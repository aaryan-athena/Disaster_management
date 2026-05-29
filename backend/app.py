from __future__ import annotations

import base64
import io
import os
import secrets
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

from db import Person, add_person, delete_person, list_persons, log_detection
from ml.mediapipe_detector import FaceDetector
from ml.facenet_embedder import FaceEmbedder
from ml.pose_detector import PoseDetector
from werkzeug.utils import secure_filename
import google.generativeai as genai
import json
from PIL import Image as PILImage


SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.6))
PI_STREAM_URL: str = os.environ.get("PI_STREAM_URL", "").strip()
PI_AUTH_TOKEN: str = os.environ.get("PI_AUTH_TOKEN", "").strip()


# ── Pi frame buffer ───────────────────────────────────────────────────────────

class _PiFrameBuffer:
    def __init__(self) -> None:
        self._frame: bytes | None = None
        self._pushed_at: float = 0.0
        self._cond = threading.Condition()

    def push(self, jpg: bytes) -> None:
        with self._cond:
            self._frame = jpg
            self._pushed_at = time.monotonic()
            self._cond.notify_all()

    def wait(self, timeout: float = 5.0) -> bytes | None:
        with self._cond:
            self._cond.wait(timeout=timeout)
            return self._frame

    @property
    def is_live(self) -> bool:
        return self._frame is not None and (time.monotonic() - self._pushed_at) < 10.0


_pi_frame_buffer = _PiFrameBuffer()


# ── Lazy thread-local ML models ───────────────────────────────────────────────
# MediaPipe has internal mutable timestamp state — not thread-safe.
# Each gunicorn thread gets its own instance via threading.local().
# FaceEmbedder (PyTorch) is stateless during inference — shared singleton is fine.

_thread_local = threading.local()
_embedder_instance: Optional[FaceEmbedder] = None
_embedder_lock = threading.Lock()


def _get_detector() -> FaceDetector:
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector = FaceDetector(min_confidence=0.5, model_selection=1)
    return _thread_local.detector


def _get_embedder() -> FaceEmbedder:
    global _embedder_instance
    if _embedder_instance is None:
        with _embedder_lock:
            if _embedder_instance is None:
                _embedder_instance = FaceEmbedder()
    return _embedder_instance


def _get_poser() -> PoseDetector:
    if not hasattr(_thread_local, "poser"):
        _thread_local.poser = PoseDetector(
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
    return _thread_local.poser


DISASTER_CLASSES = [
    "earthquake", "flood", "wildfire", "hurricane", "landslide",
    "drought", "tornado", "tsunami", "volcanic_eruption",
]


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)   # Allow all origins — frontend on Vercel calls this backend

    app.config["UPLOAD_FOLDER"] = "uploads"
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)

    cloudinary.config(
        cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
        api_key=os.environ.get("CLOUDINARY_API_KEY"),
        api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
        secure=True,
    )
    cloudinary_folder = os.environ.get("CLOUDINARY_FOLDER", "disaster-management")

    # ── In-memory caches ──────────────────────────────────────────────────────

    detection_refresh: Dict[str, datetime] = {}
    persons_cache: List[Person] = []
    embeddings_cache: Dict[str, np.ndarray] = {}
    last_people_refresh: Optional[datetime] = None
    live_location_state: Dict[str, Tuple[datetime, Optional[str], Optional[float], Optional[float]]] = {}

    DETECTION_REFRESH_TTL = timedelta(seconds=5)
    PEOPLE_REFRESH_TTL = timedelta(seconds=30)
    LIVE_LOCATION_TTL = timedelta(minutes=5)

    def _refresh_people(force: bool = False) -> None:
        nonlocal persons_cache, embeddings_cache, last_people_refresh
        now = datetime.utcnow()
        if not force and last_people_refresh and now - last_people_refresh < PEOPLE_REFRESH_TTL:
            return
        persons_cache = list_persons()
        embeddings_cache = {
            p.id: np.asarray(p.embedding, dtype=np.float32) for p in persons_cache
        }
        last_people_refresh = now

    _refresh_people(force=True)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_location_payload():
        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        if request.is_json:
            p = request.get_json(silent=True) or {}
            label = p.get("location_label") or p.get("location")
            lat = _f(p.get("latitude"))
            lon = _f(p.get("longitude"))
        else:
            label = request.form.get("location_label") or request.form.get("location")
            lat = _f(request.form.get("latitude"))
            lon = _f(request.form.get("longitude"))
        return (label.strip() or None) if label else None, lat, lon

    def _upload_image(img_bgr: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode image.")
        result = cloudinary.uploader.upload(
            io.BytesIO(buf.tobytes()), folder=cloudinary_folder, resource_type="image"
        )
        url = result.get("secure_url") or result.get("url")
        if not url:
            raise RuntimeError("Cloudinary upload did not return a URL.")
        return url

    def _log_detection(person, location=None, latitude=None, longitude=None):
        now = datetime.utcnow()
        if (last := detection_refresh.get(person.id)) and now - last < DETECTION_REFRESH_TTL:
            return
        detection_refresh[person.id] = now
        if location is None and latitude is not None:
            location = f"Browser location (~{latitude:.4f}, {longitude:.4f})"
        person.detection = log_detection(person.id, location, latitude, longitude)

    def _decode_image_from_request(field: str = "image") -> Optional[np.ndarray]:
        file = request.files.get(field)
        data_url = request.form.get("image_data")
        if not data_url and request.is_json:
            data_url = (request.json or {}).get("image_data")

        if file and file.filename:
            arr = np.frombuffer(file.read(), dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if data_url:
            try:
                _, b64 = data_url.split(",", 1)
                arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                return None
        return None

    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(np.float32), b.astype(np.float32)
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10))

    def _best_match(emb: np.ndarray):
        best_score, best_person = -1.0, None
        for person in persons_cache:
            e = embeddings_cache.get(person.id)
            if e is None:
                continue
            s = _cosine_similarity(emb, e)
            if s > best_score:
                best_score, best_person = s, person
        return best_person, float(best_score)

    def _person_to_dict(p: Person) -> dict:
        d: dict = {
            "id": p.id,
            "name": p.name,
            "location": p.location,
            "gender": p.gender,
            "image_url": p.image_url,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        if p.detection:
            d["detection"] = {
                "location": p.detection.location,
                "latitude": p.detection.latitude,
                "longitude": p.detection.longitude,
                "last_seen_at": p.detection.last_seen_at.isoformat() if p.detection.last_seen_at else None,
            }
        else:
            d["detection"] = None
        return d

    # ── Core API routes ───────────────────────────────────────────────────────

    @app.route("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    @app.route("/api/config")
    def api_config():
        return jsonify({
            "threshold": SIMILARITY_THRESHOLD,
            "pi_push_mode": bool(PI_AUTH_TOKEN),
            "pi_stream_url": PI_STREAM_URL,
        })

    @app.route("/api/register", methods=["POST"])
    def api_register():
        name = (request.form.get("name") or "").strip()
        if not name:
            return jsonify({"ok": False, "error": "Name is required."}), 400

        img = _decode_image_from_request("image")
        if img is None:
            return jsonify({"ok": False, "error": "No image received."}), 400

        face = _get_detector().crop_face(img, margin=0.25)
        if face is None:
            return jsonify({"ok": False, "error": "No face detected. Try again."}), 400

        emb = _get_embedder().embed(face)
        if emb is None:
            return jsonify({"ok": False, "error": "Failed to embed face."}), 500

        try:
            image_url = _upload_image(img)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Image upload failed: {exc}"}), 500

        try:
            add_person(
                name=name,
                location=(request.form.get("location") or "").strip() or None,
                gender=(request.form.get("gender") or "").strip() or None,
                image_url=image_url,
                embedding=emb.tolist(),
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Failed to save person: {exc}"}), 500

        _refresh_people(force=True)
        return jsonify({"ok": True, "message": f"Saved {name}."}), 200

    @app.route("/api/recognize", methods=["POST"])
    def api_recognize():
        try:
            img = _decode_image_from_request("image")
            if img is None:
                return jsonify({"ok": False, "error": "No image received."}), 400

            location_label, latitude, longitude = _extract_location_payload()
            face = _get_detector().crop_face(img, margin=0.25)
            if face is None:
                return jsonify({"ok": False, "match": False, "message": "No face detected."}), 200

            emb = _get_embedder().embed(face)
            if emb is None:
                return jsonify({"ok": False, "match": False, "message": "Failed to embed face."}), 500

            _refresh_people()
            if not persons_cache:
                return jsonify({"ok": True, "match": False, "message": "Database is empty."}), 200

            best_person, best_score = _best_match(emb)
            if best_person and best_score >= SIMILARITY_THRESHOLD:
                _log_detection(best_person, location_label, latitude, longitude)
                return jsonify({
                    "ok": True, "match": True, "score": round(best_score, 4),
                    "person": {
                        "id": best_person.id, "name": best_person.name,
                        "location": best_person.location, "gender": best_person.gender,
                        "image_url": best_person.image_url,
                    },
                }), 200

            return jsonify({"ok": True, "match": False, "score": round(best_score or -1.0, 4),
                            "message": "No match found."}), 200
        except Exception as e:
            app.logger.exception("Error in api_recognize")
            return jsonify({"ok": False, "error": f"Server error: {str(e)}"}), 500

    @app.route("/api/dashboard")
    def api_dashboard():
        _refresh_people(force=True)
        detected = sorted(
            [p for p in persons_cache if p.detection],
            key=lambda p: p.detection.last_seen_at, reverse=True,
        )
        undetected = sorted(
            [p for p in persons_cache if not p.detection],
            key=lambda p: p.name.lower(),
        )
        return jsonify({
            "summary": {
                "total_registered": len(persons_cache),
                "with_detections": len(detected),
                "never_detected": len(undetected),
            },
            "detected": [_person_to_dict(p) for p in detected],
            "undetected": [_person_to_dict(p) for p in undetected],
        })

    @app.route("/api/persons/<person_id>", methods=["DELETE"])
    def delete_person_route(person_id: str):
        try:
            delete_person(person_id)
        except Exception:
            app.logger.exception("Failed to delete person %s", person_id)
            return jsonify({"ok": False, "error": "Delete failed."}), 500

        detection_refresh.pop(person_id, None)
        embeddings_cache.pop(person_id, None)
        _refresh_people(force=True)
        return jsonify({"ok": True}), 200

    # ── Disaster prediction ───────────────────────────────────────────────────

    def _classify_disaster(image_path: str) -> dict:
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured")
        img = PILImage.open(image_path)
        try:
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = """Analyze this image and determine if it shows a natural disaster.
Classify into ONE of: earthquake, flood, wildfire, hurricane, landslide, drought, tornado, tsunami, volcanic_eruption, none.
Respond ONLY with JSON: {"disaster_type":"...","confidence":0.95,"description":"...","severity":"low/medium/high"}"""
            response = model.generate_content([prompt, img])
            text = response.text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {"disaster_type": "unknown", "confidence": 0.0,
                    "description": response.text, "severity": "unknown"}
        finally:
            img.close()

    @app.route("/api/predict-disaster", methods=["POST"])
    def predict_disaster():
        filepath = None
        try:
            if "file" not in request.files or request.files["file"].filename == "":
                return jsonify({"error": "No file uploaded"}), 400
            file = request.files["file"]
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(file.filename))
            file.save(filepath)
            return jsonify(_classify_disaster(filepath))
        except Exception as e:
            app.logger.exception("Disaster prediction error")
            return jsonify({"error": str(e)}), 500
        finally:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception:
                    pass

    # ── Raspberry Pi frame ingestion ──────────────────────────────────────────

    @app.route("/api/pi_frame", methods=["POST"])
    def receive_pi_frame():
        if PI_AUTH_TOKEN and request.headers.get("X-Pi-Token", "") != PI_AUTH_TOKEN:
            return jsonify({"ok": False, "error": "Unauthorized"}), 401
        jpg = request.data
        if not jpg:
            return jsonify({"ok": False, "error": "Empty body"}), 400
        _pi_frame_buffer.push(jpg)
        return jsonify({"ok": True}), 200

    @app.route("/api/pi_status")
    def pi_status():
        return jsonify({"live": _pi_frame_buffer.is_live})

    # ── Live video feed ───────────────────────────────────────────────────────

    @app.route("/api/live_location/<token>", methods=["POST"])
    def live_location_update(token: str):
        label, lat, lon = _extract_location_payload()
        if label is None and lat is None:
            live_location_state.pop(token, None)
            return jsonify({"ok": True, "cleared": True})
        live_location_state[token] = (datetime.utcnow(), label, lat, lon)
        return jsonify({"ok": True})

    def _get_live_location(token):
        rec = live_location_state.get(token)
        if not rec:
            return None, None, None
        ts, label, lat, lon = rec
        if datetime.utcnow() - ts > LIVE_LOCATION_TTL:
            live_location_state.pop(token, None)
            return None, None, None
        return label, lat, lon

    def _annotate_frame(frame_bgr, location_label, latitude, longitude):
        _refresh_people()
        pose_result = _get_poser().detect(frame_bgr)
        if pose_result is not None:
            pose_state = pose_result["pose_state"]
            frame_bgr = _get_poser().draw_pose(frame_bgr, pose_result)
            color = {
                "fallen": (0, 0, 255), "sitting": (0, 165, 255)
            }.get(pose_state, (0, 255, 0))
            label = {"fallen": "⚠ FALLEN", "sitting": "Sitting"}.get(pose_state, "Standing")
            if pose_result["face_visible"] and pose_result.get("bbox"):
                face_dets = _get_detector().detect(frame_bgr)
                if face_dets:
                    fx, fy, fw, fh = face_dets[0]["bbox"]
                    mx, my = int(fw * 0.15), int(fh * 0.15)
                    face = frame_bgr[max(0, fy-my):min(frame_bgr.shape[0], fy+fh+my),
                                     max(0, fx-mx):min(frame_bgr.shape[1], fx+fw+mx)]
                    if face.size > 0:
                        emb = _get_embedder().embed(face)
                        if emb is not None:
                            p, s = _best_match(emb)
                            if p and s >= SIMILARITY_THRESHOLD:
                                label = f"{p.name} - {label} ({s:.2f})"
                                color = (0, 0, 255) if pose_state == "fallen" else (80, 200, 120)
                                _log_detection(p, location_label, latitude, longitude)
            if pose_result.get("bbox"):
                x, y, w, h = pose_result["bbox"]
                cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 2)
                cv2.rectangle(frame_bgr, (x, y-30), (x+200, y), color, -1)
                cv2.putText(frame_bgr, label, (x+5, y-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
        else:
            for det in _get_detector().detect(frame_bgr):
                x, y, w, h = det["bbox"]
                mx, my = int(w*0.15), int(h*0.15)
                face = frame_bgr[max(0,y-my):min(frame_bgr.shape[0],y+h+my),
                                  max(0,x-mx):min(frame_bgr.shape[1],x+w+mx)]
                label, color = "Person (Face Only)", (0, 190, 255)
                if face.size > 0:
                    emb = _get_embedder().embed(face)
                    if emb is not None:
                        p, s = _best_match(emb)
                        if p and s >= SIMILARITY_THRESHOLD:
                            label, color = f"{p.name} ({s:.2f})", (80, 200, 120)
                            _log_detection(p, location_label, latitude, longitude)
                cv2.rectangle(frame_bgr, (x,y), (x+w,y+h), color, 2)
                cv2.rectangle(frame_bgr, (x,y-24), (x+max(150,w),y), color, -1)
                cv2.putText(frame_bgr, label, (x+4,y-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        hint = f"Threshold: {SIMILARITY_THRESHOLD:.2f}"
        cv2.putText(frame_bgr, hint, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30,30,30), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, hint, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return frame_bgr

    def _gen_mjpeg(token):
        is_production = bool(
            os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT")
            or os.environ.get("SPACE_ID")
        )
        if is_production or _pi_frame_buffer.is_live or PI_AUTH_TOKEN:
            misses = 0
            while misses < 12:
                jpg = _pi_frame_buffer.wait(timeout=10.0)
                if jpg is None:
                    misses += 1
                    continue
                misses = 0
                nparr = np.frombuffer(jpg, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                label, lat, lon = _get_live_location(token)
                frame = _annotate_frame(frame, label, lat, lon)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            return
        source: str | int = PI_STREAM_URL if PI_STREAM_URL else 0
        cap = cv2.VideoCapture(source)
        try:
            if not cap.isOpened():
                return
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                label, lat, lon = _get_live_location(token)
                frame = _annotate_frame(frame, label, lat, lon)
                ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
        finally:
            cap.release()

    @app.route("/video_feed")
    def video_feed():
        token = request.args.get("token") or ""
        return app.response_class(
            _gen_mjpeg(token), mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, port=7860)
