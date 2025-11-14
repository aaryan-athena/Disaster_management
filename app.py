from __future__ import annotations

import base64
import io
import os
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import cloudinary
import cloudinary.uploader
import cv2
import numpy as np
from flask import Flask, jsonify, redirect, render_template, request, url_for
from dotenv import load_dotenv

load_dotenv()

from db import Person, add_person, delete_person, list_persons, log_detection
from ml.mediapipe_detector import FaceDetector
from ml.facenet_embedder import FaceEmbedder


# Matching threshold (cosine similarity)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.6))


def create_app() -> Flask:
    app = Flask(__name__)

    # Lazy-init heavy ML components
    detector = FaceDetector(min_confidence=0.5, model_selection=1)
    embedder = FaceEmbedder()

    detection_refresh: Dict[str, datetime] = {}
    persons_cache: List[Person] = []
    embeddings_cache: Dict[str, np.ndarray] = {}
    last_people_refresh: Optional[datetime] = None
    live_location_state: Dict[str, Tuple[datetime, Optional[str], Optional[float], Optional[float]]] = {}

    DETECTION_REFRESH_TTL = timedelta(seconds=5)
    PEOPLE_REFRESH_TTL = timedelta(seconds=30)
    LIVE_LOCATION_TTL = timedelta(minutes=5)

    cloudinary_config = {
        "cloud_name": os.environ.get("CLOUDINARY_CLOUD_NAME"),
        "api_key": os.environ.get("CLOUDINARY_API_KEY"),
        "api_secret": os.environ.get("CLOUDINARY_API_SECRET"),
    }
    if not all(cloudinary_config.values()):
        raise RuntimeError(
            "Cloudinary configuration missing. "
            "Please set CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, and CLOUDINARY_API_SECRET."
        )
    cloudinary.config(
        cloud_name=cloudinary_config["cloud_name"],
        api_key=cloudinary_config["api_key"],
        api_secret=cloudinary_config["api_secret"],
        secure=True,
    )
    cloudinary_folder = os.environ.get("CLOUDINARY_FOLDER", "disaster-management")

    def _refresh_people(force: bool = False) -> None:
        nonlocal persons_cache, embeddings_cache, last_people_refresh
        now = datetime.utcnow()
        if not force and last_people_refresh and now - last_people_refresh < PEOPLE_REFRESH_TTL:
            return
        persons_cache = list_persons()
        embeddings_cache = {
            person.id: np.asarray(person.embedding, dtype=np.float32)
            for person in persons_cache
        }
        last_people_refresh = now

    _refresh_people(force=True)

    def _extract_location_payload() -> Tuple[Optional[str], Optional[float], Optional[float]]:
        def _coerce_float(value: Optional[object]) -> Optional[float]:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        location_label: Optional[str] = None
        latitude: Optional[float] = None
        longitude: Optional[float] = None

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            location_label = payload.get("location_label") or payload.get("location")
            latitude = _coerce_float(payload.get("latitude"))
            longitude = _coerce_float(payload.get("longitude"))
        else:
            location_label = request.form.get("location_label") or request.form.get("location")
            latitude = _coerce_float(request.form.get("latitude"))
            longitude = _coerce_float(request.form.get("longitude"))

        if location_label:
            location_label = location_label.strip() or None
        return location_label, latitude, longitude

    def _upload_image(img_bgr: np.ndarray) -> str:
        ok, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode image for Cloudinary upload.")
        image_bytes = buffer.tobytes()
        upload_result = cloudinary.uploader.upload(
            io.BytesIO(image_bytes),
            folder=cloudinary_folder,
            resource_type="image",
        )
        url = upload_result.get("secure_url") or upload_result.get("url")
        if not url:
            raise RuntimeError("Cloudinary upload did not return a URL.")
        return url

    def _log_detection(
        person: Person,
        location: Optional[str] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> None:
        now = datetime.utcnow()
        last_logged = detection_refresh.get(person.id)
        if last_logged and now - last_logged < DETECTION_REFRESH_TTL:
            return
        detection_refresh[person.id] = now

        if location is None and latitude is not None and longitude is not None:
            location = f"Browser location (~{latitude:.4f}, {longitude:.4f})"
        detection = log_detection(person.id, location, latitude, longitude)
        person.detection = detection

    def _decode_image_from_request(img_field: str = "image") -> Optional[np.ndarray]:
        file = request.files.get(img_field)
        data_url = request.form.get("image_data")
        if not data_url and request.is_json:
            data_url = (request.json or {}).get("image_data")

        if file and file.filename:
            bytes_data = file.read()
            arr = np.frombuffer(bytes_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img

        if data_url:
            try:
                _, b64data = data_url.split(",", 1)
                binary = base64.b64decode(b64data)
                arr = np.frombuffer(binary, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                return img
            except Exception:
                return None
        return None

    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-10
        return float(np.dot(a, b) / denom)

    def _best_match(embedding: np.ndarray) -> Tuple[Optional[Person], float]:
        if not persons_cache:
            return None, -1.0
        best_score = -1.0
        best_person: Optional[Person] = None
        for person in persons_cache:
            person_emb = embeddings_cache.get(person.id)
            if person_emb is None:
                continue
            score = _cosine_similarity(embedding, person_emb)
            if score > best_score:
                best_score = score
                best_person = person
        return best_person, float(best_score)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "models_loaded": True}), 200

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "GET":
            return render_template("register.html")

        name = request.form.get("name", "").strip()
        location = request.form.get("location", "").strip() or None
        gender = request.form.get("gender", "").strip() or None
        if not name:
            return render_template("register.html", error="Name is required."), 400

        img = _decode_image_from_request("image")
        if img is None:
            return render_template("register.html", error="No image received."), 400

        face = detector.crop_face(img, margin=0.25)
        if face is None:
            return render_template("register.html", error="No face detected. Try again."), 400

        emb = embedder.embed(face)
        if emb is None:
            return render_template("register.html", error="Failed to embed face."), 500

        try:
            image_url = _upload_image(img)
        except Exception as exc:
            return render_template("register.html", error=f"Image upload failed: {exc}"), 500

        try:
            add_person(
                name=name,
                location=location,
                gender=gender,
                image_url=image_url,
                embedding=emb.tolist(),
            )
        except Exception as exc:
            return render_template("register.html", error=f"Failed to save person: {exc}"), 500

        _refresh_people(force=True)
        return render_template("register.html", success=f"Saved {name}.")

    @app.route("/recognize", methods=["GET"])
    def recognize_page():
        return render_template("recognize.html", threshold=SIMILARITY_THRESHOLD)

    @app.route("/dashboard")
    def dashboard():
        _refresh_people(force=True)
        detected_people = sorted(
            (person for person in persons_cache if person.detection is not None),
            key=lambda person: person.detection.last_seen_at,
            reverse=True,
        )
        undetected_people = sorted(
            (person for person in persons_cache if person.detection is None),
            key=lambda person: person.name.lower(),
        )
        summary = {
            "total_registered": len(persons_cache),
            "with_detections": len(detected_people),
            "never_detected": len(undetected_people),
        }
        deleted_id = request.args.get("deleted")
        deleted_name = request.args.get("deleted_name")
        error_code = request.args.get("error")
        message = None
        if deleted_id:
            if deleted_name:
                message = {"kind": "success", "text": f"Deleted {deleted_name} (ID {deleted_id})."}
            else:
                message = {"kind": "success", "text": f"Deleted person {deleted_id}."}
        elif error_code == "delete_failed":
            message = {"kind": "error", "text": "Failed to delete the person. Please try again."}
        return render_template(
            "dashboard.html",
            summary=summary,
            detected_people=detected_people,
            undetected_people=undetected_people,
            message=message,
        )

    @app.route("/persons/<person_id>/delete", methods=["POST"])
    def delete_person_route(person_id: str):
        deleted_name = next((person.name for person in persons_cache if person.id == person_id), None)
        try:
            delete_person(person_id)
        except Exception:
            app.logger.exception("Failed to delete person %s", person_id)
            return redirect(url_for("dashboard", error="delete_failed"))

        detection_refresh.pop(person_id, None)
        embeddings_cache.pop(person_id, None)
        _refresh_people(force=True)
        params = {"deleted": person_id}
        if deleted_name:
            params["deleted_name"] = deleted_name
        return redirect(url_for("dashboard", **params))

    @app.route("/api/recognize", methods=["POST"])
    def api_recognize():
        try:
            app.logger.info("Recognize request received")
            
            img = _decode_image_from_request("image")
            if img is None:
                app.logger.warning("No image decoded from request")
                return jsonify({"ok": False, "error": "No image received."}), 400

            app.logger.info(f"Image decoded: shape={img.shape}")
            location_label, latitude, longitude = _extract_location_payload()

            app.logger.info("Detecting face...")
            face = detector.crop_face(img, margin=0.25)
            if face is None:
                app.logger.info("No face detected in image")
                return jsonify({"ok": False, "match": False, "message": "No face detected."}), 200

            app.logger.info(f"Face detected: shape={face.shape}")
            app.logger.info("Embedding face...")
            emb = embedder.embed(face)
            if emb is None:
                app.logger.error("Failed to generate embedding")
                return jsonify({"ok": False, "match": False, "message": "Failed to embed face."}), 500

            app.logger.info(f"Embedding generated: shape={emb.shape}")
            _refresh_people()
            if not persons_cache:
                app.logger.info("Database is empty")
                return jsonify({"ok": True, "match": False, "message": "Database is empty."}), 200

            app.logger.info(f"Matching against {len(persons_cache)} persons")
            best_person, best_score = _best_match(emb)

            if best_person and best_score >= SIMILARITY_THRESHOLD:
                app.logger.info(f"Match found: {best_person.name} with score {best_score}")
                _log_detection(best_person, location_label, latitude, longitude)
                return jsonify(
                    {
                        "ok": True,
                        "match": True,
                        "score": round(best_score, 4),
                        "person": {
                            "id": best_person.id,
                            "name": best_person.name,
                            "location": best_person.location,
                            "gender": best_person.gender,
                            "image_url": best_person.image_url,
                        },
                    }
                ), 200

            app.logger.info(f"No match found. Best score: {best_score}")
            return jsonify(
                {
                    "ok": True,
                    "match": False,
                    "score": round(best_score or -1.0, 4),
                    "message": "No match found.",
                }
            ), 200
        except Exception as e:
            app.logger.exception("Error in api_recognize")
            return jsonify({"ok": False, "error": f"Server error: {str(e)}"}), 500

    # --- Live video recognition ---
    def _get_live_location(token: Optional[str]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        if not token:
            return None, None, None
        record = live_location_state.get(token)
        if not record:
            return None, None, None
        timestamp, location_label, latitude, longitude = record
        if datetime.utcnow() - timestamp > LIVE_LOCATION_TTL:
            live_location_state.pop(token, None)
            return None, None, None
        return location_label, latitude, longitude

    def _annotate_frame(
        frame_bgr: np.ndarray,
        location_label: Optional[str],
        latitude: Optional[float],
        longitude: Optional[float],
    ) -> np.ndarray:
        _refresh_people()
        detections = detector.detect(frame_bgr)
        for det in detections:
            x, y, w, h = det["bbox"]
            mx, my = int(w * 0.15), int(h * 0.15)
            x0 = max(0, x - mx)
            y0 = max(0, y - my)
            x1 = min(frame_bgr.shape[1], x + w + mx)
            y1 = min(frame_bgr.shape[0], y + h + my)
            face = frame_bgr[y0:y1, x0:x1]
            label = "Person"
            color = (0, 190, 255)
            if face.size > 0:
                emb = embedder.embed(face)
                if emb is not None:
                    best_person, score = _best_match(emb)
                    if best_person is not None and score >= SIMILARITY_THRESHOLD:
                        label = f"{best_person.name} ({score:.2f})"
                        color = (80, 200, 120)
                        _log_detection(best_person, location_label, latitude, longitude)
                    else:
                        label = f"Unknown ({score:.2f})"
                        color = (0, 190, 255)
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame_bgr, (x, y - 24), (x + max(120, w), y), color, -1)
            cv2.putText(
                frame_bgr,
                label,
                (x + 4, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        hint = f"Threshold: {SIMILARITY_THRESHOLD:.2f} | q to stop"
        cv2.putText(frame_bgr, hint, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 3, cv2.LINE_AA)
        cv2.putText(frame_bgr, hint, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return frame_bgr

    def _gen_mjpeg(token: Optional[str]):
        cap = cv2.VideoCapture(0)
        try:
            if not cap.isOpened():
                raise RuntimeError("Cannot open camera (index 0)")
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                location_label, latitude, longitude = _get_live_location(token)
                frame = _annotate_frame(frame, location_label, latitude, longitude)
                ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ok:
                    continue
                jpg = buffer.tobytes()
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                )
        finally:
            cap.release()

    @app.route("/api/live_location/<token>", methods=["POST"])
    def live_location_update(token: str):
        if not token:
            return jsonify({"ok": False, "error": "Missing token."}), 400
        location_label, latitude, longitude = _extract_location_payload()
        if location_label is None and latitude is None and longitude is None:
            live_location_state.pop(token, None)
            return jsonify({"ok": True, "cleared": True})
        live_location_state[token] = (datetime.utcnow(), location_label, latitude, longitude)
        return jsonify({"ok": True})

    @app.route("/live")
    def live_page():
        # Check if running on a server without camera access
        is_production = os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT")
        if is_production:
            return render_template(
                "index.html",
                error="Live video feed is not available in production (requires server camera access). Use the Recognize page instead."
            )
        token = secrets.token_urlsafe(16)
        return render_template("live.html", threshold=SIMILARITY_THRESHOLD, live_token=token)

    @app.route("/video_feed")
    def video_feed():
        # Check if running on a server without camera access
        is_production = os.environ.get("RENDER") or os.environ.get("RAILWAY_ENVIRONMENT")
        if is_production:
            return jsonify({"error": "Video feed not available in production"}), 503
        token = request.args.get("token") or ""
        return app.response_class(_gen_mjpeg(token), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
