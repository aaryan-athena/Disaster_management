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
from ml.pose_detector import PoseDetector
from werkzeug.utils import secure_filename
import google.generativeai as genai
import json
from PIL import Image as PILImage


# Matching threshold (cosine similarity)
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", 0.6))

# Disaster prediction classes
DISASTER_CLASSES = ['earthquake', 'flood', 'wildfire', 'hurricane', 'landslide', 'drought', 'tornado', 'tsunami', 'volcanic_eruption']


def create_app() -> Flask:
    app = Flask(__name__)
    
    # Configure upload folder for disaster prediction
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    # Configure Gemini API for disaster prediction
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    if gemini_api_key:
        genai.configure(api_key=gemini_api_key)

    # Lazy-init heavy ML components
    detector = FaceDetector(min_confidence=0.5, model_selection=1)
    embedder = FaceEmbedder()
    pose_detector = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
        
        # First, try pose detection for full body
        pose_result = pose_detector.detect(frame_bgr)
        
        if pose_result is not None:
            # Full body detected - use pose estimation
            pose_state = pose_result['pose_state']
            face_visible = pose_result['face_visible']
            
            # Draw pose landmarks
            frame_bgr = pose_detector.draw_pose(frame_bgr, pose_result)
            
            # Determine color based on pose state
            if pose_state == "fallen":
                color = (0, 0, 255)  # Red for fallen
                status_text = "⚠️ FALLEN"
            elif pose_state == "sitting":
                color = (0, 165, 255)  # Orange for sitting
                status_text = "Sitting"
            else:  # standing
                color = (0, 255, 0)  # Green for standing
                status_text = "Standing"
            
            # If face is visible, try face recognition
            label = status_text
            if face_visible and pose_result['bbox'] is not None:
                x, y, w, h = pose_result['bbox']
                # Try to detect and recognize face
                face_detections = detector.detect(frame_bgr)
                if face_detections:
                    # Use the first detected face
                    face_det = face_detections[0]
                    fx, fy, fw, fh = face_det["bbox"]
                    mx, my = int(fw * 0.15), int(fh * 0.15)
                    fx0 = max(0, fx - mx)
                    fy0 = max(0, fy - my)
                    fx1 = min(frame_bgr.shape[1], fx + fw + mx)
                    fy1 = min(frame_bgr.shape[0], fy + fh + my)
                    face = frame_bgr[fy0:fy1, fx0:fx1]
                    
                    if face.size > 0:
                        emb = embedder.embed(face)
                        if emb is not None:
                            best_person, score = _best_match(emb)
                            if best_person is not None and score >= SIMILARITY_THRESHOLD:
                                label = f"{best_person.name} - {status_text} ({score:.2f})"
                                if pose_state == "fallen":
                                    color = (0, 0, 255)  # Keep red for fallen
                                else:
                                    color = (80, 200, 120)  # Green for recognized
                                _log_detection(best_person, location_label, latitude, longitude)
                            else:
                                label = f"Unknown - {status_text} ({score:.2f})"
            
            # Draw bounding box and label
            if pose_result['bbox'] is not None:
                x, y, w, h = pose_result['bbox']
                cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                
                # Draw label background
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame_bgr, (x, y - 30), (x + label_size[0] + 10, y), color, -1)
                cv2.putText(
                    frame_bgr,
                    label,
                    (x + 5, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
        else:
            # No full body detected - fall back to face detection only
            face_detections = detector.detect(frame_bgr)
            for det in face_detections:
                x, y, w, h = det["bbox"]
                mx, my = int(w * 0.15), int(h * 0.15)
                x0 = max(0, x - mx)
                y0 = max(0, y - my)
                x1 = min(frame_bgr.shape[1], x + w + mx)
                y1 = min(frame_bgr.shape[0], y + h + my)
                face = frame_bgr[y0:y1, x0:x1]
                label = "Person (Face Only)"
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
                cv2.rectangle(frame_bgr, (x, y - 24), (x + max(150, w), y), color, -1)
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
        
        # Draw info text
        hint = f"Threshold: {SIMILARITY_THRESHOLD:.2f} | Pose + Face Detection"
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

    # --- Disaster Prediction Routes ---
    def _classify_disaster(image_path: str) -> dict:
        """Classify disaster type using Gemini Vision API."""
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not configured in environment variables")
        
        # Open image and ensure it's closed after use
        img = PILImage.open(image_path)
        try:
            # Use gemini-2.5-flash for better free tier quota
            model = genai.GenerativeModel('gemini-2.5-flash')
            
            prompt = """Analyze this image and determine if it shows a natural disaster. 
            
Classify it into ONE of these categories:
- earthquake (collapsed buildings, cracked roads, structural damage)
- flood (water covering streets, submerged areas, water damage)
- wildfire (flames, smoke, burned areas, fire damage)
- hurricane (extreme wind damage, storm destruction)
- landslide (mudslides, collapsed hillsides, debris flow)
- drought (dried land, cracked earth, water scarcity)
- tornado (funnel clouds, tornado damage, twisted debris)
- tsunami (massive waves, coastal flooding, wave damage)
- volcanic_eruption (lava, volcanic ash, eruption)
- none (if no disaster is visible)

Respond ONLY with a JSON object in this exact format:
{
    "disaster_type": "category_name",
    "confidence": 0.95,
    "description": "brief description of what you see",
    "severity": "low/medium/high"
}"""
            
            response = model.generate_content([prompt, img])
            
            try:
                text = response.text.strip()
                if text.startswith('```'):
                    text = text.split('```')[1]
                    if text.startswith('json'):
                        text = text[4:]
                text = text.strip()
                result = json.loads(text)
                return result
            except json.JSONDecodeError:
                return {
                    "disaster_type": "unknown",
                    "confidence": 0.0,
                    "description": response.text,
                    "severity": "unknown"
                }
        finally:
            # Explicitly close the image to release the file handle
            img.close()

    @app.route("/disaster-prediction")
    def disaster_prediction_page():
        return render_template("disaster_prediction.html")

    @app.route("/api/predict-disaster", methods=["POST"])
    def predict_disaster():
        filepath = None
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file uploaded"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                result = _classify_disaster(filepath)
                return jsonify(result)
            except Exception as e:
                app.logger.exception("Error in disaster prediction")
                return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
                
        except Exception as e:
            app.logger.exception("Error in predict_disaster")
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up the uploaded file in finally block to ensure it's always deleted
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as cleanup_error:
                    app.logger.warning(f"Failed to delete file {filepath}: {cleanup_error}")

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
