People Recognition (MediaPipe + FaceNet)

Overview

- Flask web app for registering people (name, location, gender, image) and recognizing them later via webcam capture.
- Uses MediaPipe for face detection and FaceNet (facenet-pytorch) for 512-d embeddings.
- Stores metadata and embeddings in SQLite; saves original images to `data/images/`.

Quick Start

1) Create a virtual environment

   - Windows (PowerShell):
     - `python -m venv .venv`
     - `.venv\\Scripts\\Activate.ps1`

   - macOS/Linux:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`

2) Install dependencies

   - `pip install --upgrade pip`
   - `pip install -r requirements.txt`

   Note: Installing `torch` and `facenet-pytorch` can take several minutes.

3) Run the app

   - `python app.py`
   - Open `http://127.0.0.1:5000` in your browser.

Usage

- Register:
  - Go to `/register`, fill name/location/gender and capture or upload a face image.
  - The server detects the face, crops, embeds with FaceNet, and stores the record.

- Recognize:
  - Go to `/recognize`, allow camera, and click Capture.
  - If similarity passes the threshold, the person’s details appear.

Project Structure

- `app.py` — Flask application and routes
- `db.py` — SQLite models and helpers (SQLAlchemy)
- `ml/mediapipe_detector.py` — MediaPipe face detector
- `ml/facenet_embedder.py` — FaceNet embedder (facenet-pytorch)
- `templates/` — HTML templates
- `static/js/` — Frontend scripts for webcam capture
- `static/css/` — Basic styles
- `data/images/` — Saved user images
- `uploads/` — Temporary storage for disaster prediction images

Environment Variables

Create a `.env` file with the following:

```
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret
GEMINI_API_KEY=your_gemini_api_key
SIMILARITY_THRESHOLD=0.6
```

Get your Gemini API key at: https://makersuite.google.com/app/apikey

Notes

- Threshold: default cosine similarity >= 0.6 is considered a match. Adjust in `app.py` (`SIMILARITY_THRESHOLD`).
- Only one embedding is stored per person. You can extend the DB schema to store multiple images/embeddings per person and use centroid or best-of-N matching.
- Disaster prediction uses Google's Gemini 2.0 Flash model for image analysis.
- For production, consider using a WSGI/ASGI server (e.g., gunicorn/waitress) and serving static assets via a CDN or reverse proxy.

