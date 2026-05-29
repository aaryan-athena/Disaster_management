# Disaster Management Suite

Real-time face recognition and disaster response platform. Registers people with FaceNet embeddings, identifies them live via a Raspberry Pi camera stream, and classifies disaster images using Gemini AI.

---

## Architecture

```
┌─────────────────────┐     HTTPS POST frames      ┌──────────────────────────────┐
│   Raspberry Pi      │ ─────────────────────────▶  │  Backend — HuggingFace Space │
│   stream_client.py  │                             │  Flask REST API              │
│   Pi Camera / USB   │                             │  FaceNet · MediaPipe · Pose  │
└─────────────────────┘                             │  Firebase · Cloudinary       │
                                                    └──────────────┬───────────────┘
                                                                   │  JSON API + MJPEG
                                                                   ▼
                                                    ┌──────────────────────────────┐
                                                    │  Frontend — Vercel           │
                                                    │  Static HTML + CSS + JS      │
                                                    │  Register · Recognize · Live │
                                                    │  Dashboard · Disaster AI     │
                                                    └──────────────────────────────┘
```

**Data flow:**
1. Pi captures frames → pushes to backend via `POST /api/pi_frame`
2. Backend runs face + pose recognition → serves annotated MJPEG stream
3. Frontend embeds the stream and calls backend JSON endpoints for all data

---

## Repository Structure

```
├── backend/          Flask REST API → deployed on HuggingFace Spaces
├── frontend/         Static site    → deployed on Vercel
├── raspberry_pi/     Pi camera scripts
├── .env              Local development secrets (never commit)
└── README.md
```

---

## Backend — HuggingFace Spaces

### Tech stack
| Component | Library |
|-----------|---------|
| Web framework | Flask + Gunicorn |
| Face detection | MediaPipe |
| Face recognition | FaceNet (facenet-pytorch) |
| Pose detection | MediaPipe Pose |
| Image storage | Cloudinary |
| Database | Firebase Firestore |
| Disaster classification | Google Gemini 2.5 Flash |
| CORS | flask-cors |

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/config` | Returns threshold and Pi mode flags |
| POST | `/api/register` | Register a person (multipart form: name, location, gender, image) |
| POST | `/api/recognize` | Identify a face from a base64 image |
| GET | `/api/dashboard` | Returns all persons with detection history |
| DELETE | `/api/persons/<id>` | Delete a registered person |
| POST | `/api/predict-disaster` | Classify disaster type from an uploaded image |
| POST | `/api/pi_frame` | Raspberry Pi posts JPEG frames here |
| GET | `/api/pi_status` | Returns `{"live": true/false}` |
| POST | `/api/live_location/<token>` | Browser sends geolocation for overlay |
| GET | `/video_feed` | MJPEG stream with recognition annotations |

### Environment variables (set as Secrets in HF Space)

| Variable | Description |
|----------|-------------|
| `FIREBASE_CREDENTIALS_JSON` | Full JSON content of Firebase service-account key |
| `CLOUDINARY_CLOUD_NAME` | Cloudinary cloud name |
| `CLOUDINARY_API_KEY` | Cloudinary API key |
| `CLOUDINARY_API_SECRET` | Cloudinary API secret |
| `GEMINI_API_KEY` | Google Gemini API key |
| `PI_AUTH_TOKEN` | Shared secret with Raspberry Pi |
| `SIMILARITY_THRESHOLD` | Cosine similarity cutoff (default `0.6`) |

### Deploy to HuggingFace Spaces

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
   - SDK: **Docker**
   - Visibility: Public or Private

2. Push the `backend/` folder as the root of the Space:
   ```bash
   cd backend
   git init
   git remote add hf https://huggingface.co/spaces/<username>/<space-name>
   git add .
   git commit -m "initial backend"
   git push hf main
   ```

3. Go to **Settings → Variables and Secrets** and add all environment variables listed above.

4. Your backend URL will be:
   ```
   https://<username>-<space-name>.hf.space
   ```

### Local development

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
# create a .env file with all variables above
python app.py                   # runs on http://localhost:7860
```

---

## Frontend — Vercel

### Tech stack
Plain HTML, CSS, and vanilla JavaScript — no build step required.

Each page calls the backend API directly. The backend URL is configured in one place:

```
frontend/js/config.js
```

```js
const BACKEND_URL = 'https://<username>-<space-name>.hf.space';
```

**Update this value** before deploying whenever the backend URL changes.

### Pages

| File | URL (after deploy) | Description |
|------|--------------------|-------------|
| `index.html` | `/` | Landing page |
| `register.html` | `/register` | Register a person with photo |
| `recognize.html` | `/recognize` | Identify a face via webcam |
| `live.html` | `/live` | Live Pi camera feed with recognition |
| `dashboard.html` | `/dashboard` | Detection history and stats |
| `disaster.html` | `/disaster` | Gemini disaster image classifier |

### Deploy to Vercel

1. Push the full repository to GitHub.

2. Go to [vercel.com/new](https://vercel.com/new) → Import your GitHub repo.

3. In project settings:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Other (static)
   - **Build Command**: *(leave empty)*
   - **Output Directory**: `.`

4. Click **Deploy**.

5. Your frontend URL will be:
   ```
   https://<project-name>.vercel.app
   ```

### Local development

No server needed — just open files in a browser or use any static server:

```bash
cd frontend
npx serve .        # or: python -m http.server 3000
```

---

## Raspberry Pi — Camera Stream

The Pi captures camera frames and pushes them to the deployed backend over HTTPS. This works through any NAT or firewall because the Pi makes **outbound** connections only.

### Hardware

- Raspberry Pi 3B+ / 4 / 5
- Pi Camera Module v2 or v3 (CSI) **or** any USB webcam

### First-time setup

```bash
cd raspberry_pi
chmod +x setup.sh
./setup.sh
```

This installs system dependencies, creates a virtual environment, and installs Python packages.

### Configuration

```bash
cp .env.example .env   # if .env doesn't exist yet
nano .env
```

| Variable | Required | Description |
|----------|----------|-------------|
| `SERVER_URL` | ✅ | Backend URL, e.g. `https://aaryan-athena-dm.hf.space` |
| `PI_AUTH_TOKEN` | ✅ | Must match `PI_AUTH_TOKEN` set on the backend |
| `CAMERA_INDEX` | — | OpenCV device index (default `0`) |
| `FRAME_WIDTH` | — | Capture width (default `640`) |
| `FRAME_HEIGHT` | — | Capture height (default `480`) |
| `FRAME_RATE` | — | Target FPS (default `8`) |
| `JPEG_QUALITY` | — | JPEG compression 1–100 (default `75`) |

### Generate the shared auth token

Run this once and paste the output into **both** `.env` files:

```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

- `raspberry_pi/.env` → `PI_AUTH_TOKEN=<token>`
- HuggingFace Space secrets → `PI_AUTH_TOKEN=<same token>`

### Start streaming

```bash
source venv/bin/activate
python stream_client.py
```

Expected output:
```
2026-05-28 10:00:00 [INFO] Pinging server to wake it up…
2026-05-28 10:00:03 [INFO] Server is awake (attempt 1)
2026-05-28 10:00:03 [INFO] Camera opened at 640x480 @ 8 fps
2026-05-28 10:00:03 [INFO] Pushing frames to https://…/api/pi_frame at 8 fps
```

Open `/live` on the frontend — the **Pi Connected** badge turns green within a few seconds.

### Run on boot (systemd)

```bash
sudo nano /etc/systemd/system/pi-stream.service
```

```ini
[Unit]
Description=Pi Camera Stream Client
After=network-online.target
Wants=network-online.target

[Service]
User=pi
WorkingDirectory=/home/pi/Disaster_management_new/raspberry_pi
ExecStart=/home/pi/Disaster_management_new/raspberry_pi/venv/bin/python stream_client.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable pi-stream
sudo systemctl start pi-stream
```

### Two streaming modes

| Mode | Script | When to use |
|------|--------|-------------|
| **Push** (recommended) | `stream_client.py` | Backend is deployed remotely — Pi pushes frames over HTTPS |
| **Pull** (local only) | `stream_server.py` | Backend is on the same LAN — server pulls MJPEG from Pi |

---

## Firebase Setup

1. Go to [Firebase Console](https://console.firebase.google.com) → create a project.
2. Enable **Firestore Database** in Native mode.
3. Go to Project Settings → Service Accounts → **Generate new private key**.
4. Copy the entire JSON content of the downloaded file.
5. In HuggingFace Space secrets, set `FIREBASE_CREDENTIALS_JSON` to that JSON content.

For local development, save the file anywhere and set:
```env
FIREBASE_CREDENTIALS=/absolute/path/to/firebase-key.json
```

---

## Cloudinary Setup

1. Create a free account at [cloudinary.com](https://cloudinary.com).
2. From the Dashboard copy **Cloud Name**, **API Key**, **API Secret**.
3. Add them as HuggingFace secrets and to local `.env`.

---

## Local `.env` reference

```env
# Firebase — use file path for local dev
FIREBASE_CREDENTIALS=C:\path\to\firebase-admin-sdk.json

# Cloudinary
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Gemini
GEMINI_API_KEY=your_gemini_key

# Recognition
SIMILARITY_THRESHOLD=0.6

# Raspberry Pi
PI_AUTH_TOKEN=your_shared_token
PI_STREAM_URL=                     # optional: local LAN pull mode
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| HF Space crashes on boot | Missing secrets | Add all required secrets in Space settings |
| `No module named 'ml'` | `PYTHONPATH` not set | Ensure `ENV PYTHONPATH=/app` is in `backend/Dockerfile` |
| `401 Unauthorized` from Pi | Token mismatch | Regenerate and set same token in both `.env` and HF secrets |
| "Waiting for Pi…" badge stays | Pi not running or wrong `SERVER_URL` | Check `raspberry_pi/.env` and run `stream_client.py` |
| Stream freezes then recovers | Render/HF cold start | Pi warm-up ping handles this automatically |
| OOM on HF Space | ML models loading at startup | Models are lazy-loaded — first request may be slow |
| MediaPipe timestamp error | Shared instance across threads | Thread-local storage is used; restart the Space if it persists |
