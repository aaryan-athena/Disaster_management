import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise RuntimeError("mediapipe is required. Install with `pip install mediapipe`." ) from e


class FaceDetector:
    def __init__(self, min_confidence: float = 0.5, model_selection: int = 1):
        self.min_confidence = min_confidence
        self.model_selection = model_selection
        self._mp_fd = mp.solutions.face_detection
        self._detector = self._mp_fd.FaceDetection(
            model_selection=self.model_selection,
            min_detection_confidence=self.min_confidence,
        )

    def detect(self, image_bgr: np.ndarray):
        """Detect faces and return list of bounding boxes in pixel coords.

        Returns: list of dicts: {"bbox": (x, y, w, h), "score": float}
        """
        # MediaPipe expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._detector.process(image_rgb)
        h, w = image_bgr.shape[:2]
        detections = []
        if results.detections:
            for det in results.detections:
                loc = det.location_data
                if not loc or not loc.relative_bounding_box:
                    continue
                rbb = loc.relative_bounding_box
                x = max(int(rbb.xmin * w), 0)
                y = max(int(rbb.ymin * h), 0)
                bw = int(rbb.width * w)
                bh = int(rbb.height * h)
                # Clamp to image bounds
                x = max(0, min(x, w - 1))
                y = max(0, min(y, h - 1))
                bw = max(1, min(bw, w - x))
                bh = max(1, min(bh, h - y))
                score = det.score[0] if det.score else 0.0
                detections.append({"bbox": (x, y, bw, bh), "score": float(score)})
        return detections

    def crop_face(self, image_bgr: np.ndarray, margin: float = 0.2):
        """Detect the most confident face and return a cropped face image.

        margin: expand the bounding box by this fraction of its size.
        Returns: face_bgr (np.ndarray) or None.
        """
        detections = self.detect(image_bgr)
        if not detections:
            return None
        # pick the highest score
        det = max(detections, key=lambda d: d["score"])
        x, y, w, h = det["bbox"]
        # expand with margin
        mx = int(w * margin)
        my = int(h * margin)
        x0 = max(0, x - mx)
        y0 = max(0, y - my)
        x1 = min(image_bgr.shape[1], x + w + mx)
        y1 = min(image_bgr.shape[0], y + h + my)
        face = image_bgr[y0:y1, x0:x1]
        return face if face.size > 0 else None

