from __future__ import annotations

import numpy as np
import cv2
import torch
from typing import Optional

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError as e:
    raise RuntimeError(
        "facenet-pytorch is required. Install with `pip install facenet-pytorch torch torchvision`."
    ) from e


def _fixed_image_standardization(img: np.ndarray) -> np.ndarray:
    """Standardize to roughly [-1, 1] as in facenet-pytorch."""
    return (img - 127.5) / 128.0


class FaceEmbedder:
    def __init__(self, device: Optional[str] = None):
        # Force CPU for production to reduce memory usage
        self.device = torch.device("cpu")
        # pretrained='vggface2' gives 512-d embeddings
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        # Set to inference mode to reduce memory
        torch.set_grad_enabled(False)

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return 512-d embedding for the BGR face crop.
        Returns None if input is invalid.
        """
        if face_bgr is None or face_bgr.size == 0:
            return None
        # Convert BGR -> RGB
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        # Resize to 160x160 as FaceNet expects
        face_rgb = cv2.resize(face_rgb, (160, 160), interpolation=cv2.INTER_AREA)
        # Convert to float32 and standardize to ~[-1,1]
        face_rgb = face_rgb.astype(np.float32)
        face_std = _fixed_image_standardization(face_rgb)
        # HWC -> CHW, and add batch dim
        tensor = torch.from_numpy(face_std).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            emb = self.model(tensor)
        vec = emb.squeeze(0).cpu().numpy().astype(np.float32)
        # Normalize embedding to unit length for cosine similarity
        norm = np.linalg.norm(vec) + 1e-10
        return (vec / norm).astype(np.float32)
