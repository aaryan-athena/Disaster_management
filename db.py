from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    IST = ZoneInfo("Asia/Kolkata")
except ZoneInfoNotFoundError:
    IST = timezone(timedelta(hours=5, minutes=30), name="IST")

_client: Optional[firestore.Client] = None


@dataclass
class Detection:
    location: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    last_seen_at: datetime


@dataclass
class Person:
    id: str
    name: str
    location: Optional[str]
    gender: Optional[str]
    image_url: Optional[str]
    embedding: List[float]
    created_at: datetime
    detection: Optional[Detection] = None


def _format_ist(dt: datetime) -> str:
    return dt.astimezone(IST).strftime("%Y-%m-%d %H:%M:%S")


def _parse_ist(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)


def _safe_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_firestore() -> firestore.Client:
    global _client
    if _client is not None:
        return _client

    if not firebase_admin._apps:
        cred_path = os.environ.get("FIREBASE_CREDENTIALS")
        if not cred_path:
            raise RuntimeError(
                "FIREBASE_CREDENTIALS environment variable is required to initialize Firebase."
            )
        if not os.path.exists(cred_path):
            raise FileNotFoundError(f"Firebase credentials file not found: {cred_path}")
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
    _client = firestore.client()
    return _client


def _doc_to_person(doc_id: str, data: Dict) -> Person:
    created_at = _parse_ist(data.get("created_at_ist")) or datetime.now(IST)
    detection_info = data.get("detection") or {}
    detection = None
    if detection_info:
        last_seen = _parse_ist(detection_info.get("last_seen_at_ist")) or created_at
        detection = Detection(
            location=(detection_info.get("location") or None),
            latitude=_safe_float(detection_info.get("latitude")),
            longitude=_safe_float(detection_info.get("longitude")),
            last_seen_at=last_seen,
        )

    embedding = data.get("embedding") or []
    if isinstance(embedding, tuple):
        embedding = list(embedding)

    return Person(
        id=doc_id,
        name=data.get("name", ""),
        location=data.get("location"),
        gender=data.get("gender"),
        image_url=data.get("image_url"),
        embedding=list(embedding),
        created_at=created_at,
        detection=detection,
    )


def add_person(
    *,
    name: str,
    location: Optional[str],
    gender: Optional[str],
    image_url: Optional[str],
    embedding: List[float],
) -> Person:
    client = _ensure_firestore()
    doc_ref = client.collection("persons").document()
    created_at = datetime.now(IST)
    payload = {
        "name": name,
        "location": location,
        "gender": gender,
        "image_url": image_url,
        "embedding": [float(x) for x in embedding],
        "created_at_ist": _format_ist(created_at),
    }
    doc_ref.set(payload)
    return _doc_to_person(doc_ref.id, payload)


def list_persons() -> List[Person]:
    client = _ensure_firestore()
    docs = client.collection("persons").stream()
    return [_doc_to_person(doc.id, doc.to_dict()) for doc in docs]


def delete_person(person_id: str) -> None:
    client = _ensure_firestore()
    client.collection("persons").document(person_id).delete()


def log_detection(
    person_id: str,
    location: Optional[str],
    latitude: Optional[float],
    longitude: Optional[float],
) -> Detection:
    client = _ensure_firestore()
    now = datetime.now(IST)
    clean_location = (location or "").strip() or None
    detection_payload = {
        "location": clean_location,
        "latitude": _safe_float(latitude),
        "longitude": _safe_float(longitude),
        "last_seen_at_ist": _format_ist(now),
    }
    client.collection("persons").document(person_id).set(
        {"detection": detection_payload}, merge=True
    )
    return Detection(
        location=clean_location,
        latitude=detection_payload["latitude"],
        longitude=detection_payload["longitude"],
        last_seen_at=now,
    )
