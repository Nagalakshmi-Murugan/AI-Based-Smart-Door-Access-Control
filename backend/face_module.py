from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import face_recognition
except Exception:
    face_recognition = None


@dataclass
class KnownUser:
    id: int
    name: str
    encoding: np.ndarray
    user_type: str
    expiry_datetime: Optional[str]


_known_users: List[KnownUser] = []


def load_known_users(users_with_encodings: List[Dict[str, Any]]) -> int:
    global _known_users
    loaded = []
    for u in users_with_encodings:
        enc_raw = u["face_encoding"]
        if isinstance(enc_raw, str):
            enc_raw = json.loads(enc_raw)
        enc = np.array(enc_raw, dtype=np.float64)
        loaded.append(KnownUser(
            id=int(u["id"]),
            name=str(u["name"]),
            encoding=enc,
            user_type=str(u["user_type"]),
            expiry_datetime=u.get("expiry_datetime"),
        ))
    _known_users = loaded
    print(f"[face_module] Loaded {len(_known_users)} users: {[u.name for u in _known_users]}")
    return len(_known_users)


def encode_face_from_image_bytes(image_bytes: bytes) -> List[float]:
    """Encode face from uploaded image. Uses full image — most reliable."""
    if face_recognition is None:
        raise RuntimeError("face_recognition not available.")
    img = face_recognition.load_image_file(io.BytesIO(image_bytes))
    encodings = face_recognition.face_encodings(img)
    if len(encodings) == 0:
        raise ValueError("No face detected in the image.")
    if len(encodings) > 1:
        raise ValueError("Multiple faces detected. Use an image with one face.")
    return encodings[0].astype(float).tolist()


def _encode_single_face(rgb: np.ndarray) -> Optional[np.ndarray]:
    """
    Encode a face from an RGB image using face_recognition.
    Passes the full image with NO location hints — safest approach.
    Returns None if no face found.
    """
    if face_recognition is None:
        return None
    try:
        encs = face_recognition.face_encodings(rgb)
        if encs:
            return encs[0]
    except Exception as e:
        print(f"[encode] error: {e}")
    return None


def recognize_faces_in_frame(
    frame_bgr: np.ndarray,
    tolerance: float = 0.5,
    model: str = "hog",
    use_haar_first: bool = False,
    max_faces: int = 1,
) -> List[Dict[str, Any]]:
    """
    Recognize faces in a BGR frame.
    Step 1: find face locations with face_recognition (HOG)
    Step 2: for each location, crop the face + encode with NO location arg
    This avoids the dlib incompatible argument error entirely.
    """
    if face_recognition is None:
        raise RuntimeError("face_recognition not available.")

    rgb = frame_bgr[:, :, ::-1].copy()

    # Step 1: find locations
    locations = face_recognition.face_locations(rgb, model="hog")
    if not locations:
        return []

    if max_faces and len(locations) > max_faces:
        locations = locations[:max_faces]

    results = []
    h, w = rgb.shape[:2]

    for (top, right, bottom, left) in locations:
        # Step 2: generous crop around the face
        pad = 30
        t = max(0, top - pad)
        b = min(h, bottom + pad)
        l = max(0, left - pad)
        r = min(w, right + pad)
        face_crop = rgb[t:b, l:r]

        # Resize crop to at least 100x100 so dlib can detect within it
        fh, fw = face_crop.shape[:2]
        if fh < 100 or fw < 100:
            scale = max(100/fh, 100/fw)
            face_crop = cv2.resize(face_crop, (int(fw*scale), int(fh*scale)))

        # Encode with NO location argument — lets face_recognition re-detect
        face_encoding = _encode_single_face(face_crop)

        if face_encoding is None:
            # Fallback: try encoding the whole frame
            face_encoding = _encode_single_face(rgb)

        if face_encoding is None:
            continue

        match_name = "Unknown"
        match_user = None
        distance = None

        if _known_users:
            known_encs = np.stack([u.encoding for u in _known_users], axis=0)
            distances = face_recognition.face_distance(known_encs, face_encoding)
            best_idx = int(np.argmin(distances))
            distance = float(distances[best_idx])
            print(f"[recognize] best='{_known_users[best_idx].name}' dist={distance:.4f} tol={tolerance}")
            if distance <= tolerance:
                match_user = _known_users[best_idx]
                match_name = match_user.name
        else:
            print("[recognize] WARNING: No known users in memory!")

        results.append({
            "name": match_name,
            "user": None if match_user is None else {
                "id": match_user.id,
                "name": match_user.name,
                "user_type": match_user.user_type,
                "expiry_datetime": match_user.expiry_datetime,
            },
            "distance": distance,
            "box": {"top": int(top), "right": int(right), "bottom": int(bottom), "left": int(left)},
        })

    return results


def draw_overlays(frame_bgr: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    if cv2 is None:
        return frame_bgr
    out = frame_bgr.copy()
    for d in detections:
        b = d["box"]
        name = d.get("name", "Unknown")
        top, right, bottom, left = b["top"], b["right"], b["bottom"], b["left"]
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(out, (left, top), (right, bottom), color, 2)
        cv2.rectangle(out, (left, bottom - 24), (right, bottom), color, cv2.FILLED)
        cv2.putText(out, str(name), (left + 6, bottom - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return out


def get_known_users_count() -> int:
    return len(_known_users)


def resize_frame_bgr(frame_bgr: np.ndarray, max_width: int) -> np.ndarray:
    if cv2 is None or max_width <= 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    if w <= max_width:
        return frame_bgr
    scale = max_width / float(w)
    return cv2.resize(frame_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)