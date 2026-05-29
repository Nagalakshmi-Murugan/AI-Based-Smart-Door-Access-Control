from __future__ import annotations

import threading
import time
from flask import Blueprint, Response, jsonify

try:
    import cv2
except Exception:
    cv2 = None

try:
    import serial as pyserial
except Exception:
    pyserial = None

from face_module import draw_overlays, recognize_faces_in_frame, resize_frame_bgr

camera_bp = Blueprint("camera_bp", __name__)

# ── Shared camera state ────────────────────────────────
_CAP          = None
_latest_frame = None
_frame_lock   = threading.Lock()

# ── Arduino serial state ───────────────────────────────
_arduino      = None
_arduino_lock = threading.Lock()


def init_arduino(port: str, baudrate: int, simulate: bool) -> None:
    """Connect to Arduino over serial. If simulate=True, just log commands."""
    global _arduino
    if simulate:
        print(f"[arduino] SIMULATE mode — no real serial port")
        return
    if pyserial is None:
        print("[arduino] pyserial not installed — run: pip install pyserial --break-system-packages")
        return
    try:
        _arduino = pyserial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # wait for Arduino to reset
        print(f"[arduino] Connected on {port} at {baudrate} baud")
    except Exception as e:
        print(f"[arduino] Could not connect: {e}")
        print(f"[arduino] Check COM port in config.py — current: {port}")


def send_arduino_command(cmd: str) -> None:
    """Send 'O' (open) or 'C' (close) to Arduino."""
    with _arduino_lock:
        if _arduino and _arduino.is_open:
            try:
                _arduino.write(cmd.encode())
                print(f"[arduino] Sent: {cmd}")
            except Exception as e:
                print(f"[arduino] Send error: {e}")
        else:
            print(f"[arduino] SIMULATE — command: {cmd}")


def open_door() -> None:
    """Send open command in background thread so Flask doesn't block."""
    t = threading.Thread(target=send_arduino_command, args=('O',), daemon=True)
    t.start()


# ── Camera init ────────────────────────────────────────
def init_camera_global(camera_index: int, warmup_frames: int = 5) -> None:
    global _CAP
    if cv2 is None:
        return
    try:
        _CAP = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not _CAP.isOpened():
            print("❌ Camera failed to open")
            _CAP = None
            return
        print("✅ Camera opened successfully")
        _CAP.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        _CAP.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        _CAP.set(cv2.CAP_PROP_FPS, 30)
        t = threading.Thread(target=_warmup_and_read, daemon=True)
        t.start()
    except Exception as e:
        print(f"Camera error: {e}")
        _CAP = None


def _warmup_and_read() -> None:
    global _latest_frame
    print("⏳ Warming up camera...")
    time.sleep(2.0)
    for _ in range(20):
        if _CAP:
            _CAP.read()
        time.sleep(0.05)
    print("✅ Camera ready — frame reader running")
    while True:
        if _CAP is None:
            time.sleep(0.1)
            continue
        ret, frame = _CAP.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        with _frame_lock:
            _latest_frame = frame.copy()
        time.sleep(0.03)


def _get_latest_frame():
    with _frame_lock:
        return _latest_frame.copy() if _latest_frame is not None else None


# ── /recognize ─────────────────────────────────────────
@camera_bp.route("/recognize")
def recognize_route():
    if cv2 is None:
        return jsonify({"error": "cv2 not available"}), 500

    frame = _get_latest_frame()
    if frame is None:
        return jsonify({"name": "Unknown", "authorized": False, "message": "Camera warming up"})

    frame = resize_frame_bgr(frame, 640)

    try:
        detections = recognize_faces_in_frame(
            frame, tolerance=0.5, model="hog",
            use_haar_first=False, max_faces=1
        )
    except Exception as e:
        print(f"Recognition error: {e}")
        return jsonify({"name": "Unknown", "authorized": False, "message": f"Error: {e}"})

    print(f"[recognize] detections={len(detections)}")
    for d in detections:
        print(f"  → name={d['name']} distance={d.get('distance')}")

    if not detections:
        return jsonify({"name": "Unknown", "authorized": False, "message": "No face detected — look at camera"})

    user = detections[0].get("user")
    distance = detections[0].get("distance")

    if user:
        # ── Check guest expiry ──────────────────────────
        from datetime import datetime, timezone
        expiry = user.get("expiry_datetime")
        if expiry and user.get("user_type") == "guest":
            try:
                # Parse ISO datetime — handle both Z suffix and +00:00
                expiry_clean = expiry.replace("Z", "+00:00")
                expiry_dt = datetime.fromisoformat(expiry_clean)
                now_utc = datetime.now(timezone.utc)
                if now_utc > expiry_dt:
                    print(f"[recognize] Guest access EXPIRED for {user['name']} (expired {expiry_dt})")
                    # Log denied
                    try:
                        from flask import current_app
                        from database import add_log
                        add_log(current_app.config["DB_PATH"], name=user["name"], status="Denied", student_id="")
                    except Exception as e:
                        print(f"[log] {e}")
                    return jsonify({
                        "name": user["name"],
                        "authorized": False,
                        "message": f"Guest access expired — contact admin",
                        "distance": round(distance, 3) if distance else None,
                    })
            except Exception as e:
                print(f"[expiry parse error] {e}")

        # ✅ Access granted — trigger servo
        open_door()

        # Log to DB
        try:
            from flask import current_app
            from database import add_log
            add_log(
                current_app.config["DB_PATH"],
                name=user["name"],
                status="Authorized",
                student_id=""
            )
        except Exception as e:
            print(f"[log] Error saving log: {e}")

        return jsonify({
            "name": user["name"],
            "authorized": True,
            "message": "Door Opened ✅",
            "user_type": user.get("user_type", "permanent"),
            "distance": round(distance, 3) if distance else None,
        })
    else:
        # ❌ Access denied — log it
        try:
            from flask import current_app
            from database import add_log
            add_log(current_app.config["DB_PATH"], name="Unknown", status="Denied")
        except Exception as e:
            print(f"[log] Error saving log: {e}")

        return jsonify({
            "name": "Unknown",
            "authorized": False,
            "message": "Access Denied ❌",
            "distance": round(distance, 3) if distance else None,
        })


# ── /video_feed ────────────────────────────────────────
def _generate_frames():
    if cv2 is None:
        return
    print("✅ Stream started")
    while True:
        frame = _get_latest_frame()
        if frame is None:
            import numpy as np
            placeholder = cv2.imencode(".jpg",
                cv2.putText(np.zeros((480,640,3), dtype='uint8'),
                    "Camera warming up...", (160,240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,200,100), 2))[1]
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                   placeholder.tobytes() + b"\r\n")
            time.sleep(0.5)
            continue

        frame = cv2.flip(frame, 1)
        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ret:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")
        time.sleep(0.03)


@camera_bp.route("/video_feed")
def video_feed():
    return Response(_generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")