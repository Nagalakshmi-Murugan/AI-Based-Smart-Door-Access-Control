"""
test_recognize.py - Fixed version
Run WITHOUT Flask running.
"""
import time
import sqlite3
import json
import numpy as np
import cv2
import face_recognition

DB_PATH = "smart_door.db"

# ── Load users directly from SQLite (bypasses face_module entirely) ──
conn = sqlite3.connect(DB_PATH)
rows = conn.execute("SELECT id, name, student_id, face_encoding FROM users").fetchall()
conn.close()

known_names = []
known_encs = []

for row in rows:
    uid, name, student_id, enc_json = row
    try:
        enc = np.array(json.loads(enc_json), dtype=np.float64)
        known_names.append(name)
        known_encs.append(enc)
    except Exception as e:
        print(f"  Skipping {name}: {e}")

print(f"Loaded {len(known_names)} users: {known_names}")

if len(known_names) == 0:
    print("ERROR: No users loaded.")
    exit()

known_encs = np.stack(known_encs, axis=0)

# ── Open camera ──
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(2)
for _ in range(10):
    cap.read()

print("\nScanning... (10 attempts, 2 sec apart)\n")

for attempt in range(10):
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"Attempt {attempt+1}: No frame captured")
        time.sleep(1)
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb, model="hog")
    print(f"Attempt {attempt+1}: Found {len(locations)} face(s)")

    if not locations:
        time.sleep(2)
        continue

    encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)

    for enc in encodings:
        distances = face_recognition.face_distance(known_encs, enc)
        best_idx = int(np.argmin(distances))
        best_dist = float(distances[best_idx])

        print(f"  Best match: '{known_names[best_idx]}' | distance={best_dist:.4f}")
        for tol in [0.4, 0.5, 0.6, 0.7]:
            result = known_names[best_idx] if best_dist <= tol else "Unknown"
            print(f"    tolerance={tol} → {result}")

    time.sleep(2)

cap.release()
print("\nDone.")