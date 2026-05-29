from __future__ import annotations

import os
import time
import cv2

from flask import Flask, jsonify, Response, send_from_directory
from flask_cors import CORS

from config import Config
from database import init_db, list_users_with_encodings
from dataset_loader import print_dataset_sync_summary, sync_dataset_to_database
from face_module import get_known_users_count, load_known_users
from routes.camera_routes import camera_bp, init_camera_global, init_arduino
from routes.log_routes import log_bp
from routes.user_routes import user_bp


def create_app() -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    os.makedirs(app.config["DATASET_DIR"], exist_ok=True)
    init_db(app.config["DB_PATH"])

    ds_stats = sync_dataset_to_database(
        app.config["DB_PATH"], app.config["DATASET_DIR"]
    )
    print_dataset_sync_summary(ds_stats, app.config["DATASET_DIR"])

    load_known_users(list_users_with_encodings(app.config["DB_PATH"]))

    try:
        init_camera_global(
            app.config["CAMERA_INDEX"],
            int(app.config.get("CAMERA_WARMUP_FRAMES", 5)),
        )
    except Exception as e:
        print(f"[camera] Global init skipped: {e}")

    # ── Arduino init ──────────────────────────────────
    init_arduino(
        port=app.config["ARDUINO_PORT"],
        baudrate=app.config["ARDUINO_BAUDRATE"],
        simulate=app.config["ARDUINO_SIMULATE"],
    )

    app.register_blueprint(user_bp)
    app.register_blueprint(log_bp)
    app.register_blueprint(camera_bp, url_prefix="")

    @app.route("/")
    def index():
        return send_from_directory(".", "smart_door.html")

    @app.get("/health")
    def health():
        return jsonify({"ok": True, "known_users": get_known_users_count()})

    @app.route("/capture_frame")
    def capture_frame():
        from routes.camera_routes import _latest_frame, _frame_lock
        import numpy as np
        import io
        from flask import send_file
        with _frame_lock:
            grabbed = _latest_frame.copy() if _latest_frame is not None else None
        if grabbed is None:
            return jsonify({"error": "No frame"}), 500
        ret, buffer = cv2.imencode(".jpg", grabbed)
        return send_file(io.BytesIO(buffer.tobytes()), mimetype="image/jpeg")

    @app.route("/debug_users")
    def debug_users():
        from database import list_users_with_encodings
        users = list_users_with_encodings(app.config["DB_PATH"])
        return jsonify({
            "db_count": len(users),
            "cached_count": get_known_users_count(),
            "names": [u["name"] for u in users]
        })

    @app.route("/logs")
    def get_logs():
        from database import list_logs
        logs = list_logs(app.config["DB_PATH"])
        return jsonify({"logs": logs})

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)