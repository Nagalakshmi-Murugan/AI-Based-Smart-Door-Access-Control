from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

from flask import Blueprint, current_app, jsonify, request

import face_module
from database import add_user, delete_user, list_users, list_users_with_encodings
from dataset_loader import sync_dataset_to_database


user_bp = Blueprint("user_bp", __name__)


def _ensure_dataset_dir() -> str:
    """Create dataset directory if missing and return its path."""
    dataset_dir = current_app.config["DATASET_DIR"]
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir


@user_bp.route("/add-user", methods=["POST"])
def add_user_route():
    """Add a user with an image, student_id, type and optional expiry."""
    name       = (request.form.get("name") or "").strip()
    student_id = (request.form.get("student_id") or "").strip()
    user_type  = (request.form.get("user_type") or "permanent").strip().lower()
    expiry_datetime: Optional[str] = request.form.get("expiry_datetime")

    if not name:
        return jsonify({"error": "Name is required."}), 400

    if not student_id:
        return jsonify({"error": "Student / Employee ID is required."}), 400

    if user_type not in {"permanent", "guest"}:
        return jsonify({"error": "user_type must be 'permanent' or 'guest'."}), 400

    if "image" not in request.files:
        return jsonify({"error": "Image file is required (field name: image)."}), 400

    img_file  = request.files["image"]
    img_bytes = img_file.read()
    if not img_bytes:
        return jsonify({"error": "Empty image upload."}), 400

    # Encode face from image bytes
    try:
        encoding = face_module.encode_face_from_image_bytes(img_bytes)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to encode face: {e}"}), 500

    # Save uploaded image to dataset/ folder
    dataset_dir = _ensure_dataset_dir()
    ts          = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    safe_name   = (
        "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()
        or "user"
    )
    img_path = os.path.join(dataset_dir, f"{safe_name}_{ts}.jpg")
    try:
        with open(img_path, "wb") as f:
            f.write(img_bytes)
    except Exception:
        pass  # image save failure should not block enrollment

    user_id = add_user(
        current_app.config["DB_PATH"],
        name,
        encoding,
        user_type,
        expiry_datetime,
        student_id=student_id,
    )

    # Reload in-memory cache so recognition works immediately
    face_module.load_known_users(
        list_users_with_encodings(current_app.config["DB_PATH"])
    )

    return (
        jsonify(
            {
                "id":               user_id,
                "name":             name,
                "student_id":       student_id,
                "user_type":        user_type,
                "expiry_datetime":  expiry_datetime,
                "message":          f"{name} enrolled successfully!",
            }
        ),
        201,
    )


@user_bp.route("/load-dataset", methods=["POST"])
def load_dataset_route():
    """
    Re-scan dataset/ images, encode faces, upsert encodings in the database,
    and refresh the in-memory comparison cache.
    """
    default_type = (request.args.get("user_type") or "permanent").strip().lower()
    if default_type not in {"permanent", "guest"}:
        return jsonify({"error": "user_type must be 'permanent' or 'guest'."}), 400

    _ensure_dataset_dir()
    stats = sync_dataset_to_database(
        current_app.config["DB_PATH"],
        current_app.config["DATASET_DIR"],
        default_user_type=default_type,
    )
    face_module.load_known_users(
        list_users_with_encodings(current_app.config["DB_PATH"])
    )
    return jsonify({
        "ok": True,
        **stats,
        "known_users": face_module.get_known_users_count()
    })


@user_bp.route("/users", methods=["GET"])
def users_route():
    """List all users."""
    users = list_users(current_app.config["DB_PATH"])
    return jsonify({"users": users, "count": len(users)})


@user_bp.route("/users/<int:user_id>", methods=["DELETE"])
def delete_user_route(user_id: int):
    """Delete a user by ID."""
    ok = delete_user(current_app.config["DB_PATH"], user_id)
    if not ok:
        return jsonify({"error": "User not found."}), 404

    face_module.load_known_users(
        list_users_with_encodings(current_app.config["DB_PATH"])
    )
    return jsonify({"deleted": True})
