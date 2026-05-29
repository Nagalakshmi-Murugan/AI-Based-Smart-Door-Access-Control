from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from database import list_logs


log_bp = Blueprint("log_bp", __name__)


@log_bp.route("/logs", methods=["GET"])
def logs_route():
    """
    Return access logs as JSON.

    Each log: { id, name, timestamp, status } where status is Authorized or Denied.
    """
    limit_raw = request.args.get("limit", "200")
    try:
        limit = max(1, min(1000, int(limit_raw)))
    except ValueError:
        limit = 200

    logs = list_logs(current_app.config["DB_PATH"], limit=limit)
    return jsonify({"logs": logs, "count": len(logs)})

