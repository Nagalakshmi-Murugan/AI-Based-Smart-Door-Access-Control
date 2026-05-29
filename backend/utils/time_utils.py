from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional


def parse_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parse ISO datetime string (with optional Z) into a timezone-aware datetime."""
    if not value:
        return None

    v = value.strip()
    if v.endswith("Z"):
        v = v[:-1] + "+00:00"

    # Accept "YYYY-MM-DDTHH:MM" from <input type="datetime-local"> by treating as local.
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def utcnow() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(tz=timezone.utc)


def is_expiry_valid(expiry_iso: Optional[str]) -> Optional[bool]:
    """Return True/False if expiry is valid/expired, or None if no expiry provided."""
    dt = parse_datetime(expiry_iso)
    if dt is None:
        return None
    return utcnow() < dt

