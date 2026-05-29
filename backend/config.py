import os


class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    SECRET_KEY = os.environ.get("SMART_DOOR_SECRET_KEY", "dev-secret-key-change-me")
    JSON_SORT_KEYS = False

    # --- Database ---
    DB_PATH = os.path.join(BASE_DIR, "smart_door.db")

    # --- Dataset ---
    DATASET_DIR = os.environ.get(
        "SMART_DOOR_DATASET_DIR", os.path.join(BASE_DIR, "dataset")
    )

    # --- Camera ---
    CAMERA_INDEX = int(os.environ.get("SMART_DOOR_CAMERA_INDEX", "0"))
    CAMERA_WARMUP_FRAMES = int(os.environ.get("SMART_DOOR_CAMERA_WARMUP_FRAMES", "5"))
    CAMERA_FRAME_MAX_WIDTH = int(os.environ.get("SMART_DOOR_CAMERA_FRAME_MAX_WIDTH", "640"))

    # --- Face recognition ---
    # Tolerance 0.5 works well based on testing (distances 0.34–0.49)
    FACE_TOLERANCE = float(os.environ.get("SMART_DOOR_FACE_TOLERANCE", "0.5"))
    FACE_MODEL = os.environ.get("SMART_DOOR_FACE_MODEL", "hog")
    FACE_USE_OPENCV_HAAR = False   # DISABLED — causes dlib incompatibility

    # --- Arduino ---
    ARDUINO_SIMULATE = os.environ.get("SMART_DOOR_ARDUINO_SIMULATE", "1") == "1"
    ARDUINO_PORT = os.environ.get("SMART_DOOR_ARDUINO_PORT", "COM3")
    ARDUINO_BAUDRATE = int(os.environ.get("SMART_DOOR_ARDUINO_BAUDRATE", "9600"))

    # --- Logs ---
    # Maximum number of log rows returned by GET /logs.
    # Increase if you need longer history; SQLite handles millions of rows fine.
    LOG_LIMIT = int(os.environ.get("SMART_DOOR_LOG_LIMIT", "500"))