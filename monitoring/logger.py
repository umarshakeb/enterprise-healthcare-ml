import hashlib
import json
import logging
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "predictions.log"

logger = logging.getLogger("prediction_logger")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


MONITORING_FILE = LOG_DIR / "drift_data.log"
data_logger = logging.getLogger("data_monitoring_logger")
data_logger.setLevel(logging.INFO)

if not data_logger.handlers:
    file_handler = logging.FileHandler(MONITORING_FILE, encoding="utf-8")
    formatter = logging.Formatter("%(message)s")
    file_handler.setFormatter(formatter)
    data_logger.addHandler(file_handler)


def generate_input_hash(input_data):
    payload_str = json.dumps(input_data, sort_keys=True)
    return hashlib.sha256(payload_str.encode()).hexdigest()

def log_prediction(model_name, model_version, input_data, prediction):
    log_entry = {
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "model_name" : model_name,
        "model_version" : model_version,
        "input_data" : generate_input_hash(input_data),
        "prediction" : prediction
    }
    logger.info(json.dumps(log_entry))


def log_monitoring_data(model_name, model_version, input_data, prediction):
    log_entry = {
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "model_name" : model_name,
        "model_version" : model_version,
        "input_data" : input_data,
        "prediction" : prediction
    }
    data_logger.info(json.dumps(log_entry))