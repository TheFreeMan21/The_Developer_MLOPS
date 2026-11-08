import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MLFLOW_STATE_DIR = PROJECT_ROOT / "mlflow-state"
IMAGE_DATA_DIR = PROJECT_ROOT / "data" / "images" / "raw"
METADATA_PATH = PROJECT_ROOT / "data" / "images" / "metadata.csv"
TEST_DATA_DIR = PROJECT_ROOT / "data" / "test"

# Artifact Paths
PYTORCH_MODEL_PATH = ARTIFACT_DIR / "vial_cnn_model.pth"
MODEL_RESULTS_PATH = ARTIFACT_DIR / "model_results.json"
DRIFT_BASELINE_STATS_PATH = ARTIFACT_DIR / "drift_baseline_stats.json"
CLASS_NAMES_PATH = ARTIFACT_DIR / "class_names.json"
IMAGE_STATISTICS_PATH = ARTIFACT_DIR / "image_statistics.json"
INFERENCE_LOG_PATH = ARTIFACT_DIR / "inference_log.json"

# MLflow Configuration
EXPERIMENT_NAME = "vial_image_classification"
MODEL_NAME = "cnn_model"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_STATE_DIR}/mlflow.db"
MLFLOW_ARTIFACT_LOCATION = f"./mlflow-state/artifacts"

# Training Configuration
MAX_DATE = "2024-01-31"
MIN_DATE = "2024-01-01"
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
THRESHOLD = 0.5
EARLY_STOP_PATIENCE = 2
SCHEDULER_PATIENCE = 3
RANDOM_STATE = 42

# Image Specifications
IMAGE_SIZE = (64, 64)
ORIGINAL_IMAGE_SIZE = (1600, 768)
NUM_CLASSES = 2
TRANSFORM_MEAN = [0.485, 0.456, 0.406]
TRANSFORM_STD = [0.229, 0.224, 0.225]

# S3 Configuration
BASE_URL = "https://itudsip.hel1.your-objectstorage.com/data"
S3_ENDPOINT = "https://itudsip.hel1.your-objectstorage.com"
S3_BUCKET = "data"