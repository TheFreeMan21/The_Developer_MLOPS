import os
import json
import torch
import numpy as np
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import time

from src.config import *
from src.utils.mlflow_utils import setup_mlflow_tracking
from src.data.make_dataset import pull_data_from_dvc, load_image_dataset
from src.data.build_features import define_transforms, split_dataset, create_dataloaders, calculate_image_statistics
from src.models.model import SimpleCNN
from src.models.train_model import train_cnn_model, tune_threshold, evaluate_cnn_model, compute_baseline_statistics
from src.models.wrapper import SimpleLoggingWrapper

def wait_until_ready(model_name, model_version, max_attempts=10):
    client = MlflowClient()
    for attempt in range(max_attempts):
        try:
            model_version_details = client.get_model_version(name=model_name, version=model_version)
            status = ModelVersionStatus.from_string(model_version_details.status)
            if status == ModelVersionStatus.READY:
                return True
        except Exception as e:
            print(f"Error checking status: {e}")
        time.sleep(1)
    return False

def main():
    # 1. Setup Directories
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(MLFLOW_STATE_DIR, exist_ok=True)
    os.makedirs(IMAGE_DATA_DIR, exist_ok=True)

    # 2. Setup Data and MLflow
    setup_mlflow_tracking(EXPERIMENT_NAME)
    pull_data_from_dvc()
    image_dataset = load_image_dataset()

    # 3. Build Features
    train_transforms, val_transforms = define_transforms()
    train_dataset, val_dataset, test_dataset = split_dataset(image_dataset, random_seed=RANDOM_STATE)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, train_transforms, val_transforms
    )

    # Save initial artifacts
    image_stats = calculate_image_statistics(train_loader)
    with open(IMAGE_STATISTICS_PATH, "w") as f:
        json.dump(image_stats, f, indent=2)

    class_names = {0: "good", 1: "defective"}
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="training_pipeline") as run:
        mlflow.log_params({
            "model_architecture": "SimpleCNN",
            "batch_size": BATCH_SIZE,
            "num_epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "initial_threshold": THRESHOLD,
            "num_classes": NUM_CLASSES,
            "random_seed": RANDOM_STATE
        })

        # 4. Train and Evaluate
        model = SimpleCNN()
        cnn_model, history = train_cnn_model(model, train_loader, val_loader, device, random_seed=RANDOM_STATE)
        optimal_threshold = tune_threshold(cnn_model, val_loader, device)
        test_results = evaluate_cnn_model(cnn_model, test_loader, device, threshold=optimal_threshold)
        
        # 5. Baseline Drift Statistics
        baseline_stats = compute_baseline_statistics(cnn_model, val_loader, device, threshold=optimal_threshold)
        with open(DRIFT_BASELINE_STATS_PATH, 'w') as f:
            json.dump(baseline_stats, f, indent=2)
            
        mlflow.log_metric("baseline_confidence_mean", baseline_stats['confidence_statistics']['mean'])
        mlflow.log_artifact(DRIFT_BASELINE_STATS_PATH, "drift_detection")

        # 6. Save Artifacts Locally
        torch.save({
            'model_state_dict': cnn_model.state_dict(),
            'architecture': 'SimpleCNN',
            'threshold': optimal_threshold,
            'num_classes': NUM_CLASSES,
        }, PYTORCH_MODEL_PATH)

        with open(MODEL_RESULTS_PATH, "w") as f:
            json.dump(test_results, f, indent=2)

        mlflow.log_param("classification_threshold", optimal_threshold)

        # 7. Log and Register Pyfunc Wrapper
        wrapper = SimpleLoggingWrapper(model=cnn_model, threshold=optimal_threshold)
        mlflow.pyfunc.log_model("model_pyfunc", python_model=wrapper)

        model_uri = f"runs:/{run.info.run_id}/model_pyfunc"
        model_details = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
        
        if wait_until_ready(model_details.name, model_details.version):
            client = MlflowClient()
            client.transition_model_version_stage(
                name=MODEL_NAME,
                version=model_details.version,
                stage="Staging",
                archive_existing_versions=True,
            )
            client.set_model_version_tag(MODEL_NAME, model_details.version, "deployed_to", "staging")
            client.set_model_version_tag(MODEL_NAME, model_details.version, "task", "classification")
            client.set_registered_model_alias(name=MODEL_NAME, alias="best", version=str(model_details.version))
            print(f"Successfully transitioned {MODEL_NAME} v{model_details.version} to Staging")

if __name__ == "__main__":
    main()