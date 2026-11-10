model_name = "cnn_model"
experiment_name = "vial_image_classification"
artifact_path = "model_pyfunc"

artifact_dir = "artifacts"
mlflow_state_dir = "mlflow-state"
drift_baseline_stats_path = f"{artifact_dir}/drift_baseline_stats.json"
inference_log_path = f"{artifact_dir}/inference_log.json"
model_results_path = f"{artifact_dir}/model_results.json"

drift_default_days = 7
drift_ks_pvalue_threshold = 0.05

export_dir = "models/model"
model_path = export_dir

port = 5001
upstream_port = 5002
upstream_host = "127.0.0.1"

PROD_ENDPOINT = "https://mlops-itu-jtk.ondigitalocean.app"