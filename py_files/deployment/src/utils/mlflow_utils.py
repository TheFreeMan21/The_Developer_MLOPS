import os
import mlflow

def setup_mlflow_tracking(mlflow_state_dir: str, experiment_name: str = None):
    os.makedirs(mlflow_state_dir, exist_ok=True)
    mlflow_tracking_uri = f"sqlite:///{mlflow_state_dir}/mlflow.db"
    mlflow_artifact_location = f"./{mlflow_state_dir}/artifacts"
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    if experiment_name is None:
        return
        
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=mlflow_artifact_location)
    mlflow.set_experiment(experiment_name)