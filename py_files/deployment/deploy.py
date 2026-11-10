import os
from mlflow.tracking import MlflowClient
from src.config_deploy import experiment_name, model_name, mlflow_state_dir, artifact_path, export_dir
from src.utils.mlflow_utils import setup_mlflow_tracking
from src.deploy.registry import get_best_run, get_production_model, compare_and_select_model, promote_model
from src.deploy.export import export_serving_model

def main():
    setup_mlflow_tracking(mlflow_state_dir, experiment_name)
    client = MlflowClient()

    best_run = get_best_run(experiment_name)
    prod_uri, prod_version, prod_run_id = get_production_model(client, model_name)
    
    selected_run_id = compare_and_select_model(best_run, prod_uri, prod_run_id)
    
    if selected_run_id:
        promote_model(client, model_name, selected_run_id, artifact_path)
        
    export_serving_model(client, model_name, experiment_name, artifact_path, export_dir)

if __name__ == "__main__":
    main()