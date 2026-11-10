import time
import mlflow
from mlflow.tracking import MlflowClient

def get_best_run(experiment_name: str):
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_best = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    ).iloc[0]
    return experiment_best

def get_production_model(client: MlflowClient, model_name: str):
    try:
        model = client.get_model_version_by_alias(name=model_name, alias="Production")
        if model:
            return f"models:/{model_name}@Production", model.version, model.run_id
    except Exception:
        pass

    for model in client.search_model_versions(f"name='{model_name}'"):
        if dict(model).get('current_stage') == 'Production':
            return f"models:/{model_name}/{model.version}", model.version, model.run_id

    return None, None, None

def compare_and_select_model(best_run, prod_model_uri, prod_model_run_id):
    train_model_score = best_run.get("metrics.test_f1_score") or best_run.get("metrics.f1_score")
    run_id = best_run["run_id"]

    if prod_model_uri is not None:
        prod_run = mlflow.get_run(prod_model_run_id)
        prod_model_score = prod_run.data.metrics.get("test_f1_score") or prod_run.data.metrics.get("f1_score")

        if train_model_score <= prod_model_score:
            print("Current production model is better or equal. Keeping it.")
            return None

        print("New model is better. Promoting the new model.")
    else:
        print("No model in production. The trained model will be promoted.")
        
    return run_id

def wait_for_deployment(client: MlflowClient, model_name: str, version: str, stage: str = 'Staging', max_attempts: int = 20):
    for attempt in range(max_attempts):
        model_version_details = dict(client.get_model_version(name=model_name, version=version))
        if model_version_details['current_stage'] == stage:
            return True
        time.sleep(2)
    return False

def promote_model(client: MlflowClient, model_name: str, run_id: str, artifact_path: str):
    versions = [mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id]
    if not versions:
        model_details = mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name=model_name)
        version = model_details.version
    else:
        version = versions[0].version

    model_version_details = dict(client.get_model_version(name=model_name, version=version))
    if model_version_details['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Staging", archive_existing_versions=True
        )
        wait_for_deployment(client, model_name, version, 'Staging')

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production", archive_existing_versions=True
    )
    wait_for_deployment(client, model_name, version, 'Production')
    client.set_registered_model_alias(name=model_name, alias="Production", version=str(version))
    client.set_model_version_tag(model_name, version, "deployed_to", "production")
    
    return version