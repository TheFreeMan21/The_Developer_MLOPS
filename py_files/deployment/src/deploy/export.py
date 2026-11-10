import shutil
import tempfile
import json
from pathlib import Path
from datetime import datetime, timezone
import mlflow
from mlflow.tracking import MlflowClient

def resolve_model_uri(client: MlflowClient, model_name: str, experiment_name: str, artifact_path: str):
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="Production")
        source = getattr(mv, "source", None) or mv.storage_location
        return source, mv
    except Exception:
        pass

    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            mv = max(versions, key=lambda v: int(v.version))
            source = getattr(mv, "source", None) or mv.storage_location
            return source, mv
    except Exception:
        pass

    runs = mlflow.search_runs(experiment_names=[experiment_name], max_results=1, order_by=["start_time DESC"])
    if runs.empty:
        raise RuntimeError("No registered model and no runs in experiment - nothing to export")
    return f"runs:/{runs.iloc[0].run_id}/{artifact_path}", None

def export_serving_model(client: MlflowClient, model_name: str, experiment_name: str, artifact_path: str, export_dir: str):
    model_uri, model_version_obj = resolve_model_uri(client, model_name, experiment_name, artifact_path)
    
    Path(export_dir).parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=Path(export_dir).parent) as tmp:
        download_dir = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if Path(export_dir).exists():
            shutil.rmtree(export_dir)
        shutil.move(str(download_dir), export_dir)

    try:
        trained_at = datetime.fromtimestamp(
            client.get_run(model_version_obj.run_id).info.start_time / 1000, tz=timezone.utc
        ).isoformat()
        version_info = {"model_version": str(model_version_obj.version), "trained_at": trained_at}
    except Exception:
        version_info = {"model_version": "unknown", "trained_at": None}

    with open(Path(export_dir) / "version.json", "w") as f:
        json.dump(version_info, f, indent=2)