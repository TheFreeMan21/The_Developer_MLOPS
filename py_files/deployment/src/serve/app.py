import json
import urllib.request
import urllib.error
from pathlib import Path
from fastapi import FastAPI, Query, Request, Response
from src.config_deploy import upstream_host, upstream_port, model_path, inference_log_path, drift_baseline_stats_path, drift_default_days
from src.serve.drift import load_inference_logs, load_baseline_stats, detect_drift

app = FastAPI(docs_url=None, openapi_url=None, redoc_url=None)

def _upstream_health():
    try:
        with urllib.request.urlopen(f"http://{upstream_host}:{upstream_port}/health", timeout=1.0) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError):
        return None

@app.get("/health")
def health():
    upstream = _upstream_health()
    if upstream == 200:
        return Response(content='{"status": "ok"}', media_type="application/json")
    return Response(content='{"status": "loading"}', status_code=503, media_type="application/json")

@app.get("/version")
def version():
    version_file = Path(model_path) / "version.json"
    if version_file.exists():
        with open(version_file) as f:
            return json.load(f)
    return {"model_version": "unknown", "trained_at": None}

@app.post("/invocations")
@app.post("/infer")
async def invocations(request: Request):
    body = await request.body()
    req = urllib.request.Request(
        f"http://{upstream_host}:{upstream_port}/invocations",
        data=body,
        headers={"Content-Type": request.headers.get("content-type", "application/json")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return Response(content=resp.read(), status_code=resp.status, media_type=resp.headers.get("content-type"))
    except urllib.error.HTTPError as e:
        return Response(content=e.read(), status_code=e.code, media_type=e.headers.get("content-type"))
    except Exception as e:
        return Response(content=f'{{"status": "error", "detail": "{e}"}}', status_code=502, media_type="application/json")

@app.get("/drift")
def drift(days: int = Query(default=drift_default_days, ge=1)):
    logs = load_inference_logs(inference_log_path)
    baseline = load_baseline_stats(drift_baseline_stats_path)
    return detect_drift(logs, days=days, baseline_stats=baseline)