import os
import sys
import atexit
import subprocess
import uvicorn
from src.config_deploy import model_path, upstream_host, upstream_port, port
from src.serve.app import app

def start_scoring_server():
    env = os.environ.copy()
    interpreter_bin = os.path.dirname(os.path.abspath(sys.executable))
    env["PATH"] = os.pathsep.join([interpreter_bin, env.get("PATH", "")])
    
    proc = subprocess.Popen([
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", model_path,
        "-p", str(upstream_port),
        "--workers", "1",
        "--host", upstream_host,
        "--no-conda",
    ], env=env)
    
    atexit.register(lambda: proc.poll() is None and proc.terminate())
    return proc

if __name__ == "__main__":
    scoring_proc = start_scoring_server()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")