"""Real HTTP/model service with a small, explicitly recorded CPU test configuration."""

import json
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from test_core import micro_config

import server
from brain_model import model

cfg = micro_config()
cfg["runtime"].update(cpu_network_sizes=[192], cpu_default_cells=192)
model.NETWORK = cfg
Path(".cache").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="ui-service-", dir=".cache") as directory:
    httpd = server.LocalServer((server.HOST, 0), Path(directory))
    stop = threading.Event()
    threading.Thread(
        target=server.run_experiment_worker, args=(httpd.experiments, stop), daemon=True
    ).start()
    print(json.dumps({"port": httpd.server_port}), flush=True)
    try:
        httpd.serve_forever()
    finally:
        stop.set()
        httpd.server_close()
