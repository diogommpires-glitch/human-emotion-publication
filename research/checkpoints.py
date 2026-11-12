"""Resume completed nested fits without repeating prior expensive simulation methods."""

import json
import os
import uuid
from pathlib import Path

import numpy as np

from .store import atomic_json, load_arrays, sha256


class EvaluationCheckpoint:
    def __init__(self, directory, identity, budget_bytes):
        self.root = Path(directory)
        self.path = self.root / "evaluation-progress.json"
        self.budget_bytes = budget_bytes
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
            if self.manifest["identity"] != identity:
                raise ValueError("evaluation resume source/spec differs from its checkpoint")
        else:
            self.manifest = {"identity": identity, "completed": {}}

    def get(self, seed, method):
        item = self.manifest["completed"].get(f"{seed}:{method}")
        if item is None:
            return None
        path = self.root / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError("evaluation checkpoint hash mismatch")
        archive = load_arrays(path, self.budget_bytes)
        return {**item["metrics"], "predictions": archive["predictions"]}

    def save(self, seed, method, result):
        if not np.isfinite(result["predictions"]).all():
            raise ValueError("cannot checkpoint non-finite predictions")
        used = sum(path.stat().st_size for path in self.root.iterdir() if path.is_file())
        if used + result["predictions"].nbytes + 65536 > self.budget_bytes:
            raise ValueError("evaluation disk budget exceeded")
        name = f"evaluation-{seed}-{method}-{uuid.uuid4().hex}.npz"
        path = self.root / name
        with path.open("xb") as stream:
            np.savez_compressed(stream, predictions=result["predictions"])
            stream.flush()
            os.fsync(stream.fileno())
        self.manifest["completed"][f"{seed}:{method}"] = {
            "path": name,
            "sha256": sha256(path),
            "metrics": {key: value for key, value in result.items() if key != "predictions"},
        }
        atomic_json(self.path, self.manifest)
