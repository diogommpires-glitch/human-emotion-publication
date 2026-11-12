"""Scientific identities prevent mixing fits across model or observation changes."""

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .store import sha256


def scientific_identity():
    from brain_model import model

    network = json.dumps(model.NETWORK, sort_keys=True, separators=(",", ":"), allow_nan=False)
    sources = {}
    for directory in (Path(model.__file__).parent, Path(__file__).parent):
        for path in sorted(directory.glob("*.py")):
            sources[directory.name + "/" + path.name] = sha256(path)
    packages = {}
    for name in ("numpy", "torch", "scipy", "scikit-learn", "mne", "nilearn", "nibabel"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "schema_version": 1,
        "protocol_version": "emotion-research-1",
        "network_sha256": hashlib.sha256(network.encode("utf-8")).hexdigest(),
        "source_sha256": sources,
        "packages": packages,
    }
