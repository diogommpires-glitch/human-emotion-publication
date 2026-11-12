# HTTP and checkpoint validation use ValueError consistently for invalid public data.
# ruff: noqa: TRY004
"""Private transactional learned weights, using JSON and dense safetensors only."""

import hashlib
import json
import math
import os
import re
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file

from . import model, simulator
from .config import configuration_hash, validate_network
from .learning import parse_learning
from .topology import contact_matrix, dynamics_dimensions

FORMAT_VERSION = 1
DEFAULT_BUDGET = 8 * 1024**3
GROUPS = ("local_glu", "local_gaba", "long_glu", "long_gaba")
CELL_ARRAYS = ("region_of_cell", "interneuron", "vip_cell", "gaba_cell", "bin_of_cell")
CONTACT_ARRAYS = ("source", "target", "weight", "plastic", "block")


class SessionConflict(ValueError):
    """A stale revision or an idempotency key reused with different numerical input."""


class CheckpointTooLarge(ValueError):
    """Writing a checkpoint would exceed the local storage budget."""


class StorageError(OSError):
    """A checkpoint could not be durably written."""


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _read_json(path: Path):
    if path.stat().st_size > 32 * 1024**2:
        raise CheckpointTooLarge(f"JSON file exceeds 32 MiB: {path.name}")
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"Non-finite JSON {value}")),
    )


def _write_json(path: Path, value) -> None:
    with path.open("xb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())


def _save_tensors(path: Path, tensors: dict) -> None:
    save_file(
        {key: value.detach().cpu().contiguous().clone() for key, value in tensors.items()}, path
    )
    with path.open("r+b") as stream:
        os.fsync(stream.fileno())


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_tensor_headers(handle, schema):
    keys = tuple(handle.keys())
    if set(keys) != set(schema):
        raise ValueError("Checkpoint tensor keys differ from schema.")
    allocated = 0
    shapes = {}
    for key in keys:
        view = handle.get_slice(key)
        shape, dtype = tuple(view.get_shape()), view.get_dtype()
        expected_shape, expected_dtype = schema[key]
        if (
            dtype != expected_dtype
            or len(shape) != len(expected_shape)
            or any(
                want is not None and want != actual
                for want, actual in zip(expected_shape, shape, strict=True)
            )
        ):
            raise ValueError(f"Checkpoint tensor shape or dtype mismatch for {key}.")
        allocated += math.prod(shape) * {"F32": 4, "I64": 8, "BOOL": 1}[dtype]
        shapes[key] = shape
    if allocated > DEFAULT_BUDGET:
        raise CheckpointTooLarge("Tensor headers exceed the allocation budget.")
    for group in GROUPS:
        contact_shapes = {shapes[key] for key in shapes if key.startswith(f"contacts.{group}.")}
        if len(contact_shapes) > 1:
            raise ValueError("Contact header shapes disagree.")
    return keys


def _load_tensors(path: Path, expected_hash: str, schema: dict) -> dict:
    if path.stat().st_size > DEFAULT_BUDGET:
        raise CheckpointTooLarge("Tensor file exceeds the checkpoint allocation budget.")
    if _digest(path) != expected_hash:
        raise ValueError(f"Checkpoint integrity failure: {path.name}")
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = _validate_tensor_headers(handle, schema)
            tensors = {key: handle.get_tensor(key) for key in keys}
    except SafetensorError as exc:
        raise ValueError(f"Invalid safetensors file: {path.name}") from exc
    if any(not torch.isfinite(value).all() for value in tensors.values()):
        raise ValueError(f"Non-finite tensor in {path.name}")
    return tensors


def _topology_schema(manifest: dict) -> dict:
    cfg, n = manifest["network"], manifest["neurons"]
    validate_network(cfg)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("Invalid neuron count.")
    rows = {row["id"]: row for row in manifest["rows"]}
    schema: dict[str, tuple[tuple[int | None, ...], str]] = {
        key: (
            (n,),
            {
                "interneuron": "BOOL",
                "vip_cell": "BOOL",
                "gaba_cell": "BOOL",
                "region_of_cell": "I64",
                "bin_of_cell": "I64",
            }[key],
        )
        for key in CELL_ARRAYS
    }
    schema.update(
        {
            f"params.{key}": ((n,), "F32")
            for key in next(iter(cfg["cell_types"].values()))
            if key != "threshold_jitter_mV"
        }
    )
    schema.update(
        {
            f"assemblies.{channel['id']}:{region}": ((rows[region]["size"],), "BOOL")
            for channel in cfg["inputs"]
            for region in channel["targets"]
        }
    )
    schema.update(
        {
            f"contacts.{group}.{key}": (
                (None,),
                {
                    "weight": "F32",
                    "plastic": "BOOL",
                    "source": "I64",
                    "target": "I64",
                    "block": "I64",
                }[key],
            )
            for group in GROUPS
            for key in CONTACT_ARRAYS
        }
    )
    return schema


def _id(value, name="session id") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,80}", value):
        raise ValueError(
            f"{name} must contain 1..80 ASCII letters, digits, underscores or hyphens."
        )
    return value


def _revision(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 1_000_000:
        raise ValueError("from_revision must be an integer in [0,1000000].")
    return value


def _revision_path(directory: Path, revision: int) -> Path:
    return directory / f"revision-{revision:08d}"


def pack_topology(topology: dict) -> dict:
    tensors = {key: topology[key] for key in CELL_ARRAYS}
    tensors.update({f"params.{key}": value for key, value in topology["params"].items()})
    tensors.update({f"assemblies.{key}": value for key, value in topology["assemblies"].items()})
    for group, contacts in topology["contacts"].items():
        tensors.update(
            {f"contacts.{group}.{key}": value for key, value in contacts.items() if key != "slot"}
        )
    return tensors


def _matches_shape(actual, expected):
    return len(actual) == len(expected) and all(
        want is None or want == size for size, want in zip(actual, expected, strict=True)
    )


def _validate_topology_tensors(manifest, tensors):
    schema = _topology_schema(manifest)
    if set(tensors) != set(schema):
        raise ValueError("Checkpoint tensors do not match the topology schema.")
    dtypes = {"F32": torch.float32, "I64": torch.int64, "BOOL": torch.bool}
    for key, (shape, dtype) in schema.items():
        value = tensors[key]
        if value.dtype != dtypes[dtype] or not _matches_shape(tuple(value.shape), shape):
            raise ValueError(f"Invalid topology tensor shape or dtype: {key}.")
        if not torch.isfinite(value).all():
            raise ValueError(f"Non-finite topology tensor: {key}.")


def _validate_region_parameters(cfg, row, expected_classes, tensors, param_names):
    cells = slice(row["start"], row["start"] + row["size"])
    cell_types = (row["cell_type"], "fast_spiking", "vip_disinhibitory")
    for class_index, class_type in enumerate(cell_types):
        selected = (expected_classes["bin_of_cell"] % 3) == class_index
        for key in param_names - {"v_threshold_mV"}:
            if not torch.all(
                tensors[f"params.{key}"][cells][selected] == cfg["cell_types"][class_type][key]
            ):
                raise ValueError(f"Checkpoint {key} disagrees with declared cell type.")


def _validate_region_classes(cfg, row, index, tensors, param_names):
    region = tensors["region_of_cell"]
    cells = slice(row["start"], row["start"] + row["size"])
    if not torch.all(region[cells] == index):
        raise ValueError("Cell region indices disagree with layout.")
    relative = torch.arange(row["size"])
    sst = (relative >= row["projection"]) & (relative < row["projection"] + row["sst_count"])
    vip = relative >= row["projection"] + row["sst_count"]
    gaba = sst | vip | ((relative < row["projection"]) & row["gaba_projection"])
    expected_classes = {
        "interneuron": sst,
        "vip_cell": vip,
        "gaba_cell": gaba,
        "bin_of_cell": 3 * index + sst.to(torch.int64) + 2 * vip.to(torch.int64),
    }
    for key, expected_values in expected_classes.items():
        if not torch.equal(tensors[key][cells], expected_values):
            raise ValueError(f"Checkpoint {key} disagrees with declared cell classes.")
    _validate_region_parameters(cfg, row, expected_classes, tensors, param_names)


def _validate_assemblies(cfg, rows, tensors):
    by_id = {row["id"]: row for row in rows}
    for channel in cfg["inputs"]:
        for region_id in channel["targets"]:
            value = tensors[f"assemblies.{channel['id']}:{region_id}"]
            if value.dtype != torch.bool or value.shape != (by_id[region_id]["size"],):
                raise ValueError("Invalid realised stimulus assembly.")


def _validate_contact_arrays(c, n, group):
    count = c["weight"].numel()
    if any(value.shape != (count,) for value in c.values()):
        raise ValueError("Contact array shapes disagree.")
    if (
        c["source"].dtype != torch.int64
        or c["target"].dtype != torch.int64
        or c["block"].dtype != torch.int64
        or c["plastic"].dtype != torch.bool
        or c["weight"].dtype != torch.float32
    ):
        raise ValueError("Invalid contact tensor dtype.")
    if count and (
        c["source"].min() < 0
        or c["target"].min() < 0
        or c["source"].max() >= n
        or c["target"].max() >= n
        or c["weight"].min() <= 0
    ):
        raise ValueError("Contact indices or weights are outside their physical domain.")
    if group.endswith("gaba") and c["plastic"].any():
        raise ValueError("Inhibitory contacts cannot be plastic.")


def _validate_contact_blocks(c, cfg, rows, group):
    count = c["weight"].numel()
    parts = [part for part in simulator._synapse_parts(cfg, rows) if part[0] == group]
    if count:
        if c["block"].min() < 0 or c["block"].max() >= len(parts):
            raise ValueError("Invalid contact block identity.")
        block_table = torch.tensor([part[1:] for part in parts], dtype=torch.float64)
        block = block_table[c["block"]]
        valid = (
            (c["target"] >= block[:, 0])
            & (c["target"] < block[:, 0] + block[:, 1])
            & (c["source"] >= block[:, 2])
            & (c["source"] < block[:, 2] + block[:, 3])
            & (c["source"] != c["target"])
            & (c["plastic"] == block[:, 6].bool())
        )
        jitter = cfg["synapses"]["weights"]["jitter"]
        valid &= (c["weight"] >= block[:, 5] * (1 - jitter) - 1e-7) & (
            c["weight"] <= block[:, 5] * (1 + jitter) + 1e-7
        )
        if not valid.all():
            raise ValueError(
                "Contact pools, base weights or plasticity disagree with declared blocks."
            )


def unpack_topology(manifest: dict, tensors: dict, device) -> dict:
    cfg, n = manifest["network"], manifest["neurons"]
    _validate_topology_tensors(manifest, tensors)
    if not 1 <= n <= max(cfg["runtime"]["gpu_network_sizes"]):
        raise ValueError("Invalid checkpoint neuron count.")
    rows = manifest["rows"]
    if rows != simulator.layout(cfg, n):
        raise ValueError("Checkpoint cell layout does not match its configuration.")
    param_names = set(next(iter(cfg["cell_types"].values()))) - {"threshold_jitter_mV"}
    for index, row in enumerate(rows):
        _validate_region_classes(cfg, row, index, tensors, param_names)
    _validate_assemblies(cfg, rows, tensors)
    topology = {
        "n": n,
        "config_hash": configuration_hash(cfg),
        "dynamics": dynamics_dimensions(cfg),
        "device": device,
        "rows": rows,
        "region_count": len(rows),
        "build_seconds": 0.0,
        "params": {key: tensors[f"params.{key}"].to(device) for key in param_names},
        "assemblies": {
            key.removeprefix("assemblies."): value.to(device)
            for key, value in tensors.items()
            if key.startswith("assemblies.")
        },
        "contacts": {},
        "matrices": {},
        "synapse_counts": {},
        **{key: tensors[key].to(device) for key in CELL_ARRAYS},
    }
    for group in GROUPS:
        c = {key: tensors[f"contacts.{group}.{key}"] for key in CONTACT_ARRAYS}
        _validate_contact_arrays(c, n, group)
        _validate_contact_blocks(c, cfg, rows, group)
        c = {key: value.to(device) for key, value in c.items()}
        topology["matrices"][group], c["slot"] = contact_matrix(c, n)
        topology["contacts"][group] = c
        topology["synapse_counts"][group] = c["weight"].numel()
    topology["synapse_count"] = sum(topology["synapse_counts"].values())
    return topology


def _validate_weights(weights: dict, topology: dict) -> None:
    if set(weights) != set(GROUPS):
        raise ValueError("Checkpoint weight groups differ from topology.")
    for group, weight in weights.items():
        contacts = topology["contacts"][group]
        base = contacts["weight"].cpu()
        if (
            weight.dtype != torch.float32
            or weight.shape != base.shape
            or not torch.isfinite(weight).all()
        ):
            raise ValueError("Invalid learned weight shape, dtype or value.")
        if torch.any(weight < 0.1 * base) or torch.any(weight > 2 * base):
            raise ValueError("Learned weights violate bounds relative to initial contacts.")
        if not torch.equal(weight[~contacts["plastic"].cpu()], base[~contacts["plastic"].cpu()]):
            raise ValueError("Checkpoint modifies fixed contacts.")


def _validate_identity(identity: dict, action: str, manifest: dict) -> None:
    if action == "reset":
        if (
            set(identity) != {"id", "from_revision", "run_id", "action"}
            or identity["id"] != manifest["id"]
        ):
            raise ValueError("Invalid reset identity schema.")
        _revision(identity["from_revision"])
        _id(identity["run_id"], "run_id")
        return
    if not isinstance(identity, dict) or set(identity) != {"request", "session"}:
        raise ValueError("Invalid saved scientific request schema.")
    request, session = identity["request"], identity["session"]
    if (
        set(request) != {"inputs", "neurons", "seed", "person", "event", "learning_config"}
        or request["person"] != ""
        or request["event"] != ""
    ):
        raise ValueError("Saved request contains annotations or unknown fields.")
    if (
        set(session) != {"id", "from_revision", "run_id", "action"}
        or session["id"] != manifest["id"]
        or session["action"] != action
    ):
        raise ValueError("Saved session identity is inconsistent.")
    _id(session["run_id"], "run_id")
    _revision(session["from_revision"])
    _validate_saved_request(request, manifest)


def _validate_saved_request(request, manifest):
    channels = {row["id"] for row in manifest["network"]["inputs"]}
    if set(request["inputs"]) != channels or request["neurons"] != manifest["neurons"]:
        raise ValueError("Saved numerical request does not match the checkpoint.")
    for value in request["inputs"].values():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("Invalid saved input level.")
    if (
        isinstance(request["seed"], bool)
        or not isinstance(request["seed"], int)
        or not 0 <= request["seed"] < 2**31
    ):
        raise ValueError("Invalid saved random seed.")
    parse_learning({"learning": request["learning_config"]})


def _validate_response(response: dict, identity: dict) -> None:
    keys = {
        "model",
        "model_status",
        "event",
        "person",
        "inputs_used",
        "timeline",
        "dimensions",
        "evidence",
        "emotions",
        "regions",
        "chemistry",
        "spikes",
        "eeg",
        "simulation",
        "interpretation",
        "caveat",
        "session",
        "references",
    }
    if set(response) != keys or response["event"] != "" or response["person"] != "":
        raise ValueError("Saved response contains annotations or unknown fields.")
    expected_session = {
        "id",
        "start_revision",
        "revision",
        "head_revision",
        "action",
        "run_id",
        "committed",
    }
    session = response["session"]
    if set(session) != expected_session or any(
        session[key] != identity["session"][key] for key in ("id", "action", "run_id")
    ):
        raise ValueError("Saved response identity is inconsistent.")
    start = identity["session"]["from_revision"]
    committed = identity["session"]["action"] == "learn"
    expected_revision = start + 1 if committed else start
    if (
        session["start_revision"] != start
        or session["revision"] != expected_revision
        or session["committed"] is not committed
    ):
        raise ValueError("Saved response revision does not match its action and start revision.")
    head = _revision(session["head_revision"])
    if head < expected_revision or (committed and head != expected_revision):
        raise ValueError("Saved response HEAD is inconsistent.")
    _json_bytes(response)


def _remap_identity(identity: dict, session_id: str, action: str) -> None:
    if action == "reset":
        identity["id"] = session_id
    else:
        identity["session"]["id"] = session_id


def _session_numerical_request(payload, manifest, device):
    numeric = {**payload, "neurons": manifest["neurons"]}
    if not any(key in numeric for key in ("learning", "stdp", "td_learning")):
        numeric["learning"] = manifest["learning"]
    if numeric.get("seed") in (None, ""):
        raise ValueError("Session runs require an explicit seed for restart-safe idempotency.")
    request = model.parse_request(numeric, device, cfg=manifest["network"])
    request["event"] = request["person"] = ""
    return request


def _import_manifest(source):
    manifest = _read_json(source / "manifest.json")
    required = {
        "version",
        "id",
        "neurons",
        "created_at",
        "learning",
        "network",
        "rows",
        "training_reference",
        "topology_sha256",
    }
    if set(manifest) != required:
        raise ValueError("Checkpoint manifest schema mismatch.")
    parse_learning({"learning": manifest["learning"]})
    _id(manifest["id"])
    _json_bytes(manifest)
    return manifest


def _validate_initial_revision(row):
    if (
        row["action"] != "create"
        or row["request"] is not None
        or row["run_id"]
        or row["fingerprint"]
    ):
        raise ValueError("Invalid initial revision.")


def _validate_committed_revision(row, manifest, run_ids):
    if row["action"] not in ("learn", "reset") or row["run_id"] in run_ids:
        raise ValueError("Invalid revision action or duplicate run id.")
    _validate_identity(row["request"], row["action"], manifest)
    identity = row["request"] if row["action"] == "reset" else row["request"]["session"]
    if (
        identity["run_id"] != row["run_id"]
        or identity["from_revision"] != row["parent_revision"]
        or identity["action"] != row["action"]
    ):
        raise ValueError("Revision metadata disagrees with its saved request.")
    if row["fingerprint"] != hashlib.sha256(_json_bytes(row["request"])).hexdigest():
        raise ValueError("Saved request fingerprint mismatch.")
    run_ids.add(row["run_id"])


def _validate_revision_record(row, revision, manifest, run_ids):
    if set(row) != {
        "revision",
        "parent_revision",
        "action",
        "run_id",
        "fingerprint",
        "created_at",
        "weights_sha256",
        "request",
    }:
        raise ValueError("Checkpoint revision schema mismatch.")
    if row["revision"] != revision or row["parent_revision"] != (
        revision - 1 if revision else None
    ):
        raise ValueError("Checkpoint revision chain is invalid.")
    if revision == 0:
        _validate_initial_revision(row)
    else:
        _validate_committed_revision(row, manifest, run_ids)


def _read_saved_run(record_path, manifest, run_ids):
    row = _read_json(record_path)
    if set(row) != {"fingerprint", "request"}:
        raise ValueError("Saved run schema mismatch.")
    identity = row["request"]
    session = identity.get("session") if isinstance(identity, dict) else None
    if not isinstance(session, dict):
        raise ValueError("Saved run has no session identity.")
    action = session.get("action")
    if action not in ("evaluate", "replay"):
        raise ValueError("Invalid noncommitting run action.")
    _validate_identity(identity, action, manifest)
    run_id = identity["session"]["run_id"]
    if (
        record_path.parent.name != run_id
        or run_id in run_ids
        or row["fingerprint"] != hashlib.sha256(_json_bytes(identity)).hexdigest()
    ):
        raise ValueError("Invalid saved run identity or fingerprint.")
    run_ids.add(run_id)
    response = _read_json(record_path.parent / "response.json")
    _validate_response(response, identity)
    return row, response


def _stage_import(source, stage, session_id, manifest, head, files, records):
    for item in files:
        if item.suffix == ".safetensors":
            target = stage / item.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item, target)
    manifest["id"] = session_id
    _write_json(stage / "manifest.json", manifest)
    _write_json(stage / "HEAD.json", {"revision": head})
    for relative, filename, row, response in records:
        directory = stage / relative
        directory.mkdir(exist_ok=True, parents=True)
        if row["request"] is not None:
            action = row["action"] if "action" in row else row["request"]["session"]["action"]
            _remap_identity(row["request"], session_id, action)
            row["fingerprint"] = hashlib.sha256(_json_bytes(row["request"])).hexdigest()
        _write_json(directory / filename, row)
        if response is not None:
            response["session"]["id"] = session_id
            _write_json(directory / "response.json", response)


def _session_response(cfg, topology, weights, request, action):
    reference = simulator.run(
        cfg,
        topology,
        {row["id"]: row["default"] for row in cfg["inputs"]},
        model.REFERENCE_SEED,
        duration_s=cfg["runtime"]["reference_duration_seconds"],
        record=False,
        weights=weights,
    )["baseline"]
    learning = parse_learning({}) if action == "evaluate" else request["learning_config"]
    raw, state = simulator.run_episode(
        cfg,
        topology,
        request["inputs"],
        request["seed"],
        learning_config=learning,
        weights=weights,
        record=True,
    )
    response = model.build_response(request, raw, reference, topology, cfg=cfg)
    return response, state, reference


class SessionStore:
    """One execution lock spans simulation, response validation and atomic publication."""

    def __init__(self, root, *, budget_bytes: int = DEFAULT_BUDGET):
        self.root = Path(root).resolve()
        self.budget_bytes = budget_bytes
        self.root.mkdir(parents=True, exist_ok=True)

    def _budget(self, extra: int = 0) -> None:
        used = sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())
        if used + extra > self.budget_bytes:
            raise CheckpointTooLarge(
                f"Checkpoint storage needs {used + extra} bytes; budget is {self.budget_bytes} bytes. No sessions were deleted."
            )

    def _directory(self, session_id: str) -> Path:
        directory = self.root / _id(session_id)
        if not (directory / "HEAD.json").is_file():
            raise ValueError(f"Unknown session {session_id}.")
        return directory

    def _head(self, directory: Path) -> int:
        return _revision(_read_json(directory / "HEAD.json")["revision"])

    def _publish(self, directory: Path, revision: int) -> None:
        pending = directory / f".HEAD-{uuid.uuid4().hex}.json"
        _write_json(pending, {"revision": revision})
        os.replace(pending, directory / "HEAD.json")

    def _metadata(self, directory: Path) -> dict:
        manifest = _read_json(directory / "manifest.json")
        head = self._head(directory)
        revisions = []
        for revision in range(head + 1):
            row = _read_json(_revision_path(directory, revision) / "revision.json")
            revisions.append(
                {
                    key: row[key]
                    for key in ("revision", "parent_revision", "action", "run_id", "created_at")
                }
            )
        return {key: manifest[key] for key in ("id", "neurons", "created_at", "learning")} | {
            "head_revision": head,
            "revisions": revisions,
        }

    def list(self) -> list[dict]:
        return [
            self._metadata(path.parent)
            for path in sorted(self.root.glob("*/HEAD.json"))
            if not path.parent.name.startswith(".")
        ]

    def get(self, session_id: str) -> dict:
        return self._metadata(self._directory(session_id))

    def get_run(self, session_id: str, run_id: str) -> dict:
        directory = self._directory(session_id)
        _id(run_id, "run_id")
        saved = directory / "runs" / run_id / "response.json"
        if saved.is_file():
            return _read_json(saved)
        for revision in range(self._head(directory) + 1):
            path = _revision_path(directory, revision)
            row = _read_json(path / "revision.json")
            if row["run_id"] == run_id and row["action"] == "learn":
                return _read_json(path / "response.json")
        raise ValueError(f"Unknown committed run {run_id}.")

    def create(self, payload: dict) -> dict:
        with simulator.RUN_LOCK:
            device = simulator.pick_device()
            request = model.parse_request(payload, device)
            cfg = model.NETWORK
            topology = simulator.build_topology(cfg, request["neurons"], device)
            tensors = pack_topology(topology)
            self._budget(
                sum(value.numel() * value.element_size() for value in tensors.values()) * 2
                + 128 * 1024
            )
            session_id = uuid.uuid4().hex
            stage = self.root / f".create-{session_id}"
            stage.mkdir()
            _save_tensors(stage / "topology.safetensors", tensors)
            reference = simulator.run(
                cfg,
                topology,
                {row["id"]: row["default"] for row in cfg["inputs"]},
                model.REFERENCE_SEED,
                duration_s=cfg["runtime"]["reference_duration_seconds"],
                record=False,
            )["baseline"]
            manifest = {
                "version": FORMAT_VERSION,
                "id": session_id,
                "neurons": topology["n"],
                "created_at": datetime.now(UTC).isoformat(),
                "learning": request["learning_config"],
                "network": cfg,
                "rows": topology["rows"],
                "training_reference": reference,
                "topology_sha256": _digest(stage / "topology.safetensors"),
            }
            _write_json(stage / "manifest.json", manifest)
            weights = {key: value["weight"] for key, value in topology["contacts"].items()}
            self._write_revision(stage, 0, None, "create", "", "", weights, None, None)
            self._publish(stage, 0)
            self._budget()
            directory = self.root / session_id
            os.replace(stage, directory)
            return self._metadata(directory)

    def _write_revision(
        self, directory, revision, parent, action, run_id, fingerprint, weights, request, response
    ):
        stage = directory / f".stage-{uuid.uuid4().hex}"
        stage.mkdir()
        _save_tensors(stage / "weights.safetensors", weights)
        metadata = {
            "revision": revision,
            "parent_revision": parent,
            "action": action,
            "run_id": run_id,
            "fingerprint": fingerprint,
            "created_at": datetime.now(UTC).isoformat(),
            "weights_sha256": _digest(stage / "weights.safetensors"),
            "request": request,
        }
        _write_json(stage / "revision.json", metadata)
        if response is not None:
            _write_json(stage / "response.json", response)
        self._budget()
        os.replace(stage, _revision_path(directory, revision))

    def _load(self, directory, revision, device):
        manifest = _read_json(directory / "manifest.json")
        if manifest["version"] != FORMAT_VERSION:
            raise ValueError("Unsupported checkpoint version.")
        topology = unpack_topology(
            manifest,
            _load_tensors(
                directory / "topology.safetensors",
                manifest["topology_sha256"],
                _topology_schema(manifest),
            ),
            device,
        )
        path = _revision_path(directory, revision)
        metadata = _read_json(path / "revision.json")
        weights = _load_tensors(
            path / "weights.safetensors",
            metadata["weights_sha256"],
            {key: (tuple(c["weight"].shape), "F32") for key, c in topology["contacts"].items()},
        )
        _validate_weights(weights, topology)
        return manifest, topology, {key: value.to(device) for key, value in weights.items()}

    def _retry(self, directory, run_id, fingerprint):
        record = directory / "runs" / run_id
        if record.is_dir():
            metadata = _read_json(record / "request.json")
            if metadata["fingerprint"] != fingerprint:
                raise SessionConflict(
                    "run_id is already bound to a different numerical request or action."
                )
            return _read_json(record / "response.json")
        head = self._head(directory)
        for path in sorted(directory.glob("revision-*/revision.json")):
            metadata = _read_json(path)
            if metadata["run_id"] != run_id:
                continue
            if metadata["fingerprint"] != fingerprint:
                raise SessionConflict(
                    "run_id is already bound to a different numerical request or action."
                )
            revision = metadata["revision"]
            if revision > head:
                if metadata["parent_revision"] != head:
                    raise SessionConflict("Unpublished run no longer starts at HEAD.")
                self._load(directory, revision, torch.device("cpu"))
                self._publish(directory, revision)
            if metadata["action"] == "reset":
                return self._metadata(directory)
            return _read_json(path.parent / "response.json")
        return None

    def run(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Expected a JSON object.")
        if payload.get("session") is None:
            return model.simulate(payload)
        with simulator.RUN_LOCK:
            session = payload["session"]
            if not isinstance(session, dict) or set(session) != {
                "id",
                "from_revision",
                "run_id",
                "action",
            }:
                raise ValueError("session requires id, from_revision, run_id and action.")
            directory = self._directory(session["id"])
            revision = _revision(session["from_revision"])
            run_id = _id(session["run_id"], "run_id")
            action = session["action"]
            if action not in ("learn", "evaluate", "replay"):
                raise ValueError("Session action must be learn, evaluate or replay.")
            manifest = _read_json(directory / "manifest.json")
            device = simulator.pick_device()
            request = _session_numerical_request(payload, manifest, device)
            identity = {"request": request, "session": session}
            fingerprint = hashlib.sha256(_json_bytes(identity)).hexdigest()
            cached = self._retry(directory, run_id, fingerprint)
            if cached is not None:
                return cached
            head = self._head(directory)
            if revision > head or (action == "learn" and revision != head):
                raise SessionConflict(f"learn requires HEAD {head}; requested revision {revision}.")
            manifest, topology, weights = self._load(directory, revision, device)
            cfg = manifest["network"]
            response, state, reference = _session_response(cfg, topology, weights, request, action)
            next_revision = head + 1 if action == "learn" else revision
            response["session"] = {
                "id": session["id"],
                "start_revision": revision,
                "revision": next_revision,
                "head_revision": next_revision if action == "learn" else head,
                "action": action,
                "run_id": run_id,
                "committed": action == "learn",
            }
            response["references"] = {
                "start_weights": reference,
                "frozen_training": manifest["training_reference"],
            }
            _json_bytes(response)
            if action == "learn":
                learned = {key: state.tensors[f"learning.weights.{key}"] for key in GROUPS}
                self._budget(
                    sum(value.numel() * value.element_size() for value in learned.values())
                    + len(_json_bytes(response))
                    + 16384
                )
                self._write_revision(
                    directory,
                    next_revision,
                    head,
                    action,
                    run_id,
                    fingerprint,
                    learned,
                    identity,
                    response,
                )
                self._publish(directory, next_revision)
            else:
                self._budget(len(_json_bytes(response)) + len(_json_bytes(identity)) + 4096)
                stage = directory / f".run-{uuid.uuid4().hex}"
                stage.mkdir()
                _write_json(
                    stage / "request.json", {"fingerprint": fingerprint, "request": identity}
                )
                _write_json(stage / "response.json", response)
                destination = directory / "runs" / run_id
                destination.parent.mkdir(exist_ok=True)
                os.replace(stage, destination)
            return response

    def reset(self, session_id: str, payload: dict) -> dict:
        with simulator.RUN_LOCK:
            directory = self._directory(session_id)
            revision = _revision(payload.get("from_revision"))
            run_id = _id(payload.get("run_id"), "run_id")
            identity = {
                "id": session_id,
                "from_revision": revision,
                "run_id": run_id,
                "action": "reset",
            }
            fingerprint = hashlib.sha256(_json_bytes(identity)).hexdigest()
            cached = self._retry(directory, run_id, fingerprint)
            if cached is not None:
                return cached
            head = self._head(directory)
            if revision != head:
                raise SessionConflict(f"reset requires HEAD {head}.")
            _, _, weights = self._load(directory, 0, torch.device("cpu"))
            self._budget(
                sum(value.numel() * value.element_size() for value in weights.values()) + 16384
            )
            self._write_revision(
                directory, head + 1, head, "reset", run_id, fingerprint, weights, identity, None
            )
            self._publish(directory, head + 1)
            return self._metadata(directory)

    def export(self, session_id: str) -> Path:
        with simulator.RUN_LOCK:
            directory = self._directory(session_id)
            head = self._head(directory)
            destination = self.root / f".export-{uuid.uuid4().hex}"
            files = [
                directory / "manifest.json",
                directory / "topology.safetensors",
                directory / "HEAD.json",
            ]
            for revision in range(head + 1):
                files.extend(
                    path for path in _revision_path(directory, revision).iterdir() if path.is_file()
                )
            files.extend(path for path in (directory / "runs").glob("*/*.json") if path.is_file())
            self._budget(sum(path.stat().st_size for path in files))
            for path in files:
                target = destination / path.relative_to(directory)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
            return destination

    def _import_records(self, source, manifest, head, files):
        expected = {"manifest.json", "topology.safetensors", "HEAD.json"}
        records = []
        run_ids = set()
        for revision in range(head + 1):
            self._load(source, revision, torch.device("cpu"))
            directory = _revision_path(source, revision)
            row = _read_json(directory / "revision.json")
            _validate_revision_record(row, revision, manifest, run_ids)
            expected.update(
                f"{directory.name}/{name}" for name in ("revision.json", "weights.safetensors")
            )
            if row["action"] == "learn":
                response = _read_json(directory / "response.json")
                _validate_response(response, row["request"])
                expected.add(f"{directory.name}/response.json")
            else:
                response = None
            records.append((directory.relative_to(source), "revision.json", row, response))
        for record_path in (source / "runs").glob("*/request.json"):
            row, response = _read_saved_run(record_path, manifest, run_ids)
            relative = record_path.parent.relative_to(source)
            expected.update(
                f"{relative.as_posix()}/{name}" for name in ("request.json", "response.json")
            )
            records.append((relative, "request.json", row, response))
        if {item.relative_to(source).as_posix() for item in files} != expected:
            raise ValueError("Checkpoint contains unexpected files.")
        return records

    def import_checkpoint(self, path) -> dict:
        source = Path(path).resolve()
        with simulator.RUN_LOCK:
            files = [item for item in source.rglob("*") if item.is_file()]
            if any(item.is_symlink() or item.is_junction() for item in source.rglob("*")):
                raise ValueError("Checkpoint must contain ordinary files and directories.")
            self._budget(sum(item.stat().st_size for item in files))
            manifest = _import_manifest(source)
            head = self._head(source)
            if head + 1 > len(files) // 2:
                raise ValueError("Checkpoint HEAD exceeds its available revision records.")
            records = self._import_records(source, manifest, head, files)
            session_id = uuid.uuid4().hex
            stage = self.root / f".import-{session_id}"
            stage.mkdir()
            _stage_import(source, stage, session_id, manifest, head, files, records)
            os.replace(stage, self.root / session_id)
            return self._metadata(self.root / session_id)

    def cleanup_export(self, path) -> None:
        with simulator.RUN_LOCK:
            candidate = Path(path)
            if candidate.is_symlink() or candidate.is_junction():
                raise ValueError("Export cleanup refuses links.")
            resolved = candidate.resolve()
            if resolved.parent != self.root or not re.fullmatch(
                r"\.export-[0-9a-f]{32}", resolved.name
            ):
                raise ValueError(
                    "Export cleanup accepts only this store's generated export directory."
                )
            if resolved.exists():
                shutil.rmtree(resolved)
