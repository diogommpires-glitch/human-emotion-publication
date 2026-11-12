import pytest
from test_core import micro_config

from brain_model import model
from brain_model.sessions import CheckpointTooLarge, SessionConflict, SessionStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    cfg = micro_config()
    cfg["runtime"].update(cpu_network_sizes=[192], cpu_default_cells=192)
    monkeypatch.setattr(model, "NETWORK", cfg)
    monkeypatch.setenv("HUMAN_EMOTION_DEVICE", "cpu")
    return SessionStore(tmp_path / "sessions")


def request(session, run_id="first", action="learn", revision=0):
    return {
        "seed": 23,
        "inputs": {"threat_cue": 0.6},
        "learning": {"rule": "pair_stdp"},
        "session": {
            "id": session["id"],
            "from_revision": revision,
            "run_id": run_id,
            "action": action,
        },
    }


def test_commit_retry_restart_and_private_records(store):
    session = store.create({"neurons": 192, "seed": 11})
    payload = request(session)
    payload.update(
        event="PRIVATE NOTE",
        person={"name": "PRIVATE NAME", "trust": 0.5, "closeness": 0.2, "tone": 0.5},
    )
    response = store.run(payload)
    restarted = SessionStore(store.root)
    assert restarted.run(payload) == response
    assert response["session"]["revision"] == 1
    assert response["timeline"]["stimulus_seconds"] == [0.04, 0.24]
    assert restarted.get(session["id"])["head_revision"] == 1
    text = "\n".join(path.read_text() for path in store.root.rglob("*.json"))
    assert "PRIVATE" not in text
    with pytest.raises(SessionConflict):
        restarted.run({**payload, "seed": 24})


def test_evaluate_replay_reset_and_checkpoint_roundtrip(store, tmp_path):
    session = store.create({"neurons": 192, "seed": 11})
    for action in ("evaluate", "replay"):
        result = store.run(request(session, action, action))
        assert not result["session"]["committed"]
        assert store.get(session["id"])["head_revision"] == 0
    store.run(request(session))
    with pytest.raises(SessionConflict):
        store.run(request(session, "stale"))
    reset = store.reset(session["id"], {"from_revision": 1, "run_id": "reset"})
    assert reset["head_revision"] == 2
    imported = SessionStore(tmp_path / "imported").import_checkpoint(store.export(session["id"]))
    assert imported["head_revision"] == 2
    assert imported["neurons"] == 192


def test_publication_failure_preserves_head_then_recovers_retry(store, monkeypatch):
    from brain_model import sessions

    session = store.create({"neurons": 192, "seed": 11})
    replace = sessions.os.replace

    def fail_head(source, target):
        if str(target).endswith("HEAD.json"):
            raise OSError("injected HEAD publication failure")
        return replace(source, target)

    with monkeypatch.context() as context:
        context.setattr(sessions.os, "replace", fail_head)
        with pytest.raises(OSError):
            store.run(request(session))
    assert store.get(session["id"])["head_revision"] == 0
    result = SessionStore(store.root).run(request(session))
    assert result["session"]["revision"] == 1


def test_quota_and_corrupt_checkpoint_are_rejected(store, tmp_path):
    with pytest.raises(CheckpointTooLarge):
        SessionStore(tmp_path / "tiny", budget_bytes=1).create({"neurons": 192, "seed": 11})
    session = store.create({"neurons": 192, "seed": 11})
    export = store.export(session["id"])
    (export / "topology.safetensors").write_bytes(b"invalid")
    with pytest.raises(ValueError):
        SessionStore(tmp_path / "other").import_checkpoint(export)


def test_run_lookup_and_export_cleanup_preserve_committed_session(store):
    session = store.create({"neurons": 192, "seed": 11})
    response = store.run(request(session))
    assert store.get_run(session["id"], "first") == response
    export = store.export(session["id"])
    with pytest.raises(ValueError):
        store.cleanup_export(store.root / session["id"])
    store.cleanup_export(export)
    assert not export.exists()
    assert store.get(session["id"])["head_revision"] == 1


def test_checkpoint_rejects_wrong_shape_before_materializing_tensors(store, tmp_path):
    import torch
    from safetensors.torch import save_file

    from brain_model import sessions

    path = tmp_path / "wrong.safetensors"
    save_file({"weight": torch.ones(3)}, path)
    with pytest.raises(ValueError, match="shape"):
        sessions._load_tensors(path, sessions._digest(path), {"weight": ((2,), "F32")})


def test_evaluation_idempotency_survives_restart_without_revision(store):
    session = store.create({"neurons": 192, "seed": 11})
    payload = request(session, action="evaluate")
    response = store.run(payload)
    restarted = SessionStore(store.root)
    assert restarted.run(payload) == response
    assert restarted.get_run(session["id"], "first") == response
    assert restarted.get(session["id"])["head_revision"] == 0
    with pytest.raises(SessionConflict):
        restarted.run({**payload, "seed": 55})


@pytest.mark.parametrize("key", ["gaba_cell", "bin_of_cell", "contacts.local_glu.plastic"])
def test_checkpoint_cannot_change_cell_classes_or_authorize_plasticity(key):
    import torch

    from brain_model import simulator
    from brain_model.sessions import pack_topology, unpack_topology

    cfg = micro_config()
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    tensors = {key: value.clone() for key, value in pack_topology(topology).items()}
    tensors[key].fill_(key.endswith("plastic"))
    with pytest.raises(ValueError):
        unpack_topology(
            {"network": cfg, "neurons": 192, "rows": topology["rows"]}, tensors, torch.device("cpu")
        )


def test_import_rejects_private_request_and_remaps_retries(store, tmp_path):
    import json

    session = store.create({"neurons": 192, "seed": 11})
    payload = request(session)
    store.run(payload)
    export = store.export(session["id"])
    revision = export / "revision-00000001" / "revision.json"
    original = json.loads(revision.read_text())
    injected = dict(original)
    injected["request"] = {**original["request"], "event": "PRIVATE IMPORT NOTE"}
    revision.write_text(json.dumps(injected))
    with pytest.raises(ValueError):
        SessionStore(tmp_path / "bad").import_checkpoint(export)
    revision.write_text(json.dumps(original))
    destination = SessionStore(tmp_path / "good")
    imported = destination.import_checkpoint(export)
    replay_payload = {**payload, "session": {**payload["session"], "id": imported["id"]}}
    response = destination.run(replay_payload)
    assert response["session"]["id"] == imported["id"]
    assert destination.get_run(imported["id"], "first") == response


def test_metadata_reads_do_not_wait_for_numerical_execution(store):
    from concurrent.futures import ThreadPoolExecutor

    from brain_model import simulator

    session = store.create({"neurons": 192, "seed": 11})
    store.run(request(session))

    def read_metadata():
        return store.list(), store.get(session["id"]), store.get_run(session["id"], "first")

    with ThreadPoolExecutor(max_workers=1) as executor:
        simulator.RUN_LOCK.acquire()
        try:
            future = executor.submit(read_metadata)
            listed, metadata, response = future.result(timeout=1)
        finally:
            simulator.RUN_LOCK.release()
    assert listed[0]["id"] == metadata["id"] == response["session"]["id"]


@pytest.mark.parametrize("field", ["run_id", "response_revision", "start_revision"])
def test_import_rejects_inconsistent_revision_identity(store, tmp_path, field):
    import json

    session = store.create({"neurons": 192, "seed": 11})
    store.run(request(session))
    export = store.export(session["id"])
    filename = "revision.json" if field == "run_id" else "response.json"
    path = export / "revision-00000001" / filename
    value = json.loads(path.read_text())
    if field == "run_id":
        value["run_id"] = "another-run"
    else:
        value["session"]["revision" if field == "response_revision" else field] = 999
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        SessionStore(tmp_path / "imported").import_checkpoint(export)
