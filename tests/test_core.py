import copy
import json

import pytest
import torch

from brain_model import model, simulator


def micro_config():
    cfg = copy.deepcopy(model.NETWORK)
    cfg["runtime"].update(
        min_cells_per_region=8,
        duration_seconds=0.24,
        baseline_window_seconds=[0, 0.04],
        stimulus_seconds=[0.04, 0.24],
        reference_duration_seconds=0.04,
        raster_cells_per_region=8,
    )
    return cfg


def test_layout_preserves_cell_total():
    rows = simulator.layout(micro_config(), 192)
    assert sum(row["size"] for row in rows) == 192
    assert all(
        row["projection"] + row["sst_count"] + row["vip_count"] == row["size"] for row in rows
    )


def test_micro_episode_has_two_release_channels_and_finite_empty_classes():
    cfg = micro_config()
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    raw = simulator.run(cfg, topology, model.DEFAULT_INPUTS, 11)
    assert len(raw["baseline"]["release_per_cell_hz"]) == 2
    assert raw["raster"]["layout"]
    json.dumps(raw, allow_nan=False)


def test_long_range_targets_stay_inside_correct_pool():
    cfg = micro_config()
    rows = simulator.layout(cfg, 192)
    cfg["projections"] = [{"from": "sensory_thalamus", "to": "sensory_cortex", "weight": 1}]
    target = next(row for row in rows if row["id"] == "sensory_cortex")
    parts = [p for p in simulator._synapse_parts(cfg, rows) if p[0] == "long_glu"]
    assert parts[0][1:3] == (target["start"], target["projection"])
    assert parts[1][1:3] == (target["start"] + target["projection"], target["sst_count"])


def test_occupancy_rest_uses_half_saturation_constant():
    cfg = micro_config()
    cfg["neuromodulation"]["receptor_half_occupancy"] = 2
    cfg["runtime"]["time_step_ms"] = 0.01
    cfg["synapses"]["local_delay_ms"] = 0.01
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    raw, state = simulator.run_episode(
        cfg, topology, model.DEFAULT_INPUTS, 11, duration_s=0.00001, record=False
    )
    tables = simulator._modulator_tables(cfg, topology, model.DEFAULT_INPUTS)
    torch.testing.assert_close(
        torch.tensor(raw["resting_occupancy"]), tables["density"] / (tables["density"] + 2)
    )
    torch.testing.assert_close(
        state.tensors["chem"] / (state.tensors["chem"] + 2),
        tables["density"] / (tables["density"] + 2),
    )


def test_repeated_contacts_remain_distinct_but_transmit_as_canonical_csr():
    from brain_model.topology import contact_matrix

    contacts = {
        "source": torch.tensor([0, 0, 2]),
        "target": torch.tensor([1, 1, 1]),
        "weight": torch.tensor([0.2, 0.3, 0.4]),
        "plastic": torch.tensor([True, False, True]),
    }
    matrix, slots = contact_matrix(contacts, 3)
    assert matrix.col_indices().tolist() == [0, 2]
    assert slots.tolist() == [0, 0, 1]
    torch.testing.assert_close(
        matrix.to_dense() @ torch.tensor([2.0, 0.0, 3.0]), torch.tensor([0.0, 2.2, 0.0])
    )


def test_continuous_chunks_equal_uninterrupted_episode():
    cfg = micro_config()
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    whole, full = simulator.run_episode(cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.24)
    _, half = simulator.run_episode(cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.12)
    chunk, end = simulator.run_episode(
        cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.12, state=half
    )
    assert whole["spike_count"] == chunk["spike_count"]
    for key in full.tensors:
        torch.testing.assert_close(full.tensors[key], end.tensors[key], rtol=0, atol=0)
    assert whole["baseline"] == chunk["baseline"]


@pytest.mark.parametrize("rule", ["none", "pair_stdp", "dopamine_eligibility"])
def test_serialized_chunks_preserve_recording_and_learning(rule):
    from brain_model.learning import parse_learning

    cfg = micro_config()
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    learning = parse_learning({"learning": {"rule": rule}})
    whole, full = simulator.run_episode(
        cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.24, learning_config=learning
    )
    first, start = simulator.run_episode(
        cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.117, learning_config=learning
    )
    metadata, tensors = start.to_checkpoint()
    restored = simulator.RunState.from_checkpoint(metadata, tensors, topology)
    saved = {key: value.clone() for key, value in restored.tensors.items()}
    last, end = simulator.run_episode(
        cfg,
        topology,
        model.DEFAULT_INPUTS,
        23,
        duration_s=0.123,
        state=restored,
        learning_config=learning,
    )
    for key, value in saved.items():
        torch.testing.assert_close(value, restored.tensors[key], rtol=0, atol=0)
    for key in full.tensors:
        torch.testing.assert_close(full.tensors[key], end.tensors[key], rtol=0, atol=0)
    for key in ("t", "projection_hz", "occupancy", "neural_drive"):
        assert first["timecourse"][key] + last["timecourse"][key] == whole["timecourse"][key]
    for channel, values in whole["field_potential"]["channels"].items():
        assert (
            first["field_potential"]["channels"][channel]
            + last["field_potential"]["channels"][channel]
            == values
        )


def test_reference_cache_keys_configuration(monkeypatch):
    cfg = micro_config()
    monkeypatch.setattr(model, "NETWORK", cfg)
    first = simulator.build_topology(cfg, 192, torch.device("cpu"))
    model._REFERENCE.clear()
    initial = model._reference(first)
    cfg = copy.deepcopy(cfg)
    cfg["background_input"]["rate_hz"] *= 2
    monkeypatch.setattr(model, "NETWORK", cfg)
    second = simulator.build_topology(cfg, 192, torch.device("cpu"))
    changed = model._reference(second)
    assert initial != changed


def test_metadata_remains_available_without_torch():
    import subprocess
    import sys

    script = """import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch."):
        raise ImportError("test missing torch")
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from brain_model import model
assert model.model_info()["available"] is False
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_continuous_checkpoint_rejects_wrong_shape_and_configuration():
    cfg = micro_config()
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    _, state = simulator.run_episode(cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.02)
    metadata, tensors = state.to_checkpoint()
    with pytest.raises(ValueError, match="shape"):
        simulator.RunState.from_checkpoint(metadata, {**tensors, "v": torch.zeros(1)}, topology)
    changed = copy.deepcopy(cfg)
    changed["background_input"]["rate_hz"] *= 2
    other = simulator.build_topology(changed, 192, torch.device("cpu"))
    with pytest.raises(ValueError, match="configuration"):
        simulator.RunState.from_checkpoint(metadata, tensors, other)


@pytest.mark.parametrize("bad", [float("nan"), -1, 2])
def test_simulator_rejects_nonphysical_input_levels(bad):
    cfg = micro_config()
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    with pytest.raises(ValueError):
        simulator.run_episode(
            cfg, topology, {**model.DEFAULT_INPUTS, "threat_cue": bad}, 23, duration_s=0.02
        )


def test_explicit_stress_drives_modulate_tonic_state_and_resume_exactly():
    cfg = micro_config()
    topology = simulator.build_topology(cfg, 192, torch.device("cpu"))
    _, low = simulator.run_episode(
        cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.024, stimulus_drives=[{"stress": 0}]
    )
    _, high = simulator.run_episode(
        cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.024, stimulus_drives=[{"stress": 1}]
    )
    assert not torch.equal(low.tensors["v"], high.tensors["v"])
    _, first = simulator.run_episode(
        cfg, topology, model.DEFAULT_INPUTS, 23, duration_s=0.011, stimulus_drives=[{"stress": 1}]
    )
    _, resumed = simulator.run_episode(
        cfg,
        topology,
        model.DEFAULT_INPUTS,
        23,
        duration_s=0.013,
        state=first,
        stimulus_drives=[{"stress": 1}],
    )
    for key, value in high.tensors.items():
        torch.testing.assert_close(value, resumed.tensors[key], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("group", "name", "value"),
    [
        ("cell_types", "release_u", 2),
        ("cell_types", "refractory_ms", -1),
        ("synapses", "ampa_decay_ms", 0),
        ("synapses", "long_range_delay_ms", 0),
    ],
)
def test_configuration_rejects_nonphysical_release_and_times(group, name, value):
    cfg = micro_config()
    target = cfg["cell_types"]["regular_spiking"] if group == "cell_types" else cfg[group]
    target[name] = value
    with pytest.raises(ValueError):
        simulator.build_topology(cfg, 192, torch.device("cpu"))
