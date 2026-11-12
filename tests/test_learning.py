import math

import pytest
import torch

from brain_model.learning import ContactLearner, parse_learning


def learner(rule="pair_stdp"):
    contacts = {
        "long_glu": {
            "source": torch.tensor([0, 0]),
            "target": torch.tensor([1, 1]),
            "weight": torch.tensor([1.0, 1.0]),
            "plastic": torch.tensor([True, False]),
        }
    }
    topology = {
        "n": 2,
        "device": torch.device("cpu"),
        "contacts": contacts,
        "region_of_cell": torch.tensor([0, 0]),
    }
    return ContactLearner(topology, parse_learning({"learning": {"rule": rule}}), 1.0)


def test_pair_rule_causal_anticause_and_fixed_contact():
    causal = learner()
    causal.step(torch.tensor([1.0, 0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    causal.step(torch.tensor([0.0, 1.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    expected = 1 + 0.001 * math.exp(-1 / 20) * (2 - 1) / (2 - 0.1)
    assert causal.weights["long_glu"].tolist() == pytest.approx([expected, 1.0])
    anti = learner()
    anti.step(torch.tensor([0.0, 1.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    anti.step(torch.tensor([1.0, 0.0]), torch.tensor([0.0]), torch.tensor([0.0]))
    assert anti.weights["long_glu"][0] < 1
    assert anti.contacts["long_glu"]["weight"].tolist() == [1.0, 1.0]


def test_simultaneous_spikes_do_not_self_pair():
    obj = learner()
    obj.step(torch.ones(2), torch.tensor([0.0]), torch.tensor([0.0]))
    assert obj.weights["long_glu"].tolist() == [1.0, 1.0]


def test_eligibility_uses_seconds_and_regional_baseline():
    obj = learner("dopamine_eligibility")
    obj.step(torch.tensor([1.0, 0.0]), torch.tensor([0.5]), torch.tensor([0.5]))
    obj.step(torch.tensor([0.0, 1.0]), torch.tensor([0.8]), torch.tensor([0.5]))
    expected = 1 + 0.0001 * 0.001 * math.exp(-1 / 20) * 0.3
    assert obj.weights["long_glu"][0].item() == pytest.approx(expected, abs=6e-8)
    assert obj.eligibility["long_glu"][0].item() == pytest.approx(math.exp(-1 / 20))


@pytest.mark.parametrize(
    "payload",
    [
        {"stdp": {"enabled": True}, "td_learning": {"enabled": True}},
        {"stdp": {"tau_trace": 20}},
        {"td_learning": {"learning_rate_fast": 0.001}},
        {"learning": {"rule": "pair_stdp", "g": float("nan")}},
        {"learning": {"rule": "pair_stdp", "g": 1.1}},
    ],
)
def test_reject_ambiguous_or_invalid_learning(payload):
    with pytest.raises(ValueError):
        parse_learning(payload)
