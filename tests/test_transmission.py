"""Transmission conserves repeated contacts independently of CSR storage layout."""

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from brain_model.topology import contact_matrix


@settings(max_examples=30, deadline=None)
@given(
    st.lists(
        st.tuples(st.integers(0, 7), st.integers(0, 7), st.integers(1, 20)), min_size=1, max_size=80
    )
)
def test_csr_matches_explicit_contact_sum(edges):
    source, target, weight = zip(*edges, strict=True)
    contacts = {
        "source": torch.tensor(source),
        "target": torch.tensor(target),
        "weight": torch.tensor(weight, dtype=torch.float32) / 10,
    }
    matrix, slots = contact_matrix(contacts, 8)
    activity = torch.arange(8, dtype=torch.float32) / 8
    expected = torch.zeros(8)
    expected.index_add_(0, contacts["target"], contacts["weight"] * activity[contacts["source"]])
    torch.testing.assert_close(torch.mv(matrix, activity), expected, rtol=1e-5, atol=1e-6)
    assert len(slots) == len(edges)
    assert matrix.values().numel() == len(set(zip(source, target, strict=True)))


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_identical_contact_kernel_matches_cpu_and_cuda():
    generator = torch.Generator().manual_seed(23)
    contacts = {
        "source": torch.randint(32, (500,), generator=generator),
        "target": torch.randint(32, (500,), generator=generator),
        "weight": torch.rand(500, generator=generator),
    }
    activity = torch.rand(32, generator=generator)
    cpu, slots = contact_matrix(contacts, 32)
    cuda, cuda_slots = contact_matrix({name: value.cuda() for name, value in contacts.items()}, 32)
    torch.testing.assert_close(cuda_slots.cpu(), slots, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.mv(cuda, activity.cuda()).cpu(), torch.mv(cpu, activity), rtol=1e-5, atol=1e-6
    )
