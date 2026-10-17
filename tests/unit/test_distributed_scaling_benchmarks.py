#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded inputs and truthful hardware contracts of distributed drivers."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmarks import distributed_mace_scaling as mace, distributed_spectral as spectral


def test_spectral_input_block_matches_global_formula() -> None:
    shape = (8, 12, 4)
    index = (slice(4, 8), slice(0, 12), slice(0, 4))
    grid = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    phase = 3.0 / grid.size
    expected = np.asarray(
        np.sin((grid + 1.0) * phase) + 1j * np.cos((grid + 0.5) * (phase + 0.125)),
        dtype=np.complex64,
    )
    observed = spectral._input_block(shape, index, 2, np.dtype(np.complex64))
    np.testing.assert_array_equal(observed, expected[index])
    assert observed.shape == (4, 12, 4)


def test_two_devices_do_not_qualify_physical_pencil(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spectral.pxconfig, "snapshot", lambda: SimpleNamespace(cpu_device_count=None)
    )
    with pytest.raises(ValueError, match="nondegenerate"):
        spectral._campaign_contract("physical-pencil", "pencil", (2, 1))


def test_cpu_inventory_does_not_qualify_physical_slab(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spectral.pxconfig, "snapshot", lambda: SimpleNamespace(cpu_device_count=None)
    )
    monkeypatch.setattr(
        spectral.jax, "devices", lambda: (SimpleNamespace(platform="cpu"),) * 2
    )
    with pytest.raises(ValueError, match="accelerator"):
        spectral._campaign_contract("physical-slab", "slab", (2,))


def test_spectral_mesh_must_include_every_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spectral.pxconfig, "snapshot", lambda: SimpleNamespace(cpu_device_count=None)
    )
    monkeypatch.setattr(spectral.jax, "process_count", lambda: 2)
    monkeypatch.setattr(
        spectral.jax,
        "devices",
        lambda: (SimpleNamespace(platform="gpu", process_index=0),) * 2,
    )
    with pytest.raises(ValueError, match="every JAX process"):
        spectral._campaign_contract("physical-slab", "slab", (2,))


def test_mace_refuses_unimplemented_multi_process_ingress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mace.jax, "process_count", lambda: 2)
    with pytest.raises(ValueError, match="single-process"):
        mace._row("fixed-total", 2, 16, 1, "devices", 1)


def test_mace_rejects_empty_campaign() -> None:
    with pytest.raises(ValueError, match="positive"):
        mace._row("fixed-total", 1, 0, 1, "devices", 1)
