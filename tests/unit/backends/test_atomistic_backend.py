#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Admission, refusal and identity of the accelerated atomistic kernel backend."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

import phydrax.backends.atomistic as atomistic_backend
from phydrax._cuda_capabilities import CudaCompilerCapabilities
from phydrax.backends._types import BackendUnavailableError
from phydrax.backends.atomistic import (
    AtomisticKernelAdmission,
    AtomisticKernelRequest,
    AtomisticKernelTarget,
    pallas_atomistic_availability,
)


def _request(**overrides: Any) -> AtomisticKernelRequest:
    values: dict[str, Any] = {
        "precision": "float64",
        "accumulation": "deterministic",
        "channels": 96,
        "channel_tile": 128,
        "receiver_tile": 4,
        "edge_tile": 16,
        "reduction_programs": 64,
        "receiver_extent": 32,
        "source_extent": 256,
        "edge_extent": 256,
        "maximum_degree": 3,
        "path_count": 12,
        "coefficient_count": 300,
        "workspace_bytes": 64 * 1024,
        "fragment_bytes": 8 << 20,
        "fragment_budget_bytes": 16 << 20,
        "maximum_derivative_order": 2,
    }
    values.update(overrides)
    return AtomisticKernelRequest(**values)


def test_interpreter_target_is_reference_verification_not_gpu_qualification() -> None:
    target = AtomisticKernelTarget("cpu_interpret")
    admission = AtomisticKernelAdmission(_request(), target)
    assert target.lowering == "pallas_mosaic_gpu_interpret"
    assert target.compute_capability is None
    assert target.target_id == AtomisticKernelTarget("cpu_interpret").target_id
    assert "not GPU" in admission.qualification_scope
    # Channels pad to whole warpgroup tiles rather than being refused.
    assert admission.request.channel_capacity == 128


def test_cuda_target_refuses_on_hosts_without_an_admitted_device() -> None:
    availability = pallas_atomistic_availability("cuda")
    if availability.available:
        pytest.skip("An admitted CUDA device is present; refusal is not observable.")
    with pytest.raises(BackendUnavailableError, match="NVIDIA CUDA"):
        availability.require("atomistic.mace_edge_coupling.receiver")
    with pytest.raises(BackendUnavailableError):
        AtomisticKernelTarget("cuda")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"accumulation": "fast"}, "atomic scatter", id="fast-accumulation"),
        pytest.param({"channel_tile": 64}, "warpgroup lanes", id="sub-warpgroup-tile"),
        pytest.param({"channel_tile": 384}, "power-of-two", id="non-power-of-two-tile"),
        pytest.param({"channel_tile": 2048}, "maximum tile", id="oversized-tile"),
        pytest.param({"edge_tile": 4096}, "maximum tile", id="oversized-edge-tile"),
        pytest.param({"maximum_degree": 5}, "maximum degree", id="degree"),
        pytest.param({"path_count": 0}, "Path count", id="no-paths"),
        pytest.param({"coefficient_count": 5000}, "Coefficient count", id="coefficients"),
        pytest.param({"workspace_bytes": 256 * 1024}, "workspace", id="workspace"),
        pytest.param(
            {"maximum_derivative_order": 3}, "Derivative order", id="derivative-order"
        ),
        pytest.param(
            {"reduction_programs": 2048}, "reduction_programs", id="reduction-programs"
        ),
        pytest.param(
            {"fragment_bytes": 32 << 20}, "fragment budget", id="fragment-budget"
        ),
        pytest.param({"edge_extent": 0}, "edge_extent", id="empty-fragment"),
    ],
)
def test_requests_outside_the_admitted_envelope_are_refused(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AtomisticKernelAdmission(
            _request(**overrides), AtomisticKernelTarget("cpu_interpret")
        )


def test_precision_and_accumulation_are_distinct_admitted_identities() -> None:
    target = AtomisticKernelTarget("cpu_interpret")
    float64 = AtomisticKernelAdmission(_request(), target)
    float32 = AtomisticKernelAdmission(_request(precision="float32"), target)
    compensated = AtomisticKernelAdmission(_request(accumulation="compensated"), target)
    identities = {float64.admission_id, float32.admission_id, compensated.admission_id}
    assert len(identities) == 3
    assert (
        float64.admission_id == AtomisticKernelAdmission(_request(), target).admission_id
    )


@pytest.mark.parametrize(
    ("capability", "admitted"),
    [
        ((8, 0), True),
        ((8, 6), False),
        ((9, 0), True),
        ((12, 0), True),
        ((8, 9), False),
        ((10, 0), True),
        ((10, 3), True),
        ((12, 1), True),
        ((13, 0), False),
    ],
)
def test_cuda_architecture_admission_is_explicit(
    monkeypatch: pytest.MonkeyPatch, capability: tuple[int, int], admitted: bool
) -> None:
    device = SimpleNamespace(device_kind="test NVIDIA device", client=object())
    monkeypatch.setattr(atomistic_backend, "_cuda_device", lambda: (device, "test CUDA"))
    monkeypatch.setattr(atomistic_backend, "_compute_capability", lambda _: capability)
    monkeypatch.setattr(
        atomistic_backend,
        "cuda_compiler_capabilities",
        lambda _: _compiler(mosaic_gpu=True),
    )
    availability = pallas_atomistic_availability("cuda")
    assert availability.available is admitted


def _compiler(*, mosaic_gpu: bool, build_id: str = "build-a") -> CudaCompilerCapabilities:
    return CudaCompilerCapabilities(
        host_platform="win32",
        distribution="phydrax-cuda13-pjrt",
        version="test",
        build_id=build_id,
        jax_version="0.12.0",
        jaxlib_version="0.12.0",
        platform_version="cuda 13000",
        mosaic_gpu=mosaic_gpu,
        reason="The installed build does not declare Mosaic GPU.",
    )


def test_blackwell_architecture_does_not_imply_mosaic_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(
        device_kind="NVIDIA GeForce RTX 5090 Laptop GPU",
        compute_capability="12.0",
        client=SimpleNamespace(platform_version="cuda 13000"),
    )
    monkeypatch.setattr(atomistic_backend, "_cuda_device", lambda: (device, "CUDA"))
    monkeypatch.setattr(
        atomistic_backend,
        "cuda_compiler_capabilities",
        lambda _: _compiler(mosaic_gpu=False),
    )
    availability = pallas_atomistic_availability("cuda")
    assert not availability.available
    with pytest.raises(BackendUnavailableError, match="Mosaic GPU") as caught:
        AtomisticKernelTarget("cuda")
    assert caught.value.backend == "pallas-atomistic"
    assert caught.value.capability == "atomistic.mace_edge_coupling.receiver"


def test_atomistic_target_identity_distinguishes_compiler_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = SimpleNamespace(
        device_kind="test NVIDIA device",
        compute_capability="12.0",
        client=SimpleNamespace(platform_version="cuda 13000"),
    )
    compiler = _compiler(mosaic_gpu=True)
    monkeypatch.setattr(atomistic_backend, "_cuda_device", lambda: (device, "CUDA"))
    monkeypatch.setattr(
        atomistic_backend, "cuda_compiler_capabilities", lambda _: compiler
    )
    first = AtomisticKernelTarget("cuda")
    assert first.target_id == AtomisticKernelTarget("cuda").target_id
    compiler = replace(compiler, build_id="build-b")
    second = AtomisticKernelTarget("cuda")
    assert first.compiler_id != second.compiler_id
    assert first.target_id != second.target_id
