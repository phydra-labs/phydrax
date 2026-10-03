#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.execution import (
    DeviceMemorySample,
    DeviceResource,
    DistributionMode,
    ExecutionAdmissionError,
    ExecutionCandidate,
    ExecutionGroupSpec,
    ExecutionPlan,
    ExecutionPolicy,
    ExecutionRequirements,
    ExecutionResourceEvidence,
    HostMemorySample,
    MemoryEnvelope,
    MemoryMeasurement,
    PhaseMemorySampler,
    resolve_execution_plan,
    ResourceInventory,
    ResourceRequest,
    sample_device_memory,
    sample_host_memory,
)


_SUPPORTED_HOST = sys.platform == "darwin" or sys.platform.startswith("linux")
requires_host_probe = pytest.mark.skipif(
    not _SUPPORTED_HOST, reason="host memory probes exist on Linux and Darwin only"
)
_MIB = 1 << 20


@requires_host_probe
def test_host_sample_reports_resident_memory_within_kernel_and_host_bounds() -> None:
    sample = sample_host_memory()

    resident = sample.process_resident.value_bytes
    peak = sample.process_peak_resident.value_bytes
    total = sample.host_total.value_bytes
    available = sample.host_available.value_bytes
    assert resident is not None and resident > 0
    assert peak is not None and total is not None and available is not None
    # The kernel high-water mark bounds every instantaneous RSS reading.
    assert resident <= peak + sample.process_peak_resident.resolution_bytes
    assert resident < total
    assert 0 < available <= total
    assert sample.process_peak_resident.is_upper_bound
    assert not sample.process_resident.is_upper_bound
    assert sample.timestamp_ns > 0


def test_cpu_device_sample_marks_allocator_counters_unavailable() -> None:
    device = jax.devices("cpu")[0]
    held = jax.device_put(jnp.zeros((250_000,), dtype=jnp.float32), device)

    (sample,) = sample_device_memory((device,))

    for measurement in (
        sample.allocator_limit,
        sample.allocator_reserved,
        sample.allocator_in_use,
        sample.allocator_peak_in_use,
    ):
        assert not measurement.available
        assert measurement.unavailable_reason is not None
        assert "cpu" in measurement.unavailable_reason
    assert sample.allocator_headroom_bytes is None
    live = sample.live_array_bytes.value_bytes
    assert live is not None and live >= held.size * held.dtype.itemsize
    assert sample.key == (device.process_index, device.id)


@requires_host_probe
def test_phase_sampler_captures_a_deliberate_allocation() -> None:
    allocation_bytes = 128 * _MIB
    with PhaseMemorySampler("allocation", interval_seconds=0.002) as sampler:
        block = np.ones(allocation_bytes // 8, dtype=np.float64)
        time.sleep(0.2)
        del block

    evidence = sampler.evidence
    increase = evidence.sampled_peak_increase_bytes
    assert increase is not None and increase >= 0.9 * allocation_bytes
    assert evidence.sample_count > 2
    sampled = evidence.sampled_peak_resident
    assert sampled.method == "interval_sampling"
    assert sampled.scope == "peak"
    assert not sampled.is_upper_bound
    bound = evidence.resident_peak_upper_bound
    assert bound.is_upper_bound
    assert bound.value_bytes is not None and sampled.value_bytes is not None
    assert sampled.value_bytes <= bound.value_bytes + bound.resolution_bytes
    assert evidence.to_payload()["sampled_peak_resident"] == sampled.to_payload()
    with pytest.raises(RuntimeError, match="single-use"):
        sampler.__enter__()


def _measured(value: int | None) -> MemoryMeasurement:
    if value is None:
        return MemoryMeasurement.unavailable(
            "procfs_meminfo_available", "baseline", "probe not exposed"
        )
    return MemoryMeasurement.measured("procfs_meminfo_available", "baseline", value, 1)


def _host_sample(available: int | None) -> HostMemorySample:
    return HostMemorySample(
        "linux_procfs",
        0,
        1,
        MemoryMeasurement.measured("procfs_statm", "baseline", 64 * _MIB, 4096),
        MemoryMeasurement.measured("getrusage_max_rss", "peak", 96 * _MIB, 1024),
        MemoryMeasurement.measured(
            "sysconf_physical_pages", "baseline", 1024 * _MIB, 4096
        ),
        _measured(available),
    )


def _gpu_sample(limit: int | None, in_use: int) -> DeviceMemorySample:
    def allocator(value: int | None) -> MemoryMeasurement:
        if value is None:
            return MemoryMeasurement.unavailable(
                "jax_memory_stats", "baseline", "allocator statistics hidden"
            )
        return MemoryMeasurement.measured("jax_memory_stats", "baseline", value, 1)

    return DeviceMemorySample(
        0,
        0,
        "gpu",
        "cuda",
        "test-provider",
        1,
        allocator(limit),
        allocator(in_use),
        allocator(in_use),
        MemoryMeasurement.measured("jax_memory_stats", "peak", in_use, 1),
        MemoryMeasurement.measured("jax_live_arrays", "baseline", in_use, 1),
    )


def _resolve(
    inventory: ResourceInventory,
    evidence: ExecutionResourceEvidence | None,
    *,
    envelope: MemoryEnvelope = "certified",
) -> ExecutionPlan:
    device = inventory.devices[0]
    requirements = ExecutionRequirements("certified-memory")
    candidate = ExecutionCandidate(
        "candidate",
        requirements.requirements_id,
        ExecutionGroupSpec("root", (device.process_index,), (device.key,)),
        resource_evidence=evidence,
    )
    return resolve_execution_plan(
        ExecutionPolicy(
            DistributionMode.SINGLE,
            resources=ResourceRequest(1, 512 * _MIB, memory_envelope=envelope),
        ),
        inventory,
        (candidate,),
        precision_policy_id="float64",
        solver_policy_id="bounded",
    )


_CPU = DeviceResource(0, 0, 0, "cpu", "cpu")
_HOST_EVIDENCE = ExecutionResourceEvidence(
    per_host_peak_bytes=200 * _MIB, per_host_reserve_bytes=56 * _MIB
)


def test_certified_envelope_admits_declared_footprint_within_measured_memory() -> None:
    inventory = ResourceInventory(1, 0, (_CPU,), host_memory=(_host_sample(300 * _MIB),))

    plan = _resolve(inventory, _HOST_EVIDENCE)

    assert plan.resource_evidence == _HOST_EVIDENCE


@pytest.mark.parametrize(
    ("host_memory", "evidence", "reason"),
    [
        pytest.param(
            (),
            _HOST_EVIDENCE,
            "lacks measured available memory",
            id="no-host-sample",
        ),
        pytest.param(
            (_host_sample(None),),
            _HOST_EVIDENCE,
            "lacks measured available memory",
            id="available-memory-unexposed",
        ),
        pytest.param(
            (_host_sample(255 * _MIB),),
            _HOST_EVIDENCE,
            "exceeds measured available memory",
            id="footprint-exceeds-available",
        ),
        pytest.param(
            (_host_sample(300 * _MIB),),
            ExecutionResourceEvidence(per_host_peak_bytes=200 * _MIB),
            "lacks per-host memory evidence",
            id="unknown-reserve",
        ),
        pytest.param(
            (_host_sample(300 * _MIB),),
            None,
            "lacks candidate resource evidence",
            id="no-candidate-evidence",
        ),
        pytest.param(
            (_host_sample(300 * _MIB),),
            ExecutionResourceEvidence(
                per_host_peak_bytes=200 * _MIB,
                per_host_reserve_bytes=56 * _MIB,
                memory_basis="sampled",
            ),
            "sampled peak",
            id="sampled-peak-is-not-a-bound",
        ),
    ],
)
def test_certified_envelope_refuses_unknown_or_insufficient_memory_evidence(
    host_memory: tuple[HostMemorySample, ...],
    evidence: ExecutionResourceEvidence | None,
    reason: str,
) -> None:
    inventory = ResourceInventory(1, 0, (_CPU,), host_memory=host_memory)

    with pytest.raises(ExecutionAdmissionError, match=reason):
        _resolve(inventory, evidence)
    # The declared envelope keeps the owner-estimate contract unchanged.
    assert _resolve(inventory, evidence, envelope="declared") is not None


@pytest.mark.parametrize(
    ("limit", "reason"),
    [
        pytest.param(None, "lacks measured allocator headroom", id="hidden-allocator"),
        pytest.param(1024 * _MIB, "exceeds allocator headroom", id="insufficient"),
    ],
)
def test_certified_envelope_requires_accelerator_allocator_headroom(
    limit: int | None, reason: str
) -> None:
    gpu = DeviceResource(0, 0, 0, "gpu", "accelerator")
    inventory = ResourceInventory(
        1,
        0,
        (gpu,),
        host_memory=(_host_sample(300 * _MIB),),
        device_memory=(_gpu_sample(limit, 900 * _MIB),),
    )
    evidence = ExecutionResourceEvidence(
        per_device_peak_bytes=100 * _MIB,
        per_device_reserve_bytes=28 * _MIB,
        per_host_peak_bytes=200 * _MIB,
        per_host_reserve_bytes=56 * _MIB,
        memory_basis="compiler_analysis",
    )

    with pytest.raises(ExecutionAdmissionError, match=reason):
        _resolve(inventory, evidence)
    roomy = ResourceInventory(
        1,
        0,
        (gpu,),
        host_memory=(_host_sample(300 * _MIB),),
        device_memory=(_gpu_sample(2048 * _MIB, 900 * _MIB),),
    )
    assert _resolve(roomy, evidence) is not None


def test_certified_request_and_memory_basis_are_part_of_identity() -> None:
    declared = ResourceRequest(1, 4_096)
    certified = ResourceRequest(1, 4_096, memory_envelope="certified")
    sampled = ExecutionResourceEvidence(per_host_peak_bytes=1, memory_basis="sampled")

    assert ResourceRequest.from_payload(certified.to_payload()) == certified
    assert certified.resource_id != declared.resource_id
    assert ExecutionResourceEvidence.from_payload(sampled.to_payload()) == sampled
    assert (
        sampled.evidence_id
        != ExecutionResourceEvidence(per_host_peak_bytes=1).evidence_id
    )
    with pytest.raises(ValueError):
        ResourceRequest(1, 4_096, memory_envelope="optimistic")
