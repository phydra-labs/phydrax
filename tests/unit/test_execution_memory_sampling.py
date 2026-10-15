#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import solver
from phydrax.execution import (
    DeviceMemorySample,
    DeviceResource,
    discover_resource_inventory,
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
    PhaseMemoryEvidence,
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
    # A single procfs status snapshot computes VmHWM from the RSS it prints.
    # This ordering does not certify an upper bound on past RSS snapshots.
    assert resident <= peak + sample.process_peak_resident.resolution_bytes
    assert resident < total
    assert 0 < available <= total
    if sample.provider == "linux_procfs":
        assert sample.process_resident.method == "procfs_status"
        assert sample.process_peak_resident.method == "procfs_status"
        assert not sample.process_peak_resident.is_upper_bound
    else:
        assert sample.process_peak_resident.is_upper_bound
    assert not sample.process_resident.is_upper_bound
    assert sample.timestamp_ns > 0


def test_cpu_device_sample_marks_allocator_counters_unavailable() -> None:
    device = jax.devices("cpu")[0]
    held = jax.device_put(np.zeros((250_000,), dtype=np.float32), device)
    held.block_until_ready()

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
    assert held.device == device
    assert sample.live_array_bytes.method == "jax_live_arrays"
    assert live is not None and live >= held.nbytes
    assert sample.key == (device.process_index, device.id)


def test_live_array_sampling_isolates_cpu_from_the_default_accelerator() -> None:
    cpu = jax.devices("cpu")[0]
    accelerator = jax.devices()[0]
    if accelerator.platform == "cpu":
        pytest.skip("requires distinct CPU and accelerator PJRT clients")
    (before,) = sample_device_memory((cpu,))
    cpu_held = jax.device_put(np.zeros((250_000,), dtype=np.float32), cpu)
    accelerator_held = jax.device_put(
        np.zeros((2_000_000,), dtype=np.float32), accelerator
    )
    cpu_held.block_until_ready()
    accelerator_held.block_until_ready()

    (cpu_sample,) = sample_device_memory((cpu,))
    (accelerator_sample,) = sample_device_memory((accelerator,))

    assert before.live_array_bytes.value_bytes is not None
    assert cpu_sample.live_array_bytes.value_bytes == (
        before.live_array_bytes.value_bytes + cpu_held.nbytes
    )
    assert accelerator_sample.live_array_bytes.value_bytes is not None
    assert accelerator_sample.live_array_bytes.value_bytes >= accelerator_held.nbytes


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
    if evidence.final.provider == "linux_procfs":
        assert not bound.available
        assert not bound.is_upper_bound
        assert bound.method == "procfs_status"
        assert bound.unavailable_reason is not None
        assert "approximate" in bound.unavailable_reason
        assert not evidence.interval_peak_resident.available
        assert evidence.to_payload()["interval_peak_resident"] == (
            evidence.interval_peak_resident.to_payload()
        )
    else:
        assert bound.is_upper_bound
        assert bound.value_bytes is not None and sampled.value_bytes is not None
        assert sampled.value_bytes <= bound.value_bytes + bound.resolution_bytes
    assert evidence.to_payload()["sampled_peak_resident"] == sampled.to_payload()
    with pytest.raises(RuntimeError, match="single-use"):
        sampler.__enter__()


@requires_host_probe
def test_inventory_retains_memory_measurement_provenance() -> None:
    device = jax.devices()[0]
    held = jax.device_put(np.zeros((250_000,), dtype=np.float32), device)
    held.block_until_ready()

    inventory = discover_resource_inventory()

    (host,) = inventory.host_memory
    assert host.process_index == jax.process_index()
    if host.provider == "linux_procfs":
        assert host.process_peak_resident.method == "procfs_status"
        assert not host.process_peak_resident.is_upper_bound
    sample = next(
        sample
        for sample in inventory.device_memory
        if sample.key == (device.process_index, device.id)
    )
    assert sample.live_array_bytes.value_bytes is not None
    assert sample.live_array_bytes.value_bytes >= held.nbytes


@requires_host_probe
def test_production_sampling_preserves_peak_availability_in_terminal_record(
    tmp_path: Path,
) -> None:
    method = solver.SSPRK33FixedStepMethod(lambda time, state, args: jnp.ones_like(state))
    manifest = solver.ProductionCaseManifest(
        problem_id="sampled-constant-growth",
        method_id=method.method_id,
        precision_id="float64",
        topology_id="one-state",
        geometry_layout_id="static",
        dtype="float64",
    )
    store = solver.DurableCheckpointStore(
        tmp_path / "checkpoints", manifest, solver.CheckpointGenerationPolicy(2)
    )
    plan = solver.ProductionRunPlan(
        method,
        solver.RobustRetryPolicy(maximum_retries=1),
        step_size=0.1,
        end_time=0.1,
        maximum_steps=1,
        checkpoint_interval=1,
    )
    prepared = solver.PreparedProductionRun(manifest, plan, store)

    result = prepared.run(
        prepared.initial_state(jnp.asarray((0.0,), dtype=jnp.float64)),
        memory_sampling_interval=0.002,
    )

    assert result.successful
    assert result.memory is not None
    terminal = json.loads((store.root / "terminal.json").read_text())
    assert terminal["resource_evidence"] == result.memory.to_payload()
    if result.memory.final.provider == "linux_procfs":
        assert not result.memory.resident_peak_upper_bound.available
        peak = terminal["resource_evidence"]["interval_peak_resident"]
        assert peak["method"] == "procfs_status"
        assert peak["value_bytes"] is None
        assert peak["resolution_bytes"] is None
        assert "approximate" in peak["unavailable_reason"]


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
        MemoryMeasurement.measured("procfs_status", "baseline", 64 * _MIB, 1024),
        MemoryMeasurement.measured("procfs_status", "peak", 96 * _MIB, 1024),
        MemoryMeasurement.measured(
            "sysconf_physical_pages", "baseline", 1024 * _MIB, 4096
        ),
        _measured(available),
    )


@pytest.mark.parametrize(
    "final_peak",
    [
        pytest.param(96 * _MIB, id="high-water-unchanged"),
        pytest.param(128 * _MIB, id="high-water-rose"),
    ],
)
def test_linux_phase_reports_approximate_peak_without_certifying_a_bound(
    final_peak: int,
) -> None:
    baseline = _host_sample(300 * _MIB)
    final = replace(
        baseline,
        process_peak_resident=MemoryMeasurement.measured(
            "procfs_status", "peak", final_peak, 1024
        ),
    )
    # Approximate high-water accounting can miss a larger RSS observation
    # once that observation's allocation has been freed.
    sampled = MemoryMeasurement.measured("interval_sampling", "peak", 160 * _MIB, 1024)
    evidence = PhaseMemoryEvidence(
        "freed-allocation", 0.002, 0.1, 3, 0.05, baseline, final, sampled
    )

    assert evidence.sampled_peak_increase_bytes == 96 * _MIB
    assert final.process_peak_resident.available
    assert not final.process_peak_resident.is_upper_bound
    for peak in (
        evidence.resident_peak_upper_bound,
        evidence.interval_peak_resident,
    ):
        assert not peak.available
        assert peak.method == "procfs_status"
        assert peak.unavailable_reason is not None
        assert "approximate" in peak.unavailable_reason
    payload = evidence.to_payload()
    assert payload["sampled_peak_resident"] == sampled.to_payload()
    assert payload["interval_peak_resident"] == (
        evidence.interval_peak_resident.to_payload()
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
