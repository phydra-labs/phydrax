#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import pytest

from phydrax._execution_plan import (
    ExecutionAdmissionError,
    ExecutionCandidate,
    ExecutionRequirements,
    resolve_execution_plan,
)
from phydrax._execution_resources import (
    DeviceResource,
    DistributionMode,
    ExecutionGroupSpec,
    ExecutionPolicy,
    ExecutionResourceEvidence,
    ResourceInventory,
    ResourceRequest,
)


def _candidate(
    inventory: ResourceInventory,
    *,
    evidence: ExecutionResourceEvidence | None = None,
    name: str = "candidate",
) -> ExecutionCandidate:
    device = inventory.devices[0]
    requirements = ExecutionRequirements("resource-admission")
    return ExecutionCandidate(
        name,
        requirements.requirements_id,
        ExecutionGroupSpec("root", (device.process_index,), (device.key,)),
        resource_evidence=evidence,
    )


def _resolve(
    inventory: ResourceInventory,
    candidate: ExecutionCandidate,
    request: ResourceRequest,
) -> Any:
    return resolve_execution_plan(
        ExecutionPolicy(DistributionMode.SINGLE, resources=request),
        inventory,
        (candidate,),
        precision_policy_id="float64",
        solver_policy_id="bounded",
    )


def test_execution_resource_admission_scenario_1() -> None:
    inventory = ResourceInventory(
        1,
        0,
        (
            DeviceResource(
                0,
                0,
                0,
                "gpu",
                "accelerator",
                memory_bytes=16_384,
                vendor="nvidia",
            ),
        ),
    )
    request = ResourceRequest(
        1,
        32_768,
        accelerator_count=1,
        accelerator_platform="gpu",
        accelerator_vendor="nvidia",
        minimum_accelerator_memory_bytes=8_192,
    )

    plan = _resolve(inventory, _candidate(inventory), request)

    assert plan.group is not None
    assert plan.group.device_count == 1
    inventory = ResourceInventory(1, 0, (DeviceResource(0, 0, 0, "cpu", "cpu"),))
    request = ResourceRequest(1, 4_096, maximum_device_bytes=2_048)

    with pytest.raises(
        ExecutionAdmissionError, match="lacks required per-device memory evidence"
    ):
        _resolve(inventory, _candidate(inventory), request)
    inventory = ResourceInventory(1, 0, (DeviceResource(0, 0, 0, "cpu", "cpu"),))
    evidence = ExecutionResourceEvidence(
        per_device_peak_bytes=1_000,
        per_device_reserve_bytes=100,
        per_host_peak_bytes=1_200,
        per_host_reserve_bytes=200,
        compilation_cache_bytes=300,
        halo_collective_bytes=400,
        checkpoint_staging_bytes=500,
        output_backlog_bytes=600,
        dtypes=("float64",),
        backends=("jax",),
        collectives=("all-reduce",),
    )
    request = ResourceRequest(
        1,
        8_192,
        maximum_device_bytes=1_100,
        maximum_host_bytes=1_400,
        maximum_compilation_cache_bytes=300,
        maximum_halo_collective_bytes=400,
        maximum_checkpoint_staging_bytes=500,
        maximum_output_backlog_bytes=600,
        required_dtypes=("float64",),
        required_backends=("jax",),
        required_collectives=("all-reduce",),
    )

    plan = _resolve(inventory, _candidate(inventory, evidence=evidence), request)

    assert plan.resource_evidence == evidence


def test_execution_resource_admission_scenario_2() -> None:
    inventory = ResourceInventory(1, 0, (DeviceResource(0, 0, 0, "cpu", "cpu"),))
    first_evidence = ExecutionResourceEvidence(output_backlog_bytes=64)
    second_evidence = ExecutionResourceEvidence(output_backlog_bytes=65)
    first = _candidate(inventory, evidence=first_evidence)
    second = _candidate(inventory, evidence=second_evidence)
    request = ResourceRequest(1, 4_096, maximum_output_backlog_bytes=128)

    first_plan = _resolve(inventory, first, request)
    second_plan = _resolve(inventory, second, request)
    restored = type(first_plan).from_payload(first_plan.to_payload())
    restored_request = ResourceRequest.from_payload(request.to_payload())
    different_request = ResourceRequest(1, 4_096, maximum_output_backlog_bytes=127)

    assert first.candidate_id != second.candidate_id
    assert first_plan.execution_plan_id == first.candidate_id
    assert first_plan.plan_fingerprint != second_plan.plan_fingerprint
    assert restored.plan_fingerprint == first_plan.plan_fingerprint
    assert restored.resource_evidence == first_evidence
    assert restored_request.resource_id == request.resource_id
    assert different_request.resource_id != request.resource_id
    inventory = ResourceInventory(1, 0, (DeviceResource(0, 0, 0, "cpu", "cpu"),))

    plan = _resolve(
        inventory,
        _candidate(inventory),
        ResourceRequest(cpu_cores=1, memory_bytes=1_024),
    )

    assert plan.resource_evidence is None
    hosts = tuple((process, "node-a") for process in range(4))
    inventory = ResourceInventory(
        4,
        0,
        tuple(DeviceResource(process, process, 0, "cpu", "cpu") for process in range(4)),
        process_host_ids=hosts,
    )

    with pytest.raises(ExecutionAdmissionError, match="fewer distinct hosts"):
        _resolve_distributed_hosts(
            inventory,
            _distributed_host_candidate(inventory, hosts),
        )


def _distributed_host_candidate(
    inventory: ResourceInventory,
    process_host_ids: Any = (),
) -> ExecutionCandidate:
    requirements = ExecutionRequirements("distributed-host-admission")
    return ExecutionCandidate(
        "distributed-hosts",
        requirements.requirements_id,
        ExecutionGroupSpec(
            "distributed-root",
            tuple(range(inventory.process_count)),
            tuple(device.key for device in inventory.devices),
            process_host_ids=process_host_ids,
        ),
    )


def _resolve_distributed_hosts(
    inventory: ResourceInventory,
    candidate: ExecutionCandidate,
) -> Any:
    return resolve_execution_plan(
        ExecutionPolicy(
            DistributionMode.DISTRIBUTED,
            resources=ResourceRequest(
                4,
                8_192,
                host_count=2,
                process_count=4,
            ),
        ),
        inventory,
        (candidate,),
        precision_policy_id="float64",
        solver_policy_id="bounded",
    )


def test_execution_resource_admission_scenario_3() -> None:
    hosts = (
        (0, "node-a"),
        (1, "node-a"),
        (2, "node-b"),
        (3, "node-b"),
    )
    inventory = ResourceInventory(
        4,
        0,
        tuple(DeviceResource(process, process, 0, "cpu", "cpu") for process in range(4)),
        process_host_ids=hosts,
    )

    plan = _resolve_distributed_hosts(
        inventory,
        _distributed_host_candidate(inventory, hosts),
    )
    restored = type(plan).from_payload(plan.to_payload())

    assert plan.group is not None
    assert plan.group.process_host_ids == hosts
    assert restored.group == plan.group
    inventory = ResourceInventory(
        4,
        0,
        tuple(DeviceResource(process, process, 0, "cpu", "cpu") for process in range(4)),
    )

    with pytest.raises(ExecutionAdmissionError, match="host mapping evidence"):
        _resolve_distributed_hosts(
            inventory,
            _distributed_host_candidate(inventory),
        )
