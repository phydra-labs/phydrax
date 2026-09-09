#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark distributed solid-Earth operators, restart, and fail-closed policies."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment

import phydrax as phx
from phydrax.applications import geophysics as geo


def _distributed_case(part_count: int, entities_per_part: int) -> dict[str, object]:
    entity_count = part_count * entities_per_part
    owner = np.repeat(np.arange(part_count, dtype=np.int32), entities_per_part)
    adjacency = np.stack(
        (np.arange(entity_count - 1), np.arange(1, entity_count)), axis=1
    )
    halo = phx.discretization.DistributedHaloPlan(owner, adjacency, part_count)
    ids = np.asarray(halo.local_global_ids)
    valid = np.asarray(halo.local_valid)
    left_indices = np.zeros_like(ids)
    right_indices = np.zeros_like(ids)
    left_valid = np.zeros_like(valid)
    right_valid = np.zeros_like(valid)
    for part in range(part_count):
        local_map = {
            int(global_id): index
            for index, global_id in enumerate(ids[part])
            if valid[part, index]
        }
        for index, global_id in enumerate(ids[part]):
            if not valid[part, index]:
                continue
            left = local_map.get(int(global_id) - 1)
            right = local_map.get(int(global_id) + 1)
            if left is not None:
                left_indices[part, index], left_valid[part, index] = left, True
            if right is not None:
                right_indices[part, index], right_valid[part, index] = right, True
    left_indices_ = jnp.asarray(left_indices)
    right_indices_ = jnp.asarray(right_indices)
    left_valid_ = jnp.asarray(left_valid)
    right_valid_ = jnp.asarray(right_valid)

    def local_action(part, local, global_ids, local_valid, local_owned):
        del global_ids
        left = jnp.where(left_valid_[part], local[left_indices_[part]], 0.0)
        right = jnp.where(right_valid_[part], local[right_indices_[part]], 0.0)
        result = 2.0 * local - left - right
        return jnp.where(local_valid & local_owned, result, 0.0)

    def local_transpose(part, local, global_ids, local_valid, local_owned):
        del global_ids
        rows = jnp.where(local_owned, local, 0.0)
        result = 2.0 * rows
        result = result.at[left_indices_[part]].add(
            jnp.where(local_owned & left_valid_[part], -rows, 0.0)
        )
        result = result.at[right_indices_[part]].add(
            jnp.where(local_owned & right_valid_[part], -rows, 0.0)
        )
        return jnp.where(local_valid, result, 0.0)

    operator = phx.discretization.DistributedLocalOperator(
        halo,
        local_action,
        local_transpose,
        operator_name=f"qualified-dirichlet-laplacian-{entity_count}",
    )
    value = jnp.sin(0.1 * jnp.arange(entity_count))
    cotangent = jnp.cos(0.13 * jnp.arange(entity_count))
    reference = 2.0 * value
    reference = reference.at[1:].add(-value[:-1])
    reference = reference.at[:-1].add(-value[1:])
    reference_transpose = 2.0 * cotangent
    reference_transpose = reference_transpose.at[1:].add(-cotangent[:-1])
    reference_transpose = reference_transpose.at[:-1].add(-cotangent[1:])
    serial = operator.serial_reference(value)
    serial_transpose = operator.serial_transpose_reference(cotangent)
    if part_count > jax.local_device_count():
        return {
            "partitions": part_count,
            "entity_count": entity_count,
            "available": False,
            "reason": "insufficient-local-devices",
            "successful": False,
        }
    start = time.perf_counter()
    distributed = operator.distributed(value)
    distributed_transpose = operator.distributed_transpose(cotangent)
    jax.block_until_ready((distributed, distributed_transpose))
    compile_and_first_ms = 1000.0 * (time.perf_counter() - start)
    repetitions = 5
    start = time.perf_counter()
    for _ in range(repetitions):
        distributed = operator.distributed(value)
        distributed_transpose = operator.distributed_transpose(cotangent)
    jax.block_until_ready((distributed, distributed_transpose))
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    forward_error = float(jnp.max(jnp.abs(distributed - reference)))
    transpose_error = float(jnp.max(jnp.abs(distributed_transpose - reference_transpose)))
    serial_error = float(jnp.max(jnp.abs(serial - reference)))
    serial_transpose_error = float(
        jnp.max(jnp.abs(serial_transpose - reference_transpose))
    )
    pairing_residual = float(
        jnp.abs(jnp.vdot(distributed, cotangent) - jnp.vdot(value, distributed_transpose))
    )
    communication_bytes = int(halo.phase_send_valid.size * value.dtype.itemsize)
    successful = (
        max(
            forward_error,
            transpose_error,
            serial_error,
            serial_transpose_error,
            pairing_residual,
        )
        < 1e-11
    )
    return {
        "partitions": part_count,
        "entity_count": entity_count,
        "local_capacity": halo.local_capacity,
        "message_capacity": halo.message_capacity,
        "communication_bytes": communication_bytes,
        "compile_and_first_ms": compile_and_first_ms,
        "execution_ms": execution_ms,
        "forward_max_error": forward_error,
        "transpose_max_error": transpose_error,
        "serial_forward_max_error": serial_error,
        "serial_transpose_max_error": serial_transpose_error,
        "pairing_residual": pairing_residual,
        "available": True,
        "successful": successful,
    }


def _advance(
    plan: geo.ConstantDensityAcousticPlan,
    acquisition: geo.SeismicAcquisition,
    state: geo.AcousticState,
    source: np.ndarray,
    start: int,
    stop: int,
):
    value = state
    for index in range(start, stop):
        value = plan.step(value, 1.5, acquisition, source[index])
    return value


def _expected_failure(error_type, action) -> bool:
    try:
        action()
    except error_type:
        return True
    return False


def _restart_and_failure_case(directory: Path) -> dict[str, object]:
    grid = geo.AcousticGrid((9, 9), (1.0, 1.0))
    acquisition = geo.SeismicAcquisition(grid, [[4.0, 4.0]], [[5.0, 4.0]])
    plan = geo.ConstantDensityAcousticPlan(grid, 0.05, 8, 2.0)
    source = np.zeros((8, 1))
    source[0, 0] = 0.1
    initial = plan.initial_state()
    uninterrupted = _advance(plan, acquisition, initial, source, 0, 8)
    midpoint = _advance(plan, acquisition, initial, source, 0, 4)
    estimate = plan.resource_estimate(receiver_count=1, checkpoint_count=1)
    policy = geo.GeophysicalResourcePolicy(
        maximum_device_bytes=max(estimate.total_bytes, 1),
        maximum_checkpoint_bytes=10_000_000,
        maximum_sources=1,
        maximum_observations=16,
        maximum_steps=8,
    )
    checkpoint = geo.GeophysicalCheckpointPlan(
        plan.plan_id, grid.grid_id, acquisition.acquisition_id, "serial", policy
    )
    continuation = geo.GeophysicalContinuationState(
        midpoint,
        time=4 * plan.time_step,
        accepted_step=4,
        source_position=4,
        random_key=jax.random.key_data(jax.random.key(7)),
        topology_epoch=0,
    )
    path = directory / "geophysics-restart.phx"
    checkpoint.write(path, continuation)
    restored = checkpoint.read(path, continuation)
    resumed = _advance(
        plan,
        acquisition,
        restored.physical_state,
        source,
        int(restored.source_position),
        8,
    )
    uninterrupted_leaves = jax.tree_util.tree_leaves(uninterrupted)
    resumed_leaves = jax.tree_util.tree_leaves(resumed)
    restart_bitwise_equal = all(
        np.array_equal(np.asarray(left), np.asarray(right))
        for left, right in zip(uninterrupted_leaves, resumed_leaves, strict=True)
    )
    mismatched = geo.GeophysicalCheckpointPlan(
        "different-plan", grid.grid_id, acquisition.acquisition_id, "serial", policy
    )
    identity_refused = _expected_failure(
        ValueError, lambda: mismatched.read(path, continuation)
    )
    oversized = geo.GeophysicalResourceEstimate(
        retained_bytes=2,
        workspace_bytes=2,
        checkpoint_bytes=2,
        observation_bytes=2,
        source_batch_size=1,
    )
    resource_refused = _expected_failure(
        MemoryError,
        lambda: geo.GeophysicalResourcePolicy(
            maximum_device_bytes=1,
            maximum_checkpoint_bytes=1,
            maximum_sources=1,
            maximum_observations=1,
            maximum_steps=1,
        ).admit(oversized, source_count=1, observation_count=1, step_count=1),
    )
    unavailable_partitions = jax.local_device_count() + 1
    owner = np.arange(unavailable_partitions)
    unavailable_halo = phx.discretization.DistributedHaloPlan(
        owner,
        np.stack(
            (np.arange(unavailable_partitions - 1), np.arange(1, unavailable_partitions)),
            axis=1,
        ),
        unavailable_partitions,
    )
    unavailable_operator = phx.discretization.DistributedLocalOperator(
        unavailable_halo,
        lambda part, local, ids, valid, owned: jnp.where(owned, local, 0.0),
        lambda part, local, ids, valid, owned: jnp.where(valid, local, 0.0),
        operator_name="unavailable-device-refusal",
    )
    device_refused = _expected_failure(
        ValueError,
        lambda: unavailable_operator.distributed(jnp.ones(unavailable_partitions)),
    )
    successful = bool(
        restart_bitwise_equal and identity_refused and resource_refused and device_refused
    )
    return {
        "checkpoint_bytes": path.stat().st_size,
        "restart_bitwise_equal": restart_bitwise_equal,
        "checkpoint_identity_refused": identity_refused,
        "resource_budget_refused": resource_refused,
        "insufficient_devices_refused": device_refused,
        "successful": successful,
    }


def run(partitions: tuple[int, ...], entities_per_part: int) -> dict[str, object]:
    cases = [
        _distributed_case(part_count, entities_per_part) for part_count in partitions
    ]
    with TemporaryDirectory(prefix="phydrax-geophysics-production-") as temporary:
        restart = _restart_and_failure_case(Path(temporary))
    available = [case for case in cases if case["available"]]
    return {
        "kind": "geophysics-production-benchmark",
        "environment": capture_environment().to_dict(),
        "local_device_count": jax.local_device_count(),
        "distributed_cases": cases,
        "restart_and_failures": restart,
        "all_available_cases_successful": bool(
            available and all(case["successful"] for case in available)
        ),
        "successful": bool(
            available
            and all(case["successful"] for case in available)
            and restart["successful"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partitions", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--entities-per-part", type=int, default=64)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.entities_per_part < 2
        or not arguments.partitions
        or any(value <= 0 for value in arguments.partitions)
    ):
        raise ValueError("Partition counts and entities per partition must be positive.")
    payload = run(tuple(arguments.partitions), arguments.entities_per_part)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["successful"] else 1)


if __name__ == "__main__":
    main()
