# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Five distinct path/executor comparisons; selected provider failures fail the run.

Run as ``python -m benchmarks.cuda_tensor_contraction --size 32 --repeats 5``.
The slice campaign fixes the same slicing label for every executor/path and uses
explicit host placement/gather, not vendor distributed execution or host AD.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax._cuda_native import cuda_native_resources, release_cuda_component
from phydrax.tensor_network import (
    ContractionExecutionPolicy,
    ContractionLeg,
    ContractionOperand,
    ContractionPathPlanner,
    ContractionPlan,
    ContractionStructure,
    create_tensor_network_mesh,
    execute_distributed_slices,
    plan_contraction,
    plan_slice_placement,
    plan_sliced_contraction,
    prepare_contraction,
    refresh_contraction,
)
from phydrax.tensor_network._cutensor import execute_executor

from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _derivative_reference(
    matrices: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Independent real matrix-product JVP/gradient/HVP along all-one tangents."""
    size = matrices[0].shape[0]

    def product(values: tuple[np.ndarray, ...]) -> tuple[np.ndarray, np.ndarray]:
        value = np.eye(size, dtype=np.float64)
        tangent = np.zeros((size, size), dtype=np.float64)
        for matrix in values:
            tangent = tangent @ matrix + value @ np.ones_like(matrix)
            value = value @ matrix
        return value, tangent

    value, tangent = product(matrices)
    gradients = []
    hvps = []
    for index in range(len(matrices)):
        left, left_tangent = product(matrices[:index])
        right, right_tangent = product(matrices[index + 1 :])
        gradients.append(left.T @ (2 * value) @ right.T)
        hvps.append(
            left_tangent.T @ (2 * value) @ right.T
            + left.T @ (2 * tangent) @ right.T
            + left.T @ (2 * value) @ right_tangent.T
        )
    return tangent.T, tuple(gradients), tuple(hvps)


def _case(
    name: str,
    plan: ContractionPlan,
    operands: tuple[Array, ...],
    reference: np.ndarray,
    derivative_reference: tuple[
        np.ndarray, tuple[np.ndarray, ...], tuple[np.ndarray, ...]
    ],
    provider: str,
    repeats: int,
    devices: tuple[jax.Device, ...],
) -> dict[str, Any]:
    started = perf_counter()
    if provider == "native":
        policy = ContractionExecutionPolicy()
    elif provider == "cutensor":
        policy = ContractionExecutionPolicy("cutensor")
    elif provider == "cutensornet":
        policy = ContractionExecutionPolicy("cutensornet")
    else:
        raise ValueError("Unknown selected benchmark provider.")
    prepared, preparation = measure_host(
        lambda: prepare_contraction(plan, operands, execution=policy)
    )
    executor = prepared.executor

    def forward(values: tuple[Array, ...]) -> Array:
        return plan.precision.output(
            execute_executor(plan, executor, plan.precision.contraction(values))
        )

    def loss(values: tuple[Array, ...]) -> Array:
        value = forward(values)
        return jnp.real(jnp.vdot(value, value))

    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(forward).lower(operands),
        lambda lowered: lowered.compile(),
    )
    first, first_seconds = measure_synchronized(lambda: compiled(operands))
    np.testing.assert_allclose(np.asarray(first), reference, rtol=2e-10, atol=2e-10)
    _, warm = measure_repeated(lambda: compiled(operands), warmup=1, repeats=repeats)
    tangents = tuple(jnp.ones_like(value) for value in operands)

    def derivatives(
        values: tuple[Array, ...], tangent_values: tuple[Array, ...]
    ) -> tuple[Array, tuple[Array, ...], tuple[Array, ...]]:
        _, jvp = jax.jvp(forward, (values,), (tangent_values,))
        gradient, hvp = jax.jvp(jax.grad(loss), (values,), (tangent_values,))
        return jvp, gradient, hvp

    derivative_program, derivative_compile = measure_lower_and_compile(
        lambda: jax.jit(derivatives).lower(operands, tangents),
        lambda lowered: lowered.compile(),
    )
    derivative_result, derivative_first = measure_synchronized(
        lambda: derivative_program(operands, tangents)
    )
    if not all(
        np.all(np.isfinite(np.asarray(value)))
        for value in jax.tree.leaves(derivative_result)
    ):
        raise ValueError("Nonfinite benchmark derivatives.")
    for observed, expected in zip(
        jax.tree.leaves(derivative_result),
        jax.tree.leaves(derivative_reference),
        strict=True,
    ):
        np.testing.assert_allclose(np.asarray(observed), expected, rtol=2e-9, atol=2e-9)
    _, derivative_warm = measure_repeated(
        lambda: derivative_program(operands, tangents), warmup=1, repeats=repeats
    )
    changed = tuple(value + jnp.asarray(0.125, dtype=value.dtype) for value in operands)
    refreshed, refresh_seconds = measure_synchronized(
        lambda: refresh_contraction(prepared, changed)
    )
    refreshed_value, refreshed_execution = measure_synchronized(
        lambda: compiled(refreshed.operands)
    )
    changed_host = tuple(np.asarray(value) for value in changed)
    changed_reference = (
        changed_host[0] @ changed_host[1] @ changed_host[2] @ changed_host[3]
    ).T
    np.testing.assert_allclose(
        np.asarray(refreshed_value), changed_reference, rtol=2e-10, atol=2e-10
    )
    rebuild_seconds = None
    selected_provider = policy.provider
    if selected_provider != "native":
        release_cuda_component(selected_provider)
        rebuilt, rebuild_seconds = measure_synchronized(lambda: compiled(operands))
        np.testing.assert_allclose(np.asarray(rebuilt), reference, rtol=2e-10, atol=2e-10)
    sliced, slice_planning = measure_host(
        lambda: plan_sliced_contraction(plan, ("b",), batch_size=1)
    )
    placement = plan_slice_placement(sliced, create_tensor_network_mesh(devices))
    distributed, slice_end_to_end = measure_synchronized(
        lambda: execute_distributed_slices(prepared, placement)
    )
    if distributed.aggregate is None or not bool(distributed.evidence.accepted):
        raise ValueError(f"Slice campaign failed: {distributed.evidence.failure}")
    np.testing.assert_allclose(
        np.asarray(distributed.aggregate), reference, rtol=2e-10, atol=2e-10
    )
    resources = cuda_native_resources()
    cost = compiled.cost_analysis()
    return {
        "name": name,
        "provider": provider,
        "path_source": plan.optimizer,
        "path": plan.path,
        "path_provenance": plan.path_provenance,
        "plan_id": plan.plan_id,
        "execution_id": executor.lowering_id,
        "build_identity": executor.build_identity,
        "planning_included_here": False,
        "case_end_to_end_seconds": perf_counter() - started,
        "case_end_to_end_scope": "preparation, all warm samples, derivatives, oracle checks, refresh, rebuild, and host slice campaign; excludes separately reported path planning",
        "preparation_seconds": preparation,
        "tuning": "not requested; first execution includes vendor descriptor/plan reconstruction",
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_seconds": first_seconds,
        "warm": warm.to_dict(),
        "derivative_lowering_seconds": derivative_compile.lowering_seconds,
        "derivative_compilation_seconds": derivative_compile.compilation_seconds,
        "derivative_first_seconds": derivative_first,
        "derivative_warm": derivative_warm.to_dict(),
        "refresh_seconds": refresh_seconds,
        "refreshed_execution_seconds": refreshed_execution,
        "release_rebuild_seconds": rebuild_seconds,
        "compiler": {
            **asdict(
                compiler_evidence(
                    cost,
                    compiled.memory_analysis(),
                    source="jax-compiled",
                    opaque_custom_calls=provider != "native",
                )
            ),
            "raw_cost_analysis": cost,
        },
        "logical_retained_array_bytes": logical_array_bytes(prepared),
        "normalization_bound_bytes": executor.normalization_bytes,
        "external_workspace_bound_bytes": executor.external_workspace_bound_bytes,
        "external_cache_bound_bytes": executor.external_cache_bound_bytes,
        "opaque_vendor_host_bound_bytes": executor.vendor_host_bound_bytes,
        "observed_native_registry_entries": sum(
            item.live for item in resources.partitions if item.component == provider
        ),
        "slice": {
            "labels": sliced.labels,
            "count": sliced.slice_count,
            "residual_path": sliced.residual.path,
            "planning_seconds": slice_planning,
            "end_to_end_seconds": slice_end_to_end,
            "external_workspace_bound_bytes_per_device": executor.external_workspace_bound_bytes,
            "external_cache_bound_bytes_per_device": executor.external_cache_bound_bytes,
            "operand_replication_bound_bytes": sum(value.nbytes for value in operands)
            * len(devices),
            "transfer_completed_bytes": distributed.evidence.transfer.completed_bytes,
            "reduction": "host gather, global ordinal sum; transfer/reduction included in end-to-end, not independently timed",
            "device_ids": placement.mesh.device_ids,
            "host_differentiable": False,
        },
        "accepted": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.size < 2 or arguments.repeats < 1 or arguments.devices < 1:
        parser.error("size>=2, repeats>=1, and devices>=1 are required")
    record: dict[str, Any] = {"accepted": False, "cases": []}
    try:
        devices = tuple(jax.devices("gpu")[: arguments.devices])
        if len(devices) != arguments.devices:
            raise ValueError("The requested physical GPU count is unavailable.")
        size = arguments.size
        labels = ("a", "b", "c", "d", "e")
        structure = ContractionStructure(
            tuple(
                ContractionOperand(
                    f"matrix-{index}",
                    (
                        ContractionLeg(labels[index], size),
                        ContractionLeg(labels[index + 1], size),
                    ),
                )
                for index in range(4)
            ),
            ("e", "a"),
        )
        generator = np.random.default_rng(408)
        host = tuple(
            generator.normal(size=(size, size)).astype(np.float64) / np.sqrt(size)
            for _ in range(4)
        )
        operands, transfer = measure_synchronized(
            lambda: tuple(jax.device_put(value, devices[0]) for value in host)
        )
        reference = (host[0] @ host[1] @ host[2] @ host[3]).T
        derivative_reference = _derivative_reference(host)
        native, native_planning = measure_host(
            lambda: plan_contraction(structure, dtype="float64")
        )
        vendor, vendor_planning = measure_host(
            lambda: plan_contraction(
                structure, dtype="float64", path_planner=ContractionPathPlanner()
            )
        )
        record.update(
            {
                "environment": capture_environment().to_dict(),
                "identity": capture_benchmark_identity(
                    Path(__file__).resolve().parents[1],
                    Path(__file__),
                    ("cases", "planning", "accepted"),
                ).to_dict(),
                "input_transfer_seconds": transfer,
                "planning": {
                    "native_seconds": native_planning,
                    "cutensornet_seconds": vendor_planning,
                    "paths_differ": native.path != vendor.path,
                },
            }
        )
        for name, plan, provider in (
            ("native-path-native-executor", native, "native"),
            ("native-path-cutensor-executor", native, "cutensor"),
            ("native-path-cutensornet-executor", native, "cutensornet"),
            ("cutensornet-path-native-executor", vendor, "native"),
            ("cutensornet-path-cutensornet-executor", vendor, "cutensornet"),
        ):
            record["cases"].append(
                _case(
                    name,
                    plan,
                    operands,
                    reference,
                    derivative_reference,
                    provider,
                    arguments.repeats,
                    devices,
                )
            )
        record["accepted"] = True
    except Exception as error:
        record["failure"] = f"{type(error).__name__}: {error}"
    text = json.dumps(record, indent=2, allow_nan=False)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if record["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
