# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Explicit normalized-mode lowering shared by the two optional CUDA executors."""

from __future__ import annotations

from dataclasses import asdict
from math import prod
from typing import TYPE_CHECKING

import jax
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from ..ein import contract
from ._contraction_provider import (
    ContractionExecutionPolicy,
    ContractionExecutor,
    TensorOperation,
)


if TYPE_CHECKING:
    from ._contraction import ContractionPlan


def prepare_executor(
    plan: ContractionPlan, policy: ContractionExecutionPolicy, /
) -> ContractionExecutor:
    if not isinstance(policy, ContractionExecutionPolicy):
        raise TypeError("execution must be ContractionExecutionPolicy.")
    if policy.provider == "native":
        if any(len(step.input_value_ids) > 2 for step in plan.schedule.steps):
            raise ValueError(
                "Native JAX execution admits unary/binary SSA steps only; an imported multi-input step cannot be silently split."
            )
        return ContractionExecutor(
            policy,
            (),
            "native-jax",
            canonical_fingerprint(
                {
                    "kind": "native-contraction-executor",
                    "schedule": plan.schedule.schedule_id,
                    "policy": policy.policy_id,
                }
            ),
            0,
            0,
            0,
            0,
            "native JAX compositional AD and batching",
        )
    from .._cuda_native import cuda_partition_key, load_cuda_component
    from ._contraction_ad import admit_derivatives

    provider = policy.provider
    if provider not in ("cutensor", "cutensornet"):
        raise ValueError("Unknown contraction provider.")
    compute, accumulation = plan.precision.resolve_contraction(plan.dtype)
    if compute != accumulation:
        raise ValueError(
            f"{provider} prepared execution requires equal multiply and accumulation dtypes; "
            "a wider requested accumulation is not silently relabeled."
        )
    if provider == "cutensor" and any(
        len(step.input_value_ids) > 2 for step in plan.schedule.steps
    ):
        raise ValueError(
            "cuTENSOR admits unary/binary SSA steps only; splitting an imported multi-input step would change its path."
        )
    if provider == "cutensornet" and any(len(step) != 2 for step in plan.path):
        raise ValueError(
            "cuTensorNet fixed-path execution requires a binary evolving-live-list path."
        )
    from importlib.metadata import version

    identity = load_cuda_component(provider)
    distributions = (
        ("cutensor-cu13",)
        if provider == "cutensor"
        else ("cutensor-cu13", "cutensornet-cu13")
    )
    build = canonical_fingerprint(
        {
            "component": identity.name,
            "version": identity.version,
            "build": identity.build_hash,
            "vendor_distributions": tuple(
                (name, version(name)) for name in distributions
            ),
        }
    )
    base = canonical_fingerprint(
        {
            "kind": "cuda-contraction-executor",
            "plan": plan.plan_id,
            "provider": provider,
            "policy": policy.policy_id,
            "build": build,
            "normalization": "unique-mode-diagonal-gather",
            "ad": "native-multilinear-through-second-order",
            "layout": "row-major",
        }
    )
    if provider == "cutensornet":
        base = canonical_fingerprint(
            {
                "execution": base,
                "scalar_representation": "zero-copy-shared-private-unit-axis",
            }
        )
    dimensions = tuple(enumerate(plan.structure.dimensions))
    mode_ids = {label: index for index, label in enumerate(plan.structure.labels)}
    normalization_bytes = sum(
        prod(leg.dimension for leg in operand.legs) * precision_itemsize(compute)
        for operand in plan.structure.operands
        if len({leg.label for leg in operand.legs}) != len(operand.legs)
    )
    # ffi_call requests row-major buffers. XLA may materialize each input at
    # that visible boundary, and explicit storage-to-compute casts may overlap.
    # These are conservative payload bounds, separate from compiler estimates.
    input_compute_bytes = plan.cost.operand_elements * precision_itemsize(compute)
    normalization_bytes += input_compute_bytes
    if compute != plan.dtype:
        normalization_bytes += input_compute_bytes
    derivative_budget = (
        policy.maximum_derivative_bytes - plan.cost.workspace_bytes - normalization_bytes
    )
    if derivative_budget < 1:
        raise MemoryError(
            "Retained network/layout/cast buffers exhaust maximum_derivative_bytes."
        )

    def operation(
        inputs: tuple[tuple[int, ...], ...],
        output: tuple[int, ...],
        path: tuple[tuple[int, ...], ...],
    ) -> TensorOperation:
        result = TensorOperation(
            inputs,
            output,
            dimensions,
            path,
            compute,
            provider,
            policy.maximum_workspace_bytes,
            policy.maximum_cache_bytes,
            policy.maximum_cache_entries,
            policy.maximum_batch_size,
            derivative_budget,
            cuda_partition_key(base),
        )
        derivative_bytes = admit_derivatives(result)
        if (
            derivative_bytes + plan.cost.workspace_bytes + normalization_bytes
            > policy.maximum_derivative_bytes
        ):
            raise MemoryError(
                "Tensor derivatives plus retained network/layout/cast buffers exceed maximum_derivative_bytes."
            )
        return result

    if provider == "cutensornet":
        inputs = tuple(
            tuple(dict.fromkeys(mode_ids[leg.label] for leg in operand.legs))
            for operand in plan.structure.operands
        )
        operations = (
            operation(
                inputs,
                tuple(mode_ids[label] for label in plan.structure.outputs),
                plan.path,
            ),
        )
    else:
        live = {
            operand.operand_id: tuple(
                dict.fromkeys(mode_ids[leg.label] for leg in operand.legs)
            )
            for operand in plan.structure.operands
        }
        operations_ = []
        for step in plan.schedule.steps:
            inputs = tuple(live[value_id] for value_id in step.input_value_ids)
            output = tuple(mode_ids[label] for label in step.output_labels)
            operations_.append(operation(inputs, output, ()))
            for value_id in step.input_value_ids:
                del live[value_id]
            live[step.output_value_id] = output
        if not operations_:
            only = next(iter(live.values()))
            operations_.append(
                operation(
                    (only,),
                    tuple(mode_ids[label] for label in plan.structure.outputs),
                    (),
                )
            )
        operations = tuple(operations_)
    if (
        plan.cost.workspace_bytes + normalization_bytes + policy.maximum_cache_bytes
        > plan.resources.maximum_workspace_bytes
    ):
        raise MemoryError(
            "Provider normalization and retained external scratch/cache exceed maximum_workspace_bytes."
        )
    lowering_id = canonical_fingerprint(
        {
            "execution": base,
            "operations": tuple(asdict(value) for value in operations),
            "normalization_bytes": normalization_bytes,
        }
    )
    return ContractionExecutor(
        policy,
        operations,
        build,
        lowering_id,
        normalization_bytes,
        policy.maximum_workspace_bytes,
        policy.maximum_cache_bytes,
        None,
        "native-owned multilinear JVP/VJP, serial bounded batching, derivatives through order two; host orchestration excluded",
    )


def _normalize(
    plan: ContractionPlan, operands: tuple[Array, ...], /
) -> tuple[Array, ...]:
    mode_ids = {label: index for index, label in enumerate(plan.structure.labels)}
    values = []
    for value, specification in zip(operands, plan.structure.operands, strict=True):
        modes = tuple(leg.label for leg in specification.legs)
        unique = tuple(dict.fromkeys(modes))
        if unique != modes:
            # Explicit diagonal gather only: every distinct input mode is retained.
            # Its native transpose scatters the provider cotangent onto that diagonal.
            value = contract(
                value,
                tuple(mode_ids[mode] for mode in modes),
                tuple(mode_ids[mode] for mode in unique),
                optimize=[(0,)],
                backend="jax",
                precision=jax.lax.Precision.HIGHEST,
            )
        values.append(value)
    return tuple(values)


def execute_executor(
    plan: ContractionPlan, executor: ContractionExecutor, operands: tuple[Array, ...], /
) -> Array:
    if executor.policy.provider == "native":
        from ._contraction import execute_schedule

        _, accumulation = plan.precision.resolve_contraction(plan.dtype)
        return execute_schedule(plan.schedule, operands, accumulation_dtype=accumulation)
    from ._contraction_ad import execute_operation

    values = _normalize(plan, operands)
    if executor.policy.provider == "cutensornet":
        return execute_operation(executor.operations[0], values)
    if not plan.schedule.steps:
        return execute_operation(executor.operations[0], values)
    live = dict(zip(plan.schedule.initial_value_ids, values, strict=True))
    for step, operation in zip(plan.schedule.steps, executor.operations, strict=True):
        selected = tuple(live.pop(value_id) for value_id in step.input_value_ids)
        live[step.output_value_id] = execute_operation(operation, selected)
    return live[plan.schedule.output_value_id]
