#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from math import prod
from time import monotonic

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize, PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._core import MatrixProductOperator, MatrixProductState
from ._precision import TensorNetworkPrecisionPolicy
from ._schedule import (
    build_contraction_schedule,
    ContractionSchedule,
    planner_search_state_bound,
    reverse_schedule_evidence,
    ReverseScheduleEvidence,
)
from ._topology import ContractionLeg, ContractionOperand, ContractionStructure


class ContractionResourcePolicy(StrictModule):
    maximum_operand_elements: int = eqx.field(static=True)
    maximum_intermediate_elements: int = eqx.field(static=True)
    maximum_output_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_flops: int = eqx.field(static=True)
    maximum_schedule_steps: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_operand_elements: int = 100_000_000,
        maximum_intermediate_elements: int = 100_000_000,
        maximum_output_elements: int = 100_000_000,
        maximum_workspace_bytes: int = 2**31,
        maximum_flops: int = 10**15,
        maximum_schedule_steps: int = 100_000,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_operand_elements,
                maximum_intermediate_elements,
                maximum_output_elements,
                maximum_workspace_bytes,
                maximum_flops,
                maximum_schedule_steps,
            )
        )
        if any(value < 1 for value in values):
            raise ValueError("Contraction resource limits must be positive.")
        self.maximum_operand_elements = values[0]
        self.maximum_intermediate_elements = values[1]
        self.maximum_output_elements = values[2]
        self.maximum_workspace_bytes = values[3]
        self.maximum_flops = values[4]
        self.maximum_schedule_steps = values[5]
        self.policy_id = canonical_fingerprint(
            {"kind": "contraction-resource-policy", "limits": values}
        )


class ContractionPlannerPolicy(StrictModule):
    maximum_search_states: int = eqx.field(static=True)
    maximum_planning_seconds: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_search_states: int = 250_000,
        maximum_planning_seconds: float = 30.0,
    ):
        states = int(maximum_search_states)
        seconds = float(maximum_planning_seconds)
        if states < 1 or not 0.0 < seconds < float("inf"):
            raise ValueError("Planner state and time bounds must be finite and positive.")
        self.maximum_search_states = states
        self.maximum_planning_seconds = seconds
        self.policy_id = canonical_fingerprint(
            {
                "kind": "contraction-planner-policy",
                "maximum_search_states": states,
                "maximum_planning_seconds": seconds,
            }
        )


class ContractionCostEstimate(StrictModule):
    operand_elements: int = eqx.field(static=True)
    output_elements: int = eqx.field(static=True)
    largest_intermediate_elements: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    peak_live_elements: int = eqx.field(static=True)
    peak_live_bytes: int = eqx.field(static=True)
    estimated_flops: int = eqx.field(static=True)
    planner_search_bound: int = eqx.field(static=True)


class ContractionPlan(StrictModule):
    structure: ContractionStructure
    resources: ContractionResourcePolicy
    planner: ContractionPlannerPolicy
    precision: TensorNetworkPrecisionPolicy
    cost: ContractionCostEstimate
    schedule: ContractionSchedule
    equation: str = eqx.field(static=True)
    path: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    optimizer: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class PreparedContraction(StrictModule):
    plan: ContractionPlan
    operands: tuple[Array, ...]
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)


class ContractionExecutionEvidence(StrictModule):
    finite: Array
    accepted: Array
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)
    peak_live_bytes: int = eqx.field(static=True)
    numeric_version: Array


class ContractionResult(StrictModule):
    value: Array
    evidence: ContractionExecutionEvidence


class ReverseContractionResult(StrictModule):
    value: Array
    operand_cotangents: tuple[Array, ...]
    evidence: ContractionExecutionEvidence
    reverse_evidence: ReverseScheduleEvidence


class ContractionPlanCache(NonTrainableState):
    """Caller-owned, capacity-bounded host cache; no process-global cache exists."""

    def __init__(self, capacity: int = 32, /):
        capacity_ = int(capacity)
        if capacity_ < 1:
            raise ValueError("Contraction plan cache capacity must be positive.")
        self.capacity = capacity_
        self._plans: OrderedDict[str, ContractionPlan] = OrderedDict()

    def lookup(self, key: str, /) -> ContractionPlan | None:
        return self._plans.get(str(key))

    def store(self, key: str, plan: ContractionPlan, /) -> None:
        key_ = str(key)
        if not isinstance(plan, ContractionPlan):
            raise TypeError("plan must be ContractionPlan.")
        if key_ in self._plans:
            del self._plans[key_]
        elif len(self._plans) == self.capacity:
            self._plans.popitem(last=False)
        self._plans[key_] = plan

    def clear(self) -> None:
        self._plans.clear()

    def __len__(self) -> int:
        return len(self._plans)


def _equation(structure: ContractionStructure, /) -> str:
    symbols = {
        label: oe.get_symbol(index) for index, label in enumerate(structure.labels)
    }
    inputs = ",".join(
        "".join(symbols[leg.label] for leg in operand.legs)
        for operand in structure.operands
    )
    output = "".join(symbols[label] for label in structure.outputs)
    return f"{inputs}->{output}"


def _plan_cache_key(
    structure: ContractionStructure,
    resources: ContractionResourcePolicy,
    planner: ContractionPlannerPolicy,
    precision: TensorNetworkPrecisionPolicy,
    optimizer: str,
    dtype: str,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "contraction-plan-request",
            "structure": structure.structure_id,
            "resources": resources.policy_id,
            "planner": planner.policy_id,
            "precision": precision.policy_id,
            "optimizer": optimizer,
            "dtype": dtype,
        }
    )


def plan_contraction(
    structure: ContractionStructure,
    /,
    *,
    precision: TensorNetworkPrecisionPolicy | None = None,
    resources: ContractionResourcePolicy | None = None,
    planner: ContractionPlannerPolicy | None = None,
    cache: ContractionPlanCache | None = None,
    optimizer: str = "greedy",
    dtype: str = "complex128",
) -> ContractionPlan:
    """Plan exactly, refusing search and execution resources before allocation."""

    if not isinstance(structure, ContractionStructure):
        raise TypeError("structure must be ContractionStructure.")
    precision_ = TensorNetworkPrecisionPolicy() if precision is None else precision
    resources_ = ContractionResourcePolicy() if resources is None else resources
    planner_ = ContractionPlannerPolicy() if planner is None else planner
    if not isinstance(precision_, TensorNetworkPrecisionPolicy):
        raise TypeError("precision must be TensorNetworkPrecisionPolicy or None.")
    if not isinstance(resources_, ContractionResourcePolicy):
        raise TypeError("resources must be ContractionResourcePolicy or None.")
    if not isinstance(planner_, ContractionPlannerPolicy):
        raise TypeError("planner must be ContractionPlannerPolicy or None.")
    if cache is not None and not isinstance(cache, ContractionPlanCache):
        raise TypeError("cache must be ContractionPlanCache or None.")
    if optimizer not in ("greedy", "optimal"):
        raise ValueError("optimizer must be 'greedy' or 'optimal'.")
    dtype_ = jnp.dtype(dtype).name
    key = _plan_cache_key(structure, resources_, planner_, precision_, optimizer, dtype_)
    if cache is not None:
        cached = cache.lookup(key)
        if cached is not None:
            return cached

    shapes = tuple(
        tuple(leg.dimension for leg in operand.legs) for operand in structure.operands
    )
    operand_elements = sum(prod(shape) for shape in shapes)
    output_elements = structure.output_elements
    if operand_elements > resources_.maximum_operand_elements:
        raise MemoryError("Contraction operands exceed maximum_operand_elements.")
    if output_elements > resources_.maximum_output_elements:
        raise MemoryError("Contraction output exceeds maximum_output_elements.")
    search_bound = planner_search_state_bound(len(shapes), optimizer)
    if search_bound > planner_.maximum_search_states:
        raise RuntimeError(
            "Contraction planner search exceeds maximum_search_states before planning."
        )

    equation = _equation(structure)
    started = monotonic()
    path, information = oe.contract_path(
        equation, *shapes, shapes=True, optimize=optimizer
    )
    elapsed = monotonic() - started
    if elapsed > planner_.maximum_planning_seconds:
        raise TimeoutError("Contraction planning exceeded maximum_planning_seconds.")
    path_ = tuple(tuple(int(index) for index in step) for step in path)
    schedule = build_contraction_schedule(structure, path_, dtype=dtype_)
    largest = max(
        (step.output_elements for step in schedule.steps), default=output_elements
    )
    flops = max(int(information.opt_cost), schedule.total_estimated_flops)
    workspace_bytes = schedule.peak_live_bytes
    if len(schedule.steps) > resources_.maximum_schedule_steps:
        raise MemoryError("Contraction path exceeds maximum_schedule_steps.")
    if largest > resources_.maximum_intermediate_elements:
        raise MemoryError("Contraction path exceeds maximum_intermediate_elements.")
    if workspace_bytes > resources_.maximum_workspace_bytes:
        raise MemoryError("Contraction path exceeds maximum_workspace_bytes.")
    if flops > resources_.maximum_flops:
        raise MemoryError("Contraction path exceeds maximum_flops.")
    cost = ContractionCostEstimate(
        operand_elements,
        output_elements,
        largest,
        workspace_bytes,
        schedule.peak_live_elements,
        schedule.peak_live_bytes,
        flops,
        search_bound,
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "contraction-plan",
            "request": key,
            "equation": equation,
            "path": path_,
            "schedule": schedule.schedule_id,
        }
    )
    plan = ContractionPlan(
        structure,
        resources_,
        planner_,
        precision_,
        cost,
        schedule,
        equation,
        path_,
        optimizer,
        dtype_,
        True,
        "exact ordinary product-sum contraction",
        plan_id,
    )
    if cache is not None:
        cache.store(key, plan)
    return plan


def _validate_operands(plan: ContractionPlan, operands: Sequence[ArrayLike], /):
    arrays = tuple(jnp.asarray(value) for value in operands)
    if len(arrays) != len(plan.structure.operands):
        raise ValueError("Operand count differs from the contraction plan.")
    for array, specification in zip(arrays, plan.structure.operands, strict=True):
        expected = tuple(leg.dimension for leg in specification.legs)
        if array.shape != expected:
            raise ValueError(
                f"Operand {specification.operand_id!r} expected shape {expected}; "
                f"got {array.shape}."
            )
        if str(array.dtype) != plan.dtype:
            raise TypeError("Operand dtype differs from the contraction plan.")
    plan.precision.validate_storage(arrays)
    return arrays


def prepare_contraction(
    plan: ContractionPlan,
    operands: Sequence[ArrayLike],
    /,
) -> PreparedContraction:
    if not isinstance(plan, ContractionPlan):
        raise TypeError("plan must be ContractionPlan.")
    arrays = _validate_operands(plan, operands)
    prepared_id = canonical_fingerprint(
        {"kind": "prepared-contraction", "plan": plan.plan_id}
    )
    return PreparedContraction(
        plan,
        arrays,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
    )


def refresh_contraction(
    prepared: PreparedContraction,
    operands: Sequence[ArrayLike],
    /,
) -> PreparedContraction:
    if not isinstance(prepared, PreparedContraction):
        raise TypeError("prepared must be PreparedContraction.")
    arrays = _validate_operands(prepared.plan, operands)
    return PreparedContraction(
        prepared.plan,
        arrays,
        prepared.numeric_version + 1,
        prepared.prepared_id,
    )


def execute_schedule(
    schedule: ContractionSchedule,
    operands: Sequence[Array],
    /,
) -> Array:
    """Replay one admitted schedule without path search or backend switching."""

    if len(operands) != len(schedule.initial_value_ids):
        raise ValueError("Schedule operand count does not match initial SSA values.")
    value_ids = list(schedule.initial_value_ids)
    values = list(operands)
    for step in schedule.steps:
        positions = tuple(value_ids.index(value_id) for value_id in step.input_value_ids)
        selected = tuple(values[position] for position in positions)
        result = oe.contract(
            step.equation,
            *selected,
            optimize=[tuple(range(len(selected)))],
            backend="jax",
        )
        for position in sorted(positions, reverse=True):
            value_ids.pop(position)
            values.pop(position)
        value_ids.append(step.output_value_id)
        values.append(result)
    if value_ids != [schedule.output_value_id]:
        raise RuntimeError(
            "Schedule replay did not produce its declared output SSA value."
        )
    return jnp.asarray(values[0])


def _execution_evidence(
    prepared: PreparedContraction, output: Array, /
) -> ContractionExecutionEvidence:
    plan = prepared.plan
    finite = jnp.all(jnp.isfinite(output))
    replay_id = canonical_fingerprint(
        {
            "kind": "contraction-replay",
            "schedule": plan.schedule.schedule_id,
            "prepared": prepared.prepared_id,
        }
    )
    return ContractionExecutionEvidence(
        finite,
        finite,
        plan.precision.evidence_for(prepared.operands, output_value=output),
        plan.structure.structure_id,
        plan.plan_id,
        plan.schedule.schedule_id,
        prepared.prepared_id,
        replay_id,
        True,
        plan.claim,
        plan.schedule.peak_live_bytes,
        prepared.numeric_version,
    )


def execute_contraction(prepared: PreparedContraction, /) -> ContractionResult:
    if not isinstance(prepared, PreparedContraction):
        raise TypeError("prepared must be PreparedContraction.")
    plan = prepared.plan
    operands = plan.precision.contraction(prepared.operands)
    value = execute_schedule(plan.schedule, operands)
    output = plan.precision.output(value)
    return ContractionResult(output, _execution_evidence(prepared, output))


def execute_contraction_reverse(
    prepared: PreparedContraction,
    /,
    *,
    cotangent: ArrayLike | None = None,
    rematerialization: str = "store",
) -> ReverseContractionResult:
    """Replay a differentiable reverse schedule with explicit retention evidence."""

    if not isinstance(prepared, PreparedContraction):
        raise TypeError("prepared must be PreparedContraction.")
    if rematerialization not in ("store", "rematerialize"):
        raise ValueError("rematerialization must be 'store' or 'rematerialize'.")
    plan = prepared.plan
    operand_bytes = sum(
        operand.size * precision_itemsize(str(operand.dtype))
        for operand in prepared.operands
    )
    output_bytes = plan.cost.output_elements * precision_itemsize(plan.dtype)
    reverse = reverse_schedule_evidence(
        plan.schedule,
        rematerialization=rematerialization,
        operand_bytes=operand_bytes,
        output_bytes=output_bytes,
    )
    if reverse.reverse_peak_bytes > plan.resources.maximum_workspace_bytes:
        raise MemoryError(
            "Reverse contraction exceeds maximum_workspace_bytes before allocation."
        )

    def contraction(*operands):
        converted = plan.precision.contraction(operands)
        return plan.precision.output(execute_schedule(plan.schedule, converted))

    replay = (
        jax.checkpoint(contraction)
        if rematerialization == "rematerialize"
        else contraction
    )
    value, pullback = jax.vjp(replay, *prepared.operands)
    cotangent_ = jnp.ones_like(value) if cotangent is None else jnp.asarray(cotangent)
    if cotangent_.shape != value.shape:
        raise ValueError("cotangent shape must equal the contraction output shape.")
    gradients = tuple(pullback(cotangent_))
    return ReverseContractionResult(
        value,
        gradients,
        _execution_evidence(prepared, value),
        reverse,
    )


def prepare_mps_inner_contraction(
    left: MatrixProductState,
    right: MatrixProductState,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
) -> PreparedContraction:
    if left.physical_dimensions != right.physical_dimensions:
        raise ValueError("MPS physical dimensions must match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPS precision policies must match.")
    operands = []
    arrays = []
    for site, (first, second) in enumerate(zip(left.tensors, right.tensors, strict=True)):
        operands.extend(
            (
                ContractionOperand(
                    f"bra-{site}",
                    (
                        ContractionLeg(f"bra-bond-{site}", first.shape[0]),
                        ContractionLeg(f"physical-{site}", first.shape[1]),
                        ContractionLeg(f"bra-bond-{site + 1}", first.shape[2]),
                    ),
                ),
                ContractionOperand(
                    f"ket-{site}",
                    (
                        ContractionLeg(f"ket-bond-{site}", second.shape[0]),
                        ContractionLeg(f"physical-{site}", second.shape[1]),
                        ContractionLeg(f"ket-bond-{site + 1}", second.shape[2]),
                    ),
                ),
            )
        )
        arrays.extend((jnp.conj(first), second))
    for name in ("bra", "ket"):
        operands.extend(
            (
                ContractionOperand(
                    f"{name}-left-boundary",
                    (ContractionLeg(f"{name}-bond-0", 1),),
                ),
                ContractionOperand(
                    f"{name}-right-boundary",
                    (ContractionLeg(f"{name}-bond-{left.site_count}", 1),),
                ),
            )
        )
        arrays.extend((jnp.ones((1,), dtype=left.tensors[0].dtype),) * 2)
    structure = ContractionStructure(tuple(operands), ())
    plan = plan_contraction(
        structure,
        precision=left.precision,
        resources=resources,
        dtype=str(left.tensors[0].dtype),
    )
    return prepare_contraction(plan, tuple(arrays))


def prepare_mpo_inner_contraction(
    left: MatrixProductOperator,
    right: MatrixProductOperator,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
) -> PreparedContraction:
    if (
        left.output_dimensions != right.output_dimensions
        or left.input_dimensions != right.input_dimensions
    ):
        raise ValueError("MPO dimensions must match.")
    if left.precision.policy_id != right.precision.policy_id:
        raise ValueError("MPO precision policies must match.")
    operands = []
    arrays = []
    for site, (first, second) in enumerate(zip(left.tensors, right.tensors, strict=True)):
        operands.extend(
            (
                ContractionOperand(
                    f"left-{site}",
                    (
                        ContractionLeg(f"left-bond-{site}", first.shape[0]),
                        ContractionLeg(f"output-{site}", first.shape[1]),
                        ContractionLeg(f"input-{site}", first.shape[2]),
                        ContractionLeg(f"left-bond-{site + 1}", first.shape[3]),
                    ),
                ),
                ContractionOperand(
                    f"right-{site}",
                    (
                        ContractionLeg(f"right-bond-{site}", second.shape[0]),
                        ContractionLeg(f"output-{site}", second.shape[1]),
                        ContractionLeg(f"input-{site}", second.shape[2]),
                        ContractionLeg(f"right-bond-{site + 1}", second.shape[3]),
                    ),
                ),
            )
        )
        arrays.extend((jnp.conj(first), second))
    for name in ("left", "right"):
        operands.extend(
            (
                ContractionOperand(
                    f"{name}-left-boundary",
                    (ContractionLeg(f"{name}-bond-0", 1),),
                ),
                ContractionOperand(
                    f"{name}-right-boundary",
                    (ContractionLeg(f"{name}-bond-{left.site_count}", 1),),
                ),
            )
        )
        arrays.extend((jnp.ones((1,), dtype=left.tensors[0].dtype),) * 2)
    structure = ContractionStructure(tuple(operands), ())
    plan = plan_contraction(
        structure,
        precision=left.precision,
        resources=resources,
        dtype=str(left.tensors[0].dtype),
    )
    return prepare_contraction(plan, tuple(arrays))


__all__ = [
    "ContractionCostEstimate",
    "ContractionExecutionEvidence",
    "ContractionLeg",
    "ContractionOperand",
    "ContractionPlan",
    "ContractionPlanCache",
    "ContractionPlannerPolicy",
    "ContractionResourcePolicy",
    "ContractionResult",
    "ContractionStructure",
    "PreparedContraction",
    "ReverseContractionResult",
    "execute_contraction",
    "execute_contraction_reverse",
    "execute_schedule",
    "plan_contraction",
    "prepare_contraction",
    "prepare_mpo_inner_contraction",
    "prepare_mps_inner_contraction",
    "refresh_contraction",
]
