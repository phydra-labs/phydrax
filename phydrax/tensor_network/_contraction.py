#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from math import prod
from multiprocessing import get_context
from multiprocessing.connection import Connection
from numbers import Integral
from time import perf_counter
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jax import Array
from jax.core import Tracer
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import (
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionResolution,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._contraction_provider import (
    ContractionExecutionPolicy,
    ContractionExecutor,
    ContractionPathPlanner,
)
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
    ) -> None:
        values = tuple(
            (
                maximum_operand_elements,
                maximum_intermediate_elements,
                maximum_output_elements,
                maximum_workspace_bytes,
                maximum_flops,
                maximum_schedule_steps,
            )
        )
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) for value in values
        ):
            raise TypeError("Contraction resource limits must be integers.")
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
    ) -> None:
        if not isinstance(maximum_search_states, Integral) or isinstance(
            maximum_search_states, bool
        ):
            raise TypeError("maximum_search_states must be an integer.")
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
    path_provenance: str = eqx.field(static=True)


class PreparedContraction(StrictModule):
    plan: ContractionPlan
    operands: tuple[Array, ...]
    numeric_version: Array
    prepared_id: str = eqx.field(static=True)
    operand_id: str = eqx.field(static=True)
    executor: ContractionExecutor


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

    execution_id: str = eqx.field(static=True)
    provider: str = eqx.field(static=True)
    external_workspace_bound_bytes: int = eqx.field(static=True)
    external_cache_bound_bytes: int = eqx.field(static=True)
    vendor_host_bound_bytes: int | None = eqx.field(static=True)
    derivative_contract: str = eqx.field(static=True)


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

    def __init__(self, capacity: int = 32, /) -> None:
        if not isinstance(capacity, Integral) or isinstance(capacity, bool):
            raise TypeError("Contraction plan cache capacity must be an integer.")
        capacity_ = int(capacity)
        if capacity_ < 1:
            raise ValueError("Contraction plan cache capacity must be positive.")
        self.capacity = capacity_
        self._plans: OrderedDict[str, ContractionPlan] = OrderedDict()

    def lookup(self, key: str, /) -> ContractionPlan | None:
        return self._plans.get(str(key))

    @checked
    def store(self, key: str, plan: ContractionPlan, /) -> None:
        key_ = str(key)
        if key_ in self._plans:
            del self._plans[key_]
        elif len(self._plans) == self.capacity:
            self._plans.popitem(last=False)
        self._plans[key_] = plan

    def clear(self) -> None:
        self._plans.clear()

    def __len__(self) -> int:
        return len(self._plans)


def _bounded_contract_path_worker(
    connection: Connection,
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
    optimizer: Literal["greedy", "optimal"],
) -> None:
    try:
        # Readiness separates startup from search evidence; the owner bounds
        # their combined wall time and can terminate either phase.
        connection.send(None)
        started = perf_counter()
        path, information = oe.contract_path(
            equation,
            *shapes,
            shapes=True,
            optimize=optimizer,
        )
        elapsed = perf_counter() - started
        connection.send(
            (
                tuple(tuple(int(index) for index in step) for step in path),
                int(information.opt_cost),
                elapsed,
                None,
            )
        )
    except BaseException as error:
        connection.send((None, None, None, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _bounded_contract_path(
    equation: str,
    shapes: tuple[tuple[int, ...], ...],
    optimizer: str,
    timeout: float,
    /,
) -> tuple[tuple[tuple[int, ...], ...], int]:
    started = perf_counter()
    if 1 <= len(shapes) <= 2:
        # There is no path search with one or two operands. Retain opt_einsum's
        # canonical cost arithmetic without spawning/importing the package.
        fixed_path = ((0,),) if len(shapes) == 1 else ((0, 1),)
        _, information = oe.contract_path(
            equation, *shapes, shapes=True, optimize=list(fixed_path)
        )
        if perf_counter() - started > timeout:
            raise TimeoutError("Contraction planning exceeded maximum_planning_seconds.")
        return fixed_path, int(information.opt_cost)
    context = get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_bounded_contract_path_worker,
        args=(sender, equation, shapes, optimizer),
    )
    process.start()
    sender.close()
    try:
        # Bound interpreter startup as well as search; a dead readiness worker
        # cannot leave preparation waiting outside its admitted wall deadline.
        remaining = timeout - (perf_counter() - started)
        if remaining <= 0.0 or not receiver.poll(remaining):
            raise TimeoutError(
                "Contraction planner startup exceeded maximum_planning_seconds."
            )
        receiver.recv()
        remaining = timeout - (perf_counter() - started)
        if remaining <= 0.0 or not receiver.poll(remaining):
            process.terminate()
            raise TimeoutError("Contraction planning exceeded maximum_planning_seconds.")
        path, flops, elapsed, error = receiver.recv()
        if perf_counter() - started > timeout:
            raise TimeoutError("Contraction planning exceeded maximum_planning_seconds.")
    except EOFError:
        process.join(timeout=1.0)
        raise RuntimeError(
            f"Contraction planner worker exited with code {process.exitcode}."
        ) from None
    finally:
        receiver.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=1.0)
        if process.is_alive():
            process.kill()
            process.join()
    if error is not None:
        raise RuntimeError(f"Contraction planner failed: {error}")
    if elapsed > timeout:
        raise TimeoutError("Contraction planning exceeded maximum_planning_seconds.")
    return path, flops


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
    path_identity: str,
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
            "path_identity": path_identity,
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
    optimizer: str | None = None,
    dtype: str = "complex128",
    path: Sequence[Sequence[int]] | None = None,
    path_planner: ContractionPathPlanner | None = None,
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
    if sum(value is not None for value in (optimizer, path, path_planner)) > 1:
        raise ValueError(
            "path, path_planner, and an explicit native optimizer are mutually exclusive."
        )
    if path_planner is not None and not isinstance(path_planner, ContractionPathPlanner):
        raise TypeError("path_planner must be ContractionPathPlanner or None.")
    optimizer_ = "greedy" if optimizer is None else optimizer
    if optimizer_ not in ("greedy", "optimal"):
        raise ValueError("optimizer must be 'greedy' or 'optimal'.")
    imported_path = None if path is None else tuple(tuple(step) for step in path)
    path_identity = canonical_fingerprint(
        {
            "path": imported_path,
            "planner": None if path_planner is None else path_planner.planner_id,
        }
    )
    dtype_ = jnp.dtype(dtype).name
    if dtype_ not in ("float32", "float64", "complex64", "complex128"):
        raise TypeError(
            "Prepared contractions support real/complex 32/64-bit storage only."
        )
    contraction_dtype, accumulation_dtype = precision_.resolve_contraction(dtype_)
    precision_probe = jnp.empty((), dtype=accumulation_dtype)
    output_dtype = str(precision_.output(precision_probe).dtype)
    key = _plan_cache_key(
        structure, resources_, planner_, precision_, optimizer_, dtype_, path_identity
    )
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
    search_bound = (
        planner_search_state_bound(len(shapes), optimizer_)
        if imported_path is None and path_planner is None
        else 0
    )
    if search_bound > planner_.maximum_search_states:
        raise RuntimeError(
            "Contraction planner search exceeds maximum_search_states before planning."
        )

    equation = _equation(structure)
    if imported_path is not None:
        path_ = imported_path
        planner_flops = 0
        source = "imported"
        provenance = path_identity
    elif path_planner is not None:
        from ._cutensornet import plan_cutensornet_path

        path_, build = plan_cutensornet_path(structure, contraction_dtype, path_planner)
        planner_flops = 0
        source = "cutensornet"
        provenance = canonical_fingerprint(
            {"request": path_planner.planner_id, "build": build}
        )
    else:
        path_, planner_flops = _bounded_contract_path(
            equation,
            shapes,
            optimizer_,
            planner_.maximum_planning_seconds,
        )
        source = optimizer_
        provenance = canonical_fingerprint(
            {"native_optimizer": optimizer_, "planner": planner_.policy_id}
        )
        if path_ and len(path_[-1]) == 1:
            path_ = path_[:-1]
    schedule = build_contraction_schedule(
        structure,
        path_,
        dtype=contraction_dtype,
    )
    largest = max(
        (step.output_elements for step in schedule.steps), default=output_elements
    )
    flops = max(planner_flops, schedule.total_estimated_flops)
    storage_operand_bytes = operand_elements * precision_itemsize(dtype_)
    output_bytes = output_elements * precision_itemsize(output_dtype)
    workspace_bytes = (
        storage_operand_bytes
        + schedule.peak_live_elements * precision_itemsize(accumulation_dtype)
        + output_bytes
    )
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
        workspace_bytes,
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
            "path_provenance": provenance,
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
        source,
        dtype_,
        True,
        "exact ordinary product-sum contraction",
        plan_id,
        provenance,
    )
    if cache is not None:
        cache.store(key, plan)
    return plan


def _validate_operands(
    plan: ContractionPlan, operands: Sequence[ArrayLike], /
) -> tuple[Array, ...]:
    arrays = tuple(jnp.asarray(value) for value in operands)
    if len(arrays) != len(plan.structure.operands):
        raise ValueError("Operand count differs from the contraction plan.")
    for array, specification in zip(arrays, plan.structure.operands, strict=True):
        expected = tuple(leg.dimension for leg in specification.legs)
        if array.shape != expected:
            raise ValueError(
                f"Operand {specification.operand_id!r} expected shape {expected}; got {array.shape}."
            )
        if str(array.dtype) != plan.dtype:
            raise TypeError("Operand dtype differs from the contraction plan.")
    plan.precision.validate_storage(arrays)
    return arrays


def _validate_execution_devices(
    operands: tuple[Array, ...], execution: ContractionExecutionPolicy, /
) -> None:
    if execution.provider == "native":
        return
    from .._execution_device import device_platform

    if any(
        device_platform(device) != "cuda"
        for operand in operands
        for device in operand.devices()
    ):
        raise ValueError(
            f"Explicit {execution.provider} contraction requires CUDA-resident operands; no transfer or native fallback is implicit."
        )


def prepare_contraction(
    plan: ContractionPlan,
    operands: Sequence[ArrayLike],
    /,
    *,
    execution: ContractionExecutionPolicy | None = None,
) -> PreparedContraction:
    if not isinstance(plan, ContractionPlan):
        raise TypeError("plan must be ContractionPlan.")
    arrays = _validate_operands(plan, operands)
    if any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(arrays)):
        raise TypeError(
            "Host content-addressed contraction preparation cannot accept traced operands."
        )
    from ._cutensor import prepare_executor

    policy = ContractionExecutionPolicy() if execution is None else execution
    if not isinstance(policy, ContractionExecutionPolicy):
        raise TypeError("execution must be ContractionExecutionPolicy or None.")
    _validate_execution_devices(arrays, policy)
    executor = prepare_executor(plan, policy)
    operand_id = canonical_fingerprint(
        {
            "kind": "contraction-operands",
            "plan": plan.plan_id,
            "values": array_tree_fingerprint(arrays),
        }
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-contraction",
            "plan": plan.plan_id,
            "operands": operand_id,
            "numeric_version": 0,
            "execution": executor.lowering_id,
        }
    )
    return PreparedContraction(
        plan,
        arrays,
        jnp.asarray(0, dtype=jnp.int32),
        prepared_id,
        operand_id,
        executor,
    )


def refresh_contraction(
    prepared: PreparedContraction,
    operands: Sequence[ArrayLike],
    /,
) -> PreparedContraction:
    if not isinstance(prepared, PreparedContraction):
        raise TypeError("prepared must be PreparedContraction.")
    arrays = _validate_operands(prepared.plan, operands)
    if any(
        isinstance(leaf, Tracer)
        for leaf in jax.tree.leaves((arrays, prepared.numeric_version))
    ):
        raise TypeError(
            "Host content-addressed contraction refresh cannot accept traced operands or versions."
        )
    _validate_execution_devices(arrays, prepared.executor.policy)
    version = prepared.numeric_version + 1
    operand_id = canonical_fingerprint(
        {
            "kind": "contraction-operands",
            "plan": prepared.plan.plan_id,
            "values": array_tree_fingerprint(arrays),
        }
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-contraction",
            "plan": prepared.plan.plan_id,
            "operands": operand_id,
            "numeric_version": int(version),
            "execution": prepared.executor.lowering_id,
        }
    )
    return PreparedContraction(
        prepared.plan,
        arrays,
        version,
        prepared_id,
        operand_id,
        prepared.executor,
    )


def execute_schedule(
    schedule: ContractionSchedule,
    operands: Sequence[Array],
    /,
    *,
    accumulation_dtype: str | None = None,
) -> Array:
    """Replay one admitted schedule without path search or backend switching."""

    if len(operands) != len(schedule.initial_value_ids):
        raise ValueError("Schedule operand count does not match initial SSA values.")
    if any(len(step.input_value_ids) > 2 for step in schedule.steps):
        raise ValueError(
            "Native schedule execution admits unary/binary SSA steps only; multi-input steps are not silently split."
        )
    value_ids = list(schedule.initial_value_ids)
    values = list(operands)
    for step in schedule.steps:
        positions = tuple(value_ids.index(value_id) for value_id in step.input_value_ids)
        selected = tuple(values[position] for position in positions)
        result = jnp.einsum(
            step.equation,
            *selected,
            optimize=False,
            precision=jax.lax.Precision.HIGHEST,
            preferred_element_type=accumulation_dtype,
        )
        if step is not schedule.steps[-1]:
            result = result.astype(schedule.dtype)
        for position in sorted(positions, reverse=True):
            value_ids.pop(position)
            values.pop(position)
        value_ids.append(step.output_value_id)
        values.append(result)
    if value_ids != [schedule.output_value_id]:
        raise RuntimeError(
            "Schedule replay did not produce its declared output SSA value."
        )
    return jnp.asarray(values[0], dtype=accumulation_dtype)


def _execution_evidence(
    prepared: PreparedContraction, output: Array, /
) -> ContractionExecutionEvidence:
    plan = prepared.plan
    finite = jnp.all(jnp.isfinite(output))
    parent_precision = plan.precision.evidence_for(prepared.operands, output_value=output)
    child_resolution = PrecisionResolution(
        plan.precision.request,
        f"phydrax-{prepared.executor.policy.provider}-contraction",
        parent_precision.observed,
    )
    child_precision = PrecisionEvidenceEnvelope(
        child_resolution, parent_precision.observed
    )
    # prepared_id already binds the host-side numeric version; the traced
    # numeric_version array is carried as dynamic evidence below.
    replay_id = canonical_fingerprint(
        {
            "kind": "contraction-replay",
            "schedule": plan.schedule.schedule_id,
            "prepared": prepared.prepared_id,
            "operands": prepared.operand_id,
            "execution": prepared.executor.lowering_id,
        }
    )
    return ContractionExecutionEvidence(
        finite,
        finite,
        plan.precision.evidence_for(
            prepared.operands, output_value=output, children={"executor": child_precision}
        ),
        plan.structure.structure_id,
        plan.plan_id,
        plan.schedule.schedule_id,
        prepared.prepared_id,
        replay_id,
        True,
        plan.claim,
        plan.cost.peak_live_bytes,
        prepared.numeric_version,
        prepared.executor.lowering_id,
        prepared.executor.policy.provider,
        prepared.executor.external_workspace_bound_bytes,
        prepared.executor.external_cache_bound_bytes,
        prepared.executor.vendor_host_bound_bytes,
        prepared.executor.derivative_contract,
    )


def execute_contraction(prepared: PreparedContraction, /) -> ContractionResult:
    if not isinstance(prepared, PreparedContraction):
        raise TypeError("prepared must be PreparedContraction.")
    plan = prepared.plan
    operands = plan.precision.contraction(prepared.operands)
    from ._cutensor import execute_executor

    value = execute_executor(plan, prepared.executor, operands)
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
    compute_dtype, accumulation_dtype = plan.precision.resolve_contraction(plan.dtype)
    output_probe = plan.precision.output(jnp.empty((), dtype=accumulation_dtype))
    operand_bytes = sum(
        operand.size * precision_itemsize(compute_dtype) for operand in prepared.operands
    )
    output_bytes = plan.cost.output_elements * precision_itemsize(str(output_probe.dtype))
    reverse = reverse_schedule_evidence(
        plan.schedule,
        rematerialization=rematerialization,
        operand_bytes=operand_bytes,
        output_bytes=output_bytes,
        intermediate_dtype=accumulation_dtype,
    )
    reverse_peak = (
        reverse.reverse_peak_bytes
        + prepared.executor.normalization_bytes
        + prepared.executor.external_cache_bound_bytes
    )
    if reverse_peak > plan.resources.maximum_workspace_bytes:
        raise MemoryError(
            "Reverse contraction exceeds maximum_workspace_bytes before allocation."
        )

    def contraction(*operands: Array) -> Array:
        converted = plan.precision.contraction(operands)
        from ._cutensor import execute_executor

        return plan.precision.output(execute_executor(plan, prepared.executor, converted))

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
    execution: ContractionExecutionPolicy | None = None,
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
    return prepare_contraction(plan, tuple(arrays), execution=execution)


def prepare_mpo_inner_contraction(
    left: MatrixProductOperator,
    right: MatrixProductOperator,
    /,
    *,
    resources: ContractionResourcePolicy | None = None,
    execution: ContractionExecutionPolicy | None = None,
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
    return prepare_contraction(plan, tuple(arrays), execution=execution)


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
