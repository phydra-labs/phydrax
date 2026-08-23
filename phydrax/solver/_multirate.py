#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jax import lax
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._numerics._ssp_runge_kutta import ssprk33_step
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    DiscretizationBundle,
    FDExecutionPrecisionPolicy,
    FiniteVolumePrecisionPolicy,
)
from ..dynamics import TimeGrid
from ._differential import DifferentialSolution, DifferentialVectorField
from ._state_partition import StatePartition
from ._temporal_method import (
    configuration_id,
    TemporalMethodCapabilities,
    TemporalSolveEvidence,
)
from ._temporal_precision import TemporalPrecisionPolicy


class PartitionedDifferentialProblem(StrictModule):
    """Explicit slow/fast vector fields over disjoint state partitions."""

    slow_drift: DifferentialVectorField
    fast_drift: DifferentialVectorField
    initial_state: Array
    t0: Array
    t1: Array
    args: Any
    partition: StatePartition
    discretization_bundle: DiscretizationBundle | None
    problem_id: str = eqx.field(static=True)
    discretization_bundle_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        slow_drift: DifferentialVectorField,
        fast_drift: DifferentialVectorField,
        initial_state: ArrayLike,
        /,
        *,
        t0: ArrayLike,
        t1: ArrayLike,
        partition: StatePartition,
        args: Any = None,
        discretization_bundle: DiscretizationBundle | None = None,
        problem_id: str | None = None,
    ):
        if not callable(slow_drift) or not callable(fast_drift):
            raise TypeError("Partitioned drifts must be callable.")
        if not isinstance(partition, StatePartition) or len(partition.names) != 2:
            raise ValueError("Partitioned differential problems require two partitions.")
        state = jnp.asarray(initial_state)
        if state.shape != partition.state_shape:
            raise ValueError("Initial state must match the state partition.")
        start = jnp.asarray(t0, dtype=float)
        end = jnp.asarray(t1, dtype=float)
        if start.shape != () or end.shape != ():
            raise ValueError("Partitioned time bounds must be scalar.")
        end = eqx.error_if(
            end,
            ~(jnp.isfinite(start) & jnp.isfinite(end) & (end > start)),
            "Partitioned time bounds must be finite and increasing.",
        )
        for function, name in (
            (slow_drift, partition.names[0]),
            (fast_drift, partition.names[1]),
        ):
            value = jnp.asarray(function(start, state, args))
            if value.shape != state.shape:
                raise ValueError("Partitioned drifts must preserve state shape.")
            outside = jnp.where(partition.mask(name), 0, value)
            state = eqx.error_if(
                state,
                jnp.any(outside != 0),
                f"Drift {name!r} writes outside its declared partition.",
            )
        if discretization_bundle is not None and not isinstance(
            discretization_bundle, DiscretizationBundle
        ):
            raise TypeError("discretization_bundle must be DiscretizationBundle or None.")
        bundle_id = (
            None if discretization_bundle is None else discretization_bundle.bundle_id
        )
        payload = {
            "partition_id": partition.partition_id,
            "state_shape": list(state.shape),
            "state_dtype": str(state.dtype),
            "bundle_id": bundle_id,
        }
        identifier = (
            f"partitioned-problem:{canonical_fingerprint(payload)}"
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty or None.")
        self.slow_drift = slow_drift
        self.fast_drift = fast_drift
        self.initial_state = state
        self.t0 = start
        self.t1 = end
        self.args = args
        self.partition = partition
        self.discretization_bundle = discretization_bundle
        self.problem_id = identifier
        self.discretization_bundle_id = bundle_id

    def drift(self, time: Array, state: Array, args: Any, /) -> Array:
        slow = self.partition.project(
            self.partition.names[0], self.slow_drift(time, state, args)
        )
        fast = self.partition.project(
            self.partition.names[1], self.fast_drift(time, state, args)
        )
        return slow + fast


class MultiratePartitionedRK(StrictModule, NonTrainableState):
    """Fixed-ratio partitioned RK2 or RK3 subcycling method."""

    capabilities: TemporalMethodCapabilities
    order: int = eqx.field(static=True)
    refinement_ratio: int = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, order: int = 3, /, *, refinement_ratio: int = 2):
        order_ = int(order)
        ratio = int(refinement_ratio)
        if order_ not in (2, 3):
            raise ValueError("Multirate partitioned RK order must be two or three.")
        if ratio <= 1:
            raise ValueError("refinement_ratio must exceed one.")
        identifier = f"temporal:mprk:{order_}:ratio-{ratio}"
        self.order = order_
        self.refinement_ratio = ratio
        self.method_id = identifier
        self.capabilities = TemporalMethodCapabilities(
            equation_forms=("partitioned",),
            method_class="multirate-rk",
            order=order_,
            adaptive=False,
            history_depth=1,
            causal_stage_extent=1.0,
            verified=True,
            method_id=identifier,
        )


def _rk2_step(function, time, state, step_size, args, precision):
    staged_state = precision.stage(state)
    step = precision.coefficient(jnp.asarray(step_size, dtype=staged_state.real.dtype))
    first = precision.stage(function(time, staged_state, args))
    predictor = precision.stage(
        precision.accumulation(staged_state) + precision.accumulation(step * first)
    )
    second = precision.stage(function(time + step, predictor, args))
    result = precision.accumulation(staged_state) + precision.accumulation(
        0.5 * step * precision.accumulation(first + second)
    )
    return jnp.asarray(result, dtype=state.dtype)


def solve_multirate(
    problem: PartitionedDifferentialProblem,
    time_grid: TimeGrid,
    /,
    *,
    method: MultiratePartitionedRK | None = None,
    args: Any = None,
    precision: TemporalPrecisionPolicy | None = None,
) -> DifferentialSolution:
    """Integrate a partitioned problem with fixed-ratio synchronized subcycling."""
    if not isinstance(problem, PartitionedDifferentialProblem):
        raise TypeError("problem must be PartitionedDifferentialProblem.")
    if not isinstance(time_grid, TimeGrid):
        raise TypeError("time_grid must be TimeGrid.")
    times = lax.stop_gradient(time_grid.times)
    times = eqx.error_if(
        times,
        ~jnp.isclose(times[0], problem.t0) | ~jnp.isclose(times[-1], problem.t1),
        "TimeGrid endpoints must match the partitioned problem.",
    )
    selected = MultiratePartitionedRK() if method is None else method
    if not isinstance(selected, MultiratePartitionedRK):
        raise TypeError("method must be MultiratePartitionedRK or None.")
    precision_ = TemporalPrecisionPolicy() if precision is None else precision
    if not isinstance(precision_, TemporalPrecisionPolicy):
        raise TypeError("precision must be a TemporalPrecisionPolicy or None.")
    precision_.validate_state(problem.initial_state)
    runtime_args = problem.args if args is None else args

    def macro_step(state, values):
        time, width = values
        micro_width = width / selected.refinement_ratio

        def micro_step(index, current):
            micro_time = time + index * micro_width
            if selected.order == 2:
                return _rk2_step(
                    problem.drift,
                    micro_time,
                    current,
                    micro_width,
                    runtime_args,
                    precision_,
                )
            return ssprk33_step(
                problem.drift,
                micro_time,
                current,
                micro_width,
                runtime_args,
                precision=precision_,
            )

        next_state = lax.fori_loop(0, selected.refinement_ratio, micro_step, state)
        return next_state, next_state

    _, stepped = lax.scan(
        macro_step,
        problem.initial_state,
        (times[:-1], jnp.diff(times)),
    )
    states = jnp.concatenate((problem.initial_state[None, ...], stepped), axis=0)
    state_axes = tuple(range(1, states.ndim))
    valid = (
        jnp.all(jnp.isfinite(states), axis=state_axes)
        if state_axes
        else jnp.isfinite(states)
    )
    successful = jnp.all(valid)
    evidence = TemporalSolveEvidence(
        selected.capabilities,
        equation_form="partitioned",
        backend_id="backend:phydrax:multirate-rk",
        configuration_id=configuration_id(
            (
                selected,
                problem.partition.partition_id,
                precision_.policy_id,
                time_grid.time_id,
            ),
            prefix="temporal-configuration",
        ),
        controller_id=f"controller:fixed-grid:{time_grid.time_id}",
        adjoint_id="adjoint:jax-explicit-scan",
        event_id=None,
        adaptive=False,
        dense=False,
        maximum_steps=time_grid.num_steps * selected.refinement_ratio,
        precision_evidence=precision_.evidence_for(problem.initial_state, times),
    )
    output_states = precision_.output(states)
    return DifferentialSolution(
        times=times,
        states=output_states,
        valid=valid,
        backend_result=jnp.where(successful, 0, 1),
        stats={
            "macro_steps": jnp.asarray(time_grid.num_steps, dtype=jnp.int32),
            "micro_steps": jnp.asarray(
                time_grid.num_steps * selected.refinement_ratio, dtype=jnp.int32
            ),
            "refinement_ratio": selected.refinement_ratio,
        },
        solver_name=type(selected).__name__,
        interpretation="ito",
        solver_id=selected.method_id,
        resolved_method=f"partitioned-rk{selected.order}:ratio-{selected.refinement_ratio}",
        discretization_bundle=problem.discretization_bundle,
        backend_successful=successful,
        temporal_evidence=evidence,
        problem_id=problem.problem_id,
    )


def multirate_amr_subcycling_plan(
    method: MultiratePartitionedRK,
    /,
    *,
    precision: FDExecutionPrecisionPolicy | FiniteVolumePrecisionPolicy | None = None,
):
    """Bind one multirate ratio, method identity, and spatial precision to AMR."""
    from ..discretization.amr import ConservativeAMRSubcyclingPlan

    if not isinstance(method, MultiratePartitionedRK):
        raise TypeError("method must be MultiratePartitionedRK.")
    if precision is not None and not isinstance(
        precision,
        (FDExecutionPrecisionPolicy, FiniteVolumePrecisionPolicy),
    ):
        raise TypeError(
            "precision must be an FDExecutionPrecisionPolicy, "
            "FiniteVolumePrecisionPolicy, or None."
        )
    return ConservativeAMRSubcyclingPlan(
        method.refinement_ratio,
        temporal_method_id=method.method_id,
        precision=precision,
    )


__all__ = [
    "MultiratePartitionedRK",
    "multirate_amr_subcycling_plan",
    "PartitionedDifferentialProblem",
    "solve_multirate",
]
