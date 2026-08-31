#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..continuation import ParameterContinuationProblem
from ..linalg import ArraySpace
from ..nonlinear import (
    AbstractNonlinearMethod,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    prepare_nonlinear,
    PreparedNonlinearSolve,
    solve_prepared_nonlinear,
)
from ._dae import CircuitDAEDiagnostics, PreparedCircuitDAE


class CircuitOperatingPointPlan(StrictModule):
    prepared_dae: PreparedCircuitDAE
    problem: NonlinearSystemProblem
    plan_id: str = eqx.field(static=True)


class PreparedCircuitOperatingPoint(StrictModule):
    plan: CircuitOperatingPointPlan
    nonlinear: PreparedNonlinearSolve
    args: Any
    prepared_id: str = eqx.field(static=True)


class CircuitOperatingPointResult(StrictModule):
    state: Array
    nonlinear: NonlinearResult
    circuit_diagnostics: CircuitDAEDiagnostics
    prepared_id: str = eqx.field(static=True)


class _OperatingPointResidual(StrictModule):
    prepared_dae: PreparedCircuitDAE

    def __call__(self, state: Array, args: Any, /) -> Array:
        return self.prepared_dae.system.evaluate(
            jnp.asarray(0.0, dtype=state.dtype),
            state,
            jnp.zeros_like(state),
            args,
        )


class _SourceContinuationResidual(StrictModule):
    prepared_dae: PreparedCircuitDAE
    target_inputs: Any

    def __call__(self, state: Array, parameter: Array, args: Any, /) -> Array:
        del args
        if not isinstance(self.target_inputs, dict):
            raise TypeError("Source continuation requires a dictionary of scalar inputs.")
        scaled = {
            key: parameter * jnp.asarray(value)
            for key, value in self.target_inputs.items()
        }
        return self.prepared_dae.system.evaluate(
            jnp.asarray(0.0, dtype=state.dtype),
            state,
            jnp.zeros_like(state),
            {"inputs": scaled},
        )


def plan_circuit_operating_point(
    prepared_dae: PreparedCircuitDAE,
    /,
) -> CircuitOperatingPointPlan:
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    size = prepared_dae.plan.layout.size
    space = ArraySpace((size,), dtype=jnp.float64)
    problem = NonlinearSystemProblem(
        _OperatingPointResidual(prepared_dae),
        state_space=space,
        residual_space=space,
        problem_id=f"{prepared_dae.plan.circuit.circuit_id}/operating-point",
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "circuit-operating-point-plan",
            "dae": prepared_dae.prepared_id,
        }
    )
    return CircuitOperatingPointPlan(prepared_dae, problem, plan_id)


def prepare_circuit_operating_point(
    prepared_dae: PreparedCircuitDAE,
    initial_state: ArrayLike,
    /,
    *,
    args: Any = None,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
) -> PreparedCircuitOperatingPoint:
    plan = plan_circuit_operating_point(prepared_dae)
    initial = jnp.asarray(initial_state, dtype=float)
    if initial.shape != (prepared_dae.plan.layout.size,):
        raise ValueError("Operating-point initial state has the wrong shape.")
    nonlinear = prepare_nonlinear(
        plan.problem,
        initial,
        args=args,
        method=method,
        termination=termination,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-circuit-operating-point",
            "plan": plan.plan_id,
            "nonlinear": {
                "method": nonlinear.provenance.method_id,
                "linear": nonlinear.provenance.linear_plan_id,
            },
        }
    )
    return PreparedCircuitOperatingPoint(plan, nonlinear, args, prepared_id)


def solve_circuit_operating_point(
    prepared: PreparedCircuitOperatingPoint,
    /,
) -> CircuitOperatingPointResult:
    if not isinstance(prepared, PreparedCircuitOperatingPoint):
        raise TypeError("prepared must be PreparedCircuitOperatingPoint.")
    result = solve_prepared_nonlinear(prepared.nonlinear)
    state = jnp.asarray(result.state)
    diagnostics = prepared.plan.prepared_dae.diagnostics(
        jnp.asarray(0.0), state, jnp.zeros_like(state), prepared.args
    )
    return CircuitOperatingPointResult(state, result, diagnostics, prepared.prepared_id)


def circuit_source_continuation_problem(
    prepared_dae: PreparedCircuitDAE,
    target_inputs: dict[str, ArrayLike],
    /,
    *,
    problem_id: str | None = None,
) -> ParameterContinuationProblem:
    if not isinstance(prepared_dae, PreparedCircuitDAE):
        raise TypeError("prepared_dae must be PreparedCircuitDAE.")
    if not target_inputs or any(not str(key) for key in target_inputs):
        raise ValueError("target_inputs must be a nonempty mapping with nonempty keys.")
    values = {str(key): jnp.asarray(value) for key, value in target_inputs.items()}
    if any(value.shape != () for value in values.values()):
        raise ValueError("Source continuation inputs must be scalars.")
    size = prepared_dae.plan.layout.size
    space = ArraySpace((size,), dtype=jnp.float64)
    identifier = (
        f"{prepared_dae.plan.circuit.circuit_id}/source-continuation"
        if problem_id is None
        else str(problem_id)
    )
    return ParameterContinuationProblem(
        _SourceContinuationResidual(prepared_dae, values),
        parameter_lower=0.0,
        parameter_upper=1.0,
        state_space=space,
        residual_space=space,
        problem_id=identifier,
    )


def operating_point_jacobian(
    prepared: PreparedCircuitOperatingPoint,
    state: ArrayLike,
    /,
) -> Array:
    if not isinstance(prepared, PreparedCircuitOperatingPoint):
        raise TypeError("prepared must be PreparedCircuitOperatingPoint.")
    value = jnp.asarray(state)
    if value.shape != (prepared.plan.prepared_dae.plan.layout.size,):
        raise ValueError("Operating-point state has the wrong shape.")
    return jax.jacfwd(
        lambda current: prepared.plan.problem.evaluate(current, prepared.args)[0]
    )(value)


__all__ = [
    "CircuitOperatingPointPlan",
    "CircuitOperatingPointResult",
    "PreparedCircuitOperatingPoint",
    "circuit_source_continuation_problem",
    "operating_point_jacobian",
    "plan_circuit_operating_point",
    "prepare_circuit_operating_point",
    "solve_circuit_operating_point",
]
