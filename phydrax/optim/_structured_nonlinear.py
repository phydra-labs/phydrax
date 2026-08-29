#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..sparse import SparseCoordinateOperator, SparseDerivativePlan
from ._iterative import ConstrainedOptimalityCertificate


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _real_vector(value: ArrayLike, size: int | None, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{owner} must be rank one.")
    if size is not None and array.shape != (size,):
        raise ValueError(f"{owner} must have shape {(size,)}; got {array.shape}.")
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    return array if jnp.issubdtype(array.dtype, jnp.inexact) else array.astype(float)


def _maximum(values: Array, /) -> Array:
    return jnp.max(values, initial=jnp.asarray(0.0, dtype=values.dtype))


class StructuredNonlinearWarmStart(StrictModule):
    """Primal and bound-form dual values for one structured nonlinear program."""

    primal: Array
    constraint_multipliers: Array
    lower_bound_multipliers: Array
    upper_bound_multipliers: Array
    structure_id: str = eqx.field(static=True)
    source_result_id: str | None = eqx.field(static=True)
    source_program_id: str | None = eqx.field(static=True)
    source_backend: str | None = eqx.field(static=True)
    warm_start_id: str = eqx.field(static=True)

    def __init__(
        self,
        primal: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        /,
        *,
        structure_id: str,
        source_result_id: str | None = None,
        source_program_id: str | None = None,
        source_backend: str | None = None,
        warm_start_id: str | None = None,
    ):
        primal_ = _real_vector(primal, None, "warm-start primal")
        constraints = _real_vector(
            constraint_multipliers, None, "warm-start constraint multipliers"
        )
        lower = _real_vector(
            lower_bound_multipliers, int(primal_.size), "warm-start lower multipliers"
        )
        upper = _real_vector(
            upper_bound_multipliers, int(primal_.size), "warm-start upper multipliers"
        )
        primal_ = eqx.error_if(
            primal_,
            jnp.any(~jnp.isfinite(primal_)),
            "warm-start primal must be finite.",
        )
        constraints = eqx.error_if(
            constraints,
            jnp.any(~jnp.isfinite(constraints)),
            "warm-start constraint multipliers must be finite.",
        )
        lower = eqx.error_if(
            lower,
            jnp.any(~jnp.isfinite(lower)) | jnp.any(lower < 0.0),
            "warm-start lower multipliers must be finite and non-negative.",
        )
        upper = eqx.error_if(
            upper,
            jnp.any(~jnp.isfinite(upper)) | jnp.any(upper < 0.0),
            "warm-start upper multipliers must be finite and non-negative.",
        )
        self.primal = primal_
        self.constraint_multipliers = constraints
        self.lower_bound_multipliers = lower
        self.upper_bound_multipliers = upper
        self.structure_id = _identifier(structure_id, "structure_id")
        self.source_result_id = (
            None
            if source_result_id is None
            else _identifier(source_result_id, "source_result_id")
        )
        self.source_program_id = (
            None
            if source_program_id is None
            else _identifier(source_program_id, "source_program_id")
        )
        self.source_backend = (
            None
            if source_backend is None
            else _identifier(source_backend, "source_backend")
        )
        self.warm_start_id = (
            canonical_fingerprint(
                {
                    "kind": "structured-nonlinear-warm-start",
                    "structure": self.structure_id,
                    "source_result": self.source_result_id,
                    "source_program": self.source_program_id,
                    "source_backend": self.source_backend,
                    "primal_size": int(primal_.size),
                    "constraint_size": int(constraints.size),
                }
            )
            if warm_start_id is None
            else _identifier(warm_start_id, "warm_start_id")
        )


class StructuredNonlinearEvaluation(StrictModule):
    """Objective, constraints, gradient, and sparse Jacobian at one point."""

    coordinates: Array
    objective: Array
    gradient: Array
    constraints: Array
    jacobian: SparseCoordinateOperator
    finite: Array


class StructuredNonlinearProgram(StrictModule):
    """Bound-form nonlinear program with reusable exact sparse derivatives."""

    objective: Callable[[Array, Any], ArrayLike]
    constraints: Callable[[Array, Any], ArrayLike]
    jacobian_plan: SparseDerivativePlan
    hessian_plan: SparseDerivativePlan | None
    variable_lower: Array
    variable_upper: Array
    constraint_lower: Array
    constraint_upper: Array
    equality_indices: Array
    lower_indices: Array
    upper_indices: Array
    constraint_sources: tuple[str, ...] = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    program_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        objective: Callable[[Array, Any], ArrayLike],
        constraints: Callable[[Array, Any], ArrayLike],
        jacobian_plan: SparseDerivativePlan,
        /,
        *,
        variable_lower: ArrayLike,
        variable_upper: ArrayLike,
        constraint_lower: ArrayLike,
        constraint_upper: ArrayLike,
        constraint_sources: Sequence[str],
        hessian_plan: SparseDerivativePlan | None = None,
        program_id: str,
        structure_id: str,
    ):
        if not callable(objective) or not callable(constraints):
            raise TypeError("Structured nonlinear objective and constraints must be callable.")
        if not isinstance(jacobian_plan, SparseDerivativePlan):
            raise TypeError("jacobian_plan must be a SparseDerivativePlan.")
        if hessian_plan is not None and not isinstance(hessian_plan, SparseDerivativePlan):
            raise TypeError("hessian_plan must be a SparseDerivativePlan or None.")
        variables = jacobian_plan.pattern.source_size
        constraint_count = jacobian_plan.pattern.target_size
        lower_x = _real_vector(variable_lower, variables, "variable_lower")
        upper_x = _real_vector(variable_upper, variables, "variable_upper")
        lower_c = _real_vector(constraint_lower, constraint_count, "constraint_lower")
        upper_c = _real_vector(constraint_upper, constraint_count, "constraint_upper")
        if bool(np.any(np.asarray(lower_x) > np.asarray(upper_x))):
            raise ValueError("Variable lower bounds must not exceed upper bounds.")
        if bool(np.any(np.asarray(lower_c) > np.asarray(upper_c))):
            raise ValueError("Constraint lower bounds must not exceed upper bounds.")
        if hessian_plan is not None and hessian_plan.pattern.shape != (
            variables,
            variables,
        ):
            raise ValueError("hessian_plan must have one square variable-space pattern.")
        sources = tuple(str(source) for source in constraint_sources)
        if len(sources) != constraint_count or any(not source for source in sources):
            raise ValueError("constraint_sources must identify every constraint scalar.")
        lower_host = np.asarray(lower_c)
        upper_host = np.asarray(upper_c)
        equality = np.flatnonzero(
            np.isfinite(lower_host)
            & np.isfinite(upper_host)
            & (lower_host == upper_host)
        )
        lower = np.flatnonzero(np.isfinite(lower_host) & (lower_host != upper_host))
        upper = np.flatnonzero(np.isfinite(upper_host) & (lower_host != upper_host))
        self.objective = objective
        self.constraints = constraints
        self.jacobian_plan = jacobian_plan
        self.hessian_plan = hessian_plan
        self.variable_lower = lower_x
        self.variable_upper = upper_x
        self.constraint_lower = lower_c
        self.constraint_upper = upper_c
        self.equality_indices = jnp.asarray(equality, dtype=jnp.int32)
        self.lower_indices = jnp.asarray(lower, dtype=jnp.int32)
        self.upper_indices = jnp.asarray(upper, dtype=jnp.int32)
        self.constraint_sources = sources
        self.num_variables = variables
        self.num_constraints = constraint_count
        self.program_id = _identifier(program_id, "program_id")
        self.structure_id = _identifier(structure_id, "structure_id")

    def validate_coordinates(self, coordinates: ArrayLike, /) -> Array:
        return _real_vector(coordinates, self.num_variables, "coordinates")

    def evaluate(self, coordinates: ArrayLike, args: Any = None, /):
        point = self.validate_coordinates(coordinates)

        def scalar(value):
            output = jnp.asarray(self.objective(value, args))
            if output.shape != () or not jnp.issubdtype(output.dtype, jnp.floating):
                raise TypeError("Structured nonlinear objective must return one real scalar.")
            return output

        objective, gradient = jax.value_and_grad(scalar)(point)
        constraints = _real_vector(
            self.constraints(point, args), self.num_constraints, "constraints"
        )
        jacobian = self.jacobian_plan.operator(point, args)
        finite = (
            jnp.isfinite(objective)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(constraints))
            & jnp.all(jnp.isfinite(jacobian.coefficients))
        )
        return StructuredNonlinearEvaluation(
            point,
            objective,
            gradient,
            constraints,
            jacobian,
            finite,
        )

    def hessian_operator(
        self,
        coordinates: ArrayLike,
        constraint_multipliers: ArrayLike,
        objective_factor: ArrayLike,
        args: Any = None,
        /,
    ) -> SparseCoordinateOperator:
        if self.hessian_plan is None:
            raise ValueError("This structured nonlinear program has no exact Hessian plan.")
        point = self.validate_coordinates(coordinates)
        multipliers = _real_vector(
            constraint_multipliers,
            self.num_constraints,
            "constraint_multipliers",
        )
        factor = jnp.asarray(objective_factor, dtype=point.dtype)
        if factor.shape != ():
            raise ValueError("objective_factor must be scalar.")
        return self.hessian_plan.operator(point, (args, factor, multipliers))

    def warm_start(
        self,
        primal: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        /,
        *,
        source_result_id: str | None = None,
        source_backend: str | None = None,
    ) -> StructuredNonlinearWarmStart:
        warm = StructuredNonlinearWarmStart(
            primal,
            constraint_multipliers,
            lower_bound_multipliers,
            upper_bound_multipliers,
            structure_id=self.structure_id,
            source_result_id=source_result_id,
            source_program_id=self.program_id,
            source_backend=source_backend,
        )
        if warm.constraint_multipliers.shape != (self.num_constraints,):
            raise ValueError(
                "warm-start constraint multipliers do not match the program."
            )
        return warm

    def certificate(
        self,
        coordinates: ArrayLike,
        constraint_multipliers: ArrayLike,
        lower_bound_multipliers: ArrayLike,
        upper_bound_multipliers: ArrayLike,
        args: Any = None,
        /,
        *,
        active_tolerance: float,
    ) -> ConstrainedOptimalityCertificate:
        evaluation = self.evaluate(coordinates, args)
        multipliers = _real_vector(
            constraint_multipliers,
            self.num_constraints,
            "constraint_multipliers",
        )
        lower_dual = _real_vector(
            lower_bound_multipliers,
            self.num_variables,
            "lower_bound_multipliers",
        )
        upper_dual = _real_vector(
            upper_bound_multipliers,
            self.num_variables,
            "upper_bound_multipliers",
        )
        equality = multipliers[self.equality_indices]
        lower_constraint_dual = -multipliers[self.lower_indices]
        upper_constraint_dual = multipliers[self.upper_indices]
        inequality = jnp.concatenate(
            (lower_constraint_dual, upper_constraint_dual)
        )
        values = evaluation.constraints
        equality_values = values[self.equality_indices]
        equality_targets = self.constraint_lower[self.equality_indices]
        lower_slacks = (
            values[self.lower_indices] - self.constraint_lower[self.lower_indices]
        )
        upper_slacks = (
            self.constraint_upper[self.upper_indices] - values[self.upper_indices]
        )
        slacks = jnp.concatenate((lower_slacks, upper_slacks))
        stationarity = (
            evaluation.gradient
            + evaluation.jacobian.transpose_mv(multipliers)
            - lower_dual
            + upper_dual
        )
        lower_x_slack = evaluation.coordinates - self.variable_lower
        upper_x_slack = self.variable_upper - evaluation.coordinates
        lower_x_finite = jnp.isfinite(self.variable_lower)
        upper_x_finite = jnp.isfinite(self.variable_upper)
        primal = _maximum(jnp.abs(equality_values - equality_targets))
        primal = jnp.maximum(primal, _maximum(jnp.maximum(-lower_slacks, 0.0)))
        primal = jnp.maximum(primal, _maximum(jnp.maximum(-upper_slacks, 0.0)))
        primal = jnp.maximum(
            primal,
            _maximum(jnp.where(lower_x_finite, jnp.maximum(-lower_x_slack, 0.0), 0.0)),
        )
        primal = jnp.maximum(
            primal,
            _maximum(jnp.where(upper_x_finite, jnp.maximum(-upper_x_slack, 0.0), 0.0)),
        )
        dual = _maximum(jnp.maximum(-inequality, 0.0))
        dual = jnp.maximum(dual, _maximum(jnp.maximum(-lower_dual, 0.0)))
        dual = jnp.maximum(dual, _maximum(jnp.maximum(-upper_dual, 0.0)))
        complementarity = _maximum(jnp.abs(inequality * slacks))
        complementarity = jnp.maximum(
            complementarity,
            _maximum(
                jnp.where(lower_x_finite, jnp.abs(lower_dual * lower_x_slack), 0.0)
            ),
        )
        complementarity = jnp.maximum(
            complementarity,
            _maximum(
                jnp.where(upper_x_finite, jnp.abs(upper_dual * upper_x_slack), 0.0)
            ),
        )
        tolerance = float(active_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("active_tolerance must be finite and non-negative.")
        active = slacks <= tolerance
        equality_sources = tuple(
            self.constraint_sources[int(index)]
            for index in np.asarray(self.equality_indices)
        )
        inequality_sources = tuple(
            self.constraint_sources[int(index)]
            for index in np.concatenate(
                (np.asarray(self.lower_indices), np.asarray(self.upper_indices))
            )
        )
        return ConstrainedOptimalityCertificate(
            equality_multipliers=equality,
            inequality_multipliers=inequality,
            slacks=slacks,
            active_mask=active,
            stationarity_residual=stationarity,
            primal_feasibility=primal,
            dual_feasibility=dual,
            complementarity=complementarity,
            equality_sources=equality_sources,
            inequality_sources=inequality_sources,
        )


__all__ = [
    "StructuredNonlinearEvaluation",
    "StructuredNonlinearProgram",
    "StructuredNonlinearWarmStart",
]
