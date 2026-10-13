#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import AbstractLinearOperator, JacobianLinearOperator, prepare_linearization
from ._iterative import MinimizationProblem
from ._nonlinear_constraints import (
    _constraint_layout,
    _ConstraintLayout,
    _flat_constraint_values,
)


class ConstraintScalingPolicy(StrictModule):
    automatic: bool = eqx.field(static=True)
    objective_floor: float = eqx.field(static=True)
    constraint_floor: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        automatic: bool = True,
        objective_floor: float = 1.0,
        constraint_floor: float = 1.0,
    ) -> None:
        values = (float(objective_floor), float(constraint_floor))
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Constraint scaling floors must be finite and positive.")
        self.automatic = bool(automatic)
        self.objective_floor, self.constraint_floor = values


class ConstrainedModelEvaluation(StrictModule):
    parameters: PyTree[Array]
    coordinates: Array
    objective: Array
    scaled_objective: Array
    gradient: Array
    scaled_gradient: Array
    raw_constraints: Array
    equalities: Array
    lower_slacks: Array
    upper_slacks: Array
    inequality_slacks: Array
    constraint_jacobian: Array
    scaled_constraint_jacobian: Array
    primal_feasibility: Array
    finite: Array


class ConstrainedModelValues(StrictModule):
    """Objective and feasibility values without derivative work."""

    parameters: PyTree[Array]
    coordinates: Array
    objective: Array
    scaled_objective: Array
    raw_constraints: Array
    equalities: Array
    lower_slacks: Array
    upper_slacks: Array
    inequality_slacks: Array
    primal_feasibility: Array
    finite: Array


class ConstrainedModelActions(StrictModule):
    """Value/gradient model retaining only constraint derivative actions."""

    values: ConstrainedModelValues
    gradient: Array
    constraint_operator: AbstractLinearOperator


class PreparedConstrainedModel(StrictModule):
    """One canonical objective/constraint ordering and fixed numerical scaling."""

    problem: MinimizationProblem
    lower: Array
    upper: Array
    equality_indices: Array
    lower_indices: Array
    upper_indices: Array
    equality_sources: tuple[str, ...] = eqx.field(static=True)
    inequality_sources: tuple[str, ...] = eqx.field(static=True)
    objective_scale: Array
    constraint_scale: Array
    template_coordinates: Array
    unflatten: Any
    model_id: str = eqx.field(static=True)

    def evaluate_values(
        self, parameters: PyTree[Any], args: Any = None, /
    ) -> ConstrainedModelValues:
        coordinates, _ = ravel_pytree(parameters)
        objective = self.problem.value(parameters, args)[0]
        raw = _flat_constraint_values(self.problem, parameters, args)
        return self._values(parameters, coordinates, objective, raw)

    def _values(
        self,
        parameters: PyTree[Any],
        coordinates: Array,
        objective: Array,
        raw: Array,
        /,
    ) -> ConstrainedModelValues:
        equalities = raw[self.equality_indices] - self.lower[self.equality_indices]
        lower_slacks = raw[self.lower_indices] - self.lower[self.lower_indices]
        upper_slacks = self.upper[self.upper_indices] - raw[self.upper_indices]
        inequality_slacks = jnp.concatenate([lower_slacks, upper_slacks])
        primal = jnp.maximum(
            jnp.max(jnp.abs(equalities), initial=0.0),
            jnp.maximum(
                jnp.max(jnp.maximum(-lower_slacks, 0.0), initial=0.0),
                jnp.max(jnp.maximum(-upper_slacks, 0.0), initial=0.0),
            ),
        )
        return ConstrainedModelValues(
            parameters,
            coordinates,
            objective,
            objective / self.objective_scale,
            raw,
            equalities,
            lower_slacks,
            upper_slacks,
            inequality_slacks,
            primal,
            jnp.isfinite(objective) & jnp.all(jnp.isfinite(raw)),
        )

    def _evaluation(
        self, values: ConstrainedModelValues, gradient: Array, jacobian: Array, /
    ) -> ConstrainedModelEvaluation:
        return ConstrainedModelEvaluation(
            values.parameters,
            values.coordinates,
            values.objective,
            values.scaled_objective,
            gradient,
            gradient / self.objective_scale,
            values.raw_constraints,
            values.equalities,
            values.lower_slacks,
            values.upper_slacks,
            values.inequality_slacks,
            jacobian,
            jacobian / self.constraint_scale[:, None],
            values.primal_feasibility,
            values.finite
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(jacobian)),
        )

    def evaluate_derivatives(
        self, values: ConstrainedModelValues, args: Any = None, /
    ) -> ConstrainedModelEvaluation:
        gradient = jax.grad(
            lambda value: self.problem.value(self.unflatten(value), args)[0]
        )(values.coordinates)
        jacobian = jax.jacfwd(
            lambda value: _flat_constraint_values(
                self.problem, self.unflatten(value), args
            )
        )(values.coordinates)
        return self._evaluation(values, gradient, jacobian)

    def evaluate(
        self, parameters: PyTree[Any], args: Any = None, /
    ) -> ConstrainedModelEvaluation:
        coordinates, _ = ravel_pytree(parameters)
        objective, gradient = jax.value_and_grad(
            lambda value: self.problem.value(self.unflatten(value), args)[0]
        )(coordinates)
        raw = _flat_constraint_values(self.problem, parameters, args)
        jacobian = jax.jacfwd(
            lambda value: _flat_constraint_values(
                self.problem, self.unflatten(value), args
            )
        )(coordinates)
        return self._evaluation(
            self._values(parameters, coordinates, objective, raw), gradient, jacobian
        )

    def evaluate_actions(
        self, parameters: PyTree[Any], args: Any = None, /
    ) -> ConstrainedModelActions:
        coordinates, _ = ravel_pytree(parameters)
        objective, gradient = jax.value_and_grad(
            lambda value: self.problem.value(self.unflatten(value), args)[0]
        )(coordinates)
        derivative = prepare_linearization(
            lambda value: _flat_constraint_values(
                self.problem, self.unflatten(value), args
            ),
            coordinates,
        )
        values = self._values(parameters, coordinates, objective, derivative.primal)
        return ConstrainedModelActions(
            values, gradient, JacobianLinearOperator(derivative)
        )

    def lagrangian_hessian(
        self,
        parameters: PyTree[Any],
        equality_multipliers: Any,
        lower_multipliers: Any,
        upper_multipliers: Any,
        args: Any = None,
        /,
    ) -> Array:
        coordinates, _ = ravel_pytree(parameters)
        equality = jnp.asarray(equality_multipliers)
        lower = jnp.asarray(lower_multipliers)
        upper = jnp.asarray(upper_multipliers)

        def lagrangian(value: Array) -> Array:
            point = self.unflatten(value)
            objective = self.problem.value(point, args)[0]
            constraints = _flat_constraint_values(self.problem, point, args)
            equality_term = jnp.vdot(
                equality,
                constraints[self.equality_indices] - self.lower[self.equality_indices],
            )
            lower_term = jnp.vdot(
                lower,
                constraints[self.lower_indices] - self.lower[self.lower_indices],
            )
            upper_term = jnp.vdot(
                upper,
                self.upper[self.upper_indices] - constraints[self.upper_indices],
            )
            return jnp.real(objective + equality_term - lower_term - upper_term)

        return jax.hessian(lagrangian)(coordinates)


def prepare_constrained_model_evaluation(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
    scaling: ConstraintScalingPolicy | None = None,
) -> tuple[PreparedConstrainedModel, ConstrainedModelEvaluation]:
    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be MinimizationProblem.")
    if not problem.constraints and problem.bounds is None:
        raise ValueError("A constrained model requires constraints or bounds.")
    policy = ConstraintScalingPolicy() if scaling is None else scaling
    if not isinstance(policy, ConstraintScalingPolicy):
        raise TypeError("scaling must be ConstraintScalingPolicy or None.")
    coordinates, unflatten = ravel_pytree(parameters)
    layout = _constraint_layout(problem, parameters, args)
    objective, gradient = jax.value_and_grad(
        lambda value: problem.value(unflatten(value), args)[0]
    )(coordinates)
    values = _flat_constraint_values(problem, parameters, args)
    jacobian = jax.jacfwd(
        lambda value: _flat_constraint_values(problem, unflatten(value), args)
    )(coordinates)
    if policy.automatic:
        objective_scale = jnp.maximum(
            jnp.maximum(jnp.abs(objective), jnp.linalg.norm(gradient, ord=jnp.inf)),
            policy.objective_floor,
        )
        row_norms = jnp.linalg.norm(jacobian, axis=1, ord=jnp.inf)
        constraint_scale = jnp.maximum(
            jnp.maximum(jnp.abs(values), row_norms),
            policy.constraint_floor,
        )
    else:
        objective_scale = jnp.asarray(1.0, dtype=objective.dtype)
        constraint_scale = jnp.ones_like(values)
    model = _make_constrained_model(
        problem,
        coordinates,
        unflatten,
        layout,
        objective_scale,
        constraint_scale,
        automatic=policy.automatic,
    )
    initial = model._evaluation(
        model._values(parameters, coordinates, objective, values), gradient, jacobian
    )
    return model, initial


def _make_constrained_model(
    problem: MinimizationProblem,
    coordinates: Array,
    unflatten: Any,
    layout: _ConstraintLayout,
    objective_scale: Array,
    constraint_scale: Array,
    *,
    automatic: bool,
) -> PreparedConstrainedModel:
    model_id = canonical_fingerprint(
        {
            "kind": "constrained-model",
            "problem": problem.problem_id,
            "dimension": coordinates.size,
            "constraints": layout.lower.size,
            "equalities": layout.equality_indices.tolist(),
            "lower": layout.lower_indices.tolist(),
            "upper": layout.upper_indices.tolist(),
            "automatic_scaling": automatic,
        }
    )
    return PreparedConstrainedModel(
        problem=problem,
        lower=layout.lower,
        upper=layout.upper,
        equality_indices=layout.equality_indices,
        lower_indices=layout.lower_indices,
        upper_indices=layout.upper_indices,
        equality_sources=layout.equality_sources,
        inequality_sources=layout.inequality_sources,
        objective_scale=objective_scale,
        constraint_scale=constraint_scale,
        template_coordinates=coordinates,
        unflatten=unflatten,
        model_id=model_id,
    )


def prepare_constrained_model_actions(
    problem: MinimizationProblem, parameters: PyTree[Any], /, *, args: Any = None
) -> PreparedConstrainedModel:
    """Prepare unscaled sensitivity actions without any dense derivative."""
    if not isinstance(problem, MinimizationProblem):
        raise TypeError("problem must be MinimizationProblem.")
    if not problem.constraints and problem.bounds is None:
        raise ValueError("A constrained model requires constraints or bounds.")
    coordinates, unflatten = ravel_pytree(parameters)
    layout = _constraint_layout(problem, parameters, args)
    return _make_constrained_model(
        problem,
        coordinates,
        unflatten,
        layout,
        jnp.asarray(1.0, dtype=coordinates.dtype),
        jnp.ones_like(layout.lower),
        automatic=False,
    )


def prepare_constrained_model(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
    scaling: ConstraintScalingPolicy | None = None,
) -> PreparedConstrainedModel:
    return prepare_constrained_model_evaluation(
        problem, parameters, args=args, scaling=scaling
    )[0]


__all__ = [
    "ConstrainedModelEvaluation",
    "ConstrainedModelActions",
    "ConstrainedModelValues",
    "ConstraintScalingPolicy",
    "PreparedConstrainedModel",
    "prepare_constrained_model",
    "prepare_constrained_model_evaluation",
    "prepare_constrained_model_actions",
]
