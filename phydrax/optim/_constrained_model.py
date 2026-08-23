#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ._iterative import MinimizationProblem
from ._nonlinear_constraints import _constraint_layout, _flat_constraint_values


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
    ):
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

    def evaluate(
        self,
        parameters: PyTree[Any],
        args: Any = None,
        /,
    ) -> ConstrainedModelEvaluation:
        coordinates, _ = ravel_pytree(parameters)

        def objective_coordinates(value):
            return self.problem.value(self.unflatten(value), args)[0]

        objective, gradient = jax.value_and_grad(objective_coordinates)(coordinates)

        def constraint_coordinates(value):
            return _flat_constraint_values(
                self.problem,
                self.unflatten(value),
                args,
            )

        raw = constraint_coordinates(coordinates)
        jacobian = jax.jacfwd(constraint_coordinates)(coordinates)
        equalities = raw[self.equality_indices] - self.lower[self.equality_indices]
        lower_slacks = raw[self.lower_indices] - self.lower[self.lower_indices]
        upper_slacks = self.upper[self.upper_indices] - raw[self.upper_indices]
        inequality_slacks = jnp.concatenate([lower_slacks, upper_slacks])
        lower_violation = jnp.maximum(-lower_slacks, 0.0)
        upper_violation = jnp.maximum(-upper_slacks, 0.0)
        primal = jnp.maximum(
            jnp.max(jnp.abs(equalities), initial=0.0),
            jnp.maximum(
                jnp.max(lower_violation, initial=0.0),
                jnp.max(upper_violation, initial=0.0),
            ),
        )
        scale = self.constraint_scale
        finite = (
            jnp.isfinite(objective)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.all(jnp.isfinite(raw))
            & jnp.all(jnp.isfinite(jacobian))
        )
        return ConstrainedModelEvaluation(
            parameters,
            coordinates,
            objective,
            objective / self.objective_scale,
            gradient,
            gradient / self.objective_scale,
            raw,
            equalities,
            lower_slacks,
            upper_slacks,
            inequality_slacks,
            jacobian,
            jacobian / scale[:, None],
            primal,
            finite,
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

        def lagrangian(value):
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


def prepare_constrained_model(
    problem: MinimizationProblem,
    parameters: PyTree[Any],
    /,
    *,
    args: Any = None,
    scaling: ConstraintScalingPolicy | None = None,
) -> PreparedConstrainedModel:
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
    model_id = canonical_fingerprint(
        {
            "kind": "constrained-model",
            "problem": problem.problem_id,
            "dimension": coordinates.size,
            "constraints": values.size,
            "equalities": layout.equality_indices.tolist(),
            "lower": layout.lower_indices.tolist(),
            "upper": layout.upper_indices.tolist(),
            "automatic_scaling": policy.automatic,
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


__all__ = [
    "ConstrainedModelEvaluation",
    "ConstraintScalingPolicy",
    "PreparedConstrainedModel",
    "prepare_constrained_model",
]
