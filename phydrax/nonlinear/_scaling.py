#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import validate_inexact_tree
from ._types import NonlinearSystemProblem


ScalingMode: TypeAlias = Literal["none", "automatic", "explicit"]


class NonlinearScaling(StrictModule):
    """Positive physical state and residual scaling factors."""

    state_scale: PyTree[Any]
    residual_scale: PyTree[Any]
    scaling_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_scale: PyTree[Any],
        residual_scale: PyTree[Any],
        /,
        *,
        scaling_id: str | None = None,
    ):
        state_scale_ = validate_inexact_tree(state_scale, name="state_scale")
        residual_scale_ = validate_inexact_tree(
            residual_scale,
            name="residual_scale",
        )
        state_positive = jax.tree.reduce(
            lambda left, right: left & right,
            jax.tree.map(
                lambda value: jnp.all(jnp.isfinite(value) & (value > 0.0)),
                state_scale_,
            ),
            jnp.asarray(True),
        )
        residual_positive = jax.tree.reduce(
            lambda left, right: left & right,
            jax.tree.map(
                lambda value: jnp.all(jnp.isfinite(value) & (value > 0.0)),
                residual_scale_,
            ),
            jnp.asarray(True),
        )
        state_scale_ = eqx.error_if(
            state_scale_,
            ~state_positive,
            "state_scale must be finite and positive.",
        )
        residual_scale_ = eqx.error_if(
            residual_scale_,
            ~residual_positive,
            "residual_scale must be finite and positive.",
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "nonlinear-scaling",
                    "state_shapes": [
                        tuple(value.shape) for value in jax.tree.leaves(state_scale_)
                    ],
                    "residual_shapes": [
                        tuple(value.shape) for value in jax.tree.leaves(residual_scale_)
                    ],
                }
            )
            if scaling_id is None
            else str(scaling_id)
        )
        if not identifier:
            raise ValueError("scaling_id must be non-empty.")
        self.state_scale = state_scale_
        self.residual_scale = residual_scale_
        self.scaling_id = identifier

    def to_solver_state(self, physical_state, /):
        return jax.tree.map(
            lambda value, scale: value / scale,
            physical_state,
            self.state_scale,
        )

    def to_physical_state(self, solver_state, /):
        return jax.tree.map(
            lambda value, scale: value * scale,
            solver_state,
            self.state_scale,
        )

    def to_solver_residual(self, physical_residual, /):
        return jax.tree.map(
            lambda value, scale: value / scale,
            physical_residual,
            self.residual_scale,
        )

    def to_physical_residual(self, solver_residual, /):
        return jax.tree.map(
            lambda value, scale: value * scale,
            solver_residual,
            self.residual_scale,
        )


class NonlinearScalingPolicy(StrictModule):
    """Static scaling choice with explicit floors for automatic preparation."""

    mode: ScalingMode = eqx.field(static=True)
    state_floor: float = eqx.field(static=True)
    residual_floor: float = eqx.field(static=True)
    explicit: NonlinearScaling | None

    def __init__(
        self,
        mode: ScalingMode = "automatic",
        /,
        *,
        state_floor: float = 1.0,
        residual_floor: float = 1.0,
        explicit: NonlinearScaling | None = None,
    ):
        if mode not in ("none", "automatic", "explicit"):
            raise ValueError("Unknown nonlinear scaling mode.")
        values = (float(state_floor), float(residual_floor))
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Scaling floors must be finite and positive.")
        if mode == "explicit" and not isinstance(explicit, NonlinearScaling):
            raise ValueError("Explicit scaling mode requires NonlinearScaling.")
        if mode != "explicit" and explicit is not None:
            raise ValueError("explicit scaling is valid only in explicit mode.")
        self.mode = mode
        self.state_floor, self.residual_floor = values
        self.explicit = explicit

    def prepare(
        self,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        args: Any = None,
        /,
    ) -> tuple[NonlinearSystemProblem, NonlinearScaling]:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        state_ = problem.validate_state(state)
        residual, _ = problem.evaluate(state_, args)
        problem_ = problem.bind_spaces(state_, residual)
        if self.mode == "explicit":
            scaling = self.explicit
        elif self.mode == "none":
            scaling = NonlinearScaling(
                jax.tree.map(jnp.ones_like, state_),
                jax.tree.map(jnp.ones_like, residual),
                scaling_id="identity-scaling",
            )
        else:
            scaling = NonlinearScaling(
                jax.tree.map(
                    lambda value: jnp.maximum(jnp.abs(value), self.state_floor),
                    state_,
                ),
                jax.tree.map(
                    lambda value: jnp.maximum(
                        jnp.abs(value),
                        self.residual_floor,
                    ),
                    residual,
                ),
                scaling_id=f"automatic/{problem_.problem_id}",
            )
        if scaling is None:
            raise ValueError("Scaling preparation failed.")
        problem_.state_space.validate(scaling.state_scale)
        problem_.residual_space.validate(scaling.residual_scale)
        return problem_, scaling


class PreparedScaledRoot(StrictModule):
    """Physical problem and fixed scaling used by a solver-coordinate route."""

    physical_problem: NonlinearSystemProblem
    solver_problem: NonlinearSystemProblem
    scaling: NonlinearScaling

    def solver_initial(self, physical_state, /):
        return self.scaling.to_solver_state(physical_state)

    def physical_state(self, solver_state, /):
        return self.scaling.to_physical_state(solver_state)


def prepare_scaled_root(
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    /,
    *,
    policy: NonlinearScalingPolicy | None = None,
    args: Any = None,
) -> PreparedScaledRoot:
    policy_ = NonlinearScalingPolicy() if policy is None else policy
    if not isinstance(policy_, NonlinearScalingPolicy):
        raise TypeError("policy must be NonlinearScalingPolicy or None.")
    physical, scaling = policy_.prepare(problem, initial_state, args)

    def residual(solver_state, current_args):
        physical_state = scaling.to_physical_state(solver_state)
        physical_residual, physical_auxiliary = physical.evaluate(
            physical_state,
            current_args,
        )
        return scaling.to_solver_residual(physical_residual), (
            physical_state,
            physical_residual,
            physical_auxiliary,
        )

    solver_initial = scaling.to_solver_state(initial_state)
    solver_residual, _ = residual(solver_initial, args)
    solver_problem = NonlinearSystemProblem(
        residual,
        has_aux=True,
        problem_id=f"{physical.problem_id}/scaled/{scaling.scaling_id}",
    ).bind_spaces(solver_initial, solver_residual)
    return PreparedScaledRoot(physical, solver_problem, scaling)


__all__ = [
    "NonlinearScaling",
    "NonlinearScalingPolicy",
    "PreparedScaledRoot",
    "ScalingMode",
    "prepare_scaled_root",
]
