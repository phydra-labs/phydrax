#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import validate_inexact_tree
from ..linalg import (
    AbstractLinearOperator,
    AbstractVectorSpace,
    adjoint,
    FunctionLinearOperator,
    PyTreeSpace,
)
from ..linalg._spaces import _coordinate_pairing_weights, _has_diagonal_pairing
from ._preconditioning import (
    _TransformationEvaluation,
    AbstractNonlinearSystemTransformation,
)
from ._types import NonlinearSystemProblem, NonlinearTermination


ScalingMode: TypeAlias = Literal["none", "automatic", "explicit"]


class NonlinearScaling(StrictModule):
    """Positive real physical state and residual scaling factors."""

    state_scale: PyTree[Array]
    residual_scale: PyTree[Array]
    scaling_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_scale: PyTree[Any],
        residual_scale: PyTree[Any],
        /,
        *,
        scaling_id: str | None = None,
    ):
        state_scale_ = validate_inexact_tree(
            state_scale,
            name="state_scale",
            real=True,
        )
        residual_scale_ = validate_inexact_tree(
            residual_scale,
            name="residual_scale",
            real=True,
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
        label = None if scaling_id is None else str(scaling_id)
        if label == "":
            raise ValueError("scaling_id must be non-empty.")
        fingerprint = canonical_fingerprint(
            {
                "kind": "nonlinear-scaling",
                "state_scale": array_tree_fingerprint(state_scale_),
                "residual_scale": array_tree_fingerprint(residual_scale_),
            }
        )
        self.state_scale = state_scale_
        self.residual_scale = residual_scale_
        self.scaling_id = fingerprint if label is None else f"{label}/{fingerprint}"

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


def _validate_scale_tree(
    scale: PyTree[Array],
    physical: PyTree[Array],
    name: str,
    /,
) -> None:
    scale_leaves, scale_structure = jax.tree.flatten(scale)
    physical_leaves, physical_structure = jax.tree.flatten(physical)
    if scale_structure != physical_structure:
        raise ValueError(f"{name} must match the physical PyTree structure.")
    for scale_leaf, physical_leaf in zip(
        scale_leaves,
        physical_leaves,
        strict=True,
    ):
        if scale_leaf.shape != physical_leaf.shape:
            raise ValueError(f"{name} leaves must match the physical leaf shapes.")
        expected_dtype = np.dtype(jnp.real(physical_leaf).dtype)
        if np.dtype(scale_leaf.dtype) != expected_dtype:
            raise TypeError(
                f"{name} leaves must use the corresponding physical real dtype "
                f"{expected_dtype}; got {scale_leaf.dtype}."
            )


def _scale_norm_bounds(
    space: AbstractVectorSpace,
    scale: PyTree[Array],
    name: str,
    /,
) -> tuple[float, float]:
    if not _has_diagonal_pairing(space):
        raise TypeError(
            f"{name} requires a Euclidean or coordinate-diagonal physical pairing."
        )
    scale_coordinates = np.concatenate(
        tuple(np.asarray(leaf).reshape((-1,)) for leaf in jax.tree.leaves(scale))
    )
    pairing_weights = np.asarray(jnp.real(_coordinate_pairing_weights(space))).reshape(
        (-1,)
    )
    gains = np.sqrt(pairing_weights) * scale_coordinates
    if gains.size != space.size or not np.all(np.isfinite(gains) & (gains > 0.0)):
        raise ValueError(f"{name} does not define finite positive norm bounds.")
    return float(np.min(gains)), float(np.max(gains))


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
                jax.tree.map(lambda value: jnp.ones_like(jnp.real(value)), state_),
                jax.tree.map(lambda value: jnp.ones_like(jnp.real(value)), residual),
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
            )
        if scaling is None:
            raise ValueError("Scaling preparation failed.")
        _validate_scale_tree(scaling.state_scale, state_, "state_scale")
        _validate_scale_tree(scaling.residual_scale, residual, "residual_scale")
        return problem_, scaling


class ScaledRootSystem(AbstractNonlinearSystemTransformation):
    """Fixed diagonal scaling represented as a canonical root transformation."""

    scaling: NonlinearScaling
    _state_norm_lower: float = eqx.field(static=True)
    _state_norm_upper: float = eqx.field(static=True)
    _residual_norm_lower: float = eqx.field(static=True)
    _residual_norm_upper: float = eqx.field(static=True)

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        scaling: NonlinearScaling,
        /,
    ):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        if not isinstance(scaling, NonlinearScaling):
            raise TypeError("scaling must be a NonlinearScaling.")
        if problem.state_space is None or problem.residual_space is None:
            raise ValueError("A scaled root requires bound physical vector spaces.")

        physical_state = problem.state_space.zeros()
        physical_residual = problem.residual_space.zeros()
        _validate_scale_tree(scaling.state_scale, physical_state, "state_scale")
        _validate_scale_tree(
            scaling.residual_scale,
            physical_residual,
            "residual_scale",
        )
        state_lower, state_upper = _scale_norm_bounds(
            problem.state_space,
            scaling.state_scale,
            "state scaling",
        )
        residual_lower, residual_upper = _scale_norm_bounds(
            problem.residual_space,
            scaling.residual_scale,
            "residual scaling",
        )

        solver_state_template = scaling.to_solver_state(physical_state)
        solver_residual_template = scaling.to_solver_residual(physical_residual)
        solver_state_space = PyTreeSpace(solver_state_template)
        solver_residual_space = PyTreeSpace(solver_residual_template)
        transformation_id = f"scaled/{scaling.scaling_id}"

        state_to_physical = FunctionLinearOperator(
            scaling.to_physical_state,
            source=solver_state_space,
            target=problem.state_space,
            operator_id=f"{transformation_id}/state-to-physical",
        )
        physical_residual_to_solver = FunctionLinearOperator(
            scaling.to_solver_residual,
            source=problem.residual_space,
            target=solver_residual_space,
            operator_id=f"{transformation_id}/residual-to-solver",
        )
        physical_state_to_solver_adjoint = adjoint(state_to_physical)
        solver_residual_to_physical_adjoint = adjoint(physical_residual_to_solver)

        def residual(solver_state, current_args):
            state = scaling.to_physical_state(solver_state)
            physical_value, auxiliary = problem.evaluate(state, current_args)
            transformed = scaling.to_solver_residual(physical_value)
            return transformed, _TransformationEvaluation(
                state,
                physical_value,
                auxiliary,
            )

        def valid(_, __, payload, current_args):
            return problem.valid(
                payload.state,
                payload.residual,
                payload.auxiliary,
                current_args,
            )

        def transform_setup(factory):
            def setup(solver_state, current_args):
                state = scaling.to_physical_state(solver_state)
                physical_operator = factory(state, current_args)
                if not isinstance(physical_operator, AbstractLinearOperator):
                    raise TypeError("Physical linear setup must return an operator.")
                return physical_residual_to_solver @ physical_operator @ state_to_physical

            return setup

        def transform_adjoint_setup(factory):
            def setup(solver_state, current_args):
                state = scaling.to_physical_state(solver_state)
                physical_operator = factory(state, current_args)
                if not isinstance(physical_operator, AbstractLinearOperator):
                    raise TypeError("Physical adjoint setup must return an operator.")
                return (
                    physical_state_to_solver_adjoint
                    @ physical_operator
                    @ solver_residual_to_physical_adjoint
                )

            return setup

        self.original = problem
        self.scaling = scaling
        self.transformation_id = transformation_id
        self._state_norm_lower = state_lower
        self._state_norm_upper = state_upper
        self._residual_norm_lower = residual_lower
        self._residual_norm_upper = residual_upper
        self.problem = NonlinearSystemProblem(
            residual,
            state_space=solver_state_space,
            residual_space=solver_residual_space,
            validity=valid,
            trial_validity=(
                None
                if problem.trial_validity_function is None
                else lambda solver_state, current_args: problem.trial_valid(
                    scaling.to_physical_state(solver_state),
                    current_args,
                )
            ),
            trial_validity_id=problem.trial_validity_id,
            linear_setup=(
                None
                if problem.linear_setup_function is None
                else transform_setup(problem.linear_setup_function)
            ),
            tangent_linear_setup=(
                None
                if problem.tangent_linear_setup_function is None
                else transform_setup(problem.tangent_linear_setup_function)
            ),
            adjoint_linear_setup=(
                None
                if problem.adjoint_linear_setup_function is None
                else transform_adjoint_setup(problem.adjoint_linear_setup_function)
            ),
            has_aux=True,
            problem_id=f"{problem.problem_id}/{transformation_id}",
        )

    def solver_initial(
        self,
        initial_state: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        del args
        physical_state = self.original.validate_state(initial_state)
        return self.problem.validate_state(self.scaling.to_solver_state(physical_state))

    def solver_termination(
        self,
        termination: NonlinearTermination,
        /,
    ) -> NonlinearTermination:
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        residual_ratio = self._residual_norm_lower / self._residual_norm_upper
        state_ratio = self._state_norm_lower / self._state_norm_upper
        divergence = (
            termination.divergence_factor
            * max(self._residual_norm_upper, 1.0)
            / self._residual_norm_lower
        )
        values = (
            termination.absolute_residual / self._residual_norm_upper,
            termination.relative_residual * residual_ratio,
            termination.absolute_step / self._state_norm_upper,
            termination.relative_step * state_ratio,
            divergence,
        )
        maximum_residual = (
            None
            if termination.maximum_residual is None
            else termination.maximum_residual / self._residual_norm_upper
        )
        if any(not isfinite(value) for value in values) or (
            maximum_residual is not None and not isfinite(maximum_residual)
        ):
            raise ValueError("Scaling does not admit finite transformed stopping limits.")
        return NonlinearTermination(
            absolute_residual=values[0],
            relative_residual=values[1],
            maximum_residual=maximum_residual,
            absolute_step=values[2],
            relative_step=values[3],
            maximum_steps=termination.maximum_steps,
            maximum_evaluations=termination.maximum_evaluations,
            maximum_linear_iterations=termination.maximum_linear_iterations,
            divergence_factor=divergence,
        )

    def reconstruct(
        self,
        solver_state: PyTree[Any],
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        del args
        transformed = self.problem.validate_state(solver_state)
        return self.original.validate_state(self.scaling.to_physical_state(transformed))


def scale_root(
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    /,
    *,
    policy: NonlinearScalingPolicy | None = None,
    args: Any = None,
) -> ScaledRootSystem:
    """Prepare fixed scales and return a reusable canonical root transformation.

    Preparation performs one physical residual evaluation outside subsequent root
    work budgets and result diagnostics.
    """
    policy_ = NonlinearScalingPolicy() if policy is None else policy
    if not isinstance(policy_, NonlinearScalingPolicy):
        raise TypeError("policy must be NonlinearScalingPolicy or None.")
    physical, scaling = policy_.prepare(problem, initial_state, args)
    return ScaledRootSystem(physical, scaling)


__all__ = [
    "NonlinearScaling",
    "NonlinearScalingPolicy",
    "ScaledRootSystem",
    "ScalingMode",
    "scale_root",
]
