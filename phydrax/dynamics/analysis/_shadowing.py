#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from .._evolution import AbstractDifferentiableEvolution, EvolutionTrajectory
from .._grid import IterationGrid, TimeGrid


ShadowingBoundary: TypeAlias = Literal["free", "zero", "periodic"]
ShadowingTimeDilation: TypeAlias = Literal["none", "flow"]

SHADOWING_CANDIDATE_SUCCESS = 0
SHADOWING_CANDIDATE_TRAJECTORY_INVALID = 1
SHADOWING_CANDIDATE_NONFINITE = 2
SHADOWING_CANDIDATE_SHAPE_INVALID = 3


class ShadowingSensitivityProblem(StrictModule):
    """Matrix-free contract for least-squares or NILSS-style shadowing solvers.

    ``inhomogeneous_tangent`` returns the endpoint tangent contribution of one unit
    parameter perturbation over a declared evolution segment. It is not an
    instantaneous vector-field derivative unless the evolution discretization makes
    that equivalence explicit.
    """

    evolution: AbstractDifferentiableEvolution
    inhomogeneous_tangent: Callable[[Array, Array, Array, Any], Array]
    observable: Callable[[Array, Array, Any], Array]
    observable_state_gradient: Callable[[Array, Array, Any], Array] | None
    observable_parameter_derivative: Callable[[Array, Array, Any], Array] | None
    neutral_direction: Callable[[Array, Array, Any], Array] | None
    parameter_id: str = eqx.field(static=True)
    observable_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    time_dilation: ShadowingTimeDilation = eqx.field(static=True)
    forcing_semantics: str = eqx.field(static=True)

    def __init__(
        self,
        evolution: AbstractDifferentiableEvolution,
        inhomogeneous_tangent: Callable[[Array, Array, Array, Any], Array],
        observable: Callable[[Array, Array, Any], Array],
        /,
        *,
        parameter_id: str,
        observable_id: str,
        problem_id: str,
        observable_state_gradient: Callable[[Array, Array, Any], Array] | None = None,
        observable_parameter_derivative: Callable[[Array, Array, Any], Array]
        | None = None,
        neutral_direction: Callable[[Array, Array, Any], Array] | None = None,
        time_dilation: ShadowingTimeDilation = "none",
        forcing_semantics: str = "integrated-endpoint-parameter-tangent",
    ):
        if not isinstance(evolution, AbstractDifferentiableEvolution):
            raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
        callbacks = (
            inhomogeneous_tangent,
            observable,
            observable_state_gradient,
            observable_parameter_derivative,
            neutral_direction,
        )
        if any(value is not None and not callable(value) for value in callbacks):
            raise TypeError("Shadowing callbacks must be callable or None.")
        if time_dilation not in ("none", "flow"):
            raise ValueError("Unsupported time_dilation semantics.")
        if time_dilation == "flow" and neutral_direction is None:
            raise ValueError("Flow time dilation requires neutral_direction.")
        identifiers = (
            parameter_id,
            observable_id,
            problem_id,
            forcing_semantics,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("Shadowing identifiers must be non-empty strings.")
        self.evolution = evolution
        self.inhomogeneous_tangent = inhomogeneous_tangent
        self.observable = observable
        self.observable_state_gradient = observable_state_gradient
        self.observable_parameter_derivative = observable_parameter_derivative
        self.neutral_direction = neutral_direction
        self.parameter_id = parameter_id
        self.observable_id = observable_id
        self.problem_id = problem_id
        self.time_dilation = time_dilation
        self.forcing_semantics = forcing_semantics


class ShadowingCandidateResult(StrictModule):
    """Residual and response evidence for one externally supplied shadowing path."""

    tangent_path: Array
    time_dilation: Array
    defects: Array
    defect_norm: Array
    boundary_residual: Array
    boundary_norm: Array
    neutral_inner_product: Array
    observable_values: Array
    observable_directional: Array
    time_dilation_correction: Array
    quadrature_weights: Array
    observable_mean: Array
    mean_directional_response: Array
    step_valid: Array
    node_valid: Array
    valid: Array
    status: Array
    problem: ShadowingSensitivityProblem
    trajectory: EvolutionTrajectory
    boundary: ShadowingBoundary = eqx.field(static=True)
    boundary_enforced: bool = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    response_assumption: str = eqx.field(static=True)

    def least_squares_residual(
        self,
        /,
        *,
        include_neutral_constraint: bool = True,
    ) -> Array:
        pieces = [self.defects.reshape((-1,))]
        if self.boundary_enforced:
            pieces.append(self.boundary_residual.reshape((-1,)))
        if include_neutral_constraint and self.problem.neutral_direction is not None:
            pieces.append(self.neutral_inner_product.reshape((-1,)))
        return jnp.concatenate(tuple(pieces))


def _quadrature_weights(trajectory: EvolutionTrajectory, /) -> Array:
    coordinates = trajectory.grid.coordinates
    count = int(coordinates.size)
    if isinstance(trajectory.grid, TimeGrid):
        intervals = coordinates[1:] - coordinates[:-1]
        weights = jnp.zeros((count,), dtype=coordinates.dtype)
        weights = weights.at[:-1].add(0.5 * intervals)
        weights = weights.at[1:].add(0.5 * intervals)
        return weights / jnp.sum(weights)
    if isinstance(trajectory.grid, IterationGrid):
        return jnp.full((count,), 1.0 / count, dtype=float)
    raise TypeError("Unsupported trajectory grid.")


def evaluate_shadowing_candidate(
    problem: ShadowingSensitivityProblem,
    trajectory: EvolutionTrajectory,
    tangent_path: ArrayLike,
    /,
    *,
    args: Any = None,
    time_dilation: ArrayLike | None = None,
    boundary: ShadowingBoundary = "free",
) -> ShadowingCandidateResult:
    """Evaluate, but never silently solve, one shadowing sensitivity candidate."""
    if not isinstance(problem, ShadowingSensitivityProblem):
        raise TypeError("problem must be a ShadowingSensitivityProblem.")
    if not isinstance(trajectory, EvolutionTrajectory):
        raise TypeError("trajectory must be an EvolutionTrajectory.")
    if trajectory.evolution_id != problem.evolution.evolution_id:
        raise ValueError("trajectory and shadowing evolution IDs do not match.")
    if trajectory.state_layout.layout_id != problem.evolution.state_layout.layout_id:
        raise ValueError("trajectory and shadowing state layouts do not match.")
    if boundary not in ("free", "zero", "periodic"):
        raise ValueError("Unsupported shadowing boundary condition.")
    tangent = jnp.asarray(tangent_path)
    expected = trajectory.states.shape
    if tangent.shape != expected:
        raise ValueError(f"tangent_path must have shape {expected}.")
    steps = trajectory.grid.num_steps
    dilation = (
        jnp.zeros((steps,), dtype=trajectory.states.dtype)
        if time_dilation is None
        else jnp.asarray(time_dilation)
    )
    if dilation.shape != (steps,):
        raise ValueError("time_dilation must have one scalar per evolution step.")
    if problem.time_dilation == "none" and bool(jnp.any(dilation != 0.0)):
        raise ValueError("Nonzero time dilation requires time_dilation='flow'.")
    defects = []
    defect_norms = []
    step_valid_values = []
    neutral_values = []
    for index in range(steps):
        source = trajectory.grid.coordinates[index]
        target = trajectory.grid.coordinates[index + 1]
        tangent_step = problem.evolution.tangent_action(
            trajectory.states[index],
            tangent[index],
            source,
            target,
            args,
        )
        forcing = jnp.asarray(
            problem.inhomogeneous_tangent(trajectory.states[index], source, target, args)
        )
        if forcing.shape != problem.evolution.state_layout.shape:
            raise ValueError("inhomogeneous_tangent returned the wrong shape.")
        if problem.neutral_direction is None:
            neutral = jnp.zeros_like(forcing)
        else:
            neutral = jnp.asarray(
                problem.neutral_direction(target, trajectory.states[index + 1], args)
            )
            if neutral.shape != problem.evolution.state_layout.shape:
                raise ValueError("neutral_direction returned the wrong shape.")
        defect = (
            tangent[index + 1]
            - tangent_step.tangent
            - forcing
            - dilation[index] * neutral
        )
        finite = (
            tangent_step.valid
            & jnp.all(jnp.isfinite(forcing))
            & jnp.all(jnp.isfinite(neutral))
            & jnp.all(jnp.isfinite(defect))
        )
        defects.append(defect)
        defect_norms.append(jnp.linalg.norm(defect.reshape((-1,))))
        step_valid_values.append(finite)
        neutral_values.append(neutral)
    defect_array = jnp.stack(tuple(defects), axis=0)
    defect_norm = jnp.stack(tuple(defect_norms), axis=0)
    step_valid = jnp.stack(tuple(step_valid_values), axis=0)
    if boundary == "free":
        boundary_residual = jnp.zeros_like(tangent[0])
        boundary_enforced = False
    elif boundary == "zero":
        boundary_residual = tangent[0]
        boundary_enforced = True
    else:
        boundary_residual = tangent[-1] - tangent[0]
        boundary_enforced = True
    boundary_norm = jnp.linalg.norm(boundary_residual.reshape((-1,)))
    observable_values = []
    observable_directional = []
    neutral_inner = []
    node_valid_values = []
    for index in range(steps + 1):
        coordinate = trajectory.grid.coordinates[index]
        state = trajectory.states[index]
        observable = jnp.asarray(problem.observable(coordinate, state, args))
        if observable.shape != ():
            raise ValueError("observable must return a scalar.")
        if problem.observable_state_gradient is None:
            gradient = jax.grad(
                lambda value: problem.observable(coordinate, value, args)
            )(state)
        else:
            gradient = jnp.asarray(
                problem.observable_state_gradient(coordinate, state, args)
            )
        if gradient.shape != problem.evolution.state_layout.shape:
            raise ValueError("observable_state_gradient returned the wrong shape.")
        explicit = (
            jnp.asarray(0.0, dtype=observable.dtype)
            if problem.observable_parameter_derivative is None
            else jnp.asarray(
                problem.observable_parameter_derivative(coordinate, state, args)
            )
        )
        if explicit.shape != ():
            raise ValueError("observable_parameter_derivative must return a scalar.")
        directional = (
            jnp.vdot(gradient.reshape((-1,)), tangent[index].reshape((-1,))).real
            + explicit
        )
        if problem.neutral_direction is None:
            neutral = jnp.zeros_like(state)
        elif index == 0:
            neutral = jnp.asarray(problem.neutral_direction(coordinate, state, args))
        else:
            neutral = neutral_values[index - 1]
        neutral_inner.append(
            jnp.vdot(tangent[index].reshape((-1,)), neutral.reshape((-1,))).real
        )
        node_finite = (
            trajectory.valid[index]
            & jnp.isfinite(observable)
            & jnp.all(jnp.isfinite(gradient))
            & jnp.isfinite(explicit)
            & jnp.isfinite(directional)
        )
        observable_values.append(observable)
        observable_directional.append(directional)
        node_valid_values.append(node_finite)
    observables = jnp.stack(tuple(observable_values))
    directional_values = jnp.stack(tuple(observable_directional))
    neutral_inner_product = jnp.stack(tuple(neutral_inner))
    node_valid = jnp.stack(tuple(node_valid_values))
    weights = _quadrature_weights(trajectory)
    observable_mean = jnp.sum(weights * observables)
    if problem.time_dilation == "flow":
        node_dilation = jnp.concatenate((dilation, dilation[-1:]))
        correction = node_dilation * (observables - observable_mean)
    else:
        correction = jnp.zeros_like(observables)
    mean_response = jnp.sum(weights * (directional_values + correction))
    finite = (
        jnp.all(step_valid)
        & jnp.all(node_valid)
        & jnp.all(jnp.isfinite(tangent))
        & jnp.all(jnp.isfinite(dilation))
        & jnp.isfinite(boundary_norm)
        & jnp.isfinite(mean_response)
    )
    trajectory_valid = trajectory.successful
    valid = trajectory_valid & finite
    status = jnp.where(
        ~trajectory_valid,
        SHADOWING_CANDIDATE_TRAJECTORY_INVALID,
        jnp.where(
            finite,
            SHADOWING_CANDIDATE_SUCCESS,
            SHADOWING_CANDIDATE_NONFINITE,
        ),
    ).astype(jnp.int32)
    return ShadowingCandidateResult(
        tangent_path=tangent,
        time_dilation=dilation,
        defects=defect_array,
        defect_norm=defect_norm,
        boundary_residual=boundary_residual,
        boundary_norm=boundary_norm,
        neutral_inner_product=neutral_inner_product,
        observable_values=observables,
        observable_directional=directional_values,
        time_dilation_correction=correction,
        quadrature_weights=weights,
        observable_mean=observable_mean,
        mean_directional_response=mean_response,
        step_valid=step_valid,
        node_valid=node_valid,
        valid=valid,
        status=status,
        problem=problem,
        trajectory=trajectory,
        boundary=boundary,
        boundary_enforced=boundary_enforced,
        method_id="matrix-free-shadowing-candidate-residual",
        response_assumption=(
            "quadrature-average-tangent-response-with-flow-time-dilation"
            if problem.time_dilation == "flow"
            else "quadrature-average-tangent-response"
        ),
    )


__all__ = [
    "SHADOWING_CANDIDATE_NONFINITE",
    "SHADOWING_CANDIDATE_SHAPE_INVALID",
    "SHADOWING_CANDIDATE_SUCCESS",
    "SHADOWING_CANDIDATE_TRAJECTORY_INVALID",
    "ShadowingBoundary",
    "ShadowingCandidateResult",
    "ShadowingSensitivityProblem",
    "ShadowingTimeDilation",
    "evaluate_shadowing_candidate",
]
