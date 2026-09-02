#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import array_tree_fingerprint
from .._strict import StrictModule
from ._minibatch_posterior import LikelihoodBatch, MinibatchPosteriorProblem
from ._parameterized_state_space import ParameterizedStateSpaceProblem
from ._state_space_buffered import StateSpaceWindowBatch, StateSpaceWindowPlan
from ._stochastic_gradient import (
    AbstractStochasticGradientEstimator,
    STOCHASTIC_GRADIENT_INVALID,
    STOCHASTIC_GRADIENT_SUCCESS,
    StochasticGradientEstimate,
)


class BufferedParticleCorrectionDiagnostics(StrictModule):
    """Boundary-message quality and paired-buffer approximation evidence."""

    left_ess: Array
    right_ess: Array
    paired_buffer_error: Array
    anchor_distance: Array
    valid: Array
    exact: bool = eqx.field(static=True)
    status: str = eqx.field(static=True)


class ExactStateSpaceBoundaryCorrection(StrictModule):
    """Exact finite-state or linear-Gaussian boundary score-term provider."""

    score_terms_fn: Callable[..., tuple[Array, PyTree[Array]]] = eqx.field(static=True)
    diagnostics: BufferedParticleCorrectionDiagnostics
    correction_id: str = eqx.field(static=True)
    boundary_class: Literal["finite_state", "linear_gaussian"] = eqx.field(static=True)

    def score_terms(
        self,
        position: PyTree[Any],
        window: StateSpaceWindowBatch,
        key: Array,
        /,
    ) -> tuple[Array, PyTree[Array]]:
        return self.score_terms_fn(position, window, key)


class ParticleBoundaryCorrection(StrictModule):
    """Finite-particle finite-buffer boundary messages with explicit evidence."""

    left_particles: PyTree[Array]
    right_particles: PyTree[Array]
    left_weights: Array
    right_weights: Array
    left_ancestry: Array
    right_ancestry: Array
    score_terms_fn: Callable[..., tuple[Array, PyTree[Array]]] = eqx.field(static=True)
    diagnostics: BufferedParticleCorrectionDiagnostics
    correction_id: str = eqx.field(static=True)
    anchor_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        left_particles: PyTree[Array],
        right_particles: PyTree[Array],
        left_weights: ArrayLike,
        right_weights: ArrayLike,
        left_ancestry: ArrayLike,
        right_ancestry: ArrayLike,
        score_terms: Callable[..., tuple[Array, PyTree[Array]]],
        diagnostics: BufferedParticleCorrectionDiagnostics,
        correction_id: str,
        anchor_position: PyTree[Any],
    ):
        if not callable(score_terms):
            raise TypeError("score_terms must be callable.")
        if not isinstance(diagnostics, BufferedParticleCorrectionDiagnostics):
            raise TypeError("diagnostics must be BufferedParticleCorrectionDiagnostics.")
        left = _normalized_weights(left_weights, "left_weights")
        right = _normalized_weights(right_weights, "right_weights")
        left_ids = jnp.asarray(left_ancestry, dtype=jnp.int32)
        right_ids = jnp.asarray(right_ancestry, dtype=jnp.int32)
        if left_ids.shape != left.shape or right_ids.shape != right.shape:
            raise ValueError("Boundary ancestry must align with particle weights.")
        identity = str(correction_id)
        if not identity:
            raise ValueError("correction_id must be non-empty.")
        self.left_particles = left_particles
        self.right_particles = right_particles
        self.left_weights = left
        self.right_weights = right
        self.left_ancestry = left_ids
        self.right_ancestry = right_ids
        self.score_terms_fn = score_terms
        self.diagnostics = diagnostics
        self.correction_id = identity
        self.anchor_fingerprint = array_tree_fingerprint(anchor_position)["sha256"]

    def score_terms(
        self,
        position: PyTree[Any],
        window: StateSpaceWindowBatch,
        key: Array,
        /,
    ) -> tuple[Array, PyTree[Array]]:
        return self.score_terms_fn(position, window, key)


class BufferedParticleBoundaryPlan(StrictModule):
    """Fixed capacities, provider, and validation gates for boundary preparation."""

    window_plan: StateSpaceWindowPlan
    provider: Callable[..., Any] = eqx.field(static=True)
    left_capacity: int = eqx.field(static=True)
    right_capacity: int = eqx.field(static=True)
    particle_capacity: int = eqx.field(static=True)
    minimum_ess: float = eqx.field(static=True)
    maximum_paired_buffer_error: float = eqx.field(static=True)
    accept_approximate: bool = eqx.field(static=True)

    def __init__(
        self,
        window_plan: StateSpaceWindowPlan,
        /,
        *,
        left_capacity: int,
        right_capacity: int,
        particle_capacity: int,
        provider: Callable[..., Any],
        minimum_ess: float,
        maximum_paired_buffer_error: float,
        accept_approximate: bool = False,
    ):
        if not isinstance(window_plan, StateSpaceWindowPlan):
            raise TypeError("window_plan must be StateSpaceWindowPlan.")
        if not callable(provider):
            raise TypeError("provider must be callable.")
        left, right, particles = map(
            int, (left_capacity, right_capacity, particle_capacity)
        )
        ess = float(minimum_ess)
        error = float(maximum_paired_buffer_error)
        if left <= 0 or right <= 0 or particles <= 0:
            raise ValueError("Boundary and particle capacities must be positive.")
        if not math.isfinite(ess) or ess <= 0.0:
            raise ValueError("minimum_ess must be finite and positive.")
        if not math.isfinite(error) or error < 0.0:
            raise ValueError(
                "maximum_paired_buffer_error must be finite and nonnegative."
            )
        self.window_plan = window_plan
        self.provider = provider
        self.left_capacity = left
        self.right_capacity = right
        self.particle_capacity = particles
        self.minimum_ess = ess
        self.maximum_paired_buffer_error = error
        self.accept_approximate = bool(accept_approximate)


class BufferedParticleGradientEstimator(AbstractStochasticGradientEstimator):
    """Target-only inverse-inclusion additive score using frozen boundaries."""

    parameterized: ParameterizedStateSpaceProblem
    correction: ExactStateSpaceBoundaryCorrection | ParticleBoundaryCorrection
    window_plan: StateSpaceWindowPlan
    correction_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        parameterized: ParameterizedStateSpaceProblem,
        correction: ExactStateSpaceBoundaryCorrection | ParticleBoundaryCorrection,
        window_plan: StateSpaceWindowPlan,
        /,
    ):
        if not isinstance(parameterized, ParameterizedStateSpaceProblem):
            raise TypeError("parameterized must be ParameterizedStateSpaceProblem.")
        if not isinstance(
            correction, (ExactStateSpaceBoundaryCorrection, ParticleBoundaryCorrection)
        ):
            raise TypeError(
                "correction must be an exact or particle boundary correction."
            )
        if not isinstance(window_plan, StateSpaceWindowPlan):
            raise TypeError("window_plan must be StateSpaceWindowPlan.")
        self.parameterized = parameterized
        self.correction = correction
        self.window_plan = window_plan
        self.correction_fingerprint = (
            correction.correction_id
            if isinstance(correction, ExactStateSpaceBoundaryCorrection)
            else correction.correction_id + ":" + correction.anchor_fingerprint
        )

    @property
    def estimator_id(self) -> str:
        kind = (
            "exact"
            if isinstance(self.correction, ExactStateSpaceBoundaryCorrection)
            else "particle-approximate"
        )
        return f"buffered-state-space-{kind}:{self.correction_fingerprint}"

    @property
    def supports_control_variate(self) -> bool:
        return False

    def configuration(self) -> dict[str, Any]:
        return {
            "estimator_id": self.estimator_id,
            "parameterization_id": self.parameterized.parameterization_id,
            "target_length": self.window_plan.target_length,
            "left_buffer": self.window_plan.left_buffer,
            "right_buffer": self.window_plan.right_buffer,
        }

    def estimate(
        self,
        problem: MinibatchPosteriorProblem,
        position: PyTree[Any],
        batch: LikelihoodBatch,
        key: Array,
        /,
    ) -> StochasticGradientEstimate:
        del batch
        if (
            problem.parameter_space.raw_shapes
            != self.parameterized.parameter_space.raw_shapes
        ):
            raise ValueError("Buffered estimator parameter coordinates do not match.")
        window = self.window_plan.sample(key)
        log_terms, score_terms = self.correction.score_terms(position, window, key)
        log_terms_array = jnp.asarray(log_terms)
        if log_terms_array.shape != (self.window_plan.num_steps,):
            raise ValueError(
                "Boundary provider must return one additive log term per step."
            )
        inverse_probability = jnp.where(
            window.target_mask,
            1.0 / window.inclusion_probability,
            0.0,
        )
        likelihood_estimate = jnp.sum(inverse_probability * log_terms_array)

        def reduce_score(leaf: Array) -> Array:
            value = jnp.asarray(leaf)
            if value.shape[:1] != (self.window_plan.num_steps,):
                raise ValueError(
                    "Every additive score leaf must lead with the time axis."
                )
            weights = inverse_probability.reshape(
                (self.window_plan.num_steps,) + (1,) * (value.ndim - 1)
            )
            return jnp.sum(weights * value, axis=0)

        likelihood_gradient = jax.tree_util.tree_map(reduce_score, score_terms)
        prior_gradient = jax.grad(problem.parameter_space.unconstrained_log_prior)(
            position
        )
        gradient = jax.tree_util.tree_map(jnp.add, prior_gradient, likelihood_gradient)
        log_density = (
            likelihood_estimate
            + problem.parameter_space.unconstrained_log_prior(position)
        )
        gradient_norm = optax.tree.norm(gradient)
        finite = (
            jnp.isfinite(log_density)
            & jnp.isfinite(gradient_norm)
            & jnp.all(
                jnp.stack(
                    [
                        jnp.all(jnp.isfinite(leaf))
                        for leaf in jax.tree_util.tree_leaves(gradient)
                    ]
                )
            )
        )
        return StochasticGradientEstimate(
            gradient=gradient,
            log_density=log_density,
            gradient_norm=gradient_norm,
            valid=finite,
            status=jnp.where(
                finite, STOCHASTIC_GRADIENT_SUCCESS, STOCHASTIC_GRADIENT_INVALID
            ).astype(jnp.int32),
            likelihood_estimate=likelihood_estimate,
            estimator_id=self.estimator_id,
        )


def prepare_particle_boundary_correction(
    parameterized_problem: ParameterizedStateSpaceProblem,
    anchor_position: PyTree[Any],
    plan: BufferedParticleBoundaryPlan,
    /,
    *,
    key: Array,
) -> ExactStateSpaceBoundaryCorrection | ParticleBoundaryCorrection:
    """Prepare and gate immutable boundary state between SG-MCMC epochs."""
    if not isinstance(parameterized_problem, ParameterizedStateSpaceProblem):
        raise TypeError("parameterized_problem must be ParameterizedStateSpaceProblem.")
    if not isinstance(plan, BufferedParticleBoundaryPlan):
        raise TypeError("plan must be BufferedParticleBoundaryPlan.")
    correction = plan.provider(
        parameterized_problem,
        anchor_position,
        plan.window_plan,
        key,
        left_capacity=plan.left_capacity,
        right_capacity=plan.right_capacity,
        particle_capacity=plan.particle_capacity,
    )
    if not isinstance(
        correction, (ExactStateSpaceBoundaryCorrection, ParticleBoundaryCorrection)
    ):
        raise TypeError("Boundary provider returned an unsupported correction.")
    diagnostics = correction.diagnostics
    quality = (
        diagnostics.valid
        & (diagnostics.left_ess >= plan.minimum_ess)
        & (diagnostics.right_ess >= plan.minimum_ess)
        & (diagnostics.paired_buffer_error <= plan.maximum_paired_buffer_error)
    )
    if not bool(quality) and not plan.accept_approximate:
        raise ValueError(
            "Boundary correction failed ESS or paired-buffer validation gates."
        )
    return correction


def _normalized_weights(value: ArrayLike, name: str, /) -> Array:
    weights = jnp.asarray(value, dtype=float)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError(f"{name} must be a nonempty vector.")
    if bool(jnp.any(~jnp.isfinite(weights))) or bool(jnp.any(weights < 0.0)):
        raise ValueError(f"{name} must be finite and nonnegative.")
    if not bool(jnp.isclose(jnp.sum(weights), 1.0)):
        raise ValueError(f"{name} must sum to one.")
    return weights


__all__ = [
    "BufferedParticleBoundaryPlan",
    "BufferedParticleCorrectionDiagnostics",
    "BufferedParticleGradientEstimator",
    "ExactStateSpaceBoundaryCorrection",
    "ParticleBoundaryCorrection",
    "prepare_particle_boundary_correction",
]
