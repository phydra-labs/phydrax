#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree
from laplax.curv.cov import estimate_curvature, set_posterior_fn
from laplax.enums import CurvApprox

from .._frozendict import frozendict
from .._strict import StrictModule
from ._covariance import CovarianceOperator
from ._linearized import LinearizedPropagationResult, propagate_linearized
from ._posterior import AbstractBijector, IdentityBijector, PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField
from ._whitening import GaussianPriorWhitening


StructuredCurvature = Literal["full", "diagonal", "lanczos", "lobpcg"]
LikelihoodCurvature = Literal["hessian", "ggn"]


class StructuredLaplaceResult(StrictModule):
    """Matrix-free Laplax posterior over an explicit parameter PyTree."""

    problem: PosteriorProblem
    map_position: PyTree[Array]
    map_parameters: PyTree[Array]
    curvature_estimate: Any
    posterior_state: Any
    gradient_norm: Array
    prior_precision: Array
    whitening: GaussianPriorWhitening | None
    scale_mv: Callable[[PyTree[Array]], PyTree[Array]] = eqx.field(static=True)
    covariance_mv: Callable[[PyTree[Array]], PyTree[Array]] = eqx.field(static=True)
    curvature: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    rank: int | None = eqx.field(static=True)
    duration_seconds: float = eqx.field(static=True)
    likelihood_curvature: str = eqx.field(static=True)
    approximate_memory_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        map_position: PyTree[Array],
        curvature_estimate: Any,
        posterior_state: Any,
        gradient_norm: Array,
        prior_precision: Array,
        whitening: GaussianPriorWhitening | None,
        scale_mv: Callable[[PyTree[Array]], PyTree[Array]],
        covariance_mv: Callable[[PyTree[Array]], PyTree[Array]],
        curvature: str,
        dimension: int,
        rank: int | None,
        duration_seconds: float,
        likelihood_curvature: LikelihoodCurvature,
        approximate_memory_bytes: int,
    ):
        self.problem = problem
        self.map_position = map_position
        self.map_parameters = problem.parameter_space.constrain(map_position)
        self.curvature_estimate = curvature_estimate
        self.posterior_state = posterior_state
        self.gradient_norm = jnp.asarray(gradient_norm)
        self.prior_precision = jnp.asarray(prior_precision)
        self.whitening = whitening
        self.scale_mv = scale_mv
        self.covariance_mv = covariance_mv
        self.curvature = str(curvature)
        self.dimension = int(dimension)
        self.rank = rank
        self.duration_seconds = float(duration_seconds)
        self.likelihood_curvature = str(likelihood_curvature)
        self.approximate_memory_bytes = int(approximate_memory_bytes)

    def covariance_vector_product(self, vector: PyTree[Array], /) -> PyTree[Array]:
        """Apply the approximate posterior covariance without materializing it."""
        return self.covariance_mv(vector)

    def physical_covariance_vector_product(
        self, vector: PyTree[Array], /
    ) -> PyTree[Array]:
        """Apply delta-method covariance in transformed physical coordinates."""
        propagation = propagate_linearized(
            self.problem.parameter_space.constrain,
            self.map_position,
            CovarianceOperator(self.covariance_mv),
            source="epistemic",
        )
        return propagation.covariance_vector_product(vector)

    def sample_unconstrained(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
    ) -> PyTree[Array]:
        """Draw approximate posterior samples with the Laplax scale operator."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        leaves, treedef = jax.tree_util.tree_flatten(self.map_position)
        keys = jr.split(key, len(leaves))
        noise_leaves = [
            jr.normal(draw_key, (count, *leaf.shape), dtype=leaf.dtype)
            for draw_key, leaf in zip(keys, leaves, strict=True)
        ]
        noise = jax.tree_util.tree_unflatten(treedef, noise_leaves)
        perturbation = jax.vmap(self.scale_mv)(noise)
        return jax.tree_util.tree_map(
            lambda center, delta: center + delta,
            self.map_position,
            perturbation,
        )

    def sample(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
    ) -> PyTree[Array]:
        """Draw transformed physical parameters."""
        positions = self.sample_unconstrained(key, num_samples=num_samples)
        return self.problem.parameter_space.constrain(positions)
    def linearized_predict(
        self,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> LinearizedPropagationResult:
        """Propagate structured Laplace covariance through a local prediction map."""
        return propagate_linearized(
            lambda position: self.problem.predict(position, *args, **kwargs),
            self.map_position,
            CovarianceOperator(self.covariance_mv),
            source="epistemic",
        )


    def predict(
        self,
        key: Array,
        /,
        *args: Any,
        num_samples: int,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        sample_dim: str = "__phydra_uq_draw",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw and evaluate coherent structured-Laplace functions."""
        positions = self.sample_unconstrained(key, num_samples=num_samples)
        return predict_from_position_samples(
            self.problem,
            positions,
            *args,
            sample_dims=(sample_dim,),
            sample_sources=("epistemic",),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def predict_observations(
        self,
        key: Array,
        /,
        *args: Any,
        num_samples: int,
        num_observation_samples: int,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        sample_dim: str = "__phydra_uq_draw",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        """Draw structured-Laplace parameters and conditional measurements."""
        position_key, observation_key = jr.split(key)
        positions = self.sample_unconstrained(
            position_key,
            num_samples=num_samples,
        )
        return sample_observations_from_position_samples(
            self.problem,
            observation_key,
            positions,
            *args,
            num_observation_samples=num_observation_samples,
            sample_dims=(sample_dim,),
            sample_sources=("epistemic",),
            observation_dim=observation_dim,
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )


def fit_laplax(
    problem: PosteriorProblem,
    map_position: PyTree[Array] | None = None,
    /,
    *,
    curvature: StructuredCurvature,
    prior_precision: float | None,
    rank: int = 20,
    key: Array | None = None,
    tolerance: float = 1e-6,
    stationarity_tolerance: float | None = 1e-4,
    mv_jit: bool = True,
    likelihood_curvature: LikelihoodCurvature = "hessian",
) -> StructuredLaplaceResult:
    """Fit a Laplax structured approximation with explicit or whitened priors."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if curvature not in ("full", "diagonal", "lanczos", "lobpcg"):
        raise ValueError(f"Unknown structured curvature {curvature!r}.")
    if likelihood_curvature not in ("hessian", "ggn"):
        raise ValueError("likelihood_curvature must be 'hessian' or 'ggn'.")
    if likelihood_curvature == "ggn" and problem.gauss_newton_residual_fn is None:
        raise ValueError(
            "GGN curvature requires an explicit Gauss-Newton residual callback."
        )
    whitening = None
    if prior_precision is None:
        whitening = GaussianPriorWhitening.from_parameter_space(problem.parameter_space)
        prior_value = 1.0
    else:
        prior_value = float(prior_precision)
        if not jnp.isfinite(prior_value) or prior_value <= 0.0:
            raise ValueError("prior_precision must be finite and strictly positive.")
    if int(rank) <= 0:
        raise ValueError("rank must be positive.")
    if float(tolerance) <= 0.0:
        raise ValueError("tolerance must be positive.")
    if stationarity_tolerance is not None and float(stationarity_tolerance) < 0.0:
        raise ValueError("stationarity_tolerance must be non-negative or None.")

    position = problem.initial_position if map_position is None else map_position
    problem.parameter_space.constrain(position)
    base_working_position = position if whitening is None else whitening.whiten(position)
    working_position = jax.tree_util.tree_map(
        lambda value: value.reshape((1,)) if value.ndim == 0 else value,
        base_working_position,
    )
    leaves = jax.tree_util.tree_leaves(position)
    dimension = sum(int(leaf.size) for leaf in leaves)
    if dimension <= 0:
        raise ValueError("Structured Laplace position must be non-empty.")
    if curvature in ("lanczos", "lobpcg") and int(rank) >= dimension:
        raise ValueError("Low-rank curvature rank must be smaller than dimension.")
    if curvature in ("lanczos", "lobpcg") and key is None:
        raise ValueError("Low-rank curvature requires a PRNG key.")

    full_gradient = jax.grad(lambda value: problem.negative_log_density(value))(position)
    gradient_norm = jnp.sqrt(
        sum(
            (
                jnp.sum(jnp.asarray(leaf) ** 2)
                for leaf in jax.tree_util.tree_leaves(full_gradient)
            ),
            jnp.zeros(()),
        )
    )
    if stationarity_tolerance is not None and float(gradient_norm) > float(
        stationarity_tolerance
    ):
        raise ValueError(
            "Structured Laplace center is not stationary: "
            f"gradient_norm={float(gradient_norm):.6g}."
        )

    if whitening is None:
        bijectors = jax.tree_util.tree_leaves(
            problem.parameter_space.bijectors,
            is_leaf=lambda value: isinstance(value, AbstractBijector),
        )
        if any(not isinstance(value, IdentityBijector) for value in bijectors):
            raise ValueError(
                "Explicit isotropic prior_precision requires identity parameter "
                "bijectors; omit prior_precision to use declared-prior whitening."
            )

    def to_laplax_layout(tree):
        return jax.tree_util.tree_map(
            lambda value, template: (
                jnp.asarray(value).reshape((1,))
                if template.ndim == 0
                else jnp.asarray(value)
            ),
            tree,
            base_working_position,
        )

    def from_laplax_layout(tree):
        return jax.tree_util.tree_map(
            lambda value, template: jnp.asarray(value).reshape(template.shape),
            tree,
            base_working_position,
        )

    def to_position(value):
        base_value = from_laplax_layout(value)
        return base_value if whitening is None else whitening.unwhiten(base_value)

    negative_log_likelihood = lambda value: (
        -problem.log_likelihood(problem.parameter_space.constrain(to_position(value)))
    )

    if likelihood_curvature == "hessian":

        def curvature_mv(vector):
            return jax.jvp(
                jax.grad(negative_log_likelihood),
                (working_position,),
                (vector,),
            )[1]

    else:

        def residual(value):
            return problem.gauss_newton_residual(to_position(value))

        def curvature_mv(vector):
            _, tangent = jax.jvp(
                residual,
                (working_position,),
                (vector,),
            )
            _, pullback = jax.vjp(residual, working_position)
            return pullback(tangent)[0]

    kwargs: dict[str, Any] = {}
    if curvature in ("lanczos", "lobpcg"):
        kwargs.update(
            key=key,
            rank=int(rank),
            tol=float(tolerance),
        )
        kwargs["mv_jit"] = bool(mv_jit)
    started = time.perf_counter()
    curvature_type = CurvApprox(curvature)
    curvature_estimate = estimate_curvature(
        curvature_type,
        mv=curvature_mv,
        layout=working_position,
        **kwargs,
    )
    posterior_factory = set_posterior_fn(
        curvature_type,
        curvature_estimate,
        layout=working_position,
    )
    posterior = posterior_factory(
        {"prior_prec": jnp.asarray(prior_value), "sigma_squared": jnp.asarray(1.0)}
    )
    raw_scale_mv = posterior.scale_mv(posterior.state)
    raw_covariance_mv = posterior.cov_mv(posterior.state)

    def scale_mv(vector):
        scaled = from_laplax_layout(raw_scale_mv(to_laplax_layout(vector)))
        return scaled if whitening is None else whitening.unwhiten_vector(scaled)

    def covariance_mv(vector):
        base_vector = vector if whitening is None else whitening.unwhiten_vector(vector)
        covariance_vector = from_laplax_layout(
            raw_covariance_mv(to_laplax_layout(base_vector))
        )
        return (
            covariance_vector
            if whitening is None
            else whitening.unwhiten_vector(covariance_vector)
        )

    probe = jax.tree_util.tree_map(jnp.ones_like, position)
    jax.block_until_ready(scale_mv(probe))
    duration = time.perf_counter() - started
    retained_rank = (
        int(curvature_estimate.U.shape[1]) if curvature in ("lanczos", "lobpcg") else None
    )
    approximate_memory_bytes = _tree_nbytes(curvature_estimate) + _tree_nbytes(
        posterior.state
    )
    return StructuredLaplaceResult(
        problem=problem,
        map_position=position,
        curvature_estimate=curvature_estimate,
        posterior_state=posterior.state,
        gradient_norm=gradient_norm,
        prior_precision=jnp.asarray(prior_value),
        whitening=whitening,
        scale_mv=scale_mv,
        covariance_mv=covariance_mv,
        curvature=curvature,
        dimension=dimension,
        rank=retained_rank,
        duration_seconds=duration,
        likelihood_curvature=likelihood_curvature,
        approximate_memory_bytes=approximate_memory_bytes,
    )


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(
        int(jnp.asarray(leaf).nbytes)
        for leaf in jax.tree_util.tree_leaves(tree)
        if eqx.is_array(leaf)
    )


__all__ = ["StructuredLaplaceResult", "fit_laplax"]
