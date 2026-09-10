#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

import phydrax.linalg as la
from phydrax.linalg import eigen as eigen_api

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


class StructuredCurvatureEstimate(StrictModule):
    """Native curvature representation and retained spectral evidence."""

    matrix: Array
    diagonal: Array
    eigenvalues: Array
    eigenvectors: Array
    successful: Array
    representation: str = eqx.field(static=True)
    rank: int | None = eqx.field(static=True)


class StructuredPosteriorState(StrictModule):
    """Spectral posterior-precision state used by covariance and scale actions."""

    precision: Array
    eigenvalues: Array
    eigenvectors: Array
    successful: Array


class StructuredLaplaceResult(StrictModule):
    """Native structured Laplace posterior over an explicit parameter PyTree."""

    problem: PosteriorProblem
    map_position: PyTree[Array]
    map_parameters: PyTree[Array]
    curvature_estimate: StructuredCurvatureEstimate
    posterior_state: StructuredPosteriorState
    gradient_norm: Array
    prior_precision: Array
    whitening: GaussianPriorWhitening | None
    scale_mv: Callable[[PyTree[Array]], PyTree[Array]] = eqx.field(static=True)
    covariance_mv: Callable[[PyTree[Array]], PyTree[Array]] = eqx.field(static=True)
    curvature: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    rank: int | None = eqx.field(static=True)
    likelihood_curvature: str = eqx.field(static=True)
    approximate_memory_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        map_position: PyTree[Array],
        curvature_estimate: StructuredCurvatureEstimate,
        posterior_state: StructuredPosteriorState,
        gradient_norm: Array,
        prior_precision: Array,
        whitening: GaussianPriorWhitening | None,
        scale_mv: Callable[[PyTree[Array]], PyTree[Array]],
        covariance_mv: Callable[[PyTree[Array]], PyTree[Array]],
        curvature: str,
        dimension: int,
        rank: int | None,
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
        self.likelihood_curvature = str(likelihood_curvature)
        self.approximate_memory_bytes = int(approximate_memory_bytes)

    def covariance_vector_product(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.covariance_mv(vector)

    def physical_covariance_vector_product(
        self, vector: PyTree[Array], /
    ) -> PyTree[Array]:
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
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        leaves, treedef = jax.tree_util.tree_flatten(self.map_position)
        keys = jr.split(key, len(leaves))
        noise = jax.tree_util.tree_unflatten(
            treedef,
            [
                jr.normal(draw_key, (count, *leaf.shape), dtype=leaf.dtype)
                for draw_key, leaf in zip(keys, leaves, strict=True)
            ],
        )
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
        return self.problem.parameter_space.constrain(
            self.sample_unconstrained(key, num_samples=num_samples)
        )

    def linearized_predict(
        self,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> LinearizedPropagationResult:
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
        position_key, observation_key = jr.split(key)
        positions = self.sample_unconstrained(position_key, num_samples=num_samples)
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


def _native_eigh(matrix: Array, /) -> tuple[Array, Array, Array]:
    dimension = int(matrix.shape[0])
    properties = la.OperatorProperties(
        self_adjoint=True,
        evidence={"self_adjoint": "construction"},
    )
    result = eigen_api.eigensolve(
        eigen_api.Eigenproblem(
            la.DenseLinearOperator(matrix, properties=properties),
            problem_id="structured-laplace-spectrum",
        ),
        policy=eigen_api.EigenSolvePolicy(
            eigen_api.DenseEigh(),
            count=dimension,
            which="smallest-algebraic",
        ),
    )
    return result.eigenvalues, result.eigenvectors, result.successful


def fit_structured_laplace(
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
    """Fit one native dense, diagonal, or spectrally truncated Laplace posterior."""
    del mv_jit
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if curvature not in ("full", "diagonal", "lanczos", "lobpcg"):
        raise ValueError(f"Unknown structured curvature {curvature!r}.")
    if likelihood_curvature not in ("hessian", "ggn"):
        raise ValueError("likelihood_curvature must be 'hessian' or 'ggn'.")
    if likelihood_curvature == "ggn" and problem.gauss_newton_residual_fn is None:
        raise ValueError("GGN curvature requires an explicit residual callback.")
    whitening = None
    if prior_precision is None:
        whitening = GaussianPriorWhitening.from_parameter_space(problem.parameter_space)
        prior_value = 1.0
    else:
        prior_value = float(prior_precision)
        if not jnp.isfinite(prior_value) or prior_value <= 0.0:
            raise ValueError("prior_precision must be finite and strictly positive.")
    rank_ = int(rank)
    tolerance_ = float(tolerance)
    if rank_ <= 0 or tolerance_ <= 0.0:
        raise ValueError("rank and tolerance must be positive.")

    position = problem.initial_position if map_position is None else map_position
    problem.parameter_space.constrain(position)
    base_working = position if whitening is None else whitening.whiten(position)
    flat_position, unravel = ravel_pytree(base_working)
    dimension = int(flat_position.size)
    if dimension <= 0:
        raise ValueError("Structured Laplace position must be non-empty.")
    if curvature in ("lanczos", "lobpcg"):
        if rank_ >= dimension:
            raise ValueError("Low-rank curvature rank must be smaller than dimension.")
        if key is None:
            raise ValueError("Low-rank curvature requires a PRNG key.")

    def to_position(flat):
        value = unravel(flat)
        return value if whitening is None else whitening.unwhiten(value)

    def negative_log_likelihood(flat):
        value = to_position(flat)
        return -problem.log_likelihood(problem.parameter_space.constrain(value))

    if likelihood_curvature == "hessian":
        gradient_linearization = la.prepare_linearization(
            jax.grad(negative_log_likelihood),
            flat_position,
        )
        curvature_mv = gradient_linearization.jvp
    else:
        residual_linearization = la.prepare_linearization(
            lambda flat: problem.gauss_newton_residual(to_position(flat)),
            flat_position,
        )

        def curvature_mv(vector):
            return residual_linearization.vjp(residual_linearization.jvp(vector))

    retained_rank: int | None = None
    if curvature in ("lanczos", "lobpcg"):
        retained_rank = rank_
        space = la.ArraySpace((dimension,), dtype=flat_position.dtype)
        operator = la.FunctionLinearOperator(
            curvature_mv,
            source=space,
            target=space,
            properties=la.OperatorProperties(
                self_adjoint=True,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id="structured-curvature-action",
        )
        method = (
            eigen_api.RestartedLanczos(
                subspace_dimension=min(dimension, max(2 * rank_ + 4, rank_ + 2)),
                restart_dimension=rank_,
            )
            if curvature == "lanczos"
            else eigen_api.LOBPCG(block_dimension=rank_)
        )
        spectrum = eigen_api.eigensolve(
            eigen_api.Eigenproblem(
                operator,
                problem_id="structured-curvature-low-rank",
            ),
            policy=eigen_api.EigenSolvePolicy(
                method,
                count=rank_,
                which="largest-magnitude",
                max_steps=max(32, 4 * dimension),
                tolerance=eigen_api.EigenTolerancePolicy(
                    relative=tolerance_,
                    absolute=tolerance_,
                    orthogonality=max(tolerance_, 1e-6),
                ),
                key=key,
            ),
        )
        retained_values = spectrum.eigenvalues
        retained_vectors = spectrum.eigenvectors
        full_successful = spectrum.successful
        matrix = jnp.zeros((0, 0), dtype=flat_position.dtype)
        diagonal = jnp.zeros((dimension,), dtype=jnp.real(flat_position).dtype)
        approximation = (retained_vectors * retained_values[None, :]) @ jnp.conj(
            retained_vectors.T
        )
    else:
        basis = jnp.eye(dimension, dtype=flat_position.dtype)
        matrix = jax.vmap(curvature_mv)(basis).T
        matrix = 0.5 * (matrix + jnp.conj(matrix.T))
        diagonal = jnp.real(jnp.diag(matrix))
        full_values, full_vectors, full_successful = _native_eigh(matrix)
        if curvature == "full":
            approximation = matrix
            retained_values = full_values
            retained_vectors = full_vectors
        else:
            approximation = jnp.diag(diagonal)
            retained_values = diagonal
            retained_vectors = jnp.eye(dimension, dtype=matrix.dtype)

    precision = approximation + prior_value * jnp.eye(dimension, dtype=matrix.dtype)
    precision = 0.5 * (precision + jnp.conj(precision.T))
    precision_values, precision_vectors, precision_successful = _native_eigh(precision)
    positive = jnp.min(precision_values) > tolerance_
    precision_values = eqx.error_if(
        precision_values,
        ~(precision_successful & positive),
        "Structured Laplace posterior precision is not positive definite.",
    )

    def apply_spectral(vector, power):
        projected = jnp.conj(precision_vectors.T) @ vector
        return precision_vectors @ (projected * precision_values**power)

    def scale_mv(vector):
        flat, restore = ravel_pytree(vector)
        result = restore(apply_spectral(flat, -0.5))
        return result if whitening is None else whitening.unwhiten_vector(result)

    def covariance_mv(vector):
        base_vector = vector if whitening is None else whitening.unwhiten_vector(vector)
        flat, restore = ravel_pytree(base_vector)
        result = restore(apply_spectral(flat, -1.0))
        return result if whitening is None else whitening.unwhiten_vector(result)

    full_gradient = jax.grad(problem.negative_log_density)(position)
    gradient_norm = jnp.sqrt(
        sum(
            (jnp.sum(jnp.asarray(leaf) ** 2) for leaf in jax.tree.leaves(full_gradient)),
            jnp.zeros(()),
        )
    )
    if stationarity_tolerance is not None:
        gradient_norm = eqx.error_if(
            gradient_norm,
            gradient_norm > float(stationarity_tolerance),
            "Structured Laplace center is not stationary.",
        )
    if whitening is None:
        bijectors = jax.tree.leaves(
            problem.parameter_space.bijectors,
            is_leaf=lambda value: isinstance(value, AbstractBijector),
        )
        if any(not isinstance(value, IdentityBijector) for value in bijectors):
            raise ValueError(
                "Explicit prior_precision requires identity parameter bijectors."
            )

    estimate = StructuredCurvatureEstimate(
        matrix,
        diagonal,
        retained_values,
        retained_vectors,
        full_successful,
        curvature,
        retained_rank,
    )
    posterior_state = StructuredPosteriorState(
        precision,
        precision_values,
        precision_vectors,
        precision_successful & positive,
    )
    memory = _tree_nbytes((estimate, posterior_state))
    return StructuredLaplaceResult(
        problem=problem,
        map_position=position,
        curvature_estimate=estimate,
        posterior_state=posterior_state,
        gradient_norm=gradient_norm,
        prior_precision=jnp.asarray(prior_value),
        whitening=whitening,
        scale_mv=scale_mv,
        covariance_mv=covariance_mv,
        curvature=curvature,
        dimension=dimension,
        rank=retained_rank,
        likelihood_curvature=likelihood_curvature,
        approximate_memory_bytes=memory,
    )


def _tree_nbytes(tree: PyTree[Any], /) -> int:
    return sum(
        int(jnp.asarray(leaf).nbytes)
        for leaf in jax.tree.leaves(tree)
        if eqx.is_array(leaf)
    )


__all__ = [
    "LikelihoodCurvature",
    "StructuredCurvature",
    "StructuredCurvatureEstimate",
    "StructuredLaplaceResult",
    "StructuredPosteriorState",
    "fit_structured_laplace",
]
