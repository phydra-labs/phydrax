#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, overload, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from .._strict import StrictModule
from ._covariance import DenseCovariance
from ._linearized import LinearizedPropagationResult, propagate_linearized
from ._posterior import PosteriorProblem


if TYPE_CHECKING:
    from ._laplax_backend import StructuredLaplaceResult
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


class LaplaceCurvatureError(ValueError):
    """Raised when a requested Gaussian approximation has invalid curvature."""


class LaplaceResult(StrictModule):
    """Dense Gaussian posterior approximation in unconstrained coordinates."""

    problem: PosteriorProblem
    map_position: PyTree[Array]
    map_parameters: PyTree[Array]
    flat_map_position: Array
    gradient: PyTree[Array]
    gradient_norm: Array
    raw_precision: Array
    precision: Array
    covariance: Array
    scale: Array
    raw_eigenvalues: Array
    eigenvalues: Array
    damping: Array
    unravel: Any = eqx.field(static=True)
    backend: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem: PosteriorProblem,
        map_position: PyTree[Array],
        flat_map_position: Array,
        gradient: PyTree[Array],
        gradient_norm: Array,
        raw_precision: Array,
        precision: Array,
        covariance: Array,
        scale: Array,
        raw_eigenvalues: Array,
        eigenvalues: Array,
        damping: Array,
        unravel: Any,
        backend: str = "dense",
    ):
        self.problem = problem
        self.map_position = map_position
        self.map_parameters = problem.parameter_space.constrain(map_position)
        self.flat_map_position = jnp.asarray(flat_map_position)
        self.gradient = gradient
        self.gradient_norm = jnp.asarray(gradient_norm)
        self.raw_precision = jnp.asarray(raw_precision)
        self.precision = jnp.asarray(precision)
        self.covariance = jnp.asarray(covariance)
        self.scale = jnp.asarray(scale)
        self.raw_eigenvalues = jnp.asarray(raw_eigenvalues)
        self.eigenvalues = jnp.asarray(eigenvalues)
        self.damping = jnp.asarray(damping)
        self.unravel = unravel
        self.backend = str(backend)

    def physical_covariance(self) -> Array:
        """Return delta-method covariance after parameter bijectors."""
        propagation = propagate_linearized(
            self.problem.parameter_space.constrain,
            self.map_position,
            DenseCovariance(self.covariance),
            source="epistemic",
        )
        return propagation.materialize_covariance(
            max_dimension=propagation.output_dimension
        ).matrix

    def physical_correlation(self) -> Array:
        """Return delta-method physical-parameter correlations."""
        covariance = self.physical_covariance()
        scale = jnp.sqrt(jnp.maximum(jnp.diag(covariance), 0.0))
        denominator = scale[:, None] * scale[None, :]
        return jnp.where(denominator > 0.0, covariance / denominator, 0.0)

    @property
    def dimension(self) -> int:
        return int(self.flat_map_position.size)

    def sample_unconstrained(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
    ) -> PyTree[Array]:
        """Draw samples in the coordinates where the Hessian was evaluated."""
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        noise = jr.normal(
            key,
            (count, self.dimension),
            dtype=self.flat_map_position.dtype,
        )
        flat_samples = self.flat_map_position + noise @ self.scale.T
        return jax.vmap(self.unravel)(flat_samples)

    def sample(
        self,
        key: Array,
        /,
        *,
        num_samples: int,
    ) -> PyTree[Array]:
        """Draw transformed physical-parameter samples."""
        unconstrained = self.sample_unconstrained(key, num_samples=num_samples)
        return self.problem.parameter_space.constrain(unconstrained)

    def linearized_predict(
        self,
        /,
        *args: Any,
        **kwargs: Any,
    ) -> LinearizedPropagationResult:
        """Propagate dense Laplace covariance through one local prediction map."""
        return propagate_linearized(
            lambda position: self.problem.predict(position, *args, **kwargs),
            self.map_position,
            DenseCovariance(self.covariance),
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
        """Draw and evaluate coherent approximate posterior functions."""
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
        """Draw approximate posterior parameters and conditional measurements."""
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


@overload
def fit_laplace(
    problem: PosteriorProblem,
    map_position: PyTree[Array] | None = None,
    /,
    *,
    curvature: Literal["exact"] = "exact",
    damping: float = 0.0,
    stationarity_tolerance: float | None = 1e-4,
    max_dimension: int = 256,
    prior_precision: float | None = None,
    rank: int = 20,
    key: Array | None = None,
    tolerance: float = 1e-6,
    mv_jit: bool = True,
    likelihood_curvature: Literal["hessian", "ggn"] = "hessian",
) -> LaplaceResult: ...


@overload
def fit_laplace(
    problem: PosteriorProblem,
    map_position: PyTree[Array] | None = None,
    /,
    *,
    curvature: Literal["full", "diagonal", "lanczos", "lobpcg"],
    damping: float = 0.0,
    stationarity_tolerance: float | None = 1e-4,
    max_dimension: int = 256,
    prior_precision: float | None = None,
    rank: int = 20,
    key: Array | None = None,
    tolerance: float = 1e-6,
    mv_jit: bool = True,
    likelihood_curvature: Literal["hessian", "ggn"] = "hessian",
) -> StructuredLaplaceResult: ...


def fit_laplace(
    problem: PosteriorProblem,
    map_position: PyTree[Array] | None = None,
    /,
    *,
    curvature: Literal["exact", "full", "diagonal", "lanczos", "lobpcg"] = "exact",
    damping: float = 0.0,
    stationarity_tolerance: float | None = 1e-4,
    max_dimension: int = 256,
    prior_precision: float | None = None,
    rank: int = 20,
    key: Array | None = None,
    tolerance: float = 1e-6,
    mv_jit: bool = True,
    likelihood_curvature: Literal["hessian", "ggn"] = "hessian",
) -> LaplaceResult | StructuredLaplaceResult:
    """Fit an exact dense or Laplax structured posterior approximation."""
    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    if curvature not in ("exact", "full", "diagonal", "lanczos", "lobpcg"):
        raise ValueError(f"Unknown Laplace curvature {curvature!r}.")
    if curvature != "exact":
        if float(damping) != 0.0:
            raise ValueError(
                "Structured Laplax curvature uses prior_precision, not dense damping."
            )
        from ._laplax_backend import fit_laplax

        return fit_laplax(
            problem,
            map_position,
            curvature=curvature,
            prior_precision=prior_precision,
            rank=rank,
            key=key,
            tolerance=tolerance,
            stationarity_tolerance=stationarity_tolerance,
            mv_jit=mv_jit,
            likelihood_curvature=likelihood_curvature,
        )
    if likelihood_curvature != "hessian":
        raise ValueError(
            "likelihood_curvature='ggn' is available for structured Laplace only."
        )
    if prior_precision is not None:
        raise ValueError(
            "Exact curvature includes the declared prior; do not pass prior_precision."
        )
    damping_value = float(damping)
    if not jnp.isfinite(damping_value) or damping_value < 0.0:
        raise ValueError("damping must be finite and non-negative.")
    dimension_limit = int(max_dimension)
    if dimension_limit <= 0:
        raise ValueError("max_dimension must be positive.")
    if stationarity_tolerance is not None and float(stationarity_tolerance) < 0.0:
        raise ValueError("stationarity_tolerance must be non-negative or None.")

    position = problem.initial_position if map_position is None else map_position
    problem.parameter_space.constrain(position)
    flat_position, unravel = ravel_pytree(position)
    dimension = int(flat_position.size)
    if dimension == 0:
        raise ValueError("Laplace position must contain at least one scalar.")
    if dimension > dimension_limit:
        raise ValueError(
            f"Dense Laplace dimension {dimension} exceeds max_dimension={dimension_limit}."
        )

    objective = lambda flat: problem.negative_log_density(unravel(flat))
    flat_gradient = jax.grad(objective)(flat_position)
    gradient = unravel(flat_gradient)
    gradient_norm = jnp.linalg.norm(flat_gradient)
    if not bool(jnp.all(jnp.isfinite(flat_gradient))):
        raise FloatingPointError("Laplace gradient must be finite.")
    if stationarity_tolerance is not None and float(gradient_norm) > float(
        stationarity_tolerance
    ):
        raise LaplaceCurvatureError(
            "Laplace center is not stationary: "
            f"gradient_norm={float(gradient_norm):.6g}, "
            f"tolerance={float(stationarity_tolerance):.6g}."
        )

    hessian = jax.hessian(objective)(flat_position)
    raw_precision = 0.5 * (hessian + hessian.T)
    if not bool(jnp.all(jnp.isfinite(raw_precision))):
        raise FloatingPointError("Laplace curvature must be finite.")
    raw_eigenvalues = jnp.linalg.eigvalsh(raw_precision)
    precision = raw_precision + damping_value * jnp.eye(
        dimension, dtype=raw_precision.dtype
    )
    eigenvalues = jnp.linalg.eigvalsh(precision)
    minimum = float(jnp.min(eigenvalues))
    if not minimum > 0.0:
        raw_minimum = float(jnp.min(raw_eigenvalues))
        suggested = max(0.0, -raw_minimum + jnp.finfo(raw_precision.dtype).eps)
        raise LaplaceCurvatureError(
            "Laplace precision is not positive definite: "
            f"minimum_eigenvalue={minimum:.6g}, "
            f"raw_minimum_eigenvalue={raw_minimum:.6g}, "
            f"explicit damping must exceed {float(suggested):.6g}."
        )

    cholesky = jnp.linalg.cholesky(precision)
    identity = jnp.eye(dimension, dtype=precision.dtype)
    covariance = jsp.linalg.cho_solve((cholesky, True), identity)
    scale = jsp.linalg.solve_triangular(cholesky.T, identity, lower=False)
    jax.block_until_ready(scale)
    return LaplaceResult(
        problem=problem,
        map_position=position,
        flat_map_position=flat_position,
        gradient=gradient,
        gradient_norm=gradient_norm,
        raw_precision=raw_precision,
        precision=precision,
        covariance=covariance,
        scale=scale,
        raw_eigenvalues=raw_eigenvalues,
        eigenvalues=eigenvalues,
        damping=jnp.asarray(damping_value, dtype=precision.dtype),
        unravel=unravel,
    )


__all__ = ["LaplaceCurvatureError", "LaplaceResult", "fit_laplace"]
