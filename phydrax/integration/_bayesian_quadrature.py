#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import PointBatch, PointSampling, SampleLayout
from phydrax.kernels import ScaleKernel, SquaredExponentialKernel

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._sampling import design_name
from .._strict import StrictModule
from ..linalg import (
    DenseLU,
    DenseLinearOperator,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSystem,
    prepare,
    solve,
)
from ._batches import PointIntegrationBatch
from ._estimates import (
    BayesianQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._precision import IntegrationPrecisionPolicy
from ._status import IntegrationStatus
from ._targets import ProbabilityTarget


def _is_phydrax_normal(distribution: Any, /) -> bool:
    distribution_type = type(distribution)
    return (
        distribution_type.__module__ == "phydrax.uq._distributions"
        and distribution_type.__name__ == "Normal"
    )


class GaussianKernelMean(StrictModule):
    """Analytic squared-exponential kernel mean for one Gaussian expectation.

    The current probability target is scalar, so the represented diagonal Gaussian
    has one coordinate. The vector formulas are retained internally so coordinate
    shape is checked explicitly rather than silently broadcast.
    """

    kernel: SquaredExponentialKernel | ScaleKernel
    location: Array
    variance: Array
    target_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        target: ProbabilityTarget,
        kernel: SquaredExponentialKernel | ScaleKernel,
        /,
    ):
        if not isinstance(target, ProbabilityTarget) or not target.normalized:
            raise TypeError(
                "GaussianKernelMean requires one normalized ProbabilityTarget."
            )
        distribution = target.probability.distribution
        if not _is_phydrax_normal(distribution):
            raise TypeError(
                "GaussianKernelMean currently requires a phydrax.uq.Normal measure."
            )
        base = kernel.kernel if isinstance(kernel, ScaleKernel) else kernel
        if not isinstance(base, SquaredExponentialKernel) or (
            isinstance(kernel, ScaleKernel)
            and isinstance(kernel.kernel, ScaleKernel)
        ):
            raise TypeError(
                "GaussianKernelMean supports SquaredExponentialKernel, optionally "
                "wrapped once in ScaleKernel."
            )
        dimension = 1
        if base.length_scale.ndim == 1 and base.length_scale.shape[0] not in (
            1,
            dimension,
        ):
            raise ValueError(
                "Kernel length_scale dimension does not match the Gaussian measure."
            )
        self.kernel = kernel
        self.location = jnp.reshape(distribution.location, (dimension,))
        self.variance = jnp.reshape(distribution.variance, (dimension,))
        self.target_id = target.target_id
        self.dimension = dimension

    def _parameters(self, dtype: Any, /) -> tuple[Array, Array, Array, Array]:
        base = self.kernel.kernel if isinstance(self.kernel, ScaleKernel) else self.kernel
        length_scale = jnp.broadcast_to(
            jnp.asarray(base.length_scale, dtype=dtype), (self.dimension,)
        )
        amplitude = (
            jnp.asarray(self.kernel.scale, dtype=dtype)
            if isinstance(self.kernel, ScaleKernel)
            else jnp.asarray(1.0, dtype=dtype)
        )
        return (
            length_scale,
            jnp.asarray(self.location, dtype=dtype),
            jnp.asarray(self.variance, dtype=dtype),
            amplitude,
        )

    def mean(self, points: ArrayLike, /) -> Array:
        """Evaluate ∫ k(x, z) dP(z) at a point design."""
        values = jnp.asarray(points, dtype=float)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Kernel-mean points must contain only finite values.",
        )
        if self.dimension == 1 and values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                "Kernel-mean points must have shape (num_points, dimension); "
                f"expected dimension {self.dimension}, got {values.shape}."
            )
        length_scale, location, variance, amplitude = self._parameters(values.dtype)
        denominator = length_scale * length_scale + variance
        normalization = jnp.prod(length_scale / jnp.sqrt(denominator))
        exponent = -0.5 * jnp.sum((values - location) ** 2 / denominator, axis=1)
        return amplitude * normalization * jnp.exp(exponent)

    def double_mean(self, /) -> Array:
        """Evaluate ∬ k(x, z) dP(x) dP(z)."""
        dtype = jnp.result_type(self.location, self.variance)
        length_scale, _, variance, amplitude = self._parameters(dtype)
        denominator = length_scale * length_scale + 2.0 * variance
        return amplitude * jnp.prod(length_scale / jnp.sqrt(denominator))

    def __call__(self, points: ArrayLike, /) -> Array:
        return self.mean(points)


class BayesianQuadraturePlan(StrictModule):
    """Fixed-design GP quadrature for one bound Gaussian probability target."""

    kernel_mean: GaussianKernelMean
    design: PointSampling
    observation_noise: Array
    solve_regularization: Array
    solve_policy: LinearSolvePolicy
    max_points: int = eqx.field(static=True)

    def __init__(
        self,
        kernel_mean: GaussianKernelMean,
        design: PointSampling,
        /,
        *,
        observation_noise: ArrayLike = 0.0,
        solve_regularization: ArrayLike = 0.0,
        solve_policy: LinearSolvePolicy | None = None,
        max_points: int = 4096,
    ):
        if not isinstance(kernel_mean, GaussianKernelMean):
            raise TypeError("kernel_mean must be a GaussianKernelMean.")
        if not isinstance(design, PointSampling):
            raise TypeError("Bayesian quadrature requires a fixed PointSampling design.")
        if not isinstance(design.count, int) or design.count < 1:
            raise ValueError("Bayesian quadrature PointSampling count must be positive.")
        if design.layout is not None:
            raise ValueError(
                "Bayesian quadrature owns its scalar probability sample layout; "
                "PointSampling.layout must be None."
            )
        limit = int(max_points)
        if limit < 1:
            raise ValueError("max_points must be positive.")
        if design.count > limit:
            raise ValueError(
                f"Bayesian quadrature design has {design.count} points, exceeding "
                f"max_points={limit}; no kernel matrix was allocated."
            )
        noise = jnp.asarray(observation_noise, dtype=float)
        regularization = jnp.asarray(solve_regularization, dtype=float)
        if noise.ndim != 0 or regularization.ndim != 0:
            raise ValueError("Observation noise and solve regularization must be scalar.")
        self.observation_noise = eqx.error_if(
            noise,
            ~jnp.isfinite(noise) | (noise < 0.0),
            "observation_noise must be finite and nonnegative.",
        )
        self.solve_regularization = eqx.error_if(
            regularization,
            ~jnp.isfinite(regularization) | (regularization < 0.0),
            "solve_regularization must be finite and nonnegative.",
        )
        policy = (
            LinearSolvePolicy(
                DenseLU(),
                failure=FailurePolicy("status"),
            )
            if solve_policy is None
            else solve_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("solve_policy must be a LinearSolvePolicy or None.")
        if policy.failure.mode != "status":
            raise ValueError(
                "Bayesian quadrature requires solve failure='status' to retain child "
                "solve evidence."
            )
        self.kernel_mean = kernel_mean
        self.design = design
        self.solve_policy = policy
        self.max_points = limit


class BayesianQuadratureBatch(StrictModule):
    """Prepared fixed design, kernel solve, and posterior integral variance."""

    points: PointIntegrationBatch
    weights: Array
    kernel_mean: Array
    kernel_double_mean: Array
    posterior_variance: Array
    variance_roundoff_envelope: Array
    observation_noise: Array
    solve_regularization: Array
    solve_result: LinearSolveResult
    kernel_id: str = eqx.field(static=True)


def _materialize_points(
    target: ProbabilityTarget,
    plan: BayesianQuadraturePlan,
    key: Key[Array, ""],
    /,
) -> PointIntegrationBatch:
    probability = target.probability
    count = plan.design.count
    sampler = design_name(plan.design.design)
    values = probability.sample(count, sampler=sampler, key=key)
    structure = SampleLayout(((probability.label,),)).canonicalize((probability.label,))
    axis = structure.axis_for(probability.label)
    if axis is None:
        raise RuntimeError("Bayesian quadrature probability layout has no sample axis.")
    points = PointBatch(
        frozendict(
            {
                probability.label: cx.Field(
                    jnp.asarray(values).reshape((count,)), dims=(axis,)
                )
            }
        ),
        structure,
    )
    neutral = cx.Field(jnp.ones((count,), dtype=jnp.asarray(values).dtype), dims=(axis,))
    return PointIntegrationBatch(
        points,
        neutral,
        axes=(axis,),
        target_mass=jnp.asarray(1.0, dtype=neutral.data.dtype),
        provenance=f"bayesian-quadrature:{sampler}",
    )


def materialize_bayesian_quadrature(
    target: ProbabilityTarget,
    plan: BayesianQuadraturePlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    precision: IntegrationPrecisionPolicy | None = None,
) -> BayesianQuadratureBatch:
    """Prepare the fixed kernel system before any integrand is evaluated."""
    if not isinstance(target, ProbabilityTarget) or not target.normalized:
        raise TypeError("BayesianQuadraturePlan requires a normalized ProbabilityTarget.")
    if target.target_id != plan.kernel_mean.target_id:
        raise ValueError(
            "GaussianKernelMean target identity does not match the integration target."
        )
    if not _is_phydrax_normal(target.probability.distribution):
        raise TypeError("Bayesian quadrature currently requires a Gaussian target.")
    policy = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(policy, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    point_batch = _materialize_points(target, plan, key)
    label = target.probability.label
    point_values = point_batch.points[label].data
    design = policy.accumulation(policy.evaluation(point_values))
    if plan.kernel_mean.dimension != 1:
        raise ValueError(
            "Kernel-mean dimension does not match the scalar target dimension."
        )
    if isinstance(plan.solve_policy.method, DenseLU):
        kernel_matrix_bytes = (
            design.shape[0] * design.shape[0] * jnp.dtype(design.dtype).itemsize
        )
        resources = plan.solve_policy.resources
        if (
            kernel_matrix_bytes > resources.factorization_bytes
            or kernel_matrix_bytes > resources.workspace_bytes
        ):
            raise ValueError(
                "Bayesian quadrature kernel system exceeds the dense solve resource "
                "budget; no kernel matrix was allocated."
            )
    kernel_matrix = policy.accumulation(
        policy.evaluation(plan.kernel_mean.kernel.matrix(design, design))
    )
    kernel_mean = policy.accumulation(
        policy.evaluation(plan.kernel_mean.mean(design))
    )
    kernel_double_mean = policy.accumulation(
        policy.evaluation(plan.kernel_mean.double_mean())
    )
    diagonal_shift = policy.accumulation(
        plan.observation_noise + plan.solve_regularization
    )
    system_matrix = kernel_matrix + diagonal_shift * jnp.eye(
        design.shape[0], dtype=kernel_matrix.dtype
    )
    prepared = prepare(
        LinearSystem(
            DenseLinearOperator(system_matrix),
            problem_id=f"bayesian-quadrature:{target.target_id}",
        ),
        plan.solve_policy,
    )
    solve_result = solve(prepared, kernel_mean)
    weights = policy.accumulation(solve_result.value)
    contracted = oe.contract("i,i->", kernel_mean, weights)
    posterior_variance = policy.decision(kernel_double_mean - contracted)
    scale = jnp.abs(kernel_double_mean) + jnp.abs(contracted) + 1.0
    envelope = policy.decision(
        jnp.finfo(posterior_variance.dtype).eps
        * (8 * design.shape[0] + 16)
        * scale
    )
    return BayesianQuadratureBatch(
        point_batch,
        weights,
        kernel_mean,
        kernel_double_mean,
        posterior_variance,
        envelope,
        policy.accumulation(plan.observation_noise),
        policy.accumulation(plan.solve_regularization),
        solve_result,
        kernel_id=plan.kernel_mean.kernel.kernel_id,
    )


def integrate_bayesian_quadrature(
    integrand: Any,
    target: ProbabilityTarget,
    batch: BayesianQuadratureBatch,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
    kwargs: dict[str, Any] | None = None,
    precision: IntegrationPrecisionPolicy | None = None,
) -> IntegrationEstimate:
    """Apply prepared GP quadrature weights to one observable output leaf."""
    if not isinstance(batch, BayesianQuadratureBatch):
        raise TypeError("Expected a materialized BayesianQuadratureBatch.")
    policy = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(policy, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    from ._monte_carlo import _sample_values

    callback_kwargs = {} if kwargs is None else kwargs
    values, factors, normalizer, _, output_dims = _sample_values(
        integrand,
        target,
        batch.points,
        key=key,
        kwargs=callback_kwargs,
        precision=policy,
    )
    if normalizer is None:
        raise RuntimeError("Bayesian quadrature requires a normalized target.")
    finite_integrand = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(factors))
    weighted_values = values * factors.reshape(
        (factors.shape[0],) + (1,) * (values.ndim - 1)
    )
    value = oe.contract("i,i...->...", batch.weights, weighted_values)
    solve_success = batch.solve_result.status == int(LinearSolveStatus.SUCCESS)
    variance_valid = (
        jnp.isfinite(batch.posterior_variance)
        & (batch.posterior_variance >= -batch.variance_roundoff_envelope)
    )
    status = jnp.where(
        variance_valid,
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.INVALID_POSTERIOR_VARIANCE),
    )
    status = jnp.where(
        solve_success,
        status,
        int(IntegrationStatus.LINEAR_SOLVE_FAILED),
    )
    status = jnp.where(
        finite_integrand,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    successful = status == int(IntegrationStatus.CONVERGED)
    value = jnp.where(successful, value, jnp.full_like(value, jnp.nan))
    nonnegative_variance = jnp.where(
        batch.posterior_variance < 0.0,
        jnp.asarray(0.0, dtype=batch.posterior_variance.dtype),
        batch.posterior_variance,
    )
    posterior_sd = jnp.where(
        successful,
        jnp.sqrt(nonnegative_variance),
        jnp.asarray(jnp.nan, dtype=batch.posterior_variance.dtype),
    )
    diagnostics = BayesianQuadratureDiagnostics(
        status=status,
        num_evaluations=jnp.asarray(values.shape[0], dtype=jnp.int32),
        posterior_variance=batch.posterior_variance,
        variance_roundoff_envelope=batch.variance_roundoff_envelope,
        kernel_mean=batch.kernel_mean,
        kernel_double_mean=batch.kernel_double_mean,
        observation_noise=batch.observation_noise,
        solve_regularization=batch.solve_regularization,
        solve=batch.solve_result,
        target_id=target.target_id,
        kernel_id=batch.kernel_id,
    )
    return IntegrationEstimate(
        cx.Field(value, dims=output_dims),
        status=status,
        num_evaluations=values.shape[0],
        error_estimate=posterior_sd,
        error_kind="bayesian-posterior-standard-deviation",
        diagnostics=diagnostics,
        provenance=IntegrationProvenance(
            "bayesian-quadrature",
            "probability",
            batch.points.provenance,
        ),
    )




__all__ = [
    "BayesianQuadratureDiagnostics",
    "BayesianQuadraturePlan",
    "GaussianKernelMean",
]
