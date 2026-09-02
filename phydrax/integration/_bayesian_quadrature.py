#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jax import core as jax_core
from jaxtyping import Array, ArrayLike, Key

from phydrax.kernels import ScaleKernel, SquaredExponentialKernel

from .._doc import DOC_KEY0
from .._frozendict import frozendict
from .._sampling import design_name
from .._strict import StrictModule
from ..domain._structure import PointBatch, PointSampling, SampleLayout
from ..linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSolveStatus,
    LinearSystem,
    MixedPrecisionPolicy,
    OperatorProperties,
    prepare,
    solve,
)
from ._batches import PointIntegrationBatch
from ._estimates import (
    BayesianQuadratureDiagnostics,
    IntegrationEstimate,
    IntegrationProvenance,
)
from ._kernel_mean_bq import (
    FixedBayesianQuadratureDesign,
    SequentialBayesianQuadratureDesign,
)
from ._kernel_means import AbstractKernelMean
from ._precision import IntegrationPrecisionPolicy
from ._status import IntegrationStatus
from ._targets import ProbabilityTarget


def _is_phydrax_normal(distribution: Any, /) -> bool:
    distribution_type = type(distribution)
    return (
        distribution_type.__module__ == "phydrax.uq._distributions"
        and distribution_type.__name__ == "Normal"
    )


class GaussianKernelMean(AbstractKernelMean):
    """Analytic squared-exponential kernel mean for one Gaussian expectation.

    The current probability target is scalar, so the represented diagonal Gaussian
    has one coordinate. The vector formulas are retained internally so coordinate
    shape is checked explicitly rather than silently broadcast.
    """

    kernel: SquaredExponentialKernel | ScaleKernel
    location: Array
    scale: Array
    target_id: str = eqx.field(static=True)
    target_mass: Array
    normalized: bool = eqx.field(static=True)
    exactness: str = eqx.field(static=True)
    hypotheses: str = eqx.field(static=True)
    probability_label: str = eqx.field(static=True)
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
            isinstance(kernel, ScaleKernel) and isinstance(kernel.kernel, ScaleKernel)
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
        self.scale = jnp.reshape(distribution.scale, (dimension,))
        self.target_id = target.target_id
        self.probability_label = target.probability.label
        self.dimension = dimension
        self.target_mass = jnp.asarray(1.0, dtype=self.location.dtype)
        self.normalized = True
        self.exactness = "analytic"
        self.hypotheses = "normalized diagonal Gaussian and squared-exponential kernel"

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
            jnp.asarray(self.scale, dtype=dtype),
            amplitude,
        )

    def _design(self, points: ArrayLike, name: str, /) -> Array:
        values = jnp.asarray(points)
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        if self.dimension == 1 and values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[1] != self.dimension:
            raise ValueError(
                f"{name} must have shape (num_points, dimension); "
                f"expected dimension {self.dimension}, got {values.shape}."
            )
        return eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            f"{name} must contain only finite values.",
        )

    @staticmethod
    def _standardized_difference(
        left: Array,
        right: Array,
        reference_scale: Array,
        normalized_scale: Array,
        /,
    ) -> Array:
        left_negative = jnp.signbit(left)
        same_sign = left_negative == jnp.signbit(right)
        sign = jnp.where(
            left_negative,
            jnp.asarray(-1.0, dtype=left.dtype),
            jnp.asarray(1.0, dtype=left.dtype),
        )
        left_magnitude = jnp.abs(left)
        right_magnitude = jnp.abs(right)
        same_sign_difference = sign * (left_magnitude - right_magnitude) / reference_scale
        opposite_sign_difference = sign * (
            left_magnitude / reference_scale + right_magnitude / reference_scale
        )
        return (
            jnp.where(
                same_sign,
                same_sign_difference,
                opposite_sign_difference,
            )
            / normalized_scale
        )

    def matrix(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Evaluate the supported kernel without changing the operand dtype."""
        left_values = self._design(left, "left kernel points")
        right_values = self._design(right, "right kernel points")
        if left_values.dtype != right_values.dtype:
            raise ValueError("Kernel point designs must have equal dtypes.")
        length_scale, _, _, amplitude = self._parameters(left_values.dtype)
        standardized = self._standardized_difference(
            left_values[:, None, :],
            right_values[None, :, :],
            length_scale,
            jnp.ones_like(length_scale),
        )
        return amplitude * jnp.exp(-0.5 * jnp.sum(standardized**2, axis=-1))

    def mean(self, points: ArrayLike, /) -> Array:
        """Evaluate ∫ k(x, z) dP(z) at a point design."""
        values = self._design(points, "kernel-mean points")
        length_scale, location, normal_scale, amplitude = self._parameters(values.dtype)
        reference_scale = jnp.maximum(length_scale, normal_scale)
        normalized_length = length_scale / reference_scale
        normalized_normal = normal_scale / reference_scale
        normalized_combined = jnp.sqrt(normalized_length**2 + normalized_normal**2)
        normalization = jnp.prod(normalized_length / normalized_combined)
        standardized = self._standardized_difference(
            values,
            location,
            reference_scale,
            normalized_combined,
        )
        exponent = -0.5 * jnp.sum(standardized**2, axis=1)
        return amplitude * normalization * jnp.exp(exponent)

    def _double_mean(self, dtype: Any, /) -> Array:
        length_scale, _, normal_scale, amplitude = self._parameters(dtype)
        reference_scale = jnp.maximum(length_scale, normal_scale)
        normalized_length = length_scale / reference_scale
        normalized_normal = normal_scale / reference_scale
        normalization = jnp.prod(
            normalized_length
            / jnp.sqrt(normalized_length**2 + 2.0 * normalized_normal**2)
        )
        return amplitude * normalization

    def double_mean(self, /) -> Array:
        """Evaluate ∬ k(x, z) dP(x) dP(z)."""
        return self._double_mean(jnp.result_type(self.location, self.scale))

    def __call__(self, points: ArrayLike, /) -> Array:
        return self.mean(points)


class BayesianQuadraturePlan(StrictModule):
    """Bounded fixed or sequential quadrature for one declared kernel mean."""

    kernel_mean: AbstractKernelMean
    design: (
        PointSampling | FixedBayesianQuadratureDesign | SequentialBayesianQuadratureDesign
    )
    observation_noise: Array
    solve_regularization: Array
    solve_policy: LinearSolvePolicy
    max_points: int = eqx.field(static=True)

    def __init__(
        self,
        kernel_mean: AbstractKernelMean,
        design: (
            PointSampling
            | FixedBayesianQuadratureDesign
            | SequentialBayesianQuadratureDesign
        ),
        /,
        *,
        observation_noise: ArrayLike = 0.0,
        solve_regularization: ArrayLike = 0.0,
        solve_policy: LinearSolvePolicy | None = None,
        max_points: int = 4096,
    ):
        if not isinstance(kernel_mean, AbstractKernelMean):
            raise TypeError("kernel_mean must implement AbstractKernelMean.")
        if isinstance(design, PointSampling):
            if not isinstance(design.count, int) or design.count < 1:
                raise ValueError(
                    "Bayesian quadrature PointSampling count must be positive."
                )
            if design.layout is not None:
                raise ValueError(
                    "Bayesian quadrature owns its scalar probability sample layout; "
                    "PointSampling.layout must be None."
                )
            design_count = design.count
        elif isinstance(design, FixedBayesianQuadratureDesign):
            design_count = design.count
        elif isinstance(design, SequentialBayesianQuadratureDesign):
            design_count = design.total_count
        else:
            raise TypeError(
                "design must be PointSampling, FixedBayesianQuadratureDesign, or "
                "SequentialBayesianQuadratureDesign."
            )
        limit = int(max_points)
        if limit < 1:
            raise ValueError("max_points must be positive.")
        if design_count > limit:
            raise ValueError(
                f"Bayesian quadrature design has {design_count} points, exceeding "
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
                DenseCholesky(),
                failure=FailurePolicy("status"),
            )
            if solve_policy is None
            else solve_policy
        )
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("solve_policy must be a LinearSolvePolicy or None.")
        if not isinstance(policy.method, DenseCholesky):
            raise TypeError(
                "Bayesian quadrature accepts only a DenseCholesky solve policy."
            )
        if (
            policy.preconditioning is not None
            or policy.recycling is not None
            or policy.rank.relative_cutoff is not None
        ):
            raise ValueError(
                "Bayesian quadrature DenseCholesky does not accept preconditioning, "
                "recycling, or a rank cutoff."
            )
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
    if target.probability.label != plan.kernel_mean.probability_label:
        raise ValueError(
            "GaussianKernelMean probability label does not match the integration target."
        )
    distribution = target.probability.distribution
    if not _is_phydrax_normal(distribution):
        raise TypeError("Bayesian quadrature currently requires a Gaussian target.")
    binding_mismatch = (distribution.location != plan.kernel_mean.location[0]) | (
        distribution.scale != plan.kernel_mean.scale[0]
    )
    binding_message = (
        "GaussianKernelMean probability content does not match the integration target."
    )
    if isinstance(binding_mismatch, jax_core.Tracer):
        binding_anchor = eqx.error_if(
            jnp.asarray(distribution.location),
            binding_mismatch,
            binding_message,
        )
    elif bool(binding_mismatch):
        raise ValueError(binding_message)
    else:
        binding_anchor = jnp.asarray(distribution.location)
    policy = IntegrationPrecisionPolicy() if precision is None else precision
    if not isinstance(policy, IntegrationPrecisionPolicy):
        raise TypeError("precision must be an IntegrationPrecisionPolicy.")
    point_batch = _materialize_points(target, plan, key)
    label = target.probability.label
    point_values = (
        point_batch.points[label].data
        + jnp.zeros_like(point_batch.points[label].data) * binding_anchor
    )
    evaluation_design = policy.evaluation(point_values)
    solve_design = policy.accumulation(evaluation_design)
    if plan.kernel_mean.dimension != 1:
        raise ValueError(
            "Kernel-mean dimension does not match the scalar target dimension."
        )
    solve_itemsize = jnp.dtype(solve_design.dtype).itemsize
    linear_precision = plan.solve_policy.precision
    effective_solve_policy = plan.solve_policy
    if linear_precision is None:
        linear_precision = MixedPrecisionPolicy(
            operator_dtype=solve_design.dtype,
            factorization_dtype=solve_design.dtype,
            residual_dtype=solve_design.dtype,
            accumulation_dtype=solve_design.dtype,
        )
        effective_solve_policy = eqx.tree_at(
            lambda selected: selected.precision,
            plan.solve_policy,
            linear_precision,
            is_leaf=lambda value: value is None,
        )
    factorization_dtype = (
        solve_design.dtype
        if linear_precision is None or linear_precision.factorization_dtype is None
        else jnp.dtype(linear_precision.factorization_dtype)
    )
    supported_solve_dtypes = (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64))
    if (
        jnp.dtype(solve_design.dtype) not in supported_solve_dtypes
        or jnp.dtype(factorization_dtype) not in supported_solve_dtypes
    ):
        raise TypeError(
            "Bayesian quadrature DenseCholesky requires float32 or float64 solve and "
            "factorization dtypes; no kernel matrix was allocated."
        )
    entries = solve_design.shape[0] * solve_design.shape[0]
    if linear_precision is not None:
        stage_dtypes = (
            ("operator_dtype", linear_precision.operator_dtype),
            ("residual_dtype", linear_precision.residual_dtype),
            ("accumulation_dtype", linear_precision.accumulation_dtype),
        )
        mismatched_stages = tuple(
            name
            for name, dtype in stage_dtypes
            if dtype is not None and jnp.dtype(dtype) != jnp.dtype(solve_design.dtype)
        )
        if mismatched_stages:
            raise ValueError(
                "Bayesian quadrature DenseCholesky precision stages "
                f"{mismatched_stages!r} must match integration accumulation dtype "
                f"{solve_design.dtype}; no kernel matrix was allocated."
            )
        if (
            linear_precision.preconditioner_dtype is not None
            or linear_precision.krylov_dtype is not None
        ):
            raise ValueError(
                "Bayesian quadrature DenseCholesky has no preconditioner or Krylov "
                "precision stage; no kernel matrix was allocated."
            )
        lower_factorization = (
            jnp.dtype(factorization_dtype).itemsize
            < jnp.dtype(solve_design.dtype).itemsize
        )
        if (
            jnp.dtype(factorization_dtype).itemsize
            > jnp.dtype(solve_design.dtype).itemsize
        ):
            raise ValueError(
                "Bayesian quadrature DenseCholesky factorization precision cannot exceed "
                "the solve dtype; no kernel matrix was allocated."
            )
        if (
            linear_precision.maximum_refinement_steps > 0
            or linear_precision.condition_limit is not None
        ) and not lower_factorization:
            raise ValueError(
                "Bayesian quadrature DenseCholesky refinement requires a lower "
                "factorization dtype; no kernel matrix was allocated."
            )
    factorization_bytes = entries * jnp.dtype(factorization_dtype).itemsize
    gram_workspace_bytes = 2 * entries * solve_itemsize
    resources = effective_solve_policy.resources
    if (
        factorization_bytes > resources.factorization_bytes
        or gram_workspace_bytes > resources.workspace_bytes
    ):
        raise ValueError(
            "Bayesian quadrature kernel system exceeds the dense solve resource "
            "budget; no kernel matrix was allocated."
        )
    kernel_matrix = policy.accumulation(
        plan.kernel_mean.matrix(evaluation_design, evaluation_design)
    )
    kernel_mean = policy.accumulation(plan.kernel_mean.mean(evaluation_design))
    kernel_double_mean = policy.accumulation(
        plan.kernel_mean._double_mean(evaluation_design.dtype)
    )
    observation_noise = policy.accumulation(plan.observation_noise)
    solve_regularization = policy.accumulation(plan.solve_regularization)
    kernel_amplitude = policy.accumulation(
        plan.kernel_mean._parameters(evaluation_design.dtype)[3]
    )
    system_scale = jnp.maximum(
        jnp.abs(kernel_amplitude),
        jnp.maximum(jnp.abs(observation_noise), jnp.abs(solve_regularization)),
    )
    safe_system_scale = jnp.where(
        system_scale > 0.0,
        system_scale,
        jnp.asarray(1.0, dtype=system_scale.dtype),
    )
    normalized_kernel_matrix = kernel_matrix / safe_system_scale
    normalized_diagonal_shift = (
        observation_noise / safe_system_scale + solve_regularization / safe_system_scale
    )
    system_matrix = normalized_kernel_matrix + normalized_diagonal_shift * jnp.eye(
        solve_design.shape[0], dtype=kernel_matrix.dtype
    )
    normalized_kernel_mean = kernel_mean / safe_system_scale
    prepared = prepare(
        LinearSystem(
            DenseLinearOperator(
                system_matrix,
                properties=OperatorProperties(
                    self_adjoint=True,
                    positive_definite=True,
                    evidence={
                        "self_adjoint": "construction",
                        "positive_definite": "construction",
                    },
                ),
            ),
            problem_id=f"bayesian-quadrature:{target.target_id}",
        ),
        effective_solve_policy,
    )
    solve_result = solve(prepared, normalized_kernel_mean)
    weights = policy.accumulation(solve_result.value)
    decision_kernel_mean = policy.decision(kernel_mean)
    decision_weights = policy.decision(weights)
    decision_double_mean = policy.decision(kernel_double_mean)
    contracted = oe.contract("i,i->", decision_kernel_mean, decision_weights)
    posterior_variance = decision_double_mean - contracted
    arithmetic_epsilon = max(
        jnp.finfo(evaluation_design.dtype).eps,
        jnp.finfo(kernel_mean.dtype).eps,
        jnp.finfo(posterior_variance.dtype).eps,
        jnp.finfo(jnp.dtype(factorization_dtype)).eps,
    )
    variance_scale = jnp.maximum(
        jnp.maximum(jnp.abs(decision_double_mean), jnp.abs(contracted)),
        jnp.asarray(jnp.finfo(posterior_variance.dtype).tiny),
    )
    envelope = (
        jnp.asarray(arithmetic_epsilon, dtype=posterior_variance.dtype)
        * (8 * solve_design.shape[0] + 16)
        * variance_scale
    )
    return BayesianQuadratureBatch(
        point_batch,
        weights,
        kernel_mean,
        kernel_double_mean,
        posterior_variance,
        envelope,
        observation_noise,
        solve_regularization,
        solve_result,
        kernel_id=plan.kernel_mean.kernel.kernel_id,
    )


def _evaluation_batch(
    batch: PointIntegrationBatch,
    target: ProbabilityTarget,
    policy: IntegrationPrecisionPolicy,
    /,
) -> PointIntegrationBatch:
    label = target.probability.label
    source_points = batch.points
    source_field = source_points[label]
    points = PointBatch(
        frozendict(
            {
                label: cx.Field(
                    policy.evaluation(source_field.data),
                    dims=source_field.dims,
                )
            }
        ),
        source_points.structure,
        metadata=source_points.metadata,
    )
    return PointIntegrationBatch(
        points,
        batch.weights,
        axes=batch.axes,
        mask=batch.mask,
        target_mass=batch.target_mass,
        stratum_indices=batch.stratum_indices,
        num_strata=batch.num_strata,
        provenance=batch.provenance,
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
    evaluation_batch = _evaluation_batch(batch.points, target, policy)
    values, factors, normalizer, _, output_dims = _sample_values(
        integrand,
        target,
        evaluation_batch,
        key=key,
        kwargs=callback_kwargs,
        precision=policy,
    )
    if normalizer is None:
        raise RuntimeError("Bayesian quadrature requires a normalized target.")
    weighted_values = values * factors.reshape(
        (factors.shape[0],) + (1,) * (values.ndim - 1)
    )
    value = oe.contract("i,i...->...", batch.weights, weighted_values)
    finite_inputs = jnp.all(jnp.isfinite(values)) & jnp.all(jnp.isfinite(factors))
    finite_value = jnp.all(jnp.isfinite(value))
    solve_success = batch.solve_result.status == int(LinearSolveStatus.SUCCESS)
    variance_valid = jnp.isfinite(batch.posterior_variance) & (
        batch.posterior_variance >= -batch.variance_roundoff_envelope
    )
    status = jnp.where(
        variance_valid,
        int(IntegrationStatus.CONVERGED),
        int(IntegrationStatus.INVALID_POSTERIOR_VARIANCE),
    )
    status = jnp.where(
        finite_value,
        status,
        int(IntegrationStatus.NONFINITE_INTEGRAND),
    )
    status = jnp.where(
        solve_success,
        status,
        int(IntegrationStatus.LINEAR_SOLVE_FAILED),
    )
    status = jnp.where(
        finite_inputs,
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
