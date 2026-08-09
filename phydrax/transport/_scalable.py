#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._costs import SquaredEuclideanCost
from ._problem import DiscreteTransportProblem
from ._results import (
    AbstractBalancedTransportPlan,
    AbstractBalancedTransportSolver,
    SinkhornDiagnostics,
    TransportProvenance,
)
from ._status import TransportStatus


class KernelApproximationStatus(IntEnum):
    """JAX-compatible validation status for a positive kernel approximation."""

    VALID = 0
    NONFINITE_FEATURES = 1
    ZERO_KERNEL_ROW = 2
    NONFINITE_PROBE = 3
    PROBE_TOLERANCE_EXCEEDED = 4


class PositiveKernelApproximationDiagnostics(StrictModule):
    """Fixed-structure rank, replay, support, and probe-error evidence."""

    status: Array
    rank: Array
    num_probes: Array
    key_data: Array
    probe_source_indices: Array
    probe_target_indices: Array
    exact_probe_values: Array
    approximate_probe_values: Array
    relative_probe_errors: Array
    relative_probe_error: Array
    maximum_relative_probe_error: Array
    probe_tolerance: Array
    zero_source_rows: Array
    zero_target_rows: Array
    finite_features: Array

    def __init__(
        self,
        *,
        status: ArrayLike,
        rank: ArrayLike,
        num_probes: ArrayLike,
        key_data: ArrayLike,
        probe_source_indices: ArrayLike,
        probe_target_indices: ArrayLike,
        exact_probe_values: ArrayLike,
        approximate_probe_values: ArrayLike,
        relative_probe_errors: ArrayLike,
        relative_probe_error: ArrayLike,
        maximum_relative_probe_error: ArrayLike,
        probe_tolerance: ArrayLike,
        zero_source_rows: ArrayLike,
        zero_target_rows: ArrayLike,
        finite_features: ArrayLike,
    ):
        status_ = jnp.asarray(status, dtype=jnp.int32)
        rank_ = jnp.asarray(rank, dtype=jnp.int32)
        num_probes_ = jnp.asarray(num_probes, dtype=jnp.int32)
        relative_error = jnp.asarray(relative_probe_error, dtype=float)
        maximum_error = jnp.asarray(maximum_relative_probe_error, dtype=float)
        tolerance = jnp.asarray(probe_tolerance, dtype=float)
        zero_source = jnp.asarray(zero_source_rows, dtype=jnp.int32)
        zero_target = jnp.asarray(zero_target_rows, dtype=jnp.int32)
        finite = jnp.asarray(finite_features, dtype=bool)
        scalar_evidence = (
            status_,
            rank_,
            num_probes_,
            relative_error,
            maximum_error,
            tolerance,
            zero_source,
            zero_target,
            finite,
        )
        if any(item.shape != () for item in scalar_evidence):
            raise ValueError("Approximation scalar evidence must contain scalars.")
        key_evidence = jnp.asarray(key_data)
        if key_evidence.shape != (2,):
            raise ValueError("Approximation key evidence must have shape (2,).")
        source_indices = jnp.asarray(probe_source_indices, dtype=jnp.int32)
        target_indices = jnp.asarray(probe_target_indices, dtype=jnp.int32)
        exact_values = jnp.asarray(exact_probe_values, dtype=float)
        approximate_values = jnp.asarray(approximate_probe_values, dtype=float)
        relative_errors = jnp.asarray(relative_probe_errors, dtype=float)
        if source_indices.ndim != 1:
            raise ValueError("Approximation probe indices must be rank one.")
        probe_shape = source_indices.shape
        if probe_shape[0] < 1:
            raise ValueError("Approximation probe evidence must be nonempty.")
        probe_evidence = (
            target_indices,
            exact_values,
            approximate_values,
            relative_errors,
        )
        if any(item.shape != probe_shape for item in probe_evidence):
            raise ValueError("Approximation probe evidence must have one shared shape.")
        status_ = eqx.error_if(
            status_,
            (status_ < int(KernelApproximationStatus.VALID))
            | (status_ > int(KernelApproximationStatus.PROBE_TOLERANCE_EXCEEDED)),
            "Approximation status is invalid.",
        )
        rank_ = eqx.error_if(
            rank_,
            (rank_ < 1) | (num_probes_ != probe_shape[0]),
            "Approximation rank and probe count evidence is invalid.",
        )
        tolerance = eqx.error_if(
            tolerance,
            jnp.isnan(tolerance)
            | (tolerance < 0.0)
            | (zero_source < 0)
            | (zero_target < 0),
            "Approximation tolerance and zero-row evidence is invalid.",
        )
        self.status = status_
        self.rank = rank_
        self.num_probes = num_probes_
        self.key_data = key_evidence
        self.probe_source_indices = source_indices
        self.probe_target_indices = target_indices
        self.exact_probe_values = exact_values
        self.approximate_probe_values = approximate_values
        self.relative_probe_errors = relative_errors
        self.relative_probe_error = relative_error
        self.maximum_relative_probe_error = maximum_error
        self.probe_tolerance = tolerance
        self.zero_source_rows = zero_source
        self.zero_target_rows = zero_target
        self.finite_features = finite

    @property
    def successful(self) -> Array:
        """Whether feature construction passed every declared quality check."""
        return self.status == int(KernelApproximationStatus.VALID)


class PositiveKernelFactors(StrictModule):
    """Validated nonnegative factors for a positive rectangular kernel.

    The represented kernel is
    ``exp(source_log_scale[:, None] + target_log_scale[None, :])
    * (source_factors @ target_factors.T)``. Row log scales allow a
    factorizer to preserve its kernel while keeping every stored factor in a
    numerically useful range.
    """

    source_factors: Array
    target_factors: Array
    source_log_scale: Array
    target_log_scale: Array
    source_points: Array
    target_points: Array
    epsilon: Array
    diagnostics: PositiveKernelApproximationDiagnostics
    factorization_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_factors: ArrayLike,
        target_factors: ArrayLike,
        /,
        *,
        source_log_scale: ArrayLike,
        target_log_scale: ArrayLike,
        source_points: ArrayLike,
        target_points: ArrayLike,
        epsilon: ArrayLike,
        diagnostics: PositiveKernelApproximationDiagnostics,
        factorization_id: str,
    ):
        source = jnp.asarray(source_factors, dtype=float)
        target = jnp.asarray(target_factors, dtype=float)
        if source.ndim != 2 or target.ndim != 2:
            raise ValueError("Positive kernel factors must be rank-two arrays.")
        if source.shape[0] < 1 or target.shape[0] < 1:
            raise ValueError("Positive kernel factors must have nonempty row sets.")
        if source.shape[1] < 1 or source.shape[1] != target.shape[1]:
            raise ValueError(
                "Positive kernel factors must have one shared positive rank."
            )
        source_scale = jnp.asarray(source_log_scale, dtype=source.dtype)
        target_scale = jnp.asarray(target_log_scale, dtype=target.dtype)
        if source_scale.shape != (source.shape[0],):
            raise ValueError("source_log_scale must match the source factor rows.")
        if target_scale.shape != (target.shape[0],):
            raise ValueError("target_log_scale must match the target factor rows.")
        source_design = jnp.asarray(source_points, dtype=source.dtype)
        target_design = jnp.asarray(target_points, dtype=target.dtype)
        if source_design.ndim != 2 or source_design.shape[0] != source.shape[0]:
            raise ValueError("source_points must match the source factor rows.")
        if target_design.ndim != 2 or target_design.shape[0] != target.shape[0]:
            raise ValueError("target_points must match the target factor rows.")
        if source_design.shape[1] != target_design.shape[1]:
            raise ValueError("Factor source and target points must share an event size.")
        if not isinstance(diagnostics, PositiveKernelApproximationDiagnostics):
            raise TypeError(
                "diagnostics must be PositiveKernelApproximationDiagnostics."
            )
        scalar_evidence = (
            diagnostics.status,
            diagnostics.rank,
            diagnostics.num_probes,
            diagnostics.relative_probe_error,
            diagnostics.maximum_relative_probe_error,
            diagnostics.probe_tolerance,
            diagnostics.zero_source_rows,
            diagnostics.zero_target_rows,
            diagnostics.finite_features,
        )
        if any(item.shape != () for item in scalar_evidence):
            raise ValueError("Approximation scalar evidence must contain scalars.")
        if diagnostics.key_data.shape != (2,):
            raise ValueError("Approximation key evidence must have shape (2,).")
        if diagnostics.probe_source_indices.ndim != 1:
            raise ValueError("Approximation probe indices must be rank one.")
        probe_count = diagnostics.probe_source_indices.shape[0]
        if probe_count < 1:
            raise ValueError("Approximation probe evidence must be nonempty.")
        probe_shape = (probe_count,)
        probe_evidence = (
            diagnostics.probe_source_indices,
            diagnostics.probe_target_indices,
            diagnostics.exact_probe_values,
            diagnostics.approximate_probe_values,
            diagnostics.relative_probe_errors,
        )
        if any(item.shape != probe_shape for item in probe_evidence):
            raise ValueError("Approximation probe evidence must have one shared shape.")
        identity = str(factorization_id)
        if not identity:
            raise ValueError("factorization_id must be nonempty.")
        epsilon_ = jnp.asarray(epsilon, dtype=jnp.result_type(source, target)).reshape(())
        source = eqx.error_if(
            source,
            jnp.any(~jnp.isfinite(source)) | jnp.any(source < 0.0),
            "Positive kernel source factors must be finite and nonnegative.",
        )
        target = eqx.error_if(
            target,
            jnp.any(~jnp.isfinite(target)) | jnp.any(target < 0.0),
            "Positive kernel target factors must be finite and nonnegative.",
        )
        source_scale = eqx.error_if(
            source_scale,
            jnp.any(~jnp.isfinite(source_scale)),
            "Positive kernel source log scales must be finite.",
        )
        target_scale = eqx.error_if(
            target_scale,
            jnp.any(~jnp.isfinite(target_scale)),
            "Positive kernel target log scales must be finite.",
        )
        epsilon_ = eqx.error_if(
            epsilon_,
            ~jnp.isfinite(epsilon_) | (epsilon_ <= 0.0),
            "Positive kernel epsilon must be finite and positive.",
        )
        source = eqx.error_if(
            source,
            diagnostics.rank != source.shape[1],
            "Approximation rank evidence must match the factor rank.",
        )
        source = eqx.error_if(
            source,
            (diagnostics.num_probes != probe_count)
            | jnp.any(diagnostics.probe_source_indices < 0)
            | jnp.any(diagnostics.probe_source_indices >= source.shape[0])
            | jnp.any(diagnostics.probe_target_indices < 0)
            | jnp.any(diagnostics.probe_target_indices >= target.shape[0]),
            "Approximation probe evidence is inconsistent with the factors.",
        )
        source = eqx.error_if(
            source,
            (diagnostics.status < int(KernelApproximationStatus.VALID))
            | (
                diagnostics.status
                > int(KernelApproximationStatus.PROBE_TOLERANCE_EXCEEDED)
            )
            | (diagnostics.zero_source_rows < 0)
            | (diagnostics.zero_target_rows < 0),
            "Approximation status evidence is invalid.",
        )
        self.source_factors = source
        self.target_factors = target
        self.source_log_scale = source_scale
        self.target_log_scale = target_scale
        self.source_points = source_design
        self.target_points = target_design
        self.epsilon = epsilon_
        self.diagnostics = diagnostics
        self.factorization_id = identity

    @property
    def rank(self) -> int:
        """Return the stored positive feature rank."""
        return self.source_factors.shape[1]

    def kernel_matrix(self) -> Array:
        """Explicitly materialize the represented approximate kernel."""
        kernel = self.source_factors @ self.target_factors.T
        log_kernel = (
            self.source_log_scale[:, None]
            + self.target_log_scale[None, :]
            + _safe_log(kernel)
        )
        return jnp.where(kernel > 0.0, jnp.exp(log_kernel), 0.0)


class GaussianPositiveFeatures(StrictModule):
    """Deterministic positive Gaussian features for an entropic RBF kernel."""

    key: Array
    probe_tolerance: Array
    rank: int = eqx.field(static=True)
    num_probes: int = eqx.field(static=True)

    def __init__(
        self,
        key: Array,
        rank: int,
        /,
        *,
        num_probes: int = 32,
        probe_tolerance: ArrayLike = jnp.inf,
    ):
        feature_rank = int(rank)
        probes = int(num_probes)
        if feature_rank < 1:
            raise ValueError("rank must be positive.")
        if probes < 1:
            raise ValueError("num_probes must be positive.")
        key_data = jax.random.key_data(key)
        if key_data.shape != (2,):
            raise ValueError("key must be one scalar JAX PRNG key.")
        tolerance = jnp.asarray(probe_tolerance, dtype=float).reshape(())
        self.probe_tolerance = eqx.error_if(
            tolerance,
            jnp.isnan(tolerance) | (tolerance < 0.0),
            "probe_tolerance must be nonnegative and not NaN.",
        )
        self.key = key
        self.rank = feature_rank
        self.num_probes = probes

    def __call__(
        self, problem: DiscreteTransportProblem, epsilon: ArrayLike, /
    ) -> PositiveKernelFactors:
        if not isinstance(problem, DiscreteTransportProblem):
            raise TypeError("problem must be a DiscreteTransportProblem.")
        if not isinstance(problem.cost, SquaredEuclideanCost):
            raise TypeError(
                "GaussianPositiveFeatures requires SquaredEuclideanCost."
            )
        epsilon_ = jnp.asarray(
            epsilon,
            dtype=jnp.result_type(
                problem.source.points, problem.target.points, float
            ),
        ).reshape(())
        epsilon_ = eqx.error_if(
            epsilon_,
            ~jnp.isfinite(epsilon_) | (epsilon_ <= 0.0),
            "epsilon must be finite and positive.",
        )
        projection_key, source_probe_key, target_probe_key = jax.random.split(
            self.key, 3
        )
        source_points = problem.source.points.astype(epsilon_.dtype)
        target_points = problem.target.points.astype(epsilon_.dtype)
        source_center = jnp.sum(
            problem.source_probabilities[:, None] * source_points, axis=0
        )
        target_center = jnp.sum(
            problem.target_probabilities[:, None] * target_points, axis=0
        )
        center = 0.5 * (source_center + target_center)
        source_coordinates = source_points - center
        target_coordinates = target_points - center
        projections = jax.random.normal(
            projection_key,
            (self.rank, source_points.shape[1]),
            dtype=epsilon_.dtype,
        )
        inverse_scale = jnp.sqrt(2.0 / epsilon_)
        rank_normalization = 0.5 * jnp.log(
            jnp.asarray(self.rank, dtype=epsilon_.dtype)
        )

        def log_features(points):
            squared_norm = jnp.sum(jnp.square(points), axis=1)
            return (
                inverse_scale * (points @ projections.T)
                - (2.0 / epsilon_) * squared_norm[:, None]
                - rank_normalization
            )

        source_log_features = log_features(source_coordinates)
        target_log_features = log_features(target_coordinates)
        source_factors, source_log_scale, source_finite = (
            _normalize_log_features(
                source_log_features, problem.source_probabilities > 0.0
            )
        )
        target_factors, target_log_scale, target_finite = (
            _normalize_log_features(
                target_log_features, problem.target_probabilities > 0.0
            )
        )
        source_indices = jax.random.categorical(
            source_probe_key,
            _safe_log(problem.source_probabilities),
            shape=(self.num_probes,),
        ).astype(jnp.int32)
        target_indices = jax.random.categorical(
            target_probe_key,
            _safe_log(problem.target_probabilities),
            shape=(self.num_probes,),
        ).astype(jnp.int32)
        exact_probe = jnp.exp(
            -problem.cost_at(source_indices, target_indices) / epsilon_
        )
        probe_dot = jnp.sum(
            source_factors[source_indices] * target_factors[target_indices],
            axis=1,
        )
        probe_log = (
            source_log_scale[source_indices]
            + target_log_scale[target_indices]
            + _safe_log(probe_dot)
        )
        approximate_probe = jnp.where(probe_dot > 0.0, jnp.exp(probe_log), 0.0)
        absolute_error = jnp.abs(approximate_probe - exact_probe)
        relative_errors = jnp.where(
            exact_probe > 0.0,
            absolute_error / jnp.where(exact_probe > 0.0, exact_probe, 1.0),
            jnp.where(approximate_probe == 0.0, 0.0, jnp.inf),
        )
        exact_norm = jnp.linalg.norm(exact_probe)
        relative_error = jnp.where(
            exact_norm > 0.0,
            jnp.linalg.norm(approximate_probe - exact_probe) / exact_norm,
            jnp.where(jnp.linalg.norm(approximate_probe) == 0.0, 0.0, jnp.inf),
        )
        maximum_error = jnp.max(relative_errors)
        source_support = source_factors @ jnp.sum(target_factors, axis=0)
        target_support = target_factors @ jnp.sum(source_factors, axis=0)
        zero_source_rows = jnp.sum(
            (problem.source_probabilities > 0.0) & (source_support <= 0.0),
            dtype=jnp.int32,
        )
        zero_target_rows = jnp.sum(
            (problem.target_probabilities > 0.0) & (target_support <= 0.0),
            dtype=jnp.int32,
        )
        finite_features = source_finite & target_finite
        finite_probe = (
            jnp.all(jnp.isfinite(exact_probe))
            & jnp.all(jnp.isfinite(approximate_probe))
            & jnp.all(jnp.isfinite(relative_errors))
            & jnp.isfinite(relative_error)
        )
        status = jnp.where(
            ~finite_features,
            int(KernelApproximationStatus.NONFINITE_FEATURES),
            jnp.where(
                (zero_source_rows > 0) | (zero_target_rows > 0),
                int(KernelApproximationStatus.ZERO_KERNEL_ROW),
                jnp.where(
                    ~finite_probe,
                    int(KernelApproximationStatus.NONFINITE_PROBE),
                    jnp.where(
                        relative_error > self.probe_tolerance.astype(epsilon_.dtype),
                        int(KernelApproximationStatus.PROBE_TOLERANCE_EXCEEDED),
                        int(KernelApproximationStatus.VALID),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        diagnostics = PositiveKernelApproximationDiagnostics(
            status=status,
            rank=jnp.asarray(self.rank, dtype=jnp.int32),
            num_probes=jnp.asarray(self.num_probes, dtype=jnp.int32),
            key_data=jax.random.key_data(self.key),
            probe_source_indices=source_indices,
            probe_target_indices=target_indices,
            exact_probe_values=exact_probe,
            approximate_probe_values=approximate_probe,
            relative_probe_errors=relative_errors,
            relative_probe_error=relative_error,
            maximum_relative_probe_error=maximum_error,
            probe_tolerance=self.probe_tolerance.astype(epsilon_.dtype),
            zero_source_rows=zero_source_rows,
            zero_target_rows=zero_target_rows,
            finite_features=finite_features,
        )
        return PositiveKernelFactors(
            source_factors,
            target_factors,
            source_log_scale=source_log_scale,
            target_log_scale=target_log_scale,
            source_points=source_points,
            target_points=target_points,
            epsilon=epsilon_,
            diagnostics=diagnostics,
            factorization_id="gaussian-positive-features",
        )


class PositiveFeatureSinkhornResult(AbstractBalancedTransportPlan):
    """Approximate balanced transport plan represented by positive features."""

    problem: DiscreteTransportProblem
    factors: PositiveKernelFactors
    source_scaling: Array
    target_scaling: Array
    epsilon: Array
    regularized_cost: Array
    dual_cost: Array
    exact_transport_cost: Array
    exact_ground_cost_computed: bool = eqx.field(static=True)
    diagnostics: SinkhornDiagnostics
    approximation: PositiveKernelApproximationDiagnostics
    provenance: TransportProvenance

    @property
    def converged(self) -> Array:
        """Whether convergence and approximation quality both passed."""
        return self.diagnostics.status == int(TransportStatus.CONVERGED)

    @property
    def surrogate_regularized_cost(self) -> Array:
        """Return the objective induced by the approximate positive kernel."""
        return self.regularized_cost

    def regularized_objective(self) -> Array:
        """Return the physical surrogate-kernel regularized objective."""
        return self.regularized_cost

    def source_marginal(self) -> Array:
        """Return the physical source marginal induced by the factorized plan."""
        source, _ = _factor_marginals(
            self.factors,
            self.source_scaling,
            self.target_scaling,
        )
        return self.problem.mass * source

    def target_marginal(self) -> Array:
        """Return the physical target marginal induced by the factorized plan."""
        _, target = _factor_marginals(
            self.factors,
            self.source_scaling,
            self.target_scaling,
        )
        return self.problem.mass * target

    def apply_source_to_target(self, values: ArrayLike, /) -> Array:
        """Apply the physical coupling to source-indexed payloads in linear rank cost."""
        return _apply_factor_plan(
            self,
            jnp.asarray(values),
            source_to_target=True,
        )

    def apply_target_to_source(self, values: ArrayLike, /) -> Array:
        """Apply the physical coupling to target-indexed payloads in linear rank cost."""
        return _apply_factor_plan(
            self,
            jnp.asarray(values),
            source_to_target=False,
        )

    def barycentric_source_to_target(self, values: ArrayLike, /) -> Array:
        """Return target-conditioned barycenters of source payloads."""
        applied = self.apply_source_to_target(values)
        weights = self.problem.target_weights
        denominator = weights.reshape((weights.shape[0],) + (1,) * (applied.ndim - 1))
        return jnp.where(denominator > 0.0, applied / denominator, 0.0)

    def barycentric_target_to_source(self, values: ArrayLike, /) -> Array:
        """Return source-conditioned barycenters of target payloads."""
        applied = self.apply_target_to_source(values)
        weights = self.problem.source_weights
        denominator = weights.reshape((weights.shape[0],) + (1,) * (applied.ndim - 1))
        return jnp.where(denominator > 0.0, applied / denominator, 0.0)

    def dense_plan(self) -> Array:
        """Explicitly materialize the complete approximate physical plan."""
        kernel = self.factors.source_factors @ self.factors.target_factors.T
        return self.problem.mass * (
            self.source_scaling[:, None]
            * kernel
            * self.target_scaling[None, :]
        )


class PositiveFeatureSinkhorn(AbstractBalancedTransportSolver):
    """Approximate balanced Sinkhorn scaling over positive kernel features."""

    epsilon: Array
    feature_map: GaussianPositiveFeatures
    tolerance: Array
    max_iterations: int = eqx.field(static=True)
    min_iterations: int = eqx.field(static=True)
    check_every: int = eqx.field(static=True)
    early_stop: bool = eqx.field(static=True)
    store_history: bool = eqx.field(static=True)
    statistic_block_size: int = eqx.field(static=True)

    def __init__(
        self,
        epsilon: ArrayLike,
        feature_map: GaussianPositiveFeatures,
        /,
        *,
        max_iterations: int = 500,
        min_iterations: int = 1,
        tolerance: ArrayLike = 1e-7,
        check_every: int = 5,
        early_stop: bool = False,
        store_history: bool = False,
        statistic_block_size: int = 256,
    ):
        if not isinstance(feature_map, GaussianPositiveFeatures):
            raise TypeError("feature_map must be GaussianPositiveFeatures.")
        maximum = int(max_iterations)
        minimum = int(min_iterations)
        interval = int(check_every)
        block_size = int(statistic_block_size)
        if maximum < 1:
            raise ValueError("max_iterations must be positive.")
        if minimum < 0 or minimum > maximum:
            raise ValueError("min_iterations must lie in [0, max_iterations].")
        if interval < 1:
            raise ValueError("check_every must be positive.")
        if block_size < 1:
            raise ValueError("statistic_block_size must be positive.")
        epsilon_ = jnp.asarray(epsilon, dtype=float).reshape(())
        tolerance_ = jnp.asarray(tolerance, dtype=float).reshape(())
        self.epsilon = eqx.error_if(
            epsilon_,
            ~jnp.isfinite(epsilon_) | (epsilon_ <= 0.0),
            "epsilon must be finite and positive.",
        )
        self.tolerance = eqx.error_if(
            tolerance_,
            ~jnp.isfinite(tolerance_) | (tolerance_ < 0.0),
            "tolerance must be finite and nonnegative.",
        )
        self.feature_map = feature_map
        self.max_iterations = maximum
        self.min_iterations = minimum
        self.check_every = interval
        self.early_stop = bool(early_stop)
        self.store_history = bool(store_history)
        self.statistic_block_size = block_size

    def __call__(
        self,
        problem: DiscreteTransportProblem,
        /,
        *,
        factors: PositiveKernelFactors | None = None,
        initial_scalings: tuple[ArrayLike, ArrayLike] | None = None,
        exact_ground_cost: bool = False,
    ) -> PositiveFeatureSinkhornResult:
        if not isinstance(problem, DiscreteTransportProblem):
            raise TypeError("problem must be a DiscreteTransportProblem.")
        if factors is None:
            factors_ = self.feature_map(problem, self.epsilon)
        else:
            if not isinstance(factors, PositiveKernelFactors):
                raise TypeError("factors must be PositiveKernelFactors or None.")
            factors_ = factors
        source_count, target_count = problem.shape
        if factors_.source_factors.shape[0] != source_count:
            raise ValueError("Source factors must match the problem source atoms.")
        if factors_.target_factors.shape[0] != target_count:
            raise ValueError("Target factors must match the problem target atoms.")
        if factors_.source_points.shape != problem.source.points.shape:
            raise ValueError("Factor source points must match the problem point shape.")
        if factors_.target_points.shape != problem.target.points.shape:
            raise ValueError("Factor target points must match the problem point shape.")
        factors_source = eqx.error_if(
            factors_.source_factors,
            jnp.any(factors_.source_points != problem.source.points),
            "Source factors were generated for different points.",
        )
        factors_target = eqx.error_if(
            factors_.target_factors,
            jnp.any(factors_.target_points != problem.target.points),
            "Target factors were generated for different points.",
        )
        factors_ = eqx.tree_at(
            lambda item: (item.source_factors, item.target_factors),
            factors_,
            (factors_source, factors_target),
        )
        epsilon = self.epsilon.astype(
            jnp.result_type(
                problem.source.points,
                problem.target.points,
                factors_.source_factors,
            )
        )
        epsilon = eqx.error_if(
            epsilon,
            factors_.epsilon != epsilon,
            "Positive kernel factors must use the solver epsilon.",
        )
        tolerance = self.tolerance.astype(epsilon.dtype)
        source_probability = problem.source_probabilities.astype(epsilon.dtype)
        target_probability = problem.target_probabilities.astype(epsilon.dtype)
        if initial_scalings is None:
            source_initial = jnp.where(
                source_probability > 0.0,
                jnp.ones((source_count,), dtype=epsilon.dtype),
                0.0,
            )
            target_initial = jnp.where(
                target_probability > 0.0,
                jnp.ones((target_count,), dtype=epsilon.dtype),
                0.0,
            )
        else:
            source_initial = jnp.asarray(initial_scalings[0], dtype=epsilon.dtype)
            target_initial = jnp.asarray(initial_scalings[1], dtype=epsilon.dtype)
            if source_initial.shape != (source_count,):
                raise ValueError("Initial source scaling must match source atom count.")
            if target_initial.shape != (target_count,):
                raise ValueError("Initial target scaling must match target atom count.")
            source_initial = eqx.error_if(
                source_initial,
                jnp.any(~jnp.isfinite(source_initial))
                | jnp.any(source_initial < 0.0),
                "Initial source scaling must be finite and nonnegative.",
            )
            target_initial = eqx.error_if(
                target_initial,
                jnp.any(~jnp.isfinite(target_initial))
                | jnp.any(target_initial < 0.0),
                "Initial target scaling must be finite and nonnegative.",
            )
        source_probe = factors_.source_factors @ (
            factors_.target_factors.T @ target_probability
        )
        target_probe = factors_.target_factors @ (
            factors_.source_factors.T @ source_probability
        )
        active_zero_source = jnp.any(
            (source_probability > 0.0) & (source_probe <= 0.0)
        )
        active_zero_target = jnp.any(
            (target_probability > 0.0) & (target_probe <= 0.0)
        )
        zero_kernel_row = active_zero_source | active_zero_target
        approximation_failed = ~factors_.diagnostics.successful
        nonzero_approximation_failure = approximation_failed & (
            factors_.diagnostics.status
            != int(KernelApproximationStatus.ZERO_KERNEL_ROW)
        )
        initial_carry = (
            source_initial,
            target_initial,
            jnp.asarray(jnp.inf, dtype=epsilon.dtype),
            jnp.asarray(jnp.inf, dtype=epsilon.dtype),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(False),
        )

        def step(carry, index):
            (
                source_scaling,
                target_scaling,
                marginal_residual,
                scaling_residual,
                first_converged,
                converged,
                failed,
            ) = carry
            frozen = (
                failed
                | approximation_failed
                | zero_kernel_row
                | (converged if self.early_stop else False)
            )

            def update(_):
                source_denominator = factors_.source_factors @ (
                    factors_.target_factors.T @ target_scaling
                )
                source_valid = (source_probability <= 0.0) | (
                    jnp.isfinite(source_denominator) & (source_denominator > 0.0)
                )
                safe_source_denominator = jnp.where(
                    (source_probability > 0.0) & source_valid,
                    source_denominator,
                    1.0,
                )
                next_source = jnp.where(
                    source_probability > 0.0,
                    source_probability / safe_source_denominator,
                    0.0,
                )
                target_denominator = factors_.target_factors @ (
                    factors_.source_factors.T @ next_source
                )
                target_valid = (target_probability <= 0.0) | (
                    jnp.isfinite(target_denominator) & (target_denominator > 0.0)
                )
                safe_target_denominator = jnp.where(
                    (target_probability > 0.0) & target_valid,
                    target_denominator,
                    1.0,
                )
                next_target = jnp.where(
                    target_probability > 0.0,
                    target_probability / safe_target_denominator,
                    0.0,
                )
                finite = (
                    jnp.all(source_valid)
                    & jnp.all(target_valid)
                    & jnp.all(jnp.isfinite(next_source))
                    & jnp.all(jnp.isfinite(next_target))
                )
                next_source = jnp.where(finite, next_source, source_scaling)
                next_target = jnp.where(finite, next_target, target_scaling)
                source_change = jnp.max(
                    jnp.abs(
                        _active_log(next_source, source_probability)
                        - _active_log(source_scaling, source_probability)
                    )
                )
                target_change = jnp.max(
                    jnp.abs(
                        _active_log(next_target, target_probability)
                        - _active_log(target_scaling, target_probability)
                    )
                )
                return (
                    next_source,
                    next_target,
                    jnp.maximum(source_change, target_change),
                    ~finite,
                )

            def keep(_):
                return source_scaling, target_scaling, scaling_residual, failed

            next_source, next_target, next_scaling_residual, next_failed = (
                jax.lax.cond(frozen, keep, update, operand=None)
            )
            iteration = index + 1
            should_check = (
                (iteration % self.check_every == 0)
                | (iteration == self.max_iterations)
                | (iteration == self.min_iterations)
            )

            def check(_):
                source_marginal, target_marginal = _factor_marginals(
                    factors_, next_source, next_target
                )
                finite = jnp.all(jnp.isfinite(source_marginal)) & jnp.all(
                    jnp.isfinite(target_marginal)
                )
                residual = jnp.maximum(
                    jnp.sum(jnp.abs(source_marginal - source_probability)),
                    jnp.sum(jnp.abs(target_marginal - target_probability)),
                )
                return jnp.where(finite, residual, jnp.inf), ~finite

            def retain(_):
                return marginal_residual, jnp.asarray(False)

            next_residual, marginal_failed = jax.lax.cond(
                should_check & ~frozen & ~next_failed,
                check,
                retain,
                operand=None,
            )
            eligible = (
                should_check
                & (iteration >= self.min_iterations)
                & (next_residual <= tolerance)
                & ~next_failed
                & ~marginal_failed
                & ~approximation_failed
                & ~zero_kernel_row
            )
            next_first = jnp.where(
                (first_converged < 0) & eligible,
                iteration.astype(jnp.int32),
                first_converged,
            )
            next_converged = converged | eligible
            return (
                next_source,
                next_target,
                next_residual,
                next_scaling_residual,
                next_first,
                next_converged,
                next_failed | marginal_failed,
            ), next_residual

        final_carry, residuals = jax.lax.scan(
            step,
            initial_carry,
            jnp.arange(self.max_iterations, dtype=jnp.int32),
        )
        (
            source_scaling,
            target_scaling,
            _,
            scaling_residual,
            first_converged,
            _,
            failed,
        ) = final_carry
        source_marginal, target_marginal = _factor_marginals(
            factors_, source_scaling, target_scaling
        )
        final_residual = jnp.maximum(
            jnp.sum(jnp.abs(source_marginal - source_probability)),
            jnp.sum(jnp.abs(target_marginal - target_probability)),
        )
        plan_mass = jnp.sum(source_marginal)
        source_ratio = (
            _active_log(source_scaling, source_probability)
            - _active_log(source_probability, source_probability)
            - factors_.source_log_scale
        )
        target_ratio = (
            _active_log(target_scaling, target_probability)
            - _active_log(target_probability, target_probability)
            - factors_.target_log_scale
        )
        primal_probability = epsilon * (
            jnp.sum(source_marginal * source_ratio)
            + jnp.sum(target_marginal * target_ratio)
            - plan_mass
            + 1.0
        )
        dual_probability = epsilon * (
            jnp.sum(source_probability * source_ratio)
            + jnp.sum(target_probability * target_ratio)
            - plan_mass
            + 1.0
        )
        regularized_cost = problem.mass * primal_probability
        dual_cost = problem.mass * dual_probability
        finite_objective = (
            jnp.isfinite(regularized_cost)
            & jnp.isfinite(dual_cost)
            & jnp.all(jnp.isfinite(source_marginal))
            & jnp.all(jnp.isfinite(target_marginal))
        )
        final_converged = (
            (final_residual <= tolerance)
            & (self.max_iterations >= self.min_iterations)
            & ~failed
            & ~approximation_failed
            & ~zero_kernel_row
            & finite_objective
        )
        status = jnp.where(
            nonzero_approximation_failure,
            int(TransportStatus.APPROXIMATION_FAILED),
            jnp.where(
                zero_kernel_row,
                int(TransportStatus.ZERO_KERNEL_ROW),
                jnp.where(
                    failed,
                    int(TransportStatus.NONFINITE_ITERATE),
                    jnp.where(
                        ~finite_objective,
                        int(TransportStatus.NONFINITE_OBJECTIVE),
                        jnp.where(
                            final_converged,
                            int(TransportStatus.CONVERGED),
                            int(TransportStatus.MAXIMUM_ITERATIONS_REACHED),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        check_indices = tuple(
            index
            for index in range(self.max_iterations)
            if (index + 1) % self.check_every == 0
            or (index + 1) == self.max_iterations
            or (index + 1) == self.min_iterations
        )
        if self.store_history:
            history = residuals[jnp.asarray(check_indices, dtype=jnp.int32)]
        else:
            history = jnp.empty((0,), dtype=epsilon.dtype)
        actual_iterations = jnp.where(
            self.early_stop & (first_converged >= 0),
            first_converged,
            self.max_iterations,
        ).astype(jnp.int32)
        diagnostics = SinkhornDiagnostics(
            status=status,
            num_iterations=actual_iterations,
            first_converged_iteration=first_converged,
            normalized_marginal_residual=final_residual,
            physical_marginal_residual=problem.mass * final_residual,
            dual_residual=scaling_residual,
            primal_dual_gap=jnp.abs(regularized_cost - dual_cost),
            num_checks=jnp.asarray(len(check_indices), dtype=jnp.int32),
            residual_history=history,
        )
        if exact_ground_cost:
            exact_transport_cost = problem.mass * _exact_ground_cost_probability(
                problem,
                factors_,
                source_scaling,
                target_scaling,
                block_size=self.statistic_block_size,
            )
        else:
            exact_transport_cost = jnp.asarray(jnp.nan, dtype=epsilon.dtype)
        provenance = TransportProvenance(
            "positive-feature-sinkhorn",
            problem.provenance.cost,
            "factorized",
            "unrolled",
            problem.provenance.source,
            problem.provenance.target,
            approximation=(
                f"{factors_.factorization_id}(rank={factors_.rank})"
            ),
        )
        return PositiveFeatureSinkhornResult(
            problem=problem,
            factors=factors_,
            source_scaling=source_scaling,
            target_scaling=target_scaling,
            epsilon=epsilon,
            regularized_cost=regularized_cost,
            dual_cost=dual_cost,
            exact_transport_cost=exact_transport_cost,
            exact_ground_cost_computed=bool(exact_ground_cost),
            diagnostics=diagnostics,
            approximation=factors_.diagnostics,
            provenance=provenance,
        )


def _normalize_log_features(
    log_features: Array, active: Array, /
) -> tuple[Array, Array, Array]:
    finite_rows = jnp.all(jnp.isfinite(log_features), axis=1)
    valid_rows = ~active | finite_rows
    finite_log_features = jnp.where(jnp.isfinite(log_features), log_features, -jnp.inf)
    row_maximum = jnp.max(finite_log_features, axis=1)
    safe_maximum = jnp.where(active & finite_rows, row_maximum, 0.0)
    factors = jnp.where(
        (active & finite_rows)[:, None],
        jnp.exp(log_features - safe_maximum[:, None]),
        0.0,
    )
    return factors, safe_maximum, jnp.all(valid_rows)


def _factor_marginals(
    factors: PositiveKernelFactors,
    source_scaling: Array,
    target_scaling: Array,
    /,
) -> tuple[Array, Array]:
    source = source_scaling * (
        factors.source_factors @ (factors.target_factors.T @ target_scaling)
    )
    target = target_scaling * (
        factors.target_factors @ (factors.source_factors.T @ source_scaling)
    )
    return source, target


def _apply_factor_plan(
    result: PositiveFeatureSinkhornResult,
    values: Array,
    /,
    *,
    source_to_target: bool,
) -> Array:
    values_ = jnp.asarray(values)
    source_count, target_count = result.problem.shape
    expected = source_count if source_to_target else target_count
    output_count = target_count if source_to_target else source_count
    if values_.ndim < 1 or values_.shape[0] != expected:
        direction = "Source-to-target" if source_to_target else "Target-to-source"
        raise ValueError(f"{direction} values must begin with the input atom count.")
    payload_shape = values_.shape[1:]
    flat_values = values_.reshape((expected, -1))
    if source_to_target:
        reduced = result.factors.source_factors.T @ (
            result.source_scaling[:, None] * flat_values
        )
        output = result.target_scaling[:, None] * (
            result.factors.target_factors @ reduced
        )
    else:
        reduced = result.factors.target_factors.T @ (
            result.target_scaling[:, None] * flat_values
        )
        output = result.source_scaling[:, None] * (
            result.factors.source_factors @ reduced
        )
    return (result.problem.mass * output).reshape((output_count,) + payload_shape)


def _exact_ground_cost_probability(
    problem: DiscreteTransportProblem,
    factors: PositiveKernelFactors,
    source_scaling: Array,
    target_scaling: Array,
    /,
    *,
    block_size: int,
) -> Array:
    source_count, target_count = problem.shape
    source_blocks = (source_count + block_size - 1) // block_size
    target_blocks = (target_count + block_size - 1) // block_size
    dtype = jnp.result_type(source_scaling, target_scaling, problem.source.points)

    def source_body(source_block, total):
        source_indices = source_block * block_size + jnp.arange(
            block_size, dtype=jnp.int32
        )
        source_valid = source_indices < source_count
        safe_source_indices = jnp.minimum(source_indices, source_count - 1)
        source_factor = factors.source_factors[safe_source_indices]
        source_scale = source_scaling[safe_source_indices]

        def target_body(target_block, subtotal):
            target_indices = target_block * block_size + jnp.arange(
                block_size, dtype=jnp.int32
            )
            target_valid = target_indices < target_count
            safe_target_indices = jnp.minimum(target_indices, target_count - 1)
            target_factor = factors.target_factors[safe_target_indices]
            target_scale = target_scaling[safe_target_indices]
            plan = (
                source_scale[:, None]
                * (source_factor @ target_factor.T)
                * target_scale[None, :]
            )
            costs = problem.cost_at(
                safe_source_indices[:, None], safe_target_indices[None, :]
            )
            valid = source_valid[:, None] & target_valid[None, :]
            return subtotal + jnp.sum(jnp.where(valid, plan * costs, 0.0))

        return jax.lax.fori_loop(
            0,
            target_blocks,
            target_body,
            total,
        )

    return jax.lax.fori_loop(
        0,
        source_blocks,
        source_body,
        jnp.asarray(0.0, dtype=dtype),
    )


def _active_log(values: Array, active_weights: Array, /) -> Array:
    return jnp.where(active_weights > 0.0, _safe_log(values), 0.0)


def _safe_log(values: Array, /) -> Array:
    return jnp.where(values > 0.0, jnp.log(values), -jnp.inf)


__all__ = [
    "GaussianPositiveFeatures",
    "KernelApproximationStatus",
    "PositiveFeatureSinkhorn",
    "PositiveFeatureSinkhornResult",
    "PositiveKernelApproximationDiagnostics",
    "PositiveKernelFactors",
]
