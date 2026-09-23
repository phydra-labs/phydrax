#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg._local_blocks import (
    prepare_local_block_factorization,
    solve_local_blocks_detailed,
)
from ._contracts import (
    _mean_domain_result,
    _natural_domain_result,
    AbstractExponentialFamily,
    EXPONENTIAL_FAMILY_MEAN_BOUNDARY,
    EXPONENTIAL_FAMILY_NONCONVERGED,
    EXPONENTIAL_FAMILY_NONFINITE,
    EXPONENTIAL_FAMILY_SUCCESS,
    ExponentialFamilyConversionResult,
    ExponentialFamilyDomainResult,
    ExponentialFamilySignature,
    MeanCoordinates,
    NaturalCoordinates,
    StatisticBatch,
)


class FiniteSupportNaturalSolvePlan(StrictModule):
    """Fixed-work inversion of finite-support exponential-family moments."""

    maximum_steps: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    minimum_probability: float = eqx.field(static=True)
    line_search_factors: tuple[float, ...] = eqx.field(static=True)
    portable: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_steps: int = 24,
        residual_tolerance: float = 1.0e-10,
        minimum_probability: float = 0.0,
        line_search_factors: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125),
        portable: bool = False,
    ):
        steps = int(maximum_steps)
        tolerance = float(residual_tolerance)
        probability = float(minimum_probability)
        factors = tuple(float(value) for value in line_search_factors)
        if steps < 1:
            raise ValueError("maximum_steps must be positive.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("residual_tolerance must be finite and positive.")
        if not np.isfinite(probability) or probability < 0.0:
            raise ValueError("minimum_probability must be finite and nonnegative.")
        if not factors or any(
            not np.isfinite(value) or value <= 0.0 or value > 1.0 for value in factors
        ):
            raise ValueError("line_search_factors must lie in (0, 1].")
        if tuple(sorted(factors, reverse=True)) != factors:
            raise ValueError("line_search_factors must be in descending order.")
        self.maximum_steps = steps
        self.residual_tolerance = tolerance
        self.minimum_probability = probability
        self.line_search_factors = factors
        self.portable = bool(portable)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-support-natural-solve",
                "maximum_steps": steps,
                "residual_tolerance": tolerance,
                "minimum_probability": probability,
                "line_search_factors": list(factors),
                "portable": bool(portable),
            }
        )


class FiniteSupportSolveEvidence(StrictModule):
    """Numerical evidence for one finite-support mean inversion."""

    residual: Array
    iterations: Array
    minimum_probability: Array
    factorization_failed: Array
    line_search_failed: Array
    successful: Array


class FiniteSupportSolveResult(StrictModule):
    """Natural coordinates, probabilities, and evidence for one inversion."""

    conversion: ExponentialFamilyConversionResult
    probabilities: Array
    evidence: FiniteSupportSolveEvidence


def _portable_positive_definite_solve(
    matrix: Array,
    right_hand_side: Array,
    /,
) -> tuple[Array, Array]:
    """Batched Cholesky solve expressed only with portable array primitives."""
    batch_size, dimension, _ = matrix.shape
    row_indices = jnp.arange(dimension)
    factor = jnp.zeros_like(matrix)
    failed = jnp.zeros((batch_size,), dtype=jnp.bool_)

    def factor_step(index, state):
        current, current_failed = state
        selector = jax.nn.one_hot(index, dimension, dtype=matrix.dtype)
        diagonal_selector = selector[:, None] * selector[None, :]
        row = jnp.sum(current * selector[None, :, None], axis=1)
        matrix_diagonal = jnp.sum(matrix * diagonal_selector[None, :, :], axis=(-2, -1))
        diagonal_residual = matrix_diagonal - jnp.sum(row * row, axis=-1)
        valid = jnp.isfinite(diagonal_residual) & (diagonal_residual > 0.0)
        diagonal = jnp.sqrt(jnp.where(valid, diagonal_residual, 1.0))
        current = (
            current * (1.0 - diagonal_selector[None, :, :])
            + diagonal[:, None, None] * diagonal_selector[None, :, :]
        )
        pivot_row = jnp.sum(current * selector[None, :, None], axis=1)
        products = jnp.sum(current * pivot_row[:, None, :], axis=-1)
        matrix_column = jnp.sum(matrix * selector[None, None, :], axis=-1)
        column = (matrix_column - products) / diagonal[:, None]
        current_column = jnp.sum(current * selector[None, None, :], axis=-1)
        updated_column = jnp.where(
            row_indices[None, :] > index,
            column,
            current_column,
        )
        current = (
            current * (1.0 - selector[None, None, :])
            + updated_column[:, :, None] * selector[None, None, :]
        )
        return current, current_failed | ~valid

    factor, failed = jax.lax.fori_loop(
        0,
        dimension,
        factor_step,
        (factor, failed),
    )
    forward = jnp.zeros_like(right_hand_side)

    def forward_step(index, current):
        selector = jax.nn.one_hot(index, dimension, dtype=matrix.dtype)
        factor_row = jnp.sum(factor * selector[None, :, None], axis=1)
        rhs_value = jnp.sum(right_hand_side * selector[None, :], axis=-1)
        diagonal = jnp.sum(factor_row * selector[None, :], axis=-1)
        value = (rhs_value - jnp.sum(factor_row * current, axis=-1)) / diagonal
        return current * (1.0 - selector[None, :]) + value[:, None] * selector[None, :]

    forward = jax.lax.fori_loop(0, dimension, forward_step, forward)
    solution = jnp.zeros_like(right_hand_side)

    def backward_step(offset, current):
        index = dimension - 1 - offset
        selector = jax.nn.one_hot(index, dimension, dtype=matrix.dtype)
        factor_column = jnp.sum(factor * selector[None, None, :], axis=-1)
        rhs_value = jnp.sum(forward * selector[None, :], axis=-1)
        diagonal = jnp.sum(factor_column * selector[None, :], axis=-1)
        value = (rhs_value - jnp.sum(factor_column * current, axis=-1)) / diagonal
        return current * (1.0 - selector[None, :]) + value[:, None] * selector[None, :]

    solution = jax.lax.fori_loop(0, dimension, backward_step, solution)
    failed |= jnp.any(~jnp.isfinite(factor), axis=(-2, -1)) | jnp.any(
        ~jnp.isfinite(solution), axis=-1
    )
    return jnp.where(failed[:, None], 0.0, solution), failed


class FiniteSupportExponentialFamily(AbstractExponentialFamily):
    """Discrete exponential family with arbitrary finite sufficient statistics."""

    statistics: Array
    log_base_probabilities: Array
    feature_minimum: Array
    feature_maximum: Array
    solve_plan: FiniteSupportNaturalSolvePlan
    _signature: ExponentialFamilySignature = eqx.field(static=True)
    support_size: int = eqx.field(static=True)

    def __init__(
        self,
        statistics: ArrayLike,
        base_probabilities: ArrayLike,
        /,
        *,
        family_id: str,
        support_id: str | None = None,
        solve_plan: FiniteSupportNaturalSolvePlan | None = None,
    ):
        statistic_host = np.asarray(statistics)
        base_host = np.asarray(base_probabilities)
        if statistic_host.ndim != 2:
            raise ValueError("statistics must have shape (support, dimension).")
        support_size, dimension = statistic_host.shape
        if support_size < 2 or dimension < 1 or support_size <= dimension:
            raise ValueError("Finite support must contain more points than dimensions.")
        if base_host.shape != (support_size,):
            raise ValueError("base_probabilities must match the support size.")
        if not np.issubdtype(statistic_host.dtype, np.floating):
            statistic_host = statistic_host.astype(np.float64)
        if not np.issubdtype(base_host.dtype, np.floating):
            base_host = base_host.astype(np.float64)
        if not np.all(np.isfinite(statistic_host)):
            raise ValueError("statistics must be finite.")
        if not np.all(np.isfinite(base_host)) or np.any(base_host <= 0.0):
            raise ValueError("base_probabilities must be finite and positive.")
        augmented = np.column_stack((np.ones(support_size), statistic_host))
        if np.linalg.matrix_rank(augmented) != dimension + 1:
            raise ValueError("statistics must be affinely full rank.")
        normalized_base = base_host / np.sum(base_host)
        identifier = support_id or canonical_fingerprint(
            {
                "statistics": statistic_host.tolist(),
                "base_probabilities": normalized_base.tolist(),
            }
        )
        if not family_id or not identifier:
            raise ValueError("family_id and support_id must be non-empty.")
        plan = FiniteSupportNaturalSolvePlan() if solve_plan is None else solve_plan
        if not isinstance(plan, FiniteSupportNaturalSolvePlan):
            raise TypeError("solve_plan must be a FiniteSupportNaturalSolvePlan.")
        dtype = np.result_type(statistic_host.dtype, normalized_base.dtype)
        self.statistics = jnp.asarray(statistic_host, dtype=dtype)
        self.log_base_probabilities = jnp.log(jnp.asarray(normalized_base, dtype=dtype))
        self.feature_minimum = jnp.asarray(np.min(statistic_host, axis=0), dtype=dtype)
        self.feature_maximum = jnp.asarray(np.max(statistic_host, axis=0), dtype=dtype)
        self.solve_plan = plan
        self.support_size = support_size
        self._signature = ExponentialFamilySignature(
            str(family_id),
            dimension,
            (),
            "counting",
            str(identifier),
            f"finite-support-natural-{dimension}",
        )

    @property
    def signature(self) -> ExponentialFamilySignature:
        return self._signature

    def probabilities_from_natural(self, natural: NaturalCoordinates, /) -> Array:
        if natural.signature.key != self.signature.key:
            raise ValueError("Natural-coordinate signature does not match the family.")
        return self._probabilities(natural.values)

    def _logits(self, natural_values: Array, /) -> Array:
        return (
            natural_values @ jnp.swapaxes(self.statistics, -1, -2)
            + self.log_base_probabilities
        )

    def _probabilities(self, natural_values: Array, /) -> Array:
        return jax.nn.softmax(self._logits(natural_values), axis=-1)

    def _natural_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        shape = values.shape[:-1]
        return _natural_domain_result(
            self.signature,
            values,
            interior=jnp.ones(shape, dtype=jnp.bool_),
            boundary=jnp.zeros(shape, dtype=jnp.bool_),
        )

    def _mean_domain(self, values: Array, /) -> ExponentialFamilyDomainResult:
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(self.feature_minimum), jnp.abs(self.feature_maximum)), 1.0
        )
        tolerance = 32.0 * jnp.finfo(values.dtype).eps * scale
        lower_ok = jnp.all(values >= self.feature_minimum - tolerance, axis=-1)
        upper_ok = jnp.all(values <= self.feature_maximum + tolerance, axis=-1)
        strictly_lower = jnp.all(values > self.feature_minimum + tolerance, axis=-1)
        strictly_upper = jnp.all(values < self.feature_maximum - tolerance, axis=-1)
        inside_bounds = lower_ok & upper_ok
        return _mean_domain_result(
            self.signature,
            values,
            interior=inside_bounds & strictly_lower & strictly_upper,
            boundary=inside_bounds & ~(strictly_lower & strictly_upper),
        )

    def _sufficient_statistics(self, value: ArrayLike, /) -> StatisticBatch:
        raw = jnp.asarray(value)
        valid = (
            jnp.isfinite(raw)
            & (raw >= 0)
            & (raw < self.support_size)
            & (raw == jnp.floor(raw))
        )
        safe = jnp.where(valid, raw, 0).astype(jnp.int32)
        return StatisticBatch(self.statistics[safe], valid, self.signature)

    def _log_base_density(self, value: ArrayLike, /) -> Array:
        raw = jnp.asarray(value)
        valid = (
            jnp.isfinite(raw)
            & (raw >= 0)
            & (raw < self.support_size)
            & (raw == jnp.floor(raw))
        )
        safe = jnp.where(valid, raw, 0).astype(jnp.int32)
        return jnp.where(valid, self.log_base_probabilities[safe], -jnp.inf)

    def _log_normalizer(self, natural_values: Array, /) -> Array:
        return jax.nn.logsumexp(self._logits(natural_values), axis=-1)

    def _mean_values(self, natural_values: Array, /) -> Array:
        return self._probabilities(natural_values) @ self.statistics

    def _natural_from_mean_values(self, mean_values: Array, /) -> Array:
        return solve_finite_support_mean(
            self,
            MeanCoordinates(mean_values, self.signature),
            plan=self.solve_plan,
        ).conversion.natural.values

    def _natural_from_mean_result(
        self,
        mean: MeanCoordinates,
        domain: ExponentialFamilyDomainResult,
        /,
    ) -> ExponentialFamilyConversionResult:
        return solve_finite_support_mean(
            self,
            mean,
            plan=self.solve_plan,
            domain=domain,
        ).conversion

    def _sample(
        self,
        key,
        natural_values: Array,
        sample_shape: tuple[int, ...],
        /,
    ) -> Array:
        return jr.categorical(
            key,
            self._logits(natural_values),
            axis=-1,
            shape=sample_shape + natural_values.shape[:-1],
        )


def solve_finite_support_mean(
    family: FiniteSupportExponentialFamily,
    mean: MeanCoordinates | ArrayLike,
    /,
    *,
    plan: FiniteSupportNaturalSolvePlan | None = None,
    initial: NaturalCoordinates | ArrayLike | None = None,
    domain: ExponentialFamilyDomainResult | None = None,
) -> FiniteSupportSolveResult:
    """Invert finite-support means with warm-started safeguarded Newton steps."""
    if not isinstance(family, FiniteSupportExponentialFamily):
        raise TypeError("family must be a FiniteSupportExponentialFamily.")
    selected_plan = family.solve_plan if plan is None else plan
    if not isinstance(selected_plan, FiniteSupportNaturalSolvePlan):
        raise TypeError("plan must be a FiniteSupportNaturalSolvePlan.")
    target = mean if isinstance(mean, MeanCoordinates) else family.mean(mean)
    if target.signature.key != family.signature.key:
        raise ValueError("Mean-coordinate signature does not match the family.")
    admission = family.mean_domain(target) if domain is None else domain
    if admission.signature.key != family.signature.key:
        raise ValueError("Mean-domain signature does not match the family.")
    if initial is None:
        initial_values = jnp.zeros_like(target.values)
    elif isinstance(initial, NaturalCoordinates):
        if initial.signature.key != family.signature.key:
            raise ValueError("Initial natural-coordinate signature does not match.")
        initial_values = jnp.asarray(initial.values, dtype=target.values.dtype)
    else:
        initial_values = jnp.asarray(initial, dtype=target.values.dtype)
    if initial_values.shape != target.values.shape:
        raise ValueError("initial natural coordinates must match the target shape.")

    batch_shape = target.batch_shape
    dimension = family.signature.dimension
    batch_size = math.prod(batch_shape) if batch_shape else 1
    target_flat = target.values.reshape((batch_size, dimension))
    eta = initial_values.reshape((batch_size, dimension))
    admitted = admission.valid.reshape((batch_size,))
    active = admitted & jnp.all(jnp.isfinite(eta), axis=-1)
    factorization_failed = jnp.zeros((batch_size,), dtype=jnp.bool_)
    line_search_failed = jnp.zeros((batch_size,), dtype=jnp.bool_)
    iteration_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    factors = jnp.asarray(selected_plan.line_search_factors, dtype=target.values.dtype)
    statistics = family.statistics.astype(target.values.dtype)
    effective_tolerance = jnp.maximum(
        selected_plan.residual_tolerance,
        256.0 * jnp.finfo(target.values.dtype).eps,
    ) * jnp.maximum(jnp.linalg.norm(target_flat, axis=-1), 1.0)

    def evaluate(values: Array) -> tuple[Array, Array, Array]:
        probabilities = jax.nn.softmax(
            values @ jnp.swapaxes(statistics, -1, -2)
            + family.log_base_probabilities.astype(values.dtype),
            axis=-1,
        )
        means = probabilities @ statistics
        weighted = probabilities[..., :, None] * statistics
        second = jnp.swapaxes(weighted, -1, -2) @ statistics
        covariance = second - means[..., :, None] * means[..., None, :]
        return probabilities, means, covariance

    def body(_, state):
        (
            current,
            current_active,
            factor_failed,
            search_failed,
            counts,
        ) = state
        _, current_mean, covariance = evaluate(current)
        residual_vector = target_flat - current_mean
        current_residual = jnp.linalg.norm(residual_vector, axis=-1)
        preconverged = current_active & (current_residual <= effective_tolerance)
        solve_active = current_active & ~preconverged
        safe_covariance = jnp.where(
            solve_active[:, None, None],
            covariance,
            jnp.eye(dimension, dtype=covariance.dtype)[None, :, :],
        )
        safe_rhs = jnp.where(solve_active[:, None], residual_vector, 0.0)
        if selected_plan.portable:
            direction, solve_failed = _portable_positive_definite_solve(
                safe_covariance,
                safe_rhs,
            )
            local_factor_failed = solve_active & solve_failed
        else:
            factorization = prepare_local_block_factorization(
                safe_covariance,
                positive_definite=True,
            )
            solved = solve_local_blocks_detailed(factorization, safe_rhs[..., None])
            direction = solved.value[..., 0]
            local_factor_failed = solve_active & solved.failed_blocks

        candidate_values = (
            current[None, :, :] + factors[:, None, None] * direction[None, :, :]
        )
        _, candidate_means, _ = evaluate(candidate_values)
        candidate_residuals = jnp.linalg.norm(
            candidate_means - target_flat[None, :, :], axis=-1
        )
        candidate_residuals = jnp.where(
            jnp.all(jnp.isfinite(candidate_values), axis=-1),
            candidate_residuals,
            jnp.inf,
        )
        if selected_plan.portable:
            chosen = candidate_values[0]
            chosen_residual = candidate_residuals[0]
            for candidate_index in range(1, len(selected_plan.line_search_factors)):
                better = candidate_residuals[candidate_index] < chosen_residual
                chosen = jnp.where(
                    better[:, None],
                    candidate_values[candidate_index],
                    chosen,
                )
                chosen_residual = jnp.where(
                    better,
                    candidate_residuals[candidate_index],
                    chosen_residual,
                )
        else:
            best = jnp.argmin(candidate_residuals, axis=0)
            chosen = jnp.take_along_axis(
                candidate_values,
                best[None, :, None],
                axis=0,
            )[0]
            chosen_residual = jnp.take_along_axis(
                candidate_residuals,
                best[None, :],
                axis=0,
            )[0]
        local_search_failed = (
            solve_active & ~local_factor_failed & ~(chosen_residual < current_residual)
        )
        usable = solve_active & ~local_factor_failed & ~local_search_failed
        updated = jnp.where(usable[:, None], chosen, current)
        converged = preconverged | (usable & (chosen_residual <= effective_tolerance))
        next_active = (
            solve_active & ~local_factor_failed & ~local_search_failed & ~converged
        )
        next_counts = jnp.where(solve_active, counts + 1, counts)
        return (
            updated,
            next_active,
            factor_failed | local_factor_failed,
            search_failed | local_search_failed,
            next_counts,
        )

    eta, active, factorization_failed, line_search_failed, iteration_count = (
        jax.lax.fori_loop(
            0,
            selected_plan.maximum_steps,
            body,
            (eta, active, factorization_failed, line_search_failed, iteration_count),
        )
    )
    probabilities, reconstructed, _ = evaluate(eta)
    residual = jnp.linalg.norm(reconstructed - target_flat, axis=-1)
    minimum_probability = jnp.min(probabilities, axis=-1)
    finite = (
        jnp.all(jnp.isfinite(eta), axis=-1)
        & jnp.all(jnp.isfinite(probabilities), axis=-1)
        & jnp.isfinite(residual)
    )
    successful = (
        admitted
        & finite
        & ~factorization_failed
        & ~line_search_failed
        & (residual <= effective_tolerance)
        & (minimum_probability > selected_plan.minimum_probability)
    )
    domain_status = admission.status.reshape((batch_size,))
    status = jnp.where(
        successful,
        EXPONENTIAL_FAMILY_SUCCESS,
        jnp.where(
            ~admitted,
            domain_status,
            jnp.where(
                ~finite,
                EXPONENTIAL_FAMILY_NONFINITE,
                jnp.where(
                    minimum_probability <= selected_plan.minimum_probability,
                    EXPONENTIAL_FAMILY_MEAN_BOUNDARY,
                    EXPONENTIAL_FAMILY_NONCONVERGED,
                ),
            ),
        ),
    )
    natural_values = jnp.where(successful[:, None], eta, jnp.nan).reshape(
        batch_shape + (dimension,)
    )
    conversion = ExponentialFamilyConversionResult(
        mean=target,
        natural=NaturalCoordinates(natural_values, family.signature),
        valid=successful.reshape(batch_shape),
        status=status.reshape(batch_shape),
        residual=jnp.where(successful, residual, jnp.inf).reshape(batch_shape),
        iterations=iteration_count.reshape(batch_shape),
        method_id=f"finite-support-newton:{selected_plan.plan_id}",
    )
    return FiniteSupportSolveResult(
        conversion=conversion,
        probabilities=probabilities.reshape(batch_shape + (family.support_size,)),
        evidence=FiniteSupportSolveEvidence(
            residual=residual.reshape(batch_shape),
            iterations=iteration_count.reshape(batch_shape),
            minimum_probability=minimum_probability.reshape(batch_shape),
            factorization_failed=factorization_failed.reshape(batch_shape),
            line_search_failed=line_search_failed.reshape(batch_shape),
            successful=successful.reshape(batch_shape),
        ),
    )


__all__ = [
    "FiniteSupportExponentialFamily",
    "FiniteSupportNaturalSolvePlan",
    "FiniteSupportSolveEvidence",
    "FiniteSupportSolveResult",
    "solve_finite_support_mean",
]
