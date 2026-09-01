#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.special import betaln, gammaln, xlogy
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint
from ..._numerics import solve_weighted_least_squares
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, LinearSystem, prepare, solve_many
from ...optim import (
    Bounds,
    minimize,
    OptimizationTermination,
    ProjectedLBFGS,
)
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._assay import CountAssay
from ._design import ExperimentalDesign


GLM_SUCCESS = 0
GLM_RANK_DEFICIENT = 1
GLM_INSUFFICIENT_DF = 2
GLM_ALL_ZERO = 3
GLM_NONFINITE = 4
GLM_OPTIMIZATION_FAILED = 5


def _glm_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "negative-binomial-glm",
        MethodKind.EXACT_MODEL,
        ExecutionKind.ITERATIVE_TOLERANCE,
        DifferentiationKind.NONE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            "Log-linked coefficients use declared offsets; covariance is inverse "
            "expected Fisher information on each feature's observed rows."
        ),
        truncation_statement="All optimization iterations stop only by the declared tolerance policy.",
        capacity_semantics="One fixed-width native optimization is solved per assay feature.",
        assumptions=(
            "Counts follow an NB2 variance law mu + dispersion * mu^2.",
            "Experimental rows are independent after biological aggregation.",
        ),
        nondifferentiable_outputs=(
            "coefficients",
            "fitted_mean",
            "covariance",
            "log_likelihood",
            "rank",
            "residual_degrees_of_freedom",
            "status",
            "valid",
        ),
        absolute_tolerance=1.0e-6,
        relative_tolerance=1.0e-6,
    )


def _positive_mean(value: Array, /) -> Array:
    dtype = value.dtype
    lower = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    upper_log = jnp.log(jnp.asarray(jnp.finfo(dtype).max, dtype=dtype)) - 2.0
    return jnp.maximum(jnp.exp(jnp.minimum(value, upper_log)), lower)


def _negative_binomial_log_probability(y: Array, mu: Array, alpha: Array, /) -> Array:
    poisson = xlogy(y, mu) - mu - gammaln(y + 1.0)
    safe_alpha = jnp.where(alpha > 0.0, alpha, 1.0)
    inverse = 1.0 / safe_alpha
    log_normalizer = -betaln(inverse, y + 1.0) - jnp.log(inverse + y)
    log_scale = jnp.log1p(safe_alpha * mu)
    negative_binomial = (
        log_normalizer
        - log_scale / safe_alpha
        + xlogy(y, safe_alpha * mu)
        - y * log_scale
    )
    return jnp.where(alpha == 0.0, poisson, negative_binomial)


def negative_binomial_log_probability(
    counts: ArrayLike,
    mean: ArrayLike,
    dispersion: ArrayLike,
    /,
) -> Array:
    """Elementwise stable NB2 log probability with the Poisson boundary."""

    y = jnp.asarray(counts)
    mu = jnp.asarray(mean)
    alpha = jnp.asarray(dispersion)
    dtype = jnp.result_type(y, mu, alpha, float)
    y = y.astype(dtype)
    mu = mu.astype(dtype)
    alpha = alpha.astype(dtype)
    y = eqx.error_if(
        y,
        jnp.any((y < 0.0) | (y != jnp.floor(y))),
        "counts must be nonnegative integers.",
    )
    mu = eqx.error_if(
        mu,
        jnp.any(~jnp.isfinite(mu) | (mu <= 0.0)),
        "mean must be finite and positive.",
    )
    alpha = eqx.error_if(
        alpha,
        jnp.any(~jnp.isfinite(alpha) | (alpha < 0.0)),
        "dispersion must be finite and nonnegative.",
    )
    return _negative_binomial_log_probability(y, mu, alpha)


def negative_binomial_log_likelihood(
    counts: ArrayLike,
    mean: ArrayLike,
    dispersion: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
) -> Array:
    """Sum NB2 log probabilities over an explicit observation mask."""

    values = negative_binomial_log_probability(counts, mean, dispersion)
    if mask is None:
        return compensated_sum(values)
    included = jnp.asarray(mask, dtype=bool)
    if included.shape != values.shape:
        included = jnp.broadcast_to(included, values.shape)
    return compensated_sum(jnp.where(included, values, 0.0))


def _objective(parameters: Array, arguments: tuple[Array, ...], /) -> Array:
    matrix, counts, offsets, included, dispersion = arguments
    mean = _positive_mean(matrix @ parameters + offsets)
    log_probability = _negative_binomial_log_probability(counts, mean, dispersion)
    return -compensated_sum(jnp.where(included, log_probability, 0.0))


class NegativeBinomialGLMFit(StrictModule):
    """Per-feature native NB2 GLM fits and estimability projections."""

    coefficients: Array
    fitted_mean: Array
    covariance: Array
    information: Array
    coefficient_projection: Array
    log_likelihood: Array
    dispersion: Array
    offsets: Array
    design_matrix: Array
    observed_mask: Array
    rank: Array
    sample_count: Array
    residual_degrees_of_freedom: Array
    optimization_status: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    num_samples: int = eqx.field(static=True)
    num_features: int = eqx.field(static=True)
    num_coefficients: int = eqx.field(static=True)
    likelihood_data_id: str = eqx.field(static=True)


def _offset_matrix(
    offsets: ArrayLike | None,
    samples: int,
    features: int,
    dtype,
    /,
) -> Array:
    if offsets is None:
        return jnp.zeros((samples, features), dtype=dtype)
    values = jnp.asarray(offsets, dtype=dtype)
    if values.shape == (samples,):
        return jnp.broadcast_to(values[:, None], (samples, features))
    if values.shape != (samples, features):
        raise ValueError("offsets must have shape (samples,) or (samples, features).")
    return values


def _dispersion_vector(
    dispersion: ArrayLike,
    features: int,
    dtype,
    /,
) -> Array:
    values = jnp.asarray(dispersion, dtype=dtype)
    if values.ndim == 0:
        values = jnp.broadcast_to(values, (features,))
    if values.shape != (features,):
        raise ValueError(f"dispersion must be scalar or have shape ({features},).")
    return eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values) | (values < 0.0)),
        "dispersion must be finite and nonnegative.",
    )


def fit_negative_binomial_glm(
    assay: CountAssay,
    design: ExperimentalDesign,
    dispersion: ArrayLike,
    /,
    *,
    offsets: ArrayLike | None = None,
    maximum_steps: int = 256,
    rcond: float | None = None,
) -> NegativeBinomialGLMFit:
    """Fit independent feature-wise NB2 GLMs with the native L-BFGS solver."""

    if not isinstance(assay, CountAssay):
        raise TypeError("assay must be a CountAssay.")
    if not isinstance(design, ExperimentalDesign):
        raise TypeError("design must be an ExperimentalDesign.")
    if assay.num_samples != design.num_samples:
        raise ValueError("assay and design sample dimensions do not match.")
    steps = int(maximum_steps)
    if steps < 1:
        raise ValueError("maximum_steps must be positive.")

    count_values, assay_observed, _, _ = assay.dense_components()
    counts = count_values.astype(float)
    matrix = design.matrix.astype(counts.dtype)
    observed = assay_observed & design.valid_rows[:, None]
    samples, features = counts.shape
    coefficients = design.num_coefficients
    offset_matrix = _offset_matrix(offsets, samples, features, counts.dtype)
    dispersions = _dispersion_vector(dispersion, features, counts.dtype)
    if bool(jnp.any(observed & ~jnp.isfinite(offset_matrix))):
        raise ValueError("offsets must be finite on every observed design row.")
    likelihood_data_id = array_tree_fingerprint(
        (counts, observed, offset_matrix, dispersions)
    )["sha256"]
    termination = OptimizationTermination(
        absolute_optimality=1.0e-6,
        relative_optimality=1.0e-6,
        absolute_step=1.0e-8,
        relative_step=1.0e-7,
        maximum_steps=steps,
    )
    method = ProjectedLBFGS()
    unconstrained_bounds = Bounds()
    eye = jnp.eye(coefficients, dtype=counts.dtype)

    coefficient_values: list[Array] = []
    mean_values: list[Array] = []
    covariance_values: list[Array] = []
    information_values: list[Array] = []
    projection_values: list[Array] = []
    log_likelihood_values: list[Array] = []
    rank_values: list[Array] = []
    sample_count_values: list[Array] = []
    residual_df_values: list[Array] = []
    optimization_status_values: list[Array] = []
    valid_values: list[Array] = []
    status_values: list[Array] = []
    evidence_values: list[Array] = []

    for feature in range(features):
        y = counts[:, feature]
        included = observed[:, feature]
        offset = jnp.where(included, offset_matrix[:, feature], 0.0)
        initial_response = jnp.log(y + 0.5) - offset
        initial_fit = solve_weighted_least_squares(
            matrix,
            initial_response,
            mask=included,
            ridge=1.0e-8,
            rcond=rcond,
            min_samples=1,
            max_features=coefficients,
        )
        projection_fit = solve_weighted_least_squares(
            matrix,
            matrix,
            mask=included,
            rcond=rcond,
            min_samples=1,
            max_features=coefficients,
        )
        rank = projection_fit.rank.astype(jnp.int32)
        sample_count = jnp.sum(included).astype(jnp.int32)
        residual_df = sample_count - rank
        optimized = minimize(
            _objective,
            initial_fit.raw_coefficients,
            method=method,
            bounds=unconstrained_bounds,
            termination=termination,
            args=(matrix, y, offset, included, dispersions[feature]),
        )
        beta = jnp.asarray(optimized.parameters)
        mean = _positive_mean(matrix @ beta + offset)
        weight = jnp.where(
            included,
            mean / (1.0 + dispersions[feature] * mean),
            0.0,
        )
        information = jnp.swapaxes(matrix, 0, 1) @ (weight[:, None] * matrix)
        full_rank = rank == coefficients
        enough_df = residual_df > 0
        total_count = compensated_sum(jnp.where(included, y, 0.0))
        nonzero = total_count > 0.0
        safe_information = jnp.where(
            (full_rank & enough_df & nonzero)[None, None], information, eye
        )
        prepared = prepare(LinearSystem(DenseLinearOperator(safe_information)))
        inverse = solve_many(prepared, eye)
        covariance = jnp.where(
            (full_rank & enough_df & nonzero)[None, None],
            inverse.value,
            jnp.full_like(inverse.value, jnp.inf),
        )
        log_likelihood = negative_binomial_log_likelihood(
            y,
            mean,
            dispersions[feature],
            mask=included,
        )
        finite = (
            jnp.all(jnp.isfinite(beta))
            & jnp.all(jnp.isfinite(mean))
            & jnp.isfinite(log_likelihood)
            & jnp.all(jnp.isfinite(information))
        )
        optimization_ok = optimized.successful
        valid = full_rank & enough_df & nonzero & finite & optimization_ok
        status = jnp.where(
            ~finite,
            GLM_NONFINITE,
            jnp.where(
                ~full_rank,
                GLM_RANK_DEFICIENT,
                jnp.where(
                    ~enough_df,
                    GLM_INSUFFICIENT_DF,
                    jnp.where(
                        ~nonzero,
                        GLM_ALL_ZERO,
                        jnp.where(
                            ~optimization_ok,
                            GLM_OPTIMIZATION_FAILED,
                            GLM_SUCCESS,
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)

        coefficient_values.append(beta)
        mean_values.append(mean)
        covariance_values.append(covariance)
        information_values.append(information)
        projection_values.append(projection_fit.raw_coefficients)
        log_likelihood_values.append(log_likelihood)
        rank_values.append(rank)
        sample_count_values.append(sample_count)
        residual_df_values.append(residual_df)
        optimization_status_values.append(optimized.status)
        valid_values.append(valid)
        status_values.append(status)
        evidence_values.append(
            jnp.stack(
                (
                    sample_count.astype(counts.dtype),
                    rank.astype(counts.dtype),
                    residual_df.astype(counts.dtype),
                    total_count,
                    dispersions[feature],
                )
            )
        )

    return NegativeBinomialGLMFit(
        coefficients=jnp.stack(coefficient_values, axis=0),
        fitted_mean=jnp.stack(mean_values, axis=1),
        covariance=jnp.stack(covariance_values, axis=0),
        information=jnp.stack(information_values, axis=0),
        coefficient_projection=jnp.stack(projection_values, axis=0),
        log_likelihood=jnp.stack(log_likelihood_values),
        dispersion=dispersions,
        offsets=offset_matrix,
        design_matrix=matrix,
        observed_mask=observed,
        rank=jnp.stack(rank_values),
        sample_count=jnp.stack(sample_count_values),
        residual_degrees_of_freedom=jnp.stack(residual_df_values),
        optimization_status=jnp.stack(optimization_status_values),
        valid=jnp.stack(valid_values),
        status=jnp.stack(status_values),
        evidence=jnp.stack(evidence_values, axis=0),
        method_contract=_glm_contract(),
        num_samples=samples,
        num_features=features,
        num_coefficients=coefficients,
        likelihood_data_id=likelihood_data_id,
    )


__all__ = [
    "GLM_ALL_ZERO",
    "GLM_INSUFFICIENT_DF",
    "GLM_NONFINITE",
    "GLM_OPTIMIZATION_FAILED",
    "GLM_RANK_DEFICIENT",
    "GLM_SUCCESS",
    "NegativeBinomialGLMFit",
    "fit_negative_binomial_glm",
    "negative_binomial_log_likelihood",
    "negative_binomial_log_probability",
]
