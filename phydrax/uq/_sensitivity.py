#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from math import isfinite
from typing import Any, cast, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, PyTree

from .._exponential_family import (
    AbstractExponentialFamily,
    NaturalCoordinates,
)
from .._frozendict import frozendict
from .._sampling import get_sampler
from .._strict import StrictModule
from ..linalg import (
    DenseLinearOperator,
    EmpiricalGramLinearOperator,
    FactorizationPolicy,
    inverse,
    OperatorProperties,
)
from ._distributions import AbstractDistribution


class SobolResult(StrictModule):
    """First-order and total-order global sensitivity fields."""

    parameter_names: tuple[str, ...]
    first_order: cx.Field
    total_order: cx.Field
    output_variance: cx.Field
    num_samples: int
    parameter_dim: str

    def __init__(
        self,
        *,
        parameter_names: tuple[str, ...],
        first_order: cx.Field,
        total_order: cx.Field,
        output_variance: cx.Field,
        num_samples: int,
        parameter_dim: str,
    ):
        expected = len(parameter_names)
        if (
            first_order.dims != total_order.dims
            or first_order.data.shape != total_order.data.shape
        ):
            raise ValueError("Sobol first-order and total-order fields must align.")
        if (
            first_order.dims[0] != parameter_dim
            or int(first_order.data.shape[0]) != expected
        ):
            raise ValueError(
                "Sobol result parameter axis does not match parameter_names."
            )
        if first_order.dims[1:] != output_variance.dims:
            raise ValueError("Sobol output dimensions do not match output variance.")
        self.parameter_names = parameter_names
        self.first_order = first_order
        self.total_order = total_order
        self.output_variance = output_variance
        self.num_samples = int(num_samples)
        self.parameter_dim = parameter_dim


def sobol_indices(
    function,
    distributions: Mapping[str, AbstractDistribution],
    /,
    *,
    num_samples: int,
    key,
    sampler: str = "sobol_scrambled",
    batch_size: int | None = None,
    parameter_dim: str = "__phydra_uq_parameter",
    call_style: Literal["keywords", "mapping"] = "keywords",
    reduce_output: Literal["mean", "sum"] | None = None,
    mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    **kwargs: Any,
) -> SobolResult:
    """Saltelli first-order and Jansen total-order indices from one joint QMC design."""
    if not callable(function):
        raise TypeError("function must be callable.")
    names = tuple(distributions)
    if not names:
        raise ValueError("distributions must be non-empty.")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("Distribution labels must be non-empty strings.")
    if not isinstance(parameter_dim, str):
        raise TypeError("parameter_dim must be a string.")
    if parameter_dim in names or not parameter_dim:
        raise ValueError(
            "parameter_dim must be non-empty and distinct from input labels."
        )
    count = int(num_samples)
    if count < 2:
        raise ValueError("num_samples must be at least two.")
    if reduce_output not in (None, "mean", "sum"):
        raise ValueError("reduce_output must be None, 'mean', or 'sum'.")
    if call_style not in ("keywords", "mapping"):
        raise ValueError("call_style must be 'keywords' or 'mapping'.")
    for name, distribution in distributions.items():
        if not isinstance(distribution, AbstractDistribution):
            raise TypeError(f"Distribution {name!r} must implement AbstractDistribution.")
    dimension = len(names)
    unit = get_sampler(sampler)(count, 2 * dimension, key)
    a = jnp.stack(
        tuple(
            distribution.icdf(unit[:, index])
            for index, distribution in enumerate(distributions.values())
        ),
        axis=1,
    )
    b = jnp.stack(
        tuple(
            distribution.icdf(unit[:, dimension + index])
            for index, distribution in enumerate(distributions.values())
        ),
        axis=1,
    )
    f_a, output_dims = _evaluate_design(
        function,
        names,
        a,
        batch_size=batch_size,
        call_style=call_style,
        **kwargs,
    )
    f_b, b_dims = _evaluate_design(
        function,
        names,
        b,
        batch_size=batch_size,
        call_style=call_style,
        **kwargs,
    )
    if f_a.shape != f_b.shape or output_dims != b_dims:
        raise ValueError("A and B designs produced inconsistent output structure.")
    if reduce_output is not None:
        f_a = _reduce_sample_outputs(
            f_a, reduction=reduce_output, mask=mask, weights=weights
        )
        f_b = _reduce_sample_outputs(
            f_b, reduction=reduce_output, mask=mask, weights=weights
        )
        output_dims = ()
    elif mask is not None or weights is not None:
        raise ValueError("mask and weights require reduce_output='mean' or 'sum'.")
    combined = jnp.concatenate((f_a, f_b), axis=0)
    variance = jnp.var(combined, axis=0, ddof=1)
    tolerance = jnp.finfo(variance.dtype).eps * jnp.maximum(
        1.0, jnp.mean(combined**2, axis=0)
    )
    if bool(jnp.any(~jnp.isfinite(variance))) or bool(jnp.any(variance <= tolerance)):
        raise ValueError("Sobol indices require finite, non-zero output variance.")
    first = []
    total = []
    for index in range(dimension):
        hybrid = a.at[:, index].set(b[:, index])
        f_ab, hybrid_dims = _evaluate_design(
            function,
            names,
            hybrid,
            batch_size=batch_size,
            call_style=call_style,
            **kwargs,
        )
        if reduce_output is not None:
            f_ab = _reduce_sample_outputs(
                f_ab, reduction=reduce_output, mask=mask, weights=weights
            )
            hybrid_dims = ()
        if f_ab.shape != f_a.shape or hybrid_dims != output_dims:
            raise ValueError(
                "Hybrid Sobol design produced inconsistent output structure."
            )
        first.append(jnp.mean(f_b * (f_ab - f_a), axis=0) / variance)
        total.append(0.5 * jnp.mean((f_a - f_ab) ** 2, axis=0) / variance)
    first_data = jnp.stack(tuple(first), axis=0)
    total_data = jnp.stack(tuple(total), axis=0)
    return SobolResult(
        parameter_names=names,
        first_order=cx.Field(first_data, dims=(parameter_dim, *output_dims)),
        total_order=cx.Field(total_data, dims=(parameter_dim, *output_dims)),
        output_variance=cx.Field(variance, dims=output_dims),
        num_samples=count,
        parameter_dim=parameter_dim,
    )


def _evaluate_design(
    function,
    names: tuple[str, ...],
    design,
    /,
    *,
    batch_size: int | None,
    call_style: Literal["keywords", "mapping"],
    **kwargs: Any,
):
    count = int(design.shape[0])
    chunk = count if batch_size is None else int(batch_size)
    if chunk <= 0:
        raise ValueError("batch_size must be positive.")

    def evaluate(row):
        arguments = {name: row[index] for index, name in enumerate(names)}
        if call_style == "keywords":
            return function(**arguments, **kwargs)
        return function(frozendict(arguments), **kwargs)

    template = evaluate(design[0])
    if isinstance(template, cx.Field):
        template_data = jnp.asarray(template.data)
        output_dims = tuple(template.dims)
        returns_field = True
    else:
        try:
            template_data = jnp.asarray(template)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Sensitivity function must return one array or coordax.Field."
            ) from exc
        output_dims = (None,) * template_data.ndim
        returns_field = False

    def evaluate_data(row):
        value = evaluate(row)
        if returns_field:
            if not isinstance(value, cx.Field):
                raise TypeError("Sensitivity output type changed between samples.")
            if value.dims != output_dims:
                raise ValueError("Sensitivity field dimensions changed between samples.")
            return value.data
        return jnp.asarray(value)

    parts = []
    for start in range(0, count, chunk):
        data = jnp.asarray(jax.vmap(evaluate_data)(design[start : start + chunk]))
        if data.shape[1:] != template_data.shape:
            raise ValueError("Sensitivity chunks produced inconsistent output shape.")
        parts.append(data)
    values = jnp.concatenate(tuple(parts), axis=0)
    if bool(jnp.any(~jnp.isfinite(values))):
        raise FloatingPointError("Sensitivity evaluation produced non-finite outputs.")
    return values, output_dims


def _reduce_sample_outputs(
    values,
    /,
    *,
    reduction: Literal["mean", "sum"],
    mask: ArrayLike | None,
    weights: ArrayLike | None,
):
    output_shape = values.shape[1:]
    effective = jnp.ones(output_shape, dtype=values.dtype)
    if mask is not None:
        effective = effective * jnp.broadcast_to(
            jnp.asarray(mask, dtype=bool), output_shape
        )
    if weights is not None:
        weight_array = jnp.broadcast_to(jnp.asarray(weights, dtype=float), output_shape)
        if bool(jnp.any(~jnp.isfinite(weight_array))) or bool(
            jnp.any(weight_array < 0.0)
        ):
            raise ValueError("weights must be finite and non-negative.")
        effective = effective * weight_array
    flat_values = values.reshape((int(values.shape[0]), -1))
    flat_weight = effective.reshape((-1,))
    sums = jnp.sum(flat_values * flat_weight, axis=1)
    if reduction == "sum":
        return sums
    denominator = jnp.sum(flat_weight)
    if not bool(denominator > 0.0):
        raise ValueError("Sensitivity reduction has zero total weight.")
    return sums / denominator


SENSITIVITY_SUCCESS = 0
SENSITIVITY_NONFINITE = 1
SENSITIVITY_INVALID_INFORMATION = 2


class SensitivityGradientResult(StrictModule):
    """Gradient estimate with estimator and random-mechanism provenance."""

    gradient: PyTree[Array]
    standard_error: PyTree[Array] | None
    valid: Array
    status: Array
    estimator_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    resampling_id: str | None = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    num_samples: int = eqx.field(static=True)

    def __init__(
        self,
        gradient: PyTree[Array],
        standard_error: PyTree[Array] | None,
        /,
        *,
        valid: ArrayLike,
        status: ArrayLike,
        estimator_id: str,
        method_id: str,
        noise_id: str | None,
        resampling_id: str | None,
        approximation: str,
        num_samples: int,
    ):
        if not estimator_id or not method_id or not approximation:
            raise ValueError("Sensitivity provenance IDs must be non-empty.")
        if noise_id is not None and not noise_id:
            raise ValueError("noise_id must be non-empty or None.")
        if resampling_id is not None and not resampling_id:
            raise ValueError("resampling_id must be non-empty or None.")
        if int(num_samples) <= 0:
            raise ValueError("num_samples must be positive.")
        self.gradient = gradient
        self.standard_error = standard_error
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.estimator_id = str(estimator_id)
        self.method_id = str(method_id)
        self.noise_id = noise_id
        self.resampling_id = resampling_id
        self.approximation = str(approximation)
        self.num_samples = int(num_samples)


class ResamplingScoreResult(StrictModule):
    """Likelihood-ratio contribution from one declared resampling operation."""

    gradient: PyTree[Array]
    standard_error: PyTree[Array]
    centered_scores: PyTree[Array]
    expected_centered_score: PyTree[Array]
    normalized_weights: Array
    ancestor_indices: Array
    valid: Array
    status: Array
    estimator_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    noise_id: str | None = eqx.field(static=True)
    resampling_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    num_particles: int = eqx.field(static=True)
    num_draws: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        gradient: PyTree[Array],
        standard_error: PyTree[Array],
        centered_scores: PyTree[Array],
        expected_centered_score: PyTree[Array],
        normalized_weights: ArrayLike,
        ancestor_indices: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        noise_id: str | None,
        resampling_id: str,
    ):
        weights = jnp.asarray(normalized_weights)
        ancestors = jnp.asarray(ancestor_indices, dtype=jnp.int32)
        if weights.ndim != 1:
            raise ValueError("normalized_weights must be rank one.")
        if ancestors.ndim != 1 or int(ancestors.size) == 0:
            raise ValueError("ancestor_indices must be non-empty and rank one.")
        if not resampling_id:
            raise ValueError("resampling_id must be non-empty.")
        self.gradient = gradient
        self.standard_error = standard_error
        self.centered_scores = centered_scores
        self.expected_centered_score = expected_centered_score
        self.normalized_weights = weights
        self.ancestor_indices = ancestors
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.estimator_id = "resampling_score"
        self.method_id = "categorical_log_probability_score"
        self.noise_id = noise_id
        self.resampling_id = str(resampling_id)
        self.approximation = "monte_carlo_likelihood_ratio"
        self.num_particles = int(weights.size)
        self.num_draws = int(ancestors.size)


class SensitivityActionResult(StrictModule):
    """One matrix-free curvature action and its operator provenance."""

    action: PyTree[Array]
    valid: Array
    status: Array
    operator_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    num_samples: int | None = eqx.field(static=True)

    def __init__(
        self,
        action: PyTree[Array],
        /,
        *,
        valid: ArrayLike,
        status: ArrayLike,
        operator_id: str,
        method_id: str,
        approximation: str,
        regularization: float,
        num_samples: int | None,
    ):
        if not operator_id or not method_id or not approximation:
            raise ValueError("Sensitivity action provenance IDs must be non-empty.")
        self.action = action
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.operator_id = str(operator_id)
        self.method_id = str(method_id)
        self.approximation = str(approximation)
        self.regularization = float(regularization)
        self.num_samples = None if num_samples is None else int(num_samples)


class EmpiricalDirectionsResult(StrictModule):
    """Dominant local Gramian directions from matrix-free derivative actions."""

    directions: Array
    strengths: Array
    valid: Array
    status: Array
    quantity: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    ambient_shape: tuple[int, ...] = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    rank: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        directions: ArrayLike,
        strengths: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        quantity: str,
        regularization: float,
        ambient_shape: tuple[int, ...],
    ):
        vectors = jnp.asarray(directions)
        values = jnp.asarray(strengths)
        if vectors.ndim != 2 or values.shape != (vectors.shape[1],):
            raise ValueError("Directions and strengths have incompatible shapes.")
        if not quantity:
            raise ValueError("quantity must be non-empty.")
        self.directions = vectors
        self.strengths = values
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.quantity = str(quantity)
        self.method_id = "matrix_free_actions_dense_eigh"
        self.approximation = "empirical_local_linearization"
        self.regularization = float(regularization)
        self.ambient_shape = tuple(int(size) for size in ambient_shape)
        self.ambient_dimension = int(vectors.shape[0])
        self.rank = int(vectors.shape[1])


class ExperimentDesignResult(StrictModule):
    """Scalar information-design objective with an explicit validity status."""

    value: Array
    eigenvalues: Array
    valid: Array
    status: Array
    criterion: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        value: ArrayLike,
        eigenvalues: ArrayLike,
        valid: ArrayLike,
        status: ArrayLike,
        criterion: str,
        method_id: str,
        approximation: str,
        regularization: float,
    ):
        spectrum = jnp.asarray(eigenvalues)
        if spectrum.ndim != 1 or int(spectrum.size) == 0:
            raise ValueError("eigenvalues must be a non-empty rank-1 array.")
        self.value = jnp.asarray(value)
        self.eigenvalues = spectrum
        self.valid = jnp.asarray(valid, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.criterion = str(criterion)
        self.method_id = str(method_id)
        self.approximation = str(approximation)
        self.regularization = float(regularization)
        self.dimension = int(spectrum.size)


def likelihood_ratio_gradient(
    values: ArrayLike,
    scores: PyTree[Array],
    /,
    *,
    baseline: Literal["mean", "none"] | ArrayLike | None = "mean",
    noise_id: str | None = None,
    method_id: str = "score_function",
) -> SensitivityGradientResult:
    """Estimate ``E[f score]`` while retaining the score-estimator identity."""
    observations = jnp.asarray(values)
    if observations.ndim == 0 or int(observations.shape[0]) < 2:
        raise ValueError("values must have at least two samples on axis zero.")
    count = int(observations.shape[0])
    score_leaves = jax.tree_util.tree_leaves(scores)
    if not score_leaves:
        raise ValueError("scores must contain at least one array leaf.")
    if any(
        jnp.asarray(leaf).ndim == 0 or int(jnp.asarray(leaf).shape[0]) != count
        for leaf in score_leaves
    ):
        raise ValueError("Every score leaf must share the values sample axis.")
    if isinstance(baseline, str):
        if baseline not in ("mean", "none"):
            raise ValueError("baseline must be 'mean', 'none', None, or an array.")
        center = jnp.mean(observations, axis=0) if baseline == "mean" else 0.0
        covariance_correction = count / (count - 1) if baseline == "mean" else 1.0
    elif baseline is None:
        center = 0.0
        covariance_correction = 1.0
    else:
        center = jnp.broadcast_to(jnp.asarray(baseline), observations.shape[1:])
        covariance_correction = 1.0
    centered = observations - center

    contribution_tree = jax.tree_util.tree_map(
        lambda score: jax.vmap(
            lambda value, score_value: jnp.tensordot(value, score_value, axes=0)
        )(centered, jnp.asarray(score)),
        scores,
    )
    gradient = jax.tree_util.tree_map(
        lambda values: covariance_correction * jnp.mean(values, axis=0),
        contribution_tree,
    )
    standard_error = jax.tree_util.tree_map(
        lambda values: (
            covariance_correction * jnp.std(values, axis=0, ddof=1) / jnp.sqrt(count)
        ),
        contribution_tree,
    )
    finite = jnp.all(jnp.isfinite(observations)) & _tree_all_finite(scores)
    finite = finite & _tree_all_finite(gradient) & _tree_all_finite(standard_error)
    status = jnp.where(finite, SENSITIVITY_SUCCESS, SENSITIVITY_NONFINITE)
    return SensitivityGradientResult(
        gradient,
        standard_error,
        valid=finite,
        status=status,
        estimator_id="likelihood_ratio",
        method_id=method_id,
        noise_id=noise_id,
        resampling_id=None,
        approximation="monte_carlo_score_function",
        num_samples=count,
    )


def fixed_noise_pathwise_gradient(
    function: Callable[[PyTree[Array], PyTree[Array]], ArrayLike],
    parameters: PyTree[Array],
    noise: PyTree[Array],
    /,
    *,
    noise_id: str,
    method: Literal["reverse", "forward"] = "reverse",
) -> SensitivityGradientResult:
    """Differentiate a response while holding one named noise realization fixed."""
    if not callable(function):
        raise TypeError("function must be callable.")
    if not noise_id:
        raise ValueError("noise_id must be non-empty.")

    def evaluate(value):
        response = function(value, noise)
        return response, response

    if method == "reverse":
        gradient, response = jax.jacrev(evaluate, has_aux=True)(parameters)
        method_id = "jax_jacrev_fixed_noise"
    elif method == "forward":
        gradient, response = jax.jacfwd(evaluate, has_aux=True)(parameters)
        method_id = "jax_jacfwd_fixed_noise"
    else:
        raise ValueError("method must be 'reverse' or 'forward'.")
    finite = (
        _tree_all_finite(parameters)
        & _tree_all_finite(noise)
        & _tree_all_finite(response)
        & _tree_all_finite(gradient)
    )
    status = jnp.where(finite, SENSITIVITY_SUCCESS, SENSITIVITY_NONFINITE)
    return SensitivityGradientResult(
        gradient,
        None,
        valid=finite,
        status=status,
        estimator_id="fixed_noise_pathwise",
        method_id=method_id,
        noise_id=noise_id,
        resampling_id=None,
        approximation="exact_autodiff_for_fixed_realization",
        num_samples=1,
    )


def resampling_score_gradient(
    values: ArrayLike,
    log_weights: ArrayLike,
    log_weight_scores: PyTree[Array],
    ancestor_indices: ArrayLike,
    /,
    *,
    resampling_id: str,
    noise_id: str | None = None,
) -> ResamplingScoreResult:
    """Estimate the categorical resampling score, including normalization."""
    observations = jnp.asarray(values)
    logits = jnp.asarray(log_weights)
    ancestors = jnp.asarray(ancestor_indices)
    if logits.ndim != 1 or int(logits.size) == 0:
        raise ValueError("log_weights must be a non-empty rank-1 array.")
    count = int(logits.size)
    if observations.ndim == 0 or int(observations.shape[0]) != count:
        raise ValueError("values must share the particle axis of log_weights.")
    if ancestors.ndim != 1 or int(ancestors.size) == 0:
        raise ValueError("ancestor_indices must be a non-empty rank-1 array.")
    if not jnp.issubdtype(ancestors.dtype, jnp.integer):
        raise TypeError("ancestor_indices must have an integer dtype.")
    if bool(jnp.any((ancestors < 0) | (ancestors >= count))):
        raise ValueError("ancestor_indices contain an out-of-range particle index.")
    score_leaves = jax.tree_util.tree_leaves(log_weight_scores)
    if not score_leaves or any(
        jnp.asarray(leaf).ndim == 0 or int(jnp.asarray(leaf).shape[0]) != count
        for leaf in score_leaves
    ):
        raise ValueError("Every log-weight score must share the particle axis.")
    weights = jax.nn.softmax(logits)

    def score_mean(score):
        return jnp.tensordot(weights, jnp.asarray(score), axes=((0,), (0,)))

    centered_scores = jax.tree_util.tree_map(
        lambda score: jnp.asarray(score) - score_mean(score),
        log_weight_scores,
    )
    expected_score = jax.tree_util.tree_map(
        lambda score: jnp.tensordot(weights, score, axes=((0,), (0,))),
        centered_scores,
    )
    selected_values = observations[ancestors]

    contribution_tree = jax.tree_util.tree_map(
        lambda score: jax.vmap(
            lambda value, score_value: jnp.tensordot(value, score_value, axes=0)
        )(selected_values, score[ancestors]),
        centered_scores,
    )
    gradient = jax.tree_util.tree_map(
        lambda values: jnp.mean(values, axis=0),
        contribution_tree,
    )
    standard_error = jax.tree_util.tree_map(
        lambda values: jnp.std(values, axis=0, ddof=1) / jnp.sqrt(ancestors.size),
        contribution_tree,
    )
    finite = (
        jnp.all(jnp.isfinite(observations))
        & jnp.all(jnp.isfinite(logits))
        & _tree_all_finite(centered_scores)
        & _tree_all_finite(gradient)
        & _tree_all_finite(standard_error)
    )
    status = jnp.where(finite, SENSITIVITY_SUCCESS, SENSITIVITY_NONFINITE)
    return ResamplingScoreResult(
        gradient=gradient,
        standard_error=standard_error,
        centered_scores=centered_scores,
        expected_centered_score=expected_score,
        normalized_weights=weights,
        ancestor_indices=ancestors,
        valid=finite,
        status=status,
        noise_id=noise_id,
        resampling_id=resampling_id,
    )


def fisher_information_action(
    scores: ArrayLike,
    vector: ArrayLike,
    /,
    *,
    weights: ArrayLike | None = None,
    regularization: float = 0.0,
    method_id: str = "empirical_outer_product",
) -> SensitivityActionResult:
    """Apply an empirical Fisher matrix without materializing it."""
    score_array = jnp.asarray(scores)
    direction = jnp.asarray(vector)
    if score_array.ndim < 2 or int(score_array.shape[0]) == 0:
        raise ValueError("scores must have shape (sample, *parameter_shape).")
    if score_array.shape[1:] != direction.shape:
        raise ValueError("vector shape must match one score sample.")
    if jnp.iscomplexobj(score_array) or jnp.iscomplexobj(direction):
        raise TypeError(
            "fisher_information_action supports real scores; use a pairing-aware "
            "EmpiricalGramLinearOperator for complex geometry."
        )
    penalty = float(regularization)
    if not isfinite(penalty) or penalty < 0.0:
        raise ValueError("regularization must be finite and non-negative.")
    count = int(score_array.shape[0])
    flat_scores = score_array.reshape((count, -1))
    flat_direction = direction.reshape(-1)
    if weights is None:
        sample_weights = jnp.ones((count,), dtype=float)
        weights_valid = jnp.asarray(True)
    else:
        sample_weights = jnp.asarray(weights)
        if sample_weights.shape != (count,):
            raise ValueError("weights must have one entry per score sample.")
        weights_valid = (
            jnp.all(jnp.isfinite(sample_weights))
            & jnp.all(sample_weights >= 0.0)
            & (jnp.sum(sample_weights) > 0.0)
        )
    safe_weights = jnp.where(weights_valid, sample_weights, jnp.ones_like(sample_weights))
    feature_operator = DenseLinearOperator(
        flat_scores,
        operator_id="fisher-score-features",
    )
    gram = EmpiricalGramLinearOperator(
        feature_operator,
        safe_weights,
        centered=False,
        damping=penalty,
        operator_id="fisher-information",
    )
    action = gram.mv(flat_direction).reshape(direction.shape)
    finite = (
        jnp.all(jnp.isfinite(score_array))
        & jnp.all(jnp.isfinite(sample_weights))
        & weights_valid
        & jnp.all(jnp.isfinite(action))
    )
    status = jnp.where(finite, SENSITIVITY_SUCCESS, SENSITIVITY_NONFINITE)
    return SensitivityActionResult(
        action,
        valid=finite,
        status=status,
        operator_id="fisher_information",
        method_id=method_id,
        approximation="empirical_matrix_free",
        regularization=penalty,
        num_samples=count,
    )


def exponential_family_fisher_action(
    family: AbstractExponentialFamily,
    natural: NaturalCoordinates,
    direction: ArrayLike,
    /,
    *,
    regularization: float = 0.0,
    method_id: str = "jax_jvp_mean_map",
) -> SensitivityActionResult:
    """Apply an exact family Fisher action as the JVP of the mean map."""
    if not isinstance(family, AbstractExponentialFamily):
        raise TypeError("family must implement AbstractExponentialFamily.")
    if not isinstance(natural, NaturalCoordinates):
        raise TypeError("natural must be NaturalCoordinates.")
    if not method_id:
        raise ValueError("method_id must be non-empty.")
    penalty = _validate_regularization(regularization)
    vector = jnp.asarray(direction)
    if vector.shape != natural.values.shape:
        raise ValueError("direction must match the natural-coordinate shape.")
    domain = family.natural_domain(natural)
    action = family.fisher_action(natural, vector) + penalty * vector
    inputs_finite = jnp.all(jnp.isfinite(natural.values)) & jnp.all(jnp.isfinite(vector))
    action_finite = jnp.all(jnp.isfinite(action))
    domain_valid = jnp.all(domain.valid)
    valid = inputs_finite & domain_valid & action_finite
    status = jnp.where(
        ~inputs_finite,
        SENSITIVITY_NONFINITE,
        jnp.where(
            ~domain_valid,
            SENSITIVITY_INVALID_INFORMATION,
            jnp.where(
                action_finite,
                SENSITIVITY_SUCCESS,
                SENSITIVITY_NONFINITE,
            ),
        ),
    )
    return SensitivityActionResult(
        action,
        valid=valid,
        status=status,
        operator_id="fisher_information",
        method_id=method_id,
        approximation="exact_exponential_family",
        regularization=penalty,
        num_samples=None,
    )


def exponential_family_parameter_fisher_action(
    family: AbstractExponentialFamily,
    natural_fn: Callable[[PyTree[Array]], NaturalCoordinates | ArrayLike],
    parameters: PyTree[Array],
    vector: PyTree[Array],
    /,
    *,
    regularization: float = 0.0,
    method_id: str = "jax_jvp_family_pullback",
) -> SensitivityActionResult:
    """Apply ``Jηᵀ F(η) Jη`` without dense Jacobian or Fisher materialization."""
    if not isinstance(family, AbstractExponentialFamily):
        raise TypeError("family must implement AbstractExponentialFamily.")
    if not callable(natural_fn):
        raise TypeError("natural_fn must be callable.")
    if not method_id:
        raise ValueError("method_id must be non-empty.")
    penalty = _validate_regularization(regularization)

    def natural_values_fn(values):
        coordinates = natural_fn(values)
        if isinstance(coordinates, NaturalCoordinates):
            family.natural_domain(coordinates)
            return coordinates.values
        return family.natural(coordinates).values

    natural_values, linearized = jax.linearize(natural_values_fn, parameters)
    natural = family.natural(natural_values)
    natural_direction = linearized(vector)
    family_direction = family.fisher_action(natural, natural_direction)
    action = jax.linear_transpose(linearized, parameters)(family_direction)[0]
    action = jax.tree_util.tree_map(
        lambda value, direction_value: value + penalty * direction_value,
        action,
        vector,
    )
    domain = family.natural_domain(natural)
    inputs_finite = (
        _tree_all_finite(parameters)
        & _tree_all_finite(vector)
        & jnp.all(jnp.isfinite(natural_values))
        & jnp.all(jnp.isfinite(natural_direction))
    )
    action_finite = _tree_all_finite(action)
    domain_valid = jnp.all(domain.valid)
    valid = inputs_finite & domain_valid & action_finite
    status = jnp.where(
        ~inputs_finite,
        SENSITIVITY_NONFINITE,
        jnp.where(
            ~domain_valid,
            SENSITIVITY_INVALID_INFORMATION,
            jnp.where(
                action_finite,
                SENSITIVITY_SUCCESS,
                SENSITIVITY_NONFINITE,
            ),
        ),
    )
    return SensitivityActionResult(
        action,
        valid=valid,
        status=status,
        operator_id="fisher_information_pullback",
        method_id=method_id,
        approximation="exact_exponential_family",
        regularization=penalty,
        num_samples=None,
    )


def gauss_newton_action(
    residual_fn: Callable[[PyTree[Array]], PyTree[Array]],
    parameters: PyTree[Array],
    vector: PyTree[Array],
    /,
    *,
    regularization: float = 0.0,
    method_id: str = "jax_jvp_vjp",
) -> SensitivityActionResult:
    """Apply ``JᵀJ`` by one JVP and one VJP, without a dense Jacobian."""
    if not callable(residual_fn):
        raise TypeError("residual_fn must be callable.")
    penalty = float(regularization)
    if not isfinite(penalty) or penalty < 0.0:
        raise ValueError("regularization must be finite and non-negative.")
    residual, linearized = jax.linearize(residual_fn, parameters)
    tangent_residual = linearized(vector)
    pullback = jax.linear_transpose(linearized, parameters)
    action = pullback(tangent_residual)[0]
    action = jax.tree_util.tree_map(
        lambda value, direction: value + penalty * direction,
        action,
        vector,
    )
    finite = (
        _tree_all_finite(residual)
        & _tree_all_finite(tangent_residual)
        & _tree_all_finite(action)
    )
    status = jnp.where(finite, SENSITIVITY_SUCCESS, SENSITIVITY_NONFINITE)
    return SensitivityActionResult(
        action,
        valid=finite,
        status=status,
        operator_id="gauss_newton",
        method_id=method_id,
        approximation="matrix_free_local_linearization",
        regularization=penalty,
        num_samples=None,
    )


def empirical_observability_directions(
    output_fn: Callable[[Array], ArrayLike],
    state: ArrayLike,
    /,
    *,
    rank: int,
    regularization: float = 0.0,
    max_dimension: int = 256,
) -> EmpiricalDirectionsResult:
    """Return dominant directions of the local empirical observability Gramian."""
    if not callable(output_fn):
        raise TypeError("output_fn must be callable.")
    center = jnp.asarray(state)
    shape = tuple(center.shape)
    dimension = int(center.size)
    retained = _validate_direction_request(dimension, rank, max_dimension)
    penalty = _validate_regularization(regularization)
    output, pushforward = jax.linearize(
        lambda value: jnp.asarray(output_fn(value)), center
    )
    pullback = jax.linear_transpose(pushforward, center)

    def action(flat_vector):
        vector = flat_vector.reshape(shape)
        value = pullback(pushforward(vector))[0]
        return value.reshape(-1) + penalty * flat_vector

    matrix = jax.vmap(action)(jnp.eye(dimension)).T
    return _eigen_directions(
        matrix,
        rank=retained,
        quantity="observability",
        regularization=penalty,
        ambient_shape=shape,
        additional_valid=jnp.all(jnp.isfinite(output)),
    )


def empirical_controllability_directions(
    response_fn: Callable[[Array], ArrayLike],
    inputs: ArrayLike,
    /,
    *,
    rank: int,
    regularization: float = 0.0,
    max_dimension: int = 256,
) -> EmpiricalDirectionsResult:
    """Return dominant output directions of the local controllability Gramian."""
    control = jnp.asarray(inputs)
    response, pushforward = jax.linearize(
        lambda value: jnp.asarray(response_fn(value)), control
    )
    response_shape = tuple(response.shape)
    dimension = int(response.size)
    retained = _validate_direction_request(dimension, rank, max_dimension)
    penalty = _validate_regularization(regularization)
    pullback = jax.linear_transpose(pushforward, control)

    def action(flat_vector):
        vector = flat_vector.reshape(response_shape)
        input_covector = pullback(vector)[0]
        value = pushforward(input_covector)
        return value.reshape(-1) + penalty * flat_vector

    matrix = jax.vmap(action)(jnp.eye(dimension)).T
    return _eigen_directions(
        matrix,
        rank=retained,
        quantity="controllability",
        regularization=penalty,
        ambient_shape=response_shape,
        additional_valid=jnp.all(jnp.isfinite(response)),
    )


def experiment_design_objective(
    information: ArrayLike | Callable[[Array], ArrayLike],
    /,
    *,
    criterion: Literal["d_optimal", "a_optimal", "e_optimal", "mutual_information"],
    dimension: int | None = None,
    regularization: float = 0.0,
    noise_variance: float = 1.0,
    max_dimension: int = 256,
) -> ExperimentDesignResult:
    """Evaluate a guarded A-, D-, E-, or mutual-information design objective."""
    penalty = _validate_regularization(regularization)
    if callable(information):
        action = cast(Callable[[Array], ArrayLike], information)
        if dimension is None:
            raise ValueError(
                "dimension is required for a matrix-free information action."
            )
        size = int(dimension)
        if size <= 0 or size > int(max_dimension):
            raise ValueError(
                "dimension must be positive and no larger than max_dimension."
            )
        basis = jnp.eye(size)
        matrix = jax.vmap(lambda vector: jnp.asarray(action(vector)))(basis).T
        method_id = "matrix_free_actions_materialized"
        approximation = "exact_guarded_materialization"
    else:
        matrix = jnp.asarray(information)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise ValueError("information must be a non-empty square matrix.")
        size = int(matrix.shape[0])
        if dimension is not None and int(dimension) != size:
            raise ValueError("dimension does not match the information matrix.")
        if size > int(max_dimension):
            raise ValueError("Dense information exceeds max_dimension.")
        method_id = "dense_information"
        approximation = "exact_dense"
    effective = matrix + penalty * jnp.eye(size, dtype=matrix.dtype)
    eigenvalues = jnp.linalg.eigvalsh(effective)
    tolerance = (
        64.0
        * jnp.finfo(effective.dtype).eps
        * jnp.maximum(1.0, jnp.linalg.norm(effective))
    )
    symmetric = jnp.linalg.norm(effective - effective.T) <= tolerance
    positive = jnp.all(eigenvalues > 0.0)
    positive_semidefinite = jnp.all(eigenvalues >= -tolerance)
    finite = jnp.all(jnp.isfinite(effective)) & jnp.all(jnp.isfinite(eigenvalues))
    base_valid = finite & symmetric & positive_semidefinite
    if criterion == "d_optimal":
        _, log_determinant = jnp.linalg.slogdet(effective)
        raw_value = log_determinant
        criterion_valid = base_valid & positive
    elif criterion == "a_optimal":
        inverse_result = inverse(
            effective,
            FactorizationPolicy("cholesky"),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "asserted",
                    "positive_definite": "asserted",
                },
            ),
        )
        raw_value = -jnp.trace(inverse_result.value)
        criterion_valid = base_valid & positive & inverse_result.successful
    elif criterion == "e_optimal":
        raw_value = eigenvalues[0]
        criterion_valid = base_valid
    elif criterion == "mutual_information":
        variance = float(noise_variance)
        if not isfinite(variance) or variance <= 0.0:
            raise ValueError("noise_variance must be finite and positive.")
        _, log_determinant = jnp.linalg.slogdet(
            jnp.eye(size, dtype=matrix.dtype) + effective / variance
        )
        raw_value = 0.5 * log_determinant
        criterion_valid = base_valid
    else:
        raise ValueError(
            "criterion must be 'd_optimal', 'a_optimal', 'e_optimal', "
            "or 'mutual_information'."
        )
    reported_valid = criterion_valid & jnp.isfinite(raw_value)
    value = jnp.where(reported_valid, raw_value, jnp.nan)
    status = jnp.where(
        reported_valid,
        SENSITIVITY_SUCCESS,
        SENSITIVITY_INVALID_INFORMATION,
    )
    return ExperimentDesignResult(
        value=value,
        eigenvalues=eigenvalues,
        valid=reported_valid,
        status=status,
        criterion=criterion,
        method_id=method_id,
        approximation=approximation,
        regularization=penalty,
    )


def _tree_all_finite(tree: PyTree[Array], /) -> Array:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)))


def _validate_regularization(value: float, /) -> float:
    penalty = float(value)
    if not isfinite(penalty) or penalty < 0.0:
        raise ValueError("regularization must be finite and non-negative.")
    return penalty


def _validate_direction_request(
    dimension: int,
    rank: int,
    max_dimension: int,
    /,
) -> int:
    retained = int(rank)
    if dimension <= 0:
        raise ValueError("The empirical direction space must be non-empty.")
    if dimension > int(max_dimension):
        raise ValueError("Empirical direction materialization exceeds max_dimension.")
    if retained <= 0 or retained > dimension:
        raise ValueError("rank must lie between one and the ambient dimension.")
    return retained


def _eigen_directions(
    matrix: Array,
    /,
    *,
    rank: int,
    quantity: str,
    regularization: float,
    ambient_shape: tuple[int, ...],
    additional_valid: Array,
) -> EmpiricalDirectionsResult:
    eigenvalues, eigenvectors = jnp.linalg.eigh(matrix)
    strengths = eigenvalues[-rank:][::-1]
    directions = eigenvectors[:, -rank:][:, ::-1]
    tolerance = (
        64.0 * jnp.finfo(matrix.dtype).eps * jnp.maximum(1.0, jnp.linalg.norm(matrix))
    )
    valid = (
        additional_valid
        & jnp.all(jnp.isfinite(matrix))
        & jnp.all(jnp.isfinite(strengths))
        & (jnp.linalg.norm(matrix - matrix.T) <= tolerance)
        & jnp.all(strengths >= -tolerance)
    )
    status = jnp.where(valid, SENSITIVITY_SUCCESS, SENSITIVITY_INVALID_INFORMATION)
    return EmpiricalDirectionsResult(
        directions=directions,
        strengths=strengths,
        valid=valid,
        status=status,
        quantity=quantity,
        regularization=regularization,
        ambient_shape=ambient_shape,
    )


__all__ = [
    "EmpiricalDirectionsResult",
    "ExperimentDesignResult",
    "ResamplingScoreResult",
    "SENSITIVITY_INVALID_INFORMATION",
    "SENSITIVITY_NONFINITE",
    "SENSITIVITY_SUCCESS",
    "SensitivityActionResult",
    "SensitivityGradientResult",
    "empirical_controllability_directions",
    "empirical_observability_directions",
    "experiment_design_objective",
    "exponential_family_fisher_action",
    "exponential_family_parameter_fisher_action",
    "fisher_information_action",
    "fixed_noise_pathwise_gradient",
    "gauss_newton_action",
    "likelihood_ratio_gradient",
    "resampling_score_gradient",
    "SobolResult",
    "sobol_indices",
]
