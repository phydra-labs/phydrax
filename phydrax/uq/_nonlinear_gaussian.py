#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, Key, PyTree

import phydrax.ein as ein

from .._polynomial._orthogonal import standard_normal_hermite_rule_data
from .._strict import StrictModule
from ._gaussian_factor import gaussian_factor_from_covariance, GaussianFactor


NonlinearGaussianStatus = Literal[0, 1, 2, 3]
NONLINEAR_GAUSSIAN_SUCCESS: NonlinearGaussianStatus = 0
NONLINEAR_GAUSSIAN_INPUT_FACTOR_INVALID: NonlinearGaussianStatus = 1
NONLINEAR_GAUSSIAN_NONFINITE: NonlinearGaussianStatus = 2
NONLINEAR_GAUSSIAN_OUTPUT_FACTOR_INVALID: NonlinearGaussianStatus = 3


class NonlinearGaussianTransformResult(StrictModule):
    """Auditable Gaussian moments produced by one nonlinear transform."""

    mean: PyTree[Array]
    factor: GaussianFactor
    cross_covariance: Array
    valid: Array
    status: Array
    method_id: str = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    input_dimension: int = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    method_parameters: tuple[tuple[str, float], ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether the input, evaluations, and output factor are all valid."""
        return self.valid


GaussianExpectationMethod = Literal[
    "cubature", "unscented", "gauss-hermite", "monte-carlo"
]


class GaussianExpectationResult(StrictModule):
    """Auditable expectation of an arbitrary PyTree under one Gaussian law."""

    value: PyTree[Array]
    valid: Array
    status: Array
    method_id: str = eqx.field(static=True)
    point_count: int = eqx.field(static=True)
    input_dimension: int = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    method_parameters: tuple[tuple[str, float], ...] = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid


def _configuration_float(value: float, name: str, /, *, nonnegative: bool) -> float:
    resolved = float(value)
    if not np.isfinite(resolved):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and resolved < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return resolved


def _configuration_int(value: int, name: str, /, *, minimum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer.")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return resolved


def _input_coordinates(
    mean: PyTree[Array],
    factor: GaussianFactor,
    /,
) -> tuple[PyTree[Array], Array, Any]:
    if not isinstance(factor, GaussianFactor):
        raise TypeError("factor must be a GaussianFactor.")
    if factor.factor.ndim != 2:
        raise ValueError(
            "Nonlinear Gaussian transforms accept one unbatched GaussianFactor; "
            "use jax.vmap for batched factors."
        )
    arrays = jax.tree_util.tree_map(jnp.asarray, mean)
    leaves = jax.tree_util.tree_leaves(arrays)
    if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
        raise TypeError("Gaussian means must contain inexact array leaves.")
    flat_mean, unravel = ravel_pytree(arrays)
    if int(flat_mean.size) != factor.event_size:
        raise ValueError(
            "GaussianFactor event size must match the flattened mean; "
            f"expected {flat_mean.size}, got {factor.event_size}."
        )
    return arrays, flat_mean, unravel


def _evaluate_points(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    input_unravel: Any,
    points: Array,
    /,
) -> tuple[Array, Any]:
    first_output = function(input_unravel(points[0]))
    first_flat, output_unravel = ravel_pytree(first_output)
    if not jnp.issubdtype(first_flat.dtype, jnp.inexact):
        raise TypeError("Nonlinear Gaussian outputs must be inexact arrays.")
    if points.shape[0] == 1:
        return first_flat[None, :], output_unravel

    def evaluate(flat_point: Array) -> Array:
        flat_output, _ = ravel_pytree(function(input_unravel(flat_point)))
        return flat_output

    remaining = jax.vmap(evaluate)(points[1:])
    return jnp.concatenate((first_flat[None, :], remaining), axis=0), output_unravel


def _append_regularization_factor(factor: Array, regularization: float, /) -> Array:
    if regularization == 0.0:
        return factor
    event_size = int(factor.shape[0])
    scale = jnp.sqrt(jnp.asarray(regularization, dtype=jnp.real(factor).dtype))
    identity = jnp.eye(event_size, dtype=factor.dtype)
    return jnp.concatenate((factor, scale * identity), axis=1)


def _status(
    input_valid: Array,
    evaluations_finite: Array,
    output_valid: Array,
    /,
) -> tuple[Array, Array]:
    input_valid_ = jnp.asarray(input_valid, dtype=bool)
    evaluations_finite_ = jnp.asarray(evaluations_finite, dtype=bool)
    output_valid_ = jnp.asarray(output_valid, dtype=bool)
    valid = input_valid_ & evaluations_finite_ & output_valid_
    status = jnp.where(
        ~input_valid_,
        NONLINEAR_GAUSSIAN_INPUT_FACTOR_INVALID,
        jnp.where(
            ~evaluations_finite_,
            NONLINEAR_GAUSSIAN_NONFINITE,
            jnp.where(
                ~output_valid_,
                NONLINEAR_GAUSSIAN_OUTPUT_FACTOR_INVALID,
                NONLINEAR_GAUSSIAN_SUCCESS,
            ),
        ),
    )
    return valid, status


def _weighted_transform(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    mean: PyTree[Array],
    factor: GaussianFactor,
    canonical_points: Array,
    mean_weights: Array,
    covariance_weights: Array,
    /,
    *,
    method_id: str,
    regularization: float,
    method_parameters: tuple[tuple[str, float], ...],
    nonnegative_covariance_weights: bool,
    max_dense_dimension: int | None = None,
) -> NonlinearGaussianTransformResult:
    mean_tree, flat_mean, input_unravel = _input_coordinates(mean, factor)
    physical_points = flat_mean[None, :] + canonical_points @ factor.factor.T
    output_points, output_unravel = _evaluate_points(
        function,
        input_unravel,
        physical_points,
    )
    output_mean = ein.contract("p,po->o", mean_weights, output_points)
    output_centered = output_points - output_mean[None, :]
    input_centered = physical_points - flat_mean[None, :]
    cross_covariance = ein.contract(
        "p,pi,po->io",
        covariance_weights,
        input_centered,
        jnp.conj(output_centered),
    )

    if nonnegative_covariance_weights:
        weighted_deviations = output_centered.T * jnp.sqrt(covariance_weights)[None, :]
        output_factor = GaussianFactor(
            _append_regularization_factor(weighted_deviations, regularization),
            regularization=regularization,
            rank_tolerance=0.0,
            factor_id=f"{method_id}-output",
            resolved_method="weighted-sigma-point-factor",
        )
    else:
        if (
            max_dense_dimension is not None
            and int(output_mean.size) > max_dense_dimension
        ):
            raise ValueError(
                "Scaled unscented dense covariance exceeds max_output_dimension; "
                f"got {output_mean.size}, cap {max_dense_dimension}."
            )
        covariance = ein.contract(
            "p,pi,pj->ij",
            covariance_weights,
            output_centered,
            jnp.conj(output_centered),
        )
        covariance = 0.5 * (covariance + jnp.conj(covariance.T))
        output_factor = gaussian_factor_from_covariance(
            covariance,
            regularization=regularization,
            rank_tolerance=0.0,
            hermitian_tolerance=0.0,
            factor_id=f"{method_id}-output",
        )

    evaluations_finite = (
        jnp.all(jnp.isfinite(output_points))
        & jnp.all(jnp.isfinite(output_mean))
        & jnp.all(jnp.isfinite(cross_covariance))
    )
    valid, status = _status(factor.valid, evaluations_finite, output_factor.valid)
    return NonlinearGaussianTransformResult(
        mean=output_unravel(output_mean),
        factor=output_factor,
        cross_covariance=cross_covariance,
        valid=valid,
        status=status,
        method_id=method_id,
        point_count=int(canonical_points.shape[0]),
        regularization=regularization,
        input_dimension=int(flat_mean.size),
        output_dimension=int(output_mean.size),
        method_parameters=method_parameters,
    )


def _spherical_radial_rule(rank: int, dtype: jnp.dtype, /) -> tuple[Array, Array]:
    if rank == 0:
        return jnp.zeros((1, 0), dtype=dtype), jnp.ones((1,), dtype=dtype)
    scale = jnp.sqrt(jnp.asarray(rank, dtype=dtype))
    identity = jnp.eye(rank, dtype=dtype)
    points = jnp.concatenate((scale * identity, -scale * identity), axis=0)
    weights = jnp.full((2 * rank,), 1.0 / (2.0 * rank), dtype=dtype)
    return points, weights


def _scaled_unscented_rule(
    rank: int,
    dtype: jnp.dtype,
    /,
    *,
    alpha: float,
    beta: float,
    kappa: float,
) -> tuple[Array, Array, Array, bool]:
    if rank == 0:
        points = jnp.zeros((1, 0), dtype=dtype)
        weights = jnp.ones((1,), dtype=dtype)
        return points, weights, weights, True
    lambda_ = alpha**2 * (rank + kappa) - rank
    scaling = rank + lambda_
    if not np.isfinite(scaling) or scaling <= 0.0:
        raise ValueError("Scaled unscented parameters require n + lambda > 0.")
    radius = jnp.sqrt(jnp.asarray(scaling, dtype=dtype))
    identity = jnp.eye(rank, dtype=dtype)
    points = jnp.concatenate(
        (jnp.zeros((1, rank), dtype=dtype), radius * identity, -radius * identity),
        axis=0,
    )
    side_weight = 1.0 / (2.0 * scaling)
    central_mean_weight = lambda_ / scaling
    central_covariance_weight = central_mean_weight + (1.0 - alpha**2 + beta)
    mean_weights = jnp.asarray(
        (central_mean_weight, *([side_weight] * (2 * rank))), dtype=dtype
    )
    covariance_weights = jnp.asarray(
        (central_covariance_weight, *([side_weight] * (2 * rank))), dtype=dtype
    )
    return (
        points,
        mean_weights,
        covariance_weights,
        central_covariance_weight >= 0.0,
    )


def _gauss_hermite_rule(
    rank: int,
    dtype: jnp.dtype,
    /,
    *,
    order: int,
    max_dimension: int,
    max_points: int,
) -> tuple[Array, Array]:
    if rank > max_dimension:
        raise ValueError(
            "Gauss-Hermite latent dimension exceeds max_dimension; "
            f"got {rank}, cap {max_dimension}."
        )
    point_count = order**rank
    if point_count > max_points:
        raise ValueError(
            "Gauss-Hermite tensor rule exceeds max_points; "
            f"requires {point_count}, cap {max_points}."
        )
    if rank == 0:
        return jnp.zeros((1, 0), dtype=dtype), jnp.ones((1,), dtype=dtype)
    rule = standard_normal_hermite_rule_data(order, dtype=dtype)
    node_mesh = jnp.meshgrid(*([rule.nodes] * rank), indexing="ij")
    weight_mesh = jnp.meshgrid(*([rule.weights] * rank), indexing="ij")
    points = jnp.stack(tuple(mesh.reshape(-1) for mesh in node_mesh), axis=1)
    weights = jnp.prod(
        jnp.stack(tuple(mesh.reshape(-1) for mesh in weight_mesh), axis=1),
        axis=1,
    )
    return points, weights


def _weighted_expectation(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    mean: PyTree[Array],
    factor: GaussianFactor,
    canonical_points: Array,
    weights: Array,
    /,
    *,
    method_id: str,
    method_parameters: tuple[tuple[str, float], ...],
) -> GaussianExpectationResult:
    _, flat_mean, input_unravel = _input_coordinates(mean, factor)
    physical_points = flat_mean[None, :] + canonical_points @ factor.factor.T
    output_points, output_unravel = _evaluate_points(
        function, input_unravel, physical_points
    )
    value = ein.contract("p,po->o", weights, output_points)
    evaluations_finite = (
        jnp.all(jnp.isfinite(flat_mean))
        & jnp.all(jnp.isfinite(physical_points))
        & jnp.all(jnp.isfinite(output_points))
        & jnp.all(jnp.isfinite(value))
    )
    valid, status = _status(factor.valid, evaluations_finite, jnp.asarray(True))
    return GaussianExpectationResult(
        value=output_unravel(value),
        valid=valid,
        status=status,
        method_id=method_id,
        point_count=int(canonical_points.shape[0]),
        input_dimension=int(flat_mean.size),
        output_dimension=int(value.size),
        method_parameters=method_parameters,
    )


def gaussian_expectation(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    mean: PyTree[Array],
    factor: GaussianFactor,
    /,
    *,
    method: GaussianExpectationMethod = "cubature",
    key: Key[Array, ""] | None = None,
    num_samples: int = 32,
    order: int = 3,
    max_dimension: int = 5,
    max_points: int = 100_000,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> GaussianExpectationResult:
    """Evaluate an arbitrary PyTree expectation without forming output covariance."""
    if method not in ("cubature", "unscented", "gauss-hermite", "monte-carlo"):
        raise ValueError(
            "method must be 'cubature', 'unscented', 'gauss-hermite', or 'monte-carlo'."
        )
    _, flat_mean, _ = _input_coordinates(mean, factor)
    rank = factor.rank
    real_dtype = jnp.real(flat_mean).dtype
    if method == "cubature":
        points, weights = _spherical_radial_rule(rank, real_dtype)
        method_id = "spherical-radial-cubature"
        parameters: tuple[tuple[str, float], ...] = ()
    elif method == "unscented":
        alpha_ = _configuration_float(alpha, "alpha", nonnegative=False)
        beta_ = _configuration_float(beta, "beta", nonnegative=False)
        kappa_ = _configuration_float(kappa, "kappa", nonnegative=False)
        if alpha_ <= 0.0:
            raise ValueError("alpha must be positive.")
        points, weights, _, _ = _scaled_unscented_rule(
            rank,
            real_dtype,
            alpha=alpha_,
            beta=beta_,
            kappa=kappa_,
        )
        method_id = "scaled-unscented"
        parameters = (("alpha", alpha_), ("beta", beta_), ("kappa", kappa_))
    elif method == "gauss-hermite":
        order_ = _configuration_int(order, "order", minimum=1)
        max_dimension_ = _configuration_int(max_dimension, "max_dimension", minimum=1)
        max_points_ = _configuration_int(max_points, "max_points", minimum=1)
        points, weights = _gauss_hermite_rule(
            rank,
            real_dtype,
            order=order_,
            max_dimension=max_dimension_,
            max_points=max_points_,
        )
        method_id = "gauss-hermite"
        parameters = (
            ("order", float(order_)),
            ("max_dimension", float(max_dimension_)),
            ("max_points", float(max_points_)),
        )
    else:
        sample_count = _configuration_int(num_samples, "num_samples", minimum=1)
        if key is None:
            raise ValueError("key is required for method='monte-carlo'.")
        if rank == 0:
            points = jnp.zeros((1, 0), dtype=real_dtype)
            weights = jnp.ones((1,), dtype=real_dtype)
        else:
            points = jr.normal(key, (sample_count, rank), dtype=real_dtype)
            weights = jnp.full((sample_count,), 1.0 / sample_count, dtype=real_dtype)
        method_id = "fixed-sample-monte-carlo"
        parameters = (("num_samples", float(sample_count)),)
    return _weighted_expectation(
        function,
        mean,
        factor,
        points,
        weights,
        method_id=method_id,
        method_parameters=parameters,
    )


def spherical_radial_cubature(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    mean: PyTree[Array],
    factor: GaussianFactor,
    /,
    *,
    regularization: float = 0.0,
) -> NonlinearGaussianTransformResult:
    """Transform Gaussian moments with the third-degree spherical-radial rule."""
    regularization_ = _configuration_float(
        regularization,
        "regularization",
        nonnegative=True,
    )
    _, flat_mean, _ = _input_coordinates(mean, factor)
    canonical_points, weights = _spherical_radial_rule(
        factor.rank, jnp.real(flat_mean).dtype
    )
    return _weighted_transform(
        function,
        mean,
        factor,
        canonical_points,
        weights,
        weights,
        method_id="spherical-radial-cubature",
        regularization=regularization_,
        method_parameters=(),
        nonnegative_covariance_weights=True,
    )


def scaled_unscented_transform(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    mean: PyTree[Array],
    factor: GaussianFactor,
    /,
    *,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    max_output_dimension: int = 256,
    regularization: float = 0.0,
) -> NonlinearGaussianTransformResult:
    """Transform Gaussian moments with the scaled unscented sigma-point rule."""
    alpha_ = _configuration_float(alpha, "alpha", nonnegative=False)
    beta_ = _configuration_float(beta, "beta", nonnegative=False)
    kappa_ = _configuration_float(kappa, "kappa", nonnegative=False)
    max_output_dimension_ = _configuration_int(
        max_output_dimension,
        "max_output_dimension",
        minimum=1,
    )
    regularization_ = _configuration_float(
        regularization,
        "regularization",
        nonnegative=True,
    )
    if alpha_ <= 0.0:
        raise ValueError("alpha must be positive.")

    _, flat_mean, _ = _input_coordinates(mean, factor)
    rank = factor.rank
    real_dtype = jnp.real(flat_mean).dtype
    parameters = (
        ("alpha", alpha_),
        ("beta", beta_),
        ("kappa", kappa_),
        ("max_output_dimension", float(max_output_dimension_)),
    )
    (
        canonical_points,
        mean_weights,
        covariance_weights,
        nonnegative_covariance_weights,
    ) = _scaled_unscented_rule(
        rank,
        real_dtype,
        alpha=alpha_,
        beta=beta_,
        kappa=kappa_,
    )

    return _weighted_transform(
        function,
        mean,
        factor,
        canonical_points,
        mean_weights,
        covariance_weights,
        method_id="scaled-unscented",
        regularization=regularization_,
        method_parameters=parameters,
        nonnegative_covariance_weights=nonnegative_covariance_weights,
        max_dense_dimension=max_output_dimension_,
    )


def gauss_hermite_transform(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    mean: PyTree[Array],
    factor: GaussianFactor,
    /,
    *,
    order: int = 3,
    max_dimension: int = 5,
    max_points: int = 100_000,
    regularization: float = 0.0,
) -> NonlinearGaussianTransformResult:
    """Transform Gaussian moments by a guarded tensor Gauss-Hermite rule."""
    order_ = _configuration_int(order, "order", minimum=1)
    max_dimension_ = _configuration_int(max_dimension, "max_dimension", minimum=1)
    max_points_ = _configuration_int(max_points, "max_points", minimum=1)
    regularization_ = _configuration_float(
        regularization,
        "regularization",
        nonnegative=True,
    )
    _, flat_mean, _ = _input_coordinates(mean, factor)
    rank = factor.rank
    canonical_points, weights = _gauss_hermite_rule(
        rank,
        jnp.real(flat_mean).dtype,
        order=order_,
        max_dimension=max_dimension_,
        max_points=max_points_,
    )

    return _weighted_transform(
        function,
        mean,
        factor,
        canonical_points,
        weights,
        weights,
        method_id="gauss-hermite",
        regularization=regularization_,
        method_parameters=(
            ("order", float(order_)),
            ("max_dimension", float(max_dimension_)),
            ("max_points", float(max_points_)),
        ),
        nonnegative_covariance_weights=True,
    )


def first_order_gaussian_transform(
    function: Callable[[PyTree[Array]], PyTree[Array]],
    mean: PyTree[Array],
    factor: GaussianFactor,
    /,
    *,
    regularization: float = 0.0,
) -> NonlinearGaussianTransformResult:
    """Transform Gaussian moments with matrix-free factor-direction JVP actions."""
    regularization_ = _configuration_float(
        regularization,
        "regularization",
        nonnegative=True,
    )
    mean_tree, flat_mean, input_unravel = _input_coordinates(mean, factor)
    output_mean, pushforward = jax.linearize(function, mean_tree)
    flat_output_mean, _ = ravel_pytree(output_mean)
    if not jnp.issubdtype(flat_output_mean.dtype, jnp.inexact):
        raise TypeError("Nonlinear Gaussian outputs must be inexact arrays.")
    output_dimension = int(flat_output_mean.size)

    if factor.rank == 0:
        output_directions = jnp.zeros(
            (output_dimension, 0),
            dtype=flat_output_mean.dtype,
        )
        cross_covariance = jnp.zeros(
            (int(flat_mean.size), output_dimension),
            dtype=jnp.result_type(flat_mean, flat_output_mean),
        )
    else:

        def push_direction(flat_direction: Array) -> Array:
            pushed = pushforward(input_unravel(flat_direction))
            flat_pushed, _ = ravel_pytree(pushed)
            return flat_pushed

        output_directions = jax.vmap(push_direction, in_axes=1, out_axes=1)(factor.factor)
        cross_covariance = factor.factor @ jnp.conj(output_directions.T)

    output_factor = GaussianFactor(
        _append_regularization_factor(output_directions, regularization_),
        regularization=regularization_,
        rank_tolerance=0.0,
        factor_id="first-order-jvp-vjp-output",
        resolved_method="first-order-jvp-factor",
    )
    evaluations_finite = (
        jnp.all(jnp.isfinite(flat_output_mean))
        & jnp.all(jnp.isfinite(output_directions))
        & jnp.all(jnp.isfinite(cross_covariance))
    )
    valid, status = _status(factor.valid, evaluations_finite, output_factor.valid)
    return NonlinearGaussianTransformResult(
        mean=output_mean,
        factor=output_factor,
        cross_covariance=cross_covariance,
        valid=valid,
        status=status,
        method_id="first-order-jvp-vjp",
        point_count=1,
        regularization=regularization_,
        input_dimension=int(flat_mean.size),
        output_dimension=output_dimension,
        method_parameters=(),
    )
