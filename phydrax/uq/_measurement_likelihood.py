#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import prod
from typing import Any, cast, Literal

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree

from ._posterior_terms import AbstractPosteriorTerm


CovarianceBatching = Literal["shared", "per_case"]
CovarianceValue = ArrayLike | Callable[[PyTree[Any]], ArrayLike]


class LinearizedGaussianMeasurementLikelihood(AbstractPosteriorTerm):
    """Normalized Gaussian errors-in-variables likelihood linearized per case.

    The prediction callable receives physical parameters and one measured input case.
    Input uncertainty is pushed through that local input derivative and added to the
    declared observation covariance before evaluating one joint Gaussian factor.
    """

    inputs: PyTree[Array]
    targets: Array
    input_covariance: Array | None
    observation_covariance: Array | None
    predict_fn: Callable[[PyTree[Any], PyTree[Array]], ArrayLike | cx.Field] = (
        eqx.field(static=True)
    )
    input_covariance_fn: Callable[[PyTree[Any]], ArrayLike] | None = eqx.field(
        static=True
    )
    observation_covariance_fn: Callable[[PyTree[Any]], ArrayLike] | None = eqx.field(
        static=True
    )
    input_covariance_batching: CovarianceBatching = eqx.field(static=True)
    observation_covariance_batching: CovarianceBatching = eqx.field(static=True)
    num_cases: int = eqx.field(static=True)
    input_dimension: int = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    stabilization: float = eqx.field(static=True)
    max_output_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        predict_case: Callable[[PyTree[Any], PyTree[Array]], ArrayLike | cx.Field],
        measured_inputs: PyTree[ArrayLike],
        measured_targets: ArrayLike | cx.Field,
        /,
        *,
        input_covariance: CovarianceValue,
        observation_covariance: CovarianceValue,
        input_covariance_batching: CovarianceBatching = "shared",
        observation_covariance_batching: CovarianceBatching = "shared",
        stabilization: float = 0.0,
        max_output_dimension: int = 256,
        label: str = "measurement_error",
    ):
        if not callable(predict_case):
            raise TypeError("predict_case must be callable.")
        inputs = jax.tree_util.tree_map(jnp.asarray, measured_inputs)
        num_cases = _validate_case_tree(inputs, owner="Measured inputs", finite=True)
        targets = _field_data(measured_targets)
        if targets.ndim == 0 or int(targets.shape[0]) != num_cases:
            raise ValueError(
                "Measured targets must have the same non-empty leading case axis as "
                "measured inputs."
            )
        if jnp.issubdtype(targets.dtype, jnp.complexfloating):
            raise TypeError("Measurement likelihood targets must be real-valued.")
        if bool(jnp.any(~jnp.isfinite(targets))):
            raise ValueError("Measured targets must be finite.")

        one_input = jax.tree_util.tree_map(lambda value: value[0], inputs)
        flat_input, _ = ravel_pytree(one_input)
        input_dimension = int(flat_input.size)
        output_dimension = int(prod(targets.shape[1:])) if targets.ndim > 1 else 1
        maximum = int(max_output_dimension)
        if maximum <= 0:
            raise ValueError("max_output_dimension must be positive.")
        if output_dimension > maximum:
            raise ValueError(
                "Measurement output event exceeds max_output_dimension; "
                f"got {output_dimension} > {maximum}."
            )
        input_batching = _validate_batching(
            input_covariance_batching,
            owner="input_covariance_batching",
        )
        observation_batching = _validate_batching(
            observation_covariance_batching,
            owner="observation_covariance_batching",
        )
        regularization = float(stabilization)
        if not jnp.isfinite(regularization) or regularization < 0.0:
            raise ValueError("stabilization must be finite and nonnegative.")

        input_value, input_fn = _split_covariance(input_covariance, owner="input")
        observation_value, observation_fn = _split_covariance(
            observation_covariance,
            owner="observation",
        )
        if input_value is not None:
            input_value = _validate_fixed_covariance(
                input_value,
                batching=input_batching,
                num_cases=num_cases,
                dimension=input_dimension,
                positive_definite=False,
                stabilization=0.0,
                owner="Input covariance",
            )
        if observation_value is not None:
            observation_value = _validate_fixed_covariance(
                observation_value,
                batching=observation_batching,
                num_cases=num_cases,
                dimension=output_dimension,
                positive_definite=True,
                stabilization=regularization,
                owner="Observation covariance",
            )

        self.predict_fn = predict_case
        self.inputs = inputs
        self.targets = targets
        self.input_covariance = input_value
        self.observation_covariance = observation_value
        self.input_covariance_fn = input_fn
        self.observation_covariance_fn = observation_fn
        self.input_covariance_batching = input_batching
        self.observation_covariance_batching = observation_batching
        self.num_cases = num_cases
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.stabilization = regularization
        self.max_output_dimension = maximum
        self.label = _label(label)

    def per_case_log_prob(self, parameters: PyTree[Any], /) -> Array:
        """Return one normalized joint Gaussian contribution per stored case."""
        return self.log_prob_cases(
            parameters,
            self.inputs,
            self.targets,
            case_indices=jnp.arange(self.num_cases, dtype=jnp.int32),
        )

    def log_prob_cases(
        self,
        parameters: PyTree[Any],
        inputs: PyTree[ArrayLike],
        targets: ArrayLike | cx.Field,
        /,
        *,
        case_indices: ArrayLike | None = None,
    ) -> Array:
        """Evaluate external case batches for deterministic minibatch inference."""
        input_arrays = jax.tree_util.tree_map(jnp.asarray, inputs)
        batch_size = _validate_case_tree(
            input_arrays,
            owner="Measurement batch inputs",
            finite=False,
        )
        target_array = _field_data(targets)
        if target_array.ndim == 0 or int(target_array.shape[0]) != batch_size:
            raise ValueError(
                "Measurement batch targets must share the input leading case axis."
            )
        observed_dimension = (
            int(prod(target_array.shape[1:])) if target_array.ndim > 1 else 1
        )
        if observed_dimension != self.output_dimension:
            raise ValueError(
                "Measurement batch target event dimension changed; "
                f"expected {self.output_dimension}, got {observed_dimension}."
            )
        indices = _case_indices(case_indices, batch_size, self.num_cases)
        input_covariances = self._resolved_covariance(
            parameters,
            fixed=self.input_covariance,
            callback=self.input_covariance_fn,
            batching=self.input_covariance_batching,
            dimension=self.input_dimension,
            case_indices=indices,
            batch_size=batch_size,
            owner="Input covariance",
        )
        observation_covariances = self._resolved_covariance(
            parameters,
            fixed=self.observation_covariance,
            callback=self.observation_covariance_fn,
            batching=self.observation_covariance_batching,
            dimension=self.output_dimension,
            case_indices=indices,
            batch_size=batch_size,
            owner="Observation covariance",
        )
        return jax.vmap(
            lambda input_case, target_case, input_matrix, observation_matrix: (
                self._case_log_prob(
                    parameters,
                    input_case,
                    target_case,
                    input_matrix,
                    observation_matrix,
                )
            )
        )(
            input_arrays,
            target_array,
            input_covariances,
            observation_covariances,
        )

    def _resolved_covariance(
        self,
        parameters: PyTree[Any],
        /,
        *,
        fixed: Array | None,
        callback: Callable[[PyTree[Any]], ArrayLike] | None,
        batching: CovarianceBatching,
        dimension: int,
        case_indices: Array,
        batch_size: int,
        owner: str,
    ) -> Array:
        value = fixed if callback is None else jnp.asarray(callback(parameters))
        if value is None:
            raise RuntimeError(f"{owner} has neither a fixed value nor a callback.")
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError(f"{owner} must be real-valued.")
        if batching == "shared":
            if value.shape != (dimension, dimension):
                raise ValueError(
                    f"{owner} callback must return shape {(dimension, dimension)}; "
                    f"got {value.shape}."
                )
            return jnp.broadcast_to(value, (batch_size, dimension, dimension))
        expected = (self.num_cases, dimension, dimension)
        if value.shape != expected:
            raise ValueError(
                f"{owner} callback must return per-case shape {expected}; "
                f"got {value.shape}."
            )
        return jnp.take(
            value,
            case_indices,
            axis=0,
            mode="fill",
            fill_value=jnp.nan,
        )

    def _case_log_prob(
        self,
        parameters: PyTree[Any],
        input_case: PyTree[Array],
        target_case: Array,
        input_covariance: Array,
        observation_covariance: Array,
        /,
    ) -> Array:
        flat_input, unravel_input = ravel_pytree(input_case)
        target = jnp.ravel(jnp.asarray(target_case))

        def predict_flat(value):
            prediction = _field_data(self.predict_fn(parameters, unravel_input(value)))
            return jnp.ravel(prediction)

        prediction, pushforward = jax.linearize(predict_flat, flat_input)
        if prediction.shape != (self.output_dimension,):
            raise ValueError(
                "predict_case output event dimension changed; "
                f"expected {self.output_dimension}, got {prediction.size}."
            )
        input_hermitian = 0.5 * (input_covariance + input_covariance.T)
        input_eigenvalues, input_eigenvectors = jnp.linalg.eigh(input_hermitian)
        input_factors = (
            jnp.sqrt(jnp.maximum(input_eigenvalues, 0.0))[:, None]
            * input_eigenvectors.T
        )
        propagated_factors = jax.vmap(pushforward)(input_factors)
        pushed_covariance = propagated_factors.T @ propagated_factors
        observation_hermitian = 0.5 * (
            observation_covariance + observation_covariance.T
        )
        effective_covariance = (
            observation_hermitian
            + pushed_covariance
            + self.stabilization * jnp.eye(self.output_dimension)
        )
        effective_eigenvalues = jnp.linalg.eigvalsh(effective_covariance)
        input_tolerance = _covariance_tolerance(input_covariance)
        observation_tolerance = _covariance_tolerance(observation_covariance)
        effective_tolerance = _covariance_tolerance(effective_covariance)
        valid = (
            jnp.all(jnp.isfinite(prediction))
            & jnp.all(jnp.isfinite(target))
            & jnp.all(jnp.isfinite(input_covariance))
            & jnp.all(jnp.isfinite(observation_covariance))
            & (
                jnp.max(jnp.abs(input_covariance - input_covariance.T))
                <= input_tolerance
            )
            & (
                jnp.max(
                    jnp.abs(observation_covariance - observation_covariance.T)
                )
                <= observation_tolerance
            )
            & (jnp.min(input_eigenvalues) >= -input_tolerance)
            & (jnp.min(effective_eigenvalues) > effective_tolerance)
        )

        def finite_log_prob(_):
            cholesky = jnp.linalg.cholesky(effective_covariance)
            residual = target - prediction
            standardized = jsp.linalg.solve_triangular(
                cholesky,
                residual,
                lower=True,
            )
            quadratic = jnp.vdot(standardized, standardized).real
            log_determinant = 2.0 * jnp.sum(jnp.log(jnp.diag(cholesky)))
            normalizer = self.output_dimension * jnp.log(2.0 * jnp.pi)
            return -0.5 * (quadratic + log_determinant + normalizer)

        return jax.lax.cond(
            valid,
            finite_log_prob,
            lambda _: jnp.asarray(-jnp.inf, dtype=prediction.dtype),
            operand=None,
        )


def _split_covariance(
    value: CovarianceValue,
    /,
    *,
    owner: str,
) -> tuple[Array | None, Callable[[PyTree[Any]], ArrayLike] | None]:
    if callable(value):
        return None, cast(Callable[[PyTree[Any]], ArrayLike], value)
    array = jnp.asarray(value)
    if not eqx.is_inexact_array(array):
        raise TypeError(f"{owner}_covariance must be an inexact array or callable.")
    return array, None


def _validate_fixed_covariance(
    value: Array,
    /,
    *,
    batching: CovarianceBatching,
    num_cases: int,
    dimension: int,
    positive_definite: bool,
    stabilization: float,
    owner: str,
) -> Array:
    if jnp.issubdtype(value.dtype, jnp.complexfloating):
        raise TypeError(f"{owner} must be real-valued.")
    expected = (
        (dimension, dimension)
        if batching == "shared"
        else (num_cases, dimension, dimension)
    )
    if value.shape != expected:
        raise ValueError(f"{owner} must have shape {expected}; got {value.shape}.")
    matrices = value[None, ...] if batching == "shared" else value
    if bool(jnp.any(~jnp.isfinite(matrices))):
        raise ValueError(f"{owner} must be finite.")
    tolerances = jax.vmap(_covariance_tolerance)(matrices)
    symmetry_errors = jax.vmap(
        lambda matrix: jnp.max(jnp.abs(matrix - matrix.T))
    )(matrices)
    if bool(jnp.any(symmetry_errors > tolerances)):
        raise ValueError(f"{owner} must be symmetric within tolerance.")
    hermitian = 0.5 * (matrices + jnp.swapaxes(matrices, -1, -2))
    if stabilization:
        hermitian = hermitian + stabilization * jnp.eye(dimension)[None, ...]
    eigenvalues = jax.vmap(jnp.linalg.eigvalsh)(hermitian)
    if positive_definite:
        if bool(jnp.any(jnp.min(eigenvalues, axis=1) <= tolerances)):
            raise ValueError(
                f"{owner} plus explicit stabilization must be positive definite."
            )
    elif bool(jnp.any(jnp.min(eigenvalues, axis=1) < -tolerances)):
        raise ValueError(f"{owner} must be positive semidefinite.")
    symmetric = 0.5 * (value + jnp.swapaxes(value, -1, -2))
    return symmetric


def _validate_case_tree(
    value: PyTree[Array],
    /,
    *,
    owner: str,
    finite: bool,
) -> int:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
        raise TypeError(f"{owner} must be a non-empty PyTree of inexact arrays.")
    if any(leaf.ndim == 0 for leaf in leaves):
        raise ValueError(f"Every {owner.lower()} leaf needs a leading case axis.")
    count = int(leaves[0].shape[0])
    if count <= 0 or any(int(leaf.shape[0]) != count for leaf in leaves):
        raise ValueError(f"Every {owner.lower()} leaf must share one positive case axis.")
    if any(jnp.issubdtype(leaf.dtype, jnp.complexfloating) for leaf in leaves):
        raise TypeError(f"{owner} must be real-valued.")
    if finite and any(bool(jnp.any(~jnp.isfinite(leaf))) for leaf in leaves):
        raise ValueError(f"{owner} must be finite.")
    return count


def _case_indices(
    value: ArrayLike | None,
    batch_size: int,
    num_cases: int,
    /,
) -> Array:
    if value is None:
        if batch_size != num_cases:
            raise ValueError(
                "case_indices are required when evaluating a strict subset of cases."
            )
        return jnp.arange(num_cases, dtype=jnp.int32)
    indices = jnp.asarray(value)
    if indices.shape != (batch_size,) or not jnp.issubdtype(
        indices.dtype,
        jnp.integer,
    ):
        raise ValueError("case_indices must be one integer index per batch case.")
    return indices


def _field_data(value: ArrayLike | cx.Field, /) -> Array:
    return jnp.asarray(value.data if isinstance(value, cx.Field) else value)


def _validate_batching(value: str, /, *, owner: str) -> CovarianceBatching:
    if value not in ("shared", "per_case"):
        raise ValueError(f"{owner} must be 'shared' or 'per_case'.")
    return value  # type: ignore[return-value]


def _label(value: str, /) -> str:
    label = str(value)
    if not label:
        raise ValueError("Measurement likelihood labels must be non-empty.")
    return label


def _covariance_tolerance(matrix: Array, /) -> Array:
    epsilon = jnp.finfo(matrix.dtype).eps
    scale = jnp.maximum(jnp.max(jnp.abs(matrix)), jnp.ones((), dtype=matrix.dtype))
    return 100.0 * int(matrix.shape[-1]) * epsilon * scale


__all__ = ["CovarianceBatching", "LinearizedGaussianMeasurementLikelihood"]
