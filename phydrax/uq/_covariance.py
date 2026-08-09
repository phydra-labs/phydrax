#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, ArrayLike, PyTree

from .._strict import StrictModule


CovarianceRepresentation = Literal["diagonal", "dense", "factor", "operator"]
def _validated_array(value: Array, predicate: Array, message: str, /) -> Array:
    if isinstance(predicate, jax_core.Tracer):
        return eqx.error_if(value, predicate, message)
    if bool(predicate):
        raise ValueError(message)
    return value




class AbstractCovariance(StrictModule):
    """Explicit representation of a covariance on an array PyTree."""


class DiagonalCovariance(AbstractCovariance):
    """Independent variances with the same PyTree structure as an uncertain value."""

    variance: PyTree[Array]

    def __init__(self, variance: PyTree[ArrayLike], /):
        arrays = jax.tree_util.tree_map(jnp.asarray, variance)
        leaves = jax.tree_util.tree_leaves(arrays)
        if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
            raise TypeError("Diagonal covariance must contain inexact array leaves.")
        validated_leaves = []
        for leaf in leaves:
            if jnp.issubdtype(leaf.dtype, jnp.complexfloating):
                raise TypeError("Diagonal covariance variances must be real-valued.")
            validated_leaves.append(
                _validated_array(
                    leaf,
                    jnp.any(~jnp.isfinite(leaf)) | jnp.any(leaf < 0.0),
                    "Diagonal covariance variances must be finite and nonnegative.",
                )
            )
        self.variance = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(arrays),
            validated_leaves,
        )


class DenseCovariance(AbstractCovariance):
    """Dense Hermitian positive-semidefinite covariance in flattened PyTree order."""

    matrix: Array

    def __init__(self, matrix: ArrayLike, /):
        value = jnp.asarray(matrix)
        if not eqx.is_inexact_array(value):
            raise TypeError("Dense covariance must be an inexact array.")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
            raise ValueError("Dense covariance must be a non-empty square matrix.")
        value = _validated_array(
            value,
            jnp.any(~jnp.isfinite(value)),
            "Dense covariance must be finite.",
        )
        tolerance = _matrix_tolerance(value)
        hermitian_error = jnp.max(jnp.abs(value - jnp.conj(value.T)))
        value = _validated_array(
            value,
            hermitian_error > tolerance,
            "Dense covariance must be Hermitian within tolerance.",
        )
        hermitian = 0.5 * (value + jnp.conj(value.T))
        eigenvalues = jnp.linalg.eigvalsh(hermitian)
        hermitian = _validated_array(
            hermitian,
            jnp.min(eigenvalues) < -tolerance,
            "Dense covariance must be positive semidefinite.",
        )
        self.matrix = hermitian


class FactorCovariance(AbstractCovariance):
    """Covariance ``B Bᴴ`` stored as leading-rank PyTree factor directions."""

    factors: PyTree[Array]
    rank: int = eqx.field(static=True)

    def __init__(self, factors: PyTree[ArrayLike], /):
        arrays = jax.tree_util.tree_map(jnp.asarray, factors)
        leaves = jax.tree_util.tree_leaves(arrays)
        if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
            raise TypeError("Covariance factors must contain inexact array leaves.")
        if any(leaf.ndim == 0 for leaf in leaves):
            raise ValueError("Every covariance-factor leaf needs a leading rank axis.")
        rank = int(leaves[0].shape[0])
        if rank <= 0 or any(int(leaf.shape[0]) != rank for leaf in leaves):
            raise ValueError(
                "Covariance-factor leaves must share one positive leading rank axis."
            )
        validated_leaves = [
            _validated_array(
                leaf,
                jnp.any(~jnp.isfinite(leaf)),
                "Covariance factors must be finite.",
            )
            for leaf in leaves
        ]
        self.factors = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(arrays),
            validated_leaves,
        )
        self.rank = rank


class CovarianceOperator(AbstractCovariance):
    """Caller-supplied self-adjoint positive-semidefinite covariance action."""

    matvec_fn: Callable[[PyTree[Array]], PyTree[Array]] = eqx.field(static=True)

    def __init__(self, matvec: Callable[[PyTree[Array]], PyTree[Array]], /):
        if not callable(matvec):
            raise TypeError("CovarianceOperator matvec must be callable.")
        self.matvec_fn = matvec

    def __call__(self, vector: PyTree[Array], /) -> PyTree[Array]:
        return self.matvec_fn(vector)


def covariance_representation(
    covariance: AbstractCovariance,
    /,
) -> CovarianceRepresentation:
    """Return the explicit storage representation of one covariance."""
    if isinstance(covariance, DiagonalCovariance):
        return "diagonal"
    if isinstance(covariance, DenseCovariance):
        return "dense"
    if isinstance(covariance, FactorCovariance):
        return "factor"
    if isinstance(covariance, CovarianceOperator):
        return "operator"
    raise TypeError("covariance must implement AbstractCovariance.")


def _validate_covariance_template(
    covariance: AbstractCovariance,
    template: PyTree[Array],
    /,
) -> tuple[int, Any]:
    flat_template, unravel = ravel_pytree(template)
    dimension = int(flat_template.size)
    if dimension <= 0:
        raise ValueError("Covariance templates must contain at least one scalar.")
    template_structure = jax.tree_util.tree_structure(template)
    template_leaves = jax.tree_util.tree_leaves(template)

    if isinstance(covariance, DiagonalCovariance):
        _validate_tree_shapes(
            covariance.variance,
            template_structure,
            tuple(leaf.shape for leaf in template_leaves),
            owner="Diagonal covariance",
        )
    elif isinstance(covariance, DenseCovariance):
        if covariance.matrix.shape != (dimension, dimension):
            raise ValueError(
                "Dense covariance shape must match the flattened uncertain value; "
                f"expected {(dimension, dimension)}, got {covariance.matrix.shape}."
            )
        if (
            jnp.issubdtype(covariance.matrix.dtype, jnp.complexfloating)
            and not jnp.issubdtype(flat_template.dtype, jnp.complexfloating)
        ):
            raise TypeError(
                "Complex dense covariance requires a complex uncertain value."
            )
    elif isinstance(covariance, FactorCovariance):
        factor_shapes = tuple((covariance.rank, *leaf.shape) for leaf in template_leaves)
        _validate_tree_shapes(
            covariance.factors,
            template_structure,
            factor_shapes,
            owner="Factor covariance",
        )
        for factor, template_leaf in zip(
            jax.tree_util.tree_leaves(covariance.factors),
            template_leaves,
            strict=True,
        ):
            if (
                jnp.issubdtype(factor.dtype, jnp.complexfloating)
                and not jnp.issubdtype(template_leaf.dtype, jnp.complexfloating)
            ):
                raise TypeError(
                    "Complex covariance factors require complex uncertain values."
                )
    elif isinstance(covariance, CovarianceOperator):
        probe = jax.tree_util.tree_map(jnp.ones_like, template)
        output = covariance(probe)
        _validate_tree_shapes(
            output,
            template_structure,
            tuple(leaf.shape for leaf in template_leaves),
            owner="Covariance operator output",
        )
        if any(
            bool(jnp.any(~jnp.isfinite(jnp.asarray(leaf))))
            for leaf in jax.tree_util.tree_leaves(output)
        ):
            raise FloatingPointError("Covariance operator probe output must be finite.")
    else:
        raise TypeError("covariance must implement AbstractCovariance.")
    return dimension, unravel


def _apply_covariance(
    covariance: AbstractCovariance,
    vector: PyTree[Array],
    /,
    *,
    unravel: Any,
) -> PyTree[Array]:
    if isinstance(covariance, DiagonalCovariance):
        return jax.tree_util.tree_map(
            lambda variance, value: variance * value,
            covariance.variance,
            vector,
        )
    if isinstance(covariance, DenseCovariance):
        flat_vector, _ = ravel_pytree(vector)
        return unravel(covariance.matrix @ flat_vector)
    if isinstance(covariance, FactorCovariance):
        vector_leaves = jax.tree_util.tree_leaves(vector)
        factor_leaves = jax.tree_util.tree_leaves(covariance.factors)
        coefficients = jnp.zeros(
            (covariance.rank,),
            dtype=jnp.result_type(
                *(leaf.dtype for leaf in (*vector_leaves, *factor_leaves))
            ),
        )
        for factor, value in zip(factor_leaves, vector_leaves, strict=True):
            axes = tuple(range(1, factor.ndim))
            coefficients = coefficients + jnp.sum(
                jnp.conj(factor) * value[None, ...],
                axis=axes,
            )
        return jax.tree_util.tree_map(
            lambda factor: jnp.tensordot(coefficients, factor, axes=((0,), (0,))),
            covariance.factors,
        )
    if isinstance(covariance, CovarianceOperator):
        return covariance(vector)
    raise TypeError("covariance must implement AbstractCovariance.")


def _dense_factor_directions(
    covariance: DenseCovariance,
    /,
    *,
    unravel: Any,
) -> PyTree[Array]:
    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance.matrix)
    clipped = jnp.maximum(eigenvalues, jnp.zeros((), dtype=eigenvalues.dtype))
    flat_factors = jnp.sqrt(clipped)[:, None] * jnp.swapaxes(eigenvectors, 0, 1)
    return jax.vmap(unravel)(flat_factors)


def _validate_tree_shapes(
    value: PyTree[Any],
    expected_structure: Any,
    expected_shapes: tuple[tuple[int, ...], ...],
    /,
    *,
    owner: str,
) -> None:
    if jax.tree_util.tree_structure(value) != expected_structure:
        raise ValueError(f"{owner} must match the uncertain value PyTree structure.")
    leaves = jax.tree_util.tree_leaves(value)
    observed_shapes = tuple(tuple(int(size) for size in leaf.shape) for leaf in leaves)
    if observed_shapes != expected_shapes:
        raise ValueError(
            f"{owner} leaf shapes must be {expected_shapes}; got {observed_shapes}."
        )


def _matrix_tolerance(matrix: Array, /) -> Array:
    real_dtype = jnp.asarray(jnp.real(matrix)).dtype
    epsilon = jnp.finfo(real_dtype).eps
    scale = jnp.maximum(jnp.max(jnp.abs(matrix)), jnp.ones((), dtype=real_dtype))
    return 100.0 * int(matrix.shape[0]) * epsilon * scale


__all__ = [
    "AbstractCovariance",
    "CovarianceOperator",
    "CovarianceRepresentation",
    "DenseCovariance",
    "DiagonalCovariance",
    "FactorCovariance",
    "covariance_representation",
]
