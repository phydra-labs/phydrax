#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared Gegenbauer quadrature and coefficient-space operators."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.special import roots_gegenbauer

from phydrax import ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._orthogonal import OrthogonalRuleData


GegenbauerNormalization: TypeAlias = Literal["standard", "monic", "orthonormal"]
_DEFAULT_CONSTRUCTION_BYTES = 512 * 1024**2


def _nonnegative_integer(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _positive_integer(value: int, name: str, /) -> int:
    result = _nonnegative_integer(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive.")
    return result


def _static_alpha(value: float, name: str, /) -> float:
    array = np.asarray(value)
    if array.ndim != 0 or np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must be a real scalar.")
    alpha = float(array)
    if not math.isfinite(alpha) or alpha <= -0.5:
        raise ValueError(f"{name} must be finite and greater than -1/2.")
    return alpha


def _floating_dtype(dtype, /) -> jnp.dtype:
    dtype_ = jnp.dtype(dtype)
    if not jnp.issubdtype(dtype_, jnp.floating):
        raise TypeError("Gegenbauer polynomial data require a real floating dtype.")
    if dtype_ == jnp.float16 or dtype_ == jnp.bfloat16:
        return jnp.dtype(jnp.float32)
    return dtype_


def _monic_recurrence_coefficient(index: int, alpha: float, /) -> float:
    if index == 1:
        return 1.0 / (2.0 * (alpha + 1.0))
    return (
        index
        * (index + 2.0 * alpha - 1.0)
        / (4.0 * (index + alpha) * (index + alpha - 1.0))
    )


def _gegenbauer_standard_scales(alpha: float, degree: int, /, *, dtype=float) -> Array:
    """Return standard-mode leading coefficients relative to monic modes."""
    alpha_ = _static_alpha(alpha, "alpha")
    degree_ = _nonnegative_integer(degree, "degree")
    dtype_ = _floating_dtype(dtype)
    scales = np.ones((degree_ + 1,), dtype=float)
    for index in range(1, degree_ + 1):
        scales[index] = scales[index - 1] * 2.0 * (alpha_ + index - 1.0) / index
    return jnp.asarray(scales, dtype=dtype_)


def _gegenbauer_monic_scales(alpha: float, degree: int, /, *, dtype=float) -> Array:
    """Return the unit scales defining the monic Gegenbauer family."""
    _static_alpha(alpha, "alpha")
    degree_ = _nonnegative_integer(degree, "degree")
    return jnp.ones((degree_ + 1,), dtype=_floating_dtype(dtype))


def _gegenbauer_orthonormal_scales(alpha: float, degree: int, /, *, dtype=float) -> Array:
    """Return positive orthonormal-mode leading coefficients over monic modes."""
    alpha_ = _static_alpha(alpha, "alpha")
    degree_ = _nonnegative_integer(degree, "degree")
    dtype_ = _floating_dtype(dtype)
    log_norm = (
        0.5 * math.log(math.pi) + math.lgamma(alpha_ + 0.5) - math.lgamma(alpha_ + 1.0)
    )
    scales = [math.exp(-0.5 * log_norm)]
    for index in range(1, degree_ + 1):
        log_norm += math.log(_monic_recurrence_coefficient(index, alpha_))
        scales.append(math.exp(-0.5 * log_norm))
    return jnp.asarray(scales, dtype=dtype_)


def _normalization_scales(
    normalization: GegenbauerNormalization,
    alpha: float,
    degree: int,
    dtype,
    /,
) -> np.ndarray:
    functions = {
        "standard": _gegenbauer_standard_scales,
        "monic": _gegenbauer_monic_scales,
        "orthonormal": _gegenbauer_orthonormal_scales,
    }
    if normalization not in functions:
        raise ValueError(
            "Gegenbauer normalization must be 'standard', 'monic', or 'orthonormal'."
        )
    return np.asarray(functions[normalization](alpha, degree, dtype=dtype))


def _monic_coefficient_matrix(alpha: float, degree: int, dtype, /) -> np.ndarray:
    count = degree + 1
    coefficients = np.zeros((count, count), dtype=dtype)
    coefficients[0, 0] = 1.0
    if degree == 0:
        return coefficients
    coefficients[1, 1] = 1.0
    for index in range(1, degree):
        coefficients[1 : index + 2, index + 1] = coefficients[: index + 1, index]
        coefficients[:index, index + 1] -= (
            _monic_recurrence_coefficient(index, alpha) * coefficients[:index, index - 1]
        )
    return coefficients


def gauss_gegenbauer_rule_data(
    num_nodes: int, alpha: float, /, *, dtype=float
) -> OrthogonalRuleData:
    """Return Gauss quadrature for ``(1 - x**2)**(alpha - 1/2)``."""
    count = _positive_integer(num_nodes, "num_nodes")
    alpha_ = _static_alpha(alpha, "alpha")
    dtype_ = _floating_dtype(dtype)
    nodes, weights, mass = roots_gegenbauer(count, alpha_, mu=True)
    nodes_ = np.asarray(nodes, dtype=np.dtype(dtype_))
    weights_ = np.asarray(weights, dtype=np.dtype(dtype_))
    return OrthogonalRuleData(
        nodes_,
        weights_,
        exact_degree=2 * count - 1,
        family="gegenbauer",
        node_rule="gauss",
        reference_domain="minus-one-one",
        basis_measure="one-minus-x-squared-power-alpha-minus-half",
        integration_measure="gegenbauer-weight",
        measure_mass=float(mass),
        endpoint_policy="none",
        backend="scipy",
    )


def gegenbauer_differentiation_matrix(
    alpha: float,
    degree: int,
    order: int = 1,
    /,
    *,
    dtype=float,
) -> Array:
    """Map standard ``alpha`` coefficients to standard ``alpha + order``.

    Both coefficient vectors retain capacity ``degree + 1``. The returned
    matrix acts on a leading source-mode axis, and rows are target modes.
    """
    alpha_ = _static_alpha(alpha, "alpha")
    degree_ = _nonnegative_integer(degree, "degree")
    order_ = _nonnegative_integer(order, "order")
    dtype_ = _floating_dtype(dtype)
    count = degree_ + 1
    matrix = np.zeros((count, count), dtype=np.dtype(dtype_))
    if order_ == 0:
        np.fill_diagonal(matrix, 1.0)
    elif order_ <= degree_:
        factor = 2.0**order_
        for offset in range(order_):
            factor *= alpha_ + offset
        columns = np.arange(order_, count)
        matrix[columns - order_, columns] = factor
    return jnp.asarray(matrix)


class GegenbauerConnectionData(StrictModule, NonTrainableState):
    """Prepared coefficient connection from source ``alpha`` to target ``beta``."""

    matrix: Array
    source_alpha: float = eqx.field(static=True)
    target_alpha: float = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    construction_bytes: int = eqx.field(static=True)
    data_id: str = eqx.field(static=True)

    def apply(self, coefficients: ArrayLike, /) -> Array:
        """Apply the connection to a leading source-mode coefficient axis."""
        values = jnp.asarray(coefficients)
        if values.ndim < 1 or values.shape[0] != self.degree + 1:
            raise ValueError(
                "Gegenbauer coefficients need a leading axis of size degree + 1."
            )
        dtype = jnp.result_type(values, self.matrix)
        return ein.contract(
            "ij,j...->i...",
            jnp.asarray(self.matrix, dtype=dtype),
            jnp.asarray(values, dtype=dtype),
        )


def gegenbauer_connection_data(
    alpha: float,
    beta: float,
    degree: int,
    /,
    *,
    normalization: GegenbauerNormalization = "standard",
    dtype=float,
    maximum_construction_bytes: int = _DEFAULT_CONSTRUCTION_BYTES,
) -> GegenbauerConnectionData:
    """Construct the fixed-capacity ``alpha``-to-``beta`` coefficient map."""
    source_alpha = _static_alpha(alpha, "alpha")
    target_alpha = _static_alpha(beta, "beta")
    degree_ = _nonnegative_integer(degree, "degree")
    dtype_ = _floating_dtype(dtype)
    maximum_bytes = _positive_integer(
        maximum_construction_bytes, "maximum_construction_bytes"
    )
    if normalization not in ("standard", "monic", "orthonormal"):
        raise ValueError(
            "Gegenbauer normalization must be 'standard', 'monic', or 'orthonormal'."
        )
    if normalization == "standard" and target_alpha == 0.0 and degree_ > 0:
        raise ValueError("The standard Gegenbauer target basis is degenerate at beta=0.")

    count = degree_ + 1
    itemsize = np.dtype(dtype_).itemsize
    construction_bytes = (3 * count * count + 2 * count) * itemsize
    if construction_bytes > maximum_bytes:
        raise ValueError(
            "Gegenbauer connection construction exceeds maximum_construction_bytes."
        )

    source_monic = _monic_coefficient_matrix(source_alpha, degree_, np.dtype(dtype_))
    target_monic = _monic_coefficient_matrix(target_alpha, degree_, np.dtype(dtype_))
    source_scales = _normalization_scales(
        normalization,
        source_alpha,
        degree_,
        dtype_,
    )
    target_scales = _normalization_scales(
        normalization,
        target_alpha,
        degree_,
        dtype_,
    )
    source_basis = source_monic * source_scales[None, :]
    target_basis = target_monic * target_scales[None, :]
    matrix_host = np.linalg.solve(target_basis, source_basis)
    matrix_host = np.asarray(matrix_host, dtype=np.dtype(dtype_))
    rows, columns = np.indices(matrix_host.shape)
    matrix_host[(rows > columns) | ((columns - rows) % 2 != 0)] = 0.0

    data_id = canonical_fingerprint(
        {
            "kind": "gegenbauer-connection-data",
            "source_alpha": source_alpha,
            "target_alpha": target_alpha,
            "degree": degree_,
            "normalization": normalization,
            "construction_bytes": construction_bytes,
            "data": array_tree_fingerprint(matrix_host),
        }
    )
    return GegenbauerConnectionData(
        matrix=jnp.asarray(matrix_host),
        source_alpha=source_alpha,
        target_alpha=target_alpha,
        degree=degree_,
        normalization=normalization,
        construction_bytes=construction_bytes,
        data_id=data_id,
    )


__all__ = [
    "GegenbauerConnectionData",
    "gauss_gegenbauer_rule_data",
    "gegenbauer_connection_data",
    "gegenbauer_differentiation_matrix",
]
