#
#  Copyright 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._base import _AbstractScaler, _AbstractScalerSpecifier
from ._utils import _EPSILON


AxisLike = int | Sequence[int] | None
_CanonicalAxis = int | tuple[int, ...] | None


def _canonical_axis(axis: AxisLike, /) -> _CanonicalAxis:
    if axis is None or isinstance(axis, int):
        return axis
    return tuple(int(ax) for ax in axis)


def _keepdims(axis: _CanonicalAxis, /) -> bool:
    return axis is not None


class AffineScaler(_AbstractScaler):
    """Apply an affine transformation to input data.

    The transformation is `alpha * ((x - reference_value) / scale_value) + beta`.
    """

    reference_value: Array
    scale_value: Array
    alpha: Array
    beta: Array

    def __init__(
        self,
        *,
        reference_value: ArrayLike = 0.0,
        scale_value: ArrayLike = 1.0,
        alpha: ArrayLike = 1.0,
        beta: ArrayLike = 0.0,
    ):
        """Construct an affine scaler.

        **Arguments:**

        - `reference_value`: Value subtracted from input data.
        - `scale_value`: Value dividing centered input data.
        - `alpha`: Multiplicative scale applied after normalization.
        - `beta`: Additive offset applied after scaling.
        """
        self.reference_value = jnp.asarray(reference_value, dtype=float)
        self.scale_value = jnp.asarray(scale_value, dtype=float)
        self.alpha = jnp.asarray(alpha, dtype=float)
        self.beta = jnp.asarray(beta, dtype=float)

    def transform(self, x: ArrayLike) -> Array:
        x_arr = jnp.asarray(x, dtype=float)
        return (
            self.alpha * ((x_arr - self.reference_value) / self.scale_value) + self.beta
        )

    def inverse_transform(self, x: ArrayLike) -> Array:
        x_arr = jnp.asarray(x, dtype=float)
        return (
            self.scale_value * ((x_arr - self.beta) / self.alpha) + self.reference_value
        )


class MinMaxScaler(_AbstractScalerSpecifier):
    """Scale data into a target interval."""

    scaler: AffineScaler

    def __init__(
        self,
        x: ArrayLike,
        /,
        *,
        min: ArrayLike = 0.0,
        max: ArrayLike = 1.0,
        axis: AxisLike = None,
    ):
        """Construct a min-max scaler from reference data.

        **Arguments:**

        - `x`: Reference data used to infer the input range.
        - `min`: Lower bound of the transformed range.
        - `max`: Upper bound of the transformed range.
        - `axis`: Axis or axes over which statistics are computed.
        """
        x_arr = jnp.asarray(x, dtype=float)
        axis_c = _canonical_axis(axis)
        keepdims = _keepdims(axis_c)

        x_min = jnp.min(x_arr, axis=axis_c, keepdims=keepdims)
        x_max = jnp.max(x_arr, axis=axis_c, keepdims=keepdims)
        x_range = x_max - x_min
        x_range = jnp.where(x_range == 0.0, _EPSILON, x_range)

        scale_min = jnp.asarray(min, dtype=float)
        scale_max = jnp.asarray(max, dtype=float)
        scale_range = scale_max - scale_min
        if bool(jnp.any(scale_range == 0.0)):
            raise ValueError("MinMaxScaler requires `min` and `max` to differ.")

        self.scaler = AffineScaler(
            reference_value=x_min,
            scale_value=x_range,
            alpha=scale_range,
            beta=scale_min,
        )


class MaxAbsScaler(_AbstractScalerSpecifier):
    """Scale data by its maximum absolute value."""

    scaler: AffineScaler

    def __init__(
        self,
        x: ArrayLike,
        /,
        *,
        axis: AxisLike = None,
    ):
        """Construct a max-absolute-value scaler from reference data.

        **Arguments:**

        - `x`: Reference data used to infer the scale.
        - `axis`: Axis or axes over which statistics are computed.
        """
        x_arr = jnp.asarray(x, dtype=float)
        axis_c = _canonical_axis(axis)
        x_max_abs = jnp.max(jnp.abs(x_arr), axis=axis_c, keepdims=_keepdims(axis_c))
        x_max_abs = jnp.where(x_max_abs == 0.0, _EPSILON, x_max_abs)

        self.scaler = AffineScaler(scale_value=x_max_abs)


class StdScaler(_AbstractScalerSpecifier):
    """Standardize data by subtracting its mean and dividing by standard deviation."""

    scaler: AffineScaler

    def __init__(
        self,
        x: ArrayLike,
        /,
        *,
        axis: AxisLike = None,
    ):
        """Construct a standard scaler from reference data.

        **Arguments:**

        - `x`: Reference data used to infer mean and standard deviation.
        - `axis`: Axis or axes over which statistics are computed.
        """
        x_arr = jnp.asarray(x, dtype=float)
        axis_c = _canonical_axis(axis)
        keepdims = _keepdims(axis_c)
        x_mean = jnp.mean(x_arr, axis=axis_c, keepdims=keepdims)
        x_std = jnp.std(x_arr, axis=axis_c, keepdims=keepdims)
        x_std = jnp.where(x_std == 0.0, _EPSILON, x_std)

        self.scaler = AffineScaler(reference_value=x_mean, scale_value=x_std)


class NormScaler(_AbstractScalerSpecifier):
    """Scale data by its vector or matrix norm."""

    scaler: AffineScaler

    def __init__(
        self,
        x: ArrayLike,
        /,
        *,
        ord: int | float = 2,
        axis: AxisLike = None,
    ):
        """Construct a norm scaler from reference data.

        **Arguments:**

        - `x`: Reference data used to infer the norm.
        - `ord`: Norm order forwarded to `jax.numpy.linalg.norm`.
        - `axis`: Axis or axes over which the norm is computed.
        """
        x_arr = jnp.asarray(x, dtype=float)
        axis_c = _canonical_axis(axis)
        norm = jnp.linalg.norm(x_arr, ord=ord, axis=axis_c, keepdims=_keepdims(axis_c))
        norm = jnp.where(norm == 0.0, _EPSILON, norm)

        self.scaler = AffineScaler(scale_value=norm)
