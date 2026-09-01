#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


ConvolutionQuadratureMethod: TypeAlias = Literal["bdf1", "bdf2"]


def _method_order(method: ConvolutionQuadratureMethod, /) -> int:
    if method == "bdf1":
        return 1
    if method == "bdf2":
        return 2
    raise ValueError("Convolution quadrature method must be 'bdf1' or 'bdf2'.")


def bdf_symbol(
    zeta: ArrayLike,
    method: ConvolutionQuadratureMethod,
    /,
) -> Array:
    r"""Evaluate the BDF1 or BDF2 generating polynomial $\delta(\zeta)$.

    The symbols are ``1 - zeta`` and
    ``3 / 2 - 2 * zeta + zeta**2 / 2``. This supplies only the temporal
    multistep symbol; it makes no statement about the caller's PDE or
    complex-frequency operator family.
    """
    value = jnp.asarray(zeta)
    order = _method_order(method)
    if order == 1:
        return 1.0 - value
    return 1.5 - 2.0 * value + 0.5 * value * value


class ConvolutionQuadratureContourPolicy(StrictModule):
    """Explicit or roundoff-balanced circular-contour radius policy.

    The automatic policy balances with the realized real dtype and chooses
    ``radius**(2 * fft_length) == machine_epsilon``. An explicit ``tolerance``
    replaces machine epsilon. Either value is a policy target, not an a
    posteriori error bound.
    """

    radius: float | None = eqx.field(static=True)
    tolerance: float | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        radius: float | None = None,
        tolerance: float | None = None,
    ):
        radius_ = None if radius is None else float(radius)
        tolerance_ = None if tolerance is None else float(tolerance)
        if radius_ is not None and (
            not math.isfinite(radius_) or not 0.0 < radius_ < 1.0
        ):
            raise ValueError("An explicit contour radius must lie strictly in (0, 1).")
        if tolerance_ is not None and (
            not math.isfinite(tolerance_) or not 0.0 < tolerance_ < 1.0
        ):
            raise ValueError("Contour tolerance must lie strictly in (0, 1).")
        self.radius = radius_
        self.tolerance = tolerance_

    def resolved_tolerance(self, machine_epsilon: float, /) -> float:
        epsilon = float(machine_epsilon)
        if not math.isfinite(epsilon) or not 0.0 < epsilon < 1.0:
            raise ValueError("machine_epsilon must lie strictly in (0, 1).")
        if self.tolerance is not None and self.tolerance < epsilon:
            raise ValueError(
                "Contour tolerance is below the realized precision machine epsilon."
            )
        return epsilon if self.tolerance is None else self.tolerance

    def resolve(
        self,
        fft_length: int,
        /,
        *,
        machine_epsilon: float = float(np.finfo(float).eps),
    ) -> float:
        length = int(fft_length)
        if length < 1:
            raise ValueError("fft_length must be positive.")
        if self.radius is not None:
            return self.radius
        tolerance = self.resolved_tolerance(machine_epsilon)
        return math.exp(math.log(tolerance) / (2.0 * length))


class ConvolutionQuadratureContour(StrictModule):
    """Fixed BDF contour for a finite, untruncated causal history.

    ``parameters[k]`` is ``delta(radius * exp(-2 pi i k / L)) / step_size``.
    The contour is a time-discretization artifact only and carries no continuum
    PDE or boundary-integral certification.
    """

    zeta: Array
    parameters: Array
    step_size: Array
    radius: Array
    history_length: int = eqx.field(static=True)
    fft_length: int = eqx.field(static=True)
    method: ConvolutionQuadratureMethod = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    conjugate_symmetric: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        zeta: Array,
        parameters: Array,
        step_size: Array,
        radius: Array,
        history_length: int,
        fft_length: int,
        method: ConvolutionQuadratureMethod,
        tolerance: float,
        conjugate_symmetric: bool,
    ):
        self.zeta = zeta
        self.parameters = parameters
        self.step_size = step_size
        self.radius = radius
        self.history_length = int(history_length)
        self.fft_length = int(fft_length)
        self.method = method
        self.tolerance = float(tolerance)
        self.conjugate_symmetric = bool(conjugate_symmetric)

    @property
    def solved_node_indices(self) -> tuple[int, ...]:
        if self.conjugate_symmetric:
            return tuple(range(self.fft_length // 2 + 1))
        return tuple(range(self.fft_length))


def _default_fft_length(history_length: int, /) -> int:
    required = 2 * history_length
    return 1 << (required - 1).bit_length()


def prepare_convolution_quadrature_contour(
    step_size: ArrayLike,
    history_length: int,
    /,
    *,
    method: ConvolutionQuadratureMethod = "bdf2",
    fft_length: int | None = None,
    policy: ConvolutionQuadratureContourPolicy | None = None,
    conjugate_symmetric: bool = False,
) -> ConvolutionQuadratureContour:
    """Prepare fixed BDF sampling parameters for one complete history.

    The FFT length is at least twice the retained history length. All supplied
    history samples remain active; neither this choice nor conjugacy reduction
    introduces a finite-memory horizon.
    """
    _method_order(method)
    if isinstance(history_length, bool):
        raise TypeError("history_length must be an integer.")
    count = int(history_length)
    if count < 1:
        raise ValueError("history_length must be positive.")
    length = _default_fft_length(count) if fft_length is None else int(fft_length)
    if length < 2 * count:
        raise ValueError("fft_length must be at least twice history_length.")
    if length % 2:
        raise ValueError("fft_length must be even.")
    policy_ = ConvolutionQuadratureContourPolicy() if policy is None else policy
    if not isinstance(policy_, ConvolutionQuadratureContourPolicy):
        raise TypeError("policy must be ConvolutionQuadratureContourPolicy or None.")

    step = jnp.asarray(step_size)
    if step.ndim != 0:
        raise ValueError("step_size must be scalar.")
    if not jnp.issubdtype(step.dtype, jnp.inexact):
        step = step.astype(float)
    if jnp.issubdtype(step.dtype, jnp.complexfloating):
        raise TypeError("step_size must be real-valued.")
    concrete_step = float(np.asarray(step))
    if not math.isfinite(concrete_step) or concrete_step <= 0.0:
        raise ValueError("step_size must be finite and positive.")

    machine_epsilon = float(np.finfo(np.dtype(step.dtype)).eps)
    tolerance_value = policy_.resolved_tolerance(machine_epsilon)
    radius_value = policy_.resolve(length, machine_epsilon=machine_epsilon)
    real_dtype = step.dtype
    complex_dtype = jnp.result_type(real_dtype, jnp.complex64)
    indices = jnp.arange(length, dtype=real_dtype)
    phase = jnp.asarray(-2.0j * math.pi, dtype=complex_dtype) * indices / length
    radius = jnp.asarray(radius_value, dtype=real_dtype)
    zeta = radius.astype(complex_dtype) * jnp.exp(phase)
    parameters = bdf_symbol(zeta, method) / step.astype(complex_dtype)
    return ConvolutionQuadratureContour(
        zeta=zeta,
        parameters=parameters,
        step_size=step,
        radius=radius,
        history_length=count,
        fft_length=length,
        method=method,
        tolerance=tolerance_value,
        conjugate_symmetric=conjugate_symmetric,
    )


def convolution_quadrature_fft(
    history: ArrayLike,
    contour: ConvolutionQuadratureContour,
    /,
) -> Array:
    """Radially weight, zero-pad, and FFT a complete leading-axis history."""
    if not isinstance(contour, ConvolutionQuadratureContour):
        raise TypeError("contour must be ConvolutionQuadratureContour.")
    values = jnp.asarray(history)
    if values.ndim < 1:
        raise ValueError("history must have a leading time axis.")
    count = int(values.shape[0])
    if count > contour.history_length:
        raise ValueError("history exceeds the prepared history length.")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    pad = ((0, contour.fft_length - count),) + ((0, 0),) * (values.ndim - 1)
    padded = jnp.pad(values, pad)
    powers = jnp.power(contour.radius, jnp.arange(contour.fft_length))
    powers = powers.reshape((contour.fft_length,) + (1,) * (values.ndim - 1))
    return jnp.fft.fft(padded * powers, axis=0)


def convolution_quadrature_ifft(
    spectrum: ArrayLike,
    contour: ConvolutionQuadratureContour,
    /,
    *,
    history_length: int | None = None,
) -> Array:
    """Invert radial weighting and return the requested leading history prefix."""
    if not isinstance(contour, ConvolutionQuadratureContour):
        raise TypeError("contour must be ConvolutionQuadratureContour.")
    values = jnp.asarray(spectrum)
    if values.ndim < 1 or int(values.shape[0]) != contour.fft_length:
        raise ValueError("spectrum leading axis must equal contour.fft_length.")
    count = contour.history_length if history_length is None else int(history_length)
    if count < 1 or count > contour.history_length:
        raise ValueError("history_length must lie in the prepared history envelope.")
    physical = jnp.fft.ifft(values, axis=0)[:count]
    inverse_powers = jnp.power(contour.radius, -jnp.arange(count))
    inverse_powers = inverse_powers.reshape((count,) + (1,) * (values.ndim - 1))
    return physical * inverse_powers


def causal_prefix_fft(
    history: ArrayLike,
    contour: ConvolutionQuadratureContour,
    /,
) -> Array:
    """FFT every causal prefix without dropping any retained history sample.

    The result has shape ``(fft_length, history_length, ...)``. Prefix ``n``
    contains samples ``0`` through ``n`` and exact zeros thereafter, so taking
    coefficient ``n`` after a node-family action cannot receive circular
    wrap-around from future data.
    """
    if not isinstance(contour, ConvolutionQuadratureContour):
        raise TypeError("contour must be ConvolutionQuadratureContour.")
    values = jnp.asarray(history)
    if values.ndim < 2:
        raise ValueError("history must have time and coordinate axes.")
    if int(values.shape[0]) != contour.history_length:
        raise ValueError("history leading axis must match the prepared history length.")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)

    count, length = contour.history_length, contour.fft_length
    pad = ((0, length - count),) + ((0, 0),) * (values.ndim - 1)
    padded = jnp.pad(values, pad)
    target_indices = jnp.arange(count)[:, None]
    source_indices = jnp.arange(length)[None, :]
    causal = source_indices <= target_indices
    causal = causal.reshape((count, length) + (1,) * (values.ndim - 1))
    prefixes = jnp.where(causal, padded[None, ...], 0)
    powers = jnp.power(contour.radius, jnp.arange(length))
    powers = powers.reshape((1, length) + (1,) * (values.ndim - 1))
    transformed = jnp.fft.fft(prefixes * powers, axis=1)
    return jnp.swapaxes(transformed, 0, 1)


def reconstruct_causal_history(
    node_values: ArrayLike,
    contour: ConvolutionQuadratureContour,
    /,
) -> Array:
    """Recover the diagonal causal coefficient from all prefix node actions.

    ``node_values`` has axes ``(frequency, coordinate, prefix, batch...)``.
    The returned axes are ``(time, coordinate, batch...)``.
    """
    if not isinstance(contour, ConvolutionQuadratureContour):
        raise TypeError("contour must be ConvolutionQuadratureContour.")
    values = jnp.asarray(node_values)
    if values.ndim < 3:
        raise ValueError("node_values must have frequency, coordinate, and prefix axes.")
    if int(values.shape[0]) != contour.fft_length:
        raise ValueError("node_values frequency axis does not match the contour.")
    if int(values.shape[2]) != contour.history_length:
        raise ValueError("node_values prefix axis does not match the history length.")
    coefficients = jnp.fft.ifft(values, axis=0)
    indices = jnp.arange(contour.history_length)
    diagonal = coefficients[indices, :, indices, ...]
    inverse_powers = jnp.power(contour.radius, -indices)
    inverse_powers = inverse_powers.reshape(
        (contour.history_length,) + (1,) * (diagonal.ndim - 1)
    )
    return diagonal * inverse_powers


__all__ = [
    "ConvolutionQuadratureContour",
    "ConvolutionQuadratureContourPolicy",
    "ConvolutionQuadratureMethod",
    "bdf_symbol",
    "causal_prefix_fft",
    "convolution_quadrature_fft",
    "convolution_quadrature_ifft",
    "prepare_convolution_quadrature_contour",
    "reconstruct_causal_history",
]
