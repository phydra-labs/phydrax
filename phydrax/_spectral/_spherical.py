#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib.metadata import version
from math import prod
from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from jax.extend import core as jax_core
from jax.interpreters import ad, batching, mlir
from jaxtyping import Array, ArrayLike
from s2fft.precompute_transforms import (
    construct as s2fft_construct,
    spherical as s2fft_precomputed,
)
from s2fft.sampling import s2_samples
from s2fft.transforms import spherical as s2fft_spherical
from s2fft.utils import quadrature as s2fft_quadrature

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


SphericalSampling: TypeAlias = Literal["mw", "mwss", "dh", "gl"]
SphericalExecution: TypeAlias = Literal["recursive", "precomputed"]
_DEFAULT_PRECOMPUTE_BYTES = 512 * 1024**2
_SPHERICAL_MODE_NORMALIZATION = "s2fft-orthonormal-condon-shortley"


def _spherical_mode_layout_id(
    bandlimit: int,
    spin: int,
    reality: bool,
    /,
) -> str:
    degree = np.arange(bandlimit, dtype=np.int32)[:, None]
    order = np.arange(-(bandlimit - 1), bandlimit, dtype=np.int32)[None, :]
    shape = (bandlimit, 2 * bandlimit - 1)
    degrees = np.broadcast_to(degree, shape)
    orders = np.broadcast_to(order, shape)
    valid = (np.abs(orders) <= degrees) & (degrees >= abs(spin))
    independent = valid & (orders >= 0) if reality else valid
    return canonical_fingerprint(
        {
            "kind": "spherical-mode-layout-v1",
            "bandlimit": bandlimit,
            "spin": spin,
            "reality": reality,
            "valid": array_tree_fingerprint(valid),
            "independent": array_tree_fingerprint(independent),
            "normalization": _SPHERICAL_MODE_NORMALIZATION,
        }
    )


def _array_bytes(tree: object, /) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, (np.ndarray, jax.Array))
    )


def _validate_precompute_limit(
    limit: int,
    /,
    *,
    estimate: int,
    execution: SphericalExecution,
) -> int:
    value = int(limit)
    if value <= 0:
        raise ValueError("max_precompute_bytes must be positive.")
    if estimate > value:
        raise ValueError(
            f"Spherical {execution} preparation exceeds max_precompute_bytes; "
            f"estimated {estimate} bytes."
        )
    return value


def _recursive_transform_impl(
    values, precomputes, *, bandlimit, spin, sampling, reality, inverse
):
    transform = s2fft_spherical.inverse_jax if inverse else s2fft_spherical.forward_jax
    return transform(values, bandlimit, spin, None, sampling, reality, precomputes)


def _recursive_linear_impl(
    values, *precomputes, transpose, source_shape, source_dtype, **configuration
):
    def apply(field):
        return _recursive_transform_impl(field, precomputes, **configuration)

    if transpose:
        # Retain the backend's real-linear adjoint, including its FFT scaling
        # and real-field order convention. Never assume inverse == adjoint.
        _, pullback = jax.vjp(apply, jnp.zeros(source_shape, dtype=source_dtype))

        def action(field):
            return pullback(field)[0]

    else:
        action = apply
    fields = values.reshape((-1, *values.shape[-2:]))
    result = jax.vmap(action)(fields)
    return result.reshape((*values.shape[:-2], *result.shape[-2:]))


def _recursive_linear_abstract(*operands, **configuration):
    shapes = tuple(jax.ShapeDtypeStruct(value.shape, value.dtype) for value in operands)
    result = jax.eval_shape(
        lambda *values: _recursive_linear_impl(*values, **configuration), *shapes
    )
    return jax.core.ShapedArray(result.shape, result.dtype)


_recursive_linear_p = jax_core.Primitive("phydrax_recursive_spherical_linear")
_recursive_linear_p.def_impl(_recursive_linear_impl)
_recursive_linear_p.def_abstract_eval(_recursive_linear_abstract)
mlir.register_lowering(
    _recursive_linear_p,
    mlir.lower_fun(_recursive_linear_impl, multiple_results=False),
)


def _recursive_linear_jvp(primals, tangents, **configuration):
    values_tangent, *table_tangents = tangents
    if any(not isinstance(tangent, ad.Zero) for tangent in table_tangents):
        raise ValueError(
            "Recursive spherical precomputes are fixed preparation state; "
            "differentiate sampled values or harmonic coefficients, not the tables."
        )
    output = _recursive_linear_p.bind(*primals, **configuration)
    tangent = (
        ad.Zero(jax.typeof(output))
        if isinstance(values_tangent, ad.Zero)
        else _recursive_linear_p.bind(values_tangent, *primals[1:], **configuration)
    )
    return output, tangent


def _recursive_linear_transpose(cotangent, values, *precomputes, **configuration):
    if any(ad.is_undefined_primal(table) for table in precomputes):
        raise ValueError("Recursive spherical precomputes are fixed preparation state.")
    if isinstance(cotangent, ad.Zero):
        result = ad.Zero(values.aval)
    else:
        configuration = {**configuration, "transpose": not configuration["transpose"]}
        result = _recursive_linear_p.bind(cotangent, *precomputes, **configuration)
    return (result, *(None for _ in precomputes))


def _recursive_linear_batch(operands, axes, **configuration):
    value_axis, *table_axes = axes
    if any(axis is not None for axis in table_axes):
        raise ValueError(
            "Batch sampled fields, not recursive spherical preparation tables."
        )
    if value_axis is None:
        return _recursive_linear_p.bind(*operands, **configuration), None
    values = jnp.moveaxis(operands[0], value_axis, 0)
    return _recursive_linear_p.bind(values, *operands[1:], **configuration), 0


ad.primitive_jvps[_recursive_linear_p] = _recursive_linear_jvp
ad.primitive_transposes[_recursive_linear_p] = _recursive_linear_transpose
batching.primitive_batchers[_recursive_linear_p] = _recursive_linear_batch


def _fixed_recursive_transform(values, precomputes, **configuration):
    """Apply a fixed linear transform with paired native forward/adjoint rules.

    Both orientations remain linear primitives under higher AD, so neither a
    custom_vjp primal nor its native pullback is differentiated a second time.
    """
    return _recursive_linear_p.bind(
        values,
        *precomputes,
        transpose=False,
        source_shape=values.shape,
        source_dtype=values.dtype,
        **configuration,
    )


class _RecursiveSphericalExecution(StrictModule, NonTrainableState):
    forward_precomputes: tuple[Array, ...]
    inverse_precomputes: tuple[Array, ...]

    def __init__(
        self,
        *,
        bandlimit: int,
        spin: int,
        sampling: SphericalSampling,
        max_precompute_bytes: int,
    ):
        estimate = 256 * bandlimit**2
        limit = _validate_precompute_limit(
            max_precompute_bytes,
            estimate=estimate,
            execution="recursive",
        )
        forward = tuple(
            s2fft.generate_precomputes_jax(
                bandlimit,
                spin,
                sampling,
                None,
                True,
            )
        )
        inverse = tuple(
            s2fft.generate_precomputes_jax(
                bandlimit,
                spin,
                sampling,
                None,
                False,
            )
        )
        actual = _array_bytes((forward, inverse))
        if actual > limit:
            raise ValueError(
                "Spherical recursive preparation exceeds max_precompute_bytes; "
                f"actual size is {actual} bytes."
            )
        self.forward_precomputes = forward
        self.inverse_precomputes = inverse

    def forward(
        self,
        values: Array,
        /,
        *,
        bandlimit: int,
        spin: int,
        sampling: SphericalSampling,
        reality: bool,
    ) -> Array:
        return _fixed_recursive_transform(
            values,
            self.forward_precomputes,
            bandlimit=bandlimit,
            spin=spin,
            sampling=sampling,
            reality=reality,
            inverse=False,
        )

    def inverse(
        self,
        coefficients: Array,
        /,
        *,
        bandlimit: int,
        spin: int,
        sampling: SphericalSampling,
        reality: bool,
    ) -> Array:
        return _fixed_recursive_transform(
            coefficients,
            self.inverse_precomputes,
            bandlimit=bandlimit,
            spin=spin,
            sampling=sampling,
            reality=reality,
            inverse=True,
        )


class _PrecomputedSphericalExecution(StrictModule, NonTrainableState):
    forward_kernel: Array
    inverse_kernel: Array

    def __init__(
        self,
        *,
        bandlimit: int,
        spin: int,
        sampling: SphericalSampling,
        reality: bool,
        max_precompute_bytes: int,
    ):
        order_count = bandlimit if reality else 2 * bandlimit - 1
        forward_theta = (
            s2_samples.ntheta(2 * bandlimit, "mwss")
            if sampling in ("mw", "mwss")
            else s2_samples.ntheta(bandlimit, sampling)
        )
        inverse_theta = s2_samples.ntheta(bandlimit, sampling)
        estimate = (
            (forward_theta + inverse_theta)
            * bandlimit
            * order_count
            * np.dtype(float).itemsize
        )
        limit = _validate_precompute_limit(
            max_precompute_bytes,
            estimate=estimate,
            execution="precomputed",
        )
        forward = jnp.asarray(
            s2fft_construct.spin_spherical_kernel(
                bandlimit,
                spin=spin,
                reality=reality,
                sampling=sampling,
                forward=True,
            )
        )
        inverse = jnp.asarray(
            s2fft_construct.spin_spherical_kernel(
                bandlimit,
                spin=spin,
                reality=reality,
                sampling=sampling,
                forward=False,
            )
        )
        actual = _array_bytes((forward, inverse))
        if actual > limit:
            raise ValueError(
                "Spherical precomputed preparation exceeds max_precompute_bytes; "
                f"actual size is {actual} bytes."
            )
        self.forward_kernel = forward
        self.inverse_kernel = inverse

    def forward(
        self,
        values: Array,
        /,
        *,
        bandlimit: int,
        spin: int,
        sampling: SphericalSampling,
        reality: bool,
    ) -> Array:
        return s2fft_precomputed.forward_transform_jax(
            values,
            self.forward_kernel,
            bandlimit,
            sampling,
            reality,
            spin,
            None,
        )

    def inverse(
        self,
        coefficients: Array,
        /,
        *,
        bandlimit: int,
        spin: int,
        sampling: SphericalSampling,
        reality: bool,
    ) -> Array:
        return s2fft_precomputed.inverse_transform_jax(
            coefficients,
            self.inverse_kernel,
            bandlimit,
            sampling,
            reality,
            spin,
            None,
        )


class SphericalHarmonicPlan(StrictModule, NonTrainableState):
    """Prepared differentiable S2FFT analysis/synthesis on an exact sampling theorem."""

    theta: Array
    phi: Array
    theta_quadrature_weights: Array
    phi_quadrature_weights: Array
    transform: _RecursiveSphericalExecution | _PrecomputedSphericalExecution
    bandlimit: int
    sampling: SphericalSampling
    spin: int
    reality: bool
    execution: SphericalExecution
    sample_shape: tuple[int, int]
    coefficient_shape: tuple[int, int]
    fingerprint: str
    layout_id: str

    def __init__(
        self,
        bandlimit: int,
        /,
        *,
        sampling: SphericalSampling = "mw",
        spin: int = 0,
        reality: bool = True,
        execution: SphericalExecution = "recursive",
        max_precompute_bytes: int = _DEFAULT_PRECOMPUTE_BYTES,
    ):
        selected_bandlimit = int(bandlimit)
        selected_sampling = str(sampling).lower()
        selected_spin = int(spin)
        selected_reality = bool(reality)
        selected_execution = str(execution).lower()
        if selected_bandlimit <= abs(selected_spin):
            raise ValueError("bandlimit must exceed the absolute spin.")
        if selected_sampling not in ("mw", "mwss", "dh", "gl"):
            raise ValueError("sampling must be 'mw', 'mwss', 'dh', or 'gl'.")
        if selected_reality and selected_spin != 0:
            raise ValueError("reality acceleration is valid only for spin-zero fields.")
        if selected_execution not in ("recursive", "precomputed"):
            raise ValueError("execution must be 'recursive' or 'precomputed'.")
        sampling_value = selected_sampling
        execution_value = selected_execution
        theta = np.asarray(
            s2_samples.thetas(selected_bandlimit, sampling_value), dtype=float
        )
        phi = np.asarray(
            s2_samples.phis_equiang(selected_bandlimit, sampling_value), dtype=float
        )
        pixel_weights = np.asarray(
            s2fft_quadrature.quad_weights(
                selected_bandlimit,
                sampling_value,
                spin=0,
            ),
            dtype=float,
        )
        phi_weight = 2.0 * np.pi / phi.size
        theta_weights = pixel_weights / phi_weight
        if execution_value == "recursive":
            transform: _RecursiveSphericalExecution | _PrecomputedSphericalExecution
            transform = _RecursiveSphericalExecution(
                bandlimit=selected_bandlimit,
                spin=selected_spin,
                sampling=sampling_value,
                max_precompute_bytes=max_precompute_bytes,
            )
        else:
            transform = _PrecomputedSphericalExecution(
                bandlimit=selected_bandlimit,
                spin=selected_spin,
                sampling=sampling_value,
                reality=selected_reality,
                max_precompute_bytes=max_precompute_bytes,
            )
        self.theta = jnp.asarray(theta)
        self.phi = jnp.asarray(phi)
        self.theta_quadrature_weights = jnp.asarray(theta_weights)
        self.phi_quadrature_weights = jnp.full(
            phi.shape,
            phi_weight,
            dtype=self.theta.dtype,
        )
        self.transform = transform
        self.bandlimit = selected_bandlimit
        self.sampling = sampling_value
        self.spin = selected_spin
        self.reality = selected_reality
        self.execution = execution_value
        sample_shape = tuple(
            int(size) for size in s2_samples.f_shape(selected_bandlimit, sampling_value)
        )
        coefficient_shape = tuple(
            int(size) for size in s2_samples.flm_shape(selected_bandlimit)
        )
        if len(sample_shape) != 2 or len(coefficient_shape) != 2:
            raise RuntimeError("S2FFT returned an invalid rank-two transform shape.")
        self.sample_shape = (sample_shape[0], sample_shape[1])
        self.coefficient_shape = (coefficient_shape[0], coefficient_shape[1])
        self.layout_id = _spherical_mode_layout_id(
            selected_bandlimit,
            selected_spin,
            selected_reality,
        )
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "spherical-harmonic-plan-v1",
                "s2fft": version("s2fft"),
                "bandlimit": selected_bandlimit,
                "sampling": sampling_value,
                "spin": selected_spin,
                "reality": selected_reality,
            }
        )

    @property
    def transform_id(self) -> str:
        """Sampling-theorem identity shared by equivalent execution strategies."""
        return self.fingerprint

    @property
    def precompute_bytes(self) -> int:
        """Materialized recursive tables or precomputed transform kernels."""
        return _array_bytes(self.transform)

    @property
    def execution_id(self) -> str:
        """Identity of one concrete transform execution realization."""
        return canonical_fingerprint(
            {
                "kind": "spherical-harmonic-execution-v1",
                "transform": self.transform_id,
                "execution": self.execution,
                "precompute_bytes": self.precompute_bytes,
            }
        )

    def _forward_field(self, values: Array, /) -> Array:
        return self.transform.forward(
            values,
            bandlimit=self.bandlimit,
            spin=self.spin,
            sampling=self.sampling,
            reality=self.reality,
        )

    def _inverse_field(self, coefficients: Array, /) -> Array:
        return self.transform.inverse(
            coefficients,
            bandlimit=self.bandlimit,
            spin=self.spin,
            sampling=self.sampling,
            reality=self.reality,
        )

    def analysis(self, values: ArrayLike, /) -> Array:
        """Transform scalar or channel-last sampled fields to ``(ell, m)`` arrays."""
        array = jnp.asarray(values)
        scalar = tuple(int(size) for size in array.shape[-2:]) == self.sample_shape
        if scalar:
            leading_shape = tuple(int(size) for size in array.shape[:-2])
            fields = array.reshape((prod(leading_shape), *self.sample_shape))
        elif (
            array.ndim >= 3
            and tuple(int(size) for size in array.shape[-3:-1]) == self.sample_shape
        ):
            leading_shape = tuple(int(size) for size in array.shape[:-3])
            channels = int(array.shape[-1])
            fields = jnp.moveaxis(array, -1, -3).reshape(
                (prod(leading_shape) * channels, *self.sample_shape)
            )
        else:
            raise ValueError(
                "Spherical analysis expects (..., n_theta, n_phi) or "
                "(..., n_theta, n_phi, channels)."
            )
        if self.reality and jnp.issubdtype(array.dtype, jnp.complexfloating):
            raise TypeError("A reality-accelerated spherical plan requires real values.")
        coefficients = jax.vmap(self._forward_field)(fields)
        if scalar:
            return coefficients.reshape(leading_shape + self.coefficient_shape)
        result = coefficients.reshape(leading_shape + (channels, *self.coefficient_shape))
        return jnp.moveaxis(result, -3, -1)

    def synthesis(self, coefficients: ArrayLike, /) -> Array:
        """Transform scalar or channel-last ``(ell, m)`` arrays to sampled fields."""
        array = jnp.asarray(coefficients)
        scalar = tuple(int(size) for size in array.shape[-2:]) == self.coefficient_shape
        if scalar:
            leading_shape = tuple(int(size) for size in array.shape[:-2])
            fields = array.reshape((prod(leading_shape), *self.coefficient_shape))
        elif (
            array.ndim >= 3
            and tuple(int(size) for size in array.shape[-3:-1]) == self.coefficient_shape
        ):
            leading_shape = tuple(int(size) for size in array.shape[:-3])
            channels = int(array.shape[-1])
            fields = jnp.moveaxis(array, -1, -3).reshape(
                (prod(leading_shape) * channels, *self.coefficient_shape)
            )
        else:
            raise ValueError(
                "Spherical synthesis expects (..., bandlimit, 2*bandlimit-1) or "
                "(..., bandlimit, 2*bandlimit-1, channels)."
            )
        values = jax.vmap(self._inverse_field)(fields)
        if scalar:
            return values.reshape(leading_shape + self.sample_shape)
        result = values.reshape(leading_shape + (channels, *self.sample_shape))
        return jnp.moveaxis(result, -3, -1)


__all__ = [
    "SphericalExecution",
    "SphericalHarmonicPlan",
    "SphericalSampling",
]
