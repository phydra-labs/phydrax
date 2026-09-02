#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from importlib.metadata import version
from math import prod

import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from jaxtyping import Array, ArrayLike
from s2fft.precompute_transforms import (
    construct as s2fft_construct,
    wigner as s2fft_precomputed,
)
from s2fft.sampling import s2_samples, so3_samples
from s2fft.transforms import wigner as s2fft_wigner
from s2fft.utils import quadrature as s2fft_quadrature

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._spherical import SphericalExecution, SphericalSampling


_DEFAULT_PRECOMPUTE_BYTES = 512 * 1024**2
_WIGNER_NORMALIZATION = "s2fft-active-zyz-raw-haar-8pi2"


def _array_bytes(tree: object, /) -> int:
    return sum(
        int(leaf.size) * int(leaf.dtype.itemsize)
        for leaf in jax.tree_util.tree_leaves(tree)
        if isinstance(leaf, (np.ndarray, jax.Array))
    )


def _validate_limit(limit: int, estimate: int, kind: str, /) -> int:
    value = int(limit)
    if value <= 0:
        raise ValueError("max_precompute_bytes must be positive.")
    if estimate > value:
        raise ValueError(
            f"Wigner {kind} preparation exceeds max_precompute_bytes; "
            f"estimated {estimate} bytes."
        )
    return value


class _RecursiveWignerExecution(StrictModule, NonTrainableState):
    forward_precomputes: tuple[Array, ...]
    inverse_precomputes: tuple[Array, ...]

    def __init__(
        self,
        *,
        bandlimit: int,
        directional_bandlimit: int,
        sampling: SphericalSampling,
        lower_bandlimit: int,
        max_precompute_bytes: int,
    ):
        estimate = 512 * directional_bandlimit * bandlimit**2
        limit = _validate_limit(max_precompute_bytes, estimate, "recursive")
        forward = tuple(
            s2fft.generate_precomputes_wigner_jax(
                bandlimit,
                directional_bandlimit,
                sampling,
                None,
                True,
                False,
                lower_bandlimit,
            )
        )
        inverse = tuple(
            s2fft.generate_precomputes_wigner_jax(
                bandlimit,
                directional_bandlimit,
                sampling,
                None,
                False,
                False,
                lower_bandlimit,
            )
        )
        actual = _array_bytes((forward, inverse))
        if actual > limit:
            raise ValueError(
                "Wigner recursive preparation exceeds max_precompute_bytes; "
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
        directional_bandlimit: int,
        sampling: SphericalSampling,
        lower_bandlimit: int,
    ) -> Array:
        return s2fft_wigner.forward_jax(
            values,
            bandlimit,
            directional_bandlimit,
            None,
            sampling,
            False,
            self.forward_precomputes,
            lower_bandlimit,
        )

    def inverse(
        self,
        coefficients: Array,
        /,
        *,
        bandlimit: int,
        directional_bandlimit: int,
        sampling: SphericalSampling,
        lower_bandlimit: int,
    ) -> Array:
        return s2fft_wigner.inverse_jax(
            coefficients,
            bandlimit,
            directional_bandlimit,
            None,
            sampling,
            False,
            self.inverse_precomputes,
            lower_bandlimit,
        )


class _PrecomputedWignerExecution(StrictModule, NonTrainableState):
    forward_kernel: Array
    inverse_kernel: Array

    def __init__(
        self,
        *,
        bandlimit: int,
        directional_bandlimit: int,
        sampling: SphericalSampling,
        max_precompute_bytes: int,
    ):
        forward_theta = (
            s2_samples.ntheta(2 * bandlimit, "mwss")
            if sampling in ("mw", "mwss")
            else s2_samples.ntheta(bandlimit, sampling)
        )
        inverse_theta = s2_samples.ntheta(bandlimit, sampling)
        estimate = (
            (2 * directional_bandlimit - 1)
            * (forward_theta + inverse_theta)
            * bandlimit
            * (2 * bandlimit - 1)
            * np.dtype(float).itemsize
        )
        limit = _validate_limit(max_precompute_bytes, estimate, "precomputed")
        forward = jnp.asarray(
            s2fft_construct.wigner_kernel(
                bandlimit,
                directional_bandlimit,
                reality=False,
                sampling=sampling,
                forward=True,
            )
        )
        inverse = jnp.asarray(
            s2fft_construct.wigner_kernel(
                bandlimit,
                directional_bandlimit,
                reality=False,
                sampling=sampling,
                forward=False,
            )
        )
        actual = _array_bytes((forward, inverse))
        if actual > limit:
            raise ValueError(
                "Wigner precomputed preparation exceeds max_precompute_bytes; "
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
        directional_bandlimit: int,
        sampling: SphericalSampling,
        lower_bandlimit: int,
    ) -> Array:
        coefficients = s2fft_precomputed.forward_transform_jax(
            values,
            self.forward_kernel,
            bandlimit,
            directional_bandlimit,
            sampling,
            False,
            None,
        )
        if lower_bandlimit == 0:
            return coefficients
        degree = jnp.arange(bandlimit)[None, :, None]
        return jnp.where(degree >= lower_bandlimit, coefficients, 0.0)

    def inverse(
        self,
        coefficients: Array,
        /,
        *,
        bandlimit: int,
        directional_bandlimit: int,
        sampling: SphericalSampling,
        lower_bandlimit: int,
    ) -> Array:
        if lower_bandlimit:
            degree = jnp.arange(bandlimit)[None, :, None]
            coefficients = jnp.where(degree >= lower_bandlimit, coefficients, 0.0)
        return s2fft_precomputed.inverse_transform_jax(
            coefficients,
            self.inverse_kernel,
            bandlimit,
            directional_bandlimit,
            sampling,
            False,
            None,
        )


class WignerTransformPlan(StrictModule, NonTrainableState):
    """Prepared complex Wigner transform on SO(3) using raw Haar measure."""

    alpha: Array
    beta: Array
    gamma: Array
    alpha_quadrature_weights: Array
    beta_quadrature_weights: Array
    gamma_quadrature_weights: Array
    transform: _RecursiveWignerExecution | _PrecomputedWignerExecution
    _valid_mask: Array
    bandlimit: int
    directional_bandlimit: int
    lower_bandlimit: int
    sampling: SphericalSampling
    execution: SphericalExecution
    sample_shape: tuple[int, int, int]
    coefficient_shape: tuple[int, int, int]
    normalization: str
    fingerprint: str
    layout_id: str

    def __init__(
        self,
        bandlimit: int,
        directional_bandlimit: int,
        /,
        *,
        sampling: SphericalSampling = "mw",
        execution: SphericalExecution = "recursive",
        lower_bandlimit: int = 0,
        max_precompute_bytes: int = _DEFAULT_PRECOMPUTE_BYTES,
    ):
        selected_bandlimit = int(bandlimit)
        selected_directional = int(directional_bandlimit)
        selected_lower = int(lower_bandlimit)
        selected_sampling = str(sampling).lower()
        selected_execution = str(execution).lower()
        if selected_bandlimit < 1:
            raise ValueError("bandlimit must be positive.")
        if selected_directional < 1 or selected_directional > selected_bandlimit:
            raise ValueError("directional_bandlimit must satisfy 1 <= N <= L.")
        if selected_lower < 0 or selected_lower >= selected_bandlimit:
            raise ValueError("lower_bandlimit must satisfy 0 <= L_lower < L.")
        if selected_sampling not in ("mw", "mwss", "dh", "gl"):
            raise ValueError("sampling must be 'mw', 'mwss', 'dh', or 'gl'.")
        if selected_execution not in ("recursive", "precomputed"):
            raise ValueError("execution must be 'recursive' or 'precomputed'.")
        if selected_execution == "recursive" and selected_directional >= 8:
            raise ValueError(
                "recursive Wigner execution is certified only for "
                "directional_bandlimit < 8; use precomputed execution."
            )
        sampling_value = selected_sampling
        execution_value = selected_execution

        alpha = np.asarray(
            s2_samples.phis_equiang(selected_bandlimit, sampling_value), dtype=float
        )
        beta = np.asarray(
            s2_samples.thetas(selected_bandlimit, sampling_value), dtype=float
        )
        gamma = (
            2.0
            * np.pi
            * np.arange(2 * selected_directional - 1)
            / (2 * selected_directional - 1)
        )
        pixel_weights = np.asarray(
            s2fft_quadrature.quad_weights(
                selected_bandlimit,
                sampling_value,
                spin=0,
            ),
            dtype=float,
        )
        alpha_weight = 2.0 * np.pi / alpha.size
        gamma_weight = 2.0 * np.pi / gamma.size

        if execution_value == "recursive":
            transform: _RecursiveWignerExecution | _PrecomputedWignerExecution
            transform = _RecursiveWignerExecution(
                bandlimit=selected_bandlimit,
                directional_bandlimit=selected_directional,
                sampling=sampling_value,
                lower_bandlimit=selected_lower,
                max_precompute_bytes=max_precompute_bytes,
            )
        else:
            transform = _PrecomputedWignerExecution(
                bandlimit=selected_bandlimit,
                directional_bandlimit=selected_directional,
                sampling=sampling_value,
                max_precompute_bytes=max_precompute_bytes,
            )

        sample_shape = tuple(
            int(size)
            for size in so3_samples.f_shape(
                selected_bandlimit,
                selected_directional,
                sampling_value,
            )
        )
        coefficient_shape = tuple(
            int(size)
            for size in so3_samples.flmn_shape(
                selected_bandlimit,
                selected_directional,
            )
        )
        if len(sample_shape) != 3 or len(coefficient_shape) != 3:
            raise RuntimeError("S2FFT returned an invalid rank-three Wigner shape.")
        order_n = np.arange(
            -(selected_directional - 1), selected_directional, dtype=np.int32
        )[:, None, None]
        degree = np.arange(selected_bandlimit, dtype=np.int32)[None, :, None]
        order_m = np.arange(
            -(selected_bandlimit - 1), selected_bandlimit, dtype=np.int32
        )[None, None, :]
        valid = (
            (np.abs(order_n) <= degree)
            & (np.abs(order_m) <= degree)
            & (degree >= selected_lower)
        )

        self.alpha = jnp.asarray(alpha)
        self.beta = jnp.asarray(beta)
        self.gamma = jnp.asarray(gamma)
        self.alpha_quadrature_weights = jnp.full(
            alpha.shape, alpha_weight, dtype=self.alpha.dtype
        )
        self.beta_quadrature_weights = jnp.asarray(pixel_weights / alpha_weight)
        self.gamma_quadrature_weights = jnp.full(
            gamma.shape, gamma_weight, dtype=self.gamma.dtype
        )
        self.transform = transform
        self._valid_mask = jnp.asarray(valid)
        self.bandlimit = selected_bandlimit
        self.directional_bandlimit = selected_directional
        self.lower_bandlimit = selected_lower
        self.sampling = sampling_value
        self.execution = execution_value
        self.sample_shape = (sample_shape[0], sample_shape[1], sample_shape[2])
        self.coefficient_shape = (
            coefficient_shape[0],
            coefficient_shape[1],
            coefficient_shape[2],
        )
        self.normalization = _WIGNER_NORMALIZATION
        self.layout_id = canonical_fingerprint(
            {
                "kind": "wigner-mode-layout-v1",
                "bandlimit": selected_bandlimit,
                "directional_bandlimit": selected_directional,
                "lower_bandlimit": selected_lower,
                "valid": array_tree_fingerprint(valid),
                "normalization": self.normalization,
            }
        )
        self.fingerprint = canonical_fingerprint(
            {
                "kind": "wigner-transform-plan-v1",
                "s2fft": version("s2fft"),
                "bandlimit": selected_bandlimit,
                "directional_bandlimit": selected_directional,
                "lower_bandlimit": selected_lower,
                "sampling": sampling_value,
                "normalization": self.normalization,
            }
        )

    @property
    def transform_id(self) -> str:
        """SO(3) sampling-theorem identity shared across executions."""
        return self.fingerprint

    @property
    def precompute_bytes(self) -> int:
        """Bytes retained by recursive tables or precomputed Wigner kernels."""
        return _array_bytes(self.transform)

    @property
    def execution_id(self) -> str:
        """Identity of one concrete Wigner execution realization."""
        return canonical_fingerprint(
            {
                "kind": "wigner-transform-execution-v1",
                "transform": self.transform_id,
                "execution": self.execution,
                "precompute_bytes": self.precompute_bytes,
            }
        )

    def _canonicalize_real_coefficients(self, coefficients: Array, /) -> Array:
        center_n = self.directional_bandlimit - 1
        center_m = self.bandlimit - 1
        if center_n:
            negative_n = jnp.arange(
                -(self.directional_bandlimit - 1),
                0,
            )
            order_m = jnp.arange(-(self.bandlimit - 1), self.bandlimit)
            negative = jnp.conj(jnp.flip(coefficients[center_n + 1 :], axis=(0, -1)))
            negative *= (-1.0) ** jnp.abs(negative_n)[:, None, None]
            negative *= (-1.0) ** jnp.abs(order_m)[None, None, :]
            coefficients = coefficients.at[:center_n].set(negative)
        zero_n = coefficients[center_n]
        if center_m:
            negative_m = jnp.arange(-(self.bandlimit - 1), 0)
            reflected = jnp.conj(jnp.flip(zero_n[:, center_m + 1 :], axis=-1))
            reflected *= (-1.0) ** jnp.abs(negative_m)[None, :]
            zero_n = zero_n.at[:, :center_m].set(reflected)
        zero_n = zero_n.at[:, center_m].set(jnp.real(zero_n[:, center_m]))
        return coefficients.at[center_n].set(zero_n)

    def _forward_field(self, values: Array, /) -> Array:
        coefficients = self.transform.forward(
            values,
            bandlimit=self.bandlimit,
            directional_bandlimit=self.directional_bandlimit,
            sampling=self.sampling,
            lower_bandlimit=self.lower_bandlimit,
        )
        if not jnp.issubdtype(values.dtype, jnp.complexfloating):
            coefficients = self._canonicalize_real_coefficients(coefficients)
        return jnp.where(self._valid_mask, coefficients, 0.0)

    def _inverse_field(self, coefficients: Array, /) -> Array:
        sanitized = jnp.where(self._valid_mask, coefficients, 0.0)
        return self.transform.inverse(
            sanitized,
            bandlimit=self.bandlimit,
            directional_bandlimit=self.directional_bandlimit,
            sampling=self.sampling,
            lower_bandlimit=self.lower_bandlimit,
        )

    def analysis(self, values: ArrayLike, /) -> Array:
        """Transform scalar or channel-last SO(3) samples to ``(n, ell, m)``."""
        array = jnp.asarray(values)
        scalar = tuple(int(size) for size in array.shape[-3:]) == self.sample_shape
        if scalar:
            leading_shape = tuple(int(size) for size in array.shape[:-3])
            fields = array.reshape((prod(leading_shape), *self.sample_shape))
        elif (
            array.ndim >= 4
            and tuple(int(size) for size in array.shape[-4:-1]) == self.sample_shape
        ):
            leading_shape = tuple(int(size) for size in array.shape[:-4])
            channels = int(array.shape[-1])
            fields = jnp.moveaxis(array, -1, -4).reshape(
                (prod(leading_shape) * channels, *self.sample_shape)
            )
        else:
            raise ValueError(
                "Wigner analysis expects (..., n_gamma, n_beta, n_alpha) or "
                "(..., n_gamma, n_beta, n_alpha, channels)."
            )
        coefficients = jax.vmap(self._forward_field)(fields)
        if scalar:
            return coefficients.reshape(leading_shape + self.coefficient_shape)
        result = coefficients.reshape(leading_shape + (channels, *self.coefficient_shape))
        return jnp.moveaxis(result, -4, -1)

    def synthesis(self, coefficients: ArrayLike, /) -> Array:
        """Transform scalar or channel-last ``(n, ell, m)`` modes to SO(3)."""
        array = jnp.asarray(coefficients)
        scalar = tuple(int(size) for size in array.shape[-3:]) == self.coefficient_shape
        if scalar:
            leading_shape = tuple(int(size) for size in array.shape[:-3])
            fields = array.reshape((prod(leading_shape), *self.coefficient_shape))
        elif (
            array.ndim >= 4
            and tuple(int(size) for size in array.shape[-4:-1]) == self.coefficient_shape
        ):
            leading_shape = tuple(int(size) for size in array.shape[:-4])
            channels = int(array.shape[-1])
            fields = jnp.moveaxis(array, -1, -4).reshape(
                (prod(leading_shape) * channels, *self.coefficient_shape)
            )
        else:
            raise ValueError(
                "Wigner synthesis expects (..., 2*N-1, L, 2*L-1) or "
                "(..., 2*N-1, L, 2*L-1, channels)."
            )
        values = jax.vmap(self._inverse_field)(fields)
        if scalar:
            return values.reshape(leading_shape + self.sample_shape)
        result = values.reshape(leading_shape + (channels, *self.sample_shape))
        return jnp.moveaxis(result, -4, -1)


__all__ = ["WignerTransformPlan"]
