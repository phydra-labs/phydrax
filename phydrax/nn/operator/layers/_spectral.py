#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._spectral._modal import BasisTransformPlan, SpectralBasis
from ...._strict import StrictModule
from ..data import OperatorAxis


def _apply_axis_matrix(values: Array, matrix: Array, axis: int, /) -> Array:
    transformed = jnp.tensordot(matrix, values, axes=((1,), (axis,)))
    return jnp.moveaxis(transformed, 0, axis)


class BasisSpectralConvND(StrictModule):
    """Quadrature-projected convolution on one separable tensor basis."""

    in_channels: int
    out_channels: int
    n_modes: tuple[int, ...]
    bases: tuple[SpectralBasis, ...]
    weight: Array

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        n_modes: int | Sequence[int],
        bases: SpectralBasis | Sequence[SpectralBasis],
        key: Key[Array, ""] = DOC_KEY0,
    ):
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        modes = (
            (int(n_modes),)
            if isinstance(n_modes, int)
            else tuple(int(mode) for mode in n_modes)
        )
        bases_value = (
            (cast(SpectralBasis, bases),) * len(modes)
            if isinstance(bases, str)
            else tuple(bases)
        )
        if not modes or any(mode <= 0 for mode in modes):
            raise ValueError("Every basis mode count must be positive.")
        if len(bases_value) != len(modes):
            raise ValueError("bases and n_modes must have the same length.")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive.")
        for basis in bases_value:
            if basis not in ("fourier", "sine", "cosine", "legendre"):
                raise ValueError(f"Unsupported spectral basis {basis!r}.")
        self.n_modes = modes
        self.bases = bases_value
        scale = 1.0 / float(self.in_channels * self.out_channels)
        self.weight = scale * jr.normal(
            key,
            shape=(self.in_channels, self.out_channels, *self.n_modes),
        )

    def plan(self, axes: Sequence[OperatorAxis], /) -> BasisTransformPlan:
        """Precompute a nontrainable transform from operator-axis primitives."""
        axes_value = tuple(axes)
        return BasisTransformPlan(
            tuple(axis.nodes for axis in axes_value),
            tuple(axis.quadrature_weights for axis in axes_value),
            tuple(axis.periodic for axis in axes_value),
            self.bases,
            self.n_modes,
        )

    def __call__(
        self,
        values: Array,
        axes: Sequence[OperatorAxis],
        /,
        *,
        plan: BasisTransformPlan | None = None,
    ) -> Array:
        array = jnp.asarray(values)
        axes_value = tuple(axes)
        ndim = len(self.n_modes)
        if len(axes_value) != ndim:
            raise ValueError(
                f"Expected {ndim} OperatorAxis values, got {len(axes_value)}."
            )
        if array.ndim < ndim + 1 or int(array.shape[-1]) != self.in_channels:
            raise ValueError(
                "BasisSpectralConvND expects (..., spatial..., in_channels) input."
            )
        sample_shape = tuple(int(size) for size in array.shape[-ndim - 1 : -1])
        if sample_shape != tuple(axis.size for axis in axes_value):
            raise ValueError(
                f"Input spatial shape {sample_shape} does not match axis sizes "
                f"{tuple(axis.size for axis in axes_value)}."
            )
        spatial_start = array.ndim - ndim - 1
        transform = self.plan(axes_value) if plan is None else plan
        if (
            transform.sample_shape != sample_shape
            or transform.bases != self.bases
            or transform.n_modes != self.n_modes
        ):
            raise ValueError("Basis transform plan does not match this layer or grid.")
        coefficients = array
        for index, analysis in enumerate(transform.analysis_matrices):
            coefficients = _apply_axis_matrix(
                coefficients,
                analysis,
                spatial_start + index,
            )
        letters = "abcdefghijklmnopqrstuvwxyz"[:ndim]
        coefficients = oe.contract(
            f"...{letters}i,io{letters}->...{letters}o",
            coefficients,
            self.weight,
        )
        output = coefficients
        for index in range(ndim - 1, -1, -1):
            output = _apply_axis_matrix(
                output,
                transform.synthesis_matrices[index],
                spatial_start + index,
            )
        return output


__all__ = ["BasisSpectralConvND", "BasisTransformPlan", "SpectralBasis"]
