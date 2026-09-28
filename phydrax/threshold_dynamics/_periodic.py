#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact periodic heat kernel for threshold dynamics on uniform grids.

The periodic grid is an all-Fourier `TensorSpectralDiscretization`, the canonical
periodic transform owner. The heat semigroup ``G(tau) = exp(tau * Laplacian)`` is
its exact Fourier multiplier ``exp(-tau |k|^2)`` on the trigonometric interpolant,
so the kernel is exact on the grid space, self-adjoint and positive definite in
the uniform site measure. One forward transform per label field serves every
kernel time; the combination ``sum_k G(tau_k) u @ M_k`` is formed in modal space
before a single synthesis.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import DTypeLike

from .._fingerprint import canonical_fingerprint
from .._validation import finite_real_scalar, positive_integer
from ..discretization import AxisDomain
from ..discretization.spectral import (
    FourierBasisPlan,
    SpectralPrecisionPolicy,
    TensorSpectralDiscretization,
    TensorSpectralPlan,
)
from ._contracts import AbstractThresholdHeatKernel, HeatActionEvidence


def _grid(
    shape: Sequence[int], lengths: Sequence[float], /
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    shape_ = tuple(positive_integer(value, "shape") for value in shape)
    lengths_ = tuple(finite_real_scalar(value, "lengths") for value in lengths)
    if not 1 <= len(shape_) <= 3:
        raise ValueError("Periodic threshold grids have one to three axes.")
    if len(lengths_) != len(shape_):
        raise ValueError("lengths must declare one period per grid axis.")
    if any(count < 4 for count in shape_):
        raise ValueError("Periodic threshold grids need at least four sites per axis.")
    if any(length <= 0.0 for length in lengths_):
        raise ValueError("Grid periods must be positive.")
    return shape_, lengths_


class PeriodicGridHeatKernel(AbstractThresholdHeatKernel):
    """Exact heat kernel on a periodic uniform grid with cell-centered sites.

    Sites are the ``shape`` grid points of the box ``prod [0, lengths[a])``; every
    site carries measure ``prod(lengths / shape)``.
    """

    spectral: TensorSpectralDiscretization
    laplacian_eigenvalues: Array
    site_shape: tuple[int, ...] = eqx.field(static=True)
    lengths: tuple[float, ...] = eqx.field(static=True)
    resolution_length: float = eqx.field(static=True)
    equal_site_measure: bool = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    site_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: Sequence[int],
        lengths: Sequence[float],
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        shape_, lengths_ = _grid(shape, lengths)
        working = jnp.dtype(dtype)
        if working not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise ValueError("dtype must be float32 or float64.")
        spectral = TensorSpectralPlan(
            tuple(FourierBasisPlan(count) for count in shape_),
            field_name="label-indicator",
            precision=SpectralPrecisionPolicy(working),
        ).prepare(tuple(AxisDomain.periodic(0.0, length) for length in lengths_))
        self.spectral = spectral
        self.laplacian_eigenvalues = spectral.laplacian_eigenvalues().astype(working)
        self.site_shape = shape_
        self.lengths = lengths_
        self.resolution_length = max(
            length / count for length, count in zip(lengths_, shape_, strict=True)
        )
        self.equal_site_measure = True
        self.site_id = canonical_fingerprint(
            {
                "kind": "threshold-periodic-grid-sites",
                "shape": list(shape_),
                "lengths": list(lengths_),
            }
        )
        self.route_id = canonical_fingerprint(
            {
                "kind": "periodic-grid-heat-kernel",
                "spectral": spectral.prepared_id,
                "sites": self.site_id,
                "dtype": working.name,
            }
        )

    @property
    def dtype(self) -> jnp.dtype:
        return self.laplacian_eigenvalues.dtype

    @property
    def site_measure(self) -> float:
        return math.prod(
            length / count
            for length, count in zip(self.lengths, self.site_shape, strict=True)
        )

    def site_measures(self) -> Array:
        return jnp.full(self.site_shape, self.site_measure, dtype=self.dtype)

    def combine(
        self, fields: Array, times: Array, coefficients: Array, /
    ) -> tuple[Array, HeatActionEvidence]:
        dimension = len(self.site_shape)
        modal = self.spectral.project(fields.astype(self.dtype))
        values = coefficients.astype(self.dtype)
        if values.ndim not in (1, 3):
            raise ValueError(
                "coefficients must have shape (kernels,) or (kernels, labels, labels)."
            )
        structured = (
            jnp.sum(modal, axis=-1, keepdims=True) - modal
            if values.ndim == 1
            else None
        )
        combined = jnp.zeros_like(modal)
        for index in range(times.shape[0]):
            multiplier = jnp.exp(-times[index] * self.laplacian_eigenvalues)
            weighted = (
                modal @ values[index]
                if structured is None
                else values[index] * structured
            )
            combined = combined + multiplier[..., None] * weighted
        potentials = self.spectral.reconstruct(combined, real_output=True)
        finite = jnp.all(jnp.isfinite(potentials))
        return (
            potentials.reshape(self.site_shape + fields.shape[dimension:]),
            self._evidence(finite),
        )

    def smooth(self, fields: Array, time: Array, /) -> tuple[Array, HeatActionEvidence]:
        dimension = len(self.site_shape)
        modal = self.spectral.project(fields.astype(self.dtype))
        multiplier = jnp.exp(-time.astype(self.dtype) * self.laplacian_eigenvalues)
        smoothed = self.spectral.reconstruct(
            multiplier[..., None] * modal, real_output=True
        )
        finite = jnp.all(jnp.isfinite(smoothed))
        return (
            smoothed.reshape(self.site_shape + fields.shape[dimension:]),
            self._evidence(finite),
        )

    def _evidence(self, finite: Array, /) -> HeatActionEvidence:
        zero = jnp.zeros((), jnp.int32)
        return HeatActionEvidence(
            successful=finite,
            error_estimate=jnp.zeros((), self.dtype),
            native_status=zero,
            converged=finite,
            derivative_valid=finite,
            iterations=zero,
            setup_matvec_count=zero,
            action_matvec_count=zero,
            transpose_matvec_count=zero,
            breakdown_status=zero,
            numeric_version=zero,
            method="exact-periodic-fourier-multiplier",
            exact=True,
            retained_storage_bytes=0,
            workspace_bytes=0,
            operator_id=None,
            prepared_id=None,
        )


__all__ = ["PeriodicGridHeatKernel"]
