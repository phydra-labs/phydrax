#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization import TensorGridPlan, UniformCellAxisSpec
from .._free_space_convolution import FreeSpaceConvolutionPlan


class FreeSpaceVortexFFTResult(StrictModule):
    velocity: Array
    velocity_gradient: Array | None
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    circulation: Array
    boundary_vorticity_fraction: Array
    imaginary_leakage: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class FreeSpaceVortexFFTPlan(StrictModule, NonTrainableState):
    """Open-boundary Biot–Savart velocity of a cell-centered vorticity density.

    The velocity is the Hockney doubled-grid convolution of the vorticity
    density with the point-sampled Biot–Savart kernel
    (:class:`FreeSpaceConvolutionPlan` with ``"biot-savart"``); the cell measure
    is applied, so ``vorticity_density`` is a density and ``circulation`` is
    its integral. ``velocity_gradient=True`` prepares the analytic kernel
    derivatives so ``velocity_gradient[..., i, l] = ∂_l u_i`` is convolved,
    not finite-differenced.
    """

    shape: tuple[int, ...] = eqx.field(static=True)
    lower: Array
    upper: Array
    spacing: Array
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    convolution: FreeSpaceConvolutionPlan
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shape: tuple[int, ...],
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        velocity_gradient: bool = False,
    ) -> None:
        shape_ = tuple(shape)
        lower_, upper_ = (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )
        dimension = len(shape_)
        if (
            dimension not in (2, 3)
            or lower_.shape != (dimension,)
            or upper_.shape != lower_.shape
            or any(value < 2 for value in shape_)
            or np.any(upper_ <= lower_)
        ):
            raise ValueError("Free-space vortex FFT shape/bounds are invalid.")
        if not isinstance(velocity_gradient, bool):
            raise TypeError("velocity_gradient must be a bool.")
        grid = TensorGridPlan(
            tuple(UniformCellAxisSpec(count) for count in shape_)
        ).prepare(np.stack((lower_, upper_)))
        convolution = FreeSpaceConvolutionPlan(
            "biot-savart", grid, gradient=velocity_gradient
        )
        self.shape = shape_
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.spacing = convolution.spacing
        self.padded_shape = convolution.padded_shape
        self.convolution = convolution
        self.dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-space-vortex-fft",
                "shape": shape_,
                "lower": lower_.tolist(),
                "upper": upper_.tolist(),
                "velocity_gradient": velocity_gradient,
            }
        )

    def evaluate(self, vorticity_density: ArrayLike, /) -> FreeSpaceVortexFFTResult:
        omega = jnp.asarray(vorticity_density)
        if omega.shape != self.convolution.source_shape:
            raise ValueError("Free-space vorticity density shape is incompatible.")
        convolved = self.convolution.convolve(omega)
        cell_measure = jnp.prod(self.spacing)
        circulation = jnp.sum(omega, axis=tuple(range(self.dimension))) * cell_measure
        boundary_mask = jnp.zeros(self.shape, dtype=jnp.bool_)
        for axis in range(self.dimension):
            lower_index: list[slice | int] = [slice(None)] * self.dimension
            upper_index: list[slice | int] = [slice(None)] * self.dimension
            lower_index[axis], upper_index[axis] = 0, self.shape[axis] - 1
            boundary_mask = boundary_mask.at[tuple(lower_index)].set(True)
            boundary_mask = boundary_mask.at[tuple(upper_index)].set(True)
        magnitude = (
            jnp.linalg.norm(omega, axis=-1) if self.dimension == 3 else jnp.abs(omega)
        )
        boundary_fraction = jnp.sum(
            jnp.where(boundary_mask, magnitude, 0.0)
        ) / jnp.maximum(jnp.sum(magnitude), 1.0)
        successful = convolved.finite & (boundary_fraction <= 1.0e-8)
        return FreeSpaceVortexFFTResult(
            convolved.field,
            convolved.gradient,
            self.padded_shape,
            circulation,
            boundary_fraction,
            convolved.imaginary_leakage,
            convolved.finite,
            successful,
            self.plan_id,
        )


__all__ = ["FreeSpaceVortexFFTPlan", "FreeSpaceVortexFFTResult"]
