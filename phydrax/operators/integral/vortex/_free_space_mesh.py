#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


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
    shape: tuple[int, ...] = eqx.field(static=True)
    lower: Array
    upper: Array
    spacing: Array
    padded_shape: tuple[int, ...] = eqx.field(static=True)
    kernel_coefficients: Array
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], lower: ArrayLike, upper: ArrayLike, /):
        shape_ = tuple(int(value) for value in shape)
        lower_, upper_ = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        dimension = len(shape_)
        if (
            dimension not in (2, 3)
            or lower_.shape != (dimension,)
            or upper_.shape != lower_.shape
            or any(value < 2 for value in shape_)
            or np.any(upper_ <= lower_)
        ):
            raise ValueError("Free-space vortex FFT shape/bounds are invalid.")
        spacing = (upper_ - lower_) / np.asarray(shape_)
        padded = tuple(2 * value for value in shape_)
        coordinates = []
        for axis, count in enumerate(padded):
            index = np.arange(count)
            signed = np.where(index <= count // 2, index, index - count)
            coordinates.append(signed * spacing[axis])
        mesh = np.meshgrid(*coordinates, indexing="ij")
        displacement = np.stack(mesh, axis=-1)
        squared = np.sum(displacement**2, axis=-1)
        safe = np.where(squared > 0.0, squared, 1.0)
        if dimension == 2:
            kernel = np.stack((-displacement[..., 1], displacement[..., 0]), axis=-1) / (
                2.0 * np.pi * safe[..., None]
            )
        else:
            kernel = displacement / (
                4.0 * np.pi * safe[..., None] * np.sqrt(safe)[..., None]
            )
        kernel[squared == 0.0] = 0.0
        kernel_coefficients = np.fft.fftn(kernel, axes=tuple(range(dimension)))
        self.shape, self.lower, self.upper, self.spacing, self.padded_shape = (
            shape_,
            jnp.asarray(lower_),
            jnp.asarray(upper_),
            jnp.asarray(spacing),
            padded,
        )
        self.kernel_coefficients, self.dimension = (
            jnp.asarray(kernel_coefficients),
            dimension,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "free-space-vortex-fft",
                "shape": shape_,
                "lower": lower_.tolist(),
                "upper": upper_.tolist(),
            }
        )

    def evaluate(
        self, vorticity_density: ArrayLike, /, *, velocity_gradient: bool = False
    ) -> FreeSpaceVortexFFTResult:
        omega = jnp.asarray(vorticity_density)
        expected = self.shape if self.dimension == 2 else self.shape + (3,)
        if omega.shape != expected:
            raise ValueError("Free-space vorticity density shape is incompatible.")
        padding = tuple((0, count) for count in self.shape) + (
            () if self.dimension == 2 else ((0, 0),)
        )
        padded = jnp.pad(omega, padding)
        coefficients = jnp.fft.fftn(padded, axes=tuple(range(self.dimension)))
        if self.dimension == 2:
            velocity_coefficients = coefficients[..., None] * self.kernel_coefficients
        else:
            # Γ × r = -r × Γ for the convolution ordering used here.
            velocity_coefficients = -jnp.cross(
                self.kernel_coefficients, coefficients, axis=-1
            )
        velocity_padded = jnp.fft.ifftn(
            velocity_coefficients, axes=tuple(range(self.dimension))
        )
        slices = tuple(slice(0, count) for count in self.shape)
        velocity = velocity_padded[slices].real
        gradient = None
        if velocity_gradient:
            axes = tuple(
                jnp.fft.fftfreq(count, d=float(self.spacing[axis])) * 2.0 * jnp.pi
                for axis, count in enumerate(self.padded_shape)
            )
            mesh = jnp.meshgrid(*axes, indexing="ij")
            gradient_components = []
            for axis in range(self.dimension):
                derivative = jnp.fft.ifftn(
                    1j * mesh[axis][..., None] * velocity_coefficients,
                    axes=tuple(range(self.dimension)),
                )[slices].real
                gradient_components.append(derivative)
            gradient = jnp.stack(tuple(gradient_components), axis=-1)
        cell_measure = jnp.prod(self.spacing)
        circulation = jnp.sum(omega, axis=tuple(range(self.dimension))) * cell_measure
        boundary_mask = jnp.zeros(self.shape, dtype=bool)
        for axis in range(self.dimension):
            lower_index = [slice(None)] * self.dimension
            upper_index = [slice(None)] * self.dimension
            lower_index[axis], upper_index[axis] = 0, self.shape[axis] - 1
            boundary_mask = boundary_mask.at[tuple(lower_index)].set(True)
            boundary_mask = boundary_mask.at[tuple(upper_index)].set(True)
        magnitude = (
            jnp.linalg.norm(omega, axis=-1) if self.dimension == 3 else jnp.abs(omega)
        )
        boundary_fraction = jnp.sum(
            jnp.where(boundary_mask, magnitude, 0.0)
        ) / jnp.maximum(jnp.sum(magnitude), 1.0)
        leakage = jnp.max(jnp.abs(velocity_padded.imag))
        finite = jnp.all(jnp.isfinite(velocity)) & (
            gradient is None or jnp.all(jnp.isfinite(gradient))
        )
        successful = finite & (boundary_fraction <= 1.0e-8)
        return FreeSpaceVortexFFTResult(
            velocity,
            gradient,
            self.padded_shape,
            circulation,
            boundary_fraction,
            leakage,
            finite,
            successful,
            self.plan_id,
        )


__all__ = ["FreeSpaceVortexFFTPlan", "FreeSpaceVortexFFTResult"]
