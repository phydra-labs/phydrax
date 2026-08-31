#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LagrangianInitialConditionResult(StrictModule):
    density_contrast: Array
    displacement: Array
    positions: Array
    power_spectrum: Array


class LagrangianPerturbationInitialConditionPlan(StrictModule, NonTrainableState):
    shape: tuple[int, ...] = eqx.field(static=True)
    box_size: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, shape: tuple[int, ...], box_size: tuple[float, ...], /):
        shape_ = tuple(int(value) for value in shape)
        box = tuple(float(value) for value in box_size)
        if (
            len(shape_) not in (1, 2, 3)
            or len(shape_) != len(box)
            or any(value < 2 for value in shape_)
            or any(value <= 0.0 for value in box)
        ):
            raise ValueError("Lagrangian initial-condition domain is invalid.")
        self.shape = shape_
        self.box_size = box
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lpt-initial-conditions",
                "shape": list(shape_),
                "box_size": list(box),
            }
        )

    def realize(
        self,
        white_noise: ArrayLike,
        power_spectrum: ArrayLike,
        /,
        *,
        growth_factor: ArrayLike = 1.0,
    ) -> LagrangianInitialConditionResult:
        noise = jnp.asarray(white_noise)
        spectrum = jnp.asarray(power_spectrum, dtype=noise.dtype)
        if noise.shape != self.shape or spectrum.shape != self.shape:
            raise ValueError("White noise and power spectrum must match the LPT grid.")
        modes = jnp.fft.fftn(noise) * jnp.sqrt(jnp.maximum(spectrum, 0.0))
        frequencies = tuple(
            2.0 * jnp.pi * jnp.fft.fftfreq(count, length / count)
            for count, length in zip(self.shape, self.box_size, strict=True)
        )
        wavevectors = jnp.meshgrid(*frequencies, indexing="ij")
        squared = sum(component**2 for component in wavevectors)
        safe = jnp.where(squared > 0.0, squared, 1.0)
        density = jnp.fft.ifftn(modes).real
        displacement = jnp.stack(
            tuple(
                jnp.fft.ifftn(1j * component * modes / safe).real
                for component in wavevectors
            ),
            axis=-1,
        )
        displacement = jnp.asarray(growth_factor, dtype=noise.dtype) * displacement
        axes = tuple(
            (jnp.arange(count, dtype=noise.dtype) + 0.5) * length / count
            for count, length in zip(self.shape, self.box_size, strict=True)
        )
        lattice = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        positions = jnp.mod(lattice + displacement, jnp.asarray(self.box_size))
        recovered = jnp.abs(modes) ** 2 / np.prod(self.shape)
        return LagrangianInitialConditionResult(
            density_contrast=density,
            displacement=displacement,
            positions=positions,
            power_spectrum=recovered,
        )


__all__ = [
    "LagrangianInitialConditionResult",
    "LagrangianPerturbationInitialConditionPlan",
]
