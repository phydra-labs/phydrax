#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import ParticleDiscretization
from ._background import FLRWBackground
from ._particles import CosmologicalParticleState
from ._products import LagrangianGrowthHistory, MatterPowerTable
from ._scales import CODE_COSMOLOGY_SCALE, CosmologyScaleContract


LagrangianDealiasing = Literal["none", "three_halves"]


class LagrangianInitialConditionResult(StrictModule):
    density_contrast: Array
    first_order_displacement: Array
    second_order_displacement: Array
    state: CosmologicalParticleState
    power_spectrum: Array
    successful: Array

    @property
    def displacement(self) -> Array:
        return self.first_order_displacement + self.second_order_displacement

    @property
    def positions(self) -> Array:
        return self.state.positions

    @property
    def canonical_momenta(self) -> Array:
        return self.state.canonical_momenta


def _wavevectors(
    shape: tuple[int, ...], box_size: tuple[float, ...], dtype, /
) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
    components = tuple(
        2.0 * jnp.pi * jnp.fft.fftfreq(count, length / count, dtype=dtype)
        for count, length in zip(shape, box_size, strict=True)
    )
    grids = tuple(jnp.meshgrid(*components, indexing="ij"))
    gradients = []
    for axis, (count, length, component) in enumerate(
        zip(shape, box_size, grids, strict=True)
    ):
        if count % 2 == 0:
            nyquist = jnp.asarray(jnp.pi * count / length, dtype=dtype)
            component = jnp.where(jnp.abs(component) == nyquist, 0.0, component)
        gradients.append(component)
    return grids, tuple(gradients)


def _centered_resize(modes: Array, shape: tuple[int, ...], /) -> Array:
    axes = tuple(range(modes.ndim))
    shifted = jnp.fft.fftshift(modes, axes=axes)
    slices = []
    padding = []
    enlarging = all(target >= source for target, source in zip(shape, modes.shape))
    if enlarging:
        for target, source in zip(shape, modes.shape, strict=True):
            before = (target - source) // 2
            after = target - source - before
            padding.append((before, after))
        resized = jnp.pad(shifted, tuple(padding))
        resized *= prod(shape) / prod(modes.shape)
    else:
        for target, source in zip(shape, modes.shape, strict=True):
            start = (source - target) // 2
            slices.append(slice(start, start + target))
        resized = shifted[tuple(slices)]
        resized *= prod(shape) / prod(modes.shape)
    return jnp.fft.ifftshift(resized, axes=axes)


def _second_order_source(
    first_potential: Array,
    shape: tuple[int, ...],
    box_size: tuple[float, ...],
    /,
    *,
    dealiasing: LagrangianDealiasing,
) -> Array:
    work_shape = (
        shape
        if dealiasing == "none"
        else tuple(max(count + 1, (3 * count + 1) // 2) for count in shape)
    )
    potential = (
        first_potential
        if work_shape == shape
        else _centered_resize(first_potential, work_shape)
    )
    raw, gradient = _wavevectors(work_shape, box_size, potential.real.dtype)
    diagonal = tuple(jnp.fft.ifftn(-(component**2) * potential).real for component in raw)
    source = jnp.zeros(work_shape, dtype=potential.real.dtype)
    for left in range(len(work_shape)):
        for right in range(left + 1, len(work_shape)):
            mixed = jnp.fft.ifftn(-gradient[left] * gradient[right] * potential).real
            source = source + diagonal[left] * diagonal[right] - mixed**2
    source_modes = jnp.fft.fftn(source)
    if work_shape != shape:
        source_modes = _centered_resize(source_modes, shape)
    return source_modes.at[(0,) * len(shape)].set(0.0)


class LagrangianPerturbationInitialConditionPlan(StrictModule, NonTrainableState):
    """State-ready periodic 1LPT or 2LPT from tabulated linear P_delta_delta."""

    particles: ParticleDiscretization
    shape: tuple[int, ...] = eqx.field(static=True)
    box_size: tuple[float, ...] = eqx.field(static=True)
    scale: CosmologyScaleContract
    order: int = eqx.field(static=True)
    dealiasing: LagrangianDealiasing = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particles: ParticleDiscretization,
        shape: tuple[int, ...],
        box_size: tuple[float, ...],
        /,
        *,
        order: int = 1,
        dealiasing: LagrangianDealiasing = "none",
        scale: CosmologyScaleContract = CODE_COSMOLOGY_SCALE,
    ):
        shape_ = tuple(int(value) for value in shape)
        box = tuple(float(value) for value in box_size)
        order_ = int(order)
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        if not isinstance(scale, CosmologyScaleContract):
            raise TypeError("scale must be a CosmologyScaleContract.")
        if (
            len(shape_) not in (1, 2, 3)
            or len(shape_) != len(box)
            or len(shape_) != particles.ambient_dimension
            or prod(shape_) != particles.capacity
            or any(value < 2 for value in shape_)
            or any(not np.isfinite(value) or value <= 0.0 for value in box)
        ):
            raise ValueError("Lagrangian initial-condition domain is invalid.")
        if order_ not in (1, 2):
            raise ValueError("Lagrangian perturbation order must be 1 or 2.")
        if dealiasing not in ("none", "three_halves"):
            raise ValueError("Unknown Lagrangian de-aliasing policy.")
        if order_ == 1 and dealiasing != "none":
            raise ValueError("De-aliasing applies only to second-order LPT.")
        self.particles = particles
        self.shape = shape_
        self.box_size = box
        self.scale = scale
        self.order = order_
        self.dealiasing = dealiasing
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lpt-initial-conditions",
                "particles": particles.prepared_id,
                "shape": list(shape_),
                "box_size": list(box),
                "scale": scale.scale_id,
                "order": order_,
                "dealiasing": dealiasing,
            }
        )

    def realize(
        self,
        background: FLRWBackground,
        growth: LagrangianGrowthHistory,
        power: MatterPowerTable,
        white_noise: ArrayLike,
        initial_scale_factor: ArrayLike,
        /,
    ) -> LagrangianInitialConditionResult:
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be FLRWBackground.")
        if not isinstance(growth, LagrangianGrowthHistory):
            raise TypeError("growth must be LagrangianGrowthHistory.")
        if not isinstance(power, MatterPowerTable):
            raise TypeError("power must be MatterPowerTable.")
        scale_ids = (
            self.scale.scale_id,
            background.scale.scale_id,
            growth.scale.scale_id,
            power.scale.scale_id,
        )
        if len(set(scale_ids)) != 1:
            raise ValueError("LPT scale contracts disagree.")
        if power.descriptor.spatial_dimension != len(self.shape):
            raise ValueError("Matter power and LPT dimensions disagree.")
        if not power.descriptor.is_linear_cold_baryon_auto:
            raise ValueError(
                "LPT requires linear cold-baryon auto-power without shot noise."
            )
        noise = jnp.asarray(white_noise)
        if noise.shape != self.shape:
            raise ValueError("White noise must match the LPT grid.")
        scale_factor = jnp.asarray(initial_scale_factor, dtype=noise.dtype)
        if scale_factor.shape != ():
            raise ValueError("Initial scale factor must be scalar.")
        scale_factor = background.require_flat(scale_factor)
        scale_factor = background.realization.require_compatible(
            growth.realization, scale_factor
        )
        scale_factor = background.realization.require_compatible(
            power.realization, scale_factor
        )
        raw_k, gradient_k = _wavevectors(self.shape, self.box_size, noise.dtype)
        squared = sum(component**2 for component in raw_k)
        magnitude = jnp.sqrt(squared)
        safe_magnitude = jnp.where(
            squared > 0.0,
            magnitude,
            jnp.asarray(power.wavenumbers[0], dtype=magnitude.dtype),
        )
        spectral_power = power.evaluate(safe_magnitude, scale_factor)
        spectral_power = jnp.where(squared > 0.0, spectral_power, 0.0)
        grid_size = prod(self.shape)
        volume = float(prod(self.box_size))
        modes = jnp.fft.fftn(noise) * jnp.sqrt(spectral_power * grid_size / volume)
        modes = modes.at[(0,) * len(self.shape)].set(0.0)
        density = jnp.fft.ifftn(modes).real
        safe_squared = jnp.where(squared > 0.0, squared, 1.0)
        first_potential = -modes / safe_squared
        first = jnp.stack(
            tuple(
                jnp.fft.ifftn(-1j * component * first_potential).real
                for component in gradient_k
            ),
            axis=-1,
        )
        first_growth, first_rate, second_growth, second_rate = growth.evaluate(
            scale_factor
        )
        if self.order == 2:
            source = _second_order_source(
                first_potential,
                self.shape,
                self.box_size,
                dealiasing=self.dealiasing,
            )
            second_potential = -source / safe_squared
            second = jnp.stack(
                tuple(
                    jnp.fft.ifftn(-1j * component * second_potential).real
                    for component in gradient_k
                ),
                axis=-1,
            )
            second *= second_growth / first_growth**2
        else:
            second = jnp.zeros_like(first)
            second_rate = jnp.asarray(0.0, dtype=first_rate.dtype)
        axes = tuple(
            (jnp.arange(count, dtype=noise.dtype) + 0.5) * length / count
            for count, length in zip(self.shape, self.box_size, strict=True)
        )
        lattice = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        displacement = first + second
        positions = jnp.mod(lattice + displacement, jnp.asarray(self.box_size))
        hubble = background.hubble(scale_factor)
        velocity_factor = scale_factor**2 * hubble
        momentum = self.particles.safe_masses[:, None].astype(noise.dtype) * (
            velocity_factor
            * (
                first_rate * first.reshape((-1, len(self.shape)))
                + second_rate * second.reshape((-1, len(self.shape)))
            )
        )
        active = self.particles.active_mask[:, None]
        state = CosmologicalParticleState(
            jnp.where(active, positions.reshape((-1, len(self.shape))), 0.0),
            jnp.where(active, momentum, 0.0),
            scale_factor,
        )
        recovered = jnp.abs(modes) ** 2 * volume / grid_size**2
        successful = (
            jnp.isfinite(scale_factor)
            & (scale_factor > 0.0)
            & jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(state.positions))
            & jnp.all(jnp.isfinite(state.canonical_momenta))
        )
        state = CosmologicalParticleState(
            jnp.where(
                successful, state.positions, lattice.reshape(state.positions.shape)
            ),
            jnp.where(successful, state.canonical_momenta, 0.0),
            scale_factor,
        )
        return LagrangianInitialConditionResult(
            density_contrast=density,
            first_order_displacement=first,
            second_order_displacement=second,
            state=state,
            power_spectrum=recovered,
            successful=successful,
        )


__all__ = [
    "LagrangianDealiasing",
    "LagrangianInitialConditionResult",
    "LagrangianPerturbationInitialConditionPlan",
]
