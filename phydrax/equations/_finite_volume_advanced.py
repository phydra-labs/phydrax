#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._hyperbolic_systems import AbstractAdmissibleSystem, ShallowWaterSystem


class InterfacialPhaseChangeEvaluation(StrictModule):
    interface_temperature: Array
    mass_rate: Array
    limited_mass_rate: Array
    source_factor: Array
    phase_mass_defect: Array
    energy_defect: Array
    active: Array


class StefanPhaseChangePlan(StrictModule, NonTrainableState):
    saturation_law: Callable[[Array], ArrayLike]
    latent_heat: float = eqx.field(static=True)
    interfacial_velocity: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        saturation_law,
        latent_heat: float,
        /,
        *,
        interfacial_velocity: str = "mass-weighted",
    ):
        if not callable(saturation_law):
            raise TypeError("saturation_law must be callable.")
        latent = float(latent_heat)
        if (
            not np.isfinite(latent)
            or latent <= 0
            or interfacial_velocity not in ("mass-weighted", "phase-0", "phase-1")
        ):
            raise ValueError("Stefan latent heat/interfacial velocity is invalid.")
        self.saturation_law = saturation_law
        self.latent_heat = latent
        self.interfacial_velocity = interfacial_velocity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stefan-phase-change",
                "latent_heat": latent,
                "interfacial_velocity": interfacial_velocity,
            }
        )

    def evaluate(
        self,
        pressure,
        heat_flux_0,
        heat_flux_1,
        available_mass_0,
        available_mass_1,
        step_size,
        interface_area,
        /,
    ):
        temperature = jnp.asarray(self.saturation_law(jnp.asarray(pressure)))
        mass_rate = (
            jnp.asarray(heat_flux_0) - jnp.asarray(heat_flux_1)
        ) / self.latent_heat
        transfer = jnp.abs(
            mass_rate * jnp.asarray(interface_area) * jnp.asarray(step_size)
        )
        available = jnp.where(
            mass_rate >= 0, jnp.asarray(available_mass_0), jnp.asarray(available_mass_1)
        )
        factor = jnp.minimum(1.0, available / jnp.where(transfer > 0, transfer, 1.0))
        limited = factor * mass_rate
        return InterfacialPhaseChangeEvaluation(
            temperature,
            mass_rate,
            limited,
            factor,
            jnp.zeros_like(limited),
            jnp.zeros_like(limited),
            jnp.asarray(interface_area) > 0,
        )


class HydrostaticLayerCoupling(StrictModule, NonTrainableState):
    densities: Array
    gravity: float = eqx.field(static=True)
    energy_hessian: Array
    coupling_id: str = eqx.field(static=True)

    @classmethod
    def from_densities(cls, densities: ArrayLike, gravity: float = 9.81, /):
        rho, gravity_ = np.asarray(densities, dtype=float), float(gravity)
        if (
            rho.ndim != 1
            or rho.size == 0
            or np.any(~np.isfinite(rho))
            or np.any(rho <= 0)
        ):
            raise ValueError("Layer densities must be a positive finite vector.")
        if np.any(np.diff(rho) > 0):
            raise ValueError("Layer densities must be stably ordered bottom-to-top.")
        if not np.isfinite(gravity_) or gravity_ <= 0:
            raise ValueError("Layer gravity must be positive and finite.")
        hessian = gravity_ * np.minimum(rho[:, None], rho[None, :])
        if np.linalg.eigvalsh(hessian).min() < -1e-12 * np.linalg.norm(hessian):
            raise ValueError(
                "Layer hydrostatic energy Hessian is not positive semidefinite."
            )
        return cls(
            jnp.asarray(rho),
            gravity_,
            jnp.asarray(hessian),
            canonical_fingerprint(
                {
                    "kind": "hydrostatic-layer-coupling",
                    "densities": rho.tolist(),
                    "gravity": gravity_,
                }
            ),
        )

    @property
    def layer_count(self) -> int:
        return int(self.densities.size)

    def potential_energy(self, depths: ArrayLike, /) -> Array:
        values = jnp.asarray(depths)
        return 0.5 * ein.contract(
            "...i,ij,...j->...", values, self.energy_hessian, values, backend="jax"
        )

    def potential_gradient(self, depths: ArrayLike, /) -> Array:
        return ein.contract(
            "ij,...j->...i", self.energy_hessian, jnp.asarray(depths), backend="jax"
        )


class MultilayerShallowWaterSystem(AbstractAdmissibleSystem):
    coupling: HydrostaticLayerCoupling

    def __init__(self, coupling: HydrostaticLayerCoupling, dimension: int = 1, /):
        if not isinstance(coupling, HydrostaticLayerCoupling) or int(dimension) not in (
            1,
            2,
        ):
            raise TypeError(
                "Multilayer shallow water requires coupling and dimension 1/2."
            )
        self.coupling, self.dimension = coupling, int(dimension)
        k = coupling.layer_count
        self.component_names = (
            *(f"depth_{i}" for i in range(k)),
            *(f"discharge_{i}_{d}" for i in range(k) for d in range(self.dimension)),
        )
        self.system_id = canonical_fingerprint(
            {
                "kind": "multilayer-shallow-water",
                "coupling": coupling.coupling_id,
                "dimension": self.dimension,
            }
        )

    def split(self, state: ArrayLike, /) -> tuple[Array, Array]:
        value, k = jnp.asarray(state), self.coupling.layer_count
        return value[..., :k], value[..., k:].reshape(
            value.shape[:-1] + (k, self.dimension)
        )

    def conserved_to_primitive(self, state: Array, /) -> Array:
        h, q = self.split(state)
        velocity = q / jnp.where(h[..., None] > 0, h[..., None], 1)
        velocity = jnp.where(h[..., None] > 0, velocity, 0)
        return jnp.concatenate(
            (h, velocity.reshape(velocity.shape[:-2] + (-1,))),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        k = self.coupling.layer_count
        h = value[..., :k]
        velocity = value[..., k:].reshape(value.shape[:-1] + (k, self.dimension))
        q = h[..., None] * velocity
        return jnp.concatenate((h, q.reshape(q.shape[:-2] + (-1,))), axis=-1)

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        del args
        h, q = self.split(state)
        safe = jnp.where(h > 0, h, 1)
        velocity = jnp.where(h[..., None] > 0, q / safe[..., None], 0)
        mass = q[..., int(axis)]
        momentum = q * velocity[..., int(axis), None]
        pressure = (
            0.5 * jnp.diag(self.coupling.energy_hessian) * h * h / self.coupling.densities
        )
        momentum = momentum.at[..., :, int(axis)].add(pressure)
        return jnp.concatenate((mass, momentum.reshape(momentum.shape[:-2] + (-1,))), -1)

    def signal_bounds(self, left, right, axis, args=None, /):
        del args
        hl, ql = self.split(left)
        hr, qr = self.split(right)
        ul = ql[..., int(axis)] / jnp.where(hl > 0, hl, 1)
        ur = qr[..., int(axis)] / jnp.where(hr > 0, hr, 1)
        cl = jnp.sqrt(self.coupling.gravity * jnp.maximum(jnp.sum(hl, axis=-1), 0))
        cr = jnp.sqrt(self.coupling.gravity * jnp.maximum(jnp.sum(hr, axis=-1), 0))
        lower = jnp.minimum(
            jnp.min(ul, axis=-1) - cl,
            jnp.min(ur, axis=-1) - cr,
        )
        upper = jnp.maximum(
            jnp.max(ul, axis=-1) + cl,
            jnp.max(ur, axis=-1) + cr,
        )
        return lower, upper

    def max_wave_speed(self, left, right, axis, args=None, /):
        lower, upper = self.signal_bounds(left, right, axis, args)
        return jnp.maximum(jnp.abs(lower), jnp.abs(upper))

    def normal_signal_bounds(self, left, right, normal, args=None, /):
        del args
        normal_ = jnp.asarray(normal)
        if normal_.ndim == 0 or normal_.shape[-1] != self.dimension:
            raise ValueError("Normal vectors must match the multilayer system dimension.")
        hl, ql = self.split(left)
        hr, qr = self.split(right)
        ul = ql / jnp.where(hl[..., None] > 0, hl[..., None], 1)
        ur = qr / jnp.where(hr[..., None] > 0, hr[..., None], 1)
        unl = jnp.sum(ul * normal_[..., None, :], axis=-1)
        unr = jnp.sum(ur * normal_[..., None, :], axis=-1)
        cl = jnp.sqrt(self.coupling.gravity * jnp.maximum(jnp.sum(hl, axis=-1), 0))
        cr = jnp.sqrt(self.coupling.gravity * jnp.maximum(jnp.sum(hr, axis=-1), 0))
        lower = jnp.minimum(
            jnp.min(unl, axis=-1) - cl,
            jnp.min(unr, axis=-1) - cr,
        )
        upper = jnp.maximum(
            jnp.max(unl, axis=-1) + cl,
            jnp.max(unr, axis=-1) + cr,
        )
        return lower, upper

    def admissible(self, state: Array, /) -> Array:
        h, q = self.split(state)
        return (
            jnp.all(jnp.isfinite(state), -1)
            & jnp.all(h >= 0, -1)
            & jnp.all((h > 0) | jnp.all(q == 0, -1), -1)
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        h, q = self.split(state)
        q = q.at[..., :, int(axis)].multiply(-1)
        return jnp.concatenate((h, q.reshape(q.shape[:-2] + (-1,))), -1)


class BedloadSedimentPlan(StrictModule, NonTrainableState):
    relative_density: float = eqx.field(static=True)
    grain_diameter: float = eqx.field(static=True)
    critical_shields: float = eqx.field(static=True)
    porosity: float = eqx.field(static=True)
    gravity: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        relative_density: float,
        grain_diameter: float,
        critical_shields: float,
        porosity: float,
        /,
        *,
        gravity: float = 9.81,
    ):
        values = tuple(
            map(
                float,
                (
                    relative_density,
                    grain_diameter,
                    critical_shields,
                    porosity,
                    gravity,
                ),
            )
        )
        s, diameter, critical, porosity_, gravity_ = values
        if (
            not all(np.isfinite(values))
            or s <= 1
            or diameter <= 0
            or critical < 0
            or not 0 <= porosity_ < 1
            or gravity_ <= 0
        ):
            raise ValueError("Bedload parameters are outside their physical domain.")
        (
            self.relative_density,
            self.grain_diameter,
            self.critical_shields,
            self.porosity,
            self.gravity,
        ) = values
        self.plan_id = canonical_fingerprint(
            {"kind": "meyer-peter-mueller-bedload", "parameters": values}
        )

    def bedload(self, depth: ArrayLike, discharge: ArrayLike, /) -> Array:
        h, q = jnp.asarray(depth), jnp.asarray(discharge)
        velocity = q / jnp.where(h[..., None] > 0, h[..., None], 1)
        speed = jnp.sqrt(jnp.sum(velocity * velocity, -1))
        shields = (
            speed
            * speed
            / ((self.relative_density - 1) * self.gravity * self.grain_diameter)
        )
        magnitude = (
            8
            * jnp.sqrt(
                (self.relative_density - 1) * self.gravity * self.grain_diameter**3
            )
            * jnp.maximum(shields - self.critical_shields, 0) ** 1.5
        )
        return (
            magnitude[..., None]
            * velocity
            / jnp.where(speed[..., None] > 0, speed[..., None], 1)
        )


class ShallowWaterExnerSystem(AbstractAdmissibleSystem):
    base: ShallowWaterSystem
    sediment: BedloadSedimentPlan
    minimum_bed: float = eqx.field(static=True)
    maximum_bed: float = eqx.field(static=True)

    def __init__(
        self,
        base: ShallowWaterSystem,
        sediment: BedloadSedimentPlan,
        /,
        *,
        bed_bounds: tuple[float, float],
    ):
        if not isinstance(base, ShallowWaterSystem) or not isinstance(
            sediment, BedloadSedimentPlan
        ):
            raise TypeError("Exner system requires shallow water and bedload plans.")
        lower, upper = map(float, bed_bounds)
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError("Exner bed bounds are invalid.")
        self.base, self.sediment, self.minimum_bed, self.maximum_bed = (
            base,
            sediment,
            lower,
            upper,
        )
        self.dimension = base.dimension
        self.component_names = (*base.component_names, "bed_elevation")
        self.system_id = canonical_fingerprint(
            {
                "kind": "shallow-water-exner",
                "base": base.system_id,
                "sediment": sediment.plan_id,
                "bed_bounds": (lower, upper),
            }
        )

    def conserved_to_primitive(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return jnp.concatenate(
            (self.base.conserved_to_primitive(value[..., :-1]), value[..., -1:]),
            axis=-1,
        )

    def primitive_to_conserved(self, primitive: Array, /) -> Array:
        value = jnp.asarray(primitive)
        return jnp.concatenate(
            (self.base.primitive_to_conserved(value[..., :-1]), value[..., -1:]),
            axis=-1,
        )

    def signal_bounds(self, left, right, axis, args=None, /):
        return self.base.signal_bounds(
            jnp.asarray(left)[..., :-1],
            jnp.asarray(right)[..., :-1],
            axis,
            args,
        )

    def normal_signal_bounds(self, left, right, normal, args=None, /):
        return self.base.normal_signal_bounds(
            jnp.asarray(left)[..., :-1],
            jnp.asarray(right)[..., :-1],
            normal,
            args,
        )

    def physical_flux(self, state: Array, axis: int, args: Any = None, /) -> Array:
        water = jnp.asarray(state)[..., :-1]
        water_flux = self.base.physical_flux(water, axis, args)
        bedload = self.sediment.bedload(water[..., 0], water[..., 1:])
        return jnp.concatenate(
            (
                water_flux,
                (bedload[..., int(axis)] / (1 - self.sediment.porosity))[..., None],
            ),
            -1,
        )

    def max_wave_speed(self, left, right, axis, args=None, /):
        water = self.base.max_wave_speed(
            jnp.asarray(left)[..., :-1], jnp.asarray(right)[..., :-1], axis, args
        )
        return water

    def admissible(self, state: Array, /) -> Array:
        value = jnp.asarray(state)
        return (
            self.base.admissible(value[..., :-1])
            & (value[..., -1] >= self.minimum_bed)
            & (value[..., -1] <= self.maximum_bed)
        )

    def reflect_state(self, state: Array, axis: int, /) -> Array:
        value = jnp.asarray(state)
        return jnp.concatenate(
            (self.base.reflect_state(value[..., :-1], axis), value[..., -1:]), -1
        )


__all__ = [
    "BedloadSedimentPlan",
    "HydrostaticLayerCoupling",
    "InterfacialPhaseChangeEvaluation",
    "MultilayerShallowWaterSystem",
    "ShallowWaterExnerSystem",
    "StefanPhaseChangePlan",
]
