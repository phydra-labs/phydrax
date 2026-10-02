#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Radial bubble equations composed from gas, liquid and interface laws.

All radial equations share one structure. With the radiated wall quantity
`Q(R, Ṙ, z, t)` (`p_L − p_∞`, the gas pressure `p_g` for the gas-radiation
form or, for Gilmore, the enthalpy difference `H`), the equation reads
`a R̈ = b + k dQ/dt`. The wall pressure `p_L` may depend on `Ṙ` (viscous and
shell terms), so `dQ/dt = A + B R̈` hides an `R̈` term. One batched `jax.jvp`
of `Q` along `(Ṙ, 0, ż, 1)` and `(0, 1, 0, 0)` returns `A` and `B`; the
acceleration is `(b + k A)/(a − k B)`. No law hand-codes the chain rule.
"""

from __future__ import annotations

from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import parameter_field
from ..equations import AbstractBarotropicMaterial
from ..typing import checked, parse
from ._contracts import (
    AbstractBubbleGasLaw,
    AbstractBubbleInterfaceLaw,
    AbstractBubbleLiquidLaw,
    AbstractBubblePressureDrive,
    BubbleEnvironment,
    BubbleGasEvaluation,
    BubbleGasState,
    BubbleInterfaceEvaluation,
    BubbleLiquidEvaluation,
    BubbleScales,
    MOLAR_GAS_CONSTANT,
    PressureDriveEvaluation,
    scalar_parameter,
)
from ._gas import sphere_volume


RadialBubbleEquation: TypeAlias = Literal[
    "rayleigh_plesset",
    "rayleigh_plesset_radiation",
    "rayleigh_plesset_gas_radiation",
    "keller_miksis",
    "gilmore",
]


class BubbleState(StrictModule):
    """Radius, wall velocity, gas content and law internal states."""

    radius: Array
    wall_velocity: Array
    gas: BubbleGasState
    liquid: Array
    interface: Array


class BubbleWallPressure(StrictModule):
    """Every wall-pressure contribution at one state, regime and time."""

    gas: BubbleGasEvaluation
    liquid: BubbleLiquidEvaluation
    interface: BubbleInterfaceEvaluation
    drive: PressureDriveEvaluation
    wall_pressure: Array
    far_field_pressure: Array

    @property
    def admissible(self) -> Array:
        """Whether every law reports an admissible, finite evaluation."""
        return (
            self.gas.admissible
            & self.liquid.admissible
            & self.interface.admissible
            & jnp.isfinite(self.wall_pressure)
            & jnp.isfinite(self.far_field_pressure)
        )


class RadialBubbleRates(StrictModule):
    """State derivative and the diagnostics of one radial-equation evaluation."""

    derivative: BubbleState
    wall: BubbleWallPressure
    acceleration: Array
    inertia_fraction: Array
    wall_sound_speed: Array
    wall_work_rate: Array
    dissipation_rate: Array
    gas_heat_rate: Array


class BubbleEquilibrium(StrictModule):
    """Bubble at rest at its equilibrium radius under the ambient pressure."""

    state: BubbleState
    regime: Array
    gas_pressure: Array
    admissible: Array


def _barotropic_enthalpy(
    material: AbstractBarotropicMaterial, pressure: Array, /
) -> tuple[Array, Array]:
    density = material.density_from_pressure(pressure)
    enthalpy = material.specific_internal_energy(density) + pressure / density
    return enthalpy, density


class RadialBubbleModel(StrictModule):
    """Static radial-equation selector composed with gas, liquid and interface laws.

    `rayleigh_plesset`, `rayleigh_plesset_radiation`,
    `rayleigh_plesset_gas_radiation` and `keller_miksis` use a constant-density
    liquid with `liquid_density` and `liquid_sound_speed` (the latter also sets
    Mach-number evidence). `gilmore` requires a barotropic `liquid_material`
    (for example `equations.TaitBarotropicMaterial`) that owns the density,
    enthalpy and local sound speed.

    - Rayleigh–Plesset: `ρ(R R̈ + 3/2 Ṙ²) = p_L − p_∞`.
    - Radiation-corrected: adds `(R/c) d(p_L − p_∞)/dt` to the right side.
    - Gas-radiation (Marmottant et al. 2005, Eq. 3): adds `(R/c) dp_g/dt` only,
      so the capillary, viscous, shell, vapor and drive pressures do not
      radiate. For a polytropic gas this is the factor `p_g(1 − 3κṘ/c)`.
    - Keller–Miksis: `(1 − Ṙ/c) R R̈ + 3/2 (1 − Ṙ/3c) Ṙ² =
      (1 + Ṙ/c)(p_L − p_∞)/ρ + R/(ρ c) d(p_L − p_∞)/dt`.
    - Gilmore: `(1 − Ṙ/C) R R̈ + 3/2 (1 − Ṙ/3C) Ṙ² = (1 + Ṙ/C) H +
      (R/C)(1 − Ṙ/C) dH/dt` with `H = h(p_L) − h(p_∞)`, `C = c(p_L)`.
    """

    equation: RadialBubbleEquation = eqx.field(static=True)
    gas: AbstractBubbleGasLaw
    liquid: AbstractBubbleLiquidLaw
    interface: AbstractBubbleInterfaceLaw
    environment: BubbleEnvironment
    liquid_density: Array | None = parameter_field()
    liquid_sound_speed: Array | None = parameter_field()
    liquid_material: AbstractBarotropicMaterial | None
    model_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        equation: RadialBubbleEquation,
        gas: AbstractBubbleGasLaw,
        liquid: AbstractBubbleLiquidLaw,
        interface: AbstractBubbleInterfaceLaw,
        environment: BubbleEnvironment,
        /,
        *,
        liquid_density: ArrayLike | None = None,
        liquid_sound_speed: ArrayLike | None = None,
        liquid_material: AbstractBarotropicMaterial | None = None,
    ) -> None:
        selected = parse(equation, RadialBubbleEquation, "equation")
        density, sound_speed = _liquid_properties(
            selected, liquid_density, liquid_sound_speed, liquid_material
        )
        self.equation = selected
        self.gas = gas
        self.liquid = liquid
        self.interface = interface
        self.environment = environment
        self.liquid_density = density
        self.liquid_sound_speed = sound_speed
        self.liquid_material = liquid_material
        self.model_id = canonical_fingerprint(
            {
                "kind": "radial-bubble-model",
                "equation": selected,
                "gas": gas.law_id,
                "liquid": liquid.law_id,
                "interface": interface.law_id,
                "material": None if liquid_material is None else liquid_material.material_id,
            }
        )

    @property
    def requires_stiff_integration(self) -> bool:
        """Whether any composed law declares stiff internal dynamics."""
        return self.gas.capabilities.stiff or self.liquid.stiff or self.interface.stiff

    def far_field_density(self) -> Array:
        """Liquid density at the ambient pressure."""
        if self.liquid_material is not None:
            return self.liquid_material.density_from_pressure(
                self.environment.ambient_pressure
            )
        if self.liquid_density is None:
            raise ValueError("Constant-density equations require liquid_density.")
        return self.liquid_density

    def far_field_sound_speed(self) -> Array:
        """Liquid sound speed at the ambient pressure."""
        if self.liquid_material is not None:
            return self.liquid_material.sound_speed(self.far_field_density())
        if self.liquid_sound_speed is None:
            raise ValueError("Constant-density equations require liquid_sound_speed.")
        return self.liquid_sound_speed

    def characteristic_scales(
        self,
        equilibrium: BubbleEquilibrium,
        characteristic_pressure: ArrayLike,
        initial_velocity: ArrayLike,
        /,
    ) -> BubbleScales:
        """Inertial scales from the largest of ambient, gas, drive and dynamic pressure."""
        density = self.far_field_density()
        radius = equilibrium.state.radius
        velocity_ = jnp.asarray(initial_velocity, dtype=jnp.float64)
        pressure = jnp.max(
            jnp.stack(
                (
                    self.environment.ambient_pressure,
                    equilibrium.gas_pressure,
                    jnp.asarray(characteristic_pressure, dtype=jnp.float64),
                    density * velocity_**2,
                )
            )
        )
        velocity = jnp.sqrt(pressure / density)
        temperature = self.environment.ambient_temperature
        energy = pressure * sphere_volume(radius)
        return BubbleScales(
            radius,
            velocity,
            radius / velocity,
            pressure,
            temperature,
            energy,
            energy / (MOLAR_GAS_CONSTANT * temperature),
        )

    def state_scale(self, gas: BubbleGasState, scales: BubbleScales, /) -> BubbleState:
        """Positive per-leaf magnitudes of a `BubbleState` with the structure of `gas`."""
        return BubbleState(
            scales.radius,
            scales.velocity,
            BubbleGasState(
                scales.amount,
                None if gas.internal_energy is None else scales.energy,
                self.gas.internal_scale(scales),
            ),
            self.liquid.internal_scale(scales),
            self.interface.internal_scale(scales),
        )

    def equilibrium(self, equilibrium_radius: ArrayLike, /) -> BubbleEquilibrium:
        """Bubble at rest at `equilibrium_radius` with the Laplace-balanced gas pressure.

        The gas pressure is `p0 − p_v + 2σ/R + p_elastic − S(R, 0)`; a negative
        value has no equilibrium and is reported as inadmissible.
        """
        radius = jnp.asarray(equilibrium_radius, dtype=jnp.float64)
        environment = self.environment
        zero = jnp.zeros_like(radius)
        interface_internal = self.interface.initialize(radius, environment)
        regime = self.interface.initial_regime(radius, interface_internal)
        interface = self.interface.evaluate(radius, zero, interface_internal, regime)
        liquid_internal = self.liquid.initialize(interface.outer_radius, environment)
        liquid = self.liquid.evaluate(
            interface.outer_radius, interface.outer_wall_velocity, liquid_internal
        )
        gas_pressure = (
            environment.ambient_pressure
            - environment.vapor_pressure
            + interface.capillary_pressure
            + interface.elastic_pressure
            + interface.viscous_pressure
            - liquid.stress
        )
        gas = self.gas.initialize(sphere_volume(radius), gas_pressure, environment)
        state = BubbleState(radius, zero, gas, liquid_internal, interface_internal)
        wall = self.wall_pressure(state, regime, zero, None)
        admissible = (
            (radius > 0.0)
            & (gas_pressure >= 0.0)
            & gas.finite
            & wall.admissible
        )
        return BubbleEquilibrium(state, regime, gas_pressure, admissible)

    def wall_pressure(
        self,
        state: BubbleState,
        regime: Array,
        time: Array,
        drive: AbstractBubblePressureDrive | None,
        /,
    ) -> BubbleWallPressure:
        """Liquid-side wall pressure and far-field pressure (no drive when `None`)."""
        radius = state.radius
        velocity = state.wall_velocity
        environment = self.environment
        volume = sphere_volume(radius)
        volume_rate = 4.0 * jnp.pi * radius**2 * velocity
        gas = self.gas.evaluate(volume, volume_rate, state.gas, environment)
        interface = self.interface.evaluate(radius, velocity, state.interface, regime)
        liquid = self.liquid.evaluate(
            interface.outer_radius, interface.outer_wall_velocity, state.liquid
        )
        forcing = (
            PressureDriveEvaluation(
                jnp.zeros_like(time), jnp.zeros_like(time), jnp.ones_like(time, dtype=jnp.bool_)
            )
            if drive is None
            else drive.evaluate(time)
        )
        wall = (
            gas.pressure
            + environment.vapor_pressure
            - interface.capillary_pressure
            - interface.elastic_pressure
            - interface.viscous_pressure
            + liquid.stress
        )
        return BubbleWallPressure(
            gas,
            liquid,
            interface,
            forcing,
            wall,
            environment.ambient_pressure + forcing.pressure,
        )

    def _driving_quantity(self, wall: BubbleWallPressure, /) -> Array:
        match self.equation:
            case (
                "rayleigh_plesset"
                | "rayleigh_plesset_radiation"
                | "rayleigh_plesset_gas_radiation"
                | "keller_miksis"
            ):
                return wall.wall_pressure - wall.far_field_pressure
            case "gilmore":
                if self.liquid_material is None:
                    raise ValueError("Gilmore requires a barotropic liquid material.")
                wall_enthalpy, _ = _barotropic_enthalpy(self.liquid_material, wall.wall_pressure)
                far_enthalpy, _ = _barotropic_enthalpy(
                    self.liquid_material, wall.far_field_pressure
                )
                return wall_enthalpy - far_enthalpy
            case _:
                assert_never(self.equation)

    def _radiated_quantity(self, wall: BubbleWallPressure, /) -> Array:
        """Wall quantity whose time derivative carries the acoustic-radiation term."""
        match self.equation:
            case "rayleigh_plesset_gas_radiation":
                return wall.gas.pressure
            case "rayleigh_plesset_radiation" | "keller_miksis" | "gilmore":
                return self._driving_quantity(wall)
            case "rayleigh_plesset":
                raise ValueError("rayleigh_plesset has no acoustic-radiation term.")
            case _:
                assert_never(self.equation)

    def _quantity_rate(
        self,
        state: BubbleState,
        regime: Array,
        time: Array,
        drive: AbstractBubblePressureDrive | None,
        rates: BubbleState,
        /,
    ) -> tuple[Array, Array]:
        """Explicit part `A` and implicit coefficient `B` of `dQ/dt = A + B R̈`."""

        def quantity(
            radius: Array,
            velocity: Array,
            gas: BubbleGasState,
            liquid: Array,
            interface: Array,
            current_time: Array,
        ) -> Array:
            current = BubbleState(radius, velocity, gas, liquid, interface)
            return self._radiated_quantity(
                self.wall_pressure(current, regime, current_time, drive)
            )

        primals = (
            state.radius,
            state.wall_velocity,
            state.gas,
            state.liquid,
            state.interface,
            time,
        )
        explicit = (
            state.wall_velocity,
            jnp.zeros_like(state.wall_velocity),
            rates.gas,
            rates.liquid,
            rates.interface,
            jnp.ones_like(time),
        )
        implicit = (
            jnp.zeros_like(state.radius),
            jnp.ones_like(state.wall_velocity),
            jax.tree.map(jnp.zeros_like, rates.gas),
            jnp.zeros_like(rates.liquid),
            jnp.zeros_like(rates.interface),
            jnp.zeros_like(time),
        )
        tangents = jax.tree.map(lambda left, right: jnp.stack((left, right)), explicit, implicit)
        _, derivatives = jax.vmap(lambda tangent: jax.jvp(quantity, primals, tangent))(tangents)
        return derivatives[0], derivatives[1]

    def rates(
        self,
        state: BubbleState,
        regime: Array,
        time: Array,
        drive: AbstractBubblePressureDrive | None,
        /,
    ) -> RadialBubbleRates:
        """Time derivative of `state` inside the fixed interface `regime`."""
        wall = self.wall_pressure(state, regime, time, drive)
        radius = state.radius
        velocity = state.wall_velocity
        gas_rate = BubbleGasState(
            wall.gas.amount_rate, wall.gas.energy_rate, wall.gas.internal_rate
        )
        partial = BubbleState(
            velocity,
            jnp.zeros_like(velocity),
            gas_rate,
            wall.liquid.internal_rate,
            wall.interface.internal_rate,
        )
        acceleration, inertia_fraction, sound_speed = self._acceleration(
            state, regime, time, drive, wall, partial
        )
        derivative = BubbleState(
            velocity,
            acceleration,
            gas_rate,
            wall.liquid.internal_rate,
            wall.interface.internal_rate,
        )
        heat = wall.gas.heat_rate
        return RadialBubbleRates(
            derivative,
            wall,
            acceleration,
            inertia_fraction,
            sound_speed,
            4.0 * jnp.pi * radius**2 * velocity * (wall.wall_pressure - wall.far_field_pressure),
            wall.liquid.dissipation_rate + wall.interface.dissipation_rate,
            jnp.zeros_like(radius) if heat is None else heat,
        )

    def _acceleration(
        self,
        state: BubbleState,
        regime: Array,
        time: Array,
        drive: AbstractBubblePressureDrive | None,
        wall: BubbleWallPressure,
        partial: BubbleState,
        /,
    ) -> tuple[Array, Array, Array]:
        radius = state.radius
        velocity = state.wall_velocity
        match self.equation:
            case "rayleigh_plesset":
                density = self.far_field_density()
                driving = wall.wall_pressure - wall.far_field_pressure
                acceleration = (driving / density - 1.5 * velocity**2) / radius
                return acceleration, jnp.ones_like(radius), self.far_field_sound_speed()
            case "rayleigh_plesset_radiation" | "rayleigh_plesset_gas_radiation":
                density = self.far_field_density()
                sound_speed = self.far_field_sound_speed()
                driving = wall.wall_pressure - wall.far_field_pressure
                explicit, implicit = self._quantity_rate(state, regime, time, drive, partial)
                inertia = density * radius - radius / sound_speed * implicit
                acceleration = (
                    driving - 1.5 * density * velocity**2 + radius / sound_speed * explicit
                ) / inertia
                return acceleration, inertia / (density * radius), sound_speed
            case "keller_miksis":
                density = self.far_field_density()
                sound_speed = self.far_field_sound_speed()
                driving = wall.wall_pressure - wall.far_field_pressure
                explicit, implicit = self._quantity_rate(state, regime, time, drive, partial)
                mach = velocity / sound_speed
                inertia = (1.0 - mach) * radius - radius * implicit / (density * sound_speed)
                acceleration = (
                    (1.0 + mach) * driving / density
                    + radius * explicit / (density * sound_speed)
                    - 1.5 * (1.0 - mach / 3.0) * velocity**2
                ) / inertia
                return acceleration, inertia / radius, sound_speed
            case "gilmore":
                if self.liquid_material is None:
                    raise ValueError("Gilmore requires a barotropic liquid material.")
                enthalpy = self._driving_quantity(wall)
                wall_density = self.liquid_material.density_from_pressure(wall.wall_pressure)
                sound_speed = self.liquid_material.sound_speed(wall_density)
                explicit, implicit = self._quantity_rate(state, regime, time, drive, partial)
                mach = velocity / sound_speed
                memory = radius / sound_speed * (1.0 - mach)
                inertia = (1.0 - mach) * radius - memory * implicit
                acceleration = (
                    (1.0 + mach) * enthalpy
                    + memory * explicit
                    - 1.5 * (1.0 - mach / 3.0) * velocity**2
                ) / inertia
                return acceleration, inertia / radius, sound_speed
            case _:
                assert_never(self.equation)


def _liquid_properties(
    equation: RadialBubbleEquation,
    liquid_density: ArrayLike | None,
    liquid_sound_speed: ArrayLike | None,
    liquid_material: AbstractBarotropicMaterial | None,
    /,
) -> tuple[Array | None, Array | None]:
    match equation:
        case "gilmore":
            if not isinstance(liquid_material, AbstractBarotropicMaterial):
                raise TypeError("Gilmore requires an AbstractBarotropicMaterial liquid_material.")
            if liquid_density is not None or liquid_sound_speed is not None:
                raise ValueError(
                    "Gilmore takes density and sound speed from liquid_material only."
                )
            return None, None
        case (
            "rayleigh_plesset"
            | "rayleigh_plesset_radiation"
            | "rayleigh_plesset_gas_radiation"
            | "keller_miksis"
        ):
            if liquid_material is not None:
                raise ValueError(f"{equation} uses a constant-density liquid, not a material.")
            if liquid_density is None or liquid_sound_speed is None:
                raise ValueError(f"{equation} requires liquid_density and liquid_sound_speed.")
            return (
                scalar_parameter(liquid_density, "liquid_density", lower=0.0),
                scalar_parameter(liquid_sound_speed, "liquid_sound_speed", lower=0.0),
            )
        case _:
            assert_never(equation)


__all__ = [
    "BubbleEquilibrium",
    "BubbleState",
    "BubbleWallPressure",
    "RadialBubbleEquation",
    "RadialBubbleModel",
    "RadialBubbleRates",
]
