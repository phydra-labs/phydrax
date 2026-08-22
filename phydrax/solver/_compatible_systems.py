#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import StructuredCochainBridge


class CompatibleMaxwellState(StrictModule):
    electric: Array
    magnetic: Array


class CompatibleMaxwellDynamics(StrictModule, NonTrainableState):
    """Lossless Maxwell evolution on degree-one electric and degree-two magnetic cochains."""

    bridge: StructuredCochainBridge
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        permittivity: float = 1.0,
        permeability: float = 1.0,
    ):
        epsilon = float(permittivity)
        mu = float(permeability)
        if not isinstance(bridge, StructuredCochainBridge) or bridge.dimension != 3:
            raise ValueError(
                "Compatible Maxwell dynamics requires a three-dimensional bridge."
            )
        if not np.isfinite(epsilon) or not np.isfinite(mu) or epsilon <= 0.0 or mu <= 0.0:
            raise ValueError("Maxwell material coefficients must be finite and positive.")
        self.bridge = bridge
        self.permittivity = epsilon
        self.permeability = mu
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "compatible-maxwell-dynamics",
                "bridge": bridge.bridge_id,
                "permittivity": epsilon,
                "permeability": mu,
            }
        )

    def pack(self, electric: ArrayLike, magnetic: ArrayLike, /) -> CompatibleMaxwellState:
        electric_ = jnp.asarray(electric)
        magnetic_ = jnp.asarray(magnetic)
        if electric_.shape != (
            self.bridge.cochain.cell_counts[1],
        ) or magnetic_.shape != (self.bridge.cochain.cell_counts[2],):
            raise ValueError("Maxwell cochain sizes must match degrees one and two.")
        return CompatibleMaxwellState(electric=electric_, magnetic=magnetic_)

    def drift(self, state: CompatibleMaxwellState, /) -> CompatibleMaxwellState:
        if not isinstance(state, CompatibleMaxwellState):
            raise TypeError("state must be CompatibleMaxwellState.")
        electric_rate = self.bridge.codifferential(2, state.magnetic) / (
            self.permittivity * self.permeability
        )
        magnetic_rate = -self.bridge.exterior_derivative(1, state.electric)
        return self.pack(electric_rate, magnetic_rate)

    def leapfrog_step(
        self,
        state: CompatibleMaxwellState,
        step_size: ArrayLike,
        /,
    ) -> CompatibleMaxwellState:
        dt = jnp.asarray(step_size)
        magnetic_half = state.magnetic - 0.5 * dt * self.bridge.exterior_derivative(
            1, state.electric
        )
        electric_new = state.electric + dt * self.bridge.codifferential(
            2, magnetic_half
        ) / (self.permittivity * self.permeability)
        magnetic_new = magnetic_half - 0.5 * dt * self.bridge.exterior_derivative(
            1, electric_new
        )
        return self.pack(electric_new, magnetic_new)

    def energy(self, state: CompatibleMaxwellState, /) -> Array:
        electric_star = self.bridge.cochain.hodge_stars[1]
        magnetic_star = self.bridge.cochain.hodge_stars[2]
        return 0.5 * (
            self.permittivity
            * jnp.real(jnp.vdot(state.electric, electric_star * state.electric))
            + 1.0
            / self.permeability
            * jnp.real(jnp.vdot(state.magnetic, magnetic_star * state.magnetic))
        )

    def magnetic_constraint(self, state: CompatibleMaxwellState, /) -> Array:
        return self.bridge.exterior_derivative(2, state.magnetic)


class CompatibleElasticityState(StrictModule):
    displacement: Array
    velocity: Array


class CompatibleElasticityDynamics(StrictModule, NonTrainableState):
    """Compatible scalar/vector elastic-wave reference on degree-zero cochains."""

    bridge: StructuredCochainBridge
    wave_speed: float = eqx.field(static=True)
    components: int = eqx.field(static=True)
    dynamics_id: str = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        wave_speed: float = 1.0,
        components: int = 1,
    ):
        speed = float(wave_speed)
        components_ = int(components)
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or speed <= 0.0
            or components_ <= 0
        ):
            raise ValueError("Elasticity bridge, wave speed, and components are invalid.")
        self.bridge = bridge
        self.wave_speed = speed
        self.components = components_
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "compatible-elasticity-dynamics",
                "bridge": bridge.bridge_id,
                "wave_speed": speed,
                "components": components_,
            }
        )

    def _apply_components(self, function, values: Array, /) -> Array:
        return (
            function(values)
            if self.components == 1
            else jax.vmap(function, in_axes=1, out_axes=1)(values)
        )

    def pack(
        self, displacement: ArrayLike, velocity: ArrayLike, /
    ) -> CompatibleElasticityState:
        displacement_ = jnp.asarray(displacement)
        velocity_ = jnp.asarray(velocity)
        shape = (self.bridge.cochain.cell_counts[0],) + (
            () if self.components == 1 else (self.components,)
        )
        if displacement_.shape != shape or velocity_.shape != shape:
            raise ValueError(
                "Elasticity state must match degree-zero cochain/components."
            )
        return CompatibleElasticityState(displacement=displacement_, velocity=velocity_)

    def stiffness(self, displacement: ArrayLike, /) -> Array:
        value = jnp.asarray(displacement)
        return self._apply_components(
            lambda component: self.bridge.codifferential(
                1,
                self.bridge.exterior_derivative(0, component),
            ),
            value,
        )

    def drift(self, state: CompatibleElasticityState, /) -> CompatibleElasticityState:
        return self.pack(
            state.velocity,
            -(self.wave_speed**2) * self.stiffness(state.displacement),
        )

    def energy(self, state: CompatibleElasticityState, /) -> Array:
        h0 = self.bridge.cochain.hodge_stars[0]
        kinetic = jnp.sum(
            h0.reshape((-1,) + (1,) * (state.velocity.ndim - 1)) * state.velocity**2
        )
        gradient = self._apply_components(
            lambda component: self.bridge.exterior_derivative(0, component),
            state.displacement,
        )
        h1 = self.bridge.cochain.hodge_stars[1]
        potential = self.wave_speed**2 * jnp.sum(
            h1.reshape((-1,) + (1,) * (gradient.ndim - 1)) * gradient**2
        )
        return 0.5 * (kinetic + potential)


class IncompressibleProjectionResult(StrictModule):
    velocity: Array
    pressure: Array
    divergence_before: Array
    divergence_after: Array


class CompatibleIncompressibleProjection(StrictModule, NonTrainableState):
    """Degree-one velocity projection through the compatible scalar Poisson complex."""

    bridge: StructuredCochainBridge
    poisson_pseudoinverse: Array
    projection_id: str = eqx.field(static=True)

    def __init__(self, bridge: StructuredCochainBridge, /, *, size_budget: int = 4096):
        if not isinstance(bridge, StructuredCochainBridge) or bridge.dimension < 2:
            raise ValueError(
                "Incompressible projection requires a multidimensional bridge."
            )
        scalar_size = bridge.cochain.cell_counts[0]
        if scalar_size > int(size_budget):
            raise ValueError("Compatible pressure solve exceeds explicit size budget.")
        identity = jnp.eye(scalar_size)
        columns = jax.vmap(
            lambda value: bridge.codifferential(
                1,
                bridge.exterior_derivative(0, value),
            )
        )(identity)
        poisson = columns.T
        pseudoinverse = jnp.linalg.pinv(poisson, rtol=1e-12)
        self.bridge = bridge
        self.poisson_pseudoinverse = pseudoinverse
        self.projection_id = canonical_fingerprint(
            {
                "kind": "compatible-incompressible-projection",
                "bridge": bridge.bridge_id,
                "scalar_size": scalar_size,
            }
        )

    def project(self, velocity: ArrayLike, /) -> IncompressibleProjectionResult:
        value = jnp.asarray(velocity)
        if value.shape != (self.bridge.cochain.cell_counts[1],):
            raise ValueError("Incompressible velocity must be a degree-one cochain.")
        divergence = self.bridge.codifferential(1, value)
        pressure = self.poisson_pseudoinverse @ divergence
        projected = value - self.bridge.exterior_derivative(0, pressure)
        divergence_after = self.bridge.codifferential(1, projected)
        return IncompressibleProjectionResult(
            velocity=projected,
            pressure=pressure,
            divergence_before=divergence,
            divergence_after=divergence_after,
        )


class CompatibleIdealMHDState(StrictModule):
    magnetic: Array


class CompatibleIdealMHDInductionDynamics(StrictModule):
    """Constrained magnetic induction B'=-dE with caller-supplied ideal Ohm field."""

    bridge: StructuredCochainBridge
    electric_field: Any
    dynamics_id: str = eqx.field(static=True)

    def __init__(self, bridge: StructuredCochainBridge, electric_field, /):
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or bridge.dimension != 3
            or not callable(electric_field)
        ):
            raise ValueError(
                "Compatible ideal-MHD induction requires a 3D bridge and electric field."
            )
        self.bridge = bridge
        self.electric_field = electric_field
        self.dynamics_id = canonical_fingerprint(
            {
                "kind": "compatible-ideal-mhd-induction",
                "bridge": bridge.bridge_id,
                "electric_field": repr(electric_field),
            }
        )

    def pack(self, magnetic: ArrayLike, /) -> CompatibleIdealMHDState:
        value = jnp.asarray(magnetic)
        if value.shape != (self.bridge.cochain.cell_counts[2],):
            raise ValueError("MHD magnetic field must be a degree-two cochain.")
        return CompatibleIdealMHDState(magnetic=value)

    def drift(
        self,
        time: Array,
        state: CompatibleIdealMHDState,
        args: Any = None,
    ) -> CompatibleIdealMHDState:
        electric = jnp.asarray(self.electric_field(time, state.magnetic, args))
        if electric.shape != (self.bridge.cochain.cell_counts[1],):
            raise ValueError("Ideal Ohm electric field must be a degree-one cochain.")
        return self.pack(-self.bridge.exterior_derivative(1, electric))

    def step(
        self,
        time: Array,
        state: CompatibleIdealMHDState,
        step_size: ArrayLike,
        args: Any = None,
    ) -> CompatibleIdealMHDState:
        return self.pack(
            state.magnetic
            + jnp.asarray(step_size) * self.drift(time, state, args).magnetic
        )

    def magnetic_constraint(self, state: CompatibleIdealMHDState, /) -> Array:
        return self.bridge.exterior_derivative(2, state.magnetic)


class CompatibleVariableDensityProjection(StrictModule, NonTrainableState):
    """Variable-density projection δ(ρ⁻¹dp) with degree-zero density."""

    bridge: StructuredCochainBridge
    size_budget: int = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)

    def __init__(self, bridge: StructuredCochainBridge, /, *, size_budget: int = 2048):
        if not isinstance(bridge, StructuredCochainBridge) or bridge.dimension < 2:
            raise ValueError(
                "Variable-density projection requires multidimensional bridge."
            )
        if bridge.cochain.cell_counts[0] > int(size_budget):
            raise ValueError("Variable-density projection exceeds explicit size budget.")
        self.bridge = bridge
        self.size_budget = int(size_budget)
        self.projection_id = canonical_fingerprint(
            {
                "kind": "compatible-variable-density-projection",
                "bridge": bridge.bridge_id,
                "size_budget": int(size_budget),
            }
        )

    def _edge_inverse_density(self, density: Array, /) -> Array:
        incidence = self.bridge.cochain.topology.incidences[0].relation
        valid = incidence.valid
        source = jnp.where(valid, incidence.source_indices, 0)
        target = jnp.where(valid, incidence.target_indices, 0)
        sums = (
            jnp.zeros((incidence.target_size,), dtype=density.dtype)
            .at[target]
            .add(jnp.where(valid, density[source], 0.0))
        )
        counts = (
            jnp.zeros((incidence.target_size,), dtype=density.dtype)
            .at[target]
            .add(valid.astype(density.dtype))
        )
        return counts / sums

    def project(
        self,
        velocity: ArrayLike,
        density: ArrayLike,
        /,
    ) -> IncompressibleProjectionResult:
        velocity_ = jnp.asarray(velocity)
        density_ = jnp.asarray(density)
        if velocity_.shape != (self.bridge.cochain.cell_counts[1],) or density_.shape != (
            self.bridge.cochain.cell_counts[0],
        ):
            raise ValueError("Variable-density projection cochain sizes are invalid.")
        density_ = eqx.error_if(
            density_,
            jnp.any(~jnp.isfinite(density_)) | jnp.any(density_ <= 0.0),
            "Projection density must be finite and positive.",
        )
        inverse_density = self._edge_inverse_density(density_)

        def poisson_action(pressure):
            return self.bridge.codifferential(
                1,
                inverse_density * self.bridge.exterior_derivative(0, pressure),
            )

        size = self.bridge.cochain.cell_counts[0]
        matrix = jax.vmap(poisson_action)(jnp.eye(size)).T
        divergence = self.bridge.codifferential(1, velocity_)
        pressure = jnp.linalg.pinv(matrix, rtol=1e-12) @ divergence
        projected = velocity_ - inverse_density * self.bridge.exterior_derivative(
            0, pressure
        )
        return IncompressibleProjectionResult(
            velocity=projected,
            pressure=pressure,
            divergence_before=divergence,
            divergence_after=self.bridge.codifferential(1, projected),
        )


class CompatiblePoroelasticState(StrictModule):
    displacement: Array
    velocity: Array
    pressure: Array


class CompatiblePoroelasticDynamics(StrictModule, NonTrainableState):
    """Compatible scalar Biot-like elastic/pressure coupling on degree-zero forms."""

    bridge: StructuredCochainBridge
    wave_speed: float = eqx.field(static=True)
    hydraulic_diffusivity: float = eqx.field(static=True)
    coupling: float = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        wave_speed: float = 1.0,
        hydraulic_diffusivity: float = 0.1,
        coupling: float = 0.2,
    ):
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or not np.isfinite(wave_speed)
            or not np.isfinite(hydraulic_diffusivity)
            or not np.isfinite(coupling)
            or wave_speed <= 0.0
            or hydraulic_diffusivity < 0.0
        ):
            raise ValueError("Poroelastic coefficients must be finite and physical.")
        self.bridge = bridge
        self.wave_speed = float(wave_speed)
        self.hydraulic_diffusivity = float(hydraulic_diffusivity)
        self.coupling = float(coupling)

    def drift(self, state: CompatiblePoroelasticState, /) -> CompatiblePoroelasticState:
        laplace_u = self.bridge.laplace_de_rham(0, state.displacement)
        laplace_p = self.bridge.laplace_de_rham(0, state.pressure)
        laplace_v = self.bridge.laplace_de_rham(0, state.velocity)
        return CompatiblePoroelasticState(
            displacement=state.velocity,
            velocity=-(self.wave_speed**2) * laplace_u + self.coupling * laplace_p,
            pressure=-self.hydraulic_diffusivity * laplace_p - self.coupling * laplace_v,
        )


class CompatibleThermoelasticState(StrictModule):
    displacement: Array
    velocity: Array
    temperature: Array


class CompatibleThermoelasticDynamics(StrictModule, NonTrainableState):
    """Compatible scalar thermoelastic wave/heat reference."""

    bridge: StructuredCochainBridge
    wave_speed: float = eqx.field(static=True)
    thermal_diffusivity: float = eqx.field(static=True)
    expansion: float = eqx.field(static=True)

    def __init__(
        self,
        bridge: StructuredCochainBridge,
        /,
        *,
        wave_speed: float = 1.0,
        thermal_diffusivity: float = 0.1,
        expansion: float = 0.2,
    ):
        if (
            not isinstance(bridge, StructuredCochainBridge)
            or not np.isfinite(wave_speed)
            or not np.isfinite(thermal_diffusivity)
            or not np.isfinite(expansion)
            or wave_speed <= 0.0
            or thermal_diffusivity < 0.0
        ):
            raise ValueError("Thermoelastic coefficients must be finite and physical.")
        self.bridge = bridge
        self.wave_speed = float(wave_speed)
        self.thermal_diffusivity = float(thermal_diffusivity)
        self.expansion = float(expansion)

    def drift(
        self,
        state: CompatibleThermoelasticState,
        /,
    ) -> CompatibleThermoelasticState:
        laplace_u = self.bridge.laplace_de_rham(0, state.displacement)
        laplace_t = self.bridge.laplace_de_rham(0, state.temperature)
        laplace_v = self.bridge.laplace_de_rham(0, state.velocity)
        return CompatibleThermoelasticState(
            displacement=state.velocity,
            velocity=-(self.wave_speed**2) * laplace_u + self.expansion * laplace_t,
            temperature=-self.thermal_diffusivity * laplace_t
            - self.expansion * laplace_v,
        )


__all__ = [
    "CompatibleElasticityDynamics",
    "CompatibleElasticityState",
    "CompatibleIdealMHDInductionDynamics",
    "CompatibleIdealMHDState",
    "CompatibleIncompressibleProjection",
    "CompatibleMaxwellDynamics",
    "CompatibleMaxwellState",
    "CompatiblePoroelasticDynamics",
    "CompatiblePoroelasticState",
    "CompatibleThermoelasticDynamics",
    "CompatibleThermoelasticState",
    "CompatibleVariableDensityProjection",
    "IncompressibleProjectionResult",
]
