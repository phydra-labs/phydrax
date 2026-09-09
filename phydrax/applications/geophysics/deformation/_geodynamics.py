#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class SphericalShellGeometry(StrictModule, NonTrainableState):
    inner_radius_m: float = eqx.field(static=True)
    outer_radius_m: float = eqx.field(static=True)
    reference_body_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self, inner_radius_m: float, outer_radius_m: float, reference_body_id: str, /
    ):
        inner, outer = float(inner_radius_m), float(outer_radius_m)
        body = str(reference_body_id).strip()
        if (
            not np.isfinite(inner)
            or not np.isfinite(outer)
            or inner <= 0
            or outer <= inner
            or not body
        ):
            raise ValueError("Spherical shell radii and reference body are invalid.")
        self.inner_radius_m, self.outer_radius_m, self.reference_body_id = (
            inner,
            outer,
            body,
        )
        self.geometry_id = canonical_fingerprint(
            {"kind": "spherical-shell-geometry", "radii_m": (inner, outer), "body": body}
        )


class ArrheniusViscoplasticRheology(StrictModule):
    reference_viscosity_Pa_s: Array
    activation_temperature_K: Array
    reference_temperature_K: Array
    minimum_viscosity_Pa_s: Array
    maximum_viscosity_Pa_s: Array
    yield_stress_Pa: Array

    def __init__(
        self,
        reference_viscosity_Pa_s: ArrayLike,
        activation_temperature_K: ArrayLike,
        reference_temperature_K: ArrayLike,
        minimum_viscosity_Pa_s: ArrayLike,
        maximum_viscosity_Pa_s: ArrayLike,
        yield_stress_Pa: ArrayLike,
        /,
    ):
        values = jnp.broadcast_arrays(
            *(
                jnp.asarray(value)
                for value in (
                    reference_viscosity_Pa_s,
                    activation_temperature_K,
                    reference_temperature_K,
                    minimum_viscosity_Pa_s,
                    maximum_viscosity_Pa_s,
                    yield_stress_Pa,
                )
            )
        )
        invalid = any(jnp.any(~jnp.isfinite(value)) for value in values)
        invalid = (
            invalid
            | jnp.any(values[0] <= 0)
            | jnp.any(values[1] < 0)
            | jnp.any(values[2] <= 0)
            | jnp.any(values[3] <= 0)
            | jnp.any(values[4] <= values[3])
            | jnp.any(values[5] <= 0)
        )
        self.reference_viscosity_Pa_s = eqx.error_if(
            values[0],
            invalid,
            "Geodynamic rheology parameters must be finite and physical.",
        )
        (
            self.activation_temperature_K,
            self.reference_temperature_K,
            self.minimum_viscosity_Pa_s,
            self.maximum_viscosity_Pa_s,
            self.yield_stress_Pa,
        ) = values[1:]

    def viscosity(
        self, temperature_K: ArrayLike, strain_rate_s_inverse: ArrayLike, /
    ) -> tuple[Array, Array]:
        temperature, strain_rate = jnp.broadcast_arrays(
            jnp.asarray(temperature_K), jnp.asarray(strain_rate_s_inverse)
        )
        temperature = eqx.error_if(
            temperature,
            jnp.any(~jnp.isfinite(temperature))
            | jnp.any(temperature <= 0)
            | jnp.any(~jnp.isfinite(strain_rate))
            | jnp.any(strain_rate < 0),
            "Geodynamic temperature/strain rate must be finite and physical.",
        )
        arrhenius = self.reference_viscosity_Pa_s * jnp.exp(
            self.activation_temperature_K
            * (1.0 / temperature - 1.0 / self.reference_temperature_K)
        )
        plastic = self.yield_stress_Pa / jnp.maximum(2.0 * strain_rate, 1e-30)
        raw = jnp.minimum(arrhenius, plastic)
        viscosity = jnp.minimum(
            jnp.maximum(raw, self.minimum_viscosity_Pa_s),
            self.maximum_viscosity_Pa_s,
        )
        active = (
            (raw > self.minimum_viscosity_Pa_s)
            & (raw < self.maximum_viscosity_Pa_s)
            & (arrhenius != plastic)
        )
        return viscosity, active


class GeodynamicsState(StrictModule):
    velocity: Array
    pressure: Array
    temperature_K: Array
    time_s: Array
    plan_id: str = eqx.field(static=True)


class GeodynamicsStepResult(StrictModule):
    state: GeodynamicsState
    incompressibility_residual: Array
    momentum_residual: Array
    thermal_residual: Array
    successful: Array
    derivative_available: Array


class SphericalThermomechanicalPlan(StrictModule, NonTrainableState):
    """Spherical-shell Stokes KKT plus caller-supplied metric thermal rate."""

    geometry: SphericalShellGeometry
    velocity_space: la.ArraySpace
    pressure_space: la.ArraySpace
    temperature_space: la.ArraySpace
    momentum_factory: Callable[[Array], la.AbstractLinearOperator] = eqx.field(
        static=True
    )
    divergence: la.AbstractLinearOperator
    buoyancy: la.AbstractLinearOperator
    thermal_rate: Callable[[Array, Array], Array] = eqx.field(static=True)
    momentum_factory_id: str = eqx.field(static=True)
    thermal_rate_id: str = eqx.field(static=True)
    maximum_timestep_s: float = eqx.field(static=True)
    block_space: la.ArraySpace
    policy: la.LinearSolvePolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: SphericalShellGeometry,
        velocity_space: la.ArraySpace,
        pressure_space: la.ArraySpace,
        momentum_factory: Callable[[Array], la.AbstractLinearOperator],
        divergence: la.AbstractLinearOperator,
        buoyancy: la.AbstractLinearOperator,
        thermal_rate: Callable[[Array, Array], Array],
        maximum_timestep_s: float,
        /,
        *,
        momentum_factory_id: str,
        thermal_rate_id: str,
    ):
        if not isinstance(geometry, SphericalShellGeometry):
            raise TypeError("Geodynamics requires spherical shell geometry.")
        if not isinstance(velocity_space, la.ArraySpace) or not isinstance(
            pressure_space, la.ArraySpace
        ):
            raise TypeError("Geodynamics velocity/pressure spaces must be ArraySpace.")
        if not callable(momentum_factory) or not callable(thermal_rate):
            raise TypeError("Geodynamics momentum and thermal actions must be callable.")
        momentum_identity = str(momentum_factory_id).strip()
        thermal_identity = str(thermal_rate_id).strip()
        if not momentum_identity or not thermal_identity:
            raise ValueError("Geodynamic callable identities must be nonempty.")
        if not isinstance(divergence, la.AbstractLinearOperator) or not isinstance(
            buoyancy, la.AbstractLinearOperator
        ):
            raise TypeError("Geodynamics divergence/buoyancy must be native operators.")
        if not isinstance(buoyancy.source, la.ArraySpace):
            raise TypeError("Geodynamic buoyancy source must be an ArraySpace.")
        if not (
            divergence.source.compatible(velocity_space)
            and divergence.target.compatible(pressure_space)
            and buoyancy.target.compatible(velocity_space)
        ):
            raise ValueError("Geodynamic divergence/buoyancy spaces do not compose.")
        maximum = float(maximum_timestep_s)
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Geodynamic maximum timestep must be positive finite.")
        self.geometry, self.velocity_space, self.pressure_space = (
            geometry,
            velocity_space,
            pressure_space,
        )
        self.temperature_space = buoyancy.source
        self.momentum_factory, self.divergence, self.buoyancy = (
            momentum_factory,
            divergence,
            buoyancy,
        )
        self.thermal_rate, self.maximum_timestep_s = thermal_rate, maximum
        self.momentum_factory_id = momentum_identity
        self.thermal_rate_id = thermal_identity
        self.block_space = la.ArraySpace(
            (velocity_space.size + pressure_space.size,),
            dtype=jnp.result_type(velocity_space.dtype, pressure_space.dtype),
        )
        self.policy = la.LinearSolvePolicy(
            la.GMRES(restart=50, stagnation_iterations=50),
            tolerance=la.TolerancePolicy(relative=1e-8, absolute=1e-11, max_steps=2000),
            failure=la.FailurePolicy("status"),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spherical-thermomechanical-geodynamics",
                "geometry": geometry.geometry_id,
                "divergence": divergence.operator_id,
                "buoyancy": buoyancy.operator_id,
                "momentum_factory": momentum_identity,
                "thermal_rate": thermal_identity,
                "maximum_timestep_s": maximum,
            }
        )

    def initial_state(
        self,
        temperature_K: ArrayLike,
        /,
        *,
        velocity: ArrayLike = 0.0,
        pressure: ArrayLike = 0.0,
        time_s: ArrayLike = 0.0,
    ) -> GeodynamicsState:
        temperature = jnp.broadcast_to(
            jnp.asarray(temperature_K), self.temperature_space.shape
        )
        velocity_ = jnp.broadcast_to(jnp.asarray(velocity), self.velocity_space.shape)
        pressure_ = jnp.broadcast_to(jnp.asarray(pressure), self.pressure_space.shape)
        time = jnp.asarray(time_s)
        if time.shape != ():
            raise ValueError("Geodynamic initial time must be scalar.")
        temperature = eqx.error_if(
            temperature,
            jnp.any(~jnp.isfinite(temperature))
            | jnp.any(temperature <= 0)
            | jnp.any(~jnp.isfinite(velocity_))
            | jnp.any(~jnp.isfinite(pressure_))
            | ~jnp.isfinite(time),
            "Geodynamic initial state must be finite with positive temperature.",
        )
        return GeodynamicsState(velocity_, pressure_, temperature, time, self.plan_id)

    def step(self, state: GeodynamicsState, dt_s: ArrayLike, /) -> GeodynamicsStepResult:
        if not isinstance(state, GeodynamicsState) or state.plan_id != self.plan_id:
            raise ValueError("Geodynamic state belongs to a different plan.")
        dt = jnp.asarray(dt_s)
        dt = eqx.error_if(
            dt,
            ~jnp.isfinite(dt) | (dt <= 0) | (dt > self.maximum_timestep_s),
            "Geodynamic timestep must be positive and within stability evidence.",
        )
        momentum = self.momentum_factory(state.temperature_K)
        if (
            not isinstance(momentum, la.AbstractLinearOperator)
            or not momentum.source.compatible(self.velocity_space)
            or not momentum.target.compatible(self.velocity_space)
        ):
            raise TypeError("Momentum factory returned an incompatible operator.")
        nv = self.velocity_space.size

        def kkt(values):
            velocity = self.velocity_space.unflatten(values[:nv])
            pressure = self.pressure_space.unflatten(values[nv:])
            momentum_residual = momentum.mv(velocity) + self.divergence.transpose_mv(
                pressure
            )
            continuity = self.divergence.mv(velocity)
            return jnp.concatenate(
                (
                    self.velocity_space.flatten(momentum_residual),
                    self.pressure_space.flatten(continuity),
                )
            )

        operator = la.FunctionLinearOperator(
            kkt, source=self.block_space, target=self.block_space
        )
        buoyancy = self.buoyancy.mv(state.temperature_K)
        rhs = jnp.concatenate(
            (self.velocity_space.flatten(buoyancy), jnp.zeros(self.pressure_space.size))
        )
        solved = la.solve(la.LinearSystem(operator), rhs, policy=self.policy)
        velocity = self.velocity_space.unflatten(solved.value[:nv])
        pressure = self.pressure_space.unflatten(solved.value[nv:])
        rate = self.thermal_rate(velocity, state.temperature_K)
        if rate.shape != state.temperature_K.shape:
            raise ValueError("Geodynamic thermal rate changed temperature shape.")
        temperature = state.temperature_K + dt * rate
        residual = operator.mv(solved.value) - rhs
        mechanical = residual[:nv]
        continuity = residual[nv:]
        thermal_residual = (temperature - state.temperature_K) / dt - rate
        successful = (
            solved.successful
            & jnp.all(jnp.isfinite(temperature))
            & jnp.all(temperature > 0)
        )
        candidate = GeodynamicsState(
            velocity, pressure, temperature, state.time_s + dt, self.plan_id
        )
        committed = jax.tree.map(
            lambda new, old: jnp.where(successful, new, old), candidate, state
        )
        return GeodynamicsStepResult(
            committed,
            continuity,
            mechanical,
            thermal_residual,
            successful,
            successful,
        )


__all__ = [
    "ArrheniusViscoplasticRheology",
    "GeodynamicsState",
    "GeodynamicsStepResult",
    "SphericalShellGeometry",
    "SphericalThermomechanicalPlan",
]
