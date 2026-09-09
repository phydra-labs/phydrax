# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""An independently derived dry gradient/thermal-wind reference on flat ground.

This log-pressure family is not the Jablonowski--Williamson published test.
Continuous balance is exact; finite-layer hydrostatics and modal projection are
not exact and their errors are measured separately, never removed by a fixer.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._global import GlobalAtmosphereContinuation, PreparedGlobalAtmosphere


class DryGradientWindFields(StrictModule):
    east: Array
    temperature: Array
    geopotential: Array
    successful: Array


class DryBalanceDiagnostics(StrictModule):
    """Dimensional errors; no total-reference-energy normalization."""

    surface_pressure_projection_pa: Array
    temperature_projection_kelvin: Array
    wind_projection_m_per_s: Array
    hydrostatic_geopotential_error_m2_per_s2: Array
    acceleration_rms_m_per_s2: Array
    acceleration_max_m_per_s2: Array
    temperature_tendency_max_kelvin_per_s: Array
    surface_pressure_tendency_max_pa_per_s: Array


class DryGradientWindReference(StrictModule, NonTrainableState):
    """Immutable analytic reference with a declared pressure-validity interval.

    With x=log(p/p_ref), q=sin(latitude)**2 and U=U0+s*x,
    u=U*cos(latitude), Phi=-R*T0*x-(Omega*a*U+U**2/2)*q and
    T=T0+s*(Omega*a+U)*q/R. All quantities are SI, latitude is radians.
    The positive-temperature root Phi(ps,latitude)=0 fixes flat-ground ps.
    The family is admitted only when statically stable throughout the declared
    pressure interval, including the poles, not just sampled grid points.
    """

    speed: float = eqx.field(static=True)
    shear: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    reference_pressure: float = eqx.field(static=True)
    minimum_pressure: float = eqx.field(static=True)
    maximum_pressure: float = eqx.field(static=True)
    radius: float = eqx.field(static=True)
    rotation_rate: float = eqx.field(static=True)
    gas_constant: float = eqx.field(static=True)
    heat_capacity: float = eqx.field(static=True)
    reference_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        speed=20.0,
        shear=-10.0,
        temperature=288.0,
        reference_pressure=100000.0,
        minimum_pressure=1000.0,
        maximum_pressure=120000.0,
        radius=6.371e6,
        rotation_rate=7.292115e-5,
        gas_constant=287.05,
        heat_capacity=1004.0,
    ):
        parameters = dict(
            speed=float(speed),
            shear=float(shear),
            temperature=float(temperature),
            reference_pressure=float(reference_pressure),
            minimum_pressure=float(minimum_pressure),
            maximum_pressure=float(maximum_pressure),
            radius=float(radius),
            rotation_rate=float(rotation_rate),
            gas_constant=float(gas_constant),
            heat_capacity=float(heat_capacity),
        )
        if not all(np.isfinite(v) for v in parameters.values()):
            raise ValueError("Balanced reference parameters must be finite.")
        if (
            min(temperature, reference_pressure, minimum_pressure, radius, gas_constant)
            <= 0
            or maximum_pressure <= minimum_pressure
            or heat_capacity <= gas_constant
        ):
            raise ValueError(
                "Require positive scales, increasing pressure bounds and cp > R."
            )
        # T and kappa*T-dT/dlog(p) are bilinear in (log(p/p_ref), sin²(latitude)).
        # Their extrema on the rectangular validity domain occur at the corners.
        x = np.log(np.array([minimum_pressure, maximum_pressure]) / reference_pressure)
        q = np.array([0.0, 1.0])[:, None]
        t = (
            temperature
            + shear * (rotation_rate * radius + speed + shear * x) * q / gas_constant
        )
        if np.any((t <= 100.0) | (t >= 500.0)):
            raise ValueError("Analytic temperature leaves the native 100--500 K domain.")
        if np.any(gas_constant / heat_capacity * t - shear**2 * q / gas_constant <= 0):
            raise ValueError(
                "Reference must be statically stable over its pressure domain."
            )
        # The equator-connected quadratic root must exist at every latitude.
        b0 = gas_constant * temperature
        b1 = shear * (rotation_rate * radius + speed)
        c1 = rotation_rate * radius * speed + 0.5 * speed**2
        d2 = (shear * rotation_rate * radius) ** 2
        candidates = [0.0, 1.0]
        if d2 > 0:
            vertex = -b0 * b1 / d2
            if 0 < vertex < 1:
                candidates.append(vertex)
        discriminants = [b0**2 + 2 * b0 * b1 * z + d2 * z**2 for z in candidates]
        if min(b0, b0 + b1) <= 0 or min(discriminants) <= 0:
            raise ValueError("No regular positive-temperature surface-pressure branch.")
        polar_x = -2 * c1 / (b0 + b1 + np.sqrt(discriminants[1]))
        # ps is monotonic in sin²(latitude) on this positive-temperature branch.
        ps_endpoints = reference_pressure * np.exp(np.array([0.0, polar_x]))
        if np.any((ps_endpoints < minimum_pressure) | (ps_endpoints > maximum_pressure)):
            raise ValueError(
                "Surface pressure lies outside the declared pressure domain."
            )
        self.speed, self.shear, self.temperature = (
            parameters["speed"],
            parameters["shear"],
            parameters["temperature"],
        )
        self.reference_pressure = parameters["reference_pressure"]
        self.minimum_pressure, self.maximum_pressure = (
            parameters["minimum_pressure"],
            parameters["maximum_pressure"],
        )
        self.radius, self.rotation_rate = (
            parameters["radius"],
            parameters["rotation_rate"],
        )
        self.gas_constant, self.heat_capacity = (
            parameters["gas_constant"],
            parameters["heat_capacity"],
        )
        self.reference_id = canonical_fingerprint(
            {"kind": "derived-log-pressure-gradient-wind", **parameters}
        )

    def fields(self, pressure: ArrayLike, latitude: ArrayLike) -> DryGradientWindFields:
        """Evaluate without clipping; consumers must honor ``successful``."""
        p, latitude = jnp.broadcast_arrays(jnp.asarray(pressure), jnp.asarray(latitude))
        x, q = jnp.log(p / self.reference_pressure), jnp.sin(latitude) ** 2
        speed = self.speed + self.shear * x
        temperature = (
            self.temperature
            + self.shear
            * (self.rotation_rate * self.radius + speed)
            * q
            / self.gas_constant
        )
        geopotential = (
            -self.gas_constant * self.temperature * x
            - (self.rotation_rate * self.radius * speed + 0.5 * speed**2) * q
        )
        valid = (
            jnp.isfinite(p)
            & jnp.isfinite(latitude)
            & (p >= self.minimum_pressure)
            & (p <= self.maximum_pressure)
            & (jnp.abs(latitude) <= jnp.pi / 2)
        )
        return DryGradientWindFields(
            speed * jnp.cos(latitude), temperature, geopotential, jnp.all(valid)
        )

    def surface_pressure(self, latitude: ArrayLike) -> Array:
        """Cancellation-free equator-connected root; no shear/latitude division."""
        q = jnp.sin(jnp.asarray(latitude)) ** 2
        b = (
            self.gas_constant * self.temperature
            + self.shear * (self.rotation_rate * self.radius + self.speed) * q
        )
        c = (self.rotation_rate * self.radius * self.speed + 0.5 * self.speed**2) * q
        discriminant = b**2 - 2 * self.shear**2 * q * c
        x = -2 * c / (b + jnp.sqrt(discriminant))
        return self.reference_pressure * jnp.exp(x)

    def balance_residuals(
        self, pressure: ArrayLike, latitude: ArrayLike
    ) -> tuple[Array, Array, Array]:
        """Hydrostatic, gradient-wind, thermal-wind identities from AD derivatives.

        Units are respectively J/kg, m/s², m/s². The gradient and thermal-wind
        equations use their regular cos(latitude) forms even at the poles.
        Finite differences are a separate independent qualification check.
        """
        p, lat = jnp.broadcast_arrays(jnp.asarray(pressure), jnp.asarray(latitude))
        fields = self.fields(p, lat)
        dx_phi = jax.jvp(
            lambda x: self.fields(self.reference_pressure * jnp.exp(x), lat).geopotential,
            (jnp.log(p / self.reference_pressure),),
            (jnp.ones_like(p),),
        )[1]
        dlat_phi = jax.jvp(
            lambda z: self.fields(p, z).geopotential, (lat,), (jnp.ones_like(lat),)
        )[1]
        dlat_t = jax.jvp(
            lambda z: self.fields(p, z).temperature, (lat,), (jnp.ones_like(lat),)
        )[1]
        speed = self.speed + self.shear * jnp.log(p / self.reference_pressure)
        rotational = 2 * self.rotation_rate * jnp.sin(lat) * fields.east
        curvature = speed**2 * jnp.sin(lat) * jnp.cos(lat) / self.radius
        thermal = (
            2
            * self.shear
            * (self.rotation_rate + speed / self.radius)
            * jnp.sin(lat)
            * jnp.cos(lat)
        )
        return (
            dx_phi + self.gas_constant * fields.temperature,
            dlat_phi / self.radius + rotational + curvature,
            self.gas_constant / self.radius * dlat_t - thermal,
        )

    def initialize(
        self, model: PreparedGlobalAtmosphere, *, time=0.0
    ) -> GlobalAtmosphereContinuation:
        """Project continuous fields onto the native owner, not a discrete fixer."""
        if not isinstance(model, PreparedGlobalAtmosphere):
            raise TypeError("Balanced initialization requires PreparedGlobalAtmosphere.")
        if model.plan.space.layout.bandlimit < 3:
            raise ValueError(
                "Balanced family requires bandlimit >= 3 for its degree-two thermal field."
            )
        if np.any(np.asarray(model.plan.terrain) != 0):
            raise ValueError("This balanced family requires exactly flat ground.")
        if model.plan.processes.active:
            raise ValueError(
                "Balanced dry steady reference requires inactive dry processes."
            )
        if model.plan.filter_rate != 0:
            raise ValueError(
                "Balanced steady reference requires zero explicit filtering."
            )
        actual = (
            model.plan.space.radius,
            model.plan.rotation_rate,
            model.plan.gas_constant,
            model.plan.heat_capacity,
        )
        expected = (
            self.radius,
            self.rotation_rate,
            self.gas_constant,
            self.heat_capacity,
        )
        if actual != expected:
            raise ValueError("Balanced reference and model physical constants differ.")
        latitude = jnp.pi / 2 - model.work_space.transform.theta[:, None]
        ps = jnp.broadcast_to(
            self.surface_pressure(latitude), model.work_space.sample_shape
        )
        interfaces = model.plan.vertical.interfaces(ps)
        if (
            not bool(jnp.all(model.plan.vertical.valid(ps)))
            or float(jnp.min(interfaces)) < self.minimum_pressure
            or float(jnp.max(interfaces)) > self.maximum_pressure
        ):
            raise ValueError(
                "Model interfaces lie outside the admitted balanced pressure domain."
            )
        pressure = 0.5 * (interfaces[..., :-1] + interfaces[..., 1:])
        fields = self.fields(pressure, latitude[..., None])
        initial = model.initialize(
            temperature=fields.temperature,
            surface_pressure=ps,
            east=fields.east,
            time=time,
        )
        represented = model.plan.vertical.interfaces(
            model.reconstruct(initial.state.surface_pressure)
        )
        if (
            float(jnp.min(represented)) < self.minimum_pressure
            or float(jnp.max(represented)) > self.maximum_pressure
        ):
            raise ValueError(
                "Projected interfaces leave the balanced reference pressure domain."
            )
        return initial

    def diagnostics(
        self, model: PreparedGlobalAtmosphere, initial: GlobalAtmosphereContinuation
    ) -> DryBalanceDiagnostics:
        """Separate projection, finite-layer hydrostatics and actual PDE residual."""
        view = model.view(initial.state)
        latitude = jnp.pi / 2 - model.work_space.transform.theta[:, None]
        ps = jnp.broadcast_to(
            self.surface_pressure(latitude), model.work_space.sample_shape
        )
        interfaces = model.plan.vertical.interfaces(ps)
        source = self.fields(
            0.5 * (interfaces[..., :-1] + interfaces[..., 1:]), latitude[..., None]
        )
        at_represented_pressure = self.fields(view.pressure, latitude[..., None])
        rate, _, _, _ = model.tendency(initial.state, initial.held_forcing)
        east, north = model.vectors.wind(
            model.lift(rate.vorticity), model.lift(rate.divergence)
        )
        acceleration2 = east**2 + north**2
        rms = jnp.sqrt(
            jnp.sum(model.work_space.integral(view.layer_mass * acceleration2))
            / jnp.sum(model.work_space.integral(view.layer_mass))
        )
        return DryBalanceDiagnostics(
            jnp.max(jnp.abs(view.surface_pressure - ps)),
            jnp.max(jnp.abs(view.temperature - source.temperature)),
            jnp.max(jnp.hypot(view.east - source.east, view.north)),
            jnp.max(jnp.abs(view.geopotential - at_represented_pressure.geopotential)),
            rms,
            jnp.sqrt(jnp.max(acceleration2)),
            jnp.max(jnp.abs(model.reconstruct(rate.temperature))),
            jnp.max(jnp.abs(model.reconstruct(rate.surface_pressure))),
        )


__all__ = ["DryBalanceDiagnostics", "DryGradientWindFields", "DryGradientWindReference"]
