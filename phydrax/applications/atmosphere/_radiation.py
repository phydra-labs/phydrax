#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Grey LTE longwave and hemispheric two-stream diffuse shortwave transfer.

Layers and interfaces are top-to-bottom. Optical masses are kg/m²; temperature
is K; every flux and transfer is W/m². This is a declared grey reference model,
not a spectral atmospheric radiation package or a direct-solar-beam solver.
See ``docs/guides_column_radiation.md`` for equations and reference assumptions.
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
from ...ein import contract


_STEFAN_BOLTZMANN = 5.670374419e-8  # W m^-2 K^-4, SI blackbody constant.


class ColumnOpticalProperties(StrictModule, NonTrainableState):
    """Fixed, explicitly tagged grey mass coefficients, in m²/kg of each species.

    Every vector is ordered ``(dry, vapor, liquid, ice)``. Dry means the entire
    non-water gas mixture: its coefficient is not a line-by-line CO₂ model.
    There are no empirical default opacities. ``reference_id`` identifies the
    user's source/calibration declaration, not an implied validation claim.
    Asymmetry is the shortwave scattering cosine moment, in [-1, 1]. Longwave
    scattering is neglected; condensate longwave coefficients are absorption.
    Liquid/ice optical coefficients describe suspended cloud only. Falling
    rain/snow are explicitly optically transparent, not counted as dry gas.
    """

    shortwave_absorption: Array
    shortwave_scattering: Array
    shortwave_asymmetry: Array
    longwave_absorption: Array
    reference_id: str = eqx.field(static=True)
    precipitation_optics: str = eqx.field(static=True)
    optics_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        shortwave_absorption: ArrayLike,
        shortwave_scattering: ArrayLike,
        longwave_absorption: ArrayLike,
        reference_id: str,
        shortwave_asymmetry: ArrayLike = (0.0, 0.0, 0.0, 0.0),
    ):
        arrays = {
            "shortwave_absorption": np.asarray(shortwave_absorption, dtype=float),
            "shortwave_scattering": np.asarray(shortwave_scattering, dtype=float),
            "longwave_absorption": np.asarray(longwave_absorption, dtype=float),
            "shortwave_asymmetry": np.asarray(shortwave_asymmetry, dtype=float),
        }
        if not isinstance(reference_id, str) or not reference_id.strip():
            raise ValueError("Optical properties require a nonempty reference_id.")
        for name, value in arrays.items():
            if value.shape != (4,) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"{name} must be a finite (dry, vapor, liquid, ice) vector."
                )
            if name == "shortwave_asymmetry":
                if np.any(np.abs(value) > 1.0):
                    raise ValueError("Shortwave asymmetry must lie in [-1, 1].")
            elif np.any(value < 0.0):
                raise ValueError(
                    "Mass absorption/scattering coefficients must be nonnegative."
                )
        self.shortwave_absorption = jnp.asarray(arrays["shortwave_absorption"])
        self.shortwave_scattering = jnp.asarray(arrays["shortwave_scattering"])
        self.longwave_absorption = jnp.asarray(arrays["longwave_absorption"])
        self.shortwave_asymmetry = jnp.asarray(arrays["shortwave_asymmetry"])
        self.reference_id = reference_id
        self.precipitation_optics = "transparent"
        self.optics_id = canonical_fingerprint(
            {
                "kind": "grey-column-mass-optics",
                "species": ("dry", "vapor", "liquid", "ice"),
                "coefficient_units": "m2/kg",
                "reference_id": reference_id,
                "precipitation_optics": self.precipitation_optics,
                **{name: value.tolist() for name, value in arrays.items()},
            }
        )


class ColumnRadiationResult(StrictModule):
    """Positive directional flux magnitudes and signed reservoir transfers.

    ``successful`` is per batch column. Every numeric output is zero for a
    rejected column; consumers must check the flag before committing transfers.
    ``budget_residual`` is the unnormalized atmosphere + surface + space sum.
    """

    upward_flux: Array
    downward_flux: Array
    heating: Array
    surface_heating: Array
    space_heating: Array
    shortwave_upward_flux: Array
    shortwave_downward_flux: Array
    longwave_upward_flux: Array
    longwave_downward_flux: Array
    shortwave_absorption_depth: Array
    shortwave_transport_scattering_depth: Array
    longwave_absorption_depth: Array
    budget_residual: Array
    successful: Array


def _hemispheric_slab(absorption: Array, scattering: Array):
    """Reflection, transmission, absorptance of a homogeneous SW slab.

    ``scattering`` already includes (1-g). The exact exponential solution uses
    a=2*absorption+scattering, b=scattering, q=a²-b². Even power series in q
    remove the sqrt singularity at conservative scattering/transparent limits.
    Scaled exponentials avoid the growing modes of transfer matrices.
    """
    a = 2.0 * absorption + scattering
    q = 4.0 * absorption * (absorption + scattering)
    small = q < 1.0e-4
    qs = jnp.where(small, q, 0.0)
    sinhc = 1.0 + qs * (1.0 / 6.0 + qs * (1.0 / 120.0 + qs / 5040.0))
    coshm1 = qs * (0.5 + qs * (1.0 / 24.0 + qs / 720.0))
    denominator_small = 1.0 + coshm1 + a * sinhc
    reflection_small = scattering * sinhc / denominator_small
    transmission_small = 1.0 / denominator_small
    absorption_small = (coshm1 + 2.0 * absorption * sinhc) / denominator_small

    k = jnp.sqrt(jnp.where(small, 1.0, q))
    decay = jnp.exp(-k)
    scaled_sinhc = -jnp.expm1(-2.0 * k) / (2.0 * k)
    denominator = 0.5 * (1.0 + decay * decay) + a * scaled_sinhc
    reflection = scattering * scaled_sinhc / denominator
    transmission = decay / denominator
    absorbed = (0.5 * jnp.expm1(-k) ** 2 + 2.0 * absorption * scaled_sinhc) / denominator
    return (
        jnp.where(small, reflection_small, reflection),
        jnp.where(small, transmission_small, transmission),
        jnp.where(small, absorption_small, absorbed),
    )


def _shortwave_fluxes(absorption, scattering, albedo, incident):
    reflect, transmit, absorb = _hemispheric_slab(absorption, scattering)

    def add_layer(lower, slab):
        # Retain both albedo and its complement independently: 1-R loses all
        # precision in a thick conservative slab over a perfectly reflecting surface.
        r, c = lower
        reflection, transmission, absorbed = slab
        loss = transmission + absorbed
        denominator = c + r * loss
        next_r = reflection + transmission * (transmission * r / denominator)
        next_c = (loss * c + r * absorbed * (loss + transmission)) / denominator
        return (next_r, next_c), (next_r, denominator)

    _, (stack_reflection, denominators) = jax.lax.scan(
        add_layer,
        (albedo, 1.0 - albedo),
        tuple(jnp.moveaxis(x, -1, 0) for x in (reflect, transmit, absorb)),
        reverse=True,
    )

    def descend(downward, slab):
        transmission, denominator = slab
        next_downward = downward * (transmission / denominator)
        return next_downward, next_downward

    _, down = jax.lax.scan(
        descend, incident, (jnp.moveaxis(transmit, -1, 0), denominators)
    )
    down = jnp.concatenate((incident[..., None], jnp.moveaxis(down, 0, -1)), axis=-1)
    stack_reflection = jnp.concatenate(
        (jnp.moveaxis(stack_reflection, 0, -1), albedo[..., None]), axis=-1
    )
    return stack_reflection * down, down


def _longwave_fluxes(absorption, temperature, surface_temperature, emissivity, incident):
    transmission = jnp.exp(-2.0 * absorption)
    emission = -jnp.expm1(-2.0 * absorption) * _STEFAN_BOLTZMANN * temperature**4

    def propagate(flux, layer):
        attenuation, source = layer
        next_flux = attenuation * flux + source
        return next_flux, next_flux

    layers = tuple(jnp.moveaxis(x, -1, 0) for x in (transmission, emission))
    surface_down, down = jax.lax.scan(propagate, incident, layers)
    surface_up = (
        emissivity * _STEFAN_BOLTZMANN * surface_temperature**4
        + (1.0 - emissivity) * surface_down
    )
    _, up = jax.lax.scan(propagate, surface_up, layers, reverse=True)
    return (
        jnp.concatenate((jnp.moveaxis(up, 0, -1), surface_up[..., None]), axis=-1),
        jnp.concatenate((incident[..., None], jnp.moveaxis(down, 0, -1)), axis=-1),
    )


class ColumnRadiationPlan(StrictModule):
    """Native differentiable grey column transfer with explicitly fixed optics.

    Numeric calibration scales and surface parameters are trainable array leaves;
    ``optics`` is excluded by native ``partition_trainable``. Scales are scalar
    or four-species vectors; albedo/emissivity may carry broadcast batch axes.
    ``plan_id`` identifies immutable optics and closure, not current trainable
    parameter values. No hidden spectral lookup, unit conversion or provider runs.
    """

    optics: ColumnOpticalProperties
    shortwave_absorption_scale: Array
    shortwave_scattering_scale: Array
    longwave_absorption_scale: Array
    surface_albedo: Array
    surface_emissivity: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        optics: ColumnOpticalProperties,
        *,
        surface_albedo: ArrayLike = 0.1,
        surface_emissivity: ArrayLike = 1.0,
        shortwave_absorption_scale: ArrayLike = 1.0,
        shortwave_scattering_scale: ArrayLike = 1.0,
        longwave_absorption_scale: ArrayLike = 1.0,
    ):
        scales = tuple(
            np.asarray(x, dtype=float)
            for x in (
                shortwave_absorption_scale,
                shortwave_scattering_scale,
                longwave_absorption_scale,
            )
        )
        for value in scales:
            if value.shape not in ((), (4,)) or not np.all(
                np.isfinite(value) & (value >= 0)
            ):
                raise ValueError(
                    "Optical scales must be finite nonnegative scalars or four-species vectors."
                )
        albedo, emissivity = (
            np.asarray(surface_albedo, dtype=float),
            np.asarray(surface_emissivity, dtype=float),
        )
        for value in (albedo, emissivity):
            if not np.all(np.isfinite(value) & (value >= 0.0) & (value <= 1.0)):
                raise ValueError("Surface albedo and emissivity must lie in [0, 1].")
        self.optics = optics
        self.shortwave_absorption_scale = jnp.asarray(np.broadcast_to(scales[0], (4,)))
        self.shortwave_scattering_scale = jnp.asarray(np.broadcast_to(scales[1], (4,)))
        self.longwave_absorption_scale = jnp.asarray(np.broadcast_to(scales[2], (4,)))
        self.surface_albedo = jnp.asarray(albedo)
        self.surface_emissivity = jnp.asarray(emissivity)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hemispheric-grey-column-radiation",
                "optics_id": optics.optics_id,
                "layer_order": "top-to-bottom",
                "shortwave_boundary": "diffuse-hemispheric",
                "longwave": "grey-LTE-absorption-isothermal-layer",
                "diffusivity": 2.0,
                "stefan_boltzmann": _STEFAN_BOLTZMANN,
            }
        )

    def evaluate(
        self,
        temperature: ArrayLike,
        layer_mass: ArrayLike,
        vapor_mass: ArrayLike,
        liquid_mass: ArrayLike,
        ice_mass: ArrayLike,
        surface_temperature: ArrayLike,
        solar_down: ArrayLike,
        *,
        longwave_down: ArrayLike = 0.0,
        rain_mass: ArrayLike = 0.0,
        snow_mass: ArrayLike = 0.0,
    ) -> ColumnRadiationResult:
        """Evaluate top-to-bottom interface fluxes, atomically per batch column.

        Layer arrays have shape ``[..., n]`` with n >= 1 and common n; leading
        axes broadcast with boundary arrays. Masses include all condensate;
        dry mass subtracts vapor, cloud liquid/ice, and falling rain/snow.
        Rain/snow are optically transparent and may be scalar or layer arrays.
        Zero-mass transparent layers are permitted. ``solar_down`` is a diffuse
        downward TOA flux, not beam-normal irradiance. ``longwave_down`` is an
        explicit optional TOA thermal boundary (zero for empty space).
        """
        layers = tuple(
            jnp.asarray(x)
            for x in (temperature, layer_mass, vapor_mass, liquid_mass, ice_mass)
        )
        precipitation = tuple(jnp.asarray(x) for x in (rain_mass, snow_mass))
        boundaries = tuple(
            jnp.asarray(x)
            for x in (
                surface_temperature,
                solar_down,
                longwave_down,
                self.surface_albedo,
                self.surface_emissivity,
            )
        )
        dtype = jnp.result_type(
            *layers, *precipitation, *boundaries, self.shortwave_absorption_scale, float
        )
        layers = tuple(x.astype(dtype) for x in layers)
        precipitation = tuple(x.astype(dtype) for x in precipitation)
        boundaries = tuple(x.astype(dtype) for x in boundaries)
        if any(x.ndim < 1 for x in layers):
            raise ValueError("Radiation thermodynamic inputs require a final layer axis.")
        n = layers[0].shape[-1]
        if n < 1 or any(x.shape[-1] != n for x in layers):
            raise ValueError("Radiation layer axes must have a common positive length.")
        if any(x.ndim > 0 and x.shape[-1] != n for x in precipitation):
            raise ValueError(
                "Precipitation masses must be scalars or match the layer axis."
            )
        batch = jnp.broadcast_shapes(
            *(x.shape[:-1] for x in (*layers, *precipitation)),
            *(x.shape for x in boundaries),
        )
        t, mass, vapor, liquid, ice = (jnp.broadcast_to(x, batch + (n,)) for x in layers)
        rain, snow = (jnp.broadcast_to(x, batch + (n,)) for x in precipitation)
        ts, solar, thermal, albedo, emissivity = (
            jnp.broadcast_to(x, batch) for x in boundaries
        )
        dry = mass - (vapor + liquid + ice + rain + snow)
        valid = jnp.all(jnp.isfinite(t) & (t > 0.0), axis=-1)
        for value in (mass, vapor, liquid, ice, rain, snow, dry):
            valid = valid & jnp.all(jnp.isfinite(value) & (value >= 0.0), axis=-1)
        valid = valid & jnp.isfinite(ts) & (ts > 0.0)
        for value in (solar, thermal):
            valid = valid & jnp.isfinite(value) & (value >= 0.0)
        for value in (albedo, emissivity):
            valid = valid & jnp.isfinite(value) & (value >= 0.0) & (value <= 1.0)
        coefficients = (
            self.optics.shortwave_absorption * self.shortwave_absorption_scale,
            self.optics.shortwave_scattering * self.shortwave_scattering_scale,
            self.optics.longwave_absorption * self.longwave_absorption_scale,
        )
        for value in (
            self.shortwave_absorption_scale,
            self.shortwave_scattering_scale,
            self.longwave_absorption_scale,
            self.optics.shortwave_absorption,
            self.optics.shortwave_scattering,
            self.optics.longwave_absorption,
            *coefficients,
        ):
            valid = valid & jnp.all(jnp.isfinite(value) & (value >= 0.0))
        asymmetry = self.optics.shortwave_asymmetry
        valid = valid & jnp.all(jnp.isfinite(asymmetry) & (jnp.abs(asymmetry) <= 1.0))

        # Rejected lanes use safe operands, not repaired physical inputs. They
        # remain unsuccessful and all their outputs are discarded together.
        species = jnp.stack((dry, vapor, liquid, ice), axis=-1)
        species = jnp.where(valid[..., None, None], species, 0.0)
        safe_coefficients = tuple(
            jnp.where(jnp.isfinite(x) & (x >= 0.0), x, 0.0) for x in coefficients
        )
        asymmetry = jnp.where(
            jnp.isfinite(asymmetry) & (jnp.abs(asymmetry) <= 1.0), asymmetry, 0.0
        )
        sw_abs = contract("...ns,s->...n", species, safe_coefficients[0])
        sw_scat = contract(
            "...ns,s->...n", species, safe_coefficients[1] * (1.0 - asymmetry)
        )
        lw_abs = contract("...ns,s->...n", species, safe_coefficients[2])
        valid = valid & jnp.all(
            jnp.isfinite(sw_abs) & jnp.isfinite(sw_scat) & jnp.isfinite(lw_abs), axis=-1
        )
        sw_abs, sw_scat, lw_abs = (
            jnp.where(valid[..., None], x, 0.0) for x in (sw_abs, sw_scat, lw_abs)
        )
        t = jnp.where(valid[..., None], t, 1.0)
        ts, solar, thermal, albedo, emissivity = (
            jnp.where(valid, x, 0.0) for x in (ts, solar, thermal, albedo, emissivity)
        )
        sw_up, sw_down = _shortwave_fluxes(sw_abs, sw_scat, albedo, solar)
        lw_up, lw_down = _longwave_fluxes(lw_abs, t, ts, emissivity, thermal)
        up, down = sw_up + lw_up, sw_down + lw_down
        net_up = up - down
        heating = net_up[..., 1:] - net_up[..., :-1]
        surface_heating, space_heating = -net_up[..., -1], net_up[..., 0]
        residual = jnp.sum(heating, axis=-1) + surface_heating + space_heating
        for value in (sw_up, sw_down, lw_up, lw_down, up, down, heating):
            valid = valid & jnp.all(jnp.isfinite(value), axis=-1)
        valid = valid & jnp.isfinite(residual)
        return ColumnRadiationResult(
            *(jnp.where(valid[..., None], x, 0.0) for x in (up, down, heating)),
            jnp.where(valid, surface_heating, 0.0),
            jnp.where(valid, space_heating, 0.0),
            *(
                jnp.where(valid[..., None], x, 0.0)
                for x in (
                    sw_up,
                    sw_down,
                    lw_up,
                    lw_down,
                    sw_abs,
                    sw_scat,
                    lw_abs,
                )
            ),
            jnp.where(valid, residual, 0.0),
            valid,
        )


__all__ = ["ColumnOpticalProperties", "ColumnRadiationPlan", "ColumnRadiationResult"]
