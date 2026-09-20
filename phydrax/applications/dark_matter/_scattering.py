#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle import scatter_elastic_pairs
from ...qualification import ReferenceArtifactManifest


_BOLTZMANN_J_K = 1.380649e-23


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _requested_use(
    commercial_use: bool,
    redistribution: bool,
    training_use: bool,
    export: bool,
    /,
) -> tuple[tuple[str, bool], ...]:
    values = (commercial_use, redistribution, training_use, export)
    if any(not isinstance(value, bool) for value in values):
        raise TypeError("Requested-use flags must be booleans.")
    return (
        ("commercial_use", commercial_use),
        ("redistribution", redistribution),
        ("training_use", training_use),
        ("export", export),
    )


def _external_array(value: ArrayLike, /) -> Array:
    return jax.lax.stop_gradient(jnp.asarray(value, dtype=jnp.float64))


def _linear_interpolate(grid: Array, values: Array, coordinate: Array, /) -> Array:
    index = jnp.clip(
        jnp.searchsorted(grid, coordinate, side="right") - 1,
        0,
        grid.shape[0] - 2,
    )
    left = grid[index]
    fraction = jnp.clip(
        (coordinate - left) / (grid[index + 1] - left),
        0.0,
        1.0,
    )
    return values[..., index] + fraction * (values[..., index + 1] - values[..., index])


class ElasticCollisionResult(StrictModule):
    """Nonrelativistic two-body elastic result and invariant evidence in SI units."""

    projectile_velocity_m_s: Array
    target_velocity_m_s: Array
    relative_speed_m_s: Array
    momentum_before_kg_m_s: Array
    momentum_after_kg_m_s: Array
    kinetic_energy_before_J: Array
    kinetic_energy_after_J: Array
    momentum_residual_kg_m_s: Array
    energy_residual_J: Array
    scattered: Array
    mass_valid: Array
    direction_valid: Array
    conservative: Array
    finite: Array
    successful: Array


def elastic_scatter_velocity(
    projectile_velocity_m_s: ArrayLike,
    target_velocity_m_s: ArrayLike,
    outgoing_relative_direction: ArrayLike,
    projectile_mass_kg: ArrayLike,
    target_mass_kg: ArrayLike,
    /,
) -> ElasticCollisionResult:
    """Apply one elastic mark through the native particle-pair primitive."""
    projectile = jnp.asarray(projectile_velocity_m_s, dtype=jnp.float64)
    target = jnp.asarray(target_velocity_m_s, dtype=projectile.dtype)
    direction = jnp.asarray(outgoing_relative_direction, dtype=projectile.dtype)
    if projectile.shape != (3,) or target.shape != (3,) or direction.shape != (3,):
        raise ValueError(
            "Elastic scattering velocities and direction must have shape (3,)."
        )
    native = scatter_elastic_pairs(
        projectile,
        target,
        jnp.asarray(projectile_mass_kg, dtype=projectile.dtype),
        jnp.asarray(target_mass_kg, dtype=projectile.dtype),
        direction,
    )
    return ElasticCollisionResult(
        native.first_velocity,
        native.second_velocity,
        native.relative_speed,
        native.momentum_before,
        native.momentum_after,
        native.kinetic_energy_before,
        native.kinetic_energy_after,
        native.momentum_defect,
        native.kinetic_energy_defect,
        native.scattered,
        native.mass_valid,
        native.direction_valid,
        native.conservative,
        native.finite,
        native.successful,
    )


class BoundedThermalMarkSamplerPlan(StrictModule, NonTrainableState):
    """Certified fixed-capacity rejection sampler for thermal collision partners."""

    thermal_sigma_cutoff: float = eqx.field(static=True)
    maximum_proposals: int = eqx.field(static=True)
    maxwellian_tail_probability_bound: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        thermal_sigma_cutoff: float = 6.0,
        maximum_proposals: int = 128,
    ):
        cutoff = float(thermal_sigma_cutoff)
        proposals = int(maximum_proposals)
        if not math.isfinite(cutoff) or cutoff <= 0.0 or proposals <= 0:
            raise ValueError(
                "Thermal mark cutoff and proposal capacity must be positive."
            )
        tail = math.erfc(cutoff / math.sqrt(2.0)) + math.sqrt(
            2.0 / math.pi
        ) * cutoff * math.exp(-0.5 * cutoff**2)
        self.thermal_sigma_cutoff = cutoff
        self.maximum_proposals = proposals
        self.maxwellian_tail_probability_bound = tail
        self.plan_id = canonical_fingerprint(
            {
                "kind": "bounded-thermal-mark-sampler",
                "thermal_sigma_cutoff": cutoff,
                "maximum_proposals": proposals,
                "maxwellian_tail_probability_bound": tail,
            }
        )


class ElasticScatteringTable(StrictModule, NonTrainableState):
    """Manifest-qualified elastic cross sections and thermal rate table.

    Cross sections have shape ``(target, speed)`` and thermally averaged rate
    coefficients have shape ``(target, temperature, speed)`` in m3/s. Marks use
    a certified truncated-Maxwellian rejection sampler weighted by
    ``sigma(v_rel) v_rel``. Requests outside declared support return NaN.
    """

    target_masses_kg: Array
    relative_speeds_m_s: Array
    cross_sections_m2: Array
    temperatures_K: Array
    rate_coefficients_m3_s: Array
    target_ids: tuple[str, ...] = eqx.field(static=True)
    source: ReferenceArtifactManifest
    mark_sampler: BoundedThermalMarkSamplerPlan
    requested_use: tuple[tuple[str, bool], ...] = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_ids: Sequence[str],
        target_masses_kg: ArrayLike,
        relative_speeds_m_s: ArrayLike,
        cross_sections_m2: ArrayLike,
        source: ReferenceArtifactManifest,
        /,
        *,
        temperatures_K: ArrayLike,
        rate_coefficients_m3_s: ArrayLike,
        mark_sampler: BoundedThermalMarkSamplerPlan,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        targets = tuple(_identifier(value, "target ID") for value in target_ids)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("target_ids must be non-empty and unique.")
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError("Scattering tables require a ReferenceArtifactManifest.")
        if not isinstance(mark_sampler, BoundedThermalMarkSamplerPlan):
            raise TypeError("mark_sampler must be a BoundedThermalMarkSamplerPlan.")
        requested = _requested_use(commercial_use, redistribution, training_use, export)
        source.require_rights(**dict(requested))
        convention = dict(source.nondimensionalization)
        cutoff_record = convention.get("thermal_sigma_cutoff")
        tail_record = convention.get("maxwellian_tail_probability_bound")
        if (
            cutoff_record is None
            or tail_record is None
            or not np.isclose(
                cutoff_record,
                mark_sampler.thermal_sigma_cutoff,
                rtol=0.0,
                atol=1.0e-14,
            )
            or not np.isclose(
                tail_record,
                mark_sampler.maxwellian_tail_probability_bound,
                rtol=1.0e-12,
                atol=0.0,
            )
        ):
            raise ValueError(
                "Thermal rate manifest must bind the mark cutoff and tail policy."
            )
        masses = np.asarray(target_masses_kg, dtype=np.float64)
        speeds = np.asarray(relative_speeds_m_s, dtype=np.float64)
        cross_sections = np.asarray(cross_sections_m2, dtype=np.float64)
        if (
            masses.shape != (len(targets),)
            or speeds.ndim != 1
            or speeds.size < 2
            or cross_sections.shape != (len(targets), speeds.size)
            or np.any(~np.isfinite(masses))
            or np.any(masses <= 0.0)
            or np.any(~np.isfinite(speeds))
            or np.any(speeds < 0.0)
            or np.any(np.diff(speeds) <= 0.0)
            or np.any(~np.isfinite(cross_sections))
            or np.any(cross_sections < 0.0)
        ):
            raise ValueError(
                "Elastic masses, speed support, or cross sections are invalid."
            )
        temperatures = np.asarray(temperatures_K, dtype=np.float64)
        coefficients = np.asarray(rate_coefficients_m3_s, dtype=np.float64)
        if (
            temperatures.ndim != 1
            or temperatures.size < 2
            or coefficients.shape != (len(targets), temperatures.size, speeds.size)
            or np.any(~np.isfinite(temperatures))
            or np.any(temperatures <= 0.0)
            or np.any(np.diff(temperatures) <= 0.0)
            or np.any(~np.isfinite(coefficients))
            or np.any(coefficients < 0.0)
        ):
            raise ValueError("Thermal rate coefficient support is invalid.")
        if speeds[0] != 0.0:
            raise ValueError(
                "Thermal mark certification requires speed support from zero."
            )
        self.target_ids = targets
        self.target_masses_kg = _external_array(masses)
        self.relative_speeds_m_s = _external_array(speeds)
        self.cross_sections_m2 = _external_array(cross_sections)
        self.temperatures_K = _external_array(temperatures)
        self.rate_coefficients_m3_s = _external_array(coefficients)
        self.mark_sampler = mark_sampler
        self.requested_use = requested
        self.source = source
        self.table_id = canonical_fingerprint(
            {
                "kind": "elastic-scattering-table",
                "target_ids": targets,
                "source": source.manifest_id,
                "mark_sampler": mark_sampler.plan_id,
                "requested_use": requested,
                "arrays": array_tree_fingerprint(
                    {
                        "target_masses_kg": self.target_masses_kg,
                        "relative_speeds_m_s": self.relative_speeds_m_s,
                        "cross_sections_m2": self.cross_sections_m2,
                        "temperatures_K": self.temperatures_K,
                        "rate_coefficients_m3_s": self.rate_coefficients_m3_s,
                    }
                ),
            }
        )

    @property
    def num_targets(self) -> int:
        return len(self.target_ids)

    def support(self, speed_m_s: ArrayLike, temperature_K: ArrayLike, /) -> Array:
        """Return per-target joint rate-table and bounded-mark support."""
        speed = jnp.asarray(speed_m_s, dtype=self.target_masses_kg.dtype).reshape(())
        temperature = jnp.asarray(temperature_K, dtype=speed.dtype).reshape(())
        thermal_std = jnp.sqrt(_BOLTZMANN_J_K * temperature / self.target_masses_kg)
        maximum_relative_speed = (
            speed + self.mark_sampler.thermal_sigma_cutoff * thermal_std
        )
        return (
            jnp.isfinite(speed)
            & (speed >= self.relative_speeds_m_s[0])
            & jnp.isfinite(temperature)
            & (temperature >= self.temperatures_K[0])
            & (temperature <= self.temperatures_K[-1])
            & jnp.isfinite(maximum_relative_speed)
            & (maximum_relative_speed <= self.relative_speeds_m_s[-1])
        )

    def mark_support(
        self,
        projectile_velocity_m_s: ArrayLike,
        temperature_K: ArrayLike,
        /,
    ) -> Array:
        """Return per-target certification for the bounded conditional sampler."""
        velocity = jnp.asarray(projectile_velocity_m_s, dtype=self.target_masses_kg.dtype)
        temperature = jnp.asarray(temperature_K, dtype=velocity.dtype).reshape(())
        if velocity.shape != (3,):
            raise ValueError("projectile_velocity_m_s must have shape (3,).")
        projectile_speed = jnp.sqrt(jnp.sum(velocity * velocity))
        return self.support(projectile_speed, temperature) & (
            jnp.max(self.cross_sections_m2, axis=-1) > 0.0
        )

    def cross_sections(self, relative_speed_m_s: ArrayLike, /) -> Array:
        speed = jnp.asarray(relative_speed_m_s, dtype=self.relative_speeds_m_s.dtype)
        if speed.shape != ():
            raise ValueError("relative_speed_m_s must be scalar.")
        values = _linear_interpolate(
            self.relative_speeds_m_s, self.cross_sections_m2, speed
        )
        supported = (
            jnp.isfinite(speed)
            & (speed >= self.relative_speeds_m_s[0])
            & (speed <= self.relative_speeds_m_s[-1])
        )
        return jnp.where(supported, values, jnp.nan)

    def _rate_values(self, speed: Array, temperature: Array, /) -> Array:
        speed_interpolated = _linear_interpolate(
            self.relative_speeds_m_s, self.rate_coefficients_m3_s, speed
        )
        return _linear_interpolate(self.temperatures_K, speed_interpolated, temperature)

    def rate_coefficients(
        self, projectile_speed_m_s: ArrayLike, temperature_K: ArrayLike, /
    ) -> Array:
        """Return thermally averaged per-target coefficients ``<sigma v>`` in m3/s."""
        speed = jnp.asarray(
            projectile_speed_m_s, dtype=self.target_masses_kg.dtype
        ).reshape(())
        temperature = jnp.asarray(temperature_K, dtype=speed.dtype).reshape(())
        values = self._rate_values(speed, temperature)
        return jnp.where(self.support(speed, temperature), values, jnp.nan)

    def partial_rates(
        self,
        target_number_densities_m3: ArrayLike,
        projectile_velocity_m_s: ArrayLike,
        temperature_K: ArrayLike,
        /,
    ) -> Array:
        number = jnp.asarray(
            target_number_densities_m3, dtype=self.target_masses_kg.dtype
        )
        velocity = jnp.asarray(projectile_velocity_m_s, dtype=number.dtype)
        if number.shape != (self.num_targets,) or velocity.shape != (3,):
            raise ValueError(
                "Partial rates require target densities (T,) and velocity (3,)."
            )
        speed = jnp.sqrt(ein.contract("i,i->", velocity, velocity))
        temperature = jnp.asarray(temperature_K, dtype=speed.dtype).reshape(())
        positive_density = number > 0.0
        raw_coefficients = self._rate_values(speed, temperature)
        table_supported = (
            jnp.isfinite(speed)
            & (speed >= self.relative_speeds_m_s[0])
            & (speed <= self.relative_speeds_m_s[-1])
            & jnp.isfinite(temperature)
            & (temperature >= self.temperatures_K[0])
            & (temperature <= self.temperatures_K[-1])
        )
        zero_rate = table_supported & (raw_coefficients == 0.0)
        requires_mark = positive_density & ~zero_rate
        coefficients = self.rate_coefficients(speed, temperature)
        safe_coefficients = jnp.where(requires_mark, coefficients, 0.0)
        rates = number * safe_coefficients
        mark_supported = self.mark_support(velocity, temperature)
        return jnp.where(~requires_mark | mark_supported, rates, jnp.nan)

    def sample_mark(
        self,
        key: Array,
        channel: ArrayLike,
        projectile_velocity_m_s: ArrayLike,
        temperature_K: ArrayLike,
        /,
    ) -> Array:
        """Sample from bounded ``f(v_t) sigma(v_rel) v_rel / Gamma``."""
        channel_ = jnp.asarray(channel, dtype=jnp.int32).reshape(())
        projectile = jnp.asarray(
            projectile_velocity_m_s, dtype=self.target_masses_kg.dtype
        )
        temperature = jnp.asarray(temperature_K, dtype=projectile.dtype).reshape(())
        if projectile.shape != (3,):
            raise ValueError("projectile_velocity_m_s must have shape (3,).")
        proposal_key, uniform_key, direction_key = jr.split(key, 3)
        thermal_std = jnp.sqrt(
            _BOLTZMANN_J_K * temperature / self.target_masses_kg[channel_]
        )
        proposals = thermal_std * jr.normal(
            proposal_key,
            (self.mark_sampler.maximum_proposals, 3),
            dtype=temperature.dtype,
        )
        relative = projectile[None, :] - proposals
        relative_speed = jnp.sqrt(jnp.sum(relative * relative, axis=-1))
        target_speed = jnp.sqrt(jnp.sum(proposals * proposals, axis=-1))
        cross_sections = jax.vmap(
            lambda speed: _linear_interpolate(
                self.relative_speeds_m_s,
                self.cross_sections_m2[channel_],
                speed,
            )
        )(relative_speed)
        projectile_speed = jnp.sqrt(jnp.sum(projectile * projectile))
        maximum_relative_speed = (
            projectile_speed + self.mark_sampler.thermal_sigma_cutoff * thermal_std
        )
        envelope = jnp.max(self.cross_sections_m2[channel_]) * maximum_relative_speed
        certified = self.mark_support(projectile, temperature)[channel_] & (
            envelope > 0.0
        )
        eligible = (
            target_speed <= self.mark_sampler.thermal_sigma_cutoff * thermal_std
        ) & (relative_speed <= self.relative_speeds_m_s[-1])
        uniforms = jr.uniform(
            uniform_key,
            (self.mark_sampler.maximum_proposals,),
            dtype=temperature.dtype,
        )
        accepted = eligible & (uniforms * envelope <= cross_sections * relative_speed)
        found = certified & jnp.any(accepted)
        selected = jnp.argmax(accepted.astype(jnp.int32))
        target_velocity = jnp.where(found, proposals[selected], jnp.nan)
        raw_direction = jr.normal(direction_key, (3,), dtype=temperature.dtype)
        norm = jnp.sqrt(ein.contract("i,i->", raw_direction, raw_direction))
        direction = raw_direction / jnp.maximum(norm, jnp.finfo(norm.dtype).tiny)
        return jnp.concatenate((target_velocity, direction))


__all__ = [
    "BoundedThermalMarkSamplerPlan",
    "ElasticCollisionResult",
    "ElasticScatteringTable",
    "elastic_scatter_velocity",
]
