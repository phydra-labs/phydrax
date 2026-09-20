#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCSpeciesPlan


class DSMCPairCollisionParameters(StrictModule, NonTrainableState):
    """Explicit symmetric VHS pair data; no implicit species mixing."""

    reference_diameters: Array
    viscosity_exponents: Array
    reference_temperature: float = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        reference_diameters: ArrayLike,
        viscosity_exponents: ArrayLike,
        /,
        *,
        reference_temperature: float = 273.15,
        boltzmann_constant: float = 1.380649e-23,
    ) -> None:
        diameters = np.asarray(reference_diameters, dtype=np.float64)
        exponents = np.asarray(viscosity_exponents, dtype=np.float64)
        temperature = float(reference_temperature)
        boltzmann = float(boltzmann_constant)
        if (
            diameters.ndim != 2
            or diameters.shape[0] != diameters.shape[1]
            or exponents.shape != diameters.shape
            or np.any(~np.isfinite(diameters))
            or np.any(diameters <= 0.0)
            or not np.allclose(diameters, diameters.T)
            or np.any(~np.isfinite(exponents))
            or np.any((exponents < 0.5) | (exponents > 1.0))
            or not np.allclose(exponents, exponents.T)
            or not np.isfinite(temperature)
            or temperature <= 0.0
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
        ):
            raise ValueError("DSMC pair collision parameters are invalid.")
        self.reference_diameters = jnp.asarray(diameters)
        self.viscosity_exponents = jnp.asarray(exponents)
        self.reference_temperature = temperature
        self.boltzmann_constant = boltzmann
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "dsmc-pair-collision-parameters",
                "reference_diameters": array_tree_fingerprint(diameters),
                "viscosity_exponents": array_tree_fingerprint(exponents),
                "reference_temperature": temperature,
                "boltzmann_constant": boltzmann,
            }
        )

    @property
    def species_count(self) -> int:
        return self.reference_diameters.shape[0]


class DSMCCollisionEventResult(StrictModule):
    state: DSMCParticleState
    accepted: Array
    acceptance_probability: Array
    sigma_speed: Array
    momentum_defect: Array
    energy_defect: Array
    majorant_violation: Array
    finite: Array


class DSMCCollisionResult(StrictModule):
    state: DSMCParticleState
    accepted: Array
    acceptance_probability: Array
    sigma_speed: Array
    momentum_defect: Array
    energy_defect: Array
    majorant_violation: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


@runtime_checkable
class DSMCElasticCollisionPlan(Protocol):
    species: DSMCSpeciesPlan
    plan_id: str

    def collision_cross_section(
        self, first_species: Array, second_species: Array, relative_speed: Array, /
    ) -> Array: ...

    def collide_one(
        self,
        state: DSMCParticleState,
        first_index: ArrayLike,
        second_index: ArrayLike,
        valid_event: ArrayLike,
        uniforms: ArrayLike,
        majorant_sigma_speed: ArrayLike,
        /,
    ) -> DSMCCollisionEventResult: ...


class DSMCVHSCollisionPlan(StrictModule, NonTrainableState):
    """Variable-hard-sphere elastic collision with isotropic scattering."""

    species: DSMCSpeciesPlan
    parameters: DSMCPairCollisionParameters
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        parameters: DSMCPairCollisionParameters,
        /,
    ) -> None:
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or not isinstance(parameters, DSMCPairCollisionParameters)
            or parameters.species_count != species.species_count
        ):
            raise ValueError("VHS species and pair parameters are incompatible.")
        self.species = species
        self.parameters = parameters
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-vhs-collision",
                "species": species.plan_id,
                "parameters": parameters.parameter_id,
            }
        )

    def collision_cross_section(
        self, first_species: Array, second_species: Array, relative_speed: Array, /
    ) -> Array:
        return _collision_cross_section(
            self.species,
            self.parameters,
            first_species,
            second_species,
            relative_speed,
        )

    def collide_one(
        self,
        state: DSMCParticleState,
        first_index: ArrayLike,
        second_index: ArrayLike,
        valid_event: ArrayLike,
        uniforms: ArrayLike,
        majorant_sigma_speed: ArrayLike,
        /,
    ) -> DSMCCollisionEventResult:
        return _collide_one(
            self,
            state,
            first_index,
            second_index,
            valid_event,
            uniforms,
            majorant_sigma_speed,
            jnp.asarray(1.0, dtype=state.velocity.dtype),
        )

    def collide(
        self,
        state: DSMCParticleState,
        first_indices: ArrayLike,
        second_indices: ArrayLike,
        valid_events: ArrayLike,
        uniforms: ArrayLike,
        majorant_sigma_speed: ArrayLike,
        /,
    ) -> DSMCCollisionResult:
        return _collide_many(
            self,
            state,
            first_indices,
            second_indices,
            valid_events,
            uniforms,
            majorant_sigma_speed,
        )


class DSMCVSSCollisionPlan(StrictModule, NonTrainableState):
    """Variable-soft-sphere elastic collision with the actual VSS angular law."""

    species: DSMCSpeciesPlan
    parameters: DSMCPairCollisionParameters
    scattering_parameters: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        parameters: DSMCPairCollisionParameters,
        scattering_parameters: ArrayLike,
        /,
    ) -> None:
        scattering = np.asarray(scattering_parameters, dtype=np.float64)
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or not isinstance(parameters, DSMCPairCollisionParameters)
            or parameters.species_count != species.species_count
            or scattering.shape != (species.species_count, species.species_count)
            or np.any(~np.isfinite(scattering))
            or np.any(scattering < 1.0)
            or not np.allclose(scattering, scattering.T)
        ):
            raise ValueError(
                "VSS species, pair parameters, or scattering data are invalid."
            )
        self.species = species
        self.parameters = parameters
        self.scattering_parameters = jnp.asarray(scattering)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-vss-collision",
                "species": species.plan_id,
                "parameters": parameters.parameter_id,
                "scattering": array_tree_fingerprint(scattering),
            }
        )

    def collision_cross_section(
        self, first_species: Array, second_species: Array, relative_speed: Array, /
    ) -> Array:
        return _collision_cross_section(
            self.species,
            self.parameters,
            first_species,
            second_species,
            relative_speed,
        )

    def collide_one(
        self,
        state: DSMCParticleState,
        first_index: ArrayLike,
        second_index: ArrayLike,
        valid_event: ArrayLike,
        uniforms: ArrayLike,
        majorant_sigma_speed: ArrayLike,
        /,
    ) -> DSMCCollisionEventResult:
        if state.velocity.shape[-1] != 3:
            raise ValueError(
                "VSS scattering requires three molecular-velocity components."
            )
        capacity = state.capacity
        first = jnp.asarray(first_index, dtype=jnp.int32)
        second = jnp.asarray(second_index, dtype=jnp.int32)
        safe_first = jnp.clip(first, 0, capacity - 1)
        safe_second = jnp.clip(second, 0, capacity - 1)
        alpha = self.scattering_parameters[
            state.species_index[safe_first], state.species_index[safe_second]
        ]
        return _collide_one(
            self,
            state,
            first,
            second,
            valid_event,
            uniforms,
            majorant_sigma_speed,
            alpha,
        )

    def collide(
        self,
        state: DSMCParticleState,
        first_indices: ArrayLike,
        second_indices: ArrayLike,
        valid_events: ArrayLike,
        uniforms: ArrayLike,
        majorant_sigma_speed: ArrayLike,
        /,
    ) -> DSMCCollisionResult:
        if state.velocity.shape[-1] != 3:
            raise ValueError(
                "VSS scattering requires three molecular-velocity components."
            )
        return _collide_many(
            self,
            state,
            first_indices,
            second_indices,
            valid_events,
            uniforms,
            majorant_sigma_speed,
        )


def _collision_cross_section(
    species: DSMCSpeciesPlan,
    parameters: DSMCPairCollisionParameters,
    first_species: Array,
    second_species: Array,
    relative_speed: Array,
    /,
) -> Array:
    first_mass = species.molecular_masses[first_species]
    second_mass = species.molecular_masses[second_species]
    reduced_mass = first_mass * second_mass / (first_mass + second_mass)
    diameter = parameters.reference_diameters[first_species, second_species]
    omega = parameters.viscosity_exponents[first_species, second_species]
    reference_speed = jnp.sqrt(
        2.0
        * parameters.boltzmann_constant
        * parameters.reference_temperature
        / reduced_mass
    )
    speed = jnp.maximum(relative_speed, jnp.finfo(relative_speed.dtype).tiny)
    return jnp.pi * diameter**2 * (reference_speed / speed) ** (2.0 * (omega - 0.5))


def _scattering_direction(
    relative: Array,
    uniforms: Array,
    scattering_parameter: Array,
    /,
) -> Array:
    dimension = relative.shape[0]
    dtype = relative.dtype
    if dimension == 1:
        sign = jnp.where(uniforms[0] < 0.5, -1.0, 1.0)
        return jnp.asarray((sign,), dtype=dtype)
    if dimension == 2:
        angle = 2.0 * jnp.pi * uniforms[0]
        return jnp.stack((jnp.cos(angle), jnp.sin(angle)))
    speed = jnp.sqrt(jnp.sum(relative * relative))
    incident = relative / jnp.maximum(speed, jnp.finfo(dtype).tiny)
    reference = jnp.where(
        jnp.abs(incident[0]) < 0.9,
        jnp.asarray((1.0, 0.0, 0.0), dtype=dtype),
        jnp.asarray((0.0, 1.0, 0.0), dtype=dtype),
    )
    tangent_one = jnp.cross(incident, reference)
    tangent_one = tangent_one / jnp.maximum(
        jnp.sqrt(jnp.sum(tangent_one * tangent_one)), jnp.finfo(dtype).tiny
    )
    tangent_two = jnp.cross(incident, tangent_one)
    cosine = 2.0 * uniforms[0] ** (1.0 / scattering_parameter) - 1.0
    sine = jnp.sqrt(jnp.maximum(1.0 - cosine * cosine, 0.0))
    azimuth = 2.0 * jnp.pi * uniforms[1]
    return (
        cosine * incident
        + sine * jnp.cos(azimuth) * tangent_one
        + sine * jnp.sin(azimuth) * tangent_two
    )


def _collide_one(
    plan: DSMCElasticCollisionPlan,
    state: DSMCParticleState,
    first_index: ArrayLike,
    second_index: ArrayLike,
    valid_event: ArrayLike,
    uniforms: ArrayLike,
    majorant_sigma_speed: ArrayLike,
    scattering_parameter: Array,
    /,
) -> DSMCCollisionEventResult:
    first = jnp.asarray(first_index, dtype=jnp.int32)
    second = jnp.asarray(second_index, dtype=jnp.int32)
    valid = jnp.asarray(valid_event, dtype=jnp.bool_)
    random = jnp.asarray(uniforms, dtype=state.velocity.dtype)
    majorant = jnp.asarray(majorant_sigma_speed, dtype=state.velocity.dtype)
    if first.shape != () or second.shape != () or valid.shape != ():
        raise ValueError("One DSMC collision event requires scalar indices and validity.")
    if random.shape != (3,) or majorant.shape != ():
        raise ValueError(
            "One DSMC collision event requires three uniforms and one majorant."
        )
    capacity = state.capacity
    in_bounds = (first >= 0) & (first < capacity) & (second >= 0) & (second < capacity)
    first_safe = jnp.clip(first, 0, capacity - 1)
    second_safe = jnp.clip(second, 0, capacity - 1)
    valid_pair = (
        valid
        & in_bounds
        & (first != second)
        & state.active[first_safe]
        & state.active[second_safe]
        & (state.cell_id[first_safe] == state.cell_id[second_safe])
    )
    first_velocity = state.velocity[first_safe]
    second_velocity = state.velocity[second_safe]
    first_species = state.species_index[first_safe]
    second_species = state.species_index[second_safe]
    relative = first_velocity - second_velocity
    relative_speed = jnp.sqrt(jnp.sum(relative * relative))
    cross_section = plan.collision_cross_section(
        first_species, second_species, relative_speed
    )
    sigma_speed = cross_section * relative_speed
    probability = sigma_speed / jnp.maximum(
        majorant, jnp.finfo(relative_speed.dtype).tiny
    )
    epsilon = 64.0 * jnp.finfo(probability.dtype).eps
    violation = valid_pair & (
        (~jnp.isfinite(majorant)) | (majorant <= 0.0) | (probability > 1.0 + epsilon)
    )
    accepted = valid_pair & ~violation & (random[0] < probability)
    from ..particle._elastic_scattering import scatter_elastic_pairs

    first_mass = plan.species.molecular_masses[first_species]
    second_mass = plan.species.molecular_masses[second_species]
    direction = _scattering_direction(relative, random[1:], scattering_parameter)
    scattering = scatter_elastic_pairs(
        first_velocity,
        second_velocity,
        first_mass,
        second_mass,
        direction,
        mask=accepted,
    )
    first_after = scattering.first_velocity
    second_after = scattering.second_velocity
    velocity = state.velocity.at[first_safe].set(first_after)
    velocity = velocity.at[second_safe].set(second_after)
    momentum_defect = scattering.momentum_defect
    energy_defect = scattering.kinetic_energy_defect
    updated = DSMCParticleState(
        state.position,
        velocity,
        state.species_index,
        state.rotational_energy,
        state.vibrational_energy,
        state.statistical_weight,
        state.cell_id,
        state.active,
        state.incarnation,
    )
    finite = (~valid) | (
        jnp.isfinite(probability) & scattering.finite & scattering.successful
    )
    return DSMCCollisionEventResult(
        updated,
        accepted,
        probability,
        sigma_speed,
        momentum_defect,
        energy_defect,
        violation,
        finite,
    )


def _collide_many(
    plan: DSMCElasticCollisionPlan,
    state: DSMCParticleState,
    first_indices: ArrayLike,
    second_indices: ArrayLike,
    valid_events: ArrayLike,
    uniforms: ArrayLike,
    majorant_sigma_speed: ArrayLike,
    /,
) -> DSMCCollisionResult:
    first = jnp.asarray(first_indices, dtype=jnp.int32)
    second = jnp.asarray(second_indices, dtype=jnp.int32)
    valid = jnp.asarray(valid_events, dtype=jnp.bool_)
    random = jnp.asarray(uniforms, dtype=state.velocity.dtype)
    majorant = jnp.asarray(majorant_sigma_speed, dtype=state.velocity.dtype)
    event_count = first.size
    if (
        first.ndim != 1
        or second.shape != first.shape
        or valid.shape != first.shape
        or random.shape != (event_count, 3)
        or majorant.shape not in ((), (event_count,))
    ):
        raise ValueError("DSMC collision event arrays are incompatible.")
    majorant = jnp.broadcast_to(majorant, (event_count,))

    def body(particles, event):
        first_, second_, valid_, random_, majorant_ = event
        result = plan.collide_one(
            particles,
            first_,
            second_,
            valid_,
            random_,
            majorant_,
        )
        diagnostics = (
            result.accepted,
            result.acceptance_probability,
            result.sigma_speed,
            result.momentum_defect,
            result.energy_defect,
            result.majorant_violation,
            result.finite,
        )
        return result.state, diagnostics

    updated, diagnostics = jax.lax.scan(
        body,
        state,
        (first, second, valid, random, majorant),
    )
    accepted, probability, sigma_speed, momentum, energy, violation, finite = diagnostics
    energy_scale = jnp.maximum(jnp.max(jnp.abs(energy)), 1.0)
    tolerance = 512.0 * jnp.finfo(energy.dtype).eps * energy_scale
    successful = (
        jnp.all(finite) & ~jnp.any(violation) & jnp.all(jnp.abs(energy) <= tolerance)
    )
    return DSMCCollisionResult(
        updated,
        accepted,
        probability,
        sigma_speed,
        momentum,
        energy,
        violation,
        finite,
        successful,
        plan.plan_id,
    )


__all__ = [
    "DSMCCollisionEventResult",
    "DSMCCollisionResult",
    "DSMCElasticCollisionPlan",
    "DSMCPairCollisionParameters",
    "DSMCVHSCollisionPlan",
    "DSMCVSSCollisionPlan",
]
