#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from phydrax.ein import contract

from .._admissibility import (
    AdmissibilityHeader,
    AdmissibilityReason,
    AdmissibilityTransitionRequest,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.dsmc._core import DSMCParticleState, DSMCSpeciesPlan


class ContinuumDSMCReason(IntFlag):
    INTERFACE_CONSERVATION_FAILED = 1 << 8
    CONVERSION_MOMENT_FAILED = 1 << 9
    REDUCTION_FAILED = 1 << 10
    OWNERSHIP_CAPACITY_EXCEEDED = 1 << 11
    TRANSITION_CONVERSION_FAILED = 1 << 12
    TRANSITION_REDUCTION_FAILED = 1 << 13


class ContinuumDSMCConservedSchema(StrictModule, NonTrainableState):
    """Ordered conserved quantities shared by continuum and particle regions."""

    component_names: tuple[str, ...] = eqx.field(static=True)
    unit_ids: tuple[str, ...] = eqx.field(static=True)
    frame: str = eqx.field(static=True)
    energy_reference: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_names: Sequence[str],
        unit_ids: Sequence[str],
        /,
        *,
        frame: str,
        energy_reference: str,
    ) -> None:
        names = tuple(str(value) for value in component_names)
        units = tuple(str(value) for value in unit_ids)
        frame_ = str(frame)
        reference = str(energy_reference)
        if (
            not names
            or len(names) != len(units)
            or len(set(names)) != len(names)
            or any(not value or value != value.strip() for value in (*names, *units))
            or not frame_
            or not reference
        ):
            raise ValueError("Continuum-DSMC conserved schema is invalid.")
        self.component_names = names
        self.unit_ids = units
        self.frame = frame_
        self.energy_reference = reference
        self.schema_id = canonical_fingerprint(
            {
                "kind": "continuum-dsmc-conserved-schema",
                "components": names,
                "units": units,
                "frame": frame_,
                "energy_reference": reference,
            }
        )

    @property
    def component_count(self) -> int:
        return len(self.component_names)


class ContinuumDSMCInterfaceExchange(StrictModule):
    kinetic_flux: Array
    kinetic_covariance: Array
    extensive_exchange: Array
    continuum_exchange: Array
    kinetic_exchange: Array
    conservation_defect: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class ContinuumDSMCInterfacePlan(StrictModule, NonTrainableState):
    """Use measured DSMC crossing flux as the sole conservative interface flux."""

    schema: ContinuumDSMCConservedSchema
    face_measures: Array
    tolerance: float = eqx.field(static=True)
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        schema: ContinuumDSMCConservedSchema,
        face_measures: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-12,
        interface_id: str | None = None,
    ) -> None:
        measures = np.asarray(face_measures, dtype=float)
        tolerance_ = float(tolerance)
        if (
            not isinstance(schema, ContinuumDSMCConservedSchema)
            or measures.ndim != 1
            or measures.size == 0
            or np.any(~np.isfinite(measures))
            or np.any(measures <= 0.0)
            or not np.isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Continuum-DSMC interface measure or tolerance is invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "continuum-dsmc-interface",
                "schema": schema.schema_id,
                "face_measures": tuple(float(value) for value in measures),
                "tolerance": tolerance_,
                "closure": "kinetic-crossing-flux",
            }
        )
        identity = generated if interface_id is None else str(interface_id)
        if not identity:
            raise ValueError("interface_id must be nonempty.")
        self.schema = schema
        self.face_measures = jnp.asarray(measures)
        self.tolerance = tolerance_
        self.interface_id = identity

    @property
    def face_count(self) -> int:
        return self.face_measures.shape[0]

    @property
    def component_count(self) -> int:
        return self.schema.component_count

    @property
    def plan_id(self) -> str:
        return self.interface_id

    def exchange(
        self,
        kinetic_flux: ArrayLike,
        kinetic_covariance: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> ContinuumDSMCInterfaceExchange:
        flux = jnp.asarray(kinetic_flux)
        covariance = jnp.asarray(kinetic_covariance, dtype=flux.dtype)
        step = jnp.asarray(step_size, dtype=flux.dtype)
        if (
            flux.shape != (self.face_count, self.component_count)
            or covariance.shape
            != (self.face_count, self.component_count, self.component_count)
            or step.shape != ()
        ):
            raise ValueError("Continuum-DSMC interface arrays have incompatible shapes.")
        extensive = step * self.face_measures[:, None].astype(flux.dtype) * flux
        continuum = -extensive
        kinetic = extensive
        defect = continuum + kinetic
        finite = (
            jnp.all(jnp.isfinite(flux))
            & jnp.all(jnp.isfinite(covariance))
            & jnp.isfinite(step)
        )
        covariance_valid = jnp.all(jnp.diagonal(covariance, axis1=-2, axis2=-1) >= 0.0)
        scale = jnp.maximum(jnp.max(jnp.abs(extensive)), 1.0)
        conserved = jnp.max(jnp.abs(defect)) <= self.tolerance * scale
        supported = finite & covariance_valid & (step > 0.0) & conserved
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            covariance_valid & (step > 0.0),
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        reasons = jnp.where(
            conserved,
            reasons,
            reasons
            | jnp.asarray(
                int(ContinuumDSMCReason.INTERFACE_CONSERVATION_FAILED), jnp.uint32
            ),
        )
        header = AdmissibilityHeader(
            jnp.where(supported, 1.0, -1.0),
            reasons,
            self.interface_id,
            canonical_fingerprint(
                {
                    "kind": "continuum-dsmc-interface-evidence",
                    "interface": self.interface_id,
                }
            ),
        )
        return ContinuumDSMCInterfaceExchange(
            flux,
            covariance,
            extensive,
            continuum,
            kinetic,
            defect,
            header,
            self.interface_id,
        )


class ContinuumToDSMCConversionResult(StrictModule):
    velocities: Array
    statistical_weights: Array
    represented_mass: Array
    represented_momentum: Array
    represented_thermal_energy: Array
    mass_defect: Array
    momentum_defect: Array
    thermal_energy_defect: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class ContinuumToDSMCConversionPlan(StrictModule, NonTrainableState):
    """Single-species conversion with exact represented mass, momentum, and energy."""

    particle_count: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, particle_count: int, dimension: int, /) -> None:
        count = int(particle_count)
        dimension_ = int(dimension)
        if count <= dimension_ or dimension_ not in (1, 2, 3):
            raise ValueError("Continuum-to-DSMC conversion capacity is invalid.")
        self.particle_count = count
        self.dimension = dimension_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "continuum-to-dsmc-conversion",
                "particle_count": count,
                "dimension": dimension_,
                "support": "single-species-translational-equilibrium",
            }
        )

    def convert(
        self,
        key: PRNGKeyArray,
        total_mass: ArrayLike,
        mean_velocity: ArrayLike,
        translational_thermal_energy: ArrayLike,
        molecular_mass: ArrayLike,
        /,
    ) -> ContinuumToDSMCConversionResult:
        mass = jnp.asarray(total_mass)
        mean = jnp.asarray(mean_velocity, dtype=mass.dtype)
        thermal = jnp.asarray(translational_thermal_energy, dtype=mass.dtype)
        molecule = jnp.asarray(molecular_mass, dtype=mass.dtype)
        if (
            jax.random.key_data(key).shape != (2,)
            or mass.shape != ()
            or mean.shape != (self.dimension,)
            or thermal.shape != ()
            or molecule.shape != ()
        ):
            raise ValueError("Continuum-to-DSMC conversion inputs are incompatible.")
        draws = jax.random.normal(
            key, (self.particle_count, self.dimension), dtype=mass.dtype
        )
        centered = draws - jnp.mean(draws, axis=0)
        raw_norm = jnp.sum(centered**2)
        scaled = centered * jnp.sqrt(
            2.0
            * thermal
            / jnp.maximum(mass * raw_norm, jnp.finfo(mass.dtype).tiny)
            * self.particle_count
        )
        velocity = mean + scaled
        statistical_weight = mass / (self.particle_count * molecule)
        weights = jnp.full((self.particle_count,), statistical_weight)
        represented_mass = self.particle_count * statistical_weight * molecule
        represented_momentum = statistical_weight * molecule * jnp.sum(velocity, axis=0)
        represented_thermal = (
            0.5 * statistical_weight * molecule * jnp.sum((velocity - mean) ** 2)
        )
        mass_defect = represented_mass - mass
        momentum_defect = represented_momentum - mass * mean
        thermal_defect = represented_thermal - thermal
        finite = (
            jnp.all(jnp.isfinite(velocity))
            & jnp.isfinite(statistical_weight)
            & jnp.isfinite(thermal_defect)
        )
        supported = finite & (mass > 0.0) & (thermal > 0.0) & (molecule > 0.0)
        scale = jnp.maximum(jnp.maximum(jnp.abs(mass), jnp.abs(thermal)), 1.0)
        tolerance = 1024.0 * jnp.finfo(mass.dtype).eps * scale
        matched = (
            (jnp.abs(mass_defect) <= tolerance)
            & (jnp.max(jnp.abs(momentum_defect)) <= tolerance)
            & (jnp.abs(thermal_defect) <= tolerance)
        )
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            supported,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.OUTSIDE_SUPPORT), jnp.uint32),
        )
        reasons = jnp.where(
            matched,
            reasons,
            reasons
            | jnp.asarray(int(ContinuumDSMCReason.CONVERSION_MOMENT_FAILED), jnp.uint32),
        )
        header = AdmissibilityHeader(
            jnp.where(supported & matched, 1.0, -1.0),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "continuum-to-dsmc-evidence", "plan": self.plan_id}
            ),
        )
        return ContinuumToDSMCConversionResult(
            velocity,
            weights,
            represented_mass,
            represented_momentum,
            represented_thermal,
            mass_defect,
            momentum_defect,
            thermal_defect,
            header,
            self.plan_id,
        )


class DSMCToContinuumReductionResult(StrictModule):
    species_mass: Array
    momentum: Array
    total_energy: Array
    covariance: Array
    active_particles: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DSMCToContinuumReductionPlan(StrictModule, NonTrainableState):
    species: DSMCSpeciesPlan
    dimension: int = eqx.field(static=True)
    minimum_particles: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        dimension: int,
        /,
        *,
        minimum_particles: int = 2,
    ) -> None:
        dimension_ = int(dimension)
        minimum = int(minimum_particles)
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or dimension_ not in (1, 2, 3)
            or minimum < 1
        ):
            raise ValueError("DSMC-to-continuum reduction plan is invalid.")
        self.species = species
        self.dimension = dimension_
        self.minimum_particles = minimum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-to-continuum-reduction",
                "species": species.plan_id,
                "dimension": dimension_,
                "minimum_particles": minimum,
            }
        )

    def reduce(
        self,
        particles: DSMCParticleState,
        region_mask: ArrayLike | None = None,
        /,
    ) -> DSMCToContinuumReductionResult:
        if particles.velocity.shape[-1] != self.dimension:
            raise ValueError("DSMC particle velocity dimension does not match reduction.")
        region = (
            jnp.ones((particles.capacity,), dtype=bool)
            if region_mask is None
            else jnp.asarray(region_mask, dtype=bool)
        )
        if region.shape != (particles.capacity,):
            raise ValueError("DSMC reduction mask must match particle capacity.")
        safe_species = jnp.clip(
            particles.species_index, 0, self.species.species_count - 1
        )
        valid = (
            particles.active
            & region
            & (particles.species_index >= 0)
            & (particles.species_index < self.species.species_count)
            & (particles.statistical_weight > 0.0)
            & jnp.all(jnp.isfinite(particles.velocity), axis=-1)
        )
        weight = jnp.where(valid, particles.statistical_weight, 0.0)
        molecular_mass = self.species.molecular_masses[safe_species]
        represented_mass = weight * molecular_mass
        species_mass = (
            jnp.zeros((self.species.species_count,), dtype=weight.dtype)
            .at[safe_species]
            .add(represented_mass)
        )
        momentum = jnp.sum(represented_mass[:, None] * particles.velocity, axis=0)
        particle_energy = weight * (
            0.5 * molecular_mass * jnp.sum(particles.velocity**2, axis=-1)
            + particles.rotational_energy
            + particles.vibrational_energy
        )
        total_energy = jnp.sum(jnp.where(valid, particle_energy, 0.0))
        features = jnp.concatenate(
            (
                jax.nn.one_hot(safe_species, self.species.species_count)
                * represented_mass[:, None],
                represented_mass[:, None] * particles.velocity,
                particle_energy[:, None],
            ),
            axis=-1,
        )
        safe_features = jnp.where(valid[:, None], features, 0.0)
        count = jnp.sum(valid)
        mean = jnp.sum(safe_features, axis=0) / jnp.maximum(count, 1)
        centered = jnp.where(valid[:, None], safe_features - mean, 0.0)
        covariance = contract(
            "pi,pj->ij", centered, centered, backend="jax"
        ) / jnp.maximum(count - 1, 1)
        finite = (
            jnp.all(jnp.isfinite(species_mass))
            & jnp.all(jnp.isfinite(momentum))
            & jnp.isfinite(total_energy)
            & jnp.all(jnp.isfinite(covariance))
        )
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            count >= self.minimum_particles,
            reasons,
            reasons
            | jnp.asarray(int(AdmissibilityReason.UNCERTAINTY_UNRESOLVED), jnp.uint32),
        )
        header = AdmissibilityHeader(
            (count - self.minimum_particles).astype(weight.dtype),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dsmc-to-continuum-evidence", "plan": self.plan_id}
            ),
        )
        return DSMCToContinuumReductionResult(
            species_mass,
            momentum,
            total_energy,
            covariance,
            count,
            header,
            self.plan_id,
        )


class HybridOwnershipEpochState(StrictModule):
    kinetic_mask: Array
    dwell_steps: Array
    epoch: Array
    policy_id: str = eqx.field(static=True)


class HybridOwnershipRequest(StrictModule):
    candidate_mask: Array
    candidate_dwell_steps: Array
    entered: Array
    left: Array
    required_particles: Array
    capacity_available: Array
    transition: AdmissibilityTransitionRequest
    header: AdmissibilityHeader
    policy_id: str = eqx.field(static=True)


class HybridOwnershipEpochPlan(StrictModule, NonTrainableState):
    """Pure classification plus host-only ownership epoch activation."""

    enter_threshold: float = eqx.field(static=True)
    leave_threshold: float = eqx.field(static=True)
    minimum_dwell_steps: int = eqx.field(static=True)
    buffer_layers: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        enter_threshold: float,
        leave_threshold: float,
        minimum_dwell_steps: int = 4,
        buffer_layers: int = 1,
    ) -> None:
        enter = float(enter_threshold)
        leave = float(leave_threshold)
        dwell = int(minimum_dwell_steps)
        buffers = int(buffer_layers)
        if (
            not np.isfinite(enter)
            or not np.isfinite(leave)
            or not 0.0 <= leave < enter
            or dwell < 0
            or buffers < 0
        ):
            raise ValueError("Hybrid ownership thresholds, dwell, or buffer are invalid.")
        self.enter_threshold = enter
        self.leave_threshold = leave
        self.minimum_dwell_steps = dwell
        self.buffer_layers = buffers
        self.policy_id = canonical_fingerprint(
            {
                "kind": "hybrid-ownership-epoch",
                "enter_threshold": enter,
                "leave_threshold": leave,
                "minimum_dwell_steps": dwell,
                "buffer_layers": buffers,
            }
        )

    def initialize(self, kinetic_mask: ArrayLike, /) -> HybridOwnershipEpochState:
        mask = jnp.asarray(kinetic_mask, dtype=bool)
        return HybridOwnershipEpochState(
            mask,
            jnp.zeros(mask.shape, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            self.policy_id,
        )

    def classify(
        self,
        state: HybridOwnershipEpochState,
        breakdown_metric: ArrayLike,
        breakdown_header: AdmissibilityHeader,
        adjacency: ArrayLike,
        particles_per_new_cell: int,
        particle_capacity_available: ArrayLike,
        /,
    ) -> HybridOwnershipRequest:
        metric = jnp.asarray(breakdown_metric)
        adjacency_ = jnp.asarray(adjacency, dtype=bool)
        available = jnp.asarray(particle_capacity_available, dtype=jnp.int32)
        per_cell = int(particles_per_new_cell)
        if (
            not isinstance(state, HybridOwnershipEpochState)
            or state.policy_id != self.policy_id
            or not isinstance(breakdown_header, AdmissibilityHeader)
            or metric.shape != state.kinetic_mask.shape
            or breakdown_header.eligible.shape != metric.shape
            or adjacency_.shape != (metric.size, metric.size)
            or available.shape != ()
            or per_cell <= 0
        ):
            raise ValueError("Hybrid ownership state or evidence is incompatible.")
        flattened = state.kinetic_mask.reshape((-1,))
        dwell = state.dwell_steps.reshape((-1,))
        requested = jnp.where(
            metric.reshape((-1,)) >= self.enter_threshold,
            True,
            jnp.where(
                (metric.reshape((-1,)) <= self.leave_threshold)
                & (dwell >= self.minimum_dwell_steps),
                False,
                flattened,
            ),
        )
        buffered = requested
        for _ in range(self.buffer_layers):
            buffered = buffered | jnp.any(adjacency_ & buffered[None, :], axis=1)
        entered = buffered & ~flattened
        left = flattened & ~buffered
        required = jnp.sum(entered) * per_cell
        capacity_ok = available >= required
        finite = jnp.all(jnp.isfinite(metric)) & breakdown_header.globally_eligible
        reasons = breakdown_header.reason_bits.reshape(metric.shape)
        reasons = jnp.where(
            capacity_ok,
            reasons,
            reasons
            | jnp.asarray(
                int(ContinuumDSMCReason.OWNERSHIP_CAPACITY_EXCEEDED), jnp.uint32
            ),
        )
        margin = jnp.minimum(
            breakdown_header.margin,
            (available - required).astype(metric.dtype),
        )
        header = AdmissibilityHeader(
            jnp.where(finite, margin, -jnp.inf),
            reasons,
            self.policy_id,
            canonical_fingerprint(
                {"kind": "hybrid-ownership-request-evidence", "plan": self.policy_id}
            ),
        )
        candidate_mask = buffered.reshape(state.kinetic_mask.shape)
        candidate_dwell = jnp.where(
            candidate_mask == state.kinetic_mask,
            state.dwell_steps + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )
        transition = AdmissibilityTransitionRequest(
            (entered | left).reshape(state.kinetic_mask.shape),
            state.epoch + jnp.asarray(1, dtype=jnp.int32),
            header,
            "continuum-dsmc-accepted-epoch",
            "continuum-dsmc-candidate-epoch",
        )
        return HybridOwnershipRequest(
            candidate_mask,
            candidate_dwell,
            entered.reshape(state.kinetic_mask.shape),
            left.reshape(state.kinetic_mask.shape),
            required,
            available,
            transition,
            header,
            self.policy_id,
        )

    def transition_epoch(
        self,
        state: HybridOwnershipEpochState,
        request: HybridOwnershipRequest,
        conversion_evidence: AdmissibilityHeader,
        reduction_evidence: AdmissibilityHeader,
        /,
    ) -> HybridOwnershipEpochState:
        if (
            not isinstance(state, HybridOwnershipEpochState)
            or state.policy_id != self.policy_id
            or not isinstance(request, HybridOwnershipRequest)
            or request.policy_id != self.policy_id
            or not isinstance(conversion_evidence, AdmissibilityHeader)
            or not isinstance(reduction_evidence, AdmissibilityHeader)
        ):
            raise ValueError("Hybrid ownership transition inputs are incompatible.")
        entered = bool(np.any(np.asarray(request.entered)))
        left = bool(np.any(np.asarray(request.left)))
        request_ok = bool(np.asarray(request.header.globally_eligible))
        conversion_ok = bool(np.asarray(conversion_evidence.globally_eligible))
        reduction_ok = bool(np.asarray(reduction_evidence.globally_eligible))
        if (
            not request_ok
            or (entered and not conversion_ok)
            or (left and not reduction_ok)
        ):
            raise ValueError(
                "Hybrid ownership cannot change without admitted conversion and reduction."
            )
        return HybridOwnershipEpochState(
            request.candidate_mask,
            request.candidate_dwell_steps,
            state.epoch + jnp.asarray(1, dtype=jnp.int32),
            self.policy_id,
        )


__all__ = [
    "ContinuumDSMCConservedSchema",
    "ContinuumDSMCInterfaceExchange",
    "ContinuumDSMCInterfacePlan",
    "ContinuumDSMCReason",
    "ContinuumToDSMCConversionPlan",
    "ContinuumToDSMCConversionResult",
    "DSMCToContinuumReductionPlan",
    "DSMCToContinuumReductionResult",
    "HybridOwnershipEpochPlan",
    "HybridOwnershipEpochState",
    "HybridOwnershipRequest",
]
