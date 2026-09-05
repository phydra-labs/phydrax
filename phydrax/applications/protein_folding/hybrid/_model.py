#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nonperiodic reference-conditioned Cartesian/rigid conservative composition.

The elastic network conditions the protein on a supplied reference. Cross terms
are independently supplied numerical models, not sequence recognition, a DNA
force-field calibration, or a de novo folding potential. Preparation is host-only;
energy, force and split stepping differentiate on fixed admitted support.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....atomistic._dynamics import AtomisticKinematics
from ....atomistic._elastic_network import PreparedElasticNetwork
from ....atomistic._units import AtomisticUnitSystem
from ....discretization.particle._rigid_body import (
    _rigid_body_close_kick,
    _rigid_body_half_kick,
    rigid_body_drift,
    RigidBodyKinematics,
    RigidBodyLoad,
)
from ....qualification._reference import ReferenceArtifactManifest
from ...nucleic_acid_biophysics.coarse._mechanics import PreparedNucleotideModel


@dataclass(frozen=True, slots=True)
class HybridSupportMap:
    """Reversible, order-independent disjoint IDs without changing either owner.

    Each record is ``(support_kind, source_id, hybrid_id)``. DOFs and interaction
    sites have separate namespaces even when their source integers coincide.
    Padding retains an identity, but never gains active/material membership.
    """

    records: tuple[tuple[str, int, int], ...]

    def global_id(self, support_kind: str, source_id: int, /) -> int:
        for kind, source, target in self.records:
            if kind == support_kind and source == source_id:
                return target
        raise KeyError((support_kind, source_id))

    def source(self, hybrid_id: int, /) -> tuple[str, int]:
        for kind, source, target in self.records:
            if target == hybrid_id:
                return kind, source
        raise KeyError(hybrid_id)

    def fingerprint(self) -> str:
        return canonical_fingerprint(
            {"kind": "hybrid-support-map", "records": self.records}
        )


def _parameter(
    value: ArrayLike, count: int, name: str, *, signed: bool = False
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full((count,), array, dtype=np.float64)
    if array.shape != (count,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and scalar or cross-pair shaped.")
    if not signed and np.any(array < 0):
        raise ValueError(f"{name} must be nonnegative.")
    return array


class HybridCrossInteractionPlan(StrictModule, NonTrainableState):
    """Fixed sparse cross-site model with explicitly scaled independent terms.

    For each declared pair at distance r, the terms are
    ``epsilon * max(1-r/radius, 0)^4``, ``k/2 * (r-r0)^2`` and
    ``a * exp(-screening*r)/r``. Radius/r0 use length units, epsilon uses
    energy, k uses energy/length², a uses energy*length and screening uses
    inverse length. The signed electrostatic prefactor must already include
    the declared charges and dielectric convention; charges are not inferred.
    Zero coefficients disable a term. Coincident coupled sites are refused
    by numeric success evidence, not regularized into a calibrated interaction.
    """

    site_pairs: Array
    steric_energy: Array
    steric_radius: Array
    linker_stiffness: Array
    linker_length: Array
    electrostatic_prefactor: Array
    screening: Array
    units: AtomisticUnitSystem
    parameter_source: ReferenceArtifactManifest
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        site_pairs: ArrayLike,
        units: AtomisticUnitSystem,
        parameter_source: ReferenceArtifactManifest,
        /,
        *,
        steric_energy: ArrayLike = 0.0,
        steric_radius: ArrayLike = 1.0,
        linker_stiffness: ArrayLike = 0.0,
        linker_length: ArrayLike = 0.0,
        electrostatic_prefactor: ArrayLike = 0.0,
        screening: ArrayLike = 0.0,
    ):
        pairs = np.asarray(site_pairs)
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu":
            raise TypeError(
                "site_pairs must be an integer (pairs, 2) array of stable site IDs."
            )
        if pairs.dtype.kind == "u" and np.any(pairs > np.iinfo(np.int64).max):
            raise ValueError("Cross-site IDs must fit signed int64.")
        pairs = pairs.astype(np.int64, copy=False)
        if np.unique(pairs, axis=0).shape[0] != pairs.shape[0]:
            raise ValueError(
                "Duplicate cross-site pairs would double-count an interaction."
            )
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        if not isinstance(parameter_source, ReferenceArtifactManifest):
            raise TypeError("parameter_source must be ReferenceArtifactManifest.")
        count = pairs.shape[0]
        epsilon = _parameter(steric_energy, count, "steric_energy")
        radius = _parameter(steric_radius, count, "steric_radius")
        stiffness = _parameter(linker_stiffness, count, "linker_stiffness")
        length = _parameter(linker_length, count, "linker_length")
        prefactor = _parameter(
            electrostatic_prefactor, count, "electrostatic_prefactor", signed=True
        )
        screening_ = _parameter(screening, count, "screening")
        if np.any(radius <= 0):
            raise ValueError("steric_radius must be positive.")
        arrays = {
            "site_pairs": pairs,
            "steric_energy": epsilon,
            "steric_radius": radius,
            "linker_stiffness": stiffness,
            "linker_length": length,
            "electrostatic_prefactor": prefactor,
            "screening": screening_,
        }
        self.site_pairs = jnp.asarray(pairs)
        self.steric_energy = jnp.asarray(epsilon)
        self.steric_radius = jnp.asarray(radius)
        self.linker_stiffness = jnp.asarray(stiffness)
        self.linker_length = jnp.asarray(length)
        self.electrostatic_prefactor = jnp.asarray(prefactor)
        self.screening = jnp.asarray(screening_)
        self.units = units
        self.parameter_source = parameter_source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "hybrid-cross-interaction-plan",
                "arrays": array_tree_fingerprint(arrays),
                "units": units.unit_system_id,
                "source": parameter_source.manifest_id,
            }
        )

    def components(self, displacement: Array, /) -> Array:
        squared = jnp.sum(displacement * displacement, axis=-1)
        distance = jnp.sqrt(jnp.where(squared > 0.0, squared, 1.0))
        overlap = jnp.maximum(1.0 - distance / self.steric_radius, 0.0)
        components = jnp.stack(
            (
                jnp.sum(self.steric_energy * overlap**4),
                jnp.sum(
                    0.5 * self.linker_stiffness * (distance - self.linker_length) ** 2
                ),
                jnp.sum(
                    self.electrostatic_prefactor
                    * jnp.exp(-self.screening * distance)
                    / distance
                ),
            )
        )
        coupled = (
            (self.steric_energy != 0)
            | (self.linker_stiffness != 0)
            | (self.electrostatic_prefactor != 0)
        )
        valid = jnp.all(~coupled | (squared > 0.0))
        return jnp.where(valid, components, jnp.nan)


class HybridState(StrictModule):
    """Distinct Cartesian momenta and rigid world twists at one physical time."""

    protein: AtomisticKinematics
    nucleotide: RigidBodyKinematics
    time: Array
    prepared_id: str = eqx.field(static=True)


class HybridForceEvaluation(StrictModule):
    """Energy-unit loads including full fixed-support reactions.

    ``components`` order is protein network, nucleotide model, cross steric,
    cross linker, cross electrostatic. Reaction forces are forces *on fixed
    material*, not the opposite external force needed to hold that material.
    """

    energy: Array
    components: Array
    protein_forces: Array
    protein_mobile_forces: Array
    protein_reaction_forces: Array
    nucleotide_load: RigidBodyLoad
    nucleotide_mobile_load: RigidBodyLoad
    nucleotide_reaction_load: RigidBodyLoad
    successful: Array


class HybridStepResult(StrictModule):
    """Candidate and acceptance evidence; a failed candidate is not a trajectory."""

    state: HybridState
    evaluation: HybridForceEvaluation
    total_energy: Array
    successful: Array


class PreparedHybridModel(StrictModule, NonTrainableState):
    """An elastic protein network plus a native rigid nucleotide model.

    No periodic image policy or Cartesian holonomic constraint solver is added
    here. Such supports are refused. The explicit kick/drift/kick split uses
    native rigid kernels and separate Cartesian momentum updates; anisotropic
    rotation inherits the native rigid integrator's finite-step accuracy, not
    an assertion of exact free-rotor flow or exact energy preservation.
    """

    protein_network: PreparedElasticNetwork
    nucleotide_model: PreparedNucleotideModel
    cross: HybridCrossInteractionPlan
    protein_reference: ReferenceArtifactManifest
    protein_site_indices: Array
    nucleotide_site_indices: Array
    support_map: HybridSupportMap = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        protein_network: PreparedElasticNetwork,
        nucleotide_model: PreparedNucleotideModel,
        cross: HybridCrossInteractionPlan,
        protein_reference: ReferenceArtifactManifest,
        /,
        *,
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        if not isinstance(protein_network, PreparedElasticNetwork):
            raise TypeError("protein_network must be PreparedElasticNetwork.")
        if not isinstance(nucleotide_model, PreparedNucleotideModel):
            raise TypeError("nucleotide_model must be PreparedNucleotideModel.")
        if not isinstance(cross, HybridCrossInteractionPlan):
            raise TypeError("cross must be HybridCrossInteractionPlan.")
        if not isinstance(protein_reference, ReferenceArtifactManifest):
            raise TypeError("protein_reference must be ReferenceArtifactManifest.")
        system = protein_network.system
        if system.cell is not None:
            raise ValueError(
                "Mixed mechanics currently requires nonperiodic protein support."
            )
        if nucleotide_model.cell is not None:
            raise ValueError(
                "Mixed mechanics currently requires nonperiodic nucleotide support."
            )
        if nucleotide_model.bodies.ambient_dimension != 3:
            raise ValueError(
                "Hybrid nucleotide support requires three-dimensional bodies."
            )
        if protein_network.preparation.reference_id != protein_reference.manifest_id:
            raise ValueError(
                "The elastic network reference must be pinned to protein_reference.manifest_id."
            )
        if system.topology.constraint_count:
            raise ValueError(
                "Mixed mechanics does not implement Cartesian holonomic constraints."
            )
        if (
            system.plan.units.unit_system_id != nucleotide_model.units.unit_system_id
            or system.plan.units.unit_system_id != cross.units.unit_system_id
        ):
            raise ValueError(
                "Protein, nucleotide and cross interactions must use the same exact "
                "unit system; convert inputs explicitly."
            )
        for source in (
            protein_reference,
            cross.parameter_source,
            nucleotide_model.parameter_manifest,
        ):
            source.require_rights(
                commercial_use=commercial_use,
                redistribution=redistribution,
                training_use=training_use,
                export=export,
            )
        sites = system.coordinate_map.plan.sites
        markers = nucleotide_model.marker_map.markers
        protein_ids = np.asarray(sites.site_ids, dtype=np.int64)
        nucleotide_ids = np.asarray(markers.plan.marker_ids, dtype=np.int64)
        physical_nucleotide_sites = np.asarray(
            markers.active_mask & nucleotide_model.physical_site_mask
        )
        protein_lookup = {
            int(key): slot
            for slot, key in enumerate(protein_ids)
            if bool(sites.active_mask[slot])
        }
        nucleotide_lookup = {
            int(key): slot
            for slot, key in enumerate(nucleotide_ids)
            if physical_nucleotide_sites[slot]
        }
        pairs = np.asarray(cross.site_pairs)
        if any(
            int(left) not in protein_lookup or int(right) not in nucleotide_lookup
            for left, right in pairs
        ):
            raise ValueError(
                "Cross pairs must name active stable sites on their declared physical "
                "support, not differential frame markers."
            )
        records = []
        supports = (
            ("protein-dof", np.asarray(system.plan.particle_ids)),
            ("protein-site", protein_ids),
            (
                "nucleotide-body",
                np.asarray(nucleotide_model.bodies.particles.particle_ids),
            ),
            ("nucleotide-site", nucleotide_ids),
        )
        for kind, ids in supports:
            for source in sorted(int(value) for value in ids):
                records.append((kind, source, len(records)))
        self.protein_network = protein_network
        self.nucleotide_model = nucleotide_model
        self.cross = cross
        self.protein_reference = protein_reference
        self.protein_site_indices = jnp.asarray(
            [protein_lookup[int(left)] for left, _ in pairs], dtype=jnp.int32
        )
        self.nucleotide_site_indices = jnp.asarray(
            [nucleotide_lookup[int(right)] for _, right in pairs], dtype=jnp.int32
        )
        self.support_map = HybridSupportMap(tuple(records))
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-hybrid-protein-nucleotide-model",
                "protein": protein_network.prepared_id,
                "nucleotide": nucleotide_model.prepared_id,
                "cross": cross.plan_id,
                "reference_source": protein_reference.manifest_id,
                "support": self.support_map.fingerprint(),
            }
        )

    def initialize(
        self,
        protein_positions: ArrayLike,
        protein_momenta: ArrayLike,
        nucleotide: RigidBodyKinematics,
        /,
        *,
        time: float = 0.0,
    ) -> HybridState:
        """Host-admit one state; material and mobility are independent masks."""
        system = self.protein_network.system
        positions = np.asarray(
            protein_positions, dtype=np.dtype(system.plan.coordinate_dtype)
        )
        momenta = np.asarray(protein_momenta, dtype=positions.dtype)
        if positions.shape != (system.capacity, 3) or momenta.shape != positions.shape:
            raise ValueError(
                "Protein position and momentum arrays must match Cartesian capacity."
            )
        active = np.asarray(system.active_mask)
        if (
            not np.all(np.isfinite(positions[active]))
            or not np.all(np.isfinite(momenta[active]))
            or not np.isfinite(time)
        ):
            raise ValueError("Active state and time must be finite.")
        bodies = self.nucleotide_model.bodies
        rigid = bodies.kinematics(
            nucleotide.position,
            nucleotide.velocity,
            nucleotide.orientation,
            nucleotide.angular_velocity,
        )
        body_active = np.asarray(bodies.particles.active_mask)
        for value in (
            nucleotide.position,
            nucleotide.velocity,
            nucleotide.orientation,
            nucleotide.angular_velocity,
        ):
            if not np.all(np.isfinite(np.asarray(value)[body_active])):
                raise ValueError("Active nucleotide state must be finite.")
        if np.any(
            np.sum(np.asarray(nucleotide.orientation)[body_active] ** 2, axis=-1) == 0
        ):
            raise ValueError("Active nucleotide quaternions must be nonzero.")
        protein = AtomisticKinematics(
            jnp.asarray(np.where(active[:, None], positions, 0.0)),
            jnp.where(system.mobile_mask[:, None], jnp.asarray(momenta), 0.0),
            jnp.zeros((system.capacity, 3), dtype=jnp.int64),
        )
        return HybridState(
            protein,
            rigid,
            jnp.asarray(time, dtype=protein.positions.dtype),
            self.prepared_id,
        )

    def energy(
        self, protein_positions: Array, nucleotide: RigidBodyKinematics, /
    ) -> Array:
        sites = self.protein_network.system.coordinate_map.realize(protein_positions)
        rigid_sites = self.nucleotide_model.site_positions(nucleotide)
        displacement = (
            sites.positions[self.protein_site_indices]
            - rigid_sites[self.nucleotide_site_indices]
        )
        protein = self.protein_network.evaluate(protein_positions)
        total = (
            protein.energy
            + self.nucleotide_model.energy(nucleotide)
            + jnp.sum(self.cross.components(displacement))
        )
        return jnp.where(protein.successful & sites.successful, total, jnp.nan)

    def evaluate(self, state: HybridState, /) -> HybridForceEvaluation:
        if state.prepared_id != self.prepared_id:
            raise ValueError("Hybrid state belongs to a different prepared support.")
        system = self.protein_network.system
        protein = self.protein_network.evaluate(state.protein.positions)
        nucleotide = self.nucleotide_model.evaluate(state.nucleotide)
        sites = system.coordinate_map.realize(state.protein.positions)
        rigid_sites = self.nucleotide_model.site_positions(state.nucleotide)
        displacement = (
            sites.positions[self.protein_site_indices]
            - rigid_sites[self.nucleotide_site_indices]
        )

        def pair_energy(delta):
            components = self.cross.components(delta)
            return jnp.sum(components), components

        (_, cross_components), gradient = jax.value_and_grad(pair_energy, has_aux=True)(
            displacement
        )
        protein_site_forces = (
            jnp.zeros_like(sites.positions).at[self.protein_site_indices].add(-gradient)
        )
        rigid_site_forces = (
            jnp.zeros_like(rigid_sites).at[self.nucleotide_site_indices].add(gradient)
        )
        forces = protein.forces + system.coordinate_map.force_pullback(
            state.protein.positions, protein_site_forces
        )
        cross_loads = self.nucleotide_model.marker_map.site_force_load(
            state.nucleotide, rigid_site_forces
        )

        def add_load(left, right):
            return RigidBodyLoad(left.force + right.force, left.torque + right.torque)

        load = add_load(nucleotide.loads.load, cross_loads.load)
        mobile_load = add_load(nucleotide.loads.mobile_load, cross_loads.mobile_load)
        reaction_load = add_load(
            nucleotide.loads.reaction_load, cross_loads.reaction_load
        )
        components = jnp.concatenate(
            (jnp.stack((protein.energy, nucleotide.energy)), cross_components)
        )
        coupled = (
            (self.cross.steric_energy != 0)
            | (self.cross.linker_stiffness != 0)
            | (self.cross.electrostatic_prefactor != 0)
        )
        geometry_valid = jnp.all(~coupled | (jnp.sum(displacement**2, axis=-1) > 0))
        successful = (
            protein.successful
            & nucleotide.successful
            & sites.successful
            & geometry_valid
            & jnp.all(jnp.isfinite(components))
            & jnp.all(jnp.isfinite(forces))
            & jnp.all(jnp.isfinite(load.force))
            & jnp.all(jnp.isfinite(load.torque))
        )
        return HybridForceEvaluation(
            jnp.sum(components),
            components,
            forces,
            jnp.where(system.mobile_mask[:, None], forces, 0.0),
            jnp.where((system.active_mask & ~system.mobile_mask)[:, None], forces, 0.0),
            load,
            mobile_load,
            reaction_load,
            successful,
        )

    def kinetic_energy(self, state: HybridState, /) -> Array:
        system = self.protein_network.system
        momentum = jnp.where(system.mobile_mask[:, None], state.protein.momenta, 0.0)
        protein = (
            0.5
            * system.plan.units.kinetic_to_energy
            * jnp.sum(momentum**2 * system.inverse_masses[:, None])
        )
        return protein + self.nucleotide_model.kinetic_energy(state.nucleotide)

    def step(self, state: HybridState, step_size: ArrayLike, /) -> HybridStepResult:
        """One explicit conservative-force KDK candidate, without a heat bath.

        Both momenta are half-kicked at the old joint configuration, both poses
        drift, then both receive the new joint force. Forces are converted from
        energy/length and torque from energy to native mechanical units once.
        The native rigid kick includes the anisotropic gyroscopic contribution.
        Negative dt supports reversal experiments; zero/nonfinite dt fails.
        """
        dt = jnp.asarray(step_size, dtype=state.time.dtype)
        if dt.shape != ():
            raise ValueError("step_size must be scalar.")
        system = self.protein_network.system
        bodies = self.nucleotide_model.bodies
        scale = system.plan.units.force_to_momentum_rate
        old = self.evaluate(state)
        mobile = system.mobile_mask[:, None]
        half_momentum = jnp.where(
            mobile, state.protein.momenta + 0.5 * dt * scale * old.protein_forces, 0.0
        )
        positions = jnp.where(
            mobile,
            state.protein.positions + dt * system.inverse_masses[:, None] * half_momentum,
            state.protein.positions,
        )
        old_load = RigidBodyLoad(
            scale * old.nucleotide_load.force, scale * old.nucleotide_load.torque
        )
        half_rigid = _rigid_body_half_kick(bodies, state.nucleotide, old_load, dt)
        staged_rigid = rigid_body_drift(bodies, half_rigid, dt)
        staged = HybridState(
            AtomisticKinematics(positions, half_momentum, state.protein.image_counts),
            staged_rigid,
            state.time + dt,
            self.prepared_id,
        )
        new = self.evaluate(staged)
        momentum = jnp.where(
            mobile, half_momentum + 0.5 * dt * scale * new.protein_forces, 0.0
        )
        new_load = RigidBodyLoad(
            scale * new.nucleotide_load.force, scale * new.nucleotide_load.torque
        )
        rigid = _rigid_body_close_kick(bodies, staged_rigid, new_load, dt)
        result = HybridState(
            AtomisticKinematics(positions, momentum, state.protein.image_counts),
            rigid,
            staged.time,
            self.prepared_id,
        )
        total = new.energy + self.kinetic_energy(result)
        successful = (
            old.successful
            & new.successful
            & jnp.isfinite(dt)
            & (dt != 0)
            & jnp.isfinite(total)
        )
        return HybridStepResult(result, new, total, successful)


__all__ = [
    "HybridCrossInteractionPlan",
    "HybridForceEvaluation",
    "HybridState",
    "HybridStepResult",
    "HybridSupportMap",
    "PreparedHybridModel",
]
