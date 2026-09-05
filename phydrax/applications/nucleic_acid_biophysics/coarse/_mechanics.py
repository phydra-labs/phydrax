# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Published fixed-topology nucleotide energies through native marker adjoints/KDK."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....atomistic._units import AtomisticUnitSystem
from ....discretization._lagrangian_marker import LagrangianMarkerSetPlan
from ....discretization._periodic_cell import PeriodicCell
from ....discretization.particle._core import ParticleSetPlan
from ....discretization.particle._rigid_body import (
    PreparedRigidBodySet,
    rigid_body_kick_drift_kick,
    rigid_body_world_inertia,
    RigidBodyKinematics,
    RigidBodyLoad,
    RigidBodySetPlan,
    RigidBodyStepResult,
)
from ....discretization.particle._rigid_marker import (
    PreparedRigidMarkerMap,
    RigidMarkerLoadResult,
    RigidMarkerMapPlan,
)
from ....discretization.particle._rigid_thermal import PreparedRigidHeatBath
from ....ein import contract
from ....qualification._reference import ReferenceArtifactManifest
from .._construct import NucleicAcidConstruct
from ._parameters import (
    FAMILY_MODELS,
    nucleotide_reference_sites,
    NucleotideParameterArtifact,
)
from ._published import interaction_energy, radial_support


@dataclass(frozen=True)
class NucleotideModelPlan:
    construct: NucleicAcidConstruct
    body_ids: object
    site_ids: object
    reference_sites: object
    masses: object
    inertia_com: object
    parameters: NucleotideParameterArtifact
    fixed_mask: object = None
    cell: PeriodicCell | None = None

    def prepare(self) -> PreparedNucleotideModel:
        return PreparedNucleotideModel(self)


class NucleotideForceEvaluation(StrictModule):
    energy: Array
    site_forces: Array
    loads: RigidMarkerLoadResult
    successful: Array


class _InteractionGroup(StrictModule):
    pairs: Array
    strengths: Array
    charge_scale: Array
    profile: dict
    bonded: bool = eqx.field(static=True)
    model: str = eqx.field(static=True)


def _profile_cutoff(profile):
    radii = []
    for kind in ("stacking", "hydrogen-bond", "cross-stacking", "coaxial-stacking"):
        if kind in profile:
            radii.append(
                radial_support(
                    profile[kind]["radial"],
                    "morse" if kind in ("stacking", "hydrogen-bond") else "harmonic",
                )[1]
            )
    for epsilon, sigma, join in profile["excluded"].values():
        ratio = sigma / join
        value = 4 * (ratio**12 - ratio**6)
        slope = 24 / join * (ratio**6 - 2 * ratio**12)
        radii.append(join - 2 * value / slope)
    if "screening" in profile:
        radii.append(4.5 * profile["screening"][1])
    return max(radii)


class PreparedNucleotideModel(StrictModule):
    """Full selected published Hamiltonian on all fixed-topology nucleotide pairs.

    DNA1/DNA2 and RNA have distinct geometry, angular products and pair rules;
    hybrid nonbonded profiles are independently supplied DNA-form interactions.
    No mixing rule manufactures hybrid coefficients. All bonded pairs remain
    unwrapped; nonbonded pairs use one COM image for the entire body. Gradients
    hold on a fixed topology/image branch; no Hessian claim at collinear frames.
    """

    bodies: PreparedRigidBodySet
    marker_map: PreparedRigidMarkerMap
    physical_site_mask: Array
    units: AtomisticUnitSystem
    parameter_manifest: ReferenceArtifactManifest
    cell: PeriodicCell | None
    groups: tuple[_InteractionGroup, ...]
    temperature: Array
    prepared_id: str = eqx.field(static=True)
    construct_id: str = eqx.field(static=True)
    parameter_manifest_id: str = eqx.field(static=True)
    family: str = eqx.field(static=True)

    def __init__(self, plan: NucleotideModelPlan, /):
        n = plan.construct.nucleotide_count
        ids, sites, geometry = (
            np.asarray(plan.body_ids),
            np.asarray(plan.site_ids),
            np.asarray(plan.reference_sites, dtype=float),
        )
        if (
            ids.shape != (n,)
            or not np.issubdtype(ids.dtype, np.integer)
            or np.unique(ids).size != n
        ):
            raise ValueError(
                "Unique stable integer body IDs must cover the construct in nucleotide order."
            )
        if (
            sites.shape != (n, 8)
            or not np.issubdtype(sites.dtype, np.integer)
            or np.unique(sites).size != 8 * n
        ):
            raise ValueError(
                "Each nucleotide requires eight unique stable site/frame-marker IDs."
            )
        artifact = plan.parameters
        data = artifact.data()
        models = FAMILY_MODELS[artifact.family]
        chemistry = set(plan.construct.polymer_types)
        if chemistry != set(models) - {"HYBRID"}:
            raise ValueError(
                "Construct DNA/RNA chemistry does not match the parameterized model family."
            )
        expected = nucleotide_reference_sites(plan.construct, artifact)
        if (
            geometry.shape != (n, 8, 3)
            or not np.isfinite(geometry).all()
            or not np.array_equal(geometry, expected)
        ):
            raise ValueError(
                "Reference sites must exactly realize the source parameter geometry and differential frame."
            )
        if plan.cell is not None:
            vectors = np.asarray(plan.cell.vectors)
            if (
                vectors.shape != (3, 3)
                or not np.allclose(vectors, np.diag(np.diag(vectors)))
                or not all(plan.cell.periodic_axes)
            ):
                raise ValueError(
                    "Nucleotide periodic qualification currently requires a full orthorhombic PeriodicCell."
                )
        particles = ParticleSetPlan(ids, plan.masses, ambient_dimension=3).prepare()
        bodies = RigidBodySetPlan(
            np.zeros(n, dtype=np.int32), plan.inertia_com, fixed_mask=plan.fixed_mask
        ).prepare(particles)
        markers = LagrangianMarkerSetPlan(
            sites.reshape(-1), geometry.reshape(-1, 3), np.ones(8 * n)
        ).prepare()
        marker_map = RigidMarkerMapPlan(
            markers, bodies, np.repeat(np.arange(n), 8)
        ).prepare()
        polymers = tuple(
            polymer
            for polymer, sequence in zip(
                plan.construct.polymer_types, plan.construct.sequences, strict=True
            )
            for _ in sequence
        )
        bases = plan.construct.bases
        base_indices = {base: index for index, base in enumerate("AGCT")}
        base_indices["U"] = 3
        rows = {key: index for index, key in enumerate(plan.construct.nucleotide_keys)}
        directed = [(rows[a], rows[b]) for a, b in plan.construct.directed_edges]
        bonded_set = {frozenset(pair) for pair in directed}
        grouped = {}
        for a, b in directed:
            name = polymers[a]
            pair = (b, a) if models[name] != "rna" else (a, b)
            grouped.setdefault((name, True), []).append((pair, (a, b)))
        for a, b in combinations(range(n), 2):
            if frozenset((a, b)) in bonded_set:
                continue
            name = polymers[a] if polymers[a] == polymers[b] else "HYBRID"
            pair = (b, a) if name == "HYBRID" and polymers[a] == "RNA" else (a, b)
            grouped.setdefault((name, False), []).append((pair, pair))
        terminal = {rows[key] for key, _ in plan.construct.termini}
        groups = []
        for (name, bonded), records in grouped.items():
            profile = deepcopy(data["profiles"][name])
            sequence = data["sequence_strengths"][name]
            temp_factor = (
                1.0
                if name == "HYBRID"
                else 1
                + profile.pop("stacking_temperature_coefficient")
                * artifact.units.boltzmann_constant
                * artifact.temperature
            )
            charges = np.ones(len(records))
            if "screening" in profile:
                screen = profile["screening"]
                length = screen["length_per_sqrt_temperature_over_molar"] * np.sqrt(
                    artifact.temperature / artifact.salt_concentration
                )
                charges = np.asarray(
                    [
                        screen["terminal_charge_factor"]
                        ** (int(a in terminal) + int(b in terminal))
                        for (a, b), _ in records
                    ]
                )
                profile["screening"] = [screen["prefactor"], length]
            strengths = []
            pairs = []
            for (a, b), (five, three) in records:
                allowed = (bases[a], bases[b]) in (
                    ("A", "T"),
                    ("T", "A"),
                    ("G", "C"),
                    ("C", "G"),
                    ("A", "U"),
                    ("U", "A"),
                )
                if name == "RNA":
                    allowed |= (bases[a], bases[b]) in (("G", "U"), ("U", "G"))
                stack = (
                    0.0
                    if name == "HYBRID"
                    else sequence["stacking"][base_indices[bases[five]]][
                        base_indices[bases[three]]
                    ]
                    * temp_factor
                )
                hb = (
                    sequence["hydrogen-bond"][base_indices[bases[a]]][
                        base_indices[bases[b]]
                    ]
                    if allowed
                    else 0.0
                )
                strengths.append([stack, hb])
                pairs.append((a, b))
            if plan.cell is not None and not bonded:
                reach = _profile_cutoff(profile) + 2 * np.max(
                    np.sqrt(np.sum(geometry[:, :5] ** 2, -1))
                )
                if reach >= plan.cell.unique_image_radius:
                    raise ValueError(
                        "Interaction support and site offsets exceed the unique periodic COM-image radius."
                    )
            # Every profile scalar/list is numeric; dictionaries retain only fixed
            # equation names. Conversion is host preparation, not runtime lookup.
            numeric = jax.tree.map(
                lambda x: jnp.asarray(x, dtype=geometry.dtype),
                profile,
                is_leaf=lambda x: isinstance(x, list),
            )
            groups.append(
                _InteractionGroup(
                    jnp.asarray(pairs, dtype=jnp.int32),
                    jnp.asarray(strengths),
                    jnp.asarray(charges),
                    numeric,
                    bonded,
                    models[name],
                )
            )
        self.bodies, self.marker_map, self.units, self.cell = (
            bodies,
            marker_map,
            artifact.units,
            plan.cell,
        )
        self.physical_site_mask = jnp.tile(jnp.arange(8) < 5, n)
        self.parameter_manifest = artifact.manifest
        self.groups = tuple(groups)
        self.temperature = jnp.asarray(artifact.temperature)
        self.construct_id, self.parameter_manifest_id, self.family = (
            plan.construct.fingerprint(),
            artifact.manifest.manifest_id,
            artifact.family,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-nucleotide-rigid-model",
                "construct": self.construct_id,
                "parameters": self.parameter_manifest_id,
                "markers": marker_map.prepared_id,
                "units": self.units.unit_system_id,
                "cell": None if plan.cell is None else plan.cell.cell_id,
            }
        )

    def site_positions(self, state: RigidBodyKinematics, /):
        """Physical sites and differential frame markers in stable prepared order."""
        return self.marker_map.evaluate(state).position

    def _energy_sites(self, sites, com, /):
        positions = sites.reshape(self.bodies.capacity, 8, 3)
        energy = jnp.asarray(0.0, dtype=sites.dtype)
        for group in self.groups:
            shift = jnp.zeros((group.pairs.shape[0], 3), dtype=sites.dtype)
            if self.cell is not None and not group.bonded:
                delta = com[group.pairs[:, 1]] - com[group.pairs[:, 0]]
                shift = self.cell.minimum_image(delta) - delta
            values = interaction_energy(
                positions,
                group.pairs,
                group.profile,
                bonded=group.bonded,
                model=group.model,
                strengths=group.strengths,
                charge_scale=group.charge_scale,
                image_shift=shift,
            )
            energy += jnp.sum(values)
        return energy

    def energy(self, state, /):
        return self._energy_sites(self.site_positions(state), state.position)

    def evaluate(self, state, /) -> NucleotideForceEvaluation:
        energy, gradient = jax.value_and_grad(self._energy_sites)(
            self.site_positions(state), state.position
        )
        forces = -gradient
        loads = self.marker_map.site_force_load(state, forces)
        successful = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(forces))
            & jnp.all(jnp.isfinite(loads.load.torque))
        )
        return NucleotideForceEvaluation(energy, forces, loads, successful)

    def mechanical_load(self, state, /) -> RigidBodyLoad:
        load = self.evaluate(state).loads.load
        factor = self.units.force_to_momentum_rate
        return RigidBodyLoad(factor * load.force, factor * load.torque)

    def step(self, state, time, step_size, /) -> RigidBodyStepResult:
        return rigid_body_kick_drift_kick(
            self.bodies,
            state,
            self.mechanical_load(state),
            time,
            step_size,
            lambda t, q, args: self.mechanical_load(q),
        )

    def heat_bath(
        self, translation_friction, rotation_friction, /
    ) -> PreparedRigidHeatBath:
        thermal_energy = (
            float(self.temperature)
            * self.units.boltzmann_constant
            / self.units.kinetic_to_energy
        )
        return PreparedRigidHeatBath(
            self.bodies, thermal_energy, translation_friction, rotation_friction
        )

    def kinetic_energy(self, state, /):
        inertia, _ = rigid_body_world_inertia(self.bodies, state.orientation)
        momentum = contract("...ij,...j->...i", inertia, state.angular_velocity)
        mobile = self.bodies.particles.active_mask & ~self.bodies.fixed_mask
        energy = self.bodies.particles.safe_masses * jnp.sum(
            state.velocity**2, -1
        ) + jnp.sum(state.angular_velocity * momentum, -1)
        return 0.5 * self.units.kinetic_to_energy * jnp.sum(jnp.where(mobile, energy, 0))


__all__ = ["NucleotideModelPlan", "PreparedNucleotideModel", "NucleotideForceEvaluation"]
