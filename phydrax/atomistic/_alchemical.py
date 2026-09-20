#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Controlled Hamiltonians bound to canonical atomistic force-field programs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import ParticleNeighborhoodState
from ._classical import (
    HarmonicAnglePotential,
    HarmonicBondPotential,
    LennardJonesPotential,
    PeriodicTorsionPotential,
)
from ._electrostatics import (
    DirectCoulombPotential,
    EwaldReferencePotential,
    ParticleMeshEwaldPotential,
)
from ._force_field import (
    ForceFieldTermKind,
    GeneralForceFieldTerm,
    PreparedAtomisticForceField,
)
from ._potential import AtomisticPotentialRequirements
from ._potential_program import (
    AbstractPreparedAtomisticEnergyTerm,
    AbstractPreparedAtomisticHamiltonian,
    AtomisticInteractionScaleState,
    AtomisticPotentialContext,
    PreparedAtomisticPotentialProgram,
)
from ._system import PreparedAtomisticSystem
from ._thermodynamic import PreparedThermodynamicStateTable


class AlchemicalControlKind(StrEnum):
    """Canonical interaction family changed by one control coordinate."""

    STERICS = "sterics"
    ELECTROSTATICS = "electrostatics"
    BOND = "bond"
    ANGLE = "angle"
    PROPER_TORSION = "proper-torsion"
    IMPROPER_TORSION = "improper-torsion"


class AlchemicalRegionInteractionMode(StrEnum):
    """How membership in a stable-ID region selects an interaction route."""

    CROSS = "cross"
    INVOLVING = "involving"
    INTERNAL = "internal"


class SoftCorePolicy(StrictModule, NonTrainableState):
    """Endpoint-exact soft-core regularization for controlled pair routes."""

    lennard_jones_alpha: float = eqx.field(static=True)
    electrostatic_alpha: float = eqx.field(static=True)
    coupling_power: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        lennard_jones_alpha: float = 0.5,
        electrostatic_alpha: float = 0.5,
        coupling_power: int = 2,
    ):
        lj = float(lennard_jones_alpha)
        electrostatic = float(electrostatic_alpha)
        if not isinstance(coupling_power, (int, np.integer)) or isinstance(
            coupling_power, bool
        ):
            raise TypeError("coupling_power must be an integer.")
        power = int(coupling_power)
        if (
            not math.isfinite(lj)
            or lj <= 0.0
            or not math.isfinite(electrostatic)
            or electrostatic <= 0.0
            or power < 1
        ):
            raise ValueError("Soft-core alphas and coupling_power must be positive.")
        self.lennard_jones_alpha = lj
        self.electrostatic_alpha = electrostatic
        self.coupling_power = power
        self.policy_id = canonical_fingerprint(
            {
                "kind": "alchemical-soft-core-policy",
                "lennard_jones_alpha": lj.hex(),
                "electrostatic_alpha": electrostatic.hex(),
                "coupling_power": power,
            }
        )


class AlchemicalControlSchedulePlan(StrictModule, NonTrainableState):
    """Ordered thermodynamic states with typed, endpoint-exact controls."""

    state_ids: tuple[str, ...] = eqx.field(static=True)
    control_ids: tuple[str, ...] = eqx.field(static=True)
    control_kinds: tuple[AlchemicalControlKind, ...] = eqx.field(static=True)
    controls: Array
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        state_ids: Sequence[str],
        control_ids: Sequence[str],
        control_kinds: Sequence[AlchemicalControlKind | str],
        controls: ArrayLike,
        /,
    ):
        states = tuple(str(value).strip() for value in state_ids)
        names = tuple(str(value).strip() for value in control_ids)
        kinds = tuple(AlchemicalControlKind(value) for value in control_kinds)
        values = np.asarray(controls)
        if len(states) < 2 or any(not value for value in states):
            raise ValueError("A control schedule requires at least two named states.")
        if len(set(states)) != len(states):
            raise ValueError("Control schedule state IDs must be unique.")
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Control IDs must be non-empty and unique.")
        if len(kinds) != len(names):
            raise ValueError("Every control ID requires one control kind.")
        if values.shape != (len(states), len(names)) or values.dtype.kind != "f":
            raise TypeError("controls must be a floating state-by-control matrix.")
        if np.any(~np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("Control values must be finite and lie in [0, 1].")
        source, destination = values[0], values[-1]
        if (
            np.any((source != 0.0) & (source != 1.0))
            or np.any((destination != 0.0) & (destination != 1.0))
            or np.any(source == destination)
        ):
            raise ValueError(
                "Every control must have distinct exact binary source and destination endpoints."
            )
        difference = np.diff(values, axis=0)
        increasing = destination > source
        if np.any(np.where(increasing[None, :], difference < 0.0, difference > 0.0)):
            raise ValueError("Every control path must be monotone between its endpoints.")
        self.state_ids = states
        self.control_ids = names
        self.control_kinds = kinds
        self.controls = jnp.asarray(values)
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "alchemical-control-schedule",
                "state_ids": list(states),
                "control_ids": list(names),
                "control_kinds": [value.value for value in kinds],
                "controls": array_tree_fingerprint(values),
            }
        )

    @classmethod
    def linear(
        cls,
        control_ids: Sequence[str],
        control_kinds: Sequence[AlchemicalControlKind | str],
        state_count: int,
        /,
        *,
        source_controls: ArrayLike | None = None,
    ) -> "AlchemicalControlSchedulePlan":
        if not isinstance(state_count, (int, np.integer)) or isinstance(
            state_count, bool
        ):
            raise TypeError("state_count must be an integer.")
        count = int(state_count)
        names = tuple(control_ids)
        if count < 2:
            raise ValueError("A linear control schedule requires at least two states.")
        source = (
            np.ones((len(names),), dtype=np.float64)
            if source_controls is None
            else np.asarray(source_controls, dtype=np.float64)
        )
        if source.shape != (len(names),) or np.any((source != 0.0) & (source != 1.0)):
            raise ValueError("source_controls must contain one binary value per control.")
        destination = 1.0 - source
        coordinate = np.linspace(0.0, 1.0, count)[:, None]
        values = source[None, :] + coordinate * (destination - source)[None, :]
        return cls(
            tuple(f"state-{index}" for index in range(count)),
            names,
            control_kinds,
            values,
        )

    @property
    def state_count(self) -> int:
        return len(self.state_ids)

    @property
    def control_count(self) -> int:
        return len(self.control_ids)


class AlchemicalInteractionPartitionPlan(StrictModule, NonTrainableState):
    """Stable-ID regions and route-selection rules for every typed control."""

    control_ids: tuple[str, ...] = eqx.field(static=True)
    region_particle_ids: tuple[Array, ...]
    interaction_modes: tuple[AlchemicalRegionInteractionMode, ...] = eqx.field(
        static=True
    )
    mapped_particle_ids: Array
    changes_masses: bool = eqx.field(static=True)
    changes_constraints: bool = eqx.field(static=True)
    changes_virtual_geometry: bool = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        control_ids: Sequence[str],
        region_particle_ids: Sequence[ArrayLike],
        /,
        *,
        interaction_modes: Sequence[AlchemicalRegionInteractionMode | str] | None = None,
        mapped_particle_ids: ArrayLike | None = None,
        changes_masses: bool = False,
        changes_constraints: bool = False,
        changes_virtual_geometry: bool = False,
    ):
        names = tuple(str(value).strip() for value in control_ids)
        regions = tuple(np.asarray(value) for value in region_particle_ids)
        modes = (
            tuple(AlchemicalRegionInteractionMode.CROSS for _ in names)
            if interaction_modes is None
            else tuple(
                AlchemicalRegionInteractionMode(value) for value in interaction_modes
            )
        )
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Partition control IDs must be non-empty and unique.")
        if len(regions) != len(names) or len(modes) != len(names):
            raise ValueError("Every partition control requires one region and mode.")
        canonical_regions = []
        for region in regions:
            if (
                region.ndim != 1
                or region.size == 0
                or not np.issubdtype(region.dtype, np.integer)
            ):
                raise TypeError(
                    "Each alchemical region must be a non-empty integer vector."
                )
            stable_ids = np.sort(region.astype(np.int64, copy=False))
            if np.unique(stable_ids).size != stable_ids.size:
                raise ValueError("Stable particle IDs within a region must be unique.")
            canonical_regions.append(stable_ids)
        mapping = (
            np.zeros((0, 2), dtype=np.int64)
            if mapped_particle_ids is None
            else np.asarray(mapped_particle_ids)
        )
        if (
            mapping.ndim != 2
            or mapping.shape[1] != 2
            or not np.issubdtype(mapping.dtype, np.integer)
        ):
            raise TypeError("mapped_particle_ids must have integer shape (count, 2).")
        mapping = mapping.astype(np.int64, copy=False)
        if mapping.size and (
            np.unique(mapping[:, 0]).size != mapping.shape[0]
            or np.unique(mapping[:, 1]).size != mapping.shape[0]
        ):
            raise ValueError("Stable-ID mappings must be one-to-one.")
        self.control_ids = names
        self.region_particle_ids = tuple(
            jnp.asarray(value) for value in canonical_regions
        )
        self.interaction_modes = modes
        self.mapped_particle_ids = jnp.asarray(mapping)
        self.changes_masses = bool(changes_masses)
        self.changes_constraints = bool(changes_constraints)
        self.changes_virtual_geometry = bool(changes_virtual_geometry)
        self.partition_id = canonical_fingerprint(
            {
                "kind": "alchemical-interaction-partition",
                "control_ids": list(names),
                "regions": array_tree_fingerprint(tuple(canonical_regions)),
                "interaction_modes": [value.value for value in modes],
                "mapping": array_tree_fingerprint(mapping),
                "changes_masses": bool(changes_masses),
                "changes_constraints": bool(changes_constraints),
                "changes_virtual_geometry": bool(changes_virtual_geometry),
            }
        )


class AlchemicalPreparationEvidence(StrictModule, NonTrainableState):
    """Authenticated host evidence for a complete controlled route partition."""

    particle_count: int = eqx.field(static=True)
    controlled_route_counts: tuple[int, ...] = eqx.field(static=True)
    neutral_electrostatic_regions: tuple[bool, ...] = eqx.field(static=True)
    endpoint_exact: bool = eqx.field(static=True)
    unsupported_routes_absent: bool = eqx.field(static=True)
    successful: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class PreparedAlchemicalInteractionPartition(StrictModule, NonTrainableState):
    """Slot-resolved fixed-shape partition bound to one prepared system."""

    plan: AlchemicalInteractionPartitionPlan
    schedule: AlchemicalControlSchedulePlan
    system: PreparedAtomisticSystem
    region_masks: Array
    bond_control_indices: Array
    angle_control_indices: Array
    torsion_control_indices: Array
    improper_control_indices: Array
    soft_core: SoftCorePolicy
    preparation: AlchemicalPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: AlchemicalInteractionPartitionPlan,
        schedule: AlchemicalControlSchedulePlan,
        force_field: PreparedAtomisticForceField,
        soft_core: SoftCorePolicy,
        /,
    ):
        if plan.control_ids != schedule.control_ids:
            raise ValueError(
                "Schedule and partition control identities must match exactly."
            )
        if plan.changes_masses:
            raise ValueError(
                "State-dependent masses are not supported by controlled Hamiltonians."
            )
        if plan.changes_constraints:
            raise ValueError(
                "State-dependent constraints are not supported by controlled Hamiltonians."
            )
        if plan.changes_virtual_geometry:
            raise ValueError(
                "State-dependent virtual-site geometry is not supported by controlled Hamiltonians."
            )
        system = force_field.system
        particle_ids = np.asarray(system.plan.particle_ids, dtype=np.int64)
        active = np.asarray(system.active_mask, dtype=np.bool_)
        slot_by_id = {int(value): index for index, value in enumerate(particle_ids)}
        masks = np.zeros((schedule.control_count, system.capacity), dtype=np.bool_)
        for control_index, region in enumerate(plan.region_particle_ids):
            for identifier in np.asarray(region, dtype=np.int64):
                stable_id = int(identifier)
                if stable_id not in slot_by_id or not active[slot_by_id[stable_id]]:
                    raise ValueError(
                        f"Alchemical region references unknown or inactive particle ID {stable_id}."
                    )
                masks[control_index, slot_by_id[stable_id]] = True
        mapping = np.asarray(plan.mapped_particle_ids, dtype=np.int64)
        if mapping.size and not np.all(np.isin(mapping, particle_ids[active])):
            raise ValueError(
                "Stable-ID mapping references particles outside the prepared system."
            )

        changed_terms = tuple(force_field.potential.plan.terms)
        controlled_kinds = set(schedule.control_kinds)
        reciprocal_electrostatics = any(
            isinstance(term, (EwaldReferencePotential, ParticleMeshEwaldPotential))
            or (
                isinstance(term, GeneralForceFieldTerm)
                and term.kind is ForceFieldTermKind.REACTION_FIELD
            )
            for term in changed_terms
        )
        reciprocal_dispersion = any(
            isinstance(term, GeneralForceFieldTerm)
            and term.kind
            in {
                ForceFieldTermKind.LENNARD_JONES_PME,
                ForceFieldTermKind.DISPERSION_CORRECTION,
            }
            for term in changed_terms
        )
        if (
            AlchemicalControlKind.ELECTROSTATICS in controlled_kinds
            and reciprocal_electrostatics
        ):
            raise ValueError(
                "Reciprocal, self, background, exception, and real-space electrostatic "
                "controls cannot diverge; this changed route is unsupported."
            )
        if AlchemicalControlKind.STERICS in controlled_kinds and reciprocal_dispersion:
            raise ValueError(
                "Real-space and reciprocal or tail dispersion controls cannot diverge; "
                "this changed route is unsupported."
            )
        general_kind = {
            ForceFieldTermKind.HARMONIC_IMPROPER: AlchemicalControlKind.IMPROPER_TORSION,
            ForceFieldTermKind.UREY_BRADLEY: AlchemicalControlKind.ANGLE,
            ForceFieldTermKind.TORSION_SERIES: AlchemicalControlKind.PROPER_TORSION,
            ForceFieldTermKind.RYCKAERT_BELLEMANS: AlchemicalControlKind.PROPER_TORSION,
            ForceFieldTermKind.CMAP: AlchemicalControlKind.PROPER_TORSION,
            ForceFieldTermKind.PAIR_OVERRIDE: AlchemicalControlKind.STERICS,
            ForceFieldTermKind.MORSE: AlchemicalControlKind.STERICS,
            ForceFieldTermKind.BUCKINGHAM: AlchemicalControlKind.STERICS,
            ForceFieldTermKind.TABULATED_PAIR: AlchemicalControlKind.STERICS,
            ForceFieldTermKind.REACTION_FIELD: AlchemicalControlKind.ELECTROSTATICS,
            ForceFieldTermKind.DISPERSION_CORRECTION: AlchemicalControlKind.STERICS,
            ForceFieldTermKind.LENNARD_JONES_PME: AlchemicalControlKind.STERICS,
        }
        supported_types = (
            HarmonicBondPotential,
            HarmonicAnglePotential,
            PeriodicTorsionPotential,
            LennardJonesPotential,
            DirectCoulombPotential,
        )
        unsupported = []
        for term in changed_terms:
            if isinstance(term, supported_types):
                continue
            if isinstance(term, (EwaldReferencePotential, ParticleMeshEwaldPotential)):
                if AlchemicalControlKind.ELECTROSTATICS not in controlled_kinds:
                    continue
            elif isinstance(term, GeneralForceFieldTerm):
                if general_kind[term.kind] not in controlled_kinds:
                    continue
            unsupported.append(term.name)
        if unsupported:
            raise ValueError(
                "Controlled Hamiltonians refuse unsupported changed force-field routes: "
                + ", ".join(unsupported)
            )
        kinds_present = set()
        for term in changed_terms:
            if isinstance(term, HarmonicBondPotential):
                kinds_present.add(AlchemicalControlKind.BOND)
            elif isinstance(term, HarmonicAnglePotential):
                kinds_present.add(AlchemicalControlKind.ANGLE)
            elif isinstance(term, PeriodicTorsionPotential):
                kinds_present.add(
                    AlchemicalControlKind.IMPROPER_TORSION
                    if term.improper
                    else AlchemicalControlKind.PROPER_TORSION
                )
            elif isinstance(term, LennardJonesPotential):
                kinds_present.add(AlchemicalControlKind.STERICS)
            elif isinstance(term, DirectCoulombPotential):
                kinds_present.add(AlchemicalControlKind.ELECTROSTATICS)
        for kind in schedule.control_kinds:
            if kind not in kinds_present:
                raise ValueError(
                    f"No canonical {kind.value} term is available for its control."
                )

        def selected(membership: np.ndarray, mode: AlchemicalRegionInteractionMode):
            count = np.count_nonzero(membership, axis=1)
            if mode is AlchemicalRegionInteractionMode.CROSS:
                return (count > 0) & (count < membership.shape[1])
            if mode is AlchemicalRegionInteractionMode.INVOLVING:
                return count > 0
            return count == membership.shape[1]

        def route_controls(routes: Array, kind: AlchemicalControlKind) -> np.ndarray:
            host_routes = np.asarray(routes, dtype=np.int32)
            result = np.full((host_routes.shape[0],), -1, dtype=np.int32)
            for control_index, control_kind in enumerate(schedule.control_kinds):
                if control_kind is not kind:
                    continue
                chosen = selected(
                    masks[control_index, host_routes],
                    plan.interaction_modes[control_index],
                )
                if np.any(chosen & (result >= 0)):
                    raise ValueError(
                        f"{kind.value} routes overlap between alchemical controls."
                    )
                result[chosen] = control_index
            return result

        topology = system.topology
        bond = route_controls(topology.bond_indices, AlchemicalControlKind.BOND)
        angle = route_controls(topology.angle_indices, AlchemicalControlKind.ANGLE)
        torsion = route_controls(
            topology.torsion_indices, AlchemicalControlKind.PROPER_TORSION
        )
        improper = route_controls(
            topology.improper_indices, AlchemicalControlKind.IMPROPER_TORSION
        )
        constraint_pairs = {
            tuple(sorted(row))
            for row in np.asarray(topology.constraint_indices, dtype=np.int32).tolist()
        }
        controlled_bonds = {
            tuple(sorted(row))
            for row, index in zip(
                np.asarray(topology.bond_indices, dtype=np.int32).tolist(),
                bond,
                strict=True,
            )
            if index >= 0
        }
        if constraint_pairs & controlled_bonds:
            raise ValueError(
                "A controlled bond route cannot retain a state-independent distance constraint."
            )

        pair_counts = np.zeros((schedule.control_count,), dtype=np.int64)
        slots = np.flatnonzero(active)
        for left_offset, left in enumerate(slots):
            for right in slots[left_offset + 1 :]:
                owner_by_kind: dict[AlchemicalControlKind, int] = {}
                for control_index, kind in enumerate(schedule.control_kinds):
                    if kind not in (
                        AlchemicalControlKind.STERICS,
                        AlchemicalControlKind.ELECTROSTATICS,
                    ):
                        continue
                    membership = masks[control_index, [left, right]][None, :]
                    if bool(
                        selected(membership, plan.interaction_modes[control_index])[0]
                    ):
                        if kind in owner_by_kind:
                            raise ValueError(
                                f"{kind.value} pair routes overlap between alchemical controls."
                            )
                        owner_by_kind[kind] = control_index
                        pair_counts[control_index] += 1
        route_counts = []
        neutral = []
        for control_index, kind in enumerate(schedule.control_kinds):
            count = {
                AlchemicalControlKind.BOND: int(np.count_nonzero(bond == control_index)),
                AlchemicalControlKind.ANGLE: int(
                    np.count_nonzero(angle == control_index)
                ),
                AlchemicalControlKind.PROPER_TORSION: int(
                    np.count_nonzero(torsion == control_index)
                ),
                AlchemicalControlKind.IMPROPER_TORSION: int(
                    np.count_nonzero(improper == control_index)
                ),
            }.get(kind, int(pair_counts[control_index]))
            if count == 0:
                raise ValueError(
                    f"Alchemical control {schedule.control_ids[control_index]!r} selects no routes."
                )
            route_counts.append(count)
            region_charge = float(
                np.sum(
                    np.asarray(system.plan.charges)[
                        masks[control_index] & np.asarray(system.active_mask)
                    ]
                )
            )
            has_charge_change = not math.isclose(region_charge, 0.0, abs_tol=1.0e-10)
            if kind is AlchemicalControlKind.ELECTROSTATICS and has_charge_change:
                raise ValueError(
                    "Charge-changing alchemical regions are unsupported; electrostatic regions "
                    "must have zero net charge."
                )
            neutral.append(not has_charge_change)
        evidence_payload = {
            "kind": "controlled-hamiltonian-preparation-evidence",
            "system": system.prepared_id,
            "partition": plan.partition_id,
            "schedule": schedule.schedule_id,
            "controlled_route_counts": route_counts,
            "neutral_electrostatic_regions": neutral,
            "endpoint_exact": True,
            "unsupported_routes_absent": True,
        }
        evidence_id = canonical_fingerprint(evidence_payload)
        self.plan = plan
        self.schedule = schedule
        self.system = system
        self.region_masks = jnp.asarray(masks)
        self.bond_control_indices = jnp.asarray(bond)
        self.angle_control_indices = jnp.asarray(angle)
        self.torsion_control_indices = jnp.asarray(torsion)
        self.improper_control_indices = jnp.asarray(improper)
        self.soft_core = soft_core
        self.preparation = AlchemicalPreparationEvidence(
            particle_count=int(np.count_nonzero(active)),
            controlled_route_counts=tuple(route_counts),
            neutral_electrostatic_regions=tuple(neutral),
            endpoint_exact=True,
            unsupported_routes_absent=True,
            successful=True,
            evidence_id=evidence_id,
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-alchemical-interaction-partition",
                "evidence": evidence_id,
                "region_masks": array_tree_fingerprint(masks),
                "route_controls": array_tree_fingerprint(
                    {
                        "bond": bond,
                        "angle": angle,
                        "torsion": torsion,
                        "improper": improper,
                    }
                ),
                "soft_core": soft_core.policy_id,
            }
        )

    @staticmethod
    def _route_scale(indices: Array, controls: Array, /) -> Array:
        if indices.shape[0] == 0:
            return jnp.ones(indices.shape, dtype=controls.dtype)
        safe = jnp.clip(indices, 0, controls.shape[0] - 1)
        return jnp.where(indices >= 0, controls[safe], 1.0)

    def _pair_scale(
        self,
        left: Array,
        right: Array,
        controls: Array,
        kind: AlchemicalControlKind,
        /,
    ) -> Array:
        scale = jnp.ones(left.shape, dtype=controls.dtype)
        for control_index, control_kind in enumerate(self.schedule.control_kinds):
            if control_kind is not kind:
                continue
            left_member = self.region_masks[control_index, left]
            right_member = self.region_masks[control_index, right]
            mode = self.plan.interaction_modes[control_index]
            if mode is AlchemicalRegionInteractionMode.CROSS:
                selected = left_member != right_member
            elif mode is AlchemicalRegionInteractionMode.INVOLVING:
                selected = left_member | right_member
            else:
                selected = left_member & right_member
            scale = jnp.where(selected, controls[control_index], scale)
        return scale

    def interaction_scales(
        self,
        left: Array,
        right: Array,
        controls: Array,
        dtype,
        /,
    ) -> AtomisticInteractionScaleState:
        value = jnp.asarray(controls, dtype=dtype)
        return AtomisticInteractionScaleState(
            bond=self._route_scale(self.bond_control_indices, value),
            angle=self._route_scale(self.angle_control_indices, value),
            torsion=self._route_scale(self.torsion_control_indices, value),
            improper=self._route_scale(self.improper_control_indices, value),
            lennard_jones=self._pair_scale(
                left, right, value, AlchemicalControlKind.STERICS
            ),
            electrostatic=self._pair_scale(
                left, right, value, AlchemicalControlKind.ELECTROSTATICS
            ),
            lennard_jones_softcore_alpha=jnp.asarray(
                self.soft_core.lennard_jones_alpha, dtype=dtype
            ),
            electrostatic_softcore_alpha=jnp.asarray(
                self.soft_core.electrostatic_alpha, dtype=dtype
            ),
            softcore_power=self.soft_core.coupling_power,
        )


class ControlledHamiltonianStatus(StrictModule):
    finite: Array
    controls_in_range: Array
    state_index_valid: Array
    neighborhood_successful: Array
    successful: Array


class ControlledHamiltonianEvaluation(StrictModule):
    energy: Array
    term_energies: Array
    atom_energy: Array
    forces: Array
    virial: Array
    successful: Array
    neighborhood_successful: Array
    graph_overflow: Array
    program_id: str = eqx.field(static=True)
    dU_dcontrols: Array
    control_values: Array
    state_index: Array
    status: ControlledHamiltonianStatus
    schedule_id: str = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)


class AlchemicalReducedPotentialEvaluation(StrictModule):
    """Dimensionless state-by-replica reduced potentials without a UQ dependency."""

    values: Array
    energies: Array
    coverage: Array
    successful: Array
    state_indices: Array
    inverse_temperatures: Array
    controls: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    potential_ids: tuple[str, ...] = eqx.field(static=True)
    bias_ids: tuple[str | None, ...] = eqx.field(static=True)
    control_ids: tuple[str, ...] = eqx.field(static=True)
    measure_id: str = eqx.field(static=True)
    unit_system_id: str = eqx.field(static=True)
    thermodynamic_table_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)


class ControlledHamiltonianPlan(StrictModule, NonTrainableState):
    """Bind typed controls to one prepared canonical force-field Hamiltonian."""

    force_field: PreparedAtomisticForceField
    schedule: AlchemicalControlSchedulePlan
    partition: AlchemicalInteractionPartitionPlan
    soft_core: SoftCorePolicy
    requirements: AtomisticPotentialRequirements
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        force_field: PreparedAtomisticForceField,
        schedule: AlchemicalControlSchedulePlan,
        partition: AlchemicalInteractionPartitionPlan,
        /,
        *,
        soft_core: SoftCorePolicy | None = None,
    ):
        if not isinstance(force_field, PreparedAtomisticForceField):
            raise TypeError("force_field must be a PreparedAtomisticForceField.")
        if not isinstance(schedule, AlchemicalControlSchedulePlan):
            raise TypeError("schedule must be an AlchemicalControlSchedulePlan.")
        if not isinstance(partition, AlchemicalInteractionPartitionPlan):
            raise TypeError("partition must be an AlchemicalInteractionPartitionPlan.")
        policy = SoftCorePolicy() if soft_core is None else soft_core
        if not isinstance(policy, SoftCorePolicy):
            raise TypeError("soft_core must be SoftCorePolicy or None.")
        self.force_field = force_field
        self.schedule = schedule
        self.partition = partition
        self.soft_core = policy
        self.requirements = force_field.potential.plan.requirements
        self.plan_id = canonical_fingerprint(
            {
                "kind": "controlled-hamiltonian-plan",
                "force_field": force_field.prepared_id,
                "schedule": schedule.schedule_id,
                "partition": partition.partition_id,
                "soft_core": policy.policy_id,
            }
        )

    def prepare(self, /) -> "PreparedControlledHamiltonian":
        prepared_partition = PreparedAlchemicalInteractionPartition(
            self.partition, self.schedule, self.force_field, self.soft_core
        )
        return PreparedControlledHamiltonian(self, prepared_partition)


class PreparedControlledHamiltonian(AbstractPreparedAtomisticHamiltonian):
    """One cell-aware scalar differentiated for forces and control derivatives."""

    plan: ControlledHamiltonianPlan
    force_field: PreparedAtomisticForceField
    system: PreparedAtomisticSystem
    potential: PreparedAtomisticPotentialProgram
    terms: tuple[AbstractPreparedAtomisticEnergyTerm, ...]
    coefficients: Array
    partition: PreparedAlchemicalInteractionPartition
    prepared_id: str = eqx.field(static=True)
    control_ids: tuple[str, ...] = eqx.field(static=True)
    control_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ControlledHamiltonianPlan,
        partition: PreparedAlchemicalInteractionPartition,
        /,
    ):
        self.plan = plan
        self.force_field = plan.force_field
        self.system = plan.force_field.system
        self.potential = plan.force_field.potential
        self.terms = plan.force_field.potential.terms
        self.coefficients = self.potential.plan.coefficients
        self.partition = partition
        self.control_ids = plan.schedule.control_ids
        self.control_layout_id = canonical_fingerprint(
            {
                "kind": "atomistic-control-layout",
                "control_ids": list(self.control_ids),
            }
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-controlled-hamiltonian",
                "plan": plan.plan_id,
                "partition": partition.prepared_id,
                "potential": self.potential.prepared_id,
            }
        )

    def _resolve_controls(
        self,
        state_index: ArrayLike | None,
        control_values: ArrayLike | None,
        dtype,
        /,
    ) -> tuple[Array, Array, Array]:
        if state_index is not None and control_values is not None:
            raise ValueError("Supply state_index or control_values, not both.")
        if control_values is not None:
            controls = jnp.asarray(control_values, dtype=dtype)
            if controls.shape != (self.plan.schedule.control_count,):
                raise ValueError(
                    "control_values must contain one scalar per scheduled control."
                )
            return controls, jnp.asarray(-1, dtype=jnp.int32), jnp.asarray(True)
        index = jnp.asarray(0 if state_index is None else state_index, dtype=jnp.int32)
        if index.shape != ():
            raise ValueError("state_index must be scalar.")
        valid = (index >= 0) & (index < self.plan.schedule.state_count)
        safe = jnp.clip(index, 0, self.plan.schedule.state_count - 1)
        return self.plan.schedule.controls[safe].astype(dtype), index, valid

    def _interaction_scales(
        self,
        neighborhood: ParticleNeighborhoodState,
        controls: Array,
        dtype,
        /,
    ) -> AtomisticInteractionScaleState:
        pairs = neighborhood.pair_relation
        return self.partition.interaction_scales(
            pairs.left_indices, pairs.right_indices, controls, dtype
        )

    def context(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        *,
        state_index: ArrayLike | None = None,
        control_values: ArrayLike | None = None,
        **context_kwargs: Any,
    ) -> AtomisticPotentialContext:
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        controls, _, _ = self._resolve_controls(
            state_index, control_values, position.dtype
        )
        scales = self._interaction_scales(neighborhood, controls, position.dtype)
        return self.potential.context(
            position, neighborhood, interaction_scales=scales, **context_kwargs
        )

    def energy(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        *,
        state_index: ArrayLike | None = None,
        control_values: ArrayLike | None = None,
        **context_kwargs: Any,
    ) -> tuple[Array, tuple[Array, Array, Array, Array]]:
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        controls, _, state_valid = self._resolve_controls(
            state_index, control_values, position.dtype
        )
        scales = self._interaction_scales(neighborhood, controls, position.dtype)
        energy, auxiliary = self.potential.energy(
            position, neighborhood, interaction_scales=scales, **context_kwargs
        )
        term_energies, atom_energy, potential_successful, graph_overflow = auxiliary
        controls_valid = jnp.all(
            jnp.isfinite(controls) & (controls >= 0.0) & (controls <= 1.0)
        )
        successful = potential_successful & state_valid & controls_valid
        return jnp.where(successful, energy, jnp.nan), (
            term_energies,
            atom_energy,
            successful,
            graph_overflow,
        )

    def evaluate(
        self,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        *,
        state_index: ArrayLike | None = None,
        control_values: ArrayLike | None = None,
        **context_kwargs: Any,
    ) -> ControlledHamiltonianEvaluation:
        position = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        expected = (self.system.capacity, 3)
        if position.shape != expected:
            raise ValueError(f"positions must have shape {expected}.")
        controls, resolved_index, state_valid = self._resolve_controls(
            state_index, control_values, position.dtype
        )
        selected_cell = context_kwargs.get("cell")
        if selected_cell is None:
            selected_cell = self.system.cell
        vectors = context_kwargs.get("cell_vectors")
        unwrapped = context_kwargs.get("unwrapped_positions")
        fractional = context_kwargs.get("fractional_positions")
        unwrapped_offset = None
        if unwrapped is not None:
            offset = jnp.asarray(unwrapped, dtype=position.dtype) - position
            if selected_cell is None:
                unwrapped_offset = jax.lax.stop_gradient(offset)
            else:
                image_vectors = (
                    selected_cell.vectors if vectors is None else jnp.asarray(vectors)
                ).astype(position.dtype)
                images = jax.lax.stop_gradient(
                    jnp.where(
                        selected_cell.periodic_mask,
                        jnp.round(
                            contract(
                                "ni,ij->nj",
                                offset,
                                selected_cell.inverse_for_vectors(image_vectors),
                            )
                        ),
                        0.0,
                    )
                )
                translation = contract("ni,ij->nj", images, image_vectors)
                unwrapped_offset = translation + jax.lax.stop_gradient(
                    offset - translation
                )
        fractional_offset = None
        if fractional is not None and vectors is not None and selected_cell is not None:
            fractional_offset = jax.lax.stop_gradient(
                jnp.asarray(fractional, dtype=position.dtype)
                - selected_cell.fractional_with_vectors(position, vectors)
            )

        def closure(value: Array, control: Array):
            kwargs = dict(context_kwargs)
            if unwrapped_offset is not None:
                kwargs["unwrapped_positions"] = value + unwrapped_offset
            if fractional_offset is not None:
                kwargs["fractional_positions"] = (
                    selected_cell.fractional_with_vectors(value, vectors)
                    + fractional_offset
                )
            return self.energy(value, neighborhood, control_values=control, **kwargs)

        (energy, auxiliary), (position_gradient, control_gradient) = jax.value_and_grad(
            closure, argnums=(0, 1), has_aux=True
        )(position, controls)
        term_energies, atom_energy, program_successful, graph_overflow = auxiliary
        forces = jnp.where(self.system.active_mask[:, None], -position_gradient, 0.0)
        center = jnp.sum(
            jnp.where(
                self.system.active_mask[:, None],
                self.system.plan.masses[:, None] * position,
                0.0,
            ),
            axis=0,
        ) / jnp.sum(jnp.where(self.system.active_mask, self.system.plan.masses, 0.0))
        virial = -contract("ni,nj->ij", position - center, forces)
        controls_in_range = jnp.all(
            jnp.isfinite(controls) & (controls >= 0.0) & (controls <= 1.0)
        )
        finite = (
            jnp.isfinite(energy)
            & jnp.all(jnp.isfinite(term_energies))
            & jnp.all(jnp.isfinite(atom_energy))
            & jnp.all(jnp.isfinite(forces))
            & jnp.all(jnp.isfinite(virial))
            & jnp.all(jnp.isfinite(control_gradient))
        )
        successful = program_successful & finite & controls_in_range & state_valid
        status = ControlledHamiltonianStatus(
            finite=finite,
            controls_in_range=controls_in_range,
            state_index_valid=state_valid,
            neighborhood_successful=program_successful,
            successful=successful,
        )
        nan = jnp.asarray(jnp.nan, dtype=energy.dtype)
        return ControlledHamiltonianEvaluation(
            energy=jnp.where(successful, energy, nan),
            term_energies=jnp.where(successful, term_energies, nan),
            atom_energy=jnp.where(successful, atom_energy, nan),
            forces=jnp.where(successful, forces, nan),
            virial=jnp.where(successful, virial, nan),
            successful=successful,
            neighborhood_successful=program_successful,
            graph_overflow=graph_overflow,
            program_id=self.prepared_id,
            dU_dcontrols=jnp.where(successful, control_gradient, nan),
            control_values=controls,
            state_index=resolved_index,
            status=status,
            schedule_id=self.plan.schedule.schedule_id,
            partition_id=self.plan.partition.partition_id,
        )

    def force_group_energy(
        self,
        group: int,
        positions: ArrayLike,
        neighborhood: ParticleNeighborhoodState,
        /,
        *,
        state_index: ArrayLike | None = None,
        control_values: ArrayLike | None = None,
        **context_kwargs: Any,
    ) -> tuple[Array, Array]:
        value = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        controls, _, state_valid = self._resolve_controls(
            state_index, control_values, value.dtype
        )
        selected = tuple(
            index
            for index, term in enumerate(self.terms)
            if term.force_group == int(group)
        )
        if not selected:
            raise ValueError(f"Force group {int(group)} is absent from the Hamiltonian.")
        context = self.context(
            value, neighborhood, control_values=controls, **context_kwargs
        )
        evaluations = tuple(self.terms[index].energy(context) for index in selected)
        energies = jnp.stack(tuple(item.energy for item in evaluations))
        coefficients = self.potential.plan.coefficients[jnp.asarray(selected)]
        energy = jnp.sum(coefficients.astype(energies.dtype) * energies)
        successful = (
            state_valid
            & context.neighborhood_successful
            & jnp.all(jnp.stack(tuple(item.successful for item in evaluations)))
        )
        successful = successful & jnp.isfinite(energy)
        return jnp.where(successful, energy, jnp.nan), successful

    def reduced_potentials(
        self,
        positions: ArrayLike,
        neighborhoods: Sequence[ParticleNeighborhoodState],
        thermodynamic: PreparedThermodynamicStateTable,
        /,
        *,
        state_indices: ArrayLike | None = None,
        context_kwargs: Sequence[dict[str, Any]] | None = None,
    ) -> AlchemicalReducedPotentialEvaluation:
        if not isinstance(thermodynamic, PreparedThermodynamicStateTable):
            raise TypeError("thermodynamic must be a PreparedThermodynamicStateTable.")
        if (
            thermodynamic.program_id != self.prepared_id
            or thermodynamic.system_id != self.system.prepared_id
            or thermodynamic.control_ids != self.control_ids
            or thermodynamic.control_layout_id != self.control_layout_id
            or thermodynamic.unit_system_id != self.system.plan.units.unit_system_id
        ):
            raise ValueError(
                "Thermodynamic table and controlled Hamiltonian identities do not match."
            )
        coordinates = jnp.asarray(positions, dtype=self.system.plan.coordinate_dtype)
        if coordinates.ndim != 3 or coordinates.shape[1:] != (
            self.system.capacity,
            3,
        ):
            raise ValueError("positions must have shape (replicas, atom_capacity, 3).")
        neighborhoods_ = tuple(neighborhoods)
        if len(neighborhoods_) != coordinates.shape[0] or any(
            not isinstance(value, ParticleNeighborhoodState) for value in neighborhoods_
        ):
            raise ValueError("One ParticleNeighborhoodState is required per replica.")
        host_indices = (
            np.arange(thermodynamic.state_count, dtype=np.int32)
            if state_indices is None
            else np.asarray(state_indices)
        )
        if host_indices.ndim != 1 or not np.issubdtype(host_indices.dtype, np.integer):
            raise TypeError("state_indices must be an integer vector.")
        if host_indices.size == 0 or np.any(
            (host_indices < 0) | (host_indices >= thermodynamic.state_count)
        ):
            raise ValueError("state_indices must identify prepared thermodynamic states.")
        beta_host = np.asarray(thermodynamic.beta)[host_indices]
        if np.any(~np.isfinite(beta_host)) or np.any(beta_host < 0.0):
            raise ValueError(
                "Thermodynamic inverse temperatures must be finite and non-negative."
            )
        indices = jnp.asarray(host_indices, dtype=jnp.int32)
        beta = jnp.asarray(beta_host, dtype=coordinates.dtype)
        controls = thermodynamic.controls[indices].astype(coordinates.dtype)
        kwargs = (
            tuple({} for _ in neighborhoods_)
            if context_kwargs is None
            else tuple(context_kwargs)
        )
        if len(kwargs) != len(neighborhoods_):
            raise ValueError("context_kwargs must provide one mapping per replica.")
        energy_rows = []
        successful_rows = []
        for control in controls:
            energies = []
            accepted = []
            for replica, neighborhood in enumerate(neighborhoods_):
                energy, auxiliary = self.energy(
                    coordinates[replica],
                    neighborhood,
                    control_values=control,
                    **kwargs[replica],
                )
                finite = jnp.isfinite(energy)
                energies.append(energy)
                accepted.append(auxiliary[2] & finite)
            energy_rows.append(jnp.stack(tuple(energies)))
            successful_rows.append(jnp.stack(tuple(accepted)))
        energies = jnp.stack(tuple(energy_rows))
        successful = jnp.stack(tuple(successful_rows))
        values = beta[:, None] * energies
        successful = successful & jnp.isfinite(values)
        ordered_state_ids = tuple(
            thermodynamic.state_ids[int(value)] for value in host_indices
        )
        potential_ids = tuple(
            thermodynamic.potential_ids[int(value)] for value in host_indices
        )
        bias_ids = tuple(thermodynamic.bias_ids[int(value)] for value in host_indices)
        return AlchemicalReducedPotentialEvaluation(
            values=jnp.where(successful, values, jnp.nan),
            energies=jnp.where(successful, energies, jnp.nan),
            coverage=successful,
            successful=successful,
            state_indices=indices,
            inverse_temperatures=beta,
            controls=controls,
            state_ids=ordered_state_ids,
            potential_ids=potential_ids,
            bias_ids=bias_ids,
            control_ids=thermodynamic.control_ids,
            measure_id=thermodynamic.phase_space_measure_id,
            unit_system_id=thermodynamic.unit_system_id,
            thermodynamic_table_id=thermodynamic.table_id,
            prepared_id=self.prepared_id,
            schedule_id=self.plan.schedule.schedule_id,
        )


__all__ = [
    "AlchemicalControlKind",
    "AlchemicalControlSchedulePlan",
    "AlchemicalInteractionPartitionPlan",
    "AlchemicalPreparationEvidence",
    "AlchemicalReducedPotentialEvaluation",
    "AlchemicalRegionInteractionMode",
    "ControlledHamiltonianEvaluation",
    "ControlledHamiltonianPlan",
    "ControlledHamiltonianStatus",
    "PreparedAlchemicalInteractionPartition",
    "PreparedControlledHamiltonian",
    "SoftCorePolicy",
]
