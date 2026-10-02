#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic accepted-thickness rupture of multiregion foam sheets."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import parameter_field, ParameterOwner
from ..._validation import positive_integer
from ...geometry.multiregion_surface import (
    apply_surface_burst,
    MultiRegionSurfaceState,
    MultiRegionSurfaceTopology,
    SurfaceBurstResult,
    SurfaceEventPassEvidence,
    SurfaceEventPolicy,
)
from ...interfacial_transport import FilmStepStatus, SurfaceFilmEvidence
from ...typing import checked


class FoamRuptureStatus(IntEnum):
    """Outcome of one deterministic rupture transaction."""

    COMMITTED = 0
    NOT_TRIGGERED = 1
    FILM_STEP_REJECTED = 2
    SUPPORT_INVALID = 3
    LIQUID_FIELD_MISSING = 4
    TRANSACTION_ROLLED_BACK = 5


@final
class BurstProposal(StrictModule):
    """One whole-sheet burst selected from accepted slot-thickness evidence."""

    region_ids: tuple[str, str] = eqx.field(static=True)
    face_ids: tuple[int, ...] = eqx.field(static=True)
    trigger_slots: tuple[tuple[int, int], ...] = eqx.field(static=True)
    minimum_thickness_m: float = eqx.field(static=True)
    priority: float = eqx.field(static=True)
    proposal_id: str = eqx.field(static=True)

    def __init__(
        self,
        region_ids: tuple[str, str],
        face_ids: Sequence[int],
        trigger_slots: Sequence[tuple[int, int]],
        /,
        *,
        minimum_thickness_m: float,
    ) -> None:
        if (
            not isinstance(region_ids, tuple)
            or len(region_ids) != 2
            or not all(isinstance(value, str) and value for value in region_ids)
            or region_ids[0] >= region_ids[1]
        ):
            raise ValueError("region_ids must be a canonical pair of stable IDs.")
        faces = tuple(sorted(int(value) for value in face_ids))
        slots = tuple(sorted((int(vertex), int(slot)) for vertex, slot in trigger_slots))
        minimum = float(minimum_thickness_m)
        if not faces or not slots:
            raise ValueError("A burst proposal needs face and triggering-slot support.")
        if len(set(faces)) != len(faces) or len(set(slots)) != len(slots):
            raise ValueError("Burst face and trigger support must be unique.")
        if any(value < 0 for value in faces) or any(
            vertex < 0 or slot < 0 for vertex, slot in slots
        ):
            raise ValueError("Burst support indices must be nonnegative.")
        if not np.isfinite(minimum) or minimum < 0.0:
            raise ValueError("minimum_thickness_m must be finite and nonnegative.")
        self.region_ids = region_ids
        self.face_ids = faces
        self.trigger_slots = slots
        self.minimum_thickness_m = minimum
        self.priority = minimum
        self.proposal_id = canonical_fingerprint(
            {
                "kind": "foam-burst-proposal",
                "regions": list(region_ids),
                "faces": list(faces),
                "trigger_slots": [list(value) for value in slots],
                "minimum_thickness_m": minimum.hex(),
            }
        )


@final
class FoamRupturePlan(StrictModule, ParameterOwner):
    """Accepted-threshold and named extensive-field policy for foam rupture."""

    rupture_thickness_m: Array = parameter_field()
    minimum_trigger_slots: int = eqx.field(static=True)
    liquid_field_name: str = eqx.field(static=True)
    circulation_field_name: str = eqx.field(static=True)
    gas_amount_field_name: str = eqx.field(static=True)
    gas_energy_field_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rupture_thickness_m: ArrayLike,
        /,
        *,
        minimum_trigger_slots: int = 1,
        liquid_field_name: str = "film_liquid_volume",
        circulation_field_name: str = "circulation",
        gas_amount_field_name: str = "gas_amount_mol",
        gas_energy_field_name: str = "gas_internal_energy_j",
    ) -> None:
        threshold = np.asarray(rupture_thickness_m, dtype=np.float64)
        if threshold.shape != () or not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("rupture_thickness_m must be one finite positive scalar.")
        count = positive_integer(minimum_trigger_slots, "minimum_trigger_slots")
        names = tuple(
            str(value)
            for value in (
                liquid_field_name,
                circulation_field_name,
                gas_amount_field_name,
                gas_energy_field_name,
            )
        )
        if any(not value for value in names):
            raise ValueError("Rupture field names must be nonempty.")
        self.rupture_thickness_m = jnp.asarray(threshold, dtype=jnp.float64)
        self.minimum_trigger_slots = count
        (
            self.liquid_field_name,
            self.circulation_field_name,
            self.gas_amount_field_name,
            self.gas_energy_field_name,
        ) = names
        self.plan_id = canonical_fingerprint(
            {
                "kind": "foam-rupture-plan",
                "rupture_thickness_m": float(threshold).hex(),
                "minimum_trigger_slots": count,
                "liquid_field_name": names[0],
                "circulation_field_name": names[1],
                "gas_amount_field_name": names[2],
                "gas_energy_field_name": names[3],
            }
        )

    def _film_accepted(
        self,
        film_status: ArrayLike,
        evidence: SurfaceFilmEvidence,
        geometry_revision: ArrayLike,
        thickness_m: np.ndarray,
        /,
    ) -> tuple[bool, bool]:
        status = np.asarray(film_status)
        revision = np.asarray(geometry_revision)
        if status.shape != () or revision.shape != ():
            raise ValueError("film_status and geometry_revision must be scalar.")
        geometry_matches = int(revision) == int(np.asarray(evidence.geometry_revision))
        finite = np.all(np.isfinite(thickness_m))
        accepted = bool(
            int(status) == int(FilmStepStatus.ACCEPTED)
            and bool(np.asarray(evidence.converged))
            and bool(np.asarray(evidence.finite))
            and bool(np.asarray(evidence.positivity_guaranteed))
            and bool(np.asarray(evidence.conductance_admissible))
            and geometry_matches
            and finite
        )
        return accepted, geometry_matches

    @checked
    def propose(
        self,
        topology: MultiRegionSurfaceTopology,
        state: MultiRegionSurfaceState,
        thickness_m: ArrayLike,
        film_status: ArrayLike,
        film_evidence: SurfaceFilmEvidence,
        geometry_revision: ArrayLike,
        /,
    ) -> tuple[BurstProposal, ...]:
        """Return canonically ordered proposals only from accepted B evidence."""
        state.require_topology(topology)
        thickness = np.asarray(thickness_m, dtype=np.float64)
        expected = (topology.vertex_capacity, topology.slot_width)
        if thickness.shape != expected:
            raise ValueError(f"thickness_m must have shape {expected}.")
        rupture_mask = np.asarray(film_evidence.rupture_mask, dtype=np.bool_)
        if rupture_mask.shape != expected:
            raise ValueError("film_evidence.rupture_mask must use the sheet-slot layout.")
        accepted, _ = self._film_accepted(
            film_status,
            film_evidence,
            geometry_revision,
            thickness,
        )
        if not accepted:
            return ()
        threshold = float(np.asarray(self.rupture_thickness_m))
        active = np.asarray(topology.slot_active)
        triggered = active & rupture_mask & (thickness <= threshold)
        pair_slots = np.asarray(topology.vertex_pair_slots)
        labels = topology.host_face_labels()
        global_faces = np.asarray(
            topology.face_global_ids[: topology.face_count], dtype=np.int64
        )
        proposals: list[BurstProposal] = []
        for pair_index in range(topology.region_pair_count):
            slots = np.argwhere(triggered & (pair_slots == pair_index))
            if slots.shape[0] < self.minimum_trigger_slots:
                continue
            pair = np.asarray(topology.region_pairs[pair_index], dtype=np.int64)
            pair_rows = np.flatnonzero(
                np.all(np.sort(labels, axis=1) == np.sort(pair)[None, :], axis=1)
            )
            if pair_rows.size == 0:
                continue
            region_ids = (
                min(topology.region_ids[pair[0]], topology.region_ids[pair[1]]),
                max(topology.region_ids[pair[0]], topology.region_ids[pair[1]]),
            )
            minimum = float(np.min(thickness[slots[:, 0], slots[:, 1]]))
            proposals.append(
                BurstProposal(
                    region_ids,
                    tuple(int(value) for value in global_faces[pair_rows]),
                    tuple((int(row[0]), int(row[1])) for row in slots),
                    minimum_thickness_m=minimum,
                )
            )
        return tuple(
            sorted(
                proposals,
                key=lambda value: (
                    value.minimum_thickness_m,
                    value.region_ids,
                    value.face_ids,
                ),
            )
        )


@final
class FoamRuptureEvidence(StrictModule):
    """Trigger acceptance, rim ledger, transfer, and region-lineage evidence."""

    status: Array
    triggered: Array
    film_step_accepted: Array
    geometry_revision_matched: Array
    threshold_m: Array
    minimum_thickness_m: Array
    dropped_sheet_content: Array
    unresolved_rim_before: Array
    unresolved_rim_added: Array
    unresolved_rim_after: Array
    liquid_conservation_residual: Array
    removed_circulation: Array
    gas_amount_residual: Array
    gas_energy_residual: Array
    committed: Array
    event: SurfaceEventPassEvidence | None
    region_pair: tuple[str, str] | None = eqx.field(static=True)
    gas_region_lineage: tuple[tuple[str, tuple[str, ...]], ...] = eqx.field(static=True)
    derivative_available: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


@final
class FoamRuptureResult(StrictModule):
    """Committed burst state and unresolved rim ledger, or unchanged sources."""

    topology: MultiRegionSurfaceTopology
    state: MultiRegionSurfaceState
    unresolved_rim_content: Array
    proposal: BurstProposal | None
    evidence: FoamRuptureEvidence

    @property
    def successful(self) -> bool:
        return int(self.evidence.status) == FoamRuptureStatus.COMMITTED


def _field_residual(
    event: SurfaceEventPassEvidence | None,
    names: tuple[str, ...],
    field_name: str,
    /,
) -> Array:
    if event is None or event.region_transfer is None or field_name not in names:
        return jnp.asarray(0.0, dtype=jnp.float64)
    index = names.index(field_name)
    return event.region_transfer.absolute_defect[index]


def _failed_evidence(
    plan: FoamRupturePlan,
    status: FoamRuptureStatus,
    film_accepted: bool,
    geometry_matches: bool,
    rim: Array,
    /,
    *,
    triggered: bool = False,
    proposal: BurstProposal | None = None,
    event: SurfaceEventPassEvidence | None = None,
) -> FoamRuptureEvidence:
    dtype = rim.dtype
    minimum = np.inf if proposal is None else proposal.minimum_thickness_m
    return FoamRuptureEvidence(
        status=jnp.asarray(int(status), dtype=jnp.int32),
        triggered=jnp.asarray(triggered),
        film_step_accepted=jnp.asarray(film_accepted),
        geometry_revision_matched=jnp.asarray(geometry_matches),
        threshold_m=plan.rupture_thickness_m,
        minimum_thickness_m=jnp.asarray(minimum, dtype=dtype),
        dropped_sheet_content=jnp.zeros((0,), dtype=dtype),
        unresolved_rim_before=rim,
        unresolved_rim_added=jnp.asarray(0.0, dtype=dtype),
        unresolved_rim_after=rim,
        liquid_conservation_residual=jnp.asarray(0.0, dtype=dtype),
        removed_circulation=jnp.asarray(0.0, dtype=dtype),
        gas_amount_residual=jnp.asarray(0.0, dtype=dtype),
        gas_energy_residual=jnp.asarray(0.0, dtype=dtype),
        committed=jnp.asarray(False),
        event=event,
        region_pair=None if proposal is None else proposal.region_ids,
        gas_region_lineage=(),
        derivative_available=False,
        plan_id=plan.plan_id,
    )


def apply_foam_rupture(
    plan: FoamRupturePlan,
    topology: MultiRegionSurfaceTopology,
    state: MultiRegionSurfaceState,
    thickness_m: ArrayLike,
    film_status: ArrayLike,
    film_evidence: SurfaceFilmEvidence,
    geometry_revision: ArrayLike,
    unresolved_rim_content: ArrayLike,
    /,
    *,
    event_policy: SurfaceEventPolicy | None = None,
) -> FoamRuptureResult:
    """Apply the first canonical accepted burst and atomically update its rim ledger."""
    if not isinstance(plan, FoamRupturePlan):
        raise TypeError("plan must be FoamRupturePlan.")
    rim = jnp.asarray(unresolved_rim_content, dtype=state.sheet_fields.dtype)
    if rim.ndim != 0 or not bool(jnp.isfinite(rim)) or bool(rim < 0.0):
        raise ValueError("unresolved_rim_content must be one finite nonnegative scalar.")
    thickness = np.asarray(thickness_m, dtype=np.float64)
    film_accepted, geometry_matches = plan._film_accepted(
        film_status,
        film_evidence,
        geometry_revision,
        thickness,
    )
    if not film_accepted:
        evidence = _failed_evidence(
            plan,
            FoamRuptureStatus.FILM_STEP_REJECTED,
            False,
            geometry_matches,
            rim,
        )
        return FoamRuptureResult(topology, state, rim, None, evidence)
    proposals = plan.propose(
        topology,
        state,
        thickness_m,
        film_status,
        film_evidence,
        geometry_revision,
    )
    if not proposals:
        evidence = _failed_evidence(
            plan,
            FoamRuptureStatus.NOT_TRIGGERED,
            True,
            True,
            rim,
        )
        return FoamRuptureResult(topology, state, rim, None, evidence)
    proposal = proposals[0]
    if plan.liquid_field_name not in state.sheet_field_names:
        evidence = _failed_evidence(
            plan,
            FoamRuptureStatus.LIQUID_FIELD_MISSING,
            True,
            True,
            rim,
            triggered=True,
            proposal=proposal,
        )
        return FoamRuptureResult(topology, state, rim, proposal, evidence)
    burst: SurfaceBurstResult = apply_surface_burst(
        topology,
        state,
        proposal.region_ids,
        proposal.face_ids,
        policy=event_policy,
        priority=proposal.priority,
    )
    if not burst.committed:
        evidence = _failed_evidence(
            plan,
            FoamRuptureStatus.TRANSACTION_ROLLED_BACK,
            True,
            True,
            rim,
            triggered=True,
            proposal=proposal,
            event=burst.evidence,
        )
        return FoamRuptureResult(topology, state, rim, proposal, evidence)
    liquid_index = state.sheet_field_names.index(plan.liquid_field_name)
    liquid = burst.dropped_sheet_content[liquid_index]
    rim_after = rim + liquid
    source_liquid = jnp.sum(
        jnp.where(topology.slot_active, state.sheet_fields[..., liquid_index], 0.0)
    )
    target_liquid = jnp.sum(
        jnp.where(
            burst.topology.slot_active,
            burst.state.sheet_fields[..., liquid_index],
            0.0,
        )
    )
    conservation = target_liquid + rim_after - source_liquid - rim
    circulation = (
        jnp.asarray(0.0, dtype=rim.dtype)
        if plan.circulation_field_name not in state.sheet_field_names
        else burst.dropped_sheet_content[
            state.sheet_field_names.index(plan.circulation_field_name)
        ]
    )
    lineage = ()
    if burst.evidence.lineage is not None:
        lineage = burst.evidence.lineage.region_parents
    evidence = FoamRuptureEvidence(
        status=jnp.asarray(int(FoamRuptureStatus.COMMITTED), dtype=jnp.int32),
        triggered=jnp.asarray(True),
        film_step_accepted=jnp.asarray(True),
        geometry_revision_matched=jnp.asarray(True),
        threshold_m=plan.rupture_thickness_m,
        minimum_thickness_m=jnp.asarray(proposal.minimum_thickness_m, dtype=rim.dtype),
        dropped_sheet_content=burst.dropped_sheet_content,
        unresolved_rim_before=rim,
        unresolved_rim_added=liquid,
        unresolved_rim_after=rim_after,
        liquid_conservation_residual=conservation,
        removed_circulation=circulation,
        gas_amount_residual=_field_residual(
            burst.evidence,
            state.region_field_names,
            plan.gas_amount_field_name,
        ),
        gas_energy_residual=_field_residual(
            burst.evidence,
            state.region_field_names,
            plan.gas_energy_field_name,
        ),
        committed=jnp.asarray(True),
        event=burst.evidence,
        region_pair=proposal.region_ids,
        gas_region_lineage=lineage,
        derivative_available=False,
        plan_id=plan.plan_id,
    )
    return FoamRuptureResult(
        burst.topology,
        burst.state,
        rim_after,
        proposal,
        evidence,
    )


__all__ = [
    "apply_foam_rupture",
    "BurstProposal",
    "FoamRuptureEvidence",
    "FoamRupturePlan",
    "FoamRuptureResult",
    "FoamRuptureStatus",
]
