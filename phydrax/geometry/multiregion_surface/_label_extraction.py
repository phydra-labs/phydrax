#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic hard-label extraction into explicit multiregion surfaces.

The host route interprets a three-dimensional uniform label field as material
samples on a Cartesian lattice, adds an explicit free-space boundary collar,
and tetrahedralizes every cube with the globally conforming Freudenthal split.
Junction-conforming multi-label marching tetrahedra place interface vertices at
label-transition edge midpoints, interface centroids on primal faces and the
corresponding pair/triple/quadruple material point inside each tetrahedron. A
triangle is emitted for every flag ``edge < face < tetrahedron`` whose edge
endpoints carry different labels. This has no marching-cubes case-table
ambiguity. Three labels on a primal face produce one shared valence-three edge,
and four labels in a tetrahedron produce all six sheets of the region graph.

Extraction is a host seeding/repair transaction. The source label state is read
once and fingerprinted, but never retained or synchronized. A successful result
owns a new :class:`MultiRegionSurfaceTopology` and state; multiregion validation
using exact predicates, including collision certification, must accept before a prepared
surface is exposed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._host_storage import declared_integer_storage_bytes
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    canonical_identifier,
    nonnegative_integer,
    positive_finite_float,
    unique_identifiers,
)
from ...discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateTemporaryStorage,
)
from ...typing import checked, Dim, HostInt64, Identifier, parse, Scope
from .._domain_coverage import PiecewiseLinearDomain
from ._contracts import (
    MultiRegionSurfaceCapacityEvidence,
    MultiRegionSurfaceCapacityPlan,
    MultiRegionSurfaceCounts,
    MultiRegionSurfaceEvidence,
    MultiRegionSurfaceStatus,
    MultiRegionSurfaceValidationPolicy,
)
from ._geometry import PreparedMultiRegionSurface
from ._seeding import MultiRegionSurfaceSeed
from ._state import MultiRegionSurfaceState
from ._topology import MultiRegionSurfaceTopology
from ._validation import geometry_id, validate_multiregion_surface


if TYPE_CHECKING:
    import phydrax.threshold_dynamics._contracts as contracts_owner

    from ...threshold_dynamics import LabelFieldState


LabelFieldSurfaceExtractionRoute: TypeAlias = Literal["uniform-grid", "sparse-grid"]


def _admit_label_field_phase(
    budget: CoordinateEnclosureBudget,
    work_units: int,
    scratch_bytes: int,
    result_bytes: int,
    retained_work: int,
    /,
) -> None:
    budget.ticket.checkpoint()
    budget.admit_work_bound(
        work_units + retained_work,
        storage_upper=scratch_bytes + result_bytes,
    )
    budget.reserve(work_units, scratch_bytes + result_bytes)


def _finish_label_field_phase(
    budget: CoordinateEnclosureBudget,
    lifetime: CoordinateTemporaryStorage,
    result: object,
    scratch_bytes: int,
    result_bytes: int,
    /,
) -> None:
    budget.ticket.checkpoint()
    with budget.live_storage() as storage:
        storage.promote_temporary_storage(lifetime, result_bytes)
        budget.retain_owner(result, temporary_storage_credit=storage)
    budget.release_temporary_storage(scratch_bytes)
    budget.ticket.checkpoint()

def _label_string_storage_upper(values: Iterable[str], /) -> int:
    return sum(256 + 4 * len(value) for value in values)

def _label_json_text_upper(values: Iterable[str], /) -> int:
    return sum(12 * len(value) + 2 for value in values)


class _ExtractionSiteDim(Dim, minimum=1):
    """Source sites of one prepared extraction."""


class LabelFieldSurfaceExtractionStatus(IntEnum):
    """Outcome of a label-field to explicit-surface seed transaction."""

    ACCEPTED = 0
    NO_INTERFACE = 1
    AMBIGUITY_CAPACITY_EXCEEDED = 2
    CAPACITY_EXCEEDED = 3
    UNSUPPORTED_VALENCE = 4
    VALIDATION_FAILED = 5


@final
class LabelFieldSurfaceLineage(StrictModule, NonTrainableState):
    """Immutable source-to-region identity map and extraction provenance.

    ``region_source_label_indices`` follows ``region_ids`` and uses ``-1`` only
    for a synthetic free-space boundary. The source state itself is deliberately
    absent: extraction is a one-way seed transaction, not a synchronization
    alias between threshold dynamics and explicit-surface evolution.
    """

    __strict_contract__ = True

    source_label_ids: tuple[str, ...] = eqx.field(static=True)
    source_label_counts: tuple[int, ...] = eqx.field(static=True)
    region_ids: tuple[str, ...] = eqx.field(static=True)
    region_source_label_indices: tuple[int, ...] = eqx.field(static=True)
    boundary_region_id: str = eqx.field(static=True)
    source_epoch: int = eqx.field(static=True)
    source_time: float = eqx.field(static=True)
    source_state_id: Identifier = eqx.field(static=True)
    source_binding_id: Identifier = eqx.field(static=True)
    source_route_id: Identifier = eqx.field(static=True)
    source_prepared_id: Identifier = eqx.field(static=True)
    source_site_id: Identifier = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)
    candidate_seed_id: str | None = eqx.field(static=True)
    algorithm: str = eqx.field(static=True)
    lineage_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_label_ids: tuple[str, ...],
        source_label_counts: tuple[int, ...],
        region_ids: tuple[str, ...],
        region_source_label_indices: tuple[int, ...],
        boundary_region_id: str,
        source_epoch: int,
        source_time: float,
        source_state_id: str,
        source_binding_id: str,
        source_route_id: str,
        source_prepared_id: str,
        source_site_id: str,
        prepared_id: str,
        candidate_seed_id: str | None,
    ) -> None:
        if len(source_label_counts) != len(source_label_ids):
            raise ValueError(
                f"LabelFieldSurfaceLineage.source_label_counts must follow source_label_ids; got {len(source_label_counts)} counts for {len(source_label_ids)} labels"
            )
        if len(region_source_label_indices) != len(region_ids):
            raise ValueError(
                f"LabelFieldSurfaceLineage.region_source_label_indices must follow region_ids; got {len(region_source_label_indices)} indices for {len(region_ids)} regions"
            )
        algorithm = "freudenthal-multilabel-marching-tetrahedra"
        self.source_label_ids = source_label_ids
        self.source_label_counts = source_label_counts
        self.region_ids = region_ids
        self.region_source_label_indices = region_source_label_indices
        self.boundary_region_id = boundary_region_id
        self.source_epoch = source_epoch
        self.source_time = source_time
        self.source_state_id = source_state_id
        self.source_binding_id = source_binding_id
        self.source_route_id = source_route_id
        self.source_prepared_id = source_prepared_id
        self.source_site_id = source_site_id
        self.prepared_id = prepared_id
        self.candidate_seed_id = candidate_seed_id
        self.algorithm = algorithm
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "label-field-surface-lineage",
                "source_state": source_state_id,
                "source_binding": source_binding_id,
                "source_route": source_route_id,
                "source_prepared": source_prepared_id,
                "source_sites": source_site_id,
                "prepared": prepared_id,
                "region_ids": list(region_ids),
                "region_source_label_indices": list(region_source_label_indices),
                "boundary_region_id": boundary_region_id,
                "source_epoch": source_epoch,
                "source_time": source_time,
                "candidate_seed": candidate_seed_id,
                "algorithm": algorithm,
            }
        )


@final
class LabelFieldSurfaceExtractionEvidence(StrictModule, NonTrainableState):
    """Resource, ambiguity, geometry-error and validation evidence.

    ``source_*`` geometry is the independent axis-aligned voxel estimator of the
    hard label field. ``extracted_*`` geometry is measured from the
    junction-conforming marching-tetrahedra surface. Relative errors use
    ``|a-b| / max(|a|, |b|)`` and therefore remain defined when deterministic
    tetrahedral ambiguity resolution introduces a pair that had only diagonal
    voxel contact.
    """

    __strict_contract__ = True

    status: LabelFieldSurfaceExtractionStatus = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    route: LabelFieldSurfaceExtractionRoute = eqx.field(static=True)
    ambiguous_cell_count: int = eqx.field(static=True)
    multi_label_tetrahedron_count: int = eqx.field(static=True)
    ambiguity_capacity: int = eqx.field(static=True)
    ambiguity_resolved: bool = eqx.field(static=True)
    maximum_edge_valence: int = eqx.field(static=True)
    maximum_vertex_region_pairs: int = eqx.field(static=True)
    unsupported_edge_count: int = eqx.field(static=True)
    source_finite_region_volumes: tuple[float, ...] = eqx.field(static=True)
    extracted_finite_region_volumes: tuple[float, ...] = eqx.field(static=True)
    finite_region_volume_relative_errors: tuple[float, ...] = eqx.field(static=True)
    region_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    source_pair_areas: tuple[float, ...] = eqx.field(static=True)
    extracted_pair_areas: tuple[float, ...] = eqx.field(static=True)
    pair_area_relative_errors: tuple[float, ...] = eqx.field(static=True)
    capacity: MultiRegionSurfaceCapacityEvidence
    validation: MultiRegionSurfaceEvidence | None
    collision_certified: bool = eqx.field(static=True)
    lineage_id: Identifier = eqx.field(static=True)
    evidence_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        status: LabelFieldSurfaceExtractionStatus,
        route: LabelFieldSurfaceExtractionRoute,
        ambiguous_cell_count: int,
        multi_label_tetrahedron_count: int,
        ambiguity_capacity: int,
        maximum_edge_valence: int,
        maximum_vertex_region_pairs: int,
        unsupported_edge_count: int,
        source_finite_region_volumes: tuple[float, ...],
        extracted_finite_region_volumes: tuple[float, ...],
        finite_region_volume_relative_errors: tuple[float, ...],
        region_pairs: tuple[tuple[str, str], ...],
        source_pair_areas: tuple[float, ...],
        extracted_pair_areas: tuple[float, ...],
        pair_area_relative_errors: tuple[float, ...],
        capacity: MultiRegionSurfaceCapacityEvidence,
        validation: MultiRegionSurfaceEvidence | None,
        lineage_id: str,
    ) -> None:
        status = parse(status, LabelFieldSurfaceExtractionStatus, "status")
        route_ = parse(route, LabelFieldSurfaceExtractionRoute, "route")
        accepted = status is LabelFieldSurfaceExtractionStatus.ACCEPTED
        collision = bool(
            accepted
            and validation is not None
            and validation.self_intersection_checked
            and validation.intersecting_pair_count == 0
            and validation.uncertain_pair_count == 0
            and not validation.candidate_capacity_exceeded
        )
        self.status = status
        self.accepted = accepted
        self.route = route_
        self.ambiguous_cell_count = ambiguous_cell_count
        self.multi_label_tetrahedron_count = multi_label_tetrahedron_count
        self.ambiguity_capacity = ambiguity_capacity
        self.ambiguity_resolved = (
            ambiguous_cell_count <= ambiguity_capacity
            and status is not LabelFieldSurfaceExtractionStatus.NO_INTERFACE
        )
        self.maximum_edge_valence = maximum_edge_valence
        self.maximum_vertex_region_pairs = maximum_vertex_region_pairs
        self.unsupported_edge_count = unsupported_edge_count
        self.source_finite_region_volumes = source_finite_region_volumes
        self.extracted_finite_region_volumes = extracted_finite_region_volumes
        self.finite_region_volume_relative_errors = finite_region_volume_relative_errors
        self.region_pairs = region_pairs
        self.source_pair_areas = source_pair_areas
        self.extracted_pair_areas = extracted_pair_areas
        self.pair_area_relative_errors = pair_area_relative_errors
        self.capacity = capacity
        self.validation = validation
        self.collision_certified = collision
        self.lineage_id = lineage_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "label-field-surface-extraction-evidence",
                "status": status.name,
                "route": route_,
                "ambiguous_cells": ambiguous_cell_count,
                "multi_label_tetrahedra": multi_label_tetrahedron_count,
                "capacity": capacity.resource_id,
                "validation": None if validation is None else validation.evidence_id,
                "lineage": lineage_id,
            }
        )


@final
class LabelFieldSurfaceExtractionResult(StrictModule, NonTrainableState):
    """Return accepted explicit geometry or a fail-closed extraction candidate.

    A rejected result may expose ``candidate_seed`` for diagnostics or a later
    repair attempt, but topology, state, and preparation remain absent.
    Successful results expose all four and are collision-certified.
    """

    candidate_seed: MultiRegionSurfaceSeed | None
    topology: MultiRegionSurfaceTopology | None
    state: MultiRegionSurfaceState | None
    surface: PreparedMultiRegionSurface | None
    lineage: LabelFieldSurfaceLineage
    evidence: LabelFieldSurfaceExtractionEvidence

    @checked
    def __init__(
        self,
        candidate_seed: MultiRegionSurfaceSeed | None,
        topology: MultiRegionSurfaceTopology | None,
        state: MultiRegionSurfaceState | None,
        surface: PreparedMultiRegionSurface | None,
        lineage: LabelFieldSurfaceLineage,
        evidence: LabelFieldSurfaceExtractionEvidence,
        /,
    ) -> None:
        authority = (topology, state, surface)
        if evidence.accepted:
            if topology is None:
                raise TypeError(
                    "LabelFieldSurfaceExtractionResult.topology must be present for accepted extraction; got None"
                )
            if state is None:
                raise TypeError(
                    "LabelFieldSurfaceExtractionResult.state must be present for accepted extraction; got None"
                )
            if surface is None:
                raise TypeError(
                    "LabelFieldSurfaceExtractionResult.surface must be present for accepted extraction; got None"
                )
            if candidate_seed is None or not evidence.collision_certified:
                raise ValueError(
                    "LabelFieldSurfaceExtractionResult.candidate_seed must be present with collision-certified evidence for accepted extraction; got absent seed or uncertified evidence"
                )
        elif any(value is not None for value in authority):
            raise ValueError(
                "LabelFieldSurfaceExtractionResult.geometry must be absent for refused extraction; got topology, state, or prepared surface"
            )
        self.candidate_seed = candidate_seed
        self.topology = topology
        self.state = state
        self.surface = surface
        self.lineage = lineage
        self.evidence = evidence

    @property
    def accepted(self) -> bool:
        return self.evidence.accepted

    def validate_source_integrity(
        self, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> PreparedMultiRegionSurface | None:
        """Validate the accepted surface and its prepared descendants."""
        if budget is None:
            return _validate_label_extraction_result(self)
        work, scratch_bytes, result_bytes = _label_surface_validation_bounds(self)
        if budget.retained_owner_objects.get(id(self)) is self:
            result_bytes -= _label_surface_shared_topology_bytes(self)
        with budget.temporary_scope() as lifetime:
            _admit_label_field_phase(
                budget, work, scratch_bytes, result_bytes, 256
            )
            rebuilt = _validate_label_extraction_result(self)
            if rebuilt is None:
                budget.release_temporary_storage(scratch_bytes + result_bytes)
                budget.ticket.checkpoint()
            else:
                _finish_label_field_phase(
                    budget, lifetime, rebuilt, scratch_bytes, result_bytes
                )
        return rebuilt


def _label_surface_validation_bounds(
    extraction: LabelFieldSurfaceExtractionResult, /
) -> tuple[int, int, int]:
    topology = extraction.topology
    surface = extraction.surface
    if topology is None or surface is None:
        return 64, 0, 0
    vertices, edges, faces = (
        topology.vertex_capacity,
        topology.edge_capacity,
        topology.face_capacity,
    )
    active_vertices, active_edges, active_faces = (
        topology.vertex_count,
        topology.edge_count,
        topology.face_count,
    )
    regions, pairs = topology.region_capacity, topology.region_pair_capacity
    valence, slots = topology.valence_width, topology.slot_width
    candidate_capacity = surface.validation_policy.intersection_candidate_capacity
    candidate_tests = min(
        candidate_capacity, active_faces * max(0, active_faces - 1) // 2
    )
    candidate_block = min(1 << 20, active_faces * active_faces)
    work = (
        512
        + active_faces * (64 + 24 * (4 * active_faces + 1).bit_length())
        + active_edges * (8 + 8 * valence + 24 * valence * valence)
        + active_vertices * (32 + 8 * slots)
        + regions * 4
        + pairs * 4
        + 512 * candidate_tests
    )
    # The BVH overlap walker has a 16-KiB fixed frontier; its candidate blocks
    # are separately bounded below. The edge-order kernel is quadratic in valence.
    scratch = (
        16384
        + 48 * active_vertices
        + 128 * active_faces
        + 64 * active_edges * valence
        + 64 * active_edges * valence * valence
        + 32 * active_vertices * slots
        + 512 * candidate_tests
        + 64 * candidate_block
    )
    # Capacity terms cover topology arrays/routes/BVH; 1024 bytes per region
    # covers the three scalar tuples plus finite-region ids in validation evidence.
    # The fixed metadata term is 64 declared object owners at 256 bytes each.
    result = (
        vertices * (9 + 9 * slots)
        + edges * (17 + 24 * valence)
        + 512 * faces
        + 1024 * regions
        + 17 * pairs
        + _label_string_storage_upper(
            itertools.chain(
                topology.region_ids,
                (
                    topology.plan.resource_id,
                    topology.plan.plan_id,
                    topology.topology_id,
                    topology.lineage_id,
                    topology.domain,
                    surface.validation_policy.profile,
                    surface.validation_policy.policy_id,
                    surface.evidence.predicate_mode,
                    surface.evidence.geometry_id,
                    surface.evidence.evidence_id,
                    surface.prepared_id,
                ),
            )
        )
        + 16384
    )
    return work, scratch, result

def _label_surface_shared_topology_bytes(
    extraction: LabelFieldSurfaceExtractionResult, /
) -> int:
    topology = extraction.topology
    if topology is None:
        return 0
    surface = extraction.surface
    vertices, edges, faces = (
        topology.vertex_capacity,
        topology.edge_capacity,
        topology.face_capacity,
    )
    regions, pairs = topology.region_capacity, topology.region_pair_capacity
    valence, slots = topology.valence_width, topology.slot_width
    shared_strings = itertools.chain(
        topology.region_ids,
        (
            topology.plan.resource_id,
            topology.plan.plan_id,
            topology.topology_id,
            topology.lineage_id,
            topology.domain,
        ),
        ()
        if surface is None
        else (
            surface.validation_policy.profile,
            surface.validation_policy.policy_id,
        ),
    )
    return (
        vertices * (9 + 9 * slots)
        + edges * (17 + 16 * valence)
        + 137 * faces
        + 2 * regions
        + 17 * pairs
        + _label_string_storage_upper(shared_strings)
    )


def _validate_label_extraction_result(
    extraction: LabelFieldSurfaceExtractionResult, /
) -> PreparedMultiRegionSurface | None:
    LabelFieldSurfaceExtractionResult(
        extraction.candidate_seed,
        extraction.topology,
        extraction.state,
        extraction.surface,
        extraction.lineage,
        extraction.evidence,
    )
    if not extraction.accepted:
        return None
    topology, state, surface = (
        extraction.topology,
        extraction.state,
        extraction.surface,
    )
    if topology is None or state is None or surface is None:
        raise ValueError(
            "LabelFieldSurfaceExtractionResult.geometry must retain topology, state, and surface for accepted extraction; got absent retained geometry"
        )
    state.require_topology(topology)
    rebuilt = surface.validate_restored(topology, state)
    validation = extraction.evidence.validation
    if (
        validation is None
        or validation.evidence_id != surface.evidence.evidence_id
        or extraction.evidence.lineage_id != extraction.lineage.lineage_id
        or extraction.lineage.region_ids != topology.region_ids
    ):
        raise ValueError(
            "LabelFieldSurfaceExtractionResult.evidence must match retained validation, lineage, and regions; got inconsistent evidence"
        )
    return rebuilt


def _label_extraction_storage_upper(
    extraction: LabelFieldSurfaceExtractionResult, /
) -> int:
    seed = extraction.candidate_seed
    lineage, evidence = extraction.lineage, extraction.evidence
    validation = evidence.validation
    seed_bytes = 8192
    if seed is not None:
        seed_bytes += sum(
            value.nbytes
            for value in (
                seed.positions,
                seed.faces,
                seed.face_labels,
                seed.vertex_global_ids,
                seed.face_global_ids,
            )
        )
    identity_groups = (
        lineage.source_label_ids,
        lineage.region_ids,
        (
            lineage.boundary_region_id,
            lineage.source_state_id,
            lineage.source_binding_id,
            lineage.source_route_id,
            lineage.source_prepared_id,
            lineage.source_site_id,
            lineage.prepared_id,
            evidence.evidence_id,
            evidence.lineage_id,
        ),
        () if seed is None else (seed.source, seed.seed_id),
        () if validation is None else validation.finite_region_ids,
        ()
        if validation is None
        else (
            validation.predicate_mode,
            validation.topology_id,
            validation.lineage_id,
            validation.geometry_id,
            validation.evidence_id,
        ),
    )
    identity_bytes = _label_string_storage_upper(
        itertools.chain.from_iterable(identity_groups)
    )
    extraction_metadata_bytes = (
        2048 * len(lineage.region_ids)
        + 2048 * len(evidence.region_pairs)
        + 512 * len(lineage.source_label_ids)
        + (0 if validation is None else 1024 * len(validation.finite_region_ids))
    )
    topology = extraction.topology
    state = extraction.state
    if topology is None:
        return seed_bytes + extraction_metadata_bytes + identity_bytes
    state_bytes = 0
    if state is not None:
        state_bytes = (
            2
            * topology.vertex_capacity
            * 3
            * np.dtype(topology.plan.coordinate_dtype).itemsize
        )
    return (
        seed_bytes
        + state_bytes
        + extraction_metadata_bytes
        + identity_bytes
        + _label_surface_validation_bounds(extraction)[2]
    )


def _label_binding_resource_bounds(
    extraction: LabelFieldSurfaceExtractionResult,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> tuple[int, int, int]:
    topology = extraction.topology
    if topology is None:
        vertices = edges = faces = regions = pairs = 0
        vertex_capacity = edge_capacity = face_capacity = 0
        region_capacity = pair_capacity = slots = valence = 0
    else:
        vertices, edges, faces = (
            topology.vertex_count,
            topology.edge_count,
            topology.face_count,
        )
        regions, pairs = topology.region_count, topology.region_pair_count
        (
            vertex_capacity,
            edge_capacity,
            face_capacity,
            region_capacity,
            pair_capacity,
            slots,
            valence,
        ) = (
            topology.vertex_capacity,
            topology.edge_capacity,
            topology.face_capacity,
            topology.region_capacity,
            topology.region_pair_capacity,
            topology.slot_width,
            topology.valence_width,
        )
    region_names = () if topology is None else topology.region_ids
    maximum_region_name_length = max(
        (len(value) for value in region_names), default=0
    )
    interface_count = regions * max(0, regions - 1) // 2
    base_json_bytes = _label_json_text_upper(
        itertools.chain(
            region_names,
            (
                coordinate_contract.spatial_id,
                extraction.evidence.evidence_id,
                extraction.lineage.lineage_id,
            ),
        )
    )
    pair_label_json_bytes = 2 * (12 * maximum_region_name_length + 2)
    pair_entry_json_bytes = pair_label_json_bytes + 12 * 64 + 2
    binding_json_bytes = base_json_bytes + interface_count * pair_entry_json_bytes
    binding_json_work = (
        3 * base_json_bytes
        + interface_count * (pair_label_json_bytes + pair_entry_json_bytes)
    )
    unit = coordinate_contract.length_unit
    dimension = unit.dimension
    coordinate_string_bytes = _label_string_storage_upper(
        itertools.chain(
            (
                coordinate_contract.coordinate_system,
                coordinate_contract.reference_frame,
                coordinate_contract.length_coordinate_kind,
                coordinate_contract.spatial_id,
                unit.symbol,
                unit.reference_system_id,
                unit.unit_id,
                dimension.dimension_id,
            ),
            (axis for axis, _, _ in dimension.terms),
        )
    )
    coordinate_integer_bytes = (
        declared_integer_storage_bytes(unit.scale_numerator.bit_length())
        + declared_integer_storage_bytes(unit.scale_denominator.bit_length())
        + sum(
            declared_integer_storage_bytes(abs(value).bit_length())
            for _, numerator, denominator in dimension.terms
            for value in (numerator, denominator)
        )
    )
    binding = (
        8192
        + 24 * vertices
        + 40 * faces
        + 16 * regions
        + 1024 * interface_count
        + coordinate_string_bytes
        + coordinate_integer_bytes
    )
    work = (
        512
        + 16
        * (
            vertex_capacity
            + edge_capacity
            + face_capacity
            + region_capacity
            + pair_capacity
        )
        + 24 * faces * (4 * max(1, faces).bit_length())
        + 32 * vertices
        + 64 * faces
        + regions * regions
        + len(coordinate_contract.spatial_id)
        + binding_json_work
    )
    scratch = (
        8192
        + 16
        * (
            vertex_capacity
            + edge_capacity
            + face_capacity
            + region_capacity
            + pair_capacity
        )
        + 8 * vertex_capacity * slots
        + 64 * vertices
        + 256 * faces
        + 512 * region_capacity
        + 1024 * pair_capacity
        + 512
        + 2 * binding_json_bytes
    )
    return work, scratch, binding


def _prepare_label_field_binding_facts(
    extraction: LabelFieldSurfaceExtractionResult,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> tuple[
    PiecewiseLinearDomain,
    LabelFieldSurfaceExtractionResult,
    SpatialCoordinateContract,
    LabelFieldSurfaceLineage,
    LabelFieldSurfaceExtractionEvidence,
    str,
    str,
    str,
    str,
    tuple[tuple[str, str, str, bool], ...],
]:
    if not extraction.accepted or not extraction.evidence.collision_certified:
        raise ValueError(
            "LabelFieldVolumeBinding.extraction must be accepted and collision-certified; got refused or uncertified extraction"
        )
    topology = extraction.topology
    state = extraction.state
    if topology is None or state is None:
        raise ValueError(
            "LabelFieldVolumeBinding.extraction must retain represented topology and state; got absent geometry"
        )
    state.require_topology(topology)
    for mask, count in (
        (topology.vertex_active, topology.vertex_count),
        (topology.edge_active, topology.edge_count),
        (topology.face_active, topology.face_count),
        (topology.region_active, topology.region_count),
        (topology.pair_active, topology.region_pair_count),
    ):
        active = np.asarray(mask, dtype=np.bool_)
        if not np.array_equal(active, np.arange(active.size) < count):
            raise ValueError(
                f"LabelFieldVolumeBinding.active_masks must use active prefixes; got a mask inconsistent with count {count}"
            )
    if (
        not np.array_equal(state.vertex_active, topology.vertex_active)
        or not np.array_equal(state.region_active, topology.region_active)
        or not np.array_equal(state.slot_active, topology.slot_active)
    ):
        raise ValueError(
            "LabelFieldVolumeBinding.state_masks must match topology masks; got inconsistent vertex, region, or slot masks"
        )
    positions = np.asarray(state.positions[: topology.vertex_count], dtype=np.float64)
    validation = extraction.evidence.validation
    if (
        validation is None
        or validation.topology_id != topology.topology_id
        or validation.lineage_id != topology.lineage_id
        or validation.geometry_id != geometry_id(topology, positions)
        or extraction.evidence.lineage_id != extraction.lineage.lineage_id
        or extraction.lineage.region_ids != topology.region_ids
    ):
        raise ValueError(
            "LabelFieldVolumeBinding.evidence must match reconstructed topology, geometry, lineage, and regions; got stale extraction evidence"
        )
    boundary = extraction.lineage.boundary_region_id
    if boundary not in topology.region_ids:
        raise ValueError(
            f"LabelFieldVolumeBinding.boundary_region_id must identify an explicit exterior region; got {boundary!r}"
        )
    finite = tuple(value for value in topology.region_ids if value != boundary)
    if not finite:
        raise ValueError(
            "LabelFieldVolumeBinding.regions must include a finite material region; got only the boundary region"
        )
    mapping = np.asarray(
        [
            -1 if value == boundary else finite.index(value)
            for value in topology.region_ids
        ],
        dtype=np.int64,
    )
    source = canonical_fingerprint(
        {
            "kind": "label-field-volume-source",
            "lineage": extraction.lineage.lineage_id,
            "coordinates": coordinate_contract.spatial_id,
        }
    )
    domain = PiecewiseLinearDomain(
        positions,
        topology.host_faces(),
        mapping[topology.host_face_labels()],
        finite,
        source_id=source,
    )
    interior = np.unique(
        np.sort(
            domain.facet_regions[np.all(domain.facet_regions >= 0, axis=1)], axis=0
        ),
        axis=0,
    )
    interface_definitions = tuple(
        (
            canonical_fingerprint(
                {
                    "kind": "reconstructed-material-interface",
                    "source": source,
                    "regions": tuple(sorted((finite[first], finite[second]))),
                }
            ),
            finite[first],
            finite[second],
            True,
        )
        for first, second in interior.tolist()
    )
    interpretation = "freudenthal-categorical-interface-reconstruction"
    binding_id = canonical_fingerprint(
        {
            "kind": "label-field-volume-binding",
            "source": source,
            "domain": domain.domain_id,
            "topology": topology.topology_id,
            "surface_lineage": topology.lineage_id,
            "vertex_global_ids": array_tree_fingerprint(
                np.asarray(
                    topology.vertex_global_ids[: topology.vertex_count],
                    dtype=np.int64,
                )
            ),
            "face_global_ids": array_tree_fingerprint(
                np.asarray(
                    topology.face_global_ids[: topology.face_count], dtype=np.int64
                )
            ),
            "epoch": topology.epoch,
            "validation": validation.evidence_id,
            "extraction": extraction.evidence.evidence_id,
            "interpretation": interpretation,
            "interfaces": interface_definitions,
        }
    )
    return (
        domain,
        extraction,
        coordinate_contract,
        extraction.lineage,
        extraction.evidence,
        interpretation,
        source,
        domain.source_revision,
        binding_id,
        interface_definitions,
    )


def _prepare_label_field_binding(
    extraction: LabelFieldSurfaceExtractionResult,
    coordinate_contract: SpatialCoordinateContract,
    /,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> tuple[
    PiecewiseLinearDomain,
    LabelFieldSurfaceExtractionResult,
    SpatialCoordinateContract,
    LabelFieldSurfaceLineage,
    LabelFieldSurfaceExtractionEvidence,
    str,
    str,
    str,
    str,
    tuple[tuple[str, str, str, bool], ...],
]:
    if budget is None:
        extraction.validate_source_integrity()
        return _prepare_label_field_binding_facts(extraction, coordinate_contract)
    work, scratch_bytes, result_bytes = _label_surface_validation_bounds(extraction)
    result_bytes -= _label_surface_shared_topology_bytes(extraction)
    with budget.temporary_scope():
        _admit_label_field_phase(
            budget, work, scratch_bytes, result_bytes, 256
        )
        rebuilt = _validate_label_extraction_result(extraction)
        try:
            facts = _prepare_label_field_binding_facts(
                extraction, coordinate_contract
            )
        finally:
            del rebuilt
            budget.release_temporary_storage(scratch_bytes + result_bytes)
            budget.ticket.checkpoint()
    return facts


def _require_label_field_binding_facts(
    owner: LabelFieldVolumeBinding,
    facts: tuple[
        PiecewiseLinearDomain,
        LabelFieldSurfaceExtractionResult,
        SpatialCoordinateContract,
        LabelFieldSurfaceLineage,
        LabelFieldSurfaceExtractionEvidence,
        str,
        str,
        str,
        str,
        tuple[tuple[str, str, str, bool], ...],
    ],
    /,
) -> None:
    (
        domain,
        _,
        _,
        lineage,
        extraction_evidence,
        interpretation,
        source_id,
        source_revision,
        binding_id,
        interface_definitions,
    ) = facts
    if (
        binding_id != owner.binding_id
        or source_id != owner.source_id
        or source_revision != owner.source_revision
        or interface_definitions != owner.interface_definitions
        or interpretation != owner.interpretation
        or array_tree_fingerprint(domain) != array_tree_fingerprint(owner.domain)
        or lineage.lineage_id != owner.lineage.lineage_id
        or extraction_evidence.evidence_id != owner.extraction_evidence.evidence_id
    ):
        raise ValueError(
            "LabelFieldVolumeBinding.restored must match retained extraction and binding identity; got stale binding"
        )
@final
class LabelFieldVolumeBinding(StrictModule, NonTrainableState):
    """Bind represented volume to an accepted shared reconstructed interface.

    The input is categorical lattice samples. The represented volume is the
    explicitly selected Freudenthal marching-tetrahedra reconstruction, not
    occupied voxel cells and not a topology certificate for an unknown field
    between the samples. Its extraction error and source ancestry remain
    attached to the domain consumed by native constrained recovery.
    """

    domain: PiecewiseLinearDomain
    extraction: LabelFieldSurfaceExtractionResult
    coordinate_contract: SpatialCoordinateContract
    lineage: LabelFieldSurfaceLineage
    extraction_evidence: LabelFieldSurfaceExtractionEvidence
    interpretation: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    interface_definitions: tuple[tuple[str, str, str, bool], ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        extraction: LabelFieldSurfaceExtractionResult,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        if budget is None:
            facts = _prepare_label_field_binding(
                extraction, coordinate_contract, budget=None
            )
            (
                self.domain,
                self.extraction,
                self.coordinate_contract,
                self.lineage,
                self.extraction_evidence,
                self.interpretation,
                self.source_id,
                self.source_revision,
                self.binding_id,
                self.interface_definitions,
            ) = facts
            return
        work, scratch_bytes, binding_bytes = _label_binding_resource_bounds(
            extraction, coordinate_contract
        )
        input_bytes = (
            0
            if budget.retained_owner_objects.get(id(extraction)) is extraction
            else _label_extraction_storage_upper(extraction)
        )
        result_bytes = binding_bytes + input_bytes
        with budget.temporary_scope() as lifetime:
            _admit_label_field_phase(
                budget, work, scratch_bytes, result_bytes, 512
            )
            facts = _prepare_label_field_binding(
                extraction, coordinate_contract, budget=budget
            )
            (
                self.domain,
                self.extraction,
                self.coordinate_contract,
                self.lineage,
                self.extraction_evidence,
                self.interpretation,
                self.source_id,
                self.source_revision,
                self.binding_id,
                self.interface_definitions,
            ) = facts
            del facts
            _finish_label_field_phase(
                budget, lifetime, self, scratch_bytes, result_bytes
            )

    def validate_source_integrity(
        self, /, *, budget: CoordinateEnclosureBudget | None = None
    ) -> None:
        """Re-admit the retained extraction and compare its binding identity."""
        if budget is None:
            facts = _prepare_label_field_binding(
                self.extraction, self.coordinate_contract, budget=None
            )
            _require_label_field_binding_facts(self, facts)
            return
        work, scratch_bytes, binding_bytes = _label_binding_resource_bounds(
            self.extraction, self.coordinate_contract
        )
        input_bytes = (
            0
            if budget.retained_owner_objects.get(id(self.extraction))
            is self.extraction
            else _label_extraction_storage_upper(self.extraction)
        )
        result_bytes = binding_bytes + input_bytes
        with budget.temporary_scope():
            _admit_label_field_phase(
                budget, work, scratch_bytes, result_bytes, 0
            )
            facts = _prepare_label_field_binding(
                self.extraction, self.coordinate_contract, budget=budget
            )
            try:
                _require_label_field_binding_facts(self, facts)
            finally:
                del facts


@final
class LabelFieldSurfaceExtractionPlan(StrictModule, NonTrainableState):
    """Static identity, geometry, resource and validation policy for extraction.

    ``boundary_region_id`` may name one source label (that label becomes E's
    unbounded boundary) or a new synthetic label. Every other active source
    label becomes a finite E region with exactly the same stable identifier.
    The output is always free-space: periodic wrapping is intentionally not
    inferred from a threshold route because E has no unwrapped periodic volume
    representation.
    """

    __strict_contract__ = True

    label_ids: tuple[str, ...] = eqx.field(static=True)
    spacing: tuple[float, float, float] = eqx.field(static=True)
    origin: tuple[float, float, float] = eqx.field(static=True)
    boundary_region_id: str = eqx.field(static=True)
    capacity_plan: MultiRegionSurfaceCapacityPlan
    validation_policy: MultiRegionSurfaceValidationPolicy
    maximum_ambiguous_cells: int = eqx.field(static=True)
    source: Identifier = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        label_ids: Sequence[str],
        capacity_plan: MultiRegionSurfaceCapacityPlan,
        /,
        *,
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        boundary_region_id: str = "ambient",
        validation_policy: MultiRegionSurfaceValidationPolicy | None = None,
        maximum_ambiguous_cells: int = 1_000_000,
        source: str = "threshold-label-field",
    ) -> None:
        ids = unique_identifiers(label_ids, "label_ids")
        if len(spacing) != 3:
            raise ValueError(
                f"LabelFieldSurfaceExtractionPlan.spacing must contain three Cartesian values; got {spacing!r}"
            )
        spacing_: tuple[float, float, float] = (
            positive_finite_float(spacing[0], "spacing[0]"),
            positive_finite_float(spacing[1], "spacing[1]"),
            positive_finite_float(spacing[2], "spacing[2]"),
        )
        if len(origin) != 3:
            raise ValueError(
                f"LabelFieldSurfaceExtractionPlan.origin must contain three Cartesian values; got {origin!r}"
            )
        origin_: tuple[float, float, float] = (
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
        )
        if not all(math.isfinite(value) for value in origin_):
            raise ValueError(
                f"LabelFieldSurfaceExtractionPlan.origin must be finite; got {origin_!r}"
            )
        boundary = canonical_identifier(boundary_region_id, "boundary_region_id")
        policy = (
            MultiRegionSurfaceValidationPolicy()
            if validation_policy is None
            else validation_policy
        )
        if not policy.check_self_intersection:
            raise ValueError(
                "LabelFieldSurfaceExtractionPlan.validation_policy must enable collision certification; got check_self_intersection=False"
            )
        ambiguity = nonnegative_integer(
            maximum_ambiguous_cells, "maximum_ambiguous_cells"
        )
        source_ = canonical_identifier(source, "source")
        self.label_ids = ids
        self.spacing = spacing_
        self.origin = origin_
        self.boundary_region_id = boundary
        self.capacity_plan = capacity_plan
        self.validation_policy = policy
        self.maximum_ambiguous_cells = ambiguity
        self.source = source_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "label-field-surface-extraction-plan",
                "label_ids": list(ids),
                "spacing": list(spacing_),
                "origin": list(origin_),
                "boundary_region_id": boundary,
                "capacity_plan": capacity_plan.plan_id,
                "validation_policy": policy.policy_id,
                "maximum_ambiguous_cells": ambiguity,
                "source": source_,
            }
        )

    def prepare(
        self,
        grid_shape: tuple[int, int, int],
        /,
        *,
        site_coordinates: ArrayLike | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PreparedLabelFieldSurfaceExtraction:
        """Prepare uniform sites or an explicit sparse subset of a 3D grid."""
        return PreparedLabelFieldSurfaceExtraction(
            self, grid_shape, site_coordinates=site_coordinates, budget=budget
        )


def _label_grid_shape(
    grid_shape: tuple[int, int, int], /
) -> tuple[int, int, int]:
    if len(grid_shape) != 3:
        raise ValueError(
            f"PreparedLabelFieldSurfaceExtraction.grid_shape must contain three dimensions; got {grid_shape!r}"
        )
    shape: tuple[int, int, int] = (
        nonnegative_integer(grid_shape[0], "grid_shape[0]"),
        nonnegative_integer(grid_shape[1], "grid_shape[1]"),
        nonnegative_integer(grid_shape[2], "grid_shape[2]"),
    )
    if any(value == 0 for value in shape):
        raise ValueError(
            f"PreparedLabelFieldSurfaceExtraction.grid_shape must have positive dimensions; got {shape!r}"
        )
    return shape


def _label_site_layout(
    shape: tuple[int, int, int],
    site_coordinates: ArrayLike | None,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    tuple[int, ...],
    LabelFieldSurfaceExtractionRoute,
]:
    if site_coordinates is None:
        coordinates = np.stack(
            np.meshgrid(
                *(np.arange(value, dtype=np.int64) for value in shape),
                indexing="ij",
            ),
            axis=-1,
        ).reshape((-1, 3))
        order = np.arange(coordinates.shape[0], dtype=np.int64)
        source_shape = shape
        route: LabelFieldSurfaceExtractionRoute = "uniform-grid"
    else:
        raw = np.asarray(site_coordinates)
        if raw.ndim != 2 or raw.shape[1] != 3 or raw.dtype.kind not in "iu":
            raise ValueError(
                f"PreparedLabelFieldSurfaceExtraction.site_coordinates must be an integer (sites, 3) array; got {raw.dtype} {raw.shape}"
            )
        coordinates_ = raw.astype(np.int64)
        if coordinates_.shape[0] == 0:
            raise ValueError(
                "PreparedLabelFieldSurfaceExtraction.site_coordinates must be nonempty; got zero sites"
            )
        bounds = np.asarray(shape, dtype=np.int64)
        if np.any(coordinates_ < 0) or np.any(coordinates_ >= bounds):
            raise ValueError(
                f"PreparedLabelFieldSurfaceExtraction.site_coordinates must lie inside grid_shape {shape!r}; got out-of-range coordinates"
            )
        order = np.lexsort(
            (coordinates_[:, 2], coordinates_[:, 1], coordinates_[:, 0])
        )
        coordinates = coordinates_[order]
        if np.any(np.all(coordinates[1:] == coordinates[:-1], axis=1)):
            raise ValueError(
                "PreparedLabelFieldSurfaceExtraction.site_coordinates must be unique; got duplicate sites"
            )
        source_shape = (coordinates.shape[0],)
        route = "sparse-grid"
    scope = Scope()
    coordinates = parse(
        coordinates,
        HostInt64[_ExtractionSiteDim, Literal[3]],
        "site_coordinates",
        scope=scope,
    )
    order = parse(order, HostInt64[_ExtractionSiteDim], "source_order", scope=scope)
    return coordinates, order, source_shape, route


def _label_site_input_bounds(
    grid_shape: tuple[int, int, int],
    site_coordinates: ArrayLike | None,
    /,
) -> tuple[int, int]:
    if site_coordinates is None:
        return math.prod(grid_shape), 0
    if isinstance(site_coordinates, np.ndarray):
        return site_coordinates.shape[0], 0
    if isinstance(site_coordinates, Sequence):
        site_count = len(site_coordinates)
        input_elements = 0
        for row in site_coordinates:
            row_shape = getattr(row, "shape", None)
            if row_shape is not None:
                if len(row_shape) > 1:
                    raise TypeError(
                        "PreparedLabelFieldSurfaceExtraction.site_coordinates sequence rows must be one-dimensional; got a nested row"
                    )
                input_elements += math.prod(row_shape) if row_shape else 1
            elif isinstance(row, Sequence):
                if any(
                    isinstance(value, Sequence)
                    or (
                        isinstance(value, np.ndarray) and value.ndim > 0
                    )
                    or (
                        getattr(value, "shape", None) is not None
                        and len(value.shape) > 0
                    )
                    for value in row
                ):
                    raise TypeError(
                        "PreparedLabelFieldSurfaceExtraction.site_coordinates sequence rows must contain scalar values; got nested coordinates"
                    )
                input_elements += len(row)
            else:
                input_elements += 1
        return site_count, 8 * input_elements + 32 * site_count
    site_shape = getattr(site_coordinates, "shape", None)
    if site_shape is None:
        raise TypeError(
            "PreparedLabelFieldSurfaceExtraction.site_coordinates must expose an array shape or be a finite sequence; got an unbounded array-like"
        )
    site_count = int(site_shape[0]) if len(site_shape) else 0
    input_elements = math.prod(site_shape) if site_shape else 1
    dtype = getattr(site_coordinates, "dtype", None)
    input_itemsize = 8 if dtype is None else np.dtype(dtype).itemsize
    return site_count, input_elements * input_itemsize

def _set_prepared_label_layout(
    owner: PreparedLabelFieldSurfaceExtraction,
    plan: LabelFieldSurfaceExtractionPlan,
    shape: tuple[int, int, int],
    layout: tuple[
        np.ndarray,
        np.ndarray,
        tuple[int, ...],
        LabelFieldSurfaceExtractionRoute,
    ],
    /,
) -> None:
    coordinates, order, source_shape, route = layout
    owner.plan = plan
    owner.site_coordinates = coordinates
    owner.source_order = order
    owner.grid_shape = shape
    owner.source_shape = source_shape
    owner.route = route
    owner.site_count = coordinates.shape[0]
    owner.prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-label-field-surface-extraction",
            "plan": plan.plan_id,
            "grid_shape": list(shape),
            "route": route,
            "site_coordinates": array_tree_fingerprint(coordinates),
        }
    )


@final
class PreparedLabelFieldSurfaceExtraction(StrictModule, NonTrainableState):
    """Prepare a deterministic source-site layout reusable across threshold epochs."""

    __strict_contract__ = True

    plan: LabelFieldSurfaceExtractionPlan
    site_coordinates: HostInt64[_ExtractionSiteDim, Literal[3]]
    source_order: HostInt64[_ExtractionSiteDim]
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    route: LabelFieldSurfaceExtractionRoute = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plan: LabelFieldSurfaceExtractionPlan,
        grid_shape: tuple[int, int, int],
        /,
        *,
        site_coordinates: ArrayLike | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        shape = _label_grid_shape(grid_shape)
        if budget is None:
            layout = _label_site_layout(shape, site_coordinates)
            _set_prepared_label_layout(self, plan, shape, layout)
            return
        if isinstance(site_coordinates, Sequence):
            budget.reserve(len(site_coordinates))
        site_count, input_bytes = _label_site_input_bounds(shape, site_coordinates)
        if site_coordinates is None:
            # The three meshgrid tables coexist with the stacked coordinates.
            scratch_bytes = 48 * site_count + 8192
            work_units = 12 * site_count + 256
        else:
            # Cast/sort copies coexist with the retained ordered coordinates.
            scratch_bytes = 64 * site_count + input_bytes + 8192
            work_units = (
                8 + 4 * max(1, site_count.bit_length())
            ) * site_count + 256
            work_units += input_bytes
        work_units += 8 * len(plan.label_ids)
        result_bytes = (
            32 * site_count
            + 8192
            + 512 * len(plan.label_ids)
            + _label_string_storage_upper(
                itertools.chain(
                    plan.label_ids,
                    (
                        plan.boundary_region_id,
                        plan.source,
                        plan.plan_id,
                        plan.capacity_plan.resource_id,
                        plan.capacity_plan.plan_id,
                        plan.validation_policy.profile,
                        plan.validation_policy.policy_id,
                    ),
                )
            )
        )
        retained_work = 64 + 8 * len(plan.label_ids)
        with budget.temporary_scope() as lifetime:
            _admit_label_field_phase(
                budget, work_units, scratch_bytes, result_bytes, retained_work
            )
            layout = _label_site_layout(shape, site_coordinates)
            _set_prepared_label_layout(self, plan, shape, layout)
            _finish_label_field_phase(
                budget, lifetime, self, scratch_bytes, result_bytes
            )

    def extract(
        self,
        state: LabelFieldState,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> LabelFieldSurfaceExtractionResult:
        """Extract and certify geometry from one label-field epoch."""
        return _extract(self, state, budget=budget)


@dataclass(frozen=True, slots=True)
class _RegionMap:
    dense_labels: np.ndarray
    extraction_origin: tuple[float, float, float]
    region_ids: tuple[str, ...]
    region_kinds: tuple[Literal["finite", "boundary"], ...]
    source_indices: tuple[int, ...]
    boundary_index: int
    source_counts: tuple[int, ...]
    finite_source_volumes: tuple[float, ...]
    source_state_id: str
    source_binding_id: str
    source_route_id: str
    source_prepared_id: str
    source_site_id: str
    source_epoch: int
    source_time: float


@dataclass(frozen=True, slots=True)
class _DualSurface:
    positions: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray


_TETRAHEDRON_PERMUTATIONS: tuple[tuple[int, int, int], ...] = tuple(
    (permutation[0], permutation[1], permutation[2])
    for permutation in itertools.permutations((0, 1, 2))
)
_TETRAHEDRON_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
_FACE_EDGES = ((0, 1), (0, 2), (1, 2))
_CUBE_CORNERS = tuple(itertools.product((0, 1), repeat=3))
_CUBE_EDGES = tuple(
    (first, second)
    for first, a in enumerate(_CUBE_CORNERS)
    for second, b in enumerate(_CUBE_CORNERS)
    if first < second and sum(abs(x - y) for x, y in zip(a, b, strict=True)) == 1
)


def _host_state(
    prepared: PreparedLabelFieldSurfaceExtraction,
    state: contracts_owner.LabelFieldState,
    /,
) -> tuple[np.ndarray, np.ndarray, int, float]:

    # This optional source binding composes its geometry owner; resolve that upper-layer cycle only for the request.
    import phydrax.threshold_dynamics._contracts as contracts_owner

    match state:
        case contracts_owner.LabelFieldState():
            if state.label_ids != prepared.plan.label_ids:
                raise ValueError(
                    f"PreparedLabelFieldSurfaceExtraction.state_label_ids must match extraction plan label order {prepared.plan.label_ids!r}; got {state.label_ids!r}"
                )
            labels = np.asarray(state.labels, dtype=np.int64)
            if labels.shape != prepared.source_shape:
                raise ValueError(
                    f"PreparedLabelFieldSurfaceExtraction.state_labels must have prepared source shape {prepared.source_shape}; got {labels.shape}"
                )
            flat = labels.reshape(-1)[prepared.source_order]
            if np.any(flat < 0) or np.any(flat >= state.label_count):
                raise ValueError(
                    "PreparedLabelFieldSurfaceExtraction.state_labels must index extraction plan label_ids; got out-of-range labels"
                )
            active = np.asarray(state.active_labels, dtype=np.bool_)
            counts = np.bincount(flat, minlength=state.label_count)
            if np.any((counts > 0) & ~active):
                raise ValueError(
                    "PreparedLabelFieldSurfaceExtraction.active_labels must activate every label with source sites; got an inactive present label"
                )
            return (
                flat,
                active,
                int(np.asarray(state.epoch)),
                float(np.asarray(state.time)),
            )
        case _:
            raise TypeError(
                f"PreparedLabelFieldSurfaceExtraction.state must be a LabelFieldState; got {type(state).__name__}"
            )


def _region_map(
    prepared: PreparedLabelFieldSurfaceExtraction, state: LabelFieldState, /
) -> _RegionMap:
    labels, active, epoch, time = _host_state(prepared, state)
    plan = prepared.plan
    counts = np.bincount(labels, minlength=len(plan.label_ids))
    boundary_source = (
        plan.label_ids.index(plan.boundary_region_id)
        if plan.boundary_region_id in plan.label_ids
        else None
    )
    present = [
        index for index, count in enumerate(counts) if count > 0 and bool(active[index])
    ]
    if boundary_source is not None and boundary_source not in present:
        present.append(boundary_source)
        present.sort()
    region_ids = tuple(plan.label_ids[index] for index in present)
    source_indices = tuple(present)
    if boundary_source is None:
        region_ids += (plan.boundary_region_id,)
        source_indices += (-1,)
        boundary = len(region_ids) - 1
    else:
        boundary = present.index(boundary_source)
    kinds: tuple[Literal["finite", "boundary"], ...] = tuple(
        "boundary" if index == boundary else "finite" for index in range(len(region_ids))
    )
    source_to_region = np.full((len(plan.label_ids),), -1, dtype=np.int64)
    for region, source in enumerate(source_indices):
        if source >= 0:
            source_to_region[source] = region
    coordinates = np.asarray(prepared.site_coordinates, dtype=np.int64)
    match prepared.route:
        case "uniform-grid":
            offset = np.zeros((3,), dtype=np.int64)
            dense_shape = prepared.grid_shape
        case "sparse-grid":
            offset = np.min(coordinates, axis=0)
            upper = np.max(coordinates, axis=0)
            dense_shape = (
                int(upper[0] - offset[0] + 1),
                int(upper[1] - offset[1] + 1),
                int(upper[2] - offset[2] + 1),
            )
        case _:
            raise RuntimeError(
                f"PreparedLabelFieldSurfaceExtraction.route must select a supported grid route; got {prepared.route!r}"
            )
    local_coordinates = coordinates - offset
    dense = np.full(dense_shape, boundary, dtype=np.int64)
    mapped = source_to_region[labels]
    if np.any(mapped < 0):
        raise RuntimeError(
            "PreparedLabelFieldSurfaceExtraction.region_map must include every present source label; got an unmapped label"
        )
    dense[
        local_coordinates[:, 0],
        local_coordinates[:, 1],
        local_coordinates[:, 2],
    ] = mapped
    extraction_origin: tuple[float, float, float] = (
        plan.origin[0] + float(offset[0]) * plan.spacing[0],
        plan.origin[1] + float(offset[1]) * plan.spacing[1],
        plan.origin[2] + float(offset[2]) * plan.spacing[2],
    )
    measure = math.prod(plan.spacing)
    finite_volumes = tuple(
        float(counts[source] * measure)
        for region, source in enumerate(source_indices)
        if region != boundary and source >= 0
    )
    source_state_id = canonical_fingerprint(
        {
            "kind": "threshold-label-field-extraction-source",
            "prepared": prepared.prepared_id,
            "source_binding": state.binding_id,
            "source_route": state.route_id,
            "source_prepared": state.prepared_id,
            "source_sites": state.site_id,
            "dense_labels": array_tree_fingerprint(dense),
            "active_labels": array_tree_fingerprint(active),
            "coordinate_offset": array_tree_fingerprint(offset),
            "epoch": epoch,
            "time": time,
        }
    )
    return _RegionMap(
        dense_labels=dense,
        extraction_origin=extraction_origin,
        region_ids=region_ids,
        region_kinds=kinds,
        source_indices=source_indices,
        boundary_index=boundary,
        source_counts=tuple(int(value) for value in counts),
        finite_source_volumes=finite_volumes,
        source_state_id=source_state_id,
        source_binding_id=state.binding_id,
        source_route_id=state.route_id,
        source_prepared_id=state.prepared_id,
        source_site_id=state.site_id,
        source_epoch=epoch,
        source_time=time,
    )


def _tetrahedron_vertices(
    base: tuple[int, int, int],
    permutation: tuple[int, int, int],
    shape: tuple[int, int, int],
    /,
) -> tuple[int, int, int, int]:
    coordinate = list(base)
    vertices = [int(np.ravel_multi_index(tuple(coordinate), shape))]
    for axis in permutation:
        coordinate[axis] += 1
        vertices.append(int(np.ravel_multi_index(tuple(coordinate), shape)))
    return (vertices[0], vertices[1], vertices[2], vertices[3])


def _label_components(corner_labels: np.ndarray, label: int, /) -> int:
    members = {index for index, value in enumerate(corner_labels) if int(value) == label}
    components = 0
    while members:
        components += 1
        stack = [members.pop()]
        while stack:
            vertex = stack.pop()
            neighbours = [
                second if first == vertex else first
                for first, second in _CUBE_EDGES
                if first == vertex or second == vertex
            ]
            for neighbour in neighbours:
                if neighbour in members:
                    members.remove(neighbour)
                    stack.append(neighbour)
    return components


def _ambiguities(
    labels: np.ndarray, /
) -> tuple[int, int, int, int, int]:
    shape = labels.shape
    flat = labels.reshape(-1)
    ambiguous = 0
    multi_label_tetrahedra = 0
    mixed_tetrahedra = 0
    dual_face_count = 0
    dual_face_centers = 0
    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            for k in range(shape[2] - 1):
                cube = np.asarray(
                    [labels[i + di, j + dj, k + dk] for di, dj, dk in _CUBE_CORNERS],
                    dtype=np.int64,
                )
                unique = np.unique(cube)
                disconnected = any(
                    _label_components(cube, int(label)) > 1 for label in unique
                )
                if unique.size >= 3 or disconnected:
                    ambiguous += 1
                base = (i, j, k)
                for permutation in _TETRAHEDRON_PERMUTATIONS:
                    tetrahedron = _tetrahedron_vertices(base, permutation, shape)
                    tetrahedron_labels = flat[list(tetrahedron)]
                    tetrahedron_unique = np.unique(tetrahedron_labels)
                    if tetrahedron_unique.size > 1:
                        mixed_tetrahedra += 1
                    if tetrahedron_unique.size >= 3:
                        multi_label_tetrahedra += 1
                    for local_face in _TETRAHEDRON_FACES:
                        differing_edges = 0
                        for first, second in _FACE_EDGES:
                            if (
                                tetrahedron_labels[local_face[first]]
                                != tetrahedron_labels[local_face[second]]
                            ):
                                differing_edges += 1
                        dual_face_count += differing_edges
                        dual_face_centers += int(differing_edges > 0)
    return (
        ambiguous,
        multi_label_tetrahedra,
        dual_face_count,
        dual_face_centers,
        mixed_tetrahedra,
    )


def _interface_centroid(
    vertices: tuple[int, ...],
    lattice: np.ndarray,
    labels: np.ndarray,
    /,
) -> np.ndarray:
    transitions = [
        0.5 * (lattice[first] + lattice[second])
        for first, second in itertools.combinations(vertices, 2)
        if labels[first] != labels[second]
    ]
    if not transitions:
        raise RuntimeError(
            "LabelFieldSurfaceExtraction.interface_centroid must have at least one label transition; got no transitions"
        )
    return np.mean(np.asarray(transitions), axis=0)


def _tetrahedron_dual_position(
    tetrahedron: tuple[int, int, int, int],
    lattice: np.ndarray,
    labels: np.ndarray,
    /,
) -> np.ndarray:
    unique = np.unique(labels[list(tetrahedron)]).size
    if unique == 2:
        return _interface_centroid(tetrahedron, lattice, labels)
    if unique == 3:
        triple_faces = [
            tuple(tetrahedron[index] for index in local_face)
            for local_face in _TETRAHEDRON_FACES
            if np.unique(labels[[tetrahedron[index] for index in local_face]]).size == 3
        ]
        if len(triple_faces) != 2:
            raise RuntimeError(
                f"LabelFieldSurfaceExtraction.triple_faces must contain two triple-label faces for a three-label tetrahedron; got {len(triple_faces)}"
            )
        return np.mean(
            np.asarray(
                [_interface_centroid(face, lattice, labels) for face in triple_faces]
            ),
            axis=0,
        )
    if unique == 4:
        return np.mean(lattice[list(tetrahedron)], axis=0)
    raise RuntimeError(
        f"LabelFieldSurfaceExtraction.dual_point must have two to four tetrahedron labels; got {unique}"
    )


@final
class _DualSurfaceWork(StrictModule):
    """Host lattice and append-only interface ancestry in traversal order."""

    lattice: np.ndarray
    labels: np.ndarray
    positions: list[np.ndarray]
    faces: list[tuple[int, int, int]]
    face_labels: list[tuple[int, int]]
    edge_vertices: dict[tuple[int, ...], int]
    face_vertices: dict[tuple[int, ...], int]


def _prepare_dual_lattice(
    labels: np.ndarray,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    /,
) -> _DualSurfaceWork:
    """Prepare Cartesian sample coordinates with the free-space collar offset."""
    lattice = np.stack(
        np.meshgrid(
            *(np.arange(value, dtype=np.float64) for value in labels.shape),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 3))
    lattice = np.asarray(origin)[None, :] + (lattice - 1.0) * np.asarray(spacing)[None, :]
    return _DualSurfaceWork(lattice, labels.reshape(-1), [], [], [], {}, {})


def _dual_vertex(
    key: tuple[int, ...],
    table: dict[tuple[int, ...], int],
    position: np.ndarray,
    positions: list[np.ndarray],
    /,
) -> int:
    """Share one interface vertex per primal edge or face identity."""
    known = table.get(key)
    if known is not None:
        return known
    index = len(positions)
    positions.append(position)
    table[key] = index
    return index


def _emit_dual_face(
    work: _DualSurfaceWork,
    tetrahedron: tuple[int, int, int, int],
    local_face: tuple[int, int, int],
    center: int,
    /,
) -> None:
    """Emit oriented edge-face-tetrahedron flags for one material interface."""
    lattice, flat_labels = work.lattice, work.labels
    primal_face = tuple(sorted(tetrahedron[index] for index in local_face))
    differing: list[tuple[int, int]] = []
    for local_first, local_second in _FACE_EDGES:
        first = tetrahedron[local_face[local_first]]
        second = tetrahedron[local_face[local_second]]
        if flat_labels[first] != flat_labels[second]:
            differing.append((first, second))
    if not differing:
        return
    face_center = _dual_vertex(
        primal_face,
        work.face_vertices,
        _interface_centroid(primal_face, lattice, flat_labels),
        work.positions,
    )
    for first, second in differing:
        edge_key = sorted((first, second))
        primal_edge = (edge_key[0], edge_key[1])
        edge_center = _dual_vertex(
            primal_edge,
            work.edge_vertices,
            0.5 * (lattice[first] + lattice[second]),
            work.positions,
        )
        first_label = int(flat_labels[first])
        second_label = int(flat_labels[second])
        left, right = sorted((first_label, second_label))
        left_vertex = first if first_label == left else second
        right_vertex = second if first_label == left else first
        triangle = [edge_center, face_center, center]
        p0, p1, p2 = (work.positions[index] for index in triangle)
        direction = lattice[right_vertex] - lattice[left_vertex]
        orientation = float(np.dot(np.cross(p1 - p0, p2 - p0), direction))
        if orientation < 0.0:
            triangle[1], triangle[2] = triangle[2], triangle[1]
        work.faces.append((triangle[0], triangle[1], triangle[2]))
        work.face_labels.append((left, right))


def _dual_surface(
    labels: np.ndarray,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    /,
) -> _DualSurface:
    """Extract shared material interfaces in lexicographic Freudenthal order."""
    work = _prepare_dual_lattice(labels, origin, spacing)
    shape = labels.shape
    # Cartesian product has the same i, j, k order as the nested lattice loops.
    for i, j, k in itertools.product(*(range(value - 1) for value in shape)):
        for permutation in _TETRAHEDRON_PERMUTATIONS:
            tetrahedron = _tetrahedron_vertices((i, j, k), permutation, shape)
            tetrahedron_labels = work.labels[list(tetrahedron)]
            if np.all(tetrahedron_labels == tetrahedron_labels[0]):
                continue
            center = len(work.positions)
            work.positions.append(
                _tetrahedron_dual_position(tetrahedron, work.lattice, work.labels)
            )
            for local_face in _TETRAHEDRON_FACES:
                _emit_dual_face(work, tetrahedron, local_face, center)
    return _DualSurface(
        positions=np.asarray(work.positions, dtype=np.float64),
        faces=np.asarray(work.faces, dtype=np.int64),
        face_labels=np.asarray(work.face_labels, dtype=np.int64),
    )


def _source_pair_areas(
    labels: np.ndarray,
    boundary: int,
    spacing: tuple[float, float, float],
    /,
) -> dict[tuple[int, int], float]:
    padded = np.pad(labels, 1, constant_values=boundary)
    result: dict[tuple[int, int], float] = {}
    for axis in range(3):
        lower = np.take(padded, np.arange(padded.shape[axis] - 1), axis=axis)
        upper = np.take(padded, np.arange(1, padded.shape[axis]), axis=axis)
        first = lower[lower != upper]
        second = upper[lower != upper]
        area = math.prod(spacing[other] for other in range(3) if other != axis)
        for a, b in zip(first, second, strict=True):
            pair_values = sorted((int(a), int(b)))
            pair = (pair_values[0], pair_values[1])
            result[pair] = result.get(pair, 0.0) + area
    return result


def _extracted_pair_areas(surface: _DualSurface, /) -> dict[tuple[int, int], float]:
    triangles = surface.positions[surface.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    result: dict[tuple[int, int], float] = {}
    for labels, area in zip(surface.face_labels, areas, strict=True):
        pair = (int(labels[0]), int(labels[1]))
        result[pair] = result.get(pair, 0.0) + float(area)
    return result


def _relative_errors(
    first: tuple[float, ...], second: tuple[float, ...], /
) -> tuple[float, ...]:
    if len(first) != len(second):
        return ()
    return tuple(
        abs(a - b) / max(abs(a), abs(b), np.finfo(np.float64).tiny)
        for a, b in zip(first, second, strict=True)
    )


def _empty_counts(region_count: int, /) -> MultiRegionSurfaceCounts:
    return MultiRegionSurfaceCounts(
        vertex=0,
        edge=0,
        face=0,
        region=region_count,
        region_pair=0,
        edge_valence=0,
        vertex_region_pairs=0,
    )


def _lineage(
    prepared: PreparedLabelFieldSurfaceExtraction,
    regions: _RegionMap,
    seed: MultiRegionSurfaceSeed | None,
    /,
) -> LabelFieldSurfaceLineage:
    return LabelFieldSurfaceLineage(
        source_label_ids=prepared.plan.label_ids,
        source_label_counts=regions.source_counts,
        region_ids=regions.region_ids,
        region_source_label_indices=regions.source_indices,
        boundary_region_id=prepared.plan.boundary_region_id,
        source_epoch=regions.source_epoch,
        source_time=regions.source_time,
        source_state_id=regions.source_state_id,
        source_binding_id=regions.source_binding_id,
        source_route_id=regions.source_route_id,
        source_prepared_id=regions.source_prepared_id,
        source_site_id=regions.source_site_id,
        prepared_id=prepared.prepared_id,
        candidate_seed_id=None if seed is None else seed.seed_id,
    )


def _evidence(
    prepared: PreparedLabelFieldSurfaceExtraction,
    lineage: LabelFieldSurfaceLineage,
    *,
    status: LabelFieldSurfaceExtractionStatus,
    ambiguous_cells: int,
    multi_label_tetrahedra: int,
    counts: MultiRegionSurfaceCounts,
    capacity: MultiRegionSurfaceCapacityEvidence,
    validation: MultiRegionSurfaceEvidence | None = None,
    source_volumes: tuple[float, ...] = (),
    extracted_volumes: tuple[float, ...] = (),
    region_pairs: tuple[tuple[str, str], ...] = (),
    source_areas: tuple[float, ...] = (),
    extracted_areas: tuple[float, ...] = (),
) -> LabelFieldSurfaceExtractionEvidence:
    unsupported = 0 if validation is None else validation.nonphysical_edge_count
    return LabelFieldSurfaceExtractionEvidence(
        status=status,
        route=prepared.route,
        ambiguous_cell_count=ambiguous_cells,
        multi_label_tetrahedron_count=multi_label_tetrahedra,
        ambiguity_capacity=prepared.plan.maximum_ambiguous_cells,
        maximum_edge_valence=counts.edge_valence,
        maximum_vertex_region_pairs=counts.vertex_region_pairs,
        unsupported_edge_count=unsupported,
        source_finite_region_volumes=source_volumes,
        extracted_finite_region_volumes=extracted_volumes,
        finite_region_volume_relative_errors=_relative_errors(
            source_volumes, extracted_volumes
        ),
        region_pairs=region_pairs,
        source_pair_areas=source_areas,
        extracted_pair_areas=extracted_areas,
        pair_area_relative_errors=_relative_errors(source_areas, extracted_areas),
        capacity=capacity,
        validation=validation,
        lineage_id=lineage.lineage_id,
    )


def _label_field_region_bounds(
    prepared: PreparedLabelFieldSurfaceExtraction,
    dense_shape: tuple[int, int, int],
    state: LabelFieldState,
    /,
) -> tuple[int, int, int, int, int, int]:
    site_count = prepared.site_count
    dense_count = math.prod(dense_shape)
    padded_count = math.prod(tuple(value + 2 for value in dense_shape))
    cube_count = math.prod(tuple(value + 1 for value in dense_shape))
    region_count = len(prepared.plan.label_ids) + 1
    region_json_bytes = _label_json_text_upper(
        itertools.chain(
            (prepared.prepared_id,),
            (state.binding_id, state.route_id, state.prepared_id, state.site_id),
        )
    )
    region_work = (
        512
        + 16 * site_count
        + 24 * dense_count
        + 12 * padded_count
        + 8 * region_count
        + region_json_bytes
    )
    ambiguity_work = 128 + 256 * cube_count
    region_scratch = 8192 + 64 * site_count + 2 * region_json_bytes
    # Eight bounded cell labels, unique labels, and component sets/stack objects.
    ambiguity_scratch = 4096
    # Four region tuples hold references for ids/kinds and 272 bytes per integer or
    # float entry for source indices and volumes; counts scale with all source labels.
    region_bytes = (
        2048
        + 8 * dense_count
        + 576 * region_count
        + 272 * len(prepared.plan.label_ids)
        + 512
    )
    padded_bytes = 256 + 8 * padded_count
    return (
        region_work + ambiguity_work,
        region_scratch + ambiguity_scratch,
        region_scratch,
        ambiguity_scratch,
        region_bytes,
        padded_bytes,
    )



def _label_extraction_metadata_bytes(
    prepared: PreparedLabelFieldSurfaceExtraction,
    state: LabelFieldState,
    /,
    *,
    early_result: bool,
) -> tuple[int, int, int]:
    plan = prepared.plan
    strings = itertools.chain(
        plan.label_ids,
        (
            plan.boundary_region_id,
            plan.source,
            plan.plan_id,
            plan.capacity_plan.resource_id,
            plan.capacity_plan.plan_id,
            plan.validation_policy.profile,
            plan.validation_policy.policy_id,
            state.binding_id,
            state.route_id,
            state.prepared_id,
            state.site_id,
        ),
    )
    generated_hashes = 3 if early_result else 9
    seed_source_length = (
        0
        if early_result
        else len(plan.source) + len(".multilabel-marching-tetrahedra")
    )
    identity_bytes = 0
    json_text_bytes = 0
    for value in strings:
        identity_bytes += 256 + 4 * len(value)
        json_text_bytes += 12 * len(value) + 2
    identity_bytes += 512 * generated_hashes
    json_text_bytes += (12 * 64 + 2) * generated_hashes
    if seed_source_length:
        identity_bytes += 256 + 4 * seed_source_length
        json_text_bytes += 12 * seed_source_length + 2
    return identity_bytes, json_text_bytes, generated_hashes


def _label_field_extraction_bounds(
    prepared: PreparedLabelFieldSurfaceExtraction,
    dense_shape: tuple[int, int, int],
    ambiguity: tuple[int, int, int, int, int],
    early_result: bool,
    state: LabelFieldState,
    /,
) -> tuple[int, int, int]:
    metadata_bytes, metadata_json_bytes, metadata_hashes = (
        _label_extraction_metadata_bytes(
            prepared, state, early_result=early_result
        )
    )
    if early_result:
        region_capacity = max(
            prepared.plan.capacity_plan.region_capacity,
            len(prepared.plan.label_ids) + 1,
        )
        json_work = metadata_hashes * metadata_json_bytes
        return (
            256 + 64 * region_capacity + json_work,
            4096 + 256 * region_capacity + 2 * metadata_json_bytes,
            8192 + 512 * region_capacity + metadata_bytes,
        )
    capacity = prepared.plan.capacity_plan
    vertices = capacity.vertex_capacity
    edges = capacity.edge_capacity
    faces = capacity.face_capacity
    regions = capacity.region_capacity
    pairs = capacity.region_pair_capacity
    valence = capacity.maximum_edge_valence
    slots = capacity.maximum_vertex_region_pairs
    padded_count = math.prod(tuple(value + 2 for value in dense_shape))
    cube_count = math.prod(tuple(value + 1 for value in dense_shape))
    dual_faces = ambiguity[2]
    dual_face_centers = ambiguity[3]
    mixed_tetrahedra = ambiguity[4]
    dual_vertices = dual_face_centers + mixed_tetrahedra
    candidate_pairs = prepared.plan.validation_policy.intersection_candidate_capacity
    validation_face_count = min(dual_faces, faces)
    candidate_tests = min(
        candidate_pairs,
        validation_face_count * max(0, validation_face_count - 1) // 2,
    )
    candidate_block = min(1 << 20, validation_face_count * validation_face_count)
    evidence_region_count = max(
        regions, len(prepared.plan.label_ids) + 1
    )
    maximum_region_pairs = (
        (len(prepared.plan.label_ids) + 1) * len(prepared.plan.label_ids) // 2
    )
    source_pair_upper = min(maximum_region_pairs, 3 * padded_count)
    evidence_pair_count = min(maximum_region_pairs, source_pair_upper + dual_faces)
    evidence_pair_storage = max(pairs, evidence_pair_count)
    maximum_region_name_length = max(
        (len(value) for value in itertools.chain(
            prepared.plan.label_ids, (prepared.plan.boundary_region_id,)
        )),
        default=0,
    )
    pair_json_bytes = evidence_pair_count * 2 * (
        12 * maximum_region_name_length + 2
    )
    json_payload_bytes = (
        metadata_json_bytes
        + pair_json_bytes
        + 128 * evidence_region_count
        + 128 * evidence_pair_count
        + 2048
    )
    dual_sort = (4 * max(1, dual_faces) + 1).bit_length()
    capacity_sort = (4 * max(1, faces) + 1).bit_length()
    work_units = (
        512
        + 16 * len(prepared.plan.label_ids)
        + 72 * cube_count
        + dual_faces * (32 + 24 * dual_sort)
        + faces * (32 + 24 * capacity_sort)
        + edges * (8 + 4 * valence + 24 * valence * valence)
        + vertices * (8 + 4 * slots)
        + 512 * candidate_tests
        + 16 * evidence_pair_count
        + metadata_hashes * json_payload_bytes
        + 8 * padded_count
    )
    scratch_bytes = (
        65536
        + 96 * padded_count
        + 2048 * dual_faces
        + 128 * dual_vertices
        + 256 * faces
        + 64 * edges * valence
        + 64 * edges * valence * valence
        + 32 * vertices * slots
        + 512 * candidate_tests
        + 32 * pairs
        + 512 * evidence_pair_count
        + 64 * candidate_block
        + 2 * json_payload_bytes
    )
    # The seed stores 32 bytes per dual vertex and 48 per emitted face.
    # Capacity terms cover padded topology/state/routes/BVH; evidence tuples use
    # 3072 bytes per region and 2048 per pair for their retained scalar records.
    result_bytes = (
        32 * dual_vertices
        + 48 * dual_faces
        + vertices * (57 + 9 * slots)
        + edges * (17 + 24 * valence)
        + 512 * faces
        + 3072 * evidence_region_count
        + 2048 * evidence_pair_storage
        + 24576
        + metadata_bytes
    )
    return work_units, scratch_bytes, result_bytes


def _extract(
    prepared: PreparedLabelFieldSurfaceExtraction,
    state: LabelFieldState,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> LabelFieldSurfaceExtractionResult:
    if budget is None:
        return _extract_core(prepared, state)
    budget.ticket.checkpoint()
    match prepared.route:
        case "uniform-grid":
            dense_shape = prepared.grid_shape
        case "sparse-grid":
            sparse_scratch = 48 * prepared.site_count + 1024
            with budget.temporary_scope():
                budget.reserve(6 * prepared.site_count, sparse_scratch)
                coordinates = np.asarray(prepared.site_coordinates, dtype=np.int64)
                lower = np.min(coordinates, axis=0)
                upper = np.max(coordinates, axis=0)
                dense_shape = (
                    int(upper[0] - lower[0] + 1),
                    int(upper[1] - lower[1] + 1),
                    int(upper[2] - lower[2] + 1),
                )
                del coordinates, lower, upper
                budget.release_temporary_storage(sparse_scratch)
        case _ as unreachable:
            raise AssertionError(f"Unsupported label-field extraction route: {unreachable!r}")
    (
        region_work,
        stage_one_scratch,
        region_scratch,
        ambiguity_scratch,
        region_bytes,
        padded_bytes,
    ) = _label_field_region_bounds(prepared, dense_shape, state)
    with budget.temporary_scope() as lifetime:
        _admit_label_field_phase(
            budget,
            region_work,
            stage_one_scratch + region_bytes + padded_bytes,
            0,
            0,
        )
        regions = None
        padded = None
        ambiguity = None
        try:
            regions = _region_map(prepared, state)
            padded = np.pad(
                regions.dense_labels, 1, constant_values=regions.boundary_index
            )
            ambiguity = _ambiguities(padded)
            early_result = (
                sum(kind == "finite" for kind in regions.region_kinds) == 0
                or np.unique(padded).size < 2
                or ambiguity[0] > prepared.plan.maximum_ambiguous_cells
            )
            budget.release_temporary_storage(region_scratch + ambiguity_scratch)
        except BaseException:
            del regions, padded, ambiguity
            raise
        work_units, scratch_bytes, result_bytes = _label_field_extraction_bounds(
            prepared, dense_shape, ambiguity, early_result, state
        )
        try:
            _admit_label_field_phase(
                budget, work_units, scratch_bytes, result_bytes, 512
            )
            result = _extract_core(
                prepared,
                state,
                regions=regions,
                padded=padded,
                ambiguity=ambiguity,
            )
            _finish_label_field_phase(
                budget, lifetime, result, scratch_bytes, result_bytes
            )
        except BaseException:
            del regions, padded, ambiguity
            raise
        else:
            del regions, padded, ambiguity
            budget.release_temporary_storage(region_bytes + padded_bytes)
    return result


def _extract_core(
    prepared: PreparedLabelFieldSurfaceExtraction,
    state: LabelFieldState,
    /,
    *,
    regions: _RegionMap | None = None,
    padded: np.ndarray | None = None,
    ambiguity: tuple[int, int, int, int, int] | None = None,
) -> LabelFieldSurfaceExtractionResult:
    if regions is None:
        regions = _region_map(prepared, state)
    plan = prepared.plan
    if padded is None:
        padded = np.pad(regions.dense_labels, 1, constant_values=regions.boundary_index)
    if ambiguity is None:
        ambiguity = _ambiguities(padded)
    ambiguous, multi_label_tetrahedra = ambiguity[:2]
    empty = _empty_counts(len(regions.region_ids))
    empty_capacity = plan.capacity_plan.capacity_evidence(empty)
    finite_count = sum(kind == "finite" for kind in regions.region_kinds)
    if finite_count == 0 or np.unique(padded).size < 2:
        lineage = _lineage(prepared, regions, None)
        evidence = _evidence(
            prepared,
            lineage,
            status=LabelFieldSurfaceExtractionStatus.NO_INTERFACE,
            ambiguous_cells=ambiguous,
            multi_label_tetrahedra=multi_label_tetrahedra,
            counts=empty,
            capacity=empty_capacity,
        )
        return LabelFieldSurfaceExtractionResult(
            None, None, None, None, lineage, evidence
        )
    if ambiguous > plan.maximum_ambiguous_cells:
        lineage = _lineage(prepared, regions, None)
        evidence = _evidence(
            prepared,
            lineage,
            status=LabelFieldSurfaceExtractionStatus.AMBIGUITY_CAPACITY_EXCEEDED,
            ambiguous_cells=ambiguous,
            multi_label_tetrahedra=multi_label_tetrahedra,
            counts=empty,
            capacity=empty_capacity,
        )
        return LabelFieldSurfaceExtractionResult(
            None, None, None, None, lineage, evidence
        )

    dual = _dual_surface(padded, regions.extraction_origin, plan.spacing)
    seed = MultiRegionSurfaceSeed(
        dual.positions,
        dual.faces,
        dual.face_labels,
        regions.region_ids,
        regions.region_kinds,
        source=f"{plan.source}.multilabel-marching-tetrahedra",
    )
    counts = seed.counts()
    capacity = plan.capacity_plan.capacity_evidence(counts)
    lineage = _lineage(prepared, regions, seed)
    source_pair_areas = _source_pair_areas(
        regions.dense_labels, regions.boundary_index, plan.spacing
    )
    extracted_pair_areas = _extracted_pair_areas(dual)
    pair_indices = tuple(sorted(set(source_pair_areas) | set(extracted_pair_areas)))
    region_pairs = tuple(
        (regions.region_ids[first], regions.region_ids[second])
        for first, second in pair_indices
    )
    source_areas = tuple(source_pair_areas.get(pair, 0.0) for pair in pair_indices)
    extracted_areas = tuple(extracted_pair_areas.get(pair, 0.0) for pair in pair_indices)
    if not capacity.admitted:
        evidence = _evidence(
            prepared,
            lineage,
            status=LabelFieldSurfaceExtractionStatus.CAPACITY_EXCEEDED,
            ambiguous_cells=ambiguous,
            multi_label_tetrahedra=multi_label_tetrahedra,
            counts=counts,
            capacity=capacity,
            source_volumes=regions.finite_source_volumes,
            region_pairs=region_pairs,
            source_areas=source_areas,
            extracted_areas=extracted_areas,
        )
        return LabelFieldSurfaceExtractionResult(
            seed, None, None, None, lineage, evidence
        )

    topology = seed.topology(plan.capacity_plan, epoch=regions.source_epoch)
    surface_state = seed.state(topology)
    validation = validate_multiregion_surface(
        topology, surface_state, policy=plan.validation_policy
    )
    extracted_volumes = validation.signed_volumes
    if not validation.accepted:
        status = (
            LabelFieldSurfaceExtractionStatus.UNSUPPORTED_VALENCE
            if validation.status is MultiRegionSurfaceStatus.NONPHYSICAL_VALENCE
            else LabelFieldSurfaceExtractionStatus.VALIDATION_FAILED
        )
        evidence = _evidence(
            prepared,
            lineage,
            status=status,
            ambiguous_cells=ambiguous,
            multi_label_tetrahedra=multi_label_tetrahedra,
            counts=counts,
            capacity=capacity,
            validation=validation,
            source_volumes=regions.finite_source_volumes,
            extracted_volumes=extracted_volumes,
            region_pairs=region_pairs,
            source_areas=source_areas,
            extracted_areas=extracted_areas,
        )
        return LabelFieldSurfaceExtractionResult(
            seed, None, None, None, lineage, evidence
        )

    surface = PreparedMultiRegionSurface(
        topology, surface_state, policy=plan.validation_policy
    )
    evidence = _evidence(
        prepared,
        lineage,
        status=LabelFieldSurfaceExtractionStatus.ACCEPTED,
        ambiguous_cells=ambiguous,
        multi_label_tetrahedra=multi_label_tetrahedra,
        counts=counts,
        capacity=capacity,
        validation=validation,
        source_volumes=regions.finite_source_volumes,
        extracted_volumes=extracted_volumes,
        region_pairs=region_pairs,
        source_areas=source_areas,
        extracted_areas=extracted_areas,
    )
    return LabelFieldSurfaceExtractionResult(
        seed, topology, surface_state, surface, lineage, evidence
    )


__all__ = [
    "LabelFieldSurfaceExtractionEvidence",
    "LabelFieldSurfaceExtractionPlan",
    "LabelFieldSurfaceExtractionResult",
    "LabelFieldSurfaceExtractionRoute",
    "LabelFieldSurfaceExtractionStatus",
    "LabelFieldSurfaceLineage",
    "LabelFieldVolumeBinding",
    "PreparedLabelFieldSurfaceExtraction",
]
