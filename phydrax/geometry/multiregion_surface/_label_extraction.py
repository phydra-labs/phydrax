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
and four labels in a tetrahedron produce the complete six-sheet region graph.

Extraction is a host seeding/repair transaction. The source label state is read
once and fingerprinted, but never retained or synchronized. A successful result
owns a new :class:`MultiRegionSurfaceTopology` and state; exact multiregion
validation, including collision certification, must accept before a prepared
surface is exposed.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    canonical_identifier,
    nonnegative_integer,
    positive_finite_float,
    unique_identifiers,
)
from ...typing import Dim, HostInt64, Identifier, parse, Scope
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
from ._validation import validate_multiregion_surface


if TYPE_CHECKING:
    from ...threshold_dynamics import LabelFieldState


LabelFieldSurfaceExtractionRoute: TypeAlias = Literal["uniform-grid", "sparse-grid"]


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
            raise ValueError("source_label_counts must follow source_label_ids.")
        if len(region_source_label_indices) != len(region_ids):
            raise ValueError("region_source_label_indices must follow region_ids.")
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
        if not isinstance(status, LabelFieldSurfaceExtractionStatus):
            raise TypeError("status must be a LabelFieldSurfaceExtractionStatus.")
        route_ = parse(route, LabelFieldSurfaceExtractionRoute, "route")
        if not isinstance(capacity, MultiRegionSurfaceCapacityEvidence):
            raise TypeError("capacity must be MultiRegionSurfaceCapacityEvidence.")
        if validation is not None and not isinstance(
            validation, MultiRegionSurfaceEvidence
        ):
            raise TypeError("validation must be MultiRegionSurfaceEvidence or None.")
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
    """Accepted explicit authority or a fail-closed extraction candidate.

    A rejected result may expose ``candidate_seed`` for diagnostics or a later
    repair attempt, but never exposes topology/state/preparation as an accepted
    E authority. Successful results expose all four and are collision-certified.
    """

    candidate_seed: MultiRegionSurfaceSeed | None
    topology: MultiRegionSurfaceTopology | None
    state: MultiRegionSurfaceState | None
    surface: PreparedMultiRegionSurface | None
    lineage: LabelFieldSurfaceLineage
    evidence: LabelFieldSurfaceExtractionEvidence

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
        if candidate_seed is not None and not isinstance(
            candidate_seed, MultiRegionSurfaceSeed
        ):
            raise TypeError("candidate_seed must be MultiRegionSurfaceSeed or None.")
        if not isinstance(lineage, LabelFieldSurfaceLineage):
            raise TypeError("lineage must be LabelFieldSurfaceLineage.")
        if not isinstance(evidence, LabelFieldSurfaceExtractionEvidence):
            raise TypeError("evidence must be LabelFieldSurfaceExtractionEvidence.")
        authority = (topology, state, surface)
        if evidence.accepted:
            if not isinstance(topology, MultiRegionSurfaceTopology):
                raise TypeError("An accepted extraction needs a topology.")
            if not isinstance(state, MultiRegionSurfaceState):
                raise TypeError("An accepted extraction needs a state.")
            if not isinstance(surface, PreparedMultiRegionSurface):
                raise TypeError("An accepted extraction needs a prepared surface.")
            if candidate_seed is None or not evidence.collision_certified:
                raise ValueError(
                    "An accepted extraction needs a collision-certified seed."
                )
        elif any(value is not None for value in authority):
            raise ValueError("A refused extraction cannot expose a surface authority.")
        self.candidate_seed = candidate_seed
        self.topology = topology
        self.state = state
        self.surface = surface
        self.lineage = lineage
        self.evidence = evidence

    @property
    def accepted(self) -> bool:
        return self.evidence.accepted


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
        if not isinstance(capacity_plan, MultiRegionSurfaceCapacityPlan):
            raise TypeError("capacity_plan must be a MultiRegionSurfaceCapacityPlan.")
        if len(spacing) != 3:
            raise ValueError("spacing must contain three Cartesian values.")
        spacing_: tuple[float, float, float] = (
            positive_finite_float(spacing[0], "spacing[0]"),
            positive_finite_float(spacing[1], "spacing[1]"),
            positive_finite_float(spacing[2], "spacing[2]"),
        )
        if len(origin) != 3:
            raise ValueError("origin must contain three Cartesian values.")
        origin_: tuple[float, float, float] = (
            float(origin[0]),
            float(origin[1]),
            float(origin[2]),
        )
        if not all(math.isfinite(value) for value in origin_):
            raise ValueError("origin values must be finite.")
        boundary = canonical_identifier(boundary_region_id, "boundary_region_id")
        policy = (
            MultiRegionSurfaceValidationPolicy()
            if validation_policy is None
            else validation_policy
        )
        if not isinstance(policy, MultiRegionSurfaceValidationPolicy):
            raise TypeError(
                "validation_policy must be MultiRegionSurfaceValidationPolicy."
            )
        if not policy.check_self_intersection:
            raise ValueError(
                "Label-field extraction requires collision-certified validation; "
                "check_self_intersection cannot be disabled."
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
    ) -> PreparedLabelFieldSurfaceExtraction:
        """Prepare uniform sites or an explicit sparse subset of a 3D grid."""
        return PreparedLabelFieldSurfaceExtraction(
            self, grid_shape, site_coordinates=site_coordinates
        )


@final
class PreparedLabelFieldSurfaceExtraction(StrictModule, NonTrainableState):
    """Canonical source-site layout reusable across threshold epochs."""

    __strict_contract__ = True

    plan: LabelFieldSurfaceExtractionPlan
    site_coordinates: HostInt64[_ExtractionSiteDim, Literal[3]]
    source_order: HostInt64[_ExtractionSiteDim]
    grid_shape: tuple[int, int, int] = eqx.field(static=True)
    source_shape: tuple[int, ...] = eqx.field(static=True)
    route: LabelFieldSurfaceExtractionRoute = eqx.field(static=True)
    site_count: int = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        plan: LabelFieldSurfaceExtractionPlan,
        grid_shape: tuple[int, int, int],
        /,
        *,
        site_coordinates: ArrayLike | None = None,
    ) -> None:
        if not isinstance(plan, LabelFieldSurfaceExtractionPlan):
            raise TypeError("plan must be a LabelFieldSurfaceExtractionPlan.")
        if len(grid_shape) != 3:
            raise ValueError("grid_shape must contain three dimensions.")
        shape: tuple[int, int, int] = (
            nonnegative_integer(grid_shape[0], "grid_shape[0]"),
            nonnegative_integer(grid_shape[1], "grid_shape[1]"),
            nonnegative_integer(grid_shape[2], "grid_shape[2]"),
        )
        if any(value == 0 for value in shape):
            raise ValueError("grid_shape dimensions must be positive.")
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
                    "site_coordinates must be an integer array with shape (sites, 3)."
                )
            coordinates_ = raw.astype(np.int64)
            if coordinates_.shape[0] == 0:
                raise ValueError("site_coordinates cannot be empty.")
            bounds = np.asarray(shape, dtype=np.int64)
            if np.any(coordinates_ < 0) or np.any(coordinates_ >= bounds):
                raise ValueError("site_coordinates must lie inside grid_shape.")
            order = np.lexsort(
                (coordinates_[:, 2], coordinates_[:, 1], coordinates_[:, 0])
            )
            coordinates = coordinates_[order]
            if np.any(np.all(coordinates[1:] == coordinates[:-1], axis=1)):
                raise ValueError("site_coordinates must be unique.")
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
        self.plan = plan
        self.site_coordinates = coordinates
        self.source_order = order
        self.grid_shape = shape
        self.source_shape = source_shape
        self.route = route
        self.site_count = coordinates.shape[0]
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-label-field-surface-extraction",
                "plan": plan.plan_id,
                "grid_shape": list(shape),
                "route": route,
                "site_coordinates": array_tree_fingerprint(coordinates),
            }
        )

    def extract(self, state: LabelFieldState, /) -> LabelFieldSurfaceExtractionResult:
        """Extract, certify and transfer authority from one label-field epoch."""
        return _extract(self, state)


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
    prepared: PreparedLabelFieldSurfaceExtraction, state: LabelFieldState, /
) -> tuple[np.ndarray, np.ndarray, int, float]:
    from ...threshold_dynamics._contracts import LabelFieldState

    if not isinstance(state, LabelFieldState):
        raise TypeError("state must be a LabelFieldState.")
    if state.label_ids != prepared.plan.label_ids:
        raise ValueError(
            "state label_ids must exactly match the extraction plan label order."
        )
    labels = np.asarray(state.labels, dtype=np.int64)
    if labels.shape != prepared.source_shape:
        raise ValueError(
            "state labels must have prepared source shape "
            f"{prepared.source_shape}, got {labels.shape}."
        )
    flat = labels.reshape(-1)[prepared.source_order]
    if np.any(flat < 0) or np.any(flat >= state.label_count):
        raise ValueError("state labels must index the extraction plan label_ids.")
    active = np.asarray(state.active_labels, dtype=np.bool_)
    counts = np.bincount(flat, minlength=state.label_count)
    if np.any((counts > 0) & ~active):
        raise ValueError("A source label that owns sites cannot be inactive.")
    return flat, active, int(np.asarray(state.epoch)), float(np.asarray(state.time))


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
            raise RuntimeError(f"Unknown prepared extraction route {prepared.route!r}.")
    local_coordinates = coordinates - offset
    dense = np.full(dense_shape, boundary, dtype=np.int64)
    mapped = source_to_region[labels]
    if np.any(mapped < 0):
        raise RuntimeError("A present source label was omitted from the region map.")
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


def _ambiguities(labels: np.ndarray, /) -> tuple[int, int]:
    shape = labels.shape
    flat = labels.reshape(-1)
    ambiguous = 0
    multi_label_tetrahedra = 0
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
                    if np.unique(flat[list(tetrahedron)]).size >= 3:
                        multi_label_tetrahedra += 1
    return ambiguous, multi_label_tetrahedra


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
        raise RuntimeError("An interface centroid needs at least one label transition.")
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
                "A three-label tetrahedron must have two triple-label faces."
            )
        return np.mean(
            np.asarray(
                [_interface_centroid(face, lattice, labels) for face in triple_faces]
            ),
            axis=0,
        )
    if unique == 4:
        return np.mean(lattice[list(tetrahedron)], axis=0)
    raise RuntimeError("A dual point needs between two and four tetrahedron labels.")


def _dual_surface(
    labels: np.ndarray,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    /,
) -> _DualSurface:
    shape = labels.shape
    lattice = np.stack(
        np.meshgrid(
            *(np.arange(value, dtype=np.float64) for value in shape),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 3))
    lattice = np.asarray(origin)[None, :] + (lattice - 1.0) * np.asarray(spacing)[None, :]
    flat_labels = labels.reshape(-1)
    positions: list[np.ndarray] = []
    faces: list[tuple[int, int, int]] = []
    face_labels: list[tuple[int, int]] = []
    edge_vertices: dict[tuple[int, ...], int] = {}
    face_vertices: dict[tuple[int, ...], int] = {}

    def dual_vertex(
        key: tuple[int, ...],
        table: dict[tuple[int, ...], int],
        position: np.ndarray,
        /,
    ) -> int:
        known = table.get(key)
        if known is not None:
            return known
        index = len(positions)
        positions.append(position)
        table[key] = index
        return index

    for i in range(shape[0] - 1):
        for j in range(shape[1] - 1):
            for k in range(shape[2] - 1):
                base = (i, j, k)
                for permutation in _TETRAHEDRON_PERMUTATIONS:
                    tetrahedron = _tetrahedron_vertices(base, permutation, shape)
                    tetrahedron_labels = flat_labels[list(tetrahedron)]
                    if np.all(tetrahedron_labels == tetrahedron_labels[0]):
                        continue
                    center = len(positions)
                    positions.append(
                        _tetrahedron_dual_position(tetrahedron, lattice, flat_labels)
                    )
                    for local_face in _TETRAHEDRON_FACES:
                        primal_face = tuple(
                            sorted(tetrahedron[index] for index in local_face)
                        )
                        differing: list[tuple[int, int]] = []
                        for local_first, local_second in _FACE_EDGES:
                            first = tetrahedron[local_face[local_first]]
                            second = tetrahedron[local_face[local_second]]
                            if flat_labels[first] != flat_labels[second]:
                                differing.append((first, second))
                        if not differing:
                            continue
                        face_center = dual_vertex(
                            primal_face,
                            face_vertices,
                            _interface_centroid(primal_face, lattice, flat_labels),
                        )
                        for first, second in differing:
                            edge_key = sorted((first, second))
                            primal_edge = (edge_key[0], edge_key[1])
                            edge_center = dual_vertex(
                                primal_edge,
                                edge_vertices,
                                0.5 * (lattice[first] + lattice[second]),
                            )
                            first_label = int(flat_labels[first])
                            second_label = int(flat_labels[second])
                            left, right = sorted((first_label, second_label))
                            left_vertex = first if first_label == left else second
                            right_vertex = second if first_label == left else first
                            triangle = [edge_center, face_center, center]
                            p0, p1, p2 = (positions[index] for index in triangle)
                            direction = lattice[right_vertex] - lattice[left_vertex]
                            orientation = float(
                                np.dot(np.cross(p1 - p0, p2 - p0), direction)
                            )
                            if orientation < 0.0:
                                triangle[1], triangle[2] = triangle[2], triangle[1]
                            faces.append((triangle[0], triangle[1], triangle[2]))
                            face_labels.append((left, right))
    return _DualSurface(
        positions=np.asarray(positions, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        face_labels=np.asarray(face_labels, dtype=np.int64),
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


def _extract(
    prepared: PreparedLabelFieldSurfaceExtraction, state: LabelFieldState, /
) -> LabelFieldSurfaceExtractionResult:
    regions = _region_map(prepared, state)
    plan = prepared.plan
    padded = np.pad(regions.dense_labels, 1, constant_values=regions.boundary_index)
    ambiguous, multi_label_tetrahedra = _ambiguities(padded)
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
    "PreparedLabelFieldSurfaceExtraction",
]
