#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Capacity, selector, status and evidence contracts of labeled multiregion surfaces.

A multiregion surface is a triangulated, possibly non-manifold surface complex
whose every face separates two distinct regions. Each face carries the ordered
label pair ``(left, right)``; the right-handed face normal ``(x1 - x0) x (x2 - x0)``
points out of ``left`` into ``right``. Regions are either ``"finite"`` (closed
3-cells whose oriented boundary is a watertight 2-cycle and whose signed volume
is defined) or ``"boundary"`` labels (the unbounded ambient, or open regions cut
by wire frames). Boundary labels are never 3-cells and share the reference
pressure of the surrounding medium.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import final, Literal, TypeAlias

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier, nonnegative_integer, positive_integer
from ...typing import checked, parse


MultiRegionKind: TypeAlias = Literal["finite", "boundary"]
MultiRegionCoordinateDtype: TypeAlias = Literal["float64", "float32"]
MultiRegionIndexDtype: TypeAlias = Literal["int32", "int64"]
MultiRegionDomain: TypeAlias = Literal["free_space", "periodic"]
MultiRegionValidationProfile: TypeAlias = Literal[
    "general", "dry_foam", "manifold_two_region"
]

# Plateau's laws for dry foams: an accepted interior edge carries exactly three
# films and an accepted vertex touches at most four regions (six region pairs).
DRY_FOAM_EDGE_VALENCE = 3
DRY_FOAM_VERTEX_REGIONS = 4


class MultiRegionSurfaceCapacityError(ValueError):
    """A multiregion surface does not fit the declared fixed capacities."""

    def __init__(self, evidence: MultiRegionSurfaceCapacityEvidence, /) -> None:
        self.evidence = evidence
        super().__init__(
            "Multiregion surface exceeds its capacity plan: "
            + ", ".join(evidence.exceeded)
            + "."
        )


class MultiRegionSurfacePreparationError(ValueError):
    """Host validation rejected a multiregion surface; ``evidence`` says why."""

    def __init__(self, evidence: MultiRegionSurfaceEvidence, /) -> None:
        self.evidence = evidence
        super().__init__(
            f"Multiregion surface validation failed with status {evidence.status.name}."
        )


class MultiRegionSurfaceStatus(IntEnum):
    """Outcome of host multiregion surface validation (first failing check)."""

    ACCEPTED = 0
    NONFINITE_GEOMETRY = 1
    LABEL_ORIENTATION_INCONSISTENT = 2
    REGION_NOT_WATERTIGHT = 3
    NONPHYSICAL_VALENCE = 4
    DEGENERATE_FACE = 5
    NONPOSITIVE_VOLUME = 6
    SELF_INTERSECTION = 7
    UNCERTAIN_PREDICATE = 8
    INTERSECTION_CANDIDATES_EXCEEDED = 9
    SINGULAR_VERTEX = 10
    REGION_GRAPH_INCOMPLETE = 11
    DISCONNECTED_FINITE_REGION = 12


_CAPACITY_NAMES = (
    "vertex",
    "edge",
    "face",
    "region",
    "region_pair",
    "edge_valence",
    "vertex_region_pairs",
)


@final
class MultiRegionSurfaceCounts(StrictModule, NonTrainableState):
    """Exact entity counts and valence maxima required by one surface."""

    vertex: int = eqx.field(static=True)
    edge: int = eqx.field(static=True)
    face: int = eqx.field(static=True)
    region: int = eqx.field(static=True)
    region_pair: int = eqx.field(static=True)
    edge_valence: int = eqx.field(static=True)
    vertex_region_pairs: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertex: int,
        edge: int,
        face: int,
        region: int,
        region_pair: int,
        edge_valence: int,
        vertex_region_pairs: int,
    ) -> None:
        values = {
            "vertex": nonnegative_integer(vertex, "vertex"),
            "edge": nonnegative_integer(edge, "edge"),
            "face": nonnegative_integer(face, "face"),
            "region": nonnegative_integer(region, "region"),
            "region_pair": nonnegative_integer(region_pair, "region_pair"),
            "edge_valence": nonnegative_integer(edge_valence, "edge_valence"),
            "vertex_region_pairs": nonnegative_integer(
                vertex_region_pairs, "vertex_region_pairs"
            ),
        }
        self.vertex = values["vertex"]
        self.edge = values["edge"]
        self.face = values["face"]
        self.region = values["region"]
        self.region_pair = values["region_pair"]
        self.edge_valence = values["edge_valence"]
        self.vertex_region_pairs = values["vertex_region_pairs"]

    def as_mapping(self) -> Mapping[str, int]:
        return {name: _count_of(self, name) for name in _CAPACITY_NAMES}


def _count_of(counts: MultiRegionSurfaceCounts, name: str, /) -> int:
    """Count of one canonical capacity name (closed selector dispatch)."""
    match name:
        case "vertex":
            return counts.vertex
        case "edge":
            return counts.edge
        case "face":
            return counts.face
        case "region":
            return counts.region
        case "region_pair":
            return counts.region_pair
        case "edge_valence":
            return counts.edge_valence
        case "vertex_region_pairs":
            return counts.vertex_region_pairs
        case _:
            raise ValueError(f"Unknown multiregion capacity {name!r}.")


@final
class MultiRegionSurfaceCapacityEvidence(StrictModule, NonTrainableState):
    """Required counts against declared capacities; ``admitted`` iff none exceeded."""

    required: MultiRegionSurfaceCounts
    capacity: MultiRegionSurfaceCounts
    exceeded: tuple[str, ...] = eqx.field(static=True)
    admitted: bool = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)


@final
class MultiRegionSurfaceCapacityPlan(StrictModule, NonTrainableState):
    """Fixed capacities, valence limits and dtypes of one multiregion surface.

    ``region_capacity`` counts every label slot, finite regions and boundary
    labels alike; boundary labels occupy region slots but are never 3-cells.
    ``maximum_edge_valence`` bounds the faces incident to one edge and
    ``maximum_vertex_region_pairs`` bounds the distinct region pairs (sheet
    slots) meeting at one vertex. A dry-foam topology needs three and six; a
    general complex may declare more. ``event_capacity`` reserves the topology
    transaction budget of later remeshing/event passes.
    """

    vertex_capacity: int = eqx.field(static=True)
    edge_capacity: int = eqx.field(static=True)
    face_capacity: int = eqx.field(static=True)
    region_capacity: int = eqx.field(static=True)
    region_pair_capacity: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    maximum_edge_valence: int = eqx.field(static=True)
    maximum_vertex_region_pairs: int = eqx.field(static=True)
    coordinate_dtype: MultiRegionCoordinateDtype = eqx.field(static=True)
    index_dtype: MultiRegionIndexDtype = eqx.field(static=True)
    resource_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertex_capacity: int,
        edge_capacity: int,
        face_capacity: int,
        region_capacity: int,
        region_pair_capacity: int,
        maximum_edge_valence: int,
        maximum_vertex_region_pairs: int,
        resource_id: str,
        event_capacity: int = 0,
        coordinate_dtype: MultiRegionCoordinateDtype = "float64",
        index_dtype: MultiRegionIndexDtype = "int32",
    ) -> None:
        vertices = positive_integer(vertex_capacity, "vertex_capacity")
        edges = positive_integer(edge_capacity, "edge_capacity")
        faces = positive_integer(face_capacity, "face_capacity")
        regions = positive_integer(region_capacity, "region_capacity")
        pairs = positive_integer(region_pair_capacity, "region_pair_capacity")
        events = nonnegative_integer(event_capacity, "event_capacity")
        valence = positive_integer(maximum_edge_valence, "maximum_edge_valence")
        slots = positive_integer(
            maximum_vertex_region_pairs, "maximum_vertex_region_pairs"
        )
        coordinate = parse(
            coordinate_dtype, MultiRegionCoordinateDtype, "coordinate_dtype"
        )
        index = parse(index_dtype, MultiRegionIndexDtype, "index_dtype")
        resource = canonical_identifier(resource_id, "resource_id")
        if regions < 2:
            raise ValueError("region_capacity must admit at least two labels.")
        if index == "int32" and max(vertices, edges, faces) * max(valence, 3) >= 2**31:
            raise ValueError(
                "Capacities exceed int32 addressing; use index_dtype='int64'."
            )
        self.vertex_capacity = vertices
        self.edge_capacity = edges
        self.face_capacity = faces
        self.region_capacity = regions
        self.region_pair_capacity = pairs
        self.event_capacity = events
        self.maximum_edge_valence = valence
        self.maximum_vertex_region_pairs = slots
        self.coordinate_dtype = coordinate
        self.index_dtype = index
        self.resource_id = resource
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiregion-surface-capacity-plan",
                "vertex_capacity": vertices,
                "edge_capacity": edges,
                "face_capacity": faces,
                "region_capacity": regions,
                "region_pair_capacity": pairs,
                "event_capacity": events,
                "maximum_edge_valence": valence,
                "maximum_vertex_region_pairs": slots,
                "coordinate_dtype": coordinate,
                "index_dtype": index,
                "resource_id": resource,
            }
        )

    @property
    def capacity(self) -> MultiRegionSurfaceCounts:
        return MultiRegionSurfaceCounts(
            vertex=self.vertex_capacity,
            edge=self.edge_capacity,
            face=self.face_capacity,
            region=self.region_capacity,
            region_pair=self.region_pair_capacity,
            edge_valence=self.maximum_edge_valence,
            vertex_region_pairs=self.maximum_vertex_region_pairs,
        )

    @checked
    def capacity_evidence(
        self, required: MultiRegionSurfaceCounts, /
    ) -> MultiRegionSurfaceCapacityEvidence:
        """Host admission of exact required counts against this plan."""
        capacity = self.capacity
        exceeded = tuple(
            name
            for name in _CAPACITY_NAMES
            if _count_of(required, name) > _count_of(capacity, name)
        )
        return MultiRegionSurfaceCapacityEvidence(
            required=required,
            capacity=capacity,
            exceeded=exceeded,
            admitted=not exceeded,
            resource_id=self.resource_id,
        )


@final
class MultiRegionSurfaceValidationPolicy(StrictModule, NonTrainableState):
    """Host validation profile and resource bounds.

    ``profile="dry_foam"`` additionally enforces Plateau's topological laws:
    every edge whose incident films close around it carries exactly three
    films, no vertex touches more than four regions, and every region pair
    meeting at an interior vertex shares a film there (a complete region
    graph; point contacts are the unresolved stage of a T1 process).
    ``profile="manifold_two_region"`` admits exactly one finite region and one
    boundary region separated by one closed two-manifold sheet: every edge has
    valence two, every vertex touches both regions, and no wire border exists.
    All profiles refuse singular vertices (two disconnected fans of one sheet
    at one vertex) and finite regions with more than one connected component.
    ``intersection_candidate_capacity`` bounds the broad-phase face pairs
    examined by exact predicates; exceeding it is a refusal, never a silent
    pass.
    """

    profile: MultiRegionValidationProfile = eqx.field(static=True)
    check_self_intersection: bool = eqx.field(static=True)
    intersection_candidate_capacity: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        profile: MultiRegionValidationProfile = "general",
        check_self_intersection: bool = True,
        intersection_candidate_capacity: int = 4_000_000,
    ) -> None:
        profile_ = parse(profile, MultiRegionValidationProfile, "profile")
        if not isinstance(check_self_intersection, bool):
            raise TypeError("check_self_intersection must be a bool.")
        capacity = positive_integer(
            intersection_candidate_capacity, "intersection_candidate_capacity"
        )
        self.profile = profile_
        self.check_self_intersection = check_self_intersection
        self.intersection_candidate_capacity = capacity
        self.policy_id = canonical_fingerprint(
            {
                "kind": "multiregion-surface-validation-policy",
                "profile": profile_,
                "check_self_intersection": check_self_intersection,
                "intersection_candidate_capacity": capacity,
            }
        )


@final
class MultiRegionSurfaceEvidence(StrictModule, NonTrainableState):
    """Host validation evidence of one multiregion surface and geometry.

    Combinatorial checks are exact integer arithmetic; cyclic face order around
    non-manifold edges, face degeneracy and embedding intersections use exact
    ``orient2d``/``orient3d`` predicates (``predicate_mode`` records the effective
    host route). ``signed_volumes`` follows ``finite_region_ids``.
    ``region_euler_characteristics`` is ``V - E + F`` of each finite region's
    boundary surface (2 for a sphere, 0 for a torus).
    ``region_component_counts`` lists, in region-table order, the connected
    components of every label (sides of faces joined across wedges);
    ``incomplete_region_graph_vertex_count`` counts interior vertices where two
    incident regions share no film and ``singular_vertex_count`` counts vertices
    carrying a sheet in more than one disconnected fan.
    """

    status: MultiRegionSurfaceStatus = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    profile: MultiRegionValidationProfile = eqx.field(static=True)
    label_orientation_consistent: bool = eqx.field(static=True)
    inconsistent_edge_count: int = eqx.field(static=True)
    finite_regions_watertight: bool = eqx.field(static=True)
    open_finite_edge_count: int = eqx.field(static=True)
    border_edge_count: int = eqx.field(static=True)
    valence_supported: bool = eqx.field(static=True)
    maximum_edge_valence: int = eqx.field(static=True)
    maximum_vertex_regions: int = eqx.field(static=True)
    nonphysical_edge_count: int = eqx.field(static=True)
    nonphysical_vertex_count: int = eqx.field(static=True)
    degenerate_face_count: int = eqx.field(static=True)
    finite: bool = eqx.field(static=True)
    finite_region_ids: tuple[str, ...] = eqx.field(static=True)
    signed_volumes: tuple[float, ...] = eqx.field(static=True)
    positive_volumes: bool = eqx.field(static=True)
    region_euler_characteristics: tuple[int, ...] = eqx.field(static=True)
    region_component_counts: tuple[int, ...] = eqx.field(static=True)
    singular_vertex_count: int = eqx.field(static=True)
    incomplete_region_graph_vertex_count: int = eqx.field(static=True)
    self_intersection_checked: bool = eqx.field(static=True)
    intersecting_pair_count: int = eqx.field(static=True)
    uncertain_pair_count: int = eqx.field(static=True)
    candidate_pair_count: int = eqx.field(static=True)
    candidate_capacity_exceeded: bool = eqx.field(static=True)
    predicate_mode: str = eqx.field(static=True)
    capacity: MultiRegionSurfaceCapacityEvidence
    topology_id: str = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


__all__ = [
    "DRY_FOAM_EDGE_VALENCE",
    "DRY_FOAM_VERTEX_REGIONS",
    "MultiRegionCoordinateDtype",
    "MultiRegionDomain",
    "MultiRegionIndexDtype",
    "MultiRegionKind",
    "MultiRegionSurfaceCapacityError",
    "MultiRegionSurfaceCapacityEvidence",
    "MultiRegionSurfaceCapacityPlan",
    "MultiRegionSurfaceCounts",
    "MultiRegionSurfaceEvidence",
    "MultiRegionSurfacePreparationError",
    "MultiRegionSurfaceStatus",
    "MultiRegionSurfaceValidationPolicy",
    "MultiRegionValidationProfile",
]
