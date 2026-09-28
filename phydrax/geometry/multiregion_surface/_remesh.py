#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative quality remeshing of multiregion surfaces (split, collapse, flip).

Remeshing is an event pass like every other topology change: device geometry
flags candidate edges (`remesh_edge_flags`), the host turns flagged edges into
canonically ordered proposals (`propose_remesh`) and `apply_surface_events`
selects disjoint 2-rings, drafts, guards, certifies and commits them.

- **Split** inserts the edge midpoint into every incident face (junction edges
  split all their films at once), so geometry, areas and region volumes are
  preserved exactly.
- **Collapse** merges an edge into one vertex under feature rules: vertices are
  ranked sheet-interior (0), feature curve (1), feature corner (2) or fixed
  wire (3); the survivor keeps the higher-ranked position, equal-ranked curve
  vertices collapse only along their curve, corners and fixed vertices are
  never removed, and the generalized link condition (common neighbours equal
  the opposite vertices of the collapsing faces) refuses pinches and
  duplicate edges on non-manifold stars.
- **Flip** replaces the diagonal of two coplanar-enough faces of one sheet
  (Delaunay criterion); junctions, borders and label changes never flip. Its
  swept tetrahedron is certified by the CCD motion of a virtual split vertex
  from the old to the new diagonal midpoint.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import nonnegative_integer, positive_finite_float
from ...typing import Bool, Dim, Float, parse
from ._edits import (
    _CCDLeg,
    _edge_faces,
    _Edit,
    _is_feature_edge,
    _neighbors,
    _new_edit,
    _oriented_faces,
    _pair,
    _star,
    _vertex_rank,
    _WorkingMesh,
)
from ._events import SurfaceEventKind, SurfaceEventStatus
from ._geometry import PreparedMultiRegionSurface
from ._state import MultiRegionSurfaceState


MultiRegionRemeshOperation: TypeAlias = Literal["split", "collapse", "flip"]

# Virtual vertex key of the flip certification motion (never a working slot).
_VIRTUAL = -1


class _FlagEdgeDim(Dim, minimum=1):
    """Edge slots of the flagged surface."""


def _edge_pair(values: Sequence[int], name: str, /) -> tuple[int, int]:
    if isinstance(values, str) or len(values) != 2:
        raise ValueError(f"{name} must hold two vertex global ids.")
    first, second = (nonnegative_integer(value, name) for value in values)
    if first == second:
        raise ValueError(f"{name} must join two distinct vertices.")
    return (first, second) if first < second else (second, first)


def _priority(value: float, /) -> float:
    priority = float(value)
    if not math.isfinite(priority):
        raise ValueError("priority must be finite.")
    return priority


@final
class EdgeSplitProposal(StrictModule, NonTrainableState):
    """Split the edge between two vertices at its midpoint (smaller priority first)."""

    edge_vertex_ids: tuple[int, int] = eqx.field(static=True)
    priority: float = eqx.field(static=True)

    def __init__(
        self, edge_vertex_ids: Sequence[int], /, *, priority: float = 0.0
    ) -> None:
        pair = _edge_pair(edge_vertex_ids, "edge_vertex_ids")
        self.edge_vertex_ids = pair
        self.priority = _priority(priority)

    @property
    def kind(self) -> SurfaceEventKind:
        return SurfaceEventKind.SPLIT

    @property
    def support_vertex_ids(self) -> tuple[int, ...]:
        return self.edge_vertex_ids


@final
class EdgeCollapseProposal(StrictModule, NonTrainableState):
    """Collapse the edge between two vertices under the feature rules."""

    edge_vertex_ids: tuple[int, int] = eqx.field(static=True)
    priority: float = eqx.field(static=True)

    def __init__(
        self, edge_vertex_ids: Sequence[int], /, *, priority: float = 0.0
    ) -> None:
        pair = _edge_pair(edge_vertex_ids, "edge_vertex_ids")
        self.edge_vertex_ids = pair
        self.priority = _priority(priority)

    @property
    def kind(self) -> SurfaceEventKind:
        return SurfaceEventKind.COLLAPSE

    @property
    def support_vertex_ids(self) -> tuple[int, ...]:
        return self.edge_vertex_ids


@final
class EdgeFlipProposal(StrictModule, NonTrainableState):
    """Flip the diagonal shared by two faces of one sheet."""

    edge_vertex_ids: tuple[int, int] = eqx.field(static=True)
    priority: float = eqx.field(static=True)

    def __init__(
        self, edge_vertex_ids: Sequence[int], /, *, priority: float = 0.0
    ) -> None:
        pair = _edge_pair(edge_vertex_ids, "edge_vertex_ids")
        self.edge_vertex_ids = pair
        self.priority = _priority(priority)

    @property
    def kind(self) -> SurfaceEventKind:
        return SurfaceEventKind.FLIP

    @property
    def support_vertex_ids(self) -> tuple[int, ...]:
        return self.edge_vertex_ids


@final
class MultiRegionRemeshPlan(StrictModule, NonTrainableState):
    """Edge-length and angle targets of quality remeshing.

    Edges longer than ``maximum_edge_length`` split; edges shorter than
    ``minimum_edge_length`` (or the shortest edge of a face whose smallest
    angle is below ``minimum_angle``) collapse; sheet-interior edges violating
    the Delaunay criterion flip when their two faces deviate from coplanarity
    by at most ``maximum_flip_dihedral``. ``operations`` selects the enabled
    operations. Angles are radians.
    """

    minimum_edge_length: float = eqx.field(static=True)
    maximum_edge_length: float = eqx.field(static=True)
    minimum_angle: float = eqx.field(static=True)
    maximum_flip_dihedral: float = eqx.field(static=True)
    operations: tuple[MultiRegionRemeshOperation, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_edge_length: float,
        maximum_edge_length: float,
        minimum_angle: float = math.radians(12.0),
        maximum_flip_dihedral: float = math.radians(20.0),
        operations: Sequence[MultiRegionRemeshOperation] = ("split", "collapse", "flip"),
    ) -> None:
        shortest = positive_finite_float(minimum_edge_length, "minimum_edge_length")
        longest = positive_finite_float(maximum_edge_length, "maximum_edge_length")
        if not 2.0 * shortest < longest:
            raise ValueError(
                "maximum_edge_length must exceed twice minimum_edge_length so a "
                "split never produces a collapsible edge."
            )
        angle = positive_finite_float(minimum_angle, "minimum_angle")
        dihedral = positive_finite_float(maximum_flip_dihedral, "maximum_flip_dihedral")
        if angle >= math.pi / 3.0 or dihedral >= math.pi / 2.0:
            raise ValueError(
                "minimum_angle < pi/3 and maximum_flip_dihedral < pi/2 required."
            )
        if isinstance(operations, str):
            raise TypeError("operations must be a sequence of operation names.")
        selected = tuple(
            sorted(
                {
                    parse(value, MultiRegionRemeshOperation, "operations")
                    for value in operations
                }
            )
        )
        if not selected:
            raise ValueError("operations must enable at least one operation.")
        self.minimum_edge_length = shortest
        self.maximum_edge_length = longest
        self.minimum_angle = angle
        self.maximum_flip_dihedral = dihedral
        self.operations = selected
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiregion-remesh-plan",
                "minimum_edge_length": float(shortest).hex(),
                "maximum_edge_length": float(longest).hex(),
                "minimum_angle": float(angle).hex(),
                "maximum_flip_dihedral": float(dihedral).hex(),
                "operations": list(selected),
            }
        )


@final
class MultiRegionRemeshFlags(StrictModule):
    """Device geometry flags of every edge slot (inactive slots are ``False``).

    ``opposite_angle_sums`` is the sum of the two angles opposite a two-face
    sheet edge (``pi`` is the Delaunay boundary; zero elsewhere).
    """

    __strict_contract__ = True

    edge_lengths: Float[_FlagEdgeDim]
    opposite_angle_sums: Float[_FlagEdgeDim]
    split: Bool[_FlagEdgeDim]
    collapse: Bool[_FlagEdgeDim]
    flip: Bool[_FlagEdgeDim]


def _corner_angles(corners: jnp.ndarray, /) -> jnp.ndarray:
    first = jnp.roll(corners, -1, axis=1) - corners
    second = jnp.roll(corners, 1, axis=1) - corners
    cosine = jnp.sum(first * second, axis=-1) / jnp.maximum(
        jnp.linalg.norm(first, axis=-1) * jnp.linalg.norm(second, axis=-1), 1e-300
    )
    return jnp.arccos(jnp.clip(cosine, -1.0, 1.0))


def remesh_edge_flags(
    prepared: PreparedMultiRegionSurface,
    positions: ArrayLike,
    plan: MultiRegionRemeshPlan,
    /,
) -> MultiRegionRemeshFlags:
    """Device flags of split, collapse and flip candidates at ``positions``."""
    if not isinstance(prepared, PreparedMultiRegionSurface):
        raise TypeError("prepared must be a PreparedMultiRegionSurface.")
    if not isinstance(plan, MultiRegionRemeshPlan):
        raise TypeError("plan must be a MultiRegionRemeshPlan.")
    topology = prepared.topology
    points = jnp.asarray(positions)
    edges = jnp.maximum(topology.edges, 0)
    active = topology.edge_active
    lengths = jnp.where(
        active, jnp.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1), 0.0
    )
    corners = prepared.face_corner_positions(points)
    angles = _corner_angles(corners)
    face_minimum = jnp.where(topology.face_active, jnp.min(angles, axis=1), jnp.inf)
    face_edges = jnp.maximum(topology.face_edges, 0)
    shortest = jnp.take_along_axis(
        face_edges, jnp.argmin(lengths[face_edges], axis=1)[:, None], axis=1
    )[:, 0]
    needle = face_minimum < plan.minimum_angle
    needled = jnp.zeros(lengths.shape, dtype=jnp.bool_).at[shortest].max(needle)
    edge_faces = jnp.maximum(topology.edge_faces, 0)
    valence = jnp.sum(topology.edge_faces >= 0, axis=1)
    first, second = edge_faces[:, 0], edge_faces[:, min(1, topology.valence_width - 1)]
    same_labels = jnp.all(
        topology.face_labels[first] == topology.face_labels[second], axis=1
    )
    opposite = jnp.maximum(prepared.edge_opposite, 0)
    tips = points[opposite[:, : min(2, topology.valence_width)]]
    base, head = points[edges[:, 0]][:, None, :], points[edges[:, 1]][:, None, :]
    left, right = base - tips, head - tips
    cosine = jnp.sum(left * right, axis=-1) / jnp.maximum(
        jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1), 1e-300
    )
    angle_sum = jnp.sum(jnp.arccos(jnp.clip(cosine, -1.0, 1.0)), axis=1)
    area_vectors = prepared.face_area_vectors(points)
    normal_first, normal_second = area_vectors[first], area_vectors[second]
    dihedral_cosine = jnp.sum(normal_first * normal_second, axis=1) / jnp.maximum(
        jnp.linalg.norm(normal_first, axis=1) * jnp.linalg.norm(normal_second, axis=1),
        1e-300,
    )
    sheet = active & (valence == 2) & same_labels
    enabled = set(plan.operations)
    split = active & (lengths > plan.maximum_edge_length) & ("split" in enabled)
    collapse = (
        active
        & ((lengths < plan.minimum_edge_length) | needled)
        & ("collapse" in enabled)
    )
    flip = (
        sheet
        & (angle_sum > jnp.pi * (1.0 + 1.0e-9))
        & (dihedral_cosine >= math.cos(plan.maximum_flip_dihedral))
        & ("flip" in enabled)
    )
    return MultiRegionRemeshFlags(
        edge_lengths=lengths,
        opposite_angle_sums=jnp.where(sheet, angle_sum, 0.0),
        split=split,
        collapse=collapse,
        flip=flip,
    )


RemeshProposal: TypeAlias = EdgeSplitProposal | EdgeCollapseProposal | EdgeFlipProposal


def propose_remesh(
    prepared: PreparedMultiRegionSurface,
    state: MultiRegionSurfaceState,
    plan: MultiRegionRemeshPlan,
    /,
) -> tuple[RemeshProposal, ...]:
    """Canonically ordered remeshing proposals of one state (host boundary).

    Collapses come shortest first, splits longest first and flips by largest
    Delaunay violation, ties broken by vertex global ids.
    """
    topology = prepared.topology
    state.require_topology(topology)
    flags = remesh_edge_flags(prepared, state.positions, plan)
    count = topology.edge_count
    edges = np.asarray(topology.edges[:count], dtype=np.int64)
    ids = np.asarray(topology.vertex_global_ids, dtype=np.int64)[edges]
    lengths = np.asarray(flags.edge_lengths[:count], dtype=np.float64)
    sums = np.asarray(flags.opposite_angle_sums[:count], dtype=np.float64)
    proposals: list[RemeshProposal] = []
    for edge in np.flatnonzero(np.asarray(flags.collapse[:count])).tolist():
        proposals.append(EdgeCollapseProposal(ids[edge], priority=lengths[edge]))
    for edge in np.flatnonzero(np.asarray(flags.split[:count])).tolist():
        proposals.append(EdgeSplitProposal(ids[edge], priority=-lengths[edge]))
    for edge in np.flatnonzero(np.asarray(flags.flip[:count])).tolist():
        proposals.append(EdgeFlipProposal(ids[edge], priority=-sums[edge]))
    return tuple(
        sorted(
            proposals,
            key=lambda proposal: (
                int(proposal.kind),
                proposal.priority,
                proposal.edge_vertex_ids,
            ),
        )
    )


# ------------------------------------------------------------------ builders


def _split_edit(
    mesh: _WorkingMesh, first: int, second: int, /
) -> _Edit | SurfaceEventStatus:
    faces = _edge_faces(mesh, first, second)
    if not faces:
        return SurfaceEventStatus.SUPPORT_INVALID
    if bool(mesh.fixed[first]) and bool(mesh.fixed[second]):
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    middle = mesh.vertex_count
    new_faces, labels, parents = [], [], []
    for face in faces:
        start, end = (
            (first, second)
            if _oriented_faces(mesh, face, first, second)
            else (second, first)
        )
        opposite = int(
            next(v for v in mesh.faces[face].tolist() if v not in (first, second))
        )
        new_faces += [(start, middle, opposite), (middle, end, opposite)]
        labels += [mesh.labels[face], mesh.labels[face]]
        parents += [(face,), (face,)]
    return _new_edit(
        SurfaceEventKind.SPLIT,
        removed_faces=faces,
        new_faces=new_faces,
        new_labels=labels,
        new_face_parents=parents,
        new_positions=[0.5 * (mesh.positions[first] + mesh.positions[second])],
        new_vertex_parents=[(first, second)],
    )


def _collapse_target(
    mesh: _WorkingMesh, first: int, second: int, /
) -> tuple[int, int, np.ndarray] | SurfaceEventStatus:
    """``(kept, removed, position)`` under the vertex feature ranks."""
    ranks = (_vertex_rank(mesh, first), _vertex_rank(mesh, second))
    along = _is_feature_edge(mesh, first, second)
    if ranks[0] == ranks[1]:
        if ranks[0] >= 2 or (ranks[0] == 1 and not along):
            return SurfaceEventStatus.FEATURE_NOT_PRESERVED
        kept, removed = (
            (first, second)
            if mesh.vertex_ids[first] < mesh.vertex_ids[second]
            else (second, first)
        )
        return kept, removed, 0.5 * (mesh.positions[first] + mesh.positions[second])
    kept, removed = (first, second) if ranks[0] > ranks[1] else (second, first)
    if min(ranks) == 1 and not along:
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    return kept, removed, mesh.positions[kept].copy()


def _collapse_edit(
    mesh: _WorkingMesh, first: int, second: int, /
) -> _Edit | SurfaceEventStatus:
    shared = _edge_faces(mesh, first, second)
    if not shared:
        return SurfaceEventStatus.SUPPORT_INVALID
    target = _collapse_target(mesh, first, second)
    if isinstance(target, SurfaceEventStatus):
        return target
    kept, removed, position = target
    opposite = {
        int(v)
        for face in shared
        for v in mesh.faces[face].tolist()
        if v not in (first, second)
    }
    if (_neighbors(mesh, first) & _neighbors(mesh, second)) != opposite:
        return SurfaceEventStatus.LINK_CONDITION_VIOLATED
    removed_faces = _star(mesh, (removed,)).tolist()
    new_faces, labels, parents = [], [], []
    for face in removed_faces:
        row = mesh.faces[face].tolist()
        if kept in row:
            continue
        new_faces.append([kept if vertex == removed else vertex for vertex in row])
        labels.append(mesh.labels[face])
        parents.append((face,))
    star = _star(mesh, (first, second))
    keys = np.unique(mesh.faces[star])
    start = mesh.positions[keys]
    end = start.copy()
    end[np.searchsorted(keys, first)] = position
    end[np.searchsorted(keys, second)] = position
    leg = _CCDLeg(
        keys=keys,
        faces=np.searchsorted(keys, mesh.faces[star]),
        start=start,
        end=end,
        exclusions=((first, second),),
    )
    moved = {} if np.array_equal(position, mesh.positions[kept]) else {kept: position}
    return _new_edit(
        SurfaceEventKind.COLLAPSE,
        removed_faces=removed_faces,
        new_faces=new_faces,
        new_labels=labels,
        new_face_parents=parents,
        removed_vertices=(removed,),
        moved=moved,
        survivors=((kept, (kept, removed)),),
        legs=(leg,),
    )


def _flip_edit(
    mesh: _WorkingMesh, first: int, second: int, /
) -> _Edit | SurfaceEventStatus:
    faces = _edge_faces(mesh, first, second)
    if len(faces) != 2:
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    one, other = faces
    if not np.array_equal(mesh.labels[one], mesh.labels[other]) or _pair(
        mesh, one
    ) != _pair(mesh, other):
        return SurfaceEventStatus.FEATURE_NOT_PRESERVED
    if not _oriented_faces(mesh, one, first, second):
        one, other = other, one
    if not _oriented_faces(mesh, other, second, first):
        return SurfaceEventStatus.LABEL_ORIENTATION_INCONSISTENT
    tip = int(next(v for v in mesh.faces[one].tolist() if v not in (first, second)))
    base = int(next(v for v in mesh.faces[other].tolist() if v not in (first, second)))
    if tip == base or base in _neighbors(mesh, tip):
        return SurfaceEventStatus.LINK_CONDITION_VIOLATED
    keys = np.asarray((first, second, tip, base, _VIRTUAL), dtype=np.int64)
    start = np.concatenate(
        (
            mesh.positions[[first, second, tip, base]],
            (0.5 * (mesh.positions[first] + mesh.positions[second]))[None, :],
        )
    )
    end = start.copy()
    end[4] = 0.5 * (mesh.positions[tip] + mesh.positions[base])
    leg = _CCDLeg(
        keys=keys,
        faces=np.asarray(((0, 4, 2), (4, 1, 2), (1, 4, 3), (4, 0, 3)), dtype=np.int64),
        start=start,
        end=end,
        exclusions=(),
    )
    label = mesh.labels[one]
    return _new_edit(
        SurfaceEventKind.FLIP,
        removed_faces=(one, other),
        new_faces=((tip, first, base), (base, second, tip)),
        new_labels=(label, label),
        new_face_parents=((one, other), (one, other)),
        legs=(leg,),
    )


__all__ = [
    "EdgeCollapseProposal",
    "EdgeFlipProposal",
    "EdgeSplitProposal",
    "MultiRegionRemeshFlags",
    "MultiRegionRemeshOperation",
    "MultiRegionRemeshPlan",
    "RemeshProposal",
    "propose_remesh",
    "remesh_edge_flags",
]
