#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quotient discovery of trigonometric level sets on the three-torus.

The zero set is discovered once on a fundamental box of the reduced lattice
coordinates ``u in R^3 / Z^3`` with opposite faces identified. The packed
polygonizer evaluates every identified corner, root edge and ambiguous face at
one canonical representative, so the lifted contour pairs exactly across the
box faces and there is no domain boundary. Lifted vertices map to Cartesian
coordinates through the declared ``PeriodicCell``: orbit representatives by
its affine map, every translate by the exact lattice action, so the canonical
``PeriodicMeshTopology`` owns windings, orbits and the quotient complex.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._cell_mesh import CellBlock, CellMesh
from ...discretization._periodic_cell import PeriodicCell
from ...discretization._periodic_exact import apply_periodic
from ...discretization._periodic_topology import PeriodicMeshTopology
from ...typing import checked
from ..analytic._trigonometric_level_set import (
    TrigonometricLevelSet,
    TrigonometricSheetBand,
)
from ._adaptive_discovery import (
    _cell_diagonal,
    _cover_identity,
    _FieldEnclosures,
    _LATTICE_ANCHORS,
    _PACKED_OWNERSHIP,
    _root_on_edges,
    _state_identity,
    _target_level,
    _validate_discovery_inputs,
    AdaptiveImplicitCover,
    AdaptiveImplicitSurfaceEvidence,
)
from ._adaptive_packed_polygon import QuotientContour
from ._policy import (
    AdaptiveImplicitBoxIssue,
    AdaptiveImplicitSurfacePolicy,
    AdaptiveImplicitSurfaceStatus,
)


# Issues that a finer quotient lattice resolves for a level separated from
# every critical value: the zero set is a regular surface, so a cell meeting it
# without a resolved contour holds a piece smaller than the cell. A tangency
# box hugs a face no deeper than ``2**-8`` of the cell; every anchor is tried
# first at each resolution, and a finer lattice halves that slab.
_REFINABLE = int(
    AdaptiveImplicitBoxIssue.EDGE
    | AdaptiveImplicitBoxIssue.MULTIPLE_SHEETS
    | AdaptiveImplicitBoxIssue.SINGULAR
    | AdaptiveImplicitBoxIssue.FACE
    | AdaptiveImplicitBoxIssue.TANGENCY
)
_ANCHOR_BITS = 30


def _anchor_offset(anchor: int, /) -> np.ndarray:
    """Dyadic offset of quotient lattice ``anchor``; ``offset + 1`` stays exact."""
    if anchor == 0:
        return np.zeros(3, dtype=np.float64)
    shift = np.asarray(_LATTICE_ANCHORS[anchor - 1], dtype=np.float64)
    return np.floor(shift * 2.0**_ANCHOR_BITS) / 2.0**_ANCHOR_BITS


@final
class PeriodicSurfaceTopology(StrictModule, NonTrainableState):
    """Combinatorial topology of a closed quotient surface on the three-torus.

    Counts are of quotient entities. ``component_genera`` are the genera of the
    closed orientable components, each from its own Euler characteristic
    ``2 - 2 g``. A component that is not a closed orientable 2-manifold is
    reported by ``closed``/``orientable`` instead of a genus.
    """

    vertex_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    face_count: int = eqx.field(static=True)
    euler_characteristic: int = eqx.field(static=True)
    component_euler_characteristics: tuple[int, ...] = eqx.field(static=True)
    component_genera: tuple[int, ...] = eqx.field(static=True)
    closed: bool = eqx.field(static=True)
    orientable: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        vertex_count: int,
        edge_count: int,
        face_count: int,
        component_euler_characteristics: tuple[int, ...],
        closed: bool,
        orientable: bool,
    ) -> None:
        self.vertex_count = int(vertex_count)
        self.edge_count = int(edge_count)
        self.face_count = int(face_count)
        self.euler_characteristic = self.vertex_count - self.edge_count + self.face_count
        self.component_euler_characteristics = tuple(
            int(value) for value in component_euler_characteristics
        )
        if sum(self.component_euler_characteristics) != self.euler_characteristic:
            raise ValueError("Component Euler characteristics must sum to the total.")
        self.closed = bool(closed)
        self.orientable = bool(orientable)
        self.component_genera = (
            tuple((2 - value) // 2 for value in self.component_euler_characteristics)
            if self.closed and self.orientable
            else ()
        )

    @property
    def component_count(self) -> int:
        return len(self.component_euler_characteristics)


def _quotient_topology(
    faces: np.ndarray,
    orbits: np.ndarray,
    windings: np.ndarray,
    topology: PeriodicMeshTopology,
    /,
) -> PeriodicSurfaceTopology:
    """Independent quotient census, cross-checked against the topology owner."""
    vertex_count = int(topology.quotient.entity_sets[0].count)
    edge_count = int(topology.quotient.entity_sets[1].count)
    face_count = int(topology.quotient.entity_sets[2].count)
    corners = orbits[faces]
    corner_windings = windings[faces]
    # A quotient edge is its ordered orbit pair and the winding difference.
    sides: dict[tuple[int, ...], int] = {}
    for first, second in ((0, 1), (1, 2), (2, 0)):
        tail, head = corners[:, first], corners[:, second]
        delta = corner_windings[:, second] - corner_windings[:, first]
        for a, b, step in zip(tail.tolist(), head.tolist(), delta.tolist(), strict=True):
            key = (a, b, *step)
            sides[key] = sides.get(key, 0) + 1
    closed = True
    orientable = True
    undirected: set[tuple[int, ...]] = set()
    for (a, b, *step), count in sides.items():
        reverse = (b, a, *(-value for value in step))
        opposite = sides.get(reverse, 0)
        canonical = min((a, b, *step), reverse)
        undirected.add(canonical)
        if count + opposite != 2:
            closed = False
        elif count != 1:
            orientable = False
    if len(undirected) != edge_count:
        raise ValueError(
            "The quotient edge census disagrees with the periodic topology owner."
        )
    parent = list(range(int(orbits.max(initial=-1)) + 1))

    def find(item: int, /) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for a, b, c in corners.tolist():
        for x, y in ((a, b), (b, c)):
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)
    present = sorted(set(orbits.tolist()))
    roots = sorted({find(item) for item in present})
    index = {root: row for row, root in enumerate(roots)}
    vertices = np.zeros(len(roots), dtype=np.int64)
    for item in present:
        vertices[index[find(item)]] += 1
    triangles = np.zeros(len(roots), dtype=np.int64)
    for a in corners[:, 0].tolist():
        triangles[index[find(a)]] += 1
    edges = np.zeros(len(roots), dtype=np.int64)
    for key in undirected:
        edges[index[find(key[0])]] += 1
    if int(vertices.sum()) != vertex_count or int(triangles.sum()) != face_count:
        raise ValueError(
            "The quotient vertex or face census disagrees with the periodic topology owner."
        )
    return PeriodicSurfaceTopology(
        vertex_count=vertex_count,
        edge_count=edge_count,
        face_count=face_count,
        component_euler_characteristics=tuple(
            int(v - e + f) for v, e, f in zip(vertices, edges, triangles, strict=True)
        ),
        closed=closed,
        orientable=orientable,
    )


@final
class PeriodicLatticeMetrics(StrictModule, NonTrainableState):
    """Lattice metrics of one periodic cell.

    ``solid_fraction_*`` enclose the region's volume fraction of the cell
    (certified by the discovery cover); porosity is its complement.
    ``measured_surface_area`` is the area of the accepted quotient contour and
    the specific surfaces derive from it: they are measured, not enclosed.
    ``vertex_root_distance`` bounds the Cartesian distance of every contour
    vertex from a root of the field on its grid edge (the contour's source
    fidelity evidence at its vertices).
    """

    cell_volume: float = eqx.field(static=True)
    solid_fraction_lower: float = eqx.field(static=True)
    solid_fraction_upper: float = eqx.field(static=True)
    measured_surface_area: float = eqx.field(static=True)
    vertex_root_distance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        cell_volume: float,
        solid_fraction_lower: float,
        solid_fraction_upper: float,
        measured_surface_area: float,
        vertex_root_distance: float,
    ) -> None:
        if not 0.0 <= solid_fraction_lower <= solid_fraction_upper <= 1.0:
            raise ValueError("Solid fraction bounds must be ordered within [0, 1].")
        self.cell_volume = float(cell_volume)
        self.solid_fraction_lower = float(solid_fraction_lower)
        self.solid_fraction_upper = float(solid_fraction_upper)
        self.measured_surface_area = float(measured_surface_area)
        self.vertex_root_distance = float(vertex_root_distance)

    @property
    def porosity_lower(self) -> float:
        return float(np.nextafter(1.0 - self.solid_fraction_upper, -np.inf))

    @property
    def porosity_upper(self) -> float:
        return float(np.nextafter(1.0 - self.solid_fraction_lower, np.inf))

    @property
    def measured_specific_surface(self) -> float:
        """Measured contour area per cell volume."""
        return self.measured_surface_area / self.cell_volume

    @property
    def measured_solid_specific_surface(self) -> tuple[float, float]:
        """Measured area per solid volume over the certified solid-fraction range."""
        lowest = self.measured_surface_area / (
            self.solid_fraction_upper * self.cell_volume
        )
        if self.solid_fraction_lower <= 0.0:
            return lowest, math.inf
        return lowest, self.measured_surface_area / (
            self.solid_fraction_lower * self.cell_volume
        )


@final
class PeriodicImplicitSurface(StrictModule, NonTrainableState):
    """Quotient discovery of a trigonometric level set over its lattice cell.

    ``mesh`` is the lifted contour over the fundamental cell, carrying the
    canonical ``PeriodicMeshTopology`` (``None`` when discovery left unresolved
    boxes; see ``evidence``). ``reduced_points``/``faces`` are the lifted contour
    in reduced coordinates with ``vertex_orbits``/``vertex_windings``.
    ``reduced_domain`` is the fundamental box ``[offset, offset + 1]^3``.
    ``cover`` is the certified quotient cover of that box: its volume fraction
    enclosure is the region's fraction of the cell. ``root_error`` bounds the
    reduced-coordinate distance of every contour vertex from a root of the
    field on its grid edge.
    """

    source: TrigonometricLevelSet
    mesh: CellMesh | None
    topology: PeriodicSurfaceTopology | None
    cover: AdaptiveImplicitCover
    evidence: AdaptiveImplicitSurfaceEvidence
    reduced_domain: np.ndarray
    reduced_points: np.ndarray
    faces: np.ndarray
    vertex_orbits: np.ndarray
    vertex_windings: np.ndarray
    scaffold_corner_ids: np.ndarray
    scaffold_corner_signs: np.ndarray
    scaffold_root_edge_ids: np.ndarray
    scaffold_triangle_root_edge_rows: np.ndarray
    vertex_root_rows: np.ndarray
    target_level: int = eqx.field(static=True)
    root_error: float = eqx.field(static=True)
    policy: AdaptiveImplicitSurfacePolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TrigonometricLevelSet,
        mesh: CellMesh | None,
        topology: PeriodicSurfaceTopology | None,
        cover: AdaptiveImplicitCover,
        evidence: AdaptiveImplicitSurfaceEvidence,
        reduced_domain: np.ndarray,
        contour: QuotientContour | None,
        scaffold: dict[str, np.ndarray],
        vertex_root_rows: np.ndarray,
        /,
        *,
        target_level: int,
        root_error: float,
        policy: AdaptiveImplicitSurfacePolicy,
        source_id: str,
    ) -> None:
        def frozen(values: np.ndarray, dtype: type, shape: tuple[int, ...]) -> np.ndarray:
            result = np.array(values, dtype=dtype, copy=True).reshape(shape)
            result.setflags(write=False)
            return result

        self.source = source
        self.mesh = mesh
        self.topology = topology
        self.cover = cover
        self.evidence = evidence
        self.reduced_domain = frozen(reduced_domain, np.float64, (2, 3))
        empty_points = np.empty((0, 3))
        self.reduced_points = frozen(
            empty_points if contour is None else contour.points, np.float64, (-1, 3)
        )
        self.faces = frozen(
            np.empty((0, 3)) if contour is None else contour.faces, np.int64, (-1, 3)
        )
        self.vertex_orbits = frozen(
            np.empty(0) if contour is None else contour.vertex_orbits, np.int64, (-1,)
        )
        self.vertex_windings = frozen(
            np.empty((0, 3)) if contour is None else contour.vertex_windings,
            np.int64,
            (-1, 3),
        )
        self.scaffold_corner_ids = frozen(scaffold["corner_ids"], np.int64, (-1, 3))
        self.scaffold_corner_signs = frozen(scaffold["corner_signs"], np.int8, (-1,))
        self.scaffold_root_edge_ids = frozen(
            scaffold["root_edge_ids"], np.int64, (-1, 2, 3)
        )
        self.scaffold_triangle_root_edge_rows = frozen(
            scaffold["triangle_root_edge_rows"], np.int64, (-1, 3)
        )
        self.vertex_root_rows = frozen(vertex_root_rows, np.int64, (-1,))
        self.target_level = int(target_level)
        self.root_error = float(root_error)
        self.policy = policy
        self.source_id = source_id

    @property
    def cell(self) -> PeriodicCell:
        return self.source.cell

    @property
    def resolved(self) -> bool:
        return self.mesh is not None

    def lattice_metrics(self) -> PeriodicLatticeMetrics:
        """Certified volume fractions and measured surface metrics of the cell.

        Solid fraction and porosity are enclosed by the quotient cover,
        independently of any mesh. Surface area is measured on the accepted
        contour (each quotient triangle once) and is not an enclosure.
        """
        if self.mesh is None:
            raise ValueError("Surface metrics need a resolved quotient contour.")
        volume = self.cover.volume_enclosure()
        vectors = np.asarray(self.cell.vectors, dtype=np.float64)
        coordinates = np.asarray(self.mesh.coordinates, dtype=np.float64)
        corners = coordinates[np.asarray(self.mesh.blocks[0].vertices, dtype=np.int64)]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        area = float(0.5 * np.sum(np.linalg.norm(normals, axis=1)))
        # A reduced displacement d moves a point by d @ vectors in space.
        stretch = float(np.linalg.norm(vectors, ord=2))
        return PeriodicLatticeMetrics(
            cell_volume=float(abs(np.linalg.det(vectors))),
            solid_fraction_lower=volume.fraction_lower,
            solid_fraction_upper=volume.fraction_upper,
            measured_surface_area=area,
            vertex_root_distance=float(np.nextafter(stretch * self.root_error, np.inf)),
        )


def reduced_trigonometric_source(
    source: TrigonometricLevelSet, /
) -> TrigonometricLevelSet:
    """The same region on the unit cubic lattice: its field is the table in ``u``."""
    sheet = (
        None if source.half_width is None else TrigonometricSheetBand(source.half_width)
    )
    return TrigonometricLevelSet(
        source.table,
        PeriodicCell(np.eye(3, dtype=np.float64)),
        level=source.level,
        region=source.region,
        sheet=sheet,
        critical_levels=source.critical_levels,
        feature_id=source.feature_id,
    )


def lift_reduced_points(
    cell: PeriodicCell,
    points: np.ndarray,
    orbits: np.ndarray,
    windings: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian lifted coordinates, orbit representatives and windings.

    Each orbit's representative is its zero-winding row, mapped by the cell's
    affine map; every other row is its representative moved by the exact
    lattice action, as the canonical periodic topology requires.
    """
    count = points.shape[0]
    representatives = np.full(count, -1, dtype=np.int64)
    root_of_orbit: dict[int, int] = {}
    for row in np.flatnonzero(~np.any(windings != 0, axis=1)).tolist():
        root_of_orbit.setdefault(int(orbits[row]), row)
    for row in range(count):
        representative = root_of_orbit.get(int(orbits[row]))
        if representative is None:
            raise ValueError(
                "A lifted orbit has no zero-winding representative in the fundamental box."
            )
        representatives[row] = representative
    roots = np.flatnonzero(representatives == np.arange(count))
    coordinates = np.empty((count, 3), dtype=np.float64)
    coordinates[roots] = np.asarray(cell.cartesian(points[roots]), dtype=np.float64)
    moved = np.flatnonzero(representatives != np.arange(count))
    if moved.size:
        coordinates[moved] = apply_periodic(
            cell, coordinates[representatives[moved]], windings[moved]
        )
    return coordinates, representatives, windings.astype(np.int64)


def _lifted_surface(
    cell: PeriodicCell, contour: QuotientContour, /
) -> tuple[CellMesh, PeriodicMeshTopology]:
    coordinates, representatives, windings = lift_reduced_points(
        cell, contour.points, contour.vertex_orbits, contour.vertex_windings
    )
    triangles = np.asarray(contour.faces, dtype=np.int32)
    if np.linalg.det(np.asarray(cell.vectors, dtype=np.float64)) < 0.0:
        # A left-handed lattice reverses the reduced orientation.
        triangles = triangles[:, ::-1].copy()
    block = CellBlock(
        "surface",
        "triangle",
        triangles,
        global_ids=np.arange(triangles.shape[0], dtype=np.int64),
    )
    vertex_ids = np.arange(coordinates.shape[0], dtype=np.int64)
    lifted = CellMesh(coordinates, (block,), vertex_global_ids=vertex_ids)
    topology = PeriodicMeshTopology(lifted, cell, representatives, windings)
    mesh = CellMesh(
        coordinates, (block,), vertex_global_ids=vertex_ids, periodic_topology=topology
    )
    return mesh, topology


def _discover_once(
    reduced: TrigonometricLevelSet,
    domain: np.ndarray,
    policy: AdaptiveImplicitSurfacePolicy,
    source_id: str,
    anchor: int,
    /,
) -> tuple[
    AdaptiveImplicitCover,
    AdaptiveImplicitSurfaceEvidence,
    QuotientContour | None,
    dict[str, np.ndarray],
    np.ndarray,
    int,
]:
    from ._adaptive_packed_frontier import discover_leaves, PackedPolygonLeaves
    from ._adaptive_packed_polygon import polygonize

    geometry = reduced.compile()
    _validate_discovery_inputs(geometry, policy, source_id)
    bounds = _FieldEnclosures(geometry, policy.enclosure)
    if not bounds.value_rigorous:
        raise ValueError("Quotient discovery requires a rigorous value enclosure.")
    state_id = _state_identity(geometry)
    target_level = _target_level(domain, policy.relative_cell_diameter)
    leaves = discover_leaves(bounds, domain, target_level)
    leaf_count = len(leaves.level)
    surface_count = len(leaves.surface_rows)
    deepest = leaves.deepest_level
    status = leaves.status
    cover_arrays = (
        leaves.boxes,
        leaves.value_lower,
        leaves.value_upper,
        leaves.gradient_lower,
        leaves.gradient_upper,
    )
    polygon_leaves = PackedPolygonLeaves.from_leaves(leaves)
    del leaves
    polygon = polygonize(
        bounds,
        domain,
        target_level,
        policy,
        polygon_leaves,
        source_id,
        True,
        periodic=True,
    )
    status |= polygon.status
    issues = polygon.unresolved_issues
    if issues.size:
        status |= int(AdaptiveImplicitSurfaceStatus.UNRESOLVED_BOXES)
    if not surface_count:
        status |= int(AdaptiveImplicitSurfaceStatus.NO_SURFACE)
    for values in (*cover_arrays, polygon.unresolved_boxes, issues):
        values.setflags(write=False)
    cover = AdaptiveImplicitCover(
        *cover_arrays,
        domain,
        source_id,
        state_id,
        policy.enclosure,
        True,
        True,
        _cover_identity(
            cover_arrays[0],
            cover_arrays[1],
            cover_arrays[2],
            domain,
            source_id,
            state_id,
            policy.enclosure,
            True,
        ),
        _ownership=_PACKED_OWNERSHIP,
    )
    declared = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), dtype=np.float64)
    evidence = AdaptiveImplicitSurfaceEvidence(
        polygon.unresolved_boxes,
        issues,
        "enclosed",
        policy.enclosure,
        status,
        leaf_count,
        surface_count,
        deepest,
        polygon.max_surface_diagonal,
        canonical_fingerprint(
            {
                "kind": "periodic-implicit-evidence",
                "status": status,
                "unresolved_boxes": array_tree_fingerprint(polygon.unresolved_boxes),
                "unresolved_issues": array_tree_fingerprint(issues),
                "cover": cover.cover_id,
                "leaf_count": leaf_count,
                "lattice_anchor": anchor,
            }
        ),
        declared_domain=declared,
        lattice_anchor=anchor,
        _ownership=_PACKED_OWNERSHIP,
    )
    scaffold = polygon.scaffold_arrays
    contour = polygon.quotient
    vertex_rows = (
        np.empty(0, dtype=np.int64)
        if contour is None
        else _contour_root_rows(scaffold, contour)
    )
    return cover, evidence, contour, scaffold, vertex_rows, target_level


def _contour_root_rows(
    scaffold: dict[str, np.ndarray], contour: QuotientContour, /
) -> np.ndarray:
    """Root-edge row of each lifted contour vertex (its grid-edge identity)."""
    rows = np.full(contour.points.shape[0], -1, dtype=np.int64)
    triangles = np.asarray(scaffold["triangle_root_edge_rows"], dtype=np.int64)
    rows[contour.faces.reshape(-1)] = triangles.reshape(-1)
    if np.any(rows < 0):
        raise ValueError("Every lifted contour vertex must name its root edge.")
    return rows


@checked
def discover_periodic_trigonometric_surface(
    source: TrigonometricLevelSet,
    /,
    *,
    policy: AdaptiveImplicitSurfacePolicy = AdaptiveImplicitSurfacePolicy(),
    source_id: str,
) -> PeriodicImplicitSurface:
    """Discover a trigonometric level set on the quotient of its lattice cell.

    The certified boundary level is already separated from every critical
    value, so the zero set is a regular closed surface on the torus. A cell
    left unresolved either holds a corner on the zero set or a contour piece
    that the lattice cannot see: it meets a face without crossing an edge, or
    crosses one edge twice near a fold of the surface along that edge. Those
    are placements of the fundamental box relative to the surface, so every
    deterministic dyadic re-anchoring of the box is tried at the resolution
    first; only when none resolves is the whole quotient lattice refined
    (halving ``relative_cell_diameter``), until a lattice resolves or the
    unresolved cells reach ``root_tolerance``. Undecided results keep the last
    attempt's unresolved boxes as evidence and carry no mesh.
    """
    if policy.enclosure != "interval":
        raise ValueError("Quotient discovery requires the interval enclosure policy.")
    reduced = reduced_trigonometric_source(source)
    current = policy
    while True:
        for anchor in range(len(_LATTICE_ANCHORS) + 1):
            offset = _anchor_offset(anchor)
            domain = np.stack((offset, offset + 1.0))
            cover, evidence, contour, scaffold, vertex_rows, level = _discover_once(
                reduced, domain, current, source_id, anchor
            )
            issues = np.asarray(evidence.unresolved_issues, dtype=np.int64)
            if not issues.size:
                break
        if not issues.size:
            break
        boxes = np.asarray(evidence.unresolved_boxes)
        widest = max((_cell_diagonal(box[0], box[1]) for box in boxes), default=0.0)
        following = current.relative_cell_diameter * 0.5
        if (
            np.any(issues & ~_REFINABLE)
            or widest <= current.root_tolerance
            or not math.isfinite(following)
            or _target_level(domain, following) <= level
        ):
            break
        current = replace(current, relative_cell_diameter=following)
    mesh = topology = None
    if contour is not None:
        mesh, periodic = _lifted_surface(source.cell, contour)
        topology = _quotient_topology(
            contour.faces, contour.vertex_orbits, contour.vertex_windings, periodic
        )
    root_errors = np.asarray(scaffold["root_errors"], dtype=np.float64)
    return PeriodicImplicitSurface(
        source,
        mesh,
        topology,
        cover,
        evidence,
        domain,
        contour,
        scaffold,
        vertex_rows,
        target_level=level,
        root_error=float(np.max(root_errors, initial=0.0)),
        policy=current,
        source_id=source_id,
    )


# Float Newton steps toward the zero set before the certified bracket search.
_NEWTON_STEPS = 3


class QuotientZeroSet:
    """Certified points of a resolved quotient discovery's zero set.

    ``project`` places points on ``{f = 0}`` of the reduced source with the
    discovery's own root certificate. A float Newton search along the gradient
    is followed by a bracket ``q +- s d`` of strictly opposite interval point
    signs and the scaffold's interval bisection (`_root_on_edges`) to the
    discovery's ``root_tolerance``. A row may hold one reduced axis fixed. Its
    search direction then has an exact zero in that axis, so the coordinate is
    kept bit for bit (a point of the zero set's trace on a cell-face plane).
    Callers evaluate only canonical representatives: the field is exactly
    lattice periodic, so the certificate holds for every exact translate.
    """

    def __init__(self, surface: PeriodicImplicitSurface, /) -> None:
        if not surface.resolved:
            raise ValueError("Only a resolved quotient discovery has a zero set.")
        geometry = reduced_trigonometric_source(surface.source).compile()
        kernel, state = geometry.kernel, geometry.state

        def value(point: jax.Array) -> jax.Array:
            return kernel.boundary_field(state, point[None, :])[0]

        self._bounds = _FieldEnclosures(geometry, "interval")
        self._values = jax.jit(jax.vmap(value))
        self._gradients = jax.jit(jax.vmap(jax.grad(value)))
        self.tolerance = float(surface.policy.root_tolerance)

    def values(self, points: np.ndarray, /) -> np.ndarray:
        """Float field values (sampled choices only, never a certificate)."""
        rows = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if not rows.shape[0]:
            return np.empty(0, dtype=np.float64)
        return np.asarray(self._values(jnp.asarray(rows)), dtype=np.float64)

    def gradients(self, points: np.ndarray, /) -> np.ndarray:
        """Float field gradients (search directions and orientation tests only)."""
        values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        if not values.shape[0]:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(self._gradients(jnp.asarray(values)), dtype=np.float64)

    def signs(self, points: np.ndarray, /) -> np.ndarray:
        """Certified field signs at points; zero where the enclosure is undecided."""
        values = np.asarray(points, dtype=np.float64).reshape((-1, 3))
        lower, upper = self._bounds.point_values(values)
        signs = np.zeros(values.shape[0], dtype=np.int8)
        signs[upper < 0.0] = -1
        signs[lower > 0.0] = 1
        return signs

    def project(
        self, starts: np.ndarray, axes: np.ndarray, reach: float, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Certified zero-set points near ``starts`` within distance ``reach``.

        ``axes[i]`` is the reduced axis row ``i`` keeps exactly, or ``-1``.
        Returns the points, their certified distance bounds to a root (in the
        kept plane for a fixed axis), the float gradients there and the rows
        that were certified. A refused row is never widened past ``reach``.
        """
        origins = np.asarray(starts, dtype=np.float64).reshape((-1, 3))
        points = np.array(origins, copy=True)
        count = points.shape[0]
        mask = np.ones((count, 3), dtype=np.float64)
        fixed = np.flatnonzero(np.asarray(axes) >= 0)
        mask[fixed, np.asarray(axes)[fixed]] = 0.0
        for _ in range(_NEWTON_STEPS):
            gradient = self.gradients(points) * mask
            value = np.asarray(self._values(jnp.asarray(points)), dtype=np.float64)
            squared = np.sum(gradient * gradient, axis=1)
            step = np.divide(value, squared, out=np.zeros(count), where=squared > 0.0)
            points = points - step[:, None] * gradient
        gradient = self.gradients(points) * mask
        norms = np.linalg.norm(gradient, axis=1)
        usable = np.isfinite(points).all(axis=1) & (norms > 0.0)
        direction = np.divide(
            gradient, norms[:, None], out=np.zeros_like(gradient), where=usable[:, None]
        )
        brackets, signs, found = self._brackets(points, direction, usable, reach)
        result = np.array(points, copy=True)
        errors = np.full(count, np.inf)
        rows = np.flatnonzero(found)
        if rows.size:
            records = _root_on_edges(
                self._bounds, brackets[rows], signs[rows], self.tolerance
            )
            for row, record in zip(rows.tolist(), records, strict=True):
                point, error, issue = record[0], record[1], record[2]
                if issue == 0 and np.linalg.norm(point - origins[row]) <= reach:
                    result[row] = point
                    errors[row] = error
        certified = np.isfinite(errors)
        return result, errors, self.gradients(result), certified

    def _brackets(
        self, points: np.ndarray, direction: np.ndarray, usable: np.ndarray, reach: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Smallest doubling half-width ``s`` with opposite certified end signs."""
        count = points.shape[0]
        half = np.full(count, 2.0 * self.tolerance)
        brackets = np.empty((count, 2, 3), dtype=np.float64)
        signs = np.zeros((count, 2), dtype=np.int8)
        found = np.zeros(count, dtype=np.bool_)
        pending = usable.copy()
        while np.any(pending):
            rows = np.flatnonzero(pending)
            offset = half[rows, None] * direction[rows]
            low, high = points[rows] - offset, points[rows] + offset
            ends = self.signs(np.concatenate((low, high)))
            first, second = ends[: rows.size], ends[rows.size :]
            done = (first != 0) & (second != 0) & (first != second)
            brackets[rows[done], 0] = low[done]
            brackets[rows[done], 1] = high[done]
            signs[rows[done], 0] = first[done]
            signs[rows[done], 1] = second[done]
            found[rows[done]] = True
            pending[rows[done]] = False
            half[rows[~done]] *= 2.0
            pending &= half <= reach
        return brackets, signs, found


@dataclass(frozen=True, slots=True)
class QuotientBoxContour:
    """A resolved discovery's quotient contour lifted into one fundamental box.

    ``domain`` is the box ``[offset, offset + 1]^3``; ``points``/``faces`` are
    the lifted contour with ``vertex_orbits``/``vertex_windings``.
    ``root_error`` bounds every vertex's reduced distance from a root of the
    field, and ``interior`` is a point of the box certified inside the region
    (the centre of the largest certified-inside cover leaf).
    """

    domain: np.ndarray
    points: np.ndarray
    faces: np.ndarray
    vertex_orbits: np.ndarray
    vertex_windings: np.ndarray
    root_error: float
    interior: np.ndarray


def _interior_point(cover: AdaptiveImplicitCover, /) -> np.ndarray:
    boxes = np.asarray(cover.boxes, dtype=np.float64)
    inside = np.flatnonzero(np.asarray(cover.value_upper) < 0.0)
    if not inside.size:
        raise ValueError("The quotient cover has no certified interior leaf.")
    sizes = boxes[inside, 1, 0] - boxes[inside, 0, 0]
    box = boxes[inside[int(np.argmax(sizes))]]
    return 0.5 * box[0] + 0.5 * box[1]


def quotient_box_contour(
    surface: PeriodicImplicitSurface, offset: np.ndarray, /
) -> QuotientBoxContour | None:
    """The resolved discovery's contour in the fundamental box at ``offset``.

    The quotient surface does not depend on the box: any dyadic ``offset``
    (``offset + 1`` exact) cuts it along other cell-face planes. The published
    box is the surface's own contour; any other box re-runs the quotient
    polygonization at the surface's resolved policy and returns ``None`` when
    that box leaves unresolved cells. Its quotient topology must equal the
    surface's.
    """
    if surface.topology is None:
        raise ValueError("Only a resolved quotient discovery has a box contour.")
    offset = np.asarray(offset, dtype=np.float64).reshape(3)
    if not np.array_equal((offset + 1.0) - 1.0, offset):
        raise ValueError("A fundamental box offset must keep offset + 1 exact.")
    if np.array_equal(offset, surface.reduced_domain[0]):
        return QuotientBoxContour(
            np.array(surface.reduced_domain, copy=True),
            np.array(surface.reduced_points, copy=True),
            np.array(surface.faces, copy=True),
            np.array(surface.vertex_orbits, copy=True),
            np.array(surface.vertex_windings, copy=True),
            surface.root_error,
            _interior_point(surface.cover),
        )
    domain = np.stack((offset, offset + 1.0))
    cover, evidence, contour, scaffold, _, _ = _discover_once(
        reduced_trigonometric_source(surface.source),
        domain,
        surface.policy,
        surface.source_id,
        surface.evidence.lattice_anchor,
    )
    if np.asarray(evidence.unresolved_issues).size or contour is None:
        return None
    _, periodic = _lifted_surface(surface.cell, contour)
    topology = _quotient_topology(
        contour.faces, contour.vertex_orbits, contour.vertex_windings, periodic
    )
    if (
        topology.euler_characteristic != surface.topology.euler_characteristic
        or topology.component_count != surface.topology.component_count
    ):
        raise ValueError("A re-anchored quotient contour changed the surface topology.")
    return QuotientBoxContour(
        domain,
        np.asarray(contour.points, dtype=np.float64),
        np.asarray(contour.faces, dtype=np.int64),
        np.asarray(contour.vertex_orbits, dtype=np.int64),
        np.asarray(contour.vertex_windings, dtype=np.int64),
        float(np.max(np.asarray(scaffold["root_errors"]), initial=0.0)),
        _interior_point(cover),
    )
