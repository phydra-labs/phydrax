#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming Voronoi polyhedra of a PLC solid through one native call.

The route places conforming seeds for the requested cell size
(``phydrax._meshcore.conforming_voronoi``): interior seeds mirrored across
the surface by exterior ghosts, sector seeds about sharp edges, corner seeds
and a body-centred cubic lattice. Every cell is the exact Voronoi cell of its
seed in the diagram of all seeds, hence closed and convex; the boundary is
formed by the bisector faces towards the ghosts. It lies exactly on
axis-aligned planes and approximates other facets (the kernel reports the
distance and volume error). The published geometry is the binary64 rounding
of exact circumcentres; the combinatorics are decided by exact predicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

import phydrax._meshcore as meshcore
from phydrax._physical import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization import CellMesh, PolyhedralConnectivity
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.discretization._exact_voronoi_geometry import (
    ExactVoronoiCellGeometrySource,
)
from phydrax.geometry._planar_chain_embedding import triangulate_planar_polygon
from phydrax.meshing._association import GeometryAssociation, GeometrySourceEntityRole
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    VolumeMeshingSpec,
)
from phydrax.meshing._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from phydrax.meshing._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshLabel,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from phydrax.meshing._result import MeshingComplianceReport
from phydrax.meshing._sizing import (
    size_statistic,
    SizeControlStrength,
    UniformSizeControl,
)
from phydrax.meshing._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
)
from phydrax.meshing._volume_generation import declared_plc_domain
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_polyhedral import (
    _association,
    _face_group_connected,
    _mesh_scope,
    _source_scope_rows,
    polyhedral_support_issues,
)
from phydrax.meshing.providers._native_sources import NativePlcSource
from phydrax.typing import checked


# Dihedral angle above which a surface edge is a sharp feature: sector seeds
# and per-feature mirrors keep the bisector faces on both sides of the edge.
_SEED_FEATURE_ANGLE_DEGREES = 20.0

# Vertex and edge ancestry codes, indexing _ROLES.
_REGION, _FACET = 0, 3
_ROLES = (
    GeometrySourceEntityRole.REGION,
    GeometrySourceEntityRole.VERTEX,
    GeometrySourceEntityRole.EDGE,
    GeometrySourceEntityRole.FACET,
)


def voronoi_support_issues(
    source: NativePlcSource, specification: VolumeMeshingSpec, /
) -> tuple[str, ...]:
    """Admit an affine, sheet-free PLC and a uniform cell-size request."""
    issues = list(polyhedral_support_issues(source, specification))
    complex_ = source.complex
    if source.mapped_domain is not None or complex_.geometry is not None:
        issues.append("an affine PLC without a coordinate map")
    regions = np.asarray(complex_.facet_regions, dtype=np.int64)
    if np.any(regions[:, 0] == regions[:, 1]):
        issues.append("PLC facets between two regions or a region and the exterior")
    loop_edges = _polygon_edge_keys(source)
    if any(
        (min(a, b), max(a, b)) not in loop_edges
        for a, b in np.asarray(complex_.segments, dtype=np.int64).tolist()
    ):
        issues.append("PLC segments on facet polygon edges")
    polygon_vertices = set(np.asarray(complex_.polygon_vertices, dtype=np.int64).tolist())
    vertex_bank = np.asarray(source.scope_entity_ids(0), dtype=np.int64)
    for feature in specification.protected_features:
        if feature.scope.entity_dimension != 0:
            continue
        rows = np.flatnonzero(
            np.isin(vertex_bank, np.asarray(feature.scope.entity_ids, dtype=np.int64))
        )
        if not set(rows.tolist()) <= polygon_vertices:
            issues.append("protected PLC vertices on facet polygons")
    if not specification.size_controls:
        issues.append("a uniform size control giving the Voronoi cell size")
    return tuple(dict.fromkeys(issues))


def _polygon_edge_keys(source: NativePlcSource, /) -> set[tuple[int, int]]:
    offsets = np.asarray(source.complex.polygon_offsets, dtype=np.int64)
    loops = np.asarray(source.complex.polygon_vertices, dtype=np.int64)
    following = np.arange(1, loops.size + 1, dtype=np.int64)
    following[offsets[1:] - 1] = offsets[:-1]
    pairs = np.sort(np.stack((loops, loops[following]), axis=1), axis=1)
    return set(map(tuple, np.unique(pairs, axis=0).tolist()))


def _plc_triangles(source: NativePlcSource, /) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of the PLC polygons (exact ear clipping) and their polygons."""
    complex_ = source.complex
    offsets = np.asarray(complex_.polygon_offsets, dtype=np.int64)
    loops = np.asarray(complex_.polygon_vertices, dtype=np.int64)
    sizes = np.diff(offsets)
    if np.all(sizes == 3):
        return loops.reshape((-1, 3)), np.arange(sizes.size, dtype=np.int64)
    points = np.asarray(complex_.vertices, dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []
    owners: list[int] = []
    for polygon in range(sizes.size):
        loop = loops[offsets[polygon] : offsets[polygon + 1]]
        if loop.size == 3:
            triangles.append((int(loop[0]), int(loop[1]), int(loop[2])))
            owners.append(polygon)
            continue
        exact = tuple(
            (Fraction(float(x)), Fraction(float(y)), Fraction(float(z)))
            for x, y, z in points[loop]
        )
        for a, b, c in triangulate_planar_polygon(exact):
            triangles.append((int(loop[a]), int(loop[b]), int(loop[c])))
            owners.append(polygon)
    return (
        np.asarray(triangles, dtype=np.int64).reshape((-1, 3)),
        np.asarray(owners, dtype=np.int64),
    )


@dataclass(frozen=True, slots=True)
class _Published:
    """Native cells, their canonical mesh and the canonical-to-native rows."""

    cells: meshcore.ConformingVoronoi
    mesh: CellMesh
    connectivity: PolyhedralConnectivity
    # Native cell and face of every canonical cell and face.
    cell_order: np.ndarray
    face_order: np.ndarray
    cell_regions: np.ndarray
    face_facets: np.ndarray


def _sorted_loop_table(
    offsets: np.ndarray, values: np.ndarray, width: int, /
) -> np.ndarray:
    """Rows (size, sorted vertices padded with -1) identifying each face."""
    sizes = np.diff(offsets)
    rows = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    columns = np.arange(values.size, dtype=np.int64) - np.repeat(offsets[:-1], sizes)
    table = np.full((sizes.size, width + 1), -1, dtype=np.int64)
    table[rows, columns + 1] = values
    table[:, 1:] = -np.sort(-table[:, 1:], axis=1)
    table[:, 0] = sizes
    return table


def _publish(cells: meshcore.ConformingVoronoi, /) -> _Published:
    offsets = np.asarray(cells.face_offsets, dtype=np.int64)
    vertices = np.asarray(cells.face_vertices, dtype=np.int32)
    sides = np.asarray(cells.face_cells, dtype=np.int64)
    loops: list[list[np.ndarray]] = [[] for _ in range(cells.cell_seeds.shape[0])]
    for face, (left, right) in enumerate(sides.tolist()):
        loop = vertices[offsets[face] : offsets[face + 1]]
        loops[left].append(loop)
        if right >= 0:
            loops[right].append(loop[::-1])
    mesh = CellMesh.from_polyhedra(cells.vertices, loops, block_name="conforming-voronoi")
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Conforming Voronoi publication must stay polyhedral.")
    cell_order = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    canonical_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    width = int(max(np.max(np.diff(offsets)), np.max(np.diff(canonical_offsets))))
    native = _sorted_loop_table(offsets, vertices.astype(np.int64), width)
    canonical = _sorted_loop_table(
        canonical_offsets,
        np.asarray(connectivity.face_vertex_values, dtype=np.int64),
        width,
    )
    native_rows = np.lexsort(native.T[::-1])
    canonical_rows = np.lexsort(canonical.T[::-1])
    if (
        np.unique(cell_order).size != len(loops)
        or native.shape != canonical.shape
        or not np.array_equal(native[native_rows], canonical[canonical_rows])
    ):
        raise RuntimeError(
            "Canonical polyhedra must retain their native cells and faces."
        )
    face_order = np.empty((native.shape[0],), dtype=np.int64)
    face_order[canonical_rows] = native_rows
    return _Published(
        cells,
        mesh,
        connectivity,
        cell_order,
        face_order,
        np.asarray(cells.cell_regions, dtype=np.int32)[cell_order],
        np.asarray(cells.face_facets, dtype=np.int64)[face_order],
    )


def _names(
    names: tuple[str, ...], paths: tuple[tuple[str, ...], ...], rows: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    bank = np.empty((len(names),), dtype=object)
    bank[:] = names
    path_bank = np.empty((len(paths),), dtype=object)
    path_bank[:] = [tuple(path) for path in paths]
    return bank[rows], path_bank[rows]


@dataclass(frozen=True, slots=True)
class _Ancestry:
    """Per-entity PLC role code, source row and region-side rows."""

    codes: np.ndarray
    rows: np.ndarray
    sides: np.ndarray  # region rows (front, back) or -2 for no side
    residuals: np.ndarray


def _association_from(
    source: NativePlcSource,
    mesh: CellMesh,
    dimension: int,
    target_ids: np.ndarray,
    ancestry: _Ancestry,
    /,
) -> GeometryAssociation:
    complex_ = source.complex
    banks = (
        (complex_.region_ids, complex_.region_occurrence_paths),
        (complex_.vertex_ids, complex_.vertex_occurrence_paths),
        (complex_.plc_edge_ids, complex_.plc_edge_occurrence_paths),
        (complex_.facet_ids, complex_.facet_occurrence_paths),
    )
    count = ancestry.codes.size
    identifiers = np.empty((count,), dtype=object)
    paths = np.empty((count,), dtype=object)
    for code, (names, occurrences) in enumerate(banks):
        selected = ancestry.codes == code
        if np.any(selected):
            identifiers[selected], paths[selected] = _names(
                names, occurrences, ancestry.rows[selected]
            )
    region_names = np.empty((len(complex_.region_ids) + 2,), dtype=object)
    region_names[: len(complex_.region_ids)] = complex_.region_ids
    region_names[-2:] = None
    side_names = region_names[ancestry.sides]
    roles = np.empty((len(_ROLES),), dtype=object)
    roles[:] = _ROLES
    return _association(
        source,
        mesh,
        dimension,
        target_ids,
        tuple(identifiers.tolist()),
        tuple(roles[ancestry.codes].tolist()),
        ancestry.residuals,
        tuple(map(tuple, side_names.tolist())),
        tuple(paths.tolist()),
    )


def _vertex_ancestry(
    source: NativePlcSource,
    published: _Published,
    triangles: np.ndarray,
    triangle_facets: np.ndarray,
    /,
) -> _Ancestry:
    """Facet of the nearest PLC triangle for boundary vertices, else the region."""
    complex_ = source.complex
    cells = published.cells
    nearest = np.asarray(cells.vertex_triangles, dtype=np.int64)
    points = np.asarray(cells.vertices, dtype=np.float64)
    plc_points = np.asarray(complex_.vertices, dtype=np.float64)
    facet_regions = np.asarray(complex_.facet_regions, dtype=np.int64)
    count = nearest.size
    codes = np.full((count,), _REGION, dtype=np.int64)
    rows = np.zeros((count,), dtype=np.int64)
    sides = np.full((count, 2), -2, dtype=np.int64)
    residuals = np.zeros((count,), dtype=np.float64)
    # Vertices off the boundary lie in the region of any cell around them.
    offsets = np.asarray(cells.face_offsets, dtype=np.int64)
    owners = np.asarray(cells.face_cells, dtype=np.int64)[:, 0]
    rows[np.asarray(cells.face_vertices, dtype=np.int64)] = np.repeat(
        np.asarray(cells.cell_regions, dtype=np.int64)[owners], np.diff(offsets)
    )
    on_face = np.flatnonzero(nearest >= 0)
    if on_face.size:
        codes[on_face] = _FACET
        rows[on_face] = triangle_facets[nearest[on_face]]
        corners = plc_points[triangles[nearest[on_face]]]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        residuals[on_face] = np.abs(
            np.sum((points[on_face] - corners[:, 0]) * normal, axis=1)
        ) / np.linalg.norm(normal, axis=1)
        sides[on_face] = np.where(
            facet_regions[rows[on_face]] < 0, -2, facet_regions[rows[on_face]]
        )
    return _Ancestry(codes, rows, sides, residuals)


def _edge_ancestry(published: _Published, vertices: _Ancestry, /) -> _Ancestry:
    connectivity = published.connectivity
    edges = np.asarray(connectivity.edges, dtype=np.int64)
    count = edges.shape[0]
    # Facets and regions of the faces around each edge.
    face_offsets = np.asarray(connectivity.face_edge_offsets, dtype=np.int64)
    face_edges = np.asarray(connectivity.face_edge_values, dtype=np.int64)
    edge_faces = np.repeat(
        np.arange(face_offsets.size - 1, dtype=np.int64), np.diff(face_offsets)
    )
    edge_facet = np.full((count,), -1, dtype=np.int64)
    np.maximum.at(edge_facet, face_edges, published.face_facets[edge_faces])
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)[edge_faces]
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)[edge_faces]
    region_rows = np.concatenate(
        (
            np.stack((face_edges, published.cell_regions[owner]), axis=1),
            np.stack((face_edges, published.cell_regions[neighbor]), axis=1)[
                neighbor >= 0
            ],
        )
    )
    region_rows = np.unique(region_rows, axis=0)
    starts = np.searchsorted(region_rows[:, 0], np.arange(count, dtype=np.int64))
    counts = np.bincount(region_rows[:, 0], minlength=count)
    sides = np.full((count, 2), -2, dtype=np.int64)
    sides[:, 0] = region_rows[starts, 1]
    second_side = counts > 1
    sides[second_side, 1] = region_rows[starts[second_side] + 1, 1]
    codes = np.where(edge_facet >= 0, _FACET, _REGION)
    rows = np.where(edge_facet >= 0, edge_facet, sides[:, 0])
    if np.any((codes == _REGION) & (counts != 1)):
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "A conforming Voronoi edge off the PLC lies on a region interface.",
            provider_code="native_voronoi_association",
            stage="geometry_association",
        )
    first, second = edges[:, 0], edges[:, 1]
    residuals = np.maximum(vertices.residuals[first], vertices.residuals[second])
    return _Ancestry(codes, rows, sides, residuals)


def _face_ancestry(
    source: NativePlcSource, published: _Published, vertices: _Ancestry, /
) -> _Ancestry:
    connectivity = published.connectivity
    facets = published.face_facets
    owner = np.asarray(connectivity.face_owner, dtype=np.int64)
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    regions = published.cell_regions.astype(np.int64)
    bisector = facets < 0
    if np.any(
        bisector & (neighbor >= 0) & (regions[owner] != regions[np.maximum(neighbor, 0)])
    ):
        raise MeshingFailure(
            MeshingFailureCategory.ASSOCIATION_FAILED,
            "A material interface must retain its PLC facet witness.",
            provider_code="native_voronoi_association",
            stage="geometry_association",
        )
    facet_regions = np.asarray(source.complex.facet_regions, dtype=np.int64)
    codes = np.where(bisector, _REGION, _FACET)
    rows = np.where(bisector, regions[owner], facets)
    sides = np.where(
        bisector[:, None],
        regions[owner][:, None],
        facet_regions[np.maximum(facets, 0)],
    )
    sides = np.where(sides < 0, -2, sides)
    offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    residuals = np.maximum.reduceat(
        vertices.residuals[np.asarray(connectivity.face_vertex_values, dtype=np.int64)],
        offsets[:-1],
    )
    return _Ancestry(codes, rows, sides, residuals)


def _associations(
    source: NativePlcSource,
    published: _Published,
    triangles: np.ndarray,
    triangle_facets: np.ndarray,
    /,
) -> tuple[GeometryAssociation, ...]:
    mesh, connectivity = published.mesh, published.connectivity
    vertices = _vertex_ancestry(source, published, triangles, triangle_facets)
    edges = _edge_ancestry(published, vertices)
    faces = _face_ancestry(source, published, vertices)
    regions = published.cell_regions.astype(np.int64)
    cells = _Ancestry(
        np.full(regions.shape, _REGION, dtype=np.int64),
        regions,
        np.stack((regions, regions), axis=1),
        np.zeros(regions.shape, dtype=np.float64),
    )
    return (
        _association_from(
            source,
            mesh,
            0,
            np.asarray(mesh.entity_set(0).entity_ids, dtype=np.int64),
            vertices,
        ),
        _association_from(
            source,
            mesh,
            1,
            np.asarray(connectivity.edge_global_ids, dtype=np.int64),
            edges,
        ),
        _association_from(
            source,
            mesh,
            2,
            np.asarray(connectivity.face_global_ids, dtype=np.int64),
            faces,
        ),
        _association_from(
            source,
            mesh,
            3,
            np.asarray(connectivity.cell_global_ids, dtype=np.int64),
            cells,
        ),
    )


def _organization(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    published: _Published,
    associations: tuple[GeometryAssociation, ...],
    /,
) -> NativeMeshOrganization:
    mesh, connectivity = published.mesh, published.connectivity
    complex_ = source.complex
    cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    face_ids = np.asarray(connectivity.face_global_ids, dtype=np.int64)
    controls = {control.region_name: control for control in specification.region_controls}
    seeds = {seed.region_name: seed for seed in specification.region_seeds}
    zones: list[MeshZone] = []
    for region, name in enumerate(complex_.region_ids):
        members = cell_ids[published.cell_regions == region]
        if not members.size:
            raise MeshingFailure(
                MeshingFailureCategory.ASSOCIATION_FAILED,
                f"PLC region {name!r} has no conforming Voronoi cells.",
                provider_code="native_voronoi_organization",
                stage="organization",
            )
        control, seed = controls.get(name), seeds.get(name)
        zones.append(
            MeshZone(
                name,
                MeshZoneRole.REGION,
                _mesh_scope(mesh, 3, members),
                material_id=(
                    control.material_id
                    if control is not None
                    else seed.material_id
                    if seed is not None
                    else None
                ),
                region_role=(
                    control.role
                    if control is not None
                    else seed.role
                    if seed is not None
                    else None
                ),
            )
        )
    zone_ids = {zone.name: zone.zone_id for zone in zones}
    facets = published.face_facets
    facet_regions = np.asarray(complex_.facet_regions, dtype=np.int64)
    patches: list[MeshPatch] = []
    for facet in np.unique(facets[facets >= 0]).tolist():
        rows = np.flatnonzero(facets == facet)
        patches.append(
            MeshPatch(
                f"facet:{complex_.facet_ids[facet]}",
                _mesh_scope(mesh, 2, face_ids[rows]),
                connected=_face_group_connected(connectivity, rows),
                adjacent_zone_ids=tuple(
                    zone_ids[complex_.region_ids[region]]
                    for region in facet_regions[facet].tolist()
                    if region >= 0
                ),
            )
        )
    labels: list[MeshLabel] = []
    neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int64)
    interface = np.flatnonzero((facets >= 0) & (neighbor >= 0))
    if interface.size:
        labels.append(MeshLabel("interface", _mesh_scope(mesh, 2, face_ids[interface])))
    for control in specification.patch_controls:
        requested = np.asarray(control.scope.entity_ids, dtype=np.int64)
        rows = face_ids[np.isin(facets, _source_scope_rows(source, control.scope, 2))]
        if control.required and not rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Required patch {control.name!r} has no conforming Voronoi faces.",
                provider_code="native_voronoi_organization",
                stage="organization",
                entity_ids=tuple(map(int, requested)),
            )
        if rows.size:
            patches.append(
                MeshPatch(
                    control.name,
                    _mesh_scope(mesh, 2, rows),
                    adjacent_zone_ids=tuple(
                        zone_ids[name] for name in control.adjacent_region_names
                    ),
                )
            )
    # Meshing scopes store sorted entity IDs; attribute values follow that order.
    sorted_rows = np.argsort(cell_ids, kind="stable")
    all_cells = _mesh_scope(mesh, 3, cell_ids)
    seed_rows = np.asarray(published.cells.cell_seeds, dtype=np.int64)[
        published.cell_order
    ]
    attributes = (
        MeshAttribute(
            "voronoi_seed_index",
            MeshAttributeRole.MARKER,
            all_cells,
            seed_rows[sorted_rows],
        ),
        MeshAttribute(
            "voronoi_seed",
            MeshAttributeRole.USER,
            all_cells,
            np.asarray(published.cells.seeds, dtype=np.float64)[seed_rows[sorted_rows]],
        ),
    )
    return NativeMeshOrganization(
        patches=tuple(patches),
        zones=tuple(zones),
        labels=tuple(labels),
        attributes=attributes,
        associations=associations,
    )


def _scope_cells(
    source: NativePlcSource, published: _Published, control: UniformSizeControl, /
) -> np.ndarray:
    rows = _source_scope_rows(source, control.scope, control.scope.entity_dimension)
    if control.scope.entity_dimension == 3:
        return np.flatnonzero(np.isin(published.cell_regions, rows))
    faces = np.flatnonzero(np.isin(published.face_facets, rows))
    return np.unique(np.asarray(published.connectivity.face_owner, dtype=np.int64)[faces])


def _compliance(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    published: _Published,
    /,
) -> MeshingComplianceReport:
    """Judge cell size: the edge of the cube of each cell's volume."""
    volumes = np.asarray(published.cells.cell_volumes, dtype=np.float64)
    sizes = np.cbrt(volumes[published.cell_order])
    policy = specification.size_compliance
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    for control in specification.size_controls:
        if not isinstance(control, UniformSizeControl):
            issues.append(f"unsupported_size_control:{control.control_id}")
            continue
        rows = _scope_cells(source, published, control)
        if not rows.size:
            issues.append(f"empty_size_scope:{control.control_id}")
            continue
        values = sizes[rows]
        hard = control.strength is SizeControlStrength.HARD
        requested.append((f"size:{control.control_id}:target", control.target_size))
        for statistic in policy.target_statistics:
            value = size_statistic(values, statistic)
            achieved.append((f"size:{control.control_id}:{statistic}_cell_size", value))
            tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(
                control.target_size
            )
            if hard and abs(value - control.target_size) > tolerance:
                issues.append(f"size:{control.control_id}:{statistic}")
        for name, bound, value, below in (
            ("minimum", control.minimum_size, float(np.min(values)), True),
            ("maximum", control.maximum_size, float(np.max(values)), False),
        ):
            if bound is None:
                continue
            requested.append((f"size:{control.control_id}:{name}", bound))
            achieved.append((f"size:{control.control_id}:{name}_cell_size", value))
            tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(bound)
            if hard and (
                value < bound - tolerance if below else value > bound + tolerance
            ):
                issues.append(f"{name}_size:{control.control_id}")
    stats = np.asarray(published.cells.stats, dtype=np.float64)
    achieved.append(("voronoi:volume_error", float(stats[18])))
    achieved.append(("voronoi:boundary_vertex_distance_over_size", float(stats[19])))
    achieved.append(("voronoi:repair_rounds", float(stats[14])))
    if stats[15] >= 0.0:
        achieved.extend(
            (
                ("voronoi:certified", float(stats[15])),
                ("voronoi:open_cell_edges", float(stats[16])),
                ("voronoi:delaunay_violations", float(stats[17])),
                ("voronoi:boundary_distance_bound_over_size", float(stats[20])),
            )
        )
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


@final
class _PreparedNativeVoronoi(StrictModule, NonTrainableState):
    source: NativePlcSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    cell_size: float = eqx.field(static=True)

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        cell_size: float,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.cell_size = cell_size

    def execute(self, /, *, prove: bool = True) -> NativeMeshConstruction:
        """Construct the cells; ``prove`` runs the kernel's certification."""
        complex_ = self.source.complex
        triangles, polygons = _plc_triangles(self.source)
        triangle_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)[polygons]
        regions = np.asarray(complex_.facet_regions, dtype=np.int64)[triangle_facets]
        cells = meshcore.conforming_voronoi(
            np.asarray(complex_.vertices, dtype=np.float64),
            triangles,
            regions,
            triangle_facets,
            size=self.cell_size,
            feature_angle_degrees=_SEED_FEATURE_ANGLE_DEGREES,
            prove=prove,
        )
        if cells.status is not meshcore.MeshcoreStatus.OK:
            stats = np.asarray(cells.stats, dtype=np.float64)
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Conforming Voronoi cells failed after their repairs: "
                f"{int(stats[16])} open cell edges, {int(stats[17])} local Delaunay "
                f"violations, {int(stats[14])} repair rounds.",
                provider_code="native_voronoi_certification",
                stage="volume_fill",
            )
        published = _publish(cells)
        associations = _associations(self.source, published, triangles, triangle_facets)
        organization = _organization(
            self.source, self.specification, published, associations
        )
        certification = NativeCertificationRequest(
            MeshCertificationSchedule("volume_plc"),
            self.source.source_id,
            self.source.source_revision,
            domain=declared_plc_domain(
                complex_, triangles, polygons, self.source.source_id
            ),
            cell_regions=published.cell_regions,
        )
        stages = (
            MeshingStageReport(
                MeshingStageKind.VOLUME_FILL,
                MeshingStageStatus.PASSED,
                input_ids=(complex_.complex_id,),
                output_ids=(published.mesh.mesh_id,),
                created_count=published.cell_order.size,
            ),
        )
        return NativeMeshConstruction(
            published.mesh,
            self.coordinate_contract,
            _compliance(self.source, self.specification, published),
            certification,
            geometry=CellGeometrySpec.power(
                published.mesh,
                ExactVoronoiCellGeometrySource(
                    cells.vertices,
                    cells.exact_offsets,
                    cells.exact_values,
                    source_id=self.source.source_id,
                    source_revision=self.source.source_revision,
                ),
            ),
            organization=organization,
            stages=stages,
        )


@checked
def prepare_native_voronoi(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeVoronoi:
    issues = voronoi_support_issues(source, specification)
    if options.route != "plc_conforming_voronoi":
        issues = (*issues, "the plc_conforming_voronoi route")
    if issues:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(issues),
            provider_code="native_voronoi_preflight",
            stage="source_inspection",
        )
    # The finest uniform control sets the seed spacing; compliance judges each.
    cell_size = min(
        control.target_size
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl)
    )
    return _PreparedNativeVoronoi(source, specification, coordinate_contract, cell_size)


__all__ = ["prepare_native_voronoi", "voronoi_support_issues"]
