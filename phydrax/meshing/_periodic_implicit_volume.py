#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Periodic tetrahedral fill of a quotient-discovered trigonometric region.

The fundamental cell's region is closed by its lifted quotient contour and by
planar caps on the cell faces, both rebuilt from the discovery before the fill
(`periodic_box_boundary`): the contour is remeshed isotropically on the
certified zero set, its seam vertices sliding along the cell-face traces, and
each lower face's cap is a refined constrained Delaunay triangulation of the
region's trace; the upper face carries the identical triangulation moved by
the exact lattice action. The resulting piecewise-linear complex is filled by
the native fixed-boundary PLC route, which keeps every source vertex and
facet, so paired cell faces stay identical. Lifted vertex orbits then bind the
canonical ``PeriodicMeshTopology``: cap facets become interior quotient faces
and only the contour bounds the quotient region.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import final

import equinox as eqx
import numpy as np

from .._physical import SpatialCoordinateContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_mesh import CellBlock, CellMesh
from ..discretization._periodic_topology import PeriodicMeshTopology
from ..geometry.implicit._periodic_discovery import (
    lift_reduced_points,
    PeriodicImplicitSurface,
)
from ..typing import checked
from ._canonical import certify_cell_mesh
from ._contracts import CellFamilyPolicy, CellMeshingTarget, VolumeFillStrategy
from ._controls import RegionSeed
from ._organization import RegionRole
from ._periodic_box_boundary import periodic_box_boundary
from ._plc_source import PiecewiseLinearComplex
from ._result import CellMeshingResult
from ._scope import MeshingEntityKind, MeshingScope
from ._sizing import SizeControlStrength, UniformSizeControl
from ._volume_generation import NativeVolumeSchedule


# A sheet band is about one default target edge thick, so hard shape targets at
# the default resolution are a resolution-limited refusal; the sheet default
# enforces cell validity only and an explicit schedule plus a finer
# ``maximum_edge_length`` (2-3 edges across) requests the hard targets.
_SHEET_VOLUME_SCHEDULE = NativeVolumeSchedule(
    radius_edge_bound=None,
    minimum_dihedral_degrees=0.0,
    shape_strength=SizeControlStrength.SOFT,
)


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray, /) -> int:
    """Exact sign of the 2D orientation of three binary64 points."""
    ax, ay, bx, by, cx, cy = (Fraction(float(value)) for value in (*a, *b, *c))
    value = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    return (value > 0) - (value < 0)


def _triangulate_convex(
    loop: list[int], plane: Mapping[int, np.ndarray], /
) -> list[tuple[int, int, int]]:
    """Triangulate a counter-clockwise convex loop that may hold collinear vertices.

    Only strictly convex ears are clipped, and never one whose removal would
    leave the remaining loop without a strictly convex vertex, so every
    triangle is nondegenerate under exact orientation.
    """
    remaining = list(loop)
    triangles: list[tuple[int, int, int]] = []
    while len(remaining) > 3:
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            if _orientation(plane[previous], plane[current], plane[following]) <= 0:
                continue
            rest = remaining[:position] + remaining[position + 1 :]
            if not any(
                _orientation(
                    plane[rest[index - 1]],
                    plane[rest[index]],
                    plane[rest[(index + 1) % len(rest)]],
                )
                > 0
                for index in range(len(rest))
            ):
                continue
            triangles.append((previous, current, following))
            remaining = rest
            break
        else:
            raise ValueError("A cap polygon is degenerate under exact orientation.")
    if _orientation(plane[remaining[0]], plane[remaining[1]], plane[remaining[2]]) <= 0:
        raise ValueError("A cap polygon is degenerate under exact orientation.")
    triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


@final
class PeriodicImplicitVolume(StrictModule, NonTrainableState):
    """Certified periodic tetrahedral mesh of a quotient-discovered region.

    ``result`` is the certified lifted mesh carrying the canonical
    ``PeriodicMeshTopology``. ``quotient_region_components`` counts connected
    components of the quotient tetrahedra (face adjacency through paired cell
    faces). ``measure`` is the mesh volume of the fundamental cell's region.
    ``construction`` is the native fixed-boundary fill of the lifted cell with
    its compliance report (including unmet soft shape targets).
    ``boundary_triangle_count`` counts the lifted contour triangles that bound
    the quotient region, and ``root_error`` bounds the reduced-coordinate
    distance of every contour vertex from a root of the field (the remeshed
    contour's certificate, not the discovery scaffold's). ``reduced_domain`` is
    the fill's fundamental box ``[offset, offset + 1]^3``, chosen for the fill
    and in general not the discovery's box.
    """

    surface: PeriodicImplicitSurface
    result: CellMeshingResult
    construction: CellMeshingResult
    reduced_domain: np.ndarray
    cap_triangle_count: int = eqx.field(static=True)
    boundary_triangle_count: int = eqx.field(static=True)
    quotient_region_components: int = eqx.field(static=True)
    measure: float = eqx.field(static=True)
    root_error: float = eqx.field(static=True)

    def __init__(
        self,
        surface: PeriodicImplicitSurface,
        result: CellMeshingResult,
        construction: CellMeshingResult,
        reduced_domain: np.ndarray,
        /,
        *,
        cap_triangle_count: int,
        boundary_triangle_count: int,
        quotient_region_components: int,
        measure: float,
        root_error: float,
    ) -> None:
        self.surface = surface
        self.result = result
        self.construction = construction
        domain = np.array(reduced_domain, dtype=np.float64, copy=True).reshape(2, 3)
        domain.setflags(write=False)
        self.reduced_domain = domain
        self.cap_triangle_count = int(cap_triangle_count)
        self.boundary_triangle_count = int(boundary_triangle_count)
        self.quotient_region_components = int(quotient_region_components)
        self.measure = float(measure)
        self.root_error = float(root_error)

    @property
    def mesh(self) -> CellMesh:
        return self.result.mesh

    @property
    def vertex_root_distance(self) -> float:
        """Cartesian bound on every contour vertex's distance from a field root."""
        # A reduced displacement d moves a point by d @ vectors in space.
        vectors = np.asarray(self.surface.cell.vectors, dtype=np.float64)
        stretch = float(np.linalg.norm(vectors, ord=2))
        return float(np.nextafter(stretch * self.root_error, np.inf))


def _quotient_components(mesh: CellMesh, topology: PeriodicMeshTopology, /) -> int:
    """Connected components of quotient tetrahedra under shared quotient faces."""
    tetrahedra = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
    )
    orbits = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    parent = list(range(tetrahedra.shape[0]))

    def find(item: int, /) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    owner: dict[tuple[object, ...], int] = {}
    for row, cell in enumerate(tetrahedra.tolist()):
        for omitted in range(4):
            face = [cell[index] for index in range(4) if index != omitted]
            base = min(face, key=lambda vertex: (orbits[vertex], tuple(shifts[vertex])))
            key = tuple(
                sorted(
                    (
                        int(orbits[vertex]),
                        *(int(value) for value in shifts[vertex] - shifts[base]),
                    )
                    for vertex in face
                )
            )
            other = owner.setdefault(key, row)
            if other != row:
                a, b = find(other), find(row)
                if a != b:
                    parent[max(a, b)] = min(a, b)
    return len({find(row) for row in range(tetrahedra.shape[0])})


@checked
def mesh_periodic_trigonometric_region(
    surface: PeriodicImplicitSurface,
    /,
    *,
    maximum_edge_length: float | None = None,
    volume_schedule: NativeVolumeSchedule | None = None,
    coordinate_contract: SpatialCoordinateContract | None = None,
    source_revision: str = "revision-1",
) -> PeriodicImplicitVolume:
    """Fill the quotient region of a resolved periodic discovery with tetrahedra.

    ``maximum_edge_length`` is an optional hard uniform size target on the
    contour scope; it also sets the rebuilt boundary's reduced edge length. The
    native route keeps every contour and cap vertex and facet
    (``boundary="fixed"``), so the paired cell-face triangulations stay
    identical; the periodic topology then identifies them exactly. The default
    schedule for a network region is `NativeVolumeSchedule()`: hard shape
    targets (radius-edge 2, dihedral 10 degrees), met by the rebuilt boundary.
    The soft default that the discovery's marching-tetrahedra contour required
    (PeriodicCollapseSupport / TpmsDerivativeCheck) is gone for networks. A
    sheet region (``half_width`` set) is about one default target edge thick,
    so its default stays soft and validity-only; pass a hard schedule with a
    ``maximum_edge_length`` giving 2-3 edges across the sheet to require the
    shape targets.
    """
    from ._contracts import VolumeMeshingSpec
    from .providers._native import NativeMeshingProvider
    from .providers._native_options import NativeMeshingOptions
    from .providers._native_sources import NativePlcSource

    if not surface.resolved:
        raise ValueError(
            "Only a resolved quotient discovery can be meshed; see its evidence."
        )
    cell = surface.cell
    stretch = float(np.linalg.norm(np.asarray(cell.vectors, dtype=np.float64), ord=2))
    boundary = periodic_box_boundary(
        surface,
        None if maximum_edge_length is None else float(maximum_edge_length) / stretch,
    )
    polygons: list[tuple[int, int, int]] = [
        (a, b, c) for a, b, c in boundary.surface.tolist()
    ]
    facets: list[int] = [0] * len(polygons)
    segments: list[tuple[int, int]] = []
    cap_count = 0
    for axis, (lower, upper) in enumerate(boundary.caps):
        # The lower face's outward normal is -e_axis; its translate keeps the
        # counter-clockwise (+e_axis) orientation as the upper face's outward side.
        polygons.extend((a, c, b) for a, b, c in lower.tolist())
        facets.extend([1 + 2 * axis] * lower.shape[0])
        polygons.extend((a, b, c) for a, b, c in upper.tolist())
        facets.extend([2 + 2 * axis] * upper.shape[0])
        cap_count += lower.shape[0] + upper.shape[0]
        translate = dict(zip(lower.ravel().tolist(), upper.ravel().tolist(), strict=True))
        # Native source preparation merges the coplanar triangles of one cap
        # facet and drops their shared edges, after which recovery may pick
        # either diagonal of a cocircular quadrilateral on each face. Authoring
        # every interior cap edge, and its exact translate, as a segment keeps
        # both triangulations and so the lattice pairing of the cell faces.
        uses: dict[tuple[int, int], int] = {}
        for a, b, c in lower.tolist():
            for first, second in ((a, b), (b, c), (c, a)):
                key = (min(first, second), max(first, second))
                uses[key] = uses.get(key, 0) + 1
        for (first, second), count in uses.items():
            if count == 2:
                segments.append((first, second))
                segments.append((translate[first], translate[second]))
    points = boundary.points
    coordinates, representatives, windings = lift_reduced_points(
        cell, points, boundary.orbits, boundary.windings
    )
    loops = np.asarray(polygons, dtype=np.int64)
    if np.linalg.det(np.asarray(cell.vectors, dtype=np.float64)) < 0.0:
        loops = loops[:, ::-1].copy()
    facet_rows = np.asarray(facets, dtype=np.int64)
    source_id = f"{surface.source_id}-periodic-region"
    complex_ = PiecewiseLinearComplex(
        coordinates,
        loops,
        facet_rows,
        np.tile(np.asarray((-1, 0), dtype=np.int64), (7, 1)),
        ("region",),
        segments=np.asarray(segments, dtype=np.int64).reshape(-1, 2),
        vertex_ids=tuple(f"{source_id}-vertex-{row}" for row in range(points.shape[0])),
        vertex_occurrence_paths=((),) * points.shape[0],
        facet_ids=(
            f"{source_id}-contour",
            *(
                f"{source_id}-cell-face-{axis}-{side}"
                for axis in range(3)
                for side in ("lower", "upper")
            ),
        ),
        facet_occurrence_paths=((),) * 7,
        segment_ids=tuple(
            f"{source_id}-cap-edge-{first}-{second}" for first, second in segments
        ),
        segment_occurrence_paths=((),) * len(segments),
        region_occurrence_paths=((),),
        boundary="fixed",
    )
    source = NativePlcSource(complex_, source_id, source_revision)
    scope = MeshingScope(
        source.source_id,
        source.source_revision,
        MeshingEntityKind.GEOMETRY,
        2,
        source.scope_entity_set_id(2),
        source.scope_entity_ids(2),
    )
    seed = np.asarray(cell.cartesian(boundary.interior), dtype=np.float64)
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("tetrahedron",))),
        scope,
        VolumeFillStrategy.SIMPLEX,
        size_controls=(
            ()
            if maximum_edge_length is None
            else (UniformSizeControl(scope, float(maximum_edge_length)),)
        ),
        region_seeds=(
            RegionSeed(seed, "region", f"{source_id}-material", RegionRole.SOLID),
        ),
    )
    contract = (
        SpatialCoordinateContract.si()
        if coordinate_contract is None
        else coordinate_contract
    )
    lifted = (
        NativeMeshingProvider(
            NativeMeshingOptions(
                "plc_tetrahedral",
                volume_schedule=(
                    volume_schedule
                    if volume_schedule is not None
                    else _SHEET_VOLUME_SCHEDULE
                    if surface.source.half_width is not None
                    else NativeVolumeSchedule()
                ),
            )
        )
        .plan(source, specification, coordinate_contract=contract)
        .execute()
    )
    mesh = lifted.mesh
    output = np.asarray(mesh.coordinates, dtype=np.float64)
    source_rows = {coordinates[row].tobytes(): row for row in range(coordinates.shape[0])}
    count = output.shape[0]
    output_reps = np.arange(count, dtype=np.int64)
    output_shifts = np.zeros((count, 3), dtype=np.int64)
    output_of_source = np.full(coordinates.shape[0], -1, dtype=np.int64)
    for row in range(count):
        source_row = source_rows.get(output[row].tobytes())
        if source_row is not None:
            output_of_source[source_row] = row
    if np.any(output_of_source < 0):
        raise ValueError("The fixed-boundary fill dropped a source vertex.")
    for source_row in range(coordinates.shape[0]):
        row = output_of_source[source_row]
        output_reps[row] = output_of_source[representatives[source_row]]
        output_shifts[row] = windings[source_row]
    # The periodic routes and their orbit-locked adaptation owners carry
    # oriented "tetrahedron" blocks; the native fill's simplex rows are kept.
    blocks = tuple(
        CellBlock(
            block.name,
            "tetrahedron",
            np.asarray(block.vertices, dtype=np.int64),
            global_ids=np.asarray(block.global_ids, dtype=np.int64),
        )
        for block in mesh.blocks
    )
    unbound = CellMesh(output, blocks, vertex_global_ids=mesh.vertex_global_ids)
    topology = PeriodicMeshTopology(unbound, cell, output_reps, output_shifts)
    periodic = CellMesh(
        output,
        blocks,
        vertex_global_ids=mesh.vertex_global_ids,
        periodic_topology=topology,
    )
    result = certify_cell_mesh(periodic, contract)
    volumes = []
    for block in periodic.blocks:
        cells = output[np.asarray(block.vertices, dtype=np.int64)]
        edges = cells[:, 1:] - cells[:, :1]
        volumes.append(np.linalg.det(edges) / 6.0)
    return PeriodicImplicitVolume(
        surface,
        result,
        lifted,
        boundary.domain,
        cap_triangle_count=cap_count,
        boundary_triangle_count=boundary.surface.shape[0],
        quotient_region_components=_quotient_components(periodic, topology),
        measure=float(np.sum(np.abs(np.concatenate(volumes)))),
        root_error=boundary.root_error,
    )
