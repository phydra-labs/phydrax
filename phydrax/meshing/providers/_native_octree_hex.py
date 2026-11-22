#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hex-dominant meshes of a PLC solid by a snapped octree, one native call.

The route (``plc_octree_hex_dominant``, selected by
``VolumeFillStrategy.HEX_DOMINANT_POLYHEDRAL_TRANSITION``) meshes the solid of
a single-region PLC with ``phydrax._meshcore.hex_dominant``: a balanced octree
from the volume size control's target down to the boundary size control's
target (a power-of-two refinement), cut to the cells inside the surface and
clear of it; one projection layer of hexahedra joins that core's staircase
boundary to the closest-point images of its vertices, snapped to the feature
curves (dihedral angle above 45 degrees or a facet change) and corners; the
fine band, the 2:1 transitions and the wall are smoothed and then optimised for
OpenFOAM's non-orthogonality and skewness. A PROVIDER boundary-layer control
splits every projection column at its exact cumulative thicknesses from the
wall into hexahedral layers. Every cell is published as a polyhedron with
OpenFOAM face semantics: a non-planar face is the fan of triangles from its
exact vertex average, so the target is ``CellFamilyPolicy(required=
("polyhedron",))``; the ``cell_shape`` marker classifies the cells (0 six
quadrilaterals, the hexahedra; 1 a prism of a feature-curve step; 2 other, the
2:1 transitions with hanging midpoints). Consumers needing trilinear
hexahedra use ``plc_hex_dominant`` (the tet-to-hex split).

Certification contract. The wall approximates the PLC (its vertices lie on it,
its faces are chords), so the route requests ``volume_plc_fidelity``: a
certified two-sided distance bound between the mesh's exterior fan surface and
the PLC's exterior facets within the request's ``maximum_deviation``, else the
boundary size. With ``prove=True`` the kernel also certifies every cell before
publication by the exact orientation of its fan decomposition tetrahedra and a
closed genus-0 boundary; an uncertified cell refuses the request. With
``prove=False`` neither the cells nor the PLC fidelity source are prepared, so
the candidate refuses ``prove()``; the fidelity source is built on the plan's
first proved execution and reused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import final

import equinox as eqx
import numpy as np

import phydrax._meshcore as meshcore
from phydrax._physical import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization import CellMesh, PolyhedralConnectivity
from phydrax.discretization._cell_geometry import CellGeometrySpec
from phydrax.geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from phydrax.meshing._association import GeometryAssociation
from phydrax.meshing._certification_schedule import MeshCertificationSchedule
from phydrax.meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import BoundaryLayerRoute
from phydrax.meshing._native_construction import (
    NativeCertificationRequest,
    NativeMeshConstruction,
    NativeMeshOrganization,
)
from phydrax.meshing._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from phydrax.meshing._result import MeshingComplianceReport
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
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
    _face_group_connected,
    _mesh_scope,
    _source_scope_rows,
)
from phydrax.meshing.providers._native_sources import NativePlcSource
from phydrax.meshing.providers._native_voronoi import (
    _Ancestry,
    _association_from,
    _edge_ancestry,
    _face_ancestry,
    _FACET,
    _native_face_mesh,
    _plc_triangles,
    _REGION,
    plc_fidelity_query,
    plc_fidelity_request,
)
from phydrax.typing import checked


# Normal angle above which a surface edge is a feature curve (cfMesh's
# surfaceFeatureEdges -angle 45 on the reference rows).
_FEATURE_ANGLE_DEGREES = 45.0

# Cell roles of the kernel, as published in the "octree_cell_role" attribute.
_CORE, _PROJECTION, _LAYER = 0, 1, 2


def _whole_scope(
    source: NativePlcSource, scope_dimension: int, specification_scope: object, /
) -> bool:
    if not isinstance(specification_scope, MeshingScope):
        return False
    return (
        specification_scope.entity_kind is MeshingEntityKind.GEOMETRY
        and specification_scope.entity_dimension == scope_dimension
        and (specification_scope.source_id, specification_scope.source_revision)
        == (source.source_id, source.source_revision)
        and specification_scope.entity_set_id
        == source.scope_entity_set_id(scope_dimension)
        and np.array_equal(
            np.sort(np.asarray(specification_scope.entity_ids, dtype=np.int64)),
            np.sort(source.scope_entity_ids(scope_dimension)),
        )
    )


def _size_controls(
    specification: VolumeMeshingSpec, /
) -> tuple[UniformSizeControl | None, UniformSizeControl | None]:
    """The volume (core) and boundary uniform size controls."""
    volume = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl) and control.scope.entity_dimension == 3
    )
    boundary = tuple(
        control
        for control in specification.size_controls
        if isinstance(control, UniformSizeControl) and control.scope.entity_dimension == 2
    )
    return (
        volume[0] if len(volume) == 1 else None,
        boundary[0] if len(boundary) == 1 else None,
    )


def _sizes(specification: VolumeMeshingSpec, /) -> tuple[float, float] | None:
    """Core and boundary sizes, the boundary a power-of-two division of the core."""
    volume, boundary = _size_controls(specification)
    if volume is None:
        return None
    core = volume.target_size
    finest = core if boundary is None else boundary.target_size
    levels = 0
    while np.ldexp(finest, levels) < core:
        levels += 1
    if np.ldexp(finest, levels) != core:
        return None
    return core, finest


def octree_hex_dominant_support_issues(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    /,
) -> tuple[str, ...]:
    """Admit an affine single-solid PLC and the octree sizing it needs."""
    issues: list[str] = []
    if options.route != "plc_octree_hex_dominant":
        issues.append("the plc_octree_hex_dominant route")
    target = specification.target
    families = target.cell_families
    if (
        target.topological_dimension != 3
        or target.ambient_dimension != 3
        or target.geometry_order != 1
    ):
        issues.append("affine cells in three ambient dimensions")
    if (
        families.required != ("polyhedron",)
        or families.preferred
        or families.allowed_transitions
    ):
        issues.append("a polyhedral target (cells with OpenFOAM fan faces)")
    if (
        specification.fill_strategy
        is not VolumeFillStrategy.HEX_DOMINANT_POLYHEDRAL_TRANSITION
    ):
        issues.append("the hex-dominant polyhedral-transition fill strategy")
    complex_ = source.complex
    if source.mapped_domain is not None or complex_.geometry is not None:
        issues.append("an affine PLC without a coordinate map")
    regions = np.asarray(complex_.facet_regions, dtype=np.int64)
    if len(complex_.region_ids) != 1 or not np.all(np.sort(regions, axis=1) == [-1, 0]):
        issues.append("one solid region bounded by exterior facets")
    if not _whole_scope(source, 2, specification.boundary_scope):
        issues.append("the whole source PLC facet boundary on its bound revision")
    if any(
        not isinstance(control, UniformSizeControl)
        for control in specification.size_controls
    ):
        issues.append("uniform Euclidean size controls")
    volume, boundary = _size_controls(specification)
    if volume is None or not _whole_scope(source, 3, volume.scope):
        issues.append("one uniform size control on the whole solid giving the core size")
    if boundary is not None and not _whole_scope(source, 2, boundary.scope):
        issues.append("at most one uniform size control on the whole boundary")
    if volume is not None and _sizes(specification) is None:
        issues.append("a boundary size equal to the core size divided by a power of two")
    if len(specification.size_controls) > 2:
        issues.append("only the core and boundary size controls")
    layers = specification.layer_controls
    if len(layers) > 1:
        issues.append("at most one boundary-layer control")
    for control in layers:
        if control.route is not BoundaryLayerRoute.PROVIDER:
            issues.append("a PROVIDER boundary-layer control (hexahedral layers)")
        if (
            not _whole_scope(source, 2, control.wall_scope)
            or control.cap_scope is not None
        ):
            issues.append("layers grown from the whole boundary without caps")
    if specification.protected_features:
        issues.append("no protected features (feature curves follow the dihedral angle)")
    if specification.hole_seeds or specification.periodic_constraints:
        issues.append("no hole seeds or periodic constraints")
    return tuple(dict.fromkeys(issues))


@dataclass(frozen=True, slots=True)
class _Published:
    """Native mesh, its canonical mesh and the canonical-to-native face rows."""

    native: meshcore.HexDominant
    mesh: CellMesh
    connectivity: PolyhedralConnectivity
    # Native face of every canonical face; native cell rows are the canonical
    # cell IDs.
    face_order: np.ndarray
    cell_order: np.ndarray
    face_facets: np.ndarray
    cell_regions: np.ndarray


def _publish(native: meshcore.HexDominant, /) -> _Published:
    """Every cell as a polyhedron over the native face loops (fan faces)."""
    kinds = np.asarray(native.cell_kinds, dtype=np.int64)
    face_offsets = np.asarray(native.face_offsets, dtype=np.int64)
    mesh, face_order = _native_face_mesh(
        native.points,
        face_offsets,
        np.asarray(native.face_vertices, dtype=np.int32),
        np.asarray(native.face_owner, dtype=np.int64),
        np.asarray(native.face_neighbor, dtype=np.int64),
        kinds.size,
        "cells",
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Hex-dominant publication must carry polyhedral connectivity.")
    cell_order = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
    if face_order.size != face_offsets.size - 1:
        raise RuntimeError("Canonical cells must retain the native faces.")
    face_facets = np.asarray(native.face_facets, dtype=np.int64)
    return _Published(
        native,
        mesh,
        connectivity,
        face_order,
        cell_order,
        face_facets[face_order],
        np.zeros((cell_order.size,), dtype=np.int64),
    )


def _vertex_ancestry(
    source: NativePlcSource,
    published: _Published,
    triangles: np.ndarray,
    triangle_facets: np.ndarray,
    /,
) -> _Ancestry:
    """Facet of the nearest PLC triangle for wall vertices, else the region."""
    native = published.native
    nearest = np.asarray(native.vertex_triangles, dtype=np.int64)
    points = np.asarray(native.points, dtype=np.float64)
    plc_points = np.asarray(source.complex.vertices, dtype=np.float64)
    count = nearest.size
    codes = np.full((count,), _REGION, dtype=np.int64)
    rows = np.zeros((count,), dtype=np.int64)
    sides = np.full((count, 2), -2, dtype=np.int64)
    residuals = np.zeros((count,), dtype=np.float64)
    wall = np.flatnonzero(nearest >= 0)
    if wall.size:
        codes[wall] = _FACET
        rows[wall] = triangle_facets[nearest[wall]]
        corners = plc_points[triangles[nearest[wall]]]
        normal = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        residuals[wall] = np.abs(
            np.sum((points[wall] - corners[:, 0]) * normal, axis=1)
        ) / np.linalg.norm(normal, axis=1)
        facet_regions = np.asarray(source.complex.facet_regions, dtype=np.int64)
        sides[wall] = np.where(
            facet_regions[rows[wall]] < 0, -2, facet_regions[rows[wall]]
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
    edges = _edge_ancestry(
        connectivity, published.face_facets, published.cell_regions, vertices
    )
    faces = _face_ancestry(
        source, connectivity, published.face_facets, published.cell_regions, vertices
    )
    regions = published.cell_regions
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
    name = complex_.region_ids[0]
    controls = {control.region_name: control for control in specification.region_controls}
    seeds = {seed.region_name: seed for seed in specification.region_seeds}
    control, seed = controls.get(name), seeds.get(name)
    zone = MeshZone(
        name,
        MeshZoneRole.REGION,
        _mesh_scope(mesh, 3, cell_ids),
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
    facets = published.face_facets
    patches: list[MeshPatch] = []
    for facet in np.unique(facets[facets >= 0]).tolist():
        rows = np.flatnonzero(facets == facet)
        patches.append(
            MeshPatch(
                f"facet:{complex_.facet_ids[facet]}",
                _mesh_scope(mesh, 2, face_ids[rows]),
                connected=_face_group_connected(connectivity, rows),
                adjacent_zone_ids=(zone.zone_id,),
            )
        )
    for patch in specification.patch_controls:
        requested = np.asarray(patch.scope.entity_ids, dtype=np.int64)
        rows = face_ids[np.isin(facets, _source_scope_rows(source, patch.scope, 2))]
        if patch.required and not rows.size:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Required patch {patch.name!r} has no hex-dominant wall faces.",
                provider_code="native_octree_hex_organization",
                stage="organization",
                entity_ids=tuple(map(int, requested)),
            )
        if rows.size:
            patches.append(
                MeshPatch(
                    patch.name,
                    _mesh_scope(mesh, 2, rows),
                    adjacent_zone_ids=(zone.zone_id,),
                )
            )
    # Meshing scopes store sorted entity IDs; attribute values follow that order.
    native = published.native
    order = published.cell_order[np.argsort(cell_ids, kind="stable")]
    # Shape classes by face counts, as the ledger counts hexahedra: 0 six
    # quadrilaterals, 1 two triangles and three quadrilaterals, 2 other.
    owner = np.asarray(native.face_owner, dtype=np.int64)
    neighbor = np.asarray(native.face_neighbor, dtype=np.int64)
    inner = neighbor >= 0
    sizes = np.diff(np.asarray(native.face_offsets, dtype=np.int64))
    count = int(np.asarray(native.cell_kinds).size)

    def per_cell(selected: np.ndarray) -> np.ndarray:
        return np.bincount(owner[selected], minlength=count) + np.bincount(
            neighbor[selected & inner], minlength=count
        )

    faces = per_cell(np.ones_like(inner))
    quads, triangles = per_cell(sizes == 4), per_cell(sizes == 3)
    shapes = np.where(
        (faces == 6) & (quads == 6),
        0,
        np.where((faces == 5) & (quads == 3) & (triangles == 2), 1, 2),
    )
    scope = _mesh_scope(mesh, 3, cell_ids)
    attributes = (
        MeshAttribute(
            "octree_cell_role",
            MeshAttributeRole.MARKER,
            scope,
            np.asarray(native.cell_roles, dtype=np.int64)[order],
        ),
        MeshAttribute("cell_shape", MeshAttributeRole.MARKER, scope, shapes[order]),
    )
    return NativeMeshOrganization(
        patches=tuple(patches),
        zones=(zone,),
        attributes=attributes,
        associations=associations,
    )


def _cell_volumes(native: meshcore.HexDominant, /) -> np.ndarray:
    """Cell volumes by the divergence theorem over the face loops."""
    offsets = np.asarray(native.face_offsets, dtype=np.int64)
    values = np.asarray(native.face_vertices, dtype=np.int64)
    points = np.asarray(native.points, dtype=np.float64)
    sizes = np.diff(offsets)
    rows = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    following = np.arange(1, values.size + 1, dtype=np.int64)
    following[offsets[1:] - 1] = offsets[:-1]
    first, second = points[values], points[values[following]]
    flux = np.bincount(
        rows,
        np.einsum("ij,ij->i", first, np.cross(first, second)) / 6.0,
        minlength=sizes.size,
    )
    owner = np.asarray(native.face_owner, dtype=np.int64)
    neighbor = np.asarray(native.face_neighbor, dtype=np.int64)
    cells = np.asarray(native.cell_kinds).size
    inner = neighbor >= 0
    return np.bincount(owner, flux, minlength=cells) - np.bincount(
        neighbor[inner], flux[inner], minlength=cells
    )


def _compliance(
    specification: VolumeMeshingSpec, published: _Published, /
) -> MeshingComplianceReport:
    """Core cells against the volume size, wall faces against the boundary size,
    layer cells against the layer thicknesses."""
    native = published.native
    roles = np.asarray(native.cell_roles, dtype=np.int64)
    volume_control, boundary_control = _size_controls(specification)
    core_sizes = np.cbrt(_cell_volumes(native)[roles == _CORE])
    offsets = np.asarray(native.face_offsets, dtype=np.int64)
    values = np.asarray(native.face_vertices, dtype=np.int64)
    points = np.asarray(native.points, dtype=np.float64)
    # Wall face sizes: the square root of the area of each boundary loop.
    sizes = np.diff(offsets)
    rows = np.repeat(np.arange(sizes.size, dtype=np.int64), sizes)
    following = np.arange(1, values.size + 1, dtype=np.int64)
    following[offsets[1:] - 1] = offsets[:-1]
    twice = np.cross(points[values], points[values[following]])
    area = np.stack(
        [np.bincount(rows, twice[:, axis], minlength=sizes.size) for axis in range(3)],
        axis=1,
    )
    wall = np.asarray(native.face_neighbor) < 0
    wall_sizes = np.sqrt(0.5 * np.linalg.norm(area[wall], axis=1))
    policy = specification.size_compliance
    requested: list[tuple[str, float]] = []
    achieved: list[tuple[str, float]] = []
    issues: list[str] = []
    for control, sizes in ((volume_control, core_sizes), (boundary_control, wall_sizes)):
        if control is None or not sizes.size:
            continue
        hard = control.strength is SizeControlStrength.HARD
        requested.append((f"size:{control.control_id}:target", control.target_size))
        tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(
            control.target_size
        )
        for statistic in policy.target_statistics:
            value = size_statistic(sizes, statistic)
            achieved.append((f"size:{control.control_id}:{statistic}_cell_size", value))
            if hard and abs(value - control.target_size) > tolerance:
                issues.append(f"size:{control.control_id}:{statistic}")
    stats = np.asarray(native.stats, dtype=np.float64)
    cell_offsets = np.asarray(native.cell_vertex_offsets, dtype=np.int64)
    cell_values = np.asarray(native.cell_vertices, dtype=np.int64)
    counts = np.diff(cell_offsets)
    core_count = int(np.count_nonzero(roles == _CORE))
    for layer in specification.layer_controls:
        thicknesses = np.asarray(layer.schedule.thicknesses, dtype=np.float64)
        requested.append((f"layers:{layer.control_id}:count", float(thicknesses.size)))
        # Column cells run from the wall, thicknesses.size + 1 per piece; a
        # layer cell's thickness is the mean height of its upper corners over
        # the plane of its wall-side face (its last half of corners).
        rows = np.flatnonzero((roles == _LAYER) & (counts > 0))
        level = (rows - core_count) % (thicknesses.size + 1)
        achieved.append(
            (f"layers:{layer.control_id}:cells", float(np.count_nonzero(roles == _LAYER)))
        )
        heights: list[np.ndarray] = []
        levels: list[np.ndarray] = []
        for size in (6, 8):
            chosen = counts[rows] == size
            corners = cell_values[cell_offsets[rows[chosen]][:, None] + np.arange(size)]
            upper, lower = (
                points[corners[:, : size // 2]],
                points[corners[:, size // 2 :]],
            )
            normal = np.sum(np.cross(lower, np.roll(lower, -1, axis=1)), axis=1)
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            heights.append(
                np.abs(np.einsum("ckj,cj->c", upper - lower, normal)) / (size // 2)
            )
            levels.append(level[chosen])
        height, height_level = np.concatenate(heights), np.concatenate(levels)
        for index, thickness in enumerate(thicknesses):
            values = height[height_level == index]
            if not values.size:
                continue
            key = f"layers:{layer.control_id}:{index}"
            achieved.append((f"{key}:thickness_min", float(np.min(values))))
            achieved.append((f"{key}:thickness_max", float(np.max(values))))
            if np.min(values) < layer.minimum_thickness_fraction * thickness:
                issues.append(f"{key}:thickness")
    achieved.extend(
        (
            ("octree:cells", float(stats[7])),
            ("octree:hexahedra", float(stats[8])),
            ("octree:polyhedra", float(stats[9])),
            ("octree:wall_vertex_distance", float(stats[15])),
            ("octree:sampled_boundary_distance", float(stats[16])),
            ("octree:levels", float(stats[25])),
        )
    )
    if stats[22] >= 0.0:
        achieved.extend(
            (
                ("octree:certified", float(stats[22])),
                ("octree:certified_cells", float(stats[10])),
                ("octree:uncertified_cells", float(stats[23])),
            )
        )
    return MeshingComplianceReport(
        specification.specification_id,
        issues=tuple(issues),
        requested=tuple(requested),
        achieved=tuple(achieved),
    )


_UNPROVED = (
    "This hex-dominant mesh was constructed with prove=False: the kernel did "
    "not certify its cells and the PLC fidelity source was not prepared; "
    "execute(prove=True) constructs and proves it."
)


class _FidelityQueryPreparation:
    """The PLC's exterior facets as the fidelity request's source: a property
    of the source alone that only a proof consumes, prepared once on first use."""

    __slots__ = ("_query", "source")

    def __init__(self, source: NativePlcSource, /) -> None:
        self.source = source
        self._query: PiecewiseLinearBoundarySource | None = None

    def get(self, /) -> PiecewiseLinearBoundarySource:
        if self._query is None:
            triangles, polygons = _plc_triangles(self.source)
            self._query = plc_fidelity_query(
                declared_plc_domain(
                    self.source.complex, triangles, polygons, self.source.source_id
                )
            )
        return self._query


@final
class _PreparedNativeOctreeHexDominant(StrictModule, NonTrainableState):
    source: NativePlcSource
    specification: VolumeMeshingSpec
    coordinate_contract: SpatialCoordinateContract
    fidelity_query: _FidelityQueryPreparation
    core_size: float = eqx.field(static=True)
    boundary_size: float = eqx.field(static=True)

    def __init__(
        self,
        source: NativePlcSource,
        specification: VolumeMeshingSpec,
        coordinate_contract: SpatialCoordinateContract,
        core_size: float,
        boundary_size: float,
        /,
    ) -> None:
        self.source = source
        self.specification = specification
        self.coordinate_contract = coordinate_contract
        self.fidelity_query = _FidelityQueryPreparation(source)
        self.core_size = core_size
        self.boundary_size = boundary_size

    def execute(self, /, *, prove: bool = True) -> NativeMeshConstruction:
        """Construct the mesh; ``prove`` runs the kernel's cell certification."""
        complex_ = self.source.complex
        triangles, polygons = _plc_triangles(self.source)
        triangle_facets = np.asarray(complex_.polygon_facets, dtype=np.int64)[polygons]
        regions = np.asarray(complex_.facet_regions, dtype=np.int64)[triangle_facets]
        # Outward triangles: the exterior lies in front.
        outward = np.where((regions[:, 0] < 0)[:, None], triangles, triangles[:, ::-1])
        used, compact = np.unique(outward, return_inverse=True)
        layers = self.specification.layer_controls
        thicknesses = (
            np.asarray(layers[0].schedule.thicknesses, dtype=np.float64)
            if layers
            else np.zeros((0,), dtype=np.float64)
        )
        native = meshcore.hex_dominant(
            np.asarray(complex_.vertices, dtype=np.float64)[used],
            compact.reshape((-1, 3)),
            triangle_facets,
            core_size=self.core_size,
            boundary_size=self.boundary_size,
            feature_angle_degrees=_FEATURE_ANGLE_DEGREES,
            layer_thicknesses=thicknesses,
            prove=prove,
        )
        if native.status is not meshcore.MeshcoreStatus.OK:
            stats = np.asarray(native.stats, dtype=np.float64)
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                f"{int(stats[23])} hex-dominant cells failed certification after "
                "their repairs.",
                provider_code="native_octree_hex_certification",
                stage="volume_fill",
            )
        published = _publish(native)
        organization = _organization(
            self.source,
            self.specification,
            published,
            _associations(self.source, published, triangles, triangle_facets),
        )
        geometry = CellGeometrySpec.affine(published.mesh)
        tolerance = self.specification.maximum_deviation
        certification = (
            plc_fidelity_request(
                published.mesh,
                geometry,
                self.fidelity_query.get(),
                self.boundary_size if tolerance is None else tolerance,
            )
            if prove
            else NativeCertificationRequest(
                MeshCertificationSchedule("volume_plc_fidelity"),
                self.source.source_id,
                self.source.source_revision,
            )
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
            _compliance(self.specification, published),
            certification,
            geometry=geometry,
            organization=organization,
            stages=stages,
            uncertifiable=None if prove else _UNPROVED,
        )


@checked
def prepare_native_octree_hex_dominant(
    source: NativePlcSource,
    specification: VolumeMeshingSpec,
    options: NativeMeshingOptions,
    coordinate_contract: SpatialCoordinateContract,
    /,
) -> _PreparedNativeOctreeHexDominant:
    issues = octree_hex_dominant_support_issues(source, specification, options)
    sizes = _sizes(specification)
    if issues or sizes is None:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "; ".join(issues),
            provider_code="native_octree_hex_preflight",
            stage="source_inspection",
        )
    if (
        specification.maximum_deviation is not None
        and specification.maximum_deviation <= 0.0
    ):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "A snapped hex-dominant wall approximates curved facets and cannot "
            "meet a zero maximum_deviation.",
            provider_code="native_octree_hex_preflight",
            stage="source_inspection",
        )
    return _PreparedNativeOctreeHexDominant(
        source, specification, coordinate_contract, sizes[0], sizes[1]
    )


__all__ = ["octree_hex_dominant_support_issues", "prepare_native_octree_hex_dominant"]
