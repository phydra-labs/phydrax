#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Authored curved periodic layer/core source for the native W09 lifecycle.

The mapped source wall vertices and quadratic source coefficients are authored
from one ``x = 0`` representative set. Every ``x = 1`` mate is formed by the
single translation ``x -> x + 1`` and carries an explicit orbit identity.
"""

from __future__ import annotations

from typing import final

import equinox as eqx
import numpy as np
from numpy.typing import NDArray

import phydrax.meshing as M
import phydrax.meshing._layer_core_topology as _layer_core
from phydrax import array_tree_fingerprint, canonical_fingerprint, StrictModule
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    coordinate_lagrange_element,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    LayerColumnCellGeometryElement,
    PeriodicIsometryGroup,
    PeriodicMeshTopology,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry import (
    MappedDomainBoundarySource,
    MappedReferenceDomain,
    PiecewiseLinearDomain,
)
from phydrax.meshing import (
    _layer_curving,
    BoundaryLayerMesh,
    GeometryAssociation,
    PiecewiseLinearComplex,
)
from phydrax.meshing._host_banks import entity_key
from phydrax.meshing._layer_core_resources import LayerCoreSourceWork
from phydrax.meshing.providers import NativeLayerCoreSource
from phydrax.typing import checked


ORIGINAL_CURVED_PERIODIC_SOURCE_ID = "curved-periodic-narrow-gap"
ORIGINAL_CURVED_PERIODIC_SOURCE_DIGEST = (
    "ebeaa8557815eada07f50d04b77fe40d4ed2809392af56d147c6876bace97a35"
)
ORIGINAL_CURVED_PERIODIC_SOURCE_REVISION = (
    "fadef789744332ba906c2bf1b82231d331e7cc7f4128997191b2d97cb9301e4f"
)
CORRECTED_CURVED_PERIODIC_SOURCE_ID = "curved-periodic-narrow-gap-exact-x-orbits"
CURVED_PERIODIC_LAYER_PERIOD = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
CURVED_PERIODIC_LAYER_FIRST_THICKNESS = 0.01
CORRECTED_CURVED_PERIODIC_PROFILE_HEIGHT = 3.0 / 16.0
CURVED_PERIODIC_PROFILE_GAP = 1.0 / 16.0
ORIGINAL_CURVED_PERIODIC_TRACE_RESIDUALS = (
    "-101121/2361183241434822606848",
    "-6620711477/77371252455336267181195264",
)


@final
class _CoreLayout(StrictModule):
    rows: NDArray[np.int64]
    incidence: NDArray[np.int64]
    boundary_count: int = eqx.field(static=True)
    cap_polygon_ids: NDArray[np.int64]
    representatives: NDArray[np.int64]
    shifts: NDArray[np.int64]
    seam_pairs: NDArray[np.int64]


@final
class AuthoredPeriodicLayerSource(StrictModule):
    """Store corrected source controls and explicit periodic orbit identities."""

    domain: PiecewiseLinearDomain
    wall: CellMesh
    wall_association: GeometryAssociation
    source: NativeLayerCoreSource
    specification: M.VolumeMeshingSpec
    options: M.NativeMeshingOptions
    source_digest: str = eqx.field(static=True)
    wall_orbit_ids: tuple[str, ...] = eqx.field(static=True)
    coefficient_orbit_ids: tuple[str, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        domain: PiecewiseLinearDomain,
        wall: CellMesh,
        wall_association: GeometryAssociation,
        source: NativeLayerCoreSource,
        specification: M.VolumeMeshingSpec,
        options: M.NativeMeshingOptions,
        source_digest: str,
        wall_orbit_ids: tuple[str, ...],
        coefficient_orbit_ids: tuple[str, ...],
    ) -> None:
        self.domain = domain
        self.wall = wall
        self.wall_association = wall_association
        self.source = source
        self.specification = specification
        self.options = options
        self.source_digest = source_digest
        self.wall_orbit_ids = wall_orbit_ids
        self.coefficient_orbit_ids = coefficient_orbit_ids

    @property
    def source_id(self) -> str:
        return self.source.source_id

    @property
    def source_revision(self) -> str:
        return self.source.source_revision


def _periodic_group() -> PeriodicIsometryGroup:
    generator = np.eye(4, dtype=np.float64)[None]
    generator[0, :3, 3] = CURVED_PERIODIC_LAYER_PERIOD
    return PeriodicIsometryGroup(generator)


def _translated_wall(height: float, /) -> tuple[CellMesh, tuple[str, ...]]:
    """Author seam vertices from the two endpoint representatives.

    Quadratic curvature is carried by the source coefficient bank, so the
    corner carrier stays the fundamental-domain rectangle rather than a
    piecewise-linear surrogate ridge.
    """
    y = np.asarray((0.0, 1.0), dtype=np.float64)
    z = 2.0 * (1.0 - y) * y * height
    representatives = np.column_stack((np.zeros(2, dtype=np.float64), y, z))
    points = np.concatenate(
        (representatives, representatives + CURVED_PERIODIC_LAYER_PERIOD)
    )
    triangles = np.asarray(((0, 2, 3), (0, 3, 1)), dtype=np.int64)
    vertex_ids = np.arange(points.shape[0], dtype=np.int64)
    block = CellBlock(
        "fundamental-profile-strips",
        "triangle",
        triangles,
        global_ids=np.arange(100, 102, dtype=np.int64),
    )
    plain = CellMesh(points, (block,), vertex_global_ids=vertex_ids)
    roots = np.asarray((0, 1, 0, 1), dtype=np.int64)
    shifts = np.asarray(((0,), (0,), (1,), (1,)), dtype=np.int64)
    topology = PeriodicMeshTopology(plain, _periodic_group(), roots, shifts)
    wall = CellMesh(
        points,
        (block,),
        vertex_global_ids=vertex_ids,
        periodic_topology=topology,
    )
    orbit_ids = tuple(f"corrected-wall-x-orbit:{root}" for root in roots.tolist())
    return wall, orbit_ids


def _whole_mesh_scope(mesh: CellMesh) -> M.MeshingScope:
    faces = mesh.entity_set(2)
    return M.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        M.MeshingEntityKind.MESH,
        2,
        faces.entity_set_id,
        faces.entity_ids,
    )


def _prepare_reference_layers(
    wall: CellMesh, schedule: M.LayerSchedule, /
) -> BoundaryLayerMesh:
    control = M.BoundaryLayerControl(
        _whole_mesh_scope(wall),
        schedule,
        route=M.BoundaryLayerRoute.ADVANCING,
        collision=M.BoundaryLayerCollisionPolicy.FAIL,
        corner=M.BoundaryLayerCornerPolicy.SMOOTH,
        feature_angle=0.1,
        growth_rate_bounds=(0.9999999999, 1.0000000001),
    )
    layers = M.prepare_boundary_layers(
        wall,
        control,
        policy=M.BoundaryLayerPolicy(visibility_iterations=128, proximity_samples=16),
    )
    if layers.cap is None or layers.cap.periodic_topology is None:
        raise RuntimeError(
            f"reference_layers.cap must retain periodic topology; got {layers.cap!r}"
        )
    return layers


def _profile_points(
    layers: BoundaryLayerMesh, wall: CellMesh, /
) -> tuple[np.ndarray, np.ndarray]:
    cap = layers.cap
    if cap is None:
        raise RuntimeError("layer_profile.cap must be present; got None")
    roots, _ = _layer_curving.column_roots(layers)
    identifiers = np.asarray(cap.vertex_global_ids, dtype=np.int64)
    profile_ids = np.asarray([roots[int(value)] for value in identifiers], dtype=np.int64)
    if np.unique(profile_ids).size != profile_ids.size:
        raise RuntimeError(
            f"layer_profile.root_ids must identify one cap vertex per wall vertex; got {profile_ids!r}"
        )
    by_id = dict(
        zip(
            np.asarray(wall.vertex_global_ids, dtype=np.int64).tolist(),
            np.asarray(wall.coordinates, dtype=np.float64),
            strict=True,
        )
    )
    points = np.asarray([by_id[int(value)] for value in profile_ids], dtype=np.float64)
    return points, profile_ids


def _root_core_mesh(
    layers: BoundaryLayerMesh, profile_points: np.ndarray, /
) -> tuple[CellMesh, np.ndarray]:
    cap = layers.cap
    if cap is None:
        raise RuntimeError("root_core.cap must be present; got None")
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in cap.blocks]
    )
    cap_points = np.asarray(cap.coordinates, dtype=np.float64)
    upper = profile_points.copy()
    upper[:, 2] = CURVED_PERIODIC_PROFILE_GAP
    cells = np.concatenate((cap_rows, cap_rows + cap_points.shape[0]), axis=1)
    mesh = CellMesh(
        np.concatenate((cap_points, upper)),
        (
            CellBlock(
                "declared-core-roots",
                "prism",
                cells,
                global_ids=np.arange(700, 700 + cells.shape[0], dtype=np.int64),
            ),
        ),
    )
    return mesh, cap_rows


def _core_layout(
    core_mesh: CellMesh,
    cap: CellMesh,
    cap_rows: np.ndarray,
    profile_points: np.ndarray,
    /,
    *,
    work: LayerCoreSourceWork,
) -> _CoreLayout:
    topology = cap.periodic_topology
    if topology is None:
        raise RuntimeError(
            "core_layout.periodic_topology must retain cap ancestry; got None"
        )
    count = cap.coordinates.shape[0]
    roots = np.asarray(topology.vertex_representatives, dtype=np.int64)
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    image = np.arange(2 * count, dtype=np.int64)
    for vertex in np.flatnonzero(shifts[:, 0] == 1):
        image[roots[vertex]] = vertex
        image[count + roots[vertex]] = count + vertex
    polygons: list[tuple[int, ...]] = [tuple(row) for row in cap_rows.tolist()]
    incidence: list[tuple[int, int]] = [(0, -1)] * len(polygons)
    cap_keys = {tuple(sorted(row.tolist())) for row in cap_rows}
    seam_source: list[int] = []
    seam_target: dict[tuple[int, ...], int] = {}
    internal: list[tuple[int, ...]] = []
    core_faces = _layer_core.faces(
        core_mesh, np.zeros(core_mesh.entity_set(3).count, dtype=np.int64), work=work
    )
    for key, incidents in core_faces.items():
        if len(incidents) == 2:
            internal.extend(_layer_core.triangles(incidents[0].vertices))
            continue
        if key in cap_keys:
            continue
        face = incidents[0].vertices
        if all(vertex >= count for vertex in face):
            role = "upper"
            triangles = _layer_core.triangles(face)
        else:
            feet = np.asarray(
                sorted({int(vertex) % count for vertex in face}), dtype=np.int64
            )
            points = profile_points[feet]
            roles = [
                name
                for name, axis, value in (
                    ("x0", 0, 0.0),
                    ("x1", 0, 1.0),
                    ("y0", 1, 0.0),
                    ("y1", 1, 1.0),
                )
                if feet.size == 2 and np.all(points[:, axis] == value)
            ]
            if len(roles) != 1:
                raise RuntimeError(
                    f"core_layout.side_roles must contain one orbit role; got {roles!r}"
                )
            role = roles[0]
            a, b = feet.tolist()
            if (int(roots[a]), a) > (int(roots[b]), b):
                a, b = b, a
            triangles = ((a, b, count + b), (a, count + b, count + a))
            if _layer_core.cycle((a, b, count + b, count + a)) != _layer_core.cycle(face):
                triangles = tuple(tuple(reversed(row)) for row in triangles)
        for triangle in triangles:
            row = len(polygons)
            polygons.append(tuple(int(value) for value in triangle))
            incidence.append((-1, 0))
            if role == "x0":
                seam_source.append(row)
            elif role == "x1":
                seam_target[tuple(sorted(triangle))] = row
    boundary_count = len(polygons)
    pairs = np.asarray(
        [
            (
                row,
                seam_target[
                    tuple(
                        sorted(image[np.asarray(polygons[row], dtype=np.int64)].tolist())
                    )
                ],
            )
            for row in seam_source
        ],
        dtype=np.int64,
    ).reshape(-1, 2)
    polygons.extend(tuple(int(value) for value in row) for row in internal)
    incidence.extend((0, 0) for _ in internal)
    return _CoreLayout(
        np.asarray(polygons, dtype=np.int64),
        np.asarray(incidence, dtype=np.int64),
        boundary_count,
        np.arange(cap_rows.shape[0], dtype=np.int64),
        np.concatenate((roots, count + roots)),
        np.concatenate((shifts, shifts)),
        pairs,
    )


def _complex(
    points: np.ndarray,
    layout: _CoreLayout,
    *,
    reference: bool,
) -> PiecewiseLinearComplex:
    count = layout.rows.shape[0] if reference else layout.boundary_count
    return PiecewiseLinearComplex(
        points,
        tuple(layout.rows[:count]),
        np.arange(count, dtype=np.int64),
        layout.incidence[:count],
        ("core-material",),
        boundary="fixed",
    )


def _reference_source(
    layers: BoundaryLayerMesh,
    core_mesh: CellMesh,
    layout: _CoreLayout,
    /,
    *,
    work: LayerCoreSourceWork,
) -> tuple[
    NativeLayerCoreSource,
    PiecewiseLinearDomain,
    np.ndarray,
    np.ndarray,
]:
    cap = layers.cap
    if cap is None:
        raise RuntimeError("reference_source.cap must be present; got None")
    complex_ = _complex(np.asarray(core_mesh.coordinates), layout, reference=True)
    cell_count = layers.mesh.entity_set(3).count
    source = NativeLayerCoreSource(
        layers,
        complex_,
        f"{CORRECTED_CURVED_PERIODIC_SOURCE_ID}:reference-columns",
        complex_.complex_id,
        vertex_layer_ids=np.concatenate(
            (
                np.asarray(layers.cap_vertices, dtype=np.int64),
                np.full(cap.coordinates.shape[0], -1, dtype=np.int64),
            )
        ),
        cap_polygon_ids=layout.cap_polygon_ids,
        layer_regions=np.zeros(cell_count, dtype=np.int64),
        region_ids=("wall-material", "core-material"),
        core_region_map=np.asarray((1,), dtype=np.int64),
        core_vertex_representatives=layout.representatives,
        core_vertex_shifts=layout.shifts,
        core_seam_polygon_pairs=layout.seam_pairs,
    )
    mapping, cap_ids, polygons, points = _layer_core.prepare_identity(
        layers, complex_, source.vertex_layer_ids, source.cap_polygon_ids, work=work
    )
    domain, _ = _layer_core.combined_domain(
        layers,
        complex_,
        source.layer_regions,
        mapping,
        cap_ids,
        polygons,
        points,
        source.source_id,
        source.region_ids,
        source.core_region_map,
        work=work,
    )
    return source, domain, mapping, polygons


def _physical_domain(reference: PiecewiseLinearDomain, /) -> PiecewiseLinearDomain:
    return PiecewiseLinearDomain(
        reference.vertices,
        reference.facets,
        reference.facet_regions,
        reference.region_ids,
        source_id=CORRECTED_CURVED_PERIODIC_SOURCE_ID,
    )


def _facet_lookup(domain: PiecewiseLinearDomain, /) -> dict[tuple[int, ...], int]:
    return {
        tuple(sorted(np.asarray(row, dtype=np.int64).tolist())): index
        for index, row in enumerate(np.asarray(domain.facets, dtype=np.int64))
    }


def _physical_control(
    wall: CellMesh,
    reference_layers: BoundaryLayerMesh,
    domain: PiecewiseLinearDomain,
    schedule: M.LayerSchedule,
    /,
) -> tuple[M.BoundaryLayerControl, GeometryAssociation, np.ndarray]:
    lookup = _facet_lookup(domain)
    wall_vertices = np.asarray(reference_layers.wall_vertices, dtype=np.int64)
    triangles = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in wall.blocks]
    )
    source_rows = np.asarray(
        [lookup[tuple(sorted(wall_vertices[row].tolist()))] for row in triangles],
        dtype=np.int64,
    )
    facet_set = f"{domain.domain_id}:facets"
    wall_scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        facet_set,
        source_rows,
    )
    volume_scope = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        3,
        f"{domain.domain_id}:regions",
        np.asarray((0,), dtype=np.int64),
    )
    control = M.BoundaryLayerControl(
        wall_scope,
        schedule,
        route=M.BoundaryLayerRoute.ADVANCING,
        volume_scope=volume_scope,
        collision=M.BoundaryLayerCollisionPolicy.FAIL,
        corner=M.BoundaryLayerCornerPolicy.SMOOTH,
        feature_angle=0.1,
        growth_rate_bounds=(0.9999999999, 1.0000000001),
    )
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in wall.blocks]
    )
    association = GeometryAssociation(
        M.GeometryAssociationKind.PIECEWISE_LINEAR,
        domain.source_id,
        domain.source_revision,
        wall.entity_set(2).entity_set_id,
        cell_ids,
        tuple(
            entity_key(domain.source_revision, "facet", int(row)) for row in source_rows
        ),
        np.zeros(source_rows.size, dtype=np.float64),
        exact=True,
        source_dimensions=np.full(source_rows.size, 2, dtype=np.int64),
        source_indices=source_rows,
        source_entity_roles=(M.GeometrySourceEntityRole.FACET,) * source_rows.size,
        orientations=-np.ones(source_rows.size, dtype=np.int32),
    )
    return control, association, source_rows


def _physical_layers(
    wall: CellMesh,
    control: M.BoundaryLayerControl,
    association: GeometryAssociation,
    domain: PiecewiseLinearDomain,
    /,
) -> BoundaryLayerMesh:
    layers = M.prepare_boundary_layers(
        wall,
        control,
        wall_association=association,
        source_domain=domain,
        policy=M.BoundaryLayerPolicy(visibility_iterations=128, proximity_samples=16),
    )
    if layers.cap is None or layers.cap.periodic_topology is None:
        raise RuntimeError(
            f"physical_layers.cap must retain periodic topology; got {layers.cap!r}"
        )
    return layers


def _profile_geometry(
    profile: CellMesh,
) -> tuple[CellGeometrySpec, CellGeometrySpec, tuple[str, ...]]:
    element = coordinate_lagrange_element("triangle", 2)
    space = FiniteElementPlan(
        profile, FiniteElementFieldSpec("coordinates", element)
    ).prepare()
    route = np.asarray(space.dof_maps[0].cell_dofs[0], dtype=np.int64)
    parameters = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    lower = np.column_stack(
        (
            parameters[:, 0],
            parameters[:, 1],
            2.0
            * (1.0 - parameters[:, 1])
            * parameters[:, 1]
            * CORRECTED_CURVED_PERIODIC_PROFILE_HEIGHT,
        )
    )
    orbit_ids = [f"corrected-profile-coefficient:{row}" for row in range(lower.shape[0])]
    for target in np.flatnonzero(parameters[:, 0] == 1.0):
        matches = np.flatnonzero(
            (parameters[:, 0] == 0.0) & (parameters[:, 1] == parameters[target, 1])
        )
        if matches.size != 1:
            raise RuntimeError(
                f"profile.coefficient_representatives must contain one representative for row {target}; got {matches!r}"
            )
        representative = int(matches[0])
        lower[target] = lower[representative] + CURVED_PERIODIC_LAYER_PERIOD
        orbit = f"corrected-profile-x-orbit:{parameters[target, 1].hex()}"
        orbit_ids[representative] = orbit
        orbit_ids[target] = orbit
    upper = lower + np.asarray((0.0, 0.0, CURVED_PERIODIC_PROFILE_GAP), dtype=np.float64)
    elements = {profile.blocks[0].name: element}
    routes = {profile.blocks[0].name: route}
    return (
        CellGeometrySpec(elements, routes, lower),
        CellGeometrySpec(elements, routes, upper),
        tuple(orbit_ids),
    )


def _root_mesh_and_geometry(
    reference_layers: BoundaryLayerMesh,
    physical_layers: BoundaryLayerMesh,
    reference_wall: CellMesh,
    layout: _CoreLayout,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[CellMesh, CellGeometrySpec, np.ndarray, tuple[str, ...], np.ndarray]:
    reference_cap = reference_layers.cap
    physical_cap = physical_layers.cap
    if reference_cap is None or physical_cap is None:
        raise RuntimeError(
            f"mapped_roots.caps must include physical and reference caps; got {(physical_cap, reference_cap)!r}"
        )
    if not np.array_equal(
        reference_cap.vertex_global_ids, physical_cap.vertex_global_ids
    ) or any(
        not np.array_equal(left.vertices, right.vertices)
        for left, right in zip(reference_cap.blocks, physical_cap.blocks, strict=True)
    ):
        raise RuntimeError(
            f"physical_layers.cap_registry must equal the reference cap registry; got {physical_cap.mesh_id!r} for {reference_cap.mesh_id!r}"
        )
    profile_points, profile_ids = _profile_points(reference_layers, reference_wall)
    cap_rows = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in reference_cap.blocks]
    )
    profile = CellMesh(
        profile_points,
        (
            CellBlock(
                "fundamental-profile",
                "triangle",
                cap_rows,
                global_ids=np.arange(400, 400 + cap_rows.shape[0], dtype=np.int64),
            ),
        ),
        vertex_global_ids=profile_ids,
    )
    lower, upper, coefficient_orbits = _profile_geometry(profile)
    count = reference_cap.coordinates.shape[0]
    upper_reference = profile_points.copy()
    upper_reference[:, 2] = CURVED_PERIODIC_PROFILE_GAP
    layer_count = reference_layers.mesh.coordinates.shape[0]
    layer_ids = np.asarray(reference_layers.mesh.vertex_global_ids, dtype=np.int64)
    upper_ids = int(np.max(layer_ids)) + 1 + np.arange(count, dtype=np.int64)
    layer_cell_ids = np.asarray(
        reference_layers.mesh.entity_set(3).entity_ids, dtype=np.int64
    )
    core_ids = (
        int(np.max(layer_cell_ids)) + 1 + np.arange(cap_rows.shape[0], dtype=np.int64)
    )
    root_cells = np.concatenate(
        (
            np.asarray(reference_layers.cap_vertices, dtype=np.int64)[cap_rows],
            layer_count + cap_rows,
        ),
        axis=1,
    )
    plain = CellMesh(
        np.concatenate((np.asarray(reference_layers.mesh.coordinates), upper_reference)),
        (
            *reference_layers.mesh.blocks,
            CellBlock("declared-core-roots", "prism", root_cells, global_ids=core_ids),
        ),
        vertex_global_ids=np.concatenate((layer_ids, upper_ids)),
    )
    layer_periodic = reference_layers.mesh.periodic_topology
    cap_periodic = reference_cap.periodic_topology
    if layer_periodic is None or cap_periodic is None:
        raise RuntimeError(
            f"mapped_roots.periodic_topology must be present for layers and cap; got {(layer_periodic, cap_periodic)!r}"
        )
    topology = PeriodicMeshTopology(
        plain,
        layer_periodic.cell,
        np.concatenate(
            (
                np.asarray(layer_periodic.vertex_representatives, dtype=np.int64),
                layer_count
                + np.asarray(cap_periodic.vertex_representatives, dtype=np.int64),
            )
        ),
        np.concatenate(
            (
                np.asarray(layer_periodic.vertex_shifts, dtype=np.int64),
                np.asarray(cap_periodic.vertex_shifts, dtype=np.int64),
            )
        ),
    )
    root = CellMesh(
        plain.coordinates,
        plain.blocks,
        vertex_global_ids=plain.vertex_global_ids,
        periodic_topology=topology,
    )
    geometry = _layer_curving.prepare_layer_column_geometry(
        root,
        physical_layers,
        profile,
        lower,
        upper,
        upper_ids,
        fiber_graph=True,
        budget=budget,
    )
    return root, geometry, layer_cell_ids, coefficient_orbits, profile_points


def _source_facet_ids(
    domain: PiecewiseLinearDomain,
    mapping: np.ndarray,
    polygons: np.ndarray,
    layout: _CoreLayout,
    /,
) -> np.ndarray:
    lookup = _facet_lookup(domain)
    cap = set(layout.cap_polygon_ids.tolist())
    values = []
    for index in range(layout.boundary_count):
        if index in cap:
            values.append(-1)
        else:
            key = tuple(sorted(mapping[polygons[index]].tolist()))
            values.append(lookup[key])
    return np.asarray(values, dtype=np.int64)


def _physical_source(
    physical_layers: BoundaryLayerMesh,
    reference_source: NativeLayerCoreSource,
    reference_domain: PiecewiseLinearDomain,
    domain: PiecewiseLinearDomain,
    mapping: np.ndarray,
    polygons: np.ndarray,
    layout: _CoreLayout,
    root: CellMesh,
    geometry: CellGeometrySpec,
    layer_cell_ids: np.ndarray,
    profile_points: np.ndarray,
    coefficient_orbits: tuple[str, ...],
    lower_facet_ids: np.ndarray,
    /,
) -> tuple[NativeLayerCoreSource, tuple[str, ...]]:
    cap = physical_layers.cap
    if cap is None:
        raise RuntimeError("physical_source.cap must be present; got None")
    upper = profile_points.copy()
    upper[:, 2] = CURVED_PERIODIC_PROFILE_GAP
    physical_complex = _complex(
        np.concatenate((np.asarray(cap.coordinates), upper)), layout, reference=False
    )
    source_faces = _source_facet_ids(domain, mapping, polygons, layout)
    boundary_ids = np.unique(
        np.concatenate((lower_facet_ids, source_faces[source_faces >= 0]))
    )
    boundary = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        f"{domain.domain_id}:facets",
        boundary_ids,
    )
    mapped = MappedReferenceDomain(
        reference_domain,
        root,
        geometry,
        np.concatenate(
            (
                np.zeros(layer_cell_ids.size, dtype=np.int64),
                np.ones(layout.cap_polygon_ids.size, dtype=np.int64),
            )
        ),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
    )
    query = MappedDomainBoundarySource(mapped, mapped.cell_regions)
    source = NativeLayerCoreSource(
        physical_layers,
        physical_complex,
        domain.source_id,
        domain.source_revision,
        vertex_layer_ids=np.concatenate(
            (
                np.asarray(physical_layers.cap_vertices, dtype=np.int64),
                np.full(cap.coordinates.shape[0], -1, dtype=np.int64),
            )
        ),
        cap_polygon_ids=layout.cap_polygon_ids,
        layer_regions=np.zeros(layer_cell_ids.size, dtype=np.int64),
        region_ids=("wall-material", "core-material"),
        core_region_map=np.asarray((1,), dtype=np.int64),
        source_boundary_scope=boundary,
        source_domain=domain,
        fidelity_source=query,
        core_facet_source_ids=source_faces,
        core_vertex_representatives=layout.representatives,
        core_vertex_shifts=layout.shifts,
        core_seam_polygon_pairs=layout.seam_pairs,
        reference_source=reference_source,
        mapped_domain=mapped,
    )
    return source, coefficient_orbits


def _specification(
    source: NativeLayerCoreSource,
    limits: M.MeshingLimits,
    /,
) -> M.VolumeMeshingSpec:
    boundary = source.source_boundary_scope
    pairs = source.core_seam_polygon_pairs
    source_faces = source.core_facet_source_ids
    domain = source.source_domain
    if (
        boundary is None
        or pairs is None
        or source_faces is None
        or not isinstance(domain, PiecewiseLinearDomain)
    ):
        raise RuntimeError(
            f"periodic_specification.source_ancestry must include boundary scope, seam pairs, facet IDs, and a PiecewiseLinearDomain; got {(boundary, pairs, source_faces, type(domain).__name__)!r}"
        )
    first = source_faces[pairs[:, 0]]
    second = source_faces[pairs[:, 1]]
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = CURVED_PERIODIC_LAYER_PERIOD
    source_scope = M.MeshingScope(
        boundary.source_id,
        boundary.source_revision,
        boundary.entity_kind,
        2,
        boundary.entity_set_id,
        first,
    )
    target_scope = M.MeshingScope(
        boundary.source_id,
        boundary.source_revision,
        boundary.entity_kind,
        2,
        boundary.entity_set_id,
        second,
    )
    periodic = M.PeriodicConstraint(
        source_scope,
        target_scope,
        transform,
        tolerance=0.0,
        source_entity_ids=first,
        orientations=np.full(first.size, -1, dtype=np.int64),
    )
    regions = tuple(
        M.RegionControl(
            M.MeshingScope(
                boundary.source_id,
                boundary.source_revision,
                boundary.entity_kind,
                3,
                f"{domain.domain_id}:regions",
                np.asarray((index,), dtype=np.int64),
            ),
            name,
            f"material:{index}",
            M.RegionRole.FLUID,
        )
        for index, name in enumerate(source.region_ids)
    )
    return M.VolumeMeshingSpec(
        M.CellMeshingTarget(
            3,
            3,
            M.CellFamilyPolicy(
                required=("prism", "tetrahedron"),
                allowed_transitions=("hexahedron", "pyramid"),
                allow_mixed=True,
            ),
        ),
        boundary,
        M.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            M.UniformSizeControl(boundary, 2.0, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=regions,
        layer_controls=(source.layers.control,),
        periodic_constraints=(periodic,),
        limits=limits,
    )


@checked
def prepare_corrected_periodic_layer_source(
    *,
    limits: M.MeshingLimits | None = None,
) -> AuthoredPeriodicLayerSource:
    """Author the corrected curved ADVANCING layer/core source end to end."""
    limits_ = M.MeshingLimits() if limits is None else limits
    ticket = HostBatchTicket(
        limits_.maximum_work_units,
        bytes_upper=limits_.maximum_scratch_bytes,
    )
    work = LayerCoreSourceWork(limits_.maximum_work_units, ticket=ticket)
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=limits_.maximum_work_units,
        maximum_memory_bytes=limits_.maximum_scratch_bytes,
    )
    with ticket, budget.activate():
        return _prepare_corrected_periodic_layer_source(limits_, work=work, budget=budget)


def _prepare_corrected_periodic_layer_source(
    limits_: M.MeshingLimits,
    /,
    *,
    work: LayerCoreSourceWork,
    budget: CoordinateEnclosureBudget,
) -> AuthoredPeriodicLayerSource:
    schedule = M.LayerSchedule.geometric(
        2, CURVED_PERIODIC_LAYER_FIRST_THICKNESS, growth_rate=1.0
    )
    reference_wall, _ = _translated_wall(0.0)
    reference_layers = _prepare_reference_layers(reference_wall, schedule)
    reference_profile, _ = _profile_points(reference_layers, reference_wall)
    core_mesh, cap_rows = _root_core_mesh(reference_layers, reference_profile)
    reference_cap = reference_layers.cap
    if reference_cap is None:
        raise RuntimeError("reference_source.cap must be present; got None")
    layout = _core_layout(
        core_mesh, reference_cap, cap_rows, reference_profile, work=work
    )
    reference_source, reference_domain, mapping, polygons = _reference_source(
        reference_layers, core_mesh, layout, work=work
    )
    domain = _physical_domain(reference_domain)
    wall, wall_orbits = _translated_wall(CORRECTED_CURVED_PERIODIC_PROFILE_HEIGHT)
    control, association, lower_facets = _physical_control(
        wall, reference_layers, domain, schedule
    )
    physical_layers = _physical_layers(wall, control, association, domain)
    if (
        not np.array_equal(
            physical_layers.mesh.vertex_global_ids,
            reference_layers.mesh.vertex_global_ids,
        )
        or not np.array_equal(physical_layers.layer_index, reference_layers.layer_index)
        or not np.array_equal(physical_layers.column_index, reference_layers.column_index)
    ):
        raise RuntimeError(
            f"physical_layers.column_registry must equal the reference vertex, layer, and column registry; got {(physical_layers.mesh.vertex_global_ids, physical_layers.layer_index, physical_layers.column_index)!r}"
        )
    root, geometry, layer_cell_ids, coefficient_orbits, profile_points = (
        _root_mesh_and_geometry(
            reference_layers, physical_layers, reference_wall, layout, budget=budget
        )
    )
    source, coefficient_orbits = _physical_source(
        physical_layers,
        reference_source,
        reference_domain,
        domain,
        mapping,
        polygons,
        layout,
        root,
        geometry,
        layer_cell_ids,
        profile_points,
        coefficient_orbits,
        lower_facets,
    )
    specification = _specification(source, limits_)
    mapped = source.mapped_domain
    if mapped is None:
        raise RuntimeError("corrected_source.mapped_domain must be present; got None")
    digest = canonical_fingerprint(
        {
            "kind": "corrected-curved-periodic-layer-source",
            "source_id": source.source_id,
            "source_revision": source.source_revision,
            "binding_id": source.binding_id,
            "mapped_domain_id": mapped.domain_id,
            "profile_height": CORRECTED_CURVED_PERIODIC_PROFILE_HEIGHT,
            "gap": CURVED_PERIODIC_PROFILE_GAP,
            "period": CURVED_PERIODIC_LAYER_PERIOD,
            "wall_orbits": wall_orbits,
            "coefficient_orbits": coefficient_orbits,
            "fiber_graph": True,
            "coefficients": array_tree_fingerprint(mapped.source_geometry),
        }
    )
    return AuthoredPeriodicLayerSource(
        domain,
        wall,
        association,
        source,
        specification,
        M.NativeMeshingOptions("layer_core"),
        digest,
        wall_orbits,
        coefficient_orbits,
    )


@checked
def periodic_layer_source_identity_record(
    authored: AuthoredPeriodicLayerSource, /
) -> dict[str, str | float | bool | None | list[str] | list[float]]:
    """Return consumer-facing identity of one constructed corrected source."""
    mapped = authored.source.mapped_domain
    fiber_graph = mapped is not None
    if mapped is not None:
        for element in mapped.source_geometry.elements:
            match element:
                case LayerColumnCellGeometryElement(fiber_graph=True):
                    pass
                case _:
                    fiber_graph = False
                    break
    return {
        "source_id": authored.source_id,
        "source_revision": authored.source_revision,
        "source_digest": authored.source_digest,
        "source_binding_id": authored.source.binding_id,
        "mapped_domain_id": None if mapped is None else mapped.domain_id,
        "specification_id": authored.specification.specification_id,
        "wall_orbit_ids": list(authored.wall_orbit_ids),
        "coefficient_orbit_ids": list(authored.coefficient_orbit_ids),
        "fiber_graph": fiber_graph,
        "period": CURVED_PERIODIC_LAYER_PERIOD.tolist(),
        "fundamental_profile_height": CORRECTED_CURVED_PERIODIC_PROFILE_HEIGHT,
        "gap": CURVED_PERIODIC_PROFILE_GAP,
    }
