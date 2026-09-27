#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh layered volumes: advancing or provider layers around a fixed-cap core fill.

The BRep boundary is surface-meshed first. ADVANCING layers grow natively from
the wall faces; PROVIDER layers are Gmsh boundary-layer extrusions of the same
wall mesh, certified afterwards by the native validity and exact intersection
tests. Either way the cap and the remaining boundary are handed to Gmsh as
fixed discrete surfaces: the core may neither move nor split them, which is
verified bitwise and by exact face conformity before the meshes merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..._fingerprint import canonical_fingerprint
from ..._identity import SemanticProvenance
from ...discretization import CellBlock, CellGeometrySpec, CellMesh
from ...geometry.surface import SurfaceMetadata, SurfaceModel
from ...geometry.surface._model import _repair_orientations
from .._audit import audit_cell_mesh
from .._boundary_layer import (
    _analyze_wall,
    _certify_cells,
    _Environment,
    _grow_boundary_layers,
    _nearest_distances,
    _split_ridges,
    BoundaryLayerMesh,
    BoundaryLayerPolicy,
)
from .._canonical import canonicalize_cell_mesh
from .._contracts import (
    MeshingDerivativeMode,
    MeshingExecutionMode,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingProviderInfo,
    VolumeMeshingSpec,
)
from .._controls import (
    BoundaryLayerControl,
    BoundaryLayerCornerPolicy,
    BoundaryLayerRoute,
)
from .._organization import (
    MeshAttribute,
    MeshAttributeRole,
    MeshPatch,
    MeshZone,
    MeshZoneRole,
)
from .._quality import evaluate_cell_quality
from .._result import CellMeshingResult, MeshingComplianceReport, MeshingRuntimeInfo
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import (
    MeshingStageKind,
    MeshingStageReport,
    MeshingStageStatus,
    MeshingTrace,
)
from ._gmsh_evidence import (
    _boundary_association,
    _connectivity_face_incidents,
    _connectivity_face_rows,
    _patch_is_connected,
)
from ._gmsh_execute import _prepare_generation
from ._gmsh_import import _brep_model, _entity_scope, _resolve_entities


_VOLUME_ORDER = (
    ("tetrahedron", "tetrahedra"),
    ("pyramid", "pyramids"),
    ("prism", "prisms"),
    ("hexahedron", "hexahedra"),
)
_ARITY = {"tetrahedron": 4, "pyramid": 5, "prism": 6, "hexahedron": 8}
# Relative agreement required between measured and requested layer thickness;
# measurements are exact distances to the discrete wall, so curved walls carry
# a discretization-level deviation.
_LAYER_TOLERANCE = 1.0e-2
_LAYER_STAGE = MeshingStageKind.LAYER_GENERATION.value
_FILL_STAGE = MeshingStageKind.VOLUME_FILL.value


def _layered_volume(plan: Any, /) -> bool:
    """Whether a plan realizes ADVANCING or PROVIDER layers around a core fill."""
    specification = plan.specification
    return isinstance(specification, VolumeMeshingSpec) and any(
        control.route in (BoundaryLayerRoute.ADVANCING, BoundaryLayerRoute.PROVIDER)
        for control in specification.layer_controls
    )


# ---------------------------------------------------------------- boundary surface


@dataclass(frozen=True, slots=True)
class _Boundary:
    """Closed boundary surface oriented along the growth direction (into the domain)."""

    points: np.ndarray
    triangles: np.ndarray
    source_faces: np.ndarray
    wall: np.ndarray


def _domain_inward(
    points: np.ndarray, triangles: np.ndarray, /
) -> tuple[np.ndarray, float]:
    """Orient closed shells so every normal points into the enclosed domain."""
    repaired, repair = _repair_orientations(points, triangles, orient_closed_outward=True)
    components = np.asarray(repair.component_ids)
    corners = points[repaired]
    signed = np.sum(corners[:, 0] * np.cross(corners[:, 1], corners[:, 2]), axis=1) / 6.0
    volumes = np.bincount(components, signed)
    outer = int(np.argmax(volumes))
    # Each shell now faces away from itself; holes face into the domain, the
    # exterior shell out of it. Growth runs into the domain.
    flip = components == outer
    oriented = np.where(flip[:, None], repaired[:, (0, 2, 1)], repaired)
    return oriented, float(volumes[outer] - np.sum(np.delete(volumes, outer)))


def _surface_boundary(
    gmsh: Any, plan: Any, generation: Any, control: BoundaryLayerControl, /
) -> _Boundary:
    source = _brep_model(plan.source)
    node_tags, node_coordinates, _ = gmsh.model.mesh.getNodes()
    node_tags = np.asarray(node_tags, dtype=np.int64)
    order = np.argsort(node_tags, kind="stable")
    node_tags = node_tags[order]
    points = np.asarray(node_coordinates, dtype=np.float64).reshape((-1, 3))[order]
    triangle_type = gmsh.model.mesh.getElementType("Triangle", 1)
    rows = []
    owners = []
    for face in range(source.report.num_faces):
        surface = _resolve_entities(
            gmsh, source, generation.shape, _entity_scope(source, 2, (face,))
        )[0]
        element_types, _, node_blocks = gmsh.model.mesh.getElements(2, surface)
        if tuple(int(value) for value in element_types) != (triangle_type,):
            raise MeshingFailure(
                MeshingFailureCategory.CONVERSION_FAILED,
                "Layered volumes require linear triangle boundary meshes.",
                stage=MeshingStageKind.SURFACE_MESHING.value,
            )
        nodes = np.asarray(node_blocks[0], dtype=np.int64).reshape(-1, 3)
        rows.append(np.searchsorted(node_tags, nodes))
        owners.append(np.full((nodes.shape[0],), face, dtype=np.int64))
    triangles = np.concatenate(rows)
    source_faces = np.concatenate(owners)
    used = np.unique(triangles)
    remap = np.full((points.shape[0],), -1, dtype=np.int64)
    remap[used] = np.arange(used.size)
    points = points[used]
    oriented, volume = _domain_inward(points, remap[triangles])
    cad_volume = sum(
        gmsh.model.occ.getMass(3, tag) for _, tag in gmsh.model.getEntities(3)
    )
    if not volume > 0.0 or abs(volume - cad_volume) > 0.05 * cad_volume:
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            "The discrete boundary does not enclose the CAD volume with one exterior shell.",
            stage=MeshingStageKind.SURFACE_MESHING.value,
        )
    wall_faces = np.asarray(control.wall_scope.entity_ids, dtype=np.int64)
    return _Boundary(points, oriented, source_faces, np.isin(source_faces, wall_faces))


# ---------------------------------------------------------------- core fill


def _components(triangles: np.ndarray, count: int, /) -> np.ndarray:
    rows = np.repeat(np.arange(triangles.shape[0]), 3)
    graph = coo_matrix(
        (
            np.ones(rows.size, dtype=np.int8),
            (rows, triangles.reshape(-1) + triangles.shape[0]),
        ),
        shape=(triangles.shape[0] + count, triangles.shape[0] + count),
    )
    _, labels = connected_components(graph, directed=False)
    _, relabeled = np.unique(labels[: triangles.shape[0]], return_inverse=True)
    return relabeled.reshape(-1)


def _exterior_first(
    points: np.ndarray, triangles: np.ndarray, labels: np.ndarray, /
) -> Any:
    count = int(labels.max()) + 1
    diagonals = np.empty((count,))
    for label in range(count):
        corners = points[triangles[labels == label]].reshape(-1, 3)
        diagonals[label] = np.linalg.norm(np.ptp(corners, axis=0))
    exterior = int(np.argmax(diagonals))
    return (exterior, *(label for label in range(count) if label != exterior))


def _face_keys(faces: np.ndarray, /) -> np.ndarray:
    return np.sort(faces, axis=1)


def _configure_core(gmsh: Any, options: Any, maximum_size: float | None, /) -> None:
    gmsh.clear()
    gmsh.option.setNumber("General.Terminal", 1 if options.terminal_output else 0)
    gmsh.option.setNumber("General.NumThreads", options.num_threads)
    gmsh.option.setNumber("Mesh.Algorithm3D", options.algorithm_3d.gmsh_code)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.0)
    gmsh.option.setNumber(
        "Mesh.MeshSizeMax", 1.0e22 if maximum_size is None else maximum_size
    )
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 1)
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 0)
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 0)
    # Fixed nodes are identified by tag; Gmsh must not renumber them.
    gmsh.option.setNumber("Mesh.Renumber", 0)
    gmsh.model.add("phydrax-layered-core")


def _discrete_shells(gmsh: Any, points: np.ndarray, shells: Any, /) -> tuple[int, ...]:
    """One discrete surface per closed shell; node tags are vertex index + 1."""
    triangle_type = gmsh.model.mesh.getElementType("Triangle", 1)
    assigned = np.zeros((points.shape[0],), dtype=np.bool_)
    surfaces = []
    for triangles in shells:
        surface = gmsh.model.addDiscreteEntity(2)
        vertices = np.unique(triangles)
        own = vertices[~assigned[vertices]]
        assigned[own] = True
        gmsh.model.mesh.addNodes(2, surface, own + 1, points[own].reshape(-1))
        gmsh.model.mesh.addElementsByType(
            surface, triangle_type, [], (triangles + 1).reshape(-1)
        )
        surfaces.append(surface)
    return tuple(surfaces)


def _extract_volume(gmsh: Any, volume: int, name: str, corners: int, /) -> np.ndarray:
    element_types, _, node_blocks = gmsh.model.mesh.getElements(3, volume)
    expected = gmsh.model.mesh.getElementType(name, 1)
    if tuple(int(value) for value in element_types) != (expected,):
        raise MeshingFailure(
            MeshingFailureCategory.CONVERSION_FAILED,
            f"Gmsh returned elements other than linear {name.lower()} cells.",
            stage=_FILL_STAGE,
        )
    return np.asarray(node_blocks[0], dtype=np.int64).reshape(-1, corners)


def _fixed_nodes(gmsh: Any, points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Verify fixed nodes bitwise and return appended interior node coordinates."""
    tags, coordinates, _ = gmsh.model.mesh.getNodes()
    tags = np.asarray(tags, dtype=np.int64)
    order = np.argsort(tags, kind="stable")
    tags = tags[order]
    values = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)[order]
    count = points.shape[0]
    fixed = tags <= count
    if not np.array_equal(tags[fixed], np.arange(1, count + 1)) or not np.array_equal(
        values[fixed], points
    ):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "The core provider moved or dropped fixed cap/boundary nodes.",
            stage=_FILL_STAGE,
        )
    return tags, values[~fixed]


def _node_index(tags: np.ndarray, count: int, rows: np.ndarray, /) -> np.ndarray:
    """Fixed node tag t maps to vertex t - 1; appended nodes follow in tag order."""
    appended = tags[tags > count]
    return np.where(rows <= count, rows - 1, count + np.searchsorted(appended, rows))


def _require_conforming(tetrahedra: np.ndarray, fixed: np.ndarray, /) -> None:
    faces = tetrahedra[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape(-1, 3)
    keys, counts = np.unique(_face_keys(faces), axis=0, return_counts=True)
    boundary = keys[counts == 1]
    expected = np.unique(_face_keys(fixed), axis=0)
    if np.any(counts > 2) or not np.array_equal(boundary, expected):
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "Core cells do not conform exactly to the fixed cap and boundary triangles.",
            stage=_FILL_STAGE,
        )


def _fill_core(
    gmsh: Any,
    points: np.ndarray,
    fixed: np.ndarray,
    options: Any,
    maximum_size: float | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Tetrahedralize the domain bounded by fixed triangles without touching them.

    Returns the appended interior points and tetrahedra indexing ``points``
    followed by those interior points.
    """
    used = np.unique(fixed)
    local = np.searchsorted(used, fixed)
    anchors = points[used]
    _configure_core(gmsh, options, maximum_size)
    labels = _components(local, used.size)
    order = _exterior_first(anchors, local, labels)
    surfaces = _discrete_shells(
        gmsh, anchors, tuple(local[labels == label] for label in order)
    )
    loops = [gmsh.model.geo.addSurfaceLoop([surface]) for surface in surfaces]
    volume = gmsh.model.geo.addVolume(loops)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    tags, interior = _fixed_nodes(gmsh, anchors)
    triangle_type = gmsh.model.mesh.getElementType("Triangle", 1)
    for surface, label in zip(surfaces, order, strict=True):
        element_types, _, node_blocks = gmsh.model.mesh.getElements(2, surface)
        kept = np.asarray(node_blocks[0], dtype=np.int64).reshape(-1, 3) - 1
        if tuple(int(value) for value in element_types) != (
            triangle_type,
        ) or not np.array_equal(
            np.unique(_face_keys(kept), axis=0),
            np.unique(_face_keys(local[labels == label]), axis=0),
        ):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                "The core provider altered a fixed boundary surface mesh.",
                stage=_FILL_STAGE,
            )
    cells = _node_index(tags, used.size, _extract_volume(gmsh, volume, "Tetrahedron", 4))
    tetrahedra = np.where(
        cells < used.size,
        used[np.minimum(cells, used.size - 1)],
        points.shape[0] + cells - used.size,
    )
    _require_conforming(tetrahedra, fixed)
    return interior, tetrahedra


# ---------------------------------------------------------------- layer routes


@dataclass(frozen=True, slots=True)
class _Layered:
    """Layer cells, core tetrahedra, and boundary triangles in one vertex space."""

    points: np.ndarray
    layer_cells: dict[str, np.ndarray]
    layer_index: np.ndarray
    core: np.ndarray
    wall: np.ndarray
    outer: np.ndarray
    cap: np.ndarray
    outer_faces: np.ndarray
    wall_faces: np.ndarray
    thicknesses: np.ndarray
    minimum_scale: float
    counts: tuple[tuple[str, float], ...]


def _padded(triangles: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.pad(triangles, ((0, 0), (0, 1)), constant_values=-1),
        np.full((triangles.shape[0],), 3, dtype=np.int64),
    )


def _advancing_layers(
    gmsh: Any, boundary: _Boundary, control: Any, options: Any, maximum_size: Any, /
) -> _Layered:
    faces, arity = _padded(boundary.triangles)
    empty_points = np.empty((0, 3))
    empty_triangles = np.empty((0, 3), dtype=np.int64)
    layers = _grow_boundary_layers(
        boundary.points,
        faces,
        arity,
        boundary.wall,
        empty_points,
        empty_triangles,
        control,
        BoundaryLayerPolicy(),
    )
    return _merge_advancing(gmsh, layers, boundary, options, maximum_size)


def _merge_advancing(
    gmsh: Any,
    layers: BoundaryLayerMesh,
    boundary: _Boundary,
    options: Any,
    maximum_size: Any,
    /,
) -> Any:
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    wall_map = np.asarray(layers.wall_vertices, dtype=np.int64)
    outer = boundary.triangles[~boundary.wall]
    extra = np.setdiff1d(np.unique(outer), np.flatnonzero(wall_map >= 0))
    mapping = wall_map.copy()
    mapping[extra] = layer_points.shape[0] + np.arange(extra.size)
    points = np.concatenate((layer_points, boundary.points[extra]))
    cap = _cap_triangles(layers)
    fixed = np.concatenate((cap, mapping[outer]))
    interior, core = _fill_core(gmsh, points, fixed, options, maximum_size)
    evidence = layers.evidence
    cells = {
        block.cell_kind: np.asarray(block.vertices, dtype=np.int64)
        for block in layers.mesh.blocks
    }
    return _Layered(
        np.concatenate((points, interior)),
        cells,
        np.asarray(layers.layer_index, dtype=np.int32),
        core,
        mapping[boundary.triangles[boundary.wall]],
        mapping[outer],
        cap,
        boundary.source_faces[~boundary.wall],
        boundary.source_faces[boundary.wall],
        np.asarray(evidence.achieved_thicknesses, dtype=np.float64),
        evidence.minimum_scale,
        (
            ("fan_column_count", float(evidence.fan_column_count)),
            ("corner_patch_count", float(evidence.corner_patch_count)),
            ("detected_collision_count", float(evidence.detected_collision_count)),
            ("certified_valid_layer_cells", float(evidence.certified_valid_count)),
        ),
    )


def _cap_triangles(layers: BoundaryLayerMesh, /) -> np.ndarray:
    if not layers.closed_cap:
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Core fill requires a closed layer cap; open wall rims need EXACT_SWEEP.",
            stage=_FILL_STAGE,
        )
    # ty: ignore[unresolved-attribute]
    if any(block.cell_kind != "triangle" for block in layers.cap.blocks):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "A simplex core requires a triangle layer cap (enable transition pyramids).",
            stage=_FILL_STAGE,
        )
    cap_vertices = np.asarray(layers.cap_vertices, dtype=np.int64)
    # ty: ignore[unresolved-attribute]
    blocks = [np.asarray(block.vertices, dtype=np.int64) for block in layers.cap.blocks]
    return cap_vertices[np.concatenate(blocks)]


def _provider_layers(
    gmsh: Any, boundary: _Boundary, control: Any, options: Any, maximum_size: Any, /
) -> _Layered:
    """Gmsh boundary-layer extrusion of the wall mesh, certified natively."""
    faces, arity = _padded(boundary.triangles)
    if control.corner is BoundaryLayerCornerPolicy.REJECT:
        _split_ridges(
            _analyze_wall(
                boundary.points, faces, arity, boundary.wall, control.feature_angle
            ),
            control.corner,
        )
    _configure_core(gmsh, options, maximum_size)
    count = boundary.points.shape[0]
    wall_triangles = boundary.triangles[boundary.wall]
    outer = boundary.triangles[~boundary.wall]
    labels = _components(boundary.triangles, count)
    order = _exterior_first(boundary.points, boundary.triangles, labels)
    shells = tuple(boundary.triangles[labels == label] for label in order)
    surfaces = _discrete_shells(gmsh, boundary.points, shells)
    heights = np.cumsum(np.asarray(control.schedule.thicknesses, dtype=np.float64))
    loops = []
    layer_volumes = []
    for surface, label in zip(surfaces, order, strict=True):
        if np.all(boundary.wall[labels == label]):
            # Recombination keeps each extruded triangle column as prisms.
            extruded = gmsh.model.geo.extrudeBoundaryLayer(
                [(2, surface)], [1] * heights.size, heights.tolist(), True
            )
            tops = [tag for dimension, tag in extruded if dimension == 2]
            layer_volumes.extend(tag for dimension, tag in extruded if dimension == 3)
            loops.append(gmsh.model.geo.addSurfaceLoop(tops))
        else:
            loops.append(gmsh.model.geo.addSurfaceLoop([surface]))
    core_volume = gmsh.model.geo.addVolume(loops)
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(3)
    tags, interior = _fixed_nodes(gmsh, boundary.points)
    points = np.concatenate((boundary.points, interior))
    prisms = np.concatenate(
        [
            _node_index(tags, count, _extract_volume(gmsh, volume, "Prism", 6))
            for volume in layer_volumes
        ]
    )
    core = _node_index(tags, count, _extract_volume(gmsh, core_volume, "Tetrahedron", 4))
    corners = points[prisms]
    negative = (
        np.sum(
            np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
            * (corners[:, 3] - corners[:, 0]),
            axis=1,
        )
        < 0.0
    )
    prisms[negative] = prisms[negative][:, (0, 2, 1, 3, 5, 4)]
    cells = {
        name: np.empty((0, _ARITY[name]), dtype=np.int64) for name, _ in _VOLUME_ORDER
    }
    cells["prism"] = prisms
    environment = _Environment(
        boundary.triangles,
        np.zeros((boundary.triangles.shape[0],), dtype=np.bool_),
        np.empty((0, 3)),
        np.empty((0, 3), dtype=np.int64),
    )
    bad, hits = _certify_cells(points, cells, [], environment, BoundaryLayerPolicy())
    if np.any(bad["prism"]):
        vertices = prisms[bad["prism"]][:, :3].reshape(-1)
        vertices = vertices[vertices < count]
        raise MeshingFailure(
            MeshingFailureCategory.CONTROL_CONFLICT,
            f"Gmsh boundary-layer prisms collide or are invalid ({hits} certified intersections).",
            stage=_LAYER_STAGE,
            entity_ids=tuple(int(value) for value in np.unique(vertices)),
            locations=tuple(
                tuple(float(x) for x in points[value]) for value in np.unique(vertices)
            ),
        )
    level = _prism_levels(prisms, count, heights.size, points.shape[0])
    cap_faces = _top_faces(prisms, level, heights.size)
    _require_conforming(core, np.concatenate((cap_faces, outer)))
    thicknesses = _level_thicknesses(points, wall_triangles, level, heights.size)
    return _Layered(
        points,
        cells,
        level[prisms[:, 0]].astype(np.int32),
        core,
        wall_triangles,
        outer,
        cap_faces,
        boundary.source_faces[~boundary.wall],
        boundary.source_faces[boundary.wall],
        thicknesses,
        1.0,
        (),
    )


def _prism_levels(
    prisms: np.ndarray, count: int, layers: int, total: int, /
) -> np.ndarray:
    """Layer level of every vertex, following prism side edges up from the wall."""
    level = np.full((total,), -1, dtype=np.int64)
    level[:count] = 0
    for _ in range(layers):
        for corner in range(3):
            below = level[prisms[:, corner]]
            above = prisms[:, corner + 3]
            level[above] = np.where(below >= 0, below + 1, level[above])
    return level


def _top_faces(prisms: np.ndarray, level: np.ndarray, layers: int, /) -> np.ndarray:
    """Outermost prism tops; prism tops face away from the layer."""
    top = prisms[:, 3:]
    return top[np.all(level[top] == layers, axis=1)]


def _level_thicknesses(
    points: Any, wall_triangles: Any, level: Any, layers: Any, /
) -> np.ndarray:
    selected = np.flatnonzero(level > 0)
    distance = _nearest_distances(points[wall_triangles], points[selected])
    means = np.asarray(
        [np.mean(distance[level[selected] == value]) for value in range(1, layers + 1)]
    )
    return np.diff(np.concatenate(([0.0], means)))


# ---------------------------------------------------------------- merged mesh


def _merged_mesh(layered: _Layered, numeric_version: str, /) -> Any:
    """Merged cells in canonical (block-name) order with zone mask and layer indices."""
    # ``layer_index`` follows the layer mesh's kind order (_VOLUME_ORDER).
    offsets = {}
    start = 0
    for kind, _ in _VOLUME_ORDER:
        count = layered.layer_cells.get(kind, np.empty((0,))).shape[0]
        offsets[kind] = (start, start + count)
        start += count
    blocks = []
    zone_of = []
    layer_values = []
    cursor = 0
    for kind, name in sorted(_VOLUME_ORDER, key=lambda value: value[1]):
        empty = np.empty((0, _ARITY[kind]), dtype=np.int64)
        layer_rows = layered.layer_cells.get(kind, empty)
        core_rows = layered.core if kind == "tetrahedron" else empty
        rows = np.concatenate((layer_rows, core_rows))
        if not rows.size:
            continue
        blocks.append(
            CellBlock(
                name,
                kind,
                rows,
                global_ids=np.arange(cursor, cursor + rows.shape[0], dtype=np.int64),
            )
        )
        zone_of.append(
            np.concatenate(
                (
                    np.zeros(layer_rows.shape[0], dtype=np.bool_),
                    np.ones(core_rows.shape[0], dtype=np.bool_),
                )
            )
        )
        lower, upper = offsets[kind]
        layer_values.append(layered.layer_index[lower:upper])
        cursor += rows.shape[0]
    mesh = canonicalize_cell_mesh(
        CellMesh(layered.points, tuple(blocks), numeric_version=numeric_version)
    )
    return mesh, np.concatenate(zone_of), np.concatenate(layer_values)


def _organization(
    mesh: CellMesh, core: np.ndarray, layer_values: np.ndarray, layered: _Layered, /
) -> Any:
    cells = mesh.entity_set(3)
    cell_ids = np.asarray(cells.entity_ids, dtype=np.int64)

    def cell_scope(selected: Any) -> Any:
        return MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            MeshingEntityKind.MESH,
            3,
            cells.entity_set_id,
            cell_ids[selected],
        )

    layer_zone = MeshZone("boundary-layer", MeshZoneRole.REGION, cell_scope(~core))
    core_zone = MeshZone("core", MeshZoneRole.REGION, cell_scope(core))
    attribute = MeshAttribute(
        "layer_index",
        MeshAttributeRole.MARKER,
        cell_scope(~core),
        layer_values.astype(np.int32),
    )
    connectivity = mesh.connectivity
    # ty: ignore[invalid-argument-type]
    rows = _connectivity_face_rows(connectivity)
    faces = mesh.entity_set(2)
    face_ids = np.asarray(faces.entity_ids, dtype=np.int64)
    padded = np.full((len(rows), 4), np.iinfo(np.int64).max, dtype=np.int64)
    for size in (3, 4):
        selected = np.flatnonzero([len(row) == size for row in rows])
        if selected.size:
            padded[selected, :size] = np.sort(
                np.stack([rows[index] for index in selected]), axis=1
            )
    incidents = np.asarray(
        # ty: ignore[invalid-argument-type]
        [len(adjacent) for adjacent in _connectivity_face_incidents(connectivity)]
    )
    patches = []
    for name, triangles, zones in (
        ("wall", layered.wall, (layer_zone,)),
        ("layer-core-interface", layered.cap, (layer_zone, core_zone)),
        ("outer", layered.outer, (core_zone,)),
    ):
        keys = np.full((triangles.shape[0], 4), np.iinfo(np.int64).max, dtype=np.int64)
        keys[:, :3] = np.sort(triangles, axis=1)
        _, inverse = np.unique(
            np.concatenate((padded, keys)), axis=0, return_inverse=True
        )
        inverse = inverse.reshape(-1)
        position = np.full((inverse.max() + 1,), -1, dtype=np.int64)
        position[inverse[: padded.shape[0]]] = np.arange(padded.shape[0])
        selected = np.unique(position[inverse[padded.shape[0] :]])
        if selected.size and selected[0] < 0:
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Patch {name!r} triangles are not faces of the merged mesh.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        if np.any(incidents[selected] != len(zones)):
            raise MeshingFailure(
                MeshingFailureCategory.COMPLIANCE_FAILED,
                f"Patch {name!r} faces have the wrong cell incidence.",
                stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
            )
        patches.append(
            MeshPatch(
                name,
                MeshingScope(
                    mesh.mesh_id,
                    mesh.numeric_version,
                    MeshingEntityKind.MESH,
                    2,
                    faces.entity_set_id,
                    face_ids[selected],
                ),
                # ty: ignore[invalid-argument-type]
                connected=_patch_is_connected(connectivity, selected),
                adjacent_zone_ids=tuple(zone.zone_id for zone in zones),
            )
        )
    return (layer_zone, core_zone), tuple(patches), attribute


def _layer_compliance(control: BoundaryLayerControl, layered: _Layered, /) -> Any:
    requested_values = np.asarray(control.schedule.thicknesses, dtype=np.float64)
    key = f"layer:{control.control_id}"
    measured = layered.thicknesses
    requested = [
        (f"{key}:layer_count", float(requested_values.size)),
        (f"{key}:first_layer_thickness", float(requested_values[0])),
        *(
            (f"{key}:thickness:{index}", float(value))
            for index, value in enumerate(requested_values)
        ),
    ]
    achieved = [
        (f"{key}:layer_count", float(np.count_nonzero(np.isfinite(measured)))),
        (f"{key}:first_layer_thickness", float(measured[0])),
        *(
            (f"{key}:thickness:{index}", float(value))
            for index, value in enumerate(measured)
        ),
        *(
            (f"{key}:growth:{index}", float(value))
            for index, value in enumerate(measured[1:] / measured[:-1], start=1)
        ),
        (f"{key}:minimum_scale", layered.minimum_scale),
        *((f"{key}:{name}", value) for name, value in layered.counts),
        ("layer_interface_compliance", 1.0),
        ("fixed_boundary_bitwise", 1.0),
    ]
    issues = []
    # Collision policies may reduce or terminate layers; those are reported, not failures.
    if layered.minimum_scale >= 1.0 and not np.allclose(
        measured, requested_values, rtol=_LAYER_TOLERANCE, atol=0.0
    ):
        issues.append(f"layer_thickness:{control.control_id}")
    return tuple(requested), tuple(achieved), tuple(issues)


def _boundary_size_compliance(specification: Any, layered: _Layered, /) -> Any:
    """Whole-source uniform sizes govern the wall and remaining boundary surface."""
    triangles = np.concatenate((layered.wall, layered.outer))
    edges = np.unique(
        np.sort(triangles[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0
    )
    lengths = np.linalg.norm(
        layered.points[edges[:, 0]] - layered.points[edges[:, 1]], axis=1
    )
    shortest = np.full((layered.points.shape[0],), np.inf)
    longest = np.zeros((layered.points.shape[0],))
    for column in (0, 1):
        np.minimum.at(shortest, edges[:, column], lengths)
        np.maximum.at(longest, edges[:, column], lengths)
    active = np.isfinite(shortest)
    ratio = float(np.max(longest[active] / shortest[active], initial=1.0))
    minimum, maximum = float(np.min(lengths)), float(np.max(lengths))
    policy = specification.size_compliance
    requested, issues = [], []
    for control in specification.size_controls:
        key = f"size:{control.control_id}"
        for name, bound, measured, violated in (
            ("minimum_size", control.minimum_size, minimum, lambda b, m, t: m < b - t),
            ("maximum_size", control.maximum_size, maximum, lambda b, m, t: m > b + t),
            (
                "maximum_growth_rate",
                control.maximum_growth_rate,
                ratio,
                lambda b, m, t: m > b + t,
            ),
        ):
            if bound is None:
                continue
            requested.append((f"{key}:{name}", float(bound)))
            tolerance = policy.absolute_tolerance + policy.relative_tolerance * abs(bound)
            if violated(bound, measured, tolerance):
                issues.append(f"{name}:{control.control_id}")
    achieved = (
        ("boundary_minimum_edge", minimum),
        ("boundary_maximum_edge", maximum),
        ("boundary_maximum_local_edge_ratio", ratio),
    )
    return tuple(requested), achieved, tuple(issues)


def _layered_result(
    plan: Any,
    layered: _Layered,
    control: BoundaryLayerControl,
    version: str,
    info: MeshingProviderInfo,
    /,
) -> CellMeshingResult:
    source = plan.source
    specification = plan.specification
    report = source.report
    mesh, core, layer_values = _merged_mesh(layered, report.source_revision)
    zones, patches, attribute = _organization(mesh, core, layer_values, layered)
    boundary_triangles = np.concatenate((layered.wall, layered.outer))
    metadata = SurfaceMetadata(
        source_id=report.source_id,
        source_revision=report.source_revision,
        coordinate_contract=source.coordinate_contract,
        provenance=("gmsh-layered-volume", plan.plan_id),
        cell_tags=("gmsh-occ-surface",) * boundary_triangles.shape[0],
    )
    boundary = SurfaceModel.from_triangles(
        np.asarray(mesh.coordinates),
        boundary_triangles,
        metadata,
        numeric_version=report.source_revision,
        repair_orientation=True,
        orient_closed_outward=True,
    )
    association, boundary_zones, provider_attribute = _boundary_association(
        _brep_model(source), boundary.mesh, plan.options.association_tolerance_factor
    )
    del boundary_zones
    geometry = CellGeometrySpec.affine(mesh)
    attributes = (provider_attribute, attribute)
    audit = audit_cell_mesh(
        mesh,
        geometry,
        evaluate_cell_quality(mesh, mesh.coordinates),
        patches=patches,
        associations=(association,),
        attributes=attributes,
        zones=zones,
        boundary=boundary,
    )
    if not audit.passed:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=audit.quality.worst_cell_global_ids,
        )
    requested, achieved, issues = _layer_compliance(control, layered)
    size_requested, size_achieved, size_issues = _boundary_size_compliance(
        specification, layered
    )
    requested = (*requested, *size_requested)
    achieved = (*achieved, *size_achieved)
    issues = (*issues, *size_issues)
    kinds = {block.cell_kind for block in mesh.blocks}
    family = specification.target.cell_families
    requested_kinds = {*family.required, *family.preferred, *family.allowed_transitions}
    if not set(family.required) <= kinds or not kinds <= requested_kinds:
        issues = (*issues, "cell_family")
    compliance = MeshingComplianceReport(
        specification.specification_id,
        issues=issues,
        requested=requested,
        achieved=(
            *achieved,
            *(
                (f"cell_count:{block.cell_kind}", float(block.cell_count))
                for block in mesh.blocks
            ),
        ),
    )
    if not compliance.passed:
        raise MeshingFailure(
            MeshingFailureCategory.COMPLIANCE_FAILED,
            "; ".join(compliance.issues),
            stage=MeshingStageKind.SPECIFICATION_COMPLIANCE.value,
        )
    trace = MeshingTrace(
        _layered_stages(plan, control, mesh, audit, compliance, geometry, association)
    )
    runtime = MeshingRuntimeInfo(
        plan.support.provider_id,
        version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=plan.options.num_threads == 1,
        enforced_limits=("cells",),
        unenforced_limits=("provider_workspace", "converted_arrays", "wall_time"),
    )
    provenance = SemanticProvenance(
        {
            "kind": "gmsh-layered-volume-result",
            "source_revision": report.source_revision,
            "plan": plan.plan_id,
            "mesh": mesh.mesh_id,
            "control": control.control_id,
            "zones": tuple(zone.zone_id for zone in zones),
            "patches": tuple(patch.patch_id for patch in patches),
        },
        resource_ids={"source": report.source_id},
    )
    return CellMeshingResult(
        mesh,
        geometry,
        source.coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        info,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        boundary=boundary,
        patches=patches,
        zones=zones,
        attributes=attributes,
        associations=(association,),
    )


def _layered_stages(
    plan: Any,
    control: Any,
    mesh: Any,
    audit: Any,
    compliance: Any,
    geometry: Any,
    association: Any,
    /,
) -> Any:
    specification = plan.specification
    stages = (
        (
            MeshingStageKind.SOURCE_INSPECTION,
            (plan.source.report.source_revision,),
            (plan.support.source_descriptor_id,),
        ),
        (
            MeshingStageKind.SCOPE_RESOLUTION,
            (specification.specification_id,),
            (specification.boundary_scope.scope_id,),
        ),
        (MeshingStageKind.CONTROL_RESOLUTION, (control.control_id,), (plan.plan_id,)),
        (MeshingStageKind.SURFACE_MESHING, (plan.plan_id,), (plan.plan_id,)),
        (MeshingStageKind.LAYER_GENERATION, (control.control_id,), (mesh.mesh_id,)),
        (MeshingStageKind.VOLUME_FILL, (plan.plan_id,), (mesh.mesh_id,)),
        (MeshingStageKind.CANONICALIZATION, (mesh.mesh_id,), (mesh.topology_id,)),
        (
            MeshingStageKind.GEOMETRY_ASSOCIATION,
            (mesh.mesh_id,),
            (association.association_id,),
        ),
        (
            MeshingStageKind.QUALITY_EVALUATION,
            (mesh.mesh_id,),
            (audit.quality.report_id,),
        ),
        (
            MeshingStageKind.GEOMETRY_AUDIT,
            (geometry.geometry_layout_id,),
            (audit.report_id,),
        ),
        (MeshingStageKind.TOPOLOGY_AUDIT, (mesh.topology_id,), (audit.report_id,)),
        (
            MeshingStageKind.SPECIFICATION_COMPLIANCE,
            (specification.specification_id,),
            (compliance.report_id,),
        ),
    )
    return tuple(
        MeshingStageReport(
            kind, MeshingStageStatus.PASSED, input_ids=inputs, output_ids=outputs
        )
        for kind, inputs, outputs in stages
    )


def _execute_layered_volume(
    gmsh: Any,
    plan: Any,
    version: str,
    info: MeshingProviderInfo,
    cache: Any,
    background: Any,
    /,
) -> CellMeshingResult:
    """Surface-mesh the BRep, grow the layers, and fill the core with fixed caps."""
    control = plan.specification.layer_controls[0]
    generation = _prepare_generation(gmsh, plan, cache, background)
    gmsh.model.mesh.generate(2)
    boundary = _surface_boundary(gmsh, plan, generation, control)
    maximum_size = control.core_maximum_size
    match control.route:
        case BoundaryLayerRoute.ADVANCING:
            layered = _advancing_layers(
                gmsh, boundary, control, plan.options, maximum_size
            )
        case BoundaryLayerRoute.PROVIDER:
            layered = _provider_layers(
                gmsh, boundary, control, plan.options, maximum_size
            )
        case route:
            raise TypeError(
                f"Layered volumes realize ADVANCING or PROVIDER routes, not {route!r}."
            )
    return _layered_result(plan, layered, control, version, info)


# ---------------------------------------------------------------- standalone core fill


def _fill_boundary_layer_core(
    gmsh: Any,
    layers: BoundaryLayerMesh,
    boundary: SurfaceModel,
    options: Any,
    maximum_size: float | None,
    version: str,
    info: MeshingProviderInfo,
    /,
) -> CellMeshingResult:
    """Fill the domain between a layer cap and a remaining boundary surface."""
    layer_points = np.asarray(layers.mesh.coordinates, dtype=np.float64)
    boundary_points = np.asarray(boundary.mesh.coordinates, dtype=np.float64)
    outer = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in boundary.mesh.blocks]
    )
    points = np.concatenate((layer_points, boundary_points))
    fixed = np.concatenate((_cap_triangles(layers), outer + layer_points.shape[0]))
    interior, core = _fill_core(gmsh, points, fixed, options, maximum_size)
    wall = _wall_triangles(layers)
    evidence = layers.evidence
    layered = _Layered(
        np.concatenate((points, interior)),
        {
            block.cell_kind: np.asarray(block.vertices, dtype=np.int64)
            for block in layers.mesh.blocks
        },
        np.asarray(layers.layer_index, dtype=np.int32),
        core,
        wall,
        outer + layer_points.shape[0],
        _cap_triangles(layers),
        np.full((outer.shape[0],), -1, dtype=np.int64),
        np.full((wall.shape[0],), -1, dtype=np.int64),
        np.asarray(evidence.achieved_thicknesses, dtype=np.float64),
        evidence.minimum_scale,
        (),
    )
    request_id = canonical_fingerprint(
        {
            "kind": "boundary-layer-core-fill",
            "layers": layers.result_id,
            "boundary": boundary.mesh.mesh_id,
            "maximum_size": maximum_size,
            "options": options.options_id,
        }
    )
    mesh, core_mask, layer_values = _merged_mesh(layered, boundary.mesh.numeric_version)
    zones, patches, attribute = _organization(mesh, core_mask, layer_values, layered)
    geometry = CellGeometrySpec.affine(mesh)
    audit = audit_cell_mesh(
        mesh,
        geometry,
        evaluate_cell_quality(mesh, mesh.coordinates),
        patches=patches,
        attributes=(attribute,),
        zones=zones,
    )
    if not audit.passed:
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "; ".join(audit.issues),
            stage=MeshingStageKind.GEOMETRY_AUDIT.value,
            entity_ids=audit.quality.worst_cell_global_ids,
        )
    compliance = MeshingComplianceReport(
        request_id,
        achieved=(
            ("layer_interface_compliance", 1.0),
            ("fixed_boundary_bitwise", 1.0),
            *(
                (f"cell_count:{block.cell_kind}", float(block.cell_count))
                for block in mesh.blocks
            ),
        ),
    )
    trace = MeshingTrace(
        (
            MeshingStageReport(
                MeshingStageKind.VOLUME_FILL,
                MeshingStageStatus.PASSED,
                input_ids=(layers.result_id, boundary.mesh.mesh_id),
                output_ids=(mesh.mesh_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.QUALITY_EVALUATION,
                MeshingStageStatus.PASSED,
                input_ids=(mesh.mesh_id,),
                output_ids=(audit.quality.report_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.TOPOLOGY_AUDIT,
                MeshingStageStatus.PASSED,
                input_ids=(mesh.topology_id,),
                output_ids=(audit.report_id,),
            ),
            MeshingStageReport(
                MeshingStageKind.SPECIFICATION_COMPLIANCE,
                MeshingStageStatus.PASSED,
                input_ids=(request_id,),
                output_ids=(compliance.report_id,),
            ),
        )
    )
    runtime = MeshingRuntimeInfo(
        info.provider_id,
        version,
        MeshingExecutionMode.IN_PROCESS,
        deterministic=options.num_threads == 1,
        unenforced_limits=("provider_workspace", "converted_arrays", "wall_time"),
    )
    provenance = SemanticProvenance(
        {
            "kind": "gmsh-boundary-layer-core-fill",
            "request": request_id,
            "mesh": mesh.mesh_id,
            "zones": tuple(zone.zone_id for zone in zones),
            "patches": tuple(patch.patch_id for patch in patches),
        },
        resource_ids={"boundary": boundary.metadata.source_id},
    )
    return CellMeshingResult(
        mesh,
        geometry,
        boundary.metadata.coordinate_contract,
        audit,
        audit.quality,
        compliance,
        trace,
        info,
        runtime,
        MeshingDerivativeMode.NONDIFFERENTIABLE,
        provenance,
        patches=patches,
        zones=zones,
        attributes=(attribute,),
    )


def _wall_triangles(layers: BoundaryLayerMesh, /) -> np.ndarray:
    """Layer-mesh boundary triangles on the wall (faces of exactly one layer cell)."""
    wall_vertices = np.asarray(layers.wall_vertices, dtype=np.int64)
    on_wall = np.zeros((layers.mesh.coordinates.shape[0],), dtype=np.bool_)
    on_wall[wall_vertices[wall_vertices >= 0]] = True
    connectivity = layers.mesh.connectivity
    # ty: ignore[invalid-argument-type]
    rows = _connectivity_face_rows(connectivity)
    # ty: ignore[invalid-argument-type]
    incidents = _connectivity_face_incidents(connectivity)
    faces = [
        row
        for row, adjacent in zip(rows, incidents, strict=True)
        if len(adjacent) == 1 and np.all(on_wall[row])
    ]
    if any(len(row) != 3 for row in faces):
        raise MeshingFailure(
            MeshingFailureCategory.UNSUPPORTED_COMBINATION,
            "Core-fill wall patches require triangle wall faces.",
            stage=_FILL_STAGE,
        )
    return np.asarray(faces, dtype=np.int64).reshape(-1, 3)


__all__: list[str] = []
