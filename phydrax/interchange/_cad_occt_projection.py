#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit OCCT comparison queries against one persisted B-Rep revision.

Every query needs one host call into OCCT extrema; that per-point call is the
external-provider boundary and cannot be vectorized. Queries are grouped by
entity so each face and edge prepares its surface/curve projector,
trimmed-face classifier, parameter bounds, and seam/pole data exactly once.
Bookkeeping around the calls (grouping, BVH candidate search, closure
relations, frames, and status reduction) is vectorized NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

# This module is the optional OCCT provider boundary: its checked public
# contracts name the provider shape type, so the provider loads with it.
from OCP.TopoDS import TopoDS_Shape

from .._bvh import bvh_overlap_pairs_host, prepare_bvh
from .._fingerprint import canonical_fingerprint
from .._validation import positive_finite_float
from ..geometry._planar_embedding import PlanarEmbedding
from ..geometry.brep._assembly import BRepAffinePlacement, BRepAssemblyUse
from ..geometry.brep._model import BRepModel
from ..geometry.brep._projection_contracts import (
    AbstractBRepProjection,
    BRepEntityDimension,
    BRepProjectionPolicy,
    BRepProjectionResult,
    BRepProjectionStatus,
    closure_codes,
)
from ..typing import checked
from ._cad_occt import (
    _occt_modules,
    explore_unique,
    import_occt_shape,
    read_occt_shape,
    shape_index,
)


class _LazyOCCT:
    """Explicit lazy symbol access to the interchange OCCT provider."""

    def __getattr__(self, name: str) -> Any:
        return _occt_modules().symbol(name)


occt = _LazyOCCT()


def _ensure_occt() -> None:
    _occt_modules()


@dataclass(frozen=True, slots=True, eq=False)
class _Occt:
    """Host OCCT inventory and source-proved row correspondence."""

    vertices: tuple[Any, ...]
    edges: tuple[Any, ...]
    faces: tuple[Any, ...]
    solids: tuple[Any, ...]
    surfaces: tuple[Any, ...]
    curves: tuple[Any, ...]
    external_counts: tuple[int, int, int, int]
    entity_rows: dict[tuple[int, int, tuple[str, ...]], int]
    entity_paths: dict[tuple[int, int], tuple[tuple[str, ...], ...]]
    solid_sources: dict[int, tuple[tuple[int, tuple[str, ...]], ...]]
    face_orientation: dict[tuple[int, int, tuple[str, ...]], float]
    face_poles: dict[tuple[int, int, tuple[str, ...]], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class _SourceMapping:
    """Exact native occurrence identities and any proven OCCT row bindings."""

    occurrence_paths: tuple[tuple[str, ...], ...]
    entity_occurrences: tuple[tuple[int, ...], ...]
    entity_paths: dict[tuple[int, int], tuple[tuple[str, ...], ...]]
    entity_rows: dict[tuple[int, int, tuple[str, ...]], int]
    solid_sources: dict[int, tuple[tuple[int, tuple[str, ...]], ...]]
    face_orientation: dict[tuple[int, int, tuple[str, ...]], float]
    face_poles: dict[tuple[int, int, tuple[str, ...]], tuple[int, ...]]


def _source_paths(
    model: BRepModel, /
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[int, ...], ...],
    dict[tuple[int, int], tuple[tuple[str, ...], ...]],
]:
    geometry = model.geometry
    occurrence_paths = tuple(value.path for value in geometry.iter_occurrences())
    occurrence_indices = {path: index for index, path in enumerate(occurrence_paths)}
    counts = (
        model.topology.num_vertices,
        model.topology.num_edges,
        model.topology.num_faces,
        model.topology.num_solids,
    )
    kind_dimensions = {"vertex": 0, "edge": 1, "face": 2, "solid": 3}
    paths_by_entity: dict[tuple[int, int], set[tuple[str, ...]]] = {}
    for incidence in geometry.qualified_entity_incidence(
        model.source_id, model.source_revision
    ):
        entity = incidence.member
        if incidence.container != entity:
            continue
        dimension = kind_dimensions[entity.kind]
        if entity.index >= counts[dimension]:
            continue
        paths_by_entity.setdefault((dimension, entity.index), set()).add(
            entity.occurrence_path
        )
    entity_paths = {
        key: (
            (((),) if () in paths else ())
            + tuple(path for path in occurrence_paths if path in paths)
        )
        for key, paths in paths_by_entity.items()
    }
    entity_occurrences = tuple(
        tuple(
            occurrence_indices[path]
            for path in entity_paths.get((dimension, index), ())
            if path in occurrence_indices
        )
        for dimension, count in enumerate(counts)
        for index in range(count)
    )
    return occurrence_paths, entity_occurrences, entity_paths


def _children(shape: TopoDS_Shape, occt_modules: Any, /) -> tuple[TopoDS_Shape, ...]:
    iterator = occt_modules.TopoDS.TopoDS_Iterator(shape, True, True)
    values: list[TopoDS_Shape] = []
    while iterator.More():
        values.append(iterator.Value())
        iterator.Next()
    return tuple(values)


def _matches_use(use: BRepAssemblyUse, shape: TopoDS_Shape, occt_modules: Any, /) -> bool:
    shape_type = shape.ShapeType()
    if use.solid_index is not None:
        return shape_type == occt_modules.TopAbs.TopAbs_SOLID
    return shape_type in (
        occt_modules.TopAbs.TopAbs_COMPOUND,
        occt_modules.TopAbs.TopAbs_COMPSOLID,
    )


def _solid_occurrence_shapes(
    model: BRepModel, shape: TopoDS_Shape, occt_modules: Any, /
) -> tuple[tuple[tuple[str, ...], int, TopoDS_Shape], ...]:
    graph = model.geometry.assembly_graph
    if not graph.roots:
        return ()
    definitions = {
        definition.definition_id: definition for definition in graph.definitions
    }

    def complete_use_shape(use: BRepAssemblyUse, candidate: TopoDS_Shape, /) -> bool:
        if not _matches_use(use, candidate, occt_modules):
            return False
        if use.solid_index is not None:
            return True
        definition_id = use.definition_id
        if definition_id is None:
            return False
        definition = definitions.get(definition_id)
        if definition is None:
            return False
        children = _children(candidate, occt_modules)
        return len(children) == len(definition.uses) and all(
            _matches_use(child_use, child, occt_modules)
            for child_use, child in zip(definition.uses, children, strict=True)
        )

    roots = graph.roots
    if len(roots) == 1 and complete_use_shape(roots[0], shape):
        root_shapes = (shape,)
    else:
        children = _children(shape, occt_modules)
        if len(children) < len(roots):
            return ()
        root_shapes = children[: len(roots)]
        if any(
            not complete_use_shape(use, child)
            for use, child in zip(roots, root_shapes, strict=True)
        ):
            return ()
    pending: list[tuple[BRepAssemblyUse, TopoDS_Shape, tuple[str, ...]]] = [
        (use, child, ())
        for use, child in reversed(tuple(zip(roots, root_shapes, strict=True)))
    ]
    occurrences: list[tuple[tuple[str, ...], int, TopoDS_Shape]] = []
    while pending:
        use, child, parent_path = pending.pop()
        if not complete_use_shape(use, child):
            return ()
        path = (*parent_path, use.use_id)
        if use.solid_index is not None:
            occurrences.append((path, use.solid_index, child))
            continue
        definition_id = use.definition_id
        if definition_id is None:
            return ()
        definition = definitions.get(definition_id)
        if definition is None:
            return ()
        children = _children(child, occt_modules)
        pending.extend(
            (child_use, child_shape, path)
            for child_use, child_shape in reversed(
                tuple(zip(definition.uses, children, strict=True))
            )
        )
    return tuple(occurrences)


def _add_entity_binding(
    destination: dict[tuple[int, int, tuple[str, ...]], int],
    dimension: int,
    source_index: int,
    path: tuple[str, ...],
    external_index: int,
    allowed_paths: dict[tuple[int, int], tuple[tuple[str, ...], ...]],
    /,
) -> bool:
    key = (dimension, source_index, path)
    if path not in allowed_paths.get((dimension, source_index), ()):
        return False
    existing = destination.get(key)
    if existing == -1:
        return False
    if existing is not None and existing != external_index:
        destination[key] = -1
        return False
    destination[key] = external_index
    return True


def _coedge_vertices(geometry: Any, coedge: int, /) -> tuple[int, int]:
    """Return a native coedge's start and end vertices along its traversal."""
    first, last = geometry.edge_vertices[geometry.coedge_edges[coedge]]
    return (first, last) if geometry.coedge_senses[coedge] > 0 else (last, first)


def _aligned_wire(
    model: BRepModel,
    native_loop: tuple[int, ...],
    wire_edges: tuple[Any, ...],
    placement: BRepAffinePlacement | None,
    occt_modules: Any,
    /,
) -> tuple[tuple[Any, Any, Any], ...] | None:
    """Align one OCCT wire with a native loop up to rotation and reversal.

    The native decoder orients loops about the face normal and may start them
    anywhere, so the wire's cyclic edge sequence is matched in both directions.
    Each candidate must reproduce every coedge's start and end vertex at the
    placed native vertex points; only a unique alignment is accepted.
    """
    count = len(native_loop)
    if count != len(wire_edges) or not count:
        return None
    geometry = model.geometry
    points = np.asarray(geometry.vertex_points, dtype=np.float64)
    scale = float(model.report.curve_surface_scale)
    tolerance = max(
        float(model.report.curve_surface_tolerance),
        64.0 * float(np.finfo(np.float64).eps) * max(1.0, scale),
    )

    def placed(vertex: int, /) -> np.ndarray:
        point = tuple(float(value) for value in points[vertex])
        if placement is not None:
            point = tuple(float(value) for value in placement.apply(point))
        return np.asarray(point, dtype=np.float64)

    def external_point(vertex: Any, /) -> np.ndarray:
        point = occt_modules.BRep.BRep_Tool.Pnt_s(vertex)
        return np.asarray((point.X(), point.Y(), point.Z()), dtype=np.float64)

    # Cumulated orientation orders each edge's vertices along the wire.
    uses = tuple(
        (
            edge,
            occt_modules.TopExp.TopExp.FirstVertex_s(edge, True),
            occt_modules.TopExp.TopExp.LastVertex_s(edge, True),
        )
        for edge in wire_edges
    )
    if any(first.IsNull() or last.IsNull() for _, first, last in uses):
        return None
    native_ends = tuple(
        tuple(placed(vertex) for vertex in _coedge_vertices(geometry, coedge))
        for coedge in native_loop
    )
    external_ends = tuple(
        (external_point(first), external_point(last)) for _, first, last in uses
    )
    forward = occt_modules.TopAbs.TopAbs_FORWARD
    candidates: list[tuple[tuple[Any, Any, Any], ...]] = []
    for reverse in (False, True):
        start, end = (1, 0) if reverse else (0, 1)
        for shift in range(count):
            order = tuple(
                (count - 1 - (shift + position)) % count
                if reverse
                else (shift + position) % count
                for position in range(count)
            )
            # A reversed traversal flips each coedge's sense against its edge.
            if all(
                geometry.coedge_senses[coedge]
                == (1 if uses[row][0].Orientation() == forward else -1)
                * (-1 if reverse else 1)
                and np.max(np.abs(native[0] - external_ends[row][start])) <= tolerance
                and np.max(np.abs(native[1] - external_ends[row][end])) <= tolerance
                for coedge, native, row in zip(
                    native_loop, native_ends, order, strict=True
                )
            ):
                candidates.append(
                    tuple(
                        (uses[row][0], uses[row][1 + start], uses[row][1 + end])
                        for row in order
                    )
                )
    return candidates[0] if len(candidates) == 1 else None


def _map_face(
    model: BRepModel,
    face_index: int,
    path: tuple[str, ...],
    orientation_sign: int,
    placement: BRepAffinePlacement | None,
    face: Any,
    faces: tuple[Any, ...],
    edges: tuple[Any, ...],
    vertices: tuple[Any, ...],
    allowed_paths: dict[tuple[int, int], tuple[tuple[str, ...], ...]],
    occt_modules: Any,
    /,
) -> (
    tuple[
        dict[tuple[int, int, tuple[str, ...]], int],
        float,
        tuple[int, ...],
    ]
    | None
):
    geometry = model.geometry
    try:
        face_row = shape_index(faces, face)
    except RuntimeError:
        return None
    face_key = (int(BRepEntityDimension.FACE), face_index, path)
    local: dict[tuple[int, int, tuple[str, ...]], int] = {face_key: face_row}

    def bind_local(dimension: int, source_index: int, external_index: int, /) -> bool:
        key = (dimension, source_index, path)
        existing = local.get(key)
        if existing is not None and existing != external_index:
            return False
        local[key] = external_index
        return True

    loops = geometry.face_loops[face_index]
    wires = _explore_all(
        face,
        occt_modules.TopAbs.TopAbs_WIRE,
        occt_modules.TopoDS.TopoDS.Wire,
        occt_modules,
    )
    if len(loops) != len(wires):
        return None
    pole_rows: set[int] = set()
    unmatched = list(range(len(wires)))
    for native_loop in loops:
        matches = [
            (row, aligned)
            for row in unmatched
            if (
                aligned := _aligned_wire(
                    model,
                    native_loop,
                    _wire_edges(wires[row], face, occt_modules),
                    placement,
                    occt_modules,
                )
            )
            is not None
        ]
        if len(matches) != 1:
            return None
        row, aligned_edges = matches[0]
        unmatched.remove(row)
        for coedge, (edge, first, last) in zip(native_loop, aligned_edges, strict=True):
            native_edge = geometry.coedge_edges[coedge]
            try:
                edge_row = shape_index(edges, edge)
            except RuntimeError:
                return None
            if bool(occt_modules.BRep.BRep_Tool.Degenerated_s(edge)) != (
                geometry.edge_curves[native_edge] < 0
            ):
                return None
            if not bind_local(int(BRepEntityDimension.EDGE), native_edge, edge_row):
                return None
            for native_vertex, external_vertex in zip(
                _coedge_vertices(geometry, coedge), (first, last), strict=True
            ):
                try:
                    vertex_row = shape_index(vertices, external_vertex)
                except RuntimeError:
                    return None
                if not bind_local(
                    int(BRepEntityDimension.VERTEX), native_vertex, vertex_row
                ):
                    return None
                if geometry.edge_curves[native_edge] < 0:
                    pole_rows.add(vertex_row)
    for dimension, source_index, source_path in local:
        if source_path not in allowed_paths.get((dimension, source_index), ()):
            return None
    orientation = float(np.asarray(model.orientation)[face_index]) * orientation_sign
    return local, orientation, tuple(sorted(pole_rows))


def _map_free_edge(
    model: BRepModel,
    source_edge: int,
    external_row: int,
    edges: tuple[Any, ...],
    vertices: tuple[Any, ...],
    allowed_paths: dict[tuple[int, int], tuple[tuple[str, ...], ...]],
    occt_modules: Any,
    /,
) -> dict[tuple[int, int, tuple[str, ...]], int] | None:
    geometry = model.geometry
    edge = edges[external_row]
    if bool(occt_modules.BRep.BRep_Tool.Degenerated_s(edge)) != (
        geometry.edge_curves[source_edge] < 0
    ):
        return None
    first = occt_modules.TopExp.TopExp.FirstVertex_s(edge)
    last = occt_modules.TopExp.TopExp.LastVertex_s(edge)
    if first.IsNull() or last.IsNull():
        return None
    orientation = edge.Orientation()
    if orientation == occt_modules.TopAbs.TopAbs_FORWARD:
        source_vertices = geometry.edge_vertices[source_edge]
    elif orientation == occt_modules.TopAbs.TopAbs_REVERSED:
        source_vertices = geometry.edge_vertices[source_edge][::-1]
    else:
        return None
    local = {(int(BRepEntityDimension.EDGE), source_edge, ()): external_row}
    for source_vertex, external_vertex in zip(
        source_vertices, (first, last), strict=True
    ):
        try:
            vertex_row = shape_index(vertices, external_vertex)
        except RuntimeError:
            return None
        key = (int(BRepEntityDimension.VERTEX), source_vertex, ())
        existing = local.get(key)
        if existing is not None and existing != vertex_row:
            return None
        local[key] = vertex_row
    if any(
        path not in allowed_paths.get((dimension, source_index), ())
        for dimension, source_index, path in local
    ):
        return None
    return local


def _placement_sign(placement: Any, /) -> int:
    matrix = placement.matrix
    a, b, c = matrix[0][:3]
    d, e, f = matrix[1][:3]
    g, h, i = matrix[2][:3]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    return 1 if determinant > 0 else -1


def _build_source_mapping(
    model: BRepModel,
    shape: Any,
    vertices: tuple[Any, ...],
    edges: tuple[Any, ...],
    faces: tuple[Any, ...],
    solids: tuple[Any, ...],
    occt_modules: Any,
    /,
) -> _SourceMapping:
    occurrence_paths, entity_occurrences, allowed_paths = _source_paths(model)
    occurrences = tuple(model.geometry.iter_occurrences())
    occurrence_by_path = {value.path: value for value in occurrences}
    rows: dict[tuple[int, int, tuple[str, ...]], int] = {}
    orientations: dict[tuple[int, int, tuple[str, ...]], float] = {}
    poles: dict[tuple[int, int, tuple[str, ...]], tuple[int, ...]] = {}
    solid_sources: dict[int, list[tuple[int, tuple[str, ...]]]] = {}
    solid_shape_rows = _solid_occurrence_shapes(model, shape, occt_modules)
    if len(solid_shape_rows) == len(occurrences) and all(
        (path, source_index) == (occurrence.path, occurrence.solid_index)
        for (path, source_index, _), occurrence in zip(
            solid_shape_rows, occurrences, strict=True
        )
    ):
        for path, source_solid, solid_shape in solid_shape_rows:
            try:
                solid_row = shape_index(solids, solid_shape)
            except RuntimeError:
                continue
            if _add_entity_binding(
                rows,
                int(BRepEntityDimension.SOLID),
                source_solid,
                path,
                solid_row,
                allowed_paths,
            ):
                solid_sources.setdefault(solid_row, []).append((source_solid, path))
            native_faces = model.topology.solid_faces[source_solid]
            external_faces = _explore_all(
                solid_shape,
                occt_modules.TopAbs.TopAbs_FACE,
                occt_modules.TopoDS.TopoDS.Face,
                occt_modules,
            )
            if len(native_faces) != len(external_faces):
                continue
            sign = _placement_sign(occurrence_by_path[path].placement)
            for native_face, external_face in zip(
                native_faces, external_faces, strict=True
            ):
                mapped = _map_face(
                    model,
                    native_face,
                    path,
                    sign,
                    occurrence_by_path[path].placement,
                    external_face,
                    faces,
                    edges,
                    vertices,
                    allowed_paths,
                    occt_modules,
                )
                if mapped is None:
                    continue
                face_rows, orientation, face_poles = mapped
                for key, external_row in face_rows.items():
                    _add_entity_binding(
                        rows,
                        key[0],
                        key[1],
                        key[2],
                        external_row,
                        allowed_paths,
                    )
                face_key = (
                    int(BRepEntityDimension.FACE),
                    native_face,
                    path,
                )
                if rows.get(face_key, -1) >= 0:
                    orientations[face_key] = orientation
                    poles[face_key] = face_poles

    external_solid_face_rows = {
        shape_index(
            list(faces),
            external_face,
        )
        for solid in solids
        for external_face in _explore_all(
            solid,
            occt_modules.TopAbs.TopAbs_FACE,
            occt_modules.TopoDS.TopoDS.Face,
            occt_modules,
        )
    }
    native_free_faces = [
        face_index
        for face_index, owners in enumerate(model.topology.face_solids)
        if not owners
    ]
    external_free_faces = [
        face_index
        for face_index in range(len(faces))
        if face_index not in external_solid_face_rows
    ]
    if len(native_free_faces) == len(external_free_faces):
        for native_face, external_face_row in zip(
            native_free_faces, external_free_faces, strict=True
        ):
            mapped = _map_face(
                model,
                native_face,
                (),
                1,
                None,
                faces[external_face_row],
                faces,
                edges,
                vertices,
                allowed_paths,
                occt_modules,
            )
            if mapped is None:
                continue
            face_rows, orientation, face_poles = mapped
            for key, external_row in face_rows.items():
                _add_entity_binding(
                    rows,
                    key[0],
                    key[1],
                    key[2],
                    external_row,
                    allowed_paths,
                )
            face_key = (int(BRepEntityDimension.FACE), native_face, ())
            if rows.get(face_key, -1) >= 0:
                orientations[face_key] = orientation
                poles[face_key] = face_poles

    external_face_edge_rows = {
        shape_index(edges, external_edge)
        for face in faces
        for external_edge in _explore_all(
            face,
            occt_modules.TopAbs.TopAbs_EDGE,
            occt_modules.TopoDS.TopoDS.Edge,
            occt_modules,
        )
    }
    native_free_edges = [
        edge_index
        for edge_index, owners in enumerate(model.topology.edge_faces)
        if not owners
    ]
    external_free_edges = [
        edge_index
        for edge_index in range(len(edges))
        if edge_index not in external_face_edge_rows
    ]
    if len(native_free_edges) == len(external_free_edges):
        for native_edge, external_edge in zip(
            native_free_edges, external_free_edges, strict=True
        ):
            edge_rows = _map_free_edge(
                model,
                native_edge,
                external_edge,
                edges,
                vertices,
                allowed_paths,
                occt_modules,
            )
            if edge_rows is not None:
                for key, external_row in edge_rows.items():
                    _add_entity_binding(
                        rows,
                        key[0],
                        key[1],
                        key[2],
                        external_row,
                        allowed_paths,
                    )

    native_attached_vertices = {
        vertex for endpoints in model.geometry.edge_vertices for vertex in endpoints
    }
    external_attached_vertices: set[int] = set()
    for edge in edges:
        for external_vertex in (
            occt_modules.TopExp.TopExp.FirstVertex_s(edge),
            occt_modules.TopExp.TopExp.LastVertex_s(edge),
        ):
            if external_vertex.IsNull():
                continue
            external_attached_vertices.add(shape_index(vertices, external_vertex))
    native_isolated_vertices = [
        vertex
        for vertex in range(model.topology.num_vertices)
        if vertex not in native_attached_vertices
    ]
    external_isolated_vertices = [
        vertex
        for vertex in range(len(vertices))
        if vertex not in external_attached_vertices
    ]
    if len(native_isolated_vertices) == len(external_isolated_vertices):
        for native_vertex, external_vertex in zip(
            native_isolated_vertices, external_isolated_vertices, strict=True
        ):
            _add_entity_binding(
                rows,
                int(BRepEntityDimension.VERTEX),
                native_vertex,
                (),
                external_vertex,
                allowed_paths,
            )

    solid_sources_final = {row: tuple(values) for row, values in solid_sources.items()}
    return _SourceMapping(
        occurrence_paths,
        entity_occurrences,
        allowed_paths,
        rows,
        solid_sources_final,
        orientations,
        poles,
    )


@dataclass(slots=True)
class _Rows:
    """Per-query accumulation of one projection batch (world coordinates)."""

    points: np.ndarray
    parameters: np.ndarray
    residuals: np.ndarray
    status: np.ndarray
    first: np.ndarray
    second: np.ndarray
    normals: np.ndarray

    @classmethod
    def empty(cls, count: int, /) -> _Rows:
        return cls(
            np.full((count, 3), np.nan),
            np.full((count, 2), np.nan),
            np.full((count,), np.inf),
            np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8),
            np.full((count, 3), np.nan),
            np.full((count, 3), np.nan),
            np.full((count, 3), np.nan),
        )


def _xyz(value: Any, /) -> np.ndarray:
    return np.asarray((value.X(), value.Y(), value.Z()), dtype=np.float64)


def _explore_all(
    shape: Any,
    kind: Any,
    caster: Callable[[Any], Any],
    occt_modules: Any,
    /,
) -> tuple[Any, ...]:
    explorer = occt_modules.TopExp.TopExp_Explorer(shape, kind)
    entities = []
    while explorer.More():
        entities.append(caster(explorer.Current()))
        explorer.Next()
    return tuple(entities)


def _wire_edges(wire: Any, face: Any, occt_modules: Any, /) -> tuple[Any, ...]:
    explorer = occt_modules.BRepTools.BRepTools_WireExplorer(wire, face)
    edges = []
    while explorer.More():
        edges.append(explorer.Current())
        explorer.Next()
    return tuple(edges)


def _box(shape: Any, /) -> np.ndarray:

    box = occt.Bnd_Box()
    occt.BRepBndLib.Add_s(shape, box)
    # OCP 8 no longer converts ``Bnd_Box.Get``'s ``Limits`` struct; read the corners.
    return np.stack((_xyz(box.CornerMin()), _xyz(box.CornerMax())))


def _bound_shape(model: BRepModel, source: str | Path | TopoDS_Shape, /) -> TopoDS_Shape:
    """Bind explicit OCCT queries to the model's exact native source revision."""
    match source:
        case str() | Path():
            shape, _, _ = read_occt_shape(source)
        case occt.TopoDS_Shape():
            if source.IsNull():
                raise ValueError(
                    "prepare_occt_projection.source must be a nonnull OCCT shape; got a null shape"
                )
            shape = source
        case _:
            raise TypeError(
                f"prepare_occt_projection.source must be TopoDS_Shape or a CAD file path; got {type(source).__name__}"
            )
    tolerance = float(model.report.curve_surface_tolerance)
    scale = float(model.report.curve_surface_scale)
    if (
        not np.isfinite(tolerance)
        or tolerance < 0.0
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "BRepModel source accuracy must be finite and have a positive physical scale before OCCT projection."
        )
    # The report records the absolute curve-surface tolerance, i.e. the import
    # policy's relative tolerance times the source scale.
    native_source = import_occt_shape(
        shape,
        coordinate_contract=model.coordinate_contract,
        source_id=model.source_id,
        relative_geometric_tolerance=tolerance / scale,
    ).model
    if native_source.source_revision != model.source_revision:
        raise ValueError(
            "prepare_occt_projection.source must reconstruct the exact native physical source_revision; "
            f"got {native_source.source_revision!r}, expected {model.source_revision!r}."
        )
    return shape


@final
class PreparedOcctProjection(AbstractBRepProjection):
    """Explicit numerical OCCT observations bound to one native source revision.

    Entity indices are native B-Rep identities; optional OCCT subshape rows are
    used only when their occurrence-qualified incidence is structurally proved.
    Candidate points and parameters are diagnostic only: OCCT does not provide
    the certified distance bounds required to resolve a native projection row,
    so returned rows remain UNRESOLVED. With a planar ``embedding`` queries and
    results use its two-dimensional frame.
    """

    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    policy: BRepProjectionPolicy
    embedding: PlanarEmbedding | None = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    vertex_points: Array
    edge_vertices: Array
    edge_degenerate: Array
    edge_closed: Array
    edge_ranges: Array
    face_bounds: Array
    face_periods: Array
    entity_counts: tuple[int, int, int, int] = eqx.field(static=True)
    closure_codes: Array
    entity_boxes: Array
    occt: _Occt = eqx.field(static=True)
    projection_id: str = eqx.field(static=True)
    occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    entity_occurrences: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        model: BRepModel,
        shape: TopoDS_Shape,
        policy: BRepProjectionPolicy,
        embedding: PlanarEmbedding | None,
        /,
    ) -> None:
        _ensure_occt()

        occt_modules = _occt_modules()
        faces = tuple(explore_unique(shape, occt.TopAbs_FACE, occt.TopoDS.Face))
        edges = tuple(explore_unique(shape, occt.TopAbs_EDGE, occt.TopoDS.Edge))
        vertices = tuple(explore_unique(shape, occt.TopAbs_VERTEX, occt.TopoDS.Vertex))
        solids = tuple(explore_unique(shape, occt.TopAbs_SOLID, occt.TopoDS.Solid))
        topology = model.topology
        mapping = _build_source_mapping(
            model, shape, vertices, edges, faces, solids, occt_modules
        )
        vertex_points = np.asarray(
            [_xyz(occt.BRep_Tool.Pnt_s(vertex)) for vertex in vertices], dtype=np.float64
        ).reshape(-1, 3)
        edge_vertices = np.full((len(edges), 2), -1, dtype=np.int64)
        degenerate = np.zeros((len(edges),), dtype=np.bool_)
        closed = np.zeros((len(edges),), dtype=np.bool_)
        ranges = np.zeros((len(edges), 2), dtype=np.float64)
        curves = []
        for row, edge in enumerate(edges):
            first, last = occt.TopExp.FirstVertex_s(edge), occt.TopExp.LastVertex_s(edge)
            if not first.IsNull():
                edge_vertices[row, 0] = shape_index(vertices, first)
            if not last.IsNull():
                edge_vertices[row, 1] = shape_index(vertices, last)
            degenerate[row] = occt.BRep_Tool.Degenerated_s(edge)
            adaptor = occt.BRepAdaptor_Curve(edge)
            ranges[row] = (adaptor.FirstParameter(), adaptor.LastParameter())
            closed[row] = (
                not degenerate[row]
                and edge_vertices[row, 0] == edge_vertices[row, 1]
                and adaptor.IsClosed()
            )
            curves.append(
                None
                if degenerate[row]
                else occt.BRep_Tool.Curve_s(
                    edge, float(ranges[row, 0]), float(ranges[row, 1])
                )
            )
        surfaces, bounds, periods = [], [], []
        for face in faces:
            surfaces.append(occt.BRep_Tool.Surface_s(face))
            u0, u1, v0, v1 = occt.BRepTools.UVBounds_s(face)
            bounds.append((u0, u1, v0, v1))
            adaptor = occt.BRepAdaptor_Surface(face, True)
            u_period = adaptor.UPeriod() if adaptor.IsUPeriodic() else np.nan
            v_period = adaptor.VPeriod() if adaptor.IsVPeriodic() else np.nan
            # Only a face spanning its full period has a parametric seam.
            periods.append(
                (
                    u_period if u1 - u0 >= u_period * (1.0 - 1.0e-9) else np.nan,
                    v_period if v1 - v0 >= v_period * (1.0 - 1.0e-9) else np.nan,
                )
            )
        external_counts = (len(vertices), len(edges), len(faces), len(solids))
        counts = (
            topology.num_vertices,
            topology.num_edges,
            topology.num_faces,
            topology.num_solids,
        )
        closure = closure_codes(
            topology,
            model.geometry.edge_vertices,
            counts,
        )
        boxes = np.concatenate(
            (
                np.stack((vertex_points, vertex_points), axis=1),
                np.asarray([_box(edge) for edge in edges]).reshape(-1, 2, 3),
                np.asarray([_box(face) for face in faces]).reshape(-1, 2, 3),
                np.asarray([_box(solid) for solid in solids]).reshape(-1, 2, 3),
            )
        )
        ambient = 3
        if embedding is not None:
            if topology.num_solids:
                raise ValueError(
                    "PreparedOcctProjection.embedding must bind a face-only source revision."
                )
            if not vertex_points.size:
                raise ValueError(
                    "PreparedOcctProjection.embedding requires source vertices to prove the plane binding."
                )
            residual = float(np.max(np.abs(embedding.plane_residual(vertex_points))))
            if residual > policy.classifier_tolerance:
                raise ValueError(
                    f"PreparedOcctProjection.embedding must contain the source vertices within classifier tolerance={policy.classifier_tolerance}; got maximum_plane_residual={residual}"
                )
            ambient = 2
        self.source_id = model.source_id
        self.source_revision = model.source_revision
        self.frame_id = model.coordinate_contract.spatial_id
        self.model_id = model.model_id
        self.policy = policy
        self.embedding = embedding
        self.ambient_dimension = ambient
        self.vertex_points = jnp.asarray(vertex_points)
        self.edge_vertices = jnp.asarray(edge_vertices.astype(np.int32))
        self.edge_degenerate = jnp.asarray(degenerate)
        self.edge_closed = jnp.asarray(closed)
        self.edge_ranges = jnp.asarray(ranges)
        self.face_bounds = jnp.asarray(
            np.asarray(bounds, dtype=np.float64).reshape(-1, 4)
        )
        self.face_periods = jnp.asarray(
            np.asarray(periods, dtype=np.float64).reshape(-1, 2)
        )
        self.entity_counts = counts
        self.closure_codes = jnp.asarray(closure, dtype=jnp.int64)
        self.entity_boxes = jnp.asarray(boxes)
        self.occurrence_paths = mapping.occurrence_paths
        self.entity_occurrences = mapping.entity_occurrences
        self.occt = _Occt(
            vertices,
            edges,
            faces,
            solids,
            tuple(surfaces),
            tuple(curves),
            external_counts,
            mapping.entity_rows,
            mapping.entity_paths,
            mapping.solid_sources,
            mapping.face_orientation,
            mapping.face_poles,
        )
        self.projection_id = canonical_fingerprint(
            {
                "kind": "prepared-brep-projection",
                "source_revision": model.source_revision,
                "frame_id": self.frame_id,
                "source_id": model.source_id,
                "model_id": model.model_id,
                "policy": policy.policy_id,
                "embedding": None if embedding is None else embedding.embedding_id,
                "occurrence_paths": mapping.occurrence_paths,
                "entity_rows": tuple(
                    (dimension, index, path, row)
                    for (dimension, index, path), row in sorted(
                        mapping.entity_rows.items()
                    )
                ),
            }
        )

    def _queries(self, points: ArrayLike, label: str, /) -> np.ndarray:
        queries = np.asarray(points, dtype=np.float64)
        if (
            queries.ndim != 2
            or queries.shape[1] != self.ambient_dimension
            or not np.all(np.isfinite(queries))
        ):
            raise ValueError(
                f"{label} must be finite points in the source ambient dimension {self.ambient_dimension}; got shape={queries.shape}."
            )
        return queries

    def _world(self, points: np.ndarray, /) -> np.ndarray:
        if self.embedding is None:
            return points
        return self.embedding.to_world(points)

    def _ambient(self, points: np.ndarray, /) -> np.ndarray:
        if self.embedding is None:
            return points
        return self.embedding.to_planar(points)

    def _directions(self, directions: np.ndarray, /) -> np.ndarray:
        if self.embedding is None:
            return directions
        basis = np.asarray(
            (self.embedding.x_axis, self.embedding.y_axis), dtype=np.float64
        )
        return directions @ basis.T

    # -- projection -------------------------------------------------------------

    def project(
        self,
        points: ArrayLike,
        dimensions: ArrayLike,
        indices: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Return numerical OCCT candidates without claiming certified minima."""
        queries = self._queries(points, "Projection queries")
        if occurrence_paths is not None and (
            not isinstance(occurrence_paths, tuple)
            or len(occurrence_paths) != queries.shape[0]
            or any(
                not isinstance(path, tuple)
                or any(not isinstance(part, str) or not part for part in path)
                for path in occurrence_paths
            )
        ):
            raise ValueError(
                "PreparedOcctProjection.occurrence_paths must contain one exact path tuple per query."
            )
        dims = np.asarray(dimensions)
        rows = np.asarray(indices)
        if not np.issubdtype(dims.dtype, np.integer) or not np.issubdtype(
            rows.dtype, np.integer
        ):
            raise TypeError(
                f"PreparedOcctProjection.project.dimensions must use integer dtype together with indices; got dimensions_dtype={dims.dtype}, indices_dtype={rows.dtype}"
            )
        if dims.shape != (queries.shape[0],) or rows.shape != dims.shape:
            raise ValueError(
                f"PreparedOcctProjection.project.indices must contain one index per dimension and query; got queries={queries.shape[0]}, dimensions_shape={dims.shape}, indices_shape={rows.shape}"
            )
        if np.any((dims < 0) | (dims > 2)):
            raise ValueError(
                f"PreparedOcctProjection.project.dimensions must target vertices, edges or faces; got {dims!r}"
            )
        limits = np.asarray(self.entity_counts[:3])[dims]
        if np.any((rows < 0) | (rows >= limits)):
            raise ValueError(
                f"PreparedOcctProjection.project.indices must stay within the corresponding entity counts; got indices={rows!r}, limits={limits!r}"
            )
        state = _Rows.empty(queries.shape[0])
        result_dims = dims.astype(np.int8, copy=True)
        result_rows = rows.astype(np.int32, copy=True)
        result_paths: list[tuple[str, ...]] = [()] * queries.shape[0]
        external_rows = np.full((queries.shape[0],), -1, dtype=np.int64)
        source_paths: list[tuple[str, ...] | None] = [None] * queries.shape[0]
        for row, (dimension, source_index) in enumerate(
            zip(dims.tolist(), rows.tolist(), strict=True)
        ):
            available = self.occt.entity_paths.get(
                (int(dimension), int(source_index)), ()
            )
            if occurrence_paths is None:
                if len(available) != 1:
                    result_dims[row] = -1
                    result_rows[row] = -1
                    state.status[row] = BRepProjectionStatus.UNRESOLVED
                    continue
                path = available[0]
            else:
                path = occurrence_paths[row]
                if path not in available:
                    raise ValueError(
                        "PreparedOcctProjection.occurrence_paths must select an actual source occurrence for every queried entity; "
                        f"got dimension={dimension}, index={source_index}, path={path!r}, available={available!r}."
                    )
            source_paths[row] = path
            result_paths[row] = path
            external = self.occt.entity_rows.get(
                (int(dimension), int(source_index), path)
            )
            if external is None:
                state.status[row] = BRepProjectionStatus.UNRESOLVED
                continue
            external_rows[row] = external
        world = self._world(queries)
        for row, external in enumerate(external_rows.tolist()):
            if external < 0:
                if state.status[row] != BRepProjectionStatus.UNRESOLVED:
                    state.status[row] = BRepProjectionStatus.UNRESOLVED
                continue
            dimension = int(dims[row])
            path = source_paths[row]
            if path is None:
                state.status[row] = BRepProjectionStatus.UNRESOLVED
                continue
            if dimension == BRepEntityDimension.VERTEX:
                point = np.asarray(self.vertex_points)[external]
                state.points[row] = point
            elif dimension == BRepEntityDimension.EDGE:
                self._project_edge(
                    external, world, np.asarray([row], dtype=np.int64), state
                )
            elif dimension == BRepEntityDimension.FACE:
                source_key = (
                    dimension,
                    int(rows[row]),
                    path,
                )
                orientation = self.occt.face_orientation.get(source_key)
                if orientation is None:
                    state.status[row] = BRepProjectionStatus.UNRESOLVED
                    continue
                self._project_face(
                    external,
                    world,
                    np.asarray([row], dtype=np.int64),
                    state,
                    orientation,
                    self.occt.face_poles.get(source_key, ()),
                )
            state.status[row] = BRepProjectionStatus.UNRESOLVED
            state.residuals[row] = np.inf
        return self._result(
            result_dims,
            result_rows,
            queries,
            state,
            tuple(result_paths),
        )

    def classify(
        self,
        points: ArrayLike,
        /,
        *,
        tolerance: float,
        maximum_dimension: int = 2,
        groups: ArrayLike | None = None,
        allowed_groups: ArrayLike | None = None,
        allowed_dimensions: ArrayLike | None = None,
        allowed_indices: ArrayLike | None = None,
        allowed_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Return unclassified rows when OCCT cannot certify native distances."""
        queries = self._queries(points, "Classification queries")
        positive_finite_float(tolerance, "tolerance")
        if isinstance(maximum_dimension, bool) or maximum_dimension not in (0, 1, 2):
            raise ValueError("maximum_dimension must be 0, 1, or 2.")
        filters = (
            groups,
            allowed_groups,
            allowed_dimensions,
            allowed_indices,
            allowed_occurrence_paths,
        )
        if any(value is not None for value in filters):
            if any(value is None for value in filters):
                raise ValueError(
                    "Occurrence-restricted classification requires all aligned filters."
                )
            assert groups is not None
            assert allowed_groups is not None
            assert allowed_dimensions is not None
            assert allowed_indices is not None
            assert allowed_occurrence_paths is not None
            group_rows = np.asarray(groups)
            allowed_group_rows = np.asarray(allowed_groups)
            dimension_rows = np.asarray(allowed_dimensions)
            index_rows = np.asarray(allowed_indices)
            if any(
                not np.issubdtype(values.dtype, np.integer)
                for values in (
                    group_rows,
                    allowed_group_rows,
                    dimension_rows,
                    index_rows,
                )
            ):
                raise TypeError("Classification filters must use integer arrays.")
            if (
                group_rows.shape != (queries.shape[0],)
                or allowed_group_rows.ndim != 1
                or dimension_rows.shape != allowed_group_rows.shape
                or index_rows.shape != allowed_group_rows.shape
                or len(allowed_occurrence_paths) != allowed_group_rows.size
            ):
                raise ValueError(
                    "Classification filters must be aligned to their queries and allowed entities."
                )
            if np.any((dimension_rows < 0) | (dimension_rows > 2)) or np.any(
                index_rows < 0
            ):
                raise ValueError(
                    "Allowed classification entities must select vertices, edges or faces."
                )
            if any(
                int(index) >= self.entity_counts[int(dimension)]
                for dimension, index in zip(dimension_rows, index_rows, strict=True)
            ):
                raise ValueError(
                    "An allowed classification index lies outside its B-Rep dimension."
                )
            if not isinstance(allowed_occurrence_paths, tuple) or any(
                not isinstance(path, tuple)
                or any(not isinstance(part, str) or not part for part in path)
                for path in allowed_occurrence_paths
            ):
                raise TypeError(
                    "allowed_occurrence_paths must contain exact path tuples."
                )
            for dimension, index, path in zip(
                dimension_rows,
                index_rows,
                allowed_occurrence_paths,
                strict=True,
            ):
                available = self.occt.entity_paths.get((int(dimension), int(index)), ())
                if path not in available:
                    raise ValueError(
                        "An allowed classification path must own the selected source entity."
                    )

        count = queries.shape[0]
        ambient = self.ambient_dimension
        return BRepProjectionResult(
            self.source_id,
            self.source_revision,
            np.full((count,), -1, dtype=np.int8),
            np.full((count,), -1, dtype=np.int32),
            queries,
            np.full((count, ambient), np.nan),
            np.full((count, 2), np.nan),
            np.full((count,), np.inf),
            np.full(
                (count,),
                BRepProjectionStatus.UNRESOLVED,
                dtype=np.int8,
            ),
            np.full((count, ambient), np.nan),
            np.full((count, 2, ambient), np.nan),
            occurrence_paths=((),) * count,
            frame_id=self.frame_id,
        )

    def _project_edge(
        self, edge: int, queries: np.ndarray, selected: np.ndarray, state: _Rows, /
    ) -> None:

        tolerance = self.policy.ambiguity_tolerance
        if bool(np.asarray(self.edge_degenerate)[edge]):
            # A degenerate edge is geometrically its single vertex.
            point = np.asarray(self.vertex_points)[int(self.edge_vertices[edge, 0])]
            state.points[selected] = point
            state.residuals[selected] = np.linalg.norm(queries[selected] - point, axis=1)
            state.status[selected] = BRepProjectionStatus.UNIQUE
            return
        curve = self.occt.curves[edge]
        first, last = (float(value) for value in np.asarray(self.edge_ranges)[edge])
        projector = occt.GeomAPI_ProjectPointOnCurve()
        projector.Init(curve, first, last)
        ends = np.stack((_xyz(curve.Value(first)), _xyz(curve.Value(last))))
        closed = bool(np.asarray(self.edge_closed)[edge])
        span = last - first
        for row in selected:
            query = queries[row]
            projector.Perform(occt.gp_Pnt(*query))
            count = projector.NbPoints()
            # Interior extrema plus the trimmed end points bound the closest point.
            candidates = np.concatenate(
                (
                    np.asarray(
                        [_xyz(projector.Point(i)) for i in range(1, count + 1)]
                    ).reshape(-1, 3),
                    ends,
                )
            )
            parameters = np.append(
                np.asarray([projector.Parameter(i) for i in range(1, count + 1)]),
                (first, last),
            )
            best, status = _select(query, candidates, tolerance)
            same = np.linalg.norm(candidates - candidates[best], axis=1) <= tolerance
            parameter = float(np.min(parameters[same]))
            if status == BRepProjectionStatus.UNIQUE and count == 0:
                # No isolated extremum: a constant distance around the answer is a
                # continuum of minima (e.g. a circle's center).
                probes = parameter + 1.0e-3 * span * np.asarray((-1.0, 1.0))
                probes = probes[(probes >= first) & (probes <= last)]
                distances = [np.linalg.norm(_xyz(curve.Value(p)) - query) for p in probes]
                if _flat(
                    distances,
                    float(np.linalg.norm(candidates[best] - query)),
                    tolerance,
                ):
                    status = BRepProjectionStatus.AMBIGUOUS
            if status == BRepProjectionStatus.UNIQUE and closed:
                seam = min(parameter - first, last - parameter) <= (
                    self.policy.parametric_tolerance * span
                )
                status = BRepProjectionStatus.SEAM if seam else status
            point = occt.gp_Pnt()
            derivative = occt.gp_Vec()
            curve.D1(parameter, point, derivative)
            state.points[row] = _xyz(point)
            state.parameters[row, 0] = parameter
            state.residuals[row] = float(np.linalg.norm(query - state.points[row]))
            state.status[row] = status
            state.first[row] = _xyz(derivative)

    def _project_face(
        self,
        face: int,
        queries: np.ndarray,
        selected: np.ndarray,
        state: _Rows,
        orientation: float,
        poles: tuple[int, ...],
        /,
    ) -> None:

        tolerance = self.policy.ambiguity_tolerance
        classifier_tolerance = self.policy.classifier_tolerance
        shape = self.occt.faces[face]
        surface = self.occt.surfaces[face]
        u0, u1, v0, v1 = (float(value) for value in np.asarray(self.face_bounds)[face])
        projector = occt.GeomAPI_ProjectPointOnSurf()
        projector.Init(surface, u0, u1, v0, v1, classifier_tolerance)
        classifier = occt.BRepClass_FaceClassifier()
        analysis = occt.ShapeAnalysis_Surface(surface)
        for row in selected:
            query = queries[row]
            projector.Perform(occt.gp_Pnt(*query))
            candidates, parameters = [], []
            for i in range(1, projector.NbPoints() + 1):
                u, v = projector.Parameters(i)
                classifier.Perform(shape, occt.gp_Pnt2d(u, v), classifier_tolerance)
                if classifier.State() in (occt.TopAbs_IN, occt.TopAbs_ON):
                    candidates.append(_xyz(projector.Point(i)))
                    parameters.append((u, v))
            if not candidates:
                # No surface extremum inside the trim (or a continuum of minima):
                # BRepExtrema evaluates the distance to the trimmed face.
                candidates, parameters = _trimmed_candidates(
                    shape, analysis, query, classifier_tolerance
                )
            if not candidates:
                continue
            points = np.asarray(candidates, dtype=np.float64)
            params = np.asarray(parameters, dtype=np.float64)
            best, status = _select(query, points, tolerance)
            same = np.linalg.norm(points - points[best], axis=1) <= tolerance
            u, v = params[same][np.lexsort(params[same].T[::-1])[0]]
            if status == BRepProjectionStatus.UNIQUE and projector.NbPoints() == 0:
                # No isolated surface extremum: constant distance along a parameter
                # direction is a continuum of minima (e.g. a point on a cylinder axis).
                distance = float(np.linalg.norm(points[best] - query))
                for step in ((1.0e-3 * (u1 - u0), 0.0), (0.0, 1.0e-3 * (v1 - v0))):
                    probes = [
                        _xyz(surface.Value(u + sign * step[0], v + sign * step[1]))
                        for sign in (-1.0, 1.0)
                    ]
                    if _flat(
                        [np.linalg.norm(probe - query) for probe in probes],
                        distance,
                        tolerance,
                    ):
                        status = BRepProjectionStatus.AMBIGUOUS
            status = self._face_status(face, points[best], u, v, status, poles)
            properties = occt.GeomLProp_SLProps(surface, u, v, 1, tolerance)
            state.points[row] = points[best]
            state.parameters[row] = (u, v)
            state.residuals[row] = float(np.linalg.norm(query - points[best]))
            state.status[row] = status
            state.first[row] = _xyz(properties.D1U())
            state.second[row] = _xyz(properties.D1V())
            if properties.IsNormalDefined():
                state.normals[row] = _xyz(properties.Normal()) * orientation

    def _face_status(
        self,
        face: int,
        point: np.ndarray,
        u: float,
        v: float,
        status: int,
        poles: tuple[int, ...],
        /,
    ) -> int:
        if status != BRepProjectionStatus.UNIQUE:
            return status
        u0, u1, v0, v1 = np.asarray(self.face_bounds)[face]
        periods = np.asarray(self.face_periods)[face]
        tolerance = self.policy.parametric_tolerance
        seam = bool(
            (np.isfinite(periods[0]) and min(u - u0, u1 - u) <= tolerance * periods[0])
            or (np.isfinite(periods[1]) and min(v - v0, v1 - v) <= tolerance * periods[1])
        )
        pole_points = np.asarray(self.vertex_points)[list(poles)]
        singular = pole_points.size > 0 and bool(
            np.any(
                np.linalg.norm(pole_points - point, axis=1)
                <= self.policy.ambiguity_tolerance
            )
        )
        return BRepProjectionStatus.SEAM if seam or singular else status

    def _result(
        self,
        dimensions: np.ndarray,
        indices: np.ndarray,
        queries: np.ndarray,
        state: _Rows,
        paths: tuple[tuple[str, ...], ...],
        /,
    ) -> BRepProjectionResult:
        count = dimensions.size
        tangents = np.full((count, 2, 3), np.nan)
        first_norm = np.linalg.norm(state.first, axis=1)
        usable = first_norm > 0.0
        tangents[usable, 0] = state.first[usable] / first_norm[usable, None]
        normal_norm = np.linalg.norm(state.normals, axis=1)
        face = (dimensions == BRepEntityDimension.FACE) & (normal_norm > 0.0)
        normals = np.full((count, 3), np.nan)
        normals[face] = state.normals[face] / normal_norm[face, None]
        # At a pole the first derivative vanishes: seed the frame with the other.
        second_norm = np.linalg.norm(state.second, axis=1)
        swap = face & ~usable & (second_norm > 0.0)
        tangents[swap, 0] = state.second[swap] / second_norm[swap, None]
        frame = face & np.all(np.isfinite(tangents[:, 0]), axis=1)
        tangents[frame, 0] -= (
            np.sum(tangents[frame, 0] * normals[frame], axis=1)[:, None] * normals[frame]
        )
        tangents[frame, 0] /= np.linalg.norm(tangents[frame, 0], axis=1)[:, None]
        tangents[frame, 1] = np.cross(normals[frame], tangents[frame, 0])
        points = np.where(
            np.isfinite(state.points), state.points, 0.0
        )  # Unresolved rows may retain diagnostic candidate points.
        ambient_points = self._ambient(points)
        ambient_points[~np.all(np.isfinite(state.points), axis=1)] = np.nan
        if self.embedding is None:
            ambient_normals = normals
        else:
            # Face normals leave the plane: not representable in planar ambient.
            ambient_normals = np.full((count, 2), np.nan)
        return BRepProjectionResult(
            self.source_id,
            self.source_revision,
            dimensions,
            indices,
            queries,
            ambient_points,
            state.parameters,
            state.residuals,
            state.status,
            ambient_normals,
            self._directions(tangents),
            occurrence_paths=paths,
            frame_id=self.frame_id,
        )

    def locate_solids(
        self,
        points: ArrayLike,
        /,
        *,
        occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
    ) -> BRepProjectionResult:
        """Return explicit unresolved OCCT containment observations."""

        queries = self._queries(points, "Solid location queries")
        if self.ambient_dimension != 3:
            raise ValueError(
                "PreparedOcctProjection.locate_solids requires a three-dimensional source frame."
            )
        if self.entity_counts[3] == 0:
            raise ValueError(
                "PreparedOcctProjection.locate_solids requires native solid entities."
            )
        if occurrence_paths is not None and (
            not isinstance(occurrence_paths, tuple)
            or len(occurrence_paths) != queries.shape[0]
            or any(
                not isinstance(path, tuple) or path not in self.occurrence_paths
                for path in occurrence_paths
            )
        ):
            raise ValueError(
                "PreparedOcctProjection.occurrence_paths must contain one actual solid occurrence path per query."
            )
        count = queries.shape[0]
        dimensions = np.full((count,), -1, dtype=np.int8)
        indices = np.full((count,), -1, dtype=np.int32)
        output_points = np.full((count, 3), np.nan)
        statuses = np.full((count,), BRepProjectionStatus.UNRESOLVED, dtype=np.int8)
        output_paths: list[tuple[str, ...]] = [()] * count
        solid_paths = {
            path: index
            for (dimension, index), paths in self.occt.entity_paths.items()
            if dimension == BRepEntityDimension.SOLID
            for path in paths
        }
        if occurrence_paths is not None:
            for row, path in enumerate(occurrence_paths):
                source_solid = solid_paths.get(path)
                if source_solid is None:
                    raise ValueError(
                        f"PreparedOcctProjection.occurrence_paths contains no solid entity for path={path!r}."
                    )
                dimensions[row] = BRepEntityDimension.SOLID
                indices[row] = source_solid
                output_paths[row] = path
        external_solid_offset = sum(self.occt.external_counts[:3])
        boxes = np.asarray(self.entity_boxes)[external_solid_offset:]
        if boxes.shape[0] and self.occt.solids:
            tolerance = self.policy.classifier_tolerance
            first = prepare_bvh(
                queries - tolerance, queries + tolerance, dtype=jnp.float64
            )
            second = prepare_bvh(boxes[:, 0], boxes[:, 1], dtype=jnp.float64)
            query_rows, solid_rows = bvh_overlap_pairs_host(
                first, second, include_touching=True
            )
            classifier = occt.BRepClass3d_SolidClassifier()
            matches: list[set[tuple[int, tuple[str, ...]]]] = [
                set() for _ in range(count)
            ]
            for row, solid_row in zip(query_rows, solid_rows, strict=True):
                external_row = int(solid_row)
                classifier.Load(self.occt.solids[external_row])
                classifier.Perform(occt.gp_Pnt(*queries[row]), tolerance)
                if classifier.State() not in (occt.TopAbs_IN, occt.TopAbs_ON):
                    continue
                sources = self.occt.solid_sources.get(external_row, ())
                if occurrence_paths is not None:
                    sources = tuple(
                        source for source in sources if source[1] == occurrence_paths[row]
                    )
                matches[int(row)].update(sources)
            for row, sources in enumerate(matches):
                if len(sources) == 1:
                    source_solid, path = next(iter(sources))
                    dimensions[row] = BRepEntityDimension.SOLID
                    indices[row] = source_solid
                    output_points[row] = queries[row]
                    output_paths[row] = path
        return BRepProjectionResult(
            self.source_id,
            self.source_revision,
            dimensions,
            indices,
            queries,
            output_points,
            np.full((count, 2), np.nan),
            np.full((count,), np.inf),
            statuses,
            np.full((count, 3), np.nan),
            np.full((count, 2, 3), np.nan),
            occurrence_paths=tuple(output_paths),
            frame_id=self.frame_id,
        )


def _flat(distances: list[float], reference: float, tolerance: float, /) -> bool:
    """Whether every probe distance equals the minimum (a continuum of minima)."""
    return bool(distances) and all(
        abs(float(value) - reference) <= tolerance for value in distances
    )


def _select(query: np.ndarray, candidates: np.ndarray, tolerance: float, /) -> Any:
    """Nearest candidate and UNIQUE/AMBIGUOUS status among distinct tied minima."""
    distances = np.linalg.norm(candidates - query, axis=1)
    best = int(np.argmin(distances))
    tied = distances <= distances[best] + tolerance
    distinct = np.linalg.norm(candidates[tied] - candidates[best], axis=1) > tolerance
    status = (
        BRepProjectionStatus.AMBIGUOUS
        if np.any(distinct)
        else BRepProjectionStatus.UNIQUE
    )
    return best, status


def _trimmed_candidates(
    face: Any, analysis: Any, query: np.ndarray, tolerance: float, /
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:

    extrema = occt.BRepExtrema_DistShapeShape(
        occt.BRepBuilderAPI_MakeVertex(occt.gp_Pnt(*query)).Vertex(), face
    )
    if not extrema.IsDone():
        return [], []
    points, parameters = [], []
    for solution in range(1, extrema.NbSolution() + 1):
        point = extrema.PointOnShape2(solution)
        uv = analysis.ValueOfUV(point, tolerance)
        points.append(_xyz(point))
        parameters.append((uv.X(), uv.Y()))
    return points, parameters


@checked
def prepare_occt_projection(
    model: BRepModel,
    source: str | Path | TopoDS_Shape,
    /,
    *,
    policy: BRepProjectionPolicy | None = None,
    embedding: PlanarEmbedding | None = None,
) -> PreparedOcctProjection:
    """Bind explicit external queries to the verified persisted source revision."""
    _ensure_occt()
    policy_ = BRepProjectionPolicy() if policy is None else policy
    return PreparedOcctProjection(model, _bound_shape(model, source), policy_, embedding)


__all__ = [
    "PreparedOcctProjection",
    "prepare_occt_projection",
]
