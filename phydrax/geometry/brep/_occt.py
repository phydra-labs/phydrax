#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from OCP.BRep import BRep_Tool  # ty: ignore[unresolved-import]
from OCP.BRepAdaptor import (
    BRepAdaptor_Curve2d,  # ty: ignore[unresolved-import]
    BRepAdaptor_Surface,  # ty: ignore[unresolved-import]
)
from OCP.BRepMesh import BRepMesh_IncrementalMesh  # ty: ignore[unresolved-import]
from OCP.BRepTools import (
    BRepTools,  # ty: ignore[unresolved-import]
    BRepTools_WireExplorer,  # ty: ignore[unresolved-import]
)
from OCP.Geom import Geom_RectangularTrimmedSurface  # ty: ignore[unresolved-import]
from OCP.GeomAbs import (
    GeomAbs_BSplineSurface,  # ty: ignore[unresolved-import]
    GeomAbs_Cone,  # ty: ignore[unresolved-import]
    GeomAbs_Cylinder,  # ty: ignore[unresolved-import]
    GeomAbs_Plane,  # ty: ignore[unresolved-import]
    GeomAbs_Sphere,  # ty: ignore[unresolved-import]
    GeomAbs_Torus,  # ty: ignore[unresolved-import]
)
from OCP.GeomConvert import GeomConvert  # ty: ignore[unresolved-import]
from OCP.IFSelect import IFSelect_RetDone  # ty: ignore[unresolved-import]
from OCP.IGESControl import IGESControl_Reader  # ty: ignore[unresolved-import]
from OCP.STEPControl import STEPControl_Reader  # ty: ignore[unresolved-import]
from OCP.TopAbs import (
    TopAbs_EDGE,  # ty: ignore[unresolved-import]
    TopAbs_FACE,  # ty: ignore[unresolved-import]
    TopAbs_REVERSED,  # ty: ignore[unresolved-import]
    TopAbs_VERTEX,  # ty: ignore[unresolved-import]
    TopAbs_WIRE,  # ty: ignore[unresolved-import]
)
from OCP.TopExp import TopExp_Explorer  # ty: ignore[unresolved-import]
from OCP.TopLoc import TopLoc_Location  # ty: ignore[unresolved-import]
from OCP.TopoDS import TopoDS, TopoDS_Shape  # ty: ignore[unresolved-import]

from .._atlas import TrimDomain
from ._model import BRepImportReport, BRepModel, BRepTopology
from ._patches import (
    BSplineSurfacePatch,
    ConePatch,
    CylinderPatch,
    PlanePatch,
    SpherePatch,
    TorusPatch,
)


def _xyz(value: Any) -> np.ndarray:
    return np.asarray((value.X(), value.Y(), value.Z()), dtype=float)


def _xy(value: Any) -> np.ndarray:
    return np.asarray((value.X(), value.Y()), dtype=float)


def _frame(position: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        _xyz(position.Location()),
        _xyz(position.XDirection()),
        _xyz(position.YDirection()),
        _xyz(position.Direction()),
    )


def _explore(shape: TopoDS_Shape, kind: Any, caster) -> list[Any]:
    explorer = TopExp_Explorer(shape, kind)
    entities: list[Any] = []
    while explorer.More():
        entities.append(caster(explorer.Current()))
        explorer.Next()
    return entities


def _explore_unique(shape: TopoDS_Shape, kind: Any, caster) -> list[Any]:
    entities: list[Any] = []
    for candidate in _explore(shape, kind, caster):
        if not any(entity.IsSame(candidate) for entity in entities):
            entities.append(candidate)
    return entities


def _shape_index(entities: list[Any], candidate: Any) -> int:
    for index, entity in enumerate(entities):
        if entity.IsSame(candidate):
            return index
    raise RuntimeError("OCCT returned an entity absent from the global topology map.")


def _expanded_knots(surface: Any, axis: str) -> np.ndarray:
    count = surface.NbUKnots() if axis == "u" else surface.NbVKnots()
    values: list[float] = []
    for index in range(1, count + 1):
        knot = surface.UKnot(index) if axis == "u" else surface.VKnot(index)
        multiplicity = (
            surface.UMultiplicity(index) if axis == "u" else surface.VMultiplicity(index)
        )
        values.extend([float(knot)] * int(multiplicity))
    return np.asarray(values, dtype=float)


def _bspline_patch(surface: Any) -> BSplineSurfacePatch:
    num_u = int(surface.NbUPoles())
    num_v = int(surface.NbVPoles())
    control_points = np.empty((num_u, num_v, 3), dtype=float)
    weights = np.empty((num_u, num_v), dtype=float)
    for u_index in range(1, num_u + 1):
        for v_index in range(1, num_v + 1):
            control_points[u_index - 1, v_index - 1] = _xyz(
                surface.Pole(u_index, v_index)
            )
            weights[u_index - 1, v_index - 1] = float(surface.Weight(u_index, v_index))
    return BSplineSurfacePatch(
        control_points,
        weights,
        _expanded_knots(surface, "u"),
        _expanded_knots(surface, "v"),
        int(surface.UDegree()),
        int(surface.VDegree()),
    )


def _surface_patch(face: Any, bounds: np.ndarray):
    adaptor = BRepAdaptor_Surface(face, True)
    surface_type = adaptor.GetType()
    if surface_type == GeomAbs_Plane:
        origin, first, second, _ = _frame(adaptor.Plane().Position())
        return PlanePatch(origin, first, second), "plane", False
    if surface_type == GeomAbs_Cylinder:
        cylinder = adaptor.Cylinder()
        origin, first, second, axis = _frame(cylinder.Position())
        return (
            CylinderPatch(origin, first, second, axis, cylinder.Radius()),
            "cylinder",
            False,
        )
    if surface_type == GeomAbs_Cone:
        cone = adaptor.Cone()
        origin, first, second, axis = _frame(cone.Position())
        return (
            ConePatch(
                origin,
                first,
                second,
                axis,
                cone.RefRadius(),
                cone.SemiAngle(),
            ),
            "cone",
            False,
        )
    if surface_type == GeomAbs_Sphere:
        sphere = adaptor.Sphere()
        center, first, second, axis = _frame(sphere.Position())
        return (
            SpherePatch(center, first, second, axis, sphere.Radius()),
            "sphere",
            False,
        )
    if surface_type == GeomAbs_Torus:
        torus = adaptor.Torus()
        center, first, second, axis = _frame(torus.Position())
        return (
            TorusPatch(
                center,
                first,
                second,
                axis,
                torus.MajorRadius(),
                torus.MinorRadius(),
            ),
            "torus",
            False,
        )
    if surface_type == GeomAbs_BSplineSurface:
        return _bspline_patch(adaptor.BSpline()), "bspline", False

    surface = BRep_Tool.Surface_s(face)
    trimmed = Geom_RectangularTrimmedSurface(
        surface,
        float(bounds[0, 0]),
        float(bounds[1, 0]),
        float(bounds[0, 1]),
        float(bounds[1, 1]),
    )
    return (
        _bspline_patch(GeomConvert.SurfaceToBSplineSurface_s(trimmed)),
        "converted_bspline",
        True,
    )


def _ordered_wires(face: Any) -> list[Any]:
    return _explore(face, TopAbs_WIRE, TopoDS.Wire_s)


def _wire_edge_indices(wire: Any, face: Any, edges: list[Any]) -> tuple[int, ...]:
    explorer = BRepTools_WireExplorer(wire, face)
    result: list[int] = []
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        index = _shape_index(edges, edge)
        sign = -1 if edge.Orientation() == TopAbs_REVERSED else 1
        result.append(sign * (index + 1))
        explorer.Next()
    return tuple(result)


def _sample_wire(wire: Any, face: Any, samples_per_edge: int) -> np.ndarray | None:
    explorer = BRepTools_WireExplorer(wire, face)
    segments: list[np.ndarray] = []
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        curve = BRepAdaptor_Curve2d(edge, face)
        start = float(curve.FirstParameter())
        end = float(curve.LastParameter())
        if edge.Orientation() == TopAbs_REVERSED:
            start, end = end, start
        parameters = np.linspace(start, end, samples_per_edge, endpoint=False)
        segments.append(
            np.stack([_xy(curve.Value(float(value))) for value in parameters])
        )
        explorer.Next()
    if not segments:
        return None
    points = np.concatenate(segments, axis=0)
    if points.shape[0] < 3 or not np.all(np.isfinite(points)):
        return None
    keep = np.ones((points.shape[0],), dtype=bool)
    keep[1:] = np.linalg.norm(points[1:] - points[:-1], axis=1) > 1e-13
    points = points[keep]
    return points if points.shape[0] >= 3 else None


def _normalized_trim_domain(
    face: Any,
    bounds: np.ndarray,
    samples_per_edge: int,
) -> TrimDomain | None:
    outer_wire = BRepTools.OuterWire_s(face)
    if outer_wire.IsNull():
        return None
    outer = _sample_wire(outer_wire, face, samples_per_edge)
    if outer is None:
        return None
    scale = bounds[1] - bounds[0]
    normalized_outer = (outer - bounds[0]) / scale
    holes: list[np.ndarray] = []
    for wire in _ordered_wires(face):
        if wire.IsSame(outer_wire):
            continue
        loop = _sample_wire(wire, face, samples_per_edge)
        if loop is not None:
            holes.append((loop - bounds[0]) / scale)
    return TrimDomain(normalized_outer, holes)


def _extract_topology(shape: Any, faces: list[Any]) -> tuple[BRepTopology, list[Any]]:
    edges = _explore_unique(shape, TopAbs_EDGE, TopoDS.Edge_s)
    vertices = _explore_unique(shape, TopAbs_VERTEX, TopoDS.Vertex_s)
    face_edges: list[tuple[int, ...]] = []
    face_wires: list[tuple[tuple[int, ...], ...]] = []
    edge_faces: list[list[int]] = [[] for _ in edges]
    for face_index, face in enumerate(faces):
        wires = tuple(
            _wire_edge_indices(wire, face, edges) for wire in _ordered_wires(face)
        )
        face_wires.append(wires)
        unique_edges = tuple(
            dict.fromkeys(abs(index) - 1 for wire in wires for index in wire)
        )
        face_edges.append(unique_edges)
        for edge_index in unique_edges:
            edge_faces[edge_index].append(face_index)
    return (
        BRepTopology(
            face_edges=tuple(face_edges),
            edge_faces=tuple(tuple(indices) for indices in edge_faces),
            face_wires=tuple(face_wires),
            num_vertices=len(vertices),
        ),
        edges,
    )


def _extract_tessellation(
    shape: Any,
    faces: list[Any],
    *,
    linear_deflection: float,
    angular_deflection: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    BRepMesh_IncrementalMesh(
        shape,
        linear_deflection,
        False,
        angular_deflection,
        False,
    )
    scale = max(linear_deflection * 1e-7, np.finfo(float).eps * 128.0)
    vertex_lookup: dict[tuple[int, int, int], int] = {}
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_face_ids: list[int] = []
    triangle_parameters: list[np.ndarray] = []
    for face_index, face in enumerate(faces):
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation_s(face, location)
        if triangulation is None or triangulation.NbTriangles() == 0:
            raise ValueError(
                f"OCCT produced no query tessellation for face {face_index}."
            )
        transformation = location.Transformation()
        local_vertices: list[np.ndarray] = []
        local_parameters: list[np.ndarray] = []
        for node_index in range(1, triangulation.NbNodes() + 1):
            local_vertices.append(
                _xyz(triangulation.Node(node_index).Transformed(transformation))
            )
            if not triangulation.HasUVNodes():
                raise ValueError("OCCT triangulation lacks required face parameters.")
            local_parameters.append(_xy(triangulation.UVNode(node_index)))
        reversed_face = face.Orientation() == TopAbs_REVERSED
        for triangle_index in range(1, triangulation.NbTriangles() + 1):
            local = [index - 1 for index in triangulation.Triangle(triangle_index).Get()]
            if reversed_face:
                local[1], local[2] = local[2], local[1]
            global_indices: list[int] = []
            for local_index in local:
                point = local_vertices[local_index]
                key = tuple(np.rint(point / scale).astype(np.int64).tolist())
                global_index = vertex_lookup.get(key)
                if global_index is None:
                    global_index = len(vertices)
                    vertex_lookup[key] = global_index
                    vertices.append(point)
                global_indices.append(global_index)
            if len(set(global_indices)) < 3:
                continue
            triangles.append(
                (global_indices[0], global_indices[1], global_indices[2])
            )
            triangle_face_ids.append(face_index)
            triangle_parameters.append(
                np.stack([local_parameters[local_index] for local_index in local])
            )
    return (
        np.stack(vertices),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(triangle_face_ids, dtype=np.int32),
        np.stack(triangle_parameters),
    )


def _shape_revision(shape: Any) -> str:
    temporary = Path(tempfile.gettempdir()) / f"phydrax-brep-{id(shape)}.brep"
    written = BRepTools.Write_s(shape, str(temporary))
    if not written:
        raise RuntimeError("OCCT could not serialize the shape for revision hashing.")
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    temporary.unlink()
    return digest


def read_occt_shape(path: str | Path) -> tuple[Any, str, str]:
    """Read STEP, IGES, or native BREP without intermediate mesh conversion."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower()
    if suffix in {".step", ".stp"}:
        reader = STEPControl_Reader()
        if reader.ReadFile(str(source)) != IFSelect_RetDone:
            raise ValueError(f"OCCT failed to read STEP file {source}.")
        reader.TransferRoots()
        shape = reader.OneShape()
        source_format = "step"
    elif suffix in {".iges", ".igs"}:
        reader = IGESControl_Reader()
        if reader.ReadFile(str(source)) != IFSelect_RetDone:
            raise ValueError(f"OCCT failed to read IGES file {source}.")
        reader.TransferRoots()
        shape = reader.OneShape()
        source_format = "iges"
    elif suffix in {".brep", ".brp"}:
        from OCP.BRep import BRep_Builder  # ty: ignore[unresolved-import]

        shape = TopoDS_Shape()
        if not BRepTools.Read_s(shape, str(source), BRep_Builder()):
            raise ValueError(f"OCCT failed to read BREP file {source}.")
        source_format = "brep"
    else:
        raise ValueError("BRep import supports .step/.stp, .iges/.igs, and .brep/.brp.")
    if shape.IsNull():
        raise ValueError(f"CAD file {source} contains no transferable shape.")
    return shape, source_format, hashlib.sha256(source.read_bytes()).hexdigest()


def model_from_occt_shape(
    shape: Any,
    *,
    source_id: str = "occt-shape",
    source_revision: str | None = None,
    source_format: str = "occt",
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> BRepModel:
    """Extract exact patches/topology and a reported watertight query mesh."""

    if shape.IsNull():
        raise ValueError("Cannot import a null OCCT shape.")
    if linear_deflection <= 0.0 or angular_deflection <= 0.0:
        raise ValueError("Meshing deflections must be positive.")
    if trim_samples_per_edge < 3:
        raise ValueError("trim_samples_per_edge must be at least three.")
    faces = _explore_unique(shape, TopAbs_FACE, TopoDS.Face_s)
    if not faces:
        raise ValueError("The OCCT shape contains no faces.")
    topology, edges = _extract_topology(shape, faces)
    patches = []
    parameter_bounds = []
    orientations = []
    trim_domains = []
    tags = []
    converted_count = 0
    for face in faces:
        u_min, u_max, v_min, v_max = BRepTools.UVBounds_s(face)
        bounds = np.asarray(((u_min, v_min), (u_max, v_max)), dtype=float)
        if not np.all(np.isfinite(bounds)) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("Every imported face must have finite nonempty UV bounds.")
        patch, tag, converted = _surface_patch(face, bounds)
        patches.append(patch)
        parameter_bounds.append(bounds)
        orientations.append(-1.0 if face.Orientation() == TopAbs_REVERSED else 1.0)
        trim_domains.append(_normalized_trim_domain(face, bounds, trim_samples_per_edge))
        tags.append(tag)
        converted_count += int(converted)
    vertices, mesh_faces, triangle_face_ids, triangle_parameters = _extract_tessellation(
        shape,
        faces,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
    )
    revision = source_revision or _shape_revision(shape)
    report = BRepImportReport(
        source_id=source_id,
        source_revision=revision,
        source_format=source_format,
        num_faces=len(faces),
        num_edges=len(edges),
        num_vertices=topology.num_vertices,
        num_triangles=mesh_faces.shape[0],
        linear_deflection=float(linear_deflection),
        angular_deflection=float(angular_deflection),
        trim_samples_per_edge=int(trim_samples_per_edge),
        converted_surface_count=converted_count,
    )
    return BRepModel(
        patches=tuple(patches),
        parameter_bounds=np.asarray(parameter_bounds),
        orientation=np.asarray(orientations),
        trim_domains=tuple(trim_domains),
        topology=topology,
        mesh_vertices=vertices,
        mesh_faces=mesh_faces,
        triangle_face_ids=triangle_face_ids,
        triangle_parameters=triangle_parameters,
        source_id=source_id,
        source_revision=revision,
        physical_tags=tuple(tags),
        report=report,
    )


def import_brep(
    path: str | Path,
    *,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> BRepModel:
    """Import direct CAD topology, geometry, trim charts, and query tessellation."""

    shape, source_format, revision = read_occt_shape(path)
    source_id = str(Path(path).expanduser().resolve())
    return model_from_occt_shape(
        shape,
        source_id=source_id,
        source_revision=revision,
        source_format=source_format,
        linear_deflection=linear_deflection,
        angular_deflection=angular_deflection,
        trim_samples_per_edge=trim_samples_per_edge,
    )


__all__ = ["import_brep", "model_from_occt_shape", "read_occt_shape"]
