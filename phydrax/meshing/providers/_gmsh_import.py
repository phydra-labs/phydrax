#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Gmsh CAD source adaptation, session import cache, and stable entity resolution."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

import jax.numpy as jnp
import numpy as np


if TYPE_CHECKING:
    from OCP.TopoDS import TopoDS_Edge, TopoDS_Shape

from ...geometry.brep import BRepModel, BRepPartitionResult, BRepSource
from ...geometry.brep._occt import _explore_unique, read_occt_shape
from .._contracts import MeshingFailure, MeshingFailureCategory
from .._planar_bands import PlanarBandResult
from .._scope import MeshingEntityKind, MeshingScope
from .._trace import MeshingStageKind


_BRepMeshingSource = BRepModel | BRepSource | BRepPartitionResult | PlanarBandResult


def _brep_model(source: _BRepMeshingSource, /) -> BRepModel:
    if isinstance(source, PlanarBandResult):
        return source.partition.model
    if isinstance(source, BRepPartitionResult):
        return source.model
    if isinstance(source, BRepSource):
        return source.model
    if isinstance(source, BRepModel):
        return source
    raise TypeError(
        "source must be a BRepModel, BRepSource, BRepPartitionResult, or PlanarBandResult."
    )


def _entity_scope(source: BRepModel, dimension: int, identifiers: Any, /) -> MeshingScope:
    return MeshingScope(
        source.report.source_id,
        source.report.source_revision,
        MeshingEntityKind.GEOMETRY,
        dimension,
        f"{source.report.source_revision}:brep:{dimension}",
        np.asarray(identifiers, dtype=np.int64),
    )


def _source_scale(source: BRepModel, /) -> float:
    return max(float(np.ptp(np.asarray(source.mesh_vertices), axis=0).max()), 1.0)


@dataclass(slots=True)
class _CadImport:
    """One digest-verified CAD artifact owned by a Gmsh session."""

    digest: str
    snapshot: Path
    shape: object
    inventory: tuple[tuple[int, int], ...] | None
    entity_maps: dict[str, _CadEntityMap]


class _CadImportCache:
    """Session CAD imports keyed by source revision and coordinate contract.

    Entries own a private snapshot of the digest-verified source bytes, the parsed
    OCCT shape, and resolved Gmsh entity maps. Every acquisition re-digests the
    persisted source first, so replaced bytes evict the entry before any reuse,
    and Gmsh imports the verified snapshot rather than the mutable source path.
    """

    def __init__(self, capacity: int, /) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[tuple[str, str], _CadImport] = OrderedDict()
        self._workspace = TemporaryDirectory(prefix="phydrax-gmsh-cad-")
        self._created = 0
        self.hits = 0
        self.misses = 0

    @property
    def revisions(self) -> tuple[str, ...]:
        return tuple(sorted(revision for revision, _ in self._entries))

    def _evict(self, key: tuple[str, str], /) -> None:
        entry = self._entries.pop(key)
        entry.snapshot.unlink(missing_ok=True)

    def acquire(self, source: BRepModel, /) -> _CadImport:
        report = source.report
        path = Path(report.source_id)
        if not path.is_file():
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "Gmsh BRep meshing requires a reopenable STEP/IGES/BREP source path.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        key = (report.source_revision, source.coordinate_contract.spatial_id)
        entry = self._entries.get(key)
        if entry is not None and entry.digest != digest:
            self._evict(key)
            entry = None
        if digest != report.source_digest:
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The persisted BRep source bytes changed after import.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        if entry is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return entry
        self.misses += 1
        self._created += 1
        snapshot = Path(self._workspace.name) / f"{self._created}{path.suffix.lower()}"
        snapshot.write_bytes(payload)
        shape, source_format, snapshot_digest = read_occt_shape(snapshot)
        if snapshot_digest != digest or source_format != report.source_format:
            snapshot.unlink(missing_ok=True)
            raise MeshingFailure(
                MeshingFailureCategory.INVALID_SOURCE,
                "The persisted BRep source bytes changed after import.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )
        entry = _CadImport(digest, snapshot, shape, None, {})
        self._entries[key] = entry
        while len(self._entries) > self._capacity:
            self._evict(next(iter(self._entries)))
        return entry

    def import_shapes(self, gmsh: Any, entry: _CadImport, /) -> None:
        # Free curves and points must survive import so protection can embed them.
        gmsh.model.occ.importShapes(str(entry.snapshot), highestDimOnly=False)
        gmsh.model.occ.synchronize()
        inventory = tuple(
            (int(dimension), int(tag)) for dimension, tag in gmsh.model.getEntities()
        )
        if entry.inventory is None:
            entry.inventory = inventory
        elif entry.inventory != inventory:
            raise MeshingFailure(
                MeshingFailureCategory.PROVIDER_EXECUTION_FAILED,
                "Re-importing verified CAD bytes changed the Gmsh entity inventory.",
                stage=MeshingStageKind.SOURCE_INSPECTION.value,
            )

    def entity_map(
        self,
        entry: _CadImport,
        kind: str,
        resolve: Callable[[], _CadEntityMap],
        /,
    ) -> _CadEntityMap:
        resolved = entry.entity_maps.get(kind)
        if resolved is None:
            resolved = resolve()
            entry.entity_maps[kind] = resolved
        return resolved

    def close(self) -> None:
        self._entries.clear()
        self._workspace.cleanup()


@runtime_checkable
class _TopoDSEdgeCaster(Protocol):
    @staticmethod
    def Edge_s(shape: TopoDS_Shape, /) -> TopoDS_Edge: ...


def _scope_samples(
    source: BRepModel, shape: Any, scope: MeshingScope, /
) -> tuple[np.ndarray, ...]:
    """Sample stable source entities independently of Gmsh's import tag numbering."""
    ids = np.asarray(scope.entity_ids, dtype=np.int64)
    counts = {
        0: source.report.num_vertices,
        1: source.report.num_edges,
        2: source.report.num_faces,
    }
    if (
        scope.entity_kind is not MeshingEntityKind.GEOMETRY
        or scope.entity_dimension not in counts
        or scope.source_id != source.report.source_id
        or scope.source_revision != source.report.source_revision
        or np.any(ids >= counts[scope.entity_dimension])
    ):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Scope does not select an entity of the supplied BRep revision.",
        )
    if scope.entity_dimension == 2:
        face_ids = np.asarray(source.triangle_face_ids)
        parameters = np.asarray(source.triangle_parameters)
        result = []
        for face in ids:
            triangles = np.flatnonzero(face_ids == face)
            if not triangles.size:
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "Source face has no interior samples for entity resolution.",
                )
            selected = triangles[
                np.linspace(0, len(triangles) - 1, min(3, len(triangles)), dtype=np.int64)
            ]
            uv = np.mean(parameters[selected], axis=1)
            result.append(np.asarray(source.patches[int(face)].evaluate(jnp.asarray(uv))))
        return tuple(result)
    from OCP.BRep import BRep_Tool
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_VERTEX
    from OCP.TopoDS import TopoDS

    if scope.entity_dimension == 0:
        vertices = _explore_unique(shape, TopAbs_VERTEX, TopoDS.Vertex)
        result = []
        for vertex in ids:
            point = BRep_Tool.Pnt_s(vertices[int(vertex)])
            result.append(np.asarray(((point.X(), point.Y(), point.Z()),)))
        return tuple(result)
    if not isinstance(TopoDS, _TopoDSEdgeCaster):
        raise MeshingFailure(
            MeshingFailureCategory.PROVIDER_UNAVAILABLE,
            "The CAD kernel must expose the TopoDS.Edge_s edge downcast.",
        )
    edges = _explore_unique(shape, TopAbs_EDGE, TopoDS.Edge)
    result = []
    for edge in ids:
        curve = BRepAdaptor_Curve(edges[int(edge)])
        parameters = np.linspace(curve.FirstParameter(), curve.LastParameter(), 5)[1:-1]
        values = [curve.Value(float(value)) for value in parameters]
        result.append(np.asarray([(point.X(), point.Y(), point.Z()) for point in values]))
    return tuple(result)


def _match_entities(
    gmsh: Any, dimension: int, samples: Any, candidates: Any, tolerance: float, /
) -> tuple[int, ...]:
    result = []
    for points in samples:
        matches = []
        for tag in candidates:
            if tag in result:
                continue
            closest, _ = gmsh.model.getClosestPoint(
                dimension, tag, np.asarray(points).reshape(-1)
            )
            closest = np.asarray(closest).reshape((-1, 3))
            if (
                closest.shape == points.shape
                and np.max(np.linalg.norm(closest - points, axis=1)) <= tolerance
                and gmsh.model.isInside(dimension, tag, closest.reshape(-1))
                == len(points)
            ):
                matches.append(tag)
        if len(matches) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "BRep entity does not resolve uniquely to the imported Gmsh geometry.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        result.append(matches[0])
    return tuple(result)


def _resolve_entities(
    gmsh: Any, source: Any, shape: Any, scope: Any, /
) -> tuple[int, ...]:
    samples = _scope_samples(source, shape, scope)
    return _match_entities(
        gmsh,
        scope.entity_dimension,
        samples,
        [tag for _, tag in gmsh.model.getEntities(scope.entity_dimension)],
        1.0e-7 * _source_scale(source),
    )


@dataclass(frozen=True, slots=True)
class _CadEntityMap:
    face_to_surface: tuple[int, ...]
    solid_to_volume: tuple[int, ...]
    edge_to_curve: tuple[int, ...] = ()


def _validate_source_solids(source: BRepModel, shape: Any, /) -> tuple[Any, ...]:
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepExtrema import BRepExtrema_DistShapeShape
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopoDS import TopoDS

    topology = source.topology
    if (
        any(not faces for faces in topology.solid_faces)
        or any(len(owners) not in (1, 2) for owners in topology.face_solids)
        or any(
            orientations[0] == orientations[1]
            for face_index, owners in enumerate(topology.face_solids)
            if len(owners) == 2
            for orientations in (
                tuple(
                    topology.solid_face_orientations[solid][
                        topology.solid_faces[solid].index(face_index)
                    ]
                    for solid in owners
                ),
            )
        )
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Source solids require nonempty manifold boundaries with opposite shared-face orientations.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    solids = _explore_unique(shape, TopAbs_SOLID, TopoDS.Solid)
    if len(solids) != topology.num_solids or any(
        not BRepCheck_Analyzer(solid).IsValid() for solid in solids
    ):
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Reopened BRep solids do not match the valid imported solid inventory.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    scale = _source_scale(source)
    volume_tolerance = 1.0e-12 * scale**3
    contact_tolerance = 1.0e-9 * scale
    for left_index, left in enumerate(solids):
        left_faces = set(topology.solid_faces[left_index])
        for right_index, right in enumerate(
            solids[left_index + 1 :], start=left_index + 1
        ):
            distance = BRepExtrema_DistShapeShape(left, right)
            distance.Perform()
            if not distance.IsDone():
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid contact validation did not complete.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            shared_faces = left_faces & set(topology.solid_faces[right_index])
            if not shared_faces and float(distance.Value()) <= contact_tolerance:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solids touch without a complete shared topological face.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            common = BRepAlgoAPI_Common(left, right)
            common.Build()
            if not common.IsDone():
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid overlap validation did not complete.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
            properties = GProp_GProps()
            BRepGProp.VolumeProperties_s(common.Shape(), properties)
            if abs(float(properties.Mass())) > volume_tolerance:
                raise MeshingFailure(
                    MeshingFailureCategory.INVALID_SOURCE,
                    "BRep solid interiors overlap; provider-side ownership fragmentation is prohibited.",
                    stage=MeshingStageKind.SOURCE_INSPECTION.value,
                )
    return tuple(solids)


def _resolve_planar_cad_entity_map(
    gmsh: Any, source: BRepModel, shape: Any, /
) -> _CadEntityMap:
    if source.topology.num_solids:
        raise MeshingFailure(
            MeshingFailureCategory.INVALID_SOURCE,
            "Strict planar CAD meshing requires a zero-solid BRep.",
            stage=MeshingStageKind.SOURCE_INSPECTION.value,
        )
    face_tags = _resolve_entities(
        gmsh,
        source,
        shape,
        _entity_scope(source, 2, np.arange(source.report.num_faces)),
    )
    edge_tags = _resolve_entities(
        gmsh,
        source,
        shape,
        _entity_scope(source, 1, np.arange(source.report.num_edges)),
    )
    if set(face_tags) != {tag for _, tag in gmsh.model.getEntities(2)}:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh surfaces are not a bijection with planar BRep faces.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    if set(edge_tags) != {tag for _, tag in gmsh.model.getEntities(1)}:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh curves are not a bijection with planar BRep edges.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    for face_index, surface in enumerate(face_tags):
        boundary = gmsh.model.getBoundary(
            [(2, surface)], combined=False, oriented=False, recursive=False
        )
        actual = {abs(int(tag)) for dimension, tag in boundary if dimension == 1}
        expected = {edge_tags[index] for index in source.topology.face_edges[face_index]}
        if len(actual) != len(boundary) or actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh planar surface boundaries differ from source face-edge incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
    for edge_index, curve in enumerate(edge_tags):
        upward, _ = gmsh.model.getAdjacencies(1, curve)
        actual = {int(value) for value in np.asarray(upward, dtype=np.int64)}
        expected = {face_tags[index] for index in source.topology.edge_faces[edge_index]}
        if actual != expected:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh curve-to-surface adjacency differs from source planar incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
    return _CadEntityMap(tuple(face_tags), (), tuple(edge_tags))


def _volume_boundaries(
    gmsh: Any, volume_tags: tuple[int, ...], /
) -> tuple[dict[int, frozenset[int]], dict[tuple[int, int], int]]:
    boundary_faces: dict[int, frozenset[int]] = {}
    boundary_orientations: dict[tuple[int, int], int] = {}
    for volume in volume_tags:
        occurrences = gmsh.model.getBoundary(
            [(3, volume)], combined=False, oriented=True, recursive=False
        )
        if any(dimension != 2 for dimension, _ in occurrences):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An imported Gmsh volume has a non-surface boundary occurrence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        tags = tuple(abs(int(tag)) for _, tag in occurrences)
        if len(set(tags)) != len(tags):
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "An imported Gmsh volume repeats a boundary surface.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        boundary_faces[volume] = frozenset(tags)
        boundary_orientations.update(
            (
                ((volume, abs(int(tag))), 1 if int(tag) > 0 else -1)
                for _, tag in occurrences
            )
        )
    return boundary_faces, boundary_orientations


def _resolve_cad_entity_map(gmsh: Any, source: BRepModel, shape: Any, /) -> _CadEntityMap:
    _validate_source_solids(source, shape)
    face_tags = _resolve_entities(
        gmsh,
        source,
        shape,
        _entity_scope(source, 2, np.arange(source.report.num_faces)),
    )
    imported_surfaces = {tag for _, tag in gmsh.model.getEntities(2)}
    if set(face_tags) != imported_surfaces:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh surfaces are not a bijection with source BRep faces.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )

    volume_tags = tuple(tag for _, tag in sorted(gmsh.model.getEntities(3)))
    if len(volume_tags) != source.topology.num_solids:
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh volumes do not match the source solid inventory.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )
    boundary_faces, boundary_orientations = _volume_boundaries(gmsh, volume_tags)

    solid_to_volume = []
    used_volumes = set()
    for faces in source.topology.solid_faces:
        expected = frozenset(face_tags[face] for face in faces)
        candidates = tuple(
            volume
            for volume, actual in boundary_faces.items()
            if actual == expected and volume not in used_volumes
        )
        if len(candidates) != 1:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "A source solid does not resolve uniquely by exact boundary-face incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        solid_to_volume.append(candidates[0])
        used_volumes.add(candidates[0])
    if used_volumes != set(volume_tags):
        raise MeshingFailure(
            MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
            "Imported Gmsh volume ownership is not a source-solid bijection.",
            stage=MeshingStageKind.SCOPE_RESOLUTION.value,
        )

    for face_index, surface in enumerate(face_tags):
        source_owners = source.topology.face_solids[face_index]
        expected_volumes = {solid_to_volume[owner] for owner in source_owners}
        upward, _ = gmsh.model.getAdjacencies(2, surface)
        actual_volumes = {int(value) for value in np.asarray(upward, dtype=np.int64)}
        if actual_volumes != expected_volumes:
            raise MeshingFailure(
                MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                "Gmsh surface-to-volume adjacency differs from source BRep incidence.",
                stage=MeshingStageKind.SCOPE_RESOLUTION.value,
            )
        if len(source_owners) == 2:
            first, second = tuple(expected_volumes)
            if (
                boundary_orientations[first, surface]
                == boundary_orientations[second, surface]
            ):
                raise MeshingFailure(
                    MeshingFailureCategory.SCOPE_RESOLUTION_FAILED,
                    "A shared Gmsh surface has inconsistent volume orientations.",
                    stage=MeshingStageKind.SCOPE_RESOLUTION.value,
                )
    return _CadEntityMap(tuple(face_tags), tuple(solid_to_volume))
