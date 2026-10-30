#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Geometry-preserving topology repair of authored open triangle sheets."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from fractions import Fraction
from numbers import Integral
from typing import final, NoReturn

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...meshing._contracts import (
    MeshingFailure,
    MeshingFailureCategory,
    MeshingFailureEvidence,
)
from .._meshing_domain import MeshingDomain
from ._core import (
    _persistent_int64_id,
    _reconstructed_scope_banks,
    _scope_entity_ids_by_mesh_row,
    _triangle_complex_meshing_domain,
    ReconstructionReport,
)


@dataclass(frozen=True, slots=True)
class TriangleSheetRepairEvidence:
    """Exact counts of one geometry-preserving authored-sheet topology repair.

    Welding merges only bitwise-coincident coordinates, duplicate removal drops
    only triangles over an identical welded vertex set, and orientation flips
    reverse a triangle's cyclic order without moving any point. The union of
    repaired triangles is therefore exactly the authored point set; boundary
    edges remain open.
    """

    authored_vertices: int
    authored_triangles: int
    repaired_vertices: int
    repaired_triangles: int
    repaired_edges: int
    welded_vertices: int
    removed_duplicate_triangles: int
    flipped_triangles: int
    boundary_edges: int
    components: int

    @property
    def quantities(self) -> tuple[tuple[str, int], ...]:
        return tuple(asdict(self).items())


def _refuse(
    message: str,
    finding: str,
    triangles: tuple[int, ...],
    locations: tuple[tuple[float, ...], ...],
    /,
) -> NoReturn:
    raise MeshingFailure(
        MeshingFailureCategory.INVALID_SOURCE,
        message,
        provider_code="authored_triangle_sheet",
        stage="topology_repair",
        entity_ids=triangles,
        locations=locations,
        evidence=MeshingFailureEvidence(
            achieved_quantities=((f"{finding}_triangles", float(len(triangles))),),
            checkpoint="authored_sheet_topology_repair",
            logical_findings=(finding,),
        ),
    )


def _exactly_collinear(points: np.ndarray, /) -> bool:
    exact = tuple(tuple(Fraction.from_float(float(value)) for value in row) for row in points)
    first = tuple(exact[1][axis] - exact[0][axis] for axis in range(3))
    second = tuple(exact[2][axis] - exact[0][axis] for axis in range(3))
    return (
        first[1] * second[2] == first[2] * second[1]
        and first[2] * second[0] == first[0] * second[2]
        and first[0] * second[1] == first[1] * second[0]
    )


def _repair_topology(
    vertices: np.ndarray, triangles: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TriangleSheetRepairEvidence]:
    """Weld, deduplicate, and orient; refuse what cannot be repaired exactly."""
    unique, first_rows, inverse = np.unique(
        vertices, axis=0, return_index=True, return_inverse=True
    )
    birth_order = np.argsort(first_rows, kind="stable")
    rank = np.empty_like(birth_order)
    rank[birth_order] = np.arange(birth_order.size, dtype=np.int64)
    repaired_vertices = np.asarray(unique[birth_order], dtype=np.float64)
    welded = np.asarray(rank[np.asarray(inverse, dtype=np.int64).reshape(-1)], dtype=np.int64)
    mapped = welded[triangles]
    degenerate = tuple(
        row
        for row, face in enumerate(mapped.tolist())
        if len(set(face)) != 3 or _exactly_collinear(repaired_vertices[face])
    )
    if degenerate:
        _refuse(
            "Authored sheet triangles are degenerate after exact welding; "
            "a zero-area triangle has no orientation to repair.",
            "degenerate",
            degenerate,
            tuple(tuple(float(value) for value in vertices[triangles[row]].mean(axis=0)) for row in degenerate),
        )
    kept: list[tuple[int, int, int]] = []
    seen: set[tuple[int, ...]] = set()
    for face in mapped.tolist():
        key = tuple(sorted(face))
        if key in seen:
            continue
        seen.add(key)
        kept.append((int(face[0]), int(face[1]), int(face[2])))
    edge_faces: dict[tuple[int, int], list[int]] = {}
    edge_order: list[tuple[int, int]] = []
    for row, face in enumerate(kept):
        for local in range(3):
            edge = (min(face[local], face[(local + 1) % 3]), max(face[local], face[(local + 1) % 3]))
            if edge not in edge_faces:
                edge_faces[edge] = []
                edge_order.append(edge)
            edge_faces[edge].append(row)
    non_manifold = tuple(sorted({row for rows in edge_faces.values() if len(rows) > 2 for row in rows}))
    if non_manifold:
        _refuse(
            "Authored sheet edges are shared by more than two distinct triangles; "
            "orientation propagation across a non-manifold edge is not geometry-preserving.",
            "non_manifold_edge",
            non_manifold,
            tuple(
                tuple(float(value) for value in repaired_vertices[list(edge)].mean(axis=0))
                for edge in edge_order
                if len(edge_faces[edge]) > 2
            ),
        )

    def traverses(face: tuple[int, int, int], first: int, last: int, /) -> bool:
        return any((face[local], face[(local + 1) % 3]) == (first, last) for local in range(3))

    flipped = [False] * len(kept)
    component = [-1] * len(kept)
    components = 0
    for seed in range(len(kept)):
        if component[seed] >= 0:
            continue
        component[seed] = components
        pending = deque((seed,))
        while pending:
            row = pending.popleft()
            face = kept[row][::-1] if flipped[row] else kept[row]
            for local in range(3):
                first, last = face[local], face[(local + 1) % 3]
                for neighbor in edge_faces[(min(first, last), max(first, last))]:
                    if neighbor == row:
                        continue
                    # A consistent neighbor traverses the shared edge in reverse.
                    needs_flip = traverses(kept[neighbor], first, last)
                    if component[neighbor] < 0:
                        component[neighbor] = components
                        flipped[neighbor] = needs_flip
                        pending.append(neighbor)
                    elif flipped[neighbor] != needs_flip:
                        rows = tuple(index for index in range(len(kept)) if component[index] == components)
                        _refuse(
                            "An authored sheet component is non-orientable; no per-triangle "
                            "orientation assignment is consistent across its shared edges.",
                            "non_orientable_component",
                            rows,
                            (tuple(float(value) for value in repaired_vertices[[first, last]].mean(axis=0)),),
                        )
        components += 1
    repaired_faces = np.asarray(
        [kept[row][::-1] if flipped[row] else kept[row] for row in range(len(kept))],
        dtype=np.int64,
    ).reshape((-1, 3))
    edges = np.asarray(edge_order, dtype=np.int64).reshape((-1, 2))
    evidence = TriangleSheetRepairEvidence(
        authored_vertices=vertices.shape[0],
        authored_triangles=triangles.shape[0],
        repaired_vertices=repaired_vertices.shape[0],
        repaired_triangles=repaired_faces.shape[0],
        repaired_edges=edges.shape[0],
        welded_vertices=vertices.shape[0] - repaired_vertices.shape[0],
        removed_duplicate_triangles=triangles.shape[0] - repaired_faces.shape[0],
        flipped_triangles=sum(flipped),
        boundary_edges=sum(1 for rows in edge_faces.values() if len(rows) == 1),
        components=components,
    )
    return repaired_vertices, repaired_faces, edges, evidence


@final
class AuthoredTriangleSheet(StrictModule, NonTrainableState):
    """Authored, possibly open and imperfect, triangle sheet with explicit repair.

    The authored arrays are retained verbatim. The repaired complex is the exact
    same point set with welded cracks, removed duplicates, and per-component
    consistent orientation; boundaries and holes stay open. It is not a closed
    region: consumers that need a volume must derive an explicit wrap, such as
    a surface envelope, and report that interpretation.
    """

    vertices: np.ndarray
    triangles: np.ndarray
    repaired_vertices: np.ndarray
    repaired_triangles: np.ndarray
    repaired_edges: np.ndarray
    repair: TriangleSheetRepairEvidence = eqx.field(static=True)
    report: ReconstructionReport = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)
    persistent_vertex_ids: np.ndarray
    persistent_edge_ids: np.ndarray
    persistent_facet_ids: np.ndarray
    persistent_region_id: None = eqx.field(static=True)
    meshing_domain: MeshingDomain
    scope_entity_id_banks: tuple[np.ndarray, ...]
    scope_entity_key_banks: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    scope_entity_ids_by_mesh_row: tuple[np.ndarray, ...]
    scope_entity_set_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, vertices: ArrayLike, triangles: ArrayLike, /, *, feature_id: str) -> None:
        authored_vertices = np.array(vertices, dtype=np.float64, copy=True)
        raw_triangles = np.asarray(triangles)
        if (
            authored_vertices.ndim != 2
            or authored_vertices.shape[1] != 3
            or authored_vertices.shape[0] < 3
            or not np.all(np.isfinite(authored_vertices))
        ):
            raise ValueError("Authored sheet vertices must be a finite (N >= 3, 3) array.")
        if raw_triangles.ndim != 2 or raw_triangles.shape[1] != 3 or raw_triangles.shape[0] == 0 or raw_triangles.dtype.kind not in "iu":
            raise ValueError("Authored sheet triangles must be a nonempty integer (M, 3) array.")
        authored_triangles = np.array(raw_triangles, dtype=np.int64, copy=True)
        if np.any(authored_triangles < 0) or np.any(authored_triangles >= authored_vertices.shape[0]):
            raise ValueError("Authored sheet triangles must reference declared vertices.")
        feature = canonical_identifier(feature_id, "feature_id")
        source_id = f"authored_sheet:{feature}"
        repaired_vertices, repaired_faces, edges, evidence = _repair_topology(
            authored_vertices, authored_triangles
        )
        vertex_ids = np.asarray(
            [
                _persistent_int64_id(
                    {
                        "kind": "authored-sheet-vertex",
                        "source_id": source_id,
                        "point": tuple(float(value) for value in point),
                    }
                )
                for point in repaired_vertices
            ],
            dtype=np.int64,
        )
        edge_ids = np.asarray(
            [
                _persistent_int64_id(
                    {
                        "kind": "authored-sheet-edge",
                        "source_id": source_id,
                        "vertices": tuple(sorted((int(vertex_ids[first]), int(vertex_ids[last])))),
                    }
                )
                for first, last in edges.tolist()
            ],
            dtype=np.int64,
        )
        facet_ids = np.asarray(
            [
                _persistent_int64_id(
                    {
                        "kind": "authored-sheet-facet",
                        "source_id": source_id,
                        "vertices": tuple(sorted(int(vertex_ids[vertex]) for vertex in face)),
                    }
                )
                for face in repaired_faces.tolist()
            ],
            dtype=np.int64,
        )
        for values in (vertex_ids, edge_ids, facet_ids):
            if np.unique(values).size != values.size:
                raise ValueError("Authored sheet persistent entity IDs collide.")
        euler = evidence.repaired_vertices - evidence.repaired_edges + evidence.repaired_triangles
        report = ReconstructionReport(
            source_kind="authored_triangle_sheet",
            algorithm="exact_weld_deduplicate_orient",
            input_digest=canonical_fingerprint(
                {
                    "kind": "authored-triangle-sheet-input",
                    "arrays": array_tree_fingerprint((authored_vertices, authored_triangles)),
                }
            ),
            input_points=evidence.authored_vertices,
            retained_points=evidence.repaired_vertices,
            output_vertices=evidence.repaired_vertices,
            output_cells=evidence.repaired_triangles,
            connected_components=evidence.components,
            watertight=evidence.boundary_edges == 0,
            winding_consistent=True,
            recenter_offset=(0.0, 0.0, 0.0),
            parameters=tuple((name, str(value)) for name, value in evidence.quantities),
            warnings=(
                ()
                if evidence.boundary_edges == 0
                else (
                    f"open authored sheet: {evidence.boundary_edges} boundary edges remain open; "
                    "any closed region is an explicit derived wrap, not a repair",
                )
            ),
            euler_characteristic=euler,
            repair_interpretation="authored_sheet_topology_repair",
        )
        source_revision = canonical_fingerprint(
            {
                "kind": "authored-triangle-sheet-revision",
                "source_id": source_id,
                "authored": array_tree_fingerprint((authored_vertices, authored_triangles)),
                "repaired": array_tree_fingerprint((repaired_vertices, repaired_faces, edges)),
                "persistent_entity_ids": array_tree_fingerprint((vertex_ids, edge_ids, facet_ids)),
                "report": asdict(report),
            }
        )
        binding_id = canonical_fingerprint(
            {
                "kind": "authored-triangle-sheet-binding",
                "source_id": source_id,
                "source_revision": source_revision,
            }
        )
        entity_ids = (vertex_ids, edge_ids, facet_ids)
        domain = _triangle_complex_meshing_domain(
            repaired_vertices,
            repaired_faces,
            edges,
            source_id,
            source_revision,
            binding_id,
            entity_ids,
            None,
            ("authored_sheet_vertex", "authored_sheet_edge", "authored_sheet_triangle"),
        )
        id_banks, key_banks, set_ids = _reconstructed_scope_banks(domain, source_id, source_revision)
        for values in (
            authored_vertices,
            authored_triangles,
            repaired_vertices,
            repaired_faces,
            edges,
            vertex_ids,
            edge_ids,
            facet_ids,
        ):
            values.setflags(write=False)
        self.vertices = authored_vertices
        self.triangles = authored_triangles
        self.repaired_vertices = repaired_vertices
        self.repaired_triangles = repaired_faces
        self.repaired_edges = edges
        self.repair = evidence
        self.report = report
        self.feature_id = feature
        self.source_id = source_id
        self.source_revision = source_revision
        self.binding_id = binding_id
        self.persistent_vertex_ids = vertex_ids
        self.persistent_edge_ids = edge_ids
        self.persistent_facet_ids = facet_ids
        self.persistent_region_id = None
        self.meshing_domain = domain
        self.scope_entity_id_banks = id_banks
        self.scope_entity_key_banks = key_banks
        self.scope_entity_ids_by_mesh_row = _scope_entity_ids_by_mesh_row(domain)
        self.scope_entity_set_ids = set_ids

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        if isinstance(dimension, bool) or not isinstance(dimension, Integral) or int(dimension) not in (0, 1, 2, 3):
            raise ValueError("Authored sheet source dimension must be in 0..3.")
        return self.scope_entity_id_banks[int(dimension)]


__all__ = ["AuthoredTriangleSheet", "TriangleSheetRepairEvidence"]
