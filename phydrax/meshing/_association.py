#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Audited mesh-to-geometry associations and their B-Rep classification rules.

A B-Rep association classifies mesh entities on B-Rep entities of one exact
source revision: every row names the entity's dimension and index, the closest
point parameters, the projection residual, ambiguity, orientation relation, and
parent provenance. Classification of mesh edges, faces, and cells from vertex
classes follows one rule set shared by adaptation, curving, and re-derivation:

* a mesh cell lies in the unique region (solid, or face for two-dimensional
  meshes) whose closure contains every vertex class;
* a mesh entity interior to the mesh with one adjacent region lies in that region;
* a mesh entity on the mesh boundary or on a region interface lies on the
  lowest-dimensional B-Rep entity containing every vertex class in its closure
  and lying in the closure of every adjacent region; ties are ambiguous.

Through topology changes (:func:`propagate_association`) preserved vertices
(including B-Rep corner vertices) keep their class, relocated vertices keep it
and are re-projected, children of a split inherit the class of the source mesh
entity they were created on (edge children inherit its edge, face-interior
children its face) and are projected, and a collapse is legal only when the kept
vertex's class lies in the closure of the removed vertex's class; illegal
collapses raise :class:`AssociationPropagationError`. Swaps change no vertex
class. Unknown lineage (provider remeshing) is re-derived by classification
transfer and projection (:func:`rederive_association`).
"""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from typing import (
    Any,
    cast,
    final,
    NamedTuple,
    NoReturn,
    Protocol,
    SupportsIndex,
    TYPE_CHECKING,
)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._archive_columns import StaticColumnArchiveRecipe, TextRowsArchiveRecipe
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellMesh, EntitySet
from ..discretization._cell_complex import PolygonalConnectivity, TetrahedralConnectivity
from ..discretization._cell_geometry_transition_types import CellGeometryTransition
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._exact_algebraic_geometry import coefficient_payload
from ..geometry._source_boundary_records import (
    SourceBoundaryChartWitness,
    SourceBoundaryFaceImageWitness,
    SourceBoundaryImagePullback,
)
from ..geometry._source_identity import source_entity_id
from ..geometry.brep._projection_contracts import (
    AbstractBRepProjection,
    brep_entity_id,
    BRepProjectionResult,
    BRepProjectionStatus,
)
from ..typing import checked
from ._scope import resolve_mesh_scope


if TYPE_CHECKING:
    from ..discretization._cell_geometry import CellGeometrySpec
    from ..geometry._source_boundary_records import SourceBoundaryQuery
    from ..geometry._source_fidelity import ValidatedSourceFidelityWitness
    from ._certification_inputs import ScopedFidelityInput
    from ._layer_core_root_geometry import ComputedLayerCoreRoot, LayerCoreRootGeometry
    from ._lineage import MeshLineage, VertexInterpolationStencil
    from ._result import CellMeshingResult
    from .providers._native_sources import NativeLayerCoreSource

type SourceBoundaryReferencePiece = tuple[
    int, int, tuple[int, ...], tuple[tuple[tuple[int, int], ...], ...]
]
type SourceBoundaryTargetPiece = (
    SourceBoundaryReferencePiece | SourceBoundaryImagePullback
)
type SourceBoundaryTargetPieceRow = tuple[SourceBoundaryTargetPiece, ...]


def _target_piece_sort_key(piece: SourceBoundaryTargetPiece, /) -> tuple[Any, ...]:
    if isinstance(piece, SourceBoundaryImagePullback):
        return (
            1,
            piece.cell_global_id,
            piece.local_entity_index,
            piece.image_rational_tokens,
            piece.source_chart_id,
        )
    return (0, *piece)


def source_boundary_association_piece_rows(
    witness: SourceBoundaryChartWitness, /
) -> dict[tuple[int, str, tuple[str, ...]], SourceBoundaryTargetPieceRow]:
    """Group exact target pieces by occurrence-qualified source face and mesh face."""
    if (
        not isinstance(witness, SourceBoundaryChartWitness)
        or witness.chart_dimension != 2
    ):
        raise TypeError("Source-face support requires a surface chart witness.")
    polyhedra = np.asarray(witness.target_face_vertex_global_ids, dtype=np.int64)
    result: dict[
        tuple[int, str, tuple[str, ...]], list[SourceBoundaryReferencePiece]
    ] = {}
    for row, (face, cell, local, identity, path, tokens) in enumerate(
        zip(
            witness.entity_global_ids,
            witness.cell_global_ids,
            witness.local_entity_indices,
            witness.source_entity_ids,
            witness.occurrence_paths,
            witness.target_reference_rational_tokens,
            strict=True,
        )
    ):
        vertices = (
            () if polyhedra.size == 0 else tuple(int(value) for value in polyhedra[row])
        )
        piece = (int(cell), int(local), vertices, tokens)
        result.setdefault((int(face), identity, tuple(path)), []).append(piece)
    return {key: tuple(sorted(pieces)) for key, pieces in result.items()}


def source_boundary_image_association_piece_rows(
    witness: SourceBoundaryFaceImageWitness, /
) -> dict[tuple[int, str, tuple[str, ...]], SourceBoundaryTargetPieceRow]:
    """Group implicit target-face pullbacks by exact source occurrence."""
    from ..geometry._source_boundary_records import SourceBoundaryFaceImageWitness

    if type(witness) is not SourceBoundaryFaceImageWitness:
        raise TypeError(
            "Image support requires its exact source-bound face image witness."
        )
    result: dict[tuple[int, str, tuple[str, ...]], list[SourceBoundaryImagePullback]] = {}
    for face, cell, local, chart, identity, path, tokens in zip(
        witness.entity_global_ids,
        witness.cell_global_ids,
        witness.local_entity_indices,
        witness.source_chart_ids,
        witness.source_entity_ids,
        witness.occurrence_paths,
        witness.image_triangle_rational_tokens,
        strict=True,
    ):
        piece = SourceBoundaryImagePullback(int(cell), int(local), tokens, chart)
        result.setdefault((int(face), identity, tuple(path)), []).append(piece)
    return {key: tuple(sorted(pieces)) for key, pieces in result.items()}


def source_boundary_image_face_orientations(
    mesh: CellMesh, geometry: CellGeometrySpec, witness: SourceBoundaryFaceImageWitness, /
) -> dict[tuple[int, str, tuple[str, ...]], int]:
    """Resolve source normals against canonical target-face vertex orientation."""
    from ..discretization._cell_geometry_transition_types import _geometry_map_expressions
    from ..discretization._coordinate_enclosure import evaluate_expression
    from ..discretization._reference_cell import reference_cell_topology
    from ._topology_edit import entity_keys

    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = {
        int(identifier): (
            block.cell_kind,
            tuple(int(value) for value in vertices[np.asarray(rows, dtype=np.int64)]),
        )
        for block in mesh.blocks
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
    }
    resolved = geometry.resolve(mesh)
    maps = _geometry_map_expressions(mesh, geometry, (resolved[0], resolved[1]))
    faces = {
        int(identifier): tuple(int(value) for value in key if value >= 0)
        for key, identifier in zip(
            entity_keys(mesh, 2), mesh.entity_set(2).entity_ids, strict=True
        )
    }

    def normal(
        points_: tuple[tuple[Fraction, ...], ...],
    ) -> tuple[Fraction, Fraction, Fraction]:
        first, second, third = points_[:3]
        u = tuple(b - a for a, b in zip(first, second, strict=True))
        v = tuple(b - a for a, b in zip(first, third, strict=True))
        return (
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        )

    result: dict[tuple[int, str, tuple[str, ...]], int] = {}
    for row, (face, cell, local, identity, path, orientation) in enumerate(
        zip(
            witness.entity_global_ids,
            witness.cell_global_ids,
            witness.local_entity_indices,
            witness.source_entity_ids,
            witness.occurrence_paths,
            witness.orientations,
            strict=True,
        )
    ):
        face_id = int(face)
        vertex_ids = faces.get(face_id)
        if vertex_ids is None or len(vertex_ids) < 3:
            raise ValueError("An image witness names an invalid target face.")
        owner = cells.get(int(cell))
        if owner is None:
            raise ValueError("An image witness names an absent target cell.")
        kind, cell_vertices = owner
        topology = reference_cell_topology(kind)
        local_face = int(local)
        if not 0 <= local_face < len(topology.entities[2]):
            raise ValueError("An image witness names an absent local target face.")
        local_vertices = topology.entities[2][local_face]
        if tuple(sorted(cell_vertices[index] for index in local_vertices)) != vertex_ids:
            raise ValueError("An image witness changes target cell-face incidence.")
        by_global = {cell_vertices[index]: index for index in local_vertices}
        physical = tuple(
            tuple(
                evaluate_expression(
                    expression,
                    tuple(
                        Fraction.from_float(float(value))
                        for value in topology.vertices[by_global[vertex]]
                    ),
                )
                for expression in maps.expressions(int(cell))
            )
            for vertex in vertex_ids[:3]
        )
        if any(not isinstance(value, Fraction) for point in physical for value in point):
            raise ValueError(
                "Image face orientation requires exact rational target corners."
            )
        target_normal = normal(cast(tuple[tuple[Fraction, ...], ...], physical))
        image_normal = normal(witness.image_triangle(row))
        alignment = int(orientation) * sum(
            left * right for left, right in zip(target_normal, image_normal, strict=True)
        )
        if not alignment:
            raise ValueError("Image source and target face orientation is degenerate.")
        key = (face_id, identity, tuple(path))
        value = 1 if alignment > 0 else -1
        previous = result.setdefault(key, value)
        if previous != value:
            raise ValueError(
                "Image source pieces disagree on canonical face orientation."
            )
    return result


def source_boundary_edge_intervals(
    mesh: CellMesh, witness: SourceBoundaryChartWitness, /
) -> dict[tuple[int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]]:
    """Derive exact target-edge support from source chart pieces.

    The caller must first validate the witness against its source and geometry.
    Edge parameters follow increasing target vertex global IDs.
    """
    from ..discretization._reference_cell import reference_cell_topology
    from ._layer_core_strata_replay import _triangle_line_interval
    from ._topology_edit import entity_keys

    if (
        mesh.topological_dimension != 3
        or witness.chart_dimension != 2
        or witness.reference_dimension != 3
        or np.asarray(witness.target_face_vertex_global_ids).size
    ):
        raise ValueError(
            "Source facet-edge support requires explicit volume cell charts."
        )
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = {
        int(identifier): (
            block.cell_kind,
            tuple(int(value) for value in vertices[np.asarray(rows, dtype=np.int64)]),
        )
        for block in mesh.blocks
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
    }
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edges = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(entity_keys(mesh, 1), edge_ids, strict=True)
    }
    intervals: dict[
        tuple[int, str, tuple[str, ...]], list[tuple[Fraction, Fraction]]
    ] = {}
    for row, (cell_id, local_face) in enumerate(
        zip(witness.cell_global_ids, witness.local_entity_indices, strict=True)
    ):
        cell = cells.get(int(cell_id))
        if cell is None:
            raise ValueError("A source chart names an absent target cell.")
        kind, global_vertices = cell
        topology = reference_cell_topology(kind)
        local = int(local_face)
        if not 0 <= local < len(topology.entities[2]):
            raise ValueError("A source chart names an absent target face.")
        face = topology.entities[2][local]
        points = {
            global_vertices[index]: tuple(
                Fraction(float(value)) for value in topology.vertices[index]
            )
            for index in face
        }
        triangle = witness.target_reference_chart(row)
        face_vertices = set(int(index) for index in face)
        for local_edge in topology.entities[1]:
            if not set(local_edge) <= face_vertices:
                continue
            first, last = sorted(int(global_vertices[index]) for index in local_edge)
            edge_id = edges.get((first, last))
            if edge_id is None:
                raise ValueError(
                    "A source chart face edge is absent from target incidence."
                )
            interval = _triangle_line_interval(points[first], points[last], triangle)
            if interval is not None and interval[0] < interval[1]:
                key = (
                    edge_id,
                    witness.source_entity_ids[row],
                    witness.occurrence_paths[row],
                )
                intervals.setdefault(key, []).append(interval)
    result: dict[
        tuple[int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]
    ] = {}
    for key, values in intervals.items():
        merged: list[tuple[Fraction, Fraction]] = []
        for lower, upper in sorted(values):
            if merged and lower <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
            else:
                merged.append((lower, upper))
        result[key] = tuple(merged)
    return result


class _LayerImageSupportPreparation:
    """Call-local exact edge maps produced from the bound layer geometry."""

    def __init__(
        self,
        source: NativeLayerCoreSource,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        core_cell_ids: tuple[int, ...],
        /,
        *,
        root: LayerCoreRootGeometry | None = None,
        computed_root: ComputedLayerCoreRoot | None = None,
    ) -> None:
        from ._layer_core_strata_replay import core_edge_images

        self.source = source
        self.core_cell_ids = tuple(sorted(core_cell_ids))
        if len(set(self.core_cell_ids)) != len(self.core_cell_ids):
            raise ValueError("Layer image preparation requires unique core cell IDs.")
        self.mesh = mesh
        self.root = root
        self.computed_root = computed_root
        self.geometry = geometry
        self.images = core_edge_images(
            source, mesh, geometry, core_cell_ids, root=root, computed_root=computed_root
        )
        self.edge_maps = {
            (cell, edge): image
            for edge, rows in self.images.items()
            for cell, image, _ in rows
        }
        self.endpoints: dict[
            tuple[int, int], tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
        ] = {}
        self.face_support: dict[
            int,
            tuple[
                SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness,
                dict[
                    tuple[int, str, tuple[str, ...]],
                    tuple[frozenset[int], frozenset[tuple[int, int]]],
                ],
            ],
        ] = {}
        self.image_pieces: dict[
            int,
            tuple[
                SourceBoundaryFaceImageWitness,
                dict[tuple[int, int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]],
            ],
        ] = {}

    def require(
        self, source: NativeLayerCoreSource, mesh: CellMesh, geometry: CellGeometrySpec,
        core_cell_ids: tuple[int, ...], /, *,
        root: LayerCoreRootGeometry | None = None,
        computed_root: ComputedLayerCoreRoot | None = None,
    ) -> None:
        if (
            self.source is not source or self.mesh is not mesh or self.geometry is not geometry
            or self.core_cell_ids != tuple(sorted(core_cell_ids))
            or self.root is not root or self.computed_root is not computed_root
            or len(set(core_cell_ids)) != len(core_cell_ids)
        ):
            raise ValueError("Layer image preparation belongs to other source, mesh, geometry or core cells.")

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> NoReturn:
        raise TypeError("Layer image preparation is operation-local and cannot be serialized.")


def _prepare_layer_image_support(
    source: NativeLayerCoreSource,
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    core_cell_ids: tuple[int, ...],
    /,
    *,
    root: LayerCoreRootGeometry | None = None,
    computed_root: ComputedLayerCoreRoot | None = None,
) -> _LayerImageSupportPreparation:
    return _LayerImageSupportPreparation(
        source, mesh, geometry, core_cell_ids, root=root, computed_root=computed_root,
    )


def _source_boundary_image_edge_pieces(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryFaceImageWitness,
    /,
    preparation: _LayerImageSupportPreparation | None = None,
) -> dict[tuple[int, int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]]:
    """Clip actual target-edge maps against each exact physical source-face piece."""
    from ..discretization._cell_geometry_transition_types import _geometry_map_expressions
    from ..discretization._coordinate_enclosure import (
        compose_expression,
        evaluate_expression,
    )
    from ..discretization._nested_reference import _chart_from_corners
    from ..discretization._reference_cell import reference_cell_topology
    from ._layer_core_strata_replay import _triangle_line_interval, source_line_interval
    from ._topology_edit import entity_keys

    if mesh.topological_dimension != 3:
        raise ValueError("Source face image support requires a volume target.")
    if preparation is not None and (
        preparation.mesh is not mesh or preparation.geometry is not geometry
    ):
        raise ValueError("Layer image preparation belongs to another mesh or geometry.")
    if preparation is not None:
        cached = preparation.image_pieces.get(id(witness))
        if cached is not None and cached[0] is witness:
            return dict(cached[1])
    vertices = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    cells = {
        int(identifier): (
            block.cell_kind,
            tuple(int(value) for value in vertices[np.asarray(rows, dtype=np.int64)]),
        )
        for block in mesh.blocks
        for identifier, rows in zip(block.global_ids, block.vertices, strict=True)
    }
    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edges = {
        tuple(int(value) for value in key if value >= 0): int(identifier)
        for key, identifier in zip(entity_keys(mesh, 1), edge_ids, strict=True)
    }
    maps = None
    intervals: dict[
        tuple[int, int, str, tuple[str, ...]], list[tuple[Fraction, Fraction]]
    ] = {}
    edge_images: dict[
        tuple[int, int], tuple[tuple[Fraction, ...], tuple[Fraction, ...]]
    ] = {} if preparation is None else preparation.endpoints
    for row, (cell_raw, local_raw) in enumerate(
        zip(witness.cell_global_ids, witness.local_entity_indices, strict=True)
    ):
        cell_id, local = int(cell_raw), int(local_raw)
        cell = cells.get(cell_id)
        if cell is None:
            raise ValueError("A source image names an absent target cell.")
        kind, global_vertices = cell
        topology = reference_cell_topology(kind)
        if not 0 <= local < len(topology.entities[2]):
            raise ValueError("A source image names an absent target face.")
        face = set(topology.entities[2][local])
        triangle = witness.image_triangle(row)
        for local_edge in topology.entities[1]:
            if not set(local_edge) <= face:
                continue
            ordered = sorted(local_edge, key=lambda index: global_vertices[index])
            first_id, last_id = (global_vertices[index] for index in ordered)
            edge_id = edges.get((first_id, last_id))
            if edge_id is None:
                raise ValueError(
                    "An image-supported face edge is absent from target incidence."
                )
            cache_key = (cell_id, edge_id)
            endpoints = edge_images.get(cache_key)
            if endpoints is None:
                image = (
                    None if preparation is None else preparation.edge_maps.get(cache_key)
                )
                if image is None:
                    if maps is None:
                        resolved = geometry.resolve(mesh)
                        maps = _geometry_map_expressions(
                            mesh, geometry, (resolved[0], resolved[1])
                        )
                    corners = tuple(
                        tuple(
                            Fraction.from_float(float(value))
                            for value in topology.vertices[index]
                        )
                        for index in ordered
                    )
                    arguments = _chart_from_corners("interval", corners)
                    image = tuple(
                        compose_expression(value, arguments)
                        for value in maps.expressions(cell_id)
                    )
                evaluated = tuple(
                    tuple(
                        evaluate_expression(value, (Fraction(parameter),))
                        for value in image
                    )
                    for parameter in (0, 1)
                )
                if any(
                    not isinstance(value, Fraction)
                    for point in evaluated
                    for value in point
                ):
                    raise ValueError(
                        "Image-supported target edge endpoints must be exact rational points."
                    )
                exact = cast(tuple[tuple[Fraction, ...], tuple[Fraction, ...]], evaluated)
                if source_line_interval(image, exact[0], exact[1]) != (
                    Fraction(0),
                    Fraction(1),
                ):
                    raise ValueError(
                        "Image-supported target edge requires an exact affine parameter map."
                    )
                endpoints = edge_images[cache_key] = exact
            interval = _triangle_line_interval(endpoints[0], endpoints[1], triangle)
            if interval is not None:
                key = (
                    int(witness.entity_global_ids[row]),
                    edge_id,
                    witness.source_entity_ids[row],
                    witness.occurrence_paths[row],
                )
                intervals.setdefault(key, []).append(interval)
    result: dict[
        tuple[int, int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]
    ] = {}
    for key, values in intervals.items():
        merged: list[tuple[Fraction, Fraction]] = []
        for lower, upper in sorted(values):
            if merged and lower <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
            else:
                merged.append((lower, upper))
        result[key] = tuple(merged)
    if preparation is not None:
        preparation.image_pieces[id(witness)] = (witness, dict(result))
    return result


def _source_boundary_image_edge_intervals(
    mesh: CellMesh, geometry: CellGeometrySpec, witness: SourceBoundaryFaceImageWitness, /,
    preparation: _LayerImageSupportPreparation | None,
) -> dict[tuple[int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]]:
    """Union exact source-facet image coverage along target edges."""
    intervals: dict[
        tuple[int, str, tuple[str, ...]], list[tuple[Fraction, Fraction]]
    ] = {}
    for (_, edge, identity, path), values in _source_boundary_image_edge_pieces(
        mesh, geometry, witness, preparation
    ).items():
        intervals.setdefault((edge, identity, path), []).extend(
            (lower, upper) for lower, upper in values if lower < upper
        )
    result: dict[
        tuple[int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]
    ] = {}
    for key, values in intervals.items():
        merged: list[tuple[Fraction, Fraction]] = []
        for lower, upper in sorted(values):
            if merged and lower <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], upper))
            else:
                merged.append((lower, upper))
        result[key] = tuple(merged)
    return result


def source_boundary_image_edge_intervals(
    mesh: CellMesh, geometry: CellGeometrySpec, witness: SourceBoundaryFaceImageWitness, /
) -> dict[tuple[int, str, tuple[str, ...]], tuple[tuple[Fraction, Fraction], ...]]:
    return _source_boundary_image_edge_intervals(mesh, geometry, witness, None)


def _source_boundary_image_face_support(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    witness: SourceBoundaryFaceImageWitness,
    /,
    preparation: _LayerImageSupportPreparation | None,
) -> dict[tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]]:
    """Retain face-qualified exact vertex and edge support without invented diagonals."""
    from ._topology_edit import entity_keys

    edge_ids = np.asarray(mesh.entity_set(1).entity_ids, dtype=np.int64)
    edge_vertices = {
        int(identifier): tuple(int(value) for value in key if value >= 0)
        for key, identifier in zip(entity_keys(mesh, 1), edge_ids, strict=True)
    }
    result: dict[
        tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]
    ] = {
        (int(face), identity, tuple(path)): (set(), set())
        for face, identity, path in zip(
            witness.entity_global_ids,
            witness.source_entity_ids,
            witness.occurrence_paths,
            strict=True,
        )
    }
    for (face, edge, identity, path), intervals in _source_boundary_image_edge_pieces(
        mesh, geometry, witness, preparation
    ).items():
        pair = edge_vertices[edge]
        if len(pair) != 2:
            raise ValueError("An image-supported target edge must have two vertices.")
        first, last = pair
        covered_vertices, covered_edges = result.setdefault(
            (face, identity, path), (set(), set())
        )
        if any(lower < upper for lower, upper in intervals):
            covered_edges.add((first, last))
        if any(lower == 0 for lower, _ in intervals):
            covered_vertices.add(first)
        if any(upper == 1 for _, upper in intervals):
            covered_vertices.add(last)
    return result


def source_boundary_image_face_support(
    mesh: CellMesh, geometry: CellGeometrySpec, witness: SourceBoundaryFaceImageWitness, /
) -> dict[tuple[int, str, tuple[str, ...]], tuple[set[int], set[tuple[int, int]]]]:
    return _source_boundary_image_face_support(mesh, geometry, witness, None)


def _validate_target_reference_piece_rows(
    rows: tuple[SourceBoundaryTargetPieceRow | None, ...],
    roles: tuple[GeometrySourceEntityRole | None, ...],
    count: int,
    /,
) -> None:
    if len(rows) != count:
        raise ValueError("Target face piece rows must align with association rows.")
    for row, pieces in enumerate(rows):
        if pieces is None:
            continue
        if not isinstance(pieces, tuple) or not pieces:
            raise ValueError("Target face support needs nonempty immutable exact pieces.")
        for piece in pieces:
            if isinstance(piece, SourceBoundaryImagePullback):
                cell, local = piece.cell_global_id, piece.local_entity_index
                tokens, chart_id = piece.image_rational_tokens, piece.source_chart_id
                role = roles[row]
                if (
                    type(cell) is not int
                    or cell < 0
                    or type(local) is not int
                    or local < 0
                    or not isinstance(chart_id, str)
                    or not chart_id
                    or chart_id != chart_id.strip()
                    or role
                    not in (GeometrySourceEntityRole.FACET, GeometrySourceEntityRole.EDGE)
                    or not isinstance(tokens, tuple)
                    or (
                        len(tokens) != 2
                        if role is GeometrySourceEntityRole.EDGE
                        else len(tokens) < 3
                    )
                ):
                    raise ValueError(
                        "Image pullback needs exact incidence, source chart and polygon support."
                    )
                points: list[tuple[Fraction, ...]] = []
                for point in tokens:
                    if not isinstance(point, tuple) or len(point) != 3:
                        raise ValueError(
                            "Image pullback points must have three exact physical coordinates."
                        )
                    coordinates: list[Fraction] = []
                    for token in point:
                        if (
                            not isinstance(token, tuple)
                            or len(token) != 2
                            or type(token[0]) is not int
                            or type(token[1]) is not int
                            or token[1] <= 0
                        ):
                            raise ValueError(
                                "Image pullback has invalid rational coordinate tokens."
                            )
                        value = Fraction(*token)
                        if (value.numerator, value.denominator) != token:
                            raise ValueError(
                                "Image pullback rational tokens must be reduced."
                            )
                        coordinates.append(value)
                    points.append(tuple(coordinates))
                if len(set(points)) != len(points):
                    raise ValueError("Image pullback vertices must be distinct.")
                direction = tuple(
                    b - a for a, b in zip(points[0], points[1], strict=True)
                )
                if role is GeometrySourceEntityRole.FACET and not any(
                    direction[i] * (point[j] - points[0][j])
                    != direction[j] * (point[i] - points[0][i])
                    for point in points[2:]
                    for i in range(3)
                    for j in range(i + 1, 3)
                ):
                    raise ValueError("Image pullback polygon must have nonzero area.")
                continue
            if not isinstance(piece, tuple) or len(piece) != 4:
                raise ValueError(
                    "Target face piece needs its target incidence and exact chart."
                )
            cell, local, vertices, tokens = piece
            if (
                type(cell) is not int
                or cell < 0
                or type(local) is not int
                or local < -1
                or not isinstance(vertices, tuple)
                or len(vertices) not in (0, 3)
                or any(type(vertex) is not int or vertex < 0 for vertex in vertices)
                or len(set(vertices)) != len(vertices)
            ):
                raise ValueError("Target face piece has invalid target incidence.")
            corner_count = 3 if roles[row] is GeometrySourceEntityRole.FACET else 2
            if roles[row] not in (
                GeometrySourceEntityRole.FACET,
                GeometrySourceEntityRole.EDGE,
            ):
                raise ValueError(
                    "Target face pieces require an explicit source facet or edge role."
                )
            if (
                not isinstance(tokens, tuple)
                or len(tokens) != corner_count
                or not isinstance(tokens[0], tuple)
                or len(tokens[0]) not in ((2,) if vertices else (2, 3))
                or any(
                    not isinstance(point, tuple) or len(point) != len(tokens[0])
                    for point in tokens
                )
            ):
                raise ValueError(
                    "Target face piece has an invalid exact reference chart."
                )
            chart = []
            for point in tokens:
                coordinates = []
                for token in point:
                    if (
                        not isinstance(token, tuple)
                        or len(token) != 2
                        or type(token[0]) is not int
                        or type(token[1]) is not int
                        or token[1] <= 0
                    ):
                        raise ValueError("Target face piece has invalid rational tokens.")
                    value = Fraction(*token)
                    if (value.numerator, value.denominator) != token:
                        raise ValueError(
                            "Target face piece rational tokens must be reduced."
                        )
                    coordinates.append(value)
                chart.append(tuple(coordinates))
            first = tuple(b - a for a, b in zip(chart[0], chart[1], strict=True))
            if not any(first):
                raise ValueError("Target face reference pieces must be nondegenerate.")
            if corner_count == 3:
                second = tuple(b - a for a, b in zip(chart[0], chart[2], strict=True))
                if not any(
                    first[i] * second[j] != first[j] * second[i]
                    for i in range(len(first))
                    for j in range(i + 1, len(first))
                ):
                    raise ValueError(
                        "Target face reference pieces must be nondegenerate."
                    )
        if tuple(sorted(set(pieces), key=_target_piece_sort_key)) != pieces:
            raise ValueError("Target face pieces must be unique and in canonical order.")


class GeometryAssociationKind(StrEnum):
    BREP = "brep"
    SURFACE = "surface"
    IMPLICIT = "implicit"
    PIECEWISE_LINEAR = "piecewise_linear"
    MAPPED_REFERENCE = "mapped_reference"
    CURVE = "curve"


class GeometrySourceEntityRole(StrEnum):
    """Source stratum role; not inferred from ambient dimension or row position."""

    VERTEX = "vertex"
    EDGE = "edge"
    FACET = "facet"
    REGION = "region"


class GeometryAssociationProvenance(StrEnum):
    """How the rows were established.

    ``PROVIDER``: emitted by the generating mesher. ``LINEAGE``: propagated
    through an exact topology lineage from ``parent_association_id``.
    ``CLASSIFICATION``: derived by geometric classification and projection
    (optionally restricted by classification transfer from the parent).
    """

    PROVIDER = "provider"
    LINEAGE = "lineage"
    CLASSIFICATION = "classification"


class AssociationEntityClasses(NamedTuple):
    """Source stratum labels consumed by topology-adaptation constraints."""

    dimensions: np.ndarray
    indices: np.ndarray
    resolved: np.ndarray
    codes: np.ndarray


class AssociationTransfer(Protocol):
    """Common adaptation contract for revision-bound source association routes.

    ``propagate`` receives the endpoint-bound geometry transition and the
    renewed target source-chart witness when available. Geometry-free routes
    pass ``None``; source routes that require either proof must reject its absence.
    """

    @property
    def transfer_id(self) -> str: ...

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]: ...

    def validate_source(self, source: CellMeshingResult, /) -> None: ...

    def classes(
        self, source: CellMeshingResult, /
    ) -> tuple[AssociationEntityClasses, ...]: ...

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
        geometry_transition: CellGeometryTransition | None,
        target_fidelity_witness: SourceBoundaryChartWitness
        | SourceBoundaryFaceImageWitness
        | None,
        target_scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
    ) -> tuple[GeometryAssociation, ...]: ...


def _unclassified_id(
    source_id: str, source_revision: str, occurrence_path: tuple[str, ...], /
) -> str:
    return source_entity_id(
        source_id,
        source_revision,
        "unclassified",
        "unclassified",
        occurrence_path=occurrence_path,
    )


def _rows_or(value: ArrayLike | None, fill: Any, shape: Any, dtype: Any, /) -> np.ndarray:
    return (
        np.full(shape, fill, dtype=dtype) if value is None else np.asarray(value, dtype)
    )


class GeometryAssociation(StrictModule, NonTrainableState):
    """Audited map from exact mesh entities to authoritative geometry entities.

    Source rows may carry explicit ``source_dimensions``/``source_indices``;
    B-Rep rows require them and bind the occurrence-qualified source entity.
    ``-1`` marks an unresolved row without a candidate. ``parameters`` hold
    face ``(u, v)`` or edge ``(t, nan)`` coordinates (NaN otherwise).
    ``orientations`` relate canonical mesh order to the source tangent or
    oriented normal (``+1``/``-1``; ``0`` when undefined). ``parent_dimensions`` and
    ``parent_ids`` name the parent mesh entity whose class a row inherited
    (``-1`` for geometric classification). Infinite residuals are allowed only on
    unresolved rows (no closest point).
    ``target_parameter_intervals`` restrict edge rows to exact subintervals of
    the canonically oriented target edge; ``None`` denotes the entire edge.
    Source intervals and orientation then describe only that target support.
    ``target_reference_pieces`` carry exact witness-bound face restrictions;
    their parent cell, local face and optional polyhedral triangle IDs are
    retained together with the original witness token rows.
    """

    association_kind: GeometryAssociationKind = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    target_entity_set_id: str = eqx.field(static=True)
    target_global_ids: Array
    source_entity_ids: tuple[str, ...] = eqx.field(static=True)
    source_roles: tuple[GeometrySourceEntityRole | None, ...] = eqx.field(static=True)
    source_occurrence_paths: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    source_region_sides: tuple[tuple[str | None, str | None], ...] = eqx.field(
        static=True
    )
    source_parameter_intervals: tuple[tuple[Fraction, Fraction] | None, ...] = eqx.field(
        static=True
    )
    target_parameter_intervals: tuple[tuple[Fraction, Fraction] | None, ...] = eqx.field(
        static=True
    )
    target_reference_pieces: tuple[SourceBoundaryTargetPieceRow | None, ...] = eqx.field(
        static=True
    )
    source_boundary_witness_id: str | None = eqx.field(static=True)
    source_parameter_points: tuple[Fraction | None, ...] = eqx.field(static=True)
    residuals: Array
    resolved: Array
    ambiguous: Array
    source_dimensions: Array
    source_indices: Array
    parameters: Array
    orientations: Array
    parent_dimensions: Array
    parent_ids: Array
    exact: bool = eqx.field(static=True)
    provenance: GeometryAssociationProvenance = eqx.field(static=True)
    parent_association_id: str | None = eqx.field(static=True)
    association_id: str = eqx.field(static=True)

    def __init__(
        self,
        association_kind: GeometryAssociationKind,
        source_id: str,
        source_revision: str,
        target_entity_set_id: str,
        target_global_ids: ArrayLike,
        source_entity_ids: tuple[str, ...],
        residuals: ArrayLike,
        /,
        *,
        resolved: ArrayLike | None = None,
        ambiguous: ArrayLike | None = None,
        exact: bool = False,
        source_dimensions: ArrayLike | None = None,
        source_indices: ArrayLike | None = None,
        source_roles: tuple[GeometrySourceEntityRole | None, ...] | None = None,
        source_occurrence_paths: tuple[tuple[str, ...], ...] | None = None,
        source_region_sides: tuple[tuple[str | None, str | None], ...] | None = None,
        source_parameter_intervals: tuple[tuple[Fraction, Fraction] | None, ...]
        | None = None,
        target_parameter_intervals: tuple[tuple[Fraction, Fraction] | None, ...]
        | None = None,
        target_reference_pieces: tuple[SourceBoundaryTargetPieceRow | None, ...]
        | None = None,
        source_boundary_witness_id: str | None = None,
        source_parameter_points: tuple[Fraction | None, ...] | None = None,
        parameters: ArrayLike | None = None,
        orientations: ArrayLike | None = None,
        parent_dimensions: ArrayLike | None = None,
        parent_ids: ArrayLike | None = None,
        parent_association_id: str | None = None,
        provenance: GeometryAssociationProvenance = GeometryAssociationProvenance.PROVIDER,
    ) -> None:
        if not isinstance(association_kind, GeometryAssociationKind):
            raise TypeError("association_kind must be GeometryAssociationKind.")
        if not isinstance(provenance, GeometryAssociationProvenance):
            raise TypeError("provenance must be GeometryAssociationProvenance.")
        source = str(source_id).strip()
        revision = str(source_revision).strip()
        entity_set = str(target_entity_set_id).strip()
        if not source or not revision or not entity_set:
            raise ValueError("Geometry association identities must be non-empty.")
        targets = np.asarray(target_global_ids)
        if targets.ndim != 1 or not np.issubdtype(targets.dtype, np.integer):
            raise TypeError("target_global_ids must be one integer vector.")
        targets = targets.astype(np.int64, copy=False)
        if (
            targets.size == 0
            or np.any(targets < 0)
            or np.unique(targets).size != targets.size
        ):
            raise ValueError(
                "Target geometry association IDs must be unique and non-negative."
            )
        count = targets.size
        sources = tuple(str(value).strip() for value in source_entity_ids)
        distances = np.asarray(residuals, dtype=np.float64)
        if len(sources) != count or any(not value for value in sources):
            raise ValueError(
                "Source geometry IDs must match target entities and be non-empty."
            )
        resolved_ = _rows_or(resolved, True, (count,), np.bool_)
        ambiguous_ = _rows_or(ambiguous, False, (count,), np.bool_)
        if resolved_.shape != targets.shape or ambiguous_.shape != targets.shape:
            raise ValueError("Association status masks must match target entities.")
        if (
            distances.shape != targets.shape
            or np.any(np.isnan(distances))
            or np.any(distances < 0)
            or np.any(np.isinf(distances) & resolved_)
        ):
            raise ValueError(
                "Association residuals must be non-negative and aligned; infinite "
                "residuals are allowed only on unresolved rows."
            )
        if np.any(resolved_ & ambiguous_):
            raise ValueError("Resolved geometry associations cannot be ambiguous.")
        if exact and (
            not np.all(resolved_) or np.any(ambiguous_) or np.any(distances != 0.0)
        ):
            raise ValueError(
                "Exact geometry associations require zero-residual unique coverage."
            )
        roles = (
            tuple(None for _ in range(count))
            if source_roles is None
            else tuple(source_roles)
        )
        occurrences = (
            tuple(() for _ in range(count))
            if source_occurrence_paths is None
            else tuple(source_occurrence_paths)
        )
        region_sides = (
            tuple((None, None) for _ in range(count))
            if source_region_sides is None
            else tuple(source_region_sides)
        )
        parameter_intervals = (
            tuple(None for _ in range(count))
            if source_parameter_intervals is None
            else tuple(source_parameter_intervals)
        )
        target_intervals = (
            tuple(None for _ in range(count))
            if target_parameter_intervals is None
            else tuple(target_parameter_intervals)
        )
        target_pieces = (
            tuple(None for _ in range(count))
            if target_reference_pieces is None
            else tuple(target_reference_pieces)
        )
        parameter_points = (
            tuple(None for _ in range(count))
            if source_parameter_points is None
            else tuple(source_parameter_points)
        )
        if len(roles) != count or any(
            role is not None and not isinstance(role, GeometrySourceEntityRole)
            for role in roles
        ):
            raise ValueError("Source roles must align with target association rows.")
        if len(occurrences) != count or any(
            not isinstance(path, tuple)
            or any(
                not isinstance(part, str) or not part or part != part.strip()
                for part in path
            )
            for path in occurrences
        ):
            raise ValueError("Source occurrence paths must be aligned immutable tuples.")
        dimensions, indices = _source_entities(
            association_kind,
            source,
            revision,
            sources,
            source_dimensions,
            source_indices,
            occurrences,
        )
        if len(region_sides) != count or any(
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(
                value is not None
                and (not isinstance(value, str) or not value or value != value.strip())
                for value in pair
            )
            for pair in region_sides
        ):
            raise ValueError(
                "Source region sides must be aligned (minus, plus) ID pairs."
            )
        if len(parameter_intervals) != count or any(
            interval is not None
            and (
                not isinstance(interval, tuple)
                or len(interval) != 2
                or any(not isinstance(value, Fraction) for value in interval)
                or interval[0] >= interval[1]
            )
            for interval in parameter_intervals
        ):
            raise ValueError(
                "Source parameter intervals must be exact increasing Fraction pairs."
            )
        if len(target_intervals) != count or any(
            interval is not None
            and (
                not isinstance(interval, tuple)
                or len(interval) != 2
                or any(not isinstance(value, Fraction) for value in interval)
                or not (0 <= interval[0] < interval[1] <= 1)
            )
            for interval in target_intervals
        ):
            raise ValueError(
                "Target edge intervals must be exact increasing Fraction pairs in [0, 1]."
            )
        target_intervals = tuple(
            None if interval == (Fraction(0), Fraction(1)) else interval
            for interval in target_intervals
        )
        _validate_target_reference_piece_rows(target_pieces, roles, count)
        if source_boundary_witness_id is not None and (
            not isinstance(source_boundary_witness_id, str)
            or not source_boundary_witness_id
            or source_boundary_witness_id != source_boundary_witness_id.strip()
        ):
            raise ValueError("Source boundary witness identity must be nonempty.")
        if any(piece is not None for piece in target_pieces):
            if source_boundary_witness_id is None or any(
                piece is None for piece in target_pieces
            ):
                raise ValueError(
                    "Every witness-bound face row needs exact target pieces."
                )
            if any(
                not resolved_[row]
                or roles[row]
                not in (GeometrySourceEntityRole.FACET, GeometrySourceEntityRole.EDGE)
                or (
                    roles[row] is GeometrySourceEntityRole.EDGE
                    and parameter_intervals[row] is None
                )
                for row in range(count)
            ):
                raise ValueError(
                    "Exact target face pieces need resolved source facets or parameterized edges."
                )
        elif source_boundary_witness_id is not None and any(
            not resolved_[row] or roles[row] is not GeometrySourceEntityRole.FACET
            for row in range(count)
        ):
            raise ValueError(
                "Witness-bound edge support requires resolved source facets."
            )
        if len(parameter_points) != count or any(
            value is not None and not isinstance(value, Fraction)
            for value in parameter_points
        ):
            raise ValueError("Source parameter points must be exact Fraction scalars.")
        for row, (interval, point) in enumerate(
            zip(parameter_intervals, parameter_points, strict=True)
        ):
            curve_edge = (
                association_kind is GeometryAssociationKind.CURVE
                and roles[row] is GeometrySourceEntityRole.EDGE
            )
            parameterized_edge = (
                association_kind
                in (
                    GeometryAssociationKind.CURVE,
                    GeometryAssociationKind.PIECEWISE_LINEAR,
                )
                and roles[row] is GeometrySourceEntityRole.EDGE
            )
            if (
                (interval is not None or point is not None) and not parameterized_edge
            ) or (interval is not None and point is not None):
                raise ValueError(
                    "Exact source parameters belong to one edge interval or point per row."
                )
            if not resolved_[row] and (interval is not None or point is not None):
                raise ValueError("Unresolved edge rows cannot claim exact parameters.")
            facet_edge = (
                source_boundary_witness_id is not None
                and roles[row] is GeometrySourceEntityRole.FACET
            )
            if target_intervals[row] is not None and (
                not resolved_[row]
                or not ((parameterized_edge and interval is not None) or facet_edge)
            ):
                raise ValueError(
                    "Partial target support requires a proved source edge or facet."
                )
            if resolved_[row] and curve_edge and (interval is None) == (point is None):
                raise ValueError(
                    "Resolved curve-edge rows require exactly one interval or point parameter."
                )
        if association_kind is GeometryAssociationKind.PIECEWISE_LINEAR and any(
            resolved_[row]
            and roles[row] is GeometrySourceEntityRole.FACET
            and region_sides[row] == (None, None)
            for row in range(count)
        ):
            raise ValueError(
                "Resolved piecewise-linear facets need at least one incident source region."
            )
        if association_kind is GeometryAssociationKind.PIECEWISE_LINEAR and any(
            resolved_[row] and roles[row] is None for row in range(count)
        ):
            raise ValueError("Resolved piecewise-linear associations need source roles.")
        if any(
            not resolved_[row] and region_sides[row] != (None, None)
            for row in range(count)
        ):
            raise ValueError("Unresolved rows cannot claim source region sides.")
        if (
            association_kind is GeometryAssociationKind.BREP
            or source_dimensions is not None
            or source_indices is not None
        ) and np.any((dimensions < 0) & resolved_):
            raise ValueError("Resolved source-stratum rows must name a source entity.")
        parameters_, orientations_, parent_dimensions_, parent_ids_ = _row_metadata(
            count, parameters, orientations, parent_dimensions, parent_ids
        )
        parent = None if parent_association_id is None else str(parent_association_id)
        if parent is not None and not parent.strip():
            raise ValueError("parent_association_id must be non-empty when provided.")
        if provenance is GeometryAssociationProvenance.LINEAGE and parent is None:
            raise ValueError("Lineage associations require their parent association.")
        self.association_kind = association_kind
        self.source_id = source
        self.source_revision = revision
        self.target_entity_set_id = entity_set
        self.target_global_ids = jnp.asarray(targets)
        self.source_entity_ids = sources
        self.residuals = jnp.asarray(distances)
        self.resolved = jnp.asarray(resolved_)
        self.ambiguous = jnp.asarray(ambiguous_)
        self.source_dimensions = jnp.asarray(dimensions)
        self.source_indices = jnp.asarray(indices)
        self.parameters = jnp.asarray(parameters_)
        self.orientations = jnp.asarray(orientations_)
        self.parent_dimensions = jnp.asarray(parent_dimensions_)
        self.parent_ids = jnp.asarray(parent_ids_)
        self.exact = bool(exact)
        self.provenance = provenance
        self.parent_association_id = parent
        self.source_roles = roles
        self.source_occurrence_paths = occurrences
        self.source_region_sides = region_sides
        self.source_parameter_intervals = parameter_intervals
        self.target_parameter_intervals = target_intervals
        self.target_reference_pieces = target_pieces
        self.source_boundary_witness_id = source_boundary_witness_id
        self.source_parameter_points = parameter_points
        self.association_id = canonical_fingerprint(
            {
                "kind": "geometry-association",
                "association_kind": association_kind.value,
                "source_id": source,
                "source_revision": revision,
                "target_entity_set_id": entity_set,
                "target_global_ids": array_tree_fingerprint(targets),
                "source_entity_ids": sources,
                "residuals": array_tree_fingerprint(distances),
                "resolved": array_tree_fingerprint(resolved_),
                "ambiguous": array_tree_fingerprint(ambiguous_),
                "parameters": array_tree_fingerprint(parameters_),
                "source_roles": tuple(
                    None if role is None else role.value for role in roles
                ),
                "source_occurrence_paths": occurrences,
                "source_region_sides": region_sides,
                "source_parameter_intervals": tuple(
                    None
                    if interval is None
                    else tuple(coefficient_payload(value) for value in interval)
                    for interval in parameter_intervals
                ),
                **(
                    {
                        "target_parameter_intervals": tuple(
                            None
                            if interval is None
                            else tuple(coefficient_payload(value) for value in interval)
                            for interval in target_intervals
                        )
                    }
                    if any(interval is not None for interval in target_intervals)
                    else {}
                ),
                **(
                    {
                        "target_reference_pieces": tuple(
                            None
                            if pieces is None
                            else tuple(
                                {
                                    "kind": "source-boundary-image-pullback",
                                    "cell_global_id": piece.cell_global_id,
                                    "local_entity_index": piece.local_entity_index,
                                    "image_rational_tokens": piece.image_rational_tokens,
                                    "source_chart_id": piece.source_chart_id,
                                }
                                if isinstance(piece, SourceBoundaryImagePullback)
                                else piece
                                for piece in pieces
                            )
                            for pieces in target_pieces
                        ),
                        "source_boundary_witness_id": source_boundary_witness_id,
                    }
                    if source_boundary_witness_id is not None
                    else {}
                ),
                "source_parameter_points": tuple(
                    None if point is None else coefficient_payload(point)
                    for point in parameter_points
                ),
                "orientations": array_tree_fingerprint(orientations_),
                "parent_dimensions": array_tree_fingerprint(parent_dimensions_),
                "parent_ids": array_tree_fingerprint(parent_ids_),
                "exact": bool(exact),
                "provenance": provenance.value,
                "parent_association_id": parent,
            }
        )

    @property
    def complete(self) -> bool:
        return bool(np.all(np.asarray(self.resolved))) and not bool(
            np.any(np.asarray(self.ambiguous))
        )

    @property
    def maximum_residual(self) -> float:
        """Largest residual over resolved rows (0 when none is resolved)."""
        resolved = np.asarray(self.resolved)
        residuals = np.asarray(self.residuals)[resolved]
        return float(np.max(residuals)) if residuals.size else 0.0

    @checked
    def validate_target(self, entity_set: EntitySet, /) -> None:
        """Require an exact target binding, not merely resolved row statuses."""
        if self.target_entity_set_id != entity_set.entity_set_id:
            raise ValueError("Geometry association targets a different entity set.")
        if not np.all(
            np.isin(np.asarray(self.target_global_ids), np.asarray(entity_set.entity_ids))
        ):
            raise ValueError("Geometry association contains undeclared target IDs.")
        dimension = entity_set.intrinsic_dimension
        if (
            any(value is not None for value in self.target_parameter_intervals)
            and dimension != 1
        ):
            raise ValueError("Target parameter intervals require mesh-edge targets.")
        has_pieces = any(value is not None for value in self.target_reference_pieces)
        if has_pieces and dimension != 2:
            raise ValueError("Target reference pieces require mesh-face targets.")
        if (
            self.source_boundary_witness_id is not None
            and not has_pieces
            and dimension != 1
        ):
            raise ValueError(
                "Witness-bound facet-edge support requires mesh-edge targets."
            )

    def validate_source_boundary_support(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        query: SourceBoundaryQuery,
        witness: SourceBoundaryChartWitness | SourceBoundaryFaceImageWitness,
        /,
        *,
        association_source: NativeLayerCoreSource | None = None,
        layer_core_root_geometry: LayerCoreRootGeometry | None = None,
    ) -> None:
        """Reprove exact target support against its actual source face witness.

        A source edge represented by a segment in a target face additionally
        requires its original ``association_source``. Mapped descendants whose
        restriction binds a retained layer/core root also require that actual
        ``layer_core_root_geometry``; identities alone do not establish a map.
        The witness must describe the incident source facet, not an unrelated
        exterior chart.
        """
        if self.source_boundary_witness_id is None:
            raise ValueError("This association has no source-bound target support.")
        if not isinstance(
            witness, (SourceBoundaryChartWitness, SourceBoundaryFaceImageWitness)
        ):
            raise TypeError("Target support requires an exact source face witness.")
        if (
            witness.witness_id != self.source_boundary_witness_id
            or witness.source_id != self.source_id
            or witness.source_revision != self.source_revision
        ):
            raise ValueError("Target support changes its source witness identity.")
        edge_face = (
            mesh.topological_dimension >= 2
            and self.target_entity_set_id == mesh.entity_set(2).entity_set_id
            and any(role is GeometrySourceEntityRole.EDGE for role in self.source_roles)
        )
        if edge_face:
            from ..geometry._source_fidelity import validate_source_fidelity_witness
            from ._layer_core_segment_overlay import validate_source_edge_face_segments
            from .providers._native_sources import NativeLayerCoreSource

            if not isinstance(association_source, NativeLayerCoreSource):
                raise ValueError(
                    "Source-edge face support requires its actual native source."
                )
            token = validate_source_fidelity_witness(mesh, geometry, query, witness)
            validate_source_edge_face_segments(
                association_source,
                mesh,
                geometry,
                token,
                self,
                root=layer_core_root_geometry,
            )
        else:
            witness.require_current(mesh, geometry, query)
        if self.target_entity_set_id == mesh.entity_set(1).entity_set_id:
            self.validate_target(mesh.entity_set(1))
            edge_groups = (
                source_boundary_image_edge_intervals(mesh, geometry, witness)
                if isinstance(witness, SourceBoundaryFaceImageWitness)
                else source_boundary_edge_intervals(mesh, witness)
            )
            for target, identity, path, interval in zip(
                self.target_global_ids,
                self.source_entity_ids,
                self.source_occurrence_paths,
                self.target_parameter_intervals,
                strict=True,
            ):
                expected = (Fraction(0), Fraction(1)) if interval is None else interval
                if expected not in edge_groups.get(
                    (int(target), identity, tuple(path)), ()
                ):
                    raise ValueError(
                        "Target edge support differs from its exact source witness."
                    )
            return
        self.validate_target(mesh.entity_set(2))
        groups = (
            source_boundary_image_association_piece_rows(witness)
            if isinstance(witness, SourceBoundaryFaceImageWitness)
            else source_boundary_association_piece_rows(witness)
        )
        image_metadata: dict[
            tuple[int, str, tuple[str, ...]], tuple[tuple[str | None, str | None], int]
        ] = {}
        if isinstance(witness, SourceBoundaryFaceImageWitness):
            canonical_orientations = source_boundary_image_face_orientations(
                mesh, geometry, witness
            )
            for face, identity, path, sides in zip(
                witness.entity_global_ids,
                witness.source_entity_ids,
                witness.occurrence_paths,
                witness.source_region_sides,
                strict=True,
            ):
                key = (int(face), identity, tuple(path))
                value = (tuple(sides), canonical_orientations[key])
                previous = image_metadata.setdefault(key, value)
                if previous != value:
                    raise ValueError("Image witness changes source sides or orientation.")
        for row, (target, identity, path, pieces, role) in enumerate(
            zip(
                self.target_global_ids,
                self.source_entity_ids,
                self.source_occurrence_paths,
                self.target_reference_pieces,
                self.source_roles,
                strict=True,
            )
        ):
            if role is GeometrySourceEntityRole.FACET and (
                groups.get((int(target), identity, tuple(path))) != pieces
            ):
                raise ValueError(
                    "Target face support differs from its exact source witness."
                )
            if role is GeometrySourceEntityRole.FACET and image_metadata:
                expected_metadata = image_metadata.get(
                    (int(target), identity, tuple(path))
                )
                if expected_metadata != (
                    self.source_region_sides[row],
                    int(self.orientations[row]),
                ):
                    raise ValueError(
                        "Image facet support changes its source sides or orientation."
                    )
            if role not in (
                GeometrySourceEntityRole.FACET,
                GeometrySourceEntityRole.EDGE,
            ):
                raise ValueError(
                    "Target face support requires a source facet or edge role."
                )

    def target_rows(self, target_global_ids: ArrayLike, /) -> np.ndarray:
        """Host rows of exact target global IDs; every requested ID must be a row."""
        requested = np.asarray(target_global_ids)
        if requested.ndim != 1 or not np.issubdtype(requested.dtype, np.integer):
            raise TypeError("target_global_ids must be one integer vector.")
        targets = np.asarray(self.target_global_ids, dtype=np.int64)
        order = np.argsort(targets, kind="stable")
        position = np.minimum(np.searchsorted(targets[order], requested), order.size - 1)
        rows = order[position]
        if not np.array_equal(targets[rows], requested):
            raise ValueError("Geometry association does not classify every requested ID.")
        return rows


class GeometryAssociationArchiveRecipe(NamedTuple):
    """Archive form of a geometry association with per-row static banks as columns.

    Source IDs, roles, occurrence paths, region sides and exact parameters are
    one static value per target entity; the archive stores each as a value table
    plus an int64 row-index array member instead of one manifest node per row.
    :meth:`restore` rebuilds through the owning constructor and requires the
    recomputed association identity to equal the stored one.
    """

    association_kind: GeometryAssociationKind
    source_id: str
    source_revision: str
    target_entity_set_id: str
    target_global_ids: Array
    source_entity_ids: StaticColumnArchiveRecipe
    source_roles: StaticColumnArchiveRecipe
    source_occurrence_paths: StaticColumnArchiveRecipe
    source_region_sides: StaticColumnArchiveRecipe
    source_parameter_intervals: StaticColumnArchiveRecipe
    target_parameter_intervals: StaticColumnArchiveRecipe
    target_reference_pieces: StaticColumnArchiveRecipe
    source_boundary_witness_id: str | None
    source_parameter_points: StaticColumnArchiveRecipe
    residuals: Array
    resolved: Array
    ambiguous: Array
    source_dimensions: Array
    source_indices: Array
    parameters: Array
    orientations: Array
    parent_dimensions: Array
    parent_ids: Array
    exact: bool
    provenance: GeometryAssociationProvenance
    parent_association_id: str | None
    association_id: str

    @classmethod
    def from_association(
        cls, value: GeometryAssociation, /
    ) -> GeometryAssociationArchiveRecipe:
        if type(value) is not GeometryAssociation:
            raise TypeError("value must be GeometryAssociation.")
        return cls(
            value.association_kind,
            value.source_id,
            value.source_revision,
            value.target_entity_set_id,
            value.target_global_ids,
            StaticColumnArchiveRecipe.encode(value.source_entity_ids),
            StaticColumnArchiveRecipe.encode(value.source_roles),
            StaticColumnArchiveRecipe.encode(value.source_occurrence_paths),
            StaticColumnArchiveRecipe.encode(value.source_region_sides),
            StaticColumnArchiveRecipe.encode(value.source_parameter_intervals),
            StaticColumnArchiveRecipe.encode(value.target_parameter_intervals),
            StaticColumnArchiveRecipe.encode(value.target_reference_pieces),
            value.source_boundary_witness_id,
            StaticColumnArchiveRecipe.encode(value.source_parameter_points),
            value.residuals,
            value.resolved,
            value.ambiguous,
            value.source_dimensions,
            value.source_indices,
            value.parameters,
            value.orientations,
            value.parent_dimensions,
            value.parent_ids,
            value.exact,
            value.provenance,
            value.parent_association_id,
            value.association_id,
        )

    def restore(self) -> GeometryAssociation:
        if type(self.association_kind) is not GeometryAssociationKind:
            raise TypeError("Stored association kind is not registered.")
        dimensions = np.asarray(self.source_dimensions)
        indices = np.asarray(self.source_indices)
        # Rows without a source stratum are stored as -1; when no row names one
        # the constructor's absent-strata form reproduces exactly those banks.
        strata_absent = (
            self.association_kind is not GeometryAssociationKind.BREP
            and bool(np.all(dimensions == -1))
            and bool(np.all(indices == -1))
        )
        restored = GeometryAssociation(
            self.association_kind,
            self.source_id,
            self.source_revision,
            self.target_entity_set_id,
            self.target_global_ids,
            self.source_entity_ids.decode(),
            self.residuals,
            resolved=self.resolved,
            ambiguous=self.ambiguous,
            exact=self.exact,
            source_dimensions=None if strata_absent else dimensions,
            source_indices=None if strata_absent else indices,
            source_roles=self.source_roles.decode(),
            source_occurrence_paths=self.source_occurrence_paths.decode(),
            source_region_sides=self.source_region_sides.decode(),
            source_parameter_intervals=self.source_parameter_intervals.decode(),
            target_parameter_intervals=self.target_parameter_intervals.decode(),
            target_reference_pieces=self.target_reference_pieces.decode(),
            source_boundary_witness_id=self.source_boundary_witness_id,
            source_parameter_points=self.source_parameter_points.decode(),
            parameters=self.parameters,
            orientations=self.orientations,
            parent_dimensions=self.parent_dimensions,
            parent_ids=self.parent_ids,
            parent_association_id=self.parent_association_id,
            provenance=self.provenance,
        )
        if restored.association_id != self.association_id:
            raise ValueError(
                "Stored geometry association differs from its recomputed identity."
            )
        return restored


def _association_optional_text(value: str | None, /) -> str:
    return "0" if value is None else "1" + value


def _association_optional_value(value: str, /) -> str | None:
    if value == "0":
        return None
    if not value.startswith("1"):
        raise ValueError("Association bank optional text is noncanonical.")
    return value[1:]


def _association_exact_row(association: GeometryAssociation, row: int, /) -> tuple[str, ...]:
    role = association.source_roles[row]
    path = association.source_occurrence_paths[row]
    sides = association.source_region_sides[row]
    values = [
        association.source_entity_ids[row],
        _association_optional_text(None if role is None else role.value),
        str(len(path)), *path,
        _association_optional_text(sides[0]), _association_optional_text(sides[1]),
    ]
    for interval in (
        association.source_parameter_intervals[row], association.target_parameter_intervals[row],
    ):
        values.append("0" if interval is None else "1")
        if interval is not None:
            values.extend(str(item) for item in interval)
    point = association.source_parameter_points[row]
    values.append("0" if point is None else "1")
    if point is not None:
        values.append(str(point))
    pieces = association.target_reference_pieces[row]
    values.append("-1" if pieces is None else str(len(pieces)))
    if pieces is not None:
        for piece in pieces:
            if isinstance(piece, SourceBoundaryImagePullback):
                values.extend(("I", str(piece.cell_global_id), str(piece.local_entity_index),
                               piece.source_chart_id, str(len(piece.image_rational_tokens)), "3"))
                points = piece.image_rational_tokens
            else:
                cell, local, vertices, points = piece
                values.extend(("R", str(cell), str(local), str(len(vertices)),
                               *(str(vertex) for vertex in vertices),
                               str(len(points)), str(len(points[0]))))
            for coordinates in points:
                for numerator, denominator in coordinates:
                    values.extend((str(numerator), str(denominator)))
    return tuple(values)


def _association_decode_row(values: tuple[str, ...], /) -> tuple[Any, ...]:
    index = 0

    def text() -> str:
        nonlocal index
        if index >= len(values):
            raise ValueError("Association bank row is truncated.")
        result = values[index]
        index += 1
        return result

    def integer() -> int:
        value = text()
        try:
            result = int(value)
        except ValueError as error:
            raise ValueError("Association bank integer is invalid.") from error
        if str(result) != value:
            raise ValueError("Association bank integer is noncanonical.")
        return result

    def count() -> int:
        result = integer()
        if result < 0 or result > len(values) - index:
            raise ValueError("Association bank row count is invalid.")
        return result

    def fraction() -> Fraction:
        value = text()
        try:
            result = Fraction(value)
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError("Association bank rational is invalid.") from error
        if str(result) != value:
            raise ValueError("Association bank rational is noncanonical.")
        return result

    def optional_fraction_pair() -> tuple[Fraction, Fraction] | None:
        present = text()
        if present == "0":
            return None
        if present != "1":
            raise ValueError("Association bank interval marker is invalid.")
        return fraction(), fraction()

    source = text()
    role_text = _association_optional_value(text())
    role = None if role_text is None else GeometrySourceEntityRole(role_text)
    path = tuple(text() for _ in range(count()))
    sides = (_association_optional_value(text()), _association_optional_value(text()))
    source_interval, target_interval = optional_fraction_pair(), optional_fraction_pair()
    point_marker = text()
    if point_marker not in ("0", "1"):
        raise ValueError("Association bank point marker is invalid.")
    point = None if point_marker == "0" else fraction()
    piece_count = integer()
    if piece_count < -1 or piece_count > len(values) - index:
        raise ValueError("Association bank piece count is invalid.")
    pieces: list[SourceBoundaryTargetPiece] = []
    for _ in range(max(piece_count, 0)):
        kind, cell, local = text(), integer(), integer()
        if kind == "I":
            chart = text()
            vertex_ids: tuple[int, ...] = ()
        elif kind == "R":
            chart = ""
            vertex_ids = tuple(integer() for _ in range(count()))
        else:
            raise ValueError("Association bank piece kind is invalid.")
        point_count, dimension = count(), integer()
        if dimension not in (2, 3) or point_count > (len(values) - index) // (2 * dimension):
            raise ValueError("Association bank piece coordinates have an invalid extent.")
        points = tuple(
            tuple((integer(), integer()) for _ in range(dimension))
            for _ in range(point_count)
        )
        if kind == "I":
            if dimension != 3:
                raise ValueError("Association image pieces require three coordinates.")
            pieces.append(SourceBoundaryImagePullback(cell, local, points, chart))
        else:
            pieces.append((cell, local, vertex_ids, points))
    if index != len(values):
        raise ValueError("Association bank row has trailing data.")
    return (
        source, role, path, sides, source_interval, target_interval,
        None if piece_count == -1 else tuple(pieces), point,
    )


class GeometryAssociationBankArchiveRecipe(NamedTuple):
    """Compact exact columns for an ordered tuple of existing associations."""

    metadata: TextRowsArchiveRecipe
    rows: TextRowsArchiveRecipe
    ends: np.ndarray
    numeric: tuple[np.ndarray, ...]

    @classmethod
    def from_associations(
        cls, values: tuple[GeometryAssociation, ...], /
    ) -> GeometryAssociationBankArchiveRecipe:
        if not values or any(type(value) is not GeometryAssociation for value in values):
            raise TypeError("Association bank requires exact nonempty association owners.")
        names = (
            "target_global_ids", "residuals", "resolved", "ambiguous",
            "source_dimensions", "source_indices", "parameters", "orientations",
            "parent_dimensions", "parent_ids",
        )
        metadata = tuple(
            (value.association_kind.value, value.source_id, value.source_revision,
             value.target_entity_set_id, _association_optional_text(value.source_boundary_witness_id),
             "1" if value.exact else "0", value.provenance.value,
             _association_optional_text(value.parent_association_id), value.association_id)
            for value in values
        )
        lengths = tuple(len(value.source_entity_ids) for value in values)
        ends = np.cumsum(np.asarray(lengths, dtype=np.int64), dtype=np.int64)
        rows = tuple(
            _association_exact_row(value, row)
            for value in values for row in range(len(value.source_entity_ids))
        )
        arrays = tuple(
            np.concatenate(tuple(np.asarray(getattr(value, name)) for value in values), axis=0)
            for name in names
        )
        return cls(TextRowsArchiveRecipe.encode(metadata), TextRowsArchiveRecipe.encode(rows), ends, arrays)

    def restore(self) -> tuple[GeometryAssociation, ...]:
        metadata = self.metadata.decode()
        rows = self.rows.decode()
        ends = np.asarray(self.ends)
        if (
            ends.dtype != np.int64 or ends.ndim != 1 or len(metadata) != len(ends)
            or not len(ends) or np.any(ends <= np.concatenate((np.asarray((0,), dtype=np.int64), ends[:-1])))
            or int(ends[-1]) != len(rows) or len(self.numeric) != 10
        ):
            raise ValueError("Association bank offsets or field inventory are invalid.")
        arrays = tuple(np.asarray(array) for array in self.numeric)
        if any(array.ndim != (2 if index == 6 else 1) or array.shape[0] != len(rows)
               or (index == 6 and array.shape[1] != 2)
               for index, array in enumerate(arrays)):
            raise ValueError("Association bank numeric arrays are not row aligned.")
        from .._array_archive import array_payload_digest

        restored = []
        start = 0
        for meta, end in zip(metadata, ends.tolist(), strict=True):
            if len(meta) != 9 or meta[5] not in ("0", "1"):
                raise ValueError("Association bank metadata has an invalid field set.")
            parsed = tuple(_association_decode_row(row) for row in rows[start:end])
            columns = tuple(array[start:end] for array in arrays)
            dimensions, indices = columns[4], columns[5]
            kind = GeometryAssociationKind(meta[0])
            absent = kind is not GeometryAssociationKind.BREP and bool(
                np.all(dimensions == -1) and np.all(indices == -1)
            )
            item = GeometryAssociation(
                kind, meta[1], meta[2], meta[3], columns[0],
                tuple(row[0] for row in parsed), columns[1],
                resolved=columns[2], ambiguous=columns[3], exact=meta[5] == "1",
                source_dimensions=None if absent else dimensions,
                source_indices=None if absent else indices,
                source_roles=tuple(row[1] for row in parsed),
                source_occurrence_paths=tuple(row[2] for row in parsed),
                source_region_sides=tuple(row[3] for row in parsed),
                source_parameter_intervals=tuple(row[4] for row in parsed),
                target_parameter_intervals=tuple(row[5] for row in parsed),
                target_reference_pieces=tuple(row[6] for row in parsed),
                source_parameter_points=tuple(row[7] for row in parsed),
                source_boundary_witness_id=_association_optional_value(meta[4]),
                parameters=columns[6], orientations=columns[7],
                parent_dimensions=columns[8], parent_ids=columns[9],
                parent_association_id=_association_optional_value(meta[7]),
                provenance=GeometryAssociationProvenance(meta[6]),
            )
            names = (
                "target_global_ids", "residuals", "resolved", "ambiguous",
                "source_dimensions", "source_indices", "parameters", "orientations",
                "parent_dimensions", "parent_ids",
            )
            if item.association_id != meta[8] or any(
                array_payload_digest(np.asarray(getattr(item, name))) != array_payload_digest(value)
                for name, value in zip(names, columns, strict=True)
            ):
                raise ValueError("Association bank row differs from its canonical owner.")
            restored.append(item)
            start = end
        return tuple(restored)


def inherit_association_rows(
    parent: GeometryAssociation,
    target_entity_set: EntitySet,
    target_global_ids: ArrayLike,
    parent_rows: ArrayLike,
    /,
    *,
    residuals: ArrayLike,
    orientations: ArrayLike | None = None,
    source_parameter_intervals: tuple[tuple[Fraction, Fraction] | None, ...]
    | None = None,
    target_parameter_intervals: tuple[tuple[Fraction, Fraction] | None, ...]
    | None = None,
    source_parameter_points: tuple[Fraction | None, ...] | None = None,
    source_region_sides: tuple[tuple[str | None, str | None], ...] | None = None,
    resolved: ArrayLike | None = None,
    ambiguous: ArrayLike | None = None,
) -> GeometryAssociation:
    """Compose stable source classes through explicit parent-association rows."""
    if not isinstance(parent, GeometryAssociation) or not isinstance(
        target_entity_set, EntitySet
    ):
        raise TypeError(
            "Parent association and target entity set must be canonical owners."
        )
    targets = np.asarray(target_global_ids, dtype=np.int64)
    rows = np.asarray(parent_rows)
    count = targets.size
    if (
        targets.ndim != 1
        or rows.ndim != 1
        or rows.dtype.kind not in "iu"
        or rows.shape != targets.shape
        or np.any(rows < 0)
        or np.any(rows >= parent.target_global_ids.shape[0])
        or targets.shape != (target_entity_set.count,)
        or not np.array_equal(targets, np.asarray(target_entity_set.entity_ids))
    ):
        raise ValueError("Association parent rows and target IDs must align exactly.")
    residual = np.asarray(residuals, dtype=np.float64)
    if residual.shape != (count,):
        raise ValueError("Association child residuals must align with target rows.")
    if parent.source_boundary_witness_id is not None:
        raise ValueError(
            "Witness-bound target faces require an independently composed child witness."
        )
    source_ids = tuple(parent.source_entity_ids[int(row)] for row in rows)
    roles = tuple(parent.source_roles[int(row)] for row in rows)
    occurrences = tuple(parent.source_occurrence_paths[int(row)] for row in rows)
    region_sides = (
        tuple(parent.source_region_sides[int(row)] for row in rows)
        if source_region_sides is None
        else source_region_sides
    )
    intervals = (
        tuple(parent.source_parameter_intervals[int(row)] for row in rows)
        if source_parameter_intervals is None
        else source_parameter_intervals
    )
    if target_parameter_intervals is None and any(
        parent.target_parameter_intervals[int(row)] is not None for row in rows
    ):
        raise ValueError(
            "Partial target-edge support requires explicit child reference intervals."
        )
    target_intervals = (
        tuple(None for _ in rows)
        if target_parameter_intervals is None
        else target_parameter_intervals
    )
    parameter_points = (
        tuple(parent.source_parameter_points[int(row)] for row in rows)
        if source_parameter_points is None
        else source_parameter_points
    )
    resolved_ = (
        np.asarray(parent.resolved, dtype=np.bool_)[rows]
        if resolved is None
        else np.asarray(resolved, dtype=np.bool_)
    )
    ambiguous_ = (
        np.asarray(parent.ambiguous, dtype=np.bool_)[rows]
        if ambiguous is None
        else np.asarray(ambiguous, dtype=np.bool_)
    )
    orientation = (
        np.asarray(parent.orientations, dtype=np.int8)[rows]
        if orientations is None
        else np.asarray(orientations, dtype=np.int8)
    )
    source_dimensions = (
        np.asarray(parent.source_dimensions, dtype=np.int8)[rows]
        if parent.association_kind is GeometryAssociationKind.BREP
        else None
    )
    source_indices = (
        np.asarray(parent.source_indices, dtype=np.int32)[rows]
        if parent.association_kind is GeometryAssociationKind.BREP
        else None
    )
    return GeometryAssociation(
        parent.association_kind,
        parent.source_id,
        parent.source_revision,
        target_entity_set.entity_set_id,
        targets,
        source_ids,
        residual,
        resolved=resolved_,
        ambiguous=ambiguous_,
        exact=parent.exact
        and bool(np.all(resolved_))
        and not bool(np.any(ambiguous_))
        and bool(np.all(residual == 0.0)),
        source_dimensions=source_dimensions,
        source_indices=source_indices,
        source_roles=roles,
        source_occurrence_paths=occurrences,
        source_region_sides=region_sides,
        source_parameter_intervals=intervals,
        target_parameter_intervals=target_intervals,
        source_parameter_points=parameter_points,
        parameters=np.asarray(parent.parameters)[rows],
        orientations=orientation,
        parent_association_id=parent.association_id,
        provenance=GeometryAssociationProvenance.LINEAGE,
    )


def _row_metadata(
    count: int,
    parameters: ArrayLike | None,
    orientations: ArrayLike | None,
    parent_dimensions: ArrayLike | None,
    parent_ids: ArrayLike | None,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validated closest-point parameters, orientations, and parents of every row."""
    parameters_ = _rows_or(parameters, np.nan, (count, 2), np.float64)
    orientations_ = _rows_or(orientations, 0, (count,), np.int8)
    parent_dimensions_ = _rows_or(parent_dimensions, -1, (count,), np.int8)
    parent_ids_ = _rows_or(parent_ids, -1, (count,), np.int64)
    if (
        parameters_.shape != (count, 2)
        or np.any(np.isinf(parameters_))
        or orientations_.shape != (count,)
        or parent_dimensions_.shape != (count,)
        or parent_ids_.shape != (count,)
    ):
        raise ValueError(
            "Association parameters, orientations, and parents must be aligned."
        )
    if not np.all(np.isin(orientations_, (-1, 0, 1))):
        raise ValueError("Association orientations must be -1, 0, or 1.")
    if (
        np.any((parent_dimensions_ < -1) | (parent_dimensions_ > 3))
        or np.any(parent_ids_ < -1)
        or np.any((parent_dimensions_ < 0) != (parent_ids_ < 0))
    ):
        raise ValueError("Association parents must pair a dimension with an ID.")
    return parameters_, orientations_, parent_dimensions_, parent_ids_


def _source_entities(
    kind: GeometryAssociationKind,
    source: str,
    revision: str,
    sources: tuple[str, ...],
    dimensions: ArrayLike | None,
    indices: ArrayLike | None,
    occurrence_paths: tuple[tuple[str, ...], ...],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    count = len(sources)
    if dimensions is None and indices is None:
        if kind is GeometryAssociationKind.BREP:
            raise ValueError("B-Rep associations require source dimensions and indices.")
        return np.full((count,), -1, np.int8), np.full((count,), -1, np.int64)
    if dimensions is None or indices is None:
        raise ValueError("Source dimensions and indices must be supplied together.")
    dimensions_ = np.asarray(dimensions)
    indices_ = np.asarray(indices)
    if not np.issubdtype(dimensions_.dtype, np.integer) or not np.issubdtype(
        indices_.dtype, np.integer
    ):
        raise TypeError("Source dimensions and indices must be integers.")
    if dimensions_.shape != (count,) or indices_.shape != (count,):
        raise ValueError("Source dimensions and indices must align with rows.")
    if (
        np.any((dimensions_ < -1) | (dimensions_ > 3))
        or np.any(indices_ < -1)
        or np.any(indices_ > np.iinfo(np.int64).max)
        or np.any((dimensions_ >= 0) != (indices_ >= 0))
    ):
        raise ValueError("Source entities must pair a dimension with an int64 index.")
    if kind is GeometryAssociationKind.BREP:
        rows = tuple(
            zip(dimensions_.tolist(), indices_.tolist(), occurrence_paths, strict=True)
        )
        # Rows repeat a few source entities: key each distinct entity once.
        keys = {
            (dimension, index, path): (
                brep_entity_id(source, revision, dimension, index, occurrence_path=path)
                if dimension >= 0
                else _unclassified_id(source, revision, path)
            )
            for dimension, index, path in dict.fromkeys(rows)
        }
        expected = tuple(keys[row] for row in rows)
        if expected != sources:
            raise ValueError(
                "B-Rep source entity keys must match their revision, definition and occurrence."
            )
    return dimensions_.astype(np.int8), indices_.astype(np.int64)


# -- policy and errors ---------------------------------------------------------


class AssociationPropagationPolicy(StrictModule, NonTrainableState):
    """Tolerances of association propagation and re-derivation.

    ``classification_tolerance`` is the absolute distance within which geometric
    classification places a point on a B-Rep entity. ``maximum_residual`` marks
    rows whose projection residual exceeds it unresolved; ``None`` keeps residuals
    as evidence only.
    """

    classification_tolerance: float = eqx.field(static=True)
    maximum_residual: float | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        classification_tolerance: float = 1.0e-6,
        maximum_residual: float | None = None,
    ) -> None:
        tolerance = float(classification_tolerance)
        maximum = None if maximum_residual is None else float(maximum_residual)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("classification_tolerance must be finite and positive.")
        if maximum is not None and (not np.isfinite(maximum) or maximum < 0.0):
            raise ValueError("maximum_residual must be finite and non-negative or None.")
        self.classification_tolerance = tolerance
        self.maximum_residual = maximum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "association-propagation-policy",
                "classification_tolerance": tolerance,
                "maximum_residual": maximum,
            }
        )


class AssociationPropagationError(ValueError):
    """A topology change violates the B-Rep classification rules."""

    def __init__(self, message: str, target_ids: ArrayLike, /) -> None:
        super().__init__(message)
        self.target_ids = np.asarray(target_ids, dtype=np.int64)


# -- mesh topology helpers -------------------------------------------------------


def _compose(
    first: np.ndarray, second: np.ndarray, /, *, unique: bool = True
) -> np.ndarray:
    """Compose ``(a, b)`` pairs with ``(b, c)`` pairs into ``(a, c)`` pairs.

    With ``unique=False`` every path is kept, so multiplicities count the
    intermediate ``b`` connecting each ``(a, c)``.
    """
    if first.shape[0] == 0 or second.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    order = np.argsort(second[:, 0], kind="stable")
    keys = second[order, 0]
    start = np.searchsorted(keys, first[:, 1], side="left")
    counts = np.searchsorted(keys, first[:, 1], side="right") - start
    owner = np.repeat(np.arange(first.shape[0]), counts)
    offsets = np.cumsum(counts) - counts
    position = np.repeat(start - offsets, counts) + np.arange(owner.size)
    pairs = np.stack((first[owner, 0], second[order[position], 1]), axis=1)
    return np.unique(pairs, axis=0).reshape(-1, 2) if unique else pairs


def _incidence_pairs(mesh: CellMesh, low: int, high: int, /) -> np.ndarray:
    """``(low_row, high_row)`` pairs with the low entity in the high entity's closure."""
    if low == high:
        rows = np.arange(mesh.entity_set(low).count, dtype=np.int64)
        return np.stack((rows, rows), axis=1)
    pairs = None
    for degree in range(low, high):
        relation = mesh.topology.incidences[degree].relation
        valid = np.asarray(relation.valid, dtype=np.bool_).reshape(-1)
        step = np.stack(
            (
                np.asarray(relation.source_indices, dtype=np.int64).reshape(-1)[valid],
                np.asarray(relation.target_indices, dtype=np.int64).reshape(-1)[valid],
            ),
            axis=1,
        )
        pairs = step if pairs is None else _compose(pairs, step)
    # ty: ignore[no-matching-overload]
    return np.unique(pairs, axis=0).reshape(-1, 2)


def _boundary_mask(mesh: CellMesh, dimension: int, /) -> np.ndarray:
    """Entities of ``dimension`` in the closure of mesh boundary facets."""
    top = mesh.topological_dimension
    facets = _incidence_pairs(mesh, top - 1, top)
    boundary_facets = np.flatnonzero(
        np.bincount(facets[:, 0], minlength=mesh.entity_set(top - 1).count) == 1
    )
    mask = np.zeros((mesh.entity_set(dimension).count,), dtype=np.bool_)
    if dimension == top:
        return mask
    pairs = _incidence_pairs(mesh, dimension, top - 1)
    mask[pairs[np.isin(pairs[:, 1], boundary_facets), 0]] = True
    return mask


def _vertex_rows(mesh: CellMesh, identifiers: np.ndarray, /) -> np.ndarray:
    """Mesh vertex row of each global ID (-1 when absent)."""
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    order = np.argsort(vertex_ids, kind="stable")
    position = np.minimum(np.searchsorted(vertex_ids[order], identifiers), order.size - 1)
    return np.where(vertex_ids[order[position]] == identifiers, order[position], -1)


def _entity_rows(
    mesh: CellMesh, dimension: int, identifiers: np.ndarray, /
) -> np.ndarray:
    entity_ids = np.asarray(mesh.entity_set(dimension).entity_ids, dtype=np.int64)
    order = np.argsort(entity_ids, kind="stable")
    position = np.minimum(np.searchsorted(entity_ids[order], identifiers), order.size - 1)
    return np.where(entity_ids[order[position]] == identifiers, order[position], -1)


def _ordered_vertices(mesh: CellMesh, dimension: int, /) -> np.ndarray | None:
    """Canonical vertex rows of every entity of ``dimension`` (simplex routes)."""
    connectivity = mesh.connectivity
    if (
        isinstance(connectivity, SimplicialConnectivity)
        and 1 <= dimension < mesh.topological_dimension
    ):
        return np.asarray(connectivity.entities[dimension], dtype=np.int64)
    if dimension == 1 and isinstance(
        connectivity, (PolygonalConnectivity, TetrahedralConnectivity)
    ):
        return np.asarray(connectivity.edges, dtype=np.int64)
    if dimension == 2 and isinstance(connectivity, TetrahedralConnectivity):
        return np.asarray(connectivity.faces, dtype=np.int64)
    if dimension == mesh.topological_dimension == 2 and all(
        block.cell_kind == "triangle" for block in mesh.blocks
    ):
        identifiers = np.concatenate(
            [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
        )
        cells = np.concatenate(
            [np.asarray(block.vertices, dtype=np.int64) for block in mesh.blocks]
        )
        rows = _entity_rows(mesh, dimension, identifiers)
        ordered = np.empty_like(cells)
        ordered[rows] = cells
        return ordered
    return None


# -- classification --------------------------------------------------------------


_RESOLVED_STATUS = (int(BRepProjectionStatus.UNIQUE), int(BRepProjectionStatus.SEAM))


class _Classes(NamedTuple):
    """B-Rep class of every mesh entity row of one dimension (-1 unclassified)."""

    dimensions: np.ndarray
    indices: np.ndarray
    status: np.ndarray
    occurrence_paths: tuple[tuple[str, ...], ...]

    @property
    def resolved(self) -> np.ndarray:
        return np.isin(self.status, _RESOLVED_STATUS)


class _ClassificationGroups(NamedTuple):
    vertex_groups: np.ndarray
    allowed_groups: np.ndarray
    allowed_dimensions: np.ndarray
    allowed_indices: np.ndarray
    allowed_occurrence_paths: tuple[tuple[str, ...], ...]


def _common_occurrences(
    members: np.ndarray,
    member_paths: tuple[tuple[str, ...], ...],
    count: int,
    /,
) -> tuple[tuple[tuple[str, ...], ...], np.ndarray]:
    """A definition-level incidence inference must stay in one physical occurrence."""
    paths: list[tuple[str, ...] | None] = [None] * count
    conflicts = np.zeros(count, dtype=np.bool_)
    for owner, member in members:
        row = int(owner)
        path = member_paths[int(member)]
        if paths[row] is None:
            paths[row] = path
        elif paths[row] != path:
            conflicts[row] = True
    return (
        tuple(
            () if path is None or conflicts[row] else path
            for row, path in enumerate(paths)
        ),
        conflicts,
    )


def _vertex_classes(mesh: CellMesh, association: GeometryAssociation, /) -> _Classes:
    """Vertex classes in mesh row order; the association must cover every vertex."""
    rows = _entity_rows(mesh, 0, np.asarray(association.target_global_ids))
    count = mesh.coordinates.shape[0]
    if np.any(rows < 0) or np.unique(rows).size != count or rows.size != count:
        raise ValueError("The vertex association must cover exactly the mesh vertices.")
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    dims[rows] = np.asarray(association.source_dimensions, dtype=np.int64)
    indices[rows] = np.asarray(association.source_indices, dtype=np.int64)
    status[rows] = np.where(
        np.asarray(association.resolved),
        BRepProjectionStatus.UNIQUE,
        np.where(
            np.asarray(association.ambiguous),
            BRepProjectionStatus.AMBIGUOUS,
            BRepProjectionStatus.FAILED,
        ),
    )
    paths: list[tuple[str, ...]] = [()] * count
    for row, path in zip(rows.tolist(), association.source_occurrence_paths, strict=True):
        paths[row] = path
    return _Classes(dims, indices, status, tuple(paths))


def _centroids(mesh: CellMesh, members: np.ndarray, count: int, /) -> np.ndarray:
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    totals = np.zeros((count, points.shape[1]))
    np.add.at(totals, members[:, 0], points[members[:, 1]])
    return totals / np.maximum(np.bincount(members[:, 0], minlength=count), 1)[:, None]


def _candidate_residuals(
    projection: AbstractBRepProjection,
    points: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    occurrence_paths: tuple[tuple[str, ...], ...],
    /,
) -> np.ndarray:
    """Distance of each point to its candidate entity (solids: 0 inside, else inf)."""
    residuals = np.full((points.shape[0],), np.inf)
    lower = np.flatnonzero(dimensions <= 2)
    if lower.size:
        result = projection.project(
            points[lower],
            dimensions[lower],
            indices[lower],
            occurrence_paths=tuple(occurrence_paths[row] for row in lower.tolist()),
        )
        residuals[lower] = np.where(
            np.isin(np.asarray(result.status), _RESOLVED_STATUS),
            np.asarray(result.residuals),
            np.inf,
        )
    solid = np.flatnonzero(dimensions == 3)
    if solid.size:
        located = projection.locate_solids(points[solid])
        inside = (np.asarray(located.dimensions) == 3) & (
            np.asarray(located.indices) == indices[solid]
        )
        inside &= np.isin(np.asarray(located.status), _RESOLVED_STATUS)
        inside &= np.asarray(
            [
                actual == occurrence_paths[row]
                for row, actual in zip(
                    solid.tolist(), located.occurrence_paths, strict=True
                )
            ],
            dtype=np.bool_,
        )
        residuals[solid] = np.where(inside, 0.0, np.inf)
    return residuals


def _lowest_containers(
    projection: AbstractBRepProjection,
    members: np.ndarray,
    member_classes: _Classes,
    count: int,
    low: int,
    high: np.ndarray,
    within: np.ndarray | None,
    centroids: np.ndarray,
    /,
) -> _Classes:
    """Lowest-dimensional B-Rep entity containing every member class of each owner.

    ``members`` are ``(owner, member_row)`` pairs. Candidates have dimension in
    ``[low, high[owner]]`` and lie in the closure of every ``within`` entity code
    listed for their owner. Equal-dimension ties are broken by the owner
    centroid's distance to each candidate; indistinguishable ties are AMBIGUOUS.
    """
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    paths, occurrence_conflicts = _common_occurrences(
        members, member_classes.occurrence_paths, count
    )
    member_dims = member_classes.dimensions[members[:, 1]]
    classified = (member_dims >= 0) & member_classes.resolved[members[:, 1]]
    failed = np.bincount(members[~classified, 0], minlength=count) > 0
    failed |= np.bincount(members[:, 0], minlength=count) == 0
    failed |= occurrence_conflicts
    ambiguous_members = (
        np.bincount(
            members[
                member_classes.status[members[:, 1]] == BRepProjectionStatus.AMBIGUOUS, 0
            ],
            minlength=count,
        )
        > 0
    )
    status[occurrence_conflicts | ambiguous_members] = BRepProjectionStatus.AMBIGUOUS
    classified &= ~failed[members[:, 0]]
    codes = projection.entity_codes(
        member_dims[classified], member_classes.indices[members[classified, 1]]
    )
    owner_class = np.unique(
        np.stack((members[classified, 0], codes), axis=1), axis=0
    ).reshape(-1, 2)
    if owner_class.shape[0] == 0:
        return _Classes(dims, indices, status, paths)
    class_count = np.bincount(owner_class[:, 0], minlength=count)
    distinct, inverse = np.unique(owner_class[:, 1], return_inverse=True)
    row, container_dims, container_indices = projection.containers(
        *projection.code_entities(distinct)
    )
    keep = container_dims >= low
    reach = _compose(
        np.stack((owner_class[:, 0], inverse.reshape(-1)), axis=1),
        np.stack(
            (
                row[keep],
                projection.entity_codes(container_dims[keep], container_indices[keep]),
            ),
            axis=1,
        ),
        unique=False,
    )
    candidates, hits = np.unique(reach, axis=0, return_counts=True)
    candidates = candidates.reshape(-1, 2)
    candidate_dims, candidate_indices = projection.code_entities(candidates[:, 1])
    admissible = (hits == class_count[candidates[:, 0]]) & (
        candidate_dims <= high[candidates[:, 0]]
    )
    if within is not None and candidates.shape[0]:
        joined = _compose(
            np.stack((np.arange(candidates.shape[0]), candidates[:, 0]), axis=1),
            within,
            unique=False,
        )
        container_dims_, container_indices_ = projection.code_entities(joined[:, 1])
        inside = projection.contains(
            container_dims_,
            container_indices_,
            candidate_dims[joined[:, 0]],
            candidate_indices[joined[:, 0]],
        )
        admissible[joined[~inside, 0]] = False
    candidates = candidates[admissible]
    candidate_dims = candidate_dims[admissible]
    candidate_indices = candidate_indices[admissible]
    if candidates.shape[0]:
        lowest = np.full((count,), 4, dtype=np.int64)
        np.minimum.at(lowest, candidates[:, 0], candidate_dims)
        at = candidate_dims == lowest[candidates[:, 0]]
        owners = candidates[at, 0]
        at_dims, at_indices = candidate_dims[at], candidate_indices[at]
        ties = np.bincount(owners, minlength=count)
        residual = np.zeros(owners.shape)
        tied = ties[owners] > 1
        if np.any(tied):
            residual[tied] = _candidate_residuals(
                projection,
                centroids[owners[tied]],
                at_dims[tied],
                at_indices[tied],
                tuple(paths[row] for row in owners[tied].tolist()),
            )
        order = np.lexsort((at_indices, residual, owners))
        owners, at_dims, at_indices, residual = (
            owners[order],
            at_dims[order],
            at_indices[order],
            residual[order],
        )
        first = np.unique(owners, return_index=True)[1]
        second = np.minimum(first + 1, owners.size - 1)
        distinguishable = (ties[owners[first]] == 1) | (
            (owners[second] == owners[first])
            & (residual[second] > residual[first] + projection.policy.ambiguity_tolerance)
        )
        rows = owners[first]
        dims[rows] = at_dims[first]
        indices[rows] = at_indices[first]
        status[rows] = np.where(
            distinguishable, BRepProjectionStatus.UNIQUE, BRepProjectionStatus.AMBIGUOUS
        )
    dims[failed] = -1
    indices[failed] = -1
    status[failed] = BRepProjectionStatus.FAILED
    status[occurrence_conflicts | ambiguous_members] = BRepProjectionStatus.AMBIGUOUS
    return _Classes(dims, indices, status, paths)


def _classify_mesh(
    mesh: CellMesh,
    vertex_classes: _Classes,
    projection: AbstractBRepProjection,
    /,
) -> tuple[_Classes, ...]:
    """B-Rep classes of every mesh entity dimension ``0..D`` (module rules)."""
    top = mesh.topological_dimension
    classes: list[_Classes] = [vertex_classes] * (top + 1)
    members = _incidence_pairs(mesh, 0, top)[:, ::-1]
    count = mesh.entity_set(top).count
    classes[top] = _lowest_containers(
        projection,
        members,
        vertex_classes,
        count,
        top,
        np.full((count,), top),
        None,
        _centroids(mesh, members, count),
    )
    for dimension in range(top - 1, 0, -1):
        classes[dimension] = _classify_level(
            mesh, projection, vertex_classes, classes[dimension + 1], dimension
        )
    return tuple(classes)


def _classify_level(
    mesh: CellMesh,
    projection: AbstractBRepProjection,
    vertex_classes: _Classes,
    upper: _Classes,
    dimension: int,
    /,
) -> _Classes:
    """Classify entities of ``dimension`` from their classified cofaces.

    The cofaces on the lowest-dimensional adjacent classes decide: one such class
    that the entity does not bound (a coface class of higher dimension, or exactly
    two cofaces on it) is inherited; otherwise the entity lies on the lowest
    B-Rep entity containing its vertex classes within the closure of every
    adjacent lowest class.
    """
    count = mesh.entity_set(dimension).count
    pairs = _incidence_pairs(mesh, dimension, dimension + 1)
    good = np.isin(upper.status[pairs[:, 1]], _RESOLVED_STATUS)
    unresolved = np.bincount(pairs[~good, 0], minlength=count) > 0
    pairs = pairs[good]
    coface_paths, occurrence_conflicts = _common_occurrences(
        pairs, upper.occurrence_paths, count
    )
    unresolved |= occurrence_conflicts
    upper_dims = upper.dimensions[pairs[:, 1]]
    lowest = np.full((count,), 4, dtype=np.int64)
    np.minimum.at(lowest, pairs[:, 0], upper_dims)
    at = upper_dims == lowest[pairs[:, 0]]
    adjacent = np.unique(
        np.stack(
            (
                pairs[at, 0],
                projection.entity_codes(upper_dims[at], upper.indices[pairs[at, 1]]),
            ),
            axis=1,
        ),
        axis=0,
    ).reshape(-1, 2)
    distinct = np.bincount(adjacent[:, 0], minlength=count)
    cofaces = np.bincount(pairs[at, 0], minlength=count)
    inherited = (
        (distinct == 1) & ((lowest > dimension + 1) | (cofaces == 2)) & ~unresolved
    )
    members = _incidence_pairs(mesh, 0, dimension)[:, ::-1]
    # Only entities bounding their coface classes search for a lower class.
    bounding = members[~inherited[members[:, 0]]]
    bounded = _lowest_containers(
        projection,
        bounding,
        vertex_classes,
        count,
        dimension,
        np.minimum(lowest, 4) - 1,
        adjacent,
        _centroids(mesh, members, count),
    )
    code = np.full((count,), -1, dtype=np.int64)
    code[adjacent[:, 0]] = adjacent[:, 1]
    inherited_dims, inherited_indices = projection.code_entities(np.maximum(code, 0))
    dims = np.where(inherited, inherited_dims, bounded.dimensions)
    indices = np.where(inherited, inherited_indices, bounded.indices)
    status = np.where(inherited, BRepProjectionStatus.UNIQUE, bounded.status)
    status = np.where(
        unresolved & ~inherited & (status != BRepProjectionStatus.FAILED),
        BRepProjectionStatus.AMBIGUOUS,
        status,
    ).astype(np.int8)
    paths = tuple(
        coface_paths[row] if inherited[row] else bounded.occurrence_paths[row]
        for row in range(count)
    )
    return _Classes(dims.astype(np.int64), indices.astype(np.int64), status, paths)


def _orientations(
    mesh: CellMesh,
    dimension: int,
    classes: _Classes,
    normals: np.ndarray,
    tangents: np.ndarray,
    /,
) -> np.ndarray:
    """Relation of each entity's canonical vertex order to its B-Rep orientation."""
    result = np.zeros((classes.dimensions.size,), dtype=np.int8)
    ordered = _ordered_vertices(mesh, dimension)
    if ordered is None:
        return result
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    if dimension == 1:
        rows = np.flatnonzero(classes.dimensions == 1)
        tangent = tangents[rows, 0]
        direction = points[ordered[rows, 1]] - points[ordered[rows, 0]]
        sign = np.sign(np.sum(direction * tangent, axis=1))
    elif dimension == 2 and mesh.ambient_dimension == 3:
        rows = np.flatnonzero(classes.dimensions == 2)
        normal = normals[rows]
        corners = points[ordered[rows]]
        facet = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        sign = np.sign(np.sum(facet * normal, axis=1))
    else:
        return result
    result[rows] = np.where(np.isfinite(sign), sign, 0).astype(np.int8)
    return result


def _brep_association(
    projection: AbstractBRepProjection,
    mesh: CellMesh,
    dimension: int,
    classes: _Classes,
    points: np.ndarray,
    policy: AssociationPropagationPolicy,
    /,
    *,
    provenance: GeometryAssociationProvenance,
    parent_association_id: str | None,
    parent_dimensions: np.ndarray | None = None,
    parent_ids: np.ndarray | None = None,
) -> GeometryAssociation:
    """Project classified entity points and assemble the audited association."""
    count = classes.dimensions.size
    entity_set = mesh.entity_set(dimension)
    # Solid rows inherit region membership; only vertices, edges, and faces project.
    lower = np.flatnonzero((classes.dimensions >= 0) & (classes.dimensions <= 2))
    residuals = np.full(count, np.inf, dtype=np.float64)
    residuals[classes.dimensions == 3] = 0.0
    parameters = np.full((count, 2), np.nan, dtype=np.float64)
    normals = np.full(points.shape, np.nan, dtype=np.float64)
    tangents = np.full((count, 2, points.shape[1]), np.nan, dtype=np.float64)
    projection_status = np.full(count, BRepProjectionStatus.FAILED, dtype=np.int8)
    projection_status[classes.dimensions == 3] = BRepProjectionStatus.UNIQUE
    if lower.size:
        projected = projection.project(
            points[lower],
            classes.dimensions[lower],
            classes.indices[lower],
            occurrence_paths=tuple(
                classes.occurrence_paths[row] for row in lower.tolist()
            ),
        )
        residuals[lower] = np.asarray(projected.residuals)
        parameters[lower] = np.asarray(projected.parameters)
        projection_status[lower] = np.asarray(projected.status)
        normals[lower] = np.asarray(projected.normals)
        tangents[lower] = np.asarray(projected.tangents)
    ambiguous = (classes.status == BRepProjectionStatus.AMBIGUOUS) | (
        projection_status == BRepProjectionStatus.AMBIGUOUS
    )
    resolved = (
        np.isin(classes.status, _RESOLVED_STATUS)
        & np.isin(projection_status, _RESOLVED_STATUS)
        & (
            residuals <= policy.maximum_residual
            if policy.maximum_residual is not None
            else np.isfinite(residuals)
        )
    )
    return GeometryAssociation(
        GeometryAssociationKind.BREP,
        projection.source_id,
        projection.source_revision,
        entity_set.entity_set_id,
        np.asarray(entity_set.entity_ids, dtype=np.int64),
        tuple(
            brep_entity_id(
                projection.source_id,
                projection.source_revision,
                entity,
                index,
                occurrence_path=path,
            )
            if entity >= 0
            else _unclassified_id(projection.source_id, projection.source_revision, path)
            for entity, index, path in zip(
                classes.dimensions.tolist(),
                classes.indices.tolist(),
                classes.occurrence_paths,
                strict=True,
            )
        ),
        residuals,
        resolved=resolved,
        ambiguous=ambiguous & ~resolved,
        source_dimensions=classes.dimensions,
        source_indices=classes.indices,
        source_occurrence_paths=classes.occurrence_paths,
        parameters=parameters,
        orientations=_orientations(mesh, dimension, classes, normals, tangents),
        parent_dimensions=parent_dimensions,
        parent_ids=parent_ids,
        parent_association_id=parent_association_id,
        provenance=provenance,
    )


def _require_brep(
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    mesh: CellMesh,
    dimension: int,
    /,
) -> None:
    if not isinstance(association, GeometryAssociation):
        raise TypeError("association must be GeometryAssociation.")
    if not isinstance(projection, AbstractBRepProjection):
        raise TypeError("projection must be AbstractBRepProjection.")
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if association.association_kind is not GeometryAssociationKind.BREP:
        raise ValueError("B-Rep classification requires a B-Rep association.")
    if association.source_revision != projection.source_revision:
        raise ValueError("The association and projection bind different revisions.")
    if association.target_entity_set_id != mesh.entity_set(dimension).entity_set_id:
        raise ValueError("The association does not target this mesh's entities.")
    if mesh.ambient_dimension != projection.ambient_dimension:
        raise ValueError("Mesh and projection ambient dimensions differ.")


def _policy(policy: AssociationPropagationPolicy, /) -> AssociationPropagationPolicy:
    if not isinstance(policy, AssociationPropagationPolicy):
        raise TypeError("policy must be AssociationPropagationPolicy.")
    return policy


# -- public classification ----------------------------------------------------------


def _mesh_entity_classes(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    /,
) -> tuple[_Classes, ...]:
    """Host B-Rep classes ``(dimensions, indices, status)`` of mesh dimensions ``0..D``.

    ``association`` is the complete vertex association of ``mesh``. Consumers
    (adaptation constraints, curving) use the classes to protect, constrain, and
    project entities by the module rules.
    """
    _require_brep(association, projection, mesh, 0)
    return _classify_mesh(mesh, _vertex_classes(mesh, association), projection)


def associate_mesh_entities(
    mesh: CellMesh,
    association: GeometryAssociation,
    projection: AbstractBRepProjection,
    dimension: int,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Derive the B-Rep association of mesh entities of ``dimension`` (1..D).

    Classes follow the module rules from the complete vertex ``association``;
    residuals and parameters come from projecting each entity centroid (region
    rows carry zero residual), orientations from the canonical vertex order.
    """
    policy_ = _policy(policy)
    target = int(dimension)
    if not 1 <= target <= mesh.topological_dimension:
        raise ValueError("dimension must lie in 1..D; vertex associations are direct.")
    classes = _mesh_entity_classes(mesh, association, projection)[target]
    members = _incidence_pairs(mesh, 0, target)[:, ::-1]
    count = mesh.entity_set(target).count
    return _brep_association(
        projection,
        mesh,
        target,
        classes,
        _centroids(mesh, members, count),
        policy_,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=association.association_id,
    )


def _geometric_vertex_classes(
    mesh: CellMesh,
    projection: AbstractBRepProjection,
    tolerance: float,
    groups: _ClassificationGroups | None,
    /,
) -> _Classes:
    """Lowest-dimensional B-Rep entity within ``tolerance`` of every mesh vertex.

    Mesh-boundary vertices classify on entities of dimension below the mesh
    dimension (optionally restricted by transfer ``groups``); interior vertices
    of volume meshes lie on an interface face within tolerance or in the solid
    containing them; every vertex of a two-dimensional mesh classifies on its
    lowest entity within tolerance.
    """
    top = mesh.topological_dimension
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    count = points.shape[0]
    boundary = _boundary_mask(mesh, 0)
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    paths: list[tuple[str, ...]] = [()] * count

    def assign(rows: np.ndarray, result: BRepProjectionResult, /) -> None:
        dims[rows] = np.asarray(result.dimensions, dtype=np.int64)
        indices[rows] = np.asarray(result.indices, dtype=np.int64)
        status[rows] = np.asarray(result.status)
        for row, path in zip(rows.tolist(), result.occurrence_paths, strict=True):
            paths[row] = path

    rows = np.flatnonzero(boundary)
    if rows.size:
        assign(
            rows,
            projection.classify(
                points[rows],
                tolerance=tolerance,
                maximum_dimension=min(top - 1, 2),
                groups=None if groups is None else groups.vertex_groups[rows],
                allowed_groups=None if groups is None else groups.allowed_groups,
                allowed_dimensions=None if groups is None else groups.allowed_dimensions,
                allowed_indices=None if groups is None else groups.allowed_indices,
                allowed_occurrence_paths=(
                    None if groups is None else groups.allowed_occurrence_paths
                ),
            ),
        )
    rows = np.flatnonzero(~boundary)
    if rows.size:
        assign(rows, projection.classify(points[rows], tolerance=tolerance))
        if top == 3:
            inside = rows[dims[rows] < 0]
            if inside.size:
                assign(inside, projection.locate_solids(points[inside]))
    return _Classes(dims, indices, status, tuple(paths))


def associate_mesh_vertices(
    mesh: CellMesh,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Classify every mesh vertex geometrically and project it onto its class."""
    policy_ = _policy(policy)
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    if not isinstance(projection, AbstractBRepProjection):
        raise TypeError("projection must be AbstractBRepProjection.")
    if mesh.ambient_dimension != projection.ambient_dimension:
        raise ValueError("Mesh and projection ambient dimensions differ.")
    classes = _geometric_vertex_classes(
        mesh, projection, policy_.classification_tolerance, None
    )
    return _brep_association(
        projection,
        mesh,
        0,
        classes,
        np.asarray(mesh.coordinates, dtype=np.float64),
        policy_,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=None,
    )


# -- propagation through lineage --------------------------------------------------


def _child_sources(
    lineage: MeshLineage,
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    created: np.ndarray,
    vertex_stencil: VertexInterpolationStencil | None,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest-dimensional source entity containing each created vertex stencil."""
    from ._lineage import EntityLineageKind, VertexInterpolationStencil
    from ._topology_edit import entity_keys, key_rows

    top = source_mesh.topological_dimension
    dims = np.full((created.size,), -1, dtype=np.int64)
    rows = np.full((created.size,), -1, dtype=np.int64)
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    refined = (
        int(EntityLineageKind.REFINED_FROM),
        int(EntityLineageKind.SPLIT_FROM),
    )
    for dimension in range(1, top):
        record = lineage.entity_lineage(dimension)
        kinds = np.asarray(record.relation_kinds)
        mask = np.isin(kinds, refined)
        source_rows = _entity_rows(
            source_mesh, dimension, np.asarray(record.source_global_ids)[mask]
        )
        target_rows = _entity_rows(
            target_mesh, dimension, np.asarray(record.target_global_ids)[mask]
        )
        vertices = _incidence_pairs(target_mesh, 0, dimension)
        entity_vertices = _compose(
            np.stack((np.arange(target_rows.size), target_rows), axis=1),
            vertices[:, ::-1],
        )
        vertex_ids = target_ids[entity_vertices[:, 1]]
        position = np.searchsorted(created, vertex_ids)
        position = np.minimum(position, created.size - 1)
        hit = (created.size > 0) & (created[position] == vertex_ids)
        open_ = dims[position[hit]] < 0
        chosen = position[hit][open_]
        dims[chosen] = dimension
        rows[chosen] = source_rows[entity_vertices[hit, 0][open_]]

    open_rows = np.flatnonzero(dims < 0)
    if not open_rows.size or vertex_stencil is None:
        return dims, rows
    if not isinstance(vertex_stencil, VertexInterpolationStencil):
        raise TypeError("vertex_stencil must be VertexInterpolationStencil or None.")
    if (
        vertex_stencil.source_entity_set_id != source_mesh.entity_set(0).entity_set_id
        or vertex_stencil.target_entity_set_id != target_mesh.entity_set(0).entity_set_id
    ):
        raise ValueError(
            "Vertex interpolation stencil endpoints do not match association meshes."
        )
    stencil_ids = np.asarray(vertex_stencil.target_global_ids, dtype=np.int64)
    stencil_order = np.argsort(stencil_ids, kind="stable")
    stencil_sorted = stencil_ids[stencil_order]
    positions = np.searchsorted(stencil_sorted, created[open_rows])
    if np.any(positions >= stencil_sorted.size) or not np.array_equal(
        stencil_sorted[positions], created[open_rows]
    ):
        raise ValueError(
            "Vertex interpolation stencil omits created association targets."
        )
    stencil_rows = stencil_order[positions]
    stencil_sources = np.asarray(vertex_stencil.source_global_ids, dtype=np.int64)
    stencil_weights = np.asarray(vertex_stencil.weights, dtype=np.float64)
    stencil_valid = np.asarray(vertex_stencil.valid, dtype=np.bool_)
    source_cell_ids = np.concatenate(
        [np.asarray(block.global_ids, dtype=np.int64) for block in source_mesh.blocks]
    )
    source_cell_vertices = np.concatenate(
        [np.asarray(block.vertices, dtype=np.int64) for block in source_mesh.blocks]
    )
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    cell_keys = np.sort(source_vertex_ids[source_cell_vertices], axis=1)
    # Supports are looked up once per dimension: each lookup sorts its whole key table.
    supports: dict[int, list[tuple[int, np.ndarray]]] = {}
    for created_row, stencil_row in zip(open_rows, stencil_rows, strict=True):
        support = np.sort(
            np.unique(
                stencil_sources[
                    stencil_row,
                    stencil_valid[stencil_row] & (stencil_weights[stencil_row] != 0.0),
                ]
            )
        )
        if support.size < 2 or support.size > top + 1:
            continue
        supports.setdefault(int(support.size - 1), []).append((int(created_row), support))
    for dimension, entries in sorted(supports.items()):
        queries = np.stack([support for _, support in entries])
        if dimension == top:
            cell_rows = key_rows(cell_keys, queries)
            entity_rows = np.full(cell_rows.shape, -1, dtype=np.int64)
            found = cell_rows >= 0
            if np.any(found):
                entity_rows[found] = _entity_rows(
                    source_mesh, top, source_cell_ids[cell_rows[found]]
                )
        else:
            entity_rows = key_rows(entity_keys(source_mesh, dimension), queries)
        for (created_row, _), entity_row in zip(
            entries, entity_rows.tolist(), strict=True
        ):
            if entity_row >= 0:
                dims[created_row] = dimension
                rows[created_row] = entity_row
    return dims, rows


def propagate_association(
    association: GeometryAssociation,
    lineage: MeshLineage,
    source_mesh: CellMesh,
    target_mesh: CellMesh,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
    vertex_stencil: VertexInterpolationStencil | None,
) -> GeometryAssociation:
    """Carry B-Rep classes through lineage and exact interpolation support.

    Preserved vertices keep their class, relocated vertices are re-projected,
    and created vertices inherit a source entity class from their explicit
    interpolation stencil or lower-dimensional entity lineage. The
    ``VertexInterpolationStencil`` is the source of interpolation support;
    ``SPLIT_FROM`` vertex relations are not used as a substitute.
    """
    from ._lineage import EntityLineageKind, MeshLineage, VertexInterpolationStencil

    if vertex_stencil is not None and not isinstance(
        vertex_stencil, VertexInterpolationStencil
    ):
        raise TypeError("vertex_stencil must be VertexInterpolationStencil or None.")

    policy_ = _policy(policy)
    _require_brep(association, projection, source_mesh, 0)
    if not isinstance(lineage, MeshLineage):
        raise TypeError("lineage must be MeshLineage.")
    if not isinstance(target_mesh, CellMesh):
        raise TypeError("target_mesh must be CellMesh.")
    if (
        lineage.source_topology_id != source_mesh.topology_id
        or lineage.target_topology_id != target_mesh.topology_id
    ):
        raise ValueError("The lineage does not relate these source and target meshes.")
    source_classes = _vertex_classes(source_mesh, association)
    target_ids = np.asarray(target_mesh.vertex_global_ids, dtype=np.int64)
    record = lineage.entity_lineage(0)
    kinds = np.asarray(record.relation_kinds)
    sources = np.asarray(record.source_global_ids, dtype=np.int64)
    targets = np.asarray(record.target_global_ids, dtype=np.int64)
    _check_collapses(
        projection,
        source_classes,
        _entity_rows(source_mesh, 0, sources[kinds == EntityLineageKind.COLLAPSED_INTO]),
        _entity_rows(source_mesh, 0, targets[kinds == EntityLineageKind.COLLAPSED_INTO]),
        targets[kinds == EntityLineageKind.COLLAPSED_INTO],
    )
    count = target_ids.size
    dims = np.full((count,), -1, dtype=np.int64)
    indices = np.full((count,), -1, dtype=np.int64)
    status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
    paths: list[tuple[str, ...]] = [()] * count
    parent_dims = np.full((count,), -1, dtype=np.int8)
    parent_ids = np.full((count,), -1, dtype=np.int64)
    kept = np.flatnonzero(_entity_rows(source_mesh, 0, target_ids) >= 0)
    kept_source = _entity_rows(source_mesh, 0, target_ids[kept])
    dims[kept] = source_classes.dimensions[kept_source]
    indices[kept] = source_classes.indices[kept_source]
    status[kept] = source_classes.status[kept_source]
    for row, parent in zip(kept.tolist(), kept_source.tolist(), strict=True):
        paths[row] = source_classes.occurrence_paths[parent]
    parent_dims[kept] = 0
    parent_ids[kept] = target_ids[kept]
    created_rows = np.flatnonzero(_entity_rows(source_mesh, 0, target_ids) < 0)
    created = target_ids[created_rows]
    order = np.argsort(created, kind="stable")
    created_rows, created = created_rows[order], created[order]
    if created.size:
        entity_dims, entity_rows = _child_sources(
            lineage, source_mesh, target_mesh, created, vertex_stencil
        )
        found = entity_dims >= 0
        if np.any(found):
            source_levels = _classify_mesh(source_mesh, source_classes, projection)
            for level in np.unique(entity_dims[found]).tolist():
                selected = np.flatnonzero(entity_dims == level)
                level_classes = source_levels[level]
                rows = entity_rows[selected]
                target_rows = created_rows[selected]
                dims[target_rows] = level_classes.dimensions[rows]
                indices[target_rows] = level_classes.indices[rows]
                status[target_rows] = level_classes.status[rows]
                for target_row, source_row in zip(
                    target_rows.tolist(), rows.tolist(), strict=True
                ):
                    paths[target_row] = level_classes.occurrence_paths[source_row]
                parent_dims[target_rows] = level
                parent_ids[target_rows] = np.asarray(
                    source_mesh.entity_set(level).entity_ids, dtype=np.int64
                )[rows]
        orphans = created_rows[~found]
        if orphans.size:
            raise AssociationPropagationError(
                "Created B-Rep association vertices require exact source interpolation support or entity lineage.",
                target_ids[orphans],
            )
    return _brep_association(
        projection,
        target_mesh,
        0,
        _Classes(dims, indices, status, tuple(paths)),
        np.asarray(target_mesh.coordinates, dtype=np.float64),
        policy_,
        provenance=GeometryAssociationProvenance.LINEAGE,
        parent_association_id=association.association_id,
        parent_dimensions=parent_dims,
        parent_ids=parent_ids,
    )


def _check_collapses(
    projection: AbstractBRepProjection,
    classes: _Classes,
    removed: np.ndarray,
    kept: np.ndarray,
    kept_ids: np.ndarray,
    /,
) -> None:
    """A collapse keeps a vertex whose class lies in the removed vertex's closure."""
    if removed.size == 0:
        return
    if np.any(removed < 0) or np.any(kept < 0):
        raise ValueError("Collapse relations must connect source vertices.")
    removed_dims = classes.dimensions[removed]
    kept_dims = classes.dimensions[kept]
    known = (removed_dims >= 0) & (kept_dims >= 0)
    known &= classes.resolved[removed] & classes.resolved[kept]
    known &= np.asarray(
        [
            classes.occurrence_paths[first] == classes.occurrence_paths[second]
            for first, second in zip(removed.tolist(), kept.tolist(), strict=True)
        ],
        dtype=np.bool_,
    )
    legal = np.zeros(removed.shape, dtype=np.bool_)
    legal[known] = projection.contains(
        removed_dims[known],
        classes.indices[removed][known],
        kept_dims[known],
        classes.indices[kept][known],
    )
    if not np.all(legal):
        raise AssociationPropagationError(
            "A collapse removed a vertex whose B-Rep class does not bound the kept "
            "vertex's class.",
            kept_ids[~legal],
        )


# -- re-derivation after unknown lineage --------------------------------------------


def _facet_signatures(
    result: CellMeshingResult, rows: int, /
) -> tuple[np.ndarray, tuple]:
    """Facet organization signature (names of facet patches, zones, and labels)."""
    mesh = result.mesh
    dimension = mesh.topological_dimension - 1
    items = sorted(
        (value.name, value.scope)
        for value in (*result.patches, *result.zones, *result.labels)
        if value.scope.entity_dimension == dimension
    )
    columns = [
        np.asarray(resolve_mesh_scope(mesh, scope).mask, dtype=np.bool_)
        for _, scope in items
    ]
    matrix = np.stack(columns, axis=1) if columns else np.zeros((rows, 0), dtype=np.bool_)
    return matrix, tuple(name for name, _ in items)


def _transfer_groups(
    source: CellMeshingResult,
    target: CellMeshingResult,
    source_classes: tuple[_Classes, ...],
    projection: AbstractBRepProjection,
    /,
) -> _ClassificationGroups:
    """Admissible B-Rep entities per target boundary vertex from facet classes.

    Source boundary facets carrying one organization signature admit the B-Rep
    classes of those facets and their closures; target boundary vertices admit
    the union over their incident target facets with the same signatures.
    """
    top = source.mesh.topological_dimension
    facet_count = source.mesh.entity_set(top - 1).count
    source_matrix, source_names = _facet_signatures(source, facet_count)
    target_matrix, target_names = _facet_signatures(
        target, target.mesh.entity_set(top - 1).count
    )
    names = tuple(sorted(set(source_names) | set(target_names)))

    def aligned(matrix: np.ndarray, own: tuple[str, ...], /) -> np.ndarray:
        output = np.zeros((matrix.shape[0], len(names)), dtype=np.bool_)
        output[:, [names.index(name) for name in own]] = matrix
        return output

    boundary = _boundary_mask(source.mesh, top - 1)
    _, signature = np.unique(
        np.concatenate(
            (
                aligned(source_matrix, source_names),
                aligned(target_matrix, target_names),
            )
        ),
        axis=0,
        return_inverse=True,
    )
    signature = signature.reshape(-1)
    source_signature = signature[:facet_count]
    target_signature = signature[facet_count:]
    facet_classes = source_classes[top - 1]
    rows = np.flatnonzero(boundary & facet_classes.resolved)
    allowed, qualified_codes, qualified_paths = _members_closure(
        projection,
        source_signature[rows],
        facet_classes.dimensions[rows],
        facet_classes.indices[rows],
        tuple(facet_classes.occurrence_paths[row] for row in rows.tolist()),
    )
    target_boundary = _boundary_mask(target.mesh, top - 1)
    vertex_facets = _incidence_pairs(target.mesh, 0, top - 1)
    vertex_facets = vertex_facets[target_boundary[vertex_facets[:, 1]]]
    vertex_signatures = np.unique(
        np.stack((vertex_facets[:, 0], target_signature[vertex_facets[:, 1]]), axis=1),
        axis=0,
    ).reshape(-1, 2)
    vertex_count = target.mesh.coordinates.shape[0]
    width = max(1, int(np.max(np.bincount(vertex_signatures[:, 0], minlength=1))))
    # One group per distinct set of incident facet signatures (padded rows).
    padded = np.full((vertex_count, width), -1, dtype=np.int64)
    slot = np.arange(vertex_signatures.shape[0]) - np.searchsorted(
        vertex_signatures[:, 0], vertex_signatures[:, 0]
    )
    padded[vertex_signatures[:, 0], slot] = vertex_signatures[:, 1]
    combinations, groups = np.unique(padded, axis=0, return_inverse=True)
    group_signature = np.argwhere(combinations >= 0)
    group_signature[:, 1] = combinations[group_signature[:, 0], group_signature[:, 1]]
    group_codes = _compose(group_signature, allowed)
    dims, indices = projection.code_entities(qualified_codes[group_codes[:, 1]])
    return _ClassificationGroups(
        groups.reshape(-1),
        group_codes[:, 0],
        dims,
        indices,
        tuple(qualified_paths[row] for row in group_codes[:, 1].tolist()),
    )


def _members_closure(
    projection: AbstractBRepProjection,
    signatures: np.ndarray,
    dimensions: np.ndarray,
    indices: np.ndarray,
    occurrence_paths: tuple[tuple[str, ...], ...],
    /,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[str, ...], ...]]:
    """Retain qualified source identity while grouping definition closures."""
    owner, member_dims, member_indices = projection.members(dimensions, indices)
    codes = projection.entity_codes(member_dims, member_indices)
    records = tuple(
        sorted(
            {
                (int(signatures[row]), int(code), occurrence_paths[row])
                for row, code in zip(owner.tolist(), codes.tolist(), strict=True)
            }
        )
    )
    keys = tuple(sorted({(code, path) for _, code, path in records}))
    key_rows = {key: row for row, key in enumerate(keys)}
    allowed = np.asarray(
        [(signature, key_rows[(code, path)]) for signature, code, path in records],
        dtype=np.int64,
    ).reshape(-1, 2)
    return (
        allowed,
        np.asarray([code for code, _ in keys], dtype=np.int64),
        tuple(path for _, path in keys),
    )


def rederive_association(
    association: GeometryAssociation,
    source: CellMeshingResult,
    target: CellMeshingResult,
    projection: AbstractBRepProjection,
    /,
    *,
    policy: AssociationPropagationPolicy,
) -> GeometryAssociation:
    """Re-derive a vertex association on a remeshed target with unknown lineage.

    Classification transfer: source boundary facets are classified from the
    source vertex ``association``; target boundary vertices may classify only on
    the B-Rep classes (with closures) of source facets sharing the organization
    signature (facet patches, zones, labels, which remeshing providers carry as
    references/class IDs) of their incident target facets. Each vertex then takes
    the lowest-dimensional admissible entity within the classification tolerance
    and is projected; residuals are the evidence.
    """
    from ._result import CellMeshingResult

    policy_ = _policy(policy)
    if not isinstance(source, CellMeshingResult) or not isinstance(
        target, CellMeshingResult
    ):
        raise TypeError("source and target must be CellMeshingResult values.")
    _require_brep(association, projection, source.mesh, 0)
    if target.mesh.topological_dimension != source.mesh.topological_dimension:
        raise ValueError("Re-derivation requires one topological dimension.")
    source_levels = _classify_mesh(
        source.mesh, _vertex_classes(source.mesh, association), projection
    )
    groups = _transfer_groups(source, target, source_levels, projection)
    classes = _geometric_vertex_classes(
        target.mesh, projection, policy_.classification_tolerance, groups
    )
    return _brep_association(
        projection,
        target.mesh,
        0,
        classes,
        np.asarray(target.mesh.coordinates, dtype=np.float64),
        policy_,
        provenance=GeometryAssociationProvenance.CLASSIFICATION,
        parent_association_id=association.association_id,
    )


# -- result-level transfer ---------------------------------------------------------


def _target_dimension(mesh: CellMesh, association: GeometryAssociation, /) -> int:
    for dimension in range(mesh.topological_dimension + 1):
        if mesh.entity_set(dimension).entity_set_id == association.target_entity_set_id:
            return dimension
    raise ValueError("A B-Rep association does not target the source mesh entities.")


@final
class BRepAssociationTransfer(StrictModule, NonTrainableState):
    """Carries a result's B-Rep associations through topology changes.

    Binds one prepared projection and propagation policy. A source result must
    carry exactly one vertex association of the projection's revision (covering
    every vertex) and may carry B-Rep associations of higher mesh dimensions; the
    target receives the propagated or re-derived vertex association and the
    associations of the same higher dimensions derived from it.
    """

    projection: AbstractBRepProjection
    policy: AssociationPropagationPolicy
    transfer_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        projection: AbstractBRepProjection,
        /,
        *,
        policy: AssociationPropagationPolicy | None = None,
    ) -> None:
        policy_ = AssociationPropagationPolicy() if policy is None else _policy(policy)
        self.projection = projection
        self.policy = policy_
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "brep-association-transfer",
                "projection": projection.projection_id,
                "policy": policy_.policy_id,
            }
        )

    @property
    def source_namespaces(self) -> tuple[tuple[str, str], ...]:
        return ((self.projection.source_id, self.projection.source_revision),)

    def validate_source(self, source: CellMeshingResult, /) -> None:
        self.source_associations(source)

    def source_associations(
        self, source: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, tuple[int, ...]]:
        """The source vertex association and the higher dimensions to rebuild."""
        from ._result import CellMeshingResult

        if not isinstance(source, CellMeshingResult):
            raise TypeError("source must be CellMeshingResult.")
        dimensions = []
        curve = None
        vertex = None
        for association in source.associations:
            if (
                association.source_id,
                association.source_revision,
            ) != self.source_namespaces[0]:
                continue
            if association.association_kind is GeometryAssociationKind.CURVE:
                if curve is not None:
                    raise ValueError("A B-Rep source carries one CAD curve association.")
                curve = association
                continue
            if association.association_kind is not GeometryAssociationKind.BREP:
                raise ValueError("A B-Rep source namespace cannot mix association kinds.")
            dimension = _target_dimension(source.mesh, association)
            if dimension == 0:
                if vertex is not None:
                    raise ValueError("A source carries one vertex association.")
                vertex = association
            else:
                dimensions.append(dimension)
        if vertex is None:
            raise ValueError("B-Rep association transfer requires a vertex association.")
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("A source carries one association per mesh dimension.")
        _vertex_classes(source.mesh, vertex)
        if curve is not None:
            self._validate_cad_curve(source, curve)
        elif source.certification is not None:
            witness = source.certification.inputs.fidelity_witness
            if (
                isinstance(witness, SourceBoundaryChartWitness)
                and (witness.source_id, witness.source_revision) == self.source_namespaces[0]
                and witness.cad_plc is not None
                and witness.cad_plc.cad_segment_edge_ids is not None
                and np.asarray(witness.cad_plc.cad_segment_edge_ids).size
            ):
                raise ValueError("CAD source curve association is missing.")
        return vertex, tuple(sorted(dimensions))

    def _validate_cad_curve(
        self, source: CellMeshingResult, curve: GeometryAssociation, /
    ) -> None:
        """Require the accepted CAD chart and its original curve-edge authority."""
        certification = source.certification
        witness = None if certification is None else certification.inputs.fidelity_witness
        charts = source.surface_source
        if (
            not isinstance(witness, SourceBoundaryChartWitness)
            or witness.cad_curve_association is None
            or witness.cad_curve_association.association_id != curve.association_id
            or witness.mesh_id != source.mesh.mesh_id
            or witness.topology_id != source.mesh.topology_id
            or (witness.source_id, witness.source_revision) != self.source_namespaces[0]
            or charts is None
        ):
            raise ValueError("CAD curve association lacks its current source chart witness.")
        charts.require_current()
        roots = tuple(
            item for item in charts.associations
            if item.association_kind is GeometryAssociationKind.CURVE
            and item.association_id == curve.parent_association_id
        )
        if len(roots) != 1 or (
            roots[0].source_id,
            roots[0].source_revision,
        ) != self.source_namespaces[0]:
            raise ValueError("CAD curve association lost its original source-edge authority.")
        curve.validate_target(source.mesh.entity_set(1))

        plc = witness.cad_plc
        if (
            plc is None
            or plc.cad_source_charts is None
            or plc.cad_source_charts.source_chart_id != charts.source_chart_id
            or plc.cad_fidelity_tolerance is None
            or curve.provenance is not GeometryAssociationProvenance.LINEAGE
        ):
            raise ValueError("CAD curve association lacks its original PLC lineage.")
        from ..geometry._approximate_source_coverage import _cad_target_root_partition

        patch_rows = [
            charts.domain.resolve_entity_id(2, identifier, occurrence_path=occurrence)
            for identifier, occurrence in zip(
                witness.source_entity_ids, witness.occurrence_paths, strict=True
            )
        ]
        _cad_target_root_partition(
            charts.domain, source.mesh, witness, patch_rows, plc.cad_fidelity_tolerance
        )
        edge_tables = tuple(
            item for item in source.associations
            if item.association_kind is GeometryAssociationKind.BREP
            and (item.source_id, item.source_revision) == self.source_namespaces[0]
            and item.target_entity_set_id == source.mesh.entity_set(1).entity_set_id
        )
        if len(edge_tables) != 1:
            raise ValueError("CAD curve association requires one B-Rep edge table.")
        edge_table = edge_tables[0]
        edge_table.validate_target(source.mesh.entity_set(1))
        edge_rows = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(edge_table.target_global_ids))
        }
        for row, identifier in enumerate(np.asarray(curve.target_global_ids)):
            edge = edge_rows.get(int(identifier))
            if (
                edge is None
                or int(edge_table.source_dimensions[edge]) != 1
                or not bool(edge_table.resolved[edge])
                or bool(edge_table.ambiguous[edge])
                or edge_table.source_entity_ids[edge] != curve.source_entity_ids[row]
                or int(edge_table.source_indices[edge]) != int(curve.source_indices[row])
                or edge_table.source_occurrence_paths[edge] != curve.source_occurrence_paths[row]
                or (
                    int(edge_table.orientations[edge]) != 0
                    and int(edge_table.orientations[edge]) != int(curve.orientations[row])
                )
            ):
                raise ValueError("CAD curve association changes its target-edge authority.")

    def classes(
        self, source: CellMeshingResult, /
    ) -> tuple[AssociationEntityClasses, ...]:
        """B-Rep classes of every source mesh dimension (adaptation constraints)."""
        vertex, _ = self.source_associations(source)
        levels = _mesh_entity_classes(source.mesh, vertex, self.projection)
        output = []
        for level in levels:
            valid = level.dimensions >= 0
            codes = np.full(level.dimensions.shape, -1, dtype=np.int64)
            if np.any(valid):
                codes[valid] = self.projection.entity_codes(
                    level.dimensions[valid], level.indices[valid]
                )
            output.append(
                AssociationEntityClasses(
                    level.dimensions,
                    level.indices,
                    level.resolved,
                    codes,
                )
            )
        return tuple(output)

    def _derived(
        self,
        vertex: GeometryAssociation,
        dimensions: tuple[int, ...],
        mesh: CellMesh,
        /,
    ) -> tuple[GeometryAssociation, ...]:
        return (vertex,) + tuple(
            associate_mesh_entities(
                mesh, vertex, self.projection, dimension, policy=self.policy
            )
            for dimension in dimensions
        )

    def propagate(
        self,
        source: CellMeshingResult,
        lineage: MeshLineage,
        target: CellMesh,
        /,
        *,
        vertex_stencil: VertexInterpolationStencil | None,
        geometry_transition: CellGeometryTransition | None,
        target_fidelity_witness: SourceBoundaryChartWitness
        | SourceBoundaryFaceImageWitness
        | None,
        target_scoped_fidelity: tuple[ScopedFidelityInput, ...],
        validated_fidelity: tuple[ValidatedSourceFidelityWitness, ...],
    ) -> tuple[GeometryAssociation, ...]:
        """Propagate B-Rep associations with the accepted vertex stencil."""
        vertex, dimensions = self.source_associations(source)
        propagated = propagate_association(
            vertex,
            lineage,
            source.mesh,
            target,
            self.projection,
            policy=self.policy,
            vertex_stencil=vertex_stencil,
        )
        output = self._derived(propagated, dimensions, target)
        curves = tuple(
            item for item in source.associations
            if (item.source_id, item.source_revision) == self.source_namespaces[0]
            and item.association_kind is GeometryAssociationKind.CURVE
        )
        if not curves:
            return output
        from ._surface_association_transfer import _curve_edges_with_root_ancestry

        if (
            not isinstance(target_fidelity_witness, SourceBoundaryChartWitness)
            or target_fidelity_witness.mesh_id != target.mesh_id
            or target_fidelity_witness.topology_id != target.topology_id
            or (target_fidelity_witness.source_id, target_fidelity_witness.source_revision)
            != self.source_namespaces[0]
            or target_fidelity_witness.cad_curve_association is None
        ):
            raise ValueError("CAD curve transfer requires the current target chart witness.")
        if target.topology_id == source.mesh.topology_id:
            curve = curves[0]
        else:
            if vertex_stencil is None:
                raise ValueError("CAD curve transfer requires exact vertex ancestry.")
            curve = _curve_edges_with_root_ancestry(
                source, lineage, target, vertex_stencil, curves[0]
            )
        if curve.association_id != target_fidelity_witness.cad_curve_association.association_id:
            raise ValueError("CAD curve transfer differs from the target chart ancestry.")
        curve.validate_target(target.entity_set(1))
        return (*output, curve)

    def rederive(
        self, source: CellMeshingResult, target: CellMeshingResult, /
    ) -> tuple[GeometryAssociation, ...]:
        """Target associations of a remeshed result with unknown lineage."""
        vertex, dimensions = self.source_associations(source)
        if any(
            item.association_kind is GeometryAssociationKind.CURVE
            and (item.source_id, item.source_revision) == self.source_namespaces[0]
            for item in source.associations
        ):
            raise ValueError("CAD curve associations require exact topology lineage.")
        derived = rederive_association(
            vertex, source, target, self.projection, policy=self.policy
        )
        return self._derived(derived, dimensions, target.mesh)


__all__ = [
    "AssociationEntityClasses",
    "AssociationPropagationError",
    "AssociationPropagationPolicy",
    "AssociationTransfer",
    "BRepAssociationTransfer",
    "GeometryAssociation",
    "GeometryAssociationArchiveRecipe",
    "GeometryAssociationKind",
    "GeometryAssociationProvenance",
    "GeometrySourceEntityRole",
    "inherit_association_rows",
    "associate_mesh_entities",
    "associate_mesh_vertices",
    "propagate_association",
    "rederive_association",
]
