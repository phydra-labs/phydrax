#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound native overset hole cutting, donor search and field transfer.

The registration owns geometric blanking and deterministic cell-location evidence.
Field queries preserve the selected FE/FV reconstruction and its exact transpose;
physical overlap remapping remains a separate conservative route.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from fractions import Fraction
from typing import Any, assert_never, final, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._bvh import bvh_overlap_pair_blocks, prepare_bvh
from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    orient2d,
    orient3d,
    PredicateMode,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._cell_complex import (
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ..discretization._cell_geometry_validity import cell_geometry_id
from ..discretization._cell_locator import (
    AbstractCellLocator,
    CellInverseRefusal,
    CellLocationRefusalError,
    CellLocationStatus,
)
from ..discretization._cell_mesh import SimplicialConnectivity
from ..discretization._field_query import PreparedFieldQuery
from ..discretization._hexahedral import HexahedralConnectivity
from ..discretization._mapped_locator import (
    prepare_owner_local_mapped_cell_locator,
    PreparedMappedCellLocator,
    PreparedMappedCellMap,
)
from ..discretization._polyhedral_locator import (
    PreparedPolyhedralCellLocator,
    PreparedPolyhedralCellMap,
)
from ..discretization._view_support import OwnerLocalCellGeometry, OwnerLocalFieldSupport
from ..discretization._views import (
    PreparedFieldReconstruction,
    transpose_duality_evidence,
)
from ..discretization.fem import (
    discontinuous_element,
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    prepare_finite_element_field_reconstruction,
)
from ..discretization.fem._reference import FiniteElementSpec
from ..geometry import require_polyhedral_carrier_geometry
from ..geometry._common_root_volume_overlap import (
    apply_physical_overlap_extensive,
    apply_physical_overlap_extensive_with_evidence,
    apply_physical_overlap_intensive,
    CommonRootVolumeWitness,
    CurvedOverlapIntegrationEvidence,
    CurvedOverlapIntegrationPolicy,
    PhysicalOverlapActionEvidence,
    prepare_common_root_volume_refinement,
    transpose_physical_overlap_extensive,
    transpose_physical_overlap_intensive,
)
from ..geometry._overset_partition import OversetPartition, partition_overset_triangles
from ..geometry._planar_primitives import cross
from ..geometry._supermesh import (
    CommonRefinementCoverage,
    CommonRefinementEvidence,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    prepare_common_refinement,
    PreparedCommonRefinement,
)
from ..lifecycle import (
    commit_composition_rebind,
    Composition,
    CompositionEntry,
    CompositionRebind,
    CompositionRebindReceipt,
    CompositionTransport,
)
from ..typing import checked, ConvertibleToArray, parse
from ._assembly import MeshAssembly, MeshPart, MeshPartManifest
from ._coupling import (
    CouplingSearchEvidence,
    CouplingSearchStatus,
    MeshCoupling,
    MeshCouplingKind,
    OversetCoupling,
    OversetValueAction,
)
from ._result import CellMeshingResult
from ._scope import MeshingScope


if TYPE_CHECKING:
    from ._motion import MeshMotionAdvance
    from ._overset_exchange import PreparedOwnerLocalOversetExchange

type OversetDonorPriority = Literal["finest-donor", "part-priority"]
_OWNER = "phydrax.meshing.overset"
_CONNECTIVITY_ENTRY = "overset:connectivity"


class OversetVertexStatus(IntEnum):
    ACTIVE = 0
    RECEPTOR = 1
    HOLE = 2


class OversetCellStatus(IntEnum):
    ACTIVE = 0
    FRINGE = 1
    HOLE = 2


def _cell_result(part: MeshPart) -> CellMeshingResult:
    if not isinstance(part.carrier, CellMeshingResult):
        raise TypeError(
            f"Overset part {part.name!r} requires an explicit cell-mesh carrier; "
            "point, tensor and spline carriers are retained as-is and are not tessellated."
        )
    return part.carrier


def _scope_points(part: MeshPart, scope: MeshingScope | None) -> np.ndarray:
    if scope is None:
        return np.empty((0,), dtype=np.int64)
    part.require_scope(scope)
    if scope.entity_dimension != 0:
        raise ValueError("Overset point scopes must have entity_dimension zero.")
    return np.asarray(scope.entity_ids, dtype=np.int64)


def _pack_cell_vertices(
    rows: tuple[tuple[int, ...], ...], /
) -> tuple[np.ndarray, np.ndarray]:
    if not rows or any(not row for row in rows):
        raise ValueError("Overset parts require nonempty cell vertex rows.")
    width = max(map(len, rows))
    vertices = np.zeros((len(rows), width), dtype=np.int64)
    valid = np.zeros(vertices.shape, dtype=np.bool_)
    for cell, row in enumerate(rows):
        vertices[cell, : len(row)] = row
        valid[cell, : len(row)] = True
    return vertices, valid


def _polyhedral_cell_measures(locator: PreparedPolyhedralCellLocator, /) -> np.ndarray:
    connectivity = locator.cell_map.mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("Polyhedral location requires canonical polyhedral connectivity.")
    coordinates = np.asarray(locator.coordinates, dtype=np.float64)
    cell_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.int8)
    volumes = np.empty((locator.cell_map.cell_count,), dtype=np.float64)
    for cell, vertices in enumerate(locator.cell_map.cell_vertices):
        origin = tuple(
            sum(
                (Fraction(float(coordinates[vertex, axis])) for vertex in vertices),
                Fraction(0),
            )
            / len(vertices)
            for axis in range(3)
        )
        volume = Fraction(0)
        faces = cell_faces[cell_offsets[cell] : cell_offsets[cell + 1]]
        signs = cell_signs[cell_offsets[cell] : cell_offsets[cell + 1]]
        for face, sign in zip(faces, signs, strict=True):
            for triangle in locator.cell_map.face_triangles[int(face)]:
                points = tuple(
                    tuple(
                        Fraction(float(coordinates[vertex, axis])) - origin[axis]
                        for axis in range(3)
                    )
                    for vertex in triangle
                )
                first, second, third = points
                determinant = (
                    first[0] * (second[1] * third[2] - second[2] * third[1])
                    - first[1] * (second[0] * third[2] - second[2] * third[0])
                    + first[2] * (second[0] * third[1] - second[1] * third[0])
                )
                volume += int(sign) * determinant / 6
        if volume <= 0:
            raise ValueError("Overset polyhedral cells require positive oriented volume.")
        volumes[cell] = float(volume)
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError(
            "Overset polyhedral cells require positive finite physical measure."
        )
    return volumes


def _facet_rows(part: MeshPart, scope: MeshingScope) -> np.ndarray:
    part.require_scope(scope)
    mesh = _cell_result(part).mesh
    connectivity = mesh.connectivity
    dimension = mesh.topological_dimension
    if scope.entity_dimension != dimension - 1:
        raise ValueError("Overset wall/boundary scopes must identify mesh facets.")
    if isinstance(connectivity, PolyhedralConnectivity):
        identifiers = np.asarray(connectivity.face_global_ids, dtype=np.int64)
        boundary = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
        order = np.argsort(identifiers, kind="stable")
        rows = order[np.searchsorted(identifiers[order], np.asarray(scope.entity_ids))]
        if np.any(~boundary[rows]):
            raise ValueError(
                "Overset wall and receptor scopes must identify exterior facets."
            )
        return rows
    if isinstance(connectivity, SimplicialConnectivity):
        entities = np.asarray(connectivity.entities[dimension - 1], dtype=np.int64)
        boundary = np.asarray(connectivity.boundary_facets, dtype=np.bool_)
    elif isinstance(connectivity, PolygonalConnectivity):
        if dimension != 2:
            raise ValueError(
                "Polygonal connectivity only supplies two-dimensional facets."
            )
        entities = np.asarray(connectivity.edges, dtype=np.int64)
        boundary = np.asarray(connectivity.boundary_edges, dtype=np.bool_)
    elif isinstance(connectivity, TetrahedralConnectivity):
        if dimension != 3:
            raise ValueError(
                "Tetrahedral connectivity only supplies three-dimensional facets."
            )
        entities = np.asarray(connectivity.faces, dtype=np.int64)
        boundary = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    elif isinstance(connectivity, HexahedralConnectivity):
        if dimension != 3:
            raise ValueError(
                "Hexahedral connectivity only supplies three-dimensional facets."
            )
        entities = np.asarray(connectivity.faces, dtype=np.int64)
        boundary = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    elif isinstance(connectivity, IntervalConnectivity):
        if dimension != 1:
            raise ValueError(
                "Interval connectivity only supplies one-dimensional facets."
            )
        entities = np.arange(connectivity.vertex_count, dtype=np.int64)[:, None]
        boundary = np.asarray(connectivity.boundary_vertices, dtype=np.bool_)
    else:
        raise TypeError(
            "Overset facet scopes require supported canonical cell incidence."
        )
    identifiers = np.asarray(mesh.entity_set(dimension - 1).entity_ids, dtype=np.int64)
    if (
        entities.ndim != 2
        or boundary.shape != (entities.shape[0],)
        or identifiers.shape != boundary.shape
    ):
        raise ValueError("Overset facet connectivity lacks complete canonical incidence.")
    order = np.argsort(identifiers, kind="stable")
    rows = order[np.searchsorted(identifiers[order], np.asarray(scope.entity_ids))]
    if np.any(~boundary[rows]):
        raise ValueError(
            "Overset wall and receptor scopes must identify exterior facets."
        )
    return entities[rows]


def _facet_vertex_ids(part: MeshPart, scope: MeshingScope, /) -> np.ndarray:
    mesh = _cell_result(part).mesh
    rows = _facet_rows(part, scope)
    if isinstance(mesh.connectivity, PolyhedralConnectivity):
        offsets = np.asarray(mesh.connectivity.face_vertex_offsets, dtype=np.int64)
        values = np.asarray(mesh.connectivity.face_vertex_values, dtype=np.int64)
        return np.unique(
            np.concatenate([values[offsets[row] : offsets[row + 1]] for row in rows])
        )
    return np.unique(rows.reshape((-1,)))


def _facet_triangles(
    part: MeshPart,
    scope: MeshingScope,
    /,
    *,
    polyhedral_map: PreparedPolyhedralCellMap | None = None,
) -> np.ndarray:
    mesh = _cell_result(part).mesh
    rows = _facet_rows(part, scope)
    if mesh.ambient_dimension == 2:
        return rows.reshape((-1, 2))
    if isinstance(mesh.connectivity, PolyhedralConnectivity):
        if polyhedral_map is None:
            raise TypeError(
                "Polyhedral wall triangulation requires its prepared native cell map."
            )
        triangles = [
            triangle
            for face in rows
            for triangle in polyhedral_map.face_triangles[int(face)]
        ]
        return np.asarray(triangles, dtype=np.int64).reshape((-1, 3))
    if any(block.cell_kind != "tetrahedron" for block in mesh.blocks):
        raise ValueError("Non-simplex mapped walls require a source solid query.")
    return rows.reshape((-1, 3))


def _require_closed_wall(facets: np.ndarray, dimension: int, /) -> None:
    if facets.size == 0:
        raise ValueError("An overset wall scope cannot be empty.")
    if dimension == 2:
        ridges = facets.reshape((-1,))
    else:
        ridges = np.sort(
            np.concatenate(
                [np.stack((row, np.roll(row, -1)), axis=1) for row in facets],
                axis=0,
            ),
            axis=1,
        )
    _, counts = np.unique(ridges, axis=0, return_counts=True)
    if np.any(counts != 2):
        raise ValueError("Overset solid wall facets must form a closed two-manifold.")


def _wall_simplices(facets: np.ndarray, dimension: int, /) -> np.ndarray:
    if dimension == 2:
        return facets.reshape((-1, 2))
    triangles = [
        row[[0, index, index + 1]] for row in facets for index in range(1, row.size - 1)
    ]
    return np.asarray(triangles, dtype=np.int64).reshape((-1, 3))


def _image(spec: OversetPartSpec, /) -> tuple[np.ndarray, np.ndarray]:
    dim = spec.image_translation.shape[0] if spec.image_translation is not None else 0
    if spec.image_rotation is None:
        return np.eye(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64)
    return np.asarray(spec.image_rotation), np.asarray(spec.image_translation)


def _relative_image(
    source: OversetPartSpec, target: OversetPartSpec, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    source_rotation, source_translation = _image(source)
    target_rotation, target_translation = _image(target)
    if source_rotation.shape != (dimension, dimension):
        source_rotation, source_translation = np.eye(dimension), np.zeros(dimension)
    if target_rotation.shape != (dimension, dimension):
        target_rotation, target_translation = np.eye(dimension), np.zeros(dimension)
    rotation = target_rotation.T @ source_rotation
    translation = target_rotation.T @ (source_translation - target_translation)
    return rotation, translation


def _to_source(
    points: np.ndarray, source: OversetPartSpec, target: OversetPartSpec, /
) -> np.ndarray:
    rotation, translation = _relative_image(source, target, points.shape[1])
    return (points - translation) @ rotation


def _to_target(
    points: np.ndarray, source: OversetPartSpec, target: OversetPartSpec, /
) -> np.ndarray:
    rotation, translation = _relative_image(source, target, points.shape[1])
    return points @ rotation.T + translation


def _predicate_sign(value: Any) -> np.ndarray:
    if not np.all(np.asarray(value.certain)):
        raise RuntimeError("Exact overset wall predicates returned an uncertain sign.")
    return np.asarray(value.signs, dtype=np.int8)


def _perturbed_side(
    first: np.ndarray, second: np.ndarray, point: np.ndarray
) -> np.ndarray:
    exact = _predicate_sign(orient2d(first, second, point, mode=PredicateMode.EXACT))
    tie = np.where(
        second[:, 1] != first[:, 1],
        np.sign(first[:, 1] - second[:, 1]),
        np.sign(second[:, 0] - first[:, 0]),
    ).astype(np.int8)
    return np.where(exact != 0, exact, tie)


def _segment_crossings(
    points: np.ndarray, segments: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a, b = segments[:, 0], segments[:, 1]
    p = points
    sign = _predicate_sign(orient2d(a, b, p, mode=PredicateMode.EXACT))
    within = np.all((p >= np.minimum(a, b)) & (p <= np.maximum(a, b)), axis=1)
    touching = (sign == 0) & within
    crosses_y = (a[:, 1] > p[:, 1]) != (b[:, 1] > p[:, 1])
    direction = np.sign(b[:, 1] - a[:, 1]).astype(np.int8)
    crossing = crosses_y & (sign * direction > 0)
    return crossing, touching


def _triangle_crossings(
    points: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    projected = points[:, 1:]
    projected_corners = (a[:, 1:], b[:, 1:], c[:, 1:])
    sides = [
        _perturbed_side(first, second, projected)
        for first, second in (
            (projected_corners[0], projected_corners[1]),
            (projected_corners[1], projected_corners[2]),
            (projected_corners[2], projected_corners[0]),
        )
    ]
    inside_projection = (sides[0] == sides[1]) & (sides[1] == sides[2]) & (sides[0] != 0)
    normal = _predicate_sign(
        orient2d(a[:, 1:], b[:, 1:], c[:, 1:], mode=PredicateMode.EXACT)
    )
    volume = _predicate_sign(orient3d(a, b, c, points, mode=PredicateMode.EXACT))
    crossing = inside_projection & (volume != 0) & (volume == -normal)
    touching = np.zeros(points.shape[0], dtype=np.bool_)
    coplanar = np.flatnonzero(volume == 0)
    if coplanar.size:
        p = points[coplanar, 1:]
        area = normal[coplanar]
        contacts = []
        for first, second in (
            (a[coplanar, 1:], b[coplanar, 1:]),
            (b[coplanar, 1:], c[coplanar, 1:]),
            (c[coplanar, 1:], a[coplanar, 1:]),
        ):
            contacts.append(
                _predicate_sign(orient2d(first, second, p, mode=PredicateMode.EXACT))
                * area
            )
        touching[coplanar] = (area != 0) & np.all(np.stack(contacts) >= 0, axis=0)
    return crossing, touching


def _solid_query_identity(query: Any, /) -> str | None:
    if query is None:
        return None
    from ..discretization._view_support import MappedFieldSupport

    if isinstance(query, MappedFieldSupport):
        return query.support_id
    identifier = getattr(query, "query_id", None)
    if isinstance(identifier, str) and identifier:
        return identifier
    from ..geometry._contracts import compiled_geometry_identity, CompiledGeometry

    if isinstance(query, CompiledGeometry):
        return compiled_geometry_identity(query)
    raise TypeError(
        "solid_query must be a source-bound PreparedBRepQuery, CompiledGeometry, or MappedFieldSupport."
    )


def _source_solid_membership(
    query: Any, points: np.ndarray, solid_index: int, vertex_ids: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Classify source points without promoting a display tessellation to authority."""
    from ..discretization._view_support import (
        mapped_mesh_support_query,
        MappedFieldSupport,
    )
    from ..geometry import GeometryCapability

    if isinstance(query, MappedFieldSupport):
        if solid_index != 0:
            raise ValueError(
                "MappedFieldSupport identifies one certified source solid at index zero."
            )
        classified = mapped_mesh_support_query(query, points)
        resolved = np.asarray(classified.successful) & np.asarray(
            classified.location.candidates_complete
        )
        if not np.all(resolved):
            raise CellLocationRefusalError(
                "Mapped source-wall membership is unresolved or ambiguous for vertex IDs "
                f"{tuple(int(value) for value in vertex_ids[~resolved])!r}; refusal "
                "points index those query vertices.",
                classified.location.inverse_refusals,
            )
        return np.asarray(classified.inside), np.asarray(classified.boundary)
    from ..geometry._contracts import CompiledGeometry

    if isinstance(query, CompiledGeometry):
        if solid_index != 0:
            raise ValueError(
                "CompiledGeometry identifies one source solid at index zero."
            )
        sites = np.asarray(points, dtype=np.float64)
        if (
            sites.ndim != 2
            or sites.shape[0] != np.asarray(vertex_ids).size
            or sites.shape[1] != query.ambient_dimension
            or np.any(~np.isfinite(sites))
        ):
            raise ValueError(
                "Mapped source-wall query points must match their finite vertex IDs."
            )
        query.require_valid()
        query.require(GeometryCapability.REGION_QUERY)
        inside = np.asarray(query.contains(jnp.asarray(sites)))
        margin = np.asarray(query.boundary_field(jnp.asarray(sites)), dtype=np.float64)
        bounds = np.asarray(query.bounds, dtype=np.float64)
        if (
            inside.shape != (sites.shape[0],)
            or inside.dtype != np.dtype(np.bool_)
            or margin.shape != (sites.shape[0],)
            or not np.all(np.isfinite(margin))
            or bounds.shape != (2, query.ambient_dimension)
            or not np.all(np.isfinite(bounds))
        ):
            raise RuntimeError(
                "Compiled source solid returned incomplete membership evidence."
            )
        scale = float(np.linalg.norm(bounds[1] - bounds[0]))
        tolerance = float(np.asarray(query.tolerance.threshold(jnp.asarray(scale))))
        touching = np.abs(margin) <= tolerance
        return inside, touching

    contains = getattr(query, "contains", None)
    if not callable(contains):
        raise TypeError(
            "Prepared B-Rep source queries must provide contains(points, solid=...)."
        )
    classified = contains(points, solid=solid_index)
    if np.any(np.asarray(classified.resource_exhausted)):
        raise RuntimeError("Prepared B-Rep source query did not complete membership.")
    touching = np.asarray(classified.distances) <= float(query.tolerances.classifier)
    resolved = ~np.asarray(classified.unresolved)
    if np.any(~resolved & ~touching):
        raise ValueError(
            "B-Rep source-wall membership is unresolved for vertex IDs "
            f"{tuple(int(value) for value in vertex_ids[~resolved & ~touching])!r}."
        )
    return np.asarray(classified.inside), touching


class OversetPartSpec(StrictModule, NonTrainableState):
    """Declare one part's overset role and any rank-local source scopes.

    In collective registrations, ``owner_rank`` is an execution-group rank
    index; cell/vertex field-row ownership comes from the accepted sharded
    owner directory rather than this role label.
    """

    part_name: str = eqx.field(static=True)
    wall: MeshingScope | None
    boundary: MeshingScope | None
    excluded: MeshingScope | None

    solid_query: Any
    solid_index: int = eqx.field(static=True)
    priority: int = eqx.field(static=True)
    owner_rank: int = eqx.field(static=True)
    image_rotation: Array | None
    image_translation: Array | None
    spec_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_name: str,
        /,
        *,
        wall: MeshingScope | None = None,
        boundary: MeshingScope | None = None,
        excluded: MeshingScope | None = None,
        solid_query: Any = None,
        solid_index: int = 0,
        priority: int = 0,
        owner_rank: int = 0,
        image_rotation: ConvertibleToArray | None = None,
        image_translation: ConvertibleToArray | None = None,
    ) -> None:
        name = str(part_name).strip()
        if not name:
            raise ValueError("Overset part names must be non-empty.")
        for scope in (wall, boundary, excluded):
            if scope is not None and scope.source_id != name:
                raise ValueError("Overset scopes must belong to their declared part.")
        if isinstance(priority, bool) or not isinstance(priority, (int, np.integer)):
            raise TypeError("Overset priority must be an integer excluding bool.")
        if (
            isinstance(owner_rank, bool)
            or not isinstance(owner_rank, (int, np.integer))
            or owner_rank < 0
        ):
            raise ValueError(
                "Overset owner_rank must be a non-negative integer excluding bool."
            )
        if (
            isinstance(solid_index, bool)
            or not isinstance(solid_index, (int, np.integer))
            or solid_index < 0
        ):
            raise ValueError("solid_index must be a non-negative integer excluding bool.")
        if solid_query is not None and wall is None:
            raise ValueError("A source solid query requires a protected wall scope.")
        if (image_rotation is None) != (image_translation is None):
            raise ValueError(
                "image_rotation and image_translation must be supplied together."
            )
        rotation = translation = None
        if image_rotation is not None:
            rotation = np.asarray(image_rotation, dtype=np.float64)
            translation = np.asarray(image_translation, dtype=np.float64)
            if (
                rotation.ndim != 2
                or rotation.shape[0] not in (2, 3)
                or rotation.shape[1] != rotation.shape[0]
                or translation.shape != (rotation.shape[0],)
                or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(translation))
            ):
                raise ValueError(
                    "Overset image maps must be finite 2D/3D rigid transforms."
                )
            if not np.allclose(
                rotation.T @ rotation, np.eye(rotation.shape[0]), rtol=0, atol=1e-12
            ):
                raise ValueError(
                    "Overset image_rotation must be an orthogonal source-frame map."
                )
        self.part_name = name
        self.wall, self.boundary, self.excluded = wall, boundary, excluded
        self.solid_query, self.solid_index = solid_query, int(solid_index)
        self.priority, self.owner_rank = int(priority), int(owner_rank)
        self.image_rotation = None if rotation is None else jnp.asarray(rotation)
        self.image_translation = None if translation is None else jnp.asarray(translation)
        self.spec_id = canonical_fingerprint(
            {
                "kind": "overset-part-spec",
                "part": name,
                "wall": None if wall is None else wall.scope_id,
                "boundary": None if boundary is None else boundary.scope_id,
                "excluded": None if excluded is None else excluded.scope_id,
                "solid_query": _solid_query_identity(solid_query),
                "solid_index": self.solid_index,
                "priority": self.priority,
                "owner_rank": self.owner_rank,
                "image": array_tree_fingerprint((rotation, translation)),
            }
        )

    def rebound(self, part: MeshPart, /, *, solid_query: Any = None) -> OversetPartSpec:
        if part.name != self.part_name:
            raise ValueError("A rebound overset specification must retain the part name.")
        return OversetPartSpec(
            self.part_name,
            wall=None
            if self.wall is None
            else part.scope(
                self.wall.entity_dimension,
                self.wall.entity_ids,
                entity_set_id=self.wall.entity_set_id,
            ),
            boundary=None
            if self.boundary is None
            else part.scope(
                self.boundary.entity_dimension,
                self.boundary.entity_ids,
                entity_set_id=self.boundary.entity_set_id,
            ),
            excluded=None
            if self.excluded is None
            else part.scope(
                self.excluded.entity_dimension,
                self.excluded.entity_ids,
                entity_set_id=self.excluded.entity_set_id,
            ),
            solid_query=self.solid_query if solid_query is None else solid_query,
            solid_index=self.solid_index,
            priority=self.priority,
            owner_rank=self.owner_rank,
            image_rotation=self.image_rotation,
            image_translation=self.image_translation,
        )


class OversetPolicy(StrictModule, NonTrainableState):
    """Scientific donor selection and fringe-depth controls; no work quotas.

    ``"part-priority"`` prefers the donor part with the higher declared
    ``OversetPartSpec.priority`` and then the finer donor cell;
    ``"finest-donor"`` prefers the finer donor cell and breaks ties by the
    higher part priority.
    """

    fringe_layers: int = eqx.field(static=True)
    donor_priority: OversetDonorPriority = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        fringe_layers: int = 2,
        donor_priority: OversetDonorPriority = "finest-donor",
    ) -> None:
        if (
            isinstance(fringe_layers, bool)
            or not isinstance(fringe_layers, (int, np.integer))
            or fringe_layers < 1
        ):
            raise ValueError("fringe_layers must be a positive integer excluding bool.")
        donor_priority_ = parse(donor_priority, OversetDonorPriority, "donor_priority")
        self.fringe_layers = int(fringe_layers)
        self.donor_priority = donor_priority_
        self.policy_id = canonical_fingerprint(
            {
                "kind": "overset-policy",
                "fringe_layers": self.fringe_layers,
                "donor_priority": donor_priority_,
            }
        )


class OversetPartBlanking(StrictModule, NonTrainableState):
    vertex_ids: Array
    vertex_status: Array
    fringe_layer: Array
    wall_contact: Array
    protected_conflict: Array
    cell_ids: Array
    cell_status: Array
    ambiguous_cells: Array
    part_name: str = eqx.field(static=True)
    part_id: str = eqx.field(static=True)
    blanking_id: str = eqx.field(static=True)

    def __init__(
        self,
        part: MeshPart,
        vertex_status: ConvertibleToArray,
        fringe_layer: ConvertibleToArray,
        wall_contact: ConvertibleToArray,
        protected_conflict: ConvertibleToArray,
        cell_status: ConvertibleToArray,
        ambiguous_cells: ConvertibleToArray,
        /,
    ) -> None:
        if isinstance(part.carrier, OwnerLocalCellGeometry):
            vertex_ids = np.asarray(part.carrier.vertex_ids, dtype=np.int64)
            cell_ids = np.asarray(part.carrier.cell_ids, dtype=np.int64)
        else:
            mesh = _cell_result(part).mesh
            vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
            cell_ids = np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            )
        arrays = (
            np.asarray(vertex_status, dtype=np.int8),
            np.asarray(fringe_layer, dtype=np.int32),
            np.asarray(wall_contact, dtype=np.bool_),
            np.asarray(protected_conflict, dtype=np.bool_),
            np.asarray(cell_status, dtype=np.int8),
            np.asarray(ambiguous_cells, dtype=np.bool_),
        )
        if any(value.shape != vertex_ids.shape for value in arrays[:4]) or any(
            value.shape != cell_ids.shape for value in arrays[4:]
        ):
            raise ValueError(
                "Overset blanking records must align with stable vertex and cell IDs."
            )
        (
            self.vertex_ids,
            self.vertex_status,
            self.fringe_layer,
            self.wall_contact,
            self.protected_conflict,
        ) = tuple(jnp.asarray(value) for value in (vertex_ids, *arrays[:4]))
        self.cell_ids, self.cell_status, self.ambiguous_cells = tuple(
            jnp.asarray(value) for value in (cell_ids, *arrays[4:])
        )
        self.part_name, self.part_id = part.name, part.part_id
        self.blanking_id = canonical_fingerprint(
            {
                "kind": "overset-blanking",
                "part": part.part_id,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    def vertex_ids_with(self, status: int, /) -> np.ndarray:
        return np.sort(
            np.asarray(self.vertex_ids)[np.asarray(self.vertex_status) == int(status)]
        )

    def cell_ids_with(self, status: int, /) -> np.ndarray:
        return np.sort(
            np.asarray(self.cell_ids)[np.asarray(self.cell_status) == int(status)]
        )


@final
class OversetDonorRefusal(StrictModule):
    """A donor block whose mapped locator left one receptor's inverse map undecided.

    ``refusal.point`` is the receptor's slot in its receptor evidence and
    ``refusal.cell`` the donor block's local cell; the box and witness are the
    locator's exact evidence.
    """

    donor_part: str = eqx.field(static=True)
    donor_block: str = eqx.field(static=True)
    refusal: CellInverseRefusal

    def __init__(
        self, donor_part: str, donor_block: str, refusal: CellInverseRefusal, /
    ) -> None:
        if not isinstance(refusal, CellInverseRefusal):
            raise TypeError("refusal must be CellInverseRefusal.")
        self.donor_part, self.donor_block = str(donor_part), str(donor_block)
        self.refusal = refusal


class OversetReceptorEvidence(StrictModule, NonTrainableState):
    """Donor-search outcome per receptor, with the donor inverse-map refusals behind
    unresolved receptors ordered by receptor slot, donor part, and block."""

    receptor_ids: Array
    receptor_rows: Array
    status: Array
    donor_parts: Array
    donor_cells: Array
    candidate_parts: Array
    residuals: Array
    donor_measures: Array
    owner_ranks: Array
    donor_refusals: tuple[OversetDonorRefusal, ...]
    part_name: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_name: str,
        receptor_ids: ConvertibleToArray,
        receptor_rows: ConvertibleToArray,
        status: ConvertibleToArray,
        donor_parts: ConvertibleToArray,
        donor_cells: ConvertibleToArray,
        candidate_parts: ConvertibleToArray,
        residuals: ConvertibleToArray,
        donor_measures: ConvertibleToArray,
        owner_ranks: ConvertibleToArray,
        /,
        *,
        donor_refusals: tuple[OversetDonorRefusal, ...],
    ) -> None:
        arrays = (
            np.asarray(receptor_ids, dtype=np.int64),
            np.asarray(receptor_rows, dtype=np.int32),
            np.asarray(status, dtype=np.int32),
            np.asarray(donor_parts, dtype=np.int32),
            np.asarray(donor_cells, dtype=np.int64),
            np.asarray(candidate_parts, dtype=np.int32),
            np.asarray(residuals, dtype=np.float64),
            np.asarray(donor_measures, dtype=np.float64),
            np.asarray(owner_ranks),
        )
        if arrays[-1].dtype.kind not in "iu" or np.any(arrays[-1] < 0):
            raise ValueError(
                "Overset receptor owners must be non-negative integer ranks."
            )
        if arrays[0].ndim != 1 or any(array.shape != arrays[0].shape for array in arrays):
            raise ValueError(
                "Overset receptor evidence requires one record per stable receptor ID."
            )
        if np.any(np.diff(arrays[0]) <= 0):
            raise ValueError(
                "Overset receptor IDs must be strictly increasing and unique."
            )
        refusals = tuple(donor_refusals)
        if any(
            not isinstance(refusal, OversetDonorRefusal)
            or refusal.refusal.point >= arrays[0].size
            or arrays[2][refusal.refusal.point] != int(CouplingSearchStatus.UNRESOLVED)
            for refusal in refusals
        ):
            raise ValueError(
                "Overset donor refusals must name unresolved receptor slots."
            )
        (
            self.receptor_ids,
            self.receptor_rows,
            self.status,
            self.donor_parts,
            self.donor_cells,
            self.candidate_parts,
            self.residuals,
            self.donor_measures,
            self.owner_ranks,
        ) = tuple(jnp.asarray(value) for value in arrays)
        self.part_name = str(part_name)
        self.donor_refusals = refusals
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "overset-receptors",
                "part": self.part_name,
                "arrays": array_tree_fingerprint(arrays),
            }
        )

    @property
    def found(self) -> Array:
        return self.status == int(CouplingSearchStatus.FOUND)

    @property
    def orphan_ids(self) -> np.ndarray:
        return np.asarray(self.receptor_ids)[~np.asarray(self.found)]


class OversetDonorPacket(StrictModule, NonTrainableState):
    receptor_ids: Array
    receptor_slots: Array
    query_points: Array
    target_points: Array
    donor_cells: Array
    donor_cell_rows: Array
    reference_coordinates: Array
    rotation: Array
    translation: Array
    donor_block: str = eqx.field(static=True)
    donor_part: str = eqx.field(static=True)
    receptor_part: str = eqx.field(static=True)
    donor_revision: str = eqx.field(static=True)
    receptor_revision: str = eqx.field(static=True)
    donor_rank: int = eqx.field(static=True)
    receptor_rank: int = eqx.field(static=True)
    packet_id: str = eqx.field(static=True)

    def __init__(
        self,
        donor: MeshPart | MeshPartManifest,
        receptor: MeshPart | MeshPartManifest,
        donor_spec: OversetPartSpec,
        receptor_spec: OversetPartSpec,
        receptor_ids: ConvertibleToArray,
        slots: ConvertibleToArray,
        query_points: ConvertibleToArray,
        target_points: ConvertibleToArray,
        donor_cells: ConvertibleToArray,
        donor_cell_rows: ConvertibleToArray,
        reference_coordinates: ConvertibleToArray,
        /,
        *,
        donor_block: str,
        donor_rank: int | None = None,
        receptor_rank: int | None = None,
    ) -> None:
        ids = np.asarray(receptor_ids, dtype=np.int64)
        slots_ = np.asarray(slots, dtype=np.int32)
        query = np.asarray(query_points, dtype=np.float64)
        target = np.asarray(target_points, dtype=np.float64)
        cells = np.asarray(donor_cells, dtype=np.int64)
        rows = np.asarray(donor_cell_rows, dtype=np.int32)
        reference = np.asarray(reference_coordinates, dtype=np.float64)
        if ids.size == 0 or ids.ndim != 1 or np.any(np.diff(ids) <= 0):
            raise ValueError(
                "Overset donor packets require sorted non-empty receptor IDs."
            )
        if any(
            value.shape[0] != ids.size
            for value in (slots_, query, target, cells, rows, reference)
        ):
            raise ValueError(
                "Overset donor packet arrays must have one row per receptor."
            )
        if query.ndim != 2 or target.shape != query.shape:
            raise ValueError("Overset packet points must share a physical frame.")
        rotation, translation = _relative_image(donor_spec, receptor_spec, query.shape[1])
        if not np.allclose(
            _to_target(query, donor_spec, receptor_spec), target, rtol=0.0, atol=1e-10
        ):
            raise ValueError(
                "Overset donor and receptor sites do not share a physical frame."
            )
        if isinstance(donor, MeshPart):
            if isinstance(donor.carrier, OwnerLocalCellGeometry):
                block_names = tuple(block.name for block in donor.carrier.blocks)
            else:
                block_names = tuple(
                    block.name for block in _cell_result(donor).mesh.blocks
                )
            if donor_block not in block_names:
                raise ValueError(
                    "Overset donor packet block is absent from its local part."
                )
        elif not donor_block:
            raise ValueError("Owner-local donor packet requires a declared block name.")
        donor_owner = donor_spec.owner_rank if donor_rank is None else int(donor_rank)
        receptor_owner = (
            receptor_spec.owner_rank if receptor_rank is None else int(receptor_rank)
        )
        if donor_owner < 0 or receptor_owner < 0:
            raise ValueError("Overset packet owner ranks must be non-negative.")
        self.receptor_ids, self.receptor_slots = jnp.asarray(ids), jnp.asarray(slots_)
        self.query_points, self.target_points = jnp.asarray(query), jnp.asarray(target)
        self.donor_cells, self.donor_cell_rows = jnp.asarray(cells), jnp.asarray(rows)
        self.reference_coordinates = jnp.asarray(reference)
        self.rotation, self.translation = jnp.asarray(rotation), jnp.asarray(translation)
        self.donor_block = donor_block
        self.donor_part, self.receptor_part = donor.name, receptor.name
        self.donor_revision, self.receptor_revision = donor.part_id, receptor.part_id
        self.donor_rank, self.receptor_rank = donor_owner, receptor_owner
        self.packet_id = canonical_fingerprint(
            {
                "kind": "overset-donor-packet",
                "donor": donor.part_id,
                "receptor": receptor.part_id,
                "block": donor_block,
                "receptor_ids": array_tree_fingerprint(ids),
                "slots": array_tree_fingerprint(slots_),
                "query": array_tree_fingerprint(query),
                "cells": array_tree_fingerprint(cells),
                "cell_rows": array_tree_fingerprint(rows),
                "reference": array_tree_fingerprint(reference),
                "rotation": array_tree_fingerprint(rotation),
                "translation": array_tree_fingerprint(translation),
            }
        )


class OversetConnectivityError(ValueError):
    def __init__(self, message: str, connectivity: OversetConnectivity, /) -> None:
        super().__init__(message)
        self.connectivity = connectivity


class _PartGeometry(StrictModule, NonTrainableState):
    part_id: str = eqx.field(static=True)
    cell_vertices: Array
    cell_vertex_valid: Array
    cell_ids: Array
    cell_measures: Array
    cell_lower: Array
    cell_upper: Array
    cell_locators: tuple[AbstractCellLocator, ...]
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    block_names: tuple[str, ...] = eqx.field(static=True)
    wall_facets: Array | None
    wall_lower: Array | None
    wall_upper: Array | None


def _owner_local_carrier(part: MeshPart, /) -> OwnerLocalCellGeometry:
    carrier = part.carrier
    if not isinstance(carrier, OwnerLocalCellGeometry):
        raise TypeError("Owner-local overset geometry requires compact cell views.")
    return carrier


def _owner_local_manifest_binding(manifest: MeshPartManifest, /) -> tuple[str, str]:
    if (
        not isinstance(manifest, MeshPartManifest)
        or manifest.distribution_id is None
        or manifest.source_topology_id is None
        or manifest.source_geometry_id is None
    ):
        raise ValueError(
            "Owner-local manifests require current topology, geometry, and distribution identities."
        )
    return manifest.source_topology_id, manifest.source_geometry_id


def _owner_local_geometry(part: MeshPart, spec: OversetPartSpec, /) -> _PartGeometry:
    view = _owner_local_carrier(part)
    dimension = view.topological_dimension
    ambient = view.coordinates.shape[1]
    if dimension != ambient or ambient not in (2, 3):
        raise ValueError(
            "Owner-local overset requires full-dimensional 2-D or 3-D cells."
        )
    measures_by_id = dict(
        zip(
            np.asarray(view.cell_ids, dtype=np.int64).tolist(),
            np.asarray(view.cell_measures, dtype=np.float64).tolist(),
            strict=True,
        )
    )
    cell_rows: list[tuple[int, ...]] = []
    cell_ids: list[np.ndarray] = []
    measures: list[np.ndarray] = []
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    locators: list[AbstractCellLocator] = []
    block_names: list[str] = []
    block_offsets = [0]
    for block in view.blocks:
        identifiers = np.asarray(block.cell_ids, dtype=np.int64)
        if not identifiers.size:
            continue
        locator = (
            PreparedPolyhedralCellLocator.from_owner_local_geometry(view, block.name)
            if block.cell_kind == "polyhedron"
            else prepare_owner_local_mapped_cell_locator(view, block.name)
        )
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        if (
            vertices.ndim != 2
            or vertices.shape != valid.shape
            or vertices.shape[0] != identifiers.size
        ):
            raise ValueError(
                "Owner-local cell vertices do not align with stable cell IDs."
            )
        cell_rows.extend(
            tuple(
                int(vertices[row, column])
                for column in range(vertices.shape[1])
                if valid[row, column]
            )
            for row in range(vertices.shape[0])
        )
        cell_ids.append(identifiers)
        measures.append(
            np.asarray(
                [measures_by_id[int(identifier)] for identifier in identifiers],
                dtype=np.float64,
            )
        )
        if isinstance(locator, PreparedPolyhedralCellLocator):
            cell_lower = np.asarray(locator.cell_map.cell_lower, dtype=np.float64)
            cell_upper = np.asarray(locator.cell_map.cell_upper, dtype=np.float64)
        else:
            cell_lower = np.asarray(locator.cell_lower, dtype=np.float64)
            cell_upper = np.asarray(locator.cell_upper, dtype=np.float64)
        lower.append(cell_lower)
        upper.append(cell_upper)
        locators.append(locator)
        block_names.append(block.name)
        block_offsets.append(block_offsets[-1] + identifiers.size)
    if cell_rows:
        packed_vertices, vertex_valid = _pack_cell_vertices(tuple(cell_rows))
        ids_ = np.concatenate(cell_ids)
        measures_ = np.concatenate(measures)
        lower_ = np.concatenate(lower)
        upper_ = np.concatenate(upper)
    else:
        packed_vertices = np.empty((0, 0), dtype=np.int64)
        vertex_valid = np.empty((0, 0), dtype=np.bool_)
        ids_ = np.empty((0,), dtype=np.int64)
        measures_ = np.empty((0,), dtype=np.float64)
        lower_ = np.empty((0, ambient), dtype=np.float64)
        upper_ = np.empty((0, ambient), dtype=np.float64)
    if (
        ids_.shape != (len(cell_rows),)
        or measures_.shape != ids_.shape
        or lower_.shape != (ids_.size, ambient)
        or upper_.shape != lower_.shape
    ):
        raise ValueError("Owner-local locator bounds do not align with source cell rows.")
    wall_facets = wall_lower = wall_upper = None
    if spec.wall is not None:
        part.require_scope(spec.wall)
        if spec.wall.entity_dimension != dimension - 1:
            raise ValueError(
                "Owner-local wall scopes must identify codimension-one entities."
            )
        wall_loops = view.entity_vertex_loops(
            spec.wall.entity_dimension, np.asarray(spec.wall.entity_ids, dtype=np.int64)
        )
        simplices: list[np.ndarray] = []
        for loop in wall_loops:
            points = np.asarray(
                view.point_coordinates(np.asarray(view.vertex_ids)[loop]),
                dtype=np.float64,
            )
            if dimension == 2:
                if points.shape != (2, 2):
                    raise ValueError(
                        "Owner-local planar walls require two-vertex facets."
                    )
                simplices.append(points)
            else:
                if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
                    raise ValueError(
                        "Owner-local spatial walls require polygonal facets."
                    )
                simplices.extend(
                    points[[0, index, index + 1]]
                    for index in range(1, points.shape[0] - 1)
                )
        if simplices:
            wall_facets = np.asarray(simplices, dtype=np.float64)
            wall_lower = np.min(wall_facets, axis=1)
            wall_upper = np.max(wall_facets, axis=1)
    return _PartGeometry(
        part_id=part.part_id,
        cell_vertices=jnp.asarray(packed_vertices),
        cell_vertex_valid=jnp.asarray(vertex_valid),
        cell_ids=jnp.asarray(ids_),
        cell_measures=jnp.asarray(measures_),
        cell_lower=jnp.asarray(lower_),
        cell_upper=jnp.asarray(upper_),
        cell_locators=tuple(locators),
        block_offsets=tuple(block_offsets),
        block_names=tuple(block_names),
        wall_facets=None if wall_facets is None else jnp.asarray(wall_facets),
        wall_lower=None if wall_lower is None else jnp.asarray(wall_lower),
        wall_upper=None if wall_upper is None else jnp.asarray(wall_upper),
    )


def _owner_local_point_bank(
    assembly: MeshAssembly,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[tuple[tuple[str, int], ...], tuple[int, ...], np.ndarray]:
    from ._overset_exchange import _synchronize_owner_search_failure

    group_ranks = execution_group.spec.process_indices
    group_size = len(group_ranks)
    local_rank = assembly.local_rank
    if local_rank is None or not 0 <= local_rank < group_size:
        raise ValueError(
            "Owner-local assembly rank is outside the registered execution group."
        )
    part_order = {name: index for index, name in enumerate(assembly.global_part_names)}
    names = assembly.global_part_names
    dimension = assembly.part_manifest[0].ambient_dimension
    metadata: list[tuple[int, int, int]] = []
    points: list[np.ndarray] = []
    local_error: Exception | None = None
    try:
        for part in assembly.parts:
            view = part.carrier
            if (
                not isinstance(view, OwnerLocalCellGeometry)
                or view.owner_rank != local_rank
            ):
                raise TypeError(
                    "Owner-local receptor queries require this rank's compact source view."
                )
            ids = np.asarray(view.vertex_ids, dtype=np.int64)
            owners = np.asarray(view.vertex_owner_ranks, dtype=np.int32)
            selected = np.flatnonzero(owners == local_rank)
            coordinates = np.asarray(
                view.point_coordinates(ids[selected]), dtype=np.float64
            )
            if coordinates.shape != (selected.size, dimension) or np.any(
                ~np.isfinite(coordinates)
            ):
                raise ValueError(
                    "Owner-local point coordinates do not match their vertex owners."
                )
            metadata.extend(
                (part_order[part.name], int(ids[row]), int(owners[row]))
                for row in selected
            )
            points.extend(coordinates)
        local_metadata = np.asarray(metadata, dtype=np.int64).reshape((-1, 3))
        local_points = (
            np.asarray(points, dtype=np.float64).reshape((-1, dimension))
            if points
            else np.empty((0, dimension), dtype=np.float64)
        )
    except MemoryError:
        local_error = MemoryError(
            "Owner-local receptor query preparation failed allocation."
        )
        local_metadata = np.empty((0, 3), dtype=np.int64)
        local_points = np.empty((0, dimension), dtype=np.float64)
    except Exception as error:
        local_error = error
        local_metadata = np.empty((0, 3), dtype=np.int64)
        local_points = np.empty((0, dimension), dtype=np.float64)
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another rank rejected its local vertices"
        )
        raise ValueError(
            "Owner-local receptor query preparation failed collectively: " + detail
        )
    order = sorted(
        range(local_metadata.shape[0]),
        key=lambda row: (int(local_metadata[row, 0]), int(local_metadata[row, 1])),
    )
    keys = tuple(
        (names[int(local_metadata[row, 0])], int(local_metadata[row, 1])) for row in order
    )
    if len(set(keys)) != len(keys):
        raise ValueError("Owner-local vertex directory published duplicate point owners.")
    owners = tuple(int(local_metadata[row, 2]) for row in order)
    points_ = (
        local_points[np.asarray(order, dtype=np.int64)]
        if order
        else np.empty((0, dimension), dtype=np.float64)
    )
    if any(owner != local_rank for owner in owners):
        raise ValueError("Local receptor query rows disagree with their source owners.")
    return keys, owners, points_


def _owner_local_scope_vertices(
    part: MeshPart, scope: MeshingScope | None, /
) -> np.ndarray:
    if scope is None:
        return np.empty((0,), dtype=np.int64)
    carrier = part.carrier
    if not isinstance(carrier, OwnerLocalCellGeometry):
        raise TypeError("Owner-local scopes require compact cell geometry.")
    part.require_scope(scope)
    if scope.entity_dimension != carrier.topological_dimension - 1:
        raise ValueError(
            "Overset boundary scopes must identify codimension-one entities."
        )
    loops = carrier.entity_vertex_loops(
        scope.entity_dimension, np.asarray(scope.entity_ids, dtype=np.int64)
    )
    if not loops:
        return np.empty((0,), dtype=np.int64)
    rows = np.unique(
        np.concatenate(tuple(np.asarray(loop, dtype=np.int64) for loop in loops))
    )
    return np.asarray(carrier.vertex_ids, dtype=np.int64)[rows]


def _owner_local_scope_metadata(
    assembly: MeshAssembly,
    specs: tuple[OversetPartSpec, ...],
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[
    set[tuple[str, int]],
    set[tuple[str, int]],
    set[tuple[str, int]],
    set[tuple[str, int]],
    set[tuple[str, int]],
    dict[str, tuple[int, ...]],
    dict[str, tuple[int, ...]],
]:
    from ._overset_exchange import (
        _owner_local_all_gather_rows,
        _synchronize_owner_search_failure,
    )

    names = assembly.global_part_names
    part_order = {name: index for index, name in enumerate(names)}
    spec_by_name = {spec.part_name: spec for spec in specs}
    local_rank = assembly.local_rank
    if local_rank is None:
        raise ValueError("Owner-local scope metadata requires a rank-local assembly.")
    local_wall: list[tuple[int, int, int, int]] = []
    local_wall_ridges: list[tuple[int, int, int]] = []
    local_boundary_vertices: list[tuple[int, int]] = []
    local_protected_vertices: list[tuple[int, int]] = []
    local_excluded: list[tuple[int, int]] = []
    local_boundary_facets: list[tuple[int, int]] = []
    local_error: Exception | None = None
    try:
        for part in assembly.parts:
            view = part.carrier
            if not isinstance(view, OwnerLocalCellGeometry):
                raise TypeError(
                    "Owner-local scope metadata requires compact geometry views."
                )
            spec = spec_by_name[part.name]
            if spec.wall is not None:
                part.require_scope(spec.wall)
                if spec.wall.entity_dimension != view.topological_dimension - 1:
                    raise ValueError(
                        "Overset wall scopes must identify codimension-one entities."
                    )
                local_wall.extend(
                    (
                        part_order[part.name],
                        int(identifier),
                        spec.wall.entity_dimension,
                        int(local_rank),
                    )
                    for identifier in np.asarray(spec.wall.entity_ids, dtype=np.int64)
                )
                local_vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
                for loop in view.entity_vertex_loops(
                    spec.wall.entity_dimension,
                    np.asarray(spec.wall.entity_ids, dtype=np.int64),
                ):
                    identifiers = local_vertex_ids[np.asarray(loop, dtype=np.int64)]
                    if assembly.part_manifest[0].ambient_dimension == 2:
                        local_wall_ridges.extend(
                            (part_order[part.name], int(identifier), -1)
                            for identifier in identifiers
                        )
                    else:
                        local_wall_ridges.extend(
                            (
                                part_order[part.name],
                                min(int(first), int(second)),
                                max(int(first), int(second)),
                            )
                            for first, second in zip(
                                identifiers, np.roll(identifiers, -1), strict=True
                            )
                        )
                local_protected_vertices.extend(
                    (part_order[part.name], int(identifier))
                    for identifier in _owner_local_scope_vertices(part, spec.wall)
                )
            if spec.boundary is not None:
                part.require_scope(spec.boundary)
                if spec.boundary.entity_dimension != view.topological_dimension - 1:
                    raise ValueError(
                        "Overset receptor boundaries must identify codimension-one entities."
                    )
                local_boundary_facets.extend(
                    (part_order[part.name], int(identifier))
                    for identifier in np.asarray(spec.boundary.entity_ids, dtype=np.int64)
                )
                local_boundary_vertices.extend(
                    (part_order[part.name], int(identifier))
                    for identifier in _owner_local_scope_vertices(part, spec.boundary)
                )
            if spec.excluded is not None:
                part.require_scope(spec.excluded)
                if spec.excluded.entity_dimension != view.topological_dimension:
                    raise ValueError("Overset excluded scopes must identify cells.")
                local_excluded.extend(
                    (part_order[part.name], int(identifier))
                    for identifier in np.asarray(spec.excluded.entity_ids, dtype=np.int64)
                )
    except MemoryError:
        local_error = MemoryError("Owner-local scope preparation failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another rank rejected its local scopes"
        )
        raise ValueError("Owner-local scope preparation failed collectively: " + detail)
    if local_error is not None:
        raise ValueError("Owner-local scope preparation failed collectively.")
    wall_rows = _owner_local_all_gather_rows(
        np.asarray(local_wall, dtype=np.int64).reshape((-1, 4)),
        execution_group.devices,
        axis_name,
    )
    wall_ridge_rows = _owner_local_all_gather_rows(
        np.asarray(local_wall_ridges, dtype=np.int64).reshape((-1, 3)),
        execution_group.devices,
        axis_name,
    )
    boundary_rows = _owner_local_all_gather_rows(
        np.asarray(local_boundary_vertices, dtype=np.int64).reshape((-1, 2)),
        execution_group.devices,
        axis_name,
    )
    boundary_facet_rows = _owner_local_all_gather_rows(
        np.asarray(local_boundary_facets, dtype=np.int64).reshape((-1, 2)),
        execution_group.devices,
        axis_name,
    )
    protected_rows = _owner_local_all_gather_rows(
        np.asarray(local_protected_vertices, dtype=np.int64).reshape((-1, 2)),
        execution_group.devices,
        axis_name,
    )
    excluded_rows = _owner_local_all_gather_rows(
        np.asarray(local_excluded, dtype=np.int64).reshape((-1, 2)),
        execution_group.devices,
        axis_name,
    )
    wall_keys = [
        (int(row[0]), int(row[1])) for rank_rows in wall_rows for row in rank_rows
    ]
    boundary_keys = [
        (int(row[0]), int(row[1]))
        for rank_rows in boundary_facet_rows
        for row in rank_rows
    ]
    if len(set(boundary_keys)) != len(boundary_keys):
        raise ValueError(
            "Owner-local receptor-boundary scopes duplicate a global face ID."
        )
    if len(set(wall_keys)) != len(wall_keys):
        raise ValueError("Owner-local wall facet scopes duplicate a global face ID.")
    wall_dimensions: dict[str, tuple[int, ...]] = {}
    for part_index, name in enumerate(names):
        dimensions = tuple(
            sorted(
                {
                    int(row[2])
                    for rank_rows in wall_rows
                    for row in rank_rows
                    if int(row[0]) == part_index
                }
            )
        )
        if len(dimensions) > 1:
            raise ValueError("Owner-local wall facets disagree on entity dimension.")
        if (
            spec_by_name[name].wall is not None
            and not spec_by_name[name].solid_query
            and not dimensions
        ):
            raise ValueError("An authored local wall requires complete facet ownership.")
        wall_dimensions[name] = dimensions
    ridge_counts: dict[tuple[int, int, int], int] = {}
    for rank_rows in wall_ridge_rows:
        if rank_rows.ndim != 2 or rank_rows.shape[1] != 3:
            raise ValueError(
                "Owner-local wall ridge records require three integer columns."
            )
        for row in range(rank_rows.shape[0]):
            key = (int(rank_rows[row, 0]), int(rank_rows[row, 1]), int(rank_rows[row, 2]))
            ridge_counts[key] = ridge_counts.get(key, 0) + 1
    for part_index, name in enumerate(names):
        if not wall_dimensions[name] or spec_by_name[name].solid_query is not None:
            continue
        part_ridges = tuple(
            count for (index, _, _), count in ridge_counts.items() if index == part_index
        )
        if not part_ridges or any(count != 2 for count in part_ridges):
            raise ValueError(
                "Owner-local overset wall facets must form a closed two-manifold."
            )
    wall_owner_ranks = {
        name: tuple(
            sorted(
                {
                    int(row[3])
                    for rank_rows in wall_rows
                    for row in rank_rows
                    if int(row[0]) == part_index
                }
            )
        )
        for part_index, name in enumerate(names)
    }
    boundaries = {
        (names[int(row[0])], int(row[1]))
        for rank_rows in boundary_rows
        for row in rank_rows
    }
    protected = {
        (names[int(row[0])], int(row[1]))
        for rank_rows in protected_rows
        for row in rank_rows
    }
    excluded = {
        (names[int(row[0])], int(row[1]))
        for rank_rows in excluded_rows
        for row in rank_rows
    }
    wall_facets = {(names[index], identifier) for index, identifier in wall_keys}
    boundary_facets = {(names[index], identifier) for index, identifier in boundary_keys}
    if wall_facets.intersection(boundary_facets):
        raise ValueError("Overset solid walls and receptor boundaries must be disjoint.")
    return (
        boundaries,
        protected,
        excluded,
        wall_facets,
        boundary_facets,
        wall_dimensions,
        wall_owner_ranks,
    )


def _owner_local_wall_membership(
    assembly: MeshAssembly,
    specs: tuple[OversetPartSpec, ...],
    local_queries: tuple[tuple[str, int], ...],
    local_query_owners: tuple[int, ...],
    local_query_points: np.ndarray,
    wall_dimensions: Mapping[str, tuple[int, ...]],
    wall_owner_ranks: Mapping[str, tuple[int, ...]],
    geometries: Mapping[str, _PartGeometry],
    execution_group: ExecutionGroup,
    axis_name: str,
    dimension: int,
    /,
) -> tuple[dict[tuple[str, int], bool], dict[tuple[str, int], bool]]:
    from ._distribution import _owner_exchange_message_banks
    from ._overset_exchange import _synchronize_owner_search_failure

    group_ranks = execution_group.spec.process_indices
    process_rank = jax.process_index()
    if process_rank not in group_ranks:
        raise ValueError("Owner-local wall queries are outside the execution group.")
    local_rank = group_ranks.index(process_rank)
    if (
        len(local_queries) != len(local_query_owners)
        or local_query_points.shape != (len(local_queries), dimension)
        or any(owner != local_rank for owner in local_query_owners)
    ):
        raise ValueError("Local receptor queries do not match their point owners.")
    names = tuple(assembly.global_part_names)
    part_order = {name: index for index, name in enumerate(names)}
    spec_by_name = {spec.part_name: spec for spec in specs}
    wall_sources = tuple(
        spec
        for spec in specs
        if spec.solid_query is None and wall_dimensions.get(spec.part_name, ())
    )
    wall_destinations = tuple(
        sorted(
            {
                int(rank)
                for source in wall_sources
                for rank in wall_owner_ranks.get(source.part_name, ())
            }
        )
    )
    part_count = len(group_ranks)
    outgoing_ids: list[list[tuple[int, int]]] = [[] for _ in range(part_count)]
    outgoing_points: list[list[np.ndarray]] = [[] for _ in range(part_count)]
    for destination in wall_destinations:
        if destination == local_rank:
            continue
        for (part_name, identifier), point in zip(
            local_queries, local_query_points, strict=True
        ):
            outgoing_ids[destination].append((part_order[part_name], int(identifier)))
            outgoing_points[destination].append(np.asarray(point, dtype=np.float64))
    id_banks = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 2)) for rows in outgoing_ids
    )
    point_banks = tuple(
        np.asarray(rows, dtype=np.float64).reshape((-1, dimension))
        for rows in outgoing_points
    )
    incoming_ids = _owner_exchange_message_banks(
        id_banks,
        local_rank,
        execution_group,
        f"{axis_name}_wall_query_ids",
    )
    incoming_points = _owner_exchange_message_banks(
        point_banks,
        local_rank,
        execution_group,
        f"{axis_name}_wall_query_points",
    )
    local_error: Exception | None = None
    query_records: list[tuple[int, int, int, np.ndarray]] = [
        (
            local_rank,
            part_order[name],
            int(identifier),
            np.asarray(point, dtype=np.float64),
        )
        for (name, identifier), point in zip(
            local_queries, local_query_points, strict=True
        )
    ]
    try:
        if set(incoming_ids) != set(incoming_points):
            raise ValueError(
                "Wall-query IDs and coordinates have different owner routes."
            )
        for sender, rows in incoming_ids.items():
            points = incoming_points[sender]
            if points.shape != (rows.shape[0], dimension):
                raise ValueError("A wall owner received malformed query coordinates.")
            query_records.extend(
                (
                    sender,
                    int(metadata[0]),
                    int(metadata[1]),
                    np.asarray(point, dtype=np.float64),
                )
                for metadata, point in zip(rows, points, strict=True)
            )
    except MemoryError:
        local_error = MemoryError("Owner-local wall-query receive failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another wall owner rejected query coordinates"
        )
        raise ValueError("Owner-local wall-query receive failed collectively: " + detail)

    solid_flags: dict[tuple[str, int, str], tuple[bool, bool]] = {}
    local_wall_replies: list[tuple[int, int, int, int, int]] = []
    reply_rows: list[list[tuple[int, int, int, int, int]]] = [
        [] for _ in range(part_count)
    ]
    local_error = None
    try:
        for (target_name, target_id), point in zip(
            local_queries, local_query_points, strict=True
        ):
            target_spec = spec_by_name[target_name]
            for source_spec in specs:
                if (
                    source_spec.part_name == target_name
                    or source_spec.solid_query is None
                ):
                    continue
                source_point = _to_source(
                    np.asarray(point, dtype=np.float64)[None, :],
                    target_spec,
                    source_spec,
                )
                inside, touching = _source_solid_membership(
                    source_spec.solid_query,
                    source_point,
                    source_spec.solid_index,
                    np.asarray((target_id,), dtype=np.int64),
                )
                solid_flags[target_name, int(target_id), source_spec.part_name] = (
                    bool(inside[0]),
                    bool(touching[0]),
                )
        for requester, target_index, target_id, point in query_records:
            target_name = names[target_index]
            target_spec = spec_by_name[target_name]
            for source_spec in wall_sources:
                if (
                    source_spec.part_name == target_name
                    or local_rank not in wall_owner_ranks.get(source_spec.part_name, ())
                ):
                    continue
                geometry = geometries.get(source_spec.part_name)
                if geometry is None or geometry.wall_facets is None:
                    raise ValueError(
                        "A declared wall owner lacks its compact local facets."
                    )
                facets = np.asarray(geometry.wall_facets, dtype=np.float64)
                if not facets.size:
                    raise ValueError(
                        "A declared wall owner has no authored local facets."
                    )
                source_points = _to_source(
                    np.asarray(point, dtype=np.float64)[None, :],
                    target_spec,
                    source_spec,
                )
                inside, touching = _classify_facets(source_points, facets, dimension)
                row = (
                    target_index,
                    int(target_id),
                    part_order[source_spec.part_name],
                    int(inside[0]),
                    int(touching[0]),
                )
                if requester == local_rank:
                    local_wall_replies.append(row)
                else:
                    reply_rows[requester].append(row)
    except MemoryError:
        local_error = MemoryError(
            "Owner-local source-solid/wall classification failed allocation."
        )
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another owner rejected its wall classification"
        )
        raise ValueError(
            "Owner-local source-solid/wall classification failed collectively: " + detail
        )
    reply_banks = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 5)) for rows in reply_rows
    )
    incoming_replies = _owner_exchange_message_banks(
        reply_banks,
        local_rank,
        execution_group,
        f"{axis_name}_wall_query_replies",
    )
    wall_contributions: dict[tuple[int, int, int], dict[int, tuple[bool, bool]]] = {}
    contribution_error: Exception | None = None
    try:
        for target_index, target_id, source_index, inside, touching in local_wall_replies:
            key = (target_index, target_id, source_index)
            wall_contributions.setdefault(key, {})[local_rank] = (
                bool(inside),
                bool(touching),
            )
        for source_rank, rows in incoming_replies.items():
            for target_index, target_id, source_index, inside, touching in rows:
                key = (int(target_index), int(target_id), int(source_index))
                contributions = wall_contributions.setdefault(key, {})
                if source_rank in contributions:
                    raise ValueError(
                        "A wall owner published duplicate membership evidence."
                    )
                contributions[source_rank] = (bool(inside), bool(touching))
        holes_by_key: dict[tuple[str, int], bool] = {}
        contacts_by_key: dict[tuple[str, int], bool] = {}
        for target_name, target_id in local_queries:
            hole = False
            contact = False
            for source_index, source_spec in enumerate(specs):
                if source_spec.part_name == target_name:
                    continue
                key = (target_name, int(target_id))
                if source_spec.solid_query is not None:
                    inside, touching = solid_flags.get(
                        (target_name, int(target_id), source_spec.part_name),
                        (False, False),
                    )
                elif wall_dimensions.get(source_spec.part_name, ()):
                    owners = wall_owner_ranks.get(source_spec.part_name, ())
                    contributions = wall_contributions.get(
                        (part_order[target_name], int(target_id), source_index), {}
                    )
                    if set(contributions) != set(owners):
                        raise ValueError(
                            f"Wall membership for receptor {key!r} omitted a facet owner."
                        )
                    inside = bool(
                        np.count_nonzero([contributions[owner][0] for owner in owners])
                        % 2
                    )
                    touching = any(contributions[owner][1] for owner in owners)
                else:
                    inside, touching = False, False
                hole |= inside or touching
                contact |= touching
            holes_by_key[target_name, int(target_id)] = hole
            contacts_by_key[target_name, int(target_id)] = contact
    except MemoryError:
        contribution_error = MemoryError(
            "Owner-local wall evidence reduction failed allocation."
        )
        holes_by_key, contacts_by_key = {}, {}
    except Exception as error:
        contribution_error = error
        holes_by_key, contacts_by_key = {}, {}
    if _synchronize_owner_search_failure(
        contribution_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(contribution_error)
            if contribution_error is not None
            else "another point owner received incomplete wall evidence"
        )
        raise ValueError("Owner-local wall evidence failed collectively: " + detail)

    outgoing_requests: list[list[tuple[int, int]]] = [[] for _ in range(part_count)]
    for part in assembly.parts:
        view = _owner_local_carrier(part)
        vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
        owners = np.asarray(view.vertex_owner_ranks, dtype=np.int32)
        for identifier, owner in zip(vertex_ids, owners, strict=True):
            if int(owner) != local_rank:
                outgoing_requests[int(owner)].append(
                    (part_order[part.name], int(identifier))
                )
    request_banks = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 2)) for rows in outgoing_requests
    )
    incoming_requests = _owner_exchange_message_banks(
        request_banks,
        local_rank,
        execution_group,
        f"{axis_name}_vertex_flag_requests",
    )
    outgoing_replies: list[list[tuple[int, int, int, int]]] = [
        [] for _ in range(part_count)
    ]
    ghost_error: Exception | None = None
    try:
        for requester, rows in incoming_requests.items():
            for part_index, identifier in rows:
                key = (names[int(part_index)], int(identifier))
                if key not in holes_by_key or key not in contacts_by_key:
                    raise ValueError("A vertex owner lacks its local blanking result.")
                outgoing_replies[requester].append(
                    (
                        int(part_index),
                        int(identifier),
                        int(holes_by_key[key]),
                        int(contacts_by_key[key]),
                    )
                )
    except MemoryError:
        ghost_error = MemoryError("Owner-local ghost flag preparation failed allocation.")
    except Exception as error:
        ghost_error = error
    if _synchronize_owner_search_failure(
        ghost_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(ghost_error)
            if ghost_error is not None
            else "another vertex owner lacks blanking evidence"
        )
        raise ValueError(
            "Owner-local ghost blanking preparation failed collectively: " + detail
        )
    incoming_flags = _owner_exchange_message_banks(
        tuple(
            np.asarray(rows, dtype=np.int64).reshape((-1, 4)) for rows in outgoing_replies
        ),
        local_rank,
        execution_group,
        f"{axis_name}_vertex_flag_replies",
    )
    local_error = None
    try:
        for rows in incoming_flags.values():
            for part_index, identifier, hole, contact in rows:
                key = (names[int(part_index)], int(identifier))
                if key in holes_by_key:
                    raise ValueError(
                        "A local vertex has duplicate owner blanking evidence."
                    )
                holes_by_key[key] = bool(hole)
                contacts_by_key[key] = bool(contact)
        for part in assembly.parts:
            for identifier in np.asarray(
                _owner_local_carrier(part).vertex_ids, dtype=np.int64
            ):
                key = (part.name, int(identifier))
                if key not in holes_by_key or key not in contacts_by_key:
                    raise ValueError(
                        "Local vertex halo lacks completed blanking evidence."
                    )
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another rank lacks completed vertex halos"
        )
        raise ValueError(
            "Owner-local vertex blanking halos failed collectively: " + detail
        )
    return holes_by_key, contacts_by_key


def _owner_local_cut_cells(
    assembly: MeshAssembly,
    specs: tuple[OversetPartSpec, ...],
    wall_dimensions: Mapping[str, tuple[int, ...]],
    wall_owner_ranks: Mapping[str, tuple[int, ...]],
    geometries: Mapping[str, _PartGeometry],
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> set[tuple[str, int]]:
    from ._distribution import _owner_exchange_message_banks
    from ._overset_exchange import _synchronize_owner_search_failure

    names = assembly.global_part_names
    part_order = {name: index for index, name in enumerate(names)}
    spec_by_name = {spec.part_name: spec for spec in specs}
    group_ranks = execution_group.spec.process_indices
    local_rank = assembly.local_rank
    if local_rank is None or local_rank not in range(len(group_ranks)):
        raise ValueError("Owner-local cut-cell preparation requires a valid shard rank.")
    part_count = len(group_ranks)
    dimension = assembly.part_manifest[0].ambient_dimension
    wall_sources = tuple(
        spec
        for spec in specs
        if spec.solid_query is None and wall_dimensions.get(spec.part_name, ())
    )
    wall_destinations = tuple(
        sorted(
            {
                int(rank)
                for source in wall_sources
                for rank in wall_owner_ranks.get(source.part_name, ())
            }
        )
    )
    local_cells: list[tuple[str, int, np.ndarray]] = []
    local_solid_cut: dict[tuple[str, int], bool] = {}
    local_error: Exception | None = None
    try:
        for part in assembly.parts:
            view = _owner_local_carrier(part)
            geometry = geometries[part.name]
            owner_by_id = dict(
                zip(
                    np.asarray(view.cell_ids, dtype=np.int64).tolist(),
                    np.asarray(view.cell_owner_ranks, dtype=np.int32).tolist(),
                    strict=True,
                )
            )
            for row, identifier in enumerate(
                np.asarray(geometry.cell_ids, dtype=np.int64)
            ):
                identifier_ = int(identifier)
                if owner_by_id[identifier_] != local_rank:
                    continue
                key = (part.name, identifier_)
                box = np.stack(
                    (
                        np.asarray(geometry.cell_lower[row], dtype=np.float64),
                        np.asarray(geometry.cell_upper[row], dtype=np.float64),
                    )
                )
                if box.shape != (2, dimension) or np.any(~np.isfinite(box)):
                    raise ValueError("Owned cell support AABB is malformed.")
                local_cells.append((part.name, identifier_, box))
                local_solid_cut[key] = False
                for source_spec in specs:
                    if (
                        source_spec.part_name == part.name
                        or source_spec.solid_query is None
                    ):
                        continue
                    lower, upper = _source_wall_bounds(
                        source_spec.solid_query,
                        source_spec,
                        spec_by_name[part.name],
                        dimension,
                    )
                    corners = np.asarray(
                        [
                            [
                                box[0, axis] if bit == 0 else box[1, axis]
                                for axis, bit in enumerate(bits)
                            ]
                            for bits in np.ndindex(*(2,) * dimension)
                        ],
                        dtype=np.float64,
                    )
                    source_corners = _to_source(
                        corners, spec_by_name[part.name], source_spec
                    )
                    local_solid_cut[key] |= bool(
                        np.all(np.max(source_corners, axis=0) >= lower)
                        and np.all(np.min(source_corners, axis=0) <= upper)
                    )
    except MemoryError:
        local_error = MemoryError("Owner-local cut-cell preparation failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another cell owner rejected its local support"
        )
        raise ValueError(
            "Owner-local cut-cell preparation failed collectively: " + detail
        )
    if local_error is not None:
        raise ValueError("Owner-local cut-cell preparation failed collectively.")

    outgoing_ids: list[list[tuple[int, int]]] = [[] for _ in range(part_count)]
    outgoing_boxes: list[list[np.ndarray]] = [[] for _ in range(part_count)]
    for destination in wall_destinations:
        if destination == local_rank:
            continue
        for part_name, identifier, box in local_cells:
            outgoing_ids[destination].append((part_order[part_name], identifier))
            outgoing_boxes[destination].append(box.reshape((-1,)))
    id_banks = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 2)) for rows in outgoing_ids
    )
    box_banks = tuple(
        np.asarray(rows, dtype=np.float64).reshape((-1, 2 * dimension))
        for rows in outgoing_boxes
    )
    incoming_ids = _owner_exchange_message_banks(
        id_banks,
        local_rank,
        execution_group,
        f"{axis_name}_cut_query_ids",
    )
    incoming_boxes = _owner_exchange_message_banks(
        box_banks,
        local_rank,
        execution_group,
        f"{axis_name}_cut_query_boxes",
    )
    local_error = None
    cell_queries: list[tuple[int, str, int, np.ndarray]] = [
        (local_rank, name, identifier, box) for name, identifier, box in local_cells
    ]
    try:
        if set(incoming_ids) != set(incoming_boxes):
            raise ValueError(
                "Cut-cell IDs and support AABBs have different owner routes."
            )
        for sender, rows in incoming_ids.items():
            boxes = incoming_boxes[sender]
            if boxes.shape != (rows.shape[0], 2 * dimension):
                raise ValueError("A wall owner received malformed cut-cell bounds.")
            for row, box in zip(rows, boxes, strict=True):
                target_name = names[int(row[0])]
                cell_queries.append(
                    (
                        int(sender),
                        target_name,
                        int(row[1]),
                        np.asarray(box, dtype=np.float64).reshape((2, dimension)),
                    )
                )
    except MemoryError:
        local_error = MemoryError("Owner-local cut-cell query receive failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another wall owner rejected cell bounds"
        )
        raise ValueError(
            "Owner-local cut-cell query receive failed collectively: " + detail
        )

    local_wall_cut: dict[tuple[str, int], bool] = {}
    reply_rows: list[list[tuple[int, int, int]]] = [[] for _ in range(part_count)]
    local_error = None
    try:
        record_cut = np.zeros((len(cell_queries),), dtype=np.bool_)
        for source_spec in wall_sources:
            if local_rank not in wall_owner_ranks.get(source_spec.part_name, ()):
                continue
            geometry = geometries.get(source_spec.part_name)
            if (
                geometry is None
                or geometry.wall_lower is None
                or geometry.wall_upper is None
            ):
                raise ValueError(
                    "A declared wall owner lacks its compact source facet bounds."
                )
            wall_bvh = prepare_bvh(
                np.asarray(geometry.wall_lower, dtype=np.float64),
                np.asarray(geometry.wall_upper, dtype=np.float64),
                dtype=jnp.float64,
            )
            for target_spec in specs:
                if target_spec.part_name == source_spec.part_name:
                    continue
                selected_records = np.asarray(
                    [
                        index
                        for index, (_, target_name, _, _) in enumerate(cell_queries)
                        if target_name == target_spec.part_name
                    ],
                    dtype=np.int64,
                )
                if not selected_records.size:
                    continue
                target_boxes = np.empty(
                    (selected_records.size, 2, dimension), dtype=np.float64
                )
                for slot, query_index in enumerate(selected_records):
                    box = cell_queries[int(query_index)][3]
                    corners = np.asarray(
                        [
                            [
                                box[0, axis] if bit == 0 else box[1, axis]
                                for axis, bit in enumerate(bits)
                            ]
                            for bits in np.ndindex(*(2,) * dimension)
                        ],
                        dtype=np.float64,
                    )
                    source_corners = _to_source(corners, target_spec, source_spec)
                    target_boxes[slot, 0] = np.min(source_corners, axis=0)
                    target_boxes[slot, 1] = np.max(source_corners, axis=0)
                cell_bvh = prepare_bvh(
                    target_boxes[:, 0], target_boxes[:, 1], dtype=jnp.float64
                )
                selected_cells = np.zeros((selected_records.size,), dtype=np.bool_)
                for cell_rows, _ in bvh_overlap_pair_blocks(
                    cell_bvh, wall_bvh, include_touching=True
                ):
                    selected_cells[cell_rows] = True
                record_cut[selected_records[selected_cells]] = True
        for query_index, (requester, target_name, identifier, _) in enumerate(
            cell_queries
        ):
            if requester == local_rank:
                if local_rank in wall_destinations:
                    local_wall_cut[target_name, identifier] = bool(
                        record_cut[query_index]
                    )
            else:
                reply_rows[requester].append(
                    (
                        part_order[target_name],
                        identifier,
                        int(record_cut[query_index]),
                    )
                )
    except MemoryError:
        local_error = MemoryError("Owner-local wall/cut intersection failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another wall owner failed its cut-cell query"
        )
        raise ValueError(
            "Owner-local wall/cut intersection failed collectively: " + detail
        )
    incoming_replies = _owner_exchange_message_banks(
        tuple(np.asarray(rows, dtype=np.int64).reshape((-1, 3)) for rows in reply_rows),
        local_rank,
        execution_group,
        f"{axis_name}_cut_query_replies",
    )
    wall_results: dict[tuple[str, int], dict[int, bool]] = {}
    local_error = None
    try:
        for key, is_cut in local_wall_cut.items():
            wall_results[key] = {local_rank: is_cut}
        owned_keys = {(name, identifier) for name, identifier, _ in local_cells}
        for source_rank, rows in incoming_replies.items():
            for part_index, identifier, is_cut in rows:
                key = (names[int(part_index)], int(identifier))
                if key not in owned_keys:
                    raise ValueError(
                        "A wall reply names a cell not owned by its recipient."
                    )
                results = wall_results.setdefault(key, {})
                if source_rank in results:
                    raise ValueError(
                        "A wall owner published duplicate cut-cell evidence."
                    )
                results[source_rank] = bool(is_cut)
        owned_cut: dict[tuple[str, int], bool] = {}
        for key, solid_cut in local_solid_cut.items():
            wall_values = wall_results.get(key, {})
            if set(wall_values) != set(wall_destinations):
                raise ValueError(
                    f"Cut-cell wall evidence omitted an owner for local cell {key!r}."
                )
            owned_cut[key] = bool(solid_cut or any(wall_values.values()))
    except MemoryError:
        local_error = MemoryError(
            "Owner-local cut-cell ownership reduction failed allocation."
        )
        owned_cut = {}
    except Exception as error:
        local_error = error
        owned_cut = {}
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another cell owner lacks wall-cut evidence"
        )
        raise ValueError("Owner-local cut-cell ownership failed collectively: " + detail)

    ghost_requests: list[list[tuple[int, int]]] = [[] for _ in range(part_count)]
    for part in assembly.parts:
        view = _owner_local_carrier(part)
        for identifier, owner in zip(
            np.asarray(view.cell_ids, dtype=np.int64),
            np.asarray(view.cell_owner_ranks, dtype=np.int32),
            strict=True,
        ):
            if int(owner) != local_rank:
                ghost_requests[int(owner)].append(
                    (part_order[part.name], int(identifier))
                )
    incoming_requests = _owner_exchange_message_banks(
        tuple(
            np.asarray(rows, dtype=np.int64).reshape((-1, 2)) for rows in ghost_requests
        ),
        local_rank,
        execution_group,
        f"{axis_name}_cut_ghost_requests",
    )
    ghost_replies: list[list[tuple[int, int, int]]] = [[] for _ in range(part_count)]
    local_error = None
    try:
        for requester, rows in incoming_requests.items():
            for part_index, identifier in rows:
                key = (names[int(part_index)], int(identifier))
                if key not in owned_cut:
                    raise ValueError("Cell owner lacks a requested ghost cut-cell flag.")
                ghost_replies[requester].append(
                    (int(part_index), int(identifier), int(owned_cut[key]))
                )
    except MemoryError:
        local_error = MemoryError("Owner-local ghost cut-cell reply failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another owner lacks a ghost cut-cell flag"
        )
        raise ValueError(
            "Owner-local ghost cut-cell reply failed collectively: " + detail
        )
    received_ghosts = _owner_exchange_message_banks(
        tuple(
            np.asarray(rows, dtype=np.int64).reshape((-1, 3)) for rows in ghost_replies
        ),
        local_rank,
        execution_group,
        f"{axis_name}_cut_ghost_replies",
    )
    local_cut = {key: value for key, value in owned_cut.items()}
    for rows in received_ghosts.values():
        for part_index, identifier, is_cut in rows:
            local_cut[names[int(part_index)], int(identifier)] = bool(is_cut)
    for part in assembly.parts:
        for identifier in np.asarray(_owner_local_carrier(part).cell_ids, dtype=np.int64):
            if (part.name, int(identifier)) not in local_cut:
                raise ValueError("Local cell halo lacks completed cut-cell evidence.")
    return {key for key, is_cut in local_cut.items() if is_cut}


def _owner_local_prepare_blanking(
    assembly: MeshAssembly,
    specs: tuple[OversetPartSpec, ...],
    policy: OversetPolicy,
    geometry_by_name: Mapping[str, _PartGeometry],
    holes_by_vertex: Mapping[tuple[str, int], bool],
    contacts_by_vertex: Mapping[tuple[str, int], bool],
    boundary_vertices: set[tuple[str, int]],
    protected_vertices: set[tuple[str, int]],
    excluded_cells: set[tuple[str, int]],
    cut_cells: set[tuple[str, int]],
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> tuple[
    tuple[OversetPartBlanking, ...],
    tuple[OversetReceptorEvidence, ...],
    int,
    int,
]:
    from ._overset_exchange import (
        _owner_local_all_gather_rows,
        _synchronize_owner_search_failure,
    )

    rank = assembly.local_rank
    if rank is None:
        raise ValueError("Owner-local blanking requires a rank-local assembly.")
    part_order = {name: index for index, name in enumerate(assembly.global_part_names)}
    local_data: dict[str, dict[str, Any]] = {}
    initial_frontier: list[tuple[int, int]] = []
    local_error: Exception | None = None
    try:
        for part in assembly.parts:
            view = part.carrier
            if not isinstance(view, OwnerLocalCellGeometry):
                raise TypeError("Owner-local blanking requires compact cell geometry.")
            geometry = geometry_by_name[part.name]
            vertex_ids = np.asarray(view.vertex_ids, dtype=np.int64)
            vertex_owner_ranks = np.asarray(view.vertex_owner_ranks, dtype=np.int32)
            holes_local = np.asarray(
                [
                    holes_by_vertex[(part.name, int(identifier))]
                    for identifier in vertex_ids
                ],
                dtype=np.bool_,
            )
            contacts_local = np.asarray(
                [
                    contacts_by_vertex[(part.name, int(identifier))]
                    for identifier in vertex_ids
                ],
                dtype=np.bool_,
            )
            boundary_local = np.asarray(
                [
                    (part.name, int(identifier)) in boundary_vertices
                    for identifier in vertex_ids
                ],
                dtype=np.bool_,
            )
            protected_local = np.asarray(
                [
                    (part.name, int(identifier)) in protected_vertices
                    for identifier in vertex_ids
                ],
                dtype=np.bool_,
            )
            cut_local = np.asarray(
                [
                    (part.name, int(identifier)) in cut_cells
                    for identifier in np.asarray(view.cell_ids)
                ],
                dtype=np.bool_,
            )
            cell_vertices = np.asarray(geometry.cell_vertices, dtype=np.int64)
            cell_vertex_valid = np.asarray(geometry.cell_vertex_valid, dtype=np.bool_)
            geometry_cell_ids = np.asarray(geometry.cell_ids, dtype=np.int64)
            cell_row_by_id = {
                int(identifier): row for row, identifier in enumerate(geometry_cell_ids)
            }
            if (
                cell_vertices.shape != cell_vertex_valid.shape
                or cell_vertices.shape[0] != geometry_cell_ids.size
            ):
                raise ValueError(
                    "Owner-local cell incidence does not align with its geometry rows."
                )
            view_cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
            if not np.array_equal(np.sort(view_cell_ids), np.sort(geometry_cell_ids)):
                raise ValueError("Owner-local geometry omitted source cell incidence.")
            cell_holes = np.array(cut_local, copy=True)
            for row, identifier in enumerate(view_cell_ids):
                geometry_row = cell_row_by_id[int(identifier)]
                vertices = cell_vertices[geometry_row, cell_vertex_valid[geometry_row]]
                cell_holes[row] |= bool(
                    np.any(holes_local[vertices])
                    or (part.name, int(identifier)) in excluded_cells
                )
            touched = np.zeros(vertex_ids.shape, dtype=np.bool_)
            for row, is_hole in enumerate(cell_holes):
                if not is_hole:
                    continue
                identifier = int(view_cell_ids[row])
                geometry_row = cell_row_by_id[identifier]
                vertices = cell_vertices[geometry_row, cell_vertex_valid[geometry_row]]
                touched[vertices] = True
            frontier = (touched | boundary_local) & ~holes_local
            for vertex_row in np.flatnonzero(frontier):
                initial_frontier.append(
                    (part_order[part.name], int(vertex_ids[vertex_row]))
                )
            local_data[part.name] = {
                "view": view,
                "vertex_ids": vertex_ids,
                "vertex_owner_ranks": vertex_owner_ranks,
                "holes": holes_local,
                "contacts": contacts_local,
                "protected": protected_local,
                "cell_ids": view_cell_ids,
                "cell_holes": cell_holes,
                "cell_vertices": cell_vertices,
                "cell_vertex_valid": cell_vertex_valid,
                "cell_row_by_id": cell_row_by_id,
            }
    except MemoryError:
        local_error = MemoryError("Owner-local blanking preparation failed allocation.")
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(
        local_error is not None, execution_group.devices, axis_name
    ):
        detail = (
            str(local_error)
            if local_error is not None
            else "another rank rejected its local topology"
        )
        raise ValueError(
            "Owner-local blanking preparation failed collectively: " + detail
        )
    if local_error is not None:
        raise ValueError("Owner-local blanking preparation failed collectively.")
    local_frontier_rows = np.asarray(initial_frontier, dtype=np.int64).reshape((-1, 2))
    frontier_chunks = _owner_local_all_gather_rows(
        local_frontier_rows, execution_group.devices, axis_name
    )
    frontier = {
        (assembly.global_part_names[int(row[0])], int(row[1]))
        for chunk in frontier_chunks
        for row in chunk
    }
    layers_by_vertex = {key: 1 for key in frontier}
    for depth in range(2, policy.fringe_layers + 1):
        local_grown: list[tuple[int, int]] = []
        local_error = None
        try:
            for name, record in local_data.items():
                view = record["view"]
                vertex_ids = record["vertex_ids"]
                holes_local = record["holes"]
                cell_ids = record["cell_ids"]
                cell_holes = record["cell_holes"]
                cell_vertices = record["cell_vertices"]
                vertex_valid = record["cell_vertex_valid"]
                cell_row_by_id = record["cell_row_by_id"]
                cell_owners = dict(
                    zip(
                        np.asarray(view.cell_ids, dtype=np.int64).tolist(),
                        np.asarray(view.cell_owner_ranks, dtype=np.int32).tolist(),
                        strict=True,
                    )
                )
                for cell_row, identifier in enumerate(cell_ids):
                    if cell_owners[int(identifier)] != rank or cell_holes[cell_row]:
                        continue
                    geometry_row = cell_row_by_id[int(identifier)]
                    rows = cell_vertices[geometry_row, vertex_valid[geometry_row]]
                    if not any((name, int(vertex_ids[row])) in frontier for row in rows):
                        continue
                    for row in rows:
                        key = (name, int(vertex_ids[row]))
                        if not holes_local[row] and key not in layers_by_vertex:
                            local_grown.append((part_order[name], key[1]))
        except MemoryError:
            local_error = MemoryError("Owner-local fringe expansion failed allocation.")
        except Exception as error:
            local_error = error
        if _synchronize_owner_search_failure(
            local_error is not None, execution_group.devices, axis_name
        ):
            detail = (
                str(local_error)
                if local_error is not None
                else "another rank rejected fringe closure"
            )
            raise ValueError(
                "Owner-local fringe expansion failed collectively: " + detail
            )
        grown_chunks = _owner_local_all_gather_rows(
            np.asarray(local_grown, dtype=np.int64).reshape((-1, 2)),
            execution_group.devices,
            axis_name,
        )
        grown = {
            (assembly.global_part_names[int(row[0])], int(row[1]))
            for chunk in grown_chunks
            for row in chunk
        }
        if not grown:
            break
        for key in grown:
            layers_by_vertex[key] = depth
        frontier = grown
    local_blanking: list[OversetPartBlanking] = []
    local_receptor_rows: list[tuple[int, int, int, int]] = []
    local_conflict_count = 0
    local_ambiguous_count = 0
    for part in assembly.parts:
        name = part.name
        record = local_data[name]
        view = record["view"]
        vertex_ids = record["vertex_ids"]
        vertex_owner_ranks = record["vertex_owner_ranks"]
        holes_local = record["holes"]
        contacts_local = record["contacts"]
        protected_local = record["protected"]
        layers = np.asarray(
            [
                layers_by_vertex.get((name, int(identifier)), 0)
                for identifier in vertex_ids
            ],
            dtype=np.int32,
        )
        layers[protected_local & ~holes_local] = 0
        vertex_status = np.select(
            (holes_local, layers > 0),
            (OversetVertexStatus.HOLE, OversetVertexStatus.RECEPTOR),
            OversetVertexStatus.ACTIVE,
        ).astype(np.int8)
        cell_ids = record["cell_ids"]
        cell_holes = record["cell_holes"]
        cell_vertices = record["cell_vertices"]
        vertex_valid = record["cell_vertex_valid"]
        cell_row_by_id = record["cell_row_by_id"]
        cell_status = np.full(
            cell_ids.shape, int(OversetCellStatus.ACTIVE), dtype=np.int8
        )
        ambiguous = np.zeros(cell_ids.shape, dtype=np.bool_)
        for cell_row, identifier in enumerate(cell_ids):
            geometry_row = cell_row_by_id[int(identifier)]
            rows = cell_vertices[geometry_row, vertex_valid[geometry_row]]
            enclosing = bool(np.any(holes_local[rows]))
            cell_status[cell_row] = (
                int(OversetCellStatus.HOLE)
                if cell_holes[cell_row]
                else int(OversetCellStatus.FRINGE)
                if np.any(layers[rows] > 0)
                else int(OversetCellStatus.ACTIVE)
            )
            ambiguous[cell_row] = (name, int(identifier)) in cut_cells and not enclosing
        status_by_cell_id = {
            int(identifier): int(status)
            for identifier, status in zip(cell_ids, cell_status, strict=True)
        }
        ambiguous_by_cell_id = {
            int(identifier): bool(status)
            for identifier, status in zip(cell_ids, ambiguous, strict=True)
        }
        view_cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
        cell_status_view = np.asarray(
            [status_by_cell_id[int(identifier)] for identifier in view_cell_ids],
            dtype=np.int8,
        )
        ambiguous_view = np.asarray(
            [ambiguous_by_cell_id[int(identifier)] for identifier in view_cell_ids],
            dtype=np.bool_,
        )
        conflict = protected_local & (vertex_status != int(OversetVertexStatus.ACTIVE))
        local_conflict_count += int(
            np.count_nonzero(conflict & (vertex_owner_ranks == rank))
        )
        cell_owner_ranks = np.asarray(view.cell_owner_ranks, dtype=np.int32)
        local_ambiguous_count += int(
            np.count_nonzero(ambiguous_view & (cell_owner_ranks == rank))
        )
        local_blanking.append(
            OversetPartBlanking(
                part,
                vertex_status,
                layers,
                contacts_local,
                conflict,
                cell_status_view,
                ambiguous_view,
            )
        )
        part_index = part_order[name]
        for row in np.flatnonzero(
            (vertex_status == int(OversetVertexStatus.RECEPTOR))
            & (vertex_owner_ranks == rank)
        ):
            local_receptor_rows.append(
                (part_index, int(vertex_ids[row]), int(rank), int(row))
            )
    local_receptor_array = np.asarray(local_receptor_rows, dtype=np.int64).reshape(
        (-1, 4)
    )
    receptor_chunks = _owner_local_all_gather_rows(
        local_receptor_array, execution_group.devices, axis_name
    )
    receptor_records = [
        tuple(int(value) for value in row) for chunk in receptor_chunks for row in chunk
    ]
    receptor_keys = [(row[0], row[1]) for row in receptor_records]
    if len(set(receptor_keys)) != len(receptor_keys):
        raise ValueError(
            "Owner-local receptor directory published duplicate point owners."
        )
    receptor_records.sort(key=lambda row: (row[0], row[1]))
    receptor_evidence = []
    for part_index, name in enumerate(assembly.global_part_names):
        selected = [row for row in receptor_records if row[0] == part_index]
        ids = np.asarray([row[1] for row in selected], dtype=np.int64)
        rows = np.asarray([row[3] for row in selected], dtype=np.int32)
        owners = np.asarray([row[2] for row in selected], dtype=np.int32)
        count = ids.size
        receptor_evidence.append(
            OversetReceptorEvidence(
                name,
                ids,
                rows,
                np.full((count,), int(CouplingSearchStatus.OUTSIDE), dtype=np.int32),
                np.full((count,), -1, dtype=np.int32),
                np.full((count,), -1, dtype=np.int64),
                np.zeros((count,), dtype=np.int32),
                np.full((count,), np.inf, dtype=np.float64),
                np.full((count,), np.inf, dtype=np.float64),
                owners,
                donor_refusals=(),
            )
        )
    counts = _owner_local_all_gather_rows(
        np.asarray(((local_conflict_count, local_ambiguous_count),), dtype=np.int64),
        execution_group.devices,
        axis_name,
    )
    global_conflicts = sum(int(rows[0, 0]) for rows in counts)
    global_ambiguous = sum(int(rows[0, 1]) for rows in counts)
    return (
        tuple(local_blanking),
        tuple(receptor_evidence),
        global_conflicts,
        global_ambiguous,
    )


def _geometry(part: MeshPart, spec: OversetPartSpec, /) -> _PartGeometry:
    result = _cell_result(part)
    mesh = result.mesh
    if (
        mesh.topological_dimension != mesh.ambient_dimension
        or mesh.ambient_dimension not in (2, 3)
    ):
        raise ValueError("Native overset requires full-dimensional 2-D or 3-D cell maps.")
    if not mesh.blocks:
        raise ValueError("Native overset requires an explicit cell block.")
    geometry = result.geometry
    rows: list[tuple[int, ...]] = []
    identifiers: list[np.ndarray] = []
    measures: list[np.ndarray] = []
    lowers: list[np.ndarray] = []
    uppers: list[np.ndarray] = []
    locators: list[AbstractCellLocator] = []
    block_names: list[str] = []
    block_offsets = [0]
    polyhedral_map: PreparedPolyhedralCellMap | None = None
    connectivity = mesh.connectivity
    if isinstance(connectivity, PolyhedralConnectivity):
        if len(mesh.blocks) != 1:
            raise ValueError(
                "Native overset polyhedral parts require one canonical polyhedral block."
            )
        locator = PreparedPolyhedralCellLocator(mesh, geometry=geometry)
        polyhedral_map = locator.cell_map
        global_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        block_ids = np.asarray(mesh.blocks[0].global_ids, dtype=np.int64)
        if not np.array_equal(global_ids, block_ids):
            raise ValueError(
                "Polyhedral cell identities differ between block and incidence order."
            )
        rows.extend(polyhedral_map.cell_vertices)
        identifiers.append(global_ids)
        measures.append(_polyhedral_cell_measures(locator))
        lowers.append(np.asarray(polyhedral_map.cell_lower))
        uppers.append(np.asarray(polyhedral_map.cell_upper))
        locators.append(locator)
        block_names.append(mesh.blocks[0].name)
        block_offsets.append(global_ids.size)
    else:
        field = FiniteElementFieldSpec(
            "overset-location",
            {
                block.name: discontinuous_element(block.cell_kind, 0)
                for block in mesh.blocks
            },
        )
        discretization = FiniteElementPlan(
            mesh, field, coordinate_spec=geometry
        ).prepare()
        coordinates = discretization.default_runtime.coordinates
        for block_index, block in enumerate(mesh.blocks):
            cell_map = PreparedMappedCellMap(mesh, geometry, block_index)
            locator = PreparedMappedCellLocator(
                cell_map, coordinates, source_geometry=geometry
            )
            block_measure = np.asarray(
                discretization.block_geometries[0][block_index].measure,
                dtype=np.float64,
            )
            if block_measure.shape != (block.cell_count,) or np.any(
                ~np.isfinite(block_measure) | (block_measure <= 0)
            ):
                raise ValueError(
                    "Overset mapped cells require positive finite physical measures."
                )
            rows.extend(
                tuple(int(value) for value in row) for row in np.asarray(block.vertices)
            )
            identifiers.append(np.asarray(block.global_ids, dtype=np.int64))
            measures.append(block_measure)
            lowers.append(np.asarray(locator.cell_lower))
            uppers.append(np.asarray(locator.cell_upper))
            locators.append(locator)
            block_names.append(block.name)
            block_offsets.append(block_offsets[-1] + block.cell_count)
    cell_ids = np.concatenate(identifiers)
    cell_measures = np.concatenate(measures)
    cell_lower = np.concatenate(lowers)
    cell_upper = np.concatenate(uppers)
    cell_vertices, cell_vertex_valid = _pack_cell_vertices(tuple(rows))
    if (
        cell_ids.shape != (len(rows),)
        or cell_measures.shape != cell_ids.shape
        or cell_lower.shape != (cell_ids.size, mesh.ambient_dimension)
        or cell_upper.shape != cell_lower.shape
    ):
        raise ValueError(
            "Overset cell geometry banks do not align with source identities."
        )
    wall_facets = wall_lower = wall_upper = None
    if spec.wall is not None:
        if isinstance(connectivity, PolyhedralConnectivity):
            wall_rows = _facet_rows(part, spec.wall)
            wall_simplex_rows = _facet_triangles(
                part, spec.wall, polyhedral_map=polyhedral_map
            )
            if spec.solid_query is None:
                _require_closed_wall(wall_simplex_rows, mesh.ambient_dimension)
            owners = np.asarray(connectivity.face_owner, dtype=np.int64)[wall_rows]
        else:
            wall_rows = _facet_rows(part, spec.wall)
            elements = geometry.resolve(mesh)[0]
            exact_element_map = all(
                isinstance(element, FiniteElementSpec) and element.degree == 1
                for element in elements
            )
            exact_wall_map = (
                exact_element_map
                if mesh.ambient_dimension == 2
                else all(block.cell_kind == "tetrahedron" for block in mesh.blocks)
                and exact_element_map
            )
            if spec.solid_query is None and not exact_wall_map:
                raise ValueError(
                    "Mapped overset walls require a source solid query; corner facets "
                    "cannot stand in for a curved source boundary."
                )
            if spec.solid_query is None:
                _require_closed_wall(wall_rows, mesh.ambient_dimension)
            wall_simplex_rows = (
                _facet_triangles(part, spec.wall) if exact_wall_map else None
            )
            exterior = discretization.exterior_facet_domain
            exterior_rows = np.asarray(exterior.entity_indices, dtype=np.int64)
            facet_ids = np.asarray(
                mesh.entity_set(mesh.topological_dimension - 1).entity_ids,
                dtype=np.int64,
            )
            facet_order = np.argsort(facet_ids, kind="stable")
            wall_indices = facet_order[
                np.searchsorted(
                    facet_ids[facet_order],
                    np.asarray(spec.wall.entity_ids, dtype=np.int64),
                )
            ]
            positions = np.searchsorted(exterior_rows, wall_indices)
            if np.any(positions >= exterior_rows.size) or not np.array_equal(
                exterior_rows[positions], wall_indices
            ):
                raise ValueError(
                    "Overset wall facets lack canonical exterior owner routes."
                )
            owners = np.asarray(exterior.owner_cells, dtype=np.int64)[positions]
        if wall_simplex_rows is not None:
            wall_facets = np.asarray(mesh.coordinates, dtype=np.float64)[
                wall_simplex_rows
            ]
        if owners.size:
            wall_lower = cell_lower[owners]
            wall_upper = cell_upper[owners]
    return _PartGeometry(
        part_id=part.part_id,
        cell_vertices=jnp.asarray(cell_vertices),
        cell_vertex_valid=jnp.asarray(cell_vertex_valid),
        cell_ids=jnp.asarray(cell_ids),
        cell_measures=jnp.asarray(cell_measures),
        cell_lower=jnp.asarray(cell_lower),
        cell_upper=jnp.asarray(cell_upper),
        cell_locators=tuple(locators),
        block_offsets=tuple(block_offsets),
        block_names=tuple(block_names),
        wall_facets=None if wall_facets is None else jnp.asarray(wall_facets),
        wall_lower=None if wall_lower is None else jnp.asarray(wall_lower),
        wall_upper=None if wall_upper is None else jnp.asarray(wall_upper),
    )


def _classify_facets(
    points: np.ndarray, facets: np.ndarray, dimension: int, /
) -> tuple[np.ndarray, np.ndarray]:
    if facets.ndim != 3 or facets.shape[2] != dimension:
        raise ValueError("Overset wall facets have inconsistent physical dimensions.")
    lower, upper = np.min(facets, axis=(0, 1)), np.max(facets, axis=(0, 1))
    candidate = np.flatnonzero(np.all((points >= lower) & (points <= upper), axis=1))
    inside = np.zeros(points.shape[0], dtype=np.bool_)
    touching = np.zeros(points.shape[0], dtype=np.bool_)
    if candidate.size == 0:
        return inside, touching
    query = points[candidate]
    ray_upper = np.array(query, copy=True)
    ray_upper[:, 0] = upper[0]
    ray_bvh = prepare_bvh(query, ray_upper, dtype=jnp.float64)
    wall_bvh = prepare_bvh(
        np.min(facets, axis=1), np.max(facets, axis=1), dtype=jnp.float64
    )
    counts = np.zeros(candidate.size, dtype=np.int64)
    classify = _segment_crossings if dimension == 2 else _triangle_crossings
    for rows, items in bvh_overlap_pair_blocks(ray_bvh, wall_bvh, include_touching=True):
        crossing, contact = classify(query[rows], facets[items])
        counts += np.bincount(rows[crossing], minlength=candidate.size)
        touching[candidate] |= np.bincount(rows[contact], minlength=candidate.size) > 0
    inside[candidate] = counts % 2 == 1
    return inside, touching


def _segment_intersects_triangle(segment: np.ndarray, triangle: np.ndarray, /) -> bool:
    """Intersect closed affine segments and triangles with exact binary64 predicates."""
    points = [
        (Fraction.from_float(float(point[0])), Fraction.from_float(float(point[1])))
        for point in np.concatenate((segment, triangle), axis=0)
    ]
    start, end, a, b, c = points
    orientation = cross(a, b, c)
    if orientation == 0:
        raise ValueError("Overset affine triangle must be nondegenerate.")
    lower, upper = Fraction(0), Fraction(1)
    for first, second in ((a, b), (b, c), (c, a)):
        first_value = cross(first, second, start) * orientation
        last_value = cross(first, second, end) * orientation
        if first_value < 0 and last_value < 0:
            return False
        if first_value < 0:
            lower = max(lower, -first_value / (last_value - first_value))
        elif last_value < 0:
            upper = min(upper, first_value / (first_value - last_value))
        if lower > upper:
            return False
    return True


def _cut_cell_candidates(
    geometry: _PartGeometry,
    facets_in_part_frame: np.ndarray,
    triangles: np.ndarray | None = None,
    /,
) -> np.ndarray:
    if facets_in_part_frame.size == 0:
        return np.zeros(geometry.cell_ids.shape, dtype=np.bool_)
    cells = prepare_bvh(geometry.cell_lower, geometry.cell_upper, dtype=jnp.float64)
    facets = prepare_bvh(
        np.min(facets_in_part_frame, axis=1),
        np.max(facets_in_part_frame, axis=1),
        dtype=jnp.float64,
    )
    cut = np.zeros(np.asarray(geometry.cell_ids).shape, dtype=np.bool_)
    for rows, items in bvh_overlap_pair_blocks(cells, facets, include_touching=True):
        if triangles is None:
            cut[rows] = True
        else:
            for row, item in zip(rows, items, strict=True):
                if not cut[row] and _segment_intersects_triangle(
                    facets_in_part_frame[item], triangles[row]
                ):
                    cut[row] = True
    return cut


def _cut_cell_bounds(
    geometry: _PartGeometry, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray:
    cells = prepare_bvh(geometry.cell_lower, geometry.cell_upper, dtype=jnp.float64)
    wall = prepare_bvh(lower[None, :], upper[None, :], dtype=jnp.float64)
    cut = np.zeros(np.asarray(geometry.cell_ids).shape, dtype=np.bool_)
    for rows, _ in bvh_overlap_pair_blocks(cells, wall, include_touching=True):
        cut[rows] = True
    return cut


def _source_wall_bounds(
    query: Any,
    source_spec: OversetPartSpec,
    target_spec: OversetPartSpec,
    dimension: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.asarray(getattr(query, "bounds", None), dtype=np.float64)
    if bounds.shape != (2, dimension) or not np.all(np.isfinite(bounds)):
        raise ValueError(
            "Source solid query must provide a finite source-frame (2, dimension) "
            "whole-domain bound for conservative cut-cell discovery."
        )
    corners = np.asarray(
        [
            [bounds[bit, axis] for axis, bit in enumerate(bits)]
            for bits in np.ndindex(*(2,) * dimension)
        ],
        dtype=np.float64,
    )
    target_points = _to_target(corners, source_spec, target_spec)
    return np.min(target_points, axis=0), np.max(target_points, axis=0)


def _blank_part(
    index: int,
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    policy: OversetPolicy,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    part, spec = parts[index], specs[index]
    mesh = _cell_result(part).mesh
    points = np.asarray(mesh.coordinates, dtype=np.float64)
    cells = np.asarray(geometries[index].cell_vertices, dtype=np.int64)
    vertex_valid = np.asarray(geometries[index].cell_vertex_valid, dtype=np.bool_)
    safe_cells = np.where(vertex_valid, cells, 0)
    holes = np.zeros(points.shape[0], dtype=np.bool_)
    contact = np.zeros(points.shape[0], dtype=np.bool_)
    cut = np.zeros(cells.shape[0], dtype=np.bool_)
    result = _cell_result(part)
    elements, routes, _ = result.geometry.resolve(mesh)
    affine_triangles = (
        mesh.ambient_dimension == 2
        and result.geometry.source_map is None
        and all(
            block.cell_kind == "triangle"
            and isinstance(element, FiniteElementSpec)
            and element.degree == 1
            and np.array_equal(np.asarray(route), np.asarray(block.vertices))
            for block, element, route in zip(mesh.blocks, elements, routes, strict=True)
        )
    )
    triangles = points[cells] if affine_triangles else None
    for other, wall_geometry in enumerate(geometries):
        source_spec = specs[other]
        if other == index or source_spec.wall is None:
            continue
        if source_spec.solid_query is not None:
            query_points = _to_source(points, spec, source_spec)
            inside, touching = _source_solid_membership(
                source_spec.solid_query,
                query_points,
                source_spec.solid_index,
                np.asarray(mesh.vertex_global_ids),
            )
        else:
            if wall_geometry.wall_facets is None:
                raise ValueError(
                    "Overset wall membership requires its exact facets or an authoritative solid query."
                )
            wall = np.asarray(wall_geometry.wall_facets)
            source_frame = _to_source(points, spec, source_spec)
            inside, touching = _classify_facets(
                source_frame, wall, mesh.ambient_dimension
            )
        holes |= inside | touching
        contact |= touching
        if source_spec.solid_query is not None:
            lower, upper = _source_wall_bounds(
                source_spec.solid_query,
                source_spec,
                spec,
                mesh.ambient_dimension,
            )
            cut |= _cut_cell_bounds(geometries[index], lower, upper)
        elif wall_geometry.wall_facets is not None:
            wall_in_target = _to_target(
                np.asarray(wall_geometry.wall_facets).reshape(
                    (-1, mesh.ambient_dimension)
                ),
                source_spec,
                spec,
            ).reshape(np.asarray(wall_geometry.wall_facets).shape)
            cut |= _cut_cell_candidates(geometries[index], wall_in_target, triangles)
    if spec.excluded is not None:
        part.require_scope(spec.excluded)
        if spec.excluded.entity_dimension != mesh.topological_dimension:
            raise ValueError("Overset excluded scope must identify cells.")
        cut |= np.isin(
            np.asarray(geometries[index].cell_ids),
            np.asarray(spec.excluded.entity_ids),
        )
    enclosing = np.any(holes[safe_cells] & vertex_valid, axis=1)
    cell_holes = enclosing | cut
    boundary = np.zeros(points.shape[0], dtype=np.bool_)
    if spec.boundary is not None:
        boundary[_facet_vertex_ids(part, spec.boundary)] = True
    protected = np.zeros(points.shape[0], dtype=np.bool_)
    if spec.wall is not None:
        protected[_facet_vertex_ids(part, spec.wall)] = True
    touched = np.zeros(points.shape[0], dtype=np.bool_)
    selected_vertices = safe_cells[cell_holes]
    touched[selected_vertices[vertex_valid[cell_holes]]] = True
    frontier = (touched | boundary) & ~holes
    layers = np.zeros(points.shape[0], dtype=np.int32)
    layers[frontier] = 1
    for depth in range(2, policy.fringe_layers + 1):
        active_cells = np.flatnonzero(
            ~cell_holes & np.any(frontier[safe_cells] & vertex_valid, axis=1)
        )
        grown = np.zeros(points.shape[0], dtype=np.bool_)
        selected_vertices = safe_cells[active_cells]
        grown[selected_vertices[vertex_valid[active_cells]]] = True
        frontier = grown & ~holes & (layers == 0)
        layers[frontier] = depth
    layers[protected & ~holes] = 0
    vertex_status = np.select(
        (holes, layers > 0),
        (OversetVertexStatus.HOLE, OversetVertexStatus.RECEPTOR),
        OversetVertexStatus.ACTIVE,
    ).astype(np.int8)
    cell_status = np.select(
        (
            cell_holes,
            np.any((layers[safe_cells] > 0) & vertex_valid, axis=1),
        ),
        (OversetCellStatus.HOLE, OversetCellStatus.FRINGE),
        OversetCellStatus.ACTIVE,
    ).astype(np.int8)
    return (
        vertex_status,
        layers,
        contact,
        protected & (vertex_status != OversetVertexStatus.ACTIVE),
        cell_status,
        cut & ~enclosing,
    )


class _DonorChoice(StrictModule, NonTrainableState):
    parts: np.ndarray
    cells: np.ndarray
    blocks: np.ndarray
    ids: np.ndarray
    weights: np.ndarray
    explicit: np.ndarray
    references: np.ndarray
    measures: np.ndarray
    status: np.ndarray
    candidate_parts: np.ndarray
    residuals: np.ndarray
    refusals: tuple[OversetDonorRefusal, ...]


def _resolve_donors(
    points: np.ndarray,
    receptor_index: int,
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    blanking: tuple[OversetPartBlanking, ...],
    policy: OversetPolicy,
    /,
) -> _DonorChoice:
    sites = np.asarray(points, dtype=np.float64)
    count, dimension = sites.shape
    choices: list[list[tuple[Any, ...]]] = [[] for _ in range(count)]
    candidate_parts = [set() for _ in range(count)]
    blocked = np.zeros((count,), dtype=np.bool_)
    unresolved = np.zeros((count,), dtype=np.bool_)
    refusals: list[tuple[int, int, int, OversetDonorRefusal]] = []
    source_vertices = [
        np.asarray(_cell_result(part).mesh.coordinates, dtype=np.float64)
        for part in parts
    ]
    source_vertex_ids = [
        np.asarray(_cell_result(part).mesh.vertex_global_ids, dtype=np.int64)
        for part in parts
    ]
    explicit_blocks = []
    for part in parts:
        result = _cell_result(part)
        elements, routes, _ = result.geometry.resolve(result.mesh)
        explicit_blocks.append(
            tuple(
                block.cell_kind in ("triangle", "tetrahedron")
                and isinstance(element, FiniteElementSpec)
                and element.degree == 1
                and result.geometry.source_map is None
                and np.array_equal(np.asarray(route), np.asarray(block.vertices))
                for block, element, route in zip(
                    result.mesh.blocks, elements, routes, strict=True
                )
            )
        )
    for donor, spec in enumerate(specs):
        if donor == receptor_index:
            continue
        geometry = geometries[donor]
        donor_points = _to_source(sites, specs[receptor_index], spec)
        active_cells = (
            np.asarray(blanking[donor].cell_status) == int(OversetCellStatus.ACTIVE)
        ) & ~np.asarray(blanking[donor].ambiguous_cells, dtype=np.bool_)
        for block, locator in enumerate(geometry.cell_locators):
            start, _ = geometry.block_offsets[block : block + 2]
            location = locator.locate(donor_points)
            refusals.extend(
                (
                    refusal.point,
                    donor,
                    block,
                    OversetDonorRefusal(
                        spec.part_name, geometry.block_names[block], refusal
                    ),
                )
                for refusal in location.inverse_refusals
            )
            complete = np.asarray(location.candidates_complete, dtype=np.bool_)
            status = np.asarray(location.status, dtype=np.int32)
            unresolved |= ~complete | ~np.isin(
                status,
                (int(CellLocationStatus.LOCATED), int(CellLocationStatus.OUTSIDE)),
            )
            candidate_cells = np.asarray(location.candidate_cells, dtype=np.int64)
            candidate_references = np.asarray(
                location.candidate_reference, dtype=np.float64
            )
            for receptor in range(count):
                for slot, local_cell in enumerate(candidate_cells[receptor]):
                    if local_cell < 0:
                        continue
                    cell = start + int(local_cell)
                    if not active_cells[cell]:
                        blocked[receptor] = True
                        continue
                    candidate_parts[receptor].add(donor)
                    identifier = int(np.asarray(geometry.cell_ids)[cell])
                    measure = float(np.asarray(geometry.cell_measures)[cell])
                    reference = candidate_references[receptor, slot]
                    has_stencil = explicit_blocks[donor][block]
                    ids = np.full((dimension + 1,), -1, dtype=np.int64)
                    weights = np.zeros((dimension + 1,), dtype=np.float64)
                    if has_stencil:
                        vertices = np.asarray(geometry.cell_vertices[cell])[
                            np.asarray(geometry.cell_vertex_valid[cell], dtype=np.bool_)
                        ]
                        weights[1:] = reference
                        weights[0] = 1.0 - float(np.sum(reference))
                        weights = np.maximum(weights, 0.0)
                        total = float(np.sum(weights))
                        if total <= 0.0 or not np.isfinite(total):
                            unresolved[receptor] = True
                            continue
                        weights /= total
                        ids[:] = source_vertex_ids[donor][vertices]
                        residual = float(
                            np.linalg.norm(
                                weights @ source_vertices[donor][vertices]
                                - donor_points[receptor]
                            )
                        )
                    elif isinstance(locator, PreparedMappedCellLocator):
                        source_result = _cell_result(parts[donor])
                        physical = source_result.geometry.evaluate(
                            source_result.mesh,
                            np.asarray((identifier,), dtype=np.int64),
                            reference[None, :],
                        )[0, 0]
                        residual = float(
                            np.linalg.norm(physical - donor_points[receptor])
                        )
                    else:
                        residual = 0.0
                    match policy.donor_priority:
                        case "finest-donor":
                            part_priority: float | int = measure
                            size_priority: float | int = -spec.priority
                        case "part-priority":
                            # Higher declared priority donates first, as in the
                            # owner-local search and the finest-donor tiebreak.
                            part_priority = -float(spec.priority)
                            size_priority = measure
                        case other:
                            assert_never(other)
                    key = (part_priority, size_priority, spec.part_name, identifier)
                    choices[receptor].append(
                        (
                            key,
                            donor,
                            cell,
                            block,
                            ids,
                            weights,
                            reference,
                            measure,
                            residual,
                            has_stencil,
                        )
                    )
    chosen_part = np.full((count,), -1, dtype=np.int32)
    chosen_cell = np.full((count,), -1, dtype=np.int64)
    chosen_block = np.full((count,), -1, dtype=np.int32)
    chosen_ids = np.full((count, dimension + 1), -1, dtype=np.int64)
    chosen_weights = np.zeros((count, dimension + 1), dtype=np.float64)
    explicit = np.zeros((count,), dtype=np.bool_)
    references = np.zeros((count, dimension), dtype=np.float64)
    measures = np.full((count,), np.inf, dtype=np.float64)
    residuals = np.full((count,), np.inf, dtype=np.float64)
    candidate_counts = np.asarray(
        [len(values) for values in candidate_parts], dtype=np.int32
    )
    status = np.full((count,), int(CouplingSearchStatus.OUTSIDE), dtype=np.int32)
    for receptor in range(count):
        rows = choices[receptor]
        if unresolved[receptor]:
            status[receptor] = int(CouplingSearchStatus.UNRESOLVED)
        elif rows:
            (
                _,
                donor,
                cell,
                block,
                ids,
                weights,
                reference,
                measure,
                residual,
                has_stencil,
            ) = min(rows, key=lambda item: item[0])
            chosen_part[receptor], chosen_cell[receptor], chosen_block[receptor] = (
                donor,
                cell,
                block,
            )
            chosen_ids[receptor], chosen_weights[receptor] = ids, weights
            explicit[receptor] = has_stencil
            references[receptor] = reference
            measures[receptor], residuals[receptor] = measure, residual
            status[receptor] = int(CouplingSearchStatus.FOUND)
        elif blocked[receptor]:
            status[receptor] = int(CouplingSearchStatus.EXCLUDED_DONOR)
    return _DonorChoice(
        chosen_part,
        chosen_cell,
        chosen_block,
        chosen_ids,
        chosen_weights,
        explicit,
        references,
        measures,
        status,
        candidate_counts,
        residuals,
        tuple(record for *_, record in sorted(refusals, key=lambda item: item[:3])),
    )


def _make_blanking(
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    policy: OversetPolicy,
    /,
) -> tuple[OversetPartBlanking, ...]:
    return tuple(
        OversetPartBlanking(part, *_blank_part(index, parts, specs, geometries, policy))
        for index, part in enumerate(parts)
    )


def _receptor_records(
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    geometries: tuple[_PartGeometry, ...],
    blanking: tuple[OversetPartBlanking, ...],
    policy: OversetPolicy,
    /,
) -> tuple[
    tuple[OversetReceptorEvidence, ...],
    tuple[OversetDonorPacket, ...],
    tuple[OversetCoupling, ...],
]:
    evidence_records, packets, overlays = [], [], []
    for target_index, target in enumerate(parts):
        mesh = _cell_result(target).mesh
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        rows = np.flatnonzero(
            np.asarray(blanking[target_index].vertex_status)
            == OversetVertexStatus.RECEPTOR
        )
        rows = rows[np.argsort(vertex_ids[rows], kind="stable")]
        target_points = np.asarray(mesh.coordinates, dtype=np.float64)[rows]
        choices = _resolve_donors(
            target_points,
            target_index,
            parts,
            specs,
            geometries,
            blanking,
            policy,
        )
        donor_cell_ids = np.full(rows.size, -1, dtype=np.int64)
        for slot, (donor_index, row) in enumerate(
            zip(choices.parts, choices.cells, strict=True)
        ):
            if donor_index >= 0:
                donor_cell_ids[slot] = int(
                    np.asarray(geometries[donor_index].cell_ids)[row]
                )
        receptor_owner_ranks = np.full(
            rows.size, specs[target_index].owner_rank, dtype=np.int32
        )
        evidence_records.append(
            OversetReceptorEvidence(
                target.name,
                vertex_ids[rows],
                rows,
                choices.status,
                choices.parts,
                donor_cell_ids,
                choices.candidate_parts,
                choices.residuals,
                choices.measures,
                receptor_owner_ranks,
                donor_refusals=choices.refusals,
            )
        )
        for donor_index, donor in enumerate(parts):
            geometry = geometries[donor_index]
            for block_index, block_name in enumerate(geometry.block_names):
                slots = np.flatnonzero(
                    (choices.parts == donor_index) & (choices.blocks == block_index)
                )
                if slots.size == 0:
                    continue
                slots = slots[np.argsort(vertex_ids[rows[slots]], kind="stable")]
                target_site = target_points[slots]
                query_site = _to_source(
                    target_site, specs[target_index], specs[donor_index]
                )
                start = geometry.block_offsets[block_index]
                packet = OversetDonorPacket(
                    donor,
                    target,
                    specs[donor_index],
                    specs[target_index],
                    vertex_ids[rows[slots]],
                    slots,
                    query_site,
                    target_site,
                    donor_cell_ids[slots],
                    choices.cells[slots] - start,
                    choices.references[slots],
                    donor_block=block_name,
                )
                packets.append(packet)
                if not np.all(choices.explicit[slots]):
                    continue
                stencil_ids = choices.ids[slots]
                stencil_weights = choices.weights[slots]
                source_ids = np.unique(stencil_ids[stencil_ids >= 0])
                donor_scope = donor.scope(0, source_ids)
                receptor_scope = target.scope(0, vertex_ids[rows[slots]])
                hole_ids = blanking[target_index].vertex_ids_with(
                    OversetVertexStatus.HOLE
                )
                hole_scope = target.scope(0, hole_ids) if hole_ids.size else None
                search = CouplingSearchEvidence(
                    MeshCouplingKind.OVERSET,
                    "native-overset-affine-simplex-stencil",
                    donor_scope,
                    receptor_scope,
                    np.full(slots.size, int(CouplingSearchStatus.FOUND), dtype=np.int32),
                    stencil_ids,
                    choices.cells[slots],
                    choices.residuals[slots],
                )
                overlays.append(
                    OversetCoupling(
                        donor,
                        target,
                        donor_scope,
                        receptor_scope,
                        stencil_ids,
                        stencil_weights,
                        hole_scope=hole_scope,
                        search_evidence=search,
                    )
                )
    return tuple(evidence_records), tuple(packets), tuple(overlays)


def _partition_digest(partition: OversetPartition) -> str:
    return canonical_fingerprint(
        {
            "kind": "overset-exact-partition",
            "pieces": [
                (
                    p.part_name,
                    p.part_index,
                    p.cell_index,
                    tuple((str(x), str(y)) for x, y in p.vertices),
                    str(p.area),
                )
                for p in partition.pieces
            ],
            "cells": [
                (
                    c.part_name,
                    c.part_index,
                    c.cell_index,
                    str(c.source_area),
                    str(c.cavity_area),
                    str(c.owned_area),
                )
                for c in partition.cells
            ],
            "total": str(partition.total_area),
        }
    )


def _bound_partition_inputs(
    parts: tuple[MeshPart, ...],
    specs: tuple[OversetPartSpec, ...],
    priority: tuple[str, ...],
    sources: Mapping[str, Any],
) -> tuple[tuple[tuple[str, np.ndarray, np.ndarray], ...], tuple[np.ndarray, ...], str]:
    from ._association import GeometryAssociationKind, GeometrySourceEntityRole
    from ._motion_source import _rebuild_planar_source
    from .providers._native_sources import NativePlanarSource

    by_name = {part.name: part for part in parts}
    roles = {spec.part_name: spec for spec in specs}
    if len(priority) != len(parts) or set(priority) != set(by_name):
        raise ValueError("Partition priority must name every part exactly once.")
    wall_names = tuple(name for name in priority if roles[name].wall is not None)
    if set(sources) != set(wall_names):
        raise ValueError("Partition sources must bind exactly the declared wall parts.")
    cavities = []
    for name in wall_names:
        source = sources[name]
        if not isinstance(source, NativePlanarSource):
            raise TypeError("Exact partition walls require authored planar sources.")
        _rebuild_planar_source(source)
        part = by_name[name]
        result = _cell_result(part)
        if (
            result.certification is None
            or not isinstance(
                result.certification.inputs.feature_source, NativePlanarSource
            )
            or result.certification.inputs.feature_source.binding_id != source.binding_id
            or result.certification.inputs.source_revision != source.source_revision
        ):
            raise ValueError("Partition source does not bind the current certified mesh.")
        region = source.region
        offsets = np.asarray(region.loop_offsets, dtype=np.int64)
        edges = np.asarray(region.edges, dtype=np.int64)
        if offsets.shape != (3,):
            raise ValueError(
                "Exact partition requires one authored cavity loop per wall."
            )
        indices = range(int(offsets[1]), int(offsets[2]))
        keys = {
            source.source_entity_key_for_scope(1, int(source.edge_entity_ids[i]))
            for i in indices
        }
        selected, covered = set(), set()
        entity_set_id = result.mesh.entity_set(1).entity_set_id
        for association in result.associations:
            if (
                association.association_kind is not GeometryAssociationKind.CURVE
                or association.source_id != source.source_id
                or association.source_revision != source.source_revision
                or association.target_entity_set_id != entity_set_id
            ):
                continue
            for identifier, entity, path, role in zip(
                np.asarray(association.target_global_ids),
                association.source_entity_ids,
                association.source_occurrence_paths,
                association.source_roles,
                strict=True,
            ):
                key = (entity, path)
                if role is GeometrySourceEntityRole.EDGE and key in keys:
                    selected.add(int(identifier))
                    covered.add(key)
        wall = roles[name].wall
        if (
            wall is None
            or covered != keys
            or selected != set(map(int, np.asarray(wall.entity_ids)))
        ):
            raise ValueError(
                "Partition wall does not match its authored source associations."
            )
        polygon = np.array(
            np.asarray(region.vertices, dtype=np.float64)[
                edges[offsets[1] : offsets[2], 0]
            ],
            copy=True,
        )
        polygon.setflags(write=False)
        cavities.append(polygon)
    triangles = []
    for name in priority:
        result = _cell_result(by_name[name])
        mesh = result.mesh
        elements, routes, geometry_points = result.geometry.resolve(mesh)
        if (
            mesh.ambient_dimension != 2
            or result.geometry.source_map is not None
            or not np.array_equal(
                np.asarray(geometry_points), np.asarray(mesh.coordinates)
            )
            or not all(
                block.cell_kind == "triangle"
                and isinstance(element, FiniteElementSpec)
                and element.degree == 1
                and np.array_equal(np.asarray(route), np.asarray(block.vertices))
                for block, element, route in zip(
                    mesh.blocks, elements, routes, strict=True
                )
            )
        ):
            raise ValueError(
                "Exact overset partition requires affine planar P1 triangles."
            )
        triangles.append(
            (
                name,
                np.asarray(mesh.coordinates, dtype=np.float64),
                np.concatenate(
                    [np.asarray(b.vertices, dtype=np.int64) for b in mesh.blocks]
                ),
            )
        )
    binding = canonical_fingerprint(
        {
            "kind": "overset-partition-inputs",
            "priority": priority,
            "parts": [
                (
                    name,
                    by_name[name].part_id,
                    _cell_result(by_name[name]).result_id,
                    cell_geometry_id(_cell_result(by_name[name]).geometry),
                    roles[name].spec_id,
                )
                for name in priority
            ],
            "triangles": [
                (name, array_tree_fingerprint((points, cells)))
                for name, points, cells in triangles
            ],
            "sources": [(name, sources[name].binding_id) for name in wall_names],
            "cavities": [array_tree_fingerprint(polygon) for polygon in cavities],
        }
    )
    return tuple(triangles), tuple(cavities), binding


class OversetConnectivity(StrictModule, NonTrainableState):
    assembly: MeshAssembly
    specs: tuple[OversetPartSpec, ...]
    policy: OversetPolicy
    blanking: tuple[OversetPartBlanking, ...]
    receptors: tuple[OversetReceptorEvidence, ...]
    packets: tuple[OversetDonorPacket, ...]
    geometries: tuple[_PartGeometry, ...]
    bound_partition: OversetPartition | None = eqx.field(static=True)
    partition_priority: tuple[str, ...] | None = eqx.field(static=True)
    partition_sources: Mapping[str, Any]
    partition_cavities: tuple[np.ndarray, ...]
    partition_binding_id: str | None = eqx.field(static=True)
    partition_digest: str | None = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    predecessor_id: str | None = eqx.field(static=True)
    orphan_count: int = eqx.field(static=True)
    conflict_count: int = eqx.field(static=True)
    ambiguous_count: int = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    donor_search_complete: bool = eqx.field(static=True)
    connectivity_id: str = eqx.field(static=True)
    owner_local: bool = eqx.field(static=True)
    global_part_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        assembly: MeshAssembly,
        specs: tuple[OversetPartSpec, ...],
        policy: OversetPolicy,
        blanking: tuple[OversetPartBlanking, ...],
        receptors: tuple[OversetReceptorEvidence, ...],
        packets: tuple[OversetDonorPacket, ...],
        geometries: tuple[_PartGeometry, ...],
        /,
        *,
        epoch: int,
        predecessor_id: str | None = None,
        connectivity_id: str | None = None,
        global_conflict_count: int | None = None,
        global_ambiguous_count: int | None = None,
        donor_search_complete: bool = False,
        partition_priority: tuple[str, ...] | None = None,
        partition_sources: Mapping[str, Any] | None = None,
    ) -> None:
        expected_names = (
            assembly.global_part_names
            if assembly.owner_local
            else tuple(part.name for part in assembly.parts)
        )
        if tuple(spec.part_name for spec in specs) != expected_names:
            raise ValueError("Overset specs must follow the global manifest part order.")
        self.assembly, self.specs, self.policy = assembly, specs, policy
        self.blanking, self.receptors, self.packets, self.geometries = (
            blanking,
            receptors,
            packets,
            geometries,
        )
        self.epoch, self.predecessor_id = int(epoch), predecessor_id
        self.owner_local = assembly.owner_local
        self.global_part_names = assembly.global_part_names
        self.partition_priority = partition_priority
        self.partition_sources = (
            {} if partition_sources is None else dict(partition_sources)
        )
        self.bound_partition = None
        self.partition_cavities = ()
        self.partition_binding_id = None
        self.partition_digest = None
        if partition_priority is None:
            if self.partition_sources:
                raise ValueError("Partition sources require an explicit priority.")
        else:
            if assembly.owner_local:
                raise ValueError("Exact partition ownership is serial-only.")
            inputs, cavities, binding = _bound_partition_inputs(
                assembly.parts, specs, partition_priority, self.partition_sources
            )
            self.bound_partition = partition_overset_triangles(inputs, cavities)
            self.partition_cavities = cavities
            self.partition_binding_id = binding
            self.partition_digest = _partition_digest(self.bound_partition)
        self.orphan_count = sum(item.orphan_ids.size for item in receptors)
        local_conflicts = sum(
            int(np.count_nonzero(np.asarray(item.protected_conflict)))
            for item in blanking
        )
        local_ambiguous = sum(
            int(np.count_nonzero(np.asarray(item.ambiguous_cells))) for item in blanking
        )
        self.conflict_count = (
            local_conflicts
            if global_conflict_count is None
            else int(global_conflict_count)
        )
        self.ambiguous_count = (
            local_ambiguous
            if global_ambiguous_count is None
            else int(global_ambiguous_count)
        )
        if not isinstance(donor_search_complete, bool):
            raise TypeError("donor_search_complete must be a bool.")
        self.donor_search_complete = donor_search_complete
        self.complete = (
            donor_search_complete
            and self.orphan_count == 0
            and self.conflict_count == 0
            and (self.ambiguous_count == 0 or self.bound_partition is not None)
        )
        if connectivity_id is not None:
            identifier = str(connectivity_id)
            if not identifier:
                raise ValueError("connectivity_id must be non-empty.")
            self.connectivity_id = identifier
        else:
            self.connectivity_id = canonical_fingerprint(
                {
                    "kind": "native-overset-connectivity",
                    "donor_search_complete": donor_search_complete,
                    "assembly": assembly.assembly_id,
                    "specs": [item.spec_id for item in specs],
                    "policy": policy.policy_id,
                    "epoch": self.epoch,
                    "predecessor": predecessor_id,
                    "blanking": [item.blanking_id for item in blanking],
                    "receptors": [item.evidence_id for item in receptors],
                    "packets": [item.packet_id for item in packets],
                    "partition_binding": self.partition_binding_id,
                    "partition_digest": self.partition_digest,
                }
            )

    def part_index(self, name: str, /) -> int:
        try:
            return self.global_part_names.index(name)
        except ValueError as error:
            raise KeyError(name) from error

    def blanking_of(self, name: str, /) -> OversetPartBlanking:
        for item in self.blanking:
            if item.part_name == name:
                return item
        raise KeyError(f"Overset blanking for {name!r} is not local to this rank.")

    def receptors_of(self, name: str, /) -> OversetReceptorEvidence:
        for item in self.receptors:
            if item.part_name == name:
                return item
        raise KeyError(name)

    def require_complete(self) -> None:
        if self.partition_priority is not None:
            _, cavities, binding = _bound_partition_inputs(
                self.assembly.parts,
                self.specs,
                self.partition_priority,
                self.partition_sources,
            )
            expected_id = canonical_fingerprint(
                {
                    "kind": "native-overset-connectivity",
                    "donor_search_complete": self.donor_search_complete,
                    "assembly": self.assembly.assembly_id,
                    "specs": [item.spec_id for item in self.specs],
                    "policy": self.policy.policy_id,
                    "epoch": self.epoch,
                    "predecessor": self.predecessor_id,
                    "blanking": [item.blanking_id for item in self.blanking],
                    "receptors": [item.evidence_id for item in self.receptors],
                    "packets": [item.packet_id for item in self.packets],
                    "partition_binding": self.partition_binding_id,
                    "partition_digest": self.partition_digest,
                }
            )
            if (
                self.connectivity_id != expected_id
                or self.bound_partition is None
                or binding != self.partition_binding_id
                or _partition_digest(self.bound_partition) != self.partition_digest
                or len(cavities) != len(self.partition_cavities)
                or any(
                    not np.array_equal(a, b)
                    for a, b in zip(cavities, self.partition_cavities, strict=True)
                )
            ):
                raise ValueError("Overset partition is stale or has been altered.")
        elif self.bound_partition is not None or self.partition_sources:
            raise ValueError("Overset partition lacks its source binding.")
        if not self.complete:
            if not self.donor_search_complete:
                detail = "donor coverage has not been proved"
            else:
                detail = (
                    f"{self.orphan_count} orphan receptors, "
                    f"{self.conflict_count} protected-wall conflicts, and "
                    f"{self.ambiguous_count} unproved ambiguous cells"
                )
            raise OversetConnectivityError(
                f"Overset connectivity is incomplete: {detail}.",
                self,
            )

    def moved_from_advances(
        self,
        advances: Mapping[str, MeshMotionAdvance],
        /,
        *,
        solid_queries: Mapping[str, Any] | None = None,
        successor_specs: Mapping[str, OversetPartSpec] | None = None,
        owner_local_source_views: Mapping[str, OwnerLocalCellGeometry] | None = None,
        successor_assembly: MeshAssembly | None = None,
        execution_group: ExecutionGroup | None = None,
        owner_local_field_reconstructions: Mapping[str, PreparedFieldReconstruction]
        | None = None,
        owner_local_coefficient_ids: Mapping[str, ConvertibleToArray] | None = None,
        axis_name: str = "overset_motion",
        owner_local_field_name: str = "state",
    ) -> OversetConnectivity:
        """Rebuild donor evidence from accepted advances and explicit successor roles."""
        if self.partition_priority is not None:
            self.require_complete()
        from ._motion import MeshMotionAdvance

        if not isinstance(advances, Mapping) or (not advances and not self.owner_local):
            raise TypeError(
                "Motion advances must map accepted assembly parts to motion advances."
            )
        queries = {} if solid_queries is None else dict(solid_queries)
        specs_ = {} if successor_specs is None else dict(successor_specs)
        if any(query is None for query in queries.values()):
            raise TypeError("successor solid queries must be non-None.")
        if self.owner_local:
            return self._moved_from_owner_local_advances(
                advances,
                queries,
                specs_,
                owner_local_source_views,
                successor_assembly,
                execution_group,
                owner_local_field_reconstructions,
                owner_local_coefficient_ids,
                axis_name,
                owner_local_field_name,
            )
        names = {part.name for part in self.assembly.parts}
        if (
            not set(advances).issubset(names)
            or not set(queries).issubset(advances)
            or not set(specs_).issubset(advances)
        ):
            raise ValueError(
                "Motion advances and successor roles may name only moved assembly parts."
            )
        if set(queries).intersection(specs_):
            raise ValueError(
                "Provide a successor spec or a successor solid query, not both for one part."
            )
        if any(not isinstance(spec, OversetPartSpec) for spec in specs_.values()):
            raise TypeError("successor_specs must contain OversetPartSpec values.")
        if any(spec.part_name != name for name, spec in specs_.items()):
            raise ValueError("Every successor spec must retain its registered part name.")
        replacements: dict[str, MeshPart] = {}
        for part in self.assembly.parts:
            if part.name not in advances:
                continue
            advance = advances[part.name]
            if not isinstance(advance, MeshMotionAdvance):
                raise TypeError("Every moved part requires a MeshMotionAdvance.")
            if not advance.accepted or advance.result is None:
                raise ValueError(
                    "Overset connectivity requires an accepted mesh-motion result."
                )
            source = _cell_result(part)
            target = advance.result
            if (
                advance.source_mesh_id != source.mesh.mesh_id
                or advance.source_result_id != source.result_id
            ):
                raise ValueError(
                    "Mesh-motion advance does not bind the current overset source result."
                )
            if (
                target.coordinate_contract.spatial_id
                != part.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "Mesh-motion result changed the registered coordinate contract."
                )
            geometry_transition = advance.geometry_transition
            if geometry_transition is None:
                raise ValueError(
                    "Accepted motion advance lacks its exact source coordinate transition."
                )
            if (
                geometry_transition.source_mesh.mesh_id != source.mesh.mesh_id
                or geometry_transition.source_topology_id != source.mesh.topology_id
                or geometry_transition.source_geometry_id
                != cell_geometry_id(source.geometry)
            ):
                raise ValueError(
                    "Motion coordinate transition does not bind the exact overset source map."
                )
            geometry_transition.require_bound(
                source.mesh,
                source.geometry,
                geometry_transition.target_mesh,
                geometry_transition.target_geometry,
            )
            transition = advance.transition
            if transition is None:
                if (
                    target.mesh.mesh_id != geometry_transition.target_mesh.mesh_id
                    or target.mesh.topology_id != geometry_transition.target_topology_id
                    or cell_geometry_id(target.geometry)
                    != geometry_transition.target_geometry_id
                ):
                    raise ValueError(
                        "Motion result does not match its coordinate transition target."
                    )
            elif (
                transition.source_mesh_id != geometry_transition.target_mesh.mesh_id
                or transition.source_topology_id != geometry_transition.target_topology_id
                or transition.target.result_id != target.result_id
                or transition.target.mesh.mesh_id != target.mesh.mesh_id
            ):
                raise ValueError(
                    "Adaptation transition does not continue the accepted motion candidate."
                )
            elif transition.geometry_transition is not None:
                transition.geometry_transition.require_bound(
                    geometry_transition.target_mesh,
                    geometry_transition.target_geometry,
                    target.mesh,
                    target.geometry,
                )
            replacements[part.name] = MeshPart(
                part.name, target, coordinate_contract=part.coordinate_contract
            )
        if not replacements:
            raise ValueError("Motion advances do not name a registered assembly part.")
        if any(
            spec.solid_query is not None
            and spec.part_name in replacements
            and queries.get(spec.part_name) is None
            and (
                spec.part_name not in specs_ or specs_[spec.part_name].solid_query is None
            )
            for spec in self.specs
        ):
            raise ValueError(
                "Moved source solids require an explicit successor solid query."
            )
        partition_sources = dict(self.partition_sources)
        if self.partition_priority is not None:
            from ._motion_source import NativePlanarMotionSource

            for name in set(partition_sources).intersection(advances):
                transition = advances[name].source_transition
                if not isinstance(transition, NativePlanarMotionSource):
                    raise ValueError(
                        "Moved partition walls require an authored source transition."
                    )
                transition = NativePlanarMotionSource.from_stored_fields(transition)
                if transition.source.binding_id != partition_sources[name].binding_id:
                    raise ValueError(
                        "Partition wall transition does not bind the accepted source."
                    )
                partition_sources[name] = transition.successor
        return self.reregister(
            replacements,
            solid_queries=queries,
            successor_specs=specs_,
            partition_sources=partition_sources
            if self.partition_priority is not None
            else None,
            _partition_advances=advances if self.partition_priority is not None else None,
        )

    def _moved_from_owner_local_advances(
        self,
        advances: Mapping[str, Any],
        solid_queries: Mapping[str, Any],
        successor_specs: Mapping[str, OversetPartSpec],
        owner_local_source_views: Mapping[str, OwnerLocalCellGeometry] | None,
        successor_assembly: MeshAssembly | None,
        execution_group: ExecutionGroup | None,
        field_reconstructions: Mapping[str, PreparedFieldReconstruction] | None,
        coefficient_ids: Mapping[str, ConvertibleToArray] | None,
        axis_name: str,
        field_name: str,
        /,
    ) -> OversetConnectivity:
        """Collectively rebuild one owner-local motion successor without gathering meshes."""
        from ._motion import MeshMotionAdvance
        from ._overset_exchange import (
            _owner_local_all_gather_rows,
            _synchronize_owner_search_failure,
        )

        if execution_group is None or not isinstance(execution_group, ExecutionGroup):
            raise TypeError("Owner-local motion requires an ExecutionGroup.")
        if successor_assembly is None or not isinstance(successor_assembly, MeshAssembly):
            raise TypeError(
                "Owner-local motion requires its accepted successor assembly."
            )
        if not successor_assembly.owner_local:
            raise ValueError(
                "Owner-local motion cannot publish a global successor assembly."
            )
        axis = str(axis_name).strip()
        field = str(field_name).strip()
        if not axis or not field:
            raise ValueError(
                "Owner-local motion needs non-empty collective and field names."
            )
        if (
            successor_assembly.global_part_names != self.global_part_names
            or successor_assembly.local_rank != self.assembly.local_rank
            or self.assembly.local_rank is None
            or not execution_group.is_member
            or len(execution_group.devices) != len(execution_group.spec.process_indices)
        ):
            raise ValueError(
                "Owner-local successor assembly or execution group is stale."
            )
        local_parts = {part.name: part for part in self.assembly.parts}
        target_parts = {part.name: part for part in successor_assembly.parts}
        source_views: dict[str, OwnerLocalCellGeometry] = {}
        reconstruction_map: dict[str, PreparedFieldReconstruction] = {}
        coefficient_map: dict[str, ConvertibleToArray] = {}
        local_error: Exception | None = None
        try:
            if owner_local_source_views is None:
                source_views = {
                    name: part.carrier
                    for name, part in local_parts.items()
                    if isinstance(part.carrier, OwnerLocalCellGeometry)
                }
            elif isinstance(owner_local_source_views, Mapping):
                source_views = dict(owner_local_source_views)
            else:
                raise TypeError("owner_local_source_views must map local part names.")
            if field_reconstructions is not None:
                if not isinstance(field_reconstructions, Mapping):
                    raise TypeError("field_reconstructions must map local donor parts.")
                reconstruction_map = dict(field_reconstructions)
            if coefficient_ids is not None:
                if not isinstance(coefficient_ids, Mapping):
                    raise TypeError("coefficient_ids must map local reconstruction rows.")
                coefficient_map = dict(coefficient_ids)
            if (
                not isinstance(advances, Mapping)
                or not set(advances).issubset(self.global_part_names)
                or not set(advances).issubset(source_views)
                or not set(successor_specs).issubset(self.global_part_names)
                or not set(advances).issubset(successor_specs)
                or not set(solid_queries).issubset(successor_specs)
                or set(solid_queries).intersection(successor_specs)
                or solid_queries
            ):
                raise ValueError(
                    "Owner-local motion needs global moved-part names and explicit successor specs."
                )
            if set(source_views) != set(local_parts) or set(target_parts) != set(
                local_parts
            ):
                raise ValueError(
                    "Owner-local motion must retain the exact rank-local part ownership."
                )
            if any(
                not isinstance(spec, OversetPartSpec) for spec in successor_specs.values()
            ):
                raise TypeError("successor_specs must contain OversetPartSpec values.")
            if any(successor_specs[name].part_name != name for name in successor_specs):
                raise ValueError("Successor specs must retain their global part names.")
            if not set(advances).issubset(successor_specs):
                raise ValueError(
                    "Every moved owner-local part requires an explicit successor overset spec."
                )
            if set(reconstruction_map) != set(coefficient_map):
                raise ValueError(
                    "Owner-local reconstruction and coefficient-ID bindings must match."
                )
            if any(
                not isinstance(binding, PreparedFieldReconstruction)
                or binding.value_port.semantic_id != field
                for binding in reconstruction_map.values()
            ):
                raise ValueError(
                    "Owner-local donor reconstructions must bind the requested physical field."
                )
            for name, source in source_views.items():
                source_part = local_parts[name]
                target_part = target_parts[name]
                source_carrier = _owner_local_carrier(source_part)
                target = _owner_local_carrier(target_part)
                moved_here = name in advances
                if (
                    not isinstance(source, OwnerLocalCellGeometry)
                    or not isinstance(target, OwnerLocalCellGeometry)
                    or source_carrier.local_view_id != source.local_view_id
                    or target.part_name != name
                    or source.part_name != name
                    or source.owner_rank != self.assembly.local_rank
                    or target.owner_rank != self.assembly.local_rank
                    or source.local_view_id != source_carrier.local_view_id
                    or source.source_mesh_id != target.source_mesh_id
                    or (
                        not moved_here
                        and (
                            source.source_topology_id != target.source_topology_id
                            or source.source_geometry_id != target.source_geometry_id
                        )
                    )
                    or not np.array_equal(
                        np.asarray(source.cell_ids), np.asarray(target.cell_ids)
                    )
                    or not np.array_equal(
                        np.asarray(source.owned), np.asarray(target.owned)
                    )
                ):
                    raise ValueError(
                        "Owner-local successor changed an accepted source, cell, or ownership binding."
                    )
                source.require_owner_directory_current(
                    execution_group,
                    axis_name=f"{axis}_{name}_source_directory",
                )
                target.require_owner_directory_current(
                    execution_group,
                    axis_name=f"{axis}_{name}_target_directory",
                )
                if name in advances:
                    advance = advances[name]
                    if not isinstance(advance, MeshMotionAdvance):
                        raise TypeError(
                            "Every moved local part requires MeshMotionAdvance."
                        )
                    if (
                        not advance.accepted
                        or advance.result is not None
                        or advance.local_result is None
                        or advance.local_source_view_id != source.local_view_id
                        or advance.local_target_view_id != target.local_view_id
                        or advance.local_result.local_view_id != target.local_view_id
                        or advance.motion_driver_id is None
                        or not advance.motion_driver_id
                        or advance.local_result.candidate_revision_id is None
                        or advance.local_result.part_id != target.part_id
                        or advance.local_result.distribution_id != target.distribution_id
                        or advance.local_result.source_id != target.source_id
                        or advance.local_result.source_revision != target.source_revision
                        or advance.local_result.source_geometry_id
                        != target.source_geometry_id
                    ):
                        raise ValueError(
                            "Owner-local MeshMotionAdvance does not bind its accepted source and target views."
                        )
                    transition = advance.geometry_transition
                    if np.asarray(source.cell_ids).size == 0:
                        if (
                            transition is not None
                            or advance.local_target_mesh_id is not None
                            or source.local_mesh_id is not None
                            or target.local_mesh_id is not None
                        ):
                            raise ValueError(
                                "Empty owner-local motion shards cannot claim local mesh geometry."
                            )
                    elif (
                        transition is None
                        or advance.source_mesh_id != source.local_mesh_id
                        or advance.local_target_mesh_id != target.local_mesh_id
                        or transition.source_mesh.mesh_id != source.local_mesh_id
                        or transition.source_topology_id != source.local_topology_id
                        or transition.source_geometry_id != source.local_geometry_id
                        or transition.target_mesh.mesh_id != target.local_mesh_id
                        or transition.target_topology_id != target.local_topology_id
                        or transition.target_geometry_id != target.local_geometry_id
                    ):
                        raise ValueError(
                            "Owner-local motion lacks its exact canonical source-to-target geometry transition."
                        )
                    if transition is not None:
                        transition.require_bound(
                            transition.source_mesh,
                            transition.source_geometry,
                            transition.target_mesh,
                            transition.target_geometry,
                        )
                elif target.local_view_id != source.local_view_id:
                    raise ValueError(
                        "An unchanged owner-local part cannot substitute a new geometry view."
                    )
        except Exception as error:
            local_error = error
        if _synchronize_owner_search_failure(
            local_error is not None,
            execution_group.devices,
            f"{axis}_motion_admission",
        ):
            detail = (
                str(local_error)
                if local_error is not None
                else "another rank rejected its local motion transition"
            )
            raise ValueError(
                "Owner-local motion admission failed collectively: " + detail
            )
        if local_error is not None:
            raise ValueError("Owner-local motion admission failed collectively.")

        roles = {spec.part_name: spec for spec in self.specs}
        roles.update(successor_specs)
        ordered_specs = tuple(roles[name] for name in self.global_part_names)
        candidate = prepare_overset_connectivity(
            successor_assembly,
            ordered_specs,
            policy=self.policy,
            execution_group=execution_group,
            axis_name=f"{axis}_candidate",
        )
        transfer = prepare_overset_field_transfer(
            candidate,
            reconstruction_map,
            field,
        )
        exchange = transfer.prepare_owner_local_exchange(
            coefficient_map,
            execution_group,
            {name: _owner_local_carrier(part) for name, part in target_parts.items()},
            axis_name=f"{axis}_resolve",
        )
        resolved = exchange.connectivity
        if _synchronize_owner_search_failure(
            not resolved.complete,
            execution_group.devices,
            f"{axis}_resolved_admission",
        ):
            raise OversetConnectivityError(
                "Owner-local motion produced incomplete collective donor coverage.",
                resolved,
            )
        if not resolved.complete:
            raise OversetConnectivityError(
                "Owner-local motion produced incomplete donor coverage.",
                resolved,
            )
        local_motion_rows = []
        for name, advance in advances.items():
            revision = advance.local_result.candidate_revision_id
            digest = bytes.fromhex(
                canonical_fingerprint(
                    {"kind": "owner-local-motion-revision", "revision": revision}
                )
            )
            local_motion_rows.append((self.global_part_names.index(name), *digest))
        gathered_motion_rows = _owner_local_all_gather_rows(
            np.asarray(local_motion_rows, dtype=np.int64).reshape((-1, 33)),
            execution_group.devices,
            f"{axis}_motion_revisions",
        )
        revisions_by_name: dict[str, str] = {}
        for rank_rows in gathered_motion_rows:
            for row in rank_rows:
                part_index = int(row[0])
                if not 0 <= part_index < len(self.global_part_names):
                    raise ValueError("Collective motion revision names an unknown part.")
                name = self.global_part_names[part_index]
                revision_digest = bytes(
                    np.asarray(row[1:], dtype=np.uint8).tolist()
                ).hex()
                previous_digest = revisions_by_name.setdefault(name, revision_digest)
                if previous_digest != revision_digest:
                    raise ValueError(
                        "Owner-local motion shards disagree on their accepted target revision."
                    )
        if set(revisions_by_name) != set(successor_specs):
            raise ValueError(
                "Collective motion revision evidence does not cover every moved part."
            )
        motion_revisions = tuple(
            (name, revisions_by_name[name]) for name in sorted(revisions_by_name)
        )
        local_resolved_digest = bytes.fromhex(
            canonical_fingerprint(
                {
                    "kind": "owner-local-resolved-donors",
                    "connectivity": resolved.connectivity_id,
                }
            )
        )
        gathered_resolved_digests = _owner_local_all_gather_rows(
            np.asarray(
                ((int(self.assembly.local_rank), *local_resolved_digest),),
                dtype=np.int64,
            ),
            execution_group.devices,
            f"{axis}_resolved_identities",
        )
        resolved_digests_by_rank: dict[int, str] = {}
        for rank_rows in gathered_resolved_digests:
            for row in rank_rows:
                rank = int(row[0])
                if not 0 <= rank < len(execution_group.spec.process_indices):
                    raise ValueError("Owner-local donor identity names an unknown rank.")
                digest = bytes(np.asarray(row[1:], dtype=np.uint8).tolist()).hex()
                previous_digest = resolved_digests_by_rank.setdefault(rank, digest)
                if previous_digest != digest:
                    raise ValueError(
                        "Owner-local rank produced inconsistent resolved donor identities."
                    )
        if set(resolved_digests_by_rank) != set(
            range(len(execution_group.spec.process_indices))
        ):
            raise ValueError("Collective donor identity omitted an execution-group rank.")
        global_resolved_digests = tuple(sorted(resolved_digests_by_rank.items()))
        resolved_id = canonical_fingerprint(
            {
                "kind": "owner-local-motion-overset-connectivity",
                "predecessor": self.connectivity_id,
                "candidate": candidate.connectivity_id,
                "epoch": self.epoch + 1,
                "motion_revisions": motion_revisions,
                "resolved_partition_ids": global_resolved_digests,
            }
        )
        return OversetConnectivity(
            resolved.assembly,
            resolved.specs,
            resolved.policy,
            resolved.blanking,
            resolved.receptors,
            resolved.packets,
            resolved.geometries,
            epoch=self.epoch + 1,
            predecessor_id=self.connectivity_id,
            connectivity_id=resolved_id,
            global_conflict_count=resolved.conflict_count,
            global_ambiguous_count=resolved.ambiguous_count,
            donor_search_complete=True,
        )

    def reregister(
        self,
        replacements: Mapping[str, MeshPart],
        /,
        *,
        solid_queries: Mapping[str, Any] | None = None,
        successor_specs: Mapping[str, OversetPartSpec] | None = None,
        partition_sources: Mapping[str, Any] | None = None,
        _partition_advances: Mapping[str, MeshMotionAdvance] | None = None,
        couplings: tuple[MeshCoupling, ...] | None = None,
        execution_group: ExecutionGroup | None = None,
        axis_name: str = "overset_reregister",
    ) -> OversetConnectivity:
        """Register successor parts as the next connectivity epoch.

        Serial connectivity is rebuilt directly. Owner-local connectivity is a
        collective preparation candidate over the successor local views (for
        example a graph-repartitioned part); its prepared field exchange
        resolves the donor winners, and the epoch chain is kept.
        """
        if self.partition_priority is not None:
            self.require_complete()
            walls = set(replacements).intersection(self.partition_sources)
            if walls and (partition_sources is None or _partition_advances is None):
                raise ValueError(
                    "Replaced partition walls require accepted source motion."
                )
            if partition_sources is not None:
                if set(partition_sources) != set(self.partition_sources):
                    raise ValueError(
                        "Successor partition sources must retain the wall inventory."
                    )
                from ._motion import MeshMotionAdvance
                from ._motion_source import NativePlanarMotionSource

                for name, source in self.partition_sources.items():
                    successor = partition_sources[name]
                    if name not in walls:
                        if successor.binding_id != source.binding_id:
                            raise ValueError(
                                "An unchanged wall cannot change its source."
                            )
                        continue
                    advance = (
                        None
                        if _partition_advances is None
                        else _partition_advances.get(name)
                    )
                    if (
                        not isinstance(advance, MeshMotionAdvance)
                        or not advance.accepted
                        or advance.result is None
                        or advance.source_result_id
                        != _cell_result(self.assembly.part(name)).result_id
                        or advance.result.result_id
                        != _cell_result(replacements[name]).result_id
                        or not isinstance(
                            advance.source_transition, NativePlanarMotionSource
                        )
                    ):
                        raise ValueError(
                            "Partition wall lacks its accepted predecessor motion."
                        )
                    transition = NativePlanarMotionSource.from_stored_fields(
                        advance.source_transition
                    )
                    if (
                        transition.source.binding_id != source.binding_id
                        or transition.successor.binding_id != successor.binding_id
                    ):
                        raise ValueError(
                            "Partition source motion does not bind its exact endpoints."
                        )
        elif partition_sources is not None:
            raise ValueError(
                "Cannot attach partition sources to an unbound registration."
            )
        if (
            not isinstance(replacements, Mapping)
            or not replacements
            or not set(replacements).issubset({part.name for part in self.assembly.parts})
        ):
            raise ValueError(
                "reregister requires non-empty replacements for registered parts."
            )
        queries = {} if solid_queries is None else dict(solid_queries)
        specs_ = {} if successor_specs is None else dict(successor_specs)
        if any(query is None for query in queries.values()):
            raise TypeError("successor solid queries must be non-None.")
        if (
            not set(queries).issubset(replacements)
            or not set(specs_).issubset(replacements)
            or set(queries).intersection(specs_)
        ):
            raise ValueError(
                "Successor queries and specs must name distinct replaced parts."
            )
        if any(not isinstance(spec, OversetPartSpec) for spec in specs_.values()):
            raise TypeError("successor_specs must contain OversetPartSpec values.")
        parts = tuple(replacements.get(part.name, part) for part in self.assembly.parts)
        specs = tuple(
            specs_[part.name]
            if part.name in specs_
            else spec.rebound(part, solid_query=queries.get(part.name))
            for part, spec in zip(parts, self.specs, strict=True)
        )
        if any(
            spec.part_name != part.name for part, spec in zip(parts, specs, strict=True)
        ):
            raise ValueError("Successor specs must retain their registered part names.")
        retained = (
            tuple(
                link
                for link in self.assembly.couplings
                if not isinstance(link, OversetCoupling)
            )
            if couplings is None
            else tuple(couplings)
        )
        moved_names = set(replacements)
        if couplings is None and any(
            link.source_scope.source_id in moved_names
            or link.target_scope.source_id in moved_names
            for link in retained
        ):
            raise ValueError(
                "Non-overset couplings touching moved parts must be reprepared by their owner."
            )
        if self.owner_local:
            if not isinstance(execution_group, ExecutionGroup):
                raise TypeError(
                    "Owner-local reregistration requires its execution_group."
                )
            if retained:
                raise ValueError(
                    "Owner-local assemblies prepare coupling routes collectively."
                )
            successor_assembly = MeshAssembly(
                parts,
                global_parts=tuple(
                    MeshPartManifest.from_part(part)
                    if part.name in moved_names
                    else self.assembly.declared_part(part.name)
                    for part in parts
                ),
                local_rank=self.assembly.local_rank,
            )
            candidate = prepare_overset_connectivity(
                successor_assembly,
                specs,
                policy=self.policy,
                execution_group=execution_group,
                axis_name=axis_name,
            )
            return OversetConnectivity(
                candidate.assembly,
                candidate.specs,
                candidate.policy,
                candidate.blanking,
                candidate.receptors,
                candidate.packets,
                candidate.geometries,
                epoch=self.epoch + 1,
                predecessor_id=self.connectivity_id,
                connectivity_id=canonical_fingerprint(
                    {
                        "kind": "owner-local-reregistered-overset-connectivity",
                        "predecessor": self.connectivity_id,
                        "candidate": candidate.connectivity_id,
                        "epoch": self.epoch + 1,
                    }
                ),
                global_conflict_count=candidate.conflict_count,
                global_ambiguous_count=candidate.ambiguous_count,
            )
        if execution_group is not None:
            raise ValueError(
                "execution_group is only valid for owner-local connectivity."
            )
        return _prepare_connectivity(
            tuple(parts),
            retained,
            specs,
            self.policy,
            self.epoch + 1,
            self.connectivity_id,
            self.partition_priority,
            self.partition_sources if partition_sources is None else partition_sources,
        )

    def composition_entries(self) -> tuple[CompositionEntry, ...]:
        if self.owner_local or self.partition_priority is not None:
            self.require_complete()
        if self.owner_local:
            part_entries = tuple(
                CompositionEntry(
                    item,
                    entry_id=f"overset:part:{item.name}",
                    role="topology",
                    owner_id=_OWNER,
                    structure_id=_owner_local_manifest_binding(item)[0],
                    # The manifest revision carries the moved geometry; a part's
                    # semantics are its role in the overset, stable across motion.
                    revision_id=item.manifest_id,
                    semantics_id=canonical_fingerprint(
                        {"kind": "overset-owner-local-part", "name": item.name}
                    ),
                )
                for item in self.assembly.part_manifest
            )
        else:
            part_entries = tuple(
                CompositionEntry(
                    part,
                    entry_id=f"overset:part:{part.name}",
                    role="topology",
                    owner_id=_OWNER,
                    structure_id=_cell_result(part).mesh.topology_id,
                    revision_id=part.part_id,
                    semantics_id=canonical_fingerprint(
                        {"kind": "overset-part", "name": part.name}
                    ),
                )
                for part in self.assembly.parts
            )
        route = CompositionEntry(
            self,
            entry_id=_CONNECTIVITY_ENTRY,
            role="interface-route",
            owner_id=_OWNER,
            structure_id=self.connectivity_id,
            revision_id=self.connectivity_id,
            semantics_id="native-overset-authoritative-hole-cut-donor-search",
            dependencies=tuple(entry.binding("revision") for entry in part_entries),
        )
        return (*part_entries, route)

    def registration(self) -> OversetRegistration:
        if self.owner_local or self.partition_priority is not None:
            self.require_complete()
        assembly = (
            self.assembly
            if self.owner_local
            else MeshAssembly(
                self.assembly.parts,
                couplings=tuple(
                    link
                    for link in self.assembly.couplings
                    if not isinstance(link, OversetCoupling)
                ),
            )
        )
        return OversetRegistration(
            assembly,
            self.specs,
            self.policy,
            self.epoch,
            self.predecessor_id,
            self.connectivity_id,
            partition_priority=self.partition_priority,
            partition_sources=self.partition_sources,
        )


def _prepare_connectivity(
    parts: tuple[MeshPart, ...],
    retained: tuple[MeshCoupling, ...],
    specs: tuple[OversetPartSpec, ...],
    policy: OversetPolicy,
    epoch: int,
    predecessor_id: str | None,
    partition_priority: tuple[str, ...] | None = None,
    partition_sources: Mapping[str, Any] | None = None,
) -> OversetConnectivity:
    geometries = tuple(
        _geometry(part, spec) for part, spec in zip(parts, specs, strict=True)
    )
    blanking = _make_blanking(parts, specs, geometries, policy)
    receptors, packets, overlays = _receptor_records(
        parts, specs, geometries, blanking, policy
    )
    return OversetConnectivity(
        MeshAssembly(parts, couplings=(*retained, *overlays)),
        specs,
        policy,
        blanking,
        receptors,
        packets,
        geometries,
        epoch=epoch,
        predecessor_id=predecessor_id,
        donor_search_complete=True,
        partition_priority=partition_priority,
        partition_sources=partition_sources,
    )


def _prepare_owner_local_connectivity(
    assembly: MeshAssembly,
    specs: tuple[OversetPartSpec, ...],
    policy: OversetPolicy,
    execution_group: ExecutionGroup,
    axis_name: str,
    /,
) -> OversetConnectivity:
    import jax

    from ._overset_exchange import (
        _owner_metadata_mismatch,
        _synchronize_owner_search_failure,
    )

    group_ranks = execution_group.spec.process_indices
    devices = execution_group.devices
    axis = str(axis_name).strip()
    if (
        not axis
        or len(devices) != len(group_ranks)
        or tuple(device.process_index for device in devices) != group_ranks
        or not execution_group.is_member
    ):
        raise ValueError(
            "Owner-local overset requires one execution-group device per member process."
        )
    process_rank = jax.process_index()
    if process_rank not in group_ranks:
        raise ValueError("Owner-local overset process is outside its execution group.")
    partition_index = group_ranks.index(process_rank)
    names = assembly.global_part_names
    by_name = {spec.part_name: spec for spec in specs}
    local_error: Exception | None = None
    ordered_specs: tuple[OversetPartSpec, ...] = ()
    manifest_bindings: dict[str, tuple[MeshPartManifest, str, str]] = {}
    local_parts = {part.name: part for part in assembly.parts}
    try:
        if not assembly.owner_local or assembly.local_rank != partition_index:
            raise ValueError(
                "Owner-local assembly rank differs from its execution-group rank."
            )
        if len(by_name) != len(specs) or set(by_name) != set(names):
            raise ValueError(
                "Owner-local overset requires one role specification per global manifest part."
            )
        ordered_specs = tuple(by_name[name] for name in names)
        if any(
            spec.owner_rank < 0 or spec.owner_rank >= len(group_ranks)
            for spec in ordered_specs
        ):
            raise ValueError("Overset field owners must be execution-group rank indices.")
        if any(isinstance(link, OversetCoupling) for link in assembly.couplings):
            raise ValueError(
                "Remove previous overset overlays before preparing new connectivity."
            )
        if set(local_parts) != set(names):
            raise ValueError(
                "Owner-local overset requires one current local view per global part, including empty owners."
            )
        for identity in assembly.part_manifest:
            topology_id, geometry_id = _owner_local_manifest_binding(identity)
            part = local_parts[identity.name]
            view = _owner_local_carrier(part)
            if (
                identity.distribution_id is None
                or part.part_id != identity.part_id
                or part.coordinate_contract.spatial_id
                != identity.coordinate_contract.spatial_id
                or part.intrinsic_dimension != identity.intrinsic_dimension
                or part.ambient_dimension != identity.ambient_dimension
                or view.owner_rank != partition_index
                or view.part_name != identity.name
                or view.part_id != identity.part_id
                or view.distribution_id != identity.distribution_id
                or view.source_topology_id != topology_id
                or view.source_geometry_id != geometry_id
            ):
                raise ValueError(
                    "Owner-local manifest differs from its current compact local geometry view."
                )
            manifest_bindings[identity.name] = (
                identity,
                topology_id,
                geometry_id,
            )
        manifest_id = canonical_fingerprint(
            {
                "assembly": assembly.assembly_id,
                "parts": tuple(
                    (item.manifest_id, item.distribution_id)
                    for item in assembly.part_manifest
                ),
                "policy": policy.policy_id,
                "specs": tuple(
                    (
                        spec.part_name,
                        spec.priority,
                        spec.owner_rank,
                        _solid_query_identity(spec.solid_query),
                        spec.solid_index,
                        array_tree_fingerprint(
                            (spec.image_rotation, spec.image_translation)
                        ),
                    )
                    for spec in ordered_specs
                ),
            }
        )
    except Exception as error:
        local_error = error
        manifest_id = ""
    if _synchronize_owner_search_failure(local_error is not None, devices, axis):
        detail = (
            str(local_error)
            if local_error is not None
            else "another rank rejected its owner-local manifest"
        )
        raise ValueError("Owner-local overset manifest failed collectively: " + detail)
    if local_error is not None:
        raise ValueError("Owner-local overset manifest failed collectively.")
    if _owner_metadata_mismatch(manifest_id, devices):
        raise ValueError(
            "Owner-local overset manifests or global role specifications differ."
        )
    for identity in assembly.part_manifest:
        _owner_local_carrier(local_parts[identity.name]).require_owner_directory_current(
            execution_group,
            axis_name=f"{axis}_{identity.name}_directory",
        )
    geometry_by_name: dict[str, _PartGeometry] = {}
    local_error = None
    try:
        from ..discretization._view_support import MappedFieldSupport

        for part in assembly.parts:
            view = _owner_local_carrier(part)
            spec = by_name[part.name]
            identity, topology_id, geometry_id = manifest_bindings[part.name]
            if (
                view.owner_rank != partition_index
                or view.part_name != part.name
                or view.part_id != identity.part_id
                or view.distribution_id != identity.distribution_id
                or view.source_topology_id != topology_id
                or view.source_geometry_id != geometry_id
                or part.coordinate_contract.spatial_id
                != identity.coordinate_contract.spatial_id
            ):
                raise ValueError(
                    "Owner-local compact geometry view is stale against its manifest."
                )
            if isinstance(spec.solid_query, MappedFieldSupport):
                spec.solid_query.require_locator(spec.solid_query.locator)
                if (
                    spec.solid_query.mesh.mesh_id != view.source_mesh_id
                    or spec.solid_query.geometry_id != view.source_geometry_id
                ):
                    raise ValueError(
                        "Mapped source solid support must bind the accepted local source geometry."
                    )
            geometry_by_name[part.name] = _owner_local_geometry(part, spec)
    except MemoryError:
        local_error = MemoryError(
            "Owner-local overset geometry preparation failed allocation."
        )
    except Exception as error:
        local_error = error
    if _synchronize_owner_search_failure(local_error is not None, devices, axis):
        detail = (
            str(local_error)
            if local_error is not None
            else "another rank rejected its source geometry"
        )
        raise ValueError("Owner-local overset geometry failed collectively: " + detail)
    if local_error is not None:
        raise ValueError("Owner-local overset geometry failed collectively.")
    local_queries, local_query_owners, local_query_points = _owner_local_point_bank(
        assembly, execution_group, axis
    )
    (
        boundary_vertices,
        protected_vertices,
        excluded_cells,
        wall_facets,
        boundary_facets,
        wall_dimensions,
        wall_owner_ranks,
    ) = _owner_local_scope_metadata(assembly, ordered_specs, execution_group, axis)
    if wall_facets.intersection(boundary_facets):
        raise ValueError("Overset solid walls and receptor boundaries must be disjoint.")
    dimension = assembly.part_manifest[0].ambient_dimension
    holes_by_vertex, contacts_by_vertex = _owner_local_wall_membership(
        assembly,
        ordered_specs,
        local_queries,
        local_query_owners,
        local_query_points,
        wall_dimensions,
        wall_owner_ranks,
        geometry_by_name,
        execution_group,
        axis,
        dimension,
    )
    cut_cells = _owner_local_cut_cells(
        assembly,
        ordered_specs,
        wall_dimensions,
        wall_owner_ranks,
        geometry_by_name,
        execution_group,
        axis,
    )
    local_blanking, receptors, conflicts, ambiguous = _owner_local_prepare_blanking(
        assembly,
        ordered_specs,
        policy,
        geometry_by_name,
        holes_by_vertex,
        contacts_by_vertex,
        boundary_vertices,
        protected_vertices,
        excluded_cells,
        cut_cells,
        execution_group,
        axis,
    )
    global_scope_id = canonical_fingerprint(
        {
            "kind": "owner-local-overset-scopes",
            "walls": tuple(sorted(wall_facets)),
            "boundaries": tuple(sorted(boundary_facets)),
            "boundary_vertices": tuple(sorted(boundary_vertices)),
            "protected_vertices": tuple(sorted(protected_vertices)),
            "excluded_cells": tuple(sorted(excluded_cells)),
        }
    )
    connectivity_id = canonical_fingerprint(
        {
            "kind": "owner-local-overset-connectivity",
            "assembly": assembly.assembly_id,
            "manifest": manifest_id,
            "scopes": global_scope_id,
            "policy": policy.policy_id,
        }
    )
    return OversetConnectivity(
        assembly,
        ordered_specs,
        policy,
        local_blanking,
        receptors,
        (),
        tuple(geometry_by_name[part.name] for part in assembly.parts),
        epoch=0,
        connectivity_id=connectivity_id,
        global_conflict_count=conflicts,
        global_ambiguous_count=ambiguous,
    )


@checked
def prepare_overset_connectivity(
    assembly: MeshAssembly,
    parts: tuple[OversetPartSpec, ...],
    /,
    *,
    policy: OversetPolicy | None = None,
    execution_group: ExecutionGroup | None = None,
    axis_name: str = "overset",
    partition_priority: tuple[str, ...] | None = None,
    partition_sources: Mapping[str, Any] | None = None,
) -> OversetConnectivity:
    """Prepare serial connectivity or collective rank-local blanking/receptors.

    Owner-local calls require every execution-group member to provide the
    complete global role-spec tuple and its local ``MeshAssembly`` view. The
    returned owner-local connectivity is a preparation candidate; donor winners
    and field routes are resolved collectively by its prepared field exchange.

    Serial affine planar triangle assemblies may opt into exact physical
    ownership by supplying ``partition_priority`` (highest priority first)
    and ``partition_sources``, mapping every declared wall part to its current
    certified authored planar source. Each source must have one cavity loop
    whose associated mesh edges exactly equal that part's wall scope. The
    resulting bound partition is rebuilt on motion and registration replay;
    whole-cell blanking remains conservative for donor search.
    """
    policy_ = OversetPolicy() if policy is None else policy
    if not isinstance(assembly, MeshAssembly) or not isinstance(policy_, OversetPolicy):
        raise TypeError("assembly and policy must be MeshAssembly and OversetPolicy.")
    if execution_group is not None and not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be ExecutionGroup or None.")
    global_names = assembly.global_part_names
    if len(global_names) < 2:
        raise ValueError(
            "Overset registration requires at least two global assembly parts."
        )
    if any(not isinstance(spec, OversetPartSpec) for spec in parts):
        raise TypeError("parts must contain OversetPartSpec values.")
    by_name = {spec.part_name: spec for spec in parts}
    if len(by_name) != len(parts) or set(by_name) != set(global_names):
        raise ValueError(
            "Overset registration requires one role specification per global assembly part."
        )
    specs = tuple(by_name[name] for name in global_names)
    if assembly.owner_local:
        if partition_priority is not None or partition_sources is not None:
            raise ValueError("Exact partition ownership is serial-only.")
        if execution_group is None:
            raise TypeError("Owner-local overset requires execution_group.")
        return _prepare_owner_local_connectivity(
            assembly, specs, policy_, execution_group, axis_name
        )
    if execution_group is not None:
        raise ValueError("execution_group is only valid for a rank-local mesh assembly.")
    if any(isinstance(link, OversetCoupling) for link in assembly.couplings):
        raise ValueError(
            "Remove previous overset overlays before preparing new connectivity."
        )
    from ..discretization._view_support import MappedFieldSupport

    for part, spec in zip(assembly.parts, specs, strict=True):
        result = _cell_result(part)
        if isinstance(spec.solid_query, MappedFieldSupport):
            spec.solid_query.require_locator(spec.solid_query.locator)
            if (
                spec.solid_query.mesh.mesh_id != result.mesh.mesh_id
                or spec.solid_query.geometry_id != cell_geometry_id(result.geometry)
            ):
                raise ValueError(
                    "Mapped source solid support must bind the exact registered carrier map."
                )
        if spec.excluded is not None:
            part.require_scope(spec.excluded)
            if spec.excluded.entity_dimension != result.mesh.topological_dimension:
                raise ValueError("Overset excluded scopes must identify cells.")
        for scope in (spec.wall, spec.boundary):
            if scope is not None:
                part.require_scope(scope)
        if (
            spec.wall is not None
            and spec.boundary is not None
            and np.intersect1d(
                np.asarray(spec.wall.entity_ids), np.asarray(spec.boundary.entity_ids)
            ).size
        ):
            raise ValueError(
                "Overset solid walls and artificial receptor boundaries must be disjoint."
            )
    return _prepare_connectivity(
        assembly.parts,
        assembly.couplings,
        specs,
        policy_,
        0,
        None,
        partition_priority,
        partition_sources,
    )


class _PacketRoute(StrictModule, NonTrainableState):
    packet: OversetDonorPacket
    query: PreparedFieldQuery
    coupling: OversetCoupling


class PreparedOversetFieldTransfer(StrictModule, NonTrainableState):
    connectivity: OversetConnectivity
    assembly: MeshAssembly
    packets: tuple[OversetDonorPacket, ...]
    receptors: tuple[OversetReceptorEvidence, ...]
    routes: tuple[_PacketRoute, ...]
    donor_names: tuple[str, ...] = eqx.field(static=True)
    coefficient_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    conservative: bool = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    owner_local: bool = eqx.field(static=True)
    local_reconstructions: tuple[tuple[str, PreparedFieldReconstruction], ...]
    value_action: OversetValueAction = eqx.field(static=True)

    def __init__(
        self,
        connectivity: OversetConnectivity,
        discretizations: Mapping[
            str, FiniteElementDiscretization | PreparedFieldReconstruction
        ],
        field_name: str,
        /,
        *,
        value_action: OversetValueAction | None = None,
    ) -> None:
        if not isinstance(connectivity, OversetConnectivity):
            raise TypeError("connectivity must be OversetConnectivity.")
        if not isinstance(discretizations, Mapping):
            raise TypeError("discretizations must map local donor parts to field owners.")
        if connectivity.owner_local:
            if (
                not isinstance(field_name, str)
                or not field_name
                or field_name.strip() != field_name
            ):
                raise ValueError(
                    "Owner-local overset field_name must be a non-empty exact identifier."
                )
            field_name_ = field_name
            local_parts = {part.name: part for part in connectivity.assembly.parts}
            if not set(discretizations).issubset(local_parts):
                raise ValueError(
                    "Owner-local field bindings must name local assembly parts."
                )
            local_bindings: dict[str, PreparedFieldReconstruction] = {}
            for name, binding in discretizations.items():
                if not isinstance(binding, PreparedFieldReconstruction):
                    raise TypeError(
                        "Owner-local field bindings require PreparedFieldReconstruction values."
                    )
                if binding.value_port.semantic_id != field_name_:
                    raise ValueError(
                        "Owner-local field reconstruction has a different field identity."
                    )
                part = local_parts[name]
                view = _owner_local_carrier(part)
                support = binding.support_geometry
                if not isinstance(support, OwnerLocalFieldSupport):
                    raise TypeError(
                        "Owner-local field reconstruction requires its canonical source support."
                    )
                support.require_kernel(binding.kernel)
                if (
                    support.local_view_id != view.local_view_id
                    or support.source_mesh_id != view.source_mesh_id
                    or support.source_topology_id != view.source_topology_id
                    or support.source_map_id != view.source_map_id
                    or support.source_context_id != view.source_context_id
                    or support.owner_directory_id != view.distribution_id
                ):
                    raise ValueError(
                        "Owner-local field reconstruction has stale source/map/context identity."
                    )
                if not binding.coefficient_linear:
                    raise ValueError(
                        "Owner-local overset field routes require coefficient-linear reconstruction."
                    )
                local_bindings[name] = binding
            shapes = {binding.value_shape for binding in local_bindings.values()}
            if len(shapes) > 1:
                raise ValueError(
                    "Owner-local donor reconstructions must share one value shape."
                )
            requested_action = value_action
            if requested_action is None and any(
                spec.image_rotation is not None for spec in connectivity.specs
            ):
                raise ValueError(
                    "Image-bound overset fields require an explicit value_action."
                )
            action = (
                "invariant"
                if requested_action is None
                else parse(requested_action, OversetValueAction, "value_action")
            )
            self.connectivity = connectivity
            self.assembly = connectivity.assembly
            self.packets = ()
            self.receptors = connectivity.receptors
            self.routes = ()
            self.donor_names = ()
            self.coefficient_shapes = ()
            self.value_shape = next(iter(shapes), ())
            self.field_name = field_name_
            self.conservative = False
            self.owner_local = True
            self.local_reconstructions = tuple(sorted(local_bindings.items()))
            self.value_action = action
            self.transfer_id = canonical_fingerprint(
                {
                    "kind": "owner-local-overset-field-transfer",
                    "connectivity": connectivity.connectivity_id,
                    "field": self.field_name,
                    "value_action": action,
                }
            )
            return
        connectivity.require_complete()
        self.owner_local = False
        donor_names = tuple(
            sorted({packet.donor_part for packet in connectivity.packets})
        )
        if set(discretizations) != set(donor_names):
            raise ValueError(
                "Field bindings must match the donor parts selected by connectivity."
            )
        layouts: dict[str, PreparedFieldReconstruction] = {}
        block_reconstructions: dict[tuple[str, str], PreparedFieldReconstruction] = {}
        for name in donor_names:
            part = connectivity.assembly.part(name)
            result = _cell_result(part)
            binding = discretizations[name]
            if isinstance(binding, FiniteElementDiscretization):
                if binding.mesh.mesh_id != result.mesh.mesh_id or not np.array_equal(
                    np.asarray(binding.default_runtime.coordinates),
                    np.asarray(result.geometry.coordinates),
                ):
                    raise ValueError(
                        "FE overset donor fields must bind the current mesh and coordinate geometry."
                    )
                field = binding._field_index(field_name)
                block_names = binding.dof_maps[field].block_names
                for packet in connectivity.packets:
                    if packet.donor_part == name:
                        if packet.donor_block not in block_names:
                            raise ValueError(
                                "The selected FE donor block has no field DOF map."
                            )
                        reconstruction = prepare_finite_element_field_reconstruction(
                            binding, field_name, block_name=packet.donor_block
                        )
                        block_reconstructions[name, packet.donor_block] = reconstruction
                layouts[name] = next(
                    iter(
                        block_reconstructions[(name, packet.donor_block)]
                        for packet in connectivity.packets
                        if packet.donor_part == name
                    )
                )
            elif isinstance(binding, PreparedFieldReconstruction):
                from ..discretization.fem._cell_map import PreparedFiniteElementCellMap
                from ..discretization.fem._point_interpolation import (
                    FiniteElementFieldReconstructionKernel,
                )
                from ..discretization.finite_volume._field_view import (
                    UnstructuredFiniteVolumeFieldReconstructionKernel,
                )

                kernel = binding.kernel
                if not isinstance(
                    kernel,
                    (
                        FiniteElementFieldReconstructionKernel,
                        UnstructuredFiniteVolumeFieldReconstructionKernel,
                    ),
                ):
                    raise TypeError(
                        "Overset prepared reconstructions require a cell-mesh FE or FV kernel."
                    )
                locator = kernel.locator
                cell_map = locator.cell_map
                if isinstance(cell_map, PreparedPolyhedralCellMap):
                    matches_current_mesh = cell_map.mesh.mesh_id == result.mesh.mesh_id
                elif isinstance(
                    cell_map, (PreparedFiniteElementCellMap, PreparedMappedCellMap)
                ):
                    matches_current_mesh = (
                        cell_map.topology_id == result.mesh.topology_id
                        and cell_map.geometry_layout_id
                        == result.geometry.geometry_layout_id
                        and np.array_equal(
                            np.asarray(locator.coordinates),
                            np.asarray(result.geometry.coordinates),
                        )
                    )
                else:
                    raise TypeError(
                        "Overset prepared reconstruction has no canonical CellMesh locator."
                    )
                if not matches_current_mesh:
                    raise ValueError(
                        "Prepared overset reconstruction must bind the current donor mesh and geometry."
                    )
                layouts[name] = binding
                for packet in connectivity.packets:
                    if packet.donor_part == name:
                        block_reconstructions[name, packet.donor_block] = binding
            else:
                raise TypeError(
                    "Overset donor fields require FE discretizations or prepared field reconstructions."
                )
        value_shapes = {value.value_shape for value in block_reconstructions.values()}
        if len(value_shapes) != 1:
            raise ValueError("All overset donor fields must share one value shape.")
        requested_action = value_action
        if requested_action is None and any(
            spec.image_rotation is not None for spec in connectivity.specs
        ):
            raise ValueError(
                "Image-bound overset fields require an explicit value_action."
            )
        action = "invariant" if requested_action is None else requested_action
        routes = []
        couplings = []
        for packet in connectivity.packets:
            reconstruction = block_reconstructions[packet.donor_part, packet.donor_block]
            query = reconstruction.prepare_query(packet.query_points)
            donor = connectivity.assembly.part(packet.donor_part)
            receptor = connectivity.assembly.part(packet.receptor_part)
            support = donor.scope(
                donor.intrinsic_dimension, np.unique(np.asarray(packet.donor_cells))
            )
            target_scope = receptor.scope(0, packet.receptor_ids)
            hole_ids = connectivity.blanking_of(receptor.name).vertex_ids_with(
                OversetVertexStatus.HOLE
            )
            hole_scope = receptor.scope(0, hole_ids) if hole_ids.size else None
            coupling = OversetCoupling.from_field_query(
                donor,
                receptor,
                support,
                target_scope,
                query,
                hole_scope=hole_scope,
                rotation=packet.rotation,
                translation=packet.translation,
                value_action=action,
            )
            routes.append(_PacketRoute(packet, query, coupling))
            couplings.append(coupling)
        retained = tuple(
            link
            for link in connectivity.assembly.couplings
            if not isinstance(link, OversetCoupling)
        )
        self.assembly = MeshAssembly(
            connectivity.assembly.parts, couplings=(*retained, *couplings)
        )
        # Receptor parts of the transfer are the parts that receive values;
        # the connectivity's evidence also records parts with no receptors.
        self.connectivity, self.packets, self.receptors = (
            connectivity,
            connectivity.packets,
            tuple(
                evidence
                for evidence in connectivity.receptors
                if evidence.receptor_ids.size
            ),
        )
        self.routes = tuple(routes)
        self.donor_names = donor_names
        self.coefficient_shapes = tuple(
            layouts[name].coefficient_shape for name in donor_names
        )
        self.value_shape = next(iter(value_shapes))
        self.field_name = str(field_name)
        self.conservative = False
        self.local_reconstructions = ()
        self.value_action = action
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "overset-field-transfer",
                "connectivity": connectivity.connectivity_id,
                "field": self.field_name,
                "queries": [route.query.query_id for route in routes],
                "couplings": [route.coupling.coupling_id for route in routes],
            }
        )

    @property
    def coefficient_linear(self) -> bool:
        if self.owner_local:
            return all(
                reconstruction.coefficient_linear
                for _, reconstruction in self.local_reconstructions
            )
        return all(route.query.coefficient_linear for route in self.routes)

    def apply(
        self, coefficients: Mapping[str, ConvertibleToArray], /
    ) -> dict[str, Array]:
        if self.owner_local:
            raise ValueError(
                "Owner-local field transfers require prepare_owner_local_exchange()."
            )
        if set(coefficients) != set(self.donor_names):
            raise ValueError("Coefficient keys must match prepared overset donor names.")
        values = {name: jnp.asarray(coefficients[name]) for name in self.donor_names}
        if any(
            values[name].shape != shape
            for name, shape in zip(self.donor_names, self.coefficient_shapes, strict=True)
        ):
            raise ValueError(
                "Overset coefficient arrays must match their prepared full layouts."
            )
        dtype = jnp.result_type(*(value.dtype for value in values.values()))
        output = {
            evidence.part_name: jnp.zeros(
                (evidence.receptor_ids.size, *self.value_shape), dtype=dtype
            )
            for evidence in self.receptors
        }
        for route in self.routes:
            packet = route.packet
            output[packet.receptor_part] = (
                output[packet.receptor_part]
                .at[packet.receptor_slots]
                .set(route.coupling.transfer(values[packet.donor_part]))
            )
        return output

    def transpose(
        self, cotangents: Mapping[str, ConvertibleToArray], /
    ) -> dict[str, Array]:
        """Apply the exact Euclidean transpose of the coefficient-linear field action."""
        if self.owner_local:
            raise ValueError(
                "Owner-local field transfers transpose through their prepared exchange."
            )
        if not self.coefficient_linear:
            raise ValueError(
                "Overset transpose requires coefficient-linear field queries."
            )
        expected_shapes = {
            evidence.part_name: (evidence.receptor_ids.size, *self.value_shape)
            for evidence in self.receptors
        }
        if set(cotangents) != set(expected_shapes):
            raise ValueError("Cotangents must name every receptor part exactly once.")
        values = {name: jnp.asarray(cotangents[name]) for name in expected_shapes}
        if any(values[name].shape != shape for name, shape in expected_shapes.items()):
            raise ValueError(
                "Overset cotangents must match their prepared receptor layouts."
            )
        dtype = jnp.result_type(*(value.dtype for value in values.values()))
        outputs = {
            name: jnp.zeros(shape, dtype=dtype)
            for name, shape in zip(self.donor_names, self.coefficient_shapes, strict=True)
        }
        for route in self.routes:
            packet = route.packet
            dual = values[packet.receptor_part][packet.receptor_slots]
            outputs[packet.donor_part] = outputs[
                packet.donor_part
            ] + route.coupling.transpose(dual)
        return outputs

    def duality_evidence(
        self,
        coefficients: Mapping[str, ConvertibleToArray],
        cotangents: Mapping[str, ConvertibleToArray],
        /,
        *,
        tolerance: float = 1e-10,
    ) -> Any:
        values, scattered = self.apply(coefficients), self.transpose(cotangents)
        return transpose_duality_evidence(
            jnp.concatenate([values[name].ravel() for name in sorted(values)]),
            jnp.concatenate(
                [jnp.asarray(cotangents[name]).ravel() for name in sorted(values)]
            ),
            jnp.concatenate(
                [jnp.asarray(coefficients[name]).ravel() for name in self.donor_names]
            ),
            jnp.concatenate([scattered[name].ravel() for name in self.donor_names]),
            tolerance=tolerance,
        )

    def prepare_owner_local_exchange(
        self,
        coefficient_ids: Mapping[str, ConvertibleToArray],
        execution_group: ExecutionGroup,
        owner_local_geometry: Mapping[str, OwnerLocalCellGeometry],
        /,
        *,
        axis_name: str = "overset",
    ) -> PreparedOwnerLocalOversetExchange:
        """Collectively resolve owner-local donors and prepare the sparse field route.

        Every execution-group process must call this in the same order. Stable
        coefficient IDs map each local reconstruction row, including halos;
        ``exchange.apply`` accepts only locally owned coefficient rows, keyed
        by part. The returned exchange exposes the resolved winner evidence as
        ``exchange.connectivity`` and its local stable input/output IDs.
        """
        from ._overset_exchange import PreparedOwnerLocalOversetExchange

        return PreparedOwnerLocalOversetExchange(
            self,
            coefficient_ids,
            execution_group,
            owner_local_geometry,
            axis_name=axis_name,
        )


@checked
def prepare_overset_field_transfer(
    connectivity: OversetConnectivity,
    discretizations: Mapping[
        str, FiniteElementDiscretization | PreparedFieldReconstruction
    ],
    field_name: str,
    /,
    *,
    value_action: OversetValueAction | None = None,
) -> PreparedOversetFieldTransfer:
    """Prepare interpolation on complete serial connectivity or local fields.

    Owner-local bindings contain only this rank's source-bound prepared
    reconstructions. The returned transfer must be resolved with
    ``prepare_owner_local_exchange`` before any field action.
    """
    return PreparedOversetFieldTransfer(
        connectivity, discretizations, field_name, value_action=value_action
    )


@final
class PreparedOversetConservativeRemap(StrictModule, NonTrainableState):
    """Complete physical common-refinement transfer between two cell parts."""

    refinement: PreparedCommonRefinement
    integration_evidence: CurvedOverlapIntegrationEvidence | None
    source_geometry_id: str = eqx.field(static=True)
    target_geometry_id: str = eqx.field(static=True)
    common_root_id: str | None = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: MeshPart,
        target: MeshPart,
        refinement: PreparedCommonRefinement,
        /,
        *,
        common_root_id: str | None,
        integration_evidence: CurvedOverlapIntegrationEvidence | None,
    ) -> None:
        source_result, target_result = _cell_result(source), _cell_result(target)
        if not isinstance(refinement, PreparedCommonRefinement):
            raise TypeError("refinement must be a PreparedCommonRefinement.")
        if (
            refinement.source_mesh_id != source_result.mesh.mesh_id
            or refinement.target_mesh_id != target_result.mesh.mesh_id
        ):
            raise ValueError(
                "Overset refinement does not bind the requested endpoint meshes."
            )
        evidence = refinement.evidence
        if common_root_id is None:
            if integration_evidence is not None:
                raise ValueError(
                    "Curved integration evidence requires its exact common-root witness ID."
                )
            integration_complete = True
        else:
            if (
                not isinstance(common_root_id, str)
                or not common_root_id
                or not isinstance(integration_evidence, CurvedOverlapIntegrationEvidence)
                or integration_evidence.witness_id != common_root_id
            ):
                raise ValueError(
                    "Curved integration evidence must bind the exact common-root witness."
                )
            integration_complete = (
                all(
                    Fraction(value) == 0
                    for value in integration_evidence.source_reference_defects
                )
                and all(
                    Fraction(value) == 0
                    for value in integration_evidence.target_reference_defects
                )
                and np.asarray(integration_evidence.unmatched_source_rows).size == 0
                and np.asarray(integration_evidence.unmatched_target_rows).size == 0
            )
        complete = (
            refinement.status is CommonRefinementStatus.SUCCESS
            and refinement.policy.coverage is CommonRefinementCoverage.COMPLETE
            and integration_complete
            and not any(
                (
                    evidence.source_gap_count,
                    evidence.target_gap_count,
                    evidence.source_double_count,
                    evidence.target_double_count,
                    evidence.uncertain_predicate_count,
                    evidence.invalid_cell_count,
                    evidence.intersection_failure_count,
                )
            )
        )
        self.refinement = refinement
        self.integration_evidence = integration_evidence
        self.source_geometry_id = cell_geometry_id(source_result.geometry)
        self.target_geometry_id = cell_geometry_id(target_result.geometry)
        self.common_root_id = common_root_id
        self.complete = complete
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "overset-physical-overlap-transfer",
                "refinement": refinement.refinement_id,
                "source_geometry": self.source_geometry_id,
                "target_geometry": self.target_geometry_id,
                "common_root": common_root_id,
                "integration": None
                if integration_evidence is None
                else integration_evidence.evidence_id,
            }
        )

    @property
    def evidence(self) -> CommonRefinementEvidence:
        return self.refinement.evidence

    def _require_complete(self, /) -> None:
        if not self.complete:
            raise ValueError(
                f"Overset common refinement is incomplete: {self.evidence.reason}"
            )

    def apply_extensive(self, content: ConvertibleToArray, /) -> Array:
        self._require_complete()
        return apply_physical_overlap_extensive(self.refinement, content)

    def apply_extensive_with_evidence(
        self, content: ConvertibleToArray, /
    ) -> PhysicalOverlapActionEvidence:
        self._require_complete()
        return apply_physical_overlap_extensive_with_evidence(self.refinement, content)

    def apply_extensive_transpose(self, cotangent: ConvertibleToArray, /) -> Array:
        self._require_complete()
        return transpose_physical_overlap_extensive(self.refinement, cotangent)

    def apply_intensive(self, average: ConvertibleToArray, /) -> Array:
        self._require_complete()
        return apply_physical_overlap_intensive(self.refinement, average)

    def apply_intensive_transpose(self, cotangent: ConvertibleToArray, /) -> Array:
        self._require_complete()
        return transpose_physical_overlap_intensive(self.refinement, cotangent)


@checked
def prepare_overset_conservative_remap(
    source: MeshPart,
    target: MeshPart,
    /,
    *,
    policy: CommonRefinementPolicy | None = None,
    common_root: CommonRootVolumeWitness | None = None,
    integration_policy: CurvedOverlapIntegrationPolicy | None = None,
) -> PreparedOversetConservativeRemap:
    source_result, target_result = _cell_result(source), _cell_result(target)
    policy_ = (
        CommonRefinementPolicy(coverage=CommonRefinementCoverage.COMPLETE)
        if policy is None
        else policy
    )
    if not isinstance(policy_, CommonRefinementPolicy):
        raise TypeError("policy must be a CommonRefinementPolicy.")
    if policy_.coverage is not CommonRefinementCoverage.COMPLETE:
        raise ValueError("Conservative overset transfer requires complete coverage.")
    if common_root is None:
        if integration_policy is not None:
            raise ValueError("integration_policy requires an exact common-root witness.")
        require_polyhedral_carrier_geometry(source_result.mesh, source_result.geometry)
        require_polyhedral_carrier_geometry(target_result.mesh, target_result.geometry)
        refinement = prepare_common_refinement(
            source_result.mesh, target_result.mesh, policy=policy_
        )
        curved_evidence = None
    else:
        integration_policy_ = (
            CurvedOverlapIntegrationPolicy()
            if integration_policy is None
            else integration_policy
        )
        refinement, curved_evidence = prepare_common_root_volume_refinement(
            source_result.mesh,
            source_result.geometry,
            target_result.mesh,
            target_result.geometry,
            common_root,
            policy=policy_,
            integration_policy=integration_policy_,
        )
    return PreparedOversetConservativeRemap(
        source,
        target,
        refinement,
        common_root_id=None if common_root is None else common_root.witness_id,
        integration_evidence=curved_evidence,
    )


class PreparedOversetMotion(StrictModule, NonTrainableState):
    previous: OversetConnectivity
    candidate: OversetConnectivity
    rebind: CompositionRebind

    def commit(self, /, *, accepted_boundary: bool) -> CompositionRebindReceipt:
        if not isinstance(accepted_boundary, bool):
            raise TypeError("accepted_boundary must be a host bool.")
        if (
            accepted_boundary
            and self.candidate.complete
            and self.candidate.partition_priority is not None
        ):
            self.candidate.require_complete()
        return commit_composition_rebind(
            self.rebind, accepted_boundary=accepted_boundary and self.candidate.complete
        )


@checked
def prepare_overset_motion_rebind(
    composition: Composition,
    previous: OversetConnectivity,
    candidate: OversetConnectivity,
    /,
    *,
    transports: tuple[CompositionTransport, ...] = (),
    reprepare: tuple[CompositionEntry, ...] = (),
    retain: tuple[str, ...] = (),
    invalidate: tuple[str, ...] = (),
) -> PreparedOversetMotion:
    if (
        candidate.epoch != previous.epoch + 1
        or candidate.predecessor_id != previous.connectivity_id
    ):
        raise ValueError(
            "Candidate overset motion must advance exactly one epoch from the supplied predecessor."
        )
    old_entries, new_entries = (
        previous.composition_entries(),
        candidate.composition_entries(),
    )
    own_retain, own_reprepare = [], []
    for old, new in zip(old_entries, new_entries, strict=True):
        if composition.entry(old.entry_id).record_id != old.record_id:
            raise ValueError(
                "Composition does not contain the accepted overset registration."
            )
        (own_retain if old.record_id == new.record_id else own_reprepare).append(
            old.entry_id if old.record_id == new.record_id else new
        )
    return PreparedOversetMotion(
        previous,
        candidate,
        CompositionRebind(
            composition,
            retain=(*own_retain, *retain),
            reprepare=(*own_reprepare, *reprepare),
            transports=transports,
            invalidate=invalidate,
        ),
    )


class OversetRegistration(StrictModule, NonTrainableState):
    assembly: MeshAssembly
    specs: tuple[OversetPartSpec, ...]
    policy: OversetPolicy
    epoch: int = eqx.field(static=True)
    predecessor_id: str | None = eqx.field(static=True)
    connectivity_id: str = eqx.field(static=True)
    partition_priority: tuple[str, ...] | None = eqx.field(static=True)
    partition_sources: Mapping[str, Any]

    def __init__(
        self,
        assembly: MeshAssembly,
        specs: tuple[OversetPartSpec, ...],
        policy: OversetPolicy,
        epoch: int,
        predecessor_id: str | None,
        connectivity_id: str,
        /,
        *,
        partition_priority: tuple[str, ...] | None = None,
        partition_sources: Mapping[str, Any] | None = None,
    ) -> None:
        self.assembly, self.specs, self.policy = assembly, specs, policy
        self.epoch, self.predecessor_id, self.connectivity_id = (
            int(epoch),
            predecessor_id,
            str(connectivity_id),
        )
        self.partition_priority = partition_priority
        self.partition_sources = (
            {} if partition_sources is None else dict(partition_sources)
        )

    def prepare(self, /) -> OversetConnectivity:
        result = _prepare_connectivity(
            self.assembly.parts,
            self.assembly.couplings,
            self.specs,
            self.policy,
            self.epoch,
            self.predecessor_id,
            self.partition_priority,
            self.partition_sources,
        )
        if result.connectivity_id != self.connectivity_id:
            raise ValueError(
                "Restored overset connectivity identity does not match its registration."
            )
        return result


__all__ = [
    "OversetConnectivity",
    "OversetConnectivityError",
    "OversetDonorPacket",
    "OversetPartBlanking",
    "OversetPartSpec",
    "OversetPolicy",
    "OversetReceptorEvidence",
    "OversetRegistration",
    "OversetVertexStatus",
    "OversetCellStatus",
    "PreparedOversetFieldTransfer",
    "PreparedOversetConservativeRemap",
    "PreparedOversetMotion",
    "prepare_overset_connectivity",
    "prepare_overset_field_transfer",
    "prepare_overset_conservative_remap",
    "prepare_overset_motion_rebind",
]
