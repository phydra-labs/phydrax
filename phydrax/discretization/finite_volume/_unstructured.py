#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ...linalg import ArraySpace, DiagonalPairing
from ...sparse import EdgeRelation, SparseLinearMap
from ...typing import checked
from .._adaptive_simplex import MaskedSimplexMesh
from .._cell_complex import (
    polygonal_connectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from .._cell_mesh import CellBlock, CellMesh
from .._conservation_ledger import (
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._integration_domain import IntegrationDomain
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._side_actions import FacetTraceRule, PreparedTraceAction, SideTraceQuantity
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import CellComplexTopology, EntitySelection
from .._views import FieldTraceSide
from ._geometry_protocol import FiniteVolumeFaceBlock
from ._structured import _component_names


if TYPE_CHECKING:
    from .._view_support import OwnerLocalCellGeometry, OwnerLocalCompositeCellGeometry
    from ._side_trace import FiniteVolumeFaceReconstruction, PreparedNonlinearFaceTrace


Connectivity = PolygonalConnectivity | TetrahedralConnectivity | PolyhedralConnectivity

_TETRAHEDRAL_FACE_QUADRATURE_BARYCENTRIC = (
    (0.445948490915965, 0.445948490915965, 0.108103018168070),
    (0.445948490915965, 0.108103018168070, 0.445948490915965),
    (0.108103018168070, 0.445948490915965, 0.445948490915965),
    (0.091576213509771, 0.091576213509771, 0.816847572980459),
    (0.091576213509771, 0.816847572980459, 0.091576213509771),
    (0.816847572980459, 0.091576213509771, 0.091576213509771),
)
_TETRAHEDRAL_FACE_QUADRATURE_NORMALIZED_WEIGHTS = (
    0.223381589678011,
    0.223381589678011,
    0.223381589678011,
    0.109951743655322,
    0.109951743655322,
    0.109951743655322,
)


def _stable_global_ids(name: str, value: ArrayLike | None, count: int, /) -> np.ndarray:
    if value is None:
        identifiers = np.arange(count, dtype=np.int64)
    else:
        raw = np.asarray(value)
        if raw.shape != (count,) or raw.dtype.kind not in "iu":
            raise ValueError(f"{name} must contain one integer ID per entity.")
        if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
            raise ValueError(f"{name} must be representable as signed int64.")
        identifiers = raw.astype(np.int64, copy=False)
    if np.any(identifiers < 0) or np.unique(identifiers).size != count:
        raise ValueError(f"{name} must contain unique nonnegative IDs.")
    if not x64_enabled() and np.any(identifiers > np.iinfo(np.int32).max):
        raise ValueError(f"{name} must fit signed int32 when JAX x64 is disabled.")
    return identifiers


def _cross_2d(left: np.ndarray, right: np.ndarray, /) -> np.ndarray:
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def _normalized_triangles(vertices: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    if cells.shape[0] == 0:
        return cells
    points = vertices[cells]
    signed_twice_area = _cross_2d(
        points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]
    )
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=-1), axis=1)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale**2
    if np.any(~np.isfinite(signed_twice_area)) or np.any(
        np.abs(signed_twice_area) <= tolerance
    ):
        raise ValueError("Triangle cells must have positive finite nonzero area.")
    normalized = cells.copy()
    reverse = signed_twice_area < 0.0
    normalized[reverse, 1], normalized[reverse, 2] = (
        normalized[reverse, 2].copy(),
        normalized[reverse, 1].copy(),
    )
    if np.unique(np.sort(normalized, axis=1), axis=0).shape[0] != normalized.shape[0]:
        raise ValueError("Triangle cells contain duplicates.")
    return normalized


def _validated_embedded_triangles(
    vertices: np.ndarray, cells: np.ndarray, /
) -> np.ndarray:
    if cells.shape[0] == 0:
        return cells
    if cells.ndim != 2 or cells.shape[1] != 3:
        raise ValueError("Embedded surface triangles must have shape (n, 3).")
    if np.any(cells < 0) or np.any(cells >= vertices.shape[0]):
        raise ValueError("Embedded surface triangle vertices are outside the mesh.")
    points = vertices[cells]
    first = points[:, 1] - points[:, 0]
    second = points[:, 2] - points[:, 0]
    double_area = np.linalg.norm(np.cross(first, second), axis=-1)
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=-1), axis=1)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale**2
    if np.any(~np.isfinite(double_area)) or np.any(double_area <= tolerance):
        raise ValueError("Embedded surface triangles require positive finite area.")
    if np.unique(np.sort(cells, axis=1), axis=0).shape[0] != cells.shape[0]:
        raise ValueError("Embedded surface triangles contain duplicates.")
    return cells.copy()


def _quadrilateral_shape_data(points: np.ndarray, /) -> Any:
    xi = points[:, 0]
    eta = points[:, 1]
    shape = 0.25 * np.stack(
        (
            (1.0 - xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 - eta),
            (1.0 + xi) * (1.0 + eta),
            (1.0 - xi) * (1.0 + eta),
        ),
        axis=-1,
    )
    gradient = 0.25 * np.stack(
        (
            np.stack((-(1.0 - eta), -(1.0 - xi)), axis=-1),
            np.stack((1.0 - eta, -(1.0 + xi)), axis=-1),
            np.stack((1.0 + eta, 1.0 + xi), axis=-1),
            np.stack((-(1.0 + eta), 1.0 - xi), axis=-1),
        ),
        axis=1,
    )
    return shape, gradient


def _cell_volume_quadrature(
    plan: "UnstructuredFiniteVolumePlan",
    connectivity: Connectivity,
    /,
) -> tuple[Array, Array, Array]:
    nodes_host, weights_host = np.polynomial.legendre.leggauss(4)
    unit_nodes = 0.5 * (nodes_host + 1.0)
    unit_weights = 0.5 * weights_host
    points = jnp.asarray(plan.vertices)
    dtype = points.dtype
    if isinstance(connectivity, PolygonalConnectivity):
        first, second = np.meshgrid(unit_nodes, unit_nodes, indexing="ij")
        first_weight, second_weight = np.meshgrid(
            unit_weights, unit_weights, indexing="ij"
        )
        u = jnp.asarray(first.reshape((-1,)), dtype=dtype)
        v = jnp.asarray(second.reshape((-1,)), dtype=dtype)
        reference_weight = jnp.asarray(
            (first_weight * second_weight).reshape((-1,)), dtype=dtype
        )
        triangle_points = points[jnp.asarray(plan.triangles, dtype=jnp.int32)]
        triangle_quadrature = (
            triangle_points[:, :1, :]
            + u[None, :, None] * (triangle_points[:, 1:2, :] - triangle_points[:, :1, :])
            + ((1.0 - u) * v)[None, :, None]
            * (triangle_points[:, 2:3, :] - triangle_points[:, :1, :])
        )
        first_tangent = triangle_points[:, 1] - triangle_points[:, 0]
        second_tangent = triangle_points[:, 2] - triangle_points[:, 0]
        if points.shape[-1] == 2:
            triangle_jacobian = first_tangent[:, 0] * second_tangent[:, 1] - (
                first_tangent[:, 1] * second_tangent[:, 0]
            )
        else:
            triangle_jacobian = jnp.linalg.norm(
                jnp.cross(first_tangent, second_tangent), axis=-1
            )
        triangle_weights = (
            triangle_jacobian[:, None] * (1.0 - u)[None, :] * reference_weight[None, :]
        )

        reference = np.stack(
            (
                2.0 * first.reshape((-1,)) - 1.0,
                2.0 * second.reshape((-1,)) - 1.0,
            ),
            axis=-1,
        )
        shape_host, gradient_host = _quadrilateral_shape_data(reference)
        shape = jnp.asarray(shape_host, dtype=dtype)
        gradient = jnp.asarray(gradient_host, dtype=dtype)
        quadrilateral_points = points[jnp.asarray(plan.quadrilaterals, dtype=jnp.int32)]
        quadrilateral_quadrature = ein.contract(
            "qv,cvd->cqd", shape, quadrilateral_points
        )
        jacobian = ein.contract("qva,cvd->cqad", gradient, quadrilateral_points)
        determinant = (
            jacobian[..., 0, 0] * jacobian[..., 1, 1]
            - jacobian[..., 0, 1] * jacobian[..., 1, 0]
        )
        tensor_weight = jnp.asarray(
            (weights_host[:, None] * weights_host[None, :]).reshape((-1,)),
            dtype=dtype,
        )
        quadrilateral_weights = determinant * tensor_weight[None, :]
        quadrature_points = jnp.concatenate(
            (triangle_quadrature, quadrilateral_quadrature), axis=0
        )
        quadrature_weights = jnp.concatenate(
            (triangle_weights, quadrilateral_weights), axis=0
        )
    else:
        first, second, third = np.meshgrid(
            unit_nodes, unit_nodes, unit_nodes, indexing="ij"
        )
        first_weight, second_weight, third_weight = np.meshgrid(
            unit_weights, unit_weights, unit_weights, indexing="ij"
        )
        u = jnp.asarray(first.reshape((-1,)), dtype=dtype)
        v = jnp.asarray(second.reshape((-1,)), dtype=dtype)
        w = jnp.asarray(third.reshape((-1,)), dtype=dtype)
        reference_weight = jnp.asarray(
            (first_weight * second_weight * third_weight).reshape((-1,)),
            dtype=dtype,
        )
        tetrahedron_points = points[jnp.asarray(plan.tetrahedra, dtype=jnp.int32)]
        quadrature_points = (
            tetrahedron_points[:, :1, :]
            + u[None, :, None]
            * (tetrahedron_points[:, 1:2, :] - tetrahedron_points[:, :1, :])
            + ((1.0 - u) * v)[None, :, None]
            * (tetrahedron_points[:, 2:3, :] - tetrahedron_points[:, :1, :])
            + ((1.0 - u) * (1.0 - v) * w)[None, :, None]
            * (tetrahedron_points[:, 3:4, :] - tetrahedron_points[:, :1, :])
        )
        determinant = jnp.linalg.det(
            jnp.stack(
                (
                    tetrahedron_points[:, 1] - tetrahedron_points[:, 0],
                    tetrahedron_points[:, 2] - tetrahedron_points[:, 0],
                    tetrahedron_points[:, 3] - tetrahedron_points[:, 0],
                ),
                axis=-1,
            )
        )
        quadrature_weights = (
            determinant[:, None]
            * ((1.0 - u) ** 2 * (1.0 - v))[None, :]
            * reference_weight[None, :]
        )
    quadrature_weights = eqx.error_if(
        quadrature_weights,
        jnp.any(~jnp.isfinite(quadrature_weights) | (quadrature_weights <= 0.0)),
        "Unstructured cell quadrature requires positive finite weights.",
    )
    return (
        quadrature_points,
        quadrature_weights,
        jnp.ones(quadrature_weights.shape, dtype=jnp.bool_),
    )


def _quadrilateral_jacobian_determinants(
    vertices: np.ndarray, cells: np.ndarray, reference_points: np.ndarray, /
) -> np.ndarray:
    _, gradient = _quadrilateral_shape_data(reference_points)
    jacobian = ein.contract("qva,cvd->cqad", gradient, vertices[cells])
    return (
        jacobian[..., 0, 0] * jacobian[..., 1, 1]
        - jacobian[..., 0, 1] * jacobian[..., 1, 0]
    )


def _normalized_quadrilaterals(vertices: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    if cells.shape[0] == 0:
        return cells
    normalized = cells.copy()
    points = vertices[normalized]
    signed_twice_area = np.sum(_cross_2d(points, np.roll(points, -1, axis=1)), axis=1)
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=-1), axis=1)
    tolerance = 64.0 * np.finfo(np.float64).eps * scale**2
    if np.any(~np.isfinite(signed_twice_area)) or np.any(
        np.abs(signed_twice_area) <= tolerance
    ):
        raise ValueError("Quadrilateral cells must have nonzero finite signed area.")
    reverse = signed_twice_area < 0.0
    normalized[reverse] = normalized[reverse][:, [0, 3, 2, 1]]
    points = vertices[normalized]
    turns = _cross_2d(
        np.roll(points, -1, axis=1) - points,
        np.roll(points, -2, axis=1) - np.roll(points, -1, axis=1),
    )
    if np.any(turns <= tolerance[:, None]):
        raise ValueError("Quadrilateral cells must be simple and strictly convex.")
    reference = np.asarray(
        [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (-1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
            (1.0 / np.sqrt(3.0), -1.0 / np.sqrt(3.0)),
            (1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
            (-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)),
        ]
    )
    determinant = _quadrilateral_jacobian_determinants(vertices, normalized, reference)
    jacobian_tolerance = tolerance[:, None] / 4.0
    if np.any(~np.isfinite(determinant)) or np.any(determinant <= jacobian_tolerance):
        raise ValueError("Quadrilateral bilinear maps require positive finite Jacobians.")
    if np.unique(np.sort(normalized, axis=1), axis=0).shape[0] != normalized.shape[0]:
        raise ValueError("Quadrilateral cells contain duplicates.")
    return normalized


def _normalized_tetrahedra(vertices: np.ndarray, cells: np.ndarray, /) -> np.ndarray:
    if cells.shape[0] == 0:
        return cells
    points = vertices[cells]
    matrix = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=-1,
    )
    determinant = np.linalg.det(matrix)
    scale = np.max(np.linalg.norm(points - points[:, :1], axis=-1), axis=1)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale**3
    if np.any(~np.isfinite(determinant)) or np.any(np.abs(determinant) <= tolerance):
        raise ValueError("Tetrahedral cells must have nonzero finite volume.")
    normalized = cells.copy()
    reverse = determinant < 0.0
    normalized[reverse, 1], normalized[reverse, 2] = (
        normalized[reverse, 2].copy(),
        normalized[reverse, 1].copy(),
    )
    if np.unique(np.sort(normalized, axis=1), axis=0).shape[0] != normalized.shape[0]:
        raise ValueError("Tetrahedral cells contain duplicates.")
    return normalized


def _owner_neighbor(connectivity: Connectivity, cell_count: int, /) -> Any:
    if isinstance(connectivity, PolygonalConnectivity):
        cell_faces = np.asarray(connectivity.cell_edges, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_edge_signs)
        valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
        face_count = connectivity.edges.shape[0]
    elif isinstance(connectivity, TetrahedralConnectivity):
        cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_face_signs)
        valid = np.ones(cell_faces.shape, dtype=np.bool_)
        face_count = connectivity.faces.shape[0]
    elif isinstance(connectivity, PolyhedralConnectivity):
        cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_face_sign_values)
        cell_ids = np.repeat(
            np.arange(cell_count, dtype=np.int32),
            np.diff(np.asarray(connectivity.cell_face_offsets, dtype=np.int32)),
        )
        owner = np.asarray(connectivity.face_owner, dtype=np.int32)
        neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32)
        owner_sign = np.zeros((connectivity.face_count,), dtype=np.float64)
        for cell, face, sign in zip(cell_ids, cell_faces, cell_signs, strict=True):
            if owner[int(face)] == int(cell):
                owner_sign[int(face)] = float(sign)
        return owner, neighbor, owner_sign
    else:
        cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_face_signs)
        valid = np.ones(cell_faces.shape, dtype=np.bool_)
        face_count = connectivity.faces.shape[0]
    owner = np.full((face_count,), -1, dtype=np.int32)
    neighbor = np.full((face_count,), -1, dtype=np.int32)
    owner_sign = np.zeros((face_count,), dtype=np.float64)
    for cell in range(cell_count):
        for local in range(cell_faces.shape[1]):
            if not valid[cell, local]:
                continue
            face = int(cell_faces[cell, local])
            sign = float(cell_signs[cell, local])
            if owner[face] < 0:
                owner[face] = cell
                owner_sign[face] = sign
            else:
                if neighbor[face] >= 0:
                    raise ValueError(
                        "Unstructured cells must be codimension-one manifold."
                    )
                if sign == owner_sign[face]:
                    raise ValueError(
                        "Shared faces must have opposite incidence orientation."
                    )
                neighbor[face] = cell
    if np.any(owner < 0):
        raise ValueError("Every unstructured face must have an owner cell.")
    return owner, neighbor, owner_sign


_PERIODIC_SEAM_PATCH = "__periodic_seam__"


@dataclass(frozen=True, slots=True)
class _PeriodicVertexFrames:
    """Quotient identity of every local vertex row of one owner-local carrier.

    ``parts[row]`` is the periodic part index, or ``-1`` for rows of
    nonperiodic parts and remote ghosts, which never join across seams.
    ``roots`` are quotient vertex identities and ``windings`` the integer
    lattice image of each lifted row; ``lifts[part]`` maps a winding to its
    ambient translation (rows are lattice generators).
    """

    parts: np.ndarray
    roots: np.ndarray
    windings: tuple[np.ndarray | None, ...]
    lifts: tuple[np.ndarray | None, ...]


def _owner_local_periodic_vertex_frames(
    view: Any, vertex_ids: np.ndarray, /
) -> _PeriodicVertexFrames | None:
    from .._view_support import OwnerLocalCompositeCellGeometry

    parts = view.parts if isinstance(view, OwnerLocalCompositeCellGeometry) else (view,)
    row_by_id = {int(identifier): row for row, identifier in enumerate(vertex_ids)}
    part_rows = np.full(vertex_ids.shape, -1, dtype=np.int64)
    roots = np.full(vertex_ids.shape, -1, dtype=np.int64)
    windings: list[np.ndarray | None] = [None] * vertex_ids.size
    lifts: list[np.ndarray | None] = []
    periodic = False
    for index, part in enumerate(parts):
        if part.periodic_topology_id is None:
            lifts.append(None)
            continue
        periodic = True
        if not part.coordinates_are_lifted:
            raise ValueError("Owner-local periodic FV requires lifted local coordinates.")
        if part.vertex_representative_ids is None or part.vertex_windings is None:
            raise ValueError(
                "Owner-local periodic FV requires quotient vertex roots and windings."
            )
        part_ids = np.asarray(part.vertex_ids, dtype=np.int64)
        part_roots = np.asarray(part.vertex_representative_ids, dtype=np.int64)
        part_windings = np.asarray(part.vertex_windings, dtype=np.int64)
        if part_roots.shape != part_ids.shape or part_windings.shape[0] != part_ids.size:
            raise ValueError("Owner-local periodic vertex ancestry is misaligned.")
        lift = part.periodic_lift_matrix
        lifts.append(None if lift is None else np.asarray(lift, dtype=np.float64))
        for identifier, root, winding in zip(
            part_ids, part_roots, part_windings, strict=True
        ):
            row = row_by_id.get(int(identifier))
            if row is None:
                raise ValueError(
                    "Owner-local periodic FV carrier lost a resident vertex."
                )
            part_rows[row] = index
            roots[row] = int(root)
            windings[row] = winding
    if not periodic:
        return None
    return _PeriodicVertexFrames(part_rows, roots, tuple(windings), tuple(lifts))


def _periodic_seam_joins(
    face_vertex_rows: Sequence[np.ndarray] | np.ndarray,
    boundary_faces: np.ndarray,
    face_owner_owned: np.ndarray,
    coordinates: np.ndarray,
    frames: _PeriodicVertexFrames,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Join unwelded seam faces of a lifted periodic patch by quotient identity.

    A lifted face is identified by the quotient roots of its vertices and their
    windings relative to one anchor vertex, minimized over anchors; two
    boundary faces with the same identity in one periodic part are the two
    sides of one quotient face, resident at lattice images differing by the
    integer shift ``k``. The kept face (the owned side when exactly one side is
    owned) carries the flux; ``translations[j]`` is the ambient image of ``k``,
    so a neighbor-side point ``p`` in the kept owner's frame is ``p + t`` in the
    neighbor's own lifted frame. Returns ``(joins, translations, unmatched)``:
    ``joins[j] = (kept, merged)`` topology faces and the periodic boundary faces
    whose partner is not resident.
    """
    dimension = coordinates.shape[1]
    scale = max(1.0, float(np.max(np.abs(coordinates)))) if coordinates.size else 1.0
    tolerance = 1.0e-9 * scale
    groups: dict[tuple[Any, ...], list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for face in np.asarray(boundary_faces, dtype=np.int64).tolist():
        rows = np.asarray(face_vertex_rows[face], dtype=np.int64)
        parts = frames.parts[rows]
        if np.all(parts < 0):
            continue
        if np.any(parts != parts[0]):
            raise ValueError("A local face mixes periodic part frames.")
        best: tuple[tuple[Any, ...], np.ndarray, np.ndarray] | None = None
        for anchor in rows.tolist():
            anchor_winding = frames.windings[anchor]
            if anchor_winding is None:
                raise ValueError("Owner-local periodic vertex lacks its winding.")
            entries = sorted(
                (
                    int(frames.roots[row]),
                    tuple(
                        int(value)
                        for value in np.asarray(frames.windings[row]) - anchor_winding
                    ),
                    int(row),
                )
                for row in rows.tolist()
            )
            signature = tuple((root, relative) for root, relative, _ in entries)
            if best is None or signature < best[0]:
                best = (
                    signature,
                    np.asarray(anchor_winding, dtype=np.int64),
                    np.asarray([row for *_, row in entries], dtype=np.int64),
                )
        if best is None:
            continue
        groups.setdefault((int(parts[0]), best[0]), []).append(
            (int(face), best[1], best[2])
        )
    joins: list[tuple[int, int]] = []
    translations: list[np.ndarray] = []
    unmatched: list[int] = []
    for (part, _), members in sorted(groups.items(), key=lambda item: item[1][0][0]):
        if len(members) == 1:
            unmatched.append(members[0][0])
            continue
        if len(members) != 2:
            raise ValueError(
                "A quotient face is resident as more than two unwelded lifted faces."
            )
        first, second = sorted(members, key=lambda member: member[0])
        if bool(face_owner_owned[second[0]]) and not bool(face_owner_owned[first[0]]):
            first, second = second, first
        shift = second[1] - first[1]
        if not np.any(shift):
            raise ValueError(
                "Unwelded periodic seam partners must sit at distinct lattice images."
            )
        offsets = coordinates[second[2]] - coordinates[first[2]]
        lift = frames.lifts[part]
        translation = (
            np.mean(offsets, axis=0)
            if lift is None
            else shift.astype(np.float64) @ lift[:, :dimension]
        )
        if np.max(np.abs(offsets - translation[None, :])) > tolerance:
            raise ValueError(
                "Periodic seam partners are not exact lattice translates of each other."
            )
        joins.append((first[0], second[0]))
        translations.append(translation)
    return (
        np.asarray(joins, dtype=np.int32).reshape((-1, 2)),
        np.asarray(translations, dtype=np.float64).reshape((-1, dimension)),
        np.asarray(sorted(unmatched), dtype=np.int32),
    )


def _classify_owner_local_faces(
    face_vertex_rows: Sequence[np.ndarray] | np.ndarray,
    boundary_faces: np.ndarray,
    face_owner_owned: np.ndarray,
    coordinates: np.ndarray,
    frames: _PeriodicVertexFrames | None,
    /,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Boundary face groups plus periodic seam joins of one owner-local carrier.

    Owned boundary faces of a nonperiodic part are physical; boundary faces of
    ghost cells are artificial halo cuts. Periodic parts have no physical
    boundary: every seam face of an owned cell must join its resident partner.
    """
    boundary = np.asarray(boundary_faces, dtype=np.int32)
    if frames is None:
        joins = np.empty((0, 2), dtype=np.int32)
        translations = np.empty((0, coordinates.shape[1]), dtype=np.float64)
        periodic_boundary = np.zeros(boundary.shape, dtype=np.bool_)
    else:
        joins, translations, unmatched = _periodic_seam_joins(
            face_vertex_rows, boundary, face_owner_owned, coordinates, frames
        )
        periodic_boundary = np.asarray(
            [
                bool(np.any(frames.parts[np.asarray(face_vertex_rows[face])] >= 0))
                for face in boundary.tolist()
            ],
            dtype=np.bool_,
        )
        if np.any(face_owner_owned[unmatched]):
            raise ValueError(
                "Owner-local periodic FV view omits the resident periodic partner "
                "of an owned seam face."
            )
    joined = set(int(value) for value in joins.reshape(-1))
    owned = face_owner_owned[boundary]
    physical = boundary[owned & ~periodic_boundary]
    artificial = np.asarray(
        [face for face in boundary[~owned].tolist() if face not in joined],
        dtype=np.int32,
    )
    groups = {
        "__physical_boundary__": physical,
        "__owner_local_halo__": artificial,
    }
    if joins.size:
        groups[_PERIODIC_SEAM_PATCH] = np.sort(joins.reshape(-1)).astype(np.int32)
    return groups, joins, translations


def periodic_image_scale(coordinates: ArrayLike, /) -> float:
    """Length scale of the rounding that identifies periodic lattice images."""
    values = np.asarray(coordinates, dtype=np.float64)
    return max(1.0, float(np.max(np.abs(values)))) if values.size else 1.0


def periodic_image_key(image: ArrayLike, scale: float, /) -> tuple[int, ...]:
    """Total order of lattice image translations: the identity, then lexicographic.

    Translations are summed seam offsets, so they are rounded at ``1e-9 * scale``;
    distinct lattice images differ by a lattice vector, far above that tolerance.
    """
    rounded = np.rint(np.asarray(image, dtype=np.float64) / (1.0e-9 * scale)).astype(
        np.int64
    )
    return (int(np.any(rounded != 0)), *(int(value) for value in rounded))


def periodic_seam_vertex_pairs(
    plan: Any, joins: np.ndarray, translations: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertex rows `(kept, twin, translation)` of every joined seam face pair.

    The twin face's vertex `twin[i]` is the kept face's vertex `kept[i]`
    resident at the neighbor's lattice image: its coordinates are
    `coordinates[kept[i]] + translation[i]`. Matching is done once on the
    plan's own coordinates; motions must preserve it.
    """
    connectivity = None if plan.mesh is None else plan.mesh.connectivity
    if isinstance(connectivity, PolygonalConnectivity):
        face_vertices = np.asarray(connectivity.edges, dtype=np.int64)
    elif isinstance(connectivity, TetrahedralConnectivity):
        face_vertices = np.asarray(connectivity.faces, dtype=np.int64)
    else:
        raise TypeError("Periodic seam vertex pairs need edge or triangle faces.")
    points = np.asarray(plan.vertices, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(points)))) if points.size else 1.0
    kept_vertices = face_vertices[joins[:, 0]]
    twin_vertices = face_vertices[joins[:, 1]]
    distance = np.linalg.norm(
        points[twin_vertices][:, :, None, :]
        - translations[:, None, None, :]
        - points[kept_vertices][:, None, :, :],
        axis=-1,
    )
    order = np.argmin(distance, axis=-1)
    if np.any(np.min(distance, axis=-1) > 1.0e-9 * scale) or np.any(
        np.sort(order, axis=-1) != np.arange(order.shape[1])[None, :]
    ):
        raise ValueError("Periodic seam twins are not lattice translates of each other.")
    kept = np.take_along_axis(kept_vertices, order, axis=1)
    pair_translations = np.broadcast_to(
        translations[:, None, :], twin_vertices.shape + (translations.shape[1],)
    )
    return (
        kept.reshape(-1),
        twin_vertices.reshape(-1),
        pair_translations.reshape((-1, translations.shape[1])),
    )


def _triangle_measures(cell_points: Array, /) -> tuple[Array, Array]:
    """Signed areas (positive counterclockwise) and centroids of ``(..., 3, 2)``."""
    cross = (cell_points[..., 1, 0] - cell_points[..., 0, 0]) * (
        cell_points[..., 2, 1] - cell_points[..., 0, 1]
    ) - (cell_points[..., 1, 1] - cell_points[..., 0, 1]) * (
        cell_points[..., 2, 0] - cell_points[..., 0, 0]
    )
    return 0.5 * cross, jnp.mean(cell_points, axis=-2)


def _tetrahedron_measures(cell_points: Array, /) -> tuple[Array, Array]:
    """Signed volumes (positive right-handed) and centroids of ``(..., 4, 3)``."""
    determinant = jnp.linalg.det(
        jnp.stack(
            (
                cell_points[..., 1, :] - cell_points[..., 0, :],
                cell_points[..., 2, :] - cell_points[..., 0, :],
                cell_points[..., 3, :] - cell_points[..., 0, :],
            ),
            axis=-1,
        )
    )
    return determinant / 6.0, jnp.mean(cell_points, axis=-2)


def _edge_area_vectors(tangent: Array, /) -> Array:
    """Edge normals ``(t_y, -t_x)``: outward for counterclockwise traversal."""
    return jnp.stack((tangent[..., 1], -tangent[..., 0]), axis=-1)


def _triangle_area_vectors(face_points: Array, /) -> Array:
    """Right-handed area vectors of ``(..., 3, 3)`` triangular faces."""
    return 0.5 * jnp.cross(
        face_points[..., 1, :] - face_points[..., 0, :],
        face_points[..., 2, :] - face_points[..., 0, :],
    )


def _embedded_triangle_geometry(
    vertices: ArrayLike,
    triangles: ArrayLike,
    connectivity: PolygonalConnectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
) -> Any:
    points = jnp.asarray(vertices)
    cells = jnp.asarray(triangles, dtype=jnp.int32)
    cell_points = points[cells]
    first_tangent = cell_points[:, 1] - cell_points[:, 0]
    second_tangent = cell_points[:, 2] - cell_points[:, 0]
    normal_area = jnp.cross(first_tangent, second_tangent)
    double_area = jnp.linalg.norm(normal_area, axis=-1)
    cell_areas = eqx.error_if(
        0.5 * double_area,
        jnp.any(~jnp.isfinite(double_area) | (double_area <= 0.0)),
        "Embedded surface triangles require positive finite areas.",
    )
    normals = normal_area / double_area[:, None]
    cell_centers = jnp.mean(cell_points, axis=1)
    edges = jnp.asarray(connectivity.edges, dtype=jnp.int32)
    edge_points = points[edges]
    face_centers = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
    tangent = edge_points[:, 1] - edge_points[:, 0]
    owners = jnp.asarray(owner, dtype=jnp.int32)
    canonical_area = jnp.cross(tangent, normals[owners])
    area_vectors = jnp.asarray(owner_sign, dtype=points.dtype)[:, None] * canonical_area
    face_measures = jnp.linalg.norm(area_vectors, axis=-1)
    owner_centers = cell_centers[owners]
    outward = jnp.sum((face_centers - owner_centers) * area_vectors, axis=-1)
    area_vectors = eqx.error_if(
        area_vectors,
        jnp.any(~jnp.isfinite(face_measures) | (face_measures <= 0.0) | (outward <= 0.0)),
        "Embedded surface conormals must be positive and owner-outward.",
    )
    valid = jnp.asarray(connectivity.cell_edge_valid)
    cell_edges = jnp.asarray(connectivity.cell_edges, dtype=jnp.int32)
    cell_signs = jnp.asarray(connectivity.cell_edge_signs, dtype=points.dtype)
    cell_ids = jnp.broadcast_to(jnp.arange(cells.shape[0])[:, None], cell_edges.shape)
    local_tangents = tangent[cell_edges]
    local_area = jnp.cross(local_tangents, normals[:, None, :])
    contributions = jnp.where(
        valid[..., None],
        cell_signs[..., None] * local_area,
        jnp.zeros_like(local_area),
    )
    closure = (
        jnp.zeros_like(cell_centers)
        .at[cell_ids.reshape((-1,))]
        .add(contributions.reshape((-1, points.shape[-1])))
    )
    gauss_offset = tangent / (2.0 * jnp.sqrt(jnp.asarray(3.0, dtype=points.dtype)))
    quadrature_points = jnp.stack(
        (face_centers - gauss_offset, face_centers + gauss_offset), axis=1
    )
    quadrature_weights = jnp.broadcast_to(
        0.5 * face_measures[:, None], (face_measures.size, 2)
    )
    return (
        cell_areas,
        cell_centers,
        face_centers,
        area_vectors,
        face_measures,
        closure,
        quadrature_points,
        quadrature_weights,
    )


def _polygon_geometry(
    vertices: ArrayLike,
    triangles: ArrayLike,
    quadrilaterals: ArrayLike,
    connectivity: PolygonalConnectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
) -> Any:
    points = jnp.asarray(vertices)
    triangle_cells = jnp.asarray(triangles, dtype=jnp.int32)
    quadrilateral_cells = jnp.asarray(quadrilaterals, dtype=jnp.int32)
    if points.shape[1] == 3:
        if quadrilateral_cells.shape[0]:
            raise ValueError("Embedded surface FV currently accepts triangles only.")
        return _embedded_triangle_geometry(
            points, triangle_cells, connectivity, owner, owner_sign
        )
    triangle_volume, triangle_center = _triangle_measures(points[triangle_cells])

    root = 1.0 / np.sqrt(3.0)
    reference = np.asarray(((-root, -root), (root, -root), (root, root), (-root, root)))
    shape_host, gradient_host = _quadrilateral_shape_data(reference)
    shape = jnp.asarray(shape_host, dtype=points.dtype)
    gradient = jnp.asarray(gradient_host, dtype=points.dtype)
    quadrilateral_points = points[quadrilateral_cells]
    mapped = ein.contract("qv,cvd->cqd", shape, quadrilateral_points)
    jacobian = ein.contract("qva,cvd->cqad", gradient, quadrilateral_points)
    determinant = (
        jacobian[..., 0, 0] * jacobian[..., 1, 1]
        - jacobian[..., 0, 1] * jacobian[..., 1, 0]
    )
    determinant = eqx.error_if(
        determinant,
        jnp.any(~jnp.isfinite(determinant) | (determinant <= 0.0)),
        "Quadrilateral bilinear maps require positive finite Jacobians.",
    )
    quadrilateral_volume = jnp.sum(determinant, axis=1)
    quadrilateral_center = (
        jnp.sum(mapped * determinant[..., None], axis=1) / quadrilateral_volume[:, None]
    )
    cell_volumes = jnp.concatenate((triangle_volume, quadrilateral_volume))
    cell_centers = jnp.concatenate((triangle_center, quadrilateral_center), axis=0)
    cell_volumes = eqx.error_if(
        cell_volumes,
        jnp.any(~jnp.isfinite(cell_volumes) | (cell_volumes <= 0.0)),
        "Polygonal FV geometry requires positive finite cell areas.",
    )

    edges = jnp.asarray(connectivity.edges, dtype=jnp.int32)
    edge_points = points[edges]
    face_centers = 0.5 * (edge_points[:, 0] + edge_points[:, 1])
    tangent = edge_points[:, 1] - edge_points[:, 0]
    canonical_area = _edge_area_vectors(tangent)
    area_vectors = jnp.asarray(owner_sign, dtype=points.dtype)[:, None] * canonical_area
    face_measures = jnp.linalg.norm(area_vectors, axis=-1)
    owner_centers = cell_centers[jnp.asarray(owner, dtype=jnp.int32)]
    outward = jnp.sum((face_centers - owner_centers) * area_vectors, axis=-1)
    area_vectors = eqx.error_if(
        area_vectors,
        jnp.any(~jnp.isfinite(face_measures) | (face_measures <= 0.0) | (outward <= 0.0)),
        "Polygonal face vectors must be positive and owner-outward.",
    )
    valid = jnp.asarray(connectivity.cell_edge_valid)
    cell_edges = jnp.asarray(connectivity.cell_edges, dtype=jnp.int32)
    cell_signs = jnp.asarray(connectivity.cell_edge_signs, dtype=points.dtype)
    cell_ids = jnp.broadcast_to(jnp.arange(cell_volumes.size)[:, None], cell_edges.shape)
    padded_area = canonical_area[cell_edges]
    contributions = jnp.where(
        valid[..., None],
        cell_signs[..., None] * padded_area,
        jnp.zeros_like(padded_area),
    )
    closure = (
        jnp.zeros_like(cell_centers)
        .at[cell_ids.reshape((-1,))]
        .add(contributions.reshape((-1, points.shape[1])))
    )
    gauss_offset = tangent / (2.0 * jnp.sqrt(jnp.asarray(3.0, dtype=points.dtype)))
    quadrature_points = jnp.stack(
        (face_centers - gauss_offset, face_centers + gauss_offset), axis=1
    )
    quadrature_weights = jnp.broadcast_to(
        0.5 * face_measures[:, None], (face_measures.size, 2)
    )
    return (
        cell_volumes,
        cell_centers,
        face_centers,
        area_vectors,
        face_measures,
        closure,
        quadrature_points,
        quadrature_weights,
    )


def _tetrahedral_geometry(
    vertices: ArrayLike,
    tetrahedra: ArrayLike,
    connectivity: TetrahedralConnectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
) -> Any:
    points = jnp.asarray(vertices)
    cells = jnp.asarray(tetrahedra, dtype=jnp.int32)
    cell_points = points[cells]
    cell_volumes, cell_centers = _tetrahedron_measures(cell_points)
    cell_volumes = eqx.error_if(
        cell_volumes,
        jnp.any(~jnp.isfinite(cell_volumes) | (cell_volumes <= 0.0)),
        "Tetrahedral FV geometry requires positive finite cell volumes.",
    )
    faces = jnp.asarray(connectivity.faces, dtype=jnp.int32)
    face_points = points[faces]
    face_centers = jnp.mean(face_points, axis=1)
    canonical_area = _triangle_area_vectors(face_points)
    area_vectors = jnp.asarray(owner_sign, dtype=points.dtype)[:, None] * canonical_area
    face_measures = jnp.linalg.norm(area_vectors, axis=-1)
    owner_centers = cell_centers[jnp.asarray(owner, dtype=jnp.int32)]
    outward = jnp.sum((face_centers - owner_centers) * area_vectors, axis=-1)
    area_vectors = eqx.error_if(
        area_vectors,
        jnp.any(~jnp.isfinite(face_measures) | (face_measures <= 0.0) | (outward <= 0.0)),
        "Tetrahedral face vectors must be positive and owner-outward.",
    )
    cell_faces = jnp.asarray(connectivity.cell_faces, dtype=jnp.int32)
    cell_signs = jnp.asarray(connectivity.cell_face_signs, dtype=points.dtype)
    cell_ids = jnp.broadcast_to(jnp.arange(cells.shape[0])[:, None], cell_faces.shape)
    closure = (
        jnp.zeros_like(cell_centers)
        .at[cell_ids.reshape((-1,))]
        .add((cell_signs[..., None] * canonical_area[cell_faces]).reshape((-1, 3)))
    )
    barycentric = jnp.asarray(
        _TETRAHEDRAL_FACE_QUADRATURE_BARYCENTRIC, dtype=points.dtype
    )
    normalized_weights = jnp.asarray(
        _TETRAHEDRAL_FACE_QUADRATURE_NORMALIZED_WEIGHTS, dtype=points.dtype
    )
    quadrature_points = ein.contract("qv,fvd->fqd", barycentric, face_points)
    quadrature_weights = face_measures[:, None] * normalized_weights[None, :]
    return (
        cell_volumes,
        cell_centers,
        face_centers,
        area_vectors,
        face_measures,
        closure,
        quadrature_points,
        quadrature_weights,
    )


def evaluate_unstructured_fv_geometry(
    vertices: ArrayLike,
    triangles: ArrayLike,
    quadrilaterals: ArrayLike,
    tetrahedra: ArrayLike,
    connectivity: Connectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
) -> Any:
    """Evaluate owner-oriented geometry for one prepared cell complex."""

    if isinstance(connectivity, PolygonalConnectivity):
        return _polygon_geometry(
            vertices,
            triangles,
            quadrilaterals,
            connectivity,
            owner,
            owner_sign,
        )
    # ty: ignore[invalid-argument-type]
    return _tetrahedral_geometry(vertices, tetrahedra, connectivity, owner, owner_sign)


# Facet opposite each local vertex, ordered so the edge normal ``(t_y, -t_x)`` of
# a counterclockwise triangle and the right-handed area vector of a positively
# oriented tetrahedron face both point out of the cell.
_SIMPLEX_OUTWARD_FACETS = {
    2: ((1, 2), (2, 0), (0, 1)),
    3: ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
}


def _masked_cell_geometry(
    mesh: MaskedSimplexMesh, /
) -> tuple[Array, Array, Array, Array]:
    """Cell volumes/centroids and half-facet area vectors/centroids.

    Inactive cells are evaluated on the positively oriented reference simplex
    before any geometric operation, so their lanes are finite in value and
    derivative regardless of the padding rows and coordinates.
    """

    dimension = mesh.dimension
    dtype = mesh.coordinates.dtype
    reference = jnp.asarray(
        np.vstack(
            (
                np.zeros((1, dimension), dtype=np.float64),
                np.eye(dimension, dtype=np.float64),
            )
        ),
        dtype=dtype,
    )
    cell_points = jnp.where(
        mesh.cell_active[:, None, None], mesh.coordinates[mesh.cells], reference
    )
    facets = np.asarray(_SIMPLEX_OUTWARD_FACETS[dimension], dtype=np.int32)
    facet_points = cell_points[:, facets]
    match dimension:
        case 2:
            volumes, centers = _triangle_measures(cell_points)
            area_vectors = _edge_area_vectors(
                facet_points[..., 1, :] - facet_points[..., 0, :]
            )
        case 3:
            volumes, centers = _tetrahedron_measures(cell_points)
            area_vectors = _triangle_area_vectors(facet_points)
        case _:
            raise ValueError("Masked FV geometry holds triangles or tetrahedra.")
    face_count = mesh.cell_capacity * (dimension + 1)
    return (
        volumes,
        centers,
        area_vectors.reshape((face_count, dimension)),
        jnp.mean(facet_points, axis=-2).reshape((face_count, dimension)),
    )


@final
class MaskedFiniteVolumeGeometry(StrictModule):
    """Owner-oriented finite-volume geometry of a capacity-bucketed simplex layout.

    Face routes are the ``face_capacity = cell_capacity * (d + 1)`` half-facets
    of the layout: route ``c * (d + 1) + i`` is the facet of cell slot ``c``
    opposite local vertex ``i`` and is owned by ``c``. ``face_active`` keeps
    exactly one route per physical face of the active mesh: a boundary
    half-facet of an active cell, or the interior half-facet whose owner slot is
    smaller than its neighbor slot. Area vectors point out of the owner, and
    ``content_rate_map`` is the fixed-capacity sparse face-to-cell map (owner
    ``-1``, neighbor ``+1``) taking owner-outward integrated face fluxes to net
    cell content rates.

    Every quantity is exactly zero on inactive lanes (cell volumes and centroids
    of inactive cells; area vectors, measures, and centroids of inactive routes,
    whose neighbor is ``-1``), and the geometry is computed from a reference
    simplex on those lanes, so padding never yields non-finite values or
    derivatives. Active lanes use the same simplex formulas as
    `UnstructuredFiniteVolumeDiscretization`. ``signature_id`` is the capacity
    signature of the source `MaskedSimplexMesh`; it, not the active topology,
    is the compile identity of every masked finite-volume route.
    """

    cell_kind: str = eqx.field(static=True)
    cell_capacity: int = eqx.field(static=True)
    face_capacity: int = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)
    cell_active: Array
    cell_volumes: Array
    cell_centers: Array
    owner_cells: Array
    neighbor_cells: Array
    face_active: Array
    area_vectors: Array
    face_measures: Array
    face_centers: Array
    content_rate_map: SparseLinearMap

    @checked
    def __init__(self, mesh: MaskedSimplexMesh, /) -> None:
        """Evaluate the geometry of ``mesh``; see `evaluate_masked_fv_geometry`."""
        if mesh.ambient_dimension != mesh.dimension:
            raise ValueError(
                "Masked FV geometry requires triangles in 2-D or tetrahedra in 3-D."
            )
        width = mesh.dimension + 1
        face_count = mesh.cell_capacity * width
        dtype = mesh.coordinates.dtype
        volumes, centers, area_vectors, face_centers = _masked_cell_geometry(mesh)
        cell_active = mesh.cell_active
        volumes = eqx.error_if(
            volumes,
            jnp.any(cell_active & (~jnp.isfinite(volumes) | (volumes <= 0.0))),
            "Masked FV geometry requires positively oriented finite active cells.",
        )

        half = jnp.arange(face_count, dtype=jnp.int32)
        owner = half // width
        packed = mesh.facet_neighbors.reshape((face_count,))
        interior = packed >= 0
        safe_packed = jnp.where(interior, packed, 0)
        neighbor = jnp.where(interior, safe_packed // width, -1)
        live = jnp.repeat(cell_active, width)
        face_active = live & (~interior | (owner < neighbor))
        measures = jnp.linalg.norm(area_vectors, axis=-1)
        outward = jnp.sum((face_centers - centers[owner]) * area_vectors, axis=-1)
        area_vectors = eqx.error_if(
            area_vectors,
            jnp.any(
                live & (~jnp.isfinite(measures) | (measures <= 0.0) | (outward <= 0.0))
            ),
            "Masked FV half-facet vectors must be positive and owner-outward.",
        )
        # Each interior face must be seen from both sides by active cells, so the
        # owner-slot tie-break selects exactly one route per physical face.
        unpaired = (
            live
            & interior
            & ((packed[safe_packed] != half) | ~cell_active[safe_packed // width])
        )
        area_vectors = eqx.error_if(
            area_vectors,
            jnp.any(unpaired),
            "Masked simplex facet neighbors must pair active cells reciprocally.",
        )
        neighbor = jnp.where(face_active, neighbor, -1)
        zero = jnp.zeros((), dtype=dtype)
        routed = face_active[:, None]
        interior_route = face_active & (neighbor >= 0)
        relation = EdgeRelation(
            jnp.concatenate((half, half)),
            jnp.concatenate((owner, jnp.maximum(neighbor, 0))),
            source_size=face_count,
            target_size=mesh.cell_capacity,
            valid=jnp.concatenate((face_active, interior_route)),
        )
        coefficients = jnp.concatenate(
            (-jnp.ones((face_count,), dtype=dtype), jnp.ones((face_count,), dtype=dtype))
        )
        self.cell_kind = mesh.cell_kind
        self.cell_capacity = mesh.cell_capacity
        self.face_capacity = face_count
        self.signature_id = mesh.signature_id
        self.cell_active = cell_active
        self.cell_volumes = jnp.where(cell_active, volumes, zero)
        self.cell_centers = jnp.where(cell_active[:, None], centers, zero)
        self.owner_cells = owner
        self.neighbor_cells = neighbor
        self.face_active = face_active
        self.area_vectors = jnp.where(routed, area_vectors, zero)
        self.face_measures = jnp.where(face_active, measures, zero)
        self.face_centers = jnp.where(routed, face_centers, zero)
        # Explicit operator identity: the capacity signature, never route values.
        self.content_rate_map = SparseLinearMap(
            relation,
            coefficients,
            operator_id=canonical_fingerprint(
                {
                    "kind": "masked-simplex-fv-content-rate-map",
                    "signature_id": mesh.signature_id,
                }
            ),
        )

    @property
    def dimension(self) -> int:
        return self.cell_centers.shape[1]

    @property
    def boundary_faces(self) -> Array:
        """Active routes on the boundary of the active mesh, shaped ``(faces,)``."""
        return self.face_active & (self.neighbor_cells < 0)


@final
class MaskedFiniteVolumeConservation(StrictModule):
    """Exact content-rate balance of one masked finite-volume evaluation.

    ``ledger`` is the repository `ConservationStageLedger` over the capacity
    half-facet routes (route identity: the capacity signature). The compensated
    sums obey ``net_cell_sum = source_sum - boundary_outward_sum`` because every
    interior route adds its owner-outward rate to one active cell and subtracts
    it from another; ``residual`` is the round-off defect of that identity.
    Inactive cells and routes contribute exactly zero to every sum.
    """

    ledger: ConservationStageLedger
    source_sum: Array
    boundary_outward_sum: Array
    net_cell_sum: Array
    residual: Array

    @checked
    def __init__(self, ledger: ConservationStageLedger, /) -> None:
        source_sum, boundary_sum, net_cell_sum = ledger.conservation_sums()
        self.ledger = ledger
        self.source_sum = source_sum
        self.boundary_outward_sum = boundary_sum
        self.net_cell_sum = net_cell_sum
        self.residual = net_cell_sum - (source_sum - boundary_sum)


def _routed_face_flux(
    geometry: MaskedFiniteVolumeGeometry, face_flux: ArrayLike, /
) -> Array:
    if not isinstance(geometry, MaskedFiniteVolumeGeometry):
        raise TypeError("geometry must be a MaskedFiniteVolumeGeometry.")
    flux = jnp.asarray(face_flux)
    if not jnp.issubdtype(flux.dtype, jnp.floating):
        raise TypeError("face_flux must have a floating dtype.")
    if flux.ndim == 0 or flux.shape[0] != geometry.face_capacity:
        raise ValueError("face_flux must begin with the half-facet route capacity.")
    return flux


@eqx.filter_jit
def evaluate_masked_fv_geometry(mesh: MaskedSimplexMesh, /) -> MaskedFiniteVolumeGeometry:
    """Compiled finite-volume geometry of a `MaskedSimplexMesh`.

    One compilation serves every layout with the same capacity signature: the
    active counts, topology, and coordinates are dynamic data.
    """
    return MaskedFiniteVolumeGeometry(mesh)


@eqx.filter_jit
def masked_fv_flux_divergence(
    geometry: MaskedFiniteVolumeGeometry, face_flux: ArrayLike, /
) -> Array:
    """Compiled cell residual ``-(1/V_c) * sum_f s_cf F_f`` of face fluxes.

    ``face_flux`` has shape ``(face_capacity, ...)`` and holds owner-outward
    area-integrated fluxes on the half-facet routes; the owner loses and the
    neighbor gains each routed flux. Values on inactive routes (padding and the
    twin half-facet of every interior face) are ignored, and inactive cells
    receive exactly zero.
    """
    flux = _routed_face_flux(geometry, face_flux)
    routed = geometry.face_active.reshape(
        geometry.face_active.shape + (1,) * (flux.ndim - 1)
    )
    rate = geometry.content_rate_map.mv(
        jnp.where(routed, flux, jnp.zeros((), dtype=flux.dtype))
    )
    active = geometry.cell_active.reshape(
        geometry.cell_active.shape + (1,) * (flux.ndim - 1)
    )
    volume = jnp.where(geometry.cell_active, geometry.cell_volumes, 1.0).reshape(
        active.shape
    )
    return jnp.where(active, rate / volume, jnp.zeros((), dtype=rate.dtype))


@eqx.filter_jit
def evaluate_masked_fv_conservation(
    geometry: MaskedFiniteVolumeGeometry,
    face_flux: ArrayLike,
    source_rate: ArrayLike,
    /,
) -> MaskedFiniteVolumeConservation:
    """Compiled exact conservation ledger of one masked finite-volume evaluation.

    ``face_flux`` follows `masked_fv_flux_divergence`; ``source_rate`` has shape
    ``(cell_capacity, ...)``, holds integrated cell source rates, and must be
    exactly zero on inactive cells. The ledger's layout, route, and topology
    epoch identities are the capacity signature because masked topology is
    dynamic data of the bucket; its geometry and evidence versions are zero for
    this single evaluation.
    """
    flux = _routed_face_flux(geometry, face_flux)
    block = ConservationStageFluxRateBlock(
        flux,
        geometry.owner_cells,
        geometry.neighbor_cells,
        geometry.face_active,
        "masked-simplex-faces",
        "masked-simplex-half-facet",
        route_signature_id=geometry.signature_id,
    )
    version = jnp.zeros((), dtype=jnp.int32)
    ledger = ConservationStageLedger(
        (block,),
        source_rate,
        geometry.cell_active,
        geometry_family_id="masked-simplex-finite-volume",
        geometry_layout_id=geometry.signature_id,
        geometry_version=version,
        evidence_policy_id="masked-simplex-exact-balance",
        evidence_version=version,
        topology_epoch_id=geometry.signature_id,
    )
    return MaskedFiniteVolumeConservation(ledger)


class UnstructuredFiniteVolumeQualityReport(StrictModule):
    minimum_cell_measure: Array
    maximum_cell_measure: Array
    minimum_face_measure: Array
    maximum_aspect_ratio: Array
    maximum_nonorthogonality_degrees: Array
    maximum_closure_residual: Array
    worst_cell: Array


@dataclass(frozen=True, slots=True)
class _PreparedUnstructuredFiniteVolumeData:
    mesh: CellMesh | None
    vertices: Array
    triangles: Array
    quadrilaterals: Array
    tetrahedra: Array
    vertex_global_ids: Array
    cell_global_ids: Array
    cell_dimension: int
    patch_names: tuple[str, ...]
    patch_faces: tuple[Array, ...]
    field_name: str
    component_names: tuple[str, ...]
    topology_id: str
    geometry_id: str
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...]
    plan_id: str
    owner_local_view: Any | None = None
    periodic_face_joins: Array | None = None
    periodic_face_translations: Array | None = None


class UnstructuredFiniteVolumePlan(AbstractDiscretizationPlan):
    """Fixed-topology planar polygons, embedded surface triangles, or tetrahedra."""

    mesh: CellMesh | None

    vertices: Array
    triangles: Array
    quadrilaterals: Array
    tetrahedra: Array
    vertex_global_ids: Array
    cell_global_ids: Array
    cell_dimension: int = eqx.field(static=True)
    patch_names: tuple[str, ...] = eqx.field(static=True)
    patch_faces: tuple[Array, ...]
    field_name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    owner_local_view: Any | None = None
    # Owner-local periodic seam faces joined across lattice images: topology
    # faces ``(kept, merged)`` and the ambient translation from the kept
    # owner's frame to the neighbor's frame; ``None`` when nothing is joined.
    periodic_face_joins: Array | None = None
    periodic_face_translations: Array | None = None

    def __init__(
        self,
        vertices: ArrayLike | None,
        /,
        *,
        triangles: ArrayLike | None = None,
        quadrilaterals: ArrayLike | None = None,
        tetrahedra: ArrayLike | None = None,
        vertex_global_ids: ArrayLike | None = None,
        cell_global_ids: ArrayLike | None = None,
        boundary_patches: Mapping[str, ArrayLike] | None = None,
        field_name: str = "state",
        component_names: Sequence[str] = ("value",),
        _prepared: _PreparedUnstructuredFiniteVolumeData | None = None,
    ) -> None:
        if _prepared is not None:
            self.mesh = _prepared.mesh
            self.vertices = _prepared.vertices
            self.triangles = _prepared.triangles
            self.quadrilaterals = _prepared.quadrilaterals
            self.tetrahedra = _prepared.tetrahedra
            self.vertex_global_ids = _prepared.vertex_global_ids
            self.cell_global_ids = _prepared.cell_global_ids
            self.cell_dimension = _prepared.cell_dimension
            self.patch_names = _prepared.patch_names
            self.patch_faces = _prepared.patch_faces
            self.field_name = _prepared.field_name
            self.component_names = _prepared.component_names
            self.topology_id = _prepared.topology_id
            self.geometry_id = _prepared.geometry_id
            self.key = _prepared.key
            self.capabilities = _prepared.capabilities
            self.plan_id = _prepared.plan_id
            self.owner_local_view = _prepared.owner_local_view
            self.periodic_face_joins = _prepared.periodic_face_joins
            self.periodic_face_translations = _prepared.periodic_face_translations
            return
        if vertices is None:
            raise TypeError("vertices must be supplied for direct construction.")
        points = np.asarray(vertices, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Unstructured FV vertices must have shape (n, 2) or (n, 3).")
        triangle_cells = (
            np.empty((0, 3), dtype=np.int32)
            if triangles is None
            else np.asarray(triangles, dtype=np.int32)
        )
        quadrilateral_cells = (
            np.empty((0, 4), dtype=np.int32)
            if quadrilaterals is None
            else np.asarray(quadrilaterals, dtype=np.int32)
        )
        tetrahedral_cells = (
            np.empty((0, 4), dtype=np.int32)
            if tetrahedra is None
            else np.asarray(tetrahedra, dtype=np.int32)
        )
        minimum_vertices = (
            3
            if points.shape[1] == 3
            and triangle_cells.shape[0]
            and not tetrahedral_cells.shape[0]
            else points.shape[1] + 1
        )
        if points.shape[0] < minimum_vertices or np.any(~np.isfinite(points)):
            raise ValueError("Unstructured FV vertices must be finite and nonempty.")
        if points.shape[1] == 2:
            if tetrahedral_cells.shape[0]:
                raise ValueError("Tetrahedra require three-dimensional vertices.")
            if triangle_cells.ndim != 2 or triangle_cells.shape[1] != 3:
                raise ValueError("triangles must have shape (n, 3).")
            if quadrilateral_cells.ndim != 2 or quadrilateral_cells.shape[1] != 4:
                raise ValueError("quadrilaterals must have shape (n, 4).")
            triangle_cells = _normalized_triangles(points, triangle_cells)
            quadrilateral_cells = _normalized_quadrilaterals(points, quadrilateral_cells)
            connectivity: Connectivity = polygonal_connectivity(
                triangle_cells,
                quadrilateral_cells,
                points.shape[0],
            )
            dimension = 2
            face_vertices = np.asarray(connectivity.edges, dtype=np.int32)
            boundary_mask = np.asarray(connectivity.boundary_edges, dtype=np.bool_)
        elif points.shape[1] == 3 and triangle_cells.shape[0]:
            if quadrilateral_cells.shape[0] or tetrahedral_cells.shape[0]:
                raise ValueError("Embedded surface FV currently accepts triangles only.")
            triangle_cells = _validated_embedded_triangles(points, triangle_cells)
            connectivity = polygonal_connectivity(
                triangle_cells,
                np.empty((0, 4), dtype=np.int32),
                points.shape[0],
            )
            dimension = 2
            face_vertices = np.asarray(connectivity.edges, dtype=np.int32)
            boundary_mask = np.asarray(connectivity.boundary_edges, dtype=np.bool_)
        else:
            if triangle_cells.shape[0] or quadrilateral_cells.shape[0]:
                raise ValueError(
                    "Three-dimensional FV currently accepts tetrahedra only."
                )
            if tetrahedral_cells.ndim != 2 or tetrahedral_cells.shape[1] != 4:
                raise ValueError("tetrahedra must have shape (n, 4).")
            tetrahedral_cells = _normalized_tetrahedra(points, tetrahedral_cells)
            connectivity = tetrahedral_connectivity(tetrahedral_cells, points.shape[0])
            dimension = 3
            face_vertices = np.asarray(connectivity.faces, dtype=np.int32)
            boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
        cell_count = connectivity.cell_count

        patches = {} if boundary_patches is None else dict(boundary_patches)
        if not patches:
            patches = {"boundary": face_vertices[boundary_mask]}
        names = tuple(sorted(str(name) for name in patches))
        lookup = {
            tuple(sorted(int(value) for value in face)): index
            for index, face in enumerate(face_vertices.tolist())
        }
        assigned = np.zeros((face_vertices.shape[0],), dtype=np.int32)
        normalized_patch_faces = []
        face_arity = dimension
        for name in names:
            values = np.asarray(patches[name], dtype=np.int32)
            if not name or values.ndim != 2 or values.shape[1] != face_arity:
                raise ValueError(
                    f"Boundary patch faces must have shape (n, {face_arity})."
                )
            indices = []
            for face in values:
                key = tuple(sorted(int(vertex) for vertex in face))
                if key not in lookup:
                    raise ValueError(f"Boundary patch face {key!r} is not in the mesh.")
                face_index = lookup[key]
                if not boundary_mask[face_index]:
                    raise ValueError("Physical patches cannot contain interior faces.")
                assigned[face_index] += 1
                indices.append(face_index)
            normalized_patch_faces.append(np.asarray(indices, dtype=np.int32))
        if np.any(assigned[boundary_mask] != 1):
            raise ValueError(
                "Every unstructured boundary face requires exactly one patch."
            )
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        components = _component_names(component_names)
        vertex_ids = _stable_global_ids(
            "vertex_global_ids", vertex_global_ids, points.shape[0]
        )
        cell_ids = _stable_global_ids("cell_global_ids", cell_global_ids, cell_count)
        blocks = []
        offset = 0
        if triangle_cells.shape[0]:
            count = triangle_cells.shape[0]
            blocks.append(
                CellBlock(
                    "triangles",
                    "triangle",
                    triangle_cells,
                    global_ids=cell_ids[offset : offset + count],
                )
            )
            offset += count
        if quadrilateral_cells.shape[0]:
            count = quadrilateral_cells.shape[0]
            blocks.append(
                CellBlock(
                    "quadrilaterals",
                    "quadrilateral",
                    quadrilateral_cells,
                    global_ids=cell_ids[offset : offset + count],
                )
            )
            offset += count
        if tetrahedral_cells.shape[0]:
            blocks.append(
                CellBlock(
                    "tetrahedra",
                    "tetrahedron",
                    tetrahedral_cells,
                    global_ids=cell_ids,
                )
            )
        mesh = CellMesh(
            points,
            blocks,
            vertex_global_ids=vertex_ids,
        )
        topology_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-topology",
                "dimension": dimension,
                "triangles": array_tree_fingerprint(triangle_cells),
                "quadrilaterals": array_tree_fingerprint(quadrilateral_cells),
                "tetrahedra": array_tree_fingerprint(tetrahedral_cells),
                "vertex_global_ids": array_tree_fingerprint(vertex_ids),
                "cell_global_ids": array_tree_fingerprint(cell_ids),
                "patches": {
                    name: array_tree_fingerprint(value)
                    for name, value in zip(names, normalized_patch_faces, strict=True)
                },
                "field": field,
                "components": list(components),
            }
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-geometry",
                "topology": topology_id,
                "vertices": array_tree_fingerprint(points),
            }
        )
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.CONSERVATIVE_FLUX,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        global_id_dtype = jnp.int64 if x64_enabled() else jnp.int32
        self.mesh = mesh
        self.vertices = jnp.asarray(points)
        self.triangles = jnp.asarray(triangle_cells)
        self.quadrilaterals = jnp.asarray(quadrilateral_cells)
        self.tetrahedra = jnp.asarray(tetrahedral_cells)
        self.vertex_global_ids = jnp.asarray(vertex_ids, dtype=global_id_dtype)
        self.cell_global_ids = jnp.asarray(cell_ids, dtype=global_id_dtype)
        self.cell_dimension = dimension
        self.patch_names = names
        self.patch_faces = tuple(jnp.asarray(value) for value in normalized_patch_faces)
        self.field_name = field
        self.component_names = components
        self.topology_id = topology_id
        self.geometry_id = geometry_id
        self.key = DiscretizationKey(
            "unstructured_finite_volume", DiscretizationRole.PHYSICAL
        )
        self.capabilities = capabilities
        self.plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-plan",
                "topology": topology_id,
                "geometry": geometry_id,
            }
        )
        self.owner_local_view = None
        self.periodic_face_joins = None
        self.periodic_face_translations = None

    @classmethod
    @checked
    def from_cell_mesh(
        cls,
        mesh: CellMesh,
        /,
        *,
        field_name: str = "state",
        component_names: Sequence[str] = ("value",),
        boundary_face_groups: Mapping[str, ArrayLike] | None = None,
    ) -> "UnstructuredFiniteVolumePlan":
        """Construct directly from canonical polyhedral CellMesh."""
        if not isinstance(mesh.connectivity, PolyhedralConnectivity):
            raise ValueError(
                "CellMesh FV cutover currently targets polyhedral connectivity; "
                "legacy tetrahedral callers use the direct constructor."
            )
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        components = _component_names(component_names)
        connectivity = mesh.connectivity
        cell_ids = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        face_owner = np.asarray(connectivity.face_owner, dtype=np.int32)
        face_neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32)
        boundary_faces = np.flatnonzero(face_neighbor < 0).astype(np.int32)
        if boundary_face_groups is None:
            patch_names = ("boundary",)
            patch_faces = (boundary_faces,)
        else:
            groups = {
                str(name): np.asarray(indices, dtype=np.int32)
                for name, indices in boundary_face_groups.items()
            }
            patch_names = tuple(sorted(groups))
            if not patch_names or any(not name for name in patch_names):
                raise ValueError(
                    "Polyhedral boundary face-group names must be non-empty."
                )
            patch_faces = tuple(groups[name] for name in patch_names)
            assigned = np.zeros((face_owner.size,), dtype=np.int32)
            for indices in patch_faces:
                if (
                    indices.ndim != 1
                    or np.any(indices < 0)
                    or np.any(indices >= face_owner.size)
                    or np.unique(indices).size != indices.size
                    or np.any(face_neighbor[indices] >= 0)
                ):
                    raise ValueError(
                        "Polyhedral boundary groups require unique in-range boundary faces."
                    )
                assigned[indices] += 1
            if np.any(assigned[boundary_faces] != 1) or np.any(
                assigned[face_neighbor >= 0] != 0
            ):
                raise ValueError(
                    "Polyhedral boundary groups must partition every boundary face exactly."
                )
        topology_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-topology",
                "mesh": mesh.topology_id,
                "field": field,
                "components": list(components),
                "boundary_patches": {
                    name: array_tree_fingerprint(indices)
                    for name, indices in zip(patch_names, patch_faces, strict=True)
                },
            }
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-geometry",
                "topology": topology_id,
                "mesh_geometry": mesh.geometry_id,
            }
        )
        capabilities = (
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.TRACE,
            DiscretizationCapability.CONSERVATIVE_FLUX,
            DiscretizationCapability.BOUNDARY_INTEGRAL,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        tetrahedra = jnp.empty((0, 4), dtype=jnp.int32)
        key = DiscretizationKey(
            "unstructured_finite_volume",
            DiscretizationRole.PHYSICAL,
        )
        plan_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-plan",
                "topology": topology_id,
                "geometry": geometry_id,
            }
        )
        prepared = _PreparedUnstructuredFiniteVolumeData(
            mesh=mesh,
            vertices=mesh.coordinates,
            triangles=jnp.empty((0, 3), dtype=jnp.int32),
            quadrilaterals=jnp.empty((0, 4), dtype=jnp.int32),
            tetrahedra=tetrahedra,
            vertex_global_ids=jnp.asarray(vertex_ids),
            cell_global_ids=jnp.asarray(cell_ids),
            cell_dimension=3,
            patch_names=patch_names,
            patch_faces=tuple(jnp.asarray(indices) for indices in patch_faces),
            field_name=field,
            component_names=components,
            topology_id=topology_id,
            geometry_id=geometry_id,
            key=key,
            capabilities=capabilities,
            plan_id=plan_id,
        )
        return cls(None, _prepared=prepared)

    @classmethod
    def from_owner_local_geometry(
        cls,
        view: OwnerLocalCellGeometry | OwnerLocalCompositeCellGeometry,
        /,
        *,
        field_name: str = "state",
        component_names: Sequence[str] = ("value",),
    ) -> "UnstructuredFiniteVolumePlan":
        """Lower one compact owner-local cell view without assembling a root mesh.

        A composite multi-part view lowers to one disjoint local layout whose
        part blocks keep their own stable cell IDs and owner/halo classes.
        """
        from .._view_support import (
            OwnerLocalCellGeometry,
            OwnerLocalCompositeCellGeometry,
        )

        if not isinstance(
            view, (OwnerLocalCellGeometry, OwnerLocalCompositeCellGeometry)
        ):
            raise TypeError(
                "view must be OwnerLocalCellGeometry or OwnerLocalCompositeCellGeometry."
            )
        field = str(field_name)
        if not field or field.strip() != field:
            raise ValueError("field_name must be a non-empty exact identifier.")
        components = _component_names(component_names)
        dimension = view.topological_dimension
        if isinstance(view, OwnerLocalCompositeCellGeometry):
            mesh = view.union_local_mesh()
            geometry = None
        else:
            mesh, geometry = view.reconstruct_local_geometry()
        if mesh is None:
            if (
                geometry is not None
                or np.asarray(view.cell_ids).size
                or np.asarray(view.vertex_ids).size
            ):
                raise ValueError(
                    "Only a genuinely empty owner-local view has no carrier."
                )
            key = DiscretizationKey(
                "unstructured_finite_volume", DiscretizationRole.PHYSICAL
            )
            capabilities = (
                DiscretizationCapability.RECONSTRUCTION,
                DiscretizationCapability.TRACE,
                DiscretizationCapability.CONSERVATIVE_FLUX,
                DiscretizationCapability.BOUNDARY_INTEGRAL,
                DiscretizationCapability.MATRIX_FREE,
                DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
            )
            patch_names: tuple[str, ...] = ()
            patch_faces: tuple[Array, ...] = ()
            vertices = jnp.empty((0, dimension), dtype=jnp.float64)
            triangles = jnp.empty((0, 3), dtype=jnp.int32)
            quadrilaterals = jnp.empty((0, 4), dtype=jnp.int32)
            tetrahedra = jnp.empty((0, 4), dtype=jnp.int32)
            global_id_dtype = jnp.int64 if x64_enabled() else jnp.int32
            vertex_ids = jnp.empty((0,), dtype=global_id_dtype)
            cell_ids = jnp.empty((0,), dtype=global_id_dtype)
            topology_id = canonical_fingerprint(
                {
                    "kind": "owner-local-empty-fv-topology",
                    "source": view.source_topology_id,
                    "view": view.local_view_id,
                    "dimension": dimension,
                }
            )
            geometry_id = canonical_fingerprint(
                {
                    "kind": "owner-local-empty-fv-geometry",
                    "source": view.source_geometry_id,
                    "layout": view.geometry_layout_id,
                    "view": view.local_view_id,
                }
            )
            prepared = _PreparedUnstructuredFiniteVolumeData(
                mesh=None,
                vertices=vertices,
                triangles=triangles,
                quadrilaterals=quadrilaterals,
                tetrahedra=tetrahedra,
                vertex_global_ids=vertex_ids,
                cell_global_ids=cell_ids,
                cell_dimension=dimension,
                patch_names=patch_names,
                patch_faces=patch_faces,
                field_name=field,
                component_names=components,
                topology_id=topology_id,
                geometry_id=geometry_id,
                key=key,
                capabilities=capabilities,
                plan_id=canonical_fingerprint(
                    {
                        "kind": "owner-local-empty-fv-plan",
                        "topology": topology_id,
                        "geometry": geometry_id,
                        "field": field,
                        "components": components,
                    }
                ),
                owner_local_view=view,
            )
            return cls(None, _prepared=prepared)
        if isinstance(view, OwnerLocalCellGeometry) and (
            geometry is None or view.local_mesh_id != mesh.mesh_id
        ):
            raise ValueError("Owner-local FV carrier lacks its exact local geometry map.")
        cell_ids = np.asarray(
            mesh.topology.entity_sets[dimension].entity_ids, dtype=np.int64
        )
        local_cell_ids = np.asarray(view.cell_ids, dtype=np.int64)
        local_owned = np.asarray(view.owned, dtype=np.bool_)
        if cell_ids.shape != local_owned.shape or set(
            int(value) for value in cell_ids
        ) != set(int(value) for value in local_cell_ids):
            raise ValueError("Owner-local FV cells differ from the accepted source view.")
        # The view's face-neighbor CSR is keyed by its sorted owned cells.
        graph_ids = np.asarray(view.owned_cell_ids, dtype=np.int64)
        offsets = np.asarray(view.cell_neighbor_offsets, dtype=np.int64)
        neighbor_ids = np.asarray(view.cell_neighbor_ids, dtype=np.int64)
        if (
            offsets.shape != (graph_ids.size + 1,)
            or offsets[-1] != neighbor_ids.size
            or set(int(value) for value in graph_ids)
            != set(int(value) for value in local_cell_ids[local_owned])
        ):
            raise ValueError("Owner-local FV neighbor metadata is malformed.")
        available_cells = set(int(value) for value in local_cell_ids)
        for row in range(graph_ids.size):
            required = neighbor_ids[offsets[row] : offsets[row + 1]]
            if any(int(identifier) not in available_cells for identifier in required):
                raise ValueError(
                    "Owner-local FV view omits a required owned-cell neighbor halo."
                )
        owned_by_id = {
            int(identifier): bool(local_owned[row])
            for row, identifier in enumerate(local_cell_ids)
        }
        coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
        frames = _owner_local_periodic_vertex_frames(
            view, np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        )
        if isinstance(mesh.connectivity, PolyhedralConnectivity):
            connectivity = mesh.connectivity
            owner = np.asarray(connectivity.face_owner, dtype=np.int32)
            neighbor = np.asarray(connectivity.face_neighbor, dtype=np.int32)
            cell_order = np.asarray(connectivity.cell_global_ids, dtype=np.int64)
            face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
            groups, joins, translations = _classify_owner_local_faces(
                tuple(
                    face_values[face_offsets[face] : face_offsets[face + 1]]
                    for face in range(owner.size)
                ),
                np.flatnonzero(neighbor < 0),
                np.asarray(
                    [owned_by_id[int(cell_order[cell])] for cell in owner],
                    dtype=np.bool_,
                ),
                coordinates,
                frames,
            )
            plan = cls.from_cell_mesh(
                mesh,
                field_name=field,
                component_names=components,
                boundary_face_groups=groups,
            )
        elif dimension == 2:
            triangle_blocks = tuple(
                block for block in mesh.blocks if block.cell_kind == "triangle"
            )
            quadrilateral_blocks = tuple(
                block for block in mesh.blocks if block.cell_kind == "quadrilateral"
            )
            if len(triangle_blocks) + len(quadrilateral_blocks) != len(mesh.blocks):
                raise ValueError(
                    "Owner-local 2D FV currently requires canonical triangle/quadrilateral cells."
                )
            triangles = (
                np.concatenate(
                    [
                        np.asarray(block.vertices, dtype=np.int32)
                        for block in triangle_blocks
                    ],
                    axis=0,
                )
                if triangle_blocks
                else np.empty((0, 3), dtype=np.int32)
            )
            quadrilaterals = (
                np.concatenate(
                    [
                        np.asarray(block.vertices, dtype=np.int32)
                        for block in quadrilateral_blocks
                    ],
                    axis=0,
                )
                if quadrilateral_blocks
                else np.empty((0, 4), dtype=np.int32)
            )
            ordered_ids = np.concatenate(
                [
                    *(
                        [
                            np.asarray(block.global_ids, dtype=np.int64)
                            for block in triangle_blocks
                        ]
                    ),
                    *(
                        [
                            np.asarray(block.global_ids, dtype=np.int64)
                            for block in quadrilateral_blocks
                        ]
                    ),
                ]
            )
            base = cls(
                np.asarray(mesh.coordinates),
                triangles=triangles,
                quadrilaterals=quadrilaterals,
                vertex_global_ids=np.asarray(mesh.vertex_global_ids),
                cell_global_ids=ordered_ids,
                field_name=field,
                component_names=components,
            )
            base_mesh = base.mesh
            if base_mesh is None:
                raise ValueError(
                    "Owner-local polygonal FV plans require a resident mesh."
                )
            connectivity = base_mesh.connectivity
            if not isinstance(connectivity, PolygonalConnectivity):
                raise TypeError(
                    "Owner-local polygonal FV plans require polygonal topology."
                )
            owner, neighbor, _ = _owner_neighbor(connectivity, base.cell_global_ids.size)
            face_vertices = np.asarray(connectivity.edges, dtype=np.int32)
            groups, joins, translations = _classify_owner_local_faces(
                face_vertices,
                np.flatnonzero(neighbor < 0),
                np.asarray(
                    [owned_by_id[int(ordered_ids[cell])] for cell in owner],
                    dtype=np.bool_,
                ),
                coordinates,
                frames,
            )
            plan = cls(
                np.asarray(mesh.coordinates),
                triangles=triangles,
                quadrilaterals=quadrilaterals,
                vertex_global_ids=np.asarray(mesh.vertex_global_ids),
                cell_global_ids=ordered_ids,
                boundary_patches={
                    name: face_vertices[faces] for name, faces in groups.items()
                },
                field_name=field,
                component_names=components,
            )
        elif dimension == 3 and all(
            block.cell_kind == "tetrahedron" for block in mesh.blocks
        ):
            tetrahedra = np.concatenate(
                [np.asarray(block.vertices, dtype=np.int32) for block in mesh.blocks],
                axis=0,
            )
            ordered_ids = np.concatenate(
                [np.asarray(block.global_ids, dtype=np.int64) for block in mesh.blocks]
            )
            base = cls(
                np.asarray(mesh.coordinates),
                tetrahedra=tetrahedra,
                vertex_global_ids=np.asarray(mesh.vertex_global_ids),
                cell_global_ids=ordered_ids,
                field_name=field,
                component_names=components,
            )
            base_mesh = base.mesh
            if base_mesh is None:
                raise ValueError(
                    "Owner-local tetrahedral FV plans require a resident mesh."
                )
            connectivity = base_mesh.connectivity
            if not isinstance(connectivity, TetrahedralConnectivity):
                raise TypeError(
                    "Owner-local tetrahedral FV plans require tetrahedral topology."
                )
            owner, neighbor, _ = _owner_neighbor(connectivity, base.cell_global_ids.size)
            face_vertices = np.asarray(connectivity.faces, dtype=np.int32)
            groups, joins, translations = _classify_owner_local_faces(
                face_vertices,
                np.flatnonzero(neighbor < 0),
                np.asarray(
                    [owned_by_id[int(ordered_ids[cell])] for cell in owner],
                    dtype=np.bool_,
                ),
                coordinates,
                frames,
            )
            plan = cls(
                np.asarray(mesh.coordinates),
                tetrahedra=tetrahedra,
                vertex_global_ids=np.asarray(mesh.vertex_global_ids),
                cell_global_ids=ordered_ids,
                boundary_patches={
                    name: face_vertices[faces] for name, faces in groups.items()
                },
                field_name=field,
                component_names=components,
            )
        else:
            raise ValueError(
                "Owner-local unstructured FV requires triangles/quadrilaterals, tetrahedra, or polyhedral connectivity."
            )
        if plan.cell_global_ids.shape != (local_cell_ids.size,):
            raise ValueError("Owner-local FV lowering changed the local cell count.")
        return cls._with_owner_local_joins(plan, view, joins, translations)

    @classmethod
    def _with_owner_local_joins(
        cls,
        plan: UnstructuredFiniteVolumePlan,
        view: Any | None,
        joins: np.ndarray,
        translations: np.ndarray,
        /,
    ) -> UnstructuredFiniteVolumePlan:
        """Bind a lifted owner-local plan to `view` with its periodic seam joins.

        `plan` labels every unwelded seam face with the seam patch only to
        satisfy the boundary partition of the lifted patch; joined, those faces
        are interior faces of the quotient, so the patch is dropped. Without a
        view the plan is one serial owner holding the whole lifted quotient.
        """
        patch_names = plan.patch_names
        patch_faces = plan.patch_faces
        if joins.size:
            seam = patch_names.index(_PERIODIC_SEAM_PATCH)
            if not np.array_equal(
                np.sort(np.asarray(patch_faces[seam], dtype=np.int32)),
                np.sort(joins.reshape(-1)),
            ):
                raise ValueError("Owner-local FV lowering renumbered periodic seams.")
            patch_names = patch_names[:seam] + patch_names[seam + 1 :]
            patch_faces = patch_faces[:seam] + patch_faces[seam + 1 :]
        plan_identity: dict[str, Any] = {
            "kind": "owner-local-unstructured-fv-plan",
            "plan": plan.plan_id,
            "view": None if view is None else view.local_view_id,
        }
        if joins.size:
            plan_identity["periodic_face_joins"] = array_tree_fingerprint(joins)
            plan_identity["periodic_face_translations"] = array_tree_fingerprint(
                translations
            )
        return cls(
            None,
            _prepared=_PreparedUnstructuredFiniteVolumeData(
                mesh=plan.mesh,
                vertices=plan.vertices,
                triangles=plan.triangles,
                quadrilaterals=plan.quadrilaterals,
                tetrahedra=plan.tetrahedra,
                vertex_global_ids=plan.vertex_global_ids,
                cell_global_ids=plan.cell_global_ids,
                cell_dimension=plan.cell_dimension,
                patch_names=patch_names,
                patch_faces=patch_faces,
                field_name=plan.field_name,
                component_names=plan.component_names,
                topology_id=plan.topology_id,
                geometry_id=plan.geometry_id,
                key=plan.key,
                capabilities=plan.capabilities,
                plan_id=canonical_fingerprint(plan_identity),
                owner_local_view=view,
                periodic_face_joins=jnp.asarray(joins) if joins.size else None,
                periodic_face_translations=(
                    jnp.asarray(translations) if joins.size else None
                ),
            ),
        )

    def with_periodic_seam_vertices(self, vertices: ArrayLike, /) -> Any:
        """This owner-local periodic plan on moved, translation-equivariant vertices.

        Seam joins and their lattice translations are topology: every joined
        pair must stay an exact translate (``1e-9`` of the coordinate scale) of
        the other, otherwise the moved patch is not a lifted periodic patch.
        """
        joins = self.periodic_face_joins
        translations = self.periodic_face_translations
        if joins is None or translations is None:
            raise ValueError("Only periodic seam plans rejoin moved seams.")
        points = np.asarray(vertices, dtype=np.float64)
        if points.shape != np.asarray(self.vertices).shape:
            raise ValueError("Moved seam vertices must keep the plan's vertex shape.")
        joins_host = np.asarray(joins, dtype=np.int32)
        translations_host = np.asarray(translations, dtype=np.float64)
        kept_rows, twin_rows, pair_translations = periodic_seam_vertex_pairs(
            self, joins_host, translations_host
        )
        scale = max(1.0, float(np.max(np.abs(points)))) if points.size else 1.0
        if np.any(
            np.abs(points[twin_rows] - points[kept_rows] - pair_translations)
            > 1.0e-9 * scale
        ):
            raise ValueError(
                "Moved periodic seam vertices are not lattice translates of their twins."
            )
        connectivity = self.mesh.connectivity if self.mesh is not None else None
        if isinstance(connectivity, PolygonalConnectivity):
            face_vertices = np.asarray(connectivity.edges, dtype=np.int32)
        elif isinstance(connectivity, TetrahedralConnectivity):
            face_vertices = np.asarray(connectivity.faces, dtype=np.int32)
        else:
            raise TypeError("Moved periodic seams need edge or triangle faces.")
        patches = {
            name: face_vertices[np.asarray(faces, dtype=np.int32)]
            for name, faces in zip(self.patch_names, self.patch_faces, strict=True)
        }
        patches[_PERIODIC_SEAM_PATCH] = face_vertices[np.sort(joins_host.reshape(-1))]
        moved = UnstructuredFiniteVolumePlan(
            points,
            triangles=self.triangles,
            quadrilaterals=self.quadrilaterals,
            tetrahedra=self.tetrahedra,
            vertex_global_ids=self.vertex_global_ids,
            cell_global_ids=self.cell_global_ids,
            boundary_patches=patches,
            field_name=self.field_name,
            component_names=self.component_names,
        )
        if moved.topology_id != self.topology_id:
            raise ValueError("Moving periodic seam vertices changed topology identity.")
        return self._with_owner_local_joins(
            moved, self.owner_local_view, joins_host, translations_host
        )

    def prepare(self, /, *, numeric_version: str = "0") -> Any:
        return UnstructuredFiniteVolumeDiscretization(
            self, numeric_version=numeric_version
        )


class UnstructuredFiniteVolumeDiscretization(AbstractPreparedDiscretization):
    mesh: CellMesh | None
    vertices: Array
    triangles: Array
    quadrilaterals: Array
    tetrahedra: Array
    vertex_global_ids: Array
    cell_global_ids: Array
    cell_dimension: int = eqx.field(static=True)
    topology: CellComplexTopology | None
    connectivity: Connectivity | None
    face_block: FiniteVolumeFaceBlock | None
    face_blocks: tuple[FiniteVolumeFaceBlock, ...]
    cell_volumes: Array
    cell_centers: Array
    cell_quadrature_points: Array
    cell_quadrature_weights: Array
    cell_quadrature_valid: Array
    cell_quadrature_degree: int = eqx.field(static=True)
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    face_quadrature_points: Array
    face_quadrature_weights: Array
    owner_cells: Array
    owner_signs: Array
    neighbor_cells: Array
    boundary_patch_ids: Array
    boundary_patch_names: tuple[str, ...] = eqx.field(static=True)
    # Owner-local periodic seams joined across lattice images: the lifted
    # topology face of every FV face (the owner's side), the topology face its
    # neighbor holds (the unwelded twin on seams, the same face elsewhere) and,
    # per FV face, the translation from the owner's frame to the neighbor's
    # frame (zero off seams). ``None`` when no face is joined, so every FV face
    # is its topology face in one embedding.
    face_topology_ids: Array | None
    neighbor_face_topology_ids: Array | None
    periodic_face_translations: Array | None
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    cell_space: DiscreteFieldSpace | None
    face_space: DiscreteFieldSpace | None
    component_names: tuple[str, ...] = eqx.field(static=True)
    key: DiscretizationKey
    support: DiscreteSupport | None
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    owner_local_view: Any | None
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    quality: UnstructuredFiniteVolumeQualityReport

    @checked
    def __init__(
        self, plan: UnstructuredFiniteVolumePlan, /, *, numeric_version: str = "0"
    ) -> None:
        mesh = plan.mesh
        if mesh is None:
            from .._view_support import (
                OwnerLocalCellGeometry,
                OwnerLocalCompositeCellGeometry,
            )

            def genuinely_empty(view: Any, /) -> bool:
                if isinstance(view, OwnerLocalCompositeCellGeometry):
                    return view.remote_ghosts is None and all(
                        genuinely_empty(part) for part in view.parts
                    )
                return isinstance(view, OwnerLocalCellGeometry) and not (
                    np.asarray(view.cell_ids).size
                    or np.asarray(view.vertex_ids).size
                    or view.local_mesh_id is not None
                    or view.local_topology_id is not None
                    or view.local_geometry_id is not None
                )

            view = plan.owner_local_view
            if view is None or not genuinely_empty(view):
                raise ValueError(
                    "A mesh-free FV plan requires a genuinely empty owner-local view."
                )
            version = str(numeric_version)
            if not version:
                raise ValueError("numeric_version must be non-empty.")
            dimension = plan.cell_dimension
            dtype = plan.vertices.dtype
            index_dtype = jnp.int32
            empty_ids = jnp.empty((0,), dtype=index_dtype)
            empty_points = jnp.empty((0, dimension), dtype=dtype)
            empty_quadrature_points = jnp.empty((0, 0, dimension), dtype=dtype)
            empty_quadrature_weights = jnp.empty((0, 0), dtype=dtype)
            empty_cell_quadrature_valid = jnp.empty((0, 0), dtype=jnp.bool_)
            self.mesh = None
            self.vertices = plan.vertices
            self.triangles = plan.triangles
            self.quadrilaterals = plan.quadrilaterals
            self.tetrahedra = plan.tetrahedra
            self.vertex_global_ids = plan.vertex_global_ids
            self.cell_global_ids = plan.cell_global_ids
            self.cell_dimension = dimension
            self.topology = None
            self.connectivity = None
            self.face_block = None
            self.face_blocks = ()
            self.face_topology_ids = None
            self.neighbor_face_topology_ids = None
            self.periodic_face_translations = None
            self.cell_volumes = jnp.empty((0,), dtype=dtype)
            self.cell_centers = empty_points
            self.cell_quadrature_points = empty_quadrature_points
            self.cell_quadrature_weights = empty_quadrature_weights
            self.cell_quadrature_valid = empty_cell_quadrature_valid
            self.cell_quadrature_degree = 0
            self.face_centers = empty_points
            self.area_vectors = empty_points
            self.face_measures = jnp.empty((0,), dtype=dtype)
            self.face_quadrature_points = empty_quadrature_points
            self.face_quadrature_weights = empty_quadrature_weights
            self.owner_cells = empty_ids
            self.owner_signs = jnp.empty((0,), dtype=dtype)
            self.neighbor_cells = empty_ids
            self.boundary_patch_ids = empty_ids
            self.boundary_patch_names = plan.patch_names
            self.topology_id = plan.topology_id
            self.geometry_id = plan.geometry_id
            self.cell_space = None
            self.face_space = None
            self.component_names = plan.component_names
            self.key = plan.key
            self.support = None
            self.field_spaces = ()
            self.measures = ()
            self.capabilities = plan.capabilities
            self.plan_id = plan.plan_id
            self.owner_local_view = view
            self.prepared_id = canonical_fingerprint(
                {
                    "kind": "prepared-empty-owner-local-fv",
                    "plan": plan.plan_id,
                    "view": view.local_view_id,
                    "numeric_version": version,
                }
            )
            self.numeric_version = version
            self.preparation = PreparationReport(
                capabilities=plan.capabilities,
                diagnostics=("owner-local rank has no resident cells or vertices",),
                resource_counts={"cells": 0, "vertices": 0, "faces": 0},
                report_id=canonical_fingerprint(
                    {
                        "kind": "empty-owner-local-fv-preparation",
                        "view": view.local_view_id,
                        "plan": plan.plan_id,
                    }
                ),
            )
            zero = jnp.asarray(0.0, dtype=dtype)
            self.quality = UnstructuredFiniteVolumeQualityReport(
                minimum_cell_measure=zero,
                maximum_cell_measure=zero,
                minimum_face_measure=zero,
                maximum_aspect_ratio=zero,
                maximum_nonorthogonality_degrees=zero,
                maximum_closure_residual=zero,
                worst_cell=jnp.asarray(-1, dtype=jnp.int32),
            )
            return
        points = np.asarray(mesh.coordinates)
        connectivity = mesh.connectivity
        topology = mesh.topology
        cell_count = connectivity.cell_count
        if isinstance(connectivity, PolyhedralConnectivity):
            from ._polyhedral import prepare_polyhedral_finite_volume_geometry

            polyhedral = prepare_polyhedral_finite_volume_geometry(mesh)
            face_count = connectivity.face_owner.size
            owner, neighbor, owner_sign = _owner_neighbor(connectivity, cell_count)
            cell_volumes = polyhedral.cell_volumes
            cell_centers = polyhedral.cell_centers
            face_centers = polyhedral.face_centers
            area_vectors = (
                jnp.asarray(owner_sign, dtype=polyhedral.face_area_vectors.dtype)[:, None]
                * polyhedral.face_area_vectors
            )
            face_measures = polyhedral.face_measures
            closure = (
                jnp.zeros_like(cell_centers).at[:, 0].set(polyhedral.closure_residual)
            )
            quadrature_points = polyhedral.face_quadrature_points
            quadrature_weights = polyhedral.face_quadrature_weights
            cell_quadrature_points = polyhedral.cell_quadrature_points
            cell_quadrature_weights = polyhedral.cell_quadrature_weights
            cell_quadrature_valid = polyhedral.cell_quadrature_valid
        else:
            if plan.cell_dimension == 2:
                # ty: ignore[unresolved-attribute]
                face_count = connectivity.edges.shape[0]
            else:
                # ty: ignore[unresolved-attribute]
                face_count = connectivity.faces.shape[0]
            # ty: ignore[invalid-argument-type]
            owner, neighbor, owner_sign = _owner_neighbor(connectivity, cell_count)
            (
                cell_volumes,
                cell_centers,
                face_centers,
                area_vectors,
                face_measures,
                closure,
                quadrature_points,
                quadrature_weights,
            ) = evaluate_unstructured_fv_geometry(
                plan.vertices,
                plan.triangles,
                plan.quadrilaterals,
                plan.tetrahedra,
                # ty: ignore[invalid-argument-type]
                connectivity,
                owner,
                owner_sign,
            )
            (
                cell_quadrature_points,
                cell_quadrature_weights,
                cell_quadrature_valid,
                # ty: ignore[invalid-argument-type]
            ) = _cell_volume_quadrature(plan, connectivity)
            quadrature_mass = jnp.sum(cell_quadrature_weights, axis=1)
            quadrature_tolerance = (
                256.0
                * jnp.finfo(cell_quadrature_weights.dtype).eps
                * jnp.maximum(jnp.abs(cell_volumes), 1.0)
            )
            cell_quadrature_weights = eqx.error_if(
                cell_quadrature_weights,
                jnp.any(jnp.abs(quadrature_mass - cell_volumes) > quadrature_tolerance),
                "Unstructured cell quadrature must reproduce every cell measure.",
            )
        boundary_patch_ids = np.full((face_count,), -1, dtype=np.int32)
        for patch_id, face_indices in enumerate(plan.patch_faces):
            boundary_patch_ids[np.asarray(face_indices, dtype=np.int32)] = patch_id
        neighbor = np.array(neighbor, dtype=np.int32)
        neighbor_offsets: Array | None = None
        face_topology_ids: Array | None = None
        neighbor_face_topology_ids: Array | None = None
        twins_host = np.arange(face_count, dtype=np.int32)
        periodic_face_translations: Array | None = None
        if plan.periodic_face_joins is not None:
            joins = np.asarray(plan.periodic_face_joins, dtype=np.int32)
            translations = np.asarray(plan.periodic_face_translations, dtype=np.float64)
            kept, merged = joins[:, 0], joins[:, 1]
            joined = joins.reshape(-1)
            if (
                joins.ndim != 2
                or translations.shape != (kept.size, cell_centers.shape[1])
                or np.unique(joined).size != joined.size
                or np.any(joined < 0)
                or np.any(joined >= face_count)
                or np.any(neighbor[joined] >= 0)
                or np.any(boundary_patch_ids[joined] >= 0)
            ):
                raise ValueError(
                    "Periodic seam joins must pair distinct unpatched boundary faces."
                )
            # The kept face carries the quotient face; its merged twin is the
            # same face resident at the neighbor's lattice image.
            neighbor[kept] = owner[merged]
            twins_host[kept] = merged
            offsets_host = np.zeros((face_count, cell_centers.shape[1]), dtype=np.float64)
            offsets_host[kept] = translations
            neighbor_offsets = jnp.asarray(offsets_host, dtype=cell_centers.dtype)
        quality = _quality_report(
            plan,
            connectivity,
            cell_volumes,
            cell_centers,
            area_vectors,
            face_measures,
            closure,
            owner,
            neighbor,
            neighbor_offsets,
        )
        if neighbor_offsets is not None:
            keep_host = np.ones((face_count,), dtype=np.bool_)
            keep_host[np.asarray(plan.periodic_face_joins, dtype=np.int32)[:, 1]] = False
            keep_faces = np.flatnonzero(keep_host).astype(np.int32)
            keep = jnp.asarray(keep_faces)
            owner = np.asarray(owner)[keep_faces]
            neighbor = neighbor[keep_faces]
            owner_sign = np.asarray(owner_sign)[keep_faces]
            boundary_patch_ids = boundary_patch_ids[keep_faces]
            face_centers = face_centers[keep]
            area_vectors = area_vectors[keep]
            face_measures = face_measures[keep]
            quadrature_points = quadrature_points[keep]
            quadrature_weights = quadrature_weights[keep]
            periodic_face_translations = neighbor_offsets[keep]
            face_topology_ids = keep
            neighbor_face_topology_ids = jnp.asarray(twins_host[keep_faces])
            face_count = int(keep_faces.size)
        embedding_id = canonical_fingerprint(
            {
                "kind": "unstructured-fv-embedding",
                "vertices": array_tree_fingerprint(points),
            }
        )
        support = DiscreteSupport(topology, plan.cell_dimension, embedding_id)
        components = len(plan.component_names)
        cell_entities = topology.entity_sets[plan.cell_dimension]
        face_entities = topology.entity_sets[plan.cell_dimension - 1]
        cell_shape = (cell_count, components)
        cell_space = DiscreteFieldSpace(
            plan.field_name,
            support.support_id,
            EntityDofLayout(
                cell_entities.entity_set_id,
                cell_count,
                cell_count,
                component_shape=(components,),
            ),
            ArraySpace(
                cell_shape,
                pairing=DiagonalPairing(
                    jnp.broadcast_to(cell_volumes[:, None], cell_shape)
                ),
            ),
            representation="cell_average",
            conformity="discontinuous",
            reconstruction_id=canonical_fingerprint(
                {"kind": "unstructured-cell-average", "plan": plan.plan_id}
            ),
        )
        face_shape = (face_count, components)
        face_space = DiscreteFieldSpace(
            f"{plan.field_name}_face_flux",
            support.support_id,
            EntityDofLayout(
                face_entities.entity_set_id,
                face_count,
                face_count,
                component_shape=(components,),
            ),
            ArraySpace(
                face_shape,
                pairing=DiagonalPairing(
                    jnp.broadcast_to(face_measures[:, None], face_shape)
                ),
            ),
            representation="flux_moment",
            conformity="Hdiv",
            trace_space_id=cell_space.field_space_id,
        )
        face_block = FiniteVolumeFaceBlock(
            face_ids=jnp.arange(face_count, dtype=jnp.int32),
            owner_cells=jnp.asarray(owner),
            neighbor_cells=jnp.asarray(neighbor),
            boundary_patch_ids=jnp.asarray(boundary_patch_ids),
            face_centers=face_centers,
            area_vectors=area_vectors,
            face_measures=face_measures,
            active_mask=jnp.ones((face_count,), dtype=jnp.bool_),
            block_id=canonical_fingerprint(
                {"kind": "unstructured-face-block", "plan": plan.plan_id}
            ),
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=(
                "cell measures are positive",
                "face area vectors point outward from owners",
                "cell face-vector closure is satisfied",
                "boundary patches are complete",
            ),
            resource_counts={
                "vertices": points.shape[0],
                "faces": face_count,
                "cells": cell_count,
                "boundary_faces": int(np.sum(neighbor < 0)),
            },
        )
        measure_metadata = (
            DiscreteMeasure(
                "unstructured_cell_measure",
                support.support_id,
                cell_entities.entity_set_id,
                cell_volumes,
            ),
            DiscreteMeasure(
                "unstructured_face_measure",
                support.support_id,
                face_entities.entity_set_id,
                face_measures,
            ),
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=support,
            field_spaces=(cell_space, face_space),
            measures=measure_metadata,
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        self.mesh = mesh
        self.vertices = plan.vertices
        self.triangles = plan.triangles
        self.quadrilaterals = plan.quadrilaterals
        self.tetrahedra = plan.tetrahedra
        self.vertex_global_ids = plan.vertex_global_ids
        self.cell_global_ids = plan.cell_global_ids
        self.cell_dimension = plan.cell_dimension
        self.topology = topology
        # ty: ignore[invalid-assignment]
        self.connectivity = connectivity
        self.face_block = face_block
        self.face_blocks = (face_block,)
        self.face_topology_ids = face_topology_ids
        self.neighbor_face_topology_ids = neighbor_face_topology_ids
        self.periodic_face_translations = periodic_face_translations
        self.cell_volumes = cell_volumes
        self.cell_centers = cell_centers
        self.cell_quadrature_points = cell_quadrature_points
        self.cell_quadrature_weights = cell_quadrature_weights
        self.cell_quadrature_valid = cell_quadrature_valid
        self.cell_quadrature_degree = (
            1 if isinstance(connectivity, PolyhedralConnectivity) else 5
        )
        self.face_centers = face_centers
        self.area_vectors = area_vectors
        self.face_measures = face_measures
        self.face_quadrature_points = quadrature_points
        self.face_quadrature_weights = quadrature_weights
        self.owner_cells = jnp.asarray(owner)
        self.owner_signs = jnp.asarray(owner_sign)
        self.neighbor_cells = jnp.asarray(neighbor)
        self.boundary_patch_ids = jnp.asarray(boundary_patch_ids)
        self.boundary_patch_names = plan.patch_names
        self.cell_space = cell_space
        self.face_space = face_space
        self.component_names = plan.component_names
        self.topology_id = plan.topology_id
        self.geometry_id = plan.geometry_id
        self.key = plan.key
        self.support = support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.owner_local_view = plan.owner_local_view
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-finite-volume",
                "plan": plan.plan_id,
                "topology": plan.topology_id,
                "geometry": plan.geometry_id,
                "numeric_version": version,
            }
        )
        self.numeric_version = version
        self.preparation = preparation
        self.quality = quality

    @property
    def cell_count(self) -> int:
        return self.cell_volumes.size

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.cell_count, self.component_count)

    def neighbor_side_points(
        self, points: Array, face_ids: Array | None = None, /
    ) -> Array:
        """Face points ``(faces, points, dimension)`` in each neighbor's own frame.

        Owner-local periodic seam faces join a neighbor resident at another
        lattice image: a face point ``p`` in the owner's frame is
        ``p + periodic_face_translations[face]`` beside the neighbor. Without
        seams the points are returned unchanged. ``face_ids`` selects the faces
        of ``points`` when they are a subset of this discretization's faces.
        """
        values = jnp.asarray(points)
        if self.periodic_face_translations is None:
            return values
        translations = self.periodic_face_translations
        if face_ids is not None:
            translations = translations[jnp.asarray(face_ids, dtype=jnp.int32)]
        return values + translations.astype(values.dtype)[:, None, :]

    def directional_control_volume_widths(self) -> Array:
        """Return volume-normalized directional widths from control-volume faces."""

        dtype = self.cell_volumes.dtype
        if self.cell_dimension == 2 and self.vertices.shape[-1] == 3:
            cells = self.vertices[jnp.asarray(self.triangles, dtype=jnp.int32)]
            first = cells[:, 1] - cells[:, 0]
            second = cells[:, 2] - cells[:, 0]
            normal_area = jnp.cross(first, second)
            normals = normal_area / jnp.linalg.norm(normal_area, axis=-1, keepdims=True)
            first_direction = first / jnp.linalg.norm(first, axis=-1, keepdims=True)
            second_direction = jnp.cross(normals, first_direction)
            tangents = jnp.stack(
                (
                    first,
                    cells[:, 2] - cells[:, 1],
                    cells[:, 0] - cells[:, 2],
                ),
                axis=1,
            )
            conormals = jnp.cross(tangents, normals[:, None, :])
            projected_area = 0.5 * jnp.stack(
                (
                    jnp.sum(
                        jnp.abs(
                            jnp.sum(conormals * first_direction[:, None, :], axis=-1)
                        ),
                        axis=1,
                    ),
                    jnp.sum(
                        jnp.abs(
                            jnp.sum(conormals * second_direction[:, None, :], axis=-1)
                        ),
                        axis=1,
                    ),
                ),
                axis=1,
            )
            projected_area = eqx.error_if(
                projected_area,
                jnp.any(~jnp.isfinite(projected_area) | (projected_area <= 0.0)),
                "Intrinsic surface projected areas must be positive and finite.",
            )
            area = self.cell_volumes.astype(dtype)
            raw_widths = area[:, None] / projected_area
            normalization = (area / jnp.prod(raw_widths, axis=-1)) ** 0.5
            return raw_widths * normalization[:, None]
        projected_area = jnp.zeros(
            (self.cell_count, self.cell_dimension),
            dtype=dtype,
        )
        face_projection = jnp.abs(self.area_vectors.astype(dtype))
        owner = self.owner_cells
        neighbor = self.neighbor_cells
        interior = neighbor >= 0
        projected_area = projected_area.at[owner].add(0.5 * face_projection)
        projected_area = projected_area.at[jnp.maximum(neighbor, 0)].add(
            jnp.where(interior[:, None], 0.5 * face_projection, 0.0)
        )
        projected_area = eqx.error_if(
            projected_area,
            jnp.any(~jnp.isfinite(projected_area) | (projected_area <= 0.0)),
            "Directional control-volume projected areas must be positive and finite.",
        )
        volume = self.cell_volumes.astype(dtype)
        raw_widths = volume[:, None] / projected_area
        raw_product = jnp.prod(raw_widths, axis=-1)
        normalization = (volume / raw_product) ** (1.0 / self.cell_dimension)
        return raw_widths * normalization[:, None]

    def integration_domain(
        self, kind: str, selection: EntitySelection | None = None, /
    ) -> IntegrationDomain:
        """Cell or facet domain in the face table's order and orientation.

        The owner of each face is the cell its stored area vector points out
        of; local facets index each cell's edge or face table.
        """
        from ._side_trace import finite_volume_integration_domain

        return finite_volume_integration_domain(self, kind, selection)

    def prepare_side_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
        reconstruction: FiniteVolumeFaceReconstruction | None = None,
    ) -> PreparedTraceAction:
        """Prepare the cell-average or linear face state on selected faces.

        `reconstruction=None` publishes the side cell average at every site
        (`representation="cell-average"`); a
        `PreparedCellPolynomialReconstruction` publishes the k-exact face state
        (`"face-state"`, `trace_degree=k`) through a per-facet stencil route
        with an exact transpose. Sites follow the owner's facet
        parametrization (edges or triangles) and are shared by both sides of an
        interior face; normals point out of the side cell. Polyhedral faces and
        nonlinear reconstructions are refused.
        """
        from ._side_trace import prepare_finite_volume_side_trace

        return prepare_finite_volume_side_trace(
            self,
            field_name,
            domain,
            rule=rule,
            quantity=quantity,
            side=side,
            reconstruction=reconstruction,
        )

    def prepare_nonlinear_face_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        reconstruction: FiniteVolumeFaceReconstruction,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
    ) -> PreparedNonlinearFaceTrace:
        """Prepare WENO-Z face states with a linearization at a supplied state."""
        from ._side_trace import prepare_finite_volume_nonlinear_face_trace

        return prepare_finite_volume_nonlinear_face_trace(
            self,
            field_name,
            domain,
            rule=rule,
            reconstruction=reconstruction,
            quantity=quantity,
            side=side,
        )


def _quality_report(
    plan: Any,
    connectivity: Any,
    cell_volumes: Any,
    cell_centers: Any,
    area_vectors: Any,
    face_measures: Any,
    closure: Any,
    owner: Any,
    neighbor: Any,
    neighbor_offsets: Any = None,
) -> Any:
    owner_ = jnp.asarray(owner, dtype=jnp.int32)
    neighbor_ = jnp.asarray(neighbor, dtype=jnp.int32)
    interior = neighbor_ >= 0
    connector = cell_centers[jnp.maximum(neighbor_, 0)] - cell_centers[owner_]
    if neighbor_offsets is not None:
        # Seam neighbors sit at another lattice image; compare in the owner's frame.
        connector = connector - neighbor_offsets
    denominator = jnp.linalg.norm(connector, axis=-1) * face_measures
    cosine = jnp.abs(jnp.sum(connector * area_vectors, axis=-1)) / jnp.where(
        denominator > 0.0, denominator, 1.0
    )
    maximum_nonorthogonality = jnp.max(
        jnp.where(
            interior,
            jnp.degrees(jnp.arccos(jnp.clip(cosine, 0.0, 1.0))),
            0.0,
        )
    )
    points = jnp.asarray(plan.vertices)
    if isinstance(connectivity, PolygonalConnectivity):
        cell_edges = jnp.asarray(connectivity.cell_edges, dtype=jnp.int32)
        valid = jnp.asarray(connectivity.cell_edge_valid)
        lengths = jnp.linalg.norm(
            points[jnp.asarray(connectivity.edges)[:, 1]]
            - points[jnp.asarray(connectivity.edges)[:, 0]],
            axis=-1,
        )
        cell_maximum = jnp.max(jnp.where(valid, lengths[cell_edges], 0.0), axis=1)
        aspect = cell_maximum**2 / cell_volumes
    elif isinstance(connectivity, TetrahedralConnectivity):
        cells = jnp.asarray(plan.tetrahedra, dtype=jnp.int32)
        cell_points = points[cells]
        pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
        edge_lengths = jnp.stack(
            tuple(
                jnp.linalg.norm(cell_points[:, right] - cell_points[:, left], axis=-1)
                for left, right in pairs
            ),
            axis=1,
        )
        cell_faces = jnp.asarray(connectivity.cell_faces, dtype=jnp.int32)
        maximum_face = jnp.max(face_measures[cell_faces], axis=1)
        minimum_altitude = 3.0 * cell_volumes / maximum_face
        aspect = jnp.max(edge_lengths, axis=1) / minimum_altitude
    else:
        cell_faces = jnp.asarray(connectivity.cell_face_values, dtype=jnp.int32)
        counts = np.diff(np.asarray(connectivity.cell_face_offsets, dtype=np.int32))
        cell_ids = jnp.asarray(
            np.repeat(np.arange(connectivity.cell_count, dtype=np.int32), counts)
        )
        maximum_face_scale = jax.ops.segment_max(
            jnp.sqrt(face_measures[cell_faces]),
            cell_ids,
            num_segments=connectivity.cell_count,
        )
        aspect = maximum_face_scale / jnp.cbrt(cell_volumes)
    return UnstructuredFiniteVolumeQualityReport(
        minimum_cell_measure=jnp.min(cell_volumes),
        maximum_cell_measure=jnp.max(cell_volumes),
        minimum_face_measure=jnp.min(face_measures),
        maximum_aspect_ratio=jnp.max(aspect),
        maximum_nonorthogonality_degrees=maximum_nonorthogonality,
        maximum_closure_residual=jnp.max(jnp.linalg.norm(closure, axis=-1)),
        worst_cell=jnp.argmax(aspect),
    )


__all__ = [
    "MaskedFiniteVolumeConservation",
    "MaskedFiniteVolumeGeometry",
    "UnstructuredFiniteVolumeDiscretization",
    "UnstructuredFiniteVolumePlan",
    "UnstructuredFiniteVolumeQualityReport",
    "evaluate_masked_fv_conservation",
    "evaluate_masked_fv_geometry",
    "evaluate_unstructured_fv_geometry",
    "masked_fv_flux_divergence",
]
