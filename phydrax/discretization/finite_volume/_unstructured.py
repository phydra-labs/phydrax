#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import ArraySpace, DiagonalPairing
from .._cell_complex import (
    polygonal_connectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from .._cell_mesh import CellBlock, CellMesh
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    AbstractPreparedDiscretization,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import CellComplexTopology
from ._geometry_protocol import FiniteVolumeFaceBlock
from ._structured import _component_names


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
    if not bool(jax.config.read("jax_enable_x64")) and np.any(
        identifiers > np.iinfo(np.int32).max
    ):
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
    tolerance = 64.0 * np.finfo(float).eps * scale**2
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


def _quadrilateral_shape_data(points: np.ndarray, /):
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
        triangle_jacobian = (triangle_points[:, 1, 0] - triangle_points[:, 0, 0]) * (
            triangle_points[:, 2, 1] - triangle_points[:, 0, 1]
        ) - (triangle_points[:, 1, 1] - triangle_points[:, 0, 1]) * (
            triangle_points[:, 2, 0] - triangle_points[:, 0, 0]
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
        jnp.ones(quadrature_weights.shape, dtype=bool),
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
    tolerance = 64.0 * np.finfo(float).eps * scale**2
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
    tolerance = 128.0 * np.finfo(float).eps * scale**3
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


def _owner_neighbour(connectivity: Connectivity, cell_count: int, /):
    if isinstance(connectivity, PolygonalConnectivity):
        cell_faces = np.asarray(connectivity.cell_edges, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_edge_signs)
        valid = np.asarray(connectivity.cell_edge_valid, dtype=bool)
        face_count = int(connectivity.edges.shape[0])
    elif isinstance(connectivity, TetrahedralConnectivity):
        cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_face_signs)
        valid = np.ones(cell_faces.shape, dtype=bool)
        face_count = int(connectivity.faces.shape[0])
    elif isinstance(connectivity, PolyhedralConnectivity):
        cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_face_sign_values)
        cell_ids = np.repeat(
            np.arange(cell_count, dtype=np.int32),
            np.diff(np.asarray(connectivity.cell_face_offsets, dtype=np.int32)),
        )
        owner = np.asarray(connectivity.face_owner, dtype=np.int32)
        neighbour = np.asarray(connectivity.face_neighbour, dtype=np.int32)
        owner_sign = np.zeros((connectivity.face_count,), dtype=float)
        for cell, face, sign in zip(cell_ids, cell_faces, cell_signs, strict=True):
            if owner[int(face)] == int(cell):
                owner_sign[int(face)] = float(sign)
        return owner, neighbour, owner_sign
    else:
        cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
        cell_signs = np.asarray(connectivity.cell_face_signs)
        valid = np.ones(cell_faces.shape, dtype=bool)
        face_count = int(connectivity.faces.shape[0])
    owner = np.full((face_count,), -1, dtype=np.int32)
    neighbour = np.full((face_count,), -1, dtype=np.int32)
    owner_sign = np.zeros((face_count,), dtype=float)
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
                if neighbour[face] >= 0:
                    raise ValueError(
                        "Unstructured cells must be codimension-one manifold."
                    )
                if sign == owner_sign[face]:
                    raise ValueError(
                        "Shared faces must have opposite incidence orientation."
                    )
                neighbour[face] = cell
    if np.any(owner < 0):
        raise ValueError("Every unstructured face must have an owner cell.")
    return owner, neighbour, owner_sign


def _polygon_geometry(
    vertices: ArrayLike,
    triangles: ArrayLike,
    quadrilaterals: ArrayLike,
    connectivity: PolygonalConnectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
):
    points = jnp.asarray(vertices)
    triangle_cells = jnp.asarray(triangles, dtype=jnp.int32)
    quadrilateral_cells = jnp.asarray(quadrilaterals, dtype=jnp.int32)
    triangle_points = points[triangle_cells]
    triangle_cross = (triangle_points[:, 1, 0] - triangle_points[:, 0, 0]) * (
        triangle_points[:, 2, 1] - triangle_points[:, 0, 1]
    ) - (triangle_points[:, 1, 1] - triangle_points[:, 0, 1]) * (
        triangle_points[:, 2, 0] - triangle_points[:, 0, 0]
    )
    triangle_volume = 0.5 * triangle_cross
    triangle_center = jnp.mean(triangle_points, axis=1)

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
    canonical_area = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=-1)
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
):
    points = jnp.asarray(vertices)
    cells = jnp.asarray(tetrahedra, dtype=jnp.int32)
    cell_points = points[cells]
    determinant = jnp.linalg.det(
        jnp.stack(
            (
                cell_points[:, 1] - cell_points[:, 0],
                cell_points[:, 2] - cell_points[:, 0],
                cell_points[:, 3] - cell_points[:, 0],
            ),
            axis=-1,
        )
    )
    cell_volumes = determinant / 6.0
    cell_volumes = eqx.error_if(
        cell_volumes,
        jnp.any(~jnp.isfinite(cell_volumes) | (cell_volumes <= 0.0)),
        "Tetrahedral FV geometry requires positive finite cell volumes.",
    )
    cell_centers = jnp.mean(cell_points, axis=1)
    faces = jnp.asarray(connectivity.faces, dtype=jnp.int32)
    face_points = points[faces]
    face_centers = jnp.mean(face_points, axis=1)
    canonical_area = 0.5 * jnp.cross(
        face_points[:, 1] - face_points[:, 0],
        face_points[:, 2] - face_points[:, 0],
    )
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
):
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
    return _tetrahedral_geometry(vertices, tetrahedra, connectivity, owner, owner_sign)


class UnstructuredFiniteVolumeQualityReport(StrictModule):
    minimum_cell_measure: Array
    maximum_cell_measure: Array
    minimum_face_measure: Array
    maximum_aspect_ratio: Array
    maximum_nonorthogonality_degrees: Array
    maximum_closure_residual: Array
    worst_cell: Array


class UnstructuredFiniteVolumePlan(AbstractDiscretizationPlan):
    """Fixed-topology triangular, quadrilateral, mixed, or tetrahedral FV plan."""

    mesh: CellMesh

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

    def __init__(
        self,
        vertices: ArrayLike,
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
    ):
        points = np.asarray(vertices, dtype=float)
        if points.ndim != 2 or points.shape[1] not in (2, 3):
            raise ValueError("Unstructured FV vertices must have shape (n, 2) or (n, 3).")
        if points.shape[0] < points.shape[1] + 1 or np.any(~np.isfinite(points)):
            raise ValueError("Unstructured FV vertices must be finite and nonempty.")
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
            boundary_mask = np.asarray(connectivity.boundary_edges, dtype=bool)
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
            boundary_mask = np.asarray(connectivity.boundary_faces, dtype=bool)
        cell_count = connectivity.cell_count

        patches = {} if boundary_patches is None else dict(boundary_patches)
        if not patches:
            patches = {"boundary": face_vertices[boundary_mask]}
        names = tuple(sorted(str(name) for name in patches))
        lookup = {tuple(sorted(face)): index for index, face in enumerate(face_vertices)}
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
        global_id_dtype = (
            jnp.int64 if bool(jax.config.read("jax_enable_x64")) else jnp.int32
        )
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

    @classmethod
    def from_cell_mesh(
        cls,
        mesh: CellMesh,
        /,
        *,
        field_name: str = "state",
        component_names: Sequence[str] = ("value",),
    ) -> "UnstructuredFiniteVolumePlan":
        """Construct directly from canonical polyhedral CellMesh."""
        if not isinstance(mesh, CellMesh):
            raise TypeError("mesh must be CellMesh.")
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
        face_neighbour = np.asarray(connectivity.face_neighbour, dtype=np.int32)
        boundary_faces = np.flatnonzero(face_neighbour < 0).astype(np.int32)
        topology_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-topology",
                "mesh": mesh.topology_id,
                "field": field,
                "components": list(components),
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
        result = object.__new__(cls)
        object.__setattr__(result, "mesh", mesh)
        object.__setattr__(result, "vertices", mesh.coordinates)
        object.__setattr__(result, "triangles", jnp.empty((0, 3), dtype=jnp.int32))
        object.__setattr__(result, "quadrilaterals", jnp.empty((0, 4), dtype=jnp.int32))
        tetrahedra = jnp.empty((0, 4), dtype=jnp.int32)
        object.__setattr__(result, "tetrahedra", tetrahedra)
        object.__setattr__(result, "vertex_global_ids", jnp.asarray(vertex_ids))
        object.__setattr__(result, "cell_global_ids", jnp.asarray(cell_ids))
        object.__setattr__(result, "cell_dimension", 3)
        object.__setattr__(result, "patch_names", ("boundary",))
        object.__setattr__(result, "patch_faces", (jnp.asarray(boundary_faces),))
        object.__setattr__(result, "field_name", field)
        object.__setattr__(result, "component_names", components)
        object.__setattr__(result, "topology_id", topology_id)
        object.__setattr__(result, "geometry_id", geometry_id)
        object.__setattr__(
            result,
            "key",
            DiscretizationKey("unstructured_finite_volume", DiscretizationRole.PHYSICAL),
        )
        object.__setattr__(result, "capabilities", capabilities)
        object.__setattr__(
            result,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "unstructured-finite-volume-plan",
                    "topology": topology_id,
                    "geometry": geometry_id,
                }
            ),
        )
        return result

    def prepare(self, /, *, numeric_version: str = "0"):
        return UnstructuredFiniteVolumeDiscretization(
            self, numeric_version=numeric_version
        )


class UnstructuredFiniteVolumeDiscretization(AbstractPreparedDiscretization):
    mesh: CellMesh
    vertices: Array
    triangles: Array
    quadrilaterals: Array
    tetrahedra: Array
    vertex_global_ids: Array
    cell_global_ids: Array
    cell_dimension: int = eqx.field(static=True)
    topology: CellComplexTopology
    connectivity: Connectivity
    face_block: FiniteVolumeFaceBlock
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
    neighbour_cells: Array
    boundary_patch_ids: Array
    boundary_patch_names: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    cell_space: DiscreteFieldSpace
    face_space: DiscreteFieldSpace
    component_names: tuple[str, ...] = eqx.field(static=True)
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    quality: UnstructuredFiniteVolumeQualityReport

    def __init__(
        self, plan: UnstructuredFiniteVolumePlan, /, *, numeric_version: str = "0"
    ):
        if not isinstance(plan, UnstructuredFiniteVolumePlan):
            raise TypeError("plan must be UnstructuredFiniteVolumePlan.")
        mesh = plan.mesh
        points = np.asarray(mesh.coordinates)
        connectivity = mesh.connectivity
        topology = mesh.topology
        cell_count = connectivity.cell_count
        if isinstance(connectivity, PolyhedralConnectivity):
            from ._polyhedral import prepare_polyhedral_finite_volume_geometry

            polyhedral = prepare_polyhedral_finite_volume_geometry(mesh)
            face_count = int(connectivity.face_owner.size)
            owner, neighbour, owner_sign = _owner_neighbour(connectivity, cell_count)
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
                face_count = int(connectivity.edges.shape[0])
            else:
                face_count = int(connectivity.faces.shape[0])
            owner, neighbour, owner_sign = _owner_neighbour(connectivity, cell_count)
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
                connectivity,
                owner,
                owner_sign,
            )
            (
                cell_quadrature_points,
                cell_quadrature_weights,
                cell_quadrature_valid,
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
            neighbour_cells=jnp.asarray(neighbour),
            boundary_patch_ids=jnp.asarray(boundary_patch_ids),
            face_centers=face_centers,
            area_vectors=area_vectors,
            face_measures=face_measures,
            active_mask=jnp.ones((face_count,), dtype=bool),
            block_id=canonical_fingerprint(
                {"kind": "unstructured-face-block", "plan": plan.plan_id}
            ),
        )
        quality = _quality_report(
            plan,
            connectivity,
            cell_volumes,
            cell_centers,
            area_vectors,
            face_measures,
            closure,
            owner,
            neighbour,
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
                "boundary_faces": int(np.sum(neighbour < 0)),
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
        self.connectivity = connectivity
        self.face_block = face_block
        self.face_blocks = (face_block,)
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
        self.neighbour_cells = jnp.asarray(neighbour)
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
        return int(self.cell_volumes.size)

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    @property
    def state_shape(self) -> tuple[int, ...]:
        return (self.cell_count, self.component_count)

    def directional_control_volume_widths(self) -> Array:
        """Return volume-normalized directional widths from control-volume faces."""

        dtype = self.cell_volumes.dtype
        projected_area = jnp.zeros(
            (self.cell_count, self.cell_dimension),
            dtype=dtype,
        )
        face_projection = jnp.abs(self.area_vectors.astype(dtype))
        owner = self.owner_cells
        neighbour = self.neighbour_cells
        interior = neighbour >= 0
        projected_area = projected_area.at[owner].add(0.5 * face_projection)
        projected_area = projected_area.at[jnp.maximum(neighbour, 0)].add(
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


def _quality_report(
    plan,
    connectivity,
    cell_volumes,
    cell_centers,
    area_vectors,
    face_measures,
    closure,
    owner,
    neighbour,
):
    owner_ = jnp.asarray(owner, dtype=jnp.int32)
    neighbour_ = jnp.asarray(neighbour, dtype=jnp.int32)
    interior = neighbour_ >= 0
    connector = cell_centers[jnp.maximum(neighbour_, 0)] - cell_centers[owner_]
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
    "UnstructuredFiniteVolumeDiscretization",
    "UnstructuredFiniteVolumePlan",
    "UnstructuredFiniteVolumeQualityReport",
    "evaluate_unstructured_fv_geometry",
]
