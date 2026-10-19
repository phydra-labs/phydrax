#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from typing import assert_never, final, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._mapped_geometry_measure as mapped_measure_owner
import phydrax.discretization.fem._reference_rules as reference_rules_owner
import phydrax.ein as ein
from phydrax.discretization._reference_cell_types import CellBlock

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ..._trainable import fixed_field
from ...ein import contract
from ...geometry._global_embedding import GlobalEmbeddingCertificate
from ...linalg import ArraySpace, DiagonalPairing
from ...sparse import EdgeRelation, SparseLinearMap
from ...typing import checked
from .. import (
    _cell_geometry as imported_cell_geometry,
    _reference_cell as imported_reference_cell,
)
from .._adaptive_simplex import MaskedSimplexMesh
from .._cell_complex import (
    IntervalConnectivity,
    polygonal_connectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    tetrahedral_connectivity,
    TetrahedralConnectivity,
)
from .._cell_geometry import (
    cell_geometry_preparation_bounds,
    cell_geometry_preparation_query_upper,
    CellGeometrySpec,
    owned_coordinate_preparation,
)
from .._cell_geometry_validity import cell_geometry_id
from .._cell_mesh import CellMesh, SimplicialConnectivity
from .._conservation_ledger import (
    ConservationStageFluxRateBlock,
    ConservationStageLedger,
)
from .._coordinate_budget import CoordinateEnclosureBudget
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._hexahedral import HexahedralConnectivity
from .._integration_domain import IntegrationDomain
from .._lifecycle import (
    AbstractDiscretizationPlan,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._side_actions import FacetTraceRule, PreparedTraceAction, SideTraceQuantity
from .._spaces import DiscreteFieldSpace, EntityDofLayout
from .._support import DiscreteSupport
from .._topology import CellComplexTopology, EntitySelection, EntitySet
from .._views import FieldTraceSide
from . import _polyhedral as imported_polyhedral, _side_trace as imported_side_trace
from ._coordinate_source import has_power_source
from ._geometry_protocol import FiniteVolumeFaceBlock
from ._structured import normalize_component_names
from ._unstructured_types import AbstractUnstructuredFiniteVolumeDiscretization


Connectivity = (
    PolygonalConnectivity
    | TetrahedralConnectivity
    | HexahedralConnectivity
    | PolyhedralConnectivity
)

TETRAHEDRAL_FACE_QUADRATURE_BARYCENTRIC = (
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
            raise ValueError(
                f"UnstructuredFiniteVolumePlan.{name} must contain one integer ID per entity; got shape {raw.shape!r} and dtype {raw.dtype}."
            )
        if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
            raise ValueError(
                f"UnstructuredFiniteVolumePlan.{name} must be representable as signed int64; got values above the signed range."
            )
        identifiers = raw.astype(np.int64, copy=False)
    if np.any(identifiers < 0) or np.unique(identifiers).size != count:
        raise ValueError(
            f"UnstructuredFiniteVolumePlan.{name} must contain unique nonnegative IDs; got negative or repeated IDs."
        )
    if not x64_enabled() and np.any(identifiers > np.iinfo(np.int32).max):
        raise ValueError(
            f"UnstructuredFiniteVolumePlan.{name} must fit signed int32 when JAX x64 is disabled; got IDs outside the active signed range."
        )
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
        raise ValueError(
            "UnstructuredFiniteVolumePlan.triangles must have finite nonzero area; got a degenerate or nonfinite cell."
        )
    normalized = cells.copy()
    reverse = signed_twice_area < 0.0
    normalized[reverse, 1], normalized[reverse, 2] = (
        normalized[reverse, 2].copy(),
        normalized[reverse, 1].copy(),
    )
    if np.unique(np.sort(normalized, axis=1), axis=0).shape[0] != normalized.shape[0]:
        raise ValueError(
            "UnstructuredFiniteVolumePlan.triangles must contain unique vertex sets; got duplicate cells."
        )
    return normalized


def _quadrilateral_shape_data(points: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
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
        raise ValueError(
            "UnstructuredFiniteVolumePlan.quadrilaterals must have finite nonzero signed area; got a degenerate or nonfinite cell."
        )
    reverse = signed_twice_area < 0.0
    normalized[reverse] = normalized[reverse][:, [0, 3, 2, 1]]
    points = vertices[normalized]
    turns = _cross_2d(
        np.roll(points, -1, axis=1) - points,
        np.roll(points, -2, axis=1) - np.roll(points, -1, axis=1),
    )
    if np.any(turns <= tolerance[:, None]):
        raise ValueError(
            "UnstructuredFiniteVolumePlan.quadrilaterals must be simple and strictly convex; got a crossed or nonconvex cell."
        )
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
        raise ValueError(
            "UnstructuredFiniteVolumePlan.quadrilaterals must have positive finite bilinear Jacobians; got an invalid cell map."
        )
    if np.unique(np.sort(normalized, axis=1), axis=0).shape[0] != normalized.shape[0]:
        raise ValueError(
            "UnstructuredFiniteVolumePlan.quadrilaterals must contain unique vertex sets; got duplicate cells."
        )
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
        raise ValueError(
            "UnstructuredFiniteVolumePlan.tetrahedra must have finite nonzero volume; got a degenerate or nonfinite cell."
        )
    normalized = cells.copy()
    reverse = determinant < 0.0
    normalized[reverse, 1], normalized[reverse, 2] = (
        normalized[reverse, 2].copy(),
        normalized[reverse, 1].copy(),
    )
    if np.unique(np.sort(normalized, axis=1), axis=0).shape[0] != normalized.shape[0]:
        raise ValueError(
            "UnstructuredFiniteVolumePlan.tetrahedra must contain unique vertex sets; got duplicate cells."
        )
    return normalized


def _owner_neighbor(
    connectivity: Connectivity, cell_count: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    match connectivity:
        case PolygonalConnectivity():
            cell_faces = np.asarray(connectivity.cell_edges, dtype=np.int32)
            cell_signs = np.asarray(connectivity.cell_edge_signs)
            valid = np.asarray(connectivity.cell_edge_valid, dtype=np.bool_)
            face_count = connectivity.edges.shape[0]
        case TetrahedralConnectivity() | HexahedralConnectivity():
            cell_faces = np.asarray(connectivity.cell_faces, dtype=np.int32)
            cell_signs = np.asarray(connectivity.cell_face_signs)
            valid = np.ones(cell_faces.shape, dtype=np.bool_)
            face_count = connectivity.faces.shape[0]
        case PolyhedralConnectivity():
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
        case _:
            assert_never(connectivity)
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
                        "UnstructuredFiniteVolumePlan.connectivity must be codimension-one manifold; got more than two cells on a face."
                    )
                if sign == owner_sign[face]:
                    raise ValueError(
                        "UnstructuredFiniteVolumePlan.connectivity must orient shared face incidences oppositely; got matching signs."
                    )
                neighbor[face] = cell
    if np.any(owner < 0):
        raise ValueError(
            "UnstructuredFiniteVolumePlan.connectivity must assign an owner to every face; got an unowned face."
        )
    return owner, neighbor, owner_sign


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


def _polygon_geometry(
    vertices: ArrayLike,
    triangles: ArrayLike,
    quadrilaterals: ArrayLike,
    connectivity: PolygonalConnectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    points = jnp.asarray(vertices)
    triangle_cells = jnp.asarray(triangles, dtype=jnp.int32)
    quadrilateral_cells = jnp.asarray(quadrilaterals, dtype=jnp.int32)
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
    connectivity: TetrahedralConnectivity | HexahedralConnectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
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
    barycentric = jnp.asarray(TETRAHEDRAL_FACE_QUADRATURE_BARYCENTRIC, dtype=points.dtype)
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


@checked
def evaluate_unstructured_fv_geometry(
    vertices: ArrayLike,
    triangles: ArrayLike,
    quadrilaterals: ArrayLike,
    tetrahedra: ArrayLike,
    connectivity: Connectivity,
    owner: ArrayLike,
    owner_sign: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    """Evaluate owner-oriented corner geometry for a polygonal or simplex complex."""
    match connectivity:
        case PolygonalConnectivity():
            return _polygon_geometry(
                vertices, triangles, quadrilaterals, connectivity, owner, owner_sign
            )
        case TetrahedralConnectivity() | HexahedralConnectivity():
            return _tetrahedral_geometry(
                vertices, tetrahedra, connectivity, owner, owner_sign
            )
        case PolyhedralConnectivity():
            raise TypeError(
                "evaluate_unstructured_fv_geometry.connectivity must support corner simplex geometry; got polyhedral connectivity."
            )
        case _:
            assert_never(connectivity)


@contextmanager
def _fv_preparation_budget(
    mesh: CellMesh,
    geometry: CellGeometrySpec | None,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> Iterator[CoordinateEnclosureBudget]:
    if budget is not None:
        yield budget
        return
    match geometry:
        case CellGeometrySpec():
            # These are the certified-measure owner's preceding standalone maxima,
            # not meshing limits. Source preparations count queries independently.
            work = 100_000_000
            memory = sys.maxsize
            queries = cell_geometry_preparation_query_upper(geometry) * (work + 1)
        case None:
            match mesh.connectivity:
                case PolyhedralConnectivity():
                    work, memory, queries = (
                        imported_polyhedral.polyhedral_finite_volume_preparation_bounds(
                            mesh, maximum_workset_entries=100_000_000
                        )
                    )
                case (
                    IntervalConnectivity()
                    | PolygonalConnectivity()
                    | TetrahedralConnectivity()
                    | HexahedralConnectivity()
                    | SimplicialConnectivity()
                ):
                    work, memory, queries = 0, 0, 0
                case invalid_connectivity:
                    assert_never(invalid_connectivity)
        case _:
            assert_never(geometry)
    with owned_coordinate_preparation(
        work, memory, maximum_geometry_queries=queries
    ) as owner:
        yield owner


class _MappedFiniteVolumeGeometry(NamedTuple):
    """Mapped geometry and certified measure evidence in kernel return order."""

    cell_volumes: Array
    cell_centers: Array
    face_centers: Array
    area_vectors: Array
    face_measures: Array
    closure: Array
    face_quadrature_points: Array
    face_quadrature_weights: Array
    cell_quadrature_points: Array
    cell_quadrature_weights: Array
    cell_quadrature_valid: Array
    volume_errors: Array
    volume_exact: bool


def _mapped_fv_geometry(
    mesh: CellMesh,
    geometry: CellGeometrySpec,
    owner: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _MappedFiniteVolumeGeometry:
    """Evaluate the represented coordinate map; cell measures have certified bounds."""

    volumes, errors, exact = mapped_measure_owner.certified_cell_measures(
        mesh, geometry, budget=budget
    )
    elements, routes, coordinates = geometry.resolve(mesh, budget=budget)
    elements = tuple(
        imported_cell_geometry.require_scalar_coordinate_element(
            element, "Mapped FV geometry"
        )
        for element in elements
    )
    coefficients = np.asarray(coordinates, dtype=np.float64)
    dimension, ambient = mesh.topological_dimension, mesh.coordinates.shape[1]
    embedded = dimension == 2 and ambient == 3
    cells = []
    cell_points, cell_weights, centers = [], [], []
    degree = max(9 * element.degree for element in elements)
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        reference = imported_reference_cell.reference_cell_topology(block.cell_kind)
        xi, weights = reference_rules_owner.degree_aware_reference_rule(
            block.cell_kind, degree
        )
        values, gradients = element.tabulate(xi)
        values, gradients = np.asarray(values), np.asarray(gradients)
        for vertices, dofs in zip(
            np.asarray(block.vertices), np.asarray(route), strict=True
        ):
            local = coefficients[dofs]
            points = values @ local
            jacobian = np.einsum("mD,qmd->qDd", local, gradients)
            density = (
                np.sqrt(np.linalg.det(np.swapaxes(jacobian, 1, 2) @ jacobian))
                if embedded
                else np.linalg.det(jacobian)
            )
            measure_weights = np.asarray(weights) * density
            if np.any(~np.isfinite(measure_weights)) or np.any(measure_weights <= 0):
                raise ValueError(
                    "UnstructuredFiniteVolumeDiscretization.cell_quadrature_weights must have positive finite mapped densities; got a nonpositive or nonfinite weight."
                )
            cell_points.append(points)
            cell_weights.append(measure_weights)
            centers.append(
                np.sum(points * measure_weights[:, None], axis=0)
                / np.sum(measure_weights)
            )
            cells.append((vertices, reference, element, local))
    match mesh.connectivity:
        case PolyhedralConnectivity() as connectivity:
            offsets = np.asarray(connectivity.face_vertex_offsets)
            indices = np.asarray(connectivity.face_vertex_values)
            faces = [indices[offsets[f] : offsets[f + 1]] for f in range(owner.size)]
        case PolygonalConnectivity() as connectivity:
            faces = list(np.asarray(connectivity.edges))
        case (TetrahedralConnectivity() | HexahedralConnectivity()) as connectivity:
            faces = list(np.asarray(connectivity.faces))
        case _:
            raise TypeError(
                "UnstructuredFiniteVolumePlan.connectivity must be polygonal or volumetric; got unsupported connectivity."
            )
    face_points, face_weights, face_centers, area_vectors = [], [], [], []
    closure = np.zeros((len(cells), ambient), dtype=np.float64)
    for face, cell in zip(faces, owner, strict=True):
        vertices, reference, element, local = cells[int(cell)]
        facet = next(
            f
            for f in reference.entities[mesh.topological_dimension - 1]
            if set(vertices[list(f)]) == set(face)
        )
        corners = np.asarray(reference.vertices)[list(facet)]
        if len(facet) == 2:
            uv, w = reference_rules_owner.degree_aware_reference_rule("interval", degree)
            xi = corners[0] + np.asarray(uv)[:, :1] * (corners[1] - corners[0])
            tangent = np.broadcast_to((corners[1] - corners[0])[:, None], (len(w), 2, 1))
        elif len(facet) == 3:
            uv, w = reference_rules_owner.degree_aware_reference_rule("triangle", degree)
            tangent = np.broadcast_to(
                np.stack((corners[1] - corners[0], corners[2] - corners[0]), axis=1),
                (len(w), 3, 2),
            )
            xi = corners[0] + np.asarray(uv) @ tangent[0].T
        else:
            uv, w = reference_rules_owner.degree_aware_reference_rule(
                "quadrilateral", degree
            )
            u, v = np.asarray(uv).T
            shape = np.stack(((1 - u) * (1 - v), u * (1 - v), u * v, (1 - u) * v), axis=1)
            xi = shape @ corners
            du = np.stack((v - 1, 1 - v, v, -v), axis=1) @ corners
            dv = np.stack((u - 1, -u, u, 1 - u), axis=1) @ corners
            tangent = np.stack((du, dv), axis=2)
        values, gradients = element.tabulate(xi)
        points = np.asarray(values) @ local
        jacobian = np.einsum("mD,qmd->qDd", local, np.asarray(gradients))
        physical = jacobian @ tangent
        reference_normal = (
            np.stack((tangent[:, 1, 0], -tangent[:, 0, 0]), axis=1)
            if mesh.topological_dimension == 2
            else np.cross(tangent[:, :, 0], tangent[:, :, 1])
        )
        if embedded:
            gram = np.swapaxes(jacobian, 1, 2) @ jacobian
            normal = np.einsum(
                "qDd,qde,qe->qD", jacobian, np.linalg.inv(gram), reference_normal
            )
            normal *= np.sqrt(np.linalg.det(gram))[:, None]
        else:
            normal = (
                np.stack((physical[:, 1, 0], -physical[:, 0, 0]), axis=1)
                if dimension == 2
                else np.cross(physical[:, :, 0], physical[:, :, 1])
            )
        if (
            np.dot(
                reference_normal[0],
                np.mean(corners, axis=0) - np.mean(reference.vertices, axis=0),
            )
            < 0
        ):
            normal = -normal
        w = np.asarray(w)
        weights = w * np.linalg.norm(normal, axis=1)
        vector = np.sum(w[:, None] * normal, axis=0)
        face_points.append(points)
        face_weights.append(weights)
        face_centers.append(np.sum(points * weights[:, None], axis=0) / np.sum(weights))
        area_vectors.append(vector)
        closure[cell] += vector
    neighbor = _owner_neighbor(connectivity, len(cells))[1]
    for face, cell in enumerate(neighbor):
        if cell >= 0:
            closure[cell] -= area_vectors[face]

    def padded(
        points: list[np.ndarray], weights: list[np.ndarray]
    ) -> tuple[Array, Array, Array]:
        width = max(len(w) for w in weights)
        p = np.zeros((len(points), width, ambient))
        w = np.zeros((len(points), width))
        valid = np.zeros((len(points), width), dtype=np.bool_)
        for i, (pi, wi) in enumerate(zip(points, weights, strict=True)):
            p[i, : len(wi)], w[i, : len(wi)], valid[i, : len(wi)] = pi, wi, True
        return jnp.asarray(p), jnp.asarray(w), jnp.asarray(valid)

    cp, cw, cv = padded(cell_points, cell_weights)
    fp, fw, _ = padded(face_points, face_weights)
    return _MappedFiniteVolumeGeometry(
        jnp.asarray(volumes),
        jnp.asarray(centers),
        jnp.asarray(face_centers),
        jnp.asarray(area_vectors),
        jnp.asarray([np.sum(w) for w in face_weights]),
        jnp.asarray(closure),
        fp,
        fw,
        cp,
        cw,
        cv,
        jnp.asarray(errors),
        exact,
    )


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
            raise ValueError(
                "MaskedFiniteVolumeGeometry.mesh must contain triangles or tetrahedra; got another simplex dimension."
            )
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
                "MaskedFiniteVolumeGeometry.mesh must be full-dimensional triangles or tetrahedra; got an embedded simplex."
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
    """certified content-rate balance of one masked finite-volume evaluation.

    ``ledger`` is the repository `ConservationStageLedger` over the capacity
    half-facet routes (route identity: the capacity signature). The compensated
    sums obey ``net_cell_sum = source_sum - boundary_outward_sum + frame_exchange_sum``.
    Reciprocal vector fluxes are carried into their bound physical cell frames;
    ``residual`` is the round-off defect of this bundle-valued balance.
    Inactive cells and routes contribute exactly zero to every sum.
    """

    ledger: ConservationStageLedger
    source_sum: Array
    boundary_outward_sum: Array
    net_cell_sum: Array
    frame_exchange_sum: Array
    residual: Array

    @checked
    def __init__(self, ledger: ConservationStageLedger, /) -> None:
        source_sum, boundary_sum, net_cell_sum = ledger.conservation_sums()
        self.ledger = ledger
        self.source_sum = source_sum
        self.boundary_outward_sum = boundary_sum
        self.net_cell_sum = net_cell_sum
        self.frame_exchange_sum = ledger.frame_exchange_sum()
        self.residual = net_cell_sum - (
            source_sum - boundary_sum + self.frame_exchange_sum
        )


def _routed_face_flux(
    geometry: MaskedFiniteVolumeGeometry, face_flux: ArrayLike, /
) -> Array:
    flux = jnp.asarray(face_flux)
    if not jnp.issubdtype(flux.dtype, jnp.floating):
        raise TypeError(
            "masked_fv_flux_divergence.face_flux must have a floating dtype; got a nonfloating array."
        )
    if flux.ndim == 0 or flux.shape[0] != geometry.face_capacity:
        raise ValueError(
            "masked_fv_flux_divergence.face_flux must begin with the half-facet route capacity; got an incompatible shape."
        )
    return flux


@eqx.filter_jit
@checked
def evaluate_masked_fv_geometry(mesh: MaskedSimplexMesh, /) -> MaskedFiniteVolumeGeometry:
    """Compiled finite-volume geometry of a `MaskedSimplexMesh`.

    One compilation serves every layout with the same capacity signature: the
    active counts, topology, and coordinates are dynamic data.
    """
    return MaskedFiniteVolumeGeometry(mesh)


@eqx.filter_jit
@checked
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
@checked
def evaluate_masked_fv_conservation(
    geometry: MaskedFiniteVolumeGeometry,
    face_flux: ArrayLike,
    source_rate: ArrayLike,
    /,
) -> MaskedFiniteVolumeConservation:
    """Compiled certified conservation ledger of one masked finite-volume evaluation.

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


@final
class _PreparedUnstructuredFiniteVolumeData(StrictModule):
    mesh: CellMesh
    vertices: Array
    triangles: Array
    quadrilaterals: Array
    tetrahedra: Array
    vertex_global_ids: Array
    cell_global_ids: Array
    cell_dimension: int = eqx.field(static=True)
    face_active: Array
    closure_evidence_id: str = eqx.field(static=True)
    patch_names: tuple[str, ...] = eqx.field(static=True)
    patch_faces: tuple[Array, ...]
    field_name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


@final
class _DirectFiniteVolumeMesh(StrictModule):
    points: np.ndarray
    triangles: np.ndarray
    quadrilaterals: np.ndarray
    tetrahedra: np.ndarray
    connectivity: Connectivity
    dimension: int = eqx.field(static=True)
    face_vertices: np.ndarray
    boundary_mask: np.ndarray


@final
class _FiniteVolumeBoundaryPatches(StrictModule):
    names: tuple[str, ...] = eqx.field(static=True)
    faces: tuple[np.ndarray, ...]


def _prepare_direct_fv_connectivity(
    vertices: ArrayLike | None,
    triangles: ArrayLike | None,
    quadrilaterals: ArrayLike | None,
    tetrahedra: ArrayLike | None,
    /,
) -> _DirectFiniteVolumeMesh:
    """Validate coordinates and normalize the direct simplex or polygon carrier."""
    if vertices is None:
        raise TypeError(
            "UnstructuredFiniteVolumePlan.vertices must be supplied for direct construction; got None."
        )
    points = np.asarray(vertices, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (2, 3):
        raise ValueError(
            "UnstructuredFiniteVolumePlan.vertices must have shape (n, 2) or (n, 3); got an incompatible shape."
        )
    if points.shape[0] < points.shape[1] + 1 or np.any(~np.isfinite(points)):
        raise ValueError(
            "UnstructuredFiniteVolumePlan.vertices must be finite and span a cell; got nonfinite or insufficient vertices."
        )
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
            raise ValueError(
                "UnstructuredFiniteVolumePlan.tetrahedra must use three-dimensional vertices; got two-dimensional coordinates."
            )
        if triangle_cells.ndim != 2 or triangle_cells.shape[1] != 3:
            raise ValueError(
                "UnstructuredFiniteVolumePlan.triangles must have shape (n, 3); got an incompatible shape."
            )
        if quadrilateral_cells.ndim != 2 or quadrilateral_cells.shape[1] != 4:
            raise ValueError(
                "UnstructuredFiniteVolumePlan.quadrilaterals must have shape (n, 4); got an incompatible shape."
            )
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
    else:
        if triangle_cells.shape[0] or quadrilateral_cells.shape[0]:
            raise ValueError(
                "UnstructuredFiniteVolumePlan.cell_kinds must be tetrahedra for direct 3-D construction; got triangle or quadrilateral cells."
            )
        if tetrahedral_cells.ndim != 2 or tetrahedral_cells.shape[1] != 4:
            raise ValueError(
                "UnstructuredFiniteVolumePlan.tetrahedra must have shape (n, 4); got an incompatible shape."
            )
        tetrahedral_cells = _normalized_tetrahedra(points, tetrahedral_cells)
        connectivity = tetrahedral_connectivity(tetrahedral_cells, points.shape[0])
        dimension = 3
        face_vertices = np.asarray(connectivity.faces, dtype=np.int32)
        boundary_mask = np.asarray(connectivity.boundary_faces, dtype=np.bool_)
    return _DirectFiniteVolumeMesh(
        points,
        triangle_cells,
        quadrilateral_cells,
        tetrahedral_cells,
        connectivity,
        dimension,
        face_vertices,
        boundary_mask,
    )


def _prepare_direct_fv_boundary_patches(
    face_vertices: np.ndarray,
    boundary_mask: np.ndarray,
    dimension: int,
    boundary_patches: Mapping[str, ArrayLike] | None,
    /,
) -> _FiniteVolumeBoundaryPatches:
    """Resolve physical patch rows and enforce a single partition of the boundary."""
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
                f"UnstructuredFiniteVolumePlan.boundary_patches must have face shape (n, {face_arity}); got shape {values.shape!r}."
            )
        indices = []
        for face in values:
            key = tuple(sorted(int(vertex) for vertex in face))
            if key not in lookup:
                raise ValueError(
                    f"UnstructuredFiniteVolumePlan.boundary_patches must name mesh faces; got missing face {key!r}."
                )
            face_index = lookup[key]
            if not boundary_mask[face_index]:
                raise ValueError(
                    "UnstructuredFiniteVolumePlan.boundary_patches must contain boundary faces; got an interior face."
                )
            assigned[face_index] += 1
            indices.append(face_index)
        normalized_patch_faces.append(np.asarray(indices, dtype=np.int32))
    if np.any(assigned[boundary_mask] != 1):
        raise ValueError(
            "UnstructuredFiniteVolumePlan.boundary_patches must partition every boundary face once; got a missing or repeated assignment."
        )
    return _FiniteVolumeBoundaryPatches(names, tuple(normalized_patch_faces))


def _prepare_direct_fv_plan(
    vertices: ArrayLike | None,
    triangles: ArrayLike | None,
    quadrilaterals: ArrayLike | None,
    tetrahedra: ArrayLike | None,
    vertex_global_ids: ArrayLike | None,
    cell_global_ids: ArrayLike | None,
    boundary_patches: Mapping[str, ArrayLike] | None,
    field_name: str,
    component_names: Sequence[str],
    /,
) -> _PreparedUnstructuredFiniteVolumeData:
    """Prepare mesh identity and FV field metadata from a normalized carrier."""
    carrier = _prepare_direct_fv_connectivity(
        vertices, triangles, quadrilaterals, tetrahedra
    )
    points, connectivity, dimension = (
        carrier.points,
        carrier.connectivity,
        carrier.dimension,
    )
    triangle_cells, quadrilateral_cells, tetrahedral_cells = (
        carrier.triangles,
        carrier.quadrilaterals,
        carrier.tetrahedra,
    )
    face_vertices = carrier.face_vertices
    cell_count = connectivity.cell_count
    patches = _prepare_direct_fv_boundary_patches(
        face_vertices, carrier.boundary_mask, dimension, boundary_patches
    )
    names, normalized_patch_faces = patches.names, patches.faces
    field = str(field_name)
    if not field:
        raise ValueError(
            "UnstructuredFiniteVolumePlan.field_name must be non-empty; got an empty name."
        )
    components = normalize_component_names(component_names)
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
    return _PreparedUnstructuredFiniteVolumeData(
        mesh=mesh,
        vertices=jnp.asarray(points),
        triangles=jnp.asarray(triangle_cells),
        quadrilaterals=jnp.asarray(quadrilateral_cells),
        tetrahedra=jnp.asarray(tetrahedral_cells),
        vertex_global_ids=jnp.asarray(vertex_ids, dtype=global_id_dtype),
        cell_global_ids=jnp.asarray(cell_ids, dtype=global_id_dtype),
        cell_dimension=dimension,
        face_active=jnp.ones((face_vertices.shape[0],), dtype=jnp.bool_),
        closure_evidence_id="",
        patch_names=names,
        patch_faces=tuple(jnp.asarray(value) for value in normalized_patch_faces),
        field_name=field,
        component_names=components,
        topology_id=topology_id,
        geometry_id=geometry_id,
        key=DiscretizationKey("unstructured_finite_volume", DiscretizationRole.PHYSICAL),
        capabilities=capabilities,
        plan_id=canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-plan",
                "topology": topology_id,
                "geometry": geometry_id,
            }
        ),
    )


@final
class _FiniteVolumeClosure(StrictModule):
    face_active: np.ndarray
    physical_boundary: np.ndarray


def _prepare_fv_closure(
    mesh: CellMesh,
    connectivity: Connectivity,
    face_owner: np.ndarray,
    face_neighbor: np.ndarray,
    face_active: np.ndarray,
    neighborhood_complete: ArrayLike | None,
    physical_boundary_faces: ArrayLike | None,
    evidence: str,
    /,
) -> _FiniteVolumeClosure:
    """Bind owned-facet execution to the storage neighborhood and physical boundary."""
    storage = mesh.storage
    dimension = mesh.topological_dimension
    face_count = face_owner.size
    if storage is not None and storage.partition_count > 1:
        if not evidence or evidence != storage.evidence_id:
            raise ValueError(
                "UnstructuredFiniteVolumePlan.closure_evidence_id must match the storage evidence; got missing or different evidence."
            )
        complete = np.asarray(neighborhood_complete)
        physical = np.asarray(physical_boundary_faces)
        if complete.dtype != np.bool_ or complete.shape != (connectivity.cell_count,):
            raise ValueError(
                "UnstructuredFiniteVolumePlan.neighborhood_complete must be a boolean per cell; got an incompatible dtype or shape."
            )
        if physical.dtype != np.bool_ or physical.shape != (face_count,):
            raise ValueError(
                "UnstructuredFiniteVolumePlan.physical_boundary_faces must be a boolean per face; got an incompatible dtype or shape."
            )
        cell_owned = np.asarray(storage.entity_owned[dimension], dtype=np.bool_)
        face_active &= np.asarray(storage.entity_owned[dimension - 1], dtype=np.bool_)
        required = cell_owned.copy()
        required[face_owner[face_active]] = True
        required[face_neighbor[face_active & (face_neighbor >= 0)]] = True
        if np.any(required & ~complete):
            raise ValueError(
                "UnstructuredFiniteVolumePlan.neighborhood_complete must cover every owned-facet endpoint; got an incomplete required neighborhood."
            )
        incident_owned = cell_owned[face_owner] | (
            (face_neighbor >= 0) & cell_owned[np.maximum(face_neighbor, 0)]
        )
        if np.any(physical & (face_neighbor >= 0)) or np.any(
            (face_neighbor < 0) & (incident_owned | face_active) & ~physical
        ):
            raise ValueError(
                "UnstructuredFiniteVolumePlan.physical_boundary_faces must distinguish physical faces from the closure frontier; got an interior or unclassified required face."
            )
    else:
        physical = face_neighbor < 0
    return _FiniteVolumeClosure(face_active, physical)


def _prepare_fv_boundary_groups(
    physical: np.ndarray,
    face_count: int,
    boundary_face_groups: Mapping[str, ArrayLike] | None,
    /,
) -> _FiniteVolumeBoundaryPatches:
    """Validate index-based groups against the physical boundary partition."""
    boundary_faces = np.flatnonzero(physical).astype(np.int32)
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
                "UnstructuredFiniteVolumePlan.boundary_face_groups must have non-empty names; got an empty name or group set."
            )
        patch_faces = tuple(groups[name] for name in patch_names)
        assigned = np.zeros((face_count,), dtype=np.int32)
        for indices in patch_faces:
            if (
                indices.ndim != 1
                or np.any(indices < 0)
                or np.any(indices >= face_count)
                or np.unique(indices).size != indices.size
                or np.any(~physical[indices])
            ):
                raise ValueError(
                    "UnstructuredFiniteVolumePlan.boundary_face_groups must contain unique in-range physical boundary faces; got invalid face rows."
                )
            assigned[indices] += 1
        if np.any(assigned[boundary_faces] != 1) or np.any(assigned[~physical] != 0):
            raise ValueError(
                "UnstructuredFiniteVolumePlan.boundary_face_groups must partition the physical boundary once; got a missing, repeated, or interior assignment."
            )
    return _FiniteVolumeBoundaryPatches(patch_names, patch_faces)


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
    face_active: Array
    closure_evidence_id: str = eqx.field(static=True)
    patch_names: tuple[str, ...] = eqx.field(static=True)
    patch_faces: tuple[Array, ...]
    field_name: str = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
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
        if _prepared is None:
            _prepared = _prepare_direct_fv_plan(
                vertices,
                triangles,
                quadrilaterals,
                tetrahedra,
                vertex_global_ids,
                cell_global_ids,
                boundary_patches,
                field_name,
                component_names,
            )
        self.mesh = _prepared.mesh
        self.vertices = _prepared.vertices
        self.triangles = _prepared.triangles
        self.quadrilaterals = _prepared.quadrilaterals
        self.tetrahedra = _prepared.tetrahedra
        self.vertex_global_ids = _prepared.vertex_global_ids
        self.cell_global_ids = _prepared.cell_global_ids
        self.cell_dimension = _prepared.cell_dimension
        self.face_active = _prepared.face_active
        self.closure_evidence_id = _prepared.closure_evidence_id
        self.patch_names = _prepared.patch_names
        self.patch_faces = _prepared.patch_faces
        self.field_name = _prepared.field_name
        self.component_names = _prepared.component_names
        self.topology_id = _prepared.topology_id
        self.geometry_id = _prepared.geometry_id
        self.key = _prepared.key
        self.capabilities = _prepared.capabilities
        self.plan_id = _prepared.plan_id

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
        neighborhood_complete: ArrayLike | None = None,
        physical_boundary_faces: ArrayLike | None = None,
        closure_evidence_id: str | None = None,
        periodic_geometry: CellGeometrySpec | None = None,
        periodic_embedding: GlobalEmbeddingCertificate | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> "UnstructuredFiniteVolumePlan":
        """Prepare a local closure without reconstructing a mesh.

        Partitioned storage requires an evidence-bound full neighborhood
        for every owned cell and owned facet endpoint. Physical boundaries are
        explicit: the boundary of a ghost closure is not a physical boundary.

        ``budget`` shares periodic coordinate-source preparation with the
        enclosing request; it is not retained on the plan.
        """
        if periodic_geometry is not None and mesh.periodic_topology is None:
            raise ValueError(
                "UnstructuredFiniteVolumePlan.periodic_geometry must accompany periodic mesh topology; got a nonperiodic mesh."
            )
        match mesh.connectivity:
            case (
                PolygonalConnectivity()
                | TetrahedralConnectivity()
                | HexahedralConnectivity()
                | PolyhedralConnectivity()
            ) as connectivity:
                pass
            case _:
                raise ValueError(
                    "UnstructuredFiniteVolumePlan.connectivity must be polygonal or volumetric; got unsupported connectivity."
                )
        if any(block.cell_kind == "polygon" for block in mesh.blocks):
            raise ValueError(
                "UnstructuredFiniteVolumePlan.cell_kinds must support cell quadrature; got a general polygon."
            )
        field = str(field_name)
        if not field:
            raise ValueError(
                "UnstructuredFiniteVolumePlan.field_name must be non-empty; got an empty name."
            )
        components = normalize_component_names(component_names)
        dimension = mesh.topological_dimension
        cell_ids = np.asarray(mesh.topology.entity_sets[dimension].entity_ids)
        vertex_ids = np.asarray(mesh.vertex_global_ids)
        face_owner, face_neighbor, _ = _owner_neighbor(
            connectivity, connectivity.cell_count
        )
        storage = mesh.storage
        partitioned = storage is not None and storage.partition_count > 1
        face_count = face_owner.size
        face_active = np.ones((face_count,), dtype=np.bool_)
        if (
            isinstance(connectivity, PolyhedralConnectivity)
            and mesh.periodic_topology is not None
        ):
            authority = (
                mesh.periodic_topology.actual_geometry
                if periodic_geometry is None
                else periodic_geometry
            )
            if authority is None:
                raise ValueError(
                    "finite_volume_geometry.cell_geometry must bind the periodic source map; got no source geometry."
                )
            work, memory = cell_geometry_preparation_bounds(authority)
            with (
                nullcontext(budget)
                if budget is not None
                else owned_coordinate_preparation(
                    work,
                    memory,
                    maximum_geometry_queries=cell_geometry_preparation_query_upper(
                        authority
                    ),
                )
            ) as owner_budget:
                face_neighbor, quotient_active, _ = (
                    imported_polyhedral.periodic_polyhedral_faces(
                        mesh,
                        actual_geometry=periodic_geometry,
                        embedding=periodic_embedding,
                        budget=owner_budget,
                    )
                )
            face_active &= quotient_active
        evidence = "" if closure_evidence_id is None else str(closure_evidence_id)
        closure = _prepare_fv_closure(
            mesh,
            connectivity,
            face_owner,
            face_neighbor,
            face_active,
            neighborhood_complete,
            physical_boundary_faces,
            evidence,
        )
        face_active = closure.face_active
        patches = _prepare_fv_boundary_groups(
            closure.physical_boundary, face_count, boundary_face_groups
        )
        patch_names, patch_faces = patches.names, patches.faces
        topology_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-topology",
                "mesh": mesh.topology_id
                if storage is None
                else storage.logical_topology_id,
                "field": field,
                "components": list(components),
                "boundary_patches": list(patch_names)
                if partitioned
                else {
                    name: array_tree_fingerprint(indices)
                    for name, indices in zip(patch_names, patch_faces, strict=True)
                },
            }
        )
        geometry_id = canonical_fingerprint(
            {
                "kind": "unstructured-finite-volume-geometry",
                "topology": topology_id,
                "mesh_geometry": mesh.geometry_id
                if storage is None
                else storage.logical_geometry_id,
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

        def cells_of_kind(kind: str, arity: int) -> Array:
            blocks = tuple(
                block.vertices for block in mesh.blocks if block.cell_kind == kind
            )
            return (
                jnp.concatenate(blocks, axis=0)
                if blocks
                else jnp.empty((0, arity), dtype=jnp.int32)
            )

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
            triangles=cells_of_kind("triangle", 3),
            quadrilaterals=cells_of_kind("quadrilateral", 4),
            tetrahedra=cells_of_kind("tetrahedron", 4),
            vertex_global_ids=jnp.asarray(vertex_ids),
            cell_global_ids=jnp.asarray(cell_ids),
            cell_dimension=dimension,
            face_active=jnp.asarray(face_active, dtype=jnp.bool_),
            closure_evidence_id=evidence,
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

    @checked
    def prepare(
        self,
        /,
        *,
        numeric_version: str = "0",
        owned_face_capacity: int | None = None,
        cell_geometry: CellGeometrySpec | None = None,
        periodic_embedding: GlobalEmbeddingCertificate | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> UnstructuredFiniteVolumeDiscretization:
        """Prepare local geometry with optional collectively agreed facet capacity.

        A supplied capacity bounds owned-face execution across unequal ranks.
        Padding slots are masked and do not perform numerical flux solves.

        ``cell_geometry`` selects the represented represented coordinate map rather
        than corner geometry. Its certified cell measures and absolute errors
        define cell-average pairings, integration metadata, and remap inventory.
        Cell/face quadrature evaluates that same map without rescaling weights.
        ``cell_quadrature_degree == 0`` makes no polynomial-exactness claim for
        general restricted rational coordinate maps.
        Standalone native hexahedra default to their represented Q1 coordinate map,
        restored from prepared storage when present, without a cell downgrade.
        Bound nonnull coordinate sources are always restored for every supported
        topology, so omitting the keyword never selects their sampled corners.

        ``budget`` shares source resolution, certified measures and periodic
        quotient preparation with the enclosing request. Standalone mapped
        preparation retains the certified-measure owner's work/storage limits.
        """
        return UnstructuredFiniteVolumeDiscretization(
            self,
            numeric_version=numeric_version,
            owned_face_capacity=owned_face_capacity,
            cell_geometry=cell_geometry,
            periodic_embedding=periodic_embedding,
            budget=budget,
        )


@final
class _FiniteVolumeCoordinateGeometry(StrictModule):
    connectivity: Connectivity
    geometry_id: str = eqx.field(static=True)
    owner: np.ndarray
    neighbor: np.ndarray
    owner_sign: np.ndarray
    geometry: _MappedFiniteVolumeGeometry


@final
class _FiniteVolumeGeometryPhase(StrictModule):
    coordinates: _FiniteVolumeCoordinateGeometry
    cell_geometry: CellGeometrySpec | None
    neighbor: np.ndarray
    periodic_frame_maps: np.ndarray | None
    quotient_active: np.ndarray | None


@final
class _FiniteVolumeMetadata(StrictModule):
    support: DiscreteSupport
    cell_space: DiscreteFieldSpace
    face_space: DiscreteFieldSpace
    face_entities: EntitySet
    face_block: FiniteVolumeFaceBlock
    preparation: PreparationReport
    spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)


def _corner_fv_geometry(
    plan: UnstructuredFiniteVolumePlan,
    connectivity: Connectivity,
    owner: np.ndarray,
    owner_sign: np.ndarray,
    /,
) -> _MappedFiniteVolumeGeometry:
    """Evaluate corner geometry and require quadrature to reproduce cell measures."""
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
    return _MappedFiniteVolumeGeometry(
        cell_volumes,
        cell_centers,
        face_centers,
        area_vectors,
        face_measures,
        closure,
        quadrature_points,
        quadrature_weights,
        cell_quadrature_points,
        cell_quadrature_weights,
        cell_quadrature_valid,
        jnp.zeros_like(cell_volumes),
        False,
    )


def _evaluate_fv_coordinate_geometry(
    plan: UnstructuredFiniteVolumePlan,
    connectivity: Connectivity,
    cell_geometry: CellGeometrySpec | None,
    budget: CoordinateEnclosureBudget,
    /,
) -> _FiniteVolumeCoordinateGeometry:
    """Dispatch mapped, polyhedral, and corner geometry without changing charge order."""
    mesh, cell_count, geometry_id = plan.mesh, connectivity.cell_count, plan.geometry_id
    if cell_geometry is not None and not (
        all(block.cell_kind == "polyhedron" for block in mesh.blocks)
        and has_power_source(cell_geometry)
    ):
        if mesh.topological_dimension != mesh.coordinates.shape[1] and not (
            mesh.topological_dimension == 2 and mesh.coordinates.shape[1] == 3
        ):
            raise ValueError(
                "UnstructuredFiniteVolumeDiscretization.cell_geometry must be full-dimensional or an embedded surface; got unsupported topological and ambient dimensions."
            )
        owner, neighbor, owner_sign = _owner_neighbor(connectivity, cell_count)
        geometry = _mapped_fv_geometry(mesh, cell_geometry, owner, budget=budget)
        geometry_id = canonical_fingerprint(
            {
                "kind": "mapped-unstructured-finite-volume-geometry",
                "topology": plan.topology_id,
                "coordinate_geometry": cell_geometry_id(cell_geometry),
            }
        )
        return _FiniteVolumeCoordinateGeometry(
            connectivity, geometry_id, owner, neighbor, owner_sign, geometry
        )
    match connectivity:
        case PolyhedralConnectivity():
            polyhedral = imported_polyhedral.prepare_polyhedral_finite_volume_geometry(
                mesh,
                cell_geometry=cell_geometry,
                budget=budget,
            )
            if cell_geometry is not None:
                _, volume_errors, volume_exact = (
                    mapped_measure_owner.certified_cell_measures(
                        mesh,
                        cell_geometry,
                        budget=budget,
                    )
                )
                volume_errors = jnp.asarray(volume_errors)
                geometry_id = polyhedral.geometry_id
            owner, neighbor, owner_sign = _owner_neighbor(connectivity, cell_count)
            cell_volumes = polyhedral.cell_volumes
            cell_centers = polyhedral.cell_centers
            face_centers = polyhedral.face_centers
            area_vectors = jax.device_put(
                np.asarray(owner_sign, dtype=polyhedral.face_area_vectors.dtype)[:, None]
                * np.asarray(polyhedral.face_area_vectors)
            )
            face_measures = polyhedral.face_measures
            host_closure = np.zeros(cell_centers.shape, dtype=cell_centers.dtype)
            host_closure[:, 0] = np.asarray(polyhedral.closure_residual)
            closure = jax.device_put(host_closure)
            quadrature_points = polyhedral.face_quadrature_points
            quadrature_weights = polyhedral.face_quadrature_weights
            cell_quadrature_points = polyhedral.cell_quadrature_points
            cell_quadrature_weights = polyhedral.cell_quadrature_weights
            cell_quadrature_valid = polyhedral.cell_quadrature_valid
            if cell_geometry is None:
                volume_errors = jax.device_put(
                    np.zeros(cell_volumes.shape, dtype=cell_volumes.dtype)
                )
                volume_exact = False
            geometry = _MappedFiniteVolumeGeometry(
                cell_volumes,
                cell_centers,
                face_centers,
                area_vectors,
                face_measures,
                closure,
                quadrature_points,
                quadrature_weights,
                cell_quadrature_points,
                cell_quadrature_weights,
                cell_quadrature_valid,
                volume_errors,
                volume_exact,
            )
        case (
            PolygonalConnectivity() | TetrahedralConnectivity() | HexahedralConnectivity()
        ):
            owner, neighbor, owner_sign = _owner_neighbor(connectivity, cell_count)
            geometry = _corner_fv_geometry(plan, connectivity, owner, owner_sign)
        case _:
            assert_never(connectivity)
    return _FiniteVolumeCoordinateGeometry(
        connectivity, geometry_id, owner, neighbor, owner_sign, geometry
    )


def _prepare_fv_geometry(
    plan: UnstructuredFiniteVolumePlan,
    cell_geometry: CellGeometrySpec | None,
    periodic_embedding: GlobalEmbeddingCertificate | None,
    budget: CoordinateEnclosureBudget | None,
    /,
) -> _FiniteVolumeGeometryPhase:
    """Resolve coordinate-source evidence and prepare periodic neighbor frames in one budget."""
    mesh = plan.mesh
    periodic_frame_maps = None
    quotient_active = None
    if cell_geometry is None and mesh.periodic_topology is not None:
        cell_geometry = mesh.periodic_topology.actual_geometry
    match mesh.connectivity:
        case (
            PolygonalConnectivity()
            | TetrahedralConnectivity()
            | HexahedralConnectivity()
            | PolyhedralConnectivity()
        ) as connectivity:
            pass
        case _:
            raise TypeError(
                "UnstructuredFiniteVolumeDiscretization.connectivity must be polygonal or volumetric; got unsupported connectivity."
            )
    if cell_geometry is None and (
        isinstance(connectivity, HexahedralConnectivity)
        or (mesh.storage is not None and mesh.storage.local_geometry is not None)
    ):
        cell_geometry = (
            CellGeometrySpec.affine(mesh, budget=budget)
            if mesh.storage is None
            else mesh.storage.restore_geometry(budget=budget)
        )
    with _fv_preparation_budget(mesh, cell_geometry, budget) as owner_budget:
        coordinates = _evaluate_fv_coordinate_geometry(
            plan, connectivity, cell_geometry, owner_budget
        )
        neighbor = coordinates.neighbor
        if (
            isinstance(connectivity, PolyhedralConnectivity)
            and mesh.periodic_topology is not None
        ):
            neighbor, quotient_active, periodic_frame_maps = (
                imported_polyhedral.periodic_polyhedral_faces(
                    mesh,
                    actual_geometry=cell_geometry,
                    embedding=periodic_embedding,
                    budget=owner_budget,
                )
            )
    return _FiniteVolumeGeometryPhase(
        coordinates, cell_geometry, neighbor, periodic_frame_maps, quotient_active
    )


def _prepare_fv_metadata(
    plan: UnstructuredFiniteVolumePlan,
    topology: CellComplexTopology,
    points: np.ndarray,
    cell_geometry: CellGeometrySpec | None,
    cell_volumes: Array,
    face_centers: Array,
    area_vectors: Array,
    face_measures: Array,
    owner: np.ndarray,
    neighbor: np.ndarray,
    boundary_patch_ids: np.ndarray,
    face_active: Array,
    /,
) -> _FiniteVolumeMetadata:
    """Assemble support, paired field spaces, face routes, and preparation evidence."""
    mesh = plan.mesh
    cell_count, face_count = cell_volumes.size, owner.size
    embedding_id = canonical_fingerprint(
        {
            "kind": "unstructured-fv-embedding",
            "vertices": array_tree_fingerprint(points),
            "coordinate_geometry": None
            if cell_geometry is None
            else cell_geometry_id(cell_geometry),
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
                jax.device_put(
                    np.broadcast_to(np.asarray(cell_volumes)[:, None], cell_shape)
                )
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
                jax.device_put(
                    np.broadcast_to(np.asarray(face_measures)[:, None], face_shape)
                )
            ),
        ),
        representation="flux_moment",
        conformity="Hdiv",
        trace_space_id=cell_space.field_space_id,
    )
    face_block = FiniteVolumeFaceBlock(
        face_ids=jax.device_put(np.arange(face_count, dtype=np.int32)),
        owner_cells=jax.device_put(owner),
        neighbor_cells=jax.device_put(neighbor),
        boundary_patch_ids=jax.device_put(boundary_patch_ids),
        face_centers=face_centers,
        area_vectors=area_vectors,
        face_measures=face_measures,
        active_mask=face_active,
        block_id=canonical_fingerprint(
            {
                "kind": "unstructured-face-block",
                "plan": plan.plan_id,
                "storage": None if mesh.storage is None else mesh.storage.storage_id,
            }
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
            "active_faces": int(np.sum(np.asarray(face_active))),
            "reciprocal_faces": int(np.sum(np.asarray(face_active) & (neighbor >= 0))),
            "source_faces": mesh.topology.entity_sets[plan.cell_dimension - 1].count,
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
    return _FiniteVolumeMetadata(
        support,
        cell_space,
        face_space,
        face_entities,
        face_block,
        preparation,
        spaces,
        measures,
        capabilities,
    )


class UnstructuredFiniteVolumeDiscretization(
    AbstractUnstructuredFiniteVolumeDiscretization
):
    mesh: CellMesh
    vertices: Array
    triangles: Array
    quadrilaterals: Array
    tetrahedra: Array
    vertex_global_ids: Array
    cell_global_ids: Array
    face_global_ids: Array
    cell_dimension: int = eqx.field(static=True)
    cell_owned: Array
    cell_owner: Array
    owned_face_indices: Array
    owned_face_valid: Array
    owned_face_buffered: bool = eqx.field(static=True)
    global_entity_counts: tuple[int, ...] = eqx.field(static=True)
    partition_index: Array
    partition_count: int = eqx.field(static=True)
    closure_evidence_id: str = eqx.field(static=True)
    topology: CellComplexTopology
    connectivity: Connectivity
    face_block: FiniteVolumeFaceBlock
    face_blocks: tuple[FiniteVolumeFaceBlock, ...]
    cell_volumes: Array
    cell_geometry: CellGeometrySpec | None
    cell_volume_error_bounds: Array
    cell_volume_exact: bool = eqx.field(static=True)
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
    neighbor_frame_maps: Array = fixed_field()
    source_face_indices: Array = fixed_field()
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

    # Geometry/metadata preparation is separate; ordered quotient projection and Strict publication preserve constructor failure order.
    @checked
    def __init__(
        self,
        plan: UnstructuredFiniteVolumePlan,
        /,
        *,
        numeric_version: str = "0",
        owned_face_capacity: int | None = None,
        cell_geometry: CellGeometrySpec | None = None,
        periodic_embedding: GlobalEmbeddingCertificate | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        mesh = plan.mesh
        points = np.asarray(mesh.coordinates)
        geometry_phase = _prepare_fv_geometry(
            plan, cell_geometry, periodic_embedding, budget
        )
        coordinates = geometry_phase.coordinates
        cell_geometry, connectivity = (
            geometry_phase.cell_geometry,
            coordinates.connectivity,
        )
        topology = mesh.topology
        cell_count, face_count = connectivity.cell_count, coordinates.owner.size
        geometry_id = coordinates.geometry_id
        owner, neighbor, owner_sign = (
            coordinates.owner,
            geometry_phase.neighbor,
            coordinates.owner_sign,
        )
        periodic_frame_maps, quotient_active = (
            geometry_phase.periodic_frame_maps,
            geometry_phase.quotient_active,
        )
        (
            cell_volumes,
            cell_centers,
            face_centers,
            area_vectors,
            face_measures,
            closure,
            quadrature_points,
            quadrature_weights,
            cell_quadrature_points,
            cell_quadrature_weights,
            cell_quadrature_valid,
            volume_errors,
            volume_exact,
        ) = coordinates.geometry
        boundary_patch_ids = np.full((face_count,), -1, dtype=np.int32)
        for patch_id, face_indices in enumerate(plan.patch_faces):
            boundary_patch_ids[np.asarray(face_indices, dtype=np.int32)] = patch_id
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
            periodic_frame_maps,
        )
        source_face_indices = np.arange(face_count, dtype=np.int32)
        face_active = plan.face_active
        frame_maps = periodic_frame_maps
        if frame_maps is None:
            frame_maps = np.broadcast_to(
                np.eye(mesh.ambient_dimension + 1),
                (face_count, mesh.ambient_dimension + 1, mesh.ambient_dimension + 1),
            ).copy()
        if (
            isinstance(connectivity, PolyhedralConnectivity)
            and mesh.periodic_topology is not None
        ):
            periodic = mesh.periodic_topology
            degree = plan.cell_dimension - 1
            start, stop = np.asarray(periodic.lifted_offsets)[degree : degree + 2]
            orbits = np.asarray(periodic.orbit_indices)[start:stop]
            if quotient_active is None:
                raise RuntimeError(
                    "UnstructuredFiniteVolumeDiscretization.quotient_active must identify the periodic face representatives; got None."
                )
            source_face_indices = np.flatnonzero(quotient_active).astype(np.int32)
            source_face_indices = source_face_indices[
                np.argsort(orbits[source_face_indices])
            ]
            topology = periodic.quotient
            if source_face_indices.size != topology.entity_sets[degree].count:
                raise ValueError(
                    "UnstructuredFiniteVolumeDiscretization.source_face_indices must contain one SCI representative per quotient facet; got an incompatible representative count."
                )
            owner = owner[source_face_indices]
            neighbor = neighbor[source_face_indices]
            owner_sign = owner_sign[source_face_indices]
            face_centers = face_centers[source_face_indices]
            area_vectors = area_vectors[source_face_indices]
            face_measures = face_measures[source_face_indices]
            quadrature_points = quadrature_points[source_face_indices]
            quadrature_weights = quadrature_weights[source_face_indices]
            boundary_patch_ids = boundary_patch_ids[source_face_indices]
            face_active = face_active[source_face_indices]
            frame_maps = frame_maps[source_face_indices]
            face_count = source_face_indices.size
        metadata = _prepare_fv_metadata(
            plan,
            topology,
            points,
            cell_geometry,
            cell_volumes,
            face_centers,
            area_vectors,
            face_measures,
            owner,
            neighbor,
            boundary_patch_ids,
            face_active,
        )
        support, cell_space, face_space = (
            metadata.support,
            metadata.cell_space,
            metadata.face_space,
        )
        face_entities, face_block, preparation = (
            metadata.face_entities,
            metadata.face_block,
            metadata.preparation,
        )
        spaces, measures, capabilities = (
            metadata.spaces,
            metadata.measures,
            metadata.capabilities,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError(
                "UnstructuredFiniteVolumeDiscretization.numeric_version must be non-empty; got an empty string."
            )
        self.mesh = mesh
        self.cell_geometry = cell_geometry
        self.cell_volume_error_bounds = volume_errors
        self.cell_volume_exact = volume_exact
        self.vertices = plan.vertices
        self.triangles = plan.triangles
        self.quadrilaterals = plan.quadrilaterals
        self.tetrahedra = plan.tetrahedra
        self.vertex_global_ids = plan.vertex_global_ids
        self.cell_global_ids = plan.cell_global_ids
        self.face_global_ids = face_entities.entity_ids
        self.cell_dimension = plan.cell_dimension
        storage = mesh.storage
        self.cell_owned = (
            jnp.ones((cell_count,), dtype=jnp.bool_)
            if storage is None
            else storage.entity_owned[plan.cell_dimension]
        )
        self.cell_owner = (
            jnp.zeros((cell_count,), dtype=jnp.int32)
            if storage is None
            else storage.entity_owner[plan.cell_dimension]
        )
        owned_faces = np.flatnonzero(np.asarray(face_active)).astype(np.int32)
        capacity = (
            owned_faces.size if owned_face_capacity is None else owned_face_capacity
        )
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, (int, np.integer))
            or capacity < owned_faces.size
        ):
            raise ValueError(
                "UnstructuredFiniteVolumeDiscretization.owned_face_capacity must be an integer covering every owned facet; got a noninteger or insufficient capacity."
            )
        indices = np.zeros((capacity,), dtype=np.int32)
        indices[: owned_faces.size] = owned_faces
        self.owned_face_indices = jnp.asarray(indices, dtype=jnp.int32)
        self.owned_face_valid = jnp.arange(capacity, dtype=jnp.int32) < owned_faces.size
        self.owned_face_buffered = owned_face_capacity is not None
        self.global_entity_counts = (
            tuple(entity.count for entity in topology.entity_sets)
            if storage is None
            else storage.global_entity_counts
        )
        self.partition_index = jnp.asarray(
            0 if storage is None else storage.partition_index, dtype=jnp.int32
        )
        self.partition_count = 1 if storage is None else storage.partition_count
        self.closure_evidence_id = plan.closure_evidence_id
        self.topology = topology
        self.connectivity = connectivity
        self.face_block = face_block
        self.face_blocks = (face_block,)
        self.cell_volumes = cell_volumes
        self.cell_centers = cell_centers
        self.source_face_indices = jnp.asarray(source_face_indices)
        self.neighbor_frame_maps = jnp.asarray(frame_maps)
        self.cell_quadrature_points = cell_quadrature_points
        self.cell_quadrature_weights = cell_quadrature_weights
        self.cell_quadrature_valid = cell_quadrature_valid
        self.cell_quadrature_degree = (
            0
            if cell_geometry is not None
            else (1 if isinstance(connectivity, PolyhedralConnectivity) else 5)
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
        self.geometry_id = geometry_id
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
                "geometry": geometry_id,
                "numeric_version": version,
                "storage": None if storage is None else storage.storage_id,
                "closure_evidence": plan.closure_evidence_id,
                "owned_face_capacity": owned_face_capacity,
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

    def directional_control_volume_widths(self) -> Array:
        """Return volume-normalized directional widths from control-volume faces."""

        dtype = self.cell_volumes.dtype
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

    @checked
    def integration_domain(
        self, kind: str, selection: EntitySelection | None = None, /
    ) -> IntegrationDomain:
        """Cell or facet domain in the face table's order and orientation.

        The owner of each face is the cell its stored area vector points out
        of; local facets index each cell's edge or face table.
        """

        return imported_side_trace.finite_volume_integration_domain(self, kind, selection)

    @checked
    def prepare_side_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
        reconstruction: imported_side_trace.FiniteVolumeFaceReconstruction | None = None,
    ) -> PreparedTraceAction:
        """Prepare the cell-average or linear face state on selected faces.

        `reconstruction=None` publishes the side cell average at every site
        (`representation="cell-average"`); a
        `PreparedCellPolynomialReconstruction` publishes a degree-k polynomial face state
        (`"face-state"`, `trace_degree=k`) through a per-facet stencil route
        with its algebraic transpose. Sites follow the owner's facet
        parametrization (edges or triangles) and are shared by both sides of an
        interior face; normals point out of the side cell. Polyhedral faces and
        nonlinear reconstructions are refused.
        """

        return imported_side_trace.prepare_finite_volume_side_trace(
            self,
            field_name,
            domain,
            rule=rule,
            quantity=quantity,
            side=side,
            reconstruction=reconstruction,
        )

    @checked
    def prepare_nonlinear_face_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        reconstruction: imported_side_trace.FiniteVolumeFaceReconstruction,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
    ) -> imported_side_trace.PreparedNonlinearFaceTrace:
        """Prepare WENO-Z face states with a linearization at a supplied state."""

        return imported_side_trace.prepare_finite_volume_nonlinear_face_trace(
            self,
            field_name,
            domain,
            rule=rule,
            reconstruction=reconstruction,
            quantity=quantity,
            side=side,
        )


@checked
def neighbor_centers_in_owner_frame(
    discretization: UnstructuredFiniteVolumeDiscretization,
    centers: Array,
) -> Array:
    """Pull physical neighbor moments back through the prepared bound action."""
    maps = discretization.neighbor_frame_maps
    dimension = centers.shape[-1]
    neighbor = centers[jnp.maximum(discretization.neighbor_cells, 0)]
    return contract(
        "fji,fj->fi",
        maps[:, :dimension, :dimension],
        neighbor - maps[:, :dimension, dimension],
    )


def _quality_report(
    plan: UnstructuredFiniteVolumePlan,
    connectivity: Connectivity,
    cell_volumes: Array,
    cell_centers: Array,
    area_vectors: Array,
    face_measures: Array,
    closure: Array,
    owner: np.ndarray,
    neighbor: np.ndarray,
    maps: np.ndarray | None = None,
) -> UnstructuredFiniteVolumeQualityReport:
    """Prepare host-side quality diagnostics from the represented FV geometry."""
    (
        points,
        tetrahedra,
        face_active,
        volumes,
        centers,
        areas,
        measures,
        closure_vectors,
        host_connectivity,
    ) = jax.device_get(
        (
            plan.vertices,
            plan.tetrahedra,
            plan.face_active,
            cell_volumes,
            cell_centers,
            area_vectors,
            face_measures,
            closure,
            connectivity,
        )
    )
    owner_ = np.asarray(owner, dtype=np.int32)
    neighbor_ = np.asarray(neighbor, dtype=np.int32)
    interior = (neighbor_ >= 0) & face_active
    neighbor_centers = centers[np.maximum(neighbor_, 0)]
    if maps is not None:
        dimension = centers.shape[-1]
        neighbor_centers = np.einsum(
            "fji,fj->fi",
            maps[:, :dimension, :dimension],
            neighbor_centers - maps[:, :dimension, dimension],
        )
    connector = neighbor_centers - centers[owner_]
    denominator = np.linalg.norm(connector, axis=-1) * measures
    cosine = np.abs(np.sum(connector * areas, axis=-1)) / np.where(
        denominator > 0.0, denominator, 1.0
    )
    maximum_nonorthogonality = np.max(
        np.where(
            interior,
            np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))),
            0.0,
        )
    )
    match connectivity:
        case PolygonalConnectivity():
            cell_edges = np.asarray(host_connectivity.cell_edges, dtype=np.int32)
            valid = np.asarray(host_connectivity.cell_edge_valid)
            edges = np.asarray(host_connectivity.edges)
            lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=-1)
            cell_maximum = np.max(np.where(valid, lengths[cell_edges], 0.0), axis=1)
            aspect = cell_maximum**2 / volumes
        case TetrahedralConnectivity():
            cells = np.asarray(tetrahedra, dtype=np.int32)
            cell_points = points[cells]
            pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
            edge_lengths = np.stack(
                tuple(
                    np.linalg.norm(cell_points[:, right] - cell_points[:, left], axis=-1)
                    for left, right in pairs
                ),
                axis=1,
            )
            cell_faces = np.asarray(host_connectivity.cell_faces, dtype=np.int32)
            maximum_face = np.max(measures[cell_faces], axis=1)
            minimum_altitude = 3.0 * volumes / maximum_face
            aspect = np.max(edge_lengths, axis=1) / minimum_altitude
        case HexahedralConnectivity():
            maximum_face_scale = np.max(
                np.sqrt(measures[np.asarray(host_connectivity.cell_faces)]), axis=1
            )
            aspect = maximum_face_scale / np.cbrt(volumes)
        case PolyhedralConnectivity():
            cell_faces = np.asarray(host_connectivity.cell_face_values, dtype=np.int32)
            counts = np.diff(
                np.asarray(host_connectivity.cell_face_offsets, dtype=np.int32)
            )
            cell_ids = np.repeat(
                np.arange(connectivity.cell_count, dtype=np.int32), counts
            )
            maximum_face_scale = np.full(
                connectivity.cell_count, -np.inf, dtype=measures.dtype
            )
            np.maximum.at(maximum_face_scale, cell_ids, np.sqrt(measures[cell_faces]))
            aspect = maximum_face_scale / np.cbrt(volumes)
        case _:
            assert_never(connectivity)
    return UnstructuredFiniteVolumeQualityReport(
        minimum_cell_measure=jnp.asarray(np.min(volumes)),
        maximum_cell_measure=jnp.asarray(np.max(volumes)),
        minimum_face_measure=jnp.asarray(np.min(measures)),
        maximum_aspect_ratio=jnp.asarray(np.max(aspect)),
        maximum_nonorthogonality_degrees=jnp.asarray(maximum_nonorthogonality),
        maximum_closure_residual=jnp.asarray(
            np.max(np.linalg.norm(closure_vectors, axis=-1))
        ),
        worst_cell=jnp.asarray(np.argmax(aspect)),
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
