#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified geometric validity of mapped cells through Bernstein bounds.

The Jacobian determinant of a polynomial geometry map is itself a polynomial of
known degree on the reference cell. Converting it to Bernstein form gives
coefficients whose minimum and maximum enclose the determinant over the whole
cell (convex-hull property), while every corner coefficient equals the value at
that corner. Adaptive subdivision tightens the enclosure until the determinant
is proven to stay above the policy floor (CERTIFIED_VALID), a corner value of a
sub-piece proves it falls below the floor (INVALID), or the resource budget is
exhausted (UNRESOLVED).

Rational pyramids are certified through the collapsed coordinates
``(u, v, w) -> (u (1 - w) + w / 2, v (1 - w) + w / 2, w)``: the physical
determinant pulled back to the unit cube is the numerator of the rational
determinant divided by ``(1 - w)^2`` and is a tensor polynomial of degree
``(3k - 1, 3k - 1, 3k - 3)`` whose homogeneous Bernstein coefficients are bounded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from functools import cache
from itertools import product
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._geometry_predicates import (
    polygon_simplicity_2d,
    PolygonSimplicityStatus,
    PredicateMode,
    PredicateSign,
    resolve_host_predicate_mode,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry import CellGeometrySpec, CellVertexGeometryElement
from ._cell_mesh import CellMesh


class CellValidityStatus(IntEnum):
    CERTIFIED_VALID = 0
    INVALID = 1
    UNRESOLVED = 2


class CellValidityPolicy(StrictModule, NonTrainableState):
    """Resource and degeneracy controls for Bernstein validity certification.

    ``relative_determinant_floor`` scales each cell's Hadamard bound of the
    evaluated determinant; a determinant below the scaled floor is degenerate and
    therefore INVALID. For polygons it also sets the minimum edge length relative
    to the polygon diameter, preventing a numerically collapsed edge from being
    certified solely because the total area remains large. ``maximum_piece_count``
    bounds the active sub-pieces of one block at one subdivision level; exhausting
    it or the depth leaves the undecided cells UNRESOLVED.
    ``relative_planarity_tolerance`` bounds the distance of embedded polygon
    vertices from their Newell plane relative to the polygon diameter.
    """

    maximum_subdivision_depth: int = eqx.field(static=True)
    maximum_piece_count: int = eqx.field(static=True)
    relative_determinant_floor: float = eqx.field(static=True)
    relative_planarity_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_subdivision_depth: int = 8,
        maximum_piece_count: int = 1_000_000,
        relative_determinant_floor: float = 1.0e-12,
        relative_planarity_tolerance: float = 1.0e-10,
    ) -> None:
        depth = int(maximum_subdivision_depth)
        pieces = int(maximum_piece_count)
        floor = float(relative_determinant_floor)
        planarity = float(relative_planarity_tolerance)
        if depth < 0:
            raise ValueError("maximum_subdivision_depth must be non-negative.")
        if pieces <= 0:
            raise ValueError("maximum_piece_count must be positive.")
        if not math.isfinite(floor) or floor < 0.0 or floor >= 1.0:
            raise ValueError("relative_determinant_floor must lie in [0, 1).")
        if not math.isfinite(planarity) or planarity < 0.0 or planarity >= 1.0:
            raise ValueError("relative_planarity_tolerance must lie in [0, 1).")
        self.maximum_subdivision_depth = depth
        self.maximum_piece_count = pieces
        self.relative_determinant_floor = floor
        self.relative_planarity_tolerance = planarity
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cell-validity-policy",
                "maximum_subdivision_depth": depth,
                "maximum_piece_count": pieces,
                "relative_determinant_floor": floor,
                "relative_planarity_tolerance": planarity,
            }
        )


class CellValidityCertificate(StrictModule, NonTrainableState):
    """Per-cell validity status with determinant enclosures.

    ``determinant_lower``/``determinant_upper`` enclose the determinant (signed
    Jacobian determinant for full-dimensional cells, Gram determinant for
    embedded cells, twice the signed area of planar polygons, the squared vector
    area of embedded polygons, star-simplex determinants for polyhedra) over the
    final subdivision; ``depth`` is the deepest subdivision level evaluated.
    Cells of ``unsupported_block_names`` are UNRESOLVED with NaN bounds because
    their coordinate element has no polynomial degree contract.
    """

    status: Array
    determinant_lower: Array
    determinant_upper: Array
    depth: Array
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    unsupported_block_names: tuple[str, ...] = eqx.field(static=True)
    certified_valid_count: int = eqx.field(static=True)
    invalid_count: int = eqx.field(static=True)
    unresolved_count: int = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    certificate_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: Any,
        determinant_lower: Any,
        determinant_upper: Any,
        depth: Any,
        /,
        *,
        block_names: tuple[str, ...],
        block_offsets: tuple[int, ...],
        unsupported_block_names: tuple[str, ...],
        geometry_id: str,
        geometry_layout_id: str,
        policy_id: str,
    ) -> None:
        status_ = np.asarray(status, dtype=np.int32)
        lower = np.asarray(determinant_lower, dtype=np.float64)
        upper = np.asarray(determinant_upper, dtype=np.float64)
        depth_ = np.asarray(depth, dtype=np.int32)
        names = tuple(str(value) for value in block_names)
        offsets = tuple(int(value) for value in block_offsets)
        unsupported = tuple(str(value) for value in unsupported_block_names)
        if status_.ndim != 1 or any(
            value.shape != status_.shape for value in (lower, upper, depth_)
        ):
            raise ValueError("Validity certificate arrays must be aligned vectors.")
        if not np.all(
            np.isin(status_, tuple(int(value) for value in CellValidityStatus))
        ):
            raise ValueError("Validity status values must be CellValidityStatus codes.")
        if (
            len(offsets) != len(names) + 1
            or offsets[0] != 0
            or offsets[-1] != status_.size
            or any(stop < start for start, stop in zip(offsets[:-1], offsets[1:]))
        ):
            raise ValueError("Validity block offsets must partition the cells.")
        if not set(unsupported) <= set(names):
            raise ValueError("Unsupported validity blocks must be certificate blocks.")
        self.status = jnp.asarray(status_)
        self.determinant_lower = jnp.asarray(lower)
        self.determinant_upper = jnp.asarray(upper)
        self.depth = jnp.asarray(depth_)
        self.block_names = names
        self.block_offsets = offsets
        self.unsupported_block_names = unsupported
        self.certified_valid_count = int(
            np.count_nonzero(status_ == CellValidityStatus.CERTIFIED_VALID)
        )
        self.invalid_count = int(np.count_nonzero(status_ == CellValidityStatus.INVALID))
        self.unresolved_count = int(
            np.count_nonzero(status_ == CellValidityStatus.UNRESOLVED)
        )
        self.geometry_layout_id = str(geometry_layout_id)
        self.policy_id = str(policy_id)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "cell-validity-certificate",
                "geometry": str(geometry_id),
                "geometry_layout": self.geometry_layout_id,
                "policy": self.policy_id,
                "blocks": names,
                "block_offsets": offsets,
                "unsupported": unsupported,
                "status": array_tree_fingerprint(status_),
                "depth": array_tree_fingerprint(depth_),
            }
        )

    @property
    def all_certified(self) -> bool:
        return self.certified_valid_count == self.status.shape[0]


_POLYNOMIAL_FAMILIES = frozenset(
    ("Lagrange", "SimplexLagrange", "TensorProductLagrange", "HybridLagrange")
)
_EVALUATION_ENTRY_BUDGET = 1 << 23
_EPSILON = float(np.finfo(np.float64).eps)


# Bernstein parameter domains ------------------------------------------------


def _bernstein_1d(degree: int, points: np.ndarray, /) -> np.ndarray:
    powers = np.arange(degree + 1)
    binomial = np.asarray([math.comb(degree, value) for value in powers], np.float64)
    t = points[..., None]
    return binomial * t**powers * (1.0 - t) ** (degree - powers)


def _simplex_indices(degree: int, dimension: int, /) -> np.ndarray:
    return np.asarray(
        [
            index
            for index in product(range(degree + 1), repeat=dimension + 1)
            if sum(index) == degree
        ],
        dtype=np.int64,
    ).reshape(-1, dimension + 1)


def _bernstein_simplex(degree: int, dimension: int, points: np.ndarray, /) -> np.ndarray:
    indices = _simplex_indices(degree, dimension)
    barycentric = np.concatenate(
        (1.0 - np.sum(points, axis=-1, keepdims=True), points), axis=-1
    )
    multinomial = np.asarray(
        [
            math.factorial(degree) / math.prod(math.factorial(value) for value in row)
            for row in indices
        ],
        dtype=np.float64,
    )
    return multinomial * np.prod(
        barycentric[..., None, :] ** indices[None, :, :], axis=-1
    )


def _chebyshev_nodes(degree: int, /) -> np.ndarray:
    index = np.arange(degree + 1, dtype=np.float64)
    return 0.5 * (1.0 - np.cos((2.0 * index + 1.0) * np.pi / (2.0 * degree + 2.0)))


def _simplex_nodes(degree: int, dimension: int, /) -> np.ndarray:
    if degree == 0:
        return np.full((1, dimension), 1.0 / (dimension + 1.0))
    return _simplex_indices(degree, dimension)[:, 1:].astype(np.float64) / degree


def _tensor(values: tuple[np.ndarray, ...], /) -> np.ndarray:
    """Tensor product of per-axis basis tables sharing the leading point axis."""

    result = values[0]
    for value in values[1:]:
        result = (result[..., :, None] * value[..., None, :]).reshape(
            value.shape[:-1] + (-1,)
        )
    return result


def _grid(axes: tuple[np.ndarray, ...], /) -> np.ndarray:
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([value.reshape(-1) for value in mesh], axis=-1)


_INTERVAL_CHILDREN = ((0.0, 0.5), (0.5, 0.5))
_TRIANGLE_CHILDREN = (
    ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)),
    ((0.5, 0.0), (1.0, 0.0), (0.5, 0.5)),
    ((0.0, 0.5), (0.5, 0.5), (0.0, 1.0)),
    ((0.5, 0.5), (0.0, 0.5), (0.5, 0.0)),
)


def _tetrahedron_children() -> tuple[np.ndarray, ...]:
    vertices = np.eye(4, 3, k=-1)
    middle = {
        (first, second): 0.5 * (vertices[first] + vertices[second])
        for first in range(4)
        for second in range(first + 1, 4)
    }
    corners = (
        (vertices[0], middle[0, 1], middle[0, 2], middle[0, 3]),
        (middle[0, 1], vertices[1], middle[1, 2], middle[1, 3]),
        (middle[0, 2], middle[1, 2], vertices[2], middle[2, 3]),
        (middle[0, 3], middle[1, 3], middle[2, 3], vertices[3]),
    )
    # Red refinement: the inner octahedron is split around diagonal m02-m13,
    # whose equatorial cycle is m01, m03, m23, m12.
    cycle = (middle[0, 1], middle[0, 3], middle[2, 3], middle[1, 2])
    inner = tuple(
        (middle[0, 2], middle[1, 3], cycle[index], cycle[(index + 1) % 4])
        for index in range(4)
    )
    return tuple(np.asarray(value) for value in (*corners, *inner))


def _simplex_child_maps(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    if dimension == 2:
        children = tuple(np.asarray(value) for value in _TRIANGLE_CHILDREN)
    elif dimension == 3:
        children = _tetrahedron_children()
    elif dimension > 0:
        vertices = np.eye(dimension + 1, dimension, k=-1, dtype=np.float64)
        midpoint = 0.5 * (vertices[0] + vertices[1])
        first = vertices.copy()
        second = vertices.copy()
        first[1] = midpoint
        second[0] = midpoint
        children = (first, second)
    else:
        raise ValueError("Simplex subdivision requires positive dimension.")
    origins = np.stack([child[0] for child in children])
    matrices = np.stack([(child[1:] - child[0]).T for child in children])
    return origins, matrices


def _box_child_maps(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    choices = tuple(product(_INTERVAL_CHILDREN, repeat=dimension))
    origins = np.asarray([[origin for origin, _ in choice] for choice in choices])
    matrices = np.stack(
        [np.diag([scale for _, scale in choice]) for choice in choices]
    ).reshape(-1, dimension, dimension)
    return origins, matrices


def _prism_child_maps() -> tuple[np.ndarray, np.ndarray]:
    triangle_origins, triangle_matrices = _simplex_child_maps(2)
    origins = []
    matrices = []
    for origin, matrix in zip(triangle_origins, triangle_matrices, strict=True):
        for start, scale in _INTERVAL_CHILDREN:
            origins.append(np.concatenate((origin, (start,))))
            block = np.zeros((3, 3))
            block[:2, :2] = matrix
            block[2, 2] = scale
            matrices.append(block)
    return np.asarray(origins), np.asarray(matrices)


@dataclass(frozen=True)
class _BernsteinPlan:
    """Host-only interpolation and subdivision tables of one parameter domain."""

    domain: str
    nodes: np.ndarray
    coefficients_from_values: np.ndarray
    conversion_norm: float
    vertex_indices: np.ndarray
    child_origins: np.ndarray
    child_matrices: np.ndarray


@cache
def _bernstein_plan(domain: str, degrees: tuple[int, ...], /) -> _BernsteinPlan:
    match domain:
        case "box":
            axes = tuple(_chebyshev_nodes(degree) for degree in degrees)
            nodes = _grid(axes)
            basis = _tensor(
                tuple(
                    _bernstein_1d(degree, nodes[:, axis])
                    for axis, degree in enumerate(degrees)
                )
            )
            vertex_indices = np.ravel_multi_index(
                tuple(
                    np.asarray(values)
                    for values in zip(*product(*((0, degree) for degree in degrees)))
                ),
                tuple(degree + 1 for degree in degrees),
            )
            origins, matrices = _box_child_maps(len(degrees))
        case "simplex":
            degree, dimension = degrees
            nodes = _simplex_nodes(degree, dimension)
            basis = _bernstein_simplex(degree, dimension, nodes)
            indices = _simplex_indices(degree, dimension)
            vertex_indices = np.flatnonzero(np.max(indices, axis=1) == degree)
            origins, matrices = _simplex_child_maps(dimension)
        case "prism":
            triangle_degree, axial_degree = degrees
            triangle_nodes = _simplex_nodes(triangle_degree, 2)
            axial_nodes = _chebyshev_nodes(axial_degree)
            nodes = np.concatenate(
                (
                    np.repeat(triangle_nodes, axial_nodes.size, axis=0),
                    np.tile(axial_nodes, triangle_nodes.shape[0])[:, None],
                ),
                axis=1,
            )
            basis = _tensor(
                (
                    _bernstein_simplex(triangle_degree, 2, nodes[:, :2]),
                    _bernstein_1d(axial_degree, nodes[:, 2]),
                )
            )
            triangle_vertices = np.flatnonzero(
                np.max(_simplex_indices(triangle_degree, 2), axis=1) == triangle_degree
            )
            axial_vertices = np.unique(np.asarray((0, axial_degree)))
            vertex_indices = (
                triangle_vertices[:, None] * (axial_degree + 1) + axial_vertices[None, :]
            ).reshape(-1)
            origins, matrices = _prism_child_maps()
        case _:
            raise ValueError(f"Unknown Bernstein parameter domain {domain!r}.")
    # The conversion operator is applied to every piece of every level; it is
    # prepared once per (domain, degree) like other reference-element tables.
    conversion = np.linalg.solve(basis, np.eye(basis.shape[0]))
    return _BernsteinPlan(
        domain,
        nodes,
        conversion,
        float(np.max(np.sum(np.abs(conversion), axis=1))),
        np.unique(vertex_indices),
        origins,
        matrices,
    )


def _determinant_route(
    cell_kind: str, degree: int, embedded: bool, /
) -> tuple[str, tuple[int, ...]]:
    """Return the Bernstein domain and exact degree bound of the determinant."""

    scale = 2 if embedded else 1
    match cell_kind:
        case "interval":
            return "box", (scale * (degree - 1),)
        case "triangle":
            return "simplex", (scale * 2 * (degree - 1), 2)
        case "tetrahedron":
            return "simplex", (3 * (degree - 1), 3)
        case "quadrilateral":
            return "box", (scale * (2 * degree - 1),) * 2
        case "hexahedron":
            return "box", (3 * degree - 1,) * 3
        case "prism":
            return "prism", (3 * degree - 2, 3 * degree - 1)
        case "pyramid":
            return "box", (3 * degree - 1, 3 * degree - 1, 3 * degree - 3)
        case _:
            raise ValueError(f"No polynomial determinant route for {cell_kind!r}.")


def _reference_points(cell_kind: str, parameters: np.ndarray, /) -> np.ndarray:
    if cell_kind != "pyramid":
        return parameters
    height = parameters[..., 2:3]
    scale = 1.0 - height
    return np.concatenate(
        (
            parameters[..., :2] * scale + 0.5 * height,
            height,
        ),
        axis=-1,
    )


# Determinants ------------------------------------------------------------------


def _square_determinant(matrix: np.ndarray, /) -> np.ndarray:
    size = matrix.shape[-1]
    if size == 1:
        return matrix[..., 0, 0]
    if size == 2:
        return (
            matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]
        )
    return (
        matrix[..., 0, 0]
        * (matrix[..., 1, 1] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 1])
        - matrix[..., 0, 1]
        * (matrix[..., 1, 0] * matrix[..., 2, 2] - matrix[..., 1, 2] * matrix[..., 2, 0])
        + matrix[..., 0, 2]
        * (matrix[..., 1, 0] * matrix[..., 2, 1] - matrix[..., 1, 1] * matrix[..., 2, 0])
    )


def _jacobian_determinant(
    jacobian: np.ndarray, magnitude: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return the determinant and its Hadamard scale for (..., ambient, reference)."""

    column_scale = np.prod(np.linalg.norm(magnitude, axis=-2), axis=-1)
    if jacobian.shape[-2] == jacobian.shape[-1]:
        return _square_determinant(jacobian), column_scale
    gram = np.swapaxes(jacobian, -1, -2) @ jacobian
    return _square_determinant(gram), column_scale * column_scale


# Adaptive certification ----------------------------------------------------------


@dataclass(frozen=True)
class _BlockCertificate:
    status: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    depth: np.ndarray


def _evaluate_pieces(
    element: Any,
    cell_kind: str,
    plan: _BernsteinPlan,
    local: np.ndarray,
    cells: np.ndarray,
    origins: np.ndarray,
    matrices: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Bernstein coefficients and rounding margins for every piece."""

    node_count = plan.nodes.shape[0]
    dof_count = local.shape[1]
    entries_per_piece = node_count * dof_count * local.shape[2] * plan.nodes.shape[1]
    # Pieces are processed in bounded chunks so the tabulated gradients respect a
    # fixed host-memory budget independent of the subdivision state.
    chunk = max(1, _EVALUATION_ENTRY_BUDGET // max(entries_per_piece, 1))
    coefficients = []
    margins = []
    for start in range(0, cells.size, chunk):
        stop = min(start + chunk, cells.size)
        parameters = origins[start:stop, None, :] + ein.contract(
            "pij,mj->pmi", matrices[start:stop], plan.nodes
        )
        reference = _reference_points(cell_kind, parameters)
        _, gradients = element.tabulate(reference.reshape(-1, reference.shape[-1]))
        gradients = np.asarray(gradients, dtype=np.float64).reshape(
            (stop - start, node_count, dof_count, reference.shape[-1])
        )
        points = local[cells[start:stop]]
        jacobian = ein.contract("pmnk,pna->pmak", gradients, points)
        magnitude = ein.contract("pmnk,pna->pmak", np.abs(gradients), np.abs(points))
        values, scale = _jacobian_determinant(jacobian, magnitude)
        coefficient = ein.contract("rm,pm->pr", plan.coefficients_from_values, values)
        # Forward error of the determinant evaluation plus the conversion rounding.
        evaluation_error = 8.0 * (dof_count + reference.shape[-1] + 1) * _EPSILON
        margin = (
            2.0
            * plan.conversion_norm
            * (
                evaluation_error * np.max(scale, axis=1)
                + node_count * _EPSILON * np.max(np.abs(values), axis=1)
            )
        )
        coefficients.append(coefficient)
        margins.append(margin)
    return np.concatenate(coefficients), np.concatenate(margins)


def _root_scale(
    element: Any, cell_kind: str, plan: _BernsteinPlan, local: np.ndarray, /
) -> np.ndarray:
    reference = _reference_points(cell_kind, plan.nodes)
    _, gradients = element.tabulate(reference)
    gradients = np.asarray(gradients, dtype=np.float64)
    magnitude = ein.contract("mnk,cna->cmak", np.abs(gradients), np.abs(local))
    column_scale = np.prod(np.linalg.norm(magnitude, axis=-2), axis=-1)
    embedded = local.shape[-1] != reference.shape[-1]
    return np.max(column_scale * column_scale if embedded else column_scale, axis=1)


def _certify_polynomial_block(
    element: Any,
    cell_kind: str,
    local: np.ndarray,
    policy: CellValidityPolicy,
    /,
) -> _BlockCertificate:
    embedded = local.shape[-1] != element.topological_dimension
    if embedded and cell_kind not in ("interval", "triangle", "quadrilateral"):
        raise ValueError(f"Embedded {cell_kind} cells have no validity contract.")
    domain, degrees = _determinant_route(cell_kind, element.degree, embedded)
    plan = _bernstein_plan(domain, degrees)
    cell_count = local.shape[0]
    threshold = policy.relative_determinant_floor * _root_scale(
        element, cell_kind, plan, local
    )
    lower = np.full((cell_count,), np.inf)
    upper = np.full((cell_count,), -np.inf)
    depth = np.zeros((cell_count,), dtype=np.int32)
    invalid = np.zeros((cell_count,), dtype=np.bool_)
    unresolved = np.zeros((cell_count,), dtype=np.bool_)
    dimension = plan.nodes.shape[1]
    cells = np.arange(cell_count, dtype=np.int64)
    origins = np.zeros((cell_count, dimension))
    matrices = np.broadcast_to(np.eye(dimension), (cell_count, dimension, dimension))
    child_count = plan.child_origins.shape[0]
    for level in range(policy.maximum_subdivision_depth + 1):
        coefficients, margins = _evaluate_pieces(
            element, cell_kind, plan, local, cells, origins, matrices
        )
        piece_threshold = threshold[cells]
        piece_lower = np.min(coefficients, axis=1) - margins
        piece_upper = np.max(coefficients, axis=1) + margins
        corner = np.min(coefficients[:, plan.vertex_indices], axis=1)
        piece_valid = piece_lower > piece_threshold
        piece_invalid = corner + margins < piece_threshold
        np.logical_or.at(invalid, cells[piece_invalid], True)
        np.maximum.at(depth, cells, level)
        refine = ~piece_valid & ~piece_invalid & ~invalid[cells]
        exhausted = (
            level == policy.maximum_subdivision_depth
            or np.count_nonzero(refine) * child_count > policy.maximum_piece_count
        )
        if exhausted:
            unresolved[cells[refine]] = True
            refine = np.zeros_like(refine)
        leaf = ~refine
        np.minimum.at(lower, cells[leaf], piece_lower[leaf])
        np.maximum.at(upper, cells[leaf], piece_upper[leaf])
        if not np.any(refine):
            break
        parent_origins = origins[refine]
        parent_matrices = matrices[refine]
        origins = (
            parent_origins[:, None, :]
            + ein.contract("pij,kj->pki", parent_matrices, plan.child_origins)
        ).reshape(-1, dimension)
        matrices = ein.contract(
            "pij,kjl->pkil", parent_matrices, plan.child_matrices
        ).reshape(-1, dimension, dimension)
        cells = np.repeat(cells[refine], child_count)
    status = np.where(
        invalid,
        CellValidityStatus.INVALID,
        np.where(
            unresolved, CellValidityStatus.UNRESOLVED, CellValidityStatus.CERTIFIED_VALID
        ),
    ).astype(np.int32)
    return _BlockCertificate(status, lower, upper, depth)


# Star-decomposed variable-topology cells --------------------------------------------


@dataclass(frozen=True)
class PolyhedralStarTables:
    """Host CSR tables of the face-centroid star decomposition of polyhedra.

    Star entry ``e`` is the tetrahedron ``(cell centroid, face centroid,
    star_first[e], star_second[e])`` whose base edge follows the outward
    orientation of face ``star_face[e]`` in cell ``star_cell[e]``;
    ``star_sign[e]`` orients the stored face loop outward. Each cell edge
    appears in exactly two entries; ``edge_pairs`` lists them with the first
    entry fixing the edge direction.
    """

    face_sizes: np.ndarray
    face_corner_face: np.ndarray
    face_corner_vertex: np.ndarray
    face_corner_next: np.ndarray
    cell_vertex_cell: np.ndarray
    cell_vertex_values: np.ndarray
    cell_vertex_counts: np.ndarray
    star_cell: np.ndarray
    star_face: np.ndarray
    star_first: np.ndarray
    star_second: np.ndarray
    star_sign: np.ndarray
    edge_pairs: np.ndarray


def polyhedral_star_tables(
    connectivity: PolyhedralConnectivity, /
) -> PolyhedralStarTables:
    if not isinstance(connectivity, PolyhedralConnectivity):
        raise TypeError("connectivity must be PolyhedralConnectivity.")
    face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
    face_values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
    cell_face_offsets = np.asarray(connectivity.cell_face_offsets, dtype=np.int64)
    cell_faces = np.asarray(connectivity.cell_face_values, dtype=np.int64)
    cell_signs = np.asarray(connectivity.cell_face_sign_values, dtype=np.float64)
    cell_vertex_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
    cell_vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
    face_sizes = np.diff(face_offsets)
    corner_face = np.repeat(np.arange(face_sizes.size), face_sizes)
    corner_local = np.arange(face_values.size) - face_offsets[corner_face]
    corner_next = face_values[
        face_offsets[corner_face] + (corner_local + 1) % face_sizes[corner_face]
    ]
    incidence_cell = np.repeat(
        np.arange(cell_face_offsets.size - 1), np.diff(cell_face_offsets)
    )
    entry_counts = face_sizes[cell_faces]
    entry_offsets = np.concatenate(((0,), np.cumsum(entry_counts)))
    entry_incidence = np.repeat(np.arange(cell_faces.size), entry_counts)
    entry_corner = face_offsets[cell_faces[entry_incidence]] + (
        np.arange(entry_offsets[-1]) - entry_offsets[entry_incidence]
    )
    outward = cell_signs[entry_incidence] > 0.0
    first = np.where(outward, face_values[entry_corner], corner_next[entry_corner])
    second = np.where(outward, corner_next[entry_corner], face_values[entry_corner])
    star_cell = incidence_cell[entry_incidence]
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    order = np.lexsort((first, high, low, star_cell))
    if order.size % 2 or not (
        np.array_equal(star_cell[order[0::2]], star_cell[order[1::2]])
        and np.array_equal(low[order[0::2]], low[order[1::2]])
        and np.array_equal(high[order[0::2]], high[order[1::2]])
        and np.array_equal(first[order[0::2]], second[order[1::2]])
    ):
        raise ValueError(
            "Polyhedral cells must be closed, consistently oriented surfaces."
        )
    return PolyhedralStarTables(
        face_sizes=face_sizes,
        face_corner_face=corner_face,
        face_corner_vertex=face_values,
        face_corner_next=corner_next,
        cell_vertex_cell=np.repeat(
            np.arange(cell_vertex_offsets.size - 1), np.diff(cell_vertex_offsets)
        ),
        cell_vertex_values=cell_vertices,
        cell_vertex_counts=np.diff(cell_vertex_offsets),
        star_cell=star_cell,
        star_face=cell_faces[entry_incidence],
        star_first=first,
        star_second=second,
        star_sign=np.where(outward, 1.0, -1.0),
        edge_pairs=np.stack((order[0::2], order[1::2]), axis=1),
    )


def _segment_mean(values: np.ndarray, segments: np.ndarray, count: int, /) -> np.ndarray:
    totals = np.zeros((count,) + values.shape[1:])
    np.add.at(totals, segments, values)
    sizes = np.bincount(segments, minlength=count).astype(np.float64)
    return totals / sizes.reshape((-1,) + (1,) * (values.ndim - 1))


def _star_status(
    determinants: np.ndarray,
    margins: np.ndarray,
    segments: np.ndarray,
    thresholds: np.ndarray,
    total_measure: np.ndarray,
    /,
) -> _BlockCertificate:
    """Classify cells from exact star-simplex determinants.

    All star simplices above the floor certify an invertible piecewise-affine
    star map. A total signed measure below the floor proves inversion or
    degeneracy. Anything else is not star-shaped about the chosen center, which
    is not a validity disproof, so it remains UNRESOLVED.
    """

    count = thresholds.size
    lower = np.full((count,), np.inf)
    upper = np.full((count,), -np.inf)
    np.minimum.at(lower, segments, determinants - margins)
    np.maximum.at(upper, segments, determinants + margins)
    status = np.where(
        lower > thresholds,
        CellValidityStatus.CERTIFIED_VALID,
        np.where(
            total_measure < thresholds,
            CellValidityStatus.INVALID,
            CellValidityStatus.UNRESOLVED,
        ),
    ).astype(np.int32)
    return _BlockCertificate(status, lower, upper, np.zeros((count,), dtype=np.int32))


def _polygon_measure(points: np.ndarray, policy: CellValidityPolicy, /) -> tuple:
    """Determinant enclosure, floor, planarity flags, and planar loops of polygons.

    The determinant is twice the signed shoelace area about the vertex centroid
    (planar) or the squared Newell vector area (embedded, Gram convention).
    Embedded loops are measured against their Newell plane through the vertex
    centroid and projected by dropping the dominant normal axis, which keeps the
    projected coordinates exact. Returns ``(determinant, margin, threshold,
    nonplanar, ambiguous_planarity, planar_points)``.
    """

    cell_count, corner_count, ambient = points.shape
    center = np.mean(points, axis=1, keepdims=True)
    first = points - center
    second = np.roll(first, -1, axis=1)
    scale = np.sum(
        np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1), axis=1
    )
    rounding = 2.0 * (corner_count + 8) * _EPSILON * scale
    floor = policy.relative_determinant_floor
    if ambient == 2:
        area = np.sum(
            first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0], axis=1
        )
        planar = np.zeros((cell_count,), dtype=np.bool_)
        return area, rounding, floor * scale, planar, planar, points
    area = np.sum(np.cross(first, second), axis=1)
    magnitude = np.linalg.norm(area, axis=-1)
    guarded = np.maximum(magnitude, np.finfo(np.float64).tiny)
    # Vertex offsets from the Newell plane; the normal direction inherits the
    # relative rounding of the area vector.
    offset = np.max(
        np.abs(np.sum(first * (area / guarded[:, None])[:, None], axis=-1)), axis=1
    )
    slack = np.max(np.linalg.norm(first, axis=-1), axis=1) * (
        16.0 * _EPSILON + rounding / guarded
    )
    diameter = np.linalg.norm(np.max(points, axis=1) - np.min(points, axis=1), axis=-1)
    allowed = policy.relative_planarity_tolerance * diameter
    axes = np.asarray(((1, 2), (0, 2), (0, 1)))[np.argmax(np.abs(area), axis=-1)]
    return (
        magnitude * magnitude,
        2.0 * magnitude * rounding + rounding * rounding,
        floor * scale * scale,
        offset - slack > allowed,
        offset + slack > allowed,
        np.take_along_axis(points, axes[:, None, :], axis=2),
    )


def _certify_polygon_block(points: np.ndarray, policy: CellValidityPolicy, /) -> Any:
    """Certify polygon cells without assuming star-shapedness.

    A planar polygon is valid iff its vertices are finite and distinct, its
    boundary is simple, it is counterclockwise, and twice its area exceeds the
    policy floor. An embedded polygon must instead lie within
    ``relative_planarity_tolerance`` of its Newell plane, be simple in its
    dominant-axis projection, and have a squared vector area above the floor.
    """

    cell_count, corner_count, ambient = points.shape
    mode = resolve_host_predicate_mode(PredicateMode.EXACT)
    finite = np.all(np.isfinite(points), axis=(1, 2))
    safe = np.where(finite[:, None, None], points, 0.0)
    coincident = np.all(safe[:, :, None] == safe[:, None, :], axis=-1)
    repeated = np.any(coincident & ~np.eye(corner_count, dtype=np.bool_), axis=(1, 2))
    centered = safe - np.mean(safe, axis=1, keepdims=True)
    following = np.roll(centered, -1, axis=1)
    edge_vectors = following - centered
    edge_lengths = np.linalg.norm(edge_vectors, axis=-1)
    edge_roundoff = (
        4.0 * _EPSILON * np.linalg.norm(np.abs(centered) + np.abs(following), axis=-1)
    )
    diameter = np.linalg.norm(np.max(safe, axis=1) - np.min(safe, axis=1), axis=-1)
    edge_floor = policy.relative_determinant_floor * diameter[:, None]
    short_edge = np.any(edge_lengths + edge_roundoff < edge_floor, axis=1)
    uncertain_edge = np.any(edge_lengths - edge_roundoff <= edge_floor, axis=1)
    determinant, margin, threshold, nonplanar, ambiguous, planar = _polygon_measure(
        safe, policy
    )
    invalid = ~finite | repeated | nonplanar | short_edge
    candidates = np.flatnonzero(~invalid)
    simplicity = np.full((cell_count,), PolygonSimplicityStatus.SIMPLE, dtype=np.int8)
    # Embedded polygons carry no ambient orientation convention.
    orientation = np.full((cell_count,), PredicateSign.POSITIVE, dtype=np.int8)
    if candidates.size:
        result = polygon_simplicity_2d(planar[candidates], mode=mode)
        simplicity[candidates] = np.asarray(result.status)
        if ambient == 2:
            orientation[candidates] = np.asarray(result.orientation)
    simple = simplicity == PolygonSimplicityStatus.SIMPLE
    lower = determinant - margin
    upper = determinant + margin
    invalid |= (simplicity == PolygonSimplicityStatus.SELF_INTERSECTING) | (
        simple & ((orientation == PredicateSign.NEGATIVE) | (upper < threshold))
    )
    # A certified simple loop has exactly nonzero area, which meets a zero floor.
    above = (lower > threshold) | (simple & (threshold == 0.0))
    unresolved = (
        ambiguous
        | (uncertain_edge & ~short_edge)
        | ~simple
        | (orientation == PredicateSign.UNCERTAIN)
        | ~above
    )
    status = np.where(
        invalid,
        CellValidityStatus.INVALID,
        np.where(
            unresolved,
            CellValidityStatus.UNRESOLVED,
            CellValidityStatus.CERTIFIED_VALID,
        ),
    ).astype(np.int32)
    return _BlockCertificate(
        status,
        np.where(finite, lower, np.nan),
        np.where(finite, upper, np.nan),
        np.zeros((cell_count,), dtype=np.int32),
    )


def _certify_polyhedral_cells(
    tables: PolyhedralStarTables,
    coordinates: np.ndarray,
    cells: np.ndarray,
    policy: CellValidityPolicy,
    /,
) -> _BlockCertificate:
    """Certify polyhedral cells ``cells`` (global connectivity rows) of one block."""

    cell_total = tables.cell_vertex_counts.size
    face_total = tables.face_sizes.size
    center = _segment_mean(
        coordinates[tables.cell_vertex_values], tables.cell_vertex_cell, cell_total
    )
    face_center = _segment_mean(
        coordinates[tables.face_corner_vertex], tables.face_corner_face, face_total
    )
    selected = np.isin(tables.star_cell, cells)
    star_cell = tables.star_cell[selected]
    apex = center[star_cell]
    columns = np.stack(
        (
            face_center[tables.star_face[selected]] - apex,
            coordinates[tables.star_first[selected]] - apex,
            coordinates[tables.star_second[selected]] - apex,
        ),
        axis=-1,
    )
    determinants = _square_determinant(columns)
    scale = np.prod(np.linalg.norm(columns, axis=-2), axis=-1)
    local = np.searchsorted(cells, star_cell)
    thresholds = np.zeros((cells.size,))
    np.maximum.at(thresholds, local, policy.relative_determinant_floor * scale)
    total = np.zeros((cells.size,))
    np.add.at(total, local, determinants)
    return _star_status(determinants, 32.0 * _EPSILON * scale, local, thresholds, total)


# Public entry point ----------------------------------------------------------------


def _polyhedral_coordinates(
    mesh: CellMesh, routes: tuple[np.ndarray, ...], values: np.ndarray, /
) -> np.ndarray:
    """Gather vertex-geometry rows of polyhedral blocks into mesh-vertex order."""

    points = np.full((mesh.coordinates.shape[0], values.shape[1]), np.nan)
    for block, route in zip(mesh.blocks, routes, strict=True):
        if block.cell_kind == "polyhedron":
            vertices = np.asarray(block.vertices, dtype=np.int64)
            valid = np.asarray(block.vertex_valid, dtype=np.bool_)
            points[vertices[valid]] = values[route[valid]]
    return points


def certify_cell_geometry_validity(
    geometry: CellMesh | CellGeometrySpec,
    /,
    *,
    mesh: CellMesh | None = None,
    policy: CellValidityPolicy | None = None,
) -> CellValidityCertificate:
    """Certify determinant positivity of every mapped cell with Bernstein bounds.

    A ``CellMesh`` is certified with its affine vertex geometry. A
    ``CellGeometrySpec`` is certified in mesh block order when ``mesh`` is given
    (required for polyhedral blocks) and in its own sorted block order otherwise.
    """

    # The FE reference owner imports this package; resolve it lazily.
    from .fem._reference import FiniteElementSpec

    policy_ = CellValidityPolicy() if policy is None else policy
    if not isinstance(policy_, CellValidityPolicy):
        raise TypeError("policy must be CellValidityPolicy or None.")
    if isinstance(geometry, CellMesh):
        if mesh is not None:
            raise ValueError("mesh is implied when certifying a CellMesh.")
        mesh_ = geometry
        spec = CellGeometrySpec.affine(geometry)
    elif isinstance(geometry, CellGeometrySpec):
        mesh_ = mesh
        spec = geometry
    else:
        raise TypeError("geometry must be CellMesh or CellGeometrySpec.")
    if mesh_ is None:
        elements = spec.elements
        routes = tuple(np.asarray(value, dtype=np.int64) for value in spec.geometry_dofs)
        names = spec.block_names
        kinds = tuple(element.cell_kind for element in elements)
    else:
        if not isinstance(mesh_, CellMesh):
            raise TypeError("mesh must be CellMesh or None.")
        resolved, resolved_routes, _ = spec.resolve(mesh_)
        elements = resolved
        routes = tuple(np.asarray(value, dtype=np.int64) for value in resolved_routes)
        names = tuple(block.name for block in mesh_.blocks)
        kinds = tuple(block.cell_kind for block in mesh_.blocks)
    values = np.asarray(spec.coordinates, dtype=np.float64)
    tables = None
    polyhedral_points = None
    cursor = 0
    offsets = [0]
    unsupported = []
    blocks = []
    for name, kind, element, route in zip(names, kinds, elements, routes, strict=True):
        cell_count = route.shape[0]
        if isinstance(element, CellVertexGeometryElement) and kind == "polygon":
            blocks.append(_certify_polygon_block(values[route], policy_))
        elif isinstance(element, CellVertexGeometryElement):
            if mesh_ is None:
                raise ValueError("Polyhedral validity certification requires the mesh.")
            if tables is None:
                connectivity = mesh_.connectivity
                if not isinstance(connectivity, PolyhedralConnectivity):
                    raise TypeError(
                        "Polyhedral geometry requires polyhedral mesh connectivity."
                    )
                tables = polyhedral_star_tables(connectivity)
                polyhedral_points = _polyhedral_coordinates(mesh_, routes, values)
            if polyhedral_points is None:
                raise RuntimeError(
                    "Polyhedral coordinate preparation did not produce coordinates."
                )
            blocks.append(
                _certify_polyhedral_cells(
                    tables,
                    polyhedral_points,
                    np.arange(cursor, cursor + cell_count),
                    policy_,
                )
            )
        elif (
            isinstance(element, FiniteElementSpec)
            and element.family in _POLYNOMIAL_FAMILIES
            and element.conformity == "H1"
            and element.degree >= 1
        ):
            local = values[route]
            blocks.append(
                _certify_polynomial_block(element, kind, local - local[:, :1], policy_)
            )
        else:
            unsupported.append(name)
            blocks.append(
                _BlockCertificate(
                    np.full((cell_count,), CellValidityStatus.UNRESOLVED, np.int32),
                    np.full((cell_count,), np.nan),
                    np.full((cell_count,), np.nan),
                    np.zeros((cell_count,), dtype=np.int32),
                )
            )
        cursor += cell_count
        offsets.append(cursor)
    return CellValidityCertificate(
        np.concatenate([block.status for block in blocks]),
        np.concatenate([block.lower for block in blocks]),
        np.concatenate([block.upper for block in blocks]),
        np.concatenate([block.depth for block in blocks]),
        block_names=names,
        block_offsets=tuple(offsets),
        unsupported_block_names=tuple(unsupported),
        geometry_id=canonical_fingerprint(
            {
                "layout": spec.geometry_layout_id,
                "coordinates": array_tree_fingerprint(values),
            }
        ),
        geometry_layout_id=spec.geometry_layout_id,
        policy_id=policy_.policy_id,
    )


__all__ = [
    "CellValidityCertificate",
    "CellValidityPolicy",
    "CellValidityStatus",
    "certify_cell_geometry_validity",
]
