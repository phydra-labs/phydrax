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

Supported coordinate maps are H1 Lagrange families of every degree ``k >= 1`` on
intervals, triangles, tetrahedra, quadrilaterals, hexahedra, prisms and
(rational) pyramids, plus embedded intervals, triangles and quadrilaterals via
the Gram determinant. Admission uses the exact simplified coordinate
expressions: their actual Bernstein tables must fit
``CellValidityPolicy.maximum_bernstein_nodes`` or the block remains UNRESOLVED
with reason ``bernstein_node_budget``.

Source rounding enclosure. Canonical coordinate basis expressions and their
stored binary64 coefficients are converted to exact rational power coefficients.
Their determinant is formed by exact polynomial algebra, including exact
cancellation of the pyramid collapse factor. Exact power-to-Bernstein conversion
and dyadic subdivision give coefficient bounds without interpolating tabulated
values. Only the published float bounds are rounded, outward by one ULP.
An arbitrary tabulator claiming a supported family is not a source expression
and remains UNRESOLVED; partition-of-unity identities cannot establish a
pointwise basis error.

Exact coordinate sources. An exact PLC source certifies its degree-one
tetrahedra on the exact rational source points ``S`` of
``CellGeometrySpec.source_coordinates()``, never on their correctly rounded
binary64 carrier: the same polynomial route, canonical chart and relative floor
apply, so a carrier that is positively oriented while ``S`` is not is INVALID.
Exact power sources certify their source star decomposition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from itertools import product
from typing import Any, assert_never, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

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
from ..typing import parse
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry import (
    CellGeometrySpec,
    CellVertexGeometryElement,
    LayerColumnCellGeometryElement,
)
from ._cell_mesh import CellMesh
from ._coordinate_enclosure import CoordinateSourceBank
from ._exact_plc_geometry import (
    ExactPlcCellGeometryConvexSource,
    ExactPlcCellGeometrySource,
)
from ._exact_power_geometry import (
    ExactPowerCellGeometryLinearActionSource,
    ExactPowerCellGeometryRestrictionSource,
    ExactPowerCellGeometrySource,
)


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
    ``maximum_bernstein_nodes`` bounds the interpolation table of one
    determinant degree (and therefore the admitted polynomial degree).
    """

    maximum_subdivision_depth: int = eqx.field(static=True)
    maximum_piece_count: int = eqx.field(static=True)
    relative_determinant_floor: float = eqx.field(static=True)
    relative_planarity_tolerance: float = eqx.field(static=True)
    maximum_bernstein_nodes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_subdivision_depth: int = 8,
        maximum_piece_count: int = 1_000_000,
        relative_determinant_floor: float = 1.0e-12,
        relative_planarity_tolerance: float = 1.0e-10,
        maximum_bernstein_nodes: int = 4096,
    ) -> None:
        depth = int(maximum_subdivision_depth)
        pieces = int(maximum_piece_count)
        floor = float(relative_determinant_floor)
        planarity = float(relative_planarity_tolerance)
        nodes = int(maximum_bernstein_nodes)
        if depth < 0:
            raise ValueError("maximum_subdivision_depth must be non-negative.")
        if pieces <= 0:
            raise ValueError("maximum_piece_count must be positive.")
        if not math.isfinite(floor) or floor < 0.0 or floor >= 1.0:
            raise ValueError("relative_determinant_floor must lie in [0, 1).")
        if not math.isfinite(planarity) or planarity < 0.0 or planarity >= 1.0:
            raise ValueError("relative_planarity_tolerance must lie in [0, 1).")
        if nodes <= 0:
            raise ValueError("maximum_bernstein_nodes must be positive.")
        self.maximum_subdivision_depth = depth
        self.maximum_piece_count = pieces
        self.relative_determinant_floor = floor
        self.relative_planarity_tolerance = planarity
        self.maximum_bernstein_nodes = nodes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "cell-validity-policy",
                "maximum_subdivision_depth": depth,
                "maximum_piece_count": pieces,
                "relative_determinant_floor": floor,
                "relative_planarity_tolerance": planarity,
                "maximum_bernstein_nodes": nodes,
            }
        )


CellValidityUnresolvedReason: TypeAlias = Literal[
    "unsupported_element",
    "subdivision_depth",
    "piece_budget",
    "bernstein_node_budget",
    "rounding_enclosure",
    "undecided_geometry",
]


def cell_geometry_id(geometry: CellGeometrySpec, /) -> str:
    """Identity of the coordinate element layout and the actual coordinate array."""

    from ._cell_geometry import _require_storage_geometry
    from ._coordinate_enclosure import coordinate_source_signature

    storage = geometry.storage
    if storage is not None:
        if (
            geometry.storage_id != storage.storage_id
            or geometry.logical_geometry_id != storage.logical_coordinate_geometry_id
        ):
            raise ValueError(
                "Owner-local coordinate source has a different logical storage identity."
            )
        _require_storage_geometry(
            storage,
            geometry.block_names,
            geometry.elements,
            dict(zip(geometry.block_names, geometry.geometry_dofs, strict=True)),
            geometry.coordinates,
            geometry.restriction_source,
            geometry.exact_source,
            geometry.periodic_source,
        )
        return storage.logical_coordinate_geometry_id
    if geometry.storage_id is not None or geometry.logical_geometry_id is not None:
        raise ValueError(
            "Logical coordinate identity requires its checked storage source."
        )
    return canonical_fingerprint(
        {
            "layout": geometry.geometry_layout_id,
            "source_definitions": [
                coordinate_source_signature(element) for element in geometry.elements
            ],
            "source_arrays": array_tree_fingerprint(geometry.elements),
            "exact_source": None
            if geometry.exact_source is None
            else geometry.exact_source.source_id,
            **(
                {}
                if geometry.periodic_source is None
                else {
                    "periodic_coefficient_source": geometry.periodic_source.source_id,
                }
            ),
            "coordinate_routes": array_tree_fingerprint(geometry.geometry_dofs),
            "coordinates": array_tree_fingerprint(
                np.asarray(geometry.coordinates, dtype=np.float64)
            ),
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

    The certificate is bound to the certified coordinate arrays and element
    layout (``geometry_id``) and, when a mesh was supplied, to its topology
    (``topology_id``). ``unresolved_reasons`` lists ``(block, reason)`` for every
    block holding UNRESOLVED cells. Bernstein coefficients are computed exactly;
    published determinant bounds alone carry outward binary64 rounding.
    """

    status: Array
    determinant_lower: Array
    determinant_upper: Array
    depth: Array
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    unsupported_block_names: tuple[str, ...] = eqx.field(static=True)
    unresolved_reasons: tuple[tuple[str, CellValidityUnresolvedReason], ...] = eqx.field(
        static=True
    )
    certified_valid_count: int = eqx.field(static=True)
    invalid_count: int = eqx.field(static=True)
    unresolved_count: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    topology_id: str | None = eqx.field(static=True)
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
        unresolved_reasons: tuple[tuple[str, str], ...],
        geometry_id: str,
        geometry_layout_id: str,
        topology_id: str | None,
        policy_id: str,
    ) -> None:
        status_ = np.asarray(status, dtype=np.int32)
        lower = np.asarray(determinant_lower, dtype=np.float64)
        upper = np.asarray(determinant_upper, dtype=np.float64)
        depth_ = np.asarray(depth, dtype=np.int32)
        names = tuple(str(value) for value in block_names)
        offsets = tuple(int(value) for value in block_offsets)
        unsupported = tuple(str(value) for value in unsupported_block_names)
        reasons = tuple(
            (
                str(block),
                parse(reason, CellValidityUnresolvedReason, "unresolved_reasons"),
            )
            for block, reason in unresolved_reasons
        )
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
        if not set(unsupported) <= set(names) or not {
            block for block, _ in reasons
        } <= set(names):
            raise ValueError("Unresolved validity blocks must be certificate blocks.")
        self.status = jnp.asarray(status_)
        self.determinant_lower = jnp.asarray(lower)
        self.determinant_upper = jnp.asarray(upper)
        self.depth = jnp.asarray(depth_)
        self.block_names = names
        self.block_offsets = offsets
        self.unsupported_block_names = unsupported
        self.unresolved_reasons = reasons
        self.certified_valid_count = int(
            np.count_nonzero(status_ == CellValidityStatus.CERTIFIED_VALID)
        )
        self.invalid_count = int(np.count_nonzero(status_ == CellValidityStatus.INVALID))
        self.unresolved_count = int(
            np.count_nonzero(status_ == CellValidityStatus.UNRESOLVED)
        )
        self.geometry_id = str(geometry_id)
        self.geometry_layout_id = str(geometry_layout_id)
        self.topology_id = None if topology_id is None else str(topology_id)
        self.policy_id = str(policy_id)
        self.certificate_id = canonical_fingerprint(
            {
                "kind": "cell-validity-certificate",
                "geometry": self.geometry_id,
                "geometry_layout": self.geometry_layout_id,
                "topology": self.topology_id,
                "policy": self.policy_id,
                "blocks": names,
                "block_offsets": offsets,
                "unsupported": unsupported,
                "unresolved_reasons": reasons,
                "status": array_tree_fingerprint(status_),
                "depth": array_tree_fingerprint(depth_),
            }
        )

    @property
    def all_certified(self) -> bool:
        return self.certified_valid_count == self.status.shape[0]

    def require_bound(
        self, geometry: CellGeometrySpec, /, *, mesh: CellMesh | None = None
    ) -> None:
        """Refuse use of this certificate for other coordinates, layout or topology."""

        if not isinstance(geometry, CellGeometrySpec):
            raise TypeError("geometry must be CellGeometrySpec.")
        if cell_geometry_id(geometry) != self.geometry_id:
            raise ValueError(
                "Validity certificate is not bound to these coordinate arrays."
            )
        if mesh is not None and self.topology_id != mesh.topology_id:
            raise ValueError("Validity certificate is not bound to this mesh topology.")


_EPSILON = float(np.finfo(np.float64).eps)


# Bernstein parameter domains ------------------------------------------------


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


def _bernstein_node_count(domain: str, degrees: tuple[int, ...], /) -> int:
    """Conservative table size used by plans before exact expressions exist."""
    match domain:
        case "box":
            return math.prod(degree + 1 for degree in degrees)
        case "simplex":
            degree, dimension = degrees
            return math.comb(degree + dimension, dimension)
        case "prism":
            triangle_degree, axial_degree = degrees
            return math.comb(triangle_degree + 2, 2) * (axial_degree + 1)
        case _:
            raise ValueError(f"Unknown Bernstein parameter domain {domain!r}.")


def _determinant_route(
    cell_kind: str, degree: int, embedded: bool, /
) -> tuple[str, tuple[int, ...]]:
    """Return the reference domain and conservative unsimplified degree bound."""
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


# Adaptive certification ----------------------------------------------------------


@dataclass(frozen=True)
class _BlockCertificate:
    status: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    depth: np.ndarray
    reasons: tuple[CellValidityUnresolvedReason, ...] = ()


def _unresolved_block(
    count: int, reason: CellValidityUnresolvedReason, /
) -> _BlockCertificate:
    return _BlockCertificate(
        np.full((count,), CellValidityStatus.UNRESOLVED, np.int32),
        np.full((count,), np.nan),
        np.full((count,), np.nan),
        np.zeros((count,), dtype=np.int32),
        (reason,),
    )


def _unresolved_geometry(block: _BlockCertificate, /) -> _BlockCertificate:
    """Attach the generic reason to star/polygon blocks with undecided cells."""

    if not np.any(block.status == CellValidityStatus.UNRESOLVED):
        return block
    return _BlockCertificate(
        block.status, block.lower, block.upper, block.depth, ("undecided_geometry",)
    )


def _certify_polynomial_block(
    element: Any,
    cell_kind: str,
    local: np.ndarray | tuple[CoordinateSourceBank, ...],
    policy: CellValidityPolicy,
    /,
) -> _BlockCertificate:
    from contextlib import nullcontext

    from ._coordinate_enclosure import (
        _COORDINATE_BUDGET,
        affine_arguments,
        coordinate_expressions,
        coordinate_polynomials,
        Expression,
        expression_bernstein_coefficients as bernstein_coefficients,
        expression_compose as compose,
        expression_determinant,
        expression_evaluate as evaluate,
        expression_multiply,
        expression_node_count,
        expression_physical_jacobian,
        expression_sum,
        outward,
        physical_jacobian,
        RationalEnclosureError,
    )

    dimension = element.topological_dimension
    cell_count = len(local)
    embedded = cell_count > 0 and len(local[0][0]) != dimension
    if cell_kind.startswith(("simplex:", "tensor:")):
        native = {
            ("simplex", 1): "interval",
            ("simplex", 2): "triangle",
            ("simplex", 3): "tetrahedron",
            ("tensor", 1): "interval",
            ("tensor", 2): "quadrilateral",
            ("tensor", 3): "hexahedron",
        }
        cell_kind = native.get((cell_kind.partition(":")[0], dimension), cell_kind)
    if cell_kind not in (
        "interval",
        "triangle",
        "tetrahedron",
        "quadrilateral",
        "hexahedron",
        "prism",
        "pyramid",
    ):
        return _unresolved_block(cell_count, "unsupported_element")
    if embedded and cell_kind not in ("interval", "triangle", "quadrilateral"):
        return _unresolved_block(cell_count, "unsupported_element")
    domain, _ = _determinant_route(cell_kind, element.degree, embedded)
    if domain == "simplex":
        child_origins, child_matrices = _simplex_child_maps(dimension)
        vertices = tuple(
            (Fraction(0),) * dimension
            if i == 0
            else tuple(Fraction(int(j == i - 1)) for j in range(dimension))
            for i in range(dimension + 1)
        )
    elif domain == "prism":
        child_origins, child_matrices = _prism_child_maps()
        vertices = tuple(
            (*point, Fraction(height))
            for point in (
                (Fraction(0), Fraction(0)),
                (Fraction(1), Fraction(0)),
                (Fraction(0), Fraction(1)),
            )
            for height in (0, 1)
        )
    else:
        child_origins, child_matrices = _box_child_maps(dimension)
        vertices = tuple(
            tuple(Fraction(value) for value in point)
            for point in product((0, 1), repeat=dimension)
        )
    status = np.full((cell_count,), CellValidityStatus.CERTIFIED_VALID, dtype=np.int32)
    lower = np.full((cell_count,), np.inf, dtype=np.float64)
    upper = np.full((cell_count,), -np.inf, dtype=np.float64)
    depth = np.zeros((cell_count,), dtype=np.int32)
    reasons: list[CellValidityUnresolvedReason] = []
    ledger = _COORDINATE_BUDGET.get()
    for cell in range(cell_count):
        with ledger.temporary_scope() if ledger is not None else nullcontext():
            coordinates = coordinate_polynomials(element, local[cell])
            polynomial: Expression | None = None
            jacobian: tuple[tuple[Expression, ...], ...] | None = (
                None
                if coordinates is None
                else physical_jacobian(coordinates, cell_kind, dimension)
            )
            if coordinates is None:
                expressions = coordinate_expressions(element, local[cell])
                if expressions is not None:
                    jacobian = expression_physical_jacobian(
                        expressions, cell_kind, dimension
                    )
            if jacobian is not None:
                # The determinant and policy scale consume the same full physical
                # differential, including the exact removable pyramid collapse.
                metric = (
                    tuple(
                        tuple(
                            expression_sum(
                                tuple(
                                    expression_multiply(row[i], row[j])
                                    for row in jacobian
                                )
                            )
                            for j in range(dimension)
                        )
                        for i in range(dimension)
                    )
                    if embedded
                    else jacobian
                )
                polynomial = expression_determinant(metric)
            if polynomial is None or jacobian is None:
                status[cell] = CellValidityStatus.UNRESOLVED
                lower[cell] = upper[cell] = math.nan
                reasons.append("unsupported_element")
                continue
            if (
                max(
                    expression_node_count(value, domain, dimension)
                    for row in jacobian
                    for value in row
                )
                > policy.maximum_bernstein_nodes
                or expression_node_count(polynomial, domain, dimension)
                > policy.maximum_bernstein_nodes
            ):
                status[cell] = CellValidityStatus.UNRESOLVED
                lower[cell] = upper[cell] = math.nan
                reasons.append("bernstein_node_budget")
                continue
            try:
                magnitude = tuple(
                    tuple(
                        max(
                            abs(value)
                            for value in bernstein_coefficients(entry, domain, dimension)
                        )
                        for entry in row
                    )
                    for row in jacobian
                )
            except RationalEnclosureError:
                status[cell] = CellValidityStatus.UNRESOLVED
                lower[cell] = upper[cell] = math.nan
                reasons.append("rounding_enclosure")
                continue
            squared_scale = Fraction(
                math.prod(
                    sum((row[axis] * row[axis] for row in magnitude), Fraction(0))
                    for axis in range(dimension)
                )
            )
            floor = Fraction(policy.relative_determinant_floor)
            squared_threshold = (
                (floor * squared_scale) ** 2
                if embedded
                else floor * floor * squared_scale
            )
            active = [
                (
                    np.zeros((dimension,), dtype=np.float64),
                    np.eye(dimension, dtype=np.float64),
                )
            ]
            for level in range(policy.maximum_subdivision_depth + 1):
                following = []
                for origin, matrix in active:
                    piece = compose(polynomial, affine_arguments(origin, matrix))
                    try:
                        coefficients = bernstein_coefficients(piece, domain, dimension)
                    except RationalEnclosureError:
                        status[cell] = CellValidityStatus.UNRESOLVED
                        reasons.append("rounding_enclosure")
                        lower[cell] = upper[cell] = math.nan
                        following = []
                        break
                    if len(coefficients) > policy.maximum_bernstein_nodes:
                        status[cell] = CellValidityStatus.UNRESOLVED
                        reasons.append("bernstein_node_budget")
                        lower[cell] = upper[cell] = math.nan
                        following = []
                        break
                    lo, hi = min(coefficients), max(coefficients)
                    depth[cell] = level
                    try:
                        corner_values = tuple(
                            evaluate(piece, vertex) for vertex in vertices
                        )
                    except RationalEnclosureError:
                        status[cell] = CellValidityStatus.UNRESOLVED
                        reasons.append("rounding_enclosure")
                        lower[cell] = upper[cell] = math.nan
                        following = []
                        break
                    if any(
                        value <= 0 or value * value < squared_threshold
                        for value in corner_values
                    ):
                        status[cell] = CellValidityStatus.INVALID
                        lower[cell] = min(lower[cell], outward(lo, -math.inf))
                        upper[cell] = max(upper[cell], outward(hi, math.inf))
                        following = []
                        break
                    if lo > 0 and lo * lo > squared_threshold:
                        lower[cell] = min(lower[cell], outward(lo, -math.inf))
                        upper[cell] = max(upper[cell], outward(hi, math.inf))
                        continue
                    if level == policy.maximum_subdivision_depth:
                        status[cell] = CellValidityStatus.UNRESOLVED
                        reasons.append("subdivision_depth")
                        lower[cell] = min(lower[cell], outward(lo, -math.inf))
                        upper[cell] = max(upper[cell], outward(hi, math.inf))
                        continue
                    following.extend(
                        (origin + matrix @ start, matrix @ child)
                        for start, child in zip(
                            child_origins, child_matrices, strict=True
                        )
                    )
                if status[cell] == CellValidityStatus.INVALID or not following:
                    break
                if len(following) > policy.maximum_piece_count:
                    status[cell] = CellValidityStatus.UNRESOLVED
                    reasons.append("piece_budget")
                    root_coefficients = bernstein_coefficients(
                        polynomial, domain, dimension
                    )
                    lower[cell] = outward(min(root_coefficients), -math.inf)
                    upper[cell] = outward(max(root_coefficients), math.inf)
                    break
                active = following
    return _BlockCertificate(status, lower, upper, depth, tuple(reasons))


def _certify_pyramid_restriction(
    element: Any,
    local: np.ndarray | tuple[CoordinateSourceBank, ...],
    policy: CellValidityPolicy,
) -> _BlockCertificate:
    """Use the source pyramid theorem before restricting its rational chart.

    The affine target reference image is convex and must lie in the source
    pyramid, including its removable apex. The derivative is exactly
    ``J_source(A xi + b) A``; a source-wide determinant/derivative enclosure
    therefore encloses the whole restriction without pretending it polynomial.
    """
    from ._coordinate_enclosure import (
        affine_arguments,
        bernstein_coefficients,
        coordinate_polynomials,
        determinant,
        evaluate,
        outward,
        physical_jacobian,
    )
    from ._reference_cell import reference_cell_topology

    source = element.source_element
    dimension = element.topological_dimension
    arguments = affine_arguments(np.asarray(element.offset), np.asarray(element.matrix))
    for vertex in reference_cell_topology(element.cell_kind).vertices:
        point = tuple(Fraction(float(value)) for value in vertex)
        mapped = tuple(evaluate(value, point) for value in arguments)
        x, y, z = mapped
        if not (0 <= z <= 1 and z / 2 <= x <= 1 - z / 2 and z / 2 <= y <= 1 - z / 2):
            return _unresolved_block(len(local), "unsupported_element")
    matrix = tuple(
        tuple({(0,) * dimension: Fraction(float(value))} for value in row)
        for row in np.asarray(element.matrix)
    )
    determinant_a = evaluate(determinant(matrix), (Fraction(0),) * dimension)
    embedded = len(local) > 0 and len(local[0][0]) != dimension
    factor = determinant_a * determinant_a if embedded else determinant_a
    if factor <= 0:
        return _BlockCertificate(
            np.full((len(local),), CellValidityStatus.INVALID, dtype=np.int32),
            np.full((len(local),), -math.inf, dtype=np.float64),
            np.zeros((len(local),), dtype=np.float64),
            np.zeros((len(local),), dtype=np.int32),
        )
    source_policy = CellValidityPolicy(
        maximum_subdivision_depth=policy.maximum_subdivision_depth,
        maximum_piece_count=policy.maximum_piece_count,
        relative_determinant_floor=0.0,
        relative_planarity_tolerance=policy.relative_planarity_tolerance,
        maximum_bernstein_nodes=policy.maximum_bernstein_nodes,
    )
    source_block = _certify_polynomial_block(source, "pyramid", local, source_policy)
    lower = np.asarray(
        [
            outward(Fraction(float(value)) * factor, -math.inf)
            if math.isfinite(float(value))
            else float(value)
            for value in source_block.lower
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            outward(Fraction(float(value)) * factor, math.inf)
            if math.isfinite(float(value))
            else float(value)
            for value in source_block.upper
        ],
        dtype=np.float64,
    )
    status = np.full((len(local),), CellValidityStatus.UNRESOLVED, dtype=np.int32)
    reasons = list(source_block.reasons)
    for cell, points in enumerate(local):
        coordinates = coordinate_polynomials(source, points)
        jacobian = (
            None
            if coordinates is None
            else physical_jacobian(coordinates, "pyramid", dimension)
        )
        if (
            jacobian is None
            or source_block.status[cell] != CellValidityStatus.CERTIFIED_VALID
        ):
            reasons.append("unsupported_element")
            continue
        source_magnitude = tuple(
            tuple(
                max(
                    abs(value)
                    for value in bernstein_coefficients(entry, "box", dimension)
                )
                for entry in row
            )
            for row in jacobian
        )
        matrix_magnitude = tuple(
            tuple(abs(Fraction(float(value))) for value in row)
            for row in np.asarray(element.matrix)
        )
        magnitude = tuple(
            tuple(
                sum(
                    (
                        source_magnitude[i][k] * matrix_magnitude[k][j]
                        for k in range(dimension)
                    ),
                    Fraction(0),
                )
                for j in range(dimension)
            )
            for i in range(len(jacobian))
        )
        squared_scale = Fraction(
            math.prod(
                sum((row[axis] * row[axis] for row in magnitude), Fraction(0))
                for axis in range(dimension)
            )
        )
        floor = Fraction(policy.relative_determinant_floor)
        squared_threshold = (
            (floor * squared_scale) ** 2 if embedded else floor * floor * squared_scale
        )
        if (
            math.isfinite(float(lower[cell]))
            and lower[cell] > 0
            and Fraction(float(lower[cell])) ** 2 > squared_threshold
        ):
            status[cell] = CellValidityStatus.CERTIFIED_VALID
        else:
            reasons.append("rounding_enclosure")
    return _BlockCertificate(status, lower, upper, source_block.depth, tuple(reasons))


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
    from ._cell_geometry import (
        BarycentricCellGeometryElement,
        PolynomialComposedCellGeometryElement,
        RationalComposedCellGeometryElement,
        RestrictedCellGeometryElement,
        SplineCellGeometryElement,
    )
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
    exact_stars = None
    exact_piece_limit = False
    source_values: CoordinateSourceBank = ()
    match spec.exact_source:
        case (
            ExactPowerCellGeometryLinearActionSource()
            | ExactPowerCellGeometryRestrictionSource()
        ) if all(element.cell_kind == "hexahedron" for element in elements):
            source_values = spec.source_coordinates()
        case None | ExactPlcCellGeometrySource() | ExactPlcCellGeometryConvexSource():
            # The owning PLC bank defines the actual polynomial/rational map.
            source_values = spec.source_coordinates()
        case (
            ExactPowerCellGeometrySource()
            | ExactPowerCellGeometryRestrictionSource()
            | ExactPowerCellGeometryLinearActionSource()
        ):
            if mesh_ is None:
                raise ValueError("Exact power validity requires its bound mesh carrier.")
            from ..geometry._exact_polyhedral_geometry import (
                exact_vertices,
                star_tetrahedra,
            )

            connectivity = mesh_.connectivity
            if not isinstance(connectivity, PolyhedralConnectivity):
                raise TypeError(
                    "Exact power validity requires packed polyhedral connectivity."
                )
            face_sizes = np.diff(np.asarray(connectivity.face_vertex_offsets))
            expected_pieces = sum(
                int(face_sizes[face]) - 2
                for face in np.asarray(connectivity.cell_face_values)
            )
            exact_piece_limit = expected_pieces > policy_.maximum_piece_count
            if not exact_piece_limit:
                exact_stars = star_tetrahedra(
                    mesh_, exact_vertices(mesh_, spec), require_positive=False
                )
        case invalid:
            assert_never(invalid)
    values = np.asarray(spec.coordinates, dtype=np.float64)
    tables = None
    polyhedral_points = None
    cursor = 0
    offsets = [0]
    unsupported = []
    blocks = []
    for name, kind, element, route in zip(names, kinds, elements, routes, strict=True):
        cell_count = route.shape[0]
        if exact_piece_limit:
            blocks.append(_unresolved_block(cell_count, "piece_budget"))
        elif exact_stars is not None:
            from ..geometry._exact_polyhedral_geometry import determinant3
            from ._coordinate_enclosure import outward

            lower, upper, statuses = [], [], []
            floor_squared = Fraction(policy_.relative_determinant_floor) ** 2
            for star in exact_stars[cursor : cursor + cell_count]:
                determinants, admitted = [], True
                for tetrahedron in star:
                    columns = tuple(
                        tuple(
                            value - base
                            for value, base in zip(point, tetrahedron[0], strict=True)
                        )
                        for point in tetrahedron[1:]
                    )
                    determinant = determinant3(*columns)
                    squared_scale = math.prod(
                        sum((value * value for value in column), Fraction(0))
                        for column in columns
                    )
                    determinants.append(determinant)
                    admitted &= (
                        determinant > 0 and determinant**2 > floor_squared * squared_scale
                    )
                lower.append(outward(min(determinants), -math.inf))
                upper.append(outward(max(determinants), math.inf))
                statuses.append(
                    CellValidityStatus.CERTIFIED_VALID
                    if admitted
                    else CellValidityStatus.INVALID
                    if sum(determinants) <= 0
                    else CellValidityStatus.UNRESOLVED
                )
            blocks.append(
                _unresolved_geometry(
                    _BlockCertificate(
                        np.asarray(statuses, dtype=np.int32),
                        np.asarray(lower, dtype=np.float64),
                        np.asarray(upper, dtype=np.float64),
                        np.zeros(cell_count, dtype=np.int32),
                    )
                )
            )
        elif isinstance(element, CellVertexGeometryElement) and kind == "polygon":
            blocks.append(
                _unresolved_geometry(_certify_polygon_block(values[route], policy_))
            )
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
                _unresolved_geometry(
                    _certify_polyhedral_cells(
                        tables,
                        polyhedral_points,
                        np.arange(cursor, cursor + cell_count),
                        policy_,
                    )
                )
            )
        elif (
            isinstance(element, RestrictedCellGeometryElement)
            and element.source_element.cell_kind == "pyramid"
        ):
            local = tuple(tuple(source_values[index] for index in row) for row in route)
            blocks.append(_certify_pyramid_restriction(element, local, policy_))
        elif (
            isinstance(
                element,
                (
                    FiniteElementSpec,
                    BarycentricCellGeometryElement,
                    RestrictedCellGeometryElement,
                    PolynomialComposedCellGeometryElement,
                    RationalComposedCellGeometryElement,
                    SplineCellGeometryElement,
                    LayerColumnCellGeometryElement,
                ),
            )
            and element.conformity == "H1"
            and element.degree >= 1
        ):
            local = tuple(tuple(source_values[index] for index in row) for row in route)
            blocks.append(_certify_polynomial_block(element, kind, local, policy_))
        else:
            unsupported.append(name)
            blocks.append(_unresolved_block(cell_count, "unsupported_element"))
        cursor += cell_count
        offsets.append(cursor)
    return CellValidityCertificate(
        np.concatenate([block.status for block in blocks])
        if blocks
        else np.empty((0,), dtype=np.int32),
        np.concatenate([block.lower for block in blocks])
        if blocks
        else np.empty((0,), dtype=np.float64),
        np.concatenate([block.upper for block in blocks])
        if blocks
        else np.empty((0,), dtype=np.float64),
        np.concatenate([block.depth for block in blocks])
        if blocks
        else np.empty((0,), dtype=np.int32),
        block_names=names,
        block_offsets=tuple(offsets),
        unsupported_block_names=tuple(unsupported),
        unresolved_reasons=tuple(
            (name, reason)
            for name, block in zip(names, blocks, strict=True)
            for reason in dict.fromkeys(block.reasons)
        ),
        geometry_id=cell_geometry_id(spec),
        geometry_layout_id=spec.geometry_layout_id,
        topology_id=None if mesh_ is None else mesh_.topology_id,
        policy_id=policy_.policy_id,
    )


__all__ = [
    "CellValidityCertificate",
    "CellValidityPolicy",
    "CellValidityStatus",
    "certify_cell_geometry_validity",
]
