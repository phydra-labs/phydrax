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
the Gram determinant. Admission uses the simplified coordinate
expressions: their Bernstein tables must fit
``CellValidityPolicy.maximum_bernstein_nodes`` or the block remains UNRESOLVED
with reason ``bernstein_node_budget``.

Source rounding enclosure. Coordinate basis expressions retain the coefficient
field declared by their source: rational binary64 lifts, rational source points,
or multiquadratic field coefficients. Their determinant is formed by polynomial
algebra, including cancellation of the pyramid collapse factor.
Power-to-Bernstein conversion and dyadic subdivision bound the declared map
without sampled interpolation. Published binary64 endpoints round outward.
An arbitrary tabulator claiming a supported family is not a source expression
and remains UNRESOLVED; partition-of-unity identities cannot establish a
pointwise basis error.

Coordinate sources certify the field-valued points returned by
``CellGeometrySpec.source_coordinates()``, not their binary64 representatives.
Rational interpolation actions retain their parent's coefficient field. The same
polynomial route, chart, and relative floor apply, so a positively oriented
carrier cannot certify a source map with a nonpositive determinant.
Power sources certify their source star decomposition.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from itertools import product
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.geometry._rational import fraction_endpoint

from .. import _geometry_predicates as geometry_predicates_owner
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
from ..geometry import _exact_polyhedral_geometry as exact_polyhedral_geometry_owner
from ..linalg._small_batched import cofactor_determinant_2, cofactor_determinant_3
from ..typing import checked, parse
from . import (
    _cell_geometry as cell_geometry_owner,
    _cell_mesh as cell_mesh_owner,
    _coordinate_enclosure as coordinate_enclosure_owner,
    _exact_algebraic_geometry as exact_algebraic_geometry_owner,
    _exact_plc_geometry as exact_plc_geometry_owner,
    _exact_power_geometry as exact_power_geometry_owner,
    _exact_rational_geometry as exact_rational_geometry_owner,
    _reference_cell as reference_cell_owner,
)
from ._cell_complex import PolyhedralConnectivity
from ._cell_geometry_predecessor import CellGeometryPredecessor
from ._coordinate_budget import CoordinateEnclosureBudget
from ._exact_algebraic_geometry import coefficient_endpoint
from ._reference_subdivision import box_child_maps, prism_child_maps, simplex_child_maps


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
            raise ValueError(
                "CellValidityPolicy.inputs must be non-negative; got incompatible inputs."
            )
        if pieces <= 0:
            raise ValueError(
                "CellValidityPolicy.inputs must be positive; got incompatible inputs."
            )
        if not math.isfinite(floor) or floor < 0.0 or floor >= 1.0:
            raise ValueError(
                "CellValidityPolicy.inputs must lie in [0, 1); got incompatible inputs."
            )
        if not math.isfinite(planarity) or planarity < 0.0 or planarity >= 1.0:
            raise ValueError(
                "CellValidityPolicy.inputs must lie in [0, 1); got incompatible inputs."
            )
        if nodes <= 0:
            raise ValueError(
                "CellValidityPolicy.inputs must be positive; got incompatible inputs."
            )
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


@checked
def cell_geometry_id(geometry: cell_geometry_owner.CellGeometrySpec, /) -> str:
    """Identity of the coordinate element layout and the coordinate array."""

    storage = geometry.storage
    if storage is not None:
        if (
            geometry.storage_id != storage.storage_id
            or geometry.logical_geometry_id != storage.logical_coordinate_geometry_id
        ):
            raise ValueError(
                "cell_geometry_id.inputs must satisfy the condition: Owner-local coordinate source has a different logical storage identity; got incompatible inputs."
            )
        cell_geometry_owner.require_storage_geometry(
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
            "cell_geometry_id.inputs must provide its checked storage source; got incompatible inputs."
        )
    return canonical_fingerprint(
        {
            "layout": geometry.geometry_layout_id,
            "source_definitions": [
                coordinate_enclosure_owner.coordinate_source_signature(element)
                for element in geometry.elements
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
        status: ArrayLike,
        determinant_lower: ArrayLike,
        determinant_upper: ArrayLike,
        depth: ArrayLike,
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
            raise ValueError(
                "CellValidityCertificate.inputs must be aligned vectors; got incompatible inputs."
            )
        if not np.all(
            np.isin(status_, tuple(int(value) for value in CellValidityStatus))
        ):
            raise ValueError(
                "CellValidityCertificate.inputs must be CellValidityStatus codes; got incompatible inputs."
            )
        if (
            len(offsets) != len(names) + 1
            or offsets[0] != 0
            or offsets[-1] != status_.size
            or any(stop < start for start, stop in zip(offsets[:-1], offsets[1:]))
        ):
            raise ValueError(
                "CellValidityCertificate.inputs must partition the cells; got incompatible inputs."
            )
        if not set(unsupported) <= set(names) or not {
            block for block, _ in reasons
        } <= set(names):
            raise ValueError(
                "CellValidityCertificate.inputs must be certificate blocks; got incompatible inputs."
            )
        self.status, self.determinant_lower, self.determinant_upper, self.depth = (
            jax.device_put((status_, lower, upper, depth_))
        )
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

    @checked
    def require_bound(
        self,
        geometry: cell_geometry_owner.CellGeometrySpec,
        /,
        *,
        mesh: cell_mesh_owner.CellMesh | None = None,
    ) -> None:
        """Refuse use of this certificate for other coordinates, layout or topology."""

        if cell_geometry_id(geometry) != self.geometry_id:
            raise ValueError(
                "CellValidityCertificate.geometry_id must match the coordinate arrays; got another geometry ID."
            )
        if mesh is not None and self.topology_id != mesh.topology_id:
            raise ValueError(
                "CellValidityCertificate.topology_id must match the mesh topology; got another topology ID."
            )


_EPSILON = float(np.finfo(np.float64).eps)


# Bernstein parameter domains ------------------------------------------------


def bernstein_node_count(domain: str, degrees: tuple[int, ...], /) -> int:
    """Conservative table size used by plans before expressions exist."""
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
            raise ValueError(
                f"bernstein_node_count.inputs must satisfy the condition: Unknown Bernstein parameter domain{domain!r}.; got incompatible inputs."
            )


def determinant_route(
    cell_kind: str, degree: int, embedded: bool, /
) -> tuple[coordinate_enclosure_owner.BernsteinDomain, tuple[int, ...]]:
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
            raise ValueError(
                f"determinant_route.inputs must satisfy the condition: No polynomial determinant route for{cell_kind!r}.; got incompatible inputs."
            )


# Determinants ------------------------------------------------------------------


def _square_determinant(matrix: np.ndarray, /) -> np.ndarray:
    size = matrix.shape[-1]
    if size == 1:
        return np.asarray(matrix[..., 0, 0])
    if size == 2:
        return np.asarray(cofactor_determinant_2(matrix[..., 0, :], matrix[..., 1, :]))
    return np.asarray(
        cofactor_determinant_3(matrix[..., 0, :], matrix[..., 1, :], matrix[..., 2, :])
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


class _PolynomialDifferential(StrictModule):
    polynomial: coordinate_enclosure_owner.Expression | None
    jacobian: tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...] | None


def _prepare_polynomial_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    local: np.ndarray | coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> _PolynomialDifferential:
    coefficients, authority = coordinate_enclosure_owner.coordinate_preparation_key(
        element, local, budget=budget
    )
    key = (authority, cell_kind, dimension, embedded)
    earned = budget.polynomial_differential_cache.get(key)
    if earned is not None:
        budget.reserve(0)
        return earned
    coordinates = coordinate_enclosure_owner.coordinate_polynomials(
        element, coefficients, budget=budget
    )
    polynomial: coordinate_enclosure_owner.Expression | None = None
    jacobian: tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...] | None = (
        None
        if coordinates is None
        else coordinate_enclosure_owner.physical_jacobian(
            coordinates, cell_kind, dimension, budget=budget
        )
    )
    if coordinates is None:
        expressions = coordinate_enclosure_owner.coordinate_expressions(
            element, coefficients, budget=budget
        )
        if expressions is not None:
            jacobian = coordinate_enclosure_owner.expression_physical_jacobian(
                expressions,
                cell_kind,
                dimension,
                budget=budget,
            )
    if jacobian is not None:
        # The determinant and policy scale consume the same full physical
        # differential, including the removable pyramid collapse.
        polynomial = (
            coordinate_enclosure_owner.expression_gram_determinant(
                jacobian, budget=budget
            )
            if embedded
            else coordinate_enclosure_owner.expression_determinant(
                jacobian, budget=budget
            )
        )
    result = _PolynomialDifferential(polynomial, jacobian)
    if polynomial is not None and jacobian is not None:
        budget.retain_basis((key, result, polynomial, jacobian))
        budget.polynomial_differential_cache[key] = result
    return result


class _SubdivisionResult(StrictModule):
    status: int
    lower: float
    upper: float
    depth: int
    reasons: tuple[CellValidityUnresolvedReason, ...]


def _expression_magnitudes(
    jacobian: tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...],
    domain: coordinate_enclosure_owner.BernsteinDomain,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...] | None:
    rows: list[tuple[coordinate_enclosure_owner.ExactScalar, ...]] = []
    for row in jacobian:
        magnitudes: list[coordinate_enclosure_owner.ExactScalar] = []
        for entry in row:
            controls = (
                coordinate_enclosure_owner.expression_bernstein_coefficients_or_none(
                    entry, domain, dimension, budget=budget
                )
            )
            if controls is None:
                return None
            magnitudes.append(max(abs(value) for value in controls))
        rows.append(tuple(magnitudes))
    return tuple(rows)


def _expression_corner_values(
    piece: coordinate_enclosure_owner.Expression,
    vertices: tuple[tuple[Fraction, ...], ...],
    budget: CoordinateEnclosureBudget,
) -> tuple[coordinate_enclosure_owner.ExactScalar, ...] | None:
    values: list[coordinate_enclosure_owner.ExactScalar] = []
    for vertex in vertices:
        value = coordinate_enclosure_owner.expression_evaluate_or_none(
            piece, vertex, budget=budget
        )
        if value is None:
            return None
        values.append(value)
    return tuple(values)


class _PolynomialPiece(StrictModule):
    status: CellValidityStatus | None
    lower: float
    upper: float
    evaluated: bool
    terminal: bool
    subdivide: bool
    reason: CellValidityUnresolvedReason | None


def _certify_polynomial_piece(
    piece: coordinate_enclosure_owner.Expression,
    domain: coordinate_enclosure_owner.BernsteinDomain,
    dimension: int,
    vertices: tuple[tuple[Fraction, ...], ...],
    squared_threshold: coordinate_enclosure_owner.ExactScalar,
    level: int,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _PolynomialPiece:
    coefficients = coordinate_enclosure_owner.expression_bernstein_coefficients_or_none(
        piece, domain, dimension, budget=budget
    )
    if coefficients is None:
        return _PolynomialPiece(
            CellValidityStatus.UNRESOLVED,
            math.nan,
            math.nan,
            False,
            True,
            False,
            "rounding_enclosure",
        )
    if len(coefficients) > policy.maximum_bernstein_nodes:
        return _PolynomialPiece(
            CellValidityStatus.UNRESOLVED,
            math.nan,
            math.nan,
            False,
            True,
            False,
            "bernstein_node_budget",
        )
    lo, hi = min(coefficients), max(coefficients)
    corner_values = _expression_corner_values(piece, vertices, budget)
    if corner_values is None:
        return _PolynomialPiece(
            CellValidityStatus.UNRESOLVED,
            math.nan,
            math.nan,
            True,
            True,
            False,
            "rounding_enclosure",
        )
    if any(value <= 0 or value * value < squared_threshold for value in corner_values):
        return _PolynomialPiece(
            CellValidityStatus.INVALID,
            coefficient_endpoint(lo, upper=False, saturate=True),
            coefficient_endpoint(hi, upper=True, saturate=True),
            True,
            True,
            False,
            None,
        )
    if lo > 0 and lo * lo > squared_threshold:
        return _PolynomialPiece(
            CellValidityStatus.CERTIFIED_VALID,
            coefficient_endpoint(lo, upper=False, saturate=True),
            coefficient_endpoint(hi, upper=True, saturate=True),
            True,
            False,
            False,
            None,
        )
    if level == policy.maximum_subdivision_depth:
        return _PolynomialPiece(
            CellValidityStatus.UNRESOLVED,
            coefficient_endpoint(lo, upper=False, saturate=True),
            coefficient_endpoint(hi, upper=True, saturate=True),
            True,
            False,
            False,
            "subdivision_depth",
        )
    return _PolynomialPiece(None, math.inf, -math.inf, True, False, True, None)


def _certify_polynomial_subdivision(
    polynomial: coordinate_enclosure_owner.Expression,
    domain: coordinate_enclosure_owner.BernsteinDomain,
    dimension: int,
    vertices: tuple[tuple[Fraction, ...], ...],
    child_origins: np.ndarray,
    child_matrices: np.ndarray,
    squared_threshold: coordinate_enclosure_owner.ExactScalar,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _SubdivisionResult:
    status: int = CellValidityStatus.CERTIFIED_VALID
    lower, upper, depth = math.inf, -math.inf, 0
    reasons: list[CellValidityUnresolvedReason] = []
    active = [
        (
            np.zeros((dimension,), dtype=np.float64),
            np.eye(dimension, dtype=np.float64),
        )
    ]
    for level in range(policy.maximum_subdivision_depth + 1):
        following = []
        for origin, matrix in active:
            piece = coordinate_enclosure_owner.expression_compose(
                polynomial,
                coordinate_enclosure_owner.affine_arguments(
                    origin, matrix, budget=budget
                ),
                budget=budget,
            )
            result = _certify_polynomial_piece(
                piece,
                domain,
                dimension,
                vertices,
                squared_threshold,
                level,
                policy,
                budget,
            )
            if result.evaluated:
                depth = level
            if (
                result.status is not None
                and result.status != CellValidityStatus.CERTIFIED_VALID
            ):
                status = result.status
            if result.reason is not None:
                reasons.append(result.reason)
            if result.terminal and result.status == CellValidityStatus.UNRESOLVED:
                lower = upper = math.nan
            else:
                lower = min(lower, result.lower)
                upper = max(upper, result.upper)
            if result.terminal:
                following = []
                break
            if not result.subdivide:
                continue
            following.extend(
                (origin + matrix @ start, matrix @ child)
                for start, child in zip(child_origins, child_matrices, strict=True)
            )
        if status == CellValidityStatus.INVALID or not following:
            break
        if len(following) > policy.maximum_piece_count:
            status = CellValidityStatus.UNRESOLVED
            reasons.append("piece_budget")
            root_coefficients = (
                coordinate_enclosure_owner.expression_bernstein_coefficients(
                    polynomial,
                    domain,
                    dimension,
                    budget=budget,
                )
            )
            lower = coefficient_endpoint(
                min(root_coefficients), upper=False, saturate=True
            )
            upper = coefficient_endpoint(
                max(root_coefficients), upper=True, saturate=True
            )
            break
        active = following
    return _SubdivisionResult(status, lower, upper, depth, tuple(reasons))


class _PolynomialReferenceDomain(StrictModule):
    cell_kind: str
    domain: coordinate_enclosure_owner.BernsteinDomain
    child_origins: np.ndarray
    child_matrices: np.ndarray
    vertices: tuple[tuple[Fraction, ...], ...]


def _prepare_polynomial_reference_domain(
    cell_kind: str,
    dimension: int,
    degree: int,
    embedded: bool,
) -> _PolynomialReferenceDomain | None:
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
        return None
    if embedded and cell_kind not in ("interval", "triangle", "quadrilateral"):
        return None
    domain, _ = determinant_route(cell_kind, degree, embedded)
    match domain:
        case "simplex":
            child_origins, child_matrices = simplex_child_maps(dimension)
            vertices = tuple(
                (Fraction(0),) * dimension
                if i == 0
                else tuple(Fraction(int(j == i - 1)) for j in range(dimension))
                for i in range(dimension + 1)
            )
        case "prism":
            child_origins, child_matrices = prism_child_maps()
            vertices = tuple(
                (*point, Fraction(height))
                for point in (
                    (Fraction(0), Fraction(0)),
                    (Fraction(1), Fraction(0)),
                    (Fraction(0), Fraction(1)),
                )
                for height in (0, 1)
            )
        case "box":
            child_origins, child_matrices = box_child_maps(dimension)
            vertices = tuple(
                tuple(Fraction(value) for value in point)
                for point in product((0, 1), repeat=dimension)
            )
        case invalid:
            assert_never(invalid)
    return _PolynomialReferenceDomain(
        cell_kind, domain, child_origins, child_matrices, vertices
    )


def _certify_polynomial_block(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    local: np.ndarray | tuple[coordinate_enclosure_owner.CoordinateSourceBank, ...],
    policy: CellValidityPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _BlockCertificate:

    dimension = element.topological_dimension
    cell_count = len(local)
    embedded = cell_count > 0 and len(local[0][0]) != dimension
    reference = _prepare_polynomial_reference_domain(
        cell_kind, dimension, element.degree, embedded
    )
    if reference is None:
        return _unresolved_block(cell_count, "unsupported_element")
    cell_kind, domain = reference.cell_kind, reference.domain
    child_origins, child_matrices, vertices = (
        reference.child_origins,
        reference.child_matrices,
        reference.vertices,
    )
    status = np.full((cell_count,), CellValidityStatus.CERTIFIED_VALID, dtype=np.int32)
    lower = np.full((cell_count,), np.inf, dtype=np.float64)
    upper = np.full((cell_count,), -np.inf, dtype=np.float64)
    depth = np.zeros((cell_count,), dtype=np.int32)
    reasons: list[CellValidityUnresolvedReason] = []
    ledger = budget
    for cell in range(cell_count):
        with ledger.temporary_scope():
            differential = _prepare_polynomial_differential(
                element,
                local[cell],
                cell_kind,
                dimension,
                embedded,
                budget,
            )
            polynomial, jacobian = differential.polynomial, differential.jacobian
            del differential
            if polynomial is None or jacobian is None:
                status[cell] = CellValidityStatus.UNRESOLVED
                lower[cell] = upper[cell] = math.nan
                reasons.append("unsupported_element")
                continue
            if (
                max(
                    coordinate_enclosure_owner.expression_node_count(
                        value, domain, dimension
                    )
                    for row in jacobian
                    for value in row
                )
                > policy.maximum_bernstein_nodes
                or coordinate_enclosure_owner.expression_node_count(
                    polynomial, domain, dimension
                )
                > policy.maximum_bernstein_nodes
            ):
                status[cell] = CellValidityStatus.UNRESOLVED
                lower[cell] = upper[cell] = math.nan
                reasons.append("bernstein_node_budget")
                continue
            magnitude = _expression_magnitudes(jacobian, domain, dimension, budget)
            if magnitude is None:
                status[cell] = CellValidityStatus.UNRESOLVED
                lower[cell] = upper[cell] = math.nan
                reasons.append("rounding_enclosure")
                continue
            squared_scale = math.prod(
                sum((row[axis] * row[axis] for row in magnitude), Fraction(0))
                for axis in range(dimension)
            )
            floor = Fraction(policy.relative_determinant_floor)
            squared_threshold = (
                (floor * squared_scale) ** 2
                if embedded
                else floor * floor * squared_scale
            )
            result = _certify_polynomial_subdivision(
                polynomial,
                domain,
                dimension,
                vertices,
                child_origins,
                child_matrices,
                squared_threshold,
                policy,
                budget,
            )
            status[cell], lower[cell], upper[cell], depth[cell] = (
                result.status,
                result.lower,
                result.upper,
                result.depth,
            )
            reasons.extend(result.reasons)
    return _BlockCertificate(status, lower, upper, depth, tuple(reasons))


def _pyramid_restriction_threshold(
    element: cell_geometry_owner.RestrictedCellGeometryElement,
    jacobian: tuple[tuple[coordinate_enclosure_owner.Polynomial, ...], ...],
    dimension: int,
    embedded: bool,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> coordinate_enclosure_owner.ExactScalar:
    source_magnitude = tuple(
        tuple(
            max(
                abs(value)
                for value in coordinate_enclosure_owner.bernstein_coefficients(
                    entry, "box", dimension, budget=budget
                )
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
    squared_scale = math.prod(
        sum((row[axis] * row[axis] for row in magnitude), Fraction(0))
        for axis in range(dimension)
    )
    floor = Fraction(policy.relative_determinant_floor)
    squared_threshold = (
        (floor * squared_scale) ** 2 if embedded else floor * floor * squared_scale
    )
    return squared_threshold


def _certify_pyramid_restriction(
    element: cell_geometry_owner.RestrictedCellGeometryElement,
    local: np.ndarray | tuple[coordinate_enclosure_owner.CoordinateSourceBank, ...],
    policy: CellValidityPolicy,
    *,
    budget: CoordinateEnclosureBudget,
) -> _BlockCertificate:
    """Use the source pyramid before restricting its rational chart.

    The affine target reference image is convex and must lie in the source
    pyramid, including its removable apex. The derivative is exactly
    ``J_source(A xi + b) A``; a source-wide determinant/derivative enclosure
    therefore encloses the whole restriction without pretending it polynomial.
    """

    source = element.source_element
    dimension = element.topological_dimension
    arguments = coordinate_enclosure_owner.affine_arguments(
        np.asarray(element.offset), np.asarray(element.matrix), budget=budget
    )
    for vertex in reference_cell_owner.reference_cell_topology(
        element.cell_kind
    ).vertices:
        point = tuple(Fraction(float(value)) for value in vertex)
        mapped = tuple(
            coordinate_enclosure_owner.evaluate(value, point, budget=budget)
            for value in arguments
        )
        x, y, z = mapped
        if not (0 <= z <= 1 and z / 2 <= x <= 1 - z / 2 and z / 2 <= y <= 1 - z / 2):
            return _unresolved_block(len(local), "unsupported_element")
    matrix: tuple[tuple[coordinate_enclosure_owner.Polynomial, ...], ...] = tuple(
        tuple(
            coordinate_enclosure_owner.constant(
                Fraction(float(value)), dimension, budget=budget
            )
            for value in row
        )
        for row in np.asarray(element.matrix)
    )
    determinant_a = coordinate_enclosure_owner.evaluate(
        coordinate_enclosure_owner.determinant(matrix, budget=budget),
        (Fraction(0),) * dimension,
        budget=budget,
    )
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
    source_block = _certify_polynomial_block(
        source, "pyramid", local, source_policy, budget=budget
    )
    lower = np.asarray(
        [
            coefficient_endpoint(
                Fraction(float(value)) * factor, upper=False, saturate=True
            )
            if math.isfinite(float(value))
            else float(value)
            for value in source_block.lower
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            coefficient_endpoint(
                Fraction(float(value)) * factor, upper=True, saturate=True
            )
            if math.isfinite(float(value))
            else float(value)
            for value in source_block.upper
        ],
        dtype=np.float64,
    )
    status = np.full((len(local),), CellValidityStatus.UNRESOLVED, dtype=np.int32)
    reasons = list(source_block.reasons)
    for cell, points in enumerate(local):
        coordinates = coordinate_enclosure_owner.coordinate_polynomials(
            source, points, budget=budget
        )
        jacobian = (
            None
            if coordinates is None
            else coordinate_enclosure_owner.physical_jacobian(
                coordinates, "pyramid", dimension, budget=budget
            )
        )
        if (
            jacobian is None
            or source_block.status[cell] != CellValidityStatus.CERTIFIED_VALID
        ):
            reasons.append("unsupported_element")
            continue
        squared_threshold = _pyramid_restriction_threshold(
            element, jacobian, dimension, embedded, policy, budget
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


@checked
def polyhedral_star_tables(
    connectivity: PolyhedralConnectivity, /
) -> PolyhedralStarTables:
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
            "polyhedral_star_tables.inputs must be closed, consistently oriented surfaces; got incompatible inputs."
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
    """Classify cells from star-simplex determinants.

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


def _polygon_measure(
    points: np.ndarray, policy: CellValidityPolicy, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Determinant enclosure, floor, planarity flags, and planar loops of polygons.

    The determinant is twice the signed shoelace area about the vertex centroid
    (planar) or the squared Newell vector area (embedded, Gram convention).
    Embedded loops are measured against their Newell plane through the vertex
    centroid and projected by dropping the dominant normal axis, which keeps the
    projected binary64 components unchanged. Returns ``(determinant, margin, threshold,
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


def _certify_polygon_block(
    points: np.ndarray,
    policy: CellValidityPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _BlockCertificate:
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
    order = np.lexsort(np.moveaxis(safe, 2, 0), axis=-1)
    ordered = np.take_along_axis(safe, order[..., None], axis=1)
    repeated = np.any(np.all(ordered[:, 1:] == ordered[:, :-1], axis=-1), axis=1)
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
        result = polygon_simplicity_2d(planar[candidates], mode=mode, budget=budget)
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
    mesh: cell_mesh_owner.CellMesh, routes: tuple[np.ndarray, ...], values: np.ndarray, /
) -> np.ndarray:
    """Gather vertex-geometry rows of polyhedral blocks into mesh-vertex order."""

    points = np.full((mesh.coordinates.shape[0], values.shape[1]), np.nan)
    for block, route in zip(mesh.blocks, routes, strict=True):
        if block.cell_kind == "polyhedron":
            vertices = np.asarray(block.vertices, dtype=np.int64)
            valid = np.asarray(block.vertex_valid, dtype=np.bool_)
            points[vertices[valid]] = values[route[valid]]
    return points


class _RoutePreparationBounds(StrictModule):
    work: int
    memory: int


def _polynomial_route_preparation_bounds(
    element: reference_cell_types_owner.CellGeometryElement,
    local: np.ndarray,
    policy: CellValidityPolicy,
    coefficient_bits: int | None,
) -> _RoutePreparationBounds:
    work, memory = 0, 0
    coordinates = coordinate_enclosure_owner.coordinate_map_expression_estimates(
        element, local, coefficient_bits_upper=coefficient_bits
    )
    if coordinates is None:
        # The source owner already includes cache/bank normalization;
        # unsupported sources keep their existing unresolved decision.
        return _RoutePreparationBounds(0, 0)
    dimension = coordinates[0].dimension
    jacobian = coordinate_enclosure_owner.coordinate_physical_jacobian_estimates(
        coordinates, element.cell_kind, dimension
    )
    if len(coordinates) == dimension:
        metric = jacobian
    else:
        entries = []
        for first in range(dimension):
            row = []
            for second in range(dimension):
                products = tuple(
                    coordinate_enclosure_owner.coordinate_estimate_multiply(
                        values[first], values[second]
                    )
                    for values in jacobian
                )
                total = products[0]
                for product in products[1:]:
                    total = coordinate_enclosure_owner.coordinate_estimate_add(
                        total, product
                    )
                row.append(total)
            entries.append(tuple(row))
        metric = tuple(entries)
    determinant = coordinate_enclosure_owner.coordinate_determinant_estimate(
        metric, variable_dimension=dimension
    )
    domain = (
        "simplex"
        if element.cell_kind in ("interval", "triangle", "tetrahedron")
        or element.cell_kind.startswith("simplex:")
        else "prism"
        if element.cell_kind == "prism"
        else "box"
    )
    magnitudes = tuple(
        coordinate_enclosure_owner.coordinate_bernstein_estimate(entry, domain, dimension)
        for row in jacobian
        for entry in row
    )
    magnitude = coordinate_enclosure_owner.coordinate_estimate_bank(magnitudes)
    work += magnitude.work_upper + determinant.work_upper
    memory += magnitude.bytes_upper + determinant.bytes_upper
    # Absolute child frames are binary64 dyadics. This source circuit
    # covers every offset and matrix entry, including the identity
    # frame; it does not infer a polynomial source from map degree.
    coefficient = coordinate_enclosure_owner.coordinate_polynomial_estimate(
        terms=1,
        dimension=dimension,
        degree=0,
        numerator_bits=1075,
        denominator_bits=1075,
    )
    variable = coordinate_enclosure_owner.coordinate_polynomial_estimate(
        terms=1,
        dimension=dimension,
        degree=1,
        numerator_bits=1,
        denominator_bits=1,
    )
    argument = coefficient
    for _ in range(dimension):
        argument = coordinate_enclosure_owner.coordinate_estimate_add(
            argument,
            coordinate_enclosure_owner.coordinate_estimate_multiply(
                variable, coefficient
            ),
        )
    arguments = (argument,) * dimension
    piece = coordinate_enclosure_owner.coordinate_estimate_compose(determinant, arguments)
    controls = coordinate_enclosure_owner.coordinate_bernstein_estimate(
        piece, domain, dimension
    )
    corners = 1 << dimension
    evaluation = corners * (
        piece.terms + (0 if piece.denominator is None else piece.denominator.terms)
    )
    pieces = policy.maximum_piece_count * (policy.maximum_subdivision_depth + 1)
    work += pieces * (controls.work_upper + evaluation)
    memory += pieces * (controls.bytes_upper + piece.bytes_upper)
    return _RoutePreparationBounds(work, memory)


def _polynomial_validity_preparation_bounds(
    geometry: cell_geometry_owner.CellGeometrySpec, policy: CellValidityPolicy, /
) -> tuple[int, int]:
    """Bound supported validity circuits and all policy-admitted pieces."""

    host_coordinates, host_routes = jax.device_get(
        (geometry.coordinates, geometry.geometry_dofs)
    )
    values = np.asarray(host_coordinates, dtype=np.float64)
    source = geometry.exact_source
    coefficient_bits = None if source is None else max(source.maximum_bits, 1075)
    if geometry.periodic_source is not None:
        _, _, coefficient_bits = geometry.periodic_source.preparation_bounds()
    work, memory = cell_geometry_owner.cell_geometry_preparation_bounds(geometry)
    source_work_upper, source_memory_upper = work, memory
    estimates: dict[tuple[str, tuple[int, ...], int], _RoutePreparationBounds] = {}
    for element, routes in zip(geometry.elements, host_routes, strict=True):
        if isinstance(element, cell_geometry_owner.CellVertexGeometryElement):
            continue

        signature = canonical_fingerprint(
            coordinate_enclosure_owner.coordinate_source_signature(element)
        )
        for route in np.asarray(routes):
            local = values[route]
            key = (
                signature,
                local.shape,
                1075 if coefficient_bits is None else coefficient_bits,
            )
            if key not in estimates:
                estimates[key] = _polynomial_route_preparation_bounds(
                    element, local, policy, coefficient_bits
                )
            estimate = estimates[key]
            work += estimate.work
            memory += estimate.memory
    algebraic_work, algebraic_memory = (
        coordinate_enclosure_owner.coordinate_algebraic_preparation_bounds(
            work - source_work_upper,
            memory - source_memory_upper,
            field_degree_upper=cell_geometry_owner.exact_source_field_degree(source),
            polynomial_degree_upper=max(
                (
                    6 * scalar.degree * scalar.topological_dimension
                    for element in geometry.elements
                    if not isinstance(
                        element, cell_geometry_owner.CellVertexGeometryElement
                    )
                    for scalar in (
                        cell_geometry_owner.require_scalar_coordinate_element(
                            element, "validity preparation"
                        ),
                    )
                ),
                default=1,
            ),
        )
    )
    return source_work_upper + algebraic_work, source_memory_upper + algebraic_memory


def _exact_star_validity_preparation_bounds(
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    policy: CellValidityPolicy,
    /,
) -> tuple[int, int]:
    """Bound the admitted power star's face proofs and determinants."""

    connectivity = mesh.connectivity
    if not isinstance(connectivity, PolyhedralConnectivity):
        return 0, 0
    source = geometry.exact_source
    if source is None or isinstance(
        source,
        (
            exact_plc_geometry_owner.ExactPlcCellGeometrySource,
            exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource,
            exact_rational_geometry_owner.ExactRationalCellGeometrySource,
            exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource,
        ),
    ):
        return 0, 0
    offsets = np.asarray(connectivity.face_vertex_offsets)
    occurrences = np.asarray(connectivity.cell_face_values)
    sizes = np.diff(offsets)
    pieces = sum(max(int(sizes[face]) - 2, 0) for face in occurrences)
    if pieces > policy.maximum_piece_count:
        return 0, 0
    bits = max(source.maximum_bits, 1075)
    work, memory = 0, 0
    for face in occurrences:
        count = int(sizes[face])
        face_work, face_memory = exact_polyhedral_geometry_owner.triangulation_bounds(
            count, bits
        )
        # The input-bit scan visits each coordinate once.
        work += face_work + 3 * count
        memory += face_memory
    # require_positive=False leaves determinant decisions with validity itself.
    work += pieces * (4 * 3**3)
    return work, memory


@checked
def cell_validity_preparation_bounds(
    geometry: cell_geometry_owner.CellGeometrySpec,
    policy: CellValidityPolicy,
    /,
    *,
    mesh: cell_mesh_owner.CellMesh | None = None,
) -> tuple[int, int, int]:
    """Bound the source and all policy-admitted validity proof circuits."""

    work, memory = _polynomial_validity_preparation_bounds(geometry, policy)
    # Layout authentication and source-bank preparation may each renew a rational
    # source. The polynomial bound includes one of those preparations.
    source_work, source_memory = cell_geometry_owner.cell_geometry_preparation_bounds(
        geometry
    )
    work += source_work
    memory += source_memory
    queries = 2 * cell_geometry_owner.cell_geometry_preparation_query_upper(geometry)
    if mesh is not None:
        star_work, star_memory = _exact_star_validity_preparation_bounds(
            mesh, geometry, policy
        )
        work += star_work
        memory += star_memory
    for element, routes in zip(geometry.elements, geometry.geometry_dofs, strict=True):
        if (
            isinstance(element, cell_geometry_owner.CellVertexGeometryElement)
            and element.cell_kind == "polygon"
        ):
            polygon_work, polygon_memory, polygon_queries = (
                geometry_predicates_owner.polygon_simplicity_preparation_bounds(
                    routes.shape[0], routes.shape[1]
                )
            )
            work += polygon_work
            memory += polygon_memory
            queries += polygon_queries
    return work, memory, queries


class _ValidityLayout(StrictModule):
    elements: tuple[reference_cell_types_owner.CellGeometryElement, ...]
    routes: tuple[np.ndarray, ...]
    names: tuple[str, ...]
    kinds: tuple[str, ...]


def _prepare_validity_layout(
    spec: cell_geometry_owner.CellGeometrySpec,
    mesh_: cell_mesh_owner.CellMesh | None,
    budget: CoordinateEnclosureBudget,
) -> _ValidityLayout:
    if mesh_ is None:
        elements = spec.elements
        routes = tuple(
            np.asarray(value, dtype=np.int64)
            for value in jax.device_get(spec.geometry_dofs)
        )
        names = spec.block_names
        kinds = tuple(element.cell_kind for element in elements)
    else:
        resolved, resolved_routes, _ = spec.resolve(mesh_, budget=budget)
        elements = resolved
        routes = tuple(
            np.asarray(value, dtype=np.int64) for value in jax.device_get(resolved_routes)
        )
        names = tuple(block.name for block in mesh_.blocks)
        kinds = tuple(block.cell_kind for block in mesh_.blocks)
    return _ValidityLayout(elements, routes, names, kinds)


class _ValiditySources(StrictModule):
    exact_stars: (
        tuple[tuple[exact_polyhedral_geometry_owner.Tetrahedron, ...], ...] | None
    )
    exact_piece_limit: bool
    source_values: coordinate_enclosure_owner.CoordinateSourceBank


def _prepare_validity_sources(
    spec: cell_geometry_owner.CellGeometrySpec,
    mesh_: cell_mesh_owner.CellMesh | None,
    elements: tuple[reference_cell_types_owner.CellGeometryElement, ...],
    policy_: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _ValiditySources:
    exact_stars = None
    exact_piece_limit = False
    source_values: coordinate_enclosure_owner.CoordinateSourceBank = ()
    match spec.exact_source:
        case (
            exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
        ) if all(element.cell_kind == "hexahedron" for element in elements):
            source_values = spec.source_coordinates(budget=budget)
        case (
            None
            | exact_plc_geometry_owner.ExactPlcCellGeometrySource()
            | exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource()
            | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
            | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
        ):
            # The PLC bank defines the polynomial/rational map.
            source_values = spec.source_coordinates(budget=budget)
        case (
            exact_power_geometry_owner.ExactPowerCellGeometrySource()
            | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
            | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
        ):
            if mesh_ is None:
                raise ValueError(
                    "prepare_validity_sources.inputs must provide its bound mesh carrier; got incompatible inputs."
                )

            connectivity = mesh_.connectivity
            if not isinstance(connectivity, PolyhedralConnectivity):
                raise TypeError(
                    "prepare_validity_sources.inputs must provide packed polyhedral connectivity; got incompatible inputs."
                )
            face_sizes = np.diff(np.asarray(connectivity.face_vertex_offsets))
            expected_pieces = sum(
                int(face_sizes[face]) - 2
                for face in np.asarray(connectivity.cell_face_values)
            )
            exact_piece_limit = expected_pieces > policy_.maximum_piece_count
            if not exact_piece_limit:
                exact_stars = exact_polyhedral_geometry_owner.star_tetrahedra(
                    mesh_,
                    exact_polyhedral_geometry_owner.exact_vertices(
                        mesh_, spec, budget=budget
                    ),
                    require_positive=False,
                    budget=budget,
                )
        case invalid:
            assert_never(invalid)
    return _ValiditySources(exact_stars, exact_piece_limit, source_values)


def _certify_exact_star_block(
    stars: tuple[tuple[exact_polyhedral_geometry_owner.Tetrahedron, ...], ...],
    policy_: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _BlockCertificate:
    cell_count = len(stars)
    lower, upper, statuses = [], [], []
    floor_squared = Fraction(policy_.relative_determinant_floor) ** 2
    for star in stars:
        determinants, admitted = [], True
        for tetrahedron in star:
            columns = tuple(
                tuple(
                    value - base
                    for value, base in zip(point, tetrahedron[0], strict=True)
                )
                for point in tetrahedron[1:]
            )
            determinant = exact_polyhedral_geometry_owner.determinant3(
                *columns, budget=budget
            )
            squared_scale = math.prod(
                sum((value * value for value in column), Fraction(0))
                for column in columns
            )
            determinants.append(determinant)
            admitted &= determinant > 0 and determinant**2 > floor_squared * squared_scale
        lower.append(fraction_endpoint(min(determinants), upper=False, saturate=True))
        upper.append(fraction_endpoint(max(determinants), upper=True, saturate=True))
        statuses.append(
            CellValidityStatus.CERTIFIED_VALID
            if admitted
            else CellValidityStatus.INVALID
            if sum(determinants) <= 0
            else CellValidityStatus.UNRESOLVED
        )
    return _unresolved_geometry(
        _BlockCertificate(
            np.asarray(statuses, dtype=np.int32),
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
            np.zeros(cell_count, dtype=np.int32),
        )
    )


class _ValidityBlocks(StrictModule):
    blocks: tuple[_BlockCertificate, ...]
    offsets: tuple[int, ...]
    unsupported: tuple[str, ...]


@dataclass(frozen=True)
class _ValidityImport:
    rows: np.ndarray
    status: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    depth: np.ndarray


def _prepare_validity_import(
    predecessor: CellGeometryPredecessor,
    mesh: cell_mesh_owner.CellMesh | None,
    geometry: cell_geometry_owner.CellGeometrySpec,
    policy: CellValidityPolicy,
) -> _ValidityImport:
    """Authenticate the request binding before reading reusable proof payloads."""
    if mesh is None:
        raise ValueError(
            "certify_cell_geometry_validity.predecessor must provide its target mesh; got no mesh."
        )
    predecessor.require_bound(mesh, geometry)
    source = predecessor.source_validity
    if source.policy_id != policy.policy_id:
        raise ValueError(
            "certify_cell_geometry_validity.predecessor.policy_id must match the requested policy; got another policy."
        )
    if not source.all_certified:
        raise ValueError(
            "certify_cell_geometry_validity.predecessor.source_validity must certify every source cell; got uncertified cells."
        )
    status, lower, upper, depth = jax.device_get(
        (source.status, source.determinant_lower, source.determinant_upper, source.depth)
    )
    return _ValidityImport(
        np.asarray(predecessor.source_rows, dtype=np.int64),
        np.asarray(status, dtype=np.int32),
        np.asarray(lower, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
        np.asarray(depth, dtype=np.int32),
    )


def _merge_validity_import(
    block: _BlockCertificate | None,
    rows: np.ndarray,
    imported: _ValidityImport,
) -> _BlockCertificate:
    """Restore the target row order without changing newly certified evidence."""
    if block is None:
        return _BlockCertificate(
            imported.status[rows],
            imported.lower[rows],
            imported.upper[rows],
            imported.depth[rows],
        )
    changed = rows < 0
    reused = ~changed
    status = np.empty(rows.size, dtype=np.int32)
    lower = np.empty(rows.size, dtype=np.float64)
    upper = np.empty(rows.size, dtype=np.float64)
    depth = np.empty(rows.size, dtype=np.int32)
    for target, source, fresh in (
        (status, imported.status, block.status),
        (lower, imported.lower, block.lower),
        (upper, imported.upper, block.upper),
        (depth, imported.depth, block.depth),
    ):
        target[changed] = fresh
        target[reused] = source[rows[reused]]
    return _BlockCertificate(status, lower, upper, depth, block.reasons)


def _certify_validity_blocks(
    layout: _ValidityLayout,
    sources: _ValiditySources,
    spec: cell_geometry_owner.CellGeometrySpec,
    mesh_: cell_mesh_owner.CellMesh | None,
    policy_: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
    imported: _ValidityImport | None = None,
) -> _ValidityBlocks:
    values = np.asarray(spec.coordinates, dtype=np.float64)
    polyhedral: tuple[PolyhedralStarTables, np.ndarray] | None = None
    cursor = 0
    offsets = [0]
    unsupported: list[str] = []
    blocks: list[_BlockCertificate] = []
    for name, kind, element, route in zip(
        layout.names, layout.kinds, layout.elements, layout.routes, strict=True
    ):
        full_cell_count = route.shape[0]
        rows = (
            None
            if imported is None or sources.exact_piece_limit
            else imported.rows[cursor : cursor + full_cell_count]
        )
        changed = None if rows is None else np.flatnonzero(rows < 0)
        if changed is not None:
            route = route[changed]
        cell_count = route.shape[0]
        if full_cell_count > 0 and cell_count == 0:
            if imported is None or rows is None:
                raise RuntimeError(
                    "CellValidityCertificate.predecessor must provide its admitted rows; got missing import correspondence"
                )
            blocks.append(_merge_validity_import(None, rows, imported))
            cursor += full_cell_count
            offsets.append(cursor)
            continue
        if sources.exact_piece_limit:
            blocks.append(_unresolved_block(cell_count, "piece_budget"))
        elif sources.exact_stars is not None:
            blocks.append(
                _certify_exact_star_block(
                    sources.exact_stars[cursor : cursor + full_cell_count]
                    if changed is None
                    else tuple(sources.exact_stars[cursor + row] for row in changed),
                    policy_,
                    budget,
                )
            )
        else:
            match element:
                case cell_geometry_owner.CellVertexGeometryElement() if kind == "polygon":
                    blocks.append(
                        _unresolved_geometry(
                            _certify_polygon_block(values[route], policy_, budget=budget)
                        )
                    )
                case cell_geometry_owner.CellVertexGeometryElement():
                    if mesh_ is None:
                        raise ValueError(
                            "certify_validity_blocks.inputs must provide the mesh; got incompatible inputs."
                        )
                    if polyhedral is None:
                        match mesh_.connectivity:
                            case PolyhedralConnectivity() as connectivity:
                                polyhedral = (
                                    polyhedral_star_tables(connectivity),
                                    _polyhedral_coordinates(mesh_, layout.routes, values),
                                )
                            case _:
                                raise TypeError(
                                    "certify_validity_blocks.inputs must provide polyhedral mesh connectivity; got incompatible inputs."
                                )
                    tables, points = polyhedral
                    blocks.append(
                        _unresolved_geometry(
                            _certify_polyhedral_cells(
                                tables,
                                points,
                                np.arange(cursor, cursor + full_cell_count)
                                if changed is None
                                else cursor + changed,
                                policy_,
                            )
                        )
                    )
                case cell_geometry_owner.RestrictedCellGeometryElement() if (
                    element.source_element.cell_kind == "pyramid"
                ):
                    local = tuple(
                        tuple(sources.source_values[index] for index in row)
                        for row in route
                    )
                    blocks.append(
                        _certify_pyramid_restriction(
                            element, local, policy_, budget=budget
                        )
                    )
                case (
                    reference_types_owner.FiniteElementSpec()
                    | cell_geometry_owner.BarycentricCellGeometryElement()
                    | cell_geometry_owner.RestrictedCellGeometryElement()
                    | cell_geometry_owner.PolynomialComposedCellGeometryElement()
                    | cell_geometry_owner.RationalComposedCellGeometryElement()
                    | cell_geometry_owner.SplineCellGeometryElement()
                    | cell_geometry_owner.LayerColumnCellGeometryElement()
                ) if element.conformity == "H1" and element.degree >= 1:
                    local = tuple(
                        tuple(sources.source_values[index] for index in row)
                        for row in route
                    )
                    blocks.append(
                        _certify_polynomial_block(
                            element, kind, local, policy_, budget=budget
                        )
                    )
                case _:
                    unsupported.append(name)
                    blocks.append(_unresolved_block(cell_count, "unsupported_element"))
        if rows is not None and imported is not None:
            blocks[-1] = _merge_validity_import(blocks[-1], rows, imported)
        cursor += full_cell_count
        offsets.append(cursor)
    return _ValidityBlocks(tuple(blocks), tuple(offsets), tuple(unsupported))


def _validity_request_key(
    geometry: cell_geometry_owner.CellGeometrySpec,
    mesh: cell_mesh_owner.CellMesh | None,
    policy: CellValidityPolicy,
) -> str:
    """Bind reusable proof content to current sources, topology and all policy fields."""
    mesh_arrays = None
    if mesh is not None:
        # Owner-local validity consumes local coordinates, block routes/IDs and
        # connectivity. Storage owns global publication banks, not this proof.
        # Retain the full-content serial key without a placement-dependent gather.
        mesh_inputs = (
            mesh
            if mesh.storage is None or jax.process_count() == 1
            else (
                mesh.coordinates,
                mesh.blocks,
                mesh.connectivity,
                mesh.vertex_global_ids,
            )
        )
        mesh_arrays = array_tree_fingerprint(mesh_inputs)
    return canonical_fingerprint(
        {
            "geometry": cell_geometry_id(geometry),
            "mesh_arrays": mesh_arrays,
            "mesh_blocks": None
            if mesh is None
            else tuple((block.name, block.cell_kind) for block in mesh.blocks),
            "topology": None if mesh is None else mesh.topology_id,
            "policy": (
                policy.maximum_subdivision_depth,
                policy.maximum_piece_count,
                policy.relative_determinant_floor,
                policy.relative_planarity_tolerance,
                policy.maximum_bernstein_nodes,
                policy.policy_id,
            ),
        }
    )


@checked
def certify_cell_geometry_validity(
    geometry: cell_mesh_owner.CellMesh | cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    mesh: cell_mesh_owner.CellMesh | None = None,
    policy: CellValidityPolicy | None = None,
    budget: CoordinateEnclosureBudget | None = None,
    predecessor: CellGeometryPredecessor | None = None,
) -> CellValidityCertificate:
    """Certify determinant positivity of every mapped cell with Bernstein bounds.

    A ``CellMesh`` is certified with its affine vertex geometry. A
    ``CellGeometrySpec`` is certified in mesh block order when ``mesh`` is given
    (required for polyhedral blocks) and in its own sorted block order otherwise.

    A request-owned ``predecessor`` imports certified rows only after bitwise
    target binding and source payload authentication under the same policy.
    """

    policy_ = CellValidityPolicy() if policy is None else policy
    match geometry:
        case cell_mesh_owner.CellMesh():
            if mesh is not None:
                raise ValueError(
                    "certify_cell_geometry_validity.inputs must satisfy the condition: mesh is implied when certifying a CellMesh; got incompatible inputs."
                )
            mesh_ = geometry
            spec = cell_geometry_owner.CellGeometrySpec.affine(geometry, budget=budget)
        case cell_geometry_owner.CellGeometrySpec():
            mesh_ = mesh
            spec = geometry
        case invalid:
            assert_never(invalid)
    imported = (
        None
        if predecessor is None
        else _prepare_validity_import(predecessor, mesh_, spec, policy_)
    )
    reuse = budget is not None
    if budget is None:
        work_upper, bytes_upper, queries_upper = cell_validity_preparation_bounds(
            spec, policy_, mesh=mesh_
        )
        request = cell_geometry_owner.owned_coordinate_preparation(
            work_upper, bytes_upper, maximum_geometry_queries=queries_upper
        )
    else:
        request = nullcontext(budget)
    with request as budget:
        key = _validity_request_key(spec, mesh_, policy_) if reuse else None
        earned = None if key is None else budget.validity_certificate_cache.get(key)
        if earned is not None:
            return earned
        layout = _prepare_validity_layout(spec, mesh_, budget)
        names = layout.names
        sources = _prepare_validity_sources(spec, mesh_, layout.elements, policy_, budget)
        result = _certify_validity_blocks(
            layout, sources, spec, mesh_, policy_, budget, imported
        )
        blocks, offsets, unsupported = result.blocks, result.offsets, result.unsupported
        certificate = CellValidityCertificate(
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
        if key is not None:
            budget.retain_basis((key,))
            budget.retain_owner(certificate)
            budget.validity_certificate_cache[key] = certificate
        return certificate


__all__ = [
    "CellValidityCertificate",
    "CellValidityPolicy",
    "CellValidityStatus",
    "certify_cell_geometry_validity",
]
