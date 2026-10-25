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

A degree-one simplex map proven affine by its source basis, whose reference
corners are cardinal coefficient rows, has a constant Jacobian: the polynomial
route's level-zero piece then has one Bernstein control equal to every corner
value. Such blocks decide that constant from the exact source-bank edge
differences under one block admission, with identical status, outward
endpoints and depth; values exactly at the floor and nonrational corners keep
the polynomial route.
Canonical Q1 boxes also use that constant differential when every exact corner
equals its affine axis-corner combination. This is a proof of the actual Q1
map, not an assumption from straight edges: a warped corner declines to the
expanded route, and the ordinary floor, endpoints and subdivision stay intact.

Simplex maps proven affine through their degree-one root whose exact reference
corner images were already earned in the same request for the identical
element and source bank differentiate those corners as cardinal coefficients.
Square affine simplex charts reuse the exact constant source differential.
Canonical barycentric actions use their source-owned exact homogeneous rows,
and must prove exact partition of unity before taking this pullback route.
Their transformed Jacobian is still formed in full for the policy scale;
signed determinants multiply by the chart determinant and embedded Gram
determinants by its square. Constancy is checked on the actual expressions,
not inferred from the chart shape, and singular/reversed maps retain the same
rank, sign, rounding and policy decisions as expanded polynomial composition.

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
from typing import assert_never, cast, Literal, TypeAlias

import equinox as eqx
import jax
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.discretization._reference_cell_types as reference_cell_types_owner
import phydrax.discretization.fem._reference_types as reference_types_owner
from phydrax.geometry._rational import fraction_endpoint

from .. import (
    _geometry_predicates as geometry_predicates_owner,
    _meshcore as meshcore_owner,
)
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
from ._cell_geometry_predecessor import (
    _CellGeometryPredecessorSource,
    _immutable_owner_fields_match,
    _snapshot_immutable_owner_fields,
    CellGeometryPredecessor,
    prepare_cell_geometry_predecessor,
)
from ._coordinate_budget import (
    coordinate_retention_preparation_bounds,
    CoordinateEnclosureBudget,
    reserve_polynomial,
)
from ._coordinate_identity import coordinate_source_bank_key
from ._exact_algebraic_geometry import coefficient_endpoint
from ._reference_subdivision import box_child_maps, prism_child_maps, simplex_child_maps
from .fem._reference import lagrange_element


# A replaced preparer may run an arbitrary continuation after earning its
# predecessor. Only the canonical uninterrupted producer owns the atomic path.
_ATOMIC_PREDECESSOR_PREPARER = prepare_cell_geometry_predecessor


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
            "source_element_ids": [element.element_id for element in geometry.elements],
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


@dataclass(frozen=True, slots=True)
class _AffineBoxCornerAction:
    nodes: tuple[int, ...]
    axes: tuple[int, ...]
    combinations: tuple[tuple[int, tuple[int, ...]], ...]
    work_per_component: int


def _prepare_affine_box_corner_action(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> _AffineBoxCornerAction | None:
    """Authenticate a private Q1 nodal action for this call, never a cell verdict."""
    if (
        cell_kind not in ("quadrilateral", "hexahedron")
        or not isinstance(element, reference_types_owner.FiniteElementSpec)
        or element.topological_dimension != dimension
        or element.element_id
        != cell_geometry_owner.coordinate_lagrange_element(cell_kind, 1).element_id
    ):
        return None
    nodes = cell_geometry_owner._cardinal_corner_nodes(element, budget=budget)
    if nodes is None:
        return None
    axes = (1, 3) if dimension == 2 else (1, 3, 4)
    reference = reference_cell_owner.reference_cell_topology(cell_kind).vertices
    budget.reserve(len(reference) * (dimension + 1), 4096)
    combinations = tuple(
        (index, tuple(axes[axis] for axis, value in enumerate(reference[index]) if value))
        for index in range(len(nodes)) if index not in (0, *axes)
    )
    return _AffineBoxCornerAction(
        nodes, axes, combinations,
        sum(len(selected) + 2 for _, selected in combinations),
    )


def _affine_box_action_corners(
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    action: _AffineBoxCornerAction,
    budget: CoordinateEnclosureBudget,
) -> coordinate_enclosure_owner.CoordinateSourceBank | None:
    """Check every coefficient and exact affine corner identity for one cell."""
    if (
        not coefficients
        or any(type(value) is not Fraction for row in coefficients for value in row)
    ):
        return None
    corners = tuple(coefficients[node] for node in action.nodes)
    origin = corners[0]
    bits = max(
        exact_algebraic_geometry_owner.coefficient_bits(value)
        for row in corners for value in row
    )
    reserve_polynomial(
        len(origin) * action.work_per_component, len(origin) + 8,
        0, 4 * bits + 32, budget=budget,
    )
    for index, selected in action.combinations:
        for component, base in enumerate(origin):
            expected = sum(
                (corners[corner][component] for corner in selected),
                (1 - len(selected)) * base,
            )
            if corners[index][component] != expected:
                return None
    return (origin, *(corners[index] for index in action.axes))


def _affine_box_corners(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> coordinate_enclosure_owner.CoordinateSourceBank | None:
    """Authenticate the complete canonical Q1 action as an affine corner frame."""
    action = _prepare_affine_box_corner_action(element, cell_kind, dimension, budget)
    return None if action is None else _affine_box_action_corners(coefficients, action, budget)


def _affine_box_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> _PolynomialDifferential | None:
    corners = _affine_box_corners(
        element, coefficients, cell_kind, dimension, embedded, budget
    )
    return (
        None if corners is None
        else _affine_edge_differential(corners, dimension, embedded, budget)
    )


def _affine_box_block(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    route: np.ndarray,
    source_values: coordinate_enclosure_owner.CoordinateSourceBank,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _BlockCertificate | None:
    """Transport a fully authenticated affine Q1 block to its simplex frames.

    Every Q1 coefficient participates in the exact affinity check. Equal exact
    edge frames then share one zero-origin representative: translation changes
    no Jacobian, determinant, Hadamard scale or relative policy floor. The
    canonical simplex owner consumes only those actual representative frames,
    not every translated source-bank row. Original geometry bindings stay intact.
    """
    dimension = {"quadrilateral": 2, "hexahedron": 3}.get(cell_kind)
    if (
        dimension is None or route.shape[0] == 0 or not source_values
        or not isinstance(element, reference_types_owner.FiniteElementSpec)
        or element.element_id
        != cell_geometry_owner.coordinate_lagrange_element(cell_kind, 1).element_id
    ):
        return None
    simplex_kind = "triangle" if dimension == 2 else "tetrahedron"
    simplex = budget.degree_one_coordinate_element(simplex_kind)
    simplex_nodes = cell_geometry_owner._cardinal_corner_nodes(simplex, budget=budget)
    if simplex_nodes is None:
        return None
    ambient = len(source_values[0])
    with budget.temporary_scope(), budget.live_storage() as storage:
        action = _prepare_affine_box_corner_action(element, cell_kind, dimension, budget)
        if action is None:
            return None
        budget.reserve(route.shape[0], 32 * route.shape[0] + 4096)
        rows = np.empty((route.shape[0],), dtype=np.int64)
        representatives: dict[tuple[tuple[Fraction, ...], ...], int] = {}
        points: list[tuple[Fraction, ...]] = []
        entry_bytes = 0
        for cell, row in enumerate(route):
            with budget.temporary_scope():
                corners = _affine_box_action_corners(
                    tuple(source_values[int(index)] for index in row), action, budget
                )
                if corners is None:
                    return None
                bits = 2 * max(
                    exact_algebraic_geometry_owner.coefficient_bits(value)
                    for point in corners for value in point
                ) + 2
                entry_bytes = max(
                    entry_bytes,
                    512 + dimension * ambient * (
                        128 + 2 * (32 + 4 * ((bits + 29) // 30))
                    ),
                )
                storage.set_bound((len(representatives) + 1) * entry_bytes)
                reserve_polynomial(
                    dimension * ambient, dimension * ambient, 0, bits, budget=budget
                )
                edges = tuple(
                    tuple(
                        cast(Fraction, value) - cast(Fraction, base)
                        for value, base in zip(point, corners[0], strict=True)
                    )
                    for point in corners[1:]
                )
                selected = representatives.get(edges)
                if selected is None:
                    selected = len(representatives)
                    representatives[edges] = selected
                    points.append((Fraction(0),) * ambient)
                    points.extend(edges)
                rows[cell] = selected
        count = len(representatives)
        budget.reserve(
            (dimension + 1) * count,
            8 * count * (dimension + 1) + 4096,
        )
        frames = np.empty((count, dimension + 1), dtype=np.int64)
        for index in range(count):
            for corner, node in enumerate(simplex_nodes):
                frames[index, node] = index * (dimension + 1) + corner
        decided = _affine_simplex_block(
            simplex, simplex_kind, frames, tuple(points), policy, budget
        )
        if decided is None or decided.reasons:
            # Topology-dependent unresolved reasons retain the original Q1
            # polynomial route rather than broadcasting a simplex fallback.
            return None
        return _BlockCertificate(
            decided.status[rows], decided.lower[rows], decided.upper[rows],
            decided.depth[rows], (),
        )


def _affine_source_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> _PolynomialDifferential | None:
    """Form a canonical rational affine root differential without basis expansion."""
    if cell_kind in ("quadrilateral", "hexahedron"):
        return _affine_box_differential(
            element, coefficients, cell_kind, dimension, embedded, budget
        )
    if (
        cell_kind not in ("triangle", "tetrahedron")
        or not isinstance(element, reference_types_owner.FiniteElementSpec)
        or element.cell_kind != cell_kind
        or element.degree != 1
        or element.topological_dimension != dimension
        or element.local_dof_count != dimension + 1
        or not coefficients
        or not coordinate_enclosure_owner.affine_simplex_source(element, budget=budget)
    ):
        return None
    nodes = cell_geometry_owner._cardinal_corner_nodes(element, budget=budget)
    if nodes is None or len(nodes) != dimension + 1:
        return None
    return _affine_edge_differential(
        tuple(coefficients[node] for node in nodes), dimension, embedded, budget
    )


def _affine_corner_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    key: coordinate_enclosure_owner.CoordinatePreparationKey,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> _PolynomialDifferential | None:
    """Differentiate a proven affine simplex map from its already earned corners.

    Exact reference-corner images retained by this request for the identical
    element/bank binding are its cardinal P1 coefficients, so no parent
    differential or chart composition is formed.
    """
    if (
        cell_kind not in ("triangle", "tetrahedron")
        or element.cell_kind != cell_kind
        or element.topological_dimension != dimension
    ):
        return None
    corners = budget.coordinate_corner_cache.get(key)
    if (
        corners is None
        or len(corners) != dimension + 1
        or not coordinate_enclosure_owner.affine_simplex_source(element, budget=budget)
    ):
        return None
    return _affine_edge_differential(corners, dimension, embedded, budget)


def _affine_edge_differential(
    corners: coordinate_enclosure_owner.CoordinateSourceBank,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> _PolynomialDifferential | None:
    """Constant differential of exact rational reference-corner images."""
    ambient = len(corners[0])
    if (
        ambient < dimension
        or embedded != (ambient != dimension)
        or (embedded and dimension == 3)
        or any(
            len(row) != ambient or any(type(value) is not Fraction for value in row)
            for row in corners
        )
    ):
        return None
    bits = max(
        exact_algebraic_geometry_owner.coefficient_bits(value)
        for row in corners
        for value in row
    )
    differences = dimension * ambient
    work = differences + _affine_simplex_measure_operations(dimension, ambient)
    if embedded:
        work += dimension * (2 * ambient - 1)
    reserve_polynomial(
        work,
        differences + dimension + 8,
        0,
        16 * dimension * ambient * (bits + 1) + 16,
        budget=budget,
    )
    origin = corners[0]
    edges = tuple(
        tuple(
            cast(Fraction, value) - cast(Fraction, base)
            for value, base in zip(corner, origin, strict=True)
        )
        for corner in corners[1:]
    )
    measure = _affine_simplex_measure(edges, embedded=embedded)
    jacobian = tuple(
        tuple(
            coordinate_enclosure_owner.constant(edges[column][row], dimension, budget=budget)
            for column in range(dimension)
        )
        for row in range(ambient)
    )
    polynomial = coordinate_enclosure_owner.constant(measure, dimension, budget=budget)
    return _PolynomialDifferential(polynomial, jacobian)


class _AffineChart(StrictModule):
    source_element: cell_geometry_owner.ScalarCoordinateElement
    weights: tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...]
    determinant: coordinate_enclosure_owner.Expression


def _affine_chart(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> _AffineChart | None:
    """Earn one element's exact constant affine chart, independent of its cells."""
    if cell_kind not in ("triangle", "tetrahedron"):
        return None
    match element:
        case cell_geometry_owner.PolynomialComposedCellGeometryElement():
            source_element = element.source_element
            chart_element = element.chart_element
            chart_coefficients = tuple(
                tuple(Fraction(numerator, denominator) for numerator, denominator in row)
                for row in element.chart_coefficients
            )
        case cell_geometry_owner.BarycentricCellGeometryElement():
            source_element = element.source_element
            root = source_element
            while isinstance(root, cell_geometry_owner.BarycentricCellGeometryElement):
                root = root.source_element
            cell_geometry_owner.require_p1_cardinal_source(root)
            chart_element = cell_geometry_owner.coordinate_lagrange_element(cell_kind, 1)
            snapshot = cell_geometry_owner.prepare_barycentric_source_snapshot(
                element, budget=budget
            )
            action, _ = coordinate_enclosure_owner.coordinate_preparation_key(
                chart_element,
                snapshot.barycentric_weights
                if snapshot.exact_weights is None
                else snapshot.exact_weights,
                budget=budget,
            )
            # A general coefficient action need not be a reference pullback.
            # Prove partition of unity from its exact source-owned rows first.
            with budget.temporary_scope():
                budget.reserve(len(action), 1024 * len(action))
                if any(sum(row, Fraction(0)) != 1 for row in action):
                    return None
            chart_coefficients = tuple(row[1:] for row in action)
        case _:
            return None
    if (
        element.cell_kind != cell_kind
        or source_element.cell_kind != cell_kind
        or chart_element.cell_kind != cell_kind
        or source_element.degree != 1
        or chart_element.degree != 1
        or source_element.topological_dimension != dimension
        or chart_element.topological_dimension != dimension
        or any(len(row) != dimension for row in chart_coefficients)
    ):
        return None
    chart = _prepare_polynomial_differential(
        chart_element, chart_coefficients, cell_kind, dimension, False, budget
    )
    if (
        chart.polynomial is None
        or chart.jacobian is None
        or len(chart.jacobian) != dimension
        or any(len(row) != dimension for row in chart.jacobian)
    ):
        return None
    constants = _constant_differential(chart.polynomial, chart.jacobian, dimension)
    if constants is None:
        return None
    chart_constants = constants[1:]
    weights = tuple(
        tuple(chart_constants[row * dimension + column] for row in range(dimension))
        for column in range(dimension)
    )
    return _AffineChart(source_element, weights, chart.polynomial)


class AffineChartFrame(StrictModule):
    """Constant chart columns of a composed simplex over its affine P1 source.

    For ``x = S(c(xi))`` with affine ``S`` and ``c`` the physical Jacobian is
    the constant ``J_S J_c``: ``J[:, k] = sum_a J_c[a][k] e_a`` over a cell's
    source edges ``e_a`` (``columns[k][a] = J_c[a][k]``), and
    ``det J = det J_S det J_c``.
    """

    source_nodes: tuple[int, ...]
    columns: tuple[tuple[Fraction, ...], ...]
    determinant: Fraction
    bits: int


@checked
def affine_chart_frame(
    element: reference_cell_types_owner.CellGeometryElement,
    cell_kind: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> AffineChartFrame | None:
    """Earn one composed simplex element's exact constant chart, or decline."""
    if not isinstance(
        element,
        (
            cell_geometry_owner.PolynomialComposedCellGeometryElement,
            cell_geometry_owner.BarycentricCellGeometryElement,
        ),
    ):
        return None
    dimension = element.topological_dimension
    if _AFFINE_SIMPLEX_DIMENSIONS.get(cell_kind) != dimension:
        return None
    chart = _affine_chart(element, cell_kind, dimension, budget)
    if chart is None:
        return None
    source = chart.source_element
    if not isinstance(
        source, reference_types_owner.FiniteElementSpec
    ) or not coordinate_enclosure_owner.affine_simplex_source(source, budget=budget):
        return None
    nodes = cell_geometry_owner._cardinal_corner_nodes(source, budget=budget)
    determinant = _constant_polynomial_value(chart.determinant, dimension)
    if (
        nodes is None
        or len(nodes) != dimension + 1
        or type(determinant) is not Fraction
        or any(type(value) is not Fraction for column in chart.weights for value in column)
    ):
        return None
    columns = cast(tuple[tuple[Fraction, ...], ...], chart.weights)
    bits = max(
        exact_algebraic_geometry_owner.coefficient_bits(value)
        for value in (determinant, *(value for column in columns for value in column))
    )
    return AffineChartFrame(nodes, columns, determinant, bits)


def _constant_differential(
    polynomial: coordinate_enclosure_owner.Expression,
    jacobian: tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...],
    dimension: int,
    /,
) -> list[coordinate_enclosure_owner.ExactScalar] | None:
    """Exact determinant and row-major Jacobian constants, if all are constant."""
    zero = (0,) * dimension
    # Metadata alone is insufficient: every actual differential and determinant
    # must be polynomial and constant before applying the affine identities.
    constants: list[coordinate_enclosure_owner.ExactScalar] = []
    for value in (polynomial, *(entry for row in jacobian for entry in row)):
        if isinstance(value, coordinate_enclosure_owner.RationalPolynomial) or any(
            index != zero for index in value
        ):
            return None
        constants.append(value.get(zero, Fraction(0)))
    return constants


def _affine_composed_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
    chart: _AffineChart | None = None,
) -> _PolynomialDifferential | None:
    """Pull back an earned constant differential through an exact affine chart.

    A block may pass its element's chart earned once for all of its cells.
    """
    if chart is None:
        chart = _affine_chart(element, cell_kind, dimension, budget)
    if chart is None:
        return None
    parent = _prepare_polynomial_differential(
        chart.source_element, coefficients, cell_kind, dimension, embedded, budget
    )
    if (
        parent.polynomial is None
        or parent.jacobian is None
        or any(len(row) != dimension for row in parent.jacobian)
        or _constant_differential(parent.polynomial, parent.jacobian, dimension) is None
    ):
        return None
    jacobian = tuple(
        coordinate_enclosure_owner.expression_linear_combinations(
            row, chart.weights, budget=budget
        )
        for row in parent.jacobian
    )
    determinant = chart.determinant
    if embedded:
        determinant = coordinate_enclosure_owner.expression_multiply(
            determinant, determinant, budget=budget
        )
    polynomial = coordinate_enclosure_owner.expression_multiply(
        parent.polynomial, determinant, budget=budget
    )
    return _PolynomialDifferential(polynomial, jacobian)


class _ComposedChart(StrictModule):
    source_element: cell_geometry_owner.ScalarCoordinateElement
    differential: _PolynomialDifferential


def _composed_chart(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> _ComposedChart | None:
    """Earn one element's polynomial chart differential over an affine P1 source."""
    if embedded or not isinstance(
        element, cell_geometry_owner.PolynomialComposedCellGeometryElement
    ):
        return None
    source_element, chart_element = element.source_element, element.chart_element
    if (
        not isinstance(source_element, reference_types_owner.FiniteElementSpec)
        or source_element.cell_kind not in ("triangle", "tetrahedron")
        or source_element.degree != 1
        or source_element.topological_dimension != dimension
        or chart_element.cell_kind != cell_kind
        or chart_element.topological_dimension != dimension
        or element.cell_kind != cell_kind
        or any(len(row) != dimension for row in element.chart_coefficients)
    ):
        return None
    chart_coefficients = tuple(
        tuple(Fraction(numerator, denominator) for numerator, denominator in row)
        for row in element.chart_coefficients
    )
    differential = _prepare_polynomial_differential(
        chart_element, chart_coefficients, cell_kind, dimension, False, budget
    )
    if (
        differential.polynomial is None
        or differential.jacobian is None
        or len(differential.jacobian) != dimension
        or any(len(row) != dimension for row in differential.jacobian)
    ):
        return None
    return _ComposedChart(source_element, differential)


def _affine_source_composed_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
    chart: _ComposedChart | None = None,
) -> _PolynomialDifferential | None:
    """Push an element's polynomial chart differential through an affine source.

    For ``x = S(c(xi))`` with affine ``S``, ``J = J_S J_c`` and
    ``det J = det J_S det J_c`` exactly; a block passes its chart once.
    """
    if chart is None:
        chart = _composed_chart(element, cell_kind, dimension, embedded, budget)
    if chart is None:
        return None
    source = chart.source_element
    parent = _prepare_polynomial_differential(
        source, coefficients, source.cell_kind, dimension, False, budget, retain=False
    )
    if (
        parent.polynomial is None
        or parent.jacobian is None
        or len(parent.jacobian) != dimension
        or any(len(row) != dimension for row in parent.jacobian)
    ):
        return None
    constants = _constant_differential(parent.polynomial, parent.jacobian, dimension)
    if constants is None:
        return None
    source_rows = tuple(
        tuple(constants[1 + row * dimension + column] for column in range(dimension))
        for row in range(dimension)
    )
    chart_jacobian = cast(
        tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...],
        chart.differential.jacobian,
    )
    columns = tuple(
        coordinate_enclosure_owner.expression_linear_combinations(
            tuple(chart_jacobian[axis][column] for axis in range(dimension)),
            source_rows,
            budget=budget,
        )
        for column in range(dimension)
    )
    jacobian = tuple(
        tuple(columns[column][row] for column in range(dimension))
        for row in range(dimension)
    )
    polynomial = coordinate_enclosure_owner.expression_multiply(
        parent.polynomial,
        cast(coordinate_enclosure_owner.Expression, chart.differential.polynomial),
        budget=budget,
    )
    return _PolynomialDifferential(polynomial, jacobian)


_TENSOR_SIMPLEX_SOURCES = {
    "quadrilateral": "triangle",
    "hexahedron": "tetrahedron",
    "pyramid": "tetrahedron",
}


class MultilinearChartFactors(StrictModule):
    """Shared factors of a multilinear tensor chart over a proven affine simplex.

    For ``x = S(c(xi))`` with affine ``S`` the polynomial route forms
    ``J = J_S J_c`` and ``det J = det J_S det J_c``, so every cell's
    determinant controls are ``det J_S`` times the chart's shared controls.
    Every chart Jacobian entry has per-axis degree at most one, hence each
    column ``J_c[:, k]`` is the multilinear interpolation of its box-vertex
    values ``columns[k]`` (in ``product((0, 1), repeat=d)`` vertex order): each
    physical column ``J_S J_c[:, k](xi)`` is a convex combination of ``J_S v``
    over those vectors, and each per-cell Jacobian entry's own-degree Bernstein
    controls are such vertex values.
    """

    source_nodes: tuple[int, ...]
    determinant_lower: Fraction
    determinant_upper: Fraction
    columns: tuple[tuple[tuple[Fraction, ...], ...], ...]
    column_bounds: tuple[Fraction, ...]
    bits: int


def _multilinear_factors(
    chart: _ComposedChart,
    cell_kind: str,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> MultilinearChartFactors | None:
    """Earn one element's multilinear chart factors, or decline."""
    source = chart.source_element
    polynomial, jacobian = chart.differential.polynomial, chart.differential.jacobian
    if (
        _TENSOR_SIMPLEX_SOURCES.get(cell_kind) != source.cell_kind
        or not isinstance(source, reference_types_owner.FiniteElementSpec)
        or polynomial is None
        or jacobian is None
        or isinstance(polynomial, coordinate_enclosure_owner.RationalPolynomial)
        or any(
            isinstance(entry, coordinate_enclosure_owner.RationalPolynomial)
            or any(max(index) > 1 for index in entry)
            for row in jacobian
            for entry in row
        )
        or not coordinate_enclosure_owner.affine_simplex_source(source, budget=budget)
    ):
        return None
    nodes = cell_geometry_owner._cardinal_corner_nodes(source, budget=budget)
    if nodes is None or len(nodes) != dimension + 1:
        return None
    controls = coordinate_enclosure_owner.expression_bernstein_coefficients_or_none(
        polynomial, "box", dimension, budget=budget
    )
    if controls is None or any(type(value) is not Fraction for value in controls):
        return None
    rational = cast(tuple[Fraction, ...], controls)
    columns: list[tuple[tuple[Fraction, ...], ...]] = []
    bounds: list[Fraction] = []
    for column in range(dimension):
        vectors: list[tuple[Fraction, ...]] = []
        for vertex in product((Fraction(0), Fraction(1)), repeat=dimension):
            values = tuple(
                coordinate_enclosure_owner.evaluate(
                    cast(coordinate_enclosure_owner.Polynomial, jacobian[row][column]),
                    vertex,
                    budget=budget,
                )
                for row in range(dimension)
            )
            if any(type(value) is not Fraction for value in values):
                return None
            vectors.append(cast(tuple[Fraction, ...], values))
        budget.reserve(2 * dimension * len(vectors))
        columns.append(tuple(vectors))
        bounds.append(
            max(sum((value * value for value in vector), Fraction(0)) for vector in vectors)
        )
    lower, upper = min(rational), max(rational)
    bits = max(
        exact_algebraic_geometry_owner.coefficient_bits(value)
        for value in (lower, upper, *bounds)
    )
    return MultilinearChartFactors(nodes, lower, upper, tuple(columns), tuple(bounds), bits)


@checked
def multilinear_chart_factors(
    element: reference_cell_types_owner.CellGeometryElement,
    cell_kind: str,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> MultilinearChartFactors | None:
    """Earn the shared factors of one polynomial-composed element, if it has them."""
    if not isinstance(element, cell_geometry_owner.PolynomialComposedCellGeometryElement):
        return None
    dimension = element.topological_dimension
    source_kind = _TENSOR_SIMPLEX_SOURCES.get(cell_kind)
    if source_kind is None or _AFFINE_SIMPLEX_DIMENSIONS[source_kind] != dimension:
        return None
    chart = _composed_chart(element, cell_kind, dimension, False, budget)
    return None if chart is None else _multilinear_factors(chart, cell_kind, dimension, budget)


def _rational_pyramid_chart(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
) -> _ComposedChart | None:
    """Earn a collapsed pyramid chart's differential over its affine P1 source.

    A rational collapsed pyramid stores its P1 pyramid chart ``c`` at rational
    source-reference nodes, so ``x = S(c(xi))`` on the collapsed box. The
    polynomial route's physical Jacobian divides the ``u`` and ``v`` rows by
    ``1 - w`` and adds ``(u - 1/2)`` and ``(v - 1/2)`` multiples of them to the
    ``w`` row; every step is linear in the map, so for affine ``S`` it is
    exactly ``J_S J_c`` with ``J_c`` the chart element's own physical Jacobian.
    """
    if (
        embedded
        or cell_kind != "pyramid"
        or not isinstance(element, cell_geometry_owner.RationalComposedCellGeometryElement)
    ):
        return None
    source_element, chart_element = element.source_element, element.chart_element
    if (
        not isinstance(source_element, reference_types_owner.FiniteElementSpec)
        or source_element.cell_kind != "tetrahedron"
        or source_element.degree != 1
        or chart_element.cell_kind != "pyramid"
        or chart_element.degree != 1
        or any(len(row) != 3 for row in element.chart_coefficients)
    ):
        return None
    chart_coefficients = tuple(
        tuple(Fraction(numerator, denominator) for numerator, denominator in row)
        for row in element.chart_coefficients
    )
    differential = _prepare_polynomial_differential(
        chart_element, chart_coefficients, "pyramid", 3, False, budget
    )
    if (
        differential.polynomial is None
        or differential.jacobian is None
        or len(differential.jacobian) != 3
        or any(len(row) != 3 for row in differential.jacobian)
    ):
        return None
    return _ComposedChart(source_element, differential)


@checked
def pyramid_chart_factors(
    element: reference_cell_types_owner.CellGeometryElement,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> MultilinearChartFactors | None:
    """Earn the shared factors of one rational collapsed-pyramid element.

    The chart's collapsed physical Jacobian has per-axis degree at most one on
    the box (``_multilinear_factors`` verifies it), so the multilinear factor
    identities hold for every cell over its affine tetrahedral source.
    """
    if not isinstance(element, cell_geometry_owner.RationalComposedCellGeometryElement):
        return None
    chart = _rational_pyramid_chart(element, "pyramid", False, budget)
    return None if chart is None else _multilinear_factors(chart, "pyramid", 3, budget)


def multilinear_source_rows(
    factors: MultilinearChartFactors,
    local: coordinate_enclosure_owner.CoordinateSourceBank,
    /,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Return a cell's exact rational source corners, or None to decline."""
    dimension = len(factors.source_nodes) - 1
    corners = tuple(local[node] for node in factors.source_nodes)
    if any(
        len(row) != dimension or any(type(value) is not Fraction for value in row)
        for row in corners
    ):
        return None
    return cast(tuple[tuple[Fraction, ...], ...], corners)


def affine_simplex_frame(
    rows: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[tuple[Fraction, ...], ...], Fraction]:
    """Exact source edges (the columns of ``J_S``) and the shared determinant.

    Callers reserve ``dimension**2`` differences plus
    ``_affine_simplex_measure_operations(dimension, dimension)`` beforehand.
    """
    origin = rows[0]
    edges = tuple(
        tuple(value - base for value, base in zip(corner, origin, strict=True))
        for corner in rows[1:]
    )
    return edges, _affine_simplex_measure(edges, embedded=False)


def affine_simplex_frame_operations(dimension: int, /) -> int:
    """Rational operations of ``affine_simplex_frame`` for a square source."""
    return dimension * dimension + _affine_simplex_measure_operations(
        dimension, dimension
    )


class _MultilinearChart(StrictModule):
    """Policy-bound root decision of one block's multilinear chart.

    Each per-cell entry's own-degree controls are bounded by ``|J_S[r]| V_k``
    with ``V_k^2 = column_bounds[k]``, so ``|J_S|_F^(2d) prod_k V_k^2``
    dominates the polynomial route's Hadamard scale.
    """

    factors: MultilinearChartFactors
    squared_threshold_factor: Fraction


def _multilinear_chart(
    chart: _ComposedChart,
    cell_kind: str,
    dimension: int,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _MultilinearChart | None:
    """Earn one block's multilinear decision data, or decline to the polynomial route."""
    polynomial, jacobian = chart.differential.polynomial, chart.differential.jacobian
    # Every per-cell expression has the chart's determinant support and at
    # most its Jacobian degrees, so the polynomial route's node admission holds.
    if (
        polynomial is None
        or jacobian is None
        or max(
            coordinate_enclosure_owner.expression_node_count(entry, "box", dimension)
            for row in jacobian
            for entry in row
        )
        > policy.maximum_bernstein_nodes
        or coordinate_enclosure_owner.expression_node_count(polynomial, "box", dimension)
        > policy.maximum_bernstein_nodes
    ):
        return None
    factors = _multilinear_factors(chart, cell_kind, dimension, budget)
    if factors is None:
        return None
    budget.reserve(dimension + 1)
    factor = Fraction(policy.relative_determinant_floor) ** 2
    for bound in factors.column_bounds:
        factor *= bound
    return _MultilinearChart(factors, factor)


def _certify_multilinear_cell(
    chart: _MultilinearChart,
    local: np.ndarray | coordinate_enclosure_owner.CoordinateSourceBank,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> _SubdivisionResult | None:
    """Return the polynomial route's level-zero VALID decision, else decline.

    A positive lower control whose square exceeds the dominating threshold is
    exactly that route's root acceptance; every other cell keeps the route's
    corner, subdivision and refusal semantics by declining.
    """
    if isinstance(local, np.ndarray):
        return None
    factors = chart.factors
    rows = multilinear_source_rows(factors, local)
    if rows is None:
        return None
    bits = max(
        exact_algebraic_geometry_owner.coefficient_bits(value)
        for row in rows
        for value in row
    )
    # The frame, the Frobenius norm, two chart products, the squared
    # determinant, the norm power, the threshold product, the comparison and
    # the two directed endpoints.
    reserve_polynomial(
        affine_simplex_frame_operations(dimension)
        + (2 * dimension * dimension - 1)
        + 2
        + 1
        + (dimension - 1)
        + 1
        + 1
        + 2,
        dimension * dimension + dimension + 8,
        0,
        16 * dimension * dimension * (bits + 1) + 2 * factors.bits + 64,
        budget=budget,
    )
    edges, measure = affine_simplex_frame(rows)
    if measure == 0:
        return None
    lower, upper = (
        (measure * factors.determinant_lower, measure * factors.determinant_upper)
        if measure > 0
        else (measure * factors.determinant_upper, measure * factors.determinant_lower)
    )
    if lower <= 0:
        return None
    frobenius = sum((value * value for edge in edges for value in edge), Fraction(0))
    if lower * lower <= chart.squared_threshold_factor * frobenius**dimension:
        return None
    return _SubdivisionResult(
        CellValidityStatus.CERTIFIED_VALID,
        coefficient_endpoint(lower, upper=False, saturate=True),
        coefficient_endpoint(upper, upper=True, saturate=True),
        0,
        (),
    )


def _prepare_polynomial_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    local: np.ndarray | coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
    *,
    retain: bool = True,
    chart: _AffineChart | _ComposedChart | None = None,
) -> _PolynomialDifferential:
    coefficients, authority = coordinate_enclosure_owner.coordinate_preparation_key(
        element, local, budget=budget
    )
    key = (authority, cell_kind, dimension, embedded)
    earned = budget.polynomial_differential_cache.get(key)
    if earned is not None:
        budget.reserve(0)
        return earned
    differential = _affine_differential(
        element, coefficients, authority, cell_kind, dimension, embedded, budget, chart
    )
    if differential is not None:
        _remember_signed_differential_operands(
            element, coefficients, authority, cell_kind, dimension, embedded,
            differential.polynomial, differential.jacobian, budget,
        )
        if retain:
            budget.retain_basis(
                (key, differential, differential.polynomial, differential.jacobian)
            )
            budget.polynomial_differential_cache[key] = differential
        return differential
    polynomial: coordinate_enclosure_owner.Expression | None = None
    jacobian = _polynomial_jacobian(element, coefficients, cell_kind, dimension, budget)
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
    _remember_signed_differential_operands(
        element, coefficients, authority, cell_kind, dimension, embedded,
        polynomial, jacobian, budget,
    )
    result = _PolynomialDifferential(polynomial, jacobian)
    if retain and polynomial is not None and jacobian is not None:
        budget.retain_basis((key, result, polynomial, jacobian))
        budget.polynomial_differential_cache[key] = result
    return result




def _remember_signed_differential_operands(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    authority: coordinate_enclosure_owner.CoordinatePreparationKey,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    polynomial: coordinate_enclosure_owner.Expression | None,
    jacobian: tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...] | None,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Issue only the full-dimensional signed numeric output, never a verdict."""
    if (
        not budget.capture_coordinate_operands
        or not budget.capture_signed_determinant_operands
        or embedded
        or element.cell_kind != cell_kind
        or cell_kind == "pyramid"
        or element.topological_dimension != dimension
        or not coefficients
        or any(len(row) != dimension for row in coefficients)
        or polynomial is None
        or jacobian is None
        or len(jacobian) != dimension
        or any(len(row) != dimension for row in jacobian)
    ):
        return
    source = element
    while isinstance(
        source,
        (
            cell_geometry_owner.BarycentricCellGeometryElement,
            cell_geometry_owner.RestrictedCellGeometryElement,
            cell_geometry_owner.PolynomialComposedCellGeometryElement,
        ),
    ):
        if (
            isinstance(source, cell_geometry_owner.PolynomialComposedCellGeometryElement)
            and source.chart_element.topological_dimension != dimension
        ):
            return
        source = source.source_element
        if source.topological_dimension != dimension or source.cell_kind == "pyramid":
            return
    from ._coordinate_operands import _remember_signed_determinant_operands

    _remember_signed_determinant_operands(
        authority, cell_kind, dimension, polynomial, budget,
    )
def _affine_differential(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    authority: coordinate_enclosure_owner.CoordinatePreparationKey,
    cell_kind: str,
    dimension: int,
    embedded: bool,
    budget: CoordinateEnclosureBudget,
    chart: _AffineChart | _ComposedChart | None,
) -> _PolynomialDifferential | None:
    """The constant or chart-pushed differential, before any basis expansion."""
    differential = _affine_source_differential(
        element, coefficients, cell_kind, dimension, embedded, budget
    )
    if differential is None:
        differential = _affine_corner_differential(
            element, authority, cell_kind, dimension, embedded, budget
        )
    if differential is None and not isinstance(chart, _ComposedChart):
        differential = _affine_composed_differential(
            element, coefficients, cell_kind, dimension, embedded, budget, chart
        )
    if differential is None and not isinstance(chart, _AffineChart):
        differential = _affine_source_composed_differential(
            element, coefficients, cell_kind, dimension, embedded, budget, chart
        )
    return differential


def _polynomial_jacobian(
    element: cell_geometry_owner.ScalarCoordinateElement,
    coefficients: coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...] | None:
    """The full physical Jacobian of the expanded polynomial or rational map."""
    coordinates = coordinate_enclosure_owner.coordinate_polynomials(
        element, coefficients, budget=budget
    )
    if coordinates is not None:
        return coordinate_enclosure_owner.physical_jacobian(
            coordinates, cell_kind, dimension, budget=budget
        )
    expressions = coordinate_enclosure_owner.coordinate_expressions(
        element, coefficients, budget=budget
    )
    if expressions is None:
        return None
    return coordinate_enclosure_owner.expression_physical_jacobian(
        expressions, cell_kind, dimension, budget=budget
    )


class _SubdivisionResult(StrictModule):
    status: int
    lower: float
    upper: float
    depth: int
    reasons: tuple[CellValidityUnresolvedReason, ...]


def _constant_polynomial_value(
    expression: coordinate_enclosure_owner.Expression, dimension: int, /
) -> coordinate_enclosure_owner.ExactScalar | None:
    """Return an earned degree-zero coefficient, not a sampled approximation."""
    zero = (0,) * dimension
    if isinstance(expression, coordinate_enclosure_owner.RationalPolynomial) or any(
        index != zero for index in expression
    ):
        return None
    return expression.get(zero, Fraction(0))


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
            constant = _constant_polynomial_value(entry, dimension)
            if constant is not None:
                magnitudes.append(abs(constant))
                continue
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
    constant = _constant_polynomial_value(piece, dimension)
    coefficients = (
        (constant,)
        if constant is not None
        else coordinate_enclosure_owner.expression_bernstein_coefficients_or_none(
            piece, domain, dimension, budget=budget
        )
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
    # Every control and corner of an actual degree-zero polynomial is this
    # same exact value, so no basis conversion or repeated evaluation is needed.
    corner_values = (
        (constant,)
        if constant is not None
        else _expression_corner_values(piece, vertices, budget)
    )
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
            # The root piece is already expressed on this reference domain.
            # Do not reconstruct an identity chart and copy its polynomial.
            piece = (
                polynomial
                if level == 0
                else coordinate_enclosure_owner.expression_compose(
                    polynomial,
                    coordinate_enclosure_owner.affine_arguments(
                        origin, matrix, budget=budget
                    ),
                    budget=budget,
                )
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


def _certify_polynomial_cell(
    element: cell_geometry_owner.ScalarCoordinateElement,
    local: np.ndarray | coordinate_enclosure_owner.CoordinateSourceBank,
    reference: _PolynomialReferenceDomain,
    dimension: int,
    embedded: bool,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
    chart: _AffineChart | _ComposedChart | None = None,
) -> _SubdivisionResult:
    """Consume a leaf differential before its caller releases temporary storage."""
    differential = _prepare_polynomial_differential(
        element,
        local,
        reference.cell_kind,
        dimension,
        embedded,
        budget,
        retain=False,
        chart=chart,
    )
    polynomial, jacobian = differential.polynomial, differential.jacobian
    if polynomial is None or jacobian is None:
        return _SubdivisionResult(
            CellValidityStatus.UNRESOLVED, math.nan, math.nan, 0, ("unsupported_element",)
        )
    domain = reference.domain
    if (
        max(
            coordinate_enclosure_owner.expression_node_count(value, domain, dimension)
            for row in jacobian
            for value in row
        ) > policy.maximum_bernstein_nodes
        or coordinate_enclosure_owner.expression_node_count(polynomial, domain, dimension)
        > policy.maximum_bernstein_nodes
    ):
        return _SubdivisionResult(
            CellValidityStatus.UNRESOLVED, math.nan, math.nan, 0,
            ("bernstein_node_budget",),
        )
    magnitude = _expression_magnitudes(jacobian, domain, dimension, budget)
    if magnitude is None:
        return _SubdivisionResult(
            CellValidityStatus.UNRESOLVED, math.nan, math.nan, 0, ("rounding_enclosure",)
        )
    squared_threshold = _squared_threshold(magnitude, dimension, embedded, policy)
    return _certify_polynomial_subdivision(
        polynomial,
        domain,
        dimension,
        reference.vertices,
        reference.child_origins,
        reference.child_matrices,
        squared_threshold,
        policy,
        budget,
    )


def _squared_threshold(
    magnitude: tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...],
    dimension: int,
    embedded: bool,
    policy: CellValidityPolicy,
) -> coordinate_enclosure_owner.ExactScalar:
    """The relative floor against the Hadamard scale of the entry magnitudes."""
    squared_scale = math.prod(
        sum((row[axis] * row[axis] for row in magnitude), Fraction(0))
        for axis in range(dimension)
    )
    floor = Fraction(policy.relative_determinant_floor)
    return (floor * squared_scale) ** 2 if embedded else floor * floor * squared_scale


_POLYNOMIAL_REFERENCE_KINDS = (
    "interval", "triangle", "quadrilateral", "tetrahedron", "hexahedron"
)


class _PolynomialChartFactors(StrictModule):
    """Nested degree-one chart determinant factors over a proved root map.

    For ``x = R(c_1(...c_m(xi)))``, the chain rule gives
    ``det J = det J_R prod_i det J_{c_i}`` in full dimension and
    ``det(J^T J) = g_R prod_i det(J_{c_i})^2`` for an embedding.
    Canonical degree-one charts map into the convex hull of their corners,
    proved inside each source reference cell. The root and chart control
    extremes therefore enclose every factor on the chained image. Volume
    charts must preserve orientation; all other cases decline.
    """

    root: cell_geometry_owner.ScalarCoordinateElement
    root_domain: coordinate_enclosure_owner.BernsteinDomain
    lower: coordinate_enclosure_owner.ExactScalar
    upper: coordinate_enclosure_owner.ExactScalar
    embedded: bool
    axis_action: tuple[tuple[int, Fraction], ...] | None


class _RootJacobianBernsteinOperands(StrictModule):
    """Exact numerical bounds and supports, never a cell validity verdict."""

    magnitudes: tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...]
    determinant_nodes: int
    gram_nodes: int


def _compose_box_axis_action(
    differential: _PolynomialDifferential,
    prior: tuple[tuple[int, Fraction], ...] | None,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[int, Fraction], ...] | None:
    """Compose only an earned invertible signed axis scaling/permutation."""
    jacobian = differential.jacobian
    if prior is None or jacobian is None or len(jacobian) != dimension:
        return None
    budget.reserve(dimension * dimension + 2 * dimension)
    result: list[tuple[int, Fraction]] = []
    used: set[int] = set()
    for row in jacobian:
        if len(row) != dimension:
            return None
        constants = tuple(_constant_polynomial_value(value, dimension) for value in row)
        if any(type(value) is not Fraction for value in constants):
            return None
        nonzero = tuple(axis for axis, value in enumerate(constants) if value != 0)
        if len(nonzero) != 1 or nonzero[0] in used:
            return None
        axis = nonzero[0]
        used.add(axis)
        scale = cast(Fraction, constants[axis])
        result.append((prior[axis][0], scale * prior[axis][1]))
    return tuple(result)


def _reference_contains(
    domain: coordinate_enclosure_owner.BernsteinDomain, point: tuple[Fraction, ...]
) -> bool:
    match domain:
        case "simplex":
            return all(value >= 0 for value in point) and sum(point) <= 1
        case "box":
            return all(0 <= value <= 1 for value in point)
        case "prism":
            return False
        case invalid:
            assert_never(invalid)


def _polynomial_chart_factors(
    element: cell_geometry_owner.ScalarCoordinateElement,
    dimension: int,
    budget: CoordinateEnclosureBudget,
    *,
    embedded: bool,
) -> _PolynomialChartFactors | None:
    """Earn one element's nested chart determinant factors, or decline."""
    lower: coordinate_enclosure_owner.ExactScalar = Fraction(1)
    upper: coordinate_enclosure_owner.ExactScalar = Fraction(1)
    current: cell_geometry_owner.ScalarCoordinateElement = element
    axis_action: tuple[tuple[int, Fraction], ...] | None = tuple(
        (axis, Fraction(1)) for axis in range(dimension)
    )
    while isinstance(current, cell_geometry_owner.PolynomialComposedCellGeometryElement):
        source, chart = (
            cell_geometry_owner.require_scalar_coordinate_element(
                current.source_element, "Composed source"
            ),
            current.chart_element,
        )
        if (
            chart.cell_kind not in _POLYNOMIAL_REFERENCE_KINDS
            or source.cell_kind not in _POLYNOMIAL_REFERENCE_KINDS
            or chart.topological_dimension != dimension
            or source.topological_dimension != dimension
            or current.cell_kind != chart.cell_kind
            or chart.element_id
            != cell_geometry_owner.coordinate_lagrange_element(
                chart.cell_kind, 1
            ).element_id
        ):
            return None
        source_domain, _ = determinant_route(source.cell_kind, 1, False)
        chart_domain, _ = determinant_route(chart.cell_kind, 1, False)
        corners = tuple(
            tuple(Fraction(numerator, denominator) for numerator, denominator in row)
            for row in current.chart_coefficients
        )
        budget.reserve(2 * dimension * len(corners))
        if not all(_reference_contains(source_domain, corner) for corner in corners):
            return None
        differential = _prepare_polynomial_differential(
            chart, corners, chart.cell_kind, dimension, False, budget
        )
        if differential.polynomial is None:
            return None
        controls = coordinate_enclosure_owner.expression_bernstein_coefficients_or_none(
            differential.polynomial, chart_domain, dimension, budget=budget
        )
        if controls is None:
            return None
        smallest, largest = min(controls), max(controls)
        if smallest > 0:
            low, high = smallest, largest
        elif embedded and largest < 0:
            low, high = -largest, -smallest
        else:
            return None
        budget.reserve(4 if embedded else 2)
        lower *= low * low if embedded else low
        upper *= high * high if embedded else high
        axis_action = (
            _compose_box_axis_action(differential, axis_action, dimension, budget)
            if source_domain == chart_domain == "box"
            else None
        )
        current = source
    if current is element or current.cell_kind not in _POLYNOMIAL_REFERENCE_KINDS:
        return None
    root_domain, _ = determinant_route(current.cell_kind, 1, embedded)
    return _PolynomialChartFactors(current, root_domain, lower, upper, embedded, axis_action)


def _root_differential_bounds(
    factors: _PolynomialChartFactors,
    local: np.ndarray | coordinate_enclosure_owner.CoordinateSourceBank,
    dimension: int,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> tuple[coordinate_enclosure_owner.ExactScalar, coordinate_enclosure_owner.ExactScalar] | None:
    """Positive root differential bounds earned once per actual bank and request."""
    root = factors.root
    _, authority = coordinate_enclosure_owner.coordinate_preparation_key(
        root, local, budget=budget
    )
    key = (authority, root.cell_kind, dimension, policy.maximum_bernstein_nodes)
    cache = (
        budget.embedded_gram_bounds_cache
        if factors.embedded
        else budget.volume_determinant_bounds_cache
    )
    if key in cache:
        budget.reserve(0)
        return cache[key]
    bounds = None
    with budget.temporary_scope():
        polynomial = _prepare_polynomial_differential(
            root, local, root.cell_kind, dimension, factors.embedded, budget, retain=False
        ).polynomial
        if (
            polynomial is not None
            and coordinate_enclosure_owner.expression_node_count(
                polynomial, factors.root_domain, dimension
            )
            <= policy.maximum_bernstein_nodes
        ):
            controls = (
                coordinate_enclosure_owner.expression_bernstein_coefficients_or_none(
                    polynomial, factors.root_domain, dimension, budget=budget
                )
            )
            if controls is not None and min(controls) > 0:
                bounds = (min(controls), max(controls))
    budget.retain_basis((key, bounds))
    cache[key] = bounds
    return bounds


def _degree_profile(
    polynomial: coordinate_enclosure_owner.Polynomial,
    domain: coordinate_enclosure_owner.BernsteinDomain,
    dimension: int,
) -> tuple[int, ...]:
    if domain == "simplex":
        return (max((sum(index) for index in polynomial), default=0),)
    return tuple(
        max((index[axis] for index in polynomial), default=0)
        for axis in range(dimension)
    )


def _differential_node_bound(
    jacobian: tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...],
    domain: coordinate_enclosure_owner.BernsteinDomain,
    dimension: int,
    budget: CoordinateEnclosureBudget,
    *,
    embedded: bool,
) -> int | None:
    """Bound the determinant or embedded Gram determinant's Bernstein nodes.

    With a shared denominator, every order-``d`` numerator minor has degree
    at most ``d`` times the largest entry degree. Embedded Gram determinants
    square these minors and their denominator; full determinants do not.
    Exact shifts and cancellation can only lower the bound.
    """
    denominator: coordinate_enclosure_owner.Polynomial | None = None
    for row in jacobian:
        for value in row:
            if isinstance(value, coordinate_enclosure_owner.RationalPolynomial):
                if denominator is None:
                    denominator = value.denominator
                    continue
                budget.reserve(len(denominator))
                if value.denominator != denominator:
                    return None
    width = 1 if domain == "simplex" else dimension
    shift = (
        (0,) * width
        if denominator is None
        else _degree_profile(denominator, domain, dimension)
    )
    numerator = (0,) * width
    for row in jacobian:
        for value in row:
            profile = (
                _degree_profile(value.numerator, domain, dimension)
                if isinstance(value, coordinate_enclosure_owner.RationalPolynomial)
                else tuple(
                    a + b
                    for a, b in zip(
                        _degree_profile(value, domain, dimension), shift, strict=True
                    )
                )
            )
            numerator = tuple(max(a, b) for a, b in zip(numerator, profile, strict=True))
    order = (2 if embedded else 1) * dimension
    union = tuple(order * max(a, b) for a, b in zip(numerator, shift, strict=True))
    index = union if domain != "simplex" else (union[0],) + (0,) * (dimension - 1)
    count = coordinate_enclosure_owner.bernstein_node_count(
        {index: Fraction(1)}, domain, dimension
    )
    return count if denominator is None else 2 * count


def _restricted_box_jacobian_magnitudes(
    factors: _PolynomialChartFactors,
    local: np.ndarray | coordinate_enclosure_owner.CoordinateSourceBank,
    dimension: int,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[coordinate_enclosure_owner.ExactScalar, ...], ...] | None:
    """Bound the original child Bernstein scale without expanding its map.

    An invertible contained signed-axis box chart restricts each root
    Bernstein net by de Casteljau subdivision, and possibly reverses its rows.
    These are convex combinations. Its nonzero axis scales preserve actual
    polynomial degrees, so these bounds enclose the child net at the same
    minimal degrees used by the expanded route, not merely an elevated net.
    """
    action = factors.axis_action
    if factors.root_domain != "box" or action is None:
        return None
    root = factors.root
    coefficients, authority = coordinate_enclosure_owner.coordinate_preparation_key(
        root, local, budget=budget
    )
    key = (authority, root.cell_kind, dimension, policy.maximum_bernstein_nodes)
    cache = budget.root_jacobian_bernstein_operands_cache
    if key not in cache:
        operands = None
        with budget.temporary_scope():
            jacobian = _polynomial_jacobian(
                root, coefficients, root.cell_kind, dimension, budget
            )
            if jacobian is not None and all(
                not isinstance(value, coordinate_enclosure_owner.RationalPolynomial)
                and coordinate_enclosure_owner.expression_node_count(value, "box", dimension)
                <= policy.maximum_bernstein_nodes
                for row in jacobian for value in row
            ):
                magnitudes = _expression_magnitudes(jacobian, "box", dimension, budget)
                if magnitudes is not None:
                    degrees = tuple(
                        tuple(
                            _degree_profile(
                                cast(coordinate_enclosure_owner.Polynomial, value),
                                "box", dimension,
                            )
                            for value in row
                        )
                        for row in jacobian
                    )
                    budget.reserve(4 * dimension * dimension * len(degrees) + dimension)
                    maximum = tuple(
                        max(profile[axis] for row in degrees for profile in row)
                        for axis in range(dimension)
                    )
                    operands = _RootJacobianBernsteinOperands(
                        magnitudes,
                        math.prod(dimension * degree + 1 for degree in maximum),
                        math.prod(2 * dimension * degree + 1 for degree in maximum),
                    )
        budget.retain_basis((key, operands))
        cache[key] = operands
    operands = cache[key]
    if operands is None:
        return None
    budget.reserve(2 * dimension * len(operands.magnitudes) + dimension * dimension + 2)
    source_axes = tuple(
        next(source for source, (target, _) in enumerate(action) if target == axis)
        for axis in range(dimension)
    )
    # The entry-node products and determinant admission are invariant under
    # signed axis permutation; the former were admitted with the root net.
    nodes = operands.gram_nodes if factors.embedded else operands.determinant_nodes
    if nodes > policy.maximum_bernstein_nodes:
        return None
    return tuple(
        tuple(abs(action[source][1]) * row[source] for source in source_axes)
        for row in operands.magnitudes
    )


def _certify_factored_cell(
    factors: _PolynomialChartFactors,
    element: cell_geometry_owner.ScalarCoordinateElement,
    local: np.ndarray | coordinate_enclosure_owner.CoordinateSourceBank,
    reference: _PolynomialReferenceDomain,
    dimension: int,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _SubdivisionResult | None:
    """Return VALID from exact root/chart factor bounds, otherwise decline.

    Acceptance conservatively bounds the owning polynomial route's unchanged
    threshold and Bernstein-node admission. An inconclusive root scale uses
    the original expanded child. The positive bound covers the closed cell.
    """
    embedded = factors.embedded
    bounds = _root_differential_bounds(factors, local, dimension, policy, budget)
    if bounds is None:
        return None
    coefficients, authority = coordinate_enclosure_owner.coordinate_preparation_key(
        element, local, budget=budget
    )
    if (
        authority,
        reference.cell_kind,
        dimension,
        embedded,
    ) in budget.polynomial_differential_cache or _affine_differential(
        element, coefficients, authority, reference.cell_kind, dimension, embedded, budget, None
    ) is not None:
        return None
    magnitude = _restricted_box_jacobian_magnitudes(
        factors, local, dimension, policy, budget
    )
    lower = factors.lower * bounds[0]
    squared_threshold: coordinate_enclosure_owner.ExactScalar = Fraction(0)
    if magnitude is not None:
        budget.reserve(4 * dimension * len(magnitude) + 8)
        squared_threshold = _squared_threshold(magnitude, dimension, embedded, policy)
        if lower <= 0 or lower * lower <= squared_threshold:
            # A looser root bound cannot refuse a child that the original
            # expanded net might decide: retain that independent path below.
            magnitude = None
    if magnitude is None:
        jacobian = _polynomial_jacobian(
            element, coefficients, reference.cell_kind, dimension, budget
        )
        domain = reference.domain
        if jacobian is None or max(
            coordinate_enclosure_owner.expression_node_count(value, domain, dimension)
            for row in jacobian
            for value in row
        ) > policy.maximum_bernstein_nodes:
            return None
        nodes = _differential_node_bound(
            jacobian, domain, dimension, budget, embedded=embedded
        )
        if nodes is None or nodes > policy.maximum_bernstein_nodes:
            return None
        magnitude = _expression_magnitudes(jacobian, domain, dimension, budget)
        if magnitude is None:
            return None
        budget.reserve(4 * dimension * len(magnitude) + 8)
        squared_threshold = _squared_threshold(magnitude, dimension, embedded, policy)
    if lower <= 0 or lower * lower <= squared_threshold:
        return None
    return _SubdivisionResult(
        CellValidityStatus.CERTIFIED_VALID,
        coefficient_endpoint(lower, upper=False, saturate=True),
        coefficient_endpoint(factors.upper * bounds[1], upper=True, saturate=True),
        0,
        (),
    )


_QuadraticTetrahedronDerivativeTemplate: TypeAlias = tuple[
    tuple[tuple[int, ...], ...], ...
]


def _canonical_quadratic_tetrahedron_element(
    element: reference_cell_types_owner.CellGeometryElement, /
) -> reference_types_owner.FiniteElementSpec | None:
    if (
        not isinstance(element, reference_types_owner.FiniteElementSpec)
        or element.family != "SimplexLagrange"
        or element.cell_kind != "tetrahedron"
        or element.degree != 2
        or element.topological_dimension != 3
        or element.local_dof_count != 10
        or element.conformity != "H1"
        or element.mapping != "identity"
        or element.representation != "point_value"
        or element.value_shape
    ):
        return None
    coordinate_element = cell_geometry_owner.coordinate_lagrange_element(
        "tetrahedron", 2
    )
    if element.element_id == coordinate_element.element_id:
        canonical = coordinate_element
    else:
        canonical = lagrange_element("tetrahedron", 2)
        if element.element_id != canonical.element_id:
            return None
    if (
        coordinate_enclosure_owner.coordinate_source_signature(element)
        != coordinate_enclosure_owner.coordinate_source_signature(canonical)
    ):
        return None
    return canonical


def _quadratic_tetrahedron_native_input(
    spec: cell_geometry_owner.CellGeometrySpec,
    element: reference_cell_types_owner.CellGeometryElement,
    cell_kind: str,
    routes: np.ndarray,
    coordinates: np.ndarray,
    policy: CellValidityPolicy,
    template_cache: dict[tuple[str, str], _QuadraticTetrahedronDerivativeTemplate],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[
    np.ndarray,
    np.ndarray,
    _QuadraticTetrahedronDerivativeTemplate,
] | None:
    """Authenticate the exact canonical P2 source before considering array layout."""
    int64_max = (1 << 63) - 1
    if (
        cell_kind != "tetrahedron"
        or spec.exact_source is not None
        or spec.periodic_source is not None
        or spec.restriction_source is not None
        or spec.storage is not None
        or policy.maximum_bernstein_nodes > int64_max
        or policy.maximum_subdivision_depth > int64_max
        or not meshcore_owner.meshcore_provider_installed()
    ):
        return None
    canonical = _canonical_quadratic_tetrahedron_element(element)
    if (
        canonical is None
        or coordinates.ndim != 2
        or coordinates.shape[1] != 3
        or routes.ndim != 2
        or routes.shape[1] != 10
        or routes.shape[0] == 0
        or routes.shape[0] > int64_max // 30
        or routes.shape[0] * 30 * 8 > np.iinfo(np.intp).max
    ):
        return None
    identity = coordinate_enclosure_owner.element_identity(element, budget=budget)
    if identity != coordinate_enclosure_owner.element_identity(
        canonical, budget=budget
    ):
        return None
    template = template_cache.get(identity)
    if template is None:
        basis = coordinate_enclosure_owner.source_basis(element, budget=budget)
        if basis is None or len(basis) != 10:
            return None
        powers = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
        axes: list[tuple[tuple[int, ...], ...]] = []
        for axis in range(3):
            partials = tuple(
                coordinate_enclosure_owner.derivative(
                    polynomial, axis, budget=budget
                )
                for polynomial in basis
            )
            if any(
                exponent not in powers
                for partial in partials
                for exponent in partial
            ):
                return None
            axis_weights: list[tuple[int, ...]] = []
            for exponent in powers:
                weights: list[int] = []
                for partial in partials:
                    value = partial.get(exponent, Fraction(0))
                    if type(value) is not Fraction or value.denominator != 1:
                        return None
                    integer = value.numerator
                    if not -(1 << 31) <= integer < (1 << 31):
                        return None
                    weights.append(integer)
                axis_weights.append(tuple(weights))
            axes.append(tuple(axis_weights))
        template = tuple(axes)
        budget.retain_basis(template)
        template_cache[identity] = template
    return coordinates, routes, template



def _quadratic_tetrahedron_staging_bounds(count: int, /) -> tuple[int, int]:
    nodes_bytes = count * 10 * 3 * np.dtype(np.float64).itemsize
    weights_bytes = 3 * 4 * 10 * np.dtype(np.int32).itemsize
    outputs_bytes = (
        count
        * (
            np.dtype(np.int32).itemsize
            + 2 * np.dtype(np.float64).itemsize
        )
        + 6 * np.dtype(np.int64).itemsize
    )
    return (
        nodes_bytes + weights_bytes + 2 * 256,
        nodes_bytes + weights_bytes + outputs_bytes + 6 * 256,
    )



def _run_quadratic_tetrahedron_native_root(
    native_input: tuple[
        np.ndarray,
        np.ndarray,
        _QuadraticTetrahedronDerivativeTemplate,
    ],
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
    /,
) -> meshcore_owner.QuadraticTetrahedronValidityResult:
    coordinates, routes, template = native_input
    count = routes.shape[0]
    int64_max = (1 << 63) - 1
    host_staging, native_staging = _quadratic_tetrahedron_staging_bounds(count)
    ticket = budget.ticket
    staging_storage = (
        host_staging
        if ticket.native_accounting
        else host_staging + native_staging
    )
    budget.reserve(count * 30 + 120, staging_storage)
    nodes = coordinates[routes]
    weights = np.asarray(template, dtype=np.int32)
    work_limit = min(budget.remaining_work_units, int64_max)
    scratch_headroom = budget.remaining_memory_bytes
    if ticket.native_accounting:
        scratch_headroom -= native_staging
    scratch_limit = min(int64_max, max(0, scratch_headroom))
    try:
        with (
            ticket.native_target()
            if ticket.native_accounting
            else nullcontext()
        ):
            result = meshcore_owner.quadratic_tetrahedron_validity(
                nodes,
                weights,
                policy.relative_determinant_floor,
                policy.maximum_bernstein_nodes,
                policy.maximum_subdivision_depth,
                work_limit,
                scratch_limit,
            )
    except meshcore_owner.MeshcoreError as error:
        if isinstance(error, meshcore_owner.QuadraticTetrahedronValidityError):
            budget.record_native_work(int(error.evidence[0]))
        elif error.work_counters is not None:
            budget.record_native_work(error.work_counters.construction_work)
        raise
    budget.record_native_work(int(result.evidence[0]))
    return result

def _certify_polynomial_block(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    local: np.ndarray | tuple[coordinate_enclosure_owner.CoordinateSourceBank, ...],
    policy: CellValidityPolicy,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    quadratic_root: meshcore_owner.QuadraticTetrahedronValidityResult | None = None,
) -> _BlockCertificate:

    dimension = element.topological_dimension
    cell_count = (
        quadratic_root.decisions.shape[0]
        if quadratic_root is not None
        else len(local)
    )
    if quadratic_root is not None and sum(
        int(decision == meshcore_owner.QuadraticTetrahedronValidityDecision.NEEDS_SUBDIVISION)
        for decision in quadratic_root.decisions
    ) != len(local):
        raise RuntimeError(
            "_certify_polynomial_block.quadratic_root must align its subdivision rows with local source banks."
        )
    embedded = len(local) > 0 and len(local[0][0]) != dimension
    reference = _prepare_polynomial_reference_domain(
        cell_kind, dimension, element.degree, embedded
    )
    if reference is None:
        return _unresolved_block(cell_count, "unsupported_element")
    status = np.full((cell_count,), CellValidityStatus.CERTIFIED_VALID, dtype=np.int32)
    lower = np.full((cell_count,), np.inf, dtype=np.float64)
    upper = np.full((cell_count,), -np.inf, dtype=np.float64)
    depth = np.zeros((cell_count,), dtype=np.int32)
    reasons: list[CellValidityUnresolvedReason] = []

    def publish_quadratic_root(cell: int, /) -> bool:
        if quadratic_root is None:
            return False
        decision = meshcore_owner.QuadraticTetrahedronValidityDecision(
            int(quadratic_root.decisions[cell])
        )
        if decision is meshcore_owner.QuadraticTetrahedronValidityDecision.NEEDS_SUBDIVISION:
            return False
        lower[cell] = quadratic_root.lower[cell]
        upper[cell] = quadratic_root.upper[cell]
        match decision:
            case meshcore_owner.QuadraticTetrahedronValidityDecision.ROOT_VALID:
                status[cell] = CellValidityStatus.CERTIFIED_VALID
            case meshcore_owner.QuadraticTetrahedronValidityDecision.ROOT_INVALID:
                status[cell] = CellValidityStatus.INVALID
            case meshcore_owner.QuadraticTetrahedronValidityDecision.BERNSTEIN_NODE_LIMIT:
                status[cell] = CellValidityStatus.UNRESOLVED
                reasons.append("bernstein_node_budget")
            case meshcore_owner.QuadraticTetrahedronValidityDecision.DEPTH_ZERO_UNRESOLVED:
                status[cell] = CellValidityStatus.UNRESOLVED
                reasons.append("subdivision_depth")
            case meshcore_owner.QuadraticTetrahedronValidityDecision.NEEDS_SUBDIVISION:
                assert_never(decision)
        return True

    if quadratic_root is not None and len(local) == 0:
        for cell in range(cell_count):
            publish_quadratic_root(cell)
        return _BlockCertificate(status, lower, upper, depth, tuple(reasons))
    # Every cell of a block shares its element, hence one exact affine chart
    # or one polynomial chart over its affine source.
    chart: _AffineChart | _ComposedChart | None = (
        _affine_chart(element, cell_kind, dimension, budget) if len(local) else None
    )
    if chart is None and len(local):
        chart = _composed_chart(element, cell_kind, dimension, embedded, budget)
    # A rational collapsed pyramid decides through its shared chart factors,
    # while its declining cells keep the unchanged expanded polynomial route.
    pyramid = (
        _rational_pyramid_chart(element, reference.cell_kind, embedded, budget)
        if chart is None and len(local)
        else None
    )
    multilinear: _MultilinearChart | None = None
    if isinstance(chart, _ComposedChart) and reference.domain == "box":
        multilinear = _multilinear_chart(
            chart, reference.cell_kind, dimension, policy, budget
        )
    elif pyramid is not None:
        multilinear = _multilinear_chart(pyramid, "pyramid", dimension, policy, budget)
    factors = (
        _polynomial_chart_factors(element, dimension, budget, embedded=embedded)
        if chart is None and len(local)
        else None
    )
    ledger = budget
    local_cursor = 0
    for cell in range(cell_count):
        if publish_quadratic_root(cell):
            continue
        local_cell = local[local_cursor] if quadratic_root is not None else local[cell]
        local_cursor += int(quadratic_root is not None)
        if multilinear is not None:
            with ledger.temporary_scope():
                result = _certify_multilinear_cell(
                    multilinear, local_cell, dimension, budget
                )
            if result is not None:
                status[cell], lower[cell], upper[cell], depth[cell] = (
                    result.status,
                    result.lower,
                    result.upper,
                    result.depth,
                )
                continue
        # Leaf Bernstein controls are consumed within their cell; earlier
        # request-retained preparations stay reusable.
        with ledger.polynomial_preparation_scope():
            result = (
                None
                if factors is None
                else _certify_factored_cell(
                    factors,
                    element,
                    local_cell,
                    reference,
                    dimension,
                    policy,
                    budget,
                )
            )
            if result is None:
                result = _certify_polynomial_cell(
                    element,
                    local_cell,
                    reference,
                    dimension,
                    embedded,
                    policy,
                    budget,
                    chart,
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
    native_staging_upper = 0
    native_template_identities: set[str] = set()
    native_template_upper = 0
    if (
        source is None
        and geometry.periodic_source is None
        and geometry.restriction_source is None
        and values.ndim == 2
        and values.shape[1] == 3
        and policy.maximum_bernstein_nodes <= (1 << 63) - 1
        and policy.maximum_subdivision_depth <= (1 << 63) - 1
        and meshcore_owner.meshcore_available()
    ):
        for element, routes in zip(geometry.elements, host_routes, strict=True):
            route_array = np.asarray(routes)
            if route_array.ndim != 2 or route_array.shape[1] != 10:
                continue
            count = route_array.shape[0]
            if (
                _canonical_quadratic_tetrahedron_element(element) is None
                or count == 0
                or count > ((1 << 63) - 1) // 30
                or count * 30 * 8 > np.iinfo(np.intp).max
            ):
                continue
            host_staging, native_staging = (
                _quadratic_tetrahedron_staging_bounds(count)
            )
            native_staging_upper = max(
                native_staging_upper, host_staging + native_staging
            )
            if element.element_id not in native_template_identities:
                native_template_identities.add(element.element_id)
                native_template_upper += 4096
    return (
        source_work_upper + algebraic_work,
        source_memory_upper
        + algebraic_memory
        + native_staging_upper
        + native_template_upper,
    )


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
    # Source-bank retention owns its key and tuple structure even without
    # numerical rows; cell-count-scaled proof circuits do not cover that graph.
    retention_work, retention_memory = coordinate_retention_preparation_bounds(
        (coordinate_source_bank_key(geometry), ())
    )
    work += retention_work
    memory += retention_memory
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
            source_values = coordinate_enclosure_owner.prepared_coordinate_source_bank(
                spec, budget=budget
            )
        case (
            None
            | exact_plc_geometry_owner.ExactPlcCellGeometrySource()
            | exact_plc_geometry_owner.ExactPlcCellGeometryConvexSource()
            | exact_rational_geometry_owner.ExactRationalCellGeometrySource()
            | exact_algebraic_geometry_owner.ExactAlgebraicCellGeometrySource()
        ):
            # The PLC bank defines the polynomial/rational map.
            source_values = coordinate_enclosure_owner.prepared_coordinate_source_bank(
                spec, budget=budget
            )
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
            # The request-owned determinant is reused by power measures and
            # refinement on the same ordered corners.
            determinant = exact_polyhedral_geometry_owner.star_determinant(
                tetrahedron, budget=budget
            )
            columns = tuple(
                tuple(
                    value - base
                    for value, base in zip(point, tetrahedron[0], strict=True)
                )
                for point in tetrahedron[1:]
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


_AFFINE_SIMPLEX_DIMENSIONS = {"interval": 1, "triangle": 2, "tetrahedron": 3}


def _affine_simplex_measure_operations(dimension: int, ambient: int, /) -> int:
    """The existing cofactor or Gram arithmetic, excluding norms and policy."""
    if ambient == dimension:
        return (0, 0, 3, 14)[dimension]
    return 0 if dimension == 1 else 2 * ambient + 2


def _affine_simplex_operations(dimension: int, ambient: int, /) -> int:
    """Rational operations of one constant affine-simplex determinant decision.

    Edge differences and column norms precede the cofactor
    determinant of a square map, or the Gram determinant of the column norms
    and edge dot product of an embedded map. The norm product, policy threshold,
    squared comparison, sign and the exact value's two directed endpoints follow.
    """
    differences = dimension * ambient
    norms = dimension * (2 * ambient - 1)
    measure = _affine_simplex_measure_operations(dimension, ambient)
    return differences + norms + measure + (dimension - 1) + 2 + 4 + 3


def _simplex_determinant(rows: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    match len(rows):
        case 1:
            return rows[0][0]
        case 2:
            (a, b), (c, d) = rows
            return a * d - b * c
        case 3:
            (a, b, c), (d, e, f), (g, h, i) = rows
            return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
        case count:
            raise ValueError(
                f"affine_simplex.determinant must have one to three rows; got {count}"
            )


def _affine_simplex_measure(
    edges: tuple[tuple[Fraction, ...], ...],
    /,
    *,
    embedded: bool,
    norms: list[Fraction] | None = None,
) -> Fraction:
    """Shared exact scalar determinant/Gram primitive for affine source proofs."""
    if not embedded:
        return _simplex_determinant(edges)
    if norms is None:
        norms = [
            sum((value * value for value in edge), Fraction(0)) for edge in edges
        ]
    if len(edges) == 1:
        return norms[0]
    dot = sum(
        (first * second for first, second in zip(edges[0], edges[1], strict=True)),
        Fraction(0),
    )
    return norms[0] * norms[1] - dot * dot


def _strict_simplex_floor_from_edge_bounds(
    lower: np.ndarray, edge_magnitudes: np.ndarray, floor: float, /,
) -> np.ndarray:
    """Apply the unchanged strict Hadamard-floor comparison to real enclosures."""
    def upward(value: np.ndarray) -> np.ndarray:
        return np.maximum(np.nextafter(value, np.inf), np.finfo(np.float64).tiny)

    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        edges = upward(edge_magnitudes)
        squares = upward(edges * edges)
        norms = squares[:, :, 0]
        for axis in range(1, edges.shape[2]):
            norms = upward(norms + squares[:, :, axis])
        scale = norms[:, 0]
        for column in range(1, norms.shape[1]):
            scale = upward(scale * norms[:, column])
        floor_squared = max(
            float(np.nextafter(floor * floor, np.inf)), np.finfo(np.float64).tiny
        )
        threshold = upward(floor_squared * scale)
        squared = np.nextafter(lower * lower, -np.inf)
    return (
        (lower > 0.0)
        & np.isfinite(lower)
        & np.isfinite(scale)
        & np.isfinite(threshold)
        & (squared > threshold)
    )


def _rational_planar_simplex_floor(
    frames: tuple[tuple[tuple[Fraction, ...], ...], ...],
    floor: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Filter actual rational 2D frames, never their rounded source carriers.

    Directed scalar endpoints and the canonical interval operations enclose the
    already formed exact Jacobian. Every nontrivial interval-operation endpoint
    must stay normal: overflow and subnormal/possibly flushed extrema decline
    the filter. Only algebraically exact zero products and zero subtrahends
    bypass arithmetic. Exact determinants, published endpoints and all
    unresolved/near-floor decisions remain the ordinary exact circuit.
    """
    from ..geometry._interval_enclosure import interval_multiply, interval_subtract

    count = len(frames)
    with budget.live_storage() as storage:
        # Two four-lane source bounds, zero/range masks, gathered interval
        # products, determinant and shared norm/threshold temporaries coexist.
        # 512 bytes per frame covers their complete array buffers; fixed view,
        # tuple and ndarray headers have the separate 8192-byte allowance.
        base = 8192 + 512 * count
        storage.set_bound(base)
        budget.reserve(count)
        endpoints: dict[tuple[int, int], tuple[float, float]] = {}
        lower = np.zeros((count, 2, 2), dtype=np.float64)
        upper = np.zeros_like(lower)
        zeros = np.zeros(lower.shape, dtype=np.bool_)
        for row, frame in enumerate(frames):
            for column, edge in enumerate(frame):
                for axis, value in enumerate(edge):
                    key = (value.numerator, value.denominator)
                    bounds = endpoints.get(key)
                    if bounds is None:
                        storage.set_bound(base + 384 * (len(endpoints) + 1))
                        budget.reserve(1)  # Actual new numeric-cache owner.
                        if value.numerator == 0:
                            bounds = (0.0, 0.0)
                        else:
                            budget.reserve(2)  # The same two directed endpoints.
                            bounds = (
                                fraction_endpoint(value, upper=False, saturate=True),
                                fraction_endpoint(value, upper=True, saturate=True),
                            )
                        endpoints[key] = bounds
                    lower[row, column, axis], upper[row, column, axis] = bounds
                    zeros[row, column, axis] = value.numerator == 0

        def normal(values: np.ndarray) -> np.ndarray:
            # Integer bit inspection cannot itself flush a subnormal to zero.
            exponent = (values.view(np.uint64) >> np.uint64(52)) & np.uint64(0x7FF)
            return (exponent > 0) & (exponent < 0x7FF)

        safe = np.all(
            zeros | (normal(lower) & normal(upper)), axis=(1, 2),
        )

        def multiply(
            first: tuple[np.ndarray, np.ndarray],
            second: tuple[np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            zero = ((first[0] == 0) & (first[1] == 0)) | (
                (second[0] == 0) & (second[1] == 0)
            )
            active = np.flatnonzero(safe & ~zero)
            low, high = np.zeros(count), np.zeros(count)
            budget.reserve(int(active.size))
            if active.size:
                actual_low, actual_high = interval_multiply(
                    (first[0][active], first[1][active]),
                    (second[0][active], second[1][active]),
                )
                actual_low, actual_high = np.asarray(actual_low), np.asarray(actual_high)
                low[active], high[active] = actual_low, actual_high
                safe[active] &= normal(actual_low) & normal(actual_high)
            return low, high

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            first = multiply(
                (lower[:, 0, 0], upper[:, 0, 0]),
                (lower[:, 1, 1], upper[:, 1, 1]),
            )
            second = multiply(
                (lower[:, 0, 1], upper[:, 0, 1]),
                (lower[:, 1, 0], upper[:, 1, 0]),
            )
            zero_second = (second[0] == 0) & (second[1] == 0)
            active = np.flatnonzero(safe & ~zero_second)
            determinant_lower = first[0].copy()
            budget.reserve(int(active.size))
            if active.size:
                actual_low, actual_high = interval_subtract(
                    (first[0][active], first[1][active]),
                    (second[0][active], second[1][active]),
                )
                actual_low, actual_high = np.asarray(actual_low), np.asarray(actual_high)
                determinant_lower[active] = actual_low
                safe[active] &= normal(actual_low) & normal(actual_high)
            magnitudes = np.maximum(np.abs(lower), np.abs(upper))
            return safe & _strict_simplex_floor_from_edge_bounds(
                determinant_lower, magnitudes, floor,
            )


def _binary64_simplex_floor(
    source_values: coordinate_enclosure_owner.CoordinateSourceBank,
    corners: np.ndarray,
    floor: float,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Prove the relative floor cheaply; never replace exact determinant bounds."""
    # Reuse the equality-certified source view that embedding and coverage
    # consume later in this same ledger, rather than trusting mesh carriers.
    from ..geometry._global_embedding import source_bank_binary64_rows

    view, represented = source_bank_binary64_rows(source_values, budget=budget)
    budget.reserve(0, 16 * corners.shape[0])
    result = np.zeros((corners.shape[0],), dtype=np.bool_)
    selected = np.flatnonzero(np.all(represented[corners], axis=1))
    if not selected.size:
        return result
    count = selected.size
    budget.reserve(count, 1024 * count)
    points = view[corners[selected]]
    if points.shape[2] == 2:
        lower, _ = geometry_predicates_owner.orient2d_enclosure_host(
            points[:, 0], points[:, 1], points[:, 2], budget=budget
        )
    else:
        lower, _ = geometry_predicates_owner.orient3d_enclosure_host(
            points[:, 0], points[:, 1], points[:, 2], points[:, 3], budget=budget
        )
    # The same directed norm bound and strict policy comparison also consume
    # rational-frame interval enclosures; binary64 arithmetic order is unchanged.
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        result[selected] = _strict_simplex_floor_from_edge_bounds(
            lower, np.abs(points[:, 1:] - points[:, :1]), floor,
        )
    return result


def _affine_simplex_block(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    route: np.ndarray,
    source_values: coordinate_enclosure_owner.CoordinateSourceBank,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _BlockCertificate | None:
    """Decide constant P1 simplex Jacobians exactly under one block admission.

    Applies when the canonical source proof shows the degree-one simplex map
    affine and every reference corner is a cardinal coefficient row, so each
    cell's Jacobian columns are exact source-bank edge differences. The
    polynomial route's level-zero piece of such a map is this constant: its one
    Bernstein control and every corner value equal the determinant (Gram
    determinant when embedded), and its Hadamard scale is the product of the
    squared column norms. Status, outward endpoints and depth zero therefore
    coincide. Cells with a nonrational corner, or a value exactly at the policy
    floor (which the polynomial route subdivides), take the full polynomial route.
    A binary64-equal source view, or directed rational 2D Jacobian intervals,
    may prove the relative floor by an orientation enclosure and outward norm
    bounds. Only exact norm and threshold arithmetic is then omitted: the
    constant determinant and its directed endpoints still use the exact circuit.
    """
    dimension = _AFFINE_SIMPLEX_DIMENSIONS.get(cell_kind)
    count = route.shape[0]
    if (
        dimension is None
        or count == 0
        or not source_values
        or not isinstance(element, reference_types_owner.FiniteElementSpec)
        or element.cell_kind != cell_kind
        or element.degree != 1
        or element.local_dof_count != dimension + 1
        or route.ndim != 2
        or route.shape[1] != element.local_dof_count
        or not coordinate_enclosure_owner.affine_simplex_source(element, budget=budget)
    ):
        return None
    nodes = cell_geometry_owner._cardinal_corner_nodes(element, budget=budget)
    ambient = len(source_values[0])
    if (
        nodes is None
        or len(nodes) != dimension + 1
        or ambient < dimension
        or (ambient != dimension and dimension == 3)
    ):
        return None
    embedded = ambient != dimension
    corner_rows = np.asarray(route, dtype=np.int64)[:, list(nodes)]
    used = np.unique(corner_rows)
    floor = Fraction(policy.relative_determinant_floor)
    floor_numerator = floor.numerator * floor.numerator
    floor_denominator = floor.denominator * floor.denominator
    status = np.full((count,), CellValidityStatus.CERTIFIED_VALID, dtype=np.int32)
    lower = np.full((count,), np.nan, dtype=np.float64)
    upper = np.full((count,), np.nan, dtype=np.float64)
    pending: list[int] = []
    with budget.temporary_scope(), budget.live_storage() as proof_storage:
        budget.reserve(0, count * (256 + 64 * (dimension + 1)) + used.size * 16)
        rational = np.zeros((used.size,), dtype=np.bool_)
        bits = exact_algebraic_geometry_owner.coefficient_bits(floor)
        for position, row in enumerate(used.tolist()):
            point = source_values[row]
            if len(point) == ambient and all(type(value) is Fraction for value in point):
                rational[position] = True
                bits = max(
                    bits,
                    *(exact_algebraic_geometry_owner.coefficient_bits(v) for v in point),
                )
        attempted = np.all(rational[np.searchsorted(used, corner_rows)], axis=1)
        filtered = np.zeros((count,), dtype=np.bool_)
        # Form each physical Jacobian once per cell, directly in its canonical
        # rational representation. Equal exact edge tuples then share the
        # determinant, scale and policy decision; translations do not change it.
        differences = dimension * ambient
        edge_bits = 2 * bits + 2
        # An embedded triangle's Gram decision depends only on its three exact
        # squared edge lengths a=|p1-p0|^2, b=|p2-p0|^2, c=|p2-p1|^2: the
        # column dot product is (a + b - c) / 2 exactly and the Hadamard scale
        # is ab. Each undirected source-bank edge is formed and normed once.
        shared_edges = embedded and dimension == 2
        norm_bits = 2 * edge_bits + ambient
        edge_norms: dict[tuple[int, int], Fraction] = {}
        edge_entry_bytes = 512 + 2 * (32 + 4 * ((norm_bits + 29) // 30))
        decisions: dict[
            tuple[tuple[Fraction, ...], ...] | tuple[Fraction, ...],
            tuple[int, float, float] | None,
        ] = {}
        # Fraction object, two bounded integers, tuple slots and dictionary
        # capacity. Scratch for a decision is released independently below.
        entry_bytes = 512 + differences * (
            128 + 2 * (32 + 4 * ((edge_bits + 29) // 30))
        )
        # Equal exact edge frames already share the final rational decision.
        # Form that key once and query the binary64 floor only for its first
        # representative, rather than re-querying every translated cell.
        cell_keys: list[tuple[tuple[Fraction, ...], ...] | None] = []
        retained_key_bytes = 0
        if not shared_edges:
            budget.reserve(0, 512 + 8 * count)
            cell_keys = [None] * count
            representatives: dict[tuple[tuple[Fraction, ...], ...], int] = {}
            differences_by_value: dict[tuple[int, int, int, int], Fraction] = {}
            difference_entry_bytes = 384 + 2 * (
                32 + 4 * ((edge_bits + 29) // 30)
            )

            def difference(value: Fraction, base: Fraction) -> Fraction:
                key = (
                    value.numerator, value.denominator,
                    base.numerator, base.denominator,
                )
                cached = differences_by_value.get(key)
                if cached is None:
                    proof_storage.set_bound(
                        (len(representatives) + 1) * entry_bytes
                        + (len(differences_by_value) + 1) * difference_entry_bytes
                    )
                    # A genuinely new cache entry and one actual subtraction.
                    # Repeated inputs only route their existing numerical
                    # operand, just as the complete exact frame cache below.
                    budget.reserve(1)
                    reserve_polynomial(1, 1, 0, edge_bits, budget=budget)
                    cached = value - base
                    differences_by_value[key] = cached
                return cached

            for cell, corners in enumerate(corner_rows.tolist()):
                if not attempted[cell]:
                    continue
                proof_storage.set_bound(
                    (len(representatives) + 1) * entry_bytes
                    + len(differences_by_value) * difference_entry_bytes
                )
                origin = source_values[corners[0]]
                candidate = tuple(
                    tuple(
                        difference(cast(Fraction, value), cast(Fraction, base))
                        for value, base in zip(source_values[row], origin, strict=True)
                    )
                    for row in corners[1:]
                )
                previous = representatives.get(candidate)
                if previous is None:
                    representatives[candidate] = cell
                    cell_keys[cell] = candidate
                else:
                    cell_keys[cell] = cell_keys[previous]
            retained_key_bytes = (
                len(representatives) * entry_bytes
                + len(differences_by_value) * difference_entry_bytes
            )
            selected = np.fromiter(
                representatives.values(), dtype=np.int64, count=len(representatives)
            )
            if dimension == ambient and dimension in (2, 3) and selected.size:
                filtered[selected] = _binary64_simplex_floor(
                    source_values,
                    corner_rows[selected],
                    policy.relative_determinant_floor,
                    budget=budget,
                )
                if dimension == 2:
                    from ..geometry._global_embedding import source_bank_binary64_rows

                    _, represented = source_bank_binary64_rows(source_values, budget=budget)
                    rational_rows = selected[
                        ~np.all(represented[corner_rows[selected]], axis=1)
                    ]
                    if rational_rows.size:
                        frames = tuple(
                            cast(tuple[tuple[Fraction, ...], ...], cell_keys[int(row)])
                            for row in rational_rows
                        )
                        filtered[rational_rows] = _rational_planar_simplex_floor(
                            frames, policy.relative_determinant_floor, budget=budget,
                        )

        def edge_norm(first: int, second: int, /) -> Fraction:
            key = (min(first, second), max(first, second))
            value = edge_norms.get(key)
            if value is None:
                # Ambient differences and the squared-norm products and sums.
                reserve_polynomial(
                    3 * ambient - 1, ambient + 1, 0, norm_bits, budget=budget
                )
                value = sum(
                    (
                        (cast(Fraction, end) - cast(Fraction, start)) ** 2
                        for start, end in zip(
                            source_values[key[0]], source_values[key[1]], strict=True
                        )
                    ),
                    Fraction(0),
                )
                edge_norms[key] = value
            return value

        for cell, (corners, admitted) in enumerate(
            zip(corner_rows.tolist(), attempted.tolist(), strict=True)
        ):
            if not admitted:
                pending.append(cell)
                continue
            if shared_edges:
                proof_storage.set_bound(
                    len(decisions) * entry_bytes + (len(edge_norms) + 3) * edge_entry_bytes
                )
                key: tuple[tuple[Fraction, ...], ...] | tuple[Fraction, ...] = (
                    edge_norm(corners[0], corners[1]),
                    edge_norm(corners[0], corners[2]),
                    edge_norm(corners[1], corners[2]),
                )
            else:
                actual_key = cell_keys[cell]
                if actual_key is None:
                    raise RuntimeError("Affine validity must retain its admitted exact edge frame.")
                key = actual_key
            if key not in decisions and filtered[cell]:
                proof_storage.set_bound(
                    max(retained_key_bytes, (len(decisions) + 1) * entry_bytes)
                    + len(edge_norms) * edge_entry_bytes
                )
                with budget.temporary_scope():
                    # Exact cofactor determinant plus its two directed
                    # endpoints. The issued filter already proved the sign
                    # and floor, so no rational norm/threshold is recomputed.
                    reserve_polynomial(
                        _affine_simplex_measure_operations(dimension, ambient) + 2,
                        differences + 8,
                        0,
                        dimension * edge_bits + 8,
                        budget=budget,
                    )
                    measure = _simplex_determinant(
                        cast(tuple[tuple[Fraction, ...], ...], key)
                    )
                    decisions[key] = (
                        int(CellValidityStatus.CERTIFIED_VALID),
                        coefficient_endpoint(measure, upper=False, saturate=True),
                        coefficient_endpoint(measure, upper=True, saturate=True),
                    )
            if key not in decisions:
                proof_storage.set_bound(
                    max(retained_key_bytes, (len(decisions) + 1) * entry_bytes)
                    + len(edge_norms) * edge_entry_bytes
                )
                with budget.temporary_scope():
                    if shared_edges:
                        # Polarized dot (three), Gram (three), then the shared
                        # norm product, threshold, comparison and endpoints.
                        reserve_polynomial(
                            16,
                            dimension + 8,
                            0,
                            4 * dimension * (edge_bits + ambient + 1) + 2 * bits + 8,
                            budget=budget,
                        )
                        first, second, third = cast(tuple[Fraction, ...], key)
                        dot = (first + second - third) / 2
                        scale = first * second
                        measure = scale - dot * dot
                    else:
                        reserve_polynomial(
                            _affine_simplex_operations(dimension, ambient) - differences,
                            differences + dimension + 8,
                            0,
                            4 * dimension * (edge_bits + ambient + 1) + 2 * bits + 8,
                            budget=budget,
                        )
                        edges = cast(tuple[tuple[Fraction, ...], ...], key)
                        norms = [
                            sum((value * value for value in edge), Fraction(0))
                            for edge in edges
                        ]
                        scale = math.prod(norms)
                        measure = _affine_simplex_measure(
                            edges, embedded=embedded, norms=norms
                        )
                    if embedded:
                        threshold = floor_numerator * scale * scale
                    else:
                        threshold = floor_numerator * scale
                    squared = measure * measure * floor_denominator
                    if measure <= 0 or squared < threshold:
                        decision_status = CellValidityStatus.INVALID
                    elif squared == threshold:
                        decisions[key] = None
                        pending.append(cell)
                        continue
                    else:
                        decision_status = CellValidityStatus.CERTIFIED_VALID
                    decisions[key] = (
                        int(decision_status),
                        coefficient_endpoint(measure, upper=False, saturate=True),
                        coefficient_endpoint(measure, upper=True, saturate=True),
                    )
            decision = decisions[key]
            if decision is None:
                pending.append(cell)
            else:
                status[cell], lower[cell], upper[cell] = decision
            if cell % 64 == 0:
                budget.reserve(0)
    depth = np.zeros((count,), dtype=np.int32)
    reasons: tuple[CellValidityUnresolvedReason, ...] = ()
    if pending:
        fallback = _certify_polynomial_block(
            element,
            cell_kind,
            tuple(
                tuple(source_values[index] for index in route[cell]) for cell in pending
            ),
            policy,
            budget=budget,
        )
        selected = np.asarray(pending, dtype=np.int64)
        status[selected] = fallback.status
        lower[selected] = fallback.lower
        upper[selected] = fallback.upper
        depth[selected] = fallback.depth
        reasons = fallback.reasons
    return _BlockCertificate(status, lower, upper, depth, reasons)


def _affine_chart_simplex_block(
    element: cell_geometry_owner.ScalarCoordinateElement,
    cell_kind: str,
    route: np.ndarray,
    source_values: coordinate_enclosure_owner.CoordinateSourceBank,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
    *,
    corner_bank: coordinate_enclosure_owner.CoordinateSourceBank | None = None,
    corner_routes: np.ndarray | None = None,
) -> _BlockCertificate | None:
    """Decide an affine simplex chart block on its exact corner images.

    A simplex restriction, barycentric or degree-one composed chart over an
    affine degree-one root is affine, hence equal to the degree-one
    interpolant of its exact corner images on the same reference simplex.
    Its Jacobian, Bernstein control and policy decision are therefore those of
    that interpolant. Every distinct corner operation is evaluated once and
    the degree-one block decision applies, including its polynomial fallback
    for cells exactly at the policy floor.
    """
    if (
        isinstance(element, reference_types_owner.FiniteElementSpec)
        or cell_kind not in _AFFINE_SIMPLEX_DIMENSIONS
        or element.cell_kind != cell_kind
        or route.ndim != 2
        or route.shape[0] == 0
        or not coordinate_enclosure_owner.affine_simplex_source(element, budget=budget)
    ):
        return None
    if corner_bank is not None and corner_routes is not None:
        if corner_routes.shape != (
            route.shape[0], _AFFINE_SIMPLEX_DIMENSIONS[cell_kind] + 1
        ):
            raise ValueError(
                "cell_geometry_validity.corner_routes must retain the actual changed cell corners; got a different shape"
            )
        return _affine_simplex_block(
            budget.degree_one_coordinate_element(cell_kind),
            cell_kind,
            corner_routes,
            corner_bank,
            policy,
            budget,
        )
    count = route.shape[0]
    corners = _AFFINE_SIMPLEX_DIMENSIONS[cell_kind] + 1
    rows = np.arange(count * corners, dtype=np.int64).reshape((count, corners))
    images = coordinate_enclosure_owner.coordinate_vertex_images(
        ((element, np.asarray(route, dtype=np.int64), rows),),
        source_values,
        count * corners,
        mismatch="cell_geometry_validity.corner_images must bind one image per cell corner; got a repeated corner",
        budget=budget,
    )
    if images is None:
        return None
    bank: list[tuple[coordinate_enclosure_owner.ExactScalar, ...]] = []
    for image in images:
        if image is None:
            return None
        bank.append(image)
    return _affine_simplex_block(
        budget.degree_one_coordinate_element(cell_kind),
        cell_kind,
        rows,
        tuple(bank),
        policy,
        budget,
    )



class _ValidityBlocks(StrictModule):
    blocks: tuple[_BlockCertificate, ...]
    offsets: tuple[int, ...]
    unsupported: tuple[str, ...]


RestrictedValidityRefusal: TypeAlias = Literal[
    "wrong_parent",
    "coefficients",
    "chart",
    "not_contained",
    "orientation",
    "partial_proof",
    "scale",
    "unsupported",
]
"""Typed reason a restricted child does not import its parent's validity."""


@dataclass(frozen=True)
class _RestrictionParent:
    """One certified predecessor cell that may own restricted target children."""

    element: cell_geometry_owner.ScalarCoordinateElement
    kind: str
    local: coordinate_enclosure_owner.CoordinateSourceBank
    row: int


@dataclass(frozen=True)
class _TensorGridSource:
    """Borrowed roots for one uninterrupted tensor proof, never reusable authority."""

    source_mesh: cell_mesh_owner.CellMesh
    source_geometry: cell_geometry_owner.CellGeometrySpec
    source_validity: CellValidityCertificate


@dataclass(frozen=True)
class _ValidityImport:
    rows: np.ndarray
    status: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    depth: np.ndarray
    predecessor: CellGeometryPredecessor | _TensorGridSource
    parents: dict[int, tuple[_RestrictionParent, ...]]
    parent_jacobians: dict[
        int, tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...] | None
    ]
    construction: _ConstructedTensorGrid | None = None


def _prepare_validity_import(
    predecessor: CellGeometryPredecessor | _CellGeometryPredecessorSource,
    mesh: cell_mesh_owner.CellMesh | None,
    geometry: cell_geometry_owner.CellGeometrySpec,
    policy: CellValidityPolicy,
    *,
    budget: CoordinateEnclosureBudget | None,
) -> _ValidityImport:
    """Authenticate and consume ordinary owners or freshly prepared restrictions.

    The source-input branch owns preparation and import without yielding,
    awaiting or calling a supplied continuation. Its supported immutable owner
    tree was just fully earned on this ledger; it cannot escape between those
    operations. Inputs carry no authority: repeating the call prepares again.
    Ordinary predecessors and unsupported mutable owners always reauthenticate.
    """
    if mesh is None:
        raise ValueError(
            "certify_cell_geometry_validity.predecessor must provide its target mesh; got no mesh."
        )
    if isinstance(predecessor, _CellGeometryPredecessorSource):
        if budget is None:
            raise ValueError("Atomic predecessor preparation requires its caller budget")
        prepare = prepare_cell_geometry_predecessor
        predecessor = prepare(
            predecessor.mesh, predecessor.geometry, predecessor.validity, None,
            mesh, geometry, budget=budget,
        )
        # This is not a reusable admission check: the predecessor is local to
        # this uninterrupted preparation/import operation. Every general reuse
        # still goes through require_bound's complete current-owner traversal.
        budget.reserve(1)
        if (
            prepare is not _ATOMIC_PREDECESSOR_PREPARER
            or id(predecessor) not in budget.predecessor_content_admission_cache
        ):
            predecessor.require_bound(mesh, geometry, budget=budget)
    else:
        predecessor.require_bound(mesh, geometry, budget=budget)
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
        predecessor,
        {},
        {},
    )


def _restriction_parents(
    imported: _ValidityImport, budget: CoordinateEnclosureBudget, /
) -> dict[int, tuple[_RestrictionParent, ...]]:
    """Index predecessor cells by the root origin their restricted children name.

    A root predecessor names its own global cell IDs; a restricted predecessor
    names the root origins of its own restriction record.
    """
    if imported.parents:
        return imported.parents
    mesh = imported.predecessor.source_mesh
    geometry = imported.predecessor.source_geometry
    count = sum(block.cell_count for block in mesh.blocks)
    budget.reserve(count, 128 + 96 * count)
    elements, routes, _ = geometry.resolve(mesh, budget=budget)
    bank = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        geometry, budget=budget
    )
    record = geometry.restriction_source
    parents: dict[int, list[_RestrictionParent]] = {}
    flat = 0
    for block, element, route in zip(mesh.blocks, elements, routes, strict=True):
        origins = np.asarray(
            block.global_ids
            if record is None
            else record.block_parent_cell_ids[block.name],
            dtype=np.int64,
        ).tolist()
        rows = np.asarray(route, dtype=np.int64).tolist()
        for origin, row in zip(origins, rows, strict=True):
            budget.reserve(len(row))
            parents.setdefault(origin, []).append(
                _RestrictionParent(
                    cast(cell_geometry_owner.ScalarCoordinateElement, element),
                    block.cell_kind,
                    tuple(bank[index] for index in row),
                    flat,
                )
            )
            flat += 1
    imported.parents.update(
        (origin, tuple(members)) for origin, members in parents.items()
    )
    return imported.parents


type _AffineMap = tuple[tuple[tuple[Fraction, ...], ...], tuple[Fraction, ...]]


def _restriction_root_map(
    element: cell_geometry_owner.ScalarCoordinateElement,
    stop: str | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[cell_geometry_owner.ScalarCoordinateElement, _AffineMap]:
    """Compose exact stored affine restriction layers down to ``stop`` or a root.

    Binary64 restriction entries and polynomial chart coefficients stay exact.
    """
    dimension = element.topological_dimension
    matrix = tuple(
        tuple(Fraction(int(row == column)) for column in range(dimension))
        for row in range(dimension)
    )
    offset = (Fraction(0),) * dimension
    current = element
    while current.element_id != stop:
        if isinstance(current, cell_geometry_owner.RestrictedCellGeometryElement):
            layer = np.asarray(current.matrix, dtype=np.float64)
            rows = tuple(tuple(Fraction(float(value)) for value in row) for row in layer)
            shift = tuple(Fraction(float(value)) for value in np.asarray(current.offset))
        elif (
            isinstance(current, cell_geometry_owner.PolynomialComposedCellGeometryElement)
            and current.chart_element.degree == 1
        ):
            arguments = coordinate_enclosure_owner.reference_composition_arguments(
                current, budget=budget
            )
            if arguments is None or len(arguments) != dimension or any(
                type(value) is not Fraction or sum(index) > 1
                for polynomial in arguments for index, value in polynomial.items()
                if value != 0
            ):
                break
            budget.reserve(sum(map(len, arguments)) + dimension * (dimension + 1))
            rows = tuple(
                tuple(
                    cast(Fraction, polynomial.get(
                        tuple(int(axis == column) for axis in range(dimension)),
                        Fraction(0),
                    ))
                    for column in range(dimension)
                )
                for polynomial in arguments
            )
            shift = tuple(
                cast(Fraction, polynomial.get((0,) * dimension, Fraction(0)))
                for polynomial in arguments
            )
        else:
            break
        matrix = tuple(
            tuple(
                sum((row[k] * matrix[k][column] for k in range(len(row))), Fraction(0))
                for column in range(dimension)
            )
            for row in rows
        )
        offset = tuple(
            sum((row[k] * offset[k] for k in range(len(row))), Fraction(0))
            + start
            for row, start in zip(rows, shift, strict=True)
        )
        current = current.source_element
    return current, (matrix, offset)


def _fraction_determinant(matrix: tuple[tuple[Fraction, ...], ...], /) -> Fraction:
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return sum(
        (
            (-1) ** column
            * matrix[0][column]
            * _fraction_determinant(
                tuple(
                    tuple(value for index, value in enumerate(row) if index != column)
                    for row in matrix[1:]
                )
            )
            for column in range(size)
        ),
        Fraction(0),
    )


def _fraction_inverse(
    matrix: tuple[tuple[Fraction, ...], ...], /
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Exact Gauss-Jordan inverse of a small square rational matrix."""
    size = len(matrix)
    if any(len(row) != size for row in matrix):
        return None
    rows = [
        [*row, *(Fraction(int(index == column)) for column in range(size))]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            (index for index in range(column, size) if rows[index][column] != 0), None
        )
        if pivot is None:
            return None
        rows[column], rows[pivot] = rows[pivot], rows[column]
        scale = rows[column][column]
        rows[column] = [value / scale for value in rows[column]]
        for index in range(size):
            factor = rows[index][column]
            if index != column and factor != 0:
                rows[index] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(rows[index], rows[column], strict=True)
                ]
    return tuple(tuple(row[size:]) for row in rows)


def _restriction_parent_chart(
    child: cell_geometry_owner.ScalarCoordinateElement,
    parent: cell_geometry_owner.ScalarCoordinateElement,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> _AffineMap | None:
    """Exact affine ``S`` with ``child = parent o S``, if both restrict one root."""
    child_root, (child_matrix, child_offset) = _restriction_root_map(
        child, parent.element_id, budget=budget
    )
    if child_root.element_id == parent.element_id:
        if coordinate_enclosure_owner.coordinate_source_signature(
            child_root
        ) != coordinate_enclosure_owner.coordinate_source_signature(parent):
            return None
        return child_matrix, child_offset
    parent_root, (parent_matrix, parent_offset) = _restriction_root_map(
        parent, None, budget=budget
    )
    if (
        child_root.element_id != parent_root.element_id
        or coordinate_enclosure_owner.coordinate_source_signature(child_root)
        != coordinate_enclosure_owner.coordinate_source_signature(parent_root)
    ):
        return None
    inverse = _fraction_inverse(parent_matrix)
    if inverse is None:
        return None
    size = len(inverse)
    matrix = tuple(
        tuple(
            sum(
                (inverse[row][k] * child_matrix[k][column] for k in range(size)),
                Fraction(0),
            )
            for column in range(len(child_matrix[0]))
        )
        for row in range(size)
    )
    offset = tuple(
        sum(
            (
                inverse[row][k] * (child_offset[k] - parent_offset[k])
                for k in range(size)
            ),
            Fraction(0),
        )
        for row in range(size)
    )
    return matrix, offset


def _inside_parent_reference(point: tuple[Fraction, ...], kind: str, /) -> bool:
    """Closed convex reference-cell membership of an exact point."""
    if any(value < 0 or value > 1 for value in point):
        return False
    family = _cell_kind_family(kind)
    if family == "simplex":
        return sum(point, Fraction(0)) <= 1
    if family == "prism":
        return point[0] + point[1] <= 1
    return family == "box"


def _cell_kind_family(kind: str, /) -> str:
    """Reference family of a named or dimension-qualified cell kind."""
    if kind in ("interval", "triangle", "tetrahedron") or kind.startswith("simplex:"):
        return "simplex"
    if kind in ("quadrilateral", "hexahedron") or kind.startswith("tensor:"):
        return "box"
    return kind


def _physical_jacobian(
    element: cell_geometry_owner.ScalarCoordinateElement,
    local: coordinate_enclosure_owner.CoordinateSourceBank,
    cell_kind: str,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...] | None:
    """One cell's full physical differential, without its determinant."""
    coefficients, _ = coordinate_enclosure_owner.coordinate_preparation_key(
        element, local, budget=budget
    )
    coordinates = coordinate_enclosure_owner.coordinate_polynomials(
        element, coefficients, budget=budget
    )
    if coordinates is not None:
        return coordinate_enclosure_owner.physical_jacobian(
            coordinates, cell_kind, dimension, budget=budget
        )
    expressions = coordinate_enclosure_owner.coordinate_expressions(
        element, coefficients, budget=budget
    )
    if expressions is None:
        return None
    return coordinate_enclosure_owner.expression_physical_jacobian(
        expressions, cell_kind, dimension, budget=budget
    )


def _restricted_child_jacobian(
    parent: _RestrictionParent,
    imported: _ValidityImport,
    chart: _AffineMap,
    cell_kind: str,
    dimension: int,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[coordinate_enclosure_owner.Expression, ...], ...] | None:
    """Exact chain rule ``D child(xi) = D parent(S xi) M`` for ``S = M xi + o``.

    ``child = parent o S`` holds coefficientwise, so this is the child's own
    physical differential polynomial; the parent factor is earned once.
    """
    if parent.row not in imported.parent_jacobians:
        parent_reference = _prepare_polynomial_reference_domain(
            parent.kind, dimension, parent.element.degree, False
        )
        earned = (
            None
            if parent_reference is None
            else _physical_jacobian(
                parent.element,
                parent.local,
                parent_reference.cell_kind,
                dimension,
                budget,
            )
        )
        if earned is not None:
            budget.retain_basis((earned,))
        imported.parent_jacobians[parent.row] = earned
    parent_jacobian = imported.parent_jacobians[parent.row]
    if parent_jacobian is None:
        return None
    matrix, offset = chart
    values = (*offset, *(value for row in matrix for value in row))
    budget.reserve(len(values))
    # Dyadic charts share one prepared action through their exact binary64 bytes.
    exact = all(Fraction(float(value)) == value for value in values)
    dtype = np.float64 if exact else object
    arguments = coordinate_enclosure_owner.restriction_chart_arguments(
        cell_kind,
        np.asarray([float(v) if exact else v for v in offset], dtype=dtype),
        np.asarray(
            [[float(v) if exact else v for v in row] for row in matrix], dtype=dtype
        ),
        budget=budget,
    )
    composition = coordinate_enclosure_owner.ExpressionComposition(
        arguments, budget=budget
    )
    weights = tuple(
        tuple(matrix[axis][column] for axis in range(dimension))
        for column in range(dimension)
    )
    return tuple(
        coordinate_enclosure_owner.expression_linear_combinations(
            tuple(composition(entry, budget=budget) for entry in entries),
            weights,
            budget=budget,
        )
        for entries in parent_jacobian
    )


def _restricted_parent_validity(
    element: cell_geometry_owner.ScalarCoordinateElement,
    kind: str,
    local: coordinate_enclosure_owner.CoordinateSourceBank,
    parent: _RestrictionParent,
    imported: _ValidityImport,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _SubdivisionResult | RestrictedValidityRefusal:
    """Import a child's validity from an exact affine restriction of its parent.

    With identical coefficients, ``child = parent o S`` for the exact stored
    affine chart ``S(xi) = M xi + o``. Every child reference corner maps into
    the closed convex parent reference cell, so ``S`` maps the child cell into
    it and ``det D child(xi) = det D parent(S xi) det M``. The parent's
    certified whole-cell enclosure then bounds the child determinant by
    ``[L det M, U det M]`` for ``det M > 0``. The child's own Jacobian forms
    the same Hadamard scale and relative floor as fresh certification, and
    ``(L det M)^2`` must strictly exceed that squared threshold.
    """
    dimension = element.topological_dimension
    if (
        not local
        or len(local[0]) != dimension
        or parent.element.topological_dimension != dimension
        or "pyramid" in (_cell_kind_family(kind), _cell_kind_family(parent.kind))
    ):
        return "unsupported"
    budget.reserve(sum(len(point) for point in local))
    if local != parent.local:
        return "coefficients"
    chart = _restriction_parent_chart(element, parent.element, budget=budget)
    if chart is None or len(chart[0]) != dimension:
        return "chart"
    matrix, offset = chart
    corners = reference_cell_owner.reference_cell_topology(kind).vertices
    budget.reserve(len(corners) * dimension * (dimension + 1))
    for corner in corners:
        point = tuple(
            sum(
                (row[axis] * Fraction(corner[axis]) for axis in range(dimension)),
                start,
            )
            for row, start in zip(matrix, offset, strict=True)
        )
        if not _inside_parent_reference(point, parent.kind):
            return "not_contained"
    determinant = _fraction_determinant(matrix)
    if determinant <= 0:
        return "orientation"
    row = parent.row
    parent_lower, parent_upper = imported.lower[row], imported.upper[row]
    if (
        imported.status[row] != CellValidityStatus.CERTIFIED_VALID
        or not math.isfinite(parent_lower)
        or not math.isfinite(parent_upper)
        or parent_lower <= 0.0
    ):
        return "partial_proof"
    reference = _prepare_polynomial_reference_domain(
        kind, dimension, element.degree, False
    )
    if reference is None:
        return "unsupported"
    jacobian = _restricted_child_jacobian(
        parent, imported, chart, reference.cell_kind, dimension, budget
    )
    if jacobian is None or any(
        coordinate_enclosure_owner.expression_node_count(
            value, reference.domain, dimension
        )
        > policy.maximum_bernstein_nodes
        for entries in jacobian
        for value in entries
    ):
        return "unsupported"
    magnitude = _expression_magnitudes(jacobian, reference.domain, dimension, budget)
    if magnitude is None:
        return "unsupported"
    squared_scale = math.prod(
        sum((entries[axis] * entries[axis] for entries in magnitude), Fraction(0))
        for axis in range(dimension)
    )
    floor = Fraction(policy.relative_determinant_floor)
    lower = Fraction(float(parent_lower)) * determinant
    if lower * lower <= floor * floor * squared_scale:
        return "scale"
    return _SubdivisionResult(
        CellValidityStatus.CERTIFIED_VALID,
        coefficient_endpoint(lower, upper=False, saturate=True),
        coefficient_endpoint(
            Fraction(float(parent_upper)) * determinant, upper=True, saturate=True
        ),
        int(imported.depth[row]),
        (),
    )


def _restricted_child_validity(
    element: cell_geometry_owner.ScalarCoordinateElement,
    kind: str,
    local: coordinate_enclosure_owner.CoordinateSourceBank,
    origin: int,
    imported: _ValidityImport,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _SubdivisionResult | RestrictedValidityRefusal:
    """Try every predecessor cell of the declared root origin; keep the last refusal."""
    candidates = _restriction_parents(imported, budget).get(origin, ())
    refusal: RestrictedValidityRefusal = "wrong_parent"
    for parent in candidates:
        with budget.polynomial_preparation_scope():
            decision = _restricted_parent_validity(
                element, kind, local, parent, imported, policy, budget
            )
        if isinstance(decision, _SubdivisionResult):
            return decision
        refusal = decision
    return refusal


def _restricted_validity_block(
    element: cell_geometry_owner.ScalarCoordinateElement,
    kind: str,
    route: np.ndarray,
    origins: np.ndarray,
    source_values: coordinate_enclosure_owner.CoordinateSourceBank,
    imported: _ValidityImport,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _BlockCertificate:
    """Import exact restricted children; certify every refused child freshly."""
    count = route.shape[0]
    status = np.empty((count,), dtype=np.int32)
    lower = np.empty((count,), dtype=np.float64)
    upper = np.empty((count,), dtype=np.float64)
    depth = np.empty((count,), dtype=np.int32)
    pending: list[int] = []
    for cell in range(count):
        local = tuple(source_values[int(index)] for index in route[cell])
        decision = _restricted_child_validity(
            element, kind, local, int(origins[cell]), imported, policy, budget,
        )
        if not isinstance(decision, _SubdivisionResult):
            pending.append(cell)
            continue
        status[cell], lower[cell], upper[cell], depth[cell] = (
            decision.status,
            decision.lower,
            decision.upper,
            decision.depth,
        )
    reasons: tuple[CellValidityUnresolvedReason, ...] = ()
    if pending:
        selected = np.asarray(pending, dtype=np.int64)
        fresh = _affine_simplex_block(
            element, kind, route[selected], source_values, policy, budget
        )
        if fresh is None:
            fresh = _certify_polynomial_block(
                element,
                kind,
                tuple(
                    tuple(source_values[int(index)] for index in route[cell])
                    for cell in pending
                ),
                policy,
                budget=budget,
            )
        status[selected] = fresh.status
        lower[selected] = fresh.lower
        upper[selected] = fresh.upper
        depth[selected] = fresh.depth
        reasons = fresh.reasons
    return _BlockCertificate(status, lower, upper, depth, reasons)


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
    quadratic_template_cache: dict[
        tuple[str, str], _QuadraticTetrahedronDerivativeTemplate
    ] = {}
    shared_corner_bank: coordinate_enclosure_owner.CoordinateSourceBank | None = None
    share_corners = mesh_ is not None and any(
        not isinstance(element, reference_types_owner.FiniteElementSpec)
        for element in layout.elements
    ) and all(
        kind in _AFFINE_SIMPLEX_DIMENSIONS
        and not isinstance(element, cell_geometry_owner.CellVertexGeometryElement)
        and coordinate_enclosure_owner.affine_simplex_source(element, budget=budget)
        for kind, element in zip(layout.kinds, layout.elements, strict=True)
    )
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
                    record = spec.restriction_source
                    if (
                        changed is not None
                        and imported is not None
                        and record is not None
                        and isinstance(
                            element,
                            (
                                cell_geometry_owner.RestrictedCellGeometryElement,
                                cell_geometry_owner.PolynomialComposedCellGeometryElement,
                            ),
                        )
                    ):
                        block = _restricted_validity_block(
                            element, kind, route,
                            np.asarray(record.block_parent_cell_ids[name], dtype=np.int64)[changed],
                            sources.source_values, imported, policy_, budget,
                        )
                    else:
                        block = _affine_simplex_block(
                            element, kind, route, sources.source_values, policy_, budget
                        )
                        if block is None:
                            block = _affine_box_block(
                                element, kind, route, sources.source_values, policy_, budget
                            )
                        if block is None:
                            corner_routes = None
                            if share_corners and mesh_ is not None:
                                if shared_corner_bank is None:
                                    images = coordinate_enclosure_owner.prepared_coordinate_vertex_images(
                                        mesh_,
                                        spec,
                                        mismatch="cell_geometry_validity.corner_images must agree on every shared vertex; got unequal images",
                                        budget=budget,
                                    )
                                    if images is not None and all(
                                        image is not None for image in images
                                    ):
                                        shared_corner_bank = cast(
                                            coordinate_enclosure_owner.CoordinateSourceBank,
                                            images,
                                        )
                                if shared_corner_bank is not None:
                                    vertices = np.asarray(
                                        next(
                                            block.vertices
                                            for block in mesh_.blocks
                                            if block.name == name
                                        ),
                                        dtype=np.int64,
                                    )
                                    corner_routes = (
                                        vertices if changed is None else vertices[changed]
                                    )
                            block = _affine_chart_simplex_block(
                                element,
                                kind,
                                route,
                                sources.source_values,
                                policy_,
                                budget,
                                corner_bank=shared_corner_bank,
                                corner_routes=corner_routes,
                            )
                    if block is None:
                        native_input = _quadratic_tetrahedron_native_input(
                            spec,
                            element,
                            kind,
                            route,
                            values,
                            policy_,
                            quadratic_template_cache,
                            budget,
                        )
                        if native_input is None:
                            local = tuple(
                                tuple(sources.source_values[index] for index in row)
                                for row in route
                            )
                            block = _certify_polynomial_block(
                                element, kind, local, policy_, budget=budget
                            )
                        else:
                            with budget.temporary_scope():
                                root = _run_quadratic_tetrahedron_native_root(
                                    native_input, policy_, budget
                                )
                                local = tuple(
                                    tuple(
                                        sources.source_values[int(index)]
                                        for index in route[cell]
                                    )
                                    for cell, decision in enumerate(root.decisions)
                                    if int(decision)
                                    == meshcore_owner.QuadraticTetrahedronValidityDecision.NEEDS_SUBDIVISION
                                )
                                block = _certify_polynomial_block(
                                    element,
                                    kind,
                                    local,
                                    policy_,
                                    budget=budget,
                                    quadratic_root=root,
                                )
                                del root
                    blocks.append(block)
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
        else _prepare_validity_import(predecessor, mesh_, spec, policy_, budget=budget)
    )
    return _certify_prepared_cell_geometry_validity(
        spec, mesh_, policy_, budget, imported
    )


@dataclass(frozen=True)
class _TensorGridRestriction:
    """Inputs, not reusable authority: consumed inside one canonical proof call.

    For each root the three positive counts define the exact partitions
    ``[i/n, (i+1)/n]``. Their Cartesian product covers the root, has disjoint
    interiors, and has positive determinant ``1/prod(counts)``. ``order`` maps
    the actual target layout back to that exhaustive C-order enumeration.
    """

    counts: tuple[tuple[int, int, int], ...]
    order: tuple[int, ...]


@dataclass
class _ConstructedTensorGrid:
    """One-use, nonescaping handoff from the paid canonical tensor constructor."""

    budget: CoordinateEnclosureBudget
    source_mesh: cell_mesh_owner.CellMesh
    source_geometry: cell_geometry_owner.CellGeometrySpec
    mesh: cell_mesh_owner.CellMesh
    geometry: cell_geometry_owner.CellGeometrySpec
    theorem: _TensorGridRestriction
    parents: tuple[int, ...]
    owner_fields: tuple[tuple[object, type, tuple[str, ...], tuple[object, ...]], ...]
    active: bool = True

    def consume(
        self, source_mesh: cell_mesh_owner.CellMesh,
        source_geometry: cell_geometry_owner.CellGeometrySpec,
        mesh: cell_mesh_owner.CellMesh, geometry: cell_geometry_owner.CellGeometrySpec,
        theorem: _TensorGridRestriction | None, budget: CoordinateEnclosureBudget,
    ) -> None:
        active, self.active = self.active, False
        if (
            not active or self.budget is not budget
            or self.source_mesh is not source_mesh or self.source_geometry is not source_geometry
            or self.mesh is not mesh or self.geometry is not geometry
            or self.theorem is not theorem
            or not _immutable_owner_fields_match(self.owner_fields, budget)
        ):
            raise ValueError("tensor_grid.construction must be current, unchanged and consumed exactly once")


def _prepare_constructed_tensor_grid(
    source_mesh: cell_mesh_owner.CellMesh,
    source_geometry: cell_geometry_owner.CellGeometrySpec,
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    theorem: _TensorGridRestriction,
    parents: tuple[int, ...],
    budget: CoordinateEnclosureBudget,
) -> _ConstructedTensorGrid | None:
    """Seal only fresh immutable outputs; callers must not expose a continuation."""
    fields = _snapshot_immutable_owner_fields((mesh, geometry, theorem), budget)
    if fields is None:
        return None
    budget.reserve(0, 512 + 16 * len(fields))
    result = _ConstructedTensorGrid(
        budget, source_mesh, source_geometry, mesh, geometry, theorem, parents, fields
    )
    names = ("budget", "source_mesh", "source_geometry", "mesh", "geometry", "theorem", "parents")
    result.owner_fields = (*fields, (result, type(result), names, tuple(getattr(result, name) for name in names)))
    return result


def _constructed_tensor_validity_blocks(
    theorem: _TensorGridRestriction,
    layout: _ValidityLayout,
    sources: _ValiditySources,
    spec: cell_geometry_owner.CellGeometrySpec,
    imported: _ValidityImport,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _ValidityBlocks | None:
    construction = imported.construction
    assert construction is not None and not construction.active
    source = imported.predecessor.source_geometry
    source_mesh = imported.predecessor.source_mesh
    bank = coordinate_enclosure_owner.prepared_coordinate_source_bank(source, budget=budget)
    budget.reserve(sum(len(point) for point in bank))
    record = spec.restriction_source
    if (
        sources.source_values != bank or record is None
        or record.source_geometry_id != cell_geometry_id(source)
        or record.source_topology_id != source_mesh.topology_id
        or set(record.block_names) != set(layout.names)
    ):
        raise ValueError("tensor_grid.construction must bind the complete physical root source")
    root_ids = tuple(
        int(identity) for block in source_mesh.blocks for identity in np.asarray(block.global_ids)
    )
    parents = _restriction_parents(imported, budget)
    decisions = []
    for root, identity in enumerate(root_ids):
        candidates = parents.get(identity, ())
        if len(candidates) != 1:
            return None
        decision = _tensor_root_validity(candidates[0], imported, theorem.counts[root], policy, budget)
        if decision is None:
            return None
        decisions.append(decision)
    blocks, offsets = [], [0]
    cursor = 0
    for routes in layout.routes:
        count = len(routes)
        budget.reserve(5 * count + 1, 24 * count)
        status = np.empty(count, dtype=np.int32)
        lower = np.empty(count, dtype=np.float64)
        upper = np.empty(count, dtype=np.float64)
        depth = np.empty(count, dtype=np.int32)
        for local in range(count):
            decision = decisions[construction.parents[cursor + local]]
            status[local], lower[local], upper[local], depth[local] = (
                decision.status, decision.lower, decision.upper, decision.depth
            )
        blocks.append(_BlockCertificate(status, lower, upper, depth, ()))
        cursor += count
        offsets.append(cursor)
    if cursor != len(theorem.order) or cursor != len(construction.parents):
        raise ValueError("tensor_grid.construction must cover the complete final target layout")
    return _ValidityBlocks(tuple(blocks), tuple(offsets), ())


def _tensor_root_validity(
    parent: _RestrictionParent,
    imported: _ValidityImport,
    counts: tuple[int, int, int],
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _SubdivisionResult | None:
    """Transport a whole-root determinant and Hadamard bound, not a sampled child.

    Positive diagonal restriction scales the determinant and the product of
    column norms by the same factor. Root-wide magnitudes therefore establish
    the relative floor on every translated child, including nonaffine roots.
    """
    row = parent.row
    lower, upper = float(imported.lower[row]), float(imported.upper[row])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0:
        return None
    identity = tuple(
        tuple(Fraction(int(i == j)) for j in range(3)) for i in range(3)
    )
    jacobian = _restricted_child_jacobian(
        parent, imported, (identity, (Fraction(0),) * 3), "hexahedron", 3, budget
    )
    if jacobian is None:
        return None
    magnitude = _expression_magnitudes(jacobian, "box", 3, budget)
    if magnitude is None:
        return None
    budget.reserve(40, 4096)
    squared_scale = math.prod(
        sum((entries[axis] ** 2 for entries in magnitude), Fraction(0))
        for axis in range(3)
    )
    if Fraction(lower) ** 2 <= Fraction(policy.relative_determinant_floor) ** 2 * squared_scale:
        return None
    determinant = Fraction(1, math.prod(counts))
    return _SubdivisionResult(
        CellValidityStatus.CERTIFIED_VALID,
        coefficient_endpoint(Fraction(lower) * determinant, upper=False, saturate=True),
        coefficient_endpoint(Fraction(upper) * determinant, upper=True, saturate=True),
        int(imported.depth[row]),
        (),
    )


def _tensor_grid_validity_blocks(
    theorem: _TensorGridRestriction,
    layout: _ValidityLayout,
    sources: _ValiditySources,
    spec: cell_geometry_owner.CellGeometrySpec,
    imported: _ValidityImport,
    policy: CellValidityPolicy,
    budget: CoordinateEnclosureBudget,
) -> _ValidityBlocks | None:
    """Authenticate a whole uniform tensor restriction, then transport by root.

    No verdict or authority escapes. Every chart coefficient, route, parent and
    layout row is checked against the formula. Whole-root differential bounds
    cover every translation; no nonaffine child verdict is reused.
    """
    if imported.construction is not None:
        return _constructed_tensor_validity_blocks(
            theorem, layout, sources, spec, imported, policy, budget
        )
    source = imported.predecessor.source_geometry
    source_mesh = imported.predecessor.source_mesh
    root_count = sum(block.cell_count for block in source_mesh.blocks)
    counts = theorem.counts
    if len(counts) != root_count or any(
        len(row) != 3 or any(type(n) is not int or n <= 0 for n in row)
        for row in counts
    ):
        raise ValueError("tensor_grid.counts must define positive exact axis partitions")
    sizes = tuple(math.prod(row) for row in counts)
    count = sum(sizes)
    budget.reserve(
        8 * root_count + 7 * count, 8192 + 80 * count + 2048 * root_count
    )
    if len(theorem.order) != count or sum(len(row) for row in layout.routes) != count:
        raise ValueError("tensor_grid.order must cover the entire target layout")
    if any(type(row) is not int for row in theorem.order):
        raise ValueError("tensor_grid.order must contain exact integer rows")
    order = np.asarray(theorem.order, dtype=np.int64)
    if np.any(order < 0) or np.any(order >= count):
        raise ValueError("tensor_grid.order contains an out-of-range row")
    seen = np.zeros(count, dtype=np.bool_)
    seen[order] = True
    if not np.all(seen):
        raise ValueError("tensor_grid.order must enumerate each cell exactly once")
    axis_nodes = sum(n + 1 for row in counts for n in row)
    reserve_polynomial(
        2 * axis_nodes, axis_nodes, 3,
        max((n.bit_length() for row in counts for n in row), default=1),
        budget=budget,
    )
    partitions = tuple(
        tuple(
            tuple(
                (value.numerator, value.denominator)
                for value in (Fraction(i, n) for i in range(n + 1))
            )
            for n in row
        )
        for row in counts
    )
    root_rows = np.repeat(np.arange(root_count, dtype=np.int64), sizes)
    starts = np.cumsum((0, *sizes), dtype=np.int64)
    source_elements, source_routes, _ = source.resolve(source_mesh, budget=budget)
    source_bank = coordinate_enclosure_owner.prepared_coordinate_source_bank(
        source, budget=budget
    )
    budget.reserve(sum(len(point) for point in source_bank))
    if sources.source_values != source_bank:
        raise ValueError("tensor_grid.source must retain the complete root coefficient bank")
    record = spec.restriction_source
    if (
        record is None
        or record.source_geometry_id != cell_geometry_id(source)
        or record.source_topology_id != source_mesh.topology_id
        or set(record.block_names) != set(layout.names)
    ):
        raise ValueError("tensor_grid.source must retain the complete current root provenance")
    roots = []
    budget.reserve(8 * root_count, 64 * root_count)
    source_vertex_ids = np.asarray(source_mesh.vertex_global_ids, dtype=np.int64)
    for block, element, routes in zip(
        source_mesh.blocks, source_elements, source_routes, strict=True
    ):
        if block.cell_kind != "hexahedron":
            return None
        scalar_element = cell_geometry_owner.require_scalar_coordinate_element(
            element, "tensor grid root"
        )
        for identity, route, vertices in zip(
            np.asarray(block.global_ids).tolist(), np.asarray(routes),
            np.asarray(block.vertices, dtype=np.int64), strict=True,
        ):
            roots.append((
                int(identity), scalar_element, tuple(int(i) for i in route),
                tuple(int(i) for i in source_vertex_ids[vertices]),
            ))
    parents = _restriction_parents(imported, budget)
    decisions: dict[int, _SubdivisionResult] = {}
    authenticated_sources: set[tuple[int, int]] = set()
    authenticated_charts: set[int] = set()
    chart = cell_geometry_owner.coordinate_lagrange_element("hexahedron", 1)
    corners = reference_cell_owner.reference_cell_topology("hexahedron").vertices
    nodes = cell_geometry_owner._cardinal_corner_nodes(chart, budget=budget)
    if nodes is None:
        return None
    blocks = []
    offsets = [0]
    cursor = 0
    for name, kind, element, routes in zip(
        layout.names, layout.kinds, layout.elements, layout.routes, strict=True
    ):
        if (
            kind != "hexahedron"
            or not isinstance(element, cell_geometry_owner.PolynomialComposedCellGeometryElement)
            or element.chart_element.element_id != chart.element_id
        ):
            return None
        if id(element.chart_element) not in authenticated_charts:
            if coordinate_enclosure_owner.coordinate_source_signature(
                element.chart_element
            ) != coordinate_enclosure_owner.coordinate_source_signature(chart):
                raise ValueError("tensor_grid.chart must retain the canonical Q1 source law")
            authenticated_charts.add(id(element.chart_element))
        if len(element.chart_coefficients) != 8 or any(
            len(row) != 3 for row in element.chart_coefficients
        ):
            raise ValueError("tensor_grid.chart must retain all exact corner coefficients")
        origins = np.asarray(record.block_parent_cell_ids[name], dtype=np.int64)
        parent_vertices = np.asarray(record.block_parent_vertex_ids[name], dtype=np.int64)
        if origins.shape != (len(routes),) or parent_vertices.shape != (len(routes), 8):
            raise ValueError("tensor_grid.parent must bind every target row and ordered root corner")
        budget.reserve(9, 24 * len(routes))
        status = np.empty(len(routes), dtype=np.int32)
        lower = np.empty(len(routes), dtype=np.float64)
        upper = np.empty(len(routes), dtype=np.float64)
        depth = np.empty(len(routes), dtype=np.int32)
        chart_partition: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None
        for local, route in enumerate(routes):
            original = int(order[cursor + local])
            root = int(root_rows[original])
            identity, root_element, root_route, root_vertices = roots[root]
            budget.reserve(2 + len(root_route) + 6 + len(root_vertices))
            if (
                int(origins[local]) != identity
                or tuple(int(i) for i in route) != root_route
                or tuple(int(i) for i in parent_vertices[local]) != root_vertices
                or element.source_element.element_id != root_element.element_id
                or element.degree != root_element.degree
            ):
                raise ValueError("tensor_grid.parent must match its formula and complete source route")
            source_pair = (id(element.source_element), id(root_element))
            if source_pair not in authenticated_sources:
                if coordinate_enclosure_owner.coordinate_source_signature(
                    element.source_element
                ) != coordinate_enclosure_owner.coordinate_source_signature(root_element):
                    raise ValueError("tensor_grid.source must match the complete root map law")
                authenticated_sources.add(source_pair)
            _, ny, nz = counts[root]
            index = original - int(starts[root])
            integer = (index // (ny * nz), (index // nz) % ny, index % nz)
            partition = (counts[root], integer)
            if chart_partition is None:
                budget.reserve(48)
                for corner, node in zip(corners, nodes, strict=True):
                    if any(
                        pair != partitions[root][axis][integer[axis] + int(corner[axis])]
                        for axis, pair in enumerate(element.chart_coefficients[node])
                    ):
                        raise ValueError("tensor_grid.chart must match its exact partition and deterministic order")
                chart_partition = partition
            elif partition != chart_partition:
                raise ValueError("tensor_grid.chart rows must share the same exact partition member")
            if root not in decisions:
                candidates = parents.get(identity, ())
                if len(candidates) != 1:
                    return None
                parent = candidates[0]
                decision = _tensor_root_validity(
                    parent, imported, counts[root], policy, budget
                )
                if decision is None:
                    return None
                decisions[root] = decision
            decision = decisions[root]
            status[local], lower[local], upper[local], depth[local] = (
                decision.status, decision.lower, decision.upper, decision.depth
            )
        blocks.append(_BlockCertificate(status, lower, upper, depth, ()))
        cursor += len(routes)
        offsets.append(cursor)
    return _ValidityBlocks(tuple(blocks), tuple(offsets), ())


def _certify_restriction_cell_geometry_validity(
    source_mesh: cell_mesh_owner.CellMesh,
    source_geometry: cell_geometry_owner.CellGeometrySpec,
    source_validity: CellValidityCertificate,
    mesh: cell_mesh_owner.CellMesh,
    geometry: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    policy: CellValidityPolicy | None = None,
    tensor_grid: _TensorGridRestriction | None = None,
    construction: _ConstructedTensorGrid | None = None,
) -> CellValidityCertificate:
    """Prepare and consume one restriction predecessor without exporting authority.

    Only the certificate escapes. The optional private tensor inputs earn one
    exhaustive partition theorem on this target; ordinary calls prove each
    child chart and differential independently. Neither branch exports authority.
    """
    if construction is not None:
        construction.consume(source_mesh, source_geometry, mesh, geometry, tensor_grid, budget)
    policy_ = CellValidityPolicy() if policy is None else policy
    if (
        tensor_grid is not None
        and source_geometry.exact_source is None
        and geometry.exact_source is None
        and source_geometry.periodic_source is None
        and geometry.periodic_source is None
        and geometry.restriction_source is not None
        and geometry.restriction_source.carrier_publication == "source_image"
    ):
        source_validity.require_bound(source_geometry, mesh=source_mesh)
        root_count = sum(block.cell_count for block in source_mesh.blocks)
        if (
            source_validity.policy_id != policy_.policy_id
            or source_validity.geometry_layout_id != source_geometry.geometry_layout_id
            or not source_validity.all_certified
            or source_validity.status.shape != (root_count,)
            or source_validity.block_names != tuple(block.name for block in source_mesh.blocks)
        ):
            raise ValueError("tensor_grid.source_validity must bind every current root and the requested policy")
        budget.reserve(4 * root_count, 24 * root_count + 256)
        status, lower, upper, depth = jax.device_get((
            source_validity.status, source_validity.determinant_lower,
            source_validity.determinant_upper, source_validity.depth,
        ))
        # Root-sized determinant data only: no target correspondence rows,
        # predecessor content key or ordinary predecessor/import layout.
        compact = _ValidityImport(
            np.empty(0, dtype=np.int64), np.asarray(status), np.asarray(lower),
            np.asarray(upper), np.asarray(depth),
            _TensorGridSource(source_mesh, source_geometry, source_validity), {}, {}, construction,
        )
        return _certify_prepared_cell_geometry_validity(
            geometry, mesh, policy_, budget, compact, tensor_grid
        )
    imported = _prepare_validity_import(
        _CellGeometryPredecessorSource(source_mesh, source_geometry, source_validity),
        mesh, geometry, policy_, budget=budget,
    )
    return _certify_prepared_cell_geometry_validity(
        geometry, mesh, policy_, budget, imported, tensor_grid
    )


def _certify_prepared_cell_geometry_validity(
    spec: cell_geometry_owner.CellGeometrySpec,
    mesh_: cell_mesh_owner.CellMesh | None,
    policy_: CellValidityPolicy,
    budget: CoordinateEnclosureBudget | None,
    imported: _ValidityImport | None,
    tensor_grid: _TensorGridRestriction | None = None,
) -> CellValidityCertificate:
    reuse = budget is not None and tensor_grid is None
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
        compact = imported is not None and isinstance(imported.predecessor, _TensorGridSource)
        if compact:
            assert mesh_ is not None
            # The theorem checks every actual route and full source/chart law.
            # Resolve only the target structural layout once, not its complete
            # composed-source preparation for each repeated child.
            budget.reserve(4 * len(mesh_.blocks), 256 + 128 * len(mesh_.blocks))
            resolved, routes, _ = spec._resolve_layout(mesh_)
            layout = _ValidityLayout(
                resolved, tuple(np.asarray(route, dtype=np.int64) for route in routes),
                tuple(block.name for block in mesh_.blocks),
                tuple(block.cell_kind for block in mesh_.blocks),
            )
        else:
            layout = _prepare_validity_layout(spec, mesh_, budget)
        names = layout.names
        sources = _prepare_validity_sources(spec, mesh_, layout.elements, policy_, budget)
        result = (
            None if tensor_grid is None or imported is None
            else _tensor_grid_validity_blocks(
                tensor_grid, layout, sources, spec, imported, policy_, budget
            )
        )
        if result is None:
            if compact:
                # Unsupported or insufficient whole-root bounds use the ordinary theorem.
                # Start the ordinary path from its ordinary public inputs.
                assert imported is not None
                owner = imported.predecessor
                assert isinstance(owner, _TensorGridSource)
                imported = _prepare_validity_import(
                    _CellGeometryPredecessorSource(
                        owner.source_mesh, owner.source_geometry, owner.source_validity
                    ),
                    mesh_, spec, policy_, budget=budget,
                )
                layout = _prepare_validity_layout(spec, mesh_, budget)
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
