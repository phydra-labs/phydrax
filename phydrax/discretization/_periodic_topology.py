#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Quotient topology of a finite lifted periodic cell mesh.

A periodic mesh is published as an ordinary lifted `CellMesh` whose vertices
carry distinct rows and global IDs, plus this descriptor. The identification
is either a translational `PeriodicCell` lattice or a `PeriodicIsometryGroup`
of commuting proper Euclidean isometries (for example the rotation of a wedge
sector). Every lifted vertex names its quotient representative (a lifted
vertex with zero shift) and the integer group exponents ``s`` with
``x = G(s) x[representative]``; for a lattice ``G(s) x = x + s @ vectors``.
The stored binary64 coordinates are evaluations of that group lift; their
construction residual is checked against the lattice arithmetic bound or the
isometry tolerance. Acceptance must report this residual rather than treat
independently rounded image coordinates as exact group constructions.

A lifted entity with oriented corners ``(r_i, s_i)`` is identified with every
group image ``(r_i, s_i + t)``. Exponents are reduced modulo finite generator
orders, and a corner fixed by a generator (a vertex on a rotation axis) has no
image along it, so its exponent is dropped. Its canonical key takes the
lexicographically smallest oriented form over the admissible starting corners
and directions, with representative global IDs followed by the corner
exponents relative to the starting corner. The key therefore keeps winding:
two quotient edges joining the same representatives with different relative
shifts are distinct entities. The direction of the chosen form is the
orientation witness of the lifted entity relative to its quotient entity. Top
cells are keyed without orientation and must be pairwise distinct orbits, so
every quotient top cell is integrated exactly once in its local lift.

Quotient incidences are the lifted signed incidences of one representative
member, mapped through the witnesses and accumulated. Every other member of an
orbit must produce the same quotient boundary (orbit composition), accumulated
coefficients must be ``±1`` (orientation), and each quotient facet may bound at
most two top-cell sides with opposite induced orientation.
"""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import product
from typing import final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation, SparseLinearMap
from ..typing import Bool, Dim, Float64, Int32, Int64, Scalar
from ._cell_complex import (
    _first_appearance_groups,
    _variable_row_ranks,
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
)
from ._hexahedral import HexahedralConnectivity
from ._periodic_cell import PeriodicCell
from ._topology import (
    _has_duplicate_rows,
    _row_order,
    _row_run_starts,
    CellComplexTopology,
    EntitySet,
    EntitySubset,
    OrientedIncidence,
)


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh


class _LiftedVertexDim(Dim, minimum=1):
    """Number of lifted mesh vertices."""


class _LatticeRankDim(Dim, minimum=1):
    """Rank of the bound periodic lattice."""


class _CornerDim(Dim, minimum=1):
    """Number of active lifted cell corners in block, cell, corner order."""


class _LiftedEntityDim(Dim, minimum=1):
    """Number of lifted entities of every degree, concatenated by degree."""


class _QuotientEntityDim(Dim, minimum=1):
    """Number of quotient entities of every degree, concatenated by degree."""


class _DegreeBoundaryDim(Dim, minimum=3):
    """Degree offsets: topological dimension plus two."""


class _KeyBoundaryDim(Dim, minimum=2):
    """Quotient entity count plus one key offset."""


class _KeyValueDim(Dim, minimum=1):
    """Packed quotient entity key values."""


class _QuotientCellDim(Dim, minimum=1):
    """Number of quotient top cells."""


class _GeneratorDim(Dim, minimum=1):
    """Number of commuting isometry generators."""


class _HomogeneousDim(Dim, minimum=3):
    """Homogeneous coordinate extent: ambient dimension plus one."""


_INT64_MAX = np.iinfo(np.int64).max
# Finite rotation orders searched when classifying an isometry generator.
_MAXIMUM_ROTATION_ORDER = 1024
_MAXIMUM_GROUP_IMAGES = 100_000


def _isometry_inverse(matrix: np.ndarray, /) -> np.ndarray:
    """Exact inverse of a homogeneous isometry ``x -> Q x + b``."""

    dimension = matrix.shape[0] - 1
    inverse = np.eye(dimension + 1, dtype=np.float64)
    inverse[:dimension, :dimension] = matrix[:dimension, :dimension].T
    inverse[:dimension, dimension] = (
        -matrix[:dimension, :dimension].T @ matrix[:dimension, dimension]
    )
    return inverse


def _isometry_power(matrix: np.ndarray, exponent: int, /) -> np.ndarray:
    """Integer power of a homogeneous isometry by repeated squaring."""

    base = matrix if exponent >= 0 else _isometry_inverse(matrix)
    remaining = abs(exponent)
    result = np.eye(matrix.shape[0], dtype=np.float64)
    while remaining:
        if remaining & 1:
            result = base @ result
        base = base @ base
        remaining >>= 1
    return result


def _require_proper_isometries(matrices: np.ndarray, /) -> None:
    dimension = matrices.shape[1] - 1
    identity = np.eye(dimension, dtype=np.float64)
    bottom = np.eye(dimension + 1, dtype=np.float64)[-1]
    # Orthonormality within the rounding of trigonometric entries.
    bound = 128.0 * float(np.finfo(np.float64).eps) * dimension
    for index, matrix in enumerate(matrices):
        linear = matrix[:dimension, :dimension]
        if not np.array_equal(matrix[dimension], bottom):
            raise ValueError(f"generators[{index}] is not homogeneous.")
        if np.max(np.abs(linear.T @ linear - identity)) > bound:
            raise ValueError(f"generators[{index}] is not a Euclidean isometry.")
        if np.linalg.det(linear) < 0.0:
            raise ValueError(
                f"generators[{index}] reverses orientation; improper "
                "identifications have no consistent quotient orientation."
            )


@final
class PeriodicIsometryGroup(StrictModule, NonTrainableState):
    """Abelian group of proper Euclidean isometries identifying a periodic mesh.

    ``generators`` (r, d + 1, d + 1) are homogeneous maps ``x -> Q x + b`` with
    orthogonal ``Q`` of determinant ``+1``. ``orders`` records the exact full
    homogeneous order (zero means infinite); ``linear_orders`` records the
    finite exact order of Q. Infinite screws retain their original generator,
    with ``translation_periods`` giving the exponent of its translation.
    Generators must commute, so every cycle ``G_a G_b G_a^-1 G_b^-1`` is the
    identity and integer exponents ``s`` denote ``prod_c G_c^(s_c)`` in any order.
    Reflections and other improper maps reverse orientation and are refused.
    ``tolerance`` bounds the coordinate residual of every declared image.
    """

    __strict_contract__ = True

    generators: Float64[_GeneratorDim, _HomogeneousDim, _HomogeneousDim]
    orders: tuple[int, ...] = eqx.field(static=True)
    linear_orders: tuple[int, ...] = eqx.field(static=True)
    translation_periods: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    group_id: str = eqx.field(static=True)

    def __init__(self, generators: ArrayLike, /, *, tolerance: float = 1.0e-10) -> None:
        matrices = np.asarray(generators, dtype=np.float64)
        if (
            matrices.ndim != 3
            or matrices.shape[0] == 0
            or matrices.shape[1] != matrices.shape[2]
            or matrices.shape[1] < 3
        ):
            raise ValueError(
                "generators must have shape (r > 0, d + 1, d + 1) with d >= 2."
            )
        threshold = float(tolerance)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if not np.all(np.isfinite(matrices)):
            raise ValueError("generators must be finite.")
        _require_proper_isometries(matrices)
        _, orders, linear_orders, translation_periods = _admit_exact_isometry_group(
            matrices
        )
        self.generators = jnp.asarray(matrices, dtype=jnp.float64)
        self.orders = orders
        self.linear_orders = linear_orders
        self.translation_periods = translation_periods
        self.tolerance = threshold
        self.ambient_dimension = matrices.shape[1] - 1
        self.group_id = canonical_fingerprint(
            {
                "kind": "periodic-isometry-group",
                "generators": array_tree_fingerprint(matrices),
                "orders": list(orders),
                "linear_orders": list(linear_orders),
                "translation_periods": list(translation_periods),
                "tolerance": threshold,
            }
        )

    def validate_restored(self) -> None:
        """Authenticate retained original source controls and derived metadata."""
        replay = PeriodicIsometryGroup(self.generators, tolerance=self.tolerance)
        for name in (
            "orders",
            "linear_orders",
            "translation_periods",
            "ambient_dimension",
            "group_id",
        ):
            if getattr(self, name) != getattr(replay, name):
                raise ValueError(
                    f"Restored periodic isometry group {name} is not source-authentic."
                )

    @property
    def rank(self) -> int:
        return len(self.orders)

    def element(self, exponents: ArrayLike, /) -> np.ndarray:
        """Homogeneous matrix of the group element with integer ``exponents``."""

        values = np.asarray(exponents)
        if not np.issubdtype(values.dtype, np.integer) or values.shape != (self.rank,):
            raise ValueError("exponents must be one integer per generator.")
        matrices = np.asarray(self.generators, dtype=np.float64)
        result = np.eye(self.ambient_dimension + 1, dtype=np.float64)
        for matrix, exponent, order in zip(
            matrices, values.tolist(), self.orders, strict=True
        ):
            reduced = exponent % order if order else exponent
            result = _isometry_power(np.asarray(matrix), reduced) @ result
        return result

    def apply(self, points: ArrayLike, exponents: ArrayLike, /) -> np.ndarray:
        """Map each point by the group element of its exponent row."""

        array = np.asarray(points, dtype=np.float64)
        values = np.asarray(exponents)
        if (
            array.ndim != 2
            or array.shape[1] != self.ambient_dimension
            or values.shape != (array.shape[0], self.rank)
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError(
                "points must be (n, ambient dimension) with one integer exponent "
                "row per point."
            )
        reduced = _reduced(values.astype(np.int64), np.asarray(self.orders))
        unique, inverse = np.unique(reduced, axis=0, return_inverse=True)
        mapped = np.empty_like(array)
        dimension = self.ambient_dimension
        for row, exponent in enumerate(unique):
            matrix = self.element(exponent)
            members = inverse.reshape(-1) == row
            mapped[members] = (
                array[members] @ matrix[:dimension, :dimension].T
                + matrix[:dimension, dimension]
            )
        return mapped

    def fixed(self, points: ArrayLike, /) -> np.ndarray:
        """Whether each generator fixes each point within the group tolerance."""

        array = np.asarray(points, dtype=np.float64)
        dimension = self.ambient_dimension
        matrices = np.asarray(self.generators, dtype=np.float64)
        images = (
            np.asarray(
                ein.contract("gij,nj->ngi", matrices[:, :dimension, :dimension], array)
            )
            + matrices[None, :, :dimension, dimension]
        )
        return np.linalg.norm(images - array[:, None, :], axis=2) <= self.tolerance


type PeriodicMeshIdentification = PeriodicCell | PeriodicIsometryGroup


def _reduced(values: np.ndarray, orders: np.ndarray, /) -> np.ndarray:
    """Reduce exponents modulo the finite generator orders (last axis)."""

    return np.where(orders > 0, np.mod(values, np.maximum(orders, 1)), values)


def _identification_id(cell: PeriodicMeshIdentification, /) -> str:
    match cell:
        case PeriodicCell():
            return cell.cell_id
        case PeriodicIsometryGroup():
            return cell.group_id
        case _:
            raise TypeError("cell must be a PeriodicCell or PeriodicIsometryGroup.")


def _identification_orders(cell: PeriodicMeshIdentification, /) -> np.ndarray:
    match cell:
        case PeriodicCell():
            return np.zeros((cell.rank,), dtype=np.int64)
        case PeriodicIsometryGroup():
            return np.asarray(cell.orders, dtype=np.int64)
        case _:
            raise TypeError("cell must be a PeriodicCell or PeriodicIsometryGroup.")


def _exact_periodic_vertex_source(
    coordinates: np.ndarray,
    geometry: CellGeometrySpec,
    /,
) -> tuple[tuple[Fraction, ...], ...]:
    """Renew vertex-indexed source authority, never accept a detached exact bank."""
    from ._cell_geometry import CellGeometrySpec, CellVertexGeometryElement

    if not isinstance(geometry, CellGeometrySpec):
        raise TypeError("actual_geometry must be an owning CellGeometrySpec.")
    if geometry.exact_source is None:
        raise ValueError(
            "Exact periodic vertex lifts require an owning exact geometry source."
        )
    if geometry.periodic_source is not None:
        raise ValueError(
            "Periodic lift authority cannot itself retain a periodic mesh source."
        )
    if any(
        not isinstance(element, CellVertexGeometryElement)
        for element in geometry.elements
    ):
        raise ValueError(
            "Exact periodic vertex lifts require actual vertex-indexed geometry."
        )
    numeric = np.asarray(geometry.coordinates, dtype=np.float64)
    if numeric.shape != coordinates.shape or not np.array_equal(
        numeric.view(np.uint64),
        coordinates.view(np.uint64),
    ):
        raise ValueError(
            "Periodic source geometry differs from the complete original RNE carrier."
        )
    bank = geometry.source_coordinates()
    if len(bank) != coordinates.shape[0] or any(
        len(row) != coordinates.shape[1] for row in bank
    ):
        raise ValueError(
            "Periodic source geometry must own every lifted vertex coordinate."
        )
    return bank


def _exact_periodic_point_image(
    matrix: tuple[tuple[Fraction, ...], ...],
    point: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, ...]:
    _reserve_exact_isometry_work(len(point) * (len(point) + 1))
    return tuple(
        sum(
            (
                value * coordinate
                for value, coordinate in zip(row[:-1], point, strict=True)
            ),
            row[-1],
        )
        for row in matrix[:-1]
    )


def _require_exact_periodic_vertex_lift(
    bank: tuple[tuple[Fraction, ...], ...],
    representatives: np.ndarray,
    shifts: np.ndarray,
    cell: PeriodicMeshIdentification,
    /,
) -> None:
    matrices, orders = _exact_periodic_generators(cell)
    for row, root, shift in zip(bank, representatives, shifts, strict=True):
        action = _exact_periodic_element(
            matrices, orders, tuple(int(value) for value in shift)
        )
        if row != _exact_periodic_point_image(action, bank[int(root)]):
            raise ValueError(
                "Owning exact source vertices violate their authored periodic group lifts."
            )


def _fixed_generators(
    cell: PeriodicMeshIdentification,
    coordinates: np.ndarray,
    representatives: np.ndarray,
    /,
    *,
    source_coordinates: tuple[tuple[Fraction, ...], ...] | None = None,
) -> np.ndarray:
    """Per lifted vertex, whether each generator fixes its quotient vertex."""
    if source_coordinates is not None:
        matrices, orders = _exact_periodic_generators(cell)
        fixed = np.asarray(
            [
                [
                    _exact_periodic_point_image(matrix, source_coordinates[int(root)])
                    == source_coordinates[int(root)]
                    for matrix in matrices
                ]
                for root in representatives
            ],
            dtype=np.bool_,
        )
        if sum(order > 1 for order in orders) > 1:
            for root in range(len(source_coordinates)):
                if representatives[root] != root:
                    continue
                point = source_coordinates[root]
                for exponents in product(
                    *(range(order) if order else range(1) for order in orders)
                ):
                    if all(
                        exponent == 0 or fixed[root, axis]
                        for axis, exponent in enumerate(exponents)
                    ):
                        continue
                    action = _exact_periodic_element(matrices, orders, exponents)
                    if _exact_periodic_point_image(action, point) == point:
                        raise PeriodicIsometryIdentityError(
                            "Exact source has a composed vertex stabilizer outside the admitted generator-fixed quotient."
                        )
        return fixed

    match cell:
        case PeriodicCell():
            return np.zeros((representatives.shape[0], cell.rank), dtype=np.bool_)
        case PeriodicIsometryGroup():
            return cell.fixed(coordinates[representatives])
        case _:
            raise TypeError("cell must be a PeriodicCell or PeriodicIsometryGroup.")


def _lexicographic_minimum(variants: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Return the minimal variant per entity and the number of minimal variants."""

    minimal = np.ones(variants.shape[:2], dtype=np.bool_)
    for column in range(variants.shape[2]):
        values = variants[:, :, column]
        lowest = np.min(np.where(minimal, values, _INT64_MAX), axis=1)
        minimal &= values == lowest[:, None]
    return np.argmax(minimal, axis=1), np.count_nonzero(minimal, axis=1)


def _anchored_variants(
    identifiers: np.ndarray,
    shifts: np.ndarray,
    fixed: np.ndarray,
    orders: np.ndarray,
    orderings: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Corner-ordered keys relative to each ordering's first (anchor) corner.

    A corner fixed by a generator carries no exponent along it. An anchor fixed
    by a generator that moves another corner does not determine the entity's
    image, so that ordering is inadmissible and its variant is the maximal
    sentinel. Returns the variants (count, orderings, key) and the anchor
    exponents (count, orderings, rank).
    """

    count = identifiers.shape[0]
    ordered_shifts = shifts[:, orderings]
    ordered_fixed = fixed[:, orderings]
    invariant = np.all(ordered_fixed, axis=2)
    admissible = np.all(~ordered_fixed[:, :, 0] | invariant, axis=2)
    if np.any(~np.any(admissible, axis=1)):
        raise ValueError(
            "A lifted entity has no corner that anchors its group image; entities "
            "whose every corner is fixed by a generator moving another corner "
            "are not supported."
        )
    anchors = np.where(invariant, 0, ordered_shifts[:, :, 0])
    relative = _reduced(
        np.where(ordered_fixed, 0, ordered_shifts - anchors[:, :, None, :]), orders
    )
    variants = np.concatenate(
        (
            identifiers[:, orderings],
            relative.reshape((count, orderings.shape[0], -1)),
        ),
        axis=2,
    )
    return np.where(admissible[:, :, None], variants, _INT64_MAX), anchors


def _require_distinct_corners(
    identifiers: np.ndarray,
    shifts: np.ndarray,
    fixed: np.ndarray,
    orders: np.ndarray,
    noun: str,
    /,
) -> None:
    difference = _reduced(shifts[:, :, None, :] - shifts[:, None, :, :], orders)
    same = (identifiers[:, :, None] == identifiers[:, None, :]) & np.all(
        (difference == 0) | fixed[:, :, None, :], axis=-1
    )
    same[:, np.arange(identifiers.shape[1]), np.arange(identifiers.shape[1])] = False
    if np.any(same):
        raise ValueError(
            f"A lifted {noun} repeats one image of a quotient vertex and "
            "degenerates in the quotient."
        )


def _oriented_orders(arity: int, /) -> tuple[np.ndarray, np.ndarray]:
    """Corner orders equivalent to an oriented edge or face loop, with signs."""

    if arity == 2:
        return np.asarray(((0, 1), (1, 0)), dtype=np.int64), np.asarray((1, -1))
    steps = np.arange(arity, dtype=np.int64)
    orders = []
    signs = []
    for direction in (1, -1):
        for start in range(arity):
            orders.append((start + direction * steps) % arity)
            signs.append(direction)
    return np.asarray(orders, dtype=np.int64), np.asarray(signs, dtype=np.int64)


def _oriented_forms(
    identifiers: np.ndarray, shifts: np.ndarray, fixed: np.ndarray, orders: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical keys, orientation witnesses and anchor image shifts."""

    count, arity = identifiers.shape
    _require_distinct_corners(identifiers, shifts, fixed, orders, "edge or face")
    orderings, signs = _oriented_orders(arity)
    variants, anchors = _anchored_variants(identifiers, shifts, fixed, orders, orderings)
    choice, ties = _lexicographic_minimum(variants)
    if np.any(ties != 1):
        raise ValueError(
            "A lifted entity is identified with its own reversal and has no "
            "consistent quotient orientation."
        )
    rows = np.arange(count)
    return variants[rows, choice], signs[choice], anchors[rows, choice]


def _unoriented_forms(
    identifiers: np.ndarray, shifts: np.ndarray, fixed: np.ndarray, orders: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Return orientation-free top-cell keys and their anchor image shifts."""

    count, arity = identifiers.shape
    rank = shifts.shape[2]
    _require_distinct_corners(identifiers, shifts, fixed, orders, "cell")
    # One ordering per anchor corner; the remaining corners are sorted below.
    orderings = np.asarray(
        [
            [anchor, *(other for other in range(arity) if other != anchor)]
            for anchor in range(arity)
        ],
        dtype=np.int64,
    )
    variants, anchors = _anchored_variants(identifiers, shifts, fixed, orders, orderings)
    corners = np.concatenate(
        (
            variants[:, :, :arity, None],
            variants[:, :, arity:].reshape((count, arity, arity, rank)),
        ),
        axis=3,
    ).reshape((count * arity * arity, 1 + rank))
    owners = np.repeat(np.arange(count * arity, dtype=np.int64), arity)
    order = np.lexsort((*corners.T[::-1], owners))
    stacked = corners[order].reshape((count, arity, -1))
    choice, _ = _lexicographic_minimum(stacked)
    rows = np.arange(count)
    return stacked[rows, choice], anchors[rows, choice]


def _lifted_loops(
    mesh: CellMesh, degree: int, /
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Lifted edges or faces as oriented corner loops, grouped by arity."""

    connectivity = mesh.connectivity
    match connectivity:
        case IntervalConnectivity():
            raise ValueError("Interval meshes have no intermediate entities.")
        case (
            PolygonalConnectivity() | TetrahedralConnectivity() | HexahedralConnectivity()
        ) if degree == 1:
            edges = np.asarray(connectivity.edges, dtype=np.int64)
            return ((np.arange(edges.shape[0], dtype=np.int64), edges),)
        case TetrahedralConnectivity() | HexahedralConnectivity():
            faces = np.asarray(connectivity.faces, dtype=np.int64)
            return ((np.arange(faces.shape[0], dtype=np.int64), faces),)
        case PolyhedralConnectivity() if degree == 1:
            edges = np.asarray(connectivity.edges, dtype=np.int64)
            return ((np.arange(edges.shape[0], dtype=np.int64), edges),)
        case PolyhedralConnectivity():
            offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
            values = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
            lengths = np.diff(offsets)
            groups = []
            for arity in np.unique(lengths):
                rows = np.flatnonzero(lengths == arity)
                columns = offsets[rows][:, None] + np.arange(arity, dtype=np.int64)
                groups.append((rows, values[columns]))
            return tuple(groups)
        case PolygonalConnectivity():
            raise ValueError("Polygonal meshes have no lifted face entities.")
        case _:
            raise ValueError(
                "Periodic lifted loops require supported edge or face connectivity."
            )


def _lifted_cells(mesh: CellMesh, /) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Active lifted top-cell corners grouped by active corner count."""

    groups = []
    offset = 0
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        counts = np.sum(valid, axis=1)
        for arity in np.unique(counts):
            rows = np.flatnonzero(counts == arity)
            corners = vertices[rows][valid[rows]].reshape((rows.size, arity))
            groups.append((offset + rows, corners))
        offset += block.cell_count
    return tuple(groups)


def _corner_shifts(mesh: CellMesh, shifts: np.ndarray, /) -> np.ndarray:
    """Per-corner lattice shifts relative to the first active corner of each cell."""

    values = []
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        anchors = vertices[np.arange(vertices.shape[0]), np.argmax(valid, axis=1)]
        repeated = np.repeat(anchors, np.sum(valid, axis=1))
        values.append(shifts[vertices[valid]] - shifts[repeated])
    return np.concatenate(values, axis=0)


class _Degree:
    """Host orbit data of one degree before publication."""

    def __init__(
        self,
        orbit: np.ndarray,
        witness: np.ndarray,
        anchor_shift: np.ndarray,
        representatives: np.ndarray,
        keys: tuple[np.ndarray, ...],
        identifiers: np.ndarray,
        orders: np.ndarray,
    ) -> None:
        self.orbit = orbit
        self.witness = witness
        # Group element carrying each lifted entity's orbit representative onto it.
        self.shift = _reduced(anchor_shift - anchor_shift[representatives[orbit]], orders)
        self.representatives = representatives
        self.keys = keys
        self.identifiers = identifiers

    @property
    def count(self) -> int:
        return self.representatives.shape[0]


def _vertex_degree(
    representatives: np.ndarray,
    shifts: np.ndarray,
    vertex_ids: np.ndarray,
    orders: np.ndarray,
    /,
) -> _Degree:
    quotient = np.unique(representatives)
    quotient = quotient[np.argsort(vertex_ids[quotient], kind="stable")]
    ordered_ids = vertex_ids[quotient]
    orbit = np.searchsorted(ordered_ids, vertex_ids[representatives])
    return _Degree(
        orbit,
        np.ones_like(orbit),
        shifts,
        quotient,
        tuple(np.asarray((value,), dtype=np.int64) for value in ordered_ids),
        ordered_ids,
        orders,
    )


def _minimum_id_members(
    orbit: np.ndarray, lifted_ids: np.ndarray, count: int, /
) -> np.ndarray:
    order = np.lexsort((lifted_ids, orbit))
    starts = _row_run_starts(orbit[order][:, None])
    if np.count_nonzero(starts) != count:
        raise ValueError("Every quotient entity requires one lifted member.")
    return order[starts]


def _intermediate_degree(
    mesh: CellMesh,
    degree: int,
    representative_ids: np.ndarray,
    shifts: np.ndarray,
    fixed: np.ndarray,
    orders: np.ndarray,
    supplied: np.ndarray | None,
    /,
) -> _Degree:
    lifted_ids = np.asarray(mesh.topology.entity_sets[degree].entity_ids)
    size = lifted_ids.shape[0]
    witness = np.zeros((size,), dtype=np.int64)
    anchor_shift = np.zeros((size, shifts.shape[1]), dtype=np.int64)
    local_group = np.zeros((size,), dtype=np.int64)
    unique_keys: list[np.ndarray] = []
    for rows, loops in _lifted_loops(mesh, degree):
        keys, signs, anchors = _oriented_forms(
            representative_ids[loops], shifts[loops], fixed[loops], orders
        )
        groups, first = _first_appearance_groups(keys)
        local_group[rows] = groups + len(unique_keys)
        witness[rows] = signs
        anchor_shift[rows] = anchors
        unique_keys.extend(keys[first])
    if np.any(witness == 0):
        raise ValueError("Lifted connectivity does not cover its entity set.")
    offsets = np.zeros((len(unique_keys) + 1,), dtype=np.int64)
    np.cumsum([key.size for key in unique_keys], out=offsets[1:])
    ranks = _variable_row_ranks(offsets, np.concatenate(unique_keys))
    orbit = ranks[local_group]
    count = len(unique_keys)
    ordered_keys = [np.empty((0,), dtype=np.int64)] * count
    for rank, key in zip(ranks, unique_keys, strict=True):
        ordered_keys[rank] = key
    identifiers = np.arange(count, dtype=np.int64) if supplied is None else supplied
    if identifiers.shape != (count,):
        raise ValueError(
            f"Quotient entity IDs of degree {degree} must have shape {(count,)}."
        )
    return _Degree(
        orbit,
        witness,
        anchor_shift,
        _minimum_id_members(orbit, lifted_ids, count),
        tuple(ordered_keys),
        identifiers,
        orders,
    )


def _cell_degree(
    mesh: CellMesh,
    representative_ids: np.ndarray,
    shifts: np.ndarray,
    fixed: np.ndarray,
    orders: np.ndarray,
    /,
) -> _Degree:
    lifted_ids = np.asarray(
        mesh.topology.entity_sets[mesh.topological_dimension].entity_ids
    )
    size = lifted_ids.shape[0]
    anchor_shift = np.zeros((size, shifts.shape[1]), dtype=np.int64)
    keys: list[np.ndarray] = [np.empty((0,), dtype=np.int64)] * size
    for rows, corners in _lifted_cells(mesh):
        values, anchors = _unoriented_forms(
            representative_ids[corners], shifts[corners], fixed[corners], orders
        )
        if _has_duplicate_rows(values):
            raise ValueError(
                "Two lifted cells represent one quotient cell orbit; publish each "
                "top-cell orbit exactly once."
            )
        anchor_shift[rows] = anchors
        for row, value in zip(rows, values, strict=True):
            keys[row] = value
    orbit = np.arange(size, dtype=np.int64)
    return _Degree(
        orbit,
        np.ones_like(orbit),
        anchor_shift,
        orbit,
        tuple(keys),
        lifted_ids,
        orders,
    )


def _accumulated_rows(
    upper: np.ndarray, lower: np.ndarray, coefficients: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum coefficients per (lifted upper entity, quotient lower entity)."""

    pairs = np.stack((upper, lower), axis=1)
    order = _row_order(pairs)
    starts = _row_run_starts(pairs[order])
    sums = np.add.reduceat(coefficients[order], np.flatnonzero(starts))
    first = order[starts]
    kept = sums != 0
    return upper[first][kept], lower[first][kept], sums[kept]


def _require_manifold_quotient(
    lower: np.ndarray, coefficients: np.ndarray, count: int, /
) -> np.ndarray:
    """Return quotient boundary facets after the manifold and orientation checks."""

    sides = np.bincount(lower, minlength=count)
    orientation = np.bincount(lower, weights=coefficients, minlength=count)
    if np.any(sides > 2):
        raise ValueError("Quotient facets must bound at most two top-cell sides.")
    if np.any((sides == 2) & (orientation != 0)):
        raise ValueError(
            "Quotient top cells induce the same orientation on a shared quotient "
            "facet; the periodic identification has an orientation conflict."
        )
    return sides == 1


def _quotient_incidence(
    incidence: OrientedIncidence,
    lower: _Degree,
    upper: _Degree,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return quotient incidence rows plus the raw lifted-to-quotient side rows."""

    valid = np.asarray(incidence.relation.valid, dtype=np.bool_)
    source = np.asarray(incidence.relation.source_indices, dtype=np.int64)[valid]
    target = np.asarray(incidence.relation.target_indices, dtype=np.int64)[valid]
    signs = np.rint(np.asarray(incidence.signs)[valid]).astype(np.int64)
    coefficients = signs * lower.witness[source] * upper.witness[target]
    quotient_lower = lower.orbit[source]
    members, faces, sums = _accumulated_rows(target, quotient_lower, coefficients)
    if np.any(np.abs(sums) != 1):
        raise ValueError(
            "A quotient entity receives a repeated boundary coefficient; the "
            "periodic identification has an orientation conflict."
        )
    orbits = upper.orbit[members]
    is_reference = members == upper.representatives[orbits]
    reference = np.stack(
        (orbits[is_reference], faces[is_reference], sums[is_reference]), axis=1
    )
    reference = reference[_row_order(reference)]
    candidates = np.stack((orbits, faces, sums), axis=1)
    groups, _ = _first_appearance_groups(np.concatenate((reference, candidates)))
    member_counts = np.bincount(members, minlength=upper.orbit.shape[0])
    if np.any(groups[reference.shape[0] :] >= reference.shape[0]) or np.any(
        member_counts != member_counts[upper.representatives[upper.orbit]]
    ):
        raise ValueError(
            "Lifted members of one quotient orbit have different quotient "
            "boundaries; orbit composition is inconsistent."
        )
    return (
        reference[:, 1],
        reference[:, 0],
        reference[:, 2],
        quotient_lower,
        coefficients,
    )


def _require_periodic_lift(
    coordinates: np.ndarray,
    representatives: np.ndarray,
    shifts: np.ndarray,
    cell: PeriodicMeshIdentification,
    /,
    *,
    actual_geometry: CellGeometrySpec | None = None,
) -> None:
    """Require lifted coordinates to be group images of their representatives."""
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("Lifted periodic coordinates must be finite.")
    if actual_geometry is not None:
        bank = _exact_periodic_vertex_source(coordinates, actual_geometry)
        _require_exact_periodic_vertex_lift(bank, representatives, shifts, cell)
        return

    match cell:
        case PeriodicCell():
            vectors = np.asarray(cell.vectors, dtype=np.float64)
            translation = shifts.astype(np.float64) @ vectors
            deviation = coordinates - coordinates[representatives] - translation
            scale = max(
                1.0,
                float(np.max(np.abs(coordinates))),
                float(np.max(np.abs(translation), initial=0.0)),
            )
            # Rounding of one lattice translation and one coordinate difference.
            bound = 64.0 * float(np.finfo(np.asarray(cell.vectors).dtype).eps) * scale
        case PeriodicIsometryGroup():
            images = cell.apply(coordinates[representatives], shifts)
            deviation = np.linalg.norm(coordinates - images, axis=1)
            bound = cell.tolerance
        case _:
            raise TypeError("cell must be a PeriodicCell or PeriodicIsometryGroup.")
    if np.any(np.abs(deviation) > bound):
        raise ValueError(
            "Lifted vertex coordinates are inconsistent with their quotient "
            "representatives and lattice image shifts."
        )


def _periodic_vertex_orbit_indices(
    coordinates: ArrayLike,
    cell: PeriodicMeshIdentification,
    vertex_representatives: ArrayLike,
    vertex_shifts: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate authored integer group lifts without fabricating a cell carrier."""
    if not isinstance(cell, (PeriodicCell, PeriodicIsometryGroup)):
        raise TypeError("cell must be a PeriodicCell or PeriodicIsometryGroup.")
    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != cell.ambient_dimension
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "Periodic vertex coordinates must be finite ambient-dimensional rows."
        )
    representatives = np.asarray(vertex_representatives)
    shifts = np.asarray(vertex_shifts)
    count = points.shape[0]
    if not np.issubdtype(representatives.dtype, np.integer) or not np.issubdtype(
        shifts.dtype, np.integer
    ):
        raise TypeError("Vertex representatives and shifts must be integer arrays.")
    storage = np.iinfo(np.int32)
    if np.any(shifts < storage.min) or np.any(shifts > storage.max):
        raise ValueError("Periodic vertex shifts exceed the int32 image representation.")
    representatives = representatives.astype(np.int64)
    shifts = shifts.astype(np.int64)
    if representatives.shape != (count,) or shifts.shape != (count, cell.rank):
        raise ValueError(
            "Periodic vertex data must hold one representative and one "
            "lattice-rank shift per lifted vertex."
        )
    if np.any(representatives < 0) or np.any(representatives >= count):
        raise ValueError("Vertex representatives must index lifted vertices.")
    if np.any(representatives[representatives] != representatives) or np.any(
        shifts[representatives] != 0
    ):
        raise ValueError(
            "Vertex representatives must represent themselves with zero shift."
        )
    if isinstance(cell, PeriodicCell) and np.any(
        shifts[:, ~np.asarray(cell.periodic_mask, dtype=np.bool_)] != 0
    ):
        raise ValueError("Lattice shifts along nonperiodic axes must be zero.")
    return representatives, shifts


def validate_periodic_vertex_orbits(
    coordinates: ArrayLike,
    cell: PeriodicMeshIdentification,
    vertex_representatives: ArrayLike,
    vertex_shifts: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate authored integer group lifts without fabricating a cell carrier."""
    representatives, shifts = _periodic_vertex_orbit_indices(
        coordinates,
        cell,
        vertex_representatives,
        vertex_shifts,
    )
    _require_periodic_lift(
        np.asarray(coordinates, dtype=np.float64), representatives, shifts, cell
    )
    return representatives, shifts


def _supplied_quotient_ids(
    entity_global_ids: Mapping[int, ArrayLike] | None, dimension: int, /
) -> dict[int, np.ndarray]:
    supplied = (
        {}
        if entity_global_ids is None
        else {
            int(degree): np.asarray(values, dtype=np.int64)
            for degree, values in entity_global_ids.items()
        }
    )
    if any(degree <= 0 or degree >= dimension for degree in supplied):
        raise ValueError(
            "Quotient vertex and cell IDs derive from lifted identities; only "
            "intermediate quotient entity IDs may be supplied."
        )
    return supplied


def _quotient_entity_sets(
    mesh: CellMesh, degrees: tuple[_Degree, ...], boundary: tuple[np.ndarray, ...], /
) -> tuple[EntitySet, ...]:
    return tuple(
        EntitySet(
            f"quotient-{lifted.name}",
            degree,
            data.identifiers,
            subsets=(EntitySubset("boundary", mask),),
        )
        for degree, (lifted, data, mask) in enumerate(
            zip(mesh.topology.entity_sets, degrees, boundary, strict=True)
        )
    )


def _quotient_boundary(
    degrees: tuple[_Degree, ...],
    raw: tuple[tuple[np.ndarray, np.ndarray], ...],
    facet_boundary: np.ndarray,
    /,
) -> tuple[np.ndarray, ...]:
    """Close quotient boundary facets downward through raw lifted incidences."""

    dimension = len(degrees) - 1
    masks = [np.zeros((degree.count,), dtype=np.bool_) for degree in degrees]
    masks[dimension - 1] = facet_boundary
    for degree in range(dimension - 1, 0, -1):
        lower, upper = raw[degree - 1]
        masks[degree - 1][lower[masks[degree][upper]]] = True
    return tuple(masks)


class PeriodicMeshTopology(StrictModule, NonTrainableState):
    """Quotient topology of a lifted periodic `CellMesh`.

    ``cell`` is the bound identification: a translational `PeriodicCell`
    lattice or a `PeriodicIsometryGroup`. ``vertex_representatives`` and
    ``vertex_shifts`` map lifted vertices to their quotient representative and
    integer group exponents. ``corner_shifts`` holds each active cell corner's
    exponents relative to the cell's first corner (the local lift). For every
    lifted entity, concatenated by degree through ``lifted_offsets``,
    ``orbit_indices`` names its quotient entity, ``orbit_orientations`` the
    ``±1`` witness of its orientation relative to the quotient entity, and
    ``orbit_shifts`` the group exponents carrying each orbit's representative
    lifted entity onto its copy. ``representatives`` names that lifted member
    per quotient entity, concatenated through ``quotient_offsets``. ``entity_key_*``
    pack the canonical relative-shift keys of quotient entities. ``quotient`` is
    the validated oriented quotient cell complex.
    ``allocator_next_ids`` retains intermediate-entity allocation high-water
    cursors without changing geometric topology identity. A cursor of ``-1``
    means explicit entity IDs arrived without their allocation history; an edit
    cannot invent that history from the largest currently live ID.
    """

    __strict_contract__ = True

    cell: PeriodicCell | PeriodicIsometryGroup
    quotient: CellComplexTopology
    # Always a `CellGeometrySpec` (validated by `actual_geometry`); contract
    # annotations resolve at runtime, and `_cell_geometry` imports this module
    # through `_cell_mesh`, so the field names its terminal-state base.
    _actual_geometry: NonTrainableState | None
    vertex_fixed_generators: Bool[_LiftedVertexDim, _LatticeRankDim]
    vertex_representatives: Int32[_LiftedVertexDim]
    vertex_shifts: Int32[_LiftedVertexDim, _LatticeRankDim]
    corner_shifts: Int32[_CornerDim, _LatticeRankDim]
    orbit_indices: Int32[_LiftedEntityDim]
    orbit_orientations: Int32[_LiftedEntityDim]
    orbit_shifts: Int32[_LiftedEntityDim, _LatticeRankDim]
    representatives: Int32[_QuotientEntityDim]
    lifted_offsets: Int32[_DegreeBoundaryDim]
    quotient_offsets: Int32[_DegreeBoundaryDim]
    entity_key_offsets: Int64[_KeyBoundaryDim]
    entity_key_values: Int64[_KeyValueDim]
    topological_dimension: int = eqx.field(static=True)
    lifted_topology_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    allocator_next_ids: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        lifted: CellMesh,
        cell: PeriodicCell | PeriodicIsometryGroup,
        vertex_representatives: ArrayLike,
        vertex_shifts: ArrayLike,
        /,
        *,
        entity_global_ids: Mapping[int, ArrayLike] | None = None,
        entity_allocator_next_ids: Mapping[int, int] | None = None,
        actual_geometry: CellGeometrySpec | None = None,
    ) -> None:
        from ._cell_geometry_validity import cell_geometry_id
        from ._cell_mesh import CellMesh

        if not isinstance(lifted, CellMesh):
            raise TypeError("lifted must be a CellMesh.")
        if lifted.periodic_topology is not None:
            raise ValueError(
                "lifted must be the plain lifted carrier of a periodic mesh."
            )
        if not isinstance(cell, (PeriodicCell, PeriodicIsometryGroup)):
            raise TypeError("cell must be a PeriodicCell or PeriodicIsometryGroup.")
        if cell.ambient_dimension != lifted.ambient_dimension:
            raise ValueError(
                "The periodic identification and lifted mesh must share one "
                "ambient dimension."
            )
        dimension = lifted.topological_dimension
        source_coordinates = (
            _exact_periodic_vertex_source(
                np.asarray(lifted.coordinates, dtype=np.float64), actual_geometry
            )
            if actual_geometry is not None
            else None
        )
        if actual_geometry is not None:
            _, routes, _ = actual_geometry._resolve(lifted, exact_source_prepared=True)
            if any(
                not np.array_equal(np.asarray(route), np.asarray(block.vertices))
                for block, route in zip(lifted.blocks, routes, strict=True)
            ):
                raise ValueError(
                    "Exact periodic authority must use the actual lifted vertex routes."
                )
        supplied = _supplied_quotient_ids(entity_global_ids, dimension)
        representatives, shifts = _periodic_vertex_orbit_indices(
            lifted.coordinates,
            cell,
            vertex_representatives,
            vertex_shifts,
        )
        if source_coordinates is not None:
            _require_exact_periodic_vertex_lift(
                source_coordinates, representatives, shifts, cell
            )
        else:
            _require_periodic_lift(
                np.asarray(lifted.coordinates, dtype=np.float64),
                representatives,
                shifts,
                cell,
            )
        vertex_ids = np.asarray(lifted.vertex_global_ids, dtype=np.int64)
        representative_ids = vertex_ids[representatives]
        orders = _identification_orders(cell)
        fixed = _fixed_generators(
            cell,
            np.asarray(lifted.coordinates, dtype=np.float64),
            representatives,
            source_coordinates=source_coordinates,
        )
        degrees = (
            _vertex_degree(representatives, shifts, vertex_ids, orders),
            *(
                _intermediate_degree(
                    lifted,
                    degree,
                    representative_ids,
                    shifts,
                    fixed,
                    orders,
                    supplied.get(degree),
                )
                for degree in range(1, dimension)
            ),
            _cell_degree(lifted, representative_ids, shifts, fixed, orders),
        )
        allocation = (
            {} if entity_allocator_next_ids is None else dict(entity_allocator_next_ids)
        )
        if any(degree <= 0 or degree >= dimension for degree in allocation):
            raise ValueError(
                "Quotient allocation cursors name intermediate entity degrees."
            )
        cursors = [-1] * (dimension + 1)
        for degree in range(1, dimension):
            cursor = allocation.get(degree)
            if cursor is None:
                cursors[degree] = degrees[degree].count if degree not in supplied else -1
                continue
            if isinstance(cursor, bool) or not isinstance(cursor, (int, np.integer)):
                raise TypeError("Quotient allocation cursors must be integers.")
            if (
                int(cursor) <= int(np.max(degrees[degree].identifiers))
                or int(cursor) > _INT64_MAX + 1
            ):
                raise ValueError(
                    "Quotient allocation cursor does not exceed every assigned entity ID."
                )
            cursors[degree] = int(cursor)
        rows = tuple(
            _quotient_incidence(incidence, degrees[degree - 1], degrees[degree])
            for degree, incidence in enumerate(lifted.topology.incidences, start=1)
        )
        facet_boundary = _require_manifold_quotient(
            rows[-1][3], rows[-1][4], degrees[-2].count
        )
        raw = tuple(
            (
                row[3],
                degrees[degree].orbit[
                    np.asarray(incidence.relation.target_indices, dtype=np.int64)[
                        np.asarray(incidence.relation.valid, dtype=np.bool_)
                    ]
                ],
            )
            for degree, (row, incidence) in enumerate(
                zip(rows, lifted.topology.incidences, strict=True), start=1
            )
        )
        entity_sets = _quotient_entity_sets(
            lifted, degrees, _quotient_boundary(degrees, raw, facet_boundary)
        )
        quotient = CellComplexTopology(
            entity_sets,
            tuple(
                OrientedIncidence(
                    degree,
                    entity_sets[degree - 1],
                    entity_sets[degree],
                    EdgeRelation(
                        row[0].astype(np.int32),
                        row[1].astype(np.int32),
                        source_size=degrees[degree - 1].count,
                        target_size=degrees[degree].count,
                    ),
                    row[2].astype(np.float64),
                )
                for degree, row in enumerate(rows, start=1)
            ),
        )
        keys = tuple(key for degree in degrees for key in degree.keys)
        key_offsets = np.zeros((len(keys) + 1,), dtype=np.int64)
        np.cumsum([key.size for key in keys], out=key_offsets[1:])
        key_values = np.concatenate(keys)
        lifted_offsets = np.cumsum(
            [0, *(degree.orbit.shape[0] for degree in degrees)], dtype=np.int32
        )
        quotient_offsets = np.cumsum(
            [0, *(degree.count for degree in degrees)], dtype=np.int32
        )
        corner_shifts = _corner_shifts(lifted, shifts)
        orbit_indices = np.concatenate([degree.orbit for degree in degrees])
        orbit_orientations = np.concatenate([degree.witness for degree in degrees])
        orbit_shifts = np.concatenate([degree.shift for degree in degrees])
        quotient_representatives = np.concatenate(
            [degree.representatives for degree in degrees]
        )
        storage = np.iinfo(np.int32)
        if any(
            np.any(values < storage.min) or np.any(values > storage.max)
            for values in (corner_shifts, orbit_shifts)
        ):
            raise ValueError(
                "Relative periodic shifts exceed the int32 image representation."
            )
        self.cell = cell
        self.quotient = quotient
        self._actual_geometry = actual_geometry
        self.vertex_fixed_generators = jnp.asarray(fixed, dtype=jnp.bool_)
        self.vertex_representatives = jnp.asarray(representatives, dtype=jnp.int32)
        self.vertex_shifts = jnp.asarray(shifts, dtype=jnp.int32)
        self.corner_shifts = jnp.asarray(corner_shifts, dtype=jnp.int32)
        self.orbit_indices = jnp.asarray(orbit_indices, dtype=jnp.int32)
        self.orbit_orientations = jnp.asarray(orbit_orientations, dtype=jnp.int32)
        self.orbit_shifts = jnp.asarray(orbit_shifts, dtype=jnp.int32)
        self.representatives = jnp.asarray(quotient_representatives, dtype=jnp.int32)
        self.lifted_offsets = jnp.asarray(lifted_offsets, dtype=jnp.int32)
        self.quotient_offsets = jnp.asarray(quotient_offsets, dtype=jnp.int32)
        self.entity_key_offsets = jnp.asarray(key_offsets, dtype=jnp.int64)
        self.entity_key_values = jnp.asarray(key_values, dtype=jnp.int64)
        self.topological_dimension = dimension
        self.lifted_topology_id = lifted.topology_id
        self.allocator_next_ids = tuple(cursors)
        self.periodic_topology_id = canonical_fingerprint(
            {
                "kind": "periodic-mesh-topology",
                "cell": _identification_id(cell),
                "lifted_topology": lifted.topology_id,
                "quotient": quotient.topology_id,
                **(
                    {"actual_geometry": cell_geometry_id(actual_geometry)}
                    if actual_geometry is not None
                    else {}
                ),
                "arrays": array_tree_fingerprint(
                    {
                        "vertex_representatives": representatives,
                        "vertex_shifts": shifts,
                        "orbit_orientations": orbit_orientations,
                        "orbit_shifts": orbit_shifts,
                        "entity_key_offsets": key_offsets,
                        "entity_key_values": key_values,
                    }
                ),
            }
        )

    def _degree(self, degree: int, /) -> int:
        index = int(degree)
        if index < 0 or index > self.topological_dimension:
            raise ValueError(f"degree must lie in [0, {self.topological_dimension}].")
        return index

    @property
    def actual_geometry(self) -> CellGeometrySpec | None:
        """Independent exact vertex geometry authenticated by this descriptor."""
        from ._cell_geometry import CellGeometrySpec

        geometry = self._actual_geometry
        if geometry is not None and not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Periodic source authority must retain its actual CellGeometrySpec."
            )
        return geometry

    def rebuilt(self, lifted: CellMesh, /) -> PeriodicMeshTopology:
        """Revalidate this descriptor from its authored inputs on a restored lifted carrier.

        Quotient entity IDs and every resolved allocation cursor are carried
        unchanged; unresolved cursors stay unresolved. All orbit, orientation and
        quotient-complex checks of the constructor run again.
        """

        dimension = self.topological_dimension
        geometry = self.actual_geometry
        if geometry is not None:
            from ._cell_geometry import CellGeometrySpec
            from ._exact_power_geometry import (
                ExactPowerCellGeometryLinearActionSource,
                ExactPowerCellGeometryRestrictionSource,
                ExactPowerCellGeometrySource,
            )

            source = geometry.exact_source
            if not isinstance(
                source,
                (
                    ExactPowerCellGeometrySource,
                    ExactPowerCellGeometryRestrictionSource,
                    ExactPowerCellGeometryLinearActionSource,
                ),
            ):
                raise TypeError(
                    "Periodic route rebuilding requires its actual exact power source."
                )
            geometry = CellGeometrySpec.power(lifted, source)
        probe = PeriodicMeshTopology(
            lifted,
            self.cell,
            self.vertex_representatives,
            self.vertex_shifts,
            actual_geometry=geometry,
        )
        entity_ids = {}
        for degree in range(1, dimension):
            known = dict(
                zip(
                    self.entity_keys(degree),
                    np.asarray(self.quotient.entities(degree).entity_ids),
                    strict=True,
                )
            )
            keys = probe.entity_keys(degree)
            if set(keys) != set(known):
                raise ValueError(
                    "Periodic rebuilding changed the original scientific entity keys."
                )
            entity_ids[degree] = np.asarray([known[key] for key in keys], dtype=np.int64)
        return PeriodicMeshTopology(
            lifted,
            self.cell,
            self.vertex_representatives,
            self.vertex_shifts,
            actual_geometry=geometry,
            entity_global_ids=entity_ids,
            entity_allocator_next_ids={
                degree: cursor
                for degree, cursor in enumerate(self.allocator_next_ids)
                if 0 < degree < dimension and cursor >= 0
            },
        )

    def orbits(self, degree: int, /) -> tuple[Array, Array, Array]:
        """Return quotient index, orientation witness and shift per lifted entity."""

        index = self._degree(degree)
        offsets = np.asarray(self.lifted_offsets)
        start, stop = int(offsets[index]), int(offsets[index + 1])
        return (
            self.orbit_indices[start:stop],
            self.orbit_orientations[start:stop],
            self.orbit_shifts[start:stop],
        )

    def orbit_representatives(self, degree: int, /) -> Array:
        """Return the representative lifted entity of each quotient entity."""

        index = self._degree(degree)
        offsets = np.asarray(self.quotient_offsets)
        return self.representatives[int(offsets[index]) : int(offsets[index + 1])]

    def entity_vertex_permutations(
        self, lifted: CellMesh, degree: int, /
    ) -> tuple[np.ndarray, ...]:
        """Map each lifted entity's corner positions to its representative's.

        Matching uses scientific vertex representatives and relative group
        exponents, not nearest coordinates. Winding edges therefore remain
        distinct even when their endpoint representatives coincide.
        """

        index = self._degree(degree)
        if (
            lifted.periodic_topology is not self
            and lifted.topology_id != self.lifted_topology_id
        ):
            raise ValueError("The mesh is not the bound lifted topology.")
        if not 0 < index < self.topological_dimension:
            raise ValueError("Vertex permutations require an edge or face degree.")
        orbit, _, anchors = (
            np.asarray(value, dtype=np.int64) for value in self.orbits(index)
        )
        members = np.asarray(self.orbit_representatives(index))
        vertices = np.asarray(self.vertex_representatives)
        shifts = np.asarray(self.vertex_shifts, dtype=np.int64)
        orders = _identification_orders(self.cell)
        fixed = np.asarray(self.vertex_fixed_generators, dtype=np.bool_)
        loops = {}
        for rows, corners in _lifted_loops(lifted, index):
            loops.update(zip(rows.tolist(), corners, strict=True))
        result = []
        for entity in range(orbit.size):
            corners = loops[entity]
            representative = int(members[orbit[entity]])
            base = loops[representative]
            delta = anchors[entity] - anchors[representative]
            difference = _reduced(
                shifts[corners, None, :] - shifts[base][None, :, :] - delta,
                orders,
            )
            matches = (vertices[corners, None] == vertices[base][None, :]) & np.all(
                (difference == 0) | fixed[corners, None, :], axis=-1
            )
            if np.any(np.sum(matches, axis=1) != 1) or np.any(
                np.sum(matches, axis=0) != 1
            ):
                raise ValueError("An entity orbit has no unique corner permutation.")
            result.append(np.argmax(matches, axis=1).astype(np.int32))
        return tuple(result)

    def orbit_isometries(self, degree: int, /) -> np.ndarray:
        """Homogeneous maps from representative lifted entities to their copies."""

        orbit, _, anchors = (
            np.asarray(value, dtype=np.int64) for value in self.orbits(degree)
        )
        members = np.asarray(self.orbit_representatives(degree))
        exponents = anchors - anchors[members[orbit]]
        dimension = self.cell.ambient_dimension
        matrices = np.broadcast_to(
            np.eye(dimension + 1), (orbit.size, dimension + 1, dimension + 1)
        ).copy()
        match self.cell:
            case PeriodicCell():
                matrices[:, :dimension, dimension] = exponents @ np.asarray(
                    self.cell.vectors
                )
            case PeriodicIsometryGroup():
                for row, exponent in enumerate(exponents):
                    matrices[row] = self.cell.element(exponent)
        return matrices

    def entity_keys(self, degree: int, /) -> tuple[tuple[int, ...], ...]:
        """Return canonical relative-shift keys of the quotient entities."""

        index = self._degree(degree)
        offsets = np.asarray(self.quotient_offsets)
        key_offsets = np.asarray(self.entity_key_offsets)
        values = np.asarray(self.entity_key_values)
        return tuple(
            tuple(int(value) for value in values[key_offsets[row] : key_offsets[row + 1]])
            for row in range(int(offsets[index]), int(offsets[index + 1]))
        )

    def identification(self, degree: int, /) -> SparseLinearMap:
        """Return the signed quotient-to-lifted gather of oriented degree cochains.

        Its transpose accumulates lifted contributions onto quotient entities.
        """

        orbit, orientation, _ = self.orbits(degree)
        lifted_count = orbit.shape[0]
        return SparseLinearMap(
            EdgeRelation(
                np.asarray(orbit),
                np.arange(lifted_count, dtype=np.int32),
                source_size=self.quotient.entities(degree).count,
                target_size=lifted_count,
            ),
            np.asarray(orientation, dtype=np.float64),
            operator_id=canonical_fingerprint(
                {
                    "kind": "periodic-identification",
                    "degree": int(degree),
                    "periodic_topology": self.periodic_topology_id,
                }
            ),
        )

    @property
    def euler_characteristic(self) -> int:
        """Alternating quotient entity count."""

        return sum(
            (-1) ** degree * entities.count
            for degree, entities in enumerate(self.quotient.entity_sets)
        )

    def require_lift(self, coordinates: ArrayLike, /) -> None:
        """Require coordinates to remain lattice images of their representatives."""

        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != (
            self.vertex_representatives.shape[0],
            self.cell.ambient_dimension,
        ):
            raise ValueError("Coordinates must match the lifted periodic vertices.")
        _require_periodic_lift(
            points,
            np.asarray(self.vertex_representatives, dtype=np.int64),
            np.asarray(self.vertex_shifts, dtype=np.int64),
            self.cell,
            actual_geometry=self.actual_geometry,
        )


class PeriodicMeasureReport(StrictModule, NonTrainableState):
    """Measures of quotient top cells, each evaluated once in its local lift.

    ``relative_coverage_defect`` compares the total with the lattice cell
    measure when a fully periodic lattice, mesh and ambient dimensions agree, and is ``None``
    otherwise; an isometry group has no intrinsic fundamental-domain measure,
    so ``lattice_measure`` is ``None`` for it.
    """

    __strict_contract__ = True

    orbit_measures: Float64[_QuotientCellDim]
    valid: Bool[_QuotientCellDim]
    total_measure: Float64[Scalar]
    lattice_measure: float | None = eqx.field(static=True)
    relative_coverage_defect: float | None = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)


def periodic_orbit_measures(
    mesh: CellMesh,
    /,
    *,
    geometry: CellGeometrySpec | None = None,
) -> PeriodicMeasureReport:
    """Integrate the measure of every quotient top cell exactly once.

    Each lifted top cell is the unique published member of its orbit, so the
    finite-element coordinate map of its local lift integrates the orbit once.
    The rule is exact for the Jacobian determinant of the coordinate element.
    """

    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh
    from .fem import (
        discontinuous_element,
        FiniteElementFieldSpec,
        FiniteElementPlan,
        PreparedFiniteElementCellMap,
    )
    from .fem._generic import _degree_aware_reference_rule

    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be a CellMesh.")
    periodic = mesh.periodic_topology
    if periodic is None:
        raise ValueError("Periodic orbit measures require a periodic CellMesh.")
    spec = CellGeometrySpec.affine(mesh) if geometry is None else geometry
    if not isinstance(spec, CellGeometrySpec):
        raise TypeError("geometry must be CellGeometrySpec or None.")
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "periodic-measure",
            {
                block.name: discontinuous_element(block.cell_kind, 0)
                for block in mesh.blocks
            },
        ),
        coordinate_spec=spec,
    ).prepare()
    coordinates = discretization.default_runtime.coordinates
    measures = []
    valid = []
    for index, block in enumerate(mesh.blocks):
        cell_map = PreparedFiniteElementCellMap(discretization, index)
        points, weights = _degree_aware_reference_rule(
            block.cell_kind,
            cell_map.reference_dimension * cell_map.coordinate_element.degree,
        )
        point_count = weights.shape[0]
        evaluation = cell_map.evaluate(
            coordinates,
            jnp.repeat(jnp.arange(block.cell_count, dtype=jnp.int32), point_count),
            jnp.tile(points, (block.cell_count, 1)),
        )
        density = evaluation.measure.reshape((block.cell_count, point_count))
        measures.append(density @ weights)
        valid.append(jnp.all(evaluation.valid.reshape(density.shape), axis=1))
    orbit_measures = jnp.concatenate(measures)
    total = jnp.sum(orbit_measures)
    cell = periodic.cell
    match cell:
        case PeriodicCell():
            lattice = cell.cell_measure
            comparable = (
                cell.fully_periodic
                and cell.rank == mesh.topological_dimension == mesh.ambient_dimension
            )
        case PeriodicIsometryGroup():
            lattice = None
            comparable = False
        case _:
            raise TypeError("cell must be a PeriodicCell or PeriodicIsometryGroup.")
    return PeriodicMeasureReport(
        orbit_measures=orbit_measures,
        valid=jnp.concatenate(valid),
        total_measure=total,
        lattice_measure=lattice,
        relative_coverage_defect=(
            abs(float(total) - lattice) / lattice
            if comparable and lattice is not None
            else None
        ),
        periodic_topology_id=periodic.periodic_topology_id,
        geometry_layout_id=discretization.default_runtime.geometry_layout_id,
    )


class PeriodicIsometryIdentityError(ValueError):
    """Authored numerical generators do not establish an exact scientific group."""


def _reserve_exact_isometry_work(work: int, /) -> None:
    from ._coordinate_enclosure import _COORDINATE_BUDGET

    budget = _COORDINATE_BUDGET.get()
    if budget is not None:
        budget.admit_work_bound(work)
        budget.reserve(work)
    else:
        from .._meshcore import current_native_execution_budget

        native = current_native_execution_budget()
        if native is not None:
            native.charge(work=work)


def _exact_isometry_identity(dimension: int) -> tuple[tuple[Fraction, ...], ...]:
    _reserve_exact_isometry_work((dimension + 1) ** 2)
    return tuple(
        tuple(Fraction(i == j) for j in range(dimension + 1))
        for i in range(dimension + 1)
    )


def _exact_isometry_multiply(
    first: tuple[tuple[Fraction, ...], ...], second: tuple[tuple[Fraction, ...], ...]
) -> tuple[tuple[Fraction, ...], ...]:
    _reserve_exact_isometry_work(len(first) * len(second) ** 2)
    return tuple(
        tuple(
            sum((a * second[k][j] for k, a in enumerate(row)), Fraction(0))
            for j in range(len(second))
        )
        for row in first
    )


def _exact_isometry_power(
    matrix: tuple[tuple[Fraction, ...], ...], exponent: int
) -> tuple[tuple[Fraction, ...], ...]:
    if exponent < 0:
        dimension = len(matrix) - 1
        _reserve_exact_isometry_work((dimension + 1) ** 3)
        inverse = tuple(
            tuple(matrix[j][i] for j in range(dimension))
            + (
                -sum(
                    (matrix[j][i] * matrix[j][-1] for j in range(dimension)), Fraction(0)
                ),
            )
            for i in range(dimension)
        ) + (matrix[-1],)
        matrix, exponent = inverse, -exponent
    result = _exact_isometry_identity(len(matrix) - 1)
    while exponent:
        if exponent & 1:
            result = _exact_isometry_multiply(result, matrix)
        exponent >>= 1
        if exponent:
            matrix = _exact_isometry_multiply(matrix, matrix)
    return result


def _exact_group_solve(
    matrix: tuple[tuple[Fraction, ...], ...],
    rhs: tuple[Fraction, ...],
    /,
) -> tuple[Fraction, ...]:
    """Solve a nonsingular rational system, reserving elimination work."""
    size = len(rhs)
    _reserve_exact_isometry_work(size * size * (size + 1))
    rows = [list(row) + [value] for row, value in zip(matrix, rhs, strict=True)]
    for column in range(size):
        pivot = next((i for i in range(column, size) if rows[i][column]), None)
        if pivot is None:
            raise PeriodicIsometryIdentityError(
                "Derived translation periods have singular exact Gram matrix; original exponents are nondiscrete or nonfaithful."
            )
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for i in range(size):
            if i != column:
                factor = rows[i][column]
                rows[i] = [
                    a - factor * b for a, b in zip(rows[i], rows[column], strict=True)
                ]
    return tuple(row[-1] for row in rows)


def _admit_exact_isometry_group(
    source: np.ndarray,
    /,
) -> tuple[
    tuple[tuple[tuple[Fraction, ...], ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Derive closure from the original binary64 homogeneous source, without snapping."""
    dimension = source.shape[1] - 1
    _reserve_exact_isometry_work(source.size)
    matrices = tuple(
        tuple(tuple(Fraction(float(v)) for v in row) for row in matrix)
        for matrix in source
    )
    identity = _exact_isometry_identity(dimension)
    orders, linear_orders, periods, vectors = [], [], [], []
    for index, matrix in enumerate(matrices):
        if matrix[-1] != identity[-1]:
            raise PeriodicIsometryIdentityError(
                "Authored periodic transformation is not exactly homogeneous affine."
            )
        if matrix == identity:
            raise PeriodicIsometryIdentityError(
                f"generators[{index}] is the exact identity."
            )
        linear = tuple(
            tuple(row[j] for j in range(dimension)) + (Fraction(0),)
            for row in matrix[:-1]
        ) + (identity[-1],)
        _reserve_exact_isometry_work(dimension**3)
        if any(
            sum((matrix[k][i] * matrix[k][j] for k in range(dimension)), Fraction(0))
            != Fraction(i == j)
            for i in range(dimension)
            for j in range(dimension)
        ):
            raise PeriodicIsometryIdentityError(
                "Authored Q has no exact Euclidean isometry identity."
            )
        rows = [list(row[:dimension]) for row in matrix[:dimension]]
        determinant = Fraction(1)
        _reserve_exact_isometry_work(dimension**3)
        for column in range(dimension):
            pivot = next(i for i in range(column, dimension) if rows[i][column])
            if pivot != column:
                rows[column], rows[pivot] = rows[pivot], rows[column]
                determinant = -determinant
            divisor = rows[column][column]
            determinant *= divisor
            for i in range(column + 1, dimension):
                factor = rows[i][column] / divisor
                for j in range(column + 1, dimension):
                    rows[i][j] -= factor * rows[column][j]
        if determinant != 1:
            raise PeriodicIsometryIdentityError("Authored Q is not proper.")
        power = identity
        for order in range(1, _MAXIMUM_ROTATION_ORDER + 1):
            power = _exact_isometry_multiply(power, linear)
            if power == identity:
                break
        else:
            raise PeriodicIsometryIdentityError(
                f"generators[{index}] has unsupported or unproven finite exact linear order at most {_MAXIMUM_ROTATION_ORDER}; discreteness is not established."
            )
        cycle = _exact_isometry_power(matrix, order)
        full_order = order if cycle == identity else 0
        orders.append(full_order)
        linear_orders.append(order)
        periods.append(0 if full_order else order)
        if not full_order:
            vectors.append(tuple(row[-1] for row in cycle[:-1]))
    if any(
        _exact_isometry_multiply(a, b) != _exact_isometry_multiply(b, a)
        for i, a in enumerate(matrices)
        for b in matrices[i + 1 :]
    ):
        raise PeriodicIsometryIdentityError(
            "Authored periodic transformations do not commute exactly."
        )
    _reserve_exact_isometry_work(len(vectors) ** 2 * dimension)
    gram = tuple(
        tuple(
            sum((a * b for a, b in zip(v, w, strict=True)), Fraction(0)) for w in vectors
        )
        for v in vectors
    )
    _exact_group_solve(gram, (Fraction(0),) * len(vectors))
    count = 1
    for order in linear_orders:
        count *= order
    if count > _MAXIMUM_GROUP_IMAGES:
        raise PeriodicIsometryIdentityError(
            f"Exact original exponent relation validation exceeds {_MAXIMUM_GROUP_IMAGES} images."
        )
    for exponents in product(*(range(order) for order in linear_orders)):
        if not any(exponents):
            continue
        action = _exact_periodic_element(matrices, tuple(orders), exponents)
        if any(
            action[i][j] != identity[i][j]
            for i in range(dimension)
            for j in range(dimension)
        ):
            continue
        translation = tuple(row[-1] for row in action[:-1])
        _reserve_exact_isometry_work(len(vectors) * dimension * 2)
        rhs = tuple(
            sum((a * b for a, b in zip(v, translation, strict=True)), Fraction(0))
            for v in vectors
        )
        coefficients = _exact_group_solve(gram, rhs)
        if all(value.denominator == 1 for value in coefficients) and all(
            sum(
                (c * v[i] for c, v in zip(coefficients, vectors, strict=True)),
                Fraction(0),
            )
            == translation[i]
            for i in range(dimension)
        ):
            raise PeriodicIsometryIdentityError(
                "Original generators have a nontrivial exact exponent relation."
            )
    return matrices, tuple(orders), tuple(linear_orders), tuple(periods)


def _exact_periodic_generators(
    cell: PeriodicCell | PeriodicIsometryGroup,
) -> tuple[tuple[tuple[tuple[Fraction, ...], ...], ...], tuple[int, ...]]:
    dimension = cell.ambient_dimension
    if isinstance(cell, PeriodicCell):
        generators = []
        for vector in np.asarray(cell.vectors):
            matrix = [list(row) for row in _exact_isometry_identity(dimension)]
            for i, value in enumerate(vector):
                _reserve_exact_isometry_work(1)
                matrix[i][-1] = Fraction(float(value))
            generators.append(tuple(tuple(row) for row in matrix))
        return tuple(generators), tuple(0 if axis else 1 for axis in cell.periodic_axes)
    matrices, orders, linear_orders, periods = _admit_exact_isometry_group(
        np.asarray(cell.generators)
    )
    if (orders, linear_orders, periods) != (
        cell.orders,
        cell.linear_orders,
        cell.translation_periods,
    ):
        raise PeriodicIsometryIdentityError(
            "Periodic group metadata is not derived from original generators."
        )
    return matrices, orders


def _exact_periodic_element(
    matrices: tuple[tuple[tuple[Fraction, ...], ...], ...],
    orders: tuple[int, ...],
    exponents: tuple[int, ...],
) -> tuple[tuple[Fraction, ...], ...]:
    result = _exact_isometry_identity(len(matrices[0]) - 1)
    for matrix, order, exponent in zip(matrices, orders, exponents, strict=True):
        result = _exact_isometry_multiply(
            result, _exact_isometry_power(matrix, exponent % order if order else exponent)
        )
    return result


__all__ = [
    "PeriodicIsometryGroup",
    "PeriodicMeasureReport",
    "PeriodicMeshTopology",
    "periodic_orbit_measures",
    "validate_periodic_vertex_orbits",
]
