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
independently rounded image coordinates as zero-residual group constructions.

A lifted entity with oriented corners ``(r_i, s_i)`` is identified with every
group image ``(r_i, s_i + t)``. Exponents are reduced modulo finite generator
orders, and a corner fixed by a generator (a vertex on a rotation axis) has no
image along it, so its exponent is dropped. Its entity key takes the
lexicographically smallest oriented form over the admissible starting corners
and directions, with representative global IDs followed by the corner
exponents relative to the starting corner. The key therefore keeps winding:
two quotient edges joining the same representatives with different relative
shifts are distinct entities. The direction of the chosen form is the
orientation sign of the lifted entity relative to its quotient entity. Top
cells are keyed without orientation and must be pairwise distinct orbits, so
every quotient top cell is integrated exactly once in its local lift.

Quotient incidences are the lifted signed incidences of one representative
member, mapped through the orientation signs and accumulated. Every other member of an
orbit must produce the same quotient boundary (orbit composition), accumulated
coefficients must be ``±1`` (orientation), and each quotient facet may bound at
most two top-cell sides with opposite induced orientation.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack
from fractions import Fraction
from itertools import product
from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax.discretization._topology_rows import (
    has_duplicate_rows,
    row_order,
    row_run_starts,
)

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import integer
from ..sparse import EdgeRelation, SparseLinearMap
from ..typing import Bool, checked, Dim, Float64, Int32, Int64, Scalar
from . import (
    _cell_geometry as cell_geometry_owner,
    _cell_mesh as cell_mesh_owner,
    _exact_power_geometry as exact_power_geometry_owner,
    _periodic_exact as periodic_exact_owner,
)
from ._cell_complex import (
    first_appearance_groups,
    IntervalConnectivity,
    PolygonalConnectivity,
    PolyhedralConnectivity,
    TetrahedralConnectivity,
    variable_row_ranks,
)
from ._coordinate_budget import CoordinateEnclosureBudget
from ._hexahedral import HexahedralConnectivity
from ._periodic_cell import PeriodicCell
from ._periodic_exact import (
    exact_periodic_element,
    exact_periodic_generators,
    identification_id,
    identification_orders,
    isometry_element_work_upper,
    isometry_group_work_upper,
    owned_periodic_preparation,
    PeriodicIsometryGroup,
    PeriodicIsometryIdentityError,
    PeriodicMeshIdentification,
    reduce_periodic_exponents,
    reserve_exact_isometry_work,
)
from ._topology import CellComplexTopology, EntitySet, EntitySubset, OrientedIncidence


if TYPE_CHECKING:
    from ._cell_geometry import CellGeometrySpec
    from ._cell_mesh import CellMesh
    from ._coordinate_enclosure import CoordinateSourceBank, ExactScalar


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


_INT64_MAX = np.iinfo(np.int64).max


@checked
def periodic_preparation_bounds(
    cell: PeriodicMeshIdentification,
    point_count: int,
    shifts: np.ndarray,
    /,
    *,
    actual_geometry: cell_geometry_owner.CellGeometrySpec | None = None,
    stabilizers: bool = False,
) -> tuple[int, int, int]:
    """Bound the supported group, point-action and source-preparation circuits."""
    rank, dimension = cell.rank, cell.ambient_dimension
    homogeneous = dimension + 1
    exponent_bits = (
        max(
            (abs(int(value)).bit_length() for value in shifts.reshape(-1)),
            default=0,
        )
        if np.issubdtype(shifts.dtype, np.integer)
        else 0
    )
    exponent_bits = max(
        exponent_bits + 1, periodic_exact_owner.MAXIMUM_ROTATION_ORDER.bit_length()
    )
    match cell:
        case PeriodicCell():
            group_work = rank * (homogeneous**2 + dimension)
            images = 1
        case PeriodicIsometryGroup():
            rank = cell.generators.shape[0]
            dimension = cell.generators.shape[1] - 1
            homogeneous = dimension + 1
            group_work = isometry_group_work_upper(rank, dimension)
            images = int(
                np.prod(
                    tuple(order if order else 1 for order in cell.orders), dtype=object
                )
            )
        case _:
            raise TypeError(
                "PeriodicMeshTopology.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )
    point_action = dimension * homogeneous
    work = group_work + point_count * (
        isometry_element_work_upper(rank, dimension, exponent_bits) + point_action
    )
    if stabilizers:
        work += group_work + point_count * rank * point_action
        work += (
            point_count
            * images
            * (
                isometry_element_work_upper(
                    rank,
                    dimension,
                    periodic_exact_owner.MAXIMUM_ROTATION_ORDER.bit_length(),
                )
                + point_action
            )
        )
    # Binary64 dyadics, finite linear powers, translation exponents and rational
    # elimination minors bound numerator/denominator lengths without expansion.
    binary64_bits = (
        float(np.finfo(np.float64).smallest_subnormal).as_integer_ratio()[1].bit_length()
    )
    coefficient_bits = (
        periodic_exact_owner.MAXIMUM_ROTATION_ORDER * binary64_bits
        + exponent_bits
        + rank.bit_length()
        + homogeneous.bit_length()
    ) * (rank + dimension + 1) ** 2
    entries = (4 * rank + 8 + point_count) * homogeneous**2
    entries += rank * (rank + 1) + point_count * dimension
    storage = entries * (512 + 2 * ((coefficient_bits + 7) // 8))
    queries = 0
    if actual_geometry is not None and actual_geometry.exact_source is not None:
        source_work, source_bytes = cell_geometry_owner.exact_source_preparation_bounds(
            actual_geometry.exact_source, point_count, dimension
        )
        work += source_work
        storage += source_bytes
        queries = source_work
    return work, storage, queries


def _exact_periodic_vertex_source(
    coordinates: np.ndarray,
    geometry: cell_geometry_owner.CellGeometrySpec,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CoordinateSourceBank:
    """Validate the vertex-indexed coordinate source and return its rational bank."""

    if geometry.exact_source is None:
        raise ValueError(
            "PeriodicMeshTopology.actual_geometry must retain a coordinate source for vertex lifts; got None"
        )
    if geometry.periodic_source is not None:
        raise ValueError(
            "PeriodicMeshTopology.actual_geometry must not retain a periodic_source; got another periodic binding"
        )
    if any(
        not isinstance(element, cell_geometry_owner.CellVertexGeometryElement)
        for element in geometry.elements
    ):
        raise ValueError(
            "PeriodicMeshTopology.actual_geometry must use vertex-indexed coordinate elements; got another coordinate layout"
        )
    numeric = np.asarray(geometry.coordinates, dtype=np.float64)
    if numeric.shape != coordinates.shape or not np.array_equal(
        numeric.view(np.uint64),
        coordinates.view(np.uint64),
    ):
        raise ValueError(
            "PeriodicMeshTopology.actual_geometry must equal the rounded lifted carrier bitwise; got different coordinates"
        )
    bank = geometry.source_coordinates(budget=budget)
    if len(bank) != coordinates.shape[0] or any(
        len(row) != coordinates.shape[1] for row in bank
    ):
        raise ValueError(
            "PeriodicMeshTopology.actual_geometry must provide every lifted vertex coordinate; got a partial source bank"
        )
    return bank


def _exact_periodic_point_image(
    matrix: tuple[tuple[Fraction, ...], ...],
    point: tuple[ExactScalar, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ExactScalar, ...]:
    reserve_exact_isometry_work(len(point) * (len(point) + 1), budget=budget)
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
    bank: CoordinateSourceBank,
    representatives: np.ndarray,
    shifts: np.ndarray,
    cell: PeriodicMeshIdentification,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> None:
    matrices, orders = exact_periodic_generators(cell, budget=budget)
    for row, root, shift in zip(bank, representatives, shifts, strict=True):
        action = exact_periodic_element(
            matrices,
            orders,
            tuple(int(value) for value in shift),
            budget=budget,
        )
        if row != _exact_periodic_point_image(action, bank[int(root)], budget=budget):
            raise ValueError(
                "PeriodicMeshTopology.vertex_shifts must reproduce source vertices from quotient representatives; got an inconsistent source group image"
            )


def fixed_generators(
    cell: PeriodicMeshIdentification,
    coordinates: np.ndarray,
    representatives: np.ndarray,
    /,
    *,
    source_coordinates: CoordinateSourceBank | None = None,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Per lifted vertex, whether each generator fixes its quotient vertex."""
    if source_coordinates is not None:
        matrices, orders = exact_periodic_generators(cell, budget=budget)
        fixed = np.asarray(
            [
                [
                    _exact_periodic_point_image(
                        matrix, source_coordinates[int(root)], budget=budget
                    )
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
                    action = exact_periodic_element(
                        matrices, orders, exponents, budget=budget
                    )
                    if _exact_periodic_point_image(action, point, budget=budget) == point:
                        raise PeriodicIsometryIdentityError(
                            "PeriodicMeshTopology.vertex_fixed_generators must represent every source vertex stabilizer; got an unsupported composed stabilizer"
                        )
        return fixed

    match cell:
        case PeriodicCell():
            return np.zeros((representatives.shape[0], cell.rank), dtype=np.bool_)
        case PeriodicIsometryGroup():
            return cell.fixed(coordinates[representatives])
        case _:
            raise TypeError(
                "PeriodicMeshTopology.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )


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
            "PeriodicMeshTopology.entity_key_values must have a corner anchoring each group image; got an entity without an admissible anchor"
        )
    anchors = np.where(invariant, 0, ordered_shifts[:, :, 0])
    relative = reduce_periodic_exponents(
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
    difference = reduce_periodic_exponents(
        shifts[:, :, None, :] - shifts[:, None, :, :], orders
    )
    same = (identifiers[:, :, None] == identifiers[:, None, :]) & np.all(
        (difference == 0) | fixed[:, :, None, :], axis=-1
    )
    same[:, np.arange(identifiers.shape[1]), np.arange(identifiers.shape[1])] = False
    if np.any(same):
        raise ValueError(
            f"PeriodicMeshTopology.entity_key_values must distinguish every lifted {noun} corner image; got a repeated quotient vertex image"
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


def oriented_forms(
    identifiers: np.ndarray, shifts: np.ndarray, fixed: np.ndarray, orders: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return lexicographically minimal keys, orientation signs, and anchor shifts."""

    count, arity = identifiers.shape
    _require_distinct_corners(identifiers, shifts, fixed, orders, "edge or face")
    orderings, signs = _oriented_orders(arity)
    variants, anchors = _anchored_variants(identifiers, shifts, fixed, orders, orderings)
    choice, ties = _lexicographic_minimum(variants)
    if np.any(ties != 1):
        raise ValueError(
            "PeriodicMeshTopology.orbit_orientations must distinguish an entity from its reversal; got a self-reversing quotient entity"
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


@checked
def lifted_loops(
    mesh: cell_mesh_owner.CellMesh, degree: int, /
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Lifted edges or faces as oriented corner loops, grouped by arity."""

    connectivity = mesh.connectivity
    match connectivity:
        case IntervalConnectivity():
            raise ValueError(
                "lifted_loops.mesh must provide intermediate entities; got interval connectivity"
            )
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
            raise ValueError(
                "lifted_loops.degree must name available intermediate entities; got polygonal face degree"
            )
        case _:
            raise ValueError(
                "lifted_loops.mesh must provide supported edge or face connectivity; got another connectivity kind"
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
        self.shift = reduce_periodic_exponents(
            anchor_shift - anchor_shift[representatives[orbit]], orders
        )
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
    starts = row_run_starts(orbit[order][:, None])
    if np.count_nonzero(starts) != count:
        raise ValueError(
            "PeriodicMeshTopology.representatives must provide a lifted member for every quotient entity; got an empty orbit"
        )
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
    for rows, loops in lifted_loops(mesh, degree):
        keys, signs, anchors = oriented_forms(
            representative_ids[loops], shifts[loops], fixed[loops], orders
        )
        groups, first = first_appearance_groups(keys)
        local_group[rows] = groups + len(unique_keys)
        witness[rows] = signs
        anchor_shift[rows] = anchors
        unique_keys.extend(keys[first])
    if np.any(witness == 0):
        raise ValueError(
            "PeriodicMeshTopology.lifted_topology_id must bind connectivity covering every lifted entity; got incomplete entity coverage"
        )
    offsets = np.zeros((len(unique_keys) + 1,), dtype=np.int64)
    np.cumsum([key.size for key in unique_keys], out=offsets[1:])
    ranks = variable_row_ranks(offsets, np.concatenate(unique_keys))
    orbit = ranks[local_group]
    count = len(unique_keys)
    ordered_keys = [np.empty((0,), dtype=np.int64)] * count
    for rank, key in zip(ranks, unique_keys, strict=True):
        ordered_keys[rank] = key
    identifiers = np.arange(count, dtype=np.int64) if supplied is None else supplied
    if identifiers.shape != (count,):
        raise ValueError(
            f"PeriodicMeshTopology.entity_global_ids must have shape {(count,)} at degree {degree}; got an incompatible ID bank"
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
        if has_duplicate_rows(values):
            raise ValueError(
                "PeriodicMeshTopology.quotient must publish each top-cell orbit once; got repeated lifted cell orbits"
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
    order = row_order(pairs)
    starts = row_run_starts(pairs[order])
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
        raise ValueError(
            "PeriodicMeshTopology.quotient must give each facet at most two top-cell sides; got a nonmanifold facet"
        )
    if np.any((sides == 2) & (orientation != 0)):
        raise ValueError(
            "PeriodicMeshTopology.quotient must orient shared facets oppositely on their two top-cell sides; got an orientation conflict"
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
            "PeriodicMeshTopology.quotient must have unit-magnitude boundary coefficients; got a repeated boundary coefficient"
        )
    orbits = upper.orbit[members]
    is_reference = members == upper.representatives[orbits]
    reference = np.stack(
        (orbits[is_reference], faces[is_reference], sums[is_reference]), axis=1
    )
    reference = reference[row_order(reference)]
    candidates = np.stack((orbits, faces, sums), axis=1)
    groups, _ = first_appearance_groups(np.concatenate((reference, candidates)))
    member_counts = np.bincount(members, minlength=upper.orbit.shape[0])
    if np.any(groups[reference.shape[0] :] >= reference.shape[0]) or np.any(
        member_counts != member_counts[upper.representatives[upper.orbit]]
    ):
        raise ValueError(
            "PeriodicMeshTopology.quotient must give all lifted orbit members the same boundary; got inconsistent orbit composition"
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
    budget: CoordinateEnclosureBudget,
) -> None:
    """Require lifted coordinates to be group images of their representatives."""
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(
            "PeriodicMeshTopology.lifted.coordinates must be finite; got a nonfinite coordinate"
        )
    if actual_geometry is not None:
        bank = _exact_periodic_vertex_source(coordinates, actual_geometry, budget=budget)
        _require_exact_periodic_vertex_lift(
            bank, representatives, shifts, cell, budget=budget
        )
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
            raise TypeError(
                "PeriodicMeshTopology.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )
    if np.any(np.abs(deviation) > bound):
        raise ValueError(
            "PeriodicMeshTopology.vertex_shifts must reproduce lifted coordinates from quotient representatives; got an inconsistent group image"
        )


def _periodic_vertex_orbit_indices(
    coordinates: ArrayLike,
    cell: PeriodicMeshIdentification,
    vertex_representatives: ArrayLike,
    vertex_shifts: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate authored integer group lifts without fabricating a cell carrier."""
    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[0] == 0
        or points.shape[1] != cell.ambient_dimension
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            "PeriodicMeshTopology.lifted.coordinates must be finite ambient-dimensional rows; got another coordinate bank"
        )
    representatives = np.asarray(vertex_representatives)
    shifts = np.asarray(vertex_shifts)
    count = points.shape[0]
    if not np.issubdtype(representatives.dtype, np.integer) or not np.issubdtype(
        shifts.dtype, np.integer
    ):
        raise TypeError(
            "PeriodicMeshTopology.vertex_shifts must be integer arrays with integer representatives; got a noninteger dtype"
        )
    storage = np.iinfo(np.int32)
    if np.any(shifts < storage.min) or np.any(shifts > storage.max):
        raise ValueError(
            "PeriodicMeshTopology.vertex_shifts must fit signed int32; got an out-of-range image exponent"
        )
    representatives = representatives.astype(np.int64)
    shifts = shifts.astype(np.int64)
    if representatives.shape != (count,) or shifts.shape != (count, cell.rank):
        raise ValueError(
            "PeriodicMeshTopology.vertex_shifts must align lattice-rank shifts and representatives per lifted vertex; got incompatible bank shapes"
        )
    if np.any(representatives < 0) or np.any(representatives >= count):
        raise ValueError(
            "PeriodicMeshTopology.vertex_representatives must index lifted vertices; got an undeclared vertex row"
        )
    if np.any(representatives[representatives] != representatives) or np.any(
        shifts[representatives] != 0
    ):
        raise ValueError(
            "PeriodicMeshTopology.vertex_representatives must represent themselves with zero shifts; got a nontrivial representative image"
        )
    match cell:
        case PeriodicCell():
            if np.any(shifts[:, ~np.asarray(cell.periodic_mask, dtype=np.bool_)] != 0):
                raise ValueError(
                    "PeriodicMeshTopology.vertex_shifts must be zero along nonperiodic axes; got a nonzero exponent"
                )
        case PeriodicIsometryGroup():
            pass
        case _:
            raise TypeError(
                "PeriodicMeshTopology.cell must be a PeriodicCell or PeriodicIsometryGroup; got another identification kind"
            )
    return representatives, shifts


@checked
def validate_periodic_vertex_orbits(
    coordinates: ArrayLike,
    cell: PeriodicMeshIdentification,
    vertex_representatives: ArrayLike,
    vertex_shifts: ArrayLike,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate authored integer group lifts without fabricating a cell carrier."""
    representatives, shifts = _periodic_vertex_orbit_indices(
        coordinates,
        cell,
        vertex_representatives,
        vertex_shifts,
    )
    points = np.asarray(coordinates, dtype=np.float64)
    work_upper, bytes_upper, queries_upper = periodic_preparation_bounds(
        cell, points.shape[0], shifts
    )
    with owned_periodic_preparation(
        work_upper, bytes_upper, queries_upper=queries_upper, budget=budget
    ) as owner:
        _require_periodic_lift(points, representatives, shifts, cell, budget=owner)
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
            "PeriodicMeshTopology.entity_global_ids must name only intermediate quotient degrees; got a vertex or top-cell degree"
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


class _PeriodicQuotientPreparation(StrictModule):
    """Host quotient structure and publication banks in deterministic degree order."""

    quotient: CellComplexTopology
    key_offsets: np.ndarray
    key_values: np.ndarray
    lifted_offsets: np.ndarray
    quotient_offsets: np.ndarray
    corner_shifts: np.ndarray
    orbit_indices: np.ndarray
    orbit_orientations: np.ndarray
    orbit_shifts: np.ndarray
    quotient_representatives: np.ndarray
    allocator_next_ids: tuple[int, ...] = eqx.field(static=True)


def _prepare_periodic_quotient(
    lifted: cell_mesh_owner.CellMesh,
    degrees: tuple[_Degree, ...],
    shifts: np.ndarray,
    supplied: dict[int, np.ndarray],
    entity_allocator_next_ids: Mapping[int, int] | None,
) -> _PeriodicQuotientPreparation:
    """Validate allocation history and assemble the oriented quotient publication."""
    dimension = lifted.topological_dimension
    allocation = (
        {} if entity_allocator_next_ids is None else dict(entity_allocator_next_ids)
    )
    if any(degree <= 0 or degree >= dimension for degree in allocation):
        raise ValueError(
            "PeriodicMeshTopology.entity_allocator_next_ids must name intermediate entity degrees; got an unsupported degree"
        )
    cursors = [-1] * (dimension + 1)
    for degree in range(1, dimension):
        cursor = allocation.get(degree)
        if cursor is None:
            cursors[degree] = degrees[degree].count if degree not in supplied else -1
            continue
        cursor = integer(cursor, "PeriodicMeshTopology.entity_allocator_next_ids")
        if cursor <= int(np.max(degrees[degree].identifiers)) or cursor > _INT64_MAX + 1:
            raise ValueError(
                "PeriodicMeshTopology.entity_allocator_next_ids must exceed assigned entity IDs within int64 allocation range; got an invalid cursor"
            )
        cursors[degree] = cursor
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
            "PeriodicMeshTopology.orbit_shifts must fit signed int32 with corner_shifts; got an out-of-range relative exponent"
        )
    return _PeriodicQuotientPreparation(
        quotient=quotient,
        key_offsets=key_offsets,
        key_values=key_values,
        lifted_offsets=lifted_offsets,
        quotient_offsets=quotient_offsets,
        corner_shifts=corner_shifts,
        orbit_indices=orbit_indices,
        orbit_orientations=orbit_orientations,
        orbit_shifts=orbit_shifts,
        quotient_representatives=quotient_representatives,
        allocator_next_ids=tuple(cursors),
    )


class PeriodicMeshTopology(StrictModule, NonTrainableState):
    """Quotient topology of a lifted periodic `CellMesh`.

    ``cell`` is the bound identification: a translational `PeriodicCell`
    lattice or a `PeriodicIsometryGroup`. ``vertex_representatives`` and
    ``vertex_shifts`` map lifted vertices to their quotient representative and
    integer group exponents. ``corner_shifts`` holds each active cell corner's
    exponents relative to the cell's first corner (the local lift). For every
    lifted entity, concatenated by degree through ``lifted_offsets``,
    ``orbit_indices`` names its quotient entity, ``orbit_orientations`` the
    ``±1`` sign of its orientation relative to the quotient entity, and
    ``orbit_shifts`` the group exponents carrying each orbit's representative
    lifted entity onto its copy. ``representatives`` names that lifted member
    per quotient entity, concatenated through ``quotient_offsets``. ``entity_key_*``
    pack the relative-shift keys of quotient entities. ``quotient`` is
    the validated oriented quotient cell complex.
    ``allocator_next_ids`` retains intermediate-entity allocation high-water
    cursors without changing geometric topology identity. A cursor of ``-1``
    means explicit entity IDs arrived without their allocation history; an edit
    cannot invent that history from the largest currently live ID.
    """

    __strict_contract__ = True

    cell: PeriodicCell | PeriodicIsometryGroup
    quotient: CellComplexTopology
    _actual_geometry: cell_geometry_owner.CellGeometrySpec | None
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

    @checked
    def __init__(
        self,
        lifted: cell_mesh_owner.CellMesh,
        cell: PeriodicCell | PeriodicIsometryGroup,
        vertex_representatives: ArrayLike,
        vertex_shifts: ArrayLike,
        /,
        *,
        entity_global_ids: Mapping[int, ArrayLike] | None = None,
        entity_allocator_next_ids: Mapping[int, int] | None = None,
        actual_geometry: cell_geometry_owner.CellGeometrySpec | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        # Periodic topology admits geometry after the lifted CellMesh has been constructed.
        from . import _cell_geometry_validity as cell_geometry_validity_owner

        with ExitStack() as stack:
            if lifted.periodic_topology is not None:
                raise ValueError(
                    "PeriodicMeshTopology.lifted must be a nonperiodic carrier; got a periodic descriptor"
                )
            if cell.ambient_dimension != lifted.ambient_dimension:
                raise ValueError(
                    "PeriodicMeshTopology.cell must share the lifted mesh ambient dimension; got incompatible dimensions"
                )
            dimension = lifted.topological_dimension
            if budget is None:
                work_upper, bytes_upper, queries_upper = periodic_preparation_bounds(
                    cell,
                    lifted.coordinates.shape[0],
                    np.asarray(vertex_shifts),
                    actual_geometry=actual_geometry,
                    stabilizers=actual_geometry is not None,
                )
                budget = stack.enter_context(
                    owned_periodic_preparation(
                        work_upper, bytes_upper, queries_upper=queries_upper
                    )
                )
            source_coordinates = (
                _exact_periodic_vertex_source(
                    np.asarray(lifted.coordinates, dtype=np.float64),
                    actual_geometry,
                    budget=budget,
                )
                if actual_geometry is not None
                else None
            )
            if actual_geometry is not None:
                _, routes, _ = actual_geometry.resolve(
                    lifted, exact_source_prepared=True, budget=budget
                )
                if any(
                    not np.array_equal(np.asarray(route), np.asarray(block.vertices))
                    for block, route in zip(lifted.blocks, routes, strict=True)
                ):
                    raise ValueError(
                        "PeriodicMeshTopology.actual_geometry must use lifted vertex routes; got another coordinate routing"
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
                    source_coordinates,
                    representatives,
                    shifts,
                    cell,
                    budget=budget,
                )
            else:
                _require_periodic_lift(
                    np.asarray(lifted.coordinates, dtype=np.float64),
                    representatives,
                    shifts,
                    cell,
                    budget=budget,
                )
            vertex_ids = np.asarray(lifted.vertex_global_ids, dtype=np.int64)
            representative_ids = vertex_ids[representatives]
            orders = identification_orders(cell)
            fixed = fixed_generators(
                cell,
                np.asarray(lifted.coordinates, dtype=np.float64),
                representatives,
                source_coordinates=source_coordinates,
                budget=budget,
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
            preparation = _prepare_periodic_quotient(
                lifted, degrees, shifts, supplied, entity_allocator_next_ids
            )
            quotient = preparation.quotient
            key_offsets = preparation.key_offsets
            key_values = preparation.key_values
            lifted_offsets = preparation.lifted_offsets
            quotient_offsets = preparation.quotient_offsets
            corner_shifts = preparation.corner_shifts
            orbit_indices = preparation.orbit_indices
            orbit_orientations = preparation.orbit_orientations
            orbit_shifts = preparation.orbit_shifts
            quotient_representatives = preparation.quotient_representatives
            cursors = preparation.allocator_next_ids
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
                    "cell": identification_id(cell),
                    "lifted_topology": lifted.topology_id,
                    "quotient": quotient.topology_id,
                    **(
                        {
                            "actual_geometry": cell_geometry_validity_owner.cell_geometry_id(
                                actual_geometry
                            )
                        }
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
            raise ValueError(
                f"PeriodicMeshTopology.degree must lie in [0, {self.topological_dimension}]; got {degree}"
            )
        return index

    @property
    def actual_geometry(self) -> cell_geometry_owner.CellGeometrySpec | None:
        """Return the independent vertex geometry bound by this descriptor."""

        return self._actual_geometry

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicMeshTopology],
        value: PeriodicMeshTopology,
        /,
        *,
        lifted: cell_mesh_owner.CellMesh,
    ) -> PeriodicMeshTopology:
        """Rebuild stored periodic fields against their bound lifted carrier."""
        if value.lifted_topology_id != lifted.topology_id:
            raise ValueError(
                "PeriodicMeshTopology.lifted_topology_id must bind the supplied carrier; got another lifted topology"
            )
        return value.rebuilt(lifted)

    @checked
    def rebuilt(
        self,
        lifted: cell_mesh_owner.CellMesh,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> PeriodicMeshTopology:
        """Revalidate this descriptor from its authored inputs on a restored lifted carrier.

        Quotient entity IDs and every resolved allocation cursor are carried
        unchanged; unresolved cursors stay unresolved. All orbit, orientation and
        quotient-complex checks of the constructor run again.
        """

        with ExitStack() as stack:
            dimension = self.topological_dimension
            geometry = self.actual_geometry
            if budget is None:
                work_upper, bytes_upper, queries_upper = periodic_preparation_bounds(
                    self.cell,
                    lifted.coordinates.shape[0],
                    np.asarray(self.vertex_shifts, dtype=np.int64),
                    actual_geometry=geometry,
                    stabilizers=geometry is not None,
                )
                # Source renewal and the two constructor passes share one owner.
                budget = stack.enter_context(
                    owned_periodic_preparation(
                        3 * work_upper,
                        3 * bytes_upper,
                        queries_upper=3 * queries_upper,
                    )
                )
            if geometry is not None:
                match geometry.exact_source:
                    case (
                        exact_power_geometry_owner.ExactPowerCellGeometrySource()
                        | exact_power_geometry_owner.ExactPowerCellGeometryRestrictionSource()
                        | exact_power_geometry_owner.ExactPowerCellGeometryLinearActionSource()
                    ) as source:
                        pass
                    case _:
                        raise TypeError(
                            "PeriodicMeshTopology.actual_geometry must retain an ExactPowerCellGeometrySource or its derived source; got another source kind"
                        )
                geometry = cell_geometry_owner.CellGeometrySpec.power(
                    lifted, source, budget=budget
                )
            probe = PeriodicMeshTopology(
                lifted,
                self.cell,
                self.vertex_representatives,
                self.vertex_shifts,
                actual_geometry=geometry,
                budget=budget,
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
                        "PeriodicMeshTopology.entity_key_values must be retained by reconstruction; got changed scientific entity keys"
                    )
                entity_ids[degree] = np.asarray(
                    [known[key] for key in keys], dtype=np.int64
                )
            return PeriodicMeshTopology(
                lifted,
                self.cell,
                self.vertex_representatives,
                self.vertex_shifts,
                actual_geometry=geometry,
                budget=budget,
                entity_global_ids=entity_ids,
                entity_allocator_next_ids={
                    degree: cursor
                    for degree, cursor in enumerate(self.allocator_next_ids)
                    if 0 < degree < dimension and cursor >= 0
                },
            )

    def orbits(self, degree: int, /) -> tuple[Array, Array, Array]:
        """Return quotient index, orientation sign, and shift per lifted entity."""

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

    @checked
    def entity_vertex_permutations(
        self, lifted: cell_mesh_owner.CellMesh, degree: int, /
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
            raise ValueError(
                "PeriodicMeshTopology.lifted_topology_id must bind the supplied mesh; got another topology"
            )
        if not 0 < index < self.topological_dimension:
            raise ValueError(
                "PeriodicMeshTopology.degree must name an edge or face degree for vertex permutations; got another degree"
            )
        orbit, _, anchors = (
            np.asarray(value, dtype=np.int64) for value in self.orbits(index)
        )
        members = np.asarray(self.orbit_representatives(index))
        vertices = np.asarray(self.vertex_representatives)
        shifts = np.asarray(self.vertex_shifts, dtype=np.int64)
        orders = identification_orders(self.cell)
        fixed = np.asarray(self.vertex_fixed_generators, dtype=np.bool_)
        loops = {}
        for rows, corners in lifted_loops(lifted, index):
            loops.update(zip(rows.tolist(), corners, strict=True))
        result = []
        for entity in range(orbit.size):
            corners = loops[entity]
            representative = int(members[orbit[entity]])
            base = loops[representative]
            delta = anchors[entity] - anchors[representative]
            difference = reduce_periodic_exponents(
                shifts[corners, None, :] - shifts[base][None, :, :] - delta,
                orders,
            )
            matches = (vertices[corners, None] == vertices[base][None, :]) & np.all(
                (difference == 0) | fixed[corners, None, :], axis=-1
            )
            if np.any(np.sum(matches, axis=1) != 1) or np.any(
                np.sum(matches, axis=0) != 1
            ):
                raise ValueError(
                    "PeriodicMeshTopology.entity_vertex_permutations must have one corner correspondence per orbit member; got a nonunique permutation"
                )
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
        """Return the relative-shift keys of quotient entities."""

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

    def require_lift(
        self,
        coordinates: ArrayLike,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> None:
        """Require coordinates to remain lattice images of their representatives."""

        points = np.asarray(coordinates, dtype=np.float64)
        if points.shape != (
            self.vertex_representatives.shape[0],
            self.cell.ambient_dimension,
        ):
            raise ValueError(
                "PeriodicMeshTopology.coordinates must match the lifted vertex bank; got another coordinate shape"
            )
        shifts = np.asarray(self.vertex_shifts, dtype=np.int64)
        work_upper, bytes_upper, queries_upper = periodic_preparation_bounds(
            self.cell, points.shape[0], shifts, actual_geometry=self.actual_geometry
        )
        with owned_periodic_preparation(
            work_upper, bytes_upper, queries_upper=queries_upper, budget=budget
        ) as owner:
            _require_periodic_lift(
                points,
                np.asarray(self.vertex_representatives, dtype=np.int64),
                shifts,
                self.cell,
                actual_geometry=self.actual_geometry,
                budget=owner,
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


__all__ = [
    "PeriodicMeasureReport",
    "PeriodicMeshTopology",
    "validate_periodic_vertex_orbits",
]
