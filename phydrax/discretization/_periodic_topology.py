#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from itertools import permutations
from typing import final

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..sparse import EdgeRelation
from ..typing import checked
from ._cell_geometry import CellGeometrySpec
from ._cell_mesh import CellBlock, CellMesh, PolyhedralBlock
from ._exact_algebraic_geometry import ExactPeriodicAlgebraicSource
from ._exact_coordinate_bank import ExactCoordinateBank
from ._exact_plc_geometry import ExactPlcCellGeometrySource
from ._exact_power_geometry import ExactPowerCellGeometrySource
from ._periodic_cell import PeriodicCell
from ._periodic_exact import (
    apply_periodic,
    identification_id,
    identification_orders,
    PeriodicIsometryGroup,
    PeriodicMeshIdentification,
    reduce_periodic_exponents,
)
from ._topology import CellComplexTopology, EntitySet, EntitySubset, OrientedIncidence


_INT32_MAX = np.iinfo(np.int32).max
# Exact vertex banks that can witness a quotient's deck action on lifted rows.
ExactPeriodicCoordinateSource = (
    ExactPowerCellGeometrySource | ExactPlcCellGeometrySource | ExactPeriodicAlgebraicSource
)


def _require_acyclic_periodic_witness(source: ExactPeriodicCoordinateSource, /) -> None:
    """Reject power-carrier witness cycles before traversing or archiving them."""
    pending: list[tuple[ExactPeriodicCoordinateSource, bool]] = [(source, False)]
    active: set[int] = set()
    complete: set[int] = set()
    while pending:
        current, leaving = pending.pop()
        identity = id(current)
        if leaving:
            active.remove(identity)
            complete.add(identity)
            continue
        if identity in active:
            raise ValueError(
                "Exact periodic coordinate witnesses contain a carrier cycle."
            )
        if identity in complete:
            continue
        active.add(identity)
        pending.append((current, True))
        if not isinstance(current, ExactPowerCellGeometrySource):
            continue
        periodic = current.carrier_mesh.periodic_topology
        if periodic is None:
            continue
        if not hasattr(periodic, "exact_coordinate_source"):
            raise ValueError("A periodic power carrier lacks its exact witness field.")
        child = periodic.exact_coordinate_source
        if child is not None:
            pending.append((child, False))
        geometry = periodic._actual_geometry
        if geometry is not None and isinstance(
            geometry.source_map,
            (ExactPowerCellGeometrySource, ExactPlcCellGeometrySource, ExactPeriodicAlgebraicSource),
        ):
            pending.append((geometry.source_map, False))


def _rebuild_periodic_witness(
    source: ExactPeriodicCoordinateSource, cell: PeriodicMeshIdentification, /
) -> ExactPeriodicCoordinateSource:
    """Reconstruct exact source leaves, not merely their retained identity."""
    _require_acyclic_periodic_witness(source)
    if isinstance(source, ExactPlcCellGeometrySource):
        return ExactPlcCellGeometrySource.from_stored_fields(source)
    if isinstance(source, ExactPeriodicAlgebraicSource):
        return ExactPeriodicAlgebraicSource.from_stored_fields(source)
    if not isinstance(cell, PeriodicCell):
        raise TypeError("Exact periodic power witnesses require PeriodicCell.")
    return ExactPowerCellGeometrySource.from_stored_fields(
        source, target_periodic_cell=cell
    )


def _integer_rows(value: ArrayLike, shape: tuple[int, ...], name: str, /) -> np.ndarray:
    rows = np.asarray(value)
    if rows.shape != shape or rows.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array with shape {shape}.")
    if rows.dtype.kind == "u" and rows.size and np.any(rows > _INT32_MAX):
        raise ValueError(f"{name} must fit int32.")
    rows = rows.astype(np.int64, copy=False)
    if rows.size and (np.any(rows < np.iinfo(np.int32).min) or np.any(rows > _INT32_MAX)):
        raise ValueError(f"{name} must fit int32.")
    return rows


def _entity_vertex_rows(mesh: CellMesh, /) -> tuple[tuple[np.ndarray, ...], ...]:
    """Expose each entity's ordered lifted vertex row from canonical connectivity."""
    from ._cell_complex import (
        IntervalConnectivity,
        PolygonalConnectivity,
        PolyhedralConnectivity,
        TetrahedralConnectivity,
    )
    from ._cell_mesh import SimplicialConnectivity
    from ._hexahedral import HexahedralConnectivity

    connectivity = mesh.connectivity
    if isinstance(connectivity, SimplicialConnectivity):
        return tuple(
            tuple(np.asarray(row, dtype=np.int64) for row in np.asarray(values))
            for values in connectivity.entities
        )
    vertices = tuple(
        np.asarray((index,), dtype=np.int64) for index in range(mesh.coordinates.shape[0])
    )
    if isinstance(connectivity, IntervalConnectivity):
        cells = tuple(np.asarray(connectivity.cell_vertices, dtype=np.int64))
        return vertices, cells
    if isinstance(connectivity, PolygonalConnectivity):
        edges = tuple(np.asarray(connectivity.edges, dtype=np.int64))
        values = np.asarray(connectivity.cell_vertices, dtype=np.int64)
        valid = np.asarray(connectivity.cell_vertex_valid, dtype=np.bool_)
        cells = tuple(values[row, :][valid[row, :]] for row in range(values.shape[0]))
        return vertices, edges, cells
    if isinstance(connectivity, TetrahedralConnectivity):
        edges = tuple(np.asarray(connectivity.edges, dtype=np.int64))
        faces = tuple(np.asarray(connectivity.faces, dtype=np.int64))
        cells = tuple(
            np.asarray(row, dtype=np.int64)
            for block in mesh.blocks
            for row in np.asarray(block.vertices)
        )
        return vertices, edges, faces, cells
    if isinstance(connectivity, HexahedralConnectivity):
        edges = tuple(np.asarray(connectivity.edges, dtype=np.int64))
        faces = tuple(np.asarray(connectivity.faces, dtype=np.int64))
        cells = tuple(
            np.asarray(row, dtype=np.int64)
            for block in mesh.blocks
            for row in np.asarray(block.vertices)
        )
        return vertices, edges, faces, cells
    if isinstance(connectivity, PolyhedralConnectivity):
        edge_rows = tuple(np.asarray(connectivity.edges, dtype=np.int64))
        face_offsets = np.asarray(connectivity.face_vertex_offsets, dtype=np.int64)
        face_vertices = np.asarray(connectivity.face_vertex_values, dtype=np.int64)
        faces = tuple(
            face_vertices[start:stop]
            for start, stop in zip(face_offsets[:-1], face_offsets[1:], strict=True)
        )
        cell_offsets = np.asarray(connectivity.cell_vertex_offsets, dtype=np.int64)
        cell_vertices = np.asarray(connectivity.cell_vertex_values, dtype=np.int64)
        cells = tuple(
            cell_vertices[start:stop]
            for start, stop in zip(cell_offsets[:-1], cell_offsets[1:], strict=True)
        )
        return vertices, edge_rows, faces, cells
    raise TypeError(
        f"Periodic quotient topology does not support connectivity {type(connectivity).__name__}."
    )


def _orientation_orders(
    arity: int, *, simplex: bool
) -> tuple[tuple[tuple[int, ...], int], ...]:
    if arity == 1:
        return (((0,), 1),)
    # An edge is a 1-simplex: its 2-cycle rotation is its reversal, so cyclic
    # loop orientation would wrongly give the reversed edge a positive sign.
    if simplex or arity == 2:
        records = []
        for order in permutations(range(arity)):
            inversions = sum(
                order[first] > order[second]
                for first in range(arity)
                for second in range(first + 1, arity)
            )
            records.append((tuple(order), -1 if inversions % 2 else 1))
        return tuple(records)
    forward = tuple(range(arity))
    reverse = tuple(reversed(forward))
    result: dict[tuple[int, ...], int] = {}
    for direction, sign in ((forward, 1), (reverse, -1)):
        for shift in range(arity):
            order = direction[shift:] + direction[:shift]
            result.setdefault(order, sign)
    return tuple((order, sign) for order, sign in result.items())


def _interleaved_keys(
    identifiers: np.ndarray,
    exponents: np.ndarray,
    fixed: np.ndarray,
    order: tuple[int, ...],
    orders: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Interleaved ``(id, relative winding)`` keys of all rows in one corner order."""
    ranking = np.asarray(order, dtype=np.int64)
    ranked_ids = identifiers[:, ranking]
    ranked_fixed = fixed[:, ranking]
    ranked_exp = np.where(ranked_fixed, 0, exponents[:, ranking])
    # A deck shift moves only vertices outside the generator's fixed set, so
    # the anchor must be a moving vertex and fixed windings stay zero.
    movable = ~np.any(ranked_fixed, axis=2)
    first = np.argmax(movable, axis=1)
    rows = np.arange(identifiers.shape[0])
    anchors = np.where(
        movable[rows, first][:, None],
        ranked_exp[rows, first],
        np.zeros((), dtype=ranked_exp.dtype),
    )
    relative = reduce_periodic_exponents(ranked_exp - anchors[:, None, :], orders)
    relative[ranked_fixed] = 0
    keys = np.concatenate((ranked_ids[:, :, None], relative), axis=2)
    return keys.reshape((identifiers.shape[0], -1)).astype(np.int64), anchors


def _canonical_forms(
    identifiers: np.ndarray,
    shifts: np.ndarray,
    fixed: np.ndarray,
    orders: np.ndarray,
    /,
    *,
    simplex: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lexicographically least key over the orientation orders, row by row.

    The first least variant supplies the sign and anchor; every variant with
    the least key must carry the same sign.
    """
    count, arity = identifiers.shape
    variants = _orientation_orders(arity, simplex=simplex)
    rows = np.arange(count)
    keys = np.empty((count, arity * (orders.size + 1)), dtype=np.int64)
    signs = np.empty((count,), dtype=np.int32)
    anchors = np.empty((count, orders.size), dtype=np.int64)
    candidates = []
    for index, (permutation, sign) in enumerate(variants):
        key, anchor = _interleaved_keys(identifiers, shifts, fixed, permutation, orders)
        candidates.append((key, sign))
        if index == 0:
            keys[:] = key
            signs[:] = sign
            anchors[:] = anchor
            continue
        differs = key != keys
        column = np.argmax(differs, axis=1)
        smaller = differs[rows, column] & (key[rows, column] < keys[rows, column])
        keys[smaller] = key[smaller]
        signs[smaller] = sign
        anchors[smaller] = anchor[smaller]
    for key, sign in candidates:
        if np.any(np.all(key == keys, axis=1) & (signs != sign)):
            raise ValueError(
                "Periodic entity has an orientation-reversing stabilizer in the quotient."
            )
    return keys, signs, anchors


def _group_keys(keys: np.ndarray, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if keys.ndim != 2 or keys.shape[0] == 0:
        raise ValueError(
            "Periodic quotient requires non-empty entity rows at every degree."
        )
    order = np.lexsort(keys.T[::-1])
    ordered = keys[order]
    starts = np.ones((order.size,), dtype=np.bool_)
    starts[1:] = np.any(ordered[1:] != ordered[:-1], axis=1)
    start_rows = np.flatnonzero(starts)
    orbit = np.cumsum(starts, dtype=np.int64) - 1
    inverse = np.empty_like(orbit)
    inverse[order] = orbit
    return ordered[start_rows], inverse, order[start_rows]


def _group_ragged_keys(
    keys: tuple[tuple[int, ...], ...], /
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray, np.ndarray]:
    if not keys:
        raise ValueError(
            "Periodic quotient requires non-empty entity rows at every degree."
        )
    order = sorted(range(len(keys)), key=keys.__getitem__)
    unique: list[tuple[int, ...]] = []
    first: list[int] = []
    orbit = np.empty((len(keys),), dtype=np.int64)
    for row in order:
        key = keys[row]
        if not unique or key != unique[-1]:
            unique.append(key)
            first.append(row)
        orbit[row] = len(unique) - 1
    return tuple(unique), orbit, np.asarray(first, dtype=np.int64)


def _canonical_unordered_form(
    identifiers: np.ndarray,
    exponents: np.ndarray,
    fixed: np.ndarray,
    orders: np.ndarray,
    /,
) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
    """Canonicalize an unordered cell vertex set under one common deck shift."""
    ranked_fixed = np.asarray(fixed, dtype=np.bool_)
    ranked_exponents = np.asarray(exponents, dtype=np.int64).copy()
    ranked_exponents[ranked_fixed] = 0
    movable_rows = np.flatnonzero(~np.any(ranked_fixed, axis=1))
    anchor_rows = movable_rows if movable_rows.size else np.arange(identifiers.size)
    candidates = []
    for anchor_row in anchor_rows.tolist():
        anchor = ranked_exponents[anchor_row].copy()
        relative = reduce_periodic_exponents(ranked_exponents - anchor, orders)
        relative[ranked_fixed] = 0
        entries = tuple(
            sorted(
                (
                    int(identifiers[row]),
                    tuple(int(value) for value in relative[row]),
                    row,
                )
                for row in range(identifiers.size)
            )
        )
        key = tuple(
            value
            for identifier, winding, _ in entries
            for value in (identifier, *winding)
        )
        permutation = np.asarray([row for _, _, row in entries], dtype=np.int32)
        candidates.append(
            (key, tuple(int(value) for value in anchor), anchor, permutation)
        )
    key, _, anchor, permutation = min(candidates, key=lambda value: (value[0], value[1]))
    return key, anchor, permutation


def _supplied_ids(
    mapping: Mapping[int, ArrayLike] | None,
    degree: int,
    count: int,
    default: np.ndarray,
    /,
) -> np.ndarray:
    if mapping is None or degree not in mapping:
        return default.astype(np.int64, copy=False)
    values = np.asarray(mapping[degree])
    if values.shape != (count,) or values.dtype.kind not in "iu":
        raise ValueError(
            f"entity_global_ids[{degree}] must be an integer vector of length {count}."
        )
    ids = values.astype(np.int64, copy=False)
    if np.any(ids < 0) or np.unique(ids).size != count:
        raise ValueError(f"entity_global_ids[{degree}] must be unique and nonnegative.")
    return ids


def _continued_ids(
    ancestry: ArrayLike,
    degree: int,
    orbit: np.ndarray,
    count: int,
    next_id: int,
    /,
) -> tuple[np.ndarray, int]:
    """Keep each continued quotient ID once; draw every other ID from the allocator.

    ``ancestry`` names, per lifted row, the source quotient ID the entity
    continues (``-1`` for a new entity). A quotient entity whose continued rows
    name exactly one source ID that no other quotient entity claims keeps it;
    all others are new and take consecutive allocator IDs in canonical
    quotient-row order, so a retired ID never returns.
    """
    values = np.asarray(ancestry)
    if values.shape != orbit.shape or values.dtype.kind not in "iu":
        raise ValueError(
            f"quotient_id_ancestry[{degree}] must be an integer vector per lifted entity."
        )
    values = values.astype(np.int64, copy=False)
    if np.any(values < -1):
        raise ValueError(
            f"quotient_id_ancestry[{degree}] holds source quotient IDs or -1."
        )
    if next_id <= int(np.max(values, initial=-1)):
        raise ValueError(
            f"entity_allocator_next_ids[{degree}] must exceed every continued quotient ID."
        )
    # (quotient row, continued ID) pairs; a row with exactly one continued ID
    # claims it, and an ID claimed by exactly one row is kept.
    continued = values >= 0
    pairs = np.unique(
        np.stack((orbit[continued].astype(np.int64), values[continued]), axis=1), axis=0
    )
    rows, starts, sizes = np.unique(pairs[:, 0], return_index=True, return_counts=True)
    claimants = rows[sizes == 1]
    claimed = pairs[starts[sizes == 1], 1]
    identifiers, claim_counts = np.unique(claimed, return_counts=True)
    sole = np.isin(claimed, identifiers[claim_counts == 1])
    ids = np.full((count,), -1, dtype=np.int64)
    ids[claimants[sole]] = claimed[sole]
    fresh = np.flatnonzero(ids < 0)
    ids[fresh] = np.arange(next_id, next_id + fresh.size, dtype=np.int64)
    return ids, next_id + int(fresh.size)


def _quotient_incidence(
    lifted: CellMesh,
    entity_sets: tuple[EntitySet, ...],
    orbit_by_degree: tuple[np.ndarray, ...],
    orientation_by_degree: tuple[np.ndarray, ...],
    representative_by_degree: tuple[np.ndarray, ...],
    /,
) -> tuple[OrientedIncidence, ...]:
    incidences = []
    for degree, source in enumerate(lifted.topology.incidences, start=1):
        relation = source.relation
        raw_lower = np.asarray(relation.source_indices, dtype=np.int64)
        raw_upper = np.asarray(relation.target_indices, dtype=np.int64)
        valid = np.asarray(relation.valid, dtype=np.bool_)
        signs = np.asarray(source.signs, dtype=np.float64)
        upper_representatives = np.asarray(
            representative_by_degree[degree], dtype=np.int64
        )
        lower_orbits = orbit_by_degree[degree - 1]
        lower_orientation = orientation_by_degree[degree - 1]
        upper_orientation = orientation_by_degree[degree]
        # Every lifted copy of an identified entity must induce the same
        # oriented quotient boundary, not just the chosen representative.
        # Sum repeated lower orbits because a quotient boundary may cancel.
        upper_orbits = orbit_by_degree[degree]
        multiplicity = np.bincount(upper_orbits, minlength=upper_representatives.size)
        repeated = multiplicity[upper_orbits] > 1
        if np.any(repeated):
            boundaries: dict[int, dict[int, int]] = {}
            valid_routes = np.flatnonzero(valid)
            for route in valid_routes[repeated[raw_upper[valid_routes]]]:
                upper = int(raw_upper[route])
                lower = int(raw_lower[route])
                boundary = boundaries.setdefault(upper, {})
                target = int(lower_orbits[lower])
                coefficient = int(
                    signs[route] * lower_orientation[lower] * upper_orientation[upper]
                )
                boundary[target] = boundary.get(target, 0) + coefficient
            canonical: dict[int, dict[int, int]] = {}
            for upper in np.flatnonzero(repeated):
                boundary = {
                    lower: coefficient
                    for lower, coefficient in boundaries.get(int(upper), {}).items()
                    if coefficient
                }
                orbit = int(upper_orbits[upper])
                if orbit in canonical and canonical[orbit] != boundary:
                    raise ValueError(
                        "Identified periodic entities have inconsistent oriented boundaries."
                    )
                canonical[orbit] = boundary
        # Each lifted upper entity represents at most one quotient entity; routes
        # follow quotient order, then ascending route row.
        routes = np.flatnonzero(valid)
        quotient_of = np.full(
            (
                max(
                    int(np.max(raw_upper[routes], initial=-1)),
                    int(np.max(upper_representatives, initial=-1)),
                )
                + 1,
            ),
            -1,
            dtype=np.int64,
        )
        quotient_of[upper_representatives] = np.arange(
            upper_representatives.size, dtype=np.int64
        )
        routes = routes[quotient_of[raw_upper[routes]] >= 0]
        routes = routes[np.argsort(quotient_of[raw_upper[routes]], kind="stable")]
        lower = raw_lower[routes]
        lifted_upper = raw_upper[routes]
        source_rows = lower_orbits[lower]
        target_rows = quotient_of[lifted_upper]
        coefficients = (
            signs[routes]
            * lower_orientation[lower].astype(np.float64)
            * upper_orientation[lifted_upper].astype(np.float64)
        )
        edge_relation = EdgeRelation(
            np.asarray(source_rows, dtype=np.int32),
            np.asarray(target_rows, dtype=np.int32),
            source_size=entity_sets[degree - 1].count,
            target_size=entity_sets[degree].count,
        )
        incidences.append(
            OrientedIncidence(
                degree,
                entity_sets[degree - 1],
                entity_sets[degree],
                edge_relation,
                np.asarray(coefficients, dtype=np.float64),
            )
        )
    return tuple(incidences)


def _boundary_masks(
    incidences: tuple[OrientedIncidence, ...], dimension: int, /
) -> tuple[np.ndarray, ...]:
    masks = [
        np.zeros((incidence.relation.source_size,), dtype=np.bool_)
        for incidence in incidences
    ]
    top = incidences[-1]
    relation = top.relation
    valid = np.asarray(relation.valid, dtype=np.bool_)
    counts = np.bincount(
        np.asarray(relation.source_indices, dtype=np.int64)[valid],
        minlength=relation.source_size,
    )
    if np.any(counts > 2):
        raise ValueError("Periodic quotient has a nonmanifold top-cell facet orbit.")
    masks[-1] = counts == 1
    for degree in range(dimension - 2, -1, -1):
        incidence = incidences[degree]
        relation = incidence.relation
        valid = np.asarray(relation.valid, dtype=np.bool_)
        lower = np.asarray(relation.source_indices, dtype=np.int64)[valid]
        upper = np.asarray(relation.target_indices, dtype=np.int64)[valid]
        masks[degree] = np.zeros((relation.source_size,), dtype=np.bool_)
        masks[degree][lower[np.asarray(masks[degree + 1], dtype=np.bool_)[upper]]] = True
    top_cells = np.zeros((incidences[-1].relation.target_size,), dtype=np.bool_)
    return (*masks, top_cells)


def _corner_shifts(
    mesh: CellMesh, shifts: np.ndarray, orders: np.ndarray, /
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for block in mesh.blocks:
        vertices = np.asarray(block.vertices, dtype=np.int64)
        valid = np.asarray(block.vertex_valid, dtype=np.bool_)
        # Each cell's valid corners, row-major, relative to its first valid corner.
        first = vertices[np.arange(vertices.shape[0]), np.argmax(valid, axis=1)]
        relative = shifts[vertices] - shifts[first][:, None, :]
        rows.append(reduce_periodic_exponents(relative[valid], orders))
    return (
        np.concatenate(rows, axis=0)
        if rows
        else np.zeros((0, orders.size), dtype=np.int64)
    )


def fixed_generators(
    cell: PeriodicMeshIdentification,
    coordinates: ArrayLike,
    representatives: ArrayLike,
    /,
) -> np.ndarray:
    """Return generator stabilizers for every explicitly authored vertex orbit."""
    points = np.asarray(coordinates, dtype=np.float64)
    roots = np.asarray(representatives, dtype=np.int64)
    if (
        points.ndim != 2
        or roots.shape != (points.shape[0],)
        or np.any(roots < 0)
        or np.any(roots >= points.shape[0])
    ):
        raise ValueError(
            "coordinates and representative rows must cover every lifted vertex."
        )
    if isinstance(cell, PeriodicCell):
        return np.zeros((points.shape[0], cell.rank), dtype=np.bool_)
    if isinstance(cell, PeriodicIsometryGroup):
        root_rows = np.unique(roots)
        root_fixed = cell.fixed(points[root_rows])
        lookup = np.full((points.shape[0],), -1, dtype=np.int64)
        lookup[root_rows] = np.arange(root_rows.size, dtype=np.int64)
        return root_fixed[lookup[roots]]
    raise TypeError("cell must be PeriodicCell or PeriodicIsometryGroup.")


def _validate_vertex_orbits(
    coordinates: ArrayLike,
    cell: PeriodicMeshIdentification,
    representatives: ArrayLike,
    shifts: ArrayLike,
    /,
    *,
    exact_coordinate_source: ExactPeriodicCoordinateSource | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(coordinates, dtype=np.float64)
    if (
        points.ndim != 2
        or points.shape[1] != cell.ambient_dimension
        or points.shape[0] == 0
        or np.any(~np.isfinite(points))
    ):
        raise ValueError(
            "Periodic coordinates must be a nonempty finite point bank in the ambient dimension."
        )
    reps = np.asarray(representatives)
    exponents = np.asarray(shifts)
    rank = len(identification_orders(cell))
    reps = _integer_rows(reps, (points.shape[0],), "vertex_representatives")
    exponents = _integer_rows(exponents, (points.shape[0], rank), "vertex_shifts")
    if np.any(reps < 0) or np.any(reps >= points.shape[0]):
        raise ValueError("vertex_representatives contain an out-of-range lifted row.")
    if np.any(reps[reps] != reps) or np.any(exponents[reps] != 0):
        raise ValueError(
            "Every periodic representative must be a zero-winding root of its own orbit."
        )
    if exact_coordinate_source is None:
        expected = apply_periodic(cell, points[reps], exponents)
        if not np.array_equal(points, expected):
            raise ValueError(
                "vertex_shifts do not reproduce the lifted coordinates from their representatives."
            )
    else:
        if isinstance(exact_coordinate_source, ExactPowerCellGeometrySource):
            if not isinstance(cell, PeriodicCell):
                raise TypeError("Exact periodic power witnesses require PeriodicCell.")
            exact_coordinate_source.require_periodic_lift(points, cell, reps, exponents)
        else:
            exact_coordinate_source.require_periodic_lift(points, cell, reps, exponents)
    return reps.astype(np.int32), exponents.astype(np.int32)


def validate_periodic_vertex_orbits(
    coordinates: ArrayLike,
    cell: PeriodicMeshIdentification,
    representatives: ArrayLike,
    shifts: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    return _validate_vertex_orbits(coordinates, cell, representatives, shifts)


def oriented_forms(
    identifiers: ArrayLike,
    shifts: ArrayLike,
    fixed: ArrayLike,
    orders: ArrayLike,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical dihedral keys, orientation signs, and anchor windings for loops."""
    ids = np.asarray(identifiers)
    exponents = np.asarray(shifts)
    stabilizers = np.asarray(fixed, dtype=np.bool_)
    periods = np.asarray(orders, dtype=np.int64)
    if ids.ndim != 2 or ids.shape[1] == 0 or ids.dtype.kind not in "iu":
        raise ValueError(
            "identifiers must have shape (entity_count, corner_count) and integer dtype."
        )
    if (
        exponents.shape != (ids.shape[0], ids.shape[1], periods.size)
        or exponents.dtype.kind not in "iu"
    ):
        raise ValueError(
            "shifts must be integer per-corner exponent rows aligned with orders."
        )
    if stabilizers.shape != exponents.shape:
        raise ValueError("fixed must align with shifts.")
    keys, signs, anchors = _canonical_forms(
        ids.astype(np.int64),
        exponents.astype(np.int64),
        stabilizers,
        periods,
        simplex=False,
    )
    return keys, signs, anchors


def lifted_loops(
    mesh: CellMesh, degree: int, /
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Lifted entity loops grouped by their corner count, with authored winding rows."""
    topology = mesh.periodic_topology
    if topology is None:
        raise ValueError("lifted_loops requires a CellMesh with periodic topology.")
    rows_by_degree = _entity_vertex_rows(mesh)
    target = int(degree)
    if target < 0 or target >= len(rows_by_degree):
        raise ValueError("degree is outside the lifted mesh topology.")
    identifiers = np.asarray(mesh.vertex_global_ids, dtype=np.int64)[
        np.asarray(topology.vertex_representatives, dtype=np.int64)
    ]
    shifts = np.asarray(topology.vertex_shifts, dtype=np.int64)
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for row in rows_by_degree[target]:
        values = np.asarray(row, dtype=np.int64)
        grouped.setdefault(values.size, []).append((identifiers[values], shifts[values]))
    return tuple(
        (
            np.stack([entry[0] for entry in grouped[arity]]),
            np.stack([entry[1] for entry in grouped[arity]]),
        )
        for arity in sorted(grouped)
    )


@final
class PeriodicMeshTopology(StrictModule, NonTrainableState):
    """Canonical quotient of a lifted mesh with explicit representative/winding tables."""

    cell: PeriodicMeshIdentification
    quotient: CellComplexTopology
    _actual_geometry: CellGeometrySpec | None
    exact_coordinate_source: ExactPeriodicCoordinateSource | None
    coordinate_source_map_id: str | None = eqx.field(static=True)
    coordinate_shadow_fingerprint: str | None = eqx.field(static=True)
    vertex_fixed_generators: Array
    vertex_representatives: Array
    vertex_shifts: Array
    corner_shifts: Array
    orbit_indices: Array
    orbit_orientations: Array
    orbit_shifts: Array
    representatives: Array
    lifted_offsets: Array
    quotient_offsets: Array
    entity_key_offsets: Array
    entity_key_values: Array
    topological_dimension: int = eqx.field(static=True)
    lifted_complex_id: str = eqx.field(static=True)
    lifted_entity_ids: tuple[Array, ...]
    lifted_block_specs: tuple[tuple[str, str, int, str], ...] = eqx.field(static=True)
    lifted_topology_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    allocator_next_ids: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        lifted: CellMesh,
        cell: PeriodicMeshIdentification,
        vertex_representatives: ArrayLike,
        vertex_shifts: ArrayLike,
        /,
        entity_global_ids: Mapping[int, ArrayLike] | None = None,
        entity_allocator_next_ids: Mapping[int, int] | None = None,
        quotient_id_ancestry: Mapping[int, ArrayLike] | None = None,
        actual_geometry: CellGeometrySpec | None = None,
        exact_coordinate_source: ExactPeriodicCoordinateSource | None = None,
    ) -> None:
        if lifted.ambient_dimension != cell.ambient_dimension:
            raise ValueError(
                "Periodic identification must match the lifted ambient space."
            )
        if lifted.periodic_topology is not None:
            raise ValueError("lifted must be a nonperiodic fundamental-cover carrier.")
        if exact_coordinate_source is not None and not isinstance(
            exact_coordinate_source,
            (ExactPowerCellGeometrySource, ExactPlcCellGeometrySource, ExactPeriodicAlgebraicSource),
        ):
            raise TypeError(
                "exact_coordinate_source must be an exact periodic coordinate bank."
            )
        if actual_geometry is not None:
            source_map = actual_geometry.source_map
            if source_map is None:
                _, routes, geometry_coordinates = actual_geometry.resolve(lifted)
            else:
                if not isinstance(
                    source_map,
                    (ExactPowerCellGeometrySource, ExactPlcCellGeometrySource, ExactPeriodicAlgebraicSource),
                ):
                    raise ValueError(
                        "Periodic mapped geometry requires its exact coordinate witness."
                    )
                if (
                    exact_coordinate_source is not None
                    and source_map.source_map_id != exact_coordinate_source.source_map_id
                ):
                    raise ValueError(
                        "Periodic geometry and exact coordinate witnesses disagree."
                    )
                exact_coordinate_source = source_map
                # A source-bound geometry may name the final periodic mesh, not
                # this lifted carrier, so resolve() cannot be called here.
                # Still align its canonical block-name tables to lifted order.
                route_by_name = dict(
                    zip(
                        actual_geometry.block_names,
                        actual_geometry.geometry_dofs,
                        strict=True,
                    )
                )
                element_by_name = dict(
                    zip(
                        actual_geometry.block_names, actual_geometry.elements, strict=True
                    )
                )
                if set(route_by_name) != {block.name for block in lifted.blocks} or any(
                    element_by_name[block.name].cell_kind != block.cell_kind
                    for block in lifted.blocks
                ):
                    raise ValueError(
                        "Periodic geometry assignments differ from lifted blocks."
                    )
                routes = tuple(route_by_name[block.name] for block in lifted.blocks)
                geometry_coordinates = actual_geometry.coordinates
            if (
                actual_geometry.restriction_source is not None
                or not np.array_equal(
                    np.asarray(geometry_coordinates), np.asarray(lifted.coordinates)
                )
                or any(
                    not np.array_equal(np.asarray(route), np.asarray(block.vertices))
                    for route, block in zip(routes, lifted.blocks, strict=True)
                )
            ):
                raise ValueError(
                    "actual_geometry must resolve to the lifted vertex-indexed carrier."
                )
        if exact_coordinate_source is not None:
            exact_coordinate_source = _rebuild_periodic_witness(
                exact_coordinate_source, cell
            )
        if isinstance(exact_coordinate_source, (ExactPlcCellGeometrySource, ExactPeriodicAlgebraicSource)) and (
            not np.array_equal(
                np.asarray(exact_coordinate_source.vertex_global_ids, dtype=np.int64),
                np.asarray(lifted.vertex_global_ids, dtype=np.int64),
            )
        ):
            raise ValueError(
                "Exact coordinate rows must follow the lifted vertex order."
            )
        representatives, shifts = _validate_vertex_orbits(
            lifted.coordinates,
            cell,
            vertex_representatives,
            vertex_shifts,
            exact_coordinate_source=exact_coordinate_source,
        )
        dimension = lifted.topological_dimension
        next_ids = [
            -1
            if entity_allocator_next_ids is None
            or degree not in entity_allocator_next_ids
            else int(entity_allocator_next_ids[degree])
            for degree in range(dimension + 1)
        ]
        if quotient_id_ancestry is not None and any(
            degree not in range(dimension + 1)
            or next_ids[degree] < 0
            or (entity_global_ids is not None and degree in entity_global_ids)
            for degree in quotient_id_ancestry
        ):
            raise ValueError(
                "quotient_id_ancestry needs an allocator and no supplied IDs per degree."
            )
        row_groups = _entity_vertex_rows(lifted)
        if len(row_groups) != dimension + 1:
            raise ValueError("Lifted connectivity must provide every entity degree.")
        if any(
            len(rows) != lifted.topology.entity_sets[degree].count
            for degree, rows in enumerate(row_groups)
        ):
            raise ValueError(
                "Lifted connectivity entity rows must match canonical entity sets."
            )
        orders = identification_orders(cell)
        fixed_vertices = fixed_generators(cell, lifted.coordinates, representatives)
        root_ids = np.asarray(lifted.vertex_global_ids, dtype=np.int64)[representatives]
        orbit_by_degree: list[np.ndarray] = []
        orientation_by_degree: list[np.ndarray] = []
        anchor_by_degree: list[np.ndarray] = []
        representative_by_degree: list[np.ndarray] = []
        quotient_ids_by_degree: list[np.ndarray] = []
        keys_by_degree: list[tuple[tuple[int, ...], ...]] = []
        offsets_lifted = [0]
        offsets_quotient = [0]
        cell_kind_by_id = {
            int(identifier): block.cell_kind
            for block in lifted.blocks
            for identifier in np.asarray(block.global_ids, dtype=np.int64)
        }
        top_cell_ids = np.asarray(
            lifted.topology.entity_sets[dimension].entity_ids, dtype=np.int64
        )
        top_cell_kinds = tuple(
            cell_kind_by_id[int(identifier)] for identifier in top_cell_ids
        )
        for degree, rows in enumerate(row_groups):
            if not rows or any(len(row) == 0 for row in rows):
                raise ValueError("Periodic quotient cannot contain empty entities.")
            row_keys: list[tuple[int, ...] | None] = [None] * len(rows)
            signs = np.empty((len(rows),), dtype=np.int32)
            anchors = np.empty((len(rows), orders.size), dtype=np.int64)
            groups: dict[int, list[int]] = {}
            for row, vertices in enumerate(rows):
                groups.setdefault(len(vertices), []).append(row)
            for arity, group_rows in groups.items():
                unordered_rows = (
                    [
                        row
                        for row in group_rows
                        if degree == dimension and top_cell_kinds[row] == "polyhedron"
                    ]
                    if dimension == 3
                    else []
                )
                for row in unordered_rows:
                    vertices = np.asarray(rows[row], dtype=np.int64)
                    key, anchor, _ = _canonical_unordered_form(
                        root_ids[vertices],
                        shifts[vertices],
                        fixed_vertices[vertices],
                        orders,
                    )
                    row_keys[row] = key
                    signs[row] = 1
                    anchors[row] = anchor
                unordered_set = set(unordered_rows)
                ordered_rows = [row for row in group_rows if row not in unordered_set]
                if not ordered_rows:
                    continue
                vertices = np.asarray([rows[row] for row in ordered_rows], dtype=np.int64)
                ids = root_ids[vertices]
                exponents = shifts[vertices]
                fixed = fixed_vertices[vertices]
                simplex = (
                    arity == degree + 1
                    and degree == dimension
                    and all(
                        top_cell_kinds[row].startswith(
                            ("simplex:", "triangle", "tetrahedron")
                        )
                        for row in ordered_rows
                    )
                )
                keys, group_signs, group_anchors = _canonical_forms(
                    ids, exponents, fixed, orders, simplex=simplex
                )
                for local_row, row in enumerate(ordered_rows):
                    row_keys[row] = tuple(int(value) for value in keys[local_row])
                    signs[row] = group_signs[local_row]
                    anchors[row] = group_anchors[local_row]
            if any(key is None for key in row_keys):
                raise ValueError(
                    "Periodic quotient failed to canonicalize an entity row."
                )
            canonical_keys = tuple(key for key in row_keys if key is not None)
            if len({len(key) for key in canonical_keys}) == 1:
                key_table = np.asarray(canonical_keys, dtype=np.int64)
                unique_table, orbit, _ = _group_keys(key_table)
                unique_keys = tuple(
                    tuple(int(value) for value in row) for row in unique_table
                )
            else:
                unique_keys, orbit, _ = _group_ragged_keys(canonical_keys)
            lifted_ids = np.asarray(
                lifted.topology.entity_sets[degree].entity_ids, dtype=np.int64
            )
            members = np.empty((len(unique_keys),), dtype=np.int64)
            for quotient_row in range(len(unique_keys)):
                copies = np.flatnonzero(orbit == quotient_row)
                members[quotient_row] = copies[np.argmin(lifted_ids[copies])]
            defaults = lifted_ids[members]
            if quotient_id_ancestry is not None and degree in quotient_id_ancestry:
                quotient_ids, next_ids[degree] = _continued_ids(
                    quotient_id_ancestry[degree],
                    degree,
                    orbit,
                    len(unique_keys),
                    next_ids[degree],
                )
            else:
                quotient_ids = _supplied_ids(
                    entity_global_ids, degree, len(unique_keys), defaults
                )
            quotient_ids_by_degree.append(quotient_ids)
            orbit_by_degree.append(orbit.astype(np.int32))
            orientation_by_degree.append(signs)
            anchor_by_degree.append(anchors)
            representative_by_degree.append(members.astype(np.int32))
            keys_by_degree.append(unique_keys)
            offsets_lifted.append(offsets_lifted[-1] + len(rows))
            offsets_quotient.append(offsets_quotient[-1] + len(unique_keys))
        entity_sets = []
        for degree, ids in enumerate(quotient_ids_by_degree):
            entity_sets.append(EntitySet(f"periodic-{degree}-entities", degree, ids))
        quotient_entities = tuple(entity_sets)
        incidences = _quotient_incidence(
            lifted,
            quotient_entities,
            tuple(orbit_by_degree),
            tuple(orientation_by_degree),
            tuple(representative_by_degree),
        )
        masks = _boundary_masks(incidences, dimension)
        entity_sets = tuple(
            EntitySet(
                entity.name,
                entity.intrinsic_dimension,
                np.asarray(entity.entity_ids),
                subsets=(EntitySubset("boundary", masks[degree]),),
            )
            for degree, entity in enumerate(quotient_entities)
        )
        # Rebind incidence endpoint identities to the final sets containing boundary subsets.
        incidences = _quotient_incidence(
            lifted,
            entity_sets,
            tuple(orbit_by_degree),
            tuple(orientation_by_degree),
            tuple(representative_by_degree),
        )
        quotient = CellComplexTopology(entity_sets, incidences)
        all_orbits = np.concatenate(
            [
                orbit_by_degree[degree] + offsets_quotient[degree]
                for degree in range(dimension + 1)
            ]
        ).astype(np.int32)
        all_orientations = np.concatenate(orientation_by_degree).astype(np.int32)
        relative_anchor_by_degree = [
            reduce_periodic_exponents(anchors - anchors[members[orbits]], orders)
            for anchors, members, orbits in zip(
                anchor_by_degree,
                representative_by_degree,
                orbit_by_degree,
                strict=True,
            )
        ]
        all_shifts = np.concatenate(relative_anchor_by_degree, axis=0)
        all_representatives = np.concatenate(representative_by_degree).astype(np.int32)
        key_rows = [
            np.asarray(row, dtype=np.int64)
            for degree_keys in keys_by_degree
            for row in degree_keys
        ]
        key_offsets = np.zeros((len(key_rows) + 1,), dtype=np.int64)
        np.cumsum([row.size for row in key_rows], out=key_offsets[1:])
        key_values = (
            np.concatenate(key_rows).astype(np.int64, copy=False)
            if key_rows
            else np.zeros((0,), dtype=np.int64)
        )
        corner_shifts = _corner_shifts(lifted, shifts, orders)

        allocator = tuple(next_ids)
        for degree, next_id in enumerate(allocator):
            if next_id != -1 and next_id <= int(
                np.max(quotient_ids_by_degree[degree], initial=-1)
            ):
                raise ValueError(
                    f"entity_allocator_next_ids[{degree}] must exceed every retained quotient ID."
                )
        self.cell = cell
        self.quotient = quotient
        self._actual_geometry = actual_geometry
        self.exact_coordinate_source = exact_coordinate_source
        self.coordinate_source_map_id = (
            None
            if exact_coordinate_source is None
            else exact_coordinate_source.source_map_id
        )
        self.coordinate_shadow_fingerprint = (
            None
            if exact_coordinate_source is None
            else canonical_fingerprint(
                array_tree_fingerprint(np.asarray(lifted.coordinates, dtype=np.float64))
            )
        )
        self.vertex_fixed_generators = jnp.asarray(fixed_vertices, dtype=jnp.bool_)
        self.vertex_representatives = jnp.asarray(representatives, dtype=jnp.int32)
        self.vertex_shifts = jnp.asarray(shifts, dtype=jnp.int32)
        self.corner_shifts = jnp.asarray(corner_shifts, dtype=jnp.int32)
        self.orbit_indices = jnp.asarray(all_orbits, dtype=jnp.int32)
        self.orbit_orientations = jnp.asarray(all_orientations, dtype=jnp.int32)
        self.orbit_shifts = jnp.asarray(all_shifts, dtype=jnp.int32)
        self.representatives = jnp.asarray(all_representatives, dtype=jnp.int32)
        self.lifted_offsets = jnp.asarray(offsets_lifted, dtype=jnp.int32)
        self.quotient_offsets = jnp.asarray(offsets_quotient, dtype=jnp.int32)
        self.entity_key_offsets = jnp.asarray(key_offsets, dtype=jnp.int64)
        self.entity_key_values = jnp.asarray(key_values, dtype=jnp.int64)
        self.topological_dimension = dimension
        self.lifted_complex_id = lifted.topology.topology_id
        self.lifted_entity_ids = tuple(
            jnp.asarray(entity.entity_ids, dtype=jnp.int64)
            for entity in lifted.topology.entity_sets
        )
        self.lifted_block_specs = tuple(
            (
                block.name,
                block.cell_kind,
                block.cell_count,
                canonical_fingerprint(array_tree_fingerprint(block.global_ids)),
            )
            for block in lifted.blocks
        )
        self.lifted_topology_id = lifted.topology_id
        self.allocator_next_ids = allocator
        self.periodic_topology_id = self._content_id()
        # The identity was just computed from these tables; check their structure.
        self._validate_tables()

    def _content_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "periodic-mesh-topology",
                "identification": identification_id(self.cell),
                "lifted_topology": self.lifted_topology_id,
                "lifted_complex": self.lifted_complex_id,
                "lifted_blocks": self.lifted_block_specs,
                "quotient_topology": self.quotient.topology_id,
                "lifted_entity_ids": array_tree_fingerprint(self.lifted_entity_ids),
                "quotient_entity_ids": array_tree_fingerprint(
                    tuple(entity.entity_ids for entity in self.quotient.entity_sets)
                ),
                "arrays": array_tree_fingerprint(
                    {
                        "vertex_fixed_generators": self.vertex_fixed_generators,
                        "vertex_representatives": self.vertex_representatives,
                        "vertex_shifts": self.vertex_shifts,
                        "corner_shifts": self.corner_shifts,
                        "orbit_indices": self.orbit_indices,
                        "orbit_orientations": self.orbit_orientations,
                        "orbit_shifts": self.orbit_shifts,
                        "representatives": self.representatives,
                        "lifted_offsets": self.lifted_offsets,
                        "quotient_offsets": self.quotient_offsets,
                        "entity_key_offsets": self.entity_key_offsets,
                        "entity_key_values": self.entity_key_values,
                    }
                ),
            }
        )

    def validate_restored(self, /) -> None:
        """Re-validate a stored descriptor: its structural tables and identity."""
        self._validate_tables()
        if isinstance(self.exact_coordinate_source, ExactPeriodicAlgebraicSource):
            _rebuild_periodic_witness(self.exact_coordinate_source, self.cell)
        if self._content_id() != self.periodic_topology_id:
            raise ValueError(
                "Stored periodic topology identity does not authenticate its structural tables."
            )

    def _validate_tables(self, /) -> None:
        if (self.exact_coordinate_source is None) != (
            self.coordinate_source_map_id is None
        ):
            raise ValueError(
                "Stored periodic topology lacks its exact coordinate witness."
            )
        if self.exact_coordinate_source is not None and (
            not isinstance(
                self.exact_coordinate_source,
                (ExactPowerCellGeometrySource, ExactPlcCellGeometrySource, ExactPeriodicAlgebraicSource),
            )
            or self.exact_coordinate_source.source_map_id != self.coordinate_source_map_id
        ):
            raise ValueError(
                "Stored periodic topology names a different exact coordinate witness."
            )
        if (self.coordinate_source_map_id is None) != (
            self.coordinate_shadow_fingerprint is None
        ) or (
            self.coordinate_source_map_id is not None
            and (
                not self.coordinate_source_map_id
                or not self.coordinate_shadow_fingerprint
            )
        ):
            raise ValueError(
                "Stored exact periodic coordinate source binding is incomplete."
            )
        dimension = self.topological_dimension
        rank = len(identification_orders(self.cell))
        lifted_offsets = np.asarray(self.lifted_offsets, dtype=np.int64)
        quotient_offsets = np.asarray(self.quotient_offsets, dtype=np.int64)
        if (
            len(self.quotient.entity_sets) != dimension + 1
            or lifted_offsets.shape != (dimension + 2,)
            or quotient_offsets.shape != (dimension + 2,)
            or lifted_offsets[0] != 0
            or quotient_offsets[0] != 0
            or np.any(np.diff(lifted_offsets) < 0)
            or np.any(np.diff(quotient_offsets) < 0)
        ):
            raise ValueError(
                "Stored periodic degree offsets do not match the quotient dimension."
            )
        vertex_count = int(np.asarray(self.lifted_entity_ids[0]).size)
        representatives = np.asarray(self.vertex_representatives, dtype=np.int64)
        shifts = np.asarray(self.vertex_shifts, dtype=np.int64)
        fixed = np.asarray(self.vertex_fixed_generators, dtype=np.bool_)
        if (
            representatives.shape != (vertex_count,)
            or shifts.shape != (vertex_count, rank)
            or fixed.shape != shifts.shape
            or np.any(representatives < 0)
            or np.any(representatives >= vertex_count)
            or np.any(representatives[representatives] != representatives)
            or np.any(shifts[representatives] != 0)
        ):
            raise ValueError(
                "Stored periodic vertex representative and winding tables are malformed."
            )
        lifted_count = int(lifted_offsets[-1])
        quotient_count = int(quotient_offsets[-1])
        orbit = np.asarray(self.orbit_indices, dtype=np.int64)
        orientation = np.asarray(self.orbit_orientations, dtype=np.int64)
        orbit_shifts = np.asarray(self.orbit_shifts, dtype=np.int64)
        reps = np.asarray(self.representatives, dtype=np.int64)
        if (
            orbit.shape != (lifted_count,)
            or orientation.shape != orbit.shape
            or orbit_shifts.shape != (lifted_count, rank)
            or reps.shape != (quotient_count,)
            or np.any(~np.isin(orientation, (-1, 1)))
        ):
            raise ValueError(
                "Stored quotient orbit/orientation tables have incompatible shapes."
            )
        for degree in range(dimension + 1):
            left, right = map(int, lifted_offsets[degree : degree + 2])
            quotient_left, quotient_right = map(
                int, quotient_offsets[degree : degree + 2]
            )
            if right - left != self.lifted_entity_ids[degree].shape[0]:
                raise ValueError("Lifted entity IDs do not align with degree offsets.")
            if quotient_right - quotient_left != self.quotient.entity_sets[degree].count:
                raise ValueError("Quotient entity IDs do not align with degree offsets.")
            if np.any(orbit[left:right] < quotient_left) or np.any(
                orbit[left:right] >= quotient_right
            ):
                raise ValueError("Orbit rows escape their quotient degree.")
            if np.any(reps[quotient_left:quotient_right] < 0) or np.any(
                reps[quotient_left:quotient_right] >= right - left
            ):
                raise ValueError("Quotient representatives escape their lifted degree.")
            local = reps[quotient_left:quotient_right]
            if np.any(
                orbit[left:right][local] != np.arange(quotient_left, quotient_right)
            ):
                raise ValueError(
                    "Stored quotient representative does not belong to its orbit."
                )
            if np.any(orbit_shifts[left + local] != 0):
                raise ValueError(
                    "Stored quotient representative must have zero relative winding."
                )
        keys = np.asarray(self.entity_key_offsets, dtype=np.int64)
        values = np.asarray(self.entity_key_values, dtype=np.int64)
        corners = np.asarray(self.corner_shifts, dtype=np.int64)
        if (
            keys.shape != (quotient_count + 1,)
            or keys[0] != 0
            or keys[-1] != values.size
            or np.any(np.diff(keys) < 0)
            or corners.ndim != 2
            or corners.shape[1] != rank
        ):
            raise ValueError("Stored periodic corner/key tables are malformed.")

    def require_exact_geometry(
        self, mesh: CellMesh, geometry: CellGeometrySpec, /
    ) -> None:
        """Bind the retained vertex witness to every actual coordinate-map corner.

        The retained vertex bank witnesses only the quotient's vertex deck
        action. It never replaces a higher-order coordinate map or proves its
        whole-facet traces; those remain independent mapped-geometry premises.
        """
        if not isinstance(mesh, CellMesh) or not isinstance(geometry, CellGeometrySpec):
            raise TypeError(
                "Periodic exact geometry reproof requires its actual mesh and map."
            )
        if mesh.periodic_topology is not self:
            raise ValueError("Periodic exact geometry requires its actual mesh topology.")
        self.validate_restored()
        self.require_lifted(mesh.topology, mesh.lifted_topology_id, mesh.blocks)
        self.require_coordinates(mesh.coordinates)
        elements, routes, _ = geometry.resolve(mesh)
        source = self.exact_coordinate_source
        if source is None:
            return
        source = _rebuild_periodic_witness(source, self.cell)
        if source.source_map_id != self.coordinate_source_map_id:
            raise ValueError("Periodic exact coordinate witness identity changed.")
        if isinstance(source, ExactPowerCellGeometrySource):
            if not isinstance(self.cell, PeriodicCell):
                raise TypeError("Exact periodic power witnesses require PeriodicCell.")
            source.require_periodic_lift(
                mesh.coordinates,
                self.cell,
                self.vertex_representatives,
                self.vertex_shifts,
            )
        else:
            source.require_periodic_lift(
                mesh.coordinates,
                self.cell,
                self.vertex_representatives,
                self.vertex_shifts,
            )
        expected = source.source_coordinates()
        vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
        if len(expected) != vertex_ids.size:
            raise ValueError(
                "Periodic exact witness does not cover the mesh vertex bank."
            )
        if isinstance(source, (ExactPlcCellGeometrySource, ExactPeriodicAlgebraicSource)) and not np.array_equal(
            np.asarray(source.vertex_global_ids, dtype=np.int64), vertex_ids
        ):
            raise ValueError("Periodic exact witness changed vertex identities or order.")

        # Canonical vertex geometry also supports polygonal and polyhedral
        # families, including isolated vertices. Prove its full basis/routes,
        # not merely a matching number of corners.
        from ..geometry._mesh_certificates import _same_tree_content

        affine = CellGeometrySpec.affine(mesh)
        affine_elements, affine_routes, _ = affine.resolve(mesh)
        if (
            geometry.restriction_source is None
            and _same_tree_content(elements, affine_elements)
            and all(
                np.array_equal(np.asarray(left), np.asarray(right))
                for left, right in zip(routes, affine_routes, strict=True)
            )
            and geometry.source_coordinates() == ExactCoordinateBank.from_rows(expected)
        ):
            return

        from ._reference_cell import reference_cell_topology

        seen: set[int] = set()
        for block in mesh.blocks:
            if block.cell_kind in ("polygon", "polyhedron"):
                raise ValueError(
                    "A noncanonical periodic polygonal map cannot prove exact vertex binding."
                )
            reference_vertices = tuple(
                tuple(Fraction(float(value)) for value in point)
                for point in reference_cell_topology(block.cell_kind).vertices
            )
            identifiers = np.asarray(block.global_ids, dtype=np.int64)
            values = geometry.evaluate_exact(
                mesh, identifiers, (reference_vertices,) * len(identifiers)
            )
            for rows, actual in zip(block.vertices, values, strict=True):
                for row, point in zip(rows, actual, strict=True):
                    index = int(row)
                    if tuple(point) != tuple(expected[index]):
                        raise ValueError(
                            "Actual periodic cell map disagrees with its exact vertex witness."
                        )
                    seen.add(index)
        if seen != set(range(vertex_ids.size)):
            raise ValueError("Exact periodic geometry did not prove every mesh vertex.")

    @classmethod
    @checked
    def from_stored_fields(
        cls: type[PeriodicMeshTopology],
        stored: PeriodicMeshTopology,
        lifted_mesh: CellMesh,
        /,
        *,
        actual_geometry: CellGeometrySpec | None = None,
        actual_mesh: CellMesh | None = None,
    ) -> PeriodicMeshTopology:
        """Rebuild quotient topology from its current canonical lifted mesh."""
        if not isinstance(stored, cls):
            raise TypeError("stored must be PeriodicMeshTopology.")
        if actual_geometry is not None and not isinstance(
            actual_geometry, CellGeometrySpec
        ):
            raise TypeError("actual_geometry must be CellGeometrySpec or None.")
        if actual_mesh is not None and not isinstance(actual_mesh, CellMesh):
            raise TypeError("actual_mesh must be CellMesh or None.")
        if actual_mesh is not None and actual_geometry is None:
            raise ValueError("An actual periodic mesh requires its coordinate geometry.")
        stored.validate_restored()
        dimension = stored.topological_dimension
        if (
            len(stored.quotient.entity_sets) != dimension + 1
            or len(stored.allocator_next_ids) != dimension + 1
        ):
            raise ValueError(
                "Stored periodic topology has malformed quotient dimensions."
            )
        exact_coordinate_source = stored.exact_coordinate_source
        if exact_coordinate_source is not None:
            exact_coordinate_source = _rebuild_periodic_witness(
                exact_coordinate_source, stored.cell
            )
            if exact_coordinate_source.source_map_id != stored.coordinate_source_map_id:
                raise ValueError("Stored periodic exact coordinate witness changed.")
        if actual_mesh is not None:
            from ..geometry._mesh_certificates import _same_tree_content

            actual_periodic = actual_mesh.periodic_topology
            if (
                actual_periodic is None
                or not _same_tree_content(actual_periodic, stored)
                or actual_mesh.lifted_topology_id != lifted_mesh.topology_id
                or actual_mesh.numeric_version != lifted_mesh.numeric_version
                or actual_mesh.ambient_dimension != lifted_mesh.ambient_dimension
                or actual_mesh.topological_dimension != lifted_mesh.topological_dimension
                or not _same_tree_content(actual_mesh.topology, lifted_mesh.topology)
                or not _same_tree_content(actual_mesh.blocks, lifted_mesh.blocks)
                or not np.array_equal(
                    np.asarray(actual_mesh.vertex_global_ids),
                    np.asarray(lifted_mesh.vertex_global_ids),
                )
                or not np.array_equal(
                    np.asarray(actual_mesh.coordinates, dtype=np.float64).view(np.uint64),
                    np.asarray(lifted_mesh.coordinates, dtype=np.float64).view(np.uint64),
                )
            ):
                raise ValueError(
                    "Actual periodic mesh differs from its lifted source or quotient."
                )
            assert actual_geometry is not None
            actual_periodic.require_exact_geometry(actual_mesh, actual_geometry)
        elif (
            actual_geometry is not None
            and exact_coordinate_source is not None
            and stored._actual_geometry is None
        ):
            raise ValueError(
                "Exact-witness periodic geometry reproof requires its actual periodic mesh."
            )
        if stored._actual_geometry is not None and actual_geometry is None:
            raise ValueError(
                "Periodic topology restore requires the current bound exact geometry."
            )
        if stored._actual_geometry is not None:
            from ..geometry._mesh_certificates import _same_tree_content

            assert actual_geometry is not None
            bound = stored._actual_geometry
            if actual_mesh is None:
                same_map = _same_tree_content(bound, actual_geometry)
            else:
                # The retained map can be bound to the lifted carrier while
                # the actual map must remain bound to the periodic endpoint.
                # Compare every mathematical/source field, not the epoch IDs.
                same_map = all(
                    _same_tree_content(
                        getattr(bound, name), getattr(actual_geometry, name)
                    )
                    for name in (
                        "block_names",
                        "elements",
                        "geometry_dofs",
                        "coordinates",
                        "exact_coordinate_bank",
                        "source_map",
                        "restriction_source",
                    )
                )
            if not same_map:
                raise ValueError(
                    "Periodic topology changed its originally bound coordinate geometry."
                )
        quotient_ids = {
            degree: np.asarray(entity.entity_ids, dtype=np.int64)
            for degree, entity in enumerate(stored.quotient.entity_sets)
        }
        allocator = {
            degree: int(next_id)
            for degree, next_id in enumerate(stored.allocator_next_ids)
        }
        # A witness-only topology remains independent of the mapped coordinate
        # geometry; its retained exact vertex source is revalidated separately.
        renewed = cls(
            lifted_mesh,
            stored.cell,
            stored.vertex_representatives,
            stored.vertex_shifts,
            entity_global_ids=quotient_ids,
            entity_allocator_next_ids=allocator,
            actual_geometry=stored._actual_geometry,
            exact_coordinate_source=exact_coordinate_source,
        )
        array_fields = (
            "vertex_fixed_generators",
            "vertex_representatives",
            "vertex_shifts",
            "corner_shifts",
            "orbit_indices",
            "orbit_orientations",
            "orbit_shifts",
            "representatives",
            "lifted_offsets",
            "quotient_offsets",
            "entity_key_offsets",
            "entity_key_values",
        )
        if (
            renewed.periodic_topology_id != stored.periodic_topology_id
            or renewed.coordinate_source_map_id != stored.coordinate_source_map_id
            or renewed.coordinate_shadow_fingerprint
            != stored.coordinate_shadow_fingerprint
            or renewed.lifted_topology_id != stored.lifted_topology_id
            or renewed.lifted_complex_id != stored.lifted_complex_id
            or renewed.lifted_block_specs != stored.lifted_block_specs
            or renewed.allocator_next_ids != stored.allocator_next_ids
            or renewed.quotient.topology_id != stored.quotient.topology_id
            or any(
                not np.array_equal(
                    np.asarray(getattr(renewed, name)),
                    np.asarray(getattr(stored, name)),
                )
                for name in array_fields
            )
            or len(renewed.lifted_entity_ids) != len(stored.lifted_entity_ids)
            or any(
                not np.array_equal(np.asarray(current), np.asarray(saved))
                for current, saved in zip(
                    renewed.lifted_entity_ids, stored.lifted_entity_ids, strict=True
                )
            )
        ):
            raise ValueError(
                "Reconstructed periodic quotient differs from its stored current source."
            )
        return renewed

    def require_lifted(
        self,
        topology: CellComplexTopology,
        topology_id: str,
        blocks: tuple[CellBlock | PolyhedralBlock, ...],
        /,
    ) -> None:
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be a CellComplexTopology.")
        bound_topology_id = canonical_fingerprint(
            {
                "kind": "periodic-cell-mesh-topology",
                "lifted_topology": self.lifted_topology_id,
                "periodic_topology": self.periodic_topology_id,
            }
        )
        if (
            topology_id not in (self.lifted_topology_id, bound_topology_id)
            or topology.topology_id != self.lifted_complex_id
        ):
            raise ValueError(
                "Periodic topology descriptor belongs to another lifted or bound topology."
            )
        if len(topology.entity_sets) != len(self.lifted_entity_ids):
            raise ValueError(
                "Periodic topology descriptor has another entity degree count."
            )
        if any(
            not np.array_equal(np.asarray(entity.entity_ids), np.asarray(ids))
            for entity, ids in zip(
                topology.entity_sets, self.lifted_entity_ids, strict=True
            )
        ):
            raise ValueError("Periodic topology descriptor lifted entity IDs changed.")
        specs = tuple(
            (
                str(block.name),
                str(block.cell_kind),
                int(block.cell_count),
                canonical_fingerprint(array_tree_fingerprint(block.global_ids)),
            )
            for block in blocks
        )
        if specs != self.lifted_block_specs:
            raise ValueError("Periodic topology descriptor lifted blocks changed.")
        if any(
            int(self.lifted_offsets[index + 1] - self.lifted_offsets[index])
            != entity.count
            for index, entity in enumerate(topology.entity_sets)
        ):
            raise ValueError(
                "Periodic topology descriptor lifted degree offsets changed."
            )
        if (
            sum(int(np.count_nonzero(block.vertex_valid)) for block in blocks)
            != self.corner_shifts.shape[0]
        ):
            raise ValueError(
                "Periodic topology descriptor corner shifts do not cover block corners."
            )

    def require_coordinates(self, coordinates: ArrayLike, /) -> None:
        if self.coordinate_source_map_id is None:
            _validate_vertex_orbits(
                coordinates,
                self.cell,
                self.vertex_representatives,
                self.vertex_shifts,
            )
            return
        if isinstance(self.exact_coordinate_source, ExactPeriodicAlgebraicSource):
            if self.exact_coordinate_source.source_map_id != self.coordinate_source_map_id:
                raise ValueError("Periodic exact coordinate witness identity changed.")
            self.exact_coordinate_source.require_periodic_lift(
                coordinates, self.cell, self.vertex_representatives, self.vertex_shifts
            )
        if (
            canonical_fingerprint(
                array_tree_fingerprint(np.asarray(coordinates, dtype=np.float64))
            )
            != self.coordinate_shadow_fingerprint
        ):
            raise ValueError(
                "Exact periodic power execution coordinates differ from their validated source shadow."
            )

    @property
    def vertex_orbit_count(self) -> int:
        return self.quotient.entity_sets[0].count


@final
class PeriodicMeasureReport(StrictModule, NonTrainableState):
    """Independent quotient measures evaluated once per quotient top cell."""

    orbit_measures: Array
    total_measure: Array
    cell_id: str = eqx.field(static=True)
    periodic_topology_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        orbit_measures: ArrayLike,
        cell_id: str,
        periodic_topology_id: str,
        geometry_layout_id: str,
        /,
    ) -> None:
        values = np.asarray(orbit_measures, dtype=np.float64)
        if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(
                "orbit_measures must be finite positive values per quotient top cell."
            )
        self.orbit_measures = jnp.asarray(values)
        self.total_measure = jnp.asarray(np.sum(values, dtype=np.float64))
        self.cell_id = str(cell_id)
        self.periodic_topology_id = str(periodic_topology_id)
        self.geometry_layout_id = str(geometry_layout_id)


__all__ = [
    "PeriodicMeasureReport",
    "PeriodicMeshTopology",
    "fixed_generators",
    "lifted_loops",
    "oriented_forms",
    "validate_periodic_vertex_orbits",
]
