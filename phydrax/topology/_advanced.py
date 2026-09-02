#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from scipy.spatial import Delaunay

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellComplexTopology, EntitySet, OrientedIncidence
from ..sparse import EdgeRelation
from ._coefficients import PrimeField
from ._resources import TopologyResourcePolicy


def _rank_mod(matrix: np.ndarray, modulus: int, /) -> int:
    value = np.asarray(matrix, dtype=np.int64).copy() % modulus
    rows, columns = value.shape
    rank = 0
    for column in range(columns):
        pivots = np.flatnonzero(value[rank:, column])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        value[[rank, pivot]] = value[[pivot, rank]]
        value[rank] = (
            value[rank] * pow(int(value[rank, column]), modulus - 2, modulus) % modulus
        )
        for row in range(rows):
            if row != rank and value[row, column]:
                value[row] = (value[row] - value[row, column] * value[rank]) % modulus
        rank += 1
        if rank == rows:
            break
    return rank


def _closed_simplices(
    maximal: Sequence[Sequence[int]], maximum_dimension: int, /
) -> tuple[np.ndarray, ...]:
    levels: list[set[tuple[int, ...]]] = [set() for _ in range(maximum_dimension + 1)]
    for simplex in maximal:
        ordered = tuple(sorted(int(value) for value in simplex))
        for size in range(1, min(len(ordered), maximum_dimension + 1) + 1):
            levels[size - 1].update(combinations(ordered, size))
    while levels and not levels[-1]:
        levels.pop()
    return tuple(
        np.asarray(sorted(level), dtype=np.int32).reshape((-1, degree + 1))
        for degree, level in enumerate(levels)
    )


def _simplex_topology(
    simplices: tuple[np.ndarray, ...], identifier: str, /
) -> CellComplexTopology:
    entities = tuple(
        EntitySet(
            f"{identifier}:cells:{degree}",
            degree,
            np.arange(level.shape[0], dtype=np.int64),
        )
        for degree, level in enumerate(simplices)
    )
    incidences = []
    for degree in range(1, len(simplices)):
        lower_lookup = {
            tuple(simplex): index for index, simplex in enumerate(simplices[degree - 1])
        }
        source = []
        target = []
        signs = []
        for upper_index, simplex in enumerate(simplices[degree]):
            for removed in range(degree + 1):
                face = tuple(np.delete(simplex, removed).tolist())
                if face not in lower_lookup:
                    raise ValueError("Point-cloud simplex family is not face closed.")
                source.append(lower_lookup[face])
                target.append(upper_index)
                signs.append(-1.0 if removed % 2 else 1.0)
        relation = EdgeRelation(
            np.asarray(source, dtype=np.int32),
            np.asarray(target, dtype=np.int32),
            source_size=entities[degree - 1].count,
            target_size=entities[degree].count,
        )
        incidences.append(
            OrientedIncidence(
                degree,
                entities[degree - 1],
                entities[degree],
                relation,
                np.asarray(signs),
                incidence_id=f"{identifier}:incidence:{degree}",
            )
        )
    return CellComplexTopology(entities, tuple(incidences), topology_id=identifier)


class PointCloudComplexPolicy(StrictModule, NonTrainableState):
    maximum_dimension: int = eqx.field(static=True)
    maximum_simplices: int = eqx.field(static=True)
    predicate_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_dimension: int = 3,
        maximum_simplices: int = 100_000,
        predicate_tolerance: float = 1e-10,
    ):
        if (
            int(maximum_dimension) < 0
            or int(maximum_simplices) < 1
            or float(predicate_tolerance) <= 0.0
        ):
            raise ValueError(
                "Point-cloud complex capacities and tolerance must be positive."
            )
        self.maximum_dimension = int(maximum_dimension)
        self.maximum_simplices = int(maximum_simplices)
        self.predicate_tolerance = float(predicate_tolerance)


class PointCloudComplexResult(StrictModule, NonTrainableState):
    topology: CellComplexTopology
    simplices: tuple[Array, ...]
    predicate_margins: Array
    ambiguous: Array
    certified: Array
    family: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        simplices: Sequence[ArrayLike],
        predicate_margins: ArrayLike,
        /,
        *,
        ambiguous: ArrayLike,
        certified: ArrayLike,
        family: str,
    ):
        self.topology = topology
        self.simplices = tuple(jnp.asarray(value, dtype=jnp.int32) for value in simplices)
        self.predicate_margins = jnp.asarray(predicate_margins)
        self.ambiguous = jnp.asarray(ambiguous, dtype=bool)
        self.certified = jnp.asarray(certified, dtype=bool)
        self.family = str(family)


def _validate_cloud(points: ArrayLike, policy: PointCloudComplexPolicy, /) -> np.ndarray:
    cloud = np.asarray(points)
    if cloud.ndim != 2 or cloud.shape[0] == 0 or cloud.shape[1] == 0:
        raise ValueError("Point cloud must have shape (points, ambient_dimension).")
    if not np.issubdtype(cloud.dtype, np.inexact) or not np.all(np.isfinite(cloud)):
        raise TypeError("Point cloud must contain finite inexact coordinates.")
    if np.unique(cloud, axis=0).shape[0] != cloud.shape[0]:
        raise ValueError("Point-cloud complexes reject duplicate points.")
    if not isinstance(policy, PointCloudComplexPolicy):
        raise TypeError("policy must be a PointCloudComplexPolicy.")
    return cloud


def vietoris_rips_complex(
    points: ArrayLike,
    radius: float,
    /,
    *,
    policy: PointCloudComplexPolicy | None = None,
    resources: TopologyResourcePolicy | None = None,
) -> PointCloudComplexResult:
    del resources
    policy_ = PointCloudComplexPolicy() if policy is None else policy
    cloud = _validate_cloud(points, policy_)
    radius_ = float(radius)
    if radius_ <= 0.0:
        raise ValueError("Vietoris-Rips radius must be positive.")
    distances = np.linalg.norm(cloud[:, None, :] - cloud[None, :, :], axis=-1)
    maximal = []
    margins = []
    for size in range(1, min(policy_.maximum_dimension + 2, cloud.shape[0] + 1)):
        for simplex in combinations(range(cloud.shape[0]), size):
            diameter = (
                0.0 if size == 1 else float(np.max(distances[np.ix_(simplex, simplex)]))
            )
            margin = radius_ - diameter
            if abs(margin) <= policy_.predicate_tolerance:
                raise ValueError(
                    "Vietoris-Rips edge predicate is ambiguous within tolerance."
                )
            if margin > 0.0:
                maximal.append(simplex)
                margins.append(margin)
    simplices = _closed_simplices(maximal, policy_.maximum_dimension)
    if sum(level.shape[0] for level in simplices) > policy_.maximum_simplices:
        raise ValueError("Vietoris-Rips complex exceeds maximum_simplices.")
    topology = _simplex_topology(simplices, "point-cloud:vietoris-rips")
    return PointCloudComplexResult(
        topology,
        simplices,
        np.asarray(margins),
        ambiguous=False,
        certified=True,
        family="vietoris-rips",
    )


def _smallest_enclosing_radius(points: np.ndarray, tolerance: float, /) -> float:
    if points.shape[0] <= 1:
        return 0.0
    best = np.inf
    ambient = points.shape[1]
    for support_size in range(1, min(points.shape[0], ambient + 1) + 1):
        for indices in combinations(range(points.shape[0]), support_size):
            support = points[np.asarray(indices)]
            if support_size == 1:
                center = support[0]
            else:
                base = support[0]
                directions = support[1:] - base
                matrix = 2.0 * (directions @ directions.T)
                rhs = np.sum(directions * directions, axis=1)
                coefficients = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
                if (
                    np.max(
                        np.abs(matrix @ coefficients - rhs),
                        initial=0.0,
                    )
                    > tolerance
                ):
                    continue
                center = base + directions.T @ coefficients
            radius = float(np.max(np.linalg.norm(points - center, axis=1)))
            if np.all(np.linalg.norm(points - center, axis=1) <= radius + tolerance):
                best = min(best, radius)
    if not np.isfinite(best):
        raise ValueError("Smallest-enclosing-ball predicate could not be certified.")
    return best


def cech_complex(
    points: ArrayLike,
    radius: float,
    /,
    *,
    policy: PointCloudComplexPolicy | None = None,
    resources: TopologyResourcePolicy | None = None,
) -> PointCloudComplexResult:
    del resources
    policy_ = PointCloudComplexPolicy() if policy is None else policy
    cloud = _validate_cloud(points, policy_)
    radius_ = float(radius)
    if radius_ <= 0.0:
        raise ValueError("Cech radius must be positive.")
    accepted = []
    margins = []
    for size in range(1, min(policy_.maximum_dimension + 2, cloud.shape[0] + 1)):
        for simplex in combinations(range(cloud.shape[0]), size):
            enclosing = _smallest_enclosing_radius(
                cloud[np.asarray(simplex)], policy_.predicate_tolerance
            )
            margin = radius_ - enclosing
            if abs(margin) <= policy_.predicate_tolerance:
                raise ValueError("Cech nerve predicate is ambiguous within tolerance.")
            if margin > 0.0:
                accepted.append(simplex)
                margins.append(margin)
    simplices = _closed_simplices(accepted, policy_.maximum_dimension)
    if sum(level.shape[0] for level in simplices) > policy_.maximum_simplices:
        raise ValueError("Cech complex exceeds maximum_simplices.")
    topology = _simplex_topology(simplices, "point-cloud:cech")
    return PointCloudComplexResult(
        topology,
        simplices,
        np.asarray(margins),
        ambiguous=False,
        certified=True,
        family="cech",
    )


def _alpha_delaunay_simplices(
    cloud: np.ndarray, tolerance: float, /
) -> tuple[np.ndarray, ...]:
    centered = cloud - cloud[0]
    scale = float(np.max(np.linalg.norm(centered, axis=1), initial=0.0))
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    rank = int(np.count_nonzero(singular_values > tolerance * max(1.0, scale)))
    if rank == 0:
        return (np.arange(cloud.shape[0], dtype=np.int32)[:, None],)
    projected = centered @ right_vectors[:rank].T
    if cloud.shape[0] == rank + 1:
        maximal = (tuple(range(cloud.shape[0])),)
    elif rank == 1:
        order = np.argsort(projected[:, 0])
        maximal = tuple(
            tuple(sorted((int(left), int(right))))
            for left, right in zip(order[:-1], order[1:], strict=True)
        )
    else:
        triangulation = Delaunay(projected)
        maximal = tuple(
            sorted(
                {
                    tuple(sorted(int(value) for value in simplex))
                    for simplex in triangulation.simplices
                }
            )
        )
    return _closed_simplices(maximal, rank)


def _circumsphere(points: np.ndarray, tolerance: float, /) -> tuple[np.ndarray, float]:
    if points.shape[0] == 1:
        return points[0], 0.0
    base = points[0]
    directions = points[1:] - base
    matrix = 2.0 * (directions @ directions.T)
    rhs = np.sum(directions * directions, axis=1)
    coefficients = np.linalg.solve(matrix, rhs)
    residual = float(np.max(np.abs(matrix @ coefficients - rhs), initial=0.0))
    if residual > tolerance * max(1.0, float(np.max(np.abs(rhs), initial=0.0))):
        raise ValueError("Alpha circumsphere could not be certified.")
    center = base + directions.T @ coefficients
    distances = np.linalg.norm(points - center, axis=1)
    radius = float(distances[0])
    if np.max(np.abs(distances - radius), initial=0.0) > tolerance:
        raise ValueError("Alpha circumsphere could not be certified.")
    return center, radius


def alpha_complex(
    points: ArrayLike,
    radius: float,
    /,
    *,
    policy: PointCloudComplexPolicy | None = None,
    resources: TopologyResourcePolicy | None = None,
) -> PointCloudComplexResult:
    del resources
    policy_ = PointCloudComplexPolicy() if policy is None else policy
    cloud = _validate_cloud(points, policy_)
    cloud = cloud.astype(np.float64, copy=False)
    if cloud.shape[1] > 3:
        raise ValueError(
            "Alpha complexes support declared Euclidean ambient dimension at most three."
        )
    radius_ = float(radius)
    if not np.isfinite(radius_) or radius_ <= 0.0:
        raise ValueError("Alpha radius must be finite and positive.")

    candidates = _alpha_delaunay_simplices(cloud, policy_.predicate_tolerance)
    alpha_values: list[dict[tuple[int, ...], float]] = [
        {} for _ in range(len(candidates))
    ]
    for degree in range(len(candidates) - 1, -1, -1):
        for raw_simplex in candidates[degree]:
            simplex = tuple(int(value) for value in raw_simplex)
            center, intrinsic_radius = _circumsphere(
                cloud[np.asarray(simplex)], policy_.predicate_tolerance
            )
            distances = np.linalg.norm(cloud - center, axis=1)
            outside = np.ones((cloud.shape[0],), dtype=bool)
            outside[np.asarray(simplex)] = False
            outside_distances = distances[outside]
            if degree == len(candidates) - 1 and np.any(
                np.abs(outside_distances - intrinsic_radius)
                <= policy_.predicate_tolerance
            ):
                raise ValueError(
                    "Alpha Delaunay predicate is ambiguous within tolerance."
                )
            intrinsic_empty = not np.any(
                outside_distances < intrinsic_radius - policy_.predicate_tolerance
            )
            value = alpha_values[degree].get(simplex, np.inf)
            if intrinsic_empty:
                value = min(value, intrinsic_radius)
            if not np.isfinite(value):
                raise ValueError(
                    "Alpha empty-circumsphere predicate could not be certified."
                )
            alpha_values[degree][simplex] = value
            if degree:
                for face in combinations(simplex, degree):
                    inherited = alpha_values[degree - 1].get(face, np.inf)
                    alpha_values[degree - 1][face] = min(inherited, value)

    maximum_dimension = min(policy_.maximum_dimension, len(candidates) - 1)
    accepted: list[tuple[int, ...]] = []
    margins = []
    for degree in range(maximum_dimension + 1):
        for raw_simplex in candidates[degree]:
            simplex = tuple(int(value) for value in raw_simplex)
            margin = radius_ - alpha_values[degree][simplex]
            if degree == 0:
                accepted.append(simplex)
                margins.append(margin)
                continue
            if abs(margin) <= policy_.predicate_tolerance:
                raise ValueError(
                    "Alpha empty-circumsphere predicate is ambiguous within tolerance."
                )
            if margin > 0.0:
                accepted.append(simplex)
                margins.append(margin)
    simplices = _closed_simplices(accepted, maximum_dimension)
    if sum(level.shape[0] for level in simplices) > policy_.maximum_simplices:
        raise ValueError("Alpha complex exceeds maximum_simplices.")
    topology = _simplex_topology(simplices, "point-cloud:alpha")
    return PointCloudComplexResult(
        topology,
        simplices,
        np.asarray(margins),
        ambiguous=False,
        certified=True,
        family="alpha",
    )


class MultiFiltration(StrictModule, NonTrainableState):
    topology: CellComplexTopology
    grades: tuple[Array, ...]
    parameter_dimension: int = eqx.field(static=True)

    def __init__(self, topology: CellComplexTopology, grades: Sequence[ArrayLike], /):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be a CellComplexTopology.")
        grades_ = tuple(np.asarray(value) for value in grades)
        if len(grades_) != len(topology.entity_sets) or not grades_:
            raise ValueError("One grade array is required per cell degree.")
        dimension = grades_[0].shape[-1]
        if dimension < 2 or any(
            value.shape != (entities.count, dimension)
            for value, entities in zip(grades_, topology.entity_sets, strict=True)
        ):
            raise ValueError(
                "Multifiltration grades must have shape (cells, parameters), with parameters >= 2."
            )
        for incidence in topology.incidences:
            valid = np.asarray(incidence.relation.valid)
            lower = np.asarray(incidence.relation.source_indices)[valid]
            upper = np.asarray(incidence.relation.target_indices)[valid]
            if np.any(
                grades_[incidence.degree - 1][lower] > grades_[incidence.degree][upper]
            ):
                raise ValueError("Multifiltration grades violate face monotonicity.")
        self.topology = topology
        self.grades = tuple(jnp.asarray(value) for value in grades_)
        self.parameter_dimension = dimension


class FinitePersistenceModule(StrictModule, NonTrainableState):
    dimensions: Array
    edges: Array
    maps: tuple[Array, ...]
    field: PrimeField

    def __init__(
        self,
        dimensions: ArrayLike,
        edges: ArrayLike,
        maps: Sequence[ArrayLike],
        /,
        *,
        field: PrimeField,
    ):
        dimensions_ = np.asarray(dimensions, dtype=np.int32)
        edges_ = np.asarray(edges, dtype=np.int32)
        maps_ = tuple(np.asarray(value, dtype=np.int64) % field.modulus for value in maps)
        if (
            dimensions_.ndim != 1
            or np.any(dimensions_ < 0)
            or edges_.shape != (len(maps_), 2)
        ):
            raise ValueError("Finite module dimensions/edges have incompatible shapes.")
        for edge, matrix in zip(edges_, maps_, strict=True):
            source, target = map(int, edge)
            if not (
                0 <= source < len(dimensions_) and 0 <= target < len(dimensions_)
            ) or matrix.shape != (dimensions_[target], dimensions_[source]):
                raise ValueError("Finite module map shape does not match its edge.")
            if source == target and not np.array_equal(
                matrix,
                np.eye(dimensions_[source], dtype=np.int64) % field.modulus,
            ):
                raise ValueError("Finite module identity maps must be exact identities.")
        path_products: dict[tuple[int, int], list[np.ndarray]] = {}
        for first_edge, first_map in zip(edges_, maps_, strict=True):
            for second_edge, second_map in zip(edges_, maps_, strict=True):
                if int(first_edge[1]) != int(second_edge[0]):
                    continue
                pair = (int(first_edge[0]), int(second_edge[1]))
                path_products.setdefault(pair, []).append(
                    (second_map @ first_map) % field.modulus
                )
        for products in path_products.values():
            if any(not np.array_equal(products[0], product) for product in products[1:]):
                raise ValueError("Finite persistence module has noncommuting path maps.")
        self.dimensions = jnp.asarray(dimensions_)
        self.edges = jnp.asarray(edges_)
        self.maps = tuple(jnp.asarray(value, dtype=jnp.int32) for value in maps_)
        self.field = field


class MultiparameterPersistenceResult(StrictModule, NonTrainableState):
    hilbert_dimensions: Array
    rank_queries: Array
    presentation_relations: Array
    fibered_diagrams: tuple[Array, ...]
    barcode_claimed: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        hilbert_dimensions: ArrayLike,
        rank_queries: ArrayLike,
        presentation_relations: ArrayLike,
        fibered_diagrams: Sequence[ArrayLike],
    ):
        self.hilbert_dimensions = jnp.asarray(hilbert_dimensions, dtype=jnp.int32)
        self.rank_queries = jnp.asarray(rank_queries, dtype=jnp.int32)
        self.presentation_relations = jnp.asarray(presentation_relations, dtype=jnp.int32)
        self.fibered_diagrams = tuple(jnp.asarray(value) for value in fibered_diagrams)
        self.barcode_claimed = False


def compute_multiparameter_persistence(
    module: FinitePersistenceModule,
    /,
    *,
    rank_edges: Sequence[int] = (),
    fibered_diagrams: Sequence[ArrayLike] = (),
) -> MultiparameterPersistenceResult:
    if not isinstance(module, FinitePersistenceModule):
        raise TypeError("module must be a FinitePersistenceModule.")
    ranks = []
    for index in rank_edges:
        if not 0 <= int(index) < len(module.maps):
            raise ValueError("Rank query addresses an unknown module edge.")
        ranks.append(_rank_mod(np.asarray(module.maps[int(index)]), module.field.modulus))
    return MultiparameterPersistenceResult(
        hilbert_dimensions=module.dimensions,
        rank_queries=np.asarray(ranks, dtype=np.int32),
        presentation_relations=module.edges,
        fibered_diagrams=fibered_diagrams,
    )


class ZigzagIntervalResult(StrictModule, NonTrainableState):
    intervals: Array
    dimensions: Array
    reconstructed_dimensions: Array
    reconstructed_edge_ranks: Array
    valid: Array

    def __init__(
        self, intervals: ArrayLike, dimensions: ArrayLike, edge_ranks: ArrayLike, /
    ):
        intervals_ = np.asarray(intervals, dtype=np.int32).reshape((-1, 2))
        dimensions_ = np.asarray(dimensions, dtype=np.int32)
        reconstructed = np.asarray(
            [
                np.sum((intervals_[:, 0] <= node) & (intervals_[:, 1] >= node))
                for node in range(len(dimensions_))
            ],
            dtype=np.int32,
        )
        edge_reconstructed = np.asarray(
            [
                np.sum((intervals_[:, 0] <= edge) & (intervals_[:, 1] >= edge + 1))
                for edge in range(max(0, len(dimensions_) - 1))
            ],
            dtype=np.int32,
        )
        edge_ranks_ = np.asarray(edge_ranks, dtype=np.int32)
        self.intervals = jnp.asarray(intervals_)
        self.dimensions = jnp.asarray(dimensions_)
        self.reconstructed_dimensions = jnp.asarray(reconstructed)
        self.reconstructed_edge_ranks = jnp.asarray(edge_reconstructed)
        self.valid = jnp.asarray(
            np.array_equal(reconstructed, dimensions_)
            and np.array_equal(edge_reconstructed, edge_ranks_)
        )


def _generalized_rank(
    dimensions: np.ndarray,
    maps: tuple[np.ndarray, ...],
    directions: tuple[str, ...],
    start: int,
    stop: int,
    modulus: int,
    /,
) -> int:
    local_dimensions = dimensions[start : stop + 1]
    offsets = np.concatenate(([0], np.cumsum(local_dimensions)))
    total = int(offsets[-1])
    blocks = []
    for local_edge, edge in enumerate(range(start, stop)):
        left = int(local_dimensions[local_edge])
        right = int(local_dimensions[local_edge + 1])
        matrix = maps[edge] % modulus
        if directions[edge] == "forward":
            if matrix.shape != (right, left):
                raise ValueError("Forward zigzag map has incompatible shape.")
            block = np.zeros((right, total), dtype=np.int64)
            block[:, offsets[local_edge] : offsets[local_edge + 1]] = matrix
            block[:, offsets[local_edge + 1] : offsets[local_edge + 2]] = -np.eye(
                right, dtype=np.int64
            )
        else:
            if matrix.shape != (left, right):
                raise ValueError("Backward zigzag map has incompatible shape.")
            block = np.zeros((left, total), dtype=np.int64)
            block[:, offsets[local_edge] : offsets[local_edge + 1]] = -np.eye(
                left, dtype=np.int64
            )
            block[:, offsets[local_edge + 1] : offsets[local_edge + 2]] = matrix
        blocks.append(block)
    constraint = (
        np.concatenate(blocks, axis=0) % modulus
        if blocks
        else np.zeros((0, total), dtype=np.int64)
    )
    rref = constraint.copy()
    pivots = []
    row = 0
    for column in range(total):
        candidates = np.flatnonzero(rref[row:, column])
        if candidates.size == 0:
            continue
        pivot = row + int(candidates[0])
        rref[[row, pivot]] = rref[[pivot, row]]
        rref[row] = (
            rref[row] * pow(int(rref[row, column]), modulus - 2, modulus) % modulus
        )
        for other in range(rref.shape[0]):
            if other != row and rref[other, column]:
                rref[other] = (rref[other] - rref[other, column] * rref[row]) % modulus
        pivots.append(column)
        row += 1
        if row == rref.shape[0]:
            break
    free = [column for column in range(total) if column not in pivots]
    kernel = np.zeros((total, len(free)), dtype=np.int64)
    for index, column in enumerate(free):
        kernel[column, index] = 1
        for pivot_row, pivot_column in enumerate(pivots):
            kernel[pivot_column, index] = -rref[pivot_row, column] % modulus
    relation = constraint.T
    inclusion = np.zeros((total, local_dimensions[0]), dtype=np.int64)
    inclusion[: local_dimensions[0], :] = np.eye(local_dimensions[0], dtype=np.int64)
    image = inclusion @ kernel[: local_dimensions[0], :]
    return _rank_mod(np.concatenate((relation, image), axis=1), modulus) - _rank_mod(
        relation, modulus
    )


def compute_zigzag_intervals(
    dimensions: Sequence[int],
    maps: Sequence[ArrayLike],
    directions: Sequence[str],
    /,
    *,
    coefficients: PrimeField,
) -> ZigzagIntervalResult:
    dimensions_ = np.asarray(tuple(int(value) for value in dimensions), dtype=np.int32)
    maps_ = tuple(np.asarray(value, dtype=np.int64) for value in maps)
    directions_ = tuple(str(value) for value in directions)
    if (
        len(maps_) != max(0, len(dimensions_) - 1)
        or len(directions_) != len(maps_)
        or any(value not in ("forward", "backward") for value in directions_)
    ):
        raise ValueError(
            "Zigzag requires one forward/backward map per adjacent node pair."
        )
    ranks = np.zeros((len(dimensions_), len(dimensions_)), dtype=np.int32)
    for start in range(len(dimensions_)):
        for stop in range(start, len(dimensions_)):
            ranks[start, stop] = _generalized_rank(
                dimensions_, maps_, directions_, start, stop, coefficients.modulus
            )
    intervals = []
    for start in range(len(dimensions_)):
        for stop in range(start, len(dimensions_)):
            multiplicity = ranks[start, stop]
            if start > 0:
                multiplicity -= ranks[start - 1, stop]
            if stop + 1 < len(dimensions_):
                multiplicity -= ranks[start, stop + 1]
            if start > 0 and stop + 1 < len(dimensions_):
                multiplicity += ranks[start - 1, stop + 1]
            if multiplicity < 0:
                raise ValueError(
                    "Zigzag generalized ranks do not define an interval decomposition."
                )
            intervals.extend((start, stop) for _ in range(int(multiplicity)))
    edge_ranks = np.asarray(
        [_rank_mod(matrix, coefficients.modulus) for matrix in maps_], dtype=np.int32
    )
    result = ZigzagIntervalResult(
        np.asarray(intervals, dtype=np.int32), dimensions_, edge_ranks
    )
    if not bool(result.valid):
        raise ValueError(
            "Zigzag interval decomposition failed dimension/map-rank reconstruction."
        )
    return result


class CellDiagonalApproximation(StrictModule, NonTrainableState):
    source_degree: int = eqx.field(static=True)
    source_cells: Array
    left_cells: Array
    right_cells: Array
    left_degree: int = eqx.field(static=True)
    right_degree: int = eqx.field(static=True)
    coefficients: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: CellComplexTopology,
        source_degree: int,
        left_degree: int,
        right_degree: int,
        source_cells: ArrayLike,
        left_cells: ArrayLike,
        right_cells: ArrayLike,
        coefficients: ArrayLike,
        /,
    ):
        if int(left_degree) + int(right_degree) != int(source_degree):
            raise ValueError("Diagonal term degrees must sum to source_degree.")
        arrays = tuple(
            np.asarray(value)
            for value in (source_cells, left_cells, right_cells, coefficients)
        )
        if any(value.ndim != 1 or value.shape != arrays[0].shape for value in arrays):
            raise ValueError("Diagonal term arrays must be equal rank-1 arrays.")
        if (
            np.any(arrays[0] < 0)
            or np.any(arrays[0] >= topology.entity_sets[int(source_degree)].count)
            or np.any(arrays[1] < 0)
            or np.any(arrays[1] >= topology.entity_sets[int(left_degree)].count)
            or np.any(arrays[2] < 0)
            or np.any(arrays[2] >= topology.entity_sets[int(right_degree)].count)
        ):
            raise ValueError("Diagonal term addresses a cell outside the topology.")
        self.source_degree = int(source_degree)
        self.left_degree = int(left_degree)
        self.right_degree = int(right_degree)
        self.source_cells = jnp.asarray(arrays[0], dtype=jnp.int32)
        self.left_cells = jnp.asarray(arrays[1], dtype=jnp.int32)
        self.right_cells = jnp.asarray(arrays[2], dtype=jnp.int32)
        self.coefficients = jnp.asarray(arrays[3], dtype=jnp.int32)
        self.topology_id = topology.topology_id


def cup_product(
    left: ArrayLike,
    right: ArrayLike,
    diagonal: CellDiagonalApproximation,
    /,
    *,
    coefficients: PrimeField,
) -> Array:
    left_ = jnp.asarray(left, dtype=jnp.int32)
    right_ = jnp.asarray(right, dtype=jnp.int32)
    terms = (
        diagonal.coefficients * left_[diagonal.left_cells] * right_[diagonal.right_cells]
    )
    count = int(np.max(np.asarray(diagonal.source_cells), initial=-1)) + 1
    return (
        jnp.zeros((count,), dtype=jnp.int32).at[diagonal.source_cells].add(terms)
        % coefficients.modulus
    )


def _matmul_mod(left: np.ndarray, right: np.ndarray, modulus: int, /) -> np.ndarray:
    result = np.zeros((left.shape[0], right.shape[1]), dtype=np.int64)
    for index in range(left.shape[1]):
        result = (result + left[:, index, None] * right[index, None, :]) % modulus
    return result


def _cellular_sheaf_route_maps(
    topology: CellComplexTopology,
    restrictions: tuple[np.ndarray, ...],
    /,
) -> tuple[dict[tuple[int, int], np.ndarray], ...]:
    route_maps = []
    cursor = 0
    for incidence in topology.incidences:
        valid = np.asarray(incidence.relation.valid, dtype=bool)
        lower = np.asarray(incidence.relation.source_indices)
        upper = np.asarray(incidence.relation.target_indices)
        degree_maps = {}
        for route in range(incidence.relation.capacity):
            if valid[route]:
                degree_maps[(int(lower[route]), int(upper[route]))] = restrictions[cursor]
            cursor += 1
        route_maps.append(degree_maps)
    return tuple(route_maps)


def _validate_restriction_paths(
    topology: CellComplexTopology,
    dimensions: tuple[np.ndarray, ...],
    route_maps: tuple[dict[tuple[int, int], np.ndarray], ...],
    modulus: int,
    /,
) -> None:
    for start_degree in range(len(topology.entity_sets) - 2):
        for start_cell in range(topology.entity_sets[start_degree].count):
            paths = {
                start_cell: np.eye(
                    int(dimensions[start_degree][start_cell]), dtype=np.int64
                )
            }
            for upper_degree in range(start_degree + 1, len(topology.entity_sets)):
                candidates: dict[int, np.ndarray] = {}
                for (lower_cell, upper_cell), restriction in route_maps[
                    upper_degree - 1
                ].items():
                    if lower_cell not in paths:
                        continue
                    composed = _matmul_mod(restriction, paths[lower_cell], modulus)
                    if upper_cell in candidates and not np.array_equal(
                        candidates[upper_cell], composed
                    ):
                        raise ValueError(
                            "Cellular sheaf has incompatible restriction routes."
                        )
                    candidates[upper_cell] = composed
                paths = candidates


def _cellular_sheaf_differentials(
    topology: CellComplexTopology,
    dimensions: tuple[np.ndarray, ...],
    restrictions: tuple[np.ndarray, ...],
    modulus: int,
    /,
) -> tuple[np.ndarray, ...]:
    totals = tuple(int(np.sum(value, dtype=np.int64)) for value in dimensions)
    differentials = []
    cursor = 0
    for incidence in topology.incidences:
        lower_offsets = np.concatenate(
            ([0], np.cumsum(dimensions[incidence.degree - 1], dtype=np.int64))
        )
        upper_offsets = np.concatenate(
            ([0], np.cumsum(dimensions[incidence.degree], dtype=np.int64))
        )
        matrix = np.zeros(
            (totals[incidence.degree], totals[incidence.degree - 1]),
            dtype=np.int64,
        )
        valid = np.asarray(incidence.relation.valid, dtype=bool)
        for route in range(incidence.relation.capacity):
            if valid[route]:
                lower = int(incidence.relation.source_indices[route])
                upper = int(incidence.relation.target_indices[route])
                matrix[
                    upper_offsets[upper] : upper_offsets[upper + 1],
                    lower_offsets[lower] : lower_offsets[lower + 1],
                ] = (int(incidence.signs[route]) * restrictions[cursor]) % modulus
            cursor += 1
        differentials.append(matrix)
    return tuple(differentials)


def _validate_coboundaries(
    differentials: tuple[np.ndarray, ...], modulus: int, /
) -> None:
    for lower, upper in zip(differentials[:-1], differentials[1:], strict=True):
        if np.any(_matmul_mod(upper, lower, modulus)):
            raise ValueError(
                "Cellular sheaf has nonzero consecutive coboundary composition."
            )


class CellularSheaf(StrictModule, NonTrainableState):
    topology: CellComplexTopology
    stalk_dimensions: tuple[Array, ...]
    restrictions: tuple[Array, ...]
    field: PrimeField

    def __init__(
        self,
        topology: CellComplexTopology,
        stalk_dimensions: Sequence[ArrayLike],
        restrictions: Sequence[ArrayLike],
        /,
        *,
        field: PrimeField,
    ):
        if not isinstance(topology, CellComplexTopology):
            raise TypeError("topology must be a CellComplexTopology.")
        if not isinstance(field, PrimeField):
            raise TypeError("field must be a PrimeField.")
        raw_dimensions = tuple(np.asarray(value) for value in stalk_dimensions)
        if len(raw_dimensions) != len(topology.entity_sets) or any(
            value.shape != (entity.count,)
            for value, entity in zip(raw_dimensions, topology.entity_sets, strict=True)
        ):
            raise ValueError("Cellular sheaf requires one stalk dimension per cell.")
        if any(not np.issubdtype(value.dtype, np.integer) for value in raw_dimensions):
            raise TypeError("Cellular sheaf stalk dimensions must be integers.")
        if any(
            np.any(value < 0) or np.any(value > np.iinfo(np.int32).max)
            for value in raw_dimensions
        ):
            raise ValueError("Cellular sheaf stalk dimensions must be nonnegative.")
        dimensions = tuple(value.astype(np.int32, copy=False) for value in raw_dimensions)

        raw_maps = tuple(np.asarray(value) for value in restrictions)
        if any(not np.issubdtype(value.dtype, np.integer) for value in raw_maps):
            raise TypeError("Cellular sheaf restrictions must have integer coefficients.")
        maps = tuple(
            value.astype(np.int64, copy=False) % field.modulus for value in raw_maps
        )
        route_count = sum(
            incidence.relation.capacity for incidence in topology.incidences
        )
        if len(maps) != route_count:
            raise ValueError(
                "Cellular sheaf requires one restriction matrix per incidence route."
            )
        cursor = 0
        for incidence in topology.incidences:
            valid = np.asarray(incidence.relation.valid, dtype=bool)
            lower = np.asarray(incidence.relation.source_indices)
            upper = np.asarray(incidence.relation.target_indices)
            for route in range(incidence.relation.capacity):
                matrix = maps[cursor]
                if valid[route] and matrix.shape != (
                    int(dimensions[incidence.degree][upper[route]]),
                    int(dimensions[incidence.degree - 1][lower[route]]),
                ):
                    raise ValueError(
                        "Sheaf restriction matrix has incompatible stalk dimensions."
                    )
                cursor += 1

        route_maps = _cellular_sheaf_route_maps(topology, maps)
        _validate_restriction_paths(topology, dimensions, route_maps, field.modulus)
        differentials = _cellular_sheaf_differentials(
            topology, dimensions, maps, field.modulus
        )
        _validate_coboundaries(differentials, field.modulus)
        self.topology = topology
        self.stalk_dimensions = tuple(jnp.asarray(value) for value in dimensions)
        self.restrictions = tuple(jnp.asarray(value, dtype=jnp.int32) for value in maps)
        self.field = field

    def cohomology_dimensions(self, /) -> Array:
        dimensions = tuple(
            np.asarray(value, dtype=np.int32) for value in self.stalk_dimensions
        )
        restrictions = tuple(
            np.asarray(value, dtype=np.int64) for value in self.restrictions
        )
        differentials = _cellular_sheaf_differentials(
            self.topology, dimensions, restrictions, self.field.modulus
        )
        _validate_coboundaries(differentials, self.field.modulus)
        totals = tuple(int(np.sum(value, dtype=np.int64)) for value in dimensions)
        ranks = (
            [0] + [_rank_mod(value, self.field.modulus) for value in differentials] + [0]
        )
        cohomology = np.asarray(
            [
                totals[degree] - ranks[degree] - ranks[degree + 1]
                for degree in range(len(totals))
            ],
            dtype=np.int64,
        )
        if np.any(cohomology < 0):
            raise ValueError("Cellular sheaf cohomology dimensions are inconsistent.")
        return jnp.asarray(cohomology, dtype=jnp.int32)


class FilteredChainComplex(StrictModule, NonTrainableState):
    boundaries: tuple[Array, ...]
    filtration: tuple[Array, ...]
    field: PrimeField

    def __init__(
        self,
        boundaries: Sequence[ArrayLike],
        filtration: Sequence[ArrayLike],
        /,
        *,
        field: PrimeField,
    ):
        boundaries_ = tuple(
            np.asarray(value, dtype=np.int64) % field.modulus for value in boundaries
        )
        filtration_ = tuple(np.asarray(value, dtype=np.int32) for value in filtration)
        if len(filtration_) != len(boundaries_) + 1:
            raise ValueError(
                "Filtered chain complex needs one filtration vector per chain degree."
            )
        for degree, boundary in enumerate(boundaries_, start=1):
            if boundary.shape != (filtration_[degree - 1].size, filtration_[degree].size):
                raise ValueError(
                    "Boundary shape does not match filtered chain dimensions."
                )
            rows, columns = np.nonzero(boundary)
            if np.any(filtration_[degree - 1][rows] > filtration_[degree][columns]):
                raise ValueError("Boundary increases the declared filtration.")
        for lower, upper in zip(boundaries_[:-1], boundaries_[1:], strict=True):
            if np.any((lower @ upper) % field.modulus):
                raise ValueError(
                    "Filtered chain boundaries violate boundary-of-boundary zero."
                )
        self.boundaries = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in boundaries_
        )
        self.filtration = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in filtration_
        )
        self.field = field


class FilteredBicomplex(StrictModule, NonTrainableState):
    horizontal: tuple[Array, ...]
    vertical: tuple[Array, ...]
    field: PrimeField

    def __init__(
        self,
        horizontal: Sequence[ArrayLike],
        vertical: Sequence[ArrayLike],
        /,
        *,
        field: PrimeField,
    ):
        horizontal_ = tuple(
            np.asarray(value, dtype=np.int64) % field.modulus for value in horizontal
        )
        vertical_ = tuple(
            np.asarray(value, dtype=np.int64) % field.modulus for value in vertical
        )
        if len(horizontal_) != len(vertical_):
            raise ValueError(
                "Filtered bicomplex horizontal/vertical families must align."
            )
        for h, v in zip(horizontal_, vertical_, strict=True):
            if (
                h.shape != v.shape
                or h.shape[0] != h.shape[1]
                or np.any((h @ v + v @ h) % field.modulus)
            ):
                raise ValueError(
                    "Bicomplex differentials do not anticommute on the declared layout."
                )
        self.horizontal = tuple(
            jnp.asarray(value, dtype=jnp.int32) for value in horizontal_
        )
        self.vertical = tuple(jnp.asarray(value, dtype=jnp.int32) for value in vertical_)
        self.field = field


class SpectralSequenceResult(StrictModule, NonTrainableState):
    page_dimensions: Array
    differential_ranks: Array
    stabilized_page: Array
    convergence_certified: Array
    extension_resolved: bool = eqx.field(static=True)

    def __init__(
        self,
        page_dimensions: ArrayLike,
        differential_ranks: ArrayLike,
        stabilized_page: ArrayLike,
        convergence_certified: ArrayLike,
        /,
    ):
        self.page_dimensions = jnp.asarray(page_dimensions, dtype=jnp.int32)
        self.differential_ranks = jnp.asarray(differential_ranks, dtype=jnp.int32)
        self.stabilized_page = jnp.asarray(stabilized_page, dtype=jnp.int32)
        self.convergence_certified = jnp.asarray(convergence_certified, dtype=bool)
        self.extension_resolved = False


def _spectral_pairs(
    complex: FilteredChainComplex, /
) -> list[tuple[int, int, int, int, int]]:
    modulus = complex.field.modulus
    pairs: list[tuple[int, int, int, int, int]] = []
    for degree, boundary in enumerate(complex.boundaries, start=1):
        lower_filtration = np.asarray(complex.filtration[degree - 1])
        upper_filtration = np.asarray(complex.filtration[degree])
        lower_order = np.lexsort((np.arange(lower_filtration.size), lower_filtration))
        upper_order = np.lexsort((np.arange(upper_filtration.size), upper_filtration))
        reduced = (
            np.asarray(boundary, dtype=np.int64)[np.ix_(lower_order, upper_order)].copy()
            % modulus
        )
        pivot_columns: dict[int, int] = {}
        for column in range(reduced.shape[1]):
            while True:
                nonzero = np.flatnonzero(reduced[:, column])
                if nonzero.size == 0:
                    break
                pivot = int(nonzero[-1])
                owner = pivot_columns.get(pivot)
                if owner is None:
                    pivot_columns[pivot] = column
                    lower = int(lower_order[pivot])
                    upper = int(upper_order[column])
                    lower_level = int(lower_filtration[lower])
                    upper_level = int(upper_filtration[upper])
                    pairs.append(
                        (
                            upper_level - lower_level,
                            lower_level,
                            degree - 1,
                            upper_level,
                            degree,
                        )
                    )
                    break
                factor = (
                    int(reduced[pivot, column])
                    * pow(int(reduced[pivot, owner]), modulus - 2, modulus)
                    % modulus
                )
                reduced[:, column] = (
                    reduced[:, column] - factor * reduced[:, owner]
                ) % modulus
    return pairs


def compute_spectral_sequence(
    complex: FilteredChainComplex, /, *, maximum_page: int
) -> SpectralSequenceResult:
    if int(maximum_page) < 1:
        raise ValueError("maximum_page must be positive.")
    maximum_page_ = int(maximum_page)
    levels = sorted(
        set(
            np.concatenate(
                tuple(np.asarray(value) for value in complex.filtration)
            ).tolist()
        )
    )
    level_indices = {level: index for index, level in enumerate(levels)}
    e0 = np.zeros((len(levels), len(complex.filtration)), dtype=np.int32)
    for degree, filtration in enumerate(complex.filtration):
        values, counts = np.unique(np.asarray(filtration), return_counts=True)
        for level, count in zip(values.tolist(), counts.tolist(), strict=True):
            e0[level_indices[int(level)], degree] = int(count)

    pairs = _spectral_pairs(complex)
    pages = []
    differential_ranks = []
    for page in range(maximum_page_ + 1):
        dimensions = e0.copy()
        ranks = np.zeros_like(e0)
        for gap, lower_level, lower_degree, upper_level, upper_degree in pairs:
            if gap < page:
                dimensions[level_indices[lower_level], lower_degree] -= 1
                dimensions[level_indices[upper_level], upper_degree] -= 1
            if gap == page:
                ranks[level_indices[upper_level], upper_degree] += 1
        if np.any(dimensions < 0):
            raise RuntimeError(
                "Filtered reduction produced inconsistent spectral-page dimensions."
            )
        pages.append(dimensions)
        differential_ranks.append(ranks)

    stabilized_page = max((pair[0] for pair in pairs), default=-1) + 1
    convergence_certified = stabilized_page <= maximum_page_
    return SpectralSequenceResult(
        np.stack(pages),
        np.stack(differential_ranks),
        stabilized_page if convergence_certified else -1,
        convergence_certified,
    )


__all__ = [
    "CellDiagonalApproximation",
    "CellularSheaf",
    "FilteredBicomplex",
    "FilteredChainComplex",
    "FinitePersistenceModule",
    "MultiFiltration",
    "MultiparameterPersistenceResult",
    "PointCloudComplexPolicy",
    "PointCloudComplexResult",
    "SpectralSequenceResult",
    "ZigzagIntervalResult",
    "alpha_complex",
    "cech_complex",
    "compute_multiparameter_persistence",
    "compute_spectral_sequence",
    "compute_zigzag_intervals",
    "cup_product",
    "vietoris_rips_complex",
]
