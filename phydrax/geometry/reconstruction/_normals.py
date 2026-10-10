#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded-neighborhood normal estimation and consistent orientation.

Neighborhoods are the exact ``k`` nearest samples from the packed BVH owner.
Unoriented directions are the least-variance principal axes of each
neighborhood covariance (native Hermitian spectra). Orientation follows Hoppe
et al. (1992): a minimum spanning forest of the symmetrized neighbor graph under
the weight ``1 - |n_i . n_j|`` propagates relative signs from one seed per
connected component, so sign decisions travel along the most parallel pairs
first. The forest and its level-synchronous propagation are host topology
preparation; every ambiguous direction, weak propagation edge, and residual
neighbor conflict is counted in :class:`NormalEstimationEvidence`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never, Literal, TypeAlias

import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from ... import ein
from ..._bvh import bvh_nearest_items, BVHBuildPolicy, prepare_bvh
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_integer
from ...linalg import HermitianSpectrum
from ...typing import parse


NormalOrientation: TypeAlias = Literal["propagate", "supplied"]
"""``propagate`` orients directions along a minimum spanning forest of the
neighbor graph; ``supplied`` keeps the orientation of caller-supplied normals."""

_NEIGHBOR_POLICY = BVHBuildPolicy(leaf_size=16)
_QUERY_BATCH_CAPACITY = 256


@dataclass(frozen=True, slots=True)
class NormalEstimationEvidence:
    """Neighborhood, direction, and orientation evidence of point normals.

    ``ambiguous_direction_count`` counts estimated neighborhoods whose two
    smallest covariance eigenvalues are separated by at most
    ``direction_tolerance`` times the covariance trace (the least-variance axis
    is not determined). ``ambiguous_orientation_edges`` counts spanning-forest
    edges whose endpoint directions satisfy ``|n_i . n_j| < orientation_tolerance``
    (the propagated relative sign is unreliable) and
    ``conflicting_neighbor_edges`` counts neighbor-graph edges with
    ``n_i . n_j < -orientation_tolerance`` after orientation. ``reoriented_count``
    counts normals whose sign differs from the estimated or supplied direction.
    """

    route: str
    orientation: NormalOrientation
    point_count: int
    neighborhood_size: int
    graph_edges: int
    graph_components: int
    ambiguous_direction_count: int
    ambiguous_orientation_edges: int
    conflicting_neighbor_edges: int
    reoriented_count: int
    maximum_surface_variation: float
    direction_tolerance: float
    orientation_tolerance: float


class PointNormals(StrictModule, NonTrainableState):
    """Consistently oriented unit normals with their exact bounded neighborhoods.

    ``neighbor_indices[i]`` lists the ``neighborhood_size`` nearest other samples
    of point ``i`` ordered by (distance, index), ``neighbor_distances`` their
    Euclidean distances, ``surface_variation`` the PCA ratio
    ``lambda_min / trace`` of each neighborhood covariance (zero for supplied
    directions), and ``components`` the minimum sample index of each sample's
    connected component of the symmetrized neighbor graph.
    """

    points: np.ndarray
    normals: np.ndarray
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    surface_variation: np.ndarray
    components: np.ndarray
    evidence: NormalEstimationEvidence

    def __init__(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        neighbor_indices: np.ndarray,
        neighbor_distances: np.ndarray,
        surface_variation: np.ndarray,
        components: np.ndarray,
        evidence: NormalEstimationEvidence,
    ) -> None:
        if not isinstance(evidence, NormalEstimationEvidence):
            raise TypeError("evidence must be NormalEstimationEvidence.")
        count = points.shape[0]
        if (
            normals.shape != (count, 3)
            or surface_variation.shape != (count,)
            or components.shape != (count,)
        ):
            raise ValueError(
                "normals, surface_variation, and components must follow the points."
            )
        if neighbor_indices.shape != neighbor_distances.shape or (
            neighbor_indices.shape[0] != count
        ):
            raise ValueError("Neighbor arrays must have shape (num_points, k).")
        self.points = _frozen(points)
        self.normals = _frozen(normals)
        self.neighbor_indices = _frozen(neighbor_indices)
        self.neighbor_distances = _frozen(neighbor_distances)
        self.surface_variation = _frozen(surface_variation)
        self.components = _frozen(components)
        self.evidence = evidence


def _frozen(array: np.ndarray, /) -> np.ndarray:
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _validated_points(points: ArrayLike, /) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (num_points, 3).")
    if not np.all(np.isfinite(values)):
        raise ValueError("points must be finite.")
    return values


def _validated_directions(normals: ArrayLike, count: int, /) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float64)
    if values.shape != (count, 3):
        raise ValueError("normals must have shape (num_points, 3).")
    if not np.all(np.isfinite(values)):
        raise ValueError("normals must be finite.")
    lengths = np.linalg.norm(values, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("normals must be nonzero.")
    return values / lengths[:, None]


def _tolerance(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0 or result >= 1.0:
        raise ValueError(f"{name} must lie in [0, 1).")
    return result


def nearest_neighbors(
    points: np.ndarray, neighborhood_size: int, /
) -> tuple[np.ndarray, np.ndarray]:
    """Exact nearest other samples per point, ordered by (distance, index).

    Coincident samples are neighbors at distance zero; the query point itself is
    removed by index, not by distance.
    """

    count = points.shape[0]
    hierarchy = prepare_bvh(points, points, policy=_NEIGHBOR_POLICY, dtype=jnp.float64)
    nearest = bvh_nearest_items(
        hierarchy,
        points,
        k=neighborhood_size + 1,
        query_batch_capacity=_QUERY_BATCH_CAPACITY,
    )
    items = np.asarray(nearest.items, dtype=np.int64)
    squared = np.asarray(nearest.distance_squared, dtype=np.float64)
    others = items != np.arange(count, dtype=np.int64)[:, None]
    order = np.argsort(~others, axis=1, kind="stable")[:, :neighborhood_size]
    indices = np.take_along_axis(items, order, axis=1)
    distances = np.sqrt(np.take_along_axis(squared, order, axis=1))
    return indices.astype(np.int32), distances


def _principal_directions(
    points: np.ndarray, neighbors: np.ndarray, direction_tolerance: float, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-variance axes, surface variation, and direction ambiguity masks."""

    patches = np.concatenate((points[:, None, :], points[neighbors]), axis=1)
    centered = patches - np.mean(patches, axis=1, keepdims=True)
    covariance = np.asarray(ein.contract("nki,nkj->nij", centered, centered)) / float(
        patches.shape[1]
    )
    spectrum = HermitianSpectrum(jnp.asarray(covariance))
    if not bool(np.all(np.asarray(spectrum.valid))):
        raise ValueError("Neighborhood covariance spectra are not valid.")
    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    directions = np.asarray(spectrum.eigenvectors, dtype=np.float64)[:, :, 0]
    trace = np.sum(eigenvalues, axis=1)
    variation = np.where(
        trace > 0.0, eigenvalues[:, 0] / np.where(trace > 0.0, trace, 1.0), 1.0
    )
    ambiguous = (trace <= 0.0) | (
        eigenvalues[:, 1] - eigenvalues[:, 0] <= direction_tolerance * trace
    )
    return directions, variation, ambiguous


def _neighbor_edges(neighbors: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrized neighbor-graph edges ``first < second`` in lexicographic order."""

    count, width = neighbors.shape
    first = np.repeat(np.arange(count, dtype=np.int64), width)
    second = neighbors.reshape((-1,)).astype(np.int64)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    keys = np.unique(low[low != high] * count + high[low != high])
    return keys // count, keys % count


def _merge_labels(
    labels: np.ndarray, first: np.ndarray, second: np.ndarray, /
) -> np.ndarray:
    """Merge representative labels joined by edges into their minimum label."""

    parent = labels.copy()
    while True:
        low = np.minimum(parent[first], parent[second])
        np.minimum.at(parent, parent[first], low)
        np.minimum.at(parent, parent[second], low)
        while True:
            jumped = parent[parent]
            if np.array_equal(jumped, parent):
                break
            parent = jumped
        if np.array_equal(parent[first], parent[second]):
            return parent[labels]


def minimum_spanning_forest(
    count: int, first: np.ndarray, second: np.ndarray, weight: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    """Borůvka minimum spanning forest under the total order (weight, edge index).

    Returns the selected-edge mask and the minimum vertex index of every
    vertex's connected component. The strict total order makes the forest
    unique and cycle free.
    """

    edge_count = first.shape[0]
    by_rank = np.lexsort((np.arange(edge_count, dtype=np.int64), weight))
    rank = np.empty((edge_count,), dtype=np.int64)
    rank[by_rank] = np.arange(edge_count, dtype=np.int64)
    labels = np.arange(count, dtype=np.int64)
    selected = np.zeros((edge_count,), dtype=np.bool_)
    unset = np.iinfo(np.int64).max
    while True:
        first_label = labels[first]
        second_label = labels[second]
        crossing = np.flatnonzero(first_label != second_label)
        if crossing.size == 0:
            return selected, labels
        best = np.full((count,), unset, dtype=np.int64)
        np.minimum.at(best, first_label[crossing], rank[crossing])
        np.minimum.at(best, second_label[crossing], rank[crossing])
        chosen = by_rank[np.unique(best[best != unset])]
        selected[chosen] = True
        labels = _merge_labels(labels, first[chosen], second[chosen])


def _component_centroids(points: np.ndarray, components: np.ndarray, /) -> np.ndarray:
    """Centroid of every sample's connected component, gathered per sample."""

    count = points.shape[0]
    sizes = np.bincount(components, minlength=count).astype(np.float64)
    sums = np.stack(
        [
            np.bincount(components, weights=points[:, axis], minlength=count)
            for axis in range(3)
        ],
        axis=1,
    )
    return (sums / np.maximum(sizes, 1.0)[:, None])[components]


def _component_seeds(points: np.ndarray, components: np.ndarray, /) -> np.ndarray:
    """Per component, the sample farthest from its centroid (lowest index on ties)."""

    count = points.shape[0]
    distance = np.linalg.norm(points - _component_centroids(points, components), axis=1)
    order = np.lexsort((np.arange(count), -distance, components))
    first = np.ones((count,), dtype=np.bool_)
    first[1:] = components[order[1:]] != components[order[:-1]]
    return order[first]


def _propagated_signs(
    directions: np.ndarray,
    tree_first: np.ndarray,
    tree_second: np.ndarray,
    seeds: np.ndarray,
    seed_signs: np.ndarray,
    /,
) -> np.ndarray:
    """Level-synchronous propagation of relative signs over a spanning forest.

    In a forest every unvisited neighbor of the current frontier has exactly one
    visited neighbor, its parent, so each level is one vectorized update.
    """

    count = directions.shape[0]
    sources = np.concatenate((tree_first, tree_second))
    targets = np.concatenate((tree_second, tree_first))
    order = np.argsort(sources, kind="stable")
    sources, targets = sources[order], targets[order]
    offsets = np.searchsorted(sources, np.arange(count + 1, dtype=np.int64))
    signs = np.zeros((count,), dtype=np.float64)
    signs[seeds] = seed_signs
    visited = np.zeros((count,), dtype=np.bool_)
    visited[seeds] = True
    frontier = seeds
    while frontier.size:
        starts = offsets[frontier]
        widths = offsets[frontier + 1] - starts
        parents = np.repeat(frontier, widths)
        block_starts = np.cumsum(widths) - widths
        positions = np.repeat(starts - block_starts, widths) + np.arange(
            parents.size, dtype=np.int64
        )
        children = targets[positions]
        fresh = ~visited[children]
        parents, children = parents[fresh], children[fresh]
        relative = np.where(
            np.sum(directions[parents] * directions[children], axis=1) < 0.0, -1.0, 1.0
        )
        signs[children] = signs[parents] * relative
        visited[children] = True
        frontier = children
    return signs


def _orient(
    points: np.ndarray,
    directions: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    supplied: bool,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagated signs, component labels, and spanning-forest edges.

    The farthest sample from a component centroid has a normal parallel to its
    radius vector on a smooth closed surface, so estimated directions are
    pointed away from the centroid there. Supplied directions instead keep the
    component-majority sign of the caller's orientation.
    """

    count = points.shape[0]
    alignment = np.abs(np.sum(directions[first] * directions[second], axis=1))
    selected, components = minimum_spanning_forest(count, first, second, 1.0 - alignment)
    seeds = _component_seeds(points, components)
    radial = np.sum(
        directions[seeds]
        * (points[seeds] - _component_centroids(points, components)[seeds]),
        axis=1,
    )
    seed_signs = (
        np.ones((seeds.size,), dtype=np.float64)
        if supplied
        else np.where(radial < 0.0, -1.0, 1.0)
    )
    forest_first, forest_second = first[selected], second[selected]
    signs = _propagated_signs(directions, forest_first, forest_second, seeds, seed_signs)
    if supplied:
        agreement = np.bincount(components, weights=signs, minlength=count)
        signs = np.where(agreement[components] < 0.0, -signs, signs)
    return signs, components, forest_first, forest_second


def estimate_point_normals(
    points: ArrayLike,
    /,
    *,
    normals: ArrayLike | None = None,
    orientation: NormalOrientation = "propagate",
    neighborhood_size: int = 16,
    direction_tolerance: float = 1.0e-3,
    orientation_tolerance: float = 0.25,
) -> PointNormals:
    """Estimate and consistently orient unit normals of a 3D point cloud.

    Without ``normals``, each direction is the least-variance principal axis of
    the point's exact ``neighborhood_size``-nearest neighborhood, and
    ``orientation`` must be ``"propagate"``: signs are propagated along a
    minimum spanning forest of the neighbor graph from the sample farthest from
    each component centroid, whose normal is pointed away from that centroid.
    Supplied ``normals`` are normalized; ``"propagate"`` makes their signs
    consistent along the forest while keeping each component's majority
    orientation, and ``"supplied"`` keeps them unchanged. All orientation
    ambiguity is reported, never repaired silently.
    """

    points_ = _validated_points(points)
    orientation_ = parse(orientation, NormalOrientation, "orientation")
    size = positive_integer(neighborhood_size, "neighborhood_size")
    if size >= points_.shape[0]:
        raise ValueError("neighborhood_size must be smaller than the number of points.")
    direction_tolerance_ = _tolerance(direction_tolerance, "direction_tolerance")
    orientation_tolerance_ = _tolerance(orientation_tolerance, "orientation_tolerance")
    count = points_.shape[0]
    supplied = normals is not None
    if orientation_ == "supplied" and not supplied:
        raise ValueError("orientation='supplied' requires normals.")
    neighbors, distances = nearest_neighbors(points_, size)
    first, second = _neighbor_edges(neighbors)
    if normals is None:
        directions, variation, ambiguous = _principal_directions(
            points_, neighbors, direction_tolerance_
        )
        route = "bvh-knn-pca"
    else:
        directions = _validated_directions(normals, count)
        variation = np.zeros((count,), dtype=np.float64)
        ambiguous = np.zeros((count,), dtype=np.bool_)
        route = "bvh-knn-supplied"
    match orientation_:
        case "propagate":
            signs, components, forest_first, forest_second = _orient(
                points_, directions, first, second, supplied
            )
            route = f"{route}-mst-propagation"
            forest_alignment = np.abs(
                np.sum(directions[forest_first] * directions[forest_second], axis=1)
            )
            weak = int(np.count_nonzero(forest_alignment < orientation_tolerance_))
        case "supplied":
            signs = np.ones((count,), dtype=np.float64)
            components = _merge_labels(np.arange(count, dtype=np.int64), first, second)
            weak = 0
        case _:
            assert_never(orientation_)
    oriented = directions * signs[:, None]
    alignment = np.sum(oriented[first] * oriented[second], axis=1)
    evidence = NormalEstimationEvidence(
        route=route,
        orientation=orientation_,
        point_count=count,
        neighborhood_size=size,
        graph_edges=first.shape[0],
        graph_components=np.unique(components).size,
        ambiguous_direction_count=int(np.count_nonzero(ambiguous)),
        ambiguous_orientation_edges=weak,
        conflicting_neighbor_edges=int(
            np.count_nonzero(alignment < -orientation_tolerance_)
        ),
        reoriented_count=int(np.count_nonzero(signs < 0.0)),
        maximum_surface_variation=float(np.max(variation)),
        direction_tolerance=direction_tolerance_,
        orientation_tolerance=orientation_tolerance_,
    )
    return PointNormals(
        points_, oriented, neighbors, distances, variation, components, evidence
    )
