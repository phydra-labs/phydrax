#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._bvh import beam_select_leaf_items, build_packed_bvh, PackedBVH
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


def _unit_facet_normal(coordinates: Array) -> Array:
    dimension = coordinates.shape[-1]
    if dimension == 2:
        tangent = coordinates[..., 1, :] - coordinates[..., 0, :]
        normal = jnp.stack((-tangent[..., 1], tangent[..., 0]), axis=-1)
    else:
        first = coordinates[..., 1, :] - coordinates[..., 0, :]
        second = coordinates[..., 2, :] - coordinates[..., 0, :]
        normal = jnp.cross(first, second)
    norm = jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return normal / norm


def _closest_segment(
    point: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tangent = vertices[1] - vertices[0]
    parameter = float(np.dot(point - vertices[0], tangent) / np.dot(tangent, tangent))
    parameter = min(max(parameter, 0.0), 1.0)
    shape = np.asarray((1.0 - parameter, parameter), dtype=vertices.dtype)
    return shape @ vertices, shape


def _closest_triangle(
    point: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Closest-point regions from Real-Time Collision Detection, Christer Ericson.
    a, b, c = vertices
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        shape = np.asarray((1.0, 0.0, 0.0), dtype=vertices.dtype)
        return a, shape

    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        shape = np.asarray((0.0, 1.0, 0.0), dtype=vertices.dtype)
        return b, shape

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        parameter = d1 / (d1 - d3)
        shape = np.asarray((1.0 - parameter, parameter, 0.0), dtype=vertices.dtype)
        return shape @ vertices, shape

    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        shape = np.asarray((0.0, 0.0, 1.0), dtype=vertices.dtype)
        return c, shape

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        parameter = d2 / (d2 - d6)
        shape = np.asarray((1.0 - parameter, 0.0, parameter), dtype=vertices.dtype)
        return shape @ vertices, shape

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        parameter = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        shape = np.asarray((0.0, 1.0 - parameter, parameter), dtype=vertices.dtype)
        return shape @ vertices, shape

    denominator = 1.0 / (va + vb + vc)
    second = vb * denominator
    third = vc * denominator
    shape = np.asarray((1.0 - second - third, second, third), dtype=vertices.dtype)
    return shape @ vertices, shape


def _closest_point(
    point: np.ndarray, vertices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if point.shape[0] == 2:
        return _closest_segment(point, vertices)
    return _closest_triangle(point, vertices)


class ContactSurface(StrictModule, NonTrainableState):
    """One oriented contact surface represented in the current geometry."""

    node_ids: Array
    current_coordinates: Array
    facets: Array
    facet_ids: Array
    nodal_weights: Array
    surface_id: str = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_id: str,
        node_ids: ArrayLike,
        current_coordinates: ArrayLike,
        facets: ArrayLike,
        facet_ids: ArrayLike,
        /,
        *,
        nodal_weights: ArrayLike | None = None,
    ):
        identifier = str(surface_id)
        nodes = np.asarray(node_ids, dtype=np.int64)
        coordinates = np.asarray(current_coordinates)
        connectivity = np.asarray(facets, dtype=np.int32)
        facet_identifiers = np.asarray(facet_ids, dtype=np.int64)
        if not identifier:
            raise ValueError("Contact surface_id must be nonempty.")
        if coordinates.ndim != 2 or coordinates.shape[1] not in (2, 3):
            raise ValueError("Contact coordinates require shape (n, 2) or (n, 3).")
        dimension = int(coordinates.shape[1])
        if (
            nodes.shape != (coordinates.shape[0],)
            or nodes.size == 0
            or np.any(nodes < 0)
            or np.unique(nodes).size != nodes.size
            or not np.issubdtype(coordinates.dtype, np.inexact)
            or np.any(~np.isfinite(coordinates))
        ):
            raise ValueError("Contact surface nodes and current coordinates are invalid.")
        if (
            connectivity.ndim != 2
            or connectivity.shape[1] != dimension
            or connectivity.shape[0] == 0
            or facet_identifiers.shape != (connectivity.shape[0],)
            or np.any(facet_identifiers < 0)
            or np.unique(facet_identifiers).size != facet_identifiers.size
            or np.any(connectivity < 0)
            or np.any(connectivity >= coordinates.shape[0])
            or any(np.unique(row).size != dimension for row in connectivity)
        ):
            raise ValueError(
                "Contact facets require unique stable IDs and valid local nodes."
            )
        facet_coordinates = coordinates[connectivity]
        if dimension == 2:
            measure = np.linalg.norm(
                facet_coordinates[:, 1] - facet_coordinates[:, 0], axis=1
            )
        else:
            measure = np.linalg.norm(
                np.cross(
                    facet_coordinates[:, 1] - facet_coordinates[:, 0],
                    facet_coordinates[:, 2] - facet_coordinates[:, 0],
                ),
                axis=1,
            )
        if np.any(~np.isfinite(measure)) or np.any(measure <= 0.0):
            raise ValueError("Contact facets must be finite and nondegenerate.")
        weights = (
            np.ones((coordinates.shape[0],), dtype=coordinates.dtype)
            if nodal_weights is None
            else np.asarray(nodal_weights)
        )
        if (
            weights.shape != nodes.shape
            or not np.issubdtype(weights.dtype, np.inexact)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("Contact nodal weights must be positive finite scalars.")
        self.node_ids = jnp.asarray(nodes)
        self.current_coordinates = jnp.asarray(coordinates)
        self.facets = jnp.asarray(connectivity)
        self.facet_ids = jnp.asarray(facet_identifiers)
        self.nodal_weights = jnp.asarray(weights)
        self.surface_id = identifier
        self.dimension = dimension
        self.geometry_id = canonical_fingerprint(
            {
                "kind": "contact-surface-current-geometry",
                "surface": identifier,
                "nodes": array_tree_fingerprint(nodes),
                "coordinates": array_tree_fingerprint(coordinates),
                "facets": array_tree_fingerprint(connectivity),
                "facet_ids": array_tree_fingerprint(facet_identifiers),
                "weights": array_tree_fingerprint(weights),
            }
        )

    def with_current_coordinates(self, coordinates: ArrayLike, /) -> ContactSurface:
        return ContactSurface(
            self.surface_id,
            self.node_ids,
            coordinates,
            self.facets,
            self.facet_ids,
            nodal_weights=self.nodal_weights,
        )


class ContactConfiguration(StrictModule, NonTrainableState):
    """Plus/minus contact configuration at one immutable search epoch."""

    plus: ContactSurface
    minus: ContactSurface
    excluded_node_facet_pairs: Array
    epoch: int = eqx.field(static=True)
    search_radius: float = eqx.field(static=True)
    self_contact: bool = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)

    def __init__(
        self,
        plus: ContactSurface,
        minus: ContactSurface,
        /,
        *,
        epoch: int,
        search_radius: float = math.inf,
        self_contact: bool = False,
        excluded_node_facet_pairs: ArrayLike | None = None,
    ):
        if not isinstance(plus, ContactSurface) or not isinstance(minus, ContactSurface):
            raise TypeError(
                "ContactConfiguration requires plus and minus ContactSurface values."
            )
        epoch_ = int(epoch)
        radius = float(search_radius)
        self_contact_ = bool(self_contact)
        if plus.dimension != minus.dimension or epoch_ < 0:
            raise ValueError(
                "Contact surfaces must share a dimension and epoch must be nonnegative."
            )
        if math.isnan(radius) or radius <= 0.0:
            raise ValueError("Contact search_radius must be positive or infinite.")
        same_surface = plus.surface_id == minus.surface_id
        if same_surface != self_contact_:
            raise ValueError(
                "Equal surface IDs require self_contact=True; self contact requires equal IDs."
            )
        exclusions = (
            np.empty((0, 2), dtype=np.int64)
            if excluded_node_facet_pairs is None
            else np.asarray(excluded_node_facet_pairs, dtype=np.int64)
        )
        if exclusions.ndim != 2 or exclusions.shape[1:] != (2,) or np.any(exclusions < 0):
            raise ValueError(
                "Self-contact exclusions require shape (n, 2) stable ID pairs."
            )
        if (
            exclusions.shape[0]
            and np.unique(exclusions, axis=0).shape[0] != exclusions.shape[0]
        ):
            raise ValueError("Self-contact exclusion pairs must be unique.")
        if not set(exclusions[:, 0].tolist()).issubset(
            set(np.asarray(plus.node_ids).tolist())
        ):
            raise ValueError("Excluded plus node IDs must belong to the plus surface.")
        if not set(exclusions[:, 1].tolist()).issubset(
            set(np.asarray(minus.facet_ids).tolist())
        ):
            raise ValueError("Excluded minus facet IDs must belong to the minus surface.")
        self.plus = plus
        self.minus = minus
        self.excluded_node_facet_pairs = jnp.asarray(exclusions)
        self.epoch = epoch_
        self.search_radius = radius
        self.self_contact = self_contact_
        self.configuration_id = canonical_fingerprint(
            {
                "kind": "contact-configuration",
                "plus": plus.geometry_id,
                "minus": minus.geometry_id,
                "epoch": epoch_,
                "search_radius": "infinite" if math.isinf(radius) else radius,
                "self_contact": self_contact_,
                "exclusions": array_tree_fingerprint(exclusions),
            }
        )

    def next_epoch(
        self,
        plus: ContactSurface,
        minus: ContactSurface,
        /,
        *,
        search_radius: float | None = None,
        excluded_node_facet_pairs: ArrayLike | None = None,
    ) -> ContactConfiguration:
        """Advance current geometry without changing contact identities or policy."""
        if (
            plus.surface_id != self.plus.surface_id
            or minus.surface_id != self.minus.surface_id
        ):
            raise ValueError("A contact epoch cannot change its surface identities.")
        exclusions = (
            self.excluded_node_facet_pairs
            if excluded_node_facet_pairs is None
            else excluded_node_facet_pairs
        )
        return ContactConfiguration(
            plus,
            minus,
            epoch=self.epoch + 1,
            search_radius=(
                self.search_radius if search_radius is None else float(search_radius)
            ),
            self_contact=self.self_contact,
            excluded_node_facet_pairs=exclusions,
        )


class ContactPatch(StrictModule, NonTrainableState):
    """One stable plus-node/minus-facet contact patch."""

    minus_shape_values: Array
    closest_point: Array
    normal: Array
    gap: Array
    weight: Array
    pair_id: str = eqx.field(static=True)
    plus_node_index: int = eqx.field(static=True)
    minus_facet_index: int = eqx.field(static=True)
    plus_node_id: int = eqx.field(static=True)
    minus_facet_id: int = eqx.field(static=True)
    epoch: int = eqx.field(static=True)


class ContactPatchSet(StrictModule, NonTrainableState):
    """Vectorized patches frozen for one deterministic search epoch."""

    plus_node_indices: Array
    minus_facet_indices: Array
    minus_shape_values: Array
    closest_points: Array
    normals: Array
    gaps: Array
    weights: Array
    pair_ids: tuple[str, ...] = eqx.field(static=True)
    plus_node_ids: tuple[int, ...] = eqx.field(static=True)
    minus_facet_ids: tuple[int, ...] = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    patch_set_id: str = eqx.field(static=True)

    def __init__(
        self,
        pair_ids: tuple[str, ...],
        plus_node_indices: ArrayLike,
        minus_facet_indices: ArrayLike,
        plus_node_ids: tuple[int, ...],
        minus_facet_ids: tuple[int, ...],
        minus_shape_values: ArrayLike,
        closest_points: ArrayLike,
        normals: ArrayLike,
        gaps: ArrayLike,
        weights: ArrayLike,
        /,
        *,
        epoch: int,
        dimension: int,
    ):
        pairs = tuple(str(value) for value in pair_ids)
        plus_indices = np.asarray(plus_node_indices, dtype=np.int32)
        minus_indices = np.asarray(minus_facet_indices, dtype=np.int32)
        plus_ids = tuple(int(value) for value in plus_node_ids)
        minus_ids = tuple(int(value) for value in minus_facet_ids)
        shape = np.asarray(minus_shape_values)
        closest = np.asarray(closest_points)
        normal = np.asarray(normals)
        gap = np.asarray(gaps)
        weight = np.asarray(weights)
        count = len(pairs)
        dimension_ = int(dimension)
        epoch_ = int(epoch)
        if (
            len(set(pairs)) != count
            or len(plus_ids) != count
            or len(minus_ids) != count
            or plus_indices.shape != (count,)
            or minus_indices.shape != (count,)
            or shape.shape != (count, dimension_)
            or closest.shape != (count, dimension_)
            or normal.shape != (count, dimension_)
            or gap.shape != (count,)
            or weight.shape != (count,)
            or dimension_ not in (2, 3)
            or epoch_ < 0
            or not np.issubdtype(shape.dtype, np.inexact)
            or not np.issubdtype(closest.dtype, np.inexact)
            or not np.issubdtype(normal.dtype, np.inexact)
            or not np.issubdtype(gap.dtype, np.inexact)
            or not np.issubdtype(weight.dtype, np.inexact)
        ):
            raise ValueError("Contact patch arrays are inconsistent.")
        tolerance = 128.0 * max(
            np.finfo(shape.dtype).eps,
            np.finfo(normal.dtype).eps,
        )
        if count and (
            np.any(~np.isfinite(shape))
            or np.any(~np.isfinite(closest))
            or np.any(~np.isfinite(normal))
            or np.any(~np.isfinite(gap))
            or np.any(~np.isfinite(weight))
            or np.any(weight <= 0.0)
            or np.any(np.abs(np.sum(shape, axis=1) - 1.0) > tolerance)
            or np.any(np.abs(np.linalg.norm(normal, axis=1) - 1.0) > tolerance)
        ):
            raise ValueError(
                "Contact patch geometry must be finite, normalized, and affine."
            )
        self.plus_node_indices = jnp.asarray(plus_indices)
        self.minus_facet_indices = jnp.asarray(minus_indices)
        self.minus_shape_values = jnp.asarray(shape)
        self.closest_points = jnp.asarray(closest)
        self.normals = jnp.asarray(normal)
        self.gaps = jnp.asarray(gap)
        self.weights = jnp.asarray(weight)
        self.pair_ids = pairs
        self.plus_node_ids = plus_ids
        self.minus_facet_ids = minus_ids
        self.epoch = epoch_
        self.dimension = dimension_
        self.patch_set_id = canonical_fingerprint(
            {
                "kind": "contact-patch-set",
                "pair_ids": list(pairs),
                "plus_indices": array_tree_fingerprint(plus_indices),
                "minus_indices": array_tree_fingerprint(minus_indices),
                "shape": array_tree_fingerprint(shape),
                "epoch": epoch_,
            }
        )

    def __len__(self) -> int:
        return len(self.pair_ids)

    def patch(self, index: int, /) -> ContactPatch:
        position = int(index)
        if position < 0 or position >= len(self):
            raise IndexError("Contact patch index is out of range.")
        return ContactPatch(
            minus_shape_values=self.minus_shape_values[position],
            closest_point=self.closest_points[position],
            normal=self.normals[position],
            gap=self.gaps[position],
            weight=self.weights[position],
            pair_id=self.pair_ids[position],
            plus_node_index=int(self.plus_node_indices[position]),
            minus_facet_index=int(self.minus_facet_indices[position]),
            plus_node_id=self.plus_node_ids[position],
            minus_facet_id=self.minus_facet_ids[position],
            epoch=self.epoch,
        )


class ContactQueryResult(StrictModule, NonTrainableState):
    """Deterministic closest-facet result and its exclusion evidence."""

    configuration: ContactConfiguration
    patches: ContactPatchSet
    candidate_count: Array
    excluded_count: Array
    epoch: int = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    def __init__(
        self,
        configuration: ContactConfiguration,
        patches: ContactPatchSet,
        candidate_count: ArrayLike,
        excluded_count: ArrayLike,
        /,
    ):
        if not isinstance(configuration, ContactConfiguration) or not isinstance(
            patches, ContactPatchSet
        ):
            raise TypeError("Contact query requires a configuration and patch set.")
        candidates = jnp.asarray(candidate_count, dtype=jnp.int32)
        excluded = jnp.asarray(excluded_count, dtype=jnp.int32)
        if (
            patches.epoch != configuration.epoch
            or candidates.shape != ()
            or excluded.shape != ()
            or int(candidates) < len(patches)
            or int(excluded) < 0
        ):
            raise ValueError("Contact query epoch or search evidence is inconsistent.")
        self.configuration = configuration
        self.patches = patches
        self.candidate_count = candidates
        self.excluded_count = excluded
        self.epoch = configuration.epoch
        self.query_id = canonical_fingerprint(
            {
                "kind": "contact-query-result",
                "configuration": configuration.configuration_id,
                "patches": patches.patch_set_id,
                "candidates": int(candidates),
                "excluded": int(excluded),
            }
        )

    def current_kinematics(
        self,
        plus_coordinates: ArrayLike | None = None,
        minus_coordinates: ArrayLike | None = None,
        /,
    ) -> tuple[Array, Array, Array]:
        plus = (
            self.configuration.plus.current_coordinates
            if plus_coordinates is None
            else jnp.asarray(plus_coordinates)
        )
        minus = (
            self.configuration.minus.current_coordinates
            if minus_coordinates is None
            else jnp.asarray(minus_coordinates)
        )
        if (
            plus.shape != self.configuration.plus.current_coordinates.shape
            or minus.shape != self.configuration.minus.current_coordinates.shape
        ):
            raise ValueError(
                "Fixed-epoch contact coordinates must preserve surface layouts."
            )
        facet_nodes = self.configuration.minus.facets[self.patches.minus_facet_indices]
        facet_coordinates = minus[facet_nodes]
        normal = _unit_facet_normal(facet_coordinates)
        closest = jnp.sum(
            self.patches.minus_shape_values[..., None] * facet_coordinates,
            axis=1,
        )
        point = plus[self.patches.plus_node_indices]
        gap = jnp.sum((point - closest) * normal, axis=-1)
        return gap, normal, closest


class ContactQueryPlan(StrictModule, NonTrainableState):
    """Exact deterministic current-geometry query accelerated by a packed BVH."""

    configuration: ContactConfiguration
    bvh: PackedBVH
    facet_order: Array
    bbox_min: Array
    bbox_max: Array
    leaf_size: int = eqx.field(static=True)
    beam_width: int = eqx.field(static=True)
    epoch: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        configuration: ContactConfiguration,
        /,
        *,
        leaf_size: int = 8,
        beam_width: int = 4,
    ):
        if not isinstance(configuration, ContactConfiguration):
            raise TypeError("ContactQueryPlan requires ContactConfiguration.")
        leaf = int(leaf_size)
        beam = int(beam_width)
        if leaf <= 0 or beam <= 0:
            raise ValueError("Contact query leaf_size and beam_width must be positive.")
        facet_ids = np.asarray(configuration.minus.facet_ids)
        order = np.argsort(facet_ids, kind="stable").astype(np.int32)
        coordinates = np.asarray(configuration.minus.current_coordinates)
        facets = np.asarray(configuration.minus.facets)[order]
        facet_coordinates = coordinates[facets]
        minimum = np.min(facet_coordinates, axis=1)
        maximum = np.max(facet_coordinates, axis=1)
        self.configuration = configuration
        self.bvh = build_packed_bvh(
            minimum,
            maximum,
            leaf_size=leaf,
            dtype=configuration.minus.current_coordinates.dtype,
        )
        self.facet_order = jnp.asarray(order)
        self.bbox_min = jnp.asarray(minimum)
        self.bbox_max = jnp.asarray(maximum)
        self.leaf_size = leaf
        self.beam_width = beam
        self.epoch = configuration.epoch
        self.plan_id = canonical_fingerprint(
            {
                "kind": "contact-query-plan",
                "configuration": configuration.configuration_id,
                "leaf_size": leaf,
                "beam_width": beam,
                "facet_order": array_tree_fingerprint(order),
            }
        )

    def execute(self) -> ContactQueryResult:
        configuration = self.configuration
        plus = np.asarray(configuration.plus.current_coordinates)
        minus = np.asarray(configuration.minus.current_coordinates)
        minus_facets = np.asarray(configuration.minus.facets)
        minus_facet_ids = np.asarray(configuration.minus.facet_ids)
        plus_node_ids = np.asarray(configuration.plus.node_ids)
        candidates, valid = beam_select_leaf_items(
            configuration.plus.current_coordinates,
            bvh=self.bvh,
            beam_width=self.beam_width,
            steps=self.bvh.max_depth + 1,
        )
        candidate_rows = np.asarray(candidates)
        valid_rows = np.asarray(valid)
        order = np.asarray(self.facet_order)
        bbox_min = np.asarray(self.bbox_min)
        bbox_max = np.asarray(self.bbox_max)
        explicit_exclusions = {
            (int(node), int(facet))
            for node, facet in np.asarray(configuration.excluded_node_facet_pairs)
        }
        same_surface = configuration.self_contact

        pair_ids: list[str] = []
        plus_indices: list[int] = []
        minus_indices: list[int] = []
        selected_plus_ids: list[int] = []
        selected_minus_ids: list[int] = []
        shapes: list[np.ndarray] = []
        closest_points: list[np.ndarray] = []
        normals: list[np.ndarray] = []
        gaps: list[float] = []
        weights: list[float] = []
        evaluated_count = 0
        excluded_count = 0

        for plus_index, point in enumerate(plus):
            plus_id = int(plus_node_ids[plus_index])
            seed_sorted = candidate_rows[plus_index, valid_rows[plus_index]]
            seed_actual = order[seed_sorted]
            preliminary = math.inf
            for facet_index in seed_actual:
                facet_index_ = int(facet_index)
                facet_id = int(minus_facet_ids[facet_index_])
                incident = same_surface and plus_id in set(
                    np.asarray(configuration.minus.node_ids)[
                        minus_facets[facet_index_]
                    ].tolist()
                )
                if incident or (plus_id, facet_id) in explicit_exclusions:
                    continue
                closest, _ = _closest_point(point, minus[minus_facets[facet_index_]])
                preliminary = min(
                    preliminary, float(np.dot(point - closest, point - closest))
                )
            delta = np.maximum(0.0, np.maximum(bbox_min - point, point - bbox_max))
            lower_bound = np.sum(delta * delta, axis=1)
            exact_sorted = np.flatnonzero(lower_bound <= preliminary + 1.0e-14)
            exact_actual = order[exact_sorted]
            local: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
            for facet_index in exact_actual:
                facet_index_ = int(facet_index)
                facet_id = int(minus_facet_ids[facet_index_])
                facet_node_ids = np.asarray(configuration.minus.node_ids)[
                    minus_facets[facet_index_]
                ]
                incident = same_surface and plus_id in set(facet_node_ids.tolist())
                if incident or (plus_id, facet_id) in explicit_exclusions:
                    excluded_count += 1
                    continue
                closest, shape = _closest_point(point, minus[minus_facets[facet_index_]])
                distance_squared = float(np.dot(point - closest, point - closest))
                local.append((distance_squared, facet_id, facet_index_, closest, shape))
                evaluated_count += 1
            if not local:
                continue
            local.sort(key=lambda item: (item[0], item[1]))
            distance_squared, facet_id, facet_index, closest, shape = local[0]
            if math.sqrt(distance_squared) > configuration.search_radius:
                continue
            facet_coordinates = minus[minus_facets[facet_index]]
            if configuration.minus.dimension == 2:
                tangent = facet_coordinates[1] - facet_coordinates[0]
                normal = np.asarray((-tangent[1], tangent[0]))
            else:
                normal = np.cross(
                    facet_coordinates[1] - facet_coordinates[0],
                    facet_coordinates[2] - facet_coordinates[0],
                )
            normal = normal / np.linalg.norm(normal)
            gap = float(np.dot(point - closest, normal))
            pair_id = canonical_fingerprint(
                {
                    "kind": "contact-pair",
                    "plus_surface": configuration.plus.surface_id,
                    "plus_node": plus_id,
                    "minus_surface": configuration.minus.surface_id,
                    "minus_facet": facet_id,
                }
            )
            pair_ids.append(pair_id)
            plus_indices.append(plus_index)
            minus_indices.append(facet_index)
            selected_plus_ids.append(plus_id)
            selected_minus_ids.append(facet_id)
            shapes.append(shape)
            closest_points.append(closest)
            normals.append(normal)
            gaps.append(gap)
            weights.append(float(configuration.plus.nodal_weights[plus_index]))

        dimension = configuration.plus.dimension
        patch_count = len(pair_ids)
        patch_set = ContactPatchSet(
            tuple(pair_ids),
            np.asarray(plus_indices, dtype=np.int32),
            np.asarray(minus_indices, dtype=np.int32),
            tuple(selected_plus_ids),
            tuple(selected_minus_ids),
            np.asarray(shapes, dtype=plus.dtype).reshape((patch_count, dimension)),
            np.asarray(closest_points, dtype=plus.dtype).reshape(
                (patch_count, dimension)
            ),
            np.asarray(normals, dtype=plus.dtype).reshape((patch_count, dimension)),
            np.asarray(gaps, dtype=plus.dtype),
            np.asarray(weights, dtype=plus.dtype),
            epoch=configuration.epoch,
            dimension=dimension,
        )
        return ContactQueryResult(
            configuration,
            patch_set,
            jnp.asarray(evaluated_count, dtype=jnp.int32),
            jnp.asarray(excluded_count, dtype=jnp.int32),
        )


__all__ = [
    "ContactConfiguration",
    "ContactPatch",
    "ContactPatchSet",
    "ContactQueryPlan",
    "ContactQueryResult",
    "ContactSurface",
]
