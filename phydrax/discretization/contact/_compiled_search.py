#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..spatial import morton_encode_integer
from ._stencils import ContactStencilKind
from ._surface import PreparedCollisionScene


class CompiledCandidateBatch(StrictModule):
    vertex_indices: Array
    valid: Array
    actual_count: Array
    overflow_count: Array
    kind: ContactStencilKind = eqx.field(static=True)
    capacity: int = eqx.field(static=True)


class CompiledContactSearchEvidence(StrictModule):
    candidate_count: Array
    overflow_count: Array
    finite: Array
    complete: Array
    plan_id: str = eqx.field(static=True)


class CompiledContactSearchResult(StrictModule):
    edge_vertex: CompiledCandidateBatch
    edge_edge: CompiledCandidateBatch
    face_vertex: CompiledCandidateBatch
    evidence: CompiledContactSearchEvidence


class LBVHContactSearchEvidence(StrictModule):
    candidate_count: Array
    node_count: Array
    tree_depth: Array
    traversal_visits: Array
    duplicate_code_count: Array
    stack_overflow: Array
    visit_overflow: Array
    output_overflow: Array
    finite_bounds: Array
    complete: Array
    plan_id: str = eqx.field(static=True)


class LBVHContactSearchResult(StrictModule):
    edge_vertex: CompiledCandidateBatch
    edge_edge: CompiledCandidateBatch
    face_vertex: CompiledCandidateBatch
    evidence: LBVHContactSearchEvidence


class LBVHContactSearchPlan(StrictModule, NonTrainableState):
    """Deterministic fixed-budget device broad phase over Morton-ordered AABBs.

    Primitive AABBs are ordered by Morton code and assembled into a balanced
    binary hierarchy. A depth-first node-pair traversal emits candidates
    directly into fixed-capacity buffers, so neither legal-pair tables nor
    dense overlap matrices are materialized. Search selection is
    stop-gradient topology.
    """

    edges: Array
    faces: Array
    tree_left: Array
    tree_right: Array
    node_leaf_count: Array
    vertex_count: int = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)
    hierarchy_root: int = eqx.field(static=True)
    hierarchy_depth: int = eqx.field(static=True)
    edge_vertex_capacity: int = eqx.field(static=True)
    edge_edge_capacity: int = eqx.field(static=True)
    face_vertex_capacity: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    morton_bits: int = eqx.field(static=True)
    maximum_tree_depth: int = eqx.field(static=True)
    maximum_traversal_visits: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: PreparedCollisionScene,
        /,
        *,
        edge_vertex_capacity: int,
        edge_edge_capacity: int,
        face_vertex_capacity: int,
        activation_distance: float,
        morton_bits: int = 10,
        maximum_tree_depth: int = 64,
        maximum_traversal_visits: int,
    ):
        if not isinstance(scene, PreparedCollisionScene):
            raise TypeError("scene must be PreparedCollisionScene.")
        capacities = (
            int(edge_vertex_capacity),
            int(edge_edge_capacity),
            int(face_vertex_capacity),
        )
        if any(value < 0 for value in capacities) or sum(capacities) <= 0:
            raise ValueError("LBVH output capacities are invalid.")
        activation = float(activation_distance)
        bits = int(morton_bits)
        depth = int(maximum_tree_depth)
        visits = int(maximum_traversal_visits)
        if not np.isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        if bits <= 0 or bits > 10:
            raise ValueError("morton_bits must lie in [1, 10] for uint32 codes.")
        if depth <= 0 or visits <= 0:
            raise ValueError("LBVH depth and traversal visit budgets must be positive.")
        if visits > np.iinfo(np.int32).max:
            raise ValueError(
                "LBVH traversal visit budget exceeds int32 evidence capacity."
            )
        primitive_count = (
            scene.vertex_count + int(scene.edges.shape[0]) + int(scene.faces.shape[0])
        )
        left, right, leaf_count, root, hierarchy_depth = _balanced_morton_hierarchy(
            primitive_count
        )
        self.edges = jnp.asarray(scene.edges, dtype=jnp.int32)
        self.faces = jnp.asarray(scene.faces, dtype=jnp.int32)
        self.tree_left = jnp.asarray(left, dtype=jnp.int32)
        self.tree_right = jnp.asarray(right, dtype=jnp.int32)
        self.node_leaf_count = jnp.asarray(leaf_count, dtype=jnp.int32)
        self.vertex_count = scene.vertex_count
        self.ambient_dimension = scene.ambient_dimension
        self.hierarchy_root = root
        self.hierarchy_depth = hierarchy_depth
        (
            self.edge_vertex_capacity,
            self.edge_edge_capacity,
            self.face_vertex_capacity,
        ) = capacities
        self.activation_distance = activation
        self.morton_bits = bits
        self.maximum_tree_depth = depth
        self.maximum_traversal_visits = visits
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lbvh-contact-search-plan",
                "scene": scene.scene_id,
                "capacities": capacities,
                "activation_distance": activation.hex(),
                "morton_bits": bits,
                "maximum_tree_depth": depth,
                "maximum_traversal_visits": visits,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
    ) -> LBVHContactSearchResult:
        start = jnp.asarray(positions)
        end = (
            start
            if end_positions is None
            else jnp.asarray(end_positions, dtype=start.dtype)
        )
        expected = (self.vertex_count, self.ambient_dimension)
        if start.shape != expected or end.shape != expected:
            raise ValueError(f"LBVH positions must have shape {expected}.")
        finite_positions = jnp.all(jnp.isfinite(start)) & jnp.all(jnp.isfinite(end))
        if self.hierarchy_depth > self.maximum_tree_depth:
            return _empty_lbvh_result(
                self,
                finite_positions,
                stack_overflow=True,
                visit_overflow=False,
            )

        point_min = jnp.minimum(start, end)
        point_max = jnp.maximum(start, end)
        edge_start = start[self.edges]
        edge_end = end[self.edges]
        edge_min = jnp.minimum(edge_start, edge_end).min(axis=1)
        edge_max = jnp.maximum(edge_start, edge_end).max(axis=1)
        face_start = start[self.faces]
        face_end = end[self.faces]
        face_min = (
            jnp.minimum(face_start, face_end).min(axis=1)
            if self.faces.size
            else jnp.empty((0, self.ambient_dimension), dtype=start.dtype)
        )
        face_max = (
            jnp.maximum(face_start, face_end).max(axis=1)
            if self.faces.size
            else jnp.empty((0, self.ambient_dimension), dtype=start.dtype)
        )
        primitive_min = jnp.concatenate((point_min, edge_min, face_min), axis=0)
        primitive_max = jnp.concatenate((point_max, edge_max, face_max), axis=0)
        finite_bounds = jnp.all(jnp.isfinite(primitive_min)) & jnp.all(
            jnp.isfinite(primitive_max)
        )
        centroid = 0.5 * primitive_min + 0.5 * primitive_max
        global_min = jnp.min(primitive_min, axis=0)
        global_max = jnp.max(primitive_max, axis=0)
        extent = global_max - global_min
        safe_extent = jnp.where(jnp.isfinite(extent) & (extent > 0.0), extent, 1.0)
        maximum_code_coordinate = (1 << self.morton_bits) - 1
        normalized = (centroid - global_min) / safe_extent
        normalized = jnp.where(jnp.isfinite(normalized), normalized, 0.0)
        quantized = jnp.floor(
            jnp.clip(normalized, 0.0, 1.0) * maximum_code_coordinate
        ).astype(jnp.uint32)
        codes = morton_encode_integer(quantized, self.morton_bits)
        primitive_kind = jnp.concatenate(
            (
                jnp.zeros((self.vertex_count,), dtype=jnp.int32),
                jnp.ones((self.edges.shape[0],), dtype=jnp.int32),
                jnp.full((self.faces.shape[0],), 2, dtype=jnp.int32),
            )
        )
        stable_id = jnp.arange(codes.size, dtype=jnp.int32)
        order = jnp.lexsort((stable_id, primitive_kind, codes)).astype(jnp.int32)
        sorted_codes = codes[order]
        duplicate_codes = jnp.sum(sorted_codes[1:] == sorted_codes[:-1], dtype=jnp.int32)

        primitive_count = int(codes.size)
        node_count = 2 * primitive_count - 1
        node_min = jnp.zeros(
            (node_count, self.ambient_dimension), dtype=primitive_min.dtype
        )
        node_max = jnp.zeros(
            (node_count, self.ambient_dimension), dtype=primitive_max.dtype
        )
        node_min = node_min.at[:primitive_count].set(primitive_min[order])
        node_max = node_max.at[:primitive_count].set(primitive_max[order])

        if primitive_count > 1:

            def combine_bounds(index, bounds):
                lower, upper = bounds
                left = self.tree_left[index]
                right = self.tree_right[index]
                parent = primitive_count + index
                lower = lower.at[parent].set(jnp.minimum(lower[left], lower[right]))
                upper = upper.at[parent].set(jnp.maximum(upper[left], upper[right]))
                return lower, upper

            node_min, node_max = jax.lax.fori_loop(
                0,
                primitive_count - 1,
                combine_bounds,
                (node_min, node_max),
            )
        else:
            root_overlap = _aabb_mask(
                node_min[0],
                node_max[0],
                node_min[0],
                node_max[0],
                self.activation_distance,
            )
            edge_vertex = _empty_candidate_batch(
                ContactStencilKind.EDGE_VERTEX, self.edge_vertex_capacity
            )
            edge_edge = _empty_candidate_batch(
                ContactStencilKind.EDGE_EDGE, self.edge_edge_capacity
            )
            face_vertex = _empty_candidate_batch(
                ContactStencilKind.FACE_VERTEX, self.face_vertex_capacity
            )
            evidence = LBVHContactSearchEvidence(
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                jnp.asarray(1, dtype=jnp.int32),
                duplicate_codes,
                jnp.asarray(False),
                jnp.asarray(False),
                jnp.asarray(0, dtype=jnp.int32),
                finite_bounds,
                finite_bounds & root_overlap,
                self.plan_id,
            )
            return LBVHContactSearchResult(edge_vertex, edge_edge, face_vertex, evidence)

        stack_capacity = 4 * self.hierarchy_depth
        stack_first = jnp.full((stack_capacity,), -1, dtype=jnp.int32)
        stack_second = jnp.full((stack_capacity,), -1, dtype=jnp.int32)
        stack_first = stack_first.at[0].set(self.hierarchy_root)
        stack_second = stack_second.at[0].set(self.hierarchy_root)
        edge_vertex_indices = jnp.full(
            (self.edge_vertex_capacity, 4), -1, dtype=jnp.int32
        )
        edge_edge_indices = jnp.full((self.edge_edge_capacity, 4), -1, dtype=jnp.int32)
        face_vertex_indices = jnp.full(
            (self.face_vertex_capacity, 4), -1, dtype=jnp.int32
        )
        outputs = (
            edge_vertex_indices,
            jnp.asarray(0, dtype=jnp.int32),
            edge_edge_indices,
            jnp.asarray(0, dtype=jnp.int32),
            face_vertex_indices,
            jnp.asarray(0, dtype=jnp.int32),
        )
        initial = (
            stack_first,
            stack_second,
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            outputs,
        )

        def continue_traversal(state):
            _, _, stack_size, visits, stack_overflow, _ = state
            return (
                (stack_size > 0)
                & (visits < self.maximum_traversal_visits)
                & ~stack_overflow
            )

        def record_leaf_pair(first, second, current_outputs):
            (
                edge_vertex_output,
                edge_vertex_count,
                edge_edge_output,
                edge_edge_count,
                face_vertex_output,
                face_vertex_count,
            ) = current_outputs
            original_first = order[first]
            original_second = order[second]
            first_kind = primitive_kind[original_first]
            second_kind = primitive_kind[original_second]
            minus_one = jnp.asarray(-1, dtype=jnp.int32)
            if (self.ambient_dimension == 2 or not self.faces.size) and self.edges.size:
                is_edge_vertex = ((first_kind == 0) & (second_kind == 1)) | (
                    (first_kind == 1) & (second_kind == 0)
                )
                vertex = jnp.where(first_kind == 0, original_first, original_second)
                edge = (
                    jnp.where(first_kind == 1, original_first, original_second)
                    - self.vertex_count
                )
                edge = jnp.clip(edge, 0, self.edges.shape[0] - 1)
                legal = jnp.all(self.edges[edge] != vertex)
                row = jnp.stack(
                    (
                        vertex,
                        self.edges[edge, 0],
                        self.edges[edge, 1],
                        minus_one,
                    )
                )
                edge_vertex_output, edge_vertex_count = _append_traversal_candidate(
                    edge_vertex_output,
                    edge_vertex_count,
                    row,
                    is_edge_vertex & legal,
                    self.edge_vertex_capacity,
                )
            elif self.faces.size:
                is_face_vertex = ((first_kind == 0) & (second_kind == 2)) | (
                    (first_kind == 2) & (second_kind == 0)
                )
                vertex = jnp.where(first_kind == 0, original_first, original_second)
                face = (
                    jnp.where(first_kind == 2, original_first, original_second)
                    - self.vertex_count
                    - self.edges.shape[0]
                )
                face = jnp.clip(face, 0, self.faces.shape[0] - 1)
                face_legal = jnp.all(self.faces[face] != vertex)
                face_row = jnp.concatenate((vertex[None], self.faces[face]))
                face_vertex_output, face_vertex_count = _append_traversal_candidate(
                    face_vertex_output,
                    face_vertex_count,
                    face_row,
                    is_face_vertex & face_legal,
                    self.face_vertex_capacity,
                )
                if self.edges.size:
                    is_edge_edge = (
                        (first_kind == 1)
                        & (second_kind == 1)
                        & (original_first != original_second)
                    )
                    first_edge = jnp.clip(
                        original_first - self.vertex_count,
                        0,
                        self.edges.shape[0] - 1,
                    )
                    second_edge = jnp.clip(
                        original_second - self.vertex_count,
                        0,
                        self.edges.shape[0] - 1,
                    )
                    lower_edge = jnp.minimum(first_edge, second_edge)
                    upper_edge = jnp.maximum(first_edge, second_edge)
                    edge_legal = jnp.all(
                        self.edges[lower_edge, :, None] != self.edges[upper_edge, None, :]
                    )
                    edge_row = jnp.concatenate(
                        (self.edges[lower_edge], self.edges[upper_edge])
                    )
                    edge_edge_output, edge_edge_count = _append_traversal_candidate(
                        edge_edge_output,
                        edge_edge_count,
                        edge_row,
                        is_edge_edge & edge_legal,
                        self.edge_edge_capacity,
                    )
            return (
                edge_vertex_output,
                edge_vertex_count,
                edge_edge_output,
                edge_edge_count,
                face_vertex_output,
                face_vertex_count,
            )

        def traverse_one(state):
            (
                first_stack,
                second_stack,
                stack_size,
                visits,
                stack_overflow,
                current_outputs,
            ) = state
            stack_slot = stack_size - 1
            first = first_stack[stack_slot]
            second = second_stack[stack_slot]
            active = (
                first_stack,
                second_stack,
                stack_slot,
                stack_overflow,
                current_outputs,
            )
            overlap = _aabb_mask(
                node_min[first],
                node_max[first],
                node_min[second],
                node_max[second],
                self.activation_distance,
            )

            def push_pairs(active_state, new_first, new_second):
                active_size = active_state[2]
                pair_count = int(new_first.shape[0])
                has_capacity = active_size + pair_count <= stack_capacity

                def store_pairs(store_state):
                    (
                        store_first,
                        store_second,
                        store_size,
                        store_overflow,
                        store_outputs,
                    ) = store_state
                    slots = store_size + jnp.arange(pair_count, dtype=jnp.int32)
                    store_first = store_first.at[slots].set(new_first)
                    store_second = store_second.at[slots].set(new_second)
                    return (
                        store_first,
                        store_second,
                        store_size + pair_count,
                        store_overflow,
                        store_outputs,
                    )

                def report_overflow(overflow_state):
                    (
                        overflow_first,
                        overflow_second,
                        overflow_size,
                        _,
                        overflow_outputs,
                    ) = overflow_state
                    return (
                        overflow_first,
                        overflow_second,
                        overflow_size,
                        jnp.asarray(True),
                        overflow_outputs,
                    )

                return jax.lax.cond(
                    has_capacity, store_pairs, report_overflow, active_state
                )

            def process_overlap(active_state):
                first_is_leaf = first < primitive_count
                second_is_leaf = second < primitive_count

                def process_leaves(leaf_state):
                    (
                        leaf_first,
                        leaf_second,
                        leaf_size,
                        leaf_overflow,
                        leaf_outputs,
                    ) = leaf_state
                    return (
                        leaf_first,
                        leaf_second,
                        leaf_size,
                        leaf_overflow,
                        record_leaf_pair(first, second, leaf_outputs),
                    )

                def expand_nodes(node_state):
                    first_internal = jnp.clip(
                        first - primitive_count, 0, primitive_count - 2
                    )
                    second_internal = jnp.clip(
                        second - primitive_count, 0, primitive_count - 2
                    )
                    first_left = self.tree_left[first_internal]
                    first_right = self.tree_right[first_internal]
                    second_left = self.tree_left[second_internal]
                    second_right = self.tree_right[second_internal]

                    def expand_self(self_state):
                        return push_pairs(
                            self_state,
                            jnp.stack((first_right, first_left, first_left)),
                            jnp.stack((first_right, first_right, first_left)),
                        )

                    def expand_distinct(distinct_state):
                        split_first = (~first_is_leaf) & (
                            second_is_leaf
                            | (
                                self.node_leaf_count[first]
                                >= self.node_leaf_count[second]
                            )
                        )

                        def split_first_node(split_state):
                            return push_pairs(
                                split_state,
                                jnp.stack((first_right, first_left)),
                                jnp.stack((second, second)),
                            )

                        def split_second_node(split_state):
                            return push_pairs(
                                split_state,
                                jnp.stack((first, first)),
                                jnp.stack((second_right, second_left)),
                            )

                        return jax.lax.cond(
                            split_first,
                            split_first_node,
                            split_second_node,
                            distinct_state,
                        )

                    return jax.lax.cond(
                        first == second, expand_self, expand_distinct, node_state
                    )

                return jax.lax.cond(
                    first_is_leaf & second_is_leaf,
                    process_leaves,
                    expand_nodes,
                    active_state,
                )

            active = jax.lax.cond(overlap, process_overlap, lambda value: value, active)
            (
                first_stack,
                second_stack,
                stack_size,
                stack_overflow,
                current_outputs,
            ) = active
            return (
                first_stack,
                second_stack,
                stack_size,
                visits + 1,
                stack_overflow,
                current_outputs,
            )

        (
            _,
            _,
            remaining,
            traversal_visits,
            stack_overflow,
            outputs,
        ) = jax.lax.while_loop(continue_traversal, traverse_one, initial)
        (
            edge_vertex_indices,
            edge_vertex_count,
            edge_edge_indices,
            edge_edge_count,
            face_vertex_indices,
            face_vertex_count,
        ) = outputs
        edge_vertex = _traversal_candidate_batch(
            ContactStencilKind.EDGE_VERTEX,
            edge_vertex_indices,
            edge_vertex_count,
            self.edge_vertex_capacity,
        )
        edge_edge = _traversal_candidate_batch(
            ContactStencilKind.EDGE_EDGE,
            edge_edge_indices,
            edge_edge_count,
            self.edge_edge_capacity,
        )
        face_vertex = _traversal_candidate_batch(
            ContactStencilKind.FACE_VERTEX,
            face_vertex_indices,
            face_vertex_count,
            self.face_vertex_capacity,
        )
        output_overflow = (
            edge_vertex.overflow_count
            + edge_edge.overflow_count
            + face_vertex.overflow_count
        )
        visit_overflow = remaining > 0
        candidate_count = (
            edge_vertex.actual_count + edge_edge.actual_count + face_vertex.actual_count
        )
        complete = (
            finite_bounds & ~stack_overflow & ~visit_overflow & (output_overflow == 0)
        )
        evidence = LBVHContactSearchEvidence(
            candidate_count,
            jnp.asarray(node_count, dtype=jnp.int32),
            jnp.asarray(self.hierarchy_depth, dtype=jnp.int32),
            traversal_visits,
            duplicate_codes,
            stack_overflow,
            visit_overflow,
            output_overflow,
            finite_bounds,
            complete,
            self.plan_id,
        )
        return LBVHContactSearchResult(edge_vertex, edge_edge, face_vertex, evidence)


class CompiledContactSearchPlan(StrictModule, NonTrainableState):
    """Fixed-shape device search over predeclared legal primitive pairs.

    The runtime filter is fully JAX compilable. Preparation enumerates legal
    pairs once; runtime evaluates swept/static AABB separation and packs each
    kind under fixed capacity. This is the deterministic compiled authority
    before a future asymptotically faster device LBVH backend.
    """

    edge_vertex_pairs: Array
    edge_edge_pairs: Array
    face_vertex_pairs: Array
    edges: Array
    faces: Array
    edge_vertex_capacity: int = eqx.field(static=True)
    edge_edge_capacity: int = eqx.field(static=True)
    face_vertex_capacity: int = eqx.field(static=True)
    activation_distance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scene: PreparedCollisionScene,
        /,
        *,
        edge_vertex_capacity: int,
        edge_edge_capacity: int,
        face_vertex_capacity: int,
        activation_distance: float,
    ):
        if not isinstance(scene, PreparedCollisionScene):
            raise TypeError("scene must be PreparedCollisionScene.")
        capacities = (
            int(edge_vertex_capacity),
            int(edge_edge_capacity),
            int(face_vertex_capacity),
        )
        if any(value < 0 for value in capacities) or sum(capacities) <= 0:
            raise ValueError("Compiled search capacities are invalid.")
        activation = float(activation_distance)
        if not np.isfinite(activation) or activation <= 0.0:
            raise ValueError("activation_distance must be finite and positive.")
        edges = np.asarray(scene.edges, dtype=np.int32)
        faces = np.asarray(scene.faces, dtype=np.int32)
        edge_vertex = []
        edge_edge = []
        face_vertex = []
        if scene.ambient_dimension == 2 or faces.size == 0:
            for vertex in range(scene.vertex_count):
                for edge_index, edge in enumerate(edges):
                    if vertex not in edge:
                        edge_vertex.append((vertex, edge_index))
        else:
            for vertex in range(scene.vertex_count):
                for face_index, face in enumerate(faces):
                    if vertex not in face:
                        face_vertex.append((vertex, face_index))
            for first in range(edges.shape[0]):
                for second in range(first + 1, edges.shape[0]):
                    if not set(edges[first]).intersection(edges[second]):
                        edge_edge.append((first, second))
        self.edge_vertex_pairs = jnp.asarray(
            np.asarray(edge_vertex, dtype=np.int32).reshape((-1, 2))
        )
        self.edge_edge_pairs = jnp.asarray(
            np.asarray(edge_edge, dtype=np.int32).reshape((-1, 2))
        )
        self.face_vertex_pairs = jnp.asarray(
            np.asarray(face_vertex, dtype=np.int32).reshape((-1, 2))
        )
        self.edges = jnp.asarray(edges)
        self.faces = jnp.asarray(faces)
        (
            self.edge_vertex_capacity,
            self.edge_edge_capacity,
            self.face_vertex_capacity,
        ) = capacities
        self.activation_distance = activation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compiled-contact-search-plan",
                "scene": scene.scene_id,
                "edge_vertex": array_tree_fingerprint(edge_vertex),
                "edge_edge": array_tree_fingerprint(edge_edge),
                "face_vertex": array_tree_fingerprint(face_vertex),
                "capacities": capacities,
                "activation_distance": activation.hex(),
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        /,
        *,
        end_positions: ArrayLike | None = None,
    ) -> CompiledContactSearchResult:
        start = jnp.asarray(positions)
        end = (
            start
            if end_positions is None
            else jnp.asarray(end_positions, dtype=start.dtype)
        )
        if start.shape != end.shape or start.ndim != 2:
            raise ValueError("Compiled search positions have invalid shape.")
        point_min = jnp.minimum(start, end)
        point_max = jnp.maximum(start, end)
        edge_start = start[self.edges]
        edge_end = end[self.edges]
        edge_min = jnp.minimum(edge_start, edge_end).min(axis=1)
        edge_max = jnp.maximum(edge_start, edge_end).max(axis=1)
        if self.faces.size:
            face_start = start[self.faces]
            face_end = end[self.faces]
            face_min = jnp.minimum(face_start, face_end).min(axis=1)
            face_max = jnp.maximum(face_start, face_end).max(axis=1)
        else:
            face_min = jnp.empty((0, start.shape[1]), dtype=start.dtype)
            face_max = jnp.empty((0, start.shape[1]), dtype=start.dtype)

        edge_vertex = _pack_compiled_pairs(
            ContactStencilKind.EDGE_VERTEX,
            self.edge_vertex_pairs,
            self.edge_vertex_capacity,
            point_min,
            point_max,
            edge_min,
            edge_max,
            self.edges,
            self.activation_distance,
        )
        edge_edge = _pack_compiled_same_pairs(
            ContactStencilKind.EDGE_EDGE,
            self.edge_edge_pairs,
            self.edge_edge_capacity,
            edge_min,
            edge_max,
            self.edges,
            self.activation_distance,
        )
        face_vertex = _pack_compiled_pairs(
            ContactStencilKind.FACE_VERTEX,
            self.face_vertex_pairs,
            self.face_vertex_capacity,
            point_min,
            point_max,
            face_min,
            face_max,
            self.faces,
            self.activation_distance,
        )
        count = (
            edge_vertex.actual_count + edge_edge.actual_count + face_vertex.actual_count
        )
        overflow = (
            edge_vertex.overflow_count
            + edge_edge.overflow_count
            + face_vertex.overflow_count
        )
        finite = jnp.all(jnp.isfinite(start)) & jnp.all(jnp.isfinite(end))
        evidence = CompiledContactSearchEvidence(
            count,
            overflow,
            finite,
            finite & (overflow == 0),
            self.plan_id,
        )
        return CompiledContactSearchResult(edge_vertex, edge_edge, face_vertex, evidence)


def _aabb_mask(first_min, first_max, second_min, second_max, radius):
    delta = jnp.maximum(
        0.0,
        jnp.maximum(first_min - second_max, second_min - first_max),
    )
    return jnp.sum(delta * delta, axis=-1) <= radius * radius


def _balanced_morton_hierarchy(leaf_count: int, /):
    count = int(leaf_count)
    if count <= 0:
        raise ValueError("LBVH construction requires at least one primitive.")
    left_children: list[int] = []
    right_children: list[int] = []
    node_leaf_count = [1] * count

    def build(first: int, last: int):
        if last - first == 1:
            return first, 1
        middle = first + (last - first) // 2
        left, left_depth = build(first, middle)
        right, right_depth = build(middle, last)
        parent = count + len(left_children)
        left_children.append(left)
        right_children.append(right)
        node_leaf_count.append(node_leaf_count[left] + node_leaf_count[right])
        return parent, max(left_depth, right_depth) + 1

    root, depth = build(0, count)
    return (
        np.asarray(left_children, dtype=np.int32),
        np.asarray(right_children, dtype=np.int32),
        np.asarray(node_leaf_count, dtype=np.int32),
        root,
        depth,
    )


def _empty_candidate_batch(kind, capacity):
    return CompiledCandidateBatch(
        jnp.full((capacity, 4), -1, dtype=jnp.int32),
        jnp.zeros((capacity,), dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        kind,
        capacity,
    )


def _empty_lbvh_result(plan, finite_bounds, /, *, stack_overflow, visit_overflow):
    edge_vertex = _empty_candidate_batch(
        ContactStencilKind.EDGE_VERTEX, plan.edge_vertex_capacity
    )
    edge_edge = _empty_candidate_batch(
        ContactStencilKind.EDGE_EDGE, plan.edge_edge_capacity
    )
    face_vertex = _empty_candidate_batch(
        ContactStencilKind.FACE_VERTEX, plan.face_vertex_capacity
    )
    evidence = LBVHContactSearchEvidence(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(stack_overflow, dtype=bool),
        jnp.asarray(visit_overflow, dtype=bool),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(finite_bounds, dtype=bool),
        jnp.asarray(False),
        plan.plan_id,
    )
    return LBVHContactSearchResult(edge_vertex, edge_edge, face_vertex, evidence)


def _append_traversal_candidate(indices, count, row, accepted, capacity):
    accepted = jnp.asarray(accepted, dtype=bool)
    if capacity == 0:
        return indices, count + accepted.astype(jnp.int32)

    def store(values):
        output, slot = values
        return output.at[slot].set(row)

    indices = jax.lax.cond(
        accepted & (count < capacity),
        store,
        lambda values: values[0],
        (indices, count),
    )
    return indices, count + accepted.astype(jnp.int32)


def _traversal_candidate_batch(kind, indices, count, capacity):
    valid = jnp.arange(capacity, dtype=jnp.int32) < jnp.minimum(count, capacity)
    overflow = jnp.maximum(count - capacity, 0)
    return CompiledCandidateBatch(indices, valid, count, overflow, kind, capacity)


def _pack_indices(mask, capacity):
    count = jnp.sum(mask, dtype=jnp.int32)
    selected = jnp.nonzero(mask, size=capacity, fill_value=0)[0]
    valid = jnp.arange(capacity) < jnp.minimum(count, capacity)
    overflow = jnp.maximum(count - capacity, 0)
    return selected, valid, count, overflow


def _pack_compiled_pairs(
    kind,
    pairs,
    capacity,
    point_min,
    point_max,
    primitive_min,
    primitive_max,
    primitive_topology,
    radius,
    *,
    legal=None,
):
    if pairs.shape[0] == 0:
        return CompiledCandidateBatch(
            jnp.full((capacity, 4), -1, dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            kind,
            capacity,
        )
    point = pairs[:, 0]
    primitive = pairs[:, 1]
    mask = _aabb_mask(
        point_min[point],
        point_max[point],
        primitive_min[primitive],
        primitive_max[primitive],
        radius,
    )
    if legal is not None:
        mask = mask & jnp.asarray(legal, dtype=bool)
    selected, valid, count, overflow = _pack_indices(mask, capacity)
    selected_pairs = pairs[selected]
    endpoints = primitive_topology[selected_pairs[:, 1]]
    padding = 4 - (1 + endpoints.shape[1])
    indices = jnp.concatenate(
        (
            selected_pairs[:, :1],
            endpoints,
            jnp.full((capacity, padding), -1, dtype=jnp.int32),
        ),
        axis=1,
    )
    return CompiledCandidateBatch(indices, valid, count, overflow, kind, capacity)


def _pack_compiled_same_pairs(
    kind,
    pairs,
    capacity,
    lower,
    upper,
    topology,
    radius,
    *,
    legal=None,
):
    if pairs.shape[0] == 0:
        return CompiledCandidateBatch(
            jnp.full((capacity, 4), -1, dtype=jnp.int32),
            jnp.zeros((capacity,), dtype=bool),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            kind,
            capacity,
        )
    first = pairs[:, 0]
    second = pairs[:, 1]
    mask = _aabb_mask(lower[first], upper[first], lower[second], upper[second], radius)
    if legal is not None:
        mask = mask & jnp.asarray(legal, dtype=bool)
    selected, valid, count, overflow = _pack_indices(mask, capacity)
    selected_pairs = pairs[selected]
    indices = jnp.concatenate(
        (
            topology[selected_pairs[:, 0]],
            topology[selected_pairs[:, 1]],
        ),
        axis=1,
    )
    return CompiledCandidateBatch(indices, valid, count, overflow, kind, capacity)


__all__ = [
    "CompiledCandidateBatch",
    "CompiledContactSearchEvidence",
    "CompiledContactSearchPlan",
    "CompiledContactSearchResult",
    "LBVHContactSearchEvidence",
    "LBVHContactSearchPlan",
    "LBVHContactSearchResult",
]
