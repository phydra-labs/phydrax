#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import heapq
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from ..sparse import EdgeRelation
from ._method import (
    AbstractLinearCombinatorialMethod,
    CombinatorialPlan,
    make_combinatorial_plan,
)
from ._problem import AbstractCombinatorialSpace, LinearCombinatorialProblem
from ._selection import relative_gap
from ._types import (
    CombinatorialCertificate,
    CombinatorialCertification,
    CombinatorialFeasibility,
    CombinatorialMethodCapabilities,
    CombinatorialProvenance,
    CombinatorialResult,
    CombinatorialStatus,
)


class PathDecision(StrictModule):
    """Fixed-capacity source-to-target path in canonical forward order."""

    vertices: Array
    edges: Array
    length: Array


def _topology(
    relation: EdgeRelation,
    /,
) -> tuple[bool, Array, Array, Array, Array, int]:
    vertices = relation.source_size
    source = np.asarray(relation.source_indices)
    target = np.asarray(relation.target_indices)
    valid = np.asarray(relation.valid, dtype=bool)
    indegree = np.zeros((vertices,), dtype=np.int32)
    outgoing: list[list[int]] = [[] for _ in range(vertices)]
    incoming: list[list[int]] = [[] for _ in range(vertices)]
    for edge in np.nonzero(valid)[0].tolist():
        left = int(source[edge])
        right = int(target[edge])
        indegree[right] += 1
        outgoing[left].append(right)
        incoming[right].append(edge)
    ready = [int(vertex) for vertex in np.nonzero(indegree == 0)[0].tolist()]
    heapq.heapify(ready)
    order: list[int] = []
    while ready:
        vertex = heapq.heappop(ready)
        order.append(vertex)
        for right in outgoing[vertex]:
            indegree[right] -= 1
            if indegree[right] == 0:
                heapq.heappush(ready, right)
    acyclic = len(order) == vertices
    if not acyclic:
        order = list(range(vertices))
    width = max(1, max((len(edges) for edges in incoming), default=0))
    incoming_sources = np.zeros((vertices, width), dtype=np.int32)
    incoming_edges = np.full((vertices, width), relation.capacity, dtype=np.int32)
    incoming_valid = np.zeros((vertices, width), dtype=bool)
    for vertex, edges in enumerate(incoming):
        for slot, edge in enumerate(sorted(edges)):
            incoming_sources[vertex, slot] = int(source[edge])
            incoming_edges[vertex, slot] = edge
            incoming_valid[vertex, slot] = True
    return (
        acyclic,
        jnp.asarray(order, dtype=jnp.int32),
        jnp.asarray(incoming_sources),
        jnp.asarray(incoming_edges),
        jnp.asarray(incoming_valid),
        width,
    )


class ShortestPathSpace(AbstractCombinatorialSpace):
    """Directed source-to-target paths over one fixed edge relation."""

    relation: EdgeRelation
    topological_order: Array
    incoming_sources: Array
    incoming_edges: Array
    incoming_valid: Array
    source: int = eqx.field(static=True)
    target: int = eqx.field(static=True)
    vertex_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    incoming_width: int = eqx.field(static=True)
    acyclic: bool = eqx.field(static=True)
    _structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        source: int,
        target: int,
        /,
    ):
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be an EdgeRelation.")
        if relation.source_size != relation.target_size:
            raise ValueError("shortest-path relations require one shared vertex space.")
        vertices = relation.source_size
        if vertices <= 0:
            raise ValueError("shortest-path spaces require at least one vertex.")
        if isinstance(source, bool) or not isinstance(source, Integral):
            raise TypeError("source must be an integer vertex index.")
        if isinstance(target, bool) or not isinstance(target, Integral):
            raise TypeError("target must be an integer vertex index.")
        source_ = int(source)
        target_ = int(target)
        if not 0 <= source_ < vertices or not 0 <= target_ < vertices:
            raise ValueError("source and target must lie inside the vertex space.")
        (
            acyclic,
            order,
            incoming_sources,
            incoming_edges,
            incoming_valid,
            incoming_width,
        ) = _topology(relation)
        self.relation = relation
        self.topological_order = order
        self.incoming_sources = incoming_sources
        self.incoming_edges = incoming_edges
        self.incoming_valid = incoming_valid
        self.source = source_
        self.target = target_
        self.vertex_count = vertices
        self.edge_count = relation.capacity
        self.incoming_width = incoming_width
        self.acyclic = acyclic
        self._structure_id = canonical_fingerprint(
            {
                "kind": "shortest-path-space",
                "source": source_,
                "target": target_,
                "vertices": vertices,
                "edges": array_tree_fingerprint(
                    (
                        relation.source_indices,
                        relation.target_indices,
                        relation.valid,
                    )
                ),
            }
        )

    @property
    def structure_id(self) -> str:
        return self._structure_id

    def decision_spec(self, /) -> PathDecision:
        return PathDecision(
            jax.ShapeDtypeStruct((self.vertex_count,), jnp.int32),
            jax.ShapeDtypeStruct((max(self.vertex_count - 1, 0),), jnp.int32),
            jax.ShapeDtypeStruct((), jnp.int32),
        )

    def feature_spec(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.edge_count,), jnp.float32)

    def canonicalize(self, decision: PathDecision, /) -> PathDecision:
        if not isinstance(decision, PathDecision):
            raise TypeError("shortest-path decisions must be PathDecision values.")
        vertices = jnp.asarray(decision.vertices, dtype=jnp.int32)
        edges = jnp.asarray(decision.edges, dtype=jnp.int32)
        length = jnp.asarray(decision.length, dtype=jnp.int32)
        if vertices.shape[-1:] != (self.vertex_count,):
            raise ValueError(
                f"path vertices must end with shape {(self.vertex_count,)}; "
                f"got {vertices.shape}."
            )
        edge_capacity = max(self.vertex_count - 1, 0)
        if edges.shape[-1:] != (edge_capacity,):
            raise ValueError(
                f"path edges must end with shape {(edge_capacity,)}; got {edges.shape}."
            )
        if length.shape != vertices.shape[:-1]:
            raise ValueError("path length must match the decision batch shape.")
        safe_length = jnp.clip(length, 0, self.vertex_count)
        vertex_positions = jnp.arange(self.vertex_count)
        edge_positions = jnp.arange(edge_capacity)
        vertices = jnp.where(
            vertex_positions < safe_length[..., None],
            vertices,
            -1,
        )
        edges = jnp.where(
            edge_positions < jnp.maximum(safe_length - 1, 0)[..., None],
            edges,
            -1,
        )
        return PathDecision(vertices, edges, length)

    def encode(self, decision: PathDecision, /) -> Array:
        canonical = self.canonicalize(decision)
        if self.edge_count == 0:
            return jnp.zeros(canonical.length.shape + (0,), dtype=float)
        encoded = jax.nn.one_hot(
            canonical.edges,
            self.edge_count,
            dtype=float,
            axis=-1,
        )
        return jnp.sum(encoded, axis=-2)

    def audit(self, decision: PathDecision, /) -> CombinatorialFeasibility:
        canonical = self.canonicalize(decision)
        vertices = canonical.vertices
        edges = canonical.edges
        length = canonical.length
        batch_shape = length.shape
        valid_length = (length >= 1) & (length <= self.vertex_count)
        safe_length = jnp.clip(length, 1, self.vertex_count)
        first_ok = vertices[..., 0] == self.source
        last = jnp.take_along_axis(
            vertices,
            (safe_length - 1)[..., None],
            axis=-1,
        )[..., 0]
        last_ok = last == self.target
        edge_capacity = max(self.vertex_count - 1, 0)
        if edge_capacity:
            positions = jnp.arange(edge_capacity)
            active = positions < jnp.maximum(safe_length - 1, 0)[..., None]
            in_range = (edges >= 0) & (edges < self.edge_count)
            safe_edges = jnp.clip(edges, 0, max(self.edge_count - 1, 0))
            if self.edge_count:
                relation_valid = self.relation.valid[safe_edges]
                expected_source = self.relation.source_indices[safe_edges]
                expected_target = self.relation.target_indices[safe_edges]
            else:
                relation_valid = jnp.zeros_like(edges, dtype=bool)
                expected_source = jnp.zeros_like(edges)
                expected_target = jnp.zeros_like(edges)
            transition_ok = (
                in_range
                & relation_valid
                & (expected_source == vertices[..., :-1])
                & (expected_target == vertices[..., 1:])
            )
            edge_residual = jnp.sum(active & ~transition_ok, axis=-1)
        else:
            edge_residual = jnp.zeros(batch_shape, dtype=jnp.int32)
        safe_vertices = jnp.where(
            jnp.arange(self.vertex_count) < safe_length[..., None],
            vertices,
            self.vertex_count,
        )
        ordered = jnp.sort(safe_vertices, axis=-1)
        if self.vertex_count > 1:
            duplicate_residual = jnp.sum(
                (ordered[..., 1:] == ordered[..., :-1])
                & (ordered[..., 1:] < self.vertex_count),
                axis=-1,
            )
        else:
            duplicate_residual = jnp.zeros(batch_shape, dtype=jnp.int32)
        endpoint_residual = (~valid_length).astype(jnp.int32)
        endpoint_residual += (~first_ok).astype(jnp.int32)
        endpoint_residual += (~last_ok).astype(jnp.int32)
        residual = endpoint_residual + edge_residual + duplicate_residual
        return CombinatorialFeasibility(residual == 0, residual.astype(float))


def _reconstruct_one(
    predecessor: Array,
    edge_sources: Array,
    source: int,
    target: int,
    vertex_count: int,
    edge_count: int,
    /,
) -> tuple[Array, Array, Array, Array]:
    edge_capacity = max(vertex_count - 1, 0)
    reverse_vertices = jnp.full((vertex_count,), -1, dtype=jnp.int32)
    reverse_vertices = reverse_vertices.at[0].set(target)
    reverse_edges = jnp.full((edge_capacity,), -1, dtype=jnp.int32)
    if edge_count == 0:
        reached = jnp.asarray(source == target)
        vertices = reverse_vertices.at[0].set(jnp.where(reached, source, -1))
        return (
            vertices,
            reverse_edges,
            jnp.where(reached, 1, 0).astype(jnp.int32),
            reached,
        )
    initial = (
        reverse_vertices,
        reverse_edges,
        jnp.asarray(target, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(target == source),
    )

    def body(_, state):
        vertices, edges, current, steps, reached = state
        active = ~reached & (steps < edge_capacity)
        edge = predecessor[jnp.clip(current, 0, vertex_count - 1)]
        found = active & (edge >= 0) & (edge < edge_count)
        safe_edge = jnp.clip(edge, 0, max(edge_count - 1, 0))
        next_vertex = jnp.where(found, edge_sources[safe_edge], current)
        edges = jax.lax.cond(
            edge_capacity > 0,
            lambda value: value.at[jnp.minimum(steps, edge_capacity - 1)].set(
                jnp.where(found, edge, -1)
            ),
            lambda value: value,
            edges,
        )
        vertices = vertices.at[jnp.minimum(steps + 1, vertex_count - 1)].set(
            jnp.where(found, next_vertex, -1)
        )
        steps = steps + found.astype(jnp.int32)
        reached = reached | (found & (next_vertex == source))
        return vertices, edges, next_vertex, steps, reached

    reverse_vertices, reverse_edges, _, steps, reached = jax.lax.fori_loop(
        0,
        edge_capacity,
        body,
        initial,
    )
    length = jnp.where(reached, steps + 1, 0)
    vertex_positions = jnp.arange(vertex_count)
    vertex_indices = jnp.clip(length - 1 - vertex_positions, 0, vertex_count - 1)
    vertices = jnp.where(
        vertex_positions < length,
        reverse_vertices[vertex_indices],
        -1,
    )
    edge_positions = jnp.arange(edge_capacity)
    edge_indices = jnp.clip(length - 2 - edge_positions, 0, max(edge_capacity - 1, 0))
    edges = jnp.where(
        edge_positions < jnp.maximum(length - 1, 0),
        reverse_edges[edge_indices],
        -1,
    )
    return vertices, edges, length, reached


class DAGShortestPath(AbstractLinearCombinatorialMethod):
    """Exact signed-cost dynamic program over a directed acyclic graph."""

    maximum_vertices: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)
    maximum_incoming_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_vertices: int = 1_000_000,
        maximum_edges: int = 10_000_000,
        maximum_incoming_capacity: int = 10_000_000,
    ):
        limits = (maximum_vertices, maximum_edges, maximum_incoming_capacity)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in limits
        ):
            raise TypeError("DAG resource limits must be positive integers.")
        if any(int(value) <= 0 for value in limits):
            raise ValueError("DAG resource limits must be positive.")
        self.maximum_vertices = int(maximum_vertices)
        self.maximum_edges = int(maximum_edges)
        self.maximum_incoming_capacity = int(maximum_incoming_capacity)

    @property
    def method_id(self) -> str:
        return "native-dag-shortest-path"

    @property
    def capabilities(self) -> CombinatorialMethodCapabilities:
        return CombinatorialMethodCapabilities(
            exact=True,
            jax_native=True,
            jit=True,
            batched=True,
            signed_costs=True,
            deterministic_ties=True,
            optimality_certificate=True,
            surrogate_pullback=True,
        )

    @property
    def configuration(self) -> tuple[tuple[str, str], ...]:
        return (
            ("maximum_vertices", str(self.maximum_vertices)),
            ("maximum_edges", str(self.maximum_edges)),
            ("maximum_incoming_capacity", str(self.maximum_incoming_capacity)),
        )

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        if not isinstance(problem.space, ShortestPathSpace):
            raise TypeError("DAGShortestPath requires ShortestPathSpace.")
        space = problem.space
        if not space.acyclic:
            raise ValueError("DAGShortestPath requires an acyclic edge relation.")
        if len(jax.tree_util.tree_leaves(problem.costs)) != 1:
            raise ValueError("ShortestPathSpace requires one edge-cost vector.")
        if space.vertex_count > self.maximum_vertices:
            raise ValueError("shortest-path vertex count exceeds maximum_vertices.")
        if space.edge_count > self.maximum_edges:
            raise ValueError("shortest-path edge count exceeds maximum_edges.")
        incoming_capacity = space.vertex_count * space.incoming_width
        if incoming_capacity > self.maximum_incoming_capacity:
            raise ValueError(
                "shortest-path incoming topology exceeds maximum_incoming_capacity."
            )
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * incoming_capacity,
            workspace_elements=problem.batch_size
            * (2 * space.vertex_count + space.edge_count),
            certificate_kind="shortest-path-primal-dual",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, ShortestPathSpace):
            raise TypeError("DAGShortestPath requires ShortestPathSpace.")
        raw_costs = jax.tree_util.tree_leaves(problem.costs)[0]
        batch_shape = problem.batch_shape
        flat_batch = problem.batch_size
        costs = raw_costs.reshape((flat_batch, space.edge_count))
        finite = jnp.all(jnp.isfinite(costs), axis=-1)
        padded_costs = jnp.concatenate(
            (costs, jnp.full((flat_batch, 1), jnp.inf, dtype=costs.dtype)),
            axis=-1,
        )
        distances = (
            jnp.full(
                (flat_batch, space.vertex_count),
                jnp.inf,
                dtype=costs.dtype,
            )
            .at[:, space.source]
            .set(0.0)
        )
        predecessor = jnp.full(
            (flat_batch, space.vertex_count),
            -1,
            dtype=jnp.int32,
        )

        def visit(position, state):
            distance, previous = state
            vertex = space.topological_order[position]
            sources = space.incoming_sources[vertex]
            edges = space.incoming_edges[vertex]
            valid = space.incoming_valid[vertex]
            candidate = distance[:, sources] + padded_costs[:, edges]
            candidate = jnp.where(valid[None, :], candidate, jnp.inf)
            slot = jnp.argmin(candidate, axis=-1)
            best = jnp.take_along_axis(candidate, slot[:, None], axis=-1)[:, 0]
            selected_edge = edges[slot]
            available = jnp.isfinite(best)
            is_source = vertex == space.source
            next_distance = jnp.where(is_source, distance[:, vertex], best)
            next_edge = jnp.where(is_source | ~available, -1, selected_edge)
            distance = distance.at[:, vertex].set(next_distance)
            previous = previous.at[:, vertex].set(next_edge)
            return distance, previous

        distances, predecessor = jax.lax.fori_loop(
            0,
            space.vertex_count,
            visit,
            (distances, predecessor),
        )
        vertices, edges, length, reached = jax.vmap(
            _reconstruct_one,
            in_axes=(0, None, None, None, None, None),
        )(
            predecessor,
            space.relation.source_indices,
            space.source,
            space.target,
            space.vertex_count,
            space.edge_count,
        )
        decision = PathDecision(
            vertices.reshape(batch_shape + (space.vertex_count,)),
            edges.reshape(batch_shape + (max(space.vertex_count - 1, 0),)),
            length.reshape(batch_shape),
        )
        features = space.encode(decision).astype(raw_costs.dtype)
        objective = problem.objective(features)
        feasibility = space.audit(decision)
        primal = objective.reshape((flat_batch,))
        dual = distances[:, space.target]
        if space.edge_count:
            source_distance = distances[:, space.relation.source_indices]
            target_distance = distances[:, space.relation.target_indices]
            edge_valid = space.relation.valid[None, :] & jnp.isfinite(source_distance)
            violation = target_distance - source_distance - costs
            violation = jnp.where(edge_valid, violation, -jnp.inf)
            dual_residual = jnp.maximum(jnp.max(violation, axis=-1), 0.0)
        else:
            dual_residual = jnp.zeros((flat_batch,), dtype=costs.dtype)
        absolute_gap = jnp.abs(primal - dual)
        tolerance = plan.certification.threshold(primal, dual)
        objective_consistent = jnp.isfinite(primal) & (absolute_gap <= tolerance)
        solved = reached & jnp.isfinite(dual)
        certified = (
            finite
            & solved
            & feasibility.feasible.reshape((flat_batch,))
            & objective_consistent
            & (dual_residual <= tolerance)
        )
        status = jnp.where(
            ~finite,
            int(CombinatorialStatus.NONFINITE_INPUT),
            jnp.where(
                ~solved,
                int(CombinatorialStatus.INFEASIBLE),
                jnp.where(
                    certified,
                    int(CombinatorialStatus.OPTIMAL),
                    int(CombinatorialStatus.CERTIFICATION_FAILED),
                ),
            ),
        ).astype(jnp.int32)
        status = status.reshape(batch_shape)
        valid = status == int(CombinatorialStatus.OPTIMAL)
        certificate = CombinatorialCertificate(
            finite=finite.reshape(batch_shape),
            feasible=feasibility.feasible,
            objective_consistent=objective_consistent.reshape(batch_shape),
            optimality_proven=certified.reshape(batch_shape),
            primal_residual=feasibility.residual.astype(raw_costs.dtype),
            dual_residual=dual_residual.reshape(batch_shape),
            absolute_gap=absolute_gap.reshape(batch_shape),
            relative_gap=relative_gap(absolute_gap, primal, dual).reshape(batch_shape),
            tie_margin=jnp.full(batch_shape, jnp.nan, dtype=raw_costs.dtype),
            dual_available=solved.reshape(batch_shape),
            gap_available=solved.reshape(batch_shape),
            tie_available=jnp.zeros(batch_shape, dtype=bool),
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="topological-order-then-lowest-edge",
            certificate_kind=plan.certificate_kind,
            exact=True,
            signed_costs=True,
            configuration=plan.configuration,
        )
        decision = PathDecision(
            vertices=jnp.where(valid[..., None], decision.vertices, -1),
            edges=jnp.where(valid[..., None], decision.edges, -1),
            length=jnp.where(valid, decision.length, 0),
        )
        features = jnp.where(valid[..., None], features, jnp.zeros_like(features))
        return CombinatorialResult(
            decision=decision,
            features=features,
            objective_value=jnp.where(valid, objective, jnp.nan),
            status=status,
            valid=valid,
            certificate=certificate,
            iterations=jnp.full(batch_shape, space.vertex_count, dtype=jnp.int32),
            work=jnp.full(
                batch_shape,
                space.vertex_count * space.incoming_width,
                dtype=jnp.int64,
            ),
            provenance=provenance,
            batch_shape=batch_shape,
        )


__all__ = [
    "DAGShortestPath",
    "PathDecision",
    "ShortestPathSpace",
]
