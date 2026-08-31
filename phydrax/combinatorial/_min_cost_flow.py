#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
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
from ._types import (
    CombinatorialCertificate,
    CombinatorialCertification,
    CombinatorialFeasibility,
    CombinatorialMethodCapabilities,
    CombinatorialProvenance,
    CombinatorialResult,
    CombinatorialStatus,
)


class FlowDecision(StrictModule):
    """Integral flow on every edge of a fixed-capacity network."""

    flow: Array


class CapacitatedFlowSpace(AbstractCombinatorialSpace):
    """Integral capacitated flows with fixed vertex balances and edge topology."""

    relation: EdgeRelation
    balances: Array
    capacities: Array
    vertex_count: int = eqx.field(static=True)
    edge_count: int = eqx.field(static=True)
    balanced: bool = eqx.field(static=True)
    _structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation,
        balances: Any,
        capacities: Any,
        /,
    ):
        if not isinstance(relation, EdgeRelation):
            raise TypeError("relation must be an EdgeRelation.")
        if relation.source_size != relation.target_size:
            raise ValueError("flow relations require one shared vertex space.")
        vertices = relation.source_size
        edges = relation.capacity
        if vertices <= 0:
            raise ValueError("flow spaces require at least one vertex.")
        raw_balances = jnp.asarray(balances)
        raw_capacities = jnp.asarray(capacities)
        if not jnp.issubdtype(raw_balances.dtype, jnp.integer):
            raise TypeError("balances must have an integer dtype.")
        if not jnp.issubdtype(raw_capacities.dtype, jnp.integer):
            raise TypeError("capacities must have an integer dtype.")
        balances_ = raw_balances.astype(jnp.int32)
        capacities_ = raw_capacities.astype(jnp.int32)
        if balances_.shape != (vertices,):
            raise ValueError(
                f"balances must have shape {(vertices,)}; got {balances_.shape}."
            )
        if capacities_.shape != (edges,):
            raise ValueError(
                f"capacities must have shape {(edges,)}; got {capacities_.shape}."
            )
        if bool(jnp.any(capacities_ < 0)):
            raise ValueError("capacities must be non-negative.")
        capacities_ = jnp.where(relation.valid, capacities_, 0)
        self.relation = relation
        self.balances = balances_
        self.capacities = capacities_
        self.vertex_count = vertices
        self.edge_count = edges
        self.balanced = int(jnp.sum(balances_)) == 0
        self._structure_id = canonical_fingerprint(
            {
                "kind": "capacitated-flow-space",
                "topology": array_tree_fingerprint(
                    (
                        relation.source_indices,
                        relation.target_indices,
                        relation.valid,
                    )
                ),
                "balances": array_tree_fingerprint(balances_),
                "capacities": array_tree_fingerprint(capacities_),
            }
        )

    @property
    def structure_id(self) -> str:
        return self._structure_id

    def decision_spec(self, /) -> FlowDecision:
        return FlowDecision(jax.ShapeDtypeStruct((self.edge_count,), jnp.int32))

    def feature_spec(self, /) -> jax.ShapeDtypeStruct:
        return jax.ShapeDtypeStruct((self.edge_count,), jnp.float32)

    def canonicalize(self, decision: FlowDecision, /) -> FlowDecision:
        if not isinstance(decision, FlowDecision):
            raise TypeError("flow decisions must be FlowDecision values.")
        flow = jnp.asarray(decision.flow)
        if not jnp.issubdtype(flow.dtype, jnp.integer):
            raise TypeError("flow decisions must have an integer dtype.")
        flow = flow.astype(jnp.int32)
        if flow.shape[-1:] != (self.edge_count,):
            raise ValueError(
                f"flow must end with shape {(self.edge_count,)}; got {flow.shape}."
            )
        return FlowDecision(jnp.where(self.relation.valid, flow, 0))

    def encode(self, decision: FlowDecision, /) -> Array:
        return self.canonicalize(decision).flow.astype(float)

    def audit(self, decision: FlowDecision, /) -> CombinatorialFeasibility:
        canonical = self.canonicalize(decision)
        raw_flow = jnp.asarray(decision.flow, dtype=jnp.int32)
        flow = canonical.flow
        invalid_residual = jnp.sum(
            jnp.abs(jnp.where(self.relation.valid, 0, raw_flow)), axis=-1
        )
        lower_residual = jnp.sum(jnp.maximum(-flow, 0), axis=-1)
        upper_residual = jnp.sum(jnp.maximum(flow - self.capacities, 0), axis=-1)
        flat = flow.reshape((-1, self.edge_count))

        def balance_one(value):
            balance = jnp.zeros((self.vertex_count,), dtype=jnp.int32)
            balance = balance.at[self.relation.source_indices].add(value)
            balance = balance.at[self.relation.target_indices].add(-value)
            return balance

        realized = jax.vmap(balance_one)(flat).reshape(
            flow.shape[:-1] + (self.vertex_count,)
        )
        balance_residual = jnp.sum(jnp.abs(realized - self.balances), axis=-1)
        residual = invalid_residual + lower_residual + upper_residual + balance_residual
        return CombinatorialFeasibility(residual == 0, residual.astype(float))


def _residual_network(
    flow: Array,
    costs: Array,
    sources: Array,
    targets: Array,
    capacities: Array,
    valid: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    residual_sources = jnp.concatenate((sources, targets))
    residual_targets = jnp.concatenate((targets, sources))
    residual_costs = jnp.concatenate((costs, -costs))
    residual_capacity = jnp.concatenate((capacities - flow, flow))
    residual_valid = jnp.concatenate((valid, valid))
    return (
        residual_sources,
        residual_targets,
        residual_costs,
        jnp.where(residual_valid, residual_capacity, 0),
    )


def _realized_balance(
    flow: Array,
    sources: Array,
    targets: Array,
    vertex_count: int,
    /,
) -> Array:
    balance = jnp.zeros((vertex_count,), dtype=jnp.int32)
    balance = balance.at[sources].add(flow)
    balance = balance.at[targets].add(-flow)
    return balance


def _augment_feasibility(
    flow: Array,
    costs: Array,
    sources: Array,
    targets: Array,
    valid: Array,
    capacities: Array,
    balances: Array,
    vertex_count: int,
    /,
) -> tuple[Array, Array]:
    edges = flow.shape[0]
    excess = balances - _realized_balance(flow, sources, targets, vertex_count)
    if edges == 0:
        return flow, jnp.asarray(False)
    residual_sources, residual_targets, _, residual_capacity = _residual_network(
        flow, costs, sources, targets, capacities, valid
    )
    residual_edges = residual_sources.shape[0]
    source_candidates = excess > 0
    source = jnp.argmax(source_candidates.astype(jnp.int32))
    has_source = jnp.any(source_candidates)
    reached = jnp.zeros((vertex_count,), dtype=bool).at[source].set(has_source)
    predecessor = jnp.full((vertex_count,), -1, dtype=jnp.int32)

    def reach_round(_, reach_state):
        reached_, predecessor_ = reach_state

        def inspect(arc, inner):
            reached_inner, predecessor_inner = inner
            left = residual_sources[arc]
            right = residual_targets[arc]
            discover = (
                (residual_capacity[arc] > 0) & reached_inner[left] & ~reached_inner[right]
            )
            reached_inner = reached_inner.at[right].set(reached_inner[right] | discover)
            predecessor_inner = predecessor_inner.at[right].set(
                jnp.where(discover, arc, predecessor_inner[right])
            )
            return reached_inner, predecessor_inner

        return jax.lax.fori_loop(0, residual_edges, inspect, (reached_, predecessor_))

    reached, predecessor = jax.lax.fori_loop(
        0, vertex_count, reach_round, (reached, predecessor)
    )
    sink_candidates = (excess < 0) & reached
    sink = jnp.argmax(sink_candidates.astype(jnp.int32))
    found = has_source & jnp.any(sink_candidates)
    initial_delta = jnp.minimum(excess[source], -excess[sink])
    initial_path = (
        sink,
        initial_delta,
        found & (sink != source),
        found,
    )

    def inspect_path(_, state):
        current, delta, active, path_valid = state
        safe_current = jnp.clip(current, 0, vertex_count - 1)
        arc = predecessor[safe_current]
        arc_valid = active & (arc >= 0) & (arc < residual_edges)
        safe_arc = jnp.clip(arc, 0, residual_edges - 1)
        delta = jnp.where(
            arc_valid, jnp.minimum(delta, residual_capacity[safe_arc]), delta
        )
        next_vertex = jnp.where(arc_valid, residual_sources[safe_arc], current)
        active = arc_valid & (next_vertex != source)
        return next_vertex, delta, active, path_valid & (~active | arc_valid)

    terminal, delta, active, path_valid = jax.lax.fori_loop(
        0, vertex_count, inspect_path, initial_path
    )
    path_valid = path_valid & ~active & (terminal == source) & (delta > 0)

    def update_path(_, state):
        flow_, current, active_ = state
        safe_current = jnp.clip(current, 0, vertex_count - 1)
        arc = predecessor[safe_current]
        arc_valid = active_ & path_valid & (arc >= 0) & (arc < residual_edges)
        safe_arc = jnp.clip(arc, 0, residual_edges - 1)
        edge = jnp.where(safe_arc < edges, safe_arc, safe_arc - edges)
        change = jnp.where(safe_arc < edges, delta, -delta)
        flow_ = flow_.at[edge].add(jnp.where(arc_valid, change, 0))
        next_vertex = jnp.where(arc_valid, residual_sources[safe_arc], current)
        return flow_, next_vertex, arc_valid & (next_vertex != source)

    flow, _, _ = jax.lax.fori_loop(0, vertex_count, update_path, (flow, sink, found))
    return flow, path_valid


def _negative_cycle(
    flow: Array,
    costs: Array,
    sources: Array,
    targets: Array,
    valid: Array,
    capacities: Array,
    vertex_count: int,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    edges = flow.shape[0]
    if edges == 0:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return (
            empty,
            empty,
            jnp.zeros((0,), dtype=costs.dtype),
            empty,
            jnp.full((vertex_count,), -1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((vertex_count,), dtype=costs.dtype),
        )
    residual_sources, residual_targets, residual_costs, residual_capacity = (
        _residual_network(flow, costs, sources, targets, capacities, valid)
    )
    residual_edges = residual_sources.shape[0]
    initial = (
        jnp.zeros((vertex_count,), dtype=costs.dtype),
        jnp.full((vertex_count,), -1, dtype=jnp.int32),
        jnp.asarray(-1, dtype=jnp.int32),
    )

    def relaxation_round(_, round_state):
        distances, predecessor, _ = round_state
        updated = jnp.asarray(-1, dtype=jnp.int32)

        def relax(arc, state):
            distances_, predecessor_, updated_ = state
            left = residual_sources[arc]
            right = residual_targets[arc]
            candidate = distances_[left] + residual_costs[arc]
            improve = (residual_capacity[arc] > 0) & (candidate < distances_[right])
            distances_ = distances_.at[right].set(
                jnp.where(improve, candidate, distances_[right])
            )
            predecessor_ = predecessor_.at[right].set(
                jnp.where(improve, arc, predecessor_[right])
            )
            updated_ = jnp.where(improve, right, updated_)
            return distances_, predecessor_, updated_

        return jax.lax.fori_loop(
            0, residual_edges, relax, (distances, predecessor, updated)
        )

    distances, predecessor, updated = jax.lax.fori_loop(
        0, vertex_count, relaxation_round, initial
    )
    cycle_found = updated >= 0

    def enter_cycle(_, vertex):
        safe_vertex = jnp.clip(vertex, 0, vertex_count - 1)
        arc = predecessor[safe_vertex]
        safe_arc = jnp.clip(arc, 0, residual_edges - 1)
        return jnp.where(cycle_found & (arc >= 0), residual_sources[safe_arc], vertex)

    cycle_vertex = jax.lax.fori_loop(0, vertex_count, enter_cycle, updated)
    return (
        residual_sources,
        residual_targets,
        residual_costs,
        residual_capacity,
        predecessor,
        cycle_vertex,
        cycle_found,
        distances,
    )


def _cancel_cycle(
    flow: Array,
    residual_sources: Array,
    residual_capacity: Array,
    predecessor: Array,
    cycle_vertex: Array,
    cycle_found: Array,
    vertex_count: int,
    /,
) -> tuple[Array, Array]:
    edges = flow.shape[0]
    residual_edges = residual_sources.shape[0]
    if edges == 0:
        return flow, jnp.asarray(False)
    initial = (
        cycle_vertex,
        jnp.asarray(jnp.iinfo(jnp.int32).max, dtype=jnp.int32),
        cycle_found,
        cycle_found,
    )

    def inspect(_, state):
        current, delta, active, valid = state
        safe_current = jnp.clip(current, 0, vertex_count - 1)
        arc = predecessor[safe_current]
        arc_valid = active & (arc >= 0) & (arc < residual_edges)
        safe_arc = jnp.clip(arc, 0, residual_edges - 1)
        delta = jnp.where(
            arc_valid, jnp.minimum(delta, residual_capacity[safe_arc]), delta
        )
        next_vertex = jnp.where(arc_valid, residual_sources[safe_arc], current)
        active = arc_valid & (next_vertex != cycle_vertex)
        return next_vertex, delta, active, valid & (~active | arc_valid)

    terminal, delta, active, cycle_valid = jax.lax.fori_loop(
        0, vertex_count, inspect, initial
    )
    cycle_valid = cycle_valid & ~active & (terminal == cycle_vertex) & (delta > 0)

    def update(_, state):
        flow_, current, active_ = state
        safe_current = jnp.clip(current, 0, vertex_count - 1)
        arc = predecessor[safe_current]
        arc_valid = active_ & cycle_valid & (arc >= 0) & (arc < residual_edges)
        safe_arc = jnp.clip(arc, 0, residual_edges - 1)
        edge = jnp.where(safe_arc < edges, safe_arc, safe_arc - edges)
        change = jnp.where(safe_arc < edges, delta, -delta)
        flow_ = flow_.at[edge].add(jnp.where(arc_valid, change, 0))
        next_vertex = jnp.where(arc_valid, residual_sources[safe_arc], current)
        return flow_, next_vertex, arc_valid & (next_vertex != cycle_vertex)

    flow, _, _ = jax.lax.fori_loop(
        0, vertex_count, update, (flow, cycle_vertex, cycle_found)
    )
    return flow, cycle_valid


def _solve_flow_one(
    costs: Array,
    sources: Array,
    targets: Array,
    valid: Array,
    capacities: Array,
    balances: Array,
    balanced: bool,
    vertex_count: int,
    maximum_iterations: int,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    flow = jnp.zeros(capacities.shape, dtype=jnp.int32)
    initial = (
        flow,
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(not balanced),
        jnp.asarray(False),
        jnp.asarray(False),
    )

    def condition(state):
        return (state[1] < maximum_iterations) & ~state[2] & ~state[3] & ~state[4]

    def body(state):
        flow_, steps, blocked, optimal, numerical = state
        excess = balances - _realized_balance(flow_, sources, targets, vertex_count)
        feasible = jnp.all(excess == 0)

        def make_feasible(values):
            (
                current_flow,
                current_steps,
                current_blocked,
                current_optimal,
                current_numerical,
            ) = values
            next_flow, augmented = _augment_feasibility(
                current_flow,
                costs,
                sources,
                targets,
                valid,
                capacities,
                balances,
                vertex_count,
            )
            return (
                next_flow,
                current_steps + augmented.astype(jnp.int32),
                current_blocked | ~augmented,
                current_optimal,
                current_numerical,
            )

        def improve(values):
            (
                current_flow,
                current_steps,
                current_blocked,
                current_optimal,
                current_numerical,
            ) = values
            (
                residual_sources,
                _,
                _,
                residual_capacity,
                predecessor,
                cycle_vertex,
                cycle_found,
                _,
            ) = _negative_cycle(
                current_flow,
                costs,
                sources,
                targets,
                valid,
                capacities,
                vertex_count,
            )
            next_flow, cycle_valid = _cancel_cycle(
                current_flow,
                residual_sources,
                residual_capacity,
                predecessor,
                cycle_vertex,
                cycle_found,
                vertex_count,
            )
            return (
                jnp.where(cycle_found, next_flow, current_flow),
                current_steps + cycle_found.astype(jnp.int32),
                current_blocked,
                current_optimal | ~cycle_found,
                current_numerical | (cycle_found & ~cycle_valid),
            )

        return jax.lax.cond(feasible, improve, make_feasible, state)

    flow, steps, blocked, _, numerical = jax.lax.while_loop(condition, body, initial)
    realized = _realized_balance(flow, sources, targets, vertex_count)
    feasible = balanced & jnp.all(realized == balances)
    (
        _,
        _,
        _,
        _,
        _,
        _,
        negative_cycle,
        potentials,
    ) = _negative_cycle(
        flow,
        costs,
        sources,
        targets,
        valid,
        capacities,
        vertex_count,
    )
    optimal = feasible & ~negative_cycle & ~numerical
    exhausted = (steps >= maximum_iterations) & ~optimal & ~blocked & ~numerical
    return flow, feasible, optimal, blocked, exhausted, numerical, steps, potentials


class CycleCancelingMinCostFlow(AbstractLinearCombinatorialMethod):
    """Exact integral min-cost flow by deterministic augmentation and cycle canceling."""

    maximum_iterations: int = eqx.field(static=True)
    maximum_vertices: int = eqx.field(static=True)
    maximum_edges: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int = 10_000,
        maximum_vertices: int = 100_000,
        maximum_edges: int = 1_000_000,
    ):
        limits = (maximum_iterations, maximum_vertices, maximum_edges)
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) for value in limits
        ):
            raise TypeError("min-cost-flow resource limits must be positive integers.")
        if any(int(value) <= 0 for value in limits):
            raise ValueError("min-cost-flow resource limits must be positive.")
        self.maximum_iterations = int(maximum_iterations)
        self.maximum_vertices = int(maximum_vertices)
        self.maximum_edges = int(maximum_edges)

    @property
    def method_id(self) -> str:
        return "native-cycle-canceling-min-cost-flow"

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
            ("maximum_iterations", str(self.maximum_iterations)),
            ("maximum_vertices", str(self.maximum_vertices)),
            ("maximum_edges", str(self.maximum_edges)),
        )

    def plan(
        self,
        problem: LinearCombinatorialProblem,
        certification: CombinatorialCertification,
        /,
    ) -> CombinatorialPlan:
        if not isinstance(problem.space, CapacitatedFlowSpace):
            raise TypeError("CycleCancelingMinCostFlow requires CapacitatedFlowSpace.")
        if len(jax.tree_util.tree_leaves(problem.costs)) != 1:
            raise ValueError("CapacitatedFlowSpace requires one edge-cost vector.")
        space = problem.space
        if space.vertex_count > self.maximum_vertices:
            raise ValueError("flow vertex count exceeds maximum_vertices.")
        if space.edge_count > self.maximum_edges:
            raise ValueError("flow edge count exceeds maximum_edges.")
        relaxation_work = max(1, 2 * space.edge_count * space.vertex_count)
        return make_combinatorial_plan(
            problem,
            self,
            certification,
            work_estimate=problem.batch_size * self.maximum_iterations * relaxation_work,
            workspace_elements=problem.batch_size
            * (4 * space.edge_count + 4 * space.vertex_count),
            certificate_kind="residual-network-optimality",
        )

    def solve(
        self,
        problem: LinearCombinatorialProblem,
        plan: CombinatorialPlan,
        /,
    ) -> CombinatorialResult:
        space = problem.space
        if not isinstance(space, CapacitatedFlowSpace):
            raise TypeError("CycleCancelingMinCostFlow requires CapacitatedFlowSpace.")
        raw_costs = jax.tree_util.tree_leaves(problem.costs)[0]
        batch_shape = problem.batch_shape
        flat_batch = problem.batch_size
        costs = raw_costs.reshape((flat_batch, space.edge_count))
        finite = jnp.all(jnp.isfinite(costs), axis=-1)
        safe_costs = jnp.where(jnp.isfinite(costs), costs, 0.0)
        flow, solved, optimal, blocked, exhausted, numerical, steps, potentials = (
            jax.vmap(
                _solve_flow_one,
                in_axes=(0, None, None, None, None, None, None, None, None),
            )(
                safe_costs,
                space.relation.source_indices,
                space.relation.target_indices,
                space.relation.valid,
                space.capacities,
                space.balances,
                space.balanced,
                space.vertex_count,
                self.maximum_iterations,
            )
        )
        decision = FlowDecision(flow.reshape(batch_shape + (space.edge_count,)))
        features = space.encode(decision).astype(raw_costs.dtype)
        objective = problem.objective(features)
        feasibility = space.audit(decision)
        direct_objective = jnp.sum(safe_costs * flow.astype(safe_costs.dtype), axis=-1)
        direct_shaped = direct_objective.reshape(batch_shape)
        tolerance = plan.certification.threshold(objective, direct_shaped)
        objective_consistent = jnp.abs(objective - direct_shaped) <= tolerance
        if space.edge_count:
            residual_sources = jnp.concatenate(
                (space.relation.source_indices, space.relation.target_indices)
            )
            residual_targets = jnp.concatenate(
                (space.relation.target_indices, space.relation.source_indices)
            )
            residual_costs = jnp.concatenate((safe_costs, -safe_costs), axis=-1)
            residual_capacity = jnp.concatenate(
                (space.capacities[None, :] - flow, flow), axis=-1
            )
            residual_valid = jnp.concatenate((space.relation.valid, space.relation.valid))
            reduced = (
                residual_costs
                + potentials[:, residual_sources]
                - potentials[:, residual_targets]
            )
            violation = jnp.where(
                residual_valid[None, :] & (residual_capacity > 0),
                -reduced,
                -jnp.inf,
            )
            dual_residual = jnp.maximum(jnp.max(violation, axis=-1), 0.0)
        else:
            dual_residual = jnp.zeros((flat_batch,), dtype=raw_costs.dtype)
        finite_shaped = finite.reshape(batch_shape)
        solved_shaped = solved.reshape(batch_shape)
        algorithm_optimal = optimal.reshape(batch_shape)
        dual_shaped = dual_residual.reshape(batch_shape)
        certified = (
            finite_shaped
            & algorithm_optimal
            & feasibility.feasible
            & objective_consistent
            & (dual_shaped <= tolerance)
        )
        status = jnp.where(
            ~finite_shaped,
            int(CombinatorialStatus.NONFINITE_INPUT),
            jnp.where(
                numerical.reshape(batch_shape),
                int(CombinatorialStatus.NUMERICAL_FAILURE),
                jnp.where(
                    blocked.reshape(batch_shape) & ~solved_shaped,
                    int(CombinatorialStatus.INFEASIBLE),
                    jnp.where(
                        certified,
                        int(CombinatorialStatus.OPTIMAL),
                        jnp.where(
                            exhausted.reshape(batch_shape),
                            int(CombinatorialStatus.MAXIMUM_STEPS_REACHED),
                            int(CombinatorialStatus.CERTIFICATION_FAILED),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        valid_result = finite_shaped & feasibility.feasible & objective_consistent
        zero = jnp.zeros(batch_shape, dtype=raw_costs.dtype)
        gap_available = certified
        certificate = CombinatorialCertificate(
            finite=finite_shaped,
            feasible=feasibility.feasible,
            objective_consistent=objective_consistent,
            optimality_proven=certified,
            primal_residual=feasibility.residual.astype(raw_costs.dtype),
            dual_residual=dual_shaped,
            absolute_gap=jnp.where(gap_available, zero, jnp.nan),
            relative_gap=jnp.where(gap_available, zero, jnp.nan),
            tie_margin=jnp.full(batch_shape, jnp.nan, dtype=raw_costs.dtype),
            dual_available=algorithm_optimal,
            gap_available=gap_available,
            tie_available=jnp.zeros(batch_shape, dtype=bool),
        )
        provenance = CombinatorialProvenance(
            problem_id=problem.problem_id,
            structure_id=problem.structure_id,
            method_id=self.method_id,
            plan_id=plan.plan_id,
            implementation="phydrax-native-jax",
            tie_policy="lowest-vertex-then-lowest-residual-edge",
            certificate_kind=plan.certificate_kind,
            exact=True,
            signed_costs=True,
            configuration=plan.configuration,
        )
        decision = FlowDecision(jnp.where(valid_result[..., None], decision.flow, 0))
        features = jnp.where(valid_result[..., None], features, jnp.zeros_like(features))
        work = steps * max(1, 2 * space.edge_count * space.vertex_count)
        return CombinatorialResult(
            decision=decision,
            features=features,
            objective_value=jnp.where(valid_result, objective, jnp.nan),
            status=status,
            valid=valid_result,
            certificate=certificate,
            iterations=steps.reshape(batch_shape),
            work=work.reshape(batch_shape),
            provenance=provenance,
            batch_shape=batch_shape,
        )


__all__ = [
    "CapacitatedFlowSpace",
    "CycleCancelingMinCostFlow",
    "FlowDecision",
]
