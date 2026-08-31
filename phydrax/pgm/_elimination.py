#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from math import prod
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._probability import AbstractProbabilityLaw
from .._strict import StrictModule
from ._kernel import FactorGraphResourcePolicy
from ._model import (
    DiscreteFactorGraph,
    factor_graph_contains,
    factor_graph_log_score,
    factor_group_dense_tables,
    pack_evidence,
    VariableStateValues,
)


class VariableEliminationPlan(StrictModule):
    """Bounded exact elimination order with induced-clique resource evidence."""

    graph: DiscreteFactorGraph
    order: tuple[int, ...] = eqx.field(static=True)
    induced_scopes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    treewidth: int = eqx.field(static=True)
    maximum_workspace_elements: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class VariableEliminationResult(StrictModule):
    """Exact normalizer, variable marginals, and MAP assignment from elimination."""

    log_normalizer: Array
    variable_probabilities: VariableStateValues
    evidence: VariableStateValues
    map_assignment: Array
    map_log_score: Array
    valid: Array
    plan: VariableEliminationPlan

    @property
    def successful(self) -> Array:
        return self.valid


class JunctionTreePlan(StrictModule):
    """Running-intersection clique tree induced by one elimination plan."""

    elimination: VariableEliminationPlan
    cliques: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    separators: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    parents: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class JunctionTreeResult(StrictModule):
    """Exact variable and clique beliefs over a validated running-intersection tree."""

    elimination: VariableEliminationResult
    clique_probabilities: tuple[Array, ...]
    plan: JunctionTreePlan
    valid: Array


class VariableEliminationMethod(StrictModule):
    """Exact log-semiring or max-plus elimination with an explicit order policy."""

    ordering: Literal["min-fill", "min-degree", "given"] = eqx.field(static=True)
    order: tuple[int, ...] | None = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        ordering: Literal["min-fill", "min-degree", "given"] = "min-fill",
        order: Sequence[int] | None = None,
    ):
        if ordering not in ("min-fill", "min-degree", "given"):
            raise ValueError("Unknown elimination ordering policy.")
        resolved = None if order is None else tuple(int(value) for value in order)
        if ordering == "given" and resolved is None:
            raise ValueError("A given elimination order is required.")
        if ordering != "given" and resolved is not None:
            raise ValueError("order is accepted only when ordering='given'.")
        self.ordering = ordering
        self.order = resolved
        self.method_id = f"variable-elimination-{ordering}"


def _initial_scopes(graph: DiscreteFactorGraph) -> list[set[int]]:
    scopes: list[set[int]] = []
    for scope in graph.factor_scopes:
        scopes.extend({int(value) for value in row} for row in np.asarray(scope))
    scopes.extend({variable} for variable in range(graph.num_variables))
    return scopes


def _choose_order(graph: DiscreteFactorGraph, method: VariableEliminationMethod):
    if method.ordering == "given":
        assert method.order is not None
        if sorted(method.order) != list(range(graph.num_variables)):
            raise ValueError("Elimination order must be a permutation of all variables.")
        order = method.order
    else:
        active = _initial_scopes(graph)
        remaining = set(range(graph.num_variables))
        chosen: list[int] = []
        while remaining:
            candidates = []
            for variable in remaining:
                neighbors = set().union(
                    *(scope - {variable} for scope in active if variable in scope)
                )
                degree = len(neighbors)
                existing = sum(
                    1
                    for scope in active
                    for left in neighbors
                    for right in neighbors
                    if left < right and left in scope and right in scope
                )
                pairs = degree * (degree - 1) // 2
                fill = pairs - min(existing, pairs)
                key = (
                    (fill, degree, variable)
                    if method.ordering == "min-fill"
                    else (degree, fill, variable)
                )
                candidates.append((key, variable, neighbors))
            _, variable, neighbors = min(candidates)
            chosen.append(variable)
            active = [scope for scope in active if variable not in scope]
            if neighbors:
                active.append(neighbors)
            remaining.remove(variable)
        order = tuple(chosen)

    active = _initial_scopes(graph)
    induced: list[tuple[int, ...]] = []
    maximum = 1
    cards = np.asarray(graph.cardinalities, dtype=np.int64)
    for variable in order:
        involved = [scope for scope in active if variable in scope]
        union = set().union(*involved) if involved else {variable}
        clique = tuple(sorted(union))
        induced.append(clique)
        maximum = max(maximum, prod(int(cards[index]) for index in clique))
        active = [scope for scope in active if variable not in scope]
        remainder = union - {variable}
        if remainder:
            active.append(remainder)
    treewidth = max((len(scope) - 1 for scope in induced), default=0)
    return order, tuple(induced), treewidth, maximum


def plan_variable_elimination(
    graph: DiscreteFactorGraph,
    method: VariableEliminationMethod | None = None,
    /,
    *,
    resources: FactorGraphResourcePolicy | None = None,
) -> VariableEliminationPlan:
    """Plan exact elimination and reject excessive treewidth/workspace before execution."""
    if not isinstance(graph, DiscreteFactorGraph):
        raise TypeError("graph must be a DiscreteFactorGraph.")
    selected = VariableEliminationMethod() if method is None else method
    if not isinstance(selected, VariableEliminationMethod):
        raise TypeError("method must be VariableEliminationMethod or None.")
    policy = FactorGraphResourcePolicy() if resources is None else resources
    if not isinstance(policy, FactorGraphResourcePolicy):
        raise TypeError("resources must be FactorGraphResourcePolicy or None.")
    order, scopes, treewidth, maximum = _choose_order(graph, selected)
    if treewidth > policy.maximum_treewidth:
        raise ValueError(
            f"Induced treewidth {treewidth} exceeds maximum_treewidth={policy.maximum_treewidth}."
        )
    if maximum > policy.maximum_elimination_elements:
        raise ValueError(
            f"Elimination workspace {maximum} exceeds maximum_elimination_elements="
            f"{policy.maximum_elimination_elements}."
        )
    plan_id = canonical_fingerprint(
        {
            "kind": "variable-elimination-plan",
            "structure_id": graph.structure_id,
            "method": selected.method_id,
            "order": list(order),
            "scopes": [list(scope) for scope in scopes],
            "resources": policy.policy_id,
        }
    )
    return VariableEliminationPlan(
        graph=graph,
        order=order,
        induced_scopes=scopes,
        treewidth=treewidth,
        maximum_workspace_elements=maximum,
        plan_id=plan_id,
    )


def _align(
    table: Array, scope: tuple[int, ...], union: tuple[int, ...], cards: tuple[int, ...]
):
    present = tuple(variable for variable in union if variable in scope)
    permutation = tuple(scope.index(variable) for variable in present)
    transposed = (
        jnp.transpose(table, permutation)
        if permutation != tuple(range(len(scope)))
        else table
    )
    shape = tuple(
        cards[union.index(variable)] if variable in scope else 1 for variable in union
    )
    return transposed.reshape(shape)


def _factor_tables(graph: DiscreteFactorGraph, evidence: Array):
    factors: list[tuple[tuple[int, ...], Array]] = []
    for group_index, scope in enumerate(graph.factor_scopes):
        tables = factor_group_dense_tables(graph, group_index)
        for factor, row in enumerate(np.asarray(scope, dtype=np.int32)):
            factors.append((tuple(int(value) for value in row), tables[factor]))
    offsets = np.asarray(graph.variable_state_offsets)
    for variable in range(graph.num_variables):
        factors.append(
            (
                (variable,),
                evidence[offsets[variable] : offsets[variable + 1]],
            )
        )
    return factors


def _eliminate(
    plan: VariableEliminationPlan,
    evidence: Array,
    /,
    *,
    mode: Literal["sum", "max"],
) -> Array:
    graph = plan.graph
    cards_by_variable = tuple(int(value) for value in np.asarray(graph.cardinalities))
    factors = _factor_tables(graph, evidence)
    constants: list[Array] = []
    for variable in plan.order:
        involved = [(scope, table) for scope, table in factors if variable in scope]
        factors = [(scope, table) for scope, table in factors if variable not in scope]
        if not involved:
            continue
        union = tuple(sorted(set().union(*(set(scope) for scope, _ in involved))))
        cards = tuple(cards_by_variable[index] for index in union)
        combined = jnp.asarray(0.0)
        for scope, table in involved:
            combined = combined + _align(table, scope, union, cards)
        axis = union.index(variable)
        reduced = (
            jsp.special.logsumexp(combined, axis=axis)
            if mode == "sum"
            else jnp.max(combined, axis=axis)
        )
        remainder = tuple(index for index in union if index != variable)
        if remainder:
            factors.append((remainder, reduced))
        else:
            constants.append(reduced)
    for scope, table in factors:
        if scope:
            axes = tuple(range(table.ndim))
            table = (
                jsp.special.logsumexp(table, axis=axes)
                if mode == "sum"
                else jnp.max(table, axis=axes)
            )
        constants.append(table)
    return sum(constants, start=jnp.asarray(0.0))


def _clamp_evidence(
    graph: DiscreteFactorGraph,
    evidence: Array,
    assignments: dict[int, int | Array],
):
    result = evidence
    offsets = np.asarray(graph.variable_state_offsets)
    for variable, state in assignments.items():
        start, stop = int(offsets[variable]), int(offsets[variable + 1])
        mask = jnp.arange(stop - start) == jnp.asarray(state)
        result = result.at[start:stop].set(jnp.where(mask, result[start:stop], -jnp.inf))
    return result


def variable_elimination(
    plan: VariableEliminationPlan,
    /,
    *,
    evidence: ArrayLike | None = None,
) -> VariableEliminationResult:
    """Run exact bounded elimination, repeated conditionals, and deterministic MAP decoding."""
    if not isinstance(plan, VariableEliminationPlan):
        raise TypeError("plan must be VariableEliminationPlan.")
    graph = plan.graph
    packed = (
        pack_evidence(graph, evidence).values
        if evidence is not None
        else pack_evidence(graph).values
    )
    if packed.shape != (graph.num_variable_states,):
        raise ValueError(
            "Variable elimination currently requires one unbatched evidence vector."
        )
    log_normalizer = _eliminate(plan, packed, mode="sum")
    valid = jnp.isfinite(log_normalizer)
    probabilities: list[Array] = []
    for variable, cardinality in enumerate(np.asarray(graph.cardinalities)):
        values = jnp.stack(
            [
                _eliminate(
                    plan,
                    _clamp_evidence(graph, packed, {variable: state}),
                    mode="sum",
                )
                for state in range(int(cardinality))
            ]
        )
        probabilities.append(jnp.where(valid, jnp.exp(values - log_normalizer), 0.0))
    flat_probabilities = (
        jnp.concatenate(probabilities) if probabilities else jnp.zeros((0,))
    )

    chosen: dict[int, Array] = {}
    for variable, cardinality in enumerate(np.asarray(graph.cardinalities)):
        values = jnp.stack(
            [
                _eliminate(
                    plan,
                    _clamp_evidence(graph, packed, {**chosen, variable: state}),
                    mode="max",
                )
                for state in range(int(cardinality))
            ]
        )
        chosen[variable] = jnp.argmax(values).astype(jnp.int32)
    assignment = (
        jnp.stack([chosen[index] for index in range(graph.num_variables)])
        if chosen
        else jnp.zeros((0,), dtype=jnp.int32)
    )
    evidence_indices = graph.variable_state_offsets[:-1] + assignment
    map_score = factor_graph_log_score(graph, assignment) + jnp.sum(
        packed[evidence_indices]
    )
    return VariableEliminationResult(
        log_normalizer=log_normalizer,
        variable_probabilities=VariableStateValues(
            flat_probabilities, structure_id=graph.structure_id
        ),
        evidence=VariableStateValues(packed, structure_id=graph.structure_id),
        map_assignment=assignment,
        map_log_score=map_score,
        valid=valid,
        plan=plan,
    )


def plan_junction_tree(plan: VariableEliminationPlan, /) -> JunctionTreePlan:
    """Build a deterministic running-intersection tree from induced elimination cliques."""
    if not isinstance(plan, VariableEliminationPlan):
        raise TypeError("plan must be VariableEliminationPlan.")
    cliques: list[tuple[int, ...]] = []
    for scope in plan.induced_scopes:
        if (
            not any(set(scope) < set(other) for other in plan.induced_scopes)
            and scope not in cliques
        ):
            cliques.append(scope)
    clique_count = len(cliques)
    union_parent = list(range(clique_count))

    def find(index):
        while union_parent[index] != index:
            union_parent[index] = union_parent[union_parent[index]]
            index = union_parent[index]
        return index

    weighted_edges = sorted(
        (
            (
                -len(set(cliques[left]) & set(cliques[right])),
                left,
                right,
            )
            for left in range(clique_count)
            for right in range(left + 1, clique_count)
        )
    )
    adjacency: list[list[int]] = [[] for _ in range(clique_count)]
    selected_edges = 0
    for _negative_weight, left, right in weighted_edges:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        union_parent[right_root] = left_root
        adjacency[left].append(right)
        adjacency[right].append(left)
        selected_edges += 1
        if selected_edges == clique_count - 1:
            break

    parents = [-2] * clique_count
    separators: list[tuple[int, ...]] = [()] * clique_count
    parents[0] = -1
    pending = deque([0])
    while pending:
        parent = pending.popleft()
        for child in sorted(adjacency[parent]):
            if parents[child] != -2:
                continue
            parents[child] = parent
            separators[child] = tuple(sorted(set(cliques[child]) & set(cliques[parent])))
            pending.append(child)

    for variable in range(plan.graph.num_variables):
        containing = {index for index, clique in enumerate(cliques) if variable in clique}
        if len(containing) < 2:
            continue
        reached = {min(containing)}
        pending = deque(reached)
        while pending:
            current = pending.popleft()
            for neighbor in adjacency[current]:
                if neighbor in containing and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        if reached != containing:
            raise RuntimeError("Elimination cliques violate running intersection.")
    plan_id = canonical_fingerprint(
        {
            "kind": "junction-tree-plan",
            "elimination": plan.plan_id,
            "cliques": [list(scope) for scope in cliques],
            "parents": parents,
        }
    )
    return JunctionTreePlan(
        elimination=plan,
        cliques=tuple(cliques),
        separators=tuple(separators),
        parents=tuple(parents),
        plan_id=plan_id,
    )


def junction_tree_calibrate(
    plan: JunctionTreePlan,
    /,
    *,
    evidence: ArrayLike | None = None,
) -> JunctionTreeResult:
    """Return exact clique beliefs by bounded elimination-conditioned calibration."""
    if not isinstance(plan, JunctionTreePlan):
        raise TypeError("plan must be JunctionTreePlan.")
    result = variable_elimination(plan.elimination, evidence=evidence)
    graph = plan.elimination.graph
    packed = (
        pack_evidence(graph, evidence).values
        if evidence is not None
        else pack_evidence(graph).values
    )
    clique_probabilities: list[Array] = []
    cards = np.asarray(graph.cardinalities)
    for clique in plan.cliques:
        shape = tuple(int(cards[variable]) for variable in clique)
        values = []
        for flat_index in range(prod(shape)):
            configuration = np.unravel_index(flat_index, shape)
            clamped = _clamp_evidence(
                graph,
                packed,
                {variable: state for variable, state in zip(clique, configuration)},
            )
            values.append(_eliminate(plan.elimination, clamped, mode="sum"))
        log_values = jnp.stack(values).reshape(shape)
        clique_probabilities.append(jnp.exp(log_values - result.log_normalizer))
    return JunctionTreeResult(
        elimination=result,
        clique_probabilities=tuple(clique_probabilities),
        plan=plan,
        valid=result.valid,
    )


class NormalizedFactorGraphLaw(AbstractProbabilityLaw):
    """Exact normalized law backed by bounded variable elimination."""

    plan: VariableEliminationPlan
    result: VariableEliminationResult
    evidence: Array

    def __init__(
        self,
        plan: VariableEliminationPlan,
        result: VariableEliminationResult | None = None,
        /,
        *,
        evidence: ArrayLike | None = None,
    ):
        if not isinstance(plan, VariableEliminationPlan):
            raise TypeError("plan must be VariableEliminationPlan.")
        if result is not None and not isinstance(result, VariableEliminationResult):
            raise TypeError("result must be VariableEliminationResult or None.")
        if result is not None and result.plan.plan_id != plan.plan_id:
            raise ValueError("result must be the exact result from the supplied plan.")
        if result is None:
            packed = pack_evidence(plan.graph, evidence).values
            resolved = variable_elimination(plan, evidence=packed)
        else:
            resolved = result
            packed = (
                resolved.evidence.values
                if evidence is None
                else pack_evidence(plan.graph, evidence).values
            )
        if not bool(jnp.array_equal(resolved.evidence.values, packed)):
            raise ValueError("result and evidence must describe the same normalized law.")
        if not bool(resolved.valid):
            raise ValueError("Cannot normalize an infeasible factor graph.")
        self.plan = plan
        self.result = resolved
        self.evidence = packed

    @property
    def event_shape(self) -> tuple[int, ...]:
        return (self.plan.graph.num_variables,)

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return ()

    @property
    def density_measure_kind(self):
        return "counting"

    def contains(self, value: ArrayLike, /) -> Array:
        states = jnp.asarray(value)
        support = factor_graph_contains(self.plan.graph, states)
        score = factor_graph_log_score(self.plan.graph, states)
        return support & jnp.isfinite(score)

    def log_prob(self, value: ArrayLike, /) -> Array:
        states = jnp.asarray(value, dtype=jnp.int32)
        score = factor_graph_log_score(self.plan.graph, states)
        indices = self.plan.graph.variable_state_offsets[:-1] + states
        return jnp.where(
            self.contains(states),
            score + jnp.sum(self.evidence[indices], axis=-1) - self.result.log_normalizer,
            -jnp.inf,
        )

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        count = prod(tuple(int(size) for size in sample_shape)) if sample_shape else 1
        keys = jr.split(key, count)

        def one_sample(sample_key):
            chosen: dict[int, Array] = {}
            state = jnp.zeros((self.plan.graph.num_variables,), dtype=jnp.int32)
            for variable, cardinality in enumerate(
                np.asarray(self.plan.graph.cardinalities)
            ):
                log_values = jnp.stack(
                    [
                        _eliminate(
                            self.plan,
                            _clamp_evidence(
                                self.plan.graph,
                                self.evidence,
                                {**chosen, variable: candidate},
                            ),
                            mode="sum",
                        )
                        for candidate in range(int(cardinality))
                    ]
                )
                sample_key, subkey = jr.split(sample_key)
                selected = jr.categorical(subkey, log_values).astype(jnp.int32)
                chosen[variable] = selected
                state = state.at[variable].set(selected)
            return state

        samples = jax.vmap(one_sample)(keys)
        return (
            samples.reshape(tuple(sample_shape) + self.event_shape)
            if sample_shape
            else samples[0]
        )


__all__ = [
    "JunctionTreePlan",
    "JunctionTreeResult",
    "NormalizedFactorGraphLaw",
    "VariableEliminationMethod",
    "VariableEliminationPlan",
    "VariableEliminationResult",
    "junction_tree_calibrate",
    "plan_junction_tree",
    "plan_variable_elimination",
    "variable_elimination",
]
