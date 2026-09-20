#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections import defaultdict
from itertools import permutations, product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import (
    DiagramGraph,
    ExternalLeg,
    ExternalState,
    MomentumRoute,
    PropagatorLine,
    PropagatorSpec,
    VertexInsertion,
    VertexRule,
)


def _double_factorial(value: int, /) -> int:
    if value <= 0:
        return 1
    result = 1
    for factor in range(value, 0, -2):
        result *= factor
    return result


def _perfect_matchings(
    indices: tuple[int, ...], /
) -> tuple[tuple[tuple[int, int], ...], ...]:
    if not indices:
        return ((),)
    first = indices[0]
    results: list[tuple[tuple[int, int], ...]] = []
    for position in range(1, len(indices)):
        partner = indices[position]
        remaining = indices[1:position] + indices[position + 1 :]
        for tail in _perfect_matchings(remaining):
            results.append(((first, partner),) + tail)
    return tuple(results)


def _connected(vertex_count: int, endpoints: tuple[tuple[int, int], ...], /) -> bool:
    if vertex_count == 1:
        return True
    adjacency = [set() for _ in range(vertex_count)]
    for source, target in endpoints:
        adjacency[source].add(target)
        adjacency[target].add(source)
    visited = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                frontier.append(neighbor)
    return len(visited) == vertex_count


def _is_skeleton(vertex_count: int, endpoints: tuple[tuple[int, int], ...], /) -> bool:
    """Use the finite 1PI criterion: no internal line is a bridge."""
    if not _connected(vertex_count, endpoints):
        return False
    for removed, (source, target) in enumerate(endpoints):
        if source == target:
            continue
        retained = endpoints[:removed] + endpoints[removed + 1 :]
        if not _connected(vertex_count, retained):
            return False
    return True


def _route_internal_lines(
    vertex_count: int,
    endpoints: tuple[tuple[int, int], ...],
    external: tuple[ExternalLeg, ...],
    dimension: int,
    /,
) -> tuple[MomentumRoute, ...] | None:
    divergence = np.zeros((vertex_count, dimension + 1), dtype=np.float64)
    for leg in external:
        vector = np.concatenate(
            (np.asarray(leg.route.momentum), np.asarray(leg.route.frequency)[None])
        )
        vertex = int(leg.vertex.removeprefix("raw-v"))
        divergence[vertex] += vector if not leg.incoming else -vector
    required = -divergence

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(vertex_count)]
    for edge, (source, target) in enumerate(endpoints):
        if source != target:
            adjacency[source].append((target, edge))
            adjacency[target].append((source, edge))
    parent = np.full((vertex_count,), -1, dtype=np.int64)
    parent_edge = np.full((vertex_count,), -1, dtype=np.int64)
    order = [0]
    visited = {0}
    for current in order:
        for neighbor, edge in adjacency[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                parent_edge[neighbor] = edge
                order.append(neighbor)
    if len(visited) != vertex_count:
        return None

    routed = np.zeros((len(endpoints), dimension + 1), dtype=np.float64)
    balance = required.copy()
    for vertex in reversed(order[1:]):
        edge = int(parent_edge[vertex])
        source, target = endpoints[edge]
        incidence_at_vertex = 1.0 if source == vertex else -1.0
        routed[edge] = balance[vertex] / incidence_at_vertex
        parent_vertex = int(parent[vertex])
        incidence_at_parent = 1.0 if source == parent_vertex else -1.0
        balance[parent_vertex] -= incidence_at_parent * routed[edge]
    if np.max(np.abs(balance[0])) > 1.0e-10:
        return None
    return tuple(MomentumRoute(vector[:-1], vector[-1]) for vector in routed)


class DiagramGenerationEvidence(StrictModule, NonTrainableState):
    candidates_considered: Array
    disconnected_rejected: Array
    conservation_rejected: Array
    duplicate_rejected: Array
    diagrams_generated: Array
    complete: Array
    status: Array


class DiagramEnsemble(StrictModule, NonTrainableState):
    """A canonical, duplicate-free finite diagram family."""

    diagrams: tuple[DiagramGraph, ...]
    evidence: DiagramGenerationEvidence
    maximum_order: int = eqx.field(static=True)
    ensemble_id: str = eqx.field(static=True)

    def order_histogram(self, /) -> Array:
        histogram = jnp.zeros((self.maximum_order + 1,), dtype=jnp.int32)
        for diagram in self.diagrams:
            histogram = histogram.at[diagram.order].add(1)
        return histogram


class DiagramGeneratorPlan(StrictModule, NonTrainableState):
    """Immutable finite enumeration policy; preparation binds external states."""

    propagators: tuple[PropagatorSpec, ...]
    vertex_rules: tuple[VertexRule, ...]
    maximum_vertices: int = eqx.field(static=True)
    maximum_diagrams: int = eqx.field(static=True)
    maximum_candidates: int = eqx.field(static=True)
    maximum_canonical_permutations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        propagators: tuple[PropagatorSpec, ...],
        vertex_rules: tuple[VertexRule, ...],
        /,
        *,
        maximum_vertices: int = 2,
        maximum_diagrams: int = 1_024,
        maximum_candidates: int = 250_000,
        maximum_canonical_permutations: int = 40_320,
    ):
        propagators_, rules_ = tuple(propagators), tuple(vertex_rules)
        if not propagators_ or any(
            not isinstance(item, PropagatorSpec) for item in propagators_
        ):
            raise ValueError("propagators must contain PropagatorSpec values.")
        if not rules_ or any(not isinstance(item, VertexRule) for item in rules_):
            raise ValueError("vertex_rules must contain VertexRule values.")
        field_ids = tuple(item.field.field_id for item in propagators_)
        if len(set(field_ids)) != len(field_ids):
            raise ValueError("A generator needs exactly one propagator per field.")
        available = set(field_ids)
        if any(
            field.field_id not in available for rule in rules_ for field in rule.fields
        ):
            raise ValueError("Every vertex field needs a supplied propagator.")
        limits = (
            int(maximum_vertices),
            int(maximum_diagrams),
            int(maximum_candidates),
            int(maximum_canonical_permutations),
        )
        if any(value <= 0 for value in limits):
            raise ValueError("Generator resource limits must be positive.")
        if math.factorial(limits[0]) > limits[3]:
            raise ValueError(
                "maximum_vertices exceeds maximum_canonical_permutations capacity."
            )
        self.propagators = propagators_
        self.vertex_rules = rules_
        self.maximum_vertices = limits[0]
        self.maximum_diagrams = limits[1]
        self.maximum_candidates = limits[2]
        self.maximum_canonical_permutations = limits[3]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "diagram-generator-plan",
                "propagators": [item.propagator_id for item in propagators_],
                "vertex_rules": [item.rule_id for item in rules_],
                "limits": limits,
            }
        )

    def prepare(
        self, external_states: tuple[ExternalState, ...], /
    ) -> "PreparedDiagramGenerator":
        external_ = tuple(external_states)
        if any(not isinstance(item, ExternalState) for item in external_):
            raise TypeError("external_states must contain ExternalState values.")
        if len({item.label for item in external_}) != len(external_):
            raise ValueError("External-state labels must be unique.")
        if external_:
            dimension = external_[0].route.dimension
            if any(item.route.dimension != dimension for item in external_):
                raise ValueError("External states must share one momentum dimension.")
        else:
            dimension = 1
        largest_ports = self.maximum_vertices * max(
            len(rule.fields) for rule in self.vertex_rules
        )
        if len(external_) > largest_ports:
            raise ValueError("External states exceed all available vertex ports.")
        remaining = largest_ports - len(external_)
        pairable_remaining = remaining if remaining % 2 == 0 else remaining - 1
        pairing_bound = _double_factorial(pairable_remaining - 1)
        assignment_bound = math.factorial(largest_ports) // math.factorial(
            largest_ports - len(external_)
        )
        sequence_bound = sum(
            len(self.vertex_rules) ** count
            for count in range(1, self.maximum_vertices + 1)
        )
        candidate_bound = sequence_bound * assignment_bound * pairing_bound
        if candidate_bound > self.maximum_candidates:
            raise ValueError(
                "Prepared enumeration bound exceeds maximum_candidates before allocation."
            )
        return PreparedDiagramGenerator(self, external_, dimension, candidate_bound)


class PreparedDiagramGenerator(StrictModule, NonTrainableState):
    """Prepared external-state binding for fixed-order and skeleton generation."""

    plan: DiagramGeneratorPlan
    external_states: tuple[ExternalState, ...]
    dimension: int = eqx.field(static=True)
    candidate_bound: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DiagramGeneratorPlan,
        external_states: tuple[ExternalState, ...],
        dimension: int,
        candidate_bound: int,
        /,
    ):
        self.plan = plan
        self.external_states = external_states
        self.dimension = int(dimension)
        self.candidate_bound = int(candidate_bound)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-diagram-generator",
                "plan": plan.plan_id,
                "external": [
                    {
                        "field": item.field.field_id,
                        "route": item.route.route_id,
                        "incoming": item.incoming,
                        "wavefunction": (
                            complex(np.asarray(item.wavefunction)).real,
                            complex(np.asarray(item.wavefunction)).imag,
                        ),
                    }
                    for item in external_states
                ],
            }
        )

    def generate_fixed_order(self, order: int, /) -> DiagramEnsemble:
        return self._generate(order, skeleton=False)

    def generate_skeletons(self, order: int, /) -> DiagramEnsemble:
        return self._generate(order, skeleton=True)

    def _generate(self, order: int, /, *, skeleton: bool) -> DiagramEnsemble:
        requested = int(order)
        if requested <= 0:
            raise ValueError("order must be positive.")
        by_field = {item.field.field_id: item for item in self.plan.propagators}
        diagrams: dict[str, DiagramGraph] = {}
        considered = disconnected = conservation = duplicates = 0

        for vertex_count in range(1, self.plan.maximum_vertices + 1):
            for rules in product(self.plan.vertex_rules, repeat=vertex_count):
                if sum(rule.perturbative_order for rule in rules) != requested:
                    continue
                ports = tuple(
                    (vertex, local, field.field_id)
                    for vertex, rule in enumerate(rules)
                    for local, field in enumerate(rule.fields)
                )
                if len(self.external_states) > len(ports):
                    continue
                for assigned_ports in permutations(
                    range(len(ports)), len(self.external_states)
                ):
                    if any(
                        self.external_states[index].field.field_id != ports[port][2]
                        for index, port in enumerate(assigned_ports)
                    ):
                        continue
                    used = set(assigned_ports)
                    remaining_by_field: dict[str, list[int]] = defaultdict(list)
                    for port, (_, _, field_id) in enumerate(ports):
                        if port not in used:
                            remaining_by_field[field_id].append(port)
                    if any(len(indices) % 2 for indices in remaining_by_field.values()):
                        continue
                    fields = tuple(sorted(remaining_by_field))
                    matching_families = tuple(
                        _perfect_matchings(tuple(remaining_by_field[field_id]))
                        for field_id in fields
                    )
                    for matching_choice in product(*matching_families):
                        considered += 1
                        if considered > self.plan.maximum_candidates:
                            raise RuntimeError(
                                "Enumeration exceeded its proven maximum_candidates bound."
                            )
                        pairings = tuple(
                            pair for family in matching_choice for pair in family
                        )
                        endpoints = tuple(
                            (ports[left][0], ports[right][0]) for left, right in pairings
                        )
                        if not _connected(vertex_count, endpoints):
                            disconnected += 1
                            continue
                        if skeleton and not _is_skeleton(vertex_count, endpoints):
                            disconnected += 1
                            continue
                        vertices = tuple(
                            VertexInsertion(f"raw-v{index}", rule)
                            for index, rule in enumerate(rules)
                        )
                        external = tuple(
                            ExternalLeg.from_state(state, f"raw-v{ports[port][0]}")
                            for state, port in zip(
                                self.external_states, assigned_ports, strict=True
                            )
                        )
                        routes = _route_internal_lines(
                            vertex_count,
                            endpoints,
                            external,
                            self.dimension,
                        )
                        if routes is None:
                            conservation += 1
                            continue
                        lines = tuple(
                            PropagatorLine(
                                f"raw-p{index}",
                                by_field[ports[left][2]],
                                f"raw-v{ports[left][0]}",
                                f"raw-v{ports[right][0]}",
                                routes[index],
                            )
                            for index, (left, right) in enumerate(pairings)
                        )
                        diagram = DiagramGraph(
                            vertices,
                            lines,
                            external,
                            maximum_canonical_permutations=(
                                self.plan.maximum_canonical_permutations
                            ),
                        )
                        if not bool(diagram.evidence.successful):
                            conservation += 1
                            continue
                        if diagram.graph_id in diagrams:
                            duplicates += 1
                            continue
                        if len(diagrams) >= self.plan.maximum_diagrams:
                            raise ValueError(
                                "Generated diagrams exceed maximum_diagrams capacity."
                            )
                        diagrams[diagram.graph_id] = diagram

        canonical = tuple(diagrams[key] for key in sorted(diagrams))
        evidence = DiagramGenerationEvidence(
            jnp.asarray(considered, dtype=jnp.int32),
            jnp.asarray(disconnected, dtype=jnp.int32),
            jnp.asarray(conservation, dtype=jnp.int32),
            jnp.asarray(duplicates, dtype=jnp.int32),
            jnp.asarray(len(canonical), dtype=jnp.int32),
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
        )
        return DiagramEnsemble(
            canonical,
            evidence,
            requested,
            canonical_fingerprint(
                {
                    "kind": "diagram-ensemble",
                    "prepared": self.prepared_id,
                    "order": requested,
                    "skeleton": skeleton,
                    "diagrams": [item.graph_id for item in canonical],
                }
            ),
        )


__all__ = [
    "DiagramEnsemble",
    "DiagramGenerationEvidence",
    "DiagramGeneratorPlan",
    "PreparedDiagramGenerator",
]
