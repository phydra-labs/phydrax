#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DiagramGraph


class DiagramMonteCarloState(StrictModule, NonTrainableState):
    diagram_index: Array
    worm_open: Array
    worm_source: Array
    worm_target: Array
    order_histogram: Array
    proposed_moves: Array
    accepted_moves: Array
    phase_sum: Array
    phase_samples: Array
    step: Array


class DiagramMonteCarloEvidence(StrictModule, NonTrainableState):
    acceptance_rates: Array
    detailed_balance_residual: Array
    mean_phase: Array
    average_sign: Array
    signed_effective_samples: Array
    phase_standard_error: Array
    finite: Array
    successful: Array
    status: Array


class DiagramMonteCarloResult(StrictModule, NonTrainableState):
    state: DiagramMonteCarloState
    order_probabilities: Array
    evidence: DiagramMonteCarloEvidence
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class DiagramMonteCarloPlan(StrictModule, NonTrainableState):
    """Immutable birth/death/worm proposal and fixed-shape resource policy."""

    move_probabilities: Array
    worm_fugacity: Array
    steps: int = eqx.field(static=True)
    maximum_diagrams: int = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        steps: int,
        insertion_probability: float = 0.4,
        removal_probability: float = 0.4,
        worm_probability: float = 0.2,
        worm_fugacity: float = 0.5,
        maximum_diagrams: int = 4_096,
        maximum_neighbors: int = 256,
        maximum_order: int = 32,
    ):
        probabilities = np.asarray(
            (insertion_probability, removal_probability, worm_probability),
            dtype=np.float64,
        )
        fugacity = float(worm_fugacity)
        limits = (
            int(steps),
            int(maximum_diagrams),
            int(maximum_neighbors),
            int(maximum_order),
        )
        if (
            np.any(~np.isfinite(probabilities))
            or np.any(probabilities <= 0.0)
            or not np.isclose(np.sum(probabilities), 1.0)
        ):
            raise ValueError(
                "Move probabilities must be positive, finite, and sum to one."
            )
        if not np.isfinite(fugacity) or fugacity <= 0.0:
            raise ValueError("worm_fugacity must be finite and positive.")
        if any(value <= 0 for value in limits):
            raise ValueError("Monte Carlo steps and capacities must be positive.")
        self.move_probabilities = jnp.asarray(probabilities)
        self.worm_fugacity = jnp.asarray(fugacity)
        self.steps, self.maximum_diagrams, self.maximum_neighbors, self.maximum_order = (
            limits
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "diagram-monte-carlo-plan",
                "steps": limits[0],
                "probabilities": tuple(float(value) for value in probabilities),
                "worm_fugacity": fugacity,
                "limits": limits[1:],
            }
        )

    def prepare(
        self,
        diagrams: tuple[DiagramGraph, ...],
        complex_weights: ArrayLike,
        /,
    ) -> "PreparedDiagramMonteCarlo":
        diagrams_ = tuple(diagrams)
        weights = np.asarray(complex_weights, dtype=np.complex128)
        count = len(diagrams_)
        if count == 0 or any(not isinstance(item, DiagramGraph) for item in diagrams_):
            raise ValueError("diagrams must contain DiagramGraph values.")
        if count > self.maximum_diagrams:
            raise ValueError(
                "Diagram catalog exceeds maximum_diagrams before allocation."
            )
        if len({item.graph_id for item in diagrams_}) != count:
            raise ValueError("Diagram catalog graph identities must be unique.")
        if weights.shape != (count,) or not np.all(np.isfinite(weights)):
            raise ValueError("complex_weights must be one finite value per diagram.")
        magnitudes = np.abs(weights)
        if np.any(magnitudes <= 0.0):
            raise ValueError("Diagram Monte Carlo requires nonzero absolute weights.")
        orders = np.asarray([item.order for item in diagrams_], dtype=np.int32)
        if np.any(orders > self.maximum_order):
            raise ValueError(
                "Diagram order exceeds maximum_order before histogram allocation."
            )
        vertex_counts = np.asarray(
            [len(item.vertices) for item in diagrams_], dtype=np.int32
        )
        insertion_lists = [np.flatnonzero(orders == order + 1) for order in orders]
        removal_lists = [np.flatnonzero(orders == order - 1) for order in orders]
        largest = max(
            max((len(values) for values in insertion_lists), default=0),
            max((len(values) for values in removal_lists), default=0),
        )
        if largest > self.maximum_neighbors:
            raise ValueError("Birth/death adjacency exceeds maximum_neighbors.")
        insert = np.zeros((count, self.maximum_neighbors), dtype=np.int32)
        remove = np.zeros((count, self.maximum_neighbors), dtype=np.int32)
        for index, values in enumerate(insertion_lists):
            insert[index, : len(values)] = values
            insert[index, len(values) :] = index
        for index, values in enumerate(removal_lists):
            remove[index, : len(values)] = values
            remove[index, len(values) :] = index
        insert_count = np.asarray(
            [len(values) for values in insertion_lists], dtype=np.int32
        )
        remove_count = np.asarray(
            [len(values) for values in removal_lists], dtype=np.int32
        )

        probabilities = np.asarray(self.move_probabilities)
        maximum_balance = 0.0
        for source, targets in enumerate(insertion_lists):
            for target in targets:
                forward_q = probabilities[0] / len(targets)
                reverse_q = probabilities[1] / len(removal_lists[int(target)])
                acceptance = min(
                    1.0,
                    magnitudes[int(target)]
                    * reverse_q
                    / (magnitudes[source] * forward_q),
                )
                reverse_acceptance = min(
                    1.0,
                    magnitudes[source]
                    * forward_q
                    / (magnitudes[int(target)] * reverse_q),
                )
                left = magnitudes[source] * forward_q * acceptance
                right = magnitudes[int(target)] * reverse_q * reverse_acceptance
                maximum_balance = max(maximum_balance, abs(left - right))
        for index, vertex_count in enumerate(vertex_counts):
            pairs = int(vertex_count) ** 2
            opening = min(1.0, float(self.worm_fugacity) * pairs)
            closing = min(1.0, 1.0 / (float(self.worm_fugacity) * pairs))
            left = magnitudes[index] * probabilities[2] / pairs * opening
            right = (
                magnitudes[index] * float(self.worm_fugacity) * probabilities[2] * closing
            )
            maximum_balance = max(maximum_balance, abs(left - right))

        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-diagram-monte-carlo",
                "plan": self.plan_id,
                "diagrams": [item.graph_id for item in diagrams_],
                "weights": array_tree_fingerprint(weights),
            }
        )
        return PreparedDiagramMonteCarlo(
            jnp.asarray(orders),
            jnp.asarray(vertex_counts),
            jnp.asarray(weights),
            jnp.asarray(magnitudes),
            jnp.asarray(weights / magnitudes),
            jnp.asarray(insert),
            jnp.asarray(remove),
            jnp.asarray(insert_count),
            jnp.asarray(remove_count),
            self.move_probabilities,
            self.worm_fugacity,
            jnp.asarray(maximum_balance),
            self.steps,
            self.maximum_neighbors,
            self.maximum_order,
            self.plan_id,
            prepared_id,
        )


class PreparedDiagramMonteCarlo(StrictModule, NonTrainableState):
    """Fixed catalog and adjacency tables for JAX birth/death/worm transitions."""

    orders: Array
    vertex_counts: Array
    complex_weights: Array
    magnitudes: Array
    phases: Array
    insert_neighbors: Array
    remove_neighbors: Array
    insert_count: Array
    remove_count: Array
    move_probabilities: Array
    worm_fugacity: Array
    detailed_balance_residual: Array
    steps: int = eqx.field(static=True)
    maximum_neighbors: int = eqx.field(static=True)
    maximum_order: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        orders: Array,
        vertex_counts: Array,
        complex_weights: Array,
        magnitudes: Array,
        phases: Array,
        insert_neighbors: Array,
        remove_neighbors: Array,
        insert_count: Array,
        remove_count: Array,
        move_probabilities: Array,
        worm_fugacity: Array,
        detailed_balance_residual: Array,
        steps: int,
        maximum_neighbors: int,
        maximum_order: int,
        plan_id: str,
        prepared_id: str,
        /,
    ):
        self.orders = orders
        self.vertex_counts = vertex_counts
        self.complex_weights = complex_weights
        self.magnitudes = magnitudes
        self.phases = phases
        self.insert_neighbors = insert_neighbors
        self.remove_neighbors = remove_neighbors
        self.insert_count = insert_count
        self.remove_count = remove_count
        self.move_probabilities = move_probabilities
        self.worm_fugacity = worm_fugacity
        self.detailed_balance_residual = detailed_balance_residual
        self.steps = int(steps)
        self.maximum_neighbors = int(maximum_neighbors)
        self.maximum_order = int(maximum_order)
        self.plan_id = str(plan_id)
        self.prepared_id = str(prepared_id)

    def initialize(self, diagram_index: int = 0, /) -> DiagramMonteCarloState:
        index = int(diagram_index)
        if index < 0 or index >= self.orders.shape[0]:
            raise ValueError("diagram_index is outside the prepared catalog.")
        return DiagramMonteCarloState(
            jnp.asarray(index, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            jnp.zeros((self.maximum_order + 1,), dtype=jnp.int32),
            jnp.zeros((3,), dtype=jnp.int32),
            jnp.zeros((3,), dtype=jnp.int32),
            jnp.asarray(0.0 + 0.0j),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
        )

    def transition(
        self, state: DiagramMonteCarloState, key: Key[Array, ""], /
    ) -> DiagramMonteCarloState:
        move_key, choice_key, endpoint_key, acceptance_key = jr.split(key, 4)
        move = jr.categorical(move_key, jnp.log(self.move_probabilities)).astype(
            jnp.int32
        )
        insertion = move == 0
        physical_move = move < 2
        current = state.diagram_index
        neighbor_count = jnp.where(
            insertion, self.insert_count[current], self.remove_count[current]
        )
        table = jnp.where(insertion, self.insert_neighbors, self.remove_neighbors)
        slot = jnp.floor(jr.uniform(choice_key) * jnp.maximum(neighbor_count, 1)).astype(
            jnp.int32
        )
        candidate = table[current, slot]
        reverse_count = jnp.where(
            insertion, self.remove_count[candidate], self.insert_count[candidate]
        )
        forward_move_probability = jnp.where(
            insertion, self.move_probabilities[0], self.move_probabilities[1]
        )
        reverse_move_probability = jnp.where(
            insertion, self.move_probabilities[1], self.move_probabilities[0]
        )
        forward_q = forward_move_probability / jnp.maximum(neighbor_count, 1)
        reverse_q = reverse_move_probability / jnp.maximum(reverse_count, 1)
        physical_ratio = (
            self.magnitudes[candidate]
            * reverse_q
            / (self.magnitudes[current] * forward_q)
        )
        physical_available = (
            physical_move & ~state.worm_open & (neighbor_count > 0) & (reverse_count > 0)
        )
        physical_acceptance = jnp.where(
            physical_available, jnp.minimum(1.0, physical_ratio), 0.0
        )

        vertex_count = self.vertex_counts[current]
        endpoint_count = vertex_count * vertex_count
        endpoint = jnp.floor(jr.uniform(endpoint_key) * endpoint_count).astype(jnp.int32)
        proposed_source = endpoint // vertex_count
        proposed_target = endpoint % vertex_count
        worm_ratio = jnp.where(
            state.worm_open,
            1.0 / (self.worm_fugacity * endpoint_count),
            self.worm_fugacity * endpoint_count,
        )
        worm_acceptance = jnp.minimum(1.0, worm_ratio)
        acceptance_probability = jnp.where(
            physical_move, physical_acceptance, worm_acceptance
        )
        accepted = jr.uniform(acceptance_key) < acceptance_probability
        accepted_physical = accepted & physical_move
        accepted_worm = accepted & ~physical_move
        next_index = jnp.where(accepted_physical, candidate, current)
        next_worm_open = jnp.where(accepted_worm, ~state.worm_open, state.worm_open)
        opening = accepted_worm & ~state.worm_open
        closing = accepted_worm & state.worm_open
        next_source = jnp.where(
            opening,
            proposed_source,
            jnp.where(closing, -1, state.worm_source),
        )
        next_target = jnp.where(
            opening,
            proposed_target,
            jnp.where(closing, -1, state.worm_target),
        )
        proposed_moves = state.proposed_moves.at[move].add(1)
        accepted_moves = state.accepted_moves.at[move].add(accepted.astype(jnp.int32))
        histogram = state.order_histogram.at[self.orders[next_index]].add(1)
        physical_sample = ~next_worm_open
        phase_sum = state.phase_sum + jnp.where(
            physical_sample, self.phases[next_index], 0.0 + 0.0j
        )
        phase_samples = state.phase_samples + physical_sample.astype(jnp.int32)
        return DiagramMonteCarloState(
            next_index,
            next_worm_open,
            next_source,
            next_target,
            histogram,
            proposed_moves,
            accepted_moves,
            phase_sum,
            phase_samples,
            state.step + 1,
        )

    def run(
        self,
        key: Key[Array, ""],
        /,
        *,
        initial_index: int = 0,
    ) -> DiagramMonteCarloResult:
        initial = self.initialize(initial_index)
        keys = jr.split(key, self.steps)

        def advance(state, step_key):
            return self.transition(state, step_key), None

        state, _ = jax.lax.scan(advance, initial, keys)
        proposed = jnp.maximum(state.proposed_moves, 1)
        acceptance_rates = state.accepted_moves / proposed
        phase_samples = jnp.maximum(state.phase_samples, 1)
        mean_phase = state.phase_sum / phase_samples
        average_sign = jnp.abs(mean_phase)
        effective = state.phase_samples * average_sign * average_sign
        phase_standard_error = jnp.sqrt(
            jnp.maximum(0.0, 1.0 - average_sign * average_sign) / phase_samples
        )
        total = jnp.maximum(jnp.sum(state.order_histogram), 1)
        order_probabilities = state.order_histogram / total
        finite = (
            jnp.all(jnp.isfinite(acceptance_rates))
            & jnp.isfinite(mean_phase.real)
            & jnp.isfinite(mean_phase.imag)
            & jnp.isfinite(phase_standard_error)
            & jnp.isfinite(self.detailed_balance_residual)
        )
        tolerance = (
            128.0
            * jnp.finfo(self.magnitudes.dtype).eps
            * jnp.maximum(1.0, jnp.max(self.magnitudes))
        )
        successful = finite & (self.detailed_balance_residual <= tolerance)
        evidence = DiagramMonteCarloEvidence(
            acceptance_rates,
            self.detailed_balance_residual,
            mean_phase,
            average_sign,
            effective,
            phase_standard_error,
            finite,
            successful,
            jnp.where(successful, 0, 1).astype(jnp.int32),
        )
        return DiagramMonteCarloResult(
            state,
            order_probabilities,
            evidence,
            self.plan_id,
            self.prepared_id,
        )


__all__ = [
    "DiagramMonteCarloEvidence",
    "DiagramMonteCarloPlan",
    "DiagramMonteCarloResult",
    "DiagramMonteCarloState",
    "PreparedDiagramMonteCarlo",
]
