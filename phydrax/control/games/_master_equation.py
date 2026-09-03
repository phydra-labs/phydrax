#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact finite-state, finite-population master-equation reference.

This module deliberately solves a small, finite object.  The empirical law is
restricted to the count-simplex with denominator ``population_size``; no
interpolation or off-lattice evaluation is performed.  Population agents move
conditionally independently by default, producing the exact convolution of
state-wise multinomial laws.  A problem may instead declare a complete
probability vector on the same lattice.

The representative agent is separate from the population used to form the
empirical law.  Conditional on the current state and law, its physical-state
transition and the population aggregate transition are independent.  This is a
finite discrete reference, not a continuous-state or Lions-derivative solver.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import IntEnum
from functools import lru_cache
from itertools import product
from math import exp, isfinite, lgamma, log
from numbers import Integral
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


FINITE_STATE_DISCRETE_MASTER_EQUATION_REFERENCE = (
    "FINITE_STATE_DISCRETE_MASTER_EQUATION_REFERENCE"
)
DISCRETE_EMPIRICAL_LAW_NEIGHBOR_TRANSFER_DIFFERENCE = (
    "DISCRETE_EMPIRICAL_LAW_NEIGHBOR_TRANSFER_DIFFERENCE"
)


class FiniteStateMasterEquationStatus(IntEnum):
    """Stable outcomes of the finite discrete backward reference solve."""

    SUCCESS = 0
    INVALID_TERMINAL_COST = 1
    INVALID_RUNNING_COST = 2
    INVALID_TRANSITION_PROBABILITIES = 3
    INVALID_AGGREGATE_LAW_TRANSITION = 4
    NO_DETERMINISTIC_SELECTOR_FIXED_POINT = 5
    NONFINITE_BELLMAN_VALUE = 6


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _symbols(values: Sequence[Any], owner: str, /) -> tuple[Any, ...]:
    symbols = tuple(values)
    if not symbols:
        raise ValueError(f"{owner} must be non-empty.")
    try:
        distinct = len(set(symbols))
    except TypeError as error:
        raise TypeError(f"{owner} entries must be hashable.") from error
    if distinct != len(symbols):
        raise ValueError(f"{owner} entries must be unique.")
    return symbols


@lru_cache(maxsize=None)
def _weak_compositions(total: int, parts: int, /) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),)
    rows: list[tuple[int, ...]] = []
    for first in range(total + 1):
        rows.extend(
            (first, *remainder)
            for remainder in _weak_compositions(total - first, parts - 1)
        )
    return tuple(rows)


def _probability_simplex_residual(probabilities: np.ndarray, /) -> float:
    if (
        probabilities.ndim != 1
        or probabilities.size == 0
        or np.iscomplexobj(probabilities)
        or not np.all(np.isfinite(probabilities))
    ):
        return float("inf")
    negativity = max(0.0, float(-np.min(probabilities)))
    return max(abs(float(np.sum(probabilities)) - 1.0), negativity)


def _finite_scalar(value: Any, /) -> float | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.shape != () or np.iscomplexobj(array):
        return None
    scalar = float(array)
    return scalar if isfinite(scalar) else None


class FinitePopulationSimplexLattice(StrictModule):
    """All empirical laws with ``population_size`` atoms on finite states.

    ``counts`` has shape ``(num_laws, num_states)`` and integer rows summing to
    ``population_size``.  ``laws`` is exactly ``counts / population_size``.
    Directed neighbor edges move one population member from a nonempty source
    coordinate to a distinct destination coordinate.
    """

    counts: Array
    laws: Array
    neighbor_from_indices: Array
    neighbor_to_indices: Array
    neighbor_source_states: Array
    neighbor_destination_states: Array
    population_size: int = eqx.field(static=True)
    num_states: int = eqx.field(static=True)
    num_laws: int = eqx.field(static=True)
    num_neighbor_transfers: int = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)

    def __init__(self, num_states: int, population_size: int):
        if (
            isinstance(num_states, bool)
            or not isinstance(num_states, int)
            or num_states <= 0
        ):
            raise ValueError("num_states must be a positive integer.")
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size <= 0
        ):
            raise ValueError("population_size must be a positive integer.")

        count_rows = _weak_compositions(population_size, num_states)
        counts_np = np.asarray(count_rows, dtype=np.int32)
        count_to_index = {row: index for index, row in enumerate(count_rows)}
        edge_rows: list[tuple[int, int, int, int]] = []
        for from_index, row in enumerate(count_rows):
            for source in range(num_states):
                if row[source] == 0:
                    continue
                for destination in range(num_states):
                    if destination == source:
                        continue
                    neighbor = list(row)
                    neighbor[source] -= 1
                    neighbor[destination] += 1
                    edge_rows.append(
                        (
                            from_index,
                            count_to_index[tuple(neighbor)],
                            source,
                            destination,
                        )
                    )

        if edge_rows:
            edges = np.asarray(edge_rows, dtype=np.int32)
        else:
            edges = np.empty((0, 4), dtype=np.int32)
        self.counts = jnp.asarray(counts_np)
        self.laws = jnp.asarray(counts_np / float(population_size))
        self.neighbor_from_indices = jnp.asarray(edges[:, 0])
        self.neighbor_to_indices = jnp.asarray(edges[:, 1])
        self.neighbor_source_states = jnp.asarray(edges[:, 2])
        self.neighbor_destination_states = jnp.asarray(edges[:, 3])
        self.population_size = population_size
        self.num_states = num_states
        self.num_laws = len(count_rows)
        self.num_neighbor_transfers = len(edge_rows)
        self.lattice_id = canonical_fingerprint(
            {
                "kind": "finite-population-simplex-lattice",
                "num_states": num_states,
                "population_size": population_size,
                "count_order": [list(row) for row in count_rows],
            }
        )

    @property
    def empirical_law_step(self) -> float:
        return 1.0 / self.population_size

    def index_of_counts(self, counts: ArrayLike, /) -> int:
        """Return the exact lattice row for integer population counts."""

        array = np.asarray(counts)
        if array.shape != (self.num_states,) or np.iscomplexobj(array):
            raise ValueError("counts must contain one real entry per state.")
        if not np.all(np.isfinite(array)) or not np.all(array == np.rint(array)):
            raise ValueError("counts must be finite integers.")
        integer = np.asarray(array, dtype=np.int64)
        if np.any(integer < 0) or int(np.sum(integer)) != self.population_size:
            raise ValueError("counts must be nonnegative and sum to population_size.")
        matches = np.flatnonzero(
            np.all(np.asarray(self.counts) == integer[None, :], axis=1)
        )
        if matches.size != 1:
            raise ValueError("counts are not on this lattice.")
        return int(matches[0])

    def index_of_law(self, law: ArrayLike, /, *, atol: float = 1.0e-8) -> int:
        """Return a grid-law index without interpolating between lattice rows."""

        tolerance = float(atol)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("atol must be finite and nonnegative.")
        array = np.asarray(law)
        if array.shape != (self.num_states,) or np.iscomplexobj(array):
            raise ValueError("law must contain one real entry per state.")
        if not np.all(np.isfinite(array)):
            raise ValueError("law must be finite.")
        scaled = array * self.population_size
        rounded = np.rint(scaled)
        if np.max(np.abs(scaled - rounded)) > tolerance * self.population_size:
            raise ValueError("law is not an exact empirical law on this lattice.")
        return self.index_of_counts(rounded)


class FiniteStateMasterEquationProblem(StrictModule):
    """Finite data and callbacks for a discrete empirical-law master equation.

    Callback signatures are::

        transition_probabilities(time_index, state, action, law, args) -> (S,)
        running_cost(time_index, state, action, law, args) -> scalar
        terminal_cost(state, law, args) -> scalar

    If supplied, ``aggregate_law_transition`` has signature
    ``(time_index, law, selector_actions, args)`` and returns either a length-``L``
    probability vector in lattice order or a mapping from lattice indices/count
    tuples to probabilities.  Otherwise the solver forms the exact convolution
    of the state-wise multinomial population transitions.

    The deterministic selector is the first minimum in the declared action
    order.  ``selector_id`` records the caller's stable identity for that rule.
    """

    lattice: FinitePopulationSimplexLattice
    transition_probabilities: Callable[[int, Any, Any, Array, Any], ArrayLike] = (
        eqx.field(static=True)
    )
    running_cost: Callable[[int, Any, Any, Array, Any], ArrayLike] = eqx.field(
        static=True
    )
    terminal_cost: Callable[[Any, Array, Any], ArrayLike] = eqx.field(static=True)
    aggregate_law_transition: (
        Callable[[int, Array, tuple[Any, ...], Any], ArrayLike | Mapping[Any, Any]] | None
    ) = eqx.field(static=True)
    states: tuple[Any, ...] = eqx.field(static=True)
    actions: tuple[Any, ...] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    population_size: int = eqx.field(static=True)
    num_states: int = eqx.field(static=True)
    num_actions: int = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)
    aggregate_law_transition_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        states: Sequence[Any],
        actions: Sequence[Any],
        horizon: int,
        population_size: int,
        transition_probabilities: Callable[[int, Any, Any, Array, Any], ArrayLike],
        running_cost: Callable[[int, Any, Any, Array, Any], ArrayLike],
        terminal_cost: Callable[[Any, Array, Any], ArrayLike],
        /,
        *,
        selector_id: str,
        problem_id: str,
        aggregate_law_transition: Callable[
            [int, Array, tuple[Any, ...], Any], ArrayLike | Mapping[Any, Any]
        ]
        | None = None,
        aggregate_law_transition_id: str | None = None,
    ):
        state_symbols = _symbols(states, "states")
        action_symbols = _symbols(actions, "actions")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 0:
            raise ValueError("horizon must be a nonnegative integer.")
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size <= 0
        ):
            raise ValueError("population_size must be a positive integer.")
        for owner, callback in (
            ("transition_probabilities", transition_probabilities),
            ("running_cost", running_cost),
            ("terminal_cost", terminal_cost),
        ):
            if not callable(callback):
                raise TypeError(f"{owner} must be callable.")
        if aggregate_law_transition is not None and not callable(
            aggregate_law_transition
        ):
            raise TypeError("aggregate_law_transition must be callable or None.")
        if aggregate_law_transition is None:
            if aggregate_law_transition_id is not None:
                raise ValueError(
                    "aggregate_law_transition_id requires aggregate_law_transition."
                )
            aggregate_id = None
        else:
            if aggregate_law_transition_id is None:
                raise ValueError(
                    "aggregate_law_transition_id is required for a declared "
                    "aggregate transition."
                )
            aggregate_id = _identifier(
                aggregate_law_transition_id, "aggregate_law_transition_id"
            )

        self.lattice = FinitePopulationSimplexLattice(len(state_symbols), population_size)
        self.transition_probabilities = transition_probabilities
        self.running_cost = running_cost
        self.terminal_cost = terminal_cost
        self.aggregate_law_transition = aggregate_law_transition
        self.states = state_symbols
        self.actions = action_symbols
        self.horizon = horizon
        self.population_size = population_size
        self.num_states = len(state_symbols)
        self.num_actions = len(action_symbols)
        self.selector_id = _identifier(selector_id, "selector_id")
        self.aggregate_law_transition_id = aggregate_id
        self.problem_id = _identifier(problem_id, "problem_id")

    @property
    def aggregate_transition_mode(self) -> str:
        return (
            "exact-state-wise-multinomial"
            if self.aggregate_law_transition is None
            else "declared-lattice-probabilities"
        )


class FiniteStateMasterEquationEvidence(StrictModule):
    """Residual and finite-law sensitivity evidence from the backward solve."""

    bellman_residuals: Array
    action_minimum_residuals: Array
    terminal_residuals: Array
    physical_simplex_probability_residuals: Array
    law_simplex_probability_residuals: Array
    simplex_probability_residuals: Array
    neighbor_transfer_differences: Array
    bellman_residual: Array
    action_minimum_residual: Array
    terminal_residual: Array
    simplex_probability_residual: Array
    maximum_neighbor_transfer_difference: Array
    population_size: int = eqx.field(static=True)
    lattice_size: int = eqx.field(static=True)
    empirical_law_step: float = eqx.field(static=True)
    neighbor_transfer_count: int = eqx.field(static=True)
    refinement_id: str = eqx.field(static=True)
    law_sensitivity_label: str = eqx.field(static=True)
    discrete_empirical_law_difference: bool = eqx.field(static=True)
    lions_derivative_computed: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        bellman_residuals: ArrayLike,
        action_minimum_residuals: ArrayLike,
        terminal_residuals: ArrayLike,
        physical_simplex_probability_residuals: ArrayLike,
        law_simplex_probability_residuals: ArrayLike,
        neighbor_transfer_differences: ArrayLike,
        lattice: FinitePopulationSimplexLattice,
    ):
        bellman = jnp.asarray(bellman_residuals)
        minimum = jnp.asarray(action_minimum_residuals)
        terminal = jnp.asarray(terminal_residuals)
        physical_simplex = jnp.asarray(physical_simplex_probability_residuals)
        law_simplex = jnp.asarray(law_simplex_probability_residuals)
        neighbor = jnp.asarray(neighbor_transfer_differences)
        simplex = jnp.concatenate(
            (jnp.ravel(physical_simplex), jnp.ravel(law_simplex)), axis=0
        )

        self.bellman_residuals = bellman
        self.action_minimum_residuals = minimum
        self.terminal_residuals = terminal
        self.physical_simplex_probability_residuals = physical_simplex
        self.law_simplex_probability_residuals = law_simplex
        self.simplex_probability_residuals = simplex
        self.neighbor_transfer_differences = neighbor
        self.bellman_residual = _maximum_absolute(bellman)
        self.action_minimum_residual = _maximum_absolute(minimum)
        self.terminal_residual = _maximum_absolute(terminal)
        self.simplex_probability_residual = _maximum_absolute(simplex)
        self.maximum_neighbor_transfer_difference = _maximum_absolute(neighbor)
        self.population_size = lattice.population_size
        self.lattice_size = lattice.num_laws
        self.empirical_law_step = lattice.empirical_law_step
        self.neighbor_transfer_count = lattice.num_neighbor_transfers
        self.refinement_id = canonical_fingerprint(
            {
                "kind": "finite-empirical-law-refinement",
                "population_size": lattice.population_size,
                "num_states": lattice.num_states,
                "lattice_size": lattice.num_laws,
                "law_step": lattice.empirical_law_step,
            }
        )
        self.law_sensitivity_label = DISCRETE_EMPIRICAL_LAW_NEIGHBOR_TRANSFER_DIFFERENCE
        self.discrete_empirical_law_difference = True
        self.lions_derivative_computed = False


class FiniteStateMasterEquationResult(StrictModule):
    """Value table, selector, law kernel, and scope-explicit certificate."""

    problem: FiniteStateMasterEquationProblem
    lattice: FinitePopulationSimplexLattice
    values: Array
    action_values: Array
    selectors: Array
    law_transition_table: Array
    evidence: FiniteStateMasterEquationEvidence
    status: Array
    valid: Array
    problem_id: str = eqx.field(static=True)
    selector_id: str = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    termination_detail: str = eqx.field(static=True)
    aggregate_transition_mode: str = eqx.field(static=True)
    finite_state_discrete_reference: bool = eqx.field(static=True)
    exact_population_lattice_evaluated: bool = eqx.field(static=True)
    continuous_state_claimed: bool = eqx.field(static=True)
    continuous_law_claimed: bool = eqx.field(static=True)
    lions_derivative_claimed: bool = eqx.field(static=True)
    lions_derivative_evaluated: bool = eqx.field(static=True)
    continuous_master_equation_claimed: bool = eqx.field(static=True)
    global_master_equation_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    mean_field_control_claimed: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    common_noise_equilibrium_claimed: bool = eqx.field(static=True)
    common_noise_claimed: bool = eqx.field(static=True)
    common_noise_supported: bool = eqx.field(static=True)
    finite_common_state_supported: bool = eqx.field(static=True)
    finite_common_state_evaluated: bool = eqx.field(static=True)

    def __init__(
        self,
        problem: FiniteStateMasterEquationProblem,
        values: ArrayLike,
        action_values: ArrayLike,
        selectors: ArrayLike,
        law_transition_table: ArrayLike,
        evidence: FiniteStateMasterEquationEvidence,
        status: FiniteStateMasterEquationStatus,
        termination_detail: str,
        /,
    ):
        self.problem = problem
        self.lattice = problem.lattice
        self.values = jnp.asarray(values)
        self.action_values = jnp.asarray(action_values)
        self.selectors = jnp.asarray(selectors, dtype=jnp.int32)
        self.law_transition_table = jnp.asarray(law_transition_table)
        self.evidence = evidence
        self.status = jnp.asarray(int(status), dtype=jnp.int32)
        self.valid = jnp.asarray(status == FiniteStateMasterEquationStatus.SUCCESS)
        self.problem_id = problem.problem_id
        self.selector_id = problem.selector_id
        self.certificate_label = FINITE_STATE_DISCRETE_MASTER_EQUATION_REFERENCE
        self.termination_detail = _identifier(termination_detail, "termination_detail")
        self.aggregate_transition_mode = problem.aggregate_transition_mode
        self.finite_state_discrete_reference = True
        self.exact_population_lattice_evaluated = (
            status == FiniteStateMasterEquationStatus.SUCCESS
        )
        self.continuous_state_claimed = False
        self.continuous_law_claimed = False
        self.lions_derivative_claimed = False
        self.lions_derivative_evaluated = False
        self.continuous_master_equation_claimed = False
        self.global_master_equation_claimed = False
        self.mean_field_control_optimum_claimed = False
        self.mean_field_control_claimed = False
        self.mean_field_game_equilibrium_claimed = False
        self.common_noise_equilibrium_claimed = False
        self.common_noise_claimed = False
        self.common_noise_supported = False
        self.finite_common_state_supported = False
        self.finite_common_state_evaluated = False

    @property
    def U(self) -> Array:
        """Value table indexed by ``(time, state, empirical-law row)``."""

        return self.values

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def certified(self) -> Array:
        return self.valid

    def selected_action(
        self, time_index: int, state_index: int, law_index: int, /
    ) -> Any:
        """Return the declared action symbol selected at one table entry."""

        action_index = int(self.selectors[time_index, state_index, law_index])
        if action_index < 0:
            raise ValueError("No action was selected at this table entry.")
        return self.problem.actions[action_index]


def _maximum_absolute(values: Array, /) -> Array:
    if values.size == 0:
        return jnp.asarray(0.0)
    return jnp.nanmax(jnp.abs(values))


def _multinomial_probability(
    destination_counts: tuple[int, ...], probabilities: np.ndarray, /
) -> float:
    log_probability = lgamma(sum(destination_counts) + 1.0)
    for count in destination_counts:
        log_probability -= lgamma(count + 1.0)
    for count, event_probability in zip(destination_counts, probabilities, strict=True):
        if count:
            probability = float(event_probability)
            if probability == 0.0:
                return 0.0
            log_probability += count * log(probability)
    return exp(log_probability)


def _exact_population_law_transition(
    source_counts: tuple[int, ...],
    selected_transition_rows: np.ndarray,
    count_to_index: Mapping[tuple[int, ...], int],
    /,
) -> np.ndarray:
    num_states = len(source_counts)
    distribution: dict[tuple[int, ...], float] = {(0,) * num_states: 1.0}
    for state, population_count in enumerate(source_counts):
        group_distribution = {
            destination: _multinomial_probability(
                destination, selected_transition_rows[state]
            )
            for destination in _weak_compositions(population_count, num_states)
        }
        convolved: dict[tuple[int, ...], float] = {}
        for accumulated, accumulated_probability in distribution.items():
            for destination, destination_probability in group_distribution.items():
                combined = tuple(
                    left + right
                    for left, right in zip(accumulated, destination, strict=True)
                )
                convolved[combined] = convolved.get(combined, 0.0) + (
                    accumulated_probability * destination_probability
                )
        distribution = convolved

    probabilities = np.zeros((len(count_to_index),), dtype=float)
    for counts, probability in distribution.items():
        probabilities[count_to_index[counts]] = probability
    return probabilities


def _declared_population_law_transition(
    declared: ArrayLike | Mapping[Any, Any],
    lattice: FinitePopulationSimplexLattice,
    /,
) -> np.ndarray | None:
    if isinstance(declared, Mapping):
        probabilities = np.zeros((lattice.num_laws,), dtype=float)
        try:
            for key, value in declared.items():
                if isinstance(key, Integral) and not isinstance(key, bool):
                    index = int(key)
                    if not 0 <= index < lattice.num_laws:
                        return None
                else:
                    index = lattice.index_of_counts(key)
                scalar = _finite_scalar(value)
                if scalar is None:
                    return None
                probabilities[index] += scalar
        except (TypeError, ValueError):
            return None
        return probabilities
    try:
        raw_probabilities = np.asarray(declared)
    except (TypeError, ValueError):
        return None
    if raw_probabilities.shape != (lattice.num_laws,) or np.iscomplexobj(
        raw_probabilities
    ):
        return None
    try:
        return np.asarray(raw_probabilities, dtype=float)
    except (TypeError, ValueError):
        return None


def _neighbor_transfer_differences(
    values: np.ndarray, lattice: FinitePopulationSimplexLattice, /
) -> np.ndarray:
    sources = np.asarray(lattice.neighbor_from_indices, dtype=np.int64)
    destinations = np.asarray(lattice.neighbor_to_indices, dtype=np.int64)
    if sources.size == 0:
        return np.empty((values.shape[0], values.shape[1], 0), dtype=float)
    return values[:, :, destinations] - values[:, :, sources]


def _evidence(
    values: np.ndarray,
    lattice: FinitePopulationSimplexLattice,
    bellman_residuals: np.ndarray,
    action_minimum_residuals: np.ndarray,
    terminal_residuals: np.ndarray,
    physical_simplex_residuals: np.ndarray,
    law_simplex_residuals: np.ndarray,
    /,
) -> FiniteStateMasterEquationEvidence:
    return FiniteStateMasterEquationEvidence(
        bellman_residuals=bellman_residuals,
        action_minimum_residuals=action_minimum_residuals,
        terminal_residuals=terminal_residuals,
        physical_simplex_probability_residuals=physical_simplex_residuals,
        law_simplex_probability_residuals=law_simplex_residuals,
        neighbor_transfer_differences=_neighbor_transfer_differences(values, lattice),
        lattice=lattice,
    )


def solve_finite_state_master_equation_reference(
    problem: FiniteStateMasterEquationProblem,
    /,
    *,
    args: Any = None,
    probability_tolerance: float = 1.0e-7,
) -> FiniteStateMasterEquationResult:
    """Solve the exact finite-lattice backward recursion by enumeration.

    At each time and empirical law, selector profiles are considered in
    lexicographic action-index order.  A profile is accepted exactly when every
    component equals the first action minimum computed under the aggregate law
    transition induced by that profile.  The first such fixed profile is used.
    This deterministic finite enumeration is intentionally a reference method,
    not a scalable approximation.
    """

    if not isinstance(problem, FiniteStateMasterEquationProblem):
        raise TypeError("problem must be a FiniteStateMasterEquationProblem.")
    tolerance = float(probability_tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("probability_tolerance must be finite and nonnegative.")

    lattice = problem.lattice
    horizon = problem.horizon
    num_states = problem.num_states
    num_actions = problem.num_actions
    num_laws = lattice.num_laws
    laws_np = np.asarray(lattice.laws)
    counts_np = np.asarray(lattice.counts, dtype=np.int64)
    count_rows = tuple(tuple(map(int, row)) for row in counts_np)
    count_to_index = {row: index for index, row in enumerate(count_rows)}

    values = np.full((horizon + 1, num_states, num_laws), np.nan, dtype=float)
    action_values = np.full(
        (horizon, num_states, num_laws, num_actions), np.nan, dtype=float
    )
    selectors = np.full((horizon, num_states, num_laws), -1, dtype=np.int32)
    law_transition_table = np.full((horizon, num_laws, num_laws), np.nan, dtype=float)
    physical_transitions = np.full(
        (horizon, num_laws, num_states, num_actions, num_states),
        np.nan,
        dtype=float,
    )
    running_costs = np.full(
        (horizon, num_laws, num_states, num_actions), np.nan, dtype=float
    )
    physical_simplex_residuals = np.full(
        (horizon, num_laws, num_states, num_actions), np.nan, dtype=float
    )
    law_simplex_residuals = np.full((horizon, num_laws), np.nan, dtype=float)
    bellman_residuals = np.full((horizon, num_states, num_laws), np.nan)
    action_minimum_residuals = np.full((horizon, num_states, num_laws), np.nan)
    terminal_residuals = np.full((num_states, num_laws), np.nan)

    def finish(
        status: FiniteStateMasterEquationStatus, detail: str
    ) -> FiniteStateMasterEquationResult:
        evidence = _evidence(
            values,
            lattice,
            bellman_residuals,
            action_minimum_residuals,
            terminal_residuals,
            physical_simplex_residuals,
            law_simplex_residuals,
        )
        return FiniteStateMasterEquationResult(
            problem,
            values,
            action_values,
            selectors,
            law_transition_table,
            evidence,
            status,
            detail,
        )

    terminal_reference = np.full((num_states, num_laws), np.nan, dtype=float)
    for state_index, state in enumerate(problem.states):
        for law_index in range(num_laws):
            law = jnp.asarray(laws_np[law_index])
            cost = _finite_scalar(problem.terminal_cost(state, law, args))
            if cost is None:
                terminal_residuals[state_index, law_index] = float("inf")
                return finish(
                    FiniteStateMasterEquationStatus.INVALID_TERMINAL_COST,
                    "terminal_cost must return a finite real scalar at every grid row",
                )
            terminal_reference[state_index, law_index] = cost
            values[horizon, state_index, law_index] = cost
    terminal_residuals[...] = values[horizon] - terminal_reference

    for time_index in range(horizon):
        for law_index in range(num_laws):
            law = jnp.asarray(laws_np[law_index])
            for state_index, state in enumerate(problem.states):
                for action_index, action in enumerate(problem.actions):
                    cost = _finite_scalar(
                        problem.running_cost(time_index, state, action, law, args)
                    )
                    if cost is None:
                        return finish(
                            FiniteStateMasterEquationStatus.INVALID_RUNNING_COST,
                            "running_cost must return a finite real scalar at every grid row",
                        )
                    running_costs[time_index, law_index, state_index, action_index] = cost
                    try:
                        raw_transition = np.asarray(
                            problem.transition_probabilities(
                                time_index, state, action, law, args
                            )
                        )
                        transition = (
                            np.empty((0,), dtype=float)
                            if np.iscomplexobj(raw_transition)
                            else np.asarray(raw_transition, dtype=float)
                        )
                    except (TypeError, ValueError):
                        transition = np.empty((0,), dtype=float)
                    residual = _probability_simplex_residual(transition)
                    physical_simplex_residuals[
                        time_index, law_index, state_index, action_index
                    ] = residual
                    if (
                        transition.shape != (num_states,)
                        or residual > tolerance
                        or np.any(transition < 0.0)
                        or np.any(transition > 1.0)
                    ):
                        return finish(
                            FiniteStateMasterEquationStatus.INVALID_TRANSITION_PROBABILITIES,
                            "transition_probabilities must return a finite probability simplex vector",
                        )
                    physical_transitions[
                        time_index, law_index, state_index, action_index
                    ] = transition

    for time_index in range(horizon - 1, -1, -1):
        for law_index in range(num_laws):
            law = jnp.asarray(laws_np[law_index])
            selected_profile: tuple[int, ...] | None = None
            selected_q: np.ndarray | None = None
            selected_law_probabilities: np.ndarray | None = None
            for profile in product(range(num_actions), repeat=num_states):
                if problem.aggregate_law_transition is None:
                    selected_rows = np.stack(
                        [
                            physical_transitions[
                                time_index,
                                law_index,
                                state_index,
                                profile[state_index],
                            ]
                            for state_index in range(num_states)
                        ],
                        axis=0,
                    )
                    law_probabilities = _exact_population_law_transition(
                        count_rows[law_index], selected_rows, count_to_index
                    )
                else:
                    declared = problem.aggregate_law_transition(
                        time_index,
                        law,
                        tuple(problem.actions[index] for index in profile),
                        args,
                    )
                    law_probabilities = _declared_population_law_transition(
                        declared, lattice
                    )
                    if law_probabilities is None:
                        law_simplex_residuals[time_index, law_index] = float("inf")
                        return finish(
                            FiniteStateMasterEquationStatus.INVALID_AGGREGATE_LAW_TRANSITION,
                            "aggregate_law_transition must identify a probability vector on the exact lattice",
                        )

                law_residual = _probability_simplex_residual(law_probabilities)
                law_simplex_residuals[time_index, law_index] = law_residual
                if (
                    law_residual > tolerance
                    or np.any(law_probabilities < 0.0)
                    or np.any(law_probabilities > 1.0)
                ):
                    return finish(
                        FiniteStateMasterEquationStatus.INVALID_AGGREGATE_LAW_TRANSITION,
                        "aggregate law transition is not a finite probability simplex vector",
                    )

                continuation_by_state = values[time_index + 1] @ law_probabilities
                q_values = np.empty((num_states, num_actions), dtype=float)
                for state_index in range(num_states):
                    for action_index in range(num_actions):
                        q_values[state_index, action_index] = running_costs[
                            time_index, law_index, state_index, action_index
                        ] + np.dot(
                            physical_transitions[
                                time_index,
                                law_index,
                                state_index,
                                action_index,
                            ],
                            continuation_by_state,
                        )
                if not np.all(np.isfinite(q_values)):
                    return finish(
                        FiniteStateMasterEquationStatus.NONFINITE_BELLMAN_VALUE,
                        "Bellman action values must remain finite",
                    )
                first_minimum = tuple(
                    int(np.argmin(q_values[state_index]))
                    for state_index in range(num_states)
                )
                if first_minimum == profile:
                    selected_profile = profile
                    selected_q = q_values
                    selected_law_probabilities = law_probabilities
                    break

            if selected_profile is None:
                return finish(
                    FiniteStateMasterEquationStatus.NO_DETERMINISTIC_SELECTOR_FIXED_POINT,
                    "no selector profile is self-consistent under the declared first-minimum rule",
                )
            assert selected_q is not None
            assert selected_law_probabilities is not None
            law_transition_table[time_index, law_index] = selected_law_probabilities
            for state_index, action_index in enumerate(selected_profile):
                selectors[time_index, state_index, law_index] = action_index
                action_values[time_index, state_index, law_index] = selected_q[
                    state_index
                ]
                values[time_index, state_index, law_index] = selected_q[
                    state_index, action_index
                ]

    selected_action_values = np.take_along_axis(
        action_values,
        selectors[..., None],
        axis=-1,
    )[..., 0]
    bellman_residuals[...] = values[:-1] - selected_action_values
    action_minimum_residuals[...] = selected_action_values - np.min(
        action_values, axis=-1
    )
    return finish(
        FiniteStateMasterEquationStatus.SUCCESS,
        "exact finite-state empirical-law backward recursion completed",
    )


__all__ = [
    "DISCRETE_EMPIRICAL_LAW_NEIGHBOR_TRANSFER_DIFFERENCE",
    "FINITE_STATE_DISCRETE_MASTER_EQUATION_REFERENCE",
    "FinitePopulationSimplexLattice",
    "FiniteStateMasterEquationEvidence",
    "FiniteStateMasterEquationProblem",
    "FiniteStateMasterEquationResult",
    "FiniteStateMasterEquationStatus",
    "solve_finite_state_master_equation_reference",
]
