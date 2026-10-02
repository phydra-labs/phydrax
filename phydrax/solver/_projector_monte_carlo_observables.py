#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded sparse contractions in the original physical coefficient frame."""

from __future__ import annotations

from typing import final

import jax
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from ..operators.quantum.lattice._column import QuantumLatticeColumnOperator
from ..operators.quantum.lattice._guide import QuantumGuide
from ..sparse._execution import reduce_key_groups
from ..sparse._key_groups import KeyGroupPlan, KeyGroupState
from ..typing import Bool, Complex128, Scalar
from ._projector_monte_carlo_contracts import (
    ObservableDim,
    PairDim,
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloState,
    ReplicaDim,
)


@final
class ProjectorMonteCarloObservation(StrictModule):
    """One measured state; zero overlaps remain valid raw statistical records."""

    __strict_contract__ = True

    projected_numerator: Complex128[ReplicaDim]
    projected_denominator: Complex128[ReplicaDim]
    pair_numerators: Complex128[PairDim, ObservableDim]
    pair_denominators: Complex128[PairDim]
    finite: Bool[Scalar]


def _physical_coefficients(
    operator: QuantumLatticeColumnOperator,
    guide: QuantumGuide | None,
    keys: Array,
    coefficients: Array,
    active: Array,
) -> tuple[Array, Array]:
    """Undo exactly the same frozen positive similarity used by propagation."""
    if guide is None:
        physical = jnp.where(active, coefficients, jnp.zeros_like(coefficients))
        return physical, jnp.all(~active | jnp.isfinite(coefficients))

    def undo(index: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        values, successful = carry

        def evaluate(_: None) -> tuple[Array, Array]:
            address = operator.domain.from_key(keys[index])
            log_value, valid = guide.log_value(address)
            inverse = jnp.exp(-log_value)
            value = coefficients[index] * inverse.astype(jnp.complex128)
            admissible = (
                valid & jnp.isfinite(inverse) & (inverse > 0) & jnp.isfinite(value)
            )
            admissible = admissible & ((coefficients[index] == 0) | (value != 0))
            return value, admissible

        value, valid = jax.lax.cond(
            active[index],
            evaluate,
            lambda _: (jnp.asarray(0, dtype=jnp.complex128), jnp.asarray(True)),
            None,
        )
        return values.at[index].set(value), successful & valid

    return jax.lax.fori_loop(
        0,
        keys.shape[0],
        undo,
        (jnp.zeros_like(coefficients), jnp.asarray(True)),
    )


def _bra_groups(
    keys: Array, active: Array, coefficients: Array
) -> tuple[KeyGroupState, Array, Array]:
    capacity = keys.shape[0]
    plan = KeyGroupPlan(capacity, capacity, (2**32 - 1,) * keys.shape[-1])
    groups = plan.build(keys, valid=active)
    accumulated, evidence = reduce_key_groups(
        groups, jnp.conj(coefficients), value_valid=active
    )
    return groups, accumulated.value, evidence.successful


def _lookup_bra(groups: KeyGroupState, values: Array, key: Array) -> Array:
    lookup = groups.lookup(key[None, :])
    slot = jnp.maximum(lookup.group_slots[0], 0)
    return jnp.where(
        lookup.supported[0], values[slot], jnp.asarray(0, dtype=jnp.complex128)
    )


def _overlap(
    groups: KeyGroupState, bra: Array, keys: Array, ket: Array, active: Array
) -> Array:
    def add(index: Array, total: Array) -> Array:
        return jax.lax.cond(
            active[index],
            lambda _: total + _lookup_bra(groups, bra, keys[index]) * ket[index],
            lambda _: total,
            None,
        )

    return jax.lax.fori_loop(0, keys.shape[0], add, jnp.asarray(0, dtype=jnp.complex128))


def _operator_overlap(
    operator: QuantumLatticeColumnOperator,
    groups: KeyGroupState,
    bra: Array,
    keys: Array,
    ket: Array,
    active: Array,
) -> tuple[Array, Array]:
    """Stream physical outgoing routes, including all return-to-source terms."""

    def source(index: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        def contract(_: None) -> tuple[Array, Array]:
            cached = operator.prepare_source(operator.domain.from_key(keys[index]))

            def route(
                route_index: Array, running: tuple[Array, Array]
            ) -> tuple[Array, Array]:
                total, successful = running
                excitation = operator.raw_route(cached, route_index)
                contribution = jax.lax.cond(
                    excitation.valid,
                    lambda _: (
                        _lookup_bra(groups, bra, excitation.target_key)
                        * excitation.matrix_element
                        * ket[index]
                    ),
                    lambda _: jnp.asarray(0, dtype=jnp.complex128),
                    None,
                )
                return (
                    total + contribution,
                    successful & excitation.successful & jnp.isfinite(contribution),
                )

            return jax.lax.fori_loop(0, operator.raw_route_bound, route, carry)

        return jax.lax.cond(active[index], contract, lambda _: carry, None)

    return jax.lax.fori_loop(
        0,
        keys.shape[0],
        source,
        (jnp.asarray(0, dtype=jnp.complex128), jnp.asarray(True)),
    )


def observe_projector_state(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
) -> ProjectorMonteCarloObservation:
    """Observe a candidate without mutating it or allocating an S-by-S product."""
    original = prepared.original_operator
    physical: list[Array] = []
    bras: list[tuple[KeyGroupState, Array]] = []
    finite = jnp.asarray(True)
    for replica in range(prepared.plan.replicas):
        coefficients, successful = _physical_coefficients(
            original,
            prepared.guide,
            state.support_keys[replica],
            state.coefficients[replica],
            state.active[replica],
        )
        physical.append(coefficients)
        groups, values, grouped = _bra_groups(
            state.support_keys[replica], state.active[replica], coefficients
        )
        bras.append((groups, values))
        finite = finite & successful & grouped
    trial, trial_values, trial_successful = _bra_groups(
        prepared.trial_keys,
        prepared.trial_active,
        prepared.trial_coefficients,
    )
    finite = finite & trial_successful
    projected_numerator = jnp.zeros((prepared.plan.replicas,), dtype=jnp.complex128)
    projected_denominator = jnp.zeros_like(projected_numerator)
    for replica in range(prepared.plan.replicas):
        numerator, successful = _operator_overlap(
            original,
            trial,
            trial_values,
            state.support_keys[replica],
            physical[replica],
            state.active[replica],
        )
        denominator = _overlap(
            trial,
            trial_values,
            state.support_keys[replica],
            physical[replica],
            state.active[replica],
        )
        projected_numerator = projected_numerator.at[replica].set(numerator)
        projected_denominator = projected_denominator.at[replica].set(denominator)
        finite = finite & successful
    operators = (original,) + prepared.observables
    pair_numerators = jnp.zeros(
        (len(prepared.pair_ids), len(operators)), dtype=jnp.complex128
    )
    pair_denominators = jnp.zeros((len(prepared.pair_ids),), dtype=jnp.complex128)
    for pair_index, (a, b) in enumerate(prepared.pair_ids):
        groups, bra = bras[a]
        denominator = _overlap(
            groups, bra, state.support_keys[b], physical[b], state.active[b]
        )
        pair_denominators = pair_denominators.at[pair_index].set(denominator)
        for observable_index, operator in enumerate(operators):
            numerator, successful = _operator_overlap(
                operator,
                groups,
                bra,
                state.support_keys[b],
                physical[b],
                state.active[b],
            )
            pair_numerators = pair_numerators.at[pair_index, observable_index].set(
                numerator
            )
            finite = finite & successful
    finite = (
        finite
        & jnp.all(jnp.isfinite(projected_numerator))
        & jnp.all(jnp.isfinite(projected_denominator))
    )
    finite = (
        finite
        & jnp.all(jnp.isfinite(pair_numerators))
        & jnp.all(jnp.isfinite(pair_denominators))
    )
    return ProjectorMonteCarloObservation(
        projected_numerator=projected_numerator,
        projected_denominator=projected_denominator,
        pair_numerators=pair_numerators,
        pair_denominators=pair_denominators,
        finite=finite,
    )
