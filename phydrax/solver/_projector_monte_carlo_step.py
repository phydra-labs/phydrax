# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Bounded ordered event streaming and the all-replica commit boundary."""

from __future__ import annotations

from typing import assert_never, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from .._sampling._addressing import derive_key, SampleAddress
from ..lifecycle._transaction import commit_candidate, TransactionalCandidate
from ..operators.quantum.lattice._column import RawQuantumExcitation
from ..sparse import KeyGroupAccumulation, KeyGroupPlan, reduce_key_groups
from ..typing import PRNGKey
from ._projector_monte_carlo_contracts import (
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloEvidence,
    ProjectorMonteCarloHistory,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStatus,
    ProjectorMonteCarloStepResult,
)
from ._projector_monte_carlo_observables import (
    observe_projector_state,
    ProjectorMonteCarloObservation,
)


class _Stream(NamedTuple):
    keys: Array
    active: Array
    high: Array
    correction: Array
    event_keys: Array
    event_values: Array
    event_valid: Array
    cursor: Array
    status: Array
    required_groups: Array
    maximum_attempts: Array
    event_count: Array
    exact_sources: Array
    sampled_sources: Array
    pre_norm: Array
    event_ids: Array
    next_event_id: Array


class _Replica(NamedTuple):
    keys: Array
    coefficients: Array
    active: Array
    shift: Array
    population: Array
    status: Array
    required_groups: Array
    required_support: Array
    maximum_attempts: Array
    event_count: Array
    exact_sources: Array
    sampled_sources: Array
    pre_norm: Array
    post_norm: Array


def _fail(status: Array, condition: Array, code: ProjectorMonteCarloStatus) -> Array:
    return jnp.where(
        (status == 0) & condition, jnp.asarray(code, dtype=jnp.int32), status
    )


def _logical_key(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    replica: Array,
    words: Array,
    ordinal: Array,
    role: str,
) -> PRNGKey:
    address = SampleAddress(
        "projector-monte-carlo",
        "physical-step",
        target=(
            prepared.original_operator.domain.domain_id,
            prepared.original_operator.operator_id,
        ),
        role=role,
    )
    step = state.step.astype(jnp.uint64)
    attempt = ordinal.astype(jnp.uint64)
    key = derive_key(
        state.root_key,
        address,
        (step >> jnp.uint64(32)).astype(jnp.uint32),
        step.astype(jnp.uint32),
        replica.astype(jnp.uint32),
    )

    def fold(index: Array, current: PRNGKey) -> PRNGKey:
        return jr.fold_in(current, words[index])

    key = jax.lax.fori_loop(0, words.shape[0], fold, key)
    return jr.fold_in(
        jr.fold_in(key, (attempt >> jnp.uint64(32)).astype(jnp.uint32)),
        attempt.astype(jnp.uint32),
    )


def _empty_stream(prepared: PreparedProjectorMonteCarlo) -> _Stream:
    plan = prepared.plan
    words = prepared.original_operator.domain.codec.word_count
    zero = jnp.asarray(0, dtype=jnp.int64)
    return _Stream(
        jnp.zeros((plan.group_capacity, words), dtype=jnp.uint32),
        jnp.zeros((plan.group_capacity,), dtype=jnp.bool_),
        jnp.zeros((plan.group_capacity,), dtype=jnp.complex128),
        jnp.zeros((plan.group_capacity,), dtype=jnp.complex128),
        jnp.zeros((plan.event_capacity, words), dtype=jnp.uint32),
        jnp.zeros((plan.event_capacity,), dtype=jnp.complex128),
        jnp.zeros((plan.event_capacity,), dtype=jnp.bool_),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        zero,
        zero,
        zero,
        zero,
        zero,
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.zeros((plan.event_capacity,), dtype=jnp.int64),
        zero,
    )


def _flush(prepared: PreparedProjectorMonteCarlo, stream: _Stream) -> _Stream:
    g, e = prepared.plan.group_capacity, prepared.plan.event_capacity
    group_plan = KeyGroupPlan(
        g + e, g, tuple(2**32 - 1 for _ in range(stream.keys.shape[-1]))
    )
    ids = jnp.concatenate((jnp.arange(-g, 0, dtype=jnp.int64), stream.event_ids))
    groups = group_plan.build(
        jnp.concatenate((stream.keys, stream.event_keys), axis=0),
        jnp.concatenate((stream.active, stream.event_valid), axis=0),
        stable_ids=ids,
    )
    lookup = groups.lookup(stream.keys, valid=stream.active)
    slots = jnp.maximum(lookup.group_slots, 0)
    seed_high = (
        jnp.zeros_like(stream.high)
        .at[slots]
        .add(jnp.where(stream.active & lookup.supported, stream.high, jnp.complex128(0)))
    )
    seed_correction = (
        jnp.zeros_like(stream.correction)
        .at[slots]
        .add(
            jnp.where(
                stream.active & lookup.supported, stream.correction, jnp.complex128(0)
            )
        )
    )
    values = jnp.concatenate((jnp.zeros((g,), dtype=jnp.complex128), stream.event_values))
    value_valid = jnp.concatenate((jnp.zeros((g,), dtype=jnp.bool_), stream.event_valid))
    accumulation, evidence = reduce_key_groups(
        groups,
        values,
        accumulation="compensated",
        initial=KeyGroupAccumulation(seed_high, seed_correction),
        value_valid=value_valid,
    )
    status = _fail(
        stream.status,
        groups.evidence.group_overflow,
        ProjectorMonteCarloStatus.INTERMEDIATE_GROUP_OVERFLOW,
    )
    status = _fail(
        status,
        ~groups.evidence.successful | ~evidence.successful,
        ProjectorMonteCarloStatus.NONFINITE_PROPAGATION,
    )
    return stream._replace(
        keys=groups.group_keys,
        active=groups.group_active,
        high=accumulation.high,
        correction=accumulation.correction,
        event_valid=jnp.zeros_like(stream.event_valid),
        cursor=jnp.asarray(0, dtype=jnp.int32),
        status=status,
        required_groups=jnp.maximum(
            stream.required_groups, groups.evidence.required_groups.astype(jnp.int64)
        ),
    )


def _emit(
    prepared: PreparedProjectorMonteCarlo,
    stream: _Stream,
    key: Array,
    value: Array,
    valid: Array,
) -> _Stream:
    status = _fail(
        stream.status,
        valid & ~jnp.isfinite(value),
        ProjectorMonteCarloStatus.NONFINITE_PROPAGATION,
    )
    current = stream._replace(
        event_keys=stream.event_keys.at[stream.cursor].set(key),
        event_values=stream.event_values.at[stream.cursor].set(
            jnp.where(valid, value, jnp.complex128(0))
        ),
        event_valid=stream.event_valid.at[stream.cursor].set(valid),
        event_ids=stream.event_ids.at[stream.cursor].set(stream.next_event_id),
        next_event_id=stream.next_event_id + jnp.int64(1),
        cursor=stream.cursor + jnp.int32(1),
        status=status,
        event_count=stream.event_count + valid.astype(jnp.int64),
        pre_norm=stream.pre_norm + jnp.where(valid, jnp.abs(value), jnp.float64(0)),
    )
    return jax.lax.cond(
        current.cursor == prepared.plan.event_capacity,
        lambda item: _flush(prepared, item),
        lambda item: item,
        current,
    )


def _source_policy(
    prepared: PreparedProjectorMonteCarlo, coefficient: Array
) -> tuple[Array, Array, Array]:
    plan = prepared.plan
    magnitude = jnp.abs(coefficient)
    match plan.spawn_policy:
        case "exact":
            exact = jnp.asarray(True, dtype=jnp.bool_)
        case "sampled":
            exact = jnp.asarray(False, dtype=jnp.bool_)
        case "semistochastic":
            bound = prepared.original_operator.raw_route_bound
            relative = (
                jnp.asarray(True, dtype=jnp.bool_)
                if bound == 0
                else magnitude / jnp.float64(bound)
                >= jnp.float64(plan.relative_threshold / plan.boost)
            )
            exact = relative | (
                magnitude > jnp.float64(plan.absolute_threshold / plan.boost)
            )
        case _ as unreachable:
            assert_never(unreachable)
    # Saturation is evidence only: an over-limit source never executes a clamped count.
    safe_limit = 2**63 - 1024
    unrepresentable = magnitude >= jnp.float64(safe_limit / plan.boost)
    safe_weight = jnp.minimum(
        jnp.minimum(magnitude, jnp.float64(safe_limit / plan.boost))
        * jnp.float64(plan.boost),
        jnp.float64(safe_limit),
    )
    requested = jnp.maximum(jnp.int64(1), jnp.ceil(safe_weight).astype(jnp.int64))
    attempts = jnp.where(unrepresentable, jnp.int64(2**63 - 1), requested)
    exceeded = ~exact & (attempts > plan.attempt_capacity)
    return exact, attempts, exceeded


def _source(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    replica: Array,
    source_index: Array,
    stream: _Stream,
) -> _Stream:
    operator = prepared.operator
    key = state.support_keys[replica, source_index]
    coefficient = state.coefficients[replica, source_index]
    source = operator.prepare_source(operator.domain.from_key(key))
    failure = (
        ProjectorMonteCarloStatus.OPERATOR_FAILURE
        if prepared.guide is None
        else ProjectorMonteCarloStatus.GUIDE_FAILURE
    )
    exact, attempts, exceeded = _source_policy(prepared, coefficient)
    current = stream._replace(
        status=_fail(stream.status, exceeded, ProjectorMonteCarloStatus.ATTEMPT_LIMIT),
        maximum_attempts=jnp.maximum(
            stream.maximum_attempts, jnp.where(exact, jnp.int64(0), attempts)
        ),
        exact_sources=stream.exact_sources + exact.astype(jnp.int64),
        sampled_sources=stream.sampled_sources + (~exact).astype(jnp.int64),
    )

    def diagonal_route(index: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        diagonal, good = carry
        route = prepared.original_operator.raw_route(source, index.astype(jnp.int32))
        return diagonal + jnp.where(
            route.valid & ~route.off_diagonal, route.matrix_element, jnp.complex128(0)
        ), good & route.successful

    diagonal, good = jax.lax.fori_loop(
        0,
        operator.raw_route_bound,
        diagonal_route,
        (jnp.asarray(0.0, dtype=jnp.complex128), source.valid),
    )
    current = current._replace(status=_fail(current.status, ~good, failure))
    dt = jnp.complex128(prepared.problem.dt)
    current = _emit(
        prepared,
        current,
        key,
        (
            jnp.complex128(1)
            + dt * (state.shifts[replica].astype(jnp.complex128) - diagonal)
        )
        * coefficient,
        jnp.asarray(True),
    )

    def emit_route(index: Array, item: _Stream) -> _Stream:
        def exact_route(_: None) -> RawQuantumExcitation:
            return operator.raw_route(source, index.astype(jnp.int32))

        def sampled_route(_: None) -> RawQuantumExcitation:
            return operator.sample_raw_excitation(
                _logical_key(prepared, state, replica, key, index, "spawn"), source
            )

        route = jax.lax.cond(exact, exact_route, sampled_route, None)
        valid = route.valid & route.off_diagonal
        denominator = jnp.where(
            exact, jnp.float64(1), attempts.astype(jnp.float64) * route.p_raw
        )
        value = (
            -dt
            * coefficient
            * route.matrix_element
            / jnp.where(valid, denominator, jnp.float64(1)).astype(jnp.complex128)
        )
        item = item._replace(status=_fail(item.status, ~route.successful, failure))
        return _emit(prepared, item, route.target_key, value, valid)

    work = jnp.where(exact, jnp.int64(operator.raw_route_bound), attempts)

    def continue_routes(carry: tuple[Array, _Stream]) -> Array:
        index, item = carry
        return (index < work) & (item.status == 0)

    def next_route(carry: tuple[Array, _Stream]) -> tuple[Array, _Stream]:
        index, item = carry
        return index + jnp.int64(1), emit_route(index, item)

    _, current = jax.lax.while_loop(
        continue_routes, next_route, (jnp.asarray(0, dtype=jnp.int64), current)
    )
    return current


def _compress(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    replica: Array,
    stream: _Stream,
) -> _Replica:
    coefficients = stream.high + stream.correction
    post_norm = jnp.sum(
        jnp.where(stream.active, jnp.abs(coefficients), jnp.float64(0)), dtype=jnp.float64
    )
    match prepared.plan.compression:
        case "none":
            compressed = coefficients
        case "threshold":
            theta = jnp.float64(prepared.plan.theta)

            def target(index: Array) -> Array:
                value = coefficients[index]
                magnitude = jnp.abs(value)

                def project(_: None) -> Array:
                    key = _logical_key(
                        prepared,
                        state,
                        replica,
                        stream.keys[index],
                        jnp.asarray(0, dtype=jnp.int64),
                        "compression",
                    )
                    draw = jr.uniform(key, (), dtype=jnp.float64)
                    return jnp.where(
                        draw < magnitude / theta,
                        (value / magnitude.astype(jnp.complex128))
                        * theta.astype(jnp.complex128),
                        jnp.complex128(0),
                    )

                eligible = stream.active[index] & (magnitude > 0) & (magnitude < theta)
                return jax.lax.cond(eligible, project, lambda _: value, None)

            compressed = jax.lax.map(
                target, jnp.arange(prepared.plan.group_capacity, dtype=jnp.int32)
            )
        case _ as unreachable:
            assert_never(unreachable)
    active = stream.active & (compressed != jnp.complex128(0))
    required = jnp.sum(active, dtype=jnp.int64)
    indices = jnp.nonzero(active, size=prepared.plan.support_capacity, fill_value=0)[0]
    final_active = jnp.arange(prepared.plan.support_capacity, dtype=jnp.int64) < required
    keys = jnp.where(final_active[:, None], stream.keys[indices], jnp.uint32(0))
    values = jnp.where(final_active, compressed[indices], jnp.complex128(0))
    population = jnp.sum(jnp.abs(values), dtype=jnp.float64)
    status = _fail(
        stream.status,
        ~jnp.all(jnp.isfinite(coefficients))
        | ~jnp.isfinite(stream.pre_norm)
        | ~jnp.isfinite(post_norm),
        ProjectorMonteCarloStatus.NONFINITE_PROPAGATION,
    )
    status = _fail(
        status,
        required > prepared.plan.support_capacity,
        ProjectorMonteCarloStatus.FINAL_SUPPORT_OVERFLOW,
    )
    status = _fail(status, population == 0, ProjectorMonteCarloStatus.EXTINCTION)
    status = _fail(
        status, ~jnp.isfinite(population), ProjectorMonteCarloStatus.NONFINITE_PROPAGATION
    )
    shift = state.shifts[replica]
    match prepared.plan.controller:
        case "fixed-shift":
            pass
        case "double-log":

            def feedback(_: None) -> Array:
                log_next = jnp.log(population)
                return (
                    shift
                    - jnp.float64(prepared.plan.damping / prepared.problem.dt)
                    * (log_next - jnp.log(state.populations[replica]))
                    - jnp.float64(prepared.plan.restoring / prepared.problem.dt)
                    * (log_next - jnp.log(jnp.float64(prepared.plan.target_population)))
                )

            shift = jax.lax.cond(status == 0, feedback, lambda _: shift, None)
        case _ as unreachable:
            assert_never(unreachable)
    status = _fail(
        status, ~jnp.isfinite(shift), ProjectorMonteCarloStatus.NONFINITE_CONTROLLER
    )
    return _Replica(
        keys,
        values,
        final_active,
        shift,
        population,
        status,
        stream.required_groups,
        required,
        stream.maximum_attempts,
        stream.event_count,
        stream.exact_sources,
        stream.sampled_sources,
        stream.pre_norm,
        post_norm,
    )


def _replica(
    prepared: PreparedProjectorMonteCarlo, state: ProjectorMonteCarloState, replica: Array
) -> _Replica:
    stream = _empty_stream(prepared)

    def source(index: Array, item: _Stream) -> _Stream:
        return jax.lax.cond(
            state.active[replica, index] & (item.status == 0),
            lambda current: _source(prepared, state, replica, index, current),
            lambda current: current,
            item,
        )

    stream = jax.lax.fori_loop(0, prepared.plan.support_capacity, source, stream)
    stream = jax.lax.cond(
        (stream.cursor > 0) & (stream.status == 0),
        lambda item: _flush(prepared, item),
        lambda item: item,
        stream,
    )
    return _compress(prepared, state, replica, stream)


def empty_projector_evidence(
    prepared: PreparedProjectorMonteCarlo,
) -> ProjectorMonteCarloEvidence:
    zeros = jnp.zeros((prepared.plan.replicas,), dtype=jnp.int64)
    norms = jnp.zeros((prepared.plan.replicas,), dtype=jnp.float64)
    return ProjectorMonteCarloEvidence(
        replica_status=zeros.astype(jnp.int32),
        required_groups=zeros,
        required_support=zeros,
        maximum_attempts=zeros,
        event_count=zeros,
        exact_sources=zeros,
        sampled_sources=zeros,
        pre_annihilation_norm=norms,
        post_annihilation_norm=norms,
        group_capacity=prepared.plan.group_capacity,
        support_capacity=prepared.plan.support_capacity,
        attempt_capacity=prepared.plan.attempt_capacity,
        raw_route_bound=prepared.original_operator.raw_route_bound,
    )


def _append_history(
    state: ProjectorMonteCarloState,
    candidate: ProjectorMonteCarloState,
    evidence: ProjectorMonteCarloEvidence,
    observation: ProjectorMonteCarloObservation,
) -> ProjectorMonteCarloHistory:
    h = state.history
    index = h.count
    return eqx.tree_at(
        lambda item: (
            item.applied_shifts,
            item.populations,
            item.projected_numerator,
            item.projected_denominator,
            item.pair_numerators,
            item.pair_denominators,
            item.pre_annihilation_norm,
            item.post_annihilation_norm,
            item.valid,
            item.count,
        ),
        h,
        (
            h.applied_shifts.at[:, index].set(state.shifts),
            h.populations.at[:, index].set(candidate.populations),
            h.projected_numerator.at[:, index].set(observation.projected_numerator),
            h.projected_denominator.at[:, index].set(observation.projected_denominator),
            h.pair_numerators.at[:, index, :].set(observation.pair_numerators),
            h.pair_denominators.at[:, index].set(observation.pair_denominators),
            h.pre_annihilation_norm.at[:, index].set(evidence.pre_annihilation_norm),
            h.post_annihilation_norm.at[:, index].set(evidence.post_annihilation_norm),
            h.valid.at[index].set(True),
            index + jnp.int64(1),
        ),
    )


def projector_step_kernel(
    prepared: PreparedProjectorMonteCarlo, state: ProjectorMonteCarloState
) -> ProjectorMonteCarloStepResult:
    status = jnp.asarray(0, dtype=jnp.int32)
    status = _fail(
        status,
        state.history.count >= prepared.plan.history_capacity,
        ProjectorMonteCarloStatus.HISTORY_EXHAUSTED,
    )
    status = _fail(
        status,
        state.step >= jnp.iinfo(jnp.int64).max,
        ProjectorMonteCarloStatus.STEP_COUNTER_EXHAUSTED,
    )
    status = _fail(
        status, jnp.any(state.populations <= 0), ProjectorMonteCarloStatus.EXTINCTION
    )

    def propagate(_: None) -> ProjectorMonteCarloStepResult:
        replicas = jax.lax.map(
            lambda index: _replica(prepared, state, index),
            jnp.arange(prepared.plan.replicas, dtype=jnp.int32),
        )
        evidence = ProjectorMonteCarloEvidence(
            replica_status=replicas.status,
            required_groups=replicas.required_groups,
            required_support=replicas.required_support,
            maximum_attempts=replicas.maximum_attempts,
            event_count=replicas.event_count,
            exact_sources=replicas.exact_sources,
            sampled_sources=replicas.sampled_sources,
            pre_annihilation_norm=replicas.pre_norm,
            post_annihilation_norm=replicas.post_norm,
            group_capacity=prepared.plan.group_capacity,
            support_capacity=prepared.plan.support_capacity,
            attempt_capacity=prepared.plan.attempt_capacity,
            raw_route_bound=prepared.original_operator.raw_route_bound,
        )
        failed = replicas.status != 0
        first = jnp.argmax(failed.astype(jnp.int32))
        propagation_status = jnp.where(
            jnp.any(failed), replicas.status[first], jnp.int32(0)
        )
        candidate = eqx.tree_at(
            lambda item: (
                item.support_keys,
                item.coefficients,
                item.active,
                item.shifts,
                item.populations,
                item.step,
            ),
            state,
            (
                replicas.keys,
                replicas.coefficients,
                replicas.active,
                replicas.shift,
                replicas.population,
                state.step + jnp.int64(1),
            ),
        )

        def observe(_: None) -> ProjectorMonteCarloStepResult:
            observation = observe_projector_state(prepared, candidate)
            numerical_status = jnp.where(
                observation.finite,
                jnp.int32(0),
                jnp.int32(ProjectorMonteCarloStatus.OBSERVATION_FAILURE),
            )
            history = _append_history(state, candidate, evidence, observation)
            proposed = eqx.tree_at(lambda item: item.history, candidate, history)
            transaction = TransactionalCandidate(
                state, proposed, evidence, observation.finite, prepared.scientific_id
            )
            commit = commit_candidate(transaction)
            return ProjectorMonteCarloStepResult(
                state=commit.state,
                status=numerical_status,
                accepted=commit.committed,
                evidence=commit.evidence,
            )

        return jax.lax.cond(
            propagation_status == 0,
            observe,
            lambda _: ProjectorMonteCarloStepResult(
                state=state,
                status=propagation_status,
                accepted=jnp.asarray(False),
                evidence=evidence,
            ),
            None,
        )

    return jax.lax.cond(
        status == 0,
        propagate,
        lambda _: ProjectorMonteCarloStepResult(
            state=state,
            status=status,
            accepted=jnp.asarray(False),
            evidence=empty_projector_evidence(prepared),
        ),
        None,
    )
