# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Native single-device projector Monte Carlo preparation and continuation."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..operators.quantum.lattice._address import QuantumConfigurationDomain
from ..operators.quantum.lattice._guide import GuidedQuantumColumnOperator
from ..sparse import KeyGroupPlan, reduce_key_groups
from ..typing import parse, PRNGKey, validate
from ._projector_monte_carlo_contracts import (
    PreparedProjectorMonteCarlo,
    ProjectorMonteCarloEvidence,
    ProjectorMonteCarloHistory,
    ProjectorMonteCarloPlan,
    ProjectorMonteCarloProblem,
    ProjectorMonteCarloResult,
    ProjectorMonteCarloState,
    ProjectorMonteCarloStepResult,
)
from ._projector_monte_carlo_step import empty_projector_evidence, projector_step_kernel


def _require_local_arrays(
    tree: PreparedProjectorMonteCarlo
    | ProjectorMonteCarloProblem
    | ProjectorMonteCarloState,
) -> None:
    if not jax.config.x64_enabled:
        raise ValueError("Projector Monte Carlo requires jax_enable_x64.")
    if jax.process_count() != 1:
        raise ValueError(
            "Projector Monte Carlo supports one process and one device only."
        )
    devices = set()
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, Array):
            if not leaf.is_fully_addressable or len(leaf.devices()) != 1:
                raise ValueError("Distributed projector arrays are not supported.")
            devices.update(leaf.devices())
    if len(devices) > 1:
        raise ValueError("Projector Monte Carlo requires a single owning device.")


@eqx.filter_jit
def _active_keys_admissible(
    domain: QuantumConfigurationDomain,
    keys: Array,
    active: Array,
    source_capacity: int,
) -> Array:
    """Reduce address validity with at most C decoded configurations live."""
    words = keys.reshape((-1, domain.codec.word_count))
    mask = active.reshape((-1,))
    count = words.shape[0]
    if count == 0:
        return jnp.asarray(True, dtype=jnp.bool_)
    chunk = min(source_capacity, count)

    def check(carry: tuple[Array, Array], _: None) -> tuple[tuple[Array, Array], None]:
        index, good = carry
        # Clamp the final slice instead of padding/copying the packed support.
        start = jnp.minimum(index * jnp.int64(chunk), jnp.int64(count - chunk))
        chunk_keys = jax.lax.dynamic_slice(
            words, (start, jnp.int64(0)), (chunk, domain.codec.word_count)
        )
        chunk_active = jax.lax.dynamic_slice(mask, (start,), (chunk,))

        def contains(_: None) -> Array:
            return jnp.all(~chunk_active | domain.contains_keys(chunk_keys))

        good = jax.lax.cond(
            good & jnp.any(chunk_active), contains, lambda _: good, operand=None
        )
        return (index + jnp.int64(1), good), None

    (_, admissible), _ = jax.lax.scan(
        check,
        (jnp.asarray(0, dtype=jnp.int64), jnp.asarray(True, dtype=jnp.bool_)),
        xs=None,
        length=(count + chunk - 1) // chunk,
    )
    return admissible


def _canonical_vector(
    problem: ProjectorMonteCarloProblem,
    keys: Array,
    coefficients: Array,
    active: Array,
    capacity: int,
    source_capacity: int,
) -> tuple[Array, Array, Array]:
    if keys.shape[0] == 0:
        raise ValueError("Physical sparse vectors must have nonempty support.")
    admissible = _active_keys_admissible(
        problem.hamiltonian.domain, keys, active, source_capacity
    ) & jnp.all(~active | jnp.isfinite(coefficients))
    if not bool(jax.device_get(admissible)):
        raise ValueError(
            "Active physical coefficients/addresses must be finite and admissible."
        )
    groups = KeyGroupPlan(
        keys.shape[0], keys.shape[0], (2**32 - 1,) * keys.shape[-1]
    ).build(keys, active, stable_ids=jnp.arange(keys.shape[0], dtype=jnp.int64))
    accumulated, evidence = reduce_key_groups(groups, coefficients, value_valid=active)
    retained = groups.group_active & (accumulated.value != jnp.complex128(0))
    count = int(jax.device_get(jnp.sum(retained, dtype=jnp.int64)))
    if not bool(jax.device_get(evidence.successful)) or count == 0:
        raise ValueError("Canonical physical sparse vector is nonfinite or extinct.")
    if count > capacity:
        raise ValueError("Initial physical support exceeds admitted capacity.")
    indices = jnp.nonzero(retained, size=capacity, fill_value=0)[0]
    mask = jnp.arange(capacity, dtype=jnp.int64) < count
    return (
        jnp.where(mask[:, None], groups.group_keys[indices], jnp.uint32(0)),
        jnp.where(mask, accumulated.value[indices], jnp.complex128(0)),
        mask,
    )


def _resource_bytes(
    problem: ProjectorMonteCarloProblem, plan: ProjectorMonteCarloPlan
) -> tuple[int, int]:
    r, s, g, e, t = (
        plan.replicas,
        plan.support_capacity,
        plan.group_capacity,
        plan.event_capacity,
        plan.history_capacity,
    )
    k = problem.hamiltonian.domain.codec.word_count
    p, o = r * (r - 1), len(problem.observables) + 1
    bound = problem.hamiltonian.raw_route_bound
    if s * (1 + max(bound, plan.attempt_capacity)) > 2**63 - 1:
        raise ValueError("Projector event/work count exceeds signed int64.")
    if max(problem.initial_keys.shape[0], problem.trial_keys.shape[0]) > 2**31 - 1:
        raise ValueError("Physical sparse vector grouping exceeds the native work index.")
    tables = sum(
        leaf.nbytes
        for leaf in jax.tree.leaves(
            (problem.hamiltonian, problem.observables, problem.guide)
        )
        if isinstance(leaf, Array)
    )
    physical = (problem.initial_keys.shape[0] + 2 * problem.trial_keys.shape[0] + s) * (
        4 * k + 17
    )
    history = t * (r * (4 * 8 + 2 * 16) + p * (o + 1) * 16 + 1) + 8
    state = r * (s * (4 * k + 17) + 16) + 16
    evidence = r * (4 + 6 * 8 + 2 * 8) + 4 + 1 + 2 * 8
    retained = tables + physical + history + state + evidence
    decoded = plan.source_capacity * len(problem.hamiltonian.domain.site_ids) * 4
    events = e * (4 * k + 16 + 1 + 8)
    accumulator = g * (4 * k + 32 + 1)
    grouping = (g + e) * (8 * k + 64) + g * (4 * k + 64)
    observation = (r * s + problem.trial_keys.shape[0]) * (4 * k + 64) + p * (o + 1) * 16
    workspace = (
        decoded
        + events
        + accumulator
        + grouping
        + observation
        + 2 * (state + history)
        + evidence
    )
    preparation_workspace = (
        problem.initial_keys.shape[0] + problem.trial_keys.shape[0]
    ) * (8 * k + 128) + decoded
    workspace = max(workspace, preparation_workspace)
    if retained > plan.maximum_retained_bytes:
        raise ValueError(f"Projector retained byte limit exceeded: required {retained}.")
    if workspace > plan.maximum_workspace_bytes:
        raise ValueError(
            f"Projector workspace byte limit exceeded: required {workspace}."
        )
    return retained, workspace


def prepare_projector_monte_carlo(
    problem: ProjectorMonteCarloProblem, plan: ProjectorMonteCarloPlan, /
) -> PreparedProjectorMonteCarlo:
    if not isinstance(problem, ProjectorMonteCarloProblem) or not isinstance(
        plan, ProjectorMonteCarloPlan
    ):
        raise TypeError("Preparation requires a projector problem and plan.")
    if not jax.config.x64_enabled:
        raise ValueError("Projector Monte Carlo requires jax_enable_x64.")
    _require_local_arrays(problem)
    retained, workspace = _resource_bytes(problem, plan)
    initial = _canonical_vector(
        problem,
        problem.initial_keys,
        problem.initial_coefficients,
        problem.initial_active,
        plan.support_capacity,
        plan.source_capacity,
    )
    trial = _canonical_vector(
        problem,
        problem.trial_keys,
        problem.trial_coefficients,
        problem.trial_active,
        problem.trial_keys.shape[0],
        plan.source_capacity,
    )
    initial_count = int(jax.device_get(jnp.sum(initial[2], dtype=jnp.int64)))
    trial_count = int(jax.device_get(jnp.sum(trial[2], dtype=jnp.int64)))
    physical_id = array_tree_fingerprint(
        (
            initial[0][:initial_count],
            initial[1][:initial_count],
            trial[0][:trial_count],
            trial[1][:trial_count],
        )
    )
    scientific_id = canonical_fingerprint(
        {
            "problem": problem.problem_id,
            "physical_vectors": physical_id,
            "policy": plan.policy_id,
            "replicas": plan.replicas,
            "rng": "accepted-step-replica-full-address-local-role",
            "representation": "physical"
            if problem.guide is None
            else "positive-guide-similarity",
        }
    )
    operator = (
        problem.hamiltonian
        if problem.guide is None
        else GuidedQuantumColumnOperator(problem.hamiltonian, problem.guide)
    )
    return PreparedProjectorMonteCarlo(
        problem=problem,
        plan=plan,
        original_operator=problem.hamiltonian,
        operator=operator,
        guide=problem.guide,
        initial_keys=initial[0],
        initial_coefficients=initial[1],
        initial_active=initial[2],
        trial_keys=trial[0],
        trial_coefficients=trial[1],
        trial_active=trial[2],
        observables=problem.observables,
        observable_units=(problem.energy_unit, *problem.observable_units),
        pair_ids=tuple(
            (a, b) for a in range(plan.replicas) for b in range(plan.replicas) if a != b
        ),
        scientific_id=scientific_id,
        prepared_id=canonical_fingerprint(
            {
                "scientific": scientific_id,
                "plan": plan.plan_id,
                "columns": problem.hamiltonian.prepared.prepared_id,
            }
        ),
        retained_bytes=retained,
        workspace_bytes=workspace,
    )


def _empty_history(prepared: PreparedProjectorMonteCarlo) -> ProjectorMonteCarloHistory:
    r, t = prepared.plan.replicas, prepared.plan.history_capacity
    p, o = len(prepared.pair_ids), len(prepared.observables) + 1
    real = jnp.zeros((r, t), dtype=jnp.float64)
    complex_ = jnp.zeros((r, t), dtype=jnp.complex128)
    guide_id = "unguided" if prepared.guide is None else prepared.guide.guide_id
    metric_id = "euclidean" if prepared.guide is None else prepared.guide.metric_id
    return ProjectorMonteCarloHistory(
        applied_shifts=real,
        populations=real,
        projected_numerator=complex_,
        projected_denominator=complex_,
        pair_numerators=jnp.zeros((p, t, o), dtype=jnp.complex128),
        pair_denominators=jnp.zeros((p, t), dtype=jnp.complex128),
        pre_annihilation_norm=real,
        post_annihilation_norm=real,
        valid=jnp.zeros((t,), dtype=jnp.bool_),
        count=jnp.asarray(0, dtype=jnp.int64),
        scientific_id=prepared.scientific_id,
        domain_id=prepared.original_operator.domain.domain_id,
        operator_id=prepared.original_operator.operator_id,
        guide_id=guide_id,
        metric_id=metric_id,
    )


def initialize_projector_monte_carlo(
    prepared: PreparedProjectorMonteCarlo, key: PRNGKey, /
) -> ProjectorMonteCarloState:
    if not isinstance(prepared, PreparedProjectorMonteCarlo):
        raise TypeError("prepared must be PreparedProjectorMonteCarlo.")
    root = parse(key, PRNGKey, "key")
    _require_local_arrays(prepared)
    coefficients = prepared.initial_coefficients
    if prepared.guide is not None:
        guide = prepared.guide

        def transform(index: Array, carry: tuple[Array, Array]) -> tuple[Array, Array]:
            values, good = carry

            def active(_: None) -> tuple[Array, Array]:
                log_value, valid = guide.log_value(
                    prepared.original_operator.domain.from_key(
                        prepared.initial_keys[index]
                    )
                )
                scale = jnp.exp(log_value)
                value = coefficients[index] * scale.astype(jnp.complex128)
                return value, valid & jnp.isfinite(scale) & (scale > 0) & jnp.isfinite(
                    value
                ) & (value != jnp.complex128(0))

            value, valid = jax.lax.cond(
                prepared.initial_active[index],
                active,
                lambda _: (jnp.asarray(0, dtype=jnp.complex128), jnp.asarray(True)),
                None,
            )
            return values.at[index].set(value), good & valid

        coefficients, valid = jax.lax.fori_loop(
            0,
            prepared.plan.support_capacity,
            transform,
            (jnp.zeros_like(coefficients), jnp.asarray(True)),
        )
        if not bool(jax.device_get(valid)):
            raise ValueError(
                "Guide transformation is invalid or outside numerical range."
            )
    population = jnp.sum(jnp.abs(coefficients), dtype=jnp.float64)
    if not bool(jax.device_get(jnp.isfinite(population) & (population > 0))):
        raise ValueError("Initial represented population must be finite and positive.")
    r, s = prepared.plan.replicas, prepared.plan.support_capacity
    history = _empty_history(prepared)
    state = ProjectorMonteCarloState(
        support_keys=jnp.broadcast_to(
            prepared.initial_keys, (r, s, prepared.initial_keys.shape[-1])
        ),
        coefficients=jnp.broadcast_to(coefficients, (r, s)),
        active=jnp.broadcast_to(prepared.initial_active, (r, s)),
        shifts=jnp.full((r,), prepared.plan.initial_shift, dtype=jnp.float64),
        populations=jnp.broadcast_to(population, (r,)),
        step=jnp.asarray(0, dtype=jnp.int64),
        root_key=root,
        history=history,
        scientific_id=prepared.scientific_id,
        prepared_id=prepared.prepared_id,
        plan_id=prepared.plan.plan_id,
        domain_id=history.domain_id,
        codec_id=prepared.original_operator.domain.codec.codec_id,
        operator_id=history.operator_id,
        guide_id=history.guide_id,
        metric_id=history.metric_id,
    )
    validate_projector_state(prepared, state)
    return state


def _strictly_ordered(keys: Array, active: Array) -> Array:
    left, right = keys[:, :-1, :], keys[:, 1:, :]
    difference = left != right
    first = jnp.argmax(difference.astype(jnp.int32), axis=-1)
    a = jnp.take_along_axis(left, first[..., None], axis=-1)[..., 0]
    b = jnp.take_along_axis(right, first[..., None], axis=-1)[..., 0]
    ordered = jnp.any(difference, axis=-1) & (a < b)
    return jnp.all(~active[:, 1:] | (active[:, :-1] & ordered))


def validate_projector_state(
    prepared: PreparedProjectorMonteCarlo, state: ProjectorMonteCarloState, /
) -> None:
    """Validate a complete committed boundary, intentionally synchronizing once."""
    if not isinstance(prepared, PreparedProjectorMonteCarlo) or not isinstance(
        state, ProjectorMonteCarloState
    ):
        raise TypeError("Validation requires native prepared/state values.")
    validate(state)
    _require_local_arrays(state)
    guide_id = "unguided" if prepared.guide is None else prepared.guide.guide_id
    metric_id = "euclidean" if prepared.guide is None else prepared.guide.metric_id
    expected = (
        prepared.scientific_id,
        prepared.prepared_id,
        prepared.plan.plan_id,
        prepared.original_operator.domain.domain_id,
        prepared.original_operator.domain.codec.codec_id,
        prepared.original_operator.operator_id,
        guide_id,
        metric_id,
    )
    actual = (
        state.scientific_id,
        state.prepared_id,
        state.plan_id,
        state.domain_id,
        state.codec_id,
        state.operator_id,
        state.guide_id,
        state.metric_id,
    )
    if actual != expected:
        raise ValueError(
            "Projector state scientific/resource bindings do not match preparation."
        )
    r, s, t = (
        prepared.plan.replicas,
        prepared.plan.support_capacity,
        prepared.plan.history_capacity,
    )
    if (
        state.coefficients.shape != (r, s)
        or state.support_keys.shape
        != (r, s, prepared.original_operator.domain.codec.word_count)
        or state.history.applied_shifts.shape != (r, t)
        or state.history.pair_numerators.shape
        != (len(prepared.pair_ids), t, len(prepared.observables) + 1)
    ):
        raise ValueError("State/history capacities do not match the admitted plan.")
    h = state.history
    if (h.scientific_id, h.domain_id, h.operator_id, h.guide_id, h.metric_id) != (
        state.scientific_id,
        state.domain_id,
        state.operator_id,
        state.guide_id,
        state.metric_id,
    ):
        raise ValueError("History semantic bindings do not match the state.")
    population = jnp.sum(
        jnp.where(state.active, jnp.abs(state.coefficients), jnp.float64(0)),
        axis=1,
        dtype=jnp.float64,
    )
    support_good = (
        _active_keys_admissible(
            prepared.original_operator.domain,
            state.support_keys,
            state.active,
            prepared.plan.source_capacity,
        )
        & jnp.all(jnp.isfinite(state.coefficients))
        & jnp.all(state.active == (state.coefficients != jnp.complex128(0)))
        & jnp.all(
            jnp.where(state.active[..., None], jnp.uint32(0), state.support_keys)
            == jnp.uint32(0)
        )
        & _strictly_ordered(state.support_keys, state.active)
    )
    cursors = (
        (state.step == h.count)
        & (h.count >= 0)
        & (h.count <= t)
        & jnp.all(h.valid == (jnp.arange(t, dtype=jnp.int64) < h.count))
    )
    numerical = (
        jnp.all(jnp.isfinite(state.shifts))
        & jnp.all(jnp.isfinite(state.populations))
        & jnp.all(state.populations == population)
        & jnp.all(state.populations >= 0)
    )
    for record in (
        h.applied_shifts,
        h.populations,
        h.projected_numerator,
        h.projected_denominator,
        h.pair_numerators,
        h.pair_denominators,
        h.pre_annihilation_norm,
        h.post_annihilation_norm,
    ):
        numerical = numerical & jnp.all(jnp.isfinite(record))
    if not bool(jax.device_get(support_good & cursors & numerical)):
        raise ValueError(
            "Invalid committed support, controller, history, or accepted-step cursor."
        )


_compiled_step = eqx.filter_jit(projector_step_kernel)


def step_projector_monte_carlo(
    prepared: PreparedProjectorMonteCarlo, state: ProjectorMonteCarloState, /
) -> ProjectorMonteCarloStepResult:
    validate_projector_state(prepared, state)
    return _compiled_step(prepared, state)


def _solve_kernel(
    prepared: PreparedProjectorMonteCarlo, state: ProjectorMonteCarloState, steps: Array
) -> ProjectorMonteCarloResult:
    def continue_run(
        carry: tuple[Array, ProjectorMonteCarloState, Array, ProjectorMonteCarloEvidence],
    ) -> Array:
        accepted, _, status, _ = carry
        return (accepted < steps) & (status == 0)

    def advance(
        carry: tuple[Array, ProjectorMonteCarloState, Array, ProjectorMonteCarloEvidence],
    ) -> tuple[Array, ProjectorMonteCarloState, Array, ProjectorMonteCarloEvidence]:
        accepted, current, _, _ = carry
        result = projector_step_kernel(prepared, current)
        return (
            accepted + result.accepted.astype(jnp.int64),
            result.state,
            result.status,
            result.evidence,
        )

    accepted, final, status, evidence = jax.lax.while_loop(
        continue_run,
        advance,
        (
            jnp.asarray(0, dtype=jnp.int64),
            state,
            jnp.asarray(0, dtype=jnp.int32),
            empty_projector_evidence(prepared),
        ),
    )
    return ProjectorMonteCarloResult(
        state=final,
        status=status,
        evidence=evidence,
        requested_steps=jnp.asarray(steps, dtype=jnp.int64),
        accepted_steps=accepted,
    )


_compiled_solve = eqx.filter_jit(_solve_kernel)


def solve_projector_monte_carlo(
    prepared: PreparedProjectorMonteCarlo,
    state: ProjectorMonteCarloState,
    /,
    *,
    steps: int,
) -> ProjectorMonteCarloResult:
    if isinstance(steps, bool) or not isinstance(steps, int):
        raise TypeError("steps must be an integer.")
    if steps < 0 or steps > 2**63 - 1:
        raise ValueError("steps must fit nonnegative signed int64.")
    validate_projector_state(prepared, state)
    return _compiled_solve(prepared, state, jnp.asarray(steps, dtype=jnp.int64))
