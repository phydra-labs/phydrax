#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics import log_normalize
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)


MATCH = 0
INSERT = 1
DELETE = 2
STATE_COUNT = 3
PATH_PAD = -1

PAIR_HMM_OK = 0
PAIR_HMM_CONDITIONAL_BAND = 1
PAIR_HMM_INVALID_INPUT = 2
PAIR_HMM_CAPACITY_EXCEEDED = 3
PAIR_HMM_IMPOSSIBLE = 4
PAIR_HMM_TRACEBACK_INCOMPLETE = 5
PAIR_HMM_INCONSISTENT = 6


def _stable_logsumexp(values: Array, axis: int | tuple[int, ...] | None = None) -> Array:
    axes = tuple(range(values.ndim)) if axis is None else axis
    _, log_sum, _ = log_normalize(values, axes=axes)
    return log_sum


def _normalized_log_weights(logits: Array, axis: int, /) -> Array:
    normalizer = _stable_logsumexp(logits, axis=axis)
    expanded = jnp.expand_dims(normalizer, axis=axis)
    return jnp.where(jnp.isfinite(expanded), logits - expanded, -jnp.inf)


class PairHMM(StrictModule):
    """Normalized categorical parameters for a three-state pair hidden Markov model."""

    initial_logits: Array
    transition_logits: Array
    terminal_logits: Array
    match_emission_logits: Array
    insertion_emission_logits: Array
    deletion_emission_logits: Array
    alphabet_size: int = eqx.field(static=True)

    def __init__(
        self,
        initial_logits: ArrayLike,
        transition_logits: ArrayLike,
        terminal_logits: ArrayLike,
        match_emission_logits: ArrayLike,
        insertion_emission_logits: ArrayLike,
        deletion_emission_logits: ArrayLike,
        /,
    ):
        raw_initial = jnp.asarray(initial_logits)
        dtype = (
            raw_initial.dtype
            if jnp.issubdtype(raw_initial.dtype, jnp.floating)
            else jnp.asarray(0.0).dtype
        )
        initial = jnp.asarray(raw_initial, dtype=dtype)
        transition = jnp.asarray(transition_logits, dtype=dtype)
        terminal = jnp.asarray(terminal_logits, dtype=dtype)
        match = jnp.asarray(match_emission_logits, dtype=dtype)
        insertion = jnp.asarray(insertion_emission_logits, dtype=dtype)
        deletion = jnp.asarray(deletion_emission_logits, dtype=dtype)
        if initial.shape != (STATE_COUNT,):
            raise ValueError("initial_logits must have shape (3,).")
        if transition.shape != (STATE_COUNT, STATE_COUNT):
            raise ValueError("transition_logits must have shape (3, 3).")
        if terminal.shape != (STATE_COUNT,):
            raise ValueError("terminal_logits must have shape (3,).")
        if match.ndim != 2 or match.shape[0] != match.shape[1] or match.shape[0] < 1:
            raise ValueError("match_emission_logits must be a nonempty square matrix.")
        alphabet_size = match.shape[0]
        if insertion.shape != (alphabet_size,) or deletion.shape != (alphabet_size,):
            raise ValueError(
                "insertion and deletion emission logits must match the alphabet size."
            )
        self.initial_logits = initial
        self.transition_logits = transition
        self.terminal_logits = terminal
        self.match_emission_logits = match
        self.insertion_emission_logits = insertion
        self.deletion_emission_logits = deletion
        self.alphabet_size = alphabet_size

    def normalized_log_parameters(
        self, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Return log probabilities, preserving structural negative infinities."""
        return (
            _normalized_log_weights(self.initial_logits, 0),
            _normalized_log_weights(self.transition_logits, 1),
            _normalized_log_weights(self.terminal_logits, 0),
            _normalized_log_weights(self.match_emission_logits.reshape(-1), 0).reshape(
                self.alphabet_size, self.alphabet_size
            ),
            _normalized_log_weights(self.insertion_emission_logits, 0),
            _normalized_log_weights(self.deletion_emission_logits, 0),
        )


class PairHMMExecutionPlan(StrictModule):
    """Static exact, checkpoint-labeled exact, or conditional diagonal-band plan."""

    mode: str = eqx.field(static=True)
    maximum_left_length: int = eqx.field(static=True)
    maximum_right_length: int = eqx.field(static=True)
    traceback_capacity: int = eqx.field(static=True)
    band_radius: int = eqx.field(static=True)
    checkpoint_stride: int = eqx.field(static=True)
    boundary_mass_tolerance: float = eqx.field(static=True)
    expansion_step: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_left_length: int,
        maximum_right_length: int,
        /,
        *,
        traceback_capacity: int,
        mode: str = "full",
        band_radius: int | None = None,
        checkpoint_stride: int = 1,
        boundary_mass_tolerance: float = 1e-3,
        expansion_step: int = 1,
    ):
        left = int(maximum_left_length)
        right = int(maximum_right_length)
        capacity = int(traceback_capacity)
        stride = int(checkpoint_stride)
        tolerance = float(boundary_mass_tolerance)
        step = int(expansion_step)
        if mode not in ("full", "checkpointed", "diagonal_band"):
            raise ValueError("mode must be 'full', 'checkpointed', or 'diagonal_band'.")
        if left < 1 or right < 1 or capacity < 1:
            raise ValueError("maximum lengths and traceback capacity must be positive.")
        if stride < 1 or step < 1:
            raise ValueError("checkpoint_stride and expansion_step must be positive.")
        if tolerance < 0.0 or tolerance > 1.0:
            raise ValueError("boundary_mass_tolerance must lie in [0, 1].")
        radius = -1 if band_radius is None else int(band_radius)
        if mode == "diagonal_band" and radius < 0:
            raise ValueError("diagonal_band mode requires a non-negative band_radius.")
        if mode != "diagonal_band" and radius >= 0:
            raise ValueError("band_radius is only valid for diagonal_band mode.")
        self.mode = mode
        self.maximum_left_length = left
        self.maximum_right_length = right
        self.traceback_capacity = capacity
        self.band_radius = radius
        self.checkpoint_stride = stride
        self.boundary_mass_tolerance = tolerance
        self.expansion_step = step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pair-hmm-execution",
                "mode": mode,
                "maximum_left_length": left,
                "maximum_right_length": right,
                "traceback_capacity": capacity,
                "band_radius": radius,
                "checkpoint_stride": stride,
                "boundary_mass_tolerance": tolerance,
                "expansion_step": step,
            }
        )

    @classmethod
    def full(
        cls,
        maximum_left_length: int,
        maximum_right_length: int,
        /,
        *,
        traceback_capacity: int,
    ) -> PairHMMExecutionPlan:
        return cls(
            maximum_left_length,
            maximum_right_length,
            traceback_capacity=traceback_capacity,
        )

    @classmethod
    def checkpointed(
        cls,
        maximum_left_length: int,
        maximum_right_length: int,
        /,
        *,
        traceback_capacity: int,
        checkpoint_stride: int,
    ) -> PairHMMExecutionPlan:
        return cls(
            maximum_left_length,
            maximum_right_length,
            traceback_capacity=traceback_capacity,
            mode="checkpointed",
            checkpoint_stride=checkpoint_stride,
        )

    @classmethod
    def diagonal_band(
        cls,
        maximum_left_length: int,
        maximum_right_length: int,
        /,
        *,
        traceback_capacity: int,
        band_radius: int,
        boundary_mass_tolerance: float = 1e-3,
        expansion_step: int = 1,
    ) -> PairHMMExecutionPlan:
        return cls(
            maximum_left_length,
            maximum_right_length,
            traceback_capacity=traceback_capacity,
            mode="diagonal_band",
            band_radius=band_radius,
            boundary_mass_tolerance=boundary_mass_tolerance,
            expansion_step=expansion_step,
        )

    @property
    def band_limited(self) -> bool:
        return self.mode == "diagonal_band" and self.band_radius < max(
            self.maximum_left_length, self.maximum_right_length
        )


class PairHMMEvidence(StrictModule):
    """Named model, capacity, posterior, and path-domain evidence."""

    input_valid: Array
    capacity_sufficient: Array
    path_exists: Array
    traceback_complete: Array
    forward_backward_consistent: Array
    posterior_conserved: Array
    full_domain: Array
    expansion_required: Array


class PairHMMResult(StrictModule):
    """Exact or explicitly conditional pair-HMM inference on fixed lattice shapes."""

    log_partition: Array
    forward_log_partition: Array
    backward_log_partition: Array
    forward_table: Array
    backward_table: Array
    state_marginals: Array
    transition_marginals: Array
    initial_marginals: Array
    terminal_marginals: Array
    expected_state_counts: Array
    expected_transition_counts: Array
    viterbi_score: Array
    viterbi_states: Array
    viterbi_left_indices: Array
    viterbi_right_indices: Array
    viterbi_length: Array
    left_length: Array
    right_length: Array
    valid: Array
    status: Array
    evidence: PairHMMEvidence
    exact: Array
    truncated: Array
    boundary_mass: Array
    expansion_required: Array
    suggested_band_radius: Array
    forward_backward_error: Array
    posterior_conservation_error: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _pair_hmm_contract(plan: PairHMMExecutionPlan, /) -> BioinformaticsMethodContract:
    banded = plan.band_limited
    checkpoint = plan.mode == "checkpointed"
    return BioinformaticsMethodContract(
        "normalized-three-state-pair-hmm",
        MethodKind.EXACT_MODEL,
        ExecutionKind.FLOATING_POINT_DIRECT,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.PROBABILISTIC,
        conditioning_statement=(
            f"posterior conditioned to |left_index-right_index| <= {plan.band_radius}"
            if banded
            else "posterior over every path in the declared finite sequence rectangle"
        ),
        truncation_statement=(
            "diagonal path domain is truncated; boundary posterior occupancy is reported"
            if banded
            else "none; checkpoint recomputation preserves the exact declared model"
            if checkpoint
            else "none"
        ),
        capacity_semantics=(
            f"left<={plan.maximum_left_length}, right<={plan.maximum_right_length}, "
            f"Viterbi path<={plan.traceback_capacity}; overflow is an invalid result"
        ),
        assumptions=(
            "initial, transition-row, terminal, and emission logits are categorically normalized",
            "the empty/empty observation has a distinguished probability-one empty path",
        ),
        nondifferentiable_outputs=(
            "viterbi_states",
            "viterbi_left_indices",
            "viterbi_right_indices",
            "viterbi_length",
            "status",
        ),
    )


def _pad_vector(values: Array, capacity: int, fill: float, /) -> tuple[Array, bool]:
    size = values.shape[0]
    if size >= capacity:
        return values[:capacity], size > capacity
    return jnp.pad(values, (0, capacity - size), constant_values=fill), False


def _pad_matrix(
    values: Array, rows: int, columns: int, fill: float, /
) -> tuple[Array, bool]:
    overflow = values.shape[0] > rows or values.shape[1] > columns
    clipped = values[:rows, :columns]
    return (
        jnp.pad(
            clipped,
            ((0, rows - clipped.shape[0]), (0, columns - clipped.shape[1])),
            constant_values=fill,
        ),
        overflow,
    )


def _prepare_potentials(
    match: Array,
    insertion: Array,
    deletion: Array,
    left_mask: Array,
    right_mask: Array,
    plan: PairHMMExecutionPlan,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array]:
    nq = plan.maximum_left_length
    nt = plan.maximum_right_length
    match, match_overflow = _pad_matrix(match, nq, nt, -jnp.inf)
    insertion, insertion_overflow = _pad_vector(insertion, nq, -jnp.inf)
    deletion, deletion_overflow = _pad_vector(deletion, nt, -jnp.inf)
    left_mask, left_overflow = _pad_vector(left_mask, nq, False)
    right_mask, right_overflow = _pad_vector(right_mask, nt, False)
    left_mask = left_mask.astype(bool)
    right_mask = right_mask.astype(bool)
    left_length = jnp.sum(left_mask, dtype=jnp.int32)
    right_length = jnp.sum(right_mask, dtype=jnp.int32)
    left_prefix = jnp.array_equal(left_mask, jnp.arange(nq) < left_length)
    right_prefix = jnp.array_equal(right_mask, jnp.arange(nt) < right_length)
    active_match = left_mask[:, None] & right_mask[None, :]
    match_admissible = ~jnp.isnan(match) & (match != jnp.inf)
    insertion_admissible = ~jnp.isnan(insertion) & (insertion != jnp.inf)
    deletion_admissible = ~jnp.isnan(deletion) & (deletion != jnp.inf)
    input_valid = (
        ~jnp.asarray(
            match_overflow
            | insertion_overflow
            | deletion_overflow
            | left_overflow
            | right_overflow
        )
        & left_prefix
        & right_prefix
        & jnp.all(~active_match | match_admissible)
        & jnp.all(~left_mask | insertion_admissible)
        & jnp.all(~right_mask | deletion_admissible)
    )
    match = jnp.where(active_match, match, -jnp.inf)
    insertion = jnp.where(left_mask, insertion, -jnp.inf)
    deletion = jnp.where(right_mask, deletion, -jnp.inf)
    return (
        match,
        insertion,
        deletion,
        left_mask,
        right_mask,
        left_length,
        right_length,
        input_valid,
    )


def _emission_grid(
    match: Array,
    insertion: Array,
    deletion: Array,
    plan: PairHMMExecutionPlan,
    /,
) -> Array:
    nq = plan.maximum_left_length
    nt = plan.maximum_right_length
    dtype = jnp.result_type(match, insertion, deletion)
    emissions = jnp.full((nq + 1, nt + 1, STATE_COUNT), -jnp.inf, dtype=dtype)
    emissions = emissions.at[1:, 1:, MATCH].set(match)
    emissions = emissions.at[1:, :, INSERT].set(
        jnp.broadcast_to(insertion[:, None], (nq, nt + 1))
    )
    emissions = emissions.at[:, 1:, DELETE].set(
        jnp.broadcast_to(deletion[None, :], (nq + 1, nt))
    )
    return emissions


def _allowed_grid(
    left_length: Array,
    right_length: Array,
    plan: PairHMMExecutionPlan,
    /,
) -> Array:
    nq = plan.maximum_left_length
    nt = plan.maximum_right_length
    ii = jnp.arange(nq + 1, dtype=jnp.int32)[:, None, None]
    jj = jnp.arange(nt + 1, dtype=jnp.int32)[None, :, None]
    state = jnp.arange(STATE_COUNT, dtype=jnp.int32)[None, None, :]
    coordinate = (ii <= left_length) & (jj <= right_length)
    consumed = (
        ((state == MATCH) & (ii > 0) & (jj > 0))
        | ((state == INSERT) & (ii > 0))
        | ((state == DELETE) & (jj > 0))
    )
    if plan.band_radius < 0:
        band = jnp.asarray(True)
    else:
        band = jnp.abs(ii - jj) <= plan.band_radius
    return coordinate & consumed & band


def _forward_viterbi(
    emissions: Array,
    allowed: Array,
    initial_log: Array,
    transition_log: Array,
    plan: PairHMMExecutionPlan,
    /,
) -> tuple[Array, Array, Array]:
    nq = plan.maximum_left_length
    nt = plan.maximum_right_length
    dtype = emissions.dtype
    alpha = jnp.full((nq + 1, nt + 1, STATE_COUNT), -jnp.inf, dtype=dtype)
    viterbi = jnp.full_like(alpha, -jnp.inf)
    pointers = jnp.full(alpha.shape, -1, dtype=jnp.int8)
    di = jnp.asarray((1, 1, 0), dtype=jnp.int32)
    dj = jnp.asarray((1, 0, 1), dtype=jnp.int32)
    total = (nq + 1) * (nt + 1)

    def fill(
        flat_index: int, carry: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        forward, maximum, ptr = carry
        i = flat_index // (nt + 1)
        j = flat_index % (nt + 1)

        def fill_state(state: int, state_carry: tuple[Array, Array, Array]):
            forward_, maximum_, ptr_ = state_carry
            pi = jnp.maximum(i - di[state], 0)
            pj = jnp.maximum(j - dj[state], 0)
            first = (i - di[state] == 0) & (j - dj[state] == 0)
            forward_candidates = forward_[pi, pj] + transition_log[:, state]
            maximum_candidates = maximum_[pi, pj] + transition_log[:, state]
            forward_base = jnp.where(
                first, initial_log[state], _stable_logsumexp(forward_candidates)
            )
            maximum_choice = jnp.argmax(maximum_candidates)
            maximum_base = jnp.where(
                first, initial_log[state], maximum_candidates[maximum_choice]
            )
            value = forward_base + emissions[i, j, state]
            max_value = maximum_base + emissions[i, j, state]
            feasible = allowed[i, j, state]
            forward_ = forward_.at[i, j, state].set(jnp.where(feasible, value, -jnp.inf))
            maximum_ = maximum_.at[i, j, state].set(
                jnp.where(feasible, max_value, -jnp.inf)
            )
            ptr_ = ptr_.at[i, j, state].set(
                jnp.where(feasible & ~first, maximum_choice, -1).astype(jnp.int8)
            )
            return forward_, maximum_, ptr_

        return jax.lax.fori_loop(0, STATE_COUNT, fill_state, (forward, maximum, ptr))

    initial_carry = (alpha, viterbi, pointers)
    if plan.mode != "checkpointed":
        return jax.lax.fori_loop(1, total, fill, initial_carry)

    stride = plan.checkpoint_stride
    block_count = (total - 1 + stride - 1) // stride

    def fill_block(
        block_index: int, carry: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        def fill_offset(
            offset: int, current: tuple[Array, Array, Array]
        ) -> tuple[Array, Array, Array]:
            flat_index = 1 + block_index * stride + offset
            return jax.lax.cond(
                flat_index < total,
                lambda value: fill(flat_index, value),
                lambda value: value,
                current,
            )

        return jax.lax.fori_loop(0, stride, fill_offset, carry)

    return jax.lax.fori_loop(0, block_count, jax.checkpoint(fill_block), initial_carry)


def _backward(
    emissions: Array,
    allowed: Array,
    transition_log: Array,
    terminal_log: Array,
    left_length: Array,
    right_length: Array,
    plan: PairHMMExecutionPlan,
    /,
) -> Array:
    nq = plan.maximum_left_length
    nt = plan.maximum_right_length
    beta = jnp.full((nq + 1, nt + 1, STATE_COUNT), -jnp.inf, dtype=emissions.dtype)
    di = jnp.asarray((1, 1, 0), dtype=jnp.int32)
    dj = jnp.asarray((1, 0, 1), dtype=jnp.int32)
    states = jnp.arange(STATE_COUNT, dtype=jnp.int32)
    total = (nq + 1) * (nt + 1)

    def fill(reverse_index: int, values: Array) -> Array:
        flat_index = total - 1 - reverse_index
        i = flat_index // (nt + 1)
        j = flat_index % (nt + 1)
        next_i = i + di
        next_j = j + dj
        safe_i = jnp.minimum(next_i, nq)
        safe_j = jnp.minimum(next_j, nt)
        next_allowed = (next_i <= nq) & (next_j <= nt) & allowed[safe_i, safe_j, states]
        next_terms = jnp.where(
            next_allowed,
            emissions[safe_i, safe_j, states] + values[safe_i, safe_j, states],
            -jnp.inf,
        )
        terminal_cell = (i == left_length) & (j == right_length)

        def fill_state(state: int, current: Array) -> Array:
            continuation = _stable_logsumexp(transition_log[state] + next_terms)
            value = jnp.where(terminal_cell, terminal_log[state], continuation)
            return current.at[i, j, state].set(
                jnp.where(allowed[i, j, state], value, -jnp.inf)
            )

        return jax.lax.fori_loop(0, STATE_COUNT, fill_state, values)

    if plan.mode != "checkpointed":
        return jax.lax.fori_loop(0, total, fill, beta)

    stride = plan.checkpoint_stride
    block_count = (total + stride - 1) // stride

    def fill_block(block_index: int, values: Array) -> Array:
        def fill_offset(offset: int, current: Array) -> Array:
            reverse_index = block_index * stride + offset
            return jax.lax.cond(
                reverse_index < total,
                lambda value: fill(reverse_index, value),
                lambda value: value,
                current,
            )

        return jax.lax.fori_loop(0, stride, fill_offset, values)

    return jax.lax.fori_loop(0, block_count, jax.checkpoint(fill_block), beta)


def _transition_posteriors(
    alpha: Array,
    beta: Array,
    emissions: Array,
    allowed: Array,
    transition_log: Array,
    log_partition: Array,
    plan: PairHMMExecutionPlan,
    /,
) -> Array:
    nq = plan.maximum_left_length
    nt = plan.maximum_right_length
    safe_partition = jnp.where(jnp.isfinite(log_partition), log_partition, 0.0)
    marginals = jnp.zeros((nq + 1, nt + 1, STATE_COUNT, STATE_COUNT), dtype=alpha.dtype)
    di = jnp.asarray((1, 1, 0), dtype=jnp.int32)
    dj = jnp.asarray((1, 0, 1), dtype=jnp.int32)
    total = (nq + 1) * (nt + 1)

    def fill(flat_index: int, values: Array) -> Array:
        i = flat_index // (nt + 1)
        j = flat_index % (nt + 1)

        def fill_state(state: int, current: Array) -> Array:
            pi = jnp.maximum(i - di[state], 0)
            pj = jnp.maximum(j - dj[state], 0)
            has_previous = ~((i - di[state] == 0) & (j - dj[state] == 0))
            log_values = (
                alpha[pi, pj]
                + transition_log[:, state]
                + emissions[i, j, state]
                + beta[i, j, state]
                - safe_partition
            )
            feasible = allowed[i, j, state] & has_previous & jnp.isfinite(log_partition)
            posterior = jnp.where(
                feasible & jnp.isfinite(log_values), jnp.exp(log_values), 0.0
            )
            return current.at[i, j, :, state].set(posterior)

        return jax.lax.fori_loop(0, STATE_COUNT, fill_state, values)

    return jax.lax.fori_loop(1, total, fill, marginals)


def _viterbi_traceback(
    viterbi: Array,
    pointers: Array,
    terminal_log: Array,
    left_length: Array,
    right_length: Array,
    capacity_ok: Array,
    plan: PairHMMExecutionPlan,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    capacity = plan.traceback_capacity
    empty = (left_length == 0) & (right_length == 0)
    terminal_scores = viterbi[left_length, right_length] + terminal_log
    terminal_state = jnp.argmax(terminal_scores).astype(jnp.int32)
    score = jnp.where(empty, 0.0, terminal_scores[terminal_state])
    reverse_states = jnp.full((capacity,), PATH_PAD, dtype=jnp.int8)
    reverse_left = jnp.full((capacity,), -1, dtype=jnp.int32)
    reverse_right = jnp.full((capacity,), -1, dtype=jnp.int32)
    initial_done = empty | ~capacity_ok | ~jnp.isfinite(score)

    def step(
        _: int,
        carry: tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        i, j, state, length, done, complete, states, left, right = carry
        write = ~done
        slot = jnp.minimum(length, capacity - 1)
        states = states.at[slot].set(
            jnp.where(write, state.astype(jnp.int8), states[slot])
        )
        left_value = jnp.where((state == MATCH) | (state == INSERT), i - 1, -1)
        right_value = jnp.where((state == MATCH) | (state == DELETE), j - 1, -1)
        left = left.at[slot].set(jnp.where(write, left_value, left[slot]))
        right = right.at[slot].set(jnp.where(write, right_value, right[slot]))
        pointer = pointers[i, j, state]
        next_i = i - jnp.where((state == MATCH) | (state == INSERT), 1, 0)
        next_j = j - jnp.where((state == MATCH) | (state == DELETE), 1, 0)
        reached_start = (next_i == 0) & (next_j == 0)
        invalid_pointer = write & ~reached_start & (pointer < 0)
        next_done = done | (write & reached_start) | invalid_pointer
        next_state = jnp.where(pointer >= 0, pointer, state).astype(jnp.int32)
        return (
            jnp.where(write, next_i, i),
            jnp.where(write, next_j, j),
            jnp.where(write, next_state, state),
            length + write.astype(jnp.int32),
            next_done,
            complete & ~invalid_pointer,
            states,
            left,
            right,
        )

    carry = (
        left_length,
        right_length,
        terminal_state,
        jnp.asarray(0, dtype=jnp.int32),
        initial_done,
        jnp.asarray(True),
        reverse_states,
        reverse_left,
        reverse_right,
    )
    _, _, _, length, done, complete, reverse_states, reverse_left, reverse_right = (
        jax.lax.fori_loop(0, capacity, step, carry)
    )
    complete = complete & done
    positions = jnp.arange(capacity, dtype=jnp.int32)
    reverse_positions = jnp.clip(length - 1 - positions, 0, capacity - 1)
    active = positions < length
    states = jnp.where(active, reverse_states[reverse_positions], PATH_PAD)
    left = jnp.where(active, reverse_left[reverse_positions], -1)
    right = jnp.where(active, reverse_right[reverse_positions], -1)
    return score, states, left, right, length, complete


def pair_hmm_forward_backward_from_potentials(
    model: PairHMM,
    match_log_potentials: ArrayLike,
    insertion_log_potentials: ArrayLike,
    deletion_log_potentials: ArrayLike,
    plan: PairHMMExecutionPlan,
    /,
    *,
    left_mask: ArrayLike | None = None,
    right_mask: ArrayLike | None = None,
) -> PairHMMResult:
    """Infer on supplied per-cell log emissions; their gradients equal occupancies."""
    match = jnp.asarray(match_log_potentials, dtype=model.initial_logits.dtype)
    insertion = jnp.asarray(insertion_log_potentials, dtype=model.initial_logits.dtype)
    deletion = jnp.asarray(deletion_log_potentials, dtype=model.initial_logits.dtype)
    if match.ndim != 2 or insertion.ndim != 1 or deletion.ndim != 1:
        raise ValueError(
            "match, insertion, and deletion potentials must have ranks 2, 1, 1."
        )
    if match.shape != (insertion.shape[0], deletion.shape[0]):
        raise ValueError("match potential shape must match insertion/deletion lengths.")
    lmask = (
        jnp.ones(insertion.shape, dtype=bool)
        if left_mask is None
        else jnp.asarray(left_mask, dtype=bool)
    )
    rmask = (
        jnp.ones(deletion.shape, dtype=bool)
        if right_mask is None
        else jnp.asarray(right_mask, dtype=bool)
    )
    if lmask.shape != insertion.shape or rmask.shape != deletion.shape:
        raise ValueError("potential masks must match insertion and deletion shapes.")
    (
        match,
        insertion,
        deletion,
        lmask,
        rmask,
        left_length,
        right_length,
        input_valid,
    ) = _prepare_potentials(match, insertion, deletion, lmask, rmask, plan)
    (
        initial_log,
        transition_log,
        terminal_log,
        match_emission_log,
        insertion_emission_log,
        deletion_emission_log,
    ) = model.normalized_log_parameters()
    parameter_valid = (
        jnp.any(jnp.isfinite(initial_log))
        & jnp.all(jnp.any(jnp.isfinite(transition_log), axis=1))
        & jnp.any(jnp.isfinite(terminal_log))
        & jnp.any(jnp.isfinite(match_emission_log))
        & jnp.any(jnp.isfinite(insertion_emission_log))
        & jnp.any(jnp.isfinite(deletion_emission_log))
        & jnp.all(~jnp.isnan(initial_log))
        & jnp.all(~jnp.isnan(transition_log))
        & jnp.all(~jnp.isnan(terminal_log))
        & jnp.all(~jnp.isnan(match_emission_log))
        & jnp.all(~jnp.isnan(insertion_emission_log))
        & jnp.all(~jnp.isnan(deletion_emission_log))
    )
    input_valid = input_valid & parameter_valid
    emissions = _emission_grid(match, insertion, deletion, plan)
    allowed = _allowed_grid(left_length, right_length, plan)
    alpha, viterbi, pointers = _forward_viterbi(
        emissions, allowed, initial_log, transition_log, plan
    )
    beta = _backward(
        emissions,
        allowed,
        transition_log,
        terminal_log,
        left_length,
        right_length,
        plan,
    )
    empty = (left_length == 0) & (right_length == 0)
    forward_log_partition = jnp.where(
        empty,
        0.0,
        _stable_logsumexp(alpha[left_length, right_length] + terminal_log),
    )
    first_i = jnp.asarray((1, 1, 0), dtype=jnp.int32)
    first_j = jnp.asarray((1, 0, 1), dtype=jnp.int32)
    states = jnp.arange(STATE_COUNT, dtype=jnp.int32)
    safe_first_i = jnp.minimum(first_i, plan.maximum_left_length)
    safe_first_j = jnp.minimum(first_j, plan.maximum_right_length)
    first_allowed = (
        (first_i <= left_length)
        & (first_j <= right_length)
        & allowed[safe_first_i, safe_first_j, states]
    )
    first_log_joint = (
        initial_log
        + emissions[safe_first_i, safe_first_j, states]
        + beta[safe_first_i, safe_first_j, states]
    )
    backward_log_partition = jnp.where(
        empty,
        0.0,
        _stable_logsumexp(jnp.where(first_allowed, first_log_joint, -jnp.inf)),
    )
    log_partition = forward_log_partition
    safe_partition = jnp.where(jnp.isfinite(log_partition), log_partition, 0.0)
    log_state = alpha + beta - safe_partition
    state_marginals = jnp.where(
        allowed & jnp.isfinite(log_partition) & jnp.isfinite(log_state),
        jnp.exp(log_state),
        0.0,
    )
    transition_marginals = _transition_posteriors(
        alpha, beta, emissions, allowed, transition_log, log_partition, plan
    )
    initial_marginals = jnp.where(
        first_allowed & jnp.isfinite(log_partition) & jnp.isfinite(first_log_joint),
        jnp.exp(first_log_joint - safe_partition),
        0.0,
    )
    terminal_log_joint = alpha[left_length, right_length] + terminal_log
    terminal_marginals = jnp.where(
        ~empty & jnp.isfinite(log_partition) & jnp.isfinite(terminal_log_joint),
        jnp.exp(terminal_log_joint - safe_partition),
        0.0,
    )
    expected_state_counts = jnp.sum(state_marginals, axis=(0, 1))
    expected_transition_counts = jnp.sum(transition_marginals, axis=(0, 1))
    capacity_ok = plan.traceback_capacity >= left_length + right_length
    (
        viterbi_score,
        viterbi_states,
        viterbi_left,
        viterbi_right,
        viterbi_length,
        traceback_complete,
    ) = _viterbi_traceback(
        viterbi,
        pointers,
        terminal_log,
        left_length,
        right_length,
        capacity_ok,
        plan,
    )
    both_impossible = ~jnp.isfinite(forward_log_partition) & ~jnp.isfinite(
        backward_log_partition
    )
    forward_backward_error = jnp.where(
        both_impossible,
        0.0,
        jnp.abs(forward_log_partition - backward_log_partition),
    )
    scale = jnp.maximum(jnp.abs(forward_log_partition), 1.0)
    consistency_tolerance = 128.0 * jnp.finfo(alpha.dtype).eps * scale
    forward_backward_consistent = both_impossible | (
        forward_backward_error <= consistency_tolerance
    )

    initial_grid = jnp.zeros_like(state_marginals)
    initial_grid = initial_grid.at[safe_first_i, safe_first_j, states].set(
        initial_marginals
    )
    inbound = jnp.sum(transition_marginals, axis=2) + initial_grid
    posterior_conservation_error = jnp.max(jnp.abs(state_marginals - inbound))
    conservation_tolerance = 256.0 * jnp.finfo(alpha.dtype).eps
    posterior_conserved = (
        empty
        | ~jnp.isfinite(log_partition)
        | (posterior_conservation_error <= conservation_tolerance)
    )

    if plan.band_radius < 0:
        boundary_mass = jnp.asarray(0.0, dtype=alpha.dtype)
    else:
        ii = jnp.arange(plan.maximum_left_length + 1)[:, None, None]
        jj = jnp.arange(plan.maximum_right_length + 1)[None, :, None]
        boundary = jnp.abs(ii - jj) == plan.band_radius
        boundary_occupancy = jnp.sum(jnp.where(boundary, state_marginals, 0.0))
        total_occupancy = jnp.sum(state_marginals)
        boundary_mass = boundary_occupancy / jnp.maximum(total_occupancy, 1.0)
    truncated = jnp.asarray(plan.band_limited)
    path_exists = jnp.isfinite(log_partition)
    expansion_required = truncated & (
        ~path_exists | (boundary_mass > plan.boundary_mass_tolerance)
    )
    maximum_radius = max(plan.maximum_left_length, plan.maximum_right_length)
    suggested_radius = jnp.where(
        expansion_required,
        jnp.minimum(plan.band_radius + plan.expansion_step, maximum_radius),
        jnp.maximum(plan.band_radius, 0),
    ).astype(jnp.int32)
    valid = (
        input_valid
        & capacity_ok
        & path_exists
        & traceback_complete
        & forward_backward_consistent
        & posterior_conserved
    )
    status = jnp.where(
        ~input_valid,
        PAIR_HMM_INVALID_INPUT,
        jnp.where(
            ~capacity_ok,
            PAIR_HMM_CAPACITY_EXCEEDED,
            jnp.where(
                ~path_exists,
                PAIR_HMM_IMPOSSIBLE,
                jnp.where(
                    ~traceback_complete,
                    PAIR_HMM_TRACEBACK_INCOMPLETE,
                    jnp.where(
                        ~forward_backward_consistent | ~posterior_conserved,
                        PAIR_HMM_INCONSISTENT,
                        jnp.where(truncated, PAIR_HMM_CONDITIONAL_BAND, PAIR_HMM_OK),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = PairHMMEvidence(
        input_valid,
        capacity_ok,
        path_exists,
        traceback_complete,
        forward_backward_consistent,
        posterior_conserved,
        ~truncated,
        expansion_required,
    )
    return PairHMMResult(
        log_partition,
        forward_log_partition,
        backward_log_partition,
        alpha,
        beta,
        state_marginals,
        transition_marginals,
        initial_marginals,
        terminal_marginals,
        expected_state_counts,
        expected_transition_counts,
        viterbi_score,
        viterbi_states,
        viterbi_left,
        viterbi_right,
        viterbi_length,
        left_length,
        right_length,
        valid,
        status,
        evidence,
        ~truncated,
        truncated,
        boundary_mass,
        expansion_required,
        suggested_radius,
        forward_backward_error,
        posterior_conservation_error,
        _pair_hmm_contract(plan),
    )


def pair_hmm_forward_backward(
    model: PairHMM,
    left_symbol_probabilities: ArrayLike,
    right_symbol_probabilities: ArrayLike,
    plan: PairHMMExecutionPlan,
    /,
    *,
    left_mask: ArrayLike | None = None,
    right_mask: ArrayLike | None = None,
) -> PairHMMResult:
    """Infer from one-hot or ambiguity distributions over canonical symbols."""
    left = jnp.asarray(left_symbol_probabilities, dtype=model.initial_logits.dtype)
    right = jnp.asarray(right_symbol_probabilities, dtype=model.initial_logits.dtype)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != model.alphabet_size
        or right.shape[1] != model.alphabet_size
    ):
        raise ValueError(
            "symbol probabilities must be rank two with the model alphabet size."
        )
    lmask = (
        jnp.ones((left.shape[0],), dtype=bool)
        if left_mask is None
        else jnp.asarray(left_mask, dtype=bool)
    )
    rmask = (
        jnp.ones((right.shape[0],), dtype=bool)
        if right_mask is None
        else jnp.asarray(right_mask, dtype=bool)
    )
    if lmask.shape != (left.shape[0],) or rmask.shape != (right.shape[0],):
        raise ValueError("sequence masks must match their probability arrays.")
    left_sum = jnp.sum(left, axis=-1, keepdims=True)
    right_sum = jnp.sum(right, axis=-1, keepdims=True)
    left_valid = jnp.all(
        ~lmask[:, None] | (jnp.isfinite(left) & (left >= 0.0))
    ) & jnp.all(~lmask | (left_sum[:, 0] > 0.0))
    right_valid = jnp.all(
        ~rmask[:, None] | (jnp.isfinite(right) & (right >= 0.0))
    ) & jnp.all(~rmask | (right_sum[:, 0] > 0.0))
    left = jnp.where(lmask[:, None], left / jnp.where(left_sum > 0.0, left_sum, 1.0), 0.0)
    right = jnp.where(
        rmask[:, None], right / jnp.where(right_sum > 0.0, right_sum, 1.0), 0.0
    )
    _, _, _, match_log, insertion_log, deletion_log = model.normalized_log_parameters()
    match_probability = (left @ jnp.exp(match_log)) @ right.T
    insertion_probability = left @ jnp.exp(insertion_log)
    deletion_probability = right @ jnp.exp(deletion_log)
    tiny = jnp.finfo(left.dtype).tiny
    match_potential = jnp.where(
        match_probability > 0.0,
        jnp.log(jnp.maximum(match_probability, tiny)),
        -jnp.inf,
    )
    insertion_potential = jnp.where(
        insertion_probability > 0.0,
        jnp.log(jnp.maximum(insertion_probability, tiny)),
        -jnp.inf,
    )
    deletion_potential = jnp.where(
        deletion_probability > 0.0,
        jnp.log(jnp.maximum(deletion_probability, tiny)),
        -jnp.inf,
    )
    result = pair_hmm_forward_backward_from_potentials(
        model,
        match_potential,
        insertion_potential,
        deletion_potential,
        plan,
        left_mask=lmask,
        right_mask=rmask,
    )
    observation_valid = left_valid & right_valid
    valid = result.valid & observation_valid
    status = jnp.where(observation_valid, result.status, PAIR_HMM_INVALID_INPUT).astype(
        jnp.int32
    )
    evidence = eqx.tree_at(
        lambda value: value.input_valid,
        result.evidence,
        result.evidence.input_valid & observation_valid,
    )
    return eqx.tree_at(
        lambda value: (value.valid, value.status, value.evidence),
        result,
        (valid, status, evidence),
    )


__all__ = [
    "DELETE",
    "INSERT",
    "MATCH",
    "PAIR_HMM_CAPACITY_EXCEEDED",
    "PAIR_HMM_CONDITIONAL_BAND",
    "PAIR_HMM_IMPOSSIBLE",
    "PAIR_HMM_INCONSISTENT",
    "PAIR_HMM_INVALID_INPUT",
    "PAIR_HMM_OK",
    "PAIR_HMM_TRACEBACK_INCOMPLETE",
    "PATH_PAD",
    "PairHMM",
    "PairHMMEvidence",
    "PairHMMExecutionPlan",
    "PairHMMResult",
    "pair_hmm_forward_backward",
    "pair_hmm_forward_backward_from_potentials",
]
