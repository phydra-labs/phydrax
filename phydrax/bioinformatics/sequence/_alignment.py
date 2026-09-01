#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._scoring import SubstitutionScoreTable


MATCH = 0
INSERT = 1
DELETE = 2
TRACE_PAD = -1

ALIGNMENT_OK = 0
ALIGNMENT_CONDITIONAL_BAND = 1
ALIGNMENT_INVALID_INPUT = 2
ALIGNMENT_CAPACITY_EXCEEDED = 3
ALIGNMENT_IMPOSSIBLE = 4
ALIGNMENT_TRACEBACK_INCOMPLETE = 5


class AffineGapPenalties(StrictModule):
    """Affine penalties where a length-L gap costs open + (L - 1) extend."""

    open: Array
    extend: Array

    def __init__(self, open: ArrayLike, extend: ArrayLike, /):
        raw_open = jnp.asarray(open)
        dtype = (
            raw_open.dtype
            if jnp.issubdtype(raw_open.dtype, jnp.floating)
            else jnp.asarray(0.0).dtype
        )
        open_ = jnp.asarray(raw_open, dtype=dtype)
        extend_ = jnp.asarray(extend, dtype=dtype)
        if open_.ndim != 0 or extend_.ndim != 0:
            raise ValueError("Affine gap open and extend penalties must be scalars.")
        self.open = open_
        self.extend = extend_


class AlignmentExecutionPlan(StrictModule):
    """Static full-lattice or diagonal-band allocation and traceback capacity."""

    mode: str = eqx.field(static=True)
    maximum_query_length: int = eqx.field(static=True)
    maximum_target_length: int = eqx.field(static=True)
    traceback_capacity: int = eqx.field(static=True)
    band_radius: int = eqx.field(static=True)
    expansion_step: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_query_length: int,
        maximum_target_length: int,
        /,
        *,
        traceback_capacity: int,
        band_radius: int | None = None,
        expansion_step: int = 1,
    ):
        query = int(maximum_query_length)
        target = int(maximum_target_length)
        capacity = int(traceback_capacity)
        radius = -1 if band_radius is None else int(band_radius)
        step = int(expansion_step)
        if query < 1 or target < 1:
            raise ValueError("Alignment plan maximum lengths must be positive.")
        if capacity < 1:
            raise ValueError("traceback_capacity must be positive.")
        if radius < -1:
            raise ValueError("band_radius must be non-negative or None.")
        if step < 1:
            raise ValueError("expansion_step must be positive.")
        mode = "full" if radius < 0 else "diagonal_band"
        self.mode = mode
        self.maximum_query_length = query
        self.maximum_target_length = target
        self.traceback_capacity = capacity
        self.band_radius = radius
        self.expansion_step = step
        self.plan_id = canonical_fingerprint(
            {
                "kind": "affine-alignment-execution",
                "mode": mode,
                "maximum_query_length": query,
                "maximum_target_length": target,
                "traceback_capacity": capacity,
                "band_radius": radius,
                "expansion_step": step,
            }
        )

    @classmethod
    def full(
        cls,
        maximum_query_length: int,
        maximum_target_length: int,
        /,
        *,
        traceback_capacity: int,
    ) -> AlignmentExecutionPlan:
        return cls(
            maximum_query_length,
            maximum_target_length,
            traceback_capacity=traceback_capacity,
        )

    @classmethod
    def diagonal_band(
        cls,
        maximum_query_length: int,
        maximum_target_length: int,
        /,
        *,
        traceback_capacity: int,
        band_radius: int,
        expansion_step: int = 1,
    ) -> AlignmentExecutionPlan:
        return cls(
            maximum_query_length,
            maximum_target_length,
            traceback_capacity=traceback_capacity,
            band_radius=band_radius,
            expansion_step=expansion_step,
        )

    @property
    def band_limited(self) -> bool:
        return (
            0
            <= self.band_radius
            < max(self.maximum_query_length, self.maximum_target_length)
        )


class AlignmentEvidence(StrictModule):
    """Named capacity, path, score, and path-domain evidence."""

    input_valid: Array
    capacity_sufficient: Array
    path_exists: Array
    traceback_complete: Array
    score_consistent: Array
    full_domain: Array
    conditional_band: Array
    boundary_hit: Array


class AlignmentResult(StrictModule):
    """Fixed-capacity affine alignment with audited bounded execution."""

    score: Array
    traceback_score: Array
    dynamic_program: Array
    operations: Array
    query_indices: Array
    target_indices: Array
    alignment_length: Array
    query_length: Array
    target_length: Array
    terminal_query_index: Array
    terminal_target_index: Array
    valid: Array
    status: Array
    evidence: AlignmentEvidence
    exact: Array
    truncated: Array
    boundary_hit: Array
    expansion_required: Array
    suggested_band_radius: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _alignment_contract(
    plan: AlignmentExecutionPlan, mode: str, /
) -> BioinformaticsMethodContract:
    banded = plan.band_limited
    mode_semantics = {
        "global": "both sequences are consumed end-to-end",
        "local": "the best positive-scoring local path is returned; the empty path scores zero",
        "semiglobal": "unmatched leading and trailing overhangs are free",
    }[mode]
    return BioinformaticsMethodContract(
        f"affine-{mode}-alignment",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.ALMOST_EVERYWHERE,
        OutputKind.STRUCTURED,
        conditioning_statement=(
            f"paths conditioned to |query_index-target_index| <= {plan.band_radius}"
            if banded
            else "all paths in the declared finite sequence rectangle"
        ),
        truncation_statement=(
            "diagonal-band path domain; adaptive expansion is diagnosed"
            if banded
            else "none"
        ),
        capacity_semantics=(
            f"query<={plan.maximum_query_length}, target<={plan.maximum_target_length}, "
            f"traceback<={plan.traceback_capacity}; overflow is an invalid result"
        ),
        assumptions=(
            "gap length L costs open + (L-1)*extend",
            "deterministic ties prefer MATCH, INSERT, then DELETE",
            mode_semantics,
        ),
        nondifferentiable_outputs=(
            "operations",
            "query_indices",
            "target_indices",
            "alignment_length",
            "status",
        ),
    )


def _fixed_vector(
    values: Array, mask: Array, capacity: int, /
) -> tuple[Array, Array, bool]:
    size = values.shape[0]
    if size >= capacity:
        return values[:capacity], mask[:capacity], size > capacity
    padding = capacity - size
    return (
        jnp.pad(values, (0, padding)),
        jnp.pad(mask, (0, padding), constant_values=False),
        False,
    )


def _band_cell(i: ArrayLike, j: ArrayLike, plan: AlignmentExecutionPlan, /) -> Array:
    if plan.band_radius < 0:
        return jnp.asarray(True)
    return jnp.abs(jnp.asarray(i) - jnp.asarray(j)) <= plan.band_radius


def _affine_tables(
    pair_scores: Array,
    query_length: Array,
    target_length: Array,
    penalties: AffineGapPenalties,
    plan: AlignmentExecutionPlan,
    mode: str,
    /,
) -> tuple[Array, Array]:
    nq = plan.maximum_query_length
    nt = plan.maximum_target_length
    dtype = jnp.result_type(pair_scores, penalties.open, penalties.extend)
    scores = jnp.full((nq + 1, nt + 1, 3), -jnp.inf, dtype=dtype)
    pointers = jnp.full((nq + 1, nt + 1, 3), -2, dtype=jnp.int8)
    scores = scores.at[0, 0, MATCH].set(0.0)
    total = (nq + 1) * (nt + 1)

    def fill_cell(flat_index: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        values, ptr = carry
        i = flat_index // (nt + 1)
        j = flat_index % (nt + 1)
        active = (i <= query_length) & (j <= target_length) & _band_cell(i, j, plan)
        origin = (i == 0) & (j == 0)

        safe_im1 = jnp.maximum(i - 1, 0)
        safe_jm1 = jnp.maximum(j - 1, 0)
        substitution = pair_scores[safe_im1, safe_jm1]

        diagonal = values[safe_im1, safe_jm1]
        above = values[safe_im1, j]
        left = values[i, safe_jm1]

        if mode == "local":
            match_candidates = jnp.stack(
                (
                    jnp.asarray(0.0, dtype=dtype),
                    substitution,
                    diagonal[MATCH] + substitution,
                    diagonal[INSERT] + substitution,
                    diagonal[DELETE] + substitution,
                )
            )
            match_states = jnp.asarray((-2, -1, MATCH, INSERT, DELETE), dtype=jnp.int8)
            insert_candidates = jnp.stack(
                (
                    jnp.asarray(0.0, dtype=dtype),
                    above[MATCH] + penalties.open,
                    above[INSERT] + penalties.extend,
                )
            )
            insert_states = jnp.asarray((-2, MATCH, INSERT), dtype=jnp.int8)
            delete_candidates = jnp.stack(
                (
                    jnp.asarray(0.0, dtype=dtype),
                    left[MATCH] + penalties.open,
                    left[DELETE] + penalties.extend,
                )
            )
            delete_states = jnp.asarray((-2, MATCH, DELETE), dtype=jnp.int8)
        else:
            match_candidates = diagonal + substitution
            match_states = jnp.asarray((MATCH, INSERT, DELETE), dtype=jnp.int8)
            insert_candidates = jnp.stack(
                (above[MATCH] + penalties.open, above[INSERT] + penalties.extend)
            )
            insert_states = jnp.asarray((MATCH, INSERT), dtype=jnp.int8)
            delete_candidates = jnp.stack(
                (left[MATCH] + penalties.open, left[DELETE] + penalties.extend)
            )
            delete_states = jnp.asarray((MATCH, DELETE), dtype=jnp.int8)

        match_choice = jnp.argmax(match_candidates)
        insert_choice = jnp.argmax(insert_candidates)
        delete_choice = jnp.argmax(delete_candidates)
        cell_scores = jnp.stack(
            (
                match_candidates[match_choice],
                insert_candidates[insert_choice],
                delete_candidates[delete_choice],
            )
        )
        cell_ptr = jnp.stack(
            (
                match_states[match_choice],
                insert_states[insert_choice],
                delete_states[delete_choice],
            )
        )
        state_allowed = jnp.stack(((i > 0) & (j > 0), i > 0, j > 0))
        allowed = active & state_allowed & ~origin
        cell_scores = jnp.where(allowed, cell_scores, -jnp.inf)
        cell_ptr = jnp.where(allowed, cell_ptr, -2).astype(jnp.int8)

        if mode == "semiglobal":
            free_insert = active & (i > 0) & (j == 0)
            free_delete = active & (i == 0) & (j > 0)
            cell_scores = cell_scores.at[INSERT].set(
                jnp.where(free_insert, 0.0, cell_scores[INSERT])
            )
            cell_scores = cell_scores.at[DELETE].set(
                jnp.where(free_delete, 0.0, cell_scores[DELETE])
            )
            cell_ptr = cell_ptr.at[INSERT].set(
                jnp.where(free_insert, -1, cell_ptr[INSERT])
            )
            cell_ptr = cell_ptr.at[DELETE].set(
                jnp.where(free_delete, -1, cell_ptr[DELETE])
            )

        values = values.at[i, j].set(jnp.where(origin, values[i, j], cell_scores))
        ptr = ptr.at[i, j].set(jnp.where(origin, ptr[i, j], cell_ptr))
        return values, ptr

    return jax.lax.fori_loop(1, total, fill_cell, (scores, pointers))


def _terminal(
    scores: Array,
    query_length: Array,
    target_length: Array,
    plan: AlignmentExecutionPlan,
    mode: str,
    /,
) -> tuple[Array, Array, Array, Array]:
    nq = plan.maximum_query_length
    nt = plan.maximum_target_length
    if mode == "global":
        terminal_scores = scores[query_length, target_length]
        state = jnp.argmax(terminal_scores).astype(jnp.int32)
        return terminal_scores[state], query_length, target_length, state

    ii = jnp.arange(nq + 1, dtype=jnp.int32)[:, None, None]
    jj = jnp.arange(nt + 1, dtype=jnp.int32)[None, :, None]
    active = (ii <= query_length) & (jj <= target_length)
    in_band = (
        jnp.ones((nq + 1, nt + 1, 1), dtype=bool)
        if plan.band_radius < 0
        else jnp.abs(ii - jj) <= plan.band_radius
    )
    if mode == "local":
        eligible = active & in_band & (ii > 0) & (jj > 0)
    else:
        eligible = active & in_band & ((ii == query_length) | (jj == target_length))
    candidates = jnp.where(eligible, scores, -jnp.inf)
    flat = jnp.argmax(candidates.reshape(-1))
    state = (flat % 3).astype(jnp.int32)
    cell = flat // 3
    i = (cell // (nt + 1)).astype(jnp.int32)
    j = (cell % (nt + 1)).astype(jnp.int32)
    score = candidates[i, j, state]
    empty_semiglobal = (mode == "semiglobal") & (
        (query_length == 0) | (target_length == 0)
    )
    return (
        jnp.where(empty_semiglobal, 0.0, score),
        jnp.where(empty_semiglobal, 0, i),
        jnp.where(empty_semiglobal, 0, j),
        jnp.where(empty_semiglobal, MATCH, state),
    )


def _traceback(
    pointers: Array,
    terminal_i: Array,
    terminal_j: Array,
    terminal_state: Array,
    terminal_score: Array,
    capacity_ok: Array,
    plan: AlignmentExecutionPlan,
    mode: str,
    /,
) -> tuple[Array, Array, Array, Array, Array]:
    capacity = plan.traceback_capacity
    reverse_ops = jnp.full((capacity,), TRACE_PAD, dtype=jnp.int8)
    reverse_i = jnp.full((capacity,), -1, dtype=jnp.int32)
    reverse_j = jnp.full((capacity,), -1, dtype=jnp.int32)
    initial_done = ~capacity_ok | ~jnp.isfinite(terminal_score)
    initial_done = initial_done | ((mode == "local") & (terminal_score <= 0.0))
    initial_done = initial_done | ((terminal_i == 0) & (terminal_j == 0))

    def step(
        _: int,
        carry: tuple[
            Array, Array, Array, Array, Array, Array, Array, Array, Array
        ],
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        i, j, state, length, done, complete, ops, qi, tj = carry
        pointer = pointers[i, j, state]
        reset = (mode == "local") & (pointer == -2)
        boundary = (mode == "semiglobal") & ((i == 0) | (j == 0))
        write = ~done & ~reset & ~boundary
        slot = jnp.minimum(length, capacity - 1)
        ops = ops.at[slot].set(jnp.where(write, state.astype(jnp.int8), ops[slot]))
        qi_value = jnp.where((state == MATCH) | (state == INSERT), i - 1, -1)
        tj_value = jnp.where((state == MATCH) | (state == DELETE), j - 1, -1)
        qi = qi.at[slot].set(jnp.where(write, qi_value, qi[slot]))
        tj = tj.at[slot].set(jnp.where(write, tj_value, tj[slot]))
        next_i = i - jnp.where((state == MATCH) | (state == INSERT), 1, 0)
        next_j = j - jnp.where((state == MATCH) | (state == DELETE), 1, 0)
        next_length = length + write.astype(jnp.int32)
        reached_start = (next_i == 0) & (next_j == 0)
        local_start = (mode == "local") & (pointer < 0)
        semiglobal_start = (mode == "semiglobal") & ((next_i == 0) | (next_j == 0))
        invalid_pointer = write & (pointer < 0) & ~(local_start | semiglobal_start)
        next_done = (
            done
            | reset
            | boundary
            | (write & (reached_start | local_start | semiglobal_start))
        )
        next_complete = complete & ~invalid_pointer
        next_state = jnp.where(pointer >= 0, pointer, state).astype(jnp.int32)
        return (
            jnp.where(write, next_i, i),
            jnp.where(write, next_j, j),
            jnp.where(write, next_state, state),
            next_length,
            next_done,
            next_complete,
            ops,
            qi,
            tj,
        )

    carry = (
        terminal_i.astype(jnp.int32),
        terminal_j.astype(jnp.int32),
        terminal_state.astype(jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        initial_done,
        jnp.asarray(True),
        reverse_ops,
        reverse_i,
        reverse_j,
    )
    *_, length, done, complete, reverse_ops, reverse_i, reverse_j = jax.lax.fori_loop(
        0, capacity, step, carry
    )
    complete = complete & done
    positions = jnp.arange(capacity, dtype=jnp.int32)
    reverse_positions = jnp.clip(length - 1 - positions, 0, capacity - 1)
    active = positions < length
    operations = jnp.where(active, reverse_ops[reverse_positions], TRACE_PAD)
    query_indices = jnp.where(active, reverse_i[reverse_positions], -1)
    target_indices = jnp.where(active, reverse_j[reverse_positions], -1)
    return operations, query_indices, target_indices, length, complete


def _score_traceback(
    pair_scores: Array,
    operations: Array,
    query_indices: Array,
    target_indices: Array,
    length: Array,
    penalties: AffineGapPenalties,
    /,
) -> Array:
    capacity = operations.shape[0]
    dtype = jnp.result_type(pair_scores, penalties.open, penalties.extend)

    def add_step(k: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        score, previous = carry
        active = k < length
        operation = operations[k].astype(jnp.int32)
        qi = jnp.maximum(query_indices[k], 0)
        tj = jnp.maximum(target_indices[k], 0)
        substitution = pair_scores[qi, tj]
        gap = jnp.where(operation == previous, penalties.extend, penalties.open)
        contribution = jnp.where(operation == MATCH, substitution, gap)
        return score + jnp.where(active, contribution, 0.0), jnp.where(
            active, operation, previous
        )

    score, _ = jax.lax.fori_loop(
        0,
        capacity,
        add_step,
        (jnp.asarray(0.0, dtype=dtype), jnp.asarray(TRACE_PAD, dtype=jnp.int32)),
    )
    return score


def align_affine(
    query_codes: ArrayLike,
    target_codes: ArrayLike,
    scoring: SubstitutionScoreTable,
    penalties: AffineGapPenalties,
    plan: AlignmentExecutionPlan,
    /,
    *,
    mode: str = "global",
    query_mask: ArrayLike | None = None,
    target_mask: ArrayLike | None = None,
) -> AlignmentResult:
    """Align two padded encoded sequences with an affine three-state DP."""
    if mode not in ("global", "local", "semiglobal"):
        raise ValueError("mode must be 'global', 'local', or 'semiglobal'.")
    query = jnp.asarray(query_codes, dtype=jnp.int32)
    target = jnp.asarray(target_codes, dtype=jnp.int32)
    if query.ndim != 1 or target.ndim != 1:
        raise ValueError("query_codes and target_codes must be rank one.")
    qmask = (
        jnp.ones(query.shape, dtype=bool)
        if query_mask is None
        else jnp.asarray(query_mask, dtype=bool)
    )
    tmask = (
        jnp.ones(target.shape, dtype=bool)
        if target_mask is None
        else jnp.asarray(target_mask, dtype=bool)
    )
    if qmask.shape != query.shape or tmask.shape != target.shape:
        raise ValueError("sequence masks must match their code arrays.")

    query, qmask, query_shape_overflow = _fixed_vector(
        query, qmask, plan.maximum_query_length
    )
    target, tmask, target_shape_overflow = _fixed_vector(
        target, tmask, plan.maximum_target_length
    )
    query_length = jnp.sum(qmask, dtype=jnp.int32)
    target_length = jnp.sum(tmask, dtype=jnp.int32)
    query_prefix = jnp.array_equal(
        qmask, jnp.arange(plan.maximum_query_length) < query_length
    )
    target_prefix = jnp.array_equal(
        tmask, jnp.arange(plan.maximum_target_length) < target_length
    )
    query_codes_valid = jnp.all(~qmask | ((query >= 0) & (query < scoring.symbol_count)))
    target_codes_valid = jnp.all(
        ~tmask | ((target >= 0) & (target < scoring.symbol_count))
    )
    input_valid = (
        ~jnp.asarray(query_shape_overflow | target_shape_overflow)
        & query_prefix
        & target_prefix
        & query_codes_valid
        & target_codes_valid
    )
    pair_scores = scoring.pairwise_scores(query, target)
    pair_scores = jnp.where(qmask[:, None] & tmask[None, :], pair_scores, -jnp.inf)
    scores, pointers = _affine_tables(
        pair_scores, query_length, target_length, penalties, plan, mode
    )
    terminal_score, terminal_i, terminal_j, terminal_state = _terminal(
        scores, query_length, target_length, plan, mode
    )
    capacity_ok = plan.traceback_capacity >= query_length + target_length
    operations, query_indices, target_indices, length, traceback_complete = _traceback(
        pointers,
        terminal_i,
        terminal_j,
        terminal_state,
        terminal_score,
        capacity_ok,
        plan,
        mode,
    )
    traceback_score = _score_traceback(
        pair_scores,
        operations,
        query_indices,
        target_indices,
        length,
        penalties,
    )
    path_exists = jnp.isfinite(terminal_score)
    scale = jnp.maximum(jnp.abs(terminal_score), 1.0)
    tolerance = 64.0 * jnp.finfo(scores.dtype).eps * scale
    score_consistent = (~path_exists) | (
        jnp.abs(traceback_score - terminal_score) <= tolerance
    )
    if plan.band_radius < 0:
        boundary_hit = jnp.asarray(False)
    else:
        active_trace = jnp.arange(plan.traceback_capacity) < length
        query_coordinate = jnp.cumsum(
            active_trace & ((operations == MATCH) | (operations == INSERT))
        )
        target_coordinate = jnp.cumsum(
            active_trace & ((operations == MATCH) | (operations == DELETE))
        )
        boundary_hit = jnp.any(
            active_trace
            & (jnp.abs(query_coordinate - target_coordinate) == plan.band_radius)
        )
    truncated = jnp.asarray(plan.band_limited)
    expansion_required = truncated & (boundary_hit | ~path_exists)
    maximum_radius = max(plan.maximum_query_length, plan.maximum_target_length)
    suggested_radius = jnp.where(
        expansion_required,
        jnp.minimum(plan.band_radius + plan.expansion_step, maximum_radius),
        jnp.maximum(plan.band_radius, 0),
    ).astype(jnp.int32)
    valid = (
        input_valid & capacity_ok & path_exists & traceback_complete & score_consistent
    )
    status = jnp.where(
        ~input_valid,
        ALIGNMENT_INVALID_INPUT,
        jnp.where(
            ~capacity_ok,
            ALIGNMENT_CAPACITY_EXCEEDED,
            jnp.where(
                ~path_exists,
                ALIGNMENT_IMPOSSIBLE,
                jnp.where(
                    ~traceback_complete | ~score_consistent,
                    ALIGNMENT_TRACEBACK_INCOMPLETE,
                    jnp.where(truncated, ALIGNMENT_CONDITIONAL_BAND, ALIGNMENT_OK),
                ),
            ),
        ),
    ).astype(jnp.int32)
    evidence = AlignmentEvidence(
        input_valid,
        capacity_ok,
        path_exists,
        traceback_complete,
        score_consistent,
        ~truncated,
        truncated,
        boundary_hit,
    )
    return AlignmentResult(
        terminal_score,
        traceback_score,
        scores,
        operations,
        query_indices,
        target_indices,
        length,
        query_length,
        target_length,
        terminal_i,
        terminal_j,
        valid,
        status,
        evidence,
        ~truncated,
        truncated,
        boundary_hit,
        expansion_required,
        suggested_radius,
        _alignment_contract(plan, mode),
    )


__all__ = [
    "ALIGNMENT_CAPACITY_EXCEEDED",
    "ALIGNMENT_CONDITIONAL_BAND",
    "ALIGNMENT_IMPOSSIBLE",
    "ALIGNMENT_INVALID_INPUT",
    "ALIGNMENT_OK",
    "ALIGNMENT_TRACEBACK_INCOMPLETE",
    "AlignmentEvidence",
    "DELETE",
    "INSERT",
    "MATCH",
    "TRACE_PAD",
    "AffineGapPenalties",
    "AlignmentExecutionPlan",
    "AlignmentResult",
    "align_affine",
]
