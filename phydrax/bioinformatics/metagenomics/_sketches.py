#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import AlphabetPlan, SequenceBatch


class SketchStatus(IntEnum):
    SUCCESS = 0
    EMPTY_READ = 1
    CAPACITY_EXCEEDED = 2
    HASH_COLLISION = 3
    PLAN_MISMATCH = 4
    INVALID_QUERY = 5


def _count_contract(*, capacity_sufficient: bool) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "exact fixed-capacity canonical k-mer counting",
        MethodKind.EXACT_MODEL,
        ExecutionKind.EXACT_DISCRETE,
        DifferentiationKind.NONE,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Only windows composed entirely of canonical alphabet symbols are counted."
        ),
        truncation_statement=(
            "The count vector is exact over the full canonical k-mer space."
            if capacity_sufficient
            else "The declared count capacity is insufficient; no partial counts are returned."
        ),
        capacity_semantics=(
            "count_capacity must be at least alphabet_size**k; insufficient capacity is an "
            "observable failure."
        ),
        assumptions=("Sequence valid masks are left-prefix masks.",),
        nondifferentiable_outputs=("kmer_codes", "counts", "status"),
        input_dtype="int32/bool",
        compute_dtype="int32/bool",
        output_dtype="int32/bool",
    )


def _minhash_contract(hash_bits: int) -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "fixed-capacity MinHash sketch",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.STOCHASTIC_ESTIMATE,
        DifferentiationKind.NONE,
        OutputKind.ARRAY,
        conditioning_statement=(
            "Deterministic seeded hash permutations are applied to exact canonical k-mer codes."
        ),
        truncation_statement=(
            f"Hashes are retained at {hash_bits} bits and collisions are explicitly counted."
        ),
        capacity_semantics="Sketch width is fixed by the declared number of hash permutations.",
        assumptions=("MinHash equality is used as a finite-sample Jaccard estimator.",),
        nondifferentiable_outputs=("hash_values", "collision_counts", "status"),
        input_dtype="int32",
        compute_dtype="uint32",
        output_dtype="uint32/int32",
    )


def _comparison_contract() -> BioinformaticsMethodContract:
    return BioinformaticsMethodContract(
        "paired MinHash Jaccard estimation",
        MethodKind.APPROXIMATE_MODEL,
        ExecutionKind.STOCHASTIC_ESTIMATE,
        DifferentiationKind.NONE,
        OutputKind.SCALAR,
        conditioning_statement="Both sketches use the identical hash-family identity.",
        truncation_statement="All valid paired sketch slots contribute to the estimate.",
        capacity_semantics="Comparison capacity is the broadcast query-index shape.",
        assumptions=("Hash-permutation equality indicators are exchangeable estimates.",),
        nondifferentiable_outputs=("matching_hashes", "status"),
        input_dtype="uint32",
        compute_dtype="int32/float32",
        output_dtype="float32/int32",
    )


class KmerCountingPlan(StrictModule, NonTrainableState):
    """Static full-space encoding plan for exact fixed-capacity k-mer counts."""

    alphabet: AlphabetPlan = eqx.field(static=True)
    canonical_code_lut: Array
    complement_rank_lut: Array
    k: int = eqx.field(static=True)
    canonical: bool = eqx.field(static=True)
    count_capacity: int = eqx.field(static=True)
    required_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        alphabet: AlphabetPlan,
        k: int,
        /,
        *,
        count_capacity: int | None = None,
        canonical: bool = True,
    ):
        k_ = int(k)
        if k_ < 1:
            raise ValueError("k must be positive.")
        canonical_symbols = tuple(alphabet.canonical_symbols)
        alphabet_size = len(canonical_symbols)
        if alphabet_size < 1:
            raise ValueError("Alphabet must declare canonical symbols.")
        required = alphabet_size**k_
        capacity = required if count_capacity is None else int(count_capacity)
        if capacity < 1:
            raise ValueError("count_capacity must be positive.")
        canonical_rank = {symbol: index for index, symbol in enumerate(canonical_symbols)}
        code_lut = [-1] * len(alphabet.symbols)
        complement_lut = [-1] * len(alphabet.symbols)
        complements = alphabet.complement_map
        for symbol, rank in canonical_rank.items():
            code = alphabet.code(symbol)
            code_lut[code] = rank
            complement = complements[symbol]
            if complement not in canonical_rank:
                raise ValueError("Canonical reverse complements must remain canonical.")
            complement_lut[code] = canonical_rank[complement]
        self.alphabet = alphabet
        self.canonical_code_lut = jnp.asarray(code_lut, dtype=jnp.int32)
        self.complement_rank_lut = jnp.asarray(complement_lut, dtype=jnp.int32)
        self.k = k_
        self.canonical = bool(canonical)
        self.count_capacity = capacity
        self.required_capacity = required
        self.plan_id = canonical_fingerprint(
            {
                "kind": "exact-kmer-counting-plan",
                "alphabet": alphabet.fingerprint,
                "k": k_,
                "canonical_reverse_complement": bool(canonical),
                "count_capacity": capacity,
            }
        )


class KmerCountResult(StrictModule):
    record_ids: Array
    kmer_codes: Array
    window_valid: Array
    counts: Array
    distinct_counts: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    plan_id: str = eqx.field(static=True)


class MinHashPlan(StrictModule, NonTrainableState):
    sketch_size: int = eqx.field(static=True)
    hash_bits: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, sketch_size: int, /, *, hash_bits: int = 32, seed: int = 0):
        size = int(sketch_size)
        bits = int(hash_bits)
        seed_ = int(seed)
        if size < 1:
            raise ValueError("sketch_size must be positive.")
        if bits < 1 or bits > 32:
            raise ValueError("hash_bits must lie in [1, 32].")
        if seed_ < 0 or seed_ > 0xFFFFFFFF:
            raise ValueError("seed must fit in uint32.")
        self.sketch_size = size
        self.hash_bits = bits
        self.seed = seed_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "minhash-plan",
                "sketch_size": size,
                "hash_bits": bits,
                "seed": seed_,
            }
        )


class MinHashSketch(StrictModule, NonTrainableState):
    record_ids: Array
    hash_values: Array
    hash_valid: Array
    case_mask: Array
    unique_hash_counts: Array
    collision_counts: Array
    method_contract: BioinformaticsMethodContract
    plan_id: str = eqx.field(static=True)
    source_plan_id: str = eqx.field(static=True)


class MinHashSketchResult(StrictModule):
    sketch: MinHashSketch
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract


class MinHashComparisonResult(StrictModule):
    similarity: Array
    matching_hashes: Array
    compared_hashes: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract
    plan_id: str = eqx.field(static=True)


def count_kmers(reads: SequenceBatch, plan: KmerCountingPlan, /) -> KmerCountResult:
    """Count canonical k-mers exactly without hashing or bucket collisions."""
    if reads.alphabet.fingerprint != plan.alphabet.fingerprint:
        raise ValueError("Sequence and k-mer plan alphabets do not match.")
    batch_size, sequence_width = reads.token_codes.shape
    window_capacity = max(sequence_width - plan.k + 1, 0)
    positions = jnp.arange(window_capacity, dtype=jnp.int32)[None, :]
    forward = jnp.zeros((batch_size, window_capacity), dtype=jnp.int32)
    reverse = jnp.zeros_like(forward)
    valid = jnp.ones((batch_size, window_capacity), dtype=bool)
    alphabet_size = len(plan.alphabet.canonical_symbols)

    for offset in range(plan.k):
        token = reads.token_codes[:, offset : offset + window_capacity]
        canonical_rank = plan.canonical_code_lut[token]
        complement_rank = plan.complement_rank_lut[token]
        position_valid = reads.valid_mask[:, offset : offset + window_capacity]
        valid = valid & position_valid & (canonical_rank >= 0)
        forward = forward * alphabet_size + jnp.maximum(canonical_rank, 0)

        reverse_token = reads.token_codes[
            :, plan.k - 1 - offset : plan.k - 1 - offset + window_capacity
        ]
        reverse_rank = plan.complement_rank_lut[reverse_token]
        reverse = reverse * alphabet_size + jnp.maximum(reverse_rank, 0)

    codes = jnp.minimum(forward, reverse) if plan.canonical else forward
    codes = jnp.where(valid, codes, 0)
    capacity_sufficient_static = plan.count_capacity >= plan.required_capacity
    capacity_sufficient = jnp.asarray(capacity_sufficient_static, dtype=bool)
    full_counts = jnp.sum(
        jax.nn.one_hot(codes, plan.count_capacity, dtype=jnp.int32)
        * valid[:, :, None].astype(jnp.int32),
        axis=1,
        dtype=jnp.int32,
    )
    counts = jnp.where(capacity_sufficient, full_counts, jnp.zeros_like(full_counts))
    distinct = jnp.sum(counts > 0, axis=1, dtype=jnp.int32)
    window_count = jnp.sum(valid, axis=1, dtype=jnp.int32)
    record_valid = reads.case_mask & (window_count > 0) & capacity_sufficient
    status = jnp.where(
        reads.case_mask & (~capacity_sufficient),
        int(SketchStatus.CAPACITY_EXCEEDED),
        jnp.where(
            record_valid,
            int(SketchStatus.SUCCESS),
            int(SketchStatus.EMPTY_READ),
        ),
    ).astype(jnp.int32)
    contract = _count_contract(capacity_sufficient=capacity_sufficient_static)
    evidence = jnp.stack(
        (
            window_count,
            distinct,
            jnp.full((batch_size,), plan.required_capacity, dtype=jnp.int32),
            jnp.zeros((batch_size,), dtype=jnp.int32),
        ),
        axis=1,
    )
    return KmerCountResult(
        reads.record_ids,
        codes,
        valid,
        counts,
        distinct,
        record_valid,
        status,
        evidence,
        contract,
        plan.plan_id,
    )


def _hash_codes(codes: Array, seed: Array, bits: int, /) -> Array:
    value = codes.astype(jnp.uint32) ^ seed.astype(jnp.uint32)
    value = value ^ (value >> jnp.uint32(16))
    value = value * jnp.uint32(0x7FEB352D)
    value = value ^ (value >> jnp.uint32(15))
    value = value * jnp.uint32(0x846CA68B)
    value = value ^ (value >> jnp.uint32(16))
    if bits < 32:
        value = value & jnp.uint32((1 << bits) - 1)
    return value


def minhash_sketch(
    counts: KmerCountResult,
    plan: MinHashPlan,
    /,
) -> MinHashSketchResult:
    """Create seeded MinHash signatures with observable finite-bit collisions."""
    batch_size, window_capacity = counts.kmer_codes.shape
    seeds = (
        jnp.arange(plan.sketch_size, dtype=jnp.uint32) * jnp.uint32(0x9E3779B9)
        + jnp.uint32(plan.seed)
        + jnp.uint32(0x85EBCA6B)
    )
    sentinel = jnp.uint32(
        0xFFFFFFFF if plan.hash_bits == 32 else (1 << plan.hash_bits) - 1
    )
    has_windows = jnp.any(counts.window_valid, axis=1)
    hash_valid = has_windows[:, None] & counts.valid[:, None]
    if window_capacity > 0:
        hashed = _hash_codes(
            counts.kmer_codes[:, None, :], seeds[None, :, None], plan.hash_bits
        )
        material = jnp.where(counts.window_valid[:, None, :], hashed, sentinel)
        values = jnp.min(material, axis=2)
        sorted_hashes = jnp.sort(material, axis=2)
        sorted_valid = jnp.sort(counts.window_valid[:, None, :], axis=2)[:, :, ::-1]
        first = sorted_valid[:, :, :1]
        changes = sorted_valid[:, :, 1:] & (
            sorted_hashes[:, :, 1:] != sorted_hashes[:, :, :-1]
        )
        unique_hashes = jnp.sum(first, axis=2, dtype=jnp.int32) + jnp.sum(
            changes, axis=2, dtype=jnp.int32
        )
    else:
        values = jnp.full((batch_size, plan.sketch_size), sentinel, dtype=jnp.uint32)
        unique_hashes = jnp.zeros((batch_size, plan.sketch_size), dtype=jnp.int32)
    values = jnp.where(hash_valid, values, sentinel)
    collisions = jnp.maximum(counts.distinct_counts[:, None] - unique_hashes, 0)
    collision_record = jnp.any(collisions > 0, axis=1) & counts.valid
    status = jnp.where(
        counts.valid,
        jnp.where(
            collision_record,
            int(SketchStatus.HASH_COLLISION),
            int(SketchStatus.SUCCESS),
        ),
        counts.status,
    ).astype(jnp.int32)
    contract = _minhash_contract(plan.hash_bits)
    sketch = MinHashSketch(
        counts.record_ids,
        values,
        hash_valid,
        counts.valid,
        unique_hashes,
        collisions,
        contract,
        plan.plan_id,
        counts.plan_id,
    )
    evidence = jnp.stack(
        (
            jnp.sum(counts.window_valid, axis=1, dtype=jnp.int32),
            counts.distinct_counts,
            jnp.max(collisions, axis=1),
            jnp.full((batch_size,), plan.hash_bits, dtype=jnp.int32),
        ),
        axis=1,
    )
    return MinHashSketchResult(sketch, counts.valid, status, evidence, contract)


def compare_minhash(
    left: MinHashSketch,
    right: MinHashSketch,
    left_indices: ArrayLike,
    right_indices: ArrayLike,
    /,
) -> MinHashComparisonResult:
    """Estimate paired Jaccard similarities, rejecting hash-family mismatches."""
    left_index = jnp.asarray(left_indices, dtype=jnp.int32)
    right_index = jnp.asarray(right_indices, dtype=jnp.int32)
    left_index, right_index = jnp.broadcast_arrays(left_index, right_index)
    left_bounds = (left_index >= 0) & (left_index < left.record_ids.shape[0])
    right_bounds = (right_index >= 0) & (right_index < right.record_ids.shape[0])
    safe_left = jnp.clip(left_index, 0, max(left.record_ids.shape[0] - 1, 0))
    safe_right = jnp.clip(right_index, 0, max(right.record_ids.shape[0] - 1, 0))
    same_plan_static = left.plan_id == right.plan_id
    same_plan = jnp.asarray(same_plan_static, dtype=bool)
    comparable = (
        left_bounds
        & right_bounds
        & left.case_mask[safe_left]
        & right.case_mask[safe_right]
        & same_plan
    )
    valid_slots = left.hash_valid[safe_left] & right.hash_valid[safe_right]
    matches = valid_slots & (left.hash_values[safe_left] == right.hash_values[safe_right])
    compared = jnp.sum(valid_slots, axis=-1, dtype=jnp.int32)
    matching = jnp.sum(matches, axis=-1, dtype=jnp.int32)
    valid = comparable & (compared > 0)
    similarity = jnp.where(valid, matching / jnp.maximum(compared, 1), jnp.nan)
    status = jnp.where(
        ~same_plan,
        int(SketchStatus.PLAN_MISMATCH),
        jnp.where(
            valid,
            int(SketchStatus.SUCCESS),
            int(SketchStatus.INVALID_QUERY),
        ),
    ).astype(jnp.int32)
    contract = _comparison_contract()
    evidence = jnp.stack((matching, compared), axis=-1)
    return MinHashComparisonResult(
        similarity,
        matching,
        compared,
        valid,
        status,
        evidence,
        contract,
        left.plan_id if same_plan_static else "mismatch",
    )


__all__ = [
    "KmerCountResult",
    "KmerCountingPlan",
    "MinHashComparisonResult",
    "MinHashPlan",
    "MinHashSketch",
    "MinHashSketchResult",
    "SketchStatus",
    "compare_minhash",
    "count_kmers",
    "minhash_sketch",
]
