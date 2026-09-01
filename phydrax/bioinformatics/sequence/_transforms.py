#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Sequence
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax.bioinformatics.sequence._alphabet import AlphabetPlan, DNA_IUPAC
from phydrax.bioinformatics.sequence._batch import _concrete, SequenceBatch


InvalidSymbolPolicy = Literal["reject", "unknown"]
OverflowPolicy = Literal["reject", "truncate"]


class SequenceLoweringPlan(StrictModule):
    """Static capacities and explicit loss policies for host sequence lowering."""

    alphabet: AlphabetPlan = eqx.field(static=True)
    record_capacity: int = eqx.field(static=True)
    sequence_capacity: int = eqx.field(static=True)
    invalid_symbol_policy: InvalidSymbolPolicy = eqx.field(static=True)
    overflow_policy: OverflowPolicy = eqx.field(static=True)
    preserve_soft_mask: bool = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        alphabet: AlphabetPlan,
        record_capacity: int,
        sequence_capacity: int,
        *,
        invalid_symbol_policy: InvalidSymbolPolicy = "reject",
        overflow_policy: OverflowPolicy = "reject",
        preserve_soft_mask: bool = True,
    ):
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        if (
            isinstance(record_capacity, bool)
            or not isinstance(record_capacity, Integral)
            or isinstance(sequence_capacity, bool)
            or not isinstance(sequence_capacity, Integral)
        ):
            raise TypeError("Sequence lowering capacities must be integers.")
        records = int(record_capacity)
        positions = int(sequence_capacity)
        if records < 0 or positions < 0:
            raise ValueError("Sequence lowering capacities must be non-negative.")
        if invalid_symbol_policy not in ("reject", "unknown"):
            raise ValueError("invalid_symbol_policy must be 'reject' or 'unknown'.")
        if overflow_policy not in ("reject", "truncate"):
            raise ValueError("overflow_policy must be 'reject' or 'truncate'.")
        if not isinstance(preserve_soft_mask, bool):
            raise TypeError("preserve_soft_mask must be boolean.")
        preserve = preserve_soft_mask
        payload = {
            "alphabet": alphabet.fingerprint,
            "record_capacity": records,
            "sequence_capacity": positions,
            "invalid_symbol_policy": invalid_symbol_policy,
            "overflow_policy": overflow_policy,
            "preserve_soft_mask": preserve,
        }
        self.alphabet = alphabet
        self.record_capacity = records
        self.sequence_capacity = positions
        self.invalid_symbol_policy = invalid_symbol_policy
        self.overflow_policy = overflow_policy
        self.preserve_soft_mask = preserve
        self.fingerprint = canonical_fingerprint(payload)


class SequenceLoweringReport(StrictModule):
    """Array-only audit of every record, symbol mapping, and capacity loss."""

    input_record_count: Array
    retained_record_count: Array
    record_overflow_count: Array
    original_lengths: Array
    retained_lengths: Array
    truncated_symbol_counts: Array
    mapped_invalid_counts: Array
    retained_mask: Array
    plan_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        input_record_count: ArrayLike,
        retained_record_count: ArrayLike,
        record_overflow_count: ArrayLike,
        original_lengths: ArrayLike,
        retained_lengths: ArrayLike,
        truncated_symbol_counts: ArrayLike,
        mapped_invalid_counts: ArrayLike,
        retained_mask: ArrayLike,
        *,
        plan_fingerprint: str,
    ):
        inputs = jnp.asarray(input_record_count, dtype=jnp.int32)
        retained_count = jnp.asarray(retained_record_count, dtype=jnp.int32)
        overflow_count = jnp.asarray(record_overflow_count, dtype=jnp.int32)
        originals = jnp.asarray(original_lengths, dtype=jnp.int32)
        retained = jnp.asarray(retained_lengths, dtype=jnp.int32)
        truncated = jnp.asarray(truncated_symbol_counts, dtype=jnp.int32)
        mapped = jnp.asarray(mapped_invalid_counts, dtype=jnp.int32)
        keep = jnp.asarray(retained_mask)
        if inputs.shape or retained_count.shape or overflow_count.shape:
            raise ValueError("Lowering report counts must be scalar arrays.")
        if originals.ndim != 1 or any(
            values.shape != originals.shape
            for values in (retained, truncated, mapped, keep)
        ):
            raise ValueError("Per-record lowering report arrays must share one shape.")
        if keep.dtype != jnp.bool_:
            raise ValueError("retained_mask must be boolean.")
        if not plan_fingerprint:
            raise ValueError("plan_fingerprint must be non-empty.")
        concrete_inputs = _concrete(inputs)
        concrete_retained_count = _concrete(retained_count)
        concrete_overflow_count = _concrete(overflow_count)
        concrete_originals = _concrete(originals)
        concrete_retained = _concrete(retained)
        concrete_truncated = _concrete(truncated)
        concrete_mapped = _concrete(mapped)
        concrete_keep = _concrete(keep)
        if (
            concrete_inputs is not None
            and concrete_retained_count is not None
            and concrete_overflow_count is not None
            and concrete_originals is not None
            and concrete_retained is not None
            and concrete_truncated is not None
            and concrete_mapped is not None
            and concrete_keep is not None
        ):
            input_value = int(concrete_inputs)
            retained_value = int(concrete_retained_count)
            overflow_value = int(concrete_overflow_count)
            if (
                input_value != originals.shape[0]
                or retained_value != int(np.sum(concrete_keep))
                or overflow_value != input_value - retained_value
            ):
                raise ValueError("Lowering report record counts are inconsistent.")
            if (
                np.any(concrete_originals < 0)
                or np.any(concrete_retained < 0)
                or np.any(concrete_mapped < 0)
                or np.any(concrete_retained > concrete_originals)
                or np.any(concrete_truncated != concrete_originals - concrete_retained)
                or np.any(concrete_mapped > concrete_retained)
                or np.any(concrete_retained[~concrete_keep] != 0)
            ):
                raise ValueError("Lowering report per-record losses are inconsistent.")
            if np.any(concrete_keep[1:] & ~concrete_keep[:-1]):
                raise ValueError("retained_mask must be a left-aligned prefix.")
        self.input_record_count = inputs
        self.retained_record_count = retained_count
        self.record_overflow_count = overflow_count
        self.original_lengths = originals
        self.retained_lengths = retained
        self.truncated_symbol_counts = truncated
        self.mapped_invalid_counts = mapped
        self.retained_mask = keep
        self.plan_fingerprint = str(plan_fingerprint)

    @property
    def loss_occurred(self) -> Array:
        return (self.record_overflow_count > 0) | jnp.any(
            (self.truncated_symbol_counts > 0) | (self.mapped_invalid_counts > 0)
        )


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper() if symbol.isalpha() else symbol


def encode_sequence(
    sequence: str,
    alphabet: AlphabetPlan = DNA_IUPAC,
    *,
    invalid_symbol_policy: InvalidSymbolPolicy = "reject",
) -> tuple[np.ndarray, np.ndarray]:
    """Exactly encode one host string into codes and its positional soft mask."""
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a string.")
    if invalid_symbol_policy not in ("reject", "unknown"):
        raise ValueError("invalid_symbol_policy must be 'reject' or 'unknown'.")
    mapping = alphabet.symbol_to_code
    unknown_code = mapping[alphabet.unknown_symbol]
    codes = np.empty((len(sequence),), dtype=np.int32)
    soft = np.zeros((len(sequence),), dtype=bool)
    for position, raw_symbol in enumerate(sequence):
        symbol = _normalize_symbol(raw_symbol)
        if symbol not in mapping:
            if invalid_symbol_policy == "reject":
                raise ValueError(
                    f"Invalid symbol {raw_symbol!r} at position {position} for "
                    f"alphabet {alphabet.alphabet_id!r}."
                )
            code = unknown_code
        else:
            code = mapping[symbol]
        codes[position] = code
        soft[position] = raw_symbol.isalpha() and raw_symbol.islower()
    return codes, soft


def decode_sequence(
    token_codes: ArrayLike,
    alphabet: AlphabetPlan = DNA_IUPAC,
    *,
    valid_mask: ArrayLike | None = None,
    soft_mask: ArrayLike | None = None,
) -> str:
    """Exactly decode one host code vector, optionally restoring lowercase."""
    codes = np.asarray(token_codes)
    if codes.ndim != 1 or not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("token_codes must be a one-dimensional integer array.")
    valid = (
        np.ones(codes.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask)
    )
    soft = (
        np.zeros(codes.shape, dtype=bool) if soft_mask is None else np.asarray(soft_mask)
    )
    if valid.dtype != np.bool_ or valid.shape != codes.shape:
        raise ValueError("valid_mask must be boolean with token_codes shape.")
    if soft.dtype != np.bool_ or soft.shape != codes.shape:
        raise ValueError("soft_mask must be boolean with token_codes shape.")
    if np.any(codes < 0) or np.any(codes >= alphabet.size):
        raise ValueError("token_codes contains a code outside the alphabet.")
    if np.any(soft & ~valid):
        raise ValueError("soft_mask cannot mark an invalid position.")
    symbols = []
    for code, is_valid, is_soft in zip(codes, valid, soft, strict=True):
        if is_valid:
            symbol = alphabet.symbols[int(code)]
            symbols.append(symbol.lower() if is_soft else symbol)
    return "".join(symbols)


def encode_sequences(
    sequences: Sequence[str] | Iterable[str],
    alphabet: AlphabetPlan = DNA_IUPAC,
    *,
    record_ids: ArrayLike | None = None,
    invalid_symbol_policy: InvalidSymbolPolicy = "reject",
) -> SequenceBatch:
    """Exactly encode host strings without truncation or fabricated records."""
    strings = tuple(sequences)
    capacity = max((len(sequence) for sequence in strings), default=0)
    plan = SequenceLoweringPlan(
        alphabet,
        len(strings),
        capacity,
        invalid_symbol_policy=invalid_symbol_policy,
        overflow_policy="reject",
    )
    batch, _ = lower_sequences(strings, plan, record_ids=record_ids)
    return batch


def decode_sequences(
    batch: SequenceBatch,
    *,
    preserve_soft_mask: bool = True,
) -> tuple[str, ...]:
    """Decode populated records from a numeric batch to independent host strings."""
    if not isinstance(batch, SequenceBatch):
        raise TypeError("batch must be a SequenceBatch.")
    cases = np.asarray(batch.case_mask)
    codes = np.asarray(batch.token_codes)
    valid = np.asarray(batch.valid_mask)
    soft = np.asarray(batch.soft_mask) if preserve_soft_mask else np.zeros_like(valid)
    return tuple(
        decode_sequence(
            codes[index],
            batch.alphabet,
            valid_mask=valid[index],
            soft_mask=soft[index],
        )
        for index in range(batch.record_capacity)
        if cases[index]
    )


def lower_sequences(
    sequences: Sequence[str] | Iterable[str],
    plan: SequenceLoweringPlan,
    *,
    record_ids: ArrayLike | None = None,
) -> tuple[SequenceBatch, SequenceLoweringReport]:
    """Lower host strings to fixed capacities and return a complete loss audit."""
    if not isinstance(plan, SequenceLoweringPlan):
        raise TypeError("plan must be a SequenceLoweringPlan.")
    strings = tuple(sequences)
    if any(not isinstance(sequence, str) for sequence in strings):
        raise TypeError("Every sequence must be a string.")
    input_count = len(strings)
    if record_ids is None:
        input_ids = np.arange(input_count, dtype=np.int32)
    else:
        input_ids = np.asarray(record_ids)
        if input_ids.shape != (input_count,) or not np.issubdtype(
            input_ids.dtype, np.integer
        ):
            raise ValueError("record_ids must be an integer vector matching sequences.")

    retained_count = min(input_count, plan.record_capacity)
    record_overflow = input_count - retained_count
    overlong = any(
        len(sequence) > plan.sequence_capacity for sequence in strings[:retained_count]
    )
    if plan.overflow_policy == "reject" and (record_overflow or overlong):
        raise OverflowError(
            "Sequence input exceeds the declared record or sequence capacity."
        )

    pad_code = plan.alphabet.code(plan.alphabet.pad_symbol)
    tokens = np.full(
        (plan.record_capacity, plan.sequence_capacity), pad_code, dtype=np.int32
    )
    valid = np.zeros(tokens.shape, dtype=bool)
    soft = np.zeros(tokens.shape, dtype=bool)
    cases = np.zeros((plan.record_capacity,), dtype=bool)
    output_ids = np.zeros((plan.record_capacity,), dtype=input_ids.dtype)
    original_lengths = np.asarray([len(sequence) for sequence in strings], dtype=np.int32)
    retained_lengths = np.zeros((input_count,), dtype=np.int32)
    truncated_counts = original_lengths.copy()
    mapped_counts = np.zeros((input_count,), dtype=np.int32)
    retained_records = np.zeros((input_count,), dtype=bool)

    alphabet_mapping = plan.alphabet.symbol_to_code
    unknown_code = alphabet_mapping[plan.alphabet.unknown_symbol]
    for record_index in range(retained_count):
        sequence = strings[record_index]
        retained_length = min(len(sequence), plan.sequence_capacity)
        output_ids[record_index] = input_ids[record_index]
        cases[record_index] = True
        retained_records[record_index] = True
        retained_lengths[record_index] = retained_length
        truncated_counts[record_index] = len(sequence) - retained_length
        for position, raw_symbol in enumerate(sequence[:retained_length]):
            symbol = _normalize_symbol(raw_symbol)
            if symbol not in alphabet_mapping:
                if plan.invalid_symbol_policy == "reject":
                    raise ValueError(
                        f"Invalid symbol {raw_symbol!r} in record {record_index} at "
                        f"position {position}."
                    )
                code = unknown_code
                mapped_counts[record_index] += 1
            else:
                code = alphabet_mapping[symbol]
            tokens[record_index, position] = code
            valid[record_index, position] = True
            soft[record_index, position] = (
                plan.preserve_soft_mask and raw_symbol.isalpha() and raw_symbol.islower()
            )

    batch = SequenceBatch(
        output_ids,
        tokens,
        valid,
        cases,
        soft,
        plan.alphabet,
    )
    report = SequenceLoweringReport(
        np.asarray(input_count, dtype=np.int32),
        np.asarray(retained_count, dtype=np.int32),
        np.asarray(record_overflow, dtype=np.int32),
        original_lengths,
        retained_lengths,
        truncated_counts,
        mapped_counts,
        retained_records,
        plan_fingerprint=plan.fingerprint,
    )
    return batch, report


def reverse_complement(batch: SequenceBatch) -> SequenceBatch:
    """Reverse-complement every populated sequence while preserving all masks."""
    if not isinstance(batch, SequenceBatch):
        raise TypeError("batch must be a SequenceBatch.")
    if not batch.alphabet.complements:
        raise ValueError(f"Alphabet {batch.alphabet.alphabet_id!r} has no complement.")
    complement_codes = jnp.asarray(
        [
            batch.alphabet.code(batch.alphabet.complement_map[symbol])
            for symbol in batch.alphabet.symbols
        ],
        dtype=batch.token_codes.dtype,
    )
    length = batch.lengths
    positions = jnp.arange(batch.sequence_capacity)[None, :]
    source = jnp.clip(
        length[:, None] - 1 - positions, 0, max(batch.sequence_capacity - 1, 0)
    )
    if batch.sequence_capacity == 0:
        reversed_tokens = batch.token_codes
        reversed_soft = batch.soft_mask
    else:
        gathered = jnp.take_along_axis(batch.token_codes, source, axis=1)
        gathered_soft = jnp.take_along_axis(batch.soft_mask, source, axis=1)
        reversed_tokens = complement_codes[gathered]
        reversed_soft = gathered_soft
    output_valid = positions < length[:, None]
    pad_code = batch.alphabet.code(batch.alphabet.pad_symbol)
    output_tokens = jnp.where(output_valid, reversed_tokens, pad_code)
    output_soft = output_valid & reversed_soft
    return SequenceBatch(
        batch.record_ids,
        output_tokens,
        output_valid,
        batch.case_mask,
        output_soft,
        batch.alphabet,
    )


__all__ = [
    "InvalidSymbolPolicy",
    "OverflowPolicy",
    "SequenceLoweringPlan",
    "SequenceLoweringReport",
    "decode_sequence",
    "decode_sequences",
    "encode_sequence",
    "encode_sequences",
    "lower_sequences",
    "reverse_complement",
]
