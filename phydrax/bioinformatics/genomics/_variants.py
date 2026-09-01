#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Normalized bounded germline small variants.

Somatic and structural events are intentionally out of scope.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import AlphabetPlan, DNA_IUPAC


class VariantNormalizationStatus(IntEnum):
    """Machine-readable outcome of exact small-variant normalization."""

    OK = 0
    INVALID_POSITION = 1
    REFERENCE_MISMATCH = 2
    INVALID_ALLELE = 3
    CAPACITY_EXCEEDED = 4
    DUPLICATE_ALLELE = 5


VARIANT_NORMALIZATION_CONTRACT = BioinformaticsMethodContract(
    "small_variant_left_normalization",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Exact discrete normalization relative to the supplied linear reference; "
        "allele order is preserved."
    ),
    truncation_statement="Alleles and reference context are never truncated.",
    capacity_semantics=(
        "The allele-count and allele-length capacities are preflighted; insufficient "
        "capacity returns CAPACITY_EXCEEDED and an empty bounded site."
    ),
    assumptions=(
        "Germline small variants on a linear reference",
        "Zero-based positions in the scientific core",
        "Non-symbolic non-empty alleles from one explicit alphabet",
    ),
    nondifferentiable_outputs=(
        "position",
        "allele_codes",
        "allele_lengths",
        "allele_mask",
        "status",
    ),
    input_dtype="uint8",
    output_dtype="uint8",
)


class SmallVariantSite(StrictModule):
    """A normalized, fixed-capacity germline small-variant site.

    Allele zero is the reference. Populated allele rows are left-aligned and
    prefix-valid; unused rows and suffix positions contain the alphabet pad code.
    Host contig names and allele strings deliberately do not enter this PyTree.
    """

    reference_index: Array
    contig_index: Array
    position: Array
    allele_codes: Array
    allele_lengths: Array
    allele_mask: Array
    alphabet: AlphabetPlan = eqx.field(static=True)

    def __init__(
        self,
        reference_index: ArrayLike,
        contig_index: ArrayLike,
        position: ArrayLike,
        allele_codes: ArrayLike,
        allele_lengths: ArrayLike,
        allele_mask: ArrayLike,
        alphabet: AlphabetPlan = DNA_IUPAC,
    ):
        codes = jnp.asarray(allele_codes, dtype=jnp.uint8)
        lengths = jnp.asarray(allele_lengths, dtype=jnp.int32)
        mask = jnp.asarray(allele_mask, dtype=bool)
        if codes.ndim != 2:
            raise ValueError(
                "allele_codes must have shape (allele_capacity, base_capacity)."
            )
        if lengths.shape != (codes.shape[0],) or mask.shape != lengths.shape:
            raise ValueError("allele_lengths and allele_mask must match allele capacity.")
        self.reference_index = jax.lax.stop_gradient(
            jnp.asarray(reference_index, dtype=jnp.int32)
        )
        self.contig_index = jax.lax.stop_gradient(
            jnp.asarray(contig_index, dtype=jnp.int32)
        )
        self.position = jax.lax.stop_gradient(jnp.asarray(position, dtype=jnp.int64))
        self.allele_codes = jax.lax.stop_gradient(codes)
        self.allele_lengths = jax.lax.stop_gradient(lengths)
        self.allele_mask = jax.lax.stop_gradient(mask)
        self.alphabet = alphabet

    @property
    def allele_capacity(self) -> int:
        return self.allele_codes.shape[0]

    @property
    def base_capacity(self) -> int:
        return self.allele_codes.shape[1]


class VariantNormalizationEvidence(StrictModule):
    """Auditable coordinate and capacity facts from normalization."""

    original_position: Array
    normalized_position: Array
    left_shift: Array
    trimmed_prefix: Array
    trimmed_suffix: Array
    required_allele_count: Array
    required_max_allele_length: Array
    reference_match: Array


class VariantNormalizationResult(StrictModule):
    """Bounded normalized site plus explicit success, status, and evidence."""

    site: SmallVariantSite
    valid: Array
    status: Array
    evidence: VariantNormalizationEvidence
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _empty_site(
    reference_index: int,
    contig_index: int,
    position: int,
    allele_capacity: int,
    base_capacity: int,
    alphabet: AlphabetPlan,
) -> SmallVariantSite:
    pad = alphabet.code(alphabet.pad_symbol)
    return SmallVariantSite(
        jnp.asarray(reference_index, dtype=jnp.int32),
        jnp.asarray(contig_index, dtype=jnp.int32),
        jnp.asarray(position, dtype=jnp.int64),
        jnp.full((allele_capacity, base_capacity), pad, dtype=jnp.uint8),
        jnp.zeros((allele_capacity,), dtype=jnp.int32),
        jnp.zeros((allele_capacity,), dtype=bool),
        alphabet,
    )


def _normalization_result(
    site: SmallVariantSite,
    status: VariantNormalizationStatus,
    *,
    original_position: int,
    normalized_position: int,
    left_shift: int,
    trimmed_prefix: int,
    trimmed_suffix: int,
    required_allele_count: int,
    required_max_allele_length: int,
    reference_match: bool,
) -> VariantNormalizationResult:
    return VariantNormalizationResult(
        site,
        jnp.asarray(status is VariantNormalizationStatus.OK),
        jnp.asarray(int(status), dtype=jnp.int32),
        VariantNormalizationEvidence(
            jnp.asarray(original_position, dtype=jnp.int64),
            jnp.asarray(normalized_position, dtype=jnp.int64),
            jnp.asarray(left_shift, dtype=jnp.int32),
            jnp.asarray(trimmed_prefix, dtype=jnp.int32),
            jnp.asarray(trimmed_suffix, dtype=jnp.int32),
            jnp.asarray(required_allele_count, dtype=jnp.int32),
            jnp.asarray(required_max_allele_length, dtype=jnp.int32),
            jnp.asarray(reference_match),
        ),
        VARIANT_NORMALIZATION_CONTRACT,
    )


def _encode_site(
    reference_index: int,
    contig_index: int,
    position: int,
    alleles: tuple[str, ...],
    allele_capacity: int,
    base_capacity: int,
    alphabet: AlphabetPlan,
) -> SmallVariantSite:
    pad = alphabet.code(alphabet.pad_symbol)
    rows = [[pad] * base_capacity for _ in range(allele_capacity)]
    lengths = [0] * allele_capacity
    mask = [False] * allele_capacity
    for allele_index, allele in enumerate(alleles):
        encoded = [alphabet.code(symbol) for symbol in allele]
        rows[allele_index][: len(encoded)] = encoded
        lengths[allele_index] = len(encoded)
        mask[allele_index] = True
    return SmallVariantSite(
        reference_index,
        contig_index,
        position,
        jnp.asarray(rows, dtype=jnp.uint8),
        jnp.asarray(lengths, dtype=jnp.int32),
        jnp.asarray(mask, dtype=bool),
        alphabet,
    )


def decode_variant_alleles(site: SmallVariantSite, /) -> tuple[str, ...]:
    """Decode populated bounded allele rows at an explicit host boundary."""
    symbols = site.alphabet.symbols
    codes = site.allele_codes.tolist()
    lengths = site.allele_lengths.tolist()
    mask = site.allele_mask.tolist()
    return tuple(
        "".join(symbols[code] for code in row[:length])
        for row, length, populated in zip(codes, lengths, mask, strict=True)
        if populated
    )


def normalize_small_variant(
    reference_sequence: str,
    position: int,
    reference_allele: str,
    alternate_alleles: Sequence[str],
    /,
    *,
    reference_index: int = 0,
    contig_index: int = 0,
    max_alleles: int,
    max_allele_length: int,
    alphabet: AlphabetPlan = DNA_IUPAC,
) -> VariantNormalizationResult:
    """Exactly minimize and left-align one bounded germline small-variant site.

    The input position is zero-based. Capacity failure, invalid alleles, reference
    mismatch, and contig-edge errors are represented in the returned result rather
    than by partial output. Symbolic VCF alleles and breakends are intentionally out
    of scope.
    """
    allele_capacity = int(max_alleles)
    base_capacity = int(max_allele_length)
    if allele_capacity < 1 or base_capacity < 1:
        raise ValueError("max_alleles and max_allele_length must be positive.")

    reference = str(reference_sequence).upper()
    original_position = int(position)
    ref = str(reference_allele).upper()
    alts = tuple(str(allele).upper() for allele in alternate_alleles)
    alleles = (ref, *alts)
    required_count = len(alleles)
    required_length = max((len(allele) for allele in alleles), default=0)
    empty = _empty_site(
        reference_index,
        contig_index,
        max(original_position, 0),
        allele_capacity,
        base_capacity,
        alphabet,
    )

    if required_count > allele_capacity:
        return _normalization_result(
            empty,
            VariantNormalizationStatus.CAPACITY_EXCEEDED,
            original_position=original_position,
            normalized_position=original_position,
            left_shift=0,
            trimmed_prefix=0,
            trimmed_suffix=0,
            required_allele_count=required_count,
            required_max_allele_length=required_length,
            reference_match=False,
        )

    biological_symbols = set(alphabet.canonical_symbols) | {
        symbol for symbol, _ in alphabet.ambiguities
    }
    valid_alleles = (
        required_count >= 2
        and all(allele for allele in alleles)
        and all(symbol in biological_symbols for allele in alleles for symbol in allele)
    )
    if not valid_alleles:
        return _normalization_result(
            empty,
            VariantNormalizationStatus.INVALID_ALLELE,
            original_position=original_position,
            normalized_position=original_position,
            left_shift=0,
            trimmed_prefix=0,
            trimmed_suffix=0,
            required_allele_count=required_count,
            required_max_allele_length=required_length,
            reference_match=False,
        )
    if len(set(alleles)) != len(alleles):
        return _normalization_result(
            empty,
            VariantNormalizationStatus.DUPLICATE_ALLELE,
            original_position=original_position,
            normalized_position=original_position,
            left_shift=0,
            trimmed_prefix=0,
            trimmed_suffix=0,
            required_allele_count=required_count,
            required_max_allele_length=required_length,
            reference_match=False,
        )
    if original_position < 0 or original_position + len(ref) > len(reference):
        return _normalization_result(
            empty,
            VariantNormalizationStatus.INVALID_POSITION,
            original_position=original_position,
            normalized_position=original_position,
            left_shift=0,
            trimmed_prefix=0,
            trimmed_suffix=0,
            required_allele_count=required_count,
            required_max_allele_length=required_length,
            reference_match=False,
        )

    reference_match = reference[original_position : original_position + len(ref)] == ref
    if not reference_match:
        return _normalization_result(
            empty,
            VariantNormalizationStatus.REFERENCE_MISMATCH,
            original_position=original_position,
            normalized_position=original_position,
            left_shift=0,
            trimmed_prefix=0,
            trimmed_suffix=0,
            required_allele_count=required_count,
            required_max_allele_length=required_length,
            reference_match=False,
        )

    normalized = list(alleles)
    normalized_position = original_position
    trimmed_suffix = 0
    while (
        min(map(len, normalized)) > 1 and len({allele[-1] for allele in normalized}) == 1
    ):
        normalized = [allele[:-1] for allele in normalized]
        trimmed_suffix += 1

    trimmed_prefix = 0
    while (
        min(map(len, normalized)) > 1 and len({allele[0] for allele in normalized}) == 1
    ):
        normalized = [allele[1:] for allele in normalized]
        normalized_position += 1
        trimmed_prefix += 1

    left_shift = 0
    is_indel = len({len(allele) for allele in normalized}) > 1
    while is_indel and normalized_position > 0:
        if len({allele[-1] for allele in normalized}) != 1:
            break
        preceding = reference[normalized_position - 1]
        shifted = [preceding + allele[:-1] for allele in normalized]
        shifted_reference = reference[
            normalized_position - 1 : normalized_position - 1 + len(shifted[0])
        ]
        if shifted[0] != shifted_reference:
            break
        normalized = shifted
        normalized_position -= 1
        left_shift += 1

    while (
        min(map(len, normalized)) > 1 and len({allele[-1] for allele in normalized}) == 1
    ):
        normalized = [allele[:-1] for allele in normalized]
        trimmed_suffix += 1
    while (
        min(map(len, normalized)) > 1 and len({allele[0] for allele in normalized}) == 1
    ):
        normalized = [allele[1:] for allele in normalized]
        normalized_position += 1
        trimmed_prefix += 1

    normalized_alleles = tuple(normalized)
    normalized_required_length = max(map(len, normalized_alleles))
    if normalized_required_length > base_capacity:
        return _normalization_result(
            empty,
            VariantNormalizationStatus.CAPACITY_EXCEEDED,
            original_position=original_position,
            normalized_position=normalized_position,
            left_shift=left_shift,
            trimmed_prefix=trimmed_prefix,
            trimmed_suffix=trimmed_suffix,
            required_allele_count=required_count,
            required_max_allele_length=normalized_required_length,
            reference_match=True,
        )
    site = _encode_site(
        reference_index,
        contig_index,
        normalized_position,
        normalized_alleles,
        allele_capacity,
        base_capacity,
        alphabet,
    )
    return _normalization_result(
        site,
        VariantNormalizationStatus.OK,
        original_position=original_position,
        normalized_position=normalized_position,
        left_shift=left_shift,
        trimmed_prefix=trimmed_prefix,
        trimmed_suffix=trimmed_suffix,
        required_allele_count=required_count,
        required_max_allele_length=normalized_required_length,
        reference_match=True,
    )


__all__ = [
    "SmallVariantSite",
    "VARIANT_NORMALIZATION_CONTRACT",
    "VariantNormalizationEvidence",
    "VariantNormalizationResult",
    "VariantNormalizationStatus",
    "decode_variant_alleles",
    "normalize_small_variant",
]
