#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from numbers import Integral
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax._fingerprint import canonical_fingerprint
from phydrax._strict import StrictModule
from phydrax.bioinformatics.sequence._alphabet import (
    AlphabetPlan,
    PROTEIN_IUPAC,
)
from phydrax.bioinformatics.sequence._batch import _concrete, SequenceBatch
from phydrax.bioinformatics.sequence._transforms import reverse_complement


AmbiguousCodonPolicy = Literal["reject", "unknown", "consensus"]
IncompleteCodonPolicy = Literal["reject", "drop", "unknown"]
StopCodonPolicy = Literal["reject", "keep", "truncate"]
Strand = Literal["forward", "reverse"]


class GeneticCode(StrictModule):
    """Immutable complete mapping from canonical DNA codons to amino acids."""

    code_id: str = eqx.field(static=True)
    codon_table: tuple[tuple[str, str], ...] = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(self, code_id: str, codon_table: tuple[tuple[str, str], ...]):
        identifier = str(code_id).strip()
        items = tuple(
            (str(codon).upper(), str(amino).upper()) for codon, amino in codon_table
        )
        mapping = dict(items)
        expected = {
            "".join(codon) for codon in itertools.product(("T", "C", "A", "G"), repeat=3)
        }
        if not identifier:
            raise ValueError("code_id must be non-empty.")
        if len(mapping) != len(items) or set(mapping) != expected:
            raise ValueError(
                "A genetic code must map each of the 64 canonical codons once."
            )
        allowed_outputs = set(PROTEIN_IUPAC.canonical_symbols) | {"*"}
        if any(
            len(amino) != 1 or amino not in allowed_outputs for amino in mapping.values()
        ):
            raise ValueError("Genetic-code outputs must be amino-acid or stop symbols.")
        ordered = tuple(sorted(mapping.items()))
        self.code_id = identifier
        self.codon_table = ordered
        self.fingerprint = canonical_fingerprint(
            {"code_id": identifier, "codon_table": ordered}
        )

    @property
    def mapping(self) -> dict[str, str]:
        return dict(self.codon_table)


_STANDARD_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}
STANDARD_GENETIC_CODE = GeneticCode("standard", tuple(_STANDARD_TABLE.items()))


class TranslationPlan(StrictModule):
    """Static frame, strand, code, and explicit exceptional-codon policies."""

    genetic_code: GeneticCode = eqx.field(static=True)
    output_alphabet: AlphabetPlan = eqx.field(static=True)
    frame: int = eqx.field(static=True)
    strand: Strand = eqx.field(static=True)
    ambiguous_policy: AmbiguousCodonPolicy = eqx.field(static=True)
    incomplete_policy: IncompleteCodonPolicy = eqx.field(static=True)
    stop_policy: StopCodonPolicy = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        frame: int,
        strand: Strand,
        ambiguous_policy: AmbiguousCodonPolicy,
        incomplete_policy: IncompleteCodonPolicy,
        stop_policy: StopCodonPolicy,
        genetic_code: GeneticCode = STANDARD_GENETIC_CODE,
        output_alphabet: AlphabetPlan = PROTEIN_IUPAC,
    ):
        if not isinstance(genetic_code, GeneticCode):
            raise TypeError("genetic_code must be a GeneticCode.")
        if not isinstance(output_alphabet, AlphabetPlan):
            raise TypeError("output_alphabet must be an AlphabetPlan.")
        if isinstance(frame, bool) or not isinstance(frame, Integral):
            raise TypeError("frame must be an integer.")
        if frame not in (0, 1, 2):
            raise ValueError("frame must be 0, 1, or 2.")
        if strand not in ("forward", "reverse"):
            raise ValueError("strand must be 'forward' or 'reverse'.")
        if ambiguous_policy not in ("reject", "unknown", "consensus"):
            raise ValueError(
                "ambiguous_policy must be 'reject', 'unknown', or 'consensus'."
            )
        if incomplete_policy not in ("reject", "drop", "unknown"):
            raise ValueError("incomplete_policy must be 'reject', 'drop', or 'unknown'.")
        if stop_policy not in ("reject", "keep", "truncate"):
            raise ValueError("stop_policy must be 'reject', 'keep', or 'truncate'.")
        if "X" not in output_alphabet.symbols or "*" not in output_alphabet.symbols:
            raise ValueError("Translation output alphabet must contain X and * symbols.")
        if any(
            amino not in output_alphabet.symbols for _, amino in genetic_code.codon_table
        ):
            raise ValueError(
                "Translation output alphabet does not cover the genetic code."
            )
        payload = {
            "genetic_code": genetic_code.fingerprint,
            "output_alphabet": output_alphabet.fingerprint,
            "frame": frame,
            "strand": strand,
            "ambiguous_policy": ambiguous_policy,
            "incomplete_policy": incomplete_policy,
            "stop_policy": stop_policy,
        }
        self.genetic_code = genetic_code
        self.output_alphabet = output_alphabet
        self.frame = int(frame)
        self.strand = strand
        self.ambiguous_policy = ambiguous_policy
        self.incomplete_policy = incomplete_policy
        self.stop_policy = stop_policy
        self.fingerprint = canonical_fingerprint(payload)


class TranslationReport(StrictModule):
    """Array-only per-record audit of ambiguous, partial, and stop codons."""

    ambiguous_codon_counts: Array
    incomplete_base_counts: Array
    stop_codon_counts: Array
    output_lengths: Array
    valid: Array
    plan_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        ambiguous_codon_counts: ArrayLike,
        incomplete_base_counts: ArrayLike,
        stop_codon_counts: ArrayLike,
        output_lengths: ArrayLike,
        valid: ArrayLike,
        *,
        plan_fingerprint: str,
    ):
        ambiguous = jnp.asarray(ambiguous_codon_counts, dtype=jnp.int32)
        incomplete = jnp.asarray(incomplete_base_counts, dtype=jnp.int32)
        stops = jnp.asarray(stop_codon_counts, dtype=jnp.int32)
        lengths = jnp.asarray(output_lengths, dtype=jnp.int32)
        validity = jnp.asarray(valid)
        if ambiguous.ndim != 1 or any(
            values.shape != ambiguous.shape
            for values in (incomplete, stops, lengths, validity)
        ):
            raise ValueError("Translation report arrays must share shape (record,).")
        if validity.dtype != jnp.bool_:
            raise ValueError("Translation report valid must be boolean.")
        if not plan_fingerprint:
            raise ValueError("plan_fingerprint must be non-empty.")
        concrete_arrays = tuple(
            _concrete(values) for values in (ambiguous, incomplete, stops, lengths)
        )
        for values in concrete_arrays:
            if values is not None and np.any(values < 0):
                raise ValueError("Translation report counts must be non-negative.")
        concrete_incomplete = concrete_arrays[1]
        if concrete_incomplete is not None and np.any(concrete_incomplete > 2):
            raise ValueError("Incomplete base counts cannot exceed two.")
        self.ambiguous_codon_counts = ambiguous
        self.incomplete_base_counts = incomplete
        self.stop_codon_counts = stops
        self.output_lengths = lengths
        self.valid = validity
        self.plan_fingerprint = str(plan_fingerprint)


class TranslationResult(StrictModule):
    """Translated protein batch and its explicit codon-policy audit."""

    sequences: SequenceBatch
    report: TranslationReport

    def __init__(self, sequences: SequenceBatch, report: TranslationReport):
        if not isinstance(sequences, SequenceBatch):
            raise TypeError("sequences must be a SequenceBatch.")
        if not isinstance(report, TranslationReport):
            raise TypeError("report must be a TranslationReport.")
        if sequences.record_capacity != report.output_lengths.shape[0]:
            raise ValueError("Translation batch and report record shapes must match.")
        concrete_lengths = _concrete(report.output_lengths)
        sequence_lengths = _concrete(sequences.lengths)
        if (
            concrete_lengths is not None
            and sequence_lengths is not None
            and not np.array_equal(concrete_lengths, sequence_lengths)
        ):
            raise ValueError("Translation report lengths must match the protein batch.")
        self.sequences = sequences
        self.report = report


def _expansions(alphabet: AlphabetPlan, symbol: str) -> tuple[str, ...] | None:
    if symbol in alphabet.canonical_symbols:
        canonical = "T" if symbol == "U" else symbol
        return (canonical,)
    if symbol in alphabet.ambiguity_map:
        return tuple(
            "T" if value == "U" else value for value in alphabet.ambiguity_map[symbol]
        )
    return None


def _translation_lookup(
    source: AlphabetPlan,
    plan: TranslationPlan,
) -> tuple[np.ndarray, np.ndarray]:
    size = source.size
    amino_codes = np.empty((size, size, size), dtype=np.int32)
    exceptional = np.zeros((size, size, size), dtype=bool)
    unknown_code = plan.output_alphabet.code("X")
    genetic_mapping = plan.genetic_code.mapping
    for first in range(size):
        for second in range(size):
            for third in range(size):
                first_choices = _expansions(source, source.symbols[first])
                second_choices = _expansions(source, source.symbols[second])
                third_choices = _expansions(source, source.symbols[third])
                if (
                    first_choices is None
                    or second_choices is None
                    or third_choices is None
                ):
                    amino_codes[first, second, third] = unknown_code
                    exceptional[first, second, third] = True
                    continue
                choices = (first_choices, second_choices, third_choices)
                expanded = tuple(
                    genetic_mapping["".join(codon)]
                    for codon in itertools.product(*choices)
                )
                is_ambiguous = any(len(choice) > 1 for choice in choices)
                exceptional[first, second, third] = is_ambiguous
                if is_ambiguous and plan.ambiguous_policy == "unknown":
                    amino = "X"
                elif is_ambiguous and len(set(expanded)) != 1:
                    amino = "X"
                else:
                    amino = expanded[0]
                amino_codes[first, second, third] = plan.output_alphabet.code(amino)
    return amino_codes.reshape((-1,)), exceptional.reshape((-1,))


def translate(batch: SequenceBatch, plan: TranslationPlan) -> TranslationResult:
    """Translate a numeric nucleotide batch under one fully explicit policy plan."""
    if not isinstance(batch, SequenceBatch):
        raise TypeError("batch must be a SequenceBatch.")
    if not isinstance(plan, TranslationPlan):
        raise TypeError("plan must be a TranslationPlan.")
    normalized_canonical = {
        "T" if symbol == "U" else symbol for symbol in batch.alphabet.canonical_symbols
    }
    if normalized_canonical != {"A", "C", "G", "T"}:
        raise ValueError("Translation requires a DNA or RNA nucleotide alphabet.")
    if not batch.alphabet.complements and plan.strand == "reverse":
        raise ValueError("Reverse-strand translation requires a complement alphabet.")
    source_batch = reverse_complement(batch) if plan.strand == "reverse" else batch
    lookup, exceptional_lookup = _translation_lookup(source_batch.alphabet, plan)
    lookup_array = jnp.asarray(lookup, dtype=jnp.int32)
    exceptional_array = jnp.asarray(exceptional_lookup)

    available = jnp.maximum(source_batch.lengths - plan.frame, 0)
    full_counts = available // 3
    remainder = available % 3
    available_capacity = max(source_batch.sequence_capacity - plan.frame, 0)
    if plan.incomplete_policy == "unknown":
        output_capacity = (available_capacity + 2) // 3
    else:
        output_capacity = available_capacity // 3
    slots = jnp.arange(output_capacity)[None, :]
    full_valid = slots < full_counts[:, None]
    partial_valid = (
        (slots == full_counts[:, None])
        & (remainder[:, None] > 0)
        & source_batch.case_mask[:, None]
    )
    if plan.incomplete_policy != "unknown":
        partial_valid = jnp.zeros_like(partial_valid)

    if output_capacity == 0:
        amino_codes = jnp.zeros((source_batch.record_capacity, 0), dtype=jnp.int32)
        exceptional_codons = jnp.zeros_like(amino_codes, dtype=bool)
        codon_soft = jnp.zeros_like(amino_codes, dtype=bool)
    else:
        codon_positions = (
            plan.frame + 3 * slots[:, :, None] + jnp.arange(3)[None, None, :]
        )
        clipped = jnp.clip(
            codon_positions,
            0,
            max(source_batch.sequence_capacity - 1, 0),
        )
        gathered = jnp.take_along_axis(
            source_batch.token_codes[:, None, :], clipped, axis=2
        )
        gathered_soft = jnp.take_along_axis(
            source_batch.soft_mask[:, None, :], clipped, axis=2
        )
        size = source_batch.alphabet.size
        indices = (gathered[:, :, 0] * size + gathered[:, :, 1]) * size + gathered[
            :, :, 2
        ]
        amino_codes = lookup_array[indices]
        exceptional_codons = exceptional_array[indices] & full_valid
        codon_soft = jnp.any(gathered_soft, axis=2)

    concrete_remainder = _concrete(remainder)
    concrete_exceptional = _concrete(exceptional_codons)
    if (
        plan.incomplete_policy == "reject"
        and concrete_remainder is not None
        and np.any(concrete_remainder > 0)
    ):
        raise ValueError("An incomplete terminal codon is forbidden by the plan.")
    if (
        plan.ambiguous_policy == "reject"
        and concrete_exceptional is not None
        and np.any(concrete_exceptional)
    ):
        raise ValueError("An ambiguous or non-nucleotide codon is forbidden by the plan.")

    unknown_code = plan.output_alphabet.code("X")
    amino_codes = jnp.where(partial_valid, unknown_code, amino_codes)
    codon_valid = full_valid | partial_valid
    stop_code = plan.output_alphabet.code("*")
    stop_mask = (amino_codes == stop_code) & full_valid
    concrete_stops = _concrete(stop_mask)
    if (
        plan.stop_policy == "reject"
        and concrete_stops is not None
        and np.any(concrete_stops)
    ):
        raise ValueError("A stop codon is forbidden by the translation plan.")
    if plan.stop_policy == "truncate":
        codon_valid = codon_valid & (jnp.cumsum(stop_mask, axis=1) == 0)

    pad_code = plan.output_alphabet.code(plan.output_alphabet.pad_symbol)
    output_codes = jnp.where(codon_valid, amino_codes, pad_code)
    output_soft = codon_valid & codon_soft
    translated = SequenceBatch(
        source_batch.record_ids,
        output_codes,
        codon_valid,
        source_batch.case_mask,
        output_soft,
        plan.output_alphabet,
    )
    ambiguous_counts = jnp.sum(exceptional_codons, axis=1, dtype=jnp.int32)
    incomplete_counts = jnp.where(source_batch.case_mask, remainder, 0)
    stop_counts = jnp.sum(stop_mask, axis=1, dtype=jnp.int32)
    policy_valid = source_batch.case_mask
    if plan.ambiguous_policy == "reject":
        policy_valid = policy_valid & (ambiguous_counts == 0)
    if plan.incomplete_policy == "reject":
        policy_valid = policy_valid & (incomplete_counts == 0)
    if plan.stop_policy == "reject":
        policy_valid = policy_valid & (stop_counts == 0)
    report = TranslationReport(
        ambiguous_counts,
        incomplete_counts,
        stop_counts,
        translated.lengths,
        policy_valid,
        plan_fingerprint=plan.fingerprint,
    )
    return TranslationResult(translated, report)


__all__ = [
    "AmbiguousCodonPolicy",
    "GeneticCode",
    "IncompleteCodonPolicy",
    "STANDARD_GENETIC_CODE",
    "StopCodonPolicy",
    "Strand",
    "TranslationPlan",
    "TranslationReport",
    "TranslationResult",
    "translate",
]
