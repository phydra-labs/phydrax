#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from pathlib import Path

import jax
import numpy as np
import pytest

from phydrax.bioinformatics.interchange._fastx import (
    FASTARecord,
    FASTQRecord,
    lower_fasta,
    lower_fastq,
    parse_fasta,
    parse_fastq,
    write_fasta,
    write_fastq,
)
from phydrax.bioinformatics.sequence import (
    decode_sequence,
    decode_sequences,
    DNA_IUPAC,
    encode_sequence,
    encode_sequences,
    GeneticCode,
    lower_sequences,
    PhredEncoding,
    PROTEIN_IUPAC,
    QualityBatch,
    reverse_complement,
    RNA_IUPAC,
    SequenceDistribution,
    SequenceLoweringPlan,
    STANDARD_GENETIC_CODE,
    translate,
    TranslationPlan,
)


def _translation_plan(
    *,
    frame: int = 0,
    strand: str = "forward",
    ambiguous: str = "consensus",
    incomplete: str = "drop",
    stop: str = "keep",
    genetic_code: GeneticCode = STANDARD_GENETIC_CODE,
) -> TranslationPlan:
    return TranslationPlan(
        frame=frame,
        strand=strand,
        ambiguous_policy=ambiguous,
        incomplete_policy=incomplete,
        stop_policy=stop,
        genetic_code=genetic_code,
    )


def test_iupac_alphabets_keep_ambiguity_and_special_semantics_distinct() -> None:
    for alphabet in (DNA_IUPAC, RNA_IUPAC, PROTEIN_IUPAC):
        special = {
            alphabet.gap_symbol,
            alphabet.pad_symbol,
            alphabet.unknown_symbol,
            alphabet.missing_symbol,
            alphabet.mask_symbol,
        }
        assert len(special) == 5
        assert special.isdisjoint(alphabet.canonical_symbols)
        assert len(alphabet.symbols) == len(set(alphabet.symbols))
        encoded, soft = encode_sequence("".join(alphabet.symbols), alphabet)
        assert decode_sequence(encoded, alphabet, soft_mask=soft) == "".join(
            alphabet.symbols
        )

    assert dict(DNA_IUPAC.ambiguities) == {
        "R": ("A", "G"),
        "Y": ("C", "T"),
        "S": ("C", "G"),
        "W": ("A", "T"),
        "K": ("G", "T"),
        "M": ("A", "C"),
        "B": ("C", "G", "T"),
        "D": ("A", "G", "T"),
        "H": ("A", "C", "T"),
        "V": ("A", "C", "G"),
        "N": ("A", "C", "G", "T"),
    }
    assert set(RNA_IUPAC.ambiguity_map["N"]) == {"A", "C", "G", "U"}
    assert PROTEIN_IUPAC.ambiguity_map["B"] == ("D", "N")
    assert PROTEIN_IUPAC.ambiguity_map["Z"] == ("E", "Q")
    assert PROTEIN_IUPAC.ambiguity_map["J"] == ("I", "L")
    assert len(PROTEIN_IUPAC.ambiguity_map["X"]) == 22


def test_exact_host_encoding_round_trip_preserves_empty_specials_and_soft_mask() -> None:
    text = "aCgTrYsWkMbDhVn-_?.#"
    codes, soft = encode_sequence(text, DNA_IUPAC)
    assert decode_sequence(codes, DNA_IUPAC, soft_mask=soft) == text
    empty_codes, empty_soft = encode_sequence("", DNA_IUPAC)
    assert empty_codes.shape == (0,)
    assert empty_soft.shape == (0,)

    batch = encode_sequences([text, "", "NN"], DNA_IUPAC, record_ids=[7, 8, 9])
    assert batch.case_mask.shape == (3,)
    assert batch.soft_mask.shape == batch.token_codes.shape
    assert decode_sequences(batch) == (text, "", "NN")
    leaves = jax.tree_util.tree_leaves(batch)
    assert leaves
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    assert all(leaf.dtype.kind not in "OUS" for leaf in leaves)


def test_invalid_symbols_only_map_under_declared_policy() -> None:
    with pytest.raises(ValueError, match="Invalid symbol"):
        encode_sequence("ACGT%", DNA_IUPAC)
    codes, _ = encode_sequence("ACGT%", DNA_IUPAC, invalid_symbol_policy="unknown")
    assert DNA_IUPAC.symbols[int(codes[-1])] == DNA_IUPAC.unknown_symbol


def test_sequence_distribution_is_separate_and_requires_normalized_valid_rows() -> None:
    probabilities = np.zeros((1, 2, DNA_IUPAC.size), dtype=np.float32)
    probabilities[0, 0, DNA_IUPAC.code("A")] = 1.0
    distribution = SequenceDistribution(
        [4], probabilities, [[True, False]], [True], DNA_IUPAC
    )
    assert distribution.probabilities.shape == (1, 2, DNA_IUPAC.size)
    with pytest.raises(ValueError, match="sum to one"):
        SequenceDistribution([4], probabilities * 0.5, [[True, False]], [True], DNA_IUPAC)


def test_reverse_complement_is_an_involution_with_ambiguity_and_soft_masks() -> None:
    batch = encode_sequences(["aCgTRYswKMBDHVN-_.?#", ""], DNA_IUPAC)
    complemented = reverse_complement(batch)
    assert decode_sequences(complemented) == ("#?._-NBDHVKMwsRYAcGt", "")
    restored = reverse_complement(complemented)
    np.testing.assert_array_equal(restored.token_codes, batch.token_codes)
    np.testing.assert_array_equal(restored.valid_mask, batch.valid_mask)
    np.testing.assert_array_equal(restored.case_mask, batch.case_mask)
    np.testing.assert_array_equal(restored.soft_mask, batch.soft_mask)


def test_bounded_lowering_rejects_or_explicitly_reports_every_loss() -> None:
    rejecting = SequenceLoweringPlan(DNA_IUPAC, 1, 3)
    with pytest.raises(OverflowError):
        lower_sequences(["ACGT", "A"], rejecting)

    truncating = SequenceLoweringPlan(
        DNA_IUPAC,
        1,
        3,
        invalid_symbol_policy="unknown",
        overflow_policy="truncate",
    )
    batch, report = lower_sequences(["a%GT", "A"], truncating, record_ids=[9, 10])
    assert batch.token_codes.shape == (1, 3)
    assert decode_sequences(batch) == ("a?G",)
    assert bool(report.loss_occurred)
    assert int(report.record_overflow_count) == 1
    np.testing.assert_array_equal(report.original_lengths, [4, 1])
    np.testing.assert_array_equal(report.retained_lengths, [3, 0])
    np.testing.assert_array_equal(report.truncated_symbol_counts, [1, 1])
    np.testing.assert_array_equal(report.mapped_invalid_counts, [1, 0])
    np.testing.assert_array_equal(report.retained_mask, [True, False])


def test_quality_batch_has_explicit_encoding_and_rejects_shape_or_range_errors() -> None:
    qualities = QualityBatch(
        [1], [[0, 40, 0]], [[True, True, False]], [True], PhredEncoding.PHRED33
    )
    assert qualities.phred_encoding is PhredEncoding.PHRED33
    with pytest.raises(ValueError, match="shape"):
        QualityBatch([1], [[10]], [[True, False]], [True], "phred33")
    with pytest.raises(ValueError, match="between"):
        QualityBatch([1], [[94]], [[True]], [True], "phred33")
    with pytest.raises(ValueError, match="lengths must match"):
        FASTQRecord("read", "AC", "!", PhredEncoding.PHRED33)


def test_fasta_text_path_iterable_round_trip_and_no_quality_fabrication(
    tmp_path: Path,
) -> None:
    records = (
        FASTARecord("r1", "ACGT", "first"),
        FASTARecord("empty", ""),
    )
    text = write_fasta(records, line_width=2)
    assert isinstance(text, str)
    assert parse_fasta(text) == records
    assert parse_fasta(iter(text.splitlines(keepends=True))) == records
    path = tmp_path / "records.fasta"
    assert write_fasta(records, path) is None
    assert parse_fasta(path) == records

    plan = SequenceLoweringPlan(DNA_IUPAC, 2, 4)
    lowered = lower_fasta(records, plan)
    assert len(lowered) == 2
    assert decode_sequences(lowered[0]) == ("ACGT", "")
    assert not isinstance(lowered[0], QualityBatch)


def test_fastq_round_trip_requires_encoding_and_preserves_real_quality(
    tmp_path: Path,
) -> None:
    records = (
        FASTQRecord("r1", "AcN", "!I#", PhredEncoding.PHRED33, "read one"),
        FASTQRecord("empty", "", "", PhredEncoding.PHRED33),
    )
    text = write_fastq(records)
    assert isinstance(text, str)
    with pytest.raises(TypeError):
        parse_fastq(text)
    assert parse_fastq(text, phred_encoding="phred33") == records
    path = tmp_path / "records.fastq"
    assert write_fastq(records, path) is None
    assert parse_fastq(path, phred_encoding=PhredEncoding.PHRED33) == records

    plan = SequenceLoweringPlan(DNA_IUPAC, 2, 3)
    lowered = lower_fastq(records, plan)
    assert decode_sequences(lowered.sequences) == ("AcN", "")
    np.testing.assert_array_equal(lowered.qualities.phred_scores[0], [0, 40, 2])
    np.testing.assert_array_equal(
        lowered.qualities.valid_mask, lowered.sequences.valid_mask
    )


def test_translation_all_forward_frames_and_reverse_strand() -> None:
    batch = encode_sequences(["ATGAAATAG"], DNA_IUPAC)
    expected_forward = {0: "MK*", 1: "*N", 2: "EI"}
    expected_reverse = {0: "LFH", 1: "YF", 2: "IS"}
    for frame in (0, 1, 2):
        forward = translate(batch, _translation_plan(frame=frame, strand="forward"))
        reverse = translate(batch, _translation_plan(frame=frame, strand="reverse"))
        assert decode_sequences(forward.sequences) == (expected_forward[frame],)
        assert decode_sequences(reverse.sequences) == (expected_reverse[frame],)


def test_translation_ambiguity_incomplete_and_stop_policies_are_explicit() -> None:
    consensus = translate(
        encode_sequences(["ATHATN"], DNA_IUPAC),
        _translation_plan(ambiguous="consensus"),
    )
    unknown = translate(
        encode_sequences(["ATH"], DNA_IUPAC),
        _translation_plan(ambiguous="unknown"),
    )
    assert decode_sequences(consensus.sequences) == ("IX",)
    assert decode_sequences(unknown.sequences) == ("X",)
    assert int(consensus.report.ambiguous_codon_counts[0]) == 2
    with pytest.raises(ValueError, match="ambiguous"):
        translate(
            encode_sequences(["ATH"], DNA_IUPAC),
            _translation_plan(ambiguous="reject"),
        )

    incomplete = encode_sequences(["ATGA"], DNA_IUPAC)
    dropped = translate(incomplete, _translation_plan(incomplete="drop"))
    represented = translate(incomplete, _translation_plan(incomplete="unknown"))
    assert decode_sequences(dropped.sequences) == ("M",)
    assert decode_sequences(represented.sequences) == ("MX",)
    assert int(represented.report.incomplete_base_counts[0]) == 1
    with pytest.raises(ValueError, match="incomplete"):
        translate(incomplete, _translation_plan(incomplete="reject"))

    stopping = encode_sequences(["ATGTAAGGG"], DNA_IUPAC)
    kept = translate(stopping, _translation_plan(stop="keep"))
    truncated = translate(stopping, _translation_plan(stop="truncate"))
    assert decode_sequences(kept.sequences) == ("M*G",)
    assert decode_sequences(truncated.sequences) == ("M",)
    assert int(truncated.report.stop_codon_counts[0]) == 1
    with pytest.raises(ValueError, match="stop codon"):
        translate(stopping, _translation_plan(stop="reject"))


def test_translation_accepts_a_selectable_complete_genetic_code() -> None:
    alternative = dict(STANDARD_GENETIC_CODE.codon_table)
    alternative["TGA"] = "W"
    custom = GeneticCode("tga-is-tryptophan", tuple(alternative.items()))
    result = translate(
        encode_sequences(["TGA"], DNA_IUPAC),
        _translation_plan(genetic_code=custom),
    )
    assert decode_sequences(result.sequences) == ("W",)
