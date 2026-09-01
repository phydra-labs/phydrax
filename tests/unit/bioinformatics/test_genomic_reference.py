#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
import pytest

from phydrax.bioinformatics.foundation import FeatureDictionary
from phydrax.bioinformatics.genomics._annotations import (
    AnnotationStatus,
    audit_feature_parents,
    FeatureParentRelation,
    GenomicAnnotation,
    query_feature_parents,
    query_overlapping_features,
)
from phydrax.bioinformatics.genomics._coordinates import (
    CoordinateStatus,
    GraphCoordinate,
    interval_contains,
    interval_difference,
    interval_intersection,
    interval_union,
    IntervalSet,
    LinearCoordinate,
    LinearInterval,
    merge_interval_set,
    PhaseCoordinate,
    Strand,
)
from phydrax.bioinformatics.genomics._reference import (
    lower_global_coordinates,
    reference_digest,
    ReferenceContig,
    ReferenceDictionary,
    ReferenceGenome,
    ReferenceStatus,
    ReferenceWindow,
)
from phydrax.bioinformatics.genomics._transcripts import (
    assemble_cds,
    CDSModel,
    genomic_to_transcript,
    liftover_interval_to_transcript,
    liftover_transcript_coordinates,
    splice_transcript,
    transcript_to_genomic,
    TranscriptModel,
    TranscriptStatus,
    translate_cds,
)
from phydrax.bioinformatics.interchange._annotations import (
    BEDFeatureLine,
    gff3_parent_relations,
    GFF3FeatureLine,
    GTFFeatureLine,
    parse_bed,
    parse_gff3,
    parse_gtf,
    write_bed,
    write_gff3,
    write_gtf,
)
from phydrax.bioinformatics.sequence import DNA_IUPAC, SequenceBatch


def _decode(batch: SequenceBatch) -> str:
    length = int(np.asarray(batch.lengths[0]))
    return "".join(
        batch.alphabet.symbols[int(code)]
        for code in np.asarray(batch.token_codes[0, :length])
    )


def _reference(sequence: str, *, circular: bool = False) -> ReferenceGenome:
    return ReferenceGenome.from_sequences(
        {"chr1": sequence},
        assembly_id="test",
        aliases={"chr1": ("1",)},
        circular=("chr1",) if circular else (),
    )


def test_reference_digests_dictionary_aliases_and_checked_circular_window() -> None:
    digest = reference_digest("ac gt\n")
    assert digest.matches("ACGT")
    assert digest.refget_id.startswith("SQ.")
    contig = ReferenceContig("chr1", 4, digest, aliases=("1",))
    dictionary = ReferenceDictionary((contig,), assembly_id="asm")
    assert dictionary.resolve("1") == 0
    assert dictionary.contig(0).digest == digest
    assert len(dictionary.digest) == 64
    with pytest.raises(ValueError, match="not unique"):
        ReferenceDictionary(
            (
                contig,
                ReferenceContig("chr2", 4, digest, aliases=("1",)),
            )
        )

    genome = _reference("ACGTACGTAA", circular=True)
    result = genome.fetch_window("1", 8, 13, capacity=5)
    assert bool(np.asarray(result.valid))
    assert bool(np.asarray(result.window.wrapped))
    assert _decode(result.window.sequence) == "AAACG"
    lowered = lower_global_coordinates(
        result.window, np.asarray([8, 9, 0, 2, 3], dtype=np.int64)
    )
    np.testing.assert_array_equal(lowered.relative_positions, [0, 1, 2, 4, 0])
    np.testing.assert_array_equal(lowered.valid, [True, True, True, True, False])

    overflow = genome.fetch_window(0, 0, 6, capacity=5)
    assert not bool(np.asarray(overflow.valid))
    assert int(np.asarray(overflow.status)) == int(ReferenceStatus.CAPACITY_EXCEEDED)
    assert int(np.asarray(overflow.window.sequence.lengths[0])) == 0


def test_host_int64_coordinates_lower_to_window_relative_int32() -> None:
    pad = DNA_IUPAC.code(DNA_IUPAC.pad_symbol)
    sequence = SequenceBatch(
        np.asarray([0], dtype=np.int32),
        np.asarray([[DNA_IUPAC.code("A"), DNA_IUPAC.code("C"), pad]], dtype=np.int32),
        np.asarray([[True, True, False]]),
        np.asarray([True]),
        np.zeros((1, 3), dtype=bool),
        DNA_IUPAC,
    )
    window = ReferenceWindow(
        sequence,
        0,
        np.int64(3_000_000_000),
        np.int64(3_000_000_002),
        np.int64(3_000_000_000),
        np.int64(4_000_000_000),
        False,
        False,
    )
    lowered = lower_global_coordinates(
        window, np.asarray([3_000_000_000, 3_000_000_001, 3_000_000_002], dtype=np.int64)
    )
    assert lowered.relative_positions.dtype == np.dtype("int32")
    np.testing.assert_array_equal(lowered.relative_positions, [0, 1, 0])
    np.testing.assert_array_equal(lowered.valid, [True, True, False])


def test_typed_coordinates_and_zero_length_half_open_interval_algebra() -> None:
    interval = LinearInterval(0, 5, 10, strand=Strand.FORWARD)
    assert bool(np.asarray(interval_contains(interval, LinearCoordinate(0, 5))))
    assert not bool(np.asarray(interval_contains(interval, LinearCoordinate(0, 10))))
    empty = LinearInterval(0, 7, 7, strand=Strand.FORWARD)
    intersection = interval_intersection(interval, empty)
    assert not bool(np.asarray(intersection.valid))
    assert int(np.asarray(intersection.status)) == int(CoordinateStatus.DISJOINT)

    difference = interval_difference(
        interval, LinearInterval(0, 7, 8, strand=Strand.FORWARD)
    )
    np.testing.assert_array_equal(difference.intervals.valid, [True, True])
    np.testing.assert_array_equal(difference.intervals.starts, [5, 8])
    np.testing.assert_array_equal(difference.intervals.ends, [7, 10])
    union = interval_union(interval, LinearInterval(0, 12, 14, strand=Strand.FORWARD))
    np.testing.assert_array_equal(union.intervals.valid, [True, True])
    PhaseCoordinate(3, 2)
    GraphCoordinate(0, 1, 2, 3, orientation=Strand.REVERSE)
    with pytest.raises(ValueError, match="phase"):
        PhaseCoordinate(0, 3)

    intervals = IntervalSet(
        [0, 0, 0],
        [1, 1, 5],
        [4, 4, 7],
        [1, 1, 1],
        [True, True, True],
    )
    merged = merge_interval_set(intervals)
    np.testing.assert_array_equal(merged.intervals.valid, [True, True, False])
    np.testing.assert_array_equal(merged.intervals.starts[:2], [1, 5])


def test_gff3_gtf_and_bed_roundtrip_unknown_host_content_losslessly() -> None:
    gff = (
        "##gff-version 3\r\n"
        "##unknown-directive opaque value\r\n"
        "chr1\tsrc\tmRNA\t1\t9\t.\t+\t.\tID=tx%201;Unknown=a%2Cb;Flag\r\n"
        "# trailing comment"
    )
    gff_records = parse_gff3(gff)
    assert write_gff3(gff_records) == gff
    feature = next(
        record for record in gff_records if isinstance(record, GFF3FeatureLine)
    )
    assert (feature.start, feature.end) == (0, 9)
    assert feature.attribute_values("Unknown") == ("a,b",)
    assert feature.attribute_values("Flag") == ()

    gtf = 'chr1\tsrc\texon\t2\t7\t0.25\t-\t2\tgene_id "g1"; note "a; b \\"quoted\\"";\n'
    gtf_records = parse_gtf(gtf)
    assert write_gtf(gtf_records) == gtf
    gtf_feature = next(
        record for record in gtf_records if isinstance(record, GTFFeatureLine)
    )
    assert (gtf_feature.start, gtf_feature.end, gtf_feature.frame) == (1, 7, 2)
    assert gtf_feature.attribute_values("note") == ('a; b "quoted"',)

    bed = (
        "track name=opaque\n"
        "chr1\t5\t5\tzero\n"
        "chr1\t10\t30\tblocks\t0\t-\t10\t30\t0\t2\t5,10,\t0,10,\n"
    )
    bed_records = parse_bed(bed)
    assert write_bed(bed_records) == bed
    features = [record for record in bed_records if isinstance(record, BEDFeatureLine)]
    assert features[0].length == 0
    assert features[1].block_intervals == ((10, 15), (20, 30))


def test_gff3_parent_relations_preserve_duplicates_one_to_many_and_loss() -> None:
    text = (
        "chr1\ts\tgene\t1\t5\t.\t+\t.\tID=p\n"
        "chr1\ts\tgene\t6\t9\t.\t+\t.\tID=p\n"
        "chr1\ts\tmRNA\t1\t9\t.\t+\t.\tID=c;Parent=p,p,missing\n"
    )
    relation = gff3_parent_relations(parse_gff3(text))
    assert relation.duplicate_identifiers == ("p",)
    assert relation.child_rows == (2, 2, 2, 2)
    assert relation.parent_rows == (0, 1, 0, 1)
    assert relation.unresolved == ((2, "missing"),)
    assert relation.ambiguous
    assert not relation.lossless


def test_feature_queries_and_parent_relation_audit_capacity() -> None:
    features = FeatureDictionary(
        [10, 11, 12],
        namespace="test",
        version="1",
        species="human",
        reference="ref",
        annotation="ann",
    )
    parents = FeatureParentRelation(
        [1, 1, 2],
        [0, 0, 1],
        feature_count=3,
    )
    annotation = GenomicAnnotation(
        features,
        [0, 0, 0],
        [0, 2, 7],
        [10, 5, 9],
        [0, 1, -1],
        [0, 1, 1],
        [0, 0, 0],
        np.asarray([np.nan, 1.0, 2.0]),
        [-1, 0, 2],
        [True, True, True],
        parents,
    )
    overflow = query_overlapping_features(annotation, LinearInterval(0, 1, 8), capacity=2)
    assert not bool(np.asarray(overflow.valid))
    assert int(np.asarray(overflow.status)) == int(AnnotationStatus.CAPACITY_EXCEEDED)
    assert not bool(np.any(np.asarray(overflow.row_valid)))
    parent_rows = query_feature_parents(annotation, 1, capacity=2)
    np.testing.assert_array_equal(parent_rows.rows, [0, 0])
    np.testing.assert_array_equal(parent_rows.row_valid, [True, True])
    audit = audit_feature_parents(parents)
    assert int(np.asarray(audit.duplicate_edge_count)) == 1
    assert not bool(np.asarray(audit.cyclic))


def test_forward_reverse_duplicate_and_circular_transcript_splicing() -> None:
    genome = _reference("AAAACCCCGGGGTTTT")
    forward = TranscriptModel(1, 0, [0, 8], [4, 12], [True, True], strand=Strand.FORWARD)
    forward_result = splice_transcript(genome, forward, capacity=8)
    assert _decode(forward_result.sequence) == "AAAAGGGG"
    np.testing.assert_array_equal(forward_result.exon_transcript_starts, [0, 4])

    reverse = TranscriptModel(2, 0, [0, 10], [4, 14], [True, True], strand=Strand.REVERSE)
    reverse_result = splice_transcript(genome, reverse, capacity=8)
    assert _decode(reverse_result.sequence) == "AACCTTTT"
    mapped = genomic_to_transcript(reverse, [13, 0])
    np.testing.assert_array_equal(np.asarray(mapped.target_positions)[0], [0, 0])
    assert 0 in np.asarray(mapped.target_positions)[0]
    assert 7 in np.asarray(mapped.target_positions)[1]
    back = transcript_to_genomic(reverse, [0, 7])
    assert 13 in np.asarray(back.target_positions)[0]
    assert 0 in np.asarray(back.target_positions)[1]

    duplicate = TranscriptModel(
        3,
        0,
        [0, 0],
        [4, 4],
        [True, True],
        strand=Strand.FORWARD,
        exon_order=[0, 1],
    )
    duplicate_map = genomic_to_transcript(duplicate, [1])
    assert bool(np.asarray(duplicate_map.ambiguous[0]))
    np.testing.assert_array_equal(duplicate_map.target_positions[0], [1, 5])
    failed = splice_transcript(genome, duplicate, capacity=7)
    assert int(np.asarray(failed.status)) == int(TranscriptStatus.CAPACITY_EXCEEDED)
    assert _decode(failed.sequence) == ""

    circular = _reference("ACGTACGTAA", circular=True)
    circular_model = TranscriptModel(
        4,
        0,
        [8],
        [13],
        [True],
        strand=Strand.FORWARD,
        exon_order=[0],
        reference_length=10,
        circular=True,
    )
    assert (
        _decode(splice_transcript(circular, circular_model, capacity=5).sequence)
        == "AAACG"
    )
    circular_mapping = genomic_to_transcript(circular_model, [8, 1])
    np.testing.assert_array_equal(circular_mapping.target_positions[:, 0], [0, 3])
    circular_back = transcript_to_genomic(circular_model, [0, 3])
    np.testing.assert_array_equal(circular_back.target_positions[:, 0], [8, 1])

    zero_exon = TranscriptModel(5, 0, [4], [4], [True], strand=Strand.FORWARD)
    zero_result = splice_transcript(genome, zero_exon, capacity=0)
    assert bool(np.asarray(zero_result.valid))
    assert _decode(zero_result.sequence) == ""


def test_phased_multi_exon_cds_translation_and_phase_failure() -> None:
    genome = _reference("ATGACCCCGAAAATTT")
    transcript = TranscriptModel(
        7, 0, [0, 8], [4, 13], [True, True], strand=Strand.FORWARD
    )
    cds = CDSModel(
        transcript,
        [0, 8],
        [4, 13],
        [True, True],
        [0, 2],
    )
    assembly = assemble_cds(genome, cds, capacity=9)
    assert bool(np.asarray(assembly.valid))
    assert bool(np.asarray(assembly.phase_consistent))
    assert _decode(assembly.sequence) == "ATGAGAAAA"
    protein = translate_cds(assembly)
    assert bool(np.asarray(protein.valid))
    assert _decode(protein.translation.sequences) == "MRK"

    reverse_genome = _reference("AAAACCCCGGGGTTTT")
    reverse_transcript = TranscriptModel(
        9,
        0,
        [0, 10],
        [5, 14],
        [True, True],
        strand=Strand.REVERSE,
    )
    reverse_cds = CDSModel(
        reverse_transcript,
        [0, 10],
        [5, 14],
        [True, True],
        [2, 0],
    )
    reverse_assembly = assemble_cds(reverse_genome, reverse_cds, capacity=9)
    assert bool(np.asarray(reverse_assembly.phase_consistent))
    assert _decode(reverse_assembly.sequence) == "AACCGTTTT"
    assert _decode(translate_cds(reverse_assembly).translation.sequences) == "NRF"

    inconsistent = CDSModel(
        transcript,
        [0, 8],
        [4, 13],
        [True, True],
        [0, 1],
    )
    failed = assemble_cds(genome, inconsistent, capacity=9)
    assert not bool(np.asarray(failed.valid))
    assert int(np.asarray(failed.status)) == int(TranscriptStatus.PHASE_INCONSISTENT)

    ambiguous_genome = _reference("ATN")
    one_exon = TranscriptModel(8, 0, [0], [3], [True], strand=Strand.FORWARD)
    ambiguous_cds = CDSModel(one_exon, [0], [3], [True], [0])
    ambiguous = translate_cds(assemble_cds(ambiguous_genome, ambiguous_cds, capacity=3))
    assert not bool(np.asarray(ambiguous.exact))
    assert int(np.asarray(ambiguous.status)) == int(TranscriptStatus.AMBIGUOUS)
    assert _decode(ambiguous.translation.sequences) == "X"


def test_interval_and_cross_transcript_liftover_exposes_loss_ambiguity_and_capacity() -> (
    None
):
    overlapping = TranscriptModel(
        1,
        0,
        [0, 3],
        [5, 8],
        [True, True],
        strand=Strand.FORWARD,
        exon_order=[0, 1],
    )
    lifted = liftover_interval_to_transcript(overlapping, LinearInterval(0, 2, 6))
    assert int(np.asarray(lifted.ambiguous_bases)) == 2
    assert int(np.asarray(lifted.lost_bases)) == 0
    np.testing.assert_array_equal(lifted.fragment_valid, [True, True])
    empty_lift = liftover_interval_to_transcript(overlapping, LinearInterval(0, 4, 4))
    assert bool(np.asarray(empty_lift.valid))
    assert int(np.asarray(empty_lift.status)) == int(TranscriptStatus.SUCCESS)
    assert not bool(np.any(np.asarray(empty_lift.fragment_valid)))

    partial = TranscriptModel(
        2,
        0,
        [0, 7],
        [3, 9],
        [True, True],
        strand=Strand.FORWARD,
    )
    partial_result = liftover_interval_to_transcript(partial, LinearInterval(0, 0, 10))
    assert int(np.asarray(partial_result.lost_bases)) == 5
    assert int(np.asarray(partial_result.status)) == int(TranscriptStatus.PARTIAL_MAPPING)

    source = TranscriptModel(3, 0, [0], [4], [True], strand=Strand.FORWARD)
    target = TranscriptModel(
        4,
        0,
        [0, 0],
        [4, 4],
        [True, True],
        strand=Strand.FORWARD,
        exon_order=[0, 1],
    )
    one_to_many = liftover_transcript_coordinates(source, target, [1], capacity=2)
    assert bool(np.asarray(one_to_many.ambiguous[0]))
    np.testing.assert_array_equal(one_to_many.target_positions[0], [1, 5])
    overflow = liftover_transcript_coordinates(source, target, [1], capacity=1)
    assert not bool(np.asarray(overflow.valid[0]))
    assert int(np.asarray(overflow.status[0])) == int(TranscriptStatus.CAPACITY_EXCEEDED)
    assert not bool(np.any(np.asarray(overflow.route_valid)))
