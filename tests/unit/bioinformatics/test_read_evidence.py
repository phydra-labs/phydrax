#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.bioinformatics.genomics._alignment_events import expand_alignment_events
from phydrax.bioinformatics.genomics._cigar import (
    cigar_batch_from_strings,
    cigar_batch_from_tuples,
    CIGAR_STATUS_OPERATION_CAPACITY_EXCEEDED,
    CigarOp,
)
from phydrax.bioinformatics.genomics._mapping import (
    candidate_mapping_evidence,
    MAPPING_STATUS_CANDIDATE_TRUNCATED,
    MAPPING_STATUS_MAPQ_UNCALIBRATED,
    MappingCandidateBatch,
    MappingExecutionPlan,
    reference_aware_pileup_likelihood,
)
from phydrax.bioinformatics.genomics._reads import (
    read_evidence_provenance,
    ReadLayout,
    SAM_FLAG_DUPLICATE,
    SAM_FLAG_FIRST_IN_PAIR,
    SAM_FLAG_PAIRED,
    SAM_FLAG_PROPER_PAIR,
    SAM_FLAG_REVERSE,
    SAM_FLAG_SECOND_IN_PAIR,
    SAM_FLAG_SECONDARY,
    SAM_FLAG_SUPPLEMENTARY,
    SAM_FLAG_UNMAPPED,
)
from phydrax.bioinformatics.interchange._hts import (
    mapping_candidates_from_sam_like,
    read_batch_from_sam_like,
    sam_like_from_pysam,
    sam_like_records_from_pysam,
    SAMLikeRecord,
)
from phydrax.bioinformatics.sequence import DNA_IUPAC, encode_sequences


def test_packed_cigar_covers_every_operation_clip_and_splice_semantic() -> None:
    cigar = cigar_batch_from_strings(["1M1I1D1N1S1H1P1=1X"], 9)
    assert bool(cigar.valid[0])
    np.testing.assert_array_equal(cigar.operations[0], np.arange(9))
    assert int(cigar.query_length[0]) == 5
    assert int(cigar.reference_length[0]) == 5
    assert int(cigar.aligned_base_count[0]) == 3

    events = expand_alignment_events(cigar, [100], [5], [False], 9)
    assert bool(events.valid[0])
    np.testing.assert_array_equal(events.operation[0], np.arange(9))
    np.testing.assert_array_equal(events.query_index[0], [0, 1, -1, -1, 2, -1, -1, 3, 4])
    np.testing.assert_array_equal(
        events.reference_position[0], [100, -1, 101, 102, -1, -1, -1, 103, 104]
    )
    assert int(events.reference_anchor_position[0, 1]) == 100

    reverse = expand_alignment_events(cigar, [100], [5], [True], 9)
    np.testing.assert_array_equal(reverse.read_cycle[0], [4, 3, -1, -1, 2, -1, -1, 1, 0])


def test_cigar_capacity_failure_is_observable_and_never_partially_usable() -> None:
    cigar = cigar_batch_from_tuples([[(CigarOp.MATCH, 1), (CigarOp.INSERTION, 1)]], 1)
    assert not bool(cigar.valid[0])
    assert int(cigar.status[0]) == CIGAR_STATUS_OPERATION_CAPACITY_EXCEEDED
    assert int(cigar.evidence[0]) == 2
    assert int(cigar.op_count[0]) == 0
    assert int(cigar.packed_ops[0, 0]) == 0

    malformed = cigar_batch_from_strings(["2Mgarbage"], 2)
    assert not bool(malformed.valid[0])


def _sam_records() -> tuple[SAMLikeRecord, ...]:
    first_flag = (
        SAM_FLAG_PAIRED
        | SAM_FLAG_PROPER_PAIR
        | SAM_FLAG_FIRST_IN_PAIR
        | SAM_FLAG_DUPLICATE
    )
    second_flag = (
        SAM_FLAG_PAIRED
        | SAM_FLAG_PROPER_PAIR
        | SAM_FLAG_SECOND_IN_PAIR
        | SAM_FLAG_REVERSE
    )
    return (
        SAMLikeRecord(
            "pair",
            "ACGT",
            (30, 30, 30, 30),
            first_flag,
            "chr1",
            10,
            42,
            ((0, 4),),
            "chr1",
            12,
            6,
            "rg1",
            "AC",
            "ref-v1",
        ),
        SAMLikeRecord(
            "pair",
            "TGCA",
            (25, 25, 25, 25),
            second_flag,
            "chr1",
            12,
            None,
            ((0, 4),),
            "chr1",
            10,
            -6,
            "rg1",
            None,
            "ref-v1",
        ),
        SAMLikeRecord(
            "secondary", "AA", (20, 20), SAM_FLAG_SECONDARY, "chr1", 20, 10, ((0, 2),)
        ),
        SAMLikeRecord(
            "supplementary",
            "CC",
            (20, 20),
            SAM_FLAG_SUPPLEMENTARY,
            "chr1",
            30,
            10,
            ((0, 2),),
        ),
        SAMLikeRecord("unmapped", "NN", None, SAM_FLAG_UNMAPPED, None, None, None, None),
    )


def test_read_lowering_preserves_flags_unknowns_overlap_groups_and_umi() -> None:
    reads = read_batch_from_sam_like(
        _sam_records(),
        ReadLayout(6, 4, 2, 2),
        reference_ids={"chr1": 7},
        read_group_ids={"rg1": 3},
    )
    assert bool(jnp.all(reads.valid))
    assert bool(reads.reverse_strand[1])
    assert bool(reads.secondary[2])
    assert bool(reads.supplementary[3])
    assert bool(reads.unmapped[4])
    assert not bool(reads.mapping_quality_known[1])
    assert not bool(jnp.any(reads.quality.valid_mask[4]))
    assert bool(reads.pair.overlapping_mate[0])
    assert bool(reads.pair.overlapping_mate[1])
    np.testing.assert_array_equal(reads.pair.overlap_start[:2], [12, 12])
    np.testing.assert_array_equal(reads.pair.overlap_end[:2], [14, 14])

    provenance = read_evidence_provenance(reads)
    assert bool(provenance.umi_duplicate[0])
    assert not bool(provenance.quality_known[4])
    leaves = jax.tree_util.tree_leaves(reads)
    assert all(leaf.dtype.kind not in "OUS" for leaf in leaves)


def test_reference_likelihood_exposes_mismatch_and_unknown_quality() -> None:
    read_record = SAMLikeRecord("r", "ACGT", None, 0, "chr1", 0, None, ((0, 4),))
    reads = read_batch_from_sam_like(
        [read_record], ReadLayout(1, 4, 2), reference_ids={"chr1": 100}
    )
    candidate_records = (
        (
            SAMLikeRecord("r", "ACGT", None, 0, "chr1", 0, None, ((7, 4),)),
            SAMLikeRecord("r", "ACGT", None, 0, "chr2", 0, None, ((0, 4),)),
        ),
    )
    candidates, candidate_cigar = mapping_candidates_from_sam_like(
        candidate_records,
        2,
        2,
        reference_ids={"chr1": 100, "chr2": 200},
    )
    events = expand_alignment_events(
        candidate_cigar,
        candidates.reference_start,
        jnp.full((1, 2), 4),
        candidates.reverse_strand,
        4,
    )
    references = encode_sequences(["ACGA", "ACGT"], DNA_IUPAC, record_ids=[100, 200])
    plan = MappingExecutionPlan(2, 4)
    pileup = reference_aware_pileup_likelihood(
        reads, candidates, events, references, plan
    )
    assert bool(pileup.valid[0, 0])
    assert bool(pileup.reference_mismatch[0, 0])
    assert int(pileup.mismatch_count[0, 0]) == 1
    np.testing.assert_allclose(
        pileup.candidate_log_likelihood[0, 1], 4.0 * math.log(0.25), rtol=1e-6
    )


def test_mapping_evidence_is_conditional_with_null_pair_score_and_mapq_status() -> None:
    reads = read_batch_from_sam_like(
        [SAMLikeRecord("r", "ACGT", (30, 30, 30, 30), 0, "chr1", 0, None, ((0, 4),))],
        ReadLayout(1, 4, 2),
        reference_ids={"chr1": 100},
    )
    candidate_records = (
        (
            SAMLikeRecord("r", "ACGT", None, 0, "chr1", 0, None, ((0, 4),)),
            SAMLikeRecord("r", "ACGT", None, 0, "chr2", 0, None, ((0, 4),)),
        ),
    )
    candidates, cigars = mapping_candidates_from_sam_like(
        candidate_records, 2, 2, reference_ids={"chr1": 100, "chr2": 200}
    )
    events = expand_alignment_events(
        cigars, candidates.reference_start, [[4, 4]], [[False, False]], 4
    )
    references = encode_sequences(["ACGA", "ACGT"], DNA_IUPAC, record_ids=[100, 200])
    plan = MappingExecutionPlan(2, 4)
    pileup = reference_aware_pileup_likelihood(
        reads, candidates, events, references, plan
    )
    evidence = candidate_mapping_evidence(
        candidates, pileup, [[10.0, 0.0]], [-50.0], [0.0], plan
    )
    assert int(evidence.best_candidate_index[0]) == 0
    np.testing.assert_allclose(
        jnp.sum(jnp.exp(evidence.conditional_candidate_log_probability[0])),
        1.0,
        rtol=1e-6,
    )
    assert bool(evidence.conditional_on_supplied_candidates[0])
    assert not bool(evidence.mapq_calibrated[0])
    assert not bool(evidence.mapping_quality_known[0])
    assert int(evidence.status[0]) == MAPPING_STATUS_MAPQ_UNCALIBRATED

    truncated = MappingCandidateBatch(
        candidates.reference_id,
        candidates.reference_start,
        candidates.reverse_strand,
        candidates.candidate_mask,
        candidates.candidate_log_prior,
        [True],
    )
    truncated_result = candidate_mapping_evidence(
        truncated, pileup, [[0.0, 0.0]], [0.0], [0.0], plan
    )
    assert int(truncated_result.status[0]) == MAPPING_STATUS_CANDIDATE_TRUNCATED
    assert not bool(truncated_result.mapq_calibrated[0])

    null_result = candidate_mapping_evidence(
        candidates, pileup, [[0.0, 0.0]], [100.0], [0.0], plan
    )
    assert not bool(null_result.mapped[0])
    assert int(null_result.best_candidate_index[0]) == -1


def test_candidate_adapter_makes_truncation_observable() -> None:
    records = tuple(
        SAMLikeRecord(f"r{index}", "A", (30,), 0, "chr1", index, 20, ((0, 1),))
        for index in range(3)
    )
    candidates, _ = mapping_candidates_from_sam_like(
        (records,), 2, 1, reference_ids={"chr1": 1}
    )
    assert bool(candidates.retrieval_truncated[0])
    assert int(candidates.candidate_count[0]) == 2


def test_pysam_adapter_normalizes_unknown_mapq_and_requires_cram_identity() -> None:
    class FakeRecord:
        query_name = "r"
        query_sequence = "A"
        query_qualities = None
        flag = 0
        reference_name = "chr1"
        reference_start = 0
        mapping_quality = 255
        cigartuples = ((0, 1),)
        next_reference_name = None
        next_reference_start = -1
        template_length = 0

        @staticmethod
        def has_tag(name: str) -> bool:
            return name == "RG"

        @staticmethod
        def get_tag(name: str) -> str:
            assert name == "RG"
            return "group"

    converted = sam_like_from_pysam(FakeRecord())
    assert converted.mapping_quality is None
    assert converted.read_group == "group"

    class FakeCRAM:
        is_cram = True

        @staticmethod
        def fetch(*, until_eof: bool):
            assert until_eof
            return (FakeRecord(),)

    with pytest.raises(ValueError, match="reference_identity"):
        sam_like_records_from_pysam(FakeCRAM())
    records = sam_like_records_from_pysam(FakeCRAM(), reference_identity="ref-v1")
    assert records[0].reference_identity == "ref-v1"
