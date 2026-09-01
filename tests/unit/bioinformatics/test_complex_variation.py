#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.bioinformatics.genomics._copy_number import (
    CopyNumberObservations,
    CopyNumberReferencePlan,
    CopyNumberSegmentationPlan,
    CopyNumberStatus,
    segment_copy_number,
)
from phydrax.bioinformatics.genomics._somatic import (
    somatic_likelihoods,
    SomaticCopyContext,
    SomaticLikelihoodPlan,
    SomaticPanelProvenance,
    SomaticStatus,
    TumorNormalAlleleCounts,
)
from phydrax.bioinformatics.genomics._structural_variants import (
    BreakendGraph,
    BreakendOrientation,
    BreakpointEvidence,
    CandidateLimitation,
    evaluate_breakend_graph,
    EventLinkKind,
    structural_variant_candidate_evidence,
    StructuralVariantCandidatePlan,
    StructuralVariantStatus,
)
from phydrax.bioinformatics.sequence import DNA_IUPAC, SequenceBatch


def _inserted_sequences(count: int) -> SequenceBatch:
    a = DNA_IUPAC.code("A")
    pad = DNA_IUPAC.code(DNA_IUPAC.pad_symbol)
    return SequenceBatch(
        jnp.arange(count, dtype=jnp.int64),
        jnp.asarray([[a, pad]] * count, dtype=jnp.int32),
        jnp.asarray([[True, False]] * count),
        jnp.ones((count,), dtype=bool),
        jnp.zeros((count, 2), dtype=bool),
        DNA_IUPAC,
    )


def _breakpoint_evidence(count: int) -> BreakpointEvidence:
    return BreakpointEvidence(
        jnp.full((count,), 4.0),
        jnp.full((count,), 2.0),
        jnp.full((count,), 1.0),
        jnp.full((count,), 0.7),
        jnp.full((count,), 55.0),
        jnp.full((count,), 0.5),
        jnp.full((count,), 0.2),
    )


def test_breakend_graph_preserves_reciprocal_unpaired_imprecise_and_complex_links():
    graph = BreakendGraph(
        [0, 1, 1],
        [100, 900, 1_500],
        [BreakendOrientation.RIGHT, BreakendOrientation.LEFT, BreakendOrientation.RIGHT],
        [1, 0, -1],
        [0, 0, 1],
        [100, 900, 1_490],
        [100, 900, 1_510],
        _inserted_sequences(3),
        [[1, 2]],
        [EventLinkKind.ORDERED_CHAIN],
    )
    plan = StructuralVariantCandidatePlan(
        maximum_breakends=8,
        regions_exhaustive=True,
    )
    candidates = structural_variant_candidate_evidence(plan, 3)
    result = evaluate_breakend_graph(graph, _breakpoint_evidence(3), candidates)

    assert bool(result.valid)
    assert int(result.status) == StructuralVariantStatus.SUCCESS
    assert result.evidence.reciprocal_breakends.tolist() == [True, True, False]
    assert result.evidence.unpaired_breakends.tolist() == [False, False, True]
    assert result.evidence.imprecise_breakends.tolist() == [False, False, True]
    assert result.evidence.linked_breakends.tolist() == [False, True, True]
    assert result.graph.inserted_sequence.valid_mask[:, 0].tolist() == [True, True, True]


def test_nonreciprocal_mate_is_observable_and_candidate_limitations_are_explicit():
    graph = BreakendGraph(
        [0, 0],
        [10, 20],
        [1, -1],
        [1, -1],
        [0, 0],
        [10, 20],
        [10, 20],
        _inserted_sequences(2),
        jnp.zeros((0, 2), dtype=jnp.int32),
        jnp.zeros((0,), dtype=jnp.int8),
    )
    plan = StructuralVariantCandidatePlan(
        maximum_breakends=2,
        intercontig_search=False,
        retain_unpaired=False,
        assembly_search=False,
        regions_exhaustive=False,
        precomputed_candidates=True,
    )
    candidates = structural_variant_candidate_evidence(plan, 2)
    result = evaluate_breakend_graph(graph, _breakpoint_evidence(2), candidates)

    assert not bool(result.valid)
    assert int(result.status) == StructuralVariantStatus.INVALID_MATE
    mask = int(candidates.limitation_mask)
    assert mask & CandidateLimitation.REGIONS_NOT_EXHAUSTIVE
    assert mask & CandidateLimitation.INTERCONTIG_DISABLED
    assert mask & CandidateLimitation.UNPAIRED_DROPPED
    assert mask & CandidateLimitation.ASSEMBLY_DISABLED
    assert mask & CandidateLimitation.PRECOMPUTED_CANDIDATES


def test_structural_variant_capacity_failure_never_silently_truncates():
    candidates = structural_variant_candidate_evidence(
        StructuralVariantCandidatePlan(maximum_breakends=2),
        3,
    )
    assert not bool(candidates.capacity_sufficient)
    assert int(candidates.generated_breakends) == 0
    assert int(candidates.dropped_breakends) == 3
    assert int(candidates.limitation_mask) & CandidateLimitation.CAPACITY_EXCEEDED


def test_reference_baseline_handles_haploid_sex_contigs_and_par():
    reference = CopyNumberReferencePlan(
        2.0,
        [2.0, 1.0],
        par_contig_index=[1],
        par_start=[100],
        par_end=[200],
        par_baseline_copy=[2.0],
    )
    baseline = reference.expected_copy([0, 1, 1], [50, 50, 150])
    assert baseline.tolist() == [2.0, 1.0, 2.0]


def test_finite_state_copy_number_segmentation_resolves_loh_with_noisy_depth_and_baf():
    reference = CopyNumberReferencePlan(2.0, [2.0, 1.0])
    observations = CopyNumberObservations(
        [0, 0, 0, 1],
        [0, 100, 200, 0],
        [100, 200, 300, 100],
        [1.04, 0.98, 0.51, 1.03],
        [0.08, 0.08, 0.07, 0.08],
        [0.48, 0.46, 0.03, 0.04],
        [0.06, 0.07, 0.05, 0.06],
        [True, True, True, True],
    )
    result = segment_copy_number(
        observations,
        reference,
        CopyNumberSegmentationPlan(
            maximum_bins=8,
            maximum_total_copy=4,
            transition_penalty=0.2,
        ),
    )

    assert bool(result.valid)
    assert int(result.status) == CopyNumberStatus.SUCCESS
    assert result.state.total_copy.tolist() == [2, 2, 1, 1]
    assert result.state.minor_copy.tolist() == [1, 1, 0, 0]
    assert result.state.loss_of_heterozygosity.tolist() == [False, False, True, True]
    assert result.segment_index.tolist() == [0, 0, 1, 2]
    assert jnp.allclose(jnp.sum(result.posterior_probability, axis=-1), 1.0)
    assert bool(jnp.all(jnp.isfinite(result.evidence.posterior_entropy)))
    assert bool(result.evidence.posterior_exact)
    assert bool(result.evidence.candidates.unmodeled_above_bound)
    assert not bool(result.evidence.candidates.interval_generation_performed)
    assert bool(result.evidence.candidates.intervals_precomputed)
    assert not bool(result.evidence.candidates.interval_search_exhaustive)


def test_copy_number_missing_baf_and_capacity_failure_are_observable():
    observations = CopyNumberObservations(
        [0, 0],
        [0, 10],
        [10, 20],
        [1.0, 0.5],
        [0.2, 0.2],
        [jnp.nan, jnp.nan],
        [jnp.nan, jnp.nan],
        [False, False],
    )
    result = segment_copy_number(
        observations,
        CopyNumberReferencePlan(2.0, [2.0]),
        CopyNumberSegmentationPlan(maximum_bins=1, maximum_total_copy=2),
    )
    assert not bool(result.valid)
    assert int(result.status) == CopyNumberStatus.CAPACITY_EXCEEDED
    assert not bool(result.evidence.candidates.capacity_sufficient)
    assert result.posterior_probability.shape[0] == 2


def _somatic_inputs(matched_normal):
    counts = TumorNormalAlleleCounts(
        [0, 10, 10],
        [30, 20, 20],
        [0, 2, 0],
        [30, 30, 0],
        matched_normal,
    )
    context = SomaticCopyContext(
        [0.0, 1.0, 1.0],
        [0.0, 0.2, 0.0],
        [2.0, 2.0, 2.0],
        [2.0, 2.0, 2.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.5],
    )
    panel = SomaticPanelProvenance(
        "panel-fingerprint",
        40,
        [True, True, False],
        [0.01, 0.01, 0.0],
    )
    return counts, context, panel


def test_somatic_likelihoods_cover_purity_zero_one_and_normal_contamination():
    counts, context, panel = _somatic_inputs([True, True, False])
    result = somatic_likelihoods(
        counts,
        context,
        panel,
        SomaticLikelihoodPlan(maximum_candidates=3),
    )

    error = 1e-3
    assert jnp.isclose(result.evidence.expected_tumor_alt_fraction[0, 0], error)
    assert jnp.isclose(result.evidence.expected_tumor_alt_fraction[1, 0], 0.5)
    assert result.evidence.expected_normal_alt_fraction[1, 0] > error
    assert bool(result.valid[0]) and bool(result.valid[1])
    assert int(result.status[0]) == SomaticStatus.SUCCESS
    assert int(result.status[2]) == SomaticStatus.NO_MATCHED_NORMAL
    assert bool(result.evidence.subclonal_context_used[2])
    assert len(result.panel_provenance.panel_id) == 64


def test_somatic_without_matched_normal_is_valid_but_never_claims_normal_evidence():
    counts, context, panel = _somatic_inputs([False, False, False])
    result = somatic_likelihoods(
        counts,
        context,
        panel,
        SomaticLikelihoodPlan(maximum_candidates=3),
    )

    assert bool(jnp.all(result.valid))
    assert result.status.tolist() == [
        SomaticStatus.NO_MATCHED_NORMAL,
        SomaticStatus.NO_MATCHED_NORMAL,
        SomaticStatus.NO_MATCHED_NORMAL,
    ]
    assert not bool(jnp.any(result.evidence.normal_likelihood_used))
    assert result.posterior_probability.shape == (3, 3)
    assert bool(jnp.allclose(jnp.sum(result.posterior_probability, axis=-1), 1.0))


def test_somatic_candidate_capacity_failure_scores_nothing():
    counts, context, panel = _somatic_inputs([True, True, False])
    result = somatic_likelihoods(
        counts,
        context,
        panel,
        SomaticLikelihoodPlan(maximum_candidates=2),
    )
    assert not bool(jnp.any(result.valid))
    assert result.status.tolist() == [
        SomaticStatus.CAPACITY_EXCEEDED,
        SomaticStatus.CAPACITY_EXCEEDED,
        SomaticStatus.CAPACITY_EXCEEDED,
    ]
    assert int(result.evidence.candidate_generation.candidates_scored) == 0
