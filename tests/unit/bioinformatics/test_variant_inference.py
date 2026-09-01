from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.bioinformatics.genomics._genotyping import (
    allele_frequency_genotype_prior,
    enumerate_genotype_states,
    genotype_likelihoods_from_gl,
    genotype_likelihoods_from_pl,
    genotype_likelihoods_from_reads,
    genotype_likelihoods_to_gl,
    genotype_likelihoods_to_pl,
    GenotypingStatus,
    infer_genotype,
    local_allele_evidence_from_calls,
    uniform_genotype_prior,
)
from phydrax.bioinformatics.genomics._phasing import (
    PedigreePhaseEvidence,
    phase_small_variants,
    PhasingStatus,
    read_phase_evidence_from_calls,
)
from phydrax.bioinformatics.genomics._variants import (
    decode_variant_alleles,
    normalize_small_variant,
    VariantNormalizationStatus,
)
from phydrax.bioinformatics.interchange._variants import (
    parse_vcf,
    VariantInterchangeStatus,
    vcf_record_to_small_variant,
    vcf_sample_likelihoods,
    write_vcf,
)


@pytest.mark.parametrize(
    ("alleles", "ploidy", "expected"),
    ((2, 1, 2), (2, 2, 3), (3, 3, 10)),
)
def test_complete_genotype_enumeration_for_bounded_ploidies(
    alleles: int, ploidy: int, expected: int
) -> None:
    states = enumerate_genotype_states(alleles, ploidy, expected)
    assert bool(states.valid)
    assert int(states.state_count) == expected
    populated = np.asarray(states.states)[np.asarray(states.state_mask)]
    assert len({tuple(row) for row in populated}) == expected
    assert np.all(populated[:, 1:] >= populated[:, :-1]) if ploidy > 1 else True


def test_multiallelic_vcf_genotype_order_and_capacity_failure() -> None:
    diploid = enumerate_genotype_states(3, 2, 6)
    np.testing.assert_array_equal(
        np.asarray(diploid.states)[:6],
        np.asarray(((0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2))),
    )
    overflow = enumerate_genotype_states(3, 3, 9)
    assert not bool(overflow.valid)
    assert int(overflow.status) == GenotypingStatus.CAPACITY_EXCEEDED
    assert int(overflow.evidence.required_state_count) == 10
    assert not np.asarray(overflow.state_mask).any()


def test_gl_pl_conversion_uses_separate_natural_log_likelihoods() -> None:
    states = enumerate_genotype_states(2, 2, 4)
    from_gl = genotype_likelihoods_from_gl((0.0, -1.0, -2.0), states, depth=7)
    from_pl = genotype_likelihoods_from_pl((0, 10, 20), states, depth=7)
    np.testing.assert_allclose(
        np.asarray(from_gl.log_likelihoods)[:3],
        np.asarray(from_pl.log_likelihoods)[:3],
        rtol=1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(genotype_likelihoods_to_gl(from_pl))[:3],
        np.asarray((0.0, -1.0, -2.0)),
        atol=1.0e-6,
    )
    np.testing.assert_array_equal(
        np.asarray(genotype_likelihoods_to_pl(from_gl))[:3],
        np.asarray((0, 10, 20)),
    )
    assert math.isnan(float(genotype_likelihoods_to_gl(from_gl)[3]))


def test_no_coverage_preserves_posterior_but_forces_no_call() -> None:
    states = enumerate_genotype_states(2, 2, 3)
    evidence = local_allele_evidence_from_calls(
        jnp.asarray((0, 0), dtype=jnp.int32),
        jnp.asarray((30.0, 30.0)),
        jnp.asarray((60.0, 60.0)),
        allele_count=2,
        read_mask=jnp.asarray((False, False)),
    )
    likelihoods = genotype_likelihoods_from_reads(evidence, states)
    result = infer_genotype(likelihoods, uniform_genotype_prior(states), states)
    assert bool(result.valid)
    assert bool(result.hard_call.no_call)
    assert not bool(result.hard_call.called)
    assert int(result.status) == GenotypingStatus.NO_COVERAGE
    np.testing.assert_allclose(
        np.asarray(result.posterior.probabilities)[:3], np.full((3,), 1.0 / 3.0)
    )
    np.testing.assert_array_equal(np.asarray(result.hard_call.alleles), (-1, -1))


def test_calibrated_quality_resolves_contradictory_haploid_reads() -> None:
    states = enumerate_genotype_states(2, 1, 2)
    evidence = local_allele_evidence_from_calls(
        jnp.asarray((1, 0), dtype=jnp.int32),
        jnp.asarray((55.0, 4.0)),
        jnp.asarray((60.0, 60.0)),
        allele_count=2,
    )
    result = infer_genotype(
        genotype_likelihoods_from_reads(evidence, states),
        uniform_genotype_prior(states),
        states,
        min_posterior=0.8,
    )
    assert bool(result.hard_call.called)
    np.testing.assert_array_equal(np.asarray(result.hard_call.alleles), (1,))
    assert float(result.hard_call.genotype_quality) > 10.0
    assert float(result.posterior.dosage[1]) > 0.99


def test_triploid_multiallelic_prior_and_dosage_are_not_hard_calls() -> None:
    states = enumerate_genotype_states(3, 3, 10)
    prior = allele_frequency_genotype_prior(jnp.asarray((0.5, 0.3, 0.2)), states)
    likelihoods = genotype_likelihoods_from_gl(
        jnp.asarray((-9.0, -8.0, -7.0, -6.0, -5.0, 0.0, -4.0, -3.0, -2.0, -1.0)),
        states,
        depth=12,
    )
    result = infer_genotype(likelihoods, prior, states, min_posterior=0.0)
    assert bool(result.valid)
    assert np.isclose(float(jnp.sum(result.posterior.probabilities)), 1.0)
    assert np.isclose(float(jnp.sum(result.posterior.dosage)), 3.0)
    assert result.posterior.dosage.shape == (3,)
    assert result.hard_call.alleles.shape == (3,)


def test_candidate_omission_is_observable_and_never_hard_called() -> None:
    states = enumerate_genotype_states(2, 2, 3)
    evidence = local_allele_evidence_from_calls(
        jnp.asarray((0, 2), dtype=jnp.int32),
        jnp.asarray((40.0, 40.0)),
        jnp.asarray((60.0, 60.0)),
        allele_count=2,
    )
    likelihoods = genotype_likelihoods_from_reads(evidence, states)
    result = infer_genotype(likelihoods, uniform_genotype_prior(states), states)
    assert not bool(evidence.candidate_complete)
    assert int(likelihoods.status) == GenotypingStatus.CANDIDATE_OMITTED
    assert not bool(result.valid)
    assert bool(result.hard_call.no_call)


def test_normalization_minimizes_repeats_and_checks_contig_edges() -> None:
    repeat = normalize_small_variant(
        "CAAAAAG",
        3,
        "AA",
        ("A",),
        max_alleles=3,
        max_allele_length=4,
    )
    assert bool(repeat.valid)
    assert int(repeat.site.position) == 0
    assert int(repeat.evidence.left_shift) == 3
    assert decode_variant_alleles(repeat.site) == ("CA", "C")

    minimized = normalize_small_variant(
        "GACGT",
        1,
        "ACG",
        ("ATG",),
        max_alleles=2,
        max_allele_length=3,
    )
    assert int(minimized.site.position) == 2
    assert decode_variant_alleles(minimized.site) == ("C", "T")

    at_edge = normalize_small_variant(
        "AAAA",
        0,
        "AA",
        ("A",),
        max_alleles=2,
        max_allele_length=2,
    )
    assert int(at_edge.site.position) == 0
    over_edge = normalize_small_variant(
        "AAAA",
        3,
        "AA",
        ("A",),
        max_alleles=2,
        max_allele_length=2,
    )
    assert not bool(over_edge.valid)
    assert int(over_edge.status) == VariantNormalizationStatus.INVALID_POSITION


def test_normalization_capacity_failure_does_not_truncate_alleles() -> None:
    result = normalize_small_variant(
        "ACGT",
        1,
        "C",
        ("A", "G"),
        max_alleles=2,
        max_allele_length=1,
    )
    assert not bool(result.valid)
    assert int(result.status) == VariantNormalizationStatus.CAPACITY_EXCEEDED
    assert int(result.evidence.required_allele_count) == 3
    assert not np.asarray(result.site.allele_mask).any()


def test_vcf_like_parse_write_no_call_gl_pl_and_normalization() -> None:
    text = (
        "##fileformat=VCFv4.3\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
        "chr1\t2\t.\tA\tC\t50\tPASS\tDP=8\tGT:GQ:DP:GL:PL:PS\t"
        "0|1:40:8:-3,0,-3:30,0,30:2\t./.:.:0:.:.:.\n"
    )
    parsed = parse_vcf(text, max_records=2, max_samples=2, max_alleles=3)
    assert parsed.valid
    assert parsed.records[0].samples[0].called
    assert parsed.records[0].samples[0].phased
    assert parsed.records[0].samples[0].phase_set == 2
    assert parsed.records[0].samples[1].no_call

    states = enumerate_genotype_states(2, 2, 3)
    likelihoods = vcf_sample_likelihoods(parsed.records[0].samples[0], states)
    np.testing.assert_allclose(
        np.asarray(genotype_likelihoods_to_gl(likelihoods))[:3],
        np.asarray((-3.0, 0.0, -3.0)),
        atol=1.0e-6,
    )
    normalized = vcf_record_to_small_variant(
        parsed.records[0],
        "GACG",
        reference_index=4,
        contig_index=0,
        max_alleles=3,
        max_allele_length=2,
    )
    assert bool(normalized.valid)
    assert int(normalized.site.position) == 1

    written = write_vcf(parsed.header, parsed.records, max_records=2)
    assert written.valid
    reparsed = parse_vcf(written.text, max_records=2, max_samples=2, max_alleles=3)
    assert reparsed.valid
    assert reparsed.records == parsed.records


def test_vcf_capacity_overflow_returns_no_partial_records() -> None:
    text = (
        "##fileformat=VCFv4.3\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "chr1\t1\t.\tA\tC\t.\t.\t.\n"
        "chr1\t2\t.\tC\tG\t.\t.\t.\n"
    )
    parsed = parse_vcf(text, max_records=1, max_samples=0, max_alleles=2)
    assert not parsed.valid
    assert parsed.status == VariantInterchangeStatus.CAPACITY_EXCEEDED
    assert parsed.records == ()
    assert parsed.evidence.record_count == 2


def _switch_reads() -> tuple[jnp.ndarray, jnp.ndarray]:
    calls = jnp.asarray(((0, 0, 0), (1, 1, 1), (0, 0, 1), (1, 1, 0)), dtype=jnp.int32)
    mask = jnp.asarray(
        (
            (True, True, False),
            (True, True, False),
            (False, True, True),
            (False, True, True),
        )
    )
    return calls, mask


def test_read_backed_phasing_reports_phase_sets_and_switch_evidence() -> None:
    calls, mask = _switch_reads()
    evidence = read_phase_evidence_from_calls(
        calls,
        jnp.full(calls.shape, 55.0),
        jnp.full(calls.shape, 60.0),
        mask,
        allele_count=2,
    )
    result = phase_small_variants(
        jnp.asarray(((0, 1), (0, 1), (0, 1)), dtype=jnp.int32),
        evidence,
        max_phase_states=2,
        min_switch_log_odds=math.log(2.0),
    )
    assert bool(result.valid)
    assert int(result.status) == PhasingStatus.OK
    assert float(result.switch_log_odds[0]) < 0.0
    assert float(result.switch_log_odds[1]) > 0.0
    np.testing.assert_array_equal(np.asarray(result.switch_supported), (False, True))
    assert len(set(np.asarray(result.phase_sets).tolist())) == 1
    assert np.asarray(result.phase_mask).all()
    assert not np.array_equal(
        np.asarray(result.phased_genotypes[1]),
        np.asarray(result.phased_genotypes[2]),
    )


def test_pedigree_phasing_and_mendelian_inconsistency_are_explicit() -> None:
    calls = jnp.zeros((1, 2), dtype=jnp.int32)
    observations = jnp.zeros((1, 2), dtype=bool)
    evidence = read_phase_evidence_from_calls(
        calls,
        jnp.zeros_like(calls, dtype=jnp.float32),
        jnp.zeros_like(calls, dtype=jnp.float32),
        observations,
        allele_count=2,
        read_mask=jnp.asarray((False,)),
    )
    pedigree = PedigreePhaseEvidence(
        jnp.asarray(((0, 0), (0, 0))),
        jnp.asarray(((1, 1), (1, 1))),
    )
    consistent = phase_small_variants(
        jnp.asarray(((0, 1), (0, 1))),
        evidence,
        max_phase_states=2,
        pedigree_evidence=pedigree,
    )
    assert bool(consistent.valid)
    assert np.asarray(consistent.phase_mask).all()
    np.testing.assert_array_equal(
        np.asarray(consistent.phased_genotypes), np.asarray(((0, 1), (0, 1)))
    )
    assert np.asarray(consistent.mendelian_consistent).all()

    inconsistent = phase_small_variants(
        jnp.asarray(((1, 1), (1, 1))),
        evidence,
        max_phase_states=2,
        pedigree_evidence=PedigreePhaseEvidence(
            jnp.asarray(((0, 0), (0, 0))),
            jnp.asarray(((0, 0), (0, 0))),
        ),
    )
    assert not bool(inconsistent.valid)
    assert int(inconsistent.status) == PhasingStatus.MENDELIAN_INCONSISTENT
    assert not np.asarray(inconsistent.mendelian_consistent).any()


def test_phasing_capacity_and_candidate_omission_fail_observably() -> None:
    calls = jnp.asarray(((3,),), dtype=jnp.int32)
    evidence = read_phase_evidence_from_calls(
        calls,
        jnp.asarray(((40.0,),)),
        jnp.asarray(((60.0,),)),
        jnp.asarray(((True,),)),
        allele_count=2,
    )
    omitted = phase_small_variants(jnp.asarray(((0, 1),)), evidence, max_phase_states=2)
    assert not bool(omitted.valid)
    assert int(omitted.status) == PhasingStatus.CANDIDATE_OMITTED

    triploid_calls = jnp.zeros((1, 1), dtype=jnp.int32)
    triploid_evidence = read_phase_evidence_from_calls(
        triploid_calls,
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1)),
        jnp.zeros((1, 1), dtype=bool),
        allele_count=2,
        read_mask=jnp.asarray((False,)),
    )
    overflow = phase_small_variants(
        jnp.asarray(((0, 1, 1),)),
        triploid_evidence,
        max_phase_states=5,
    )
    assert not bool(overflow.valid)
    assert int(overflow.status) == PhasingStatus.CAPACITY_EXCEEDED
    assert int(overflow.evidence.required_phase_state_count) == 6
