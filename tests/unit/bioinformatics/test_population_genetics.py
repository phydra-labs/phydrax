#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.bioinformatics.population._association import (
    AssociationStatus,
    binary_association,
    leave_one_chromosome_out_kinship,
    quantitative_association,
    UNSUPPORTED_ASSOCIATION_ANALYSES,
)
from phydrax.bioinformatics.population._cohort import GenotypeCohort
from phydrax.bioinformatics.population._demography import (
    pairwise_coalescent_log_density,
    PiecewiseConstantDemography,
    sfs_log_likelihood,
    standard_neutral_expected_sfs,
)
from phydrax.bioinformatics.population._imputation import (
    ImputationPlan,
    ImputationStatus,
    impute_genotypes,
    ReferenceHaplotypePanel,
)
from phydrax.bioinformatics.population._pedigree import (
    infer_pedigree,
    PedigreeInferencePlan,
)
from phydrax.bioinformatics.population._recombination import (
    infer_local_ancestry,
    infer_recombination_mosaic,
    RecombinationMap,
)
from phydrax.bioinformatics.population._summary import (
    allele_counts,
    genomic_kinship,
    hardy_weinberg,
    linkage_disequilibrium,
    PopulationSummaryStatus,
    site_frequency_spectrum,
)
from phydrax.bioinformatics.population._tree_sequence import (
    EdgeTable,
    marginal_tree,
    MutationTable,
    NodeTable,
    SAMPLE_NODE,
    SiteTable,
    summarize_tree_sequence,
    TreeSequenceTables,
)


def _uncertain_mixed_cohort() -> GenotypeCohort:
    probabilities = jnp.asarray(
        [
            [[0.9, 0.1, 0.0], [0.1, 0.2, 0.7], [0.0, 1.0, 0.0]],
            [[0.2, 0.8, 0.0], [0.7, 0.3, 0.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.2, 0.8], [0.4, 0.2, 0.4], [0.3, 0.7, 0.0]],
        ]
    )
    return GenotypeCohort(
        probabilities,
        jnp.asarray([[2, 2, 2], [1, 1, 1], [2, 2, 1]]),
        jnp.asarray([[True, True, False], [True, True, True], [True, True, True]]),
        jnp.asarray([10.0, 20.0, 30.0]),
        jnp.asarray([0, 1, 1]),
        sample_ids=("s0", "s1", "s2"),
        chromosome_labels=("1", "X"),
        reference_alleles=("A", "C", "G"),
        alternate_alleles=("G", "T", "A"),
        ancestral_is_alternate=jnp.asarray([False, True, False]),
    )


def test_genotype_uncertainty_missingness_and_sex_chromosome_summaries():
    cohort = _uncertain_mixed_cohort()
    # Missing posterior is canonical and uninformative, but observed-data counts omit it.
    np.testing.assert_allclose(cohort.genotype_probabilities[0, 2], [1 / 3, 1 / 3, 1 / 3])
    np.testing.assert_allclose(cohort.dosage[0, :2], [0.1, 1.6])
    likelihood_cohort = GenotypeCohort.from_log_likelihoods(
        jnp.log(cohort.genotype_probabilities),
        cohort.ploidy,
        cohort.observed,
        cohort.positions,
        cohort.chromosome_index,
        sample_ids=cohort.sample_ids,
        chromosome_labels=cohort.chromosome_labels,
        reference_alleles=cohort.reference_alleles,
        alternate_alleles=cohort.alternate_alleles,
        ancestral_is_alternate=cohort.ancestral_is_alternate,
        polarization_known=cohort.polarization_known,
    )
    np.testing.assert_allclose(
        likelihood_cohort.genotype_probabilities,
        cohort.genotype_probabilities,
        atol=1e-6,
    )
    counts = allele_counts(cohort)
    np.testing.assert_allclose(counts.alternate_count, [1.7, 1.1, 3.5])
    np.testing.assert_array_equal(counts.allele_number, [4, 3, 5])
    assert bool(jnp.all(counts.successful))

    hwe = hardy_weinberg(cohort)
    # Haploid X observations are not misclassified as homozygous diploids.
    assert int(hwe.diploid_samples[1]) == 0
    assert int(hwe.status[1]) == int(PopulationSummaryStatus.NON_DIPLOID)
    np.testing.assert_allclose(hwe.genotype_counts[0], [1.0, 0.3, 0.7])

    ld = linkage_disequilibrium(cohort)
    assert int(ld.overlapping_samples[0, 2]) == 2
    np.testing.assert_allclose(ld.r_squared, ld.correlation**2, equal_nan=True)
    kinship = genomic_kinship(cohort)
    np.testing.assert_allclose(kinship.relationship, kinship.relationship.T, atol=1e-6)


def test_sfs_polarization_and_capacity_are_observable():
    cohort = GenotypeCohort.from_calls(
        jnp.asarray([[0, 1], [2, 1]]),
        jnp.full((2, 2), 2),
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([0, 0]),
        sample_ids=("a", "b"),
        chromosome_labels=("1",),
        ancestral_is_alternate=jnp.asarray([False, True]),
    )
    spectrum = site_frequency_spectrum(cohort)
    # Both sites carry one derived allele: the second site is reversed by polarization.
    np.testing.assert_allclose(spectrum.spectrum, [0.0, 2.0, 0.0, 0.0, 0.0])
    assert bool(spectrum.successful)

    folded = site_frequency_spectrum(cohort, folded=True)
    np.testing.assert_allclose(folded.spectrum, [0.0, 2.0, 0.0, 0.0, 0.0])
    failed = site_frequency_spectrum(cohort, maximum_allele_number=3)
    assert not bool(failed.valid)
    assert int(failed.status) == int(PopulationSummaryStatus.CAPACITY_EXCEEDED)
    np.testing.assert_array_equal(failed.evidence, [4, 3])


def test_bounded_pedigree_uses_uncertain_mendelian_evidence():
    # Individuals 0/1 are homozygous opposite parents; child 2 is heterozygous.
    calls = jnp.asarray([[0, 2, 1, 0], [0, 2, 1, 2], [0, 2, 1, 0]])
    cohort = GenotypeCohort.from_calls(
        calls,
        jnp.full(calls.shape, 2),
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.zeros((3,), dtype=jnp.int32),
        sample_ids=("mother", "father", "child", "other"),
        chromosome_labels=("1",),
    )
    candidates = (
        jnp.zeros((4, 4), dtype=bool)
        .at[2, 0]
        .set(True)
        .at[2, 1]
        .set(True)
        .at[2, 3]
        .set(True)
    )
    result = infer_pedigree(
        cohort,
        PedigreeInferencePlan(3),
        candidate_parent=candidates,
        sex=jnp.asarray([1, 2, 0, 2]),
    )
    np.testing.assert_array_equal(result.map_parent_pair[2], [0, 1])
    assert bool(result.valid[2])
    assert len(result.pgm_results) == 1


def test_recombination_and_local_ancestry_mosaics_switch_once():
    recombination_map = RecombinationMap(
        jnp.arange(6.0), jnp.asarray([0.0, 0.01, 0.02, 1.0, 1.01, 1.02])
    )
    emissions = jnp.asarray(
        [
            [0.0, -12.0],
            [0.0, -12.0],
            [0.0, -12.0],
            [-12.0, 0.0],
            [-12.0, 0.0],
            [-12.0, 0.0],
        ]
    )
    mosaic = infer_recombination_mosaic(
        emissions, recombination_map, jnp.asarray([0.5, 0.5])
    )
    np.testing.assert_array_equal(mosaic.state_path, [0, 0, 0, 1, 1, 1])
    assert bool(mosaic.successful)
    assert float(mosaic.switch_probability[2]) > 0.9

    ancestry = infer_local_ancestry(
        emissions,
        recombination_map,
        jnp.asarray([0.5, 0.5]),
        generations=5.0,
    )
    np.testing.assert_array_equal(ancestry.ancestry_path, [0, 0, 0, 1, 1, 1])
    assert bool(ancestry.successful)


def test_mixed_ploidy_imputation_and_capacity_failure():
    calls = jnp.asarray([[-1, -1], [1, 2], [-1, -1]])
    ploidy = jnp.asarray([[1, 2], [1, 2], [1, 2]])
    cohort = GenotypeCohort.from_calls(
        calls,
        ploidy,
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.zeros((3,), dtype=jnp.int32),
        sample_ids=("male-x", "diploid"),
        chromosome_labels=("X",),
        maximum_ploidy=2,
    )
    panel = ReferenceHaplotypePanel(
        jnp.asarray([[0, 0, 0], [1, 1, 1]]),
        cohort.positions,
        cohort.chromosome_index,
    )
    recombination_map = RecombinationMap(cohort.positions, jnp.asarray([0.0, 0.01, 0.02]))
    result = impute_genotypes(cohort, panel, recombination_map, ImputationPlan(4))
    np.testing.assert_array_equal(result.required_copying_states, [2, 4])
    np.testing.assert_allclose(
        jnp.sum(result.genotype_probabilities, axis=-1), 1.0, atol=1e-6
    )
    assert bool(jnp.all(result.successful))

    failed = impute_genotypes(cohort, panel, recombination_map, ImputationPlan(3))
    assert bool(jnp.all(failed.status == int(ImputationStatus.CAPACITY_EXCEEDED)))
    assert not bool(jnp.any(failed.valid))


def test_numeric_tree_sequence_accepts_multiple_roots():
    nodes = NodeTable(
        jnp.asarray([SAMPLE_NODE, SAMPLE_NODE, SAMPLE_NODE, SAMPLE_NODE, 0, 0]),
        jnp.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.5]),
    )
    edges = EdgeTable(
        jnp.zeros((4,)),
        jnp.full((4,), 10.0),
        jnp.asarray([4, 4, 5, 5]),
        jnp.asarray([0, 1, 2, 3]),
    )
    sites = SiteTable(jnp.asarray([2.0]), jnp.asarray([0]))
    mutations = MutationTable(jnp.asarray([0]), jnp.asarray([0]), jnp.asarray([1]))
    tables = TreeSequenceTables(nodes, edges, 10.0, sites=sites, mutations=mutations)
    tree = marginal_tree(tables, 5.0)
    assert bool(tree.successful)
    assert int(tree.root_count) == 2
    np.testing.assert_array_equal(jnp.flatnonzero(tree.root_mask), [4, 5])
    summary = summarize_tree_sequence(tables)
    assert summary.tree_count == 1
    np.testing.assert_array_equal(summary.root_count, [2])


def _association_cohort(sample_count: int = 20) -> GenotypeCohort:
    first = np.tile(np.asarray([0, 1, 2, 1]), sample_count // 4 + 1)[:sample_count]
    second = np.tile(np.asarray([0, 0, 1, 2, 2]), sample_count // 5 + 1)[:sample_count]
    third = np.roll(first, 2)
    calls = jnp.asarray(np.stack((first, second, third)))
    return GenotypeCohort.from_calls(
        calls,
        jnp.full(calls.shape, 2),
        jnp.asarray([1.0, 2.0, 3.0]),
        jnp.asarray([0, 0, 1]),
        sample_ids=tuple(f"s{index}" for index in range(sample_count)),
        chromosome_labels=("1", "2"),
    )


def test_quantitative_association_accounts_for_stratification_relatedness_and_loco():
    cohort = _association_cohort()
    ancestry = jnp.concatenate((jnp.zeros((10,)), jnp.ones((10,))))
    phenotype = 0.8 * cohort.dosage[0] + 2.0 * ancestry + jnp.linspace(-0.1, 0.1, 20)
    kinship = genomic_kinship(cohort).relationship
    loco = leave_one_chromosome_out_kinship(cohort)
    result = quantitative_association(
        cohort,
        phenotype,
        covariates=ancestry[:, None],
        loco=loco,
        relatedness_scale=0.2,
    )
    assert result.used_loco
    assert bool(result.valid[0])
    assert float(result.effect[0]) > 0.0
    np.testing.assert_array_equal(loco.informative_variant_count, [1, 2])

    related = quantitative_association(
        cohort,
        phenotype,
        covariates=ancestry[:, None],
        kinship=kinship,
        relatedness_scale=0.2,
    )
    assert bool(related.valid[0])


def test_binary_association_reports_imbalance_and_separation_status():
    cohort = _association_cohort()
    outcome = jnp.zeros((20,)).at[0].set(1.0)
    result = binary_association(
        cohort,
        outcome,
        covariates=jnp.tile(jnp.asarray([0.0, 1.0]), 10)[:, None],
        maximum_iterations=96,
        imbalance_threshold=0.1,
    )
    assert bool(result.imbalanced)
    assert int(result.case_count) == 1
    assert int(result.control_count) == 19
    assert bool(jnp.any(result.valid))

    separated = binary_association(
        cohort,
        jnp.asarray([0.0] * 10 + [1.0] * 10),
        covariates=jnp.asarray([0.0] * 10 + [1.0] * 10)[:, None],
        separation_threshold=5.0,
        maximum_iterations=96,
    )
    assert bool(jnp.all(separated.separated))
    assert bool(jnp.all(separated.status == int(AssociationStatus.SEPARATION)))


def test_demographic_and_sfs_likelihood_primitives_are_stable():
    demography = PiecewiseConstantDemography(
        jnp.asarray([100.0, jnp.inf]),
        jnp.asarray([[1_000.0], [10_000.0]]),
    )
    density = pairwise_coalescent_log_density(jnp.asarray([10.0, 200.0]), demography)
    assert bool(jnp.all(density.successful))
    assert float(density.cumulative_hazard[1]) > float(density.cumulative_hazard[0])

    expected = standard_neutral_expected_sfs(4, 2.0)
    np.testing.assert_allclose(
        expected.expected_spectrum, [0.0, 2.0, 1.0, 2.0 / 3.0, 0.0]
    )
    likelihood = sfs_log_likelihood(
        jnp.asarray([0.0, 3.0, 1.0, 1.0, 0.0]), expected.expected_spectrum
    )
    assert bool(likelihood.successful)
    assert bool(jnp.isfinite(likelihood.total_log_likelihood))


def test_fine_mapping_prs_and_survival_are_explicitly_out_of_scope():
    assert UNSUPPORTED_ASSOCIATION_ANALYSES == (
        "fine-mapping",
        "polygenic-risk-score",
        "survival",
    )
