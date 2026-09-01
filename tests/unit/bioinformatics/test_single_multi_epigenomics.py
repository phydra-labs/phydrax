#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.bioinformatics.foundation import (
    BiospecimenLineage,
    ExchangeabilityPlan,
    ExperimentalUnitPlan,
    FeatureDictionary,
    FeatureMapping,
    MethodKind,
    OntologyGraph,
)
from phydrax.bioinformatics.genomics._epigenomics import (
    bisulfite_methylation_statistics,
    BisulfiteCallBatch,
    BisulfiteMethylationPlan,
    chromatin_fragment_statistics,
    ChromatinFragmentBatch,
    EPIGENOMICS_CAPACITY_EXCEEDED,
    EPIGENOMICS_CONVERSION_FAILED,
    EPIGENOMICS_INSUFFICIENT_COVERAGE,
    EPIGENOMICS_INVALID_CONTEXT,
    EPIGENOMICS_ML_UNCALIBRATED,
    EPIGENOMICS_MM_ML_ORIENTATION_INVALID,
    mm_ml_modification_statistics,
    MMMLCalibrationPlan,
    MMMLModificationBatch,
    PeakControlBlacklistPlan,
)
from phydrax.bioinformatics.omics._composition import (
    donor_composition_inputs,
    donor_composition_logratio_contrast,
)
from phydrax.bioinformatics.omics._integration import (
    INTEGRATION_CONFIRMATORY_USE_FORBIDDEN,
    INTEGRATION_PROVENANCE_MISMATCH,
    transport_exploratory_integration,
    TransportIntegrationPlan,
)
from phydrax.bioinformatics.omics._multiomics import (
    align_modalities,
    MultiomicAlignmentPlan,
)
from phydrax.bioinformatics.omics._pathways import (
    ontology_feature_set_test,
    OntologyFeatureSetTestPlan,
    PATHWAY_CAPACITY_EXCEEDED,
    PATHWAY_CORRECTED_EMBEDDING_FORBIDDEN,
)
from phydrax.bioinformatics.omics._single_cell import (
    CellQCThresholds,
    summarize_cell_qc,
)
from phydrax.bioinformatics.omics._transcript_abundance import (
    differential_transcript_usage,
    estimate_transcript_abundance,
    TRANSCRIPT_NONCONVERGED,
    TranscriptEquivalenceBatch,
)
from phydrax.bioinformatics.omics._velocity import (
    kinetic_rna_velocity,
    KineticRNAVelocityPlan,
    VELOCITY_LAYER_ASSUMPTION_FAILED,
)
from phydrax.sparse import RowRelation
from phydrax.transport import AbstractBalancedTransportPlan


def _exchangeability(condition_ids: tuple[int, ...]) -> ExchangeabilityPlan:
    donor_count = len(condition_ids)
    entity_count = 5 * donor_count
    entity_ids = jnp.arange(entity_count)
    entity_kinds = jnp.tile(jnp.asarray([0, 1, 2, 3, 5]), donor_count)
    parents = []
    children = []
    for donor in range(donor_count):
        base = 5 * donor
        parents.extend((base, base + 1, base + 2, base + 3))
        children.extend((base + 1, base + 2, base + 3, base + 4))
    replicate_ids = -jnp.ones((entity_count,), dtype=jnp.int32)
    observation_indices = jnp.arange(4, entity_count, 5)
    replicate_ids = replicate_ids.at[observation_indices].set(
        jnp.arange(donor_count, dtype=jnp.int32)
    )
    lineage = BiospecimenLineage(
        entity_ids,
        entity_kinds,
        jnp.asarray(parents),
        jnp.asarray(children),
        replicate_ids,
        replicate_ids,
        study_id="unit-study",
    )
    units = ExperimentalUnitPlan(
        lineage,
        observation_indices,
        jnp.arange(0, entity_count, 5),
        jnp.asarray(condition_ids),
    )
    return ExchangeabilityPlan(units, jnp.zeros((donor_count,), dtype=jnp.int32))


def _features(size: int, namespace: str) -> FeatureDictionary:
    return FeatureDictionary(
        jnp.arange(size),
        namespace=namespace,
        version="1",
        species="human",
        reference="test",
        annotation="test",
    )


class _DenseTestTransport(AbstractBalancedTransportPlan):
    coupling: Array
    source_weights: Array
    target_weights: Array
    regularized_cost: Array
    converged_value: Array

    @property
    def converged(self):
        return self.converged_value

    def regularized_objective(self):
        return self.regularized_cost

    def source_marginal(self):
        return jnp.sum(self.coupling, axis=1)

    def target_marginal(self):
        return jnp.sum(self.coupling, axis=0)

    def apply_source_to_target(self, values):
        return self.coupling.T @ jnp.asarray(values)

    def apply_target_to_source(self, values):
        return self.coupling @ jnp.asarray(values)

    def barycentric_source_to_target(self, values):
        applied = self.apply_source_to_target(values)
        return applied / self.target_weights[:, None]

    def barycentric_target_to_source(self, values):
        applied = self.apply_target_to_source(values)
        return applied / self.source_weights[:, None]

    def dense_plan(self):
        return self.coupling


def test_cell_qc_and_composition_keep_biological_replicates() -> None:
    qc = summarize_cell_qc(
        jnp.asarray([100, 10, 80, 70]),
        jnp.asarray([50, 5, 40, 35]),
        jnp.asarray([5, 8, 4, 3]),
        jnp.asarray([0.1, 0.9, 0.1, 0.2]),
        jnp.asarray([0, 0, 1, 1]),
        sample_count=2,
        thresholds=CellQCThresholds(
            minimum_total_counts=20,
            minimum_detected_features=10,
            maximum_mitochondrial_fraction=0.2,
            maximum_doublet_score=0.5,
        ),
    )
    np.testing.assert_array_equal(qc.evidence.cells_observed, [2, 2])
    np.testing.assert_array_equal(qc.evidence.cells_accepted, [1, 2])
    assert qc.evidence.replicate_unit == "sample"
    assert qc.evidence.decision_reason_counts[0, 0] == 1
    assert qc.evidence.decision_reason_counts[0, 3] == 1
    assert qc.method_contract.method_kind is MethodKind.HEURISTIC
    assert qc.claim_kind == "heuristic_qc_decision"

    exchangeability = _exchangeability((0, 0, 1, 1))
    composition = donor_composition_inputs(
        jnp.asarray([0] * 20 + [1] * 2 + [2] * 3 + [3] * 30),
        jnp.asarray([0] * 10 + [1] * 10 + [0, 1] + [0, 0, 1] + [0] * 10 + [1] * 20),
        jnp.ones((4, 1)),
        donor_count=4,
        cell_type_count=2,
        exchangeability=exchangeability,
    )
    contrast = donor_composition_logratio_contrast(
        composition, jnp.asarray([False, False, True, True])
    )
    np.testing.assert_array_equal(composition.evidence.donor_observed, [True] * 4)
    assert composition.evidence.replicate_unit == "donor"
    assert int(contrast.evidence.donor_replicates_used) == 4
    assert bool(contrast.valid)


def test_multimodal_alignment_preserves_missingness_one_to_many_and_fit_provenance() -> (
    None
):
    query_features = _features(2, "query")
    reference_features = _features(2, "reference")
    mapping = FeatureMapping(
        query_features,
        reference_features,
        jnp.asarray([0, 0, 1]),
        jnp.asarray([0, 1, 1]),
    )
    plan = MultiomicAlignmentPlan(
        jnp.asarray([True, True, False]),
        mapping,
        jnp.asarray([1.0, 0.5, 0.5]),
        fit_provenance_id="train-split",
    )
    result = align_modalities(
        jnp.asarray([[1.0, 2.0], [2.0, 4.0], [9.0, 9.0]]),
        jnp.asarray([[2.0, 1.0], [4.0, 2.0], [8.0, 8.0]]),
        jnp.asarray([True, True, False]),
        jnp.asarray([True, True, True]),
        plan,
        expected_fit_provenance_id="train-split",
    )
    np.testing.assert_array_equal(
        result.evidence.modality_presence,
        [[True, True], [True, True], [False, True]],
    )
    assert bool(result.evidence.one_to_many_source_feature[0])
    assert bool(result.evidence.fitted_provenance_match)
    assert result.method_contract.method_kind is MethodKind.LEARNED
    assert result.claim_kind == "learned_alignment"
    assert np.all(np.asarray(result.standardized_reference[2]) == 0.0)

    mismatch = align_modalities(
        jnp.ones((3, 2)),
        jnp.ones((3, 2)),
        jnp.ones((3,), dtype=bool),
        jnp.ones((3,), dtype=bool),
        plan,
        expected_fit_provenance_id="test-split",
    )
    assert not bool(jnp.all(mismatch.valid))


def test_equivalence_abundance_and_differential_usage_use_sample_replicates() -> None:
    compatibility = RowRelation(
        jnp.asarray([[0, 1], [1, 2]]),
        source_size=3,
    )
    batch = TranscriptEquivalenceBatch(
        jnp.asarray([[30, 10], [20, 20], [10, 30], [12, 28]]),
        compatibility,
        jnp.asarray([1.0, 1.0, 1.0]),
    )
    abundance = estimate_transcript_abundance(batch, maximum_iterations=128)
    np.testing.assert_allclose(
        jnp.sum(abundance.relative_abundance, axis=1), jnp.ones((4,)), rtol=1e-5
    )
    assert bool(jnp.all(abundance.valid))
    assert abundance.method_contract.method_kind is MethodKind.APPROXIMATE_MODEL
    bounded = estimate_transcript_abundance(batch, maximum_iterations=1, tolerance=1e-12)
    assert bool(jnp.any(bounded.status == TRANSCRIPT_NONCONVERGED))

    genes = RowRelation(jnp.asarray([[0, 1], [2, 0]]), source_size=3)
    usage = differential_transcript_usage(
        abundance,
        genes,
        jnp.asarray([False, False, True, True]),
        exchangeability=_exchangeability((0, 0, 1, 1)),
    )
    assert int(usage.evidence.first_group_samples) == 2
    assert int(usage.evidence.second_group_samples) == 2
    assert usage.evidence.replicate_unit == "sample"
    assert bool(usage.valid)


def test_ontology_test_requires_exchangeability_and_rejects_corrected_embedding() -> None:
    genes = _features(3, "gene")
    sets = _features(2, "feature-set")
    membership = FeatureMapping(
        genes,
        sets,
        jnp.asarray([0, 1, 1, 2]),
        jnp.asarray([0, 0, 1, 1]),
    )
    ontology = OntologyGraph(sets, jnp.asarray([1]), jnp.asarray([0]))
    statistics = jnp.asarray([2.0, 1.0, -1.0])
    permutations = jnp.asarray([[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]])
    exchangeability = _exchangeability((0, 0, 1, 1))
    result = ontology_feature_set_test(
        statistics,
        permutations,
        membership,
        ontology,
        OntologyFeatureSetTestPlan(
            maximum_permutations=3,
            maximum_ontology_steps=2,
            input_provenance_id="raw-count-statistic",
        ),
        exchangeability=exchangeability,
    )
    assert bool(result.valid)
    assert bool(result.evidence.exchangeability_declared)
    assert result.ontology_adjusted_p_value[1] >= result.p_value[0]

    forbidden = ontology_feature_set_test(
        statistics,
        permutations,
        membership,
        ontology,
        OntologyFeatureSetTestPlan(
            maximum_permutations=3,
            maximum_ontology_steps=2,
            input_provenance_id="corrected",
            corrected_embedding_used=True,
        ),
        exchangeability=exchangeability,
    )
    assert int(forbidden.status) == PATHWAY_CORRECTED_EMBEDDING_FORBIDDEN
    assert not bool(forbidden.valid)
    over_capacity = ontology_feature_set_test(
        statistics,
        permutations,
        membership,
        ontology,
        OntologyFeatureSetTestPlan(
            maximum_permutations=2,
            maximum_ontology_steps=2,
            input_provenance_id="raw-count-statistic",
        ),
        exchangeability=exchangeability,
    )
    assert int(over_capacity.status) == PATHWAY_CAPACITY_EXCEEDED


def test_transport_integration_is_exploratory_and_checks_training_provenance() -> None:
    coupling = jnp.asarray([[0.5, 0.0], [0.0, 0.5]])
    transport = _DenseTestTransport(
        coupling=coupling,
        source_weights=jnp.asarray([0.5, 0.5]),
        target_weights=jnp.asarray([0.5, 0.5]),
        regularized_cost=jnp.asarray(0.1),
        converged_value=jnp.asarray(True),
    )
    plan = TransportIntegrationPlan(
        jnp.asarray([True, False]),
        jnp.asarray([True, False]),
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([0.5, 0.5]),
        fitted_on_split_id="train",
    )
    result = transport_exploratory_integration(
        transport,
        jnp.asarray([[0.0], [2.0]]),
        jnp.asarray([[0.0], [2.0]]),
        plan,
        expected_split_id="train",
    )
    assert bool(result.valid)
    assert result.claim_kind == "exploratory_regularized_transport"
    assert result.method_contract.method_kind is MethodKind.RELAXED_OBJECTIVE
    assert float(result.evidence.source_marginal_residual) == 0.0

    wrong_split = transport_exploratory_integration(
        transport,
        jnp.asarray([[0.0], [2.0]]),
        jnp.asarray([[0.0], [2.0]]),
        plan,
        expected_split_id="test",
    )
    assert int(wrong_split.status) == INTEGRATION_PROVENANCE_MISMATCH
    confirmatory = transport_exploratory_integration(
        transport,
        jnp.asarray([[0.0], [2.0]]),
        jnp.asarray([[0.0], [2.0]]),
        plan,
        expected_split_id="train",
        requested_use="confirmatory",
    )
    assert int(confirmatory.status) == INTEGRATION_CONFIRMATORY_USE_FORBIDDEN


def test_velocity_requires_aligned_spliced_unspliced_count_layers() -> None:
    plan = KineticRNAVelocityPlan(
        parameter_origin="training_fitted", fitted_on_split_id="train"
    )
    result = kinetic_rna_velocity(
        jnp.asarray([[5.0, 2.0], [3.0, 1.0]]),
        jnp.asarray([[2.0, 1.0], [1.0, 0.0]]),
        jnp.asarray([3.0, 2.0]),
        jnp.asarray([1.0, 1.0]),
        jnp.asarray([0.5, 0.5]),
        plan,
        expected_split_id="train",
    )
    np.testing.assert_allclose(result.unspliced_velocity[0], [1.0, 1.0])
    assert bool(jnp.all(result.valid))
    assert result.claim_kind == "kinetic_model_estimate"

    invalid_plan = KineticRNAVelocityPlan(
        count_layers=False,
        parameter_origin="externally_calibrated",
        fitted_on_split_id="train",
    )
    invalid = kinetic_rna_velocity(
        jnp.ones((1, 2)),
        jnp.ones((1, 2)),
        1.0,
        1.0,
        1.0,
        invalid_plan,
        expected_split_id="train",
    )
    assert int(invalid.status[0]) == VELOCITY_LAYER_ASSUMPTION_FAILED


def test_atac_chip_peak_control_and_blacklist_are_distinct() -> None:
    fragments = ChromatinFragmentBatch(
        jnp.asarray([0, 0, 0, 0]),
        jnp.asarray([10, 16, 31, 12]),
        jnp.asarray([12, 17, 33, 14]),
        jnp.asarray([0, 0, 0, 1]),
        assay_kind="chip",
    )
    intervals = PeakControlBlacklistPlan(
        jnp.asarray([0]),
        jnp.asarray([10]),
        jnp.asarray([20]),
        jnp.asarray([0]),
        jnp.asarray([30]),
        jnp.asarray([40]),
        jnp.asarray([0]),
        jnp.asarray([15]),
        jnp.asarray([18]),
        maximum_fragments=8,
        maximum_intervals=4,
    )
    result = chromatin_fragment_statistics(fragments, intervals, sample_count=2)
    np.testing.assert_array_equal(result.peak_counts[:, 0], [1, 1])
    np.testing.assert_array_equal(result.control_counts[:, 0], [1, 0])
    np.testing.assert_array_equal(result.evidence.fragments_blacklisted, [1, 0])
    assert result.evidence.assay_kind == "chip"
    assert result.method_contract.method_name == (
        "chip_fragment_peak_control_blacklist_statistics"
    )
    assert bool(jnp.all(result.valid))
    bounded_intervals = PeakControlBlacklistPlan(
        intervals.peak_contig,
        intervals.peak_start,
        intervals.peak_end,
        intervals.control_contig,
        intervals.control_start,
        intervals.control_end,
        intervals.blacklist_contig,
        intervals.blacklist_start,
        intervals.blacklist_end,
        maximum_fragments=2,
        maximum_intervals=4,
    )
    bounded = chromatin_fragment_statistics(fragments, bounded_intervals, sample_count=2)
    assert bool(jnp.all(bounded.status == EPIGENOMICS_CAPACITY_EXCEEDED))


def test_bisulfite_statistics_protect_coverage_context_and_conversion() -> None:
    calls = BisulfiteCallBatch(
        jnp.asarray([8, 4, 1, 6, 3, 1]),
        jnp.asarray([10, 10, 10, 10, 10, 10]),
        jnp.asarray([0, 1, 2, 0, 1, 2]),
        jnp.asarray([0, 0, 0, 1, 1, 1]),
    )
    plan = BisulfiteMethylationPlan(
        minimum_coverage_per_context=10,
        minimum_conversion_rate=0.95,
        maximum_calls=8,
    )
    result = bisulfite_methylation_statistics(
        calls,
        jnp.asarray([1, 0]),
        jnp.asarray([100, 100]),
        plan,
        sample_count=2,
    )
    np.testing.assert_array_equal(result.coverage, jnp.full((2, 3), 10))
    np.testing.assert_allclose(result.methylation_fraction[0], [0.8, 0.4, 0.1])
    assert result.evidence.context_names == ("CG", "CHG", "CHH")
    assert bool(jnp.all(result.valid))

    failed_conversion = bisulfite_methylation_statistics(
        calls,
        jnp.asarray([10, 0]),
        jnp.asarray([100, 100]),
        plan,
        sample_count=2,
    )
    assert int(failed_conversion.status[0]) == EPIGENOMICS_CONVERSION_FAILED
    assert not bool(failed_conversion.valid[0])
    low_coverage_plan = BisulfiteMethylationPlan(
        minimum_coverage_per_context=11,
        minimum_conversion_rate=0.95,
        maximum_calls=8,
    )
    low_coverage = bisulfite_methylation_statistics(
        calls,
        jnp.asarray([1, 0]),
        jnp.asarray([100, 100]),
        low_coverage_plan,
        sample_count=2,
    )
    assert bool(jnp.all(low_coverage.status == EPIGENOMICS_INSUFFICIENT_COVERAGE))
    invalid_context_calls = BisulfiteCallBatch(
        jnp.asarray([1]),
        jnp.asarray([10]),
        jnp.asarray([3]),
        jnp.asarray([0]),
    )
    invalid_context = bisulfite_methylation_statistics(
        invalid_context_calls,
        jnp.asarray([0]),
        jnp.asarray([100]),
        plan,
        sample_count=1,
    )
    assert int(invalid_context.status[0]) == EPIGENOMICS_INVALID_CONTEXT
    bounded_methylation = bisulfite_methylation_statistics(
        calls,
        jnp.asarray([1, 0]),
        jnp.asarray([100, 100]),
        BisulfiteMethylationPlan(
            minimum_coverage_per_context=10,
            minimum_conversion_rate=0.95,
            maximum_calls=5,
        ),
        sample_count=2,
    )
    assert bool(jnp.all(bounded_methylation.status == EPIGENOMICS_CAPACITY_EXCEEDED))


def test_mm_ml_orientation_and_calibration_are_auditable() -> None:
    calls = MMMLModificationBatch(
        jnp.asarray([0, 1]),
        jnp.asarray([2, 2]),
        jnp.asarray([102, 202]),
        jnp.asarray([0, 0]),
        jnp.asarray([1, 1]),
        jnp.asarray([255, 128]),
        jnp.asarray([10, 10]),
        jnp.asarray([0, 0]),
        jnp.asarray([False, True]),
    )
    calibrated = MMMLCalibrationPlan(
        ("C+m",),
        maximum_calls=4,
        calibrated=True,
        calibration_provenance_id="isotonic-training",
    )
    result = mm_ml_modification_statistics(calls, calibrated, sample_count=1)
    np.testing.assert_array_equal(result.evidence.reference_forward_strand, [True, False])
    np.testing.assert_array_equal(result.forward_strand_count, [[1]])
    np.testing.assert_array_equal(result.reverse_strand_count, [[1]])
    assert bool(result.valid[0])
    assert result.evidence.calibration_provenance_id == "isotonic-training"

    uncalibrated = MMMLCalibrationPlan(
        ("C+m",),
        maximum_calls=4,
        calibrated=False,
        calibration_provenance_id="raw-ml",
    )
    invalid = mm_ml_modification_statistics(calls, uncalibrated, sample_count=1)
    assert int(invalid.status) == EPIGENOMICS_ML_UNCALIBRATED
    assert not bool(invalid.valid[0])
    invalid_orientation_calls = MMMLModificationBatch(
        jnp.asarray([0]),
        jnp.asarray([2]),
        jnp.asarray([102]),
        jnp.asarray([0]),
        jnp.asarray([0]),
        jnp.asarray([255]),
        jnp.asarray([10]),
        jnp.asarray([0]),
        jnp.asarray([False]),
    )
    invalid_orientation = mm_ml_modification_statistics(
        invalid_orientation_calls, calibrated, sample_count=1
    )
    assert int(invalid_orientation.status) == EPIGENOMICS_MM_ML_ORIENTATION_INVALID
    bounded_calibration = MMMLCalibrationPlan(
        ("C+m",),
        maximum_calls=1,
        calibrated=True,
        calibration_provenance_id="bounded",
    )
    bounded = mm_ml_modification_statistics(calls, bounded_calibration, sample_count=1)
    assert int(bounded.status) == EPIGENOMICS_CAPACITY_EXCEEDED
