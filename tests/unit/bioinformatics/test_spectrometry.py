#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from phydrax.bioinformatics.interchange._spectrometry import (
    lower_mzml_record,
    MzMLLoweringPlan,
    MzMLReadStatus,
)
from phydrax.bioinformatics.sequence import PROTEIN_IUPAC, SequenceBatch
from phydrax.bioinformatics.spectrometry._acquisition import (
    AcquisitionEvidence,
    AcquisitionKind,
    AcquisitionMetadata,
    AcquisitionRun,
    lookup_spectrum,
    PrecursorBatch,
)
from phydrax.bioinformatics.spectrometry._calibration import (
    apply_mass_calibration,
    bin_mass_spectrum,
    CalibrationStatus,
    mass_calibration_model,
    MassBinningPlan,
)
from phydrax.bioinformatics.spectrometry._features import (
    FeatureMatchPlan,
    FeatureMatchStatus,
    LCMSFeatureBatch,
    match_lcms_features,
)
from phydrax.bioinformatics.spectrometry._metabolomics import (
    AdductBatch,
    classify_metabolite_confidence,
    FormulaCandidateBatch,
    IsotopeEnvelopeBatch,
    match_metabolite_candidates,
    MetaboliteConfidenceLevel,
    MetaboliteFeatureBatch,
    MetaboliteMatchingPlan,
)
from phydrax.bioinformatics.spectrometry._protein_inference import (
    CompetitionLevel,
    infer_proteins_from_shared_peptides,
    PeptideEvidenceBatch,
    ProteinPeptideRelation,
    target_decoy_competition,
    TargetDecoyCompetitionBatch,
)
from phydrax.bioinformatics.spectrometry._proteomics import (
    ModificationBatch,
    ModificationSiteKind,
    PeptideCandidateBatch,
    PSMScoringInput,
    PSMScoringPlan,
    reverse_peptide_decoys,
    score_psms,
)
from phydrax.bioinformatics.spectrometry._quantification import (
    QuantificationBatch,
    QuantificationEvidence,
    QuantificationPlan,
    QuantificationSampleKind,
    quantify_replicates,
)
from phydrax.bioinformatics.spectrometry._spectrum import (
    Chromatogram,
    IntensityUnit,
    IonPolarity,
    MassSpectrum,
    MassToChargeUnit,
    SpectrometryUnits,
    SpectrumBatch,
    SpectrumRepresentation,
    TimeUnit,
)


def _protein_sequences(
    record_ids: list[int], sequences: list[str], width: int
) -> SequenceBatch:
    pad = PROTEIN_IUPAC.code(PROTEIN_IUPAC.pad_symbol)
    tokens = np.full((len(sequences), width), pad, dtype=np.int32)
    valid = np.zeros((len(sequences), width), dtype=bool)
    for row, sequence in enumerate(sequences):
        tokens[row, : len(sequence)] = [PROTEIN_IUPAC.code(symbol) for symbol in sequence]
        valid[row, : len(sequence)] = True
    return SequenceBatch(
        np.asarray(record_ids, dtype=np.int64),
        tokens,
        valid,
        np.ones((len(sequences),), dtype=bool),
        np.zeros_like(valid),
        PROTEIN_IUPAC,
    )


def test_profile_centroid_chromatogram_and_bin_boundaries() -> None:
    units = SpectrometryUnits(
        mass_to_charge=MassToChargeUnit.MZ,
        intensity=IntensityUnit.COUNTS,
        time=TimeUnit.SECOND,
    )
    profile = MassSpectrum(
        [100.0, 101.0, 102.0],
        [1.0, 2.0, 3.0],
        representation=SpectrumRepresentation.PROFILE,
        polarity=IonPolarity.POSITIVE,
        units=units,
    )
    centroid = MassSpectrum(
        [100.0, 101.0, 102.0],
        [1.0, 2.0, 3.0],
        representation=SpectrumRepresentation.CENTROID,
        polarity=IonPolarity.POSITIVE,
        units=units,
    )
    chromatogram = Chromatogram(
        [0.0, 1.0, 2.0],
        [0.0, 5.0, 1.0],
        precursor_mass_to_charge=500.2,
        product_mass_to_charge=200.1,
        polarity=IonPolarity.POSITIVE,
        units=units,
    )
    assert profile.representation is SpectrumRepresentation.PROFILE
    assert centroid.representation is SpectrumRepresentation.CENTROID
    assert bool(np.asarray(chromatogram.has_product))

    result = bin_mass_spectrum(centroid, MassBinningPlan([100.0, 101.0, 102.0]))
    np.testing.assert_allclose(result.intensity, [1.0, 5.0])
    np.testing.assert_array_equal(result.point_count, [1, 2])
    assert bool(np.asarray(result.valid))


def test_unit_and_polarity_mismatch_are_observable() -> None:
    spectrum = MassSpectrum([100.0], [1.0])
    wrong_unit = bin_mass_spectrum(
        spectrum,
        MassBinningPlan([99.0, 101.0], mass_to_charge_unit=MassToChargeUnit.THOMSON),
    )
    assert int(np.asarray(wrong_unit.status)) == int(CalibrationStatus.UNIT_MISMATCH)
    assert not bool(np.asarray(wrong_unit.valid))

    query = LCMSFeatureBatch(
        [1], [500.0], [10.0], [100.0], [2], polarity=IonPolarity.POSITIVE
    )
    reference = LCMSFeatureBatch(
        [2], [500.0], [10.0], [100.0], [-2], polarity=IonPolarity.NEGATIVE
    )
    matched = match_lcms_features(
        query,
        reference,
        FeatureMatchPlan(
            mass_tolerance_ppm=10.0,
            retention_time_tolerance=1.0,
            query_capacity=1,
            reference_capacity=1,
        ),
    )
    assert int(np.asarray(matched.status)) == int(FeatureMatchStatus.POLARITY_MISMATCH)
    assert not bool(np.asarray(matched.valid))


def test_chimeric_and_missing_spectrum_queries_and_mzml_capacity() -> None:
    spectra = SpectrumBatch(
        [[100.0, 200.0], [0.0, 0.0]],
        [[10.0, 20.0], [0.0, 0.0]],
        point_mask=[[True, True], [False, False]],
        scan_mask=[True, False],
        scan_ids=[42, 0],
        ms_levels=[2, 0],
        retention_time=[5.0, 0.0],
        polarity=IonPolarity.POSITIVE,
    )
    precursors = PrecursorBatch(
        [[500.2, 501.2], [0.0, 0.0]],
        [[2, 3], [0, 0]],
        [[0.5, 0.5], [0.0, 0.0]],
        [[0.5, 0.5], [0.0, 0.0]],
        [[30.0, 30.0], [0.0, 0.0]],
        [[True, True], [False, False]],
    )
    run = AcquisitionRun(
        spectra,
        precursors,
        AcquisitionMetadata(AcquisitionKind.DATA_DEPENDENT),
    )
    found = lookup_spectrum(run, 42)
    missing = lookup_spectrum(run, 99)
    assert bool(np.asarray(found.valid))
    assert int(np.asarray(found.evidence)) & int(AcquisitionEvidence.CHIMERIC_PRECURSOR)
    assert not bool(np.asarray(missing.valid))
    assert int(np.asarray(missing.scan_index)) == -1
    assert not np.any(np.asarray(missing.active_mask))

    record = {
        "index": 42,
        "ms level": 2,
        "centroid spectrum": "",
        "positive scan": "",
        "m/z array": np.asarray([100.0, 200.0]),
        "intensity array": np.asarray([10.0, 20.0]),
        "scanList": {"scan": [{"scan start time": 5.0}]},
        "precursorList": {
            "precursor": [
                {
                    "selectedIonList": {
                        "selectedIon": [
                            {"selected ion m/z": 500.2, "charge state": 2},
                            {"selected ion m/z": 501.2, "charge state": 3},
                        ]
                    },
                    "isolationWindow": {
                        "isolation window lower offset": 0.5,
                        "isolation window upper offset": 0.5,
                    },
                    "activation": {"collision energy": 30.0},
                }
            ]
        },
    }
    lowered = lower_mzml_record(
        record,
        MzMLLoweringPlan(scan_capacity=1, point_capacity=2, precursor_capacity=2),
    )
    assert bool(np.asarray(lowered.valid))
    assert int(np.asarray(lowered.precursors.active_mask.sum())) == 2
    overflow = lower_mzml_record(
        record,
        MzMLLoweringPlan(scan_capacity=1, point_capacity=1, precursor_capacity=2),
    )
    assert int(np.asarray(overflow.status)) == int(MzMLReadStatus.POINT_CAPACITY_EXCEEDED)
    assert not np.any(np.asarray(overflow.spectrum.active_mask))


def test_mass_calibration_extrapolation_is_not_silent() -> None:
    model = mass_calibration_model([0.1], 100.0, 200.0)
    spectrum = MassSpectrum([99.0, 150.0], [1.0, 1.0])
    result = apply_mass_calibration(spectrum, model)
    np.testing.assert_array_equal(result.extrapolated_mask, [True, False])
    np.testing.assert_allclose(result.calibrated_mass_to_charge, [99.1, 150.1])
    assert int(np.asarray(result.status)) == int(CalibrationStatus.EXTRAPOLATION)
    assert not bool(np.asarray(result.valid))


def test_peptide_modification_charge_decoys_and_psm_score() -> None:
    sequences = _protein_sequences([7], ["ACD"], 4)
    modifications = ModificationBatch(
        [[10, 0]],
        [[0, 0]],
        [[int(ModificationSiteKind.RESIDUE), 0]],
        [[15.9949, 0.0]],
        [[True, False]],
    )
    candidates = PeptideCandidateBatch(
        sequences,
        [3],
        [350.0],
        [False],
        modifications,
    )
    decoy_result = reverse_peptide_decoys(candidates)
    decoy = decoy_result.decoys
    expected = [PROTEIN_IUPAC.code(symbol) for symbol in "DCA"]
    np.testing.assert_array_equal(decoy.sequences.token_codes[0, :3], expected)
    assert int(np.asarray(decoy.charge[0])) == 3
    assert int(np.asarray(decoy.modifications.positions[0, 0])) == 2
    assert int(np.asarray(decoy.sequences.record_ids[0])) == -8

    inputs = PSMScoringInput(
        [1], [42], [0], [3], [2.0], [6], [10], [0.8], [0.9], [False], [True]
    )
    scored = score_psms(inputs, PSMScoringPlan(precursor_tolerance_ppm=10.0))
    np.testing.assert_allclose(scored.matched_ion_fraction, [0.6])
    np.testing.assert_allclose(scored.score, [2.26], rtol=1.0e-6)
    assert bool(np.asarray(scored.valid))


def test_target_decoy_competition_at_all_levels_and_shared_peptides() -> None:
    for level in CompetitionLevel:
        batch = TargetDecoyCompetitionBatch(
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0],
            [10, 11, 12, 13, 14, 15],
            [0, 0, 1, 1, 2, 2],
            [False, True, True, False, False, True],
            [True, True, True, True, True, True],
            level=level,
            fdr_group_ids=[0, 0, 0, 0, 1, 1],
        )
        result = target_decoy_competition(batch, decoy_pseudocount=0)
        np.testing.assert_array_equal(
            result.winner_mask, [True, False, True, False, True, False]
        )
        np.testing.assert_allclose(np.asarray(result.q_value)[[0, 4]], [0.0, 0.0])
        assert result.level is level

    relation = ProteinPeptideRelation(
        [100, 200, 300],
        [[10, 11], [11, 12], [11, 12]],
        [[True, True], [True, True], [True, True]],
    )
    peptide_evidence = PeptideEvidenceBatch(
        [10, 11, 12], [5.0, 3.0, 2.0], [0.01, 0.02, 0.03], [True, True, True]
    )
    inferred = infer_proteins_from_shared_peptides(relation, peptide_evidence)
    np.testing.assert_array_equal(inferred.shared_peptide_mask, [False, True, True])
    np.testing.assert_array_equal(inferred.razor_protein_index, [0, 0, 1])
    assert int(np.asarray(inferred.group_ids[1])) == int(
        np.asarray(inferred.group_ids[2])
    )
    np.testing.assert_array_equal(inferred.representative_mask, [True, True, False])


def test_adduct_isotope_ambiguity_and_candidate_confidence_levels() -> None:
    feature_batch = LCMSFeatureBatch(
        [1],
        [101.007276],
        [60.0],
        [1000.0],
        [1],
        polarity=IonPolarity.POSITIVE,
    )
    isotope_batch = IsotopeEnvelopeBatch([[0.0, 1.003355]], [[1.0, 0.1]], [[True, True]])
    features = MetaboliteFeatureBatch(feature_batch, isotope_batch)
    formulas = FormulaCandidateBatch(
        [10, 11],
        [[5, 10], [6, 8]],
        [6, 1],
        [100.0, 100.0],
        [1.0, 2.0],
        [[0.0, 1.003355], [0.0, 1.003355]],
        [[1.0, 0.1], [1.0, 0.1]],
        [[True, True], [True, True]],
        [True, True],
    )
    adducts = AdductBatch([1], [1], [1.007276], [1], [int(IonPolarity.POSITIVE)], [True])
    matches = match_metabolite_candidates(
        features,
        formulas,
        adducts,
        MetaboliteMatchingPlan(
            mass_tolerance_ppm=5.0,
            isotope_mass_tolerance=0.001,
            minimum_isotope_cosine=0.99,
            candidate_capacity=2,
        ),
    )
    np.testing.assert_array_equal(matches.matched_mask, [True, True])
    np.testing.assert_array_equal(matches.ambiguous_mask, [True, True])
    confidence = classify_metabolite_confidence(
        matches,
        authenticated_standard_match=[True, False],
        reference_msms_match=[True, True],
    )
    np.testing.assert_array_equal(
        confidence.confidence_level,
        [
            int(MetaboliteConfidenceLevel.AUTHENTICATED_STANDARD),
            int(MetaboliteConfidenceLevel.PUTATIVELY_ANNOTATED),
        ],
    )
    np.testing.assert_array_equal(confidence.ambiguous_mask, [True, True])


def test_blank_run_order_censoring_and_missing_are_not_zero() -> None:
    roles = [
        int(QuantificationSampleKind.BLANK),
        int(QuantificationSampleKind.QUALITY_CONTROL),
        int(QuantificationSampleKind.QUALITY_CONTROL),
        int(QuantificationSampleKind.BIOLOGICAL),
        int(QuantificationSampleKind.BIOLOGICAL),
        int(QuantificationSampleKind.BIOLOGICAL),
    ]
    present = np.asarray(
        [[True, True, True, True, True, False], [True, True, True, True, True, False]]
    )
    censored = np.asarray(
        [
            [False, False, False, False, True, False],
            [False, False, False, False, True, False],
        ]
    )
    batch = QuantificationBatch(
        [1, 2],
        [10, 11, 12, 13, 14, 15],
        [[10.0, 20.0, 30.0, 40.0, 0.0, 0.0], [10.0, 20.0, 30.0, 5.0, 0.0, 0.0]],
        [[1.0, 1.0, 1.0, 1.0, 20.0, 0.0], [1.0, 1.0, 1.0, 1.0, 20.0, 0.0]],
        present,
        censored,
        [True, True],
        [True, True, True, True, True, True],
        roles,
        [0, 1, 2, 3, 4, 5],
        [0, 0, 0, 7, 7, 7],
        [0, 0, 0, 0, 0, 0],
    )
    result = quantify_replicates(
        batch,
        QuantificationPlan([7], [True], minimum_blank_count=1, minimum_qc_count=2),
    )
    np.testing.assert_allclose(result.blank_estimate, [10.0, 10.0])
    assert int(np.asarray(result.evidence)) & int(
        QuantificationEvidence.RUN_ORDER_CORRECTED
    )
    assert bool(np.asarray(result.point_estimate_mask[1, 3]))
    assert float(np.asarray(result.corrected_intensity[1, 3])) == 0.0
    assert np.isnan(np.asarray(result.corrected_intensity[0, 4]))
    assert float(np.asarray(result.lower_bound[0, 4])) == 0.0
    assert float(np.asarray(result.upper_bound[0, 4])) > 0.0
    assert np.isnan(np.asarray(result.corrected_intensity[0, 5]))
    assert bool(np.asarray(result.missing_mask[0, 5]))
    assert int(np.asarray(result.replicate_observed_count[0, 0])) == 1
    assert int(np.asarray(result.replicate_censored_count[0, 0])) == 1
    assert int(np.asarray(result.replicate_missing_count[0, 0])) == 1
    assert np.isnan(np.asarray(result.replicate_lower_bound[0, 0]))
