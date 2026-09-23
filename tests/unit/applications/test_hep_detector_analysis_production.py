#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
from io import BytesIO

import h5py
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _geometry_and_conditions():
    detector = phx.applications.detector
    conditions = detector.DetectorConditions(
        magnetic_field=jnp.zeros(3),
        electric_field=jnp.zeros(3),
        momentum_unit=phx.units.GIGAELECTRONVOLT,
        length_unit=phx.units.MILLIMETER,
        time_unit=phx.units.NANOSECOND,
        geometry_id="test-geometry",
        material_id="test-material",
        field_id="zero",
        alignment_id="nominal",
        calibration_id="test",
        validity_interval=(0, 10),
    )
    geometry = detector.calorimetry.CalorimeterGeometry(
        cell_ids=jnp.asarray([10, 11]),
        channel_ids=jnp.asarray([0, 1]),
        layer_ids=jnp.asarray([0, 1]),
        subdetector_ids=jnp.zeros(2, dtype=jnp.int32),
        material_ids=jnp.zeros(2, dtype=jnp.int32),
        readout_ids=jnp.zeros(2, dtype=jnp.int32),
        centroids=jnp.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 2.0]]),
        volumes=jnp.ones(2),
        active=jnp.ones(2, dtype="bool"),
        dead=jnp.asarray([False, True]),
        senders=jnp.asarray([0, 1]),
        receivers=jnp.asarray([1, 0]),
        conditions_id=conditions.conditions_id,
    )
    return conditions, geometry


def test_calorimeter_energy_ledger_keeps_dead_and_unknown_leakage_distinct():
    detector = phx.applications.detector
    conditions, geometry = _geometry_and_conditions()
    hits = detector.SensitiveHitBank(
        event_ids=jnp.asarray([1]),
        hit_ids=jnp.asarray([[0, 1, 2]]),
        detector_element_ids=jnp.asarray([[0, 1, 2]]),
        channel_ids=jnp.asarray([[0, 1, 99]]),
        source_step_indices=jnp.asarray([[0, 1, 2]]),
        positions=jnp.zeros((1, 3, 3)),
        times=jnp.zeros((1, 3)),
        energies=jnp.asarray([[2.0, 3.0, 4.0]]),
        active=jnp.ones((1, 3), dtype="bool"),
        conditions_id=conditions.conditions_id,
    )
    truth = detector.calorimetry.route_calorimeter_hits(
        geometry,
        hits,
        jnp.asarray([10.0]),
        jnp.asarray([9.0]),
        source_id="ledger-test",
    )
    assert bool(truth.successful)
    assert jnp.isclose(truth.cell_energies[0, 0], 2.0)
    assert jnp.isclose(truth.dead_energy[0], 3.0)
    assert jnp.isclose(truth.unmapped_energy[0], 4.0)
    assert not bool(truth.leakage_known[0])
    assert jnp.isnan(truth.leakage_energy[0])


def test_calorimeter_response_and_observables_are_conditioned_on_live_cells():
    detector = phx.applications.detector
    conditions, geometry = _geometry_and_conditions()
    hits = detector.SensitiveHitBank(
        event_ids=jnp.asarray([3]),
        hit_ids=jnp.asarray([[0]]),
        detector_element_ids=jnp.asarray([[0]]),
        channel_ids=jnp.asarray([[0]]),
        source_step_indices=jnp.asarray([[0]]),
        positions=jnp.zeros((1, 1, 3)),
        times=jnp.zeros((1, 1)),
        energies=jnp.asarray([[2.0]]),
        active=jnp.ones((1, 1), dtype="bool"),
        conditions_id=conditions.conditions_id,
    )
    truth = detector.calorimetry.route_calorimeter_hits(
        geometry,
        hits,
        jnp.asarray([2.0]),
        jnp.asarray([2.0]),
        leakage_energy=jnp.asarray([0.0]),
        leakage_known=jnp.asarray([True]),
        source_id="response-test",
    )
    response = detector.calorimetry.apply_calorimeter_response(
        detector.calorimetry.CalorimeterResponsePlan(
            geometry,
            gain=jnp.ones(2),
            noise_standard_deviation=jnp.zeros(2),
            crosstalk=jnp.zeros((2, 2)),
            adc_lsb=0.1,
            threshold=0.0,
            maximum_adc=100,
        ),
        truth,
        jr.key(4),
    )
    observables = detector.calorimetry.calorimeter_observables(
        geometry, response.reconstructed_cell_energy
    )
    assert jnp.isclose(observables.total_energy[0], 2.0)
    assert int(observables.occupancy[0]) == 1
    assert not bool(response.digits.active[0, 1])
    other_geometry = detector.calorimetry.CalorimeterGeometry(
        cell_ids=jnp.asarray([10, 11]),
        channel_ids=jnp.asarray([0, 1]),
        layer_ids=jnp.asarray([0, 1]),
        subdetector_ids=jnp.zeros(2, dtype=jnp.int32),
        material_ids=jnp.zeros(2, dtype=jnp.int32),
        readout_ids=jnp.zeros(2, dtype=jnp.int32),
        centroids=geometry.centroids + 1.0,
        volumes=jnp.ones(2),
        active=jnp.ones(2, dtype="bool"),
        dead=jnp.asarray([False, True]),
        senders=jnp.asarray([0, 1]),
        receivers=jnp.asarray([1, 0]),
        conditions_id=conditions.conditions_id,
    )
    clustering = detector.calorimetry.CalorimeterClusteringPlan(
        other_geometry,
        jnp.asarray([0, -1]),
        cluster_capacity=1,
    )
    with pytest.raises(ValueError, match="geometry"):
        detector.calorimetry.reconstruct_calorimeter_clusters(clustering, response)


def test_corpus_split_and_sparse_velocity_preserve_geometry_support():
    detector = phx.applications.detector
    _, geometry = _geometry_and_conditions()
    payload = b"calorimeter-corpus"
    manifest = phx.qualification.ReferenceArtifactManifest(
        "tiny-corpus",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="test",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted-test",
        nondimensionalization={"energy": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )
    corpus = detector.calorimetry.prepare_calorimeter_corpus(
        geometry,
        jnp.asarray([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
        jnp.asarray([[1.0], [2.0], [3.0], [4.0]]),
        condition_names=("incident_energy",),
        ledger=jnp.asarray(
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
        record_ids=("a", "b", "c", "d"),
        source_manifest_ids=(manifest.manifest_id,) * 4,
        split_group_ids=("train-a", "train-b", "validation", "test"),
        validation_groups=("validation",),
        test_groups=("test",),
        rights=(manifest,),
    )
    model = detector.calorimetry.ConditionalCalorimeterVelocity(
        geometry,
        corpus.condition_names,
        width=4,
        depth=1,
        key=jr.key(5),
    )
    velocity = model(jnp.asarray([0.2, 0.0]), jnp.asarray(0.5), jnp.asarray([2.0]))
    assert velocity.shape == (2,)
    assert velocity[1] == 0.0


def test_fixed_association_track_fit_recovers_observable_trajectory():
    detector = phx.applications.detector
    times = jnp.asarray([0.0, 1.0, 2.0])
    intercept = jnp.asarray([1.0, -2.0, 0.5])
    velocity = jnp.asarray([0.1, 0.2, -0.3])
    positions = intercept[None, :] + times[:, None] * velocity[None, :]
    measurements = detector.TrackMeasurementBank(
        event_ids=jnp.asarray([5]),
        positions=positions[None, None, :, :],
        times=times[None, None, :],
        variances=jnp.full((1, 1, 3, 3), 1.0e-4),
        surface_ids=jnp.asarray([[[0, 1, 2]]]),
        active=jnp.ones((1, 1, 3), dtype="bool"),
        association_id="fixed-test",
        conditions_id="conditions-test",
    )
    fitted = detector.fit_associated_tracks(
        detector.TrackFitPlan(
            association_id="fixed-test",
            conditions_id="conditions-test",
            regularization=1.0e-14,
        ),
        measurements,
    )
    assert bool(fitted.valid[0, 0])
    assert jnp.allclose(fitted.parameters[0, 0, :3], intercept, atol=1.0e-10)
    assert jnp.allclose(fitted.parameters[0, 0, 3:], velocity, atol=1.0e-10)
    assert fitted.chi_square[0, 0] < 1.0e-16
    particles = detector.particles_from_straight_tracks(
        fitted,
        pdg_hypothesis=13,
        rest_energy=4.0,
        charge=-1.0,
        speed_of_light=2.0,
    )
    speed_squared = jnp.sum(velocity * velocity)
    gamma = 1.0 / jnp.sqrt(1.0 - speed_squared / 4.0)
    np.testing.assert_allclose(
        particles.momenta[0, 0],
        4.0 * gamma * velocity / 4.0,
        rtol=1e-12,
    )
    np.testing.assert_allclose(particles.energies[0, 0], 4.0 * gamma, rtol=1e-12)


def test_detector_conditions_identity_and_energy_loss_are_enforced():
    detector = phx.applications.detector
    conditions, _ = _geometry_and_conditions()
    digits = detector.DigitBank(
        event_ids=jnp.asarray([1]),
        digit_ids=jnp.asarray([[0]]),
        channel_ids=jnp.asarray([[0]]),
        signals=jnp.asarray([[1.0]]),
        times=jnp.asarray([[0.0]]),
        active=jnp.asarray([[True]]),
        saturated=jnp.asarray([[False]]),
        conditions_id=conditions.conditions_id,
    )
    interval = phx.measurement.OperationalInterval(
        phx.measurement.OperationalCoordinate("detector", {"run": 1}),
        phx.measurement.OperationalCoordinate("detector", {"run": 2}),
    )
    payload = detector.DetectorCalibrationPayload(
        jnp.asarray([0]),
        jnp.asarray([1.0]),
        jnp.asarray([0.0]),
        jnp.eye(2),
        interval,
        conditions_snapshot_id="different-conditions",
        authority=detector.CalibrationAuthority.CANDIDATE,
        source_id="test",
    )
    with pytest.raises(ValueError, match="conditions snapshot"):
        detector.apply_detector_calibration(payload, digits)

    tracks = detector.TransportTrackBank(
        event_ids=jnp.asarray([1]),
        track_ids=jnp.asarray([[0]]),
        parent_track_ids=jnp.asarray([[-1]]),
        pdg_ids=jnp.asarray([[13]]),
        positions=jnp.zeros((1, 1, 3)),
        momenta=jnp.asarray([[[3.0, 0.0, 0.0]]]),
        rest_energies=jnp.asarray([[4.0]]),
        charges=jnp.asarray([[0.0]]),
        active=jnp.asarray([[True]]),
        conditions_id=conditions.conditions_id,
    )
    propagated = detector.propagate_charged_tracks(
        detector.ChargedPropagationPlan(
            conditions,
            step_size=1.0,
            step_count=1,
            speed_of_light=1.0,
            mean_energy_loss_per_length=1.0,
        ),
        tracks,
    )
    distance = jnp.linalg.norm(propagated.position_history[0, 0, 0])
    final_momentum = jnp.linalg.norm(propagated.tracks.momenta[0, 0])
    final_energy = jnp.sqrt(final_momentum**2 + 4.0**2)
    np.testing.assert_allclose(final_energy, 5.0 - distance, rtol=1e-6)


def test_weighted_analysis_retains_negative_bins_and_nested_cutflow():
    analysis = phx.applications.collider_analysis
    weights = phx.particle_physics.EventWeightSet(
        jnp.asarray([[1.0], [-2.0], [0.5]]),
        names=("nominal",),
        variation_kinds=(phx.particle_physics.WeightVariationKind.NOMINAL,),
        correlation_groups=("nominal",),
    )
    histogram = analysis.fill_weighted_histogram(
        analysis.HistogramPlan(
            jnp.asarray([0.0, 1.0, 2.0]), observable_id="x", unit_id="1"
        ),
        jnp.asarray([0.2, 0.8, 1.2]),
        weights.nominal,
    )
    assert jnp.allclose(histogram.sum_weights, jnp.asarray([-1.0, 0.5]))
    cutflow = analysis.build_cutflow(
        weights,
        jnp.asarray([[True, True], [True, False], [False, False]]),
        cut_names=("preselection", "signal"),
    )
    assert bool(cutflow.nested)
    assert jnp.allclose(cutflow.sum_weights, jnp.asarray([-1.0, 1.0]))


def test_calochallenge_profile_admits_pinned_resident_hdf5_arrays():
    _, geometry = _geometry_and_conditions()
    stream = BytesIO()
    with h5py.File(stream, "w") as handle:
        handle.create_dataset("incident_energies", data=jnp.asarray([[2.0], [3.0]]))
        handle.create_dataset("showers", data=jnp.asarray([[2.0, 0.0], [3.0, 0.0]]))
    payload = stream.getvalue()
    manifest = phx.qualification.ReferenceArtifactManifest(
        "calochallenge-test",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="test",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted-test",
        nondimensionalization={"energy": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )
    imported = phx.interchange.hep.import_calochallenge_hdf5(
        payload,
        manifest,
        phx.interchange.hep.CaloChallengeProfile(
            "incident_energies",
            "showers",
            geometry.geometry_id,
            "GeV",
            8,
        ),
        geometry,
        training_use=True,
    )
    assert imported.report.valid
    assert jnp.allclose(imported.incident_energies, jnp.asarray([2.0, 3.0]))
    assert jnp.allclose(imported.showers[:, 0], imported.incident_energies)


def test_likelihood_and_hepdata_adapters_preserve_admitted_bin_semantics():
    analysis = phx.applications.collider_analysis
    likelihood = analysis.BinnedLikelihoodPlan(
        jnp.asarray([10.0, 20.0]),
        jnp.asarray([[1.0, -2.0]]),
        jnp.asarray([1.0]),
        nuisance_names=("shape",),
        channel_names=("low", "high"),
    )
    workspace = phx.interchange.hep.binned_likelihood_to_pyhf_workspace(
        likelihood,
        jnp.asarray([11.0, 19.0]),
        parameter_of_interest="shape",
    )
    modifier = workspace["channels"][0]["samples"][0]["modifiers"][0]
    assert modifier["data"]["hi_data"] == [11.0, 18.0]
    assert modifier["data"]["lo_data"] == [9.0, 22.0]

    plan = analysis.HistogramPlan(
        jnp.asarray([0.0, 1.0, 2.0]),
        observable_id="energy",
        unit_id="GeV",
    )
    histogram = analysis.fill_weighted_histogram(
        plan,
        jnp.asarray([0.25, 1.25]),
        jnp.asarray([2.0, -0.5]),
    )
    table = phx.interchange.hep.weighted_histogram_to_hepdata(
        plan,
        histogram,
        dependent_name="events",
        qualifiers={"SQRT(S)": "13 TeV"},
    )
    assert table["dependent_variables"][0]["values"][1]["value"] == -0.5
    assert table["independent_variables"][0]["values"][0] == {
        "low": 0.0,
        "high": 1.0,
    }
