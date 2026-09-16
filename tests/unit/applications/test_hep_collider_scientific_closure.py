#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_native_sequential_and_fuzzy_jets_preserve_constituents():
    analysis = phx.applications.collider_analysis
    inputs = analysis.JetInputBatch(
        jnp.asarray([1]),
        jnp.asarray(
            [[[10.0, 6.0, 0.0, 8.0], [5.0, 3.0, 0.0, 4.0], [4.0, -4.0, 0.0, 0.0]]]
        ),
        jnp.asarray([[True, True, True]]),
        source_collection_id="particles",
        momentum_unit_id="GeV",
    )
    jets = analysis.cluster_sequential_jets(
        analysis.JetDefinition(analysis.JetAlgorithm.ANTI_KT, 0.5),
        inputs,
    )
    assert int(jnp.sum(jets.active)) == 2
    assert jnp.allclose(
        jnp.sum(jets.momenta[0], axis=0),
        jnp.sum(inputs.momenta[0], axis=0),
    )
    assert jnp.allclose(jnp.sum(jets.constituent_weights[0], axis=0), jnp.ones(3))

    fuzzy = analysis.cluster_fuzzy_jets(
        analysis.FuzzyJetPlan(2, 0.6, iteration_count=8),
        inputs,
    )
    assert fuzzy.responsibilities.shape == (1, 2, 3)
    assert jnp.all(fuzzy.pileup_responsibility >= 0.0)
    assert jnp.isfinite(fuzzy.log_likelihood[0])


def test_trigger_menu_buffer_and_fragment_replay_are_fail_closed():
    trigger = phx.applications.trigger_replay
    fragments = (
        trigger.RawFragmentRecord(
            1, "A", 10, 3, 100, "a", 4, trigger.FragmentStatus.COMPLETE
        ),
        trigger.RawFragmentRecord(
            1, "B", 10, 3, 101, "b", 5, trigger.FragmentStatus.COMPLETE
        ),
        trigger.RawFragmentRecord(
            2, "A", 10, 4, 110, "c", 4, trigger.FragmentStatus.COMPLETE
        ),
    )
    built = trigger.build_fragment_events(
        trigger.EventBuildingPlan(("A", "B"), maximum_timestamp_spread=2),
        fragments,
    )
    assert built[0].status is trigger.EventBuildStatus.COMPLETE
    assert built[1].status is trigger.EventBuildStatus.MISSING_SOURCE

    menu = trigger.TriggerMenu(
        (
            trigger.TriggerLine(
                "seed",
                stream_names=("calibration",),
                maximum_latency=1.0,
                maximum_resource_units=1.0,
            ),
            trigger.TriggerLine(
                "physics",
                seed_names=("seed",),
                stream_names=("physics",),
                maximum_latency=2.0,
                maximum_resource_units=2.0,
            ),
        )
    )
    replay = trigger.replay_trigger_menu(
        menu,
        jnp.asarray([1, 2, 3]),
        jnp.asarray([[True, True], [False, True], [True, False]]),
        measured_latency=jnp.zeros((3, 2)),
        measured_resource_units=jnp.zeros((3, 2)),
    )
    assert jnp.array_equal(replay.post_prescale[:, 1], jnp.asarray([True, False, False]))
    buffer = trigger.replay_trigger_buffer(
        trigger.TriggerBufferPlan(3, 1),
        jnp.asarray([2, 3, 0, 0]),
    )
    assert int(buffer.total_dropped) == 1
    assert bool(buffer.valid)


def test_collider_corrections_backgrounds_histograms_and_eft():
    analysis = phx.applications.collider_analysis
    correction = analysis.CorrectionMap(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([1.0, 1.1, 1.2]),
        jnp.asarray([0.01, 0.02, 0.03]),
        input_name="pt",
        input_unit_id="GeV",
        output_name="scale",
        correlation_id="jes",
        source_id="test",
    )
    corrected = analysis.apply_correction(
        correction, jnp.asarray([0.5, 1.5]), jnp.asarray([10.0, 10.0])
    )
    assert jnp.allclose(corrected.corrected_values, jnp.asarray([10.5, 11.5]))

    background = analysis.estimate_abcd_background(
        analysis.ABCDBackgroundPlan(0.1, correlation_id="abcd"),
        jnp.asarray([20.0]),
        jnp.asarray([10.0]),
        jnp.asarray([5.0]),
    )
    assert jnp.isclose(background.prediction[0], 40.0)
    assert bool(background.valid[0])

    histogram = analysis.fill_multidimensional_histogram(
        analysis.MultiHistogramPlan(
            (jnp.asarray([0.0, 1.0, 2.0]), jnp.asarray([0.0, 5.0, 10.0])),
            axis_names=("x", "y"),
            unit_ids=("1", "GeV"),
        ),
        jnp.asarray([[0.5, 2.0], [1.5, 7.0]]),
        jnp.asarray([1.0, -0.5]),
    )
    assert jnp.allclose(histogram.sum_weights, jnp.asarray([[1.0, 0.0], [0.0, -0.5]]))

    theory = phx.applications.collider_theory
    eft = theory.EFTMorphingPlan(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([[1.0, 0.0]]),
        jnp.asarray([[[0.5, 0.25]]]),
        jnp.asarray([-1.0]),
        jnp.asarray([1.0]),
        coefficient_names=("c1",),
        observable_names=("bin0", "bin1"),
        source_prediction_id="grid",
    )
    morphed = theory.evaluate_eft_morphing(eft, jnp.asarray([0.5]))
    assert bool(morphed.valid)
    assert jnp.allclose(morphed.values, jnp.asarray([1.625, 2.0625]))
