import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import phydrax as phx


def _measure():
    return phx.integration.WeightedSampleTarget(
        {"x": jnp.asarray([-1.0, 0.0, 1.0])},
        jnp.zeros(3),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance="three-point-posterior",
    )


def test_posterior_reweighting_preserves_raw_weighted_measure_and_overlap():
    plan = phx.uq.PosteriorReweightingPlan(
        lambda sample: jnp.asarray(0.0),
        lambda sample: sample["x"],
        old_target_id="old",
        new_target_id="new",
        policy=phx.uq.PosteriorReweightingPolicy(1.0, 0.1),
    )
    result = phx.uq.reweight_posterior(_measure(), plan)
    expected_log_ratio = jsp.special.logsumexp(jnp.asarray([-1.0, 0.0, 1.0])) - jnp.log(
        3.0
    )

    np.testing.assert_allclose(result.log_normalizer_ratio, expected_log_ratio)
    np.testing.assert_allclose(
        jsp.special.logsumexp(result.target.log_weights), 0.0, atol=2e-15
    )
    assert bool(result.valid)
    assert result.target.samples["x"].shape == (3,)


def test_population_recycling_and_selection_match_manual_event_integrals():
    event = phx.uq.EventPosterior(
        _measure(),
        jnp.zeros(3),
        event_id="event-0",
        parameterization_id="scalar-x",
        likelihood_id="event-likelihood",
        provider_id="native",
        inference_method="nested",
        approximation="exact",
        source_effective_sample_size=3.0,
    )
    second_event = phx.uq.EventPosterior(
        _measure(),
        jnp.zeros(3),
        event_id="event-1",
        parameterization_id="scalar-x",
        likelihood_id="event-likelihood",
        provider_id="native",
        inference_method="nested",
        approximation="exact",
        source_effective_sample_size=3.0,
    )
    batch = phx.uq.prepare_population_sample_batch((event, second_event))

    def population_log_prob(hyperparameters, sample):
        return -0.5 * (sample["x"] - hyperparameters["mean"]) ** 2 - 0.5 * jnp.log(
            2.0 * jnp.pi
        )

    hyperparameters = {"mean": jnp.asarray(0.0)}
    event_values = population_log_prob(hyperparameters, _measure().samples)
    proposal = event_values
    selection = phx.uq.SelectionInjectionSet(
        _measure().samples,
        proposal,
        jnp.ones(3),
        campaign_id="complete-detection",
        parameterization_id="scalar-x",
    )
    term = phx.uq.PopulationPosteriorTerm(
        batch,
        population_log_prob,
        selection=selection,
        minimum_event_effective_sample_size=1.0,
    )
    expected = jsp.special.logsumexp(event_values) - jnp.log(3.0)
    diagnostics = term.diagnostics(hyperparameters)

    np.testing.assert_allclose(diagnostics.event_log_factors, expected)
    np.testing.assert_allclose(diagnostics.selection.efficiency, 1.0)
    np.testing.assert_allclose(term.per_case_log_prob(hyperparameters), expected)
    assert bool(diagnostics.valid)
    invalid_rate = phx.uq.PoissonPopulationPosteriorTerm(
        batch,
        population_log_prob,
        lambda _: jnp.asarray(-1.0),
        selection,
        minimum_event_effective_sample_size=1.0,
    )
    assert bool(jnp.all(jnp.isneginf(invalid_rate.per_case_log_prob(hyperparameters))))


def test_simulation_calibration_retains_failures_and_uniform_rank_evidence():
    posterior_values = jnp.linspace(0.025, 0.975, 20)
    posterior = phx.integration.WeightedSampleTarget(
        {"x": posterior_values},
        jnp.zeros(20),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance="uniform-reference",
    )
    cases = (
        *(
            phx.uq.SimulationCalibrationCase(
                {"x": value},
                posterior,
                case_id=f"case-{index}",
                analysis_id="rank-campaign",
            )
            for index, value in enumerate(posterior_values)
        ),
        phx.uq.SimulationCalibrationCase(
            {"x": jnp.ones(2)},
            posterior,
            case_id="shape-failure",
            analysis_id="rank-campaign",
        ),
    )
    plan = phx.uq.SimulationCalibrationPlan(
        ("['x']",),
        num_bins=5,
        minimum_valid_cases=20,
    )
    result = phx.uq.simulation_calibration(cases, plan)

    np.testing.assert_array_equal(result.histogram_counts, np.full((1, 5), 4))
    assert result.valid_case_count == 20
    assert bool(result.passed)
    assert result.failed_case_ids == ("shape-failure",)


def test_result_export_binds_explicit_context(tmp_path):
    reweighting = phx.uq.reweight_posterior(
        _measure(),
        phx.uq.PosteriorReweightingPlan(
            lambda _: jnp.asarray(0.0),
            lambda sample: -0.5 * sample["x"] ** 2,
            old_target_id="flat",
            new_target_id="normal",
            policy=phx.uq.PosteriorReweightingPolicy(1.0, 0.1),
        ),
    )
    context = phx.uq.UQResultContext(
        analysis_id="context-test",
        problem_id="problem",
        likelihood_id="likelihood",
        parameterization_id="scalar-x",
        data_ids=("data",),
        provider_ids=("provider",),
        approximation_id="exact",
        normalization="absolute",
    )
    path = phx.uq.export_result(reweighting, tmp_path / "reweighted.phx", context=context)
    archive = phx.uq.read_result_archive(path)

    assert archive.kind == "posterior_reweighting"
    assert archive.metadata["context"]["context_id"] == context.context_id
    assert "log_weights" in archive.fields


def test_population_and_calibration_results_have_portable_archives(tmp_path):
    event_target = phx.integration.WeightedSampleTarget(
        _measure().samples,
        jnp.zeros(3),
        normalized=True,
        independent=False,
        mask=jnp.asarray([True, True, False]),
        sample_axes=0,
        provenance="masked-event-archive",
    )
    event = phx.uq.EventPosterior(
        event_target,
        jnp.zeros(3),
        event_id="event-archive",
        parameterization_id="scalar-x",
        likelihood_id="event-likelihood",
        provider_id="native",
        inference_method="nested",
        approximation="exact",
        source_effective_sample_size=2.0,
    )
    batch = phx.uq.prepare_population_sample_batch((event,))
    injections = phx.uq.SelectionInjectionSet(
        _measure().samples,
        jnp.zeros(3),
        jnp.ones(3),
        campaign_id="archive-selection",
        parameterization_id="scalar-x",
    )
    selection = phx.uq.estimate_selection_efficiency(
        injections,
        {},
        lambda _, __: jnp.asarray(0.0),
    )
    posterior = phx.integration.WeightedSampleTarget(
        {"x": jnp.linspace(0.1, 0.9, 5)},
        jnp.zeros(5),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance="archive-calibration",
    )
    cases = tuple(
        phx.uq.SimulationCalibrationCase(
            {"x": value},
            posterior,
            case_id=f"archive-case-{index}",
            analysis_id="archive-calibration",
        )
        for index, value in enumerate(jnp.linspace(0.1, 0.9, 5))
    )
    calibration = phx.uq.simulation_calibration(
        cases,
        phx.uq.SimulationCalibrationPlan(
            ("['x']",),
            num_bins=5,
            minimum_valid_cases=5,
        ),
    )

    for name, result, expected_kind in (
        ("event", event, "event_posterior"),
        ("batch", batch, "population_sample_batch"),
        ("selection", selection, "selection_efficiency"),
        ("calibration", calibration, "simulation_calibration"),
    ):
        path = phx.uq.export_result(result, tmp_path / f"{name}.phx")
        archive = phx.uq.read_result_archive(path)
        assert archive.kind == expected_kind
        if name == "event":
            assert "mask" in archive.fields
        if name == "calibration":
            assert archive.metadata["analysis_id"] == "archive-calibration"
