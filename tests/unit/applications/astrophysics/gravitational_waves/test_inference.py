import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np

import phydrax as phx


def test_phase_marginalization_matches_explicit_periodic_quadrature_and_reconstructs(
    wave_problem,
):
    gw, _, _, _, injected, likelihood = wave_problem()
    phase = gw.PhaseMarginalizationPlan(reconstruction_nodes=128)
    plan = gw.GravitationalWaveMarginalizationPlan(likelihood, phase=phase)
    marginalized_parameters = dict(injected)
    marginalized_parameters.pop("phase")
    nodes = jnp.linspace(0.0, 2.0 * jnp.pi, 2048, endpoint=False)
    explicit = jsp.special.logsumexp(
        jax.vmap(
            lambda value: likelihood.log_probability(
                {**marginalized_parameters, "phase": value}
            )
        )(nodes)
    ) - jnp.log(float(nodes.size))

    np.testing.assert_allclose(
        plan.log_probability(marginalized_parameters), explicit, atol=2e-6
    )
    samples = {
        name: jnp.stack((value, value)) for name, value in marginalized_parameters.items()
    }
    first = gw.reconstruct_marginalized_parameters(
        plan, jr.key(9), samples, sample_ndim=1
    )
    second = gw.reconstruct_marginalized_parameters(
        plan, jr.key(9), samples, sample_ndim=1
    )
    assert first.samples["phase"].shape == (2,)
    assert jnp.array_equal(first.samples["phase"], second.samples["phase"])
    assert bool(jnp.all(first.valid))


def test_distance_and_time_marginalization_match_declared_discrete_measure(
    wave_problem,
):
    gw, _, _, _, injected, likelihood = wave_problem()
    distance = gw.DistanceMarginalizationPlan(
        0.8,
        1.2,
        phx.uq.Uniform(0.8, 1.2),
        order=24,
    )
    time = gw.TimeMarginalizationPlan(0.0, 0.04, num_nodes=5)
    plan = gw.GravitationalWaveMarginalizationPlan(
        likelihood,
        distance=distance,
        time=time,
    )
    parameters = dict(injected)
    parameters.pop("luminosity_distance")
    parameters.pop("geocent_time")
    manual = []
    for distance_index in range(distance.nodes.size):
        for time_index in range(time.nodes.size):
            manual.append(
                likelihood.log_probability(
                    {
                        **parameters,
                        "luminosity_distance": distance.nodes[distance_index],
                        "geocent_time": time.nodes[time_index],
                    }
                )
                + distance.log_weights[distance_index]
                + time.log_weights[time_index]
            )
    expected = jsp.special.logsumexp(jnp.stack(manual))
    result = plan.evaluate(parameters)

    np.testing.assert_allclose(result.log_probability, expected, atol=1e-12)
    assert result.evaluated_points == distance.nodes.size * time.nodes.size
    assert bool(result.valid)


def test_calibration_marginalization_matches_declared_response_ensemble(wave_problem):
    gw, provenance, _, _, injected, likelihood = wave_problem()
    responses = jnp.stack(
        (
            jnp.ones_like(likelihood.network.strain),
            1.05 * jnp.ones_like(likelihood.network.strain),
        )
    ).astype(complex)
    ensemble = gw.CalibrationResponseEnsemble(
        likelihood.network.frequency,
        responses,
        likelihood.network.detector_ids,
        provenance,
        convention="template",
        log_weights=jnp.log(jnp.asarray([0.4, 0.6])),
    )
    calibrated = gw.GravitationalWaveLikelihoodPlan(
        likelihood.network,
        likelihood.response,
        likelihood.waveform,
        likelihood.waveform_parameters_fn,
        likelihood.extrinsic_parameters_fn,
        parameterization_id=likelihood.parameterization_id,
        calibration_id=ensemble.ensemble_id,
        calibration=lambda values, frequency: ensemble.response(
            values["calibration_index"], frequency
        ),
    )
    plan = gw.GravitationalWaveMarginalizationPlan(
        calibrated,
        calibration=gw.CalibrationMarginalizationPlan(ensemble),
    )
    manual = jsp.special.logsumexp(
        jnp.stack(
            tuple(
                calibrated.log_probability(
                    {**injected, "calibration_index": jnp.asarray(index)}
                )
                + ensemble.log_weights[index]
                for index in range(ensemble.count)
            )
        )
    )

    np.testing.assert_allclose(plan.log_probability(injected), manual, atol=1e-12)


def test_relative_binning_is_exact_at_fiducial_and_qualified_nearby(wave_problem):
    gw, _, _, _, injected, likelihood = wave_problem(sample_count=128)
    nearby = {**injected, "center_frequency": jnp.asarray(10.05)}
    qualified = gw.prepare_relative_binning_likelihood(
        likelihood,
        injected,
        (injected, nearby),
        ("fiducial", "nearby"),
        num_bins=12,
        policy=gw.LikelihoodApproximationPolicy(0.5, 0.4),
    )

    np.testing.assert_allclose(
        qualified.log_probability(injected),
        likelihood.log_probability(injected),
        atol=1e-10,
    )
    assert qualified.qualification.passed
    assert qualified.qualification.maximum_absolute_error <= 0.5


def test_multiband_stride_one_is_an_exact_prepared_route(wave_problem):
    gw, _, _, _, injected, likelihood = wave_problem()
    active_frequency = likelihood.network.frequency[
        jnp.any(likelihood.network.active, axis=0)
    ]
    qualified = gw.prepare_multiband_likelihood(
        likelihood,
        (float(active_frequency[0]), float(active_frequency[-1])),
        (1,),
        (injected,),
        ("injected",),
        policy=gw.LikelihoodApproximationPolicy(1e-10, 1e-10),
    )

    np.testing.assert_allclose(
        qualified.log_probability(injected),
        likelihood.log_probability(injected),
        atol=1e-10,
    )
    assert qualified.qualification.passed


def test_full_rank_roq_preserves_exact_likelihood(wave_problem):
    gw, _, _, _, injected, likelihood = wave_problem(sample_count=32)
    size = int(likelihood.network.frequency.size)
    basis = np.eye(size)
    artifact = phx.rom.ROMArtifact(
        "full-frequency-corpus",
        "full-frequency-validity",
        phx.rom.LinearPODProfile(size),
        basis,
        np.ones(size),
        basis,
        tuple(f"basis-{index}" for index in range(size)),
        phx.lifecycle.NumericRevision("0" * 64, label="full-frequency-test"),
    )
    interpolation = phx.rom.prepare_empirical_interpolation(artifact).prepare()
    qualified = gw.prepare_reduced_order_quadrature_likelihood(
        likelihood,
        interpolation,
        interpolation,
        (injected,),
        ("injected",),
        policy=gw.LikelihoodApproximationPolicy(1e-10, 1e-10),
    )

    np.testing.assert_allclose(
        qualified.log_probability(injected),
        likelihood.log_probability(injected),
        atol=1e-10,
    )
    assert qualified.qualification.passed


def test_parameter_plan_identity_includes_prior_values():
    gw = phx.applications.astrophysics.gravitational_waves

    def plan(high):
        return gw.GravitationalWaveParameterPlan(
            {"amplitude": jnp.asarray(0.5)},
            {"amplitude": phx.uq.Uniform(0.1, high)},
            continuous_paths=("['amplitude']",),
            waveform_parameters=lambda values: values,
            extrinsic_parameters=lambda _: (
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
            ),
            parameterization_id="prior-identity-test",
        )

    assert plan(1.0).plan_id != plan(2.0).plan_id
