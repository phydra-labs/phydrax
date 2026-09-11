import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_exact_network_likelihood_preserves_absolute_and_ratio_semantics(wave_problem):
    _, _, _, _, injected, likelihood = wave_problem()
    evaluation = likelihood.evaluate(injected)
    residual = likelihood.network.strain - evaluation.detector_signal
    direct = jnp.sum(
        jnp.where(
            likelihood.network.active,
            -(jnp.abs(residual) ** 2) / likelihood.network.variance
            - jnp.log(jnp.pi * likelihood.network.variance),
            0.0,
        ),
        axis=-1,
    )

    np.testing.assert_allclose(evaluation.log_probability_by_detector, direct, atol=1e-12)
    np.testing.assert_allclose(
        evaluation.log_probability,
        evaluation.noise_log_probability + evaluation.log_likelihood_ratio,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jnp.where(likelihood.network.active, evaluation.detector_signal, 0.0j),
        likelihood.network.strain,
    )
    assert bool(evaluation.valid)
    assert bool(jnp.all(evaluation.optimal_snr_squared > 0.0))


def test_geocentric_time_applies_common_frequency_domain_translation(wave_problem):
    _, _, _, _, injected, likelihood = wave_problem()
    reference = likelihood.detector_signal(injected)[0]
    shift = jnp.asarray(0.007)
    translated = likelihood.detector_signal(
        {**injected, "geocent_time": injected["geocent_time"] + shift}
    )[0]
    expected = reference * jnp.exp(
        -2.0j * jnp.pi * likelihood.network.frequency[None, :] * shift
    )

    np.testing.assert_allclose(translated, expected, atol=1e-12)


def test_exact_likelihood_is_jittable_and_detector_order_invariant(wave_problem):
    gw, provenance, psd, geometries, injected, likelihood = wave_problem()
    eager = likelihood.log_probability(injected)
    compiled = eqx.filter_jit(likelihood.log_probability)(injected)
    reversed_network = gw.DetectorNetworkData(
        tuple(
            gw.DetectorStrainData(
                detector_id,
                likelihood.network.strain[index],
                psd,
                provenance,
                start_time_gps=0.0,
                active=likelihood.network.active[index],
            )
            for index, detector_id in ((1, "D2"), (0, "D1"))
        )
    )
    reversed_response = gw.DetectorResponsePlan(
        reversed_network,
        (geometries[1], geometries[0]),
    )
    reversed_likelihood = gw.GravitationalWaveLikelihoodPlan(
        reversed_network,
        reversed_response,
        likelihood.waveform,
        likelihood.waveform_parameters_fn,
        likelihood.extrinsic_parameters_fn,
        parameterization_id=likelihood.parameterization_id,
    )

    np.testing.assert_allclose(compiled, eager, atol=1e-12)
    np.testing.assert_allclose(
        reversed_likelihood.log_probability(injected), eager, atol=1e-12
    )


def test_invalid_waveform_support_returns_negative_infinity(wave_problem):
    gw, _, _, _, injected, likelihood = wave_problem()
    invalid = {**injected, "amplitude": jnp.asarray(-0.1)}
    evaluation = likelihood.evaluate(invalid)

    assert not bool(evaluation.valid)
    assert int(evaluation.status) == int(
        gw.GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT
    )
    assert jnp.isneginf(likelihood.log_probability(invalid))
    assert jnp.isneginf(eqx.filter_jit(likelihood.log_probability)(invalid))


def test_likelihood_rejects_undeclared_parameter_mapping(wave_problem):
    gw, _, _, _, _, likelihood = wave_problem()
    with pytest.raises(ValueError, match="parameterization identities"):
        gw.GravitationalWaveLikelihoodPlan(
            likelihood.network,
            likelihood.response,
            likelihood.waveform,
            likelihood.waveform_parameters_fn,
            likelihood.extrinsic_parameters_fn,
            parameterization_id="different-parameterization",
        )


def test_callable_waveform_uses_declared_polarization_order(wave_problem):
    gw, provenance, _, _, _, likelihood = wave_problem()
    capabilities = gw.WaveformCapabilities(
        ("plus", "cross"),
        arbitrary_frequencies=True,
        jittable=True,
        batched=False,
        derivative_level="first",
        parameterization_id="callable-order-test",
    )
    waveform = gw.CallableFrequencyDomainWaveform(
        lambda frequency, _: {
            "cross": 2.0j * jnp.ones_like(frequency),
            "plus": jnp.ones_like(frequency, dtype=complex),
        },
        capabilities,
        provenance,
        waveform_id="callable-order-test",
    )
    result = waveform.evaluate(likelihood.network.frequency, {})

    np.testing.assert_allclose(result.values[0], 1.0)
    np.testing.assert_allclose(result.values[1], 2.0j)


def test_time_series_data_plan_prepares_psd_band_and_window(wave_problem):
    gw, provenance, _, _, _, _ = wave_problem()
    plan = gw.GravitationalWaveDataPlan(
        64,
        1.0 / 64.0,
        start_time_gps=10.0,
        minimum_frequency=4.0,
        maximum_frequency=20.0,
        notches=((11.5, 12.5),),
        window="tukey",
        tukey_alpha=0.2,
    )
    reference = np.random.default_rng(19).normal(size=256)
    psd = plan.estimate_psd(reference, provenance, average="median")
    data = plan.prepare("D1", jnp.asarray(reference[:64]), psd, provenance)

    assert data.strain.shape == (33,)
    assert data.start_time_gps == 10.0
    assert bool(jnp.all(data.frequency[data.active] >= 4.0))
    assert bool(jnp.all(data.frequency[data.active] <= 20.0))
    assert not bool(data.active[12])
    assert 0.0 < float(data.window_power) < 1.0


def test_detector_tensor_has_known_overhead_plus_response(wave_problem):
    _, _, _, _, _, likelihood = wave_problem()
    response = likelihood.response.evaluate(0.0, 0.5 * jnp.pi, 0.0, 0.0)

    np.testing.assert_allclose(response.antenna[0], jnp.asarray([1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(response.time_delay_seconds[0], 0.0, atol=1e-12)
    assert bool(response.valid)


def test_data_contract_rejects_active_endpoints_and_mismatched_grids(wave_problem):
    gw, provenance, _, _, _, _ = wave_problem(sample_count=32)
    frequency = jnp.fft.rfftfreq(32, 1.0 / 32.0)
    with pytest.raises(ValueError, match="DC and Nyquist"):
        gw.OneSidedPowerSpectralDensity(
            frequency,
            jnp.ones_like(frequency),
            provenance,
            sample_count=32,
            sample_interval=1.0 / 32.0,
            active=jnp.ones_like(frequency, dtype=bool),
        )
    with pytest.raises(ValueError, match="canonical"):
        gw.OneSidedPowerSpectralDensity(
            frequency.at[3].add(0.01),
            jnp.ones_like(frequency),
            provenance,
            sample_count=32,
            sample_interval=1.0 / 32.0,
        )


def test_old_frequency_response_exports_are_removed():
    astrophysics = phx.applications.astrophysics
    assert "FrequencyDomainSignal" not in astrophysics.__all__
    assert "FrequencyResponsePlan" not in astrophysics.__all__
    assert "DetectorNetworkPlan" not in astrophysics.__all__
