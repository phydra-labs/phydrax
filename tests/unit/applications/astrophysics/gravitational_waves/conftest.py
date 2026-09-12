import jax.numpy as jnp
import pytest

import phydrax as phx


@pytest.fixture
def wave_problem():
    return build_wave_problem


def build_wave_problem(*, sample_count=64):
    gw = phx.applications.astrophysics.gravitational_waves
    sample_interval = 1.0 / sample_count
    frequency = jnp.fft.rfftfreq(sample_count, sample_interval)
    active = (frequency >= 4.0) & (frequency <= 24.0) & (frequency < frequency[-1])
    provenance = phx.applications.astrophysics.ObservationDataProvenance.native(
        "gravitational-wave-test"
    )
    psd = gw.OneSidedPowerSpectralDensity(
        frequency,
        jnp.ones_like(frequency),
        provenance,
        sample_count=sample_count,
        sample_interval=sample_interval,
        active=active,
    )
    geometries = (
        gw.InterferometerGeometry(
            "D1",
            jnp.asarray([0.0, 0.0, 0.0]),
            jnp.asarray([1.0, 0.0, 0.0]),
            jnp.asarray([0.0, 1.0, 0.0]),
            provenance,
        ),
        gw.InterferometerGeometry(
            "D2",
            jnp.asarray([1000.0, 0.0, 0.0]),
            jnp.asarray([0.0, 1.0, 0.0]),
            jnp.asarray([-1.0, 0.0, 0.0]),
            provenance,
        ),
    )
    zero_data = tuple(
        gw.DetectorStrainData(
            detector_id,
            jnp.zeros_like(frequency, dtype=complex),
            psd,
            provenance,
            start_time_gps=0.0,
            active=active,
        )
        for detector_id in ("D1", "D2")
    )
    zero_network = gw.DetectorNetworkData(zero_data)
    zero_response = gw.DetectorResponsePlan(zero_network, geometries)
    waveform = gw.SineGaussianWaveformPlan(provenance)
    waveform_keys = (
        "amplitude",
        "center_frequency",
        "quality_factor",
        "phase",
        "ellipticity",
        "luminosity_distance",
    )
    waveform_parameters = lambda values: {name: values[name] for name in waveform_keys}
    injected = {
        "amplitude": jnp.asarray(0.4),
        "center_frequency": jnp.asarray(10.0),
        "quality_factor": jnp.asarray(5.0),
        "phase": jnp.asarray(0.3),
        "ellipticity": jnp.asarray(0.25),
        "luminosity_distance": jnp.asarray(1.0),
        "right_ascension": jnp.asarray(0.4),
        "declination": jnp.asarray(0.2),
        "polarization": jnp.asarray(0.1),
        "geocent_time": jnp.asarray(0.02),
    }
    zero_likelihood = gw.GravitationalWaveLikelihoodPlan(
        zero_network,
        zero_response,
        waveform,
        waveform_parameters,
        gw.default_extrinsic_parameters,
        parameterization_id=waveform.capabilities.parameterization_id,
    )
    signal = zero_likelihood.detector_signal(injected)[0]
    data = tuple(
        gw.DetectorStrainData(
            detector_id,
            signal[index],
            psd,
            provenance,
            start_time_gps=0.0,
            active=active,
        )
        for index, detector_id in enumerate(("D1", "D2"))
    )
    network = gw.DetectorNetworkData(data)
    response = gw.DetectorResponsePlan(network, geometries)
    likelihood = gw.GravitationalWaveLikelihoodPlan(
        network,
        response,
        waveform,
        waveform_parameters,
        gw.default_extrinsic_parameters,
        parameterization_id=waveform.capabilities.parameterization_id,
    )
    return gw, provenance, psd, geometries, injected, likelihood
