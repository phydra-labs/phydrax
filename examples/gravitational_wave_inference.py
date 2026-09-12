# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Self-contained normalized gravitational-wave inference with native nested sampling."""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def run(*, output: str | Path = ".tmp/gravitational-wave-example.phxresult"):
    gw = phx.applications.astrophysics.gravitational_waves
    provenance = phx.applications.astrophysics.ObservationDataProvenance.native(
        "sine-gaussian-example"
    )
    sample_count = 64
    sample_interval = 1.0 / sample_count
    frequency = jnp.fft.rfftfreq(sample_count, sample_interval)
    active = (frequency >= 4.0) & (frequency <= 24.0) & (frequency < frequency[-1])
    psd = gw.OneSidedPowerSpectralDensity(
        frequency,
        jnp.ones_like(frequency),
        provenance,
        sample_count=sample_count,
        sample_interval=sample_interval,
        active=active,
    )
    geometry = gw.InterferometerGeometry(
        "D1",
        jnp.zeros(3),
        jnp.asarray([1.0, 0.0, 0.0]),
        jnp.asarray([0.0, 1.0, 0.0]),
        provenance,
    )
    fixed_waveform = {
        "center_frequency": jnp.asarray(10.0),
        "quality_factor": jnp.asarray(5.0),
        "phase": jnp.asarray(0.3),
        "ellipticity": jnp.asarray(0.2),
        "luminosity_distance": jnp.asarray(1.0),
    }
    waveform_parameters = lambda values: {
        **fixed_waveform,
        "amplitude": values["amplitude"],
    }
    extrinsic_parameters = lambda _: (
        jnp.asarray(0.4),
        jnp.asarray(0.2),
        jnp.asarray(0.1),
        jnp.asarray(0.0),
    )
    waveform = gw.SineGaussianWaveformPlan(
        provenance, parameterization_id="sine-gaussian-amplitude"
    )
    zero_data = gw.DetectorStrainData(
        "D1",
        jnp.zeros_like(frequency, dtype=complex),
        psd,
        provenance,
        start_time_gps=0.0,
        active=active,
    )
    zero_network = gw.DetectorNetworkData((zero_data,))
    zero_response = gw.DetectorResponsePlan(zero_network, (geometry,))
    zero_likelihood = gw.GravitationalWaveLikelihoodPlan(
        zero_network,
        zero_response,
        waveform,
        waveform_parameters,
        extrinsic_parameters,
        parameterization_id=waveform.capabilities.parameterization_id,
    )
    injected_amplitude = jnp.asarray(0.42)
    injected_signal = zero_likelihood.detector_signal({"amplitude": injected_amplitude})[
        0
    ][0]
    data = gw.DetectorStrainData(
        "D1",
        injected_signal,
        psd,
        provenance,
        start_time_gps=0.0,
        active=active,
    )
    network = gw.DetectorNetworkData((data,))
    response = gw.DetectorResponsePlan(network, (geometry,))
    likelihood = gw.GravitationalWaveLikelihoodPlan(
        network,
        response,
        waveform,
        waveform_parameters,
        extrinsic_parameters,
        parameterization_id=waveform.capabilities.parameterization_id,
    )
    parameter_plan = gw.GravitationalWaveParameterPlan(
        {"amplitude": jnp.asarray(0.3)},
        {"amplitude": phx.uq.Uniform(0.05, 0.8)},
        continuous_paths=("['amplitude']",),
        waveform_parameters=waveform_parameters,
        extrinsic_parameters=extrinsic_parameters,
        parameterization_id="sine-gaussian-amplitude",
    )
    prepared = gw.prepare_gravitational_wave_inference(
        parameter_plan,
        likelihood,
        analysis_id="native-sine-gaussian-amplitude",
        nested_capacity=phx.uq.NestedSamplingCapacity(
            max_live=24,
            max_dead_points=192,
            max_likelihood_evaluations=4096,
            max_dynamic_batches=1,
            max_clusters=2,
            max_phantoms=24,
        ),
        initial_live=24,
    )
    result = phx.uq.sample_nested(
        prepared.posterior,
        key=jr.key(20260911),
        plan=prepared.nested_sampling_plan,
        remaining_evidence_tolerance=0.1,
    )
    weights = jnp.exp(result.posterior_log_weights)
    posterior_mean = jnp.sum(weights * result.samples["amplitude"])
    destination = phx.uq.export_result(
        result,
        Path(output),
        context=prepared.result_context,
    )
    summary = {
        "status": phx.uq.nested_sampling_status_name(int(result.status)),
        "valid": bool(result.valid),
        "injected_amplitude": float(injected_amplitude),
        "posterior_mean_amplitude": float(posterior_mean),
        "absolute_error": float(jnp.abs(posterior_mean - injected_amplitude)),
        "log_evidence": float(result.log_evidence),
        "result": str(destination),
    }
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
