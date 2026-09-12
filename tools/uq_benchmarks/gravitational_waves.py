# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Deterministic gravitational-wave accuracy and performance scenarios."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import phydrax as phx

from .configuration import BenchmarkConfiguration
from .report import Metric, metric, ScenarioResult


def _timed(function: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = function()
    jax.block_until_ready(value)
    return value, time.perf_counter() - started


def _problem(configuration: BenchmarkConfiguration):
    gw = phx.applications.astrophysics.gravitational_waves
    frequency_bins = configuration.gravitational_wave_frequency_bins
    sample_count = 2 * (frequency_bins - 1)
    sample_interval = 1.0 / sample_count
    frequency = jnp.fft.rfftfreq(sample_count, sample_interval)
    active = (frequency >= 4.0) & (frequency <= 0.75 * frequency[-1])
    active = active & (frequency < frequency[-1])
    provenance = phx.applications.astrophysics.ObservationDataProvenance.native(
        f"gravitational-wave-benchmark-{configuration.profile}"
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
            jnp.zeros(3),
            jnp.asarray([1.0, 0.0, 0.0]),
            jnp.asarray([0.0, 1.0, 0.0]),
            provenance,
        ),
        gw.InterferometerGeometry(
            "D2",
            jnp.asarray([2000.0, -1000.0, 500.0]),
            jnp.asarray([0.0, 1.0, 0.0]),
            jnp.asarray([0.0, 0.0, 1.0]),
            provenance,
        ),
    )
    zero = tuple(
        gw.DetectorStrainData(
            geometry.detector_id,
            jnp.zeros_like(frequency, dtype=complex),
            psd,
            provenance,
            start_time_gps=0.0,
            active=active,
        )
        for geometry in geometries
    )
    waveform = gw.SineGaussianWaveformPlan(
        provenance, parameterization_id="benchmark-sine-gaussian"
    )
    parameters = {
        "amplitude": jnp.asarray(0.4),
        "center_frequency": jnp.asarray(0.35 * frequency[-1]),
        "quality_factor": jnp.asarray(6.0),
        "phase": jnp.asarray(0.3),
        "ellipticity": jnp.asarray(0.2),
        "luminosity_distance": jnp.asarray(1.0),
        "right_ascension": jnp.asarray(0.4),
        "declination": jnp.asarray(0.2),
        "polarization": jnp.asarray(0.1),
        "geocent_time": jnp.asarray(0.0),
    }
    waveform_parameters = lambda values: {
        name: values[name]
        for name in (
            "amplitude",
            "center_frequency",
            "quality_factor",
            "phase",
            "ellipticity",
            "luminosity_distance",
        )
    }
    extrinsic_parameters = lambda values: (
        values["right_ascension"],
        values["declination"],
        values["polarization"],
        values["geocent_time"],
    )
    zero_network = gw.DetectorNetworkData(zero)
    zero_likelihood = gw.GravitationalWaveLikelihoodPlan(
        zero_network,
        gw.DetectorResponsePlan(zero_network, geometries),
        waveform,
        waveform_parameters,
        extrinsic_parameters,
        parameterization_id=waveform.capabilities.parameterization_id,
    )
    signal = zero_likelihood.detector_signal(parameters)[0]
    data = tuple(
        gw.DetectorStrainData(
            geometry.detector_id,
            signal[index],
            psd,
            provenance,
            start_time_gps=0.0,
            active=active,
        )
        for index, geometry in enumerate(geometries)
    )
    network = gw.DetectorNetworkData(data)
    likelihood = gw.GravitationalWaveLikelihoodPlan(
        network,
        gw.DetectorResponsePlan(network, geometries),
        waveform,
        waveform_parameters,
        extrinsic_parameters,
        parameterization_id=waveform.capabilities.parameterization_id,
    )
    nearby = (
        parameters,
        {**parameters, "center_frequency": parameters["center_frequency"] * 1.01},
        {**parameters, "quality_factor": parameters["quality_factor"] * 0.97},
    )
    return gw, parameters, nearby, likelihood


def gravitational_wave_exact_likelihood(
    configuration: BenchmarkConfiguration, seed: int
) -> ScenarioResult:
    """Check normalized network likelihood identity and compiled evaluation cost."""
    _, parameters, _, likelihood = _problem(configuration)
    evaluation = likelihood.evaluate(parameters)
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
    identity_error = float(
        jnp.max(jnp.abs(evaluation.log_probability_by_detector - direct))
    )
    compiled = eqx.filter_jit(likelihood.log_probability)
    _, cold_seconds = _timed(lambda: compiled(parameters))
    _, warm_seconds = _timed(lambda: compiled(parameters))
    return ScenarioResult(
        name="gravitational_wave_exact_likelihood",
        description=gravitational_wave_exact_likelihood.__doc__ or "",
        seed=seed,
        metrics={
            "normalized_identity_error": metric(
                identity_error, "accuracy", maximum=1.0e-10
            ),
            "cold_seconds": metric(cold_seconds, "performance", unit="s"),
            "warm_seconds": metric(warm_seconds, "performance", unit="s"),
        },
        metadata={
            "profile": configuration.profile,
            "detectors": len(likelihood.network.detector_ids),
            "frequency_bins": int(likelihood.network.frequency.size),
        },
    )


def gravitational_wave_marginalization(
    configuration: BenchmarkConfiguration, seed: int
) -> ScenarioResult:
    """Compare analytic phase marginalization with periodic numerical quadrature."""
    gw, parameters, _, likelihood = _problem(configuration)
    phase = gw.PhaseMarginalizationPlan(reconstruction_nodes=128)
    marginalized = gw.GravitationalWaveMarginalizationPlan(likelihood, phase=phase)
    reduced = dict(parameters)
    reduced.pop("phase")
    nodes = jnp.linspace(0.0, 2.0 * jnp.pi, 1024, endpoint=False)
    explicit = jsp.special.logsumexp(
        jax.vmap(lambda value: likelihood.log_probability({**reduced, "phase": value}))(
            nodes
        )
    ) - jnp.log(float(nodes.size))
    value, elapsed = _timed(lambda: marginalized.log_probability(reduced))
    return ScenarioResult(
        name="gravitational_wave_marginalization",
        description=gravitational_wave_marginalization.__doc__ or "",
        seed=seed,
        metrics={
            "phase_log_probability_error": metric(
                float(jnp.abs(value - explicit)), "accuracy", maximum=2.0e-5
            ),
            "evaluation_seconds": metric(elapsed, "performance", unit="s"),
        },
        metadata={"profile": configuration.profile, "quadrature_nodes": 1024},
    )


def gravitational_wave_relative_binning(
    configuration: BenchmarkConfiguration, seed: int
) -> ScenarioResult:
    """Qualify relative binning against held-out exact likelihood evaluations."""
    gw, parameters, validation, likelihood = _problem(configuration)
    prepared = gw.prepare_relative_binning_likelihood(
        likelihood,
        parameters,
        validation,
        tuple(f"validation-{index}" for index in range(len(validation))),
        num_bins=configuration.gravitational_wave_relative_bins,
        policy=gw.LikelihoodApproximationPolicy(2.0, 1.0),
    )
    _, elapsed = _timed(lambda: prepared.log_probability(validation[-1]))
    return ScenarioResult(
        name="gravitational_wave_relative_binning",
        description=gravitational_wave_relative_binning.__doc__ or "",
        seed=seed,
        metrics={
            "maximum_log_probability_error": metric(
                float(prepared.qualification.maximum_absolute_error),
                "accuracy",
                maximum=2.0,
            ),
            "evaluation_seconds": metric(elapsed, "performance", unit="s"),
        },
        metadata={
            "profile": configuration.profile,
            "relative_bins": configuration.gravitational_wave_relative_bins,
            "qualification_id": prepared.qualification.report_id,
        },
    )


def gravitational_wave_multiband(
    configuration: BenchmarkConfiguration, seed: int
) -> ScenarioResult:
    """Qualify fixed-policy multiband compression against exact evaluation."""
    gw, _, validation, likelihood = _problem(configuration)
    active_frequency = likelihood.network.frequency[
        jnp.any(likelihood.network.active, axis=0)
    ]
    prepared = gw.prepare_multiband_likelihood(
        likelihood,
        (float(active_frequency[0]), float(active_frequency[-1])),
        (2,),
        validation,
        tuple(f"validation-{index}" for index in range(len(validation))),
        policy=gw.LikelihoodApproximationPolicy(4.0, 2.0),
    )
    _, elapsed = _timed(lambda: prepared.log_probability(validation[-1]))
    return ScenarioResult(
        name="gravitational_wave_multiband",
        description=gravitational_wave_multiband.__doc__ or "",
        seed=seed,
        metrics={
            "maximum_log_probability_error": metric(
                float(prepared.qualification.maximum_absolute_error),
                "accuracy",
                maximum=4.0,
            ),
            "evaluation_seconds": metric(elapsed, "performance", unit="s"),
        },
        metadata={
            "profile": configuration.profile,
            "qualification_id": prepared.qualification.report_id,
        },
    )


def _train_roq_interpolation(likelihood, parameters, *, quadratic: bool):
    frequency = np.asarray(likelihood.network.frequency)
    center = float(parameters["center_frequency"])
    quality = float(parameters["quality_factor"])
    centers = np.linspace(0.94 * center, 1.06 * center, 16)
    qualities = np.linspace(0.88 * quality, 1.12 * quality, 4)
    snapshots = []
    for candidate_center in centers:
        for candidate_quality in qualities:
            envelope = np.exp(
                -0.5
                * (
                    (frequency - candidate_center)
                    / (candidate_center / candidate_quality)
                )
                ** 2
            )
            snapshots.append(envelope**2 if quadratic else envelope)
    cases = tuple(
        phx.rom.ROMCaseSpec(f"roq-{int(quadratic)}-{index}", (("index", float(index)),))
        for index in range(len(snapshots))
    )

    def truth(case):
        index = int(dict(case.parameters)["index"])
        return phx.rom.TruthSample(snapshots[index], f"snapshot-{index}")

    corpus = phx.rom.create_corpus(
        cases,
        truth,
        truth_model_id=f"sine-gaussian-envelope-{int(quadratic)}",
        truth_model_revision="benchmark",
        split=phx.rom.CorpusSplit(tuple(case.case_id for case in cases)),
    )
    artifact = phx.rom.train_profile(corpus, phx.rom.LinearPODProfile(12))
    return phx.rom.prepare_empirical_interpolation(artifact).prepare()


def gravitational_wave_reduced_order_quadrature(
    configuration: BenchmarkConfiguration, seed: int
) -> ScenarioResult:
    """Train, prepare, and qualify linear/quadratic reduced-order quadrature."""
    gw, parameters, validation, likelihood = _problem(configuration)
    started = time.perf_counter()
    linear = _train_roq_interpolation(likelihood, parameters, quadratic=False)
    quadratic = _train_roq_interpolation(likelihood, parameters, quadratic=True)
    prepared = gw.prepare_reduced_order_quadrature_likelihood(
        likelihood,
        linear,
        quadratic,
        validation,
        tuple(f"validation-{index}" for index in range(len(validation))),
        policy=gw.LikelihoodApproximationPolicy(5.0, 3.0),
    )
    preparation_seconds = time.perf_counter() - started
    _, elapsed = _timed(lambda: prepared.log_probability(validation[-1]))
    return ScenarioResult(
        name="gravitational_wave_reduced_order_quadrature",
        description=gravitational_wave_reduced_order_quadrature.__doc__ or "",
        seed=seed,
        metrics={
            "maximum_log_probability_error": metric(
                float(prepared.qualification.maximum_absolute_error),
                "accuracy",
                maximum=5.0,
            ),
            "preparation_seconds": metric(preparation_seconds, "performance", unit="s"),
            "evaluation_seconds": metric(elapsed, "performance", unit="s"),
        },
        metadata={
            "profile": configuration.profile,
            "linear_nodes": int(linear.node_indices.size),
            "quadratic_nodes": int(quadratic.node_indices.size),
            "qualification_id": prepared.qualification.report_id,
        },
    )


def gravitational_wave_population_recycling(
    configuration: BenchmarkConfiguration, seed: int
) -> ScenarioResult:
    """Evaluate hierarchical event recycling, selection normalization, and ESS."""
    event_count = configuration.gravitational_wave_population_events
    sample_count = configuration.gravitational_wave_population_samples
    grid = jnp.linspace(-3.0, 3.0, sample_count)
    events = []
    for event_index in range(event_count):
        shifted = grid + 0.05 * event_index
        target = phx.integration.WeightedSampleTarget(
            {"x": shifted},
            jnp.zeros(sample_count),
            normalized=True,
            independent=False,
            sample_axes=0,
            provenance=f"event-{event_index}",
        )
        events.append(
            phx.uq.EventPosterior(
                target,
                jnp.zeros(sample_count),
                event_id=f"event-{event_index}",
                parameterization_id="scalar-x",
                likelihood_id="benchmark-event-likelihood",
                provider_id="native",
                inference_method="nested",
                approximation="exact",
                source_effective_sample_size=float(sample_count),
            )
        )
    batch = phx.uq.prepare_population_sample_batch(tuple(events))

    def population_log_probability(hyperparameters, sample):
        scale = hyperparameters["scale"]
        return (
            -0.5 * ((sample["x"] - hyperparameters["mean"]) / scale) ** 2
            - jnp.log(scale)
            - 0.5 * jnp.log(2.0 * jnp.pi)
        )

    hyperparameters = {"mean": jnp.asarray(0.1), "scale": jnp.asarray(1.1)}
    proposal = population_log_probability(hyperparameters, {"x": grid})
    selection = phx.uq.SelectionInjectionSet(
        {"x": grid},
        proposal,
        jnp.ones(sample_count),
        campaign_id="complete-detection",
        parameterization_id="scalar-x",
    )
    term = phx.uq.PopulationPosteriorTerm(
        batch,
        population_log_probability,
        selection=selection,
        minimum_event_effective_sample_size=1.0,
    )
    hyperparameters = {"mean": jnp.asarray(0.1), "scale": jnp.asarray(1.1)}
    diagnostics, elapsed = _timed(lambda: term.diagnostics(hyperparameters))
    return ScenarioResult(
        name="gravitational_wave_population_recycling",
        description=gravitational_wave_population_recycling.__doc__ or "",
        seed=seed,
        metrics={
            "minimum_event_effective_sample_size": Metric(
                float(jnp.min(diagnostics.event_importance_effective_sample_size)),
                "diagnostic",
                minimum=1.0,
            ),
            "selection_efficiency": Metric(
                float(diagnostics.selection.efficiency),
                "diagnostic",
                minimum=0.0,
                maximum=1.0,
            ),
            "evaluation_seconds": metric(elapsed, "performance", unit="s"),
        },
        metadata={
            "profile": configuration.profile,
            "events": event_count,
            "samples_per_event": sample_count,
        },
    )


def gravitational_wave_simulation_calibration(
    configuration: BenchmarkConfiguration, seed: int
) -> ScenarioResult:
    """Check an exactly stratified simulation-calibration rank campaign."""
    case_count = max(20, configuration.calibration_cases)
    num_bins = 10
    posterior_values = (jnp.arange(case_count) + 0.5) / case_count
    posterior = phx.integration.WeightedSampleTarget(
        {"x": posterior_values},
        jnp.zeros(case_count),
        normalized=True,
        independent=False,
        sample_axes=0,
        provenance="uniform-calibration-reference",
    )
    cases = tuple(
        phx.uq.SimulationCalibrationCase(
            {"x": value},
            posterior,
            case_id=f"case-{index}",
            analysis_id="gravitational-wave-sbc",
        )
        for index, value in enumerate(posterior_values)
    )
    plan = phx.uq.SimulationCalibrationPlan(
        ("['x']",), num_bins=num_bins, minimum_valid_cases=case_count
    )
    result, elapsed = _timed(lambda: phx.uq.simulation_calibration(cases, plan))
    expected = case_count / num_bins
    maximum_deviation = float(jnp.max(jnp.abs(result.histogram_counts - expected)))
    return ScenarioResult(
        name="gravitational_wave_simulation_calibration",
        description=gravitational_wave_simulation_calibration.__doc__ or "",
        seed=seed,
        metrics={
            "maximum_rank_count_deviation": metric(
                maximum_deviation, "calibration", maximum=1.0
            ),
            "evaluation_seconds": metric(elapsed, "performance", unit="s"),
        },
        metadata={
            "profile": configuration.profile,
            "cases": case_count,
            "bins": num_bins,
        },
    )


GRAVITATIONAL_WAVE_SCENARIOS = {
    "gravitational_wave_exact_likelihood": gravitational_wave_exact_likelihood,
    "gravitational_wave_marginalization": gravitational_wave_marginalization,
    "gravitational_wave_relative_binning": gravitational_wave_relative_binning,
    "gravitational_wave_multiband": gravitational_wave_multiband,
    "gravitational_wave_reduced_order_quadrature": gravitational_wave_reduced_order_quadrature,
    "gravitational_wave_population_recycling": gravitational_wave_population_recycling,
    "gravitational_wave_simulation_calibration": gravitational_wave_simulation_calibration,
}


__all__ = ["GRAVITATIONAL_WAVE_SCENARIOS"]
