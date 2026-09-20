#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Benchmark native NR polynomial-EIM, waveform match, and remnant kernels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx


jax.config.update("jax_enable_x64", True)


def _compiler_record(compiled) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    record = asdict(evidence)
    record["estimated_device_memory_bytes"] = evidence.estimated_device_memory_bytes
    return record


def _measure(function, arguments, warmup, repeats):
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(function).lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments), warmup=warmup, repeats=repeats
    )
    return result, {
        "compilation": asdict(compilation),
        "execution": execution.to_seconds_dict(),
        "compiler": _compiler_record(compiled),
    }


def _field(reconstruction, coefficient_scale, field_id):
    gw = phx.applications.astrophysics.gravitational_waves
    coefficients = coefficient_scale * jnp.asarray(
        [
            [1.0, 0.08, 0.03, -0.02],
            [0.9, -0.04, 0.02, 0.01],
            [0.7, 0.03, -0.04, 0.02],
            [0.5, -0.02, 0.01, -0.03],
        ]
    )
    orders = jnp.asarray(
        [
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ]
        * 4,
        dtype=jnp.int32,
    )
    return gw.PolynomialEmpiricalField(
        reconstruction,
        coefficients,
        orders,
        jnp.ones((4, 4), dtype="bool"),
        field_id=field_id,
    )


def _setup(time_samples: int):
    gw = phx.applications.astrophysics.gravitational_waves
    time = np.linspace(-100.0, 20.0, time_samples)
    centers = np.linspace(-100.0, 20.0, 4)
    width = 0.35 * (centers[-1] - centers[0])
    weights = np.exp(-(((time[:, None] - centers[None, :]) / width) ** 2))
    reconstruction = jnp.asarray(weights / np.sum(weights, axis=1, keepdims=True))
    modes = ((2, 1), (2, 2), (3, 3))
    real_fields = tuple(
        _field(reconstruction, 1.0 / index, f"benchmark:real:{mode}")
        for index, mode in enumerate(modes, start=1)
    )
    imaginary_fields = tuple(
        _field(reconstruction, 0.1 / index, f"benchmark:imaginary:{mode}")
        for index, mode in enumerate(modes, start=1)
    )
    provenance = phx.applications.astrophysics.ObservationDataProvenance.native(
        "benchmark:nr-polynomial-surrogate"
    )
    artifact = gw.AlignedNRSurrogateArtifact(
        time,
        modes,
        real_fields,
        imaginary_fields,
        jnp.asarray([0.0, -1.0, -1.0]),
        jnp.asarray([3.0, 1.0, 1.0]),
        provenance,
        maximum_mass_ratio=8.0,
        maximum_spin_magnitude=0.8,
        frame_id="benchmark:inertial",
        artifact_id="benchmark:polynomial-eim",
        time_origin_id="benchmark:analytic-grid-origin",
        mode_normalization="s2fft-orthonormal-condon-shortley",
    )
    precision = phx.discretization.spectral.SpectralPrecisionPolicy(
        jnp.complex128,
        output_dtype=jnp.complex128,
    )
    angular = phx.discretization.spectral.SphericalSpectralPlan(
        4,
        spin=-2,
        reality=False,
        precision=precision,
    ).prepare()
    surrogate = gw.AlignedNRSurrogatePlan(
        artifact,
        angular,
        phx.RelativityScaleContract.si(),
    )

    sample_count = 256
    sample_interval = 1.0 / 256.0
    frequency = jnp.fft.rfftfreq(sample_count, sample_interval)
    psd = gw.OneSidedPowerSpectralDensity(
        frequency,
        jnp.ones_like(frequency),
        provenance,
        sample_count=sample_count,
        sample_interval=sample_interval,
        psd_id="benchmark:unit-psd",
    )
    match = gw.WaveformMatchPlan(psd)
    first = jnp.exp(-(((frequency - 40.0) / 12.0) ** 2)).astype(jnp.complex128)
    second = first * jnp.exp(-2.0j * jnp.pi * frequency * (7 * sample_interval) + 0.4j)
    remnant = phx.applications.compact_objects.AlignedBinaryRemnantPlan()
    return surrogate, match, remnant, first, second


def run(time_samples: int, warmup: int, repeats: int) -> dict[str, object]:
    environment = capture_environment().to_dict()
    setup, setup_seconds = measure_synchronized(lambda: _setup(time_samples))
    surrogate, match_plan, remnant_plan, first, second = setup
    ratio = jnp.asarray(2.0)
    primary_spin = jnp.asarray(0.2)
    secondary_spin = jnp.asarray(-0.1)

    def surrogate_kernel(q, chi1, chi2):
        result = surrogate.evaluate_modes(
            {
                "mass_ratio": q,
                "primary_spin": chi1,
                "secondary_spin": chi2,
            }
        )
        return (
            result.coefficients,
            result.symmetry_defect,
            result.valid,
            result.status,
            result.qualified,
        )

    def surrogate_jvp_kernel(q, chi1, chi2):
        return jax.jvp(
            lambda value: jnp.real(surrogate_kernel(value, chi1, chi2)[0]).sum(),
            (q,),
            (jnp.ones_like(q),),
        )

    def match_kernel(first_, second_):
        result = match_plan.match(first_, second_)
        return result.match, result.lag_seconds, result.valid, result.status

    def remnant_kernel(m1, m2, chi1, chi2):
        result = remnant_plan.evaluate(m1, m2, chi1, chi2)
        return (
            result.final_mass,
            result.final_dimensionless_spin,
            result.radiated_energy_fraction,
            result.physically_valid,
            result.status,
        )

    surrogate_result, surrogate_performance = _measure(
        surrogate_kernel,
        (ratio, primary_spin, secondary_spin),
        warmup,
        repeats,
    )
    derivative_result, derivative_performance = _measure(
        surrogate_jvp_kernel,
        (ratio, primary_spin, secondary_spin),
        warmup,
        repeats,
    )
    match_result, match_performance = _measure(
        match_kernel,
        (first, second),
        warmup,
        repeats,
    )
    remnant_result, remnant_performance = _measure(
        remnant_kernel,
        (jnp.asarray(36.0), jnp.asarray(24.0), jnp.asarray(0.5), jnp.asarray(0.2)),
        warmup,
        repeats,
    )
    derivative_finite = bool(jnp.isfinite(derivative_result[1]))
    expected_lag = 7.0 * match_plan.psd.sample_interval
    successful = (
        bool(
            surrogate_result[2]
            & match_result[2]
            & remnant_result[3]
            & (jnp.abs(match_result[0] - 1.0) < 1.0e-12)
            & (jnp.abs(match_result[1] - expected_lag) < 1.0e-12)
            & ~surrogate_result[4]
        )
        and derivative_finite
    )
    return {
        "identities": {
            "benchmark": "gravitational-wave-surrogates",
            "surrogate_plan": surrogate.plan_id,
            "surrogate_artifact": surrogate.artifact.artifact_id,
            "surrogate_trust": surrogate.artifact.trust_id,
            "match_plan": match_plan.plan_id,
            "remnant_plan": remnant_plan.plan_id,
            "remnant_source": remnant_plan.source_id,
        },
        "configuration": {
            "time_samples": time_samples,
            "mode_count": len(surrogate.artifact.modes),
            "fit_coordinates": 3,
            "empirical_nodes": 4,
            "polynomial_terms": 4,
            "match_sample_count": match_plan.psd.sample_count,
            "warmup": warmup,
            "repeats": repeats,
        },
        "environment": environment,
        "physics": {
            "successful": successful,
            "surrogate_valid": bool(surrogate_result[2]),
            "surrogate_source_authenticated": surrogate.artifact.source_authenticated,
            "surrogate_qualified": bool(surrogate_result[4]),
            "surrogate_status": int(surrogate_result[3]),
            "surrogate_symmetry_defect": float(surrogate_result[1]),
            "surrogate_derivative_finite": derivative_finite,
            "match_valid": bool(match_result[2]),
            "match_status": int(match_result[3]),
            "match": float(match_result[0]),
            "lag_seconds": float(match_result[1]),
            "remnant_physically_valid": bool(remnant_result[3]),
            "remnant_status": int(remnant_result[4]),
            "final_mass": float(remnant_result[0]),
            "final_spin": float(remnant_result[1]),
            "radiated_energy_fraction": float(remnant_result[2]),
        },
        "performance": {
            "setup_seconds": setup_seconds,
            "surrogate": surrogate_performance,
            "surrogate_jvp": derivative_performance,
            "match": match_performance,
            "remnant": remnant_performance,
            "logical_bytes": {
                "surrogate": logical_array_bytes(surrogate),
                "surrogate_output": logical_array_bytes(surrogate_result),
                "match_plan": logical_array_bytes(match_plan),
                "match_inputs": logical_array_bytes((first, second)),
                "remnant_plan": logical_array_bytes(remnant_plan),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--time-samples", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not 16 <= arguments.time_samples <= 1_000_000:
        raise ValueError("time-samples must be between 16 and 1,000,000.")
    if not 0 <= arguments.warmup <= 100:
        raise ValueError("warmup must be between 0 and 100.")
    if not 1 <= arguments.repeats <= 1_000:
        raise ValueError("repeats must be between 1 and 1,000.")
    payload = run(arguments.time_samples, arguments.warmup, arguments.repeats)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["physics"]["successful"] else 1)


if __name__ == "__main__":
    main()
