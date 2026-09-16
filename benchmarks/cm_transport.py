#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Decision benchmark for distinct periodic and open electronic transport paths.

Run from the repository root with ``PYTHONPATH=.:benchmarks JAX_ENABLE_X64=1``.
Kubo, constant-tau Boltzmann, one-energy retarded embedding, and explicit
disorder statistics are timed separately.  Timings and logical bytes are
resource observations, never scientific qualification or release evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment, logical_array_bytes, measure_repeated

from phydrax.chemistry.periodic._boltzmann import (
    ConstantRelaxationTime,
    PeriodicBoltzmannPlan,
)
from phydrax.chemistry.periodic._kubo import (
    finite_frequency_kubo_response,
    KuboDiamagneticSumRule,
    KuboLinewidth,
    PeriodicKuboPlan,
)
from phydrax.operators.quantum._electronic_transport import (
    ElasticDisorderEnsemblePlan,
    evaluate_elastic_disorder_ensemble,
    retarded_embedding,
    retarded_open_system_point,
)


_E = 1.602176634e-19
_KB = 1.380649e-23


def _periodic_inputs(k_points: int, bands: int):
    coordinate = np.linspace(-0.5, 0.5, k_points, endpoint=False)
    centers = (np.arange(bands) - 0.5 * (bands - 1)) * 0.35 * _E
    energies = centers[None, :] + 0.08 * _E * np.cos(2.0 * np.pi * coordinate[:, None])
    dimension = 3
    velocity = np.zeros((k_points, bands, bands, dimension), dtype=complex)
    diagonal = 1.5e5 * np.sin(2.0 * np.pi * coordinate)
    for band in range(bands):
        velocity[:, band, band, :] = diagonal[:, None] * np.asarray(
            [1.0, 0.6 + 0.15 * band, 0.3 + 0.07 * band * band]
        )
    for left in range(bands):
        for right in range(left + 1, bands):
            amplitude = 3.0e4 / (1 + right - left)
            vector = amplitude * np.asarray([1.0, 0.5j, 0.3 + 0.2j])
            velocity[:, left, right, :] = vector
            velocity[:, right, left, :] = vector.conj()
    weights = np.full(k_points, 1.0 / k_points)
    return energies, velocity, weights


def _independent_f_sum(energies, velocity, weights, temperature, volume):
    occupations = 1.0 / (1.0 + np.exp(energies / (_KB * temperature)))
    derivative = occupations * (1.0 - occupations) / (_KB * temperature)
    dimension = velocity.shape[-1]
    drude = np.zeros((dimension, dimension))
    regular = np.zeros((dimension, dimension))
    for k, weight in enumerate(weights):
        for band in range(energies.shape[1]):
            diagonal = velocity[k, band, band]
            drude += (
                weight
                * derivative[k, band]
                * np.real(np.outer(diagonal, diagonal.conj()))
            )
            for target in range(band + 1, energies.shape[1]):
                gap = energies[k, target] - energies[k, band]
                difference = occupations[k, band] - occupations[k, target]
                element = velocity[k, band, target]
                regular += (
                    weight * difference / gap * np.real(np.outer(element, element.conj()))
                )
    prefactor = np.pi * _E**2 / volume
    return prefactor * (0.5 * drude + regular)


def run_case(
    k_points: int,
    bands: int,
    frequencies: int,
    disorder_realizations: int,
    warmup: int,
    repeats: int,
):
    started = time.perf_counter()
    energies, velocity, weights = _periodic_inputs(k_points, bands)
    volume, temperature = 1.0e-28, 300.0
    sum_rule = KuboDiamagneticSumRule(
        _independent_f_sum(energies, velocity, weights, temperature, volume),
        source_id="benchmark-independent-loop-f-sum",
    )
    kubo = PeriodicKuboPlan(
        energies,
        velocity,
        weights,
        sum_rule,
        chemical_potential_joule=0.0,
        temperature_kelvin=temperature,
        cell_volume_m3=volume,
    )
    boltzmann = PeriodicBoltzmannPlan(
        energies,
        np.real(np.diagonal(velocity, axis1=1, axis2=2).transpose(0, 2, 1)),
        weights,
        ConstantRelaxationTime(1.0e-14, mechanism_id="benchmark-supplied-tau"),
        chemical_potential_joule=0.0,
        temperature_kelvin=temperature,
        cell_volume_m3=volume,
    )
    frequency = jnp.linspace(1.0e12, 5.0e15, frequencies)
    linewidth = KuboLinewidth(0.03 * _E, mechanism_id="benchmark-physical-linewidth")
    preparation_seconds = time.perf_counter() - started

    raw, raw_timing = measure_repeated(
        kubo.raw_transitions, warmup=warmup, repeats=repeats
    )
    optical, optical_timing = measure_repeated(
        lambda: finite_frequency_kubo_response(kubo, frequency, linewidth),
        warmup=warmup,
        repeats=repeats,
    )
    thermoelectric, boltzmann_timing = measure_repeated(
        boltzmann.evaluate, warmup=warmup, repeats=repeats
    )

    energy = jnp.asarray(0.0 + 0.0j)
    contact = retarded_embedding(energy, jnp.asarray([[-1.0j]]), jnp.asarray([[1.0]]))
    open_point, embedding_timing = measure_repeated(
        lambda: retarded_open_system_point(
            energy, jnp.asarray([[0.0]]), jnp.asarray([[1.0]]), (contact, contact)
        ),
        warmup=warmup,
        repeats=repeats,
    )
    disorder_plan = ElasticDisorderEnsemblePlan(
        tuple(f"realization-{index}" for index in range(disorder_realizations)),
        np.full(disorder_realizations, 1.0 / disorder_realizations),
    )
    disorder_values = jnp.linspace(0.8, 1.2, disorder_realizations)[:, None]
    disorder, disorder_timing = measure_repeated(
        lambda: evaluate_elastic_disorder_ensemble(
            disorder_plan, disorder_values, jnp.ones(disorder_realizations, dtype=bool)
        ),
        warmup=warmup,
        repeats=repeats,
    )
    outputs = (raw, optical, thermoelectric, open_point, disorder)
    return {
        "axes": {
            "k_points": k_points,
            "bands": bands,
            "angular_frequencies": frequencies,
            "device_modes": 1,
            "leads": 2,
            "retarded_energies": 1,
            "disorder_realizations": disorder_realizations,
        },
        "preparation_seconds": preparation_seconds,
        "timings_seconds": {
            "kubo_raw": raw_timing.to_seconds_dict(),
            "kubo_regular_frequency": optical_timing.to_seconds_dict(),
            "boltzmann_constant_tau": boltzmann_timing.to_seconds_dict(),
            "retarded_embedding_point": embedding_timing.to_seconds_dict(),
            "elastic_disorder_statistics": disorder_timing.to_seconds_dict(),
        },
        "resources": {
            "logical_input_bytes": logical_array_bytes((kubo, boltzmann, disorder_plan)),
            "logical_output_bytes": logical_array_bytes(outputs),
            "measured_peak_host_bytes": None,
            "measured_peak_device_bytes": None,
            "peak_measurement_unavailable_reason": "no allocator telemetry in this benchmark path",
        },
        "scientific_residuals": {
            "kubo_f_sum_relative": float(raw.f_sum_relative_residual),
            "kubo_minimum_dissipative_eigenvalue": float(
                optical.evidence.minimum_dissipative_eigenvalue
            ),
            "boltzmann_onsager_relative": float(
                thermoelectric.evidence.kelvin_onsager_relative_residual
            ),
            "retarded_spectral_identity_relative": float(
                open_point.spectral_identity_residual
            ),
            "disorder_effective_sample_size": float(disorder.effective_sample_size),
        },
        "successful": bool(optical.successful)
        and bool(thermoelectric.successful)
        and bool(open_point.successful)
        and bool(disorder.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k-points", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--bands", type=int, default=4)
    parser.add_argument("--frequencies", type=int, default=128)
    parser.add_argument("--disorder-realizations", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        min(arguments.k_points) < 1
        or arguments.bands < 3
        or arguments.frequencies < 2
        or arguments.disorder_realizations < 2
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError("Transport benchmark axes or sample counts are invalid.")
    if not jax.config.x64_enabled:
        raise ValueError("This physical-units benchmark requires JAX_ENABLE_X64=1.")
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "sample_policy": "retain-every-synchronized-sample",
        },
        "runs": [
            run_case(
                count,
                arguments.bands,
                arguments.frequencies,
                arguments.disorder_realizations,
                arguments.warmup,
                arguments.repeats,
            )
            for count in arguments.k_points
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    payload["artifact_bytes_before_artifact_byte_field"] = len(encoded.encode("utf-8"))
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
