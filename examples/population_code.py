"""Physical LIF function approximation and filtered stationary spike decoding.

Run from the repository: python examples/population_code.py
"""

from __future__ import annotations

import json

import jax.numpy as jnp
import jax.random as jr

from phydrax.applications import electrophysiology as ep
from phydrax.domain import HyperRectangle
from phydrax.nn import population as pc


def target(point):
    return jnp.asarray([point[0] ** 2, jnp.sin(2.0 * point[0])])


def main() -> None:
    population_key, training_key, held_out_key, phase_key = jr.split(jr.key(42), 4)
    neuron = ep.LeakyIntegrateAndFire(0.2, 0.01, -65.0, -50.0, -62.0, refractory_ms=2.0)
    population = pc.prepare_lif_population(
        HyperRectangle([-1.0], [1.0]), neuron, 64, key=population_key
    )
    training = pc.sample_population_points(population, 512, key=training_key)
    held_out = pc.sample_population_points(population, 256, key=held_out_key)
    code = pc.fit_population_decoder(population, training, target, ridge=1e-6)
    approximation = pc.assess_population_code(code, held_out, target)

    # Under constant input, a deterministic physical LIF neuron has period
    # charge(reset -> threshold) + refractory. Random stationary phases yield
    # exact periodic physical LIF event counts, not a new simulator or Poisson
    # surrogate. Silent neurons have rate zero and emit no events.
    dt_ms = 1.0
    point = jnp.asarray([0.4])
    rates_hz = population.rates(point)
    edges_ms = jnp.arange(2001) * dt_ms
    phases = jr.uniform(phase_key, (population.neuron_count,))
    cumulative = jnp.floor(edges_ms[:, None] * rates_hz[None, :] / 1000.0 + phases)
    spike_counts = jnp.diff(cumulative, axis=0)
    filtered = pc.filter_population_spikes(
        spike_counts, dt_ms=dt_ms, time_constant_ms=30.0
    )
    trajectory = jnp.broadcast_to(point, (spike_counts.shape[0], 1))
    temporal = pc.assess_population_spikes(code, trajectory, filtered, target)
    payload = {
        "neurons": population.neuron_count,
        "valid_fit": bool(code.least_squares.valid),
        "rank": int(code.least_squares.rank),
        "silent_neurons": int(jnp.sum(code.silent_neurons)),
        "condition_number": float(code.least_squares.condition_number),
        "held_out_rmse": approximation.rmse.tolist(),
        "temporal_rmse": {
            "rate_approximation": temporal.approximation.rmse.tolist(),
            "deterministic_filtering": temporal.filtering.rmse.tolist(),
            "spike_variability": temporal.spike_variability.rmse.tolist(),
            "total": temporal.total.rmse.tolist(),
        },
        "signed_decomposition_error": float(
            jnp.max(
                jnp.abs(
                    temporal.total.residual
                    - temporal.approximation.residual
                    - temporal.filtering.residual
                    - temporal.spike_variability.residual
                )
            )
        ),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
