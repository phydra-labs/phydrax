#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.applications.dark_matter._profiles import LayeredTerrestrialProfile
from phydrax.applications.dark_matter._rates import spherical_surface_crossings
from phydrax.applications.dark_matter._scattering import (
    BoundedThermalMarkSamplerPlan,
    ElasticScatteringTable,
)
from phydrax.applications.dark_matter._terrestrial import (
    layered_analytic_optical_depth,
    TerrestrialTransportPlan,
)
from phydrax.integration import WeightedSampleBatch
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.stochastic import PoissonClockRealization


def _manifest(name: str) -> ReferenceArtifactManifest:
    payload = name.encode()
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-Benchmark-Synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="benchmark-only",
        nondimensionalization={
            "length_m": 1.0,
            "thermal_sigma_cutoff": 6.0,
            "maxwellian_tail_probability_bound": 7.488376948795484e-08,
        },
        uncertainty={"relative": 0.0},
        lineage_ids=(f"synthetic:{name}",),
    )


def _inputs(path_count: int):
    profile = LayeredTerrestrialProfile(
        jnp.asarray((1.0, 2.0)),
        jnp.asarray((1.0, 1.0)),
        jnp.asarray(((0.4,), (0.8,))),
        jnp.asarray((100.0, 200.0)),
        ("target",),
        _manifest("dark-matter-transport-benchmark-profile"),
        frame_id="benchmark-body-frame",
    )
    speeds = jnp.asarray((0.0, 10.0))
    cross_sections = jnp.asarray(((0.5, 0.5),))
    rates = jnp.broadcast_to(
        cross_sections[:, None, :] * speeds[None, None, :], (1, 2, 2)
    )
    scattering = ElasticScatteringTable(
        ("target",),
        jnp.asarray((1.0,)),
        speeds,
        cross_sections,
        _manifest("dark-matter-transport-benchmark-scattering"),
        temperatures_K=jnp.asarray((100.0, 200.0)),
        rate_coefficients_m3_s=rates,
        mark_sampler=BoundedThermalMarkSamplerPlan(maximum_proposals=64),
    )
    impacts = jnp.linspace(-1.99, 1.99, path_count)
    positions = jnp.stack(
        (-3.0 * jnp.ones_like(impacts), impacts, jnp.zeros_like(impacts)), axis=-1
    )
    return profile, scattering, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", type=int, default=16384)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--jump-event-capacity", type=int, default=64)
    parser.add_argument("--guard-event-capacity", type=int, default=32)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.paths <= 1
        or arguments.repetitions <= 0
        or arguments.jump_event_capacity <= 0
        or arguments.guard_event_capacity <= 0
    ):
        raise ValueError(
            "paths must exceed one; repetitions and event capacities must be positive."
        )
    profile, scattering, positions = _inputs(arguments.paths)
    direction = jnp.asarray((1.0, 0.0, 0.0))

    action = eqx.filter_jit(
        jax.vmap(
            lambda position: layered_analytic_optical_depth(
                profile, scattering, position, direction, 6.0, 4.0
            )
        )
    )
    start = time.perf_counter()
    first = action(positions)
    jax.block_until_ready(first.total_optical_depth)
    compile_and_first_s = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(arguments.repetitions):
        result = action(positions)
    jax.block_until_ready(result.total_optical_depth)
    execution_s = (time.perf_counter() - start) / arguments.repetitions

    thresholds = jr.exponential(jr.key(7), (arguments.paths,))
    empirical_interaction = thresholds <= result.total_optical_depth
    cdf_error = jnp.abs(
        jnp.mean(empirical_interaction) - jnp.mean(result.interaction_cdf)
    )

    crossing_states = jnp.concatenate(
        (
            jnp.broadcast_to(jnp.asarray((2.0, 0.0, 0.0)), (arguments.paths, 3)),
            jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), (arguments.paths, 3)),
        ),
        axis=-1,
    )
    source = WeightedSampleBatch(
        crossing_states,
        -jnp.log(jnp.asarray(float(arguments.paths))) * jnp.ones((arguments.paths,)),
        sample_axes=0,
        provenance="synthetic-benchmark-unit-flux",
        independent=True,
    )
    crossings = spherical_surface_crossings(
        crossing_states[:, None, :],
        jnp.ones((arguments.paths, 1), dtype="bool"),
        source,
        2.0,
        direction="outward",
    )
    crossing_total = jnp.exp(crossings.diagnostics.log_total_flux_weight)
    event_path_count = min(arguments.paths, 512)
    event_states = jnp.broadcast_to(
        jnp.asarray((-3.0, 0.0, 0.0, 1.0, 0.0, 0.0)),
        (event_path_count, 6),
    )
    event_paths = WeightedSampleBatch(
        event_states,
        -jnp.log(jnp.asarray(float(event_path_count))) * jnp.ones((event_path_count,)),
        support_valid=jnp.asarray(True),
        sample_axes=0,
        provenance="synthetic-benchmark-transport-flux",
        independent=True,
    )
    transport = TerrestrialTransportPlan(
        profile,
        scattering,
        1.0,
        detector_depth_m=0.5,
        maximum_jump_events=arguments.jump_event_capacity,
        maximum_guard_events=arguments.guard_event_capacity,
    )
    clocks = PoissonClockRealization(
        jr.key(19),
        scattering.num_targets,
        support=(0.0, 6.0),
        max_events_per_channel=arguments.jump_event_capacity,
        sample_shape=(event_path_count,),
        process_id=transport.process.process_id,
    )
    event_result = transport.simulate(event_paths, clocks, jnp.asarray((0.0, 6.0)))
    deterministic = event_result.solution.deterministic_events
    if deterministic is None:
        raise RuntimeError("Guarded benchmark produced no deterministic evidence.")
    if not bool(jnp.all(event_result.evidence.successful)):
        raise RuntimeError("Transport benchmark contains invalid path evidence.")
    if not bool(event_result.detector_crossings.diagnostics.successful):
        raise RuntimeError("Transport benchmark detector crossings are invalid.")
    if not bool(crossings.diagnostics.successful):
        raise RuntimeError("Surface crossing benchmark evidence is invalid.")
    observed_jump_events = jnp.sum(event_result.solution.events.valid)
    observed_guard_events = jnp.sum(deterministic.valid)
    if not bool((observed_jump_events > 0) & (observed_guard_events > 0)):
        raise RuntimeError("Benchmark did not exercise both jump and guard capacities.")
    payload = {
        "path_count": arguments.paths,
        "event_path_count": event_path_count,
        "compile_and_first_ms": 1000.0 * compile_and_first_s,
        "execution_ms": 1000.0 * execution_s,
        "paths_per_second": arguments.paths / execution_s,
        "interaction_cdf_absolute_error": float(cdf_error),
        "surface_crossing_absolute_error": float(jnp.abs(crossing_total - 1.0)),
        "surface_crossing_effective_sample_size": float(
            crossings.diagnostics.effective_sample_size
        ),
        "jump_event_capacity": arguments.jump_event_capacity,
        "guard_event_capacity": arguments.guard_event_capacity,
        "observed_jump_events": int(observed_jump_events),
        "observed_guard_events": int(observed_guard_events),
        "jump_capacity_utilization": float(
            observed_jump_events / (event_path_count * arguments.jump_event_capacity)
        ),
        "guard_capacity_utilization": float(
            observed_guard_events / (event_path_count * arguments.guard_event_capacity)
        ),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
