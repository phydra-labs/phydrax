#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytical labelled A/T CTMC: exact exp(Q t), first hits and native SSA.

Run with ``python -m benchmarks.nucleic_secondary_kinetics``. This independently
specified mathematical model measures numerical behavior, not experimental DNA
kinetics. No third-party parameter tables are bundled or scientific calibration
claimed. Increase ``--copies`` to test competing labelled binding partners.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.secondary_kinetics import (
    AssociationConvention,
    prepare_secondary_kinetics,
    SecondaryEnergyModel,
    SecondaryRateLaw,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.solver._jump import _direct_ssa_paths, solve_direct_ssa
from phydrax.solver._jump_hitting import event_first_hit, finite_generator_hitting
from phydrax.stochastic import PoissonClockRealization


def run(*, paths: int, copies: int, capacity: int, repeats: int) -> dict:
    if paths <= 0 or copies <= 0 or capacity <= 0 or repeats <= 0:
        raise ValueError("All benchmark sizes must be positive.")
    content = json.dumps(
        {
            "profile": "pair_loop",
            "chemistry": "DNA",
            "pairing_rule": "watson_crick",
            "temperature": 300.0,
            "energy_convention": "dimensionless_molar_G_over_RT",
            "minimum_hairpin_unpaired": 3,
            "pair_energies": {"AT": math.log(1.5)},
            "stack_energies": {},
            "hairpin_energies": {},
            "bulge_energies": {},
            "internal_energies": {},
            "multibranch": [0.0, 0.0, 0.0],
            "association_initiation": 0.0,
        },
        sort_keys=True,
    ).encode()
    manifest = ReferenceArtifactManifest(
        "independent-analytical-labelled-binding-benchmark",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"temperature_kelvin": 300.0},
        uncertainty={"analytical_definition": 0.0},
        lineage_ids=("independently-defined-equations",),
    )
    model = SecondaryEnergyModel.from_bytes(
        content,
        manifest,
        requested_use={
            "commercial_use": True,
            "redistribution": False,
            "training_use": False,
            "export": False,
        },
    )
    construct = NucleicAcidConstruct(
        tuple(["a"] + [f"t{i}" for i in range(copies)]),
        tuple(["A"] + ["T"] * copies),
        ("DNA",) * (copies + 1),
        (False,) * (copies + 1),
    )
    prepared = prepare_secondary_kinetics(
        construct,
        model,
        AssociationConvention(
            mode="fixed_volume",
            standard_concentration=1000.0,
            volume=1 / (1000 * 6.02214076e23),
        ),
        SecondaryRateLaw("association_metropolis", 3.0, 3.0),
        temperature=300.0,
    )
    initial = prepared.encode(prepared.states[0])
    target = prepared.pair_count_target(1)
    duration = 1.0
    clocks = PoissonClockRealization(
        jr.key(731),
        prepared.process.num_channels,
        support=(0.0, duration),
        max_events_per_channel=capacity,
        sample_shape=(paths,),
        process_id=prepared.process.process_id,
    )
    start, end = jnp.asarray(0.0, dtype=float), jnp.asarray(duration, dtype=float)
    # Separate native SSA kernel lowering/compilation; execution below still uses
    # the public solver and its event-ledger/status assembly, not a second SSA.
    _, compilation = measure_lower_and_compile(
        lambda: _direct_ssa_paths.lower(
            prepared.process,
            initial,
            start,
            end,
            clocks.direct_event_keys,
            clocks.mark_keys,
            None,
            capacity,
        ),
        lambda lowered: lowered.compile(),
    )
    solve = lambda: solve_direct_ssa(
        prepared.process,
        clocks,
        initial,
        t0=0.0,
        t1=duration,
        save_times=jnp.asarray([0.0, duration]),
        max_events=capacity,
    )
    _, first_call_seconds = measure_synchronized(solve)
    solution, repeated = measure_repeated(solve, warmup=0, repeats=repeats)
    hits = event_first_hit(solution, initial, target, t0=0.0, t1=duration)
    generator = prepared.generator()
    exact = generator.transition_matrix(duration)[0]
    exact_hitting = finite_generator_hitting(generator, target.mask)
    valid = np.asarray(solution.successful)
    final = np.asarray(solution.states[:, -1, 0])
    observed = np.bincount(final[valid], minlength=len(prepared.states)) / max(
        1, int(valid.sum())
    )
    grid = np.linspace(0, duration, 9)
    # Unobserved incomplete paths form a CDF interval, never false censoring.
    hit_times = np.asarray(hits.time)
    cdf_lower = np.asarray([(hit_times <= time).mean() for time in grid])
    incomplete = np.asarray(hits.incomplete)
    observation_end = np.asarray(hits.observation_end)
    cdf_upper = cdf_lower + np.asarray(
        [(incomplete & (observation_end < time)).mean() for time in grid]
    )
    exact_cdf = 1 - np.exp(-2 * copies * grid)
    se = np.sqrt(exact_cdf * (1 - exact_cdf) / paths)
    leaked = prepared.generator((prepared.states[0],), boundary_policy="leak")
    return {
        "scientific_profile": "independent analytical CTMC; no experimental calibration",
        "environment": capture_environment().to_dict(),
        "paths": paths,
        "strand_copies": copies + 1,
        "active_states": len(prepared.states),
        "channels": prepared.process.num_channels,
        "event_capacity": capacity,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first_public_solve_seconds": first_call_seconds,
        "repeated_execution": repeated.to_dict(),
        "logical_array_bytes": logical_array_bytes((prepared.process, solution)),
        "successful_paths": int(valid.sum()),
        "capacity_failures": int(np.asarray(hits.capacity_failure).sum()),
        "censored_paths": int(np.asarray(hits.censored).sum()),
        "incomplete_first_hits": int(incomplete.sum()),
        "maximum_observed_event_count": int(np.asarray(solution.events.counts).max()),
        "exact_final_probabilities": np.asarray(exact).tolist(),
        "empirical_final_probabilities_successful_only": observed.tolist(),
        "final_probability_max_error": float(np.max(np.abs(observed - np.asarray(exact))))
        if valid.all()
        else None,
        "final_probability_comparison_qualified": bool(valid.all()),
        "first_hit_grid": grid.tolist(),
        "exact_first_hit_cdf": exact_cdf.tolist(),
        "empirical_first_hit_cdf_lower": cdf_lower.tolist(),
        "empirical_first_hit_cdf_upper": cdf_upper.tolist(),
        "first_hit_monte_carlo_standard_error": se.tolist(),
        "exact_initial_mfpt": float(exact_hitting.mean_first_passage_time[0]),
        "hitting_probability_residual": float(exact_hitting.probability_residual),
        "mfpt_residual": float(exact_hitting.mfpt_residual),
        "generator_row_sum_residual": float(
            jnp.max(jnp.abs(generator.matrix.sum(axis=1)))
        ),
        "closed_support_escaped_rates": np.asarray(generator.escaped_rates).tolist(),
        "omitted_support_escaped_rates": np.asarray(leaked.escaped_rates).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", type=int, default=2048)
    parser.add_argument("--copies", type=int, default=1)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()
    print(
        json.dumps(
            run(
                paths=options.paths,
                copies=options.copies,
                capacity=options.capacity,
                repeats=options.repeats,
            ),
            indent=2,
        )
    )
