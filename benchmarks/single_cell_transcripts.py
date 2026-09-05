#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native generate→observe→fit→held-out prediction, not external biological validation.

Run from the repository root with ``python benchmarks/single_cell_transcripts.py``.
SSA cold orchestration includes compilation; separately reported observation-map
lowering/compilation is measured directly, not inferred by subtracting timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict

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

from phydrax.applications.systems_biology import (
    CountMeasurementPlan,
    TelegraphGeneExpressionPlan,
)
from phydrax.applications.systems_biology.single_cell import (
    CellIdentity,
    fit_stationary_counts,
    GeneIdentity,
    generate_transcripts,
    observe_transcripts,
    PiecewiseConstantRates,
    predict_transcript_velocity,
    predicted_count_moments,
    ScenarioSegment,
    scheduled_transcript_mean,
    StationaryCountTarget,
    TranscriptCountAssay,
    TranscriptCounts,
    TranscriptScenario,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import derived_unit, SECOND


def _calibration():
    content = (
        b"Synthetic benchmark: exact independently specified telegraph rates "
        b"and binomial/Poisson observation coefficients."
    )
    return ReferenceArtifactManifest(
        "single-cell-synthetic-calibration",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"time_seconds": 1.0, "transcript_count": 1.0},
        uncertainty={"synthetic_coefficient_error": 0.0},
        lineage_ids=("independent-analytic-law",),
    )


def _split(counts, start, stop):
    return TranscriptCounts(
        counts.gene,
        counts.cell_ids[start:stop],
        counts.counts[start:stop],
        valid=counts.valid[start:stop],
        coordinates=counts.coordinates[start:stop],
        coordinate_semantics=counts.coordinate_semantics,
        time_unit=counts.time_unit,
        assay_id=counts.assay_id,
        source_id=counts.observation_id,
        preprocessing_id="disjoint-cell-split",
    )


def _target(counts):
    # Delta-method standard errors of empirical moments, using only measured cells.
    values = np.asarray(counts.counts)
    centered = values - values.mean(axis=0)
    influences = np.column_stack(
        (
            values,
            centered[:, 0] ** 2,
            centered[:, 1] ** 2,
            centered[:, 0] * centered[:, 1],
        )
    )
    errors = influences.std(axis=0, ddof=1) / np.sqrt(len(values))
    if np.any(errors <= 0):
        raise ValueError(
            "Too few informative sampled cells to estimate all moment uncertainties; increase --cells."
        )
    return StationaryCountTarget.from_counts(
        counts,
        errors,
        equilibrium_evidence_id="synthetic-constant-terminal-window;finite-time-bias-audited-below",
    )


def run(cells, genes, repeats, capacity):
    calibration = _calibration()
    assay = TranscriptCountAssay.from_plans(
        CountMeasurementPlan(0.7, 0.15, observation_capacity=1024),
        CountMeasurementPlan(0.6, 0.2, observation_capacity=1024),
        calibration,
    )
    rates = np.asarray([2.0, 3.0, 12.0, 4.0, 1.5])
    runtime = np.broadcast_to(rates, (2, genes, 5)).copy()
    runtime[0, :, 2] = 4.0
    runtime[:, :, 2] *= np.arange(1, genes + 1)[None, :]
    schedule = PiecewiseConstantRates(
        (0.0, 1.0, 12.0), runtime, rate_unit=derived_unit("per-second", ((SECOND, -1),))
    )
    scenario = TranscriptScenario(
        tuple(CellIdentity(10_000 + i, f"cell-{i}") for i in range(cells)),
        tuple(GeneIdentity(1_000 + g, f"gene-{g}") for g in range(genes)),
        (ScenarioSegment(11, schedule, (0.0, 1.0, 4.0, 12.0)),),
        np.zeros((cells, genes, 3)),
        max_paths=cells * genes,
        max_events_per_interval=capacity,
    )
    operation = lambda: generate_transcripts(scenario, jax.random.key(51))
    cold, cold_seconds = measure_synchronized(operation)
    experiment, execution = measure_repeated(operation, warmup=0, repeats=repeats)
    measured, observe_seconds = measure_synchronized(
        lambda: observe_transcripts(
            experiment, assay, jax.random.key(71), gene_id=1_000, segment_id=11
        )
    )
    training = _target(_split(measured, 0, cells // 2))
    held_out = _target(_split(measured, cells // 2, cells))
    initial = rates.copy()
    initial[2] = 5.0
    fit, fit_seconds = measure_synchronized(
        lambda: fit_stationary_counts(
            training,
            assay,
            initial,
            fixed_rates={0: rates[0], 1: rates[1], 3: rates[3], 4: rates[4]},
            rate_calibration=calibration,
        )
    )
    if not bool(fit.result.successful):
        raise RuntimeError(
            f"Native moment fit failed with status {int(fit.result.status)}."
        )
    velocity = predict_transcript_velocity(fit, held_out.observations)
    model = TelegraphGeneExpressionPlan(*rates).prepare()
    forward = jax.jit(lambda values: predicted_count_moments(model, values, assay))
    compiled, compilation = measure_lower_and_compile(
        lambda: forward.lower(jnp.asarray(rates)), lambda lowered: lowered.compile()
    )
    _, prediction_timing = measure_repeated(
        lambda: compiled(jnp.asarray(rates)), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="compiled-count-moment-map",
    )
    selected = tuple(p for p in experiment.paths if p.gene.gene_id == 1_000)
    latent_final = np.stack(tuple(np.asarray(p.latent.values[-1]) for p in selected))
    latent_boundary = np.stack(tuple(np.asarray(p.latent.values[1]) for p in selected))
    exact = np.asarray(scheduled_transcript_mean(schedule, np.zeros(3)))
    stationary = model.stationary_moments()
    stationary_mean = np.asarray(
        [stationary.promoter_mean, stationary.nascent_mean, stationary.mature_mean]
    )
    count_expected = np.asarray(predicted_count_moments(model, rates, assay))
    captures, backgrounds = np.asarray([0.7, 0.6]), np.asarray([0.15, 0.2])
    noise = np.asarray(measured.counts) - (latent_final[:, 1:] * captures + backgrounds)
    true_drift = (
        rates[3] * latent_final[cells // 2 :, 1]
        - rates[4] * latent_final[cells // 2 :, 2]
    )
    return {
        "profile": "synthetic-exact-piecewise-telegraph;calibrated-count-moment-fit",
        "cells": cells,
        "genes": genes,
        "paths": len(experiment.paths),
        "intervals_per_path": 2,
        "event_capacity_per_interval": capacity,
        "active_events": sum(
            int(jnp.sum(s.events.valid)) for p in experiment.paths for s in p.intervals
        ),
        "ssa_cold_orchestration_including_compilation_seconds": cold_seconds,
        "ssa_repeated_orchestration": execution.to_dict(),
        "observe_seconds": observe_seconds,
        "fit_seconds": fit_seconds,
        "count_prediction_compilation": asdict(compilation),
        "count_prediction_execution": prediction_timing.to_dict(),
        "count_prediction_compiler": asdict(evidence),
        "logical_result_bytes": logical_array_bytes(experiment),
        "boundary_mean_error": (latent_boundary.mean(axis=0) - exact[1]).tolist(),
        "terminal_mean_error": (latent_final.mean(axis=0) - exact[-1]).tolist(),
        "terminal_mean_monte_carlo_standard_error": (
            latent_final.std(axis=0, ddof=1) / np.sqrt(cells)
        ).tolist(),
        "terminal_nonstationary_mean_bias": (exact[-1] - stationary_mean).tolist(),
        "stationary_moments": np.asarray(stationary.fitting_vector).tolist(),
        "expected_observed_stationary_moments": count_expected.tolist(),
        "held_out_standardized_moment_residuals": np.asarray(
            fit.held_out_residuals(held_out)
        ).tolist(),
        "capture_noise_mean": noise.mean(axis=0).tolist(),
        "capture_cross_channel_noise_covariance": float(np.cov(noise.T)[0, 1]),
        "held_out_drift_rmse_against_stored_truth_for_benchmark_only": float(
            np.sqrt(np.mean((np.asarray(velocity.estimates) - true_drift) ** 2))
        ),
        "fit_rank": fit.identifiability.rank,
        "fit_free_parameters": len(fit.identifiability.free_parameter_indices),
        "scientific_gates": [
            "No experimental calibration/biological-timescale validation is claimed.",
            "Velocity is a count-derived drift estimate, not a path derivative or lineage.",
            "Moment-fit covariance is first-order conditional covariance, not a posterior.",
        ],
        "cold_paths": len(cold.paths),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--genes", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--event-capacity", type=int, default=1024)
    args = parser.parse_args()
    if args.cells < 16 or args.genes < 1:
        parser.error(
            "Use at least 16 cells and one gene; meaningful empirical uncertainty needs replicate cells."
        )
    result = run(args.cells, args.genes, args.repeats, args.event_capacity)
    print(
        json.dumps(
            {"environment": capture_environment().to_dict(), "benchmark": result},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
