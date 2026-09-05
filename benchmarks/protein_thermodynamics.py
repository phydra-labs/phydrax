# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Analytical sampling control for paired-state enthalpy and experimental closure.

This benchmark checks estimator throughput/errors, not force-field accuracy.
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
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)

from phydrax.applications.protein_folding.experiments import ThermodynamicConvention
from phydrax.applications.protein_folding.thermodynamics import (
    close_free_energy_at_reference,
    EnthalpyReplica,
    fit_heat_capacity_slope,
    paired_state_enthalpy,
    ProteinEnsembleComposition,
)
from phydrax.applications.protein_folding.workflows import ProteinFreeEnergyWorkflow
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.series import SampledSeries, SeriesSupport
from phydrax.units import KILOJOULE_PER_MOLE, PICOSECOND


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1024)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    rng = np.random.default_rng(2026)
    convention = ThermodynamicConvention()
    composition = ProteinEnsembleComposition(
        "analytical-fixed-construct",
        "fixed-chemical-composition",
        (("protein", 1), ("water", 1000), ("counterion", 1)),
        "analytical-sampling-control",
    )
    ensembles = []
    for basin in ("folded-control", "unfolded-control"):
        replicas = []
        for temperature in (280.0, 290.0, 300.0, 310.0, 320.0):
            for replica in range(args.replicas):
                identity = f"independent-iid:{basin}:{temperature}:{replica}"
                mean = 1000.0 + (
                    30.0 + 0.5 * (temperature - 300.0)
                    if basin == "unfolded-control"
                    else 0.0
                )
                values = mean + rng.normal(size=args.samples)
                source = ScientificArtifactEnvelope(
                    artifact_kind="analytical-ensemble-control",
                    content_digest=hashlib.sha256(values.tobytes()).hexdigest(),
                    producer="independent-normal-control",
                    producer_version="native",
                    build_id="known-mean-and-variance",
                    license_id="CC0-1.0",
                    resource_id=identity,
                    status="complete",
                )
                series = SampledSeries(
                    SeriesSupport(
                        jnp.arange(args.samples, dtype=float),
                        coordinate_id=PICOSECOND.unit_id,
                    ),
                    values,
                    series_id=identity,
                )
                replicas.append(
                    EnthalpyReplica(
                        series,
                        composition,
                        basin,
                        temperature,
                        KILOJOULE_PER_MOLE,
                        PICOSECOND,
                        identity,
                        identity,
                        source,
                        32,
                        0.01,
                        "iid-zero-correlation-with-positive-upper-bound",
                        "stationary-iid-control",
                        "matched-pressure-control",
                    )
                )
        ensembles.append(tuple(replicas))
    estimate, estimation_seconds = measure_synchronized(
        lambda: paired_state_enthalpy(*ensembles, convention=convention)
    )
    execute = jax.jit(fit_heat_capacity_slope)
    compiled, compilation = measure_lower_and_compile(
        lambda: execute.lower(estimate), lambda lowered: lowered.compile()
    )
    fit, repeated = measure_repeated(
        lambda: compiled(estimate), warmup=1, repeats=args.repeats
    )
    reference = ReferenceArtifactManifest(
        "synthetic-melting-control",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(b"Tm=300K;standard_error=0.5K").hexdigest(),
        size_bytes=27,
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"kelvin": 1.0},
        uncertainty={"temperature_standard_error_kelvin": 0.5},
        lineage_ids=("synthetic-not-experimental-data",),
    )
    closed = close_free_energy_at_reference(
        fit,
        jnp.asarray([280.0, 300.0, 320.0]),
        reference_temperature=300.0,
        reference_delta_g=0.0,
        experimental_covariance=[[0.25, 0.0], [0.0, 0.0]],
        reference=reference,
        closure_kind="measured-melting-temperature",
    )
    free_energy = ProteinFreeEnergyWorkflow(
        ("offset-A", "offset-B"),
        composition.fingerprint(),
        300.0,
        convention,
        "analytical-energy-offset-control",
        "iid-control",
    ).fep(jnp.full(1024, 2.4), energy_unit=KILOJOULE_PER_MOLE)
    print(
        json.dumps(
            {
                "claim": "analytical-estimator-control-not-protein-thermodynamic-validation",
                "samples_per_replica": args.samples,
                "replicas_per_state_temperature": args.replicas,
                "temperatures": np.asarray(estimate.temperatures).tolist(),
                "delta_h_error_kJ_per_mol": np.asarray(
                    estimate.delta_enthalpy
                    - (30.0 + 0.5 * (estimate.temperatures - 300.0))
                ).tolist(),
                "delta_h_standard_errors": np.asarray(estimate.standard_errors).tolist(),
                "delta_cp_error_kJ_per_mol_kelvin": float(fit.delta_heat_capacity - 0.5),
                "linear_model_residuals": np.asarray(fit.residuals).tolist(),
                "experimental_closure_dependency": closed.experimental_dependencies,
                "closed_g_standard_errors": np.asarray(closed.standard_errors).tolist(),
                "constant_offset_fep_error": float(free_energy.free_energies[1] - 2.4),
                "estimation_seconds": estimation_seconds,
                "slope_compilation": asdict(compilation),
                "slope_execution": repeated.to_dict(),
                "environment": capture_environment().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
