# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Analytic electronic compiler/solver benchmark, not DNA parameter calibration.

Run from the repository root with JAX_ENABLE_X64=true:
    .venv/bin/python -m benchmarks.nucleic_electronics --sizes 2 4 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.electronics import (
    electronic_coherences,
    electronic_populations,
    ElectronicChannel,
    ElectronicParameterArtifact,
    ElectronicSiteGraph,
    evolve_electronic_jumps,
    evolve_electronics,
    prepare_electronics,
)
from phydrax.atomistic import AtomisticScaleContract, AtomisticUnitSystem
from phydrax.discretization import TemporalMesh
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.solver import integrate_finite_cptp
from phydrax.units import (
    ANGSTROM,
    conversion_factor,
    DALTON,
    derived_unit,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
    FEMTOSECOND,
    JOULE,
    KELVIN,
    PICOSECOND,
)


USE = dict(commercial_use=False, redistribution=False, training_use=False, export=False)
UNITS = AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
PER_FS = derived_unit("1/fs", ((FEMTOSECOND, -1),))


def fixture(size, *, dephasing=0.0, coupling=True, units=UNITS, energy_unit=ELECTRONVOLT):
    construct = NucleicAcidConstruct(("benchmark",), ("A" * size,), ("DNA",), (False,))
    sites = tuple(10_000 + index * 7 for index in range(size))
    graph = ElectronicSiteGraph(
        construct,
        sites,
        construct.nucleotide_keys,
        ("pi",) * size,
        tuple(zip(sites[:-1], sites[1:], strict=True)),
    )
    keys = tuple((site,) for site in sites)
    energy = UNITS.reduced_planck_constant * float(
        conversion_factor(ELECTRONVOLT, energy_unit)
    )
    edges = (
        tuple(
            (left, right, energy) for left, right in zip(keys[:-1], keys[1:], strict=True)
        )
        if coupling
        else ()
    )
    channels = (
        tuple(
            ElectronicChannel(f"local-{index}", "dephasing", key, None, dephasing, PER_FS)
            for index, key in enumerate(keys)
        )
        if dephasing
        else ()
    )
    declaration = {
        "keys": keys,
        "diagonal": [0.0] * size,
        "edges": edges,
        "channels": [channel.record() for channel in channels],
        "unit": energy_unit.unit_id,
        "scope": "analytic tight-binding chain, not a published DNA parameter set",
    }
    raw = json.dumps(declaration, sort_keys=True).encode()
    manifest = ReferenceArtifactManifest(
        "independently-declared-electronic-benchmark",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(raw).hexdigest(),
        size_bytes=len(raw),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_J": float(energy_unit.scale_to_reference)},
        uncertainty=None,
        lineage_ids=("analytic-numerical-benchmark-not-experimental-reference",),
    )
    parameters = ElectronicParameterArtifact(
        keys,
        (0.0,) * size,
        edges,
        channels,
        energy_unit,
        declaration["scope"],
        "real orbital gauge",
        manifest,
        raw,
    )
    return prepare_electronics(graph, parameters, units=units, requested_use=USE)


def density(state):
    return state[:, None] * jnp.conj(state[None, :])


def analytic_cases(trajectory_count):
    model = fixture(2)
    initial = model.basis_state(model.basis_keys[0])
    coherent = evolve_electronics(
        model,
        density(initial),
        step_size=0.05,
        time_unit=FEMTOSECOND,
        steps=20,
        requested_use=USE,
    )
    times = coherent.densities.support.coordinates
    analytic_error = float(
        jnp.max(
            jnp.abs(
                electronic_populations(model, coherent.densities.values)[:, 1]
                - jnp.sin(times) ** 2
            )
        )
    )
    dephasing = fixture(2, dephasing=0.4, coupling=False)
    plus = jnp.ones(2, dtype=complex) / jnp.sqrt(2.0)
    decoherence = evolve_electronics(
        dephasing,
        density(plus),
        step_size=0.05,
        time_unit=FEMTOSECOND,
        steps=20,
        requested_use=USE,
        method="cptp",
    )
    observed = electronic_coherences(
        dephasing, decoherence.densities.values, (dephasing.graphs[0].site_ids,)
    )[:, 0]
    dephasing_error = float(jnp.max(jnp.abs(observed - 0.5 * jnp.exp(-0.4 * times))))
    ps_units = AtomisticUnitSystem(
        AtomisticScaleContract(ANGSTROM, JOULE),
        mass_unit=DALTON,
        time_unit=PICOSECOND,
        charge_unit=ELEMENTARY_CHARGE,
        temperature_unit=KELVIN,
        constant_set_id="codata-2018",
    )
    equivalent = fixture(2, units=ps_units, energy_unit=JOULE)
    converted = evolve_electronics(
        equivalent,
        density(initial),
        step_size=0.00005,
        time_unit=PICOSECOND,
        steps=20,
        requested_use=USE,
    )
    unit_error = float(
        jnp.max(jnp.abs(converted.densities.values - coherent.densities.values))
    )
    jumps, jump_seconds = measure_synchronized(
        lambda: evolve_electronic_jumps(
            dephasing,
            plus,
            jax.random.PRNGKey(8701),
            step_size=0.005,
            time_unit=FEMTOSECOND,
            steps=200,
            trajectory_count=trajectory_count,
            requested_use=USE,
        )
    )
    unraveling_error = float(
        jnp.max(
            jnp.abs(jumps.mean_densities.values[-1] - decoherence.densities.values[-1])
        )
    )
    return {
        "coherent_population_max_error": analytic_error,
        "dephasing_coherence_max_error": dephasing_error,
        "unit_equivalence_density_max_error": unit_error,
        "quantum_jump_density_max_error": unraveling_error,
        "quantum_jump_total_seconds": jump_seconds,
        "quantum_jump_count": trajectory_count,
        "quantum_jump_step_fs": 0.005,
        "quantum_jump_statistical_scale": 1 / np.sqrt(trajectory_count),
        "quantum_jump_valid": bool(jumps.native_result.valid),
        "density_valid": bool(
            coherent.native_result.valid
            & decoherence.native_result.valid
            & converted.native_result.valid
        ),
        "qualification": "analytic numerical laws only; no experimental DNA calibration",
    }


def scaling_case(size, repeats, steps):
    model, preparation_seconds = measure_synchronized(
        lambda: fixture(size, dephasing=0.02)
    )
    initial = density(model.basis_state(model.basis_keys[0]))
    # Prepare the native finite ABI outside tracing; its rates and jumps remain
    # dynamic numeric inputs. No duplicate solver or application propagator.
    slicing = TemporalMesh.uniform(0.0, 0.1 * steps, steps)
    plan = model.finite_plan(slicing, time_unit=FEMTOSECOND)
    kernel = jax.jit(lambda rho, native_plan: integrate_finite_cptp(native_plan, rho))
    compiled, compilation = measure_lower_and_compile(
        lambda: kernel.lower(initial, plan), lambda lowered: lowered.compile()
    )
    result, execution = measure_repeated(
        lambda: compiled(initial, plan), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-finite-cptp",
    )
    return {
        "sites": size,
        "basis_dimension": model.dimension,
        "channel_capacity": int(model.rates.shape[0]),
        "active_channels": int(jnp.sum(model.active_jumps)),
        "liouville_elements": model.dimension**4,
        "steps": steps,
        "preparation_seconds": preparation_seconds,
        "compilation": asdict(compilation),
        "execution": execution.to_dict(),
        "compiler": asdict(evidence),
        "model_logical_array_bytes": logical_array_bytes(model),
        "result_logical_array_bytes": logical_array_bytes(result),
        "trace_residual": float(jnp.max(result.density_trace_residuals)),
        "hermiticity_residual": float(jnp.max(result.density_hermiticity_residuals)),
        "minimum_density_eigenvalue": float(jnp.min(result.density_minimum_eigenvalues)),
        "minimum_choi_eigenvalue": float(jnp.min(result.cp_margins)),
        "valid": bool(result.valid),
        "method_claim": result.method_claim,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--trajectories", type=int, default=1024)
    args = parser.parse_args()
    if not jax.config.x64_enabled:
        raise ValueError("Run this physical-unit benchmark with JAX_ENABLE_X64=true.")
    print(
        json.dumps(
            {
                "environment": capture_environment().to_dict(),
                "analytic": analytic_cases(args.trajectories),
                "scaling": [
                    scaling_case(size, args.repeats, args.steps) for size in args.sizes
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
