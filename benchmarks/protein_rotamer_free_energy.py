"""Exact/BP scalar, marginal and force errors on analytic uncalibrated rotamers.

Run: .venv/bin/python benchmarks/protein_rotamer_free_energy.py --sizes 3 6 9
This proves the numerical bridge only. No biological parameter corpus is bundled.
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
)

from phydrax.applications.protein_folding._construct import ProteinConstruct
from phydrax.applications.protein_folding.potentials import (
    RotamerFreeEnergyTerm,
    RotamerGeometryPlan,
    RotamerParameterPlan,
)
from phydrax.atomistic import AtomisticSystemPlan, AtomisticUnitSystem
from phydrax.qualification import ReferenceArtifactManifest


def numerical_model(size, loop, method, tolerance):
    units = AtomisticUnitSystem.reduced()
    ids = np.arange(3 * size, dtype=np.int64) * 17 + 100
    positions = jnp.asarray(
        [
            [1.3 * i + dx, dy, 0.2 * i]
            for i in range(size)
            for dx, dy in ((0, 0), (1, 0), (0, 1))
        ]
    )
    cards = tuple(2 + i % 2 for i in range(size))
    payload = b"Analytical reduced Gaussian numerical benchmark; no biological calibration or data."
    source = ReferenceArtifactManifest(
        "analytical-rotamer-numerical-benchmark",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="benchmark-author-owned",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted-analytical-fixture",
        nondimensionalization={"length": 1.0, "energy": 1.0},
        uncertainty=None,
        lineage_ids=("numerical-bridge-only",),
    )
    geometry = RotamerGeometryPlan(
        ProteinConstruct(("A",), ("A" * size,)),
        ids.reshape((size, 3)),
        tuple(
            np.asarray([[0.1 * k, -0.2 * k, 0.4 * (-1) ** k] for k in range(card)])
            for card in cards
        ),
        source,
        units=units,
    )
    pairs = tuple((i, i + 1) for i in range(size - 1)) + (
        ((0, size - 1),) if loop else ()
    )
    parameters = RotamerParameterPlan(
        units,
        1.0,
        tuple(np.linspace(-0.1, 0.2, card) for card in cards),
        pairs,
        tuple(
            0.2
            * np.asarray(
                [[(-1) ** (i + j) for j in range(cards[b])] for i in range(cards[a])]
            )
            for a, b in pairs
        ),
        tuple(np.full((cards[a], cards[b]), 1.3) for a, b in pairs),
        source,
    )
    system = AtomisticSystemPlan(
        ids, np.full(3 * size, 6), np.ones(3 * size), units
    ).prepare()
    term = RotamerFreeEnergyTerm(
        geometry,
        parameters,
        ids[::3],
        np.full(size, 1.0 / size),
        sampling_temperature=1.0,
        inference_method=method,
        absolute_tolerance=tolerance,
        relative_tolerance=tolerance,
    ).prepare(system)
    return term, positions


def compile_execution(term, positions):
    def scalar(q):
        result = term.evaluate(q)
        if term.exact is None:
            diagnostics = result.inference.inference.diagnostics
            residual = diagnostics.final_residual
            iterations = diagnostics.iterations
        else:
            residual = jnp.asarray(0.0)
            iterations = jnp.asarray(0)
        return result.energy, (
            result.variable_probabilities,
            result.successful,
            residual,
            iterations,
            result.contraction_bound,
            result.status,
        )

    function = jax.jit(jax.value_and_grad(scalar, has_aux=True))
    executable, compilation = measure_lower_and_compile(
        lambda: function.lower(positions),
        lambda lowered: lowered.compile(),
    )
    evidence = compiler_evidence(
        executable.cost_analysis(),
        executable.memory_analysis(),
        source="jax-compiled-executable",
    )
    return executable, compilation, evidence


def run_case(size, loop, tolerances, repeats):
    reference, positions = numerical_model(size, loop, "exact", 1e-12)
    executable, compilation, memory = compile_execution(reference, positions)
    exact, timings = measure_repeated(
        lambda: executable(positions), warmup=1, repeats=repeats
    )
    (exact_energy, (exact_marginals, successful, _, _, _, _)), exact_gradient = exact
    if not bool(successful):
        raise RuntimeError("Exact numerical reference failed.")
    rows = [
        {
            "method": "exact",
            "compilation": asdict(compilation),
            "execution": timings.to_dict(),
            "compiler_evidence": asdict(memory),
            "logical_plan_bytes": logical_array_bytes(reference),
            "energy": float(exact_energy),
            "configurations": int(np.prod(reference.plan.geometry.cardinalities)),
            "successful": True,
        }
    ]
    for tolerance in tolerances:
        term, _ = numerical_model(size, loop, "bethe", tolerance)
        executable, compilation, memory = compile_execution(term, positions)
        value, timings = measure_repeated(
            lambda: executable(positions), warmup=1, repeats=repeats
        )
        (
            (energy, (marginals, successful, residual, iterations, bound, status)),
            gradient,
        ) = value
        rows.append(
            {
                "method": "bethe",
                "tolerance": tolerance,
                "normalizer_kind": "bethe" if loop else "exact-tree",
                "compilation": asdict(compilation),
                "execution": timings.to_dict(),
                "compiler_evidence": asdict(memory),
                "logical_plan_bytes": logical_array_bytes(term),
                "message_entries": term.bp.message_count,
                "energy": float(energy),
                "absolute_energy_error": float(jnp.abs(energy - exact_energy)),
                "maximum_marginal_error": float(
                    jnp.max(jnp.abs(marginals - exact_marginals))
                ),
                "maximum_force_error": float(jnp.max(jnp.abs(gradient - exact_gradient))),
                "root_residual": float(residual),
                "iterations": int(iterations),
                "contraction_bound": float(bound),
                "successful": bool(successful),
                "status": int(status),
            }
        )
    return {
        "residues": size,
        "active_atoms": 3 * size,
        "capacity": 3 * size,
        "graph": "loop" if loop else "tree",
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[3, 6, 9])
    parser.add_argument("--tolerances", nargs="+", type=float, default=[1e-5, 1e-9])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if any(size < 3 for size in args.sizes):
        parser.error("sizes must be at least three for loop comparisons")
    jax.config.update("jax_enable_x64", True)
    results = [
        run_case(size, loop, args.tolerances, args.repeats)
        for size in args.sizes
        for loop in (False, True)
    ]
    print(
        json.dumps(
            {
                "environment": capture_environment().to_dict(),
                "qualification": "analytical numerical bridge only",
                "biological_acceptance_gate": (
                    "missing right-cleared calibrated mini-protein parameters "
                    "and independent measured validation corpus"
                ),
                "branch_policy": "zero messages; globally contraction-qualified loops; no warm starts",
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
