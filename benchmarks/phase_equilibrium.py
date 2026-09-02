#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


def _model():
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("methane", "ethane"),
        (phx.equations.ChemicalPhaseKind.GAS,) * 2,
        jnp.asarray((0.016043, 0.03007)),
        ("C", "H"),
        jnp.asarray(((1, 2), (4, 6)), dtype=jnp.int32),
        jnp.zeros((2,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,),) * 2),
        jnp.zeros((2,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=1000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, species)
    parameters = phx.equations.PengRobinsonParameters(
        schema.catalog,
        jnp.asarray((190.564, 305.322)),
        jnp.asarray((4.5992e6, 4.8722e6)),
        jnp.asarray((0.01142, 0.0995)),
        jnp.zeros((2, 2)),
        provenance="public methane/ethane constants; zero interaction assumption",
    )
    return phx.equations.HomogeneousHelmholtzPlan(
        ideal,
        phx.equations.PengRobinsonResidualHelmholtzTerm(schema, parameters),
        maximum_molar_density=5.0e4,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    model = _model()
    plan = phx.solver.FixedTwoPhaseTPFlashPlan(model, tolerance=1.0e-6, maximum_steps=40)
    start = time.perf_counter()
    result = plan.solve(jnp.asarray(180.0), jnp.asarray(1.0e6), jnp.asarray((0.5, 0.5)))
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    payload = {
        "elapsed_ms": elapsed_ms,
        "status": int(result.status),
        "successful": bool(result.successful),
        "maximum_material_residual": float(jnp.max(jnp.abs(result.material_residual))),
        "maximum_fugacity_residual": float(jnp.max(jnp.abs(result.fugacity_residual))),
        "phase_fraction": [float(value) for value in result.phase_fraction],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
