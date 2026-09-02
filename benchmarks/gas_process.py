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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    schema = phx.equations.ChemicalSpeciesSchema.from_unique_species(
        ("air",),
        (phx.equations.ChemicalPhaseKind.GAS,),
        jnp.asarray((0.02897,)),
        ("air",),
        jnp.asarray(((1,),), dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        gas_standard_pressure=1.0e5,
    )
    species = phx.equations.PolynomialSpeciesThermodynamicsPlan(
        schema,
        jnp.asarray(((2.5 * phx.equations.UNIVERSAL_GAS_CONSTANT,),)),
        jnp.zeros((1,)),
        reference_temperature=300.0,
        minimum_temperature=100.0,
        maximum_temperature=2000.0,
    )
    ideal = phx.equations.IdealGasReferenceHelmholtzTerm(schema, species)
    model = phx.equations.HomogeneousHelmholtzPlan(
        ideal, phx.equations.ZeroResidualHelmholtzTerm(schema)
    )
    tf = phx.applications.thermofluids
    performance_map = tf.CompressorMapPlan(
        jnp.asarray((0.8, 1.0)),
        jnp.asarray((0.0, 1.0)),
        jnp.asarray(((8.0, 9.0), (10.0, 11.0))),
        jnp.asarray(((1.5, 1.6), (2.0, 2.1))),
        jnp.asarray(((0.75, 0.76), (0.8, 0.81))),
        reference_temperature=288.15,
        reference_pressure=101325.0,
        provenance="synthetic benchmark map",
    )
    compressor = tf.CompressorPlan(model, performance_map)
    design = compressor.design(
        jnp.asarray(1.0),
        jnp.asarray(0.0),
        corrected_flow=10.0,
        pressure_ratio=2.0,
        isentropic_efficiency=0.8,
    )
    inlet = compressor.station(
        jnp.asarray(300.0),
        jnp.asarray(1.0e5),
        jnp.asarray(1.0),
        jnp.asarray((1.0,)),
    )
    start = time.perf_counter()
    result = compressor.evaluate(inlet, jnp.asarray(1.0), jnp.asarray(0.0), design)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    payload = {
        "elapsed_ms": elapsed_ms,
        "successful": bool(result.successful),
        "outlet_pressure": float(result.outlet.total_pressure),
        "shaft_power": float(result.shaft_power),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
