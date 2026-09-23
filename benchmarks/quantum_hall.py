#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from time import perf_counter

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.operators.quantum import evaluate_local_operator
from phydrax.units import ANGSTROM, ELECTRONVOLT


_ENERGY_SCALE = qh.QuantumHallEnergyScale(
    ELECTRONVOLT,
    1.602_176_634e-19,
    "electronvolt",
)


def _elapsed(function):
    start = perf_counter()
    value = function()
    jax.block_until_ready(value)
    return value, perf_counter() - start


def lattice(mesh: int):
    plan = qh.HaldaneModelPlan(
        1.0,
        0.0,
        0.2,
        np.pi / 2.0,
        1.0,
        ELECTRONVOLT,
        ANGSTROM,
    )
    result, elapsed = _elapsed(
        lambda: qh.evaluate_haldane_topology(plan, mesh_shape=(mesh, mesh))
    )
    return {
        "route": "lattice",
        "mesh": mesh,
        "elapsed_seconds": elapsed,
        "chern": int(np.asarray(result.chern.nearest_integer)),
        "quantization_residual": float(np.asarray(result.chern.quantization_residual)),
    }


def sphere(particles: int):
    flux = 3 * (particles - 1)
    sphere_plan = qh.HaldaneSpherePlan(
        particles,
        qh.MonopoleLandauLevel(flux, 0, qh.SPIN_POLARIZED_ELECTRON),
        "fermion",
        _ENERGY_SCALE,
    )
    interaction = qh.HaldanePseudopotentialPlan(
        sphere_plan,
        {index: float(index == 1) for index in range(1, flux + 1, 2)},
        "v1-parent",
    )
    start = perf_counter()
    prepared = qh.prepare_haldane_sphere_hamiltonian(
        interaction,
        twice_projection=0 if particles * flux % 2 == 0 else 1,
    )
    elapsed = perf_counter() - start
    vector = jax.random.normal(
        jr.key(0), (prepared.many_body.dimension,), dtype=jnp.float64
    ).astype(jnp.complex128)
    _, warm = _elapsed(lambda: prepared.many_body.operator.mv(vector))
    return {
        "route": "sphere",
        "particles": particles,
        "flux": flux,
        "dimension": prepared.many_body.dimension,
        "routes": prepared.many_body.evidence.nonzero_routes,
        "preparation_seconds": elapsed,
        "warmed_matvec_seconds": warm,
    }


def vmc(particles: int):
    flux = 3 * (particles - 1)
    sphere_plan = qh.HaldaneSpherePlan(
        particles,
        qh.MonopoleLandauLevel(flux, 0, qh.SPIN_POLARIZED_ELECTRON),
        "fermion",
        _ENERGY_SCALE,
    )
    prepared = qh.prepare_landau_level_mixing_vmc(
        qh.LandauLevelMixingVMCPlan(
            sphere_plan,
            0.5,
            chain_count=4,
            hidden_dimension=8,
            layer_count=1,
            determinant_count=2,
        ),
        jr.key(0),
    )
    local, elapsed = _elapsed(
        lambda: evaluate_local_operator(
            prepared.model,
            prepared.operator,
            prepared.problem.initial_configurations,
        )
    )
    return {
        "route": "vmc",
        "particles": particles,
        "elapsed_seconds": elapsed,
        "valid": int(np.sum(np.asarray(local.successful))),
    }


def higher_landau(level: int):
    start = perf_counter()
    result = qh.planar_coulomb_pseudopotentials(
        level,
        2 * level + 9,
        quadrature_order=128,
    )
    elapsed = perf_counter() - start
    return {
        "route": "higher-landau",
        "level": level,
        "channels": len(result.relative_channels),
        "refinement_residual": float(np.asarray(result.refinement_residual)),
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("route", choices=("higher", "lattice", "sphere", "vmc"))
    parser.add_argument("--size", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.route == "higher":
        record = higher_landau(arguments.size)
    elif arguments.route == "lattice":
        record = lattice(arguments.size)
    elif arguments.route == "sphere":
        record = sphere(arguments.size)
    else:
        record = vmc(arguments.size)
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
