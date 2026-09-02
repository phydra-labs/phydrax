#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
from _runtime import capture_environment, logical_array_bytes, measure_repeated

import phydrax as phx


def _problem(sites: int):
    identity = jnp.eye(2, dtype=jnp.complex128)
    pauli_x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    pauli_z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    terms = []
    for bond in range(sites - 1):
        local = [identity] * sites
        local[bond] = -pauli_z
        local[bond + 1] = pauli_z
        terms.append(phx.tensor_network.product_mpo(jnp.stack(local)))
    for site in range(sites):
        local = [identity] * sites
        local[site] = -0.5 * pauli_x
        terms.append(phx.tensor_network.product_mpo(jnp.stack(local)))
    hamiltonian = terms[0]
    for term in terms[1:]:
        hamiltonian = phx.tensor_network.add_mpo(hamiltonian, term)
    state = phx.tensor_network.product_mps(
        jnp.tile(jnp.asarray([[1.0, 0.0]], dtype=jnp.complex128), (sites, 1))
    )
    return state, hamiltonian


def _case(sites: int, bond: int, repeats: int):
    state, hamiltonian = _problem(sites)
    dmrg_problem = phx.solver.DMRGProblem(state, hamiltonian)
    dmrg_policy = phx.solver.DMRGPolicy(
        maximum_bond_dimension=bond,
        maximum_sweeps=2,
        eigen_policy=phx.linalg.eigen.EigenSolvePolicy(
            phx.linalg.eigen.DenseEigh(), count=1, which="smallest-algebraic"
        ),
    )
    prepared = phx.solver.prepare_dmrg(dmrg_problem, dmrg_policy)
    dmrg, dmrg_timing = measure_repeated(
        lambda: phx.solver.solve_dmrg(prepared), warmup=1, repeats=repeats
    )
    tdvp_policy = phx.solver.MatrixProductTDVPPolicy(
        "real-time", step_size=0.01, steps=1, norm_tolerance=1e-5
    )
    tdvp, tdvp_timing = measure_repeated(
        lambda: phx.solver.solve_matrix_product_tdvp(
            phx.solver.MatrixProductTDVPProblem(state, hamiltonian), tdvp_policy
        ),
        warmup=1,
        repeats=repeats,
    )
    return {
        "sites": sites,
        "maximum_bond_dimension": bond,
        "logical_input_bytes": logical_array_bytes((state, hamiltonian)),
        "dmrg": {
            "execution": dmrg_timing.to_milliseconds_dict(),
            "energy": float(dmrg.energy),
            "status": int(dmrg.diagnostics.status),
            "maximum_local_residual": float(
                jnp.nanmax(dmrg.diagnostics.local_residual_history)
            ),
        },
        "tdvp": {
            "execution": tdvp_timing.to_milliseconds_dict(),
            "status": int(tdvp.diagnostics.status),
            "norm_residual": float(jnp.abs(tdvp.diagnostics.norm_history[1] - 1.0)),
            "maximum_local_residual": float(
                jnp.nanmax(tdvp.diagnostics.local_residual_history)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--bond-dimensions", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        any(value < 2 for value in arguments.sites)
        or any(value < 1 for value in arguments.bond_dimensions)
        or arguments.repeats < 1
    ):
        raise ValueError("Benchmark sizes and repeats are outside supported bounds.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            _case(sites, bond, arguments.repeats)
            for sites in arguments.sites
            for bond in arguments.bond_dimensions
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
