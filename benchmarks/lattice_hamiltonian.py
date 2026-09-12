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


def _z2_case(repeats: int) -> dict:
    topology = phx.discretization.polygonal_cell_complex(
        jnp.asarray([[0, 1, 2]]), None, 3
    )
    model = phx.applications.lattice_field.Z2GaugeModel(
        topology,
        electric_coupling=0.4,
        magnetic_coupling=1.2,
    )
    sector, sector_timing = measure_repeated(
        lambda: phx.applications.lattice_field.prepare_z2_gauss_sector(model),
        warmup=0,
        repeats=repeats,
    )
    hamiltonian = phx.applications.lattice_field.z2_gauge_hamiltonian(model)
    dense, dense_timing = measure_repeated(
        lambda: phx.solver.materialize_local_hamiltonian(hamiltonian),
        warmup=1,
        repeats=repeats,
    )
    maximum_commutator = jnp.asarray(0.0)
    for term in phx.applications.lattice_field.z2_gauss_terms(model):
        generator = phx.solver.materialize_local_hamiltonian(
            phx.solver.LocalHamiltonian(model.layout, (term,))
        )
        maximum_commutator = jnp.maximum(
            maximum_commutator,
            jnp.max(jnp.abs(dense @ generator - generator @ dense)),
        )
    return {
        "vertices": model.num_vertices,
        "edges": model.num_edges,
        "faces": model.num_faces,
        "physical_dimension": model.layout.dimension,
        "gauss_sector_dimension": sector.subspace.logical_dimension,
        "constraint_rank": sector.constraint_rank,
        "maximum_commutator_residual": float(maximum_commutator),
        "dense_bytes": logical_array_bytes(dense),
        "sector_preparation": sector_timing.to_milliseconds_dict(),
        "dense_materialization": dense_timing.to_milliseconds_dict(),
    }


def _schwinger_case(sites: int, repeats: int) -> dict:
    model = phx.applications.lattice_field.SchwingerChainModel(
        sites,
        lattice_spacing=0.4,
        mass=0.3,
        gauge_coupling=0.8,
        left_boundary_flux=0.1,
        external_flux=jnp.linspace(-0.05, 0.08, sites - 1),
    )
    mpo, mpo_timing = measure_repeated(
        lambda: phx.applications.lattice_field.schwinger_mpo(model),
        warmup=1,
        repeats=repeats,
    )
    dense_residual = None
    if sites <= 8:
        local = phx.applications.lattice_field.schwinger_local_hamiltonian(model)
        local_dense = phx.solver.materialize_local_hamiltonian(local)
        mpo_dense = mpo.operator.to_dense(maximum_elements=1 << 20)
        dense_residual = float(jnp.max(jnp.abs(local_dense - mpo_dense)))
    staggered = jnp.asarray([1 if site % 2 == 0 else 0 for site in range(sites)])
    flux = phx.applications.lattice_field.reconstruct_schwinger_flux(model, staggered)
    gauss_residual = phx.applications.lattice_field.schwinger_gauss_residual(
        model,
        staggered,
        flux,
    )

    local_states = jnp.asarray(
        [[0.0, 1.0] if site % 2 == 0 else [1.0, 0.0] for site in range(sites)],
        dtype=jnp.complex128,
    )
    state = phx.tensor_network.product_mps(local_states)
    dmrg = phx.solver.prepare_finite_dmrg(
        phx.solver.FiniteDMRGProblem(state, mpo.operator),
        phx.solver.FiniteDMRGPolicy(
            maximum_bond_dimension=min(8, 1 << (sites // 2)),
            maximum_sweeps=2,
            eigen_policy=phx.linalg.eigen.EigenSolvePolicy(
                phx.linalg.eigen.DenseEigh(),
                count=1,
                which="smallest-algebraic",
            ),
        ),
    )
    dmrg_result, dmrg_timing = measure_repeated(
        lambda: phx.solver.solve_finite_dmrg(dmrg),
        warmup=1,
        repeats=repeats,
    )
    tdvp_result, tdvp_timing = measure_repeated(
        lambda: phx.solver.solve_finite_tdvp(
            phx.solver.FiniteTDVPProblem(state, mpo.operator),
            phx.solver.FiniteTDVPPolicy(
                "real-time",
                step_size=0.01,
                steps=1,
                norm_tolerance=1e-5,
            ),
        ),
        warmup=1,
        repeats=repeats,
    )
    return {
        "sites": sites,
        "local_term_count": None if sites > 8 else len(local.terms),
        "prefix_maximum_bond_dimension": mpo.electric_evidence.maximum_bond_dimension,
        "full_mpo_bond_dimensions": list(mpo.operator.bond_dimensions),
        "dense_crosscheck_residual": dense_residual,
        "gauss_residual": float(gauss_residual),
        "mpo_construction": mpo_timing.to_milliseconds_dict(),
        "dmrg": {
            "execution": dmrg_timing.to_milliseconds_dict(),
            "energy": float(dmrg_result.energy),
            "status": int(dmrg_result.diagnostics.status),
        },
        "tdvp": {
            "execution": tdvp_timing.to_milliseconds_dict(),
            "status": int(tdvp_result.diagnostics.status),
            "norm_residual": float(
                jnp.abs(tdvp_result.diagnostics.norm_history[-1] - 1.0)
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", nargs="+", type=int, default=[4, 6])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if any(site < 2 or site > 8 for site in arguments.sites):
        raise ValueError("sites must lie between two and eight for this qualification.")
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "z2_triangle": _z2_case(arguments.repeats),
        "schwinger": [
            _schwinger_case(sites, arguments.repeats) for sites in arguments.sites
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
