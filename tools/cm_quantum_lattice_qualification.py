#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Tiny candidate-only quantum-lattice smoke and invariant evidence emitter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.operators.quantum import CARPolynomial, FermionicFockBasis, FermionModeOrder
from phydrax.operators.quantum.lattice import (
    FixedCardinalityFermionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    lower_quantum_lattice_to_vmc,
    prepare_quantum_lattice,
    quantum_lattice_candidate_profiles,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)


jax.config.update("jax_enable_x64", True)


def run_smoke() -> dict[str, object]:
    order = FermionModeOrder(("left", "middle", "right"))
    spaces = tuple(LocalSpacePlan.fermion(label, label) for label in order.labels)
    create = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    annihilate = create.T
    operators = {
        space.site_id: (
            LocalOperatorPlan(space, "create", create, (1,)),
            LocalOperatorPlan(space, "annihilate", annihilate, (-1,)),
        )
        for space in spaces
    }
    term = QuantumLatticeTerm(
        (operators["left"][0], operators["right"][1]),
        coefficient=1.0,
        add_adjoint=True,
        label="long-range-hopping",
    )
    prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(spaces, (term,), fermion_mode_order=order),
        QuantumLatticeResourcePolicy(
            maximum_terms=4,
            maximum_factors_per_term=2,
            maximum_branches_per_input=16,
            maximum_sector_dimension=16,
            maximum_workspace_bytes=20_000,
        ),
    )
    basis = FixedCardinalityFermionBasis(
        order,
        2,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=16, maximum_table_bytes=10_000
        ),
    )
    sector = QuantumSectorOperator(prepared, SectorChargeMap(basis, basis, 0))
    identity = jnp.eye(basis.dimension, dtype=jnp.complex128)
    matrix = jnp.stack(tuple(sector.mv(column) for column in identity), axis=1)
    full_basis = FermionicFockBasis(order)
    full = CARPolynomial(
        order,
        (
            (1.0, (("left", "create"), ("right", "annihilate"))),
            (1.0, (("right", "create"), ("left", "annihilate"))),
        ),
    ).dense_matrix(maximum_elements=512)
    indices = np.asarray(
        [
            full_basis.basis_index(np.asarray(basis.coordinate(index)).tolist())
            for index in range(basis.dimension)
        ]
    )
    reference = jnp.asarray(np.asarray(full)[np.ix_(indices, indices)])
    vmc = lower_quantum_lattice_to_vmc(sector)
    configuration = basis.coordinate(0)
    connections = vmc.connections(configuration)
    parity_residual = jnp.max(jnp.abs(matrix - reference))
    hermiticity_residual = jnp.max(jnp.abs(matrix - jnp.conj(matrix.T)))
    charge_residual = max(
        abs(int(jnp.sum(basis.coordinate(index))) - basis.particle_count)
        for index in range(basis.dimension)
    )
    successful = bool(
        parity_residual <= 1e-12
        and hermiticity_residual <= 1e-12
        and charge_residual == 0
    )
    return {
        "kind": "quantum-lattice-candidate-smoke",
        "profiles": [
            profile.to_record() for profile in quantum_lattice_candidate_profiles()
        ],
        "case": {
            "model": "three-mode-long-range-hopping",
            "sector_dimension": basis.dimension,
            "prepared_id": prepared.prepared_id,
            "operator_id": sector.operator_id,
            "fermion_mode_order_id": order.order_id,
            "matrix_free": not sector.capabilities.materialize,
            "branch_capacity": prepared.plan.total_branches_per_input,
            "action_workspace_bytes": sector.action_workspace_bytes,
        },
        "raw": {
            "sector_matrix_real": np.asarray(jnp.real(matrix)).tolist(),
            "sector_matrix_imag": np.asarray(jnp.imag(matrix)).tolist(),
            "vmc_connected_configurations": np.asarray(
                connections.configurations
            ).tolist(),
            "vmc_matrix_elements_real": np.asarray(
                jnp.real(connections.matrix_elements)
            ).tolist(),
            "vmc_matrix_elements_imag": np.asarray(
                jnp.imag(connections.matrix_elements)
            ).tolist(),
            "vmc_valid": np.asarray(connections.valid).tolist(),
        },
        "criteria": {
            "full_space_car_parity_residual": float(parity_residual),
            "hermiticity_residual": float(hermiticity_residual),
            "maximum_charge_residual": charge_residual,
        },
        "successful": successful,
        "claim": "candidate-smoke-only-not-release-evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    text = json.dumps(run_smoke(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
