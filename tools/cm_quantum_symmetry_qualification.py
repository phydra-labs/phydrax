#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-group orbit-sector candidate evidence on a periodic spin ring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.operators.quantum.lattice import (
    CharacterSectorPlan,
    FiniteGroupActionPlan,
    FixedSpinProjectionBasis,
    LocalOperatorPlan,
    LocalSpacePlan,
    MonomialConfigurationGenerator,
    OrbitOperatorResourcePolicy,
    OrbitSectorResourcePolicy,
    prepare_finite_group_action,
    prepare_orbit_sector_basis,
    prepare_quantum_lattice,
    prepare_quantum_orbit_sector_operator,
    quantum_lattice_candidate_profiles,
    QuantumLatticeResourcePolicy,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
    QuantumSectorOperator,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)


jax.config.update("jax_enable_x64", True)


def _matrix(operator):
    identity = jnp.eye(operator.source.size, dtype=jnp.complex128)
    return jnp.stack(tuple(operator.mv(column) for column in identity), axis=1)


def _ring():
    site_count = 3
    spaces = tuple(LocalSpacePlan.spin(f"s{index}", 1) for index in range(site_count))
    raising = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    lowering = raising.T
    terms = tuple(
        QuantumLatticeTerm(
            (
                LocalOperatorPlan(spaces[index], f"raise-{index}", raising, (2,)),
                LocalOperatorPlan(
                    spaces[(index + 1) % site_count],
                    f"lower-{(index + 1) % site_count}",
                    lowering,
                    (-2,),
                ),
            ),
            coefficient=1.0,
            add_adjoint=True,
            label=f"exchange-{index}",
        )
        for index in range(site_count)
    )
    prepared = prepare_quantum_lattice(
        QuantumLatticeSpecification(spaces, terms),
        QuantumLatticeResourcePolicy(
            maximum_terms=16,
            maximum_factors_per_term=2,
            maximum_branches_per_input=128,
            maximum_sector_dimension=32,
            maximum_workspace_bytes=100_000,
        ),
    )
    direct = FixedSpinProjectionBasis(
        tuple(space.site_id for space in spaces),
        (1,) * site_count,
        -1,
        resources=SectorBasisResourcePolicy(
            maximum_dimension=32, maximum_table_bytes=20_000
        ),
    )
    translation = MonomialConfigurationGenerator(
        "translation",
        direct.site_ids,
        direct.site_dimensions,
        (1, 2, 0),
        order=3,
    )
    return prepared, direct, translation


def run_qualification() -> dict[str, object]:
    prepared, direct, translation = _ring()
    direct_operator = QuantumSectorOperator(prepared, SectorChargeMap(direct, direct, 0))
    direct_matrix = _matrix(direct_operator)
    sectors = []
    reduced_eigenvalues = []
    maximum_projection_residual = 0.0
    maximum_invariance_residual = 0.0
    for momentum in range(3):
        character = np.exp(2.0j * np.pi * momentum / 3.0)
        action = prepare_finite_group_action(
            FiniteGroupActionPlan(
                direct,
                (translation,),
                OrbitSectorResourcePolicy(
                    maximum_group_order=6,
                    maximum_orbit_dimension=8,
                    maximum_table_bytes=100_000,
                ),
            ),
            CharacterSectorPlan(f"momentum-{momentum}", {"translation": character}),
        )
        basis = prepare_orbit_sector_basis(action)
        operator = prepare_quantum_orbit_sector_operator(
            prepared,
            basis,
            OrbitOperatorResourcePolicy(
                maximum_routes=64,
                maximum_workspace_bytes=100_000,
            ),
        )
        reduced = _matrix(operator)
        embedding = np.zeros((direct.dimension, basis.dimension), dtype=np.complex128)
        raw_to_orbit = np.asarray(basis.raw_to_orbit)
        coefficients = np.asarray(basis.embedding_coefficients)
        for raw_index, orbit_index in enumerate(raw_to_orbit):
            if orbit_index >= 0:
                embedding[raw_index, orbit_index] = coefficients[raw_index]
        reference = np.conj(embedding.T) @ np.asarray(direct_matrix) @ embedding
        residual = float(np.max(np.abs(np.asarray(reduced) - reference)))
        maximum_projection_residual = max(maximum_projection_residual, residual)
        maximum_invariance_residual = max(
            maximum_invariance_residual,
            float(operator.evidence.maximum_invariance_residual),
        )
        values = np.linalg.eigvalsh(np.asarray(reduced))
        reduced_eigenvalues.extend(values.tolist())
        sectors.append(
            {
                "momentum": momentum,
                "character_real": float(np.real(character)),
                "character_imag": float(np.imag(character)),
                "basis_id": basis.basis_id,
                "operator_id": operator.operator_id,
                "dimension": basis.dimension,
                "orbit_sizes": np.asarray(basis.orbit_sizes).tolist(),
                "embedding_real": np.real(embedding).tolist(),
                "embedding_imag": np.imag(embedding).tolist(),
                "matrix_real": np.real(np.asarray(reduced)).tolist(),
                "matrix_imag": np.imag(np.asarray(reduced)).tolist(),
                "projection_residual": residual,
                "invariance_residuals": np.asarray(
                    operator.evidence.invariance_residuals
                ).tolist(),
                "hermiticity_residual": float(operator.evidence.hermiticity_residual),
                "route_count": operator.evidence.route_count,
            }
        )
    direct_eigenvalues = np.linalg.eigvalsh(np.asarray(direct_matrix))
    spectrum_residual = float(
        np.max(
            np.abs(np.sort(np.asarray(reduced_eigenvalues)) - np.sort(direct_eigenvalues))
        )
    )
    successful = bool(
        maximum_projection_residual <= 1e-12
        and maximum_invariance_residual <= 1e-12
        and spectrum_residual <= 1e-12
    )
    return {
        "kind": "quantum-orbit-sector-candidate-qualification",
        "profiles": [
            profile.to_record() for profile in quantum_lattice_candidate_profiles()
        ],
        "case": {
            "model": "three-site-periodic-spin-exchange",
            "direct_dimension": direct.dimension,
            "prepared_id": prepared.prepared_id,
            "direct_operator_id": direct_operator.operator_id,
        },
        "raw": {
            "direct_matrix_real": np.real(np.asarray(direct_matrix)).tolist(),
            "direct_matrix_imag": np.imag(np.asarray(direct_matrix)).tolist(),
            "direct_eigenvalues": direct_eigenvalues.tolist(),
            "sectors": sectors,
        },
        "criteria": {
            "maximum_projection_residual": maximum_projection_residual,
            "maximum_invariance_residual": maximum_invariance_residual,
            "spectrum_union_residual": spectrum_residual,
        },
        "successful": successful,
        "claim": "finite-character-sector-candidate-only-not-thermodynamic-evidence",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(run_qualification(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
