#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite SU(2) coupling and two-particle fuzzy-sphere qualification record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phydrax.applications import fuzzy_space
from phydrax.operators.quantum.lattice import (
    prepare_su2_sector_basis,
    SU2CouplingTreePlan,
    SU2SectorResourcePolicy,
)


def _resources():
    return SU2SectorResourcePolicy(
        maximum_product_dimension=128,
        maximum_sector_dimension=128,
        maximum_matrix_elements=16_384,
    )


def run_qualification() -> dict[str, object]:
    coupled = prepare_su2_sector_basis(
        SU2CouplingTreePlan(
            ("a", "b", "c"),
            (1, 1, 1),
            1,
            _resources(),
        )
    )
    fermion = fuzzy_space.prepare_fuzzy_sphere_two_particle(
        fuzzy_space.FuzzySphereTwoParticlePlan(
            2,
            "fermion",
            {2: 1.25},
            _resources(),
        )
    )
    boson = fuzzy_space.prepare_fuzzy_sphere_two_particle(
        fuzzy_space.FuzzySphereTwoParticlePlan(
            2,
            "boson",
            {0: 0.5, 4: 2.0},
            _resources(),
        )
    )
    fermion_spectrum = fuzzy_space.fuzzy_sphere_spectrum(fermion)
    boson_spectrum = fuzzy_space.fuzzy_sphere_spectrum(boson)
    successful = bool(
        coupled.evidence.accepted
        and fermion.evidence.accepted
        and boson.evidence.accepted
    )
    return {
        "kind": "fuzzy-space-candidate-qualification",
        "profiles": [
            profile.to_record()
            for profile in fuzzy_space.fuzzy_space_candidate_profiles()
        ],
        "case": {
            "coupled_basis_id": coupled.prepared_id,
            "fermion_model_id": fermion.prepared_id,
            "boson_model_id": boson.prepared_id,
            "twice_monopole_flux": 2,
        },
        "raw": {
            "coupling_transform": np.asarray(coupled.transform).tolist(),
            "fermion_hamiltonian": np.asarray(fermion.physical_hamiltonian).tolist(),
            "boson_hamiltonian": np.asarray(boson.physical_hamiltonian).tolist(),
            "fermion_energies": np.asarray(fermion_spectrum.energies).tolist(),
            "fermion_total_twice_spins": np.asarray(
                fermion_spectrum.total_twice_spins
            ).tolist(),
            "boson_energies": np.asarray(boson_spectrum.energies).tolist(),
            "boson_total_twice_spins": np.asarray(
                boson_spectrum.total_twice_spins
            ).tolist(),
        },
        "criteria": {
            "coupling_orthonormality": float(coupled.evidence.orthonormality_residual),
            "coupling_projector_idempotence": float(
                coupled.evidence.projector_idempotence_residual
            ),
            "fermion_exchange_complete": bool(fermion.evidence.exchange_sector_complete),
            "fermion_rotational_commutator": float(
                fermion.evidence.rotational_commutator_residual
            ),
            "boson_exchange_complete": bool(boson.evidence.exchange_sector_complete),
            "boson_rotational_commutator": float(
                boson.evidence.rotational_commutator_residual
            ),
        },
        "successful": successful,
        "claim": "finite-two-particle-fuzzy-sphere-reference-no-continuum-cft-identification",
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
