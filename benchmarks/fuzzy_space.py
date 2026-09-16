#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json

from benchmarks._runtime import capture_environment, logical_array_bytes, measure_host
from phydrax.applications import fuzzy_space
from phydrax.operators.quantum.lattice import SU2SectorResourcePolicy
from phydrax.tensor_network import su2_fusion


def benchmark_case(twice_flux: int, statistics: str):
    resources = SU2SectorResourcePolicy(
        maximum_product_dimension=10_000,
        maximum_sector_dimension=10_000,
        maximum_matrix_elements=100_000_000,
    )
    allowed = tuple(
        total
        for total in su2_fusion(twice_flux, twice_flux)
        if (((2 * twice_flux - total) // 2) % 2 == 0) == (statistics == "boson")
    )
    plan = fuzzy_space.FuzzySphereTwoParticlePlan(
        twice_flux,
        statistics,
        {spin: float(index + 1) for index, spin in enumerate(allowed)},
        resources,
    )
    prepared, prepare_seconds = measure_host(
        lambda: fuzzy_space.prepare_fuzzy_sphere_two_particle(plan)
    )
    spectrum, spectrum_seconds = measure_host(
        lambda: fuzzy_space.fuzzy_sphere_spectrum(prepared)
    )
    return {
        "axes": {
            "twice_monopole_flux": twice_flux,
            "orbital_count": plan.orbital_count,
            "statistics": statistics,
            "allowed_pair_spins": list(allowed),
            "physical_dimension": int(prepared.physical_hamiltonian.shape[0]),
        },
        "ids": {"plan": plan.plan_id, "prepared": prepared.prepared_id},
        "host_seconds": {
            "prepare": prepare_seconds,
            "spectrum": spectrum_seconds,
        },
        "logical_bytes": {
            "prepared": logical_array_bytes(prepared),
            "spectrum": logical_array_bytes(spectrum),
        },
        "scientific_residuals": {
            "orthonormality": float(prepared.evidence.transform_orthonormality_residual),
            "projector_idempotence": float(
                prepared.evidence.physical_projector_idempotence_residual
            ),
            "rotational_commutator": float(
                prepared.evidence.rotational_commutator_residual
            ),
        },
        "successful": bool(prepared.evidence.accepted and spectrum.finite),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fluxes", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    if any(value < 1 for value in arguments.fluxes):
        raise ValueError("twice flux values must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": [
            benchmark_case(flux, statistics)
            for flux in arguments.fluxes
            for statistics in ("boson", "fermion")
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output:
        with open(arguments.output, "w", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    else:
        print(encoded)


if __name__ == "__main__":
    main()
