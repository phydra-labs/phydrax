"""Tiny executable smoke for magnetism/superconductivity candidate profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from phydrax.applications.magnetism import magnetism_support_tuples
from phydrax.applications.superconductivity import superconductivity_support_tuples
from phydrax.chemistry.periodic._magnetism import (
    prepare_spin_orbit_operator,
    SpinorBasisConvention,
    SpinOrbitCouplingPlan,
)
from phydrax.discretization import PeriodicCell
from phydrax.discretization._reciprocal import ReciprocalMeshPlan
from phydrax.operators.periodic import (
    PeriodicBlochGauge,
    PeriodicOrbitalBasisPlan,
)
from phydrax.operators.periodic._family import (
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
)
from phydrax.operators.quantum._fermionic_fock import FermionModeOrder
from phydrax.operators.quantum._magnetism import spin_matrices
from phydrax.operators.quantum._spin_wave import (
    evaluate_linear_spin_wave,
    LinearSpinWavePlan,
    prepare_linear_spin_wave,
    SpinWaveReferenceState,
)
from phydrax.operators.quantum._superconductivity import (
    evaluate_bdg_spectrum,
    FermionicBdGPlan,
    FermionicPairingPlan,
    NambuConvention,
    prepare_fermionic_bdg,
)
from phydrax.sparse import EdgeRelation
from phydrax.units import ANGSTROM


def _constant_family(matrix, *, hermitian):
    value = np.asarray(matrix, dtype="complex128")
    count = value.shape[0]
    target = np.repeat(np.arange(count), count)
    source = np.tile(np.arange(count), count)
    reverse = np.arange(count * count).reshape((count, count)).T.reshape((-1,))
    relation = EdgeRelation(source, target, source_size=count, target_size=count)
    plan = PeriodicTranslationFamilyPlan(
        relation,
        np.zeros((count * count, 1), dtype="int64"),
        reverse,
        hermitian=hermitian,
    )
    state = PeriodicTranslationFamilyState(
        plan, value[target, source].reshape((-1, 1, 1))
    )
    return prepare_periodic_translation_family(plan, state)


def _soc_residuals():
    basis = PeriodicOrbitalBasisPlan(
        PeriodicCell(np.eye(3)),
        ("px", "py", "pz"),
        np.zeros((3, 3)),
        ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )
    angular = np.asarray(
        [
            [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
            [[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]],
            [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        ]
    )
    result = prepare_spin_orbit_operator(
        SpinOrbitCouplingPlan(
            SpinorBasisConvention(basis),
            angular,
            source_id="analytic-p-shell",
            maximum_spinor_count=6,
        ),
        1.0,
    )
    spectrum = jnp.linalg.eigvalsh(result.onsite_matrix)
    return {
        "hermiticity_residual": float(result.hermiticity_residual),
        "angular_momentum_residual": float(result.angular_momentum_residual),
        "p_shell_eigenvalues": [float(value) for value in spectrum],
    }


def _lswt_residuals():
    reference = SpinWaveReferenceState([[0.0, 0.0, 1.0]], [1.0])
    mesh = ReciprocalMeshPlan.monkhorst_pack(PeriodicCell([[1.0]]), (1,))
    plan = LinearSpinWavePlan(
        reference,
        mesh,
        [0],
        [0],
        energy_unit="reduced-energy",
        maximum_site_count=1,
    )
    result = evaluate_linear_spin_wave(
        prepare_linear_spin_wave(
            plan,
            _constant_family([[1.5]], hermitian=True),
            _constant_family([[0.0]], hermitian=False),
            [[0.0, 0.0, 0.0]],
        )
    )
    return {
        "frequency": float(result.frequencies[0, 0]),
        "paraunitarity_residual": float(result.paraunitarity_residual),
        "frequency_pairing_residual": float(result.frequency_pairing_residual),
        "minimum_stability_eigenvalue": float(result.minimum_stability_eigenvalue),
    }


def _bdg_residuals():
    order = FermionModeOrder(("up", "down"))
    plan = FermionicBdGPlan(
        NambuConvention(order),
        FermionicPairingPlan(
            order,
            [0],
            [1.0],
            mesh_id="analytic-gamma",
            energy_unit="reduced-energy",
        ),
        maximum_mode_count=2,
    )
    result = evaluate_bdg_spectrum(
        prepare_fermionic_bdg(
            plan,
            [[[0.1, 0.0], [0.0, 0.1]]],
            [[[0.0, 0.5], [-0.5, 0.0]]],
            chemical_potential=0.0,
            normal_family_id="analytic-normal",
            pairing_family_id="analytic-singlet",
        )
    )
    return {
        "particle_hole_residual": float(result.particle_hole_residual),
        "spectral_pairing_residual": float(result.spectral_pairing_residual),
        "minimum_direct_gap": float(result.minimum_direct_gap),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sx, sy, sz, _, _ = spin_matrices(1)
    report = {
        "maturity": "candidate",
        "capability_ids": tuple(
            support.capability
            for support in (
                *magnetism_support_tuples(),
                *superconductivity_support_tuples(),
            )
        ),
        "spin_half_su2_residual": float(jnp.max(jnp.abs(sx @ sy - sy @ sx - 1.0j * sz))),
        "soc": _soc_residuals(),
        "linear_spin_wave": _lswt_residuals(),
        "fermionic_bdg": _bdg_residuals(),
        "passed": False,
        "release_claim": False,
    }
    finite_values = np.asarray(
        (
            report["spin_half_su2_residual"],
            report["soc"]["hermiticity_residual"],
            report["soc"]["angular_momentum_residual"],
            report["linear_spin_wave"]["paraunitarity_residual"],
            report["linear_spin_wave"]["frequency_pairing_residual"],
            report["linear_spin_wave"]["minimum_stability_eigenvalue"],
            report["fermionic_bdg"]["particle_hole_residual"],
            report["fermionic_bdg"]["spectral_pairing_residual"],
            report["fermionic_bdg"]["minimum_direct_gap"],
        )
    )
    report["passed"] = bool(
        np.all(np.isfinite(finite_values))
        and report["spin_half_su2_residual"] <= 1.0e-12
        and report["soc"]["hermiticity_residual"] <= 1.0e-12
        and report["soc"]["angular_momentum_residual"] <= 1.0e-12
        and report["linear_spin_wave"]["paraunitarity_residual"] <= 1.0e-10
        and report["linear_spin_wave"]["frequency_pairing_residual"] <= 1.0e-10
        and report["linear_spin_wave"]["minimum_stability_eigenvalue"] >= 0.0
        and report["fermionic_bdg"]["particle_hole_residual"] <= 1.0e-12
        and report["fermionic_bdg"]["spectral_pairing_residual"] <= 1.0e-12
        and report["fermionic_bdg"]["minimum_direct_gap"] >= 0.0
    )
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
