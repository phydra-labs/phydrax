#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Optional restricted coupled-cluster execution over native MO integral stores."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..electronic_structure.correlation import (
    AbstractCoupledClusterProvider,
    CoupledClusterCheckpoint,
    CoupledClusterPlan,
    CoupledClusterResult,
    MolecularOrbitalIntegralStore,
)


def is_pyscf_correlation_available() -> bool:
    return importlib.util.find_spec("pyscf") is not None


def require_pyscf_correlation():
    if not is_pyscf_correlation_available():
        raise ImportError(
            "Coupled-cluster execution requires optional dependency 'pyscf'."
        )
    return (
        importlib.import_module("pyscf.gto"),
        importlib.import_module("pyscf.scf"),
        importlib.import_module("pyscf.ao2mo"),
        importlib.import_module("pyscf.cc"),
        importlib.import_module("pyscf.cc.ccsd_lambda"),
    )


def _mean_field_from_store(store: MolecularOrbitalIntegralStore):
    gto, scf, ao2mo, _, _ = require_pyscf_correlation()
    orbital_count = store.partition.orbital_count
    occupied = tuple(
        sorted(
            set(store.partition.frozen_occupied)
            | set(store.partition.correlated_occupied)
        )
    )
    if occupied != tuple(range(len(occupied))):
        raise ValueError(
            "Restricted coupled-cluster occupied orbitals must be canonical leading columns."
        )
    atom = [("H", (0.0, 0.0, 4.0 * index)) for index in range(orbital_count)]
    molecule = gto.M(
        atom=atom,
        basis={"H": [[0, [1.0, 1.0]]]},
        unit="Bohr",
        charge=orbital_count - 2 * len(occupied),
        spin=0,
        verbose=0,
    )
    if molecule.nao_nr() != orbital_count:
        raise RuntimeError(
            "Synthetic coupled-cluster molecule did not realize the MO dimension."
        )
    mean_field = scf.RHF(molecule)
    mean_field.mo_coeff = np.eye(orbital_count)
    mean_field.mo_occ = np.asarray(
        [2.0 if index in occupied else 0.0 for index in range(orbital_count)]
    )
    mean_field.mo_energy = np.asarray(store.orbital_energies)
    mean_field.e_tot = float(store.reference_energy)
    mean_field.converged = True
    mean_field._eri = ao2mo.restore(8, np.asarray(store.dense_two_body()), orbital_count)
    mean_field.get_hcore = lambda *args: np.asarray(store.one_body)
    mean_field.get_ovlp = lambda *args: np.eye(orbital_count)
    return mean_field


class PySCFCoupledClusterProvider(AbstractCoupledClusterProvider):
    provider_version: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)

    def __init__(self):
        version = (
            importlib.metadata.version("pyscf")
            if is_pyscf_correlation_available()
            else "unavailable"
        )
        self.provider_version = version
        self.provider_id = canonical_fingerprint(
            {"kind": "pyscf-coupled-cluster-provider", "version": version}
        )

    def evaluate(
        self,
        plan: CoupledClusterPlan,
        integrals: MolecularOrbitalIntegralStore,
        /,
        *,
        checkpoint: CoupledClusterCheckpoint | None = None,
    ) -> CoupledClusterResult:
        if not is_pyscf_correlation_available():
            raise ImportError(
                "Coupled-cluster execution requires optional dependency 'pyscf'."
            )
        if not isinstance(plan, CoupledClusterPlan) or not isinstance(
            integrals, MolecularOrbitalIntegralStore
        ):
            raise TypeError("PySCF coupled cluster requires typed plan and MO integrals.")
        if checkpoint is not None and (
            checkpoint.provider_id != self.provider_id
            or checkpoint.plan_id != plan.plan_id
            or checkpoint.store_id != integrals.store_id
        ):
            raise ValueError("Coupled-cluster checkpoint belongs to another calculation.")
        _, _, _, cc, ccsd_lambda = require_pyscf_correlation()
        mean_field = _mean_field_from_store(integrals)
        frozen = tuple(
            sorted(
                set(integrals.partition.frozen_occupied)
                | set(integrals.partition.frozen_virtual)
            )
        )
        calculation = cc.CCSD(
            mean_field,
            frozen=None if not frozen else frozen,
        )
        calculation.conv_tol = plan.convergence_tolerance
        calculation.conv_tol_normt = plan.convergence_tolerance
        calculation.max_cycle = plan.maximum_iterations
        calculation.verbose = 0
        eris = calculation.ao2mo(calculation.mo_coeff)
        correlation, singles, doubles = calculation.kernel(
            t1=None if checkpoint is None else np.asarray(checkpoint.singles_amplitudes),
            t2=None if checkpoint is None else np.asarray(checkpoint.doubles_amplitudes),
            eris=eris,
        )
        updated_singles, updated_doubles = calculation.update_amps(singles, doubles, eris)
        amplitude_residual = max(
            float(np.max(np.abs(updated_singles - singles), initial=0.0)),
            float(np.max(np.abs(updated_doubles - doubles), initial=0.0)),
        )
        lambda_singles = lambda_doubles = None
        lambda_residual = 0.0
        lambda_converged = True
        if plan.solve_lambda:
            lambda_singles, lambda_doubles = calculation.solve_lambda(
                t1=singles,
                t2=doubles,
                l1=None
                if checkpoint is None or checkpoint.lambda_singles is None
                else np.asarray(checkpoint.lambda_singles),
                l2=None
                if checkpoint is None or checkpoint.lambda_doubles is None
                else np.asarray(checkpoint.lambda_doubles),
                eris=eris,
            )
            updated_lambda_singles, updated_lambda_doubles = ccsd_lambda.update_lambda(
                calculation,
                singles,
                doubles,
                lambda_singles,
                lambda_doubles,
                eris,
            )
            lambda_residual = max(
                float(
                    np.max(np.abs(updated_lambda_singles - lambda_singles), initial=0.0)
                ),
                float(
                    np.max(np.abs(updated_lambda_doubles - lambda_doubles), initial=0.0)
                ),
            )
            lambda_converged = bool(calculation.converged_lambda)
        triples = calculation.ccsd_t(eris=eris) if plan.level == "ccsd(t)" else 0.0
        successful = (
            bool(calculation.converged)
            and lambda_converged
            and np.isfinite(correlation)
            and np.isfinite(triples)
            and amplitude_residual <= plan.convergence_tolerance
            and lambda_residual <= plan.convergence_tolerance
        )
        return CoupledClusterResult(
            correlation,
            triples,
            float(integrals.reference_energy) + correlation + triples,
            singles,
            doubles,
            amplitude_residual,
            lambda_residual,
            int(calculation.cycles)
            + (0 if checkpoint is None else checkpoint.completed_iterations),
            successful,
            self.provider_id,
            plan.plan_id,
            integrals.store_id,
            lambda_singles=lambda_singles,
            lambda_doubles=lambda_doubles,
        )


__all__ = [
    "PySCFCoupledClusterProvider",
    "is_pyscf_correlation_available",
    "require_pyscf_correlation",
]
