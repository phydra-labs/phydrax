#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Governed optional molecular RHF/UHF/ROHF CCSD and CCSD(T) gradients."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
from collections.abc import Callable
from math import isfinite

import equinox as eqx
import numpy as np
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..electronic_structure.correlation import (
    AbstractMolecularCoupledClusterGradientProvider,
    CoupledClusterPlan,
    MolecularCoupledClusterGradientResult,
)


PySCFMoleculeBuilder = Callable[[np.ndarray], object]


def is_pyscf_molecular_correlation_available() -> bool:
    return importlib.util.find_spec("pyscf") is not None


def require_pyscf_molecular_correlation():
    if not is_pyscf_molecular_correlation_available():
        raise ImportError(
            "Molecular coupled-cluster gradients require optional dependency 'pyscf'."
        )
    return (
        importlib.import_module("pyscf.gto"),
        importlib.import_module("pyscf.scf"),
        importlib.import_module("pyscf.cc"),
    )


class PySCFMolecularCoupledClusterGradientProvider(
    AbstractMolecularCoupledClusterGradientProvider
):
    molecule_builder: PySCFMoleculeBuilder = eqx.field(static=True)
    coupled_cluster: CoupledClusterPlan
    reference: str = eqx.field(static=True)
    builder_definition_id: str = eqx.field(static=True)
    frozen_orbitals: tuple[int, ...] = eqx.field(static=True)
    scf_tolerance: float = eqx.field(static=True)
    scf_maximum_iterations: int = eqx.field(static=True)
    provider_version: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        molecule_builder: PySCFMoleculeBuilder,
        builder_definition_id: str,
        coupled_cluster: CoupledClusterPlan,
        reference: str,
        /,
        *,
        frozen_orbitals: tuple[int, ...] = (),
        scf_tolerance: float = 1.0e-11,
        scf_maximum_iterations: int = 128,
    ):
        if not callable(molecule_builder):
            raise TypeError("molecule_builder must be callable.")
        if not isinstance(coupled_cluster, CoupledClusterPlan):
            raise TypeError("coupled_cluster must be CoupledClusterPlan.")
        if not coupled_cluster.solve_lambda:
            raise ValueError("Analytic coupled-cluster gradients require a lambda solve.")
        definition = str(builder_definition_id).strip()
        reference_ = str(reference).strip().lower()
        frozen = tuple(sorted(int(value) for value in frozen_orbitals))
        tolerance = float(scf_tolerance)
        maximum = int(scf_maximum_iterations)
        if (
            not definition
            or reference_ not in ("rhf", "uhf", "rohf")
            or len(set(frozen)) != len(frozen)
            or any(value < 0 for value in frozen)
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or maximum <= 0
        ):
            raise ValueError(
                "Molecular CC provider reference, frozen space, or SCF policy is invalid."
            )
        version = (
            importlib.metadata.version("pyscf")
            if is_pyscf_molecular_correlation_available()
            else "unavailable"
        )
        self.molecule_builder = molecule_builder
        self.coupled_cluster = coupled_cluster
        self.reference = reference_
        self.builder_definition_id = definition
        self.frozen_orbitals = frozen
        self.scf_tolerance = tolerance
        self.scf_maximum_iterations = maximum
        self.provider_version = version
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pyscf-molecular-coupled-cluster-gradient-plan",
                "builder_definition": definition,
                "coupled_cluster": coupled_cluster.plan_id,
                "reference": reference_,
                "frozen_orbitals": list(frozen),
                "scf_tolerance": tolerance,
                "scf_maximum_iterations": maximum,
            }
        )
        self.provider_id = canonical_fingerprint(
            {
                "kind": "pyscf-molecular-coupled-cluster-gradient-provider",
                "version": version,
                "plan": self.plan_id,
            }
        )

    def evaluate(self, positions: ArrayLike, /) -> MolecularCoupledClusterGradientResult:
        coordinate = np.asarray(positions, dtype=np.float64)
        if (
            coordinate.ndim != 2
            or coordinate.shape[1] != 3
            or np.any(~np.isfinite(coordinate))
        ):
            raise ValueError(
                "Molecular CC positions must have shape (atom, 3) and be finite."
            )
        gto, scf, cc = require_pyscf_molecular_correlation()
        molecule = self.molecule_builder(coordinate)
        if not isinstance(molecule, gto.Mole):
            raise TypeError("molecule_builder must return a PySCF Mole.")
        built_positions = molecule.atom_coords(unit="Bohr")
        if built_positions.shape != coordinate.shape or not np.allclose(
            built_positions, coordinate, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                "PySCF molecule geometry differs from requested Bohr positions."
            )
        if self.reference == "rhf":
            mean_field = scf.RHF(molecule)
        elif self.reference == "uhf":
            mean_field = scf.UHF(molecule)
        else:
            mean_field = scf.ROHF(molecule)
        mean_field.conv_tol = self.scf_tolerance
        mean_field.max_cycle = self.scf_maximum_iterations
        mean_field.verbose = 0
        reference_energy = mean_field.kernel()
        calculation = cc.CCSD(
            mean_field,
            frozen=None if not self.frozen_orbitals else self.frozen_orbitals,
        )
        calculation.conv_tol = self.coupled_cluster.convergence_tolerance
        calculation.conv_tol_normt = self.coupled_cluster.convergence_tolerance
        calculation.max_cycle = self.coupled_cluster.maximum_iterations
        calculation.verbose = 0
        eris = calculation.ao2mo()
        correlation, singles, doubles = calculation.kernel(eris=eris)
        triples = (
            calculation.ccsd_t(eris=eris)
            if self.coupled_cluster.level == "ccsd(t)"
            else 0.0
        )
        if self.coupled_cluster.level == "ccsd(t)":
            if self.reference == "rhf":
                lambda_module = importlib.import_module("pyscf.cc.ccsd_t_lambda")
                gradient_module = importlib.import_module("pyscf.grad.ccsd_t")
            else:
                lambda_module = importlib.import_module("pyscf.cc.uccsd_t_lambda")
                gradient_module = importlib.import_module("pyscf.grad.uccsd_t")
            lambda_converged, lambda_singles, lambda_doubles = lambda_module.kernel(
                calculation,
                eris=eris,
                t1=singles,
                t2=doubles,
                max_cycle=self.coupled_cluster.maximum_iterations,
                tol=self.coupled_cluster.convergence_tolerance,
                verbose=0,
            )
            gradient = gradient_module.Gradients(calculation).kernel(
                singles,
                doubles,
                lambda_singles,
                lambda_doubles,
                eris=eris,
            )
        else:
            lambda_singles, lambda_doubles = calculation.solve_lambda(
                t1=singles,
                t2=doubles,
                eris=eris,
            )
            lambda_converged = bool(calculation.converged_lambda)
            gradient = calculation.nuc_grad_method().kernel(
                singles,
                doubles,
                lambda_singles,
                lambda_doubles,
                eris=eris,
            )
        return MolecularCoupledClusterGradientResult(
            reference_energy,
            correlation,
            triples,
            gradient,
            int(calculation.cycles),
            bool(mean_field.converged),
            bool(calculation.converged),
            bool(lambda_converged),
            self.reference,
            self.provider_id,
            self.plan_id,
        )


__all__ = [
    "PySCFMolecularCoupledClusterGradientProvider",
    "is_pyscf_molecular_correlation_available",
    "require_pyscf_molecular_correlation",
]
