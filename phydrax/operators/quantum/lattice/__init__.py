#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical finite quantum-lattice compilation and fixed-sector execution."""

from ._compile import (
    certify_charge_map,
    ChargeMapCertification,
    CompiledMonomial,
    plan_quantum_lattice,
    prepare_quantum_lattice,
    PreparedQuantumLattice,
    QuantumLatticeCompilerPlan,
    QuantumLatticeResourcePolicy,
    refresh_quantum_lattice,
)
from ._lifecycle import (
    QuantumLatticeArchiveArtifact,
    QuantumLatticeArtifactKind,
    read_quantum_lattice_artifact_archive,
    write_quantum_lattice_artifact_archive,
)
from ._model import (
    LocalOperatorPlan,
    LocalSpacePlan,
    LocalStatistics,
    QuantumLatticeSpecification,
    QuantumLatticeTerm,
)
from ._operator import (
    apply_compiled_to_coordinate,
    apply_monomial_to_coordinate,
    QuantumSectorOperator,
)
from ._periodic import (
    FermionInteractionPlan,
    FermionInteractionTerm,
    periodic_finite_to_fermion_lattice,
)
from ._qualification import (
    quantum_lattice_candidate_profiles,
    quantum_lattice_candidate_support_tuples,
)
from ._sector import (
    AbstractSectorBasis,
    FixedBosonNumberBasis,
    FixedCardinalityFermionBasis,
    FixedSpinProjectionBasis,
    SectorBasisResourcePolicy,
    SectorChargeMap,
)
from ._stochastic import (
    assess_sign_free_stochastic_candidate,
    SignFreeStochasticCandidateEvidence,
    SignFreeStochasticCandidatePlan,
)
from ._vmc import lower_quantum_lattice_to_vmc, QuantumLatticeVMCOperator


__all__ = [
    "AbstractSectorBasis",
    "ChargeMapCertification",
    "CompiledMonomial",
    "FermionInteractionPlan",
    "FermionInteractionTerm",
    "FixedBosonNumberBasis",
    "FixedCardinalityFermionBasis",
    "FixedSpinProjectionBasis",
    "LocalOperatorPlan",
    "LocalSpacePlan",
    "LocalStatistics",
    "QuantumLatticeArchiveArtifact",
    "QuantumLatticeArtifactKind",
    "PreparedQuantumLattice",
    "QuantumLatticeCompilerPlan",
    "QuantumLatticeResourcePolicy",
    "QuantumLatticeSpecification",
    "QuantumLatticeTerm",
    "QuantumLatticeVMCOperator",
    "QuantumSectorOperator",
    "SectorBasisResourcePolicy",
    "SectorChargeMap",
    "SignFreeStochasticCandidateEvidence",
    "SignFreeStochasticCandidatePlan",
    "apply_compiled_to_coordinate",
    "apply_monomial_to_coordinate",
    "assess_sign_free_stochastic_candidate",
    "certify_charge_map",
    "lower_quantum_lattice_to_vmc",
    "read_quantum_lattice_artifact_archive",
    "write_quantum_lattice_artifact_archive",
    "periodic_finite_to_fermion_lattice",
    "quantum_lattice_candidate_profiles",
    "quantum_lattice_candidate_support_tuples",
    "plan_quantum_lattice",
    "prepare_quantum_lattice",
    "refresh_quantum_lattice",
]
