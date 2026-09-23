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
from ._irrep_sector import (
    CovariantOperatorEvidence,
    FiniteGroupIrrepEvidence,
    FiniteGroupIrrepPlan,
    IrrepMultiplicityEvidence,
    prepare_covariant_irrep_operator,
    prepare_finite_group_irrep,
    prepare_finite_group_irrep_basis,
    PreparedFiniteGroupIrrep,
    PreparedFiniteGroupIrrepBasis,
    ReducedCovariantOperator,
    run_symmetry_resolved_finite_size_study,
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
from ._orbit_operator import (
    OrbitOperatorEvidence,
    OrbitOperatorResourcePolicy,
    prepare_quantum_orbit_sector_operator,
    QuantumOrbitSectorOperator,
)
from ._orbit_sector import (
    CharacterSectorPlan,
    FiniteGroupActionPlan,
    MonomialConfigurationGenerator,
    OrbitSectorResourcePolicy,
    prepare_finite_group_action,
    prepare_orbit_sector_basis,
    PreparedFiniteGroupAction,
    PreparedOrbitSectorBasis,
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
from ._reduced_irrep import (
    prepare_su2_sector_basis,
    PreparedSU2SectorBasis,
    project_product_operator_to_su2_sector,
    SU2CouplingTreePlan,
    SU2ProjectedOperator,
    SU2ProjectedOperatorEvidence,
    SU2SectorEvidence,
    SU2SectorResourcePolicy,
)
from ._sector import (
    AbstractSectorBasis,
    FixedAbelianChargeBasis,
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
    "CharacterSectorPlan",
    "ChargeMapCertification",
    "CovariantOperatorEvidence",
    "FiniteGroupIrrepEvidence",
    "FiniteGroupIrrepPlan",
    "IrrepMultiplicityEvidence",
    "CompiledMonomial",
    "FermionInteractionPlan",
    "FermionInteractionTerm",
    "FixedBosonNumberBasis",
    "FixedCardinalityFermionBasis",
    "FixedAbelianChargeBasis",
    "FixedSpinProjectionBasis",
    "FiniteGroupActionPlan",
    "LocalOperatorPlan",
    "LocalSpacePlan",
    "LocalStatistics",
    "MonomialConfigurationGenerator",
    "OrbitOperatorEvidence",
    "OrbitOperatorResourcePolicy",
    "OrbitSectorResourcePolicy",
    "QuantumLatticeArchiveArtifact",
    "QuantumLatticeArtifactKind",
    "PreparedQuantumLattice",
    "PreparedFiniteGroupAction",
    "PreparedOrbitSectorBasis",
    "PreparedFiniteGroupIrrep",
    "PreparedFiniteGroupIrrepBasis",
    "PreparedSU2SectorBasis",
    "QuantumLatticeCompilerPlan",
    "QuantumLatticeResourcePolicy",
    "QuantumLatticeSpecification",
    "QuantumLatticeTerm",
    "QuantumLatticeVMCOperator",
    "QuantumSectorOperator",
    "QuantumOrbitSectorOperator",
    "ReducedCovariantOperator",
    "SectorBasisResourcePolicy",
    "SectorChargeMap",
    "SignFreeStochasticCandidateEvidence",
    "SignFreeStochasticCandidatePlan",
    "SU2CouplingTreePlan",
    "SU2ProjectedOperator",
    "SU2ProjectedOperatorEvidence",
    "SU2SectorEvidence",
    "SU2SectorResourcePolicy",
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
    "prepare_covariant_irrep_operator",
    "prepare_finite_group_irrep",
    "prepare_finite_group_irrep_basis",
    "prepare_finite_group_action",
    "prepare_orbit_sector_basis",
    "prepare_quantum_orbit_sector_operator",
    "prepare_su2_sector_basis",
    "project_product_operator_to_su2_sector",
    "run_symmetry_resolved_finite_size_study",
    "refresh_quantum_lattice",
]
