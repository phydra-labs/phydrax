#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._stationary import (
    analytic_double_well_kink,
    DoubleWellKinkEvidence,
    DoubleWellKinkPlan,
    DoubleWellKinkResult,
    solve_double_well_kink,
)
from ._stationary_qualification import stationary_soliton_candidate_profiles
from ._topology import phase_field_topology_plan
from ._workflows import (
    AllenCahnAcceptedState,
    AllenCahnFEMPlan,
    BinaryPhaseFieldModel,
    CahnHilliardAcceptedState,
    CahnHilliardFEMPlan,
    PhaseFieldAcceptancePolicy,
    PhaseFieldProductionCase,
    PhaseFieldResolutionEvidence,
    PhaseFieldStepEvidence,
    PhaseFieldStepResult,
    PreparedAllenCahnFEM,
    PreparedCahnHilliardFEM,
)


__all__ = [
    "DoubleWellKinkEvidence",
    "DoubleWellKinkPlan",
    "DoubleWellKinkResult",
    "AllenCahnAcceptedState",
    "AllenCahnFEMPlan",
    "BinaryPhaseFieldModel",
    "CahnHilliardAcceptedState",
    "CahnHilliardFEMPlan",
    "PhaseFieldAcceptancePolicy",
    "PhaseFieldProductionCase",
    "PhaseFieldResolutionEvidence",
    "PhaseFieldStepEvidence",
    "PhaseFieldStepResult",
    "PreparedAllenCahnFEM",
    "PreparedCahnHilliardFEM",
    "analytic_double_well_kink",
    "solve_double_well_kink",
    "stationary_soliton_candidate_profiles",
    "phase_field_topology_plan",
]
