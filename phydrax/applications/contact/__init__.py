#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._barrier import (
    clamped_log_barrier,
    clamped_log_barrier_first_derivative,
    clamped_log_barrier_second_derivative,
    physical_barrier_scale,
    physical_clamped_log_barrier,
)
from ._friction import (
    ContactFrictionEvaluation,
    ContactFrictionState,
    LaggedCoulombFrictionPlan,
    PreparedLaggedCoulombFriction,
    smooth_coulomb_potential,
)
from ._potential import (
    ContactPotentialEvaluation,
    ConvergentContactPotentialPlan,
    PreparedConvergentContactPotential,
)
from ._sensitivity import (
    contact_dynamics_solution_jvp,
    contact_dynamics_solution_vjp,
    contact_equilibrium_solution_jvp,
    contact_equilibrium_solution_vjp,
    ContactDynamicsSensitivityArguments,
    ContactSensitivityArguments,
    ContactSensitivityResult,
)
from ._solver import (
    ContactDynamicsState,
    ContactEnergyLedger,
    ContactRejectionReason,
    ContactSolveDiagnostics,
    ContactSolvePolicy,
    FiniteElementContactDynamicsPlan,
    FiniteElementContactEquilibriumPlan,
    FiniteElementContactEquilibriumResult,
    FiniteElementContactResult,
    prepare_finite_element_contact_dynamics,
    prepare_finite_element_contact_equilibrium,
    prepare_finite_element_contact_step,
    PreparedFiniteElementContactStep,
    solve_finite_element_contact_equilibrium,
    solve_finite_element_contact_step,
)


__all__ = [
    "ContactDynamicsSensitivityArguments",
    "ContactDynamicsState",
    "ContactEnergyLedger",
    "ContactFrictionEvaluation",
    "ContactFrictionState",
    "ContactPotentialEvaluation",
    "ContactRejectionReason",
    "ContactSensitivityArguments",
    "ContactSensitivityResult",
    "ContactSolveDiagnostics",
    "ContactSolvePolicy",
    "ConvergentContactPotentialPlan",
    "FiniteElementContactDynamicsPlan",
    "FiniteElementContactEquilibriumPlan",
    "FiniteElementContactEquilibriumResult",
    "FiniteElementContactResult",
    "LaggedCoulombFrictionPlan",
    "PreparedConvergentContactPotential",
    "PreparedFiniteElementContactStep",
    "PreparedLaggedCoulombFriction",
    "clamped_log_barrier",
    "clamped_log_barrier_first_derivative",
    "clamped_log_barrier_second_derivative",
    "contact_dynamics_solution_jvp",
    "contact_dynamics_solution_vjp",
    "contact_equilibrium_solution_jvp",
    "contact_equilibrium_solution_vjp",
    "physical_barrier_scale",
    "physical_clamped_log_barrier",
    "prepare_finite_element_contact_dynamics",
    "prepare_finite_element_contact_equilibrium",
    "prepare_finite_element_contact_step",
    "smooth_coulomb_potential",
    "solve_finite_element_contact_equilibrium",
    "solve_finite_element_contact_step",
]
