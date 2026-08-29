#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._execution import (
    ElementTensorOperator,
    FiniteElementPreconditionerData,
    PartialAssemblyOperator,
    TensorProductAction,
    TensorProductPartialAssemblyOperator,
)
from ._interpreter import evaluate_differential_operator, execute_local_action
from ._ir import (
    ActionKind,
    DifferentialOperator,
    FieldSlot,
    FieldSlotRole,
    FiniteElementActionIR,
    LocalActionIR,
    RegionIR,
    RegionKind,
)
from ._kernels import KernelBinding, KernelTable
from ._lowering import (
    compile_workset_program,
    kernel_table_from_form,
    lower_finite_element_form,
)
from ._materials import (
    FiniteElementAuxiliaryEvaluation,
    LocalImplicitDiagnostics,
    LocalImplicitMaterial,
)
from ._observations import (
    CoordinateObservation,
    finite_element_parameter_gradient,
    FiniteElementAdjointResult,
    FiniteElementLeastSquaresObjective,
    solve_finite_element_adjoint,
)
from ._operators import (
    average,
    curl,
    divergence,
    FacetJet,
    FieldJet,
    jump,
    normal_trace,
    symmetric_gradient,
    tangential_trace,
)
from ._proofs import (
    darcy_form,
    HDGPoissonSolution,
    linear_elasticity_form,
    maxwell_form,
    sipg_dirichlet,
    sipg_neumann,
    sipg_poisson_form,
    sipg_robin,
    SIPGBoundaryCondition,
    SIPGPenaltyPolicy,
    smoothed_elasticity_form,
    solve_hdg_poisson,
    stokes_form,
    upwind_advection_form,
)
from ._worksets import CompiledWorkset, WorksetProgram, WorksetSignature


__all__ = [
    "ElementTensorOperator",
    "PartialAssemblyOperator",
    "CoordinateObservation",
    "ActionKind",
    "CompiledWorkset",
    "HDGPoissonSolution",
    "DifferentialOperator",
    "FiniteElementPreconditionerData",
    "FiniteElementAuxiliaryEvaluation",
    "FiniteElementAdjointResult",
    "FiniteElementLeastSquaresObjective",
    "LocalImplicitDiagnostics",
    "KernelBinding",
    "KernelTable",
    "LocalImplicitMaterial",
    "FacetJet",
    "FieldJet",
    "FieldSlot",
    "FieldSlotRole",
    "LocalActionIR",
    "FiniteElementActionIR",
    "RegionIR",
    "RegionKind",
    "TensorProductAction",
    "TensorProductPartialAssemblyOperator",
    "WorksetProgram",
    "WorksetSignature",
    "average",
    "darcy_form",
    "linear_elasticity_form",
    "maxwell_form",
    "SIPGBoundaryCondition",
    "SIPGPenaltyPolicy",
    "finite_element_parameter_gradient",
    "sipg_dirichlet",
    "sipg_neumann",
    "sipg_poisson_form",
    "sipg_robin",
    "smoothed_elasticity_form",
    "solve_hdg_poisson",
    "stokes_form",
    "upwind_advection_form",
    "compile_workset_program",
    "kernel_table_from_form",
    "curl",
    "divergence",
    "evaluate_differential_operator",
    "solve_finite_element_adjoint",
    "execute_local_action",
    "jump",
    "lower_finite_element_form",
    "normal_trace",
    "symmetric_gradient",
    "tangential_trace",
]
