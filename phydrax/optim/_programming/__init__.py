#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical linear, quadratic, and conic mathematical programs."""

from ._barrier import cone_barrier_oracle, ConeBarrierOracle
from ._cones import (
    AbstractConvexCone,
    NonnegativeCone,
    ProductCone,
    RotatedSecondOrderCone,
    SecondOrderCone,
    ZeroCone,
)
from ._conic_sensitivity import (
    conic_primal_jvp,
    conic_primal_vjp,
    ConicProgramData,
    ConicSensitivityResult,
    prepare_conic_sensitivity,
    PreparedConicSensitivity,
)
from ._cvxpy import (
    CVXPYConstraintSlice,
    CVXPYProgramBinding,
    CVXPYVariableSlice,
    export_cvxpy_program,
    import_cvxpy_problem,
    refresh_cvxpy_program,
    restore_cvxpy_solution,
)
from ._exponential_cone import ExponentialCone
from ._lifecycle import (
    bind_convex_numeric,
    CanonicalProgram,
    ConvexProgramExecution,
    ConvexProgramPlan,
    ConvexProgramTemplate,
    ConvexWarmStart,
    plan_convex_program,
    prepare_convex_program,
    prepare_convex_template,
    PreparedConvexProgram,
    refresh_convex_program,
    solve_conic_program,
    solve_convex_program,
    solve_linear_program,
    solve_prepared_convex_program,
)
from ._matrix_free_conic_sensitivity import PreparedMatrixFreeConicSensitivity
from ._mixed_integer import (
    MixedIntegerBranchingRule,
    MixedIntegerProgram,
    MixedIntegerResult,
    MixedIntegerSolvePolicy,
    MixedIntegerStatus,
    solve_mixed_integer_program,
)
from ._policy import (
    AbstractConvexProgramMethod,
    ClarabelInteriorPoint,
    ConicGeneralizedDerivativePolicy,
    ConvexDifferentiationMode,
    ConvexDifferentiationPolicy,
    ConvexSolvePolicy,
    ConvexTermination,
    DensePrimalDualQP,
    MPAXr2HPDHG,
    MPAXraPDHG,
    NativeHomogeneousConic,
    QPaxInteriorPoint,
)
from ._power_cone import PowerCone
from ._problem import ConicProgram, LinearProgram
from ._psd_cone import PositiveSemidefiniteCone
from ._quadratic import (
    ConvexProgramResult,
    QuadraticProgram,
    solve_quadratic_program,
    solve_quadratic_program_primal,
)
from ._types import (
    convex_program_status_message,
    ConvexProgramCapabilities,
    ConvexProgramCertificate,
    ConvexProgramProvenance,
    ConvexProgramStatus,
)


__all__ = [
    "AbstractConvexCone",
    "AbstractConvexProgramMethod",
    "CanonicalProgram",
    "ConicProgram",
    "ConeBarrierOracle",
    "CVXPYConstraintSlice",
    "ConicProgramData",
    "CVXPYProgramBinding",
    "CVXPYVariableSlice",
    "ConicSensitivityResult",
    "ConvexDifferentiationMode",
    "ConicGeneralizedDerivativePolicy",
    "ConvexDifferentiationPolicy",
    "ClarabelInteriorPoint",
    "ConvexProgramCapabilities",
    "ConvexProgramCertificate",
    "ConvexProgramExecution",
    "ConvexProgramPlan",
    "ConvexProgramProvenance",
    "ConvexProgramStatus",
    "ConvexProgramTemplate",
    "ConvexSolvePolicy",
    "ConvexTermination",
    "ConvexWarmStart",
    "DensePrimalDualQP",
    "ExponentialCone",
    "MixedIntegerBranchingRule",
    "MixedIntegerProgram",
    "MixedIntegerResult",
    "MixedIntegerSolvePolicy",
    "MixedIntegerStatus",
    "NativeHomogeneousConic",
    "LinearProgram",
    "NonnegativeCone",
    "PositiveSemidefiniteCone",
    "PowerCone",
    "PreparedConvexProgram",
    "PreparedMatrixFreeConicSensitivity",
    "PreparedConicSensitivity",
    "ProductCone",
    "MPAXr2HPDHG",
    "MPAXraPDHG",
    "QPaxInteriorPoint",
    "QuadraticProgram",
    "ConvexProgramResult",
    "RotatedSecondOrderCone",
    "SecondOrderCone",
    "ZeroCone",
    "bind_convex_numeric",
    "convex_program_status_message",
    "cone_barrier_oracle",
    "conic_primal_jvp",
    "conic_primal_vjp",
    "export_cvxpy_program",
    "import_cvxpy_problem",
    "refresh_cvxpy_program",
    "restore_cvxpy_solution",
    "plan_convex_program",
    "prepare_convex_program",
    "prepare_convex_template",
    "prepare_conic_sensitivity",
    "refresh_convex_program",
    "solve_convex_program",
    "solve_conic_program",
    "solve_prepared_convex_program",
    "solve_linear_program",
    "solve_mixed_integer_program",
    "solve_quadratic_program",
    "solve_quadratic_program_primal",
]
