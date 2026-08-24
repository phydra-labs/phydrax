#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Canonical linear, quadratic, and conic mathematical programs."""

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
from ._policy import (
    AbstractConvexProgramMethod,
    ClarabelInteriorPoint,
    ConvexDifferentiationMode,
    ConvexDifferentiationPolicy,
    ConvexSolvePolicy,
    ConvexTermination,
    DensePrimalDualQP,
    MPAXr2HPDHG,
    MPAXraPDHG,
    QPaxInteriorPoint,
)
from ._problem import ConicProgram, LinearProgram
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
    "ConicProgramData",
    "ConicSensitivityResult",
    "ConvexDifferentiationMode",
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
    "LinearProgram",
    "NonnegativeCone",
    "PreparedConvexProgram",
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
    "conic_primal_jvp",
    "conic_primal_vjp",
    "plan_convex_program",
    "prepare_convex_program",
    "prepare_convex_template",
    "prepare_conic_sensitivity",
    "refresh_convex_program",
    "solve_convex_program",
    "solve_conic_program",
    "solve_prepared_convex_program",
    "solve_linear_program",
    "solve_quadratic_program",
    "solve_quadratic_program_primal",
]
