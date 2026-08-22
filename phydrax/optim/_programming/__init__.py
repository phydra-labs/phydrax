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
    "plan_convex_program",
    "prepare_convex_program",
    "prepare_convex_template",
    "refresh_convex_program",
    "solve_convex_program",
    "solve_conic_program",
    "solve_prepared_convex_program",
    "solve_linear_program",
    "solve_quadratic_program",
    "solve_quadratic_program_primal",
]
