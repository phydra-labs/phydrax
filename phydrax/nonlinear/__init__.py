#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nonlinear algebraic systems, fixed points, and implicit root calculus."""

from .._bounds import Bounds
from ._fas import (
    fas_cycle,
    FASCycleKind,
    FASCyclePolicy,
    FASDiagnostics,
    FASHierarchy,
    FASLevel,
    FASNonlinearPreconditioner,
    FASResult,
)
from ._fixed_point import AndersonAcceleration, FixedPointIteration, PicardIteration
from ._implicit import implicit_root
from ._linearization import (
    JacobianMode,
    JacobianPolicy,
    prepare_jacobian,
    PreparedJacobian,
)
from ._newton import (
    JacobianRefreshPolicy,
    JacobianRefreshStrategy,
    NewtonForcingPolicy,
    NewtonForcingStrategy,
    NewtonKrylov,
    NewtonTrustRegion,
    root,
    RootLineSearch,
    RootTrustRegion,
)
from ._ngmres import NonlinearGMRES
from ._preconditioning import (
    AbstractLeftNonlinearPreconditioner,
    AbstractNonlinearSystemTransformation,
    AbstractRightNonlinearPreconditioner,
    FunctionLeftNonlinearPreconditioner,
    FunctionRightNonlinearPreconditioner,
    left_precondition,
    LeftPreconditionedSystem,
    right_precondition,
    RightPreconditionedSystem,
)
from ._prepared import (
    prepare_nonlinear,
    PreparedNonlinearSolve,
    refresh_nonlinear,
    solve_prepared_nonlinear,
)
from ._types import (
    AbstractNonlinearMethod,
    FixedPointProblem,
    nonlinear_status_message,
    NonlinearCapabilities,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
    NonlinearTransformationEvidence,
)
from ._vi import (
    complementarity_certificate,
    ComplementarityCertificate,
    ComplementarityFormulation,
    GeneralizedDerivativePolicy,
    SemismoothNewton,
    VariationalInequalityProblem,
    VariationalInequalityResult,
)


__all__ = [
    "AbstractNonlinearMethod",
    "AbstractLeftNonlinearPreconditioner",
    "AbstractNonlinearSystemTransformation",
    "AndersonAcceleration",
    "AbstractRightNonlinearPreconditioner",
    "Bounds",
    "ComplementarityCertificate",
    "ComplementarityFormulation",
    "FASCycleKind",
    "FASCyclePolicy",
    "FASDiagnostics",
    "FASHierarchy",
    "FASLevel",
    "FASNonlinearPreconditioner",
    "FASResult",
    "FixedPointIteration",
    "FixedPointProblem",
    "FunctionLeftNonlinearPreconditioner",
    "FunctionRightNonlinearPreconditioner",
    "GeneralizedDerivativePolicy",
    "JacobianRefreshPolicy",
    "JacobianRefreshStrategy",
    "JacobianMode",
    "JacobianPolicy",
    "LeftPreconditionedSystem",
    "NewtonKrylov",
    "NewtonTrustRegion",
    "NewtonForcingPolicy",
    "NewtonForcingStrategy",
    "NonlinearCapabilities",
    "NonlinearGMRES",
    "NonlinearDiagnostics",
    "NonlinearProvenance",
    "NonlinearTransformationEvidence",
    "NonlinearResult",
    "NonlinearStatus",
    "NonlinearSystemProblem",
    "NonlinearTermination",
    "PicardIteration",
    "PreparedJacobian",
    "PreparedNonlinearSolve",
    "RightPreconditionedSystem",
    "RootLineSearch",
    "RootTrustRegion",
    "SemismoothNewton",
    "VariationalInequalityProblem",
    "VariationalInequalityResult",
    "complementarity_certificate",
    "fas_cycle",
    "implicit_root",
    "left_precondition",
    "nonlinear_status_message",
    "prepare_jacobian",
    "prepare_nonlinear",
    "root",
    "refresh_nonlinear",
    "right_precondition",
    "solve_prepared_nonlinear",
]
