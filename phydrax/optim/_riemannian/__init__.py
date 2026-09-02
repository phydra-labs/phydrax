#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._adaptive import riemannian_adam, RiemannianAdam, RiemannianAdamState
from ._advanced import (
    AbstractRiemannianLineSearchOptimizer,
    riemannian_conjugate_gradient,
    riemannian_lbfgs,
    RiemannianConjugateGradient,
    RiemannianConjugateGradientState,
    RiemannianLBFGS,
    RiemannianLBFGSState,
)
from ._first_order import (
    AbstractRiemannianOptimizer,
    riemannian_momentum,
    riemannian_sgd,
    RiemannianMomentum,
    RiemannianMomentumState,
    RiemannianSGD,
    RiemannianSGDState,
    RiemannianStepMetrics,
)
from ._line_search import armijo_backtracking, ArmijoLineSearch, ArmijoResult
from ._parameter_geometry import ParameterGeometry
from ._private import (
    PrivateRiemannianSGD,
    PrivateRiemannianSGDState,
    PrivateRiemannianStepEvidence,
)


__all__ = [
    "PrivateRiemannianSGD",
    "PrivateRiemannianSGDState",
    "PrivateRiemannianStepEvidence",
    "AbstractRiemannianLineSearchOptimizer",
    "ArmijoLineSearch",
    "ArmijoResult",
    "AbstractRiemannianOptimizer",
    "ParameterGeometry",
    "RiemannianAdam",
    "RiemannianAdamState",
    "RiemannianMomentum",
    "RiemannianMomentumState",
    "RiemannianSGD",
    "RiemannianSGDState",
    "RiemannianConjugateGradient",
    "RiemannianConjugateGradientState",
    "RiemannianLBFGS",
    "RiemannianLBFGSState",
    "RiemannianStepMetrics",
    "riemannian_adam",
    "riemannian_momentum",
    "riemannian_sgd",
    "armijo_backtracking",
    "riemannian_conjugate_gradient",
    "riemannian_lbfgs",
]
