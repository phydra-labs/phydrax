#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

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


__all__ = [
    "AbstractRiemannianLineSearchOptimizer",
    "ArmijoLineSearch",
    "ArmijoResult",
    "AbstractRiemannianOptimizer",
    "ParameterGeometry",
    "RiemannianMomentum",
    "RiemannianMomentumState",
    "RiemannianSGD",
    "RiemannianSGDState",
    "RiemannianConjugateGradient",
    "RiemannianConjugateGradientState",
    "RiemannianLBFGS",
    "RiemannianLBFGSState",
    "RiemannianStepMetrics",
    "riemannian_momentum",
    "riemannian_sgd",
    "armijo_backtracking",
    "riemannian_conjugate_gradient",
    "riemannian_lbfgs",
]
