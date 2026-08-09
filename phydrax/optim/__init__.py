#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phydrax-owned optimization algorithms and workflow configurations."""

from .._model import KFACAffineBlock, KFACLayoutProvider
from ._differential_evolution import DifferentialEvolutionSearch
from ._kfac._config import kfac
from ._quadratic_program import (
    QP_INFEASIBLE,
    QP_MAX_ITERATIONS,
    QP_NONFINITE,
    QP_SUCCESS,
    QPDifferentiableMethod,
    QPMethod,
    QuadraticProgram,
    QuadraticProgramResult,
    solve_quadratic_program,
    solve_quadratic_program_primal,
)
from ._riemannian import (
    ArmijoLineSearch,
    ParameterGeometry,
    riemannian_conjugate_gradient,
    riemannian_lbfgs,
    riemannian_momentum,
    riemannian_sgd,
)


__all__ = [
    "ArmijoLineSearch",
    "DifferentialEvolutionSearch",
    "KFACAffineBlock",
    "KFACLayoutProvider",
    "ParameterGeometry",
    "QP_INFEASIBLE",
    "QP_MAX_ITERATIONS",
    "QP_NONFINITE",
    "QP_SUCCESS",
    "QPDifferentiableMethod",
    "QPMethod",
    "QuadraticProgram",
    "QuadraticProgramResult",
    "kfac",
    "riemannian_conjugate_gradient",
    "riemannian_lbfgs",
    "riemannian_momentum",
    "riemannian_sgd",
    "solve_quadratic_program",
    "solve_quadratic_program_primal",
]
