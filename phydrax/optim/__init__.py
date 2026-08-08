#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phydrax-owned optimization algorithms and workflow configurations."""

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


__all__ = [
    "DifferentialEvolutionSearch",
    "QP_INFEASIBLE",
    "QP_MAX_ITERATIONS",
    "QP_NONFINITE",
    "QP_SUCCESS",
    "QPDifferentiableMethod",
    "QPMethod",
    "QuadraticProgram",
    "QuadraticProgramResult",
    "kfac",
    "solve_quadratic_program",
    "solve_quadratic_program_primal",
]
