#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming finite element discretizations."""

from ._p1 import (
    p1_local_matrices,
    P1DirichletElimination,
    P1FiniteElementDiscretization,
    P1FiniteElementPlan,
    P1HeatDynamics,
)


__all__ = [
    "P1DirichletElimination",
    "P1FiniteElementDiscretization",
    "P1FiniteElementPlan",
    "P1HeatDynamics",
    "p1_local_matrices",
]
