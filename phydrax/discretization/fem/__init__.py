#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming finite element discretizations."""

from ._constraints import dirichlet_constraint, FiniteElementDirichletConstraint
from ._generic import (
    FiniteElementDiscretization,
    FiniteElementDofMap,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    IntegrationDomain,
)
from ._precision import FiniteElementPrecisionPolicy
from ._reference import FiniteElementSpec, lagrange_element


__all__ = [
    "FiniteElementDirichletConstraint",
    "dirichlet_constraint",
    "FiniteElementDiscretization",
    "FiniteElementDofMap",
    "FiniteElementFieldSpec",
    "FiniteElementPlan",
    "FiniteElementPrecisionPolicy",
    "FiniteElementSpec",
    "IntegrationDomain",
    "lagrange_element",
]
