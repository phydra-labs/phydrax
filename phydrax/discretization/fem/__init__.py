#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming finite element discretizations."""

from ._adaptivity import (
    dual_weighted_residual_estimate,
    FiniteElementErrorEstimate,
    FiniteElementRefinementMap,
    FiniteElementTransferPlan,
    FiniteElementTransferRole,
    residual_jump_estimate,
)
from ._constraints import (
    affine_dof_constraint,
    dirichlet_constraint,
    FiniteElementDirichletConstraint,
)
from ._distributed import (
    DistributedFiniteElementConstraint,
    FiniteElementHaloPlan,
    PartitionedFiniteElementDofMap,
)
from ._embedded import (
    EmbeddedQuadrature,
    FiniteElementEnrichment,
    MultiscaleFiniteElementBasis,
)
from ._generic import (
    FiniteElementCoordinateSpec,
    FiniteElementDiscretization,
    FiniteElementDofMap,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementRuntimeData,
    IntegrationDomain,
)
from ._hdg import HDGCondensationPlan, HDGTraceSpace
from ._io import evaluate_finite_element_field, write_finite_element_field
from ._local_elimination import (
    FiniteElementLocalEliminationPlan,
    LocalEliminationResult,
)
from ._precision import FiniteElementPrecisionPolicy
from ._reference import (
    discontinuous_element,
    FiniteElementSpec,
    lagrange_element,
    nedelec_element,
    raviart_thomas_element,
)


__all__ = [
    "FiniteElementErrorEstimate",
    "FiniteElementRefinementMap",
    "FiniteElementTransferPlan",
    "FiniteElementTransferRole",
    "affine_dof_constraint",
    "dual_weighted_residual_estimate",
    "residual_jump_estimate",
    "DistributedFiniteElementConstraint",
    "EmbeddedQuadrature",
    "FiniteElementEnrichment",
    "MultiscaleFiniteElementBasis",
    "FiniteElementHaloPlan",
    "PartitionedFiniteElementDofMap",
    "FiniteElementDirichletConstraint",
    "evaluate_finite_element_field",
    "write_finite_element_field",
    "discontinuous_element",
    "dirichlet_constraint",
    "FiniteElementCoordinateSpec",
    "FiniteElementDiscretization",
    "FiniteElementDofMap",
    "FiniteElementLocalEliminationPlan",
    "FiniteElementFieldSpec",
    "FiniteElementPlan",
    "HDGCondensationPlan",
    "HDGTraceSpace",
    "FiniteElementRuntimeData",
    "FiniteElementPrecisionPolicy",
    "FiniteElementSpec",
    "IntegrationDomain",
    "nedelec_element",
    "raviart_thomas_element",
    "lagrange_element",
    "LocalEliminationResult",
]
