#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming finite element discretizations."""

from . import smoothing
from ._adaptivity import (
    adaptive_uniform_solve,
    dual_weighted_residual_estimate,
    FiniteElementErrorEstimate,
    FiniteElementRefinementMap,
    FiniteElementTransferPlan,
    FiniteElementTransferRole,
    refine_triangles_uniform,
    residual_jump_estimate,
)
from ._constraints import (
    affine_dof_constraint,
    dirichlet_constraint,
    FiniteElementDirichletConstraint,
)
from ._distributed import (
    DistributedFiniteElementConstraint,
    DistributedFiniteElementOperator,
    FiniteElementHaloPlan,
    FiniteElementPartition,
    JaxCollectiveBackend,
    partition_cells_contiguous,
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
from ._high_order import (
    lagrange_1d_tabulation,
    local_diagonal,
    NodeSet,
    QuadratureChunkPolicy,
    ReferenceNodalFamily,
    SumFactorizationPlan,
    TensorProductTabulation,
)
from ._io import evaluate_finite_element_field, write_finite_element_field
from ._local_elimination import (
    FiniteElementLocalEliminationPlan,
    LocalEliminationResult,
)
from ._multigrid import PTransferData, quadrilateral_p_transfer
from ._precision import FiniteElementPrecisionPolicy
from ._reference import (
    discontinuous_element,
    FiniteElementSpec,
    lagrange_element,
    nedelec_element,
    raviart_thomas_element,
)


__all__ = [
    "smoothing",
    "FiniteElementErrorEstimate",
    "FiniteElementRefinementMap",
    "FiniteElementTransferPlan",
    "NodeSet",
    "QuadratureChunkPolicy",
    "ReferenceNodalFamily",
    "SumFactorizationPlan",
    "TensorProductTabulation",
    "lagrange_1d_tabulation",
    "local_diagonal",
    "FiniteElementTransferRole",
    "adaptive_uniform_solve",
    "PTransferData",
    "quadrilateral_p_transfer",
    "refine_triangles_uniform",
    "affine_dof_constraint",
    "dual_weighted_residual_estimate",
    "residual_jump_estimate",
    "DistributedFiniteElementConstraint",
    "DistributedFiniteElementOperator",
    "EmbeddedQuadrature",
    "FiniteElementEnrichment",
    "MultiscaleFiniteElementBasis",
    "FiniteElementHaloPlan",
    "FiniteElementPartition",
    "JaxCollectiveBackend",
    "PartitionedFiniteElementDofMap",
    "partition_cells_contiguous",
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
