#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conforming finite element discretizations."""

from . import smoothing
from ._adaptivity import (
    coarsen_triangles_local,
    dorfler_mark,
    dual_weighted_residual_estimate,
    FiniteElementAdaptationMap,
    FiniteElementDWRIndicators,
    FiniteElementErrorEstimate,
    FiniteElementRefinementMap,
    FiniteElementTransferBundle,
    FiniteElementTransferPlan,
    FiniteElementTransferRole,
    local_dual_weighted_residual,
    maximum_mark,
    refine_triangles_local,
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
    SimplexNodalFamily,
    SumFactorizationPlan,
    TensorProductTabulation,
)
from ._io import evaluate_finite_element_field, write_finite_element_field
from ._local_elimination import (
    FiniteElementLocalEliminationPlan,
    LocalEliminationResult,
)
from ._low_order_auxiliary import (
    LowOrderAuxiliaryOperatorPlan,
    LowOrderAuxiliaryPreconditioner,
)
from ._multigrid import (
    finite_element_p_transfer,
    FiniteElementPTransfer,
    PTransferData,
    PTransferRole,
    quadrilateral_p_transfer,
)
from ._p_multigrid import (
    finite_element_p_multigrid_plan,
    FiniteElementPMultigridPlan,
    FiniteElementPMultigridPolicy,
    PDegreeCoarsening,
)
from ._patch_preconditioning import (
    FiniteElementPatchPlan,
    FiniteElementPatchPreconditioner,
    one_ring_patch_plan,
)
from ._precision import FiniteElementPrecisionPolicy
from ._reference import (
    discontinuous_element,
    FiniteElementSpec,
    lagrange_element,
    nedelec_element,
    raviart_thomas_element,
)
from ._reference_topology import (
    reference_cell_topology,
    REFERENCE_TOPOLOGIES,
    ReferenceCellTopology,
)


__all__ = [
    "smoothing",
    "FiniteElementAdaptationMap",
    "FiniteElementDWRIndicators",
    "FiniteElementErrorEstimate",
    "FiniteElementRefinementMap",
    "FiniteElementTransferBundle",
    "FiniteElementTransferPlan",
    "NodeSet",
    "QuadratureChunkPolicy",
    "ReferenceNodalFamily",
    "SimplexNodalFamily",
    "SumFactorizationPlan",
    "TensorProductTabulation",
    "lagrange_1d_tabulation",
    "REFERENCE_TOPOLOGIES",
    "ReferenceCellTopology",
    "reference_cell_topology",
    "local_diagonal",
    "FiniteElementTransferRole",
    "coarsen_triangles_local",
    "LowOrderAuxiliaryOperatorPlan",
    "LowOrderAuxiliaryPreconditioner",
    "dorfler_mark",
    "local_dual_weighted_residual",
    "maximum_mark",
    "refine_triangles_local",
    "finite_element_p_transfer",
    "FiniteElementPTransfer",
    "PTransferData",
    "PTransferRole",
    "quadrilateral_p_transfer",
    "finite_element_p_multigrid_plan",
    "FiniteElementPMultigridPlan",
    "FiniteElementPMultigridPolicy",
    "PDegreeCoarsening",
    "FiniteElementPatchPlan",
    "FiniteElementPatchPreconditioner",
    "one_ring_patch_plan",
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
