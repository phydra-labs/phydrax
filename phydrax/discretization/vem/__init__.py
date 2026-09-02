#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Projection-native virtual element discretizations on polygonal meshes."""

from ._constraints import (
    virtual_element_dirichlet_constraint,
    VirtualElementDirichletConstraint,
)
from ._dofs import VirtualElementDofMap
from ._extended import (
    adapt_virtual_element_hp,
    adapt_virtual_element_p,
    CurvedVirtualElementEdge,
    prepare_polyhedral_polynomial_vem_3d,
    PreparedPolyhedralPolynomialVEM3D,
    VirtualElementAdaptationResult,
    VirtualElementAdaptivityPolicy,
    VirtualElementEpoch,
    VirtualElementProductPlan,
)
from ._operator import FactorizedVirtualElementOperator
from ._polyhedral import (
    PolyhedralVEMEvidence3D,
    prepare_polyhedral_h1_virtual_element_3d,
    PreparedPolyhedralH1VirtualElement3D,
)
from ._precision import VirtualElementPrecisionPolicy, VirtualElementResourceBudget
from ._projection import (
    prepare_virtual_element_projections,
    VirtualElementProjectionData,
    VirtualElementProjectionEvidence,
)
from ._space import (
    VirtualElementDiscretization,
    VirtualElementPlan,
    VirtualElementRuntimeData,
)
from ._spec import (
    conforming_h1_virtual_element,
    conforming_hcurl_virtual_element,
    conforming_hdiv_virtual_element,
    discontinuous_l2_virtual_element,
    VirtualElementFieldSpec,
    VirtualElementSpec,
)
from ._stabilization import (
    stabilize_virtual_element_tensor,
    StabilizedVirtualElementTensor,
    VirtualElementStabilizationEvidence,
    VirtualElementStabilizationPolicy,
)


__all__ = [
    "adapt_virtual_element_hp",
    "adapt_virtual_element_p",
    "CurvedVirtualElementEdge",
    "FactorizedVirtualElementOperator",
    "PolyhedralVEMEvidence3D",
    "prepare_polyhedral_h1_virtual_element_3d",
    "PreparedPolyhedralH1VirtualElement3D",
    "StabilizedVirtualElementTensor",
    "VirtualElementDirichletConstraint",
    "VirtualElementDiscretization",
    "VirtualElementDofMap",
    "VirtualElementFieldSpec",
    "VirtualElementPlan",
    "VirtualElementAdaptationResult",
    "VirtualElementAdaptivityPolicy",
    "VirtualElementEpoch",
    "prepare_polyhedral_polynomial_vem_3d",
    "PreparedPolyhedralPolynomialVEM3D",
    "VirtualElementProductPlan",
    "VirtualElementPrecisionPolicy",
    "VirtualElementProjectionData",
    "VirtualElementProjectionEvidence",
    "VirtualElementResourceBudget",
    "VirtualElementRuntimeData",
    "VirtualElementSpec",
    "VirtualElementStabilizationEvidence",
    "VirtualElementStabilizationPolicy",
    "conforming_h1_virtual_element",
    "conforming_hcurl_virtual_element",
    "conforming_hdiv_virtual_element",
    "discontinuous_l2_virtual_element",
    "prepare_virtual_element_projections",
    "virtual_element_dirichlet_constraint",
    "stabilize_virtual_element_tensor",
]
