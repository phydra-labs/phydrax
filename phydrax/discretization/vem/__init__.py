#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Projection-native virtual element discretizations on polygonal meshes."""

from ._constraints import (
    virtual_element_dirichlet_constraint,
    VirtualElementDirichletConstraint,
)
from ._dofs import VirtualElementDofMap
from ._operator import FactorizedVirtualElementOperator
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
    "FactorizedVirtualElementOperator",
    "StabilizedVirtualElementTensor",
    "VirtualElementDirichletConstraint",
    "VirtualElementDiscretization",
    "VirtualElementDofMap",
    "VirtualElementFieldSpec",
    "VirtualElementPlan",
    "VirtualElementPrecisionPolicy",
    "VirtualElementProjectionData",
    "VirtualElementProjectionEvidence",
    "VirtualElementResourceBudget",
    "VirtualElementRuntimeData",
    "VirtualElementSpec",
    "VirtualElementStabilizationEvidence",
    "VirtualElementStabilizationPolicy",
    "conforming_h1_virtual_element",
    "prepare_virtual_element_projections",
    "virtual_element_dirichlet_constraint",
    "stabilize_virtual_element_tensor",
]
