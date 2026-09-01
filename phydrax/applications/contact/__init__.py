#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._adapters import (
    DeformableMPMContactAdapter,
    DeformableMPMContactEvaluation,
    FiniteElementContactAssembly,
    FiniteElementContactBoundary,
    FixedEpochNeuralContactAdapter,
    FixedEpochNeuralContactEvaluation,
)
from ._geometry import (
    ContactConfiguration,
    ContactPatch,
    ContactPatchSet,
    ContactQueryPlan,
    ContactQueryResult,
    ContactSurface,
)
from ._laws import (
    AbstractNormalContactLaw,
    AugmentedLagrangianContactLaw,
    CoulombContactLaw,
    CoulombContactResponse,
    FrictionlessPDASContactLaw,
    NormalContactResponse,
    PenaltyContactLaw,
    PenaltyConvergenceEvidence,
)
from ._mechanics import FixedEpochContactOperator
from ._state import (
    AcceptedContactState,
    CONTACT_OPEN,
    CONTACT_SLIP,
    CONTACT_STICK,
    ContactEpochTransaction,
    ContactEvaluation,
    ContactStateTransaction,
)
from ._weak_forms import (
    ContactMortarEvidence,
    ContactMortarSpace,
    NitscheContactEvidence,
    NitscheContactPolicy,
)


__all__ = [
    "AbstractNormalContactLaw",
    "AcceptedContactState",
    "AugmentedLagrangianContactLaw",
    "CONTACT_OPEN",
    "CONTACT_SLIP",
    "CONTACT_STICK",
    "ContactConfiguration",
    "ContactEpochTransaction",
    "ContactEvaluation",
    "ContactMortarEvidence",
    "ContactMortarSpace",
    "ContactPatch",
    "ContactPatchSet",
    "ContactQueryPlan",
    "ContactQueryResult",
    "ContactStateTransaction",
    "ContactSurface",
    "CoulombContactLaw",
    "CoulombContactResponse",
    "DeformableMPMContactAdapter",
    "DeformableMPMContactEvaluation",
    "FiniteElementContactAssembly",
    "FiniteElementContactBoundary",
    "FixedEpochContactOperator",
    "FixedEpochNeuralContactAdapter",
    "FixedEpochNeuralContactEvaluation",
    "FrictionlessPDASContactLaw",
    "NitscheContactEvidence",
    "NitscheContactPolicy",
    "NormalContactResponse",
    "PenaltyContactLaw",
    "PenaltyConvergenceEvidence",
]
