#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from ._contact import CrackFaceContactAdapter
from ._diffuse import (
    BoundedNeuralFixedHistoryController,
    FixedHistoryNeuralBlock,
    PhaseFieldFractureModel,
    PhaseFieldFractureParameters,
    PhaseFieldHistoryState,
    PhaseFieldHistoryTransaction,
)
from ._enrichment import (
    CrackEnrichmentValues,
    CrackTipMaterial,
    IsotropicWilliamsCrackTipBasis,
    shifted_heaviside_enrichment,
    shifted_williams_enrichment,
    ShiftedCrackEnrichment,
)
from ._geometry import (
    build_sharp_crack_topology,
    CrackFrontGeometry,
    CrackProjection,
    SharpCrackTopology,
)
from ._growth import (
    CrackGrowthProposal,
    CrackGrowthTransaction,
    prepare_crack_growth_transaction,
    propose_mixed_mode_growth,
    SharpFractureState,
)
from ._observables import evaluate_interaction_integral, StressIntensityFactors
from ._quadrature import (
    build_sharp_crack_quadrature,
    CrackFaceQuadrature,
    CrackQuadratureEvidence,
    CrackTipQuadrature,
    CrackVolumeQuadrature,
    SharpCrackQuadrature,
)
from ._topology import diffuse_fracture_topology_plan


__all__ = [
    "BoundedNeuralFixedHistoryController",
    "CrackEnrichmentValues",
    "CrackFaceContactAdapter",
    "CrackFaceQuadrature",
    "CrackFrontGeometry",
    "CrackGrowthProposal",
    "CrackGrowthTransaction",
    "CrackProjection",
    "CrackQuadratureEvidence",
    "CrackTipMaterial",
    "CrackTipQuadrature",
    "CrackVolumeQuadrature",
    "FixedHistoryNeuralBlock",
    "IsotropicWilliamsCrackTipBasis",
    "PhaseFieldFractureModel",
    "PhaseFieldFractureParameters",
    "PhaseFieldHistoryState",
    "PhaseFieldHistoryTransaction",
    "SharpCrackQuadrature",
    "SharpCrackTopology",
    "SharpFractureState",
    "ShiftedCrackEnrichment",
    "StressIntensityFactors",
    "build_sharp_crack_quadrature",
    "build_sharp_crack_topology",
    "diffuse_fracture_topology_plan",
    "evaluate_interaction_integral",
    "prepare_crack_growth_transaction",
    "propose_mixed_mode_growth",
    "shifted_heaviside_enrichment",
    "shifted_williams_enrichment",
]
