#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""External ionizing-radiation ledgers to qualified initial-lesion observables.

Transport/spatial chemistry remain external. Initial lesions, candidate probabilities,
clusters and yield calibration are distinct from biological repair or survival.
"""

from . import interchange
from ._clusters import (
    cluster_radiation_lesions,
    contour_distance,
    LesionCluster,
    RadiationClusters,
)
from ._interactions import (
    InteractionLedger,
    PhysicalInteraction,
    PrimaryHistoryKey,
    RadiationEventKey,
    RadiationSource,
)
from ._lesions import (
    candidate_radiation_lesions,
    IndirectLesionRule,
    InitialLesion,
    InitialLesionLedger,
    LesionCandidate,
    LesionCandidates,
    LesionPolicy,
    realize_radiation_lesions,
)
from ._qualification import (
    calibrate_radiation_lesions,
    expected_initial_lesion_yield,
    LesionExpectationSupport,
    prepare_lesion_expectation,
    RadiationCalibrationData,
    RadiationCalibrationResult,
    RadiationCondition,
    RadiationStageEvidence,
)
from ._quantities import GRAY, HistoryExposure, radiation_yield, RadiationYield
from ._reactions import ChemicalReaction, ReactionLedger
from ._targets import (
    map_radiation_targets,
    prepare_radiation_targets,
    PreparedRadiationTargets,
    RadiationTargetGeometry,
    SourceTargetRoute,
    TargetHit,
    TargetMapping,
    TargetMolecule,
    TargetSite,
)


__all__ = [
    "interchange",
    "InteractionLedger",
    "PhysicalInteraction",
    "PrimaryHistoryKey",
    "RadiationEventKey",
    "RadiationSource",
    "ChemicalReaction",
    "ReactionLedger",
    "PreparedRadiationTargets",
    "RadiationTargetGeometry",
    "SourceTargetRoute",
    "TargetHit",
    "TargetMapping",
    "TargetMolecule",
    "TargetSite",
    "map_radiation_targets",
    "prepare_radiation_targets",
    "IndirectLesionRule",
    "InitialLesion",
    "InitialLesionLedger",
    "LesionCandidate",
    "LesionCandidates",
    "LesionPolicy",
    "candidate_radiation_lesions",
    "realize_radiation_lesions",
    "LesionCluster",
    "RadiationClusters",
    "cluster_radiation_lesions",
    "contour_distance",
    "GRAY",
    "HistoryExposure",
    "RadiationYield",
    "radiation_yield",
    "LesionExpectationSupport",
    "RadiationCalibrationData",
    "RadiationCalibrationResult",
    "RadiationCondition",
    "RadiationStageEvidence",
    "calibrate_radiation_lesions",
    "expected_initial_lesion_yield",
    "prepare_lesion_expectation",
]
