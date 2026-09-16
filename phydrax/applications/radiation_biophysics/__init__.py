#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Governed ionizing-radiation ledgers to qualified initial-lesion observables.

Interaction ledgers may originate from an external engine or the explicit native
radiation-transport host adapter. Spatial radiation chemistry remains external.
Initial lesions, candidate probabilities, clusters and yield calibration are
distinct from biological repair or survival.
"""

from . import circulating_blood, interchange
from ._clusters import (
    cluster_radiation_lesions,
    contour_distance,
    LesionCluster,
    RadiationClusters,
)
from ._initial_lesion_qualification import (
    assess_radiation_initial_lesions,
    RadiationInitialLesionAssessment,
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
from ._plasmid_gel import (
    evaluate_plasmid_gel,
    PLASMID_FORMS,
    PlasmidFormPrediction,
    PlasmidGelAssay,
    PlasmidGelEvaluation,
    PlasmidGelObservations,
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
from ._quantities import HistoryExposure, radiation_yield, RadiationYield
from ._reactions import ChemicalReaction, ReactionLedger
from ._scores import (
    ExternalRadiationRunIdentity,
    ExternalRadiationScoreResult,
    radiation_score_content_id,
    RadiationEstimatorEvidence,
    RadiationScoreDefinition,
)
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
from .circulating_blood import (
    ABSORBED_DOSE_RATE_REFERENCE,
    BloodCompartment,
    BloodFlow,
    BloodTransitJumpProcess,
    circulating_blood_dose_rate_quantity,
    CIRCULATING_BLOOD_DOSE_RATE_REFERENCES,
    CIRCULATING_BLOOD_DOSE_RATE_SUPPORT,
    CirculatingBloodModel,
    CirculationCapacityEvidence,
    DeterministicBloodDoseResult,
    DOSE_TO_MEDIUM_RATE_REFERENCE,
    DOSE_TO_WATER_RATE_REFERENCE,
    DoseRateInterval,
    HistoryCapacityEvidence,
    integrate_circulating_blood_dose,
    PiecewiseConstantDoseRateSchedule,
    prepare_circulating_blood_model,
    prepare_spatial_compartment_mixture,
    PreparedCirculatingBloodModel,
    PreparedSpatialCompartmentMixture,
    score_circulating_blood_histories,
    simulate_circulating_blood_dose,
    StochasticBloodDoseResult,
)


__all__ = [
    "circulating_blood",
    "ABSORBED_DOSE_RATE_REFERENCE",
    "CIRCULATING_BLOOD_DOSE_RATE_REFERENCES",
    "CIRCULATING_BLOOD_DOSE_RATE_SUPPORT",
    "DOSE_TO_MEDIUM_RATE_REFERENCE",
    "DOSE_TO_WATER_RATE_REFERENCE",
    "BloodCompartment",
    "BloodFlow",
    "BloodTransitJumpProcess",
    "CirculatingBloodModel",
    "CirculationCapacityEvidence",
    "DeterministicBloodDoseResult",
    "DoseRateInterval",
    "ExternalRadiationRunIdentity",
    "ExternalRadiationScoreResult",
    "HistoryCapacityEvidence",
    "PiecewiseConstantDoseRateSchedule",
    "PreparedCirculatingBloodModel",
    "PreparedSpatialCompartmentMixture",
    "RadiationEstimatorEvidence",
    "RadiationScoreDefinition",
    "StochasticBloodDoseResult",
    "circulating_blood_dose_rate_quantity",
    "integrate_circulating_blood_dose",
    "prepare_circulating_blood_model",
    "prepare_spatial_compartment_mixture",
    "radiation_score_content_id",
    "score_circulating_blood_histories",
    "simulate_circulating_blood_dose",
    "interchange",
    "PLASMID_FORMS",
    "PlasmidFormPrediction",
    "PlasmidGelAssay",
    "PlasmidGelEvaluation",
    "PlasmidGelObservations",
    "RadiationInitialLesionAssessment",
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
    "HistoryExposure",
    "RadiationYield",
    "radiation_yield",
    "assess_radiation_initial_lesions",
    "evaluate_plasmid_gel",
    "LesionExpectationSupport",
    "RadiationCalibrationData",
    "RadiationCalibrationResult",
    "RadiationCondition",
    "RadiationStageEvidence",
    "calibrate_radiation_lesions",
    "expected_initial_lesion_yield",
    "prepare_lesion_expectation",
]
