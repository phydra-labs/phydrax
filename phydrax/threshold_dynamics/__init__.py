#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hard-label threshold dynamics for capillarity-driven multiphase geometry."""

from ..geometry.multiregion_surface import (
    LabelFieldSurfaceExtractionEvidence,
    LabelFieldSurfaceExtractionPlan,
    LabelFieldSurfaceExtractionResult,
    LabelFieldSurfaceExtractionRoute,
    LabelFieldSurfaceExtractionStatus,
    LabelFieldSurfaceLineage,
    PreparedLabelFieldSurfaceExtraction,
)
from ._coarsening import (
    GasDiffusionCoarsening,
    GasDiffusionEvidence,
    GasDiffusionRunResult,
)
from ._contracts import (
    AbstractThresholdHeatKernel,
    decompose_threshold_kernel,
    HeatActionEvidence,
    LabelFieldState,
    SparseCandidateEvidence,
    ThresholdDynamicsEvidence,
    ThresholdDynamicsResourcePolicy,
    ThresholdDynamicsRunResult,
    ThresholdDynamicsStepResult,
    ThresholdKernelDecomposition,
    ThresholdKernelForm,
    ThresholdPotentials,
    VolumeConstraintEvidence,
)
from ._mesh import MeshHeatKernel
from ._periodic import PeriodicGridHeatKernel
from ._profiles import threshold_dynamics_candidate_profiles
from ._qualification import (
    label_contacts,
    label_neighbor_counts,
    triple_junction_angles,
    von_neumann_mullins_fit,
)
from ._sparse import SparseLabelGrid
from ._status import threshold_dynamics_status_message, ThresholdDynamicsStatus
from ._step import PreparedThresholdDynamics, ThresholdDynamicsPlan, ThresholdRoute
from ._volume_constraints import LabelVolumeConstraint


__all__ = [
    "AbstractThresholdHeatKernel",
    "GasDiffusionCoarsening",
    "GasDiffusionEvidence",
    "GasDiffusionRunResult",
    "HeatActionEvidence",
    "LabelFieldState",
    "LabelFieldSurfaceExtractionEvidence",
    "LabelFieldSurfaceExtractionPlan",
    "LabelFieldSurfaceExtractionResult",
    "LabelFieldSurfaceExtractionRoute",
    "LabelFieldSurfaceExtractionStatus",
    "LabelFieldSurfaceLineage",
    "LabelVolumeConstraint",
    "MeshHeatKernel",
    "PeriodicGridHeatKernel",
    "PreparedLabelFieldSurfaceExtraction",
    "PreparedThresholdDynamics",
    "SparseCandidateEvidence",
    "SparseLabelGrid",
    "ThresholdDynamicsEvidence",
    "ThresholdDynamicsPlan",
    "ThresholdDynamicsResourcePolicy",
    "ThresholdDynamicsRunResult",
    "ThresholdDynamicsStatus",
    "ThresholdDynamicsStepResult",
    "ThresholdKernelDecomposition",
    "ThresholdKernelForm",
    "ThresholdPotentials",
    "ThresholdRoute",
    "VolumeConstraintEvidence",
    "decompose_threshold_kernel",
    "label_contacts",
    "label_neighbor_counts",
    "threshold_dynamics_candidate_profiles",
    "threshold_dynamics_status_message",
    "triple_junction_angles",
    "von_neumann_mullins_fit",
]
