#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed and adaptive quantum-mechanical/molecular-mechanical coupling."""

from ._advanced import (
    AbstractPolarizableEmbeddedRegionProvider,
    AdaptivePartitionedQMMMSurface,
    AdaptiveQMMMEvaluation,
    CallablePolarizableEmbeddedRegionProvider,
    multipole_embedding_for_region,
    MutualPolarizableQMMMSurface,
    MutualPolarizationResult,
    PeriodicMultilevelQMMMSurface,
    PolarizableEmbeddedRegionEvaluation,
)
from ._core import (
    AbstractEmbeddedRegionProvider,
    CallableEmbeddedRegionProvider,
    ElectrostaticEmbeddingQMMMSurface,
    EmbeddedRegionEvaluation,
    NativeRHFEmbeddedRegionProvider,
    PreparedQuantumRegion,
    QMMMEvaluation,
    QuantumRegionPlan,
    SubtractiveQMMMSurface,
)


__all__ = [
    "AbstractPolarizableEmbeddedRegionProvider",
    "AdaptivePartitionedQMMMSurface",
    "AdaptiveQMMMEvaluation",
    "AbstractEmbeddedRegionProvider",
    "CallableEmbeddedRegionProvider",
    "CallablePolarizableEmbeddedRegionProvider",
    "ElectrostaticEmbeddingQMMMSurface",
    "EmbeddedRegionEvaluation",
    "NativeRHFEmbeddedRegionProvider",
    "MutualPolarizableQMMMSurface",
    "MutualPolarizationResult",
    "PeriodicMultilevelQMMMSurface",
    "PolarizableEmbeddedRegionEvaluation",
    "PreparedQuantumRegion",
    "QMMMEvaluation",
    "QuantumRegionPlan",
    "SubtractiveQMMMSurface",
    "multipole_embedding_for_region",
]
