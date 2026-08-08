#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reusable neural network layers."""

from ._dropout import Dropout, inference_mode
from ._fourier_embeddings import (
    ExplicitFourierFeatureEmbeddings,
    HybridFourierFeatureEmbeddings,
    MultiscaleFourierFeatureEmbeddings,
    RandomFourierFeatureEmbeddings,
    TrainableFourierFeatureEmbeddings,
)
from ._fourier_sampling import FourierEvaluationMethod, sample_fourier_grid
from ._linear import Linear
from ._manifold_warp import (
    ManifoldMultiheadWarp,
    ManifoldWarpDiagnostics,
    sphere_retraction,
    sphere_tangent_projection,
)
from ._measure_attention import (
    AttentionExecution,
    AttentionKernel,
    MeasureAwareAttention,
)
from ._probabilistic_warp import ProbabilisticMultiheadWarp
from ._warp import MultiheadWarp, WarpBoundaryMode
from ._warp_geometry import (
    conservative_remap,
    GaussianWarpRoute,
    normalized_axis_nodes,
    normalized_lattice_from_nodes,
    RectilinearWarpDiagnostics,
    sample_rectilinear_grid,
    warp_field,
    warp_jacobian,
    WarpMaskMode,
)


__all__ = [
    "AttentionExecution",
    "AttentionKernel",
    "MeasureAwareAttention",
    "ExplicitFourierFeatureEmbeddings",
    "HybridFourierFeatureEmbeddings",
    "MultiscaleFourierFeatureEmbeddings",
    "RandomFourierFeatureEmbeddings",
    "TrainableFourierFeatureEmbeddings",
    "FourierEvaluationMethod",
    "sample_fourier_grid",
    "Dropout",
    "Linear",
    "MultiheadWarp",
    "ProbabilisticMultiheadWarp",
    "ManifoldMultiheadWarp",
    "ManifoldWarpDiagnostics",
    "sphere_retraction",
    "sphere_tangent_projection",
    "RectilinearWarpDiagnostics",
    "GaussianWarpRoute",
    "WarpMaskMode",
    "conservative_remap",
    "normalized_axis_nodes",
    "normalized_lattice_from_nodes",
    "sample_rectilinear_grid",
    "warp_field",
    "warp_jacobian",
    "WarpBoundaryMode",
    "inference_mode",
]
