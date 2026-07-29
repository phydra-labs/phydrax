#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reusable neural network layers."""

from ._dropout import Dropout, inference_mode
from ._graph_transfer import (
    GeometryMomentEmbedding,
    GraphAttentionTransfer,
    GraphKernelTransfer,
    MultiscaleGraphTransfer,
)
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
from ._operator_attention import (
    AxialOperatorAttention,
    CodomainAttention,
    OperatorAttention,
    SliceAttention,
)
from ._operator_transformer import OperatorTransformerProcessor
from ._probabilistic_warp import ProbabilisticMultiheadWarp
from ._regional_processor import RegionalGraphProcessor
from ._spectral import BasisSpectralConvND, BasisTransformPlan, SpectralBasis
from ._warp import MultiheadWarp, WarpBoundaryMode
from ._warp_geometry import (
    conservative_remap,
    DENSITY_WARP_FIELD,
    GaussianWarpRoute,
    normalized_axis_nodes,
    normalized_lattice_from_nodes,
    RectilinearWarpDiagnostics,
    sample_rectilinear_grid,
    SCALAR_WARP_FIELD,
    warp_field,
    warp_jacobian,
    WarpFieldSpec,
    WarpMaskMode,
    WarpVariance,
)


__all__ = [
    "BasisSpectralConvND",
    "BasisTransformPlan",
    "AxialOperatorAttention",
    "CodomainAttention",
    "OperatorAttention",
    "SliceAttention",
    "AttentionExecution",
    "AttentionKernel",
    "MeasureAwareAttention",
    "OperatorTransformerProcessor",
    "GeometryMomentEmbedding",
    "GraphAttentionTransfer",
    "GraphKernelTransfer",
    "MultiscaleGraphTransfer",
    "RegionalGraphProcessor",
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
    "WarpFieldSpec",
    "SCALAR_WARP_FIELD",
    "DENSITY_WARP_FIELD",
    "WarpMaskMode",
    "WarpVariance",
    "conservative_remap",
    "normalized_axis_nodes",
    "normalized_lattice_from_nodes",
    "sample_rectilinear_grid",
    "warp_field",
    "warp_jacobian",
    "WarpBoundaryMode",
    "SpectralBasis",
    "inference_mode",
]
