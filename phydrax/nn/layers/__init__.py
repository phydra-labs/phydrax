#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reusable neural network layers."""

from ._adaptive_residual import AdaptiveResidual
from ._causal_recurrent import (
    CausalRecurrentConfig,
    CausalRecurrentDiagnostics,
    CausalRecurrentFailurePolicy,
    CausalRecurrentResult,
    run_causal_recurrent,
)
from ._complex_linear import ComplexLinear
from ._diffusion_conditioning import (
    SinusoidalTimeEmbedding,
    TimeConditionedVectorModel,
)
from ._dropout import Dropout, inference_mode
from ._fourier_embeddings import (
    ExplicitFourierFeatureEmbeddings,
    HybridFourierFeatureEmbeddings,
    MultiscaleFourierFeatureEmbeddings,
    RandomFourierFeatureEmbeddings,
    TrainableFourierFeatureEmbeddings,
)
from ._fourier_sampling import FourierEvaluationMethod, sample_fourier_grid
from ._interface import InterfaceDistanceSemantics, InterfaceFeatureLift
from ._linear import Linear
from ._linear_recurrent_unit import LinearRecurrentUnit
from ._low_rank_complex_linear import (
    LowRankComplexLinear,
    LowRankComplexLinearInitializationReport,
)
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
from ._measure_convolution import MeasureNormalizedConvND
from ._probabilistic_warp import ProbabilisticMultiheadWarp
from ._recurrent import (
    AbstractRecurrentCell,
    AffineRecurrence,
    RecurrentBatch,
    RecurrentResult,
    run_affine_recurrence,
    run_recurrent,
)
from ._recurrent_cells import GRUCell, LSTMCell, RNNCell, StackedRecurrentCell
from ._selective_sequence import (
    ResetAwareCausalConv1D,
    SelectiveStateSpaceBlock,
    SelectiveStateSpaceState,
)
from ._sine import SineLayer
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
from ._weight_space_recurrence import (
    WeightSpaceRecurrence,
    WeightSpaceState,
)


__all__ = [
    "AdaptiveResidual",
    "AbstractRecurrentCell",
    "AffineRecurrence",
    "AttentionExecution",
    "CausalRecurrentConfig",
    "CausalRecurrentDiagnostics",
    "CausalRecurrentFailurePolicy",
    "CausalRecurrentResult",
    "AttentionKernel",
    "MeasureAwareAttention",
    "MeasureNormalizedConvND",
    "ExplicitFourierFeatureEmbeddings",
    "HybridFourierFeatureEmbeddings",
    "MultiscaleFourierFeatureEmbeddings",
    "RandomFourierFeatureEmbeddings",
    "TrainableFourierFeatureEmbeddings",
    "FourierEvaluationMethod",
    "GRUCell",
    "sample_fourier_grid",
    "SinusoidalTimeEmbedding",
    "TimeConditionedVectorModel",
    "Dropout",
    "ComplexLinear",
    "InterfaceDistanceSemantics",
    "InterfaceFeatureLift",
    "Linear",
    "LowRankComplexLinear",
    "LowRankComplexLinearInitializationReport",
    "LinearRecurrentUnit",
    "MultiheadWarp",
    "ProbabilisticMultiheadWarp",
    "ManifoldMultiheadWarp",
    "ManifoldWarpDiagnostics",
    "sphere_retraction",
    "LSTMCell",
    "sphere_tangent_projection",
    "RectilinearWarpDiagnostics",
    "RecurrentBatch",
    "RecurrentResult",
    "ResetAwareCausalConv1D",
    "RNNCell",
    "StackedRecurrentCell",
    "SineLayer",
    "SelectiveStateSpaceBlock",
    "SelectiveStateSpaceState",
    "WeightSpaceRecurrence",
    "WeightSpaceState",
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
    "run_affine_recurrence",
    "run_causal_recurrent",
    "run_recurrent",
]
