#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Neural-operator architecture implementations."""

from ...._model import (
    OperatorArchitectureCodec,
    register_operator_architecture_codec,
)
from .attention._codano import CoDABlock, CoDANO, CoDAOperatorState
from .attention._dpot import DPOT, dpot_corrupt_history
from .attention._gaot import GAOT
from .attention._gnot import GNOT
from .attention._in_context import (
    InContextOperator,
    InContextOperatorState,
    OperatorPromptState,
)
from .attention._poseidon import Poseidon
from .attention._transolver import Transolver
from .attention._upt import ABUPT, LatentTokenBlock, LatentTokenProcessor, UPT
from .conditioning._deeponet import (
    AbstractBasisTrunk,
    AbstractBranchEncoder,
    DeepONet,
    FixedBranchEncoder,
    IntegralBranchEncoder,
    PODBasis,
)
from .conditioning._equation_conditioning import (
    attach_pde_condition,
    PDEConditionEncoder,
)
from .conditioning._function_frame import (
    FUNCTION_PROJECTION_INSUFFICIENT_SUPPORT,
    FUNCTION_PROJECTION_INVALID_MEASURE,
    FUNCTION_PROJECTION_NONFINITE,
    FUNCTION_PROJECTION_RANK_DEFICIENT,
    FUNCTION_PROJECTION_REGULARIZED,
    FUNCTION_PROJECTION_SUCCESS,
    FunctionFrameEncoding,
    FunctionFrameReconstructor,
    FunctionProjectionPolicy,
    FunctionProjectionRankPolicy,
    FunctionProjectionReport,
    LearnedFunctionFrame,
    ProjectionBranchEncoder,
)
from .conditioning._holomorphic_deeponet import (
    ConditionalHarmonicOperator2D,
    ConditionalHolomorphicDeepONet,
    ConditionalHolomorphicMapCertificate,
    HolomorphicBasisTrunk,
    HolomorphicTrunkMode,
    TargetAugmentedBranchEncoder,
)
from .conditioning._nonlinear_decoder import (
    CoordinateConditionedOperator,
    CoordinateDecoderState,
    FiLMCoordinateDecoder,
)
from .conditioning._pde_conditioned import (
    PDEConditionedInput,
    PDEConditionedOperator,
)
from .dynamics._diagonal_state_space_mixer import DiagonalStateSpaceMixer
from .dynamics._flower import (
    Flower,
    FlowerDiagnostics,
    FlowerQueryMode,
    FlowerTransitionMode,
)
from .dynamics._koopman import KoopmanTemporalOperator
from .dynamics._linear_recurrent_operator import LinearRecurrentOperator
from .dynamics._selective_state_space_mixer import (
    SelectiveStateSpaceDiagnostics,
    SelectiveStateSpaceMixer,
)
from .dynamics._weight_space import WeightSpaceOperator
from .geometric._cochain_neural_operator import (
    CochainNeuralOperator,
    TopologicalCochainBlock,
    TopologicalRouteConfig,
)
from .geometric._equivariant_geometry import EqGINO, EquivariantGeometryOperator
from .geometric._geometry_informed_flower import GeometryInformedFlower
from .geometric._geometry_operator import GeometryOperatorDiagnostics
from .geometric._gino import GINO
from .geometric._green_kernel import GreenKernelOperator
from .geometric._lattice_equivariant_cno import LatticeEquivariantCNO
from .geometric._local_operator import (
    LocalDifferentialOperator,
    LocalGlobalOperator,
    LocalIntegralOperator,
)
from .geometric._native_graph import NativeGraphOperator
from .geometric._rigno import RIGNO
from .probabilistic._flowjax_operator import (
    conditional_coupling_flow_operator,
    ConditionalFlowFunctionOperator,
    FlowJAXOperatorDistribution,
    OperatorBatchConditioner,
)
from .probabilistic._probabilistic_operator import (
    gaussian_operator_nll,
    GaussianFunctionOperator,
)
from .spectral._cno import AntiAliasedConvND, CNO, UNO
from .spectral._fno import (
    AxialFactorizedFNO,
    FNO,
    IFNO,
    IFNOConvergence,
    MultiScaleSpectralConvND,
    spectral_resample,
    SpectralConvND,
)
from .spectral._hofno import HOFNO
from .spectral._laplace import LaplaceTemporalOperator
from .spectral._manifold_spectral import ManifoldSpectralOperator
from .spectral._sfno import SFNO, SphericalSpectralConv
from .spectral._wavelet import MultiwaveletOperator, WaveletNeuralOperator


_PORTABLE_ARCHITECTURES = (
    ("ABUPT", ABUPT),
    ("AxialFactorizedFNO", AxialFactorizedFNO),
    ("CNO", CNO),
    ("CochainNeuralOperator", CochainNeuralOperator),
    ("CoDANO", CoDANO),
    ("ConditionalFlowFunctionOperator", ConditionalFlowFunctionOperator),
    ("CoordinateConditionedOperator", CoordinateConditionedOperator),
    ("DeepONet", DeepONet),
    ("DPOT", DPOT),
    ("DiagonalStateSpaceMixer", DiagonalStateSpaceMixer),
    ("EquivariantGeometryOperator", EquivariantGeometryOperator),
    ("Flower", Flower),
    ("FNO", FNO),
    ("FunctionFrameReconstructor", FunctionFrameReconstructor),
    ("GAOT", GAOT),
    ("GaussianFunctionOperator", GaussianFunctionOperator),
    ("GeometryInformedFlower", GeometryInformedFlower),
    ("GINO", GINO),
    ("GNOT", GNOT),
    ("GreenKernelOperator", GreenKernelOperator),
    ("HOFNO", HOFNO),
    ("IFNO", IFNO),
    ("InContextOperator", InContextOperator),
    ("KoopmanTemporalOperator", KoopmanTemporalOperator),
    ("LatticeEquivariantCNO", LatticeEquivariantCNO),
    ("LaplaceTemporalOperator", LaplaceTemporalOperator),
    ("LocalDifferentialOperator", LocalDifferentialOperator),
    ("LocalGlobalOperator", LocalGlobalOperator),
    ("LocalIntegralOperator", LocalIntegralOperator),
    ("ManifoldSpectralOperator", ManifoldSpectralOperator),
    ("MultiwaveletOperator", MultiwaveletOperator),
    ("NativeGraphOperator", NativeGraphOperator),
    ("PDEConditionedOperator", PDEConditionedOperator),
    ("Poseidon", Poseidon),
    ("RIGNO", RIGNO),
    ("SelectiveStateSpaceMixer", SelectiveStateSpaceMixer),
    ("SFNO", SFNO),
    ("Transolver", Transolver),
    ("UNO", UNO),
    ("UPT", UPT),
    ("WaveletNeuralOperator", WaveletNeuralOperator),
)

for _architecture_name, _architecture_type in _PORTABLE_ARCHITECTURES:
    register_operator_architecture_codec(
        OperatorArchitectureCodec(
            f"phydrax.operator.architecture:{_architecture_name}",
            _architecture_type,
        )
    )

del _architecture_name, _architecture_type


__all__ = [
    "AntiAliasedConvND",
    "CNO",
    "UNO",
    "CochainNeuralOperator",
    "TopologicalCochainBlock",
    "TopologicalRouteConfig",
    "CoDABlock",
    "CoDANO",
    "AbstractBasisTrunk",
    "AbstractBranchEncoder",
    "CoDAOperatorState",
    "DeepONet",
    "FixedBranchEncoder",
    "IntegralBranchEncoder",
    "PODBasis",
    "FUNCTION_PROJECTION_INSUFFICIENT_SUPPORT",
    "FUNCTION_PROJECTION_INVALID_MEASURE",
    "FUNCTION_PROJECTION_NONFINITE",
    "FUNCTION_PROJECTION_RANK_DEFICIENT",
    "FUNCTION_PROJECTION_REGULARIZED",
    "FUNCTION_PROJECTION_SUCCESS",
    "FunctionFrameEncoding",
    "FunctionFrameReconstructor",
    "FunctionProjectionPolicy",
    "FunctionProjectionRankPolicy",
    "FunctionProjectionReport",
    "LearnedFunctionFrame",
    "ConditionalHarmonicOperator2D",
    "ConditionalHolomorphicDeepONet",
    "ConditionalHolomorphicMapCertificate",
    "HolomorphicBasisTrunk",
    "HolomorphicTrunkMode",
    "TargetAugmentedBranchEncoder",
    "ProjectionBranchEncoder",
    "DPOT",
    "DiagonalStateSpaceMixer",
    "dpot_corrupt_history",
    "attach_pde_condition",
    "PDEConditionEncoder",
    "EqGINO",
    "EquivariantGeometryOperator",
    "Flower",
    "FlowerDiagnostics",
    "FlowerQueryMode",
    "FlowerTransitionMode",
    "conditional_coupling_flow_operator",
    "ConditionalFlowFunctionOperator",
    "FlowJAXOperatorDistribution",
    "OperatorBatchConditioner",
    "AxialFactorizedFNO",
    "FNO",
    "IFNO",
    "IFNOConvergence",
    "MultiScaleSpectralConvND",
    "spectral_resample",
    "SpectralConvND",
    "GAOT",
    "GeometryInformedFlower",
    "GeometryOperatorDiagnostics",
    "GINO",
    "GNOT",
    "GreenKernelOperator",
    "HOFNO",
    "InContextOperator",
    "InContextOperatorState",
    "OperatorPromptState",
    "KoopmanTemporalOperator",
    "LaplaceTemporalOperator",
    "LatticeEquivariantCNO",
    "LocalDifferentialOperator",
    "LocalGlobalOperator",
    "LinearRecurrentOperator",
    "LocalIntegralOperator",
    "ManifoldSpectralOperator",
    "NativeGraphOperator",
    "CoordinateConditionedOperator",
    "CoordinateDecoderState",
    "FiLMCoordinateDecoder",
    "PDEConditionedInput",
    "PDEConditionedOperator",
    "Poseidon",
    "gaussian_operator_nll",
    "GaussianFunctionOperator",
    "RIGNO",
    "SelectiveStateSpaceDiagnostics",
    "SelectiveStateSpaceMixer",
    "SFNO",
    "WeightSpaceOperator",
    "SphericalSpectralConv",
    "Transolver",
    "ABUPT",
    "LatentTokenBlock",
    "LatentTokenProcessor",
    "UPT",
    "MultiwaveletOperator",
    "WaveletNeuralOperator",
]
