"""Pointwise, separable, and process neural models."""

from ..._model import ModelBinding
from .._loss import add_model_loss, ModelWithLoss
from . import wrappers
from ._algebra_analytic import (
    AlgebraAnalyticLayer,
    AlgebraAnalyticNetwork,
    AnalyticityEvidence,
    AnalyticityKind,
    AnalyticityOperator,
)
from ._constitutive import DeformationGradientMinors, PolyconvexPotential
from ._constrained_constitutive import (
    CoercivePolyconvexEnvelope,
    ConstrainedPolyconvexPotential,
    MaterialConstraintReport,
    PolyconvexMaterialConstraints,
    ReferenceConfiguration,
)
from ._feynmann import FeynmaNN
from ._flowjax_process import (
    conditional_coupling_flow_process,
    FlowJAXProcessDistribution,
    IdentityCoefficientTransition,
    LatentFlowJAXCoefficientProcess,
    StateTimeProcessConditioner,
)
from ._holomorphic import HolomorphicMLP
from ._input_convex import InputConvexNetwork, PartiallyInputConvexNetwork
from ._kan import KAN, KANEdgeBlock
from ._kan_adaptation import (
    adapt_kan_grids,
    KANGridAdaptationPlan,
    KANGridAdaptationReport,
)
from ._kan_basis import (
    AbstractEdgeBasis,
    BSplineEdgeBasis,
    BSplineGrid,
    BSplineGridBank,
    OrthogonalPolynomialEdgeBasis,
    RationalBSplineEdgeBasis,
    RationalBSplineEdgeParameters,
    TrainableBSplineGrid,
    TrainableBSplineGridBank,
)
from ._kan_capacity import (
    coarsen_kan_edges,
    KANCapacityAdaptationReport,
    refine_kan_edges,
)
from ._linear_recurrent import LinearRecurrentModel
from ._mlp import MLP
from ._modified_mlp import ModifiedMLP
from ._ordinal import OrdinalCumulativeLinkHead
from ._piratenet import PirateNet
from ._port_hamiltonian import (
    DissipationStructure,
    FeatureNormPotential,
    PortHamiltonianVectorField,
)
from ._projective_potential import ProjectiveInvariantPotential
from ._recurrent import (
    BidirectionalRecurrentSequenceModel,
    RecurrentSequenceModel,
)
from ._selective_sequence import SelectiveSequenceModel
from ._separable_feynmann import SeparableFeynmaNN
from ._separable_kan import SeparableKAN
from ._separable_mlp import SeparableMLP
from ._separable_modified_mlp import SeparableModifiedMLP
from ._siren import SIREN
from ._weight_space_recurrent import (
    FunctionalStateDecoder,
    WeightSpaceRecurrentModel,
)
from .wrappers import (
    AutoencodedOnsagerDiagnostics,
    AutoencodedOnsagerModel,
    CausalCoordinateNetwork,
    CausalCoordinatePlan,
    CausalCoordinateResult,
    ComplexOutputModel,
    ConcatenatedModel,
    DecayAggregation,
    DifferentialFieldDecoder,
    DifferentialNormalization,
    EquinoxModel,
    EquinoxStructuredModel,
    ExponentialSpectralEnvelope,
    FixedSubspaceOnsagerModel,
    FixedSubspaceProjectionReport,
    ImplicitModalField,
    LatentContractionModel,
    LatentExecutionPolicy,
    LinearDifferentialTransform,
    MagnitudeDirectionModel,
    MaskedSeriesPoolingModel,
    PortHamiltonianResidualClosure,
    RaggedSeriesBatchInput,
    RaggedSeriesModel,
    Separable,
    Sequential,
    SparseImplicitModalField,
    SpectralBasisModulation,
)


__all__ = [
    "AlgebraAnalyticLayer",
    "AlgebraAnalyticNetwork",
    "AnalyticityEvidence",
    "AnalyticityKind",
    "AnalyticityOperator",
    "ProjectiveInvariantPotential",
    "AbstractEdgeBasis",
    "BSplineEdgeBasis",
    "BSplineGrid",
    "BSplineGridBank",
    "BidirectionalRecurrentSequenceModel",
    "AutoencodedOnsagerDiagnostics",
    "AutoencodedOnsagerModel",
    "CausalCoordinateNetwork",
    "CausalCoordinatePlan",
    "CausalCoordinateResult",
    "ComplexOutputModel",
    "CoercivePolyconvexEnvelope",
    "ConstrainedPolyconvexPotential",
    "DeformationGradientMinors",
    "ConcatenatedModel",
    "DecayAggregation",
    "DifferentialFieldDecoder",
    "DissipationStructure",
    "DifferentialNormalization",
    "EquinoxModel",
    "EquinoxStructuredModel",
    "FeynmaNN",
    "FlowJAXProcessDistribution",
    "FeatureNormPotential",
    "ExponentialSpectralEnvelope",
    "FunctionalStateDecoder",
    "FixedSubspaceOnsagerModel",
    "FixedSubspaceProjectionReport",
    "IdentityCoefficientTransition",
    "InputConvexNetwork",
    "KAN",
    "HolomorphicMLP",
    "ImplicitModalField",
    "SparseImplicitModalField",
    "OrdinalCumulativeLinkHead",
    "KANCapacityAdaptationReport",
    "KANEdgeBlock",
    "KANGridAdaptationPlan",
    "KANGridAdaptationReport",
    "LatentContractionModel",
    "LatentExecutionPolicy",
    "LatentFlowJAXCoefficientProcess",
    "LinearDifferentialTransform",
    "LinearRecurrentModel",
    "MLP",
    "MagnitudeDirectionModel",
    "MaskedSeriesPoolingModel",
    "ModelWithLoss",
    "MaterialConstraintReport",
    "ModifiedMLP",
    "ModelBinding",
    "PirateNet",
    "PartiallyInputConvexNetwork",
    "PolyconvexPotential",
    "PolyconvexMaterialConstraints",
    "PortHamiltonianVectorField",
    "PortHamiltonianResidualClosure",
    "OrthogonalPolynomialEdgeBasis",
    "RaggedSeriesBatchInput",
    "ReferenceConfiguration",
    "RaggedSeriesModel",
    "RationalBSplineEdgeBasis",
    "RationalBSplineEdgeParameters",
    "RecurrentSequenceModel",
    "SelectiveSequenceModel",
    "Separable",
    "SeparableFeynmaNN",
    "SeparableKAN",
    "SeparableMLP",
    "SeparableModifiedMLP",
    "SIREN",
    "SpectralBasisModulation",
    "Sequential",
    "StateTimeProcessConditioner",
    "TrainableBSplineGrid",
    "TrainableBSplineGridBank",
    "WeightSpaceRecurrentModel",
    "adapt_kan_grids",
    "add_model_loss",
    "coarsen_kan_edges",
    "conditional_coupling_flow_process",
    "refine_kan_edges",
    "wrappers",
]
