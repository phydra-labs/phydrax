"""Pointwise, separable, and process neural models."""

from ..._model import ModelBinding
from .._loss import add_model_loss, ModelWithLoss
from . import wrappers
from ._feynmann import FeynmaNN
from ._flowjax_process import (
    conditional_coupling_flow_process,
    FlowJAXProcessDistribution,
    IdentityCoefficientTransition,
    LatentFlowJAXCoefficientProcess,
    StateTimeProcessConditioner,
)
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
)
from ._kan_capacity import (
    coarsen_kan_edges,
    KANCapacityAdaptationReport,
    refine_kan_edges,
)
from ._mlp import MLP
from ._modified_mlp import ModifiedMLP
from ._separable_feynmann import SeparableFeynmaNN
from ._separable_kan import SeparableKAN
from ._separable_mlp import SeparableMLP
from ._separable_modified_mlp import SeparableModifiedMLP
from .wrappers import (
    ComplexOutputModel,
    ConcatenatedModel,
    DifferentialFieldDecoder,
    DifferentialNormalization,
    EquinoxModel,
    EquinoxStructuredModel,
    LatentContractionModel,
    LatentExecutionPolicy,
    LinearDifferentialTransform,
    MagnitudeDirectionModel,
    MaskedSeriesPoolingModel,
    RaggedSeriesBatchInput,
    RaggedSeriesModel,
    Separable,
    Sequential,
)


__all__ = [
    "AbstractEdgeBasis",
    "BSplineEdgeBasis",
    "BSplineGrid",
    "BSplineGridBank",
    "ComplexOutputModel",
    "ConcatenatedModel",
    "DifferentialFieldDecoder",
    "DifferentialNormalization",
    "EquinoxModel",
    "EquinoxStructuredModel",
    "FeynmaNN",
    "FlowJAXProcessDistribution",
    "IdentityCoefficientTransition",
    "KAN",
    "KANCapacityAdaptationReport",
    "KANEdgeBlock",
    "KANGridAdaptationPlan",
    "KANGridAdaptationReport",
    "LatentContractionModel",
    "LatentExecutionPolicy",
    "LatentFlowJAXCoefficientProcess",
    "LinearDifferentialTransform",
    "MLP",
    "MagnitudeDirectionModel",
    "MaskedSeriesPoolingModel",
    "ModelWithLoss",
    "ModifiedMLP",
    "ModelBinding",
    "OrthogonalPolynomialEdgeBasis",
    "RaggedSeriesBatchInput",
    "RaggedSeriesModel",
    "RationalBSplineEdgeBasis",
    "RationalBSplineEdgeParameters",
    "Separable",
    "SeparableFeynmaNN",
    "SeparableKAN",
    "SeparableMLP",
    "SeparableModifiedMLP",
    "Sequential",
    "StateTimeProcessConditioner",
    "TrainableBSplineGrid",
    "adapt_kan_grids",
    "add_model_loss",
    "coarsen_kan_edges",
    "conditional_coupling_flow_process",
    "refine_kan_edges",
    "wrappers",
]
