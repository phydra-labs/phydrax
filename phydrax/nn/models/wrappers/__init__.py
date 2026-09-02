"""Model wrappers and composition utilities."""

from ._causal_coordinate import (
    CausalCoordinateNetwork,
    CausalCoordinatePlan,
    CausalCoordinateResult,
)
from ._complex_output import ComplexOutputModel
from ._concatenated import ConcatenatedModel
from ._differential_output import (
    DifferentialFieldDecoder,
    DifferentialNormalization,
    LinearDifferentialTransform,
)
from ._equinox import EquinoxModel, EquinoxStructuredModel
from ._implicit_modal import (
    DecayAggregation,
    ExponentialSpectralEnvelope,
    ImplicitModalField,
    SparseImplicitModalField,
    SpectralBasisModulation,
)
from ._magnitude_direction import MagnitudeDirectionModel
from ._onsager import (
    AutoencodedOnsagerDiagnostics,
    AutoencodedOnsagerModel,
    FixedSubspaceOnsagerModel,
    FixedSubspaceProjectionReport,
    PortHamiltonianResidualClosure,
)
from ._ragged_series import (
    MaskedSeriesPoolingModel,
    RaggedSeriesBatchInput,
    RaggedSeriesModel,
)
from ._separable_wrappers import (
    LatentContractionModel,
    LatentExecutionPolicy,
    Separable,
)
from ._sequential import Sequential


__all__ = [
    "AutoencodedOnsagerDiagnostics",
    "AutoencodedOnsagerModel",
    "CausalCoordinateNetwork",
    "CausalCoordinatePlan",
    "CausalCoordinateResult",
    "ComplexOutputModel",
    "ConcatenatedModel",
    "DecayAggregation",
    "DifferentialFieldDecoder",
    "DifferentialNormalization",
    "EquinoxModel",
    "ExponentialSpectralEnvelope",
    "ImplicitModalField",
    "EquinoxStructuredModel",
    "FixedSubspaceOnsagerModel",
    "FixedSubspaceProjectionReport",
    "LatentContractionModel",
    "LatentExecutionPolicy",
    "LinearDifferentialTransform",
    "MagnitudeDirectionModel",
    "MaskedSeriesPoolingModel",
    "RaggedSeriesBatchInput",
    "RaggedSeriesModel",
    "PortHamiltonianResidualClosure",
    "SparseImplicitModalField",
    "Separable",
    "SpectralBasisModulation",
    "Sequential",
]
