"""Model wrappers and composition utilities."""

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
    SpectralBasisModulation,
)
from ._magnitude_direction import MagnitudeDirectionModel
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
    "ComplexOutputModel",
    "ConcatenatedModel",
    "DecayAggregation",
    "DifferentialFieldDecoder",
    "DifferentialNormalization",
    "EquinoxModel",
    "ExponentialSpectralEnvelope",
    "ImplicitModalField",
    "EquinoxStructuredModel",
    "LatentContractionModel",
    "LatentExecutionPolicy",
    "LinearDifferentialTransform",
    "MagnitudeDirectionModel",
    "MaskedSeriesPoolingModel",
    "RaggedSeriesBatchInput",
    "RaggedSeriesModel",
    "Separable",
    "SpectralBasisModulation",
    "Sequential",
]
