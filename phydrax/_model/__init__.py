"""Internal model-evaluation contracts shared across Phydrax subsystems."""

from ._artifacts import (
    artifact_value,
    artifact_value_id,
    operator_architecture_codec,
    operator_architecture_codec_for,
    OperatorArchitectureCodec,
    register_artifact_value,
    register_operator_architecture_codec,
)
from ._binding import ModelBatchMode, ModelBinding, ModelInputMode
from ._kfac import KFACAffineBlock, KFACLayoutProvider
from ._objectives import (
    iter_model_objective_providers,
    model_objective_labels,
    model_objective_values,
    ModelObjectiveProvider,
)
from ._protocols import (
    AxisModelEvaluator,
    ModelEvaluator,
    StructuredDerivativeProvider,
)
from ._spectral import SpectralDiscretizationProvider


__all__ = [
    "AxisModelEvaluator",
    "artifact_value",
    "artifact_value_id",
    "OperatorArchitectureCodec",
    "operator_architecture_codec",
    "operator_architecture_codec_for",
    "register_artifact_value",
    "register_operator_architecture_codec",
    "iter_model_objective_providers",
    "KFACAffineBlock",
    "KFACLayoutProvider",
    "ModelBatchMode",
    "ModelBinding",
    "ModelEvaluator",
    "ModelObjectiveProvider",
    "model_objective_labels",
    "model_objective_values",
    "ModelInputMode",
    "SpectralDiscretizationProvider",
    "StructuredDerivativeProvider",
]
