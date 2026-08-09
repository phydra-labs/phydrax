"""Internal model-evaluation contracts shared across Phydrax subsystems."""

from ._array import AbstractArrayModel
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
from ._frozen import FrozenModel
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
from ._structure import (
    deserialise_model_leaf,
    model_from_structure_recipe,
    model_structure_recipe,
    serialise_model_leaf,
)


__all__ = [
    "AbstractArrayModel",
    "AxisModelEvaluator",
    "artifact_value",
    "artifact_value_id",
    "deserialise_model_leaf",
    "OperatorArchitectureCodec",
    "operator_architecture_codec",
    "operator_architecture_codec_for",
    "register_artifact_value",
    "register_operator_architecture_codec",
    "iter_model_objective_providers",
    "KFACAffineBlock",
    "FrozenModel",
    "KFACLayoutProvider",
    "ModelBatchMode",
    "ModelBinding",
    "ModelEvaluator",
    "ModelObjectiveProvider",
    "model_objective_labels",
    "model_from_structure_recipe",
    "model_structure_recipe",
    "model_objective_values",
    "ModelInputMode",
    "serialise_model_leaf",
    "StructuredDerivativeProvider",
]
