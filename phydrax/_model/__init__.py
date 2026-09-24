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
from ._ports import (
    ModelPorts,
    PortBindingEvidence,
    PortMapping,
    PortProvider,
    PortVariance,
    resolve_port_mapping,
    ValuePort,
)
from ._protocols import (
    AxisModelEvaluator,
    INPUT_CONVEX_CERTIFICATE_KEY,
    MODEL_CONSTRUCTION_CERTIFICATE_KEYS,
    ModelEvaluator,
    ModelMetadataProvider,
    StructuredDerivativeProvider,
    TRIAL_SPACE_CERTIFICATE_KEY,
)
from ._structure import (
    deserialize_model_leaf,
    model_from_structure_recipe,
    model_structure_recipe,
    serialize_model_leaf,
)


__all__ = [
    "AbstractArrayModel",
    "AxisModelEvaluator",
    "artifact_value",
    "artifact_value_id",
    "deserialize_model_leaf",
    "OperatorArchitectureCodec",
    "operator_architecture_codec",
    "operator_architecture_codec_for",
    "register_artifact_value",
    "register_operator_architecture_codec",
    "iter_model_objective_providers",
    "INPUT_CONVEX_CERTIFICATE_KEY",
    "KFACAffineBlock",
    "FrozenModel",
    "KFACLayoutProvider",
    "ModelBatchMode",
    "MODEL_CONSTRUCTION_CERTIFICATE_KEYS",
    "ModelBinding",
    "ModelEvaluator",
    "ModelPorts",
    "ModelMetadataProvider",
    "ModelObjectiveProvider",
    "model_objective_labels",
    "model_from_structure_recipe",
    "model_structure_recipe",
    "model_objective_values",
    "ModelInputMode",
    "PortBindingEvidence",
    "PortMapping",
    "PortProvider",
    "PortVariance",
    "resolve_port_mapping",
    "serialize_model_leaf",
    "StructuredDerivativeProvider",
    "TRIAL_SPACE_CERTIFICATE_KEY",
    "ValuePort",
]
