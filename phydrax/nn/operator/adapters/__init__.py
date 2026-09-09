"""Adapters between operator contracts and external or pointwise runtimes."""

from ...._model import (
    OperatorArchitectureCodec,
    register_operator_architecture_codec,
)
from ._conditional_affine import TrainedChemicalConditionalAffineTransition
from ._context import bind_operator_context, OperatorContextModel
from ._external import (
    checkpoint_sha256,
    ExternalOperatorAdapter,
    load_external_operator_adapter,
    load_operator_manifest,
    OperatorCheckpointManifest,
    save_operator_manifest,
    verify_operator_checkpoint,
)
from ._group_average import GroupAveragedOperator
from ._particle_exchange import (
    LearnedPairwiseExchangeResult,
    PairwiseExchangeBindingPlan,
    PairwiseExchangeFeatureSchema,
    PairwiseExchangeKind,
    particle_pair_operator_batch,
    PreparedPairwiseExchangeBinding,
)


register_operator_architecture_codec(
    OperatorArchitectureCodec(
        "phydrax.operator.architecture:ExternalOperatorAdapter",
        ExternalOperatorAdapter,
    )
)


__all__ = [
    "GroupAveragedOperator",
    "ExternalOperatorAdapter",
    "OperatorCheckpointManifest",
    "OperatorContextModel",
    "LearnedPairwiseExchangeResult",
    "particle_pair_operator_batch",
    "PairwiseExchangeBindingPlan",
    "PairwiseExchangeFeatureSchema",
    "PairwiseExchangeKind",
    "PreparedPairwiseExchangeBinding",
    "bind_operator_context",
    "checkpoint_sha256",
    "load_external_operator_adapter",
    "load_operator_manifest",
    "save_operator_manifest",
    "verify_operator_checkpoint",
    "TrainedChemicalConditionalAffineTransition",
]
