"""Adapters between operator contracts and external or pointwise runtimes."""

from ...._model import (
    OperatorArchitectureCodec,
    register_operator_architecture_codec,
)
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


register_operator_architecture_codec(
    OperatorArchitectureCodec(
        "phydrax.operator.architecture:ExternalOperatorAdapter@1",
        ExternalOperatorAdapter,
    )
)


__all__ = [
    "ExternalOperatorAdapter",
    "OperatorCheckpointManifest",
    "OperatorContextModel",
    "bind_operator_context",
    "checkpoint_sha256",
    "load_external_operator_adapter",
    "load_operator_manifest",
    "save_operator_manifest",
    "verify_operator_checkpoint",
]
