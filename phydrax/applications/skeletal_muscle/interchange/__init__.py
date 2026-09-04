#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict external skeletal-model identity and channel interchange."""

from ._descriptor import (
    ExternalModelAsset,
    ExternalModelChannelBinding,
    ExternalModelDescriptor,
    ExternalModelDimensionalContract,
    ExternalModelHostInventory,
    ExternalModelPreparationError,
    ExternalModelPreparationEvidence,
    ExternalModelQuantity,
    ExternalModelSource,
    ExternalModelTransformation,
    prepare_external_model_descriptor,
    PreparedExternalModelDescriptor,
)


__all__ = [
    "ExternalModelAsset",
    "ExternalModelChannelBinding",
    "ExternalModelDescriptor",
    "ExternalModelDimensionalContract",
    "ExternalModelHostInventory",
    "ExternalModelPreparationError",
    "ExternalModelPreparationEvidence",
    "ExternalModelQuantity",
    "ExternalModelSource",
    "ExternalModelTransformation",
    "PreparedExternalModelDescriptor",
    "prepare_external_model_descriptor",
]
