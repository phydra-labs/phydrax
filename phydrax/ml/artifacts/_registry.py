#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib

from ..._model import FrozenModel, register_artifact_value
from ...units._dimension import DimensionSignature
from .._schema import FeatureSchema, TargetSchema


_ARTIFACT_FAMILIES = (
    "preprocessing",
    "compose",
    "linear",
    "covariance",
    "discriminant",
    "naive_bayes",
    "kernel_methods",
    "neighbors",
    "decomposition",
    "manifold",
    "clustering",
    "mixture",
    "outliers",
    "tree",
    "ensemble",
    "feature_selection",
    "multiclass",
    "quantum",
    "semi_supervised",
    "calibration",
)
_REGISTERED = False


def register_native_ml_artifacts() -> None:
    """Register stable public identities before saving or loading native ML models."""
    global _REGISTERED
    if _REGISTERED:
        return
    register_artifact_value("phydrax.ml.core:FrozenModel", FrozenModel)
    register_artifact_value("phydrax.ml.core:FeatureSchema", FeatureSchema)
    register_artifact_value("phydrax.ml.core:TargetSchema", TargetSchema)
    register_artifact_value("phydrax.units:DimensionSignature", DimensionSignature)
    for family in _ARTIFACT_FAMILIES:
        module = importlib.import_module(f"phydrax.ml.{family}")
        namespace = vars(module)
        exports = namespace.get("__all__", ())
        for name in exports:
            value = namespace[name]
            if not isinstance(value, type) or not value.__module__.startswith(
                f"phydrax.ml.{family}."
            ):
                continue
            register_artifact_value(
                f"phydrax.ml.{family}:{name}",
                value,
            )
    kernel_module = importlib.import_module("phydrax.kernels")
    for name in vars(kernel_module).get("__all__", ()):
        value = vars(kernel_module)[name]
        if isinstance(value, type) and value.__module__.startswith("phydrax.kernels."):
            register_artifact_value(
                f"phydrax.kernels:{name}",
                value,
            )
    _REGISTERED = True


__all__ = ["register_native_ml_artifacts"]
