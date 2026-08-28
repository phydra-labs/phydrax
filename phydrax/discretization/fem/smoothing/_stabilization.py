#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


SmoothingStabilizationKind = Literal[
    "none",
    "compatible-blend",
    "projected-gradient",
    "rank-complement",
    "selective-volumetric",
]


class SmoothingStabilizationPolicy(StrictModule, NonTrainableState):
    kind: SmoothingStabilizationKind = eqx.field(static=True)
    parameter: float = eqx.field(static=True)
    preserves_rigid_modes: bool = eqx.field(static=True)
    preserves_affine_fields: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SmoothingStabilizationKind = "none",
        /,
        *,
        parameter: float = 0.0,
        preserves_rigid_modes: bool = True,
        preserves_affine_fields: bool = True,
    ):
        if kind not in (
            "none",
            "compatible-blend",
            "projected-gradient",
            "rank-complement",
            "selective-volumetric",
        ):
            raise ValueError("Unknown smoothing stabilization kind.")
        parameter_ = float(parameter)
        if parameter_ < 0.0 or (kind != "none" and parameter_ == 0.0):
            raise ValueError("Active stabilization requires a positive parameter.")
        self.kind = kind
        self.parameter = parameter_
        self.preserves_rigid_modes = bool(preserves_rigid_modes)
        self.preserves_affine_fields = bool(preserves_affine_fields)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "smoothing-stabilization-policy",
                "stabilization": kind,
                "parameter": parameter_,
                "preserves_rigid_modes": bool(preserves_rigid_modes),
                "preserves_affine_fields": bool(preserves_affine_fields),
            }
        )

    def apply(
        self,
        smoothed: ArrayLike,
        compatible: ArrayLike,
        /,
        *,
        complement_projector: ArrayLike | None = None,
    ) -> Array:
        smoothed_ = jnp.asarray(smoothed)
        compatible_ = jnp.asarray(compatible)
        if smoothed_.shape != compatible_.shape:
            raise ValueError("Smoothed and compatible local tensors must match.")
        if self.kind == "none":
            return smoothed_
        if self.kind in ("compatible-blend", "selective-volumetric"):
            return smoothed_ + self.parameter * (compatible_ - smoothed_)
        if complement_projector is None:
            raise ValueError("Projected/rank-complement stabilization needs a projector.")
        projector = jnp.asarray(complement_projector)
        if projector.shape[-2:] != smoothed_.shape[-2:]:
            raise ValueError("Stabilization projector has incompatible matrix shape.")
        return smoothed_ + self.parameter * projector


__all__ = ["SmoothingStabilizationKind", "SmoothingStabilizationPolicy"]
