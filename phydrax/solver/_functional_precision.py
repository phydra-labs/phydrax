#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint
from .._precision import (
    precision_dtype_name,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState


FunctionalMatmulPrecision = Literal[
    "default",
    "high",
    "highest",
    "BF16_BF16_F32",
    "TF32_TF32_F32",
    "F32_F32_F32",
]


class FunctionalPrecisionPolicy(StrictModule, NonTrainableState):
    """Experimental standard-Optax model-evaluation contraction precision."""

    matmul_precision: FunctionalMatmulPrecision = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(self, matmul_precision: FunctionalMatmulPrecision = "default", /):
        if matmul_precision not in (
            "default",
            "high",
            "highest",
            "BF16_BF16_F32",
            "TF32_TF32_F32",
            "F32_F32_F32",
        ):
            raise ValueError("Unsupported FunctionalSolver matmul precision.")
        self.matmul_precision = matmul_precision
        self.policy_id = canonical_fingerprint(
            {
                "kind": "functional-precision-policy",
                "matmul_precision": matmul_precision,
                "scope": "standard-optax-model-evaluation",
            }
        )

    def evidence(self, value_dtype: Any, /) -> PrecisionEvidenceEnvelope:
        dtype = precision_dtype_name(jnp.dtype(value_dtype))
        request = PrecisionRequest(
            "functional-solver",
            {
                "compute": dtype,
                "accumulation": dtype,
                "output": dtype,
            },
        )
        resolution = PrecisionResolution(
            request,
            f"phydrax-functional:{self.matmul_precision}",
            dict(request.requested),
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


__all__ = ["FunctionalMatmulPrecision", "FunctionalPrecisionPolicy"]
