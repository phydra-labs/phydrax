#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._precision import (
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
    RealPrecisionDType,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class FiniteElementPrecisionPolicy(StrictModule, NonTrainableState):
    """Geometry, evaluation, accumulation, and output precision for FE kernels."""

    geometry_dtype: RealPrecisionDType = eqx.field(static=True)
    evaluation_dtype: RealPrecisionDType = eqx.field(static=True)
    accumulation_dtype: RealPrecisionDType = eqx.field(static=True)
    output_dtype: RealPrecisionDType = eqx.field(static=True)
    compensated_accumulation: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_dtype: Any = "float64",
        evaluation_dtype: Any = "float64",
        accumulation_dtype: Any | None = None,
        output_dtype: Any | None = None,
        compensated_accumulation: bool = True,
    ):
        geometry = real_precision_dtype_name(geometry_dtype)
        evaluation = real_precision_dtype_name(evaluation_dtype)
        accumulation = real_precision_dtype_name(
            evaluation if accumulation_dtype is None else accumulation_dtype
        )
        output = real_precision_dtype_name(
            evaluation if output_dtype is None else output_dtype
        )
        self.geometry_dtype = geometry
        self.evaluation_dtype = evaluation
        self.accumulation_dtype = accumulation
        self.output_dtype = output
        self.compensated_accumulation = bool(compensated_accumulation)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-precision-policy",
                "geometry_dtype": geometry,
                "evaluation_dtype": evaluation,
                "accumulation_dtype": accumulation,
                "output_dtype": output,
                "compensated_accumulation": bool(compensated_accumulation),
            }
        )

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "finite-element",
            {
                "storage": self.evaluation_dtype,
                "basis": self.geometry_dtype,
                "compute": self.evaluation_dtype,
                "accumulation": self.accumulation_dtype,
                "output": self.output_dtype,
            },
        )

    def geometry(self, value: Any, /):
        return jnp.asarray(value, dtype=self.geometry_dtype)

    def evaluation(self, value: Any, /):
        return jnp.asarray(value, dtype=self.evaluation_dtype)

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(self.accumulation_dtype)

    def output(self, value: Any, /):
        return jnp.asarray(value, dtype=self.output_dtype)

    def evidence(self) -> PrecisionEvidenceEnvelope:
        resolution = PrecisionResolution(
            self.request,
            "phydrax-finite-element",
            {
                "storage": self.evaluation_dtype,
                "basis": self.geometry_dtype,
                "compute": self.evaluation_dtype,
                "accumulation": self.accumulation_dtype,
                "output": self.output_dtype,
            },
        )
        return PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
        )


__all__ = ["FiniteElementPrecisionPolicy"]
