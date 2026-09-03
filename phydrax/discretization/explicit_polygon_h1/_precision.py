#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
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


class ExplicitPolygonH1PrecisionPolicy(StrictModule, NonTrainableState):
    """Geometry, basis, factorization, accumulation, and certification precision."""

    geometry_dtype: RealPrecisionDType = eqx.field(static=True)
    basis_dtype: RealPrecisionDType = eqx.field(static=True)
    factorization_dtype: RealPrecisionDType = eqx.field(static=True)
    accumulation_dtype: RealPrecisionDType = eqx.field(static=True)
    output_dtype: RealPrecisionDType = eqx.field(static=True)
    certification_dtype: RealPrecisionDType = eqx.field(static=True)
    compensated_accumulation: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_dtype: Any = "float64",
        basis_dtype: Any = "float64",
        factorization_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        output_dtype: Any | None = None,
        certification_dtype: Any = "float64",
        compensated_accumulation: bool = True,
    ):
        geometry = real_precision_dtype_name(geometry_dtype)
        basis = real_precision_dtype_name(basis_dtype)
        factorization = real_precision_dtype_name(
            basis if factorization_dtype is None else factorization_dtype
        )
        accumulation = real_precision_dtype_name(
            basis if accumulation_dtype is None else accumulation_dtype
        )
        output = real_precision_dtype_name(
            basis if output_dtype is None else output_dtype
        )
        certification = real_precision_dtype_name(certification_dtype)
        self.geometry_dtype = geometry
        self.basis_dtype = basis
        self.factorization_dtype = factorization
        self.accumulation_dtype = accumulation
        self.output_dtype = output
        self.certification_dtype = certification
        self.compensated_accumulation = bool(compensated_accumulation)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "explicit-polygon-h1-precision",
                "geometry": geometry,
                "basis": basis,
                "factorization": factorization,
                "accumulation": accumulation,
                "output": output,
                "certification": certification,
                "compensated_accumulation": bool(compensated_accumulation),
            }
        )
        self.validate_backend()

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "explicit-polygon-h1",
            {
                "storage": self.geometry_dtype,
                "basis": self.basis_dtype,
                "compute": self.basis_dtype,
                "factorization": self.factorization_dtype,
                "accumulation": self.accumulation_dtype,
                "output": self.output_dtype,
                "certification": self.certification_dtype,
            },
        )

    def geometry(self, value: Any, /):
        return jnp.asarray(value, dtype=self.geometry_dtype)

    def basis(self, value: Any, /):
        return jnp.asarray(value, dtype=self.basis_dtype)

    def factorization(self, value: Any, /):
        return jnp.asarray(value, dtype=self.factorization_dtype)

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(self.accumulation_dtype)

    def output(self, value: Any, /):
        return jnp.asarray(value, dtype=self.output_dtype)

    def certification(self, value: Any, /):
        return jnp.asarray(value, dtype=self.certification_dtype)

    def evidence(self) -> PrecisionEvidenceEnvelope:
        request = self.request
        resolution = PrecisionResolution(
            request,
            "phydrax-explicit-polygon-h1",
            dict(request.requested),
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))

    def validate_backend(self) -> None:
        if (
            "float64"
            in (
                self.geometry_dtype,
                self.basis_dtype,
                self.factorization_dtype,
                self.accumulation_dtype,
                self.output_dtype,
                self.certification_dtype,
            )
            and not jax.config.x64_enabled
        ):
            raise ValueError("Requested explicit polygon float64 precision is disabled.")


__all__ = ["ExplicitPolygonH1PrecisionPolicy"]
