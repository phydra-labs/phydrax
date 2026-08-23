#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._precision import (
    complex_precision_dtype,
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState


def _real_component(value: Any, /) -> str:
    name = precision_dtype_name(value)
    if name == "complex64":
        return "float32"
    if name == "complex128":
        return "float64"
    return real_precision_dtype_name(name)


def _effective_dtype(requested: str | None, observed: str, /) -> str:
    if requested is None:
        return observed
    if observed in ("complex64", "complex128"):
        return complex_precision_dtype(requested)
    return requested


class HermitianPrecisionPolicy(StrictModule, NonTrainableState):
    """Hermitian compute, factorization, reduction, decision, and output precision."""

    compute_dtype: str | None = eqx.field(static=True)
    factorization_dtype: str | None = eqx.field(static=True)
    accumulation_dtype: str | None = eqx.field(static=True)
    decision_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        compute_dtype: Any | None = None,
        factorization_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        decision_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ):
        compute = (
            None if compute_dtype is None else real_precision_dtype_name(compute_dtype)
        )
        factorization = (
            None
            if factorization_dtype is None
            else real_precision_dtype_name(factorization_dtype)
        )
        accumulation = (
            None
            if accumulation_dtype is None
            else real_precision_dtype_name(accumulation_dtype)
        )
        decision = (
            None if decision_dtype is None else real_precision_dtype_name(decision_dtype)
        )
        output = None if output_dtype is None else real_precision_dtype_name(output_dtype)
        requested = tuple(
            value
            for value in (compute, factorization, accumulation, decision, output)
            if value is not None
        )
        if any(value not in ("float32", "float64") for value in requested):
            raise ValueError("Hermitian precision supports float32 and float64 bases.")
        if (
            accumulation is not None
            and factorization is not None
            and precision_itemsize(accumulation) < precision_itemsize(factorization)
        ):
            raise ValueError(
                "Hermitian accumulation precision cannot be narrower than factorization."
            )
        if (
            decision is not None
            and accumulation is not None
            and precision_itemsize(decision) < precision_itemsize(accumulation)
        ):
            raise ValueError(
                "Hermitian decision precision cannot be narrower than accumulation."
            )
        request = PrecisionRequest(
            "hermitian-spectral",
            {
                "compute": compute,
                "factorization": factorization,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        self.compute_dtype = compute
        self.factorization_dtype = factorization
        self.accumulation_dtype = accumulation
        self.decision_dtype = decision
        self.output_dtype = output
        self.policy_id = request.request_id

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "hermitian-spectral",
            {
                "compute": self.compute_dtype,
                "factorization": self.factorization_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.decision_dtype,
                "output": self.output_dtype,
            },
        )

    def _cast(self, value: Any, requested: str | None, /):
        array = jnp.asarray(value)
        if requested is None or not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        observed = precision_dtype_name(array.dtype)
        return array.astype(_effective_dtype(requested, observed))

    def compute(self, value: Any, /):
        return self._cast(value, self.compute_dtype)

    def factorization(self, value: Any, /):
        return self._cast(value, self.factorization_dtype)

    def accumulation(self, value: Any, /):
        return self._cast(value, self.accumulation_dtype)

    def decision(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.decision_dtype is None else array.astype(self.decision_dtype)

    def output(self, value: Any, /):
        return self._cast(value, self.output_dtype)

    def evidence_for(self, value: Any, /) -> PrecisionEvidenceEnvelope:
        observed = precision_dtype_name(jnp.asarray(value).dtype)
        compute = _effective_dtype(self.compute_dtype, observed)
        factorization = _effective_dtype(self.factorization_dtype, compute)
        accumulation = _effective_dtype(self.accumulation_dtype, factorization)
        decision = (
            _real_component(accumulation)
            if self.decision_dtype is None
            else self.decision_dtype
        )
        output = _effective_dtype(self.output_dtype, observed)
        resolution = PrecisionResolution(
            self.request,
            "phydrax-hermitian-spectral",
            {
                "compute": compute,
                "factorization": factorization,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


__all__ = ["HermitianPrecisionPolicy"]
