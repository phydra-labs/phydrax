#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._precision import (
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    PrecisionResourceAssumptions,
    real_precision_dtype_name,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState


_SUPPORTED_FD_DTYPES = frozenset(("float32", "float64", "complex64", "complex128"))


def _real_component_dtype(value: Any, /) -> str:
    name = precision_dtype_name(value)
    if name == "complex64":
        return "float32"
    if name == "complex128":
        return "float64"
    return real_precision_dtype_name(name)


def _is_complex(value: Any, /) -> bool:
    return precision_dtype_name(value) in ("complex64", "complex128")


class FDExecutionPrecisionPolicy(StrictModule, NonTrainableState):
    """Executable coefficient, field, accumulation, and certification dtypes."""

    coefficient_dtype: str = eqx.field(static=True)
    field_dtype: str = eqx.field(static=True)
    accumulation_dtype: str = eqx.field(static=True)
    certification_dtype: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coefficient_dtype: Any = jnp.float64,
        field_dtype: Any = jnp.float64,
        accumulation_dtype: Any | None = None,
        certification_dtype: Any | None = None,
    ):
        coefficient = precision_dtype_name(coefficient_dtype)
        field = precision_dtype_name(field_dtype)
        accumulation = (
            field
            if accumulation_dtype is None
            else precision_dtype_name(accumulation_dtype)
        )
        certification = (
            _real_component_dtype(field)
            if certification_dtype is None
            else real_precision_dtype_name(certification_dtype)
        )
        if any(
            value not in _SUPPORTED_FD_DTYPES
            for value in (coefficient, field, accumulation)
        ):
            raise ValueError(
                "FD execution initially supports float32/float64 and complex64/complex128."
            )
        if _is_complex(coefficient) and not _is_complex(field):
            raise ValueError("Complex FD coefficients require a complex field dtype.")
        if _is_complex(accumulation) != _is_complex(field):
            raise ValueError("FD accumulation and field dtypes must have the same kind.")
        if precision_itemsize(_real_component_dtype(accumulation)) < precision_itemsize(
            _real_component_dtype(field)
        ):
            raise ValueError("FD accumulation precision cannot be narrower than fields.")
        if precision_itemsize(certification) < precision_itemsize(
            _real_component_dtype(field)
        ):
            raise ValueError("FD certification precision cannot be narrower than fields.")
        request = PrecisionRequest(
            "finite-difference",
            {
                "coefficient": coefficient,
                "storage": field,
                "compute": field,
                "accumulation": accumulation,
                "certification": certification,
                "communication": field,
                "checkpoint": field,
                "output": field,
            },
        )
        resolution = PrecisionResolution(request, "phydrax-fd", dict(request.requested))
        self.coefficient_dtype = coefficient
        self.field_dtype = field
        self.accumulation_dtype = accumulation
        self.certification_dtype = certification
        self.policy_id = resolution.resolution_id

    def coefficient(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.coefficient_dtype))

    def field(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.field_dtype))

    def accumulation(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.accumulation_dtype))

    def certification(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.certification_dtype))

    @property
    def resource_assumptions(self) -> PrecisionResourceAssumptions:
        return PrecisionResourceAssumptions(
            "finite-difference",
            {
                "coefficient": self.coefficient_dtype,
                "storage": self.field_dtype,
                "compute": self.field_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.certification_dtype,
                "communication": self.field_dtype,
                "checkpoint": self.field_dtype,
                "output": self.field_dtype,
            },
        )

    @property
    def precision_request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "finite-difference",
            {
                "coefficient": self.coefficient_dtype,
                "storage": self.field_dtype,
                "compute": self.field_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.certification_dtype,
                "communication": self.field_dtype,
                "checkpoint": self.field_dtype,
                "output": self.field_dtype,
            },
        )

    @property
    def precision_resolution(self) -> PrecisionResolution:
        request = self.precision_request
        return PrecisionResolution(request, "phydrax-fd", dict(request.requested))

    def evidence(self) -> PrecisionEvidenceEnvelope:
        resolution = self.precision_resolution
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


__all__ = ["FDExecutionPrecisionPolicy"]
