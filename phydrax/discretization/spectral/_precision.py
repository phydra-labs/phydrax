#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._precision import (
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    PrecisionResourceAssumptions,
    real_precision_dtype_name,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_SUPPORTED = frozenset(("float32", "float64", "complex64", "complex128"))


def _dtype(value: Any, /) -> str:
    name = precision_dtype_name(value)
    if name not in _SUPPORTED:
        raise ValueError(
            "Spectral precision supports float32/float64 and complex64/complex128."
        )
    return name


def _real_dtype(value: Any, /) -> str:
    name = _dtype(value)
    if name == "complex64":
        return "float32"
    if name == "complex128":
        return "float64"
    return real_precision_dtype_name(name)


def _complex_dtype(value: Any, /) -> str:
    name = _dtype(value)
    return "complex64" if _real_dtype(name) == "float32" else "complex128"


class SpectralPrecisionPolicy(StrictModule, NonTrainableState):
    """Physical, modal, transform, nonlinear, reduction, and output precision."""

    physical_dtype: str = eqx.field(static=True)
    coefficient_dtype: str = eqx.field(static=True)
    transform_dtype: str = eqx.field(static=True)
    nonlinear_dtype: str = eqx.field(static=True)
    reduction_dtype: str = eqx.field(static=True)
    certification_dtype: str = eqx.field(static=True)
    output_dtype: str = eqx.field(static=True)
    checkpoint_dtype: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        physical_dtype: Any = jnp.float64,
        /,
        *,
        coefficient_dtype: Any | None = None,
        transform_dtype: Any | None = None,
        nonlinear_dtype: Any | None = None,
        reduction_dtype: Any | None = None,
        certification_dtype: Any | None = None,
        output_dtype: Any | None = None,
        checkpoint_dtype: Any | None = None,
    ):
        physical = _dtype(physical_dtype)
        coefficient = _dtype(
            _complex_dtype(physical) if coefficient_dtype is None else coefficient_dtype
        )
        transform = _dtype(coefficient if transform_dtype is None else transform_dtype)
        nonlinear = _dtype(physical if nonlinear_dtype is None else nonlinear_dtype)
        reduction = _real_dtype(physical if reduction_dtype is None else reduction_dtype)
        certification = _real_dtype(
            reduction if certification_dtype is None else certification_dtype
        )
        output = _dtype(physical if output_dtype is None else output_dtype)
        checkpoint = _dtype(coefficient if checkpoint_dtype is None else checkpoint_dtype)
        if not coefficient.startswith("complex"):
            raise ValueError("Spectral coefficient storage must use a complex dtype.")
        if not transform.startswith("complex"):
            raise ValueError("Spectral transform compute must use a complex dtype.")
        physical_real = _real_dtype(physical)
        if precision_itemsize(_real_dtype(coefficient)) < precision_itemsize(
            physical_real
        ):
            raise ValueError(
                "Spectral coefficient precision cannot be narrower than physical storage."
            )
        if precision_itemsize(_real_dtype(transform)) < precision_itemsize(
            _real_dtype(coefficient)
        ):
            raise ValueError(
                "Spectral transform precision cannot be narrower than coefficients."
            )
        if precision_itemsize(_real_dtype(nonlinear)) < precision_itemsize(physical_real):
            raise ValueError(
                "Spectral nonlinear precision cannot be narrower than physical storage."
            )
        if precision_itemsize(reduction) < max(
            precision_itemsize(_real_dtype(transform)),
            precision_itemsize(_real_dtype(nonlinear)),
        ):
            raise ValueError(
                "Spectral reduction precision cannot be narrower than transform or "
                "nonlinear precision."
            )
        if precision_itemsize(certification) < precision_itemsize(reduction):
            raise ValueError(
                "Spectral certification precision cannot be narrower than reductions."
            )
        request = PrecisionRequest(
            "spectral-method",
            {
                "storage": coefficient,
                "coefficient": coefficient,
                "basis": physical,
                "compute": transform,
                "residual": nonlinear,
                "accumulation": reduction,
                "certification": certification,
                "output": output,
                "checkpoint": checkpoint,
            },
        )
        resolution = PrecisionResolution(
            request,
            "phydrax-spectral",
            dict(request.requested),
        )
        self.physical_dtype = physical
        self.coefficient_dtype = coefficient
        self.transform_dtype = transform
        self.nonlinear_dtype = nonlinear
        self.reduction_dtype = reduction
        self.certification_dtype = certification
        self.output_dtype = output
        self.checkpoint_dtype = checkpoint
        self.policy_id = resolution.resolution_id

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "spectral-method",
            {
                "storage": self.coefficient_dtype,
                "coefficient": self.coefficient_dtype,
                "basis": self.physical_dtype,
                "compute": self.transform_dtype,
                "residual": self.nonlinear_dtype,
                "accumulation": self.reduction_dtype,
                "certification": self.certification_dtype,
                "output": self.output_dtype,
                "checkpoint": self.checkpoint_dtype,
            },
        )

    def physical(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.physical_dtype))

    def coefficients(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.coefficient_dtype))

    def transform(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.transform_dtype))

    def nonlinear(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.nonlinear_dtype))

    def reduction(self, value: Any, /):
        array = jnp.asarray(value)
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        dtype = (
            jnp.complex64
            if self.reduction_dtype == "float32" and jnp.iscomplexobj(array)
            else jnp.complex128
            if self.reduction_dtype == "float64" and jnp.iscomplexobj(array)
            else jnp.dtype(self.reduction_dtype)
        )
        return array.astype(dtype)

    def certification(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.certification_dtype))

    def output(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.output_dtype))

    def checkpoint(self, value: Any, /):
        return jnp.asarray(value, dtype=jnp.dtype(self.checkpoint_dtype))

    def evidence(self) -> PrecisionEvidenceEnvelope:
        resolution = PrecisionResolution(
            self.request,
            "phydrax-spectral",
            dict(self.request.requested),
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))

    def resource_assumptions(self) -> PrecisionResourceAssumptions:
        return PrecisionResourceAssumptions(
            "spectral-method",
            dict(self.evidence().observed),
        )


__all__ = ["SpectralPrecisionPolicy"]
