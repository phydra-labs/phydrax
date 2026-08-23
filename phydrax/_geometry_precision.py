#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ._precision import (
    complex_precision_dtype,
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
    ScalarPrecisionDType,
)
from ._strict import StrictModule
from ._trainable import NonTrainableState


_SUPPORTED_GEOMETRY_DTYPES = frozenset(("float32", "float64", "complex64", "complex128"))


def _real_component(value: Any, /) -> str:
    name = precision_dtype_name(value)
    if name == "complex64":
        return "float32"
    if name == "complex128":
        return "float64"
    return real_precision_dtype_name(name)


def _effective_dtype(
    requested: str | None,
    observed: ScalarPrecisionDType,
    /,
) -> ScalarPrecisionDType:
    if requested is None:
        return observed
    if observed in ("complex64", "complex128") and requested in (
        "float32",
        "float64",
    ):
        return complex_precision_dtype(requested)
    return precision_dtype_name(requested)


class GeometryPrecisionPolicy(StrictModule, NonTrainableState):
    """Coordinate, local compute, reduction, decision, and output precision."""

    coordinate_dtype: str | None = eqx.field(static=True)
    compute_dtype: str | None = eqx.field(static=True)
    accumulation_dtype: str | None = eqx.field(static=True)
    decision_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coordinate_dtype: Any | None = None,
        compute_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        decision_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ):
        coordinate = (
            None if coordinate_dtype is None else precision_dtype_name(coordinate_dtype)
        )
        compute = None if compute_dtype is None else precision_dtype_name(compute_dtype)
        accumulation = (
            None
            if accumulation_dtype is None
            else precision_dtype_name(accumulation_dtype)
        )
        decision = (
            None if decision_dtype is None else real_precision_dtype_name(decision_dtype)
        )
        output = None if output_dtype is None else precision_dtype_name(output_dtype)
        requested = tuple(
            value
            for value in (coordinate, compute, accumulation, decision, output)
            if value is not None
        )
        if any(value not in _SUPPORTED_GEOMETRY_DTYPES for value in requested):
            raise ValueError("Geometry precision supports float32/64 and complex64/128.")
        if (
            accumulation is not None
            and compute is not None
            and precision_itemsize(_real_component(accumulation))
            < precision_itemsize(_real_component(compute))
        ):
            raise ValueError(
                "Geometry accumulation precision cannot be narrower than compute."
            )
        if (
            decision is not None
            and accumulation is not None
            and precision_itemsize(decision)
            < precision_itemsize(_real_component(accumulation))
        ):
            raise ValueError(
                "Geometry decision precision cannot be narrower than accumulation."
            )
        request = PrecisionRequest(
            "geometry",
            {
                "storage": coordinate,
                "compute": compute,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        self.coordinate_dtype = coordinate
        self.compute_dtype = compute
        self.accumulation_dtype = accumulation
        self.decision_dtype = decision
        self.output_dtype = output
        self.policy_id = request.request_id

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "geometry",
            {
                "storage": self.coordinate_dtype,
                "compute": self.compute_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.decision_dtype,
                "output": self.output_dtype,
            },
        )

    def validate_coordinates(self, coordinates: Any, /) -> ScalarPrecisionDType:
        array = jnp.asarray(coordinates)
        observed = precision_dtype_name(array.dtype)
        if self.coordinate_dtype is not None and observed != self.coordinate_dtype:
            raise TypeError(
                f"Geometry coordinate dtype {observed} does not match "
                f"{self.coordinate_dtype}."
            )
        return observed

    def compute(self, value: Any, /):
        array = jnp.asarray(value)
        if self.compute_dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(
            _effective_dtype(self.compute_dtype, precision_dtype_name(array.dtype))
        )

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        if self.accumulation_dtype is None or not jnp.issubdtype(
            array.dtype, jnp.inexact
        ):
            return array
        return array.astype(
            _effective_dtype(
                self.accumulation_dtype,
                precision_dtype_name(array.dtype),
            )
        )

    def decision(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.decision_dtype is None else array.astype(self.decision_dtype)

    def output(self, value: Any, /):
        array = jnp.asarray(value)
        if self.output_dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(
            _effective_dtype(self.output_dtype, precision_dtype_name(array.dtype))
        )

    def sum(self, value: Any, /, *, axis: Any = None, keepdims: bool = False):
        return jnp.sum(
            self.accumulation(value),
            axis=axis,
            keepdims=keepdims,
        )

    def norm(self, value: Any, /, *, axis: Any = None, keepdims: bool = False):
        accumulated = self.accumulation(value)
        squared = self.sum(jnp.abs(accumulated) ** 2, axis=axis, keepdims=keepdims)
        return self.decision(jnp.sqrt(squared))

    def evidence_for(
        self,
        coordinates: Any,
        /,
        *,
        children: dict[str, PrecisionEvidenceEnvelope] | None = None,
    ) -> PrecisionEvidenceEnvelope:
        observed = self.validate_coordinates(coordinates)
        compute = _effective_dtype(self.compute_dtype, observed)
        accumulation = _effective_dtype(self.accumulation_dtype, compute)
        decision = (
            _real_component(accumulation)
            if self.decision_dtype is None
            else self.decision_dtype
        )
        output = _effective_dtype(self.output_dtype, observed)
        resolution = PrecisionResolution(
            self.request,
            "phydrax-geometry",
            {
                "storage": observed,
                "compute": compute,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        return PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
            children={} if children is None else children,
        )


__all__ = ["GeometryPrecisionPolicy"]
