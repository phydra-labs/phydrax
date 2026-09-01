#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._precision import precision_itemsize, real_precision_dtype_name
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class VortexPrecisionPolicy(StrictModule, NonTrainableState):
    """Coordinate, kernel, accumulation, and output precision for vortex work."""

    coordinate_dtype: str | None = eqx.field(static=True)
    compute_dtype: str | None = eqx.field(static=True)
    accumulation_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coordinate_dtype: Any | None = None,
        compute_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ):
        coordinate = (
            None
            if coordinate_dtype is None
            else real_precision_dtype_name(coordinate_dtype)
        )
        compute = (
            None if compute_dtype is None else real_precision_dtype_name(compute_dtype)
        )
        accumulation = (
            None
            if accumulation_dtype is None
            else real_precision_dtype_name(accumulation_dtype)
        )
        output = None if output_dtype is None else real_precision_dtype_name(output_dtype)
        if (
            compute is not None
            and accumulation is not None
            and precision_itemsize(accumulation) < precision_itemsize(compute)
        ):
            raise ValueError(
                "Vortex accumulation precision cannot be narrower than compute."
            )
        self.coordinate_dtype = coordinate
        self.compute_dtype = compute
        self.accumulation_dtype = accumulation
        self.output_dtype = output
        self.policy_id = canonical_fingerprint(
            {
                "kind": "vortex-precision-policy",
                "coordinate": coordinate,
                "compute": compute,
                "accumulation": accumulation,
                "output": output,
            }
        )

    def validate_coordinates(self, value: Any, /) -> str:
        observed = real_precision_dtype_name(jnp.asarray(value).dtype)
        if self.coordinate_dtype is not None and observed != self.coordinate_dtype:
            raise TypeError(
                f"Vortex coordinate dtype {observed} does not match "
                f"{self.coordinate_dtype}."
            )
        return observed

    def compute(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.compute_dtype is None else array.astype(self.compute_dtype)

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        return (
            array
            if self.accumulation_dtype is None
            else array.astype(self.accumulation_dtype)
        )

    def output(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.output_dtype is None else array.astype(self.output_dtype)

    def sum(self, value: Any, /, *, axis: Any = None, keepdims: bool = False):
        return jnp.sum(self.accumulation(value), axis=axis, keepdims=keepdims)


__all__ = ["VortexPrecisionPolicy"]
