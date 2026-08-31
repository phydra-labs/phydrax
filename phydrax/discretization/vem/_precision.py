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


class VirtualElementPrecisionPolicy(StrictModule, NonTrainableState):
    geometry_dtype: RealPrecisionDType = eqx.field(static=True)
    projection_dtype: RealPrecisionDType = eqx.field(static=True)
    accumulation_dtype: RealPrecisionDType = eqx.field(static=True)
    output_dtype: RealPrecisionDType = eqx.field(static=True)
    certification_dtype: RealPrecisionDType = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry_dtype: Any = "float64",
        projection_dtype: Any = "float64",
        accumulation_dtype: Any | None = None,
        output_dtype: Any | None = None,
        certification_dtype: Any = "float64",
    ):
        geometry = real_precision_dtype_name(geometry_dtype)
        projection = real_precision_dtype_name(projection_dtype)
        accumulation = real_precision_dtype_name(
            projection if accumulation_dtype is None else accumulation_dtype
        )
        output = real_precision_dtype_name(
            projection if output_dtype is None else output_dtype
        )
        certification = real_precision_dtype_name(certification_dtype)
        self.geometry_dtype = geometry
        self.projection_dtype = projection
        self.accumulation_dtype = accumulation
        self.output_dtype = output
        self.certification_dtype = certification
        self.policy_id = canonical_fingerprint(
            {
                "kind": "virtual-element-precision",
                "geometry": geometry,
                "projection": projection,
                "accumulation": accumulation,
                "output": output,
                "certification": certification,
            }
        )

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "virtual-element",
            {
                "basis": self.geometry_dtype,
                "factorization": self.projection_dtype,
                "compute": self.projection_dtype,
                "accumulation": self.accumulation_dtype,
                "output": self.output_dtype,
                "certification": self.certification_dtype,
            },
        )

    def geometry(self, value: Any, /):
        return jnp.asarray(value, dtype=self.geometry_dtype)

    def projection(self, value: Any, /):
        return jnp.asarray(value, dtype=self.projection_dtype)

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        return (
            array
            if not jnp.issubdtype(array.dtype, jnp.inexact)
            else array.astype(self.accumulation_dtype)
        )

    def output(self, value: Any, /):
        return jnp.asarray(value, dtype=self.output_dtype)

    def evidence(self) -> PrecisionEvidenceEnvelope:
        request = self.request
        resolution = PrecisionResolution(
            request,
            "phydrax-virtual-element",
            dict(request.requested),
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


class VirtualElementResourceBudget(StrictModule, NonTrainableState):
    maximum_local_dofs: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_projector_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_local_dofs: int = 256,
        maximum_cells: int = 1_000_000,
        maximum_projector_bytes: int = 1 << 30,
    ):
        local = int(maximum_local_dofs)
        cells = int(maximum_cells)
        storage = int(maximum_projector_bytes)
        if local <= 0 or cells <= 0 or storage <= 0:
            raise ValueError("Virtual-element resource budgets must be positive.")
        self.maximum_local_dofs = local
        self.maximum_cells = cells
        self.maximum_projector_bytes = storage
        self.budget_id = canonical_fingerprint(
            {
                "kind": "virtual-element-resource-budget",
                "local_dofs": local,
                "cells": cells,
                "projector_bytes": storage,
            }
        )


__all__ = ["VirtualElementPrecisionPolicy", "VirtualElementResourceBudget"]
