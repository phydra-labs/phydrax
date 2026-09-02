#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations.fem._ir import LoweredOperatorProgram
from ..equations.fem._worksets import WorksetProgram


class ProductionResourceBudget(StrictModule, NonTrainableState):
    maximum_compile_units: int = eqx.field(static=True)
    maximum_host_bytes: int = eqx.field(static=True)
    maximum_device_bytes: int = eqx.field(static=True)
    maximum_output_queue_bytes: int = eqx.field(static=True)
    budget_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_compile_units: int,
        maximum_host_bytes: int,
        maximum_device_bytes: int,
        maximum_output_queue_bytes: int,
    ):
        values = tuple(
            int(value)
            for value in (
                maximum_compile_units,
                maximum_host_bytes,
                maximum_device_bytes,
                maximum_output_queue_bytes,
            )
        )
        if any(value <= 0 for value in values):
            raise ValueError("Production resource budgets must be positive.")
        (
            self.maximum_compile_units,
            self.maximum_host_bytes,
            self.maximum_device_bytes,
            self.maximum_output_queue_bytes,
        ) = values
        self.budget_id = canonical_fingerprint(
            {
                "kind": "production-resource-budget",
                "values": values,
            }
        )


class ProductionResourceForecast(StrictModule, NonTrainableState):
    compile_units: int = eqx.field(static=True)
    host_bytes: int = eqx.field(static=True)
    device_bytes: int = eqx.field(static=True)
    ad_bytes: int = eqx.field(static=True)
    output_queue_bytes: int = eqx.field(static=True)
    admitted: bool = eqx.field(static=True)
    forecast_id: str = eqx.field(static=True)


def prepare_production_resource_forecast(
    worksets: WorksetProgram,
    state: Array,
    budget: ProductionResourceBudget,
    /,
    *,
    ad_multiplier: float = 3.0,
    output_snapshots: int = 2,
) -> ProductionResourceForecast:
    if not isinstance(worksets, WorksetProgram) or not isinstance(
        budget, ProductionResourceBudget
    ):
        raise TypeError("Resource forecast requires worksets and budget.")
    state_bytes = int(np.asarray(state).nbytes)
    workset_bytes = sum(bucket.resident_bytes for bucket in worksets.buckets)
    compile_units = len(worksets.buckets) * len(worksets.operator_program.nodes)
    host_bytes = state_bytes + workset_bytes
    device_bytes = 2 * state_bytes + workset_bytes
    ad_bytes = int(float(ad_multiplier) * device_bytes)
    output_bytes = int(output_snapshots) * state_bytes
    admitted = bool(
        compile_units <= budget.maximum_compile_units
        and host_bytes <= budget.maximum_host_bytes
        and device_bytes + ad_bytes <= budget.maximum_device_bytes
        and output_bytes <= budget.maximum_output_queue_bytes
    )
    forecast_id = canonical_fingerprint(
        {
            "kind": "production-resource-forecast",
            "worksets": worksets.program_id,
            "budget": budget.budget_id,
            "compile_units": compile_units,
            "host_bytes": host_bytes,
            "device_bytes": device_bytes,
            "ad_bytes": ad_bytes,
            "output_queue_bytes": output_bytes,
        }
    )
    return ProductionResourceForecast(
        compile_units,
        host_bytes,
        device_bytes,
        ad_bytes,
        output_bytes,
        admitted,
        forecast_id,
    )


class PreparedCompilationService:
    """Single-process in-memory executable cache keyed by semantic program IDs."""

    def __init__(self):
        self._executables: dict[str, Any] = {}

    @property
    def entry_count(self) -> int:
        return len(self._executables)

    def compile(
        self,
        lowered: LoweredOperatorProgram,
        sample_inputs: Mapping[str, Any],
        /,
    ):
        if not isinstance(lowered, LoweredOperatorProgram):
            raise TypeError("compile requires LoweredOperatorProgram.")
        backend = jax.default_backend()
        device = str(jax.devices()[0])
        key = canonical_fingerprint(
            {
                "kind": "production-executable-cache-key",
                "lowered": lowered.lowered_id,
                "backend": backend,
                "device": device,
                "jax": jax.__version__,
                "input_shapes": {
                    name: (tuple(np.shape(value)), str(np.asarray(value).dtype))
                    for name, value in sorted(sample_inputs.items())
                },
            }
        )
        if key not in self._executables:
            function = jax.jit(lambda inputs: lowered(inputs))
            self._executables[key] = function.lower(sample_inputs).compile()
        return self._executables[key]


__all__ = [
    "ProductionResourceBudget",
    "ProductionResourceForecast",
    "PreparedCompilationService",
    "prepare_production_resource_forecast",
]
