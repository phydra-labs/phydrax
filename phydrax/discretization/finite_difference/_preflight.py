#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_support import PreparedTensorGrid
from ..amr import BlockLevelPlan
from ._execution import PreparedStencilExecutionOperator
from ._operators import PreparedStencilOperator
from ._precision import FDExecutionPrecisionPolicy


class FDResourceEstimate(StrictModule, NonTrainableState):
    state_bytes: int = eqx.field(static=True)
    halo_bytes: int = eqx.field(static=True)
    stencil_metadata_bytes: int = eqx.field(static=True)
    temporary_bytes: int = eqx.field(static=True)
    amr_bytes: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)
    memory_budget_bytes: int | None = eqx.field(static=True)
    fits_budget: bool = eqx.field(static=True)
    estimate_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    precision_resource_assumptions_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_bytes: int,
        halo_bytes: int,
        stencil_metadata_bytes: int,
        temporary_bytes: int,
        amr_bytes: int,
        checkpoint_bytes: int,
        memory_budget_bytes: int | None,
        plan_id: str,
        precision_policy_id: str,
        precision_resource_assumptions_id: str,
    ):
        values = tuple(
            int(value)
            for value in (
                state_bytes,
                halo_bytes,
                stencil_metadata_bytes,
                temporary_bytes,
                amr_bytes,
                checkpoint_bytes,
            )
        )
        if any(value < 0 for value in values):
            raise ValueError("FD resource byte estimates must be non-negative.")
        total = sum(values)
        budget = None if memory_budget_bytes is None else int(memory_budget_bytes)
        if budget is not None and budget <= 0:
            raise ValueError("FD memory budget must be positive or None.")
        self.state_bytes = values[0]
        self.halo_bytes = values[1]
        self.stencil_metadata_bytes = values[2]
        self.temporary_bytes = values[3]
        self.amr_bytes = values[4]
        self.checkpoint_bytes = values[5]
        self.total_bytes = total
        self.memory_budget_bytes = budget
        self.fits_budget = budget is None or total <= budget
        self.precision_policy_id = str(precision_policy_id)
        self.precision_resource_assumptions_id = str(precision_resource_assumptions_id)
        self.estimate_id = canonical_fingerprint(
            {
                "kind": "fd-resource-estimate",
                "plan": plan_id,
                "components": list(values),
                "total": total,
                "budget": budget,
                "precision_policy": self.precision_policy_id,
                "precision_resource_assumptions": (
                    self.precision_resource_assumptions_id
                ),
            }
        )


class FDExecutionPreflightPlan(StrictModule, NonTrainableState):
    """Static memory/capacity estimate before compiling an FD execution."""

    grid: PreparedTensorGrid
    field_count: int = eqx.field(static=True)
    halo_widths: tuple[tuple[int, int], ...] = eqx.field(static=True)
    operators: tuple[PreparedStencilOperator | PreparedStencilExecutionOperator, ...]
    amr_levels: tuple[BlockLevelPlan, ...]
    temporary_fields: int = eqx.field(static=True)
    checkpoint_copies: int = eqx.field(static=True)
    precision: FDExecutionPrecisionPolicy
    memory_budget_bytes: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        field_count: int,
        halo_widths: Sequence[tuple[int, int]] | None = None,
        operators: Sequence[
            PreparedStencilOperator | PreparedStencilExecutionOperator
        ] = (),
        amr_levels: Sequence[BlockLevelPlan] = (),
        temporary_fields: int = 2,
        checkpoint_copies: int = 1,
        precision: FDExecutionPrecisionPolicy | None = None,
        memory_budget_bytes: int | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("FD execution preflight requires PreparedTensorGrid.")
        fields = int(field_count)
        temporaries = int(temporary_fields)
        copies = int(checkpoint_copies)
        halos = (
            ((0, 0),) * len(grid.shape)
            if halo_widths is None
            else tuple((int(value[0]), int(value[1])) for value in halo_widths)
        )
        operators_ = tuple(operators)
        levels = tuple(amr_levels)
        precision_ = FDExecutionPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, FDExecutionPrecisionPolicy):
            raise TypeError("precision must be an FDExecutionPrecisionPolicy.")
        for operator in operators_:
            operator_precision = (
                operator.reference_operator.precision
                if isinstance(operator, PreparedStencilExecutionOperator)
                else operator.precision
            )
            if operator_precision.policy_id != precision_.policy_id:
                raise ValueError(
                    "FD preflight operators must share one precision policy."
                )
        if (
            fields <= 0
            or temporaries < 0
            or copies < 0
            or len(halos) != len(grid.shape)
            or any(lower < 0 or upper < 0 for lower, upper in halos)
            or not all(
                isinstance(
                    value,
                    (PreparedStencilOperator, PreparedStencilExecutionOperator),
                )
                for value in operators_
            )
            or not all(isinstance(value, BlockLevelPlan) for value in levels)
        ):
            raise ValueError("FD execution preflight controls are invalid.")
        self.grid = grid
        self.field_count = fields
        self.halo_widths = halos
        self.operators = operators_
        self.amr_levels = levels
        self.temporary_fields = temporaries
        self.checkpoint_copies = copies
        self.precision = precision_
        self.memory_budget_bytes = (
            None if memory_budget_bytes is None else int(memory_budget_bytes)
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fd-execution-preflight",
                "grid": grid.prepared_id,
                "field_count": fields,
                "halos": [list(value) for value in halos],
                "operators": [value.operator_id for value in operators_],
                "amr_levels": [value.plan_id for value in levels],
                "temporary_fields": temporaries,
                "checkpoint_copies": copies,
                "precision": precision_.policy_id,
                "budget": self.memory_budget_bytes,
            }
        )

    def estimate(self, /) -> FDResourceEstimate:
        assumptions = self.precision.resource_assumptions
        itemsize = assumptions.itemsize("storage")
        state_bytes = self.grid.size * self.field_count * itemsize
        checkpoint_itemsize = assumptions.itemsize("checkpoint")
        halo_shape = tuple(
            size + lower + upper
            for size, (lower, upper) in zip(
                self.grid.shape,
                self.halo_widths,
                strict=True,
            )
        )
        halo_bytes = (
            (int(np.prod(halo_shape)) - self.grid.size) * self.field_count * itemsize
        )
        metadata = 0
        for operator in self.operators:
            if isinstance(operator, PreparedStencilExecutionOperator):
                metadata += operator.execution.report.lowered_metadata_bytes
            else:
                metadata += sum(
                    int(np.asarray(value).nbytes)
                    for value in (operator.indices, operator.weights, operator.valid)
                )
        temporary_bytes = self.grid.size * self.temporary_fields * itemsize
        amr_bytes = sum(
            level.maximum_blocks
            * int(np.prod(level.block_shape))
            * self.field_count
            * itemsize
            for level in self.amr_levels
        )
        checkpoint_bytes = (
            self.grid.size
            * self.field_count
            * checkpoint_itemsize
            * self.checkpoint_copies
            + sum(
                level.maximum_blocks
                * int(np.prod(level.block_shape))
                * self.field_count
                * checkpoint_itemsize
                * self.checkpoint_copies
                for level in self.amr_levels
            )
        )
        estimate = FDResourceEstimate(
            state_bytes=state_bytes,
            halo_bytes=halo_bytes,
            stencil_metadata_bytes=metadata,
            temporary_bytes=temporary_bytes,
            amr_bytes=amr_bytes,
            checkpoint_bytes=checkpoint_bytes,
            memory_budget_bytes=self.memory_budget_bytes,
            precision_policy_id=self.precision.policy_id,
            precision_resource_assumptions_id=assumptions.assumptions_id,
            plan_id=self.plan_id,
        )
        if not estimate.fits_budget:
            raise ValueError(
                f"FD execution requires {estimate.total_bytes} bytes, exceeding "
                f"budget {estimate.memory_budget_bytes}."
            )
        return estimate


__all__ = [
    "FDExecutionPreflightPlan",
    "FDResourceEstimate",
]
