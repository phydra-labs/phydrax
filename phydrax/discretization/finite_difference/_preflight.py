#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_support import PreparedTensorGrid
from ..amr import BlockLevelPlan
from ._execution import PreparedStencilExecutionOperator
from ._operators import PreparedStencilOperator


class FDPrecisionPolicy(StrictModule, NonTrainableState):
    """Explicit coefficient, compute, reduction, and checkpoint dtypes."""

    coefficient_dtype: str = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    reduction_dtype: str = eqx.field(static=True)
    checkpoint_dtype: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coefficient_dtype: Any = np.float64,
        compute_dtype: Any = np.float64,
        reduction_dtype: Any = np.float64,
        checkpoint_dtype: Any | None = None,
    ):
        coefficient = _canonical_dtype(coefficient_dtype)
        compute = _canonical_dtype(compute_dtype)
        reduction = _canonical_dtype(reduction_dtype)
        checkpoint = (
            compute if checkpoint_dtype is None else _canonical_dtype(checkpoint_dtype)
        )
        if any(
            not np.issubdtype(value, np.inexact)
            for value in (coefficient, compute, reduction, checkpoint)
        ):
            raise TypeError("FD precision dtypes must be real or complex inexact types.")
        if reduction.itemsize < compute.itemsize:
            raise ValueError(
                "FD reduction precision cannot be narrower than compute precision."
            )
        self.coefficient_dtype = coefficient.str
        self.compute_dtype = compute.str
        self.reduction_dtype = reduction.str
        self.checkpoint_dtype = checkpoint.str
        self.policy_id = canonical_fingerprint(
            {
                "kind": "fd-precision-policy",
                "coefficient": coefficient.str,
                "compute": compute.str,
                "reduction": reduction.str,
                "checkpoint": checkpoint.str,
            }
        )

    @property
    def compute_itemsize(self) -> int:
        return np.dtype(self.compute_dtype).itemsize

    @property
    def checkpoint_itemsize(self) -> int:
        return np.dtype(self.checkpoint_dtype).itemsize


def _canonical_dtype(value: Any, /) -> np.dtype:
    return np.dtype(jax.dtypes.canonicalize_dtype(np.dtype(value)))


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
        self.estimate_id = canonical_fingerprint(
            {
                "kind": "fd-resource-estimate",
                "plan": plan_id,
                "components": list(values),
                "total": total,
                "budget": budget,
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
    precision: FDPrecisionPolicy
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
        precision: FDPrecisionPolicy | None = None,
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
        precision_ = FDPrecisionPolicy() if precision is None else precision
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
        itemsize = self.precision.compute_itemsize
        state_bytes = self.grid.size * self.field_count * itemsize
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
            * self.precision.checkpoint_itemsize
            * self.checkpoint_copies
            + sum(
                level.maximum_blocks
                * int(np.prod(level.block_shape))
                * self.field_count
                * self.precision.checkpoint_itemsize
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
    "FDPrecisionPolicy",
    "FDResourceEstimate",
]
