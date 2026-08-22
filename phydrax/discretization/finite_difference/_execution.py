#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    OperatorCapabilities,
)
from ._operators import PreparedStencilOperator


class InteriorStencilKernel(StrictModule, NonTrainableState):
    """One offset/weight row reused by every regular interior target."""

    axis: int = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    weights: Array
    target_start: int = eqx.field(static=True)
    target_stop: int = eqx.field(static=True)
    periodic: bool = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        offsets: tuple[int, ...],
        weights: ArrayLike,
        target_start: int,
        target_stop: int,
        /,
        *,
        periodic: bool,
        stencil_id: str,
    ):
        weights_ = jnp.asarray(weights)
        start = int(target_start)
        stop = int(target_stop)
        if (
            not offsets
            or weights_.shape != (len(offsets),)
            or stop <= start
            or any(not np.isfinite(value) for value in np.asarray(weights_))
        ):
            raise ValueError(
                "Interior stencil offsets, weights, or target range are invalid."
            )
        self.axis = int(axis)
        self.offsets = tuple(int(value) for value in offsets)
        self.weights = weights_
        self.target_start = start
        self.target_stop = stop
        self.periodic = bool(periodic)
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "interior-stencil-kernel",
                "stencil": stencil_id,
                "axis": int(axis),
                "offsets": list(offsets),
                "target_range": [start, stop],
                "periodic": bool(periodic),
            }
        )

    def apply(self, source: Array, target_shape: tuple[int, ...], /) -> Array:
        moved = jnp.moveaxis(source, self.axis, 0)
        target_axis = target_shape[self.axis]
        output_shape = (target_axis,) + moved.shape[1:]
        output = jnp.zeros(output_shape, dtype=jnp.result_type(source, self.weights))
        if self.periodic:
            result = jnp.zeros_like(output)
            for offset, weight in zip(self.offsets, self.weights, strict=True):
                result = result + weight * jnp.roll(moved, -offset, axis=0)
            output = result
        else:
            count = self.target_stop - self.target_start
            result = jnp.zeros((count,) + moved.shape[1:], dtype=output.dtype)
            for offset, weight in zip(self.offsets, self.weights, strict=True):
                start = self.target_start + offset
                result = result + weight * moved[start : start + count]
            output = output.at[self.target_start : self.target_stop].set(result)
        return jnp.moveaxis(output, 0, self.axis)


class ClosureStencilKernel(StrictModule, NonTrainableState):
    """Only irregular closure rows retained from the canonical masked bank."""

    axis: int = eqx.field(static=True)
    target_indices: Array
    source_indices: Array
    weights: Array
    valid: Array
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        axis: int,
        target_indices: ArrayLike,
        source_indices: ArrayLike,
        weights: ArrayLike,
        valid: ArrayLike,
        /,
        *,
        stencil_id: str,
    ):
        targets = jnp.asarray(target_indices, dtype=jnp.int32)
        sources = jnp.asarray(source_indices, dtype=jnp.int32)
        weights_ = jnp.asarray(weights)
        valid_ = jnp.asarray(valid, dtype=bool)
        if (
            targets.ndim != 1
            or sources.ndim != 2
            or weights_.shape != sources.shape
            or valid_.shape != sources.shape
            or sources.shape[0] != targets.size
        ):
            raise ValueError("Closure stencil rows and targets must align.")
        self.axis = int(axis)
        self.target_indices = targets
        self.source_indices = sources
        self.weights = weights_
        self.valid = valid_
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "closure-stencil-kernel",
                "stencil": stencil_id,
                "axis": int(axis),
                "rows": int(targets.size),
                "capacity": int(sources.shape[1]),
            }
        )

    def apply(self, source: Array, target_shape: tuple[int, ...], /) -> Array:
        moved = jnp.moveaxis(source, self.axis, 0)
        safe_indices = jnp.where(self.valid, self.source_indices, 0)
        gathered = moved[safe_indices]
        mask_shape = self.valid.shape + (1,) * (gathered.ndim - 2)
        safe_values = jnp.where(
            self.valid.reshape(mask_shape),
            gathered,
            jnp.zeros((), dtype=gathered.dtype),
        )
        safe_weights = jnp.where(self.valid, self.weights, 0.0)
        weight_shape = safe_weights.shape + (1,) * (gathered.ndim - 2)
        rows = jnp.sum(safe_weights.reshape(weight_shape) * safe_values, axis=1)
        target_axis = target_shape[self.axis]
        output = jnp.zeros((target_axis,) + moved.shape[1:], dtype=rows.dtype)
        output = output.at[self.target_indices].set(rows)
        return jnp.moveaxis(output, 0, self.axis)


class StencilExecutionReport(StrictModule, NonTrainableState):
    """Interior/closure counts, metadata compression, and parity evidence."""

    interior_rows: int = eqx.field(static=True)
    closure_rows: int = eqx.field(static=True)
    canonical_metadata_bytes: int = eqx.field(static=True)
    lowered_metadata_bytes: int = eqx.field(static=True)
    maximum_parity_residual: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        interior_rows: int,
        closure_rows: int,
        canonical_metadata_bytes: int,
        lowered_metadata_bytes: int,
        maximum_parity_residual: float,
        stencil_id: str,
    ):
        residual = float(maximum_parity_residual)
        self.interior_rows = int(interior_rows)
        self.closure_rows = int(closure_rows)
        self.canonical_metadata_bytes = int(canonical_metadata_bytes)
        self.lowered_metadata_bytes = int(lowered_metadata_bytes)
        self.maximum_parity_residual = residual
        self.passed = (
            residual <= 1e-11 and lowered_metadata_bytes <= canonical_metadata_bytes
        )
        self.report_id = canonical_fingerprint(
            {
                "kind": "stencil-execution-report",
                "stencil": stencil_id,
                "interior_rows": int(interior_rows),
                "closure_rows": int(closure_rows),
                "canonical_metadata_bytes": int(canonical_metadata_bytes),
                "lowered_metadata_bytes": int(lowered_metadata_bytes),
                "maximum_parity_residual": residual,
            }
        )


class StencilExecutionPlan(StrictModule, NonTrainableState):
    """Compact regular interior plus masked irregular closures for one operator."""

    reference_operator: PreparedStencilOperator
    interior: InteriorStencilKernel | None
    closure: ClosureStencilKernel | None
    report: StencilExecutionReport
    plan_id: str = eqx.field(static=True)

    def __init__(self, operator: PreparedStencilOperator, /):
        if not isinstance(operator, PreparedStencilOperator):
            raise TypeError(
                "Stencil execution lowering requires PreparedStencilOperator."
            )
        stencil = operator.stencil_set.stencil
        reports = stencil.row_reports
        interior_rows = np.asarray(
            [index for index, report in enumerate(reports) if report.kind == "interior"],
            dtype=np.int32,
        )
        closure_rows = np.asarray(
            [index for index, report in enumerate(reports) if report.kind != "interior"],
            dtype=np.int32,
        )
        interior = _prepare_interior(operator, interior_rows)
        closure = (
            None
            if closure_rows.size == 0
            else ClosureStencilKernel(
                operator.axis,
                closure_rows,
                np.asarray(operator.indices)[closure_rows],
                np.asarray(operator.weights)[closure_rows],
                np.asarray(operator.valid)[closure_rows],
                stencil_id=stencil.stencil_id,
            )
        )
        self.reference_operator = operator
        self.interior = interior
        self.closure = closure
        probe_index = jnp.arange(operator.source.size, dtype=operator.source.dtype)
        probe = (jnp.sin(0.17 * probe_index) + 0.3 * jnp.cos(0.07 * probe_index)).reshape(
            operator.source.shape
        )
        lowered = self.apply(probe)
        reference = operator.mv(probe)
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(reference)))
        residual = float(np.asarray(jnp.max(jnp.abs(lowered - reference)) / scale))
        canonical_bytes = sum(
            int(np.asarray(value).nbytes)
            for value in (operator.indices, operator.weights, operator.valid)
        )
        lowered_arrays = []
        if interior is not None:
            lowered_arrays.append(interior.weights)
        if closure is not None:
            lowered_arrays.extend(
                (
                    closure.target_indices,
                    closure.source_indices,
                    closure.weights,
                    closure.valid,
                )
            )
        lowered_bytes = sum(int(np.asarray(value).nbytes) for value in lowered_arrays)
        report = StencilExecutionReport(
            interior_rows=int(interior_rows.size),
            closure_rows=int(closure_rows.size),
            canonical_metadata_bytes=canonical_bytes,
            lowered_metadata_bytes=lowered_bytes,
            maximum_parity_residual=residual,
            stencil_id=stencil.stencil_id,
        )
        if not report.passed:
            raise RuntimeError(
                "Compact stencil lowering failed parity or metadata evidence."
            )
        self.report = report
        self.plan_id = canonical_fingerprint(
            {
                "kind": "stencil-execution-plan",
                "operator": operator.operator_id,
                "interior": None if interior is None else interior.kernel_id,
                "closure": None if closure is None else closure.kernel_id,
                "report": report.report_id,
            }
        )

    def apply(self, values: ArrayLike, /) -> Array:
        source = self.reference_operator.source.validate(jnp.asarray(values))
        output = jnp.zeros(
            self.reference_operator.target.shape,
            dtype=jnp.result_type(source, self.reference_operator.weights),
        )
        if self.interior is not None:
            output = output + self.interior.apply(
                source,
                self.reference_operator.target.shape,
            )
        if self.closure is not None:
            output = output + self.closure.apply(
                source,
                self.reference_operator.target.shape,
            )
        return self.reference_operator.target.validate(output)


def _prepare_interior(
    operator: PreparedStencilOperator,
    rows: np.ndarray,
    /,
) -> InteriorStencilKernel | None:
    if rows.size == 0:
        return None
    if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
        return None
    indices = np.asarray(operator.indices)
    weights = np.asarray(operator.weights)
    valid = np.asarray(operator.valid)
    periodic = operator.stencil_set.kind == "periodic"

    def relative(row: int, row_valid: np.ndarray) -> tuple[int, ...]:
        values = indices[row, row_valid] - row
        if periodic:
            count = operator.source.shape[operator.axis]
            values = np.mod(values, count)
            values = np.where(values > count // 2, values - count, values)
        return tuple(values.tolist())

    first_valid = valid[rows[0]]
    offsets = relative(int(rows[0]), first_valid)
    row_weights = weights[rows[0], first_valid]
    for row in rows[1:]:
        row_valid = valid[row]
        if relative(int(row), row_valid) != offsets or not np.allclose(
            weights[row, row_valid],
            row_weights,
            rtol=1e-12,
            atol=1e-14,
        ):
            return None
    return InteriorStencilKernel(
        operator.axis,
        offsets,
        row_weights,
        int(rows[0]),
        int(rows[-1] + 1),
        periodic=periodic,
        stencil_id=operator.stencil_set.stencil.stencil_id,
    )


class PreparedStencilExecutionOperator(AbstractLinearOperator):
    source: ArraySpace
    target: ArraySpace
    execution: StencilExecutionPlan

    def __init__(self, execution: StencilExecutionPlan, /):
        if not isinstance(execution, StencilExecutionPlan):
            raise TypeError("execution must be a StencilExecutionPlan.")
        reference = execution.reference_operator
        self.source = reference.source
        self.target = reference.target
        self.properties = reference.properties
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=True,
            diagonal_assembly=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "prepared-stencil-execution-operator",
                "execution": execution.plan_id,
                "reference": reference.operator_id,
            }
        )
        self.execution = execution

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.execution.apply(vector)

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        return self.reference_operator.transpose_mv(vector)

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        return self.reference_operator.adjoint_mv(vector)

    def _materialize(self, /) -> Array:
        return self.reference_operator._materialize()

    @property
    def reference_operator(self) -> PreparedStencilOperator:
        return self.execution.reference_operator


def lower_stencil_operator(
    operator: PreparedStencilOperator,
    /,
) -> PreparedStencilExecutionOperator:
    return PreparedStencilExecutionOperator(StencilExecutionPlan(operator))


__all__ = [
    "ClosureStencilKernel",
    "InteriorStencilKernel",
    "lower_stencil_operator",
    "PreparedStencilExecutionOperator",
    "StencilExecutionPlan",
    "StencilExecutionReport",
]
