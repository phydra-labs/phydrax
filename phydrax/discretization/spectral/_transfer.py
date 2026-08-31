#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._spectral._fourier import resize_fourier_axis
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import FunctionLinearOperator
from .._core import DiscretizationCapability, PreparationReport
from .._transfer import FieldTransfer, TransferProperties
from ._basis import AbstractSpectralBasisPlan, PreparedSpectralAxis
from ._constraints import _constraint_matrix, ConstrainedBasisPlan
from ._rational import (
    RationalChebyshevHalfLineBasisPlan,
    RationalChebyshevLineBasisPlan,
)
from ._space import _apply_axis_transform, TensorSpectralDiscretization


class SpectralModalTransferReport(StrictModule, NonTrainableState):
    """Resolution relation, losslessness, trace residual, and transfer work."""

    trace_residual: Array
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    axis_actions: tuple[str, ...] = eqx.field(static=True)
    lossless: bool = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_shape: Sequence[int],
        target_shape: Sequence[int],
        axis_actions: Sequence[str],
        lossless: bool,
        trace_residual: ArrayLike,
        workspace_bytes: int,
    ):
        source = tuple(int(size) for size in source_shape)
        target = tuple(int(size) for size in target_shape)
        actions = tuple(str(action) for action in axis_actions)
        workspace = int(workspace_bytes)
        if (
            not source
            or len(source) != len(target)
            or len(source) != len(actions)
            or any(size <= 0 for size in source + target)
            or any(not action for action in actions)
            or workspace < 0
        ):
            raise ValueError("Spectral transfer report metadata is invalid.")
        residual = jnp.asarray(trace_residual, dtype=float).reshape(())
        residual = eqx.error_if(
            residual,
            ~(jnp.isfinite(residual) & (residual >= 0.0)),
            "Spectral transfer trace residual must be finite and non-negative.",
        )
        self.trace_residual = residual
        self.source_shape = source
        self.target_shape = target
        self.axis_actions = actions
        self.lossless = bool(lossless)
        self.workspace_bytes = workspace
        self.report_id = canonical_fingerprint(
            {
                "kind": "spectral-modal-transfer-report",
                "source_shape": list(source),
                "target_shape": list(target),
                "axis_actions": list(actions),
                "lossless": bool(lossless),
                "trace_residual": float(np.asarray(residual)),
                "workspace_bytes": workspace,
            }
        )


class SpectralModalTransferPlan(StrictModule, NonTrainableState):
    """Symbolic modal transfer between compatible prepared tensor spaces."""

    source: TensorSpectralDiscretization
    target: TensorSpectralDiscretization
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TensorSpectralDiscretization,
        target: TensorSpectralDiscretization,
        /,
    ):
        if not isinstance(source, TensorSpectralDiscretization) or not isinstance(
            target, TensorSpectralDiscretization
        ):
            raise TypeError("source and target must be tensor spectral discretizations.")
        if len(source.axes) != len(target.axes):
            raise ValueError("Spectral transfer ranks must match.")
        self.source = source
        self.target = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-modal-transfer-plan",
                "source": source.prepared_id,
                "target": target.prepared_id,
            }
        )

    def prepare(self, /) -> "PreparedSpectralModalTransfer":
        matrices: list[Array | None] = []
        actions: list[str] = []
        trace_residual = 0.0
        for source_axis, target_axis in zip(
            self.source.axes,
            self.target.axes,
            strict=True,
        ):
            _validate_axis_compatibility(source_axis, target_axis)
            if source_axis.family == "fourier":
                matrices.append(None)
                actions.append("fourier-mode-map")
            elif isinstance(source_axis.plan, ConstrainedBasisPlan):
                matrix, residual = _constrained_axis_matrix(source_axis, target_axis)
                matrices.append(jnp.asarray(matrix))
                actions.append("constrained-base-projection")
                trace_residual = max(trace_residual, residual)
            else:
                matrices.append(None)
                actions.append("degree-prefix")

        source = self.source
        target = self.target

        def action(coefficients):
            result = source._validate_leading(
                coefficients,
                source.modal_shape,
                "Transferred modal coefficients",
            )
            for axis, (source_axis, target_axis, matrix) in enumerate(
                zip(source.axes, target.axes, matrices, strict=True)
            ):
                if source_axis.family == "fourier":
                    result = resize_fourier_axis(result, axis, target_axis.mode_count)
                elif matrix is not None:
                    result = _apply_axis_transform(
                        result,
                        axis,
                        lambda vector, matrix=matrix: matrix @ vector,
                    )
                else:
                    result = _resize_degree_axis(result, axis, target_axis.mode_count)
            return result

        operator = FunctionLinearOperator(
            action,
            source=source.modal_space.vector_space,
            target=target.modal_space.vector_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "spectral-modal-transfer-operator",
                    "plan": self.plan_id,
                    "actions": actions,
                }
            ),
        )
        itemsize = np.dtype(source.plan.precision.coefficient_dtype).itemsize
        workspace = max(
            int(np.prod(source.modal_shape)), int(np.prod(target.modal_shape))
        )
        report = SpectralModalTransferReport(
            source_shape=source.modal_shape,
            target_shape=target.modal_shape,
            axis_actions=actions,
            lossless=all(
                target_size >= source_size
                for source_size, target_size in zip(
                    source.modal_shape,
                    target.modal_shape,
                    strict=True,
                )
            ),
            trace_residual=trace_residual,
            workspace_bytes=workspace * itemsize,
        )
        preparation = PreparationReport(
            capabilities=(DiscretizationCapability.FIELD_TRANSFER,),
            diagnostics=(
                f"lossless:{report.lossless}",
                f"trace_residual:{float(np.asarray(report.trace_residual)):.3e}",
            ),
            resource_counts={"workspace_bytes": report.workspace_bytes},
        )
        transfer = FieldTransfer(
            source.modal_space,
            target.modal_space,
            operator,
            properties=TransferProperties(
                nested=True,
                differentiable_geometry=True,
                exact_on=("retained-modal-subspace",),
            ),
            preparation=preparation,
        )
        return PreparedSpectralModalTransfer(self, transfer, report)


class PreparedSpectralModalTransfer(StrictModule, NonTrainableState):
    plan: SpectralModalTransferPlan
    transfer: FieldTransfer
    report: SpectralModalTransferReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralModalTransferPlan,
        transfer: FieldTransfer,
        report: SpectralModalTransferReport,
        /,
    ):
        if not isinstance(plan, SpectralModalTransferPlan):
            raise TypeError("plan must be a SpectralModalTransferPlan.")
        if not isinstance(transfer, FieldTransfer):
            raise TypeError("transfer must be a FieldTransfer.")
        if not isinstance(report, SpectralModalTransferReport):
            raise TypeError("report must be a SpectralModalTransferReport.")
        self.plan = plan
        self.transfer = transfer
        self.report = report
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-modal-transfer",
                "plan": plan.plan_id,
                "transfer": transfer.transfer_id,
                "report": report.report_id,
            }
        )

    def __call__(self, coefficients: ArrayLike, /) -> Array:
        source = self.plan.source
        values = source._validate_leading(
            coefficients,
            source.modal_shape,
            "Transferred modal coefficients",
        )
        payload_shape = values.shape[len(source.modal_shape) :]
        if not payload_shape:
            return self.transfer.operator.mv(values)
        flattened = values.reshape(source.modal_shape + (-1,))
        transferred = jax.vmap(
            self.transfer.operator.mv,
            in_axes=-1,
            out_axes=-1,
        )(flattened)
        return transferred.reshape(self.plan.target.modal_shape + payload_shape)


def prepare_spectral_modal_transfer(
    source: TensorSpectralDiscretization,
    target: TensorSpectralDiscretization,
    /,
) -> PreparedSpectralModalTransfer:
    return SpectralModalTransferPlan(source, target).prepare()


def _validate_axis_compatibility(
    source: PreparedSpectralAxis,
    target: PreparedSpectralAxis,
    /,
) -> None:
    if source.domain.domain_id != target.domain.domain_id:
        raise ValueError("Spectral transfer axes must use the same physical domain.")
    if source.family != target.family:
        raise ValueError("Spectral transfer basis families must match.")
    if isinstance(source.plan, ConstrainedBasisPlan) != isinstance(
        target.plan, ConstrainedBasisPlan
    ):
        raise ValueError("Constrained and unconstrained bases cannot be mixed.")
    if isinstance(source.plan, ConstrainedBasisPlan):
        if not isinstance(target.plan, ConstrainedBasisPlan):
            raise TypeError("Constrained transfer target is inconsistent.")
        if source.plan.conditions.plan_id != target.plan.conditions.plan_id:
            raise ValueError("Constrained transfer trace plans must match.")
        _validate_plan_family(source.plan.base, target.plan.base)
    else:
        _validate_plan_family(source.plan, target.plan)


def _validate_plan_family(
    source: AbstractSpectralBasisPlan,
    target: AbstractSpectralBasisPlan,
    /,
) -> None:
    if type(source) is not type(target):
        raise ValueError("Spectral transfer basis plan types must match.")
    if source.family in (
        "fourier",
        "sine",
        "cosine",
        "chebyshev",
        "legendre",
    ):
        return
    if isinstance(
        source,
        (RationalChebyshevLineBasisPlan, RationalChebyshevHalfLineBasisPlan),
    ):
        if not np.array_equal(np.asarray(source.scale), np.asarray(target.scale)):
            raise ValueError(
                "Rational spectral transfers require the same map family and scale."
            )
        return
    if source.plan_id != target.plan_id:
        raise ValueError("Mapped spectral transfers require identical basis plans.")


def _resize_degree_axis(
    coefficients: Array,
    axis: int,
    target_size: int,
    /,
) -> Array:
    source_size = int(coefficients.shape[axis])
    target = int(target_size)
    if target == source_size:
        return coefficients
    if target < source_size:
        return jnp.take(coefficients, jnp.arange(target), axis=axis)
    padding = [(0, 0)] * coefficients.ndim
    padding[axis] = (0, target - source_size)
    return jnp.pad(coefficients, tuple(padding))


def _constrained_axis_matrix(
    source: PreparedSpectralAxis,
    target: PreparedSpectralAxis,
    /,
) -> tuple[np.ndarray, float]:
    if not isinstance(source.plan, ConstrainedBasisPlan) or not isinstance(
        target.plan, ConstrainedBasisPlan
    ):
        raise TypeError("Constrained axis transfer requires constrained plans.")
    source_base = source.plan.base.prepare(source.domain, precision=source.precision)
    target_base = target.plan.base.prepare(target.domain, precision=target.precision)
    if (
        source.modal_transform is None
        or target.modal_transform is None
        or source_base.modal_transform is None
        or target_base.modal_transform is None
    ):
        raise RuntimeError("Constrained transfer requires dense modal metadata.")
    source_to_base = np.asarray(source_base.modal_transform.analysis) @ np.asarray(
        source.modal_transform.synthesis
    )
    base_resize = np.zeros((target_base.mode_count, source_base.mode_count))
    retained = min(target_base.mode_count, source_base.mode_count)
    base_resize[:retained, :retained] = np.eye(retained)
    target_from_base = np.asarray(target.modal_transform.analysis) @ np.asarray(
        target_base.modal_transform.synthesis
    )
    matrix = target_from_base @ base_resize @ source_to_base
    target_base_coefficients = (
        np.asarray(target_base.modal_transform.analysis)
        @ np.asarray(target.modal_transform.synthesis)
        @ matrix
    )
    residual = (
        _constraint_matrix(target_base, target.plan.conditions) @ target_base_coefficients
    )
    return matrix, float(np.max(np.abs(residual), initial=0.0))


__all__ = [
    "PreparedSpectralModalTransfer",
    "SpectralModalTransferPlan",
    "SpectralModalTransferReport",
    "prepare_spectral_modal_transfer",
]
