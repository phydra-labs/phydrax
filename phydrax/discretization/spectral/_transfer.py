#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
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
from ._lattice import LatticeHarmonicDiscretization
from ._rational import (
    RationalChebyshevHalfLineBasisPlan,
    RationalChebyshevLineBasisPlan,
)
from ._space import _apply_axis_transform, TensorSpectralDiscretization
from ._spherical import SphericalSpectralDiscretization


class SpectralModalTransferReport(StrictModule, NonTrainableState):
    """Resolution relation, losslessness, trace residual, and transfer work."""

    trace_residual: Array
    source_shape: tuple[int, ...] = eqx.field(static=True)
    target_shape: tuple[int, ...] = eqx.field(static=True)
    axis_actions: tuple[str, ...] = eqx.field(static=True)
    lossless: bool = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    removed_mode_count: int = eqx.field(static=True)
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
        source_space_id: str,
        target_space_id: str,
        removed_mode_count: int,
    ):
        source = tuple(int(size) for size in source_shape)
        target = tuple(int(size) for size in target_shape)
        actions = tuple(str(action) for action in axis_actions)
        workspace = int(workspace_bytes)
        source_space = str(source_space_id)
        target_space = str(target_space_id)
        removed = int(removed_mode_count)
        if (
            not source
            or len(source) != len(target)
            or len(source) != len(actions)
            or any(size <= 0 for size in source + target)
            or any(not action for action in actions)
            or workspace < 0
            or not source_space
            or not target_space
            or removed < 0
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
        self.source_space_id = source_space
        self.target_space_id = target_space
        self.removed_mode_count = removed
        self.report_id = canonical_fingerprint(
            {
                "kind": "spectral-modal-transfer-report",
                "source_shape": list(source),
                "target_shape": list(target),
                "axis_actions": list(actions),
                "lossless": bool(lossless),
                "trace_residual": float(np.asarray(residual)),
                "workspace_bytes": workspace,
                "source_space_id": source_space,
                "target_space_id": target_space,
                "removed_mode_count": removed,
            }
        )


class SpectralModalTransferResult(StrictModule):
    coefficients: Array
    removed_coefficient_energy: Array
    source_space_id: str = eqx.field(static=True)
    target_space_id: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)


class SpectralModalTransferPlan(StrictModule, NonTrainableState):
    """Symbolic modal transfer between compatible prepared spectral spaces."""

    source: (
        TensorSpectralDiscretization
        | SphericalSpectralDiscretization
        | LatticeHarmonicDiscretization
    )
    target: (
        TensorSpectralDiscretization
        | SphericalSpectralDiscretization
        | LatticeHarmonicDiscretization
    )
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: TensorSpectralDiscretization
        | SphericalSpectralDiscretization
        | LatticeHarmonicDiscretization,
        target: TensorSpectralDiscretization
        | SphericalSpectralDiscretization
        | LatticeHarmonicDiscretization,
        /,
    ):
        tensor_pair = isinstance(source, TensorSpectralDiscretization) and isinstance(
            target, TensorSpectralDiscretization
        )
        spherical_pair = isinstance(
            source, SphericalSpectralDiscretization
        ) and isinstance(target, SphericalSpectralDiscretization)
        lattice_pair = isinstance(source, LatticeHarmonicDiscretization) and isinstance(
            target, LatticeHarmonicDiscretization
        )
        if not tensor_pair and not spherical_pair and not lattice_pair:
            raise TypeError("source and target must be one compatible spectral family.")
        if tensor_pair and len(source.axes) != len(target.axes):
            raise ValueError("Spectral transfer ranks must match.")
        if spherical_pair:
            if (
                source.layout.spin != target.layout.spin
                or source.layout.reality != target.layout.reality
                or source.layout.normalization != target.layout.normalization
                or source.radius != target.radius
            ):
                raise ValueError(
                    "Spherical transfers require one spin, reality, normalization, "
                    "and radius."
                )
        if lattice_pair:
            if (
                source.periodic_dimension != target.periodic_dimension
                or source.primitive_vectors.shape != target.primitive_vectors.shape
                or not bool(
                    jnp.allclose(
                        source.primitive_vectors,
                        target.primitive_vectors,
                        rtol=0.0,
                        atol=0.0,
                    )
                )
            ):
                raise ValueError(
                    "Lattice harmonic transfers require identical primitive geometry."
                )
        self.source = source
        self.target = target
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-modal-transfer-plan",
                "source": _preparation_id(source),
                "target": _preparation_id(target),
            }
        )

    def prepare(self, /) -> "PreparedSpectralModalTransfer":
        if isinstance(self.source, SphericalSpectralDiscretization):
            return self._prepare_spherical()
        if isinstance(self.source, LatticeHarmonicDiscretization):
            return self._prepare_lattice()
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
            source_space_id=source.modal_space.field_space_id,
            target_space_id=target.modal_space.field_space_id,
            removed_mode_count=max(
                int(np.prod(source.modal_shape))
                - int(
                    np.prod(
                        tuple(
                            min(source_size, target_size)
                            for source_size, target_size in zip(
                                source.modal_shape,
                                target.modal_shape,
                                strict=True,
                            )
                        )
                    )
                ),
                0,
            ),
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

    def _prepare_spherical(self, /) -> "PreparedSpectralModalTransfer":
        source = self.source
        target = self.target
        if not isinstance(source, SphericalSpectralDiscretization) or not isinstance(
            target, SphericalSpectralDiscretization
        ):
            raise TypeError("Spherical transfer preparation requires spherical spaces.")
        source_offset = source.layout.bandlimit - 1
        target_offset = target.layout.bandlimit - 1
        retained_limit = min(source.layout.bandlimit, target.layout.bandlimit)

        def action(coefficients):
            values = jnp.asarray(coefficients)
            if values.shape != source.coefficient_shape:
                raise ValueError(
                    "Transferred spherical coefficients must match the modal shape."
                )
            values = source.layout.mask_invalid(values)
            result = jnp.zeros(target.coefficient_shape, dtype=values.dtype)
            for degree in range(retained_limit):
                source_slice = slice(
                    source_offset - degree,
                    source_offset + degree + 1,
                )
                target_slice = slice(
                    target_offset - degree,
                    target_offset + degree + 1,
                )
                result = result.at[degree, target_slice].set(values[degree, source_slice])
            result = target.layout.mask_invalid(result)
            if target.layout.reality:
                result = target.layout.canonicalize_reality(result)
            return result

        operator = FunctionLinearOperator(
            action,
            source=source.modal_space.vector_space,
            target=target.modal_space.vector_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "spherical-modal-transfer-operator",
                    "plan": self.plan_id,
                    "source_layout": source.layout.layout_id,
                    "target_layout": target.layout.layout_id,
                }
            ),
        )
        source_modes = source.layout.logical_mode_count
        retained_modes = sum(source.layout.level_multiplicities[:retained_limit])
        removed_modes = max(source_modes - retained_modes, 0)
        itemsize = np.dtype(source.plan.precision.coefficient_dtype).itemsize
        report = SpectralModalTransferReport(
            source_shape=source.coefficient_shape,
            target_shape=target.coefficient_shape,
            axis_actions=("spherical-degree-map", "spherical-order-map"),
            lossless=target.layout.bandlimit >= source.layout.bandlimit,
            trace_residual=0.0,
            workspace_bytes=max(
                math.prod(source.coefficient_shape),
                math.prod(target.coefficient_shape),
            )
            * itemsize,
            source_space_id=source.modal_space.field_space_id,
            target_space_id=target.modal_space.field_space_id,
            removed_mode_count=removed_modes,
        )
        preparation = PreparationReport(
            capabilities=(DiscretizationCapability.FIELD_TRANSFER,),
            diagnostics=(
                f"lossless:{report.lossless}",
                f"removed_modes:{removed_modes}",
                f"spin:{source.layout.spin}",
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
                exact_on=("retained-spherical-modal-subspace",),
            ),
            preparation=preparation,
        )
        return PreparedSpectralModalTransfer(self, transfer, report)

    def _prepare_lattice(self, /) -> "PreparedSpectralModalTransfer":
        source = self.source
        target = self.target
        if not isinstance(source, LatticeHarmonicDiscretization) or not isinstance(
            target, LatticeHarmonicDiscretization
        ):
            raise TypeError("Lattice transfer preparation requires lattice spaces.")
        target_lookup = {
            mode_id: mode_index
            for mode_index, mode_id in enumerate(target.plan.layout.mode_ids)
        }
        source_indices = []
        target_indices = []
        for source_index, mode_id in enumerate(source.plan.layout.mode_ids):
            if mode_id in target_lookup:
                source_indices.append(source_index)
                target_indices.append(target_lookup[mode_id])
        source_indices_array = jnp.asarray(source_indices, dtype=jnp.int32)
        target_indices_array = jnp.asarray(target_indices, dtype=jnp.int32)

        def action(coefficients):
            values = jnp.asarray(coefficients)
            if values.shape != (source.harmonic_count,):
                raise ValueError(
                    "Transferred lattice coefficients must match harmonic_count."
                )
            result = jnp.zeros((target.harmonic_count,), dtype=values.dtype)
            return result.at[target_indices_array].set(values[source_indices_array])

        operator = FunctionLinearOperator(
            action,
            source=source.modal_space,
            target=target.modal_space,
            operator_id=canonical_fingerprint(
                {
                    "kind": "lattice-harmonic-modal-transfer-operator",
                    "plan": self.plan_id,
                    "source_layout": source.plan.layout.layout_id,
                    "target_layout": target.plan.layout.layout_id,
                }
            ),
        )
        removed_modes = source.harmonic_count - len(source_indices)
        itemsize = np.dtype(source.plan.precision.coefficient_dtype).itemsize
        report = SpectralModalTransferReport(
            source_shape=(source.harmonic_count,),
            target_shape=(target.harmonic_count,),
            axis_actions=("lattice-mode-id-map",),
            lossless=removed_modes == 0,
            trace_residual=0.0,
            workspace_bytes=max(source.harmonic_count, target.harmonic_count) * itemsize,
            source_space_id=source.modal_space.space_id,
            target_space_id=target.modal_space.space_id,
            removed_mode_count=removed_modes,
        )
        return PreparedSpectralModalTransfer(self, operator, report)


class PreparedSpectralModalTransfer(StrictModule, NonTrainableState):
    plan: SpectralModalTransferPlan
    transfer: FieldTransfer | FunctionLinearOperator
    report: SpectralModalTransferReport
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralModalTransferPlan,
        transfer: FieldTransfer | FunctionLinearOperator,
        report: SpectralModalTransferReport,
        /,
    ):
        if not isinstance(plan, SpectralModalTransferPlan):
            raise TypeError("plan must be a SpectralModalTransferPlan.")
        if not isinstance(transfer, (FieldTransfer, FunctionLinearOperator)):
            raise TypeError("transfer must be a FieldTransfer or FunctionLinearOperator.")
        if not isinstance(report, SpectralModalTransferReport):
            raise TypeError("report must be a SpectralModalTransferReport.")
        self.plan = plan
        self.transfer = transfer
        self.report = report
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-modal-transfer",
                "plan": plan.plan_id,
                "transfer": (
                    transfer.transfer_id
                    if isinstance(transfer, FieldTransfer)
                    else transfer.operator_id
                ),
                "report": report.report_id,
            }
        )

    def __call__(self, coefficients: ArrayLike, /) -> Array:
        source = self.plan.source
        source_shape = _modal_shape(source)
        target_shape = _modal_shape(self.plan.target)
        values = _validate_modal_coefficients(
            source,
            coefficients,
            "Transferred modal coefficients",
        )
        operator = (
            self.transfer.operator
            if isinstance(self.transfer, FieldTransfer)
            else self.transfer
        )
        payload_shape = values.shape[len(source_shape) :]
        if not payload_shape:
            return operator.mv(values)
        flattened = values.reshape(source_shape + (-1,))
        transferred = jax.vmap(
            operator.mv,
            in_axes=-1,
            out_axes=-1,
        )(flattened)
        return transferred.reshape(target_shape + payload_shape)

    def apply_with_evidence(
        self, coefficients: ArrayLike, /
    ) -> SpectralModalTransferResult:
        values = _validate_modal_coefficients(
            self.plan.source,
            coefficients,
            "Transferred modal coefficients",
        )
        transferred = self(values)
        source_energy = jnp.sum(jnp.abs(values) ** 2)
        retained_energy = jnp.sum(jnp.abs(transferred) ** 2)
        removed = jnp.maximum(
            source_energy - retained_energy,
            jnp.asarray(0.0, dtype=source_energy.dtype),
        )
        return SpectralModalTransferResult(
            coefficients=transferred,
            removed_coefficient_energy=removed,
            source_space_id=self.report.source_space_id,
            target_space_id=self.report.target_space_id,
            report_id=self.report.report_id,
        )


def prepare_spectral_modal_transfer(
    source: TensorSpectralDiscretization
    | SphericalSpectralDiscretization
    | LatticeHarmonicDiscretization,
    target: TensorSpectralDiscretization
    | SphericalSpectralDiscretization
    | LatticeHarmonicDiscretization,
    /,
) -> PreparedSpectralModalTransfer:
    return SpectralModalTransferPlan(source, target).prepare()


def _preparation_id(
    discretization: TensorSpectralDiscretization
    | SphericalSpectralDiscretization
    | LatticeHarmonicDiscretization,
    /,
) -> str:
    if isinstance(discretization, LatticeHarmonicDiscretization):
        return discretization.preparation_id
    return discretization.prepared_id


def _modal_shape(
    discretization: TensorSpectralDiscretization
    | SphericalSpectralDiscretization
    | LatticeHarmonicDiscretization,
    /,
) -> tuple[int, ...]:
    if isinstance(discretization, TensorSpectralDiscretization):
        return discretization.modal_shape
    if isinstance(discretization, SphericalSpectralDiscretization):
        return discretization.coefficient_shape
    return (discretization.harmonic_count,)


def _validate_modal_coefficients(
    discretization: TensorSpectralDiscretization
    | SphericalSpectralDiscretization
    | LatticeHarmonicDiscretization,
    coefficients: ArrayLike,
    name: str,
    /,
) -> Array:
    shape = _modal_shape(discretization)
    if isinstance(discretization, TensorSpectralDiscretization):
        return discretization._validate_leading(coefficients, shape, name)
    if isinstance(discretization, LatticeHarmonicDiscretization):
        values = jnp.asarray(coefficients)
        if values.ndim < 1 or values.shape[0] != discretization.harmonic_count:
            raise ValueError(
                f"{name} must begin with ({discretization.harmonic_count},); "
                f"got {values.shape}."
            )
        return values
    values = jnp.asarray(coefficients)
    if values.ndim < len(shape) or tuple(values.shape[: len(shape)]) != shape:
        raise ValueError(f"{name} must begin with shape {shape}; got {values.shape}.")
    return discretization.layout.mask_invalid(values)


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
    "SpectralModalTransferResult",
    "SpectralModalTransferReport",
    "prepare_spectral_modal_transfer",
]
