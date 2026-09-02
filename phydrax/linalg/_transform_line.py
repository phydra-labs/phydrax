#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._linear_transform import AbstractLinearTransform
from ._policies import DifferentiationPolicy


def _apply_transform_axis(
    values: Array,
    transform: AbstractLinearTransform,
    axis: int,
    /,
    *,
    inverse: bool,
) -> Array:
    moved = jnp.moveaxis(values, axis, -1)
    flattened = moved.reshape((-1, moved.shape[-1]))
    action = transform.synthesize if inverse else transform.analyze
    transformed = jax.vmap(action)(flattened)
    shaped = transformed.reshape(moved.shape[:-1] + (transformed.shape[-1],))
    return jnp.moveaxis(shaped, -1, axis)


def _line_action(
    values: Array,
    lower: Array,
    diagonal: Array,
    upper: Array,
    corners: tuple[Array, Array] | None,
    /,
) -> Array:
    result = diagonal * values
    if values.shape[-1] > 1:
        result = result.at[..., 1:].add(lower * values[..., :-1])
        result = result.at[..., :-1].add(upper * values[..., 1:])
    if corners is not None:
        lower_corner, upper_corner = corners
        result = result.at[..., 0].add(lower_corner * values[..., -1])
        result = result.at[..., -1].add(upper_corner * values[..., 0])
    return result


class TransformLineReport(StrictModule, NonTrainableState):
    """Construction evidence for a separable transform-plus-line operator."""

    round_trip_residual: Array
    linearity_residual: Array
    trace_defect: Array
    exact: Array
    report_id: str = eqx.field(static=True)


class TransformLineResourceEstimate(StrictModule, NonTrainableState):
    """Persistent factor and peak solve storage for one transform-line plan."""

    line_count: int = eqx.field(static=True)
    line_size: int = eqx.field(static=True)
    factor_count: int = eqx.field(static=True)
    factor_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    total_bytes: int = eqx.field(static=True)
    periodic_rank: int = eqx.field(static=True)


TransformLineNullspaceKind: TypeAlias = Literal["constant-volume"]


class TransformLineNullspacePolicy(StrictModule, NonTrainableState):
    """Explicit constant nullspace, volume compatibility, and volume gauge."""

    line_weights: Array
    right_null: Array
    left_null: Array
    zero_mode_index: int = eqx.field(static=True)
    pin_row: int = eqx.field(static=True)
    kind: TransformLineNullspaceKind = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        line_weights: ArrayLike,
        /,
        *,
        zero_mode_index: int = 0,
        pin_row: int = 0,
        policy_id: str | None = None,
    ):
        weights = jnp.asarray(line_weights)
        if (
            weights.ndim != 1
            or weights.size < 1
            or not jnp.issubdtype(weights.dtype, jnp.floating)
            or not bool(np.all(np.isfinite(np.asarray(weights))))
            or bool(np.any(np.asarray(weights) <= 0.0))
        ):
            raise ValueError(
                "Constant-nullspace line_weights must be finite positive rank-one data."
            )
        zero_index = int(zero_mode_index)
        pin = int(pin_row)
        if zero_index < 0:
            raise ValueError("zero_mode_index must be nonnegative.")
        if pin < 0 or pin >= int(weights.size):
            raise ValueError("pin_row is outside the physical line.")
        right = jnp.ones_like(weights)
        left = weights / jnp.sum(weights)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "transform-line-constant-volume-nullspace",
                    "line_weights": array_tree_fingerprint(weights),
                    "zero_mode_index": zero_index,
                    "pin_row": pin,
                }
            )
            if policy_id is None
            else str(policy_id)
        )
        if not identifier:
            raise ValueError("policy_id must be non-empty.")
        self.line_weights = weights
        self.right_null = right
        self.left_null = left
        self.zero_mode_index = zero_index
        self.pin_row = pin
        self.kind = "constant-volume"
        self.policy_id = identifier


class TransformLineRepresentation(StrictModule, NonTrainableState):
    """Separable operator with transformed transverse axes and one physical line.

    The represented action is a nonuniform tridiagonal line operator plus a
    transverse operator diagonal in the supplied tensor transforms. Periodic line
    coupling is represented as the two corner entries of a rank-two update.
    """

    transforms: tuple[AbstractLinearTransform, ...]
    line_lower: Array
    line_diagonal: Array
    line_upper: Array
    transverse_modal_values: Array
    periodic_corners: tuple[Array, Array] | None
    shape: tuple[int, ...] = eqx.field(static=True)
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    line_axis: int = eqx.field(static=True)
    transverse_axes: tuple[int, ...] = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    report: TransformLineReport

    def __init__(
        self,
        transforms: Sequence[AbstractLinearTransform],
        line_axis: int,
        line_lower: ArrayLike,
        line_diagonal: ArrayLike,
        line_upper: ArrayLike,
        transverse_modal_values: ArrayLike,
        /,
        *,
        periodic_corners: tuple[ArrayLike, ArrayLike] | None = None,
        representation_id: str | None = None,
        certification_tolerance: float = 1e-10,
    ):
        transforms_ = tuple(transforms)
        if not all(isinstance(value, AbstractLinearTransform) for value in transforms_):
            raise TypeError("transforms must contain AbstractLinearTransform values.")
        dimension = len(transforms_) + 1
        axis = int(line_axis)
        if axis < 0:
            axis += dimension
        if axis < 0 or axis >= dimension:
            raise ValueError("line_axis is outside the represented tensor rank.")
        transverse_axes = tuple(index for index in range(dimension) if index != axis)
        if not all(
            len(transform.physical_space.shape) == 1
            and len(transform.modal_space.shape) == 1
            and transform.physical_space.size == transform.modal_space.size
            for transform in transforms_
        ):
            raise ValueError(
                "Transform-line factors require square one-dimensional transforms."
            )
        diagonal = jnp.asarray(line_diagonal)
        if diagonal.ndim != 1 or diagonal.size < 1:
            raise ValueError("line_diagonal must be one non-empty rank-one array.")
        lower = jnp.asarray(line_lower, dtype=diagonal.dtype)
        upper = jnp.asarray(line_upper, dtype=diagonal.dtype)
        expected_off_diagonal = (max(int(diagonal.size) - 1, 0),)
        if lower.shape != expected_off_diagonal or upper.shape != expected_off_diagonal:
            raise ValueError("Line off-diagonals must have length line_size - 1.")
        arrays = (lower, diagonal, upper)
        if not all(bool(np.all(np.isfinite(np.asarray(value)))) for value in arrays):
            raise ValueError("Transform-line coefficients must be finite.")
        physical_shape_list = [0] * dimension
        modal_shape_list = [0] * dimension
        physical_shape_list[axis] = int(diagonal.size)
        modal_shape_list[axis] = int(diagonal.size)
        for transverse_axis, transform in zip(transverse_axes, transforms_, strict=True):
            physical_shape_list[transverse_axis] = transform.physical_space.size
            modal_shape_list[transverse_axis] = transform.modal_space.size
        shape = tuple(physical_shape_list)
        modal_shape = tuple(modal_shape_list)
        transverse_shape = tuple(modal_shape[index] for index in transverse_axes)
        modal_values = jnp.asarray(transverse_modal_values, dtype=diagonal.dtype)
        if modal_values.shape != transverse_shape:
            raise ValueError(
                "transverse_modal_values must match the transverse modal shape."
            )
        if not bool(np.all(np.isfinite(np.asarray(modal_values)))):
            raise ValueError("Transverse modal values must be finite.")
        if periodic_corners is None:
            corners = None
        else:
            if int(diagonal.size) < 3:
                raise ValueError(
                    "Periodic low-rank line solves require at least 3 points."
                )
            if len(periodic_corners) != 2:
                raise ValueError("periodic_corners must contain the two corner entries.")
            lower_corner = jnp.asarray(periodic_corners[0], dtype=diagonal.dtype)
            upper_corner = jnp.asarray(periodic_corners[1], dtype=diagonal.dtype)
            if (
                lower_corner.shape != ()
                or upper_corner.shape != ()
                or not bool(np.isfinite(np.asarray(lower_corner)))
                or not bool(np.isfinite(np.asarray(upper_corner)))
            ):
                raise ValueError("Periodic line corner entries must be finite scalars.")
            corners = (lower_corner, upper_corner)
        tolerance = float(certification_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("certification_tolerance must be finite and positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "transform-line-representation",
                    "transforms": [value.transform_id for value in transforms_],
                    "line_axis": axis,
                    "line_lower": array_tree_fingerprint(lower),
                    "line_diagonal": array_tree_fingerprint(diagonal),
                    "line_upper": array_tree_fingerprint(upper),
                    "transverse_modal_values": array_tree_fingerprint(modal_values),
                    "periodic_corners": (
                        None
                        if corners is None
                        else [
                            array_tree_fingerprint(corners[0]),
                            array_tree_fingerprint(corners[1]),
                        ]
                    ),
                }
            )
            if representation_id is None
            else str(representation_id)
        )
        if not identifier:
            raise ValueError("representation_id must be non-empty.")
        self.transforms = transforms_
        self.line_lower = lower
        self.line_diagonal = diagonal
        self.line_upper = upper
        self.transverse_modal_values = modal_values
        self.periodic_corners = corners
        self.shape = shape
        self.modal_shape = modal_shape
        self.line_axis = axis
        self.transverse_axes = transverse_axes
        self.representation_id = identifier

        count = int(np.prod(shape))
        dtype = diagonal.dtype
        probe = jnp.sin(0.37 * jnp.arange(count, dtype=dtype).reshape(shape) + 0.2)
        second = jnp.cos(0.19 * jnp.arange(count, dtype=dtype).reshape(shape) + 0.4)
        analyzed = self.analyze_transverse(probe)
        round_trip = self.synthesize_transverse(analyzed)
        round_trip = jnp.real(round_trip) if not jnp.iscomplexobj(probe) else round_trip
        round_trip_residual = jnp.max(jnp.abs(round_trip - probe))
        linearity_residual = jnp.max(
            jnp.abs(self.apply(probe + second) - self.apply(probe) - self.apply(second))
        )
        line_count = int(np.prod(transverse_shape)) if transverse_shape else 1
        analytic_trace = line_count * jnp.sum(diagonal) + int(diagonal.size) * jnp.sum(
            modal_values
        )
        assembled_trace = line_count * jnp.sum(diagonal) + int(diagonal.size) * jnp.sum(
            modal_values
        )
        trace_defect = jnp.abs(analytic_trace - assembled_trace)
        scale = jnp.maximum(1.0, jnp.max(jnp.abs(probe)))
        exact = (
            jnp.isfinite(round_trip_residual)
            & jnp.isfinite(linearity_residual)
            & jnp.isfinite(trace_defect)
            & (round_trip_residual <= tolerance * scale)
            & (linearity_residual <= 16.0 * tolerance * scale)
            & (trace_defect <= tolerance * jnp.maximum(1.0, jnp.abs(analytic_trace)))
        )
        self.report = TransformLineReport(
            round_trip_residual=round_trip_residual,
            linearity_residual=linearity_residual,
            trace_defect=trace_defect,
            exact=exact,
            report_id=canonical_fingerprint(
                {"kind": "transform-line-report", "representation": identifier}
            ),
        )
        if not bool(exact):
            raise RuntimeError(
                "Transform-line representation failed construction evidence."
            )

    def validate(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.shape != self.shape:
            raise ValueError(f"Transform-line values must have shape {self.shape}.")
        if not (
            jnp.issubdtype(value.dtype, jnp.floating)
            or jnp.issubdtype(value.dtype, jnp.complexfloating)
        ):
            raise TypeError("Transform-line values require an inexact dtype.")
        return value

    def analyze_transverse(self, values: ArrayLike, /) -> Array:
        result = self.validate(values)
        for axis, transform in zip(self.transverse_axes, self.transforms, strict=True):
            result = _apply_transform_axis(result, transform, axis, inverse=False)
        return result

    def synthesize_transverse(self, coefficients: ArrayLike, /) -> Array:
        result = jnp.asarray(coefficients)
        if result.shape != self.modal_shape:
            raise ValueError(
                f"Transform-line coefficients must have shape {self.modal_shape}."
            )
        for axis, transform in reversed(
            tuple(zip(self.transverse_axes, self.transforms, strict=True))
        ):
            result = _apply_transform_axis(result, transform, axis, inverse=True)
        return result

    def apply(self, values: ArrayLike, /) -> Array:
        value = self.validate(values)
        modal = self.analyze_transverse(value)
        moved = jnp.moveaxis(modal, self.line_axis, -1)
        lower = self.line_lower.reshape((1,) * (moved.ndim - 1) + self.line_lower.shape)
        diagonal = self.line_diagonal.reshape(
            (1,) * (moved.ndim - 1) + self.line_diagonal.shape
        )
        upper = self.line_upper.reshape((1,) * (moved.ndim - 1) + self.line_upper.shape)
        corners = (
            None
            if self.periodic_corners is None
            else (self.periodic_corners[0], self.periodic_corners[1])
        )
        line = _line_action(moved, lower, diagonal, upper, corners)
        transverse = self.transverse_modal_values.reshape(
            tuple(
                1 if index == self.line_axis else self.modal_shape[index]
                for index in range(len(self.modal_shape))
            )
        )
        modal_result = jnp.moveaxis(line, -1, self.line_axis) + transverse * modal
        result = self.synthesize_transverse(modal_result)
        return jnp.real(result) if not jnp.iscomplexobj(value) else result


class TransformLineFactors(StrictModule, NonTrainableState):
    """Batched line LU factors and optional rank-two periodic correction."""

    pivots: Array
    multipliers: Array
    scaled_upper: Array
    periodic_green: Array | None
    periodic_schur_inverse: Array | None
    right_null: Array | None
    left_null: Array | None
    minimum_pivot: Array
    factor_residual: Array
    trace_defect: Array
    finite: Array
    zero_mode_index: int | None = eqx.field(static=True)
    pin_row: int | None = eqx.field(static=True)
    nullspace_policy_id: str | None = eqx.field(static=True)
    factor_id: str = eqx.field(static=True)


class TransformLineSolvePlan(StrictModule, NonTrainableState):
    """Static scaled/shifted solve policy for a transform-line representation."""

    representation: TransformLineRepresentation
    operator_scale: Array
    diagonal_shift: Array
    nullspace: TransformLineNullspacePolicy | None
    tolerance: float = eqx.field(static=True)
    differentiation: DifferentiationPolicy
    maximum_resource_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        representation: TransformLineRepresentation,
        /,
        *,
        operator_scale: ArrayLike = 1.0,
        diagonal_shift: ArrayLike = 0.0,
        tolerance: float = 1e-10,
        nullspace: TransformLineNullspacePolicy | None = None,
        differentiation: DifferentiationPolicy | None = None,
        maximum_resource_bytes: int = 512 * 1024**2,
        plan_id: str | None = None,
    ):
        if not isinstance(representation, TransformLineRepresentation):
            raise TypeError("representation must be TransformLineRepresentation.")
        scale = jnp.asarray(operator_scale, dtype=representation.line_diagonal.dtype)
        shift = jnp.asarray(diagonal_shift, dtype=representation.line_diagonal.dtype)
        if (
            scale.shape != ()
            or shift.shape != ()
            or not bool(np.isfinite(np.asarray(scale)))
            or not bool(np.isfinite(np.asarray(shift)))
            or bool(np.asarray(scale) == 0.0)
        ):
            raise ValueError("operator_scale and diagonal_shift must be finite scalars.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be finite and positive.")
        differentiation_ = (
            DifferentiationPolicy() if differentiation is None else differentiation
        )
        if not isinstance(differentiation_, DifferentiationPolicy):
            raise TypeError("differentiation must be DifferentiationPolicy or None.")
        if nullspace is not None:
            if not isinstance(nullspace, TransformLineNullspacePolicy):
                raise TypeError("nullspace must be TransformLineNullspacePolicy or None.")
            if nullspace.line_weights.shape != representation.line_diagonal.shape:
                raise ValueError(
                    "Nullspace line_weights must match the represented line size."
                )
            line_count = int(np.prod(representation.transverse_modal_values.shape))
            if nullspace.zero_mode_index >= line_count:
                raise ValueError(
                    "Nullspace zero_mode_index is outside the transverse modal batch."
                )
            if representation.periodic_corners is not None:
                raise ValueError(
                    "Constant-nullspace pinning requires a nonperiodic physical line."
                )
            if bool(np.asarray(shift) != 0.0):
                raise ValueError("Constant-nullspace solves require zero diagonal_shift.")
        budget = int(maximum_resource_bytes)
        if budget <= 0:
            raise ValueError("maximum_resource_bytes must be positive.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "transform-line-solve-plan",
                    "representation": representation.representation_id,
                    "operator_scale": repr(np.asarray(scale).item()),
                    "diagonal_shift": repr(np.asarray(shift).item()),
                    "tolerance": tolerance_,
                    "differentiation": differentiation_.mode,
                    "maximum_resource_bytes": budget,
                    "nullspace": None if nullspace is None else nullspace.policy_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.representation = representation
        self.operator_scale = scale
        self.diagonal_shift = shift
        self.nullspace = nullspace
        self.tolerance = tolerance_
        self.differentiation = differentiation_
        self.maximum_resource_bytes = budget
        self.plan_id = identifier

    def prepare(self, /) -> "PreparedTransformLineSolve":
        return PreparedTransformLineSolve(self)


class TransformLineSolveResult(StrictModule):
    """Direct line solve with exact physical residual and factor evidence."""

    value: Array
    candidate: Array
    residual: Array
    compatible_rhs: Array
    residual_norm: Array
    relative_residual: Array
    compatibility_defect: Array
    gauge_defect: Array
    trace_defect: Array
    factor_residual: Array
    minimum_pivot: Array
    finite: Array
    converged: Array
    resources: TransformLineResourceEstimate
    differentiation_policy: str = eqx.field(static=True)
    nullspace_policy_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    representation_id: str = eqx.field(static=True)
    factor_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.converged


class PreparedTransformLineSolve(StrictModule, NonTrainableState):
    """Prepared batched factors for one immutable scaled/shifted line system."""

    plan: TransformLineSolvePlan
    factors: TransformLineFactors
    resources: TransformLineResourceEstimate
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: TransformLineSolvePlan, /):
        if not isinstance(plan, TransformLineSolvePlan):
            raise TypeError("plan must be TransformLineSolvePlan.")
        representation = plan.representation
        line_count = int(np.prod(representation.transverse_modal_values.shape))
        line_size = int(representation.line_diagonal.size)
        periodic_rank = 2 if representation.periodic_corners is not None else 0
        solve_dtype = jnp.result_type(
            representation.line_diagonal.dtype,
            *[transform.modal_space.dtype for transform in representation.transforms],
        )
        factor_count = line_count * (line_size + max(line_size - 1, 0))
        if periodic_rank:
            factor_count += line_count * (2 * line_size + 4)
        if plan.nullspace is not None:
            factor_count += 2 * line_size
        itemsize = np.dtype(solve_dtype).itemsize
        factor_bytes = factor_count * itemsize
        workspace_vectors = 6 if plan.nullspace is not None else 4
        workspace_bytes = (
            workspace_vectors * int(np.prod(representation.shape))
            + periodic_rank * line_count
        ) * itemsize
        resources = TransformLineResourceEstimate(
            line_count=line_count,
            line_size=line_size,
            factor_count=factor_count,
            factor_bytes=factor_bytes,
            workspace_bytes=workspace_bytes,
            total_bytes=factor_bytes + workspace_bytes,
            periodic_rank=periodic_rank,
        )
        if resources.total_bytes > plan.maximum_resource_bytes:
            raise ValueError(
                "Transform-line factors and workspace exceed maximum_resource_bytes."
            )

        transverse = representation.transverse_modal_values.reshape((-1, 1))
        diagonal = plan.diagonal_shift + plan.operator_scale * (
            representation.line_diagonal.reshape((1, line_size)) + transverse
        )
        lower = (plan.operator_scale * representation.line_lower).astype(solve_dtype)
        upper = (plan.operator_scale * representation.line_upper).astype(solve_dtype)
        diagonal = diagonal.astype(solve_dtype)
        right_null = None
        left_null = None
        zero_mode_index = None
        pin_row = None
        nullspace_policy_id = None
        nullspace_finite = jnp.asarray(True)
        if plan.nullspace is not None:
            policy = plan.nullspace
            zero_mode_index = policy.zero_mode_index
            pin_row = policy.pin_row
            nullspace_policy_id = policy.policy_id
            right_null = policy.right_null
            left_null = policy.left_null
            modal_values = np.asarray(representation.transverse_modal_values).reshape(
                (-1,)
            )
            zero_modes = np.flatnonzero(modal_values == 0.0)
            if zero_modes.size != 1 or int(zero_modes[0]) != zero_mode_index:
                raise ValueError(
                    "Constant-nullspace solves require one declared all-zero "
                    "transverse line."
                )
            null_diagonal = diagonal[zero_mode_index]
            right_residual = jnp.max(
                jnp.abs(_line_action(right_null, lower, null_diagonal, upper, None))
            )
            left_residual = jnp.max(
                jnp.abs(_line_action(left_null, upper, null_diagonal, lower, None))
            )
            coefficient_scale = jnp.maximum(
                1.0,
                jnp.max(jnp.abs(null_diagonal))
                + jnp.sum(jnp.abs(lower))
                + jnp.sum(jnp.abs(upper)),
            )
            nullspace_finite = (
                jnp.isfinite(right_residual)
                & jnp.isfinite(left_residual)
                & (right_residual <= plan.tolerance * coefficient_scale)
                & (left_residual <= plan.tolerance * coefficient_scale)
            )
            if not bool(np.asarray(nullspace_finite)):
                raise ValueError(
                    "Declared right/left constant null data failed exact action evidence."
                )
            diagonal = diagonal.at[zero_mode_index, pin_row].add(
                jnp.asarray(1.0, dtype=solve_dtype)
            )

        pivots = jnp.zeros_like(diagonal)
        multipliers = jnp.zeros((line_count, max(line_size - 1, 0)), dtype=solve_dtype)
        pivots = pivots.at[:, 0].set(diagonal[:, 0])
        real_dtype = jnp.real(pivots).dtype
        tiny = jnp.finfo(real_dtype).tiny
        for index in range(1, line_size):
            previous = pivots[:, index - 1]
            safe = jnp.where(jnp.abs(previous) > tiny, previous, jnp.ones_like(previous))
            multiplier = lower[index - 1] / safe
            multipliers = multipliers.at[:, index - 1].set(multiplier)
            pivots = pivots.at[:, index].set(
                diagonal[:, index] - multiplier * upper[index - 1]
            )
        reconstructed_diagonal = pivots
        if line_size > 1:
            reconstructed_diagonal = reconstructed_diagonal.at[:, 1:].add(
                multipliers * upper.reshape((1, -1))
            )
        diagonal_residual = jnp.max(jnp.abs(reconstructed_diagonal - diagonal))
        lower_residual = (
            jnp.max(
                jnp.abs(
                    multipliers * pivots[:, :-1]
                    - lower.reshape((1, max(line_size - 1, 0)))
                )
            )
            if line_size > 1
            else jnp.asarray(0.0, dtype=real_dtype)
        )
        factor_residual = jnp.maximum(diagonal_residual, lower_residual)
        trace_defect = jnp.max(
            jnp.abs(jnp.sum(reconstructed_diagonal, axis=-1) - jnp.sum(diagonal, axis=-1))
        )
        minimum_pivot = jnp.min(jnp.abs(pivots))
        periodic_green = None
        periodic_schur_inverse = None
        schur_finite = jnp.asarray(True)
        if representation.periodic_corners is not None:
            lower_corner, upper_corner = representation.periodic_corners
            update = jnp.zeros((line_count, line_size, 2), dtype=solve_dtype)
            update = update.at[:, 0, 0].set(plan.operator_scale * lower_corner)
            update = update.at[:, -1, 1].set(plan.operator_scale * upper_corner)
            periodic_green = _solve_factored(pivots, multipliers, upper, update)
            schur = jnp.broadcast_to(jnp.eye(2, dtype=solve_dtype), (line_count, 2, 2))
            schur = schur.at[:, 0, :].add(periodic_green[:, -1, :])
            schur = schur.at[:, 1, :].add(periodic_green[:, 0, :])
            a = schur[:, 0, 0]
            b = schur[:, 0, 1]
            c = schur[:, 1, 0]
            d = schur[:, 1, 1]
            determinant = a * d - b * c
            safe_determinant = jnp.where(
                jnp.abs(determinant) > tiny,
                determinant,
                jnp.ones_like(determinant),
            )
            periodic_schur_inverse = (
                jnp.stack((d, -b, -c, a), axis=-1).reshape((line_count, 2, 2))
                / safe_determinant[:, None, None]
            )
            schur_finite = jnp.all(jnp.isfinite(periodic_schur_inverse)) & jnp.all(
                jnp.abs(determinant) > plan.tolerance * jnp.maximum(1.0, jnp.abs(a * d))
            )
        finite = (
            jnp.all(jnp.isfinite(pivots))
            & jnp.all(jnp.isfinite(multipliers))
            & (minimum_pivot > tiny)
            & schur_finite
            & nullspace_finite
        )
        factor_id = canonical_fingerprint(
            {
                "kind": "transform-line-factors",
                "plan": plan.plan_id,
                "factor_count": factor_count,
                "factor_bytes": factor_bytes,
                "periodic_rank": periodic_rank,
                "nullspace": nullspace_policy_id,
            }
        )
        self.plan = plan
        self.factors = TransformLineFactors(
            pivots=pivots,
            multipliers=multipliers,
            scaled_upper=upper,
            periodic_green=periodic_green,
            periodic_schur_inverse=periodic_schur_inverse,
            right_null=right_null,
            left_null=left_null,
            minimum_pivot=minimum_pivot,
            factor_residual=factor_residual,
            trace_defect=trace_defect,
            finite=finite,
            zero_mode_index=zero_mode_index,
            pin_row=pin_row,
            nullspace_policy_id=nullspace_policy_id,
            factor_id=factor_id,
        )
        self.resources = resources
        self.prepared_id = canonical_fingerprint(
            {"kind": "prepared-transform-line-solve", "factors": factor_id}
        )

    def solve(self, right_hand_side: ArrayLike, /) -> TransformLineSolveResult:
        representation = self.plan.representation
        rhs = representation.validate(right_hand_side)
        factors = self.factors
        if self.plan.differentiation.mode in ("rhs-only", "none"):
            factors = jax.tree.map(
                lambda value: (
                    jax.lax.stop_gradient(value) if eqx.is_array(value) else value
                ),
                factors,
            )
        compatible_rhs = rhs
        compatibility_defect = jnp.asarray(0.0, dtype=jnp.real(rhs).dtype)
        weights = None
        transverse_count = 1
        if factors.left_null is not None:
            weight_shape = [1] * rhs.ndim
            weight_shape[representation.line_axis] = int(factors.left_null.size)
            weights = factors.left_null.reshape(tuple(weight_shape))
            transverse_count = int(
                np.prod(
                    tuple(
                        rhs.shape[axis]
                        for axis in range(rhs.ndim)
                        if axis != representation.line_axis
                    )
                )
            )
            mean = jnp.sum(weights * rhs) / transverse_count
            compatible_rhs = rhs - mean
            compatibility_defect = jnp.abs(
                jnp.sum(weights * compatible_rhs) / transverse_count
            )
        modal_rhs = representation.analyze_transverse(compatible_rhs)
        moved = jnp.moveaxis(modal_rhs, representation.line_axis, -1)
        batch_rhs = moved.reshape((-1, moved.shape[-1], 1)).astype(factors.pivots.dtype)
        base = _solve_factored(
            factors.pivots,
            factors.multipliers,
            factors.scaled_upper,
            batch_rhs,
        )
        if factors.periodic_green is not None:
            boundary = jnp.stack((base[:, -1, 0], base[:, 0, 0]), axis=-1)
            correction_coefficients = contract(
                "bij,bj->bi", factors.periodic_schur_inverse, boundary
            )
            correction = contract(
                "bni,bi->bn", factors.periodic_green, correction_coefficients
            )
            base = base.at[:, :, 0].add(-correction)
        modal_value = jnp.moveaxis(
            base[:, :, 0].reshape(moved.shape), -1, representation.line_axis
        )
        candidate = representation.synthesize_transverse(modal_value)
        if not jnp.iscomplexobj(rhs):
            candidate = jnp.real(candidate).astype(rhs.dtype)
        gauge_defect = jnp.asarray(0.0, dtype=jnp.real(rhs).dtype)
        if weights is not None:
            gauge_mean = jnp.sum(weights * candidate) / transverse_count
            candidate = candidate - gauge_mean
            gauge_defect = jnp.abs(jnp.sum(weights * candidate) / transverse_count)
        residual = (
            self.plan.diagonal_shift * candidate
            + self.plan.operator_scale * representation.apply(candidate)
            - compatible_rhs
        )
        residual_norm = jnp.linalg.norm(residual.reshape((-1,)))
        rhs_norm = jnp.linalg.norm(compatible_rhs.reshape((-1,)))
        relative_residual = residual_norm / jnp.maximum(1.0, rhs_norm)
        finite = (
            factors.finite
            & jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(compatibility_defect)
            & jnp.isfinite(gauge_defect)
        )
        converged = (
            finite
            & (relative_residual <= self.plan.tolerance)
            & (compatibility_defect <= self.plan.tolerance)
            & (gauge_defect <= self.plan.tolerance)
        )
        value = jnp.where(converged, candidate, jnp.zeros_like(candidate))
        result = TransformLineSolveResult(
            value=value,
            candidate=candidate,
            residual=residual,
            compatible_rhs=compatible_rhs,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            compatibility_defect=compatibility_defect,
            gauge_defect=gauge_defect,
            trace_defect=factors.trace_defect,
            factor_residual=factors.factor_residual,
            minimum_pivot=factors.minimum_pivot,
            finite=finite,
            converged=converged,
            resources=self.resources,
            differentiation_policy=self.plan.differentiation.mode,
            nullspace_policy_id=factors.nullspace_policy_id,
            plan_id=self.plan.plan_id,
            representation_id=representation.representation_id,
            factor_id=factors.factor_id,
        )
        if self.plan.differentiation.mode == "none":
            result = jax.tree.map(
                lambda array: (
                    jax.lax.stop_gradient(array) if eqx.is_array(array) else array
                ),
                result,
            )
        return result


def _solve_factored(
    pivots: Array,
    multipliers: Array,
    upper: Array,
    right_hand_side: Array,
    /,
) -> Array:
    line_size = int(pivots.shape[-1])
    result = jnp.zeros_like(right_hand_side)
    result = result.at[:, 0, :].set(right_hand_side[:, 0, :])
    for index in range(1, line_size):
        result = result.at[:, index, :].set(
            right_hand_side[:, index, :]
            - multipliers[:, index - 1, None] * result[:, index - 1, :]
        )
    safe_last = jnp.where(
        jnp.abs(pivots[:, -1]) > 0.0,
        pivots[:, -1],
        jnp.ones_like(pivots[:, -1]),
    )
    result = result.at[:, -1, :].set(result[:, -1, :] / safe_last[:, None])
    for index in reversed(range(line_size - 1)):
        safe = jnp.where(
            jnp.abs(pivots[:, index]) > 0.0,
            pivots[:, index],
            jnp.ones_like(pivots[:, index]),
        )
        result = result.at[:, index, :].set(
            (result[:, index, :] - upper[index] * result[:, index + 1, :]) / safe[:, None]
        )
    return result


__all__ = [
    "PreparedTransformLineSolve",
    "TransformLineNullspaceKind",
    "TransformLineNullspacePolicy",
    "TransformLineFactors",
    "TransformLineReport",
    "TransformLineRepresentation",
    "TransformLineResourceEstimate",
    "TransformLineSolvePlan",
    "TransformLineSolveResult",
]
