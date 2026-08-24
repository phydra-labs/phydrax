#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..finite_difference._certification import FDStabilityReport
from ..finite_difference._sbp import PreparedSBPOperator
from ._core import (
    _physical_coordinates,
    _reference_grid,
    _trace,
    BlockInterface,
    BlockSide,
    PreparedMultiblockGrid,
)
from ._interpolation import NormCompatibleInterpolationPlan


MultiblockNumericalFlux: TypeAlias = Literal["central", "upwind"]


class MultiblockSATCoupling(StrictModule, NonTrainableState):
    """Conforming or 2:1 mortar SAT for scalar advection across one interface."""

    multiblock: PreparedMultiblockGrid
    interface: BlockInterface
    left: PreparedSBPOperator
    right: PreparedSBPOperator
    interpolation: NormCompatibleInterpolationPlan | None
    left_trace_weights: Array
    right_trace_weights: Array
    speed: float = eqx.field(static=True)
    flux: MultiblockNumericalFlux = eqx.field(static=True)
    local_speeds: tuple[float, float] = eqx.field(static=True)
    stability_report: FDStabilityReport
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        multiblock: PreparedMultiblockGrid,
        interface_name: str,
        left: PreparedSBPOperator,
        right: PreparedSBPOperator,
        speed: float,
        /,
        *,
        flux: MultiblockNumericalFlux = "central",
        interpolation_order: int = 4,
    ):
        if (
            not isinstance(multiblock, PreparedMultiblockGrid)
            or not isinstance(left, PreparedSBPOperator)
            or not isinstance(right, PreparedSBPOperator)
        ):
            raise TypeError(
                "Multiblock SAT requires prepared topology and SBP operators."
            )
        selected = tuple(
            value for value in multiblock.plan.interfaces if value.name == interface_name
        )
        if len(selected) != 1:
            raise ValueError("interface_name must select exactly one block interface.")
        interface = selected[0]
        left_block = multiblock.block(interface.left_block)
        right_block = multiblock.block(interface.right_block)
        if (
            left.grid.prepared_id != _reference_grid(left_block).prepared_id
            or right.grid.prepared_id != _reference_grid(right_block).prepared_id
            or left.axis != interface.left_axis
            or right.axis != interface.right_axis
        ):
            raise ValueError(
                "SBP operators must align with the interface blocks and normals."
            )
        speed_ = float(speed)
        if (
            not np.isfinite(speed_)
            or speed_ == 0.0
            or flux
            not in (
                "central",
                "upwind",
            )
        ):
            raise ValueError("Multiblock SAT speed/flux is invalid.")
        left_trace_shape = (
            left.grid.shape[: left.axis_index] + left.grid.shape[left.axis_index + 1 :]
        )
        right_trace_shape = (
            right.grid.shape[: right.axis_index]
            + right.grid.shape[right.axis_index + 1 :]
        )
        oriented_right_shape = tuple(
            right_trace_shape[index] for index in interface.orientation.permutation
        )
        left_trace_weights = _trace_tangential_weight(
            left_block,
            left,
            interface.left_side,
        )
        right_trace_weights = interface.orientation.apply(
            _trace_tangential_weight(
                right_block,
                right,
                interface.right_side,
            )
        )
        interpolation = None
        if left_trace_shape != oriented_right_shape:
            if len(left_trace_shape) != 1:
                raise ValueError(
                    "Nonconforming SAT currently requires one tangential axis."
                )
            left_coordinates, left_weights = _trace_coordinate_and_weight(
                left_block,
                left,
                interface.left_side,
            )
            right_coordinates, right_weights = _trace_coordinate_and_weight(
                right_block,
                right,
                interface.right_side,
            )
            right_coordinates = interface.orientation.apply(right_coordinates)
            right_weights = interface.orientation.apply(right_weights)
            interpolation = NormCompatibleInterpolationPlan(
                left_coordinates,
                right_coordinates,
                left_weights,
                right_weights,
                interpolation_order=interpolation_order,
            )
        left_sign = -1.0 if interface.left_side == "lower" else 1.0
        right_sign = -1.0 if interface.right_side == "lower" else 1.0
        identifier = canonical_fingerprint(
            {
                "kind": "multiblock-sat-coupling",
                "multiblock": multiblock.prepared_id,
                "interface": interface.interface_id,
                "left_sbp": left.prepared_id,
                "right_sbp": right.prepared_id,
                "interpolation": (
                    None if interpolation is None else interpolation.plan_id
                ),
                "speed": speed_,
                "flux": flux,
            }
        )
        self.multiblock = multiblock
        self.interface = interface
        self.left = left
        self.right = right
        self.interpolation = interpolation
        self.left_trace_weights = left_trace_weights
        self.right_trace_weights = right_trace_weights
        self.speed = speed_
        self.flux = flux
        self.local_speeds = (left_sign * speed_, -right_sign * speed_)
        self.stability_report = FDStabilityReport(
            "multiblock_sat_energy",
            residual=0.0,
            tolerance=1e-12,
            assumptions=(
                "norm-compatible mortar interpolation",
                "constant scalar advection speed",
            ),
            evidence="analytic",
            subject_id=identifier,
        )
        self.coupling_id = identifier

    def _trace_values(
        self,
        left_state: Array,
        right_state: Array,
        /,
    ) -> tuple[Array, Array, tuple[int, ...], tuple[int, ...]]:
        left_trace = _trace(
            self.multiblock.block(self.interface.left_block),
            self.interface.left_axis,
            self.interface.left_side,
            left_state,
        )
        right_trace = self.interface.orientation.apply(
            _trace(
                self.multiblock.block(self.interface.right_block),
                self.interface.right_axis,
                self.interface.right_side,
                right_state,
            )
        )
        return (
            left_trace.reshape((-1,)),
            right_trace.reshape((-1,)),
            left_trace.shape,
            right_trace.shape,
        )

    def corrections(
        self,
        left_state: ArrayLike,
        right_state: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        left = self.left.operator.source.validate(jnp.asarray(left_state))
        right = self.right.operator.source.validate(jnp.asarray(right_state))
        left_trace, right_trace, left_shape, right_shape = self._trace_values(left, right)
        if self.interpolation is None:
            if left_trace.shape != right_trace.shape:
                raise RuntimeError(
                    "Conforming interface trace sizes diverged at runtime."
                )
            numerical_flux = (
                0.5 * self.speed * (left_trace + right_trace)
                if self.flux == "central"
                else self.speed * left_trace
                if self.speed > 0.0
                else self.speed * right_trace
            )
            left_residual = self.speed * left_trace - numerical_flux
            right_residual = numerical_flux - self.speed * right_trace
        else:
            left_mortar = self.interpolation.left_to_mortar(left_trace)
            right_mortar = self.interpolation.right_to_mortar(right_trace)
            right_on_left = self.interpolation.mortar_to_left(right_mortar)
            left_on_right = self.interpolation.mortar_to_right(left_mortar)
            if self.flux == "central":
                left_residual = 0.5 * self.speed * (left_trace - right_on_left)
                right_residual = 0.5 * self.speed * (left_on_right - right_trace)
            elif self.speed > 0.0:
                left_residual = jnp.zeros_like(left_trace)
                right_residual = self.speed * (left_on_right - right_trace)
            else:
                left_residual = self.speed * (left_trace - right_on_left)
                right_residual = jnp.zeros_like(right_trace)
        left_residual = left_residual * self.left_trace_weights.reshape((-1,))
        right_residual = right_residual * self.right_trace_weights.reshape((-1,))
        right_residual = self.interface.orientation.inverse(
            right_residual.reshape(right_shape)
        ).reshape((-1,))
        left_residual = left_residual.reshape(left_shape)
        right_trace_shape = (
            self.right.grid.shape[: self.right.axis_index]
            + self.right.grid.shape[self.right.axis_index + 1 :]
        )
        right_residual = right_residual.reshape(right_trace_shape)
        left_correction = _inject_trace(
            jnp.zeros_like(left),
            self.left.axis_index,
            self.interface.left_side,
            left_residual,
        )
        right_correction = _inject_trace(
            jnp.zeros_like(right),
            self.right.axis_index,
            self.interface.right_side,
            right_residual,
        )
        return (
            left_correction / self.left.norm_weights,
            right_correction / self.right.norm_weights,
        )


def _trace_tangential_weight(
    block,
    sbp: PreparedSBPOperator,
    side: BlockSide,
    /,
) -> Array:
    trace_weight = _trace(
        block,
        sbp.axis,
        side,
        sbp.norm_weights,
    )
    normal_index = 0 if side == "lower" else sbp.axis_norm_weights.size - 1
    return trace_weight / sbp.axis_norm_weights[normal_index]


def _trace_coordinate_and_weight(
    block,
    sbp: PreparedSBPOperator,
    side: BlockSide,
    /,
) -> tuple[Array, Array]:
    physical = _trace(
        block,
        sbp.axis,
        side,
        _physical_coordinates(block),
    )
    if physical.ndim != 2:
        raise ValueError("Nonconforming interpolation requires one-dimensional traces.")
    distances = jnp.linalg.norm(jnp.diff(physical, axis=0), axis=-1)
    coordinate = jnp.concatenate((jnp.zeros((1,)), jnp.cumsum(distances)))
    return coordinate, _trace_tangential_weight(block, sbp, side)


def _inject_trace(
    output: Array,
    axis: int,
    side: BlockSide,
    trace: Array,
    /,
) -> Array:
    index: list[slice | int] = [slice(None)] * output.ndim
    index[axis] = 0 if side == "lower" else output.shape[axis] - 1
    return output.at[tuple(index)].set(trace)


__all__ = ["MultiblockNumericalFlux", "MultiblockSATCoupling"]
