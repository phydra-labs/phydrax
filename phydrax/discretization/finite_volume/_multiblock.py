#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum_chunks
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..multiblock import InterfaceOrientation
from ._riemann import AbstractNumericalFluxPlan
from ._structured import FiniteVolumeDiscretization


InterfaceSide: TypeAlias = Literal["lower", "upper"]


def _boundary(values: Array, axis: int, side: InterfaceSide, /) -> Array:
    index = 0 if side == "lower" else values.shape[axis] - 1
    return jnp.take(values, index, axis=axis)


def _tangential_shape(shape: tuple[int, ...], axis: int, /) -> tuple[int, ...]:
    return shape[:axis] + shape[axis + 1 :]


def _repeat_to_shape(values: Array, target: tuple[int, ...], /) -> Array:
    output = values
    if output.shape[:-1] == target:
        return output
    if len(output.shape[:-1]) != len(target):
        raise ValueError("Multiblock traces must have matching rank.")
    for axis, (source, destination) in enumerate(
        zip(output.shape[:-1], target, strict=True)
    ):
        if destination % source:
            raise ValueError("Nested interface ratios must be integral.")
        output = jnp.repeat(output, destination // source, axis=axis)
    return output


def _sum_to_shape(values: Array, target: tuple[int, ...], /) -> Array:
    output = values
    for axis, destination in enumerate(target):
        source = output.shape[axis]
        if source == destination:
            continue
        if source % destination:
            raise ValueError("Fine interface flux cannot be reduced to coarse shape.")
        ratio = source // destination
        shape = output.shape[:axis] + (destination, ratio) + output.shape[axis + 1 :]
        output = output.reshape(shape).sum(axis=axis + 1)
    return output


class ConservativeMultiblockFluxResult(StrictModule):
    common_integrated_flux: Array
    left_integrated_flux: Array
    right_integrated_flux: Array
    conservation_defect: Array
    max_speed: Array


class ConservativeMultiblockInterfacePlan(StrictModule, NonTrainableState):
    """Conforming or nested 2:1 shared FV interface flux."""

    left: FiniteVolumeDiscretization
    right: FiniteVolumeDiscretization
    left_axis: int = eqx.field(static=True)
    right_axis: int = eqx.field(static=True)
    left_side: InterfaceSide = eqx.field(static=True)
    right_side: InterfaceSide = eqx.field(static=True)
    orientation: InterfaceOrientation
    interface_solver: AbstractNumericalFluxPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left: FiniteVolumeDiscretization,
        right: FiniteVolumeDiscretization,
        left_axis: int,
        right_axis: int,
        orientation: InterfaceOrientation,
        interface_solver: AbstractNumericalFluxPlan,
        /,
        *,
        left_side: InterfaceSide = "upper",
        right_side: InterfaceSide = "lower",
    ):
        if not isinstance(left, FiniteVolumeDiscretization) or not isinstance(
            right, FiniteVolumeDiscretization
        ):
            raise TypeError("Multiblock FV interfaces require two discretizations.")
        left_axis_ = int(left_axis)
        right_axis_ = int(right_axis)
        if (
            not 0 <= left_axis_ < len(left.cell_shape)
            or not 0 <= right_axis_ < len(right.cell_shape)
            or left_side not in ("lower", "upper")
            or right_side not in ("lower", "upper")
        ):
            raise ValueError("Multiblock FV axis or side is invalid.")
        if left_side != "upper" or right_side != "lower":
            raise ValueError(
                "Initial multiblock FV orientation requires upper-to-lower sides."
            )
        if left.component_names != right.component_names:
            raise ValueError("Multiblock FV component layouts must match.")
        if not isinstance(orientation, InterfaceOrientation):
            raise TypeError("orientation must be an InterfaceOrientation.")
        if orientation.trace_rank != len(left.cell_shape) - 1:
            raise ValueError("Interface orientation rank must match tangential rank.")
        if not isinstance(interface_solver, AbstractNumericalFluxPlan):
            raise TypeError("interface_solver must be a numerical flux plan.")
        left_shape = _tangential_shape(left.cell_shape, left_axis_)
        right_shape = _tangential_shape(right.cell_shape, right_axis_)
        oriented_right_shape = tuple(
            right_shape[index] for index in orientation.permutation
        )
        for left_count, right_count in zip(left_shape, oriented_right_shape, strict=True):
            if (
                max(left_count, right_count) != 2 * min(left_count, right_count)
                and left_count != right_count
            ):
                raise ValueError("Multiblock FV interfaces must conform or nest 2:1.")
        self.left = left
        self.right = right
        self.left_axis = left_axis_
        self.right_axis = right_axis_
        self.left_side = left_side
        self.right_side = right_side
        self.orientation = orientation
        self.interface_solver = interface_solver
        self.plan_id = canonical_fingerprint(
            {
                "kind": "conservative-multiblock-fv-interface",
                "left": left.prepared_id,
                "right": right.prepared_id,
                "left_axis": left_axis_,
                "right_axis": right_axis_,
                "orientation": orientation.orientation_id,
                "flux": interface_solver.flux_id,
            }
        )

    def flux(
        self,
        system: Any,
        left_state: Array,
        right_state: Array,
        args: Any = None,
        /,
    ) -> ConservativeMultiblockFluxResult:
        left_trace = _boundary(left_state, self.left_axis, self.left_side)
        right_trace = self.orientation.apply(
            _boundary(right_state, self.right_axis, self.right_side),
            trailing_axes=1,
        )
        left_measure = _boundary(
            self.left.face_measures[self.left_axis], self.left_axis, self.left_side
        )
        right_measure = self.orientation.apply(
            _boundary(
                self.right.face_measures[self.right_axis],
                self.right_axis,
                self.right_side,
            )
        )
        left_shape = left_trace.shape[:-1]
        right_shape = right_trace.shape[:-1]
        mortar_shape = tuple(
            max(left_count, right_count)
            for left_count, right_count in zip(left_shape, right_shape, strict=True)
        )
        left_mortar = _repeat_to_shape(left_trace, mortar_shape)
        right_mortar = _repeat_to_shape(right_trace, mortar_shape)
        left_measure_mortar = _repeat_to_shape(left_measure[..., None], mortar_shape)[
            ..., 0
        ]
        right_measure_mortar = _repeat_to_shape(right_measure[..., None], mortar_shape)[
            ..., 0
        ]
        mortar_measure = jnp.minimum(left_measure_mortar, right_measure_mortar)
        result = self.interface_solver.face_flux(
            system, left_mortar, right_mortar, self.left_axis, args
        )
        common = result.normal_flux * mortar_measure[..., None]
        left_integrated = _sum_to_shape(common, left_shape)
        right_oriented = -_sum_to_shape(common, right_shape)
        right_integrated = self.orientation.inverse(right_oriented, trailing_axes=1)
        defect = compensated_sum_chunks(
            (left_integrated, right_integrated),
            output_ndim=1,
        )
        return ConservativeMultiblockFluxResult(
            common_integrated_flux=common,
            left_integrated_flux=left_integrated,
            right_integrated_flux=right_integrated,
            conservation_defect=defect,
            max_speed=result.max_speed,
        )


__all__ = [
    "ConservativeMultiblockFluxResult",
    "ConservativeMultiblockInterfacePlan",
]
