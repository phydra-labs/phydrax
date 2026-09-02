#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._tensor_support import PreparedTensorGrid
from ..finite_difference._mapped_grid import PreparedMappedTensorGrid


BlockSide: TypeAlias = Literal["lower", "upper"]
PreparedBlock: TypeAlias = PreparedTensorGrid | PreparedMappedTensorGrid


class InterfaceOrientation(StrictModule, NonTrainableState):
    """Permutation and reflection of tangential trace axes."""

    trace_rank: int = eqx.field(static=True)
    permutation: tuple[int, ...] = eqx.field(static=True)
    flips: tuple[bool, ...] = eqx.field(static=True)
    orientation_id: str = eqx.field(static=True)

    def __init__(
        self,
        trace_rank: int,
        /,
        *,
        permutation: Sequence[int] | None = None,
        flips: Sequence[bool] | None = None,
    ):
        rank = int(trace_rank)
        if rank < 0:
            raise ValueError("Trace rank must be non-negative.")
        permutation_ = (
            tuple(range(rank))
            if permutation is None
            else tuple(int(value) for value in permutation)
        )
        flips_ = (
            (False,) * rank if flips is None else tuple(bool(value) for value in flips)
        )
        if sorted(permutation_) != list(range(rank)) or len(flips_) != rank:
            raise ValueError("Interface orientation must permute every trace axis once.")
        self.trace_rank = rank
        self.permutation = permutation_
        self.flips = flips_
        self.orientation_id = canonical_fingerprint(
            {
                "kind": "interface-orientation",
                "trace_rank": rank,
                "permutation": list(permutation_),
                "flips": list(flips_),
            }
        )

    def apply(self, values: Array, /, *, trailing_axes: int = 0) -> Array:
        value = jnp.asarray(values)
        trailing = int(trailing_axes)
        if value.ndim != self.trace_rank + trailing:
            raise ValueError("Interface trace rank does not match oriented values.")
        axes = self.permutation + tuple(range(self.trace_rank, value.ndim))
        result = jnp.transpose(value, axes)
        for axis, flip in enumerate(self.flips):
            if flip:
                result = jnp.flip(result, axis=axis)
        return result

    def inverse(self, values: Array, /, *, trailing_axes: int = 0) -> Array:
        value = jnp.asarray(values)
        result = value
        for axis, flip in enumerate(self.flips):
            if flip:
                result = jnp.flip(result, axis=axis)
        inverse = tuple(np.argsort(np.asarray(self.permutation)).tolist())
        axes = inverse + tuple(range(self.trace_rank, result.ndim))
        return jnp.transpose(result, axes)


class BlockInterface(StrictModule, NonTrainableState):
    """One oriented physical face connection between two named blocks."""

    name: str = eqx.field(static=True)
    left_block: str = eqx.field(static=True)
    left_axis: str = eqx.field(static=True)
    left_side: BlockSide = eqx.field(static=True)
    right_block: str = eqx.field(static=True)
    right_axis: str = eqx.field(static=True)
    right_side: BlockSide = eqx.field(static=True)
    orientation: InterfaceOrientation
    interface_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        left_block: str,
        left_axis: str,
        left_side: BlockSide,
        right_block: str,
        right_axis: str,
        right_side: BlockSide,
        orientation: InterfaceOrientation,
        /,
    ):
        values = tuple(
            str(value) for value in (name, left_block, left_axis, right_block, right_axis)
        )
        if (
            any(not value for value in values)
            or left_side
            not in (
                "lower",
                "upper",
            )
            or right_side not in ("lower", "upper")
        ):
            raise ValueError("Block interface names, axes, and sides must be valid.")
        if left_block == right_block:
            raise ValueError("A physical interface must connect distinct blocks.")
        if not isinstance(orientation, InterfaceOrientation):
            raise TypeError("orientation must be InterfaceOrientation.")
        self.name = values[0]
        self.left_block = values[1]
        self.left_axis = values[2]
        self.left_side = left_side
        self.right_block = values[3]
        self.right_axis = values[4]
        self.right_side = right_side
        self.orientation = orientation
        self.interface_id = canonical_fingerprint(
            {
                "kind": "block-interface",
                "name": values[0],
                "left": [values[1], values[2], left_side],
                "right": [values[3], values[4], right_side],
                "orientation": orientation.orientation_id,
            }
        )


class MultiblockInterfaceReport(StrictModule, NonTrainableState):
    """Conforming/nonconforming geometry and nesting evidence for one interface."""

    interface_name: str = eqx.field(static=True)
    left_trace_shape: tuple[int, ...] = eqx.field(static=True)
    right_trace_shape: tuple[int, ...] = eqx.field(static=True)
    conforming: bool = eqx.field(static=True)
    nesting_ratio: int = eqx.field(static=True)
    geometry_residual: float = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        interface: BlockInterface,
        left_trace_shape: tuple[int, ...],
        right_trace_shape: tuple[int, ...],
        /,
        *,
        nesting_ratio: int,
        geometry_residual: float,
        tolerance: float,
    ):
        residual = float(geometry_residual)
        ratio = int(nesting_ratio)
        conforming = left_trace_shape == right_trace_shape
        self.interface_name = interface.name
        self.left_trace_shape = left_trace_shape
        self.right_trace_shape = right_trace_shape
        self.conforming = conforming
        self.nesting_ratio = ratio
        self.geometry_residual = residual
        self.passed = ratio in (1, 2) and residual <= float(tolerance)
        self.report_id = canonical_fingerprint(
            {
                "kind": "multiblock-interface-report",
                "interface": interface.interface_id,
                "left_shape": list(left_trace_shape),
                "right_shape": list(right_trace_shape),
                "ratio": ratio,
                "geometry_residual": residual,
                "tolerance": float(tolerance),
            }
        )


class MultiblockGridPlan(StrictModule, NonTrainableState):
    """Named mapped/reference blocks with explicit oriented face topology."""

    block_names: tuple[str, ...] = eqx.field(static=True)
    blocks: tuple[PreparedBlock, ...]
    interfaces: tuple[BlockInterface, ...]
    geometry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[tuple[str, PreparedBlock]],
        interfaces: Sequence[BlockInterface],
        /,
        *,
        geometry_tolerance: float = 1e-9,
    ):
        block_values = tuple(blocks)
        names = tuple(str(name) for name, _ in block_values)
        prepared = tuple(value for _, value in block_values)
        interfaces_ = tuple(interfaces)
        tolerance = float(geometry_tolerance)
        if (
            not names
            or len(set(names)) != len(names)
            or any(not name for name in names)
            or not all(
                isinstance(value, (PreparedTensorGrid, PreparedMappedTensorGrid))
                for value in prepared
            )
        ):
            raise ValueError("Multiblock plans require unique named prepared blocks.")
        dimensions = {len(_reference_grid(value).shape) for value in prepared}
        if len(dimensions) != 1:
            raise ValueError("Every multiblock grid block must have one dimension.")
        if not all(isinstance(value, BlockInterface) for value in interfaces_):
            raise TypeError("interfaces must contain BlockInterface values.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("geometry_tolerance must be finite and positive.")
        known = set(names)
        if any(
            interface.left_block not in known or interface.right_block not in known
            for interface in interfaces_
        ):
            raise ValueError("Block interface references an unknown block.")
        faces = tuple(
            face
            for interface in interfaces_
            for face in (
                (interface.left_block, interface.left_axis, interface.left_side),
                (interface.right_block, interface.right_axis, interface.right_side),
            )
        )
        if len(set(faces)) != len(faces):
            raise ValueError(
                "Each physical block face may participate in at most one interface."
            )
        self.block_names = names
        self.blocks = prepared
        self.interfaces = interfaces_
        self.geometry_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multiblock-grid-plan",
                "blocks": [
                    [name, _block_id(value)]
                    for name, value in zip(names, prepared, strict=True)
                ],
                "interfaces": [value.interface_id for value in interfaces_],
                "geometry_tolerance": tolerance,
            }
        )

    def prepare(
        self,
        /,
        *,
        interface_coordinates: Sequence[tuple[Array, Array]] | None = None,
    ) -> "PreparedMultiblockGrid":
        """Validate interfaces against block or caller-supplied physical traces.

        ``interface_coordinates`` supplies one ``(left, right)`` pair per
        interface.  It is intended for embedded manifolds whose physical
        dimension differs from their logical block dimension.
        """
        return PreparedMultiblockGrid(
            self, interface_coordinates=interface_coordinates
        )


def _reference_grid(block: PreparedBlock, /) -> PreparedTensorGrid:
    return block.reference_grid if isinstance(block, PreparedMappedTensorGrid) else block


def _block_id(block: PreparedBlock, /) -> str:
    return block.prepared_id


def _physical_coordinates(block: PreparedBlock, /) -> Array:
    if isinstance(block, PreparedMappedTensorGrid):
        return block.physical_coordinates
    grid = block
    dimension = len(grid.shape)
    components = []
    for axis, coordinates in enumerate(grid.primary_entity_layout.coordinates_by_axis):
        reshape = [1] * dimension
        reshape[axis] = int(coordinates.size)
        components.append(jnp.broadcast_to(coordinates.reshape(reshape), grid.shape))
    return jnp.stack(components, axis=-1)


def _trace(
    block: PreparedBlock,
    axis: str,
    side: BlockSide,
    values: Array,
    /,
) -> Array:
    grid = _reference_grid(block)
    axis_index = grid.axis_names.index(axis)
    index = 0 if side == "lower" else grid.shape[axis_index] - 1
    return jnp.take(values, index, axis=axis_index)


class PreparedMultiblockGrid(StrictModule, NonTrainableState):
    """Validated oriented block topology with physical trace evidence."""

    plan: MultiblockGridPlan
    interface_reports: tuple[MultiblockInterfaceReport, ...]
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: MultiblockGridPlan,
        /,
        *,
        interface_coordinates: Sequence[tuple[Array, Array]] | None = None,
    ):
        if not isinstance(plan, MultiblockGridPlan):
            raise TypeError("plan must be MultiblockGridPlan.")
        supplied = (
            None
            if interface_coordinates is None
            else tuple(
                (jnp.asarray(left), jnp.asarray(right))
                for left, right in interface_coordinates
            )
        )
        if supplied is not None and len(supplied) != len(plan.interfaces):
            raise ValueError(
                "Interface coordinates must provide one trace pair per interface."
            )
        reports = []
        for index, interface in enumerate(plan.interfaces):
            left = self._block_from(plan, interface.left_block)
            right = self._block_from(plan, interface.right_block)
            left_grid = _reference_grid(left)
            right_grid = _reference_grid(right)
            if (
                interface.left_axis not in left_grid.axis_names
                or interface.right_axis not in right_grid.axis_names
            ):
                raise ValueError("Interface axis is not present in its block.")
            trace_rank = len(left_grid.shape) - 1
            if (
                interface.orientation.trace_rank != trace_rank
                or len(right_grid.shape) - 1 != trace_rank
            ):
                raise ValueError(
                    "Interface orientation rank must match both block traces."
                )
            if supplied is None:
                left_trace = _trace(
                    left,
                    interface.left_axis,
                    interface.left_side,
                    _physical_coordinates(left),
                )
                right_trace = _trace(
                    right,
                    interface.right_axis,
                    interface.right_side,
                    _physical_coordinates(right),
                )
            else:
                left_trace, right_trace = supplied[index]
                if (
                    left_trace.ndim != trace_rank + 1
                    or right_trace.ndim != trace_rank + 1
                    or left_trace.shape[-1] != right_trace.shape[-1]
                ):
                    raise ValueError(
                        "Supplied physical interface traces have invalid dimensions."
                    )
            right_trace = interface.orientation.apply(
                right_trace, trailing_axes=1
            )
            left_shape = left_trace.shape[:-1]
            right_shape = right_trace.shape[:-1]
            if left_shape == right_shape:
                ratio = 1
                residual = float(np.max(np.abs(np.asarray(left_trace - right_trace))))
            elif trace_rank == 1 and (
                right_shape[0] == 2 * (left_shape[0] - 1) + 1
                or left_shape[0] == 2 * (right_shape[0] - 1) + 1
            ):
                ratio = 2
                left_endpoints = jnp.stack((left_trace[0], left_trace[-1]))
                right_endpoints = jnp.stack((right_trace[0], right_trace[-1]))
                residual = float(
                    np.max(np.abs(np.asarray(left_endpoints - right_endpoints)))
                )
            else:
                raise ValueError(
                    "Only conforming and nested 2:1 interface traces are supported."
                )
            report = MultiblockInterfaceReport(
                interface,
                left_shape,
                right_shape,
                nesting_ratio=ratio,
                geometry_residual=residual,
                tolerance=plan.geometry_tolerance,
            )
            if not report.passed:
                raise ValueError(
                    f"Interface {interface.name!r} physical traces do not coincide."
                )
            reports.append(report)
        self.plan = plan
        self.interface_reports = tuple(reports)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-multiblock-grid",
                "plan": plan.plan_id,
                "reports": [value.report_id for value in reports],
                "interface_coordinates": (
                    None if supplied is None else array_tree_fingerprint(supplied)
                ),
            }
        )

    @staticmethod
    def _block_from(plan: MultiblockGridPlan, name: str, /) -> PreparedBlock:
        return plan.blocks[plan.block_names.index(name)]

    def block(self, name: str, /) -> PreparedBlock:
        if name not in self.plan.block_names:
            raise KeyError(f"Unknown multiblock grid block {name!r}.")
        return self._block_from(self.plan, name)

    def trace(
        self,
        block_name: str,
        axis: str,
        side: BlockSide,
        values: Array,
        /,
    ) -> Array:
        return _trace(self.block(block_name), axis, side, jnp.asarray(values))


__all__ = [
    "BlockInterface",
    "BlockSide",
    "InterfaceOrientation",
    "MultiblockGridPlan",
    "MultiblockInterfaceReport",
    "PreparedBlock",
    "PreparedMultiblockGrid",
]
