#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..multiblock import InterfaceOrientation
from ._discretization import LatticeBoltzmannDiscretization
from ._geometry import LatticeBoltzmannGeometryKind


LatticeBoltzmannBlockSide: TypeAlias = Literal["lower", "upper"]


def _side_sign(side: LatticeBoltzmannBlockSide, /) -> int:
    return -1 if side == "lower" else 1


def _boundary(values: Array, axis: int, side: LatticeBoltzmannBlockSide, /) -> Array:
    return jnp.take(values, 0 if side == "lower" else values.shape[axis] - 1, axis=axis)


def _tangential_shape(shape: tuple[int, ...], axis: int, /) -> tuple[int, ...]:
    return shape[:axis] + shape[axis + 1 :]


class LatticeBoltzmannBlockTracePair(StrictModule):
    """Conforming traces expressed in the left block's spatial and Q ordering."""

    left: Array
    right_in_left_coordinates: Array


class LatticeBoltzmannBlockInterfacePlan(StrictModule, NonTrainableState):
    """Conforming block interface with an exact integer-velocity Q permutation."""

    left: LatticeBoltzmannDiscretization
    right: LatticeBoltzmannDiscretization
    orientation: InterfaceOrientation
    population_permutation: Array
    inverse_population_permutation: Array
    coordinate_transform: Array
    left_axis: int = eqx.field(static=True)
    right_axis: int = eqx.field(static=True)
    left_side: LatticeBoltzmannBlockSide = eqx.field(static=True)
    right_side: LatticeBoltzmannBlockSide = eqx.field(static=True)
    population_permutation_indices: tuple[int, ...] = eqx.field(static=True)
    inverse_population_permutation_indices: tuple[int, ...] = eqx.field(static=True)
    geometry_kind: LatticeBoltzmannGeometryKind = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        left: LatticeBoltzmannDiscretization,
        right: LatticeBoltzmannDiscretization,
        left_axis: int,
        right_axis: int,
        orientation: InterfaceOrientation,
        /,
        *,
        left_side: LatticeBoltzmannBlockSide = "upper",
        right_side: LatticeBoltzmannBlockSide = "lower",
        scale_tolerance: float = 1.0e-12,
    ):
        if not isinstance(left, LatticeBoltzmannDiscretization) or not isinstance(
            right, LatticeBoltzmannDiscretization
        ):
            raise TypeError("Block interfaces require two LBM discretizations.")
        if not isinstance(orientation, InterfaceOrientation):
            raise TypeError("orientation must be an InterfaceOrientation.")
        left_axis_ = int(left_axis)
        right_axis_ = int(right_axis)
        dimension = left.velocity_set.dimension
        tolerance = float(scale_tolerance)
        if right.velocity_set.dimension != dimension:
            raise ValueError("LBM block dimensions must match.")
        if (
            not 0 <= left_axis_ < dimension
            or not 0 <= right_axis_ < dimension
            or left_side not in ("lower", "upper")
            or right_side not in ("lower", "upper")
        ):
            raise ValueError("LBM block interface axes or sides are invalid.")
        if orientation.trace_rank != dimension - 1:
            raise ValueError("Interface orientation rank must match tangential rank.")
        if left.periodic[left_axis_] or right.periodic[right_axis_]:
            raise ValueError("A multiblock interface face cannot be periodic.")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("scale_tolerance must be finite and positive.")
        left_scale = float(left.cell_size)
        right_scale = float(right.cell_size)
        if not np.isclose(left_scale, right_scale, rtol=tolerance, atol=tolerance):
            raise ValueError("Conforming LBM blocks must use one lattice scale.")

        left_shape = _tangential_shape(left.grid.shape, left_axis_)
        right_shape = _tangential_shape(right.grid.shape, right_axis_)
        oriented_right_shape = tuple(
            right_shape[index] for index in orientation.permutation
        )
        if left_shape != oriented_right_shape:
            raise ValueError("LBM block traces must be conforming after orientation.")

        left_tangential = tuple(axis for axis in range(dimension) if axis != left_axis_)
        right_tangential = tuple(axis for axis in range(dimension) if axis != right_axis_)
        transform = np.zeros((dimension, dimension), dtype=np.int32)
        transform[left_axis_, right_axis_] = -_side_sign(left_side) * _side_sign(
            right_side
        )
        for left_position, right_position in enumerate(orientation.permutation):
            left_coordinate = left_tangential[left_position]
            right_coordinate = right_tangential[right_position]
            transform[left_coordinate, right_coordinate] = (
                -1 if orientation.flips[left_position] else 1
            )
        if not np.array_equal(transform @ transform.T, np.eye(dimension, dtype=np.int32)):
            raise ValueError(
                "Interface orientation does not define an orthogonal axis map."
            )

        left_velocities = np.asarray(left.velocity_set.velocities, dtype=np.int32)
        right_velocities = np.asarray(right.velocity_set.velocities, dtype=np.int32)
        transformed_right = right_velocities @ transform.T
        if left_velocities.shape != transformed_right.shape:
            raise ValueError("LBM blocks must have the same population count.")
        left_lookup = {
            tuple(int(value) for value in velocity): index
            for index, velocity in enumerate(left_velocities)
        }
        right_to_left = []
        for velocity in transformed_right:
            key = tuple(int(value) for value in velocity)
            if key not in left_lookup:
                raise ValueError(
                    "Interface orientation is incompatible with the lattice velocity set."
                )
            right_to_left.append(left_lookup[key])
        if len(set(right_to_left)) != left.velocity_set.population_count:
            raise ValueError("Interface velocity transformation is not a Q permutation.")
        permutation = np.empty((left.velocity_set.population_count,), dtype=np.int32)
        for right_index, left_index in enumerate(right_to_left):
            permutation[left_index] = right_index
        inverse = np.argsort(permutation).astype(np.int32)
        left_weights = np.asarray(left.velocity_set.weights)
        right_weights = np.asarray(right.velocity_set.weights)
        left_cs2 = float(left.velocity_set.sound_speed_squared)
        right_cs2 = float(right.velocity_set.sound_speed_squared)
        if not np.allclose(
            left_weights,
            right_weights[permutation],
            rtol=tolerance,
            atol=tolerance,
        ) or not np.isclose(left_cs2, right_cs2, rtol=tolerance, atol=tolerance):
            raise ValueError(
                "Oriented LBM blocks must have identical quadrature weights."
            )

        self.left = left
        self.right = right
        self.orientation = orientation
        self.population_permutation = jnp.asarray(permutation, dtype=jnp.int32)
        self.inverse_population_permutation = jnp.asarray(inverse, dtype=jnp.int32)
        self.coordinate_transform = jnp.asarray(transform, dtype=jnp.int32)
        self.left_axis = left_axis_
        self.right_axis = right_axis_
        self.left_side = left_side
        self.right_side = right_side
        self.population_permutation_indices = tuple(int(value) for value in permutation)
        self.inverse_population_permutation_indices = tuple(
            int(value) for value in inverse
        )
        self.geometry_kind = LatticeBoltzmannGeometryKind.BLOCKWISE
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-block-interface",
                "geometry_kind": self.geometry_kind.value,
                "left": left.prepared_id,
                "right": right.prepared_id,
                "left_face": [left_axis_, left_side],
                "right_face": [right_axis_, right_side],
                "orientation": orientation.orientation_id,
                "coordinate_transform": transform.tolist(),
                "population_permutation": permutation.tolist(),
                "lattice_scale": left_scale,
            }
        )

    def left_trace(self, populations: ArrayLike, /) -> Array:
        values = self.left.validate_populations(populations)
        return _boundary(values, self.left_axis, self.left_side)

    def orient_right_trace(self, populations: ArrayLike, /) -> Array:
        values = self.right.validate_populations(populations)
        trace = _boundary(values, self.right_axis, self.right_side)
        oriented = self.orientation.apply(trace, trailing_axes=1)
        return oriented[..., self.population_permutation]

    def orient_left_trace_to_right(self, left_trace: ArrayLike, /) -> Array:
        values = jnp.asarray(left_trace)
        expected = (
            *_tangential_shape(self.left.grid.shape, self.left_axis),
            self.left.velocity_set.population_count,
        )
        if values.shape != expected:
            raise ValueError("Left LBM trace does not match the interface shape.")
        right_order = values[..., self.inverse_population_permutation]
        return self.orientation.inverse(right_order, trailing_axes=1)

    def paired_traces(
        self,
        left_populations: ArrayLike,
        right_populations: ArrayLike,
        /,
    ) -> LatticeBoltzmannBlockTracePair:
        return LatticeBoltzmannBlockTracePair(
            self.left_trace(left_populations),
            self.orient_right_trace(right_populations),
        )


class LatticeBoltzmannBlockConnection(StrictModule, NonTrainableState):
    """One fixed conforming interface between two indexed kinetic blocks."""

    interface: LatticeBoltzmannBlockInterfacePlan
    left_block: int = eqx.field(static=True)
    right_block: int = eqx.field(static=True)
    connection_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_block: int,
        right_block: int,
        interface: LatticeBoltzmannBlockInterfacePlan,
        /,
    ):
        left = int(left_block)
        right = int(right_block)
        if left < 0 or right < 0 or left == right:
            raise ValueError(
                "A block connection requires two distinct nonnegative indices."
            )
        if not isinstance(interface, LatticeBoltzmannBlockInterfacePlan):
            raise TypeError("interface must be LatticeBoltzmannBlockInterfacePlan.")
        self.interface = interface
        self.left_block = left
        self.right_block = right
        self.connection_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-block-connection",
                "left_block": left,
                "right_block": right,
                "interface": interface.plan_id,
            }
        )


class LatticeBoltzmannMultiblockState(StrictModule):
    populations: tuple[Array, ...]

    def __init__(self, populations: Sequence[ArrayLike], /):
        values = tuple(jnp.asarray(value) for value in populations)
        if not values:
            raise ValueError("A multiblock state requires at least one block.")
        self.populations = values


class LatticeBoltzmannMultiblockExchangeEvidence(StrictModule):
    maximum_reciprocity_residual: Array
    incoming_write_count: Array
    interface_count: Array
    successful: Array


class LatticeBoltzmannMultiblockExchangeResult(StrictModule):
    state: LatticeBoltzmannMultiblockState
    write_masks: tuple[Array, ...]
    evidence: LatticeBoltzmannMultiblockExchangeEvidence


def _replace_boundary_directions(
    populations: Array,
    axis: int,
    side: LatticeBoltzmannBlockSide,
    incoming: Array,
    replacement: Array,
    /,
) -> Array:
    moved = jnp.moveaxis(populations, axis, 0)
    location = 0 if side == "lower" else moved.shape[0] - 1
    boundary = moved[location]
    updated = jnp.where(incoming, replacement, boundary)
    return jnp.moveaxis(moved.at[location].set(updated), 0, axis)


def _mark_boundary_directions(
    mask: Array,
    axis: int,
    side: LatticeBoltzmannBlockSide,
    incoming: Array,
    /,
) -> Array:
    moved = jnp.moveaxis(mask, axis, 0)
    location = 0 if side == "lower" else moved.shape[0] - 1
    marked = moved[location] | jnp.broadcast_to(incoming, moved[location].shape)
    return jnp.moveaxis(moved.at[location].set(marked), 0, axis)


class LatticeBoltzmannMultiblockCouplingPlan(StrictModule, NonTrainableState):
    """Same-step exchange over a fixed set of nonoverlapping block interfaces."""

    blocks: tuple[LatticeBoltzmannDiscretization, ...]
    connections: tuple[LatticeBoltzmannBlockConnection, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        blocks: Sequence[LatticeBoltzmannDiscretization],
        connections: Sequence[LatticeBoltzmannBlockConnection],
        /,
    ):
        blocks_ = tuple(blocks)
        connections_ = tuple(connections)
        if not blocks_ or any(
            not isinstance(block, LatticeBoltzmannDiscretization) for block in blocks_
        ):
            raise TypeError("blocks must be a nonempty sequence of LBM discretizations.")
        occupied: set[tuple[int, int, str]] = set()
        for connection in connections_:
            if not isinstance(connection, LatticeBoltzmannBlockConnection):
                raise TypeError("connections must contain block connections.")
            if connection.left_block >= len(blocks_) or connection.right_block >= len(
                blocks_
            ):
                raise ValueError("A connection names a block outside this plan.")
            interface = connection.interface
            if (
                interface.left.prepared_id != blocks_[connection.left_block].prepared_id
                or interface.right.prepared_id
                != blocks_[connection.right_block].prepared_id
            ):
                raise ValueError(
                    "A connection interface does not match its indexed blocks."
                )
            faces = (
                (connection.left_block, interface.left_axis, interface.left_side),
                (connection.right_block, interface.right_axis, interface.right_side),
            )
            if any(face in occupied for face in faces):
                raise ValueError("Each block face may belong to only one interface.")
            occupied.update(faces)
        self.blocks = blocks_
        self.connections = connections_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-multiblock-coupling",
                "blocks": [block.prepared_id for block in blocks_],
                "connections": [connection.connection_id for connection in connections_],
                "schedule": "same-step",
            }
        )

    def exchange(
        self, state: LatticeBoltzmannMultiblockState, /
    ) -> LatticeBoltzmannMultiblockExchangeResult:
        if not isinstance(state, LatticeBoltzmannMultiblockState):
            raise TypeError("state must be LatticeBoltzmannMultiblockState.")
        if len(state.populations) != len(self.blocks):
            raise ValueError("Multiblock state and plan block counts do not match.")
        source = tuple(
            block.validate_populations(values)
            for block, values in zip(self.blocks, state.populations, strict=True)
        )
        exchanged = list(source)
        write_masks = [jnp.zeros(values.shape, dtype=bool) for values in source]
        reciprocity_residual = jnp.asarray(0.0, dtype=source[0].dtype)
        for connection in self.connections:
            interface = connection.interface
            left_index = connection.left_block
            right_index = connection.right_block
            right_in_left = interface.orient_right_trace(source[right_index])
            left_trace = interface.left_trace(source[left_index])
            left_in_right = interface.orient_left_trace_to_right(left_trace)
            right_round_trip = interface.orient_left_trace_to_right(right_in_left)
            left_round_trip = interface.orientation.apply(left_in_right, trailing_axes=1)[
                ..., interface.population_permutation
            ]
            right_trace = _boundary(
                source[right_index], interface.right_axis, interface.right_side
            )
            reciprocity_residual = jnp.maximum(
                reciprocity_residual,
                jnp.maximum(
                    jnp.max(jnp.abs(right_round_trip - right_trace)),
                    jnp.max(jnp.abs(left_round_trip - left_trace)),
                ),
            )
            left_velocities = interface.left.velocity_set.velocities
            right_velocities = interface.right.velocity_set.velocities
            left_incoming = (
                left_velocities[:, interface.left_axis] * _side_sign(interface.left_side)
                < 0
            )
            right_incoming = (
                right_velocities[:, interface.right_axis]
                * _side_sign(interface.right_side)
                < 0
            )
            exchanged[left_index] = _replace_boundary_directions(
                exchanged[left_index],
                interface.left_axis,
                interface.left_side,
                left_incoming,
                right_in_left,
            )
            exchanged[right_index] = _replace_boundary_directions(
                exchanged[right_index],
                interface.right_axis,
                interface.right_side,
                right_incoming,
                left_in_right,
            )
            write_masks[left_index] = _mark_boundary_directions(
                write_masks[left_index],
                interface.left_axis,
                interface.left_side,
                left_incoming,
            )
            write_masks[right_index] = _mark_boundary_directions(
                write_masks[right_index],
                interface.right_axis,
                interface.right_side,
                right_incoming,
            )
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in exchanged))
        )
        tolerance = 128.0 * jnp.finfo(source[0].dtype).eps
        successful = finite & (reciprocity_residual <= tolerance)
        write_count = sum(jnp.sum(mask) for mask in write_masks)
        evidence = LatticeBoltzmannMultiblockExchangeEvidence(
            reciprocity_residual,
            jnp.asarray(write_count, dtype=jnp.int32),
            jnp.asarray(len(self.connections), dtype=jnp.int32),
            successful,
        )
        return LatticeBoltzmannMultiblockExchangeResult(
            LatticeBoltzmannMultiblockState(exchanged),
            tuple(write_masks),
            evidence,
        )


__all__ = [
    "LatticeBoltzmannBlockConnection",
    "LatticeBoltzmannBlockInterfacePlan",
    "LatticeBoltzmannBlockSide",
    "LatticeBoltzmannBlockTracePair",
    "LatticeBoltzmannMultiblockCouplingPlan",
    "LatticeBoltzmannMultiblockExchangeEvidence",
    "LatticeBoltzmannMultiblockExchangeResult",
    "LatticeBoltzmannMultiblockState",
]
