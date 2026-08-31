#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._transfer import PreparedPICParticleCochainTransfer
from ._types import PICCurrentDepositResult


def _flat_index(indices: tuple[Array, Array, Array], shape: tuple[int, ...], /) -> Array:
    return (indices[0] * shape[1] + indices[1]) * shape[2] + indices[2]


def _linear_factor(start: Array, end: Array, bit: int, /) -> tuple[Array, Array]:
    delta = end - start
    return (start, delta) if bit else (1.0 - start, -delta)


def _integrated_product(
    first: tuple[Array, Array], second: tuple[Array, Array], /
) -> Array:
    a, b = first
    c, d = second
    return a * c + 0.5 * (a * d + b * c) + (b * d) / 3.0


class ChargeConservingCurrentPlan(StrictModule, NonTrainableState):
    """Local cubical Whitney current satisfying exact discrete continuity."""

    transfer: PreparedPICParticleCochainTransfer
    maximum_segments_per_particle: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedPICParticleCochainTransfer,
        /,
        *,
        maximum_segments_per_particle: int = 4,
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(transfer, PreparedPICParticleCochainTransfer):
            raise TypeError("transfer must be PreparedPICParticleCochainTransfer.")
        if transfer.bridge.dimension != 3:
            raise ValueError("Charge-conserving current currently requires a 3-D bridge.")
        if any(not axis.periodic for axis in transfer.bridge.grid.structured_axes):
            raise ValueError("Charge-conserving current currently requires periodic axes.")
        widths = tuple(np.asarray(axis.interval_widths) for axis in transfer.bridge.grid.structured_axes)
        if any(not np.allclose(value, value[0], rtol=1e-12, atol=1e-14) for value in widths):
            raise ValueError("Charge-conserving current currently requires uniform axes.")
        segments = int(maximum_segments_per_particle)
        tolerance_ = float(tolerance)
        if segments != 4:
            raise ValueError(
                "maximum_segments_per_particle must be four for one-cell-per-axis paths."
            )
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("tolerance must be positive and finite.")
        self.transfer = transfer
        self.maximum_segments_per_particle = segments
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "charge-conserving-whitney-current",
                "transfer": transfer.prepared_id,
                "segments": segments,
                "tolerance": tolerance_,
            }
        )

    def _segments(self, start: Array, end: Array, /):
        axes = self.transfer.bridge.grid.structured_axes
        lower = jnp.asarray([axis.bounds[0] for axis in axes], dtype=start.dtype)
        spacing = jnp.asarray([axis.interval_widths[0] for axis in axes], dtype=start.dtype)
        q0 = (start - lower) / spacing
        q1 = (end - lower) / spacing
        delta = q1 - q0
        epsilon = 32.0 * jnp.finfo(start.dtype).eps
        direction = jnp.sign(delta)
        q0_side = q0 + epsilon * direction
        start_cell = jnp.floor(q0_side)
        boundary = jnp.where(delta > 0.0, start_cell + 1.0, start_cell)
        safe_delta = jnp.where(jnp.abs(delta) > epsilon, delta, 1.0)
        crossing = (boundary - q0) / safe_delta
        valid_crossing = (
            (jnp.abs(delta) > epsilon)
            & (crossing > epsilon)
            & (crossing < 1.0 - epsilon)
        )
        crossing = jnp.where(valid_crossing, crossing, 1.0)
        times = jnp.sort(
            jnp.concatenate(
                (
                    jnp.zeros((start.shape[0], 1), dtype=start.dtype),
                    crossing,
                    jnp.ones((start.shape[0], 1), dtype=start.dtype),
                ),
                axis=1,
            ),
            axis=1,
        )
        segment_start = times[:, :-1]
        segment_end = times[:, 1:]
        valid = segment_end - segment_start > epsilon
        midpoint_t = 0.5 * (segment_start + segment_end)
        midpoint = q0[:, None, :] + midpoint_t[..., None] * delta[:, None, :]
        cell_unwrapped = jnp.floor(midpoint + epsilon * direction[:, None, :]).astype(jnp.int32)
        local_start = (
            q0[:, None, :]
            + segment_start[..., None] * delta[:, None, :]
            - cell_unwrapped
        )
        local_end = (
            q0[:, None, :]
            + segment_end[..., None] * delta[:, None, :]
            - cell_unwrapped
        )
        overflow = jnp.any(jnp.abs(delta) > 1.0 + epsilon, axis=-1)
        return local_start, local_end, cell_unwrapped, valid, overflow

    def deposit(
        self,
        start_position: ArrayLike,
        end_position: ArrayLike,
        step_size: ArrayLike,
        /,
    ) -> PICCurrentDepositResult:
        start = jnp.asarray(start_position)
        end = jnp.asarray(end_position, dtype=start.dtype)
        expected = (self.transfer.species.capacity, 3)
        if start.shape != expected or end.shape != expected:
            raise ValueError(f"Current-deposition positions must have shape {expected}.")
        dt = jnp.asarray(step_size, dtype=start.dtype).reshape(())
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0.0), "step_size must be positive and finite."
        )
        start_routes = self.transfer.build(start)
        end_routes = self.transfer.build(end)
        start_charge = self.transfer.deposit_charge(start_routes)
        end_charge = self.transfer.deposit_charge(end_routes)
        local_start, local_end, cell, segment_valid, particle_overflow = self._segments(
            start, end
        )
        active = self.transfer.species.particles.active_mask
        segment_valid = segment_valid & active[:, None]
        counts = jnp.sum(segment_valid, axis=1, dtype=jnp.int32)
        overflow = jnp.any(particle_overflow & active) | jnp.any(
            counts > self.maximum_segments_per_particle
        )
        bridge = self.transfer.bridge
        shapes = bridge.orientation_shapes[1]
        offsets = bridge.orientation_offsets[1]
        interval_counts = jnp.asarray(
            [axis.interval_centers.size for axis in bridge.grid.structured_axes],
            dtype=jnp.int32,
        )
        point_counts = jnp.asarray(
            [axis.point_coordinates.size for axis in bridge.grid.structured_axes],
            dtype=jnp.int32,
        )
        charges = self.transfer.species.charges.astype(start.dtype)
        contribution_indices = []
        contribution_values = []
        contribution_valid = []
        for axis in range(3):
            transverse = tuple(value for value in range(3) if value != axis)
            for first_bit in (0, 1):
                for second_bit in (0, 1):
                    first_factor = _linear_factor(
                        local_start[..., transverse[0]],
                        local_end[..., transverse[0]],
                        first_bit,
                    )
                    second_factor = _linear_factor(
                        local_start[..., transverse[1]],
                        local_end[..., transverse[1]],
                        second_bit,
                    )
                    integral = (
                        local_end[..., axis] - local_start[..., axis]
                    ) * _integrated_product(first_factor, second_factor)
                    index_components = []
                    for coordinate_axis in range(3):
                        if coordinate_axis == axis:
                            index_components.append(
                                jnp.mod(cell[..., coordinate_axis], interval_counts[coordinate_axis])
                            )
                        else:
                            bit = first_bit if coordinate_axis == transverse[0] else second_bit
                            index_components.append(
                                jnp.mod(
                                    cell[..., coordinate_axis] + bit,
                                    point_counts[coordinate_axis],
                                )
                            )
                    flat = offsets[axis] + _flat_index(
                        tuple(index_components), shapes[axis]
                    )
                    contribution_indices.append(flat)
                    contribution_values.append(-charges[:, None] * integral / dt)
                    contribution_valid.append(segment_valid)
        indices = jnp.stack(tuple(contribution_indices), axis=-1).reshape((-1,))
        values = jnp.stack(tuple(contribution_values), axis=-1).reshape((-1,))
        valid = jnp.stack(tuple(contribution_valid), axis=-1).reshape((-1,))
        flux_content = jnp.zeros(
            (bridge.cochain.cell_counts[1],), dtype=start.dtype
        )

        def scatter(index, carry):
            return carry.at[indices[index]].add(jnp.where(valid[index], values[index], 0.0))

        flux_content = jax.lax.fori_loop(0, indices.size, scatter, flux_content)
        current = bridge.cochain.solve_hodge(1, flux_content)
        continuity = (
            end_charge.cochain - start_charge.cochain
        ) / dt + bridge.codifferential(1, current)
        maximum = jnp.max(jnp.abs(continuity), initial=0.0)
        scale = jnp.maximum(
            1.0,
            jnp.max(
                jnp.abs((end_charge.cochain - start_charge.cochain) / dt),
                initial=0.0,
            ),
        )
        finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(continuity))
            & jnp.all(jnp.isfinite(start))
            & jnp.all(jnp.isfinite(end))
        )
        successful = (
            start_charge.successful
            & end_charge.successful
            & ~overflow
            & finite
            & (maximum <= self.tolerance * scale)
        )
        return PICCurrentDepositResult(
            start_charge,
            end_charge,
            current,
            continuity,
            maximum,
            jnp.sum(counts, dtype=jnp.int32),
            overflow,
            finite,
            successful,
            self.plan_id,
        )


class PICMaxwellCurrentArguments(StrictModule):
    particle_current: Array
    external_arguments: object


class PICMaxwellCurrentSource(StrictModule, NonTrainableState):
    """Stable Maxwell callback extracting a deposited midpoint edge current."""

    source_id: str = eqx.field(static=True)

    def __init__(self, source_id: str = "pic-midpoint-current", /):
        identifier = str(source_id)
        if not identifier:
            raise ValueError("source_id must be nonempty.")
        self.source_id = identifier

    def __call__(self, time: Array, coordinates: Array, args: object, /) -> Array:
        del time, coordinates
        if not isinstance(args, PICMaxwellCurrentArguments):
            raise TypeError("Maxwell PIC current requires PICMaxwellCurrentArguments.")
        return args.particle_current


__all__ = [
    "ChargeConservingCurrentPlan",
    "PICMaxwellCurrentArguments",
    "PICMaxwellCurrentSource",
]
