#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._numerics._quadrature_rules import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import canonical_row_route_ids, EdgeRelation, RelationExecutionPlan
from ..splatting._assignment import _basis_and_derivative
from ._binning import PICCellBinningPlan
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
    """Physical tail-to-head spline-Whitney current with ``rho_dot - delta(J) = 0``.

    The shape order is the transfer's ``shape_order``: order one integrates the
    lowest-order Whitney forms in closed form; orders two and three integrate
    the spline-Whitney path integrals exactly (`_spline_whitney_flux`).
    """

    transfer: PreparedPICParticleCochainTransfer
    binning: PICCellBinningPlan
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
    ) -> None:
        if not isinstance(transfer, PreparedPICParticleCochainTransfer):
            raise TypeError("transfer must be PreparedPICParticleCochainTransfer.")
        if transfer.bridge.dimension != 3:
            raise ValueError("Charge-conserving current currently requires a 3-D bridge.")
        if any(not axis.periodic for axis in transfer.bridge.grid.structured_axes):
            raise ValueError(
                "Charge-conserving current currently requires periodic axes."
            )
        widths = tuple(
            np.asarray(axis.interval_widths)
            for axis in transfer.bridge.grid.structured_axes
        )
        if any(
            not np.allclose(value, value[0], rtol=1e-12, atol=1e-14) for value in widths
        ):
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
        self.binning = PICCellBinningPlan(
            tuple(float(axis.bounds[0]) for axis in transfer.bridge.grid.structured_axes),
            tuple(float(axis.bounds[1]) for axis in transfer.bridge.grid.structured_axes),
            tuple(
                axis.interval_centers.size
                for axis in transfer.bridge.grid.structured_axes
            ),
            (True, True, True),
        )
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

    def _segments(
        self, start: Array, end: Array, shift: float, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Split paths at the knot lattice ``shift + ℤ`` of every axis.

        Returns per-segment local coordinates inside the knot cell
        ``cell + shift`` (normalized grid units), the unwrapped cell, validity,
        and per-particle overflow of the one-cell-per-axis path bound.
        """
        axes = self.transfer.bridge.grid.structured_axes
        lower = jnp.asarray([axis.bounds[0] for axis in axes], dtype=start.dtype)
        spacing = jnp.asarray(
            [axis.interval_widths[0] for axis in axes], dtype=start.dtype
        )
        q0 = (start - lower) / spacing - shift
        q1 = (end - lower) / spacing - shift
        delta = q1 - q0
        epsilon = 32.0 * jnp.finfo(start.dtype).eps
        direction = jnp.sign(delta)
        q0_side = q0 + epsilon * direction
        start_cell = jnp.floor(q0_side)
        boundary = jnp.where(delta > 0.0, start_cell + 1.0, start_cell)
        safe_delta = jnp.where(jnp.abs(delta) > epsilon, delta, 1.0)
        crossing = (boundary - q0) / safe_delta
        valid_crossing = (
            (jnp.abs(delta) > epsilon) & (crossing > epsilon) & (crossing < 1.0 - epsilon)
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
        cell_unwrapped = jnp.floor(midpoint + epsilon * direction[:, None, :]).astype(
            jnp.int32
        )
        local_start = (
            q0[:, None, :] + segment_start[..., None] * delta[:, None, :] - cell_unwrapped
        )
        local_end = (
            q0[:, None, :] + segment_end[..., None] * delta[:, None, :] - cell_unwrapped
        )
        overflow = jnp.any(jnp.abs(delta) > 1.0 + epsilon, axis=-1)
        return local_start, local_end, cell_unwrapped, valid, overflow

    def _whitney_flux(
        self, start: Array, end: Array, charges: Array, active: Array, dt: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        """Lowest-order Whitney flux content by closed-form segment integrals."""
        local_start, local_end, cell, segment_valid, particle_overflow = self._segments(
            start, end, 0.0
        )
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
                                jnp.mod(
                                    cell[..., coordinate_axis],
                                    interval_counts[coordinate_axis],
                                )
                            )
                        else:
                            bit = (
                                first_bit
                                if coordinate_axis == transverse[0]
                                else second_bit
                            )
                            index_components.append(
                                jnp.mod(
                                    cell[..., coordinate_axis] + bit,
                                    point_counts[coordinate_axis],
                                )
                            )
                    flat = offsets[axis] + _flat_index(
                        (index_components[0], index_components[1], index_components[2]),
                        shapes[axis],
                    )
                    contribution_indices.append(flat)
                    contribution_values.append(charges[:, None] * integral / dt)
                    contribution_valid.append(segment_valid)
        indices = jnp.stack(tuple(contribution_indices), axis=-1).reshape((-1,))
        values = jnp.stack(tuple(contribution_values), axis=-1).reshape((-1,))
        valid = jnp.stack(tuple(contribution_valid), axis=-1).reshape((-1,))
        flux_content = jnp.zeros((bridge.cochain.cell_counts[1],), dtype=start.dtype)

        def scatter(index: Array, carry: Array) -> Array:
            return carry.at[indices[index]].add(
                jnp.where(valid[index], values[index], 0.0)
            )

        flux_content = jax.lax.fori_loop(0, indices.size, scatter, flux_content)
        return flux_content, counts, overflow, jnp.asarray(True)

    def _spline_whitney_flux(
        self, start: Array, end: Array, charges: Array, active: Array, dt: Array, /
    ) -> tuple[Array, Array, Array, Array]:
        """Order-``p`` spline-Whitney flux content by exact segment quadrature.

        Along a straight path the edge current of axis ``a`` is
        ``q/Δt ∫ N^{p-1}(q_a − e − ½) N^p(q_b − j) N^p(q_c − k) dq_a``. Splitting
        each path at the common knot lattice of those splines (integers for odd
        ``p``, half-integers for even ``p``) leaves polynomial integrands of
        degree ``3p − 1`` in the path parameter, integrated exactly by
        ``⌈3p/2⌉``-point Gauss–Legendre. ``dN^p(x − i)/dx = N^{p-1}(x − i + ½) −
        N^{p-1}(x − i − ½)`` then makes the discrete continuity with the
        degree-``p`` vertex charge exact to roundoff. Contributions reduce in
        canonical cell-binned particle order, so the current is invariant to
        particle slot order.
        """
        order = self.transfer.plan.shape_order
        shift = 0.5 * ((order - 1) % 2)
        local_start, local_end, cell, segment_valid, particle_overflow = self._segments(
            start, end, shift
        )
        segment_valid = segment_valid & active[:, None]
        counts = jnp.sum(segment_valid, axis=1, dtype=jnp.int32)
        overflow = jnp.any(particle_overflow & active) | jnp.any(
            counts > self.maximum_segments_per_particle
        )
        dtype = start.dtype
        origin = shift + cell.astype(dtype)
        q_start = origin + local_start
        q_end = origin + local_end
        delta = q_end - q_start
        midpoint = 0.5 * (q_start + q_end)
        rule = gauss_legendre_data((3 * order + 1) // 2)
        nodes = (0.5 * (rule.nodes + 1.0)).astype(dtype)
        weights = (0.5 * rule.weights).astype(dtype)
        points = q_start[..., None, :] + nodes[:, None] * delta[..., None, :]

        def axis_shape(axis: int, degree: int, offset: float) -> tuple[Array, Array]:
            # Nodes base..base+degree carry the piece containing the segment.
            base = jnp.floor(midpoint[..., axis] - offset - 0.5 * (degree - 1))
            node = base[..., None] + jnp.arange(degree + 1, dtype=dtype)
            local = (points[..., axis] - offset)[..., None] - node[..., None, :]
            value, _ = _basis_and_derivative(degree, local)
            return node.astype(jnp.int32), value

        bridge = self.transfer.bridge
        shapes = bridge.orientation_shapes[1]
        offsets = bridge.orientation_offsets[1]
        axes = bridge.grid.structured_axes
        indices = []
        values = []
        valid = []
        for axis in range(3):
            first, second = tuple(value for value in range(3) if value != axis)
            along_index, along = axis_shape(axis, order - 1, 0.5)
            first_index, first_value = axis_shape(first, order, 0.0)
            second_index, second_value = axis_shape(second, order, 0.0)
            integral = delta[..., axis, None, None, None] * ein.contract(
                "g,nsgi,nsgj,nsgk->nsijk", weights, along, first_value, second_value
            )
            grids = {
                axis: along_index[..., :, None, None] % axes[axis].interval_centers.size,
                first: (
                    first_index[..., None, :, None] % axes[first].point_coordinates.size
                ),
                second: (
                    second_index[..., None, None, :] % axes[second].point_coordinates.size
                ),
            }
            flat = offsets[axis] + _flat_index(
                (grids[0], grids[1], grids[2]), shapes[axis]
            )
            indices.append(
                jnp.broadcast_to(flat, integral.shape).reshape((start.shape[0], -1))
            )
            values.append(
                (charges[:, None, None, None, None] * integral / dt).reshape(
                    (start.shape[0], -1)
                )
            )
            valid.append(
                jnp.broadcast_to(
                    segment_valid[..., None, None, None], integral.shape
                ).reshape((start.shape[0], -1))
            )
        route_indices = jnp.concatenate(indices, axis=1)
        route_values = jnp.concatenate(values, axis=1)
        route_valid = jnp.concatenate(valid, axis=1)
        routes = route_indices.shape[1]
        bins = self.binning.bin(
            start,
            active,
            identity=(
                *(start[:, axis] for axis in range(3)),
                *(end[:, axis] for axis in range(3)),
                charges,
            ),
        )
        relation = EdgeRelation(
            jnp.arange(route_indices.size, dtype=jnp.int32),
            route_indices.reshape((-1,)),
            source_size=route_indices.size,
            target_size=bridge.cochain.cell_counts[1],
            valid=route_valid.reshape((-1,)),
        )
        execution = RelationExecutionPlan().prepare(
            relation, stable_route_ids=canonical_row_route_ids(bins.order, routes)
        )
        flux_content, evidence = execution.reduce(
            route_values.reshape((-1,)), accumulation="fast", output="dense"
        )
        return flux_content, counts, overflow, evidence.successful

    def deposit(
        self,
        start_position: ArrayLike,
        end_position: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        macrocharge: ArrayLike | None = None,
        active_mask: ArrayLike | None = None,
    ) -> PICCurrentDepositResult:
        """Deposit tail-to-head current and endpoint charge.

        ``macrocharge`` and ``active_mask`` are the runtime population charge and
        activity; omitted, the prepared species charges and activity are used.
        """
        start = jnp.asarray(start_position)
        end = jnp.asarray(end_position, dtype=start.dtype)
        expected = (self.transfer.species.capacity, 3)
        if start.shape != expected or end.shape != expected:
            raise ValueError(f"Current-deposition positions must have shape {expected}.")
        charges = (
            self.transfer.species.charges
            if macrocharge is None
            else jnp.asarray(macrocharge)
        ).astype(start.dtype)
        active = self.transfer.species.particles.active_mask
        if active_mask is not None:
            runtime_active = jnp.asarray(active_mask, dtype=jnp.bool_)
            if runtime_active.shape != active.shape:
                raise ValueError("active_mask must have particle-capacity shape.")
            active = active & runtime_active
        if charges.shape != active.shape:
            raise ValueError("macrocharge must have particle-capacity shape.")
        dt = jnp.asarray(step_size, dtype=start.dtype).reshape(())
        dt = eqx.error_if(
            dt, ~jnp.isfinite(dt) | (dt <= 0.0), "step_size must be positive and finite."
        )
        start_routes = self.transfer.build(start, active_mask=active_mask)
        end_routes = self.transfer.build(end, active_mask=active_mask)
        start_charge = self.transfer.deposit_macrocharge(start_routes, charges)
        end_charge = self.transfer.deposit_macrocharge(end_routes, charges)
        flux_content, counts, overflow, reduced = (
            self._whitney_flux(start, end, charges, active, dt)
            if self.transfer.plan.shape_order == 1
            else self._spline_whitney_flux(start, end, charges, active, dt)
        )
        bridge = self.transfer.bridge
        current = bridge.cochain.solve_hodge(1, flux_content)
        continuity = (
            end_charge.cochain - start_charge.cochain
        ) / dt - bridge.codifferential(1, current)
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
            & reduced
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


__all__ = [
    "ChargeConservingCurrentPlan",
    "PICMaxwellCurrentArguments",
]
