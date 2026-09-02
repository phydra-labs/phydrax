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
from .._tensor_support import PreparedTensorGrid


def _forward(value, axis, spacing, periodic):
    shifted = (
        jnp.roll(value, -1, axis=axis)
        if periodic
        else jnp.concatenate(
            (
                jnp.take(value, jnp.arange(1, value.shape[axis]), axis=axis),
                jnp.take(value, jnp.asarray([value.shape[axis] - 1]), axis=axis),
            ),
            axis=axis,
        )
    )
    return (shifted - value) / spacing


def _backward(value, axis, spacing, periodic):
    if periodic:
        previous = jnp.roll(value, 1, axis=axis)
    else:
        pad_shape = list(value.shape)
        pad_shape[axis] = 1
        previous = jnp.concatenate(
            (
                jnp.zeros(tuple(pad_shape), dtype=value.dtype),
                jnp.take(value, jnp.arange(value.shape[axis] - 1), axis=axis),
            ),
            axis=axis,
        )
    return (value - previous) / spacing


class ReducedPICCurrentResult(StrictModule):
    start_charge: Array
    end_charge: Array
    current: tuple[Array, Array, Array]
    continuity_residual: Array
    maximum_continuity_defect: Array
    finite: Array
    boundary_flux: Array
    global_charge_defect: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReducedPICTransferPlan(StrictModule, NonTrainableState):
    """dD3V CIC transfer with conservative physical-boundary assignment."""

    grid: PreparedTensorGrid
    dimension: int = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    lower: tuple[float, ...] = eqx.field(static=True)
    spacing: tuple[float, ...] = eqx.field(static=True)
    upper: tuple[float, ...] = eqx.field(static=True)
    periodic: tuple[bool, ...] = eqx.field(static=True)
    maximum_path_segments: int = eqx.field(static=True)
    cell_volume: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        tolerance: float = 1.0e-9,
        maximum_path_segments: int = 16,
    ):
        if not isinstance(grid, PreparedTensorGrid) or len(grid.shape) not in (1, 2):
            raise TypeError("ReducedPICTransferPlan requires a prepared 1-D or 2-D grid.")
        segments = int(maximum_path_segments)
        if segments < 1:
            raise ValueError("maximum_path_segments must be positive.")
        widths = tuple(np.asarray(axis.interval_widths) for axis in grid.structured_axes)
        if any(not np.allclose(value, value[0]) for value in widths):
            raise ValueError("Reduced PIC currently requires uniform axes.")
        tolerance_ = float(tolerance)
        if tolerance_ <= 0.0 or not np.isfinite(tolerance_):
            raise ValueError("tolerance must be positive and finite.")
        self.grid = grid
        self.dimension = len(grid.shape)
        self.shape = tuple(
            int(axis.interval_centers.size) for axis in grid.structured_axes
        )
        self.lower = tuple(float(axis.bounds[0]) for axis in grid.structured_axes)
        self.upper = tuple(float(axis.bounds[1]) for axis in grid.structured_axes)
        self.periodic = tuple(bool(axis.periodic) for axis in grid.structured_axes)
        self.maximum_path_segments = segments
        self.spacing = tuple(float(value[0]) for value in widths)
        self.cell_volume = float(np.prod(self.spacing))
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reduced-pic-transfer",
                "grid": grid.prepared_id,
                "tolerance": tolerance_,
                "maximum_path_segments": segments,
            }
        )

    def _routes(self, position: Array):
        count = position.shape[0]
        lower = jnp.asarray(self.lower, dtype=position.dtype)
        spacing = jnp.asarray(self.spacing, dtype=position.dtype)
        coordinate = (position - lower) / spacing - 0.5
        base = jnp.floor(coordinate).astype(jnp.int32)
        fraction = coordinate - base
        route_count = 2**self.dimension
        indices = []
        weights = []
        for route in range(route_count):
            bits = tuple((route >> axis) & 1 for axis in range(self.dimension))
            component = []
            for axis in range(self.dimension):
                raw = base[:, axis] + bits[axis]
                if self.periodic[axis]:
                    index = jnp.mod(raw, self.shape[axis])
                else:
                    # Merge the exterior half-stencil into the boundary cell.
                    index = jnp.clip(raw, 0, self.shape[axis] - 1)
                component.append(index)
            if self.dimension == 1:
                flat = component[0]
            else:
                flat = component[0] * self.shape[1] + component[1]
            weight = jnp.ones((count,), dtype=position.dtype)
            for axis, bit in enumerate(bits):
                weight = weight * (fraction[:, axis] if bit else 1.0 - fraction[:, axis])
            indices.append(flat)
            weights.append(weight)
        return jnp.stack(tuple(indices), axis=-1), jnp.stack(tuple(weights), axis=-1)

    def deposit(
        self,
        position: ArrayLike,
        content: ArrayLike,
        active_mask: ArrayLike,
        /,
    ) -> Array:
        points = jnp.asarray(position)
        values = jnp.asarray(content, dtype=points.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        if points.ndim != 2 or points.shape[1] != self.dimension:
            raise ValueError("Reduced PIC positions have incompatible spatial dimension.")
        if values.shape != active.shape or values.shape != (points.shape[0],):
            raise ValueError("Reduced PIC payload/activity must preserve capacity.")
        indices, weights = self._routes(points)
        target = jnp.zeros((int(np.prod(self.shape)),), dtype=points.dtype)
        for route in range(indices.shape[1]):
            target = target.at[indices[:, route]].add(
                jnp.where(active, values * weights[:, route], 0.0)
            )
        return target.reshape(self.shape) / self.cell_volume

    def gather(
        self,
        position: ArrayLike,
        field: tuple[ArrayLike, ArrayLike, ArrayLike],
        active_mask: ArrayLike,
        /,
    ) -> Array:
        points = jnp.asarray(position)
        active = jnp.asarray(active_mask, dtype=bool)
        components = tuple(jnp.asarray(value) for value in field)
        if any(value.shape != self.shape for value in components):
            raise ValueError("Reduced PIC field components must match the grid shape.")
        indices, weights = self._routes(points)
        gathered = []
        for component in components:
            flat = component.reshape((-1,))
            value = jnp.zeros((points.shape[0],), dtype=component.dtype)
            for route in range(indices.shape[1]):
                value = value + weights[:, route] * flat[indices[:, route]]
            gathered.append(jnp.where(active, value, 0.0))
        return jnp.stack(tuple(gathered), axis=-1)

    def current(
        self,
        start_position: ArrayLike,
        end_position: ArrayLike,
        macrocharge: ArrayLike,
        velocity: ArrayLike,
        active_mask: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        boundary_result=None,
    ) -> ReducedPICCurrentResult:
        start = jnp.asarray(start_position)
        end = jnp.asarray(end_position, dtype=start.dtype)
        charge = jnp.asarray(macrocharge, dtype=start.dtype)
        velocity_ = jnp.asarray(velocity, dtype=start.dtype)
        active = jnp.asarray(active_mask, dtype=bool)
        dt = jnp.asarray(step_size, dtype=start.dtype).reshape(())
        if start.shape != end.shape or start.shape[1] != self.dimension:
            raise ValueError("Reduced PIC current positions are incompatible.")
        if charge.shape != active.shape or velocity_.shape != (start.shape[0], 3):
            raise ValueError("Reduced PIC current payloads preserve particle capacity.")
        displacement = jnp.abs(end - start) / jnp.asarray(self.spacing, dtype=start.dtype)
        required_segments = jnp.max(jnp.ceil(displacement), initial=1.0).astype(jnp.int32)
        path_capacity_exceeded = required_segments > self.maximum_path_segments
        rho_start = self.deposit(start, charge, active)
        rho_end = self.deposit(end, charge, active)
        midpoint = 0.5 * (start + end)
        raw = []
        for axis in range(self.dimension):
            raw.append(self.deposit(midpoint, charge * velocity_[:, axis], active))
        while len(raw) < 3:
            raw.append(self.deposit(midpoint, charge * velocity_[:, len(raw)], active))
        divergence = jnp.sum(
            jnp.stack(
                tuple(
                    _backward(raw[axis], axis, self.spacing[axis], self.periodic[axis])
                    for axis in range(self.dimension)
                )
            ),
            axis=0,
        )
        residual = (rho_end - rho_start) / dt + divergence
        corrected = list(raw)
        boundary_flux = jnp.zeros((self.dimension, 2), dtype=start.dtype)
        if all(self.periodic):
            transformed = jnp.fft.fftn(residual)
            eigenvalue = jnp.zeros(self.shape, dtype=start.dtype)
            for axis in range(self.dimension):
                frequency = 2.0 * jnp.pi * jnp.fft.fftfreq(self.shape[axis])
                axis_shape = [1] * self.dimension
                axis_shape[axis] = self.shape[axis]
                eigenvalue = (
                    eigenvalue
                    + (2.0 - 2.0 * jnp.cos(frequency)).reshape(axis_shape)
                    / self.spacing[axis] ** 2
                )
            safe = jnp.where(eigenvalue > 0.0, eigenvalue, 1.0)
            potential_hat = jnp.where(eigenvalue > 0.0, -transformed / safe, 0.0)
            potential = jnp.real(jnp.fft.ifftn(potential_hat))
            for axis in range(self.dimension):
                corrected[axis] = raw[axis] - _forward(
                    potential, axis, self.spacing[axis], True
                )
        else:
            correction_axis = self.periodic.index(False)
            correction = -self.spacing[correction_axis] * jnp.cumsum(
                residual, axis=correction_axis
            )
            corrected[correction_axis] = corrected[correction_axis] + correction
            upper_flux = jnp.take(
                correction,
                jnp.asarray([self.shape[correction_axis] - 1]),
                axis=correction_axis,
            )
            boundary_flux = boundary_flux.at[correction_axis, 1].set(
                jnp.sum(upper_flux) * self.cell_volume / self.spacing[correction_axis]
            )
        final_residual = (rho_end - rho_start) / dt + jnp.sum(
            jnp.stack(
                tuple(
                    _backward(
                        corrected[axis],
                        axis,
                        self.spacing[axis],
                        self.periodic[axis],
                    )
                    for axis in range(self.dimension)
                )
            ),
            axis=0,
        )
        maximum = jnp.max(jnp.abs(final_residual), initial=0.0)
        scale = jnp.maximum(
            1.0, jnp.max(jnp.abs((rho_end - rho_start) / dt), initial=0.0)
        )
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in corrected))
        ) & jnp.all(jnp.isfinite(final_residual))
        supplied_boundary_flux = (
            jnp.asarray(0.0, dtype=start.dtype)
            if boundary_result is None
            else jnp.sum(jnp.asarray(boundary_result.boundary_charge_flux)) / dt
        )
        global_defect = (
            jnp.sum(rho_end - rho_start) * self.cell_volume / dt
            + jnp.sum(boundary_flux)
            + supplied_boundary_flux
        )
        global_scale = jnp.maximum(
            jnp.sum(jnp.abs(rho_end - rho_start)) * self.cell_volume / dt,
            1.0,
        )
        successful = (
            finite
            & ~path_capacity_exceeded
            & (maximum <= self.tolerance * scale)
            & (jnp.abs(global_defect) <= self.tolerance * global_scale)
        )
        return ReducedPICCurrentResult(
            rho_start,
            rho_end,
            tuple(corrected),
            final_residual,
            maximum,
            finite,
            boundary_flux,
            global_defect,
            successful,
            self.plan_id,
        )


__all__ = ["ReducedPICCurrentResult", "ReducedPICTransferPlan"]
