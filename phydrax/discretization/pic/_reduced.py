#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._reduced_differences import backward_difference, forward_difference
from .._tensor_support import PreparedTensorGrid
from ._boundary import PICBoundaryResult


def _backward_gram_basis(
    count: int, spacing: float, periodic: bool, /
) -> tuple[Array, Array]:
    """Eigenpairs of ``B Bᵀ`` for the reduced backward difference ``B`` of one axis.

    Host preparation of the Poisson operator ``−div ∘ grad`` of the reduced
    Yee complex: periodic ``B`` is circulant with one null mode, nonperiodic
    ``B`` reads ``value[-1] = 0`` and is invertible.
    """
    identity = np.eye(count)
    shift = np.roll(identity, 1, axis=0) if periodic else np.eye(count, k=-1)
    backward = (identity - shift) / spacing
    values, vectors = np.linalg.eigh(backward @ backward.T)
    # The periodic null eigenvalue is rounded to zero so it stays masked exactly.
    values = np.where(values < 1.0e-12 / spacing**2, 0.0, values)
    return jnp.asarray(values), jnp.asarray(vectors)


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
    laplacian_bases: tuple[tuple[Array, Array], ...]
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
    ) -> None:
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
        self.shape = tuple(axis.interval_centers.size for axis in grid.structured_axes)
        self.lower = tuple(float(axis.bounds[0]) for axis in grid.structured_axes)
        self.upper = tuple(float(axis.bounds[1]) for axis in grid.structured_axes)
        self.periodic = tuple(bool(axis.periodic) for axis in grid.structured_axes)
        self.maximum_path_segments = segments
        self.spacing = tuple(float(value[0]) for value in widths)
        self.laplacian_bases = (
            ()
            if self.dimension == 1
            else tuple(
                _backward_gram_basis(count, spacing, periodic)
                for count, spacing, periodic in zip(
                    self.shape, self.spacing, self.periodic, strict=True
                )
            )
        )
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

    def _routes(self, position: Array) -> tuple[Array, Array]:
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
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
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
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
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
        boundary_result: PICBoundaryResult | None = None,
    ) -> ReducedPICCurrentResult:
        start = jnp.asarray(start_position)
        end = jnp.asarray(end_position, dtype=start.dtype)
        charge = jnp.asarray(macrocharge, dtype=start.dtype)
        velocity_ = jnp.asarray(velocity, dtype=start.dtype)
        active = jnp.asarray(active_mask, dtype=jnp.bool_)
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
                    backward_difference(
                        raw[axis], axis, self.spacing[axis], self.periodic[axis]
                    )
                    for axis in range(self.dimension)
                )
            ),
            axis=0,
        )
        residual = (rho_end - rho_start) / dt + divergence
        corrected = list(raw)
        if self.dimension == 2:
            # Poisson correction J ← J + Bᵀψ with (Σ_a B_a B_aᵀ)ψ = −residual,
            # diagonalized by the per-axis eigenbases; B is the divergence the
            # reduced Yee field pairs with (a nonperiodic axis reads J[-1] = 0).
            (x_values, x_vectors), (y_values, y_vectors) = self.laplacian_bases
            eigenvalue = x_values[:, None] + y_values[None, :]
            safe = jnp.where(eigenvalue > 0.0, eigenvalue, 1.0)
            potential = (
                x_vectors
                @ jnp.where(
                    eigenvalue > 0.0, -(x_vectors.T @ residual @ y_vectors) / safe, 0.0
                )
                @ y_vectors.T
            )
            for axis in range(2):
                corrected[axis] = raw[axis] - forward_difference(
                    potential, axis, self.spacing[axis], self.periodic[axis]
                )
        elif self.periodic[0]:
            transformed = jnp.fft.fft(residual)
            eigenvalue = (
                2.0 - 2.0 * jnp.cos(2.0 * jnp.pi * jnp.fft.fftfreq(self.shape[0]))
            ) / self.spacing[0] ** 2
            safe = jnp.where(eigenvalue > 0.0, eigenvalue, 1.0)
            potential = jnp.real(
                jnp.fft.ifft(jnp.where(eigenvalue > 0.0, -transformed / safe, 0.0))
            )
            corrected[0] = raw[0] - forward_difference(
                potential, 0, self.spacing[0], True
            )
        else:
            corrected[0] = raw[0] - self.spacing[0] * jnp.cumsum(residual)
        # The corrected current on the upper face of a nonperiodic axis is the
        # physical outflow; the lower wall carries none.
        boundary_flux = jnp.zeros((self.dimension, 2), dtype=start.dtype)
        for axis in range(self.dimension):
            if not self.periodic[axis]:
                upper_flux = jnp.take(
                    corrected[axis], jnp.asarray([self.shape[axis] - 1]), axis=axis
                )
                boundary_flux = boundary_flux.at[axis, 1].set(
                    jnp.sum(upper_flux) * self.cell_volume / self.spacing[axis]
                )
        final_residual = (rho_end - rho_start) / dt + jnp.sum(
            jnp.stack(
                tuple(
                    backward_difference(
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
            (corrected[0], corrected[1], corrected[2]),
            final_residual,
            maximum,
            finite,
            boundary_flux,
            global_defect,
            successful,
            self.plan_id,
        )


__all__ = ["ReducedPICCurrentResult", "ReducedPICTransferPlan"]
