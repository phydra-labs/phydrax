#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import PreparedTensorGrid


def _forward(value: Array, axis: int, spacing: float) -> Array:
    return (jnp.roll(value, -1, axis=axis) - value) / spacing


def _backward(value: Array, axis: int, spacing: float) -> Array:
    return (value - jnp.roll(value, 1, axis=axis)) / spacing


class ReducedMaxwellDiagnostics(StrictModule):
    energy: Array
    electric_constraint_linf: Array
    magnetic_constraint_linf: Array
    source_power: Array
    power_balance_residual: Array
    step_fraction: Array
    finite: Array
    stable: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class CompatibleMaxwell2DState(StrictModule):
    electric: tuple[Array, Array, Array]
    magnetic: tuple[Array, Array, Array]
    charge: Array


class CompatibleMaxwell2DPlan(StrictModule, NonTrainableState):
    """Periodic 2D3V Yee/de-Rham Maxwell block with explicit staggering."""

    grid: PreparedTensorGrid
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    courant_factor: float = eqx.field(static=True)
    shape: tuple[int, int] = eqx.field(static=True)
    spacing: tuple[float, float] = eqx.field(static=True)
    stable_dt: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        courant_factor: float = 0.95,
    ):
        if not isinstance(grid, PreparedTensorGrid) or len(grid.shape) != 2:
            raise TypeError(
                "CompatibleMaxwell2DPlan requires a prepared 2-D tensor grid."
            )
        if any(not axis.periodic for axis in grid.structured_axes):
            raise ValueError("Initial reduced Maxwell requires periodic axes.")
        epsilon, mu, courant = (
            float(permittivity),
            float(permeability),
            float(courant_factor),
        )
        if epsilon <= 0.0 or mu <= 0.0 or not 0.0 < courant <= 1.0:
            raise ValueError("Reduced Maxwell material/Courant values are invalid.")
        widths = tuple(np.asarray(axis.interval_widths) for axis in grid.structured_axes)
        if any(not np.allclose(value, value[0]) for value in widths):
            raise ValueError("Reduced Maxwell currently requires uniform axes.")
        spacing = (float(widths[0][0]), float(widths[1][0]))
        shape = (
            int(grid.structured_axes[0].interval_centers.size),
            int(grid.structured_axes[1].interval_centers.size),
        )
        wave_speed = 1.0 / np.sqrt(epsilon * mu)
        stable = courant / (
            wave_speed * np.sqrt(sum(1.0 / value**2 for value in spacing))
        )
        self.grid = grid
        self.permittivity = epsilon
        self.permeability = mu
        self.courant_factor = courant
        self.shape = shape
        self.spacing = spacing
        self.stable_dt = stable
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compatible-maxwell-2d3v",
                "grid": grid.prepared_id,
                "epsilon": epsilon,
                "mu": mu,
                "courant": courant,
            }
        )

    def initialize(
        self,
        *,
        electric: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        magnetic: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
        charge: ArrayLike | None = None,
    ) -> CompatibleMaxwell2DState:
        zero = jnp.zeros(self.shape)
        e = (
            (zero, zero, zero)
            if electric is None
            else tuple(jnp.asarray(v) for v in electric)
        )
        b = (
            (zero, zero, zero)
            if magnetic is None
            else tuple(jnp.asarray(v) for v in magnetic)
        )
        rho = zero if charge is None else jnp.asarray(charge)
        if any(value.shape != self.shape for value in e + b) or rho.shape != self.shape:
            raise ValueError(
                "Reduced 2-D Maxwell fields must share the cell-count shape."
            )
        return CompatibleMaxwell2DState(e, b, rho)

    def divergence_electric(self, state: CompatibleMaxwell2DState, /) -> Array:
        ex, ey, _ = state.electric
        dx, dy = self.spacing
        return self.permittivity * (_backward(ex, 0, dx) + _backward(ey, 1, dy))

    def divergence_magnetic(self, state: CompatibleMaxwell2DState, /) -> Array:
        bx, by, _ = state.magnetic
        dx, dy = self.spacing
        return _backward(bx, 0, dx) + _backward(by, 1, dy)

    def energy(self, state: CompatibleMaxwell2DState, /) -> Array:
        volume = self.spacing[0] * self.spacing[1]
        e2 = jnp.sum(jnp.stack(tuple(jnp.sum(value * value) for value in state.electric)))
        b2 = jnp.sum(jnp.stack(tuple(jnp.sum(value * value) for value in state.magnetic)))
        return 0.5 * volume * (self.permittivity * e2 + b2 / self.permeability)

    def step(
        self,
        state: CompatibleMaxwell2DState,
        electric_current: tuple[ArrayLike, ArrayLike, ArrayLike],
        step_size: ArrayLike,
        /,
    ) -> tuple[CompatibleMaxwell2DState, ReducedMaxwellDiagnostics]:
        dt = jnp.asarray(step_size).reshape(())
        current = tuple(jnp.asarray(value) for value in electric_current)
        if any(value.shape != self.shape for value in current):
            raise ValueError("Reduced 2-D current components must match the field shape.")
        ex, ey, ez = state.electric
        bx, by, bz = state.magnetic
        dx, dy = self.spacing
        half_bx = bx - 0.5 * dt * _forward(ez, 1, dy)
        half_by = by + 0.5 * dt * _forward(ez, 0, dx)
        half_bz = bz - 0.5 * dt * (_forward(ey, 0, dx) - _forward(ex, 1, dy))
        jx, jy, jz = current
        next_ex = ex + dt / self.permittivity * (
            _backward(half_bz / self.permeability, 1, dy) - jx
        )
        next_ey = ey + dt / self.permittivity * (
            -_backward(half_bz / self.permeability, 0, dx) - jy
        )
        next_ez = ez + dt / self.permittivity * (
            _backward(half_by / self.permeability, 0, dx)
            - _backward(half_bx / self.permeability, 1, dy)
            - jz
        )
        next_bx = half_bx - 0.5 * dt * _forward(next_ez, 1, dy)
        next_by = half_by + 0.5 * dt * _forward(next_ez, 0, dx)
        next_bz = half_bz - 0.5 * dt * (
            _forward(next_ey, 0, dx) - _forward(next_ex, 1, dy)
        )
        next_charge = state.charge - dt * (_backward(jx, 0, dx) + _backward(jy, 1, dy))
        candidate = CompatibleMaxwell2DState(
            (next_ex, next_ey, next_ez),
            (next_bx, next_by, next_bz),
            next_charge,
        )
        old_energy = self.energy(state)
        new_energy = self.energy(candidate)
        source_power = (
            self.spacing[0]
            * self.spacing[1]
            * sum(
                jnp.sum(e * current_component)
                for e, current_component in zip(candidate.electric, current, strict=True)
            )
        )
        gauss = jnp.max(jnp.abs(self.divergence_electric(candidate) - next_charge))
        magnetic = jnp.max(jnp.abs(self.divergence_magnetic(candidate)))
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in candidate.electric + candidate.magnetic
                )
            )
        ) & jnp.all(jnp.isfinite(next_charge))
        stable = jnp.isfinite(dt) & (dt > 0.0) & (dt <= self.stable_dt)
        successful = finite & stable
        accepted = CompatibleMaxwell2DState(
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.electric, state.electric, strict=True)
            ),
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.magnetic, state.magnetic, strict=True)
            ),
            jnp.where(successful, candidate.charge, state.charge),
        )
        diagnostics = ReducedMaxwellDiagnostics(
            new_energy,
            gauss,
            magnetic,
            source_power,
            new_energy - old_energy + dt * source_power,
            dt / self.stable_dt,
            finite,
            stable,
            successful,
            self.plan_id,
        )
        return accepted, diagnostics


class CompatibleMaxwell1DState(StrictModule):
    electric: tuple[Array, Array, Array]
    magnetic: tuple[Array, Array, Array]
    charge: Array


class CompatibleMaxwell1DPlan(StrictModule, NonTrainableState):
    """Periodic 1D3V compatible longitudinal/transverse Maxwell blocks."""

    grid: PreparedTensorGrid
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    courant_factor: float = eqx.field(static=True)
    count: int = eqx.field(static=True)
    spacing: float = eqx.field(static=True)
    stable_dt: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        /,
        *,
        permittivity: float = 1.0,
        permeability: float = 1.0,
        courant_factor: float = 0.95,
    ):
        if not isinstance(grid, PreparedTensorGrid) or len(grid.shape) != 1:
            raise TypeError(
                "CompatibleMaxwell1DPlan requires a prepared 1-D tensor grid."
            )
        axis = grid.structured_axes[0]
        if not axis.periodic:
            raise ValueError("Initial reduced Maxwell requires a periodic axis.")
        epsilon, mu, courant = (
            float(permittivity),
            float(permeability),
            float(courant_factor),
        )
        widths = np.asarray(axis.interval_widths)
        if (
            epsilon <= 0.0
            or mu <= 0.0
            or not 0.0 < courant <= 1.0
            or not np.allclose(widths, widths[0])
        ):
            raise ValueError("Reduced 1-D Maxwell parameters/grid are invalid.")
        spacing = float(widths[0])
        stable = courant * spacing * np.sqrt(epsilon * mu)
        self.grid = grid
        self.permittivity = epsilon
        self.permeability = mu
        self.courant_factor = courant
        self.count = int(axis.interval_centers.size)
        self.spacing = spacing
        self.stable_dt = stable
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compatible-maxwell-1d3v",
                "grid": grid.prepared_id,
                "epsilon": epsilon,
                "mu": mu,
                "courant": courant,
            }
        )

    def initialize(self) -> CompatibleMaxwell1DState:
        zero = jnp.zeros((self.count,))
        return CompatibleMaxwell1DState((zero, zero, zero), (zero, zero, zero), zero)

    def energy(self, state: CompatibleMaxwell1DState, /) -> Array:
        electric_energy = jnp.sum(
            jnp.stack(tuple(jnp.sum(value**2) for value in state.electric))
        )
        magnetic_energy = jnp.sum(
            jnp.stack(tuple(jnp.sum(value**2) for value in state.magnetic))
        )
        return (
            0.5
            * self.spacing
            * (self.permittivity * electric_energy + magnetic_energy / self.permeability)
        )

    def step(
        self,
        state: CompatibleMaxwell1DState,
        electric_current: tuple[ArrayLike, ArrayLike, ArrayLike],
        step_size: ArrayLike,
        /,
    ) -> tuple[CompatibleMaxwell1DState, ReducedMaxwellDiagnostics]:
        dt = jnp.asarray(step_size).reshape(())
        current = tuple(jnp.asarray(value) for value in electric_current)
        if any(value.shape != (self.count,) for value in current):
            raise ValueError("Reduced 1-D currents must match the grid count.")
        ex, ey, ez = state.electric
        bx, by, bz = state.magnetic
        half_by = by + 0.5 * dt * _forward(ez, 0, self.spacing)
        half_bz = bz - 0.5 * dt * _forward(ey, 0, self.spacing)
        jx, jy, jz = current
        next_ex = ex - dt * jx / self.permittivity
        next_ey = ey + dt / self.permittivity * (
            -_backward(half_bz / self.permeability, 0, self.spacing) - jy
        )
        next_ez = ez + dt / self.permittivity * (
            _backward(half_by / self.permeability, 0, self.spacing) - jz
        )
        next_by = half_by + 0.5 * dt * _forward(next_ez, 0, self.spacing)
        next_bz = half_bz - 0.5 * dt * _forward(next_ey, 0, self.spacing)
        charge = state.charge - dt * _backward(jx, 0, self.spacing)
        candidate = CompatibleMaxwell1DState(
            (next_ex, next_ey, next_ez), (bx, next_by, next_bz), charge
        )
        old_energy, new_energy = self.energy(state), self.energy(candidate)
        source_power = self.spacing * sum(
            jnp.sum(e * value)
            for e, value in zip(candidate.electric, current, strict=True)
        )
        gauss = jnp.max(
            jnp.abs(self.permittivity * _backward(next_ex, 0, self.spacing) - charge)
        )
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in candidate.electric + candidate.magnetic
                )
            )
        ) & jnp.all(jnp.isfinite(charge))
        stable = jnp.isfinite(dt) & (dt > 0.0) & (dt <= self.stable_dt)
        successful = finite & stable
        accepted = CompatibleMaxwell1DState(
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.electric, state.electric, strict=True)
            ),
            tuple(
                jnp.where(successful, new, old)
                for new, old in zip(candidate.magnetic, state.magnetic, strict=True)
            ),
            jnp.where(successful, candidate.charge, state.charge),
        )
        return accepted, ReducedMaxwellDiagnostics(
            new_energy,
            gauss,
            jnp.asarray(0.0, dtype=charge.dtype),
            source_power,
            new_energy - old_energy + dt * source_power,
            dt / self.stable_dt,
            finite,
            stable,
            successful,
            self.plan_id,
        )


__all__ = [
    "CompatibleMaxwell1DPlan",
    "CompatibleMaxwell1DState",
    "CompatibleMaxwell2DPlan",
    "CompatibleMaxwell2DState",
    "ReducedMaxwellDiagnostics",
]
