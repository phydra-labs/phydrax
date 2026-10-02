#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Numerical Cherenkov instability (NCI) monitor and Godfrey–Vay linear reference.

`SpectralNCIMonitorPlan` reduces the electromagnetic energy of a spectral field
state into isotropic ``|k|`` shells (`PeriodicFourierShellPlan`) and reports the
energy above a high-``|k|`` threshold, where the NCI of relativistically
drifting plasma first appears; `SpectralNCIMonitorPlan.fit` fits the
exponential growth of that energy. `godfrey_vay_growth_rate` solves the linear
PSATD–Esirkepov cold-beam dispersion relation of Godfrey, Vay & Haber (J.
Comput. Phys. 258, 689 (2014), Eqs. 14–25) with the solver's current scaling
``ζ`` on every resolved grid mode of the drift/transverse plane and returns the
largest predicted field growth rate — the linear reference the monitored
growth is compared against.
"""

from __future__ import annotations

from collections.abc import Sequence
from math import pi

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import PeriodicFourierShellPlan
from ....linalg import determinant_small_linear, SmallLinearSolvePlan
from ....nonlinear import VectorLocalRootPlan
from ....typing import checked
from ._psatd import PreparedSpectralMaxwell, SpectralMaxwellState


class SpectralNCISample(StrictModule):
    """Electromagnetic energy per ``|k|`` shell and above the high-``|k|`` threshold."""

    shell_energy: Array
    high_energy: Array
    total_energy: Array


class NCIGrowthFit(StrictModule):
    """Least-squares fit ``ln W(t) ≈ a + 2Γt`` of monitored high-``|k|`` energy.

    ``rate`` is the field-amplitude growth rate ``Γ`` (half the energy rate);
    ``residual`` the RMS misfit of ``ln W``.
    """

    rate: Array
    energy_rate: Array
    intercept: Array
    residual: Array
    samples: int = eqx.field(static=True)


class SpectralNCIMonitorPlan(StrictModule, NonTrainableState):
    """High-``|k|`` shell energy of one prepared spectral solver's fields."""

    shells: PeriodicFourierShellPlan
    high: Array
    permittivity: float = eqx.field(static=True)
    permeability: float = eqx.field(static=True)
    high_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        solver: PreparedSpectralMaxwell,
        /,
        *,
        shell_count: int = 8,
        high_fraction: float = 0.5,
    ) -> None:
        count = int(shell_count)
        fraction = float(high_fraction)
        if count < 2:
            raise ValueError("shell_count must be at least two.")
        if not 0.0 < fraction < 1.0:
            raise ValueError("high_fraction must lie in (0, 1).")
        plan = solver.plan
        lengths = tuple(n * h for n, h in zip(plan.counts, plan.spacing, strict=True))
        maximum = float(np.sqrt(sum((pi / h) ** 2 for h in plan.spacing)))
        edges = np.linspace(0.0, maximum, count + 1)
        self.shells = PeriodicFourierShellPlan(
            plan.counts, lengths, edges, source_id="spectral-maxwell-nci"
        )
        self.high = jnp.asarray(edges[:-1] >= fraction * maximum)
        self.permittivity = plan.permittivity
        self.permeability = plan.permeability
        self.high_fraction = fraction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-nci-monitor",
                "solver": solver.solver_id,
                "shell_count": count,
                "high_fraction": fraction,
            }
        )

    def sample(self, field: SpectralMaxwellState, /) -> SpectralNCISample:
        """Shell energies ``½∫(ε|E|² + |B|²/μ)`` by Parseval over ``|k|`` shells."""
        counts = self.shells.weighted_mode_count
        energy = jnp.zeros_like(counts)
        for values, weight in (
            (field.electric, 0.5 * self.permittivity),
            (field.magnetic, 0.5 / self.permeability),
        ):
            for axis in range(3):
                power = self.shells.auto_power(self.shells.transform(values[..., axis]))
                energy = energy + weight * power.shell_values * counts
        return SpectralNCISample(
            energy, jnp.sum(jnp.where(self.high, energy, 0.0)), jnp.sum(energy)
        )

    @staticmethod
    def fit(
        times: ArrayLike, energies: ArrayLike, /, *, start: float, stop: float
    ) -> NCIGrowthFit:
        """Fit the exponential growth of ``energies`` over ``start ≤ t ≤ stop``."""
        t = np.asarray(times, dtype=np.float64)
        w = np.asarray(energies, dtype=np.float64)
        if t.shape != w.shape or t.ndim != 1:
            raise ValueError("times and energies must be matching vectors.")
        window = (t >= float(start)) & (t <= float(stop)) & (w > 0.0)
        if np.count_nonzero(window) < 3:
            raise ValueError("A growth fit needs at least three positive samples.")
        x, y = t[window], np.log(w[window])
        centered = x - x.mean()
        slope = float(np.sum(centered * (y - y.mean())) / np.sum(centered**2))
        intercept = float(y.mean() - slope * x.mean())
        residual = float(np.sqrt(np.mean((y - intercept - slope * x) ** 2)))
        return NCIGrowthFit(
            jnp.asarray(0.5 * slope),
            jnp.asarray(slope),
            jnp.asarray(intercept),
            jnp.asarray(residual),
            int(np.count_nonzero(window)),
        )


class GodfreyVayReference(StrictModule):
    """Linear NCI prediction over the drift/transverse grid-mode plane.

    ``growth_rates[i, j]`` is the largest root ``Im ω`` found for the mode
    ``(wavenumbers_drift[i], wavenumbers_transverse[j])`` (zero where none
    converged unstable); ``maximum_growth_rate`` is its maximum.
    """

    wavenumbers_drift: Array
    wavenumbers_transverse: Array
    growth_rates: Array
    frequencies: Array
    maximum_growth_rate: Array
    converged_fraction: Array


def _zeta(wavenumber: Array, spacing: float, vay: bool, /) -> Array:
    """Current scaling of the solver relative to the k-space Esirkepov current."""
    if vay:
        return jnp.ones_like(wavenumber)
    half = 0.5 * wavenumber * spacing
    return jnp.where(half == 0.0, 1.0, half / jnp.sin(jnp.where(half == 0.0, 1.0, half)))


def _dispersion(
    omega: Array,
    kz: Array,
    kx: Array,
    zeta_z: Array,
    zeta_x: Array,
    parameters: tuple[float, float, float, float, float, float, int, int],
    /,
) -> Array:
    """Godfrey–Vay–Haber 3×3 determinant (Eqs. 14–25) at complex ``ω`` (c = 1)."""
    velocity, density, gamma, dz, dx, dt, order, aliases = parameters
    k = jnp.sqrt(kz**2 + kx**2)
    bracket_k = jnp.sin(0.5 * k * dt) / (0.5 * k * dt)
    mesh_z, mesh_x = kz * bracket_k, kx * bracket_k
    mesh_omega = jnp.sin(0.5 * omega * dt) / (0.5 * dt)
    alias = jnp.arange(-aliases, aliases + 1, dtype=jnp.float64)
    kzp = kz + 2.0 * pi * alias[:, None] / dz
    kxp = kx + 2.0 * pi * alias[None, :] / dx

    def sinc(value: Array) -> Array:
        safe = jnp.where(value == 0.0, 1.0, value)
        return jnp.where(value == 0.0, 1.0, jnp.sin(safe) / safe)

    shape = (sinc(0.5 * kzp * dz) * sinc(0.5 * kxp * dx)) ** (order + 1)
    electric = shape * shape
    magnetic = electric * jnp.cos(0.5 * omega * dt) / jnp.cos(0.5 * k * dt)
    phase = 0.5 * (omega - kzp * velocity) * dt
    csc = 1.0 / jnp.sin(phase)
    cot = jnp.cos(phase) * csc
    sine = jnp.sin(k * dt)
    drift_sin = jnp.sin(0.5 * kzp * velocity * dt)
    drift_cos = jnp.cos(0.5 * kzp * velocity * dt)
    along = k * kz**2 * dt + zeta_z * kx**2 * sine
    eta_z = cot * along * drift_sin + (k * dt - zeta_x * sine) * kz**2 * drift_cos
    eta_x = (
        cot * (k * dt - zeta_z * sine) * kx**2 * drift_sin
        + (k * kx**2 * dt + zeta_x * kz**2 * sine) * drift_cos
    )
    longitudinal = density / gamma**2
    xi_zz = -longitudinal * jnp.sum(
        electric * csc**2 * along * dt * mesh_omega * kzp / (4.0 * k**3 * kz)
    )
    xi_zx = -density * jnp.sum(electric * csc * eta_z * kxp / (2.0 * k**3 * kz))
    xi_zy = density * velocity * jnp.sum(magnetic * csc * eta_z * kxp / (2.0 * k**3 * kz))
    xi_xz = -longitudinal * jnp.sum(
        electric
        * csc**2
        * (k * dt - zeta_z * sine)
        * dt
        * mesh_omega
        * kx
        * kzp
        / (4.0 * k**3)
    )
    xi_xx = -density * jnp.sum(electric * csc * eta_x * kxp / (2.0 * k**3 * kx))
    xi_xy = density * velocity * jnp.sum(magnetic * csc * eta_x * kxp / (2.0 * k**3 * kx))
    matrix = jnp.stack(
        (
            jnp.stack((xi_zz + mesh_omega, xi_zx, xi_zy + mesh_x)),
            jnp.stack((xi_xz, xi_xx + mesh_omega, xi_xy - mesh_z)),
            jnp.stack((mesh_x, -mesh_z, mesh_omega)),
        )
    )
    return determinant_small_linear(SmallLinearSolvePlan(3), matrix)


def godfrey_vay_growth_rate(
    solver: PreparedSpectralMaxwell,
    /,
    *,
    drift_axis: int,
    transverse_axis: int,
    drift_speed: float,
    plasma_frequency: float,
    step_size: float,
    aliases: int = 2,
    guesses: Sequence[float] = (0.05, 0.3),
) -> GodfreyVayReference:
    """Largest linear NCI growth rate of a cold beam drifting along ``drift_axis``.

    The beam of rest-frame-unaware lab plasma frequency ``plasma_frequency``
    (``ω_p² = n q²/(ε m)``) drifts at ``drift_speed``; the reference solves
    ``det M(ω) = 0`` for every grid mode with nonzero drift and transverse
    wavenumbers (third-axis wavenumber zero) by Newton iteration seeded at the
    beam aliases ``ω = k′_z v`` with growth guesses ``guesses/Δt``, and keeps
    converged roots with ``Im ω > 0``. It covers the configuration the
    Godfrey–Vay analysis describes: standard, constant-J, infinite-order,
    collocated PSATD with the Esirkepov current (``ζ = 1`` for Vay deposition,
    ``ζ = (kΔ/2)/sin(kΔ/2)`` otherwise) and equal deposit/gather shapes.
    """
    plan = solver.plan
    if (
        plan.variant != "standard"
        or plan.time_dependency != "constant-j"
        or plan.stencil != "infinite-order"
        or plan.grid != "collocated"
        or plan.decomposition != "global-fft"
    ):
        raise ValueError(
            "The Godfrey–Vay reference covers the standard constant-j infinite-order "
            "collocated global PSATD."
        )
    if not solver.transfers:
        raise ValueError("The Godfrey–Vay reference needs the solver's particle shape.")
    orders = {transfer.plan.shape_order for transfer in solver.transfers}
    if len(orders) != 1:
        raise ValueError("Every species must share one shape order.")
    axes = (int(drift_axis), int(transverse_axis))
    if len(set(axes)) != 2 or any(axis not in (0, 1, 2) for axis in axes):
        raise ValueError("drift_axis and transverse_axis must be distinct axes.")
    speed = plan.speed_of_light
    velocity = float(drift_speed) / speed
    if not 0.0 < velocity < 1.0:
        raise ValueError("drift_speed must be positive and subluminal.")
    gamma = 1.0 / np.sqrt(1.0 - velocity**2)
    density = float(plasma_frequency) ** 2 / gamma
    dt = float(step_size)
    dz = plan.spacing[axes[0]] / speed
    dx = plan.spacing[axes[1]] / speed
    parameters = (velocity, density, float(gamma), dz, dx, dt, orders.pop(), int(aliases))
    counts = (plan.counts[axes[0]], plan.counts[axes[1]])
    kz_values = 2.0 * pi * np.fft.fftfreq(counts[0], d=dz)
    kx_values = 2.0 * pi * np.fft.fftfreq(counts[1], d=dx)
    kz_values = kz_values[(kz_values != 0.0) & (np.arange(counts[0]) != counts[0] // 2)]
    kx_values = kx_values[(kx_values > 0.0) & (np.arange(counts[1]) != counts[1] // 2)]
    vay = plan.charge_conservation == "vay-deposition"
    root = VectorLocalRootPlan(
        2, maximum_steps=60, tolerance=1e-12, plan_id="godfrey-vay"
    )
    alias_range = np.arange(-int(aliases), int(aliases) + 1)
    kz_grid, kx_grid, alias_grid, guess_grid = np.meshgrid(
        kz_values,
        kx_values,
        alias_range,
        np.asarray(guesses, dtype=np.float64),
        indexing="ij",
    )
    beam = (kz_grid + 2.0 * pi * alias_grid / dz) * velocity
    beam = np.angle(np.exp(1j * beam * dt)) / dt
    initial = np.stack((beam, guess_grid / dt), axis=-1).reshape((-1, 2))
    kz_flat = jnp.asarray(kz_grid.reshape(-1))
    kx_flat = jnp.asarray(kx_grid.reshape(-1))
    beam_flat = jnp.asarray(beam.reshape(-1))
    scale = (pi / dt) ** 3

    def solve_one(values: tuple[Array, Array, Array, Array]) -> tuple[Array, Array]:
        kz, kx, pole, guess = values
        zeta_z = _zeta(kz, dz, vay)
        zeta_x = _zeta(kx, dx, vay)

        def residual(x: Array) -> Array:
            omega = x[0] + 1j * x[1]
            value = (
                _dispersion(omega, kz, kx, zeta_z, zeta_x, parameters)
                * jnp.sin(0.5 * (omega - pole) * dt) ** 2
                / scale
            )
            return jnp.stack((jnp.real(value), jnp.imag(value)))

        found, diagnostics = root.solve_with_diagnostics(residual, guess)
        return found, diagnostics.converged

    roots, converged = jax.lax.map(
        solve_one, (kz_flat, kx_flat, beam_flat, jnp.asarray(initial))
    )
    roots_host = np.asarray(roots).reshape((*kz_grid.shape, 2))
    accepted = (
        np.asarray(converged).reshape(kz_grid.shape)
        & (roots_host[..., 1] > 0.0)
        & (np.abs(roots_host[..., 0]) <= pi / dt)
        & np.all(np.isfinite(roots_host), axis=-1)
    )
    growth = np.where(accepted, roots_host[..., 1], 0.0)
    best = growth.reshape((*growth.shape[:2], -1))
    index = np.argmax(best, axis=-1)
    rates = np.take_along_axis(best, index[..., None], axis=-1)[..., 0]
    frequency = np.take_along_axis(
        roots_host[..., 0].reshape((*growth.shape[:2], -1)), index[..., None], axis=-1
    )[..., 0]
    return GodfreyVayReference(
        wavenumbers_drift=jnp.asarray(kz_values / speed),
        wavenumbers_transverse=jnp.asarray(kx_values / speed),
        growth_rates=jnp.asarray(rates),
        frequencies=jnp.asarray(frequency),
        maximum_growth_rate=jnp.asarray(np.max(rates, initial=0.0)),
        converged_fraction=jnp.asarray(np.mean(np.asarray(converged))),
    )


__all__ = [
    "GodfreyVayReference",
    "NCIGrowthFit",
    "SpectralNCIMonitorPlan",
    "SpectralNCISample",
    "godfrey_vay_growth_rate",
]
