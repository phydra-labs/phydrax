#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact one-interval Maxwell propagators in (Galilean) spectral coordinates.

In every Fourier mode the vacuum Maxwell system ``∂ₜU = A U + F`` for
``U = (E, B)`` with ``A U = (c² ∇⁻×B, −∇⁺×E)`` is integrated analytically. The
discrete derivatives ``D±`` are the solver's (staggered or collocated,
finite- or infinite-order) symbols, ``[k]² = Σ|D_a|²``, and Galilean
coordinates translating at ``v_gal`` add the scalar advection ``iκ`` with
``κ = [k]·v_gal`` (Lehe et al. 2016; Kirchen et al. 2016). Because
``A² = −c²[k]²`` on transverse fields and ``A`` annihilates longitudinal ones,
any analytic function of the shifted operator is

    f(A + iκ) = f(iκ) P∥ + ½[f(z₊) + f(z₋)] P⊥ + ½[f(z₊) − f(z₋)] A/(ic[k]),

with ``z± = i(κ ± c[k])``. With ``φ₀ = exp`` and ``φ_{j+1}(z) = (φ_j(z) −
1/j!)/z`` the step with ``J(τ) = J₀ + J₁ τ`` and a constant magnetic current
``M`` (``∂ₜB = −∇×E − M``, the equivalent source of sheet antennas) is

    U(h) = φ₀(Ah) U₀ − h φ₁(Ah) (J₀/ε, M) − h² φ₂(Ah) (J₁/ε, 0),

which for ``κ = 0`` and constant ``J`` is Haber's PSATD update (Haber et al.
1973; Vay, Haber, Godfrey 2013), for ``κ ≠ 0`` the Galilean PSATD of Lehe et
al. (2016)/Kirchen et al. (2016), and for linear or piecewise (multi-J)
currents the time-polynomial PSATD of Shapoval et al. (Phys. Rev. E 110,
025206, 2024). Window integrals ``∫ₐᵇ U dτ`` follow from
``d/dτ[τ^{j+1} φ_{j+1}(Aτ)] = τ^j φ_j(Aτ)`` and give the averaged Galilean
fields (Shapoval et al., Phys. Rev. E 104, 055311, 2021) and the split-field
PML increments.
"""

from __future__ import annotations

from math import factorial

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ...._strict import StrictModule
from ...._trainable import NonTrainableState


_SERIES_RADIUS = 0.5
_SERIES_TERMS = 18


def phi_functions(z: Array, count: int, /) -> tuple[Array, ...]:
    """``(φ₀(z), …, φ_{count−1}(z))`` stable at ``z → 0``.

    Small arguments use the Taylor series ``φ_j(z) = Σ zⁿ/(n + j)!`` (truncation
    below ``0.5¹⁸/18!``); larger ones the recurrence, which is well conditioned
    there.
    """
    small = jnp.abs(z) < _SERIES_RADIUS
    near = jnp.where(small, z, 0.0)
    far = jnp.where(small, 1.0, z)
    values = []
    recurrence = jnp.exp(far)
    for order in range(count):
        series = sum(
            near**term / factorial(term + order) for term in range(_SERIES_TERMS)
        )
        values.append(jnp.where(small, series, recurrence))
        recurrence = (recurrence - 1.0 / factorial(order)) / far
    return tuple(values)


class SpectralOperators(StrictModule, NonTrainableState):
    """Prepared per-mode derivative symbols of one spectral grid.

    Arrays broadcast against spectral stacks ``[K₀, K₁, K₂, blocks, …]``.
    ``plus``/``minus`` are the point→interval and interval→point symbols per
    axis (identical ``i[k]`` on collocated grids); ``squared`` is ``[k]²``;
    ``advection`` is the Galilean ``κ``; ``resolved`` marks ``[k]² > 0``.
    """

    plus: Array
    minus: Array
    squared: Array
    advection: Array
    resolved: Array
    speed: float = eqx.field(static=True)
    permittivity: float = eqx.field(static=True)

    def curl_plus(self, value: Array, /) -> Array:
        return _curl(self.plus, value)

    def curl_minus(self, value: Array, /) -> Array:
        return _curl(self.minus, value)

    def electric_divergence(self, electric: Array, /) -> Array:
        return jnp.sum(self.minus * electric, axis=-1)

    def magnetic_divergence(self, magnetic: Array, /) -> Array:
        return jnp.sum(self.plus * magnetic, axis=-1)

    def safe_squared(self) -> Array:
        return jnp.where(self.resolved, self.squared, 1.0)

    def longitudinal_electric(self, electric: Array, /) -> Array:
        """``P∥E = −∇⁺(∇⁻·E)/[k]²`` on resolved modes, zero elsewhere."""
        scale = jnp.where(self.resolved, -1.0 / self.safe_squared(), 0.0)
        return self.plus * (scale * self.electric_divergence(electric))[..., None]

    def longitudinal_magnetic(self, magnetic: Array, /) -> Array:
        scale = jnp.where(self.resolved, -1.0 / self.safe_squared(), 0.0)
        return self.minus * (scale * self.magnetic_divergence(magnetic))[..., None]

    def coulomb_field(self, charge: Array, /) -> Array:
        """Longitudinal ``E = −∇⁺ρ/(ε[k]²)`` with ``∇⁻·E = ρ/ε`` on resolved modes."""
        scale = jnp.where(
            self.resolved, -1.0 / (self.permittivity * self.safe_squared()), 0.0
        )
        return self.plus * (scale * charge)[..., None]

    def magnetic_charge_field(self, charge: Array, /) -> Array:
        """Longitudinal ``B = −∇⁻ρ_m/[k]²`` with ``∇⁺·B = ρ_m`` on resolved modes."""
        scale = jnp.where(self.resolved, -1.0 / self.safe_squared(), 0.0)
        return self.minus * (scale * charge)[..., None]

    def charge_current(self, charge: Array, /) -> Array:
        """Longitudinal current ``∇⁺ρ/[k]²`` whose divergence is ``−ρ``."""
        scale = jnp.where(self.resolved, 1.0 / self.safe_squared(), 0.0)
        return self.plus * (scale * charge)[..., None]

    def transverse_current(self, current: Array, /) -> Array:
        return current - self.longitudinal_electric(current)

    def arguments(self, duration: Array | float, /) -> tuple[Array, Array, Array]:
        """``(z₀, z₊, z₋) = i(κ, κ + c[k], κ − c[k])·duration``."""
        frequency = self.speed * jnp.sqrt(self.squared)
        return (
            1j * self.advection * duration,
            1j * (self.advection + frequency) * duration,
            1j * (self.advection - frequency) * duration,
        )


def _curl(symbol: Array, value: Array, /) -> Array:
    return jnp.stack(
        (
            symbol[..., 1] * value[..., 2] - symbol[..., 2] * value[..., 1],
            symbol[..., 2] * value[..., 0] - symbol[..., 0] * value[..., 2],
            symbol[..., 0] * value[..., 1] - symbol[..., 1] * value[..., 0],
        ),
        axis=-1,
    )


def apply_function(
    operators: SpectralOperators,
    values: tuple[Array, Array, Array],
    electric: Array,
    magnetic: Array,
    /,
) -> tuple[Array, Array]:
    """``f(A + iκ)(E, B)`` from scalar values ``(f(z₀), f(z₊), f(z₋))``."""
    center, upper, lower = values
    even = (0.5 * (upper + lower))[..., None]
    odd = (0.5 * (upper - lower))[..., None]
    frequency = operators.speed * jnp.sqrt(operators.safe_squared())
    rotation = jnp.where(operators.resolved, 1.0 / (1j * frequency), 0.0)[..., None]
    parallel_e = operators.longitudinal_electric(electric)
    parallel_b = operators.longitudinal_magnetic(magnetic)
    rotated_e = rotation * operators.speed**2 * operators.curl_minus(magnetic)
    rotated_b = -rotation * operators.curl_plus(electric)
    return (
        center[..., None] * parallel_e + even * (electric - parallel_e) + odd * rotated_e,
        center[..., None] * parallel_b + even * (magnetic - parallel_b) + odd * rotated_b,
    )


def apply_source(
    operators: SpectralOperators,
    values: tuple[Array, Array, Array],
    current: Array,
    magnetic_current: Array | None = None,
    /,
) -> tuple[Array, Array]:
    """``f(A + iκ)(−J/ε, −M)``; ``M`` is the magnetic current of ``∂ₜB = −∇×E − M``."""
    return apply_function(
        operators,
        values,
        -current / operators.permittivity,
        jnp.zeros_like(current) if magnetic_current is None else -magnetic_current,
    )


def exact_interval(
    operators: SpectralOperators,
    electric: Array,
    magnetic: Array,
    current: Array,
    current_slope: Array | None,
    duration: Array,
    /,
    *,
    magnetic_current: Array | None = None,
) -> tuple[Array, Array]:
    """Exact ``U(h)`` for ``J(τ) = J₀ + J₁τ`` and a constant magnetic current ``M``."""
    count = 2 if current_slope is None else 3
    columns = tuple(phi_functions(z, count) for z in operators.arguments(duration))
    at = tuple(tuple(column[order] for column in columns) for order in range(count))
    e, b = apply_function(operators, (at[0][0], at[0][1], at[0][2]), electric, magnetic)
    se, sb = apply_source(
        operators, (at[1][0], at[1][1], at[1][2]), current, magnetic_current
    )
    e = e + duration * se
    b = b + duration * sb
    if current_slope is not None:
        se, sb = apply_source(operators, (at[2][0], at[2][1], at[2][2]), current_slope)
        e = e + duration**2 * se
        b = b + duration**2 * sb
    return e, b


def _integral_values(
    operators: SpectralOperators, start: Array, stop: Array, order: int, /
) -> tuple[Array, Array, Array]:
    """``(b^{j+1}φ_{j+1}(zb) − a^{j+1}φ_{j+1}(za))`` at ``z₀, z₊, z₋``."""
    result = []
    for upper, lower in zip(
        operators.arguments(stop), operators.arguments(start), strict=True
    ):
        high = phi_functions(upper, order + 2)[order + 1]
        low = phi_functions(lower, order + 2)[order + 1]
        result.append(stop ** (order + 1) * high - start ** (order + 1) * low)
    return result[0], result[1], result[2]


def window_integral(
    operators: SpectralOperators,
    electric: Array,
    magnetic: Array,
    current: Array,
    current_slope: Array | None,
    start: Array,
    stop: Array,
    /,
    *,
    magnetic_current: Array | None = None,
) -> tuple[Array, Array]:
    """``∫ₐᵇ U(τ) dτ`` of the analytic interval solution (τ from interval start)."""
    e, b = apply_function(
        operators, _integral_values(operators, start, stop, 0), electric, magnetic
    )
    se, sb = apply_source(
        operators, _integral_values(operators, start, stop, 1), current, magnetic_current
    )
    e, b = e + se, b + sb
    if current_slope is not None:
        se, sb = apply_source(
            operators, _integral_values(operators, start, stop, 2), current_slope
        )
        e, b = e + se, b + sb
    return e, b


def gauss_following_increment(
    operators: SpectralOperators,
    start_charge: Array,
    end_charge: Array,
    duration: Array,
    /,
) -> Array:
    """Longitudinal ``E`` increment ``−∇⁺(ρ₁ − θρ₀)/(ε[k]²)``, ``θ = e^{iκh}``.

    Added to a propagation driven by the transverse current only, it makes
    ``E∥(h) + ∇⁺ρ₁/(ε[k]²) = θ(E∥(0) + ∇⁺ρ₀/(ε[k]²))``: the Gauss residual is
    advected unchanged whatever the charge history inside the interval.
    """
    phase = jnp.exp(1j * operators.advection * duration)
    return operators.coulomb_field(end_charge - phase * start_charge)


def galilean_charge_current(
    operators: SpectralOperators,
    start_charge: Array,
    end_charge: Array,
    duration: Array,
    /,
) -> Array:
    """Constant longitudinal current carrying ``ρ₀ → ρ₁`` in Galilean coordinates.

    ``J∥ = ∇⁺(ρ₁ − θρ₀)/([k]² h φ₁(iκh))`` is the discrete Galilean continuity
    of Lehe et al. (2016); for ``κ = 0`` it is ``∇⁺(ρ₁ − ρ₀)/([k]² h)``.
    """
    z = 1j * operators.advection * duration
    phase = jnp.exp(z)
    first = phi_functions(z, 2)[1]
    return operators.charge_current(end_charge - phase * start_charge) / (
        duration * first[..., None]
    )


def gauss_following_window(
    operators: SpectralOperators,
    start_charge: Array,
    end_charge: Array,
    duration: Array,
    start: Array,
    stop: Array,
    /,
) -> Array:
    """``∫ₐᵇ`` of the ρ-driven longitudinal field for a constant-current interval.

    With ``ρ(τ) = ρ₀ + (ρ₁ − ρ₀) s(τ)/s(h)``, ``s(τ) = τφ₁(iκτ)`` (the exact
    charge history of a constant Galilean current), the longitudinal field
    beyond the ``θ(τ)E∥(0)`` part carried by the propagator is
    ``∇⁺[θ(τ)ρ₀ − ρ(τ)]/(ε[k]²)``.
    """
    kappa = operators.advection

    def first(value: Array) -> Array:
        return value * phi_functions(1j * kappa * value, 2)[1]

    def second(value: Array) -> Array:
        return value**2 * phi_functions(1j * kappa * value, 3)[2]

    phase_integral = first(stop) - first(start)
    history_integral = second(stop) - second(start)
    width = stop - start
    charge_integral = start_charge * width + (
        end_charge - start_charge
    ) * history_integral / first(duration)
    return -operators.coulomb_field(start_charge * phase_integral - charge_integral)


__all__ = [
    "SpectralOperators",
    "apply_function",
    "apply_source",
    "exact_interval",
    "galilean_charge_current",
    "gauss_following_increment",
    "gauss_following_window",
    "phi_functions",
    "window_integral",
]
