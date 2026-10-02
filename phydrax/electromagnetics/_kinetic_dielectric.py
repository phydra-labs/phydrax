#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Hot magnetized kinetic plasma dielectric, relativistic weak growth, dispersion roots.

Conventions (shared with `ColdPlasmaDielectric`, Stix 1992):

- phasors carry ``exp(−iωt)``; the Stix frame has ``B₀ ∥ ẑ`` and
  ``k = (k⊥, 0, k∥)``; the signed cyclotron frequency is ``Ω_s = q_s |B₀| / m_s``;
- ``ε = I + Σ_s χ_s`` with the linear Vlasov susceptibility

  ``χ = (q²/ε₀ω) Σ_n ∫ d³p (U/v⊥) V_n* V_nᵀ / (ω − k∥v∥ − nΩ/γ) − ẑẑ (q²/ε₀ω²) ∫ d³p v∥ G``,

  ``V_n = (v⊥ nJ_n(a)/a, i v⊥ J_n'(a), v∥ J_n(a))``, ``a = k⊥v⊥γ/Ω``,
  ``U = (1 − k∥v∥/ω) ∂f/∂p⊥ + (k∥v⊥/ω) ∂f/∂p∥`` and
  ``G = (v∥ ∂f/∂p⊥ − v⊥ ∂f/∂p∥)/v⊥``, derived directly from the unperturbed
  orbit integral (``Im ω > 0``) and continued analytically to ``Im ω ≤ 0``.

The nonrelativistic model evaluates this for drifting bi-Maxwellians with the
plasma dispersion function ``Z(ζ) = i√π w(ζ)``. ``w`` is entire, so ``Z`` is
Landau's continuation for every ``Im ζ``; for ``k∥ < 0`` the causal parallel
integral is ``sgn(k∥) Z(sgn(k∥) ζ)``, which this module uses for every
parallel moment. Large ``|ζ|`` moments use the asymptotic series with the
Landau residue so that ``1 + ζZ`` keeps its relative accuracy.

The weakly relativistic model expands ``γ ≈ 1 + u²/2`` in the resonance and
keeps the lowest finite-Larmor-radius order of each harmonic (Shkarofsky 1966;
Bornatici et al. 1983). Its parallel-perpendicular momentum integrals are the
Shkarofsky functions ``F_q(z, a) = −i ∫₀^∞ (1 − it)^{−q} exp(izt − at²/(1 − it)) dt``
with ``z = μ(1 − nΩ/ω)``, ``a = μN∥²/2`` and ``μ = mc²/T``, evaluated from
closed forms in ``Z`` on the sheet whose branch cut runs from ``z = a`` into
the lower half-plane (the continuation of the ``Im ω > 0`` response).
"""

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from math import erfc, exp, factorial, isfinite, lgamma, log, pi, sqrt
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._numerics._quadrature_rules import gauss_legendre_data
from .._physical import ElectromagneticScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..ein import contract
from ..linalg import determinant_small_linear, SmallLinearSolvePlan
from ..nonlinear import VectorLocalRootPlan
from ..special import ive, jv, wofz
from ..typing import (
    as_host_array,
    Bool,
    checked,
    Complex128,
    ConvertibleToArray,
    Dim,
    Float64,
    HostFloat64,
    Identifier,
    Int32,
    parse,
    Scope,
    Size,
    VariadicDim,
)
from ._cold_plasma import ColdPlasmaDielectric, ColdPlasmaWaveStatus, PlasmaWaveMode
from ._magnetobremsstrahlung import AbstractGyrotropicDistribution


KineticSusceptibilityModel: TypeAlias = Literal["nonrelativistic", "weakly-relativistic"]
KineticDispersionModel: TypeAlias = Literal["electromagnetic", "electrostatic"]

_SQRT_PI = sqrt(pi)
_MOMENT_ASYMPTOTIC_RADIUS = 8.0
"""``|ζ|`` beyond which Landau moments use the asymptotic series (error ≲ e^{−64})."""
_MOMENT_SERIES_TERMS = 48
_SHKAROFSKY_ASYMPTOTIC_RADIUS = 6.0
"""Pole distance from the Gaussian bulk beyond which the Shkarofsky radial integral
is summed asymptotically; below it the closed form loses at most ~6^{2m+1} ulps."""
_SHKAROFSKY_SERIES_TERMS = 72
_SHKAROFSKY_RECURRENCE_MINIMUM = 1.0
"""``|a|`` above which the upward ``q`` recurrence is stable (it divides by ``a``)."""
_PERPENDICULAR_LIMIT = 1.0e-100
"""``|k∥ w∥ / (ω − k∥V − nΩ)|`` below which the ``k∥ = 0`` limit is exact in float64."""


class _SpeciesDim(Dim, minimum=1):
    """Charged species of the kinetic plasma."""


class _BatchDims(VariadicDim):
    """Broadcast batch of evaluation points."""


class _HarmonicDim(Dim, minimum=1):
    """Cyclotron harmonics ``−N … N``."""


class _PathDim(Dim, minimum=1):
    """Points along a dispersion continuation path."""


class KineticSusceptibilityStatus(IntFlag):
    """Evidence flags of `KineticSusceptibilityResult.status`."""

    NONE = 0
    HARMONIC_TRUNCATION = 1
    LARMOR_RADIUS_LIMIT = 2
    NONFINITE = 4


class KineticDispersionStatus(IntFlag):
    """Per-point evidence flags of `KineticDispersionResult.status`."""

    NONE = 0
    NOT_CONVERGED = 1
    ILL_CONDITIONED = 2
    BRANCH_JUMP = 4
    HARMONIC_TRUNCATION = 8
    LARMOR_RADIUS_LIMIT = 16
    NONFINITE = 32


class WeakGrowthStatus(IntFlag):
    """Evidence flags of `RelativisticWeakGrowthResult.status`."""

    NONE = 0
    MODE_NOT_PROPAGATING = 1
    RESONANCE_NOT_ELLIPTIC = 2
    NO_RESONANCE = 4
    QUADRATURE_UNRESOLVED = 8
    WEAK_GROWTH_VIOLATED = 16
    MODE_LABEL_AMBIGUOUS = 32
    NONFINITE = 64


def _complex(value: Array, /) -> Array:
    return value.astype(jnp.complex128)


def _float64_argument(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.dtype != jnp.float64:
        raise TypeError(f"{name} must be a float64 array.")
    return array


def _frequency_argument(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if array.dtype == jnp.float64:
        return _complex(array)
    if array.dtype != jnp.complex128:
        raise TypeError(f"{name} must be a float64 or complex128 array.")
    return array


def _gaussian_moment(order: int, /) -> float:
    """``∫ tᵏ e^{−t²} dt / √π``."""
    if order % 2:
        return 0.0
    return float(np.prod(np.arange(1, order, 2, dtype=np.float64))) / 2.0 ** (order // 2)


# ---------------------------------------------------------------------------
# Landau moments  M_k(ξ) = (1/√π) ∫ tᵏ e^{−t²} / (t − ξ) dt,  k = 0 … 3
# ---------------------------------------------------------------------------

_MOMENT_SERIES = np.asarray(
    [
        [_gaussian_moment(order + term) for term in range(_MOMENT_SERIES_TERMS)]
        for order in range(4)
    ],
    dtype=np.float64,
)


def _inverse_powers(argument: Array, count: int, /) -> Array:
    """``argument^{−1} … argument^{−count}`` along a new trailing axis."""
    inverse = 1.0 / argument
    return jnp.cumprod(
        jnp.broadcast_to(inverse[..., None], inverse.shape + (count,)), axis=-1
    )


def _landau_residue(argument: Array, /) -> Array:
    """``σ i√π exp(−w²)``: the Landau residue absent above the real axis, half on
    it and full below it. The exponential is evaluated only where ``σ ≠ 0`` so
    that the vanishing weight never meets an overflowed ``exp(−w²)``.
    """
    stokes = jnp.where(argument.imag == 0.0, 1.0, 2.0).astype(jnp.complex128)
    present = argument.imag <= 0.0
    exponent = jnp.where(present, -(argument * argument), 0.0j)
    return jnp.where(present, stokes * 1j * _SQRT_PI * jnp.exp(exponent), 0.0j)


def _landau_moments(xi: Array, /) -> Array:
    """``M_0 … M_3`` (trailing axis) continued from ``Im ξ > 0`` (Landau contour)."""
    far = jnp.abs(xi) >= _MOMENT_ASYMPTOTIC_RADIUS
    near_xi = jnp.where(far, 0.0j, xi)
    z = 1j * _SQRT_PI * wofz(near_xi)
    first = 1.0 + near_xi * z
    second = near_xi * first
    third = 0.5 + near_xi * second
    near = jnp.stack((z, first, second, third), axis=-1)
    far_xi = jnp.where(far, xi, _MOMENT_ASYMPTOTIC_RADIUS + 0.0j)
    algebraic = -(
        _inverse_powers(far_xi, _MOMENT_SERIES_TERMS)
        @ jnp.asarray(_MOMENT_SERIES.T).astype(jnp.complex128)
    )
    residue = _landau_residue(far_xi)
    orders = jnp.stack(
        (jnp.ones_like(far_xi), far_xi, far_xi * far_xi, far_xi * far_xi * far_xi),
        axis=-1,
    )
    asymptotic = algebraic + residue[..., None] * orders
    return jnp.where(far[..., None], asymptotic, near)


# ---------------------------------------------------------------------------
# Shkarofsky functions F_{j+3/2}(z, a), j = 0 … count − 1
# ---------------------------------------------------------------------------


def _plasma_z(argument: Array, /) -> Array:
    return 1j * _SQRT_PI * wofz(argument)


def _resonance_root(offset: Array, /) -> Array:
    """``φ = √(z − a)`` with the cut along ``arg(z − a) = −π/2``.

    Real ``z − a`` gives ``φ > 0`` above tangency and ``φ = i|φ|`` below it (the
    ``+i0`` prescription); crossing into ``Im(z − a) < 0`` continues that value.
    """
    rotated = jnp.exp(0.25j * pi) * jnp.sqrt(-1j * offset)
    # Exact values on the real axis, where the rotation would leave rounding.
    real = offset.real
    magnitude = jnp.sqrt(jnp.abs(real)).astype(jnp.complex128)
    on_axis = jnp.where(real >= 0.0, magnitude, 1j * magnitude)
    return jnp.where(offset.imag == 0.0, on_axis, rotated)


def _shkarofsky_recurrence(z: Array, a: Array, count: int, /) -> Array:
    """Closed ``F_{1/2}``, ``F_{3/2}`` and the upward recurrence (stable for ``|a| ≥ 1``).

    ``a F_{q+1} = 1 + (1 − q) F_q − (z − a) F_{q−1}`` follows from integrating the
    Laplace representation by parts.
    """
    safe = jnp.where(jnp.abs(a) >= _SHKAROFSKY_RECURRENCE_MINIMUM, a, 1.0 + 0.0j)
    psi = jnp.sqrt(safe)
    phi = _resonance_root(z - safe)
    plus = _plasma_z(1j * phi + psi)
    minus = _plasma_z(1j * phi - psi)
    previous_scaled = phi * (plus + minus) / 2j  # (z − a) F_{1/2}
    current = (minus - plus) / (2.0 * psi)  # F_{3/2}
    values = [current]
    order = 1.5
    for _ in range(count - 1):
        following = (1.0 + (1.0 - order) * current - previous_scaled) / safe
        previous_scaled = (z - safe) * current
        current = following
        order += 1.0
        values.append(current)
    return jnp.stack(values, axis=-1)


def _shkarofsky_asymptotic_table(count: int, /) -> tuple[np.ndarray, np.ndarray]:
    """``gaussian[j, i] = E[t^{i+j}]`` and ``binomial[m − 1, i] = C(2m, i)``."""
    top = 2 * count
    gaussian = np.asarray(
        [
            [_gaussian_moment(i + j) for i in range(top + 1)]
            for j in range(_SHKAROFSKY_SERIES_TERMS)
        ],
        dtype=np.float64,
    )
    binomial = np.zeros((count, top + 1), dtype=np.float64)
    for m in range(1, count + 1):
        for i in range(2 * m + 1):
            binomial[m - 1, i] = float(factorial(2 * m)) / (
                factorial(i) * factorial(2 * m - i)
            )
    return gaussian, binomial


def _integer_powers(value: Array, top: int, /) -> list[Array]:
    """``value⁰ … value^top`` by repeated multiplication (exact ``0⁰ = 1``)."""
    powers = [jnp.ones_like(value)]
    for _ in range(top):
        powers.append(powers[-1] * value)
    return powers


def _shifted_gaussian_moments(beta: Array, count: int, /) -> list[Array]:
    """``M_j(β) = ∫ r^{2j} e^{−(r+β)²} dr / √π`` for ``j = 0 … count − 1``."""
    powers = _integer_powers(-beta, 2 * count)
    moments = []
    for j in range(count):
        total = jnp.zeros_like(beta)
        for i in range(0, 2 * j + 1, 2):
            coefficient = float(factorial(2 * j)) / (factorial(i) * factorial(2 * j - i))
            total = total + coefficient * _gaussian_moment(i) * powers[2 * j - i]
        moments.append(total)
    return moments


def _radial_integrals(phi: Array, beta: Array, count: int, /) -> Array:
    """``I_m = (1/√π) ∫ r^{2m} e^{−(r+β)²} / (r² + φ²) dr`` for ``m = 1 … count``.

    The partial fractions of ``1/(r² + φ²)`` place the pole ``iφ`` above and
    ``−iφ`` below the real line (``Re φ ≥ 0`` on the physical sheet), giving
    ``φ I_0 = [Z(iφ + β) + Z(iφ − β)]/(2i)``; ``I_m = M_{m−1} − φ² I_{m−1}``
    raises the power. When both poles are far from the Gaussian bulk the
    polynomial part and the pole part cancel, so the asymptotic expansion of
    each partial fraction is summed instead, with the Landau residue restored on
    the continued sheet.
    """
    plus_argument = 1j * phi + beta
    minus_argument = 1j * phi - beta
    far = (
        (jnp.abs(plus_argument) >= _SHKAROFSKY_ASYMPTOTIC_RADIUS)
        & (jnp.abs(minus_argument) >= _SHKAROFSKY_ASYMPTOTIC_RADIUS)
        & (jnp.abs(phi) >= 1.0)
    )
    near_phi = jnp.where(far, 0.0j, phi)
    near_beta = jnp.where(far, 0.0j, beta)
    moments = _shifted_gaussian_moments(near_beta, count)
    scaled = (
        _plasma_z(1j * near_phi + near_beta) + _plasma_z(1j * near_phi - near_beta)
    ) / 2j
    closed = []
    current = moments[0] - near_phi * scaled
    closed.append(current)
    for m in range(2, count + 1):
        current = moments[m - 1] - near_phi * near_phi * current
        closed.append(current)
    closed_values = jnp.stack(closed, axis=-1)

    far_phi = jnp.where(far, phi, 2.0 * _SHKAROFSKY_ASYMPTOTIC_RADIUS + 0.0j)
    far_beta = jnp.where(far, beta, 0.0j)
    gaussian, binomial = _shkarofsky_asymptotic_table(count)
    gaussian_table = jnp.asarray(gaussian).astype(jnp.complex128)
    pole_powers = _integer_powers(-(far_phi * far_phi), count)[1:]
    total = jnp.zeros(far_phi.shape + (count,), dtype=jnp.complex128)
    for sign in (1.0, -1.0):
        shift = sign * far_beta
        argument = 1j * far_phi + shift
        # tails[..., i] = Σ_j E[t^{i+j}] w^{−(j+1)}: expansion of 1/(t − w).
        tails = _inverse_powers(argument, _SHKAROFSKY_SERIES_TERMS) @ gaussian_table
        shift_powers = _integer_powers(-shift, 2 * count)
        residue = _landau_residue(argument)
        partials = []
        for m in range(1, count + 1):
            algebraic = jnp.zeros_like(argument)
            for i in range(2 * m + 1):
                algebraic = algebraic + (
                    float(binomial[m - 1, i]) * shift_powers[2 * m - i] * tails[..., i]
                )
            partials.append(residue * pole_powers[m - 1] - algebraic)
        total = total + jnp.stack(partials, axis=-1)
    asymptotic = total / (2j * far_phi)[..., None]
    return jnp.where(far[..., None], asymptotic, closed_values)


def _shkarofsky_quadrature(z: Array, a: Array, count: int, nodes: int, /) -> Array:
    """``F_{k+3/2}`` from its momentum-space form, used for ``|a| < 1``.

    ``F_{k+3/2} = (1/k!) ∫ d³r r^{2k} sin^{2k}α e^{−r² − 2ψr cos α − a} / (π^{3/2}(r² + φ²))``
    with ``ψ = √a``: the radial integral is closed with `_radial_integrals` and
    the entire, slowly varying ``cos α`` integral uses Gauss–Legendre nodes.
    No step divides by ``a`` or ``ψ``, so ``N∥ → 0`` is regular.
    """
    rule = gauss_legendre_data(nodes)
    safe = jnp.where(jnp.abs(a) < _SHKAROFSKY_RECURRENCE_MINIMUM, a, 0.0j)
    node_shape = (1,) * max(safe.ndim, z.ndim) + (nodes,)
    cosine = jnp.asarray(rule.nodes, dtype=jnp.float64).reshape(node_shape)
    weight = jnp.asarray(rule.weights, dtype=jnp.float64).reshape(node_shape)
    psi = jnp.sqrt(safe)
    phi = _resonance_root(z - safe)
    radial = _radial_integrals(
        phi[..., None], psi[..., None] * _complex(cosine), count
    )  # [..., nodes, count]
    sine_squared = _complex(1.0 - cosine * cosine)
    envelope = jnp.exp(-safe[..., None] * sine_squared) * _complex(weight)
    values = []
    for k in range(count):
        integrand = envelope * sine_squared**k * radial[..., k]
        values.append(jnp.sum(integrand, axis=-1) / float(factorial(k)))
    return jnp.stack(values, axis=-1)


def _shkarofsky(z: Array, a: Array, count: int, nodes: int, /) -> Array:
    """``F_{j+3/2}(z, a)`` for ``j = 0 … count − 1`` (trailing axis)."""
    recurrence = _shkarofsky_recurrence(z, a, count)
    quadrature = _shkarofsky_quadrature(z, a, count, nodes)
    use_recurrence = jnp.abs(a) >= _SHKAROFSKY_RECURRENCE_MINIMUM
    return jnp.where(use_recurrence[..., None], recurrence, quadrature)


# ---------------------------------------------------------------------------
# Tensor assembly
# ---------------------------------------------------------------------------


def _assemble(
    xx: Array, xy: Array, yy: Array, xz: Array, yz: Array, zz: Array, /
) -> Array:
    """Onsager-structured ``χ`` from its six independent entries."""
    return jnp.stack(
        (
            jnp.stack((xx, xy, xz), axis=-1),
            jnp.stack((-xy, yy, yz), axis=-1),
            jnp.stack((xz, -yz, zz), axis=-1),
        ),
        axis=-2,
    )


def _harmonic_split(terms: Array, /) -> tuple[Array, Array]:
    """Sum of harmonics ``|n| ≤ N`` and of the first omitted pair ``|n| = N + 1``.

    ``terms[..., h, 3, 3]`` indexes ``n = −(N+1) … N+1``.
    """
    retained = jnp.sum(terms[..., 1:-1, :, :], axis=-3)
    omitted = terms[..., 0, :, :] + terms[..., -1, :, :]
    return retained, omitted


class KineticSusceptibilityResult(StrictModule):
    """Hot-plasma susceptibility with harmonic-truncation and FLR evidence.

    ``species_susceptibility`` and ``dielectric_tensor = I + Σ_s χ_s`` are in
    the Stix frame. ``truncation_ratio`` is, per species, the Frobenius norm of
    the first omitted harmonic pair ``n = ±(N + 1)`` relative to the retained
    susceptibility — an evaluated estimate of the truncation error, not a
    bound. ``larmor_parameter`` is ``λ = k⊥² T⊥ / (m Ω²)``; the weakly
    relativistic model is lowest order in ``λ`` for every harmonic and flags
    ``LARMOR_RADIUS_LIMIT`` above its tolerance.
    """

    __strict_contract__ = True

    angular_frequency: Complex128[_BatchDims]
    parallel_wavenumber: Float64[_BatchDims]
    perpendicular_wavenumber: Float64[_BatchDims]
    species_susceptibility: Complex128[_BatchDims, _SpeciesDim, Literal[3], Literal[3]]
    dielectric_tensor: Complex128[_BatchDims, Literal[3], Literal[3]]
    truncation_ratio: Float64[_BatchDims, _SpeciesDim]
    larmor_parameter: Float64[_BatchDims, _SpeciesDim]
    status: Int32[_BatchDims]
    harmonic_count: int = eqx.field(static=True)
    model: KineticSusceptibilityModel = eqx.field(static=True)


class KineticPlasmaDielectric(StrictModule, NonTrainableState):
    """Hot magnetized multi-species plasma of drifting bi-Maxwellians.

    ``densities`` are per cubic length unit of ``scale``; ``charge_numbers`` in
    elementary charges; ``mass_ratios`` in electron masses; ``magnetic_field`` is
    ``B₀`` (only ``|B₀|`` enters, ``B₀ ∥ ẑ`` in the Stix frame);
    ``parallel_temperatures`` and ``perpendicular_temperatures`` are ``k_B T`` in
    the scale's energy unit; ``parallel_drifts`` are field-aligned drift
    velocities (the only drift of a gyrotropic equilibrium without ``E₀``).
    ``harmonic_count = N`` retains ``n = −N … N``; the pair ``±(N + 1)`` is
    evaluated as truncation evidence. ``model="weakly-relativistic"`` requires
    isotropic, drift-free species and uses Shkarofsky functions with
    ``shkarofsky_nodes`` Gauss–Legendre nodes in the small-``N∥`` path.
    """

    __strict_contract__ = True

    scale: ElectromagneticScaleContract = eqx.field(static=True)
    densities: Float64[_SpeciesDim]
    charge_numbers: Float64[_SpeciesDim]
    mass_ratios: Float64[_SpeciesDim]
    parallel_temperatures: Float64[_SpeciesDim]
    perpendicular_temperatures: Float64[_SpeciesDim]
    parallel_drifts: Float64[_SpeciesDim]
    magnetic_field: Float64[Literal[3]]
    species_count: Size[_SpeciesDim] = eqx.field(static=True)
    model: KineticSusceptibilityModel = eqx.field(static=True)
    harmonic_count: int = eqx.field(static=True)
    truncation_tolerance: float = eqx.field(static=True)
    larmor_tolerance: float = eqx.field(static=True)
    shkarofsky_nodes: int = eqx.field(static=True)
    dielectric_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        /,
        *,
        densities: ConvertibleToArray,
        charge_numbers: ConvertibleToArray,
        mass_ratios: ConvertibleToArray,
        magnetic_field: ConvertibleToArray,
        parallel_temperatures: ConvertibleToArray,
        perpendicular_temperatures: ConvertibleToArray,
        parallel_drifts: ConvertibleToArray | None = None,
        model: KineticSusceptibilityModel = "nonrelativistic",
        harmonic_count: int = 8,
        truncation_tolerance: float = 1.0e-8,
        larmor_tolerance: float = 0.1,
        shkarofsky_nodes: int = 32,
    ) -> None:
        scope = Scope()
        density = as_host_array(
            densities, HostFloat64[_SpeciesDim], "densities", scope=scope
        )
        charge = as_host_array(
            charge_numbers, HostFloat64[_SpeciesDim], "charge_numbers", scope=scope
        )
        mass = as_host_array(
            mass_ratios, HostFloat64[_SpeciesDim], "mass_ratios", scope=scope
        )
        parallel = as_host_array(
            parallel_temperatures,
            HostFloat64[_SpeciesDim],
            "parallel_temperatures",
            scope=scope,
        )
        perpendicular = as_host_array(
            perpendicular_temperatures,
            HostFloat64[_SpeciesDim],
            "perpendicular_temperatures",
            scope=scope,
        )
        drift = as_host_array(
            np.zeros(density.shape, dtype=np.float64)
            if parallel_drifts is None
            else parallel_drifts,
            HostFloat64[_SpeciesDim],
            "parallel_drifts",
            scope=scope,
        )
        field = as_host_array(
            magnetic_field, HostFloat64[Literal[3]], "magnetic_field", scope=scope
        )
        species_count = parse(
            density.shape[0], Size[_SpeciesDim], "species_count", scope=scope
        )
        model_ = parse(model, KineticSusceptibilityModel, "model")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("densities must be finite and nonnegative.")
        if not np.all(np.isfinite(charge)) or np.any(charge == 0.0):
            raise ValueError("charge_numbers must be finite and nonzero.")
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("mass_ratios must be finite and strictly positive.")
        for name, values in (
            ("parallel_temperatures", parallel),
            ("perpendicular_temperatures", perpendicular),
        ):
            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                raise ValueError(
                    f"{name} must be finite and strictly positive; the cold limit "
                    "is ColdPlasmaDielectric."
                )
        if not np.all(np.isfinite(drift)):
            raise ValueError("parallel_drifts must be finite.")
        if not np.all(np.isfinite(field)) or not np.any(field != 0.0):
            raise ValueError("magnetic_field must be finite and nonzero.")
        count = int(harmonic_count)
        if count < 1:
            raise ValueError("harmonic_count must be a positive integer.")
        truncation = float(truncation_tolerance)
        larmor = float(larmor_tolerance)
        if not isfinite(truncation) or truncation <= 0.0:
            raise ValueError("truncation_tolerance must be finite and positive.")
        if not isfinite(larmor) or larmor <= 0.0:
            raise ValueError("larmor_tolerance must be finite and positive.")
        nodes = int(shkarofsky_nodes)
        if nodes < 4:
            raise ValueError("shkarofsky_nodes must be at least 4.")
        speed = float(scale.speed_of_light)
        match model_:
            case "nonrelativistic":
                pass
            case "weakly-relativistic":
                if np.any(parallel != perpendicular) or np.any(drift != 0.0):
                    raise ValueError(
                        "the weakly relativistic (Shkarofsky) model requires isotropic, "
                        "drift-free Maxwellian species."
                    )
                rest = mass * float(scale.electron_mass) * speed * speed
                if np.any(parallel >= rest):
                    raise ValueError(
                        "the weakly relativistic model requires k_B T < m c² per species."
                    )
            case _:
                raise ValueError(f"unknown kinetic susceptibility model {model_!r}.")
        self.scale = scale
        self.densities = jnp.asarray(density)
        self.charge_numbers = jnp.asarray(charge)
        self.mass_ratios = jnp.asarray(mass)
        self.parallel_temperatures = jnp.asarray(parallel)
        self.perpendicular_temperatures = jnp.asarray(perpendicular)
        self.parallel_drifts = jnp.asarray(drift)
        self.magnetic_field = jnp.asarray(field)
        self.species_count = species_count
        self.model = model_
        self.harmonic_count = count
        self.truncation_tolerance = truncation
        self.larmor_tolerance = larmor
        self.shkarofsky_nodes = nodes
        self.dielectric_id = canonical_fingerprint(
            {
                "kind": "kinetic-plasma-dielectric",
                "scale": scale.scale_id,
                "model": model_,
                "harmonic_count": count,
                "truncation_tolerance": truncation,
                "larmor_tolerance": larmor,
                "shkarofsky_nodes": nodes,
                "content": array_tree_fingerprint(
                    {
                        "densities": density,
                        "charge_numbers": charge,
                        "mass_ratios": mass,
                        "parallel_temperatures": parallel,
                        "perpendicular_temperatures": perpendicular,
                        "parallel_drifts": drift,
                        "magnetic_field": field,
                    }
                ),
            }
        )

    @property
    def speed_of_light(self) -> float:
        return float(self.scale.speed_of_light)

    @property
    def species_masses(self) -> Array:
        return self.mass_ratios * float(self.scale.electron_mass)

    @property
    def plasma_frequency_squared(self) -> Array:
        """``ω_ps² = n_s (Z_s e)² / (ε₀ m_s)`` per species."""
        charge = self.charge_numbers * float(self.scale.elementary_charge)
        return (
            self.densities
            * charge
            * charge
            / (float(self.scale.vacuum_permittivity) * self.species_masses)
        )

    @property
    def cyclotron_frequency(self) -> Array:
        """Signed ``Ω_s = Z_s e |B₀| / m_s`` per species."""
        charge = self.charge_numbers * float(self.scale.elementary_charge)
        return charge * jnp.linalg.norm(self.magnetic_field) / self.species_masses

    @property
    def parallel_thermal_speed(self) -> Array:
        """``w∥ = √(2 T∥ / m)`` per species."""
        return jnp.sqrt(2.0 * self.parallel_temperatures / self.species_masses)

    @property
    def perpendicular_thermal_speed(self) -> Array:
        """``w⊥ = √(2 T⊥ / m)`` per species."""
        return jnp.sqrt(2.0 * self.perpendicular_temperatures / self.species_masses)

    def _harmonics(self) -> Array:
        top = self.harmonic_count + 1
        return jnp.arange(-top, top + 1, dtype=jnp.float64)

    def _nonrelativistic_terms(
        self, omega: Array, k_par: Array, k_perp: Array, /
    ) -> tuple[Array, Array]:
        """Per-harmonic resonant terms ``[..., S, H, 3, 3]`` and the nonresonant ``zz``."""
        leading = (1,) * omega.ndim
        column = leading + (self.species_count, 1)

        def species(value: Array, /) -> Array:
            return value.reshape(column)

        harmonics = self._harmonics().reshape(leading + (1, -1))
        orders = jnp.abs(harmonics)
        frequency = omega[..., None, None]
        parallel = k_par[..., None, None]
        perpendicular = k_perp[..., None, None]
        gyro = species(self.cyclotron_frequency)
        w_par = species(self.parallel_thermal_speed)
        w_perp = species(self.perpendicular_thermal_speed)
        drift = species(self.parallel_drifts)
        anisotropy = species(self.perpendicular_temperatures / self.parallel_temperatures)
        lam = (perpendicular * w_perp / gyro) ** 2 / 2.0  # [..., S, 1]
        scaled_center = ive(orders, lam)
        scaled_lower = ive(jnp.abs(harmonics - 1.0), lam)
        scaled_upper = ive(jnp.abs(harmonics + 1.0), lam)
        n_gamma_over_lambda = 0.5 * (scaled_lower - scaled_upper)
        gamma_prime = 0.5 * (scaled_lower + scaled_upper) - scaled_center
        gyro_ = _complex(gyro)
        drift_ = _complex(drift)
        w_ = _complex(w_par)
        detuning = frequency - _complex(parallel) * drift_ - _complex(harmonics) * gyro_
        kw = parallel * w_par  # [..., S, 1] real
        limit = jnp.abs(kw) <= _PERPENDICULAR_LIMIT * jnp.abs(detuning)
        safe_kw = jnp.where(limit, 1.0, kw)
        sign = jnp.where(safe_kw < 0.0, -1.0, 1.0)
        xi = detuning / _complex(jnp.abs(safe_kw))
        moments = _landau_moments(xi)
        sign_c = _complex(sign)
        m0e = sign_c * moments[..., 0]
        m1e = moments[..., 1]
        m2e = sign_c * moments[..., 2]
        m3e = moments[..., 3]
        h0 = 1.0 - _complex(parallel) * drift_ / frequency
        h1 = _complex(kw) * _complex(1.0 - anisotropy) / frequency
        g0 = h0 * m0e - h1 * m1e
        g1 = drift_ * h0 * m0e + (w_ * h0 - drift_ * h1) * m1e - w_ * h1 * m2e
        g2 = (
            drift_ * drift_ * h0 * m0e
            + (2.0 * drift_ * w_ * h0 - drift_ * drift_ * h1) * m1e
            + (w_ * w_ * h0 - 2.0 * drift_ * w_ * h1) * m2e
            - w_ * w_ * h1 * m3e
        )
        scale_kw = -1.0 / _complex(safe_kw)
        second_moment = drift_ * drift_ + 0.5 * w_ * w_
        mom0 = jnp.where(limit, 1.0 / detuning, scale_kw * g0)
        mom1 = jnp.where(limit, drift_ / detuning, scale_kw * g1)
        mom2 = jnp.where(limit, second_moment / detuning, scale_kw * g2)
        prefactor = _complex(species(self.plasma_frequency_squared)) / frequency
        n_ = _complex(harmonics)
        ratio = _complex(perpendicular / gyro)
        xx = -prefactor * _complex(harmonics * n_gamma_over_lambda) * mom0
        xy = -1j * prefactor * n_ * _complex(gamma_prime) * mom0
        yy = (
            -prefactor
            * _complex(harmonics * n_gamma_over_lambda - 2.0 * lam * gamma_prime)
            * mom0
        )
        xz = -prefactor * _complex(n_gamma_over_lambda) * ratio * mom1
        yz = 1j * prefactor * ratio * _complex(gamma_prime) * mom1
        zz = -2.0 * prefactor / _complex(w_perp * w_perp) * _complex(scaled_center) * mom2
        nonresonant = (
            _complex(species(self.plasma_frequency_squared))
            / (frequency * frequency)
            * _complex((2.0 * drift * drift + w_par * w_par) / (w_perp * w_perp) - 1.0)
        )[..., 0]
        return _assemble(xx, xy, yy, xz, yz, zz), nonresonant

    def _weakly_relativistic_terms(
        self, omega: Array, k_par: Array, k_perp: Array, /
    ) -> tuple[Array, Array]:
        """Lowest-FLR Shkarofsky terms ``[..., S, H, 3, 3]`` (no nonresonant part).

        With ``u = p/(mc)``, ``χ = −(ω_p²/ω²) μ² Σ_n ∫ d³u g(u) V̂_n* V̂_nᵀ / X_n`` for
        the Maxwellian ``g``, ``X_n = z_n + μu²/2 − μN∥u∥`` and the leading term
        ``J_n(a)² ≈ ℓ_n u⊥^{2|n|}``, ``ℓ_n = (b/2)^{2|n|}/(|n|!)²``, ``b = ck⊥/Ω``.
        The moments ``∫ g u⊥^{2k} u∥^m / X`` are ``(2/μ)^k k!`` times
        ``F_{k+3/2}``, ``N∥(F_{k+3/2} − F_{k+5/2})`` and
        ``[F_{k+5/2} + 2a(F_{k+3/2} − 2F_{k+5/2} + F_{k+7/2})]/μ``.
        """
        speed = self.speed_of_light
        top = self.harmonic_count + 1
        leading = (1,) * omega.ndim
        column = leading + (self.species_count, 1)
        row = leading + (self.species_count,)
        harmonics = self._harmonics().reshape(leading + (1, -1))
        frequency = omega[..., None, None]
        mu = (self.species_masses * speed * speed / self.parallel_temperatures).reshape(
            row
        )
        mu_ = _complex(mu.reshape(column))
        gyro = self.cyclotron_frequency.reshape(row)
        z = mu_ * (1.0 - _complex(harmonics) * _complex(gyro.reshape(column)) / frequency)
        n_par = _complex(speed * k_par[..., None, None]) / frequency
        a = 0.5 * mu_ * n_par * n_par
        shk = _shkarofsky(z, a, top + 3, self.shkarofsky_nodes)  # [..., S, H, q]
        b = _complex(speed * k_perp[..., None] / gyro)  # [..., S]
        half_b = 0.5 * b
        n_par_ = n_par[..., 0]
        a_ = a[..., 0]
        inverse_mu = 1.0 / _complex(mu)

        def moments(h: int, k: int, /) -> tuple[Array, Array, Array]:
            f0, f1, f2 = shk[..., h, k], shk[..., h, k + 1], shk[..., h, k + 2]
            weight = (2.0 * inverse_mu) ** k * float(factorial(k))
            return (
                weight * f0,
                weight * n_par_ * (f0 - f1),
                weight * inverse_mu * (f1 + 2.0 * a_ * (f0 - 2.0 * f1 + f2)),
            )

        columns = []
        for h, n in enumerate(range(-top, top + 1)):
            order = abs(n)
            square = float(factorial(order)) ** 2
            m0, m1, m2 = moments(h, order)
            ell = half_b ** (2 * order) / square
            if n == 0:
                zero = jnp.zeros_like(m0)
                yy = 0.25 * b * b * moments(h, 2)[0]
                yz = 0.5j * b * moments(h, 1)[1]
                columns.append((zero, zero, yy, zero, yz, ell * m2))
                continue
            # ℓ_n/b² and ℓ_n/b without division, so k⊥ → 0 stays regular.
            over_b2 = half_b ** (2 * order - 2) / (4.0 * square)
            over_b = half_b ** (2 * order - 1) / (2.0 * square)
            columns.append(
                (
                    n * n * over_b2 * m0,
                    1j * n * order * over_b2 * m0,
                    n * n * over_b2 * m0,
                    n * over_b * m1,
                    -1j * order * over_b * m1,
                    ell * m2,
                )
            )
        entries = [jnp.stack(values, axis=-1) for values in zip(*columns, strict=True)]
        prefactor = (
            -_complex(self.plasma_frequency_squared.reshape(column))
            / (frequency * frequency)
            * mu_
            * mu_
        )
        terms = _assemble(*(prefactor * entry for entry in entries))
        return terms, jnp.zeros(omega.shape + (self.species_count,), dtype=jnp.complex128)

    def susceptibility(
        self,
        omega: ArrayLike,
        parallel_wavenumber: ArrayLike,
        perpendicular_wavenumber: ArrayLike,
        /,
    ) -> KineticSusceptibilityResult:
        """``χ_s`` and ``ε`` at (possibly complex) ``ω`` and real ``(k∥, k⊥)``.

        ``ω`` is float64 or complex128 (``Im ω < 0`` is the analytic continuation);
        ``k∥`` is signed and ``k⊥ ≥ 0``. Arguments broadcast.
        """
        frequency, k_par, k_perp = jnp.broadcast_arrays(
            _frequency_argument(omega, "omega"),
            _float64_argument(parallel_wavenumber, "parallel_wavenumber"),
            _float64_argument(perpendicular_wavenumber, "perpendicular_wavenumber"),
        )
        match self.model:
            case "nonrelativistic":
                terms, nonresonant = self._nonrelativistic_terms(frequency, k_par, k_perp)
            case "weakly-relativistic":
                terms, nonresonant = self._weakly_relativistic_terms(
                    frequency, k_par, k_perp
                )
            case _:
                raise ValueError(f"unknown kinetic susceptibility model {self.model!r}.")
        retained, omitted = _harmonic_split(terms)
        species = retained.at[..., 2, 2].add(nonresonant)
        identity = jnp.eye(3, dtype=jnp.complex128).reshape(
            (1,) * frequency.ndim + (3, 3)
        )
        dielectric = identity + jnp.sum(species, axis=-3)
        norm = jnp.sqrt(jnp.sum(jnp.abs(species) ** 2, axis=(-2, -1)))
        tail = jnp.sqrt(jnp.sum(jnp.abs(omitted) ** 2, axis=(-2, -1)))
        ratio = tail / jnp.where(norm > 0.0, norm, 1.0)
        row = (1,) * frequency.ndim + (self.species_count,)
        larmor = (k_perp[..., None] / self.cyclotron_frequency.reshape(row)) ** 2 * (
            self.perpendicular_temperatures / self.species_masses
        ).reshape(row)
        finite = jnp.all(jnp.isfinite(dielectric), axis=(-2, -1))
        truncated = jnp.any(~(ratio <= self.truncation_tolerance), axis=-1)
        match self.model:
            case "nonrelativistic":
                flr = jnp.zeros(finite.shape, dtype=jnp.bool_)
            case "weakly-relativistic":
                flr = jnp.any(larmor > self.larmor_tolerance, axis=-1)
            case _:
                raise ValueError(f"unknown kinetic susceptibility model {self.model!r}.")
        status = (
            truncated.astype(jnp.int32)
            * KineticSusceptibilityStatus.HARMONIC_TRUNCATION.value
            | flr.astype(jnp.int32)
            * KineticSusceptibilityStatus.LARMOR_RADIUS_LIMIT.value
            | (~finite).astype(jnp.int32) * KineticSusceptibilityStatus.NONFINITE.value
        )
        return KineticSusceptibilityResult(
            angular_frequency=frequency,
            parallel_wavenumber=k_par,
            perpendicular_wavenumber=k_perp,
            species_susceptibility=species,
            dielectric_tensor=dielectric,
            truncation_ratio=ratio,
            larmor_parameter=larmor,
            status=status,
            harmonic_count=self.harmonic_count,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Kinetic dispersion roots
# ---------------------------------------------------------------------------


class KineticDispersionResult(StrictModule):
    """Roots ``ω(k)`` along a continuation path with per-point failure evidence.

    ``frequencies`` are the Newton iterates at every path point, including
    failed ones (never zero-filled); ``converged`` and ``status`` say which are
    roots. ``predicted_frequencies`` are the secant predictions from the last two
    accepted roots, and a root farther than ``branch_tolerance`` (relative) from
    its prediction is flagged ``BRANCH_JUMP`` and not used to continue the
    branch. ``residual_norm`` is the normalized dispersion function at the
    iterate and ``condition_estimate`` that of its real 2×2 Jacobian.
    ``truncation_ratio`` is the harmonic-truncation evidence of the dielectric
    at each iterate.
    """

    __strict_contract__ = True

    wavenumbers: Float64[_PathDim]
    angles: Float64[_PathDim]
    frequencies: Complex128[_PathDim]
    predicted_frequencies: Complex128[_PathDim]
    residual_norm: Float64[_PathDim]
    condition_estimate: Float64[_PathDim]
    converged: Bool[_PathDim]
    truncation_ratio: Float64[_PathDim, _SpeciesDim]
    status: Int32[_PathDim]
    model: KineticDispersionModel = eqx.field(static=True)


class KineticDispersionProblem(StrictModule, NonTrainableState):
    """Complex-frequency roots of the kinetic dispersion relation.

    ``model="electromagnetic"`` solves ``det Λ / (1 + n²)² = 0`` with
    ``Λ = n²(κ̂κ̂ − I) + ε`` and ``n = ck/ω`` (the analytic factor keeps the
    residual ``O(1)`` from fast waves to whistlers); ``model="electrostatic"``
    solves ``κ̂·ε·κ̂ = 0``. ``ω`` is found as a real 2-vector root with
    `phydrax.nonlinear.VectorLocalRootPlan` (Newton with implicit-function
    derivatives), scaled by the magnitude of the initial frequency. A point is
    converged when the residual norm is below ``tolerance`` with a nonsingular
    Jacobian; ``maximum_condition`` bounds the Jacobian condition estimate.
    """

    __strict_contract__ = True

    dielectric: KineticPlasmaDielectric
    model: KineticDispersionModel = eqx.field(static=True)
    root: VectorLocalRootPlan
    maximum_condition: float = eqx.field(static=True)
    branch_tolerance: float = eqx.field(static=True)
    problem_id: Identifier = eqx.field(static=True)

    @checked
    def __init__(
        self,
        dielectric: KineticPlasmaDielectric,
        /,
        *,
        model: KineticDispersionModel = "electromagnetic",
        maximum_steps: int = 40,
        tolerance: float = 1.0e-10,
        maximum_condition: float = 1.0e12,
        branch_tolerance: float = 0.25,
    ) -> None:
        model_ = parse(model, KineticDispersionModel, "model")
        condition = float(maximum_condition)
        branch = float(branch_tolerance)
        if not isfinite(condition) or condition <= 1.0:
            raise ValueError("maximum_condition must be finite and above one.")
        if not isfinite(branch) or branch <= 0.0:
            raise ValueError("branch_tolerance must be finite and positive.")
        root = VectorLocalRootPlan(
            2,
            maximum_steps=maximum_steps,
            tolerance=tolerance,
            plan_id="kinetic-dispersion",
        )
        self.dielectric = dielectric
        self.model = model_
        self.root = root
        self.maximum_condition = condition
        self.branch_tolerance = branch
        self.problem_id = canonical_fingerprint(
            {
                "kind": "kinetic-dispersion-problem",
                "dielectric": dielectric.dielectric_id,
                "model": model_,
                "root": root.plan_id,
                "maximum_condition": condition,
                "branch_tolerance": branch,
            }
        )

    def dispersion_function(
        self, omega: ArrayLike, wavenumber: ArrayLike, angle: ArrayLike, /
    ) -> Array:
        """Normalized dispersion function at complex ``ω``, ``|k|`` and angle to ``B₀``."""
        frequency, k, theta = jnp.broadcast_arrays(
            _frequency_argument(omega, "omega"),
            _float64_argument(wavenumber, "wavenumber"),
            _float64_argument(angle, "angle"),
        )
        return self._dispersion(frequency, k, theta)

    def _dispersion(self, frequency: Array, k: Array, theta: Array, /) -> Array:
        sine = jnp.sin(theta)
        cosine = jnp.cos(theta)
        epsilon = self.dielectric.susceptibility(
            frequency, k * cosine, k * jnp.abs(sine)
        ).dielectric_tensor
        normal = _complex(
            jnp.stack((jnp.abs(sine), jnp.zeros_like(sine), cosine), axis=-1)
        )
        match self.model:
            case "electrostatic":
                return jnp.sum(normal * (epsilon @ normal[..., None])[..., 0], axis=-1)
            case "electromagnetic":
                n_squared = _complex(self.dielectric.speed_of_light * k) ** 2 / (
                    frequency * frequency
                )
                wave = (
                    n_squared[..., None, None]
                    * (
                        normal[..., :, None] * normal[..., None, :]
                        - jnp.eye(3, dtype=jnp.complex128).reshape((1,) * k.ndim + (3, 3))
                    )
                    + epsilon
                )
                determinant = determinant_small_linear(SmallLinearSolvePlan(3), wave)
                return determinant / (1.0 + n_squared) ** 2
            case _:
                raise ValueError(f"unknown kinetic dispersion model {self.model!r}.")

    def solve(
        self,
        wavenumbers: ArrayLike,
        angles: ArrayLike,
        initial_frequency: ArrayLike,
        /,
    ) -> KineticDispersionResult:
        """Continue one branch along ``wavenumbers[K]`` from ``initial_frequency``.

        ``angles`` (radians between ``k`` and ``B₀``) broadcast to the path. The
        first point starts from ``initial_frequency``; later points start from
        the secant extrapolation of the last two accepted roots.
        """
        k_path = _float64_argument(wavenumbers, "wavenumbers")
        if k_path.ndim != 1 or k_path.shape[0] < 1:
            raise ValueError("wavenumbers must be a nonempty one-dimensional path.")
        theta_path = jnp.broadcast_to(_float64_argument(angles, "angles"), k_path.shape)
        start = _frequency_argument(initial_frequency, "initial_frequency")
        if start.ndim != 0:
            raise ValueError("initial_frequency must be a scalar.")
        frequency_scale = jnp.abs(start)

        # carry: (last, before-last) accepted roots, their wavenumbers, accepted count.
        type Carry = tuple[Array, Array, Array, Array, Array]

        def step(
            carry: Carry, point: tuple[Array, Array]
        ) -> tuple[Carry, tuple[Array, ...]]:
            last, before_last, last_k, before_last_k, accepted = carry
            k, theta = point
            span = last_k - before_last_k
            slope = jnp.where(
                accepted >= 2,
                (last - before_last) / _complex(jnp.where(span == 0.0, 1.0, span)),
                0.0j,
            )
            predicted = last + slope * _complex(k - last_k)

            def residual(x: Array, /) -> Array:
                value = self._dispersion(
                    _complex(frequency_scale) * jax.lax.complex(x[0], x[1]), k, theta
                )
                return jnp.stack((value.real, value.imag))

            guess = jnp.stack((predicted.real, predicted.imag)) / frequency_scale
            solution, diagnostics = self.root.solve_with_diagnostics(residual, guess)
            frequency = _complex(frequency_scale) * jax.lax.complex(
                solution[0], solution[1]
            )
            jump = jnp.abs(frequency - predicted) > self.branch_tolerance * jnp.abs(
                predicted
            )
            conditioned = diagnostics.condition_estimate <= self.maximum_condition
            accept = diagnostics.converged & conditioned & ~jump
            updated = (
                jnp.where(accept, frequency, last),
                jnp.where(accept, last, before_last),
                jnp.where(accept, k, last_k),
                jnp.where(accept, last_k, before_last_k),
                accepted + accept.astype(jnp.int32),
            )
            return updated, (
                frequency,
                predicted,
                diagnostics.residual_norm,
                diagnostics.condition_estimate,
                diagnostics.converged,
                jump,
            )

        initial = (
            start,
            start,
            k_path[0],
            k_path[0],
            jnp.asarray(0, dtype=jnp.int32),
        )
        _, (frequencies, predicted, residual, condition, converged, jump) = jax.lax.scan(
            step, initial, (k_path, theta_path)
        )
        evidence = self.dielectric.susceptibility(
            frequencies,
            k_path * jnp.cos(theta_path),
            k_path * jnp.abs(jnp.sin(theta_path)),
        )
        ill = ~(condition <= self.maximum_condition)
        finite = jnp.isfinite(frequencies) & jnp.isfinite(residual)
        status = (
            (~converged).astype(jnp.int32) * KineticDispersionStatus.NOT_CONVERGED.value
            | ill.astype(jnp.int32) * KineticDispersionStatus.ILL_CONDITIONED.value
            | jump.astype(jnp.int32) * KineticDispersionStatus.BRANCH_JUMP.value
            | (
                (evidence.status & KineticSusceptibilityStatus.HARMONIC_TRUNCATION.value)
                != 0
            ).astype(jnp.int32)
            * KineticDispersionStatus.HARMONIC_TRUNCATION.value
            | (
                (evidence.status & KineticSusceptibilityStatus.LARMOR_RADIUS_LIMIT.value)
                != 0
            ).astype(jnp.int32)
            * KineticDispersionStatus.LARMOR_RADIUS_LIMIT.value
            | (~finite).astype(jnp.int32) * KineticDispersionStatus.NONFINITE.value
        )
        return KineticDispersionResult(
            wavenumbers=k_path,
            angles=theta_path,
            frequencies=frequencies,
            predicted_frequencies=predicted,
            residual_norm=residual,
            condition_estimate=condition,
            converged=converged & ~ill & finite,
            truncation_ratio=evidence.truncation_ratio,
            status=status,
            model=self.model,
        )


# ---------------------------------------------------------------------------
# Driver distributions for relativistic weak growth
# ---------------------------------------------------------------------------


def _upper_gamma_half_integer(order: int, x: float, /) -> float:
    """Regularized ``Q(order + 1/2, x)`` by the exact upward recurrence from ``erfc``."""
    value = erfc(sqrt(x))
    a = 0.5
    for _ in range(order):
        value += exp(a * log(x) - x - lgamma(a + 1.0)) if x > 0.0 else 0.0
        a += 1.0
    return value


def _bisect_support(tail: Callable[[float], float], tolerance: float, /) -> float:
    """Smallest bracketed radius whose monotone tail mass is below ``tolerance``."""
    lower, upper = 0.0, 1.0
    while tail(upper) > tolerance:
        lower, upper = upper, 2.0 * upper
    for _ in range(200):
        middle = 0.5 * (lower + upper)
        if tail(middle) > tolerance:
            lower = middle
        else:
            upper = middle
    return upper


def _positive(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return number


def _tail_tolerance(value: float, /) -> float:
    tolerance = _positive(value, "tail_tolerance")
    if tolerance >= 0.5:
        raise ValueError("tail_tolerance must be below 1/2.")
    return tolerance


def _shell_moment(center: float, width: float, start: float, /) -> float:
    """``∫_start^∞ u² exp(−(u − center)²/width²) du`` in closed form."""
    lower = (start - center) / width
    gauss = exp(-lower * lower)
    return width * (
        center * center * 0.5 * _SQRT_PI * erfc(lower)
        + center * width * gauss
        + width * width * (0.5 * lower * gauss + 0.25 * _SQRT_PI * erfc(lower))
    )


def _ring_moment(center: float, width: float, start: float, /) -> float:
    """``∫_start^∞ u exp(−(u − center)²/width²) du`` in closed form."""
    lower = (start - center) / width
    return width * (
        0.5 * width * exp(-lower * lower) + center * 0.5 * _SQRT_PI * erfc(lower)
    )


class LossConeDistribution(AbstractGyrotropicDistribution):
    """Dory–Guest–Harris loss cone ``f ∝ (u⊥/u_t)^{2j} exp(−u²/u_t²)``.

    ``thermal_momentum = u_t`` and integer ``index = j ≥ 1`` (``j = 0`` is the
    Maxwellian); ``∂f/∂u⊥ > 0`` for ``u⊥ < √j u_t`` drives cyclotron maser
    growth. The normalization is ``π^{3/2} u_t³ j!`` and the support radius
    bounds the omitted mass by ``tail_tolerance`` through the exact
    ``Q(j + 3/2, R²/u_t²)``.
    """

    thermal_momentum: float = eqx.field(static=True)
    index: int = eqx.field(static=True)
    support_radius: float = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)

    def __init__(
        self,
        thermal_momentum: float,
        /,
        *,
        index: int,
        tail_tolerance: float = 1.0e-15,
    ) -> None:
        width = _positive(thermal_momentum, "thermal_momentum")
        order = int(index)
        if order < 1:
            raise ValueError("index must be a positive integer.")
        tolerance = _tail_tolerance(tail_tolerance)
        radius = _bisect_support(
            lambda r: _upper_gamma_half_integer(order + 1, (r / width) ** 2), tolerance
        )
        self.thermal_momentum = width
        self.index = order
        self.support_radius = radius
        self.tail_bound = _upper_gamma_half_integer(order + 1, (radius / width) ** 2)

    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        width = self.thermal_momentum
        ratio = u_perp / width
        return self.index * jnp.log(ratio * ratio) - (
            ratio * ratio + (u_par / width) ** 2
        )

    def log_normalization(self) -> Array:
        return jnp.asarray(
            1.5 * log(pi) + 3.0 * log(self.thermal_momentum) + lgamma(self.index + 1.0),
            dtype=jnp.float64,
        )

    def momentum_support(self) -> tuple[Array, Array]:
        return jnp.zeros((), dtype=jnp.float64), jnp.asarray(
            self.support_radius, dtype=jnp.float64
        )

    def tail_mass_bound(self) -> Array:
        return jnp.asarray(self.tail_bound, dtype=jnp.float64)


class RingDistribution(AbstractGyrotropicDistribution):
    """Ring ``f ∝ exp(−(u⊥ − u_r)²/Δ⊥² − u∥²/Δ∥²)`` in normalized momentum."""

    ring_momentum: float = eqx.field(static=True)
    perpendicular_spread: float = eqx.field(static=True)
    parallel_spread: float = eqx.field(static=True)
    support_radius: float = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)

    def __init__(
        self,
        ring_momentum: float,
        /,
        *,
        perpendicular_spread: float,
        parallel_spread: float,
        tail_tolerance: float = 1.0e-15,
    ) -> None:
        center = float(ring_momentum)
        if not isfinite(center) or center < 0.0:
            raise ValueError("ring_momentum must be finite and nonnegative.")
        perpendicular = _positive(perpendicular_spread, "perpendicular_spread")
        parallel = _positive(parallel_spread, "parallel_spread")
        tolerance = _tail_tolerance(tail_tolerance)
        total = _ring_moment(center, perpendicular, 0.0)

        def tail(t: float) -> float:
            # u > R with R² = (u_r + tΔ⊥)² + (tΔ∥)² needs u⊥ > u_r + tΔ⊥ or |u∥| > tΔ∥.
            return _ring_moment(
                center, perpendicular, center + t * perpendicular
            ) / total + erfc(t)

        t = _bisect_support(tail, tolerance)
        self.ring_momentum = center
        self.perpendicular_spread = perpendicular
        self.parallel_spread = parallel
        self.support_radius = sqrt(
            (center + t * perpendicular) ** 2 + (t * parallel) ** 2
        )
        self.tail_bound = tail(t)

    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        return -(
            ((u_perp - self.ring_momentum) / self.perpendicular_spread) ** 2
            + (u_par / self.parallel_spread) ** 2
        )

    def log_normalization(self) -> Array:
        perpendicular = _ring_moment(self.ring_momentum, self.perpendicular_spread, 0.0)
        return jnp.asarray(
            log(2.0 * pi * _SQRT_PI * self.parallel_spread * perpendicular),
            dtype=jnp.float64,
        )

    def momentum_support(self) -> tuple[Array, Array]:
        return jnp.zeros((), dtype=jnp.float64), jnp.asarray(
            self.support_radius, dtype=jnp.float64
        )

    def tail_mass_bound(self) -> Array:
        return jnp.asarray(self.tail_bound, dtype=jnp.float64)


class HorseshoeDistribution(AbstractGyrotropicDistribution):
    """Shell with a loss cone: ``f ∝ exp(−(u − u_s)²/Δ²) L(α)``.

    ``L(α) = 1/(1 + exp(−(α_p − α_c)/w))`` with the pitch angle folded to
    ``α_p ∈ [0, π/2]`` depletes both loss cones of half-angle
    ``loss_cone_angle = α_c`` over the edge width ``w`` (radians, ``≥ 0.01``).
    The radial normalization is closed form; the pitch-angle integral uses a
    256-node Gauss–Legendre rule on host preparation.
    """

    shell_momentum: float = eqx.field(static=True)
    spread: float = eqx.field(static=True)
    loss_cone_angle: float = eqx.field(static=True)
    edge_width: float = eqx.field(static=True)
    angular_weight: float = eqx.field(static=True)
    support_radius: float = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)

    def __init__(
        self,
        shell_momentum: float,
        /,
        *,
        spread: float,
        loss_cone_angle: float,
        edge_width: float,
        tail_tolerance: float = 1.0e-15,
    ) -> None:
        center = _positive(shell_momentum, "shell_momentum")
        width = _positive(spread, "spread")
        angle = float(loss_cone_angle)
        if not isfinite(angle) or not 0.0 <= angle < 0.5 * pi:
            raise ValueError("loss_cone_angle must lie in [0, π/2).")
        edge = float(edge_width)
        if not isfinite(edge) or edge < 0.01:
            raise ValueError("edge_width must be finite and at least 0.01 rad.")
        tolerance = _tail_tolerance(tail_tolerance)
        total = _shell_moment(center, width, 0.0)
        radius = center + width * _bisect_support(
            lambda t: _shell_moment(center, width, center + t * width) / total, tolerance
        )
        rule = gauss_legendre_data(256)
        cosine = 0.5 * (np.asarray(rule.nodes, dtype=np.float64) + 1.0)
        pitch = np.arccos(cosine)
        loss_cone = 1.0 / (1.0 + np.exp(-(pitch - angle) / edge))
        self.shell_momentum = center
        self.spread = width
        self.loss_cone_angle = angle
        self.edge_width = edge
        # ∫_{−1}^{1} L dμ = 2 ∫_0^1 L dμ by the folding symmetry.
        self.angular_weight = float(np.sum(np.asarray(rule.weights) * loss_cone))
        self.support_radius = radius
        self.tail_bound = _shell_moment(center, width, radius) / total

    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        magnitude = jnp.sqrt(u_perp * u_perp + u_par * u_par)
        pitch = jnp.arctan2(u_perp, jnp.abs(u_par))
        radial = -(((magnitude - self.shell_momentum) / self.spread) ** 2)
        return radial - jax.nn.softplus(-(pitch - self.loss_cone_angle) / self.edge_width)

    def log_normalization(self) -> Array:
        radial = _shell_moment(self.shell_momentum, self.spread, 0.0)
        return jnp.asarray(
            log(2.0 * pi * radial * self.angular_weight), dtype=jnp.float64
        )

    def momentum_support(self) -> tuple[Array, Array]:
        return jnp.zeros((), dtype=jnp.float64), jnp.asarray(
            self.support_radius, dtype=jnp.float64
        )

    def tail_mass_bound(self) -> Array:
        return jnp.asarray(self.tail_bound, dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Relativistic weak growth on the resonance ellipse
# ---------------------------------------------------------------------------


def _signed_bessel(order: Array, argument: Array, /) -> Array:
    """``J_s(x)`` for integer-valued ``s`` and real ``x`` of either sign.

    ``J_{−s}(x) = J_s(−x) = (−1)^s J_s(x)``; the native ``jv`` takes ``x ≥ 0``.
    """
    magnitude = jnp.abs(order)
    value = jv(magnitude, jnp.abs(argument))
    odd = jnp.mod(magnitude, 2.0) == 1.0
    flipped = odd & ((argument < 0.0) != (order < 0.0))
    return jnp.where(flipped, -value, value)


class RelativisticWeakGrowthResult(StrictModule):
    """Weak growth (``> 0``) or damping (``< 0``) rate of one cold-plasma mode.

    ``anti_hermitian_susceptibility = (χ − χ†)/(2i)`` of the energetic species
    at real ``(ω, k)`` of the selected mode; ``growth_rate`` is
    ``−ω² e*·χᴬ·e / e*·∂_ω(ω² Kᴴ)·e`` (per time unit, amplitude growth) with the
    cold background's Hermitian ``Kᴴ`` and unit polarization ``e``.
    ``harmonic_growth_rates`` resolves it by harmonic ``s = −N … N`` and
    ``resonant`` marks harmonics whose resonance ellipse
    ``γ − N∥u∥ − sΩ/ω = 0`` exists. ``quadrature_error`` is the relative
    difference from the half-order ellipse rule. Unsupported points
    (non-propagating mode, ``N∥² ≥ 1``) carry NaN rates with their status bit.
    """

    __strict_contract__ = True

    angular_frequency: Float64[_BatchDims]
    angle: Float64[_BatchDims]
    refractive_index: Float64[_BatchDims]
    polarization: Complex128[_BatchDims, Literal[3]]
    anti_hermitian_susceptibility: Complex128[_BatchDims, Literal[3], Literal[3]]
    harmonic_growth_rates: Float64[_BatchDims, _HarmonicDim]
    resonant: Bool[_BatchDims, _HarmonicDim]
    growth_rate: Float64[_BatchDims]
    quadrature_error: Float64[_BatchDims]
    status: Int32[_BatchDims]
    harmonic_count: int = eqx.field(static=True)


class RelativisticWeakGrowthPlan(StrictModule, NonTrainableState):
    """Weak-growth rate of cold-plasma modes driven by an energetic gyrotropic species.

    The energetic species (``density`` per cubic length unit, ``charge_number``,
    ``mass_ratio``, normalized momentum ``distribution``) is dilute: it does not
    change the cold modes of ``background`` and enters only through the
    anti-Hermitian part of its fully relativistic susceptibility (Wu & Lee 1979;
    Melrose & Dulk 1982)

    ``χᴬ = −2π² (ω_h²/ω²) Σ_s ∫ du∥ γ² [sY ∂_{u⊥}f/u⊥ + N∥ ∂_{u∥}f] Ṽ_s* Ṽ_sᵀ``

    on the resonance ellipse ``γ = sY + N∥u∥`` (``Y = Ω₀/ω`` signed,
    ``N∥² < 1``), with ``Ṽ_s = (β⊥ sJ_s/a, iβ⊥J_s', β∥J_s)`` and
    ``a = N⊥u⊥/Y``. The ellipse is parameterized by its eccentric anomaly and
    integrated with a ``quadrature_order`` Gauss–Legendre rule.
    """

    background: ColdPlasmaDielectric
    distribution: AbstractGyrotropicDistribution
    density: float = eqx.field(static=True)
    charge_number: float = eqx.field(static=True)
    mass_ratio: float = eqx.field(static=True)
    harmonic_count: int = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    quadrature_tolerance: float = eqx.field(static=True)
    weak_growth_tolerance: float = eqx.field(static=True)

    @checked
    def __init__(
        self,
        background: ColdPlasmaDielectric,
        distribution: AbstractGyrotropicDistribution,
        /,
        *,
        density: float,
        charge_number: float = -1.0,
        mass_ratio: float = 1.0,
        harmonic_count: int = 4,
        quadrature_order: int = 64,
        quadrature_tolerance: float = 1.0e-6,
        weak_growth_tolerance: float = 0.1,
    ) -> None:
        charge = float(charge_number)
        if not isfinite(charge) or charge == 0.0:
            raise ValueError("charge_number must be finite and nonzero.")
        count = int(harmonic_count)
        order = int(quadrature_order)
        if count < 1:
            raise ValueError("harmonic_count must be a positive integer.")
        if order < 8 or order % 2:
            raise ValueError("quadrature_order must be an even integer of at least 8.")
        self.background = background
        self.distribution = distribution
        self.density = _positive(density, "density")
        self.charge_number = charge
        self.mass_ratio = _positive(mass_ratio, "mass_ratio")
        self.harmonic_count = count
        self.quadrature_order = order
        self.quadrature_tolerance = _positive(
            quadrature_tolerance, "quadrature_tolerance"
        )
        self.weak_growth_tolerance = _positive(
            weak_growth_tolerance, "weak_growth_tolerance"
        )

    @property
    def plasma_frequency_squared(self) -> float:
        scale = self.background.scale
        charge = self.charge_number * float(scale.elementary_charge)
        return (
            self.density
            * charge
            * charge
            / (
                float(scale.vacuum_permittivity)
                * self.mass_ratio
                * float(scale.electron_mass)
            )
        )

    @property
    def cyclotron_frequency(self) -> Array:
        """Signed rest-mass ``Ω₀ = Z e |B₀| / m`` of the energetic species."""
        scale = self.background.scale
        return (
            self.charge_number
            * float(scale.elementary_charge)
            * self.background.magnetic_field_magnitude
            / (self.mass_ratio * float(scale.electron_mass))
        )

    def _anti_hermitian(
        self, n_par: Array, n_perp: Array, y: Array, order: int, /
    ) -> tuple[Array, Array]:
        """Per-harmonic ``χᴬ / (ω_h²/ω²)`` ``[..., H, 3, 3]`` and resonance mask.

        Harmonics are a vectorized axis so that one Bessel evaluation serves
        every order ``s − 1, s, s + 1``.
        """
        rule = gauss_legendre_data(order)
        leading = (1,) * n_par.ndim
        anomaly = (0.5 * pi * (jnp.asarray(rule.nodes, dtype=jnp.float64) + 1.0)).reshape(
            leading + (1, order)
        )
        weight = (0.5 * pi * jnp.asarray(rule.weights, dtype=jnp.float64)).reshape(
            leading + (1, order)
        )
        count = self.harmonic_count
        harmonics = jnp.arange(-count, count + 1, dtype=jnp.float64).reshape(
            leading + (2 * count + 1, 1)
        )
        parallel = n_par[..., None, None]
        perpendicular = n_perp[..., None, None]
        ratio = y[..., None, None]
        eccentricity = 1.0 - parallel * parallel
        elliptic = eccentricity > 0.0
        safe_ecc = jnp.where(elliptic, eccentricity, 1.0)
        sy = harmonics * ratio  # [..., H, 1]
        discriminant = sy * sy - safe_ecc
        resonant = elliptic & (sy > 0.0) & (discriminant > 0.0)
        root = jnp.sqrt(jnp.where(resonant, discriminant, 1.0))
        half_par = root / safe_ecc
        u_par = sy * parallel / safe_ecc + half_par * jnp.cos(anomaly)
        u_perp = root / jnp.sqrt(safe_ecc) * jnp.sin(anomaly)  # [..., H, P]
        gamma = sy + parallel * u_par
        d_perp, d_par = self.distribution.density_gradient(u_perp, u_par)
        drive = sy * d_perp / u_perp + parallel * d_par
        argument = perpendicular * u_perp / ratio
        shifts = jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float64).reshape(
            (3,) + leading + (1, 1)
        )
        lower, center, upper = _signed_bessel(harmonics[None] + shifts, argument[None])
        beta_perp = u_perp / gamma
        beta_par = u_par / gamma
        vector = jnp.stack(
            (
                _complex(0.5 * beta_perp * (lower + upper)),
                0.5j * _complex(beta_perp * (lower - upper)),
                _complex(beta_par * center),
            ),
            axis=-1,
        )
        measure = weight * half_par * jnp.sin(anomaly) * gamma * gamma * drive
        outer = jnp.conj(vector)[..., :, None] * vector[..., None, :]
        block = (
            -2.0 * pi * pi * jnp.sum(_complex(measure)[..., None, None] * outer, axis=-3)
        )
        return jnp.where(resonant[..., None], block, 0.0j), resonant[..., 0]

    def evaluate(
        self, omega: ArrayLike, theta: ArrayLike, mode: PlasmaWaveMode, /
    ) -> RelativisticWeakGrowthResult:
        """Growth rate of ``mode`` at real ``ω`` and wave-normal angle ``θ``."""
        mode_ = parse(mode, PlasmaWaveMode, "mode")
        frequency, angle = jnp.broadcast_arrays(
            _float64_argument(omega, "omega"), _float64_argument(theta, "theta")
        )
        wave = self.background.refractive_indices(frequency, angle)
        n_squared = wave.select(mode_, wave.n_squared)
        polarization = wave.select(mode_, wave.polarization)
        mode_status = wave.select(mode_, wave.status)
        match mode_:
            case PlasmaWaveMode.RIGHT | PlasmaWaveMode.LEFT:
                ambiguous_bit = ColdPlasmaWaveStatus.PARALLEL_LABEL_AMBIGUOUS.value
            case PlasmaWaveMode.ORDINARY | PlasmaWaveMode.EXTRAORDINARY:
                ambiguous_bit = ColdPlasmaWaveStatus.PERPENDICULAR_LABEL_AMBIGUOUS.value
            case _:
                raise ValueError(f"unknown plasma wave mode {mode_!r}.")
        propagating = (
            (n_squared.imag == 0.0)
            & (n_squared.real > 0.0)
            & jnp.isfinite(n_squared)
            & (
                (
                    mode_status
                    & (
                        ColdPlasmaWaveStatus.RESONANT.value
                        | ColdPlasmaWaveStatus.POLARIZATION_UNDEFINED.value
                    )
                )
                == 0
            )
        )
        index = jnp.sqrt(jnp.where(propagating, n_squared.real, 1.0))
        n_par = index * jnp.cos(angle)
        n_perp = index * jnp.abs(jnp.sin(angle))
        y = self.cyclotron_frequency / frequency
        elliptic = n_par * n_par < 1.0
        scale = self.plasma_frequency_squared / (frequency * frequency)
        blocks, resonant = _resonant_blocks(self, n_par, n_perp, y, self.quadrature_order)
        coarse, _ = _resonant_blocks(self, n_par, n_perp, y, self.quadrature_order // 2)
        blocks = _complex(scale)[..., None, None, None] * blocks
        coarse = _complex(scale)[..., None, None] * jnp.sum(coarse, axis=-3)
        anti_hermitian = jnp.sum(blocks, axis=-3)

        def hermitian_response(value: Array, /) -> Array:
            tensor = self.background.dielectric_tensor(value)
            hermitian = 0.5 * (tensor + jnp.conj(jnp.swapaxes(tensor, -1, -2)))
            return _complex(value * value)[..., None, None] * hermitian

        _, response = jax.jvp(
            hermitian_response, (frequency,), (jnp.ones_like(frequency),)
        )
        energy = contract(
            "...i,...ij,...j->...", jnp.conj(polarization), response, polarization
        ).real
        omega_squared = frequency * frequency
        harmonic_rates = (
            -omega_squared[..., None]
            * contract(
                "...i,...hij,...j->...h", jnp.conj(polarization), blocks, polarization
            ).real
            / energy[..., None]
        )
        growth = jnp.sum(harmonic_rates, axis=-1)
        coarse_growth = (
            -omega_squared
            * contract(
                "...i,...ij,...j->...", jnp.conj(polarization), coarse, polarization
            ).real
            / energy
        )
        error = jnp.abs(growth - coarse_growth) / jnp.maximum(
            jnp.abs(growth), jnp.finfo(jnp.float64).tiny
        )
        supported = propagating & elliptic
        growth = jnp.where(supported, growth, jnp.nan)
        harmonic_rates = jnp.where(supported[..., None], harmonic_rates, jnp.nan)
        any_resonant = jnp.any(resonant, axis=-1)
        status = (
            (~propagating).astype(jnp.int32) * WeakGrowthStatus.MODE_NOT_PROPAGATING.value
            | (propagating & ~elliptic).astype(jnp.int32)
            * WeakGrowthStatus.RESONANCE_NOT_ELLIPTIC.value
            | (supported & ~any_resonant).astype(jnp.int32)
            * WeakGrowthStatus.NO_RESONANCE.value
            | (supported & any_resonant & ~(error <= self.quadrature_tolerance)).astype(
                jnp.int32
            )
            * WeakGrowthStatus.QUADRATURE_UNRESOLVED.value
            | (
                supported & ~(jnp.abs(growth) <= self.weak_growth_tolerance * frequency)
            ).astype(jnp.int32)
            * WeakGrowthStatus.WEAK_GROWTH_VIOLATED.value
            | ((mode_status & ambiguous_bit) != 0).astype(jnp.int32)
            * WeakGrowthStatus.MODE_LABEL_AMBIGUOUS.value
            | (supported & ~jnp.isfinite(growth)).astype(jnp.int32)
            * WeakGrowthStatus.NONFINITE.value
        )
        return RelativisticWeakGrowthResult(
            angular_frequency=frequency,
            angle=angle,
            refractive_index=jnp.where(propagating, index, jnp.nan),
            polarization=polarization,
            anti_hermitian_susceptibility=anti_hermitian,
            harmonic_growth_rates=harmonic_rates,
            resonant=resonant,
            growth_rate=growth,
            quadrature_error=error,
            status=status,
            harmonic_count=self.harmonic_count,
        )


@eqx.filter_jit
def _resonant_blocks(
    plan: RelativisticWeakGrowthPlan,
    n_par: Array,
    n_perp: Array,
    y: Array,
    order: int,
    /,
) -> tuple[Array, Array]:
    """Compiled resonance-ellipse integral (stable entry point; ``order`` is static)."""
    return plan._anti_hermitian(n_par, n_perp, y, order)


__all__ = [
    "HorseshoeDistribution",
    "KineticDispersionModel",
    "KineticDispersionProblem",
    "KineticDispersionResult",
    "KineticDispersionStatus",
    "KineticPlasmaDielectric",
    "KineticSusceptibilityModel",
    "KineticSusceptibilityResult",
    "KineticSusceptibilityStatus",
    "LossConeDistribution",
    "RelativisticWeakGrowthPlan",
    "RelativisticWeakGrowthResult",
    "RingDistribution",
    "WeakGrowthStatus",
]
