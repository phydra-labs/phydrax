#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Mode-resolved magnetobremsstrahlung (cyclotron, gyrosynchrotron, synchrotron).

A gyrotropic population of charges ``q = Z e``, mass ``m = μ m_e``, number density
``N`` and normalized momentum distribution ``f(u⊥, u∥)`` (``u = p/(m c)``,
``∫ f d³u = 1``) radiates into the two cold-plasma modes ``σ`` of a
`ColdPlasmaDielectric` (Melrose 1968; Ramaty 1969; Melrose & McPhedran 1991).
For a mode with refractive index ``n``, unit polarization ``e`` (Stix frame,
``B₀ ∥ ẑ``, ``κ̂ = (sin θ, 0, cos θ)``) and transverse power ``|e_T|²``, the
emission coefficient per unit volume, angular frequency and wave-normal solid
angle is::

    j_σ = q² ω² n / (8π² ε₀ c³ |e_T|²) Σ_s ∫ d³p N f |e*·V_s|² δ(ω − sΩ/γ − k∥ v∥)

with ``Ω = |q| B / m``, ``V_s = (v⊥ (J_{s−1} + J_{s+1})/2, iε v⊥ (J_{s−1} − J_{s+1})/2,
v∥ J_s)`` at ``x = k⊥ v⊥ γ/Ω`` and ``ε = −sign(q)`` the gyration sense. The
normalization ``n/|e_T|²`` is the exact mode-energy factor
``n² R ∂(nω)/∂ω = n/(2|e_T|²)`` of a Hermitian (collisionless) dielectric. The
absorption coefficient per unit length along the wave normal is::

    α_σ = −π q² / (ε₀ ω c n |e_T|²) Σ_s ∫ d³p N |e*·V_s|² δ(…) [(sΩ/(γv⊥)) ∂f/∂p⊥ + k∥ ∂f/∂p∥]

so a thermal (Jüttner) population obeys Kirchhoff's law ``j_σ = n² (k T ω²/(8π³c²)) α_σ``
exactly (classical emission gives the Rayleigh–Jeans limit).

Routes:

- ``"harmonic-sum"``: exact integer harmonic sum. The delta function is
  resolved on the resonance curve ``γ = sY + N∥ u∥`` (``Y = Ω/ω``, ``N∥ = n cos θ``)
  parametrized by ``u∥`` with ``∫ d³u δ(…) F = (2π/ω) ∫ du∥ γ² F``. The
  harmonic set is every ``s`` whose resonance ellipse (``N∥² < 1``) or hyperbola
  (``N∥² ≥ 1``, including ``s ≤ 0`` anomalous-Doppler harmonics) intersects the
  distribution's momentum support. Each resonance interval is integrated in a
  logarithmic ``γ`` map with an embedded Gauss–Kronrod rule.
- ``"continuous-harmonic"``: the Fleishman & Kuznetsov (2010) continuous-harmonic
  limit, ``Σ_s → ∫ ds``, with exact real-order Bessel functions over the full
  ``(u, cos α)`` plane, pitch nodes clustered on the emission cone of width
  ``√(1 − n²β²)`` (the Razin-suppressed beaming width). Valid when many
  harmonics contribute; the emission-weighted harmonic number is reported.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from enum import IntFlag
from math import exp, isfinite, log, pi
from typing import assert_never, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._interpolation import apply_gather_stencil, rectilinear_stencil
from .._numerics._quadrature_rules import gauss_kronrod_data
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..special import jv, kve
from ..typing import (
    as_host_array,
    checked,
    ConvertibleToArray,
    Dim,
    Float64,
    HostFloat64,
    Int32,
    parse,
    Scalar,
    Scope,
    VariadicDim,
)
from ._cold_plasma import (
    ColdPlasmaDielectric,
    ColdPlasmaWaveStatus,
    FaradayCoefficients,
    PlasmaWaveMode,
)


MagnetobremsstrahlungRoute: TypeAlias = Literal["harmonic-sum", "continuous-harmonic"]


class _BatchDims(VariadicDim):
    """Broadcast batch of ``(ω, θ)`` evaluation points."""


class _MomentumDim(Dim, minimum=2):
    """Momentum-magnitude nodes of a tabulated distribution."""


class _PitchDim(Dim, minimum=2):
    """Pitch-angle-cosine nodes of a tabulated distribution."""


class MagnetobremsstrahlungStatus(IntFlag):
    """Per-root evidence flags of `MagnetobremsstrahlungResult.status`.

    ``EVANESCENT``, ``RESONANCE_CONE``, ``POLARIZATION_UNDEFINED``,
    ``HARMONIC_CAPACITY_EXCEEDED``, ``ANOMALOUS_DOPPLER`` and ``NONFINITE`` make the
    root unsupported (NaN coefficients). ``QUADRATURE_UNRESOLVED``,
    ``LOW_HARMONIC`` and ``COLLISIONAL_MEDIUM`` qualify supported values.
    """

    NONE = 0
    EVANESCENT = 1
    RESONANCE_CONE = 2
    POLARIZATION_UNDEFINED = 4
    HARMONIC_CAPACITY_EXCEEDED = 8
    ANOMALOUS_DOPPLER = 16
    NONFINITE = 32
    QUADRATURE_UNRESOLVED = 64
    LOW_HARMONIC = 128
    COLLISIONAL_MEDIUM = 256


_UNSUPPORTED = (
    MagnetobremsstrahlungStatus.EVANESCENT
    | MagnetobremsstrahlungStatus.RESONANCE_CONE
    | MagnetobremsstrahlungStatus.POLARIZATION_UNDEFINED
    | MagnetobremsstrahlungStatus.HARMONIC_CAPACITY_EXCEEDED
    | MagnetobremsstrahlungStatus.ANOMALOUS_DOPPLER
    | MagnetobremsstrahlungStatus.NONFINITE
).value


def _host_positive(value: float, name: str, /) -> float:
    number = float(value)
    if not isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _composite_rule(order: int, panels: int, /) -> tuple[Array, Array, Array]:
    """Composite embedded Gauss–Kronrod nodes and weights on ``[0, 1]``."""
    rule = gauss_kronrod_data(order)
    if rule.embedded_weights is None:
        raise RuntimeError("The Gauss–Kronrod rule carries no embedded Gauss weights.")
    unit = 0.5 * (rule.nodes + 1.0)
    offsets = jnp.arange(panels, dtype=jnp.float64)[:, None]
    nodes = ((offsets + unit[None, :]) / panels).reshape((-1,))
    kronrod = jnp.tile(0.5 * rule.weights, panels) / panels
    gauss = jnp.tile(0.5 * rule.embedded_weights, panels) / panels
    return nodes, kronrod, gauss


def _integer_bessel(order: Array, x: Array, /) -> Array:
    """``J_s(x)`` for integer-valued ``s`` of either sign, ``x ≥ 0``."""
    magnitude = jnp.abs(order)
    value = jv(magnitude, x)
    odd = jnp.mod(magnitude, 2.0) == 1.0
    return jnp.where((order < 0.0) & odd, -value, value)


def _bessel_triplet(
    order: Array, x: Array, bessel: Callable[[Array, Array], Array], /
) -> tuple[Array, Array, Array]:
    """``(J_{s−1}, J_s, J_{s+1})`` from two evaluations and one downward recurrence step.

    ``J_{s−1} = (2s/x) J_s − J_{s+1}`` is a single algebraically exact step; at
    ``x = 0`` it is replaced by its limit ``δ_{s,1}``.
    """
    center = bessel(order, x)
    upper = bessel(order + 1.0, x)
    positive = x > 0.0
    lower = jnp.where(
        positive,
        2.0 * order * center / jnp.where(positive, x, 1.0) - upper,
        (order == 1.0).astype(x.dtype),
    )
    return lower, center, upper


# ---------------------------------------------------------------------------
# Distributions
# ---------------------------------------------------------------------------


class AbstractGyrotropicDistribution(StrictModule):
    """Gyrotropic momentum distribution ``f(u⊥, u∥)`` per ``d³u``, ``u = p/(m c)``.

    Implementations return the unnormalized ``log f`` (``−∞`` outside the
    support), its logarithmic normalization ``log ∫ exp(log f) d³u``, the
    momentum-magnitude support used for integration, and a bound on the
    normalized probability mass outside that support.
    """

    @abc.abstractmethod
    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def log_normalization(self) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def momentum_support(self) -> tuple[Array, Array]:
        raise NotImplementedError

    @abc.abstractmethod
    def tail_mass_bound(self) -> Array:
        raise NotImplementedError

    def density(self, u_perp: ArrayLike, u_par: ArrayLike, /) -> Array:
        """Normalized ``f(u⊥, u∥)`` with ``∫ f d³u = 1``."""
        perp, par = jnp.broadcast_arrays(
            jnp.asarray(u_perp, dtype=jnp.float64), jnp.asarray(u_par, dtype=jnp.float64)
        )
        return jnp.exp(self.log_density(perp, par) - self.log_normalization())

    def density_gradient(
        self, u_perp: ArrayLike, u_par: ArrayLike, /
    ) -> tuple[Array, Array]:
        """``(∂f/∂u⊥, ∂f/∂u∥)`` of the normalized density (exact JVPs of ``log f``)."""
        perp, par = jnp.broadcast_arrays(
            jnp.asarray(u_perp, dtype=jnp.float64), jnp.asarray(u_par, dtype=jnp.float64)
        )
        log_f, d_perp = jax.jvp(
            lambda value: self.log_density(value, par), (perp,), (jnp.ones_like(perp),)
        )
        _, d_par = jax.jvp(
            lambda value: self.log_density(perp, value), (par,), (jnp.ones_like(par),)
        )
        density = jnp.exp(log_f - self.log_normalization())
        inside = jnp.isfinite(log_f)
        return (
            jnp.where(inside, density * d_perp, 0.0),
            jnp.where(inside, density * d_par, 0.0),
        )


def _thermal_tail_bound(exponent: float, /) -> float:
    """Upper bound of the Jüttner mass beyond ``(γ − 1)/θ = T``, uniform in ``θ``."""
    poly = exp(-exponent) * (exponent * exponent + 2.0 * exponent + 2.0)
    return poly / (2.0 - poly)


class ThermalJuttnerDistribution(AbstractGyrotropicDistribution):
    """Isotropic Maxwell–Jüttner ``f ∝ exp(−γ/θ)`` with ``θ = kT/(m c²)``.

    The integration support is ``(γ − 1)/θ ≤ T`` with ``T`` the smallest value for
    which the rigorous bound ``e^{−T}(T² + 2T + 2)/(2 − e^{−T}(T² + 2T + 2))`` on
    the omitted mass is below ``tail_tolerance``.
    """

    __strict_contract__ = True

    temperature: Float64[Scalar]
    tail_exponent: float = eqx.field(static=True)

    def __init__(
        self, temperature: ConvertibleToArray, /, *, tail_tolerance: float = 1.0e-15
    ) -> None:
        theta = _host_positive(float(np.asarray(temperature)), "temperature")
        tolerance = _host_positive(tail_tolerance, "tail_tolerance")
        if tolerance >= 0.5:
            raise ValueError("tail_tolerance must be below 1/2.")
        lower, upper = 1.0, 800.0
        for _ in range(200):
            middle = 0.5 * (lower + upper)
            if _thermal_tail_bound(middle) > tolerance:
                lower = middle
            else:
                upper = middle
        self.temperature = jnp.asarray(theta, dtype=jnp.float64)
        self.tail_exponent = upper

    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        squared = u_perp * u_perp + u_par * u_par
        kinetic = squared / (1.0 + jnp.sqrt(1.0 + squared))
        return -kinetic / self.temperature

    def log_normalization(self) -> Array:
        theta = self.temperature
        return jnp.log(4.0 * pi * theta * kve(2.0, 1.0 / theta))

    def momentum_support(self) -> tuple[Array, Array]:
        kinetic = self.temperature * self.tail_exponent
        return jnp.zeros((), dtype=jnp.float64), jnp.sqrt(kinetic * (kinetic + 2.0))

    def tail_mass_bound(self) -> Array:
        return jnp.asarray(_thermal_tail_bound(self.tail_exponent), dtype=jnp.float64)


class PowerLawDistribution(AbstractGyrotropicDistribution):
    """Isotropic power law ``dN/du ∝ u^{−index}`` on ``[minimum_momentum, maximum_momentum]``."""

    __strict_contract__ = True

    index: Float64[Scalar]
    minimum_momentum: Float64[Scalar]
    maximum_momentum: Float64[Scalar]

    def __init__(
        self,
        index: ConvertibleToArray,
        minimum_momentum: ConvertibleToArray,
        maximum_momentum: ConvertibleToArray,
        /,
    ) -> None:
        delta = float(np.asarray(index))
        lower = _host_positive(float(np.asarray(minimum_momentum)), "minimum_momentum")
        upper = _host_positive(float(np.asarray(maximum_momentum)), "maximum_momentum")
        if not isfinite(delta) or upper <= lower:
            raise ValueError(
                "index must be finite and maximum_momentum above minimum_momentum."
            )
        self.index = jnp.asarray(delta, dtype=jnp.float64)
        self.minimum_momentum = jnp.asarray(lower, dtype=jnp.float64)
        self.maximum_momentum = jnp.asarray(upper, dtype=jnp.float64)

    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        u = jnp.sqrt(u_perp * u_perp + u_par * u_par)
        inside = (u >= self.minimum_momentum) & (u <= self.maximum_momentum)
        safe = jnp.where(inside, u, self.minimum_momentum)
        return jnp.where(inside, -(self.index + 2.0) * jnp.log(safe), -jnp.inf)

    def log_normalization(self) -> Array:
        lower = jnp.log(self.minimum_momentum)
        upper = jnp.log(self.maximum_momentum)
        power = 1.0 - self.index
        flat = jnp.abs(power) < 1.0e-12
        safe_power = jnp.where(flat, 1.0, power)
        # ∫ u^{−δ} du = (e^{(1−δ) ln u₂} − e^{(1−δ) ln u₁})/(1 − δ), via log-sum.
        log_integral = jnp.where(
            flat,
            jnp.log(upper - lower),
            safe_power * lower
            + jnp.log(jnp.expm1(safe_power * (upper - lower)) / safe_power),
        )
        return jnp.log(4.0 * pi) + log_integral

    def momentum_support(self) -> tuple[Array, Array]:
        return self.minimum_momentum, self.maximum_momentum

    def tail_mass_bound(self) -> Array:
        return jnp.zeros((), dtype=jnp.float64)


def _kappa_log_density(u: Array, theta: Array, kappa: Array, /) -> Array:
    kinetic = u * u / (1.0 + jnp.sqrt(1.0 + u * u))
    return -(kappa + 1.0) * jnp.log1p(kinetic / (kappa * theta))


class KappaDistribution(AbstractGyrotropicDistribution):
    """Isotropic relativistic kappa ``f ∝ (1 + (γ − 1)/(κ θ))^{−(κ+1)}`` (Pandya et al. 2016).

    Truncated at ``maximum_momentum``; ``κ > 2`` for a finite density. The
    normalization is integrated once at construction and the omitted mass is
    bounded by ``4π(1 + κθ)² κθ y^{2−κ}/((κ − 2) Z)`` with ``y = 1 + (γ_max − 1)/(κθ)``.
    """

    __strict_contract__ = True

    temperature: Float64[Scalar]
    kappa: Float64[Scalar]
    maximum_momentum: Float64[Scalar]
    normalization: Float64[Scalar]
    tail_bound: Float64[Scalar]

    def __init__(
        self,
        temperature: ConvertibleToArray,
        kappa: ConvertibleToArray,
        maximum_momentum: ConvertibleToArray,
        /,
        *,
        quadrature_order: int = 21,
        quadrature_panels: int = 24,
    ) -> None:
        theta = _host_positive(float(np.asarray(temperature)), "temperature")
        kappa_ = float(np.asarray(kappa))
        upper = _host_positive(float(np.asarray(maximum_momentum)), "maximum_momentum")
        if not isfinite(kappa_) or kappa_ <= 2.0:
            raise ValueError("kappa must be finite and above 2.")
        theta_ = jnp.asarray(theta, dtype=jnp.float64)
        kappa_array = jnp.asarray(kappa_, dtype=jnp.float64)
        core = (kappa_ * theta) * (kappa_ * theta + 2.0)
        scale_u = max(core, 1.0e-300) ** 0.5
        nodes, weights, _ = _composite_rule(quadrature_order, quadrature_panels)
        span = log(1.0 + upper / scale_u)
        # u = s (eᵗ − 1), du = s eᵗ dt on t ∈ [0, ln(1 + u_max/s)].
        t = span * nodes
        u = scale_u * jnp.expm1(t)
        integrand = (
            4.0
            * pi
            * u
            * u
            * jnp.exp(_kappa_log_density(u, theta_, kappa_array))
            * scale_u
            * jnp.exp(t)
        )
        normalization = float(span * jnp.sum(weights * integrand))
        gamma_max = (1.0 + upper * upper) ** 0.5
        y = 1.0 + (gamma_max - 1.0) / (kappa_ * theta)
        tail = (
            4.0
            * pi
            * (1.0 + kappa_ * theta) ** 2
            * kappa_
            * theta
            * y ** (2.0 - kappa_)
            / ((kappa_ - 2.0) * normalization)
        )
        self.temperature = theta_
        self.kappa = kappa_array
        self.maximum_momentum = jnp.asarray(upper, dtype=jnp.float64)
        self.normalization = jnp.asarray(normalization, dtype=jnp.float64)
        self.tail_bound = jnp.asarray(tail, dtype=jnp.float64)

    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        u = jnp.sqrt(u_perp * u_perp + u_par * u_par)
        inside = u <= self.maximum_momentum
        value = _kappa_log_density(
            jnp.where(inside, u, 0.0), self.temperature, self.kappa
        )
        return jnp.where(inside, value, -jnp.inf)

    def log_normalization(self) -> Array:
        return jnp.log(self.normalization)

    def momentum_support(self) -> tuple[Array, Array]:
        return jnp.zeros((), dtype=jnp.float64), self.maximum_momentum

    def tail_mass_bound(self) -> Array:
        return self.tail_bound


def _tabulated_log_density(
    log_momenta: Array,
    pitch_cosines: Array,
    values: Array,
    u_perp: Array,
    u_par: Array,
    /,
) -> Array:
    u = jnp.sqrt(u_perp * u_perp + u_par * u_par)
    positive = u > 0.0
    safe_u = jnp.where(positive, u, jnp.exp(log_momenta[0]))
    log_u = jnp.log(safe_u)
    mu = jnp.where(positive, u_par / safe_u, 0.0)
    inside = (
        positive
        & (log_u >= log_momenta[0])
        & (log_u <= log_momenta[-1])
        & (mu >= pitch_cosines[0])
        & (mu <= pitch_cosines[-1])
    )
    stencil = rectilinear_stencil(
        (log_momenta, pitch_cosines),
        jnp.stack((log_u, mu), axis=-1),
        boundary=("clamp", "clamp"),
    )
    interpolated = apply_gather_stencil(values.reshape((-1,)), stencil).values
    return jnp.where(inside, interpolated, -jnp.inf)


class TabulatedGyrotropicDistribution(AbstractGyrotropicDistribution):
    """Tabulated ``log f(u, cos α)`` on a rectilinear ``(ln u, cos α)`` grid.

    ``log_density`` is bilinear in ``(ln u, cos α)``; ``f = 0`` outside the grid.
    The normalization integrates that interpolant once at construction with a
    Gauss–Kronrod rule per grid cell.
    """

    __strict_contract__ = True

    log_momenta: Float64[_MomentumDim]
    pitch_cosines: Float64[_PitchDim]
    log_values: Float64[_MomentumDim, _PitchDim]
    normalization: Float64[Scalar]

    def __init__(
        self,
        momenta: ConvertibleToArray,
        pitch_cosines: ConvertibleToArray,
        log_density: ConvertibleToArray,
        /,
        *,
        quadrature_order: int = 15,
    ) -> None:
        scope = Scope()
        momentum = as_host_array(
            momenta, HostFloat64[_MomentumDim], "momenta", scope=scope
        )
        cosine = as_host_array(
            pitch_cosines, HostFloat64[_PitchDim], "pitch_cosines", scope=scope
        )
        table = as_host_array(
            log_density, HostFloat64[_MomentumDim, _PitchDim], "log_density", scope=scope
        )
        if not np.all(np.isfinite(momentum)) or np.any(momentum <= 0.0):
            raise ValueError("momenta must be finite and positive.")
        if np.any(np.diff(momentum) <= 0.0) or np.any(np.diff(cosine) <= 0.0):
            raise ValueError("momenta and pitch_cosines must be strictly increasing.")
        if not np.all(np.isfinite(cosine)) or cosine[0] < -1.0 or cosine[-1] > 1.0:
            raise ValueError("pitch_cosines must lie in [-1, 1].")
        if not np.all(np.isfinite(table)):
            raise ValueError("log_density must be finite on the grid.")
        log_momenta = jnp.asarray(np.log(momentum))
        cosines = jnp.asarray(cosine)
        values = jnp.asarray(table)
        nodes, weights, _ = _composite_rule(quadrature_order, 1)
        u_low, u_high = log_momenta[:-1], log_momenta[1:]
        mu_low, mu_high = cosines[:-1], cosines[1:]
        log_u = (u_low[:, None] + (u_high - u_low)[:, None] * nodes[None, :]).reshape(-1)
        log_w = ((u_high - u_low)[:, None] * weights[None, :]).reshape(-1)
        mu = (mu_low[:, None] + (mu_high - mu_low)[:, None] * nodes[None, :]).reshape(-1)
        mu_w = ((mu_high - mu_low)[:, None] * weights[None, :]).reshape(-1)
        u = jnp.exp(log_u)[:, None]
        cos = mu[None, :]
        # Stay strictly inside the grid so bilinear evaluation is not clamped.
        log_f = _tabulated_log_density(
            log_momenta, cosines, values, u * jnp.sqrt(1.0 - cos * cos), u * cos
        )
        # d³u = 2π u³ d(ln u) dμ.
        normalization = jnp.sum(
            2.0 * pi * u**3 * jnp.exp(log_f) * log_w[:, None] * mu_w[None, :]
        )
        self.log_momenta = log_momenta
        self.pitch_cosines = cosines
        self.log_values = values
        self.normalization = normalization

    def log_density(self, u_perp: Array, u_par: Array, /) -> Array:
        return _tabulated_log_density(
            self.log_momenta, self.pitch_cosines, self.log_values, u_perp, u_par
        )

    def log_normalization(self) -> Array:
        return jnp.log(self.normalization)

    def momentum_support(self) -> tuple[Array, Array]:
        return jnp.exp(self.log_momenta[0]), jnp.exp(self.log_momenta[-1])

    def tail_mass_bound(self) -> Array:
        return jnp.zeros((), dtype=jnp.float64)


# ---------------------------------------------------------------------------
# Emission kernels
# ---------------------------------------------------------------------------


class _ModeInput(StrictModule):
    """Per-point, per-root inputs of the emission kernels (dynamic leaves only)."""

    omega: Array
    index: Array
    cosine: Array
    sine: Array
    polarization: Array


class _ModeOutput(StrictModule):
    emission: Array
    absorption: Array
    error: Array
    effective_harmonic: Array
    lowest_harmonic: Array
    highest_harmonic: Array
    harmonic_count: Array


def _projection(
    polarization: Array,
    sense: float,
    u_perp: Array,
    u_par: Array,
    gamma: Array,
    lower: Array,
    center: Array,
    upper: Array,
    /,
) -> Array:
    """``|e*·V_s/c|²`` from ``J_{s−1}``, ``J_s``, ``J_{s+1}``."""
    beta_perp = u_perp / gamma
    beta_par = u_par / gamma
    vx = 0.5 * beta_perp * (lower + upper)
    vy = 0.5 * sense * beta_perp * (lower - upper)
    ex, ey, ez = polarization[0], polarization[1], polarization[2]
    projection = (
        jnp.conj(ex) * vx + jnp.conj(ey) * (1j * vy) + jnp.conj(ez) * beta_par * center
    )
    return jnp.abs(projection) ** 2


def _resonance_interval(
    sy: Array, n_par: Array, gamma_low: Array, gamma_high: Array, /
) -> tuple[Array, Array, Array]:
    """``u∥`` interval of the resonance curve ``γ = sY + N∥u∥`` inside ``γ ∈ [γ₋, γ₊]``."""
    a = n_par * n_par - 1.0
    b_half = sy * n_par
    c = sy * sy - 1.0
    discriminant = sy * sy + a
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    q = -(b_half + jnp.where(b_half >= 0.0, root, -root))
    safe_q = jnp.where(q == 0.0, 1.0, q)
    safe_a = jnp.where(a == 0.0, 1.0, a)
    first = jnp.where(a == 0.0, jnp.where(q >= 0.0, jnp.inf, -jnp.inf), q / safe_a)
    second = jnp.where(q == 0.0, 0.0, c / safe_q)
    ellipse = a < 0.0
    ellipse_low = jnp.minimum(first, second)
    ellipse_high = jnp.maximum(first, second)
    first_physical = jnp.isfinite(first) & (
        sy + n_par * jnp.where(jnp.isfinite(first), first, 0.0) > 0.0
    )
    branch = jnp.where(first_physical, first, second)
    hyperbola_low = jnp.where(n_par > 0.0, branch, -jnp.inf)
    hyperbola_high = jnp.where(n_par > 0.0, jnp.inf, branch)
    curve_low = jnp.where(ellipse, ellipse_low, hyperbola_low)
    curve_high = jnp.where(ellipse, ellipse_high, hyperbola_high)
    safe_n = jnp.where(n_par == 0.0, 1.0, n_par)
    edge_low = (gamma_low - sy) / safe_n
    edge_high = (gamma_high - sy) / safe_n
    window_open = (sy >= gamma_low) & (sy <= gamma_high)
    window_low = jnp.where(
        n_par == 0.0,
        jnp.where(window_open, -jnp.inf, jnp.inf),
        jnp.minimum(edge_low, edge_high),
    )
    window_high = jnp.where(
        n_par == 0.0,
        jnp.where(window_open, jnp.inf, -jnp.inf),
        jnp.maximum(edge_low, edge_high),
    )
    low = jnp.maximum(curve_low, window_low)
    high = jnp.minimum(curve_high, window_high)
    valid = (discriminant >= 0.0) & jnp.isfinite(low) & jnp.isfinite(high) & (high > low)
    return jnp.where(valid, low, 0.0), jnp.where(valid, high, 1.0), valid


def _harmonic_bounds(
    y: Array, n_par: Array, u_low: Array, u_high: Array, /
) -> tuple[Array, Array]:
    """Integer harmonics whose resonance curve meets the momentum shell."""
    magnitude = jnp.abs(n_par)
    subluminal = magnitude < 1.0
    turning = magnitude / jnp.sqrt(
        jnp.where(subluminal, 1.0 - magnitude * magnitude, 1.0)
    )
    extreme = jnp.where(subluminal, jnp.clip(turning, u_low, u_high), u_high)
    minimum = jnp.sqrt(1.0 + extreme * extreme) - magnitude * extreme
    maximum = jnp.sqrt(1.0 + u_high * u_high) + magnitude * u_high
    return jnp.ceil(minimum / y), jnp.floor(maximum / y)


class MagnetobremsstrahlungResult(StrictModule):
    """Mode-resolved emission and absorption with Stokes transfer coefficients.

    Trailing axis ``2`` indexes the cold-plasma roots of ``faraday.wave``; pick one
    with `select`. ``emission`` is ``j_σ`` per volume, unit angular frequency and
    wave-normal steradian; ``absorption`` is ``α_σ`` per length along the wave
    normal. ``stokes_emission`` and ``propagation_matrix`` combine both roots in the
    `FaradayCoefficients` basis (``ê₁`` in the ``k``–``B₀`` plane) with the
    cold-plasma Faraday rotation and conversion ``2 Re ρ``; the matrix uses
    ``d S/ds = j − K S`` with
    ``K = [[α_I, α_Q, α_U, α_V], [α_Q, α_I, ρ_V, −ρ_U], [α_U, −ρ_V, α_I, ρ_Q],
    [α_V, ρ_U, −ρ_Q, α_I]]``. Unsupported roots (see `MagnetobremsstrahlungStatus`)
    carry NaN, and so do the Stokes quantities when either root is unsupported.
    ``harmonic_range`` is ``(lowest, highest)`` resonant harmonic (continuous route:
    the range spanned by the quadrature nodes with non-negligible emission),
    ``harmonic_count`` the evaluated harmonic count (0 on the continuous route),
    ``effective_harmonic`` the emission-weighted harmonic number,
    ``quadrature_error`` the relative Kronrod–Gauss difference, and
    ``tail_mass_bound`` the omitted distribution mass.
    """

    __strict_contract__ = True

    emission: Float64[_BatchDims, Literal[2]]
    absorption: Float64[_BatchDims, Literal[2]]
    stokes_emission: Float64[_BatchDims, Literal[4]]
    propagation_matrix: Float64[_BatchDims, Literal[4], Literal[4]]
    status: Int32[_BatchDims, Literal[2]]
    harmonic_range: Float64[_BatchDims, Literal[2], Literal[2]]
    harmonic_count: Int32[_BatchDims, Literal[2]]
    effective_harmonic: Float64[_BatchDims, Literal[2]]
    quadrature_error: Float64[_BatchDims, Literal[2]]
    tail_mass_bound: Float64[Scalar]
    faraday: FaradayCoefficients
    route: MagnetobremsstrahlungRoute = eqx.field(static=True)

    def select(self, mode: PlasmaWaveMode, values: ArrayLike, /) -> Array:
        """Pick the root labeled ``mode`` from a ``[..., 2]`` root-indexed array."""
        return self.faraday.wave.select(mode, values)

    @property
    def supported(self) -> Array:
        return (self.status & _UNSUPPORTED) == 0


class MagnetobremsstrahlungPlan(StrictModule, NonTrainableState):
    """Magnetobremsstrahlung of a gyrotropic population in a cold magnetized plasma.

    ``plasma`` supplies ``B₀``, the refractive indices ``n_σ(ω, θ)`` and mode
    polarizations; ``distribution`` is the emitting population's normalized
    momentum distribution and ``emitter_density`` its number density per cubic
    length unit of the plasma's scale. The emitters have charge
    ``emitter_charge_number · e`` and mass ``emitter_mass_ratio · m_e``.

    Resources are explicit: ``maximum_harmonics`` is the harmonic capacity of the
    harmonic sum (larger sets report ``HARMONIC_CAPACITY_EXCEEDED``);
    ``quadrature_order`` selects the embedded Gauss–Kronrod rule per resonance
    interval, ``continuous_panels`` the composite panels per axis of the
    continuous route; ``batch_size`` bounds how many ``(ω, θ)`` points are
    evaluated together. ``quadrature_tolerance`` flags larger relative
    Kronrod–Gauss differences, ``maximum_refractive_index`` marks the resonance
    cone, and ``minimum_continuous_harmonic`` flags continuous-route results whose
    emission-weighted harmonic number is lower.
    """

    __strict_contract__ = True

    plasma: ColdPlasmaDielectric
    distribution: AbstractGyrotropicDistribution
    emitter_density: Float64[Scalar]
    route: MagnetobremsstrahlungRoute = eqx.field(static=True)
    emitter_charge_number: float = eqx.field(static=True)
    emitter_mass_ratio: float = eqx.field(static=True)
    quadrature_order: int = eqx.field(static=True)
    maximum_harmonics: int = eqx.field(static=True)
    continuous_panels: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    quadrature_tolerance: float = eqx.field(static=True)
    maximum_refractive_index: float = eqx.field(static=True)
    minimum_continuous_harmonic: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        plasma: ColdPlasmaDielectric,
        distribution: AbstractGyrotropicDistribution,
        /,
        *,
        emitter_density: ConvertibleToArray,
        route: MagnetobremsstrahlungRoute = "harmonic-sum",
        emitter_charge_number: float = -1.0,
        emitter_mass_ratio: float = 1.0,
        quadrature_order: int = 41,
        maximum_harmonics: int = 256,
        continuous_panels: int = 4,
        batch_size: int = 8,
        quadrature_tolerance: float = 1.0e-6,
        maximum_refractive_index: float = 100.0,
        minimum_continuous_harmonic: float = 10.0,
    ) -> None:
        route_ = parse(route, MagnetobremsstrahlungRoute, "route")
        density = float(np.asarray(emitter_density))
        if not isfinite(density) or density < 0.0:
            raise ValueError("emitter_density must be finite and nonnegative.")
        charge = float(emitter_charge_number)
        if not isfinite(charge) or charge == 0.0:
            raise ValueError("emitter_charge_number must be finite and nonzero.")
        mass = _host_positive(emitter_mass_ratio, "emitter_mass_ratio")
        gauss_kronrod_data(quadrature_order)
        counts = (maximum_harmonics, continuous_panels, batch_size)
        if any(not isinstance(count, int) or count < 1 for count in counts):
            raise ValueError(
                "maximum_harmonics, continuous_panels and batch_size must be positive integers."
            )
        tolerance = _host_positive(quadrature_tolerance, "quadrature_tolerance")
        index_limit = _host_positive(maximum_refractive_index, "maximum_refractive_index")
        harmonic_floor = _host_positive(
            minimum_continuous_harmonic, "minimum_continuous_harmonic"
        )
        self.plasma = plasma
        self.distribution = distribution
        self.emitter_density = jnp.asarray(density, dtype=jnp.float64)
        self.route = route_
        self.emitter_charge_number = charge
        self.emitter_mass_ratio = mass
        self.quadrature_order = quadrature_order
        self.maximum_harmonics = maximum_harmonics
        self.continuous_panels = continuous_panels
        self.batch_size = batch_size
        self.quadrature_tolerance = tolerance
        self.maximum_refractive_index = index_limit
        self.minimum_continuous_harmonic = harmonic_floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "magnetobremsstrahlung",
                "plasma": plasma.dielectric_id,
                "distribution": type(distribution).__name__,
                "route": route_,
                "emitter_charge_number": charge,
                "emitter_mass_ratio": mass,
                "quadrature_order": quadrature_order,
                "maximum_harmonics": maximum_harmonics,
                "continuous_panels": continuous_panels,
                "quadrature_tolerance": tolerance,
                "maximum_refractive_index": index_limit,
                "minimum_continuous_harmonic": harmonic_floor,
            }
        )

    @property
    def gyrofrequency(self) -> Array:
        """Nonrelativistic emitter gyrofrequency ``Ω = |q| B₀ / m``."""
        scale = self.plasma.scale
        charge = abs(self.emitter_charge_number) * float(scale.elementary_charge)
        mass = self.emitter_mass_ratio * float(scale.electron_mass)
        return charge * self.plasma.magnetic_field_magnitude / mass

    @property
    def _sense(self) -> float:
        return -1.0 if self.emitter_charge_number > 0.0 else 1.0

    def _harmonic_sum(self, mode: _ModeInput, /) -> _ModeOutput:
        y = self.gyrofrequency / mode.omega
        n_par = mode.index * mode.cosine
        n_perp = mode.index * mode.sine
        u_low, u_high = self.distribution.momentum_support()
        gamma_low = jnp.sqrt(1.0 + u_low * u_low)
        gamma_high = jnp.sqrt(1.0 + u_high * u_high)
        s_low, s_high = _harmonic_bounds(y, n_par, u_low, u_high)
        harmonics = s_low + jnp.arange(self.maximum_harmonics, dtype=jnp.float64)
        active = harmonics <= s_high
        sy = harmonics * y
        low, high, valid = _resonance_interval(sy, n_par, gamma_low, gamma_high)
        nodes, kronrod, gauss = _composite_rule(self.quadrature_order, 1)
        gamma_a = sy + n_par * low
        gamma_b = sy + n_par * high
        delta = jnp.log(jnp.maximum(gamma_b, 1.0) / jnp.maximum(gamma_a, 1.0))
        small = jnp.abs(delta) < 1.0e-10
        safe_delta = jnp.where(small, 1.0, delta)[:, None]
        t = nodes[None, :]
        fraction = jnp.where(
            small[:, None], t, jnp.expm1(t * safe_delta) / jnp.expm1(safe_delta)
        )
        slope = jnp.where(
            small[:, None],
            1.0,
            safe_delta * jnp.exp(t * safe_delta) / jnp.expm1(safe_delta),
        )
        span = (high - low)[:, None]
        u_par = low[:, None] + span * fraction
        gamma = sy[:, None] + n_par * u_par
        u_perp = jnp.sqrt(jnp.maximum(gamma * gamma - 1.0 - u_par * u_par, 0.0))
        argument = n_perp * u_perp / y
        order = harmonics[:, None]
        power = _projection(
            mode.polarization,
            self._sense,
            u_perp,
            u_par,
            gamma,
            *_bessel_triplet(order, argument, _integer_bessel),
        )
        density = self.distribution.density(u_perp, u_par)
        d_perp, d_par = self.distribution.density_gradient(u_perp, u_par)
        positive = u_perp > 0.0
        drive = (
            sy[:, None]
            * jnp.where(positive, d_perp / jnp.where(positive, u_perp, 1.0), 0.0)
            + n_par * d_par
        )
        mask = (active & valid)[:, None]
        measure = jnp.where(mask, gamma * gamma * power * span * slope, 0.0)
        emission_terms = measure * density
        absorption_terms = -measure * drive
        per_harmonic = jnp.sum(emission_terms * kronrod, axis=-1)
        emission = jnp.sum(per_harmonic)
        absorption = jnp.sum(absorption_terms * kronrod)
        emission_error = jnp.abs(emission - jnp.sum(emission_terms * gauss))
        absorption_error = jnp.abs(absorption - jnp.sum(absorption_terms * gauss))
        error = jnp.maximum(
            emission_error / jnp.abs(emission), absorption_error / jnp.abs(absorption)
        )
        effective = jnp.sum(harmonics * per_harmonic) / emission
        count = jnp.maximum(s_high - s_low + 1.0, 0.0)
        return _ModeOutput(
            emission=emission,
            absorption=absorption,
            error=jnp.where(emission > 0.0, error, 0.0),
            effective_harmonic=effective,
            lowest_harmonic=s_low,
            highest_harmonic=s_high,
            harmonic_count=count,
        )

    def _continuous(self, mode: _ModeInput, /) -> _ModeOutput:
        y = self.gyrofrequency / mode.omega
        n_par = mode.index * mode.cosine
        n_perp = mode.index * mode.sine
        u_low, u_high = self.distribution.momentum_support()
        u_start = jnp.maximum(u_low, 1.0e-6 * u_high)
        nodes, kronrod, gauss = _composite_rule(
            self.quadrature_order, self.continuous_panels
        )
        log_span = jnp.log(u_high / u_start)
        u = (u_start * jnp.exp(log_span * nodes))[:, None]
        gamma = jnp.sqrt(1.0 + u * u)
        beta_squared = (u / gamma) ** 2
        cone = jnp.sqrt(
            jnp.maximum(
                jnp.abs(1.0 - mode.index**2 * beta_squared), 1.0 / (gamma * gamma)
            )
        )
        width = jnp.minimum(cone * jnp.maximum(mode.sine, cone), 2.0)
        center = mode.cosine
        t_low = jnp.arcsinh((-1.0 - center) / width)
        t_high = jnp.arcsinh((1.0 - center) / width)
        t = t_low + (t_high - t_low) * nodes[None, :]
        mu = jnp.clip(center + width * jnp.sinh(t), -1.0, 1.0)
        pitch_measure = (t_high - t_low) * width * jnp.cosh(t)
        u_par = u * mu
        u_perp = u * jnp.sqrt(jnp.maximum(1.0 - mu * mu, 0.0))
        sy = gamma - n_par * u_par
        harmonic = sy / y
        active = harmonic >= 1.0
        order = jnp.where(active, harmonic, 1.0)
        argument = n_perp * u_perp / y
        power = _projection(
            mode.polarization,
            self._sense,
            u_perp,
            u_par,
            gamma,
            *_bessel_triplet(order, argument, jv),
        )
        density = self.distribution.density(u_perp, u_par)
        d_perp, d_par = self.distribution.density_gradient(u_perp, u_par)
        # Σ_s ∫ du∥ → ∫ du⊥ du∥ ∂s/∂u⊥ with du⊥ du∥ = u du dμ/sin α and
        # ∂s/∂u⊥ = u sin α/(γY): the measure is u³/(γY) d(ln u) dμ.
        measure = jnp.where(
            active,
            log_span * u * u * pitch_measure * (u / (gamma * y)) * gamma * gamma * power,
            0.0,
        )
        positive = u_perp > 0.0
        drive = (
            sy * jnp.where(positive, d_perp / jnp.where(positive, u_perp, 1.0), 0.0)
            + n_par * d_par
        )
        emission_terms = measure * density
        absorption_terms = -measure * drive
        kk = kronrod[:, None] * kronrod[None, :]
        gg = gauss[:, None] * gauss[None, :]
        emission = jnp.sum(emission_terms * kk)
        absorption = jnp.sum(absorption_terms * kk)
        error = jnp.maximum(
            jnp.abs(emission - jnp.sum(emission_terms * gg)) / jnp.abs(emission),
            jnp.abs(absorption - jnp.sum(absorption_terms * gg)) / jnp.abs(absorption),
        )
        weighted = emission_terms * kk
        effective = jnp.sum(harmonic * weighted) / emission
        significant = active & (weighted > 1.0e-12 * jnp.max(weighted))
        return _ModeOutput(
            emission=emission,
            absorption=absorption,
            error=jnp.where(emission > 0.0, error, 0.0),
            effective_harmonic=effective,
            lowest_harmonic=jnp.min(jnp.where(significant, harmonic, jnp.inf)),
            highest_harmonic=jnp.max(jnp.where(significant, harmonic, -jnp.inf)),
            harmonic_count=jnp.zeros((), dtype=jnp.float64),
        )

    def _kernel(self, mode: _ModeInput, /) -> _ModeOutput:
        match self.route:
            case "harmonic-sum":
                return self._harmonic_sum(mode)
            case "continuous-harmonic":
                return self._continuous(mode)
            case _:
                assert_never(self.route)

    def evaluate(
        self, omega: ArrayLike, theta: ArrayLike, /
    ) -> MagnetobremsstrahlungResult:
        """Coefficients at real float64 angular frequencies ``ω`` and angles ``θ`` to ``B₀``."""
        faraday = self.plasma.faraday_coefficients(omega, theta)
        wave = faraday.wave
        batch = wave.angle.shape
        n_squared = wave.n_squared
        index = wave.refractive_index.real
        c1 = wave.status
        index_limit = self.maximum_refractive_index
        propagating = (n_squared.real > 0.0) & (
            c1 & ColdPlasmaWaveStatus.EVANESCENT.value == 0
        )
        resonant = (c1 & ColdPlasmaWaveStatus.RESONANT.value != 0) | ~(
            jnp.abs(wave.refractive_index) <= index_limit
        )
        undefined = (
            c1
            & (
                ColdPlasmaWaveStatus.POLARIZATION_UNDEFINED
                | ColdPlasmaWaveStatus.ROOT_DEGENERATE
            ).value
            != 0
        )
        anomalous = (self.route == "continuous-harmonic") & (
            jnp.abs(index * jnp.cos(wave.angle)[..., None]) >= 1.0
        )
        usable = propagating & ~resonant & ~undefined & ~anomalous
        safe_index = jnp.where(usable, index, 1.0)
        transverse = faraday.transverse_fraction
        safe_transverse = jnp.where(usable, transverse, 1.0)
        polarization = jnp.where(
            usable[..., None],
            wave.polarization,
            jnp.asarray((0.0, 1.0, 0.0), dtype=jnp.complex128),
        )
        omega_b = jnp.broadcast_to(wave.angular_frequency[..., None], batch + (2,))
        cosine = jnp.broadcast_to(jnp.cos(wave.angle)[..., None], batch + (2,))
        sine = jnp.broadcast_to(jnp.sin(wave.angle)[..., None], batch + (2,))
        count = int(np.prod(batch, dtype=np.int64)) * 2
        modes = _ModeInput(
            omega=omega_b.reshape((count,)),
            index=safe_index.reshape((count,)),
            cosine=cosine.reshape((count,)),
            sine=sine.reshape((count,)),
            polarization=polarization.reshape((count, 3)),
        )
        outputs = jax.lax.map(self._kernel, modes, batch_size=self.batch_size)
        shape = batch + (2,)
        sums = jax.tree_util.tree_map(lambda value: value.reshape(shape), outputs)
        scale = self.plasma.scale
        q = abs(self.emitter_charge_number) * float(scale.elementary_charge)
        mass = self.emitter_mass_ratio * float(scale.electron_mass)
        epsilon = float(scale.vacuum_permittivity)
        light = float(scale.speed_of_light)
        density = self.emitter_density
        omega_r = omega_b
        emission = (
            (q * q * omega_r * safe_index * density / (4.0 * pi * epsilon * light))
            * sums.emission
            / safe_transverse
        )
        absorption = (
            (
                2.0
                * pi
                * pi
                * q
                * q
                * density
                / (epsilon * omega_r * safe_index * mass * light)
            )
            * sums.absorption
            / safe_transverse
        )
        collisional = jnp.abs(n_squared.imag) > 0.0
        capacity = sums.harmonic_count > self.maximum_harmonics
        unresolved = ~(sums.error <= self.quadrature_tolerance)
        low_harmonic = (
            (self.route == "continuous-harmonic")
            & ~(sums.effective_harmonic >= self.minimum_continuous_harmonic)
            & (sums.emission > 0.0)
        )
        status = MagnetobremsstrahlungStatus
        flags = (
            (~propagating).astype(jnp.int32) * status.EVANESCENT.value
            | resonant.astype(jnp.int32) * status.RESONANCE_CONE.value
            | undefined.astype(jnp.int32) * status.POLARIZATION_UNDEFINED.value
            | capacity.astype(jnp.int32) * status.HARMONIC_CAPACITY_EXCEEDED.value
            | anomalous.astype(jnp.int32) * status.ANOMALOUS_DOPPLER.value
            | (unresolved & usable).astype(jnp.int32) * status.QUADRATURE_UNRESOLVED.value
            | (low_harmonic & usable).astype(jnp.int32) * status.LOW_HARMONIC.value
            | collisional.astype(jnp.int32) * status.COLLISIONAL_MEDIUM.value
        )
        finite = jnp.isfinite(emission) & jnp.isfinite(absorption)
        flags = flags | ((~finite) & usable & ~capacity).astype(jnp.int32) * (
            status.NONFINITE.value
        )
        supported = (flags & _UNSUPPORTED) == 0
        nan = jnp.asarray(jnp.nan, dtype=jnp.float64)
        emission = jnp.where(supported, emission, nan)
        absorption = jnp.where(supported, absorption, nan)
        mode_stokes = faraday.mode_stokes
        stokes_emission = jnp.concatenate(
            (
                jnp.sum(emission, axis=-1)[..., None],
                jnp.sum(emission[..., None] * mode_stokes, axis=-2),
            ),
            axis=-1,
        )
        absorption_i = 0.5 * jnp.sum(absorption, axis=-1)
        a_q, a_u, a_v = jnp.moveaxis(
            0.5 * jnp.sum(absorption[..., None] * mode_stokes, axis=-2), -1, 0
        )
        r_q, r_u, r_v = jnp.moveaxis(2.0 * faraday.coefficients.real, -1, 0)
        propagation = jnp.stack(
            (
                jnp.stack((absorption_i, a_q, a_u, a_v), axis=-1),
                jnp.stack((a_q, absorption_i, r_v, -r_u), axis=-1),
                jnp.stack((a_u, -r_v, absorption_i, r_q), axis=-1),
                jnp.stack((a_v, r_u, -r_q, absorption_i), axis=-1),
            ),
            axis=-2,
        )
        both = jnp.all(supported, axis=-1)
        stokes_emission = jnp.where(both[..., None], stokes_emission, nan)
        propagation = jnp.where(both[..., None, None], propagation, nan)
        return MagnetobremsstrahlungResult(
            emission=emission,
            absorption=absorption,
            stokes_emission=stokes_emission,
            propagation_matrix=propagation,
            status=flags,
            harmonic_range=jnp.stack(
                (sums.lowest_harmonic, sums.highest_harmonic), axis=-1
            ),
            harmonic_count=sums.harmonic_count.astype(jnp.int32),
            effective_harmonic=sums.effective_harmonic,
            quadrature_error=sums.error,
            tail_mass_bound=self.distribution.tail_mass_bound(),
            faraday=faraday,
            route=self.route,
        )


__all__ = [
    "AbstractGyrotropicDistribution",
    "KappaDistribution",
    "MagnetobremsstrahlungPlan",
    "MagnetobremsstrahlungResult",
    "MagnetobremsstrahlungRoute",
    "MagnetobremsstrahlungStatus",
    "PowerLawDistribution",
    "TabulatedGyrotropicDistribution",
    "ThermalJuttnerDistribution",
]
