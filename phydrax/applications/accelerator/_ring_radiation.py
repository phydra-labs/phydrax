#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Storage-ring synchrotron radiation: integrals, equilibrium, radiative tracking.

Conventions
-----------
- A ``RingLattice`` is one periodic cell of ``RingElement`` values repeated
  ``periodicity`` times; the bends of the complete ring must close
  (``Σ h L = 2π``). Bends are horizontal normal-entry sector bends with an
  optional gradient; edge angles, vertical or skew bends, and nonlinear
  multipoles cannot be expressed.
- Linear optics use the canonical coordinates ``(x, px, y, py, z, δ)`` with
  ``z = −β₀ c Δt`` (positive early), conjugate to ``δ = Δp/p₀``. Bunch
  coordinates follow the bound ``AcceleratorConvention``: ``ζ = −σ z`` with
  ``σ = +1`` for ``"positive-late"`` and ``−1`` for ``"positive-early"``.
- Hill equations: ``x'' + (h² + k₁) x = h δ`` and ``y'' − k₁ y = 0`` with
  curvature ``h = 1/ρ`` and normalized gradient ``k₁ = (∂B_y/∂x)/(B ρ)``;
  path lengthening advances ``z`` by ``−h x`` per unit length and velocity
  slip by ``δ/γ₀²``.
- Radiation integrals (Sands 1970): ``I₁ = ∮ η h``, ``I₂ = ∮ h²``,
  ``I₃ = ∮ |h|³``, ``I₄ = ∮ η h (h² + 2 k₁)``, ``I₅ = ∮ |h|³ ℋ`` with
  ``ℋ = γ_x η² + 2 α_x η η' + β_x η'²``. Damping partition
  ``J = (1 − I₄/I₂, 1, 2 + I₄/I₂)``, energy loss per turn
  ``U₀ = (2/3) r E_rest γ₀⁴ I₂`` with ``r = q²/(4π ε₀ E_rest)``, damping times
  ``τᵢ = 2 E₀ T₀ / (Jᵢ U₀)``, equilibrium energy spread
  ``σ_δ² = C_q γ₀² I₃/(J_s I₂)`` and emittance ``ε_x = C_q γ₀² I₅/(J_x I₂)`` with
  ``C_q = 55 ħ c / (32 √3 E_rest)``.
- Radiation is ultrarelativistic and classical: the reference Lorentz factor
  must exceed ``minimum_lorentz_factor`` and the quantum parameter
  ``χ = ħ c γ₀² |h| / E_rest`` must stay below ``maximum_quantum_parameter``.
  Photons are emitted along the particle direction (no vertical opening-angle
  excitation), so the vertical equilibrium emittance of an uncoupled ring is
  zero.
- A particle at ``(x, δ)`` in a bend slice of length ``L`` radiates the mean
  energy ``(2/3) q² γ₀⁴ (1 + δ)² (h + k₁ x)² (1 + h x) L / (4π ε₀)``. Stochastic
  emission draws a Poisson photon count of mean
  ``(5/(2√3)) α_q γ₀ |h + k₁ x| (1 + h x) L`` with ``α_q = q²/(4π ε₀ ħ c)`` and
  photon energies ``u = ξ u_c``, ``u_c = (3/2) ħ c γ₀³ (1 + δ)² |h + k₁ x|``,
  where ``ξ`` follows the classical photon-number spectrum ``F(ξ)/ξ`` of
  ``phydrax.special.synchrotron_f``. Emission removes momentum ``u/c`` along
  the direction of motion: ``δ → δ − u/(p₀ c)`` and ``(px, py)`` scale by
  ``1 − u/(p₀ c (1 + δ))``.
- RF cavities are thin; a particle gains ``|q| V sin(φ_s − k_rf z)`` with
  ``k_rf = 2π h_rf / C``. The synchronous phase restores the model's mean
  loss per turn on the stable side of transition.
- Every quantity is in the bound ``ElectromagneticScaleContract`` units: lengths
  in its length unit, energies in its energy unit, charge in its charge unit,
  and cavity voltages as energy per charge.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from enum import IntFlag
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics import gauss_kronrod_data, gauss_legendre_data
from ..._physical import ElectromagneticScaleContract
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import (
    canonical_identifier,
    finite_real_scalar,
    positive_finite_float,
    positive_integer,
)
from ...linalg import determinant_small_linear, SmallLinearSolvePlan, solve_small_linear
from ...special import synchrotron_f
from ...typing import Bool, checked, Dim, Float64, Int32, parse, PRNGKey, Scalar, Size
from ._advanced import _symplectic_form, SymplecticMapPlan
from ._beam import _late_sign, AcceleratorBunch, AcceleratorConvention


RingElementKind: TypeAlias = Literal["drift", "quadrupole", "sector-bend", "rf-cavity"]
RingRadiationModel: TypeAlias = Literal["none", "classical", "stochastic"]

_SQRT_THREE = math.sqrt(3.0)
# Mellin moments of K_{5/3}: ∫₀^∞ F(ξ)/ξ dξ = 5π/3, ⟨ξ⟩ = 8/(15√3), ⟨ξ²⟩ = 11/27.
_PHOTON_NUMBER_INTEGRAL = 5.0 * math.pi / 3.0
_MEAN_FRACTION = 8.0 / (15.0 * _SQRT_THREE)
_SECOND_MOMENT = 11.0 / 27.0
# F(ξ) → 4π/(√3 Γ(1/3)) (ξ/2)^{1/3}, so the density 3 F(t³)/t in t = ξ^{1/3}
# tends to 4√3 π / (2^{1/3} Γ(1/3)) at t = 0.
_ORIGIN_DENSITY = (
    4.0 * _SQRT_THREE * math.pi / (2.0 ** (1.0 / 3.0) * math.gamma(1.0 / 3.0))
)
_PHOTONS_PER_RADIAN = 5.0 / (2.0 * _SQRT_THREE)
_QUANTUM_EXCITATION = 55.0 / (32.0 * _SQRT_THREE)
_SPECTRUM_PANELS = 64
_SPECTRUM_PANEL_ORDER = 16
_SERIES_PHASE = 0.5
_STABILITY_MARGIN = 1.0e-12
_MAXIMUM_PHOTON_CAPACITY = 256
_MOMENT_CHANNELS = 6 + 36 + 5

_STEP_LINEAR = 0
_STEP_RADIATION = 1
_STEP_RF = 2


class _ElementDim(Dim, minimum=1):
    """Elements of one lattice cell."""


class _BoundaryDim(Dim, minimum=2):
    """Element boundaries of one lattice cell."""


class _SpectrumNodeDim(Dim, minimum=3):
    """Photon-spectrum table nodes."""


class _StepDim(Dim, minimum=1):
    """Tracking steps of one lattice cell."""


class _TurnDim(Dim, minimum=1):
    """Tracked turns."""


class _ParticleDim(Dim, minimum=1):
    """Bunch capacity."""


class RingOpticsError(ValueError):
    """The lattice or its optics is outside the supported radiation contract."""


class RadiativeRingTrackingResourceError(ValueError):
    """Radiative ring tracking exceeds its declared memory budget."""


class RadiativeRingTrackingStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    PHOTON_OVERFLOW = 2
    PARTICLE_LOSS = 4


# --------------------------------------------------------------------------- #
# Lattice description.
# --------------------------------------------------------------------------- #


class RingElement(StrictModule):
    """One element of a storage-ring cell.

    ``"drift"`` needs a positive ``length``; ``"quadrupole"`` a positive
    ``length`` and nonzero ``gradient`` ``k₁``; ``"sector-bend"`` a positive
    ``length`` and nonzero ``curvature`` ``h`` with optional ``gradient``;
    ``"rf-cavity"`` is thin with positive peak ``voltage`` and ``harmonic``
    number. Parameters that do not belong to the kind must stay zero.
    """

    kind: RingElementKind = eqx.field(static=True)
    element_id: str = eqx.field(static=True)
    length: float = eqx.field(static=True)
    curvature: float = eqx.field(static=True)
    gradient: float = eqx.field(static=True)
    voltage: float = eqx.field(static=True)
    harmonic: int = eqx.field(static=True)

    def __init__(
        self,
        kind: RingElementKind,
        element_id: str,
        /,
        *,
        length: float = 0.0,
        curvature: float = 0.0,
        gradient: float = 0.0,
        voltage: float = 0.0,
        harmonic: int = 0,
    ) -> None:
        kind_ = parse(kind, RingElementKind, "kind")
        identifier = canonical_identifier(element_id, "element_id")
        length_ = finite_real_scalar(length, "length")
        curvature_ = finite_real_scalar(curvature, "curvature")
        gradient_ = finite_real_scalar(gradient, "gradient")
        voltage_ = finite_real_scalar(voltage, "voltage")
        if isinstance(harmonic, bool) or not isinstance(harmonic, int):
            raise TypeError("harmonic must be an integer.")
        match kind_:
            case "drift":
                valid = length_ > 0.0 and curvature_ == gradient_ == voltage_ == 0.0
                valid = valid and harmonic == 0
            case "quadrupole":
                valid = length_ > 0.0 and gradient_ != 0.0 and curvature_ == 0.0
                valid = valid and voltage_ == 0.0 and harmonic == 0
            case "sector-bend":
                valid = length_ > 0.0 and curvature_ != 0.0 and voltage_ == 0.0
                valid = valid and harmonic == 0
            case "rf-cavity":
                valid = length_ == 0.0 and curvature_ == gradient_ == 0.0
                valid = valid and voltage_ > 0.0 and harmonic >= 1
            case _:
                assert_never(kind_)
        if not valid:
            raise ValueError(
                f"Ring element {identifier!r} has parameters outside the {kind_!r} contract."
            )
        self.kind = kind_
        self.element_id = identifier
        self.length = length_
        self.curvature = curvature_
        self.gradient = gradient_
        self.voltage = voltage_
        self.harmonic = harmonic


class RingLattice(StrictModule):
    """One periodic cell of a closed storage ring.

    The ring repeats the cell ``periodicity`` times and must close: the total
    bend angle ``periodicity · Σ h L`` equals ``2π`` within
    ``closure_tolerance`` radians. A lattice without bends is refused.
    """

    elements: tuple[RingElement, ...] = eqx.field(static=True)
    periodicity: int = eqx.field(static=True)
    cell_length: float = eqx.field(static=True)
    circumference: float = eqx.field(static=True)
    lattice_id: str = eqx.field(static=True)

    def __init__(
        self,
        elements: Sequence[RingElement],
        /,
        *,
        periodicity: int = 1,
        closure_tolerance: float = 1.0e-9,
    ) -> None:
        elements_ = tuple(elements)
        if not elements_ or not all(
            isinstance(value, RingElement) for value in elements_
        ):
            raise TypeError("elements must be a non-empty sequence of RingElement.")
        identifiers = [value.element_id for value in elements_]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Ring element identities must be unique within the cell.")
        periods = positive_integer(periodicity, "periodicity")
        tolerance = positive_finite_float(closure_tolerance, "closure_tolerance")
        bends = [value for value in elements_ if value.kind == "sector-bend"]
        if not bends:
            raise RingOpticsError("A radiating ring needs at least one sector bend.")
        angle = periods * math.fsum(value.curvature * value.length for value in bends)
        if abs(angle - 2.0 * math.pi) > tolerance:
            raise RingOpticsError(
                f"Ring bends do not close: total angle {angle!r} rad differs from 2π."
            )
        cell_length = math.fsum(value.length for value in elements_)
        self.elements = elements_
        self.periodicity = periods
        self.cell_length = cell_length
        self.circumference = periods * cell_length
        self.lattice_id = canonical_fingerprint(
            {
                "kind": "accelerator-ring-lattice",
                "elements": [
                    [
                        value.kind,
                        value.element_id,
                        value.length,
                        value.curvature,
                        value.gradient,
                        value.voltage,
                        value.harmonic,
                    ]
                    for value in elements_
                ],
                "periodicity": periods,
            }
        )


# --------------------------------------------------------------------------- #
# Host linear optics.
# --------------------------------------------------------------------------- #


def _phase_minus_sine(phase: float, circular: bool, /) -> float:
    """``φ − sin φ`` (circular) or ``sinh φ − φ`` without cancellation."""
    if phase < _SERIES_PHASE:
        term = phase**3 / 6.0
        total = 0.0
        sign = -1.0 if circular else 1.0
        for order in range(3, 23, 2):
            total += term
            term *= sign * phase * phase / ((order + 1) * (order + 2))
        return total
    return phase - math.sin(phase) if circular else math.sinh(phase) - phase


def _hill_functions(
    strength: float, length: float, /
) -> tuple[float, float, float, float]:
    """``C``, ``S``, ``∫S`` and ``∫∫S`` of ``u'' + K u = 0`` over ``length``."""
    if strength > 0.0:
        root = math.sqrt(strength)
        phase = root * length
        return (
            math.cos(phase),
            math.sin(phase) / root,
            2.0 * math.sin(0.5 * phase) ** 2 / strength,
            _phase_minus_sine(phase, True) / (strength * root),
        )
    if strength < 0.0:
        root = math.sqrt(-strength)
        phase = root * length
        return (
            math.cosh(phase),
            math.sinh(phase) / root,
            2.0 * math.sinh(0.5 * phase) ** 2 / (-strength),
            _phase_minus_sine(phase, False) / (-strength * root),
        )
    return 1.0, length, 0.5 * length * length, length**3 / 6.0


def _element_matrix(
    curvature: float, gradient: float, length: float, inverse_gamma_sq: float, /
) -> np.ndarray:
    """First-order canonical transfer matrix of a sector bend, quadrupole, or drift."""
    horizontal = curvature * curvature + gradient
    cx, sx, first, second = _hill_functions(horizontal, length)
    cy, sy, _, _ = _hill_functions(-gradient, length)
    matrix = np.eye(6, dtype=np.float64)
    matrix[0, 0] = cx
    matrix[0, 1] = sx
    matrix[1, 0] = -horizontal * sx
    matrix[1, 1] = cx
    matrix[0, 5] = curvature * first
    matrix[1, 5] = curvature * sx
    matrix[2, 2] = cy
    matrix[2, 3] = sy
    matrix[3, 2] = gradient * sy
    matrix[3, 3] = cy
    matrix[4, 0] = -curvature * sx
    matrix[4, 1] = -curvature * first
    matrix[4, 5] = -curvature * curvature * second + length * inverse_gamma_sq
    return matrix


def _periodic_twiss(block: np.ndarray, plane: str, /) -> tuple[float, float]:
    cosine = 0.5 * (block[0, 0] + block[1, 1])
    if not abs(cosine) < 1.0 - _STABILITY_MARGIN:
        raise RingOpticsError(
            f"The {plane} cell optics are unstable or on an integer/half-integer "
            f"resonance: cos μ = {cosine!r}."
        )
    sine = math.copysign(math.sqrt(1.0 - cosine * cosine), block[0, 1])
    return block[0, 1] / sine, (block[0, 0] - block[1, 1]) / (2.0 * sine)


def _twiss_matrix(beta: float, alpha: float, /) -> np.ndarray:
    return np.asarray(
        [[beta, -alpha], [-alpha, (1.0 + alpha * alpha) / beta]], dtype=np.float64
    )


def _phase_advance(block: np.ndarray, beta: float, alpha: float, /) -> float:
    return math.atan2(block[0, 1], beta * block[0, 0] - alpha * block[0, 1])


class RingOptics(StrictModule):
    """Periodic linear optics of one cell at its element boundaries.

    ``positions`` holds the ``E + 1`` boundary coordinates ``s``; the Twiss and
    dispersion arrays hold the periodic solution there. Tunes and momentum
    compaction refer to the complete ring; ``slip_factor`` is
    ``α_c − 1/γ₀²``. ``cell_matrix`` is the canonical first-order cell map
    without RF.
    """

    __strict_contract__ = True

    positions: Float64[_BoundaryDim]
    beta_x: Float64[_BoundaryDim]
    alpha_x: Float64[_BoundaryDim]
    beta_y: Float64[_BoundaryDim]
    alpha_y: Float64[_BoundaryDim]
    dispersion: Float64[_BoundaryDim]
    dispersion_slope: Float64[_BoundaryDim]
    horizontal_tune: Float64[Scalar]
    vertical_tune: Float64[Scalar]
    momentum_compaction: Float64[Scalar]
    slip_factor: Float64[Scalar]
    dispersion_condition: Float64[Scalar]
    cell_matrix: Float64[Literal[6], Literal[6]]
    symplectic_residual: Float64[Scalar]


class RingRadiationIntegrals(StrictModule):
    """Synchrotron radiation integrals of the complete ring.

    ``element_contributions`` holds ``(I₁, …, I₅)`` of each cell element; the
    ring values are ``periodicity`` times their sum. Bends are integrated with
    the 15-point Gauss–Kronrod rule on the exact in-bend optics, and
    ``quadrature_error`` is the ring-total Kronrod–Gauss difference.
    """

    __strict_contract__ = True

    i1: Float64[Scalar]
    i2: Float64[Scalar]
    i3: Float64[Scalar]
    i4: Float64[Scalar]
    i5: Float64[Scalar]
    element_contributions: Float64[_ElementDim, Literal[5]]
    quadrature_error: Float64[Literal[5]]


class RingRadiationEquilibrium(StrictModule):
    """Radiation damping and quantum-excitation equilibrium of the reference beam.

    ``damping_partition``, ``damping_times`` (scale time unit), and
    ``damping_turns`` are ordered ``(x, y, s)``. ``critical_energy`` and
    ``quantum_parameter`` refer to the strongest bend; ``photons_per_turn`` is
    the mean emitted photon count per particle and turn.
    """

    __strict_contract__ = True

    energy_loss_per_turn: Float64[Scalar]
    revolution_period: Float64[Scalar]
    damping_partition: Float64[Literal[3]]
    damping_times: Float64[Literal[3]]
    damping_turns: Float64[Literal[3]]
    horizontal_emittance: Float64[Scalar]
    energy_spread: Float64[Scalar]
    photons_per_turn: Float64[Scalar]
    critical_energy: Float64[Scalar]
    quantum_parameter: Float64[Scalar]


# --------------------------------------------------------------------------- #
# Classical photon spectrum.
# --------------------------------------------------------------------------- #


def _spectrum_density(nodes: np.ndarray, /) -> np.ndarray:
    """Photon-number density ``3 F(t³)/t`` in ``t = ξ^{1/3}`` (unnormalized)."""
    interior = nodes > 0.0
    safe = np.where(interior, nodes, 1.0)
    values = np.asarray(synchrotron_f(jnp.asarray(safe**3, dtype=jnp.float64)))
    return np.where(interior, 3.0 * values / safe, _ORIGIN_DENSITY)


class SynchrotronPhotonSpectrum(StrictModule, NonTrainableState):
    """Classical synchrotron photon-number spectrum ``F(ξ)/ξ`` for sampling.

    ``ξ = u/u_c`` is the photon energy in units of the critical energy. The
    density is tabulated in ``t = ξ^{1/3}`` (where it is smooth and finite at
    the origin) on ``node_count`` uniform nodes over ``[0, maximum_fraction^{1/3}]``
    from ``phydrax.special.synchrotron_f`` and sampled by the exact inverse of
    its piecewise-linear interpolant. ``table_*`` are the exact moments of the
    sampled distribution; ``quadrature_*`` integrate ``F`` directly with
    composite Gauss–Legendre panels; ``tail_bound`` bounds the omitted photon
    fraction beyond ``maximum_fraction``.
    """

    __strict_contract__ = True

    nodes: Float64[_SpectrumNodeDim]
    densities: Float64[_SpectrumNodeDim]
    cdf: Float64[_SpectrumNodeDim]
    table_mean_fraction: float = eqx.field(static=True)
    table_second_moment: float = eqx.field(static=True)
    quadrature_photon_number: float = eqx.field(static=True)
    quadrature_mean_fraction: float = eqx.field(static=True)
    quadrature_second_moment: float = eqx.field(static=True)
    tail_bound: float = eqx.field(static=True)
    maximum_fraction: float = eqx.field(static=True)
    node_count: Size[_SpectrumNodeDim] = eqx.field(static=True)
    spectrum_id: str = eqx.field(static=True)

    def __init__(self, *, node_count: int = 4097, maximum_fraction: float = 50.0) -> None:
        count = positive_integer(node_count, "node_count")
        if count < 3:
            raise ValueError("node_count must be at least three.")
        maximum = positive_finite_float(maximum_fraction, "maximum_fraction")
        if maximum < 20.0:
            raise ValueError("maximum_fraction must be at least 20 critical energies.")
        upper = maximum ** (1.0 / 3.0)
        nodes = np.linspace(0.0, upper, count, dtype=np.float64)
        raw = _spectrum_density(nodes)
        widths = np.diff(nodes)
        cumulative = np.concatenate(
            ([0.0], np.cumsum(0.5 * widths * (raw[:-1] + raw[1:])))
        )
        total = cumulative[-1]
        densities = raw / total
        cdf = cumulative / total
        rule = gauss_legendre_data(5)
        points = np.asarray(rule.nodes)
        weights = np.asarray(rule.weights)
        # The piecewise-linear density times t³ or t⁶ is a polynomial of degree
        # at most seven per segment, integrated exactly by five Gauss points.
        abscissae = nodes[:-1, None] + 0.5 * widths[:, None] * (points[None, :] + 1.0)
        fraction = (points[None, :] + 1.0) * 0.5
        local = densities[:-1, None] + fraction * (
            densities[1:, None] - densities[:-1, None]
        )
        scaled = 0.5 * widths[:, None] * weights[None, :] * local
        table_mean = float(np.sum(scaled * abscissae**3))
        table_second = float(np.sum(scaled * abscissae**6))
        panel = gauss_legendre_data(_SPECTRUM_PANEL_ORDER)
        edges = np.linspace(0.0, upper, _SPECTRUM_PANELS + 1, dtype=np.float64)
        panel_width = np.diff(edges)
        panel_points = edges[:-1, None] + 0.5 * panel_width[:, None] * (
            np.asarray(panel.nodes)[None, :] + 1.0
        )
        panel_weights = 0.5 * panel_width[:, None] * np.asarray(panel.weights)[None, :]
        density = _spectrum_density(panel_points)
        number = float(np.sum(panel_weights * density))
        mean = float(np.sum(panel_weights * density * panel_points**3)) / number
        second = float(np.sum(panel_weights * density * panel_points**6)) / number
        # F(ξ) ≤ √(πξ/2) e^{-ξ} (1 + 55/(72ξ)) for ξ ≥ 1, so ∫_Ξ^∞ F/ξ dξ is bounded by
        # √(π/(2Ξ)) e^{-Ξ} (1 + 55/(72Ξ)) relative to 5π/3.
        tail = (
            math.sqrt(math.pi / (2.0 * maximum))
            * math.exp(-maximum)
            * (1.0 + 55.0 / (72.0 * maximum))
            / _PHOTON_NUMBER_INTEGRAL
        )
        self.nodes = jnp.asarray(nodes)
        self.densities = jnp.asarray(densities)
        self.cdf = jnp.asarray(cdf)
        self.table_mean_fraction = table_mean
        self.table_second_moment = table_second
        self.quadrature_photon_number = number
        self.quadrature_mean_fraction = mean
        self.quadrature_second_moment = second
        self.tail_bound = tail
        self.maximum_fraction = maximum
        self.node_count = count
        self.spectrum_id = canonical_fingerprint(
            {
                "kind": "classical-synchrotron-photon-spectrum",
                "table": array_tree_fingerprint((nodes, densities, cdf)),
            }
        )

    def sample_fractions(self, uniforms: ArrayLike, /) -> Array:
        """Photon energies ``u/u_c`` for uniforms in ``[0, 1]`` (any shape)."""
        values = jnp.asarray(uniforms, dtype=jnp.float64)
        return _sample_fractions(self.nodes, self.densities, self.cdf, values)


def _sample_fractions(
    nodes: Array, densities: Array, cdf: Array, uniforms: Array
) -> Array:
    """Exact inverse of the piecewise-linear density CDF, returned as ``ξ = t³``.

    Within a segment the density ``f₀ + s τ`` integrates to ``f₀ τ + s τ²/2``;
    the root uses the cancellation-free form ``2 r / (f₀ + √(f₀² + 2 s r))``.
    """
    last = nodes.shape[0] - 2
    index = jnp.clip(jnp.searchsorted(cdf, uniforms, side="right") - 1, 0, last)
    start = nodes[index]
    width = nodes[index + 1] - start
    first = densities[index]
    slope = (densities[index + 1] - first) / width
    residual = jnp.maximum(uniforms - cdf[index], 0.0)
    root = jnp.sqrt(jnp.maximum(first * first + 2.0 * slope * residual, 0.0))
    denominator = first + root
    offset = jnp.where(denominator > 0.0, 2.0 * residual / denominator, 0.0)
    t = start + jnp.clip(offset, 0.0, width)
    return t * t * t


# --------------------------------------------------------------------------- #
# Radiation plan.
# --------------------------------------------------------------------------- #


class _Reference(NamedTuple):
    rest_energy: float
    momentum_energy: float
    charge: float
    gamma: float
    beta: float


class _OpticsSolution(NamedTuple):
    optics: RingOptics
    boundary_dispersion: np.ndarray
    boundary_twiss: np.ndarray


def _optics(
    lattice: RingLattice, inverse_gamma_sq: float, /
) -> tuple[_OpticsSolution, list[np.ndarray]]:
    matrices = [
        _element_matrix(value.curvature, value.gradient, value.length, inverse_gamma_sq)
        for value in lattice.elements
    ]
    cell = np.eye(6, dtype=np.float64)
    for matrix in matrices:
        cell = matrix @ cell
    form = _symplectic_form(-1.0)
    residual = max(float(np.max(np.abs(m.T @ form @ m - form))) for m in matrices)
    beta_x, alpha_x = _periodic_twiss(cell[0:2, 0:2], "horizontal")
    beta_y, alpha_y = _periodic_twiss(cell[2:4, 2:4], "vertical")
    # Solve (I − M) η = m in the dimensionless variables (η/L, η') so the
    # conditioning reflects the optics, not the length unit.
    unit = np.asarray([lattice.cell_length, 1.0], dtype=np.float64)
    periodic = solve_small_linear(
        SmallLinearSolvePlan(2),
        jnp.asarray((np.eye(2) - cell[0:2, 0:2]) * unit[None, :] / unit[:, None]),
        jnp.asarray(cell[0:2, 5] / unit),
    )
    if not bool(periodic.successful):
        raise RingOpticsError("The periodic dispersion solve failed.")
    dispersion = unit * np.asarray(periodic.value, dtype=np.float64)
    count = len(matrices)
    boundary_dispersion = np.zeros((count + 1, 2), dtype=np.float64)
    boundary_twiss = np.zeros((count + 1, 4), dtype=np.float64)
    positions = np.zeros((count + 1,), dtype=np.float64)
    boundary_dispersion[0] = dispersion
    boundary_twiss[0] = (beta_x, alpha_x, beta_y, alpha_y)
    phase_x = 0.0
    phase_y = 0.0
    twiss_x = _twiss_matrix(beta_x, alpha_x)
    twiss_y = _twiss_matrix(beta_y, alpha_y)
    for index, (element, matrix) in enumerate(
        zip(lattice.elements, matrices, strict=True)
    ):
        bx, ax, by, ay = boundary_twiss[index]
        phase_x += _phase_advance(matrix[0:2, 0:2], bx, ax)
        phase_y += _phase_advance(matrix[2:4, 2:4], by, ay)
        twiss_x = matrix[0:2, 0:2] @ twiss_x @ matrix[0:2, 0:2].T
        twiss_y = matrix[2:4, 2:4] @ twiss_y @ matrix[2:4, 2:4].T
        boundary_twiss[index + 1] = (
            twiss_x[0, 0],
            -twiss_x[0, 1],
            twiss_y[0, 0],
            -twiss_y[0, 1],
        )
        boundary_dispersion[index + 1] = (
            matrix[0:2, 0:2] @ boundary_dispersion[index] + matrix[0:2, 5]
        )
        positions[index + 1] = positions[index] + element.length
    # Path lengthening of the periodic dispersive orbit x = η δ over one cell.
    lengthening = cell[4, 0] * dispersion[0] + cell[4, 1] * dispersion[1] + cell[4, 5]
    compaction = -lattice.periodicity * (
        lengthening - lattice.cell_length * inverse_gamma_sq
    )
    compaction /= lattice.circumference
    optics = RingOptics(
        jnp.asarray(positions),
        jnp.asarray(boundary_twiss[:, 0]),
        jnp.asarray(boundary_twiss[:, 1]),
        jnp.asarray(boundary_twiss[:, 2]),
        jnp.asarray(boundary_twiss[:, 3]),
        jnp.asarray(boundary_dispersion[:, 0]),
        jnp.asarray(boundary_dispersion[:, 1]),
        jnp.asarray(lattice.periodicity * phase_x / (2.0 * math.pi)),
        jnp.asarray(lattice.periodicity * phase_y / (2.0 * math.pi)),
        jnp.asarray(compaction),
        jnp.asarray(compaction - inverse_gamma_sq),
        jnp.asarray(periodic.condition_estimate, dtype=jnp.float64),
        jnp.asarray(cell),
        jnp.asarray(residual),
    )
    return _OpticsSolution(optics, boundary_dispersion, boundary_twiss), matrices


def _bend_integrals(
    element: RingElement,
    dispersion: np.ndarray,
    twiss: np.ndarray,
    inverse_gamma_sq: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Kronrod integrals of the five integrands over one bend and their Gauss gap."""
    rule = gauss_kronrod_data(15)
    if rule.embedded_weights is None:
        raise RuntimeError("The Gauss–Kronrod rule lacks its embedded Gauss weights.")
    half = 0.5 * element.length
    points = half * (np.asarray(rule.nodes) + 1.0)
    h = element.curvature
    k1 = element.gradient
    twiss_x = _twiss_matrix(twiss[0], twiss[1])
    samples = np.zeros((points.size, 5), dtype=np.float64)
    for row, position in enumerate(points):
        matrix = _element_matrix(h, k1, float(position), inverse_gamma_sq)
        eta, slope = matrix[0:2, 0:2] @ dispersion + matrix[0:2, 5]
        local = matrix[0:2, 0:2] @ twiss_x @ matrix[0:2, 0:2].T
        curly = local[1, 1] * eta * eta - 2.0 * local[0, 1] * eta * slope
        curly += local[0, 0] * slope * slope
        cube = abs(h) ** 3
        samples[row] = (h * eta, h * h, cube, h * eta * (h * h + 2.0 * k1), cube * curly)
    kronrod = half * np.asarray(rule.weights) @ samples
    gauss = half * np.asarray(rule.embedded_weights) @ samples
    return kronrod, np.abs(kronrod - gauss)


def _integrals(
    lattice: RingLattice,
    solution: _OpticsSolution,
    inverse_gamma_sq: float,
    /,
) -> RingRadiationIntegrals:
    count = len(lattice.elements)
    contributions = np.zeros((count, 5), dtype=np.float64)
    error = np.zeros((5,), dtype=np.float64)
    for index, element in enumerate(lattice.elements):
        if element.kind != "sector-bend":
            continue
        value, gap = _bend_integrals(
            element,
            solution.boundary_dispersion[index],
            solution.boundary_twiss[index],
            inverse_gamma_sq,
        )
        contributions[index] = value
        error += gap
    totals = jnp.asarray(lattice.periodicity * np.sum(contributions, axis=0))
    return RingRadiationIntegrals(
        totals[0],
        totals[1],
        totals[2],
        totals[3],
        totals[4],
        jnp.asarray(contributions),
        jnp.asarray(lattice.periodicity * error),
    )


def _reference(
    scale: ElectromagneticScaleContract,
    rest_energy: float,
    momentum: float,
    charge: float,
    minimum_lorentz_factor: float,
    /,
) -> _Reference:
    rest = positive_finite_float(rest_energy, "reference_rest_energy")
    momentum_ = positive_finite_float(momentum, "reference_momentum")
    charge_ = finite_real_scalar(charge, "reference_charge")
    if charge_ == 0.0:
        raise ValueError("A radiating ring needs a charged reference particle.")
    momentum_energy = momentum_ * float(scale.speed_of_light)
    ratio = momentum_energy / rest
    gamma = math.sqrt(1.0 + ratio * ratio)
    if gamma < minimum_lorentz_factor:
        raise RingOpticsError(
            f"Reference Lorentz factor {gamma!r} is below the ultrarelativistic "
            f"radiation threshold {minimum_lorentz_factor!r}."
        )
    return _Reference(rest, momentum_energy, charge_, gamma, ratio / gamma)


def _equilibrium(
    lattice: RingLattice,
    scale: ElectromagneticScaleContract,
    reference: _Reference,
    integrals: RingRadiationIntegrals,
    maximum_quantum_parameter: float,
    /,
) -> RingRadiationEquilibrium:
    speed = float(scale.speed_of_light)
    hbar_c = float(scale.reduced_planck_constant) * speed
    coupling = reference.charge**2 / (4.0 * math.pi * float(scale.vacuum_permittivity))
    gamma = reference.gamma
    i2 = float(integrals.i2)
    i3 = float(integrals.i3)
    i4 = float(integrals.i4)
    i5 = float(integrals.i5)
    partition = np.asarray((1.0 - i4 / i2, 1.0, 2.0 + i4 / i2), dtype=np.float64)
    if not (partition[0] > 0.0 and partition[2] > 0.0):
        raise RingOpticsError(
            "The ring has no radiation-damped equilibrium: damping partition "
            f"J = {tuple(float(value) for value in partition)!r}."
        )
    maximum_curvature = max(
        abs(value.curvature) for value in lattice.elements if value.kind == "sector-bend"
    )
    quantum = hbar_c * gamma * gamma * maximum_curvature / reference.rest_energy
    if quantum > maximum_quantum_parameter:
        raise RingOpticsError(
            f"Quantum parameter {quantum!r} exceeds the classical-radiation limit "
            f"{maximum_quantum_parameter!r}."
        )
    loss = (2.0 / 3.0) * coupling * gamma**4 * i2
    energy = gamma * reference.rest_energy
    period = lattice.circumference / (reference.beta * speed)
    times = 2.0 * energy * period / (partition * loss)
    excitation = _QUANTUM_EXCITATION * hbar_c / reference.rest_energy
    bend_angle = lattice.periodicity * math.fsum(
        abs(value.curvature) * value.length
        for value in lattice.elements
        if value.kind == "sector-bend"
    )
    photons = _PHOTONS_PER_RADIAN * coupling / hbar_c * gamma * bend_angle
    return RingRadiationEquilibrium(
        jnp.asarray(loss),
        jnp.asarray(period),
        jnp.asarray(partition),
        jnp.asarray(times),
        jnp.asarray(times / period),
        jnp.asarray(excitation * gamma * gamma * i5 / (partition[0] * i2)),
        jnp.asarray(math.sqrt(excitation * gamma * gamma * i3 / (partition[2] * i2))),
        jnp.asarray(photons),
        jnp.asarray(1.5 * hbar_c * gamma**3 * maximum_curvature),
        jnp.asarray(quantum),
    )


def _convention_matrix(late_sign: float, /) -> np.ndarray:
    """``S`` mapping canonical ``z`` to the bunch ``ζ = −σ z`` (``S = S⁻¹``)."""
    matrix = np.eye(6, dtype=np.float64)
    matrix[4, 4] = -late_sign
    return matrix


class RingRadiationPlan(StrictModule, NonTrainableState):
    """Optics, radiation integrals, and equilibrium of a ring for one reference beam.

    Construction solves the periodic optics, integrates ``I₁…I₅``, and derives
    the damping partition, damping times, energy loss per turn ``U₀``, and the
    equilibrium emittance and energy spread. Unstable optics, rings without a
    radiation-damped equilibrium (``J_x ≤ 0`` or ``J_s ≤ 0``), non-closed
    lattices, sub-threshold Lorentz factors, and quantum parameters above
    ``maximum_quantum_parameter`` raise ``RingOpticsError``. ``one_turn`` is
    the non-radiating first-order one-turn map in the bunch convention (RF
    linearized about its zero-loss synchronous phase) for ``track_ring`` and
    ``linear_ring_optics``.
    """

    lattice: RingLattice
    convention: AcceleratorConvention
    optics: RingOptics
    integrals: RingRadiationIntegrals
    equilibrium: RingRadiationEquilibrium
    spectrum: SynchrotronPhotonSpectrum
    one_turn: SymplecticMapPlan
    scale: ElectromagneticScaleContract = eqx.field(static=True)
    reference_rest_energy: float = eqx.field(static=True)
    reference_momentum: float = eqx.field(static=True)
    reference_charge: float = eqx.field(static=True)
    lorentz_factor: float = eqx.field(static=True)
    late_sign: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        lattice: RingLattice,
        scale: ElectromagneticScaleContract,
        /,
        *,
        reference_rest_energy: float,
        reference_momentum: float,
        reference_charge: float,
        convention: AcceleratorConvention | None = None,
        spectrum: SynchrotronPhotonSpectrum | None = None,
        minimum_lorentz_factor: float = 100.0,
        maximum_quantum_parameter: float = 1.0e-2,
    ) -> None:
        convention_ = AcceleratorConvention() if convention is None else convention
        if not isinstance(convention_, AcceleratorConvention):
            raise TypeError("convention must be an AcceleratorConvention.")
        spectrum_ = SynchrotronPhotonSpectrum() if spectrum is None else spectrum
        if not isinstance(spectrum_, SynchrotronPhotonSpectrum):
            raise TypeError("spectrum must be a SynchrotronPhotonSpectrum.")
        sign = _late_sign(convention_)
        minimum = positive_finite_float(minimum_lorentz_factor, "minimum_lorentz_factor")
        quantum = positive_finite_float(
            maximum_quantum_parameter, "maximum_quantum_parameter"
        )
        reference = _reference(
            scale, reference_rest_energy, reference_momentum, reference_charge, minimum
        )
        inverse_gamma_sq = 1.0 / reference.gamma**2
        solution, matrices = _optics(lattice, inverse_gamma_sq)
        integrals = _integrals(lattice, solution, inverse_gamma_sq)
        equilibrium = _equilibrium(lattice, scale, reference, integrals, quantum)
        slip = float(solution.optics.slip_factor)
        if slip == 0.0:
            raise RingOpticsError("The ring operates exactly at transition.")
        one_turn = _one_turn_map(lattice, matrices, reference, slip, sign, convention_)
        self.lattice = lattice
        self.convention = convention_
        self.optics = solution.optics
        self.integrals = integrals
        self.equilibrium = equilibrium
        self.spectrum = spectrum_
        self.one_turn = one_turn
        self.scale = scale
        self.reference_rest_energy = reference.rest_energy
        self.reference_momentum = float(reference_momentum)
        self.reference_charge = reference.charge
        self.lorentz_factor = reference.gamma
        self.late_sign = sign
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-ring-radiation-plan",
                "lattice": lattice.lattice_id,
                "scale": scale.scale_id,
                "convention": convention_.convention_id,
                "spectrum": spectrum_.spectrum_id,
                "reference": [
                    reference.rest_energy,
                    float(reference_momentum),
                    reference.charge,
                ],
                "minimum_lorentz_factor": minimum,
                "maximum_quantum_parameter": quantum,
            }
        )


def _rf_amplitude(element: RingElement, reference: _Reference, /) -> float:
    """``|q| V / (β₀ p₀ c)``: relative momentum gain of a crest passage."""
    return (
        abs(reference.charge)
        * element.voltage
        / (reference.beta * reference.momentum_energy)
    )


def _rf_wavenumber(element: RingElement, lattice: RingLattice, /) -> float:
    return 2.0 * math.pi * element.harmonic / lattice.circumference


def _one_turn_map(
    lattice: RingLattice,
    matrices: list[np.ndarray],
    reference: _Reference,
    slip: float,
    late_sign: float,
    convention: AcceleratorConvention,
    /,
) -> SymplecticMapPlan:
    """Non-radiating one-turn map; RF linearized about its zero-loss phase."""
    phase = math.pi if slip > 0.0 else 0.0
    cell = np.eye(6, dtype=np.float64)
    for element, matrix in zip(lattice.elements, matrices, strict=True):
        step = matrix
        if element.kind == "rf-cavity":
            step = np.eye(6, dtype=np.float64)
            step[5, 4] = (
                -_rf_amplitude(element, reference)
                * _rf_wavenumber(element, lattice)
                * math.cos(phase)
            )
        cell = step @ cell
    turn = np.linalg.matrix_power(cell, lattice.periodicity)
    convert = _convention_matrix(late_sign)
    return SymplecticMapPlan(
        convert @ turn @ convert,
        np.zeros((6,), dtype=np.float64),
        convention,
        element_id=f"{lattice.lattice_id}:one-turn",
        maximum_symplectic_residual=1.0e-9,
    )


# --------------------------------------------------------------------------- #
# Radiative tracking.
# --------------------------------------------------------------------------- #


def _poisson_tail(mean: float, capacity: int, /) -> float:
    """``P(N > capacity)`` for ``N ~ Poisson(mean)`` by direct tail summation.

    Capacities below the mean are reported with the bound ``1``; otherwise the
    tail terms decrease monotonically and the first one is formed in log space
    so that a large mean cannot underflow it.
    """
    if mean == 0.0:
        return 0.0
    if capacity < mean:
        return 1.0
    count = capacity + 1
    term = math.exp(count * math.log(mean) - mean - math.lgamma(count + 1))
    total = 0.0
    while term > 0.0 and term > 1.0e-18 * total:
        total += term
        count += 1
        term *= mean / count
    return total


class _StepTable(NamedTuple):
    matrices: np.ndarray
    kinds: np.ndarray
    curvatures: np.ndarray
    gradients: np.ndarray
    lengths: np.ndarray
    rf_amplitudes: np.ndarray
    rf_wavenumbers: np.ndarray


def _step_table(
    plan: RingRadiationPlan, model: RingRadiationModel, slices: int, /
) -> _StepTable:
    inverse_gamma_sq = 1.0 / plan.lorentz_factor**2
    reference = _Reference(
        plan.reference_rest_energy,
        plan.reference_momentum * float(plan.scale.speed_of_light),
        plan.reference_charge,
        plan.lorentz_factor,
        math.sqrt(1.0 - inverse_gamma_sq),
    )
    match model:
        case "none":
            radiation_kind = _STEP_LINEAR
        case "classical" | "stochastic":
            radiation_kind = _STEP_RADIATION
        case _:
            assert_never(model)
    rows: list[tuple[np.ndarray, int, float, float, float, float, float]] = []
    identity = np.eye(6, dtype=np.float64)
    for element in plan.lattice.elements:
        match element.kind:
            case "drift" | "quadrupole":
                matrix = _element_matrix(
                    0.0, element.gradient, element.length, inverse_gamma_sq
                )
                rows.append((matrix, _STEP_LINEAR, 0.0, 0.0, 0.0, 0.0, 0.0))
            case "sector-bend":
                piece = element.length / slices
                half = _element_matrix(
                    element.curvature, element.gradient, 0.5 * piece, inverse_gamma_sq
                )
                for _ in range(slices):
                    rows.append(
                        (
                            half,
                            radiation_kind,
                            element.curvature,
                            element.gradient,
                            piece,
                            0.0,
                            0.0,
                        )
                    )
                    rows.append((half, _STEP_LINEAR, 0.0, 0.0, 0.0, 0.0, 0.0))
            case "rf-cavity":
                rows.append(
                    (
                        identity,
                        _STEP_RF,
                        0.0,
                        0.0,
                        0.0,
                        _rf_amplitude(element, reference),
                        _rf_wavenumber(element, plan.lattice),
                    )
                )
            case _:
                assert_never(element.kind)
    return _StepTable(
        np.stack([row[0] for row in rows]),
        np.asarray([row[1] for row in rows], dtype=np.int32),
        *(
            np.asarray([row[column] for row in rows], dtype=np.float64)
            for column in range(2, 7)
        ),
    )


class RadiativeRingTrackingPlan(StrictModule, NonTrainableState):
    """Turn-by-turn tracking of a bunch through a radiating ring.

    Each cell element applies its first-order canonical map; sector bends are
    split into ``slices_per_bend`` slices with a radiation kick at each slice
    center, and thin RF cavities apply ``|q| V sin(φ_s − k_rf z)``. ``model``
    selects ``"none"`` (symplectic), ``"classical"`` (deterministic mean loss:
    damping without excitation), or ``"stochastic"`` (Poisson photon emission
    from the classical spectrum: damping and quantum excitation). The
    synchronous phase restores the model's reference loss per turn and is
    refused when the cavities cannot, or when the lumped synchrotron motion is
    unstable. ``photon_capacity`` is the smallest per-slice photon capacity
    whose Poisson overflow probability at the reference rate is at most
    ``photon_overflow_probability``; realized overflows are counted, never
    silently accepted. Tracking memory (per-turn moments plus the per-step
    photon workspace) is refused above ``maximum_bytes``.
    """

    __strict_contract__ = True

    radiation: RingRadiationPlan
    step_matrices: Float64[_StepDim, Literal[6], Literal[6]]
    step_kinds: Int32[_StepDim]
    step_curvatures: Float64[_StepDim]
    step_gradients: Float64[_StepDim]
    step_lengths: Float64[_StepDim]
    step_rf_amplitudes: Float64[_StepDim]
    step_rf_wavenumbers: Float64[_StepDim]
    model: RingRadiationModel = eqx.field(static=True)
    turn_count: int = eqx.field(static=True)
    slices_per_bend: int = eqx.field(static=True)
    step_count: Size[_StepDim] = eqx.field(static=True)
    horizontal_aperture: float = eqx.field(static=True)
    vertical_aperture: float = eqx.field(static=True)
    synchronous_phase: float = eqx.field(static=True)
    synchrotron_tune: float = eqx.field(static=True)
    photon_capacity: int = eqx.field(static=True)
    overflow_probability: float = eqx.field(static=True)
    loss_coefficient: float = eqx.field(static=True)
    photon_rate_coefficient: float = eqx.field(static=True)
    critical_coefficient: float = eqx.field(static=True)
    momentum_energy: float = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        radiation: RingRadiationPlan,
        turn_count: int,
        /,
        *,
        model: RingRadiationModel,
        horizontal_aperture: float,
        vertical_aperture: float,
        slices_per_bend: int = 4,
        photon_overflow_probability: float = 1.0e-12,
        maximum_bytes: int = 2**30,
    ) -> None:
        model_ = parse(model, RingRadiationModel, "model")
        turns = positive_integer(turn_count, "turn_count")
        horizontal = positive_finite_float(horizontal_aperture, "horizontal_aperture")
        vertical = positive_finite_float(vertical_aperture, "vertical_aperture")
        slices = positive_integer(slices_per_bend, "slices_per_bend")
        overflow = positive_finite_float(
            photon_overflow_probability, "photon_overflow_probability"
        )
        if overflow >= 1.0:
            raise ValueError("photon_overflow_probability must lie in (0, 1).")
        budget = positive_integer(maximum_bytes, "maximum_bytes")
        table = _step_table(radiation, model_, slices)
        speed = float(radiation.scale.speed_of_light)
        hbar_c = float(radiation.scale.reduced_planck_constant) * speed
        coupling = radiation.reference_charge**2 / (
            4.0 * math.pi * float(radiation.scale.vacuum_permittivity)
        )
        gamma = radiation.lorentz_factor
        rate = _PHOTONS_PER_RADIAN * coupling / hbar_c * gamma
        phase, tune = _synchronous_phase(radiation, table, model_)
        capacity, probability = _photon_capacity(table, rate, model_, overflow)
        self.radiation = radiation
        self.step_matrices = jnp.asarray(table.matrices)
        self.step_kinds = jnp.asarray(table.kinds)
        self.step_curvatures = jnp.asarray(table.curvatures)
        self.step_gradients = jnp.asarray(table.gradients)
        self.step_lengths = jnp.asarray(table.lengths)
        self.step_rf_amplitudes = jnp.asarray(table.rf_amplitudes)
        self.step_rf_wavenumbers = jnp.asarray(table.rf_wavenumbers)
        self.model = model_
        self.turn_count = turns
        self.slices_per_bend = slices
        self.step_count = table.kinds.size
        self.horizontal_aperture = horizontal
        self.vertical_aperture = vertical
        self.synchronous_phase = phase
        self.synchrotron_tune = tune
        self.photon_capacity = capacity
        self.overflow_probability = probability
        self.loss_coefficient = (2.0 / 3.0) * coupling * gamma**4
        self.photon_rate_coefficient = rate
        self.critical_coefficient = 1.5 * hbar_c * gamma**3
        self.momentum_energy = radiation.reference_momentum * speed
        self.maximum_bytes = budget
        self.plan_id = canonical_fingerprint(
            {
                "kind": "accelerator-radiative-ring-tracking-plan",
                "radiation": radiation.plan_id,
                "model": model_,
                "turns": turns,
                "slices_per_bend": slices,
                "apertures": [horizontal, vertical],
                "photon_overflow_probability": overflow,
                "maximum_bytes": budget,
            }
        )

    def memory_bytes(self, particle_count: int, /) -> int:
        """Per-turn moment history plus the per-step particle and photon workspace."""
        count = positive_integer(particle_count, "particle_count")
        history = self.turn_count * _MOMENT_CHANNELS * 8
        workspace = count * (6 * 8 * 3 + (self.photon_capacity + 4) * 8)
        return history + workspace


def _synchronous_phase(
    plan: RingRadiationPlan, table: _StepTable, model: RingRadiationModel, /
) -> tuple[float, float]:
    """Synchronous phase for the model's loss and the lumped synchrotron tune."""
    match model:
        case "none":
            loss = 0.0
        case "classical" | "stochastic":
            loss = float(plan.equilibrium.energy_loss_per_turn)
        case _:
            assert_never(model)
    periods = plan.lattice.periodicity
    rf = table.kinds == _STEP_RF
    momentum_energy = plan.reference_momentum * float(plan.scale.speed_of_light)
    beta = math.sqrt(1.0 - 1.0 / plan.lorentz_factor**2)
    # Crest energy gain per turn |q| V_total = Σ amplitude · β₀ p₀ c.
    crest = periods * float(np.sum(table.rf_amplitudes[rf])) * beta * momentum_energy
    slip = float(plan.optics.slip_factor)
    if not np.any(rf):
        if loss > 0.0:
            raise RingOpticsError("Radiative tracking needs RF cavities to restore U₀.")
        return 0.0, 0.0
    if not loss < crest:
        raise RingOpticsError(
            f"RF crest gain {crest!r} per turn cannot restore the energy loss {loss!r}."
        )
    ratio = loss / crest
    phase = math.pi - math.asin(ratio) if slip > 0.0 else math.asin(ratio)
    focusing = (
        -periods
        * float(np.sum(table.rf_amplitudes[rf] * table.rf_wavenumbers[rf]))
        * math.cos(phase)
    )
    half_trace = 1.0 - 0.5 * slip * plan.lattice.circumference * focusing
    if not abs(half_trace) < 1.0:
        raise RingOpticsError(
            f"Synchrotron motion is unstable: lumped half-trace {half_trace!r}."
        )
    return phase, math.acos(half_trace) / (2.0 * math.pi)


def _photon_capacity(
    table: _StepTable, rate: float, model: RingRadiationModel, overflow: float, /
) -> tuple[int, float]:
    match model:
        case "none" | "classical":
            return 0, 0.0
        case "stochastic":
            pass
        case _:
            assert_never(model)
    radiating = table.kinds == _STEP_RADIATION
    mean = rate * float(
        np.max(np.abs(table.curvatures[radiating]) * table.lengths[radiating])
    )
    for capacity in range(1, _MAXIMUM_PHOTON_CAPACITY + 1):
        tail = _poisson_tail(mean, capacity)
        if tail <= overflow:
            return capacity, tail
    raise RingOpticsError(
        f"A bend slice emits {mean!r} photons on average; the photon capacity for "
        f"overflow probability {overflow!r} exceeds {_MAXIMUM_PHOTON_CAPACITY}. "
        "Increase slices_per_bend."
    )


class RadiativeRingTrackingEvidence(StrictModule):
    """Turn-resolved beam moments, radiation ledger, and integrity evidence.

    Moments are weighted over lanes alive at the end of each turn, in the bunch
    convention. ``horizontal_emittance`` removes the statistical correlation
    with ``δ`` (dispersion-free betatron emittance); ``energy_spread`` is the
    RMS ``δ``. ``radiated_energy`` and ``rf_energy`` are weighted mean energies
    per particle and turn (scale energy unit); ``photon_count`` is the mean
    emitted photon count per particle and turn. ``photon_overflow_count``
    counts slice emissions whose Poisson count exceeded ``photon_capacity``
    (those emissions are truncated and the run is not accepted).
    """

    __strict_contract__ = True

    status: Int32[Scalar]
    accepted: Bool[Scalar]
    finite: Bool[Scalar]
    mean: Float64[_TurnDim, Literal[6]]
    covariance: Float64[_TurnDim, Literal[6], Literal[6]]
    horizontal_emittance: Float64[_TurnDim]
    vertical_emittance: Float64[_TurnDim]
    energy_spread: Float64[_TurnDim]
    radiated_energy: Float64[_TurnDim]
    rf_energy: Float64[_TurnDim]
    photon_count: Float64[_TurnDim]
    transmission: Float64[_TurnDim]
    loss_turn: Int32[_ParticleDim]
    photon_overflow_count: Int32[Scalar]
    nonfinite_count: Int32[Scalar]
    turn_count: Size[_TurnDim] = eqx.field(static=True)
    particle_count: Size[_ParticleDim] = eqx.field(static=True)
    memory_bytes: int = eqx.field(static=True)


class RadiativeRingTrackingResult(StrictModule):
    """Final bunch (lost lanes inactive) and tracking evidence."""

    bunch: AcceleratorBunch
    evidence: RadiativeRingTrackingEvidence
    plan_id: str = eqx.field(static=True)


class _StepData(NamedTuple):
    matrix: Array
    kind: Array
    curvature: Array
    gradient: Array
    length: Array
    rf_amplitude: Array
    rf_wavenumber: Array
    index: Array


class _Kick(NamedTuple):
    coordinates: Array
    radiated: Array
    rf: Array
    photons: Array
    overflow: Array


class _TurnCarry(NamedTuple):
    coordinates: Array
    alive: Array
    loss_turn: Array
    nonfinite: Array


class _Ledger(NamedTuple):
    radiated: Array
    rf: Array
    photons: Array
    overflow: Array


type _RadiationKick = Callable[[Array, Array, Array, Array, _StepData], _Kick]


def _apply_emission(
    coordinates: Array,
    momentum_loss: Array,
    radiated: Array,
    photons: Array,
    overflow: Array,
) -> _Kick:
    """Remove photon momentum along the particle direction."""
    delta = coordinates[:, 5]
    factor = 1.0 - momentum_loss / (1.0 + delta)
    updated = coordinates.at[:, 1].multiply(factor)
    updated = updated.at[:, 3].multiply(factor)
    updated = updated.at[:, 5].add(-momentum_loss)
    return _Kick(updated, radiated, jnp.zeros_like(radiated), photons, overflow)


def _field_and_path(coordinates: Array, data: _StepData) -> tuple[Array, Array]:
    x = coordinates[:, 0]
    return data.curvature + data.gradient * x, (1.0 + data.curvature * x) * data.length


def _classical_kick(plan: RadiativeRingTrackingPlan, /) -> _RadiationKick:
    def kick(
        coordinates: Array, alive: Array, turn: Array, cell: Array, data: _StepData
    ) -> _Kick:
        del alive, turn, cell
        field, path = _field_and_path(coordinates, data)
        delta = coordinates[:, 5]
        energy = plan.loss_coefficient * (1.0 + delta) ** 2 * field * field * path
        return _apply_emission(
            coordinates,
            energy / plan.momentum_energy,
            energy,
            jnp.zeros(delta.shape, dtype=jnp.int32),
            jnp.zeros(delta.shape, dtype=jnp.bool_),
        )

    return kick


def _stochastic_kick(
    plan: RadiativeRingTrackingPlan, root: PRNGKey, identities: Array, /
) -> _RadiationKick:
    spectrum = plan.radiation.spectrum
    address = SampleAddress(
        "phydrax.applications.accelerator",
        "ring-radiation-photon-emission",
        target=plan.radiation.lattice.lattice_id,
    )
    capacity = plan.photon_capacity
    slots = jnp.arange(capacity, dtype=jnp.int32)
    orders = jnp.arange(1, capacity + 1, dtype=jnp.float64)

    def kick(
        coordinates: Array, alive: Array, turn: Array, cell: Array, data: _StepData
    ) -> _Kick:
        field, path = _field_and_path(coordinates, data)
        delta = coordinates[:, 5]
        rate = jnp.where(alive, plan.photon_rate_coefficient * jnp.abs(field) * path, 0.0)
        rate = jnp.maximum(rate, 0.0)
        critical = plan.critical_coefficient * (1.0 + delta) ** 2 * jnp.abs(field)

        def emit(identity: Array, mean: Array) -> tuple[Array, Array]:
            key = derive_key(root, address, turn, cell, data.index, identity)
            uniforms = jr.uniform(key, (capacity + 1,), dtype=jnp.float64)
            # Inverse-transform Poisson count; ``capacity + 1`` marks overflow.
            probabilities = jnp.exp(-mean) * jnp.concatenate(
                (jnp.ones((1,), dtype=jnp.float64), jnp.cumprod(mean / orders))
            )
            count = jnp.sum(uniforms[0] > jnp.cumsum(probabilities), dtype=jnp.int32)
            fractions = _sample_fractions(
                spectrum.nodes, spectrum.densities, spectrum.cdf, uniforms[1:]
            )
            return jnp.sum(jnp.where(slots < count, fractions, 0.0)), count

        fraction_sum, counts = jax.vmap(emit)(identities, rate)
        energy = jnp.where(alive, critical * fraction_sum, 0.0)
        return _apply_emission(
            coordinates,
            energy / plan.momentum_energy,
            energy,
            counts,
            alive & (counts > capacity),
        )

    return kick


def _moments(coordinates: Array, weights: Array) -> tuple[Array, Array]:
    total = jnp.sum(weights)
    normalized = weights / jnp.maximum(total, jnp.finfo(weights.dtype).tiny)
    mean = ein.contract("n,ni->i", normalized, coordinates)
    centered = coordinates - mean
    covariance = ein.contract("n,ni,nj->ij", normalized, centered, centered)
    return mean, covariance


def _emittances(covariance: Array, /) -> tuple[Array, Array, Array]:
    """Dispersion-free horizontal, vertical emittance and RMS δ per turn."""
    variance = covariance[:, 5, 5]
    safe = jnp.where(variance > 0.0, variance, 1.0)
    correlation = jnp.where(
        variance[:, None] > 0.0, covariance[:, 0:2, 5] / safe[:, None], 0.0
    )
    horizontal = covariance[:, 0:2, 0:2] - variance[:, None, None] * (
        correlation[:, :, None] * correlation[:, None, :]
    )
    plan = SmallLinearSolvePlan(2)
    emittance_x = jnp.sqrt(jnp.maximum(determinant_small_linear(plan, horizontal), 0.0))
    emittance_y = jnp.sqrt(
        jnp.maximum(determinant_small_linear(plan, covariance[:, 2:4, 2:4]), 0.0)
    )
    return emittance_x, emittance_y, jnp.sqrt(jnp.maximum(variance, 0.0))


def _validate_bunch(plan: RadiativeRingTrackingPlan, bunch: AcceleratorBunch, /) -> Array:
    radiation = plan.radiation
    if bunch.coordinates.dtype != jnp.float64:
        raise TypeError("Radiative ring tracking requires float64 bunch coordinates.")
    if bunch.convention.convention_id != radiation.convention.convention_id:
        raise ValueError("Ring radiation plan and bunch coordinate conventions differ.")
    reference = (
        float(bunch.reference_rest_energy),
        float(bunch.reference_momentum),
        float(bunch.reference_charge),
    )
    expected = (
        radiation.reference_rest_energy,
        radiation.reference_momentum,
        radiation.reference_charge,
    )
    if not np.allclose(reference, expected, rtol=1.0e-12, atol=0.0):
        raise ValueError("Bunch reference particle differs from the ring radiation plan.")
    tracked = bunch.active & bunch.valid
    identities = np.asarray(bunch.particle_ids)[np.asarray(tracked)]
    if np.unique(identities).size != identities.size:
        raise ValueError("Tracked particles must carry unique particle_ids.")
    return tracked


def track_radiative_ring(
    plan: RadiativeRingTrackingPlan,
    bunch: AcceleratorBunch,
    /,
    *,
    key: PRNGKey | None = None,
) -> RadiativeRingTrackingResult:
    """Track ``bunch`` for ``plan.turn_count`` turns with synchrotron radiation.

    The ``"stochastic"`` model requires ``key``; each emission derives its
    stream from ``(turn, cell, step, particle_id)`` so results do not depend on
    slot order or batching. The other models refuse a key.
    """
    if not isinstance(plan, RadiativeRingTrackingPlan) or not isinstance(
        bunch, AcceleratorBunch
    ):
        raise TypeError(
            "plan and bunch must be RadiativeRingTrackingPlan and AcceleratorBunch."
        )
    initial_alive = _validate_bunch(plan, bunch)
    required = plan.memory_bytes(bunch.capacity)
    if required > plan.maximum_bytes:
        raise RadiativeRingTrackingResourceError(
            f"Radiative ring tracking needs {required} bytes; the budget is "
            f"{plan.maximum_bytes}."
        )
    match plan.model:
        case "stochastic":
            if key is None:
                raise ValueError("The stochastic radiation model requires a key.")
            radiate = _stochastic_kick(
                plan, parse(key, PRNGKey, "key"), bunch.particle_ids
            )
        case "classical" | "none":
            if key is not None:
                raise ValueError(f"The {plan.model!r} radiation model consumes no key.")
            radiate = _classical_kick(plan)
        case _:
            assert_never(plan.model)
    sign = plan.radiation.late_sign
    canonical = bunch.coordinates.at[:, 4].multiply(-sign)
    weights = bunch.weights
    final, history = _track(plan, radiate, canonical, initial_alive, weights)
    mean, covariance, radiated, rf, photons, overflow, transmission = history
    convert = jnp.asarray(_convention_matrix(sign))
    mean = mean @ convert
    covariance = convert @ covariance @ convert
    emittance_x, emittance_y, spread = _emittances(covariance)
    overflow_count = jnp.sum(overflow, dtype=jnp.int32)
    finite = (final.nonfinite == 0) & jnp.all(jnp.isfinite(covariance))
    status = (
        jnp.where(finite, 0, int(RadiativeRingTrackingStatus.NONFINITE))
        | jnp.where(
            overflow_count > 0, int(RadiativeRingTrackingStatus.PHOTON_OVERFLOW), 0
        )
        | jnp.where(
            jnp.any(final.loss_turn >= 0),
            int(RadiativeRingTrackingStatus.PARTICLE_LOSS),
            0,
        )
    ).astype(jnp.int32)
    evidence = RadiativeRingTrackingEvidence(
        status,
        finite & (overflow_count == 0),
        finite,
        mean,
        covariance,
        emittance_x,
        emittance_y,
        spread,
        radiated,
        rf,
        photons,
        transmission,
        final.loss_turn,
        overflow_count,
        final.nonfinite,
        plan.turn_count,
        bunch.capacity,
        plan.memory_bytes(bunch.capacity),
    )
    result_bunch = AcceleratorBunch(
        final.coordinates.at[:, 4].multiply(-sign),
        bunch.weights,
        bunch.particle_ids,
        active=final.alive,
        reference_rest_energy=float(bunch.reference_rest_energy),
        reference_momentum=float(bunch.reference_momentum),
        reference_charge=float(bunch.reference_charge),
        convention=bunch.convention,
        bunch_id=f"{bunch.bunch_id}:{plan.plan_id}",
    )
    return RadiativeRingTrackingResult(result_bunch, evidence, plan.plan_id)


def _track(
    plan: RadiativeRingTrackingPlan,
    radiate: _RadiationKick,
    coordinates: Array,
    alive: Array,
    weights: Array,
    /,
) -> tuple[_TurnCarry, tuple[Array, ...]]:
    steps = _StepData(
        plan.step_matrices,
        plan.step_kinds,
        plan.step_curvatures,
        plan.step_gradients,
        plan.step_lengths,
        plan.step_rf_amplitudes,
        plan.step_rf_wavenumbers,
        jnp.arange(plan.step_count, dtype=jnp.int32),
    )
    phase = plan.synchronous_phase
    energy_scale = (
        math.sqrt(1.0 - 1.0 / plan.radiation.lorentz_factor**2) * plan.momentum_energy
    )
    horizontal = plan.horizontal_aperture
    vertical = plan.vertical_aperture

    def linear(
        coordinates: Array, alive: Array, turn: Array, cell: Array, data: _StepData
    ) -> _Kick:
        del alive, turn, cell, data
        zeros = jnp.zeros(coordinates.shape[:1], dtype=coordinates.dtype)
        return _Kick(
            coordinates,
            zeros,
            zeros,
            jnp.zeros(zeros.shape, dtype=jnp.int32),
            jnp.zeros(zeros.shape, dtype=jnp.bool_),
        )

    def cavity(
        coordinates: Array, alive: Array, turn: Array, cell: Array, data: _StepData
    ) -> _Kick:
        del alive, turn, cell
        gain = data.rf_amplitude * jnp.sin(phase - data.rf_wavenumber * coordinates[:, 4])
        zeros = jnp.zeros(gain.shape, dtype=gain.dtype)
        return _Kick(
            coordinates.at[:, 5].add(gain),
            zeros,
            gain * energy_scale,
            jnp.zeros(gain.shape, dtype=jnp.int32),
            jnp.zeros(gain.shape, dtype=jnp.bool_),
        )

    branches = (linear, radiate, cavity)

    def step(
        carry: tuple[Array, Array, Array, _Ledger, Array, Array], data: _StepData
    ) -> tuple[tuple[Array, Array, Array, _Ledger, Array, Array], None]:
        coordinates, alive, nonfinite, ledger, turn, cell = carry
        moved = coordinates @ data.matrix.T
        kick = jax.lax.switch(data.kind, branches, moved, alive, turn, cell, data)
        candidate = kick.coordinates
        finite = jnp.all(jnp.isfinite(candidate), axis=1)
        inside = (
            (jnp.abs(candidate[:, 0]) <= horizontal)
            & (jnp.abs(candidate[:, 2]) <= vertical)
            & (candidate[:, 5] > -1.0)
        )
        survives = alive & finite & inside
        next_coordinates = jnp.where(survives[:, None], candidate, coordinates)
        mask = jnp.where(alive, 1.0, 0.0)
        ledger = _Ledger(
            ledger.radiated + mask * kick.radiated,
            ledger.rf + mask * kick.rf,
            ledger.photons + jnp.where(alive, kick.photons, 0),
            ledger.overflow + jnp.sum(alive & kick.overflow, dtype=jnp.int32),
        )
        nonfinite = nonfinite + jnp.sum(alive & ~finite, dtype=jnp.int32)
        return (next_coordinates, survives, nonfinite, ledger, turn, cell), None

    def one_cell(
        carry: tuple[Array, Array, Array, _Ledger, Array], cell: Array
    ) -> tuple[tuple[Array, Array, Array, _Ledger, Array], None]:
        coordinates, alive, nonfinite, ledger, turn = carry
        (coordinates, alive, nonfinite, ledger, _, _), _ = jax.lax.scan(
            step, (coordinates, alive, nonfinite, ledger, turn, cell), steps
        )
        return (coordinates, alive, nonfinite, ledger, turn), None

    cells = jnp.arange(plan.radiation.lattice.periodicity, dtype=jnp.int32)

    def one_turn(carry: _TurnCarry, turn: Array) -> tuple[_TurnCarry, tuple[Array, ...]]:
        zeros = jnp.zeros(carry.alive.shape, dtype=coordinates.dtype)
        ledger = _Ledger(
            zeros,
            zeros,
            jnp.zeros(carry.alive.shape, dtype=jnp.int32),
            jnp.zeros((), dtype=jnp.int32),
        )
        (next_coordinates, next_alive, nonfinite, ledger, _), _ = jax.lax.scan(
            one_cell,
            (carry.coordinates, carry.alive, carry.nonfinite, ledger, turn),
            cells,
        )
        lost = carry.alive & ~next_alive
        loss_turn = jnp.where(lost & (carry.loss_turn < 0), turn, carry.loss_turn)
        live_weights = jnp.where(next_alive, weights, 0.0)
        mean, covariance = _moments(next_coordinates, live_weights)
        total = jnp.maximum(jnp.sum(live_weights), jnp.finfo(weights.dtype).tiny)
        record = (
            mean,
            covariance,
            jnp.sum(live_weights * ledger.radiated) / total,
            jnp.sum(live_weights * ledger.rf) / total,
            jnp.sum(live_weights * ledger.photons) / total,
            ledger.overflow,
            jnp.sum(live_weights)
            / jnp.maximum(jnp.sum(weights), jnp.finfo(weights.dtype).tiny),
        )
        return _TurnCarry(next_coordinates, next_alive, loss_turn, nonfinite), record

    initial = _TurnCarry(
        coordinates,
        alive,
        jnp.full(alive.shape, -1, dtype=jnp.int32),
        jnp.zeros((), dtype=jnp.int32),
    )
    return jax.lax.scan(one_turn, initial, jnp.arange(plan.turn_count, dtype=jnp.int32))


__all__ = [
    "RadiativeRingTrackingEvidence",
    "RadiativeRingTrackingPlan",
    "RadiativeRingTrackingResourceError",
    "RadiativeRingTrackingResult",
    "RadiativeRingTrackingStatus",
    "RingElement",
    "RingElementKind",
    "RingLattice",
    "RingOptics",
    "RingOpticsError",
    "RingRadiationEquilibrium",
    "RingRadiationIntegrals",
    "RingRadiationModel",
    "RingRadiationPlan",
    "SynchrotronPhotonSpectrum",
    "track_radiative_ring",
]
