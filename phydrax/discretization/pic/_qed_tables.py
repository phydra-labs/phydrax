#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strong-field QED rate and spectrum tables in the locally constant field.

`QEDTable` prepares, on the host, the dimensionless total rate and the
cumulative spectrum of one strong-field process from the synchrotron kernels
``F(x) = x∫ₓ^∞K_{5/3}`` and ``G(x) = xK_{2/3}(x)`` of `phydrax.special` and
composite Gauss–Legendre quadrature:

- ``"nonlinear-compton"``: a lepton of energy ``ε = γmc²`` and quantum
  parameter ``χ`` emits a photon of energy fraction ``ξ = ω/ε`` at
  ``d²N/(dt dξ) = (α mc²/(ħγ)) s(χ, ξ)`` with, for ``δ = 2ξ/(3χ(1−ξ))``,

      s(χ, ξ) = [F(δ) + ξ²/(1−ξ) G(δ)] / (√3 π δ),

  total ``K(χ) = ∫₀¹ s dξ`` (``→ 5χ/(2√3)`` as ``χ → 0``);
- ``"nonlinear-breit-wheeler"``: a photon of energy ``ε_γ`` and parameter
  ``χ_γ`` creates a pair whose electron carries ``ξ = ε₋/ε_γ`` at
  ``d²N/(dt dξ) = (α m²c⁴/(ħ ε_γ)) s(χ_γ, ξ)`` with, for
  ``δ = 2/(3χ_γ ξ(1−ξ))``,

      s(χ_γ, ξ) = [G(δ)/(ξ(1−ξ)) − F(δ)] / (√3 π δ),

  total ``R(χ_γ) = ∫₀¹ s dξ = χ_γ T(χ_γ)`` in Erber's normalization
  (``T → (3/16)√(3/2) e^{−8/(3χ_γ)}`` as ``χ_γ → 0``).

Both brackets are the Baier–Katkov/Ritus spectra rewritten with
``∫_δ^∞K_{1/3} = 2K_{2/3}(δ) − ∫_δ^∞K_{5/3}``. The cumulative spectra
``P(χ, ξ)`` are tabulated on uniform grids of a per-process variable in which
the spectrum is smooth for every ``χ``: ``x = log δ`` for Compton (with the
exact ``δ^{1/3}`` power-law tail below ``δ = 1e-12``) and
``v = q / q_max(χ)`` for Breit–Wheeler, where ``ξ = 1/(1 + e^{2q})`` and
``δ = δ₀ cosh² q`` with ``δ₀ = 8/(3χ_γ)`` (the spectrum is symmetric in
``ξ ↔ 1 − ξ``). Both are truncated at ``δ = δ₀ + 80`` (``e^{−80}`` of the
mass). Rows are spaced in ``log χ`` and interpolated linearly, as are nodes
along each row, so every interpolated row is a monotone CDF and inverse
sampling is exact for the piecewise-linear CDF whose distance from the exact
one is the recorded interpolation error. The log-rate (with the Breit–Wheeler
``8/(3χ)`` exponent removed) is interpolated by cubic Hermite segments with
exact forward-mode slopes; below the table the exact small-``χ`` asymptotics
continue it.

``polarization`` selects the spectrum of a polarized incoming particle
(Seipt & King, PRA 102, 052805, 2020, rewritten with
``∫_δ^∞K_{1/3} = (2G − F)/δ``, ``K_{1/3} = H/δ``, ``H(x) = xK_{1/3}(x)``):

- ``"nonlinear-compton"``: a lepton whose spin projection on its spin
  quantization axis ``ê = v̂ × F̂`` (the rest-frame magnetic field direction
  for an electron; ``F`` the Lorentz force) is ``P = ±1`` has
  ``s_P = s − P ξ H(δ)/(√3π δ)``;
- ``"nonlinear-breit-wheeler"``: a photon whose linear Stokes parameter along
  the transverse field ``E + c k̂×B`` is ``τ = ±1`` has
  ``s_τ = s − τ G(δ)/(√3π δ)``.

``"positive"``/``"negative"`` tabulate ``P, τ = +1``/``−1``; the average of
the two is the ``"averaged"`` (unpolarized) table, and a partially polarized
particle is the mixture with weights ``(1 ± P)/2``.

The module also declares the shared strong-field QED vocabulary: which
quantity a process conserves exactly (`QEDConservation`), which particle
polarizations a run resolves (`QEDPolarizationModel`), and the per-event flags
(`QEDEventFlag`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import IntFlag
from typing import assert_never, Literal, NamedTuple, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import cubic_hermite_interpolate
from ..._numerics import gauss_legendre_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import positive_finite_float, positive_integer
from ...special import synchrotron_f, synchrotron_g, synchrotron_h
from ...typing import Dim, Float64, parse, Size


QEDProcess: TypeAlias = Literal["nonlinear-compton", "nonlinear-breit-wheeler"]
QEDConservation: TypeAlias = Literal["momentum", "energy"]
QEDTablePolarization: TypeAlias = Literal["averaged", "positive", "negative"]
QEDPolarizationModel: TypeAlias = Literal[
    "unpolarized", "photon-polarized", "spin-and-photon-polarized"
]


class QEDEventFlag(IntFlag):
    """Per-particle and per-event strong-field QED flags.

    ``BELOW_THRESHOLD`` marks a declared exemption (lepton below the minimum
    Lorentz factor, photon below the pair threshold ``2mc²``).
    ``CHI_EXCEEDED``, ``EVENT_PROBABILITY_EXCEEDED`` and ``NONFINITE`` are
    support failures, which depend only on the state (never on a random draw,
    so a rejected step cannot be retried into success);
    ``SCALE_UNSEPARATED`` is a radiation-ownership conflict with the field
    grid. ``KINEMATICS_REFUSED`` marks a sampled event the declared
    conservation cannot realize (a photon fraction above ``β`` or a product
    below its rest energy, possible only near ``γ ~ 1`` where the LCFA does not
    apply); the event is consumed without effect and counted. The remaining
    flags are validity evidence: ``OUTSIDE_LCFA_VALIDITY`` marks an event whose formation
    time exceeds the local field-variation time, ``INFRARED_CORRECTED`` an
    improved-LCFA emission from the flat infrared part of the spectrum, and
    ``ONE_STEP_TRIDENT`` / ``PHOTON_SPLITTING`` a lepton / photon whose
    formation is not local, where the unmodeled one-step trident and photon
    splitting channels are not bounded by the modeled two-step description.
    """

    NONE = 0
    BELOW_THRESHOLD = 1
    CHI_EXCEEDED = 2
    EVENT_PROBABILITY_EXCEEDED = 4
    KINEMATICS_REFUSED = 8
    SCALE_UNSEPARATED = 16
    NONFINITE = 32
    OUTSIDE_LCFA_VALIDITY = 64
    INFRARED_CORRECTED = 128
    ONE_STEP_TRIDENT = 256
    PHOTON_SPLITTING = 512


QED_SUPPORT_FAILURE = (
    QEDEventFlag.CHI_EXCEEDED
    | QEDEventFlag.EVENT_PROBABILITY_EXCEEDED
    | QEDEventFlag.NONFINITE
)

_INVERSE_NORMALIZATION = 1.0 / (math.sqrt(3.0) * math.pi)
# Spectra are cut where F, G ~ √(πδ/2) e^{−δ} fall by e^{−80} below their
# threshold value; the omitted mass is reported by the table evidence.
_EXPONENT_CUT = 80.0
_COMPTON_MINIMUM_DELTA = 1.0e-12
_COMPTON_MINIMUM_CHI = 1.0e-6
_BREIT_WHEELER_MINIMUM_CHI = 1.0e-2
_BREIT_WHEELER_EXPONENT = 8.0 / 3.0


class _ChiNodeDim(Dim, minimum=4):
    """Table rows in ``log χ``."""


class _SpectrumNodeDim(Dim, minimum=16):
    """Cumulative-spectrum nodes along one row."""


def _polarization_sign(polarization: QEDTablePolarization, /) -> float:
    match polarization:
        case "averaged":
            return 0.0
        case "positive":
            return 1.0
        case "negative":
            return -1.0
        case _:
            assert_never(polarization)


def _compton_density(chi: Array, x: Array, sign: float, /) -> Array:
    """``d(√3π N)/dx`` at ``x = log δ``: ``6χ[F + ξ²(2+a)/2 G − PξH]/(2+a)²``, ``a = 3χδ``."""
    delta = jnp.exp(x)
    a = 3.0 * chi * delta
    fraction = a / (2.0 + a)
    bracket = synchrotron_f(delta) + 0.5 * fraction**2 * (2.0 + a) * synchrotron_g(delta)
    if sign != 0.0:
        bracket = bracket - sign * fraction * synchrotron_h(delta)
    return 6.0 * chi * bracket / (2.0 + a) ** 2


def _breit_wheeler_extent(chi: Array, /) -> Array:
    """``q_max`` with ``δ₀ cosh² q_max = δ₀ + 80``."""
    threshold = _BREIT_WHEELER_EXPONENT / chi
    return jnp.arccosh(jnp.sqrt(1.0 + _EXPONENT_CUT / threshold))


def _breit_wheeler_density(chi: Array, v: Array, sign: float, /) -> Array:
    """``d(√3π N_half)/dv`` on ``ξ ∈ (0, 1/2]`` at ``v = q/q_max``."""
    extent = _breit_wheeler_extent(chi)
    q = v * extent
    squared = jnp.cosh(q) ** 2
    delta = (_BREIT_WHEELER_EXPONENT / chi) * squared
    bracket = 1.5 * chi * synchrotron_g(delta) - synchrotron_f(delta) / delta
    if sign != 0.0:
        bracket = bracket - sign * synchrotron_g(delta) / delta
    return bracket * 0.5 * extent / squared


def _segment_integrals(
    density: Callable[[Array], Array], edges: Array, order: int, /
) -> Array:
    rule = gauss_legendre_data(order)
    widths = edges[1:] - edges[:-1]
    points = edges[:-1, None] + 0.5 * widths[:, None] * (rule.nodes[None, :] + 1.0)
    weights = 0.5 * widths[:, None] * rule.weights[None, :]
    return jnp.sum(density(points) * weights, axis=-1)


class _Segments(NamedTuple):
    """Unnormalized spectral masses of one process on the table rows."""

    head: Array
    segments: Array
    upper_density: Array
    symmetry: float


class _Spectra(NamedTuple):
    """Host totals and normalized cumulative rows of one process."""

    total: np.ndarray
    cumulative: np.ndarray
    upper_density: np.ndarray


def _variable_bounds(process: QEDProcess, /) -> tuple[float, float]:
    match process:
        case "nonlinear-compton":
            return math.log(_COMPTON_MINIMUM_DELTA), math.log(_EXPONENT_CUT)
        case "nonlinear-breit-wheeler":
            return 0.0, 1.0
        case _:
            assert_never(process)


def _segments(
    process: QEDProcess, sign: float, log_chi: Array, nodes: int, order: int, /
) -> _Segments:
    """Head mass, per-segment masses (composite Gauss–Legendre) and cut density."""
    lower, upper = _variable_bounds(process)
    edges = jnp.linspace(lower, upper, nodes)
    chi = jnp.exp(log_chi)[:, None, None]
    top = jnp.full((1, 1), upper)
    match process:
        case "nonlinear-compton":
            segments = _segment_integrals(
                lambda points: _compton_density(chi, points[None], sign), edges, order
            )
            # Below δ = 1e-12 the density is exactly ∝ δ^{1/3} to O(δ^{2/3}, χδ),
            # so the tail mass is three times the density at the lower node.
            head = 3.0 * _compton_density(chi[:, :, 0], jnp.full((1, 1), lower), sign)
            return _Segments(
                head, segments, _compton_density(chi[:, :, 0], top, sign)[:, 0], 1.0
            )
        case "nonlinear-breit-wheeler":
            segments = _segment_integrals(
                lambda points: _breit_wheeler_density(chi, points[None], sign),
                edges,
                order,
            )
            return _Segments(
                jnp.zeros((log_chi.shape[0], 1)),
                segments,
                _breit_wheeler_density(chi[:, :, 0], top, sign)[:, 0],
                2.0,
            )
        case _:
            assert_never(process)


def _log_rate(
    process: QEDProcess, sign: float, log_chi: Array, nodes: int, order: int, /
) -> Array:
    """Smooth log-rate: ``log K`` or ``log R + 8/(3χ)``."""
    parts = _segments(process, sign, log_chi, nodes, order)
    total = parts.head[:, 0] + jnp.sum(parts.segments, axis=-1)
    return jnp.log(parts.symmetry * _INVERSE_NORMALIZATION * total) + _log_rate_offset(
        process, jnp.exp(log_chi)
    )


def _spectra(
    process: QEDProcess, sign: float, log_chi: Array, nodes: int, order: int, /
) -> _Spectra:
    """Host totals ``K`` or ``R`` and normalized CDF rows."""
    parts = _segments(process, sign, log_chi, nodes, order)
    head = np.asarray(parts.head)
    # NumPy's sequential cumulative sum of nonnegative masses is monotone in
    # floating point; parallel prefix sums need not be.
    cumulative = np.concatenate(
        (head, head + np.cumsum(np.asarray(parts.segments), axis=-1)), axis=-1
    )
    total = cumulative[:, -1]
    return _Spectra(
        parts.symmetry * _INVERSE_NORMALIZATION * total,
        cumulative / total[:, None],
        np.asarray(parts.upper_density) / total,
    )


def _log_rate_offset(process: QEDProcess, chi: Array, /) -> Array:
    """Exponent removed from the log-rate so that it interpolates smoothly."""
    match process:
        case "nonlinear-compton":
            return jnp.zeros_like(chi)
        case "nonlinear-breit-wheeler":
            return _BREIT_WHEELER_EXPONENT / chi
        case _:
            assert_never(process)


def _minimum_chi(process: QEDProcess, /) -> float:
    match process:
        case "nonlinear-compton":
            return _COMPTON_MINIMUM_CHI
        case "nonlinear-breit-wheeler":
            return _BREIT_WHEELER_MINIMUM_CHI
        case _:
            assert_never(process)


class QEDTable(StrictModule, NonTrainableState):
    """Rate and cumulative spectrum of one strong-field process for ``χ ≤ maximum_chi``.

    Rows are spaced by ``1/nodes_per_decade`` decades in ``χ`` from ``1e-6``
    (Compton) or ``1e-2`` (Breit–Wheeler) to ``maximum_chi``; each row holds the
    normalized cumulative spectrum on ``spectrum_nodes`` uniform nodes. Below
    the first row the rate continues with its exact small-``χ`` asymptote
    (``K ∝ χ``; ``R ∝ χ e^{−8/(3χ)}``) and the first row's spectrum shape.
    ``polarization`` selects the averaged spectrum or the spectrum of an
    incoming particle polarized with parameter ``+1``/``−1`` (module docstring).

    Host evidence, all measured at construction:

    - ``quadrature_error``: largest relative change of the rate and largest
      absolute change of the CDF when the Gauss–Legendre order is halved;
    - ``rate_interpolation_error``: largest relative rate error at row midpoints;
    - ``cdf_row_error`` / ``cdf_node_error``: largest absolute CDF error of
      linear interpolation between rows / between nodes against direct
      quadrature at the midpoints (their sum bounds the Kolmogorov–Smirnov
      distance of sampled spectra from the exact ones);
    - ``minimum_cdf_increment``: smallest node-to-node CDF increment
      (``≥ 0``: every row is monotone);
    - ``truncated_mass``: largest relative spectral mass beyond the cut,
      estimated as the normalized density at the cut times one node spacing.

    Construction refuses tables whose rate interpolation error exceeds
    ``rate_tolerance``, whose CDF error exceeds ``cdf_tolerance``, or with a
    decreasing row.
    """

    __strict_contract__ = True

    log_chi: Float64[_ChiNodeDim]
    rate_values: Float64[_ChiNodeDim]
    rate_slopes: Float64[_ChiNodeDim]
    cumulative: Float64[_ChiNodeDim, _SpectrumNodeDim]
    process: QEDProcess = eqx.field(static=True)
    polarization: QEDTablePolarization = eqx.field(static=True)
    minimum_chi: float = eqx.field(static=True)
    maximum_chi: float = eqx.field(static=True)
    variable_lower: float = eqx.field(static=True)
    variable_upper: float = eqx.field(static=True)
    quadrature_error: float = eqx.field(static=True)
    rate_interpolation_error: float = eqx.field(static=True)
    cdf_row_error: float = eqx.field(static=True)
    cdf_node_error: float = eqx.field(static=True)
    minimum_cdf_increment: float = eqx.field(static=True)
    truncated_mass: float = eqx.field(static=True)
    row_count: Size[_ChiNodeDim] = eqx.field(static=True)
    node_count: Size[_SpectrumNodeDim] = eqx.field(static=True)
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        process: QEDProcess,
        /,
        *,
        maximum_chi: float,
        polarization: QEDTablePolarization = "averaged",
        nodes_per_decade: int = 16,
        spectrum_nodes: int = 513,
        quadrature_order: int = 8,
        rate_tolerance: float = 1.0e-6,
        cdf_tolerance: float = 1.0e-3,
    ) -> None:
        process_ = parse(process, QEDProcess, "process")
        polarization_ = parse(polarization, QEDTablePolarization, "polarization")
        sign = _polarization_sign(polarization_)
        minimum = _minimum_chi(process_)
        maximum = positive_finite_float(maximum_chi, "maximum_chi")
        if maximum <= 10.0 * minimum:
            raise ValueError(f"maximum_chi must exceed {10.0 * minimum:g}.")
        density = positive_integer(nodes_per_decade, "nodes_per_decade")
        nodes = positive_integer(spectrum_nodes, "spectrum_nodes")
        order = positive_integer(quadrature_order, "quadrature_order")
        if density < 4 or nodes < 17 or order < 4:
            raise ValueError(
                "nodes_per_decade ≥ 4, spectrum_nodes ≥ 17 and quadrature_order ≥ 4 "
                "are required."
            )
        rate_limit = positive_finite_float(rate_tolerance, "rate_tolerance")
        cdf_limit = positive_finite_float(cdf_tolerance, "cdf_tolerance")
        count = math.ceil(density * math.log10(maximum / minimum)) + 1
        log_chi = jnp.linspace(math.log(minimum), math.log(maximum), count)

        def log_rate(value: Array) -> Array:
            return _log_rate(process_, sign, value, nodes, order)

        spectra = _spectra(process_, sign, log_chi, nodes, order)
        rates, slopes = jax.jvp(log_rate, (log_chi,), (jnp.ones_like(log_chi),))
        coarse = _spectra(process_, sign, log_chi, nodes, order // 2)
        quadrature = max(
            float(np.max(np.abs(coarse.total / spectra.total - 1.0))),
            float(np.max(np.abs(coarse.cumulative - spectra.cumulative))),
        )
        midpoints = 0.5 * (log_chi[:-1] + log_chi[1:])
        between = _spectra(process_, sign, midpoints, nodes, order)
        interpolated = cubic_hermite_interpolate(
            log_chi, rates, midpoints, slopes=slopes
        ).values - _log_rate_offset(process_, jnp.exp(midpoints))
        rate_error = float(
            np.max(np.abs(np.exp(np.asarray(interpolated)) / between.total - 1.0))
        )
        row_error = float(
            np.max(
                np.abs(
                    0.5 * (spectra.cumulative[:-1] + spectra.cumulative[1:])
                    - between.cumulative
                )
            )
        )
        refined = _spectra(process_, sign, log_chi, 2 * nodes - 1, order)
        node_error = float(
            np.max(
                np.abs(
                    refined.cumulative[:, 1::2]
                    - 0.5 * (spectra.cumulative[:, :-1] + spectra.cumulative[:, 1:])
                )
            )
        )
        increment = float(np.min(np.diff(spectra.cumulative, axis=-1)))
        lower, upper = _variable_bounds(process_)
        truncated = float(np.max(spectra.upper_density)) * (upper - lower) / (nodes - 1)
        if not rate_error <= rate_limit:
            raise ValueError(
                f"QED rate interpolation error {rate_error:.2e} exceeds "
                f"{rate_limit:.1e}; increase nodes_per_decade."
            )
        if not row_error + node_error <= cdf_limit:
            raise ValueError(
                f"QED spectrum interpolation error {row_error + node_error:.2e} "
                f"exceeds {cdf_limit:.1e}; increase nodes_per_decade or "
                "spectrum_nodes."
            )
        if not increment >= 0.0:
            raise ValueError("QED cumulative spectrum is not monotone.")
        self.log_chi = log_chi
        self.rate_values = rates
        self.rate_slopes = slopes
        self.cumulative = jnp.asarray(spectra.cumulative)
        self.process = process_
        self.polarization = polarization_
        self.minimum_chi = minimum
        self.maximum_chi = maximum
        self.variable_lower = lower
        self.variable_upper = upper
        self.quadrature_error = quadrature
        self.rate_interpolation_error = rate_error
        self.cdf_row_error = row_error
        self.cdf_node_error = node_error
        self.minimum_cdf_increment = increment
        self.truncated_mass = truncated
        self.row_count = count
        self.node_count = nodes
        identity: dict[str, object] = {
            "kind": "qed-table",
            "process": process_,
            "minimum_chi": minimum,
            "maximum_chi": maximum,
            "nodes_per_decade": density,
            "spectrum_nodes": nodes,
            "quadrature_order": order,
            "variable_bounds": [lower, upper],
            "exponent_cut": _EXPONENT_CUT,
        }
        # The averaged table is the unpolarized one; only polarized components
        # carry their polarization in the identity.
        if sign != 0.0:
            identity["polarization"] = polarization_
        self.table_id = canonical_fingerprint(identity)

    # -- rate ------------------------------------------------------------------

    def rate_function(self, chi: ArrayLike, /) -> Array:
        """Dimensionless total ``K(χ)`` (Compton) or ``R(χ) = χT(χ)`` (Breit–Wheeler).

        ``χ`` above ``maximum_chi`` is clipped (callers flag it); ``χ = 0``
        gives zero.
        """
        value = jnp.asarray(chi, dtype=jnp.float64)
        positive = value > 0.0
        safe = jnp.where(positive, value, self.minimum_chi)
        clipped = jnp.clip(safe, self.minimum_chi, self.maximum_chi)
        log_clipped = jnp.log(clipped)
        tabulated = cubic_hermite_interpolate(
            self.log_chi, self.rate_values, log_clipped, slopes=self.rate_slopes
        ).values
        # Below the table the log-rate (offset removed) has slope one.
        below = self.rate_values[0] + jnp.log(safe / self.minimum_chi)
        smooth = jnp.where(safe < self.minimum_chi, below, tabulated)
        rate = jnp.exp(smooth - _log_rate_offset(self.process, safe))
        return jnp.where(positive, rate, 0.0)

    def spectrum(self, chi: ArrayLike, fraction: ArrayLike, /) -> Array:
        """Dimensionless spectral density ``s(χ, ξ)``, evaluated directly from F5."""
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        xi = jnp.asarray(fraction, dtype=jnp.float64)
        interior = (xi > 0.0) & (xi < 1.0) & (chi_ > 0.0)
        safe_xi = jnp.where(interior, xi, 0.5)
        safe_chi = jnp.where(interior, chi_, 1.0)
        sign = _polarization_sign(self.polarization)
        match self.process:
            case "nonlinear-compton":
                delta = 2.0 * safe_xi / (3.0 * safe_chi * (1.0 - safe_xi))
                numerator = synchrotron_f(delta) + safe_xi**2 / (
                    1.0 - safe_xi
                ) * synchrotron_g(delta)
                if sign != 0.0:
                    numerator = numerator - sign * safe_xi * synchrotron_h(delta)
            case "nonlinear-breit-wheeler":
                product = safe_xi * (1.0 - safe_xi)
                delta = 2.0 / (3.0 * safe_chi * product)
                numerator = synchrotron_g(delta) / product - synchrotron_f(delta)
                if sign != 0.0:
                    numerator = numerator - sign * synchrotron_g(delta)
            case _:
                assert_never(self.process)
        return jnp.where(interior, _INVERSE_NORMALIZATION * (numerator / delta), 0.0)

    # -- cumulative spectrum ---------------------------------------------------

    def _rows(self, chi: Array, /) -> tuple[Array, Array]:
        """Lower row index and linear weight in ``log χ`` (clipped to the table)."""
        spacing = (self.log_chi[-1] - self.log_chi[0]) / (self.row_count - 1)
        position = (
            jnp.log(jnp.clip(chi, self.minimum_chi, self.maximum_chi)) - self.log_chi[0]
        ) / spacing
        index = jnp.clip(jnp.floor(position), 0, self.row_count - 2).astype(jnp.int32)
        return index, jnp.clip(position - index, 0.0, 1.0)

    def _node_value(self, index: Array, weight: Array, node: Array, /) -> Array:
        lower = self.cumulative[index, node]
        upper = self.cumulative[index + 1, node]
        return lower + weight * (upper - lower)

    def _variable(self, chi: Array, fraction: Array, /) -> Array:
        """Row variable of a fraction (on the lower half for Breit–Wheeler)."""
        match self.process:
            case "nonlinear-compton":
                return jnp.log(2.0 * fraction / (3.0 * chi * (1.0 - fraction)))
            case "nonlinear-breit-wheeler":
                half = jnp.minimum(fraction, 1.0 - fraction)
                return 0.5 * jnp.log((1.0 - half) / half) / _breit_wheeler_extent(chi)
            case _:
                assert_never(self.process)

    def _fraction(self, chi: Array, variable: Array, /) -> Array:
        match self.process:
            case "nonlinear-compton":
                a = 3.0 * chi * jnp.exp(variable)
                return a / (2.0 + a)
            case "nonlinear-breit-wheeler":
                return 1.0 / (1.0 + jnp.exp(2.0 * variable * _breit_wheeler_extent(chi)))
            case _:
                assert_never(self.process)

    def _row_cdf(self, chi: Array, variable: Array, /) -> Array:
        """Interpolated row CDF at a row variable (clipped to the row)."""
        index, weight = self._rows(chi)
        spacing = (self.variable_upper - self.variable_lower) / (self.node_count - 1)
        position = (variable - self.variable_lower) / spacing
        node = jnp.clip(jnp.floor(position), 0, self.node_count - 2).astype(jnp.int32)
        local = jnp.clip(position - node, 0.0, 1.0)
        left = self._node_value(index, weight, node)
        right = self._node_value(index, weight, node + 1)
        value = left + local * (right - left)
        match self.process:
            case "nonlinear-compton":
                # Exact δ^{1/3} law below the first node.
                head = left * jnp.exp(
                    jnp.minimum(variable - self.variable_lower, 0.0) / 3.0
                )
                return jnp.where(variable < self.variable_lower, head, value)
            case "nonlinear-breit-wheeler":
                return value
            case _:
                assert_never(self.process)

    def cdf(self, chi: ArrayLike, fraction: ArrayLike, /) -> Array:
        """Cumulative spectrum ``P(χ, ξ) = ∫₀^ξ s / ∫₀¹ s`` of the tabulated rows."""
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        xi = jnp.asarray(fraction, dtype=jnp.float64)
        chi_, xi = jnp.broadcast_arrays(chi_, xi)
        interior = (xi > 0.0) & (xi < 1.0)
        safe_xi = jnp.where(interior, xi, 0.5)
        safe_chi = jnp.where(chi_ > 0.0, chi_, self.minimum_chi)
        match self.process:
            case "nonlinear-compton":
                value = self._row_cdf(safe_chi, self._variable(safe_chi, safe_xi))
            case "nonlinear-breit-wheeler":
                # The lower-half row holds the half-spectrum's CDF, reversed in v.
                half = 1.0 - self._row_cdf(safe_chi, self._variable(safe_chi, safe_xi))
                value = jnp.where(safe_xi <= 0.5, 0.5 * half, 1.0 - 0.5 * half)
            case _:
                assert_never(self.process)
        return jnp.where(interior, value, jnp.where(xi <= 0.0, 0.0, 1.0))

    def _invert_row(self, chi: Array, target: Array, /) -> Array:
        """Row variable where the interpolated row CDF reaches ``target``."""
        index, weight = self._rows(chi)
        lower = jnp.zeros(target.shape, dtype=jnp.int32)
        upper = jnp.full(target.shape, self.node_count - 1, dtype=jnp.int32)

        def bisect(_: int, bounds: tuple[Array, Array]) -> tuple[Array, Array]:
            low, high = bounds
            middle = (low + high) // 2
            below = self._node_value(index, weight, middle) <= target
            return jnp.where(below, middle, low), jnp.where(below, high, middle)

        steps = math.ceil(math.log2(self.node_count - 1)) + 1
        lower, _ = jax.lax.fori_loop(0, steps, bisect, (lower, upper))
        lower = jnp.minimum(lower, self.node_count - 2)
        left = self._node_value(index, weight, lower)
        right = self._node_value(index, weight, lower + 1)
        width = right - left
        local = jnp.where(
            width > 0.0, (target - left) / jnp.where(width > 0.0, width, 1.0), 0.0
        )
        spacing = (self.variable_upper - self.variable_lower) / (self.node_count - 1)
        variable = self.variable_lower + spacing * (lower + jnp.clip(local, 0.0, 1.0))
        match self.process:
            case "nonlinear-compton":
                first = self._node_value(index, weight, jnp.zeros_like(lower))
                head = self.variable_lower + 3.0 * jnp.log(
                    jnp.maximum(target, jnp.finfo(jnp.float64).tiny)
                    / jnp.where(first > 0.0, first, 1.0)
                )
                return jnp.where(target < first, head, variable)
            case "nonlinear-breit-wheeler":
                return variable
            case _:
                assert_never(self.process)

    def quantile(self, chi: ArrayLike, probability: ArrayLike, /) -> Array:
        """Fraction ``ξ`` with ``P(χ, ξ) = probability`` (inverse-CDF sampling)."""
        chi_ = jnp.asarray(chi, dtype=jnp.float64)
        target = jnp.clip(jnp.asarray(probability, dtype=jnp.float64), 0.0, 1.0)
        chi_, target = jnp.broadcast_arrays(chi_, target)
        safe_chi = jnp.where(chi_ > 0.0, chi_, self.minimum_chi)
        match self.process:
            case "nonlinear-compton":
                return self._fraction(safe_chi, self._invert_row(safe_chi, target))
            case "nonlinear-breit-wheeler":
                lower_half = target < 0.5
                half = jnp.where(lower_half, 2.0 * target, 2.0 * (1.0 - target))
                # The half-row CDF increases from ξ = 1/2 (v = 0) outward.
                fraction = self._fraction(
                    safe_chi, self._invert_row(safe_chi, 1.0 - half)
                )
                return jnp.where(lower_half, fraction, 1.0 - fraction)
            case _:
                assert_never(self.process)


__all__ = [
    "QED_SUPPORT_FAILURE",
    "QEDConservation",
    "QEDEventFlag",
    "QEDPolarizationModel",
    "QEDProcess",
    "QEDTable",
    "QEDTablePolarization",
]
