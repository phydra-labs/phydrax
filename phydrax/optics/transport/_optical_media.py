#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Spectral optical media: tabulated dispersion, absorption, and volume processes.

`SpectralOpticalMedium` implements the `OpticalMedium` protocol from explicit
per-medium tables on one vacuum-wavelength grid: refractive index,
absorption length, and optional Rayleigh, Henyey--Greenstein, Lorenz--Mie, and
wavelength-shifting processes. Tables are interpolated linearly in wavelength
with the native piecewise-linear substrate; wavelengths outside the grid are
reported unsupported and never extrapolated.

Polarized scattering uses the amplitude functions ``S1`` (perpendicular) and
``S2`` (parallel) of Bohren and Huffman (time dependence ``exp(-i omega t)``,
absorbing index ``m = n + i kappa``). For a unit Jones vector ``J`` with
carried-frame Stokes parameters ``Q`` and ``U`` the joint density of the
scattering angle ``theta`` and azimuth ``phi`` factors exactly into the
unpolarized phase function ``(|S1|^2 + |S2|^2) / 2`` in ``theta`` and the
conditional azimuth density ``(1 - a cos 2(phi - psi)) / (2 pi)`` with
``a = L (|S1|^2 - |S2|^2) / (|S1|^2 + |S2|^2)``, ``L = sqrt(Q^2 + U^2)``, and
``psi = atan2(U, Q) / 2``. Both factors are sampled exactly and the scattered
Jones vector is ``(S1 J_s, S2 J_p)`` normalized, so the weight carries power
and the Jones vector carries the polarization state.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._interpolation import linear_interpolate
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import (
    as_array,
    as_host_array,
    Complex128,
    ConvertibleToArray,
    Dim,
    Float64,
    HostComplex128,
    HostFloat64,
    Int32,
    parse,
    Scalar,
    Scope,
    Size,
)
from ._optical_monte_carlo import (
    _henyey_greenstein_cosine,
    OpticalScatteringSample,
    rotate_jones_to_scattering_frame,
)


WavelengthShiftDelay: TypeAlias = Literal["exponential", "delta"]


class _MediumDim(Dim, minimum=1):
    """Number of homogeneous media."""


class _WavelengthDim(Dim, minimum=2):
    """Nodes of the shared vacuum-wavelength grid."""


class _EmissionDim(Dim, minimum=2):
    """Nodes of the wavelength-shifter emission grid."""


class _AngleDim(Dim, minimum=3):
    """Nodes of the Lorenz--Mie scattering-cosine grid."""


class _TermDim(Dim, minimum=1):
    """Retained Lorenz--Mie multipole orders."""


class _CosineDim(Dim):
    """Requested scattering cosines."""


class _Channel(IntEnum):
    """Columns of the packed per-medium spectral table."""

    REFRACTIVE_INDEX = 0
    ABSORPTION = 1
    RAYLEIGH = 2
    HENYEY_GREENSTEIN = 3
    ANISOTROPY = 4
    MIE_SCATTERING = 5
    MIE_ABSORPTION = 6
    WAVELENGTH_SHIFTING = 7
    GRID_POSITION = 8


class _Process(IntEnum):
    """Volume process selected at one event."""

    RAYLEIGH = 0
    HENYEY_GREENSTEIN = 1
    MIE = 2
    WAVELENGTH_SHIFTING = 3


# Lentz continued fractions converge to this relative increment.
_LENTZ_TOLERANCE = 1e-15
_LENTZ_MAXIMUM_TERMS = 10_000_000
_TINY = 1e-300


# --------------------------------------------------------------------------- #
# Lorenz--Mie series (host preparation) and amplitude sums (JAX).
# --------------------------------------------------------------------------- #


def _wiscombe_term_count(size_parameter: float) -> int:
    """Wiscombe (1980) series truncation ``N_stop`` for size parameter ``x``."""

    x = size_parameter
    if x <= 8.0:
        return math.floor(x + 4.0 * x ** (1.0 / 3.0) + 1.0)
    if x < 4200.0:
        return math.floor(x + 4.05 * x ** (1.0 / 3.0) + 2.0)
    return math.floor(x + 4.0 * x ** (1.0 / 3.0) + 2.0)


def _lentz_logarithmic_derivative(order: int, argument: complex) -> tuple[complex, int]:
    """``D_n(z) = psi_n'(z) / psi_n(z)`` by Lentz's (1976) continued fraction.

    ``D_n(z) = -n / z + J_{n-1/2}(z) / J_{n+1/2}(z)`` and the Bessel ratio is
    ``a_1 + 1 / (a_2 + 1 / (a_3 + ...))`` with
    ``a_k = (-1)^(k+1) (2 n + 2 k - 1) / z``, evaluated with the modified Lentz
    recurrence until the multiplicative increment is within one ulp of one.
    """

    def partial(index: int) -> complex:
        sign = 1.0 if index % 2 == 1 else -1.0
        return sign * (2 * order + 2 * index - 1) / argument

    value = partial(1)
    numerator = value if value != 0.0 else complex(_TINY)
    denominator = 0j
    for index in range(2, _LENTZ_MAXIMUM_TERMS):
        term = partial(index)
        denominator = term + denominator
        denominator = 1.0 / (denominator if denominator != 0.0 else complex(_TINY))
        numerator = term + 1.0 / numerator
        numerator = numerator if numerator != 0.0 else complex(_TINY)
        increment = numerator * denominator
        value *= increment
        if abs(increment - 1.0) < _LENTZ_TOLERANCE:
            return -order / argument + value, index
    raise ArithmeticError(
        f"Lentz continued fraction for D_{order}({argument}) did not converge."
    )


def _mie_coefficients(
    size_parameter: float, relative_index: complex
) -> tuple[np.ndarray, np.ndarray, int]:
    """External coefficients ``a_n``, ``b_n`` for ``n = 1 .. N_stop``.

    ``D_n(m x)`` is recurred downward from ``N_stop`` (Wiscombe 1980), started
    exactly by Lentz's continued fraction, which is stable for every complex
    ``m x``; ``psi_n`` and ``chi_n`` of the real size parameter are recurred
    upward, which is stable up to ``N_stop``.
    """

    x = size_parameter
    m = relative_index
    count = _wiscombe_term_count(x)
    argument = m * x
    derivative = np.zeros(count + 1, dtype=np.complex128)
    derivative[count], iterations = _lentz_logarithmic_derivative(count, argument)
    for order in range(count, 0, -1):
        ratio = order / argument
        derivative[order - 1] = ratio - 1.0 / (derivative[order] + ratio)
    a = np.zeros(count, dtype=np.complex128)
    b = np.zeros(count, dtype=np.complex128)
    psi_previous, psi_current = math.cos(x), math.sin(x)
    chi_previous, chi_current = -math.sin(x), math.cos(x)
    for order in range(1, count + 1):
        psi = (2 * order - 1) / x * psi_current - psi_previous
        chi = (2 * order - 1) / x * chi_current - chi_previous
        xi = complex(psi, -chi)
        xi_previous = complex(psi_current, -chi_current)
        electric = derivative[order] / m + order / x
        magnetic = m * derivative[order] + order / x
        a[order - 1] = (electric * psi - psi_current) / (electric * xi - xi_previous)
        b[order - 1] = (magnetic * psi - psi_current) / (magnetic * xi - xi_previous)
        psi_previous, psi_current = psi_current, psi
        chi_previous, chi_current = chi_current, chi
    return a, b, iterations


def _mie_efficiencies(
    size_parameter: float, a: np.ndarray, b: np.ndarray
) -> tuple[float, float, float, float, float]:
    """Extinction, scattering, backscattering efficiencies, asymmetry, tail share."""

    x2 = size_parameter * size_parameter
    orders = np.arange(1, a.shape[0] + 1, dtype=np.float64)
    weights = 2.0 * orders + 1.0
    extinction = 2.0 / x2 * float(np.sum(weights * np.real(a + b)))
    partial = weights * (np.abs(a) ** 2 + np.abs(b) ** 2)
    scattering = 2.0 / x2 * float(np.sum(partial))
    adjacent = orders[:-1] * (orders[:-1] + 2.0) / (orders[:-1] + 1.0)
    cross = float(
        np.sum(adjacent * np.real(a[:-1] * np.conj(a[1:]) + b[:-1] * np.conj(b[1:])))
        + np.sum(weights / (orders * (orders + 1.0)) * np.real(a * np.conj(b)))
    )
    asymmetry = 4.0 / (x2 * scattering) * cross if scattering > 0.0 else 0.0
    alternating = np.where(orders % 2.0 == 0.0, 1.0, -1.0)
    backscattering = abs(complex(np.sum(weights * alternating * (a - b)))) ** 2 / x2
    tail = float(partial[-1] / np.sum(partial)) if scattering > 0.0 else 0.0
    return extinction, scattering, backscattering, asymmetry, tail


def _mie_amplitudes(a: Array, b: Array, cosines: Array) -> tuple[Array, Array]:
    """``S1`` and ``S2`` by the upward ``pi_n``/``tau_n`` recurrences.

    Zero-padded coefficients beyond a node's ``N_stop`` contribute nothing,
    so one static order count serves every table row.
    """

    orders = jnp.arange(1, a.shape[0] + 1, dtype=jnp.float64)
    zero = jnp.zeros(cosines.shape, dtype=jnp.complex128)

    def order_step(
        carry: tuple[Array, Array, Array, Array],
        inputs: tuple[Array, Array, Array],
    ) -> tuple[tuple[Array, Array, Array, Array], None]:
        previous, current, s1, s2 = carry
        order, a_n, b_n = inputs
        tau = order * cosines * current - (order + 1.0) * previous
        factor = (2.0 * order + 1.0) / (order * (order + 1.0))
        s1 = s1 + factor * (a_n * current + b_n * tau)
        s2 = s2 + factor * (a_n * tau + b_n * current)
        following = (
            (2.0 * order + 1.0) * cosines * current - (order + 1.0) * previous
        ) / order
        return (current, following, s1, s2), None

    initial = (
        jnp.zeros(cosines.shape, dtype=jnp.float64),
        jnp.ones(cosines.shape, dtype=jnp.float64),
        zero,
        zero,
    )
    (_, _, s1, s2), _ = jax.lax.scan(order_step, initial, (orders, a, b))
    return s1, s2


def _validated_particle(size_parameter: float, relative_index: complex) -> None:
    if not math.isfinite(size_parameter) or size_parameter <= 0.0:
        raise ValueError("Lorenz--Mie size parameters must be finite and positive.")
    if (
        not (math.isfinite(relative_index.real) and math.isfinite(relative_index.imag))
        or relative_index.real <= 0.0
        or relative_index.imag < 0.0
    ):
        raise ValueError(
            "Lorenz--Mie relative indices need a positive real part and a "
            "non-negative imaginary part (m = n + i kappa)."
        )


class LorenzMieResult(StrictModule, NonTrainableState):
    """Efficiencies, amplitude functions, and series evidence of one sphere.

    ``s1``/``s2`` are the Bohren--Huffman amplitude functions at the requested
    scattering cosines; Wiscombe's (1979) tabulated values are their complex
    conjugates because MIEV0 uses ``m = n - i kappa``. ``term_count`` is
    Wiscombe's ``N_stop``, ``lentz_iterations`` the continued-fraction length
    that started the downward ``D_n`` recurrence, and
    ``truncation_contribution`` the share of the scattering efficiency carried
    by the last retained order.
    """

    __strict_contract__ = True

    extinction_efficiency: Float64[Scalar]
    scattering_efficiency: Float64[Scalar]
    absorption_efficiency: Float64[Scalar]
    backscattering_efficiency: Float64[Scalar]
    asymmetry: Float64[Scalar]
    s1: Complex128[_CosineDim]
    s2: Complex128[_CosineDim]
    a_coefficients: Complex128[_TermDim]
    b_coefficients: Complex128[_TermDim]
    size_parameter: float = eqx.field(static=True)
    relative_index: complex = eqx.field(static=True)
    term_count: Size[_TermDim] = eqx.field(static=True)
    lentz_iterations: int = eqx.field(static=True)
    truncation_contribution: float = eqx.field(static=True)


def lorenz_mie(
    size_parameter: float,
    relative_index: complex,
    cosines: ConvertibleToArray,
    /,
) -> LorenzMieResult:
    """Lorenz--Mie solution for a homogeneous sphere (Bohren--Huffman convention).

    ``size_parameter`` is ``x = 2 pi r n_medium / lambda_vacuum`` and
    ``relative_index`` is ``m = n_particle / n_medium`` with non-negative
    imaginary part. The series uses Wiscombe's truncation, Lentz-started
    downward recurrence for ``D_n(m x)``, and upward Riccati--Bessel
    recurrences; ``cosines`` select the scattering angles of ``s1``/``s2``.
    """

    x = float(size_parameter)
    m = complex(relative_index)
    _validated_particle(x, m)
    host_cosines = as_host_array(cosines, HostFloat64[_CosineDim], "cosines")
    if not np.all(np.isfinite(host_cosines)) or np.any(np.abs(host_cosines) > 1.0):
        raise ValueError("cosines must be finite and lie in [-1, 1].")
    a, b, iterations = _mie_coefficients(x, m)
    extinction, scattering, backscattering, asymmetry, tail = _mie_efficiencies(x, a, b)
    a_device = jnp.asarray(a)
    b_device = jnp.asarray(b)
    s1, s2 = _mie_amplitudes(a_device, b_device, jnp.asarray(host_cosines))
    return LorenzMieResult(
        jnp.asarray(extinction, dtype=jnp.float64),
        jnp.asarray(scattering, dtype=jnp.float64),
        jnp.asarray(extinction - scattering, dtype=jnp.float64),
        jnp.asarray(backscattering, dtype=jnp.float64),
        jnp.asarray(asymmetry, dtype=jnp.float64),
        s1,
        s2,
        a_device,
        b_device,
        x,
        m,
        a.shape[0],
        iterations,
        tail,
    )


# --------------------------------------------------------------------------- #
# Process descriptions.
# --------------------------------------------------------------------------- #


def _inverse_lengths(lengths: np.ndarray, name: str, /) -> np.ndarray:
    """Coefficients ``1 / L`` from positive lengths; ``+inf`` means absent."""

    if np.any(np.isnan(lengths)) or np.any(lengths <= 0.0):
        raise ValueError(f"{name} must be positive; +inf denotes an absent process.")
    finite = np.isfinite(lengths)
    return np.where(finite, 1.0 / np.where(finite, lengths, 1.0), 0.0)


class RayleighScattering(StrictModule, NonTrainableState):
    """Point-dipole scattering with tabulated scattering lengths per medium.

    Amplitudes are ``S1 = 1`` and ``S2 = cos(theta)``: the unpolarized phase
    function is ``3 (1 + cos^2 theta) / (16 pi)`` and a linearly polarized
    photon is never scattered along its field.
    """

    __strict_contract__ = True

    scattering_lengths: Float64[_MediumDim, _WavelengthDim]

    def __init__(self, scattering_lengths: ConvertibleToArray, /) -> None:
        self.scattering_lengths = as_array(
            scattering_lengths,
            Float64[_MediumDim, _WavelengthDim],
            "scattering_lengths",
        )


class HenyeyGreensteinScattering(StrictModule, NonTrainableState):
    """Scalar Henyey--Greenstein scattering with tabulated lengths and ``g``.

    Amplitudes are ``S1 = S2``: the Jones components on the scattering-plane
    frame are unchanged, exactly as in `TissueOpticalMedium`.
    """

    __strict_contract__ = True

    scattering_lengths: Float64[_MediumDim, _WavelengthDim]
    anisotropy: Float64[_MediumDim, _WavelengthDim]

    def __init__(
        self, scattering_lengths: ConvertibleToArray, anisotropy: ConvertibleToArray, /
    ) -> None:
        scope = Scope()
        self.scattering_lengths = as_array(
            scattering_lengths,
            Float64[_MediumDim, _WavelengthDim],
            "scattering_lengths",
            scope=scope,
        )
        self.anisotropy = as_array(
            anisotropy, Float64[_MediumDim, _WavelengthDim], "anisotropy", scope=scope
        )


class MieParticles(StrictModule, NonTrainableState):
    """Monodisperse homogeneous spheres suspended in each medium.

    ``radii`` are in the transport length unit, ``number_densities`` per cubic
    transport length, and ``refractive_indices`` the complex particle indices
    ``n + i kappa`` on the medium's wavelength grid. ``length_per_wavelength_unit``
    is the transport length of one wavelength unit (for example ``100`` for
    wavelengths in metres and lengths in centimetres); it is the only link
    between the two unit systems. ``angle_count`` is the number of scattering
    angles, uniform in ``theta``, of the inverse-CDF tables (default
    ``max(1025, 32 N_stop + 1)`` for the largest node); tables whose
    normalization or asymmetry deviates from the series by more than
    ``table_tolerance`` are refused.
    """

    __strict_contract__ = True

    radii: Float64[_MediumDim]
    refractive_indices: Complex128[_MediumDim, _WavelengthDim]
    number_densities: Float64[_MediumDim]
    length_per_wavelength_unit: float = eqx.field(static=True)
    angle_count: int | None = eqx.field(static=True)
    table_tolerance: float = eqx.field(static=True)

    def __init__(
        self,
        radii: ConvertibleToArray,
        refractive_indices: ConvertibleToArray,
        number_densities: ConvertibleToArray,
        /,
        *,
        length_per_wavelength_unit: float,
        angle_count: int | None = None,
        table_tolerance: float = 1e-3,
    ) -> None:
        scope = Scope()
        self.radii = as_array(radii, Float64[_MediumDim], "radii", scope=scope)
        self.refractive_indices = as_array(
            refractive_indices,
            Complex128[_MediumDim, _WavelengthDim],
            "refractive_indices",
            scope=scope,
        )
        self.number_densities = as_array(
            number_densities, Float64[_MediumDim], "number_densities", scope=scope
        )
        scale = float(length_per_wavelength_unit)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("length_per_wavelength_unit must be finite and positive.")
        if angle_count is not None and (
            isinstance(angle_count, bool) or not isinstance(angle_count, int)
        ):
            raise TypeError("angle_count must be an int or None.")
        if angle_count is not None and angle_count < 3:
            raise ValueError("angle_count must be at least three.")
        tolerance = float(table_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("table_tolerance must be finite and positive.")
        self.length_per_wavelength_unit = scale
        self.angle_count = angle_count
        self.table_tolerance = tolerance


class WavelengthShifter(StrictModule, NonTrainableState):
    """Wavelength-shifting absorption, emission spectrum, yield, and delay.

    ``absorption_lengths`` are on the medium's wavelength grid;
    ``emission_spectra`` are non-negative spectral densities per unit
    wavelength on ``emission_wavelengths``, interpreted as piecewise linear and
    sampled exactly. Each absorbed photon is re-emitted isotropically and
    unpolarized with probability ``quantum_yields`` after a delay that is
    exponential with mean ``delay_times`` or exactly ``delay_times``
    (``delay="delta"``).
    """

    __strict_contract__ = True

    absorption_lengths: Float64[_MediumDim, _WavelengthDim]
    emission_wavelengths: Float64[_EmissionDim]
    emission_spectra: Float64[_MediumDim, _EmissionDim]
    quantum_yields: Float64[_MediumDim]
    delay_times: Float64[_MediumDim]
    delay: WavelengthShiftDelay = eqx.field(static=True)

    def __init__(
        self,
        absorption_lengths: ConvertibleToArray,
        emission_wavelengths: ConvertibleToArray,
        emission_spectra: ConvertibleToArray,
        quantum_yields: ConvertibleToArray,
        delay_times: ConvertibleToArray,
        /,
        *,
        delay: WavelengthShiftDelay = "exponential",
    ) -> None:
        scope = Scope()
        self.absorption_lengths = as_array(
            absorption_lengths,
            Float64[_MediumDim, _WavelengthDim],
            "absorption_lengths",
            scope=scope,
        )
        self.emission_wavelengths = as_array(
            emission_wavelengths,
            Float64[_EmissionDim],
            "emission_wavelengths",
            scope=scope,
        )
        self.emission_spectra = as_array(
            emission_spectra,
            Float64[_MediumDim, _EmissionDim],
            "emission_spectra",
            scope=scope,
        )
        self.quantum_yields = as_array(
            quantum_yields, Float64[_MediumDim], "quantum_yields", scope=scope
        )
        self.delay_times = as_array(
            delay_times, Float64[_MediumDim], "delay_times", scope=scope
        )
        self.delay = parse(delay, WavelengthShiftDelay, "delay")


# --------------------------------------------------------------------------- #
# Prepared tables.
# --------------------------------------------------------------------------- #


def _piecewise_linear_cdf(nodes: ArrayLike, densities: ArrayLike) -> Array:
    """Normalized trapezoid CDF along the last axis; zero rows stay zero.

    Host preparation and device kernels share this one construction.
    """

    grid = jnp.asarray(nodes, dtype=jnp.float64)
    values = jnp.asarray(densities, dtype=jnp.float64)
    segments = 0.5 * jnp.diff(grid) * (values[..., :-1] + values[..., 1:])
    cumulative = jnp.concatenate(
        (
            jnp.zeros(values.shape[:-1] + (1,), dtype=jnp.float64),
            jnp.cumsum(segments, axis=-1),
        ),
        axis=-1,
    )
    total = cumulative[..., -1:]
    return jnp.where(total > 0.0, cumulative / jnp.where(total > 0.0, total, 1.0), 0.0)


def _sample_piecewise_linear(
    nodes: Array, densities: Array, cdf: Array, uniforms: Array
) -> Array:
    """Exact inverse CDF of piecewise-linear densities, one row per lane.

    Within a segment the density ``f0 + s t`` integrates to
    ``f0 t + s t^2 / 2``; the root is taken in the cancellation-free form
    ``2 r / (f0 + sqrt(f0^2 + 2 s r))``.
    """

    last = nodes.shape[0] - 2

    def one(density: Array, cumulative: Array, uniform: Array) -> Array:
        index = jnp.clip(jnp.searchsorted(cumulative, uniform, side="right") - 1, 0, last)
        start = nodes[index]
        width = nodes[index + 1] - start
        first = density[index]
        slope = (density[index + 1] - first) / width
        residual = jnp.maximum(uniform - cumulative[index], 0.0)
        root = jnp.sqrt(jnp.maximum(first * first + 2.0 * slope * residual, 0.0))
        denominator = first + root
        offset = jnp.where(denominator > 0.0, 2.0 * residual / denominator, 0.0)
        return start + jnp.clip(offset, 0.0, width)

    return jax.vmap(one)(densities, cdf, uniforms)


class MieTableEvidence(StrictModule, NonTrainableState):
    """Series and table evidence of the prepared Lorenz--Mie nodes.

    ``normalization_residuals`` compare the scattering efficiency integrated
    from the tabulated phase function with the series value, and
    ``asymmetry_residuals`` the tabulated and series asymmetry parameters.
    """

    __strict_contract__ = True

    size_parameters: Float64[_MediumDim, _WavelengthDim]
    relative_indices: Complex128[_MediumDim, _WavelengthDim]
    term_counts: Int32[_MediumDim, _WavelengthDim]
    lentz_iterations: Int32[_MediumDim, _WavelengthDim]
    truncation_contributions: Float64[_MediumDim, _WavelengthDim]
    scattering_efficiencies: Float64[_MediumDim, _WavelengthDim]
    extinction_efficiencies: Float64[_MediumDim, _WavelengthDim]
    asymmetry: Float64[_MediumDim, _WavelengthDim]
    normalization_residuals: Float64[_MediumDim, _WavelengthDim]
    asymmetry_residuals: Float64[_MediumDim, _WavelengthDim]
    angle_count: int = eqx.field(static=True)
    maximum_term_count: int = eqx.field(static=True)
    table_bytes: int = eqx.field(static=True)


class _MieTables(StrictModule, NonTrainableState):
    __strict_contract__ = True

    a: Complex128[_MediumDim, _WavelengthDim, _TermDim]
    b: Complex128[_MediumDim, _WavelengthDim, _TermDim]
    cosines: Float64[_AngleDim]
    densities: Float64[_MediumDim, _WavelengthDim, _AngleDim]
    cdf: Float64[_MediumDim, _WavelengthDim, _AngleDim]


class _EmissionTables(StrictModule, NonTrainableState):
    __strict_contract__ = True

    wavelengths: Float64[_EmissionDim]
    densities: Float64[_MediumDim, _EmissionDim]
    cdf: Float64[_MediumDim, _EmissionDim]
    quantum_yields: Float64[_MediumDim]
    delay_times: Float64[_MediumDim]
    delay: WavelengthShiftDelay = eqx.field(static=True)


def _prepare_mie(
    particles: MieParticles,
    wavelengths: np.ndarray,
    medium_indices: np.ndarray,
    scope: Scope,
) -> tuple[_MieTables, MieTableEvidence, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Series coefficients, inverse-CDF tables, coefficients, and evidence."""

    radii = as_host_array(particles.radii, HostFloat64[_MediumDim], "radii", scope=scope)
    particle_indices = as_host_array(
        particles.refractive_indices,
        HostComplex128[_MediumDim, _WavelengthDim],
        "refractive_indices",
        scope=scope,
    )
    densities = as_host_array(
        particles.number_densities,
        HostFloat64[_MediumDim],
        "number_densities",
        scope=scope,
    )
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("Mie particle radii must be finite and positive.")
    if not np.all(np.isfinite(densities)) or np.any(densities < 0.0):
        raise ValueError("Mie number densities must be finite and non-negative.")
    media, nodes = medium_indices.shape
    size_parameters = (
        2.0
        * np.pi
        * radii[:, None]
        * medium_indices
        / (wavelengths[None, :] * particles.length_per_wavelength_unit)
    )
    relative = particle_indices / medium_indices
    series = [
        [
            _mie_series_node(float(size_parameters[i, j]), complex(relative[i, j]))
            for j in range(nodes)
        ]
        for i in range(media)
    ]
    terms = max(node[0].shape[0] for row in series for node in row)
    count = (
        max(1025, 32 * terms + 1)
        if particles.angle_count is None
        else particles.angle_count
    )
    a = np.zeros((media, nodes, terms), dtype=np.complex128)
    b = np.zeros((media, nodes, terms), dtype=np.complex128)
    for i in range(media):
        for j in range(nodes):
            node_terms = series[i][j][0].shape[0]
            a[i, j, :node_terms] = series[i][j][0]
            b[i, j, :node_terms] = series[i][j][1]
    efficiencies = np.asarray(
        [[node[3] for node in row] for row in series], dtype=np.float64
    )
    extinction_efficiency = efficiencies[..., 0]
    scattering_efficiency = efficiencies[..., 1]
    asymmetry = efficiencies[..., 3]
    cosines, normalized, normalization, asymmetry_residual = _mie_phase_tables(
        a,
        b,
        size_parameters,
        scattering_efficiency,
        asymmetry,
        count,
        particles.table_tolerance,
    )
    geometric = np.pi * radii * radii
    scattering = densities[:, None] * geometric[:, None] * scattering_efficiency
    # Q_ext - Q_sca is zero for real m up to series roundoff of either sign.
    particle_absorption = (
        densities[:, None]
        * geometric[:, None]
        * np.maximum(extinction_efficiency - scattering_efficiency, 0.0)
    )
    tables = _MieTables(
        jnp.asarray(a),
        jnp.asarray(b),
        jnp.asarray(cosines),
        jnp.asarray(normalized),
        jnp.asarray(_piecewise_linear_cdf(cosines, normalized)),
    )
    evidence = MieTableEvidence(
        jnp.asarray(size_parameters),
        jnp.asarray(relative),
        jnp.asarray([[node[0].shape[0] for node in row] for row in series], jnp.int32),
        jnp.asarray([[node[2] for node in row] for row in series], jnp.int32),
        jnp.asarray(efficiencies[..., 4]),
        jnp.asarray(scattering_efficiency),
        jnp.asarray(extinction_efficiency),
        jnp.asarray(asymmetry),
        jnp.asarray(normalization),
        jnp.asarray(asymmetry_residual),
        count,
        terms,
        2 * a.nbytes + 2 * normalized.nbytes + cosines.nbytes,
    )
    content = {
        "a": a,
        "b": b,
        "cosines": cosines,
        "radii": radii,
        "number_densities": densities,
        "particle_indices": particle_indices,
    }
    return tables, evidence, scattering, particle_absorption, content


def _mie_phase_tables(
    a: np.ndarray,
    b: np.ndarray,
    size_parameters: np.ndarray,
    scattering_efficiency: np.ndarray,
    asymmetry: np.ndarray,
    count: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalized piecewise-linear phase densities and their series residuals.

    The density is linear in the scattering cosine between grid nodes uniform
    in the scattering angle, so its normalization is the trapezoid sum and
    its first moment is exact per segment; both are compared with the series
    and a table outside ``tolerance`` is refused.
    """

    media, nodes, terms = a.shape
    cosines = np.cos(np.linspace(np.pi, 0.0, count))
    cosines[0], cosines[-1] = -1.0, 1.0
    device_cosines = jnp.asarray(cosines)

    def intensity(row: tuple[Array, Array]) -> Array:
        s1, s2 = _mie_amplitudes(row[0], row[1], device_cosines)
        return 0.5 * (jnp.abs(s1) ** 2 + jnp.abs(s2) ** 2)

    # Bounded host preparation: one node's angular grid at a time.
    phase = np.asarray(
        jax.lax.map(
            intensity,
            (
                jnp.asarray(a.reshape((media * nodes, terms))),
                jnp.asarray(b.reshape((media * nodes, terms))),
            ),
        )
    ).reshape((media, nodes, count))
    widths = np.diff(cosines)
    mass = np.sum(0.5 * widths * (phase[..., :-1] + phase[..., 1:]), axis=-1)
    moment = np.sum(
        widths
        / 6.0
        * (
            phase[..., :-1] * (2.0 * cosines[:-1] + cosines[1:])
            + phase[..., 1:] * (cosines[:-1] + 2.0 * cosines[1:])
        ),
        axis=-1,
    )
    normalization = np.abs(
        2.0 * mass / (size_parameters * size_parameters) / scattering_efficiency - 1.0
    )
    asymmetry_residual = np.abs(moment / mass - asymmetry)
    worst = max(float(np.max(normalization)), float(np.max(asymmetry_residual)))
    if not math.isfinite(worst) or worst > tolerance:
        raise ValueError(
            "Lorenz--Mie angular tables miss the series by "
            f"{worst:.3e} > table_tolerance {tolerance:.3e}; increase angle_count."
        )
    return cosines, phase / mass[..., None], normalization, asymmetry_residual


def _mie_series_node(
    size_parameter: float, relative_index: complex
) -> tuple[np.ndarray, np.ndarray, int, tuple[float, float, float, float, float]]:
    _validated_particle(size_parameter, relative_index)
    a, b, iterations = _mie_coefficients(size_parameter, relative_index)
    extinction, scattering, backscattering, asymmetry, tail = _mie_efficiencies(
        size_parameter, a, b
    )
    return a, b, iterations, (extinction, scattering, backscattering, asymmetry, tail)


def _prepare_emission(
    shifter: WavelengthShifter,
    wavelengths: np.ndarray,
    scope: Scope,
) -> tuple[_EmissionTables, np.ndarray, dict[str, np.ndarray]]:
    absorption = as_host_array(
        shifter.absorption_lengths,
        HostFloat64[_MediumDim, _WavelengthDim],
        "wavelength_shifter.absorption_lengths",
        scope=scope,
    )
    emission_grid = as_host_array(
        shifter.emission_wavelengths,
        HostFloat64[_EmissionDim],
        "emission_wavelengths",
    )
    spectra = as_host_array(
        shifter.emission_spectra,
        HostFloat64[_MediumDim, _EmissionDim],
        "emission_spectra",
        scope=scope,
    )
    yields = as_host_array(
        shifter.quantum_yields, HostFloat64[_MediumDim], "quantum_yields", scope=scope
    )
    delays = as_host_array(
        shifter.delay_times, HostFloat64[_MediumDim], "delay_times", scope=scope
    )
    coefficients = _inverse_lengths(absorption, "wavelength_shifter.absorption_lengths")
    if not np.all(np.isfinite(emission_grid)) or np.any(np.diff(emission_grid) <= 0.0):
        raise ValueError("emission_wavelengths must be finite and strictly increasing.")
    if emission_grid[0] < wavelengths[0] or emission_grid[-1] > wavelengths[-1]:
        raise ValueError(
            "emission_wavelengths must lie inside the medium wavelength grid."
        )
    if not np.all(np.isfinite(spectra)) or np.any(spectra < 0.0):
        raise ValueError("emission_spectra must be finite and non-negative.")
    if not np.all(np.isfinite(yields)) or np.any((yields < 0.0) | (yields > 1.0)):
        raise ValueError("quantum_yields must lie in [0, 1].")
    if not np.all(np.isfinite(delays)) or np.any(delays < 0.0):
        raise ValueError("delay_times must be finite and non-negative.")
    widths = np.diff(emission_grid)
    totals = np.sum(0.5 * widths * (spectra[:, :-1] + spectra[:, 1:]), axis=-1)
    shifting = np.any(coefficients > 0.0, axis=-1)
    if np.any(shifting & (totals <= 0.0)):
        raise ValueError(
            "Every wavelength-shifting medium needs an emission spectrum with "
            "positive integral."
        )
    densities = spectra / np.where(totals > 0.0, totals, 1.0)[:, None]
    tables = _EmissionTables(
        jnp.asarray(emission_grid),
        jnp.asarray(densities),
        jnp.asarray(_piecewise_linear_cdf(emission_grid, densities)),
        jnp.asarray(yields),
        jnp.asarray(delays),
        shifter.delay,
    )
    content = {
        "emission_wavelengths": emission_grid,
        "emission_spectra": spectra,
        "quantum_yields": yields,
        "delay_times": delays,
    }
    return tables, coefficients, content


# --------------------------------------------------------------------------- #
# Medium.
# --------------------------------------------------------------------------- #


def _uniform_rows(keys: Array, count: int) -> Array:
    return jax.vmap(lambda key: jr.uniform(key, (count,), dtype=jnp.float64))(keys)


def _normal_rows(keys: Array, count: int) -> Array:
    return jax.vmap(lambda key: jr.normal(key, (count,), dtype=jnp.float64))(keys)


def _rayleigh_cosine(uniforms: Array) -> Array:
    """Exact inverse CDF of ``(1 + mu^2)``: the real root of ``mu^3 + 3 mu = 8u - 4``."""

    shifted = 4.0 * uniforms - 2.0
    root = jnp.cbrt(shifted + jnp.sqrt(shifted * shifted + 1.0))
    return jnp.clip(root - 1.0 / root, -1.0, 1.0)


def _polarized_azimuth(
    jones_vectors: Array,
    s1: Array,
    s2: Array,
    uniforms: Array,
    normals: Array,
) -> Array:
    """Exact azimuth draw from ``(1 - a cos 2(phi - psi)) / (2 pi)``.

    The density is the mixture of the uniform law (weight ``1 - |a|``) and
    ``sin^2`` (``a > 0``) or ``cos^2`` (``a < 0``) of ``phi - psi``. The
    ``sin^2`` law is the polar angle of ``(g, c)`` with ``g`` standard normal
    and ``c`` signed chi with three degrees of freedom, whose joint density
    ``c^2 exp(-(g^2 + c^2) / 2)`` factors into ``sin^2`` times a radial law.
    """

    first = jones_vectors[:, 0]
    second = jones_vectors[:, 1]
    stokes_q = jnp.abs(first) ** 2 - jnp.abs(second) ** 2
    stokes_u = 2.0 * jnp.real(jnp.conj(first) * second)
    linear = jnp.sqrt(stokes_q * stokes_q + stokes_u * stokes_u)
    perpendicular = jnp.abs(s1) ** 2
    parallel = jnp.abs(s2) ** 2
    total = perpendicular + parallel
    contrast = jnp.clip(
        linear * (perpendicular - parallel) / jnp.where(total > 0.0, total, 1.0),
        -1.0,
        1.0,
    )
    orientation = 0.5 * jnp.arctan2(stokes_u, stokes_q)
    chi = jnp.sqrt(jnp.sum(normals[:, :3] ** 2, axis=-1))
    signed = jnp.where(normals[:, 3] >= 0.0, chi, -chi)
    squared_sine = jnp.arctan2(signed, normals[:, 4])
    peaked = jnp.where(contrast > 0.0, squared_sine, squared_sine + 0.5 * jnp.pi)
    offset = jnp.where(
        uniforms[:, 0] < jnp.abs(contrast), peaked, 2.0 * jnp.pi * uniforms[:, 1]
    )
    return orientation + offset


class SpectralOpticalMedium(StrictModule, NonTrainableState):
    """Tabulated spectral media with polarized volume processes.

    ``wavelengths`` is the strictly increasing vacuum-wavelength grid shared
    by every table; ``refractive_indices`` are real and positive and
    ``absorption_lengths`` positive (``+inf`` for transparent entries), both of
    shape ``(media, wavelengths)``. Coefficients (inverse lengths) and the
    Henyey--Greenstein ``g`` are interpolated linearly in wavelength with the
    native piecewise-linear substrate; Lorenz--Mie angular laws between two
    nodes are the interpolation-weighted mixture of the two node laws.
    Wavelengths outside the grid are unsupported.

    The extinction is the sum of bulk absorption, Rayleigh, Henyey--Greenstein,
    Mie extinction, and wavelength-shifter absorption; the albedo counts
    elastic scattering and quantum-yield-weighted re-emission. `mie_evidence`
    reports the series truncation and angular-table evidence.
    """

    __strict_contract__ = True

    wavelengths: Float64[_WavelengthDim]
    table: Float64[_MediumDim, _WavelengthDim, Literal[9]]
    mie_tables: _MieTables | None
    mie_evidence: MieTableEvidence | None
    emission: _EmissionTables | None
    medium_count: Size[_MediumDim] = eqx.field(static=True)
    medium_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavelengths: ConvertibleToArray,
        refractive_indices: ConvertibleToArray,
        absorption_lengths: ConvertibleToArray,
        /,
        *,
        rayleigh: RayleighScattering | None = None,
        henyey_greenstein: HenyeyGreensteinScattering | None = None,
        mie: MieParticles | None = None,
        wavelength_shifter: WavelengthShifter | None = None,
    ) -> None:
        scope = Scope()
        grid = as_host_array(
            wavelengths, HostFloat64[_WavelengthDim], "wavelengths", scope=scope
        )
        index = as_host_array(
            refractive_indices,
            HostFloat64[_MediumDim, _WavelengthDim],
            "refractive_indices",
            scope=scope,
        )
        absorption = as_host_array(
            absorption_lengths,
            HostFloat64[_MediumDim, _WavelengthDim],
            "absorption_lengths",
            scope=scope,
        )
        if (
            not np.all(np.isfinite(grid))
            or grid[0] <= 0.0
            or np.any(np.diff(grid) <= 0.0)
        ):
            raise ValueError("wavelengths must be positive and strictly increasing.")
        if not np.all(np.isfinite(index)) or np.any(index <= 0.0):
            raise ValueError("refractive_indices must be finite and positive.")
        media, nodes = index.shape
        channels = np.zeros((media, nodes, len(_Channel)), dtype=np.float64)
        channels[..., _Channel.REFRACTIVE_INDEX] = index
        channels[..., _Channel.ABSORPTION] = _inverse_lengths(
            absorption, "absorption_lengths"
        )
        channels[..., _Channel.GRID_POSITION] = np.arange(nodes, dtype=np.float64)
        content: dict[str, object] = {
            "wavelengths": grid,
            "refractive_indices": index,
            "absorption_lengths": absorption,
        }
        if rayleigh is not None:
            if not isinstance(rayleigh, RayleighScattering):
                raise TypeError("rayleigh must be a RayleighScattering.")
            lengths = as_host_array(
                rayleigh.scattering_lengths,
                HostFloat64[_MediumDim, _WavelengthDim],
                "rayleigh.scattering_lengths",
                scope=scope,
            )
            channels[..., _Channel.RAYLEIGH] = _inverse_lengths(
                lengths, "rayleigh.scattering_lengths"
            )
            content["rayleigh"] = lengths
        if henyey_greenstein is not None:
            if not isinstance(henyey_greenstein, HenyeyGreensteinScattering):
                raise TypeError("henyey_greenstein must be a HenyeyGreensteinScattering.")
            lengths = as_host_array(
                henyey_greenstein.scattering_lengths,
                HostFloat64[_MediumDim, _WavelengthDim],
                "henyey_greenstein.scattering_lengths",
                scope=scope,
            )
            anisotropy = as_host_array(
                henyey_greenstein.anisotropy,
                HostFloat64[_MediumDim, _WavelengthDim],
                "henyey_greenstein.anisotropy",
                scope=scope,
            )
            if not np.all(np.isfinite(anisotropy)) or np.any(np.abs(anisotropy) >= 1.0):
                raise ValueError(
                    "Henyey--Greenstein g must lie strictly between -1 and 1."
                )
            channels[..., _Channel.HENYEY_GREENSTEIN] = _inverse_lengths(
                lengths, "henyey_greenstein.scattering_lengths"
            )
            channels[..., _Channel.ANISOTROPY] = anisotropy
            content["henyey_greenstein"] = {"lengths": lengths, "g": anisotropy}
        mie_tables = None
        mie_evidence = None
        if mie is not None:
            if not isinstance(mie, MieParticles):
                raise TypeError("mie must be MieParticles.")
            mie_tables, mie_evidence, scattering, particle_absorption, mie_content = (
                _prepare_mie(mie, grid, index, scope)
            )
            channels[..., _Channel.MIE_SCATTERING] = scattering
            channels[..., _Channel.MIE_ABSORPTION] = particle_absorption
            content["mie"] = {
                **mie_content,
                "length_per_wavelength_unit": mie.length_per_wavelength_unit,
            }
        emission = None
        if wavelength_shifter is not None:
            if not isinstance(wavelength_shifter, WavelengthShifter):
                raise TypeError("wavelength_shifter must be a WavelengthShifter.")
            emission, shifting, emission_content = _prepare_emission(
                wavelength_shifter, grid, scope
            )
            channels[..., _Channel.WAVELENGTH_SHIFTING] = shifting
            content["wavelength_shifter"] = {
                **emission_content,
                "coefficients": shifting,
                "delay": wavelength_shifter.delay,
            }
        self.wavelengths = jnp.asarray(grid)
        self.table = jnp.asarray(channels)
        self.mie_tables = mie_tables
        self.mie_evidence = mie_evidence
        self.emission = emission
        self.medium_count = media
        self.medium_id = canonical_fingerprint(
            {
                "kind": "spectral-optical-medium",
                "content": array_tree_fingerprint(content),
            }
        )

    def _channels(self, medium_indices: Array, wavelengths: Array) -> tuple[Array, Array]:
        """Interpolated channels ``(lanes, channel)`` and the support mask."""

        result = linear_interpolate(
            self.wavelengths,
            self.table,
            wavelengths,
            axis=1,
            bounds="fill",
            fill_value=jnp.nan,
        )
        lanes = jnp.arange(wavelengths.shape[0])
        return result.values[lanes, medium_indices], result.support

    def refractive_index(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        values, _ = self._channels(medium_indices, wavelengths)
        return values[:, _Channel.REFRACTIVE_INDEX]

    def extinction(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        values, _ = self._channels(medium_indices, wavelengths)
        return (
            values[:, _Channel.ABSORPTION]
            + values[:, _Channel.RAYLEIGH]
            + values[:, _Channel.HENYEY_GREENSTEIN]
            + values[:, _Channel.MIE_SCATTERING]
            + values[:, _Channel.MIE_ABSORPTION]
            + values[:, _Channel.WAVELENGTH_SHIFTING]
        )

    def _process_weights(self, medium_indices: Array, values: Array) -> Array:
        """Per-lane rates of Rayleigh, HG, Mie, and re-emitting shifting."""

        yields = (
            jnp.zeros(medium_indices.shape, dtype=jnp.float64)
            if self.emission is None
            else self.emission.quantum_yields[medium_indices]
        )
        return jnp.stack(
            (
                values[:, _Channel.RAYLEIGH],
                values[:, _Channel.HENYEY_GREENSTEIN],
                values[:, _Channel.MIE_SCATTERING],
                yields * values[:, _Channel.WAVELENGTH_SHIFTING],
            ),
            axis=-1,
        )

    def albedo(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        values, _ = self._channels(medium_indices, wavelengths)
        total = self.extinction(medium_indices, wavelengths)
        surviving = jnp.sum(self._process_weights(medium_indices, values), axis=-1)
        return surviving / jnp.where(total > 0.0, total, 1.0)

    def spectral_support(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        _, support = self._channels(medium_indices, wavelengths)
        return support

    def _mie_event(
        self,
        medium_indices: Array,
        grid_position: Array,
        node_uniforms: Array,
        cosine_uniforms: Array,
    ) -> tuple[Array, Array, Array]:
        """Cosine and amplitudes from the interpolation-weighted node mixture."""

        tables = self.mie_tables
        if tables is None:
            raise RuntimeError("Mie tables are absent.")
        last = self.wavelengths.shape[0] - 1
        safe_position = jnp.where(jnp.isfinite(grid_position), grid_position, 0.0)
        lower = jnp.clip(jnp.floor(safe_position), 0, last - 1).astype(jnp.int32)
        node = lower + (node_uniforms < safe_position - lower).astype(jnp.int32)
        cosines = _sample_piecewise_linear(
            tables.cosines,
            tables.densities[medium_indices, node],
            tables.cdf[medium_indices, node],
            cosine_uniforms,
        )
        s1, s2 = jax.vmap(_mie_amplitudes)(
            tables.a[medium_indices, node], tables.b[medium_indices, node], cosines
        )
        return cosines, s1, s2

    def _emission_event(
        self, medium_indices: Array, uniforms: Array
    ) -> tuple[Array, Array]:
        """Emission wavelengths and re-emission delays."""

        emission = self.emission
        if emission is None:
            raise RuntimeError("Wavelength-shifter tables are absent.")
        wavelengths = _sample_piecewise_linear(
            emission.wavelengths,
            emission.densities[medium_indices],
            emission.cdf[medium_indices],
            uniforms[:, 0],
        )
        mean = emission.delay_times[medium_indices]
        match emission.delay:
            case "exponential":
                delays = -mean * jnp.log1p(-uniforms[:, 1])
            case "delta":
                delays = mean
            case _:
                raise ValueError(f"Unknown wavelength-shift delay {emission.delay!r}.")
        return wavelengths, delays

    def scatter(
        self,
        medium_indices: Array,
        wavelengths: Array,
        jones_vectors: Array,
        keys: Array,
        /,
    ) -> OpticalScatteringSample:
        values, _ = self._channels(medium_indices, wavelengths)
        rates = self._process_weights(medium_indices, values)
        uniform_keys = jax.vmap(lambda key: jr.fold_in(key, 0))(keys)
        normal_keys = jax.vmap(lambda key: jr.fold_in(key, 1))(keys)
        # Columns: process, cosine, Mie node, azimuth mixture, azimuth,
        # emission wavelength, emission delay.
        uniforms = _uniform_rows(uniform_keys, 7)
        # Columns 0-4: azimuth sin^2 law; 5-8: re-emitted Jones vector.
        normals = _normal_rows(normal_keys, 9)
        cumulative = jnp.cumsum(rates, axis=-1)
        draw = uniforms[:, 0] * cumulative[:, -1]
        process = jnp.sum(draw[:, None] >= cumulative[:, :-1], axis=-1)
        rayleigh = process == _Process.RAYLEIGH
        henyey_greenstein = process == _Process.HENYEY_GREENSTEIN
        mie = process == _Process.MIE
        shifting = process == _Process.WAVELENGTH_SHIFTING

        cosine_uniforms = uniforms[:, 1]
        cosines = jnp.where(
            rayleigh,
            _rayleigh_cosine(cosine_uniforms),
            jnp.where(
                henyey_greenstein,
                _henyey_greenstein_cosine(
                    values[:, _Channel.ANISOTROPY], cosine_uniforms
                ),
                2.0 * cosine_uniforms - 1.0,
            ),
        )
        unit = jnp.ones(wavelengths.shape, dtype=jnp.complex128)
        s1 = unit
        s2 = jnp.where(rayleigh, cosines.astype(jnp.complex128), unit)
        if self.mie_tables is not None:
            mie_cosines, mie_s1, mie_s2 = self._mie_event(
                medium_indices,
                values[:, _Channel.GRID_POSITION],
                uniforms[:, 2],
                cosine_uniforms,
            )
            cosines = jnp.where(mie, mie_cosines, cosines)
            s1 = jnp.where(mie, mie_s1, s1)
            s2 = jnp.where(mie, mie_s2, s2)
        azimuths = _polarized_azimuth(
            jones_vectors, s1, s2, uniforms[:, 3:5], normals[:, :5]
        )
        frame = rotate_jones_to_scattering_frame(jones_vectors, azimuths)
        scattered = jnp.stack((s1 * frame[:, 0], s2 * frame[:, 1]), axis=-1)
        if self.emission is None:
            emitted_wavelengths = wavelengths
            delays = jnp.zeros(wavelengths.shape, dtype=jnp.float64)
        else:
            emitted, emission_delays = self._emission_event(
                medium_indices, uniforms[:, 5:7]
            )
            emitted_wavelengths = jnp.where(shifting, emitted, wavelengths)
            delays = jnp.where(shifting, emission_delays, 0.0)
            # Unpolarized re-emission: a Jones vector uniform on the sphere.
            random_jones = normals[:, 5:9:2] + 1j * normals[:, 6:9:2]
            scattered = jnp.where(shifting[:, None], random_jones, scattered)
        norm = jnp.sqrt(jnp.sum(jnp.abs(scattered) ** 2, axis=-1))
        jones = scattered / jnp.maximum(norm, _TINY)[:, None]
        return OpticalScatteringSample(
            jnp.clip(cosines, -1.0, 1.0),
            azimuths,
            jones,
            emitted_wavelengths,
            delays,
        )


__all__ = [
    "HenyeyGreensteinScattering",
    "LorenzMieResult",
    "MieParticles",
    "MieTableEvidence",
    "RayleighScattering",
    "SpectralOpticalMedium",
    "WavelengthShiftDelay",
    "WavelengthShifter",
    "lorenz_mie",
]
