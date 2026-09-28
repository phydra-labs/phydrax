#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cold magnetized multi-species plasma: Stix dielectric, wave modes, Faraday effect.

Conventions (Stix 1992):

- phasors carry ``exp(−iωt)``; a collision frequency ``ν_s`` enters the species
  momentum equation as ``ω → ω + iν_s`` while the current still divides by the
  real ``ω``, so ``Im(n²) > 0`` is absorption;
- the Stix frame has ``B₀ ∥ ẑ`` and the wave normal ``κ̂ = (sin θ, 0, cos θ)`` in the
  ``x–z`` plane with ``ŷ = ẑ × x̂``; the signed cyclotron frequency is
  ``Ω_s = q_s |B₀| / m_s`` (negative for electrons);
- ``R = 1 − Σ ω_ps² / (ω (ω + iν_s + Ω_s))``, ``L`` with ``−Ω_s``,
  ``P = 1 − Σ ω_ps² / (ω (ω + iν_s))``, ``S = (R + L)/2``, ``D = (R − L)/2``;
- the wave operator is ``n² (κ̂κ̂ − I) + ε`` and its determinant is the Stix
  biquadratic ``A n⁴ − B n² + C = 0`` with ``A = S sin²θ + P cos²θ``,
  ``B = RL sin²θ + PS (1 + cos²θ)``, ``C = PRL``.

Both roots are carried as normalized homogeneous pairs ``(u, v)`` with
``n² = u/v`` so that resonances (``v = 0``) and cutoffs (``u = 0``) are ordinary
points. The discriminant is formed as
``F² = (RL − PS)² sin⁴θ + 4P²D² cos²θ``, which keeps the root splitting
accurate when the two roots nearly coincide. Branch identity is not read from
the sign of the discriminant at one angle: the branch ``(B + F)/(2A)`` is
continuous in ``θ`` whenever ``F`` is, so ``F`` is continued numerically in
``θ`` from its signed closed forms ``2PD`` at ``θ = 0`` (where the branch is
``R``) and ``RL − PS`` at ``θ = π/2`` (where it is ``RL/S``, the extraordinary
mode). ``F²`` is pole-free, so resonances do not obstruct the continuation; the
minimum relative root separation and a per-step certificate along each path are
reported as evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum, IntFlag
from math import isfinite, pi
from typing import assert_never, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from numpy.polynomial import polynomial as host_polynomial

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import (
    as_host_array,
    Bool,
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


_CANCELLATION_TOLERANCE = 64.0 * float(np.finfo(np.float64).eps)
"""Relative size below which a pole's leading Laurent coefficient is canceled."""


class _SpeciesDim(Dim, minimum=1):
    """Charged species of the cold plasma."""


class _BatchDims(VariadicDim):
    """Broadcast batch of ``(ω, θ)`` evaluation points."""


class _RightCutoffDim(Dim):
    """Zeros of the Stix ``R`` parameter."""


class _LeftCutoffDim(Dim):
    """Zeros of the Stix ``L`` parameter."""


class _PlasmaCutoffDim(Dim):
    """Zeros of the Stix ``P`` parameter."""


class _HybridResonanceDim(Dim):
    """Zeros of the Stix ``S`` parameter."""


class PlasmaWaveMode(IntEnum):
    """Cold-plasma branch identity at the parallel and perpendicular endpoints.

    ``RIGHT``/``LEFT`` name the root that continues from ``n² = R``/``n² = L`` at
    ``θ = 0``; ``ORDINARY``/``EXTRAORDINARY`` name the root that continues from
    ``n² = P``/``n² = RL/S`` at ``θ = π/2``. The integer values are the codes
    stored in `ColdPlasmaWaveResult.parallel_mode` and `perpendicular_mode`.
    """

    RIGHT = 0
    LEFT = 1
    ORDINARY = 2
    EXTRAORDINARY = 3


class ColdPlasmaWaveStatus(IntFlag):
    """Per-root evidence flags of `ColdPlasmaWaveResult.status`."""

    NONE = 0
    EVANESCENT = 1
    RESONANT = 2
    ROOT_DEGENERATE = 4
    PARALLEL_LABEL_AMBIGUOUS = 8
    PERPENDICULAR_LABEL_AMBIGUOUS = 16
    POLARIZATION_UNDEFINED = 32
    NONFINITE = 64


class StixParameters(StrictModule):
    """Stix ``S``, ``D``, ``P``, ``R``, ``L`` at one batch of angular frequencies."""

    __strict_contract__ = True

    sum_term: Complex128[_BatchDims]
    difference_term: Complex128[_BatchDims]
    plasma_term: Complex128[_BatchDims]
    right_term: Complex128[_BatchDims]
    left_term: Complex128[_BatchDims]


class ColdPlasmaWaveResult(StrictModule):
    """Both roots of the Stix biquadratic with identity, polarization, and evidence.

    Trailing axis ``2`` indexes the two roots in numerically stable order; use
    `select` to pick a root by `PlasmaWaveMode`. ``refractive_index`` is the
    principal square root of ``n²`` (``Re n ≥ 0``, ``Im n ≥ 0`` for absorbing
    roots). ``polarization`` is the unit null vector of the wave operator in the
    Stix frame with its largest component real and positive;
    ``transverse_ratio`` is Stix's ``K = i E_x / E_y`` and
    ``longitudinal_component`` is ``κ̂ · E`` for that unit vector. Resonant roots
    carry infinite ``n²`` and degenerate or undefined quantities carry NaN, each
    with the matching `ColdPlasmaWaveStatus` bit set. ``*_branch_separation`` is
    the minimum relative root separation ``|n₊² − n₋²| / (|n₊²| + |n₋²|)`` along
    each continuation path and ``*_branch_turn`` the maximum per-step turn
    ``|Δ arg F²| / π`` of the discriminant (infinite where it vanishes); labels
    are certified when the turn is below ``1/2``, otherwise the corresponding
    ambiguity bit is set and `ColdPlasmaDielectric.continuation_steps` should be
    raised. A collisionless plasma has real ``F² ≥ 0`` and zero turn unless the
    roots coincide.
    ``quasi_longitudinal_term = |4 P² D² cos²θ|`` and
    ``quasi_transverse_term = |(RL − PS)² sin⁴θ|`` are the two parts of the
    discriminant whose ratio decides the QL/QT regime.
    """

    __strict_contract__ = True

    angular_frequency: Float64[_BatchDims]
    angle: Float64[_BatchDims]
    n_squared: Complex128[_BatchDims, Literal[2]]
    refractive_index: Complex128[_BatchDims, Literal[2]]
    polarization: Complex128[_BatchDims, Literal[2], Literal[3]]
    transverse_ratio: Complex128[_BatchDims, Literal[2]]
    longitudinal_component: Complex128[_BatchDims, Literal[2]]
    parallel_mode: Int32[_BatchDims, Literal[2]]
    perpendicular_mode: Int32[_BatchDims, Literal[2]]
    status: Int32[_BatchDims, Literal[2]]
    polarization_margin: Float64[_BatchDims, Literal[2]]
    parallel_branch_separation: Float64[_BatchDims]
    parallel_branch_turn: Float64[_BatchDims]
    perpendicular_branch_separation: Float64[_BatchDims]
    perpendicular_branch_turn: Float64[_BatchDims]
    quasi_longitudinal_term: Float64[_BatchDims]
    quasi_transverse_term: Float64[_BatchDims]
    stix: StixParameters
    continuation_steps: int = eqx.field(static=True)

    def select(self, mode: PlasmaWaveMode, values: ArrayLike, /) -> Array:
        """Pick, from a ``[..., 2]`` root-indexed array, the root labeled ``mode``."""
        mode_ = parse(mode, PlasmaWaveMode, "mode")
        array = jnp.asarray(values)
        if array.shape[: self.status.ndim] != self.status.shape:
            raise ValueError(
                "select requires an array whose leading axes are the batch and the "
                "root axis of this result."
            )
        match mode_:
            case PlasmaWaveMode.RIGHT | PlasmaWaveMode.LEFT:
                labels = self.parallel_mode
            case PlasmaWaveMode.ORDINARY | PlasmaWaveMode.EXTRAORDINARY:
                labels = self.perpendicular_mode
            case _:
                assert_never(mode_)
        root_axis = self.status.ndim - 1
        first = labels[..., 0] == mode_.value
        first = first.reshape(first.shape + (1,) * (array.ndim - self.status.ndim))
        return jnp.where(
            first, jnp.take(array, 0, axis=root_axis), jnp.take(array, 1, axis=root_axis)
        )


class ColdPlasmaResonanceCone(StrictModule):
    """``tan²θ_res = −P/S``; ``angle`` is real only where ``exists`` holds."""

    __strict_contract__ = True

    tangent_squared: Complex128[_BatchDims]
    angle: Float64[_BatchDims]
    exists: Bool[_BatchDims]


class ColdPlasmaCharacteristicFrequencies(StrictModule):
    """Zeros of ``R``, ``L``, ``P`` (cutoffs) and ``S`` (hybrid resonances).

    Every zero of the rational Stix parameter is reported, including negative
    and complex frequencies; a negative real zero of ``R`` is the opposite of a
    zero of ``L`` in a collisionless plasma. Each ``*_residuals`` entry is the
    magnitude of the Stix parameter re-evaluated at the reported zero.
    """

    __strict_contract__ = True

    right_cutoffs: Complex128[_RightCutoffDim]
    left_cutoffs: Complex128[_LeftCutoffDim]
    plasma_cutoffs: Complex128[_PlasmaCutoffDim]
    hybrid_resonances: Complex128[_HybridResonanceDim]
    right_residuals: Float64[_RightCutoffDim]
    left_residuals: Float64[_LeftCutoffDim]
    plasma_residuals: Float64[_PlasmaCutoffDim]
    hybrid_residuals: Float64[_HybridResonanceDim]


class FaradayCoefficients(StrictModule):
    """Half the Poincaré-sphere rotation generator of the two cold-plasma modes.

    ``coefficients = (ρ_Q, ρ_U, ρ_V)`` per unit length in the wave-normal
    transverse basis ``ê₁ = (cos θ, 0, −sin θ)`` (in the ``k``–``B₀`` plane, so
    ``Q > 0`` is linear polarization along the projected field), ``ê₂ = ŷ``, with
    ``V = 2 Im(E₁* E₂)`` positive for the right-handed sense about ``κ̂``. Along
    ``B₀`` this gives ``ρ_V = (ω/2c)(n_L − n_R)``, the rotation rate of the plane
    of linear polarization; across ``B₀`` it gives ``ρ_Q = (ω/2c)(n_X − n_O)``.
    The Stokes vector obeys ``d(Q,U,V)/ds = 2 Re ρ × (Q,U,V)``; imaginary parts
    are half the differential attenuation between the modes. The generator is
    exact for orthogonal transverse modes (collisionless plasma) and
    ``mode_overlap = |ŝ₀ + ŝ₁| / 2`` measures the departure from orthogonality
    under collisions.
    ``transverse_fraction`` is the transverse power of each unit mode vector and
    ``mode_stokes`` the normalized transverse Stokes vector ``(Q, U, V)/I`` of each
    root in the same basis; a purely longitudinal root gives NaN coefficients.
    """

    __strict_contract__ = True

    coefficients: Complex128[_BatchDims, Literal[3]]
    mode_overlap: Float64[_BatchDims]
    transverse_fraction: Float64[_BatchDims, Literal[2]]
    mode_stokes: Float64[_BatchDims, Literal[2], Literal[3]]
    wave: ColdPlasmaWaveResult

    @property
    def rotation(self) -> Array:
        """``ρ_V``: Faraday rotation per unit length (real part)."""
        return self.coefficients[..., 2]

    @property
    def conversion(self) -> Array:
        """``ρ_Q``: Faraday conversion per unit length (real part)."""
        return self.coefficients[..., 0]


def _float64_argument(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.inexact) and array.dtype != jnp.float64:
        raise TypeError(f"{name} requires float64 values; received {array.dtype}.")
    return array.astype(jnp.float64)


def _normalize(u: Array, v: Array, /) -> tuple[Array, Array, Array]:
    norm = jnp.sqrt(jnp.abs(u) ** 2 + jnp.abs(v) ** 2)
    scale = norm.astype(jnp.complex128)
    return u / scale, v / scale, norm


def _quartic_coefficients(
    stix: StixParameters, sin_squared: Array, cos_squared: Array, /
) -> tuple[Array, Array, Array, Array]:
    """``A``, ``B``, ``C`` and the discriminant ``F² = B² − 4AC`` of the biquadratic.

    ``F²`` uses the exact decomposition ``(RL − PS)² sin⁴θ + 4 P² D² cos²θ``;
    forming ``B² − 4AC`` directly loses ``log₁₀(B²/F²)`` digits where the two
    roots nearly coincide, as for weakly magnetized plasmas.
    """
    s, d, p = stix.sum_term, stix.difference_term, stix.plasma_term
    rl = stix.right_term * stix.left_term
    sin2 = sin_squared.astype(jnp.complex128)
    cos2 = cos_squared.astype(jnp.complex128)
    a = s * sin2 + p * cos2
    b = rl * sin2 + p * s * (1.0 + cos2)
    anisotropy = (rl - p * s) * sin2
    gyrotropy = p * d
    discriminant_squared = anisotropy * anisotropy + 4.0 * gyrotropy * gyrotropy * cos2
    return a, b, p * rl, discriminant_squared


def _homogeneous_roots(
    a: Array, b: Array, c: Array, discriminant: Array, /
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Normalized homogeneous roots ``(u₀, v₀)``, ``(u₁, v₁)`` and their raw norms.

    Slot 0 is ``(B ± F)/(2A)`` with the sign that avoids cancellation and slot 1
    is the Vieta partner ``2C/(B ± F)``.
    """
    plus = b + discriminant
    minus = b - discriminant
    q = jnp.where(jnp.abs(plus) >= jnp.abs(minus), plus, minus)
    u0, v0, norm0 = _normalize(q, 2.0 * a)
    u1, v1, norm1 = _normalize(2.0 * c, q)
    return u0, v0, u1, v1, norm0, norm1


def _continue_splitting(offset: Array, splitting: Array, /) -> tuple[Array, Array, Array]:
    """Continue ``F = ±√F²`` along a sampled path (last axis).

    ``offset`` is ``B`` and ``splitting`` the principal ``√F²`` at each sample,
    except the first sample, which carries the signed closed form of ``F`` at
    the path endpoint. Each step keeps the sign of ``F`` whose argument turns
    by less than ``π/2``; the choice ties only when ``arg F²`` turns by ``π``.
    Returns the parity of sign flips that continue the first sample to the last,
    the minimum relative root separation
    ``2|F| / (|B + F| + |B − F|) = |n₊² − n₋²| / (|n₊²| + |n₋²|)``, and the
    maximum per-step turn ``|Δ arg F²| / π`` (infinite where ``F`` vanishes),
    which certifies every sign choice when below ``1/2``. Growth of ``|F|``
    alone never makes the choice ambiguous: only winding of ``F²`` near zero
    does.
    """
    previous = splitting[..., :-1]
    current = splitting[..., 1:]
    keep = jnp.abs(current - previous)
    flip = jnp.abs(current + previous)
    parity = jnp.sum(flip < keep, axis=-1, dtype=jnp.int32) % 2
    squared = splitting * splitting
    turn = squared[..., 1:] * jnp.conj(squared[..., :-1])
    ratio = jnp.where(turn == 0.0, jnp.inf, jnp.abs(jnp.angle(turn)) / pi)
    magnitude = jnp.abs(splitting)
    separation = (2.0 * magnitude) / (
        jnp.abs(offset + splitting) + jnp.abs(offset - splitting)
    )
    return parity, jnp.min(separation, axis=-1), jnp.max(ratio, axis=-1)


def _with_trailing_axis(stix: StixParameters, /) -> StixParameters:
    return StixParameters(
        sum_term=stix.sum_term[..., None],
        difference_term=stix.difference_term[..., None],
        plasma_term=stix.plasma_term[..., None],
        right_term=stix.right_term[..., None],
        left_term=stix.left_term[..., None],
    )


def _label_pair(parity: Array, first: PlasmaWaveMode, second: PlasmaWaveMode, /) -> Array:
    """Labels ``[..., 2]`` of the two roots; even parity keeps ``first`` in slot 0."""
    even = parity == 0
    slot0 = jnp.where(even, first.value, second.value).astype(jnp.int32)
    slot1 = jnp.where(even, second.value, first.value).astype(jnp.int32)
    return jnp.stack((slot0, slot1), axis=-1)


def _null_vector(
    u: Array, v: Array, stix: StixParameters, sine: Array, cosine: Array, /
) -> tuple[Array, Array]:
    """Unit null vector of ``u (κ̂κ̂ − I) + v ε`` and its relative margin.

    The polarization vector is the requested dense result of a statically 3×3
    rank-two operator: it is the largest cross product of two rows, and the
    reported margin ``max ‖rᵢ × rⱼ‖ / max ‖r_k‖²`` is of order one for a
    one-dimensional null space and of roundoff size when the operator has rank
    one (isotropic plasma) or vanishes.
    """
    s, d, p = stix.sum_term, stix.difference_term, stix.plasma_term
    sine = sine.astype(jnp.complex128)
    cosine = cosine.astype(jnp.complex128)
    zero = jnp.zeros_like(u)
    shear = u * sine * cosine
    row1 = jnp.stack((v * s - u * cosine * cosine, -1j * v * d, shear), axis=-1)
    row2 = jnp.stack((1j * v * d, v * s - u, zero), axis=-1)
    row3 = jnp.stack((shear, zero, v * p - u * sine * sine), axis=-1)
    candidates = jnp.stack(
        (jnp.cross(row1, row2), jnp.cross(row2, row3), jnp.cross(row3, row1)),
        axis=-2,
    )
    norms = jnp.linalg.norm(candidates, axis=-1)
    row_scale = jnp.maximum(
        jnp.maximum(jnp.linalg.norm(row1, axis=-1), jnp.linalg.norm(row2, axis=-1)),
        jnp.linalg.norm(row3, axis=-1),
    )
    best = jnp.argmax(norms, axis=-1)
    vector = jnp.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    best_norm = jnp.take_along_axis(norms, best[..., None], axis=-1)[..., 0]
    margin = best_norm / (row_scale * row_scale)
    unit = vector / best_norm.astype(jnp.complex128)[..., None]
    leading = jnp.argmax(jnp.abs(unit), axis=-1)
    reference = jnp.take_along_axis(unit, leading[..., None], axis=-1)[..., 0]
    phase = reference / jnp.abs(reference).astype(jnp.complex128)
    return unit * jnp.conj(phase)[..., None], margin


def _rational_zeros(
    terms: Sequence[tuple[np.ndarray, Sequence[complex]]], /
) -> np.ndarray:
    """Zeros of ``1 − Σ_j num_j(ω) / Π_{p ∈ poles_j} (ω − p)`` on the host.

    Numerators are low-to-high coefficient arrays. The common denominator is the
    least common multiple of the pole multisets, so pole–zero cancellations
    inside one term must already be removed by the caller. A pole whose leading
    Laurent coefficient cancels across terms to within `_CANCELLATION_TOLERANCE`
    of the individual contributions (the ``ω = 0`` pole of ``R`` and ``L`` in a
    charge-neutral collisionless plasma) is a root of the cleared numerator but
    not of the rational function; it is deflated once before the roots of the
    monic numerator are taken and sorted.
    """
    multiplicity: dict[complex, int] = {}
    for _, poles in terms:
        own: dict[complex, int] = {}
        for pole in poles:
            own[pole] = own.get(pole, 0) + 1
        for pole, count in own.items():
            multiplicity[pole] = max(multiplicity.get(pole, 0), count)
    denominator = np.ones((1,), dtype=np.complex128)
    for pole, count in multiplicity.items():
        for _ in range(count):
            denominator = host_polynomial.polymul(
                denominator, np.asarray([-pole, 1.0], dtype=np.complex128)
            )
    numerator = denominator
    cofactors: list[np.ndarray] = []
    for coefficients, poles in terms:
        remaining = dict(multiplicity)
        for pole in poles:
            remaining[pole] -= 1
        cofactor = np.asarray(coefficients, dtype=np.complex128)
        for pole, count in remaining.items():
            for _ in range(count):
                cofactor = host_polynomial.polymul(
                    cofactor, np.asarray([-pole, 1.0], dtype=np.complex128)
                )
        cofactors.append(cofactor)
        numerator = host_polynomial.polysub(numerator, cofactor)
    for pole in multiplicity:
        contributions = np.asarray(
            [host_polynomial.polyval(pole, cofactor) for cofactor in cofactors]
        )
        if abs(contributions.sum()) <= _CANCELLATION_TOLERANCE * float(
            np.abs(contributions).sum()
        ):
            numerator, _ = host_polynomial.polydiv(
                numerator, np.asarray([-pole, 1.0], dtype=np.complex128)
            )
    roots = np.asarray(host_polynomial.polyroots(numerator), dtype=np.complex128)
    order = np.lexsort((roots.imag, roots.real))
    return roots[order]


def _species_frequencies(
    scale: ElectromagneticScaleContract,
    densities: Array,
    charge_numbers: Array,
    mass_ratios: Array,
    field_magnitude: Array,
    /,
) -> tuple[Array, Array]:
    """``ω_ps² = n_s (Z_s e)² / (ε₀ m_s)`` and signed ``Ω_s = Z_s e |B₀| / m_s``."""
    charge = charge_numbers * float(scale.elementary_charge)
    mass = mass_ratios * float(scale.electron_mass)
    plasma = densities * charge * charge / (float(scale.vacuum_permittivity) * mass)
    return plasma, charge * field_magnitude / mass


def _species_stix(
    omega: Array,
    plasma_frequency_squared: Array,
    cyclotron_frequency: Array,
    collision_frequencies: Array,
    /,
) -> StixParameters:
    """Stix parameters at complex-typed ``ω`` from per-species frequencies (last axis)."""
    frequency = omega[..., None]
    batch = (1,) * omega.ndim + (plasma_frequency_squared.shape[-1],)
    collision = collision_frequencies.astype(jnp.complex128).reshape(batch)
    shifted = frequency + 1j * collision
    weight = plasma_frequency_squared.astype(jnp.complex128).reshape(batch) / frequency
    gyro = cyclotron_frequency.astype(jnp.complex128).reshape(batch)
    right = 1.0 - jnp.sum(weight / (shifted + gyro), axis=-1)
    left = 1.0 - jnp.sum(weight / (shifted - gyro), axis=-1)
    plasma = 1.0 - jnp.sum(weight / shifted, axis=-1)
    return StixParameters(
        sum_term=0.5 * (right + left),
        difference_term=0.5 * (right - left),
        plasma_term=plasma,
        right_term=right,
        left_term=left,
    )


def _dispersion_polynomial(
    stix: StixParameters, parallel_squared: Array, perpendicular_squared: Array, /
) -> Array:
    """Stix determinant ``A n⁴ − B n² + C`` in Cartesian refractive-index components.

    With ``n∥² = n² cos²θ`` and ``n⊥² = n² sin²θ`` the biquadratic is the
    polynomial ``n² (S n⊥² + P n∥²) − RL n⊥² − PS (n² + n∥²) + PRL``, which stays
    smooth at ``n = 0`` and along ``B₀``.
    """
    s, p = stix.sum_term, stix.plasma_term
    rl = stix.right_term * stix.left_term
    parallel = parallel_squared.astype(jnp.complex128)
    perpendicular = perpendicular_squared.astype(jnp.complex128)
    total = parallel + perpendicular
    return (
        total * (s * perpendicular + p * parallel)
        - rl * perpendicular
        - p * s * (total + parallel)
        + p * rl
    )


def _continue_branches(
    steps: int,
    stix: StixParameters,
    offset: Array,
    discriminant: Array,
    theta: Array,
    start: float,
    endpoint_offset: Array,
    endpoint_splitting: Array,
    /,
) -> tuple[Array, Array, Array]:
    """Parity that places the ``+endpoint_splitting`` branch in root slot 0.

    The branch ``(B + F)/(2A)`` is continuous in ``θ`` (as a homogeneous
    pair it passes through resonances) whenever ``F`` is, so identity reduces
    to continuing the pole-free scalar ``F`` from its signed closed form at
    ``start`` to the principal ``√F²`` used for the slots at ``θ``.
    """
    fractions = jnp.arange(1, steps, dtype=jnp.float64).reshape(
        (1,) * theta.ndim + (steps - 1,)
    ) / float(steps)
    angles = start + (theta - start)[..., None] * fractions
    sine = jnp.sin(angles)
    cosine = jnp.cos(angles)
    _, path_offset, _, path_squared = _quartic_coefficients(
        _with_trailing_axis(stix), sine * sine, cosine * cosine
    )
    parity, separation, ratio = _continue_splitting(
        jnp.concatenate(
            (endpoint_offset[..., None], path_offset, offset[..., None]), axis=-1
        ),
        jnp.concatenate(
            (
                endpoint_splitting[..., None],
                jnp.sqrt(path_squared),
                discriminant[..., None],
            ),
            axis=-1,
        ),
    )
    slot0_plus = jnp.abs(offset + discriminant) >= jnp.abs(offset - discriminant)
    return (parity + (~slot0_plus).astype(jnp.int32)) % 2, separation, ratio


def _cold_plasma_waves(
    stix: StixParameters,
    frequency: Array,
    angle: Array,
    continuation_steps: int,
    polarization_tolerance: float,
    /,
) -> ColdPlasmaWaveResult:
    """Labeled roots, polarization and evidence of the biquadratic at ``(ω, θ)``.

    ``stix`` holds the Stix parameters at the real angular frequencies
    ``frequency``; ``angle`` is ``θ`` broadcast against them.
    """
    sine = jnp.sin(angle)
    cosine = jnp.cos(angle)
    sin_squared = sine * sine
    cos_squared = cosine * cosine
    a, b, c, discriminant_squared = _quartic_coefficients(stix, sin_squared, cos_squared)
    discriminant = jnp.sqrt(discriminant_squared)
    u0, v0, u1, v1, norm0, norm1 = _homogeneous_roots(a, b, c, discriminant)
    s, d, p = stix.sum_term, stix.difference_term, stix.plasma_term
    rl = stix.right_term * stix.left_term
    # θ = 0: B = 2PS and F = 2PD give (B + F)/(2A) = S + D = R.
    parallel = _continue_branches(
        continuation_steps, stix, b, discriminant, angle, 0.0, 2.0 * p * s, 2.0 * p * d
    )
    # θ = π/2: B = RL + PS and F = RL − PS give (B + F)/(2A) = RL/S (X mode).
    perpendicular = _continue_branches(
        continuation_steps, stix, b, discriminant, angle, 0.5 * pi, rl + p * s, rl - p * s
    )
    u = jnp.stack((u0, u1), axis=-1)
    v = jnp.stack((v0, v1), axis=-1)
    norms = jnp.stack((norm0, norm1), axis=-1)
    n_squared = u / v
    polarization, margin = _null_vector(
        u, v, _with_trailing_axis(stix), sine[..., None], cosine[..., None]
    )
    parallel_parity, parallel_separation, parallel_turn = parallel
    perp_parity, perp_separation, perp_turn = perpendicular
    parallel_ambiguous = ~(parallel_turn < 0.5)
    perp_ambiguous = ~(perp_turn < 0.5)
    status = (
        (n_squared.real < 0.0).astype(jnp.int32) * ColdPlasmaWaveStatus.EVANESCENT.value
        | ((v == 0.0) & (norms != 0.0)).astype(jnp.int32)
        * ColdPlasmaWaveStatus.RESONANT.value
        | (norms == 0.0).astype(jnp.int32) * ColdPlasmaWaveStatus.ROOT_DEGENERATE.value
        | parallel_ambiguous[..., None].astype(jnp.int32)
        * ColdPlasmaWaveStatus.PARALLEL_LABEL_AMBIGUOUS.value
        | perp_ambiguous[..., None].astype(jnp.int32)
        * ColdPlasmaWaveStatus.PERPENDICULAR_LABEL_AMBIGUOUS.value
        | (~(margin > polarization_tolerance)).astype(jnp.int32)
        * ColdPlasmaWaveStatus.POLARIZATION_UNDEFINED.value
        | (~(jnp.isfinite(u) & jnp.isfinite(v))).astype(jnp.int32)
        * ColdPlasmaWaveStatus.NONFINITE.value
    )
    parallel_labels = _label_pair(
        parallel_parity, PlasmaWaveMode.RIGHT, PlasmaWaveMode.LEFT
    )
    perpendicular_labels = _label_pair(
        perp_parity, PlasmaWaveMode.EXTRAORDINARY, PlasmaWaveMode.ORDINARY
    )
    kappa = jnp.stack((sine, jnp.zeros_like(sine), cosine), axis=-1).astype(
        jnp.complex128
    )
    return ColdPlasmaWaveResult(
        angular_frequency=frequency,
        angle=angle,
        n_squared=n_squared,
        refractive_index=jnp.sqrt(n_squared),
        polarization=polarization,
        transverse_ratio=1j * polarization[..., 0] / polarization[..., 1],
        longitudinal_component=jnp.sum(polarization * kappa[..., None, :], axis=-1),
        parallel_mode=parallel_labels,
        perpendicular_mode=perpendicular_labels,
        status=status,
        polarization_margin=margin,
        parallel_branch_separation=parallel_separation,
        parallel_branch_turn=parallel_turn,
        perpendicular_branch_separation=perp_separation,
        perpendicular_branch_turn=perp_turn,
        quasi_longitudinal_term=jnp.abs(
            4.0 * (stix.plasma_term * stix.difference_term) ** 2
        )
        * cos_squared,
        quasi_transverse_term=jnp.abs(
            (stix.right_term * stix.left_term - stix.plasma_term * stix.sum_term) ** 2
        )
        * sin_squared
        * sin_squared,
        stix=stix,
        continuation_steps=continuation_steps,
    )


def _faraday_coefficients(
    wave: ColdPlasmaWaveResult, speed_of_light: float, /
) -> FaradayCoefficients:
    """Faraday generator of both roots in the wave-normal transverse basis."""
    sine = jnp.sin(wave.angle).astype(jnp.complex128)[..., None]
    cosine = jnp.cos(wave.angle).astype(jnp.complex128)[..., None]
    first = wave.polarization[..., 0] * cosine - wave.polarization[..., 2] * sine
    second = wave.polarization[..., 1]
    product = jnp.conj(first) * second
    transverse = jnp.abs(first) ** 2 + jnp.abs(second) ** 2
    stokes = (
        jnp.stack(
            (
                jnp.abs(first) ** 2 - jnp.abs(second) ** 2,
                2.0 * product.real,
                2.0 * product.imag,
            ),
            axis=-1,
        )
        / transverse[..., None]
    )
    wavenumber = wave.angular_frequency / speed_of_light
    difference = wave.refractive_index[..., 1] - wave.refractive_index[..., 0]
    generator = (
        0.25
        * (wavenumber.astype(jnp.complex128) * difference)[..., None]
        * (stokes[..., 0, :] - stokes[..., 1, :]).astype(jnp.complex128)
    )
    overlap = 0.5 * jnp.linalg.norm(stokes[..., 0, :] + stokes[..., 1, :], axis=-1)
    return FaradayCoefficients(
        coefficients=generator,
        mode_overlap=overlap,
        transverse_fraction=transverse,
        mode_stokes=stokes,
        wave=wave,
    )


class ColdPlasmaDielectric(StrictModule, NonTrainableState):
    """Cold multi-species magnetized plasma bound to an electromagnetic scale.

    ``densities`` are number densities per cubic length unit of the scale,
    ``charge_numbers`` are signed multiples of the elementary charge,
    ``mass_ratios`` are species masses in units of the electron mass,
    ``magnetic_field`` is ``B₀`` in the scale's field unit, and
    ``collision_frequencies`` are per-species momentum-transfer rates per time
    unit. ``continuation_steps`` samples each branch-identification path and
    ``polarization_tolerance`` is the smallest `ColdPlasmaWaveResult.polarization_margin`
    (a proxy of the wave operator's ``σ₂/σ₁``) for which a mode polarization is
    reported as defined; it becomes small in the isotropic limit ``D → 0``.
    """

    __strict_contract__ = True

    scale: ElectromagneticScaleContract = eqx.field(static=True)
    densities: Float64[_SpeciesDim]
    charge_numbers: Float64[_SpeciesDim]
    mass_ratios: Float64[_SpeciesDim]
    collision_frequencies: Float64[_SpeciesDim]
    magnetic_field: Float64[Literal[3]]
    species_count: Size[_SpeciesDim] = eqx.field(static=True)
    continuation_steps: int = eqx.field(static=True)
    polarization_tolerance: float = eqx.field(static=True)
    dielectric_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        scale: ElectromagneticScaleContract,
        /,
        *,
        densities: ConvertibleToArray,
        charge_numbers: ConvertibleToArray,
        mass_ratios: ConvertibleToArray,
        magnetic_field: ConvertibleToArray,
        collision_frequencies: ConvertibleToArray | None = None,
        continuation_steps: int = 64,
        polarization_tolerance: float = 1e-12,
    ) -> None:
        if not isinstance(scale, ElectromagneticScaleContract):
            raise TypeError("scale must be an ElectromagneticScaleContract.")
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
        collision = as_host_array(
            np.zeros(density.shape, dtype=np.float64)
            if collision_frequencies is None
            else collision_frequencies,
            HostFloat64[_SpeciesDim],
            "collision_frequencies",
            scope=scope,
        )
        field = as_host_array(
            magnetic_field, HostFloat64[Literal[3]], "magnetic_field", scope=scope
        )
        species_count = parse(
            density.shape[0], Size[_SpeciesDim], "species_count", scope=scope
        )
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError("densities must be finite and nonnegative.")
        if not np.all(np.isfinite(charge)) or np.any(charge == 0.0):
            raise ValueError("charge_numbers must be finite and nonzero.")
        if not np.all(np.isfinite(mass)) or np.any(mass <= 0.0):
            raise ValueError("mass_ratios must be finite and strictly positive.")
        if not np.all(np.isfinite(collision)) or np.any(collision < 0.0):
            raise ValueError("collision_frequencies must be finite and nonnegative.")
        if not np.all(np.isfinite(field)):
            raise ValueError("magnetic_field must be finite.")
        steps = int(continuation_steps)
        if steps < 1:
            raise ValueError("continuation_steps must be a positive integer.")
        tolerance = float(polarization_tolerance)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("polarization_tolerance must be finite and positive.")
        self.scale = scale
        self.densities = jnp.asarray(density)
        self.charge_numbers = jnp.asarray(charge)
        self.mass_ratios = jnp.asarray(mass)
        self.collision_frequencies = jnp.asarray(collision)
        self.magnetic_field = jnp.asarray(field)
        self.species_count = species_count
        self.continuation_steps = steps
        self.polarization_tolerance = tolerance
        self.dielectric_id = canonical_fingerprint(
            {
                "kind": "cold-plasma-dielectric",
                "scale": scale.scale_id,
                "continuation_steps": steps,
                "polarization_tolerance": tolerance,
                "content": array_tree_fingerprint(
                    {
                        "densities": density,
                        "charge_numbers": charge,
                        "mass_ratios": mass,
                        "collision_frequencies": collision,
                        "magnetic_field": field,
                    }
                ),
            }
        )

    @property
    def speed_of_light(self) -> float:
        return float(self.scale.speed_of_light)

    @property
    def magnetic_field_magnitude(self) -> Array:
        return jnp.linalg.norm(self.magnetic_field)

    @property
    def plasma_frequency_squared(self) -> Array:
        """``ω_ps² = n_s (Z_s e)² / (ε₀ m_s)`` per species."""
        return _species_frequencies(
            self.scale,
            self.densities,
            self.charge_numbers,
            self.mass_ratios,
            self.magnetic_field_magnitude,
        )[0]

    @property
    def cyclotron_frequency(self) -> Array:
        """Signed ``Ω_s = Z_s e |B₀| / m_s`` per species."""
        return _species_frequencies(
            self.scale,
            self.densities,
            self.charge_numbers,
            self.mass_ratios,
            self.magnetic_field_magnitude,
        )[1]

    def _stix(self, omega: Array, /) -> StixParameters:
        return _species_stix(
            omega,
            self.plasma_frequency_squared,
            self.cyclotron_frequency,
            self.collision_frequencies,
        )

    def stix_parameters(self, omega: ArrayLike, /) -> StixParameters:
        """Stix parameters at real float64 angular frequencies ``ω``."""
        return self._stix(_float64_argument(omega, "omega").astype(jnp.complex128))

    def dielectric_tensor(self, omega: ArrayLike, /) -> Array:
        """``ε(ω)[..., 3, 3]`` in the Stix frame (``B₀ ∥ ẑ``)."""
        stix = self.stix_parameters(omega)
        s, d, p = stix.sum_term, stix.difference_term, stix.plasma_term
        zero = jnp.zeros_like(s)
        return jnp.stack(
            (
                jnp.stack((s, -1j * d, zero), axis=-1),
                jnp.stack((1j * d, s, zero), axis=-1),
                jnp.stack((zero, zero, p), axis=-1),
            ),
            axis=-2,
        )

    def resonance_cone(self, omega: ArrayLike, /) -> ColdPlasmaResonanceCone:
        """Resonance-cone angle ``tan²θ_res = −P/S`` at real ``ω``."""
        stix = self.stix_parameters(omega)
        tangent_squared = -stix.plasma_term / stix.sum_term
        exists = (tangent_squared.imag == 0.0) & (tangent_squared.real > 0.0)
        return ColdPlasmaResonanceCone(
            tangent_squared=tangent_squared,
            angle=jnp.arctan(jnp.sqrt(tangent_squared.real)),
            exists=exists,
        )

    def characteristic_frequencies(self) -> ColdPlasmaCharacteristicFrequencies:
        """Cutoffs (``R``, ``L``, ``P`` zeros) and hybrid resonances (``S`` zeros).

        Host preparation from concrete species data: the rational Stix parameter
        is cleared to one monic polynomial and its roots are taken from the
        companion matrix. A pole that cancels across species (the ``ω = 0`` pole
        of ``R`` and ``L`` in a charge-neutral collisionless plasma) is removed
        with its spurious numerator root. Traced species data are refused.
        """
        weights = as_host_array(
            self.plasma_frequency_squared, HostFloat64[_SpeciesDim], "weights"
        )
        gyro = as_host_array(
            self.cyclotron_frequency, HostFloat64[_SpeciesDim], "cyclotron"
        )
        collision = as_host_array(
            self.collision_frequencies, HostFloat64[_SpeciesDim], "collision"
        )
        right: list[tuple[np.ndarray, list[complex]]] = []
        left: list[tuple[np.ndarray, list[complex]]] = []
        plasma: list[tuple[np.ndarray, list[complex]]] = []
        hybrid: list[tuple[np.ndarray, list[complex]]] = []
        for weight, omega_c, nu in zip(weights, gyro, collision, strict=True):
            constant = np.asarray([weight], dtype=np.complex128)
            damping = complex(0.0, -float(nu))
            right.append((constant, [0j, damping - float(omega_c)]))
            left.append((constant, [0j, damping + float(omega_c)]))
            plasma.append((constant, [0j, damping]))
            # S_s = ω_ps² (ω + iν) / (ω (ω + iν − Ω)(ω + iν + Ω)); the numerator
            # cancels the ω pole when ν = 0 and one gyro pole when Ω = 0.
            if omega_c == 0.0:
                hybrid.append((constant, [0j, damping]))
            elif nu == 0.0:
                hybrid.append((constant, [float(omega_c), -float(omega_c)]))
            else:
                hybrid.append(
                    (
                        np.asarray([weight * -damping, weight], dtype=np.complex128),
                        [0j, damping - float(omega_c), damping + float(omega_c)],
                    )
                )
        right_roots = _rational_zeros(right)
        left_roots = _rational_zeros(left)
        plasma_roots = _rational_zeros(plasma)
        hybrid_roots = _rational_zeros(hybrid)
        return ColdPlasmaCharacteristicFrequencies(
            right_cutoffs=jnp.asarray(right_roots),
            left_cutoffs=jnp.asarray(left_roots),
            plasma_cutoffs=jnp.asarray(plasma_roots),
            hybrid_resonances=jnp.asarray(hybrid_roots),
            right_residuals=jnp.abs(self._stix(jnp.asarray(right_roots)).right_term),
            left_residuals=jnp.abs(self._stix(jnp.asarray(left_roots)).left_term),
            plasma_residuals=jnp.abs(self._stix(jnp.asarray(plasma_roots)).plasma_term),
            hybrid_residuals=jnp.abs(self._stix(jnp.asarray(hybrid_roots)).sum_term),
        )

    def refractive_indices(
        self, omega: ArrayLike, theta: ArrayLike, /
    ) -> ColdPlasmaWaveResult:
        """Both roots of ``A n⁴ − B n² + C = 0`` at real ``ω`` and wave-normal angle ``θ``.

        ``θ`` is the angle between ``k`` and ``B₀`` in radians; ``ω`` and ``θ``
        broadcast. See `ColdPlasmaWaveResult` for the reported identity,
        polarization, and evidence.
        """
        frequency, angle = jnp.broadcast_arrays(
            _float64_argument(omega, "omega"), _float64_argument(theta, "theta")
        )
        return _cold_plasma_waves(
            self._stix(frequency.astype(jnp.complex128)),
            frequency,
            angle,
            self.continuation_steps,
            self.polarization_tolerance,
        )

    def faraday_coefficients(
        self, omega: ArrayLike, theta: ArrayLike, /
    ) -> FaradayCoefficients:
        """Faraday rotation and conversion per unit length from the two mode indices."""
        return _faraday_coefficients(
            self.refractive_indices(omega, theta), self.speed_of_light
        )


__all__ = [
    "ColdPlasmaCharacteristicFrequencies",
    "ColdPlasmaDielectric",
    "ColdPlasmaResonanceCone",
    "ColdPlasmaWaveResult",
    "ColdPlasmaWaveStatus",
    "FaradayCoefficients",
    "PlasmaWaveMode",
    "StixParameters",
]
