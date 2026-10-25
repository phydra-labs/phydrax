"""Independent host references for degree-two multipoles and induced dipoles.

Pure NumPy/SciPy equations sharing no code with the native electrostatic,
polarization, linear-algebra or derivative kernels. Every dense object and
every real/reciprocal block is bounded before allocation; unsupported inputs
raise :class:`ReferenceUnsupportedError` and exhausted resources (atom bound,
image/mode shells, pair-image terms, unconverged final level) raise
:class:`ReferenceResourceError` carrying the convergence history so far.

Definitions
-----------
Sites carry primitive Cartesian moments: charge ``q``, dipole ``mu`` and the
raw (traceful) second moment ``Theta = int rho s s``, acting through
``O(grad) = q - mu.grad + 1/2 Theta : grad grad`` (``O(-grad)`` for the
receiving site). The energy is ``E = 1/2 B(M, M)`` of the symmetric form

``B(L, R) = k sum'_{i, j, n} O_{L_i}(-grad) O_{R_j}(grad) phi_ij(R)``,
``R = r_i - r_j + n H`` (rows of ``H`` are lattice vectors),

where ``sum'`` omits the co-centered term ``i = j, n = 0``. Point sites use
``phi = 1/R``; Gaussian sites (unit-normalized, standard deviation ``sigma``)
use ``phi_ij = erf(beta_ij R)/R`` with ``beta_ij = 1/sqrt(2(s_i^2 + s_j^2))``.
``include_coincident_self`` adds, for Gaussian sites only, the physical
co-centered term ``k/2 O_i(-grad) O_i(grad) erf(r/(2 sigma_i))/r |_0``; for point
sites that term is infinite and the request is refused. Distinct sites at the
same position interact physically (Gaussian) or are refused (point).

With ``M^0 = q``, ``M^1 = mu``, ``M^2 = Theta/2``, one ordered pair contributes
``sum_{l,m} (-1)^m M_i^l . T^(l+m)(R) . M_j^m`` with ``T^(n) = grad^n phi``.
For a radial ``phi`` let ``B_0 = phi`` and ``B_{n+1} = -(1/R) dB_n/dR``; then
``T^(n) = sum_k (-1)^(n-k) B_(n-k) Sym_k(delta^k R^(n-2k))`` where ``Sym_k``
sums every placement of ``k`` disjoint index pairs (induction: ``d/dR_z`` of
``B_m`` gives ``-R_z B_(m+1)`` and of an ``R_a`` factor gives ``delta_az``).
Radial coefficients are derived here, not copied:

* ``1/R``: ``R^2 B_n = (2n-1) B_(n-1)``.
* ``erfc(bR)/R``: ``R^2 B_n = (2n-1) B_(n-1) + c_n exp(-b^2 R^2)`` with
  ``c_n = (2b^2)^n/(b sqrt(pi))``; induction applies ``-(1/R) d/dR`` to both
  sides using ``-(1/R) d/dR exp(-b^2R^2) = 2b^2 exp(-b^2R^2)``.
* ``erf(bR)/R``: the Taylor series
  ``B_n = 2b/sqrt(pi) (2b^2)^n sum_j (-b^2R^2)^j/(j! (2n+2j+1))`` (termwise
  application of ``-(1/R) d/dR`` to ``r^(2m)``) for ``bR < 1`` (this also gives
  the co-centered ``R = 0`` limits), else ``1/R - erfc`` above.

Periodic convention (tin-foil, ``k = 0`` excluded): the Fourier amplitude of
site ``j`` is ``A_j(k) = q_j + i k.mu_j - 1/2 k.Theta_j.k`` (``grad -> i k``),
so the full Fourier form is ``(k/V) sum_{k != 0} Phi(k) Re S_L(k) conj S_R(k)``
with ``S(k) = sum_j A_j exp(-sigma_j^2 k^2/2) exp(i k.r_j)``.

* Ewald split (point and Gaussian): ``Phi = 4 pi exp(-k^2/(4 a^2))/k^2``. The
  long-range kernel is the Gaussian pair kernel of effective
  ``s_eff^2 = s_ij^2 + 1/(2a^2)`` (point: ``erf(aR)/R``); real space sums
  ``phi_ij - erf(beta'_ij R)/R`` (``beta' = 1/sqrt(2 s_ij^2 + 1/a^2)``) over
  every image inside the level cutoff, including nonzero self images, and the
  smooth co-centered term ``k/2 O_i O_i erf(beta'_ii r)/r|_0`` contained in the
  reciprocal sum is subtracted. ``background=True`` (uniform background) adds
  ``-pi k Q^2/(2 a^2 V)``: the ``a``-dependent part of the charge-charge
  ``k -> 0`` limit ``4 pi (1/k^2 - s_eff^2/2)``; the ``a``-independent Gaussian
  form-factor constant belongs to the excluded ``k = 0`` term. Dipole and
  second-moment couplings have no ``a``-dependent ``k -> 0`` constant, so the
  sum is ``a``-independent.
* Absolutely convergent Gaussian Fourier series
  (:func:`gaussian_fourier_multipole_reference`): ``Phi = 4 pi/k^2`` with the
  physical co-centered term subtracted unless included. It equals the Ewald
  split plus its background term, because
  ``(phi_hat - phi_long_hat)(0) = pi/a^2``; no further term is added.

``background=False`` declares ``require-neutral`` and refuses a charged cell.
Strain is the column deformation ``F = I + e``: positions ``r F^T``, row vectors
``H F^T``, lab-frame moments and widths fixed. Real-space pair terms give
``dE/de_ab = sum dE/dR_a R_b``; ``k.r`` is invariant while
``dk_c/de_ab = -delta_bc k_a`` and ``dV/de_ab = V delta_ab``.

Polarization functional (isotropic ``alpha > 0`` in charge^2 length^2/energy):
``E(u) = 1/2 sum |u_i|^2/alpha_i + 1/2 B(u, u) + B(u, M)`` with induced
dipoles ``u`` on the site Gaussians (same widths), co-centered interactions
excluded, local cost retained and no permanent-permanent energy. With
``B(u, u) = u.G.u`` and ``c = d B(u, M)/du = -E_p`` the dense independent
operator is ``A = diag(1/alpha) + G`` and the stationary point solves
``A u = E_p``. At stationarity the energy is ``1/2 sum|u|^2/alpha +
1/2 B(u, u + 2M)`` and forces/strain follow from ``B(u, u + 2M)`` at fixed
``u`` (Hellmann-Feynman). TCG/truncated solvers approximate this stationary
reference; their difference is an approximation gap, never reference error.

Convergence: periodic results increase the dimensionless level ``x``
(``erfc``/Gaussian decay argument at both cutoffs) until energy, forces,
strain derivative (and induced dipoles) change by at most ``tolerance`` times
``max(1, |value|)`` on two successive refinements. Finite results are exact
pair sums. Derivatives are analytic throughout.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Iterator
from typing import Any, NamedTuple

import numpy as np
from scipy import special


__all__ = [
    "ReferenceResourceError",
    "ReferenceUnsupportedError",
    "gaussian_fourier_multipole_reference",
    "multipole_reference",
    "polarization_reference",
]


class ReferenceUnsupportedError(ValueError):
    """The input lies outside the declared reference definitions."""


class ReferenceResourceError(RuntimeError):
    """A declared reference bound was exhausted; ``history`` is retained."""

    def __init__(self, message: str, history: list[dict[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.history: list[dict[str, Any]] = [] if history is None else history


#: Highest radial derivative order (quadrupole-quadrupole gradient).
_ORDER = 5
#: Pair-image terms evaluated per dense derivative-tensor block.
_PAIR_CHUNK = 2048
#: Bound on selected real-space pair-image terms per level.
_MAXIMUM_REAL_TERMS = 2_000_000
#: Translations examined per enumeration block (times the atom count).
_TRANSLATION_BLOCK = 1 << 16
#: Taylor terms of the erf-kernel series (used for ``beta R < 1``).
_SERIES_TERMS = 40
#: Point sites closer than this (any image) are refused as singular.
_POINT_SEPARATION = 1.0e-6
#: Ewald split ``a = factor (N / V^2)^(1/6)``.
_ALPHA_FACTOR = 3.0
#: Dimensionless decay-argument schedule ``x`` of successive levels.
_FIRST_LEVEL = 4.0
_LEVEL_STEP = 0.5
_LAST_LEVEL = 9.0
#: Successive within-tolerance refinements required to accept a level.
_STABLE_COMPARISONS = 2
#: Relative total-charge bound of ``require-neutral``.
_NEUTRALITY = 1.0e-10
#: Relative antisymmetry bound of a second-moment matrix.
_SYMMETRY = 1.0e-12

_LETTERS = "abcdefg"
_BASIS = (
    "primitive Cartesian (q, mu, raw Theta = int rho s s); "
    "O(grad) = q - mu.grad + 1/2 Theta:grad grad"
)
_STRAIN = (
    "analytic column-strain dE/de: positions r(I+e)^T, rows H(I+e)^T, "
    "lab-frame moments and widths fixed"
)


# ------------------------------------------------------------------ radial


def _point_b(distance: np.ndarray, order: int, /) -> np.ndarray:
    out = np.empty((order + 1,) + distance.shape)
    inverse = 1.0 / (distance * distance)
    out[0] = 1.0 / distance
    for n in range(1, order + 1):
        out[n] = (2 * n - 1) * out[n - 1] * inverse
    return out


def _erf_series(distance: np.ndarray, beta: np.ndarray, order: int, /) -> np.ndarray:
    x = (beta * distance) ** 2
    terms = np.empty((_SERIES_TERMS,) + distance.shape)
    terms[0] = 1.0
    for j in range(1, _SERIES_TERMS):
        terms[j] = terms[j - 1] * (-x) / j
    out = np.empty((order + 1,) + distance.shape)
    prefactor = 2.0 * beta / math.sqrt(math.pi)
    for n in range(order + 1):
        denominators = 2.0 * (n + np.arange(_SERIES_TERMS)) + 1.0
        series = np.tensordot(1.0 / denominators, terms, axes=1)
        out[n] = prefactor * (2.0 * beta * beta) ** n * series
    return out


def _erfc_recursion(distance: np.ndarray, beta: np.ndarray, order: int, /) -> np.ndarray:
    out = np.empty((order + 1,) + distance.shape)
    inverse = 1.0 / (distance * distance)
    gauss = np.exp(-((beta * distance) ** 2))
    out[0] = special.erfc(beta * distance) / distance
    coefficient = 1.0 / (beta * math.sqrt(math.pi))
    for n in range(1, order + 1):
        coefficient = coefficient * 2.0 * beta * beta
        out[n] = ((2 * n - 1) * out[n - 1] + coefficient * gauss) * inverse
    return out


def _erf_b(distance: np.ndarray, beta: np.ndarray, order: int, /) -> np.ndarray:
    beta = np.broadcast_to(beta, distance.shape)
    out = np.empty((order + 1,) + distance.shape)
    near = beta * distance < 1.0
    far = ~near
    if np.any(near):
        out[:, near] = _erf_series(distance[near], beta[near], order)
    if np.any(far):
        out[:, far] = _point_b(distance[far], order) - _erfc_recursion(
            distance[far], beta[far], order
        )
    return out


def _erfc_b(distance: np.ndarray, beta: np.ndarray, order: int, /) -> np.ndarray:
    beta = np.broadcast_to(beta, distance.shape)
    out = np.empty((order + 1,) + distance.shape)
    near = beta * distance < 1.0
    far = ~near
    if np.any(near):
        out[:, near] = _point_b(distance[near], order) - _erf_series(
            distance[near], beta[near], order
        )
    if np.any(far):
        out[:, far] = _erfc_recursion(distance[far], beta[far], order)
    return out


def _real_b(
    distance: np.ndarray,
    physical: np.ndarray | None,
    smooth: np.ndarray | None,
    order: int,
    /,
) -> np.ndarray:
    """Radial coefficients of ``phi - phi_long`` (``smooth=None``: ``phi``)."""
    if smooth is None:
        return (
            _point_b(distance, order)
            if physical is None
            else _erf_b(distance, physical, order)
        )
    if physical is None:
        return _erfc_b(distance, smooth, order)
    # erf(b R)/R - erf(b' R)/R with b > b': erfc difference once b'R >= 1.
    out = np.empty((order + 1,) + distance.shape)
    far = smooth * distance >= 1.0
    near = ~far
    if np.any(far):
        out[:, far] = _erfc_recursion(
            distance[far], smooth[far], order
        ) - _erfc_recursion(distance[far], physical[far], order)
    if np.any(near):
        out[:, near] = _erf_b(distance[near], physical[near], order) - _erf_series(
            distance[near], smooth[near], order
        )
    return out


# ------------------------------------------------------- derivative tensors


def _partial_matchings(
    slots: tuple[int, ...], /
) -> Iterator[tuple[tuple[tuple[int, int], ...], tuple[int, ...]]]:
    if not slots:
        yield (), ()
        return
    first, rest = slots[0], slots[1:]
    for pairs, singles in _partial_matchings(rest):
        yield pairs, (first,) + singles
    for index, partner in enumerate(rest):
        remaining = rest[:index] + rest[index + 1 :]
        for pairs, singles in _partial_matchings(remaining):
            yield ((first, partner),) + pairs, singles


def _matching_table(order: int, /) -> list[list[tuple[int, str, int]]]:
    """Per rank: ``(pair count, einsum subscripts, single count)``."""
    table: list[list[tuple[int, str, int]]] = []
    for n in range(order + 1):
        rows: list[tuple[int, str, int]] = []
        for pairs, singles in _partial_matchings(tuple(range(n))):
            subscripts: list[str] = [_LETTERS[a] + _LETTERS[b] for a, b in pairs]
            subscripts += ["p" + _LETTERS[s] for s in singles]
            subscripts.append("p")
            rows.append(
                (
                    len(pairs),
                    ",".join(subscripts) + "->p" + _LETTERS[:n],
                    len(singles),
                )
            )
        table.append(rows)
    return table


_MATCHINGS = _matching_table(_ORDER)


def _derivative_tensors(
    vectors: np.ndarray, b: np.ndarray, order: int, /
) -> list[np.ndarray]:
    """``T^(n) = grad^n phi`` for ``n <= order`` of radial ``phi`` (``b`` its B_n)."""
    eye = np.eye(3)
    count = vectors.shape[0]
    tensors = []
    for n in range(order + 1):
        total = np.zeros((count,) + (3,) * n)
        for pairs, subscripts, singles in _MATCHINGS[n]:
            coefficient = (-1.0) ** (n - pairs) * b[n - pairs]
            operands = [eye] * pairs + [vectors] * singles + [coefficient]
            total += np.einsum(subscripts, *operands)
        tensors.append(total)
    return tensors


def _contract(
    tensor: np.ndarray,
    left_degree: int,
    left: np.ndarray,
    right_degree: int,
    right: np.ndarray,
    free: int,
    /,
) -> np.ndarray:
    lt = _LETTERS[:left_degree]
    rt = _LETTERS[left_degree : left_degree + right_degree]
    ft = _LETTERS[left_degree + right_degree : left_degree + right_degree + free]
    return np.einsum(f"p{lt}{rt}{ft},p{lt},p{rt}->p{ft}", tensor, left, right)


# ----------------------------------------------------------------- moments


class _Moments(NamedTuple):
    charge: np.ndarray
    dipole: np.ndarray
    half_second: np.ndarray

    def parts(self, index: np.ndarray, /) -> tuple[np.ndarray, ...]:
        return self.charge[index], self.dipole[index], self.half_second[index]

    def total(self) -> float:
        return float(np.sum(self.charge))

    def plus_scaled(self, other: _Moments, factor: float, /) -> _Moments:
        """``self + factor * other`` componentwise."""
        return _Moments(
            self.charge + factor * other.charge,
            self.dipole + factor * other.dipole,
            self.half_second + factor * other.half_second,
        )


def _pair_terms(
    tensors: list[np.ndarray],
    left: tuple[np.ndarray, ...],
    right: tuple[np.ndarray, ...],
    /,
    *,
    gradient: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    """``e = sum (-1)^m L^l.T^(l+m).R^m`` and (optionally) ``de/dR``."""
    value = np.zeros(tensors[0].shape[0])
    slope = np.zeros((tensors[0].shape[0], 3)) if gradient else None
    for l in range(3):
        for m in range(3):
            sign = -1.0 if m % 2 else 1.0
            value += sign * _contract(tensors[l + m], l, left[l], m, right[m], 0)
            if slope is not None:
                slope += sign * _contract(tensors[l + m + 1], l, left[l], m, right[m], 1)
    return value, slope


def _field_terms(
    tensors: list[np.ndarray], right: tuple[np.ndarray, ...], /
) -> np.ndarray:
    """``d e(u, R)/du = sum_m (-1)^m T^(1+m).R^m`` (minus the field)."""
    count = tensors[0].shape[0]
    ones = np.ones(count)
    total = np.zeros((count, 3))
    for m in range(3):
        sign = -1.0 if m % 2 else 1.0
        total += sign * _contract(tensors[1 + m], 0, ones, m, right[m], 1)
    return total


# ----------------------------------------------------------------- geometry


class _Geometry(NamedTuple):
    positions: np.ndarray
    vectors: np.ndarray | None
    volume: float | None
    widths: np.ndarray | None

    @property
    def count(self) -> int:
        return self.positions.shape[0]

    @property
    def site_kind(self) -> str:
        return "point" if self.widths is None else "gaussian"


class _Level(NamedTuple):
    method: str  # "finite" | "ewald" | "fourier"
    parameter: float | None
    alpha: float | None
    real_cutoff: float | None
    reciprocal_cutoff: float | None


class _Stats(NamedTuple):
    real_terms: int
    real_extent: list[int] | None
    modes: int
    reciprocal_extent: list[int] | None


def _float_array(value: Any, name: str, shape: tuple[int, ...], /) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape:
        raise ReferenceUnsupportedError(
            f"{name} must have shape {shape}, got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ReferenceUnsupportedError(f"{name} must be finite.")
    return array


def _positive_int(value: Any, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ReferenceUnsupportedError(f"{name} must be a positive integer.")
    return int(value)


def _validated(
    positions: Any,
    charges: Any,
    dipoles: Any,
    second_moments: Any,
    vectors: Any,
    widths: Any,
    /,
    *,
    background: bool,
    coulomb_constant: float,
    tolerance: float,
    maximum_shells: int,
    mode_chunk: int,
    maximum_atoms: int,
) -> tuple[_Geometry, _Moments, dict[str, Any]]:
    maximum_atoms = _positive_int(maximum_atoms, "maximum_atoms")
    maximum_shells = _positive_int(maximum_shells, "maximum_shells")
    mode_chunk = _positive_int(mode_chunk, "mode_chunk")
    shape = np.shape(positions)
    if len(shape) != 2 or shape[1] != 3 or shape[0] < 1:
        raise ReferenceUnsupportedError("positions must have shape (N, 3), N >= 1.")
    count = int(shape[0])
    if count > maximum_atoms:
        raise ReferenceResourceError(
            f"{count} atoms exceed the reference bound maximum_atoms={maximum_atoms}."
        )
    x = _float_array(positions, "positions", (count, 3))
    q = _float_array(charges, "charges", (count,))
    mu = _float_array(dipoles, "dipoles", (count, 3))
    theta = _float_array(second_moments, "second_moments", (count, 3, 3))
    asymmetry = np.max(np.abs(theta - np.swapaxes(theta, 1, 2)))
    if asymmetry > _SYMMETRY * max(1.0, float(np.max(np.abs(theta)))):
        raise ReferenceUnsupportedError(
            "second_moments must be symmetric primitive Cartesian matrices."
        )
    theta = 0.5 * (theta + np.swapaxes(theta, 1, 2))
    if not (math.isfinite(tolerance) and tolerance > 0.0):
        raise ReferenceUnsupportedError("tolerance must be positive and finite.")
    if not (math.isfinite(coulomb_constant) and coulomb_constant > 0.0):
        raise ReferenceUnsupportedError("coulomb_constant must be positive and finite.")
    sigma = None
    if widths is not None:
        sigma = _float_array(widths, "widths", (count,))
        if np.any(sigma <= 0.0):
            raise ReferenceUnsupportedError(
                "Gaussian widths must be positive; widths=None selects point sites."
            )
    cell = None
    volume = None
    if vectors is not None:
        cell = _float_array(vectors, "vectors", (3, 3))
        volume = abs(float(np.linalg.det(cell)))
        if volume <= 1.0e-12 * float(np.prod(np.linalg.norm(cell, axis=1))):
            raise ReferenceUnsupportedError("vectors must span a full 3D cell.")
        fractional = x @ np.linalg.inv(cell)
        x = (fractional - np.floor(fractional)) @ cell
        total = float(np.sum(q))
        if not background and abs(total) > _NEUTRALITY * max(
            1.0, float(np.sum(np.abs(q)))
        ):
            raise ReferenceUnsupportedError(
                f"Charged cell (Q={total!r}) under require-neutral; "
                "pass background=True for the uniform-background convention."
            )
    elif background:
        raise ReferenceUnsupportedError(
            "background applies to periodic cells only; finite sums have no k = 0 term."
        )
    support = {
        "atoms": count,
        "maximum_atoms": maximum_atoms,
        "maximum_shells": maximum_shells,
        "mode_chunk": mode_chunk,
        "pair_chunk": _PAIR_CHUNK,
        "maximum_real_terms": _MAXIMUM_REAL_TERMS,
        "level_schedule": [_FIRST_LEVEL, _LEVEL_STEP, _LAST_LEVEL],
        "stable_comparisons": _STABLE_COMPARISONS,
    }
    return (
        _Geometry(x, cell, volume, sigma),
        _Moments(q, mu, 0.5 * theta),
        support,
    )


def _heights(vectors: np.ndarray, /) -> np.ndarray:
    volume = abs(float(np.linalg.det(vectors)))
    return np.asarray(
        [
            volume
            / float(np.linalg.norm(np.cross(vectors[(a + 1) % 3], vectors[(a + 2) % 3])))
            for a in range(3)
        ]
    )


def _pair_betas(
    geometry: _Geometry, left: np.ndarray, right: np.ndarray, alpha: float | None, /
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Physical ``beta_ij`` (``None``: point) and Ewald ``beta'_ij`` (``None``: none)."""
    if geometry.widths is None:
        return None, None if alpha is None else np.full(left.shape, alpha)
    pair = geometry.widths[left] ** 2 + geometry.widths[right] ** 2
    physical = 1.0 / np.sqrt(2.0 * pair)
    if alpha is None:
        return physical, None
    return physical, 1.0 / np.sqrt(2.0 * pair + 1.0 / alpha**2)


def _real_pairs(
    geometry: _Geometry, level: _Level, maximum_shells: int, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int] | None]:
    """Ordered pair-image terms ``(i, j, R)`` without the co-centered term."""
    count = geometry.count
    x = geometry.positions
    if level.method == "finite":
        left, right = np.nonzero(~np.eye(count, dtype=np.bool_))
        terms = (left, right, x[left] - x[right], None)
    else:
        assert geometry.vectors is not None and level.real_cutoff is not None
        cutoff = level.real_cutoff
        extent = [
            int(math.ceil(cutoff / height)) + 1 for height in _heights(geometry.vectors)
        ]
        if max(extent) > maximum_shells:
            raise ReferenceResourceError(
                f"Real-space image extent {extent} exceeds maximum_shells="
                f"{maximum_shells} at cutoff {cutoff!r}."
            )
        integers = np.asarray(
            list(itertools.product(*(range(-e, e + 1) for e in extent))),
            dtype=np.float64,
        )
        zero = np.flatnonzero(np.all(integers == 0.0, axis=1))[0]
        translations = integers @ geometry.vectors
        block = max(1, _TRANSLATION_BLOCK // count)
        lefts, rights, displacements = [], [], []
        selected = 0
        for i in range(count):
            base = x[i] - x
            for start in range(0, translations.shape[0], block):
                shift = translations[start : start + block]
                vectors = base[:, None, :] + shift[None, :, :]
                keep = np.sum(vectors * vectors, axis=-1) <= cutoff * cutoff
                if start <= zero < start + block:
                    keep[i, zero - start] = False
                rows, columns = np.nonzero(keep)
                selected += rows.size
                if selected > _MAXIMUM_REAL_TERMS:
                    raise ReferenceResourceError(
                        f"More than {_MAXIMUM_REAL_TERMS} real-space pair-image "
                        f"terms at cutoff {cutoff!r}."
                    )
                lefts.append(np.full(rows.size, i))
                rights.append(rows)
                displacements.append(vectors[rows, columns])
        terms = (
            np.concatenate(lefts),
            np.concatenate(rights),
            np.concatenate(displacements).reshape(-1, 3),
            extent,
        )
    if geometry.widths is None and terms[2].shape[0]:
        closest = float(np.min(np.linalg.norm(terms[2], axis=-1)))
        if closest < _POINT_SEPARATION:
            raise ReferenceUnsupportedError(
                f"Distinct point sites (or images) at separation {closest!r}: "
                "the point interaction is singular."
            )
    return terms


def _modes(
    geometry: _Geometry, cutoff: float, maximum_shells: int, /
) -> tuple[np.ndarray, list[int]]:
    assert geometry.vectors is not None
    lengths = np.linalg.norm(geometry.vectors, axis=1)
    extent = [int(math.ceil(cutoff * length / (2.0 * math.pi))) for length in lengths]
    if max(extent) > maximum_shells:
        raise ReferenceResourceError(
            f"Reciprocal extent {extent} exceeds maximum_shells={maximum_shells} "
            f"at cutoff {cutoff!r}."
        )
    integers = np.asarray(
        list(itertools.product(*(range(-e, e + 1) for e in extent))), dtype=np.float64
    )
    integers = integers[np.any(integers != 0.0, axis=1)]
    k = integers @ (2.0 * math.pi * np.linalg.inv(geometry.vectors).T)
    return k[np.sum(k * k, axis=1) <= cutoff * cutoff], extent


def _self_tensors(beta: np.ndarray, order: int, /) -> list[np.ndarray]:
    zeros = np.zeros(beta.shape)
    return _derivative_tensors(
        np.zeros(beta.shape + (3,)), _erf_series(zeros, beta, order), order
    )


def _smooth_self_beta(geometry: _Geometry, alpha: float | None, /) -> np.ndarray:
    """Co-centered ``beta`` of the reciprocal kernel (``alpha=None``: physical)."""
    if geometry.widths is None:
        assert alpha is not None
        return np.full(geometry.count, alpha)
    split = 0.0 if alpha is None else 1.0 / alpha**2
    return 1.0 / np.sqrt(4.0 * geometry.widths**2 + split)


# ------------------------------------------------------------ bilinear form


def _bilinear(
    geometry: _Geometry,
    left: _Moments,
    right: _Moments,
    level: _Level,
    /,
    *,
    include_self: bool,
    background: bool,
    maximum_shells: int,
    mode_chunk: int,
) -> tuple[float, np.ndarray, np.ndarray, _Stats]:
    """``B(L, R)/k`` with ``dB/dr`` and ``dB/de`` (analytic)."""
    count = geometry.count
    index = np.arange(count)
    value = 0.0
    gradient = np.zeros((count, 3))
    strain = np.zeros((3, 3))
    real_terms, real_extent = 0, None
    if level.method != "fourier":
        lefts, rights, displacements, real_extent = _real_pairs(
            geometry, level, maximum_shells
        )
        real_terms = int(lefts.size)
        for start in range(0, real_terms, _PAIR_CHUNK):
            i = lefts[start : start + _PAIR_CHUNK]
            j = rights[start : start + _PAIR_CHUNK]
            r = displacements[start : start + _PAIR_CHUNK]
            physical, smooth = _pair_betas(geometry, i, j, level.alpha)
            b = _real_b(np.linalg.norm(r, axis=1), physical, smooth, _ORDER)
            tensors = _derivative_tensors(r, b, _ORDER)
            e, g = _pair_terms(tensors, left.parts(i), right.parts(j), gradient=True)
            assert g is not None
            value += float(np.sum(e))
            np.add.at(gradient, i, g)
            np.add.at(gradient, j, -g)
            strain += g.T @ r
    if include_self:
        assert geometry.widths is not None
        tensors = _self_tensors(0.5 / geometry.widths, _ORDER - 1)
        e, _ = _pair_terms(
            tensors,
            left.parts(index),
            right.parts(index),
            gradient=False,
        )
        value += float(np.sum(e))
    modes, reciprocal_extent = 0, None
    if level.method != "finite":
        assert geometry.volume is not None and level.reciprocal_cutoff is not None
        k_all, reciprocal_extent = _modes(
            geometry, level.reciprocal_cutoff, maximum_shells
        )
        modes = int(k_all.shape[0])
        rec, rec_gradient, rec_strain = _reciprocal_bilinear(
            geometry, k_all, level.alpha, left, right, mode_chunk
        )
        value += rec
        gradient += rec_gradient
        strain += rec_strain
        tensors = _self_tensors(_smooth_self_beta(geometry, level.alpha), _ORDER - 1)
        e, _ = _pair_terms(
            tensors,
            left.parts(index),
            right.parts(index),
            gradient=False,
        )
        value -= float(np.sum(e))
        if background and level.alpha is not None:
            term = (
                -math.pi
                * left.total()
                * right.total()
                / (level.alpha**2 * geometry.volume)
            )
            value += term
            strain -= term * np.eye(3)
    return (
        value,
        gradient,
        strain,
        _Stats(real_terms, real_extent, modes, reciprocal_extent),
    )


def _amplitudes(moments: _Moments, k: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    """``A_j(k)`` (C, N) and ``dA_j/dk`` (C, N, 3) of primitive moments."""
    amplitude = (
        moments.charge[None, :]
        + 1j * (k @ moments.dipole.T)
        - np.einsum("ca,nab,cb->cn", k, moments.half_second, k)
    )
    slope = 1j * moments.dipole[None, :, :] - 2.0 * np.einsum(
        "nab,cb->cna", moments.half_second, k
    )
    return amplitude, slope


def _kernel_factor(
    k2: np.ndarray, alpha: float | None, /
) -> tuple[np.ndarray, np.ndarray]:
    """``Phi(k)`` and ``(dPhi/dk_b)/(Phi k_b)``."""
    if alpha is None:
        return 4.0 * math.pi / k2, -2.0 / k2
    return (
        4.0 * math.pi * np.exp(-k2 / (4.0 * alpha**2)) / k2,
        -(2.0 / k2 + 1.0 / (2.0 * alpha**2)),
    )


def _reciprocal_bilinear(
    geometry: _Geometry,
    k_all: np.ndarray,
    alpha: float | None,
    left: _Moments,
    right: _Moments,
    mode_chunk: int,
    /,
) -> tuple[float, np.ndarray, np.ndarray]:
    assert geometry.volume is not None
    count = geometry.count
    x = geometry.positions
    variance = np.zeros(count) if geometry.widths is None else geometry.widths**2
    value = 0.0
    gradient = np.zeros((count, 3))
    strain = np.zeros((3, 3))
    for start in range(0, k_all.shape[0], mode_chunk):
        k = k_all[start : start + mode_chunk]
        k2 = np.sum(k * k, axis=1)
        phi, slope_ratio = _kernel_factor(k2, alpha)
        z = np.exp(1j * (k @ x.T))
        w = np.exp(-0.5 * k2[:, None] * variance[None, :])
        dw = -variance[None, :, None] * k[:, None, :] * w[:, :, None]
        a_left, da_left = _amplitudes(left, k)
        a_right, da_right = _amplitudes(right, k)
        weighted_left = a_left * w * z
        weighted_right = a_right * w * z
        s_left = np.sum(weighted_left, axis=1)
        s_right = np.sum(weighted_right, axis=1)
        product = np.real(s_left * np.conj(s_right))
        value += float(np.sum(phi * product))
        difference = np.imag(
            weighted_left * np.conj(s_right)[:, None]
            - s_left[:, None] * np.conj(weighted_right)
        )
        gradient -= np.einsum("c,ca,cn->na", phi, k, difference)
        ds_left = np.sum(
            (da_left * w[:, :, None] + a_left[:, :, None] * dw) * z[:, :, None], axis=1
        )
        ds_right = np.sum(
            (da_right * w[:, :, None] + a_right[:, :, None] * dw) * z[:, :, None],
            axis=1,
        )
        inner = (
            phi[:, None]
            * np.real(
                ds_left * np.conj(s_right)[:, None] + s_left[:, None] * np.conj(ds_right)
            )
            + (phi * slope_ratio * product)[:, None] * k
        )
        strain -= k.T @ inner
    value /= geometry.volume
    gradient /= geometry.volume
    strain = strain / geometry.volume - value * np.eye(3)
    return value, gradient, strain


# ------------------------------------------------------------- polarization


def _polarization_system(
    geometry: _Geometry,
    permanent: _Moments,
    level: _Level,
    /,
    *,
    maximum_shells: int,
    mode_chunk: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense ``G`` (3N, 3N) with ``B(u, u) = u.G.u`` and ``c = dB(u, M)/du``, over ``k``."""
    count = geometry.count
    index = np.arange(count)
    pair_blocks = np.zeros((count * count, 3, 3))
    c = np.zeros((count, 3))
    lefts, rights, displacements, _ = _real_pairs(geometry, level, maximum_shells)
    for start in range(0, lefts.size, _PAIR_CHUNK):
        i = lefts[start : start + _PAIR_CHUNK]
        j = rights[start : start + _PAIR_CHUNK]
        r = displacements[start : start + _PAIR_CHUNK]
        physical, smooth = _pair_betas(geometry, i, j, level.alpha)
        b = _real_b(np.linalg.norm(r, axis=1), physical, smooth, 3)
        tensors = _derivative_tensors(r, b, 3)
        np.add.at(pair_blocks, i * count + j, -tensors[2])
        np.add.at(c, i, _field_terms(tensors, permanent.parts(j)))
    g = (
        pair_blocks.reshape(count, count, 3, 3)
        .transpose(0, 2, 1, 3)
        .reshape(3 * count, 3 * count)
    )
    if level.method == "ewald":
        assert geometry.volume is not None and level.reciprocal_cutoff is not None
        k_all, _ = _modes(geometry, level.reciprocal_cutoff, maximum_shells)
        variance = np.zeros(count) if geometry.widths is None else geometry.widths**2
        for start in range(0, k_all.shape[0], mode_chunk):
            k = k_all[start : start + mode_chunk]
            k2 = np.sum(k * k, axis=1)
            phi, _ = _kernel_factor(k2, level.alpha)
            z = np.exp(1j * (k @ geometry.positions.T))
            w = np.exp(-0.5 * k2[:, None] * variance[None, :])
            cosine = (k[:, None, :] * (w * z.real)[:, :, None]).reshape(len(k), -1)
            sine = (k[:, None, :] * (w * z.imag)[:, :, None]).reshape(len(k), -1)
            g += (
                (cosine * phi[:, None]).T @ cosine + (sine * phi[:, None]).T @ sine
            ) / geometry.volume
            amplitude, _ = _amplitudes(permanent, k)
            structure = np.sum(amplitude * w * z, axis=1)
            c -= (
                np.einsum(
                    "c,ca,cn->na", phi, k, w * np.imag(z * np.conj(structure)[:, None])
                )
                / geometry.volume
            )
        tensors = _self_tensors(_smooth_self_beta(geometry, level.alpha), 3)
        for site in range(count):
            block = slice(3 * site, 3 * site + 3)
            g[block, block] += tensors[2][site]
        c -= _field_terms(tensors, permanent.parts(index))
    return g, c.reshape(-1)


# ---------------------------------------------------------------- schedule


def _ewald_alpha(geometry: _Geometry, factor: float, /) -> float:
    assert geometry.volume is not None
    return factor * (geometry.count / geometry.volume**2) ** (1.0 / 6.0)


def _level(geometry: _Geometry, method: str, x: float, alpha: float | None, /) -> _Level:
    if method == "ewald":
        assert alpha is not None
        if geometry.widths is None:
            slowest = alpha
        else:
            widest = float(np.max(geometry.widths))
            slowest = 1.0 / math.sqrt(4.0 * widest**2 + 1.0 / alpha**2)
        return _Level("ewald", x, alpha, x / slowest, 2.0 * alpha * x)
    assert method == "fourier" and geometry.widths is not None
    return _Level("fourier", x, None, None, x / float(np.min(geometry.widths)))


def _change(current: np.ndarray | float, previous: np.ndarray | float, /) -> float:
    return float(np.max(np.abs(np.asarray(current) - np.asarray(previous))))


def _scale(value: np.ndarray | float, /) -> float:
    return max(1.0, float(np.max(np.abs(np.asarray(value)))))


def _converge(
    geometry: _Geometry,
    method: str,
    alpha: float | None,
    tolerance: float,
    evaluate: Callable[[_Level], tuple[dict[str, Any], _Stats]],
    quantities: tuple[str, ...],
    /,
) -> tuple[dict[str, Any], _Stats, _Level, list[dict[str, Any]]]:
    history: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    stable = 0
    steps = int(round((_LAST_LEVEL - _FIRST_LEVEL) / _LEVEL_STEP))
    for step in range(steps + 1):
        level = _level(geometry, method, _FIRST_LEVEL + step * _LEVEL_STEP, alpha)
        try:
            outcome, stats = evaluate(level)
        except ReferenceResourceError as error:
            raise ReferenceResourceError(
                f"Reference resources exhausted before convergence: {error}", history
            ) from error
        record: dict[str, Any] = {
            "level": step,
            "parameter": level.parameter,
            "alpha": level.alpha,
            "real_cutoff": level.real_cutoff,
            "reciprocal_cutoff": level.reciprocal_cutoff,
            "real_extent": stats.real_extent,
            "reciprocal_extent": stats.reciprocal_extent,
            "real_terms": stats.real_terms,
            "modes": stats.modes,
            "energy": float(outcome["energy"]),
        }
        within = previous is not None
        for name in quantities:
            if previous is None:
                record[f"{name}_change"] = None
                continue
            change = _change(outcome[name], previous[name])
            record[f"{name}_change"] = change
            within = within and change <= tolerance * _scale(outcome[name])
        record["within_tolerance"] = bool(within)
        history.append(record)
        stable = stable + 1 if within else 0
        if stable >= _STABLE_COMPARISONS:
            return outcome, stats, level, history
        previous = outcome
    raise ReferenceResourceError(
        f"Reference did not converge to tolerance {tolerance!r} by level "
        f"{_LAST_LEVEL!r}.",
        history,
    )


# ------------------------------------------------------------------ public


def _self_convention(geometry: _Geometry, include: bool, /) -> str:
    images = (
        "nonzero periodic self images included"
        if geometry.vectors is not None
        else ("finite: no images")
    )
    if geometry.widths is None:
        return f"co-centered (i=j, n=0) point terms excluded; {images}"
    if include:
        return (
            "co-centered Gaussian term included as k/2 O(-grad)O(grad) "
            f"erf(r/(2 sigma))/r at 0; {images}"
        )
    return f"co-centered (i=j, n=0) Gaussian terms excluded; {images}"


def _convention(geometry: _Geometry, method: str, background: bool, /) -> str:
    kernel = "phi=1/R" if geometry.widths is None else "phi=erf(R/sqrt(2(s_i^2+s_j^2)))/R"
    if method == "finite":
        return f"finite exact ordered pair sum, E=1/2 B(M,M), {kernel}"
    neutrality = (
        "uniform-background (-pi k Q^2/(2 a^2 V) in the Ewald split)"
        if background
        else "require-neutral"
    )
    route = (
        "Ewald split (explicit real images, spherical reciprocal modes)"
        if method == "ewald"
        else "absolutely convergent Gaussian Fourier series"
    )
    return (
        f"periodic tin-foil, k=0 excluded, {neutrality}, alpha-independent Gaussian "
        f"form-factor constant excluded with k=0; {route}; {kernel}"
    )


def _multipole(
    positions: Any,
    charges: Any,
    dipoles: Any,
    second_moments: Any,
    vectors: Any,
    /,
    *,
    widths: Any,
    include_coincident_self: bool,
    background: bool,
    coulomb_constant: float,
    tolerance: float,
    maximum_shells: int,
    mode_chunk: int,
    maximum_atoms: int,
    method: str,
    alpha_factor: float = _ALPHA_FACTOR,
) -> dict[str, Any]:
    geometry, moments, support = _validated(
        positions,
        charges,
        dipoles,
        second_moments,
        vectors,
        widths,
        background=background,
        coulomb_constant=coulomb_constant,
        tolerance=tolerance,
        maximum_shells=maximum_shells,
        mode_chunk=mode_chunk,
        maximum_atoms=maximum_atoms,
    )
    if include_coincident_self and geometry.widths is None:
        raise ReferenceUnsupportedError(
            "The co-centered point self energy is infinite; include_coincident_self "
            "requires Gaussian widths."
        )
    if geometry.vectors is None:
        method = "finite"
    elif method == "fourier" and geometry.widths is None:
        raise ReferenceUnsupportedError(
            "The Fourier series diverges for point sites; use the Ewald split."
        )
    constant = float(coulomb_constant)

    def evaluate(level: _Level) -> tuple[dict[str, Any], _Stats]:
        value, gradient, strain, stats = _bilinear(
            geometry,
            moments,
            moments,
            level,
            include_self=include_coincident_self,
            background=background,
            maximum_shells=support["maximum_shells"],
            mode_chunk=support["mode_chunk"],
        )
        return {
            "energy": 0.5 * constant * value,
            "forces": -0.5 * constant * gradient,
            "strain_derivative": 0.5 * constant * strain,
        }, stats

    quantities = ("energy", "forces", "strain_derivative")
    if method == "finite":
        level = _Level("finite", None, None, None, None)
        outcome, stats = evaluate(level)
        history = [
            {
                "level": "exact",
                "real_terms": stats.real_terms,
                "energy": float(outcome["energy"]),
                "energy_change": None,
                "forces_change": None,
                "strain_derivative_change": None,
                "within_tolerance": True,
            }
        ]
    else:
        alpha = _ewald_alpha(geometry, alpha_factor) if method == "ewald" else None
        outcome, stats, level, history = _converge(
            geometry, method, alpha, tolerance, evaluate, quantities
        )
    return _result(
        geometry,
        outcome,
        stats,
        level,
        history,
        support,
        kind=f"independent-{method}-{geometry.site_kind}-multipole",
        convention=_convention(geometry, method, background),
        self_convention=_self_convention(geometry, include_coincident_self),
        background=background,
        tolerance=tolerance,
        constant=constant,
    )


def _result(
    geometry: _Geometry,
    outcome: dict[str, Any],
    stats: _Stats,
    level: _Level,
    history: list[dict[str, Any]],
    support: dict[str, Any],
    /,
    *,
    kind: str,
    convention: str,
    self_convention: str,
    background: bool,
    tolerance: float,
    constant: float,
) -> dict[str, Any]:
    periodic = geometry.vectors is not None
    strain = np.asarray(outcome["strain_derivative"], dtype=np.float64)
    support = dict(support)
    support.update(
        {
            "method": level.method,
            "level_parameter": level.parameter,
            "alpha": level.alpha,
            "real_cutoff": level.real_cutoff,
            "reciprocal_cutoff": level.reciprocal_cutoff,
            "real_extent": stats.real_extent,
            "reciprocal_extent": stats.reciprocal_extent,
            "real_terms": stats.real_terms,
            "modes": stats.modes,
        }
    )
    result = {
        "kind": kind,
        "independent": True,
        "site_kind": geometry.site_kind,
        "basis": _BASIS,
        "coulomb_constant": constant,
        "energy": float(outcome["energy"]),
        "forces": np.asarray(outcome["forces"], dtype=np.float64),
        "strain_derivative": strain,
        "stress": strain / geometry.volume if periodic else None,
        "strain_method": _STRAIN,
        "convention": convention,
        "self_convention": self_convention,
        "neutrality": (
            None
            if not periodic
            else ("uniform-background" if background else "require-neutral")
        ),
        "converged": True,
        "convergence": (
            "exact finite sum"
            if not periodic
            else f"E/F/S changes <= tolerance*max(1,|value|) on "
            f"{_STABLE_COMPARISONS} successive refinements"
        ),
        "tolerance": tolerance,
        "history": history,
        "support": support,
        "derivatives": "analytic",
    }
    for name, value in outcome.items():
        result.setdefault(name, value)
    return result


def multipole_reference(
    positions: Any,
    charges: Any,
    dipoles: Any,
    second_moments: Any,
    vectors: Any,
    /,
    *,
    widths: Any = None,
    include_coincident_self: bool = False,
    background: bool = False,
    coulomb_constant: float = 1.0,
    tolerance: float = 1.0e-10,
    maximum_shells: int = 24,
    mode_chunk: int = 512,
    maximum_atoms: int = 64,
) -> dict[str, Any]:
    """Independent finite or periodic (Ewald-split) degree-two multipole energy.

    ``vectors=None`` is the exact finite pair sum; a ``(3, 3)`` row-vector cell
    selects the converged tin-foil Ewald split for point and Gaussian sites.
    Returns energy, forces, analytic strain derivative (finite systems
    included), stress (periodic only), conventions, support and history.
    """
    return _multipole(
        positions,
        charges,
        dipoles,
        second_moments,
        vectors,
        widths=widths,
        include_coincident_self=include_coincident_self,
        background=background,
        coulomb_constant=coulomb_constant,
        tolerance=tolerance,
        maximum_shells=maximum_shells,
        mode_chunk=mode_chunk,
        maximum_atoms=maximum_atoms,
        method="ewald",
    )


def gaussian_fourier_multipole_reference(
    positions: Any,
    charges: Any,
    dipoles: Any,
    second_moments: Any,
    vectors: Any,
    /,
    *,
    widths: Any,
    include_coincident_self: bool = False,
    background: bool = False,
    coulomb_constant: float = 1.0,
    tolerance: float = 1.0e-10,
    maximum_shells: int = 24,
    mode_chunk: int = 512,
    maximum_atoms: int = 64,
) -> dict[str, Any]:
    """Periodic Gaussian multipoles by the absolutely convergent Fourier series.

    A separately derived route (no real-space split, no background term) of
    the same definition as :func:`multipole_reference`; requires widths and a
    cell.
    """
    if widths is None or vectors is None:
        raise ReferenceUnsupportedError(
            "The Fourier-series reference needs Gaussian widths and a periodic cell."
        )
    return _multipole(
        positions,
        charges,
        dipoles,
        second_moments,
        vectors,
        widths=widths,
        include_coincident_self=include_coincident_self,
        background=background,
        coulomb_constant=coulomb_constant,
        tolerance=tolerance,
        maximum_shells=maximum_shells,
        mode_chunk=mode_chunk,
        maximum_atoms=maximum_atoms,
        method="fourier",
    )


def _polarization(
    positions: Any,
    charges: Any,
    dipoles: Any,
    second_moments: Any,
    polarizabilities: Any,
    vectors: Any,
    /,
    *,
    widths: Any,
    background: bool,
    coulomb_constant: float,
    tolerance: float,
    maximum_shells: int,
    mode_chunk: int,
    maximum_atoms: int,
    alpha_factor: float = _ALPHA_FACTOR,
) -> dict[str, Any]:
    geometry, permanent, support = _validated(
        positions,
        charges,
        dipoles,
        second_moments,
        vectors,
        widths,
        background=background,
        coulomb_constant=coulomb_constant,
        tolerance=tolerance,
        maximum_shells=maximum_shells,
        mode_chunk=mode_chunk,
        maximum_atoms=maximum_atoms,
    )
    count = geometry.count
    polarizability = _float_array(polarizabilities, "polarizabilities", (count,))
    if np.any(polarizability <= 0.0):
        raise ReferenceUnsupportedError(
            "Polarizabilities must be positive isotropic values (alpha > 0)."
        )
    constant = float(coulomb_constant)
    local = np.repeat(1.0 / polarizability, 3)
    zeros = np.zeros(count)
    stationarity: dict[str, Any] = {}

    def evaluate(level: _Level) -> tuple[dict[str, Any], _Stats]:
        g, c = _polarization_system(
            geometry,
            permanent,
            level,
            maximum_shells=support["maximum_shells"],
            mode_chunk=support["mode_chunk"],
        )
        g, c = constant * g, constant * c
        operator = g + np.diag(local)
        asymmetry = float(np.max(np.abs(operator - operator.T)))
        symmetric = 0.5 * (operator + operator.T)
        eigenvalues = np.linalg.eigvalsh(symmetric)
        if not eigenvalues[0] > 0.0:
            raise ReferenceUnsupportedError(
                "The dense polarization operator diag(1/alpha) + G is not positive "
                f"definite (minimum eigenvalue {float(eigenvalues[0])!r}): the "
                "functional has no stationary minimum."
            )
        induced = np.linalg.solve(symmetric, -c)
        residual = symmetric @ induced + c
        induced -= np.linalg.solve(symmetric, residual)
        residual = symmetric @ induced + c
        u = induced.reshape(count, 3)
        left = _Moments(zeros, u, np.zeros((count, 3, 3)))
        value, gradient, strain, stats = _bilinear(
            geometry,
            left,
            left.plus_scaled(permanent, 2.0),
            level,
            include_self=False,
            background=False,
            maximum_shells=support["maximum_shells"],
            mode_chunk=support["mode_chunk"],
        )
        energy = 0.5 * float(induced @ (local * induced)) + 0.5 * constant * value
        stationary = 0.5 * float(induced @ c)
        stationarity.update(
            {
                "residual_max_abs": float(np.max(np.abs(residual))),
                "residual_relative": float(
                    np.linalg.norm(residual)
                    / max(np.linalg.norm(c), np.finfo(np.float64).tiny)
                ),
                "minimum_eigenvalue": float(eigenvalues[0]),
                "maximum_eigenvalue": float(eigenvalues[-1]),
                "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
                "operator_asymmetry": asymmetry,
                "stationary_energy": stationary,
                "functional_consistency": abs(energy - stationary),
            }
        )
        return {
            "energy": energy,
            "forces": -0.5 * constant * gradient,
            "strain_derivative": 0.5 * constant * strain,
            "induced_dipoles": u,
            "permanent_field": -c.reshape(count, 3),
        }, stats

    quantities = ("energy", "forces", "strain_derivative", "induced_dipoles")
    if geometry.vectors is None:
        level = _Level("finite", None, None, None, None)
        outcome, stats = evaluate(level)
        history = [
            {
                "level": "exact",
                "real_terms": stats.real_terms,
                "energy": float(outcome["energy"]),
                **{f"{name}_change": None for name in quantities},
                "within_tolerance": True,
            }
        ]
    else:
        alpha = _ewald_alpha(geometry, alpha_factor)
        outcome, stats, level, history = _converge(
            geometry, "ewald", alpha, tolerance, evaluate, quantities
        )
    result = _result(
        geometry,
        outcome,
        stats,
        level,
        history,
        support,
        kind=f"independent-{level.method}-{geometry.site_kind}-polarization",
        convention=(
            "stationary induced-dipole functional 1/2 sum|u|^2/alpha + 1/2 B(u,u) + "
            "B(u,M), no permanent-permanent energy; dense diag(1/alpha)+G solve; "
            "induced dipoles on the site Gaussians; "
            + _convention(geometry, level.method, background)
        ),
        self_convention=(
            "co-centered electrostatics excluded, local |u|^2/alpha cost retained; "
            + _self_convention(geometry, False).split("; ", 1)[1]
        ),
        background=background,
        tolerance=tolerance,
        constant=constant,
    )
    result.update(stationarity)
    result["polarizabilities"] = polarizability
    result["derivatives"] = (
        "analytic Hellmann-Feynman: d/dx 1/2 B(u, u + 2M) at the stationary u"
    )
    result["comparison_semantics"] = (
        "stationary minimiser of the declared functional; TCG or truncated solvers "
        "approximate it and their difference is an approximation gap"
    )
    return result


def polarization_reference(
    positions: Any,
    charges: Any,
    dipoles: Any,
    second_moments: Any,
    polarizabilities: Any,
    vectors: Any,
    /,
    *,
    widths: Any = None,
    background: bool = False,
    coulomb_constant: float = 1.0,
    tolerance: float = 1.0e-10,
    maximum_shells: int = 24,
    mode_chunk: int = 512,
    maximum_atoms: int = 64,
) -> dict[str, Any]:
    """Independent stationary induced-dipole polarization (finite or periodic).

    Builds the bounded dense field ``E_p`` and operator ``diag(1/alpha) + G``
    with host NumPy, solves, and reports induced dipoles, residual,
    eigenvalue/conditioning evidence, analytic forces and strain derivative.
    ``background`` only admits a charged periodic cell (no charge-charge term
    enters the functional).
    """
    return _polarization(
        positions,
        charges,
        dipoles,
        second_moments,
        polarizabilities,
        vectors,
        widths=widths,
        background=background,
        coulomb_constant=coulomb_constant,
        tolerance=tolerance,
        maximum_shells=maximum_shells,
        mode_chunk=mode_chunk,
        maximum_atoms=maximum_atoms,
    )
