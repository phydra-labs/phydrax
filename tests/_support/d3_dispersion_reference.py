"""Independent host references for conventional D3(BJ) and FourierD3.

Pure NumPy/SciPy float64 implementations that share no code with the native
terms: brute-force image-route enumeration, Grimme's conventional counting
function, the FourierD3 modified counting function written from the
published formula plus the inspected source regularizer, the Gaussian C6
interpolation as an explicit double loop, the BJ pair energy, the S5 switch,
direct lattice sums over a smooth near/far partition with a continuum far field
and the radial Fourier transform by adaptive oscillatory quadrature. Lengths are
bohr and energies hartree; the conversion constants below are CODATA 2018 values
written independently of ``phydrax.units``. Synthetic tables here are invented
fixtures: they qualify no element and are not Grimme's reference data.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import TypedDict

import numpy as np
from scipy import integrate, special


BOHR_ANGSTROM = 0.529177210903
HARTREE_ELECTRONVOLT = 4.3597447222071e-18 / 1.602176634e-19
# Near-field lattice sums stop where the partition weight is erfc(6.5)/2 < 2e-20.
PARTITION_REACH = 6.5


class SyntheticD3Tables(TypedDict):
    """Host reference tables in the layout of ``D3ReferenceTables`` arguments."""

    species: tuple[int, ...]
    covalent_radii: np.ndarray
    r4r2: np.ndarray
    c6_reference: np.ndarray
    reference_coordination: np.ndarray
    reference_valid: np.ndarray


def synthetic_tables() -> SyntheticD3Tables:
    """Two invented species with three reference slots (one unpopulated for Z=1)."""
    species = (1, 6)
    valid = np.asarray([[True, True, False], [True, True, True]])
    coordination = np.asarray([[0.0, 0.9, 0.0], [0.0, 1.1, 2.4]])
    c6 = np.zeros((2, 2, 3, 3))
    c6[0, 0, :2, :2] = [[3.1, 2.2], [2.2, 1.7]]
    c6[1, 1] = [[49.1, 43.2, 29.4], [43.2, 38.0, 25.1], [29.4, 25.1, 18.2]]
    c6[0, 1, :2, :] = [[12.1, 10.6, 7.9], [8.6, 7.6, 5.5]]
    c6[1, 0] = c6[0, 1].T
    return {
        "species": species,
        "covalent_radii": np.asarray([0.8, 1.9]),
        "r4r2": np.asarray([2.0, 3.1]),
        "c6_reference": c6,
        "reference_coordination": coordination,
        "reference_valid": valid,
    }


def single_species_block() -> np.ndarray:
    """A positive symmetric 4x4 reference block with a signed, nonmonotone rank error.

    Ordered by magnitude its eigenvalues are about (62.7, 12.3, -11.3, -1.68)
    and the maximum relative entry errors at ranks 1..4 about
    (0.484, 0.703, 0.0766, 0).
    """
    return np.asarray(
        [
            [27.0, 11.0, 14.0, 15.0],
            [11.0, 9.0, 21.0, 17.0],
            [14.0, 21.0, 11.0, 16.0],
            [15.0, 17.0, 16.0, 15.0],
        ]
    )


def rank_error_curve(block: np.ndarray) -> list[float]:
    """Maximum relative entry error of the magnitude-ordered signed truncations."""
    values, vectors = np.linalg.eigh(block)
    order = np.argsort(-np.abs(values), kind="stable")
    values, vectors = values[order], vectors[:, order]
    errors = []
    for rank in range(1, values.size + 1):
        approx = (vectors[:, :rank] * values[:rank]) @ vectors[:, :rank].T
        errors.append(float(np.max(np.abs(approx - block) / np.abs(block))))
    return errors


def image_routes(
    positions: np.ndarray,
    vectors: np.ndarray | None,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Every directed route ``(receiver t, source s, n)`` with ``|x_t - x_s + n H| < radius``.

    Repeated and nonzero self images are included; only ``t == s, n == 0`` is
    excluded. ``vectors is None`` is the finite (nonperiodic) case.
    """
    points = np.asarray(positions, dtype=np.float64)
    count = points.shape[0]
    if vectors is None:
        shifts = np.zeros((1, 3), dtype=np.int64)
        lattice = np.zeros((3, 3))
    else:
        lattice = np.asarray(vectors, dtype=np.float64)
        inverse = np.linalg.inv(lattice)
        span = np.ptp(points @ inverse, axis=0) if count > 1 else np.zeros(3)
        extents = [
            int(math.ceil(radius * np.linalg.norm(inverse[:, axis]) + span[axis])) + 1
            for axis in range(3)
        ]
        shifts = np.asarray(
            list(itertools.product(*(range(-e, e + 1) for e in extents))), dtype=np.int64
        )
    receivers, sources, images = [], [], []
    for t in range(count):
        for s in range(count):
            displacement = points[t] - points[s] + shifts @ lattice
            inside = np.sum(displacement**2, axis=1) < radius * radius
            if t == s:
                inside &= np.any(shifts != 0, axis=1)
            for shift in shifts[inside]:
                receivers.append(t)
                sources.append(s)
                images.append(shift)
    return (
        np.asarray(receivers, dtype=np.int32),
        np.asarray(sources, dtype=np.int32),
        np.asarray(images, dtype=np.int64).reshape(-1, 3),
    )


def canonical_half(
    receivers: np.ndarray, sources: np.ndarray, shifts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One route of every reversal pair ``(t, s, n) <-> (s, t, -n)``."""
    keep = []
    for index, (t, s, shift) in enumerate(zip(receivers, sources, shifts)):
        if t > s:
            keep.append(index)
        elif t == s:
            first = shift[np.flatnonzero(shift)[0]]
            if first > 0:
                keep.append(index)
    keep = np.asarray(keep, dtype=np.int64)
    return receivers[keep], sources[keep], shifts[keep]


def conventional_count(
    distance: float, covalent: float, steepness: float = 16.0
) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (covalent / distance - 1.0)))


def modified_count(
    distance: float,
    covalent: float,
    cutoff: float,
    epsilon: float,
    *,
    folded_scale: float = 4.0 / 3.0,
    steepness: float = 16.0,
) -> float:
    """FourierD3 Eq. (18) with the source regularizer ``epsilon`` (area units)."""
    if not 0.0 < distance < cutoff:
        return 0.0
    middle = 0.5 * (covalent / folded_scale + cutoff)
    steep = steepness
    if distance > middle:
        steep += (distance - middle) ** 2 / ((cutoff - distance) ** 2 + epsilon)
    argument = steep * (covalent / distance - 1.0)
    if argument < -700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-argument))


def coordination_numbers(
    positions_bohr: np.ndarray,
    vectors_bohr: np.ndarray | None,
    rows: np.ndarray,
    radii: np.ndarray,
    cutoff_bohr: float,
    count: Callable[[float, float], float],
) -> np.ndarray:
    receivers, sources, shifts = image_routes(positions_bohr, vectors_bohr, cutoff_bohr)
    lattice = np.zeros((3, 3)) if vectors_bohr is None else vectors_bohr
    result = np.zeros(positions_bohr.shape[0])
    for t, s, shift in zip(receivers, sources, shifts):
        distance = float(
            np.linalg.norm(positions_bohr[t] - positions_bohr[s] + shift @ lattice)
        )
        result[t] += count(distance, float(radii[rows[t]] + radii[rows[s]]))
    return result


def interpolated_c6(
    tables: SyntheticD3Tables,
    a: int,
    b: int,
    cn_a: float,
    cn_b: float,
    *,
    threshold: float | None = None,
) -> float:
    """Grimme's normalized Gaussian C6 as an explicit double loop."""
    exponents, values = [], []
    for p in range(tables["reference_valid"].shape[1]):
        for q in range(tables["reference_valid"].shape[1]):
            if not (tables["reference_valid"][a, p] and tables["reference_valid"][b, q]):
                continue
            exponents.append(
                -4.0
                * (
                    (cn_a - tables["reference_coordination"][a, p]) ** 2
                    + (cn_b - tables["reference_coordination"][b, q]) ** 2
                )
            )
            values.append(tables["c6_reference"][a, b, p, q])
    exponents = np.asarray(exponents)
    values = np.asarray(values)
    if threshold is not None:
        keep = exponents - exponents.max() >= -threshold
        exponents, values = exponents[keep], values[keep]
    weights = np.exp(exponents)
    return float(np.sum(weights * values) / np.sum(weights))


def s5_switch(distance: float, on: float, off: float) -> float:
    if distance <= on:
        return 1.0
    if distance >= off:
        return 0.0
    t = (distance - on) / (off - on)
    return 1.0 - (10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5)


def bj_pair_potential(
    distance: float, r4r2_a: float, r4r2_b: float, damping: dict
) -> float:
    """``s6 / (r^6 + R0^6) + 3 s8 q / (r^8 + R0^8)`` (multiplies ``C6``)."""
    product = 3.0 * r4r2_a * r4r2_b
    radius = damping["a1"] * math.sqrt(product) + damping["a2"]
    return damping["s6"] / (distance**6 + radius**6) + damping["s8"] * product / (
        distance**8 + radius**8
    )


def d3_energy(
    positions_angstrom: np.ndarray,
    vectors_angstrom: np.ndarray | None,
    rows: np.ndarray,
    tables: SyntheticD3Tables,
    damping: dict,
    *,
    coordination_cutoff: float,
    dispersion_cutoff: float,
    switch: tuple[float, float] | None = None,
    threshold: float | None = None,
) -> float:
    """Conventional two-body D3(BJ) energy in electronvolt (inputs in angstrom)."""
    x = np.asarray(positions_angstrom, dtype=np.float64) / BOHR_ANGSTROM
    h = None if vectors_angstrom is None else np.asarray(vectors_angstrom) / BOHR_ANGSTROM
    cn = coordination_numbers(
        x,
        h,
        rows,
        tables["covalent_radii"],
        coordination_cutoff / BOHR_ANGSTROM,
        lambda r, covalent: conventional_count(r, covalent),
    )
    receivers, sources, shifts = image_routes(x, h, dispersion_cutoff / BOHR_ANGSTROM)
    lattice = np.zeros((3, 3)) if h is None else h
    energy = 0.0
    for t, s, shift in zip(receivers, sources, shifts):
        distance = float(np.linalg.norm(x[t] - x[s] + shift @ lattice))
        a, b = rows[t], rows[s]
        c6 = interpolated_c6(tables, a, b, cn[t], cn[s], threshold=threshold)
        value = (
            -0.5
            * c6
            * bj_pair_potential(distance, tables["r4r2"][a], tables["r4r2"][b], damping)
        )
        if switch is not None:
            value *= s5_switch(distance * BOHR_ANGSTROM, switch[0], switch[1])
        energy += value
    return energy * HARTREE_ELECTRONVOLT


def fourier_d3_direct_energy(
    positions_angstrom: np.ndarray,
    vectors_angstrom: np.ndarray,
    rows: np.ndarray,
    tables: SyntheticD3Tables,
    damping: dict,
    *,
    coordination_cutoff: float,
    epsilon_bohr2: float,
    shells: int | None = None,
    radius: float | None = None,
    width: float | None = None,
) -> float:
    """FourierD3 lattice energy in electronvolt by direct summation.

    Coordination numbers use the modified counting function; ``C6_ij`` is the
    exact (untruncated) normalized-Gaussian interpolation. ``shells`` sums the
    fixed parallelepiped ``|n_i| <= shells`` without a tail (smooth in the
    positions, for finite differences in atom positions; it lacks the volume
    response of the ``r^-6`` tail, so cell-strain references use the partition).

    ``radius`` ``R`` and ``width`` ``L`` (angstrom) instead split every pair
    potential with ``w(r) = erfc((r - R) / L) / 2`` into a near part ``w phi``,
    summed over the lattice out to ``R + 6.5 L``, and a far part
    ``(1 - w) phi`` replaced by its continuum integral
    ``(4 pi / V) int_0^inf r^2 (1 - w) phi dr`` (the ``G = 0`` term of its
    Poisson sum; the excluded self image ``(1 - w(0)) phi(0)`` is subtracted).
    The far part varies on the scale ``L``, so the neglected ``G != 0`` terms
    are of order ``(4 pi R phi(R) / (V G^2)) exp(-G^2 L^2 / 4)`` at the shortest
    nonzero reciprocal vector, plus the damping poles of ``phi`` near
    ``|r| ~ R0`` weighted by ``erfc((R - R0) / L)``. For the 5 angstrom test
    cells (``G >= 1.22 / angstrom``) ``R = 50, L = 8`` and ``R = 40, L = 7``
    therefore agree to roundoff. A sharp sphere with a continuum tail instead
    carries the oscillating lattice-point discrepancy of the sphere, which
    decays only algebraically (relative changes near ``1e-7`` between
    ``R = 60`` and ``80`` angstrom).
    """
    x = np.asarray(positions_angstrom, dtype=np.float64) / BOHR_ANGSTROM
    h = np.asarray(vectors_angstrom, dtype=np.float64) / BOHR_ANGSTROM
    cutoff = coordination_cutoff / BOHR_ANGSTROM
    cn = coordination_numbers(
        x,
        h,
        rows,
        tables["covalent_radii"],
        cutoff,
        lambda r, covalent: modified_count(r, covalent, cutoff, epsilon_bohr2),
    )
    count = x.shape[0]
    c6 = np.asarray(
        [
            [
                interpolated_c6(tables, rows[i], rows[j], cn[i], cn[j])
                for j in range(count)
            ]
            for i in range(count)
        ]
    )
    if shells is not None:
        grid = np.asarray(
            list(itertools.product(range(-shells, shells + 1), repeat=3)),
            dtype=np.float64,
        )
    else:
        if radius is None or width is None:
            raise ValueError("Direct lattice sums need shells or a radius and width.")
        if not 0.0 < width < radius:
            raise ValueError("The partition needs 0 < width < radius.")
        center = radius / BOHR_ANGSTROM
        spread = width / BOHR_ANGSTROM
        bound = center + PARTITION_REACH * spread
        inverse = np.linalg.inv(h)
        extents = [
            int(math.ceil(bound * np.linalg.norm(inverse[:, i]))) + 2 for i in range(3)
        ]
        grid = np.asarray(
            list(itertools.product(*(range(-e, e + 1) for e in extents))),
            dtype=np.float64,
        )
        volume = abs(np.linalg.det(h))
    translations = grid @ h
    energy = 0.0
    for i in range(count):
        for j in range(count):
            r4r2_a, r4r2_b = tables["r4r2"][rows[i]], tables["r4r2"][rows[j]]
            separation = x[i] - x[j] + translations
            distance = np.sqrt(np.sum(separation**2, axis=1))
            distance = distance[distance > 1.0e-12]
            if shells is not None:
                pair = float(np.sum(_bj_values(distance, r4r2_a, r4r2_b, damping)))
            else:
                pair = _partitioned_pair(
                    distance, r4r2_a, r4r2_b, damping, center, spread, volume, i == j
                )
            energy -= 0.5 * c6[i, j] * pair
    return energy * HARTREE_ELECTRONVOLT


def _bj_values(
    distance: np.ndarray, r4r2_a: float, r4r2_b: float, damping: dict
) -> np.ndarray:
    """:func:`bj_pair_potential` over an array of distances."""
    product = 3.0 * r4r2_a * r4r2_b
    radius = damping["a1"] * math.sqrt(product) + damping["a2"]
    return damping["s6"] / (distance**6 + radius**6) + damping["s8"] * product / (
        distance**8 + radius**8
    )


def _partitioned_pair(
    distance: np.ndarray,
    r4r2_a: float,
    r4r2_b: float,
    damping: dict,
    center: float,
    spread: float,
    volume: float,
    self_pair: bool,
) -> float:
    """Near lattice sum of ``w phi`` plus the continuum integral of ``(1 - w) phi`` (bohr)."""

    def far_density(r: float) -> float:
        weight = 0.5 * math.erfc((center - r) / spread)
        return r * r * weight * bj_pair_potential(r, r4r2_a, r4r2_b, damping)

    bound = center + PARTITION_REACH * spread
    r = distance[distance < bound]
    weights = 0.5 * special.erfc((r - center) / spread)
    near = float(np.sum(weights * _bj_values(r, r4r2_a, r4r2_b, damping)))
    edges = (0.0, max(center - PARTITION_REACH * spread, 0.0), center, bound, np.inf)
    far = sum(
        integrate.quad(far_density, lower, upper, epsabs=0.0, epsrel=1.0e-13, limit=200)[
            0
        ]
        for lower, upper in zip(edges[:-1], edges[1:])
    )
    far *= 4.0 * math.pi / volume
    if self_pair:
        weight = 0.5 * math.erfc(center / spread)
        far -= weight * bj_pair_potential(0.0, r4r2_a, r4r2_b, damping)
    return near + far


def radial_fourier_transform(
    potential: Callable[[float], float], wavenumber: float
) -> float:
    """``int phi(|r|) exp(-i k.r) d^3r`` by oscillatory quadrature (``k = 0`` allowed)."""
    if wavenumber == 0.0:
        value, _ = integrate.quad(
            lambda r: 4.0 * math.pi * r * r * potential(r),
            0.0,
            np.inf,
            epsabs=0.0,
            epsrel=1.0e-13,
            limit=400,
        )
        return value
    value, _ = integrate.quad(
        lambda r: r * potential(r),
        0.0,
        np.inf,
        weight="sin",
        wvar=wavenumber,
        epsabs=1.0e-15,
        limlst=200,
    )
    return 4.0 * math.pi * value / wavenumber


def central_difference(
    function: Callable[[np.ndarray], float], point: np.ndarray, step: float
) -> np.ndarray:
    base = np.asarray(point, dtype=np.float64)
    gradient = np.zeros_like(base)
    for index in np.ndindex(base.shape):
        forward = base.copy()
        backward = base.copy()
        forward[index] += step
        backward[index] -= step
        gradient[index] = (function(forward) - function(backward)) / (2.0 * step)
    return gradient


def strained(points: np.ndarray, strain: np.ndarray) -> np.ndarray:
    """Rows mapped by the column deformation ``F = I + strain``: ``x @ F.T``."""
    return np.asarray(points) @ (np.eye(3) + strain).T
