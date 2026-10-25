#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Analytic and physics controls of the independent multipole/polarization reference.

Each control is derived separately from the reference implementation (closed
forms, lattice constants, charge-cluster limits, a spherical Fourier integral,
an alternative Fourier route, split-parameter invariance or finite
differences) so that plausible sign, normalization, self-term and background
mistakes fail.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from scipy import integrate, special

from tests._support.atomistic_long_range_reference import (
    converged_ewald_energy,
    NACL_MADELUNG,
)
from tests._support.atomistic_multipole_reference import (
    _multipole,
    _polarization,
    gaussian_fourier_multipole_reference,
    multipole_reference,
    polarization_reference,
    ReferenceResourceError,
    ReferenceUnsupportedError,
)


#: Simple-cubic Wigner (uniform-background) Madelung constant per lattice constant.
SIMPLE_CUBIC_WIGNER = 2.837297479480620

_TRICLINIC = np.asarray([[4.0, 0.0, 0.0], [0.7, 4.3, 0.0], [-0.5, 0.4, 4.6]])
_FRACTIONAL = np.asarray([[0.1, 0.2, 0.3], [0.55, 0.45, 0.7], [0.8, 0.9, 0.15]])
_FINITE = np.asarray([[0.0, 0.0, 0.0], [2.3, 0.4, -0.3], [-0.5, 2.1, 0.8]])
_WIDTHS = np.asarray([0.5, 0.7, 0.6])


def _moments(seed: int, count: int, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    charges = rng.uniform(-1.0, 1.0, count)
    charges -= np.mean(charges)
    dipoles = rng.uniform(-0.3, 0.3, (count, 3))
    raw = rng.uniform(-0.2, 0.2, (count, 3, 3))
    return charges, dipoles, raw + np.swapaxes(raw, 1, 2)


def _derivative(energy: Any, step: float, /) -> float:
    """Fourth-order central difference of ``energy(t)`` at ``t = 0``."""
    return (
        -energy(2.0 * step)
        + 8.0 * energy(step)
        - 8.0 * energy(-step)
        + energy(-2.0 * step)
    ) / (12.0 * step)


def _difference_forces(
    energy: Any, positions: np.ndarray, cell: np.ndarray | None, step: float, /
) -> np.ndarray:
    forces = np.zeros_like(positions)
    for atom in range(positions.shape[0]):
        for axis in range(3):

            def moved(t: float, atom: int = atom, axis: int = axis) -> float:
                shifted = positions.copy()
                shifted[atom, axis] += t
                return energy(shifted, cell)

            forces[atom, axis] = -_derivative(moved, step)
    return forces


def _difference_strain(
    energy: Any, positions: np.ndarray, cell: np.ndarray | None, step: float, /
) -> np.ndarray:
    """Full column-strain derivative of ``E(x (I+e)^T, H (I+e)^T)``."""
    derivative = np.zeros((3, 3))
    for first in range(3):
        for second in range(3):

            def strained(t: float, first: int = first, second: int = second) -> float:
                deformation = np.eye(3)
                deformation[first, second] += t
                return energy(
                    positions @ deformation.T,
                    None if cell is None else cell @ deformation.T,
                )

            derivative[first, second] = _derivative(strained, step)
    return derivative


def _split(alpha_factor: float, /, *args: Any, **keywords: Any) -> dict[str, Any]:
    options = {
        "widths": None,
        "include_coincident_self": False,
        "background": False,
        "coulomb_constant": 1.0,
        "tolerance": 1.0e-11,
        "maximum_shells": 24,
        "mode_chunk": 512,
        "maximum_atoms": 64,
        "method": "ewald",
    }
    options.update(keywords)
    return _multipole(*args, alpha_factor=alpha_factor, **options)


# --------------------------------------------------------- lattice controls


def test_point_charge_lattice_reduces_to_scalar_ewald_oracle() -> None:
    positions = _FRACTIONAL @ _TRICLINIC
    charges = np.asarray([1.0, -0.4, -0.6])
    reference = multipole_reference(
        positions, charges, np.zeros((3, 3)), np.zeros((3, 3, 3)), _TRICLINIC
    )
    oracle, _ = converged_ewald_energy(
        positions, charges, _TRICLINIC, 0.875, tolerance=1.0e-12
    )
    assert reference["converged"] and reference["independent"]
    assert reference["energy"] == pytest.approx(oracle, rel=1.0e-9, abs=1.0e-11)


def test_rock_salt_madelung_energy_forces_and_strain() -> None:
    sodium = np.asarray([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64)
    chloride = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=np.float64)
    positions = np.concatenate([sodium, chloride])
    charges = np.repeat([1.0, -1.0], 4)
    reference = multipole_reference(
        positions, charges, np.zeros((8, 3)), np.zeros((8, 3, 3)), 2.0 * np.eye(3)
    )
    energy = reference["energy"]
    assert energy == pytest.approx(-4.0 * NACL_MADELUNG, rel=1.0e-10)
    np.testing.assert_allclose(reference["forces"], 0.0, atol=1.0e-10)
    np.testing.assert_allclose(
        reference["strain_derivative"], -energy / 3.0 * np.eye(3), atol=1.0e-9
    )
    np.testing.assert_allclose(
        reference["stress"], reference["strain_derivative"] / 8.0, rtol=1.0e-14
    )


def test_uniform_background_simple_cubic_wigner_constant() -> None:
    side = 1.5
    reference = multipole_reference(
        np.zeros((1, 3)),
        np.asarray([1.0]),
        np.zeros((1, 3)),
        np.zeros((1, 3, 3)),
        side * np.eye(3),
        background=True,
    )
    energy = reference["energy"]
    assert reference["neutrality"] == "uniform-background"
    assert energy == pytest.approx(-SIMPLE_CUBIC_WIGNER / (2.0 * side), rel=1.0e-10)
    np.testing.assert_allclose(reference["forces"], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(
        reference["strain_derivative"], -energy / 3.0 * np.eye(3), atol=1.0e-10
    )


@pytest.mark.parametrize("route", ["point", "gaussian-ewald", "gaussian-fourier"])
def test_tinfoil_cubic_dipole_lattice_lorentz_energy(route: str) -> None:
    """Tin-foil cubic dipole lattice: ``E = -2 pi |mu|^2 / (3 V)`` (Lorentz sum 0)."""
    side = 2.0
    dipole = np.asarray([[0.3, -0.2, 0.5]])
    arguments = (
        np.asarray([[0.3, 0.7, 1.1]]),
        np.zeros(1),
        dipole,
        np.zeros((1, 3, 3)),
        side * np.eye(3),
    )
    if route == "point":
        reference = multipole_reference(*arguments)
    elif route == "gaussian-ewald":
        reference = multipole_reference(*arguments, widths=np.asarray([0.15]))
    else:
        reference = gaussian_fourier_multipole_reference(
            *arguments, widths=np.asarray([0.15])
        )
    expected = -2.0 * math.pi * float(dipole[0] @ dipole[0]) / (3.0 * side**3)
    assert reference["energy"] == pytest.approx(expected, rel=1.0e-9)
    np.testing.assert_allclose(reference["forces"], 0.0, atol=1.0e-12)


@pytest.mark.parametrize(
    ("widths", "include"),
    [(None, False), (_WIDTHS, False), (_WIDTHS, True)],
    ids=["point", "gaussian-exclude", "gaussian-include"],
)
def test_ewald_split_is_alpha_independent_through_degree_two(
    widths: np.ndarray | None, include: bool
) -> None:
    charges, dipoles, seconds = _moments(3, 3)
    arguments = (_FRACTIONAL @ _TRICLINIC, charges, dipoles, seconds, _TRICLINIC)
    narrow = _split(2.0, *arguments, widths=widths, include_coincident_self=include)
    wide = _split(4.5, *arguments, widths=widths, include_coincident_self=include)
    assert narrow["support"]["alpha"] != wide["support"]["alpha"]
    assert narrow["energy"] == pytest.approx(wide["energy"], rel=1.0e-9, abs=1.0e-10)
    np.testing.assert_allclose(narrow["forces"], wide["forces"], atol=1.0e-9)
    np.testing.assert_allclose(
        narrow["strain_derivative"], wide["strain_derivative"], atol=1.0e-9
    )


@pytest.mark.parametrize(
    ("include", "charged"),
    [(False, False), (True, False), (False, True)],
    ids=["exclude", "include", "charged-background"],
)
def test_gaussian_ewald_split_matches_absolutely_convergent_fourier_series(
    include: bool, charged: bool
) -> None:
    charges, dipoles, seconds = _moments(5, 3)
    if charged:
        charges = charges + 0.3
    arguments = (_FRACTIONAL @ _TRICLINIC, charges, dipoles, seconds, _TRICLINIC)
    split = multipole_reference(
        *arguments,
        widths=_WIDTHS,
        include_coincident_self=include,
        background=charged,
        tolerance=1.0e-11,
    )
    fourier = gaussian_fourier_multipole_reference(
        *arguments,
        widths=_WIDTHS,
        include_coincident_self=include,
        background=charged,
        tolerance=1.0e-11,
    )
    assert split["support"]["method"] == "ewald"
    assert fourier["support"]["method"] == "fourier"
    assert split["energy"] == pytest.approx(fourier["energy"], rel=1.0e-9, abs=1.0e-10)
    np.testing.assert_allclose(split["forces"], fourier["forces"], atol=1.0e-9)
    np.testing.assert_allclose(
        split["strain_derivative"], fourier["strain_derivative"], atol=1.0e-9
    )


# ---------------------------------------------------------- finite controls


def _cluster(
    center: np.ndarray,
    charge: float,
    dipole: np.ndarray,
    second: np.ndarray,
    size: float,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Charges with moments ``(q, mu, Theta)``; odd/even symmetric parts."""
    charges = [charge]
    points = [center]
    for axis in range(3):
        offset = 0.5 * size * np.eye(3)[axis]
        charges += [dipole[axis] / size, -dipole[axis] / size]
        points += [center + offset, center - offset]
    values, vectors = np.linalg.eigh(second)
    for value, vector in zip(values, vectors.T):
        weight = value / (2.0 * size * size)
        charges += [weight, weight, -2.0 * weight]
        points += [center + size * vector, center - size * vector, center]
    return np.asarray(charges), np.asarray(points)


def _cluster_energy(
    positions: np.ndarray,
    charges: np.ndarray,
    dipoles: np.ndarray,
    seconds: np.ndarray,
    widths: np.ndarray | None,
    size: float,
    /,
) -> float:
    clusters = [
        _cluster(positions[i], charges[i], dipoles[i], seconds[i], size)
        for i in range(positions.shape[0])
    ]
    energy = 0.0
    for first in range(len(clusters)):
        for second in range(first + 1, len(clusters)):
            qa, pa = clusters[first]
            qb, pb = clusters[second]
            distance = np.linalg.norm(pa[:, None, :] - pb[None, :, :], axis=-1)
            if widths is None:
                kernel = 1.0 / distance
            else:
                pair = math.sqrt(2.0 * (widths[first] ** 2 + widths[second] ** 2))
                kernel = special.erf(distance / pair) / distance
            energy += float(np.sum(qa[:, None] * qb[None, :] * kernel))
    return energy


@pytest.mark.parametrize("widths", [None, _WIDTHS], ids=["point", "gaussian"])
def test_finite_moments_match_charge_cluster_limits(widths: np.ndarray | None) -> None:
    """Moments of small charge clusters: Richardson limit in the even ``d^2`` series."""
    charges, dipoles, seconds = _moments(7, 3)
    charges = charges + np.asarray([0.2, -0.5, 0.1])
    size = 0.08
    values = [
        _cluster_energy(_FINITE, charges, dipoles, seconds, widths, size / 2**level)
        for level in range(3)
    ]
    first = [(4.0 * values[i + 1] - values[i]) / 3.0 for i in range(2)]
    limit = (16.0 * first[1] - first[0]) / 15.0
    reference = multipole_reference(
        _FINITE, charges, dipoles, seconds, None, widths=widths
    )
    assert reference["energy"] == pytest.approx(limit, rel=1.0e-7, abs=1.0e-9)


def test_gaussian_coincident_self_energy_matches_spherical_fourier_integral() -> None:
    sigma = 0.45
    charge = 0.8
    dipole = np.asarray([0.3, -0.4, 0.2])
    raw = np.asarray([[0.3, 0.1, -0.2], [0.1, -0.1, 0.05], [-0.2, 0.05, 0.4]])
    constant = 1.3
    trace = float(np.trace(raw))
    square = float(np.sum(raw * raw))

    def integrand(k: float) -> float:
        average = (
            charge**2
            - charge * k * k * trace / 3.0
            + k * k * float(dipole @ dipole) / 3.0
            + k**4 * (trace**2 + 2.0 * square) / 60.0
        )
        return math.exp(-((sigma * k) ** 2)) * average

    integral, _ = integrate.quad(integrand, 0.0, math.inf, epsabs=0.0, epsrel=1.0e-13)
    arguments = (
        np.asarray([[0.2, -0.1, 0.4]]),
        np.asarray([charge]),
        dipole[None, :],
        raw[None, :, :],
        None,
    )
    included = multipole_reference(
        *arguments,
        widths=np.asarray([sigma]),
        include_coincident_self=True,
        coulomb_constant=constant,
    )
    excluded = multipole_reference(
        *arguments, widths=np.asarray([sigma]), coulomb_constant=constant
    )
    assert included["energy"] == pytest.approx(constant / math.pi * integral, rel=1.0e-11)
    assert excluded["energy"] == 0.0
    np.testing.assert_array_equal(excluded["forces"], 0.0)


@pytest.mark.parametrize(
    ("widths", "include"),
    [(None, False), (_WIDTHS, True)],
    ids=["point", "gaussian-include"],
)
def test_finite_forces_and_strain_match_central_differences(
    widths: np.ndarray | None, include: bool
) -> None:
    charges, dipoles, seconds = _moments(11, 3)

    def energy(x: np.ndarray, cell: np.ndarray | None) -> float:
        return multipole_reference(
            x,
            charges,
            dipoles,
            seconds,
            cell,
            widths=widths,
            include_coincident_self=include,
        )["energy"]

    reference = multipole_reference(
        _FINITE,
        charges,
        dipoles,
        seconds,
        None,
        widths=widths,
        include_coincident_self=include,
    )
    assert reference["stress"] is None
    np.testing.assert_allclose(
        reference["forces"],
        _difference_forces(energy, _FINITE, None, 1.0e-3),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        reference["strain_derivative"],
        _difference_strain(energy, _FINITE, None, 1.0e-3),
        rtol=1.0e-8,
        atol=1.0e-10,
    )


@pytest.mark.parametrize("widths", [None, _WIDTHS[:2]], ids=["point", "gaussian"])
def test_periodic_forces_and_strain_match_converged_central_differences(
    widths: np.ndarray | None,
) -> None:
    charges, dipoles, seconds = _moments(13, 2)
    positions = _FRACTIONAL[:2] @ _TRICLINIC

    def energy(x: np.ndarray, cell: np.ndarray | None) -> float:
        return multipole_reference(
            x, charges, dipoles, seconds, cell, widths=widths, tolerance=1.0e-12
        )["energy"]

    reference = multipole_reference(
        positions, charges, dipoles, seconds, _TRICLINIC, widths=widths, tolerance=1.0e-12
    )
    np.testing.assert_allclose(
        reference["forces"],
        _difference_forces(energy, positions, _TRICLINIC, 1.0e-3),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        reference["strain_derivative"],
        _difference_strain(energy, positions, _TRICLINIC, 1.0e-3),
        rtol=1.0e-6,
        atol=1.0e-7,
    )


# ------------------------------------------------------ polarization controls


def _axial_kernel(
    distance: float, widths: tuple[float, float] | None, /
) -> tuple[float, float]:
    """Independent ``phi'(R)`` and ``phi''(R)`` of the point or Gaussian pair kernel."""
    if widths is None:
        return -1.0 / distance**2, 2.0 / distance**3
    beta = 1.0 / math.sqrt(2.0 * (widths[0] ** 2 + widths[1] ** 2))
    gauss = 2.0 * beta / math.sqrt(math.pi) * math.exp(-((beta * distance) ** 2))
    error = math.erf(beta * distance)
    first = gauss / distance - error / distance**2
    second = -gauss * (2.0 * beta**2 + 2.0 / distance**2) + 2.0 * error / distance**3
    return first, second


def _two_site_energy(
    distance: float,
    charges: tuple[float, float],
    alphas: tuple[float, float],
    widths: tuple[float, float] | None,
    constant: float,
    /,
) -> float:
    """Closed-form axial solve: sites at ``0`` and ``R z``; charges only."""
    first, second = _axial_kernel(distance, widths)
    field_one = constant * charges[1] * first  # along +z at site 1
    field_two = -constant * charges[0] * first
    coupling = constant * second
    determinant = 1.0 - coupling**2 * alphas[0] * alphas[1]
    u_one = alphas[0] * (field_one + coupling * alphas[1] * field_two) / determinant
    u_two = alphas[1] * (field_two + coupling * alphas[0] * field_one) / determinant
    return -0.5 * (u_one * field_one + u_two * field_two)


@pytest.mark.parametrize("widths", [None, (0.4, 0.6)], ids=["point", "gaussian"])
def test_two_site_polarization_closed_form(widths: tuple[float, float] | None) -> None:
    distance = 2.2
    charges = (0.9, -0.4)
    alphas = (0.35, 0.5)
    constant = 1.7
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    reference = polarization_reference(
        positions,
        np.asarray(charges),
        np.zeros((2, 3)),
        np.zeros((2, 3, 3)),
        np.asarray(alphas),
        None,
        widths=None if widths is None else np.asarray(widths),
        coulomb_constant=constant,
    )
    expected = _two_site_energy(distance, charges, alphas, widths, constant)
    assert reference["energy"] == pytest.approx(expected, rel=1.0e-12)
    np.testing.assert_allclose(reference["induced_dipoles"][:, :2], 0.0, atol=1.0e-14)
    axial = _derivative(
        lambda t: _two_site_energy(distance + t, charges, alphas, widths, constant),
        1.0e-3,
    )
    np.testing.assert_allclose(
        reference["forces"][:, 2], [axial, -axial], rtol=1.0e-8, atol=1.0e-12
    )


@pytest.mark.parametrize("widths", [None, np.asarray([0.15])], ids=["point", "gaussian"])
def test_periodic_polarizable_dipole_lattice_clausius_mossotti(
    widths: np.ndarray | None,
) -> None:
    """One site per cubic tin-foil cell: Lorentz field ``4 pi mu/(3V)``.

    The site's charge and second moment produce no field at their own lattice
    sites (inversion symmetry), so the induced dipole depends only on ``mu``.
    """
    side = 2.0
    volume = side**3
    alpha = 0.6
    dipole = np.asarray([0.2, -0.1, 0.35])
    second = np.asarray([[0.2, 0.05, 0.0], [0.05, -0.1, 0.1], [0.0, 0.1, 0.3]])
    reference = polarization_reference(
        np.asarray([[0.4, 0.1, 1.3]]),
        np.asarray([0.7]),
        dipole[None, :],
        second[None, :, :],
        np.asarray([alpha]),
        side * np.eye(3),
        widths=widths,
        background=True,
    )
    lorentz = 4.0 * math.pi / (3.0 * volume)
    induced = lorentz * dipole / (1.0 / alpha - lorentz)
    np.testing.assert_allclose(reference["induced_dipoles"][0], induced, rtol=1.0e-9)
    assert reference["energy"] == pytest.approx(
        -0.5 * lorentz * float(dipole @ induced), rel=1.0e-9
    )
    assert reference["minimum_eigenvalue"] == pytest.approx(
        1.0 / alpha - lorentz, rel=1.0e-9
    )


def test_periodic_polarization_is_alpha_independent() -> None:
    charges, dipoles, seconds = _moments(17, 3)
    arguments = (
        _FRACTIONAL @ _TRICLINIC,
        charges,
        dipoles,
        seconds,
        np.asarray([0.3, 0.45, 0.25]),
        _TRICLINIC,
    )
    narrow, wide = (
        _polarization(
            *arguments,
            widths=_WIDTHS,
            background=False,
            coulomb_constant=1.0,
            tolerance=1.0e-11,
            maximum_shells=24,
            mode_chunk=512,
            maximum_atoms=64,
            alpha_factor=factor,
        )
        for factor in (2.0, 4.5)
    )
    assert narrow["energy"] == pytest.approx(wide["energy"], rel=1.0e-9, abs=1.0e-11)
    np.testing.assert_allclose(
        narrow["induced_dipoles"], wide["induced_dipoles"], atol=1.0e-10
    )
    np.testing.assert_allclose(narrow["forces"], wide["forces"], atol=1.0e-9)
    np.testing.assert_allclose(
        narrow["strain_derivative"], wide["strain_derivative"], atol=1.0e-9
    )


@pytest.mark.parametrize(
    ("widths", "periodic"),
    [(None, False), (_WIDTHS, False), (_WIDTHS[:2], True)],
    ids=["finite-point", "finite-gaussian", "periodic-gaussian"],
)
def test_polarization_forces_and_strain_match_central_differences(
    widths: np.ndarray | None, periodic: bool
) -> None:
    count = 2 if periodic else 3
    charges, dipoles, seconds = _moments(19, count)
    alphas = np.asarray([0.3, 0.45, 0.25])[:count]
    positions = _FRACTIONAL[:count] @ _TRICLINIC if periodic else _FINITE
    cell = _TRICLINIC if periodic else None

    def energy(x: np.ndarray, vectors: np.ndarray | None) -> float:
        return polarization_reference(
            x,
            charges,
            dipoles,
            seconds,
            alphas,
            vectors,
            widths=widths,
            tolerance=1.0e-12,
        )["energy"]

    reference = polarization_reference(
        positions,
        charges,
        dipoles,
        seconds,
        alphas,
        cell,
        widths=widths,
        tolerance=1.0e-12,
    )
    rtol, atol = (1.0e-6, 1.0e-7) if periodic else (1.0e-8, 1.0e-10)
    np.testing.assert_allclose(
        reference["forces"],
        _difference_forces(energy, positions, cell, 1.0e-3),
        rtol=rtol,
        atol=atol,
    )
    np.testing.assert_allclose(
        reference["strain_derivative"],
        _difference_strain(energy, positions, cell, 1.0e-3),
        rtol=rtol,
        atol=atol,
    )


def test_polarization_reports_stationarity_and_conditioning() -> None:
    charges, dipoles, seconds = _moments(23, 3)
    reference = polarization_reference(
        _FINITE, charges, dipoles, seconds, np.asarray([0.3, 0.45, 0.25]), None
    )
    assert reference["converged"] and reference["independent"]
    assert reference["induced_dipoles"].shape == (3, 3)
    assert reference["permanent_field"].shape == (3, 3)
    assert reference["residual_relative"] < 1.0e-12
    assert 0.0 < reference["minimum_eigenvalue"] <= reference["maximum_eigenvalue"]
    assert reference["condition_number"] >= 1.0
    assert reference["functional_consistency"] < 1.0e-12 * max(
        1.0, abs(reference["energy"])
    )
    assert reference["energy"] == pytest.approx(
        reference["stationary_energy"], rel=1.0e-12
    )
    assert reference["energy"] < 0.0


def test_polarization_refuses_indefinite_finite_operator() -> None:
    with pytest.raises(ReferenceUnsupportedError):
        polarization_reference(
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),
            np.asarray([1.0, -1.0]),
            np.zeros((2, 3)),
            np.zeros((2, 3, 3)),
            np.asarray([1.0, 1.0]),
            None,
        )


def test_polarization_refuses_periodic_polarization_catastrophe() -> None:
    side = 1.0
    catastrophic = 1.1 * 3.0 * side**3 / (4.0 * math.pi)
    with pytest.raises(ReferenceUnsupportedError):
        polarization_reference(
            np.zeros((1, 3)),
            np.zeros(1),
            np.asarray([[0.0, 0.0, 0.1]]),
            np.zeros((1, 3, 3)),
            np.asarray([catastrophic]),
            side * np.eye(3),
        )


# ------------------------------------------------------- refusal contracts


def test_point_reference_refuses_coincident_self_energy() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    with pytest.raises(ReferenceUnsupportedError):
        multipole_reference(
            _FINITE, charges, dipoles, seconds, None, include_coincident_self=True
        )


def test_periodic_reference_refuses_undeclared_charge_background() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    with pytest.raises(ReferenceUnsupportedError):
        multipole_reference(
            _FRACTIONAL @ _TRICLINIC, charges + 0.1, dipoles, seconds, _TRICLINIC
        )


def test_finite_reference_refuses_periodic_background() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    with pytest.raises(ReferenceUnsupportedError):
        multipole_reference(_FINITE, charges, dipoles, seconds, None, background=True)


def test_reference_refuses_asymmetric_primitive_second_moment() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    asymmetric = seconds.copy()
    asymmetric[0, 0, 1] += 0.1
    with pytest.raises(ReferenceUnsupportedError):
        multipole_reference(_FINITE, charges, dipoles, asymmetric, None)


def test_gaussian_reference_refuses_nonpositive_width() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    with pytest.raises(ReferenceUnsupportedError):
        multipole_reference(
            _FINITE,
            charges,
            dipoles,
            seconds,
            None,
            widths=np.asarray([0.5, 0.0, 0.5], dtype=np.float64),
        )


def test_point_reference_refuses_distinct_coincident_sites() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    with pytest.raises(ReferenceUnsupportedError):
        multipole_reference(
            np.concatenate([_FINITE[:2], _FINITE[:1]]), charges, dipoles, seconds, None
        )


def test_gaussian_fourier_reference_refuses_point_sites() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    with pytest.raises(ReferenceUnsupportedError):
        gaussian_fourier_multipole_reference(
            _FRACTIONAL @ _TRICLINIC, charges, dipoles, seconds, _TRICLINIC, widths=None
        )


def test_polarization_reference_refuses_nonpositive_polarizability() -> None:
    charges, dipoles, seconds = _moments(29, 3)
    with pytest.raises(ReferenceUnsupportedError):
        polarization_reference(
            _FINITE,
            charges,
            dipoles,
            seconds,
            np.asarray([0.3, -0.1, 0.2], dtype=np.float64),
            None,
        )


def test_reference_refuses_exhausted_atom_budget() -> None:
    charges, dipoles, seconds = _moments(31, 3)
    with pytest.raises(ReferenceResourceError):
        multipole_reference(
            _FRACTIONAL @ _TRICLINIC,
            charges,
            dipoles,
            seconds,
            _TRICLINIC,
            maximum_atoms=2,
        )


def test_reference_refuses_exhausted_image_shell_budget() -> None:
    charges, dipoles, seconds = _moments(31, 3)
    with pytest.raises(ReferenceResourceError):
        multipole_reference(
            _FRACTIONAL @ _TRICLINIC,
            charges,
            dipoles,
            seconds,
            _TRICLINIC,
            maximum_shells=1,
        )


def test_reference_nonconvergence_preserves_joint_observable_history() -> None:
    charges, dipoles, seconds = _moments(31, 3)
    with pytest.raises(ReferenceResourceError) as unconverged:
        multipole_reference(
            _FRACTIONAL @ _TRICLINIC,
            charges,
            dipoles,
            seconds,
            _TRICLINIC,
            tolerance=1.0e-30,
        )
    history = unconverged.value.history
    assert len(history) >= 2
    assert all(record["within_tolerance"] is False for record in history)
    for key in ("energy_change", "forces_change", "strain_derivative_change"):
        assert all(record[key] is not None for record in history[1:])
