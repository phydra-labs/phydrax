"""Certified fidelity bounds hold on every platform without libm or LAPACK.

Every angle, cosine and conditioner below is decided by the rigorous native
elementary kernels or by correctly rounded IEEE operations. mpmath at 100
digits is the independent oracle; the adversarial inputs sit one ulp around
the cardinal angles, where a libm result's last bit decides a bound.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import Any

import mpmath as mp
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.geometry._domain_boundary import exact_plane_chart
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._rational import fraction_down, fraction_up
from phydrax.geometry._source_fidelity import _acos_lower, _acos_upper, _asin_upper
from phydrax.geometry.brep._constructors import brep_cylinder
from phydrax.geometry.implicit import _analytic_profile
from phydrax.geometry.implicit._source_charts import (
    _normal_angle_bounds,
    _ordered_contraction,
    _tangent_conditioner,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(), reason="rigorous enclosures need meshcore"
    ),
]


def _mp(value: Fraction, /) -> mp.mpf:
    return mp.mpf(value.numerator) / value.denominator


def _fraction(value: mp.mpf, /) -> Fraction:
    mantissa, exponent = mp.mpf(value).man_exp
    return Fraction(int(mantissa)) * Fraction(2) ** int(exponent)


def _cardinal_cosines() -> tuple[Fraction, ...]:
    """Exact cosines and sines of binary64 angles one ulp around 0, pi/2 and pi."""
    values = {Fraction(0), Fraction(1), Fraction(-1), Fraction(1, 2), Fraction(-1, 2)}
    with mp.workdps(100):
        for nearest in (0.0, 0.5 * np.pi, np.pi):
            for angle in (
                np.nextafter(nearest, -np.inf),
                nearest,
                np.nextafter(nearest, np.inf),
            ):
                for function in (mp.cos, mp.sin):
                    values.add(_fraction(function(mp.mpf(float(angle)))))
    for step in (1, 2, 3):
        below = 1.0
        for _ in range(step):
            below = float(np.nextafter(below, 0.0))
        values.update((Fraction(below), Fraction(-below)))
    values.update((Fraction(1, 2**60), Fraction(-1, 2**60), Fraction(1, 3)))
    return tuple(sorted(values))


@pytest.mark.parametrize("cosine", _cardinal_cosines())
def test_acos_bounds_contain_the_exact_angle(cosine: Fraction) -> None:
    with mp.workdps(100):
        exact = mp.acos(_mp(cosine))
        assert _mp(_acos_lower(cosine)) <= exact <= _mp(_acos_upper(cosine))
        # Tight: the angle range of the cosine's binary64 bracket plus a few
        # ulps of the kernel, not a bisection's resolution.
        bracket = mp.acos(max(mp.mpf(-1), mp.mpf(fraction_down(cosine)))) - mp.acos(
            min(mp.mpf(1), mp.mpf(fraction_up(cosine)))
        )
        width = _mp(_acos_upper(cosine) - _acos_lower(cosine))
        upper = _mp(_acos_upper(cosine))
        assert width <= bracket + 8 * mp.mpf(2) ** -52 * max(upper, mp.mpf(2) ** -1022)
        if cosine >= 0:
            sine = _fraction(mp.sqrt(1 - _mp(cosine) ** 2))
            assert mp.asin(_mp(sine)) <= _mp(_asin_upper(sine))


def test_acos_bounds_saturate_outside_the_unit_interval() -> None:
    with mp.workdps(100):
        assert _acos_lower(Fraction(2)) == 0 == _acos_upper(Fraction(1))
        assert _mp(_acos_lower(Fraction(-2))) <= mp.pi <= _mp(_acos_upper(Fraction(-2)))
        assert _mp(_asin_upper(Fraction(2))) >= mp.pi / 2
        assert _asin_upper(Fraction(-1)) == 0


@pytest.mark.parametrize("cosine", _cardinal_cosines())
def test_implicit_normal_angle_bounds_contain_the_exact_angle(
    cosine: Fraction,
) -> None:
    point = float(cosine)
    tangent_u = ((1.0, 1.0), (0.0, 0.0), (0.0, 0.0))
    tangent_v = ((0.0, 0.0), (1.0, 1.0), (0.0, 0.0))
    # Gradient (sqrt(1 - c^2), 0, c) makes the cosine with the z normal c.
    side = float(np.sqrt(max(0.0, 1.0 - point * point)))
    gradient = ((side, side), (0.0, 0.0), (point, point))
    lower, upper = _normal_angle_bounds(tangent_u, tangent_v, gradient, 1)
    with mp.workdps(100):
        norm = mp.sqrt(mp.mpf(side) ** 2 + mp.mpf(point) ** 2)
        exact = mp.acos(mp.mpf(point) / norm)
        assert mp.mpf(lower) <= exact <= mp.mpf(upper)


def test_plane_chart_points_are_rigorous_cosine_centers() -> None:
    domain = MeshingDomain.from_brep(
        brep_cylinder(0.5, 1.0, coordinate_contract=phx.SpatialCoordinateContract.si())
    )
    charts = [
        chart
        for chart in (
            exact_plane_chart(domain, patch) for patch in range(len(domain.patches))
        )
        if chart is not None
    ]
    assert charts
    chart = charts[0]
    profile = -chart.radius_offset + chart.radius_sign * Fraction(1, 2)
    radius = chart.radius_sign * (chart.radius_offset + profile)
    with mp.workdps(100):
        for angle in (0.0, 0.5 * np.pi, np.pi, 1.0, np.nextafter(2.0 * np.pi, 0.0)):
            point = chart.chart_point((Fraction(angle), profile))
            exact = (
                _mp(radius) * mp.cos(mp.mpf(angle)),
                -chart.radius_sign * _mp(radius) * mp.sin(mp.mpf(angle)),
            )
            for value, reference in zip(point, exact, strict=True):
                assert abs(_mp(value) - reference) <= 4 * mp.mpf(2) ** -53
        assert chart.chart_point((Fraction(0), profile)) == (radius, Fraction(0))


def _perturb_libm(monkeypatch: pytest.MonkeyPatch, /) -> None:
    """Every platform-libm elementary function rounds one ulp away.

    ``math.hypot`` is CPython's own vector norm (not the platform libm), so
    it is the same on every platform and is not perturbed.
    """
    for name in ("cos", "sin", "hypot", "arctan2", "arccos"):
        original = getattr(np, name)
        monkeypatch.setattr(
            np,
            name,
            lambda *args, _original=original, **kwargs: np.nextafter(
                _original(*args, **kwargs), np.inf
            ),
        )
    for name in ("cos", "sin", "acos", "atan2"):
        original = getattr(math, name)
        monkeypatch.setattr(
            math,
            name,
            lambda *args, _original=original: math.nextafter(_original(*args), math.inf),
        )


@pytest.mark.parametrize("family", ("sphere", "ring-torus"))
def test_analytic_profile_cover_ignores_platform_libm(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    geometry = (
        phx.geometry.Sphere((0.013, -0.021, 0.007), 0.4)
        if family == "sphere"
        else phx.geometry.Torus((0.012, -0.007, 0.017), 0.3, 0.9)
    ).compile()
    profile = _analytic_profile.AnalyticImplicitProfile(
        geometry, phx.SpatialCoordinateContract.si()
    )
    here = profile.boundary_cover(0.2)
    _perturb_libm(monkeypatch)
    there = profile.boundary_cover(0.2)
    for name in ("points", "covering_radius", "point_error"):
        assert np.array_equal(getattr(here, name), getattr(there, name)), name
    monkeypatch.undo()
    # mpmath oracle: each sample is within its certified error of the surface.
    center = np.asarray(profile.center, dtype=np.float64)
    with mp.workdps(60):
        for row in range(0, here.points.shape[0], 7):
            delta = [
                mp.mpf(float(here.points[row, axis])) - mp.mpf(float(center[axis]))
                for axis in range(3)
            ]
            if family == "sphere":
                distance = abs(
                    mp.sqrt(sum(value**2 for value in delta)) - mp.mpf(profile.radius)
                )
            else:
                ring = mp.sqrt(delta[0] ** 2 + delta[1] ** 2) - mp.mpf(
                    profile.major_radius
                )
                distance = abs(mp.sqrt(ring**2 + delta[2] ** 2) - mp.mpf(profile.radius))
            assert distance <= mp.mpf(float(here.point_error[row]))


def test_tangent_conditioner_decides_conditioning_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Constant:
        def __init__(self, value: float) -> None:
            self.center = {(0, 0): Fraction(value)} if value else {}

    def tangents(values: np.ndarray, /) -> Any:
        return tuple(tuple(_Constant(float(value)) for value in row) for row in values)

    def forbid(*args: object, **kwargs: object) -> object:
        raise AssertionError("LAPACK decided a conditioner.")

    for name in ("svd", "qr", "inv", "det"):
        monkeypatch.setattr(np.linalg, name, forbid)
    # Singular values exactly 8 and 1: not conditioned (strict threshold).
    boundary = np.asarray(((8.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    assert _tangent_conditioner(tangents(boundary)) is None
    needle = np.asarray(((1.0, 1.0), (0.0, 2.0**-10), (0.0, 0.0)))
    conditioner = _tangent_conditioner(tangents(needle))
    assert conditioner is not None
    monkeypatch.undo()
    # Oracle: R^-1 makes the centroid tangents orthonormal.
    with mp.workdps(60):
        matrix = mp.matrix([[mp.mpf(float(value)) for value in row] for row in needle])
        inverse = mp.matrix([[_mp(value) for value in row] for row in conditioner])
        product = matrix * inverse
        gram = product.T * product
        for row in range(2):
            for column in range(2):
                expected = 1 if row == column else 0
                assert abs(gram[row, column] - expected) <= mp.mpf(2) ** -40


def test_ordered_contraction_matches_exact_sum() -> None:
    generator = np.random.default_rng(7)
    coefficients = generator.standard_normal((5, 3, 10))
    values = generator.standard_normal((5, 10))
    result = _ordered_contraction(coefficients, values)
    for face in range(5):
        for axis in range(3):
            exact = sum(
                (
                    Fraction(float(coefficients[face, axis, index]))
                    * Fraction(float(values[face, index]))
                    for index in range(10)
                ),
                Fraction(0),
            )
            assert abs(Fraction(float(result[face, axis])) - exact) <= Fraction(1, 2**40)
