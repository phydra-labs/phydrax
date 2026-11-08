from fractions import Fraction
from pathlib import Path
from typing import Any, cast, Literal

import equinox as eqx
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._meshing_domain import (
    _PlacedSurfacePatch,
    MeshingDomain,
    SourcePreparation,
)
from phydrax.geometry.brep import brep_box
from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._patches import CylinderPatch, SpherePatch, TorusPatch


F = Fraction
CENTER = (F(1, 3), F(-2, 7), F(4, 11))
FIRST = (F(3, 5), F(4, 5), F(0))
SECOND = (F(-4, 5), F(3, 5), F(0))
AXIS = (F(0), F(0), F(1))
MAJOR = F(7, 3)
MINOR = F(2, 7)


def _torus() -> TorusPatch:
    return TorusPatch(
        tuple(map(float, CENTER)),
        tuple(map(float, FIRST)),
        tuple(map(float, SECOND)),
        tuple(map(float, AXIS)),
        float(MAJOR),
        float(MINOR),
        center_exact=CENTER,
        first_axis_exact=FIRST,
        second_axis_exact=SECOND,
        axis_exact=AXIS,
        major_radius_exact=MAJOR,
        minor_radius_exact=MINOR,
    )


@pytest.fixture
def preparation() -> SourcePreparation:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    return SourcePreparation(MeshingDomain.from_brep(model))


def _mp_fraction(value: Fraction) -> mp.mpf:
    return mp.mpf(value.numerator) / value.denominator


def _exact_values(u: float, v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Analytic derivatives of the authored coefficients, independent of JAX and
    # the interval implementation. Float parameters are converted exactly.
    u, v = mp.mpf(u), mp.mpf(v)
    cu, su, cv, sv = mp.cos(u), mp.sin(u), mp.cos(v), mp.sin(v)
    major, minor = _mp_fraction(MAJOR), _mp_fraction(MINOR)
    point = np.empty(3, dtype=object)
    jac = np.empty((3, 2), dtype=object)
    hess = np.empty((3, 2, 2), dtype=object)
    for i in range(3):
        center, first, second, axis = map(
            _mp_fraction, (CENTER[i], FIRST[i], SECOND[i], AXIS[i])
        )
        radial = cu * first + su * second
        tangent = -su * first + cu * second
        ring = major + minor * cv
        point[i] = center + ring * radial + minor * sv * axis
        jac[i, 0] = ring * tangent
        jac[i, 1] = -minor * sv * radial + minor * cv * axis
        hess[i, 0, 0] = -ring * radial
        hess[i, 0, 1] = hess[i, 1, 0] = -minor * sv * tangent
        hess[i, 1, 1] = -minor * cv * radial - minor * sv * axis
    return point, jac, hess


def _assert_enclosed(lower: np.ndarray, upper: np.ndarray, exact: np.ndarray) -> None:
    assert lower.shape == upper.shape == exact.shape
    for index in np.ndindex(exact.shape):
        assert mp.mpf(float(lower[index])) <= exact[index] <= mp.mpf(float(upper[index]))


@pytest.mark.parametrize(
    "box",
    [
        ((0.0, 0.0), (0.0, 0.0)),
        ((0.2, -0.7), (0.2, -0.7)),
        ((-0.3, -0.8), (0.9, 0.4)),
        ((1.4, 2.8), (2.1, 3.4)),
    ],
)
def test_source_program_encloses_exact_point_jacobian_and_hessian(
    preparation: SourcePreparation, box: tuple[tuple[float, float], ...]
) -> None:
    patch = _torus()
    low, high = np.asarray(box, dtype=np.float64)
    samples = (low, high, (low + high) / 2)
    with mp.workdps(100):
        references = [_exact_values(*point) for point in samples]
        for order in range(3):
            selected = cast(Literal[0, 1, 2], order)
            program = preparation.derivative_program(patch, selected)
            assert preparation.derivative_program(patch, selected) is program
            lower, upper = program.evaluate(low[None, :], high[None, :])
            for reference in references:
                _assert_enclosed(lower[0], upper[0], reference[order])


@pytest.mark.parametrize(
    "box",
    [((-0.3, -0.8), (0.9, 0.4)), ((1.4, 2.8), (2.1, 3.4))],
)
def test_surface_box_encloses_exact_source(
    preparation: SourcePreparation, box: tuple[tuple[float, float], ...]
) -> None:
    patch = _torus()
    low, high = np.asarray(box, dtype=np.float64)
    bounds = preparation.surface_box(patch, np.asarray(box))
    with mp.workdps(100):
        for u in np.linspace(low[0], high[0], 5):
            for v in np.linspace(low[1], high[1], 5):
                _assert_enclosed(bounds[0], bounds[1], _exact_values(u, v)[0])


def test_source_differs_from_binary_shadow_at_exact_chart_origin(
    preparation: SourcePreparation,
) -> None:
    patch = _torus()
    source = CENTER[0] + (MAJOR + MINOR) * FIRST[0]
    shadow = F.from_float(float(CENTER[0])) + (
        F.from_float(float(MAJOR)) + F.from_float(float(MINOR))
    ) * F.from_float(float(FIRST[0]))
    assert source != shadow
    point = np.zeros((1, 2))
    lower, upper = preparation.derivative_program(patch, 0).evaluate(point, point)
    assert F.from_float(float(lower[0, 0])) <= source <= F.from_float(float(upper[0, 0]))


def test_degenerate_box_and_inconsistent_shadow_are_refused(
    preparation: SourcePreparation,
) -> None:
    patch = _torus()
    with pytest.raises(ValueError, match="increasing"):
        preparation.surface_box(patch, ((0.0, 0.0), (0.0, 1.0)))
    with pytest.raises(ValueError, match="order"):
        preparation.derivative_program(patch, cast(Any, 3))
    inconsistent = eqx.tree_at(lambda p: p.major_radius, patch, jnp.asarray(9.0))
    with pytest.raises(ValueError, match="shadow"):
        preparation.derivative_program(inconsistent, 0)


PLACEMENT = BRepAffinePlacement.from_matrix(
    (
        (F(4, 3), F(1, 7), F(0), F(2, 9)),
        (F(0), F(5, 4), F(-2, 5), F(-3, 11)),
        (F(1, 6), F(0), F(7, 5), F(1, 13)),
        (F(0), F(0), F(0), F(1)),
    )
)


def _placed_reference(values: np.ndarray, order: int) -> np.ndarray:
    result = np.empty_like(values)
    for index in np.ndindex(values.shape):
        row, rest = index[0], index[1:]
        result[index] = sum(
            _mp_fraction(PLACEMENT.matrix[row][column]) * values[(column, *rest)]
            for column in range(3)
        )
        if order == 0:
            result[index] += _mp_fraction(PLACEMENT.matrix[row][3])
    return result


@pytest.mark.parametrize(
    "box",
    [((0.0, 0.0), (0.0, 0.0)), ((-0.4, 0.2), (0.7, 1.1))],
)
def test_placed_source_program_encloses_exact_affine_torus(
    preparation: SourcePreparation, box: tuple[tuple[float, float], ...]
) -> None:
    patch = _PlacedSurfacePatch(_torus(), PLACEMENT)
    low, high = np.asarray(box, dtype=np.float64)
    with mp.workdps(100):
        for order in range(3):
            program = preparation.derivative_program(patch, cast(Literal[0, 1, 2], order))
            lower, upper = program.evaluate(low[None, :], high[None, :])
            for point in (low, high, (low + high) / 2):
                exact = _placed_reference(_exact_values(*point)[order], order)
                _assert_enclosed(lower[0], upper[0], exact)


def test_placed_sphere_program_encloses_binary_source_and_derivatives(
    preparation: SourcePreparation,
) -> None:
    center = (0.1, -0.2, 0.3)
    first = (0.7, 0.2, -0.1)
    second = (-0.4, 0.8, 0.2)
    axis = (0.1, -0.3, 1.2)
    radius = 1.3
    sphere = SpherePatch(center, first, second, axis, radius)
    surface = _PlacedSurfacePatch(sphere, PLACEMENT)
    low, high = np.asarray(((-0.4, 0.2), (0.7, 1.1)))
    with mp.workdps(100):
        for order in range(3):
            program = preparation.derivative_program(
                surface, cast(Literal[0, 1, 2], order)
            )
            lower, upper = program.evaluate(low[None, :], high[None, :])
            for u, v in (low, high, (low + high) / 2):
                cu, su = mp.cos(mp.mpf(u)), mp.sin(mp.mpf(u))
                cv, sv = mp.cos(mp.mpf(v)), mp.sin(mp.mpf(v))
                values = np.empty(((3,), (3, 2), (3, 2, 2))[order], dtype=object)
                for i in range(3):
                    c, a, b, z, r = map(
                        mp.mpf, (center[i], first[i], second[i], axis[i], radius)
                    )
                    radial, tangent = cu * a + su * b, -su * a + cu * b
                    if order == 0:
                        values[i] = c + r * (cv * radial + sv * z)
                    elif order == 1:
                        values[i, 0] = r * cv * tangent
                        values[i, 1] = r * (-sv * radial + cv * z)
                    else:
                        values[i, 0, 0] = -r * cv * radial
                        values[i, 0, 1] = values[i, 1, 0] = -r * sv * tangent
                        values[i, 1, 1] = -r * (cv * radial + sv * z)
                _assert_enclosed(lower[0], upper[0], _placed_reference(values, order))
    inconsistent = eqx.tree_at(
        lambda value: value.linear, surface, surface.linear.at[0, 0].set(9.0)
    )
    with pytest.raises(ValueError, match="shadow"):
        preparation.derivative_program(inconsistent, 0)


def _cylinder() -> CylinderPatch:
    return CylinderPatch(
        tuple(map(float, CENTER)),
        tuple(map(float, FIRST)),
        tuple(map(float, SECOND)),
        tuple(map(float, AXIS)),
        float(MINOR),
        origin_exact=CENTER,
        first_axis_exact=FIRST,
        second_axis_exact=SECOND,
        axis_exact=AXIS,
        radius_exact=MINOR,
    )


def _cylinder_exact_values(
    u: float, v: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u, v = mp.mpf(u), mp.mpf(v)
    cu, su = mp.cos(u), mp.sin(u)
    radius = _mp_fraction(MINOR)
    point = np.empty(3, dtype=object)
    jac = np.empty((3, 2), dtype=object)
    hess = np.empty((3, 2, 2), dtype=object)
    for i in range(3):
        origin, first, second, axis = map(
            _mp_fraction, (CENTER[i], FIRST[i], SECOND[i], AXIS[i])
        )
        radial = cu * first + su * second
        tangent = -su * first + cu * second
        point[i] = origin + radius * radial + v * axis
        jac[i, 0], jac[i, 1] = radius * tangent, axis
        hess[i, 0, 0] = -radius * radial
        hess[i, 0, 1] = hess[i, 1, 0] = hess[i, 1, 1] = mp.mpf(0)
    return point, jac, hess


@pytest.mark.parametrize("placed", (False, True))
@pytest.mark.parametrize(
    "box",
    [((0.0, 0.0), (0.0, 0.0)), ((-0.4, 0.2), (0.7, 1.1))],
)
def test_cylinder_source_program_encloses_exact_jet(
    preparation: SourcePreparation, box: tuple[tuple[float, float], ...], placed: bool
) -> None:
    patch = _cylinder()
    surface = _PlacedSurfacePatch(patch, PLACEMENT) if placed else patch
    low, high = np.asarray(box, dtype=np.float64)
    with mp.workdps(100):
        for order in range(3):
            program = preparation.derivative_program(
                surface, cast(Literal[0, 1, 2], order)
            )
            lower, upper = program.evaluate(low[None, :], high[None, :])
            for point in (low, high, (low + high) / 2):
                exact = _cylinder_exact_values(*point)[order]
                if placed:
                    exact = _placed_reference(exact, order)
                _assert_enclosed(lower[0], upper[0], exact)


def test_cylinder_source_program_refuses_mutated_radius_shadow(
    preparation: SourcePreparation,
) -> None:
    patch = _cylinder()
    inconsistent = eqx.tree_at(lambda p: p.radius, patch, jnp.asarray(9.0))
    with pytest.raises(ValueError, match="shadow"):
        preparation.derivative_program(inconsistent, 0)


@pytest.mark.parametrize("field", ("linear", "offset"))
def test_placed_source_program_refuses_mutated_placement_shadow(
    preparation: SourcePreparation, field: str
) -> None:
    patch = _PlacedSurfacePatch(_torus(), PLACEMENT)
    if field == "linear":
        inconsistent = eqx.tree_at(
            lambda p: p.linear, patch, patch.linear.at[0, 0].set(9.0)
        )
    else:
        inconsistent = eqx.tree_at(lambda p: p.offset, patch, patch.offset.at[0].set(9.0))
    with pytest.raises(ValueError, match="shadow"):
        preparation.derivative_program(inconsistent, 0)


def test_repaired_external_boolean_torus_cover_and_fidelity(tmp_path: Path) -> None:
    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.geometry._source_fidelity import _surface_jacobian_bounds
    from tools.meshing_qualification.scenarios.native_cad_profiles import _external_case

    # This fixture imports and roundtrips the donor STEP, explicitly repairs it
    # within its declared source budget, and performs the real Boolean cut.
    model, _, _, _ = _external_case("external-combined", "step", tmp_path)
    domain = MeshingDomain.from_brep(model)
    preparation = SourcePreparation(domain)
    torus_faces = []
    for index, patch in enumerate(domain.patches):
        surface = patch.surface
        native = surface.source if isinstance(surface, _PlacedSurfacePatch) else surface
        if isinstance(native, TorusPatch):
            torus_faces.append((index, surface, native))
    assert len(torus_faces) >= 2
    assert any(
        isinstance(radius, Fraction) and Fraction.from_float(float(radius)) != radius
        for _, _, patch in torus_faces
        for radius in (patch.major_radius_exact, patch.minor_radius_exact)
    )

    source = MeshingDomainBoundarySource(domain, tuple(range(len(domain.patches))))
    cover = source.boundary_chart_cover()
    assert cover.complete
    assert cover.semantics == "certified"
    assert not cover.findings
    assert cover.simplices.size

    triangle = ((F(0), F(0)), (F(1, 10), F(0)), (F(0), F(1, 10)))
    charts = np.asarray([[tuple(map(float, point)) for point in triangle]])
    box = np.asarray(((0.0, 0.0), (0.1, 0.1)))
    for index, surface, _ in torus_faces:
        bounds = preparation.surface_box(surface, box)
        assert np.all(np.isfinite(bounds))
        interpolation = domain.interpolation_bounds(
            index, charts, preparation=preparation
        )
        assert np.all(np.isfinite(interpolation))
        assert np.all(interpolation >= 0.0)
        jacobian = _surface_jacobian_bounds(surface, triangle, preparation)
        assert jacobian is not None
        assert np.all(np.isfinite(jacobian[0]))
        assert np.all(np.isfinite(jacobian[1]))


def test_repaired_step_cylinder_cover_uses_exact_source(tmp_path: Path) -> None:
    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.geometry._source_fidelity import _surface_jacobian_bounds
    from tools.meshing_qualification.scenarios.native_cad_profiles import _external_case

    model, _, _, _ = _external_case("external-routine", "step", tmp_path)
    domain = MeshingDomain.from_brep(model)
    preparation = SourcePreparation(domain)
    cylinder_faces = [
        (index, patch.surface)
        for index, patch in enumerate(domain.patches)
        if isinstance(
            patch.surface.source
            if isinstance(patch.surface, _PlacedSurfacePatch)
            else patch.surface,
            CylinderPatch,
        )
    ]
    assert len(cylinder_faces) == 2
    cover = MeshingDomainBoundarySource(
        domain, tuple(range(len(domain.patches)))
    ).boundary_chart_cover(preparation=preparation)
    assert cover.complete and cover.semantics == "certified"
    assert not cover.findings
    triangle = ((F(0), F(0)), (F(1, 10), F(0)), (F(0), F(1, 10)))
    charts = np.asarray([[tuple(map(float, point)) for point in triangle]])
    for index, surface in cylinder_faces:
        interpolation = domain.interpolation_bounds(
            index, charts, preparation=preparation
        )
        assert np.all(np.isfinite(interpolation))
        jacobian = _surface_jacobian_bounds(surface, triangle, preparation)
        assert jacobian is not None
        assert np.all(np.isfinite(jacobian[0])) and np.all(np.isfinite(jacobian[1]))
