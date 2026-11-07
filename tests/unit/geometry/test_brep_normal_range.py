from fractions import Fraction
from math import atan, comb, pi

import jax.numpy as jnp
import numpy as np

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._constructors import (
    brep_cone,
    brep_cylinder,
    brep_planar_face,
    brep_sphere,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._normal_range import brep_normal_ranges, BRepNormalRange
from phydrax.geometry.brep._patches import BSplineSurfacePatch


_CONTRACT = SpatialCoordinateContract.si()
type _Poly = dict[tuple[int, int], Fraction]
_PULL = (0.0, 0.0, 1.0)


def _ranges(model: BRepModel) -> tuple[BRepNormalRange, ...]:
    return brep_normal_ranges(model, _PULL, draft_angle=0.0, angular_tolerance=2.0e-2)


def _face_at_height(model: BRepModel, height: float) -> int:
    """Select a face by an evaluated interior point, independent of carrier type."""
    for face, patch in enumerate(model.patches):
        bounds = np.asarray(model.parameter_bounds, dtype=np.float64)[face]
        point = np.asarray(patch.evaluate(jnp.asarray(0.5 * (bounds[0] + bounds[1]))))
        if abs(point[2] - height) <= 1.0e-12:
            return face
    raise AssertionError(f"No face has an interior point at height {height}.")


def test_cylinder_normal_ranges_match_analytic_side_and_cap_cones() -> None:
    model = brep_cylinder(1.0, 2.0, coordinate_contract=_CONTRACT)
    ranges = _ranges(model)
    side = ranges[_face_at_height(model, 1.0)]
    bottom = ranges[_face_at_height(model, 0.0)]
    top = ranges[_face_at_height(model, 2.0)]
    side_enclosure = np.asarray(side.angle_enclosure)
    bottom_enclosure = np.asarray(bottom.angle_enclosure)
    top_enclosure = np.asarray(top.angle_enclosure)

    # Outward side normals are horizontal; disc caps point along -z and +z,
    # including at the profile point where each cap meets the revolution axis.
    assert side.status == "vertical"
    assert side_enclosure[0] <= pi / 2.0 <= side_enclosure[1]
    assert side_enclosure[1] - side_enclosure[0] < 1.0e-10
    assert bottom.status == "undercut"
    assert bottom_enclosure[0] <= pi <= bottom_enclosure[1]
    assert bottom_enclosure[1] - bottom_enclosure[0] < 1.0e-6
    assert top.status == "positive"
    assert top_enclosure[0] <= 0.0 <= top_enclosure[1]
    assert top_enclosure[1] - top_enclosure[0] < 1.0e-6
    for face in (side, bottom, top):
        assert np.asarray(face.unresolved_cells).shape == (0, 2, 2)
        assert np.asarray(face.singular_points).shape == (0, 2, 3)


def test_cone_normal_range_contains_analytic_draft_angle() -> None:
    model = brep_cone(1.0, 2.0, 3.0, coordinate_contract=_CONTRACT)
    side = _ranges(model)[_face_at_height(model, 1.5)]
    # Radius grows by 1 over height 3: the outward normal is (3 e_r - e_z) / sqrt(10).
    expected = pi / 2.0 + atan(1.0 / 3.0)
    enclosure = np.asarray(side.angle_enclosure)

    assert side.status == "undercut"
    assert enclosure[0] <= expected <= enclosure[1]
    assert enclosure[1] - enclosure[0] <= 4.0e-2


def test_sphere_poles_are_declared_singular_points_with_exact_evidence() -> None:
    model = brep_sphere(1.0, coordinate_contract=_CONTRACT)
    (face,) = _ranges(model)
    enclosure = np.asarray(face.angle_enclosure)
    cells = np.asarray(face.singular_cells)
    points = np.asarray(face.singular_points)

    assert enclosure[0] <= 0.0
    assert enclosure[1] >= pi
    assert enclosure[1] <= np.nextafter(pi, np.inf)
    # Both poles are degenerate edges whose exact collapse the model proved, so
    # every cell is decided: the remaining status reflects normals on both sides
    # of the pull plane, not undecided cells.
    assert face.status == "unresolved"
    assert np.asarray(face.unresolved_cells).shape == (0, 2, 2)
    assert face.unresolved_reasons == ()
    assert {item.kind for item in face.singularities} == {"declared_degenerate_edge"}
    poles = {item.point: item for item in face.singularities}
    north = (Fraction(0), Fraction(0), Fraction(1))
    south = (Fraction(0), Fraction(0), Fraction(-1))
    assert set(poles) == {north, south}
    assert poles[north].edge_id != poles[south].edge_id
    # The pole latitude is its exact conic root, not the binary64 latitude
    # (077ec39e8; docs/api/geometry.md), so the evidence box is the root's
    # certified enclosure: it must contain the exact latitude +-pi/2, which
    # lies strictly between float(pi)/2 and nextafter(float(pi), inf)/2.
    half_pi = (Fraction(pi) / 2, Fraction(float(np.nextafter(pi, np.inf))) / 2)
    for cell, spatial, item in zip(cells, points, face.singularities, strict=True):
        sign = 1 if item.point == north else -1
        lower, upper = sorted(sign * value for value in item.parameter_box[1])
        assert lower <= half_pi[0] and half_pi[1] <= upper
        assert cell[0, 1] <= sign * 0.5 * pi <= cell[1, 1]
        # Shrunk toward the pole while each split-off strip is regular.
        assert cell[1, 1] - cell[0, 1] <= 1.0e-12
        assert np.array_equal(spatial, np.asarray((item.point, item.point), float))


_U: _Poly = {(1, 0): Fraction(1)}
_V: _Poly = {(0, 1): Fraction(1)}


def _constant(value: Fraction) -> _Poly:
    return {(0, 0): value}


def _add(*polys: _Poly) -> _Poly:
    result: _Poly = {}
    for poly in polys:
        for key, value in poly.items():
            result[key] = result.get(key, Fraction(0)) + value
    return {key: value for key, value in result.items() if value}


def _multiply(*polys: _Poly) -> _Poly:
    result: _Poly = {(0, 0): Fraction(1)}
    for poly in polys:
        product: _Poly = {}
        for (a, b), left in result.items():
            for (c, d), right in poly.items():
                product[(a + c, b + d)] = product.get((a + c, b + d), 0) + left * right
        result = {key: value for key, value in product.items() if value}
    return result


def _scale(value: Fraction, poly: _Poly) -> _Poly:
    return {key: value * item for key, item in poly.items()}


def _evaluate(poly: _Poly, u: Fraction, v: Fraction) -> Fraction:
    return sum((value * u**a * v**b for (a, b), value in poly.items()), Fraction(0))


def _spline_face(components: tuple[_Poly, _Poly, _Poly], degree: int) -> BRepModel:
    """Unit-square face of a polynomial patch equal to ``(u, v, 0)`` on its border."""

    def bernstein(poly: _Poly, i: int, j: int) -> Fraction:
        return sum(
            (
                Fraction(comb(i, a), comb(degree, a))
                * Fraction(comb(j, b), comb(degree, b))
                * poly.get((a, b), Fraction(0))
                for a in range(i + 1)
                for b in range(j + 1)
            ),
            Fraction(0),
        )

    controls = tuple(
        tuple(
            tuple(bernstein(poly, i, j) for poly in components) for j in range(degree + 1)
        )
        for i in range(degree + 1)
    )
    knots = (Fraction(0),) * (degree + 1) + (Fraction(1),) * (degree + 1)
    weights = tuple(tuple(Fraction(1) for _ in row) for row in controls)
    patch = BSplineSurfacePatch(
        np.asarray(controls, dtype=np.float64),
        np.ones((degree + 1, degree + 1), dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        degree,
        degree,
        exact_control_points=controls,
        exact_weights=weights,
        exact_u_knots=knots,
        exact_v_knots=knots,
    )
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=_CONTRACT,
        source_id="normal-range-spline",
    )
    return BRepModel(
        geometry=source.geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=_CONTRACT,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def _bump() -> _Poly:
    # 16 u (1 - u) v (1 - v): zero on the unit-square border.
    return _scale(
        Fraction(16),
        _multiply(
            _U,
            _add(_constant(Fraction(1)), _scale(Fraction(-1), _U)),
            _V,
            _add(_constant(Fraction(1)), _scale(Fraction(-1), _V)),
        ),
    )


def test_cross_cap_patch_certifies_exact_interior_singular_points() -> None:
    # x = u, y = v + bump * k(v), z = bump * (u - 1/3) / 2. Along u = 1/3 the
    # restriction y_v = 1 + (P(v) - 1) with P = 150/17 (v - 2/5)(v - 1/5), so
    # S_v = 0 at v = 1/5 and v = 2/5: two Whitney-umbrella points (rank-two
    # cross-product Jacobian) at rational, non-dyadic parameters. On v = 1/2,
    # the other line where z_v = 0, y_v >= 1 - 9/8 * 25/34 > 0.
    # k = (9/32) int_0^v (P - 1) / (v (1 - v)) = -(45/544) (10 v + 1).
    k = {(0, 0): Fraction(-45, 544), (0, 1): Fraction(-225, 272)}
    components = (
        _U,
        _add(_V, _multiply(_bump(), k)),
        _scale(Fraction(1, 2), _multiply(_bump(), _add(_U, _constant(Fraction(-1, 3))))),
    )
    (face,) = _ranges(_spline_face(components, 3))

    assert np.asarray(face.unresolved_cells).shape == (0, 2, 2)
    assert face.unresolved_reasons == ()
    assert np.asarray(face.angle_enclosure).tolist() == [
        0.0,
        float(np.nextafter(pi, np.inf)),
    ]
    expected = {(Fraction(1, 3), Fraction(1, 5)), (Fraction(1, 3), Fraction(2, 5))}
    found = {}
    for item, cell, spatial in zip(
        face.singularities,
        np.asarray(face.singular_cells),
        np.asarray(face.singular_points),
        strict=True,
    ):
        assert item.kind == "isolated_cross_product_zero"
        (u, u_), (v, v_) = item.parameter_box
        assert (u, v) == (u_, v_)
        assert cell[0, 0] <= u <= cell[1, 0] and cell[0, 1] <= v <= cell[1, 1]
        assert item.point == tuple(_evaluate(poly, u, v) for poly in components)
        assert np.all(spatial[0] <= np.asarray(item.point, float))
        assert np.all(np.asarray(item.point, float) <= spatial[1])
        found[(u, v)] = item
    assert set(found) == expected


def test_near_singular_regular_patch_resolves_below_former_singular_width() -> None:
    # A planar sheet x = u, y = v - (1 - eta) alpha(u) beta(v) with alpha <= 1
    # peaking at u = 1/3 and beta' = 1 - 12 (v - 1/2)^2 <= 1: |S_u x S_v| = y_v
    # >= eta = 2^-18 everywhere, with the minimum at the non-dyadic u = 1/3. The
    # enclosure there needs cells far below 2^-8 of the face width, where the
    # previous fixed singular resolution left the face unresolved.
    eta = Fraction(1, 2**18)
    one = _constant(Fraction(1))
    alpha = _scale(
        Fraction(27, 4),
        _multiply(
            _U, _add(one, _scale(Fraction(-1), _U)), _add(one, _scale(Fraction(-1), _U))
        ),
    )
    centered = _add(_V, _constant(Fraction(-1, 2)))
    beta = _multiply(
        centered, _add(one, _scale(Fraction(-4), _multiply(centered, centered)))
    )
    components = (_U, _add(_V, _scale(eta - 1, _multiply(alpha, beta))), {})
    model = _spline_face(components, 3)
    (face,) = brep_normal_ranges(model, _PULL, draft_angle=0.0, angular_tolerance=1.0)

    assert face.status == "positive"
    assert np.asarray(face.unresolved_cells).shape == (0, 2, 2)
    assert face.singularities == ()
    enclosure = np.asarray(face.angle_enclosure)
    assert enclosure[0] == 0.0 and enclosure[1] < pi / 2.0


def test_insufficient_draft_needs_a_lower_angle_above_the_exact_limit() -> None:
    from phydrax.geometry.brep._normal_range import _status

    lower = float.fromhex("0x1.a5ba180991559p-2")
    upper = float.fromhex("0x1.a5ba18099155ap-2")
    draft = float.fromhex("0x1.28b12f41de7c2p+0")
    # The 36-digit truncation of pi is a rational lower bound of pi.
    pi_lower = Fraction(314159265358979323846264338327950288, 10**35)
    # The face's angle can still lie at or below pi/2 - draft.
    assert Fraction(lower) < pi_lower / 2 - Fraction(draft)

    assert _status(lower, upper, draft, False, False) != "insufficient"
