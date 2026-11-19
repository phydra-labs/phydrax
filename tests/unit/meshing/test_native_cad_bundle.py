#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native CAD geometry bundle against the JAX source evaluation.

References are the source classes' own JAX ``evaluate`` (values, and central
differences of it for derivatives) and, on fixtures, the 3D source curves:
every entity is evaluated natively at random parameters and compared within
4 ulp of the coordinate scale; derivative jets match Richardson-extrapolated
differences; on the quadric qualification fixture and a two-representation
spline fixture each curve's p-curve image on both incident faces matches the
curve within the domain tolerance plus binary64 resolution.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._interpolation._tspline import TMesh2D
from phydrax._meshcore import (
    cad_geometry_evaluate,
    CadEntityKind,
    CadGeometryQuantity,
    CadModel,
)
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
)
from phydrax.geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from phydrax.geometry.brep._intersection import intersect_surface_regions
from phydrax.geometry.brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    SurfaceRegion,
)
from phydrax.geometry.brep._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    HyperbolaCurve,
    LineCurve,
    OffsetCurve,
    OffsetSurface,
    ParabolaCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
    TorusRootedCirclePCurve,
)
from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface
from phydrax.geometry.brep._tspline import TSplineSurfacePatch
from phydrax.meshing.providers._native_cad_bundle import (
    cad_bundle_support_issues,
    cad_entity_model,
    cad_geometry_bundle,
    CadBundleSupportError,
)
from tools.generate_cad_geometry_programs import OUTPUT, render


_EPSILON = float(np.finfo(np.float64).eps)
_SAMPLES = 100_000
_JET_SAMPLES = 200
_E1 = np.asarray((0.36, 0.48, 0.8), dtype=np.float64)
_E2 = np.asarray((0.8, -0.6, 0.0), dtype=np.float64)
_AXIS = np.cross(_E1, _E2)
_ORIGIN = np.asarray((0.5, -1.25, 2.0), dtype=np.float64)
_ROTATION = np.asarray(((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
_TRANSLATION = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)

type _Entity = AbstractSurfacePatch | AbstractCurve
# entity, its parameter box (lower, upper per input), and for p-curves a
# carrier surface of the evaluation model.
type _Case = tuple[_Entity, tuple[tuple[float, float], ...]]


def _rotated_plane() -> PlanePatch:
    return PlanePatch(_ORIGIN, _E1, _E2)


def _cylinder() -> CylinderPatch:
    return CylinderPatch(_ORIGIN, _E1, _E2, _AXIS, 1.5)


def _cone() -> ConePatch:
    return ConePatch(_ORIGIN, _E1, _E2, _AXIS, 0.7, 0.3)


def _sphere() -> SpherePatch:
    return SpherePatch(_ORIGIN, _E1, _E2, _AXIS, 1.5)


def _torus() -> TorusPatch:
    return TorusPatch(_ORIGIN, _E1, _E2, _AXIS, 1.5, 0.4)


def _conic_root() -> ConicParameterRoot:
    circle = CircleCurve((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 1.0)
    return ConicParameterRoot(circle, Fraction(3, 5), Fraction(4, 5))


def _rational_curve(dimension: int, /) -> BSplineCurve:
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 0.5),
            (3.0, -1.0, 1.0),
            (4.0, 0.5, -1.0),
            (5.0, 1.0, 0.0),
        )
    )[:, :dimension]
    return BSplineCurve(
        points,
        np.asarray((1.0, 0.8, 1.3, 0.9, 1.0)),
        np.asarray((0.0, 0.0, 0.0, 0.0, 0.4, 1.0, 1.0, 1.0, 1.0)),
        3,
    )


def _rational_surface() -> BSplineSurfacePatch:
    u_count, v_count = 4, 3
    grid = np.empty((u_count, v_count, 3), dtype=np.float64)
    for row in range(u_count):
        for column in range(v_count):
            grid[row, column] = (row, column, 0.3 * np.sin(row + 2.0 * column))
    weights = 1.0 + 0.2 * np.cos(np.arange(u_count * v_count)).reshape((u_count, v_count))
    return BSplineSurfacePatch(
        grid,
        weights,
        np.asarray((0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)),
        np.asarray((0.0, 0.0, 0.0, 1.0, 1.0, 1.0)),
        2,
        2,
    )


def _rational_spline_surfaces() -> tuple[BSplineSurfacePatch, TSplineSurfacePatch]:
    """One exact rational surface (u / (1 + u), v, u v / (1 + u)) in two
    representations: a B-spline and a T-spline over the same T-mesh."""
    zero, knot, one = Fraction(0), Fraction(1, 2), Fraction(1)
    exact_points = (
        ((zero, zero, zero), (zero, one, zero)),
        ((Fraction(1, 3), zero, zero), (Fraction(1, 3), one, Fraction(1, 3))),
        ((Fraction(1, 2), zero, zero), (Fraction(1, 2), one, Fraction(1, 2))),
    )
    exact_weights = (
        (one, one),
        (Fraction(3, 2), Fraction(3, 2)),
        (Fraction(2), Fraction(2)),
    )
    knots = (zero, zero, knot, one, one)
    bspline = BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in exact_points
            ],
            dtype=np.float64,
        ),
        np.asarray([[float(value) for value in row] for row in exact_weights]),
        np.asarray(knots, dtype=np.float64),
        np.asarray((zero, zero, one, one), dtype=np.float64),
        1,
        1,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
        exact_u_knots=knots,
        exact_v_knots=(zero, zero, one, one),
    )
    mesh = TMesh2D.tensor_product((zero, knot, one), (zero, one))
    points: list[tuple[Fraction, Fraction, Fraction]] = []
    weights: list[Fraction] = []
    for anchor in mesh.anchors:
        u, v = anchor.local_knots.greville_parameter
        weight = one + u
        points.append((u / weight, v, u * v / weight))
        weights.append(weight)
    tspline = TSplineSurfacePatch(
        mesh,
        np.asarray([[float(value) for value in row] for row in points]),
        np.asarray([float(value) for value in weights]),
        exact_control_points=tuple(points),
        exact_weights=tuple(weights),
    )
    return bspline, tspline


_SURFACE_BOX = ((-7.0, 7.0), (-2.0, 2.0))


def _surface_cases() -> dict[str, _Case]:
    bspline, tspline = _rational_spline_surfaces()
    line = LineCurve(_ORIGIN + 1.5 * _E1, _AXIS)
    profile = _rational_curve(3)
    return {
        "plane": (_rotated_plane(), _SURFACE_BOX),
        "cylinder": (_cylinder(), _SURFACE_BOX),
        "cone": (_cone(), _SURFACE_BOX),
        "sphere": (_sphere(), ((-7.0, 7.0), (-1.5, 1.5))),
        "torus": (_torus(), ((-7.0, 7.0), (-7.0, 7.0))),
        "bspline": (_rational_surface(), ((0.0, 1.0), (0.0, 1.0))),
        "bspline_span": (bspline, ((0.0, 1.0), (0.0, 1.0))),
        "tspline": (tspline, ((0.0, 1.0), (0.0, 1.0))),
        "placed": (PlacedSurface(_torus(), _ROTATION, _TRANSLATION), _SURFACE_BOX),
        "offset_plane_chain": (
            OffsetSurface(OffsetSurface(_rotated_plane(), 0.25), -0.75),
            _SURFACE_BOX,
        ),
        "offset_sphere": (OffsetSurface(_sphere(), 0.25), ((-7.0, 7.0), (-1.4, 1.4))),
        "offset_cylinder": (OffsetSurface(_cylinder(), 0.25), _SURFACE_BOX),
        "offset_torus": (OffsetSurface(_torus(), 0.25), ((-7.0, 7.0), (-7.0, 7.0))),
        "offset_cone": (OffsetSurface(_cone(), 0.25), _SURFACE_BOX),
        "offset_generic": (
            OffsetSurface(_rational_surface(), 0.1),
            ((0.0, 1.0), (0.0, 1.0)),
        ),
        "offset_of_offset": (
            OffsetSurface(OffsetSurface(_cylinder(), 0.25), 0.25),
            _SURFACE_BOX,
        ),
        "ruled": (
            RuledSurface(
                CircleCurve(_ORIGIN, _E1, _E2, 1.5),
                EllipseCurve(_ORIGIN + _AXIS, _E1, _E2, 2.0, 1.0),
            ),
            _SURFACE_BOX,
        ),
        "extrusion": (ExtrusionSurface(profile, _AXIS), ((0.0, 1.0), (-2.0, 2.0))),
        "revolution_line": (RevolutionSurface(line, _ORIGIN, _AXIS), _SURFACE_BOX),
        "revolution_spline": (
            RevolutionSurface(profile, np.zeros(3), np.asarray((0.0, 0.0, 1.0))),
            ((-7.0, 7.0), (0.0, 1.0)),
        ),
    }


_LINE = ((-3.0, 3.0),)
_TURNS = ((-7.0, 7.0),)


def _curve_cases() -> dict[str, _Case]:
    circle = CircleCurve(_ORIGIN, _E1, _E2, 1.25)
    return {
        "line": (LineCurve(_ORIGIN, _E1), _LINE),
        "exact_line": (ExactLineCurve(_ORIGIN, _ORIGIN + _E2), _LINE),
        "circle": (circle, _TURNS),
        "ellipse": (EllipseCurve(_ORIGIN, _E1, _E2, 2.0, 0.75), _TURNS),
        "parabola": (ParabolaCurve(_ORIGIN, _E1, _E2, 0.5), _LINE),
        "hyperbola": (HyperbolaCurve(_ORIGIN, _E1, _E2, 2.0, 0.75), ((-2.0, 2.0),)),
        "bspline": (_rational_curve(3), ((0.0, 1.0),)),
        "placed": (PlacedCurve(circle, _ROTATION, _TRANSLATION), _TURNS),
        "offset_line_chain": (
            OffsetCurve(OffsetCurve(LineCurve(_ORIGIN, _E1), 0.5, _E2), 0.25, _AXIS),
            _LINE,
        ),
        "offset_generic": (
            OffsetCurve(EllipseCurve(_ORIGIN, _E1, _E2, 2.0, 0.75), 0.3, _AXIS),
            _TURNS,
        ),
        "surface_mapped": (
            SurfaceMappedCurve(_sphere(), LineCurve((0.0, 0.0), (1.0, 0.25))),
            _LINE,
        ),
        "isoparametric": (SurfaceIsoparametricCurve(_torus(), 1, 0.3), _TURNS),
    }


def _pcurve_cases() -> dict[str, _Case]:
    root = _conic_root()
    circle = CircleCurve((0.5, -0.25), (0.6, 0.8), (-0.8, 0.6), 1.25)
    return {
        "line": (LineCurve((0.5, -1.0), (2.0, 3.0)), _LINE),
        "exact_line": (ExactLineCurve((0.5, -1.0), (2.0, 3.0)), _LINE),
        "circle": (circle, _TURNS),
        "ellipse": (
            EllipseCurve((0.5, -0.25), (0.6, 0.8), (-0.8, 0.6), 2.0, 0.75),
            _TURNS,
        ),
        "parabola": (ParabolaCurve((0.5, -0.25), (0.6, 0.8), (-0.8, 0.6), 0.5), _LINE),
        "hyperbola": (
            HyperbolaCurve((0.5, -0.25), (0.6, 0.8), (-0.8, 0.6), 2.0, 0.75),
            ((-2.0, 2.0),),
        ),
        "bspline": (_rational_curve(2), ((0.0, 1.0),)),
        "conic_root_line": (
            ConicRootLinePCurve(root, Fraction(1, 2), Fraction(3, 2)),
            _LINE,
        ),
        "torus_rooted_forward": (TorusRootedCirclePCurve(root, Fraction(1, 4), 1), _LINE),
        "torus_rooted_reverse": (
            TorusRootedCirclePCurve(root, Fraction(1, 4), -1),
            _LINE,
        ),
        "plane_cylinder_section": (
            PlaneCylinderSectionPCurve(
                (Fraction(1, 10), Fraction(1, 5)),
                (Fraction(3, 10), Fraction(-2, 5)),
                (Fraction(1, 2), Fraction(3, 5)),
                (Fraction(-7, 10), Fraction(4, 5)),
                parameter_range=(-3.0, 3.0),
                angle_offset_root=ConicParameterRoot(
                    CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0),
                    Fraction(3, 5),
                    Fraction(4, 5),
                ),
            ),
            _LINE,
        ),
        "affine": (
            AffinePCurve(circle, np.asarray(((1, 2), (0, 1))), np.asarray((0.5, 0.0))),
            _TURNS,
        ),
        "periodic": (PeriodicPCurve(circle, _torus(), (1, -1)), _TURNS),
        "offset_circle": (OffsetCurve(circle, 0.5), _TURNS),
        "offset_line_chain": (
            OffsetCurve(OffsetCurve(LineCurve((0.5, -1.0), (2.0, 3.0)), 0.5), -0.2),
            _LINE,
        ),
        "trim_reversed": (
            CurveTrimSegment(circle, 0.25, 2.5, reversed=True),
            ((0.0, 1.0),),
        ),
    }


def _random_parameters(
    box: tuple[tuple[float, float], ...], count: int, seed: int, /
) -> np.ndarray:
    generator = np.random.default_rng(seed)
    columns = [generator.uniform(lower, upper, count) for lower, upper in box]
    return np.stack(columns, axis=1) if len(box) == 2 else columns[0]


def _evaluate(
    kind: CadEntityKind,
    entity: _Entity,
    parameters: np.ndarray,
    quantity: CadGeometryQuantity,
    /,
) -> np.ndarray:
    if isinstance(entity, AbstractSurfacePatch):
        model = cad_entity_model((entity,), (), ())
    elif kind is CadEntityKind.CURVE:
        model = cad_entity_model((), (entity,), ())
    else:
        model = cad_entity_model((_rotated_plane(),), (), ((entity, 0.0, 1.0),))
    return cad_geometry_evaluate(model, kind, 0, parameters, quantity)


def _assert_values(
    kind: CadEntityKind, entity: _Entity, box: tuple[tuple[float, float], ...], /
) -> None:
    parameters = _random_parameters(box, _SAMPLES, 7)
    native = _evaluate(kind, entity, parameters, CadGeometryQuantity.VALUE)
    reference = np.asarray(entity.evaluate(jnp.asarray(parameters)), dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(reference))))
    error = float(np.max(np.abs(native - reference.reshape(native.shape))))
    assert error <= 4.0 * _EPSILON * scale, (error / (_EPSILON * scale), scale)


def _difference(
    function: Callable[[np.ndarray], np.ndarray],
    parameters: np.ndarray,
    direction: np.ndarray,
    order: int,
    /,
) -> np.ndarray:
    """Central difference along ``direction`` with one Richardson step."""
    step = 1e-3

    def central(size: float, /) -> np.ndarray:
        plus = function(parameters + size * direction)
        minus = function(parameters - size * direction)
        if order == 1:
            return (plus - minus) / (2.0 * size)
        return (plus - 2.0 * function(parameters) + minus) / (size * size)

    return (4.0 * central(0.5 * step) - central(step)) / 3.0


def _assert_jets(
    kind: CadEntityKind, entity: _Entity, box: tuple[tuple[float, float], ...], /
) -> None:
    # Keep differences inside one polynomial span of every spline knot.
    inner = tuple((lower + 0.01, upper - 0.01) for lower, upper in box)
    parameters = _random_parameters(inner, _JET_SAMPLES, 11)
    native = _evaluate(kind, entity, parameters, CadGeometryQuantity.SECOND_DERIVATIVES)

    def reference(points: np.ndarray, /) -> np.ndarray:
        return np.asarray(entity.evaluate(jnp.asarray(points)), dtype=np.float64)

    values = reference(parameters)
    width = values.shape[-1]
    scale = max(1.0, float(np.max(np.abs(values))))
    if parameters.ndim == 1:
        expected = (
            values,
            _difference(reference, parameters, np.asarray(1.0), 1),
            _difference(reference, parameters, np.asarray(1.0), 2),
        )
    else:
        u, v = np.asarray((1.0, 0.0)), np.asarray((0.0, 1.0))
        mixed = 0.25 * (
            _difference(reference, parameters, u + v, 2)
            - _difference(reference, parameters, u - v, 2)
        )
        expected = (
            values,
            _difference(reference, parameters, u, 1),
            _difference(reference, parameters, v, 1),
            _difference(reference, parameters, u, 2),
            mixed,
            _difference(reference, parameters, v, 2),
        )
    for slot, block in enumerate(expected):
        actual = native[:, slot * width : (slot + 1) * width]
        assert np.max(np.abs(actual - block)) <= 1e-6 * scale, slot


def test_generated_template_programs_are_current() -> None:
    assert OUTPUT.read_text(encoding="utf-8") == render()


@pytest.mark.parametrize("name", tuple(_surface_cases()))
def test_surface_values_match_source_evaluate(name: str) -> None:
    entity, box = _surface_cases()[name]
    _assert_values(CadEntityKind.SURFACE, entity, box)


@pytest.mark.parametrize("name", tuple(_curve_cases()))
def test_curve_values_match_source_evaluate(name: str) -> None:
    entity, box = _curve_cases()[name]
    _assert_values(CadEntityKind.CURVE, entity, box)


@pytest.mark.parametrize("name", tuple(_pcurve_cases()))
def test_pcurve_values_match_source_evaluate(name: str) -> None:
    entity, box = _pcurve_cases()[name]
    _assert_values(CadEntityKind.PCURVE, entity, box)


# Jet boxes inside one knot span: differences across a knot are not smooth.
_SPAN = (0.52, 0.98)
_SURFACE_JET_BOXES = {
    "bspline": (_SPAN, (0.02, 0.98)),
    "bspline_span": (_SPAN, (0.02, 0.98)),
    "offset_generic": (_SPAN, (0.02, 0.98)),
    "revolution_spline": ((-7.0, 7.0), (0.42, 0.98)),
    "extrusion": ((0.42, 0.98), (-2.0, 2.0)),
}
_CURVE_JET_BOXES = {"bspline": ((0.42, 0.98),)}


@pytest.mark.parametrize("name", tuple(_surface_cases()))
def test_surface_jets_match_differences(name: str) -> None:
    entity, box = _surface_cases()[name]
    _assert_jets(CadEntityKind.SURFACE, entity, _SURFACE_JET_BOXES.get(name, box))


@pytest.mark.parametrize("name", tuple(_curve_cases()))
def test_curve_jets_match_differences(name: str) -> None:
    entity, box = _curve_cases()[name]
    _assert_jets(CadEntityKind.CURVE, entity, _CURVE_JET_BOXES.get(name, box))


@pytest.mark.parametrize("name", tuple(_pcurve_cases()))
def test_pcurve_jets_match_differences(name: str) -> None:
    entity, box = _pcurve_cases()[name]
    _assert_jets(CadEntityKind.PCURVE, entity, _CURVE_JET_BOXES.get(name, box))


def test_sphere_differential_has_umbilic_curvature() -> None:
    parameters = _random_parameters(((-7.0, 7.0), (-1.4, 1.4)), 100, 3)
    differential = _evaluate(
        CadEntityKind.SURFACE, _sphere(), parameters, CadGeometryQuantity.DIFFERENTIAL
    )
    latitude = parameters[:, 1]
    np.testing.assert_allclose(
        differential[:, 0], 2.25 * np.cos(latitude) ** 2, rtol=1e-13
    )
    np.testing.assert_allclose(differential[:, 2], 2.25, rtol=1e-13)
    np.testing.assert_allclose(differential[:, 6:], -1.0 / 1.5, rtol=1e-12)


def _quadric_model() -> phx.geometry.brep.BRepModel:
    contract = phx.SpatialCoordinateContract.si()
    vertical = phx.geometry.brep.brep_cylinder(
        1.0,
        2.4,
        base_center=(0.0, 0.0, -1.2),
        axis=(0.0, 0.0, 1.0),
        coordinate_contract=contract,
        source_id="qualification:native-cad-quadric-vertical",
    )
    transverse = phx.geometry.brep.brep_cylinder(
        0.6,
        3.0,
        base_center=(-1.5, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=contract,
        source_id="qualification:native-cad-quadric-transverse",
    )
    return phx.geometry.brep.boolean_brep(
        vertical,
        transverse,
        "intersection",
        policy=phx.geometry.brep.BRepBooleanPolicy(),
    ).model


@pytest.fixture(scope="module")
def quadric_domain() -> MeshingDomain:
    return MeshingDomain.from_brep(_quadric_model())


def _spline_domain() -> MeshingDomain:
    """Both representations of one exact rational surface as the two faces of
    a closed shell glued along the four boundary curves of the unit square."""
    bspline, tspline = _rational_spline_surfaces()
    corners = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    pairs = ((0, 1), (1, 2), (2, 3), (3, 0))

    def loop(sense: int, /) -> tuple[PatchCurveUse, ...]:
        uses = tuple(
            PatchCurveUse(
                curve,
                LineCurve(
                    np.asarray(corners[first]),
                    np.asarray(corners[last]) - np.asarray(corners[first]),
                ),
                0.0,
                1.0,
                sense=sense,
            )
            for curve, (first, last) in enumerate(pairs)
        )
        return uses if sense > 0 else tuple(reversed(uses))

    return MeshingDomain(
        (
            MeshingSurfacePatch(bspline, (loop(1),)),
            MeshingSurfacePatch(tspline, (loop(-1),), orientation=-1),
        ),
        tuple(MeshingDomainCurve(first, last) for first, last in pairs),
        4,
        source_id="native-cad-bundle-spline-shell",
        source_revision="r1",
    )


def _assert_curve_images_agree(domain: MeshingDomain, /) -> None:
    model = cad_geometry_bundle(domain).model
    bound = domain.tolerance + 16.0 * _EPSILON * domain.scale
    curves_seen: set[int] = set()
    for use in range(model.use_kinds.size):
        if model.use_kinds[use] != 0:
            continue
        loop = int(np.searchsorted(model.loop_use_offsets, use, side="right")) - 1
        face = int(model.loop_surfaces[loop])
        curve = int(model.use_entities[use])
        lower, upper = sorted(float(value) for value in model.use_ranges[use])
        parameters = np.linspace(lower, upper, 4097)
        chart = cad_geometry_evaluate(
            model, CadEntityKind.PCURVE, use, parameters, CadGeometryQuantity.VALUE
        )
        image = cad_geometry_evaluate(
            model, CadEntityKind.SURFACE, face, chart, CadGeometryQuantity.VALUE
        )
        carrier = cad_geometry_evaluate(
            model, CadEntityKind.CURVE, curve, parameters, CadGeometryQuantity.VALUE
        )
        gap = float(np.max(np.linalg.norm(image - carrier, axis=1)))
        assert gap <= bound, (use, curve, face, gap)
        curves_seen.add(curve)
    assert curves_seen == set(range(model.curve_nodes.size))


def test_quadric_fixture_curve_images_agree_on_both_faces(
    quadric_domain: MeshingDomain,
) -> None:
    model = cad_geometry_bundle(quadric_domain).model
    faces_per_curve = np.bincount(
        model.use_entities[model.use_kinds == 0], minlength=model.curve_nodes.size
    )
    assert np.all(faces_per_curve == 2)
    _assert_curve_images_agree(quadric_domain)


def test_spline_fixture_curve_images_agree_on_both_faces() -> None:
    _assert_curve_images_agree(_spline_domain())


def test_quadric_fixture_entities_match_source_evaluate(
    quadric_domain: MeshingDomain,
) -> None:
    model = cad_geometry_bundle(quadric_domain).model
    generator = np.random.default_rng(5)
    for face, patch in enumerate(quadric_domain.patches):
        parameters = np.stack(
            (generator.uniform(-7.0, 7.0, 10_000), generator.uniform(-1.2, 1.2, 10_000)),
            axis=1,
        )
        native = cad_geometry_evaluate(
            model, CadEntityKind.SURFACE, face, parameters, CadGeometryQuantity.VALUE
        )
        reference = np.asarray(patch.surface.evaluate(jnp.asarray(parameters)))
        assert np.max(np.abs(native - reference)) <= 4.0 * _EPSILON * quadric_domain.scale
    uses = tuple(
        use for patch in quadric_domain.patches for loop in patch.loops for use in loop
    )
    for row, use in enumerate(uses):
        lower, upper = sorted((use.first, use.last))
        parameters = generator.uniform(lower, upper, 10_000)
        native = cad_geometry_evaluate(
            model, CadEntityKind.PCURVE, row, parameters, CadGeometryQuantity.VALUE
        )
        reference = np.asarray(use.pcurve.evaluate(jnp.asarray(parameters)))
        scale = max(1.0, float(np.max(np.abs(reference))))
        assert np.max(np.abs(native - reference)) <= 4.0 * _EPSILON * scale, row


def test_quadric_fixture_intersection_curves_match_source_evaluate(
    quadric_domain: MeshingDomain,
) -> None:
    authority = quadric_domain.brep_authority
    assert authority is not None
    geometry = authority.geometry
    model = cad_geometry_bundle(quadric_domain).model
    generator = np.random.default_rng(9)
    for curve in range(model.curve_nodes.size):
        source = geometry.curves[
            geometry.edge_curves[quadric_domain.source_indices[1][curve]]
        ]
        lower, upper = model.curve_ranges[curve]
        parameters = generator.uniform(lower, upper, _SAMPLES)
        native = cad_geometry_evaluate(
            model, CadEntityKind.CURVE, curve, parameters, CadGeometryQuantity.VALUE
        )
        reference = np.asarray(source.evaluate(jnp.asarray(parameters)))
        assert isinstance(source, (IntersectionCurve, LineCurve))
        assert np.max(np.abs(native - reference)) <= 4.0 * _EPSILON * quadric_domain.scale


def test_intersection_pcurve_kinds_are_native(quadric_domain: MeshingDomain) -> None:
    kinds = {
        type(use.pcurve)
        for patch in quadric_domain.patches
        for loop in patch.loops
        for use in loop
    }
    assert IntersectionPCurve in kinds
    assert cad_bundle_support_issues(quadric_domain) == ()


def test_bundle_tables_mirror_the_domain(quadric_domain: MeshingDomain) -> None:
    model = cad_geometry_bundle(quadric_domain).model
    uses = tuple(
        use for patch in quadric_domain.patches for loop in patch.loops for use in loop
    )
    assert model.surface_nodes.size == len(quadric_domain.patches)
    assert model.curve_nodes.size == len(quadric_domain.curves)
    assert model.use_kinds.size == len(uses)
    np.testing.assert_array_equal(
        model.use_ranges, np.asarray([(use.first, use.last) for use in uses])
    )
    np.testing.assert_array_equal(model.use_senses, [use.sense for use in uses])
    np.testing.assert_array_equal(
        model.surface_orientations,
        [patch.orientation for patch in quadric_domain.patches],
    )
    np.testing.assert_array_equal(model.corner_points, quadric_domain.corner_points)
    assert model.region_surface_offsets.size == len(quadric_domain.regions) + 1
    assert all(not array.flags.writeable for array in model)


@pytest.mark.parametrize("kind", ("sphere", "cylinder", "cone"))
def test_pole_edges_do_not_alias_curve_rows(kind: str) -> None:
    contract = phx.SpatialCoordinateContract.si()
    if kind == "sphere":
        source = phx.geometry.brep.brep_sphere(1.0, coordinate_contract=contract)
    elif kind == "cylinder":
        source = phx.geometry.brep.brep_cylinder(1.0, 1.0, coordinate_contract=contract)
    else:
        source = phx.geometry.brep.brep_cone(1.0, 0.0, 1.0, coordinate_contract=contract)
    domain = MeshingDomain.from_brep(source)
    bundle = cad_geometry_bundle(domain)
    model = bundle.model
    rows = bundle.edge_curve_rows
    geometry = source.geometry
    expected = [
        geometry.edge_curves[domain.source_indices[1][edge]] >= 0
        for edge in range(len(domain.curves))
    ]
    np.testing.assert_array_equal(rows >= 0, expected)
    np.testing.assert_array_equal(rows[rows >= 0], np.arange(sum(expected)))
    assert not rows.flags.writeable
    assert model.curve_nodes.size == sum(expected)
    for edge, row in enumerate(rows):
        if row < 0:
            continue
        source_curve = geometry.curves[
            geometry.edge_curves[domain.source_indices[1][edge]]
        ]
        lower, upper = model.curve_ranges[row]
        parameters = np.linspace(lower, upper, 7)
        actual = cad_geometry_evaluate(
            model, CadEntityKind.CURVE, int(row), parameters, CadGeometryQuantity.VALUE
        )
        np.testing.assert_allclose(
            actual, source_curve.evaluate(jnp.asarray(parameters)), rtol=0.0, atol=1e-14
        )
        np.testing.assert_array_equal(
            model.curve_corners[row], (domain.curves[edge].start, domain.curves[edge].end)
        )
    uses = [use for patch in domain.patches for loop in patch.loops for use in loop]
    for index, use in enumerate(uses):
        entity = rows[use.curve] if isinstance(use, PatchCurveUse) else use.corner
        assert model.use_entities[index] == entity


def test_intersection_pcurve_crosses_declared_periodic_chart_transition() -> None:
    first = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    second = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    bounds = np.asarray(((0.0, -2.0), (math.tau, 2.0)), dtype=np.float64)
    result = intersect_surface_regions(
        SurfaceRegion(first, bounds), SurfaceRegion(second, bounds)
    )
    assert result.complete
    candidates = [
        (curve, node)
        for curve in result.curves
        for node in range(1, curve.num_charts)
        if np.any(
            np.asarray(curve.node_period_shifts[node, 0, :2])
            != np.asarray(curve.node_period_shifts[node, 1, :2])
        )
    ]
    assert candidates
    curve, node = candidates[0]
    pcurve = curve.p_curve("first")
    parameters = np.asarray((node - 1e-6, float(node), node + 1e-6), dtype=np.float64)
    model = cad_entity_model((first,), (), ((pcurve, pcurve.first, pcurve.last),))
    native = cad_geometry_evaluate(
        model, CadEntityKind.PCURVE, 0, parameters, CadGeometryQuantity.VALUE
    )
    reference = np.asarray(pcurve.evaluate(jnp.asarray(parameters)))
    assert np.max(np.abs(native - reference)) <= 4.0 * _EPSILON * max(
        1.0, float(np.max(np.abs(reference)))
    )


def test_bundle_is_cached_by_domain_identity(quadric_domain: MeshingDomain) -> None:
    first = cad_geometry_bundle(quadric_domain)
    second = cad_geometry_bundle(quadric_domain)
    assert first.bundle_id == quadric_domain.domain_id
    assert first.model is second.model


class _UnsupportedSurface(AbstractSurfacePatch):
    """A kind without a native node: reported, never approximated."""

    def evaluate(self, parameters: jnp.ndarray, /) -> jnp.ndarray:
        values = jnp.asarray(parameters, dtype=jnp.float64)
        u, v = values[..., 0], values[..., 1]
        return jnp.stack((u / (1.0 + u), v, u * v / (1.0 + u)), axis=-1)


def test_unsupported_kinds_are_reported_not_approximated() -> None:
    domain = _spline_domain()
    unsupported = MeshingDomain(
        (
            MeshingSurfacePatch(_UnsupportedSurface(), domain.patches[0].loops),
            domain.patches[1],
        ),
        domain.curves,
        4,
        source_id="native-cad-bundle-unsupported",
        source_revision="r1",
    )
    issues = cad_bundle_support_issues(unsupported)
    assert any("_UnsupportedSurface" in issue for issue in issues)
    with pytest.raises(CadBundleSupportError, match="_UnsupportedSurface"):
        cad_geometry_bundle(unsupported)


def test_spline_curve_refuses_parameters_outside_its_knots() -> None:
    model: CadModel = cad_entity_model((), (_rational_curve(3),), ())
    with pytest.raises(ValueError, match="cad_geometry_evaluate"):
        cad_geometry_evaluate(
            model,
            CadEntityKind.CURVE,
            0,
            np.asarray((0.5, 1.5)),
            CadGeometryQuantity.VALUE,
        )
