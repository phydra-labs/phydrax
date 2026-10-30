from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._exact_algebraic_geometry import RadicalElement
from phydrax.geometry._atlas import BoundaryAtlas
from phydrax.geometry.brep._assembly import BRepAffinePlacement
from phydrax.geometry.brep._boolean_core import AffinePCurve
from phydrax.geometry.brep._conic_gaps import native_face_gap_squared
from phydrax.geometry.brep._constructors import (
    brep_planar_face,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._correspondence import exact_source_lift
from phydrax.geometry.brep._intersection_curve import (
    exact_curve_parameter_point,
    SurfaceRegion,
)
from phydrax.geometry.brep._model import BRepBoundaryMap, BRepModel
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
    TorusPatch,
)
from phydrax.geometry.brep._placed import PlacedCurve, PlacedSurface


def _surface(
    control_points: tuple[tuple[tuple[Fraction, ...], ...], ...],
) -> BSplineSurfacePatch:
    knots = (Fraction(0), Fraction(0), Fraction(0), Fraction(1), Fraction(1), Fraction(1))
    return BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in control_points
            ],
            dtype=np.float64,
        ),
        np.ones((3, 3), dtype=np.float64),
        np.asarray([float(value) for value in knots], dtype=np.float64),
        np.asarray([float(value) for value in knots], dtype=np.float64),
        2,
        2,
        exact_control_points=control_points,
        exact_weights=tuple(tuple(Fraction(1) for _ in range(3)) for _ in range(3)),
        exact_u_knots=knots,
        exact_v_knots=knots,
    )


def _model_with_patch(source: BRepModel, patch: BSplineSurfacePatch, /) -> BRepModel:
    return BRepModel(
        geometry=source.geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def test_spline_boundary_lift_is_replayed_for_valid_and_changed_sources() -> None:
    contract = SpatialCoordinateContract.si()
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
        source_id="model-gap-authority",
    )
    controls = tuple(
        tuple(
            (
                Fraction(u, 2),
                Fraction(v, 2),
                Fraction(1, 4) if (u, v) == (1, 1) else Fraction(0),
            )
            for v in range(3)
        )
        for u in range(3)
    )
    valid = _model_with_patch(source, _surface(controls))
    valid.require_current()

    changed = [list(row) for row in controls]
    changed[0][1] = (Fraction(0), Fraction(1, 2), Fraction(1, 4))
    with pytest.raises(ValueError):
        _model_with_patch(source, _surface(tuple(tuple(row) for row in changed)))


@pytest.mark.parametrize("height", (0.0, 2.0**-20), ids=("exact-lift", "positive-gap"))
def test_affine_spline_pcurve_preserves_parameter_interval_and_gap(height: float) -> None:
    pcurve = BSplineCurve(
        np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64),
        np.ones((2,), dtype=np.float64),
        np.asarray((2.0, 2.0, 4.0, 4.0), dtype=np.float64),
        1,
    )
    source = LineCurve((-1.0, 0.0, height), (0.5, 0.0, 0.0))
    region = SurfaceRegion(
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
    )
    assert exact_source_lift(source, pcurve, region, 2.0, 4.0) == (height == 0.0)
    assert (
        native_face_gap_squared(source, pcurve, region, 2.0, 4.0) == Fraction(height) ** 2
    )


def test_spline_pcurve_same_image_does_not_prove_same_parameter_lift() -> None:
    pcurve = BSplineCurve(
        np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64),
        np.asarray((1.0, 2.0), dtype=np.float64),
        np.asarray((2.0, 2.0, 4.0, 4.0), dtype=np.float64),
        1,
    )
    source = LineCurve((-1.0, 0.0, 0.0), (0.5, 0.0, 0.0))
    region = SurfaceRegion(
        PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
    )
    assert not exact_source_lift(source, pcurve, region, 2.0, 4.0)
    squared_gap = native_face_gap_squared(source, pcurve, region, 2.0, 4.0)
    # At parameter 3 the rational chart gives 2/3, while the edge gives 1/2.
    assert squared_gap is None or squared_gap >= Fraction(1, 36)


@pytest.mark.parametrize(
    ("cosine", "sine", "matches"),
    ((Fraction(0), Fraction(1), True), (Fraction(1), Fraction(0), False)),
    ids=("upper-tube-circle", "different-tube-angle"),
)
def test_torus_ring_lift_binds_exact_meridian_root(
    cosine: Fraction, sine: Fraction, matches: bool
) -> None:
    patch = TorusPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        3.0,
        1.0,
    )
    meridian = CircleCurve((3.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
    root = ConicParameterRoot(meridian, cosine, sine)
    chart = AffinePCurve(
        ConicRootLinePCurve(
            root, Fraction(0), Fraction(1), parameter_range=(0.0, 2 * np.pi)
        ),
        ((0, 1), (1, 0)),
        (0, 0),
    )
    upper_ring = CircleCurve((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 3.0)
    region = SurfaceRegion(
        patch, np.asarray(((0.0, 0.0), (2 * np.pi, 2.0)), dtype=np.float64)
    )
    assert exact_source_lift(upper_ring, chart, region, 0.0, 2 * np.pi) is matches
    if matches:
        assert native_face_gap_squared(upper_ring, chart, region, 0.0, 2 * np.pi) == 0
        rounded_chart = LineCurve((0.0, np.pi / 2), (1.0, 0.0))
        assert not exact_source_lift(upper_ring, rounded_chart, region, 0.0, 2 * np.pi)


@pytest.mark.parametrize(
    "upper", (0.0, -1.0, np.nan), ids=("empty", "reversed", "nonfinite")
)
def test_traced_boundary_map_refuses_invalid_chart_interval(upper: float) -> None:
    @eqx.filter_jit
    def evaluate(bound: Array) -> Array:
        patch = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        limits = jnp.stack(
            (jnp.zeros((2,), dtype=jnp.float64), jnp.stack((bound, jnp.ones_like(bound))))
        )[None, ...]
        mapping = BRepBoundaryMap((patch,), limits)
        return mapping.map(
            jnp.asarray((0,), dtype=jnp.int32),
            jnp.asarray(((0.5, 0.5),), dtype=jnp.float64),
        )

    with pytest.raises(eqx.EquinoxRuntimeError, match="finite and nonempty"):
        evaluate(jnp.asarray(upper, dtype=jnp.float64)).block_until_ready()


@pytest.mark.parametrize("sign", (0.0, np.nan), ids=("zero", "nonfinite"))
def test_traced_boundary_atlas_refuses_invalid_orientation(sign: float) -> None:
    @eqx.filter_jit
    def normal(orientation: Array) -> Array:
        patch = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        mapping = BRepBoundaryMap(
            (patch,), jnp.asarray((((0.0, 0.0), (1.0, 1.0)),), dtype=jnp.float64)
        )
        atlas = BoundaryAtlas(
            mapping,
            source_entity_ids=jnp.asarray((0,), dtype=jnp.int32),
            source_id="traced-orientation",
            orientation=orientation[None],
        )
        return atlas.frame(
            jnp.asarray((0,), dtype=jnp.int32),
            jnp.asarray(((0.5, 0.5),), dtype=jnp.float64),
        ).normal

    with pytest.raises(eqx.EquinoxRuntimeError, match="orientation entries"):
        normal(jnp.asarray(sign, dtype=jnp.float64)).block_until_ready()


def test_radical_torus_source_and_placement_preserve_exact_geometry() -> None:
    radius = RadicalElement.sqrt(Fraction(2))
    patch = TorusPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        float(radius),
        0.25,
        major_radius_exact=radius,
    )
    rounded = TorusPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        float(radius),
        0.25,
    )
    ring_radius = radius + Fraction(1, 4)
    ring = CircleCurve(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        float(ring_radius),
        radius_exact=ring_radius,
    )
    chart = LineCurve((0.0, 0.0), (1.0, 0.0))
    domain = np.asarray(((0.0, 0.0), (2 * np.pi, 2 * np.pi)), dtype=np.float64)
    assert exact_source_lift(ring, chart, SurfaceRegion(patch, domain), 0.0, 2 * np.pi)
    assert not exact_source_lift(
        ring, chart, SurfaceRegion(rounded, domain), 0.0, 2 * np.pi
    )
    rotation = np.eye(3, dtype=np.float64)
    translation = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    placed = PlacedSurface(patch, rotation, translation)
    placed_rounded = PlacedSurface(rounded, rotation, translation)
    assert placed.identity_payload() != placed_rounded.identity_payload()
    bounds = placed.bounding_box(domain)
    derivative_lower, derivative_upper = patch.derivative_bounds(
        np.asarray(((0.0, 0.0), (0.125, 0.125)), dtype=np.float64)
    )
    with localcontext() as context:
        context.prec = 80
        reach = Decimal(2).sqrt() + Decimal("0.25")
        assert Decimal.from_float(float(bounds[0, 0])) <= Decimal(1) - reach
        assert Decimal.from_float(float(bounds[1, 0])) >= Decimal(1) + reach
        assert Decimal.from_float(float(bounds[0, 1])) <= -reach
        assert Decimal.from_float(float(bounds[1, 1])) >= reach
        assert Decimal.from_float(float(derivative_lower[1, 0])) <= reach
        assert Decimal.from_float(float(derivative_upper[1, 0])) >= reach
    assert bounds[0, 2] <= -0.25 and bounds[1, 2] >= 0.25


def test_torus_refuses_inconsistent_exact_radius_shadow() -> None:
    with pytest.raises(ValueError):
        TorusPatch(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
            0.25,
            major_radius_exact=RadicalElement.sqrt(Fraction(2)),
        )


def test_torus_refuses_incomplete_exact_center() -> None:
    with pytest.raises(ValueError):
        TorusPatch(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            2.0,
            0.25,
            center_exact=(Fraction(0), Fraction(0)),
        )


def test_exact_curve_endpoint_uses_declared_affine_not_rounded_pose() -> None:
    curve = ExactLineCurve(
        (Fraction(1, 3), Fraction(0), Fraction(0)),
        (Fraction(4, 3), Fraction(0), Fraction(0)),
    )
    affine = BRepAffinePlacement(
        (
            (Fraction(1), Fraction(0), Fraction(0), Fraction(-1, 3)),
            (Fraction(0), Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(1), Fraction(0)),
            (Fraction(0), Fraction(0), Fraction(0), Fraction(1)),
        )
    )
    placed = PlacedCurve(
        curve,
        np.eye(3, dtype=np.float64),
        np.asarray((-1 / 3, 0.0, 0.0), dtype=np.float64),
        affine=affine,
    )
    assert exact_curve_parameter_point(placed, Fraction(0)) == (
        Fraction(0),
        Fraction(0),
        Fraction(0),
    )
