"""Source-certified domain admission is invariant under binary similarities."""

from fractions import Fraction
from typing import Literal

import equinox as eqx
import numpy as np
import pytest
from hypothesis import example, given, settings

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry import (
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    EllipseCurve,
    LineCurve,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    PlanePatch,
)
from phydrax.geometry._meshing_domain import _use_endpoint_box, _use_point_box
from phydrax.geometry.brep import (
    brep_box,
    BRepGeometry,
    BRepOccurrence,
    BRepTessellationPolicy,
    NativePeriodEndpoint,
)
from phydrax.geometry.brep._constructors import assemble_brep_model
from phydrax.geometry.brep._model import BRepCurve
from tests._support.brep_metamorphic import binary_similarities, BinarySimilarity
from tests._support.meshing_domain import square_patch


def _transform_plane(
    surface: AbstractSurfacePatch, transform: BinarySimilarity
) -> PlanePatch:
    assert isinstance(surface, PlanePatch)
    return PlanePatch(
        transform.apply(np.asarray(surface.origin)),
        transform.apply(np.asarray(surface.first_axis)),
        transform.apply(np.asarray(surface.second_axis)),
    )


def _transform_line(curve: BRepCurve, transform: BinarySimilarity) -> LineCurve:
    assert isinstance(curve, LineCurve)
    return LineCurve(
        transform.apply(np.asarray(curve.origin)),
        transform.apply(np.asarray(curve.direction)),
    )


def _patches(
    transform: BinarySimilarity, defect: Literal["none", "roundoff", "curve", "corner"]
) -> tuple[MeshingSurfacePatch, ...]:
    first = square_patch((0, 1, 2, 3))
    second = MeshingSurfacePatch(first.surface, first.loops, reversed=True)
    if defect == "roundoff":
        first = MeshingSurfacePatch(
            PlanePatch((0, 0, 1), (1, 0, 0), (0, 1, 0)), first.loops
        )
        second = MeshingSurfacePatch(
            PlanePatch((0, 0, np.nextafter(1.0, 2.0)), (1, 0, 0), (0, 1, 0)),
            first.loops,
            reversed=True,
        )
    if defect == "curve":
        second = MeshingSurfacePatch(
            PlanePatch((0.0, 0.0, 2.0**-20), (1, 0, 0), (0, 1, 0)),
            first.loops,
            reversed=True,
        )
    elif defect == "corner":
        changed = PatchCurveUse(1, LineCurve((1.0 + 2.0**-20, 0.0), (0.0, 1.0)), 0.0, 1.0)
        first = MeshingSurfacePatch(
            first.surface, ((first.loops[0][0], changed, *first.loops[0][2:]),)
        )
        second = MeshingSurfacePatch(first.surface, first.loops, reversed=True)
    return tuple(
        MeshingSurfacePatch(
            _transform_plane(patch.surface, transform),
            patch.loops,
            reversed=patch.reversed,
        )
        for patch in (first, second)
    )


def _domain_decision(
    transform: BinarySimilarity, defect: Literal["none", "roundoff", "curve", "corner"]
) -> tuple[str, tuple[tuple[int, int], ...], tuple[int, ...]]:
    curves = tuple(MeshingDomainCurve(*ends) for ends in ((0, 1), (1, 2), (2, 3), (3, 0)))
    try:
        domain = MeshingDomain(
            _patches(transform, defect),
            curves,
            4,
            source_id="binary-similarity",
            source_revision="r1",
        )
    except ValueError as error:
        return str(error).split(" must", 1)[0], (), ()
    return (
        "admitted",
        tuple((curve.start, curve.end) for curve in domain.curves),
        tuple(len(domain.curve_uses(curve)) for curve in range(len(domain.curves))),
    )


@pytest.mark.parametrize("defect", ("none", "roundoff", "curve", "corner"))
@settings(max_examples=12, deadline=None)
@given(transform=binary_similarities())
@example(transform=BinarySimilarity((0, 1, 2), (1, 1, 1), -20))
@example(transform=BinarySimilarity((2, 0, 1), (-1, 1, -1), 20))
def test_binary_similarity_preserves_domain_admission(
    defect: Literal["none", "roundoff", "curve", "corner"], transform: BinarySimilarity
) -> None:
    expected = {
        "none": "admitted",
        "roundoff": "admitted",
        "curve": "MeshingDomain.curve_geometry",
        "corner": "MeshingDomain.curve_corners",
    }[defect]
    source = _domain_decision(BinarySimilarity((0, 1, 2), (1, 1, 1), 0), defect)
    assert source[0] == expected
    assert _domain_decision(transform, defect) == source


def _native_domain(
    transform: BinarySimilarity, defect: Literal["none", "curve", "corner"] = "none"
) -> MeshingDomain:
    source = brep_box(
        (-0.5, -0.5, -0.5),
        (0.5, 0.5, 0.5),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    geometry = source.require_geometry()
    patches = source.patches
    if defect == "curve":
        patch = patches[0]
        assert isinstance(patch, PlanePatch)
        changed_patch = PlanePatch(
            np.asarray(patch.origin) + np.asarray((0.0, 0.0, 2.0**-20)),
            patch.first_axis,
            patch.second_axis,
        )
        patches = (changed_patch, *patches[1:])
    elif defect == "corner":
        points = np.array(geometry.vertex_points, dtype=np.float64, copy=True)
        points[0, 0] += 2.0**-20
        geometry = eqx.tree_at(lambda value: value.vertex_points, geometry, points)
    transformed_curves = tuple(
        _transform_line(curve, transform) for curve in geometry.curves
    )
    transformed_geometry = eqx.tree_at(
        lambda value: (value.vertex_points, value.curves),
        geometry,
        (transform.apply(np.asarray(geometry.vertex_points)), transformed_curves),
    )
    transformed_surfaces = tuple(_transform_plane(patch, transform) for patch in patches)
    return MeshingDomain.from_brep_geometry(
        transformed_geometry,
        transformed_surfaces,
        np.asarray(source.orientation),
        source_id=source.source_id,
        source_revision=source.source_revision,
    )


@settings(max_examples=8, deadline=None)
@given(transform=binary_similarities())
@example(transform=BinarySimilarity((0, 1, 2), (1, 1, 1), -20))
@example(transform=BinarySimilarity((2, 0, 1), (-1, 1, -1), 20))
def test_binary_similarity_preserves_native_brep_domain(
    transform: BinarySimilarity,
) -> None:
    before = _native_domain(BinarySimilarity((0, 1, 2), (1, 1, 1), 0))
    after = _native_domain(transform)
    np.testing.assert_array_equal(
        after.corner_points, transform.apply(before.corner_points)
    )
    assert [(curve.start, curve.end) for curve in after.curves] == [
        (curve.start, curve.end) for curve in before.curves
    ]
    assert [len(after.curve_uses(i)) for i in range(len(after.curves))] == [
        len(before.curve_uses(i)) for i in range(len(before.curves))
    ]
    assert after.source_indices == before.source_indices


@pytest.mark.parametrize("defect", ("curve", "corner"))
@settings(max_examples=8, deadline=None)
@given(transform=binary_similarities())
@example(transform=BinarySimilarity((0, 1, 2), (1, 1, 1), -20))
@example(transform=BinarySimilarity((2, 0, 1), (-1, 1, -1), 20))
def test_binary_similarity_preserves_native_brep_refusal(
    defect: Literal["curve", "corner"], transform: BinarySimilarity
) -> None:
    with pytest.raises(ValueError) as source:
        _native_domain(BinarySimilarity((0, 1, 2), (1, 1, 1), 0), defect)
    with pytest.raises(ValueError) as target:
        _native_domain(transform, defect)
    assert (
        str(target.value).split(" must", 1)[0] == str(source.value).split(" must", 1)[0]
    )


def _placed_domain(transform: BinarySimilarity, defect: bool = False) -> MeshingDomain:
    radius = np.ldexp(0.5, transform.exponent)
    source = brep_box(
        (-radius, -radius, -radius),
        (radius, radius, radius),
        coordinate_contract=SpatialCoordinateContract.si(),
        tessellation=BRepTessellationPolicy(realize=False),
    )
    original = source.require_geometry()
    rotation = np.eye(3, dtype=np.float64)[np.asarray(transform.permutation)]
    rotation *= np.asarray(transform.signs)[:, None]
    geometry = BRepGeometry(
        vertex_points=original.vertex_points,
        curves=original.curves,
        edge_curves=original.edge_curves,
        edge_ranges=original.edge_ranges,
        edge_vertices=original.edge_vertices,
        pcurves=original.pcurves,
        coedge_edges=original.coedge_edges,
        coedge_senses=original.coedge_senses,
        face_loops=original.face_loops,
        shell_faces=original.shell_faces,
        shell_orientations=original.shell_orientations,
        solid_shells=original.solid_shells,
        occurrences=(
            BRepOccurrence(
                ("placed",),
                0,
                rotation,
                transform.apply(np.asarray((2.0, 3.0, 4.0))),
            ),
        ),
    )
    model = assemble_brep_model(
        geometry,
        source.patches,
        source.parameter_bounds,
        source.orientation,
        source.physical_tags,
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="placed-similarity",
        source_format="native",
        source_digest=source.source_digest,
        import_policy_id=source.import_policy_id,
        tessellation=BRepTessellationPolicy(realize=False),
    )
    if defect:
        points = np.array(geometry.vertex_points, dtype=np.float64, copy=True)
        points[0, 0] += np.ldexp(2.0**-20, transform.exponent)
        changed = eqx.tree_at(lambda value: value.vertex_points, geometry, points)
        model = eqx.tree_at(lambda value: value.geometry, model, changed)
    return MeshingDomain.from_brep(model)


@settings(max_examples=8, deadline=None)
@given(transform=binary_similarities())
@example(transform=BinarySimilarity((0, 1, 2), (1, 1, 1), -20))
@example(transform=BinarySimilarity((2, 0, 1), (-1, 1, -1), 20))
def test_binary_similarity_preserves_placed_brep_corners(
    transform: BinarySimilarity,
) -> None:
    source = _placed_domain(BinarySimilarity((0, 1, 2), (1, 1, 1), 0))
    target = _placed_domain(transform)
    np.testing.assert_array_equal(
        target.corner_points, transform.apply(source.corner_points)
    )
    assert target.source_indices == source.source_indices
    assert target.source_occurrences == source.source_occurrences
    assert [(curve.start, curve.end) for curve in target.curves] == [
        (curve.start, curve.end) for curve in source.curves
    ]
    with pytest.raises(ValueError, match="MeshingDomain.curve_corners"):
        _placed_domain(transform, defect=True)


def _partial_arc_domain(
    transform: BinarySimilarity, ellipse: bool, *, rooted: bool = True
) -> MeshingDomain:
    radius = np.ldexp(1.0, transform.exponent)
    if ellipse:
        curve = EllipseCurve((0, 0), (1, 0), (0, 1), 2 * radius, radius)
        turns = Fraction(1, 4)
        closing = (
            PatchCurveUse(1, LineCurve((0, radius), (0, -radius)), 0.0, 1.0),
            PatchCurveUse(2, LineCurve((0, 0), (2 * radius, 0)), 0.0, 1.0),
        )
        curves = ((0, 1), (1, 2), (2, 0))
    else:
        curve = CircleCurve((0, 0), (1, 0), (0, 1), radius)
        turns = Fraction(1, 2)
        closing = (PatchCurveUse(1, LineCurve((-radius, 0), (2 * radius, 0)), 0.0, 1.0),)
        curves = ((0, 1), (1, 0))
    arc = PatchCurveUse(
        0,
        curve,
        0.0,
        float(turns) * 2 * np.pi,
        last_root=NativePeriodEndpoint(curve, turns=turns) if rooted else None,
    )
    axes = BinarySimilarity(transform.permutation, transform.signs, 0)
    return MeshingDomain(
        (
            MeshingSurfacePatch(
                PlanePatch(
                    (0, 0, 0),
                    axes.apply(np.asarray((1.0, 0.0, 0.0))),
                    axes.apply(np.asarray((0.0, 1.0, 0.0))),
                ),
                ((arc, *closing),),
            ),
        ),
        tuple(MeshingDomainCurve(*ends) for ends in curves),
        len(curves),
        source_id="partial-arc",
        source_revision="r1",
    )


@pytest.mark.parametrize("ellipse", (False, True), ids=("half-circle", "quarter-ellipse"))
@settings(max_examples=8, deadline=None)
@given(transform=binary_similarities())
@example(transform=BinarySimilarity((0, 1, 2), (1, 1, 1), -20))
@example(transform=BinarySimilarity((2, 0, 1), (-1, 1, -1), 20))
def test_binary_similarity_preserves_rooted_partial_arc_joins(
    ellipse: bool, transform: BinarySimilarity
) -> None:
    source = _partial_arc_domain(BinarySimilarity((0, 1, 2), (1, 1, 1), 0), ellipse)
    target = _partial_arc_domain(transform, ellipse)
    np.testing.assert_array_equal(
        target.corner_points, transform.apply(source.corner_points)
    )
    assert [(curve.start, curve.end) for curve in target.curves] == [
        (curve.start, curve.end) for curve in source.curves
    ]
    # The declared modeling tolerance admits the tiny binary-source gap too;
    # the rooted endpoint still carries a distinct source-realization enclosure.
    unrooted = _partial_arc_domain(transform, ellipse, rooted=False)
    np.testing.assert_array_equal(unrooted.corner_points, target.corner_points)
    patch = target.patches[0]
    use = patch.loops[0][0]
    assert isinstance(use, PatchCurveUse)
    component = 0 if ellipse else 1
    binary_box = _use_point_box(use, use.last)
    rooted_box = _use_endpoint_box(patch, use, 1)
    assert not binary_box[0, component] <= 0.0 <= binary_box[1, component]
    assert rooted_box[0, component] <= 0.0 <= rooted_box[1, component]


@pytest.mark.parametrize("within_allowance", (False, True))
@settings(max_examples=8, deadline=None)
@given(transform=binary_similarities())
@example(transform=BinarySimilarity((0, 1, 2), (1, 1, 1), -20))
@example(transform=BinarySimilarity((2, 0, 1), (-1, 1, -1), 20))
def test_binary_similarity_preserves_declared_modeling_allowance(
    within_allowance: bool, transform: BinarySimilarity
) -> None:
    radius = np.ldexp(1.0, transform.exponent)
    gap = np.ldexp(2.0**-50 if within_allowance else 2.0**-20, transform.exponent)
    curve = BSplineCurve.bezier(
        ((radius, 0.0), (0.0, radius), (0.0, 0.0), (radius - gap, gap)),
        (1.0, 1.0, 1.0, 1.0),
    )
    axes = BinarySimilarity(transform.permutation, transform.signs, 0)
    patch = MeshingSurfacePatch(
        PlanePatch(
            (0, 0, 0),
            axes.apply(np.asarray((1.0, 0.0, 0.0))),
            axes.apply(np.asarray((0.0, 1.0, 0.0))),
        ),
        ((PatchCurveUse(0, curve, 0.0, 1.0),),),
    )
    if within_allowance:
        domain = MeshingDomain(
            (patch,),
            (MeshingDomainCurve(0, 0),),
            1,
            source_id="modeled-loop-gap",
            source_revision="r1",
        )
        assert domain.corner_count == 1
        assert len(domain.curve_uses(0)) == 1
    else:
        with pytest.raises(ValueError, match="MeshingDomain.curve_corners"):
            MeshingDomain(
                (patch,),
                (MeshingDomainCurve(0, 0),),
                1,
                source_id="modeled-loop-gap",
                source_revision="r1",
            )
