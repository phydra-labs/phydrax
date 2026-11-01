from collections.abc import Callable
from math import pi

import build123d as bd
import numpy as np
import pytest
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.TopoDS import TopoDS_Shape

from phydrax._mass import EstimatedMass, ExactMass
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_cylinder,
    brep_revolution,
    brep_sphere,
    brep_torus,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._defeature import _hole_wall_data, brep_defeature
from phydrax.geometry.brep._measure import (
    brep_mass_properties,
    brep_shell_volume_sign,
    BRepMassProperties,
    BRepMeasureRefusal,
)
from phydrax.geometry.brep._model import BRepGeometry, BRepModel
from phydrax.geometry.brep._patches import (
    BSplineSurfacePatch,
    PlanePatch,
    RevolutionSurface,
)
from phydrax.geometry.brep._source import BRepSource
from phydrax.interchange import import_occt_shape


_COORDINATES = SpatialCoordinateContract.si()


def _contains(bounds: np.ndarray, reference: float, /) -> None:
    assert bounds[0] <= reference <= bounds[1]


def _with_assembly_graph(model: BRepModel, graph: BRepAssemblyGraph, /) -> BRepModel:
    source = model.geometry
    geometry = BRepGeometry(
        vertex_points=source.vertex_points,
        vertex_exact_points=source.vertex_exact_points,
        vertex_roots=source.vertex_roots,
        edge_endpoint_roots=source.edge_endpoint_roots,
        coedge_endpoint_roots=source.coedge_endpoint_roots,
        edge_vertex_gap_bounds=source.edge_vertex_gap_bounds,
        edge_face_gap_bounds=source.edge_face_gap_bounds,
        curves=source.curves,
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=graph,
    )
    return BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=np.asarray(model.parameter_bounds),
        orientation=np.asarray(model.orientation),
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
    )


def test_box_measure_is_source_bound_and_shell_sign_remains_distinct() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        coordinate_contract=_COORDINATES,
    )

    measured = brep_mass_properties(model, relative_tolerance=1e-6)

    assert measured.status == ("certified",)
    assert measured.model_id == model.model_id
    assert measured.solid_ids[0].source_revision == model.source_revision
    assert measured.solid_ids[0].occurrence_path == ("solid-0",)
    _contains(np.asarray(measured.volume_enclosures)[0], 6.0)
    # Affine faces with line trims integrate exactly in rationals.
    assert measured.exact == (True,)
    assert np.array_equal(np.asarray(measured.volume_enclosures)[0], (6.0, 6.0))
    assert measured.evaluations == (0,)
    assert brep_shell_volume_sign(model, 0) == 1


@pytest.mark.parametrize(
    ("radius", "height"),
    ((1.0, 2.0), (0.0315, 0.0015)),
    ids=("regular", "thin-band"),
)
def test_curved_shell_volume_sign_resolves_trims_along_chart_edges(
    radius: float, height: float
) -> None:
    # The lateral trim runs along the chart box edges; a thin band would leave
    # every closed cell touching it, so only open cell interiors decide.
    model = brep_cylinder(radius, height, coordinate_contract=_COORDINATES)
    assert brep_shell_volume_sign(model, 0) == 1
    assert (
        brep_shell_volume_sign(brep_sphere(radius, coordinate_contract=_COORDINATES), 0)
        == 1
    )


def _occt_volume(shape: TopoDS_Shape, /) -> tuple[float, float]:
    """OCCT's independent adaptive volume and its reported error estimate."""
    properties = GProp_GProps()
    error = BRepGProp.VolumeProperties_s(shape, properties, 1e-12, True)
    return float(properties.Mass()), float(error)


def _three_section_loft() -> TopoDS_Shape:
    wires = [
        bd.Wire.make_circle(1.0, plane=bd.Plane(origin=(0.0, 0.0, 0.0))),
        bd.Wire.make_circle(1.25, plane=bd.Plane(origin=(0.15, -0.05, 0.8))),
        bd.Wire.make_circle(0.9, plane=bd.Plane(origin=(-0.1, 0.1, 1.7))),
    ]
    return bd.Solid.make_loft(wires).wrapped


def _assert_certified(
    measured: BRepMassProperties, reference: float, tolerance: float, /
) -> None:
    assert measured.status == ("certified",), measured.reasons
    lower, upper = (float(value) for value in measured.volume_enclosures[0])
    assert lower <= reference <= upper
    assert (upper - lower) / 2.0 <= tolerance * lower
    assert measured.evaluations[0] > 0


@pytest.mark.parametrize(
    ("build", "reference"),
    (
        (
            lambda: brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES),
            2.0 * pi,
        ),
        (
            lambda: brep_sphere(1.0, coordinate_contract=_COORDINATES),
            4.0 * pi / 3.0,
        ),
        (
            lambda: brep_torus(2.0, 0.5, coordinate_contract=_COORDINATES),
            2.0 * pi**2 * 2.0 * 0.5**2,
        ),
        (
            # Planar caps trimmed by circle p-curves integrate their Green area.
            lambda: (
                import_occt_shape(
                    bd.Cylinder(1.0, 2.0).wrapped, coordinate_contract=_COORDINATES
                ).model
            ),
            2.0 * pi,
        ),
    ),
    ids=("cylinder", "sphere", "torus", "occt-cylinder"),
)
def test_curved_volume_meets_requested_certified_half_width(
    build: Callable[[], BRepModel], reference: float
) -> None:
    tolerance = 1e-4
    measured = brep_mass_properties(build(), relative_tolerance=tolerance)
    _assert_certified(measured, reference, tolerance)
    assert measured.exact == (False,)


def test_bspline_loft_volume_matches_independent_occt_integration() -> None:
    shape = _three_section_loft()
    model = import_occt_shape(shape, coordinate_contract=_COORDINATES).model
    assert isinstance(model.patches[0], BSplineSurfacePatch)
    reference, error = _occt_volume(shape)

    measured = brep_mass_properties(model, relative_tolerance=1e-10)

    # Polynomial Bernstein spans integrate exactly; only the certified edge
    # gaps of the imported source widen the enclosure.
    assert measured.status == ("certified",), measured.reasons
    lower, upper = (float(value) for value in measured.volume_enclosures[0])
    assert (upper - lower) / 2.0 <= 1e-10 * lower
    assert lower - error <= reference <= upper + error


def test_unattainable_tolerance_refuses_with_reached_enclosure() -> None:
    model = brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES)

    measured = brep_mass_properties(model, relative_tolerance=1e-17)

    assert measured.status == ("unresolved",)
    assert "resolution" in measured.reasons[0]
    assert np.all(np.isinf(np.asarray(measured.volume_enclosures)[0]))
    reached = np.asarray(measured.reached_enclosures)[0]
    assert np.all(np.isfinite(reached))
    _contains(reached, 2.0 * pi)


def test_source_interior_mass_is_exact_for_planar_and_estimated_for_curved() -> None:
    box = BRepSource(
        brep_box((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=_COORDINATES)
    ).compile()
    assert isinstance(box.interior_mass, ExactMass)
    assert float(box.measure) == 6.0

    tolerance = 1e-4
    sphere = BRepSource(
        brep_sphere(1.0, coordinate_contract=_COORDINATES),
        measure_relative_tolerance=tolerance,
    ).compile()
    mass = sphere.interior_mass
    assert isinstance(mass, EstimatedMass)
    value, uncertainty = float(mass.value), float(mass.uncertainty)
    assert abs(value - 4.0 * pi / 3.0) <= uncertainty <= tolerance * value

    refused = BRepSource(
        brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES),
        measure_relative_tolerance=1e-17,
    ).compile()
    with pytest.raises(BRepMeasureRefusal) as refusal:
        _ = refused.measure
    assert refusal.value.properties is not None
    assert refusal.value.properties.status == ("unresolved",)
    lower, upper = refusal.value.enclosure
    assert lower <= 2.0 * pi <= upper


def test_hollow_boolean_measure_subtracts_oriented_cavity_shell() -> None:
    outer = brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 2.0, 2.0),
        coordinate_contract=_COORDINATES,
    )
    inner = brep_box(
        (0.5, 0.5, 0.5),
        (1.5, 1.5, 1.5),
        coordinate_contract=_COORDINATES,
    )
    result = boolean_brep(outer, inner, "difference")
    assert result.model is not None

    measured = brep_mass_properties(result.model, relative_tolerance=1e-6)

    assert measured.status == ("certified",), measured.reasons
    _contains(np.asarray(measured.volume_enclosures)[0], 7.0)


def test_assembly_measure_applies_exact_placement_per_occurrence() -> None:
    definition = brep_box(
        (0.0, 0.0, 0.0), (1.0, 2.0, 3.0), coordinate_contract=_COORDINATES
    )
    reflected_scale = BRepAffinePlacement.from_matrix(
        ((-2, 0, 0, 10), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    graph = BRepAssemblyGraph(
        roots=(
            BRepAssemblyUse("unit-copy", solid_index=0),
            BRepAssemblyUse("reflected-double", placement=reflected_scale, solid_index=0),
        ),
        definitions=(),
        solid_count=1,
    )
    assembly = _with_assembly_graph(definition, graph)

    measured = brep_mass_properties(assembly, relative_tolerance=1e-6)

    assert measured.status == ("certified", "certified")
    bounds = np.asarray(measured.volume_enclosures)
    _contains(bounds[0], 6.0)
    _contains(bounds[1], 12.0)
    assert measured.occurrence_paths == (("unit-copy",), ("reflected-double",))
    assert measured.solid_ids[0].index == measured.solid_ids[1].index == 0
    assert measured.solid_ids[0].occurrence_path != measured.solid_ids[1].occurrence_path


def test_missing_curved_source_trim_reports_unsupported_without_finite_result() -> None:
    source = brep_sphere(1.0, coordinate_contract=_COORDINATES)
    face = next(
        index
        for index, patch in enumerate(source.patches)
        if not isinstance(patch, PlanePatch)
    )
    trims = list(source.trim_domains)
    trims[face] = None
    incomplete = BRepModel(
        geometry=source.geometry,
        patches=source.patches,
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=tuple(trims),
    )

    measured = brep_mass_properties(incomplete, relative_tolerance=1e-6)

    assert measured.status == ("unsupported",)
    assert measured.reasons[0]
    assert np.all(np.isinf(np.asarray(measured.volume_enclosures)[0]))
    assert np.isnan(np.asarray(measured.volume)[0])


def test_conical_hole_defeature_publishes_measured_shape_change() -> None:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop.polygon(((0.5, 0.0), (2.0, 0.0), (2.0, 1.0), (0.75, 1.0))),
    )
    source = brep_revolution(
        profile, (0.0, 0.0), (0.0, 1.0), coordinate_contract=_COORDINATES
    )
    face = next(
        index
        for index, patch in enumerate(source.patches)
        if isinstance(patch, RevolutionSurface)
        and (wall := _hole_wall_data(source, index)) is not None
        and wall.family == "revolved-cone"
    )

    result = brep_defeature(source, source.face_ids[face], operation="hole")

    assert result.status == "certified", result.reason
    assert result.volume_kind == "enclosed"
    expected = pi * (0.5**2 + 0.5 * 0.75 + 0.75**2) / 3.0
    _contains(np.asarray(result.volume_change_enclosure), expected)
