from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._interpolation import TMesh2D
from phydrax._interpolation._bspline_extraction import (
    _rational_bernstein_surface_span,
    RationalBSplineSurfaceSpan,
)
from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
)
from phydrax.geometry._domain_boundary import (
    exact_plane_chart,
    MeshingDomainBoundarySource,
)
from phydrax.geometry._meshing_domain import (
    _prepare_surface_spans,
    _span_enclosures_for_chart,
    MeshingDomain,
    MeshingDomainCurve,
    MeshingDomainRegion,
    MeshingSurfacePatch,
    PatchCurveUse,
    PatchPoleUse,
    use_endpoints,
)
from phydrax.geometry._source_identity import source_entity_id
from phydrax.geometry.brep import brep_box, brep_cylinder, brep_sphere
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._constructors import (
    assemble_brep_model,
    brep_extrusion,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._correspondence import native_edge_pcurve
from phydrax.geometry.brep._intersection_curve import exact_curve_parameter_point
from phydrax.geometry.brep._model import BRepGeometry, BRepModel
from phydrax.geometry.brep._patches import (
    AbstractCurve,
    BSplineSurfacePatch,
    CircleCurve,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.brep._tspline import TSplineSurfacePatch


def test_rooted_quadric_trim_joins_require_source_authority(tmp_path: Path) -> None:
    from phydrax.geometry._domain_boundary import (
        _exact_trim_triangles,
        _line_endpoints,
        _patch_chart_polygon,
        _rooted_patch_chart_endpoints,
        _rooted_patch_chart_junctions,
        _rooted_representative,
        CurvedTrimLens,
    )
    from phydrax.geometry._meshing_domain import (
        _same_brep_trim_endpoint,
        SourcePreparation,
    )
    from phydrax.geometry._source_fidelity import _curved_trim_lens_caps
    from phydrax.geometry.brep._boolean_core import PeriodicPCurve
    from phydrax.linalg._hermitian_spectral import fraction_sqrt_interval
    from phydrax.meshing.providers._native_surface import _trim_lens_bound
    from tools.meshing_qualification.scenarios.native_cad_profiles import _quadric_case

    model, _, _, _ = _quadric_case("iges", tmp_path)
    domain = MeshingDomain.from_brep(model)
    patch = 2
    smallest = float(np.nextafter(0.0, 1.0))
    box = np.asarray(((-smallest, 0.5), (smallest, 0.5)), dtype=np.float64)
    outside = (Fraction.from_float(-smallest), Fraction(1, 2))
    inside = (Fraction(0), Fraction(1, 2))
    assert (
        _rooted_representative(
            _rational_spline_surfaces()[0], outside, inside, (box, box)
        )
        == inside
    )
    uses = domain.patches[patch].loops[0]
    original = tuple((use.first, use.last, use.sense) for use in uses)
    raw = tuple(_line_endpoints(use) for use in uses)
    mismatches = tuple(
        slot
        for slot, (_, last) in enumerate(raw)
        if last != raw[(slot + 1) % len(raw)][0]
    )
    assert mismatches
    slot = mismatches[0]
    following = (slot + 1) % len(uses)
    endpoints = _rooted_patch_chart_endpoints(domain, patch)
    assert endpoints[0][slot][1] == endpoints[0][following][0]
    assert all(
        last == loop[(index + 1) % len(loop)][0]
        for loop in endpoints
        for index, (_, last) in enumerate(loop)
    )
    assert endpoints == _rooted_patch_chart_endpoints(domain, patch)
    assert original == tuple((use.first, use.last, use.sense) for use in uses)

    preparation = SourcePreparation(domain)
    junction = _rooted_patch_chart_junctions(domain, patch, preparation)[0][slot]
    assert junction.representative == endpoints[0][slot][1]
    assert junction.lower != junction.upper
    radius = fraction_sqrt_interval(
        sum(
            (
                max(
                    abs(junction.lower[axis] - junction.representative[axis]),
                    abs(junction.upper[axis] - junction.representative[axis]),
                )
                ** 2
                for axis in range(2)
            ),
            Fraction(0),
        )
    )[1]
    lens = CurvedTrimLens(
        (junction.representative, junction.representative),
        junction.lower,
        junction.upper,
        radius * 2,
    )
    representative = junction.representative
    chart = (
        representative,
        (representative[0] + radius, representative[1]),
        (representative[0], representative[1] + radius),
    )
    caps = _curved_trim_lens_caps(
        domain.patches[patch].surface, [(0, chart)], (lens,), preparation
    )
    assert caps[0] > 0
    reference = _patch_chart_polygon(domain, patch, rooted=True)
    assert reference is not None
    triangles = _exact_trim_triangles(*reference)
    source = MeshingDomainBoundarySource(domain, (patch,))
    _, actual_lenses = source._trim_partition_reference_unchecked(
        patch, triangles, preparation
    )
    assert any(
        value.chord == (representative, representative) and value.parameter_bound > 0
        for value in actual_lenses
    )

    # Use a proper subinterval of a real intersection branch rather than the
    # effectively straight p-curves on patch 2.
    branch_patch = 0
    use = domain.patches[branch_patch].loops[0][0]
    assert isinstance(use, PatchCurveUse)
    oriented = _rooted_patch_chart_endpoints(domain, branch_patch)[0][0]
    native_endpoints = oriented if use.sense > 0 else oriented[::-1]
    lower, middle, upper = Fraction(1, 8), Fraction(1, 4), Fraction(3, 8)
    whole = _trim_lens_bound(
        domain, branch_patch, use, native_endpoints, lower, upper, preparation
    )
    left = _trim_lens_bound(
        domain, branch_patch, use, native_endpoints, lower, middle, preparation
    )
    right = _trim_lens_bound(
        domain, branch_patch, use, native_endpoints, middle, upper, preparation
    )
    assert whole > 0
    assert left < whole
    assert right < whole

    authority = domain.brep_authority
    assert authority is not None
    face = domain.source_indices[2][patch]
    coedges = authority.geometry.face_loops[face][0]
    first, second = uses[slot], uses[following]
    assert _same_brep_trim_endpoint(
        authority, face, coedges[slot], coedges[following], first, second
    )
    surface = authority.patches[face]
    periods = surface.native_period_symbols()
    axis = periods.index("two_pi")
    shifts = (1, 0) if axis == 0 else (0, 1)
    shifted = eqx.tree_at(
        lambda use: use.pcurve,
        second,
        PeriodicPCurve(second.pcurve, surface, shifts),
    )
    assert not _same_brep_trim_endpoint(
        authority, face, coedges[slot], coedges[following], first, shifted
    )
    first_end = 1 if first.sense > 0 else 0
    vertex = authority.geometry.edge_vertices[
        authority.geometry.coedge_edges[coedges[slot]]
    ][first_end]
    other = next(
        index
        for index, use in enumerate(uses)
        if authority.geometry.edge_vertices[
            authority.geometry.coedge_edges[coedges[index]]
        ][0 if use.sense > 0 else 1]
        != vertex
    )
    assert not _same_brep_trim_endpoint(
        authority, face, coedges[slot], coedges[other], first, uses[other]
    )

    unbound = eqx.tree_at(lambda value: value.brep_authority, domain, None)
    with pytest.raises(ValueError):
        _rooted_patch_chart_endpoints(unbound, patch)


def test_coincident_conic_endpoint_shadows_require_root_authority() -> None:
    from phydrax.geometry._domain_boundary import (
        _line_endpoints,
        _rooted_patch_chart_junctions,
    )
    from tests.unit.interchange._native_cad_archive_worker import (
        _boolean_curve_carrier_model,
    )

    model = _boolean_curve_carrier_model(SpatialCoordinateContract.si())
    domain = MeshingDomain.from_brep(model)
    uses = domain.patches[0].loops[0]
    assert _line_endpoints(uses[0])[1] == _line_endpoints(uses[1])[0]
    junction = _rooted_patch_chart_junctions(domain, 0)[0][0]
    assert junction.lower == junction.upper == junction.representative

    if domain.brep_authority is None:
        raise ValueError("The conic fixture requires a BRep authority.")
    modified = eqx.tree_at(
        lambda value: value.geometry.vertex_roots[1], domain.brep_authority, None
    )
    unbound = eqx.tree_at(lambda value: value.brep_authority, domain, modified)
    with pytest.raises(ValueError, match="shared same-sheet UV root enclosure"):
        _rooted_patch_chart_junctions(unbound, 0)


def _rebuild_brep_model(
    model: BRepModel,
    /,
    *,
    assembly_graph: BRepAssemblyGraph,
) -> BRepModel:
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
        assembly_graph=assembly_graph,
    )
    return BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=np.asarray(model.parameter_bounds),
        orientation=np.asarray(model.orientation),
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=(None,) * len(model.patches),
    )


def _rational_parallelepiped() -> BRepModel:
    """An exact solid with thirds-valued corners and integer plane frames."""
    cube = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    source = cube.geometry
    cube_points = np.asarray(source.vertex_points)
    exact_points = tuple(
        (
            (Fraction(float(point[0])) + Fraction(float(point[1])) + 1) / 3,
            (-2 * Fraction(float(point[0])) + Fraction(float(point[1])) + 1) / 3,
            Fraction(float(point[2])),
        )
        for point in cube_points
    )
    curves = tuple(
        ExactLineCurve(exact_points[start], exact_points[end])
        for start, end in source.edge_vertices
    )
    patches: list[PlanePatch] = []
    pcurves: dict[int, AbstractCurve] = {}
    bounds: list[np.ndarray] = []
    for face, loops in enumerate(source.face_loops):
        original = cube.patches[face]
        if not isinstance(original, PlanePatch):
            raise TypeError("The fixture requires planar unit-cube source faces.")
        vertices = sorted(
            {
                vertex
                for loop in loops
                for coedge in loop
                for vertex in source.edge_vertices[source.coedge_edges[coedge]]
            }
        )
        points = cube_points[np.asarray(vertices, dtype=np.int64)]
        constant_axes = np.flatnonzero(np.all(points == points[0], axis=0))
        if constant_axes.size != 1:
            raise ValueError("Each unit-cube face must have one constant coordinate.")
        axis = constant_axes[0]
        level = points[0, axis]
        match axis:
            case 0:
                origin = (level, 0.0, 0.0)
            case 1:
                origin = (0.0, level + 1.0, 0.0)
            case 2:
                origin = (0.0, 0.0, level)
            case _:
                raise ValueError("The fixture has an invalid spatial axis.")
        # Positive rescaling of the affine-image axes preserves face orientation.
        axes = tuple(
            (vector[0] + vector[1], -2 * vector[0] + vector[1], 3 * vector[2])
            for vector in (
                np.asarray(original.first_axis),
                np.asarray(original.second_axis),
            )
        )
        patch = PlanePatch(origin, axes[0], axes[1])
        patches.append(patch)
        uv_points: list[tuple[ExactScalar, ...]] = []
        for loop in loops:
            for coedge in loop:
                pcurve = native_edge_pcurve(curves[source.coedge_edges[coedge]], patch)
                if pcurve is None:
                    raise ValueError(
                        "The exact solid edge must have a planar source lift."
                    )
                pcurves[coedge] = pcurve
                for parameter in (Fraction(0), Fraction(1)):
                    point = exact_curve_parameter_point(pcurve, parameter)
                    if point is None or len(point) != 2:
                        raise ValueError(
                            "The planar source lift needs exact UV endpoints."
                        )
                    uv_points.append(point)
        bounds.append(
            np.asarray(
                (
                    tuple(
                        min(coefficient_interval(point[axis])[0] for point in uv_points)
                        for axis in range(2)
                    ),
                    tuple(
                        max(coefficient_interval(point[axis])[1] for point in uv_points)
                        for axis in range(2)
                    ),
                ),
                dtype=np.float64,
            )
        )
    geometry = BRepGeometry(
        vertex_points=np.asarray(exact_points, dtype=np.float64),
        vertex_exact_points=exact_points,
        curves=curves,
        edge_curves=tuple(range(len(curves))),
        edge_ranges=np.tile(np.asarray((0.0, 1.0), dtype=np.float64), (len(curves), 1)),
        edge_vertices=source.edge_vertices,
        pcurves=tuple(pcurves[index] for index in range(len(source.coedge_edges))),
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
    )
    return assemble_brep_model(
        geometry,
        tuple(patches),
        np.asarray(bounds),
        cube.orientation,
        cube.physical_tags,
        coordinate_contract=cube.coordinate_contract,
        source_id="exact-rational-placement-solid",
        trim_domains=(None,) * len(patches),
    )


def _rebuild_brep_domain(
    domain: MeshingDomain,
    model: BRepModel,
    /,
    *,
    regions: tuple[MeshingDomainRegion, ...] | None = None,
    source_indices: tuple[tuple[int, ...], ...] | None = None,
    source_occurrences: tuple[tuple[tuple[str, ...], ...], ...] | None = None,
    region_source_indices: tuple[int, ...] | None = None,
    region_source_occurrences: tuple[tuple[str, ...], ...] | None = None,
    source_kinds: tuple[str, ...] | None = None,
) -> MeshingDomain:
    return MeshingDomain(
        domain.patches,
        domain.curves,
        domain.corner_count,
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        regions=domain.regions if regions is None else regions,
        tolerance=10.0,
        source_indices=(
            domain.source_indices if source_indices is None else source_indices
        ),
        source_occurrences=(
            domain.source_occurrences
            if source_occurrences is None
            else source_occurrences
        ),
        region_source_indices=(
            domain.region_source_indices
            if region_source_indices is None
            else region_source_indices
        ),
        region_source_occurrences=(
            domain.region_source_occurrences
            if region_source_occurrences is None
            else region_source_occurrences
        ),
        source_kinds=domain.source_kinds if source_kinds is None else source_kinds,
        authority_id=model.model_id,
        brep_authority=model,
        corner_points=domain.corner_points,
    )


def _square_domain() -> MeshingDomain:
    surface = PlanePatch((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(3, LineCurve((0.0, 0.0), (0.0, 1.0)), 0.0, 1.0, sense=-1),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(0, 3),
        ),
        4,
        source_id="square-source",
        source_revision="r1",
        source_indices=((11, 17, 23, 41), (101, 202, 303, 404), (900,)),
        source_occurrences=(
            ((), (), (), ()),
            (("instance-A",), (), (), ()),
            ((),),
        ),
        corner_points=np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
            dtype=np.float64,
        ),
    )


def _rational_spline_surfaces() -> tuple[BSplineSurfacePatch, TSplineSurfacePatch]:
    zero = Fraction(0)
    one = Fraction(1)
    knot = Fraction(1, 3)
    u_values = (zero, knot, one)
    v_values = (zero, one)
    exact_points = tuple(
        tuple(
            (
                u / (one + u),
                v,
                u * v / (one + u),
            )
            for v in v_values
        )
        for u in u_values
    )
    exact_weights = tuple(tuple(one + u for _ in v_values) for u in u_values)
    bspline = BSplineSurfacePatch(
        np.asarray(
            [
                [[float(value) for value in point] for point in row]
                for row in exact_points
            ],
            dtype=np.float64,
        ),
        np.asarray(
            [[float(value) for value in row] for row in exact_weights],
            dtype=np.float64,
        ),
        np.asarray((zero, zero, knot, one, one), dtype=np.float64),
        np.asarray((zero, zero, one, one), dtype=np.float64),
        1,
        1,
        exact_control_points=exact_points,
        exact_weights=exact_weights,
        exact_u_knots=(zero, zero, knot, one, one),
        exact_v_knots=(zero, zero, one, one),
    )

    mesh = TMesh2D.tensor_product((zero, knot, one), (zero, one))
    exact_tspline_points: list[tuple[Fraction, Fraction, Fraction]] = []
    exact_tspline_weights: list[Fraction] = []
    for anchor in mesh.anchors:
        u, v = anchor.local_knots.greville_parameter
        weight = one + u
        exact_tspline_points.append((u / weight, v, u * v / weight))
        exact_tspline_weights.append(weight)
    tspline = TSplineSurfacePatch(
        mesh,
        np.asarray(
            [[float(value) for value in row] for row in exact_tspline_points],
            dtype=np.float64,
        ),
        np.asarray([float(value) for value in exact_tspline_weights], dtype=np.float64),
        exact_control_points=tuple(exact_tspline_points),
        exact_weights=tuple(exact_tspline_weights),
    )
    return bspline, tspline


@pytest.mark.parametrize("surface_index", (0, 1), ids=("bspline", "tspline"))
def test_surface_span_exact_points_match_independent_rational_map(
    surface_index: int,
) -> None:
    surface = _rational_spline_surfaces()[surface_index]
    for span in surface.prepare_spans().spans:
        (u0, u1), (v0, v1) = span.parameter_bounds
        for u, v in ((u0, v0), (u1, v1), ((u0 + u1) / 2, (v0 + v1) / 2)):
            expected = (u / (1 + u), v, u * v / (1 + u))
            assert span.evaluate_exact((u, v)) == expected, (span.span_indices, u, v)


@pytest.mark.parametrize("surface_index", (0, 1), ids=("bspline", "tspline"))
def test_surface_span_exact_point_refuses_outside_span(surface_index: int) -> None:
    span = _rational_spline_surfaces()[surface_index].prepare_spans().spans[0]
    (u0, _), (v0, _) = span.parameter_bounds
    with pytest.raises(ValueError, match="declared nonempty span"):
        span.evaluate_exact((u0 - 1, v0))


def _rational_spline_domain(
    surface: BSplineSurfacePatch | TSplineSurfacePatch, source_id: str
) -> MeshingDomain:
    uses = (
        PatchCurveUse(0, LineCurve((0.0, 0.0), (1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(1, LineCurve((1.0, 0.0), (0.0, 1.0)), 0.0, 1.0),
        PatchCurveUse(2, LineCurve((1.0, 1.0), (-1.0, 0.0)), 0.0, 1.0),
        PatchCurveUse(3, LineCurve((0.0, 0.0), (0.0, 1.0)), 0.0, 1.0, sense=-1),
    )
    return MeshingDomain(
        (MeshingSurfacePatch(surface, (uses,)),),
        (
            MeshingDomainCurve(0, 1),
            MeshingDomainCurve(1, 2),
            MeshingDomainCurve(2, 3),
            MeshingDomainCurve(0, 3),
        ),
        4,
        source_id=source_id,
        source_revision="r1",
    )


def _expected_interpolation_error_squared(
    chart: tuple[tuple[Fraction, Fraction], ...],
) -> Fraction:
    lower_u = min(point[0] for point in chart)
    upper_u = max(point[0] for point in chart)
    lower_v = min(point[1] for point in chart)
    upper_v = max(point[1] for point in chart)
    width_u = upper_u - lower_u
    width_v = upper_v - lower_v
    denominator = Fraction(1) + lower_u
    x_error = width_u**2 / denominator**3
    z_error = upper_v * width_u**2 / denominator**3 + width_u * width_v / denominator**2
    return x_error**2 + z_error**2


def _assert_hessian_encloses_exact_extrema(
    span: RationalBSplineSurfaceSpan,
) -> None:
    enclosure = span.enclose_hessian(span.parameter_bounds)
    (lower_u, upper_u), (lower_v, upper_v) = span.parameter_bounds
    lower_denominator = Fraction(1) + lower_u
    upper_denominator = Fraction(1) + upper_u
    expected_du = (
        (
            Fraction(1) / upper_denominator**2,
            Fraction(1) / lower_denominator**2,
        ),
        (Fraction(0), Fraction(0)),
        (
            lower_v / upper_denominator**2,
            upper_v / lower_denominator**2,
        ),
    )
    expected_dv = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(1)),
        (lower_u / lower_denominator, upper_u / upper_denominator),
    )
    expected_uu = (
        (Fraction(-2) / lower_denominator**3, Fraction(-2) / upper_denominator**3),
        (Fraction(0), Fraction(0)),
        (
            Fraction(-2) * upper_v / lower_denominator**3,
            Fraction(-2) * lower_v / upper_denominator**3,
        ),
    )
    expected_uv = (
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(1) / upper_denominator**2, Fraction(1) / lower_denominator**2),
    )
    expected_vv = ((Fraction(0), Fraction(0)),) * 3
    assert enclosure.positive_denominator
    for actual, expected in zip(
        (
            enclosure.du_bounds,
            enclosure.dv_bounds,
            enclosure.duu_bounds,
            enclosure.duv_bounds,
            enclosure.dvv_bounds,
        ),
        (expected_du, expected_dv, expected_uu, expected_uv, expected_vv),
        strict=True,
    ):
        for actual_component, expected_component in zip(actual, expected, strict=True):
            assert actual_component[0] <= expected_component[0]
            assert actual_component[1] >= expected_component[1]


def test_rational_bernstein_surface_hessian_encloses_exact_extrema() -> None:
    for surface in _rational_spline_surfaces():
        spans = _prepare_surface_spans(surface)
        assert spans is not None
        assert spans.spans
        for span in spans.spans:
            assert span.has_hessian
            _assert_hessian_encloses_exact_extrema(span)


def test_rational_bernstein_surface_span_refuses_nonpositive_denominator() -> None:
    with pytest.raises(ValueError, match="did not certify a positive denominator"):
        _rational_bernstein_surface_span(
            "negative-weight-source",
            (0, 0),
            (1, 1),
            (0, 1, 2, 3),
            ((0, 1), (0, 1)),
            (
                ((0, 0, 1), (1, 0, 1)),
                ((0, 1, 1), (1, 1, -1)),
            ),
            include_hessian=True,
        )


def test_sphere_domain_retains_exact_poles_and_continuous_period_lifts() -> None:
    model = brep_sphere(
        1.0, coordinate_contract=SpatialCoordinateContract.si(), source_id="sphere-domain"
    )
    domain = MeshingDomain.from_brep(model)

    np.testing.assert_array_equal(
        domain.corner_points, ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
    )
    assert domain.authority_id == model.model_id
    for patch in domain.patches:
        for loop in patch.loops:
            for index, use in enumerate(loop):
                np.testing.assert_array_equal(
                    use_endpoints(use)[1], use_endpoints(loop[(index + 1) % len(loop)])[0]
                )
    pole_corners = {
        use.corner
        for patch in domain.patches
        for loop in patch.loops
        for use in loop
        if isinstance(use, PatchPoleUse)
    }
    assert pole_corners == {0, 1}
    domain.require_current()


def test_box_domain_retains_authoritative_solid_boundary() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="box-domain",
    )
    domain = MeshingDomain.from_brep(model)

    assert domain.source_kinds == ("vertex", "edge", "face")
    assert domain.region_source_indices == (0,)
    assert domain.region_source_occurrences == (("solid-0",),)
    assert domain.regions[0].boundary == tuple(
        zip(
            model.topology.solid_faces[0],
            model.topology.solid_face_orientations[0],
            strict=True,
        )
    )


def test_brep_domain_rejects_constructor_default_source_kinds() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    domain = MeshingDomain.from_brep(model)

    with pytest.raises(ValueError, match="source kinds"):
        MeshingDomain(
            domain.patches,
            domain.curves,
            domain.corner_count,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
            regions=domain.regions,
            source_indices=domain.source_indices,
            source_occurrences=domain.source_occurrences,
            authority_id=model.model_id,
            brep_authority=model,
            corner_points=domain.corner_points,
        )


def test_brep_geometry_cannot_claim_exact_domain_authority() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    geometry = model.geometry
    domain = MeshingDomain.from_brep_geometry(
        geometry,
        model.patches,
        np.asarray(model.orientation),
        source_id=model.source_id,
        source_revision=model.source_revision,
        corner_points=np.asarray(geometry.vertex_points),
    )
    assert domain.brep_authority is None
    assert domain.source_kinds == ("corner", "curve", "surface")
    first = domain.patches[0]
    assert isinstance(first.surface, PlanePatch)
    shifted = PlanePatch(
        np.asarray(first.surface.origin) + np.asarray((0.25, 0.0, 0.0)),
        first.surface.first_axis,
        first.surface.second_axis,
    )
    patches = (
        MeshingSurfacePatch(shifted, first.loops, orientation=first.orientation),
        *domain.patches[1:],
    )

    with pytest.raises(TypeError, match="Bare BRepGeometry"):
        MeshingDomain(
            patches,
            domain.curves,
            domain.corner_count,
            source_id=domain.source_id,
            tolerance=10.0,
            source_revision=domain.source_revision,
            regions=domain.regions,
            source_indices=domain.source_indices,
            source_occurrences=domain.source_occurrences,
            region_source_indices=domain.region_source_indices,
            region_source_occurrences=domain.region_source_occurrences,
            source_kinds=domain.source_kinds,
            authority_id=domain.authority_id,
            brep_authority=geometry,
            corner_points=domain.corner_points,
        )


def _brep_model_for_source_leaf(leaf: str, /) -> BRepModel:
    contract = SpatialCoordinateContract.si()
    if leaf == "patch":
        profile = PlanarProfile(ProfilePlane(), ProfileLoop.circle((0.0, 0.0), 1.0))
        model = brep_extrusion(profile, (0.0, 0.0, 2.0), coordinate_contract=contract)
        assert any(isinstance(patch, CylinderPatch) for patch in model.patches)
        return model
    if leaf == "geometry":
        return brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    raise ValueError(f"Unknown BRep source leaf {leaf!r}.")


def _mutate_brep_source_leaf(model: BRepModel, leaf: str, /) -> BRepModel:
    if leaf == "patch":
        patch_index, patch = next(
            (index, patch)
            for index, patch in enumerate(model.patches)
            if isinstance(patch, CylinderPatch)
        )
        return eqx.tree_at(
            lambda value: value.patches[patch_index].origin,
            model,
            patch.origin + jnp.asarray((0.125, 0.0, 0.0)),
        )
    if leaf == "geometry":
        vertex_points = jnp.asarray(model.geometry.vertex_points)
        changed_points = vertex_points.at[0, 0].add(0.125)
        return eqx.tree_at(
            lambda value: value.geometry.vertex_points, model, changed_points
        )
    raise ValueError(f"Unknown BRep source leaf {leaf!r}.")


@pytest.mark.parametrize("leaf", ("patch", "geometry"))
def test_brep_domain_rejects_tree_mutated_stale_authority(leaf: str) -> None:
    model = _brep_model_for_source_leaf(leaf)
    stale_model = _mutate_brep_source_leaf(model, leaf)

    with pytest.raises(ValueError):
        MeshingDomain.from_brep(stale_model)


@pytest.mark.parametrize("leaf", ("patch", "geometry"))
def test_meshing_domain_currentness_rechecks_brep_authority(leaf: str) -> None:
    model = _brep_model_for_source_leaf(leaf)
    domain = MeshingDomain.from_brep(model)
    stale_model = _mutate_brep_source_leaf(model, leaf)
    stale_domain = eqx.tree_at(lambda value: value.brep_authority, domain, stale_model)

    with pytest.raises(ValueError):
        stale_domain.require_current()


@pytest.mark.parametrize("use_kind", ("curve", "pole"))
def test_meshing_domain_currentness_rechecks_trim_pcurve(use_kind: str) -> None:
    model = (
        brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
        if use_kind == "pole"
        else brep_box(
            (0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0),
            coordinate_contract=SpatialCoordinateContract.si(),
        )
    )
    domain = MeshingDomain.from_brep(model)
    use_type = PatchPoleUse if use_kind == "pole" else PatchCurveUse
    patch_index, loop_index, use_index, pcurve = next(
        (
            (patch_index, loop_index, use_index, use.pcurve)
            for patch_index, patch in enumerate(domain.patches)
            for loop_index, loop in enumerate(patch.loops)
            for use_index, use in enumerate(loop)
            if isinstance(use, use_type) and isinstance(use.pcurve, LineCurve)
        )
    )
    stale_domain = eqx.tree_at(
        lambda value: (
            value.patches[patch_index].loops[loop_index][use_index].pcurve.origin
        ),
        domain,
        pcurve.origin + jnp.asarray((0.125, 0.0)),
    )

    with pytest.raises(ValueError):
        stale_domain.require_current()


@pytest.mark.parametrize(
    "mutation",
    (
        "missing-face",
        "reversed-face",
        "absent-solid",
        "wrong-occurrence",
        "missing-region",
    ),
)
def test_brep_domain_rejects_incomplete_or_misoriented_region_boundary(
    mutation: str,
) -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    domain = MeshingDomain.from_brep(model)
    region = domain.regions[0]
    regions = domain.regions
    region_indices = domain.region_source_indices
    region_occurrences = domain.region_source_occurrences
    if mutation == "missing-face":
        regions = (MeshingDomainRegion(region.name, region.boundary[:-1]),)
    elif mutation == "reversed-face":
        first_patch, first_sign = region.boundary[0]
        regions = (
            MeshingDomainRegion(
                region.name,
                ((first_patch, -first_sign), *region.boundary[1:]),
            ),
        )
    elif mutation == "missing-region":
        regions = ()
        region_indices = ()
        region_occurrences = ()
    elif mutation == "wrong-occurrence":
        region_occurrences = (("missing-occurrence",),)
    else:
        region_indices = (model.topology.num_solids,)

    with pytest.raises(ValueError, match="region|solid boundary"):
        _rebuild_brep_domain(
            domain,
            model,
            regions=regions,
            region_source_indices=region_indices,
            region_source_occurrences=region_occurrences,
        )


def test_brep_domain_rejects_source_and_region_keys_outside_occurrence_solids() -> None:
    contract = SpatialCoordinateContract.si()
    first = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    second = brep_box((2.0, 0.0, 0.0), (3.0, 1.0, 1.0), coordinate_contract=contract)
    model = boolean_brep(first, second, "union").model
    assert model.topology.num_solids == 2
    assert all(
        occurrence.placement == BRepAffinePlacement.identity()
        for occurrence in model.geometry.assembly_graph.iter_occurrences()
    )
    domain = MeshingDomain.from_brep(model)
    first_path, second_path = domain.region_source_occurrences
    wrong_region_paths = list(domain.region_source_occurrences)
    wrong_region_paths[0] = second_path
    with pytest.raises(ValueError, match="solid occurrence"):
        _rebuild_brep_domain(
            domain,
            model,
            region_source_occurrences=tuple(wrong_region_paths),
        )

    wrong_region_indices = list(domain.region_source_indices)
    wrong_region_indices[0] = 1
    with pytest.raises(ValueError, match="solid occurrence"):
        _rebuild_brep_domain(
            domain,
            model,
            region_source_indices=tuple(wrong_region_indices),
        )

    for dimension, entity_count in enumerate(
        (
            model.topology.num_vertices,
            model.topology.num_edges,
            model.topology.num_faces,
        )
    ):
        source_row = next(
            row
            for row, path in enumerate(domain.source_occurrences[dimension])
            if path == first_path
        )
        paths = [list(bank) for bank in domain.source_occurrences]
        paths[dimension][source_row] = second_path
        with pytest.raises(ValueError, match="occurrence solid"):
            _rebuild_brep_domain(
                domain,
                model,
                source_occurrences=tuple(tuple(bank) for bank in paths),
            )

        indices = [list(bank) for bank in domain.source_indices]
        indices[dimension][source_row] = entity_count
        with pytest.raises(ValueError, match="absent authoritative BRep entity"):
            _rebuild_brep_domain(
                domain,
                model,
                source_indices=tuple(tuple(bank) for bank in indices),
            )


def test_placed_brep_domain_rejects_binary64_singular_query_placement() -> None:
    box = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    placement = BRepAffinePlacement.from_matrix(
        (
            (Fraction(1, 10**400), 0, 0, 0),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    graph = BRepAssemblyGraph(
        (BRepAssemblyUse("binary64-underflow", placement, solid_index=0),),
        (),
        1,
    )
    model = _rebuild_brep_model(box, assembly_graph=graph)

    with pytest.raises(ValueError, match="binary64 realization must be nonsingular"):
        MeshingDomain.from_brep(model)


def test_brep_exact_vertex_placement_cancels_before_binary64_rounding() -> None:
    box = _rational_parallelepiped()
    source = box.geometry
    exact_point = (Fraction(1, 3), Fraction(1, 3), Fraction(0))
    vertex = source.vertex_exact_points.index(exact_point)
    placement = BRepAffinePlacement.from_matrix(
        (
            (1, 0, 0, Fraction(-1, 3)),
            (0, 1, 0, Fraction(-1, 3)),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    graph = BRepAssemblyGraph(
        (BRepAssemblyUse("cancel-thirds", placement, solid_index=0),),
        (),
        1,
    )
    model = _rebuild_brep_model(box, assembly_graph=graph)
    domain = MeshingDomain.from_brep(model)
    corner = domain.source_indices[0].index(vertex)

    np.testing.assert_array_equal(domain.corner_points[corner], (0.0, 0.0, 0.0))
    assert Fraction(float(source.vertex_points[vertex, 0])) != exact_point[0]


def test_brep_bounded_root_placement_preserves_numeric_source_realization() -> None:
    contract = SpatialCoordinateContract.si()
    first = brep_cylinder(
        1.0,
        4.0,
        base_center=(0.0, 0.0, -2.0),
        coordinate_contract=contract,
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=(-2.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=contract,
    )
    model = boolean_brep(first, second, "intersection").model
    geometry = model.geometry
    root_vertex: int | None = None
    bounds: (
        tuple[
            tuple[Fraction, Fraction, Fraction],
            tuple[Fraction, Fraction, Fraction],
        ]
        | None
    ) = None
    for vertex, (exact, root) in enumerate(
        zip(geometry.vertex_exact_points, geometry.vertex_roots, strict=True)
    ):
        if exact is not None or root is None:
            continue
        lower, upper = geometry.source_vertex_point_bounds(vertex)
        if any(low < high for low, high in zip(lower, upper, strict=True)):
            root_vertex, bounds = vertex, (lower, upper)
            break
    assert root_vertex is not None and bounds is not None
    lower, upper = bounds
    shadow = tuple(
        Fraction.from_float(float(value))
        for value in np.asarray(geometry.vertex_points[root_vertex])
    )
    midpoint = tuple((low + high) / 2 for low, high in zip(lower, upper, strict=True))
    candidate_axes = tuple(
        axis
        for axis, (low, high) in enumerate(zip(lower, upper, strict=True))
        if low < high
    )
    axis = next(
        (
            candidate
            for candidate in candidate_axes
            if midpoint[candidate] != shadow[candidate]
        ),
        candidate_axes[0],
    )
    difference = midpoint[axis] - shadow[axis]
    scale = Fraction(1) / abs(difference) if difference else Fraction(1)
    matrix = [
        [Fraction(1), 0, 0, 0],
        [0, Fraction(1), 0, 0],
        [0, 0, Fraction(1), 0],
        [0, 0, 0, Fraction(1)],
    ]
    matrix[axis][axis] = scale
    matrix[axis][3] = -scale * shadow[axis]
    placement = BRepAffinePlacement.from_matrix(matrix)
    topology = geometry.topology
    solid = next(
        solid_index
        for solid_index, faces in enumerate(topology.solid_faces)
        if any(
            root_vertex in geometry.edge_vertices[edge]
            for face in faces
            for edge in topology.face_edges[face]
        )
    )
    graph = BRepAssemblyGraph(
        (BRepAssemblyUse("bounded-root-placement", placement, solid_index=solid),),
        (),
        len(geometry.solid_shells),
    )
    model = _rebuild_brep_model(model, assembly_graph=graph)
    domain = MeshingDomain.from_brep(model)
    corner = next(
        index
        for index, (source_vertex, path) in enumerate(
            zip(
                domain.source_indices[0],
                domain.source_occurrences[0],
                strict=True,
            )
        )
        if source_vertex == root_vertex and path == ("bounded-root-placement",)
    )
    expected = placement.apply(shadow)

    np.testing.assert_array_equal(
        domain.corner_points[corner],
        tuple(float(value) for value in expected),
    )
    if difference:
        midpoint_image = placement.apply(midpoint)
        assert domain.corner_points[corner, axis] != float(midpoint_image[axis])


@pytest.mark.parametrize("mutation", ("corner", "pcurve"), ids=("corner", "pcurve"))
def test_brep_domain_authority_refuses_changed_source_even_with_tolerance(
    mutation: str,
) -> None:
    model = brep_sphere(
        1.0, coordinate_contract=SpatialCoordinateContract.si(), source_id="sphere-domain"
    )
    domain = MeshingDomain.from_brep(model)
    points = np.array(domain.corner_points, copy=True)
    patches = domain.patches
    if mutation == "corner":
        points[0, 0] += 0.25
    else:
        patch = patches[0]
        loop = list(patch.loops[0])
        index, pole = next(
            (index, use)
            for index, use in enumerate(loop)
            if isinstance(use, PatchPoleUse)
        )
        loop[index] = PatchPoleUse(
            pole.corner,
            LineCurve((0.0, -np.pi / 2.0 + 0.125), (1.0, 0.0)),
            pole.first,
            pole.last,
            sense=pole.sense,
        )
        patches = (
            MeshingSurfacePatch(
                patch.surface,
                (tuple(loop), *patch.loops[1:]),
                orientation=patch.orientation,
            ),
            *patches[1:],
        )

    with pytest.raises(ValueError, match="authoritative|source lift"):
        MeshingDomain(
            patches,
            domain.curves,
            domain.corner_count,
            source_id=domain.source_id,
            source_revision=domain.source_revision,
            regions=domain.regions,
            tolerance=10.0,
            source_indices=domain.source_indices,
            source_occurrences=domain.source_occurrences,
            source_kinds=domain.source_kinds,
            authority_id=model.model_id,
            brep_authority=model,
            corner_points=points,
        )


def test_meshing_domain_spline_spans_cover_crossing_and_touching_knot_charts() -> None:
    knot = Fraction(1, 3)
    quarter = Fraction(1, 4)
    three_quarters = Fraction(3, 4)
    five_twelfths = Fraction(5, 12)
    crossing = (
        (quarter, quarter),
        (five_twelfths, quarter),
        (five_twelfths, three_quarters),
        (quarter, three_quarters),
    )
    touching = (
        (knot, quarter),
        (five_twelfths, quarter),
        (five_twelfths, three_quarters),
        (knot, three_quarters),
    )
    chart_bank = (crossing, touching)
    sample = np.asarray(((1 / 6, 1 / 4), (1 / 3, 2 / 3), (3 / 4, 1 / 2)))
    expected_points = np.stack(
        (
            sample[:, 0] / (1.0 + sample[:, 0]),
            sample[:, 1],
            sample[:, 0] * sample[:, 1] / (1.0 + sample[:, 0]),
        ),
        axis=-1,
    )

    for index, surface in enumerate(_rational_spline_surfaces()):
        spans = _prepare_surface_spans(surface)
        assert spans is not None
        np.testing.assert_allclose(
            np.asarray(surface.evaluate(jnp.asarray(sample))),
            expected_points,
            rtol=0.0,
            atol=2e-14,
        )
        for chart in chart_bank:
            enclosures = _span_enclosures_for_chart(spans, chart)
            intersected_u_spans = {
                enclosure.span_parameter_bounds[0] for enclosure in enclosures
            }
            assert any(lower < knot == upper for lower, upper in intersected_u_spans)
            assert any(lower == knot < upper for lower, upper in intersected_u_spans)
            assert all(enclosure.positive_denominator for enclosure in enclosures)

        domain = _rational_spline_domain(surface, f"rational-source-{index}")
        bounds = domain.interpolation_bounds(0, chart_bank)
        assert bounds.shape == (2,)
        assert np.all(np.isfinite(bounds))
        for bound, chart in zip(bounds, chart_bank, strict=True):
            exact_float_bound = Fraction.from_float(float(bound))
            assert exact_float_bound**2 >= _expected_interpolation_error_squared(chart)

        turns = domain.normal_turn_bounds(0, chart_bank)
        assert turns.shape == (2,)
        assert np.all(np.isfinite(turns))
        # The nearest binary64 pi is below pi; an upper enclosure must round up.
        assert np.all((0.0 <= turns) & (turns <= np.nextafter(np.pi, np.inf)))


def test_meshing_domain_preserves_trim_orientation_and_batched_queries() -> None:
    domain = _square_domain()
    point = domain.curve_atlas.map(jnp.asarray((0,)), jnp.asarray(((0.5,),)))
    normals, regular = domain.oriented_normals(
        np.asarray((0,)), np.asarray(((0.5, 0.5),))
    )

    np.testing.assert_array_equal(np.asarray(point), ((0.5, 0.0, 0.0),))
    np.testing.assert_allclose(normals, ((0.0, 0.0, 1.0),))
    np.testing.assert_array_equal(regular, (True,))


def test_meshing_domain_entity_lookup_requires_occurrence_disambiguation() -> None:
    domain = _square_domain()
    entity = domain.entity_id(1, 0)

    assert entity == source_entity_id(
        "square-source", "r1", "curve", 101, occurrence_path=("instance-A",)
    )
    assert domain.resolve_entity_id(1, entity) == 0
    assert domain.resolve_entity_id(1, entity, occurrence_path=("instance-A",)) == 0
    empty_occurrence = source_entity_id("square-source", "r1", "curve", 101)
    with pytest.raises(ValueError, match="uniquely"):
        domain.resolve_entity_id(1, empty_occurrence)


def test_meshing_domain_boundary_cover_and_normal_turn_are_continuous() -> None:
    domain = _square_domain()
    source = MeshingDomainBoundarySource(domain, (0,))
    cover = source.boundary_chart_cover()
    maximum_radius = 0.25
    samples = source.boundary_cover(maximum_radius)
    distances = source.boundary_distance(
        np.asarray(((0.5, 0.5, 0.0), (0.5, 0.5, 1.0)), dtype=np.float64)
    )
    turn = domain.normal_turn_bounds(
        0,
        np.asarray(
            (((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),),
            dtype=np.float64,
        ),
    )

    assert cover.complete
    assert cover.semantics == "certified"
    assert cover.simplices.shape == (2, 3, 3)
    assert samples.complete
    assert samples.semantics == "certified"
    assert np.all(samples.covering_radius <= maximum_radius)
    assert distances.semantics == "certified"
    assert distances.lower[0] == 0.0
    assert distances.lower[1] <= 1.0 <= distances.upper[1]
    assert 0.0 <= turn[0] < 1.0e-10


def test_sphere_adaptive_interpolation_bound_refines_wide_polar_chart() -> None:
    model = brep_sphere(
        1.0,
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="sphere-interpolation",
    )
    domain = MeshingDomain.from_brep(model)
    # A first-ring cell next to a pole: wide in angle, narrow in profile.
    chart = (
        (Fraction(np.pi), Fraction(-np.pi / 4.0)),
        (Fraction(2.0 * np.pi / 3.0), Fraction(-np.pi / 4.0)),
        (Fraction(2.0 * np.pi / 3.0), Fraction(-3.0 * np.pi / 8.0)),
    )
    corners = np.asarray(chart, dtype=np.float64)
    weights = np.asarray(
        [
            (first, second)
            for first in np.linspace(0.0, 1.0, 81)
            for second in np.linspace(0.0, 1.0, 81)
            if first + second <= 1.0
        ],
        dtype=np.float64,
    )
    barycentric = np.column_stack((1.0 - weights.sum(axis=1), weights))
    points = barycentric @ corners
    patches = np.zeros((points.shape[0],), dtype=np.int64)
    corner_values = domain.evaluate(np.zeros((3,), dtype=np.int64), corners)
    sampled = float(
        np.max(
            np.linalg.norm(
                domain.evaluate(patches, points) - barycentric @ corner_values,
                axis=1,
            )
        )
    )
    box_bound = float(domain.interpolation_bounds(0, corners[None])[0])
    budget = Fraction(1, 4)

    (refined,) = domain.adaptive_interpolation_bounds(0, (chart,), (budget,))
    (unreachable,) = domain.adaptive_interpolation_bounds(
        0, (chart,), (Fraction(sampled) / 2,)
    )

    assert box_bound > 1.0
    assert 0.1 < sampled < 0.2
    assert refined is not None and sampled <= refined <= budget
    assert unreachable is not None and unreachable >= sampled


def test_meshing_domain_scope_banks_preserve_stable_keys_and_occurrences() -> None:
    domain = _square_domain()
    identifiers = np.asarray(domain.scope_entity_ids(1), dtype=np.int64)
    keys = tuple(
        domain.source_entity_key_for_scope(1, int(identifier))
        for identifier in identifiers.tolist()
    )

    assert identifiers.shape == (4,)
    assert np.unique(identifiers).size == identifiers.size
    assert not np.array_equal(identifiers, np.asarray(domain.scope_indices(1)))
    assert not identifiers.flags.writeable
    assert set(keys) == {
        (domain.entity_id(1, slot), domain.source_occurrences[1][slot])
        for slot in range(len(domain.curves))
    }
    assert ("instance-A",) in {path for _, path in keys}
    assert domain.scope_entity_set_id(1)
    assert domain.scope_entity_ids(2).shape == (1,)
    assert domain.scope_entity_ids(0).shape == (4,)
    assert domain.scope_entity_ids(3).shape == (0,)


def test_meshing_domain_prepares_scope_banks_once_per_stratum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="scope-bank-box",
    )
    table = MeshingDomain._source_entity_table
    calls = []

    def counted_table(self: MeshingDomain, dimension: int, /):  # noqa: ANN202
        calls.append(dimension)
        return table(self, dimension)

    monkeypatch.setattr(MeshingDomain, "_source_entity_table", counted_table)
    domain = MeshingDomain.from_brep(model)
    # Bank preparation reads each stratum table a fixed number of times rather
    # than once per entity.
    assert sorted(calls) == [0, 0, 1, 1, 2, 2, 3, 3]
    monkeypatch.undo()

    for dimension, count in enumerate((8, 12, 6, 1)):
        keys = domain.stratum_entity_ids(dimension)
        assert keys == tuple(domain.entity_id(dimension, slot) for slot in range(count))
        assert (
            domain.entity_ids(np.full((count,), dimension), np.arange(count)[::-1])
            == keys[::-1]
        )
        assert set(domain.scope_entity_key_banks[dimension]) == set(
            zip(keys, domain._source_entity_table(dimension)[2], strict=True)
        )
        paths = domain.source_occurrences[dimension] if dimension < 3 else ()
        for slot, key in enumerate(keys):
            assert domain.resolve_entity_id(dimension, key) == slot
            if paths:
                assert (
                    domain.resolve_entity_id(dimension, key, occurrence_path=paths[slot])
                    == slot
                )
    with pytest.raises(ValueError, match="resolve uniquely"):
        domain.resolve_entity_id(2, "absent-source-entity")
    with pytest.raises(ValueError, match="outside its source stratum"):
        domain.entity_ids(np.asarray((2,)), np.asarray((6,)))


def test_planar_revolution_caps_partition_in_their_exact_plane_chart() -> None:
    # A cylinder cap revolves a radial profile line: its authored polar chart is
    # not affine, but the carrier proves it exactly planar, so it has an exact
    # affine plane chart with the authored orientation whose trim is the rim
    # circle; the authored seam and pole are interior to it.
    domain = MeshingDomain.from_brep(
        brep_cylinder(0.5, 1.0, coordinate_contract=SpatialCoordinateContract.si())
    )
    charts = {
        patch: exact_plane_chart(domain, patch) for patch in range(len(domain.patches))
    }
    caps = tuple(patch for patch, chart in charts.items() if chart is not None)
    assert len(caps) == 2
    source = MeshingDomainBoundarySource(domain, caps)
    turn = Fraction.from_float(2.0 * np.pi)
    for patch in caps:
        chart = charts[patch]
        assert chart is not None
        assert chart.chart_id == f"{domain.entity_id(2, patch)}#plane"
        origin, first, second = (
            np.asarray([float(value) for value in row], dtype=np.float64)
            for row in (chart.origin, chart.first_axis, chart.second_axis)
        )
        loops = domain.patches[patch].loops[0]
        rim_profile = next(
            Fraction.from_float(float(use.traversal_endpoints()[0, 1]))
            for use in loops
            if isinstance(use, PatchCurveUse)
            and use.traversal_endpoints()[0, 1] == use.traversal_endpoints()[1, 1]
        )
        pole_profile = 1 - rim_profile
        for angle, profile in ((0.3, 0.2), (2.0, 0.9), (5.5, 0.5)):
            point = np.asarray(
                domain.evaluate(
                    np.asarray((patch,), dtype=np.int64),
                    np.asarray(((angle, profile),), dtype=np.float64),
                ),
                dtype=np.float64,
            )[0]
            s, t = chart.chart_point(
                (Fraction.from_float(angle), Fraction.from_float(profile))
            )
            np.testing.assert_allclose(
                origin + float(s) * first + float(t) * second, point, atol=1.0e-15
            )
            step = 1.0e-6
            derivatives = np.asarray(
                domain.evaluate(
                    np.full((4,), patch, dtype=np.int64),
                    np.asarray(
                        (
                            (angle + step, profile),
                            (angle - step, profile),
                            (angle, profile + step),
                            (angle, profile - step),
                        ),
                        dtype=np.float64,
                    ),
                ),
                dtype=np.float64,
            )
            authored_normal = np.cross(
                derivatives[0] - derivatives[1], derivatives[2] - derivatives[3]
            )
            assert float(authored_normal @ np.cross(first, second)) > 0.0
        assert chart.chart_point((Fraction(0), Fraction(1, 2))) == chart.chart_point(
            (turn, Fraction(1, 2))
        )
        assert chart.chart_point((Fraction(1), pole_profile)) == (
            Fraction(0),
            Fraction(0),
        )
        assert len(chart.loops) == 1 and len(chart.loops[0]) == 1
        rim_circle = chart.loops[0][0].pcurve
        assert isinstance(rim_circle, CircleCurve) and rim_circle.radius_exact == 1
        rim = tuple(
            chart.chart_point((Fraction.from_float(2.0 * np.pi * row / 8), rim_profile))
            for row in range(8)
        )
        assert rim[0] == (Fraction(1), Fraction(0))
        target = []
        for row in range(8):
            triangle = ((Fraction(0), Fraction(0)), rim[row], rim[(row + 1) % 8])
            area = _area2(triangle)
            target.append(
                triangle if area > 0 else (triangle[0], triangle[2], triangle[1])
            )
        declared, lenses = source.trim_partition_reference(
            patch, tuple(target), plane=True
        )
        assert len(lenses) == 8
        assert sum(abs(_area2(triangle)) for triangle in declared) == sum(
            abs(_area2(triangle)) for triangle in target
        )
    assert exact_plane_chart(domain, next(iter(set(range(3)) - set(caps)))) is None


def _area2(triangle: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
    return (triangle[1][0] - triangle[0][0]) * (triangle[2][1] - triangle[0][1]) - (
        triangle[1][1] - triangle[0][1]
    ) * (triangle[2][0] - triangle[0][0])
