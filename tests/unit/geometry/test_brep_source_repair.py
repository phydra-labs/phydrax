from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.discretization._exact_algebraic_geometry import (
    coefficient_interval,
    RadicalElement,
)
from phydrax.geometry._interval_enclosure import prepare_interval_function
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry._rational import fraction_down, fraction_up
from phydrax.geometry.brep._boolean_core import AffinePCurve, PeriodicPCurve
from phydrax.geometry.brep._constructors import (
    assemble_brep_model,
    brep_cone,
    brep_planar_face,
    brep_sphere,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._correspondence import (
    _exact_brep_pole_collapse,
    _exact_vertex_root_source,
    exact_source_lift,
    torus_rooted_circle_pcurve,
)
from phydrax.geometry.brep._intersection import (
    BranchRootEndpoint,
    ConicParameterEndpoint,
    intersect_surface_regions,
    NativePeriodEndpoint,
    TrimRootEndpoint,
)
from phydrax.geometry.brep._intersection_curve import (
    _orthogonal_cylinder_curve_branch_sign,
    IntersectionPCurve,
    orthogonal_cylinder_family,
    orthogonal_cylinder_graph_bounds,
    SurfaceRegion,
)
from phydrax.geometry.brep._model import (
    _certify_edge_face_gap_bounds,
    _validate_degenerate_coedge_poles,
    BRepGeometry,
    BRepModel,
)
from phydrax.geometry.brep._patches import (
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    LineCurve,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    TorusPatch,
)
from phydrax.geometry.brep._projection import prepare_brep_projection
from phydrax.geometry.brep._projection_contracts import (
    BRepEntityDimension,
    BRepProjectionStatus,
)
from phydrax.geometry.brep._root_bindings import (
    BRepConicVertex,
    BRepCurveSurfaceLift,
    BRepVertexRoot,
)
from phydrax.geometry.brep._source_repair import repair_brep_source
from phydrax.geometry.brep._source_repair_carriers import (
    _cylinder_circle_chart,
    _cylinder_line_chart,
    build_target_carriers,
)
from phydrax.geometry.brep._source_repair_topology import target_topology
from phydrax.geometry.brep._trim_helpers import period_offset, period_scalar_bounds
from phydrax.interchange import CadImportPolicy, CadImportResult, read_step
from phydrax.interchange._cad_archive import _register_brep_types, save_brep_archive
from phydrax.meshing._model_codec import write_model_artifact
from tools.meshing_qualification.scenarios.native_cad_profiles import (
    _external_raw,
    _quadric_case,
    _require_geometric_losslessness,
    _roundtrip,
)


@pytest.fixture
def donor_import(tmp_path: Path) -> CadImportResult:
    path = tmp_path / "nist-ftc-11.step"
    path.write_bytes(_external_raw())
    return read_step(
        path,
        CadImportPolicy(SpatialCoordinateContract.si(), pcurve_fit=None),
        trusted_root=tmp_path,
    )


@pytest.fixture
def donor(donor_import: CadImportResult, tmp_path: Path) -> BRepModel:
    model, _ = _roundtrip(donor_import.model, "external-washer", "step", tmp_path)
    return model


@pytest.fixture
def repair_tolerance(donor_import: CadImportResult) -> float:
    source_tolerance = donor_import.source_tolerance
    assert source_tolerance is not None
    assert donor_import.source_tolerances
    return min(
        source_tolerance,
        float(
            cast(
                float,
                _require_geometric_losslessness(donor_import)[
                    "minimum_declared_tolerance"
                ],
            )
        ),
        *(
            value * donor_import.source_length_unit_meters
            for _, value in donor_import.source_tolerances
        ),
    )


def _assert_exact_closed_source(model: BRepModel) -> None:
    model.require_current()
    geometry = model.geometry
    assert geometry.edge_vertex_gap_bounds is None
    _validate_degenerate_coedge_poles(geometry, model.patches)
    gaps = _certify_edge_face_gap_bounds(
        geometry,
        model.patches,
        parameter_bounds=model.parameter_bounds,
        exact_unsupported=True,
        certified_degenerate_coedges=True,
    )
    assert gaps is not None and all(gap == 0.0 for gap in gaps)
    assert geometry.solid_shells
    for shells in geometry.solid_shells:
        for shell in shells:
            uses = defaultdict(list)
            for face, orientation in zip(
                geometry.shell_faces[shell],
                geometry.shell_orientations[shell],
                strict=True,
            ):
                for loop in geometry.face_loops[face]:
                    for coedge in loop:
                        edge = geometry.coedge_edges[coedge]
                        uses[edge].append(
                            int(model.orientation[face])
                            * orientation
                            * geometry.coedge_senses[coedge]
                        )
            assert uses
            assert all(sorted(signs) == [-1, 1] for signs in uses.values())


def test_external_source_repair_preserves_source_and_proves_exact_incidence(
    donor: BRepModel, repair_tolerance: float
) -> None:
    source_revision = donor.source_revision
    source_geometry_id = donor.geometry.geometry_id
    source_points = np.asarray(donor.geometry.vertex_points).copy()
    source_face_gaps = donor.geometry.edge_face_gap_bounds
    source_vertex_gaps = donor.geometry.edge_vertex_gap_bounds
    tolerance = repair_tolerance

    result = repair_brep_source(donor, tolerance=tolerance)
    target = result.model

    assert donor.source_revision == source_revision
    assert donor.geometry.geometry_id == source_geometry_id
    np.testing.assert_array_equal(donor.geometry.vertex_points, source_points)
    assert donor.geometry.edge_face_gap_bounds == source_face_gaps
    assert donor.geometry.edge_vertex_gap_bounds == source_vertex_gaps
    donor.require_current()
    assert result.source_revision == source_revision
    assert result.target_revision == target.source_revision != source_revision
    assert target.source_id == donor.source_id
    assert result.certificate_id
    assert 0.0 < result.max_displacement <= tolerance
    for kind in ("solid", "face", "edge", "vertex"):
        mapping = getattr(result, f"{kind}_map")
        assert set(mapping) == set(getattr(donor, f"{kind}_ids"))
        assert set(mapping.values()) == set(getattr(target, f"{kind}_ids"))
        assert all(a.kind == b.kind == kind for a, b in mapping.items())
        assert all(a.source_revision == source_revision for a in mapping)
        assert all(b.source_revision == target.source_revision for b in mapping.values())
    displacement = np.linalg.norm(
        np.asarray(target.geometry.vertex_points) - source_points, axis=1
    )
    assert np.max(displacement, initial=0.0) <= result.max_displacement + 1e-15
    _assert_exact_closed_source(target)


def test_repaired_donor_conic_arc_endpoints_prove_shared_exact_uv(
    donor: BRepModel, repair_tolerance: float
) -> None:
    target = repair_brep_source(donor, tolerance=repair_tolerance).model
    geometry = target.geometry
    patch = target.patches[3]
    seen_period_pair = False
    seen_native = False
    assert geometry.face_loops[3]
    for loop in geometry.face_loops[3]:
        assert len(loop) == 2
        for position, coedge in enumerate(loop):
            following = loop[(position + 1) % len(loop)]
            edge, next_edge = (
                geometry.coedge_edges[coedge],
                geometry.coedge_edges[following],
            )
            sense, next_sense = (
                geometry.coedge_senses[coedge],
                geometry.coedge_senses[following],
            )
            end_slot = 1 if sense > 0 else 0
            next_slot = 0 if next_sense > 0 else 1
            vertex = geometry.edge_vertices[edge][end_slot]
            assert vertex == geometry.edge_vertices[next_edge][next_slot]
            root = geometry.vertex_roots[vertex]
            assert root is not None
            first = geometry.coedge_endpoint_roots[coedge][end_slot]
            second = geometry.coedge_endpoint_roots[following][next_slot]
            first_curve, second_curve = (
                geometry.pcurves[coedge],
                geometry.pcurves[following],
            )
            assert isinstance(first_curve, PlaneCylinderSectionPCurve)
            assert isinstance(second_curve, PlaneCylinderSectionPCurve)
            assert isinstance(first, ConicParameterEndpoint)
            assert isinstance(second, ConicParameterEndpoint)
            assert root.same_uv_endpoint(patch, first_curve, first, second_curve, second)
            for curve, endpoint in ((first_curve, first), (second_curve, second)):
                box = root.source_endpoint_uv_box(patch, curve, endpoint)
                exact = root._exact_source_endpoint_uv(patch, curve, endpoint)
                assert box is not None and exact is not None and exact[0] == "plane"
                for axis, coordinate in enumerate(exact[1:]):
                    assert isinstance(coordinate, (Fraction, RadicalElement))
                    lower, upper = coefficient_interval(coordinate, bits=128)
                    assert (
                        Fraction(float(box[0, axis]))
                        <= lower
                        <= upper
                        <= Fraction(float(box[1, axis]))
                    )
            if (
                first.root.cosine_exact == second.root.cosine_exact
                and first.root.sine_exact == second.root.sine_exact
                and {first.root.turns, second.root.turns} == {0, 1}
            ):
                seen_period_pair = True
            for curve, endpoint in ((first_curve, first), (second_curve, second)):
                if (
                    endpoint.root.cosine_exact == 1
                    and endpoint.root.sine_exact == 0
                    and endpoint.root.turns == 0
                    and curve.native_period_symbol() == "two_pi"
                    and curve.angle_offset_root is None
                ):
                    native = NativePeriodEndpoint(curve, turns=0)
                    assert root.same_uv_endpoint(patch, curve, endpoint, curve, native)
                    assert root.source_endpoint_uv_box(patch, curve, native) is not None
                    seen_native = True
            wrong_curve = LineCurve((0.0, 0.0), (1.0, 0.0))
            assert not root.same_uv_endpoint(
                patch, first_curve, first, wrong_curve, second
            )
            assert not root.same_uv_endpoint(
                target.patches[4], first_curve, first, second_curve, second
            )
            other = geometry.coedge_endpoint_roots[coedge][1 - end_slot]
            assert isinstance(other, ConicParameterEndpoint)
            assert not root.same_uv_endpoint(
                patch, first_curve, first, first_curve, other
            )
    assert seen_period_pair
    assert seen_native


def test_iges_quadric_shared_branch_and_trim_uv_roots(tmp_path: Path) -> None:
    model, _, _, _ = _quadric_case("iges", tmp_path)
    domain = MeshingDomain.from_brep(model)
    face = int(domain.source_indices[2][2])
    patch = model.patches[face]
    geometry = model.geometry
    loop = geometry.face_loops[face][0]
    for coedge, following in ((2, 3), (5, 2)):
        assert loop[(loop.index(coedge) + 1) % len(loop)] == following
        edge, next_edge = geometry.coedge_edges[coedge], geometry.coedge_edges[following]
        end_slot = 1 if geometry.coedge_senses[coedge] > 0 else 0
        next_slot = 0 if geometry.coedge_senses[following] > 0 else 1
        vertex = geometry.edge_vertices[edge][end_slot]
        assert vertex == geometry.edge_vertices[next_edge][next_slot]
        root = geometry.vertex_roots[vertex]
        assert root is not None
        first = geometry.coedge_endpoint_roots[coedge][end_slot]
        second = geometry.coedge_endpoint_roots[following][next_slot]
        assert {type(first), type(second)} == {BranchRootEndpoint, TrimRootEndpoint}
        first_curve, second_curve = geometry.pcurves[coedge], geometry.pcurves[following]
        assert root.same_uv_endpoint(patch, first_curve, first, second_curve, second)
        first_box = root.source_endpoint_uv_box(patch, first_curve, first)
        second_box = root.source_endpoint_uv_box(patch, second_curve, second)
        assert first_box is not None and second_box is not None
        assert np.all(first_box[0] <= second_box[1])
        assert np.all(second_box[0] <= first_box[1])
        unrelated = LineCurve((100.0, 100.0), (1.0, 0.0))
        assert not root.same_uv_endpoint(patch, first_curve, first, unrelated, second)
        assert root.source_endpoint_uv_box(patch, unrelated, second) is None


def test_repaired_donor_projection_distinguishes_containment_and_uncertainty(
    donor: BRepModel, repair_tolerance: float
) -> None:
    target = repair_brep_source(donor, tolerance=repair_tolerance).model
    projection = prepare_brep_projection(target)
    outside = np.asarray((-0.04, -0.00010321, -0.00086266))
    inside = np.asarray((0.023, 0.0, -0.00075))
    locations = projection.locate_solids(np.stack((outside, inside)))
    assert np.asarray(locations.status).tolist() == [
        BRepProjectionStatus.FAILED,
        BRepProjectionStatus.UNIQUE,
    ]

    # A certified separation does not turn an unresolved closest point into a
    # unique projection. These four donor faces have nonzero separation here.
    face_indices = np.asarray((1, 3, 4, 5), dtype=np.int64)
    faces = projection.project(
        np.tile(outside, (len(face_indices), 1)),
        np.full(len(face_indices), int(BRepEntityDimension.FACE), dtype=np.int64),
        face_indices,
    )
    assert np.all(np.asarray(faces.distance_lower_bounds) > 0.0)
    assert np.asarray(faces.status).tolist() == [BRepProjectionStatus.UNRESOLVED] * 4

    vertices = np.asarray(target.geometry.vertex_points)
    ring = np.asarray((0.016, 0.0, 0.0))
    boundary = vertices[np.argmin(np.linalg.norm(vertices - ring, axis=1))]
    boundary_result = projection.locate_solids(boundary[None, :])
    assert int(np.asarray(boundary_result.status)[0]) in (
        BRepProjectionStatus.AMBIGUOUS,
        BRepProjectionStatus.UNRESOLVED,
    )

    sheet = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    with pytest.raises(ValueError, match="no solids"):
        prepare_brep_projection(sheet).locate_solids(inside[None, :])


def test_repair_refuses_insufficient_displacement_budget(donor: BRepModel) -> None:
    revision = donor.source_revision
    with pytest.raises(ValueError):
        repair_brep_source(donor, tolerance=1e-30)
    assert donor.source_revision == revision
    donor.require_current()


def test_repair_refuses_ambiguous_coincident_face_candidates() -> None:
    contract = SpatialCoordinateContract.si()
    face = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
    )
    g = face.geometry
    vertices = g.topology.num_vertices
    edges = g.topology.num_edges
    coedges = len(g.coedge_edges)
    curves = len(g.curves)
    # Three distinct coincident sheets give every edge two possible partners;
    # source identity does not authorize either proximity-based weld.
    geometry = BRepGeometry(
        vertex_points=np.tile(np.asarray(g.vertex_points), (3, 1)),
        vertex_exact_points=g.vertex_exact_points * 3,
        curves=g.curves * 3,
        edge_curves=tuple(
            index + copy * curves for copy in range(3) for index in g.edge_curves
        ),
        edge_ranges=np.tile(np.asarray(g.edge_ranges), (3, 1)),
        edge_vertices=tuple(
            (a + copy * vertices, b + copy * vertices)
            for copy in range(3)
            for a, b in g.edge_vertices
        ),
        pcurves=g.pcurves * 3,
        coedge_edges=tuple(
            index + copy * edges for copy in range(3) for index in g.coedge_edges
        ),
        coedge_senses=g.coedge_senses * 3,
        face_loops=tuple(
            tuple(
                tuple(index + copy * coedges for index in loop)
                for loop in g.face_loops[0]
            )
            for copy in range(3)
        ),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    ambiguous = assemble_brep_model(
        geometry,
        face.patches * 3,
        np.tile(np.asarray(face.parameter_bounds), (3, 1, 1)),
        np.tile(np.asarray(face.orientation), 3),
        ("first", "second", "third"),
        coordinate_contract=contract,
        source_id="ambiguous-coincident-sheets",
        trim_domains=face.trim_domains * 3,
    )
    revision = ambiguous.source_revision
    with pytest.raises(ValueError):
        repair_brep_source(ambiguous, tolerance=1e-6)
    assert ambiguous.source_revision == revision
    ambiguous.require_current()


@pytest.mark.parametrize("phase_quarter", (0, 1), ids=("zero-phase", "quarter-turn"))
def test_cylinder_native_period_endpoint_keeps_physical_phase_and_sheet(
    phase_quarter: int,
) -> None:
    cylinder = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    unit = CircleCurve((0, 0), (1, 0), (0, 1), 1.0)
    phase = ConicParameterRoot(unit, Fraction(0), Fraction(1)) if phase_quarter else None
    first_axis = (0, 1, 0) if phase_quarter else (1, 0, 0)
    second_axis = (-1, 0, 0) if phase_quarter else (0, 1, 0)
    edge = CircleCurve((0, 0, 0), first_axis, second_axis, 1.0)
    edge_root = ConicParameterRoot(edge, Fraction(1), Fraction(0))
    coordinates = edge_root.exact_point()
    assert len(coordinates) == 3
    point = (coordinates[0], coordinates[1], coordinates[2])
    vertex = BRepVertexRoot(BRepConicVertex(edge_root))
    pcurve = PlaneCylinderSectionPCurve(
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0)),
        (Fraction(0), Fraction(0)),
        parameter_range=(0.0, float(2 * np.pi)),
        angle_offset_root=phase,
    )
    native_zero = NativePeriodEndpoint(pcurve, cylinder, 0, turns=0)
    native_full = NativePeriodEndpoint(pcurve, cylinder, 0, turns=1)
    conic = ConicParameterEndpoint(edge_root, pcurve, None, point)
    assert vertex.same_uv_endpoint(cylinder, pcurve, native_zero, pcurve, conic)
    assert not vertex.same_uv_endpoint(cylinder, pcurve, native_zero, pcurve, native_full)
    for endpoint, turns in ((native_zero, 0), (native_full, 1)):
        box = vertex.source_endpoint_uv_box(cylinder, pcurve, endpoint)
        assert box is not None
        angle = ConicParameterRoot(
            unit,
            Fraction(0 if phase_quarter else 1),
            Fraction(1 if phase_quarter else 0),
            turns=turns,
        )
        lower, upper = angle.parameter_enclosure()
        assert box[0, 0] <= lower <= upper <= box[1, 0]
        assert box[0, 1] <= 0 <= box[1, 1]


def test_repair_topology_preserves_nonzero_cylinder_circle_phase(
    donor: BRepModel,
) -> None:
    carriers = build_target_carriers(donor)
    geometry = donor.geometry
    original = target_topology(donor, carriers.curves, carriers.patches, carriers.pcurves)
    coedge_faces = {
        coedge: face
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
    }
    coedge = next(
        coedge
        for coedge, edge in enumerate(geometry.coedge_edges)
        if isinstance(carriers.patches[coedge_faces[coedge]], CylinderPatch)
        and isinstance(carriers.curves[edge], CircleCurve)
        and not isinstance(carriers.pcurves[coedge], PeriodicPCurve)
        and geometry.edge_vertices[edge][0] == geometry.edge_vertices[edge][1]
        and all(
            isinstance(value, NativePeriodEndpoint)
            for value in original.coedge_endpoint_roots[coedge]
        )
    )
    face = coedge_faces[coedge]
    old = carriers.patches[face]
    assert isinstance(old, CylinderPatch)
    rotated = CylinderPatch(
        np.asarray(old.origin),
        np.asarray(old.second_axis),
        -np.asarray(old.first_axis),
        np.asarray(old.axis),
        np.asarray(old.radius),
        origin_exact=old.origin_exact,
        first_axis_exact=old.second_axis_exact,
        second_axis_exact=tuple(-x for x in old.first_axis_exact),
        axis_exact=old.axis_exact,
        radius_exact=old.radius_exact,
    )
    ring = carriers.curves[geometry.coedge_edges[coedge]]
    assert isinstance(ring, CircleCurve)
    if _cylinder_circle_chart(ring, rotated, (-10.0, 10.0)).angle_offset_root is None:
        rotated = CylinderPatch(
            np.asarray(old.origin),
            -np.asarray(old.second_axis),
            np.asarray(old.first_axis),
            np.asarray(old.axis),
            np.asarray(old.radius),
            origin_exact=old.origin_exact,
            first_axis_exact=tuple(-x for x in old.second_axis_exact),
            second_axis_exact=old.first_axis_exact,
            axis_exact=old.axis_exact,
            radius_exact=old.radius_exact,
        )
    patches = list(carriers.patches)
    patches[face] = rotated
    pcurves = list(carriers.pcurves)
    for use, owner in coedge_faces.items():
        if owner != face:
            continue
        source = carriers.pcurves[use]
        shifts = (0, 0)
        if isinstance(source, PeriodicPCurve):
            shifts = source.period_shifts
        edge_curve = carriers.curves[geometry.coedge_edges[use]]
        if isinstance(edge_curve, CircleCurve):
            chart = _cylinder_circle_chart(edge_curve, rotated, (-10.0, 10.0))
        else:
            assert isinstance(edge_curve, LineCurve)
            chart = _cylinder_line_chart(edge_curve, rotated, (-10.0, 10.0))
        pcurves[use] = PeriodicPCurve(chart, rotated, shifts) if any(shifts) else chart
    chart = pcurves[coedge]
    assert isinstance(chart, PlaneCylinderSectionPCurve)
    assert chart.angle_offset_root is not None
    topology = target_topology(donor, carriers.curves, tuple(patches), tuple(pcurves))
    vertex = geometry.edge_vertices[geometry.coedge_edges[coedge]][0]
    binding = topology.vertex_roots[vertex]
    assert binding is not None
    for turns, endpoint in enumerate(topology.coedge_endpoint_roots[coedge]):
        assert isinstance(endpoint, NativePeriodEndpoint)
        assert endpoint.rational == 0 and endpoint.turns == turns
        box = binding.source_endpoint_uv_box(rotated, chart, endpoint)
        assert box is not None
        phase = chart.angle_offset_root
        angle = ConicParameterRoot(
            phase.conic, phase.cosine_exact, phase.sine_exact, turns=turns
        )
        lower, upper = angle.parameter_enclosure()
        assert box[0, 0] <= lower <= upper <= box[1, 0]
    seam_matches = []
    for use, owner in coedge_faces.items():
        edge = geometry.coedge_edges[use]
        if owner != face or not isinstance(carriers.curves[edge], LineCurve):
            continue
        for slot, endpoint_vertex in enumerate(geometry.edge_vertices[edge]):
            if endpoint_vertex != vertex:
                continue
            seam_endpoint = topology.coedge_endpoint_roots[use][slot]
            for ring_endpoint in topology.coedge_endpoint_roots[coedge]:
                seam_matches.append(
                    binding.same_uv_endpoint(
                        rotated, chart, ring_endpoint, pcurves[use], seam_endpoint
                    )
                )
    assert any(seam_matches)


def _assert_zero_phase_cylinder_trim_seam(model: BRepModel) -> None:
    geometry = model.geometry
    count = 0
    for face, loops in enumerate(geometry.face_loops):
        patch = model.patches[face]
        if not isinstance(patch, CylinderPatch):
            continue
        for loop in loops:
            for position, current in enumerate(loop):
                following = loop[(position + 1) % len(loop)]
                edge, next_edge = (
                    geometry.coedge_edges[current],
                    geometry.coedge_edges[following],
                )
                end = 1 if geometry.coedge_senses[current] > 0 else 0
                start = 0 if geometry.coedge_senses[following] > 0 else 1
                vertex = geometry.edge_vertices[edge][end]
                assert vertex == geometry.edge_vertices[next_edge][start]
                root = geometry.vertex_roots[vertex]
                endpoints = (
                    geometry.coedge_endpoint_roots[current][end],
                    geometry.coedge_endpoint_roots[following][start],
                )
                curves = (geometry.pcurves[current], geometry.pcurves[following])
                for index in (0, 1):
                    ring, native = curves[index], endpoints[index]
                    other, opposite = curves[1 - index], endpoints[1 - index]
                    if not (
                        isinstance(ring, PlaneCylinderSectionPCurve)
                        and ring.angle_offset_root is None
                        and isinstance(native, NativePeriodEndpoint)
                        and isinstance(opposite, ConicParameterEndpoint)
                    ):
                        continue
                    assert root is not None
                    assert root.same_uv_endpoint(patch, ring, native, other, opposite)
                    first_box = root.source_endpoint_uv_box(patch, ring, native)
                    second_box = root.source_endpoint_uv_box(patch, other, opposite)
                    assert first_box is not None and second_box is not None
                    assert np.all(first_box[0] <= second_box[1])
                    assert np.all(second_box[0] <= first_box[1])
                    count += 1
    assert count > 0


def test_zero_phase_cylinder_trim_seam_survives_cold_restore(
    donor: BRepModel, repair_tolerance: float, tmp_path: Path
) -> None:
    target = repair_brep_source(donor, tolerance=repair_tolerance).model
    _assert_zero_phase_cylinder_trim_seam(target)
    path = tmp_path / "zero-phase-cylinder.phx"
    save_brep_archive(target, path, ingress_limits=None)
    code = """
import sys
from phydrax.interchange._cad_archive import load_brep_archive
from tests.unit.geometry.test_brep_source_repair import _assert_zero_phase_cylinder_trim_seam
model = load_brep_archive(sys.argv[1], ingress_limits=None)
model.require_current()
_assert_zero_phase_cylinder_trim_seam(model)
"""
    subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("kind", ("sphere", "cone"))
def test_native_revolution_poles_publish_source_owned_carriers(kind: str) -> None:
    contract = SpatialCoordinateContract.si()
    model = (
        brep_sphere(1.0, coordinate_contract=contract)
        if kind == "sphere"
        else brep_cone(1.0, 0.0, 2.0, coordinate_contract=contract)
    )
    model.require_current()
    geometry = model.geometry
    poles = [
        (face, coedge)
        for face, loops in enumerate(geometry.face_loops)
        for loop in loops
        for coedge in loop
        if geometry.edge_curves[geometry.coedge_edges[coedge]] < 0
    ]
    assert len(poles) == 2
    for face, coedge in poles:
        edge = geometry.coedge_edges[coedge]
        vertex = geometry.edge_vertices[edge][0]
        assert geometry.edge_vertices[edge][1] == vertex
        patch = model.patches[face]
        pcurve = geometry.pcurves[coedge]
        root = geometry.vertex_roots[vertex]
        first, last = (float(value) for value in np.asarray(geometry.edge_ranges[edge]))
        endpoints = geometry.coedge_endpoint_roots[coedge]
        assert _exact_brep_pole_collapse(
            None,
            patch,
            pcurve,
            first,
            last,
            geometry.vertex_exact_points[vertex],
            root,
            endpoints,
        )
        if kind == "sphere":
            assert isinstance(pcurve, AffinePCurve)
            assert isinstance(pcurve.curve, ConicRootLinePCurve)
            assert root is not None
            point, profile_root = _exact_vertex_root_source(root)
            assert point is not None and profile_root is not None
            assert pcurve.curve.root.root_id == profile_root.root_id
            assert all(
                isinstance(endpoint, NativePeriodEndpoint) for endpoint in endpoints
            )
            assert all(
                root.source_endpoint_uv_box(patch, pcurve, endpoint) is not None
                for endpoint in endpoints
            )
            shadow = LineCurve((0.0, profile_root.parameter), (1.0, 0.0))
            assert not _exact_brep_pole_collapse(
                None,
                patch,
                shadow,
                first,
                last,
                geometry.vertex_exact_points[vertex],
                root,
                endpoints,
            )
        else:
            assert isinstance(pcurve, LineCurve)
            exact_point = geometry.vertex_exact_points[vertex]
            assert exact_point is not None
            wrong = (exact_point[0] + 1, exact_point[1], exact_point[2])
            assert not _exact_brep_pole_collapse(
                None, patch, pcurve, first, last, wrong, root, endpoints
            )
    MeshingDomain.from_brep(model).require_current()


def test_revolution_pole_uv_requires_exact_rooted_carrier() -> None:
    model = brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    patch = model.patches[0]
    assert isinstance(patch, RevolutionSurface)
    poles = tuple(root for root in model.geometry.vertex_roots if root is not None)
    assert len(poles) == 2
    for vertex in poles:
        point, profile_root = _exact_vertex_root_source(vertex)
        assert point is not None and profile_root is not None
        meridian = LineCurve((0.0, 0.0), (0.0, 1.0))
        meridian_endpoint = ConicParameterEndpoint(profile_root, meridian, None, point)
        rooted_line = ConicRootLinePCurve(
            profile_root, 0, 1, parameter_range=(0.0, float(2 * np.pi))
        )
        pole_curve = AffinePCurve(rooted_line, ((0, 1), (1, 0)), (0, 0))
        native_zero = NativePeriodEndpoint(pole_curve, patch, 0, turns=0)
        native_full = NativePeriodEndpoint(pole_curve, patch, 0, turns=1)
        assert vertex.same_uv_endpoint(
            patch, pole_curve, native_zero, meridian, meridian_endpoint
        )
        for endpoint, turns in ((native_zero, 0), (native_full, 1)):
            box = vertex.source_endpoint_uv_box(patch, pole_curve, endpoint)
            assert box is not None
            longitude = period_scalar_bounds(period_offset(Fraction(0), Fraction(turns)))
            latitude = profile_root.parameter_enclosure()
            assert box[0, 0] <= longitude[0] <= longitude[1] <= box[1, 0]
            assert box[0, 1] <= latitude[0] <= latitude[1] <= box[1, 1]
        assert not vertex.same_uv_endpoint(
            patch, pole_curve, native_full, meridian, meridian_endpoint
        )
        shifted = PeriodicPCurve(meridian, patch, (1, 0))
        shifted_endpoint = ConicParameterEndpoint(profile_root, shifted, None, point)
        assert vertex.same_uv_endpoint(
            patch, pole_curve, native_full, shifted, shifted_endpoint
        )

        # A binary shadow of an irrational latitude is not the exact root.
        shadow = LineCurve((0.0, profile_root.parameter), (1.0, 0.0))
        shadow_endpoint = NativePeriodEndpoint(shadow, patch, 0, turns=0)
        assert vertex.source_endpoint_uv_box(patch, shadow, shadow_endpoint) is None
        assert not vertex.same_uv_endpoint(
            patch, shadow, shadow_endpoint, meridian, meridian_endpoint
        )
        wrong_shadow = LineCurve((0.0, profile_root.parameter + 0.125), (1.0, 0.0))
        wrong_endpoint = NativePeriodEndpoint(wrong_shadow, patch, 0, turns=0)
        assert vertex.source_endpoint_uv_box(patch, wrong_shadow, wrong_endpoint) is None
        unrelated_patch = RevolutionSurface(patch.curve, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        assert not vertex.same_uv_endpoint(
            unrelated_patch, pole_curve, native_zero, meridian, meridian_endpoint
        )

        assert isinstance(patch.curve, CircleCurve)
        off_axis = ConicParameterRoot(patch.curve, Fraction(1), Fraction(0))
        if off_axis.exact_point()[0] == 0 and off_axis.exact_point()[1] == 0:
            off_axis = ConicParameterRoot(patch.curve, Fraction(0), Fraction(1))
        coordinates = off_axis.exact_point()
        assert len(coordinates) == 3
        off_point = (coordinates[0], coordinates[1], coordinates[2])
        assert off_point[0] != 0 or off_point[1] != 0
        off_vertex = BRepVertexRoot(BRepConicVertex(off_axis))
        off_meridian = ConicParameterEndpoint(off_axis, meridian, None, off_point)
        assert not off_vertex.same_uv_endpoint(
            patch, pole_curve, native_zero, meridian, off_meridian
        )


@pytest.mark.parametrize("orientation", (1, -1), ids=("forward", "reverse"))
def test_rooted_torus_quarter_turn_lift_and_cold_restore(
    orientation: int, tmp_path: Path
) -> None:
    torus = TorusPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        3.0,
        1.0,
    )
    meridian = CircleCurve((3.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), 1.0)
    root = ConicParameterRoot(meridian, Fraction(0), Fraction(1))
    curve = CircleCurve(
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
        (-float(orientation), 0.0, 0.0),
        3.0,
    )
    pcurve = torus_rooted_circle_pcurve(root, Fraction(1, 4), orientation)
    first, last = 0.0, float(2 * np.pi)
    region = SurfaceRegion(torus, np.asarray(((0.0, 0.0), (last, last))))
    assert exact_source_lift(curve, pcurve, region, first, last)
    wrong_phase = torus_rooted_circle_pcurve(root, Fraction(0), orientation)
    wrong_direction = torus_rooted_circle_pcurve(root, Fraction(1, 4), -orientation)
    for wrong in (wrong_phase, wrong_direction):
        assert not exact_source_lift(curve, wrong, region, first, last)
        with pytest.raises(ValueError):
            BRepCurveSurfaceLift(curve, torus, wrong, first, last, source_region=region)

    lift = BRepCurveSurfaceLift(curve, torus, pcurve, first, last, source_region=region)
    path = tmp_path / "rooted-torus-lift.phx"
    _register_brep_types()
    write_model_artifact(path, lift, limits=None)
    code = """
import json, sys
from phydrax.geometry.brep._correspondence import exact_source_lift
from phydrax.geometry.brep._intersection_curve import SurfaceRegion
from phydrax.geometry.brep._root_bindings import BRepCurveSurfaceLift
from phydrax.interchange._cad_archive import _register_brep_types, _reconstruct_brep_graph
from phydrax.meshing._model_codec import read_model_artifact
_register_brep_types()
lift = read_model_artifact(sys.argv[1], expected_type=BRepCurveSurfaceLift,
                           restore=_reconstruct_brep_graph, limits=None)
region = SurfaceRegion(lift.surface, lift.parameter_box)
assert exact_source_lift(lift.curve, lift.pcurve, region, lift.first, lift.last)
print(json.dumps({'root_id': lift.root_id, 'turns': str(lift.pcurve.turns),
                  'orientation': lift.pcurve.orientation,
                  'conic_root': lift.pcurve.root_line.root.root_id}))
"""
    restored = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(restored.stdout) == {
        "root_id": lift.root_id,
        "turns": "1/4",
        "orientation": orientation,
        "conic_root": root.root_id,
    }


@pytest.mark.parametrize("mixed", (False, True), ids=("cylinders", "revolved-small"))
@pytest.mark.parametrize("swap", (False, True))
def test_intersection_chart_source_bounds_enclose_exact_carriers(
    mixed: bool, swap: bool
) -> None:
    large_radius, small_radius = Fraction(4, 3), Fraction(1, 3)
    large = CylinderPatch(
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        float(large_radius),
        radius_exact=large_radius,
    )
    if mixed:
        profile = LineCurve(
            (-2.0, float(small_radius), 0.0),
            (4.0, 0.0, 0.0),
            origin_exact=(Fraction(-2), small_radius, Fraction(0)),
        )
        small = RevolutionSurface(profile, (0, 0, 0), (1, 0, 0))
        small_box = ((0.0, 0.0), (float(2 * np.pi), 1.0))
    else:
        small = CylinderPatch(
            (0, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 0),
            float(small_radius),
            radius_exact=small_radius,
        )
        small_box = ((0.0, -2.0), (float(2 * np.pi), 2.0))
    large_region = SurfaceRegion(large, np.asarray(((0.0, -2.0), (2 * np.pi, 2.0))))
    small_region = SurfaceRegion(small, np.asarray(small_box))
    first, second = (small_region, large_region) if swap else (large_region, small_region)
    result = intersect_surface_regions(first, second)
    assert result.complete and len(result.curves) == 2
    period_lower, period_upper = period_scalar_bounds(
        period_offset(Fraction(0), Fraction(1))
    )
    assert period_lower <= 2 * np.pi <= period_upper
    for curve in result.curves:
        family = orthogonal_cylinder_family(curve.first, curve.second)
        assert family is not None
        sign = _orthogonal_cylinder_curve_branch_sign(curve, family)
        assert sign in (-1, 1)
        theta_axis = 2 * family.small_index
        theta_start = Fraction.from_float(
            float(np.asarray(curve.chart_start)[0, theta_axis])
        )
        theta_end = Fraction.from_float(float(np.asarray(curve.chart_end)[0, theta_axis]))
        theta = theta_start + Fraction(3, 8) * (theta_end - theta_start)
        reference_angle = float(np.asarray(curve.chart_start)[0, 2 * family.large_index])
        reference = orthogonal_cylinder_graph_bounds(
            family, fraction_down(theta), fraction_up(theta), sign, reference_angle
        )
        assert reference is not None
        reference_lower, reference_upper = reference
        for side, offset in (("first", 0), ("second", 2)):
            for reversed_source in (False, True):
                pcurve = IntersectionPCurve(
                    curve, side, first=0.0, last=1.0, reversed=reversed_source
                )
                evaluate, lower, upper = pcurve._chart_evaluator_with_source_bounds(0)
                for index, exact in ((2, large_radius), (3, small_radius)):
                    assert (
                        Fraction(float(lower[index]))
                        <= exact
                        <= Fraction(float(upper[index]))
                    )
                    assert lower[index] < upper[index]
                parameter = 0.625 if reversed_source else 0.375
                interval = prepare_interval_function(
                    lambda x: evaluate(x[0], x[1], x[2:]),
                    2 + len(lower),
                    execution="host",
                )
                box_lower = np.concatenate(
                    (
                        np.asarray((parameter, float(period_lower)), dtype=np.float64),
                        lower,
                    )
                )[None, :]
                box_upper = np.concatenate(
                    (
                        np.asarray((parameter, float(period_upper)), dtype=np.float64),
                        upper,
                    )
                )[None, :]
                actual_lower, actual_upper = interval.evaluate(box_lower, box_upper)
                assert np.all(np.isfinite(actual_lower)) and np.all(
                    np.isfinite(actual_upper)
                )
                assert np.all(actual_lower[0] <= reference_upper[offset : offset + 2])
                assert np.all(actual_upper[0] >= reference_lower[offset : offset + 2])
                reference_midpoint = (
                    reference_lower[offset : offset + 2]
                    + reference_upper[offset : offset + 2]
                ) / 2
                assert np.all(actual_lower[0] <= reference_midpoint)
                assert np.all(reference_midpoint <= actual_upper[0])
                numeric = np.asarray(
                    evaluate(
                        jnp.asarray(parameter), jnp.asarray(2 * np.pi), jnp.asarray(lower)
                    )
                )
                assert np.all(actual_lower[0] <= numeric) and np.all(
                    numeric <= actual_upper[0]
                )
        with pytest.raises(ValueError):
            IntersectionPCurve(curve, "first", first=-1.0, last=1.0)


def test_bounded_chart_refuses_non_cylinder_family() -> None:
    plane = SurfaceRegion(
        PlanePatch((0, 0, 0.5), (1, 0, 0), (0, 1, 0)),
        np.asarray(((-2.0, -2.0), (2.0, 2.0))),
    )
    sphere = SurfaceRegion(
        SpherePatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0),
        np.asarray(((0.0, -np.pi / 2), (2 * np.pi, np.pi / 2))),
    )
    result = intersect_surface_regions(plane, sphere)
    assert result.complete and len(result.curves) == 1
    with pytest.raises(NotImplementedError, match="orthogonal cylinder"):
        result.curves[0].p_curve("first")._chart_evaluator_with_source_bounds(0)


def test_repair_source_and_certificate_replay_in_cold_process(
    donor: BRepModel, repair_tolerance: float, tmp_path: Path
) -> None:
    tolerance = repair_tolerance
    result = repair_brep_source(donor, tolerance=tolerance)
    source_path = tmp_path / "source.phx"
    target_path = tmp_path / "target.phx"
    save_brep_archive(donor, source_path, ingress_limits=None)
    save_brep_archive(result.model, target_path, ingress_limits=None)
    expected = {
        "source": result.source_revision,
        "target": result.target_revision,
        "certificate": result.certificate_id,
        "displacement": result.max_displacement,
        "maps": {
            kind: sorted(
                (a.index, b.index) for a, b in getattr(result, f"{kind}_map").items()
            )
            for kind in ("solid", "face", "edge", "vertex")
        },
    }
    code = """
import json, sys
from phydrax.geometry.brep._source_repair import repair_brep_source
from phydrax.interchange._cad_archive import load_brep_archive
source = load_brep_archive(sys.argv[1], ingress_limits=None)
target = load_brep_archive(sys.argv[2], ingress_limits=None)
source.require_current()
target.require_current()
result = repair_brep_source(source, tolerance=float(sys.argv[3]))
assert result.model.source_revision == target.source_revision
print(json.dumps({
    'source': result.source_revision,
    'target': result.target_revision,
    'certificate': result.certificate_id,
    'displacement': result.max_displacement,
    'maps': {kind: sorted((a.index, b.index) for a, b in getattr(result, kind + '_map').items())
             for kind in ('solid', 'face', 'edge', 'vertex')},
}, sort_keys=True))
"""
    restored = subprocess.run(
        [sys.executable, "-c", code, str(source_path), str(target_path), str(tolerance)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(restored.stdout) == json.loads(json.dumps(expected))
