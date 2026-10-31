from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._cad_revision import AssociationStatus
from phydrax.geometry.brep import brep_box, brep_cylinder, brep_sphere
from phydrax.geometry.brep._boolean import boolean_brep
from phydrax.geometry.brep._boolean_core import (
    AffinePCurve,
    BRepBooleanFailure,
    BRepBooleanPolicy,
    BRepBooleanResult,
    pcurve_affine_correspondence,
    pcurve_surface_period_shifts,
    PeriodicPCurve,
)
from phydrax.geometry.brep._intersection import IntersectionCurvePointRoot
from phydrax.geometry.brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    IntersectionSide,
)
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import AbstractCurve, CylinderPatch, LineCurve
from phydrax.geometry.brep._root_bindings import (
    BRepBranchNodeRoots,
    BRepVertexRoot,
    vertex_root_payload,
)
from phydrax.interchange._cad_archive import load_brep_archive, save_brep_archive
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)
_OTHER_COORDINATES = SpatialCoordinateContract(MILLIMETER, reference_frame="other")


def _source_identity(model: BRepModel) -> tuple[str, str]:
    geometry = model.geometry
    assert geometry is not None
    return model.model_id, geometry.geometry_id


def _branch_side(curve: AbstractCurve) -> IntersectionSide | None:
    source = curve.curve if isinstance(curve, CurveTrimSegment) else curve
    if not isinstance(source, IntersectionPCurve):
        return None
    return source.side


def _assert_exhaustive_oriented_lineage(result: BRepBooleanResult) -> None:
    graph = result.association_graph
    transaction = graph.transaction
    coverage = transaction.coverage
    assert coverage.source_exhaustive
    assert coverage.target_exhaustive
    assert not transaction.unresolved_source_ids
    assert not transaction.unresolved_target_ids

    source = {item.occurrence_id: item for item in graph.source_revision.occurrences}
    assert {item.kind for item in source.values()} >= {
        "assembly",
        "solid",
        "face",
        "edge",
        "vertex",
    }
    target = {item.occurrence_id: item for item in graph.target_revision.occurrences}
    if target:
        assert {item.kind for item in target.values()} >= {
            "assembly",
            "solid",
            "face",
            "edge",
            "vertex",
        }
    else:
        assert result.model.topology.num_solids == 0
    target_faces = {
        identifier: item for identifier, item in target.items() if item.kind == "face"
    }
    face_edges = tuple(
        edge
        for edge in transaction.correspondences
        if edge.target_occurrence_id in target_faces
    )
    assert {edge.target_occurrence_id for edge in face_edges} == set(target_faces)
    assert all(
        target_faces[edge.target_occurrence_id].orientation in (-1, 1)
        for edge in face_edges
    )

    mapped_source = {edge.source_occurrence_id for edge in transaction.correspondences}
    mapped_target = {edge.target_occurrence_id for edge in transaction.correspondences}
    assert set(result.deleted_source_ids) == set(source) - mapped_source
    assert mapped_source | set(result.deleted_source_ids) == set(source)
    for identifier in source:
        resolution = graph.resolve_target(graph.source_revision.select(identifier))
        if identifier in mapped_source:
            assert resolution.status in (
                AssociationStatus.UNIQUE,
                AssociationStatus.MULTIPLE,
            )
        else:
            assert resolution.status is AssociationStatus.NO_PREIMAGE
            assert resolution.relation == "deleted"
    for identifier in target:
        resolution = graph.resolve_preimage(graph.target_revision.select(identifier))
        if identifier in mapped_target:
            assert resolution.status in (
                AssociationStatus.UNIQUE,
                AssociationStatus.MULTIPLE,
            )
        else:
            assert resolution.status is AssociationStatus.NO_PREIMAGE
            assert resolution.relation == "created"


def test_boolean_pcurve_execution_shadows_match_exact_source_fields() -> None:
    patch = CylinderPatch(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    source = LineCurve((0.0, 0.0), (1.0, 1.0))
    identity = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
    zero = (Fraction(0), Fraction(0))

    periodic = PeriodicPCurve(source, patch, (1, 0))
    assert pcurve_surface_period_shifts(source, periodic, identity, zero, patch) == (1, 0)
    tampered_periodic = eqx.tree_at(
        lambda curve: curve.offset_value,
        periodic,
        jnp.asarray((0.125, 0.0), dtype=jnp.float64),
    )
    assert (
        pcurve_surface_period_shifts(source, tampered_periodic, identity, zero, patch)
        is None
    )
    assert pcurve_affine_correspondence(source, tampered_periodic) is None

    affine = AffinePCurve(
        source, np.asarray(identity, dtype=object), np.asarray(zero, dtype=object)
    )
    assert pcurve_surface_period_shifts(source, affine, identity, zero, patch) == (
        0,
        0,
    )
    tampered_matrix = eqx.tree_at(
        lambda curve: curve.matrix_value,
        affine,
        jnp.asarray(((1.0, 0.0), (0.0, 1.125)), dtype=jnp.float64),
    )
    assert (
        pcurve_surface_period_shifts(tampered_matrix, source, identity, zero, patch)
        is None
    )
    assert pcurve_affine_correspondence(tampered_matrix, source) is None
    tampered_offset = eqx.tree_at(
        lambda curve: curve.offset_value,
        affine,
        jnp.asarray((0.125, 0.0), dtype=jnp.float64),
    )
    assert (
        pcurve_surface_period_shifts(source, tampered_offset, identity, zero, patch)
        is None
    )
    assert pcurve_affine_correspondence(source, tampered_offset) is None

    underflow_matrix = np.asarray(
        (
            (Fraction(1, 10**400), Fraction(0)),
            (Fraction(0), Fraction(1)),
        ),
        dtype=object,
    )
    with pytest.raises(ValueError, match="nonsingular in binary64"):
        AffinePCurve(source, underflow_matrix, np.asarray(zero, dtype=object))


@pytest.mark.parametrize(
    ("second_base", "second_axis"),
    (
        ((-2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((2.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
        ((0.0, -2.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 2.0, 0.0), (0.0, -1.0, 0.0)),
    ),
    ids=(
        "positive-profile",
        "negative-profile",
        "seam-crossing-positive-profile",
        "seam-crossing-negative-profile",
    ),
)
def test_nonrational_quadric_boolean_publishes_complete_branches_and_pcurves(
    second_base: tuple[float, float, float],
    second_axis: tuple[float, float, float],
) -> None:
    first = brep_cylinder(
        1.0,
        4.0,
        base_center=(0.0, 0.0, -2.0),
        coordinate_contract=_COORDINATES,
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=second_base,
        axis=second_axis,
        coordinate_contract=_COORDINATES,
    )
    original = _source_identity(first), _source_identity(second)

    result = boolean_brep(first, second, "intersection")

    assert result.model.topology.num_solids == 1
    geometry = result.model.geometry
    assert geometry is not None
    branches = tuple(
        curve for curve in geometry.curves if isinstance(curve, IntersectionCurve)
    )
    assert branches
    for branch in branches:
        assert isinstance(branch, AbstractCurve)
        assert branch.fully_certified
        parameters = jnp.linspace(0.15, branch.num_charts - 0.15, 11, dtype=jnp.float64)
        coupled = branch.evaluate_coupled(parameters)
        first_uv = coupled.first_parameters
        second_uv = coupled.second_parameters
        points = np.asarray(coupled.point)
        np.testing.assert_allclose(
            np.asarray(branch.evaluate(parameters)), points, rtol=0.0, atol=1.0e-12
        )
        np.testing.assert_allclose(
            np.asarray(branch.first.patch.evaluate(first_uv)),
            points,
            rtol=0.0,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            np.asarray(branch.second.patch.evaluate(second_uv)),
            points,
            rtol=0.0,
            atol=1.0e-10,
        )
        first_pcurve = branch.p_curve("first")
        second_pcurve = branch.p_curve("second")
        assert isinstance(first_pcurve, IntersectionPCurve)
        assert isinstance(second_pcurve, IntersectionPCurve)
        for chart in range(branch.num_charts):
            interval = (chart + 0.1, chart + 0.9)
            assert branch.is_c1_on(*interval)
            assert first_pcurve.is_c1_on(*interval)
            assert second_pcurve.is_c1_on(*interval)
        np.testing.assert_allclose(
            np.asarray(first_pcurve.evaluate(parameters)),
            np.asarray(first_uv),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            np.asarray(second_pcurve.evaluate(parameters)),
            np.asarray(second_uv),
            rtol=0.0,
            atol=1.0e-12,
        )

        curve_index = next(
            index for index, curve in enumerate(geometry.curves) if curve is branch
        )
        branch_edges = tuple(
            edge
            for edge, index in enumerate(geometry.edge_curves)
            if index == curve_index
        )
        assert branch_edges
        for edge in branch_edges:
            assert len(geometry.topology.edge_faces[edge]) == 2
            coedges = tuple(
                coedge
                for coedge, used_edge in enumerate(geometry.coedge_edges)
                if used_edge == edge
            )
            sides = tuple(_branch_side(geometry.pcurves[coedge]) for coedge in coedges)
            assert len(sides) == 2
            assert all(side is not None for side in sides)
            assert {side for side in sides if side is not None} == {"first", "second"}
        branch_vertices = {
            vertex for edge in branch_edges for vertex in geometry.edge_vertices[edge]
        }
        assert any(
            geometry.vertex_roots[vertex] is not None for vertex in branch_vertices
        )

    _assert_exhaustive_oriented_lineage(result)
    assert (_source_identity(first), _source_identity(second)) == original


def test_closed_branch_endpoints_isolated_on_opposite_seam_bind_one_chart_point() -> None:
    first = brep_cylinder(
        1.0, 4.0, base_center=(0.0, 0.0, -2.0), coordinate_contract=_COORDINATES
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=(-2.0, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=_COORDINATES,
    )
    model = boolean_brep(first, second, "intersection").model
    geometry = model.geometry
    assert geometry is not None
    closed_faces = tuple(
        (face, loops[0][0])
        for face, loops in enumerate(geometry.face_loops)
        if len(loops) == 1 and len(loops[0]) == 1
    )
    # The larger cylinder keeps two disks, each bounded by one closed branch
    # whose roots were isolated on the smaller cylinder's seam generator.
    assert len(closed_faces) == 2
    for face, coedge in closed_faces:
        start_vertex, end_vertex = geometry.edge_vertices[geometry.coedge_edges[coedge]]
        assert start_vertex == end_vertex
        root = geometry.vertex_roots[start_vertex]
        assert root is not None
        start, end = geometry.coedge_endpoint_roots[coedge]
        pcurve = geometry.pcurves[coedge]
        assert root.same_uv_endpoint(model.patches[face], pcurve, end, pcurve, start)
        other_faces = tuple(
            index
            for index, patch in enumerate(model.patches)
            if patch is not model.patches[face] and index != face
        )
        assert other_faces
        # The branch p-curve of this face does not bind on a patch carrying the
        # other branch side.
        assert not root.same_uv_endpoint(
            model.patches[other_faces[-1]], pcurve, end, pcurve, start
        )


def test_seam_crossing_quadric_loop_closes_under_exact_per_chart_decks(
    tmp_path: Path,
) -> None:
    # The second cylinder runs along y, so one loop crosses the first
    # cylinder's seam twice; one crossing coincides with the second
    # cylinder's seam, where the loop closes.
    first = brep_cylinder(
        1.0, 4.0, base_center=(0.0, 0.0, -2.0), coordinate_contract=_COORDINATES
    )
    second = brep_cylinder(
        0.8,
        4.0,
        base_center=(0.0, -2.0, 0.0),
        axis=(0.0, 1.0, 0.0),
        coordinate_contract=_COORDINATES,
    )
    model = boolean_brep(first, second, "intersection").model
    geometry = model.geometry
    assert geometry is not None
    assert model.topology.num_solids == 1
    branches = {
        curve.branch_id: curve
        for curve in geometry.curves
        if isinstance(curve, IntersectionCurve)
    }
    crossing = tuple(
        branch
        for branch in branches.values()
        if np.any(np.asarray(branch.node_axes) == 0)
    )
    others = tuple(
        branch
        for branch in branches.values()
        if not np.any(np.asarray(branch.node_axes) == 0)
    )
    assert len(crossing) == 1 and len(others) == 1
    branch = crossing[0]
    # Each source chart closes under its own exact native turn.
    np.testing.assert_array_equal(branch.closure_period_shifts(), (1, 0, 1, 0))
    np.testing.assert_array_equal(others[0].closure_period_shifts(), (0, 0, 1, 0))
    # The interior crossing fixes the first cylinder's angle at its exact seam
    # value, and the charts on either side differ by one exact deck turn.
    seam_nodes = tuple(
        int(node)
        for node in np.flatnonzero(np.asarray(branch.node_axes) == 0)
        if 0 < node < branch.num_charts
    )
    assert len(seam_nodes) == 1
    node = seam_nodes[0]
    assert float(branch.node_values[node]) == 0.0
    shifts = np.asarray(branch.node_period_shifts[node])
    assert abs(int(shifts[0, 0]) - int(shifts[1, 0])) == 1
    np.testing.assert_array_equal(shifts[:, 1:], 0)
    # The start node lies exactly on both seams.
    assert set(branch.node_exact_coordinates(0)) == {(2, 0.0), (0, 0.0)}
    # Every chart lies on the authored first-cylinder sheet.
    angles = np.concatenate(
        (np.asarray(branch.chart_start)[:, 0], np.asarray(branch.chart_end)[:, 0])
    )
    assert np.all(angles >= -1.0e-12) and np.all(angles <= 2.0 * np.pi + 1.0e-12)

    # The disk the loop bounds on the first cylinder is split along that
    # cylinder's seam into two faces sharing one seam segment.
    def on_first_cylinder(face: int) -> bool:
        point = np.asarray(model.patches[face].evaluate(jnp.asarray((0.5, 0.5))))
        return bool(np.isclose(np.hypot(point[0], point[1]), 1.0, rtol=0.0, atol=1e-12))

    seam_edges = tuple(
        edge
        for edge, faces in enumerate(geometry.topology.edge_faces)
        if isinstance(geometry.curves[geometry.edge_curves[edge]], LineCurve)
        and len(set(faces)) == 2
        and all(on_first_cylinder(face) for face in faces)
    )
    assert len(seam_edges) == 1

    # Both seam roots of the start node bind one vertex through a validated
    # branch-node witness; every other vertex keeps its unwitnessed identity.
    roots = tuple(root for root in geometry.vertex_roots if root is not None)
    witnessed = tuple(root for root in roots if root.branch_node is not None)
    assert len(witnessed) == 1
    vertex_root = witnessed[0]
    witness = vertex_root.branch_node
    assert witness is not None and vertex_root.spatial_root is None
    assert witness.node.curve.branch_id == branch.branch_id
    assert witness.node.parameter == 0.0
    assert len({endpoint.root.root_id for endpoint in witness.endpoints}) == 2
    assert all(witness.binds(endpoint.root) for endpoint in witness.endpoints)
    assert all(
        "branch_node" not in vertex_root_payload(root)
        for root in roots
        if root.branch_node is None
    )
    with pytest.raises(ValueError, match="certify the branch node"):
        BRepBranchNodeRoots(
            IntersectionCurvePointRoot(branch, float(node)), witness.endpoints
        )
    with pytest.raises(ValueError, match="certify the vertex source root"):
        BRepVertexRoot(witness.node, branch_node=witness)

    archive = tmp_path / "seam-crossing-quadric.phx"
    save_brep_archive(model, archive)
    restored = load_brep_archive(archive).geometry
    assert restored is not None
    assert tuple(
        None if root is None else root.root_id for root in restored.vertex_roots
    ) == tuple(None if root is None else root.root_id for root in geometry.vertex_roots)


def _notch_branches(
    lower: tuple[float, float, float], upper: tuple[float, float, float]
) -> tuple[BRepBooleanResult, tuple[IntersectionCurve, ...]]:
    cylinder = brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES)
    notch = brep_box(lower, upper, coordinate_contract=_COORDINATES)
    original = _source_identity(cylinder), _source_identity(notch)
    result = boolean_brep(cylinder, notch, "difference")
    assert (_source_identity(cylinder), _source_identity(notch)) == original
    assert result.model.topology.num_solids == 1
    geometry = result.model.geometry
    assert geometry is not None
    branches = {
        curve.branch_id: curve
        for curve in geometry.curves
        if isinstance(curve, IntersectionCurve)
    }
    for branch in branches.values():
        assert branch.fully_certified
        assert not branch.closed
        assert (branch.start_kind, branch.end_kind) == ("boundary", "boundary")
    _assert_exhaustive_oriented_lineage(result)
    return result, tuple(branches.values())


def test_notch_arc_across_the_cylinder_seam_splits_under_an_exact_deck() -> None:
    # The notch sits on the cylinder's seam generator, so each cap-plane arc
    # runs from the end of the angle window into its start.
    _, branches = _notch_branches((-0.25, 0.75, 0.5), (0.25, 1.25, 1.5))
    arcs = tuple(
        branch for branch in branches if np.any(np.asarray(branch.node_axes)[1:-1] == 0)
    )
    assert len(arcs) == 2
    for arc in arcs:
        interior = np.asarray(arc.node_axes)[1:-1]
        values = np.asarray(arc.node_values)[1:-1]
        shifts = np.asarray(arc.node_period_shifts)[1:-1]
        seam = [
            index
            for index in range(len(interior))
            if interior[index] == 0 and values[index] == 0.0
        ]
        assert len(seam) == 1
        incoming, outgoing = shifts[seam[0]]
        np.testing.assert_array_equal(incoming - outgoing, (1, 0, 0, 0))
        # Both ends fix the exact plane edge coordinate they lie on.
        assert {int(arc.node_axes[0]), int(arc.node_axes[-1])} == {2}
        assert {float(arc.node_values[0]), float(arc.node_values[-1])} == {0.0, 0.5}


def test_notch_tangent_contacts_outside_the_trim_are_exactly_empty() -> None:
    # The notch face y = 1 is a tangent plane of the cylinder and its cap edges
    # are tangent lines, touching at x = 0, outside the notch (x >= 0.5). The
    # exact zero discriminants place those double roots outside each trim.
    _, branches = _notch_branches((0.5, 0.25, 0.5), (1.5, 1.0, 1.5))
    assert len(branches) == 4
    lines = [branch for branch in branches if int(branch.chart_axes[0]) == 1]
    arcs = [branch for branch in branches if int(branch.chart_axes[0]) == 0]
    assert len(lines) == 2 and len(arcs) == 2
    for arc in arcs:
        # Each arc runs from the edge y = 0.25 to the edge x = 0.5 of its cap.
        assert {int(arc.node_axes[0]), int(arc.node_axes[-1])} == {2, 3}
        assert {float(arc.node_values[0]), float(arc.node_values[-1])} == {0.0}


def test_tangent_cylinder_contact_has_empty_regularized_intersection() -> None:
    first = brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES)
    second = brep_cylinder(
        1.0,
        2.0,
        base_center=(0.0, 0.0, 2.0),
        coordinate_contract=_COORDINATES,
    )
    original = _source_identity(first), _source_identity(second)

    result = boolean_brep(first, second, "intersection")

    assert result.empty
    assert result.model.topology.num_faces == 0
    assert result.model.topology.num_solids == 0
    _assert_exhaustive_oriented_lineage(result)
    assert (_source_identity(first), _source_identity(second)) == original


def test_coincident_cylinder_overlay_keeps_one_oriented_source_boundary() -> None:
    first = brep_cylinder(1.0, 2.0, coordinate_contract=_COORDINATES)
    second = brep_cylinder(
        1.0,
        2.0,
        base_center=(0.0, 0.0, 1.0),
        coordinate_contract=_COORDINATES,
    )

    result = boolean_brep(first, second, "intersection")

    assert result.model.topology.num_solids == 1
    assert result.model.topology.num_faces > 0
    _assert_exhaustive_oriented_lineage(result)


def test_singular_point_contact_is_empty_or_explicitly_unresolved() -> None:
    first = brep_sphere(1.0, center=(0.0, 0.0, 0.0), coordinate_contract=_COORDINATES)
    second = brep_sphere(1.0, center=(2.0, 0.0, 0.0), coordinate_contract=_COORDINATES)
    original = _source_identity(first), _source_identity(second)

    try:
        result = boolean_brep(first, second, "intersection")
    except BRepBooleanFailure as error:
        assert error.reason
        assert error.source_entities
    else:
        assert result.empty
        assert result.model.topology.num_faces == 0
        _assert_exhaustive_oriented_lineage(result)
    assert (_source_identity(first), _source_identity(second)) == original


def test_identical_difference_and_disconnected_union_publish_complete_results() -> None:
    source = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_COORDINATES,
    )
    original = _source_identity(source)

    empty = boolean_brep(source, source, "difference")
    assert empty.empty
    _assert_exhaustive_oriented_lineage(empty)

    first = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_COORDINATES,
    )
    second = brep_box(
        (2.0, 0.0, 0.0),
        (3.0, 1.0, 1.0),
        coordinate_contract=_COORDINATES,
    )
    union = boolean_brep(first, second, "union")
    assert union.model.topology.num_solids == 2
    _assert_exhaustive_oriented_lineage(union)
    graph = union.association_graph
    for target_entity in graph.target_revision.occurrences:
        if target_entity.kind not in ("edge", "vertex"):
            continue
        resolution = graph.resolve_preimage(
            graph.target_revision.select(target_entity.occurrence_id)
        )
        assert resolution.status in (
            AssociationStatus.UNIQUE,
            AssociationStatus.MULTIPLE,
        )
        assert all(
            source_entity.kind == target_entity.kind
            for source_entity in resolution.candidate_selectors
        )
    assert _source_identity(source) == original


def test_boolean_failure_does_not_mutate_sources_and_policy_has_no_work_quotas() -> None:
    first = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_COORDINATES,
    )
    second = brep_box(
        (0.5, 0.0, 0.0),
        (1.5, 1.0, 1.0),
        coordinate_contract=_OTHER_COORDINATES,
    )
    original = _source_identity(first), _source_identity(second)
    with pytest.raises(ValueError):
        boolean_brep(first, second, "intersection")
    assert (_source_identity(first), _source_identity(second)) == original

    policy = BRepBooleanPolicy()
    assert not hasattr(policy, "maximum_cells")
    assert not hasattr(policy, "maximum_faces")
    assert not hasattr(policy.intersection, "maximum_boxes")
    assert not hasattr(policy.intersection, "maximum_march_steps")
    assert not hasattr(policy.sewing, "maximum_coedges")
