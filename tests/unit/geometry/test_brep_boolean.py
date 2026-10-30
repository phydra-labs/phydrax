from __future__ import annotations

from fractions import Fraction

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
from phydrax.geometry.brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    IntersectionSide,
)
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._patches import AbstractCurve, CylinderPatch, LineCurve
from phydrax.units import MILLIMETER


_COORDINATES = SpatialCoordinateContract(MILLIMETER)
_OTHER_COORDINATES = SpatialCoordinateContract(MILLIMETER, reference_frame="other")


def _source_identity(model: BRepModel) -> tuple[str, str]:
    geometry = model.geometry
    assert geometry is not None
    return model.model_id, geometry.geometry_id


def _branch_side(curve: AbstractCurve) -> IntersectionSide | None:
    if not isinstance(curve, CurveTrimSegment):
        return None
    source = curve.curve
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
    ),
    ids=("positive-profile", "negative-profile"),
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
