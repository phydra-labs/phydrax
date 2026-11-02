#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry.brep import (
    brep_box,
    brep_planar_face,
    BRepAffinePlacement,
    BRepEntityId,
    BRepModel,
    BRepPartitionRole,
    edit_brep_occurrence_placement,
    PlanarEmbedding,
    PlanarPartitionOperand,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)


def _translation(y: float, z: float = 0.0) -> BRepAffinePlacement:
    return BRepAffinePlacement.from_matrix(
        ((1, 0, 0, 0), (0, 1, 0, y), (0, 0, 1, z), (0, 0, 0, 1))
    )


def _source_edge(model: BRepModel, path: tuple[str, ...]) -> BRepEntityId:
    # brep_box authors its first profile segment as base edge zero.
    edge = BRepEntityId(model.source_id, model.source_revision, "edge", 0, path)
    assert edge in {
        record.member
        for record in model.geometry.qualified_entity_incidence(
            model.source_id, model.source_revision
        )
    }
    return edge


def _scene(
    model: BRepModel, edge: BRepEntityId
) -> tuple[phx.linalg.ArraySpace, phx.discretization.PreparedCollisionScene, np.ndarray]:
    assert (
        edge.source_id == model.source_id
        and edge.source_revision == model.source_revision
    )
    path = edge.occurrence_path
    endpoints = model.geometry.edge_vertices[edge.index]
    occurrence = next(
        item for item in model.geometry.iter_occurrences() if item.path == path
    )
    local = np.asarray(model.geometry.vertex_points)[list(endpoints)]
    world = np.asarray(
        [occurrence.placement.apply(point) for point in local], dtype=float
    )
    space = phx.linalg.ArraySpace((2, 2), dtype=np.float64)
    moving_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((0, 1)), ambient_dimension=2, edges=jnp.asarray(((0, 1),))
    )
    moving = phx.discretization.PreparedCollisionSurface(
        moving_plan,
        jnp.asarray(world[:, :2]),
        phx.discretization.selection_collision_operator(space, jnp.asarray((0, 1))),
    )
    obstacle_plan = phx.discretization.CollisionSurfacePlan(
        jnp.asarray((10, 11)),
        ambient_dimension=2,
        edges=jnp.asarray(((0, 1),)),
        body_ids=1,
        static_mask=True,
    )
    obstacle = phx.discretization.PreparedCollisionSurface(
        obstacle_plan,
        jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
        phx.discretization.static_collision_operator(space, 2, 2),
    )
    return space, phx.discretization.PreparedCollisionScene((moving, obstacle)), world


def test_contact_scene_rebuilt_from_placement_and_rejected_candidate_rolls_back() -> None:
    source = brep_box(
        (0, 0, 0), (1, 1, 1), coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    occurrence = next(iter(source.geometry.iter_occurrences()))
    path = occurrence.path
    accepted = edit_brep_occurrence_placement(source, path, _translation(1.0)).model
    edge = _source_edge(accepted, path)
    endpoints = accepted.geometry.edge_vertices[edge.index]
    space, scene, initial_world = _scene(accepted, edge)
    search = phx.discretization.SweepAndPruneContactSearchPlan(
        edge_vertex_capacity=16,
        edge_edge_capacity=0,
        face_vertex_capacity=0,
        activation_distance=0.1,
    )
    positions = scene.positions(space.zeros())
    initial = search.build(scene, positions)
    assert int(initial.edge_vertex.actual_count) == 0
    np.testing.assert_allclose(initial_world[:, 1:], ((1.0, 0.0), (1.0, 0.0)))
    np.testing.assert_allclose(np.asarray(positions)[:2], initial_world[:, :2])

    edit = edit_brep_occurrence_placement(accepted, path, _translation(0.05))
    trial = edit.model
    trial_edge = _source_edge(trial, path)
    trial_space, trial_scene, trial_world = _scene(trial, trial_edge)
    trial_positions = trial_scene.positions(trial_space.zeros())
    candidate = search.build(trial_scene, trial_positions)
    assert int(candidate.edge_vertex.actual_count) > 0
    np.testing.assert_allclose(
        trial_world - initial_world, ((0, -0.95, 0), (0, -0.95, 0))
    )
    np.testing.assert_allclose(np.asarray(trial_positions)[:2], trial_world[:, :2])
    assert next(iter(trial.geometry.iter_occurrences())).path == path
    assert tuple(trial.geometry.edge_vertices[trial_edge.index]) == endpoints
    # The application rejects a pose with active proximity candidates; the prior
    # immutable scene and accepted source remain usable without reconstructing them.
    assert (
        int(search.build(scene, scene.positions(space.zeros())).edge_vertex.actual_count)
        == 0
    )
    np.testing.assert_allclose(scene.positions(space.zeros()), positions)
    assert next(iter(accepted.geometry.iter_occurrences())).placement == _translation(1.0)
    with pytest.raises(ValueError, match="nonsingular"):
        BRepAffinePlacement.from_matrix(
            ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
        )
    assert (
        int(search.build(scene, scene.positions(space.zeros())).edge_vertex.actual_count)
        == 0
    )


def test_occurrence_embedding_is_refused_and_world_frame_revalidated() -> None:
    source = brep_box(
        (0, 0, 0), (1, 1, 1), coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    path = next(iter(source.geometry.iter_occurrences())).path
    accepted = edit_brep_occurrence_placement(source, path, _translation(1.0)).model
    moved = edit_brep_occurrence_placement(accepted, path, _translation(1.0, 2.0)).model
    _, _, accepted_world = _scene(accepted, _source_edge(accepted, path))
    _, _, moved_world = _scene(moved, _source_edge(moved, path))
    original_frame = PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    moved_frame = PlanarEmbedding((0, 0, 2), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    np.testing.assert_allclose(original_frame.plane_residual(accepted_world), 0)
    np.testing.assert_allclose(original_frame.plane_residual(moved_world), 2)
    np.testing.assert_allclose(moved_frame.plane_residual(moved_world), 0)
    np.testing.assert_allclose(
        moved_frame.to_world(moved_frame.to_planar(moved_world)), moved_world
    )
    with pytest.raises(ValueError, match="face-only"):
        phx.geometry.prepare_brep_projection(moved, embedding=moved_frame)
    with pytest.raises(ValueError, match="root faces and zero solids"):
        PlanarPartitionOperand("placed", moved, BRepPartitionRole.REGION)
    np.testing.assert_allclose(
        original_frame.to_world(original_frame.to_planar(accepted_world)), accepted_world
    )
    assert next(iter(accepted.geometry.iter_occurrences())).placement == _translation(1.0)


def test_face_projection_revalidates_changed_world_embedding() -> None:
    loop = ProfileLoop.polygon(((0, 0), (1, 0), (1, 1), (0, 1)))
    contract = phx.SpatialCoordinateContract.si()
    old_face = brep_planar_face(
        PlanarProfile(ProfilePlane(), loop), coordinate_contract=contract
    )
    new_face = brep_planar_face(
        PlanarProfile(ProfilePlane(origin=(0, 0, 2)), loop), coordinate_contract=contract
    )
    old_frame = PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    new_frame = PlanarEmbedding((0, 0, 2), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    old_projection = phx.geometry.prepare_brep_projection(old_face, embedding=old_frame)
    with pytest.raises(ValueError, match="embedding must contain every source vertex"):
        phx.geometry.prepare_brep_projection(new_face, embedding=old_frame)
    new_projection = phx.geometry.prepare_brep_projection(new_face, embedding=new_frame)
    dimension = np.asarray([int(phx.geometry.BRepEntityDimension.FACE)])
    index = np.asarray([0])
    query = np.asarray([[0.25, 0.5]])
    before = old_projection.project(query, dimension, index)
    after = new_projection.project(query, dimension, index)
    np.testing.assert_allclose(before.points, after.points)
    np.testing.assert_allclose(old_frame.to_world(before.points), ((0.25, 0.5, 0),))
    np.testing.assert_allclose(new_frame.to_world(after.points), ((0.25, 0.5, 2),))
    np.testing.assert_allclose(
        old_projection.project(query, dimension, index).points, before.points
    )
