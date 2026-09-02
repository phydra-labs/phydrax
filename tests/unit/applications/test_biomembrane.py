#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cellular_mechanics._membrane import (
    _self_intersection_free,
    _vertex_links_valid,
    BiomembranePlan,
)


jax.config.update("jax_enable_x64", True)


def _tetrahedron():
    vertices = np.asarray(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    ) / np.sqrt(3.0)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return vertices, faces


def _octahedron():
    vertices = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    faces = np.asarray(
        [
            [4, 0, 2],
            [4, 2, 1],
            [4, 1, 3],
            [4, 3, 0],
            [5, 2, 0],
            [5, 1, 2],
            [5, 3, 1],
            [5, 0, 3],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _icosphere(level: int):
    ratio = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, ratio, 0),
            (1, ratio, 0),
            (-1, -ratio, 0),
            (1, -ratio, 0),
            (0, -1, ratio),
            (0, 1, ratio),
            (0, -1, -ratio),
            (0, 1, -ratio),
            (ratio, 0, -1),
            (ratio, 0, 1),
            (-ratio, 0, -1),
            (-ratio, 0, 1),
        ],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int32,
    )
    for _ in range(level):
        vertex_list = list(vertices)
        midpoint: dict[tuple[int, int], int] = {}

        def middle(first: int, second: int) -> int:
            edge = (min(first, second), max(first, second))
            if edge not in midpoint:
                point = vertices[first] + vertices[second]
                point /= np.linalg.norm(point)
                midpoint[edge] = len(vertex_list)
                vertex_list.append(point)
            return midpoint[edge]

        refined = []
        for first, second, third in faces:
            ab = middle(int(first), int(second))
            bc = middle(int(second), int(third))
            ca = middle(int(third), int(first))
            refined.extend(
                (
                    (first, ab, ca),
                    (second, bc, ab),
                    (third, ca, bc),
                    (ab, bc, ca),
                )
            )
        vertices = np.asarray(vertex_list)
        faces = np.asarray(refined, dtype=np.int32)
    return vertices, faces


def _prepared(*, species: bool = False):
    vertices, faces = _octahedron()
    keywords = {}
    if species:
        keywords = {
            "species_diffusivity": (0.2, 0.1),
            "reaction_matrix": ((-0.3, 0.2), (0.3, -0.2)),
            "curvature_coupling": (0.1, -0.05),
        }
    plan = BiomembranePlan(
        faces,
        vertex_ids=np.arange(100, 106),
        face_ids=np.arange(200, 208),
        bending_rigidity=np.linspace(0.8, 1.2, 6),
        gaussian_rigidity=-0.15,
        spontaneous_curvature=0.2,
        local_area_modulus=0.4,
        global_area_modulus=1.3,
        volume_modulus=1.7,
        tension=0.05,
        pressure=0.07,
        mobility=np.linspace(0.4, 0.9, 6),
        **keywords,
    )
    return plan.prepare(vertices)


def test_plan_rejects_nonmanifold_and_inconsistently_oriented_topology():
    vertices, faces = _tetrahedron()
    del vertices
    open_faces = faces[:-1]
    with pytest.raises(ValueError, match="at least four faces"):
        BiomembranePlan(open_faces)
    reversed_face = faces.copy()
    reversed_face[0] = reversed_face[0, ::-1]
    with pytest.raises(ValueError, match="opposite edge orientations"):
        BiomembranePlan(reversed_face)


def test_energy_is_rigid_motion_invariant_and_internal_resultants_vanish():
    prepared = _prepared()
    state = prepared.state()
    angle = 0.43
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translated = np.asarray(state.positions) @ rotation.T + np.asarray([1.2, -0.7, 0.3])
    original = prepared.evaluate(state)
    moved = prepared.evaluate(prepared.state(translated))
    np.testing.assert_allclose(
        moved.energy.total,
        original.energy.total,
        rtol=2.0e-11,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        original.energy.gaussian, -0.15 * 4.0 * np.pi, atol=2.0e-12
    )
    assert float(original.geometry.conservative_force_residual) < 2.0e-11
    assert float(original.geometry.conservative_torque_residual) < 2.0e-11
    np.testing.assert_allclose(original.geometry.area_residual, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(original.geometry.volume_residual, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(original.geometry.local_area_residual, 0.0, atol=1.0e-14)


def test_spherical_helfrich_energy_converges_to_eight_pi():
    errors = []
    for level in (0, 1, 2):
        vertices, faces = _icosphere(level)
        prepared = BiomembranePlan(faces, bending_rigidity=1.0).prepare(vertices)
        energy = float(prepared.evaluate(prepared.state()).energy.helfrich)
        errors.append(abs(energy - 8.0 * np.pi))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.8


def test_force_is_negative_energy_gradient_by_virtual_work():
    prepared = _prepared(species=True)
    mass = 0.03 + 0.01 * np.arange(12, dtype=np.float64).reshape(6, 2)
    state = prepared.state(species_mass=mass)
    direction = np.sin(np.arange(18, dtype=np.float64)).reshape(6, 3)
    direction -= np.mean(direction, axis=0, keepdims=True)
    step = 2.0e-6
    plus = prepared.state(np.asarray(state.positions) + step * direction, mass)
    minus = prepared.state(np.asarray(state.positions) - step * direction, mass)
    finite_difference = (
        float(prepared.energy(plus).total) - float(prepared.energy(minus).total)
    ) / (2.0 * step)
    virtual_work = -float(
        np.sum(np.asarray(prepared.evaluate(state).conservative_force) * direction)
    )
    np.testing.assert_allclose(finite_difference, virtual_work, rtol=3.0e-6, atol=3.0e-7)


def test_surface_diffusion_reaction_conserves_total_species_mass():
    prepared = _prepared(species=True)
    mass = np.asarray(
        [
            [0.30, 0.10],
            [0.22, 0.15],
            [0.18, 0.12],
            [0.25, 0.20],
            [0.16, 0.14],
            [0.19, 0.11],
        ]
    )
    result = prepared.diffuse_react(prepared.state(species_mass=mass), 0.01)
    assert bool(result.evidence.successful)
    assert bool(result.evidence.conservative)
    np.testing.assert_allclose(result.evidence.total_mass_residual, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(np.sum(result.mass_rate), 0.0, atol=2.0e-14)
    assert not np.allclose(result.accepted_state.species_mass, mass)


def test_thermal_increment_has_fdt_covariance_and_stable_rng_identity():
    vertices, faces = _tetrahedron()
    prepared = BiomembranePlan(faces, bending_rigidity=0.0, mobility=0.7).prepare(
        vertices
    )
    state = prepared.state()
    keys = jax.random.split(jax.random.key(71), 1024)
    increments = jax.vmap(
        lambda key: (
            prepared.thermal_step(
                state, key, 0.02, 0.4, boltzmann_constant=1.3, step_index=9
            ).evidence.stochastic_displacement
        )
    )(keys)
    expected = 2.0 * 1.3 * 0.4 * 0.02 * 0.7
    sample = np.asarray(increments).reshape((-1, 3))
    covariance = np.cov(sample, rowvar=False, bias=True)
    np.testing.assert_allclose(np.diag(covariance), expected, rtol=0.08, atol=0.0)
    np.testing.assert_allclose(
        covariance - np.diag(np.diag(covariance)), 0.0, atol=0.0012
    )
    first = prepared.thermal_step(state, keys[0], 0.02, 0.4, step_index=9)
    repeated = prepared.thermal_step(state, keys[0], 0.02, 0.4, step_index=9)
    np.testing.assert_array_equal(
        first.candidate_state.positions,
        repeated.candidate_state.positions,
    )
    assert first.evidence.rng_identity == repeated.evidence.rng_identity


def test_split_transaction_conserves_fields_and_changes_preparation_identity():
    prepared = _prepared(species=True)
    mass = 0.02 + 0.005 * np.arange(12).reshape(6, 2)
    state = prepared.state(species_mass=mass)
    proposal = prepared.propose_split(state, (100, 102))
    assert proposal.manifold
    assert proposal.oriented
    assert proposal.self_intersection_free
    assert proposal.candidate.prepared_id != prepared.prepared_id
    np.testing.assert_allclose(
        np.sum(proposal.candidate_state.species_mass, axis=0),
        np.sum(mass, axis=0),
        atol=2.0e-15,
    )
    evidence = prepared.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=10.0,
    )
    assert bool(evidence.accepted)
    np.testing.assert_allclose(evidence.species_mass_jump, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(evidence.material_integral_jump, 0.0, atol=2.0e-13)
    committed = prepared.commit_remesh(proposal, evidence)
    assert committed.committed
    assert committed.prepared.prepared_id == proposal.candidate.prepared_id
    assert set(np.asarray(prepared.plan.vertex_ids)).issubset(
        set(np.asarray(committed.prepared.plan.vertex_ids))
    )


def test_rejected_remesh_rolls_back_exact_source_objects():
    prepared = _prepared(species=True)
    state = prepared.state(species_mass=np.full((6, 2), 0.1))
    proposal = prepared.propose_split(state, (100, 102))
    evidence = prepared.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=0.0,
        maximum_relative_volume_jump=0.0,
        maximum_relative_energy_jump=0.0,
    )
    assert not bool(evidence.accepted)
    result = prepared.commit_remesh(proposal, evidence)
    assert not result.committed
    assert result.prepared is prepared
    assert result.state is state


def test_state_and_remesh_evidence_are_bound_to_preparation_epoch():
    vertices, faces = _tetrahedron()
    first = BiomembranePlan(faces, species_diffusivity=(0.1,)).prepare(vertices)
    second = BiomembranePlan(
        faces,
        vertex_ids=(10, 11, 12, 13),
        species_diffusivity=(0.1,),
    ).prepare(vertices)
    state = first.state(species_mass=np.full((4, 1), 0.1))
    with pytest.raises(ValueError, match="different membrane preparation"):
        second.evaluate(state)

    changed = first.state(species_mass=np.full((4, 1), 0.2))
    first_proposal = first.propose_split(state, (0, 1))
    second_proposal = first.propose_split(changed, (0, 1))
    assert first_proposal.proposal_id != second_proposal.proposal_id
    evidence = first.evaluate_remesh(
        first_proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=10.0,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        first.commit_remesh(second_proposal, evidence)


def test_split_preserves_uniform_concentration_and_transferred_rest_area():
    vertices, faces = _tetrahedron()
    vertices = vertices.copy()
    vertices[0] *= 1.04
    prepared = BiomembranePlan(
        faces,
        local_area_modulus=2.0,
        species_diffusivity=(0.1,),
    ).prepare(vertices)
    deformed = vertices.copy()
    deformed[:, 2] *= 1.05
    deformed_state = prepared.state(deformed)
    dual_area = np.asarray(prepared.evaluate(deformed_state).geometry.vertex_area)
    state = prepared.state(deformed, (0.7 * dual_area)[:, None])
    proposal = prepared.propose_split(state, (0, 1))
    assert proposal.stencil_valid
    candidate_evaluation = proposal.candidate.evaluate(proposal.candidate_state)
    np.testing.assert_allclose(
        candidate_evaluation.species_concentration,
        0.7,
        rtol=2.0e-10,
        atol=2.0e-10,
    )
    assert float(candidate_evaluation.energy.local_area) > 0.0
    assert not np.allclose(
        proposal.candidate.reference_face_area,
        candidate_evaluation.geometry.face_area,
    )


def test_constant_active_normal_traction_has_zero_closed_surface_resultant():
    vertices, faces = _tetrahedron()
    vertices = vertices.copy()
    vertices[0] *= 1.03
    prepared = BiomembranePlan(faces, bending_rigidity=0.0, active_traction=0.8).prepare(
        vertices
    )
    evaluation = prepared.evaluate(prepared.state())
    np.testing.assert_allclose(np.sum(evaluation.active_force, axis=0), 0.0, atol=2.0e-14)


def test_collapse_keeps_removed_patch_material_in_its_one_ring():
    vertices, faces = _octahedron()
    modulus = np.arange(1.0, 9.0)
    prepared = BiomembranePlan(
        faces,
        vertex_ids=np.arange(100, 106),
        face_ids=np.arange(200, 208),
        local_area_modulus=modulus,
    ).prepare(vertices)
    proposal = prepared.propose_collapse(prepared.state(), (100, 102))
    assert proposal.candidate.prepared_id != prepared.prepared_id
    far_ids = [
        int(prepared.plan.face_ids[index])
        for index, face in enumerate(faces)
        if 0 not in face and 2 not in face
    ]
    source_ids = np.asarray(prepared.plan.face_ids)
    candidate_ids = np.asarray(proposal.candidate.plan.face_ids)
    for face_id in far_ids:
        source = int(np.flatnonzero(source_ids == face_id)[0])
        candidate = int(np.flatnonzero(candidate_ids == face_id)[0])
        np.testing.assert_allclose(
            proposal.candidate.reference_face_area[candidate],
            prepared.reference_face_area[source],
            atol=2.0e-14,
        )
        np.testing.assert_allclose(
            proposal.candidate.plan.local_area_modulus[candidate],
            prepared.plan.local_area_modulus[source],
            atol=2.0e-14,
        )


def test_centered_volume_is_invariant_at_large_global_offset():
    vertices, faces = _tetrahedron()
    prepared = BiomembranePlan(faces).prepare(vertices)
    shifted = vertices + np.asarray((1.0e9, -2.0e9, 3.0e9))
    shifted_prepared = BiomembranePlan(faces).prepare(shifted)
    np.testing.assert_allclose(
        shifted_prepared.reference_volume,
        prepared.reference_volume,
        rtol=2.0e-7,
        atol=2.0e-7,
    )


def test_transport_rejects_nonfinite_derived_geometry_and_is_conservative():
    vertices, faces = _tetrahedron()
    prepared = BiomembranePlan(
        faces,
        species_diffusivity=(0.1,),
        reaction_matrix=((1.0e-13,),),
    ).prepare(vertices)
    np.testing.assert_allclose(
        np.sum(prepared.plan.reaction_matrix, axis=0), 0.0, atol=0.0
    )
    overflow = prepared.state(vertices * 1.0e200, np.ones((4, 1)))
    result = prepared.diffuse_react(overflow, 0.01)
    assert not bool(result.evidence.successful)
    assert not bool(result.evidence.finite)


def test_transport_uses_signed_cotangent_on_deformed_non_delaunay_edge():
    vertices, faces = _tetrahedron()
    prepared = BiomembranePlan(faces, species_diffusivity=(0.3,)).prepare(vertices)
    deformed = vertices.copy()
    midpoint = 0.5 * (vertices[1] + vertices[2])
    deformed[0] = midpoint + 0.02 * (vertices[0] - midpoint)
    (
        _,
        _,
        vertex_area,
        vertex_normal,
        _,
        _,
    ) = prepared._surface_geometry(jnp.asarray(deformed))
    _, _, cotangent = prepared._curvature(
        jnp.asarray(deformed), vertex_area, vertex_normal
    )
    edges = np.asarray(prepared.plan.edge_vertices)
    edge = int(
        np.flatnonzero(
            ((edges[:, 0] == 1) & (edges[:, 1] == 2))
            | ((edges[:, 0] == 2) & (edges[:, 1] == 1))
        )[0]
    )
    assert float(cotangent[edge]) < 0.0
    concentration = np.arange(4, dtype=np.float64)[:, None]
    state = prepared.state(deformed, np.asarray(vertex_area)[:, None] * concentration)
    result = prepared.diffuse_react(state, 1.0e-4)
    first, second = edges[edge]
    expected = (
        0.5
        * float(cotangent[edge])
        * 0.3
        * (concentration[second, 0] - concentration[first, 0])
    )
    np.testing.assert_allclose(
        result.edge_flux[edge, 0], expected, rtol=2.0e-12, atol=2.0e-12
    )


def test_invalid_species_state_cannot_be_certified_by_remeshing():
    prepared = _prepared(species=True)
    mass = np.full((6, 2), 0.1)
    mass[0, 0] = -0.01
    proposal = prepared.propose_split(prepared.state(species_mass=mass), (100, 102))
    evidence = prepared.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=10.0,
    )
    assert not bool(evidence.accepted)


def test_vertex_link_and_shared_edge_intersection_guards_are_explicit():
    _, tetra_faces = _tetrahedron()
    second = tetra_faces + 3
    second[second == 3] = 0
    pinched = np.concatenate((tetra_faces, second), axis=0)
    assert not _vertex_links_valid(pinched, 7)

    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.2, 1.0, 0.0),
            (0.8, 1.0, 0.0),
        )
    )
    overlapping = np.asarray(((0, 1, 2), (1, 0, 3)), dtype=np.int32)
    assert not _self_intersection_free(positions, overlapping, 1.0e-12)
