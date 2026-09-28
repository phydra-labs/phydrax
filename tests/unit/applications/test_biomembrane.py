#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.cellular_mechanics import BiomembranePlan
from phydrax.geometry.multiregion_surface import (
    EdgeFlipProposal,
    EdgeSplitProposal,
    MultiRegionSurfaceValidationPolicy,
    SurfaceEventKind,
    SurfaceEventPassStatus,
    SurfaceEventPolicy,
)
from tests._support.assertions import assert_tree_equal


jax.config.update("jax_enable_x64", True)


def _tetrahedron() -> Any:
    vertices = np.asarray(
        [[1.0, 1.0, 1.0], [-1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, -1.0]]
    ) / np.sqrt(3.0)
    faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
    return vertices, faces


def _octahedron() -> Any:
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


def _cube() -> Any:
    vertices = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    faces = np.asarray(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def _icosphere(level: int) -> Any:
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


def _prepared(*, species: bool = False) -> Any:
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
        # ty: ignore[invalid-argument-type]
        **keywords,
    )
    return plan.prepare(vertices)


def test_biomembrane_scenario_1() -> None:
    vertices, faces = _tetrahedron()
    del vertices
    open_faces = faces[:-1]
    with pytest.raises(ValueError, match="at least four faces"):
        BiomembranePlan(open_faces)
    reversed_face = faces.copy()
    reversed_face[0] = reversed_face[0, ::-1]
    with pytest.raises(ValueError, match="opposite edge orientations"):
        BiomembranePlan(reversed_face)
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
    errors = []
    for level in (0, 1, 2):
        vertices, faces = _icosphere(level)
        prepared = BiomembranePlan(faces, bending_rigidity=1.0).prepare(vertices)
        energy = float(prepared.evaluate(prepared.state()).energy.helfrich)
        errors.append(abs(energy - 8.0 * np.pi))
    assert errors[2] < errors[1] < errors[0]
    assert errors[2] < 0.8


def test_biomembrane_scenario_2() -> None:
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


def test_biomembrane_scenario_3() -> None:
    prepared = _prepared(species=True)
    mass = 0.02 + 0.005 * np.arange(12).reshape(6, 2)
    state = prepared.state(species_mass=mass)
    proposal = prepared.propose_remesh(state, EdgeSplitProposal((100, 102)))
    assert proposal.surface_result.evidence.status is SurfaceEventPassStatus.COMMITTED
    assert proposal.surface_result.evidence.validation is not None
    assert proposal.surface_result.evidence.validation.profile == "manifold_two_region"
    assert proposal.surface_result.evidence.validation.self_intersection_checked
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
    np.testing.assert_allclose(evidence.area_jump, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(evidence.volume_jump, 0.0, atol=2.0e-13)
    np.testing.assert_allclose(evidence.species_mass_jump, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(evidence.material_integral_jump, 0.0, atol=2.0e-13)
    assert proposal.vertex_transfer_evidence is not None
    assert proposal.face_transfer_evidence is not None
    assert bool(proposal.vertex_transfer_evidence.successful)
    np.testing.assert_allclose(
        proposal.vertex_transfer_evidence.absolute_defect, 0.0, atol=2.0e-13
    )
    np.testing.assert_allclose(
        proposal.face_transfer_evidence.absolute_defect, 0.0, atol=2.0e-13
    )
    assert bool(proposal.face_transfer_evidence.successful)
    committed = prepared.commit_remesh(proposal, evidence)
    assert committed.committed
    assert committed.prepared.prepared_id == proposal.candidate.prepared_id
    assert set(np.asarray(prepared.plan.vertex_ids)).issubset(
        set(np.asarray(committed.prepared.plan.vertex_ids))
    )

    prepared = _prepared(species=True)
    state = prepared.state(species_mass=np.full((6, 2), 0.1))
    proposal = prepared.propose_remesh(state, EdgeSplitProposal((100, 102)))
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
    assert result.lineage is None
    assert result.vertex_transition is None
    assert result.face_transition is None
    vertices, faces = _tetrahedron()
    first = BiomembranePlan(faces, species_diffusivity=(0.1,)).prepare(vertices)
    second = BiomembranePlan(
        faces,
        # ty: ignore[invalid-argument-type]
        vertex_ids=(10, 11, 12, 13),
        species_diffusivity=(0.1,),
    ).prepare(vertices)
    state = first.state(species_mass=np.full((4, 1), 0.1))
    with pytest.raises(ValueError, match="different membrane preparation"):
        second.evaluate(state)

    changed = first.state(species_mass=np.full((4, 1), 0.2))
    first_proposal = first.propose_remesh(state, EdgeSplitProposal((0, 1)))
    second_proposal = first.propose_remesh(changed, EdgeSplitProposal((0, 1)))
    assert first_proposal.proposal_id != second_proposal.proposal_id
    evidence = first.evaluate_remesh(
        first_proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=10.0,
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        first.commit_remesh(second_proposal, evidence)


def test_biomembrane_scenario_4() -> None:
    vertices, faces = _tetrahedron()
    vertices = vertices.copy()
    vertices[0] *= 1.03
    prepared = BiomembranePlan(faces, bending_rigidity=0.0, active_traction=0.8).prepare(
        vertices
    )
    evaluation = prepared.evaluate(prepared.state())
    np.testing.assert_allclose(np.sum(evaluation.active_force, axis=0), 0.0, atol=2.0e-14)

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


def test_biomembrane_scenario_5() -> None:
    vertices, faces = _tetrahedron()
    prepared = BiomembranePlan(
        faces,
        species_diffusivity=(0.1,),
        # ty: ignore[invalid-argument-type]
        reaction_matrix=((1.0e-13,),),
    ).prepare(vertices)
    np.testing.assert_allclose(
        np.sum(prepared.plan.reaction_matrix, axis=0), 0.0, atol=0.0
    )
    overflow = prepared.state(vertices * 1.0e200, np.ones((4, 1)))
    result = prepared.diffuse_react(overflow, 0.01)
    assert not bool(result.evidence.successful)
    assert not bool(result.evidence.finite)
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
    prepared = _prepared(species=True)
    mass = np.full((6, 2), 0.1)
    mass[0, 0] = -0.01
    proposal = prepared.propose_remesh(
        prepared.state(species_mass=mass), EdgeSplitProposal((100, 102))
    )
    evidence = prepared.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=10.0,
    )
    assert not bool(evidence.accepted)
    prepared = _prepared(species=True)
    cold = prepared.plan.prepare(np.asarray(prepared.reference_positions))
    mass = 0.1 + np.arange(12, dtype=np.float64).reshape((6, 2)) / 100.0
    event = EdgeSplitProposal((100, 102))
    warm_proposal = prepared.propose_remesh(prepared.state(species_mass=mass), event)
    cold_proposal = cold.propose_remesh(cold.state(species_mass=mass), event)
    assert (
        warm_proposal.surface_result.topology.topology_id
        == cold_proposal.surface_result.topology.topology_id
    )
    assert (
        warm_proposal.surface_result.evidence.evidence_id
        == cold_proposal.surface_result.evidence.evidence_id
    )
    assert_tree_equal(
        warm_proposal.candidate_state,
        cold_proposal.candidate_state,
    )
    _, tetra_faces = _tetrahedron()
    second = tetra_faces + 3
    second[second == 3] = 0
    pinched = np.concatenate((tetra_faces, second), axis=0)
    with pytest.raises(ValueError, match="vertex link"):
        BiomembranePlan(pinched)

    with pytest.raises(ValueError, match="manifold_two_region"):
        BiomembranePlan(
            tetra_faces,
            remesh_policy=SurfaceEventPolicy(
                validation=MultiRegionSurfaceValidationPolicy(profile="general")
            ),
        )


def test_coplanar_face_flip_commits_shared_epoch_and_preserves_content() -> None:
    vertices, faces = _cube()
    prepared = BiomembranePlan(
        faces,
        bending_rigidity=0.0,
        vertex_ids=np.arange(10, 18),
        face_ids=np.arange(30, 42),
        species_diffusivity=(0.1,),
        species_ids=("lipid",),
    ).prepare(vertices)
    state = prepared.state(species_mass=np.arange(8, dtype=np.float64)[:, None] + 1.0)
    proposal = prepared.propose_remesh(state, EdgeFlipProposal((14, 16)))
    assert proposal.surface_result.committed
    assert proposal.event_kind is SurfaceEventKind.FLIP
    evidence = prepared.evaluate_remesh(
        proposal,
        maximum_relative_area_jump=1.0,
        maximum_relative_volume_jump=1.0,
        maximum_relative_energy_jump=1.0,
    )
    result = prepared.commit_remesh(proposal, evidence)
    assert result.committed
    assert result.lineage is not None
    assert result.vertex_transition is not None
    assert result.face_transition is not None
    assert result.prepared.remesh_topology.epoch == prepared.remesh_topology.epoch + 1
    np.testing.assert_allclose(
        np.sum(result.state.species_mass, axis=0),
        np.sum(state.species_mass, axis=0),
        atol=2.0e-13,
    )
