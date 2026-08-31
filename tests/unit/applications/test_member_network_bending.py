from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network


def _beam_properties(count):
    material = mn.LinearElasticMaterial(1000.0, 400.0, 1.0)
    section = mn.BeamSection(1.0, 1.0, 1.0, 0.5, 100.0, 100.0)
    return mn.MemberPropertyMap((material,), (section,), (0,) * count, (0,) * count)


def test_corotational_beam_is_objective_under_large_rigid_rotation():
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        fixed_nodes=(0, 1),
    )
    reference_positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    reference = mn.MemberReferenceState(structure, reference_positions)
    dofs = mn.MemberDOFLayout(
        structure, rotation_constrained=jnp.ones((2, 1), dtype=bool)
    )
    definition = mn.MemberNetworkDefinition(
        structure, reference, _beam_properties(1), dofs
    )
    block = mn.CorotationalFrameBlock((0,))
    rotated = mn.MemberKinematics(
        jnp.asarray(((0.0, 0.0), (0.0, 1.0))),
        jnp.full((2, 1), jnp.pi / 2.0),
    )
    evaluated = block.evaluate(definition, rotated)
    assert evaluated.valid
    assert evaluated.energy == pytest.approx(0.0, abs=1.0e-9)
    assert evaluated.axial_force[0] == pytest.approx(0.0, abs=1.0e-9)
    assert evaluated.bending_moment[0, 0] == pytest.approx(0.0, abs=1.0e-9)


def test_corotational_cantilever_matches_small_deflection_limit():
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        constrained_dofs=jnp.asarray(((True, True), (False, False))),
    )
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure,
        rotation_constrained=jnp.asarray(((True,), (False,))),
    )
    definition = mn.MemberNetworkDefinition(
        structure, reference, _beam_properties(1), dofs
    )
    assembly = mn.MemberNetworkAssembly((mn.CorotationalFrameBlock((0,)),))
    problem = mn.MemberNetworkProblem(definition, assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((2, 1)))
    inputs = mn.MemberNetworkInputs(
        structure.prescribed_values(positions),
        dofs.prescribed_rotations(initial.rotation_vectors),
        jnp.asarray(((0.0, 0.0), (0.0, -1.0))),
        jnp.zeros((2, 1)),
        jnp.asarray((1.0,)),
    )
    result = mn.member_network_equilibrium(problem, inputs, initial)
    assert result.successful
    assert result.state.kinematics.positions[1, 1] < 0.0
    assert result.state.kinematics.positions[1, 1] == pytest.approx(
        -1.0 / 3000.0,
        rel=0.12,
    )
    assert result.state.assembly.bending_moment[0, 0] != 0.0


def test_discrete_rod_bending_and_twist_energy_detect_deformation():
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        3,
        3,
        fixed_nodes=(0, 1, 2),
    )
    positions = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure, rotation_constrained=jnp.ones((3, 3), dtype=bool)
    )
    definition = mn.MemberNetworkDefinition(
        structure, reference, _beam_properties(2), dofs
    )
    rod = mn.DiscreteRodBlock((0, 1, 2), (0, 1))
    rest = rod.evaluate(definition, mn.MemberKinematics(positions, jnp.zeros((3, 3))))
    bent_positions = positions.at[1, 1].set(0.2)
    bent_rotations = jnp.zeros((3, 3)).at[2, 0].set(0.1)
    bent = rod.evaluate(definition, mn.MemberKinematics(bent_positions, bent_rotations))
    assert rest.energy == pytest.approx(0.0, abs=1.0e-10)
    assert bent.valid
    assert bent.energy > 0.0
    assert jnp.any(jnp.abs(bent.bending_moment) > 0.0)


def test_hinge_bending_energy_is_zero_at_rest_and_positive_when_folded():
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (1, 2), (2, 0), (1, 3), (3, 0)), dtype=jnp.int32),
        4,
        3,
        fixed_nodes=(0, 1, 2, 3),
    )
    positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0),
        )
    )
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure, rotation_constrained=jnp.ones((4, 3), dtype=bool)
    )
    definition = mn.MemberNetworkDefinition(
        structure, reference, _beam_properties(5), dofs
    )
    hinge = mn.HingeBendingBlock(((0, 1, 2, 3),), (10.0,), (0.0,))
    flat = hinge.evaluate(definition, mn.MemberKinematics(positions, jnp.zeros((4, 3))))
    folded = hinge.evaluate(
        definition,
        mn.MemberKinematics(positions.at[3, 2].set(0.25), jnp.zeros((4, 3))),
    )
    assert flat.energy == pytest.approx(0.0, abs=1.0e-10)
    assert folded.energy > 0.0
