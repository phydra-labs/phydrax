from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network


def _axial_problem(*, cable: bool = False, rest_length: float = 1.0):
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        constrained_dofs=jnp.asarray(((True, True), (False, True))),
    )
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    material = mn.LinearElasticMaterial(100.0, 40.0, 1.0, yield_strength=50.0)
    section = mn.AxialSection(1.0)
    properties = mn.MemberPropertyMap((material,), (section,), (0,), (0,))
    reference = mn.MemberReferenceState(
        structure, positions, rest_lengths=jnp.asarray((rest_length,))
    )
    dofs = mn.MemberDOFLayout(
        structure, rotation_constrained=jnp.ones((2, 1), dtype=bool)
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    law = mn.TensionOnlyCableLaw() if cable else mn.LinearAxialLaw()
    assembly = mn.MemberNetworkAssembly((mn.AxialMemberBlock((0,), law),))
    problem = mn.MemberNetworkProblem(definition, assembly)
    rotations = jnp.zeros((2, 1))
    initial = mn.MemberKinematics(positions, rotations)
    return structure, definition, problem, initial


def _inputs(structure, definition, load, rest_length):
    rotations = jnp.zeros((2, 1))
    return mn.MemberNetworkInputs(
        structure.prescribed_values(definition.reference.positions),
        definition.dofs.prescribed_rotations(rotations),
        jnp.asarray(((0.0, 0.0), (load, 0.0))),
        jnp.zeros((2, 1)),
        jnp.asarray((rest_length,)),
    )


def test_product_state_geometry_composes_euclidean_blocks():
    geometry = phx.metrix.ProductStateGeometry(
        (
            phx.metrix.ProductStateGeometryBlock(
                phx.metrix.EuclideanStateGeometry(), (2,), block_id="position"
            ),
            phx.metrix.ProductStateGeometryBlock(
                phx.metrix.EuclideanStateGeometry(), (1,), block_id="rotation"
            ),
        )
    )
    state = geometry.combine_point((jnp.asarray((1.0, 2.0)), jnp.asarray((0.5,))))
    step = jnp.asarray((0.1, -0.2, 0.3))
    assert bool(geometry.contains(state))
    assert jnp.allclose(geometry.retract(state, step), state + step)
    assert geometry.split_point(state)[0] == pytest.approx(jnp.asarray((1.0, 2.0)))


def test_axial_member_equilibrium_matches_closed_form_and_derivative():
    structure, definition, problem, initial = _axial_problem()
    inputs = _inputs(structure, definition, 10.0, 1.0)
    result = mn.member_network_equilibrium(problem, inputs, initial)
    assert result.successful
    assert result.state.kinematics.positions[1, 0] == pytest.approx(1.1, abs=1.0e-8)
    assert result.state.assembly.axial_force[0] == pytest.approx(10.0, abs=1.0e-8)
    assert result.diagnostics.residual_norm <= 1.0e-8

    plan = mn.plan_member_network(problem, inputs, initial)

    def displacement(load):
        dynamic = _inputs(structure, definition, load, 1.0)
        solved = mn.solve_member_network(
            mn.prepare_member_network(plan, dynamic, initial)
        )
        return solved.state.kinematics.positions[1, 0]

    assert jax.grad(displacement)(jnp.asarray(10.0)) == pytest.approx(0.01, rel=1.0e-5)


def test_tension_only_cable_slackens_and_retensions_with_active_set_evidence():
    structure, definition, problem, initial = _axial_problem(cable=True, rest_length=1.1)
    slack_inputs = _inputs(structure, definition, 0.0, 1.1)
    plan = mn.plan_member_network(problem, slack_inputs, initial)
    slack = mn.solve_cable_slackness(
        mn.prepare_member_network(plan, slack_inputs, initial),
        initial_active=jnp.asarray((True,)),
    )
    assert slack.successful
    assert not bool(slack.final_active[0])
    assert slack.derivative_mode == "ambiguous-active-set"
    assert slack.equilibrium.state.assembly.axial_force[0] == pytest.approx(0.0)

    tension_inputs = _inputs(structure, definition, 5.0, 1.1)
    tension = mn.solve_cable_slackness(
        mn.prepare_member_network(plan, tension_inputs, initial),
        initial_active=slack.final_active,
    )
    assert tension.successful
    assert bool(tension.final_active[0])
    assert tension.equilibrium.state.assembly.axial_force[0] == pytest.approx(
        5.0, abs=1e-7
    )


def test_force_density_bridge_infers_compatible_rest_lengths():
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        3,
        2,
        fixed_nodes=(0, 2),
    )
    positions = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
    loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
    force_problem = sm.ForceDensityProblem(structure, sign_mode="tension")
    force_inputs = sm.ForceDensityInputs(
        jnp.ones((2,)), structure.prescribed_values(positions), loads
    )
    force_result = sm.force_density_equilibrium(force_problem, force_inputs)
    material = mn.LinearElasticMaterial(100.0, 40.0, 1.0)
    properties = mn.MemberPropertyMap(
        (material,), (mn.AxialSection(1.0),), (0, 0), (0, 0)
    )
    assembly = mn.MemberNetworkAssembly((mn.AxialMemberBlock((0, 1)),))
    target, definition, inputs, initial = mn.member_network_from_force_density(
        force_result, structure, properties, assembly
    )
    lengths = force_result.state.member_lengths
    expected = 100.0 * lengths / (100.0 + target.axial_forces)
    assert jnp.allclose(definition.reference.rest_lengths, expected)
    constitutive = mn.member_network_equilibrium(
        mn.MemberNetworkProblem(definition, assembly), inputs, initial
    )
    assert constitutive.successful
    assert jnp.allclose(
        constitutive.state.assembly.axial_force,
        target.axial_forces,
        atol=1.0e-7,
    )
    policy = mn.PrestressFabricationPolicy(
        0.5 * expected,
        1.5 * expected,
        -jnp.ones_like(expected),
        jnp.ones_like(expected),
        require_stability=True,
        require_sequence=True,
    )
    realizability = mn.assess_prestress_realizability(
        target,
        definition,
        mn.LinearAxialLaw(),
        policy,
        stability_valid=True,
        sequence_valid=True,
        member_roles="tension-only",
    )
    assert realizability.successful
    assert realizability.constitutive_valid
    assert realizability.equilibrium_valid
