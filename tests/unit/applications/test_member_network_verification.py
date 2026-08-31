from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import pytest

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network


def _axial_network():
    structure = sm.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        constrained_dofs=jnp.asarray(((True, True), (False, True))),
        node_ids=("base", "tip"),
        member_ids=("column",),
    )
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    material = mn.LinearElasticMaterial(
        100.0,
        40.0,
        2.0,
        tension_allowable=20.0,
        compression_allowable=20.0,
    )
    section = mn.BeamSection(1.0, 1.0, 1.0, 0.5, 1.0, 1.0)
    properties = mn.MemberPropertyMap((material,), (section,), (0,), (0,))
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure, rotation_constrained=jnp.ones((2, 1), dtype=bool)
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    assembly = mn.MemberNetworkAssembly((mn.AxialMemberBlock((0,)),))
    problem = mn.MemberNetworkProblem(definition, assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((2, 1)))
    return structure, definition, problem, initial


def _inputs(structure, definition, load):
    return mn.MemberNetworkInputs(
        structure.prescribed_values(definition.reference.positions),
        definition.dofs.prescribed_rotations(definition.reference.rotation_vectors),
        jnp.asarray(((0.0, 0.0), (load, 0.0))),
        jnp.zeros((2, 1)),
        definition.reference.rest_lengths,
    )


def test_local_and_generalized_buckling_match_closed_forms():
    structure, definition, _, _ = _axial_network()
    local = mn.local_euler_buckling(
        definition,
        jnp.asarray((-50.0,)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
    )
    assert local.critical_load[0] == pytest.approx(jnp.pi**2 * 100.0)
    assert local.utilization[0] == pytest.approx(50.0 / (jnp.pi**2 * 100.0))
    assert local.valid[0]

    linear = mn.linearized_buckling(
        jnp.diag(jnp.asarray((10.0, 20.0))),
        jnp.diag(jnp.asarray((-2.0, -4.0))),
        proportional_load_verified=True,
        conservative_verified=True,
    )
    assert linear.successful
    assert linear.critical_factor == pytest.approx(5.0)
    assert jnp.allclose(linear.load_factors, 5.0)


def test_tangent_stability_and_continuation_are_native():
    structure, definition, problem, initial = _axial_network()
    inputs = _inputs(structure, definition, 5.0)
    result = mn.member_network_equilibrium(problem, inputs, initial)
    stability = mn.tangent_stability(problem, inputs, result.state.kinematics)
    continuation = mn.member_network_continuation_problem(problem, inputs)
    assert stability.stable
    assert stability.minimum_eigenvalue > 0.0
    assert continuation.problem_id.endswith("load-continuation")


def test_construction_sequence_transfers_state_and_load_operations():
    structure, definition, problem, initial = _axial_network()
    empty = _inputs(structure, definition, 0.0)
    loaded = _inputs(structure, definition, 5.0)
    stage_one = mn.ConstructionStage(
        problem,
        empty,
        (mn.InstallationRule.DECLARED_STRESS_FREE_LENGTH,),
        stage_id="install",
    )
    stage_two = mn.ConstructionStage(
        problem,
        loaded,
        (mn.InstallationRule.DECLARED_STRESS_FREE_LENGTH,),
        load_operation=mn.LoadOperation.ADD,
        stage_id="load",
    )
    plan = mn.plan_construction_sequence((stage_one, stage_two), initial)
    result = mn.solve_construction_sequence(plan, initial)
    assert result.successful
    assert len(result.stages) == 2
    assert result.stages[0].equilibrium.state.kinematics.positions[1, 0] == pytest.approx(
        1.0
    )
    assert result.stages[1].equilibrium.state.kinematics.positions[1, 0] == pytest.approx(
        1.05
    )
    assert result.checkpoint.completed_stage == 1


def test_sizing_catalog_and_verification_report_governing_evidence():
    structure, definition, problem, initial = _axial_network()
    inputs = _inputs(structure, definition, 5.0)
    equilibrium = mn.member_network_equilibrium(problem, inputs, initial)
    local = mn.local_euler_buckling(
        definition,
        equilibrium.state.assembly.axial_force,
        jnp.ones((1,)),
        jnp.asarray((1.05,)),
    )
    sizing = mn.evaluate_member_sizing(definition, equilibrium, local_buckling=local)
    assert sizing.valid
    assert sizing.mass == pytest.approx(2.1)
    assert sizing.axial_stress[0] == pytest.approx(5.0)
    assert int(sizing.governing_member) == 0

    selected = mn.select_catalog_member_sizing(
        ("light", "heavy"),
        lambda index: (
            sizing
            if index == 0
            else eqx.tree_at(lambda value: value.mass, sizing, 2.0 * sizing.mass)
        ),
    )
    assert selected.successful
    assert selected.selected_label == "light"

    certified = mn.verify_member_structure(
        equilibrium=equilibrium,
        sizing=sizing,
        local_buckling=local,
        required=("equilibrium", "sizing", "local_buckling"),
    )
    incomplete = mn.verify_member_structure(
        equilibrium=equilibrium,
        required=("equilibrium", "prestress"),
    )
    assert certified.successful
    assert incomplete.verdict == int(mn.StructuralEvidenceVerdict.INCOMPLETE)
    assert incomplete.missing == ("prestress",)
