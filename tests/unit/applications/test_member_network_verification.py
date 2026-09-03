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


def _axial_modal_network(*, repeated: bool = False):
    if repeated:
        edges = jnp.asarray(((0, 1), (0, 2)), dtype=jnp.int32)
        node_count = 3
        constraints = jnp.asarray(((True, True), (False, True), (True, False)))
        positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    else:
        edges = jnp.asarray(((0, 1),), dtype=jnp.int32)
        node_count = 2
        constraints = jnp.asarray(((True, True), (False, False)))
        positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    structure = sm.ForceDensityStructure.from_edges(
        edges,
        node_count,
        2,
        constrained_dofs=constraints,
    )
    material = mn.LinearElasticMaterial(100.0, 40.0, 1.0)
    properties = mn.MemberPropertyMap(
        (material,),
        (mn.AxialSection(1.0),),
        (0,) * edges.shape[0],
        (0,) * edges.shape[0],
    )
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure,
        rotation_constrained=jnp.ones((node_count, 1), dtype=bool),
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    assembly = mn.MemberNetworkAssembly(
        (mn.AxialMemberBlock(jnp.arange(edges.shape[0])),)
    )
    problem = mn.MemberNetworkProblem(definition, assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((node_count, 1)))
    inputs = mn.MemberNetworkInputs(
        structure.prescribed_values(positions),
        dofs.prescribed_rotations(initial.rotation_vectors),
        jnp.zeros_like(positions),
        jnp.zeros((node_count, 1)),
        reference.rest_lengths,
    )
    return problem, inputs, initial


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
    stability = mn.tangent_stability(
        problem,
        result,
        mass=jnp.ones((1, 1)),
    )
    continuation = mn.member_network_continuation_problem(problem, inputs)
    assert stability.stable
    assert stability.minimum_eigenvalue > 0.0
    assert stability.modal_valid
    assert stability.eigen_residual < 1.0e-8
    assert stability.mass_orthogonality_error < 1.0e-8
    assert continuation.problem_id.endswith("load-continuation")


def test_tangent_stability_cannot_mix_equilibrium_and_independent_inputs():
    structure, definition, problem, initial = _axial_network()
    accepted_inputs = _inputs(structure, definition, 5.0)
    equilibrium = mn.member_network_equilibrium(
        problem,
        accepted_inputs,
        initial,
    )
    mismatched_inputs = eqx.tree_at(
        lambda value: value.rest_lengths,
        accepted_inputs,
        1.1 * accepted_inputs.rest_lengths,
    )
    assert jnp.array_equal(equilibrium.inputs.rest_lengths, accepted_inputs.rest_lengths)
    with pytest.raises(TypeError):
        mn.tangent_stability(problem, mismatched_inputs, equilibrium)


def test_modal_stability_handles_rigid_modes_and_rejects_nonpositive_mass():
    problem, inputs, initial = _axial_modal_network()
    equilibrium = mn.member_network_equilibrium(
        problem,
        inputs,
        initial,
        nonlinear_method=phx.nonlinear.NewtonKrylov(),
    )
    modal = mn.tangent_stability(
        problem,
        equilibrium,
        mass=jnp.eye(2),
        rigid_mode_count=1,
        differentiate_eigenvalues=True,
    )
    assert modal.tangent.shape == (2, 2)
    assert modal.equilibrium_accepted
    assert modal.physical_tangent
    assert modal.minimum_mass_eigenvalue > 0.0
    assert modal.mass_positive
    assert modal.rigid_modes_valid
    assert modal.modal_valid
    assert modal.mode_gap_valid
    assert modal.mode_derivatives_available
    assert modal.angular_frequencies[0] == pytest.approx(0.0, abs=1.0e-8)
    assert modal.eigen_residual < 1.0e-8
    assert modal.mass_orthogonality_error < 1.0e-8

    with pytest.raises(
        (ValueError, eqx.EquinoxRuntimeError),
        match="positive-definite",
    ):
        invalid = mn.tangent_stability(
            problem,
            equilibrium,
            mass=jnp.diag(jnp.asarray((1.0, 0.0))),
            rigid_mode_count=1,
        )
        invalid.eigenvalues.block_until_ready()


def test_modal_tracking_marks_low_overlap_and_crossings_ambiguous():
    problem, inputs, initial = _axial_modal_network()
    equilibrium = mn.member_network_equilibrium(
        problem,
        inputs,
        initial,
        nonlinear_method=phx.nonlinear.NewtonKrylov(),
    )
    reference = mn.tangent_stability(
        problem,
        equilibrium,
        mass=jnp.eye(2),
        rigid_mode_count=1,
    )
    tracking_plan = phx.linalg.eigen.plan_hermitian_eigenspace_tracking(
        reference.eigenvalues
    )
    rotation = jnp.asarray(((1.0, -1.0), (1.0, 1.0))) / jnp.sqrt(2.0)
    ambiguous = mn.tangent_stability(
        problem,
        equilibrium,
        mass=jnp.eye(2),
        rigid_mode_count=1,
        tracking_plan=tracking_plan,
        reference_modes=reference.modes @ rotation,
        differentiate_eigenvalues=True,
    )
    assert ambiguous.tracking is not None
    assert not ambiguous.tracking.successful
    assert not ambiguous.modal_valid
    assert not ambiguous.mode_derivatives_available
    assert ambiguous.tracking.diagnostics.assignment_margin <= 1.0e-8

    repeated_problem, repeated_inputs, repeated_initial = _axial_modal_network(
        repeated=True
    )
    repeated_equilibrium = mn.member_network_equilibrium(
        repeated_problem,
        repeated_inputs,
        repeated_initial,
    )
    crossing = mn.tangent_stability(
        repeated_problem,
        repeated_equilibrium,
        mass=jnp.eye(2),
        differentiate_eigenvalues=True,
    )
    assert not crossing.mode_gap_valid
    assert crossing.stable
    assert not crossing.mode_derivatives_available
    assert not crossing.modal_valid


def test_tangent_stability_does_not_certify_an_unaccepted_equilibrium():
    structure, definition, problem, initial = _axial_network()
    inputs = _inputs(structure, definition, 5.0)
    equilibrium = mn.member_network_equilibrium(problem, inputs, initial)
    rejected = eqx.tree_at(
        lambda value: value.status,
        equilibrium,
        jnp.asarray(int(mn.MemberNetworkStatus.NONLINEAR_SOLVE_FAILED)),
    )
    stability = mn.tangent_stability(
        problem,
        rejected,
        mass=jnp.ones((1, 1)),
    )
    assert not stability.equilibrium_accepted
    assert not stability.physical_tangent
    assert not stability.modal_valid


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
