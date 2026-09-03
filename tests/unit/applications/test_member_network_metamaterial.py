from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network


def _beam_properties(member_count: int):
    material = mn.LinearElasticMaterial(2_000.0, 800.0, 1.0)
    section = mn.BeamSection(0.1, 0.002, 0.002, 0.001, 0.08, 0.08)
    return mn.MemberPropertyMap(
        (material,),
        (section,),
        (0,) * member_count,
        (0,) * member_count,
    )


def _ligament_comparison_definition():
    edges = jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32)
    positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, -1.0, 0.0),
        )
    )
    structure = sm.ForceDensityStructure.from_edges(
        edges,
        5,
        3,
        fixed_nodes=(0, 1, 2, 3, 4),
    )
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure,
        rotation_constrained=jnp.ones((5, 3), dtype=bool),
    )
    definition = mn.MemberNetworkDefinition(
        structure,
        reference,
        _beam_properties(2),
        dofs,
    )
    blocks = (
        mn.CorotationalFrameBlock((0, 1), block_id="ligament-frame"),
        mn.DiscreteRodBlock((0, 1, 2), (0, 1), block_id="ligament-rod"),
        mn.HingeBendingBlock(
            ((0, 1, 3, 4),),
            (5.0,),
            (0.0,),
            block_id="ligament-hinge",
        ),
    )
    return definition, positions, blocks


@pytest.mark.parametrize("block_index", (0, 1, 2))
def test_existing_ligament_blocks_supply_energy_force_tangent_and_geometry_derivatives(
    block_index,
):
    definition, positions, blocks = _ligament_comparison_definition()
    block = blocks[block_index]
    rotations = jnp.asarray(
        (
            (0.01, -0.02, 0.03),
            (0.02, 0.01, -0.01),
            (-0.01, 0.02, 0.01),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    )
    displaced = positions.at[1, 1].set(0.04).at[4, 2].set(0.08)

    def energy(flat_positions):
        kinematics = mn.MemberKinematics(
            flat_positions.reshape(positions.shape), rotations
        )
        return block.evaluate(definition, kinematics).energy

    point = displaced.reshape((-1,))
    force = jax.grad(energy)(point)
    tangent = jax.jacfwd(jax.grad(energy))(point)
    hessian = jax.hessian(energy)(point)
    direction = jnp.linspace(-0.3, 0.4, point.size)
    direction = direction / jnp.sqrt(jnp.sum(direction**2))
    step = 1.0e-5
    finite_force = (
        energy(point + step * direction) - energy(point - step * direction)
    ) / (2.0 * step)
    finite_tangent = (
        jax.grad(energy)(point + step * direction)
        - jax.grad(energy)(point - step * direction)
    ) / (2.0 * step)
    mode_quantity = jnp.vdot(direction, tangent @ direction)

    assert jnp.all(jnp.isfinite(force))
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.vdot(force, direction) == pytest.approx(finite_force, rel=2.0e-4)
    assert jnp.allclose(tangent, hessian, atol=1.0e-9, rtol=1.0e-9)
    assert jnp.allclose(tangent @ direction, finite_tangent, atol=2.0e-4, rtol=2.0e-4)
    assert jnp.sum(tangent**2) > 0.0
    assert jnp.isfinite(mode_quantity)


def test_existing_ligament_blocks_are_objective_and_frame_route_needs_no_new_block():
    definition, positions, blocks = _ligament_comparison_definition()
    rotation_vector = jnp.asarray((0.35, -0.2, 0.15))
    rotation = mn.rotation_vector_matrix(rotation_vector)
    translated = positions @ rotation.T + jnp.asarray((1.2, -0.7, 0.4))
    nodal_rotations = jnp.broadcast_to(rotation_vector, (positions.shape[0], 3))
    rigid = mn.MemberKinematics(translated, nodal_rotations)

    for block in blocks:
        assert block.evaluate(definition, rigid).energy == pytest.approx(0.0, abs=2.0e-8)
        assert block.member_indices.size == jnp.unique(block.member_indices).size

    selected = mn.MemberNetworkAssembly((blocks[0],))
    selected_members = selected.blocks[0].member_indices
    assert selected_members.size == definition.structure.member_count
    assert jnp.unique(selected_members).size == selected_members.size


def _rigid_unit_base():
    edges = jnp.asarray(((0, 1), (2, 3), (0, 2), (1, 3)), dtype=jnp.int32)
    positions = jnp.asarray(((0.0, 0.0), (0.0, 1.0), (1.0, 0.1), (1.0, 1.1)))
    constraints = jnp.asarray(
        ((True, True), (True, True), (False, False), (False, False))
    )
    structure = sm.ForceDensityStructure.from_edges(
        edges,
        4,
        2,
        constrained_dofs=constraints,
    )
    material = mn.LinearElasticMaterial(20_000.0, 8_000.0, 1.0)
    rigid = mn.BeamSection(0.8, 0.02, 0.02, 0.01, 0.8, 0.8)
    ligament = mn.BeamSection(0.08, 0.002, 0.002, 0.001, 0.08, 0.08)
    properties = mn.MemberPropertyMap(
        (material,),
        (rigid, ligament),
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        fabrication_group=(0, 1, 2, 2),
    )
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure,
        rotation_constrained=jnp.asarray(((True,), (True,), (False,), (False,))),
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    assembly = mn.MemberNetworkAssembly((mn.CorotationalFrameBlock((0, 1, 2, 3)),))
    return mn.MemberNetworkProblem(definition, assembly), positions


def _designed_geometry(design):
    gap, offset = design
    positions = jnp.asarray(((0.0, 0.0), (0.0, 1.0), (gap, offset), (gap, 1.0 + offset)))
    edges = jnp.asarray(((0, 1), (2, 3), (0, 2), (1, 3)), dtype=jnp.int32)
    vectors = positions[edges[:, 1]] - positions[edges[:, 0]]
    lengths = jnp.sqrt(jnp.sum(vectors**2, axis=-1))
    return positions, lengths


def _realized_problem(base_problem, design):
    positions, lengths = _designed_geometry(design)
    reference = eqx.tree_at(
        lambda value: (value.positions, value.rest_lengths),
        base_problem.definition.reference,
        (positions, lengths),
    )
    definition = eqx.tree_at(
        lambda value: value.reference,
        base_problem.definition,
        reference,
    )
    return eqx.tree_at(lambda value: value.definition, base_problem, definition)


def _nodal_loads(node_count: int):
    vertical = jnp.zeros((node_count, 2)).at[2, 1].set(-0.005).at[3, 1].set(-0.005)
    horizontal = jnp.zeros((node_count, 2)).at[2, 0].set(-0.005).at[3, 0].set(-0.005)
    return vertical, horizontal


def _member_inputs(problem, design, nodal_forces):
    positions, lengths = _designed_geometry(design)
    rotations = jnp.zeros((problem.definition.structure.node_count, 1))
    return mn.MemberNetworkInputs(
        problem.definition.structure.prescribed_values(positions),
        problem.definition.dofs.prescribed_rotations(rotations),
        nodal_forces,
        jnp.zeros_like(rotations),
        lengths,
    )


def _state_design_components(base_problem, load_cases, aggregation, target=None):
    dofs = base_problem.definition.dofs

    def reference_state(design):
        positions, _ = _designed_geometry(design)
        return dofs.reduce(positions, jnp.zeros((4, 1)))

    def residual(states, design, _args):
        realized = _realized_problem(base_problem, design)
        positions, _ = _designed_geometry(design)
        prescribed = realized.definition.structure.prescribed_values(positions)
        prescribed_rotations = realized.definition.dofs.prescribed_rotations(
            jnp.zeros((4, 1))
        )
        undeformed = reference_state(design)

        def energy(current):
            kinematics = realized.definition.dofs.expand(
                current,
                prescribed,
                prescribed_rotations,
            )
            return realized.assembly.evaluate(realized.definition, kinematics).energy

        tangent = jax.hessian(energy)(undeformed)
        return jnp.stack(
            tuple(
                tangent @ (states[index] - undeformed) - case.load
                for index, case in enumerate(load_cases)
            )
        )

    def responses(states, design):
        reference = reference_state(design)
        return jnp.stack(
            tuple(
                case.value(states[index], design, reference)
                for index, case in enumerate(load_cases)
            )
        )

    def objective(states, design, _args):
        values = responses(states, design)
        weights = jnp.asarray(tuple(case.weight for case in load_cases))
        if target is None:
            return aggregation(values, weights)
        mismatch = (values - target) ** 2
        return aggregation(mismatch, weights) + 1.0e-4 * jnp.sum(
            (design - jnp.asarray((1.0, 0.1))) ** 2
        )

    constraints = (
        phx.optim.StateDesignConstraint(
            lambda _state, design, _args: 0.8 - design[0],
            upper=0.0,
            constraint_id="minimum-unit-gap",
            depends_on_state=False,
        ),
        phx.optim.StateDesignConstraint(
            lambda _state, design, _args: design[1] ** 2 - 0.25**2,
            upper=0.0,
            constraint_id="ligament-offset-clearance",
            depends_on_state=False,
        ),
    )
    problem = phx.optim.StateDesignProblem(
        residual,
        objective,
        state_solver=phx.optim.LeastSquaresStateSolver(
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1.0e-9,
                relative_optimality=0.0,
                absolute_step=1.0e-13,
                relative_step=0.0,
                maximum_steps=100,
            )
        ),
        acceptance_policy=phx.optim.StateAcceptancePolicy(
            state_absolute_tolerance=1.0e-9,
            state_relative_tolerance=1.0e-8,
            accepted_state_statuses=(
                phx.optim.OptimizationStatus.SUCCESS,
                phx.optim.OptimizationStatus.STAGNATION,
            ),
        ),
        design_bounds=phx.optim.Bounds(
            jnp.asarray((0.7, -0.3)),
            jnp.asarray((1.4, 0.3)),
        ),
        constraints=constraints,
        problem_id="rigid-unit-reference-response",
    )
    return problem, reference_state, responses


def _refined_reanalysis(design, nodal_load):
    gap, offset = design
    positions = jnp.asarray(
        (
            (0.0, 0.0),
            (0.0, 1.0),
            (gap, offset),
            (gap, 1.0 + offset),
            (0.5 * gap, 0.5 * offset),
            (0.5 * gap, 1.0 + 0.5 * offset),
        )
    )
    edges = jnp.asarray(
        ((0, 1), (2, 3), (0, 4), (4, 2), (1, 5), (5, 3)),
        dtype=jnp.int32,
    )
    constraints = jnp.asarray(
        (
            (True, True),
            (True, True),
            (False, False),
            (False, False),
            (False, False),
            (False, False),
        )
    )
    structure = sm.ForceDensityStructure.from_edges(
        edges,
        6,
        2,
        constrained_dofs=constraints,
    )
    material = mn.LinearElasticMaterial(20_000.0, 8_000.0, 1.0)
    rigid = mn.BeamSection(0.8, 0.02, 0.02, 0.01, 0.8, 0.8)
    ligament = mn.BeamSection(0.08, 0.002, 0.002, 0.001, 0.08, 0.08)
    properties = mn.MemberPropertyMap(
        (material,),
        (rigid, ligament),
        (0,) * 6,
        (0, 0, 1, 1, 1, 1),
    )
    reference = mn.MemberReferenceState(structure, positions)
    dofs = mn.MemberDOFLayout(
        structure,
        rotation_constrained=jnp.asarray(
            ((True,), (True,), (False,), (False,), (False,), (False,))
        ),
    )
    definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
    assembly = mn.MemberNetworkAssembly((mn.CorotationalFrameBlock(jnp.arange(6)),))
    problem = mn.MemberNetworkProblem(definition, assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((6, 1)))
    loads = jnp.zeros((6, 2)).at[:4].set(nodal_load)
    inputs = mn.MemberNetworkInputs(
        structure.prescribed_values(positions),
        dofs.prescribed_rotations(initial.rotation_vectors),
        loads,
        jnp.zeros((6, 1)),
        reference.rest_lengths,
    )
    solved = mn.member_network_equilibrium(problem, inputs, initial)
    displacement = solved.state.kinematics.positions - positions
    return solved, jnp.sum(loads * displacement)


def test_rigid_unit_static_multicase_state_design_mma_and_refined_reanalysis():
    base_problem, _ = _rigid_unit_base()
    design = jnp.asarray((1.0, 0.1))
    nodal_loads = _nodal_loads(4)
    generalized_loads = tuple(
        jnp.concatenate(
            (
                base_problem.definition.structure.reduce(load),
                jnp.zeros(
                    (base_problem.definition.dofs.rotation_size,),
                    dtype=load.dtype,
                ),
            )
        )
        for load in nodal_loads
    )
    load_cases = tuple(
        sm.LoadCase(
            load,
            objective=lambda state, _design, case, reference: jnp.vdot(
                case.load, state - reference
            ),
            weight=weight,
            case_id=case_id,
        )
        for load, weight, case_id in zip(
            generalized_loads,
            (1.0, 0.75),
            ("vertical-shear", "horizontal-compression"),
            strict=True,
        )
    )
    aggregation = sm.Aggregation("weighted_sum")
    provisional, reference_state, responses = _state_design_components(
        base_problem,
        load_cases,
        aggregation,
    )
    initial_state = jnp.broadcast_to(
        reference_state(design), (2, reference_state(design).size)
    )
    state = provisional.solve_state(design, initial_state)
    assert state.successful
    assert state.residual_norm <= 1.0e-9

    direct_states = []
    direct_responses = []
    for nodal_load, generalized_load in zip(nodal_loads, generalized_loads, strict=True):
        realized = _realized_problem(base_problem, design)
        inputs = _member_inputs(realized, design, nodal_load)
        positions, _ = _designed_geometry(design)
        initial = mn.MemberKinematics(positions, jnp.zeros((4, 1)))
        solved = mn.member_network_equilibrium(realized, inputs, initial)
        assert solved.successful
        if not direct_states:

            def position_energy(current_positions):
                kinematics = mn.MemberKinematics(
                    current_positions,
                    solved.state.kinematics.rotation_vectors,
                )
                return realized.assembly.evaluate(
                    realized.definition,
                    kinematics,
                ).energy

            position_gradient = jax.grad(position_energy)(
                solved.state.kinematics.positions
            )
            assert jnp.allclose(
                solved.state.internal_forces,
                position_gradient,
                atol=1.0e-8,
                rtol=1.0e-8,
            )

            def reduced_energy(current):
                kinematics = realized.definition.dofs.expand(
                    current,
                    inputs.prescribed_positions,
                    inputs.prescribed_rotations,
                )
                return realized.assembly.evaluate(
                    realized.definition,
                    kinematics,
                ).energy

            assert jnp.allclose(
                mn.member_network_tangent(
                    realized,
                    inputs,
                    solved.state.kinematics,
                ),
                jax.hessian(reduced_energy)(
                    realized.definition.dofs.reduce(
                        solved.state.kinematics.positions,
                        solved.state.kinematics.rotation_vectors,
                    )
                ),
                atol=1.0e-8,
                rtol=1.0e-8,
            )
        reduced = realized.definition.dofs.reduce(
            solved.state.kinematics.positions,
            solved.state.kinematics.rotation_vectors,
        )
        direct_states.append(reduced)
        direct_responses.append(
            jnp.vdot(generalized_load, reduced - reference_state(design))
        )
    target = responses(state.state, design)
    assert jnp.allclose(
        jnp.asarray(direct_responses),
        target,
        atol=1.0e-9,
        rtol=2.0e-2,
    )

    design_problem, _, _ = _state_design_components(
        base_problem,
        load_cases,
        aggregation,
        target,
    )
    optimized = phx.optim.solve_state_design(
        design_problem,
        state.state,
        design,
        method=phx.optim.ReducedMMA(),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=2.0e-6,
            relative_optimality=0.0,
            absolute_step=1.0e-11,
            relative_step=0.0,
            maximum_steps=8,
        ),
    )
    assert optimized.successful
    assert optimized.state_acceptance.accepted
    constraints = design_problem.constraint_values(optimized.state, optimized.design)
    assert all(value <= 2.0e-6 for value in constraints)

    refined_responses = []
    for nodal_load in nodal_loads:
        refined, response = _refined_reanalysis(optimized.design, nodal_load)
        assert refined.successful
        refined_responses.append(response)
    assert jnp.allclose(
        jnp.asarray(refined_responses),
        target,
        atol=2.0e-5,
        rtol=0.15,
    )
