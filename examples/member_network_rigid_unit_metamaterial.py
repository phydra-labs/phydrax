#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static multi-case rigid-unit response matching with independent reanalysis."""

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network

edges = jnp.asarray(((0, 1), (2, 3), (0, 2), (1, 3)), dtype=jnp.int32)
nominal_positions = jnp.asarray(((0.0, 0.0), (0.0, 1.0), (1.0, 0.1), (1.0, 1.1)))
structure = sm.ForceDensityStructure.from_edges(
    edges,
    4,
    2,
    constrained_dofs=jnp.asarray(
        ((True, True), (True, True), (False, False), (False, False))
    ),
)
material = mn.LinearElasticMaterial(20_000.0, 8_000.0, 1.0)
rigid_section = mn.BeamSection(0.8, 0.02, 0.02, 0.01, 0.8, 0.8)
ligament_section = mn.BeamSection(0.08, 0.002, 0.002, 0.001, 0.08, 0.08)
properties = mn.MemberPropertyMap(
    (material,),
    (rigid_section, ligament_section),
    (0, 0, 0, 0),
    (0, 0, 1, 1),
    fabrication_group=(0, 1, 2, 2),
)
reference = mn.MemberReferenceState(structure, nominal_positions)
dofs = mn.MemberDOFLayout(
    structure,
    rotation_constrained=jnp.asarray(((True,), (True,), (False,), (False,))),
)
definition = mn.MemberNetworkDefinition(structure, reference, properties, dofs)
assembly = mn.MemberNetworkAssembly((mn.CorotationalFrameBlock((0, 1, 2, 3)),))
base_problem = mn.MemberNetworkProblem(definition, assembly)


def geometry(design):
    gap, offset = design
    positions = jnp.asarray(((0.0, 0.0), (0.0, 1.0), (gap, offset), (gap, 1.0 + offset)))
    vectors = positions[edges[:, 1]] - positions[edges[:, 0]]
    rest_lengths = jnp.sqrt(jnp.sum(vectors**2, axis=-1))
    return positions, rest_lengths


def realized_problem(design):
    positions, rest_lengths = geometry(design)
    current_reference = eqx.tree_at(
        lambda value: (value.positions, value.rest_lengths),
        base_problem.definition.reference,
        (positions, rest_lengths),
    )
    current_definition = eqx.tree_at(
        lambda value: value.reference,
        base_problem.definition,
        current_reference,
    )
    return eqx.tree_at(
        lambda value: value.definition,
        base_problem,
        current_definition,
    )


vertical = jnp.zeros((4, 2)).at[2:, 1].set(-0.005)
horizontal = jnp.zeros((4, 2)).at[2:, 0].set(-0.005)


def generalized_load(nodal_load):
    return jnp.concatenate(
        (
            structure.reduce(nodal_load),
            jnp.zeros((dofs.rotation_size,), dtype=nodal_load.dtype),
        )
    )


load_cases = (
    sm.LoadCase(
        generalized_load(vertical),
        objective=lambda state, _design, case, reference_state: jnp.vdot(
            case.load, state - reference_state
        ),
        weight=1.0,
        case_id="vertical-shear",
    ),
    sm.LoadCase(
        generalized_load(horizontal),
        objective=lambda state, _design, case, reference_state: jnp.vdot(
            case.load, state - reference_state
        ),
        weight=0.75,
        case_id="horizontal-compression",
    ),
)
aggregation = sm.Aggregation("weighted_sum")
initial_design = jnp.asarray((1.0, 0.1))


def reference_state(design):
    positions, _ = geometry(design)
    return dofs.reduce(positions, jnp.zeros((4, 1)))


def state_residual(states, design, _args):
    problem = realized_problem(design)
    positions, _ = geometry(design)
    prescribed = structure.prescribed_values(positions)
    prescribed_rotations = dofs.prescribed_rotations(jnp.zeros((4, 1)))
    undeformed = reference_state(design)

    def energy(state):
        kinematics = dofs.expand(state, prescribed, prescribed_rotations)
        return problem.assembly.evaluate(problem.definition, kinematics).energy

    tangent = jax.hessian(energy)(undeformed)
    return jnp.stack(
        tuple(
            tangent @ (states[index] - undeformed) - case.load
            for index, case in enumerate(load_cases)
        )
    )


def responses(states, design):
    current_reference = reference_state(design)
    return jnp.stack(
        tuple(
            case.value(states[index], design, current_reference)
            for index, case in enumerate(load_cases)
        )
    )


state_solver = phx.optim.LeastSquaresStateSolver(
    termination=phx.optim.OptimizationTermination(
        absolute_optimality=1.0e-9,
        relative_optimality=0.0,
        absolute_step=1.0e-13,
        relative_step=0.0,
        maximum_steps=100,
    )
)


sample_state = jnp.broadcast_to(reference_state(initial_design), (2, dofs.reduced_size))
state_only = phx.optim.StateDesignProblem(
    state_residual,
    lambda states, design, _args: aggregation(
        responses(states, design),
        jnp.asarray(tuple(case.weight for case in load_cases)),
    ),
    state_solver=state_solver,
    acceptance_policy=phx.optim.StateAcceptancePolicy(
        state_absolute_tolerance=1.0e-9,
        state_relative_tolerance=1.0e-8,
        accepted_state_statuses=(
            phx.optim.OptimizationStatus.SUCCESS,
            phx.optim.OptimizationStatus.STAGNATION,
        ),
    ),
    problem_id="rigid-unit-sample-response",
).solve_state(initial_design, sample_state)
assert state_only.successful
assert state_only.residual_norm <= 1.0e-9
reference_responses = responses(state_only.state, initial_design)

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
design_problem = phx.optim.StateDesignProblem(
    state_residual,
    lambda states, design, _args: (
        aggregation(
            (responses(states, design) - reference_responses) ** 2,
            jnp.asarray(tuple(case.weight for case in load_cases)),
        )
        + 1.0e-4 * jnp.sum((design - initial_design) ** 2)
    ),
    state_solver=state_solver,
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
optimized = phx.optim.solve_state_design(
    design_problem,
    state_only.state,
    initial_design,
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
assert all(
    value <= 2.0e-6
    for value in design_problem.constraint_values(optimized.state, optimized.design)
)


def refined_reanalysis(design, nodal_load):
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
    refined_edges = jnp.asarray(
        ((0, 1), (2, 3), (0, 4), (4, 2), (1, 5), (5, 3)),
        dtype=jnp.int32,
    )
    refined_structure = sm.ForceDensityStructure.from_edges(
        refined_edges,
        6,
        2,
        constrained_dofs=jnp.asarray(
            (
                (True, True),
                (True, True),
                (False, False),
                (False, False),
                (False, False),
                (False, False),
            )
        ),
    )
    refined_properties = mn.MemberPropertyMap(
        (material,),
        (rigid_section, ligament_section),
        (0,) * 6,
        (0, 0, 1, 1, 1, 1),
    )
    refined_reference = mn.MemberReferenceState(refined_structure, positions)
    refined_dofs = mn.MemberDOFLayout(
        refined_structure,
        rotation_constrained=jnp.asarray(
            ((True,), (True,), (False,), (False,), (False,), (False,))
        ),
    )
    refined_definition = mn.MemberNetworkDefinition(
        refined_structure,
        refined_reference,
        refined_properties,
        refined_dofs,
    )
    refined_assembly = mn.MemberNetworkAssembly(
        (mn.CorotationalFrameBlock(jnp.arange(6)),)
    )
    problem = mn.MemberNetworkProblem(refined_definition, refined_assembly)
    initial = mn.MemberKinematics(positions, jnp.zeros((6, 1)))
    loads = jnp.zeros((6, 2)).at[:4].set(nodal_load)
    inputs = mn.MemberNetworkInputs(
        refined_structure.prescribed_values(positions),
        refined_dofs.prescribed_rotations(initial.rotation_vectors),
        loads,
        jnp.zeros((6, 1)),
        refined_reference.rest_lengths,
    )
    solved = mn.member_network_equilibrium(problem, inputs, initial)
    assert solved.successful
    displacement = solved.state.kinematics.positions - positions
    return jnp.sum(loads * displacement)


reanalysis_responses = jnp.stack(
    tuple(refined_reanalysis(optimized.design, load) for load in (vertical, horizontal))
)
assert jnp.allclose(reanalysis_responses, reference_responses, atol=2.0e-5, rtol=0.15)
print("design", optimized.design)
print("reference responses", reference_responses)
print("refined reanalysis", reanalysis_responses)
