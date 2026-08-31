from __future__ import annotations

import jax.numpy as jnp
import pytest

import phydrax as phx


fd = phx.applications.solid_mechanics


def _design_setup():
    structure = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        3,
        2,
        fixed_nodes=(0, 2),
    )
    positions = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
    prescribed = structure.prescribed_values(positions)
    loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
    sample = fd.ForceDensityInputs(jnp.ones((2,)), prescribed, loads)
    equilibrium = fd.ForceDensityProblem(
        structure,
        sign_mode="tension",
        problem_id="inverse-cable",
    )
    plan = fd.plan_force_density(equilibrium, sample)

    def decode(design, _):
        return fd.ForceDensityInputs(
            jnp.repeat(design.reshape(()), 2),
            prescribed,
            loads,
        )

    return equilibrium, plan, sample, decode


def test_force_density_reduced_design_hits_target_shape_and_recertifies_state():
    _, plan, _, decode = _design_setup()
    problem = fd.ForceDensityDesignProblem(
        plan,
        decode,
        lambda state, design, _: (state.positions[1, 1] + 0.25) ** 2,
        design_bounds=phx.optim.Bounds(0.2, 5.0),
        problem_id="target-height-cable",
    )
    result = fd.solve_force_density_design(
        problem,
        jnp.asarray(1.0),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-6,
            relative_optimality=0.0,
            absolute_step=1.0e-11,
            relative_step=0.0,
            maximum_steps=400,
        ),
    )
    assert result.successful
    assert float(result.state_design.design) == pytest.approx(2.0, abs=2.0e-3)
    assert result.equilibrium.state.positions[1, 1] == pytest.approx(-0.25, abs=3.0e-4)
    assert result.equilibrium.diagnostics.free_residual_norm <= 1.0e-9


def test_force_density_design_decoder_can_move_supports_and_loads():
    equilibrium, plan, sample, _ = _design_setup()

    def decode(design, _):
        prescribed = sample.prescribed_values.at[1].set(design[1])
        loads = sample.load_parameters.at[1, 1].set(design[2])
        return fd.ForceDensityInputs(jnp.repeat(design[0], 2), prescribed, loads)

    problem = fd.ForceDensityDesignProblem(
        plan,
        decode,
        lambda state, design, _: jnp.sum(state.positions**2),
    )
    design = jnp.asarray((2.0, 0.1, -0.8))
    inputs = problem.inputs(design)
    state_problem = problem.as_state_design_problem()
    solved = equilibrium.structure.reduce(
        fd.force_density_equilibrium(equilibrium, inputs).state.positions
    )
    state = problem.physical_state(solved, design)

    assert inputs.prescribed_values[1] == pytest.approx(0.1)
    assert inputs.load_parameters[1, 1] == pytest.approx(-0.8)
    assert state.positions[0, 1] == pytest.approx(0.1)
    assert jnp.allclose(state_problem.residual(solved, design), 0.0, atol=1.0e-10)


def test_force_density_physical_constraints_lower_to_structured_state_design():
    _, plan, _, decode = _design_setup()
    length_constraint = fd.ForceDensityDesignConstraint(
        lambda state, design, _: state.member_lengths,
        lower=jnp.asarray((1.0, 1.0)),
        upper=jnp.asarray((2.0, 2.0)),
        constraint_id="member-lengths",
    )
    problem = fd.ForceDensityDesignProblem(
        plan,
        decode,
        lambda state, design, _: fd.force_density_load_path(state),
        design_bounds=phx.optim.Bounds(0.2, 5.0),
        constraints=(length_constraint,),
        problem_id="bounded-member-cable",
    )
    inputs = problem.inputs(jnp.asarray(1.0))
    initial = fd.force_density_equilibrium(plan.problem, inputs)
    reduced = plan.problem.structure.reduce(initial.state.positions)
    compilation = phx.optim.compile_structured_state_design(
        problem.as_state_design_problem(),
        reduced,
        jnp.asarray(1.0),
        exact_hessian=False,
    )
    program = compilation.optimization.program

    assert program.num_constraints == plan.problem.structure.free_dof_count + 2
    assert jnp.array_equal(program.constraint_lower[-2:], jnp.asarray((1.0, 1.0)))
    assert jnp.array_equal(program.constraint_upper[-2:], jnp.asarray((2.0, 2.0)))


def test_force_density_design_constraint_preserves_physical_state_callback():
    _, plan, _, decode = _design_setup()
    constraint = fd.ForceDensityDesignConstraint(
        lambda state, design, _: state.positions[1, 1],
        lower=-0.6,
        upper=-0.2,
        constraint_id="center-height",
    )
    problem = fd.ForceDensityDesignProblem(
        plan,
        decode,
        lambda state, design, _: jnp.sum(state.member_lengths),
        constraints=(constraint,),
    )
    inputs = problem.inputs(jnp.asarray(1.0))
    equilibrium = fd.force_density_equilibrium(plan.problem, inputs)
    reduced = plan.problem.structure.reduce(equilibrium.state.positions)
    lowered = problem.as_state_design_problem().constraints[0]
    assert lowered.value(reduced, jnp.asarray(1.0)) == pytest.approx(-0.5)
