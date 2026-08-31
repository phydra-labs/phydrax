from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


fd = phx.applications.solid_mechanics


def _cable():
    structure = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        3,
        2,
        fixed_nodes=(0, 2),
    )
    positions = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
    loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
    inputs = fd.ForceDensityInputs(
        jnp.ones((2,)), structure.prescribed_values(positions), loads
    )
    problem = fd.ForceDensityProblem(structure, sign_mode="tension")
    return structure, positions, problem, inputs


def _tetrahedron():
    positions = jnp.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    connectivity = phx.discretization.polygonal_connectivity(
        jnp.asarray(((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)), dtype=jnp.int32),
        None,
        4,
    )
    structure = fd.ForceDensityStructure.from_edges(
        connectivity.edges,
        4,
        3,
        fixed_nodes=(0, 1, 2),
        surface_connectivity=connectivity,
    )
    return structure, positions


def test_nonlinear_plan_refresh_preserves_template_and_updates_setup_numerics():
    structure, positions, _, _ = _cable()
    model = fd.EdgeLineLoadModel(measure="current")
    problem = fd.ForceDensityProblem(structure, load_model=model, sign_mode="tension")
    inputs = fd.ForceDensityInputs(
        jnp.full((2,), 10.0),
        structure.prescribed_values(positions),
        jnp.asarray(((0.0, -0.1), (0.0, -0.1))),
    )
    plan = fd.plan_force_density(problem, inputs, initial_positions=positions)
    prepared = fd.prepare_force_density(plan, inputs, initial_positions=positions)
    changed = fd.ForceDensityInputs(
        jnp.full((2,), 12.0), inputs.prescribed_values, inputs.load_parameters
    )
    refreshed = fd.refresh_force_density(prepared, changed, initial_positions=positions)

    assert plan.nonlinear_template is not None
    assert prepared.nonlinear_solve is not None
    assert refreshed.nonlinear_solve is not None
    assert plan.nonlinear_uses_setup
    assert (
        refreshed.nonlinear_solve.linear_template_id
        == prepared.nonlinear_solve.linear_template_id
    )
    assert int(refreshed.nonlinear_solve.numeric_version) > int(
        prepared.nonlinear_solve.numeric_version
    )
    assert fd.solve_force_density(refreshed).successful


def test_force_density_plan_identity_covers_termination_and_load_tree_contract():
    structure, positions, _, _ = _cable()
    problem = fd.ForceDensityProblem(
        structure,
        load_model=fd.EdgeLineLoadModel(measure="current"),
        sign_mode="tension",
    )
    inputs = fd.ForceDensityInputs(
        jnp.full((2,), 10.0),
        structure.prescribed_values(positions),
        jnp.asarray(((0.0, -0.1), (0.0, -0.1))),
    )
    short = fd.plan_force_density(
        problem,
        inputs,
        initial_positions=positions,
        nonlinear_termination=phx.nonlinear.NonlinearTermination(maximum_steps=20),
    )
    long = fd.plan_force_density(
        problem,
        inputs,
        initial_positions=positions,
        nonlinear_termination=phx.nonlinear.NonlinearTermination(maximum_steps=40),
    )
    assert short.plan_id != long.plan_id
    with pytest.raises(ValueError, match="input PyTree"):
        fd.prepare_force_density(
            short,
            fd.ForceDensityInputs(
                inputs.force_densities,
                inputs.prescribed_values,
                (inputs.load_parameters,),
            ),
            initial_positions=positions,
        )


def test_self_weight_surface_traction_and_component_ledger_conserve_loads():
    structure, positions = _tetrahedron()
    lengths = jnp.sqrt(
        jnp.sum(
            (positions[structure.receivers] - positions[structure.senders]) ** 2,
            axis=-1,
        )
    )
    weight = fd.ReferenceMemberSelfWeightModel(lengths, jnp.asarray((0.0, 0.0, -9.81)))
    traction = fd.SurfaceTractionLoadModel(
        measure="reference", reference_positions=positions
    )
    composite = fd.CompositeForceDensityLoadModel((weight, traction))
    parameters = (
        jnp.ones((structure.member_count,)),
        jnp.zeros((structure.surface_connectivity.cell_count, 3)).at[:, 2].set(2.0),
    )
    state = fd.evaluate_force_density_load(composite, structure, positions, parameters)
    expected_weight = -9.81 * jnp.sum(lengths)
    assert len(state.components) == 2
    assert len(state.component_ids) == 2
    assert state.valid
    assert jnp.sum(state.components[0][:, 2]) == pytest.approx(expected_weight)
    assert jnp.allclose(state.total, state.components[0] + state.components[1])


def test_pneumatic_pressure_uses_closed_volume_law_and_rejects_wrong_orientation():
    structure, positions = _tetrahedron()
    volume = fd.enclosed_surface_volume(structure, positions)
    assert volume == pytest.approx(1.0 / 6.0)
    model = fd.PneumaticPressureLoadModel(
        "ideal-gas", reference_volume=1.0 / 6.0, exponent=1.4
    )
    pressure = model.pressure(structure, positions, jnp.asarray(2.0))
    loads = model.nodal_loads(structure, positions, jnp.asarray(2.0))
    assert pressure == pytest.approx(2.0)
    assert bool(model.valid(structure, positions, jnp.asarray(2.0)))
    assert jnp.all(jnp.isfinite(loads))
    reflected = positions.at[:, 2].multiply(-1.0)
    assert not bool(model.valid(structure, reflected, jnp.asarray(2.0)))


def test_surface_pressure_rejects_folded_q4_and_observables_detect_warp():
    connectivity = phx.discretization.polygonal_connectivity(
        None, jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32), 4
    )
    structure = fd.ForceDensityStructure.from_edges(
        connectivity.edges,
        4,
        3,
        fixed_nodes=(0, 1, 2, 3),
        surface_connectivity=connectivity,
    )
    planar = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    warped = planar.at[2, 2].set(0.2)
    folded = jnp.asarray(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0))
    )
    model = fd.SurfacePressureLoadModel()
    assert bool(model.valid(structure, planar, jnp.asarray((1.0,))))
    assert not bool(model.valid(structure, folded, jnp.asarray((1.0,))))
    assert fd.surface_planarity_residual(structure, planar, 1.0) == pytest.approx(0.0)
    assert jnp.abs(fd.surface_planarity_residual(structure, warped, 1.0)[0]) > 0.0
    assert jnp.allclose(fd.surface_rectangularity_residual(structure, planar, 1.0), 0.0)


def test_pressure_loaded_tetrahedron_solves_and_has_implicit_derivative():
    structure, positions = _tetrahedron()
    problem = fd.ForceDensityProblem(
        structure,
        load_model=fd.SurfacePressureLoadModel(),
        sign_mode="tension",
    )
    sample = fd.ForceDensityInputs(
        jnp.full((structure.member_count,), 20.0),
        structure.prescribed_values(positions),
        jnp.full((structure.surface_connectivity.cell_count,), 0.01),
    )
    plan = fd.plan_force_density(problem, sample, initial_positions=positions)

    def top_height(pressure):
        inputs = fd.ForceDensityInputs(
            sample.force_densities,
            sample.prescribed_values,
            jnp.full((structure.surface_connectivity.cell_count,), pressure),
        )
        solved = fd.solve_force_density(
            fd.prepare_force_density(plan, inputs, initial_positions=positions)
        )
        return solved.state.positions[3, 2]

    result = fd.solve_force_density(
        fd.prepare_force_density(plan, sample, initial_positions=positions)
    )
    derivative = jax.grad(top_height)(jnp.asarray(0.01))
    assert result.successful
    assert result.nonlinear_result is not None
    assert jnp.isfinite(derivative)


def test_batch_affine_reciprocal_and_per_graph_evidence():
    structure, positions, problem, inputs = _cable()
    plan = fd.plan_force_density(problem, inputs)
    batched = fd.solve_force_density_batch(
        plan,
        jnp.asarray(((1.0, 1.0), (2.0, 2.0))),
        jnp.stack((inputs.prescribed_values, inputs.prescribed_values)),
        jnp.stack((inputs.load_parameters, inputs.load_parameters)),
    )
    assert jnp.all(batched.successful)
    assert batched.results.state.positions[:, 1, 1] == pytest.approx(
        jnp.asarray((-0.5, -0.25))
    )
    assert batched.results.diagnostics.graph_free_residual_norms.shape == (2, 1)

    graph = phx.graph.GraphIR(
        senders=jnp.asarray((0, 1, 1, 2)),
        receivers=jnp.asarray((1, 0, 2, 1)),
        n_node=jnp.asarray((3,), dtype=jnp.int32),
        n_edge=jnp.asarray((4,), dtype=jnp.int32),
    )
    reciprocal = fd.ForceDensityStructure.from_graph(
        graph,
        2,
        fixed_nodes=(0, 2),
        edge_semantics="reciprocal-pairs",
        node_ids=("left", "middle", "right"),
    )
    assert reciprocal.member_count == 2
    assert reciprocal.node_ids[1] == "middle"

    prolongation = jnp.asarray(((0.0,), (0.0,), (1.0,), (0.0,)))
    prescribed_map = jnp.asarray(
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    affine = fd.ForceDensityStructure.from_affine_constraints(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        prolongation,
        prescribed_map,
    )
    reference = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    affine_problem = fd.ForceDensityProblem(affine, sign_mode="tension")
    affine_inputs = fd.ForceDensityInputs(
        jnp.asarray((1.0,)),
        affine.prescribed_values(reference),
        jnp.asarray(((0.0, 0.0), (1.0, 0.0))),
    )
    affine_result = fd.force_density_equilibrium(affine_problem, affine_inputs)
    assert affine_result.successful
    assert affine_result.state.positions[1, 0] == pytest.approx(1.0)


def test_mechanism_self_stress_and_constitutive_stability_are_distinct():
    structure, straight, problem, inputs = _cable()
    mechanism = fd.analyze_force_density_mechanisms(structure, straight)
    assert mechanism.successful
    assert int(mechanism.mechanism_count) == 1
    assert int(mechanism.self_stress_count) == 1

    solved = fd.force_density_equilibrium(problem, inputs)
    stability = fd.analyze_force_density_tangent_stability(
        structure, solved.state, jnp.full((2,), 100.0)
    )
    assert stability.successful
    assert stability.stable
    assert stability.minimum_eigenvalue > 0.0


def test_linear_solution_taylor_remainder_is_second_order():
    _, _, problem, sample = _cable()
    plan = fd.plan_force_density(problem, sample)

    def objective(q):
        inputs = fd.ForceDensityInputs(
            q, sample.prescribed_values, sample.load_parameters
        )
        return jnp.sum(
            fd.solve_force_density(fd.prepare_force_density(plan, inputs)).state.positions
            ** 2
        )

    direction = jnp.asarray((0.2, -0.1))
    value, derivative = jax.jvp(objective, (sample.force_densities,), (direction,))
    first = jnp.abs(
        objective(sample.force_densities + 1.0e-2 * direction)
        - value
        - 1.0e-2 * derivative
    )

    second = jnp.abs(
        objective(sample.force_densities + 5.0e-3 * direction)
        - value
        - 5.0e-3 * derivative
    )
    assert first / second == pytest.approx(4.0, rel=0.12)


def test_discrete_arch_converges_to_analytical_load_path_and_rise():
    span = 10.0
    density = 1.0
    node_count = 20
    coordinates = jnp.stack(
        (
            jnp.linspace(-span / 2.0, span / 2.0, node_count),
            jnp.zeros((node_count,)),
        ),
        axis=-1,
    )
    edges = jnp.stack(
        (jnp.arange(node_count - 1), jnp.arange(1, node_count)), axis=-1
    ).astype(jnp.int32)
    structure = fd.ForceDensityStructure.from_edges(
        edges, node_count, 2, fixed_nodes=(0, node_count - 1)
    )
    loads = jnp.zeros((node_count, 2)).at[:, 1].set(-(density * span) / node_count)
    prescribed = structure.prescribed_values(coordinates)
    sample = fd.ForceDensityInputs(jnp.full((node_count - 1,), -10.0), prescribed, loads)
    equilibrium = fd.ForceDensityProblem(structure, sign_mode="compression")
    plan = fd.plan_force_density(equilibrium, sample)

    def decode(magnitude, _):
        return fd.ForceDensityInputs(
            jnp.full((node_count - 1,), -magnitude.reshape(())),
            prescribed,
            loads,
        )

    design = fd.ForceDensityDesignProblem(
        plan,
        decode,
        lambda state, magnitude, _: fd.force_density_load_path(state),
        design_bounds=phx.optim.Bounds(1.0e-3, 1.0e3),
    )
    result = fd.solve_force_density_design(
        design,
        jnp.asarray(10.0),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1.0e-6,
            relative_optimality=0.0,
            maximum_steps=500,
        ),
    )
    rise = jnp.max(jnp.abs(result.equilibrium.state.positions[:, 1]))
    load_path = fd.force_density_load_path(result.equilibrium.state)
    target_rise = jnp.sqrt(3.0) * span / 4.0
    target_load_path = density * span**2 / jnp.sqrt(3.0)
    assert result.successful
    assert rise == pytest.approx(target_rise, rel=0.02)
    assert load_path == pytest.approx(target_load_path, rel=0.08)


def test_structured_force_density_wrapper_compiles_physical_constraints():
    structure, _, equilibrium, sample = _cable()
    plan = fd.plan_force_density(equilibrium, sample)

    def decode(magnitude, _):
        return fd.ForceDensityInputs(
            jnp.repeat(magnitude.reshape(()), 2),
            sample.prescribed_values,
            sample.load_parameters,
        )

    constraint = fd.ForceDensityDesignConstraint(
        lambda state, magnitude, _: state.positions[1, 1],
        lower=-0.6,
        upper=-0.2,
        constraint_id="center-height",
    )
    design = fd.ForceDensityDesignProblem(
        plan,
        decode,
        lambda state, magnitude, _: fd.force_density_load_path(state),
        design_bounds=phx.optim.Bounds(0.2, 8.0),
        constraints=(constraint,),
    )
    with pytest.raises(ValueError, match="require an explicit"):
        fd.solve_force_density_design(design, jnp.asarray(1.0))
    compiled = fd.compile_structured_force_density_design(
        design,
        jnp.asarray(1.0),
        exact_hessian=False,
    )
    assert isinstance(compiled, fd.StructuredForceDensityDesignCompilation)
    assert (
        compiled.state_design.optimization.program.num_constraints
        == structure.free_dof_count + 1
    )
