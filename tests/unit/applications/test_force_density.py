from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


fd = phx.applications.solid_mechanics


def _three_node_problem(*, compression: bool = False):
    structure = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (1, 2)), dtype=jnp.int32),
        3,
        2,
        fixed_nodes=(0, 2),
    )
    reference = jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
    loads = jnp.asarray(((0.0, 0.0), (0.0, -1.0), (0.0, 0.0)))
    q = jnp.full((2,), -1.0 if compression else 1.0)
    inputs = fd.ForceDensityInputs(
        q,
        structure.prescribed_values(reference),
        loads,
    )
    problem = fd.ForceDensityProblem(
        structure,
        sign_mode="compression" if compression else "tension",
        problem_id="three-node-arch" if compression else "three-node-cable",
    )
    return problem, inputs


def test_force_density_structure_rejects_unanchored_coordinate_and_self_loop():
    with pytest.raises(ValueError, match="constrain every translation"):
        fd.ForceDensityStructure.from_edges(
            jnp.asarray(((0, 1),), dtype=jnp.int32),
            2,
            2,
            constrained_dofs=jnp.asarray(((True, False), (False, False))),
        )
    with pytest.raises(ValueError, match="self-loops"):
        fd.ForceDensityStructure.from_edges(
            jnp.asarray(((0, 0),), dtype=jnp.int32),
            1,
            2,
            fixed_nodes=(0,),
        )


def test_force_density_structure_preserves_parallel_members_and_partial_restraints():
    structure = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1), (0, 1)), dtype=jnp.int32),
        2,
        2,
        constrained_dofs=jnp.asarray(((True, True), (False, True))),
    )
    assert structure.member_count == 2
    assert structure.free_dof_count == 1
    assert structure.equilibrium_relation.source_size == 1
    reconstructed = fd.ForceDensityStructure.from_graph(
        structure.graph,
        2,
        constrained_dofs=structure.constrained_dofs,
    )
    assert reconstructed.structure_id == structure.structure_id


def test_three_node_tension_and_compression_are_mirrors_with_balanced_reactions():
    tension_problem, tension_inputs = _three_node_problem()
    compression_problem, compression_inputs = _three_node_problem(compression=True)
    tension = fd.force_density_equilibrium(tension_problem, tension_inputs)
    compression = fd.force_density_equilibrium(compression_problem, compression_inputs)

    assert tension.successful
    assert compression.successful
    assert tension.state.positions[1] == pytest.approx(jnp.asarray((0.0, -0.5)))
    assert compression.state.positions[1] == pytest.approx(jnp.asarray((0.0, 0.5)))
    assert tension.diagnostics.free_residual_norm <= 1.0e-10
    assert compression.diagnostics.free_residual_norm <= 1.0e-10
    assert jnp.sum(tension.state.support_reactions[:, 1]) == pytest.approx(1.0)
    assert tension.diagnostics.global_balance_norm <= 1.0e-10
    assert jnp.all(tension.state.axial_forces > 0.0)
    assert jnp.all(compression.state.axial_forces < 0.0)


def test_member_orientation_does_not_change_equilibrium():
    problem, inputs = _three_node_problem()
    expected = fd.force_density_equilibrium(problem, inputs)
    reversed_structure = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((1, 0), (2, 1)), dtype=jnp.int32),
        3,
        2,
        fixed_nodes=(0, 2),
    )
    reversed_problem = fd.ForceDensityProblem(
        reversed_structure,
        sign_mode="tension",
    )
    reversed_inputs = fd.ForceDensityInputs(
        inputs.force_densities,
        reversed_structure.prescribed_values(
            jnp.asarray(((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0)))
        ),
        inputs.load_parameters,
    )
    actual = fd.force_density_equilibrium(reversed_problem, reversed_inputs)
    assert jnp.allclose(actual.state.positions, expected.state.positions)
    assert jnp.allclose(actual.state.member_lengths, expected.state.member_lengths)
    assert jnp.allclose(actual.state.axial_forces, expected.state.axial_forces)


def test_sparse_force_density_solve_matches_direct_dense_equations():
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (0, 2)), dtype=np.int32)
    structure = fd.ForceDensityStructure.from_edges(edges, 4, 2, fixed_nodes=(0, 3))
    positions = jnp.asarray(((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (3.0, 0.0)))
    q = jnp.asarray((1.5, 2.0, 0.7, 1.1))
    loads = jnp.asarray(((0.0, 0.0), (0.0, -0.3), (0.2, -0.5), (0.0, 0.0)))
    inputs = fd.ForceDensityInputs(q, structure.prescribed_values(positions), loads)
    problem = fd.ForceDensityProblem(structure, sign_mode="tension")
    result = fd.force_density_equilibrium(problem, inputs)

    incidence = np.zeros((edges.shape[0], 4))
    incidence[np.arange(edges.shape[0]), edges[:, 0]] = -1.0
    incidence[np.arange(edges.shape[0]), edges[:, 1]] = 1.0
    matrix = incidence.T @ np.diag(np.asarray(q)) @ incidence
    free = np.asarray((1, 2))
    fixed = np.asarray((0, 3))
    expected = np.linalg.solve(
        matrix[np.ix_(free, free)],
        np.asarray(loads)[free]
        - matrix[np.ix_(free, fixed)] @ np.asarray(positions)[fixed],
    )
    assert result.successful
    assert jnp.allclose(result.state.positions[free], expected, atol=1.0e-10)


def test_all_constrained_structure_returns_reactions_without_linear_solve():
    structure = fd.ForceDensityStructure.from_edges(
        jnp.asarray(((0, 1),), dtype=jnp.int32),
        2,
        2,
        fixed_nodes=(0, 1),
    )
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0)))
    loads = jnp.asarray(((0.0, -1.0), (0.0, 0.0)))
    problem = fd.ForceDensityProblem(structure, sign_mode="tension")
    inputs = fd.ForceDensityInputs(
        jnp.asarray((2.0,)), structure.prescribed_values(positions), loads
    )
    result = fd.force_density_equilibrium(problem, inputs)
    assert result.successful
    assert result.linear_result is None
    assert result.state.positions == pytest.approx(positions)
    assert jnp.allclose(
        jnp.sum(
            result.state.applied_nodal_loads + result.state.support_reactions, axis=0
        ),
        0.0,
    )


def test_prepared_refresh_reuses_symbolic_plan_and_changes_numeric_solution():
    problem, inputs = _three_node_problem()
    plan = fd.plan_force_density(problem, inputs)
    prepared = fd.prepare_force_density(plan, inputs)
    first = fd.solve_force_density(prepared)
    changed = fd.ForceDensityInputs(
        jnp.asarray((2.0, 2.0)),
        inputs.prescribed_values,
        inputs.load_parameters,
    )
    refreshed = fd.refresh_force_density(prepared, changed)
    second = fd.solve_force_density(refreshed)
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert int(refreshed.numeric_version) == 1
    assert first.state.positions[1, 1] == pytest.approx(-0.5)
    assert second.state.positions[1, 1] == pytest.approx(-0.25)


def test_force_density_solution_map_has_finite_q_load_and_support_derivatives():
    problem, sample = _three_node_problem()
    plan = fd.plan_force_density(problem, sample)

    def objective(q, load_y, support_y):
        prescribed = sample.prescribed_values.at[1].set(support_y)
        loads = sample.load_parameters.at[1, 1].set(load_y)
        inputs = fd.ForceDensityInputs(q, prescribed, loads)
        result = fd.solve_force_density(fd.prepare_force_density(plan, inputs))
        return jnp.sum(result.state.positions**2) + fd.force_density_load_path(
            result.state
        )

    q = sample.force_densities
    value, pullback = jax.vjp(objective, q, -1.0, 0.0)
    gradients = pullback(jnp.ones_like(value))
    tangent = jax.jvp(
        objective,
        (q, -1.0, 0.0),
        (jnp.asarray((0.2, -0.1)), 0.1, 0.05),
    )[1]
    epsilon = 1.0e-5
    finite_difference = (
        objective(
            q + epsilon * jnp.asarray((0.2, -0.1)), -1.0 + epsilon * 0.1, epsilon * 0.05
        )
        - objective(
            q - epsilon * jnp.asarray((0.2, -0.1)), -1.0 - epsilon * 0.1, -epsilon * 0.05
        )
    ) / (2.0 * epsilon)
    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in gradients)
    assert tangent == pytest.approx(finite_difference, rel=2.0e-5, abs=2.0e-6)


def test_force_density_sign_and_magnitude_contracts_fail_closed():
    problem, inputs = _three_node_problem()
    with pytest.raises(Exception, match="magnitude or sign"):
        fd.force_density_equilibrium(
            problem,
            fd.ForceDensityInputs(
                jnp.asarray((1.0, -1.0)),
                inputs.prescribed_values,
                inputs.load_parameters,
            ),
        )
    with pytest.raises(Exception, match="magnitude or sign"):
        fd.force_density_equilibrium(
            problem,
            fd.ForceDensityInputs(
                jnp.asarray((1.0, 0.0)),
                inputs.prescribed_values,
                inputs.load_parameters,
            ),
        )


def test_force_density_load_output_must_share_coordinate_dtype():
    problem, inputs = _three_node_problem()
    mismatched = fd.ForceDensityInputs(
        inputs.force_densities,
        inputs.prescribed_values,
        inputs.load_parameters.astype(jnp.float32),
    )
    with pytest.raises(TypeError, match="share the force-density coordinate dtype"):
        fd.force_density_equilibrium(problem, mismatched)
