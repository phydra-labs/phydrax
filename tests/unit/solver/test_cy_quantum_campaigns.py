#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_tetrahedral_qubit_tomography_improves_likelihood():
    true_density = jnp.asarray([[0.7 + 0.0j, 0.1], [0.1, 0.3 + 0.0j]])
    problem = phx.solver.tetrahedral_qubit_tomography(true_density, shots=2000)
    initial = phx.uq.tomography_log_likelihood(
        problem.povm, problem.data, problem.initial_density
    ).log_likelihood
    result = phx.solver.solve_quantum_tomography(
        problem,
        policy=phx.solver.QuantumTomographyPolicy(iterations=20, learning_rate=0.05),
    )
    assert bool(result.valid)
    assert result.log_likelihood_history[-1] >= initial
    assert bool(problem.manifold.contains(result.density))
    artifact = phx.solver.freeze_quantum_tomography(result, problem)
    assert artifact.povm_id == problem.povm.povm_id


def test_lindblad_amplitude_damping_preserves_density_invariants():
    initial = jnp.asarray([[0.01 + 0.0j, 0.0j], [0.0j, 0.99 + 0.0j]])
    problem = phx.solver.amplitude_damping_problem(0.5, initial)
    result = phx.solver.solve_lindblad(problem, step_size=0.05, steps=10)
    assert bool(result.valid)
    assert result.states[-1, 1, 1] < initial[1, 1]
    assert jnp.max(result.trace_residuals) < 1e-8


def test_calabi_yau_campaign_problem_artifact_and_point_inference():
    key = jax.random.PRNGKey(12)
    campaign = phx.solver.prepare_elliptic_curve(key, line_count=2)
    assert bool(jnp.all(campaign.problem.samples.valid))
    result = phx.solver.solve_calabi_yau_metric(
        campaign.problem,
        policy=phx.solver.CalabiYauSolvePolicy(
            iterations=1,
            learning_rate=1e-4,
            maximum_backtracks=1,
        ),
    )
    assert result.objective_history.shape == (1,)
    artifact = phx.solver.freeze_calabi_yau_result(result, campaign.hypersurface)
    point = campaign.problem.samples.homogeneous_points[0]
    evaluation = artifact.evaluate(campaign.hypersurface, point)
    assert jnp.all(jnp.isfinite(evaluation.metric))
    assert evaluation.metric.shape == (2, 2)
