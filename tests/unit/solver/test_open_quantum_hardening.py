#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_shared_scalar_root_event_and_semantic_replay():
    problem = phx.solver.amplitude_damping_trajectory_problem(
        2.0, jnp.asarray([0.0j, 1.0 + 0.0j])
    )
    plan = phx.solver.QuantumTrajectoryPlan(maximum_events=8, root_method="toms748")
    first = phx.solver.solve_event_driven_quantum_jump(
        problem,
        jax.random.PRNGKey(1),
        step_size=0.1,
        steps=10,
        trajectory_plan=plan,
    )
    second = phx.solver.solve_event_driven_quantum_jump(
        problem,
        jax.random.PRNGKey(1),
        step_size=0.1,
        steps=10,
        trajectory_plan=plan,
    )
    assert bool(first.valid)
    assert jnp.array_equal(first.events.times, second.events.times)


def test_environment_mps_and_nonnormalizing_tebd():
    state = phx.tensor_network.product_mps(
        jnp.asarray([[2.0, 0.0], [0.0, 1.0]], dtype=complex)
    )
    assert jnp.allclose(state.norm(), jnp.linalg.norm(state.to_dense()))
    hamiltonian = phx.tensor_network.NearestNeighborHamiltonian(
        (jnp.zeros((4, 4), dtype=complex),),
        (2, 2),
        hamiltonian_id="zero",
    )
    evolved, _ = phx.tensor_network.tebd_step(
        state,
        hamiltonian,
        0.1,
        maximum_bond_dimension=2,
        normalize=False,
    )
    assert jnp.allclose(evolved.norm(), state.norm())


def test_lpdo_raw_trace_canonicalization_and_xxz_strang():
    problem = phx.solver.boundary_driven_xxz_problem(
        2, half_step=0.005, boundary_rate=0.2
    )
    initial_trace = problem.initial_state.raw_trace()
    assert jnp.allclose(initial_trace, 1.0)
    canonical, evidence = phx.tensor_network.canonicalize_lpdo(
        problem.initial_state, center=1
    )
    assert bool(evidence.valid)
    result = phx.solver.solve_purified_strang(
        phx.solver.PurifiedStrangProblem(
            canonical,
            problem.hamiltonian,
            problem.half_step_channels,
        ),
        step_size=0.01,
        steps=1,
        maximum_bond_dimension=4,
        maximum_purification_dimension=8,
    )
    assert bool(result.valid)
    assert jnp.all(jnp.isfinite(result.raw_trace_history))


def test_bath_decomposition_scaled_and_implicit_heom():
    with pytest.raises(ValueError):
        phx.operators.quantum.drude_lorentz_matsubara(0.1, 2.0 * jnp.pi, 1.0, 2)
    expansion = phx.operators.quantum.underdamped_brownian_two_pole(1.0, 0.2, 0.1)
    hierarchy = phx.solver.HEOMHierarchy(expansion.rank, 1)
    scaled = phx.solver.ScaledHEOMTopology(hierarchy, expansion)
    assert bool(scaled.valid)
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    problem = phx.solver.HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.asarray([[1, 0], [0, -1]], dtype=complex),
        expansion,
        hierarchy,
        density,
    )
    result = phx.solver.solve_heom_backward_euler(problem, step_size=0.001, steps=1)
    assert bool(result.valid)


def test_heom_continuation_uses_common_initial_state():
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    expansion = phx.operators.quantum.drude_lorentz_matsubara(0.02, 1.0, 2.0, 1)
    problem = phx.solver.HEOMProblem(
        jnp.zeros((2, 2), dtype=complex),
        jnp.asarray([[1, 0], [0, -1]], dtype=complex),
        expansion,
        phx.solver.HEOMHierarchy(1, 1),
        density,
    )
    continuation = phx.solver.solve_heom_continuation(
        problem, (1, 2), step_size=0.001, steps=1, tolerance=1.0
    )
    assert jnp.allclose(
        continuation.solutions[0].root_states[0],
        continuation.solutions[1].root_states[0],
    )


def test_conditioned_map_and_matched_spin_boson():
    identity = jnp.eye(4, dtype=complex)
    report = phx.operators.quantum.analyze_dynamical_map_series(
        jnp.stack((identity, identity)), 2
    )
    assert bool(report.valid)
    assert jnp.allclose(report.intermediate_solve_residuals, 0.0)
    density = jnp.asarray([[0.6 + 0j, 0j], [0j, 0.4 + 0j]])
    comparison = phx.solver.spin_boson_dephasing_comparison(
        density, heom_depth=1, step_size=0.001, steps=1
    )
    assert bool(comparison.valid)


def test_causal_process_tomography_and_compression_status():
    spec = phx.tensor_network.CombLegSpec(2, 1, 1)
    density = jnp.asarray([[0.7 + 0j, 0j], [0j, 0.3 + 0j]])
    process = phx.tensor_network.CausalProcessTensor(
        spec,
        density,
        (jnp.eye(2, dtype=complex)[None, ...],),
        process_id="identity-process",
    )
    kraus = jnp.eye(2, dtype=complex)[None, None, ...]
    instrument = phx.tensor_network.QuantumInstrument(
        kraus,
        jnp.asarray([True]),
        jnp.asarray([[True]]),
        instrument_id="identity-instrument",
    )
    experiment = phx.solver.ProcessTomographyExperiment(
        (instrument,), (0,), 10.0, experiment_id="identity-count"
    )
    problem = phx.solver.CausalProcessTomographyProblem(process, (experiment,))
    result = phx.solver.fit_causal_process_initial_state(
        problem, jnp.eye(2, dtype=complex), iterations=1
    )
    assert bool(result.underidentified)
    assert not bool(result.valid)
    compressed = phx.tensor_network.project_process_memory_subspace(process, 1)
    assert bool(compressed.valid)


def test_neural_no_jump_tdvp_lifecycle():
    problem = phx.solver.NeuralNoJumpTDVPProblem(
        jnp.asarray([0.2]),
        lambda parameters, vector: vector,
        lambda parameters: -parameters,
        lambda parameters: jnp.asarray([0.0]),
        lambda channel, parameters: (parameters, jnp.asarray(0.0)),
    )
    result = phx.solver.solve_neural_no_jump_tdvp(
        problem,
        jax.random.PRNGKey(0),
        step_size=0.1,
        steps=2,
    )
    assert bool(result.valid)
    assert result.parameter_history.shape == (3, 1)
