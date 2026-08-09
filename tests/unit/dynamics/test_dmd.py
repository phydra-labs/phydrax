#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _linear_trajectory(*, controlled: bool):
    generator = np.random.default_rng(2026)
    cases = 12
    capacity = 7
    state_matrix = np.asarray([[0.8, 0.25], [-0.15, 0.9]])
    input_matrix = np.asarray([[0.5], [-0.2]])
    states = np.empty((cases, capacity, 2))
    controls = generator.normal(size=(cases, capacity - 1, 1))
    states[:, 0] = generator.normal(size=(cases, 2))
    for step in range(capacity - 1):
        states[:, step + 1] = states[:, step] @ state_matrix.T
        if controlled:
            states[:, step + 1] += controls[:, step] @ input_matrix.T
    coordinates = np.broadcast_to(np.arange(capacity), (cases, capacity))
    data = phx.dynamics.TrajectoryData(
        coordinates,
        states,
        state_layout=phx.dynamics.StateLayout((2,), component_names=("x", "y")),
        inputs=controls if controlled else None,
        input_layout=(
            phx.dynamics.InputLayout((1,), component_names=("u",), roles="control")
            if controlled
            else None
        ),
        case_axes=("trajectory",),
        case_axis_roles=("case",),
        coordinate_id="step",
        source_id="linear-benchmark",
    )
    return data, state_matrix, input_matrix


def test_exact_dmd_recovers_linear_map_and_executes_as_system():
    data, expected, _ = _linear_trajectory(controlled=False)

    result = phx.dynamics.identification.fit_dmd(data, mode="exact")

    assert bool(result.valid)
    assert int(result.diagnostics.retained_rank) == 2
    np.testing.assert_allclose(
        np.asarray(result.state_matrix), expected, rtol=1e-11, atol=1e-11
    )
    system = result.to_system()
    prediction = system.evaluate(jnp.asarray(0), jnp.asarray([0.4, -0.3]), None)
    np.testing.assert_allclose(
        np.asarray(prediction), expected @ np.asarray([0.4, -0.3]), atol=1e-11
    )


def test_controlled_dmd_recovers_state_and_input_matrices():
    data, expected_state, expected_input = _linear_trajectory(controlled=True)

    result = phx.dynamics.identification.fit_dmd(data, rank=3)

    assert bool(result.valid)
    np.testing.assert_allclose(
        np.asarray(result.state_matrix), expected_state, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(
        np.asarray(result.input_matrix), expected_input, rtol=1e-11, atol=1e-11
    )
    predicted = result.to_system().evaluate(
        jnp.asarray(0),
        jnp.asarray([0.2, 0.7]),
        None,
        inputs=jnp.asarray([-0.4]),
    )
    expected = expected_state @ np.asarray([0.2, 0.7]) + expected_input[:, 0] * -0.4
    np.testing.assert_allclose(np.asarray(predicted), expected, atol=1e-11)


def test_continuous_dmd_eigenvalue_conversion_rejects_irregular_spacing():
    states = jnp.asarray([[1.0], [0.8], [0.6], [0.4]])
    data = phx.dynamics.TrajectoryData(
        jnp.asarray([0.0, 0.2, 0.7, 1.0]),
        states,
        state_layout=phx.dynamics.StateLayout((1,)),
        source_id="irregular",
    )

    with pytest.raises(ValueError, match="uniform"):
        phx.dynamics.identification.fit_dmd(data, continuous_eigenvalues=True)


def test_edmd_recovers_quadratic_map_and_decoder():
    generator = np.random.default_rng(18)
    cases = 20
    capacity = 5
    states = np.empty((cases, capacity, 1))
    states[:, 0, 0] = generator.uniform(-0.8, 0.8, size=cases)
    for step in range(capacity - 1):
        value = states[:, step, 0]
        states[:, step + 1, 0] = 0.2 + 0.8 * value - 0.3 * value**2
    coordinates = np.broadcast_to(np.arange(capacity), (cases, capacity))
    layout = phx.dynamics.StateLayout((1,), component_names=("x",))
    data = phx.dynamics.TrajectoryData(
        coordinates,
        states,
        state_layout=layout,
        case_axes=("initial_condition",),
        case_axis_roles=("case",),
        source_id="quadratic-map",
    )
    library = phx.dynamics.identification.PolynomialFeatureLibrary(layout, degree=2)

    result = phx.dynamics.identification.fit_edmd(data, library)

    assert bool(result.valid)
    query = jnp.asarray([[-0.5], [0.1], [0.7]])
    expected = 0.2 + 0.8 * query - 0.3 * query**2
    np.testing.assert_allclose(
        np.asarray(result.predict(query)), np.asarray(expected), atol=2e-11
    )
    evolution = phx.dynamics.DiscreteEvolution(result.to_system())
    rollout = phx.dynamics.evolve(
        evolution,
        jnp.asarray([0.25]),
        phx.dynamics.IterationGrid.from_steps(4, iteration_id="edmd-rollout"),
    )
    assert bool(rollout.successful)
