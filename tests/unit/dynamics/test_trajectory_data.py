#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _ragged_data():
    coordinates = jnp.asarray([[0.0, 0.2, 0.7, 1.1, 1.6], [0.0, 0.4, 1.0, 1.0, 1.0]])
    states = coordinates[..., None] ** 2
    sample_valid = jnp.asarray(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    states = jnp.where(sample_valid[..., None], states, jnp.nan)
    return phx.dynamics.TrajectoryData(
        coordinates,
        states,
        state_layout=phx.dynamics.StateLayout(
            (1,), axes=("state",), component_names=("x",)
        ),
        sample_valid=sample_valid,
        reset_mask=jnp.asarray(
            [[False, False, True, False], [False, False, False, False]]
        ),
        case_axes=("case",),
        case_axis_roles=("case",),
        coordinate_id="time",
        source_id="ragged",
    )


def test_trajectory_data_preserves_padding_resets_and_case_axes():
    data = _ragged_data()

    assert data.case_shape == (2,)
    assert data.state_layout.component_names == ("x",)
    assert not bool(data.transition_valid[0, 2])
    assert not bool(data.transition_valid[1, 2])
    pairs = data.transitions()
    assert pairs.source_states.shape == (2, 4, 1)
    np.testing.assert_array_equal(
        np.asarray(pairs.valid),
        np.asarray([[True, True, False, True], [True, True, False, False]]),
    )


def test_trajectory_data_rejects_cross_reset_transition_marked_valid():
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="reset"):
        phx.dynamics.TrajectoryData(
            jnp.arange(4.0),
            jnp.arange(4.0)[:, None],
            state_layout=phx.dynamics.StateLayout((1,)),
            reset_mask=jnp.asarray([False, True, False]),
            transition_valid=jnp.ones((3,), dtype=bool),
            source_id="invalid-reset",
        )


def test_delay_embedding_never_crosses_a_reset():
    coordinates = jnp.arange(7.0)
    data = phx.dynamics.TrajectoryData(
        coordinates,
        jnp.stack((coordinates, coordinates**2), axis=-1),
        state_layout=phx.dynamics.StateLayout((2,), component_names=("x", "y")),
        reset_mask=jnp.asarray([False, False, False, True, False, False]),
        source_id="delay-source",
    )

    embedded = phx.dynamics.identification.delay_embed(data, (0, 2))

    assert embedded.state_layout.component_names == (
        "x[t-0]",
        "y[t-0]",
        "x[t-2]",
        "y[t-2]",
    )
    np.testing.assert_array_equal(
        np.asarray(embedded.sample_valid),
        np.asarray([False, False, True, True, False, False, True]),
    )
    np.testing.assert_allclose(
        np.asarray(embedded.states[2]), np.asarray([2.0, 4.0, 0.0, 0.0])
    )


def test_evolution_adapter_retains_system_layout_and_provenance():
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: state + args,
        state_layout=phx.dynamics.StateLayout((1,), component_names=("population",)),
        system_id="growth-map",
    )
    evolution = phx.dynamics.DiscreteEvolution(system)
    trajectory = phx.dynamics.evolve(
        evolution,
        jnp.asarray([1.0]),
        phx.dynamics.IterationGrid.from_steps(3, iteration_id="observations"),
        args=jnp.asarray([0.5]),
    )

    data = phx.dynamics.identification.trajectory_data_from_evolution(trajectory)

    np.testing.assert_allclose(
        np.asarray(data.states[:, 0]), np.asarray([1.0, 1.5, 2.0, 2.5])
    )
    assert data.state_layout.component_names == ("population",)
    assert data.coordinate_id == "observations"
    assert data.source_id == f"evolution:{trajectory.evolution_id}"


def test_memory_solution_adapter_preserves_delay_masks_and_solver_identity():
    solution = phx.solver.MemoryEquationSolution(
        times=jnp.asarray([0.0, 0.25, 0.5, 0.75]),
        states=jnp.asarray([[1.0], [0.9], [0.82], [0.75]]),
        valid=jnp.asarray([True, True, True, False]),
        realization=None,
        state_shape=(1,),
        solver_name="method-of-steps",
        solver_id="delay:retarded",
        resolved_method="diffrax-method-of-steps",
    )

    data = phx.dynamics.identification.trajectory_data_from_differential_solution(
        solution,
        state_layout=phx.dynamics.StateLayout((1,), component_names=("population",)),
    )

    assert data.source_id == ("memory:delay:retarded:diffrax-method-of-steps")
    np.testing.assert_array_equal(
        np.asarray(data.sample_valid), [True, True, True, False]
    )
    np.testing.assert_array_equal(np.asarray(data.transition_valid), [True, True, False])
