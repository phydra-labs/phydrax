import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_stochastic_transition_view_preserves_paths_masks_and_split_provenance():
    times = jnp.asarray([0.0, 0.1, 0.2, 0.4])
    states = jnp.arange(4 * 4 * 3, dtype=float).reshape((4, 4, 3))
    valid = jnp.ones((4, 4), dtype=bool).at[0, 2].set(False)
    trajectory = phx.stochastic.StochasticTrajectory(
        times,
        states,
        valid=valid,
        realization_axes=("path",),
        realization_shape=(4,),
        state_axes=("space",),
        realizations=(None,),
        case_ids=("heat-case",),
        parameter_ids=("diffusivity:0.1",),
        discretization_id="grid:3",
        basis_id="noise:2",
    )
    transitions = trajectory.adjacent_transitions()

    assert transitions.source_states.shape == (4, 3, 3)
    assert transitions.target_states.shape == (4, 3, 3)
    assert jnp.array_equal(
        transitions.valid,
        jnp.asarray(
            [
                [True, False, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        ),
    )
    assert transitions.num_valid == 10
    sampled = transitions.sample_flat_indices(jr.key(0), 64)
    assert jnp.all(transitions.valid.reshape((-1,))[sampled])

    discretized_axis = phx.discretization.UniformAxisSpec(3).materialize(0.0, 1.0)
    axis = phx.nn.operator.OperatorAxis.from_discretization("x", discretized_axis)
    dataset = transitions.operator_dataset(
        source_axes=(axis,),
        source_time_name="source_time",
    )
    split = phx.nn.operator.training.split_operator_dataset(
        dataset,
        train_fraction=0.5,
        validation_fraction=0.25,
        policy=phx.nn.operator.training.OperatorSplitPolicy(
            group_by=("trajectory",), seed=3
        ),
    )
    trajectory_groups = []
    for partition in (split.train, split.validation, split.test):
        assert partition.provenance is not None
        trajectory_groups.append(
            {record.identities["trajectory"] for record in partition.provenance}
        )

    assert dataset.size == transitions.num_valid
    assert len(transitions.driver_segment_references()) == transitions.num_valid
    assert trajectory_groups[0].isdisjoint(trajectory_groups[1])
    assert trajectory_groups[0].isdisjoint(trajectory_groups[2])
    assert trajectory_groups[1].isdisjoint(trajectory_groups[2])
    assert set.union(*trajectory_groups) == set(trajectory.trajectory_ids)
    assert all(
        record.identities["physical_case"] == "heat-case" for record in dataset.provenance
    )
    assert all(
        record.identities["parameters"] == "diffusivity:0.1"
        for record in dataset.provenance
    )
