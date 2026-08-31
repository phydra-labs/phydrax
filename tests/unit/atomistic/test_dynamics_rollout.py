from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _dynamics():
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        [0, 1], [1, 1], [1.0, 1.0], units, atom_type_ids=[0, 0]
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([1.0], [1.0], 2.5)]
    ).prepare(system)
    return phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()


def test_bounded_rollout_retains_only_planned_samples_and_replays():
    dynamics = _dynamics()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    initial = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(2)
    )
    trajectory_plan = phx.atomistic.AtomisticTrajectoryPlan(
        9, sample_stride=4, include_initial=True
    )
    full = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        trajectory_plan,
        replay=phx.atomistic.AtomisticReplayPolicy("full"),
    ).rollout(initial)
    step = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        trajectory_plan,
        replay=phx.atomistic.AtomisticReplayPolicy("step"),
    ).rollout(initial)
    assert trajectory_plan.capacity == 4
    assert int(full.trajectory.count) == 4
    assert bool(full.successful)
    assert bool(phx.atomistic.atomistic_replay_matches(full.replay, step.replay))
    np.testing.assert_allclose(
        full.final_state.kinematics.positions,
        step.final_state.kinematics.positions,
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_roundtrip_continues_exact_state(tmp_path: Path):
    dynamics = _dynamics()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    initial = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(3)
    )
    advanced = dynamics.step_detailed(initial).accepted_state
    plan = phx.atomistic.AtomisticCheckpointPlan(dynamics)
    path = tmp_path / "atomistic.chk"
    written = phx.atomistic.write_atomistic_checkpoint(path, plan, advanced)
    restored = phx.atomistic.read_atomistic_checkpoint(path, plan, initial)
    assert written.payload_id == restored.payload_id
    observed = dynamics.step_detailed(restored.state).accepted_state
    expected = dynamics.step_detailed(advanced).accepted_state
    leaves_observed = jax.tree.leaves(observed)
    leaves_expected = jax.tree.leaves(expected)
    for left, right in zip(leaves_observed, leaves_expected, strict=True):
        np.testing.assert_array_equal(left, right)
