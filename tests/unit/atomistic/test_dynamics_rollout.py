from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tests._support.atomistic_bundles import (
    prepare_program,
    role_capacity,
    search_capacity,
)


def _dynamics() -> Any:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    system = phx.atomistic.AtomisticSystemPlan(
        # ty: ignore[invalid-argument-type]
        [0, 1],
        # ty: ignore[invalid-argument-type]
        [1, 1],
        # ty: ignore[invalid-argument-type]
        [1.0, 1.0],
        units,
        # ty: ignore[invalid-argument-type]
        atom_type_ids=[0, 0],
    ).prepare()
    # Finite pair: explicit nonperiodic search frame and the one-pair storage of
    # the former dense neighborhood.
    potential = prepare_program(
        phx.atomistic.AtomisticPotentialProgram(
            # ty: ignore[invalid-argument-type]
            [phx.atomistic.LennardJonesPotential([1.0], [1.0], 2.5)]
        ),
        system,
        search=phx.atomistic.AtomisticCellListImageSearch(maximum_candidate_slots=4096),
        skin=0.5,
        search_capacities=(search_capacity(8, particles_per_cell=2, images=27),),
        role_capacities=(role_capacity(1, 1),),
        frame=phx.discretization.PeriodicCell(
            20.0 * jnp.eye(3), origin=jnp.full((3,), -10.0), periodic_axes=(False,) * 3
        ),
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    return dynamics, thermodynamic


def test_bounded_rollout_retains_only_planned_samples_and_replays() -> None:
    dynamics, thermodynamic = _dynamics()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    initial = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(2),
    )
    trajectory_plan = phx.atomistic.AtomisticTrajectoryPlan(
        9, sample_stride=4, include_initial=True
    )
    full = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        thermodynamic,
        trajectory_plan,
        replay=phx.atomistic.AtomisticReplayPolicy("full"),
    ).rollout(initial)
    step = phx.atomistic.AtomisticRolloutPlan(
        dynamics,
        thermodynamic,
        trajectory_plan,
        replay=phx.atomistic.AtomisticReplayPolicy("step"),
    ).rollout(initial)
    assert trajectory_plan.capacity == 4
    assert int(full.trajectory.count) == 4
    assert (
        full.trajectory.units.unit_system_id == dynamics.system.plan.units.unit_system_id
    )
    assert bool(full.successful)
    assert bool(phx.atomistic.atomistic_replay_matches(full.replay, step.replay))
    np.testing.assert_allclose(
        full.final_state.kinematics.positions,
        step.final_state.kinematics.positions,
        rtol=0.0,
        atol=0.0,
    )


def test_checkpoint_roundtrip_continues_exact_state(tmp_path: Path) -> None:
    dynamics, thermodynamic = _dynamics()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    initial = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(3),
    )
    advanced = dynamics.step_detailed(initial, thermodynamic).accepted_state
    plan = phx.atomistic.AtomisticCheckpointPlan(dynamics, thermodynamic)
    path = tmp_path / "atomistic.chk"
    written = phx.atomistic.write_atomistic_checkpoint(path, plan, advanced)
    restored = phx.atomistic.read_atomistic_checkpoint(path, plan, initial)
    assert written.payload_id == restored.payload_id
    assert written.units.unit_system_id == dynamics.system.plan.units.unit_system_id
    assert restored.units.unit_system_id == dynamics.system.plan.units.unit_system_id
    observed = dynamics.step_detailed(restored.state, thermodynamic).accepted_state
    expected = dynamics.step_detailed(advanced, thermodynamic).accepted_state
    leaves_observed = jax.tree.leaves(observed)
    leaves_expected = jax.tree.leaves(expected)
    for left, right in zip(leaves_observed, leaves_expected, strict=True):
        np.testing.assert_array_equal(left, right)
