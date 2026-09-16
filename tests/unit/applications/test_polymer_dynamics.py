import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications import cellular_mechanics as cm, polymer_liquids as pl


def _runtime():
    system = phx.atomistic.AtomisticSystemPlan(
        [10, 20, 30, 40],
        [0, 0, 0, 0],
        [1.0, 1.0, 1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0, 0, 0],
        element_mask=[False, False, False, False],
        molecule_ids=[0, 0, 0, 0],
    ).prepare()
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.01], [0.5], 1.0)]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(6).prepare(
        system.particles
    )
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()
    thermodynamic = phx.atomistic.AtomisticThermodynamicStatePlan(
        phx.atomistic.AtomisticPhaseSpaceMeasurePlan(system), ensemble="nve"
    ).prepare(dynamics)
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0], [3.6, 0.0, 0.0]]
    )
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(7),
    )
    return dynamics, thermodynamic, state, positions


def test_chromatin_uses_live_coordinates_and_joint_checkpoint_replays():
    dynamics, thermodynamic, atomistic, positions = _runtime()
    chromatin = cm.ChromatinDynamicsPlan(
        4,
        1,
        ambient_dimension=3,
        capture_distance=10.0,
        spring_stiffness=0.1,
        spring_rest_length=1.0,
    ).prepare()
    chromatin_state = chromatin.initialize(left=[0], right=[3])
    coupling = cm.ChromatinAtomisticCouplingPlan(
        [10, 20, 30, 40], maximum_spring_energy=100.0
    ).prepare(dynamics, chromatin)
    state = coupling.initialize(atomistic, chromatin_state)

    first = coupling.step(state, thermodynamic, jax.random.key(9))
    checkpoint = coupling.checkpoint(state)
    replay = coupling.step(coupling.restore(checkpoint), thermodynamic, jax.random.key(9))

    assert first.successful & replay.successful
    np.testing.assert_allclose(
        first.accepted_state.atomistic.kinematics.positions,
        replay.accepted_state.atomistic.kinematics.positions,
    )
    np.testing.assert_allclose(
        first.evidence.initial_springs.energy,
        chromatin.springs.energy(chromatin_state.relations, positions),
    )


def test_overdamped_runtime_has_stable_addressed_replay():
    dynamics, _, _, positions = _runtime()
    runtime = phx.atomistic.OverdampedAtomisticPlan(
        1.0e-3, 0.2, 1.0, realization_id=11
    ).prepare(dynamics)
    state = runtime.initialize(positions, key=jax.random.key(3))

    left = runtime.step(state)
    right = runtime.step(state)

    assert left.successful & right.successful
    np.testing.assert_array_equal(
        left.accepted_state.positions, right.accepted_state.positions
    )
    assert int(left.accepted_state.step_index) == 1


def test_generalized_langevin_runtime_enforces_discrete_fdt_and_replays():
    dynamics, thermodynamic, atomistic, _ = _runtime()
    decay = 0.9
    transition = np.eye(2) * decay
    noise = np.eye(2) * np.sqrt(1.0 - decay**2)
    runtime = phx.atomistic.GeneralizedLangevinRuntimePlan(
        transition, noise, 1.0, realization_id=5
    ).prepare(dynamics)
    state = runtime.initialize(atomistic)

    left = runtime.step(state, thermodynamic)
    right = runtime.step(state, thermodynamic)

    assert left.successful & right.successful
    np.testing.assert_allclose(left.covariance_residual, 0.0, atol=1.0e-12)
    np.testing.assert_array_equal(
        left.accepted_state.atomistic.kinematics.momenta,
        right.accepted_state.atomistic.kinematics.momenta,
    )
    assert left.accepted_state.auxiliary.shape == (4, 3, 1)


def test_equilibrium_rheology_retains_correlation_uncertainty_evidence():
    time = np.arange(64, dtype=float)
    shear = np.sin(0.2 * time) + 0.2 * np.cos(0.7 * time)
    stress = np.zeros((64, 3, 3), dtype=float)
    stress[:, 0, 1] = shear
    stress[:, 1, 0] = shear
    result = pl.polymer_green_kubo_viscosity(
        pl.PolymerStressCorrelationPlan(
            maximum_frames=64,
            maximum_lag=8,
            block_count=4,
            minimum_origins=16,
            maximum_relative_standard_error=1.0e9,
            maximum_stationarity_drift=1.0e9,
        ),
        stress,
        0.1,
        10.0,
        1.0,
        1.0,
    )
    assert result.successful
    assert float(result.shear_correlation[0]) > 0.0
    assert jnp.isfinite(result.viscosity)
