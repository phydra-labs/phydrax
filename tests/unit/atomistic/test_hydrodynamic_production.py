import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _system(*, periodic=False):
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 10.0) if periodic else None
    return phx.atomistic.AtomisticSystemPlan(
        [0, 1],
        [0, 0],
        [1.0, 1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0, 0],
        element_mask=[False, False],
        cell=cell,
    ).prepare()


def _dynamics():
    system = _system()
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.1], [1.0], 2.5)]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(1).prepare(
        system.particles
    )
    return phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1.0e-3),
    ).prepare()


def test_free_space_rpy_and_generic_brownian_runtime():
    system = _system()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    mobility = phx.atomistic.FreeSpaceRPYMobilityPlan(
        1.0, 1.0, maximum_particles=2
    ).prepare(system, [0, 1])
    matrix = phx.atomistic.materialize_mobility(mobility, positions, maximum_dofs=6)
    np.testing.assert_allclose(matrix, matrix.T, atol=1.0e-12)
    assert np.min(np.linalg.eigvalsh(np.asarray(matrix))) > 0.0
    np.testing.assert_allclose(
        np.diag(np.asarray(matrix)), np.full(6, 1.0 / (6.0 * np.pi))
    )
    assert not mobility.configuration_valid(jnp.zeros_like(positions))

    runtime = phx.atomistic.HydrodynamicBrownianPlan(
        1.0e-3, 0.0, maximum_displacement=1.0
    ).prepare(
        _dynamics(),
        phx.atomistic.FreeSpaceRPYMobilityPlan(1.0, 1.0, maximum_particles=2),
    )
    state = runtime.initialize(positions, key=jax.random.key(5))
    step = runtime.step(state)
    assert step.successful
    np.testing.assert_allclose(step.accepted_state.positions, positions)


def test_direct_and_positive_split_periodic_mobility_are_spd():
    system = _system(periodic=True)
    positions = jnp.asarray([[1.0, 1.0, 1.0], [4.0, 1.0, 1.0]])
    direct = phx.atomistic.DirectPeriodicRPYMobilityPlan(
        0.5,
        1.0,
        2,
        maximum_particles=2,
        convergence_tolerance=1.0,
    ).prepare(system, [0, 1])
    evidence = direct.evaluate(positions)
    assert evidence.successful

    split = phx.atomistic.PositiveSplitPeriodicRPYMobilityPlan(
        0.5,
        1.0,
        2,
        1,
        maximum_particles=2,
        convergence_tolerance=1.0,
    ).prepare(system, [0, 1])
    split_evidence = split.evaluate(positions)
    assert split_evidence.successful
    np.testing.assert_allclose(
        split_evidence.total_matrix,
        split_evidence.wave_matrix + split_evidence.local_matrix,
        atol=1.0e-12,
    )


def test_confined_fib_adapter_composes_marker_transfer_and_fluid_inverse():
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(8) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    marker_position = jnp.asarray([[0.5, 0.5, 0.5]])
    markers = phx.discretization.LagrangianMarkerSetPlan(
        [0], marker_position, [1.0]
    ).prepare()
    transfer = phx.discretization.MACMarkerTransferPlan(operators, markers).prepare()
    inverse = phx.linalg.IdentityLinearOperator(operators.velocity_space)
    system = phx.atomistic.AtomisticSystemPlan(
        [0],
        [0],
        [1.0],
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0],
        element_mask=[False],
    ).prepare()
    mobility = phx.atomistic.ConfinedFIBMobilityPlan(
        transfer, inverse, maximum_particles=1
    ).prepare(system, [0])
    evidence = mobility.evaluate(marker_position, maximum_dofs=3)
    assert evidence.successful


def test_hydrodynamic_stress_and_lubrication_ledgers():
    stress = phx.atomistic.hydrodynamic_stress(
        phx.atomistic.HydrodynamicStressPlan(2.0, 10.0, 1.0),
        [[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]],
        2,
    )
    assert stress.successful
    np.testing.assert_allclose(stress.solvent_cauchy_stress[0, 1], 2.0)
    np.testing.assert_allclose(stress.pressure, 0.2)
    compressible = phx.atomistic.hydrodynamic_stress(
        phx.atomistic.HydrodynamicStressPlan(2.0, 10.0, 1.0),
        jnp.eye(3),
        2,
    )
    assert not compressible.successful

    system = _system()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [2.2, 0.0, 0.0]])
    forces = jnp.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    mobility = phx.atomistic.FreeSpaceRPYMobilityPlan(
        1.0, 1.0, maximum_particles=2
    ).prepare(system, [0, 1])
    result = phx.atomistic.hard_sphere_lubrication_correction(
        phx.atomistic.HardSphereLubricationPlan(1.0, 1.0, 1.0, 0.01, maximum_dofs=6),
        mobility,
        positions,
        forces,
    )
    assert result.successful & (result.pair_count == 1)
    uncorrected_relative = (
        result.uncorrected_velocity[0, 0] - result.uncorrected_velocity[1, 0]
    )
    corrected_relative = result.corrected_velocity[0, 0] - result.corrected_velocity[1, 0]
    assert abs(corrected_relative) < abs(uncorrected_relative)
    assert np.min(np.linalg.eigvalsh(np.asarray(result.resistance_matrix))) >= -1.0e-12
