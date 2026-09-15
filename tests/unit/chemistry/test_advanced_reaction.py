import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _units():
    return phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()


def _surface(system, evaluator, provider_id):
    def wrapped(positions, _cell):
        energy, forces = evaluator(jnp.asarray(positions))
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            None,
            True,
            provider_id=provider_id,
            source_result_id=phx.chemistry.electronic_geometry_id(system, positions),
        )

    return phx.chemistry.CallablePotentialEnergySurface(
        wrapped,
        system.system_id,
        system.units,
        provider_id,
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )


def test_internal_coordinate_retraction_and_bfgs_optimization_reach_bond_minimum():
    coordinates = phx.chemistry.MolecularCoordinateSystemPlan(2, bonds=[(0, 1)])
    initial_positions = jnp.asarray([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    retracted = coordinates.retract(initial_positions, [-0.2])

    units = _units()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        initial_positions,
        [1.0, 1.0],
        units.scale,
        particle_ids=[1, 2],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )

    def harmonic_bond(positions):
        delta = positions[1] - positions[0]
        distance = jnp.linalg.norm(delta)
        energy = 0.5 * (distance - 1.0) ** 2
        direction = delta / distance
        force_second = -(distance - 1.0) * direction
        forces = jnp.stack((-force_second, force_second))
        return energy, forces

    optimized = phx.chemistry.InternalCoordinateOptimizationPlan(
        system,
        _surface(system, harmonic_bond, "harmonic-bond"),
        coordinates,
        force_tolerance=1.0e-7,
        internal_gradient_tolerance=1.0e-7,
        trust_radius=0.2,
        maximum_iterations=20,
    ).run(structure)

    assert bool(retracted.successful)
    np.testing.assert_allclose(retracted.achieved_values, [1.2], atol=1.0e-10)
    assert bool(optimized.successful)
    np.testing.assert_allclose(optimized.internal_coordinates[-1], [1.0], atol=2.0e-7)


def test_dimer_refines_double_well_saddle_and_rates_propagate_conservatively():
    units = _units()
    structure = phx.atomistic.AtomicStructure(
        [1], [[0.2, 0.0, 0.0]], [1.0], units.scale, particle_ids=[7]
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0]
    )

    def double_well(positions):
        x, y, z = positions[0]
        energy = (x * x - 1.0) ** 2 + 0.5 * y * y + 0.5 * z * z
        forces = jnp.asarray([[-4.0 * x * (x * x - 1.0), -y, -z]])
        return energy, forces

    saddle = phx.chemistry.DimerSaddleRefinementPlan(
        system,
        _surface(system, double_well, "dimer-double-well"),
        translation_step=0.05,
        rotation_step=0.1,
        force_tolerance=1.0e-7,
        maximum_iterations=100,
    ).run(structure, [[1.0, 0.0, 0.0]])
    rate = phx.chemistry.TransitionStateRatePlan(
        300.0,
        1.0,
        2.0,
        hbar=1.0,
    ).evaluate(0.0, 3.0, imaginary_angular_frequency=0.5)
    network = phx.chemistry.ReactionNetworkPlan([[0.0, 2.0], [1.0, 0.0]]).propagate(
        [1.0, 0.0], [0.0, 0.5, 2.0]
    )

    assert bool(saddle.successful)
    np.testing.assert_allclose(saddle.positions[0, 0], 0.0, atol=2.0e-7)
    assert float(saddle.curvature) < 0.0
    assert bool(rate.successful) and float(rate.rate) > float(rate.classical_rate)
    assert bool(network.successful)
    np.testing.assert_allclose(np.sum(network.populations, axis=1), 1.0, atol=1.0e-12)
