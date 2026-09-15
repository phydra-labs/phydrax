import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _system():
    units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
    structure = phx.atomistic.AtomicStructure(
        [1, 1],
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        [1.0, 1.0],
        units.scale,
        particle_ids=[11, 13],
    )
    system = phx.atomistic.AtomisticSystemPlan.from_structure(
        structure, units, molecule_ids=[0, 0]
    )
    return structure, system


def _surface(system, energy_function, provider_id):
    def evaluate(positions, _cell):
        coordinate = jnp.asarray(positions)
        energy = energy_function(coordinate)
        forces = -jax.grad(energy_function)(coordinate)
        return phx.chemistry.PotentialEnergySurfaceEvaluation(
            energy,
            forces,
            None,
            True,
            provider_id=provider_id,
            source_result_id=phx.chemistry.electronic_geometry_id(system, coordinate),
        )

    return phx.chemistry.CallablePotentialEnergySurface(
        evaluate,
        system.system_id,
        system.units,
        provider_id,
        phx.chemistry.PotentialEnergySurfaceCapabilities(),
    )


def test_boundary_multipoles_and_mutual_polarization_close_variational_energy():
    structure, system = _system()
    region = phx.chemistry.QuantumRegionPlan(system, [11], spin_multiplicity=2).prepare()
    multipoles = phx.atomistic.PermanentMultipoleSiteData(
        [0.0, 0.0],
        np.zeros((2, 3)),
        np.zeros((2, 3, 3)),
        [0.0, 1.0],
        [1.0, 1.0],
    )
    permanent = phx.chemistry.multipole_embedding_for_region(
        region, structure.positions, multipoles
    )

    def quantum(region_positions, embedding, induced):
        field = jnp.asarray([[0.2, 0.0, 0.0]])
        embedded = phx.chemistry.EmbeddedRegionEvaluation(
            -jnp.sum(jnp.asarray(induced) * field),
            jnp.zeros_like(region_positions),
            jnp.zeros_like(embedding.positions),
            True,
            "toy-polarizable",
        )
        return phx.chemistry.PolarizableEmbeddedRegionEvaluation(
            embedded, field, True, "toy-polarizable"
        )

    provider = phx.chemistry.CallablePolarizableEmbeddedRegionProvider(
        quantum,
        "toy-polarizable",
        region.region_system.system_id,
        system.units,
    )
    surface = phx.chemistry.MutualPolarizableQMMMSurface(
        region,
        _surface(system, lambda positions: jnp.asarray(0.0), "zero-classical"),
        provider,
        multipoles,
        phx.atomistic.PolarizationOperatorPlan(),
        residual_tolerance=1.0e-10,
    )
    result = surface.evaluate_components(structure.positions)

    assert permanent.multipoles.site_capacity == 1
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.polarizable_embedding.induced_dipoles,
        [[0.2, 0.0, 0.0]],
        atol=1.0e-10,
    )
    np.testing.assert_allclose(result.qmmm.total_energy, -0.02, atol=1.0e-10)


def test_adaptive_partition_weights_add_exact_force_correction_and_epoch_identity():
    structure, system = _system()
    zero = _surface(system, lambda positions: jnp.asarray(0.0), "partition-zero")
    one = _surface(system, lambda positions: jnp.asarray(1.0), "partition-one")

    def weights(positions):
        logits = jnp.asarray([positions[0, 0], -positions[0, 0]])
        return jax.nn.softmax(logits)

    adaptive = phx.chemistry.AdaptivePartitionedQMMMSurface(
        (zero, one), weights, ("left", "right")
    )
    result = adaptive.evaluate_partition(structure.positions, epoch_index=4)
    derivative = 0.5

    assert bool(result.successful)
    assert result.topology_epoch.index == 4
    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1.0e-14)
    np.testing.assert_allclose(result.evaluation.energy, 0.5, atol=1.0e-14)
    np.testing.assert_allclose(result.evaluation.forces[0, 0], derivative, atol=1.0e-14)


def test_periodic_multilevel_surface_combines_energy_and_force_ledgers():
    structure, system = _system()
    first = _surface(system, lambda positions: jnp.sum(positions**2), "level-one")
    second = _surface(system, lambda positions: 2.0 * jnp.sum(positions**2), "level-two")
    surface = phx.chemistry.PeriodicMultilevelQMMMSurface(
        (first, second), (1.0, -0.25), "excited-state-1"
    )
    result = surface.evaluate(structure.positions, jnp.eye(3) * 10.0)

    assert bool(result.successful)
    np.testing.assert_allclose(result.energy, 0.5 * np.sum(structure.positions**2))
    np.testing.assert_allclose(result.forces, -np.asarray(structure.positions))
