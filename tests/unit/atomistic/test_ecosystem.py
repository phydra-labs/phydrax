from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _system(*, coordinate_map=None, cell=None):
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    plan = phx.atomistic.AtomisticSystemPlan(
        [10, 20, 30],
        [1, 1, 1],
        [1.0, 1.0, 1.0],
        units,
        atom_type_ids=[0, 0, 0],
        charges=[0.4, -0.2, -0.2],
        coordinate_map=coordinate_map,
        cell=cell,
    )
    return plan.prepare(), units


def _runtime():
    system, _ = _system()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(
        system.particles
    )
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.2], [1.0], 2.5)]
    ).prepare(system)
    dynamics = phx.atomistic.AtomisticDynamicsPlan(
        system,
        potential,
        neighborhood,
        phx.atomistic.VelocityVerletPlan(1e-3),
    ).prepare()
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
    state = dynamics.initialize_state(
        positions, velocity=jnp.zeros_like(positions), key=jax.random.key(0)
    )
    return system, neighborhood, potential, dynamics, state


def test_identity_and_virtual_site_force_pullback():
    physical_ids = np.asarray([10, 20, 30])
    sites = phx.atomistic.AtomisticInteractionSitePlan(
        [10, 20, 30, 40],
        [1, 1, 1, 0],
        [0, 0, 0, 1],
        [0.4, -0.2, -0.2, 0.1],
        physical_mask=[True, True, True, False],
    )
    rule = phx.atomistic.VirtualSiteRule(
        phx.atomistic.VirtualSiteKind.LOCAL_FRAME,
        40,
        physical_ids,
        [0.2, 0.1, 0.0],
    )
    mapping = phx.atomistic.AtomisticCoordinateMapPlan(
        physical_ids, sites, [0, 1, 2, -1], virtual_rules=(rule,)
    )
    system, _ = _system(coordinate_map=mapping)
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    realized = system.coordinate_map.realize(positions)
    assert bool(realized.successful)
    np.testing.assert_allclose(realized.positions[3], [0.2, 0.1, 0.0], atol=1e-12)
    site_force = jnp.zeros((4, 3)).at[3, 0].set(1.0)
    pulled = system.coordinate_map.force_pullback(positions, site_force)
    gradient = jax.grad(
        lambda value: system.coordinate_map.realize(value).positions[3, 0]
    )(positions)
    np.testing.assert_allclose(pulled, gradient, atol=1e-12)


def test_force_field_bundle_and_new_terms_are_energy_derived():
    system, units = _system()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(3).prepare(
        system.particles
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.2, 0.0]])
    terms = [
        phx.atomistic.MorsePotential([[0.2]], [[2.0]], [[1.0]], 2.5),
        phx.atomistic.BuckinghamPotential([[1.0]], [[2.0]], [[0.1]], 2.5),
    ]
    program = phx.atomistic.AtomisticPotentialProgram(terms).prepare(system)
    evaluation = program.evaluate(positions, neighborhood.build(positions))
    assert bool(evaluation.successful)
    assert bool(jnp.isfinite(evaluation.energy))
    provenance = phx.atomistic.AtomisticForceFieldProvenance(
        "native", ("source",), "custom", "explicit"
    )
    bundle = phx.atomistic.AtomisticForceFieldPlan(
        system.plan,
        phx.atomistic.AtomisticPotentialProgram(
            (*terms, phx.atomistic.DirectCoulombPotential())
        ),
        phx.atomistic.AtomisticNonbondedPolicy(2.5, electrostatics="direct"),
        provenance,
    ).prepare()
    assert dict(bundle.preparation.resource_counts)["interaction_sites"] == 3
    assert bundle.system.plan.units.unit_system_id == units.unit_system_id


def test_frame_xyz_h5md_and_rerun_roundtrip(tmp_path: Path):
    system, neighborhood, potential, dynamics, state = _runtime()
    reporter = phx.atomistic.AtomisticReporterPlan(
        phx.atomistic.interchange.ExtendedXYZTrajectoryPlan(tmp_path / "trajectory.xyz")
    )
    frame = reporter.frame(dynamics, state)
    with reporter.sink.open(append=False) as writer:
        writer.write(frame)
    with reporter.sink.open() as reader:
        observed = tuple(reader)
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0].positions, frame.positions)
    h5 = phx.atomistic.interchange.H5MDTrajectoryPlan(tmp_path / "trajectory.h5")
    with h5.open(append=False) as writer:
        writer.write(frame)
    with h5.open() as reader:
        h5_frames = tuple(reader)
    np.testing.assert_allclose(h5_frames[0].positions, frame.positions)
    source = phx.atomistic.InMemoryTrajectorySourcePlan((frame,))
    rerun = phx.atomistic.AtomisticRerunPlan(source, potential, neighborhood).run()
    assert bool(rerun.successful)
    np.testing.assert_allclose(
        rerun.evaluations[0][0].energy, state.force.potential_energy
    )


def test_collective_variables_bias_replica_and_free_energy():
    system, _, _, dynamics, state = _runtime()
    cv = phx.atomistic.sampling.CollectiveVariablePlan(
        phx.atomistic.sampling.CollectiveVariableKind.DISTANCE, [0, 1]
    ).prepare(system)
    program = phx.atomistic.sampling.CollectiveVariableProgram((cv,))
    value, valid = program.evaluate(state.kinematics.positions)
    np.testing.assert_allclose(value, [1.2])
    assert bool(valid)
    bias = phx.atomistic.sampling.PreparedAtomisticBias(
        phx.atomistic.sampling.AtomisticBiasPlan(
            phx.atomistic.sampling.BiasKind.HARMONIC,
            program,
            center=[1.0],
            stiffness=[2.0],
        ),
        dynamics,
    )
    bias_state = bias.plan.initialize()
    bias_value = bias.evaluate(state.kinematics.positions, bias_state, state.time)
    assert bool(bias_value.successful)
    np.testing.assert_allclose(bias_value.energy, 0.04, atol=1e-12)
    replica_plan = phx.atomistic.sampling.AtomisticReplicaEnsemblePlan([1.0, 2.0])
    replica = phx.atomistic.sampling.initialize_replica_state(
        replica_plan,
        jnp.stack((state.kinematics.positions, state.kinematics.positions)),
        jnp.stack((state.kinematics.momenta, state.kinematics.momenta)),
        [[0.0, 1.0], [1.0, 0.0]],
        jax.random.key(4),
    )
    exchanged = phx.atomistic.sampling.replica_exchange_step(replica_plan, replica, 1.0)
    assert bool(exchanged.successful)
    fep = phx.uq.free_energy_perturbation([0.0, 0.0, 0.0])
    np.testing.assert_allclose(fep.free_energies[1], 0.0, atol=1e-12)
    bar = phx.uq.bennett_acceptance_ratio([1.0, 1.0], [-1.0, -1.0])
    assert bool(jnp.isfinite(bar.free_energies[1]))
    np.testing.assert_allclose(bar.free_energies[1], 1.0, atol=1.0e-12)


def test_committee_advanced_physics_and_distributed_contracts():
    system, neighborhood, potential, _, state = _runtime()
    other = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.LennardJonesPotential([0.21], [1.0], 2.5)]
    ).prepare(system)
    committee = phx.atomistic.CommitteeAtomisticPotential(
        (potential, other),
        phx.atomistic.CommitteeReductionPolicy(1.0, 1.0, 1.0),
    )
    evidence = committee.evaluate(state.kinematics.positions, state.neighborhood)
    assert bool(evidence.successful)
    multipoles = phx.atomistic.PermanentMultipoleSiteData(
        [0.4, -0.2, -0.2],
        jnp.zeros((3, 3)),
        jnp.zeros((3, 3, 3)),
        [0.1, 0.1, 0.1],
        [1.0, 1.0, 1.0],
    )
    polarization = phx.atomistic.evaluate_polarization(
        phx.atomistic.PolarizationPlan(maximum_iterations=200, tolerance=1e-6),
        state.kinematics.positions,
        multipoles,
    )
    assert bool(polarization.successful)
    normal = phx.atomistic.RingPolymerNormalModePlan(2, 1.0)
    q = jnp.stack((state.kinematics.positions, state.kinematics.positions))
    p = jnp.zeros_like(q)
    propagated_q, propagated_p = normal.propagate(q, p, system.plan.masses, 0.01)
    assert propagated_q.shape == q.shape
    assert propagated_p.shape == p.shape
    box = phx.discretization.ParticleBox([0.0, 0.0, 0.0], [4.0, 4.0, 4.0])
    distributed = phx.atomistic.DistributedAtomisticPlan(
        system,
        phx.discretization.ParticleDomainDecompositionPlan(2, 2.5, box),
    ).prepare(state.kinematics.positions)
    distributed_eval, local_energy = phx.atomistic.halo_short_range_evaluate(
        phx.atomistic.DistributedAtomisticPlan(
            system, phx.discretization.ParticleDomainDecompositionPlan(2, 2.5, box)
        ),
        distributed,
        potential,
        state.neighborhood,
    )
    assert bool(distributed_eval.successful)
    assert local_energy.shape == (2,)
    np.testing.assert_allclose(distributed_eval.forces, state.force.forces, atol=1.0e-12)
    np.testing.assert_allclose(
        jnp.sum(local_energy), state.force.potential_energy, atol=1.0e-12
    )
