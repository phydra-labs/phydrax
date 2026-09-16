import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.polymer_liquids import entanglement as ent


def _snapshot_runtime(*, periodic=False):
    particle_ids = [10, 20, 30, 40, 50, 60]
    topology = phx.atomistic.MolecularTopologyPlan(
        bonds=[[10, 20], [20, 30], [40, 50], [50, 60]],
        bond_type_ids=[0, 0, 0, 0],
    )
    cell = phx.discretization.PeriodicCell(jnp.eye(3) * 10.0) if periodic else None
    system = phx.atomistic.AtomisticSystemPlan(
        particle_ids,
        [0] * 6,
        [1.0] * 6,
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=[0] * 6,
        element_mask=[False] * 6,
        molecule_ids=[0, 0, 0, 1, 1, 1],
        topology=topology,
        cell=cell,
    ).prepare()
    potential = phx.atomistic.AtomisticPotentialProgram(
        [phx.atomistic.HarmonicBondPotential([1.0], [1.0])]
    ).prepare(system)
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(15, box=cell).prepare(
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
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [1.0, 5.0, 1.0],
            [2.0, 5.0, 1.0],
            [3.0, 5.0, 1.0],
        ]
    )
    state = dynamics.initialize_state(
        positions,
        thermodynamic,
        velocity=jnp.zeros_like(positions),
        key=jax.random.key(9),
    )
    layout = phx.atomistic.PolymerChainLayoutPlan(
        [[0, 1, 2], [3, 4, 5]],
        [[True, True, True], [True, True, True]],
        maximum_frames=8,
    )
    prepared = ent.PrimitivePathSnapshotPlan(
        maximum_particles=6,
        maximum_chains=2,
        maximum_beads_per_chain=3,
        maximum_bond_length=1.1,
    ).prepare(dynamics, layout, ("left", "right"))
    return prepared, state


def test_force_primitive_path_preserves_fixed_straight_chains():
    prepared, state = _snapshot_runtime()
    snapshot, snapshot_evidence = prepared.capture(state)
    ppa = ent.ForcePrimitivePathPlan(
        bond_tension=1.0,
        excluded_volume_energy=1.0,
        excluded_volume_sigma=1.0,
        contact_distance=1.5,
        minimum_interchain_distance=0.5,
        maximum_bond_length=1.1,
        maximum_iterations=16,
        maximum_contacts=8,
    ).prepare(prepared)
    result = ppa.evaluate(snapshot)

    assert snapshot_evidence.successful & result.successful
    np.testing.assert_allclose(result.contour_lengths, [2.0, 2.0])
    np.testing.assert_allclose(result.endpoint_residual, 0.0)
    assert int(result.contact_state.count) == 0


def test_entanglement_estimators_retain_named_conventions_and_uncertainty():
    result = ent.estimate_entanglement(
        ent.EntanglementEstimatorPlan(
            0.85,
            1.0,
            block_count=4,
            minimum_frames=8,
            maximum_relative_standard_error=1.0e-12,
        ),
        [2, 2],
        jnp.full((8, 2), 4.0),
        jnp.full((8, 2), 2.0),
    )
    assert result.successful
    np.testing.assert_allclose(result.coil_entanglement_length, 2.0)
    np.testing.assert_allclose(result.plateau_modulus, (4.0 / 5.0) * 0.85 / 2.0)

    multi = ent.multi_length_kink_entanglement([50.0, 100.0, 150.0], [1.0, 2.0, 3.0])
    assert multi.successful
    np.testing.assert_allclose(multi.entanglement_length, 50.0)


def test_periodic_snapshot_and_z1plus_interchange_preserve_source_identity():
    prepared, state = _snapshot_runtime(periodic=True)
    snapshot, _ = prepared.capture(state)
    periodic = ent.periodic_primitive_path_evidence(snapshot)
    artifact = ent.export_z1plus_lammps_dump(
        ent.Z1PlusExportPlan(maximum_bytes=100_000, tool_version="3.1"), snapshot
    )
    primitive = np.asarray(snapshot.unwrapped_positions)[
        np.asarray(snapshot.chain_indices)
    ]
    oracle = ent.import_z1plus_result(
        snapshot,
        {
            "source_snapshot_id": snapshot.snapshot_id,
            "primitive_positions": primitive,
            "primitive_mask": np.asarray(snapshot.chain_mask),
            "contour_lengths": [2.0, 2.0],
            "kink_counts": [0, 0],
        },
        tool_version="3.1",
        tool_digest="fixture-digest",
    )
    assert periodic.successful & oracle.successful
    assert artifact.source_snapshot_id == oracle.source_snapshot_id
    assert b"ITEM: ATOMS id mol xu yu zu" in artifact.payload


def test_periodic_evidence_uses_last_active_bead_for_ragged_chains():
    positions = jnp.asarray(
        [
            [1.0, 1.0, 1.0],
            [2.0, 1.0, 1.0],
            [3.0, 1.0, 1.0],
            [1.0, 5.0, 1.0],
            [2.0, 5.0, 1.0],
        ]
    )
    snapshot = ent.PrimitivePathSnapshot(
        positions,
        jnp.asarray([[0, 1, 2], [3, 4, 0]], dtype=jnp.int32),
        jnp.asarray([[True, True, True], [True, True, False]]),
        jnp.arange(5),
        jnp.asarray([4.0, 1.0]),
        jnp.zeros((5, 3), dtype=jnp.int32),
        jnp.eye(3) * 10.0,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        "state",
        "snapshot",
        "prepared",
    )
    evidence = ent.periodic_primitive_path_evidence(snapshot)
    assert evidence.successful
    np.testing.assert_allclose(evidence.winding_residual, 0.0)
