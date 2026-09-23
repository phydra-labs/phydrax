#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.numerical_relativity._distributed import (
    NumericalRelativityAMRDistributionPlan,
)
from phydrax.applications.numerical_relativity._einstein_vlasov import (
    EinsteinVlasovMatterState,
)
from phydrax.applications.numerical_relativity._einstein_vlasov_amr import (
    EinsteinVlasovAMRStressTransferPlan,
    EinsteinVlasovCheckpointPlan,
    EinsteinVlasovParticleMigrationPlan,
)
from phydrax.applications.numerical_relativity._state import flat_z4c_state
from phydrax.discretization import (
    BlockHierarchyPlan,
    BlockLevelPlan,
    BlockTopologyCompiler,
    FDAMRHierarchyPlan,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization.particle._relativistic_stress_transfer import (
    RelativisticParticleState,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.metrix import ADMGridGeometry, StressEnergyProjection


def _geometry(shape, token, topology):
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    return ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype="bool"),
        jnp.ones(shape, dtype="bool"),
        snapshot_token=token,
        chart_id="cartesian",
        convention_id="mostly-plus",
        scale_id="geometric",
        topology_id=topology,
        geometry_lineage_id=f"geometry:{topology}",
    )


def _projection(geometry):
    shape = geometry.leading_shape
    momentum = jnp.broadcast_to(jnp.asarray((0.2, -0.1, 0.05)), shape + (3,))
    stress = jnp.broadcast_to(jnp.diag(jnp.asarray((0.3, 0.2, 0.1))), shape + (3, 3))
    return StressEnergyProjection(
        jnp.ones(shape),
        momentum,
        stress,
        geometry.active,
        geometry.valid,
        jnp.zeros(shape),
        jnp.zeros(shape),
        snapshot_token=geometry.snapshot_token,
        geometry_lineage_id=geometry.geometry_lineage_id,
        convention_id=geometry.convention_id,
        scale_id=geometry.scale_id,
        topology_id=geometry.topology_id,
        projection_id=f"projection:{geometry.topology_id}",
    )


def _distribution():
    base = TensorGridPlan(
        tuple(UniformCellAxisSpec(4, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))))
    hierarchy = BlockHierarchyPlan(
        base,
        (
            BlockLevelPlan(0, (2, 2, 2), 8, halo_width=1),
            BlockLevelPlan(1, (2, 2, 2), 8, halo_width=1),
        ),
    )
    topology = BlockTopologyCompiler(hierarchy).initialize()
    prepared = FDAMRHierarchyPlan(hierarchy).prepare()
    return NumericalRelativityAMRDistributionPlan("z4c", hierarchy, 1).prepare(
        topology, prepared
    )


def _particles():
    ids = jnp.asarray((8, 2, 6, 4), dtype=jnp.int64)
    active = jnp.ones((4,), dtype="bool")
    return RelativisticParticleState(
        ids,
        jnp.ones((4,)),
        jnp.asarray(((0.1, 0.1, 0.1), (0.3, 0.2, 0.1), (0.6, 0.3, 0.2), (0.8, 0.8, 0.8))),
        jnp.zeros((4, 3)),
        jnp.zeros((4, 3)),
        jnp.ones((4,), dtype=jnp.int32),
        active,
        jnp.zeros((4,), dtype=jnp.int32),
        ids,
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        jnp.asarray(0, dtype=jnp.int32),
        frame_id="eulerian-frame",
        topology_id="particle-grid",
        frame_lineage_id="frame-lineage",
    )


def _repository(tmp_path):
    profile = HPCFilesystemProfile(
        "posix.einstein-vlasov",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    return POSIXArtifactRepository(
        tmp_path / "repository",
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=4096,
            maximum_metadata_bytes=64 * 1024,
        ),
    )


def test_amr_stress_transfer_preserves_all_adm_source_integrals():
    coarse_geometry = _geometry((2, 2, 2), 1, "coarse")
    fine_geometry = _geometry((4, 4, 4), 2, "fine")
    coarse = _projection(coarse_geometry)
    transfer = EinsteinVlasovAMRStressTransferPlan(conservation_tolerance=1.0e-6)
    prolonged = transfer.transfer_projection(
        coarse,
        fine_geometry,
        1.0,
        direction="prolong",
        target_proper_volume=0.125,
    )

    assert bool(prolonged.evidence.qualified)
    np.testing.assert_allclose(
        prolonged.evidence.target_integrals,
        prolonged.evidence.source_integrals,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert bool(prolonged.projection.compatible_with(fine_geometry))

    restricted = transfer.transfer_projection(
        prolonged.projection,
        coarse_geometry,
        0.125,
        direction="restrict",
        target_proper_volume=1.0,
    )
    assert bool(restricted.evidence.qualified)
    np.testing.assert_allclose(
        restricted.projection.energy_density,
        coarse.energy_density,
        rtol=0.0,
        atol=1.0e-6,
    )


def test_owner_computes_routes_are_stable_and_capacity_fail_closed():
    distribution = _distribution()
    particles = _particles()
    migration = EinsteinVlasovParticleMigrationPlan(distribution, 4)
    first = migration.route(
        particles,
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.asarray((3, 1, 2, 0), dtype=jnp.int32),
    )

    assert bool(first.committed)
    np.testing.assert_array_equal(first.route.owner, 0)
    ordered_ids = particles.particle_ids[first.route.stable_order]
    np.testing.assert_array_equal(ordered_ids, jnp.asarray((2, 4, 6, 8)))
    packed = migration.owner_packed(first.route, particles.particle_ids)
    np.testing.assert_array_equal(packed[0], ordered_ids)

    moved = migration.route(
        particles,
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.asarray((2, 1, 3, 0), dtype=jnp.int32),
        predecessor=first.route,
    )
    assert int(moved.route.migration_count) == 2
    assert not bool(moved.derivative_valid)

    overflow = EinsteinVlasovParticleMigrationPlan(distribution, 3).route(
        particles,
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.asarray((3, 1, 2, 0), dtype=jnp.int32),
    )
    assert not bool(overflow.committed)
    assert not bool(overflow.route.capacity_valid)


def test_checkpoint_restart_preserves_particles_routes_and_allows_resharding(tmp_path):
    distribution = _distribution()
    particles = _particles()
    migration = EinsteinVlasovParticleMigrationPlan(distribution, 4)
    routed = migration.route(
        particles,
        jnp.zeros((4,), dtype=jnp.int32),
        jnp.asarray((3, 1, 2, 0), dtype=jnp.int32),
    ).route
    z4c = flat_z4c_state((5, 5, 5), grid_id="ev-grid")
    state = EinsteinVlasovMatterState(
        z4c,
        particles,
        particles.time,
        3,
        1,
        0,
        False,
        runtime_id="einstein-vlasov-runtime",
    )
    checkpoint = EinsteinVlasovCheckpointPlan(
        state,
        routed,
        migration,
        topology_epoch=4,
        analysis_plan_id="analysis",
        numeric_revision_id="numeric",
        execution_plan_id="execution",
        stress_plan_id="stress",
        frame_provider_id="frame-provider",
        placement_id="one-device",
    )
    local_path = tmp_path / "einstein-vlasov.phxcheckpoint"
    checkpoint.write_local(local_path, state, routed)
    local = checkpoint.read_local(
        local_path,
        state,
        routed,
        target_placement_id="new-placement",
    )
    assert bool(local.exact)
    assert bool(local.placement_changed)
    np.testing.assert_array_equal(
        local.state.particles.particle_ids, particles.particle_ids
    )
    np.testing.assert_array_equal(local.route.owner, routed.owner)

    repository = _repository(tmp_path)
    publication = checkpoint.publish(repository, state, routed, writer_id="rank-0")
    manifest = checkpoint.assemble(repository, (publication,), expected_process_count=1)
    restored = checkpoint.restore(
        repository,
        manifest,
        state,
        routed,
        target_placement_id="resharded-layout",
    )
    assert bool(restored.exact)
    assert bool(restored.placement_changed)
    assert not bool(restored.derivative_valid)
    np.testing.assert_array_equal(restored.state.z4c.values, state.z4c.values)
    np.testing.assert_array_equal(
        restored.state.particles.frame_token, state.particles.frame_token
    )
    np.testing.assert_array_equal(restored.route.local_slot, routed.local_slot)
    child_checkpoint = EinsteinVlasovCheckpointPlan(
        state,
        routed,
        migration,
        topology_epoch=5,
        analysis_plan_id="analysis",
        numeric_revision_id="numeric",
        execution_plan_id="execution",
        stress_plan_id="stress",
        frame_provider_id="frame-provider",
        placement_id="one-device",
    )
    child_publication = child_checkpoint.publish(
        repository,
        state,
        routed,
        writer_id="rank-0-child",
        parent_manifest=manifest,
    )
    child_manifest = child_checkpoint.assemble(
        repository,
        (child_publication,),
        expected_process_count=1,
        parent_manifest=manifest,
    )
    assert child_manifest.parent_manifest_id == manifest.manifest_id
    with pytest.raises(ValueError, match="exact parent"):
        child_checkpoint.restore(
            repository,
            child_manifest,
            state,
            routed,
        )
    child_restored = child_checkpoint.restore(
        repository,
        child_manifest,
        state,
        routed,
        parent_manifest=manifest,
    )
    assert bool(child_restored.exact)
