#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


class IntervalGeometry:
    bounds = jnp.asarray([[0.0], [1.0]])

    @staticmethod
    def signed_distance(points):
        x = points[:, 0]
        return jnp.minimum(x, 1.0 - x)

    @staticmethod
    def boundary_normal(points):
        return jnp.where(points[:, :1] < 0.5, -1.0, 1.0)


def _population(name, count=6):
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=1, name=name
    ).prepare()
    return phx.discretization.ParticlePopulation(
        name, particles, role="material-phase", state_shape=(count, 3)
    )


def test_native_multi_population_cells_match_dense_bipartite_pairs():
    target = _population("target")
    source = _population("source")
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.MultiPopulationCellPlan(box, 0.25, (4, 4)).prepare(
        (target, source)
    )
    target_position = (jnp.arange(6, dtype=float) + 0.25)[:, None] / 6.0
    source_position = (jnp.arange(6, dtype=float) + 0.75)[:, None] / 6.0
    state = prepared.build((target_position, source_position))
    key = phx.discretization.ParticleSearchKey(
        target.population_id, source.population_id, 0.3
    )
    relation = prepared.bipartite_relation(
        state, (target_position, source_position), key, 36
    )
    dense = (
        phx.discretization.DenseBipartiteParticleNeighborhoodPlan(36)
        .prepare(
            target.particles,
            source.particles,
            target_population_id=target.population_id,
            source_population_id=source.population_id,
        )
        .build(target_position, source_position)
    )
    valid = np.asarray(relation.relation.valid)
    pairs = set(
        zip(
            np.asarray(relation.relation.target_particle_ids)[valid].tolist(),
            np.asarray(relation.relation.source_particle_ids)[valid].tolist(),
            strict=True,
        )
    )
    dense_valid = np.asarray(dense.relation.valid)
    dense_pairs = {
        pair
        for pair in zip(
            np.asarray(dense.relation.target_particle_ids)[dense_valid].tolist(),
            np.asarray(dense.relation.source_particle_ids)[dense_valid].tolist(),
            strict=True,
        )
        if abs(float(target_position[pair[0], 0] - source_position[pair[1], 0])) % 1.0
        < 0.3
        or 1.0 - abs(float(target_position[pair[0], 0] - source_position[pair[1], 0]))
        < 0.3
    }

    assert state.successful
    assert relation.successful
    assert pairs == dense_pairs


def test_small_batched_solver_and_adaptive_root_report_residuals():
    matrix = jnp.asarray([[[2.0, 0.5], [0.5, 1.5]], [[1.0, 0.0], [0.0, 3.0]]])
    rhs = jnp.asarray([[1.0, 2.0], [2.0, 3.0]])
    result = phx.linalg.solve_small_linear(
        phx.linalg.SmallLinearSolvePlan(2), matrix, rhs
    )
    root = phx.discretization.solve_adaptive_h_root(
        phx.discretization.AdaptiveHRootPlan(1.2, 1, 0.1, 1.0),
        jnp.asarray([1.0, 2.0]),
        lambda h: jnp.asarray([4.0, 8.0]),
        jnp.asarray([0.8, 0.8]),
    )

    assert jnp.all(result.successful)
    assert jnp.max(result.residual_norm) < 1e-12
    assert root.successful
    assert root.residual < 1e-10


def test_production_boundary_reconstruction_and_moments_are_explicit():
    kernel = phx.discretization.WendlandC2SPHKernel(1)
    wall = phx.discretization.WallParticleGenerationPlan(
        IntervalGeometry(), kernel, 0.25, 0.3, layers=1
    ).prepare()
    features = phx.discretization.classify_boundary_features(wall)
    certification = phx.discretization.certify_wall_moments(
        wall, kernel, 0.3, zeroth_tolerance=2.0, first_tolerance=2.0
    )

    assert features.kind_code.shape == (wall.quality.particle_count,)
    assert jnp.isfinite(certification.zeroth_moment_error)
    assert certification.successful


def test_shock_sensor_shifting_and_precision_certification_are_finite():
    previous = phx.discretization.ShockViscosityState(
        jnp.zeros((3,)), jnp.zeros((3,)), jnp.ones((3,)), jnp.asarray(0)
    )
    updated = phx.discretization.update_shock_viscosity(
        phx.discretization.ShockViscositySensorPlan(),
        previous,
        jnp.asarray([-1.0, 0.0, 1.0]),
        jnp.asarray([0.1, 1.0, 0.1]),
        0.01,
    )
    certified = phx.discretization.certify_particle_precision(
        jnp.asarray([1.0, 2.0]),
        jnp.asarray([1.0, 2.0 + 1e-8]),
        phx.discretization.ParticlePrecisionPolicy(),
        tolerance=1e-6,
    )

    assert jnp.all(updated.alpha >= 0.0)
    assert certified.successful


def test_iisph_operator_oracle_and_projection_qualification_are_separate():
    count = 5
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2,
        box=phx.discretization.ParticleBox([0.0], [1.0]),
    ).prepare(particles)
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    iisph = phx.discretization.PreparedIISPH(
        particles,
        neighborhood,
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
        phx.discretization.IISPHMethodPlan(1.0, maximum_iterations=2, tolerance=1.0),
    )
    oracle = phx.discretization.assemble_iisph_operator(iisph, position, 0.001)
    diagnostics = phx.discretization.diagnose_iisph_operator(oracle)
    step = iisph.step_detailed(
        0.0, iisph.initialize_state(position, jnp.zeros_like(position)), 0.001
    )

    assert oracle.action_error < 1e-12
    assert diagnostics.finite
    assert step.successful
    assert not step.production_qualified


def test_reference_domain_decomposition_updates_halos_and_migration():
    box = phx.discretization.ParticleBox([0.0], [1.0])
    plan = phx.discretization.ParticleDomainDecompositionPlan(2, 0.15, box)
    position = jnp.asarray([[0.1], [0.45], [0.55], [0.9]])
    halo = phx.discretization.prepare_particle_halos(plan, position, jnp.ones((4,), bool))
    local = phx.discretization.halo_update(jnp.arange(4.0)[:, None], halo)
    migrated = phx.discretization.migrate_particle_halos(
        plan,
        halo,
        position + jnp.asarray([[0.5], [0.0], [0.0], [-0.5]]),
        jnp.ones((4,), bool),
    )

    assert halo.successful
    assert local.shape == (2, 4, 1)
    assert migrated.migration_count == 2


def test_benchmark_registry_and_replay_round_trip(tmp_path):
    profile = phx.discretization.ParticleQualificationProfile()
    qualification = phx.discretization.ParticleQualificationResult(
        phx.discretization.ParticleMethodMaturity.EXPERIMENTAL,
        profile,
        (),
        True,
        False,
    )
    identity = phx.discretization.ParticleBenchmarkIdentity(
        "test", "configuration:test", "source:test"
    )
    record = phx.discretization.ParticleBenchmarkRecord(
        identity, qualification, (("error", 0.1),)
    )
    registry = phx.discretization.ParticleBenchmarkRegistry((record,))
    artifact = phx.discretization.ParticleQualificationArtifact(
        registry, "method:test", "code:test", "packages:test"
    )
    artifact_path = tmp_path / "qualification.json"
    phx.discretization.write_particle_qualification_artifact(artifact_path, artifact)
    packet = phx.discretization.ParticleReplayPacket(
        jnp.asarray([2.0]),
        0.1,
        3,
        jnp.asarray([1.0]),
        problem_id="problem:test",
        method_id="method:test",
        failure_status="overflow",
    )
    packet_path = tmp_path / "replay.npz"
    phx.discretization.write_particle_replay(packet_path, packet)
    recovered = phx.discretization.read_particle_replay(packet_path)

    assert artifact_path.exists()
    assert recovered.packet_id == packet.packet_id
    assert jnp.array_equal(recovered.state, packet.state)
