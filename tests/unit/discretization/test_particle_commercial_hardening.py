#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh
from numpy.typing import NDArray

import phydrax as phx


class IntervalGeometry:
    bounds = jnp.asarray([[0.0], [1.0]])

    @staticmethod
    def signed_distance(points: Any) -> Any:
        x = points[:, 0]
        return jnp.minimum(x, 1.0 - x)

    @staticmethod
    def boundary_normal(points: Any) -> Any:
        return jnp.where(points[:, :1] < 0.5, -1.0, 1.0)


def _population(name: Any, count: Any = 6) -> Any:
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.ones((count,)), ambient_dimension=1, name=name
    ).prepare()
    return phx.discretization.ParticlePopulation(
        name, particles, role="material-phase", state_shape=(count, 3)
    )


def test_particle_commercial_hardening_scenario_1() -> None:
    target = _population("target")
    source = _population("source")
    # ty: ignore[invalid-argument-type]
    box = phx.discretization.ParticleBox([0.0], [1.0])
    prepared = phx.discretization.MultiPopulationCellPlan(box, 0.25, (4, 4)).prepare(
        (target, source)
    )
    target_position = (jnp.arange(6, dtype="float64") + 0.25)[:, None] / 6.0
    source_position = (jnp.arange(6, dtype="float64") + 0.75)[:, None] / 6.0
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


def test_particle_commercial_hardening_scenario_2() -> None:
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
    count = 5
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    neighborhood = phx.discretization.DenseParticleNeighborhoodPlan(
        count * (count - 1) // 2,
        # ty: ignore[invalid-argument-type]
        box=phx.discretization.ParticleBox([0.0], [1.0]),
    ).prepare(particles)
    position = (jnp.arange(count, dtype="float64") + 0.5)[:, None] * spacing
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
        # ty: ignore[invalid-argument-type]
        0.0,
        iisph.initialize_state(position, jnp.zeros_like(position)),
        # ty: ignore[invalid-argument-type]
        0.001,
    )

    assert oracle.action_error < 1e-12
    assert diagnostics.finite
    assert step.successful
    assert not step.production_qualified
    # ty: ignore[invalid-argument-type]
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
    assert bool(halo.halo_mask[1, 0])
    assert bool(halo.halo_mask[0, 3])


def test_distributed_runtime_rejects_invalid_active_stable_ids() -> None:
    layout = phx.solver.DistributedParticleLayout(
        1,
        2,
        np.asarray((0, np.iinfo(np.uint32).max), dtype=np.uint32),
    )
    mesh = Mesh(np.asarray(jax.devices()[:1]), ("particles",))
    runtime = phx.discretization.DistributedParticleRuntimePlan(
        layout,
        (1.0,),
    ).prepare(mesh, "particles")
    with pytest.raises(RuntimeError, match="invalid stable IDs"):
        runtime.initialize(
            jnp.asarray(((0.25,), (0.75,))),
            jnp.zeros((2, 1)),
            jnp.ones((2,)),
            jnp.asarray((-1, 4), dtype=jnp.int64),
            jnp.asarray((True, True)),
            1.0,
        )


def test_benchmark_registry_and_replay_round_trip(tmp_path: Any) -> None:
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
    changed_packet = phx.discretization.ParticleReplayPacket(
        jnp.asarray([3.0]),
        0.1,
        3,
        jnp.asarray([1.0]),
        problem_id="problem:test",
        method_id="method:test",
        failure_status="overflow",
    )
    assert changed_packet.packet_id != packet.packet_id
    packet_path = tmp_path / "replay.npz"
    phx.discretization.write_particle_replay(packet_path, packet)
    recovered = phx.discretization.read_particle_replay(packet_path)

    assert artifact_path.exists()
    assert recovered.packet_id == packet.packet_id
    assert jnp.array_equal(recovered.state, packet.state)


def _feature_wall(
    positions: NDArray[np.floating], normals: NDArray[np.floating], spacing: float
) -> phx.discretization.PreparedWallParticles:
    count, dimension = positions.shape
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count, dtype=jnp.int32),
        jnp.ones((count,), dtype=positions.dtype),
        ambient_dimension=dimension,
    ).prepare()

    def scalar(value: float) -> jax.Array:
        return jnp.asarray(value, dtype=positions.dtype)

    quality = phx.discretization.WallParticleQualityReport(
        count, scalar(1.0), scalar(1.0), scalar(spacing), scalar(0.0), "feature-quality"
    )
    return phx.discretization.PreparedWallParticles(
        jnp.asarray(positions),
        jnp.asarray(normals),
        jnp.ones((count,), dtype=positions.dtype),
        particles,
        quality,
        "feature-wall",
    )


def _feature_reference(
    positions: NDArray[np.floating],
    normals: NDArray[np.floating],
    spacing: float,
    edge_angle: float,
    corner_angle: float,
) -> tuple[
    NDArray[np.int32], NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]
]:
    count = len(positions)
    codes = np.zeros((count,), dtype=np.int32)
    variation = np.zeros((count,), dtype=np.float64)
    minimum = np.full((count,), np.inf, dtype=np.float64)
    for target in range(count):
        for source in range(count):
            distance = math.sqrt(
                math.fsum(
                    (float(a) - float(b)) ** 2
                    for a, b in zip(positions[target], positions[source], strict=True)
                )
            )
            if distance > 0.0:
                minimum[target] = min(minimum[target], distance)
                if distance < 2.5 * spacing:
                    dot = float(np.dot(normals[target], normals[source]))
                    angle = math.acos(min(1.0, max(-1.0, dot)))
                    variation[target] = max(variation[target], angle)
        if variation[target] >= corner_angle:
            codes[target] = 3
        elif variation[target] >= edge_angle:
            codes[target] = 1
    ambiguous = ~np.isfinite(minimum) | (minimum < 0.25 * spacing)
    return codes, variation, minimum, ambiguous


@pytest.mark.parametrize(
    "count", [1, 9, 257], ids=["isolated", "parallel", "ragged-tiles"]
)
@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64], ids=["float32", "float64"])
def test_boundary_features_brute_force_reference(
    count: int, dimension: int, dtype: type[np.floating]
) -> None:
    indices = np.arange(count, dtype=np.int32)
    positions = np.zeros((count, dimension), dtype=dtype)
    positions[:, 0] = indices * 0.11
    positions[:, 1] = 0.03 * np.sin(indices)
    # Duplicates are excluded from the minimum, while nearby distinct points
    # mark ambiguous thin gaps and remote points do not influence feature kind.
    if count > 1:
        positions[1] = positions[0]
        positions[2] = positions[0] + 0.005
    angles = np.where(indices < 3, 0.0, np.where(indices < 6, 0.5, 1.2))
    normals = np.zeros_like(positions)
    normals[:, 0] = np.cos(angles)
    normals[:, 1] = np.sin(angles)
    wall = _feature_wall(positions, normals, 0.1)
    result = jax.jit(phx.discretization.classify_boundary_features)(wall)
    expected = _feature_reference(positions, normals, 0.1, 0.35, 0.8)
    tolerance = 5.0e-4 if dtype == np.float32 else 3.0e-8
    np.testing.assert_array_equal(result.kind_code, expected[0])
    np.testing.assert_allclose(result.normal_variation, expected[1], atol=tolerance)
    np.testing.assert_allclose(result.minimum_separation, expected[2], rtol=tolerance)
    np.testing.assert_array_equal(result.ambiguous, expected[3])
    assert wall.prepared_id == "feature-wall"


def test_boundary_features_threshold_equality_and_strict_neighbor_radius() -> None:
    # Non-unit normals exercise clipping at both ends of the arccos domain.
    positions = np.asarray([[0.0, 0.0], [0.25, 0.0], [0.5, 0.0]], dtype=np.float64)
    normals = np.asarray([[2.0, 0.0], [-2.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    wall = _feature_wall(positions, normals, 0.1)
    isolated = phx.discretization.classify_boundary_features(wall)
    np.testing.assert_array_equal(isolated.kind_code, 0)
    np.testing.assert_array_equal(isolated.normal_variation, 0.0)
    np.testing.assert_array_equal(isolated.ambiguous, False)
    close_wall = _feature_wall(positions, normals, 0.11)
    result = phx.discretization.classify_boundary_features(
        close_wall, edge_angle=math.pi / 2.0, corner_angle=math.pi
    )
    np.testing.assert_array_equal(result.kind_code, [3, 3, 1])
    np.testing.assert_allclose(result.normal_variation, [math.pi, math.pi, math.pi / 2.0])
    np.testing.assert_array_equal(result.ambiguous, False)


@pytest.mark.parametrize("invalid_field", ["positions", "normals"])
def test_boundary_features_nonfinite_evidence(invalid_field: str) -> None:
    positions = np.asarray([[0.0, 0.0], [0.1, 0.0]], dtype=np.float64)
    normals = np.asarray([[1.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    if invalid_field == "positions":
        positions[:] = np.nan
    else:
        normals[:] = np.nan
    wall = _feature_wall(positions, normals, 0.1)
    result = phx.discretization.classify_boundary_features(wall)
    np.testing.assert_array_equal(result.kind_code, 0)
    if invalid_field == "positions":
        np.testing.assert_array_equal(result.normal_variation, 0.0)
        np.testing.assert_array_equal(result.minimum_separation, np.inf)
        np.testing.assert_array_equal(result.ambiguous, True)
    else:
        assert np.isnan(np.asarray(result.normal_variation)).all()
        np.testing.assert_allclose(result.minimum_separation, 0.1)
        np.testing.assert_array_equal(result.ambiguous, False)


def test_boundary_features_empty_wall_reduction_error_contract() -> None:
    wall = _feature_wall(
        np.zeros((1, 2), dtype=np.float64),
        np.asarray([[1.0, 0.0]], dtype=np.float64),
        0.1,
    )
    empty = jnp.zeros((0, 2), dtype=jnp.float64)
    wall = eqx.tree_at(
        lambda value: (value.positions, value.normals), wall, (empty, empty)
    )
    with pytest.raises(
        ValueError,
        match="zero-size array to reduction operation max which has no identity",
    ):
        phx.discretization.classify_boundary_features(wall)
