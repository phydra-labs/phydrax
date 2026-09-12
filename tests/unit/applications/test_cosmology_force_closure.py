import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


cosmology = phx.applications.cosmology


def test_local_curvature_validity_is_flat_exact_and_quadratic():
    plan = cosmology.LocalCurvatureValidityPlan(
        light_speed=1.0,
        geometry_error_budget=1.0e-3,
        support_kind="periodic-box-diagonal",
    )
    flat = plan.evaluate(cosmology.FLRWBackground(1.0, 0.3), 0.1)
    assert bool(flat.successful)
    np.testing.assert_allclose(flat.support_ratio, 0.0)
    curved = plan.evaluate(
        cosmology.FLRWBackground(1.0, 0.3, curvature_density=0.01), 0.1
    )
    np.testing.assert_allclose(curved.support_ratio, 0.01)
    np.testing.assert_allclose(curved.volume_indicator, 0.01**2 / 5.0)
    assert bool(curved.successful)


def test_periodic_ewald_is_symmetric_and_near_field_gate_is_fail_closed():
    ewald = cosmology.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.02,
        alpha=5.0,
        real_shells=2,
        reciprocal_modes=4,
    )
    positions = jnp.asarray([[0.25, 0.5, 0.5], [0.75, 0.5, 0.5]])
    masses = jnp.ones((2,))
    result = ewald.evaluate(positions, masses)
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.acceleration[0], -result.acceleration[1], atol=1e-10
    )
    gate = cosmology.MeshMatchedNearFieldGate(
        cutoff=0.75,
        maximum_pairs=4,
        maximum_relative_error=1e-6,
    )
    accepted = gate.evaluate(positions, result.acceleration, result.acceleration)
    assert bool(accepted["approved"])
    rejected = gate.evaluate(
        positions, result.acceleration, jnp.zeros_like(result.acceleration)
    )
    assert not bool(rejected["approved"])


def test_screened_ewald_radius_route_matches_zero_shell_and_fails_closed():
    positions = jnp.asarray([[0.20, 0.30, 0.40], [0.70, 0.60, 0.50], [0.40, 0.80, 0.20]])
    masses = jnp.asarray([1.0, 0.8, 1.2])
    common = {
        "softening": 0.02,
        "alpha": 4.0,
        "real_shells": 0,
        "reciprocal_modes": 3,
    }
    direct = cosmology.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        **common,
    ).evaluate(positions, masses)
    radius = cosmology.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        real_space_execution="screened_radius",
        real_cutoff=2.0,
        maximum_real_pairs=positions.shape[0] ** 2,
        **common,
    ).evaluate(positions, masses)
    assert bool(radius.successful)
    assert radius.evidence.real_space_execution == "screened_radius"
    assert int(radius.evidence.required_real_pairs) == 6
    np.testing.assert_allclose(radius.acceleration, direct.acceleration)

    exhausted = cosmology.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        real_space_execution="screened_radius",
        real_cutoff=2.0,
        maximum_real_pairs=1,
        **common,
    ).evaluate(positions, masses)
    assert not bool(exhausted.successful)
    assert bool(exhausted.evidence.real_pair_overflow)
    np.testing.assert_array_equal(exhausted.acceleration, 0.0)


def test_screened_ewald_radius_route_is_filter_jittable():
    plan = cosmology.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.02,
        alpha=4.0,
        real_shells=0,
        reciprocal_modes=3,
        real_space_execution="screened_radius",
        real_cutoff=2.0,
        maximum_real_pairs=9,
    )
    positions = jnp.asarray([[0.20, 0.30, 0.40], [0.70, 0.60, 0.50], [0.40, 0.80, 0.20]])
    result = eqx.filter_jit(plan.evaluate)(positions, jnp.asarray([1.0, 0.8, 1.2]))
    assert bool(result.successful)
    assert int(result.evidence.required_real_pairs) == 6


def test_screened_ewald_radius_route_resolves_periodic_image_seam():
    positions = jnp.asarray([[0.05, 0.5, 0.5], [0.95, 0.5, 0.5]])
    masses = jnp.asarray([1.0, 0.8])
    plan = cosmology.PeriodicEwaldForcePlan(
        (1.0, 1.0, 1.0),
        1.0,
        softening=0.02,
        alpha=4.0,
        real_shells=1,
        reciprocal_modes=2,
        real_space_execution="screened_radius",
        real_cutoff=0.2,
        maximum_real_pairs=2 * 2 * 27,
    )
    result = plan.evaluate(positions, masses)
    target = positions[:, None, None, :]
    source = positions[None, :, None, :] + plan.real_offsets[None, None, :, :]
    displacement = source - target
    radius_squared = jnp.sum(displacement**2, axis=-1) + plan.softening**2
    radius = jnp.sqrt(radius_squared)
    geometric_radius = jnp.sqrt(jnp.sum(displacement**2, axis=-1))
    zero_offset = (
        jnp.arange(plan.real_offsets.shape[0], dtype=jnp.int32) == plan.zero_offset_index
    )
    self_pair = (
        jnp.eye(positions.shape[0], dtype=bool)[:, :, None] & zero_offset[None, None, :]
    )
    valid = ~self_pair & (geometric_radius <= plan.real_cutoff)
    expected = jnp.sum(
        jnp.where(
            valid[..., None],
            masses[None, :, None, None]
            * displacement
            * (plan._screening(radius) / radius**3)[..., None],
            0.0,
        ),
        axis=(1, 2),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.evidence.real_space_acceleration, expected)


def test_snapshot_and_distributed_feasibility_contracts():
    artifact = cosmology.ScientificArtifactEnvelope(
        artifact_kind="snapshot",
        content_digest="snapshot-fixture",
        producer="test",
        producer_version="current",
        build_id="fixture",
        license_id="internal",
        resource_id="static",
        status="complete",
    )
    snapshot = cosmology.CosmologySnapshotProduct(
        [1, 2],
        [[0.1, 0.2, 0.3], [0.5, 0.6, 0.7]],
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
        [1.0, 1.0],
        0.5,
        (1.0, 1.0, 1.0),
        artifact,
    )
    assert snapshot.snapshot_id
    feasible = cosmology.DistributedPMFeasibilityEvidence(
        (64, 64, 64),
        (2, 2, 1),
        100_000,
        byte_budget_per_device=1_000_000_000,
    )
    assert feasible.divisible
    assert feasible.feasible
