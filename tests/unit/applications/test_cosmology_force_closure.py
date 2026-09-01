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
