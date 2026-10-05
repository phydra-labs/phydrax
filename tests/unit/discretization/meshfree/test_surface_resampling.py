# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization.meshfree._capacity import MeshfreeCapacityPolicy
from phydrax.discretization.meshfree._resampling import SurfaceResamplingPolicy


def _quality(points: Array) -> tuple[Array, Array]:
    return jnp.ones(points.shape[0]), jnp.ones(points.shape[0])


def test_deterministic_insertion_and_removal_project_without_fake_convergence() -> None:
    points = jnp.array([[0.0, 0.0], [0.01, 0.0], [1.0, 0.0]])
    probes = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    policy = SurfaceResamplingPolicy(maximum_fill=0.3, minimum_separation=0.1)
    result = policy.repair(
        points,
        probes,
        MeshfreeCapacityPolicy((3, 4)),
        lambda p: p.at[:, 1].set(0),
        lambda p: p[:, 1],
        _quality,
    )
    assert result.converged
    assert result.removed_sources == (1,) and result.inserted_probes == (1,)
    # Sample lineage: survivors name their input index, the probe is new.
    np.testing.assert_array_equal(result.source_indices, [0, 2, -1])
    np.testing.assert_allclose(
        np.sort(np.asarray(result.points)[:, 0]), [0.0, 0.5, 1.0], atol=1e-12
    )
    assert not result.after.fill_is_certified_global


def test_resampling_reports_capacity_refusal_and_unresolved_stencil_quality() -> None:
    points, probes = jnp.array([[0.0, 0.0], [1.0, 0.0]]), jnp.array([[0.5, 0.0]])
    policy = SurfaceResamplingPolicy(
        maximum_fill=0.1, minimum_separation=0.01, maximum_iterations=2
    )
    full = policy.repair(
        points,
        probes,
        MeshfreeCapacityPolicy((2,)),
        lambda p: p,
        lambda p: p[:, 1],
        _quality,
    )
    assert full.capacity_refused and not full.converged
    bad = policy.repair(
        points,
        points,
        MeshfreeCapacityPolicy((2,)),
        lambda p: p,
        lambda p: p[:, 1],
        lambda p: (jnp.full(p.shape[0], 1e12), jnp.ones(p.shape[0])),
    )
    assert not bad.converged and bad.after.maximum_condition == 1e12


def test_coincident_point_removal_keeps_lower_index_without_admitting_bad_fit() -> None:
    points = jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    probes = jnp.array([[0.0, 0.0], [1.0, 0.0]])
    result = SurfaceResamplingPolicy(maximum_fill=0.1, minimum_separation=0.1).repair(
        points,
        probes,
        MeshfreeCapacityPolicy((3,)),
        lambda p: p,
        lambda p: p[:, 1],
        _quality,
    )
    assert result.converged and result.removed_sources == (1,)
    assert result.before.separation == 0
    np.testing.assert_allclose(result.points, [[0.0, 0.0], [1.0, 0.0]], atol=0)


def test_repair_proposals_are_deterministic_and_identified() -> None:
    points = jnp.array([[0.0, 0.0], [0.01, 0.0], [1.0, 0.0]])
    probes = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    policy = SurfaceResamplingPolicy(maximum_fill=0.3, minimum_separation=0.1)

    def repair() -> tuple[str, np.ndarray]:
        result = policy.repair(
            points,
            probes,
            MeshfreeCapacityPolicy((3, 4)),
            lambda p: p.at[:, 1].set(0),
            lambda p: p[:, 1],
            _quality,
        )
        return result.proposal_id, np.asarray(result.points)

    first, second = repair(), repair()
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
