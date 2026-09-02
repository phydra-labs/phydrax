#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


track = phx.linalg.eigen


def test_eigenspace_tracking_recovers_permutation_and_complex_phase():
    reference_values = jnp.asarray([1.0, 2.0, 4.0])
    reference_vectors = jnp.eye(3, dtype=jnp.complex128)
    permutation = jnp.asarray([2, 0, 1])
    phases = jnp.exp(1j * jnp.asarray([0.3, -0.7, 1.2]))
    candidate_vectors = reference_vectors[:, permutation] * phases[None, :]
    candidate_values = reference_values[permutation]
    plan = track.plan_hermitian_eigenspace_tracking(reference_values)

    result = track.track_hermitian_eigenspaces(
        plan,
        reference_vectors,
        candidate_values,
        candidate_vectors,
    )

    assert bool(result.successful)
    assert jnp.array_equal(result.assignment, jnp.asarray([1, 2, 0]))
    assert jnp.allclose(result.values, reference_values)
    assert jnp.allclose(result.vectors, reference_vectors)
    assert float(result.diagnostics.assignment_margin) > 0.99


def test_eigenspace_tracking_aligns_a_degenerate_rotated_subspace():
    reference_values = jnp.asarray([0.0, 0.0, 3.0])
    reference_vectors = jnp.eye(3, dtype=jnp.complex128)
    angle = jnp.asarray(0.37)
    rotation = jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ],
        dtype=jnp.complex128,
    )
    candidate_vectors = reference_vectors.at[:2, :2].set(rotation)
    plan = track.plan_hermitian_eigenspace_tracking(reference_values)

    result = jax.jit(track.track_hermitian_eigenspaces)(
        plan,
        reference_vectors,
        reference_values,
        candidate_vectors,
    )

    assert plan.clusters == ((0, 1), (2,))
    assert bool(result.successful)
    assert jnp.allclose(result.vectors, reference_vectors, atol=1e-12)
    assert jnp.all(result.diagnostics.cluster_minimum_overlaps > 1.0 - 1e-12)


def test_eigenspace_tracking_reports_ambiguous_nondegenerate_matching():
    reference_values = jnp.asarray([0.0, 1.0])
    reference_vectors = jnp.eye(2)
    candidate_vectors = jnp.asarray(
        [[1.0, -1.0], [1.0, 1.0]], dtype=jnp.float64
    ) / jnp.sqrt(2.0)
    plan = track.plan_hermitian_eigenspace_tracking(reference_values)

    result = track.track_hermitian_eigenspaces(
        plan,
        reference_vectors,
        reference_values,
        candidate_vectors,
    )

    assert not bool(result.successful)
    assert int(result.status) == int(track.HermitianEigenspaceTrackingStatus.AMBIGUOUS)
    assert float(result.diagnostics.assignment_margin) == pytest.approx(0.0)


def test_eigenspace_tracking_rejects_cluster_change_and_bad_shapes():
    reference_values = jnp.asarray([0.0, 0.0, 2.0])
    reference_vectors = jnp.eye(3)
    plan = track.plan_hermitian_eigenspace_tracking(reference_values)
    split_values = jnp.asarray([0.0, 0.5, 2.0])
    result = track.track_hermitian_eigenspaces(
        plan,
        reference_vectors,
        split_values,
        reference_vectors,
    )

    assert not bool(result.successful)
    assert int(result.status) == int(
        track.HermitianEigenspaceTrackingStatus.CLUSTER_MISMATCH
    )
    with pytest.raises(ValueError, match="column count"):
        track.track_hermitian_eigenspaces(
            plan,
            reference_vectors[:, :2],
            split_values,
            reference_vectors,
        )
    with pytest.raises(ValueError, match="nonempty"):
        track.plan_hermitian_eigenspace_tracking(jnp.asarray([]))
    with pytest.raises(ValueError, match="finite"):
        track.plan_hermitian_eigenspace_tracking(jnp.asarray([0.0, np.nan]))
