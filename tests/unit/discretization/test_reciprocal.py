import numpy as np
import pytest

from phydrax.discretization import (
    PeriodicCell,
    ReciprocalConnectivityPlan,
    ReciprocalMeshPlan,
    ReciprocalPathPlan,
    ReciprocalResourceError,
)


@pytest.mark.parametrize("rank,shape", [(1, (5,)), (2, (3, 4)), (3, (2, 3, 4))])
def test_rank_aware_mesh_has_unique_points_and_normalized_weights(rank, shape):
    cell = PeriodicCell(np.eye(rank))
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, shape)

    assert mesh.fractional_points.shape == (int(np.prod(shape)), rank)
    np.testing.assert_allclose(np.sum(mesh.weights), 1.0, atol=1.0e-14)
    wrapped = np.asarray(mesh.fractional_points) % 1.0
    assert np.unique(np.round(wrapped, 14), axis=0).shape[0] == wrapped.shape[0]


def test_regular_connectivity_closes_oriented_wrapped_plaquettes():
    cell = PeriodicCell([[2.0, 0.0], [0.5, 1.5]])
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (3, 4))
    prepared = ReciprocalConnectivityPlan.regular(mesh).prepare()
    plan = prepared.plan

    displacements = np.asarray(prepared.fractional_displacements)
    for edges, signs in zip(
        plan.plaquette_edges, plan.plaquette_orientations, strict=True
    ):
        np.testing.assert_allclose(
            np.sum(displacements[np.asarray(edges)] * np.asarray(signs)[:, None], axis=0),
            0.0,
            atol=1.0e-14,
        )
    reverse = np.asarray(plan.reverse_indices)
    np.testing.assert_array_equal(reverse[reverse], np.arange(reverse.size))
    np.testing.assert_allclose(displacements[reverse], -displacements)


def test_path_and_mesh_refuse_cell_identity_mismatch():
    cell = PeriodicCell([[1.0]])
    other = PeriodicCell([[2.0]])
    mesh = ReciprocalMeshPlan.monkhorst_pack(cell, (4,))
    path = ReciprocalPathPlan(cell, [[0.0], [0.5]])

    with pytest.raises(ValueError, match="different PeriodicCell"):
        mesh.require_cell(other)
    with pytest.raises(ValueError, match="different PeriodicCell"):
        path.require_cell(other)


def test_reciprocal_capacities_refuse_before_grid_allocation():
    cell = PeriodicCell(np.eye(3))
    with pytest.raises(ReciprocalResourceError, match="maximum_points"):
        ReciprocalMeshPlan.monkhorst_pack(cell, (20, 20, 20), maximum_points=100)
