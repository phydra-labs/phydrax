from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellMesh, PeriodicCell, PeriodicMeshTopology
from phydrax.lifecycle._meshing_source_families import rebuild_native_family_value
from phydrax.meshing._curving import _straight_geometry
from phydrax.meshing._periodic import certify_periodic_embedding


def _strip() -> CellMesh:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    block = CellBlock(
        "root", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
    )
    plain = CellMesh(points, (block,))
    topology = PeriodicMeshTopology(
        plain,
        PeriodicCell(np.eye(2, dtype=np.float64), periodic_axes=(True, False)),
        np.asarray((0, 0, 3, 3)),
        np.asarray(((0, 0), (1, 0), (1, 0), (0, 0))),
    )
    return CellMesh(points, (block,), periodic_topology=topology)


@pytest.mark.parametrize("degree", (3, 4))
def test_quotient_source_keeps_exact_group_trace_and_dynamic_representatives(
    degree: int,
) -> None:
    mesh = _strip()
    straight = _straight_geometry(mesh, degree)
    coordinates = np.asarray(straight.coordinates).copy()
    # A non-dyadic translation image requires carrier rounding even though the
    # declared scientific trace is exactly a translate of its representative.
    coordinates[:, 0] += 0.02 * coordinates[:, 1] * (1.0 - coordinates[:, 1])
    curved = straight.with_coordinates(coordinates)
    source = curved.periodic_source
    assert source is not None
    exact = curved.source_coordinates()
    for node in source.active_indices:
        root = source.representative_nodes[source.node_orbits[node]]
        exponent = source.node_exponents[node]
        assert exact[node][0] - exact[root][0] == Fraction(exponent[0])
        assert exact[node][1] == exact[root][1]
    assert curved.source_execution_error(mesh) > 0.0
    evidence = certify_periodic_embedding(mesh, geometry=curved)
    assert (
        evidence.global_embedding is not None
        and evidence.global_embedding.status == "certified"
    )
    assert evidence.maximum_coordinate_residual > 0.0
    renewed = rebuild_native_family_value(source)
    assert renewed is not None and renewed.source_id == source.source_id
    indices = np.asarray(tuple(reversed(range(len(exact)))), dtype=np.int64)
    assert source.reindexed(indices).source_coordinates() == tuple(
        exact[int(index)] for index in indices
    )
    derivative = jax.jacrev(
        lambda values: eqx.tree_at(
            lambda item: item.representative_coordinates, source, values
        ).runtime_coordinates()
    )(source.representative_coordinates)
    root = source.representative_nodes[0]
    orbit_node = next(
        node
        for node in source.active_indices
        if source.node_orbits[node] == 0 and node != root
    )
    np.testing.assert_array_equal(
        np.asarray(derivative[orbit_node, :, 0, :]), np.eye(2, dtype=np.float64)
    )
    assert jnp.all(jnp.isfinite(derivative))
