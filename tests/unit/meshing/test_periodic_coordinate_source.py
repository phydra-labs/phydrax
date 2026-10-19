from contextlib import AbstractContextManager
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    CellBlock,
    CellGeometrySpec,
    CellMesh,
    PeriodicCell,
    PeriodicMeshTopology,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    coordinate_bernstein_estimate,
    coordinate_estimate_bank,
    coordinate_map_expression_estimates,
    coordinate_preparation_metadata_estimate,
)
from phydrax.discretization._periodic_exact import owned_periodic_preparation
from phydrax.discretization._periodic_topology import periodic_preparation_bounds
from phydrax.lifecycle._meshing_sources import reconstruct_meshing_source
from phydrax.meshing._curving import straight_geometry
from phydrax.meshing._periodic import certify_periodic_embedding


def _source_budget(
    mesh: CellMesh, geometry: CellGeometrySpec, /
) -> AbstractContextManager[CoordinateEnclosureBudget]:
    periodic = mesh.periodic_topology
    source = geometry.periodic_source
    if periodic is None or source is None:
        raise ValueError(
            "geometry.periodic_source must accompany mesh.periodic_topology; got a missing source or topology."
        )
    work, storage, queries = periodic_preparation_bounds(
        periodic.cell,
        geometry.coordinates.shape[0],
        np.asarray(periodic.orbit_shifts, dtype=np.int64),
    )
    matrix_bits = max(
        max(abs(numerator).bit_length(), denominator.bit_length())
        for matrix in source.exact_isometries
        for row in matrix
        for numerator, denominator in row
    )
    binary64_bits = (
        float(np.finfo(np.float64).smallest_subnormal).as_integer_ratio()[1].bit_length()
    )
    error_bits = (
        2
        * (
            (mesh.ambient_dimension + 1) * (binary64_bits + matrix_bits)
            + mesh.ambient_dimension.bit_length()
        )
        + 1
    )
    estimates = []
    for block, element, route in zip(
        mesh.blocks, geometry.elements, geometry.geometry_dofs, strict=True
    ):
        domain = (
            "simplex"
            if block.cell_kind in ("interval", "triangle", "tetrahedron")
            else "prism"
            if block.cell_kind == "prism"
            else "box"
        )
        for row in np.asarray(route):
            local = np.asarray(geometry.coordinates)[row[row >= 0]]
            expressions = coordinate_map_expression_estimates(
                element, local, coefficient_bits_upper=error_bits
            )
            if expressions is None:
                raise ValueError(
                    "geometry.elements must have a supported coordinate recipe; got an unsupported recipe."
                )
            estimates.append(
                coordinate_preparation_metadata_estimate(
                    element, local, coefficient_bits_upper=error_bits
                )
            )
            estimates.extend(
                coordinate_bernstein_estimate(value, domain, mesh.topological_dimension)
                for value in expressions
            )
    estimate = coordinate_estimate_bank(tuple(estimates))
    return owned_periodic_preparation(
        2 * work + estimate.work_upper,
        2 * storage + estimate.bytes_upper,
        queries_upper=2 * queries,
    )


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
    straight = straight_geometry(mesh, degree)
    coordinates = np.asarray(straight.coordinates).copy()
    # A non-dyadic translation image requires carrier rounding even though the
    # declared scientific trace is exactly a translate of its representative.
    coordinates[:, 0] += 0.02 * coordinates[:, 1] * (1.0 - coordinates[:, 1])
    curved = straight.with_coordinates(coordinates)
    source = curved.periodic_source
    assert source is not None
    with _source_budget(mesh, curved) as budget:
        exact = curved.source_coordinates(budget=budget)
        for node in source.active_indices:
            root = source.representative_nodes[source.node_orbits[node]]
            exponent = source.node_exponents[node]
            assert exact[node][0] - exact[root][0] == Fraction(exponent[0])
            assert exact[node][1] == exact[root][1]
        assert curved.source_execution_error(mesh, budget=budget) > 0.0
        evidence = certify_periodic_embedding(mesh, geometry=curved)
        assert (
            evidence.global_embedding is not None
            and evidence.global_embedding.status == "certified"
        )
        assert evidence.maximum_coordinate_residual > 0.0
        renewed = reconstruct_meshing_source(source)
        assert renewed is not None and renewed.source_id == source.source_id
        indices = np.asarray(tuple(reversed(range(len(exact)))), dtype=np.int64)
        assert source.reindexed(indices, budget=budget).source_coordinates(
            budget=budget
        ) == tuple(exact[int(index)] for index in indices)
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
