"""Prepared spatial families and explicit graph query plans remain typed."""

from jax import Array
from typing_extensions import assert_type

from phydrax.discretization.spatial import (
    BVHNeighborQueryPlan,
    BVHNeighborQueryResult,
    BVHSourcePlan,
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonNeighborQueryResult,
    MortonSourcePlan,
    PreparedBVHSources,
    PreparedMortonSources,
)
from phydrax.graph import (
    batched_knn_graph,
    batched_knn_query_graph,
    GraphIR,
    query_neighbors,
    QueryGraph,
    QueryNeighborhood,
)


def spatial_queries(points: Array, targets: Array, address: MortonAddressPlan) -> None:
    morton_source = MortonSourcePlan(address, 8)
    bvh_source = BVHSourcePlan(3, 8)
    assert_type(morton_source.prepare(points), PreparedMortonSources)
    assert_type(bvh_source.prepare(points), PreparedBVHSources)
    morton = MortonNeighborQueryPlan(morton_source, 8, 2)
    bvh = BVHNeighborQueryPlan(bvh_source, 8, 2)
    prepared_morton = morton_source.prepare(points)
    prepared_bvh = bvh_source.prepare(points)
    assert_type(morton.query(prepared_morton, targets), MortonNeighborQueryResult)
    assert_type(bvh.query(prepared_bvh, targets), BVHNeighborQueryResult)
    assert_type(query_neighbors(prepared_morton, targets, plan=morton), QueryNeighborhood)
    assert_type(query_neighbors(prepared_bvh, targets, plan=bvh), QueryNeighborhood)
    assert_type(query_neighbors(points, targets, plan=bvh), QueryNeighborhood)
    assert_type(batched_knn_query_graph(prepared_bvh, targets, k=2, plan=bvh), QueryGraph)
    assert_type(batched_knn_graph(prepared_morton, k=2, plan=morton), GraphIR)
    morton.query(prepared_bvh, targets)  # ty: ignore[invalid-argument-type]
    bvh.query(prepared_morton, targets)  # ty: ignore[invalid-argument-type]
