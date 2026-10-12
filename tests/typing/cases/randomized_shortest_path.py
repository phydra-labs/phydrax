"""Public randomized-shortest-path result contracts."""

from jax import Array
from typing_extensions import assert_type

from phydrax.graph import RandomizedShortestPathPlan, RandomizedShortestPathResult
from phydrax.sparse import EdgeRelation


def randomized_shortest_path_types(
    relation: EdgeRelation, nodes: Array, affinities: Array, costs: Array, theta: Array
) -> None:
    plan = RandomizedShortestPathPlan(relation, nodes)
    result = plan.solve(affinities, costs, theta)
    assert_type(result, RandomizedShortestPathResult)
    assert_type(result.free_energy, Array)
    assert_type(result.expected_cost, Array)
    assert_type(result.reachable, Array)
    assert_type(result.status, Array)
    assert_type(result.nonlinear_status, Array)
    assert_type(result.newton_iterations, Array)
    assert_type(result.residual_norm, Array)
    assert_type(result.linear_status, Array)
    assert_type(result.free_energy_distance, Array)
    assert_type(result.expected_cost_dissimilarity, Array)
    assert_type(plan.maximum_derivative_order, int)
    assert_type(result.maximum_derivative_order, int)
    assert_type(plan.resource_estimate, dict[str, int])
    RandomizedShortestPathPlan("wrong-kind", nodes)  # ty: ignore[invalid-argument-type]
