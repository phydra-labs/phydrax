"""Public directed shortest-path field contracts."""

from typing import Literal

from jax import Array
from typing_extensions import assert_type

from phydrax.applications.geophysics import TravelTimeGraphPlan, TravelTimeResult
from phydrax.combinatorial import (
    ShortestPathFieldPlan,
    ShortestPathFieldResult,
    ShortestPathFieldStatus,
)
from phydrax.sparse import EdgeRelation
from phydrax.typing import Bool, Float64, Int32


def fields(relation: EdgeRelation, costs: Array, sources: Array) -> None:
    plan = ShortestPathFieldPlan(relation, method="host-dijkstra")
    assert_type(plan, ShortestPathFieldPlan)
    result = plan.solve(costs, sources)
    assert_type(result, ShortestPathFieldResult)
    assert_type(result.distance, Float64)
    assert_type(result.reachable, Bool)
    assert_type(result.predecessor_edge, Int32)
    assert_type(result.plan_id, str)
    assert_type(plan.maximum_host_bytes, int)
    assert_type(plan.projected_host_bytes, int)
    assert_type(plan.host_projected_work, int)
    assert_type(ShortestPathFieldStatus.SUCCESS, Literal[ShortestPathFieldStatus.SUCCESS])
    ShortestPathFieldPlan(relation, method="dijkstra")  # ty: ignore[invalid-argument-type]


def travel_field_evidence(plan: TravelTimeGraphPlan, speed: Array) -> None:
    result = plan.solve(speed, 0)
    assert_type(result, TravelTimeResult)
    assert_type(result.field_status, Array)
    assert_type(result.bellman_residual, Array)
    assert_type(result.dual_residual, Array)
    assert_type(result.reachable, Array)
