"""Public effective resistance result contracts."""

from jax import Array
from typing_extensions import assert_type

from phydrax.graph import (
    EffectiveResistancePlan,
    EffectiveResistanceResult,
    SketchedEffectiveResistancePlan,
    SketchedResistanceEmbedding,
)
from phydrax.sparse import EdgeRelation
from phydrax.typing import PRNGKey


def resistance_types(relation: EdgeRelation, nodes: Array, conductances: Array) -> None:
    plan = EffectiveResistancePlan(relation, nodes)
    result = plan.solve(conductances)
    assert_type(result, EffectiveResistanceResult)
    assert_type(result.resistance, Array)
    assert_type(result.connected, Array)
    assert_type(result.status, Array)
    assert_type(result.solve_status, Array)
    assert_type(result.residual_norm, Array)
    assert_type(result.iterations, Array)
    assert_type(result.derivative_regular, Array)
    EffectiveResistancePlan("wrong-kind", nodes)  # ty: ignore[invalid-argument-type]


def sketch_types(
    relation: EdgeRelation, conductances: Array, key: PRNGKey, nodes: Array
) -> None:
    plan = SketchedEffectiveResistancePlan(relation, sketch_dimension=8)
    result = plan.embed(conductances, key)
    assert_type(result, SketchedResistanceEmbedding)
    assert_type(result.embedding, Array)
    assert_type(result.component_label, Array)
    assert_type(result.status, Array)
    assert_type(result.solve_status, Array)
    assert_type(result.residual_norm, Array)
    assert_type(result.iterations, Array)
    assert_type(result.derivative_regular, Array)
    assert_type(result.sketch_dimension, int)
    assert_type(result.resistance(nodes, nodes), Array)
