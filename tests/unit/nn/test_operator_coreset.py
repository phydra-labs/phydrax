# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import pytest

import phydrax as phx


def _dataset():
    coordinates = jnp.linspace(0.0, 1.0, 4)[:, None]
    source = phx.nn.operator.FunctionSamples(
        values=jnp.arange(12.0).reshape((3, 4, 1)),
        coordinates=coordinates,
    )
    query = phx.nn.operator.FunctionSamples(values=None, coordinates=coordinates)
    batch = phx.nn.operator.OperatorBatch(
        inputs={"source": source},
        queries={"query": query},
        case_axes=("case",),
        case_shape=(3,),
    )
    targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
        {"field": jnp.arange(12.0).reshape((3, 4, 1))},
        batch,
        query_names={"field": "query"},
    )
    return phx.nn.operator.training.OperatorDataset(batch, targets)


def test_atomic_operator_case_coreset_carries_weights_and_provenance():
    dataset = _dataset()
    result = phx.nn.operator.training.compress_operator_cases(
        dataset,
        phx.coresets.MomentRecombination(),
        features=jnp.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]]),
    )
    assert result.dataset.size == result.selection.capacity
    assert jnp.array_equal(result.dataset.case_mask, result.selection.mask)
    source = phx.nn.operator.InMemoryOperatorCaseSource(result.dataset)
    for index in range(result.dataset.size):
        case = source.read_case(index)
        assert case.case_log_weight == float(result.dataset.case_log_weights[index])
        assert case.case_active == bool(result.selection.mask[index])
    assert (
        tuple(record.case_id for record in result.dataset.provenance)
        == result.source_case_ids
    )


def test_named_query_coreset_aligns_geometry_targets_and_mass():
    dataset = _dataset()
    result = phx.nn.operator.training.compress_operator_queries(
        dataset,
        "query",
        phx.coresets.MomentRecombination(),
        features=jnp.stack((jnp.ones((4,)), jnp.linspace(0.0, 1.0, 4)), axis=1),
    )
    query = result.dataset.batch.query("query")
    target = result.dataset.targets.field("field")
    assert target.values.shape[1] == query.sample_shape[0]
    assert jnp.allclose(jnp.sum(query.quadrature_weights), result.source_physical_mass[0])


def test_custom_weighted_losses_require_explicit_per_case_protocol():
    term = phx.nn.operator.training.OperatorLossTerm(
        "per-case",
        lambda *args, **kwargs: jnp.ones((3,)),
        case_reduction="per_case",
    )
    assert term.case_reduction == "per_case"
    with pytest.raises(ValueError, match="case_reduction"):
        phx.nn.operator.training.OperatorLossTerm(
            "invalid",
            lambda *args, **kwargs: 0.0,
            case_reduction="automatic",
        )
