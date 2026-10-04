#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Behavior of prepared bounded streamed relations against independent references.

The reference evaluates every admitted event exactly once with plain JAX over
host-selected routes, sums messages per receiver and applies the receiver
epilogue to complete aggregates. It shares no tiling, schedule, grouping or
reducer code with the streamed implementation.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.tree_util import Partial

import phydrax as phx


SOURCE_COUNT = 6
RECEIVER_COUNT = 7
# Receiver 2 has degree ten with duplicate (source, receiver) routes, receiver 0
# a duplicate pair, receiver 5 no routes; the last three routes are invalid and
# carry out-of-range padding indices.
_ROUTES = (
    [(source, 2) for source in (0, 1, 2, 3, 4, 5, 0, 1, 2, 3)]
    + [(1, 0), (1, 0), (5, 1), (2, 3), (4, 3), (0, 3), (3, 4), (0, 6), (5, 6)]
    + [(-1, 3), (99, 0), (2, 77)]
)
_VALID = np.asarray([True] * 19 + [False] * 3)
_STORAGE = np.random.default_rng(11).permutation(len(_ROUTES))


def _relation(
    storage: np.ndarray = _STORAGE,
) -> tuple[phx.sparse.EdgeRelation, np.ndarray, np.ndarray, np.ndarray]:
    routes = np.asarray(_ROUTES, dtype=np.int32)[storage]
    valid = _VALID[storage]
    relation = phx.sparse.EdgeRelation(
        routes[:, 0],
        routes[:, 1],
        source_size=SOURCE_COUNT,
        target_size=RECEIVER_COUNT,
        valid=valid,
    )
    return relation, routes[:, 0], routes[:, 1], valid


def _data() -> tuple[dict[str, Array], dict[str, Array], dict[str, Array], Array]:
    rng = np.random.default_rng(3)
    parameters = {
        "w": jnp.asarray(0.8),
        "u": jnp.asarray(rng.normal(size=(2, 2))),
    }
    sources = {
        "x": jnp.asarray(rng.normal(size=(SOURCE_COUNT, 3))),
        "h": jnp.asarray(rng.normal(size=(SOURCE_COUNT, 2))),
    }
    receivers = {
        "x": jnp.asarray(rng.normal(size=(RECEIVER_COUNT, 3))),
        "h": jnp.asarray(rng.normal(size=(RECEIVER_COUNT, 2))),
    }
    edges = jnp.asarray(rng.uniform(0.5, 1.5, size=(len(_ROUTES),)))
    return parameters, sources, receivers, edges


def _radial(
    parameters: dict[str, Array],
    source: dict[str, Array],
    receiver: dict[str, Array],
    edge: Array,
) -> tuple[Array, Array]:
    displacement = receiver["x"] - source["x"]
    # log is singular at coincident endpoints: only admitted events may reach it.
    distance = jnp.sqrt(jnp.sum(displacement * displacement))
    return jnp.tanh(parameters["w"] * jnp.log(distance)) * edge, displacement


def _message(
    parameters: dict[str, Array],
    source: dict[str, Array],
    receiver: dict[str, Array],
    edge: Array,
) -> dict[str, Array]:
    radial, displacement = _radial(parameters, source, receiver, edge)
    return {
        "s": radial * source["h"],
        "v": radial * displacement[:, None] * source["h"][None, :],
    }


def _message_with_edge_output(
    parameters: dict[str, Array],
    source: dict[str, Array],
    receiver: dict[str, Array],
    edge: Array,
) -> tuple[dict[str, Array], Array]:
    message = _message(parameters, source, receiver, edge)
    return message, jnp.sin(message["s"])


def _epilogue(
    parameters: dict[str, Array], receiver: dict[str, Array], aggregate: dict[str, Array]
) -> dict[str, Array]:
    return {
        "h": jnp.tanh(aggregate["s"] @ parameters["u"] + receiver["h"]),
        "n": jnp.sum(aggregate["v"] ** 2),
    }


_MESSAGE = {
    "s": jax.ShapeDtypeStruct((2,), jnp.float64),
    "v": jax.ShapeDtypeStruct((3, 2), jnp.float64),
}
_OUTPUT = {
    "h": jax.ShapeDtypeStruct((2,), jnp.float64),
    "n": jax.ShapeDtypeStruct((), jnp.float64),
}
_EDGE_OUTPUT = jax.ShapeDtypeStruct((2,), jnp.float64)


def _reference(
    parameters: dict[str, Array],
    sources: dict[str, Array],
    receivers: dict[str, Array],
    edges: Array,
    *,
    source_index: np.ndarray,
    target_index: np.ndarray,
    admitted: np.ndarray,
    receiver_valid: np.ndarray | None = None,
) -> tuple[dict[str, Array], Array]:
    routes = np.flatnonzero(admitted)
    source = jax.tree.map(lambda value: value[source_index[routes]], sources)
    receiver = jax.tree.map(lambda value: value[target_index[routes]], receivers)
    messages, edge_values = jax.vmap(
        lambda s, r, e: _message_with_edge_output(parameters, s, r, e)
    )(source, receiver, edges[routes])
    aggregate = jax.tree.map(
        lambda value: (
            jnp.zeros((RECEIVER_COUNT,) + value.shape[1:], value.dtype)
            .at[target_index[routes]]
            .add(value)
        ),
        messages,
    )
    outputs = jax.vmap(lambda r, a: _epilogue(parameters, r, a))(receivers, aggregate)
    if receiver_valid is not None:
        outputs = jax.tree.map(
            lambda value: jnp.where(
                receiver_valid.reshape((-1,) + (1,) * (value.ndim - 1)), value, 0.0
            ),
            outputs,
        )
    edge_outputs = jnp.zeros((len(admitted), 2)).at[routes].set(edge_values)
    return outputs, edge_outputs


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize(
    ("receiver_tile", "edge_tile"),
    [(1, 1), (2, 3), (3, 4), (8, 32)],
    ids=["single-event-fragments", "split-high-degree", "mixed-fill", "one-tile"],
)
def test_irregular_relation_matches_independent_reference(
    accumulation: phx.sparse.RelationAccumulation, receiver_tile: int, edge_tile: int
) -> None:
    relation, source_index, target_index, valid = _relation()
    parameters, sources, receivers, edges = _data()
    plan = phx.sparse.StreamedRelationPlan(
        receiver_tile=receiver_tile,
        edge_tile=edge_tile,
        channel_capacity=16,
        accumulation=accumulation,
    )
    prepared = plan.prepare(relation, owner_id="test-irregular")
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT, _EDGE_OUTPUT)

    result = prepared.evaluate(
        payload,
        _message_with_edge_output,
        _epilogue,
        parameters,
        sources,
        receivers,
        edges,
    )
    expected, expected_edges = _reference(
        parameters,
        sources,
        receivers,
        edges,
        source_index=source_index,
        target_index=target_index,
        admitted=valid,
    )

    np.testing.assert_allclose(result.receiver_outputs["h"], expected["h"], atol=1e-13)
    np.testing.assert_allclose(result.receiver_outputs["n"], expected["n"], rtol=1e-12)
    assert result.edge_outputs is not None
    np.testing.assert_allclose(result.edge_outputs, expected_edges, atol=1e-14)
    evidence = result.evidence
    assert bool(evidence.successful)
    assert int(evidence.active_routes) == 19
    assert int(evidence.evaluated_routes) == 19
    assert int(evidence.committed_receivers) == RECEIVER_COUNT
    assert int(evidence.maximum_receiver_degree) == 10
    # A degree-ten receiver cannot fit a fragment of fewer than ten events.
    assert (int(evidence.fragmented_receivers) > 0) == (edge_tile < 10)


def _fragment_aggregate(
    parameters: dict[str, Array],
    sources: dict[str, Array],
    receivers: dict[str, Array],
    edges: Array,
    fragment: phx.sparse.StreamedFragment,
    seed: dict[str, Any],
) -> tuple[dict[str, Any], Array]:
    # Lane rows reach the per-event law only through the fragment routing:
    # distinct fragment sources and receiver slots.
    local_sources = jax.tree.map(lambda value: value[fragment.source_ids], sources)
    source = jax.tree.map(lambda value: value[fragment.lane_local_sources], local_sources)
    receiver = jax.tree.map(lambda value: value[fragment.lane_slots], receivers)
    messages, edge_values = jax.vmap(
        lambda s, r, e: _message_with_edge_output(parameters, s, r, e)
    )(source, receiver, edges)
    return {
        name: fragment.reduce(messages[name], seed[name]) for name in messages
    }, edge_values


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize(
    ("receiver_tile", "edge_tile"),
    [(1, 1), (2, 3), (8, 32)],
    ids=["single-event-fragments", "split-high-degree", "one-tile"],
)
def test_fragment_aggregation_matches_reference_and_event_streaming(
    accumulation: phx.sparse.RelationAccumulation, receiver_tile: int, edge_tile: int
) -> None:
    relation, source_index, target_index, valid = _relation()
    parameters, sources, receivers, edges = _data()
    active = jnp.asarray(valid & (np.arange(len(_ROUTES)) % 5 != 1))
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=receiver_tile,
        edge_tile=edge_tile,
        channel_capacity=16,
        accumulation=accumulation,
    ).prepare(relation, owner_id="test-fragments")
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT, _EDGE_OUTPUT)

    def run(
        theta: dict[str, Array], fragments: bool
    ) -> phx.sparse.StreamedRelationResult:
        if fragments:
            return prepared.evaluate_fragments(
                payload,
                _fragment_aggregate,
                _epilogue,
                theta,
                sources,
                receivers,
                edges,
                edge_active=active,
            )
        return prepared.evaluate(
            payload,
            _message_with_edge_output,
            _epilogue,
            theta,
            sources,
            receivers,
            edges,
            edge_active=active,
        )

    result = run(parameters, True)
    expected, expected_edges = _reference(
        parameters,
        sources,
        receivers,
        edges,
        source_index=source_index,
        target_index=target_index,
        admitted=np.asarray(active),
    )
    np.testing.assert_allclose(result.receiver_outputs["h"], expected["h"], atol=1e-13)
    np.testing.assert_allclose(result.receiver_outputs["n"], expected["n"], rtol=1e-12)
    assert result.edge_outputs is not None
    np.testing.assert_allclose(result.edge_outputs, expected_edges, atol=1e-14)
    assert bool(result.evidence.successful)
    assert int(result.evidence.evaluated_routes) == int(np.sum(active))

    def loss(theta: dict[str, Array], fragments: bool) -> Array:
        outputs = run(theta, fragments)
        return jnp.sum(outputs.receiver_outputs["h"] ** 2) + jnp.sum(
            outputs.edge_outputs
            * outputs.receiver_outputs["n"][target_index % RECEIVER_COUNT, None]
        )

    event_gradient = jax.grad(loss)(parameters, False)
    fragment_gradient = jax.grad(loss)(parameters, True)
    for name in parameters:
        np.testing.assert_allclose(
            fragment_gradient[name], event_gradient[name], rtol=1e-12, atol=1e-13
        )


def test_prepared_fragment_routing_partitions_lanes_by_slot_and_source() -> None:
    relation, _, _, valid = _relation()
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3).prepare(
        relation, owner_id="test-fragment-routing"
    )
    fragments = jax.device_get(prepared.fragments())
    tiles = prepared.schedule.tile_count
    lane_valid = np.asarray(jax.device_get(prepared.schedule.lane_valid))
    covered = 0
    for tile in range(tiles):
        slots = np.asarray(fragments.lane_slots[tile])
        sources = np.asarray(fragments.lane_sources[tile])
        for slot in range(2):
            start = int(fragments.receiver_lane_offsets[tile, slot])
            count = int(fragments.receiver_lane_counts[tile, slot])
            assert np.all(lane_valid[tile, start : start + count])
            assert np.all(slots[start : start + count] == slot)
            assert count == int(np.sum(lane_valid[tile] & (slots == slot)))
        ids = np.asarray(fragments.source_ids[tile])[
            np.asarray(fragments.source_valid[tile])
        ]
        assert np.all(np.diff(ids) > 0)
        for local, source in enumerate(ids):
            start = int(fragments.source_lane_offsets[tile, local])
            count = int(fragments.source_lane_counts[tile, local])
            lanes = np.asarray(fragments.source_lanes[tile, start : start + count])
            assert np.all(lane_valid[tile, lanes]) and np.all(sources[lanes] == source)
            assert np.all(np.asarray(fragments.lane_local_sources[tile, lanes]) == local)
            assert count == int(np.sum(lane_valid[tile] & (sources == source)))
            covered += count
        invalid = ~np.asarray(fragments.source_valid[tile])
        assert np.all(np.asarray(fragments.source_lane_counts[tile])[invalid] == 0)
    assert covered == int(np.sum(valid))


def test_cancellation_split_across_fragments_keeps_seeded_correction() -> None:
    relation = phx.sparse.EdgeRelation(
        np.zeros(4, dtype=np.int32),
        np.zeros(4, dtype=np.int32),
        source_size=1,
        target_size=1,
    )
    events = jnp.asarray([1.0e16, 1.0, -1.0e16, 1.0])
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)
    )

    def total(edge_tile: int, accumulation: phx.sparse.RelationAccumulation) -> float:
        prepared = phx.sparse.StreamedRelationPlan(
            receiver_tile=1, edge_tile=edge_tile, accumulation=accumulation
        ).prepare(relation, owner_id="test-cancellation")
        result = prepared.evaluate(
            payload,
            lambda p, s, r, e: e,
            lambda p, r, a: a,
            None,
            jnp.zeros((1,)),
            jnp.zeros((1,)),
            events,
        )
        return float(result.receiver_outputs[0])

    # One event per fragment: only a correction carried across every fragment
    # recovers 2; summing fragment subtotals or dropping it yields 0 or 1.
    assert total(1, "compensated") == 2.0
    assert total(4, "compensated") == 2.0
    assert total(1, "deterministic") == total(4, "deterministic") == 1.0


def test_nonlinear_epilogue_runs_once_after_final_fragment() -> None:
    relation = phx.sparse.EdgeRelation(
        np.arange(5, dtype=np.int32),
        np.asarray([1, 1, 1, 1, 1], dtype=np.int32),
        source_size=5,
        target_size=2,
    )
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)
    )
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=2).prepare(
        relation, owner_id="test-epilogue"
    )

    result = prepared.evaluate(
        payload,
        lambda p, source, r, e: source,
        lambda p, receiver, aggregate: aggregate**2 + receiver,
        None,
        jnp.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        jnp.asarray([10.0, 20.0]),
        jnp.zeros((5,)),
    )

    # Squaring the complete aggregate, not each fragment or event.
    np.testing.assert_array_equal(result.receiver_outputs, [10.0, 245.0])
    assert int(result.evidence.fragmented_receivers) == 1


def test_empty_rows_get_zero_aggregate_and_masked_receivers_commit_zero() -> None:
    relation, source_index, target_index, valid = _relation()
    parameters, sources, receivers, edges = _data()
    receiver_valid = np.ones(RECEIVER_COUNT, dtype=np.bool_)
    receiver_valid[3] = False
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3).prepare(
        relation, owner_id="test-masks", receiver_valid=jnp.asarray(receiver_valid)
    )
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT)

    result = prepared.evaluate(
        payload, _message, _epilogue, parameters, sources, receivers, edges
    )
    admitted = valid & np.isin(target_index, np.flatnonzero(receiver_valid))
    expected, _ = _reference(
        parameters,
        sources,
        receivers,
        edges,
        source_index=source_index,
        target_index=target_index,
        admitted=admitted,
        receiver_valid=receiver_valid,
    )

    np.testing.assert_allclose(result.receiver_outputs["h"], expected["h"], atol=1e-13)
    # Receiver 5 has no routes: its lawful epilogue of a zero aggregate.
    np.testing.assert_allclose(
        result.receiver_outputs["h"][5], jnp.tanh(receivers["h"][5]), atol=1e-15
    )
    np.testing.assert_array_equal(result.receiver_outputs["h"][3], [0.0, 0.0])
    assert int(result.evidence.committed_receivers) == RECEIVER_COUNT - 1
    assert int(result.evidence.active_routes) == int(admitted.sum())


def test_inactive_and_invalid_events_never_reach_the_callback_domain() -> None:
    relation, source_index, target_index, valid = _relation()
    parameters, sources, receivers, edges = _data()
    # Route storage slot of (1, 0) duplicates: coincident endpoints make log
    # singular, so these events must be masked before evaluation.
    coincident = np.flatnonzero((source_index == 1) & (target_index == 0))
    receivers = {
        "x": receivers["x"].at[0].set(sources["x"][1]),
        "h": receivers["h"],
    }
    active = np.ones(len(_ROUTES), dtype=np.bool_)
    active[coincident] = False
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=1, edge_tile=2).prepare(
        relation, owner_id="test-domain"
    )
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT)

    def observable(theta: dict[str, Array], x: Array) -> Array:
        result = prepared.evaluate(
            payload,
            _message,
            _epilogue,
            theta,
            {"x": x, "h": sources["h"]},
            receivers,
            edges,
            edge_active=jnp.asarray(active),
        )
        return jnp.sum(result.receiver_outputs["h"]) + jnp.sum(
            result.receiver_outputs["n"]
        )

    value, gradients = jax.value_and_grad(observable, argnums=(0, 1))(
        parameters, sources["x"]
    )
    expected, _ = _reference(
        parameters,
        sources,
        receivers,
        edges,
        source_index=source_index,
        target_index=target_index,
        admitted=valid & active,
    )

    np.testing.assert_allclose(
        value, jnp.sum(expected["h"]) + jnp.sum(expected["n"]), rtol=1e-13
    )
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(gradients))
    result = prepared.evaluate(
        payload,
        _message,
        _epilogue,
        parameters,
        sources,
        receivers,
        edges,
        edge_active=jnp.asarray(active),
    )
    # Receiver 0's only routes are inactive: its epilogue sees a zero aggregate.
    np.testing.assert_allclose(
        result.receiver_outputs["h"][0], jnp.tanh(receivers["h"][0]), atol=1e-15
    )
    assert int(result.evidence.evaluated_routes) == int((valid & active).sum())


def test_stable_ids_fix_event_order_under_route_storage_permutation() -> None:
    rng = np.random.default_rng(5)
    # Mixed magnitudes make the floating-point sum order-sensitive.
    events = jnp.asarray(
        rng.normal(size=len(_ROUTES)) * 10.0 ** rng.integers(-8, 9, size=len(_ROUTES))
    )
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    plan = phx.sparse.StreamedRelationPlan(
        receiver_tile=2, edge_tile=3, accumulation="deterministic"
    )

    def outputs(storage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        relation, *_ = _relation(storage)
        prepared = plan.prepare(
            relation,
            owner_id="test-order",
            stable_route_ids=jnp.asarray(storage, dtype=jnp.int32),
        )
        result = prepared.evaluate(
            payload,
            lambda p, s, r, e: (e, 2.0 * e),
            lambda p, r, a: a,
            None,
            jnp.zeros((SOURCE_COUNT,)),
            jnp.zeros((RECEIVER_COUNT,)),
            events[storage],
        )
        edge_outputs = np.empty(len(_ROUTES))
        edge_outputs[storage] = np.asarray(result.edge_outputs)
        return np.asarray(result.receiver_outputs), edge_outputs

    reference_storage = np.arange(len(_ROUTES))
    first = outputs(reference_storage)
    second = outputs(_STORAGE)

    # Deterministic sums follow (receiver, stable ID) order bitwise.
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])


def test_row_relation_cases_stay_isolated() -> None:
    sources = np.asarray([[[0, 1, 1], [2, 0, 0]], [[1, 1, 2], [0, 2, 1]]], dtype=np.int32)
    valid = np.asarray(
        [
            [[True, True, True], [True, False, False]],
            [[True, True, False], [False, False, False]],
        ]
    )
    relation = phx.sparse.RowRelation(
        sources, source_size=3, valid=valid, case_shape=(2,)
    )
    values = jnp.asarray([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=1, edge_tile=2).prepare(
        relation, owner_id="test-cases"
    )
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((), jnp.float64), jax.ShapeDtypeStruct((), jnp.float64)
    )

    result = prepared.evaluate(
        payload,
        lambda p, source, r, e: jnp.exp(source),
        lambda p, r, aggregate: jnp.log1p(aggregate),
        None,
        values,
        jnp.zeros((2, 2)),
        jnp.zeros(sources.shape),
    )

    expected = np.log1p(
        [
            [np.exp(1.0) + 2 * np.exp(2.0), np.exp(3.0)],
            [2 * np.exp(20.0), 0.0],
        ]
    )
    np.testing.assert_allclose(result.receiver_outputs, expected, rtol=1e-14)


def test_traced_preparation_matches_host_preparation() -> None:
    relation, *_ = _relation()
    parameters, sources, receivers, edges = _data()
    plan = phx.sparse.StreamedRelationPlan(
        receiver_tile=2, edge_tile=3, accumulation="deterministic"
    )
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT)

    def run(source: Array, target: Array, valid: Array, epoch: Array) -> Any:
        traced = phx.sparse.EdgeRelation(
            source,
            target,
            source_size=SOURCE_COUNT,
            target_size=RECEIVER_COUNT,
            valid=valid,
        )
        prepared = plan.prepare(traced, owner_id="test-rebuild", epoch=epoch)
        return prepared.evaluate(
            payload, _message, _epilogue, parameters, sources, receivers, edges
        )

    traced = jax.jit(run)(
        relation.source_indices, relation.target_indices, relation.valid, jnp.asarray(4)
    )
    host = plan.prepare(relation, owner_id="test-rebuild").evaluate(
        payload, _message, _epilogue, parameters, sources, receivers, edges
    )

    for left, right in zip(
        jax.tree.leaves(traced.receiver_outputs), jax.tree.leaves(host.receiver_outputs)
    ):
        np.testing.assert_array_equal(left, right)
    assert bool(traced.evidence.successful)
    assert int(traced.evidence.tiles_used) == host.evidence.tile_count
    assert traced.evidence.tile_count >= host.evidence.tile_count
    assert int(traced.evidence.binding.epoch) == 4
    assert traced.evidence.binding.content_id is None
    assert host.evidence.binding.content_id is not None


def test_host_topology_binding_distinguishes_equal_shape_topologies() -> None:
    relation, *_ = _relation()
    permuted, *_ = _relation(np.roll(_STORAGE, 1))
    plan = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3)

    first = plan.prepare(relation, owner_id="test-binding")
    second = plan.prepare(permuted, owner_id="test-binding")
    again = plan.prepare(relation, owner_id="test-binding")

    assert first.binding.binding_id != second.binding.binding_id
    assert first.execution_id != second.execution_id
    assert first.execution_id == again.execution_id


def test_streamed_workspace_is_independent_of_route_capacity() -> None:
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT)
    parameters, *_ = _data()

    def resources(receivers: int, degree: int) -> phx.sparse.StreamedRelationResources:
        routes = receivers * degree
        relation = phx.sparse.EdgeRelation(
            np.arange(routes, dtype=np.int32) % receivers,
            np.repeat(np.arange(receivers, dtype=np.int32), degree),
            source_size=receivers,
            target_size=receivers,
        )
        prepared = phx.sparse.StreamedRelationPlan(receiver_tile=4, edge_tile=16).prepare(
            relation, owner_id="test-resources"
        )
        nodes = {"x": jnp.zeros((receivers, 3)), "h": jnp.zeros((receivers, 2))}
        return prepared.resources(payload, parameters, nodes, nodes, jnp.zeros((routes,)))

    small = resources(8, 3)
    large = resources(64, 12)

    assert small.edge_workspace_bytes == large.edge_workspace_bytes
    assert small.receiver_workspace_bytes == large.receiver_workspace_bytes
    assert small.spill_carry_bytes == large.spill_carry_bytes
    assert large.output_bytes == 8 * small.output_bytes
    assert large.tile_count > small.tile_count


def test_array_bearing_callbacks_count_in_cotangent_bound() -> None:
    relation, *_ = _relation()
    parameters, sources, receivers, edges = _data()
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT, _EDGE_OUTPUT)
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3).prepare(
        relation, owner_id="test-callback-cotangents"
    )
    # 10,000 float64 trainable weights bound outside the explicit parameters.
    weights = jnp.linspace(0.5, 1.5, 10_000)
    rows = sum(
        leaf.nbytes for leaf in jax.tree.leaves((parameters, sources, receivers, edges))
    )

    def edge(w: Array, p: Any, s: Any, r: Any, e: Array) -> Any:
        return _message_with_edge_output(p, s, r, e * jnp.mean(w))

    def aggregate(w: Array, p: Any, s: Any, r: Any, e: Array, f: Any, seed: Any) -> Any:
        return _fragment_aggregate(p, s, r, e * jnp.mean(w), f, seed)

    def epilogue(w: Array, p: Any, r: Any, a: Any) -> Any:
        return jax.tree.map(lambda value: value * jnp.mean(w), _epilogue(p, r, a))

    def evaluated(
        edge_function: Any = _message_with_edge_output,
        receiver_epilogue: Any = _epilogue,
        fragment_aggregate: Any = None,
    ) -> int:
        if fragment_aggregate is None:
            result = prepared.evaluate(
                payload,
                edge_function,
                receiver_epilogue,
                parameters,
                sources,
                receivers,
                edges,
            )
        else:
            result = prepared.evaluate_fragments(
                payload,
                fragment_aggregate,
                receiver_epilogue,
                parameters,
                sources,
                receivers,
                edges,
            )
        return result.resources.cotangent_accumulator_bytes

    bound_edge = Partial(edge, weights)
    assert evaluated() == rows
    assert evaluated(bound_edge) == rows + weights.nbytes
    assert (
        evaluated(fragment_aggregate=Partial(aggregate, weights)) == rows + weights.nbytes
    )
    assert evaluated(bound_edge, Partial(epilogue, weights)) == rows + 2 * weights.nbytes
    assert (
        prepared.resources(
            payload,
            parameters,
            sources,
            receivers,
            edges,
            callbacks=(bound_edge, _epilogue),
        ).cotangent_accumulator_bytes
        == rows + weights.nbytes
    )


def test_small_topology_is_not_padded_to_requested_tiles() -> None:
    relation, source_index, target_index, valid = _relation()
    parameters, sources, receivers, edges = _data()
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT, _EDGE_OUTPUT)
    default = phx.sparse.StreamedRelationPlan(transpose=True)
    exact = phx.sparse.StreamedRelationPlan(
        receiver_tile=RECEIVER_COUNT, edge_tile=len(_ROUTES), transpose=True
    )

    prepared = default.prepare(relation, owner_id="test-effective-widths")
    matched = exact.prepare(relation, owner_id="test-effective-widths")
    used = prepared.resources(payload, parameters, sources, receivers, edges)
    needed = matched.resources(payload, parameters, sources, receivers, edges)

    # The default 256 x 8192 request is an upper bound: a 7-receiver, 22-route
    # relation is scheduled exactly as if those extents had been requested.
    assert (prepared.schedule.receiver_tile, prepared.schedule.edge_tile) == (
        RECEIVER_COUNT,
        len(_ROUTES),
    )
    assert prepared.transpose().schedule.receiver_tile == SOURCE_COUNT
    assert used.edge_workspace_bytes == needed.edge_workspace_bytes
    assert used.receiver_workspace_bytes == needed.receiver_workspace_bytes
    assert used.output_staging_bytes == needed.output_staging_bytes
    assert used.persistent_schedule_bytes == needed.persistent_schedule_bytes
    # The declared plan keeps its identity; only the schedule adapts.
    assert prepared.plan.plan_id == default.plan_id
    result = prepared.evaluate(
        payload,
        _message_with_edge_output,
        _epilogue,
        parameters,
        sources,
        receivers,
        edges,
    )
    expected, expected_edges = _reference(
        parameters,
        sources,
        receivers,
        edges,
        source_index=source_index,
        target_index=target_index,
        admitted=valid,
    )
    np.testing.assert_allclose(result.receiver_outputs["h"], expected["h"], atol=1e-13)
    assert result.edge_outputs is not None
    np.testing.assert_allclose(result.edge_outputs, expected_edges, atol=1e-14)


@pytest.mark.parametrize("traced", [False, True], ids=["host", "traced"])
def test_relation_without_routes_commits_zero_aggregate_epilogues(traced: bool) -> None:
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((2,), jnp.float64),
        jax.ShapeDtypeStruct((2,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    plan = phx.sparse.StreamedRelationPlan(transpose=True)
    receivers = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def run(source: Array, target: Array) -> Any:
        relation = phx.sparse.EdgeRelation(source, target, source_size=2, target_size=3)
        prepared = plan.prepare(relation, owner_id="test-zero-routes")
        return prepared.evaluate(
            payload,
            lambda p, s, r, e: (s, jnp.sum(s)),
            lambda p, receiver, aggregate: jnp.exp(aggregate) + receiver,
            None,
            jnp.ones((2, 2)),
            receivers,
            jnp.zeros((0,)),
        )

    empty = jnp.zeros((0,), dtype=jnp.int32)
    result = jax.jit(run)(empty, empty) if traced else run(empty, empty)

    np.testing.assert_array_equal(result.receiver_outputs, receivers + 1.0)
    assert result.edge_outputs.shape == (0,)
    assert bool(result.evidence.successful)
    assert int(result.evidence.evaluated_routes) == 0
    assert int(result.evidence.committed_receivers) == 3


def test_payload_beyond_channel_capacity_is_refused() -> None:
    relation, *_ = _relation()
    parameters, sources, receivers, edges = _data()
    prepared = phx.sparse.StreamedRelationPlan(channel_capacity=7).prepare(
        relation, owner_id="test-capacity"
    )
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT)

    with pytest.raises(ValueError, match="channel elements"):
        prepared.evaluate(
            payload, _message, _epilogue, parameters, sources, receivers, edges
        )


def test_callback_results_must_match_declared_payload() -> None:
    relation, *_ = _relation()
    parameters, sources, receivers, edges = _data()
    prepared = phx.sparse.StreamedRelationPlan().prepare(relation, owner_id="test-spec")
    wide = phx.sparse.StreamedPayloadSpec(
        {"s": jax.ShapeDtypeStruct((3,), jnp.float64), "v": _MESSAGE["v"]}, _OUTPUT
    )
    single = phx.sparse.StreamedPayloadSpec(
        {"s": jax.ShapeDtypeStruct((2,), jnp.float32), "v": _MESSAGE["v"]}, _OUTPUT
    )

    with pytest.raises(ValueError, match="declared"):
        prepared.evaluate(
            wide, _message, _epilogue, parameters, sources, receivers, edges
        )
    with pytest.raises(TypeError, match="dtype"):
        prepared.evaluate(
            single, _message, _epilogue, parameters, sources, receivers, edges
        )


def test_duplicate_stable_ids_are_refused_on_host_and_poison_traced_results() -> None:
    relation, *_ = _relation()
    parameters, sources, receivers, edges = _data()
    plan = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3)
    duplicated = jnp.zeros((len(_ROUTES),), dtype=jnp.int32)
    payload = phx.sparse.StreamedPayloadSpec(_MESSAGE, _OUTPUT)

    with pytest.raises(ValueError, match="duplicate stable route IDs"):
        plan.prepare(relation, owner_id="test-duplicates", stable_route_ids=duplicated)

    @jax.jit
    def traced(ids: Array) -> Any:
        prepared = plan.prepare(
            relation, owner_id="test-duplicates", stable_route_ids=ids
        )
        return prepared.evaluate(
            payload, _message, _epilogue, parameters, sources, receivers, edges
        )

    result = traced(duplicated)
    assert not bool(result.evidence.successful)
    assert int(result.evidence.duplicate_stable_ids) > 0
    assert bool(jnp.all(jnp.isnan(result.receiver_outputs["h"])))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"receiver_tile": 0}, "receiver_tile must be positive"),
        ({"replay": "scheduled"}, "Scheduled replay"),
        ({"replay": "block"}, "replay_block_size"),
        ({"replay_block_size": 2}, "Only block replay"),
    ],
    ids=["empty-tile", "scheduled-replay", "block-without-size", "size-without-block"],
)
def test_invalid_plans_are_refused(options: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        phx.sparse.StreamedRelationPlan(**options)


def test_transpose_requires_prepared_source_major_schedule() -> None:
    relation, *_ = _relation()
    prepared = phx.sparse.StreamedRelationPlan().prepare(
        relation, owner_id="test-transpose"
    )

    with pytest.raises(ValueError, match="transpose=True"):
        prepared.transpose()
