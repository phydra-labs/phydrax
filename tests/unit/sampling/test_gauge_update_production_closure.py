#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.sampling._gauge_updates as gauge_updates
from phydrax.discretization import polygonal_cell_complex, prepare_cell_boundary_paths
from phydrax.graph import MatrixGaugeLinkSpace
from phydrax.graph._gauge_transport import GaugeStaplePlan
from phydrax.metrix import SpecialUnitaryGroup, UnitaryGroup
from phydrax.sampling._gauge_updates import (
    gauge_replica_exchange,
    gauge_update_sweeps,
    GaugeReplicaExchangePlan,
    GaugeUpdatePlan,
    initialize_gauge_replica_state,
    initialize_gauge_update_state,
    prepare_gauge_update,
)


def _prepared(group: Any, kind: Any = "heatbath", attempts: Any = 128) -> Any:
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    boundaries = prepare_cell_boundary_paths(topology)
    space = MatrixGaugeLinkSpace(topology, group)
    staples = GaugeStaplePlan(space, boundaries)
    colors = jnp.arange(space.num_edges)
    conflicts = ~jnp.eye(space.num_edges, dtype="bool")
    name = "u1" if group.dimension == 1 else f"su{group.dimension}"
    plan = GaugeUpdatePlan(
        # ty: ignore[invalid-argument-type]
        name,
        coupling=0.8,
        update=kind,
        rejection_attempts=attempts,
    )
    return space, prepare_gauge_update(plan, staples, colors, conflicts)


def test_gauge_update_production_closure_scenario_1() -> None:
    space, prepared = _prepared(UnitaryGroup(1), attempts=256)
    state = initialize_gauge_update_state(prepared, space.identity())
    result = gauge_update_sweeps(prepared, state, key=jax.random.key(11))

    evidence = result.evidence
    assert space.contains(result.state.links)
    assert jnp.all(evidence.membership_preserved)
    assert jnp.allclose(
        evidence.log_forward_reverse_ratio + evidence.log_target_ratio,
        0.0,
        atol=2e-6,
    )
    assert jnp.all(evidence.rejection_attempts <= prepared.rejection_attempts)
    assert jnp.all((evidence.status == 0) | (evidence.status == 1))
    assert result.reference_measure == "product-haar"
    space, heatbath = _prepared(SpecialUnitaryGroup(2), attempts=256)
    initial = initialize_gauge_update_state(heatbath, space.identity())
    sampled = gauge_update_sweeps(heatbath, initial, key=jax.random.key(12))
    assert space.contains(sampled.state.links)
    assert jnp.all(sampled.evidence.exact_target_correction)

    _, reflection = _prepared(SpecialUnitaryGroup(2), kind="overrelaxation")
    reflected_state = initialize_gauge_update_state(reflection, sampled.state.links)
    reflected = gauge_update_sweeps(reflection, reflected_state, key=jax.random.key(13))
    assert space.contains(reflected.state.links)
    assert jnp.allclose(reflected.evidence.log_target_ratio, 0.0, atol=3e-5)
    assert jnp.all(reflected.evidence.exact_target_correction)
    space, prepared = _prepared(SpecialUnitaryGroup(3), attempts=256)
    state = initialize_gauge_update_state(prepared, space.identity())
    result = gauge_update_sweeps(prepared, state, key=jax.random.key(14))

    assert space.contains(result.state.links)
    assert result.evidence.link_index.shape == (3 * space.num_edges,)
    assert jnp.all(result.evidence.membership_preserved)
    assert jnp.allclose(
        result.evidence.log_forward_reverse_ratio + result.evidence.log_target_ratio,
        0.0,
        atol=3e-5,
    )


def test_failed_overrelaxation_correction_rolls_back_every_link(monkeypatch: Any) -> None:
    space, prepared = _prepared(SpecialUnitaryGroup(2), kind="overrelaxation")
    initial = initialize_gauge_update_state(prepared, space.identity())
    calls = 0

    def non_microcanonical_weight(*_args: Any) -> Any:
        nonlocal calls
        value = jnp.asarray(float(calls % 2))
        calls += 1
        return value

    monkeypatch.setattr(gauge_updates, "_local_log_weight", non_microcanonical_weight)
    result = gauge_update_sweeps(prepared, initial, key=jax.random.key(131))

    assert jnp.any(
        result.evidence.status
        == gauge_updates.GaugeUpdateStatus.MICROCANONICAL_INVARIANCE_FAILURE
    )
    assert not jnp.any(result.evidence.accepted)
    assert jnp.array_equal(result.state.links, initial.links)


def test_gauge_update_production_closure_scenario_2() -> None:
    topology = polygonal_cell_complex(jnp.asarray([[0, 1, 2]]), None, 3)
    boundaries = prepare_cell_boundary_paths(topology)
    space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(2))
    staples = GaugeStaplePlan(space, boundaries)
    conflicts = ~jnp.eye(space.num_edges, dtype="bool")

    with pytest.raises(ValueError, match="sharing a sweep color conflict"):
        prepare_gauge_update(
            GaugeUpdatePlan("su2", coupling=1.0),
            staples,
            jnp.zeros(space.num_edges, dtype="int64"),
            conflicts,
        )
    with pytest.raises(ValueError, match="link count, coloring"):
        prepare_gauge_update(
            GaugeUpdatePlan("su3", coupling=1.0),
            staples,
            jnp.arange(space.num_edges),
            conflicts,
        )
    plan = GaugeReplicaExchangePlan(jnp.asarray([1.0, 0.7, 0.4]))
    configurations = jnp.arange(6.0).reshape((3, 2))
    reduced = jnp.asarray(
        [
            [0.0, 4.0, 8.0],
            [8.0, 0.0, 4.0],
            [4.0, 8.0, 0.0],
        ]
    )
    state = initialize_gauge_replica_state(plan, configurations, reduced)
    first = gauge_replica_exchange(plan, state, key=jax.random.key(15))
    second = gauge_replica_exchange(plan, first.state, key=jax.random.key(15))

    assert jnp.array_equal(first.attempted, jnp.asarray([True, False]))
    assert jnp.array_equal(second.attempted, jnp.asarray([False, True]))
    assert first.log_target_ratio[0] == -12.0
    assert jnp.allclose(first.detailed_balance_residual, 0.0)
    assert jnp.allclose(first.log_forward_reverse_ratio, 0.0)
    assert jnp.array_equal(first.status, jnp.asarray([0, 4], dtype=jnp.int32))
    assert first.state.attempted_swaps == 1
    assert second.state.attempted_swaps == 2


def test_native_staple_dependencies_preserve_colored_microcanonical_action() -> None:
    faces = ((0, 1, 2), (0, 2, 3))
    topology = polygonal_cell_complex(jnp.asarray(faces), None, 4)
    boundaries = prepare_cell_boundary_paths(topology)
    space = MatrixGaugeLinkSpace(topology, SpecialUnitaryGroup(2))
    staples = GaugeStaplePlan(space, boundaries)
    tails = np.asarray(space.tail_vertices)
    heads = np.asarray(space.head_vertices)
    lookup = {
        tuple(sorted((int(tail), int(head)))): edge
        for edge, (tail, head) in enumerate(zip(tails, heads, strict=True))
    }
    conflicts = np.zeros((space.num_edges, space.num_edges), dtype=np.bool_)
    for face in faces:
        face_edges = tuple(
            lookup[tuple(sorted(pair))]
            for pair in zip(face, face[1:] + face[:1], strict=True)
        )
        for edge in face_edges:
            for other in face_edges:
                conflicts[edge, other] = edge != other
    omitted = conflicts.copy()
    first = lookup[(0, 1)]
    shared = lookup[(0, 2)]
    omitted[first, shared] = omitted[shared, first] = False
    with pytest.raises(ValueError, match="omits a dependency"):
        prepare_gauge_update(
            GaugeUpdatePlan("su2", coupling=1.0),
            staples,
            np.arange(space.num_edges, dtype=np.int32),
            omitted,
        )
    colors = np.arange(space.num_edges, dtype=np.int32)
    # These edges lie in different triangles and may update simultaneously.
    colors[lookup[(2, 3)]] = colors[first]
    colors = np.searchsorted(np.unique(colors), colors)
    prepared = prepare_gauge_update(
        GaugeUpdatePlan("su2", coupling=1.0, update="overrelaxation"),
        staples,
        colors,
        conflicts,
    )
    coordinates = 0.2 * jnp.sin(
        jnp.arange(space.num_edges * 3, dtype=jnp.float64).reshape((space.num_edges, 3))
    )
    links = jax.vmap(lambda value: space.group.exp(space.group.hat(value)))(coordinates)

    def wilson_action(values: jax.Array) -> jax.Array:
        traces: list[jax.Array] = []
        for face in faces:
            product = jnp.eye(2, dtype=values.dtype)
            for tail, head in zip(face, face[1:] + face[:1], strict=True):
                edge = lookup[tuple(sorted((tail, head)))]
                factor = values[edge]
                if int(tails[edge]) != tail:
                    factor = jnp.conj(factor.T)
                product = product @ factor
            traces.append(jnp.real(jnp.trace(product)) / 2)
        return jnp.sum(1 - jnp.stack(traces))

    initial = initialize_gauge_update_state(prepared, links)
    result = gauge_update_sweeps(prepared, initial, key=jax.random.key(132))
    np.testing.assert_allclose(
        wilson_action(result.state.links), wilson_action(links), atol=3e-5
    )
    assert jnp.max(jnp.abs(result.state.links - links)) > 1e-4
