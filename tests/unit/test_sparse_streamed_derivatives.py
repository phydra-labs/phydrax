#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Ordinary-JAX derivatives of prepared streamed relations.

Streamed evaluations are differentiated by JAX itself; these tests compare
their JVP, VJP, nested force-loss and HVP actions with an independent plain-JAX
per-event reference, finite differences and linear transpose/adjoint algebra.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from jax.tree_util import Partial

import phydrax as phx


SOURCES = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 4, 3, 3, 0, 2], dtype=np.int32)
TARGETS = np.asarray([2, 2, 2, 2, 2, 2, 2, 0, 0, 1, 3, 3, 9], dtype=np.int32)
VALID = np.asarray([True] * 12 + [False])
NODE_COUNT = 5
_SCALAR = jax.ShapeDtypeStruct((), jnp.float64)
_PAYLOAD = phx.sparse.StreamedPayloadSpec(
    {"m": jax.ShapeDtypeStruct((2,), jnp.float64)}, _SCALAR
)


def _relation() -> phx.sparse.EdgeRelation:
    return phx.sparse.EdgeRelation(
        SOURCES, TARGETS, source_size=NODE_COUNT, target_size=NODE_COUNT, valid=VALID
    )


def _edge(
    theta: dict[str, Array], source: Array, receiver: Array, edge: Array
) -> dict[str, Array]:
    displacement = receiver - source
    distance = jnp.sqrt(jnp.sum(displacement * displacement) + 1.0)
    return {
        "m": jnp.tanh(theta["a"] @ displacement) * jnp.exp(-theta["b"] * distance) * edge
    }


def _readout(
    theta: dict[str, Array], receiver: Array, aggregate: dict[str, Array]
) -> Array:
    return jnp.sum(jnp.sin(aggregate["m"] * theta["c"])) + 0.1 * jnp.sum(receiver**2)


def _parameters() -> dict[str, Array]:
    rng = np.random.default_rng(17)
    return {
        "a": jnp.asarray(rng.normal(size=(2, 3))),
        "b": jnp.asarray(0.3),
        "c": jnp.asarray(rng.normal(size=(2,))),
    }


def _positions() -> Array:
    return jnp.asarray(np.random.default_rng(23).normal(size=(NODE_COUNT, 3)))


_EDGES = jnp.asarray(np.random.default_rng(29).uniform(0.5, 1.5, size=SOURCES.shape))


def _streamed_energy(
    plan: phx.sparse.StreamedRelationPlan,
) -> Any:
    prepared = plan.prepare(_relation(), owner_id="test-derivatives")

    def energy(theta: dict[str, Array], positions: Array) -> Array:
        result = prepared.evaluate(
            _PAYLOAD, _edge, _readout, theta, positions, positions, _EDGES
        )
        return jnp.sum(result.receiver_outputs)

    return energy


def _reference_energy(theta: dict[str, Array], positions: Array) -> Array:
    routes = np.flatnonzero(VALID)
    messages = jax.vmap(lambda s, r, e: _edge(theta, s, r, e))(
        positions[SOURCES[routes]], positions[TARGETS[routes]], _EDGES[routes]
    )
    aggregate = {"m": jnp.zeros((NODE_COUNT, 2)).at[TARGETS[routes]].add(messages["m"])}
    return jnp.sum(jax.vmap(lambda r, a: _readout(theta, r, a))(positions, aggregate))


_PLANS = [
    phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3, accumulation="fast"),
    phx.sparse.StreamedRelationPlan(
        receiver_tile=1, edge_tile=2, accumulation="deterministic"
    ),
    phx.sparse.StreamedRelationPlan(
        receiver_tile=2,
        edge_tile=4,
        accumulation="compensated",
        replay="block",
        replay_block_size=2,
    ),
    phx.sparse.StreamedRelationPlan(receiver_tile=4, edge_tile=16, replay="full"),
]
_PLAN_IDS = ["step-fast", "step-deterministic", "block-compensated", "full-replay"]


@pytest.mark.parametrize("plan", _PLANS, ids=_PLAN_IDS)
def test_coordinate_jvp_vjp_match_finite_differences_and_duality(
    plan: phx.sparse.StreamedRelationPlan,
) -> None:
    energy = _streamed_energy(plan)
    theta = _parameters()
    positions = _positions()
    direction = jnp.asarray(np.random.default_rng(31).normal(size=positions.shape))

    def node_outputs(x: Array) -> Array:
        return jax.grad(energy, argnums=1)(theta, x)

    value, tangent = jax.jvp(lambda x: energy(theta, x), (positions,), (direction,))
    step = 1.0e-6
    difference = (
        energy(theta, positions + step * direction)
        - energy(theta, positions - step * direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(value, _reference_energy(theta, positions), rtol=1e-13)
    np.testing.assert_allclose(tangent, difference, rtol=1e-7)

    # Duality of the force map's linearization: <u, J v> = <J^T u, v>.
    cotangent = jnp.asarray(np.random.default_rng(37).normal(size=positions.shape))
    _, forward = jax.jvp(node_outputs, (positions,), (direction,))
    _, pullback = jax.vjp(node_outputs, positions)
    (backward,) = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(cotangent, forward), jnp.vdot(backward, direction), rtol=1e-11
    )


def test_array_bearing_callbacks_share_parameter_cotangents() -> None:
    weights = jnp.asarray([0.7, -0.4])
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3).prepare(
        _relation(), owner_id="test-callables"
    )
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((2,), jnp.float64), _SCALAR
    )

    def edge(w: Array, _: None, source: Array, receiver: Array, scale: Array) -> Array:
        return jnp.tanh(w * jnp.sum(receiver - source)) * scale

    def readout(w: Array, _: None, receiver: Array, aggregate: Array) -> Array:
        return jnp.sum(jnp.cos(w * aggregate)) + jnp.sum(receiver)

    def streamed(w: Array, positions: Array) -> Array:
        result = prepared.evaluate(
            payload,
            Partial(edge, w),
            Partial(readout, w),
            None,
            positions,
            positions,
            _EDGES,
        )
        return jnp.sum(result.receiver_outputs)

    def reference(w: Array, positions: Array) -> Array:
        routes = np.flatnonzero(VALID)
        messages = jax.vmap(lambda s, r, e: edge(w, None, s, r, e))(
            positions[SOURCES[routes]], positions[TARGETS[routes]], _EDGES[routes]
        )
        aggregate = jnp.zeros((NODE_COUNT, 2)).at[TARGETS[routes]].add(messages)
        return jnp.sum(
            jax.vmap(lambda r, a: readout(w, None, r, a))(positions, aggregate)
        )

    positions = _positions()
    gradient = jax.grad(streamed, argnums=(0, 1))(weights, positions)
    expected = jax.grad(reference, argnums=(0, 1))(weights, positions)

    np.testing.assert_allclose(gradient[0], expected[0], rtol=1e-12)
    np.testing.assert_allclose(gradient[1], expected[1], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("plan", _PLANS, ids=_PLAN_IDS)
def test_force_loss_parameter_gradient_matches_reference(
    plan: phx.sparse.StreamedRelationPlan,
) -> None:
    energy = _streamed_energy(plan)
    positions = _positions()
    labels = jnp.asarray(np.random.default_rng(41).normal(size=positions.shape))

    def force_loss(model: Any) -> Any:
        def loss(theta: dict[str, Array]) -> Array:
            forces = -jax.grad(model, argnums=1)(theta, positions)
            return jnp.sum((forces - labels) ** 2) + model(theta, positions) ** 2

        return loss

    gradient = jax.jit(jax.grad(force_loss(energy)))(_parameters())
    expected = jax.grad(force_loss(_reference_energy))(_parameters())

    for name in ("a", "b", "c"):
        np.testing.assert_allclose(gradient[name], expected[name], rtol=1e-11, atol=1e-13)


@pytest.mark.parametrize("plan", _PLANS, ids=_PLAN_IDS)
def test_coordinate_hessian_vector_product_matches_reference(
    plan: phx.sparse.StreamedRelationPlan,
) -> None:
    energy = _streamed_energy(plan)
    theta = _parameters()
    positions = _positions()
    direction = jnp.asarray(np.random.default_rng(43).normal(size=positions.shape))

    def hvp(model: Any) -> Array:
        return jax.jvp(
            lambda x: jax.grad(model, argnums=1)(theta, x), (positions,), (direction,)
        )[1]

    np.testing.assert_allclose(
        hvp(energy), hvp(_reference_energy), rtol=1e-11, atol=1e-13
    )


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize("edge_tile", [1, 4], ids=["seeded-fragments", "one-fragment"])
def test_valid_zero_valued_events_keep_their_derivatives(
    accumulation: phx.sparse.RelationAccumulation, edge_tile: int
) -> None:
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1, 0], dtype=np.int32),
        np.asarray([0, 0, 0], dtype=np.int32),
        source_size=2,
        target_size=1,
        valid=np.asarray([True, True, False]),
    )
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=1, edge_tile=edge_tile, accumulation=accumulation
    ).prepare(relation, owner_id="test-zero-events")
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR, _SCALAR)

    def energy(theta: Array, x: Array) -> Array:
        result = prepared.evaluate(
            payload,
            lambda t, source, r, e: t * source + source * source,
            lambda t, r, aggregate: aggregate,
            theta,
            x,
            jnp.zeros((1,)),
            jnp.zeros((3,)),
        )
        return result.receiver_outputs[0]

    theta = jnp.asarray(2.0)
    x = jnp.zeros((2,))
    # Every primal message is exactly zero, yet both active events own
    # derivatives; the invalid third event owns none.
    assert float(energy(theta, x)) == 0.0
    np.testing.assert_array_equal(jax.grad(energy, argnums=1)(theta, x), [2.0, 2.0])
    np.testing.assert_array_equal(
        jax.jacfwd(jax.grad(energy, argnums=1), argnums=0)(theta, x), [1.0, 1.0]
    )


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
@pytest.mark.parametrize("edge_tile", [1, 2], ids=["seeded-fragments", "one-fragment"])
def test_valid_zero_valued_events_keep_their_curvature(
    accumulation: phx.sparse.RelationAccumulation, edge_tile: int
) -> None:
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1, 0], dtype=np.int32),
        np.asarray([0, 0, 0], dtype=np.int32),
        source_size=2,
        target_size=1,
        valid=np.asarray([True, True, False]),
    )
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=1, edge_tile=edge_tile, accumulation=accumulation
    ).prepare(relation, owner_id="test-zero-event-curvature")
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR, _SCALAR, _SCALAR)
    residual = jnp.asarray([0.5])
    # The invalid route's NaN edge row is padding: it must stay inert.
    edges = jnp.asarray([0.0, 0.0, jnp.nan])

    def node_edge(theta: Array, x: Array) -> tuple[Array, Array]:
        result = prepared.evaluate(
            payload,
            lambda t, source, r, e: ((t - 1.0) + source + e, source * source),
            lambda t, r, aggregate: (aggregate + r) ** 2,
            theta,
            x,
            residual,
            edges,
        )
        assert result.edge_outputs is not None
        return result.receiver_outputs[0], jnp.sum(result.edge_outputs[:2] ** 2)

    def node_edge_reference(theta: Array, x: Array) -> tuple[Array, Array]:
        return (2.0 * (theta - 1.0) + x[0] + x[1] + residual[0]) ** 2, jnp.sum(x**4)

    def checks(model: Any) -> dict[str, Array]:
        def energy(theta: Array, x: Array) -> Array:
            return model(theta, x)[0]

        def loss(theta: Array, x: Array) -> Array:
            return sum(model(theta, x))

        def force_loss(theta: Array) -> Array:
            forces = -jax.grad(loss, argnums=1)(theta, x)
            return jnp.sum((forces - labels) ** 2) + loss(theta, x) ** 2

        coordinate_gradient = jax.grad(energy, argnums=1)
        return {
            "energy": energy(theta, x),
            "forward-over-reverse": jax.jacfwd(coordinate_gradient, argnums=1)(theta, x),
            "reverse-over-reverse": jax.jacrev(coordinate_gradient, argnums=1)(theta, x),
            "combined-hvp": jax.jvp(
                lambda y: jax.grad(loss, argnums=1)(theta, y), (x,), (direction,)
            )[1],
            "mixed": jax.jacfwd(coordinate_gradient, argnums=0)(theta, x),
            "force-loss": jax.grad(force_loss)(theta),
        }

    # At theta = 1 and x = 0 every valid message is exactly zero, yet both
    # events own their curvature: the energy is (2 (theta - 1) + x0 + x1 + r)^2.
    theta = jnp.asarray(1.0)
    x = jnp.zeros((2,))
    labels = jnp.asarray([0.3, -0.7])
    direction = jnp.asarray([1.0, 2.0])
    streamed = checks(node_edge)
    expected = checks(node_edge_reference)

    np.testing.assert_array_equal(streamed["forward-over-reverse"], np.full((2, 2), 2.0))
    np.testing.assert_array_equal(streamed["reverse-over-reverse"], np.full((2, 2), 2.0))
    for name in expected:
        np.testing.assert_allclose(
            streamed[name], expected[name], rtol=1e-14, atol=1e-15, err_msg=name
        )


def test_complex_linear_subcase_transpose_and_adjoint() -> None:
    rng = np.random.default_rng(47)
    coefficients = jnp.asarray(
        rng.normal(size=SOURCES.shape) + 1j * rng.normal(size=SOURCES.shape)
    )
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=2, edge_tile=3, accumulation="deterministic", transpose=True
    ).prepare(_relation(), owner_id="test-linear")
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((), jnp.complex128), jax.ShapeDtypeStruct((), jnp.complex128)
    )

    def apply(
        relation: phx.sparse.PreparedStreamedRelation, values: Array, weights: Array
    ) -> Array:
        return relation.evaluate(
            payload,
            lambda p, source, r, coefficient: coefficient * source,
            lambda p, r, aggregate: aggregate,
            None,
            values,
            jnp.zeros((NODE_COUNT,), jnp.complex128),
            weights,
        ).receiver_outputs

    dense = np.zeros((NODE_COUNT, NODE_COUNT), dtype=np.complex128)
    for route in np.flatnonzero(VALID):
        dense[TARGETS[route], SOURCES[route]] += complex(coefficients[route])
    x = jnp.asarray(rng.normal(size=NODE_COUNT) + 1j * rng.normal(size=NODE_COUNT))
    y = jnp.asarray(rng.normal(size=NODE_COUNT) + 1j * rng.normal(size=NODE_COUNT))

    transposed = apply(prepared.transpose(), y, coefficients)
    adjoint = apply(prepared.transpose(), y, jnp.conj(coefficients))
    _, pullback = jax.vjp(lambda values: apply(prepared, values, coefficients), x)

    np.testing.assert_allclose(apply(prepared, x, coefficients), dense @ x, rtol=1e-13)
    np.testing.assert_allclose(transposed, dense.T @ y, rtol=1e-13)
    np.testing.assert_allclose(adjoint, dense.conj().T @ y, rtol=1e-13)
    np.testing.assert_allclose(pullback(y)[0], transposed, rtol=1e-13)
    np.testing.assert_allclose(
        jnp.vdot(y, apply(prepared, x, coefficients)), jnp.vdot(adjoint, x), rtol=1e-13
    )
    assert prepared.transpose().transpose().direction == "target"


def test_failed_traced_schedule_invalidates_derivatives() -> None:
    plan = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=3)
    duplicated = jnp.zeros(SOURCES.shape, dtype=jnp.int32)

    @jax.jit
    def gradient(ids: Array, positions: Array) -> Array:
        def energy(x: Array) -> Array:
            prepared = plan.prepare(
                _relation(), owner_id="test-failure", stable_route_ids=ids
            )
            result = prepared.evaluate(
                _PAYLOAD, _edge, _readout, _parameters(), x, x, _EDGES
            )
            return jnp.sum(result.receiver_outputs)

        return jax.grad(energy)(positions)

    assert bool(jnp.all(jnp.isnan(gradient(duplicated, _positions()))))
    assert bool(
        jnp.all(
            jnp.isfinite(
                gradient(jnp.arange(SOURCES.size, dtype=jnp.int32), _positions())
            )
        )
    )


def _single_receiver(
    edge_tile: int, receivers: int = 1
) -> phx.sparse.PreparedStreamedRelation:
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 0], dtype=np.int32),
        source_size=2,
        target_size=receivers,
    )
    return phx.sparse.StreamedRelationPlan(receiver_tile=1, edge_tile=edge_tile).prepare(
        relation, owner_id="test-epilogue-domain"
    )


@pytest.mark.parametrize("edge_tile", [1, 2], ids=["split-receiver", "whole-receiver"])
def test_epilogue_never_sees_partial_fragment_aggregates(edge_tile: int) -> None:
    # Events 0 then 1 into one receiver: with one-event fragments the partial
    # aggregate after the first fragment is exactly 0, outside log's domain,
    # while the final aggregate 1 is lawful.
    prepared = _single_receiver(edge_tile)
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR, _SCALAR)

    def energy(x: Array) -> Array:
        result = prepared.evaluate(
            payload,
            lambda p, source, r, e: source,
            lambda p, r, aggregate: jnp.log(aggregate),
            None,
            x,
            jnp.zeros((1,)),
            jnp.zeros((2,)),
        )
        assert result.receiver_outputs.shape == (1,)
        return result.receiver_outputs[0]

    x = jnp.asarray([0.0, 1.0])
    # Independent reference: E = log(x0 + x1), grad = 1/s, Hessian = -1/s**2.
    assert float(energy(x)) == 0.0
    np.testing.assert_array_equal(jax.grad(energy)(x), [1.0, 1.0])
    np.testing.assert_array_equal(
        jax.jvp(energy, (x,), (jnp.asarray([1.0, 1.0]),))[1], 2.0
    )
    np.testing.assert_array_equal(
        jax.jvp(jax.grad(energy), (x,), (jnp.asarray([1.0, 0.0]),))[1], [-1.0, -1.0]
    )


def test_masked_receiver_epilogue_never_sees_its_own_aggregate() -> None:
    relation = phx.sparse.EdgeRelation(
        np.asarray([0, 1], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
        source_size=2,
        target_size=2,
    )
    prepared = phx.sparse.StreamedRelationPlan(receiver_tile=2, edge_tile=2).prepare(
        relation, owner_id="test-masked-domain", receiver_valid=jnp.asarray([True, False])
    )
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR, _SCALAR)

    def energy(x: Array) -> Array:
        outputs = prepared.evaluate(
            payload,
            lambda p, source, r, e: source,
            lambda p, r, aggregate: jnp.log(aggregate),
            None,
            x,
            jnp.zeros((2,)),
            jnp.zeros((2,)),
        ).receiver_outputs
        return jnp.sum(outputs)

    # Receiver 1 is masked: its zero aggregate never reaches log.
    x = jnp.asarray([2.0, 5.0])
    np.testing.assert_allclose(energy(x), np.log(2.0), rtol=1e-15)
    np.testing.assert_array_equal(jax.grad(energy)(x), [0.5, 0.0])


def _edge_output_result(edge: Array, active: Array) -> phx.sparse.StreamedRelationResult:
    prepared = _single_receiver(2)
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR, _SCALAR, _SCALAR)
    return prepared.evaluate(
        payload,
        lambda p, s, r, e: (jnp.ones(()), jnp.log(e)),
        lambda p, r, aggregate: aggregate,
        None,
        jnp.zeros((2,)),
        jnp.zeros((1,)),
        edge,
        edge_active=active,
    )


def test_non_finite_requested_edge_output_fails_the_evaluation() -> None:
    failed = _edge_output_result(jnp.asarray([1.0, -1.0]), jnp.asarray([True, True]))

    assert not bool(failed.evidence.finite)
    assert not bool(failed.evidence.successful)
    assert bool(jnp.all(jnp.isnan(failed.receiver_outputs)))
    assert failed.edge_outputs is not None
    assert bool(jnp.all(jnp.isnan(failed.edge_outputs)))


def test_inactive_route_edge_output_domain_is_inert() -> None:
    inert = _edge_output_result(jnp.asarray([1.0, -1.0]), jnp.asarray([True, False]))

    assert bool(inert.evidence.successful)
    np.testing.assert_array_equal(inert.receiver_outputs, [1.0])
    np.testing.assert_array_equal(inert.edge_outputs, [0.0, 0.0])


@pytest.mark.parametrize(
    ("scale", "admitted"), [(0.0, False), (1.0, True)], ids=["infinite-message", "finite"]
)
def test_non_finite_message_poisons_residual_outputs_and_derivatives(
    scale: float, admitted: bool
) -> None:
    prepared = _single_receiver(2)
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR, _SCALAR)

    def evaluate(residual: Array) -> phx.sparse.StreamedRelationResult:
        # The epilogue reads only its residual: a failed message would otherwise
        # leave a finite output and Jacobian behind failed evidence.
        return prepared.evaluate(
            payload,
            lambda p, s, r, e: 1.0 / e,
            lambda p, receiver, aggregate: 2.0 * receiver,
            None,
            jnp.zeros((2,)),
            residual,
            jnp.full((2,), scale),
        )

    residual = jnp.asarray([3.0])
    result = evaluate(residual)
    gradient = jax.grad(lambda r: evaluate(r).receiver_outputs[0])(residual)

    assert bool(result.evidence.successful) == admitted
    if admitted:
        np.testing.assert_array_equal(result.receiver_outputs, [6.0])
        np.testing.assert_array_equal(gradient, [2.0])
    else:
        assert bool(jnp.all(jnp.isnan(result.receiver_outputs)))
        assert bool(jnp.all(jnp.isnan(gradient)))


@pytest.mark.parametrize(
    ("theta", "admitted"), [(0.0, False), (2.0, True)], ids=["infinite-message", "finite"]
)
@pytest.mark.parametrize("bound", [False, True], ids=["parameters", "callback-array"])
def test_failed_message_poisons_derivatives_of_unread_inputs(
    theta: float, admitted: bool, bound: bool
) -> None:
    prepared = _single_receiver(2)
    payload = phx.sparse.StreamedPayloadSpec(_SCALAR, _SCALAR)

    def message(t: Array, s: Array) -> Array:
        return 1.0 / t + s

    def evaluate(t: Array, x: Array) -> phx.sparse.StreamedRelationResult:
        # The residual-only epilogue never reads the message, so ``t`` (read
        # only by the failed message) does not reach the output.
        if bound:
            edge, parameters = Partial(lambda w, p, s, r, e: message(w, s), t), None
        else:
            edge, parameters = (lambda p, s, r, e: message(p, s)), t
        return prepared.evaluate(
            payload,
            edge,
            lambda p, receiver, aggregate: 2.0 * receiver**2,
            parameters,
            x,
            jnp.asarray([3.0]),
            jnp.zeros((2,)),
        )

    x = jnp.asarray([0.25, -0.5])
    t = jnp.asarray(theta)

    def output(t: Array) -> Array:
        return evaluate(t, x).receiver_outputs[0]

    def force_loss(t: Array) -> Array:
        forces = jax.grad(lambda y: evaluate(t, y).receiver_outputs[0])(x)
        return jnp.sum((forces - 1.0) ** 2)

    derivatives = [
        jax.grad(output)(t),
        jax.jvp(output, (t,), (jnp.ones(()),))[1],
        jax.grad(jax.grad(output))(t),
        jax.jacfwd(jax.jacfwd(output))(t),
        jax.grad(force_loss)(t),
    ]

    assert bool(evaluate(t, x).evidence.successful) == admitted
    if admitted:
        np.testing.assert_array_equal(output(t), 18.0)
        for derivative in derivatives:
            np.testing.assert_array_equal(derivative, 0.0)
    else:
        assert bool(jnp.isnan(output(t)))
        for derivative in derivatives:
            assert bool(jnp.all(jnp.isnan(derivative)))
