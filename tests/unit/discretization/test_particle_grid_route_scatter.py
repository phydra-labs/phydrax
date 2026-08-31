#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _prepared(*, accumulation: str = "deterministic", particle_ids=(7, 2, 11)):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray(particle_ids),
        jnp.ones((len(particle_ids),)),
        ambient_dimension=1,
    ).prepare()
    return phx.discretization.ParticleGridSplatPlan(
        grid,
        assignment=phx.discretization.TensorBSplineSplatAssignment(2),
        execution=phx.discretization.SplatExecutionPolicy(accumulation=accumulation),
    ).prepare(particles)


def _manual(state, payload, target_size):
    valid = state.stencil.valid[..., None]
    material = jnp.where(valid, payload, 0.0)
    return (
        jnp.zeros((target_size, payload.shape[-1]), dtype=payload.dtype)
        .at[state.stencil.indices.reshape((-1,))]
        .add(material.reshape((-1, payload.shape[-1])))
    )


@pytest.mark.parametrize("accumulation", ["fast", "deterministic", "compensated"])
def test_route_payload_scatter_matches_manual_sum(accumulation):
    prepared = _prepared(accumulation=accumulation)
    position = jnp.asarray([[0.13], [0.52], [0.88]])
    state = prepared.build(position)
    payload = jnp.arange(
        prepared.particles.capacity * prepared.route_width * 2,
        dtype=jnp.float64,
    ).reshape((prepared.particles.capacity, prepared.route_width, 2))

    result = prepared.scatter_route_payload(state, payload)
    expected = _manual(state, payload, prepared.target_size).reshape(
        prepared.target_shape + (2,)
    )

    assert isinstance(result, phx.discretization.SplatRouteScatterResult)
    assert bool(result.successful)
    assert int(result.valid_route_count) == int(jnp.sum(state.stencil.valid))
    np.testing.assert_allclose(result.values, expected, rtol=2e-13, atol=2e-13)


def test_deterministic_route_scatter_is_particle_id_order_invariant():
    position = jnp.asarray([[0.13], [0.52], [0.88]])
    payload = jnp.arange(18, dtype=jnp.float64).reshape((3, 3, 2))
    first = _prepared(particle_ids=(7, 2, 11))
    first_result = first.scatter_route_payload(first.build(position), payload).values

    permutation = jnp.asarray((2, 0, 1))
    second = _prepared(particle_ids=(11, 7, 2))
    second_result = second.scatter_route_payload(
        second.build(position[permutation]), payload[permutation]
    ).values

    np.testing.assert_array_equal(first_result, second_result)


def test_route_scatter_jit_vmap_jvp_vjp_and_finite_difference():
    prepared = _prepared()
    positions = jnp.asarray(
        [
            [[0.13], [0.52], [0.88]],
            [[0.16], [0.55], [0.91]],
        ]
    )
    base_payload = jnp.linspace(-0.5, 0.7, 18).reshape((3, 3, 2))

    @jax.jit
    def apply(position, payload):
        state = prepared.build(position)
        weighted = payload * state.stencil.weights[..., None]
        return prepared.scatter_route_payload(state, weighted).values

    vmapped = jax.vmap(lambda value: apply(value, base_payload))(positions)
    sequential = jnp.stack([apply(value, base_payload) for value in positions])
    np.testing.assert_allclose(vmapped, sequential, rtol=1e-13, atol=1e-13)

    position = positions[0]
    tangent = jnp.linspace(-0.2, 0.3, base_payload.size).reshape(base_payload.shape)
    cotangent = jnp.linspace(-0.4, 0.5, apply(position, base_payload).size).reshape(
        apply(position, base_payload).shape
    )
    _, forward = jax.jvp(
        lambda payload: apply(position, payload),
        (base_payload,),
        (tangent,),
    )
    _, pullback = jax.vjp(lambda payload: apply(position, payload), base_payload)
    reverse = pullback(cotangent)[0]
    np.testing.assert_allclose(
        jnp.vdot(forward, cotangent),
        jnp.vdot(tangent, reverse),
        rtol=1e-12,
        atol=1e-12,
    )

    objective = lambda payload: jnp.sum(apply(position, payload) ** 2)
    epsilon = 1.0e-5
    directional = jnp.vdot(jax.grad(objective)(base_payload), tangent)
    finite = (
        objective(base_payload + epsilon * tangent)
        - objective(base_payload - epsilon * tangent)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(directional, finite, rtol=2e-8, atol=2e-9)


def test_route_scatter_rejects_wrong_layout_and_foreign_state():
    prepared = _prepared()
    state = prepared.build(jnp.asarray([[0.13], [0.52], [0.88]]))
    with pytest.raises(ValueError, match="Route payload must begin"):
        prepared.scatter_route_payload(state, jnp.ones((3, 2)))

    foreign = _prepared(particle_ids=(1, 2, 3))
    with pytest.raises(ValueError, match="different prepared transfer"):
        foreign.scatter_route_payload(
            state,
            jnp.ones((3, prepared.route_width, 1)),
        )
