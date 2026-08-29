#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax as phx


def _prepared(*, geometry_ad="piecewise", accumulation="deterministic"):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(5),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([4, 1]),
        jnp.ones((2,)),
        ambient_dimension=1,
    ).prepare()
    execution = phx.discretization.SplatExecutionPolicy(
        geometry_ad=geometry_ad, accumulation=accumulation
    )
    return phx.discretization.ParticleGridSplatPlan(grid, execution=execution).prepare(
        particles
    )


def test_payload_jvp_vjp_and_gather_are_the_same_linear_pair():
    prepared = _prepared()
    position = jnp.asarray([[0.2], [0.7]])
    state = prepared.build(position)
    payload = jnp.asarray([2.0, -1.0])
    tangent = jnp.asarray([0.4, 0.7])

    def deposit(value):
        return prepared.deposit_content(state, value).content

    _, jvp = jax.jvp(deposit, (payload,), (tangent,))
    expected_jvp = deposit(tangent)
    cotangent = jnp.asarray([0.3, -0.5, 0.9, 0.2, -0.1])
    _, pullback = jax.vjp(deposit, payload)
    payload_cotangent = pullback(cotangent)[0]
    gathered = prepared.gather(state, cotangent)

    assert jnp.allclose(jvp, expected_jvp)
    assert jnp.allclose(payload_cotangent, gathered.values)
    assert jnp.allclose(
        jnp.vdot(deposit(payload), cotangent),
        jnp.vdot(payload, payload_cotangent),
    )


def test_complex_adjoint_identity_in_every_accumulation_mode():
    position = jnp.asarray([[0.2], [0.7]])
    payload = jnp.asarray([2.0 + 0.5j, -1.0 + 3.0j])
    cotangent = jnp.asarray([0.3 - 0.1j, -0.5 + 0.7j, 0.9, 0.2 - 0.8j, -0.1 + 0.4j])

    for accumulation in ("fast", "deterministic", "compensated"):
        prepared = _prepared(accumulation=accumulation)
        state = prepared.build(position)

        def deposit(value):
            return prepared.deposit_content(state, value).content

        output, pullback = jax.vjp(deposit, payload)
        payload_cotangent = pullback(cotangent)[0]
        gathered = prepared.gather(state, cotangent)

        assert jnp.allclose(payload_cotangent, gathered.values)
        assert jnp.allclose(
            jnp.vdot(output, cotangent), jnp.vdot(payload, payload_cotangent)
        )


def test_position_jvp_and_vjp_match_finite_difference_inside_cells():
    prepared = _prepared()
    position = jnp.asarray([[0.2], [0.7]])
    direction = jnp.asarray([[0.3], [-0.2]])
    content = jnp.asarray([2.0, 4.0])
    weight = jnp.asarray([0.2, -0.7, 0.4, 0.8, -0.1])

    def output(value):
        state = prepared.build(value)
        return prepared.deposit_content(state, content).content

    def loss(value):
        return jnp.vdot(output(value), weight)

    _, jvp = jax.jvp(output, (position,), (direction,))
    analytic = jnp.vdot(jvp, weight)
    gradient = jax.grad(loss)(position)
    eps = 1e-5
    finite = (loss(position + eps * direction) - loss(position - eps * direction)) / (
        2.0 * eps
    )

    assert jnp.allclose(jnp.vdot(gradient, direction), analytic)
    assert jnp.allclose(analytic, finite, rtol=1e-8, atol=1e-9)


def test_reconstruction_derivative_matches_finite_difference():
    prepared = _prepared()
    position = jnp.asarray([[0.2], [0.7]])
    samples = jnp.asarray([1.0, 3.0])
    weights = jnp.asarray([2.0, 1.0])

    def loss(value):
        state = prepared.build(value)
        result = prepared.reconstruct(state, samples, weights)
        return jnp.sum(result.values**2)

    direction = jnp.asarray([[0.1], [-0.15]])
    gradient = jax.grad(loss)(position)
    eps = 1e-5
    finite = (loss(position + eps * direction) - loss(position - eps * direction)) / (
        2.0 * eps
    )

    assert jnp.allclose(jnp.vdot(gradient, direction), finite, rtol=1e-7, atol=1e-8)


def test_jit_vmap_and_scan_preserve_transfer_results():
    prepared = _prepared()
    content = jnp.asarray([2.0, 4.0])
    positions = jnp.asarray(
        [
            [[0.2], [0.7]],
            [[0.25], [0.75]],
            [[0.3], [0.8]],
        ]
    )

    @jax.jit
    def apply(position):
        return prepared.deposit_content(prepared.build(position), content).content

    vmapped = jax.vmap(apply)(positions)

    def step(_, position):
        value = apply(position)
        return None, value

    _, scanned = jax.lax.scan(step, None, positions)
    sequential = jnp.stack([apply(position) for position in positions])

    assert jnp.allclose(vmapped, sequential)
    assert jnp.allclose(scanned, sequential)


def test_frozen_geometry_is_zero_under_jit_vmap_and_scan():
    prepared = _prepared(geometry_ad="frozen")
    content = jnp.asarray([2.0, 4.0])
    positions = jnp.asarray([[[0.2], [0.7]], [[0.25], [0.75]], [[0.3], [0.8]]])

    def loss(position):
        value = prepared.deposit_content(prepared.build(position), content).content
        return jnp.sum(value**2)

    gradients = jax.jit(jax.vmap(jax.grad(loss)))(positions)
    _, scanned = jax.lax.scan(
        lambda _, position: (None, jax.grad(loss)(position)), None, positions
    )

    assert jnp.all(gradients == 0.0)
    assert jnp.all(scanned == 0.0)


def test_periodic_total_content_has_zero_position_gradient():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([2, 1]), jnp.ones((2,)), ambient_dimension=1
    ).prepare()
    prepared = phx.discretization.ParticleGridSplatPlan(grid).prepare(particles)
    content = jnp.asarray([2.0, -3.0])

    def total(position):
        return jnp.sum(
            prepared.deposit_content(prepared.build(position), content).content
        )

    gradient = jax.grad(total)(jnp.asarray([[0.01], [0.99]]))
    assert jnp.allclose(gradient, 0.0, atol=1e-12)


def test_exact_grid_node_reports_zero_route_weight_margin():
    prepared = _prepared()
    state = prepared.build(jnp.asarray([[0.25], [0.75]]))

    assert state.minimum_route_weight == 0.0
    assert jnp.allclose(state.partition_sums, 1.0)
