#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp

import phydrax as phx


def _splat():
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.27, 0.31], [0.43, 0.38]])
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(2), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    prepared = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    return prepared, position


def test_route_superset_jvp_vjp_and_topology_guard():
    prepared, position = _splat()
    deformation = jnp.broadcast_to(jnp.eye(2), (2, 2, 2))
    plan = phx.solver.MPMRouteSupersetPlan(prepared, minimum_margin=1e-10)
    state = plan.build(position)
    cotangent = (
        jnp.ones_like(state.base_routes.stencil.weights),
        jnp.ones_like(state.base_routes.weight_gradients),
        jnp.ones_like(state.base_routes.route_offsets),
    )
    result = plan.linearize(
        state,
        position,
        deformation,
        None,
        jnp.full_like(position, 1e-3),
        jnp.zeros_like(deformation),
        None,
        cotangent,
    )

    assert bool(result.route_topology_stable)
    assert bool(result.successful)
    assert jnp.all(jnp.isfinite(result.weight_jvp))
    assert jnp.all(jnp.isfinite(result.position_transpose))


def test_compact_residual_jvp_transpose_match_dense_operator():
    prepared, position = _splat()
    routes = prepared.build(position)
    blocks = phx.discretization.MPMActiveBlockPlan((8, 8), (4, 4), 4).build(routes)
    storage = phx.discretization.BlockSparseMPMNodalStoragePlan(
        phx.discretization.MPMActiveBlockPlan((8, 8), (4, 4), 4)
    )
    operator = phx.solver.MPMCompactImplicitOperator(storage, blocks)
    dense = jnp.arange(8 * 8.0).reshape((8, 8))
    compact = storage.pack(dense, blocks)
    direction = jnp.ones_like(compact)
    cotangent = 0.5 * jnp.ones_like(compact)
    result = operator.apply(
        lambda value: 2.0 * value + jnp.roll(value, 1, axis=0),
        compact,
        direction,
        cotangent,
    )

    assert bool(result.successful)
    assert result.dense_compact_residual_defect < 1e-10
    assert result.dense_compact_jvp_defect < 1e-10
    assert result.dense_compact_transpose_defect < 1e-10


def test_implicit_unknown_layout_and_contact_generalized_actions():
    free = jnp.ones((2, 1, 2), dtype=bool)
    essential = jnp.zeros_like(free).at[0, 0, 1].set(True)
    free = free & ~essential
    layout = phx.solver.MPMImplicitUnknownLayout(
        free,
        essential,
        contact_multiplier_capacity=1,
        rigid_dof_capacity=3,
    )
    packed = layout.pack(jnp.zeros((2, 1, 2)))
    velocity, multipliers, rigid = layout.unpack(packed)
    assert velocity.shape == (2, 1, 2)
    assert multipliers.shape == (1,)
    assert rigid.shape == (3,)

    contact = phx.discretization.KWayMPMContactPlan(
        2,
        friction=phx.discretization.SmoothCoulombMPMFrictionPlan(
            0.1, regularization=1e-3
        ),
        smoothing=1e-3,
        maximum_steps=100,
        tolerance=1e-8,
    )
    mass = jnp.asarray([[1.0], [1.0]])
    velocity = jnp.asarray([[[0.5, 0.0]], [[-0.5, 0.0]]])
    gradients = jnp.asarray([[[1.0, 0.0]], [[-1.0, 0.0]]])
    graph = contact.build_graph(mass, gradients)
    linearized = phx.solver.linearize_kway_contact(
        contact,
        mass,
        velocity,
        graph,
        0.01,
        jnp.ones_like(velocity),
        jnp.ones_like(velocity),
    )
    assert bool(linearized.successful)
    assert jnp.all(jnp.isfinite(linearized.jvp))
    assert jnp.all(jnp.isfinite(linearized.transpose))
