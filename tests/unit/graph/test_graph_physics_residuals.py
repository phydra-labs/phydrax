#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _line_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes=jnp.array([[0.0], [1.0], [3.0]]),
        edges=jnp.array([[2.0], [3.0]]),
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 2], dtype=jnp.int32),
        n_node=jnp.array([3], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _domain_and_node_batch():
    domain = phx.domain.GraphDomain(_line_graph())
    nodes = domain.component({"graph": phx.domain.Nodes()})
    batch = nodes.sample(phx.domain.PointSampling(3, layout=phx.domain.SampleLayout((("graph",),))))
    return domain, nodes, batch


def test_graph_poisson_residual_composes_laplacian_and_source():
    domain, _nodes, batch = _domain_and_node_batch()

    @domain.Function("graph")
    def u(node):
        return node[0]

    @domain.Function("graph")
    def source(node):
        return jnp.where(node[0] < 0.5, -2.0, jnp.where(node[0] < 2.0, -4.0, 6.0))

    @domain.Function("graph")
    def diffusivity(edge):
        return edge[0]

    residual = phx.operators.graph_poisson_residual(
        u,
        source=source,
        weight=diffusivity,
    )

    assert jnp.allclose(residual(batch).data, jnp.zeros((3,)))


def test_graph_conservation_residual_composes_divergence_and_source():
    domain, _nodes, batch = _domain_and_node_batch()

    @domain.Function("graph")
    def flux(edge):
        return edge[0]

    @domain.Function("graph")
    def source(node):
        return jnp.where(node[0] < 0.5, -2.0, jnp.where(node[0] < 2.0, -1.0, 3.0))

    residual = phx.operators.graph_conservation_residual(flux, source=source)

    assert jnp.allclose(residual(batch).data, jnp.zeros((3,)))


def test_graph_advection_diffusion_residual_adds_advective_flux():
    domain, _nodes, batch = _domain_and_node_batch()

    @domain.Function("graph")
    def u(node):
        return node[0]

    @domain.Function("graph")
    def flux(edge):
        return edge[0]

    @domain.Function("graph")
    def source(node):
        return jnp.where(node[0] < 0.5, -3.0, jnp.where(node[0] < 2.0, -2.0, 5.0))

    residual = phx.operators.graph_advection_diffusion_residual(
        u,
        advective_flux=flux,
        source=source,
    )

    assert jnp.allclose(residual(batch).data, jnp.zeros((3,)))


def test_graph_heat_residual_zero_for_constant_implicit_step():
    domain, nodes, _batch = _domain_and_node_batch()
    structure = phx.domain.SampleLayout((("graph",),))

    @domain.Function("graph")
    def u_current(node):
        del node
        return 2.0

    @domain.Function("graph")
    def u_next(node):
        del node
        return 2.0

    def residual(next_fn, current_fn):
        return phx.operators.graph_heat_residual(next_fn, current_fn, dt=0.25)

    constraint = phx.constraints.FunctionalConstraint.from_operator(component=nodes,
    operator=residual,
    constraint_vars=("u_next", "u_current"), sampling=phx.domain.PointSampling(3, layout=structure), )

    assert constraint.loss({"u_next": u_next, "u_current": u_current}) < 1e-12


def test_graph_euler_residual_matches_explicit_rate():
    domain, _nodes, batch = _domain_and_node_batch()

    @domain.Function("graph")
    def u_current(node):
        return node[0]

    @domain.Function("graph")
    def rate(node):
        return node[0] + 1.0

    @domain.Function("graph")
    def u_next(node):
        return node[0] + 0.5 * (node[0] + 1.0)

    residual = phx.operators.graph_euler_residual(
        u_next,
        u_current,
        lambda _u: rate,
        dt=0.5,
    )

    assert jnp.allclose(residual(batch).data, jnp.zeros((3,)))
