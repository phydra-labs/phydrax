#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_metric_cochain_assembly_uses_paired_primal_and_dual_measures():
    chart = phx.metrix.CoordinateChart("line", ("x",))
    complex = phx.graph.cochain_complex_from_incidences(
        (2, 1),
        (jnp.asarray([[-1.0], [1.0]]),),
        (jnp.ones((2,)), jnp.ones((1,))),
        coordinates=(jnp.asarray([[0.0], [1.0]]), jnp.asarray([[0.5]])),
    )
    vertices = phx.graph.OrientedCellParameterization(
        0,
        2,
        1,
        lambda cell, reference: jnp.asarray([cell], dtype=float),
        lambda cell, reference: jnp.zeros((1, 0)),
        jnp.zeros((1, 0)),
        jnp.ones((1,)),
        jnp.ones((2,)),
    )
    edge = phx.graph.OrientedCellParameterization(
        1,
        1,
        1,
        lambda cell, reference: jnp.asarray([reference[0]]),
        lambda cell, reference: jnp.ones((1, 1)),
        jnp.asarray([[0.5]]),
        jnp.ones((1,)),
        jnp.ones((1,)),
    )
    bridge = phx.graph.ContinuousCochainBridge(complex, chart, (vertices, edge))

    vertex_duals = phx.graph.OrientedCellParameterization(
        1,
        2,
        1,
        lambda cell, reference: jnp.asarray([0.5 * (cell + reference[0])]),
        lambda cell, reference: jnp.asarray([[0.5]]),
        jnp.asarray([[0.5]]),
        jnp.ones((1,)),
        jnp.ones((2,)),
    )
    edge_dual = phx.graph.OrientedCellParameterization(
        0,
        1,
        1,
        lambda cell, reference: jnp.asarray([0.5]),
        lambda cell, reference: jnp.zeros((1, 0)),
        jnp.zeros((1, 0)),
        jnp.ones((1,)),
        jnp.ones((1,)),
    )
    metric = phx.metrix.diagonal_metric(lambda q: jnp.asarray([4.0]), chart=chart)
    assembly = phx.graph.assemble_metric_cochain_complex(
        bridge,
        metric,
        (vertex_duals, edge_dual),
    )

    assert jnp.allclose(assembly.primal_measures[0], jnp.ones((2,)))
    assert jnp.allclose(assembly.primal_measures[1], jnp.asarray([2.0]))
    assert jnp.allclose(assembly.dual_measures[0], jnp.ones((2,)))
    assert jnp.allclose(assembly.dual_measures[1], jnp.ones((1,)))
    assert jnp.allclose(assembly.hodge_stars[0], jnp.ones((2,)))
    assert jnp.allclose(assembly.hodge_stars[1], jnp.asarray([0.5]))
