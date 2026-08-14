#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _flux_graph() -> phx.graph.GraphIR:
    return phx.graph.GraphIR(
        nodes={"area": jnp.array([2.0, 1.0])},
        edges={"flux": jnp.array([6.0, -2.0])},
        senders=jnp.array([0, 1], dtype=jnp.int32),
        receivers=jnp.array([1, 0], dtype=jnp.int32),
        n_node=jnp.array([2], dtype=jnp.int32),
        n_edge=jnp.array([2], dtype=jnp.int32),
    )


def _square_dual() -> phx.graph.MeshDualGraph:
    vertices = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    faces = jnp.array([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    return phx.graph.mesh_to_dual_graph(vertices, faces)


def test_finite_volume_divergence_conserves_internal_flux_without_volume_normalization():
    out = phx.graph.GraphFiniteVolumeDivergence(
        output_key="div",
        normalize_by_volume=False,
    )(_flux_graph())

    assert jnp.allclose(out.nodes["div"], jnp.array([-8.0, 8.0]))
    assert jnp.allclose(jnp.sum(out.nodes["div"]), 0.0)


def test_finite_volume_divergence_normalizes_by_cell_volume():
    out = phx.graph.GraphFiniteVolumeDivergence(output_key="div")(_flux_graph())

    assert jnp.allclose(out.nodes["div"], jnp.array([-4.0, 8.0]))


def test_finite_volume_diffusion_is_zero_for_constant_dual_cell_field():
    dual = _square_dual()
    graph = dual.graph.replace(
        nodes={**dual.graph.nodes, "u": jnp.ones((2,))},
        validate=False,
    )

    out = phx.graph.GraphFiniteVolumeDiffusion(input_key="u", output_key="du")(graph)

    assert jnp.allclose(out.nodes["du"], jnp.zeros((2, 1)))


def test_finite_volume_diffusion_is_conservative_on_dual_graph():
    dual = _square_dual()
    graph = dual.graph.replace(
        nodes={**dual.graph.nodes, "u": jnp.array([1.0, 3.0])},
        validate=False,
    )

    out = phx.graph.GraphFiniteVolumeDiffusion(input_key="u", output_key="du")(graph)
    weighted_total = jnp.sum(out.nodes["du"][:, 0] * graph.nodes["area"][:, 0])

    assert out.nodes["du"].shape == (2, 1)
    assert jnp.allclose(weighted_total, 0.0, atol=1e-6)
    assert out.nodes["du"][0, 0] > 0.0
    assert out.nodes["du"][1, 0] < 0.0


def test_finite_volume_diffusion_wraps_as_graph_model_on_dual_graph():
    dual = _square_dual()
    domain = phx.domain.GraphDomain(dual.graph)
    faces = domain.component({"graph": dual.face_nodes_component()})
    batch = faces.sample(
        phx.domain.PointSampling(2, layout=phx.domain.SampleLayout((("graph",),)))
    )
    values = jnp.array([1.0, 3.0])

    @domain.Function("graph")
    def u(face):
        return values[face.get("face_index")]

    model = domain.GraphModel(
        phx.graph.GraphFiniteVolumeDiffusion(input_key="u", output_key="du"),
        input_fn=u,
        input_key="u",
        output_key="du",
    )

    out = model(batch).data

    assert out.shape == (2, 1)
    assert out[0, 0] > 0.0
    assert out[1, 0] < 0.0
