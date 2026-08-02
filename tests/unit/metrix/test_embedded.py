import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _sphere(radius=1.0):
    chart = phx.metrix.CoordinateChart("sphere", ("theta", "phi"))

    def embedding(q):
        theta, phi = q
        return radius * jnp.array(
            [
                jnp.sin(theta) * jnp.cos(phi),
                jnp.sin(theta) * jnp.sin(phi),
                jnp.cos(theta),
            ]
        )

    return phx.metrix.EmbeddedChart(
        chart,
        embedding,
        3,
        retraction=lambda x: radius * x / jnp.linalg.norm(x),
    )


def test_plane_embedding_has_identity_metric_and_zero_extrinsic_curvature():
    chart = phx.metrix.CoordinateChart("plane", ("u", "v"))
    embedded = phx.metrix.EmbeddedChart(
        chart,
        lambda q: jnp.array([q[0], q[1], 0.0]),
        3,
    )
    points = jnp.array([[0.2, -0.4], [1.0, 2.0]])

    metric = embedded.induced_metric()
    tangent = embedded.tangent_projector(points)
    normal = embedded.normal_projector(points)

    assert jnp.allclose(metric(points), jnp.broadcast_to(jnp.eye(2), (2, 2, 2)))
    assert jnp.allclose(embedded.volume_density(points), 1.0)
    assert jnp.allclose(tangent + normal, jnp.eye(3))
    assert jnp.allclose(tangent @ tangent, tangent)
    assert jnp.allclose(normal @ normal, normal)
    assert jnp.allclose(embedded.second_fundamental_form(points), 0.0)
    assert jnp.allclose(embedded.mean_curvature_vector(points), 0.0)
    assert jnp.allclose(phx.metrix.scalar_curvature(metric, points), 0.0)


def test_sphere_induced_and_extrinsic_geometry_agree():
    radius = 2.0
    embedded = _sphere(radius)
    point = jnp.array([1.1, 0.4])
    ambient = embedded(point)
    metric = embedded.induced_metric()
    expected_metric = radius**2 * jnp.diag(jnp.array([1.0, jnp.sin(point[0]) ** 2]))

    assert jnp.allclose(metric(point), expected_metric, atol=1e-9)
    assert jnp.allclose(embedded.volume_density(point), radius**2 * jnp.sin(point[0]))
    assert jnp.allclose(phx.metrix.scalar_curvature(metric, point), 2.0 / radius**2)

    tangent = embedded.tangent_projector(point)
    normal = embedded.normal_projector(point)
    unit_normal = ambient / radius
    assert jnp.allclose(tangent @ unit_normal, 0.0, atol=1e-9)
    assert jnp.allclose(normal @ unit_normal, unit_normal, atol=1e-9)
    assert jnp.allclose(tangent + normal, jnp.eye(3), atol=1e-9)

    mean_curvature = embedded.mean_curvature_vector(point)
    shape = embedded.shape_operator(unit_normal, point)
    assert jnp.allclose(jnp.linalg.norm(mean_curvature), 1.0 / radius, atol=1e-9)
    assert jnp.allclose(shape, -jnp.eye(2) / radius, atol=1e-9)

    moved = ambient + jnp.array([0.4, -0.2, 0.1])
    retracted = embedded.retract(moved)
    assert jnp.allclose(jnp.linalg.norm(retracted), radius)
    assert jnp.allclose(jax.jit(embedded.tangent_projector)(point), tangent)


def test_cylinder_intrinsic_flatness_and_principal_curvatures():
    radius = 1.7
    chart = phx.metrix.CoordinateChart("cylinder", ("theta", "z"))

    def embedding(q):
        return jnp.array([radius * jnp.cos(q[0]), radius * jnp.sin(q[0]), q[1]])

    embedded = phx.metrix.EmbeddedChart(chart, embedding, 3)
    point = jnp.array([0.6, -0.2])
    unit_normal = jnp.array([jnp.cos(point[0]), jnp.sin(point[0]), 0.0])
    shape = embedded.shape_operator(unit_normal, point)

    assert jnp.allclose(
        embedded.induced_metric()(point),
        jnp.diag(jnp.array([radius**2, 1.0])),
        atol=1e-9,
    )
    assert jnp.allclose(
        phx.metrix.scalar_curvature(embedded.induced_metric(), point),
        0.0,
        atol=1e-9,
    )
    assert jnp.allclose(jnp.linalg.eigvals(shape), jnp.array([-1.0 / radius, 0.0]))


def test_normal_projector_helper_is_batched_and_rejects_zero_normals():
    normals = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    projector = phx.metrix.tangent_projector_from_normal(normals)

    assert projector.shape == (2, 3, 3)
    assert jnp.allclose(projector @ normals[..., None], 0.0)
    assert jnp.allclose(projector @ projector, projector)
    assert jnp.allclose(
        jax.jit(phx.metrix.tangent_projector_from_normal)(normals),
        projector,
    )

    with pytest.raises(eqx.EquinoxRuntimeError, match="nonzero"):
        result = phx.metrix.tangent_projector_from_normal(jnp.zeros(3))
        jax.block_until_ready(result)
