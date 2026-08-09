#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _orthonormal(rows, columns):
    matrix = jnp.arange(1, rows * columns + 1, dtype=float).reshape(rows, columns)
    orthogonal, _ = jnp.linalg.qr(matrix)
    return orthogonal


def _manifold_cases():
    stiefel = _orthonormal(4, 2)
    return (
        (
            phx.metrix.EuclideanManifold((3,)),
            jnp.array([1.0, -2.0, 0.5]),
            jnp.array([0.3, -0.4, 0.2]),
        ),
        (
            phx.metrix.SphereManifold(3),
            jnp.array([1.0, 2.0, 3.0]) / jnp.sqrt(14.0),
            jnp.array([0.3, -0.4, 0.2]),
        ),
        (
            phx.metrix.StiefelManifold(4, 2),
            stiefel,
            jnp.arange(8.0).reshape(4, 2) / 10.0,
        ),
        (
            phx.metrix.GrassmannManifold(4, 2),
            stiefel,
            jnp.arange(8.0).reshape(4, 2) / 10.0,
        ),
        (
            phx.metrix.SpecialOrthogonalManifold(3),
            jnp.eye(3),
            jnp.array([[0.2, -0.1, 0.3], [0.4, -0.2, 0.1], [-0.3, 0.2, 0.5]]),
        ),
        (
            phx.metrix.AffineInvariantSPDManifold(2),
            jnp.array([[2.0, 0.25], [0.25, 1.25]]),
            jnp.array([[0.2, -0.1], [0.3, 0.4]]),
        ),
    )


def test_manifold_contract_is_public_and_abstract():
    with pytest.raises(TypeError):
        phx.metrix.AbstractRiemannianManifold()

    for symbol in (
        "AbstractRiemannianManifold",
        "AffineInvariantSPDManifold",
        "EuclideanManifold",
        "GrassmannManifold",
        "SpecialOrthogonalManifold",
        "SphereManifold",
        "StiefelManifold",
    ):
        assert symbol in phx.metrix.__all__
        assert getattr(phx.metrix, symbol) is not None


@pytest.mark.parametrize("manifold,point,ambient", _manifold_cases())
def test_manifold_metric_retraction_and_transport_laws(manifold, point, ambient):
    tangent = manifold.project_tangent(point, ambient)
    cotangent = 0.7 - 0.3 * ambient
    rgradient = manifold.egrad_to_rgrad(point, cotangent)

    metric_pairing = manifold.inner(point, rgradient, tangent)
    ambient_pairing = jnp.real(jnp.vdot(cotangent, tangent))
    assert jnp.allclose(metric_pairing, ambient_pairing, rtol=2e-9, atol=2e-9)

    projected_twice = manifold.project_tangent(point, tangent)
    assert jnp.allclose(projected_twice, tangent, rtol=2e-9, atol=2e-9)

    step = 0.03 * tangent
    destination = jax.jit(manifold.retract)(point, step)
    assert bool(manifold.contains(destination))
    assert jnp.asarray(manifold.constraint_residual(destination)).shape == ()
    assert jnp.asarray(manifold.norm(point, tangent)).shape == ()

    _, derivative = jax.jvp(
        lambda value: manifold.retract(point, value),
        (jnp.zeros_like(tangent),),
        (tangent,),
    )
    assert jnp.allclose(derivative, tangent, rtol=2e-8, atol=2e-8)

    transported = jax.jit(manifold.transport)(point, step, destination, tangent)
    target_projection = manifold.project_tangent(destination, transported)
    assert jnp.allclose(target_projection, transported, rtol=2e-8, atol=2e-8)
    zero_transport = manifold.transport(
        point,
        jnp.zeros_like(tangent),
        point,
        tangent,
    )
    assert jnp.allclose(zero_transport, tangent, rtol=2e-9, atol=2e-9)


@pytest.mark.parametrize("method", ["exponential", "cayley"])
def test_so_manifold_delegates_existing_state_retraction(method):
    manifold = phx.metrix.SpecialOrthogonalManifold(3, retraction=method)
    state_geometry = phx.metrix.SpecialOrthogonalStateGeometry(3, retraction=method)
    point = jnp.eye(3)
    ambient = jnp.array([[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]])
    tangent = manifold.project_tangent(point, ambient)
    local = state_geometry.to_local(point, tangent)

    assert jnp.allclose(
        manifold.retract(point, tangent),
        state_geometry.retract(point, local),
    )
    assert manifold.transport_method == "tangent-projection"
    assert not manifold.transport_is_parallel


def test_spd_metric_gradient_and_transport_are_affine_invariant():
    manifold = phx.metrix.AffineInvariantSPDManifold(3)
    point = jnp.array([[2.0, 0.2, -0.1], [0.2, 1.4, 0.15], [-0.1, 0.15, 1.1]])
    left = manifold.project_tangent(
        point,
        jnp.array([[0.4, -0.2, 0.1], [0.1, 0.3, -0.1], [0.2, 0.05, -0.2]]),
    )
    right = manifold.project_tangent(
        point,
        jnp.array([[-0.1, 0.2, 0.3], [0.0, 0.1, -0.2], [0.4, -0.1, 0.2]]),
    )
    step = 0.08 * left
    destination = manifold.retract(point, step)
    transported_left = manifold.transport(point, step, destination, left)
    transported_right = manifold.transport(point, step, destination, right)

    assert manifold.transport_is_parallel
    assert manifold.transport_is_isometric
    assert jnp.allclose(
        manifold.inner(destination, transported_left, transported_right),
        manifold.inner(point, left, right),
        rtol=2e-9,
        atol=2e-9,
    )
    assert jnp.all(jnp.linalg.eigvalsh(destination) > 0.0)


def test_grassmann_operations_are_invariant_under_basis_change():
    manifold = phx.metrix.GrassmannManifold(5, 2)
    point = _orthonormal(5, 2)
    rotation = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    ambient = jnp.arange(10.0).reshape(5, 2) / 7.0
    tangent = manifold.project_tangent(point, ambient)
    transformed_tangent = manifold.project_tangent(point @ rotation, ambient @ rotation)

    assert jnp.allclose(transformed_tangent, tangent @ rotation)
    destination = manifold.retract(point, 0.05 * tangent)
    transformed_destination = manifold.retract(
        point @ rotation,
        0.05 * transformed_tangent,
    )
    assert jnp.allclose(
        destination @ destination.T,
        transformed_destination @ transformed_destination.T,
        atol=2e-9,
    )


def test_manifolds_support_leading_product_axes():
    sphere = phx.metrix.SphereManifold(3)
    points = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    ambient = jnp.array([[0.0, 0.2, -0.1], [0.3, 0.0, 0.4]])
    tangent = sphere.project_tangent(points, ambient)
    destinations = sphere.retract(points, 0.1 * tangent)

    assert bool(sphere.contains(destinations))
    assert destinations.shape == points.shape
    assert jnp.asarray(sphere.inner(points, tangent, tangent)).shape == ()

    stiefel = phx.metrix.StiefelManifold(3, 2)
    matrices = jnp.broadcast_to(_orthonormal(3, 2), (4, 3, 2))
    projected = stiefel.project_tangent(matrices, jnp.ones_like(matrices))
    updated = stiefel.retract(matrices, 0.01 * projected)
    assert bool(stiefel.contains(updated))
    assert updated.shape == matrices.shape


def test_manifold_constructor_and_shape_failures_are_explicit():
    with pytest.raises(ValueError, match="at least two"):
        phx.metrix.SphereManifold(1)
    with pytest.raises(ValueError, match="must not exceed"):
        phx.metrix.StiefelManifold(2, 3)
    with pytest.raises(ValueError, match="strictly less"):
        phx.metrix.GrassmannManifold(2, 2)
    with pytest.raises(ValueError, match="trailing shape"):
        phx.metrix.SphereManifold(3).contains(jnp.ones((2,)))
