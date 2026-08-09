#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _mixed_parameters():
    return {
        "euclidean": jnp.array([2.0, -1.0]),
        "sphere": jnp.array([1.0, 0.0, 0.0]),
    }


def _mixed_geometry(parameters=None):
    parameters = _mixed_parameters() if parameters is None else parameters
    return phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['sphere']": phx.metrix.SphereManifold(3)},
    )


def test_parameter_geometry_paths_binding_and_public_exports():
    parameters = _mixed_parameters()
    paths = phx.optim.ParameterGeometry.array_leaf_paths(parameters)
    geometry = _mixed_geometry(parameters)

    assert paths == ("['euclidean']", "['sphere']")
    assert geometry.num_manifold_leaves == 1
    assert geometry.manifold_ids == ("manifold:sphere:3",)
    assert bool(geometry.contains(parameters))
    assert "ParameterGeometry" in phx.optim.__all__
    assert "riemannian_sgd" in phx.optim.__all__
    assert "riemannian_momentum" in phx.optim.__all__


def test_parameter_geometry_mixes_leaf_metrics_without_flattening():
    parameters = _mixed_parameters()
    geometry = _mixed_geometry(parameters)
    gradients = {
        "euclidean": jnp.array([1.0, 2.0]),
        "sphere": jnp.array([3.0, 4.0, 0.0]),
    }
    rgradient = geometry.egrad_to_rgrad(parameters, gradients)

    assert jnp.array_equal(rgradient["euclidean"], gradients["euclidean"])
    assert jnp.allclose(rgradient["sphere"], jnp.array([0.0, 4.0, 0.0]))
    assert jnp.allclose(geometry.norm(parameters, rgradient), jnp.sqrt(21.0))

    step = jax.tree.map(lambda leaf: -0.1 * leaf, rgradient)
    destination = geometry.retract(parameters, step)
    assert destination["euclidean"].shape == (2,)
    assert destination["sphere"].shape == (3,)
    assert jnp.allclose(jnp.linalg.norm(destination["sphere"]), 1.0)
    assert geometry.constraint_residuals(destination).keys() == {"['sphere']"}
    assert geometry.maximum_constraint_residual(destination) < 1e-12


def test_parameter_geometry_transport_preserves_mixed_tree_structure():
    parameters = _mixed_parameters()
    geometry = _mixed_geometry(parameters)
    tangent = {
        "euclidean": jnp.array([0.3, -0.2]),
        "sphere": jnp.array([0.0, 0.4, -0.1]),
    }
    step = jax.tree.map(lambda leaf: 0.1 * leaf, tangent)
    destination = geometry.retract(parameters, step)
    transported = jax.jit(geometry.transport)(
        parameters,
        step,
        destination,
        tangent,
    )

    assert jax.tree.structure(transported) == jax.tree.structure(parameters)
    assert jnp.array_equal(transported["euclidean"], tangent["euclidean"])
    assert jnp.allclose(
        jnp.vdot(destination["sphere"], transported["sphere"]),
        0.0,
        atol=1e-12,
    )


def test_parameter_geometry_supports_leading_manifold_product_axes():
    parameters = {
        "directions": jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['directions']": phx.metrix.SphereManifold(3)},
    )
    ambient = {"directions": jnp.ones((3, 3))}
    tangent = geometry.egrad_to_rgrad(parameters, ambient)
    destination = geometry.retract(
        parameters,
        jax.tree.map(lambda leaf: 0.05 * leaf, tangent),
    )

    assert bool(geometry.contains(destination))
    assert destination["directions"].shape == (3, 3)


def test_parameter_geometry_rejects_invalid_bindings_and_reuse():
    parameters = _mixed_parameters()

    with pytest.raises(ValueError, match="at least one manifold"):
        phx.optim.ParameterGeometry.from_leaf_paths(parameters, {})
    with pytest.raises(ValueError, match="Unknown.*missing"):
        phx.optim.ParameterGeometry.from_leaf_paths(
            parameters,
            {"['missing']": phx.metrix.SphereManifold(3)},
        )
    with pytest.raises(ValueError, match="outside"):
        phx.optim.ParameterGeometry.from_leaf_paths(
            {"sphere": jnp.ones((3,))},
            {"['sphere']": phx.metrix.SphereManifold(3)},
        )
    with pytest.raises(ValueError, match="trailing shape"):
        phx.optim.ParameterGeometry.from_leaf_paths(
            {"sphere": jnp.array([1.0, 0.0])},
            {"['sphere']": phx.metrix.SphereManifold(3)},
        )
    with pytest.raises(TypeError, match="real floating-point"):
        phx.optim.ParameterGeometry.from_leaf_paths(
            {"sphere": jnp.array([1.0 + 0.0j, 0.0j, 0.0j])},
            {"['sphere']": phx.metrix.SphereManifold(3)},
        )

    geometry = _mixed_geometry(parameters)
    with pytest.raises(ValueError, match="PyTree structure"):
        geometry.validate({"different": jnp.ones((3,))})
    with pytest.raises(ValueError, match="must have shape"):
        geometry.validate(
            {"euclidean": jnp.ones((3,)), "sphere": jnp.array([1.0, 0.0, 0.0])}
        )


def test_unselected_complex_leaf_uses_real_hermitian_metric():
    parameters = {
        "complex": jnp.array([1.0 + 2.0j, -0.5j]),
        "sphere": jnp.array([1.0, 0.0, 0.0]),
    }
    geometry = phx.optim.ParameterGeometry.from_leaf_paths(
        parameters,
        {"['sphere']": phx.metrix.SphereManifold(3)},
    )
    tangent = {
        "complex": jnp.array([1.0j, 2.0 - 0.5j]),
        "sphere": jnp.array([0.0, 1.0, 0.0]),
    }
    expected = jnp.real(jnp.vdot(tangent["complex"], tangent["complex"])) + 1.0
    assert jnp.allclose(geometry.inner(parameters, tangent, tangent), expected)
