import io

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
import pytest

from phydrax.nn.models import (
    DeformationGradientMinors,
    InputConvexNetwork,
    PolyconvexPotential,
)


@pytest.mark.parametrize("dimension", (2, 3))
def test_deformation_gradient_minors_match_determinant_derivatives(dimension):
    minors = DeformationGradientMinors(dimension)
    gradient = jnp.eye(dimension) + 0.2 * jr.normal(
        jr.key(dimension), (dimension, dimension), dtype=jnp.float64
    )
    expected_cofactor = jax.grad(jnp.linalg.det)(gradient)

    assert jnp.allclose(minors.determinant(gradient), jnp.linalg.det(gradient))
    assert jnp.allclose(minors.cofactor(gradient), expected_cofactor)
    assert minors(gradient).shape == (2 * dimension**2 + 1,)


@pytest.mark.parametrize("dimension", (2, 3))
def test_polynomial_minors_and_constitutive_derivatives_remain_finite_at_singularity(
    dimension,
):
    gradient = jnp.eye(dimension).at[-1, -1].set(0.0)
    model = PolyconvexPotential(
        dimension,
        width_size=8,
        depth=2,
        key=jr.key(10 + dimension),
    )

    energy = model(gradient)
    stress = model.first_piola_stress(gradient)
    tangent = model.material_tangent(gradient)
    assert energy.shape == ()
    assert stress.shape == (dimension, dimension)
    assert tangent.shape == (dimension, dimension, dimension, dimension)
    assert jnp.isfinite(energy)
    assert jnp.all(jnp.isfinite(stress))
    assert jnp.all(jnp.isfinite(tangent))


def test_polyconvex_outer_potential_has_positive_semidefinite_lifted_hessian():
    model = PolyconvexPotential(
        3,
        width_size=12,
        depth=3,
        key=jr.key(20),
    )
    lifted = model.minors(
        jnp.eye(3) + 0.1 * jr.normal(jr.key(21), (3, 3), dtype=jnp.float64)
    )
    hessian = jax.hessian(model.lifted_energy)(lifted)
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (hessian + hessian.T))
    assert eigenvalues.min() >= -1e-9


def test_polyconvex_potential_requires_a_proven_convex_outer_network():
    wrong_size = InputConvexNetwork(in_size=8, width_size=4, depth=1, key=jr.key(22))
    with pytest.raises(ValueError, match="input size must be 9"):
        PolyconvexPotential(2, potential=wrong_size)
    with pytest.raises(TypeError, match="InputConvexNetwork"):
        PolyconvexPotential(2, potential=lambda value: jnp.sum(value))


def test_material_tangent_matches_directional_stress_derivative():
    model = PolyconvexPotential(
        3,
        width_size=10,
        depth=2,
        key=jr.key(23),
    )
    gradient = jnp.eye(3) + 0.1 * jr.normal(jr.key(24), (3, 3))
    direction = jr.normal(jr.key(25), (3, 3))
    tangent = model.material_tangent(gradient)
    expected = oe.contract("ijkl,kl->ij", tangent, direction)
    _, derivative = jax.jvp(model.first_piola_stress, (gradient,), (direction,))
    assert jnp.allclose(derivative, expected, atol=2e-8, rtol=2e-8)


def test_polyconvex_model_is_batched_jittable_differentiable_and_serializable():
    model = PolyconvexPotential(
        2,
        width_size=8,
        depth=2,
        key=jr.key(26),
    )
    gradients = jnp.stack(
        (
            jnp.eye(2),
            jnp.array([[1.1, 0.2], [-0.1, 0.8]]),
            jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        )
    )
    energy = eqx.filter_jit(lambda current: current(gradients))(model)
    stress = eqx.filter_jit(lambda current: current.first_piola_stress(gradients))(model)
    tangent = eqx.filter_jit(lambda current: current.material_tangent(gradients))(model)
    assert energy.shape == (3,)
    assert stress.shape == (3, 2, 2)
    assert tangent.shape == (3, 2, 2, 2, 2)
    assert jnp.all(jnp.isfinite(jax.grad(lambda value: jnp.sum(model(value)))(gradients)))

    buffer = io.BytesIO()
    eqx.tree_serialise_leaves(buffer, model)
    buffer.seek(0)
    restored = eqx.tree_deserialise_leaves(buffer, model)
    assert jnp.array_equal(restored(gradients), energy)
