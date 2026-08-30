import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from opt_einsum import contract

from phydrax.nn.operator.layers import O3TensorProduct, O3TensorProductPlan
from phydrax.nn.operator.representations import O3Representation


def _rotation(key):
    matrix = jr.normal(key, (3, 3), dtype=jnp.float64)
    orthogonal, triangular = jnp.linalg.qr(matrix)
    signs = jnp.where(jnp.diag(triangular) < 0.0, -1.0, 1.0)
    orthogonal = orthogonal * signs[None, :]
    return orthogonal.at[:, 0].multiply(jnp.linalg.det(orthogonal))


def test_known_scalar_and_vector_products_have_component_normalization():
    scalar = O3Representation(scalars=1)
    vector = O3Representation(vectors=1)
    scalar_product = O3TensorProduct(
        O3TensorProductPlan(scalar, scalar, scalar), internal_weights=False
    )
    np.testing.assert_allclose(
        scalar_product(jnp.asarray([3.0]), jnp.asarray([4.0]), jnp.asarray([2.0])),
        24.0,
    )

    scalar_vector = O3TensorProduct(
        O3TensorProductPlan(scalar, vector, vector), internal_weights=False
    )
    observed = scalar_vector(
        jnp.asarray([2.0]),
        jnp.asarray([1.0, -2.0, 3.0]),
        jnp.ones((1,)),
    )
    np.testing.assert_allclose(observed, [2.0, -4.0, 6.0])


def test_vector_vector_decomposes_into_known_zero_one_two_maps():
    vector = O3Representation(vectors=1)
    output = O3Representation(scalars=1, pseudovectors=1, tensors=1)
    plan = O3TensorProductPlan(vector, vector, output)
    product = O3TensorProduct(plan, internal_weights=False)
    left = jnp.asarray([0.3, -0.5, 0.8], dtype=jnp.float64)
    right = jnp.asarray([-0.2, 0.7, 0.4], dtype=jnp.float64)
    features = output.split(product(left, right, jnp.ones((3,), dtype=left.dtype)))
    expected_scalar = contract("i,i->", left, right) / jnp.sqrt(3.0)
    expected_axial = jnp.cross(left, right) / jnp.sqrt(2.0)
    outer = 0.5 * (
        contract("i,j->ij", left, right) + contract("i,j->ij", right, left)
    )
    expected_tensor = outer - jnp.trace(outer) * jnp.eye(3) / 3.0
    np.testing.assert_allclose(features.scalars[0], expected_scalar, rtol=1e-13)
    np.testing.assert_allclose(features.pseudovectors[0], expected_axial, rtol=1e-13)
    np.testing.assert_allclose(features.tensors[0], expected_tensor, rtol=1e-13)


def test_all_low_degree_paths_obey_inversion_parity_and_random_so3_equivariance():
    left_representation = O3Representation(vectors=1, pseudovectors=1)
    right_representation = O3Representation(vectors=1)
    output_representation = O3Representation(
        scalars=1,
        pseudoscalars=1,
        vectors=1,
        pseudovectors=1,
        tensors=1,
        pseudotensors=1,
    )
    plan = O3TensorProductPlan(
        left_representation, right_representation, output_representation
    )
    product = O3TensorProduct(plan, internal_weights=False, dtype=jnp.float64)
    left = jr.normal(jr.key(2), (left_representation.packed_size,), dtype=jnp.float64)
    right = jr.normal(jr.key(3), (right_representation.packed_size,), dtype=jnp.float64)
    weights = jr.normal(jr.key(4), (plan.parameter_count,), dtype=jnp.float64)
    reference = product(left, right, weights)

    inversion = -jnp.eye(3, dtype=jnp.float64)
    inverted = product(
        left_representation.transform(left, inversion),
        right_representation.transform(right, inversion),
        weights,
    )
    np.testing.assert_allclose(
        inverted,
        output_representation.transform(reference, inversion),
        rtol=2e-13,
        atol=2e-13,
    )

    rotation = _rotation(jr.key(5))
    rotated = product(
        left_representation.transform(left, rotation),
        right_representation.transform(right, rotation),
        weights,
    )
    np.testing.assert_allclose(
        rotated,
        output_representation.transform(reference, rotation),
        rtol=3e-13,
        atol=3e-13,
    )


def test_successive_rotations_batching_and_gradients_preserve_layout():
    representation = O3Representation(
        scalars=1, vectors=1, pseudovectors=1, tensors=1
    )
    edge = O3Representation(scalars=1, vectors=1, tensors=1)
    plan = O3TensorProductPlan(representation, edge, representation)
    product = O3TensorProduct(plan, internal_weights=False, dtype=jnp.float64)
    left = jr.normal(jr.key(10), (4, representation.packed_size), dtype=jnp.float64)
    right = jr.normal(jr.key(11), (4, edge.packed_size), dtype=jnp.float64)
    weights = jr.normal(jr.key(12), (4, plan.parameter_count), dtype=jnp.float64)
    observed = product(left, right, weights)
    assert observed.shape == (4, representation.packed_size)
    gradient = jax.grad(lambda value: jnp.sum(product(value, right, weights) ** 2))(
        left
    )
    assert gradient.shape == left.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))

    first = _rotation(jr.key(13))
    second = _rotation(jr.key(14))
    combined = second @ first
    direct = representation.transform(observed, combined)
    successive = representation.transform(
        representation.transform(observed, first), second
    )
    np.testing.assert_allclose(direct, successive, rtol=4e-13, atol=4e-13)


def test_plan_counts_identity_and_resource_rejection_are_resolved_before_layer():
    left = O3Representation(vectors=2)
    right = O3Representation(vectors=3)
    output = O3Representation(scalars=4, pseudovectors=5, tensors=6)
    plan = O3TensorProductPlan(left, right, output)
    assert plan.path_count == 3
    assert plan.parameter_count == 90
    assert O3TensorProduct(plan, key=jr.key(20)).weight.size == plan.parameter_count
    assert plan.resource_evidence["parameter_count"] == 90
    assert plan.content_id == plan.plan_id
    assert plan.plan_id == O3TensorProductPlan(left, right, output).plan_id
    with pytest.raises(ValueError, match="parameters"):
        O3TensorProductPlan(left, right, output, maximum_parameters=89)
    with pytest.raises(ValueError, match="no legal"):
        O3TensorProductPlan(
            O3Representation(scalars=1),
            O3Representation(scalars=1),
            O3Representation(vectors=1),
        )
    with pytest.raises(ValueError, match="only 'uvw'"):
        O3TensorProductPlan(left, right, output, connection_mode="uvu")


def test_product_rejects_mismatched_layouts_and_path_weight_axes():
    scalar = O3Representation(scalars=1)
    plan = O3TensorProductPlan(scalar, scalar, scalar)
    product = O3TensorProduct(plan, internal_weights=False)
    with pytest.raises(ValueError, match="Left values"):
        product(jnp.ones((2,)), jnp.ones((1,)), jnp.ones((1,)))
    with pytest.raises(ValueError, match="leading axes"):
        product(jnp.ones((2, 1)), jnp.ones((3, 1)), jnp.ones((1,)))
    with pytest.raises(ValueError, match="path weights"):
        product(jnp.ones((2, 1)), jnp.ones((2, 1)), jnp.ones((3, 1)))
