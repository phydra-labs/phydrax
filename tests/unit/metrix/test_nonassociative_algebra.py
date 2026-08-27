from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _exact_basis(dimension, position):
    return tuple(Fraction(int(index == position)) for index in range(dimension))


def test_exact_derived_products_distinguish_quaternion_and_octonion_bracketing():
    quaternion = phx.metrix.algebra.QuaternionAlgebraSpec()
    octonion = phx.metrix.algebra.OctonionAlgebraSpec()
    quaternion_basis = tuple(_exact_basis(4, index) for index in range(4))
    witness = octonion.properties.claim("associative").witness
    octonion_basis = tuple(_exact_basis(8, index) for index in range(8))
    positions = tuple(octonion.basis_index(label) for label in witness)

    assert (
        quaternion.associator_exact(
            quaternion_basis[1], quaternion_basis[2], quaternion_basis[3]
        )
        == (Fraction(0),) * 4
    )
    assert quaternion.commutator_exact(quaternion_basis[1], quaternion_basis[2]) == tuple(
        2 * value for value in quaternion_basis[3]
    )
    assert (
        octonion.associator_exact(*(octonion_basis[position] for position in positions))
        != (Fraction(0),) * 8
    )


def test_octonion_associator_is_alternating_for_exact_linear_combinations():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    left = tuple(Fraction(value) for value in (1, 2, -1, 3, 0, 1, -2, 4))
    right = tuple(Fraction(value) for value in (0, -1, 2, 1, 3, -2, 1, 0))
    third = tuple(Fraction(value) for value in (2, 0, 1, -1, 4, 3, 0, -2))
    zero = (Fraction(0),) * 8

    assert algebra.associator_exact(left, left, right) == zero
    assert algebra.associator_exact(right, left, left) == zero
    forward = algebra.associator_exact(left, right, third)
    reverse = algebra.associator_exact(right, left, third)
    assert forward == tuple(-value for value in reverse)


def test_numeric_derived_products_preserve_bracketing_jit_and_jvp():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    product = algebra.prepare_product(backend="sparse")
    left = jnp.asarray([0.3, -0.2, 0.7, 0.1, -0.4, 0.8, 0.5, -0.6])
    middle = jnp.asarray([-0.1, 0.4, 0.2, -0.8, 0.3, 0.6, -0.5, 0.7])
    right = jnp.asarray([0.2, 0.5, -0.3, 0.9, -0.7, 0.1, 0.4, -0.2])

    expected = product(product(left, middle), right) - product(
        left, product(middle, right)
    )
    compiled = eqx.filter_jit(product.associator)(left, middle, right)
    tangent = jax.jvp(
        lambda value: product.associator(value, middle, right),
        (left,),
        (jnp.ones_like(left),),
    )[1]

    assert jnp.allclose(compiled, expected, atol=1e-12)
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.linalg.norm(compiled) > 0.0


def test_jordan_product_promotes_integer_coordinates_without_truncation():
    algebra = phx.metrix.algebra.FiniteRealAlgebraSpec(
        "rational-product",
        ("1", "u"),
        (
            (0, 0, 0, 1, 1),
            (0, 1, 1, 1, 1),
            (1, 0, 0, 1, 1),
        ),
        (1, 0),
        ((1, 0), (0, 1)),
    )
    product = algebra.prepare_product(backend="sparse")
    left = jnp.asarray([0, 1], dtype=jnp.int32)
    right = jnp.asarray([1, 0], dtype=jnp.int32)

    value = product.jordan_product(left, right)

    assert jnp.issubdtype(value.dtype, jnp.floating)
    assert jnp.array_equal(value, jnp.asarray([0.5, 0.5]))


def test_higher_cayley_dickson_does_not_inherit_octonion_alternation():
    algebra = phx.metrix.algebra.CayleyDicksonAlgebraSpec(4)
    witness = algebra.properties.claim("left_alternative").witness
    basis = jnp.eye(algebra.coordinate_dimension)
    positions = tuple(algebra.basis_index(label) for label in witness)
    product = algebra.prepare_product(backend="sparse")
    left, middle, right = (basis[position] for position in positions)

    polarized = product.associator(left, middle, right) + product.associator(
        middle, left, right
    )

    assert jnp.linalg.norm(polarized) > 0.0
