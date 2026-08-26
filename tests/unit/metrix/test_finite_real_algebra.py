from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def test_complex_and_quaternion_tables_have_exact_declared_laws():
    complex_algebra = phx.metrix.algebra.ComplexAlgebraSpec()
    quaternion = phx.metrix.algebra.QuaternionAlgebraSpec()
    complex_product = complex_algebra.prepare_product(backend="sparse")
    quaternion_product = quaternion.prepare_product(backend="sparse")
    complex_basis = jnp.eye(2)
    quaternion_basis = jnp.eye(4)

    assert jnp.array_equal(
        complex_product(complex_basis[1], complex_basis[1]),
        jnp.asarray([-1.0, 0.0]),
    )
    assert jnp.array_equal(
        quaternion_product(quaternion_basis[1], quaternion_basis[2]),
        quaternion_basis[3],
    )
    assert jnp.array_equal(
        quaternion_product(quaternion_basis[2], quaternion_basis[1]),
        -quaternion_basis[3],
    )
    assert complex_algebra.properties.proven("commutative")
    assert quaternion.properties.proven("associative")
    assert quaternion.properties.claim("commutative").status == "disproven"
    assert quaternion.properties.proven("division_algebra")


def test_octonion_bracketing_and_cayley_dickson_property_loss_are_explicit():
    octonion = phx.metrix.algebra.OctonionAlgebraSpec()
    product = octonion.prepare_product(backend="sparse")
    basis = jnp.eye(8)
    witness = octonion.properties.claim("associative").witness
    left = octonion.basis_index(witness[0])
    middle = octonion.basis_index(witness[1])
    right = octonion.basis_index(witness[2])
    left_bracket = product(product(basis[left], basis[middle]), basis[right])
    right_bracket = product(basis[left], product(basis[middle], basis[right]))
    sedenion = phx.metrix.algebra.CayleyDicksonAlgebraSpec(4)

    assert not jnp.array_equal(left_bracket, right_bracket)
    assert octonion.properties.claim("associative").status == "disproven"
    assert octonion.properties.proven("alternative")
    assert sedenion.properties.claim("alternative").status == "disproven"
    assert sedenion.properties.claim("division_algebra").status == "disproven"
    assert sedenion.properties.proven("has_zero_divisors")


def test_multicomplex_zero_divisor_is_not_quaternion_multiplication():
    algebra = phx.metrix.algebra.MulticomplexAlgebraSpec(2)
    product = algebra.prepare_product(backend="sparse")
    left = jnp.asarray([1.0, 0.0, 0.0, 1.0])
    right = jnp.asarray([1.0, 0.0, 0.0, -1.0])
    quaternion = phx.metrix.algebra.QuaternionAlgebraSpec()
    restored = phx.metrix.algebra.FiniteRealAlgebraSpec.from_dict(algebra.to_dict())

    assert jnp.array_equal(product(left, right), jnp.zeros((4,)))
    assert algebra.properties.proven("commutative")
    assert algebra.properties.proven("has_zero_divisors")
    assert algebra.algebra_id != quaternion.algebra_id
    assert restored.algebra_id == algebra.algebra_id
    assert restored.spec_id == algebra.spec_id


def test_sparse_dense_lowered_and_differentiated_products_agree():
    algebra = phx.metrix.algebra.QuaternionAlgebraSpec()
    sparse = algebra.prepare_product(backend="sparse")
    dense = algebra.prepare_product(backend="dense")
    left = jnp.asarray([[1.0, 0.2, -0.3, 0.4], [0.1, 0.5, 0.7, -0.2]])
    right = jnp.asarray([0.3, -0.4, 0.2, 0.8])
    compiled = eqx.filter_jit(sparse)(left, right)
    derivative = jax.jvp(
        lambda value: sparse(value, right), (left,), (jnp.ones_like(left),)
    )[1]
    lowered = sparse.lower((2,), jnp.float64)
    lowered_state = {
        "left": left,
        "right": jnp.broadcast_to(right, left.shape),
        "output": jnp.zeros_like(left),
    }
    report = phx.discretization.compare_lowered_backends(lowered, lowered_state)

    assert jnp.allclose(compiled, dense(left, right), atol=1e-12)
    assert jnp.all(jnp.isfinite(derivative))
    assert bool(report.passed)


def test_custom_rational_table_and_resource_failures_are_exact():
    budget = phx.metrix.algebra.AlgebraResourceBudget(maximum_coordinates=2)
    algebra = phx.metrix.algebra.FiniteRealAlgebraSpec(
        "dual-number",
        ("1", "epsilon"),
        (
            (0, 0, 0, 1, 1),
            (0, 1, 1, 1, 1),
            (1, 0, 1, 1, 1),
        ),
        (1, 0),
        ((1, 0), (0, -1)),
        budget=budget,
    )
    epsilon = (Fraction(0), Fraction(1))

    restored = phx.metrix.algebra.FiniteRealAlgebraSpec.from_dict(algebra.to_dict())
    assert algebra.product_exact(epsilon, epsilon) == (Fraction(0), Fraction(0))
    assert algebra.properties.proven("associative")
    assert restored.algebra_id == algebra.algebra_id
    assert restored.spec_id == algebra.spec_id
    with pytest.raises(ValueError, match="maximum"):
        phx.metrix.algebra.CayleyDicksonAlgebraSpec(2, budget=budget)


def test_clifford_provider_implements_finite_algebra_protocol():
    clifford = phx.metrix.clifford.CliffordAlgebraSpec((1, 1, 1))
    provider = phx.metrix.clifford.CliffordFiniteAlgebraProvider(clifford)
    product = provider.prepare_product()
    basis = jnp.eye(provider.coordinate_dimension)

    assert isinstance(provider, phx.metrix.algebra.FiniteRealAlgebraProvider)
    assert provider.coordinate_dimension == 8
    assert product(basis[1], basis[2]).shape == (8,)
