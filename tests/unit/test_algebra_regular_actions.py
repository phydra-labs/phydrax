import jax.numpy as jnp
import pytest

import phydrax as phx


def test_left_and_right_quaternion_actions_are_distinct_native_operators():
    algebra = phx.metrix.algebra.QuaternionAlgebraSpec()
    product = algebra.prepare_product(backend="sparse")
    space = phx.linalg.AlgebraArraySpace((), algebra, dtype=jnp.float64)
    multiplier = jnp.asarray([0.0, 1.0, 0.0, 0.0])
    value = jnp.asarray([0.0, 0.0, 1.0, 0.0])
    left = phx.linalg.algebra_regular_action_operator(
        product, multiplier, space, side="left"
    )
    right = phx.linalg.algebra_regular_action_operator(
        product, multiplier, space, side="right"
    )

    assert jnp.array_equal(left.mv(value), jnp.asarray([0.0, 0.0, 0.0, 1.0]))
    assert jnp.array_equal(right.mv(value), jnp.asarray([0.0, 0.0, 0.0, -1.0]))
    assert left.operator_id != right.operator_id


def test_octonion_action_composition_does_not_collapse_to_one_multiplier():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    product = algebra.prepare_product(backend="sparse")
    space = phx.linalg.AlgebraArraySpace((), algebra, dtype=jnp.float64)
    witness = algebra.properties.claim("associative").witness
    positions = tuple(algebra.basis_index(label) for label in witness)
    basis = jnp.eye(8)
    left, middle, value = (basis[position] for position in positions)
    left_action = phx.linalg.algebra_regular_action_operator(
        product, left, space, side="left"
    )
    middle_action = phx.linalg.algebra_regular_action_operator(
        product, middle, space, side="left"
    )
    collapsed = phx.linalg.algebra_regular_action_operator(
        product, product(left, middle), space, side="left"
    )

    composed_value = (left_action @ middle_action).mv(value)

    assert not jnp.array_equal(composed_value, collapsed.mv(value))
    assert jnp.array_equal(
        composed_value - collapsed.mv(value),
        -product.associator(left, middle, value),
    )


def test_regular_action_materialization_and_transpose_match_pairing():
    algebra = phx.metrix.algebra.OctonionAlgebraSpec()
    product = algebra.prepare_product(backend="sparse")
    space = phx.linalg.AlgebraArraySpace((), algebra, dtype=jnp.float64)
    multiplier = jnp.linspace(-0.4, 0.6, 8)
    value = jnp.linspace(0.2, 0.9, 8)
    probe = jnp.linspace(-0.7, 0.3, 8)
    action = phx.linalg.algebra_regular_action_operator(
        product, multiplier, space, side="right"
    )
    matrix = phx.linalg.materialize(action, phx.linalg.MaterializationPolicy())

    assert jnp.allclose(action.mv(value), matrix @ value, atol=1e-12)
    assert jnp.allclose(
        space.inner(action.mv(value), probe),
        space.inner(value, action.transpose_mv(probe)),
        atol=1e-12,
    )


def test_regular_actions_support_nonfinal_axes_and_base_shaped_multipliers():
    algebra = phx.metrix.algebra.QuaternionAlgebraSpec()
    layout = phx.metrix.algebra.AlgebraElementLayout(algebra, algebra_axis=0)
    product = algebra.prepare_product(layout=layout, backend="sparse")
    space = phx.linalg.AlgebraArraySpace((2,), algebra, algebra_axis=0, dtype=jnp.float64)
    multiplier = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    value = jnp.asarray([[1.0, 1.0], [0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    action = phx.linalg.algebra_regular_action_operator(
        product, multiplier, space, side="left"
    )

    assert jnp.array_equal(action.mv(value), product(multiplier, value))


def test_regular_action_rejects_ambiguous_layout_dtype_and_side():
    quaternion = phx.metrix.algebra.QuaternionAlgebraSpec()
    octonion = phx.metrix.algebra.OctonionAlgebraSpec()
    product = quaternion.prepare_product(backend="sparse")
    space = phx.linalg.AlgebraArraySpace((), quaternion, dtype=jnp.float64)
    multiplier = jnp.asarray([0.0, 1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="side"):
        phx.linalg.algebra_regular_action_operator(
            product, multiplier, space, side="center"
        )
    with pytest.raises(TypeError, match="dtype"):
        phx.linalg.algebra_regular_action_operator(
            product, multiplier.astype(jnp.float32), space, side="left"
        )
    with pytest.raises(ValueError, match="match"):
        phx.linalg.algebra_regular_action_operator(
            product,
            multiplier,
            phx.linalg.AlgebraArraySpace((), octonion, dtype=jnp.float64),
            side="left",
        )
