#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


cl = phx.metrix.clifford


def _basis(layout):
    return tuple(cl.basis_blade(layout, bitmap) for bitmap in layout.bitmaps)


def test_euclidean_plane_multiplication_table_and_associativity():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    layout = cl.CliffordBladeLayout.full(algebra)
    product = cl.prepare_product(algebra, layout, layout, backend="sparse")
    blades = _basis(layout)
    expected = (
        ((0, 1), (1, 1), (2, 1), (3, 1)),
        ((1, 1), (0, 1), (3, 1), (2, 1)),
        ((2, 1), (3, -1), (0, 1), (1, -1)),
        ((3, 1), (2, -1), (1, 1), (0, -1)),
    )

    for left, row in zip(blades, expected):
        for right, (position, sign) in zip(blades, row):
            assert jnp.array_equal(product(left, right), sign * blades[position])

    for left in blades:
        for middle in blades:
            for right in blades:
                assert jnp.array_equal(
                    product(product(left, middle), right),
                    product(left, product(middle, right)),
                )


def test_signature_radical_and_sparse_closure_are_explicit():
    algebra = cl.CliffordAlgebraSpec((1, -1, 0))
    full = cl.CliffordBladeLayout.full(algebra)
    product = cl.prepare_product(algebra, full, full, backend="sparse")
    e0, e1, e2 = (
        cl.basis_blade(full, 1),
        cl.basis_blade(full, 2),
        cl.basis_blade(full, 4),
    )
    scalar = cl.basis_blade(full, 0)

    assert jnp.array_equal(product(e0, e0), scalar)
    assert jnp.array_equal(product(e1, e1), -scalar)
    assert jnp.array_equal(product(e2, e2), jnp.zeros_like(scalar))

    vectors = cl.CliffordBladeLayout.grades_layout(algebra, (1,))
    vector_product = cl.prepare_product(
        algebra,
        vectors,
        vectors,
        kind="geometric",
        backend="sparse",
    )
    assert vector_product.output_layout.grade_set == (0, 2)
    with pytest.raises(ValueError, match="drops nonzero"):
        cl.prepare_product(
            algebra,
            vectors,
            vectors,
            output_layout=cl.CliffordBladeLayout.grades_layout(algebra, (0,)),
        )


def test_product_kinds_involutions_and_layout_maps():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    full = cl.CliffordBladeLayout.full(algebra)
    vectors = cl.CliffordBladeLayout.grades_layout(algebra, (1,))
    exterior = cl.prepare_product(
        algebra,
        vectors,
        vectors,
        kind="exterior",
        backend="sparse",
    )
    vector_blades = _basis(vectors)
    assert exterior.output_layout.bitmaps == (3,)
    assert jnp.array_equal(exterior(vector_blades[0], vector_blades[0]), jnp.zeros((1,)))
    assert jnp.array_equal(exterior(vector_blades[0], vector_blades[1]), jnp.ones((1,)))
    assert jnp.array_equal(exterior(vector_blades[1], vector_blades[0]), -jnp.ones((1,)))

    values = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    assert jnp.array_equal(
        cl.grade_involution(values, full), jnp.asarray([1.0, -2.0, -3.0, 4.0])
    )
    assert jnp.array_equal(cl.reverse(values, full), jnp.asarray([1.0, 2.0, 3.0, -4.0]))
    assert jnp.array_equal(
        cl.clifford_conjugate(values, full), jnp.asarray([1.0, -2.0, -3.0, -4.0])
    )
    extracted, grade_one = cl.project_grades(values, full, (1,))
    assert grade_one.layout_id == vectors.layout_id
    assert jnp.array_equal(extracted, jnp.asarray([2.0, 3.0]))
    assert jnp.array_equal(
        cl.embed_layout(extracted, vectors, full), jnp.asarray([0.0, 2.0, 3.0, 0.0])
    )


def test_product_preserves_dtype_and_jax_transformability():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    layout = cl.CliffordBladeLayout.full(algebra)
    sparse = cl.prepare_product(algebra, layout, layout, backend="sparse")
    dense = cl.prepare_product(algebra, layout, layout, backend="dense")
    values = jnp.arange(8, dtype=jnp.float64).reshape((2, 4)) / 7.0

    assert jnp.allclose(jax.jit(sparse)(values, values), dense(values, values))
    assert jax.vmap(lambda value: sparse(value, value))(values).shape == (2, 4)
    gradient = jax.grad(lambda value: cl.scalar_part(sparse(value, value), layout))(
        values[0]
    )
    assert gradient.dtype == jnp.float64
    complex_values = values.astype(jnp.complex128) * (1.0 + 1.0j)
    assert sparse(complex_values, complex_values).dtype == jnp.complex128


def test_resource_budget_rejects_before_layout_allocation():
    budget = cl.CliffordResourceBudget(maximum_blades=3)
    algebra = cl.CliffordAlgebraSpec((1, 1), budget=budget)
    with pytest.raises(ValueError, match="budget allows 3"):
        cl.CliffordBladeLayout.full(algebra)
    assert cl.CliffordBladeLayout.grades_layout(algebra, (0, 1)).blade_count == 3


def test_algebra_identity_is_independent_of_execution_budget_and_orientation():
    first = cl.CliffordAlgebraSpec((1, -1))
    second = cl.CliffordAlgebraSpec(
        (1, -1),
        orientation=-1,
        budget=cl.CliffordResourceBudget(maximum_plan_bytes=128 * 1024**2),
    )
    assert first.algebra_id == second.algebra_id
    assert first.spec_id != second.spec_id
    first.require_compatible(second)
