from __future__ import annotations

import tracemalloc
from itertools import product
from math import comb

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._holomorphic_linear import HolomorphicMultiIndexSet
from phydrax._polynomial._scaled_monomial import ScaledMonomialBasis


@pytest.mark.parametrize("dimension,degree", [(1, 0), (2, 3), (4, 2)])
def test_holomorphic_total_degree_order_and_identity(dimension: int, degree: int) -> None:
    reference = tuple(
        sorted(
            (
                value
                for value in product(range(degree + 1), repeat=dimension)
                if sum(value) <= degree
            ),
            key=lambda value: (sum(value), value),
        )
    )
    limit = comb(dimension + degree, degree)
    actual = HolomorphicMultiIndexSet.total_degree(dimension, degree, maximum_count=limit)
    explicit = HolomorphicMultiIndexSet(
        dimension, reference, require_downward_closed=True, maximum_count=limit
    )
    assert actual.indices == reference
    assert actual.index_set_id == explicit.index_set_id
    with pytest.raises(ValueError, match="exceeds"):
        HolomorphicMultiIndexSet.total_degree(
            dimension, degree, maximum_count=limit - 1
        ) if limit > 1 else HolomorphicMultiIndexSet.total_degree(
            dimension, degree + 1, maximum_count=limit
        )


@pytest.mark.parametrize("dimension,degree", [(2, 3), (4, 2)])
def test_scaled_total_degree_order_identity_and_values(
    dimension: int, degree: int
) -> None:
    reference = np.asarray(
        sorted(
            (
                value
                for value in product(range(degree + 1), repeat=dimension)
                if sum(value) <= degree
            ),
            key=lambda value: (sum(value), tuple(-entry for entry in value)),
        ),
        dtype=np.int32,
    )
    limit = comb(dimension + degree, degree)
    basis = ScaledMonomialBasis(dimension, degree, maximum_features=limit)
    np.testing.assert_array_equal(basis.exponents, reference)
    expected_id = canonical_fingerprint(
        {
            "kind": "scaled-total-degree-monomials",
            "dimension": dimension,
            "degree": degree,
            "exponents": array_tree_fingerprint(reference),
        }
    )
    assert basis.basis_id == expected_id
    points = jnp.asarray(np.arange(dimension, dtype=np.float32) * 0.25)
    np.testing.assert_allclose(
        basis.evaluate(
            points, jnp.zeros_like(points), jnp.asarray(1.0, dtype=jnp.float32)
        ),
        np.prod(np.asarray(points)[None, :] ** reference, axis=-1),
    )
    with pytest.raises(ValueError, match="budget"):
        ScaledMonomialBasis(dimension, degree, maximum_features=limit - 1)


@pytest.mark.parametrize("kind", ["holomorphic", "scaled"])
def test_huge_polynomial_request_refuses_without_enumeration(kind: str) -> None:
    # Neither the complete simplex nor its enclosing Cartesian box is enumerable.
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match="exceeds|budget"):
            if kind == "holomorphic":
                HolomorphicMultiIndexSet.total_degree(100, 100, maximum_count=8)
            else:
                ScaledMonomialBasis(100, 100, maximum_features=8)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Count arithmetic and exception evidence may allocate, but no index workset.
    assert peak < 64 * 1024
