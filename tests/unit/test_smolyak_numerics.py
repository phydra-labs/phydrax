#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._numerics import (
    barycentric_interpolate,
    clenshaw_curtis_data,
    dense_index,
    smolyak_axis_data,
    smolyak_terms,
    weighted_total_degree_indices,
)


def _brute_indices(dimension, level, anisotropy):
    budget = level - 1
    maxima = tuple(int(budget // weight) for weight in anisotropy)
    return {
        index
        for index in itertools.product(*(range(maximum + 1) for maximum in maxima))
        if sum(weight * entry for weight, entry in zip(anisotropy, index, strict=True))
        <= np.nextafter(float(budget), np.inf)
    }


def _brute_coefficients(indices):
    dimension = len(next(iter(indices)))
    coefficients = {}
    for index in indices:
        coefficient = 0
        for corner in itertools.product((0, 1), repeat=dimension):
            upper = tuple(entry + step for entry, step in zip(index, corner, strict=True))
            if upper in indices:
                coefficient += (-1) ** sum(corner)
        if coefficient:
            coefficients[index] = coefficient
    return coefficients


@pytest.mark.parametrize(
    ("dimension", "level", "anisotropy"),
    [
        (1, 5, (1.0,)),
        (2, 4, (1.0, 1.0)),
        (3, 4, (1.0, 2.0, 0.75)),
        (4, 3, (2.0, 1.0, 1.5, 0.5)),
    ],
)
def test_sparse_indices_and_mobius_coefficients_match_brute_reference(
    dimension, level, anisotropy
):
    expected_indices = _brute_indices(dimension, level, anisotropy)
    actual_indices = {
        dense_index(index, dimension)
        for index in weighted_total_degree_indices(dimension, level, anisotropy)
    }
    actual_coefficients = {
        dense_index(term.index, dimension): term.coefficient
        for term in smolyak_terms(dimension, level, anisotropy)
    }

    assert actual_indices == expected_indices
    assert actual_coefficients == _brute_coefficients(expected_indices)


def test_real_anisotropy_includes_threshold_boundary_and_is_axis_equivariant():
    indices = {
        dense_index(index, 2) for index in weighted_total_degree_indices(2, 2, (0.1, 0.2))
    }
    permuted = {
        dense_index(index, 2)[::-1]
        for index in weighted_total_degree_indices(2, 2, (0.2, 0.1))
    }

    assert (10, 0) in indices
    assert (0, 5) in indices
    assert indices == permuted


@pytest.mark.parametrize(
    "anisotropy",
    [(1.0,), (1.0, 2.0, 3.0), (0.0, 1.0), (-1.0, 1.0), (np.nan, 1.0), (np.inf, 1.0)],
)
def test_invalid_anisotropy_is_rejected(anisotropy):
    with pytest.raises(ValueError, match="anisotropy"):
        weighted_total_degree_indices(2, 3, anisotropy)


def test_high_dimensional_low_level_construction_is_sparse():
    indices = weighted_total_degree_indices(32, 3)
    terms = smolyak_terms(32, 3)

    assert len(indices) == 561
    assert len(terms) == 561
    assert max(len(index) for index in indices) == 2


def test_clenshaw_curtis_has_one_point_base_and_structural_nested_ids():
    base = clenshaw_curtis_data(1)
    assert jnp.array_equal(base.nodes, jnp.asarray([0.0]))
    assert jnp.array_equal(base.weights, jnp.asarray([2.0]))

    seen = {}
    for level in range(5):
        data = smolyak_axis_data("clenshaw-curtis", level)
        for identifier, node in zip(data.node_ids, data.nodes, strict=True):
            if identifier in seen:
                assert node == pytest.approx(seen[identifier], abs=1e-15)
            seen[identifier] = node
    assert smolyak_axis_data("clenshaw-curtis", 0).node_ids[0] in set(
        smolyak_axis_data("clenshaw-curtis", 4).node_ids
    )


def test_leja_sequence_is_nested_by_identity():
    previous = ()
    for level in range(8):
        current = smolyak_axis_data("leja", level)
        assert current.node_ids[: len(previous)] == previous
        previous = current.node_ids


def test_gauss_hermite_data_preserves_standard_normal_moments():
    data = smolyak_axis_data("gauss-hermite", 5)
    nodes = data.nodes
    weights = data.quadrature_weights
    assert weights is not None

    assert np.sum(weights) == pytest.approx(1.0, abs=1e-14)
    assert np.sum(weights * nodes) == pytest.approx(0.0, abs=1e-14)
    assert np.sum(weights * nodes**2) == pytest.approx(1.0, abs=1e-14)
    assert np.sum(weights * nodes**4) == pytest.approx(3.0, abs=1e-13)


def test_barycentric_derivatives_are_finite_and_exact_at_nodes():
    data = smolyak_axis_data("clenshaw-curtis", 3)
    nodes = jnp.asarray(data.nodes)
    weights = jnp.asarray(data.barycentric_weights)
    values = nodes**4 - 2.0 * nodes**2 + nodes

    def interpolated(x):
        return barycentric_interpolate(x, nodes, weights, values)

    first = jax.vmap(jax.grad(interpolated))(nodes)
    second = jax.vmap(jax.grad(jax.grad(interpolated)))(nodes)

    assert jnp.all(jnp.isfinite(first))
    assert jnp.all(jnp.isfinite(second))
    assert jnp.allclose(first, 4.0 * nodes**3 - 4.0 * nodes + 1.0, atol=1e-11)
    assert jnp.allclose(second, 12.0 * nodes**2 - 4.0, atol=1e-10)
