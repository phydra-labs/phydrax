from __future__ import annotations

import math

import numpy as np
import pytest

from phydrax._polynomial._cubature import simplex_rule_data, tensor_product_rule_data
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.discretization._side_actions import FacetTraceRule


@pytest.mark.parametrize("dimension", [1, 2, 3, 4, 5])
def test_simplex_rule_integrates_independent_dirichlet_moments(dimension: int) -> None:
    rule = simplex_rule_data(dimension, 4)
    points, weights = np.asarray(rule.points), np.asarray(rule.weights)
    exponents = np.zeros((dimension,), dtype=np.int64)
    exponents[0] = 4 if dimension == 1 else 2
    if dimension > 1:
        exponents[-1] = 1
    expected = math.prod(
        math.factorial(int(value)) for value in exponents
    ) / math.factorial(int(np.sum(exponents)) + dimension)
    actual = weights @ np.prod(points**exponents, axis=1)
    assert actual == pytest.approx(expected, rel=2.0e-13, abs=2.0e-14)
    assert np.sum(weights) == pytest.approx(1.0 / math.factorial(dimension), rel=2.0e-13)
    assert np.all(points >= 0.0) and np.all(np.sum(points, axis=1) <= 1.0 + 1.0e-14)
    assert rule.exact_degree >= 4
    assert rule.reference_dimension == dimension


@pytest.mark.parametrize("family", ["gauss", "lobatto"])
@pytest.mark.parametrize("dimension", [1, 3, 4])
def test_tensor_rule_integrates_product_moments(family: str, dimension: int) -> None:
    if family == "gauss":
        rule = tensor_product_rule_data(dimension, 3, family="gauss")
    else:
        rule = tensor_product_rule_data(dimension, 4, family="lobatto")
    points, weights = np.asarray(rule.points), np.asarray(rule.weights)
    exponents = np.full((dimension,), 3, dtype=np.int64)
    actual = weights @ np.prod(points**exponents, axis=1)
    assert actual == pytest.approx(4.0**-dimension, rel=2.0e-13, abs=2.0e-14)
    assert np.sum(weights) == pytest.approx(1.0, rel=2.0e-13)
    assert np.all(points >= 0.0) and np.all(points <= 1.0)


@pytest.mark.parametrize("kind", ["simplex:4", "tensor:4"])
def test_generic_facet_rule_integrates_reference_measure(kind: str) -> None:
    shape = reference_cell_topology(kind)
    rule = FacetTraceRule(points=4)
    points, weights = rule.reference(shape)
    expected = 1.0 / math.factorial(4) if kind == "simplex:4" else 1.0
    assert np.sum(weights) == pytest.approx(expected, rel=2.0e-13)
    assert points.shape[1] == 4
    assert rule.exact_degree(shape) == (4 if kind == "simplex:4" else 7)


def test_simplex_facet_rule_refuses_insufficient_constant_measure_order() -> None:
    with pytest.raises(ValueError, match="insufficient"):
        FacetTraceRule(points=1).reference(reference_cell_topology("simplex:4"))


@pytest.mark.parametrize("dimension", [4, 10_000])
def test_n_dimensional_rule_refuses_capacity_before_tensor_allocation(
    dimension: int,
) -> None:
    with pytest.raises(ValueError, match="maximum_rule_bytes"):
        tensor_product_rule_data(dimension, 8, maximum_rule_bytes=128)
