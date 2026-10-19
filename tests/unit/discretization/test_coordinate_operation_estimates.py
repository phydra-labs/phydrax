#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Structural estimates bound charged work and storage of polynomial circuits."""

import math
from fractions import Fraction

import pytest

from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    coordinate_estimate_multiply,
    coordinate_estimate_power,
    coordinate_expression_estimate,
    coordinate_polynomial_estimate,
    multiply,
    Polynomial,
    power,
)
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient


@pytest.mark.parametrize("dimension", (1, 2, 3), ids=("line", "plane", "volume"))
@pytest.mark.parametrize(
    "exponent", (0, 1, 2, 5, 12), ids=("zero", "one", "two", "five", "twelve")
)
def test_power_estimate_bounds_merged_support_and_charged_operation(
    dimension: int, exponent: int
) -> None:
    polynomial: Polynomial = {(0,) * dimension: Fraction(1, 3)}
    for axis in range(dimension):
        index = tuple(int(column == axis) for column in range(dimension))
        polynomial[index] = Fraction(axis + 1, 5 + axis)
    estimate = coordinate_estimate_power(
        coordinate_expression_estimate(polynomial), exponent
    )
    ticket = HostBatchTicket(estimate.work_upper, bytes_upper=estimate.bytes_upper)
    ledger = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=estimate.work_upper,
        maximum_memory_bytes=estimate.bytes_upper,
    )
    with ticket, ledger.activate():
        result = power(polynomial, exponent, dimension, budget=ledger)
    assert len(result) <= estimate.terms <= math.comb(exponent + dimension, dimension)
    assert result[(0,) * dimension] == Fraction(1, 3) ** exponent
    assert ledger.work_units <= estimate.work_upper
    assert ledger.peak_bytes_upper <= estimate.bytes_upper
    assert ticket.work_units == ledger.work_units


def test_large_power_estimate_uses_monomial_support_not_expansion_tree() -> None:
    leaf = coordinate_polynomial_estimate(
        terms=4, dimension=3, degree=1, numerator_bits=3, denominator_bits=7
    )
    estimate = coordinate_estimate_power(leaf, 100_000)
    assert estimate.terms == math.comb(100_003, 3)
    # Sequential merged-support multiplication is polynomial in the exponent.
    # This bound rejects construction of the uncancelled 4**100000 tree.
    assert estimate.work_upper.bit_length() < 128
    assert estimate.bytes_upper.bit_length() < 128


@pytest.mark.parametrize("exponent", (0, 1, 7), ids=("zero", "one", "seven"))
def test_zero_power_estimate_bounds_constant_and_annihilation(exponent: int) -> None:
    estimate = coordinate_estimate_power(coordinate_expression_estimate({}), exponent)
    ticket = HostBatchTicket(estimate.work_upper, bytes_upper=estimate.bytes_upper)
    ledger = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=estimate.work_upper,
        maximum_memory_bytes=estimate.bytes_upper,
    )
    with ticket, ledger.activate():
        result = power({}, exponent, 0, budget=ledger)
    assert result == ({(): Fraction(1)} if exponent == 0 else {})
    assert ledger.work_units <= estimate.work_upper
    assert ledger.peak_bytes_upper <= estimate.bytes_upper


def test_product_estimate_bounds_coprime_common_denominator_growth() -> None:
    first: Polynomial = {
        (0,): Fraction(1, 101),
        (1,): Fraction(1, 103),
        (2,): Fraction(1, 107),
    }
    second: Polynomial = {
        (0,): Fraction(1, 109),
        (1,): Fraction(1, 113),
        (2,): Fraction(1, 127),
    }
    estimate = coordinate_estimate_multiply(
        coordinate_expression_estimate(first), coordinate_expression_estimate(second)
    )
    ticket = HostBatchTicket(estimate.work_upper, bytes_upper=estimate.bytes_upper)
    ledger = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=estimate.work_upper,
        maximum_memory_bytes=estimate.bytes_upper,
    )
    with ticket, ledger.activate():
        result = multiply(first, second, budget=ledger)
    assert len(result) <= estimate.terms == 5
    assert (
        max(
            abs(
                rational_coefficient(value, role="fixture coefficient").numerator
            ).bit_length()
            for value in result.values()
        )
        <= estimate.numerator_bits
    )
    assert ledger.work_units <= estimate.work_upper
    assert ledger.peak_bytes_upper <= estimate.bytes_upper


def test_sparse_wide_dimension_estimates_do_not_expand_unused_box_bounds() -> None:
    monomial = coordinate_polynomial_estimate(
        terms=1,
        dimension=100_000,
        degree=1,
        numerator_bits=3,
        denominator_bits=7,
    )
    powered = coordinate_estimate_power(monomial, 100_000)
    assert powered.terms == 1
    assert powered.degree == 100_000
    assert powered.work_upper.bit_length() < 128
    assert powered.bytes_upper.bit_length() < 128
    sparse = coordinate_polynomial_estimate(
        terms=2,
        dimension=100_000,
        degree=100_000,
        numerator_bits=3,
        denominator_bits=7,
    )
    product = coordinate_estimate_multiply(sparse, sparse)
    assert product.terms == 4
    assert product.work_upper.bit_length() < 128
    assert product.bytes_upper.bit_length() < 128
