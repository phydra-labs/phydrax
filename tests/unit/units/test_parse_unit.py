from __future__ import annotations

from fractions import Fraction

import pytest

import phydrax.units as units
from phydrax.units import (
    AMOUNT,
    DIMENSIONLESS,
    DimensionSignature,
    ENERGY,
    LENGTH,
    MASS,
    parse_unit,
    PRESSURE,
    TIME,
    UnitDefinition,
    VELOCITY,
)


def test_every_exported_catalog_unit_resolves_from_its_own_symbol():
    catalog = {
        name: value
        for name, value in vars(units).items()
        if isinstance(value, UnitDefinition)
    }
    assert catalog
    for name, unit in catalog.items():
        assert parse_unit(unit.symbol) == unit, name


@pytest.mark.parametrize(
    ("expression", "dimension", "scale"),
    [
        ("kg/m^3", MASS / LENGTH**3, Fraction(1)),
        ("kg/(m^2*s)", MASS / (LENGTH**2 * TIME), Fraction(1)),
        ("(m/s)^2", VELOCITY**2, Fraction(1)),
        ("1/s", DIMENSIONLESS / TIME, Fraction(1)),
        ("1", DIMENSIONLESS, Fraction(1)),
        ("J/kg", ENERGY / MASS, Fraction(1)),
        ("kJ/mol", ENERGY / AMOUNT, Fraction(1000)),
        ("km/ms", VELOCITY, Fraction(10**6)),
        ("cm^-1", DIMENSIONLESS / LENGTH, Fraction(100)),
        ("Pa*s2/m3", PRESSURE * TIME**2 / LENGTH**3, Fraction(1)),
        ("kPa m^-1", PRESSURE / LENGTH, Fraction(1000)),
        ("cm^(1/2)", LENGTH ** Fraction(1, 2), Fraction(1, 10)),
        ("s J^-1/2 m^-1/2", TIME / (ENERGY * LENGTH) ** Fraction(1, 2), Fraction(1)),
        ("mm^+2 / ms", LENGTH**2 / TIME, Fraction(1, 1000)),
    ],
)
def test_compound_expressions_resolve_to_exact_dimension_and_scale(
    expression, dimension, scale
):
    unit = parse_unit(expression)
    assert unit.symbol == expression
    assert unit.dimension == dimension
    assert unit.scale_to_reference == scale


def test_equivalent_spellings_share_dimension_and_scale():
    left = parse_unit("kg/(m^2*s)")
    right = parse_unit("kg m^-2 s^-1")
    assert (left.dimension, left.scale_to_reference) == (
        right.dimension,
        right.scale_to_reference,
    )
    assert parse_unit("kg/m3") == units.KILOGRAM_PER_CUBIC_METER


def test_quotients_associate_left_to_right():
    assert parse_unit("kg/m/s").dimension == DimensionSignature(
        {"mass": 1, "length": -1, "time": -1}
    )


@pytest.mark.parametrize("expression", ["furlong", "m/fortnight", "W", "degC"])
def test_unknown_symbols_raise_value_error_naming_the_expression(expression):
    with pytest.raises(ValueError, match="unknown unit symbol") as error:
        parse_unit(expression)
    assert repr(expression) in str(error.value)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        " m",
        "kg//m",
        "m^",
        "m^x",
        "(m",
        "m)",
        "kg*",
        "2*m",
        "m(s)",
        "m^1/0",
        "m.s",
        "J/kg K",
    ],
)
def test_malformed_expressions_raise_value_error(expression):
    with pytest.raises(ValueError, match="Unit expression"):
        parse_unit(expression)


def test_rational_power_without_exact_scale_fails_closed():
    with pytest.raises(ValueError, match="exact rational scale"):
        parse_unit("mm^(1/2)")


def test_non_string_expression_is_a_type_error():
    with pytest.raises(TypeError):
        parse_unit(units.METER)
