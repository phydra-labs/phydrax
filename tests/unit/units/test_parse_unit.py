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


def test_parse_unit_scenario_1() -> None:
    catalog = {
        name: value
        for name, value in vars(units).items()
        if isinstance(value, UnitDefinition)
    }
    assert catalog
    for name, unit in catalog.items():
        assert parse_unit(unit.symbol) == unit, name
    cases: tuple[tuple[str, DimensionSignature, Fraction], ...] = (
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
        (
            "s J^-1/2 m^-1/2",
            TIME / (ENERGY * LENGTH) ** Fraction(1, 2),
            Fraction(1),
        ),
        ("mm^+2 / ms", LENGTH**2 / TIME, Fraction(1, 1000)),
    )
    for expression, dimension, scale in cases:
        unit = parse_unit(expression)
        assert unit.symbol == expression
        assert unit.dimension == dimension, expression
        assert unit.scale_to_reference == scale, expression
    left = parse_unit("kg/(m^2*s)")
    right = parse_unit("kg m^-2 s^-1")
    assert (left.dimension, left.scale_to_reference) == (
        right.dimension,
        right.scale_to_reference,
    )
    assert parse_unit("kg/m3") == units.KILOGRAM_PER_CUBIC_METER


def test_parse_unit_scenario_2() -> None:
    assert parse_unit("kg/m/s").dimension == DimensionSignature(
        {"mass": 1, "length": -1, "time": -1}
    )
    for expression in ("furlong", "m/fortnight", "W", "degC"):
        with pytest.raises(ValueError, match="unknown unit symbol") as error:
            parse_unit(expression)
        assert repr(expression) in str(error.value)
    malformed = (
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
    )
    for expression in malformed:
        with pytest.raises(ValueError, match="Unit expression"):
            parse_unit(expression)


def test_parse_unit_scenario_3() -> None:
    with pytest.raises(ValueError, match="exact rational scale"):
        parse_unit("mm^(1/2)")
    with pytest.raises(TypeError):
        # ty: ignore[invalid-argument-type]
        parse_unit(units.METER)
