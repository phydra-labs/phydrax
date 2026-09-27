from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_units_scenario_1() -> None:
    left = phx.units.DimensionSignature(
        (("time", -1), ("length", Fraction(1, 2)), ("time", 1))
    )
    right = phx.units.DimensionSignature({"length": Fraction(2, 4)})
    assert left == right
    assert left.dimension_id == right.dimension_id
    assert left.terms == (("length", 1, 2),)
    assert (left / left).is_dimensionless
    assert phx.units.LENGTH * phx.units.TIME**-1 == phx.units.VELOCITY
    with pytest.raises(TypeError, match="integers or fractions"):
        # ty: ignore[invalid-argument-type]
        phx.units.DimensionSignature({"length": 0.5})

    dimension = phx.units.ENERGY / phx.units.AMOUNT
    assert phx.units.DimensionSignature.from_dict(dimension.to_dict()) == dimension
    tampered = dimension.to_dict()
    tampered["terms"][0]["numerator"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        phx.units.DimensionSignature.from_dict(tampered)
    ambiguous = dimension.to_dict()
    ambiguous["physical_dimension"] = [1.0]
    with pytest.raises(ValueError, match="canonical fields"):
        phx.units.DimensionSignature.from_dict(ambiguous)
    assert phx.units.conversion_factor(phx.units.KILOMETER, phx.units.METER) == 1000
    converted = phx.units.convert_value(
        jnp.asarray([1.0, 2.0]),
        source=phx.units.KILOMETER,
        target=phx.units.METER,
    )
    np.testing.assert_allclose(converted, [1000.0, 2000.0])
    integer = phx.units.convert_value(
        jnp.asarray([100], dtype=jnp.int16),
        source=phx.units.KILOMETER,
        target=phx.units.METER,
    )
    assert jnp.issubdtype(integer.dtype, jnp.inexact)
    np.testing.assert_allclose(integer, [100_000.0])

    compiled = jax.jit(
        lambda value: phx.units.convert_value(
            value,
            source=phx.units.KILOMETER,
            target=phx.units.METER,
        )
    )
    assert float(jax.grad(compiled)(jnp.asarray(2.0))) == 1000.0
    with pytest.raises(ValueError, match="matching dimensions"):
        phx.units.conversion_factor(phx.units.METER, phx.units.SECOND)
    code_meter = phx.units.UnitDefinition(
        "code_length",
        phx.units.LENGTH,
        "fixture-code",
    )
    with pytest.raises(ValueError, match="shared reference system"):
        phx.units.conversion_factor(code_meter, phx.units.METER)
    unit = phx.units.KILOCALORIE_PER_MOLE
    restored = phx.units.UnitDefinition.from_dict(unit.to_dict())
    assert restored == unit
    assert restored.dimension == phx.units.ENERGY / phx.units.AMOUNT
    tampered = unit.to_dict()
    tampered["scale_numerator"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        phx.units.UnitDefinition.from_dict(tampered)
    ambiguous = unit.to_dict()
    ambiguous["offset_to_reference"] = 0
    with pytest.raises(ValueError, match="canonical fields"):
        phx.units.UnitDefinition.from_dict(ambiguous)

    velocity = phx.units.derived_unit(
        "km/s",
        ((phx.units.KILOMETER, 1), (phx.units.SECOND, -1)),
    )
    assert velocity.dimension == phx.units.VELOCITY
    assert velocity.scale_to_reference == 1000
    huge = phx.units.UnitDefinition(
        "huge-length",
        phx.units.LENGTH,
        "fixture",
        10**400,
    )
    assert huge.scale_to_reference == 10**400
    with pytest.raises(TypeError, match="exact positive rational"):
        # ty: ignore[invalid-argument-type]
        phx.units.UnitDefinition("bad", phx.units.LENGTH, "si", 0.1)
    with pytest.raises(ValueError, match="exact rational root"):
        phx.units.derived_unit(
            "sqrt-km",
            ((phx.units.KILOMETER, Fraction(1, 2)),),
        )
    square_meter = phx.units.derived_unit("m2-fixture", ((phx.units.METER, 2),))
    assert phx.units.conversion_factor(phx.units.BARN, square_meter) == Fraction(
        1, 10**28
    )
    assert phx.units.conversion_factor(
        phx.units.KILOELECTRONVOLT,
        phx.units.ELECTRONVOLT,
    ) == Fraction(1000)
    assert phx.units.conversion_factor(
        phx.units.MEGAELECTRONVOLT,
        phx.units.ELECTRONVOLT,
    ) == Fraction(1_000_000)
    assert phx.units.BECQUEREL.dimension == phx.units.FREQUENCY
    assert phx.units.WEBER.dimension == phx.units.VOLTAGE * phx.units.TIME
    assert phx.units.TESLA.dimension == phx.units.WEBER.dimension / phx.units.AREA
    assert phx.units.HENRY.dimension == phx.units.WEBER.dimension / phx.units.CURRENT
    assert phx.units.HERTZ.dimension == phx.units.FREQUENCY
    assert phx.units.HERTZ.scale_to_reference == 1
    assert phx.units.HERTZ.symbol == "Hz"
