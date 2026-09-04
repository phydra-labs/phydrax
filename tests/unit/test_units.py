from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_dimension_signature_is_exact_and_canonical():
    left = phx.units.DimensionSignature(
        (("time", -1), ("length", Fraction(1, 2)), ("time", 1))
    )
    right = phx.units.DimensionSignature({"length": Fraction(2, 4)})
    assert left == right
    assert left.dimension_id == right.dimension_id
    assert left.terms == (("length", 1, 2),)
    assert (left / left).is_dimensionless
    assert (phx.units.LENGTH * phx.units.TIME**-1) == phx.units.VELOCITY


def test_dimension_signature_rejects_inexact_exponents():
    with pytest.raises(TypeError, match="integers or fractions"):
        phx.units.DimensionSignature({"length": 0.5})


def test_dimension_payload_roundtrip_and_tamper_detection():
    dimension = phx.units.ENERGY / phx.units.AMOUNT
    assert phx.units.DimensionSignature.from_dict(dimension.to_dict()) == dimension
    payload = dimension.to_dict()
    payload["terms"][0]["numerator"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        phx.units.DimensionSignature.from_dict(payload)
    ambiguous = dimension.to_dict()
    ambiguous["physical_dimension"] = [1.0]
    with pytest.raises(ValueError, match="canonical fields"):
        phx.units.DimensionSignature.from_dict(ambiguous)


def test_unit_conversion_is_exact_and_differentiable():
    assert phx.units.conversion_factor(phx.units.KILOMETER, phx.units.METER) == Fraction(
        1000
    )
    converted = phx.units.convert_value(
        jnp.asarray([1.0, 2.0]),
        source=phx.units.KILOMETER,
        target=phx.units.METER,
    )
    np.testing.assert_allclose(converted, [1000.0, 2000.0])
    integer_converted = phx.units.convert_value(
        jnp.asarray([100], dtype=jnp.int16),
        source=phx.units.KILOMETER,
        target=phx.units.METER,
    )
    assert jnp.issubdtype(integer_converted.dtype, jnp.inexact)
    np.testing.assert_allclose(integer_converted, [100_000.0])

    convert = jax.jit(
        lambda value: phx.units.convert_value(
            value,
            source=phx.units.KILOMETER,
            target=phx.units.METER,
        )
    )
    assert float(jax.grad(convert)(jnp.asarray(2.0))) == 1000.0


def test_unit_conversion_rejects_dimension_and_reference_mismatches():
    with pytest.raises(ValueError, match="matching dimensions"):
        phx.units.conversion_factor(phx.units.METER, phx.units.SECOND)

    code_meter = phx.units.UnitDefinition("code_length", phx.units.LENGTH, "fixture-code")
    with pytest.raises(ValueError, match="shared reference system"):
        phx.units.conversion_factor(code_meter, phx.units.METER)


def test_unit_payload_roundtrip_and_tamper_detection():
    unit = phx.units.KILOCALORIE_PER_MOLE
    restored = phx.units.UnitDefinition.from_dict(unit.to_dict())
    assert restored == unit
    assert restored.dimension == phx.units.ENERGY / phx.units.AMOUNT

    payload = unit.to_dict()
    payload["scale_numerator"] += 1
    with pytest.raises(ValueError, match="fingerprint"):
        phx.units.UnitDefinition.from_dict(payload)
    ambiguous = unit.to_dict()
    ambiguous["offset_to_reference"] = 0
    with pytest.raises(ValueError, match="canonical fields"):
        phx.units.UnitDefinition.from_dict(ambiguous)


def test_derived_units_require_exact_scales_and_shared_references():
    velocity = phx.units.derived_unit(
        "km/s", ((phx.units.KILOMETER, 1), (phx.units.SECOND, -1))
    )
    assert velocity.dimension == phx.units.VELOCITY
    assert velocity.scale_to_reference == 1000
    huge = phx.units.UnitDefinition("huge-length", phx.units.LENGTH, "fixture", 10**400)
    assert huge.scale_to_reference == 10**400

    with pytest.raises(TypeError, match="exact positive rational"):
        phx.units.UnitDefinition("bad", phx.units.LENGTH, "si", 0.1)
    with pytest.raises(ValueError, match="exact rational root"):
        phx.units.derived_unit("sqrt-km", ((phx.units.KILOMETER, Fraction(1, 2)),))
