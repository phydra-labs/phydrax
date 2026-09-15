from fractions import Fraction

import jax
import jax.numpy as jnp
import pytest

from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.units import KILOGRAM


def test_si_relativity_scale_round_trips_mass_and_quantum_quantities():
    scale = RelativityScaleContract.si()

    assert scale.gravitational_constant == Fraction(66_743, 10**15)
    assert scale.speed_of_light == Fraction(299_792_458)
    assert scale.reduced_planck_constant == Fraction(1_054_571_817, 10**43)
    assert scale.boltzmann_constant == Fraction(1_380_649, 10**29)
    assert scale.quantum_constants_explicit

    masses = jnp.asarray([1.0, 2.5, 10.0])
    lengths = jax.jit(scale.mass_to_geometric_length)(masses)
    times = jax.jit(scale.mass_to_geometric_time)(masses)
    assert jnp.allclose(scale.geometric_length_to_mass(lengths), masses, rtol=2e-6)
    assert jnp.allclose(scale.geometric_time_to_mass(times), masses, rtol=2e-6)

    accelerations = jnp.asarray([1.0e12, 2.0e12])
    temperatures = scale.surface_gravity_to_temperature(accelerations)
    assert jnp.all(jnp.isfinite(temperatures))
    assert jnp.allclose(
        scale.temperature_to_surface_gravity(temperatures),
        accelerations,
        rtol=2e-6,
    )

    areas = jnp.asarray([1.0e-30, 2.0e-30])
    entropies = scale.area_to_entropy(areas)
    assert jnp.all(jnp.isfinite(entropies))
    assert jnp.all(entropies > 0.0)
    assert jnp.allclose(scale.entropy_to_area(entropies), areas, rtol=2e-6)


def test_relativity_conversion_does_not_clip_or_repair_numeric_inputs():
    scale = RelativityScaleContract.si()
    values = jnp.asarray([-1.0, jnp.nan])

    converted = scale.mass_to_geometric_length(values)

    assert converted[0] < 0.0
    assert jnp.isnan(converted[1])


def test_geometric_scale_sets_classical_constants_to_one_without_erasing_quantum_constants():
    scale = RelativityScaleContract.geometric(KILOGRAM)

    assert scale.gravitational_constant == 1
    assert scale.speed_of_light == 1
    assert scale.reduced_planck_constant > 0
    assert scale.boltzmann_constant > 0
    assert float(scale.mass_to_geometric_length(jnp.asarray(3.0))) == pytest.approx(3.0)
    assert float(scale.mass_to_geometric_time(jnp.asarray(3.0))) == pytest.approx(3.0)
    assert scale.temperature_unit.dimension.terms == (("temperature", 1, 1),)
    assert scale.entropy_unit == scale.boltzmann_constant_unit


def test_relativity_scale_payload_is_content_addressed_and_tamper_evident():
    scale = RelativityScaleContract.si()
    restored = RelativityScaleContract.from_dict(scale.to_dict())

    assert restored.scale_id == scale.scale_id
    assert restored.to_dict() == scale.to_dict()

    payload = scale.to_dict()
    payload["speed_of_light"] = {"numerator": 1, "denominator": 1}
    with pytest.raises(ValueError, match="fingerprint"):
        RelativityScaleContract.from_dict(payload)


def test_relativity_scale_rejects_invalid_constants_and_requires_quantum_declaration_for_quantum_maps():
    dimensional = DimensionalScaleContract.si()
    constants = ("6.67430e-11", 299_792_458, "1.054571817e-34", "1.380649e-23")

    for index, invalid in enumerate((0.0, -1.0, float("inf"), float("nan"))):
        values = list(constants)
        values[index] = invalid
        with pytest.raises(ValueError, match="finite and positive"):
            RelativityScaleContract(dimensional, *values)

    classical = RelativityScaleContract(
        dimensional,
        *constants,
        quantum_constants_explicit=False,
    )
    with pytest.raises(ValueError, match="explicitly declared"):
        classical.area_to_entropy(1.0)
    with pytest.raises(ValueError, match="explicitly declared"):
        classical.surface_gravity_to_temperature(1.0)
