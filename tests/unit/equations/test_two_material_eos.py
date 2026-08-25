#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.equations import (
    IdealGasMaterial,
    StiffenedGasMaterial,
    TwoMaterialEOSClosure,
    TwoMaterialPrimitiveState,
)


def _materials():
    ideal = IdealGasMaterial(1.4, 287.0, density_floor=1.0e-10)
    stiff = StiffenedGasMaterial(
        4.4,
        6.0e8,
        1816.0,
        reference_energy=2.0e5,
        density_floor=1.0e-10,
    )
    return ideal, stiff


def test_ideal_ideal_round_trip_and_report_coefficients():
    ideal, _ = _materials()
    closure = TwoMaterialEOSClosure(
        ideal,
        IdealGasMaterial(1.67, 287.0),
        mass_floor=1.0e-10,
        energy_floor=1.0e-10,
    )
    primitive = jnp.asarray([[1.2, 0.8, 30.0, -4.0, 101325.0, 0.35]], dtype=jnp.float64)
    conserved = closure.primitive_to_conserved(primitive)
    recovered = closure.conserved_to_primitive(conserved)
    np.testing.assert_allclose(recovered, primitive, rtol=2.0e-12)
    report = closure.report(conserved)
    np.testing.assert_allclose(report.pressure, primitive[..., -2], rtol=2.0e-12)
    np.testing.assert_allclose(
        report.pressure_coefficient * report.internal_energy_coefficient,
        1.0,
        rtol=2.0e-12,
    )
    with pytest.raises(Exception):
        report.pressure = jnp.zeros_like(report.pressure)


def test_ideal_stiffened_common_pressure_and_primitive_state():
    ideal, stiff = _materials()
    closure = TwoMaterialEOSClosure(ideal, stiff)
    state = TwoMaterialPrimitiveState(
        density_0=jnp.asarray([1.2], dtype=jnp.float64),
        density_1=jnp.asarray([950.0], dtype=jnp.float64),
        velocity=jnp.asarray([[3.0, -0.5]], dtype=jnp.float64),
        pressure=jnp.asarray([1.0e5], dtype=jnp.float64),
        alpha_0=jnp.asarray([0.3], dtype=jnp.float64),
    )
    conserved = closure.primitive_to_conserved(state)
    np.testing.assert_allclose(
        closure.conserved_to_primitive(conserved), state.as_array(), rtol=2.0e-12
    )
    np.testing.assert_allclose(closure.pressure(conserved), state.pressure, rtol=2.0e-12)
    assert jnp.all(closure.temperature(conserved) > 0.0)
    assert jnp.all(closure.sound_speed(conserved) > 0.0)


def test_pure_phase_limits_are_exact_and_finite():
    ideal, stiff = _materials()
    closure = TwoMaterialEOSClosure(ideal, stiff)
    for alpha in (0.0, 1.0):
        primitive = jnp.asarray([[2.0, 950.0, 0.25, 1.0e5, alpha]], dtype=jnp.float64)
        conserved = closure.primitive_to_conserved(primitive)
        np.testing.assert_allclose(
            closure.pressure(conserved), primitive[..., -2], rtol=2.0e-12
        )
        assert jnp.all(jnp.isfinite(closure.report(conserved).temperature))
        assert jnp.all(jnp.isfinite(closure.report(conserved).sound_speed))
        assert jnp.all(closure.admissible(conserved))


@pytest.mark.parametrize(
    ("alpha", "floor_mass_index"),
    ((1.0e-4, 0), (1.0 - 1.0e-4, 1)),
)
def test_alpha_floor_boundaries_round_trip_and_are_active(alpha, floor_mass_index):
    ideal, stiff = _materials()
    closure = TwoMaterialEOSClosure(
        ideal,
        stiff,
        alpha_floor=1.0e-4,
        mass_floor=1.0e-8,
    )
    primitive = jnp.asarray([[2.0, 950.0, 0.25, 1.0e5, alpha]], dtype=jnp.float64)

    conserved = closure.primitive_to_conserved(primitive)
    compiled_conserved = jax.jit(closure.primitive_to_conserved)(primitive)
    recovered = closure.conserved_to_primitive(conserved)
    compiled_recovered = jax.jit(closure.conserved_to_primitive)(compiled_conserved)

    assert conserved[..., floor_mass_index].item() > 0.0
    assert bool(closure.report(conserved).admissible)
    assert bool(jax.jit(closure.admissible)(compiled_conserved))
    np.testing.assert_allclose(recovered, primitive, rtol=2.0e-12)
    np.testing.assert_allclose(compiled_recovered, primitive, rtol=2.0e-12)


@pytest.mark.parametrize(
    ("alpha", "inactive_mass_index"),
    ((0.0, 0), (1.0, 1)),
)
def test_exact_zero_phase_rejects_nonzero_partial_mass(alpha, inactive_mass_index):
    ideal, stiff = _materials()
    closure = TwoMaterialEOSClosure(ideal, stiff, alpha_floor=1.0e-4)
    primitive = jnp.asarray([[2.0, 950.0, 0.25, 1.0e5, alpha]], dtype=jnp.float64)
    pure = closure.primitive_to_conserved(primitive)
    invalid = pure.at[..., inactive_mass_index].set(closure.mass_floor)

    assert not bool(closure.admissible(invalid))
    assert not bool(closure.report(invalid).admissible)
    assert not bool(jax.jit(closure.admissible)(invalid))


def test_alpha_mass_energy_and_finite_admissibility_checks_fail_closed():
    ideal, stiff = _materials()
    closure = TwoMaterialEOSClosure(
        ideal, stiff, alpha_floor=1.0e-4, mass_floor=1.0e-3, energy_floor=1.0e-3
    )
    valid = jnp.asarray([[1.0, 950.0, 0.0, 1.0e5, 0.5]], dtype=jnp.float64)
    conserved = closure.primitive_to_conserved(valid)
    assert bool(jnp.all(closure.admissible(conserved)))
    for alpha in (-0.1, 1.1, jnp.nan):
        invalid = valid.at[..., -1].set(alpha)
        assert not bool(jnp.any(closure.admissible(invalid)))
    low_mass = conserved.at[..., 0].set(1.0e-8)
    low_energy = conserved.at[..., -2].set(1.0e-8)
    assert not bool(jnp.any(closure.admissible(low_mass)))
    assert not bool(jnp.any(closure.admissible(low_energy)))
    nonfinite = conserved.at[..., -2].set(jnp.inf)
    assert not bool(jnp.any(closure.admissible(nonfinite)))


def test_jit_grad_and_dtype_are_preserved():
    ideal, stiff = _materials()
    closure = TwoMaterialEOSClosure(ideal, stiff)
    primitive = jnp.asarray([[1.2, 950.0, 0.2, 1.0e5, 0.4]], dtype=jnp.float32)
    compiled = jax.jit(closure.primitive_to_conserved)
    conserved = compiled(primitive)
    assert conserved.dtype == primitive.dtype
    assert closure.conserved_to_primitive(conserved).dtype == primitive.dtype
    derivative = jax.grad(
        lambda pressure: closure.pressure(
            closure.primitive_to_conserved(primitive.at[..., -2].set(pressure))
        ).sum()
    )(jnp.asarray(1.0e5, dtype=jnp.float32))
    assert jnp.isfinite(derivative)
    assert derivative > 0.0


def test_material_parameter_changes_change_closure_identity():
    ideal, stiff = _materials()
    first = TwoMaterialEOSClosure(ideal, stiff)
    second = TwoMaterialEOSClosure(IdealGasMaterial(1.41, 287.0), stiff)
    assert first.closure_id != second.closure_id
    floor_changed = TwoMaterialEOSClosure(ideal, stiff, alpha_floor=1.0e-8)
    assert first.closure_id != floor_changed.closure_id
    assert first.eos_id == first.closure_id
