import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax.applications.compact_objects._black_hole_thermodynamics import (
    classify_kerr,
    evaluate_fixed_angular_momentum_response,
    evaluate_fixed_angular_velocity_response,
    evaluate_kerr_entropy_temperature,
    evaluate_kerr_first_law,
    evaluate_stationary_kerr_horizon,
    KerrBranchCode,
    KerrInput,
)


jax.config.update("jax_enable_x64", True)


def test_schwarzschild_and_extremal_limits_are_exact_and_branch_explicit():
    mass = 2.0
    schwarzschild = eqx.filter_jit(evaluate_stationary_kerr_horizon)(KerrInput(mass, 0.0))
    assert int(schwarzschild.branch.code) == int(KerrBranchCode.SUBEXTREMAL)
    np.testing.assert_allclose(schwarzschild.outer_radius, 2.0 * mass, rtol=0, atol=0)
    np.testing.assert_allclose(schwarzschild.inner_radius, 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(schwarzschild.area, 16.0 * np.pi * mass**2)
    np.testing.assert_allclose(schwarzschild.irreducible_mass, mass, rtol=0, atol=0)
    np.testing.assert_allclose(schwarzschild.surface_gravity, 1.0 / (4.0 * mass))
    np.testing.assert_allclose(schwarzschild.angular_velocity, 0.0, rtol=0, atol=0)
    assert bool(schwarzschild.qualified)
    assert bool(schwarzschild.derivative_valid)

    extremal = evaluate_stationary_kerr_horizon(KerrInput(mass, mass**2))
    assert int(extremal.branch.code) == int(KerrBranchCode.EXTREMAL)
    np.testing.assert_allclose(extremal.outer_radius, mass, rtol=0, atol=0)
    np.testing.assert_allclose(extremal.inner_radius, mass, rtol=0, atol=0)
    np.testing.assert_allclose(extremal.area, 8.0 * np.pi * mass**2)
    np.testing.assert_allclose(extremal.irreducible_mass, mass / np.sqrt(2.0))
    np.testing.assert_allclose(extremal.surface_gravity, 0.0, rtol=0, atol=0)
    np.testing.assert_allclose(extremal.angular_velocity, 1.0 / (2.0 * mass))
    assert bool(extremal.finite)
    assert bool(extremal.physically_valid)
    assert bool(extremal.qualified)
    assert not bool(extremal.derivative_valid)


def test_near_extremal_evaluation_retains_the_resolved_gap_without_clipping():
    spin = np.nextafter(1.0, 0.0)
    expected_root = np.sqrt((1.0 - spin) * (1.0 + spin))
    horizon = evaluate_stationary_kerr_horizon(KerrInput(1.0, spin))
    recovered_root = 2.0 * horizon.surface_gravity * horizon.outer_radius

    assert bool(horizon.branch.subextremal)
    assert float(horizon.branch.extremality_margin) == 1.0 - spin
    np.testing.assert_allclose(recovered_root, expected_root, rtol=2.0e-15)
    assert float(horizon.surface_gravity) > 0.0
    assert bool(horizon.derivative_valid)


def test_overextremal_and_indeterminate_inputs_do_not_claim_a_horizon():
    over = evaluate_stationary_kerr_horizon(KerrInput(1.0, 1.01))
    assert bool(over.branch.overextremal)
    assert int(over.branch.code) == int(KerrBranchCode.OVEREXTREMAL)
    assert bool(over.converged)
    assert not bool(over.finite)
    assert not bool(over.physically_valid)
    assert not bool(over.qualified)
    assert bool(jnp.isnan(over.area))

    indeterminate_input = KerrInput(jnp.nan, 0.0)
    indeterminate = classify_kerr(indeterminate_input)
    assert bool(indeterminate.indeterminate)
    assert int(indeterminate.code) == int(KerrBranchCode.INDETERMINATE)
    assert indeterminate_input.input_id != KerrInput(1.0, 0.0).input_id


def test_entropy_and_temperature_require_and_retain_an_explicit_scale():
    scale = RelativityScaleContract.si()
    mass = 1_000.0
    horizon = evaluate_stationary_kerr_horizon(KerrInput(mass, 0.0))
    thermal = evaluate_kerr_entropy_temperature(horizon, scale)

    gravitational_constant = float(scale.gravitational_constant)
    speed_of_light = float(scale.speed_of_light)
    reduced_planck_constant = float(scale.reduced_planck_constant)
    boltzmann_constant = float(scale.boltzmann_constant)
    expected_entropy = (
        boltzmann_constant
        * speed_of_light**3
        * (16.0 * np.pi * mass**2)
        / (4.0 * reduced_planck_constant * gravitational_constant)
    )
    expected_temperature = (
        reduced_planck_constant
        * speed_of_light
        / (8.0 * np.pi * boltzmann_constant * mass)
    )
    np.testing.assert_allclose(thermal.entropy, expected_entropy, rtol=3.0e-15)
    np.testing.assert_allclose(thermal.temperature, expected_temperature, rtol=3.0e-15)
    np.testing.assert_allclose(
        thermal.surface_gravity_acceleration,
        speed_of_light**2 / (4.0 * mass),
        rtol=3.0e-15,
    )
    assert thermal.scale.scale_id == scale.scale_id
    assert thermal.scale.entropy_unit == scale.entropy_unit
    assert thermal.scale.temperature_unit == scale.temperature_unit
    assert bool(thermal.qualified)

    implicit_quantum_scale = RelativityScaleContract(
        scale.dimensional_scale,
        scale.gravitational_constant,
        scale.speed_of_light,
        scale.reduced_planck_constant,
        scale.boltzmann_constant,
        quantum_constants_explicit=False,
    )
    with pytest.raises(ValueError, match="explicitly declared"):
        evaluate_kerr_entropy_temperature(horizon, implicit_quantum_scale)


def test_first_law_jvp_and_smarr_evidence_are_directional_and_qualified():
    parameters = KerrInput(2.0, 0.8)
    evidence = eqx.filter_jit(evaluate_kerr_first_law)(parameters, 0.3, -0.2)

    assert bool(evidence.finite)
    assert bool(evidence.converged)
    assert bool(evidence.physically_valid)
    assert bool(evidence.qualified)
    assert bool(evidence.derivative_valid)
    assert bool(evidence.first_law_satisfied)
    assert bool(evidence.smarr_satisfied)
    np.testing.assert_allclose(evidence.first_law_residual, 0.0, atol=2.0e-15)
    np.testing.assert_allclose(evidence.smarr_residual, 0.0, atol=2.0e-15)

    extremal = evaluate_kerr_first_law(KerrInput(2.0, 4.0), 1.0, 0.0)
    assert bool(extremal.physically_valid)
    assert not bool(extremal.derivative_valid)
    assert not bool(extremal.qualified)
    assert not bool(extremal.first_law_satisfied)
    assert bool(extremal.smarr_satisfied)
    assert bool(jnp.isnan(extremal.area_tangent))
    np.testing.assert_allclose(extremal.smarr_residual, 0.0, atol=2.0e-15)


def test_fixed_angular_momentum_response_exposes_davies_singularity():
    mass = 2.0
    schwarzschild = evaluate_fixed_angular_momentum_response(KerrInput(mass, 0.0))
    np.testing.assert_allclose(
        schwarzschild.mass_temperature_response, -8.0 * np.pi * mass**2
    )
    np.testing.assert_allclose(
        schwarzschild.thermal_mass_response,
        schwarzschild.mass_temperature_response,
        rtol=0,
        atol=0,
    )
    assert not bool(schwarzschild.singular)
    assert bool(schwarzschild.qualified)

    root = np.sqrt(3.0) - 1.0
    spin = np.sqrt(1.0 - root**2)
    davies = evaluate_fixed_angular_momentum_response(
        KerrInput(1.0, spin), conditioning_tolerance=1.0e-10
    )
    assert bool(davies.physically_valid)
    assert bool(davies.converged)
    assert bool(davies.singular)
    assert not bool(davies.finite)
    assert not bool(davies.qualified)
    assert not bool(davies.derivative_valid)
    assert bool(jnp.isnan(davies.mass_temperature_response))


def test_fixed_angular_velocity_response_separates_heat_and_rotational_work():
    mass = 2.0
    spin = 0.6
    response = evaluate_fixed_angular_velocity_response(KerrInput(mass, spin * mass**2))
    root = np.sqrt(1.0 - spin**2)

    np.testing.assert_allclose(
        response.mass_temperature_response,
        -4.0 * np.pi * mass**2 * (1.0 + root),
    )
    np.testing.assert_allclose(
        response.thermal_mass_response,
        -2.0 * np.pi * mass**2 * root * (1.0 + root) ** 2,
    )
    np.testing.assert_allclose(
        response.mass_temperature_response,
        response.thermal_mass_response + response.rotational_mass_response,
        rtol=2.0e-15,
    )
    np.testing.assert_allclose(response.first_law_response_residual, 0.0, atol=2.0e-14)
    assert bool(response.qualified)
    assert bool(response.derivative_valid)

    extremal = evaluate_fixed_angular_velocity_response(KerrInput(mass, mass**2))
    assert bool(extremal.singular)
    assert not bool(extremal.derivative_valid)
    assert not bool(extremal.qualified)
