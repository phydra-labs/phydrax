#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.skeletal_muscle.continuum import (
    Almonacid2024MaterialParameters,
    almonacid_2024_material_response,
)
from phydrax.applications.skeletal_muscle.continuum._almonacid_2024_material import (
    _active_force_length,
    _active_force_velocity,
    _passive_fiber,
)


@pytest.fixture(autouse=True)
def _double_precision():
    with jax.enable_x64(True):
        yield


def _parameters(tissue=1, fraction=0.0):
    return Almonacid2024MaterialParameters(
        maximum_fiber_stress_Pa=2e5,
        bulk_modulus_Pa=1e7,
        maximum_base_stress_Pa=2e5,
        base_scale=1.0,
        maximum_strain_rate_per_s=5.0,
        fat_bulk_modulus_Pa=1.7e6 if tissue == 1 else 0.0,
        fat_scale=0.8 if tissue == 1 else 0.0,
        fat_c1_Pa=1.3e5 if tissue == 1 else 0.0,
        fat_fraction=fraction,
        base_coefficients=(0.1990559575103343, 0.3662334826469149, 0.0)
        if tissue == 1
        else (4.6896264975, -3.4551410975, 484.92055395),
    )


def _response(
    F,
    *,
    tissue=1,
    parameters=None,
    previous=None,
    activation=0.63,
    pressure=1700.0,
    dilation=1.013,
    direction=(1.0, 0.0, 0.0),
    dt=0.01,
    dynamic=True,
):
    return almonacid_2024_material_response(
        _parameters(tissue) if parameters is None else parameters,
        F,
        F if previous is None else previous,
        pressure,
        dilation,
        activation,
        dt,
        jnp.asarray(direction),
        tissue,
        dynamic=dynamic,
    )


@pytest.mark.parametrize("stretch", (0.4, 1.75))
def test_active_length_support_retains_source_endpoint_value_and_one_sided_derivative(
    stretch,
):
    harmonics = (
        (0.642587074375392, 1.290128342448810, 0.629168420414746),
        (0.325979591577056, 5.308969899884336, -4.520101562237307),
        (0.328204247867325, 6.744187042136006, 1.689155892259429),
        (0.015388902741327, 19.823676877725276, -7.386155292116579),
        (0.139240359517525, 8.038287396059996, 2.543022326676525),
        (0.001801867529599, 32.237736486095052, -6.454098315528945),
        (0.012560837549867, 23.117614057963024, -2.643346778503341),
    )
    value = sum(a * math.sin(w * stretch + p) for a, w, p in harmonics)
    derivative = sum(a * w * math.cos(w * stretch + p) for a, w, p in harmonics)
    outside = math.nextafter(stretch, -math.inf if stretch == 0.4 else math.inf)
    actual, tangent = jax.value_and_grad(_active_force_length)(jnp.asarray(stretch))
    np.testing.assert_allclose(actual, value, rtol=0, atol=2e-15)
    np.testing.assert_allclose(tangent, derivative, rtol=0, atol=4e-15)
    assert float(_active_force_length(jnp.asarray(outside))) == 0.0
    assert float(jax.grad(_active_force_length)(jnp.asarray(outside))) == 0.0


@pytest.mark.parametrize(
    "rate,value,derivative",
    (
        (-1.2, 0.0, 0.0),
        (-0.25, 0.3503552027590582, 0.9703687419567255),
        (0.0, 1.0, 6.090887473114851),
        (0.05, 1.3743240862369306, 0.9678639805763023),
        (0.75, 1.5950466421954534, 0.0),
    ),
)
def test_velocity_knots_have_source_right_branch_value_and_derivative(
    rate, value, derivative
):
    actual, tangent = jax.value_and_grad(_active_force_velocity)(jnp.asarray(rate))
    np.testing.assert_allclose(actual, value, rtol=0, atol=2e-15)
    np.testing.assert_allclose(tangent, derivative, rtol=0, atol=2e-15)
    left = float(_active_force_velocity(jnp.asarray(rate - 1e-9)))
    right = float(_active_force_velocity(jnp.asarray(rate + 1e-9)))
    assert abs(left - value) < 2e-8
    assert abs(right - value) < 2e-8


@pytest.mark.parametrize(
    "aponeurosis,stretch,stress,tangent",
    (
        (False, 1.0, 0.0, 0.0),
        (False, 1.25, 0.1471153017268245, 1.176922413814596),
        (False, 1.5, 0.6561181989622077, 2.8951007640684696),
        (False, 1.65, 1.1, 3.023323249768765),
        (True, 0.8, 0.008, 0.01),
        (True, 1.0, 0.01, 0.01),
        (True, 1.01, 0.06168820342030671, 10.327640684061333),
        (True, 1.02, 0.22502363448694518, 22.33944552926633),
        (True, 1.15, 2.9605686155904847, 19.745861872326614),
    ),
)
def test_passive_source_knots_and_energy_primitive(aponeurosis, stretch, stress, tangent):
    stress_function = lambda x: _passive_fiber(x, aponeurosis=aponeurosis)[0]
    energy_function = lambda x: _passive_fiber(x, aponeurosis=aponeurosis)[1]
    value, derivative = jax.value_and_grad(stress_function)(jnp.asarray(stretch))
    np.testing.assert_allclose(value, stress, rtol=0, atol=3e-14)
    np.testing.assert_allclose(derivative, tangent, rtol=0, atol=3e-13)
    np.testing.assert_allclose(
        jax.grad(energy_function)(jnp.asarray(stretch)),
        stress / stretch,
        rtol=0,
        atol=3e-13,
    )
    np.testing.assert_allclose(
        energy_function(jnp.asarray(stretch - 1e-9)),
        energy_function(jnp.asarray(stretch + 1e-9)),
        rtol=0,
        atol=1e-8,
    )


@pytest.mark.parametrize("tissue,fraction", ((1, 0.37), (2, 0.0)))
def test_passive_mixed_energy_generates_stress_and_symmetric_consistent_tangent(
    tissue, fraction
):
    parameters = _parameters(tissue, fraction)
    F = jnp.asarray(((1.19, 0.12, -0.03), (0.04, 0.94, 0.08), (0.01, 0.0, 1.02)))

    def response(deformation, dilation):
        return _response(
            deformation,
            tissue=tissue,
            parameters=parameters,
            activation=0.0,
            dilation=dilation,
            dynamic=False,
        )

    def potential(deformation, dilation):
        point = response(deformation, dilation)
        return point.passive_energy_density_J_per_m3 + 1700.0 * point.volume_constraint

    point = response(F, 1.013)
    gradient = jax.grad(potential, argnums=(0, 1))(F, 1.013)
    np.testing.assert_allclose(gradient[0], point.first_piola_Pa, rtol=3e-12, atol=2e-7)
    np.testing.assert_allclose(
        gradient[1], point.dilation_residual_Pa, rtol=3e-13, atol=2e-8
    )
    stress_tangent = jax.jacfwd(lambda x: response(x, 1.013).first_piola_Pa)(F)
    hessian = jax.hessian(lambda x: potential(x, 1.013))(F)
    np.testing.assert_allclose(stress_tangent, hessian, rtol=2e-11, atol=3e-6)
    np.testing.assert_allclose(
        stress_tangent, stress_tangent.transpose(2, 3, 0, 1), rtol=2e-11, atol=3e-6
    )


def test_dynamic_stress_is_objective_and_uses_spatial_isochoric_rate_not_secant_length():
    F = jnp.asarray(((1.17, 0.12, -0.03), (0.04, 0.92, 0.08), (0.01, 0.0, 1.04)))
    previous = jnp.asarray(((1.16, 0.10, -0.01), (0.03, 0.93, 0.07), (0.0, 0.01, 1.03)))
    direction = jnp.asarray((0.8, 0.0, 0.6))
    rotation = jnp.asarray(((0.8, -0.6, 0.0), (0.6, 0.8, 0.0), (0.0, 0.0, 1.0)))
    point = _response(F, previous=previous, direction=direction)
    rotated = _response(rotation @ F, previous=rotation @ previous, direction=direction)
    spatial = np.asarray((F - previous) / 0.01) @ np.linalg.inv(F)
    d = (spatial + spatial.T) / 2
    orientation = np.linalg.det(F) ** (-1 / 3) * np.asarray(F @ direction)
    expected_rate = (
        orientation
        @ (d - np.trace(d) / 3 * np.eye(3))
        @ orientation
        / (np.linalg.norm(orientation) * 5)
    )
    np.testing.assert_allclose(point.normalized_strain_rate, expected_rate, rtol=2e-13)
    previous_length = np.linalg.det(previous) ** (-1 / 3) * np.linalg.norm(
        previous @ direction
    )
    secant = (np.linalg.norm(orientation) - previous_length) / (0.01 * 5)
    assert abs(expected_rate - secant) > 1e-5
    np.testing.assert_allclose(
        rotated.first_piola_Pa, rotation @ point.first_piola_Pa, rtol=2e-12, atol=1e-8
    )
    np.testing.assert_allclose(
        rotated.kirchhoff_Pa,
        rotation @ point.kirchhoff_Pa @ rotation.T,
        rtol=2e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        rotated.normalized_strain_rate, point.normalized_strain_rate, rtol=2e-12
    )
    np.testing.assert_allclose(
        rotated.passive_energy_density_J_per_m3,
        point.passive_energy_density_J_per_m3,
        rtol=2e-12,
    )


def test_fat_mixture_scales_actual_parts_and_bulk_without_double_normalizing_fat():
    F = jnp.diag(jnp.asarray((1.2, 0.96, 0.91)))
    pure = _response(F, parameters=_parameters(fraction=0.0))
    mixed = _response(F, parameters=_parameters(fraction=0.37))
    fat = _response(F, parameters=_parameters(fraction=1.0))
    np.testing.assert_allclose(
        mixed.kirchhoff_Pa,
        0.63 * pure.kirchhoff_Pa + 0.37 * fat.kirchhoff_Pa,
        rtol=2e-13,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        mixed.dilation_residual_Pa,
        0.63 * pure.dilation_residual_Pa + 0.37 * fat.dilation_residual_Pa,
        rtol=2e-13,
    )
    np.testing.assert_allclose(
        mixed.kirchhoff_active_Pa, 0.63 * pure.kirchhoff_active_Pa, rtol=2e-13
    )
    np.testing.assert_allclose(
        mixed.kirchhoff_passive_fiber_Pa,
        0.63 * pure.kirchhoff_passive_fiber_Pa,
        rtol=2e-13,
    )
    b = np.asarray(F @ F.T) * np.linalg.det(F) ** (-2 / 3)
    expected = 2 * 0.8 * 1.3e5 * (b - np.trace(b) / 3 * np.eye(3))
    np.testing.assert_allclose(fat.kirchhoff_base_Pa, expected, rtol=2e-13, atol=1e-8)
    np.testing.assert_allclose(fat.kirchhoff_active_Pa, 0.0, atol=0.0)
    np.testing.assert_allclose(fat.kirchhoff_passive_fiber_Pa, 0.0, atol=0.0)


def test_active_power_is_separate_from_passive_energy_and_dynamic_tangent_includes_rate():
    F = jnp.diag(jnp.asarray((1.1, 0.97, 0.96)))
    previous = 0.997 * F + jnp.diag(jnp.asarray((0.0, 0.002, 0.0)))
    active = _response(F, previous=previous)
    passive = _response(F, previous=previous, activation=0.0)
    np.testing.assert_allclose(
        active.passive_energy_density_J_per_m3,
        passive.passive_energy_density_J_per_m3,
        rtol=0,
        atol=0,
    )
    difference = active.kirchhoff_Pa - passive.kirchhoff_Pa
    np.testing.assert_allclose(
        difference, active.kirchhoff_active_Pa, rtol=2e-13, atol=1e-8
    )
    perturbation = jnp.asarray(((0.1, 0.02, 0.0), (0.0, -0.03, 0.0), (0.01, 0.0, 0.02)))
    function = lambda x: _response(x, previous=previous).first_piola_Pa
    tangent = jax.jvp(function, (F,), (perturbation,))[1]
    h = 1e-6
    finite_difference = (
        function(F + h * perturbation) - function(F - h * perturbation)
    ) / (2 * h)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-8, atol=0.02)


@pytest.mark.parametrize(
    "F",
    (
        ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((float("nan"), 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    ),
)
def test_invalid_deformation_never_becomes_finite_stress_or_zero_gradient(F):
    deformation = jnp.asarray(F)
    point = _response(deformation, previous=jnp.eye(3))
    assert not bool(point.admissible)
    assert not bool(jnp.any(jnp.isfinite(point.first_piola_Pa)))
    assert not bool(jnp.isfinite(point.passive_energy_density_J_per_m3))
    stress = lambda x: _response(x, previous=jnp.eye(3)).first_piola_Pa
    _, tangent = jax.jvp(stress, (deformation,), (jnp.ones((3, 3)),))
    gradient = jax.grad(lambda x: jnp.sum(stress(x)))(deformation)
    assert not bool(jnp.any(jnp.isfinite(tangent)))
    assert not bool(jnp.any(jnp.isfinite(gradient)))


def test_small_nonunit_direction_retains_finite_passive_energy_gradient():
    deformation = jnp.eye(3)
    response = _response(
        deformation,
        activation=0.0,
        pressure=0.0,
        dilation=1.0,
        direction=(1e-18, 0.0, 0.0),
    )
    assert bool(response.admissible)
    gradient = jax.grad(
        lambda value: (
            _response(
                value,
                activation=0.0,
                pressure=0.0,
                dilation=1.0,
                direction=(1e-18, 0.0, 0.0),
            ).passive_energy_density_J_per_m3
        )
    )(deformation)
    np.testing.assert_allclose(gradient, response.first_piola_Pa, rtol=0, atol=1e-9)


def test_invalid_mixed_state_and_aponeurosis_fat_are_rejected_inside_transforms():
    parameters = eqx.tree_at(lambda p: p.fat_fraction, _parameters(2), jnp.asarray(0.1))
    incompatible = _response(jnp.eye(3), tissue=2, parameters=parameters)
    assert not bool(incompatible.admissible)
    response = jax.jit(
        jax.vmap(lambda dilation: _response(jnp.eye(3), dilation=dilation))
    )
    batch = response(jnp.asarray((1.0, 0.0, -1.0)))
    np.testing.assert_array_equal(batch.admissible, (True, False, False))
    assert bool(jnp.all(jnp.isnan(batch.first_piola_Pa[1:])))
    invalid_parameters = eqx.tree_at(
        lambda p: p.maximum_strain_rate_per_s, _parameters(), jnp.asarray(0.0)
    )
    assert not bool(_response(jnp.eye(3), parameters=invalid_parameters).admissible)
