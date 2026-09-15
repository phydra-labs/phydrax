#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.geometry import RigidFrame
from phydrax.optics.wave._nonlinear_response import (
    AbstractCarrierResolvedResponse,
    CarrierResolvedResponseEvaluation,
    instantaneous_nonlinear_polarization,
    InstantaneousScalarSusceptibility,
    OrientedTensorSusceptibility,
    PreparedCarrierResolvedResponse,
)
from phydrax.optics.wave._pulse_time import PulseTimeSpace


jax.config.update("jax_enable_x64", True)


def _positive_mask(count: int) -> jax.Array:
    return jnp.fft.fftfreq(count) > 0.0


def _time_space(count: int) -> PulseTimeSpace:
    grid = TensorGridPlan((FourierAxisSpec(count),), axis_names=("time",)).prepare(
        jnp.asarray([[0.0], [2.0 * jnp.pi]])
    )
    return PulseTimeSpace(grid, topology="periodic-cell")


def test_scalar_projection_keeps_sum_and_difference_frequency_mixing():
    count = 64
    samples = jnp.arange(count)
    fundamental = jnp.exp(-2j * jnp.pi * 7.0 * samples / count)
    second_harmonic = 0.2 * jnp.exp(-2j * jnp.pi * 14.0 * samples / count)
    field = fundamental + second_harmonic
    response = InstantaneousScalarSusceptibility(2.0, 0.0)

    polarization = instantaneous_nonlinear_polarization(
        response, field, _positive_mask(count)
    )
    spectrum = jnp.fft.ifft(polarization, norm="ortho")

    # E_real**2 contains both the sum-frequency fundamental square and the
    # difference-frequency product E_2 * conj(E_1).
    assert jnp.abs(spectrum[14]) > 0.0
    assert jnp.abs(spectrum[7]) > 0.0
    assert jnp.max(jnp.abs(spectrum[~_positive_mask(count)])) < 1.0e-11


def test_instantaneous_susceptibility_implements_prepared_response_contract():
    count = 32
    time_space = _time_space(count)
    mask = _positive_mask(count)
    samples = time_space.coordinates
    field = 0.8 * jnp.exp(-6j * samples)
    response = InstantaneousScalarSusceptibility(0.2, 0.03)

    prepared = response.prepare(time_space, mask, field.shape, temporal_axis=0)
    evaluation = prepared.evaluate(field)

    assert isinstance(response, AbstractCarrierResolvedResponse)
    assert isinstance(prepared, PreparedCarrierResolvedResponse)
    assert isinstance(evaluation, CarrierResolvedResponseEvaluation)
    assert jnp.allclose(
        evaluation.analytic_nonlinear_polarization,
        instantaneous_nonlinear_polarization(response, field, mask),
    )
    assert jnp.all(evaluation.analytic_free_current == 0.0)
    assert evaluation.physical_state.shape == field.shape + (0,)
    assert evaluation.evidence.initialized_at_window_entrance
    assert evaluation.successful
    assert jnp.all(evaluation.ledger.free_current_work_density == 0.0)
    assert jnp.allclose(
        evaluation.ledger.material_work_density,
        evaluation.ledger.terminal_polarization_energy_density,
    )
    assert jnp.allclose(
        evaluation.ledger.energy_closure_defect,
        evaluation.ledger.optical_work_density - evaluation.ledger.material_work_density,
    )


def test_oriented_tensor_response_is_rotation_covariant():
    chi2 = jnp.zeros((3, 3, 3), dtype=jnp.float64)
    chi2 = chi2.at[0, 1, 1].set(1.7e-12)
    chi2 = chi2.at[2, 0, 1].set(-0.4e-12)
    chi3 = jnp.zeros((3, 3, 3, 3), dtype=jnp.float64)
    chi3 = chi3.at[1, 0, 0, 2].set(2.3e-22)
    identity = RigidFrame.identity(3)
    rotation = RigidFrame.from_axis_angle((1.0, -2.0, 0.5), 0.63)
    crystal_response = OrientedTensorSusceptibility(chi2, chi3, identity)
    rotated_response = OrientedTensorSusceptibility(chi2, chi3, rotation)
    crystal_field = jnp.asarray([0.7, -0.2, 0.4], dtype=jnp.float64)
    lab_field = rotation.rotation @ crystal_field

    expected = rotation.rotation @ crystal_response.physical_polarization(crystal_field)
    actual = rotated_response.physical_polarization(lab_field)

    assert jnp.allclose(actual, expected, rtol=1.0e-10, atol=1.0e-25)


def test_zero_tensor_components_are_exactly_inactive():
    response = OrientedTensorSusceptibility(
        jnp.zeros((3, 3, 3)),
        jnp.zeros((3, 3, 3, 3)),
        RigidFrame.identity(3),
    )
    field = jnp.asarray([[0.4, -0.1, 0.2], [-0.7, 0.3, 0.5]])
    assert jnp.all(response.physical_polarization(field) == 0.0)


def test_instantaneous_response_has_smooth_field_and_susceptibility_gradients():
    count = 32
    samples = jnp.arange(count)
    carrier = jnp.exp(-2j * jnp.pi * 6.0 * samples / count)
    mask = _positive_mask(count)

    def objective(amplitude, chi3):
        response = InstantaneousScalarSusceptibility(0.0, chi3)
        polarization = instantaneous_nonlinear_polarization(
            response, amplitude * carrier, mask
        )
        return jnp.sum(jnp.abs(polarization) ** 2)

    field_gradient, susceptibility_gradient = jax.grad(objective, argnums=(0, 1))(
        jnp.asarray(0.8), jnp.asarray(2.0e-20)
    )
    assert jnp.isfinite(field_gradient)
    assert jnp.isfinite(susceptibility_gradient)
    assert field_gradient != 0.0
    assert susceptibility_gradient != 0.0
