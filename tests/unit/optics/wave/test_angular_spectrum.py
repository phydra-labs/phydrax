#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization import FourierAxisSpec, TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    AngularSpectrumPlan,
    AngularSpectrumStatus,
    PlaneFieldSpace,
    propagate_angular_spectrum,
    ScalarPlaneField,
    TangentialPlaneField,
)


def _periodic_space(shape=(16, 18), bounds=((-jnp.pi, -jnp.pi), (jnp.pi, jnp.pi))):
    grid = TensorGridPlan(
        tuple(FourierAxisSpec(size) for size in shape),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(bounds))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")


def _finite_space(shape=(31, 32)):
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(size) for size in shape),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray([[-2.0, -2.0], [2.0, 2.0]]))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "finite-window")


def test_topology_requires_periodic_no_pad_or_finite_explicit_pad_crop():
    with pytest.raises(ValueError, match="requires padding=None"):
        AngularSpectrumPlan(2).prepare(_periodic_space())
    with pytest.raises(ValueError, match="requires explicit positive padding"):
        AngularSpectrumPlan().prepare(_finite_space())


def test_zero_distance_is_identity_on_periodic_cell():
    space = _periodic_space()
    coordinates = space.transverse_coordinates
    values = jnp.exp(-(coordinates[..., 0] ** 2 + coordinates[..., 1] ** 2))
    field = ScalarPlaneField(space, values, 10.0, -0.25)
    prepared = AngularSpectrumPlan().prepare(space)

    result = propagate_angular_spectrum(prepared, field, 0.0, 12.0 + 0.0j)

    assert jnp.allclose(result.field.values, field.values, rtol=1e-6, atol=1e-6)
    assert result.field.longitudinal_coordinate == -0.25
    assert result.successful
    assert result.status == int(AngularSpectrumStatus.SUCCESS)
    assert jnp.allclose(result.cropped_energy, 0.0)


def test_single_fourier_mode_accumulates_exact_longitudinal_phase():
    space = _periodic_space(shape=(18, 20))
    coordinates = space.transverse_coordinates
    transverse_wavevector = jnp.asarray([2.0, -3.0])
    values = jnp.exp(1j * jnp.sum(coordinates * transverse_wavevector, axis=-1))
    field = ScalarPlaneField(space, values, 13.0, 0.0)
    distance = 0.37
    medium_wavenumber = 8.0
    result = (
        AngularSpectrumPlan().prepare(space).execute(field, distance, medium_wavenumber)
    )
    longitudinal_wavenumber = jnp.sqrt(
        medium_wavenumber**2 - jnp.sum(transverse_wavevector**2)
    )
    expected = values * jnp.exp(1j * longitudinal_wavenumber * distance)

    assert jnp.allclose(result.field.values, expected, rtol=2e-5, atol=2e-5)


def test_complex_medium_wavenumber_has_explicit_phase_and_attenuation():
    space = _periodic_space(shape=(10, 12))
    field = ScalarPlaneField(space, jnp.ones(space.shape), 13.0, 0.0)
    distance = 0.7
    medium_wavenumber = 6.0 + 0.2j

    result = (
        AngularSpectrumPlan().prepare(space).execute(field, distance, medium_wavenumber)
    )

    expected = jnp.exp(1j * medium_wavenumber * distance)
    assert jnp.allclose(result.field.values, expected, rtol=2e-5, atol=2e-5)


def test_outgoing_branch_decays_evanescent_mode():
    space = _periodic_space(shape=(16, 16))
    coordinates = space.transverse_coordinates
    transverse_wavenumber = 3.0
    values = jnp.exp(1j * transverse_wavenumber * coordinates[..., 0])
    field = ScalarPlaneField(space, values, 4.0, 0.0)
    distance = 0.4
    medium_wavenumber = 2.0
    result = (
        AngularSpectrumPlan().prepare(space).execute(field, distance, medium_wavenumber)
    )
    expected_decay = jnp.exp(
        -jnp.sqrt(transverse_wavenumber**2 - medium_wavenumber**2) * distance
    )

    assert jnp.allclose(
        result.field.values,
        expected_decay * values,
        rtol=2e-5,
        atol=2e-5,
    )


def test_finite_window_odd_even_padding_crops_to_same_grid_and_spreads_gaussian():
    space = _finite_space()
    coordinates = space.transverse_coordinates
    values = jnp.exp(
        -(coordinates[..., 0] ** 2 + coordinates[..., 1] ** 2) / (2.0 * 0.3**2)
    )
    field = ScalarPlaneField(space, values, 20.0, 0.0)
    prepared = AngularSpectrumPlan(
        ((3, 4), (4, 5)), maximum_leakage_fraction=0.05
    ).prepare(space)

    identity = prepared.execute(field, 0.0, 35.0)
    propagated = prepared.execute(field, 0.6, 35.0)

    assert prepared.working_shape == (38, 41)
    assert identity.field.values.shape == space.shape
    assert jnp.allclose(identity.field.values, values, rtol=1e-5, atol=1e-5)
    assert propagated.field.values.shape == space.shape
    assert jnp.max(jnp.abs(propagated.field.values)) < jnp.max(jnp.abs(values))
    assert propagated.successful


def test_finite_window_leakage_is_explicit_failure():
    space = _finite_space(shape=(17, 18))
    values = jnp.zeros(space.shape, dtype=complex).at[8, 9].set(1.0)
    field = ScalarPlaneField(space, values, 5.0, 0.0)
    prepared = AngularSpectrumPlan(
        ((2, 3), (3, 2)), maximum_leakage_fraction=0.0
    ).prepare(space)

    result = prepared.execute(field, 1.0, 8.0)

    assert result.leakage_fraction > 0.0
    assert not result.successful
    assert result.status & int(AngularSpectrumStatus.LEAKAGE_EXCEEDED)
    assert result.cropped_energy > 0.0


def test_dynamic_distance_gradient_and_scalar_tangential_parity():
    space = _periodic_space(shape=(12, 14))
    coordinates = space.transverse_coordinates
    values = jnp.exp(2.0j * coordinates[..., 1])
    scalar = ScalarPlaneField(space, values, 9.0, 0.0)
    tangential = TangentialPlaneField(
        space,
        jnp.stack((values, 2.0j * values), axis=-1),
        9.0,
        0.0,
    )
    prepared = AngularSpectrumPlan().prepare(space)

    scalar_result = prepared.execute(scalar, 0.2, 7.0)
    tangential_result = prepared.execute(tangential, 0.2, 7.0)
    assert jnp.allclose(
        tangential_result.field.values[..., 0], scalar_result.field.values
    )
    assert jnp.allclose(
        tangential_result.field.values[..., 1],
        2.0j * scalar_result.field.values,
    )

    def propagated_real(distance):
        return jnp.real(prepared.execute(scalar, distance, 7.0).field.values[0, 0])

    derivative = jax.grad(propagated_real)(jnp.asarray(0.2))
    assert jnp.isfinite(derivative)
    assert jnp.abs(derivative) > 0.0
