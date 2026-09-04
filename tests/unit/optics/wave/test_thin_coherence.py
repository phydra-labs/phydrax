#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

from phydrax.discretization import FourierAxisSpec, TensorGridPlan
from phydrax.geometry import RigidFrame
from phydrax.optics.wave import (
    coherent_mode_intensity,
    JonesThinTransmission,
    PlaneFieldSpace,
    ScalarPlaneField,
    ScalarThinTransmission,
    TangentialPlaneField,
    thin_lens,
)


def _space(shape=(6, 8)):
    grid = TensorGridPlan(
        tuple(FourierAxisSpec(size) for size in shape),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray([[-1.0, -2.0], [1.0, 2.0]]))
    return PlaneFieldSpace(grid, RigidFrame.identity(3), "periodic-cell")


def test_scalar_thin_mask_acts_equally_on_scalar_and_tangential_fields():
    space = _space()
    coordinates = space.transverse_coordinates
    transmission = jnp.exp(0.3j * coordinates[..., 0])
    action = ScalarThinTransmission(space, transmission)
    scalar = ScalarPlaneField(space, jnp.ones(space.shape), 3.0, 0.0)
    tangential = TangentialPlaneField(
        space,
        jnp.ones(space.shape + (2,)),
        3.0,
        0.0,
    )

    scalar_output = action.apply(scalar)
    tangential_output = action.apply(tangential)
    assert jnp.allclose(scalar_output.values, transmission)
    assert jnp.allclose(tangential_output.values[..., 0], transmission)
    assert jnp.allclose(tangential_output.values[..., 1], transmission)


def test_jones_thin_action_uses_local_tangential_basis_order():
    space = _space()
    matrix = jnp.broadcast_to(
        jnp.asarray([[0.0, 1.0j], [1.0, 0.0]], dtype=complex),
        space.shape + (2, 2),
    )
    action = JonesThinTransmission(space, matrix)
    values = jnp.zeros(space.shape + (2,), dtype=complex)
    values = values.at[..., 0].set(2.0)
    field = TangentialPlaneField(space, values, 3.0, 0.0)

    output = action(field)

    assert jnp.allclose(output.values[..., 0], 0.0)
    assert jnp.allclose(output.values[..., 1], 2.0)


def test_thin_lens_factory_has_explicit_paraxial_phase():
    space = _space()
    focal_length = -2.5
    medium_wavenumber = 7.0
    lens = thin_lens(space, focal_length, medium_wavenumber)
    radius_squared = jnp.sum(space.transverse_coordinates**2, axis=-1)
    expected = jnp.exp(-0.5j * medium_wavenumber * radius_squared / focal_length)

    assert jnp.allclose(lens.transmission, expected)
    assert jnp.allclose(jnp.abs(lens.transmission), 1.0)


def test_thin_lens_phase_is_differentiable_in_focal_length():
    space = _space()

    def sampled_real_phase(focal_length):
        return jnp.real(thin_lens(space, focal_length, 7.0).transmission[1, 2])

    derivative = jax.grad(sampled_real_phase)(jnp.asarray(2.5))
    assert jnp.isfinite(derivative)
    assert jnp.abs(derivative) > 0.0


def test_coherent_mode_reduction_has_no_cross_terms():
    space = _space()
    first = ScalarPlaneField(space, jnp.ones(space.shape), 4.0, 1.0)
    second = ScalarPlaneField(space, 1.0j * jnp.ones(space.shape), 4.0, 1.0)

    intensity = coherent_mode_intensity(
        (first, second),
        jnp.asarray([2.0, 3.0]),
        jnp.asarray([True, True]),
    )

    assert jnp.allclose(intensity.values, 5.0)


def test_inactive_coherent_mode_nans_are_masked_before_arithmetic():
    space = _space()
    active = TangentialPlaneField(
        space,
        jnp.stack((jnp.ones(space.shape), 2.0 * jnp.ones(space.shape)), axis=-1),
        4.0,
        1.0,
    )
    inactive = TangentialPlaneField(
        space,
        jnp.full(space.shape + (2,), jnp.nan + 1.0j * jnp.nan),
        4.0,
        1.0,
    )

    intensity = coherent_mode_intensity(
        (active, inactive),
        jnp.asarray([2.0, jnp.nan]),
        jnp.asarray([True, False]),
    )

    assert jnp.all(jnp.isfinite(intensity.values))
    assert jnp.allclose(intensity.values, 10.0)
