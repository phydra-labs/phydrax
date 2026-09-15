#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from phydrax.discretization import (
    AxisDiscretization,
    AxisDomain,
    FourierAxisSpec,
    PreparedTensorGrid,
    TensorGridPlan,
    UniformAxisSpec,
)
from phydrax.geometry import RigidFrame
from phydrax.optics.wave._fields import (
    PlaneFieldSpace,
    ScalarPlaneField,
    TangentialPlaneField,
)
from phydrax.optics.wave._fresnel import (
    DirectFresnelPlan,
    FresnelPropagationStatus,
    prepare_direct_fresnel,
    propagate_direct_fresnel,
)


def _finite_space(shape, bounds, frame=None):
    grid = TensorGridPlan(
        tuple(UniformAxisSpec(size) for size in shape), axis_names=("u", "v")
    ).prepare(jnp.asarray(bounds))
    return PlaneFieldSpace(
        grid,
        RigidFrame.identity(3) if frame is None else frame,
        "finite-window",
    )


def test_identical_space_zero_distance_is_exact_identity_for_both_polarizations():
    space = _finite_space((17, 18), ((-2.0, -1.5), (2.0, 1.5)))
    coordinates = space.transverse_coordinates
    scalar_values = jnp.exp(
        -(coordinates[..., 0] ** 2 + coordinates[..., 1] ** 2)
    ) * jnp.exp(0.2j * coordinates[..., 0])
    prepared = prepare_direct_fresnel(DirectFresnelPlan(space, space))
    scalar = ScalarPlaneField(space, scalar_values, 9.0, -0.3)
    tangential = TangentialPlaneField(
        space,
        jnp.stack((scalar_values, 0.4j * scalar_values), axis=-1),
        9.0,
        -0.3,
    )

    for field in (scalar, tangential):
        result = propagate_direct_fresnel(prepared, field, 0.0, 12.0)
        assert jnp.array_equal(result.field.values, field.values)
        assert result.field.longitudinal_coordinate == field.longitudinal_coordinate
        assert result.successful
        assert result.status == int(FresnelPropagationStatus.SUCCESS)
        assert result.evidence.relative_power_error == 0.0


def test_different_grid_direct_fresnel_reproduces_gaussian_beam_width_and_power():
    input_space = _finite_space((81, 83), ((-4.0, -4.0), (4.0, 4.0)))
    output_space = _finite_space((101, 103), ((-6.0, -6.0), (6.0, 6.0)))
    coordinates = input_space.transverse_coordinates
    waist = 0.7
    values = jnp.exp(-(coordinates[..., 0] ** 2 + coordinates[..., 1] ** 2) / waist**2)
    field = ScalarPlaneField(input_space, values, 15.0, 0.0)
    wavenumber = 20.0
    distance = 5.0
    prepared = prepare_direct_fresnel(
        DirectFresnelPlan(
            input_space,
            output_space,
            maximum_sampling_phase_step=100.0,
            maximum_paraxial_angle=1.5,
            maximum_power_error=2.0e-2,
        )
    )

    result = prepared.execute(field, distance, wavenumber)
    output_intensity = jnp.abs(result.field.values) ** 2
    output_weights = output_space.area_weights
    output_coordinates = output_space.transverse_coordinates
    normalization = jnp.sum(output_weights * output_intensity)
    rms_radius_squared = (
        jnp.sum(
            output_weights * output_intensity * jnp.sum(output_coordinates**2, axis=-1)
        )
        / normalization
    )
    rayleigh_range = 0.5 * wavenumber * waist**2
    expected_radius_squared = 0.5 * waist**2 * (1.0 + (distance / rayleigh_range) ** 2)

    assert result.field.space.space_id == output_space.space_id
    assert result.field.values.shape == output_space.shape
    assert jnp.allclose(rms_radius_squared, expected_radius_squared, rtol=1.5e-2)
    assert result.evidence.relative_power_error < 2.0e-2
    assert result.successful


def test_direct_fresnel_distance_gradient_is_finite_and_nonzero():
    input_space = _finite_space((25, 27), ((-2.0, -2.0), (2.0, 2.0)))
    output_space = _finite_space((29, 31), ((-2.5, -2.5), (2.5, 2.5)))
    coordinates = input_space.transverse_coordinates
    field = ScalarPlaneField(
        input_space,
        jnp.exp(-jnp.sum(coordinates**2, axis=-1)),
        10.0,
        0.0,
    )
    prepared = prepare_direct_fresnel(
        DirectFresnelPlan(
            input_space,
            output_space,
            maximum_sampling_phase_step=100.0,
            maximum_paraxial_angle=1.5,
            maximum_power_error=1.0,
        )
    )

    def objective(distance):
        propagated = propagate_direct_fresnel(prepared, field, distance, 16.0)
        return jnp.real(propagated.field.values[14, 15])

    derivative = jax.grad(objective)(jnp.asarray(4.0))
    assert jnp.isfinite(derivative)
    assert derivative != 0.0


def test_prepare_rejects_unsupported_topology_grid_frame_and_resources():
    finite = _finite_space((9, 10), ((-1.0, -1.0), (1.0, 1.0)))
    periodic_grid = TensorGridPlan(
        (FourierAxisSpec(8), FourierAxisSpec(8)), axis_names=("u", "v")
    ).prepare(jnp.asarray([[-1.0, -1.0], [1.0, 1.0]]))
    periodic = PlaneFieldSpace(periodic_grid, RigidFrame.identity(3), "periodic-cell")
    shifted_frame = RigidFrame(jnp.eye(3), jnp.asarray([0.1, 0.0, 0.0]))
    shifted = _finite_space((9, 10), ((-1.0, -1.0), (1.0, 1.0)), frame=shifted_frame)
    nonuniform_axis = AxisDiscretization(
        nodes=jnp.asarray([-1.0, -0.4, 0.2, 1.0]),
        quad_weights=jnp.asarray([0.3, 0.6, 0.7, 0.4]),
        basis="nonuniform",
        domain=AxisDomain.interval(-1.0, 1.0),
        lower_endpoint_included=True,
        upper_endpoint_included=True,
    )
    uniform_axis = UniformAxisSpec(5).materialize(jnp.asarray(-1.0), jnp.asarray(1.0))
    nonuniform = PlaneFieldSpace(
        PreparedTensorGrid((nonuniform_axis, uniform_axis), axis_names=("u", "v")),
        RigidFrame.identity(3),
        "finite-window",
    )

    with pytest.raises(ValueError, match="finite-window"):
        DirectFresnelPlan(periodic, finite)
    with pytest.raises(ValueError, match="uniform point axes"):
        prepare_direct_fresnel(DirectFresnelPlan(nonuniform, finite))
    with pytest.raises(ValueError, match="identical.*frames"):
        prepare_direct_fresnel(DirectFresnelPlan(finite, shifted))
    with pytest.raises(ValueError, match="maximum_kernel_elements"):
        prepare_direct_fresnel(
            DirectFresnelPlan(finite, finite, maximum_kernel_elements=10)
        )
    with pytest.raises(ValueError, match="maximum_workspace_bytes"):
        prepare_direct_fresnel(
            DirectFresnelPlan(finite, finite, maximum_workspace_bytes=128)
        )


def test_zero_distance_different_spaces_and_accuracy_limits_are_explicit_failures():
    input_space = _finite_space((13, 15), ((-1.0, -1.0), (1.0, 1.0)))
    output_space = _finite_space((17, 19), ((-1.5, -1.5), (1.5, 1.5)))
    field = ScalarPlaneField(input_space, jnp.ones(input_space.shape), 7.0, 0.0)
    prepared = prepare_direct_fresnel(
        DirectFresnelPlan(
            input_space,
            output_space,
            maximum_sampling_phase_step=1.0e-4,
            maximum_paraxial_angle=1.0e-4,
            maximum_power_error=0.0,
        )
    )

    zero = prepared.execute(field, 0.0, 10.0)
    propagated = prepared.execute(field, 0.5, 10.0)

    assert not zero.successful
    assert zero.status & int(FresnelPropagationStatus.ZERO_DISTANCE_SPACE_MISMATCH)
    assert not propagated.successful
    assert propagated.status & int(FresnelPropagationStatus.SAMPLING_LIMIT)
    assert propagated.status & int(FresnelPropagationStatus.PARAXIAL_LIMIT)
    assert propagated.status & int(FresnelPropagationStatus.POWER_LIMIT)
