#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._mac_passive_tracer import (
    MACPassiveTracerMacCormackPlan,
    MACPassiveTracerStatus,
)


def _core(*, cells=16, periodic=True, correction_strength=1.0):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    tracer_space = grid.field_space(
        "passive_tracer",
        entity_layout=discretization.cell_layout,
        dtype=operators.pressure_space.dtype,
        representation="point_value",
    )
    transport = MACPassiveTracerMacCormackPlan(
        operators,
        tracer_space,
        correction_strength=correction_strength,
    ).prepare()
    return grid, discretization, operators, tracer_space, transport


def test_passive_tracer_plan_rejects_nonperiodic_wrong_representation_and_location():
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    centered = grid.field_space(
        "tracer",
        entity_layout=discretization.cell_layout,
        dtype=operators.pressure_space.dtype,
        representation="point_value",
    )
    with pytest.raises(ValueError, match="periodic"):
        MACPassiveTracerMacCormackPlan(operators, centered)

    grid, discretization, operators, _, _ = _core(cells=8)
    cell_average = grid.field_space(
        "tracer",
        entity_layout=discretization.cell_layout,
        dtype=operators.pressure_space.dtype,
        representation="cell_average",
    )
    with pytest.raises(ValueError, match="point_value"):
        MACPassiveTracerMacCormackPlan(operators, cell_average)

    face_value = grid.field_space(
        "tracer",
        entity_layout=discretization.face_layouts[0],
        dtype=operators.pressure_space.dtype,
        representation="point_value",
    )
    with pytest.raises(ValueError, match="centered cell location"):
        MACPassiveTracerMacCormackPlan(operators, face_value)


def test_zero_velocity_and_constant_tracers_are_preserved_exactly():
    _, discretization, _, tracer_space, transport = _core()
    dtype = tracer_space.vector_space.dtype
    zero_velocity = tuple(
        jnp.zeros(layout.shape, dtype=dtype) for layout in discretization.face_layouts
    )
    tracer = jnp.arange(discretization.cell_shape[0], dtype=dtype)
    stationary = transport.advance(tracer, zero_velocity, jnp.asarray(0.25, dtype=dtype))
    np.testing.assert_array_equal(stationary.values, tracer)
    assert stationary.success
    assert int(stationary.status) == MACPassiveTracerStatus.SUCCESS

    constant = jnp.full(discretization.cell_shape, 2.75, dtype=dtype)
    moving_velocity = tuple(
        jnp.full(layout.shape, 0.37, dtype=dtype)
        for layout in discretization.face_layouts
    )
    translated = transport.advance(
        constant, moving_velocity, jnp.asarray(0.19, dtype=dtype)
    )
    np.testing.assert_array_equal(translated.values, constant)
    assert translated.differentiation == "almost_everywhere"
    assert translated.conservation == "diagnostic_only"
    assert translated.field_space_id == tracer_space.field_space_id
    assert translated.layout_id == tracer_space.layout.layout_id
    assert translated.support_id == tracer_space.support_id
    assert translated.values.shape == discretization.cell_shape
    assert translated.limiter_active.shape == discretization.cell_shape


def test_maccormack_result_is_donor_bounded_and_integral_defect_is_diagnostic():
    _, discretization, _, tracer_space, transport = _core(cells=24)
    dtype = tracer_space.vector_space.dtype
    centers = discretization.cell_centers[..., 0]
    face_points = discretization.face_centers[0][..., 0]
    tracer = jnp.exp(-80.0 * (centers - 0.35) ** 2).astype(dtype)
    velocity = ((0.25 + 0.2 * jnp.sin(2.0 * jnp.pi * face_points)).astype(dtype),)
    result = transport.advance(tracer, velocity, jnp.asarray(0.08, dtype=dtype))

    assert result.success
    assert result.donor_bounded
    assert result.donor_bound_defect <= 8.0 * jnp.finfo(dtype).eps
    assert jnp.all(result.values >= result.donor_lower_bound)
    assert jnp.all(result.values <= result.donor_upper_bound)
    assert jnp.abs(result.integral_defect) > jnp.finfo(dtype).eps
    assert result.maximum_displacement_cell_widths > 0.0


def test_zero_correction_is_one_pass_predictor_and_nonfinite_velocity_fails_closed():
    _, discretization, _, tracer_space, transport = _core(
        cells=10, correction_strength=0.0
    )
    dtype = tracer_space.vector_space.dtype
    tracer = jnp.linspace(-0.4, 1.2, discretization.cell_shape[0], dtype=dtype)
    velocity = (jnp.full(discretization.face_layouts[0].shape, 0.2, dtype=dtype),)
    predictor = transport.advance(tracer, velocity, jnp.asarray(0.04, dtype=dtype))
    np.testing.assert_allclose(predictor.values, predictor.raw_values)
    np.testing.assert_array_equal(predictor.maximum_maccormack_correction, 0.0)

    nonfinite = (jnp.full(discretization.face_layouts[0].shape, jnp.nan, dtype=dtype),)
    failed = transport.advance(tracer, nonfinite, jnp.asarray(0.04, dtype=dtype))
    assert not failed.success
    assert not failed.finite
    assert int(failed.status) == MACPassiveTracerStatus.NONFINITE


def test_passive_tracer_has_fixed_jit_shapes_and_finite_branchwise_gradient():
    _, discretization, _, tracer_space, transport = _core(cells=12)
    dtype = tracer_space.vector_space.dtype
    centers = discretization.cell_centers[..., 0]
    tracer = (0.6 + 0.2 * jnp.sin(2.0 * jnp.pi * centers)).astype(dtype)
    velocity = (jnp.full(discretization.face_layouts[0].shape, 0.1, dtype=dtype),)
    step = jnp.asarray(0.03, dtype=dtype)

    compiled = jax.jit(lambda values: transport.advance(values, velocity, step).values)
    values = compiled(tracer)
    gradient = jax.grad(lambda source: jnp.sum(compiled(source)))(tracer)

    assert values.shape == tracer.shape
    assert gradient.shape == tracer.shape
    assert jnp.all(jnp.isfinite(gradient))
