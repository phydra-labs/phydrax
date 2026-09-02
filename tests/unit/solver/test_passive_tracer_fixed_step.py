#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.finite_volume._mac_passive_tracer import (
    MACPassiveTracerMacCormackPlan,
)
from phydrax.solver._fixed_step import CallableFixedStepMethod, FixedStepResult
from phydrax.solver._passive_tracer import (
    MACPassiveTracerContinuationState,
    MACPassiveTracerFixedStepMethod,
)


def _transport(cells=12):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
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
    transport = MACPassiveTracerMacCormackPlan(operators, tracer_space).prepare()
    return discretization, transport


def _base_step(step_index, time, state, step_size, args):
    del step_index, time, step_size
    candidate = state + jnp.asarray(args["increment"], dtype=state.dtype)
    successful = jnp.asarray(args["successful"])
    return FixedStepResult(
        candidate_state=candidate,
        accepted_state=jnp.where(successful, candidate, state),
        successful=successful,
        residual=jnp.asarray(0.25, dtype=state.dtype),
        iterations=jnp.asarray(3, dtype=jnp.int32),
        work=jnp.asarray(7, dtype=jnp.int32),
        transform_applied=jnp.asarray(True),
        transform_correction_norm=jnp.asarray(0.125, dtype=state.dtype),
    )


def _base_method():
    return CallableFixedStepMethod(_base_step, "passive-tracer-test-base")


def test_fixed_step_wrapper_accepts_base_and_tracer_atomically_and_preserves_evidence():
    discretization, transport = _transport()
    dtype = transport.tracer_space.vector_space.dtype
    velocity = (jnp.zeros(discretization.face_layouts[0].shape, dtype=dtype),)
    method = MACPassiveTracerFixedStepMethod(
        _base_method(), transport, lambda state: velocity, "test-zero-velocity"
    )
    tracer = jnp.sin(2.0 * jnp.pi * discretization.cell_centers[..., 0]).astype(dtype)
    state = MACPassiveTracerContinuationState(
        base_state=jnp.asarray(2.0, dtype=dtype), tracer=tracer
    )
    result = method.step(
        jnp.asarray(0),
        jnp.asarray(0.0, dtype=dtype),
        state,
        jnp.asarray(0.1, dtype=dtype),
        {"increment": 1.0, "successful": True},
    )

    assert result.successful
    np.testing.assert_array_equal(result.accepted_state.base_state, 3.0)
    np.testing.assert_array_equal(result.accepted_state.tracer, tracer)
    np.testing.assert_array_equal(result.residual, 0.25)
    np.testing.assert_array_equal(result.iterations, 3)
    np.testing.assert_array_equal(result.work, 7 + transport.work_count)
    assert result.transform_applied
    np.testing.assert_array_equal(result.transform_correction_norm, 0.125)
    assert method.velocity_provider_id == "test-zero-velocity"
    assert method.method_id


def test_fixed_step_wrapper_rolls_back_tracer_when_base_rejects():
    discretization, transport = _transport()
    dtype = transport.tracer_space.vector_space.dtype
    velocity = (jnp.full(discretization.face_layouts[0].shape, 0.3, dtype=dtype),)
    method = MACPassiveTracerFixedStepMethod(
        _base_method(), transport, lambda state: velocity, "test-moving-velocity"
    )
    tracer = jnp.arange(discretization.cell_shape[0], dtype=dtype)
    state = MACPassiveTracerContinuationState(
        base_state=jnp.asarray(4.0, dtype=dtype), tracer=tracer
    )
    result = method.step(
        jnp.asarray(1),
        jnp.asarray(0.1, dtype=dtype),
        state,
        jnp.asarray(0.07, dtype=dtype),
        {"increment": 2.0, "successful": False},
    )

    assert not result.successful
    np.testing.assert_array_equal(result.candidate_state.base_state, 6.0)
    assert not jnp.array_equal(result.candidate_state.tracer, tracer)
    np.testing.assert_array_equal(result.accepted_state.base_state, state.base_state)
    np.testing.assert_array_equal(result.accepted_state.tracer, state.tracer)


def test_fixed_step_wrapper_rolls_back_successful_base_when_tracer_fails():
    discretization, transport = _transport()
    dtype = transport.tracer_space.vector_space.dtype
    nonfinite_velocity = (
        jnp.full(discretization.face_layouts[0].shape, jnp.nan, dtype=dtype),
    )
    method = MACPassiveTracerFixedStepMethod(
        _base_method(),
        transport,
        lambda state: nonfinite_velocity,
        "test-nonfinite-velocity",
    )
    tracer = jnp.linspace(0.0, 1.0, discretization.cell_shape[0], dtype=dtype)
    state = MACPassiveTracerContinuationState(
        base_state=jnp.asarray(5.0, dtype=dtype), tracer=tracer
    )
    result = method.step(
        jnp.asarray(2),
        jnp.asarray(0.2, dtype=dtype),
        state,
        jnp.asarray(0.05, dtype=dtype),
        {"increment": 3.0, "successful": True},
    )

    assert not result.successful
    np.testing.assert_array_equal(result.candidate_state.base_state, 8.0)
    np.testing.assert_array_equal(result.accepted_state.base_state, state.base_state)
    np.testing.assert_array_equal(result.accepted_state.tracer, state.tracer)


def test_fixed_step_wrapper_requires_explicit_velocity_provider_identity():
    _, transport = _transport()
    with pytest.raises(ValueError, match="velocity_provider_id"):
        MACPassiveTracerFixedStepMethod(_base_method(), transport, lambda state: (), "")
