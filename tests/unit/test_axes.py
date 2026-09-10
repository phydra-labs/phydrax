#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def test_axis_identity_layout_alignment_and_reduction():
    case = phx.axes.Axis(phx.axes.AxisKey("experiment", "case"), 3)
    component = phx.axes.Axis(phx.axes.AxisKey("state", "component"), 2)
    values = phx.axes.AxisArray(
        jnp.arange(6.0).reshape((3, 2)),
        axes=phx.axes.AxisLayout((case.ref(), component.ref())),
    )
    weights = phx.axes.AxisArray(
        jnp.asarray((1.0, 2.0, 3.0)),
        axes=phx.axes.AxisLayout((case.ref(),)),
    )
    weighted = values * weights
    reduced = phx.axes.reduce_axes(weighted, (case.key,))

    assert weighted.layout == values.layout
    np.testing.assert_allclose(reduced.data, jnp.asarray((16.0, 22.0)))
    assert reduced.layout == phx.axes.AxisLayout((component.ref(),))


def test_axis_keys_prevent_equal_size_identity_collisions():
    left = phx.axes.Axis(phx.axes.AxisKey("left", "sample"), 2)
    right = phx.axes.Axis(phx.axes.AxisKey("right", "sample"), 2)
    assert left.key != right.key
    with pytest.raises(ValueError, match="repeat"):
        phx.axes.AxisLayout((left.ref(), left.ref()))


def test_axis_contraction_and_jit_preserve_layout():
    state = phx.axes.Axis(phx.axes.AxisKey("system", "state"), 2)
    target = phx.axes.Axis(phx.axes.AxisKey("system", "target"), 3)
    source_ref = state.ref(slot="source", variance="dual")
    vector_ref = state.ref(slot="source", variance="dual")
    target_ref = target.ref()
    matrix = phx.axes.AxisArray(
        jnp.arange(6.0).reshape((3, 2)),
        axes=phx.axes.AxisLayout((target_ref, source_ref)),
    )
    vector = phx.axes.AxisArray(
        jnp.asarray((2.0, -1.0)),
        axes=phx.axes.AxisLayout((vector_ref,)),
    )
    plan = phx.axes.AxisContractionPlan(
        matrix.layout,
        vector.layout,
        ((source_ref, vector_ref),),
    )
    result = jax.jit(plan.apply)(matrix, vector)

    np.testing.assert_allclose(result.data, jnp.asarray((-1.0, 1.0, 3.0)))
    assert result.layout == phx.axes.AxisLayout((target_ref,))
