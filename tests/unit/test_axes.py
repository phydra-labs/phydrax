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
    plan = phx.axes.PairwiseAxisContractionPlan(
        matrix.layout,
        vector.layout,
        ((source_ref, vector_ref),),
    )
    result = jax.jit(plan.apply)(matrix, vector)

    np.testing.assert_allclose(result.data, jnp.asarray((-1.0, 1.0, 3.0)))
    assert result.layout == phx.axes.AxisLayout((target_ref,))


def test_alignment_preserves_namespaced_axis_references():
    left_axis = phx.axes.Axis(phx.axes.AxisKey("left", "sample"), 2)
    right_axis = phx.axes.Axis(phx.axes.AxisKey("right", "sample"), 3)
    left_ref = left_axis.ref(slot="source")
    right_ref = right_axis.ref(slot="target")
    field = phx.axes.AxisArray(
        jnp.arange(6.0).reshape((2, 3)),
        axes=phx.axes.AxisLayout((left_ref, right_ref)),
    )

    reordered = field.order_as(right_ref, left_ref)
    np.testing.assert_allclose(reordered.data, field.data.T)
    assert reordered.layout == phx.axes.AxisLayout((right_ref, left_ref))
    with pytest.raises(ValueError, match="ambiguous"):
        field.order_as("sample", "sample")

    plan = phx.axes.AxisAlignmentPlan(
        field.layout,
        phx.axes.AxisLayout((right_ref, left_ref)),
    )
    aligned = plan.apply(field)
    np.testing.assert_allclose(aligned.data, field.data.T)
    assert aligned.layout == plan.target


def test_broadcast_like_keeps_disjoint_namespaces_distinct():
    left_ref = phx.axes.Axis(phx.axes.AxisKey("left", "sample"), 2).ref()
    right_ref = phx.axes.Axis(phx.axes.AxisKey("right", "sample"), 3).ref()
    left = phx.axes.AxisArray(
        jnp.asarray((1.0, 2.0)),
        axes=phx.axes.AxisLayout((left_ref,)),
    )
    right = phx.axes.AxisArray(
        jnp.asarray((3.0, 4.0, 5.0)),
        axes=phx.axes.AxisLayout((right_ref,)),
    )

    result = left.broadcast_like(right)

    assert result.layout == phx.axes.AxisLayout((left_ref, right_ref))
    np.testing.assert_allclose(
        result.data,
        jnp.asarray(((1.0, 1.0, 1.0), (2.0, 2.0, 2.0))),
    )


def test_factorized_contraction_transposes_into_declared_axis_order():
    tensor = jnp.arange(6.0).reshape((2, 3, 1, 1))
    factor = phx.integration.AxisFactor("factor", tensor, ("b", "a"))
    plan = phx.integration.AxisContractionPlan(
        (phx.integration.AxisProductTerm(("factor",)),),
        output_axes=("a", "b"),
    )

    result = phx.integration.contract_axis_factors({"factor": factor}, plan)

    assert result.axes == ("a", "b")
    np.testing.assert_allclose(
        result.data,
        jnp.transpose(tensor, (1, 0, 2, 3)).sum(axis=-2),
    )
