#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _block(bounds, shape=(33, 17)):
    return phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformAxisSpec(shape[0]),
            phx.discretization.UniformAxisSpec(shape[1]),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(bounds))


def _interface(*, orientation=None):
    return phx.discretization.BlockInterface(
        "middle",
        "left",
        "x",
        "upper",
        "right",
        "x",
        "lower",
        phx.discretization.InterfaceOrientation(1)
        if orientation is None
        else orientation,
    )


def test_conforming_multiblock_topology_certifies_physical_trace_coincidence():
    left = _block(((0.0, 0.0), (0.5, 1.0)))
    right = _block(((0.5, 0.0), (1.0, 1.0)))
    prepared = phx.discretization.MultiblockGridPlan(
        (("left", left), ("right", right)),
        (_interface(),),
    ).prepare()

    report = prepared.interface_reports[0]

    assert report.passed
    assert report.conforming
    assert report.geometry_residual < 1e-14
    assert prepared.block("left").prepared_id == left.prepared_id


def test_reflected_tangential_orientation_aligns_reversed_mapped_block():
    left_reference = _block(((0.0, 0.0), (1.0, 1.0)), shape=(17, 17))
    right_reference = _block(((0.0, 0.0), (1.0, 1.0)), shape=(17, 17))
    left = phx.discretization.MappedTensorGridPlan(
        left_reference,
        lambda q: jnp.asarray((0.5 * q[0], q[1])),
    ).prepare()
    right = phx.discretization.MappedTensorGridPlan(
        right_reference,
        lambda q: jnp.asarray((1.0 - 0.5 * q[0], 1.0 - q[1])),
    ).prepare()
    orientation = phx.discretization.InterfaceOrientation(1, flips=(True,))
    interface = phx.discretization.BlockInterface(
        "middle",
        "left",
        "x",
        "upper",
        "right",
        "x",
        "upper",
        orientation,
    )

    prepared = phx.discretization.MultiblockGridPlan(
        (("left", left), ("right", right)),
        (interface,),
    ).prepare()

    assert prepared.interface_reports[0].passed
    assert prepared.interface_reports[0].geometry_residual < 2e-12


def test_nonconforming_nested_interface_and_norm_compatible_mortar():
    left = _block(((0.0, 0.0), (0.5, 1.0)), shape=(33, 9))
    right = _block(((0.5, 0.0), (1.0, 1.0)), shape=(33, 17))
    prepared = phx.discretization.MultiblockGridPlan(
        (("left", left), ("right", right)),
        (_interface(),),
    ).prepare()
    coarse_x = left.axes[1].nodes
    fine_x = right.axes[1].nodes
    coarse_h = left.structured_axes[1].point_measures
    fine_h = right.structured_axes[1].point_measures
    interpolation = phx.discretization.NormCompatibleInterpolationPlan(
        coarse_x,
        fine_x,
        coarse_h,
        fine_h,
        interpolation_order=4,
    )
    coarse = jnp.sin(jnp.pi * coarse_x)
    fine_probe = jnp.cos(2.0 * jnp.pi * fine_x)

    prolonged = interpolation.left_to_mortar(coarse)
    restricted = interpolation.mortar_to_left(fine_probe)

    assert prepared.interface_reports[0].nesting_ratio == 2
    assert interpolation.compatibility_residual < 1e-12
    assert interpolation.constant_residual < 1e-12
    np.testing.assert_allclose(
        jnp.sum(fine_h * prolonged * fine_probe),
        jnp.sum(coarse_h * coarse * restricted),
        rtol=2e-12,
        atol=2e-12,
    )


def test_multiblock_sat_central_flux_conserves_energy_and_upwind_dissipates():
    left_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(33),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [0.5]]))
    right_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(33),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.5], [1.0]]))
    interface = phx.discretization.BlockInterface(
        "middle",
        "left",
        "x",
        "upper",
        "right",
        "x",
        "lower",
        phx.discretization.InterfaceOrientation(0),
    )
    multiblock = phx.discretization.MultiblockGridPlan(
        (("left", left_grid), ("right", right_grid)),
        (interface,),
    ).prepare()
    left_sbp = phx.discretization.SBPDerivativePlan(
        left_grid,
        "x",
        interior_order=4,
    ).prepare()
    right_sbp = phx.discretization.SBPDerivativePlan(
        right_grid,
        "x",
        interior_order=4,
    ).prepare()
    left_state = 2.0 * left_grid.axes[0].nodes
    right_state = 0.3 * 2.0 * (1.0 - right_grid.axes[0].nodes)

    rates = []
    for flux in ("central", "upwind"):
        coupling = phx.discretization.MultiblockSATCoupling(
            multiblock,
            "middle",
            left_sbp,
            right_sbp,
            1.0,
            flux=flux,
        )
        left_sat, right_sat = coupling.corrections(left_state, right_state)
        left_rhs = -coupling.local_speeds[0] * left_sbp.operator.mv(left_state) + left_sat
        right_rhs = (
            -coupling.local_speeds[1] * right_sbp.operator.mv(right_state) + right_sat
        )
        rates.append(
            2.0
            * (
                jnp.sum(left_sbp.norm_weights * left_state * left_rhs)
                + jnp.sum(right_sbp.norm_weights * right_state * right_rhs)
            )
        )
        assert coupling.stability_report.passed

    np.testing.assert_allclose(rates[0], 0.0, rtol=0.0, atol=2e-10)
    assert rates[1] < rates[0]


def test_nonconforming_multiblock_sat_uses_norm_adjoint_mortar_transfer():
    left_grid = _block(((0.0, 0.0), (0.5, 1.0)), shape=(33, 9))
    right_grid = _block(((0.5, 0.0), (1.0, 1.0)), shape=(33, 17))
    multiblock = phx.discretization.MultiblockGridPlan(
        (("left", left_grid), ("right", right_grid)),
        (_interface(),),
    ).prepare()
    left_sbp = phx.discretization.SBPDerivativePlan(
        left_grid,
        "x",
        interior_order=4,
    ).prepare()
    right_sbp = phx.discretization.SBPDerivativePlan(
        right_grid,
        "x",
        interior_order=4,
    ).prepare()
    left_x = left_grid.axes[0].nodes[:, None]
    left_y = left_grid.axes[1].nodes[None, :]
    right_x = right_grid.axes[0].nodes[:, None]
    right_y = right_grid.axes[1].nodes[None, :]
    left_state = (left_x / 0.5) * (1.0 + 0.2 * jnp.sin(jnp.pi * left_y))
    right_state = ((1.0 - right_x) / 0.5) * (0.4 + 0.1 * jnp.cos(2.0 * jnp.pi * right_y))
    coupling = phx.discretization.MultiblockSATCoupling(
        multiblock,
        "middle",
        left_sbp,
        right_sbp,
        1.0,
        flux="central",
    )

    left_sat, right_sat = coupling.corrections(left_state, right_state)
    left_rhs = -left_sbp.operator.mv(left_state) + left_sat
    right_rhs = -right_sbp.operator.mv(right_state) + right_sat
    energy_rate = 2.0 * (
        jnp.sum(left_sbp.norm_weights * left_state * left_rhs)
        + jnp.sum(right_sbp.norm_weights * right_state * right_rhs)
    )

    assert coupling.interpolation is not None
    assert coupling.interpolation.compatibility_residual < 1e-12
    np.testing.assert_allclose(energy_rate, 0.0, rtol=0.0, atol=2e-9)


def test_duplicate_physical_face_connections_are_rejected():
    left = _block(((0.0, 0.0), (0.5, 1.0)))
    right = _block(((0.5, 0.0), (1.0, 1.0)))

    with pytest.raises(ValueError, match="at most one interface"):
        phx.discretization.MultiblockGridPlan(
            (("left", left), ("right", right)),
            (_interface(), _interface()),
        )
