#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.fem import (
    FiniteElementMortarMetricData,
    serial_finite_element_mortar_plan,
)
from phydrax.equations.fem._executor import execute_finite_element_mortar_flux
from phydrax.equations.fem._worksets import CompiledWorkset, WorksetSignature


def _mortar_and_metric():
    left_nodes = np.linspace(-1.0, 1.0, 4)
    right_nodes = np.linspace(-1.0, 1.0, 2)
    quadrature, weights = np.polynomial.legendre.leggauss(6)
    coordinates = np.stack((quadrature, 0.2 * quadrature**2), axis=1)
    measure = np.sqrt(1.0 + (0.4 * quadrature) ** 2)
    mortar = serial_finite_element_mortar_plan(
        left_nodes,
        right_nodes,
        left_nodes,
        quadrature,
        weights,
        right_orientation=np.asarray((1, 0), dtype=np.int32),
        declared_reproduction_degree=1,
        left_physical_coordinates=coordinates,
        right_physical_coordinates=coordinates,
        coordinate_measure=measure,
        interface_id="asymmetric-hp-interface",
    )
    physical_weights = jnp.asarray(weights * measure)
    tangent = jnp.stack(
        (jnp.ones_like(jnp.asarray(quadrature)), 0.4 * quadrature), axis=1
    )
    normal = jnp.stack((tangent[:, 1], -tangent[:, 0]), axis=1)
    metric = FiniteElementMortarMetricData(
        coordinates,
        physical_weights,
        normal,
        -normal,
    )
    return mortar, metric


def test_asymmetric_workset_accepts_independent_side_widths_and_mortar_data():
    mortar, metric = _mortar_and_metric()
    signature = WorksetSignature(
        "interior_facet",
        "left-block",
        "quadrilateral",
        mortar.plan_id,
        {"u": 4},
        support_id="support",
        entity_set_id="hp-interface",
        reference_action_ids=("left-reference", "right-reference"),
        field_layout_ids=("u-layout",),
        geometry_action_id="mortar-geometry",
        precision_id="precision",
        ir_semantics_id="interior-action",
        local_kernel="mortar",
        neighbour_local_widths={"u": 2},
    )
    workset = CompiledWorkset(
        signature,
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
        jnp.asarray((1,), dtype=jnp.int32),
        {"u": jnp.asarray(((0, 1, 2, 3),), dtype=jnp.int32)},
        neighbour_gathers={"u": jnp.asarray(((4, 5),), dtype=jnp.int32)},
        mortar=mortar,
        mortar_metric=metric,
    )

    assert dict(workset.gathers)["u"].shape == (1, 4)
    assert dict(workset.neighbour_gathers)["u"].shape == (1, 2)
    assert workset.mortar_metric.opposite_normal_error == 0.0


def test_mortar_flux_is_conservative_and_preserves_jvp_vjp_transposes():
    mortar, metric = _mortar_and_metric()
    workset = SimpleNamespace(mortar=mortar, mortar_metric=metric)

    def action(left, right):
        return execute_finite_element_mortar_flux(
            workset,
            left,
            right,
            lambda plus, minus, points, weights, normal, context: 0.5 * (plus + minus),
            None,
        )

    left = jnp.asarray((0.2, 0.5, 0.8, 1.1))
    right = jnp.asarray((1.0, 0.0))
    left_lift, right_lift = action(left, right)
    np.testing.assert_allclose(
        jnp.sum(left_lift) + jnp.sum(right_lift),
        0.0,
        atol=2.0e-13,
    )

    tangent_left = jnp.asarray((0.3, -0.2, 0.1, 0.4))
    tangent_right = jnp.asarray((-0.5, 0.7))
    _, tangent = jax.jvp(action, (left, right), (tangent_left, tangent_right))
    cotangent = (
        jnp.linspace(-0.4, 0.6, left_lift.size),
        jnp.linspace(0.2, -0.3, right_lift.size),
    )
    _, pullback = jax.vjp(action, left, right)
    cotangent_left, cotangent_right = pullback(cotangent)
    np.testing.assert_allclose(
        jnp.vdot(tangent[0], cotangent[0]) + jnp.vdot(tangent[1], cotangent[1]),
        jnp.vdot(tangent_left, cotangent_left) + jnp.vdot(tangent_right, cotangent_right),
        atol=2.0e-13,
    )
