#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.nn.models import (
    AlgebraAnalyticLayer,
    AlgebraAnalyticNetwork,
    AnalyticityOperator,
)


def _two_chart_atlas():
    first = phx.metrix.CoordinateChart("first", ("z0", "z1"))
    second = phx.metrix.CoordinateChart("second", ("w0", "w1"))
    identity = lambda value: value
    return phx.metrix.CoordinateAtlas(
        (first, second),
        (
            phx.metrix.ChartTransition(first, second, identity, inverse=identity),
            phx.metrix.ChartTransition(second, first, identity, inverse=identity),
        ),
    )


def test_cartier_divisor_overlap_units_cocycle_clearance_and_intersection():
    atlas = _two_chart_atlas()
    divisor = phx.geometry.complex.CartierDivisor(
        atlas,
        (
            phx.geometry.complex.DivisorChart(0, lambda z: z[0], component_id="z0"),
            phx.geometry.complex.DivisorChart(1, lambda z: z[0], component_id="z0"),
        ),
        {
            (0, 1): lambda z: jnp.asarray(1.0 + 0.0j),
            (1, 0): lambda z: jnp.asarray(1.0 + 0.0j),
        },
        divisor_id="coordinate-hyperplane",
    )
    point = jnp.asarray([0.3 + 0.1j, 0.4 - 0.2j])
    assert divisor.overlap_residual(0, 1, point) < 1e-8
    sampled = divisor.clearance(0, jnp.stack((point, 2.0 * point)))
    assert bool(sampled.clear)
    assert bool(sampled.sampled)
    assert not bool(sampled.certified)
    certified = divisor.clearance(
        0, jnp.stack((point, 2.0 * point)), certified_lower_bounds=jnp.asarray([0.2, 0.4])
    )
    assert bool(certified.certified)

    other = phx.geometry.complex.CartierDivisor(
        atlas,
        (
            phx.geometry.complex.DivisorChart(0, lambda z: z[1], component_id="z1"),
            phx.geometry.complex.DivisorChart(1, lambda z: z[1], component_id="z1"),
        ),
        {
            (0, 1): lambda z: jnp.asarray(1.0 + 0.0j),
            (1, 0): lambda z: jnp.asarray(1.0 + 0.0j),
        },
        divisor_id="other-coordinate-hyperplane",
    )
    intersection = divisor.intersection(other, 0, jnp.zeros((2,), dtype=complex))
    assert bool(intersection.transverse)
    assert bool(intersection.valid)


def test_algebra_analytic_network_names_operator_side_and_bracketing():
    product = phx.metrix.algebra.ComplexAlgebraSpec().prepare_product(backend="sparse")
    layer = AlgebraAnalyticLayer(
        jnp.asarray([[[1.0, 0.0]]]),
        jnp.asarray([[0.0, 0.0]]),
        product,
        side="left",
    )
    operator = AnalyticityOperator(
        "complex_holomorphic",
        lambda function, coordinate: jnp.zeros_like(function(coordinate)),
        operator_id="declared-cr",
    )
    network = AlgebraAnalyticNetwork(
        (layer,),
        lambda value: value,
        operator,
        phx.metrix.algebra.BracketingPlan(0, operand_count=1),
        network_id="complex-linear-network",
    )
    value = network(jnp.asarray([[0.2, -0.1]]))
    assert jnp.allclose(value, jnp.asarray([[0.2, -0.1]]))
    evidence = network.analyticity_evidence(jnp.asarray([[0.2, -0.1]]))
    assert evidence.operator_kind == "complex_holomorphic"
    assert evidence.side == "left"
    assert bool(evidence.valid)
    with pytest.raises(ValueError, match="kind"):
        AnalyticityOperator(
            "octonion_holomorphic", lambda function, point: point, operator_id="ambiguous"
        )


def test_gauge_renormalization_uses_group_inverse_for_reverse_state_transport():
    def certify(plan, parameters, gauge, state):
        evidence = plan.evidence(
            parameters,
            gauge,
            state,
            lambda value: jnp.asarray(0.0),
            lambda value: jnp.asarray(0.0),
        )
        assert bool(evidence.finite)
        assert bool(evidence.valid)
        assert evidence.inverse_residual < 1e-6
        assert evidence.state_residual < 1e-6

    def make_plan(action, inverse, gauge_kind):
        return phx.metrix.GaugeRenormalizationPlan(
            action,
            action,
            action,
            gauge_inverse=inverse,
            inverse_state_transport=action,
            gauge_kind=gauge_kind,
            tolerance=1e-6,
            plan_id=f"{gauge_kind}-round-trip",
        )

    scaling = make_plan(
        lambda value, gauge: value * gauge,
        lambda gauge: 1.0 / gauge,
        "positive_scaling",
    )
    certify(scaling, jnp.asarray([2.0]), jnp.asarray(2.0), jnp.asarray([3.0]))
    with pytest.raises(ValueError, match="Gauge element"):
        scaling.apply(jnp.asarray([2.0]), jnp.asarray(0.0), jnp.asarray([3.0]))

    mismatched_state = phx.metrix.GaugeRenormalizationPlan(
        lambda value, gauge: value * gauge,
        lambda value, gauge: value * gauge,
        lambda value, gauge: value * gauge,
        gauge_inverse=lambda gauge: 1.0 / gauge,
        inverse_state_transport=lambda value, gauge: {"value": value * gauge},
        gauge_kind="positive_scaling",
        plan_id="mismatched-state-tree",
    ).evidence(
        jnp.asarray([2.0]),
        jnp.asarray(2.0),
        jnp.asarray([3.0]),
        lambda value: jnp.asarray(0.0),
        lambda value: jnp.asarray(0.0),
    )
    assert not bool(mismatched_state.finite)
    assert not bool(mismatched_state.valid)

    complex_scalar = make_plan(
        lambda value, gauge: value * gauge,
        lambda gauge: jnp.conj(gauge) / jnp.abs(gauge) ** 2,
        "complex_scalar",
    )
    certify(
        complex_scalar,
        jnp.asarray(1.0 + 0.3j),
        jnp.exp(0.4j),
        jnp.asarray(-0.2 + 0.7j),
    )

    def quaternion_multiply(left, right):
        lw, lx, ly, lz = left
        rw, rx, ry, rz = right
        return jnp.asarray(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            )
        )

    quaternion = make_plan(
        lambda value, gauge: quaternion_multiply(gauge, value),
        lambda gauge: gauge * jnp.asarray((1.0, -1.0, -1.0, -1.0)) / jnp.sum(gauge**2),
        "quaternion_unit",
    )
    quaternion_gauge = jnp.asarray((jnp.cos(0.3), jnp.sin(0.3), 0.0, 0.0))
    certify(
        quaternion,
        jnp.asarray((0.2, -0.1, 0.4, 0.3)),
        quaternion_gauge,
        jnp.asarray((0.7, 0.0, -0.2, 0.1)),
    )

    angle = jnp.asarray(0.27)
    g2_gauge = (
        jnp.eye(7)
        .at[0, 0]
        .set(jnp.cos(angle))
        .at[0, 1]
        .set(-jnp.sin(angle))
        .at[1, 0]
        .set(jnp.sin(angle))
        .at[1, 1]
        .set(jnp.cos(angle))
    )
    g2 = make_plan(
        lambda value, gauge: gauge @ value,
        lambda gauge: gauge.T,
        "g2",
    )
    certify(g2, jnp.arange(7.0), g2_gauge, jnp.arange(7.0) - 1.0)

    field = phx.metrix.clifford.CliffordMetricField(
        lambda coordinates: jnp.eye(2) + 0.0 * coordinates[0],
        dimension=2,
        signature=(2, 0),
        field_id="spin-gauge-cl2",
    )
    product = phx.metrix.clifford.PreparedCliffordMetricProduct(field)
    coordinates = jnp.asarray([0.0])
    spin_gauge = jnp.asarray((jnp.cos(angle), 0.0, 0.0, jnp.sin(angle)))
    spin_action = lambda value, gauge: product(coordinates, gauge, value)
    spin = make_plan(
        spin_action,
        lambda gauge: gauge * jnp.asarray((1.0, 1.0, 1.0, -1.0)),
        "spin",
    )
    certify(
        spin,
        jnp.asarray((0.3, -0.1, 0.2, 0.4)),
        spin_gauge,
        jnp.asarray((-0.2, 0.5, 0.1, -0.3)),
    )
