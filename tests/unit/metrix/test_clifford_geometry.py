#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cl = phx.metrix.clifford


def test_form_bridge_round_trip_and_chevalley_vector_action():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    chart = phx.metrix.CoordinateChart("plane", ("x", "y"))
    bridge = cl.CliffordMetricBridge(algebra, chart)
    point = jnp.asarray([0.2, -0.4])
    form = phx.metrix.DifferentialForm(
        lambda coordinates: jnp.asarray([2.0 + coordinates[0], 3.0 + coordinates[1]]),
        chart=chart,
        degree=1,
    )
    field = bridge.embed(form)

    assert jnp.allclose(bridge.extract(field, 1)(point), form(point))

    e0_form = phx.metrix.DifferentialForm(
        lambda coordinates: jnp.asarray([1.0, 0.0]),
        chart=chart,
        degree=1,
    )
    wedge = bridge.embed(phx.metrix.wedge(e0_form, form))(point)
    contraction = bridge.embed(
        phx.metrix.interior_product(lambda coordinates: jnp.asarray([1.0, 0.0]), form)
    )(point)
    full = bridge.layout
    product = cl.prepare_product(algebra, full, full, output_layout=full)
    e0 = cl.basis_blade(full, 1)
    assert jnp.allclose(product(e0, field(point)), wedge + contraction)


def test_signed_form_bridge_raises_and_lowers_indices():
    algebra = cl.CliffordAlgebraSpec((1, -1))
    chart = phx.metrix.CoordinateChart("minkowski", ("t", "x"))
    bridge = cl.CliffordMetricBridge(algebra, chart)
    form = phx.metrix.DifferentialForm(
        lambda coordinates: jnp.asarray([2.0, 3.0]),
        chart=chart,
        degree=1,
    )
    point = jnp.zeros((2,))
    embedded = bridge.embed(form)
    assert jnp.array_equal(embedded(point), jnp.asarray([0.0, 2.0, -3.0, 0.0]))
    assert jnp.array_equal(bridge.extract(embedded, 1)(point), form(point))

    with pytest.raises(ValueError, match="nondegenerate"):
        cl.CliffordMetricBridge(cl.CliffordAlgebraSpec((1, 0)), chart)


def test_standalone_boost_and_orthogonal_actions_preserve_products():
    minkowski = cl.CliffordAlgebraSpec((1, -1))
    boost = cl.lorentz_boost_action(minkowski, 1, 0.4)
    inverse = cl.lorentz_boost_action(minkowski, 1, -0.4)
    identity = boost.compose(inverse)

    assert jnp.allclose(identity.matrix, jnp.eye(2), atol=1e-12)
    report = cl.audit_clifford_action(boost)
    assert bool(report.valid)
    assert float(report.automorphism_defect) < 1e-12

    audit_set = cl.MetricIsometryAuditSet(minkowski, (boost, inverse))
    assert all(bool(value.valid) for value in cl.audit_clifford_actions(audit_set))


def test_finite_metric_group_requires_actual_composition_closure():
    algebra = cl.CliffordAlgebraSpec((1, -1))
    identity = np.eye(2)
    reflection = np.diag([1.0, -1.0])
    group = cl.FiniteMetricIsometryGroup(
        algebra,
        np.stack((identity, reflection)),
    )
    assert group.order == 2
    assert group.multiplication_table == ((0, 1), (1, 0))
    assert group.inverse_indices == (0, 1)

    boost = np.asarray(cl.lorentz_boost_action(algebra, 1, 0.3).matrix)
    with pytest.raises(ValueError, match="not uniquely closed"):
        cl.FiniteMetricIsometryGroup(algebra, np.stack((identity, boost)))


def test_three_dimensional_o3_adapter_preserves_reflection_action():
    algebra = cl.CliffordAlgebraSpec((1, 1, 1))
    representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (1, 1, 1, 1),
    )
    values = jnp.arange(8, dtype=float)
    assert jnp.array_equal(representation.from_o3(representation.to_o3(values)), values)

    reflection = jnp.diag(jnp.asarray([-1.0, 1.0, 1.0]))
    action = cl.MetricIsometryAction(algebra, reflection)
    transformed = representation.transform(values, action)
    o3 = representation.o3_representation()
    assert jnp.allclose(
        representation.to_o3(transformed),
        o3.transform(representation.to_o3(values), reflection),
    )
