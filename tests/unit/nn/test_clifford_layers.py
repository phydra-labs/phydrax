#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._trainable import partition_trainable


cl = phx.metrix.clifford


def _euclidean_actions(algebra):
    angle = 0.37
    rotation = jnp.asarray(
        [
            [jnp.cos(angle), -jnp.sin(angle)],
            [jnp.sin(angle), jnp.cos(angle)],
        ]
    )
    reflection = jnp.diag(jnp.asarray([-1.0, 1.0]))
    return cl.MetricIsometryAuditSet(
        algebra,
        (
            cl.MetricIsometryAction(algebra, rotation),
            cl.MetricIsometryAction(algebra, reflection),
        ),
    )


def test_clifford_grade_representation_round_trip_and_field_schema():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (1, 1, 1),
    )
    values = jnp.arange(12, dtype=float).reshape((3, 4))
    assert jnp.array_equal(representation.join(representation.split(values)), values)
    restored = phx.nn.operator.representations.CliffordGradeRepresentation.from_dict(
        representation.to_dict()
    )
    assert restored.representation_id == representation.representation_id

    field = phx.nn.operator.OperatorFieldSpec(
        "state",
        channels=representation.packed_size,
        representation="clifford_multivector",
        clifford_layout=representation,
        scale=(2.0, 3.0, 3.0, 4.0),
        offset=(1.0, 0.0, 0.0, 0.0),
    )
    serialized = phx.nn.operator.OperatorFieldSpec.from_dict(field.to_dict())
    assert serialized.clifford_layout is not None
    assert (
        serialized.clifford_layout.representation_id == representation.representation_id
    )

    with pytest.raises(ValueError, match="constant over the blades"):
        phx.nn.operator.OperatorFieldSpec(
            "bad_scale",
            channels=representation.packed_size,
            representation="clifford_multivector",
            clifford_layout=representation,
            scale=(1.0, 2.0, 3.0, 1.0),
        )
    with pytest.raises(ValueError, match="Non-scalar"):
        phx.nn.operator.OperatorFieldSpec(
            "bad_offset",
            channels=representation.packed_size,
            representation="clifford_multivector",
            clifford_layout=representation,
            offset=(0.0, 1.0, 0.0, 0.0),
        )


def test_grade_linear_matches_explicit_channel_mixing_with_leading_axes():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    input_representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (2, 3, 0),
    )
    output_representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (3, 2, 1),
    )
    layer = phx.nn.operator.layers.CliffordGradeLinear(
        input_representation,
        output_representation,
        key=jr.key(11),
    )
    values = jr.normal(jr.key(12), (2, 3, input_representation.packed_size))
    features = input_representation.split(values)
    leading = values.shape[:-1]
    expected_grades = []
    for grade, (grade_values, weight, output_count, layout) in enumerate(
        zip(
            features.grades,
            layer.weights,
            output_representation.multiplicities,
            output_representation.grade_layouts,
            strict=True,
        )
    ):
        if weight is None:
            mixed = jnp.zeros(
                leading + (output_count, layout.blade_count),
                dtype=values.dtype,
            )
        else:
            expanded_weight = weight.reshape((1,) * len(leading) + weight.shape + (1,))
            mixed = jnp.sum(
                expanded_weight * grade_values[..., None, :, :],
                axis=len(leading) + 1,
            )
        if grade == 0 and layer.scalar_bias is not None:
            mixed = mixed + layer.scalar_bias.reshape(
                (1,) * len(leading) + layer.scalar_bias.shape + (1,)
            )
        expected_grades.append(mixed)

    expected = output_representation.join(
        phx.nn.operator.representations.CliffordGradeFeatures(tuple(expected_grades))
    )
    result = layer(values)
    assert result.shape == expected.shape
    assert result.dtype == expected.dtype
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(jax.jit(layer)(values), expected)


def test_grade_linear_and_gate_are_euclidean_equivariant():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (2, 2, 2),
    )
    layer = phx.nn.operator.layers.CliffordGradeLinear(
        representation,
        representation,
        key=jr.key(1),
    )
    values = jr.normal(jr.key(2), (7, representation.packed_size))
    actions = _euclidean_actions(algebra)
    report = phx.nn.operator.layers.audit_clifford_equivariance(
        layer,
        values,
        representation,
        representation,
        actions,
    )
    gate_report = phx.nn.operator.layers.audit_clifford_equivariance(
        lambda value: phx.nn.operator.layers.clifford_gated_activation(
            value,
            representation,
        ),
        values,
        representation,
        representation,
        actions,
    )

    assert layer.certificate.group_scope == "orthogonal-euclidean"
    assert bool(report.valid)
    assert bool(gate_report.valid)
    trainable, _ = partition_trainable(layer)
    assert sum(leaf.size for leaf in jax.tree.leaves(trainable)) > 0


def test_geometric_product_layer_is_equivariant_and_transformable():
    algebra = cl.CliffordAlgebraSpec((1, 1))
    representation = phx.nn.operator.representations.CliffordGradeRepresentation(
        algebra,
        (2, 2, 2),
    )
    layer = phx.nn.operator.layers.CliffordGeometricProductLayer(
        representation,
        key=jr.key(3),
    )
    values = jr.normal(jr.key(4), (5, representation.packed_size))
    report = phx.nn.operator.layers.audit_clifford_equivariance(
        layer,
        values,
        representation,
        representation,
        _euclidean_actions(algebra),
        tolerance=1e-8,
    )

    assert bool(report.valid)
    assert jax.jit(layer)(values).shape == values.shape
    assert jnp.all(
        jnp.isfinite(jax.grad(lambda value: jnp.sum(layer(value) ** 2))(values))
    )


def test_indefinite_gate_and_nonuniform_product_channels_are_rejected():
    indefinite = phx.nn.operator.representations.CliffordGradeRepresentation(
        cl.CliffordAlgebraSpec((1, -1)),
        (1, 1, 1),
    )
    with pytest.raises(ValueError, match="positive-definite"):
        phx.nn.operator.layers.clifford_gated_activation(
            jnp.zeros((indefinite.packed_size,)),
            indefinite,
        )

    nonuniform = phx.nn.operator.representations.CliffordGradeRepresentation(
        cl.CliffordAlgebraSpec((1, 1)),
        (1, 2, 1),
    )
    with pytest.raises(ValueError, match="common positive latent multiplicity"):
        phx.nn.operator.layers.CliffordGeometricProductLayer(nonuniform)


def test_finite_metric_group_cannot_substitute_for_sampled_boost_set():
    algebra = cl.CliffordAlgebraSpec((1, -1))
    boost = cl.lorentz_boost_action(algebra, 1, 0.2)
    audit_set = cl.MetricIsometryAuditSet(algebra, (boost, boost.inverse()))
    assert len(audit_set.actions) == 2
    with pytest.raises(ValueError, match="not uniquely closed"):
        cl.FiniteMetricIsometryGroup(
            algebra,
            np.stack((np.eye(2), np.asarray(boost.matrix))),
        )
