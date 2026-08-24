#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.nn.operator.data import (
    FunctionSamples,
    OperatorAxis,
    OperatorBatch,
    OperatorClassificationSpec,
    OperatorFieldBatch,
    OperatorOutputSpec,
    OperatorPrediction,
    OperatorTargetBatch,
)
from phydrax.nn.operator.field import OperatorFieldSpec
from phydrax.nn.operator.training._classification_losses import (
    OperatorClassificationNLL,
    OperatorFocalClassificationLoss,
    OperatorOverlapLoss,
    OperatorSoftClassificationLoss,
)
from phydrax.nn.operator.training._fingerprint import operator_fit_schema
from phydrax.nn.operator.training._normalization import fit_operator_normalization


def _point_batch(*, mask=None, weights=None, topology=None):
    coordinates = jnp.asarray([[0.0], [0.5], [1.0]])
    source = FunctionSamples(values=jnp.zeros((3,)), coordinates=coordinates)
    query = FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=weights,
        mask=mask,
        topology=topology,
    )
    return OperatorBatch(inputs={"u": source}, queries={"query": query})


def _grid_batch():
    x = OperatorAxis(
        "x",
        jnp.asarray([0.0, 1.0]),
        quadrature_weights=jnp.asarray([0.25, 0.75]),
    )
    y = OperatorAxis(
        "y",
        jnp.asarray([0.0, 2.0]),
        quadrature_weights=jnp.asarray([1.0, 2.0]),
    )
    samples = FunctionSamples(values=jnp.zeros((2, 2)), axes=(x, y))
    return OperatorBatch(
        inputs={"u": samples},
        queries={"query": FunctionSamples(values=None, axes=(x, y))},
    )


def _topology_batch():
    graph = phx.graph.GraphIR(
        nodes={"type": jnp.asarray([0, 1, 1], dtype=jnp.int32)},
        edges={"weight": jnp.ones((2,))},
        senders=jnp.asarray([0, 1]),
        receivers=jnp.asarray([1, 2]),
        n_node=jnp.asarray([3]),
        n_edge=jnp.asarray([2]),
        node_mask=jnp.asarray([True, True, False]),
        edge_mask=jnp.asarray([True, True]),
        graph_mask=jnp.asarray([True]),
    )
    topology = phx.nn.operator.OperatorTopology.from_graph(
        graph, jnp.asarray([0, 1, -1], dtype=jnp.int32)
    )
    return _point_batch(
        weights=jnp.asarray([0.2, 0.8, 100.0]),
        topology=topology,
    )


def _paired(batch, spec, logits, target):
    predicted = OperatorFieldBatch(logits, query_name="query", spec=spec)
    truth = OperatorFieldBatch(target, query_name="query", spec=spec)
    prediction = OperatorPrediction(
        {"label": predicted},
        batch.queries,
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    targets = OperatorTargetBatch(
        {"label": truth},
        case_axes=batch.case_axes,
        case_shape=batch.case_shape,
    )
    targets.validate(batch)
    return prediction, targets


def _evaluate(loss, prediction, batch, targets):
    return loss(
        None,
        prediction,
        batch,
        targets,
        key=jr.PRNGKey(0),
        step=jnp.asarray(0),
        training=True,
        context=None,
    )


def test_output_spec_splits_prediction_and_target_shapes_without_casting_labels():
    batch = _point_batch()
    hard = OperatorClassificationSpec("multiclass", ("cold", "warm", "hot"))
    soft = OperatorClassificationSpec(
        "multiclass",
        ("cold", "warm", "hot"),
        target="soft",
    )
    hard_output = OperatorOutputSpec(3, classification=hard)
    soft_output = OperatorOutputSpec(3, classification=soft)

    assert hard_output.prediction_shape(batch) == (3, 3)
    assert hard_output.target_shape(batch) == (3,)
    assert soft_output.prediction_shape(batch) == (3, 3)
    assert soft_output.target_shape(batch) == (3, 3)
    labels = jnp.asarray([0, 2, 1], dtype=jnp.int16)
    assert hard_output.validate_target(labels, batch).dtype == jnp.int16
    with pytest.raises(ValueError, match="target shape"):
        hard_output.validate_target(jnp.eye(3), batch)
    with pytest.raises(TypeError, match="integer or Boolean"):
        hard_output.validate_target(labels.astype(float), batch)


@pytest.mark.parametrize(
    ("kind", "classes", "target", "channels", "prediction_tail", "target_tail"),
    [
        ("binary", ("no", "yes"), "hard", "scalar", (), ()),
        ("binary", ("no", "yes"), "soft", "scalar", (), ()),
        ("multiclass", ("a", "b", "c"), "hard", 3, (3,), ()),
        ("multiclass", ("a", "b", "c"), "soft", 3, (3,), (3,)),
        ("multilabel", ("a", "b", "c"), "hard", 3, (3,), (3,)),
        ("multilabel", ("a", "b", "c"), "soft", 3, (3,), (3,)),
        ("ordinal", ("low", "mid", "high"), "hard", "scalar", (), ()),
    ],
)
def test_all_classification_kinds_have_explicit_statistical_shapes(
    kind,
    classes,
    target,
    channels,
    prediction_tail,
    target_tail,
):
    thresholds = (-0.5, 0.75) if kind == "ordinal" else ()
    classification = OperatorClassificationSpec(
        kind,
        classes,
        target=target,
        thresholds=thresholds,
    )
    output = OperatorOutputSpec(channels, classification=classification)
    assert output.channel_shape == prediction_tail
    assert output.target_channel_shape == target_tail


def test_ordinal_spec_rejects_soft_targets_and_noncanonical_thresholds():
    with pytest.raises(ValueError, match="Soft ordinal"):
        OperatorClassificationSpec(
            "ordinal", ("low", "mid", "high"), target="soft", thresholds=(-1.0, 1.0)
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        OperatorClassificationSpec(
            "ordinal", ("low", "mid", "high"), thresholds=(1.0, 1.0)
        )
    with pytest.raises(TypeError, match="must be strings"):
        OperatorClassificationSpec("binary", (0, 1))


def test_json_roundtrip_is_primitive_only_and_continuous_shape_is_unchanged():
    continuous = OperatorOutputSpec(2, component_names=("x", "y"))
    assert continuous.to_dict() == {
        "channels": 2,
        "component_names": ["x", "y"],
    }
    classification = OperatorClassificationSpec(
        "ordinal",
        ("low", "medium", "high"),
        thresholds=(-0.25, 1.5),
    )
    output = OperatorOutputSpec("scalar", classification=classification)
    payload = output.to_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert OperatorOutputSpec.from_dict(payload).to_dict() == payload
    assert payload["classification"]["classes"] == ["low", "medium", "high"]


def test_classification_field_is_dimensionless_identity_and_target_only():
    classification = OperatorClassificationSpec("binary", ("off", "on"))
    output = OperatorOutputSpec("scalar", classification=classification)
    field = OperatorFieldSpec("phase", role="target", output_spec=output)
    labels = jnp.asarray([False, True])
    assert field.physical_dimension == ()
    assert field.scale == (1.0,)
    assert field.offset == (0.0,)
    assert jnp.array_equal(field.nondimensionalize(labels), labels)
    assert field.nondimensionalize(labels).dtype == labels.dtype
    with pytest.raises(ValueError, match="target-only"):
        OperatorFieldSpec("phase", role="both", output_spec=output)


def test_normalization_skips_classification_in_mixed_targets():
    batch = _point_batch(weights=jnp.asarray([1.0, 2.0, 1.0]))
    classification = OperatorClassificationSpec("binary", ("off", "on"))
    class_output = OperatorOutputSpec("scalar", classification=classification)
    continuous_output = OperatorOutputSpec("scalar")
    targets = OperatorTargetBatch(
        {
            "temperature": OperatorFieldBatch(
                jnp.asarray([1.0, 2.0, 4.0]),
                query_name="query",
                spec=continuous_output,
            ),
            "phase": OperatorFieldBatch(
                jnp.asarray([0, 1, 1], dtype=jnp.int8),
                query_name="query",
                spec=class_output,
            ),
        }
    )
    policy = fit_operator_normalization(batch, targets)
    assert set(policy.targets) == {"temperature"}
    normalized = policy.normalize_targets(targets)
    assert normalized.field("phase").values.dtype == jnp.int8
    assert jnp.array_equal(
        normalized.field("phase").values, targets.field("phase").values
    )
    prediction = OperatorPrediction(
        {
            "temperature": OperatorFieldBatch(
                jnp.asarray([1.5, 2.5, 3.5]),
                query_name="query",
                spec=continuous_output,
            ),
            "phase": OperatorFieldBatch(
                jnp.asarray([-2.0, 0.0, 2.0]),
                query_name="query",
                spec=class_output,
            ),
        },
        batch.queries,
    )
    normalized_prediction = policy.normalize_prediction(prediction)
    assert jnp.array_equal(
        normalized_prediction.field("phase").values,
        prediction.field("phase").values,
    )


def test_nll_uses_point_grid_and_topology_geometry_masks_and_quadrature():
    classification = OperatorClassificationSpec("binary", ("off", "on"))
    spec = OperatorOutputSpec("scalar", classification=classification)
    loss_mean = OperatorClassificationNLL(classification, zero_measure="zero")
    loss_integral = OperatorClassificationNLL(
        classification,
        support_reduction="integral",
        zero_measure="zero",
    )

    point = _point_batch(
        mask=jnp.asarray([True, True, False]),
        weights=jnp.asarray([1.0, 3.0, 100.0]),
    )
    point_prediction, point_targets = _paired(
        point,
        spec,
        jnp.zeros((3,)),
        jnp.asarray([0, 1, 99], dtype=jnp.int32),
    )
    assert jnp.allclose(
        _evaluate(loss_mean, point_prediction, point, point_targets), jnp.log(2.0)
    )
    assert jnp.allclose(
        _evaluate(loss_integral, point_prediction, point, point_targets),
        4.0 * jnp.log(2.0),
    )

    grid = _grid_batch()
    grid_prediction, grid_targets = _paired(
        grid, spec, jnp.zeros((2, 2)), jnp.zeros((2, 2), dtype=int)
    )
    assert jnp.allclose(
        _evaluate(loss_integral, grid_prediction, grid, grid_targets),
        3.0 * jnp.log(2.0),
    )

    topology = _topology_batch()
    topology_prediction, topology_targets = _paired(
        topology,
        spec,
        jnp.zeros((3,)),
        jnp.asarray([0, 1, 99], dtype=int),
    )
    assert jnp.allclose(
        _evaluate(loss_mean, topology_prediction, topology, topology_targets),
        jnp.log(2.0),
    )


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel"])
def test_hard_nll_focal_and_overlap_are_finite_for_supported_kinds(kind):
    classes = ("off", "on") if kind == "binary" else ("a", "b", "c")
    classification = OperatorClassificationSpec(kind, classes)
    channels = "scalar" if kind == "binary" else len(classes)
    spec = OperatorOutputSpec(channels, classification=classification)
    batch = _point_batch(weights=jnp.asarray([0.2, 0.3, 0.5]))
    if kind == "binary":
        logits = jnp.asarray([-1.0, 0.0, 1.0])
        target = jnp.asarray([0, 1, 1], dtype=jnp.int32)
    elif kind == "multiclass":
        logits = jnp.asarray([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]])
        target = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    else:
        logits = jnp.asarray([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 1.0, 2.0]])
        target = jnp.asarray([[1, 0, 1], [0, 1, 0], [1, 1, 1]], dtype=jnp.int32)
    prediction, targets = _paired(batch, spec, logits, target)
    for loss in (
        OperatorClassificationNLL(classification),
        OperatorFocalClassificationLoss(classification),
        OperatorOverlapLoss(classification, empty="one"),
    ):
        assert jnp.isfinite(_evaluate(loss, prediction, batch, targets))


@pytest.mark.parametrize("kind", ["binary", "multiclass", "multilabel"])
def test_soft_cross_entropy_and_overlap_are_finite_for_supported_kinds(kind):
    classes = ("off", "on") if kind == "binary" else ("a", "b", "c")
    classification = OperatorClassificationSpec(kind, classes, target="soft")
    channels = "scalar" if kind == "binary" else len(classes)
    spec = OperatorOutputSpec(channels, classification=classification)
    batch = _point_batch(weights=jnp.asarray([0.2, 0.3, 0.5]))
    if kind == "binary":
        logits = jnp.asarray([-1.0, 0.0, 1.0])
        target = jnp.asarray([0.1, 0.5, 0.9])
    elif kind == "multiclass":
        logits = jnp.asarray([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]])
        target = jnp.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    else:
        logits = jnp.asarray([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, 1.0, 2.0]])
        target = jnp.asarray([[0.8, 0.1, 0.6], [0.2, 0.9, 0.1], [0.5, 0.7, 0.9]])
    prediction, targets = _paired(batch, spec, logits, target)
    for loss in (
        OperatorSoftClassificationLoss(classification),
        OperatorOverlapLoss(classification, empty="one"),
    ):
        assert jnp.isfinite(_evaluate(loss, prediction, batch, targets))


def test_soft_focal_and_overlap_have_distinct_complete_fingerprints():
    hard = OperatorClassificationSpec("multiclass", ("a", "b", "c"))
    reordered = OperatorClassificationSpec("multiclass", ("b", "a", "c"))
    soft = OperatorClassificationSpec("multiclass", ("a", "b", "c"), target="soft")
    nll = OperatorClassificationNLL(hard)
    focal = OperatorFocalClassificationLoss(hard, gamma=1.5, alpha=(1.0, 2.0, 3.0))
    soft_loss = OperatorSoftClassificationLoss(soft)
    overlap = OperatorOverlapLoss(
        hard,
        overlap="tversky",
        class_reduction="weighted",
        alpha=0.3,
        beta=0.7,
        empty="one",
    )
    assert (
        len(
            {
                nll.fingerprint,
                focal.fingerprint,
                soft_loss.fingerprint,
                overlap.fingerprint,
            }
        )
        == 4
    )
    assert nll.fingerprint != OperatorClassificationNLL(reordered).fingerprint
    assert (
        focal.fingerprint != OperatorFocalClassificationLoss(hard, gamma=2.0).fingerprint
    )
    assert overlap.fingerprint != OperatorOverlapLoss(hard, overlap="jaccard").fingerprint


def test_class_order_changes_fit_schema_for_exact_resume_rejection():
    batch = _point_batch()
    first = OperatorClassificationSpec("multiclass", ("a", "b", "c"))
    second = OperatorClassificationSpec("multiclass", ("b", "a", "c"))
    logits = jnp.zeros((3, 3))
    labels = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    _, first_targets = _paired(
        batch, OperatorOutputSpec(3, classification=first), logits, labels
    )
    _, second_targets = _paired(
        batch, OperatorOutputSpec(3, classification=second), logits, labels
    )
    first_schema = operator_fit_schema(batch, target=first_targets)
    second_schema = operator_fit_schema(batch, target=second_targets)
    assert first_schema != second_schema
    assert first_schema["targets"]["label"]["classification"]["classes"] == [
        "a",
        "b",
        "c",
    ]


def test_overlap_reduces_ratio_once_and_handles_zero_measure_explicitly():
    classification = OperatorClassificationSpec("binary", ("off", "on"))
    spec = OperatorOutputSpec("scalar", classification=classification)
    batch = _point_batch(weights=jnp.asarray([1.0, 3.0, 0.0]))
    prediction, targets = _paired(
        batch,
        spec,
        jnp.asarray([20.0, -20.0, 0.0]),
        jnp.asarray([1, 1, 0], dtype=jnp.int32),
    )
    dice = OperatorOverlapLoss(classification, overlap="dice", zero_measure="zero")
    # I=1, P=1, T=4 under physical weights, hence loss=1-2/5.
    assert jnp.allclose(
        _evaluate(dice, prediction, batch, targets),
        0.6,
        atol=1e-5,
    )

    empty_batch = _point_batch(
        mask=jnp.asarray([False, False, False]),
        weights=jnp.asarray([1.0, 1.0, 1.0]),
    )
    empty_prediction, empty_targets = _paired(
        empty_batch,
        spec,
        jnp.zeros((3,)),
        jnp.zeros((3,), dtype=jnp.int32),
    )
    assert _evaluate(dice, empty_prediction, empty_batch, empty_targets) == 0.0


def test_ordinal_hard_nll_and_overlap_use_scalar_location_and_fixed_thresholds():
    classification = OperatorClassificationSpec(
        "ordinal",
        ("low", "medium", "high"),
        thresholds=(-1.0, 1.0),
    )
    spec = OperatorOutputSpec("scalar", classification=classification)
    batch = _point_batch(weights=jnp.ones((3,)))
    prediction, targets = _paired(
        batch,
        spec,
        jnp.asarray([-2.0, 0.0, 2.0]),
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
    )
    assert jnp.isfinite(
        _evaluate(OperatorClassificationNLL(classification), prediction, batch, targets)
    )
    assert jnp.isfinite(
        _evaluate(
            OperatorOverlapLoss(classification, empty="one"),
            prediction,
            batch,
            targets,
        )
    )


def test_operator_zero_weight_skips_nonfinite_predictions():
    classification = OperatorClassificationSpec("binary", ("off", "on"))
    spec = OperatorOutputSpec("scalar", classification=classification)
    batch = _point_batch()
    prediction, targets = _paired(
        batch,
        spec,
        jnp.full((3,), jnp.nan),
        jnp.zeros((3,), dtype=jnp.int32),
    )

    assert (
        _evaluate(
            OperatorClassificationNLL(classification, weight=0.0),
            prediction,
            batch,
            targets,
        )
        == 0.0
    )
