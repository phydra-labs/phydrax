#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import ast
import builtins
import copy
import importlib
import json
import math
import struct

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml.interop import (
    ConversionError,
    from_xgboost_artifact,
    UnsupportedConversionError,
)
from phydrax.ml.tree import TreeEnsemble


_TYPED_ARRAYS = {
    "base_weights": "d",
    "categories": "l",
    "categories_nodes": "l",
    "categories_segments": "L",
    "categories_sizes": "L",
    "default_left": "U",
    "left_children": "l",
    "loss_changes": "d",
    "parents": "l",
    "right_children": "l",
    "split_conditions": "d",
    "split_indices": "l",
    "split_type": "U",
    "sum_hessian": "d",
}
_TYPED_FORMATS = {"d": ">f", "l": ">i", "L": ">q", "U": ">B"}


def _scalar_tree(
    identifier,
    *,
    threshold=None,
    left_value=0.0,
    right_value=0.0,
    leaf_value=0.0,
    default_left=True,
    categories=(),
):
    if threshold is None and not categories:
        node_count = 1
        left = [-1]
        right = [-1]
        parents = [-1]
        conditions = [float(leaf_value)]
        indices = [0]
        defaults = [0]
        split_type = [0]
        category_nodes = []
        category_segments = []
        category_sizes = []
        category_values = []
    else:
        node_count = 3
        left = [1, -1, -1]
        right = [2, -1, -1]
        parents = [-1, 0, 0]
        if categories:
            split_condition = math.nan
        else:
            assert threshold is not None
            split_condition = float(threshold)
        conditions = [split_condition, float(left_value), float(right_value)]
        indices = [0, 0, 0]
        defaults = [int(default_left), 0, 0]
        split_type = [int(bool(categories)), 0, 0]
        category_nodes = [0] if categories else []
        category_segments = [0] if categories else []
        category_sizes = [len(categories)] if categories else []
        category_values = list(categories)
    return {
        "base_weights": [0.0] * node_count,
        "categories": category_values,
        "categories_nodes": category_nodes,
        "categories_segments": category_segments,
        "categories_sizes": category_sizes,
        "default_left": defaults,
        "id": identifier,
        "left_children": left,
        "loss_changes": [0.0] * node_count,
        "parents": parents,
        "right_children": right,
        "split_conditions": conditions,
        "split_indices": indices,
        "split_type": split_type,
        "sum_hessian": [2.0] + [1.0] * (node_count - 1),
        "tree_param": {
            "num_deleted": "0",
            "num_feature": "1",
            "num_nodes": str(node_count),
            "size_leaf_vector": "1",
        },
    }


def _vector_tree(identifier=0):
    return {
        "base_weights": [0.0] * 6,
        "categories": [],
        "categories_nodes": [],
        "categories_segments": [],
        "categories_sizes": [],
        "default_left": [1, 0, 0],
        "id": identifier,
        "leaf_weights": [1.0, 2.0, -1.0, 3.0],
        "left_children": [1, -1, -1],
        "loss_changes": [1.0, 0.0, 0.0],
        "parents": [-1, 0, 0],
        "right_children": [2, 0, 1],
        "split_conditions": [0.0, math.nan, math.nan],
        "split_indices": [0, 0, 0],
        "split_type": [0, 0, 0],
        "sum_hessian": [2.0, 1.0, 1.0],
        "tree_param": {
            "num_deleted": "0",
            "num_feature": "1",
            "num_nodes": "3",
            "size_leaf_vector": "2",
        },
    }


def _objective(name, num_class):
    if name in {
        "binary:logistic",
        "binary:logitraw",
        "reg:gamma",
        "reg:logistic",
        "reg:squarederror",
    }:
        return {"name": name, "reg_loss_param": {"scale_pos_weight": "1"}}
    if name in {"multi:softmax", "multi:softprob"}:
        return {
            "name": name,
            "softmax_multiclass_param": {"num_class": str(num_class)},
        }
    return {"name": name}


def _saved_model(
    trees,
    *,
    tree_info=None,
    iteration_indptr=None,
    objective="reg:squarederror",
    num_class=0,
    num_target=1,
    base_score="0E0",
    feature_type="q",
    num_parallel_tree=1,
    weight_drop=None,
):
    tree_info = [0] * len(trees) if tree_info is None else list(tree_info)
    iteration_indptr = (
        list(range(len(trees) + 1))
        if iteration_indptr is None
        else list(iteration_indptr)
    )
    booster_model = {
        "gbtree_model_param": {
            "num_parallel_tree": str(num_parallel_tree),
            "num_trees": str(len(trees)),
        },
        "iteration_indptr": iteration_indptr,
        "tree_info": tree_info,
        "trees": trees,
    }
    if weight_drop is not None:
        booster_model["weight_drop"] = list(weight_drop)
    return {
        "learner": {
            "attributes": {"training_tag": "fixture"},
            "feature_names": ["feature"],
            "feature_types": [feature_type],
            "gradient_booster": {"model": booster_model, "name": "gbtree"},
            "learner_model_param": {
                "base_score": base_score,
                "boost_from_average": "1",
                "num_class": str(num_class),
                "num_feature": "1",
                "num_target": str(num_target),
            },
            "objective": _objective(objective, num_class),
        },
        "version": [3, 0, 0],
    }


def _ubjson_key(value):
    encoded = value.encode("utf-8")
    return b"L" + struct.pack(">q", len(encoded)) + encoded


def _ubjson_integer(value):
    if -128 < value < 127:
        return b"i" + struct.pack(">b", value)
    if -32768 < value < 32767:
        return b"I" + struct.pack(">h", value)
    if -2_147_483_648 < value < 2_147_483_647:
        return b"l" + struct.pack(">i", value)
    return b"L" + struct.pack(">q", value)


def _ubjson(value, field=None):
    if type(value) is dict:
        return (
            b"{"
            + b"".join(
                _ubjson_key(key) + _ubjson(item, key)
                for key, item in sorted(value.items())
            )
            + b"}"
        )
    if type(value) is list:
        marker = _TYPED_ARRAYS.get(field)
        if marker is not None:
            payload = b"".join(
                struct.pack(_TYPED_FORMATS[marker], item) for item in value
            )
            return (
                b"[$"
                + marker.encode("ascii")
                + b"#L"
                + struct.pack(">q", len(value))
                + payload
            )
        return (
            b"[#L"
            + struct.pack(">q", len(value))
            + b"".join(_ubjson(item) for item in value)
        )
    if type(value) is str:
        return b"S" + _ubjson_key(value)
    if type(value) is bool:
        return b"T" if value else b"F"
    if type(value) is int:
        return _ubjson_integer(value)
    if type(value) is float:
        return b"d" + struct.pack(">f", value)
    if value is None:
        return b"Z"
    raise TypeError(type(value).__name__)


def _configuration(result):
    return {
        key: ast.literal_eval(value) for key, value in result.provenance.configuration
    }


def _guard_xgboost_import(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "xgboost" or name.startswith("xgboost."):
            raise AssertionError("Conversion must not import XGBoost.")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)


def test_converter_module_import_never_requests_xgboost(monkeypatch):
    _guard_xgboost_import(monkeypatch)
    from phydrax.ml.interop import _xgboost

    importlib.reload(_xgboost)


def test_scalar_regression_converts_all_json_containers_and_canonical_ubjson(
    tmp_path, monkeypatch
):
    document = _saved_model(
        [
            _scalar_tree(
                0,
                threshold=0.0,
                left_value=2.0,
                right_value=-1.0,
                default_left=True,
            )
        ],
        base_score="1E0",
    )
    text = json.dumps(document, separators=(",", ":"))
    json_path = tmp_path / "model.json"
    json_path.write_text(text, encoding="utf-8")
    sources = {
        "json-mapping": document,
        "json-text": text,
        "json-bytes": text.encode("utf-8"),
        "json-path": json_path,
        "ubjson": _ubjson(document),
    }
    expected_formats = {
        "json-mapping": "json-mapping",
        "json-text": "json",
        "json-bytes": "json",
        "json-path": "json",
        "ubjson": "ubjson",
    }
    _guard_xgboost_import(monkeypatch)

    array_checksums = set()
    for label, source in sources.items():
        result = from_xgboost_artifact(source)
        model = result.model
        native = model.model
        configuration = _configuration(result)

        assert isinstance(native, TreeEnsemble)
        assert jnp.array_equal(
            model(jnp.array([[-1.0], [0.0], [2.0], [jnp.nan]])),
            jnp.array([3.0, 0.0, 0.0, 3.0]),
        )
        assert result.provenance.source == "xgboost"
        assert result.provenance.source_version == "3.0.0"
        assert result.provenance.source_model == "gbtree"
        assert result.provenance.feature_names == ("feature",)
        assert result.provenance.license_id == "Apache-2.0"
        assert configuration["source_format"] == expected_formats[label]
        assert len(configuration["artifact_sha256"]) == 64
        assert len(configuration["array_sha256"]) == 64
        assert configuration["numeric_routing"].startswith("XGBoost strict float32")
        assert configuration["semantic_notes"][-1].startswith(
            "prediction is fully native"
        )
        array_checksums.add(configuration["array_sha256"])

    assert len(array_checksums) == 1


def test_binary_sigmoid_preserves_base_score_link_strict_tie_and_missing_direction():
    document = _saved_model(
        [
            _scalar_tree(
                0,
                threshold=1.0,
                left_value=-2.0,
                right_value=2.0,
                default_left=False,
            )
        ],
        objective="binary:logistic",
        base_score="5E-1",
    )
    result = from_xgboost_artifact(document)
    points = jnp.array([[0.0], [1.0], [jnp.nan]])
    native = result.model.model
    assert isinstance(native, TreeEnsemble)

    assert jnp.array_equal(native.predict_raw(points), jnp.array([-2.0, 2.0, 2.0]))
    assert jnp.allclose(result.model(points), jax.nn.sigmoid(jnp.array([-2.0, 2.0, 2.0])))
    assert native.target_schema.kind == "binary"
    assert _configuration(result)["base_score_margin"] == (0.0,)


def test_multiclass_tree_info_groups_margins_before_softprob():
    document = _saved_model(
        [
            _scalar_tree(0, leaf_value=1.0),
            _scalar_tree(1, leaf_value=2.0),
            _scalar_tree(2, leaf_value=-1.0),
        ],
        tree_info=[0, 1, 2],
        iteration_indptr=[0, 3],
        objective="multi:softprob",
        num_class=3,
        base_score="5E-1",
    )
    result = from_xgboost_artifact(document)
    points = jnp.array([[0.0], [4.0]])
    raw = jnp.array([[1.5, 2.5, -0.5], [1.5, 2.5, -0.5]])
    native = result.model.model
    assert isinstance(native, TreeEnsemble)

    assert jnp.array_equal(native.predict_raw(points), raw)
    assert jnp.allclose(result.model(points), jax.nn.softmax(raw, axis=-1))
    assert native.target_schema.kind == "multiclass"
    assert _configuration(result)["tree_info"] == (0, 1, 2)


def test_categorical_selected_set_routes_right_and_missing_uses_persisted_default():
    document = _saved_model(
        [
            _scalar_tree(
                0,
                left_value=10.0,
                right_value=-3.0,
                default_left=True,
                categories=(2, 4),
            )
        ],
        feature_type="c",
    )
    result = from_xgboost_artifact(document)

    assert jnp.array_equal(
        result.model(jnp.array([[2.0], [3.0], [4.0], [jnp.nan]])),
        jnp.array([-3.0, 10.0, -3.0, 10.0]),
    )
    native = result.model.model
    assert isinstance(native, TreeEnsemble)
    assert native.feature_schema.kinds == ("categorical",)
    assert native.left_child[0, 0] == 2
    assert native.right_child[0, 0] == 1
    assert not native.default_left[0, 0]
    assert _configuration(result)["categorical_features"] == (0,)


def test_vector_leaf_layout_maps_leaf_indices_to_complete_output_vectors():
    document = _saved_model(
        [_vector_tree()],
        num_target=2,
        base_score="5E-1",
    )
    result = from_xgboost_artifact(document)

    assert jnp.array_equal(
        result.model(jnp.array([[-1.0], [1.0], [jnp.nan]])),
        jnp.array([[1.5, 2.5], [-0.5, 3.5], [1.5, 2.5]]),
    )
    assert result.model.model.out_size == 2
    assert _configuration(result)["vector_leaf"] is True


def test_dart_weight_drop_scales_each_tree_before_sum():
    document = _saved_model(
        [_scalar_tree(0, leaf_value=2.0), _scalar_tree(1, leaf_value=4.0)],
        weight_drop=[0.25, 0.5],
        base_score="1E0",
    )
    result = from_xgboost_artifact(document)

    assert jnp.array_equal(result.model(jnp.array([[0.0], [9.0]])), jnp.array([3.5, 3.5]))
    configuration = _configuration(result)
    assert configuration["dart_weighted"] is True
    assert configuration["tree_weights"] == (0.25, 0.5)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("missing-version", ConversionError),
        ("unknown-top-field", UnsupportedConversionError),
        ("unsupported-booster", UnsupportedConversionError),
        ("unsupported-objective", UnsupportedConversionError),
        ("noncanonical-parallel-layout", UnsupportedConversionError),
        ("categorical-segment-mismatch", UnsupportedConversionError),
        ("vector-width-mismatch", UnsupportedConversionError),
        ("malformed-parent-links", ConversionError),
    ],
)
def test_malformed_and_unsupported_saved_models_fail_closed(mutation, error):
    document = _saved_model(
        [
            _scalar_tree(
                0,
                threshold=0.0,
                left_value=-1.0,
                right_value=1.0,
            )
        ]
    )
    if mutation == "missing-version":
        del document["version"]
    elif mutation == "unknown-top-field":
        document["configuration"] = {}
    elif mutation == "unsupported-booster":
        document["learner"]["gradient_booster"]["name"] = "dart"
    elif mutation == "unsupported-objective":
        document["learner"]["objective"] = {"name": "binary:hinge"}
    elif mutation == "noncanonical-parallel-layout":
        document["learner"]["gradient_booster"]["model"]["gbtree_model_param"][
            "num_parallel_tree"
        ] = "2"
    elif mutation == "categorical-segment-mismatch":
        tree = document["learner"]["gradient_booster"]["model"]["trees"][0]
        tree["split_type"][0] = 1
        tree["categories_nodes"] = [0]
        tree["categories_segments"] = [1]
        tree["categories_sizes"] = [1]
        tree["categories"] = [2]
    elif mutation == "vector-width-mismatch":
        document = _saved_model([_vector_tree()], num_target=3)
    elif mutation == "malformed-parent-links":
        document["learner"]["gradient_booster"]["model"]["trees"][0]["parents"][2] = 1

    with pytest.raises(error):
        from_xgboost_artifact(document)


def test_bad_ubjson_and_duplicate_json_keys_fail_as_conversion_errors():
    document = _saved_model([_scalar_tree(0, leaf_value=1.0)])
    with pytest.raises(ConversionError, match="trailing bytes"):
        from_xgboost_artifact(_ubjson(document) + b"x")
    with pytest.raises(ConversionError, match="Duplicate JSON object key"):
        from_xgboost_artifact('{"learner":{},"learner":{},"version":[3,0,0]}')


def test_source_mapping_is_copied_and_not_mutated():
    document = _saved_model([_scalar_tree(0, leaf_value=1.0)])
    original = copy.deepcopy(document)

    from_xgboost_artifact(document)

    assert document == original
