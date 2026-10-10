#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import io
import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from typing import NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._array_archive import ArrayArchiveLimits
from phydrax._model import _structure as structure_module, register_artifact_value
from phydrax._model._structure import (
    deserialize_model_leaf,
    deserialize_model_tree,
    model_from_array_recipe,
    model_from_logical_array_recipe,
    model_from_structure_recipe,
    model_recipe_array_inventory,
    model_recipe_array_pairs,
    model_recipe_array_values,
    model_recipe_from_wire_json,
    model_recipe_template,
    model_recipe_wire_json,
    model_structure_recipe,
    serialize_model_leaf,
    validate_model_structure_recipe,
)
from phydrax.equations import ChemicalComponentCatalog
from phydrax.exterior import FormType, FormValueSpec
from phydrax.geometry.brep._patches import BSplineCurve
from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts


def _wire(*nodes: Mapping[str, object], root: int | None = None) -> str:
    return json.dumps(
        {"nodes": list(nodes), "root": len(nodes) - 1 if root is None else root},
        separators=(",", ":"),
        sort_keys=True,
    )


def _curve() -> BSplineCurve:
    register_meshing_source_artifacts()
    return BSplineCurve(
        np.asarray([[0.0, 0.0], [1.0, 2.0]], dtype=np.float64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float64),
        1,
    )


def test_native_recipe_preserves_immutable_form_identity_and_array_payload() -> None:
    register_meshing_source_artifacts()
    spec = FormValueSpec(FormType(3, 2, twist="twisted"), proxy="flux")
    value = spec, np.asarray((0.5, 1.5, 2.5), dtype=np.float64)
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="form")
    restored_spec, restored_array = model_from_logical_array_recipe(
        recipe,
        arrays,
        prefix="form",
    )
    assert type(restored_spec) is FormValueSpec
    assert restored_spec.to_dict() == spec.to_dict()
    np.testing.assert_array_equal(restored_array, value[1])


def test_native_recipe_refuses_incompatible_form_proxy() -> None:
    register_meshing_source_artifacts()
    spec = FormValueSpec(FormType(3, 2, twist="twisted"), proxy="flux")
    recipe = model_structure_recipe(spec)
    payload = spec.to_dict()
    payload["proxy"] = "circulation"
    recipe["payload"] = model_structure_recipe(payload)
    with pytest.raises(ValueError):
        model_from_logical_array_recipe(recipe, {}, prefix="form")


def test_native_recipe_restores_host_coefficients_and_immutable_curve() -> None:
    curve = _curve()
    pieces = curve.bezier_pieces()
    value = curve, pieces
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="curve")
    restored_curve, restored_pieces = model_from_logical_array_recipe(
        recipe,
        arrays,
        prefix="curve",
    )

    np.testing.assert_allclose(
        restored_curve.evaluate(jnp.asarray(0.25, dtype=jnp.float64)),
        [0.25, 0.5],
    )
    np.testing.assert_array_equal(
        restored_pieces[0].homogeneous_controls,
        [[0.0, 0.0, 1.0], [1.0, 2.0, 1.0]],
    )
    np.testing.assert_array_equal(
        restored_pieces[0].homogeneous_lower,
        pieces[0].homogeneous_lower,
    )
    np.testing.assert_array_equal(
        restored_pieces[0].homogeneous_upper,
        pieces[0].homogeneous_upper,
    )
    with pytest.raises(ValueError, match="read-only"):
        restored_pieces[0].homogeneous_controls[0, 0] = 7.0
    with pytest.raises(AttributeError):
        restored_curve.degree = 2


@pytest.mark.parametrize("field_change", ["missing", "extra"])
def test_native_recipe_refuses_inexact_registered_field_set(field_change: str) -> None:
    curve = _curve()
    recipe = model_structure_recipe(curve)
    arrays = model_recipe_array_values(curve, recipe, prefix="curve")
    if field_change == "missing":
        recipe["items"].pop()
    else:
        recipe["items"].append({"kind": "literal", "value": None})

    with pytest.raises(ValueError, match="field count"):
        model_from_array_recipe(recipe, arrays, prefix="curve")


def test_native_recipe_refuses_corrupted_numerical_dtype() -> None:
    curve = _curve()
    recipe = model_structure_recipe(curve)
    arrays = model_recipe_array_values(curve, recipe, prefix="curve")
    name = next(iter(arrays))
    arrays[name] = np.asarray(arrays[name], dtype=np.float32)

    with pytest.raises(ValueError, match="shape/dtype"):
        model_from_logical_array_recipe(recipe, arrays, prefix="curve")


@pytest.mark.parametrize("array_change", ["missing", "extra"])
def test_native_recipe_refuses_inexact_array_coverage(array_change: str) -> None:
    curve = _curve()
    value = curve, curve.bezier_pieces()
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="curve")
    if array_change == "missing":
        del arrays[next(reversed(arrays))]
    else:
        arrays["unbound-coefficients"] = np.asarray([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="exact inventory"):
        model_from_logical_array_recipe(recipe, arrays, prefix="curve")


def test_native_recipe_bounds_static_payload_before_reconstruction() -> None:
    recipe = model_structure_recipe(("native-source-description" * 10, _curve()))
    limits = replace(ArrayArchiveLimits(), max_manifest_bytes=128)

    with pytest.raises(ValueError, match="manifest byte limit"):
        validate_model_structure_recipe(recipe, limits=limits)


def test_model_structural_template_refuses_inconsistent_scientific_dimensions() -> None:
    catalog = ChemicalComponentCatalog(
        ("H2", "O2"),
        np.asarray([2.016, 31.998], dtype=np.float64),
        ("H", "O"),
        np.asarray([[2, 0], [0, 2]], dtype=np.int64),
    )
    recipe = model_structure_recipe(catalog)
    molar_masses = next(
        index
        for index, field in enumerate(fields(type(catalog)))
        if field.name == "molar_masses"
    )
    recipe["items"][molar_masses]["shape"] = [3]

    with pytest.raises(ValueError, match="molar_masses"):
        model_from_structure_recipe(recipe)


def test_native_quality_recipe_retains_unavailable_metric_and_report_identity() -> None:
    from phydrax.discretization import CellMesh
    from phydrax.meshing._quality import evaluate_cell_quality, summarize_cell_quality

    register_meshing_source_artifacts()
    mesh = CellMesh.from_triangles(
        np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.asarray([[0, 1, 2]], dtype=np.int32),
    )
    quality = summarize_cell_quality(evaluate_cell_quality(mesh))
    recipe = json.loads(json.dumps(model_structure_recipe(quality), allow_nan=False))
    arrays = model_recipe_array_values(quality, recipe, prefix="quality")
    restored = model_from_logical_array_recipe(recipe, arrays, prefix="quality")

    assert np.isnan(restored.minimum_metric_quality)
    assert restored.report_id == quality.report_id
    np.testing.assert_array_equal(
        restored.evaluation.metric_quality,
        quality.evaluation.metric_quality,
    )
    np.testing.assert_array_equal(
        restored.evaluation.measures,
        np.asarray([0.5], dtype=np.float64),
    )


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), float("-inf")],
    ids=["unavailable", "positive-infinity", "negative-infinity"],
)
def test_recipe_retains_nonfinite_static_float_semantics(value: float) -> None:
    recipe = json.loads(json.dumps(model_structure_recipe(value), allow_nan=False))
    restored = model_from_structure_recipe(recipe)

    assert type(restored) is float
    if np.isnan(value):
        assert np.isnan(restored)
    else:
        assert restored == value


@pytest.mark.parametrize(
    ("recipe", "error_type"),
    [
        ({"kind": "nonfinite_float", "value": "unknown"}, ValueError),
        ({"kind": "nonfinite_float", "value": 0}, TypeError),
        ({"kind": "nonfinite_float", "value": "nan", "extra": 1}, ValueError),
    ],
    ids=["unknown-token", "wrong-token-type", "extra-field"],
)
def test_recipe_refuses_malformed_nonfinite_float(
    recipe: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        model_from_structure_recipe(recipe)


@pytest.mark.parametrize("template_kind", ["descriptor", "materialized"])
@pytest.mark.parametrize("writable", [False, True], ids=["immutable", "mutable"])
def test_model_stream_preserves_numpy_coefficient_mutability(
    template_kind: str,
    writable: bool,
) -> None:
    coefficient = np.asarray([2.0, 3.0], dtype=np.float64)
    coefficient.setflags(write=writable)
    model = {"coefficient": coefficient}
    recipe = model_structure_recipe(model)
    template = (
        model_recipe_template(recipe)
        if template_kind == "descriptor"
        else model_from_structure_recipe(recipe)
    )
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, model, filter_spec=serialize_model_leaf)
    stream.seek(0)
    if template_kind == "descriptor":
        restored = {
            "coefficient": deserialize_model_leaf(stream, template["coefficient"]),
        }
    else:
        restored = eqx.tree_deserialise_leaves(
            stream,
            template,
            filter_spec=deserialize_model_leaf,
        )

    np.testing.assert_array_equal(restored["coefficient"], coefficient)
    assert model_structure_recipe(restored) == recipe
    if writable:
        restored["coefficient"][0] = 5.0
        assert restored["coefficient"][0] == 5.0
    else:
        with pytest.raises(ValueError, match="read-only"):
            restored["coefficient"][0] = 5.0


@dataclass(frozen=True)
class _Epoch:
    source: object
    evidence: object
    carriers: tuple[object, ...]


register_artifact_value("test.recipe:shared-epoch", _Epoch)


def _epoch_chain(device: jax.Array, host: np.ndarray, depth: int) -> _Epoch:
    # Each epoch reaches its predecessor through three paths, so a tree recipe
    # would hold 3**depth copies of the oldest epoch and its arrays.
    epoch = _Epoch(device, {"host": host}, ())
    for _ in range(depth):
        epoch = _Epoch(epoch, {"source": epoch}, (epoch,))
    return epoch


def test_recipe_limits_stores_and_restores_shared_objects_once() -> None:
    device = jnp.arange(4, dtype=jnp.float64)
    host = np.asarray([2.0, 3.0], dtype=np.float64)
    value = _epoch_chain(device, host, 8)
    limits = replace(
        ArrayArchiveLimits(),
        max_aggregate_bytes=device.nbytes + host.nbytes,
        max_manifest_bytes=16384,
    )
    recipe = json.loads(json.dumps(model_structure_recipe(value)))

    validate_model_structure_recipe(recipe, limits=limits)
    inventory = model_recipe_array_inventory(recipe, prefix="chain", limits=limits)
    arrays = model_recipe_array_values(value, recipe, prefix="chain", limits=limits)
    assert [entry.shape for entry in inventory] == [(4,), (2,)]
    assert set(arrays) == {entry.name for entry in inventory}
    restored = model_from_array_recipe(recipe, arrays, prefix="chain", limits=limits)

    epoch = restored
    for _ in range(8):
        assert epoch.source is epoch.evidence["source"] is epoch.carriers[0]
        epoch = epoch.source
    assert isinstance(epoch.source, jax.Array)
    assert isinstance(epoch.evidence["host"], np.ndarray)
    assert epoch.evidence["host"].flags.writeable
    np.testing.assert_array_equal(epoch.source, device)
    np.testing.assert_array_equal(epoch.evidence["host"], host)
    assert model_structure_recipe(restored) == recipe


def test_recipe_restores_array_shared_by_runtime_leaves_as_one_object() -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    value = {"encoder": weight, "decoder": (weight,)}
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="tied")

    restored = model_from_logical_array_recipe(recipe, arrays, prefix="tied")

    assert len(arrays) == 1
    assert restored["encoder"] is restored["decoder"][0]
    np.testing.assert_array_equal(restored["encoder"], weight)


def test_recipe_array_pairs_ignore_storage_sharing_and_mutability() -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    copy = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    host = np.asarray([1.0], dtype=np.float64)
    frozen = host.copy()
    frozen.setflags(write=False)
    shared = ({"a": weight, "b": weight}, host)

    pairs = model_recipe_array_pairs(shared, ({"a": weight, "b": copy}, frozen))

    assert pairs is not None
    assert [(id(left), id(right)) for left, right in pairs] == [
        (id(weight), id(weight)),
        (id(weight), id(copy)),
        (id(host), id(frozen)),
    ]
    assert model_recipe_array_pairs(shared, ({"a": weight}, host)) is None
    assert (
        model_recipe_array_pairs(shared, ({"a": weight, "b": weight[:1]}, host)) is None
    )


def test_recipe_limits_refuse_distinct_arrays_with_equal_content() -> None:
    first = jnp.arange(4, dtype=jnp.float64)
    second = jnp.arange(4, dtype=jnp.float64)
    limits = replace(ArrayArchiveLimits(), max_aggregate_bytes=first.nbytes)
    validate_model_structure_recipe(
        model_structure_recipe((first, first)),
        limits=limits,
    )

    with pytest.raises(ValueError, match="total array-byte limit"):
        validate_model_structure_recipe(
            model_structure_recipe((first, second)),
            limits=limits,
        )


_ARRAY_RECIPE = {"kind": "array", "shape": [2], "dtype": "float64", "backend": "jax"}


@pytest.mark.parametrize(
    "items",
    [
        [{"kind": "reference", "target": 0}, _ARRAY_RECIPE],
        [_ARRAY_RECIPE, {"kind": "reference", "target": 1}],
        [_ARRAY_RECIPE, {"kind": "reference", "target": True}],
        [_ARRAY_RECIPE, {"kind": "reference", "target": -1}],
        [_ARRAY_RECIPE, {"kind": "reference", "target": 0, "shape": [2]}],
        [
            _ARRAY_RECIPE,
            {"kind": "frozenset", "items": [{"kind": "reference", "target": 0}]},
        ],
        [
            {
                "kind": "dataclass",
                "type": "test.recipe:shared-epoch",
                "items": [
                    {"kind": "reference", "target": 0},
                    {"kind": "literal", "value": None},
                    {"kind": "tuple", "items": []},
                ],
            }
        ],
    ],
    ids=[
        "forward",
        "out-of-range",
        "boolean",
        "negative",
        "extra-field",
        "set-item",
        "ancestor",
    ],
)
def test_recipe_refuses_reference_without_completed_earlier_object(
    items: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError):
        model_from_structure_recipe({"kind": "tuple", "items": items})


_ARRAY_NODE = {"kind": "array", "shape": [2], "dtype": "float64", "backend": "jax"}


@pytest.mark.parametrize(
    "text",
    [
        _wire({"kind": "tuple", "items": [1]}, _ARRAY_NODE),
        _wire(_ARRAY_NODE, {"kind": "tuple", "items": [1]}),
        _wire({"kind": "tuple", "items": [1]}, {"kind": "tuple", "items": [0]}),
        _wire(_ARRAY_NODE, {"kind": "tuple", "items": [5]}),
        _wire(_ARRAY_NODE, {"kind": "tuple", "items": [True]}),
        _wire(_ARRAY_NODE, {"kind": "tuple", "items": [-1]}),
        _wire(_ARRAY_NODE, {"kind": "tuple", "items": [0], "shape": [2]}),
        _wire(_ARRAY_NODE, {"kind": "frozenset", "items": [0]}),
        _wire(
            _ARRAY_NODE,
            {"kind": "literal", "value": None},
            {"kind": "tuple", "items": [1]},
        ),
        _wire(_ARRAY_NODE, {"kind": "tuple", "items": [0]}, root=0),
        _wire(
            _ARRAY_NODE,
            {"kind": "reference", "target": 0},
            {"kind": "tuple", "items": [0, 1]},
        ),
        _wire(
            {"kind": "literal", "value": 1},
            {"kind": "tuple", "items": [0]},
            {"kind": "tuple", "items": [1, 1]},
        ),
        _wire(
            {"kind": "literal", "value": 1},
            {"kind": "literal", "value": 1},
            {"kind": "tuple", "items": [0, 1]},
        ),
        _wire(
            _ARRAY_NODE,
            {"kind": "literal", "value": None},
            {
                "kind": "dataclass",
                "type": "test.recipe:shared-epoch",
                "items": [0, 1],
            },
        ),
    ],
    ids=[
        "forward",
        "self-cycle",
        "two-node-cycle",
        "out-of-range",
        "boolean",
        "negative",
        "extra-field",
        "set-item-array",
        "unreachable",
        "root-not-final",
        "wire-reference",
        "shared-container",
        "uninterned-static",
        "type-field-mismatch",
    ],
)
def test_wire_recipe_refuses_hostile_node_table(text: str) -> None:
    with pytest.raises(ValueError):
        model_recipe_from_wire_json(text)


@pytest.mark.parametrize(
    "repetitions", (2, 8), ids=("within-logical-budget", "expanded-over-budget")
)
def test_wire_static_sharing_obeys_logical_manifest_budget(repetitions: int) -> None:
    literal = {"kind": "literal", "value": "x" * 2048}
    text = _wire(literal, {"kind": "tuple", "items": [0] * repetitions})
    limits = ArrayArchiveLimits(max_manifest_bytes=8192)
    assert len(text.encode("ascii")) < limits.max_manifest_bytes
    if repetitions == 8:
        with pytest.raises(ValueError, match="manifest byte limit"):
            model_recipe_from_wire_json(text, limits=limits)
    else:
        recipe = model_recipe_from_wire_json(text, limits=limits)
        assert recipe == {"kind": "tuple", "items": [literal] * repetitions}
        assert model_recipe_wire_json(recipe, limits=limits) == text


def test_wire_recipe_keeps_constant_nesting_and_exact_logical_sharing() -> None:
    device = jnp.arange(3, dtype=jnp.float64)
    host = np.asarray([2.0, 3.0], dtype=np.float64)
    value = _epoch_chain(device, host, 64)
    recipe = model_structure_recipe(value)
    shallow = replace(ArrayArchiveLimits(), max_manifest_nesting=5)
    text = model_recipe_wire_json(recipe, limits=shallow)

    admitted = model_recipe_from_wire_json(text, limits=shallow)
    arrays = model_recipe_array_values(value, admitted, prefix="deep")
    restored = model_from_logical_array_recipe(admitted, arrays, prefix="deep")

    assert admitted == recipe
    assert len(arrays) == 2
    epoch = restored
    for _ in range(64):
        assert epoch.source is epoch.evidence["source"] is epoch.carriers[0]
        epoch = epoch.source
    np.testing.assert_array_equal(epoch.source, device)
    assert model_structure_recipe(restored) == recipe
    with pytest.raises(ValueError, match="not canonical"):
        model_recipe_from_wire_json(json.dumps(json.loads(text)), limits=shallow)


def test_wire_boundary_leaves_logical_recipe_identity_unchanged() -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    host = np.asarray([5.0, 7.0], dtype=np.float64)
    value = _Epoch(weight, {"host": host}, (weight,))
    # Independent baseline: the shared device array is logical ordinal 0 and its
    # second path is a reference; the wire encoding must not alter this record.
    expected = {
        "kind": "dataclass",
        "type": "test.recipe:shared-epoch",
        "items": [
            {
                "kind": "array",
                "shape": [2],
                "dtype": "float64",
                "backend": "jax",
            },
            {
                "kind": "mapping",
                "items": [
                    [
                        {"kind": "literal", "value": "host"},
                        {
                            "kind": "array",
                            "shape": [2],
                            "dtype": "float64",
                            "backend": "numpy",
                            "writable": True,
                        },
                    ]
                ],
            },
            {"kind": "tuple", "items": [{"kind": "reference", "target": 0}]},
        ],
    }

    recipe = model_structure_recipe(value)
    restored = model_recipe_from_wire_json(model_recipe_wire_json(recipe))

    assert recipe == expected
    assert restored == expected


def test_model_stream_restores_recipe_shared_leaves_as_one_array() -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    model = {"encoder": weight, "decoder": weight}
    recipe = model_structure_recipe(model)
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(stream, model, filter_spec=serialize_model_leaf)
    stream.seek(0)

    restored = deserialize_model_tree(stream, recipe)

    assert restored["encoder"] is restored["decoder"]
    np.testing.assert_array_equal(restored["encoder"], weight)
    assert model_structure_recipe(restored) == recipe


def test_model_stream_refuses_inconsistent_shared_leaf_payloads() -> None:
    weight = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    recipe = model_structure_recipe({"encoder": weight, "decoder": weight})
    stream = io.BytesIO()
    eqx.tree_serialise_leaves(
        stream,
        {"encoder": weight, "decoder": weight + 1.0},
        filter_spec=serialize_model_leaf,
    )
    stream.seek(0)

    with pytest.raises(ValueError, match="inconsistent serialized payloads"):
        deserialize_model_tree(stream, recipe)


@pytest.mark.parametrize(
    ("rejection", "message"),
    [
        ("recipe-budget", "limit"),
        ("extra-array", "exact inventory"),
        ("corrupt-dtype", "shape/dtype"),
    ],
)
def test_host_recipe_rejects_before_any_payload_materialization(
    monkeypatch: pytest.MonkeyPatch,
    rejection: str,
    message: str,
) -> None:
    value = jnp.asarray([2.0, 3.0], dtype=jnp.float64)
    recipe = model_structure_recipe(value)
    arrays = {"model/000000": value}
    limits = ArrayArchiveLimits()
    if rejection == "recipe-budget":
        limits = replace(limits, max_total_array_elements=1)
    elif rejection == "extra-array":
        arrays["unexpected"] = value
    else:
        arrays["model/000000"] = jnp.asarray([2.0, 3.0], dtype=jnp.float32)

    def forbid_materialization(
        _value: object,
        *_args: object,
        **_kwargs: object,
    ) -> NoReturn:
        raise RuntimeError("A refused host recipe attempted to materialize a payload.")

    monkeypatch.setattr(structure_module.np, "asarray", forbid_materialization)
    with pytest.raises(ValueError, match=message):
        model_from_array_recipe(recipe, arrays, prefix="model", limits=limits)


def test_nan_mapping_key_roundtrip_retains_its_actual_array_value() -> None:
    value = {float("nan"): np.asarray([2.0, 3.0], dtype=np.float64)}
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="mapping")
    restored = model_from_array_recipe(recipe, arrays, prefix="mapping")
    key, coefficient = next(iter(restored.items()))

    assert type(key) is float
    assert np.isnan(key)
    np.testing.assert_array_equal(coefficient, [2.0, 3.0])


def test_mapping_refuses_indistinguishable_nan_key_recipes() -> None:
    value = {
        float("nan"): np.asarray([2.0], dtype=np.float64),
        float("nan"): np.asarray([3.0], dtype=np.float64),
    }
    with pytest.raises(TypeError, match="same canonical encoding"):
        model_structure_recipe(value)


def test_fraction_recipe_retains_exact_nonbinary_rational_value() -> None:
    value = Fraction((1 << 100) + 1, (1 << 100) + 3)
    recipe = json.loads(json.dumps(model_structure_recipe(value), allow_nan=False))
    restored = model_from_structure_recipe(recipe)

    assert type(restored) is Fraction
    assert restored == value
    assert restored != Fraction(float(value))


def test_native_affine_pcurve_recipe_retains_exact_chart_and_numerical_fields() -> None:
    from phydrax.geometry.brep._intersection_curve import AffinePCurve
    from phydrax.geometry.brep._patches import LineCurve

    register_meshing_source_artifacts()
    matrix = (
        (Fraction(1, 3), Fraction(0)),
        (Fraction(0), Fraction(2, 7)),
    )
    offset = Fraction(1, 11), Fraction(-1, 13)
    source = AffinePCurve(
        LineCurve(
            np.asarray([0.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 2.0], dtype=np.float64),
        ),
        np.asarray(matrix, dtype=np.object_),
        np.asarray(offset, dtype=np.object_),
    )
    recipe = model_structure_recipe(source)
    arrays = model_recipe_array_values(source, recipe, prefix="affine-chart")
    restored = model_from_logical_array_recipe(recipe, arrays, prefix="affine-chart")

    assert restored.matrix == matrix
    assert restored.offset == offset
    assert restored.source_id == source.source_id
    np.testing.assert_array_equal(restored.matrix_value, source.matrix_value)
    np.testing.assert_array_equal(restored.offset_value, source.offset_value)
    np.testing.assert_allclose(
        restored.evaluate(jnp.asarray(0.5, dtype=jnp.float64)),
        np.asarray(
            [
                float(Fraction(1, 6) + Fraction(1, 11)),
                float(Fraction(2, 7) - Fraction(1, 13)),
            ],
            dtype=np.float64,
        ),
    )


@pytest.mark.parametrize(
    ("recipe", "error_type"),
    [
        ({"kind": "fraction", "numerator": True, "denominator": 3}, TypeError),
        ({"kind": "fraction", "numerator": 1, "denominator": 0}, ValueError),
        ({"kind": "fraction", "numerator": 1, "denominator": -3}, ValueError),
        ({"kind": "fraction", "numerator": 2, "denominator": 6}, ValueError),
        ({"kind": "fraction", "numerator": 1, "denominator": 3, "extra": 0}, ValueError),
    ],
    ids=[
        "noninteger-component",
        "zero-denominator",
        "negative-denominator",
        "noncanonical-reduction",
        "extra-field",
    ],
)
def test_fraction_recipe_refuses_noncanonical_components(
    recipe: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        model_from_structure_recipe(recipe)


def test_fraction_recipe_bounds_integer_components_before_reduction() -> None:
    recipe = {"kind": "fraction", "numerator": 1 << 600, "denominator": 1}
    limits = replace(ArrayArchiveLimits(), max_manifest_bytes=128)

    with pytest.raises(ValueError, match="manifest byte limit"):
        model_from_structure_recipe(recipe, limits=limits)
