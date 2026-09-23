#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import zipfile

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
import phydrax._model._structure as structure_module
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax._model import (
    model_from_structure_recipe,
    model_structure_recipe,
    register_artifact_value,
)
from phydrax.ml.artifacts import read_ml_artifact, save_ml_artifact


def _first_recipe_key():
    return None


def _second_recipe_key():
    return None


def test_ml_artifact_round_trip_preserves_model_and_contract(tmp_path):
    model = phx.nn.layers.Linear(
        in_size=2,
        out_size=1,
        rwf=False,
        key=jr.key(17),
    )
    diagnostics = phx.ml.FitDiagnostics(
        valid=True,
        status=phx.ml.ML_SUCCESS,
        objective=0.25,
        method="test-fit",
    )
    result = phx.ml.FitResult(
        model,
        diagnostics,
        valid=True,
        status=phx.ml.ML_SUCCESS,
        method="test-fit",
        gradient_contract=phx.ml.GradientContract.direct(conditions=("full rank",)),
    )
    feature_schema = phx.ml.FeatureSchema(("x", "t"))
    target_schema = phx.ml.TargetSchema("continuous", names=("u",))
    destination = tmp_path / "linear.phxml"

    save_ml_artifact(
        destination,
        result.model,
        fit_result=result,
        feature_schema=feature_schema,
        target_schema=target_schema,
        provenance={"source": "native", "revision": 3},
        licenses=("PNPL-2.2",),
    )
    restored = read_ml_artifact(destination)
    assert restored.manifest.feature_schema is not None
    assert restored.manifest.target_schema is not None
    assert restored.manifest.fit is not None
    points = jnp.array([[1.0, 2.0], [-3.0, 0.5]])

    assert jnp.allclose(restored.model(points), result.model(points))
    assert restored.manifest.model_type == "phydrax.ml.core:FrozenModel"
    assert restored.manifest.feature_schema["names"] == ["x", "t"]
    assert restored.manifest.target_schema["names"] == ["u"]
    assert restored.manifest.fit["gradient_contract"]["fit_mode"] == "direct"
    assert restored.manifest.provenance == {"revision": 3, "source": "native"}
    assert restored.manifest.licenses == ("PNPL-2.2",)


def test_ml_artifact_rejects_checksum_corruption(tmp_path):
    model = phx.nn.layers.Linear(
        in_size=1,
        out_size=1,
        rwf=False,
        key=jr.key(2),
    )
    destination = tmp_path / "model.phxml"
    save_ml_artifact(destination, model)
    payload = bytearray(destination.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    destination.write_bytes(payload)

    with pytest.raises(ArrayArchiveCorruptionError):
        read_ml_artifact(destination)


def test_quantum_feature_artifact_round_trip_preserves_execution(tmp_path):
    layout = phx.operators.quantum.HilbertRegisterLayout(("q",), (2,))
    model = phx.ml.quantum.projected_iqp_feature_map(layout, axes=("Z",))
    destination = tmp_path / "quantum-feature.phxml"
    point = jnp.asarray([0.37], dtype=jnp.float64)

    save_ml_artifact(destination, model)
    restored = read_ml_artifact(destination)

    assert jnp.allclose(restored.model(point), model(point))


def _rewrite_archive_manifest(path, mutate):
    with zipfile.ZipFile(path, mode="r") as source:
        members = {
            information.filename: source.read(information)
            for information in source.infolist()
            if information.filename != "manifest.json"
        }
        manifest = json.loads(source.read("manifest.json"))
    mutate(manifest)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as target:
        target.writestr(
            "manifest.json",
            json.dumps(manifest, allow_nan=False, sort_keys=True).encode("utf-8"),
        )
        for name, payload in members.items():
            target.writestr(name, payload)


def test_ml_artifact_rejects_billion_element_recipe_before_device_allocation(
    tmp_path,
    monkeypatch,
):
    model = phx.nn.layers.Linear(
        in_size=1,
        out_size=1,
        rwf=False,
        key=jr.key(8),
    )
    destination = tmp_path / "oversized-recipe.phxml"
    save_ml_artifact(destination, model)

    def mutate(manifest):
        pending = [manifest["model_recipe"]]
        while pending:
            node = pending.pop()
            if node["kind"] == "array":
                node["shape"] = [65_536, 65_536]
                return
            if node["kind"] == "dataclass":
                pending.extend(node["fields"].values())
            elif node["kind"] in ("tuple", "list", "set", "frozenset"):
                pending.extend(node["items"])
            elif node["kind"] in ("mapping", "frozendict", "mappingproxy"):
                pending.extend(item for pair in node["items"] for item in pair)
        raise AssertionError("fixture model recipe contains no array")

    _rewrite_archive_manifest(destination, mutate)
    monkeypatch.setattr(
        structure_module.jnp,
        "asarray",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("device allocation was reached before recipe admission")
        ),
    )
    with pytest.raises(ArrayArchiveCorruptionError, match="recipe"):
        read_ml_artifact(destination)


def test_ml_artifact_rejects_nonmapping_optional_metadata_as_corruption(tmp_path):
    model = phx.nn.layers.Linear(
        in_size=1,
        out_size=1,
        rwf=False,
        key=jr.key(9),
    )
    destination = tmp_path / "invalid-metadata.phxml"
    save_ml_artifact(destination, model)
    _rewrite_archive_manifest(
        destination,
        lambda manifest: manifest.__setitem__("feature_schema", []),
    )

    with pytest.raises(ArrayArchiveCorruptionError, match="metadata"):
        read_ml_artifact(destination)


def test_ml_artifact_binds_recipe_shape_to_exact_leaf_inventory(
    tmp_path,
    monkeypatch,
):
    model = phx.nn.layers.Linear(
        in_size=1,
        out_size=1,
        rwf=False,
        key=jr.key(10),
    )
    destination = tmp_path / "mismatched-leaf.phxml"
    save_ml_artifact(destination, model)

    def mutate(manifest):
        pending = [manifest["model_recipe"]]
        while pending:
            node = pending.pop()
            if node["kind"] == "array":
                node["shape"] = [3, 3]
                return
            if node["kind"] == "dataclass":
                pending.extend(node["fields"].values())
            elif node["kind"] in ("tuple", "list", "set", "frozenset"):
                pending.extend(node["items"])
            elif node["kind"] in ("mapping", "frozendict", "mappingproxy"):
                pending.extend(item for pair in node["items"] for item in pair)
        raise AssertionError("fixture model recipe contains no array")

    _rewrite_archive_manifest(destination, mutate)
    monkeypatch.setattr(
        structure_module.jnp,
        "asarray",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("device allocation was reached before leaf admission")
        ),
    )

    with pytest.raises(ArrayArchiveCorruptionError, match="incompatible"):
        read_ml_artifact(destination)


def test_native_ml_artifact_refuses_registered_nonmodel_recipes(tmp_path):
    with pytest.raises(TypeError, match="AbstractArrayModel"):
        save_ml_artifact(
            tmp_path / "recipe.phxml",
            phx.ml.preprocessing.StandardScaler(),
        )


def test_model_recipe_orders_set_and_mapping_keys_by_canonical_encoding():
    register_artifact_value("test.recipe:z-key", _first_recipe_key)
    register_artifact_value("test.recipe:a-key", _second_recipe_key)

    mapping = model_structure_recipe(
        {_first_recipe_key: "first", _second_recipe_key: "second"}
    )
    members = model_structure_recipe({_first_recipe_key, _second_recipe_key})

    assert [pair[0]["value"] for pair in mapping["items"]] == [
        "test.recipe:a-key",
        "test.recipe:z-key",
    ]
    assert [item["value"] for item in members["items"]] == [
        "test.recipe:a-key",
        "test.recipe:z-key",
    ]


def test_model_recipe_rejects_malformed_prng_key_before_allocation(monkeypatch):
    monkeypatch.setattr(
        structure_module.jnp,
        "zeros",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("device allocation was reached before PRNG admission")
        ),
    )

    with pytest.raises(ValueError, match="PRNG key-data shape"):
        model_from_structure_recipe(
            {
                "kind": "prng_key",
                "shape": [1],
                "implementation": "threefry2x32",
            }
        )
