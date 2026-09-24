#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import zipfile

import equinox as eqx
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
from phydrax.ml.artifacts import (
    executable_identity,
    load_ml_model,
    read_ml_artifact,
    save_ml_artifact,
)


def _first_recipe_key():
    return None


def _second_recipe_key():
    return None


def _dimension(axis):
    return phx.units.DimensionSignature({axis: 1})


def _schema_bound_fit():
    features = jr.normal(jr.key(3), (24, 2))
    targets = features @ jnp.asarray([[1.5], [-0.5]]) + 0.25
    feature_schema = phx.ml.FeatureSchema(
        ("x", "t"),
        layout_id="probe",
        dimensions=(_dimension("length"), _dimension("time")),
    )
    target_schema = phx.ml.TargetSchema(
        "continuous", names=("u",), dimensions=(_dimension("length"),)
    )
    result = phx.ml.fit(
        phx.ml.linear.OLSRecipe(),
        features,
        targets,
        feature_schema=feature_schema,
        target_schema=target_schema,
    )
    return result, feature_schema, target_schema


def test_fitted_executable_retains_schemas_and_ports_through_jit_and_artifacts(
    tmp_path,
):
    result, feature_schema, target_schema = _schema_bound_fit()
    executable = result.as_trainable()
    ports = result.model_ports()
    points = jnp.asarray([[1.0, 2.0], [-3.0, 0.5]])

    (port,) = ports.inputs
    assert port.component_ids == ("x", "t")
    assert port.dimensions == feature_schema.dimensions
    assert [output.component_ids for output in ports.outputs] == [("u",)]
    assert ports.outputs[0].dimensions == target_schema.dimensions

    passed = eqx.filter_jit(lambda model: model)(executable)
    dynamic, static = eqx.partition(executable, eqx.is_inexact_array)
    recombined = eqx.combine(dynamic, static)
    for model in (passed, recombined):
        assert model.feature_schema == feature_schema
        assert model.model_ports().ports_id == ports.ports_id

    destination = tmp_path / "bound.phxml"
    save_ml_artifact(destination, executable, fit_result=result)
    loaded = load_ml_model(destination)

    assert loaded.feature_schema == feature_schema
    assert loaded.target_schema == target_schema
    assert loaded.model_ports().ports_id == ports.ports_id
    assert jnp.allclose(loaded(points), executable(points))


def _owner_ports():
    position = phx.ValuePort(
        "position",
        event_shape=(2,),
        component_ids=("x", "y"),
        representation="coordinates",
        dimensions=(_dimension("length"),) * 2,
        axis_keys=(phx.axes.AxisKey("owner:position", "component"),),
    )
    time = phx.ValuePort(
        "time", event_shape=(), component_ids=("t",), representation="coordinates"
    )
    speed = phx.ValuePort(
        "speed",
        event_shape=(),
        component_ids=("s",),
        representation="field",
        dimensions=(_dimension("length"),),
    )
    return position, time, speed


def test_port_backed_schemas_declare_owner_ports_through_artifacts(tmp_path):
    position, time, speed = _owner_ports()
    feature_schema = phx.ml.FeatureSchema.from_ports((position, time))
    target_schema = phx.ml.TargetSchema.from_port(speed)
    features = jr.normal(jr.key(5), (24, 3))
    result = phx.ml.fit(
        phx.ml.linear.OLSRecipe(),
        features,
        features @ jnp.asarray([1.0, -2.0, 0.5]),
        feature_schema=feature_schema,
        target_schema=target_schema,
    )

    assert feature_schema.names == ("x", "y", "t")
    # Time declares no dimensions, so the concatenated feature axis declares none.
    assert feature_schema.dimensions is None
    ports = result.model_ports()
    assert ports.inputs == (position, time)
    assert ports.outputs == (speed,)

    destination = tmp_path / "ports.phxml"
    save_ml_artifact(destination, result.model, fit_result=result)
    artifact = read_ml_artifact(destination)
    assert artifact.manifest.ports == ports
    assert artifact.model.as_trainable().model_ports() == ports


def test_port_backed_schemas_reject_contradicting_declarations():
    position, time, speed = _owner_ports()

    with pytest.raises(ValueError, match="component IDs in port order"):
        phx.ml.FeatureSchema(("y", "x", "t"), ports=(position, time))
    with pytest.raises(ValueError, match="declared dimensions"):
        phx.ml.FeatureSchema(("x", "y"), ports=(position,))
    with pytest.raises(ValueError, match="must be continuous"):
        phx.ml.TargetSchema("count", names=("s",), port=speed)
    two_ports = phx.ml.FeatureSchema.from_ports((position, time))
    assert two_ports.value_ports() == (position, time)
    assert two_ports.select((0, 1)).ports is None

    features = jr.normal(jr.key(6), (12, 2))
    vector_target = phx.ml.fit(
        phx.ml.linear.OLSRecipe(),
        features,
        jnp.stack((features[:, 0], features[:, 1]), axis=-1),
        feature_schema=phx.ml.FeatureSchema.from_ports((position,)),
        target_schema=phx.ml.TargetSchema.from_port(position),
    )
    assert vector_target.model_ports().outputs == (position,)
    scalar_target = phx.ml.fit(
        phx.ml.linear.OLSRecipe(),
        features,
        jnp.stack((features[:, 0], features[:, 1]), axis=-1),
        feature_schema=phx.ml.FeatureSchema.from_ports((position,)),
        target_schema=phx.ml.TargetSchema.from_port(speed),
    )
    with pytest.raises(ValueError, match="event shape"):
        scalar_target.model_ports()


def test_ml_artifact_round_trip_preserves_contract_and_identity(tmp_path):
    result, feature_schema, target_schema = _schema_bound_fit()
    executable = result.as_trainable()
    destination = tmp_path / "linear.phxml"

    save_ml_artifact(
        destination,
        executable,
        fit_result=result,
        provenance={"source": "native", "revision": 3},
        licenses=("PNPL-2.2",),
    )
    restored = read_ml_artifact(destination)
    manifest = restored.manifest
    semantic, numeric, signature = executable_identity(executable)

    assert manifest.derivative_contract == result.derivative_contract
    assert manifest.derivative_contract.contract_id == (
        result.derivative_contract.contract_id
    )
    assert manifest.semantic_provenance.semantic_id == semantic.semantic_id
    assert manifest.numeric_revision.revision_id == numeric.revision_id
    assert manifest.executable_signature.signature_id == signature.signature_id
    assert manifest.feature_schema == feature_schema
    assert manifest.target_schema == target_schema
    assert manifest.ports.ports_id == result.model_ports().ports_id
    assert manifest.fit["method"] == result.method
    assert manifest.provenance == {"revision": 3, "source": "native"}
    assert manifest.licenses == ("PNPL-2.2",)


def test_ml_artifact_identity_distinguishes_numeric_revisions():
    result, _, _ = _schema_bound_fit()
    executable = result.as_trainable()
    shifted = eqx.tree_at(
        lambda model: model.intercept, executable, executable.intercept + 1.0
    )

    semantic, numeric, signature = executable_identity(executable)
    shifted_semantic, shifted_numeric, shifted_signature = executable_identity(shifted)

    assert shifted_semantic.semantic_id == semantic.semantic_id
    assert shifted_signature.signature_id == signature.signature_id
    assert shifted_numeric.revision_id != numeric.revision_id


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest["identity"].__setitem__(
                "numeric_revision_id", "0" * 64
            ),
            "identity",
        ),
        (
            lambda manifest: manifest["feature_schema"].__setitem__("names", ["x", "z"]),
            "schemas or ports",
        ),
    ],
)
def test_ml_artifact_rejects_records_inconsistent_with_executable(
    tmp_path, mutate, message
):
    result, _, _ = _schema_bound_fit()
    destination = tmp_path / "tampered.phxml"
    save_ml_artifact(destination, result.as_trainable(), fit_result=result)
    _rewrite_archive_manifest(destination, mutate)

    with pytest.raises(ArrayArchiveCorruptionError, match=message):
        read_ml_artifact(destination)


def test_ml_artifact_refuses_previous_format_record(tmp_path):
    result, _, _ = _schema_bound_fit()
    destination = tmp_path / "previous.phxml"
    save_ml_artifact(destination, result.as_trainable(), fit_result=result)

    def previous_format(manifest):
        for field in ("ports", "derivative_contract", "identity"):
            del manifest[field]
        manifest["fit"]["gradient_contract"] = {"fit_mode": "direct"}

    _rewrite_archive_manifest(destination, previous_format)

    with pytest.raises(ArrayArchiveCorruptionError, match="manifest fields"):
        read_ml_artifact(destination)


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
