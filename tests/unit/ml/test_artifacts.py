#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax.ml.artifacts import read_ml_artifact, save_ml_artifact


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
    assert restored.manifest.model_type == "phydrax.artifact:FrozenModel@1"
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
