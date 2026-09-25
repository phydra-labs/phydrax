#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


pytest.importorskip("dp_accounting")
pytest.importorskip("jax_privacy")


class _LinearOperator(phx.nn.operator.AbstractOperatorModel):
    weight: jax.Array
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.weight = jnp.asarray([[1.0]], dtype=jnp.float32)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("state").values
        assert values is not None
        return (values[..., None] @ self.weight)[..., 0]

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _dataset(cases: int = 4):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 8),
        quadrature_weights=jnp.full((8,), 1.0 / 8.0),
    )
    offsets = jnp.arange(cases, dtype="float64")[:, None]
    values = offsets + axis.nodes[None, :]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"output": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _task():
    return phx.nn.operator.OperatorTask(
        "private-map",
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "input",
                role="source",
                source_name="state",
                dimension=phx.units.DIMENSIONLESS,
            ),
            phx.nn.operator.OperatorFieldSpec(
                "output",
                role="target",
                query_name="query",
                dimension=phx.units.DIMENSIONLESS,
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=(phx.units.LENGTH,),
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=False,
        ),
    )


def _privacy(iterations: int = 2):
    definition = phx.privacy.PrivacyDefinition(
        phx.privacy.PrivacyUnit("operator-case"),
        phx.privacy.NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    scope = phx.privacy.PrivateDataScope("operator-study", definition)
    return phx.privacy.PrivateTrainingPlan(
        phx.privacy.DPSGDPlan(
            scope,
            phx.privacy.PrivacyBudget(5.0, 1e-5),
            sampling_probability=0.5,
            iterations=iterations,
            clipping_norm=1.0,
            normalize_by=2.0,
            dtype="float32",
        )
    )


def _fit(dataset, privacy, *, steps, checkpoint_path=None, resume=False, key_seed=7):
    task = _task()
    output_port = task.field_by_name["output"].value_port()
    return phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        task=task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_ports={"output": output_port},
        port_mapping=phx.PortMapping(
            outputs=((output_port.port_id, output_port.port_id),)
        ),
        steps=steps,
        privacy=privacy,
        include_model_losses=False,
        loss_terms=(phx.nn.operator.training.SupervisedOperatorLoss(),),
        key=jr.key(key_seed),
        checkpoint_path=checkpoint_path,
        checkpoint_every=1,
        resume=resume,
        jit=True,
    )


def _assert_models_equal(left, right):
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        if isinstance(left_leaf, jax.Array):
            assert jnp.array_equal(left_leaf, right_leaf)


def test_private_operator_fit_resume_and_artifact_preserve_release_boundary(tmp_path):
    dataset = _dataset()
    privacy = _privacy()
    uninterrupted = _fit(dataset, privacy, steps=2)

    assert uninterrupted.completed_steps == 2
    assert uninterrupted.privacy_certificate is not None
    assert not uninterrupted.privacy_certificate.public_release_allowed
    assert uninterrupted.history.initial_metrics == {}
    assert uninterrupted.history.final_metrics == {}
    assert uninterrupted.history.train_metrics == ({}, {})
    with pytest.raises(ValueError, match="not released"):
        _ = uninterrupted.initial_loss

    checkpoint = tmp_path / "private-checkpoint"
    partial = _fit(dataset, privacy, steps=1, checkpoint_path=checkpoint)
    assert partial.privacy_certificate is not None
    assert partial.privacy_certificate.guarantee.epsilon < (
        uninterrupted.privacy_certificate.guarantee.epsilon
    )
    checkpoint_manifest = json.loads(
        (checkpoint / "manifest.json").read_text(encoding="utf-8")
    )
    assert checkpoint_manifest["metadata"]["privacy_classification"] == "restricted"
    resumed = _fit(
        dataset,
        privacy,
        steps=2,
        checkpoint_path=checkpoint,
        resume=True,
        key_seed=999,
    )
    assert resumed.resumed_from_step == 1
    _assert_models_equal(uninterrupted.execution_model, resumed.execution_model)
    assert resumed.privacy_certificate == uninterrupted.privacy_certificate

    trained = uninterrupted.trained_operator
    assert trained is not None
    assert trained.privacy_certificate == uninterrupted.privacy_certificate
    artifact = tmp_path / "private-operator"
    with pytest.raises(ValueError, match="restricted training state"):
        phx.nn.operator.training.save_operator_artifact(
            artifact,
            trained,
            training_state=jnp.asarray(1.0),
            portable=False,
            execution_model_factory_id="tests.private-linear",
        )
    with pytest.raises(PermissionError, match="not qualified"):
        phx.nn.operator.training.save_operator_artifact(
            tmp_path / "public-operator",
            trained,
            portable=False,
            execution_model_factory_id="tests.private-linear",
            public_release=True,
        )
    phx.nn.operator.training.save_operator_artifact(
        artifact,
        trained,
        portable=False,
        execution_model_factory_id="tests.private-linear",
    )
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    serialized = json.dumps(manifest, sort_keys=True)
    assert manifest["privacy_certificate"]["certificate_id"] == (
        uninterrupted.privacy_certificate.certificate_id
    )
    assert manifest["privacy_classification"] == "restricted"
    assert "train_metrics" not in serialized
    assert "sampler_seed" not in serialized
    assert "key_data" not in serialized
    manifest_path = artifact / "manifest.json"
    manifest["privacy_classification"] = "public"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="not qualified"):
        phx.nn.operator.training.load_operator_artifact_manifest(artifact)
    manifest["privacy_classification"] = "restricted"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    restored = phx.nn.operator.training.load_trained_operator(
        artifact,
        execution_model_like=_LinearOperator(),
    )
    assert restored.privacy_certificate == uninterrupted.privacy_certificate


def test_private_operator_fit_rejects_unaccounted_surfaces():
    dataset = _dataset()
    privacy = _privacy(iterations=1)
    common = {
        "privacy": privacy,
        "steps": 1,
        "include_model_losses": False,
        "loss_terms": (phx.nn.operator.training.SupervisedOperatorLoss(),),
        "jit": False,
    }
    with pytest.raises(ValueError, match="normalization"):
        phx.nn.operator.training.fit_operator(
            _LinearOperator(), dataset, normalization="fit", **common
        )
    with pytest.raises(ValueError, match="include_model_losses"):
        phx.nn.operator.training.fit_operator(
            _LinearOperator(),
            dataset,
            **(common | {"include_model_losses": True}),
        )
    with pytest.raises(ValueError, match="explicitly public"):
        phx.nn.operator.training.fit_operator(
            _LinearOperator(), dataset, validation=dataset, **common
        )
