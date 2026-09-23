#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import pytest
from jaxtyping import Array

import phydrax as phx


def _scale_operator_contract(model):
    return phx.nn.operator.ConfiguredOperatorContract(
        architecture="test-scale-operator",
        configuration=(),
        capabilities=model.capability,
        training=model.training,
    )


class _ScaleOperator(phx.nn.operator.AbstractOperatorModel):
    operator_architecture = "test-scale-operator"
    _operator_contract_builder = staticmethod(_scale_operator_contract)

    scale: Array
    capability: phx.nn.operator.OperatorCapabilitySpec = eqx.field(static=True)
    training: phx.nn.operator.OperatorTrainingRequirement = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale, *, capability=None, training=None):
        self.scale = jnp.asarray(scale)
        self.capability = (
            phx.nn.operator.OperatorCapabilitySpec(
                source_geometries=("abstract",),
                query_geometries=("abstract",),
            )
            if capability is None
            else capability
        )
        self.training = (
            phx.nn.operator.OperatorTrainingRequirement()
            if training is None
            else training
        )
        self.in_size = 1
        self.out_size = 1

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        values = batch.input("state").values
        assert values is not None
        return self.scale * values

    def __call__(self, batch, /, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _path():
    low = phx.fidelity.FidelityLevelSpec(
        "low",
        problem_id="operator-problem",
        observable_id="field",
        model_id="coarse-operator",
        approximation_id="coarse",
        observable_contract_id="field-on-target-grid",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "high",
        problem_id="operator-problem",
        observable_id="field",
        model_id="fine-operator",
        approximation_id="fine",
        observable_contract_id="field-on-target-grid",
    )
    return phx.fidelity.FidelityHierarchy(
        (low, high),
        (phx.fidelity.FidelityRelation("low", "high"),),
        target_level_id="high",
    ).linear_path()


def _dataset(case_ids, *, scale):
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, 4))
    values = jnp.stack(
        tuple(jnp.linspace(float(index), float(index) + 1.0, 4) for index in case_ids)
    )
    provenance = tuple(
        phx.nn.operator.OperatorCaseProvenance(
            f"case-{index}",
            identities={"physical_case_id": f"physical-{index}"},
        )
        for index in case_ids
    )
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"output": scale * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
        provenance=provenance,
    )


def test_fidelity_correction_operator_adds_baseline_and_correction():
    dataset = _dataset((0, 1, 2), scale=3.0)
    model = phx.nn.operator.architectures.FidelityCorrectionOperator(
        _ScaleOperator(1.0),
        _ScaleOperator(2.0),
        _path(),
    )
    values = model.__call_operator_batch__(dataset.batch)
    source = dataset.batch.input("state").values
    assert source is not None
    assert jnp.allclose(values, 3.0 * source)


def test_fidelity_contract_intersects_both_child_capabilities():
    baseline_capability = phx.nn.operator.OperatorCapabilitySpec(
        source_geometries=("tensor_grid",),
        query_geometries=("tensor_grid",),
        spatial_dimensions=(2,),
        axis_requirement="uniform",
        quadrature="physical_required",
        masks="all_valid_only",
        resolution_transfer=True,
        multiple_queries=True,
    )
    correction_capability = phx.nn.operator.OperatorCapabilitySpec(
        source_geometries=("tensor_grid", "point_cloud"),
        query_geometries=("tensor_grid", "point_cloud"),
        spatial_dimensions=(2, 3),
        quadrature="optional",
        masks="supported",
        resolution_transfer=False,
        multiple_queries=False,
    )
    model = phx.nn.operator.architectures.FidelityCorrectionOperator(
        _ScaleOperator(1.0, capability=baseline_capability),
        _ScaleOperator(2.0, capability=correction_capability),
        _path(),
    )
    capability = model.operator_contract.capabilities

    assert capability.source_geometries == ("tensor_grid",)
    assert capability.query_geometries == ("tensor_grid",)
    assert capability.spatial_dimensions == (2,)
    assert capability.axis_requirement == "uniform"
    assert capability.quadrature == "physical_required"
    assert capability.masks == "all_valid_only"
    assert not capability.resolution_transfer
    assert not capability.multiple_queries


def test_fidelity_contract_rejects_unrepresentable_training_requirements():
    baseline = _ScaleOperator(
        1.0,
        training=phx.nn.operator.OperatorTrainingRequirement(),
    )
    correction = _ScaleOperator(
        2.0,
        training=phx.nn.operator.OperatorTrainingRequirement(
            regime="pretrained_system",
            pretrained_weights_required=True,
        ),
    )
    with pytest.raises(ValueError, match="training requirement"):
        phx.nn.operator.architectures.FidelityCorrectionOperator(
            baseline,
            correction,
            _path(),
        )


def test_fidelity_operator_preparation_pairs_physical_cases_explicitly():
    low = _dataset((0, 1, 2, 3), scale=1.0)
    target = _dataset((0, 1, 2), scale=3.0)
    prepared = phx.nn.operator.training.prepare_fidelity_operator_dataset(
        low,
        target,
        _path(),
    )

    assert prepared.dataset.size == 3
    assert prepared.paired_low_indices == (0, 1, 2)
    assert prepared.paired_target_indices == (0, 1, 2)
    assert prepared.low_only_case_ids == ("case-3",)
    assert not prepared.target_only_case_ids
    assert all(
        record.identities["target_fidelity_id"] == "high"
        for record in prepared.dataset.provenance
    )

    with pytest.raises(ValueError, match="unpaired target"):
        phx.nn.operator.training.prepare_fidelity_operator_dataset(
            _dataset((0, 1), scale=1.0),
            _dataset((0, 1, 2), scale=3.0),
            _path(),
        )
