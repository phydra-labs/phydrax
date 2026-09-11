from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._identity import SemanticProvenance
from phydrax._model import AbstractArrayModel, FrozenModel
from phydrax._trainable import NonTrainableState
from phydrax.closure_data._dataset import (
    ChunkedClosureDatasetManifest,
    ClosureDatasetChunk,
    ClosureSample,
    ClosureSampleKey,
    DatasetExtent,
    LeakageSafePartition,
    LeakageSafePartitionPlan,
    NormalizerProvenance,
    PartitionAssignment,
    TrainOnlyNormalizer,
)
from phydrax.closure_data._kinetic_equilibrium import (
    energy_equilibrium_numeric_revision,
    EnergyEquilibriumTrainingPair,
    LearnedEnergyEquilibriumBindingPlan,
    prepare_energy_equilibrium_dataset,
    PreparedLearnedEnergyEquilibriumBinding,
)
from phydrax.closure_data._state import FlowStateSchema
from phydrax.discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from phydrax.discretization.discrete_velocity._quadrature import d2v17_quadrature


class _AffineDualModel(AbstractArrayModel):
    weight: jax.Array
    bias: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, in_size: int = 4, out_size: int = 2, *, offset: float = 0.0):
        self.weight = jnp.zeros((out_size, in_size), dtype=jnp.float64)
        self.bias = jnp.full((out_size,), offset, dtype=jnp.float64)
        self.in_size = in_size
        self.out_size = out_size

    def __call__(self, values, /, *, key=None):
        del key
        return self.weight @ values + self.bias


def _schema() -> FlowStateSchema:
    return FlowStateSchema(
        ("density", "momentum_x", "momentum_y", "total_energy"),
        ("kg/m^3", "kg/(m^2*s)", "kg/(m^2*s)", "J/m^3"),
        (1.0, 1.0, 1.0, 1.0),
        density_name="density",
        total_energy_name="total_energy",
    )


def _sample_key(index: int) -> ClosureSampleKey:
    return ClosureSampleKey(
        case_id=f"case-{index}",
        trajectory_id="trajectory",
        realization_id="realization",
        time_block_id="block",
        time_index=0,
    )


def _manifest(
    keys: tuple[ClosureSampleKey, ...], schema_id: str
) -> ChunkedClosureDatasetManifest:
    extents = tuple(
        DatasetExtent(
            case_id=key.case_id,
            trajectory_id=key.trajectory_id,
            realization_id=key.realization_id,
            time_block_id=key.time_block_id,
            sample_count=1,
        )
        for key in keys
    )
    chunks = tuple(
        ClosureDatasetChunk.from_payload(
            f"payload-{index}".encode(),
            extent_id=extent.extent_id,
            logical_name=f"aligned-pair-{index}",
            chunk_index=0,
            sample_start=0,
            sample_stop=1,
            byte_offset=0,
        )
        for index, extent in enumerate(extents)
    )
    return ChunkedClosureDatasetManifest(
        dataset_id="energy-equilibrium-oracle",
        schema_id=schema_id,
        analysis_dag_id="oracle-generation-dag",
        extents=extents,
        chunks=chunks,
    )


def _aligned_inputs():
    schema = _schema()
    equilibrium = PositiveEnergyEquilibriumPlan(d2v17_quadrature())
    conserved_values = (
        (1.0, 0.0, 0.0, 2.0),
        (3.0, 0.2, -0.1, 4.0),
        (100.0, 3.0, 2.0, 120.0),
        (200.0, -4.0, 1.0, 250.0),
    )
    oracle_values = ((0.0, 0.0), (0.1, -0.1), (0.2, 0.1), (-0.2, 0.3))
    keys = tuple(_sample_key(index) for index in range(len(conserved_values)))
    conserved = tuple(
        ClosureSample(jnp.asarray(values), key, schema_id=schema.schema_id)
        for values, key in zip(conserved_values, keys, strict=True)
    )
    oracle = tuple(
        ClosureSample(jnp.asarray(values), key, schema_id=schema.schema_id)
        for values, key in zip(oracle_values, keys, strict=True)
    )
    pairs = tuple(
        EnergyEquilibriumTrainingPair(
            state,
            dual,
            quadrature_id=equilibrium.quadrature.quadrature_id,
            material_id="calorically-perfect-gas",
            oracle_plan_id=equilibrium.plan_id,
        )
        for state, dual in zip(conserved, oracle, strict=True)
    )
    split_names = ("train", "train", "validation", "test")
    partition_plan = LeakageSafePartitionPlan(
        "case",
        train_fraction=0.5,
        validation_fraction=0.25,
        test_fraction=0.25,
        salt="energy-equilibrium",
    )
    assignments = tuple(
        PartitionAssignment(
            sample_id=sample.sample_id,
            group_key=sample.key.group_key("case"),
            split=split,
        )
        for sample, split in zip(conserved, split_names, strict=True)
    )
    partition = LeakageSafePartition(partition_plan, assignments)
    normalizer = TrainOnlyNormalizer.fit(
        conserved,
        partition,
        feature_name="conserved-state",
        epsilon=1e-12,
    )
    manifest = _manifest(keys, schema.schema_id)
    return schema, equilibrium, conserved, oracle, pairs, manifest, partition, normalizer


def _prepared_dataset():
    inputs = _aligned_inputs()
    dataset = prepare_energy_equilibrium_dataset(
        inputs[4], inputs[5], inputs[6], inputs[7]
    )
    return (*inputs, dataset)


def _binding_plan():
    schema, equilibrium, *_, dataset = _prepared_dataset()
    semantic = SemanticProvenance(
        {
            "kind": "learned-positive-energy-equilibrium-dual",
            "architecture": "affine-test-model",
        },
        resource_ids={"dataset": dataset.preparation_id},
    )
    plan = LearnedEnergyEquilibriumBindingPlan(
        equilibrium,
        schema,
        dataset,
        semantic,
        input_component_names=schema.component_names,
        material_id="calorically-perfect-gas",
    )
    return plan, semantic, dataset


def test_aligned_pairs_prepare_leakage_safe_splits_and_train_only_statistics():
    (
        _,
        _,
        conserved,
        _,
        pairs,
        manifest,
        partition,
        normalizer,
        dataset,
    ) = _prepared_dataset()

    assert all(
        pair.conserved.key.sample_id == pair.oracle_dual.key.sample_id for pair in pairs
    )
    assert tuple(pair.conserved.sample_id for pair in dataset.train_pairs) == tuple(
        sample.sample_id for sample in conserved[:2]
    )
    assert len(dataset.validation_pairs) == 1
    assert len(dataset.test_pairs) == 1
    train_cases = {pair.conserved.key.case_id for pair in dataset.train_pairs}
    holdout_cases = {
        pair.conserved.key.case_id
        for pair in (*dataset.validation_pairs, *dataset.test_pairs)
    }
    assert train_cases.isdisjoint(holdout_cases)
    np.testing.assert_allclose(normalizer.mean, (2.0, 0.1, -0.05, 3.0))
    assert normalizer.provenance.partition_id == partition.partition_id
    assert normalizer.provenance.training_sample_ids == tuple(
        sample.sample_id for sample in conserved[:2]
    )
    assert all(
        sample.sample_id not in normalizer.provenance.training_sample_ids
        for sample in conserved[2:]
    )

    train_only_plan = LeakageSafePartitionPlan(
        "case",
        train_fraction=1.0,
        validation_fraction=0.0,
        test_fraction=0.0,
        salt="energy-equilibrium-train-only",
    )
    train_only_partition = LeakageSafePartition(
        train_only_plan,
        tuple(
            PartitionAssignment(
                sample_id=sample.sample_id,
                group_key=sample.key.group_key("case"),
                split="train",
            )
            for sample in conserved
        ),
    )
    train_only_normalizer = TrainOnlyNormalizer.fit(
        conserved,
        train_only_partition,
        feature_name="conserved-state",
        epsilon=1e-12,
    )
    train_only = prepare_energy_equilibrium_dataset(
        pairs,
        manifest,
        train_only_partition,
        train_only_normalizer,
    )
    assert not train_only.validation_pairs
    assert not train_only.test_pairs


def test_pair_manifest_partition_and_normalizer_identity_mismatches_are_rejected():
    schema, equilibrium, conserved, oracle, pairs, manifest, partition, normalizer = (
        _aligned_inputs()
    )
    other_key = ClosureSampleKey(
        case_id="foreign-case",
        trajectory_id="trajectory",
        realization_id="realization",
        time_block_id="block",
        time_index=0,
    )
    misaligned_dual = ClosureSample(
        oracle[0].values, other_key, schema_id=schema.schema_id
    )
    with pytest.raises(ValueError, match="share one key"):
        EnergyEquilibriumTrainingPair(
            conserved[0],
            misaligned_dual,
            quadrature_id=equilibrium.quadrature.quadrature_id,
            material_id="calorically-perfect-gas",
            oracle_plan_id=equilibrium.plan_id,
        )

    foreign_pair = EnergyEquilibriumTrainingPair(
        conserved[-1],
        oracle[-1],
        quadrature_id="foreign-quadrature",
        material_id="calorically-perfect-gas",
        oracle_plan_id=equilibrium.plan_id,
    )
    with pytest.raises(ValueError, match="pair identities"):
        prepare_energy_equilibrium_dataset(
            (*pairs[:-1], foreign_pair), manifest, partition, normalizer
        )

    foreign_manifest = _manifest(
        tuple(value.key for value in conserved), "foreign-schema"
    )
    with pytest.raises(ValueError, match="manifest schema"):
        prepare_energy_equilibrium_dataset(pairs, foreign_manifest, partition, normalizer)

    incomplete_partition = LeakageSafePartition(
        partition.plan, partition.assignments[:-1]
    )
    with pytest.raises(ValueError, match="exactly match"):
        prepare_energy_equilibrium_dataset(
            pairs, manifest, incomplete_partition, normalizer
        )

    no_train_partition = LeakageSafePartition(
        partition.plan,
        tuple(
            PartitionAssignment(
                sample_id=sample.sample_id,
                group_key=sample.key.group_key("case"),
                split="test",
            )
            for sample in conserved
        ),
    )
    with pytest.raises(ValueError, match="nonempty training split"):
        prepare_energy_equilibrium_dataset(
            pairs, manifest, no_train_partition, normalizer
        )

    leaked_provenance = NormalizerProvenance(
        partition_id=partition.partition_id,
        training_assignment_ids=tuple(
            value.assignment_id for value in partition.assignments
        ),
        training_sample_ids=tuple(value.sample_id for value in conserved),
        feature_name="conserved-state",
        schema_id=schema.schema_id,
    )
    leaked_normalizer = TrainOnlyNormalizer(
        normalizer.mean,
        normalizer.scale,
        leaked_provenance,
        epsilon=normalizer.epsilon,
    )
    with pytest.raises(ValueError, match="authoritative conserved training"):
        prepare_energy_equilibrium_dataset(pairs, manifest, partition, leaked_normalizer)

    counterfeit_normalizer = TrainOnlyNormalizer(
        normalizer.mean + 1.0,
        normalizer.scale,
        normalizer.provenance,
        epsilon=normalizer.epsilon,
    )
    with pytest.raises(ValueError, match="authoritative conserved training"):
        prepare_energy_equilibrium_dataset(
            pairs, manifest, partition, counterfeit_normalizer
        )


def test_binding_rejects_revision_and_model_shape_mismatches():
    plan, semantic, dataset = _binding_plan()
    model = _AffineDualModel()
    stale_revision = energy_equilibrium_numeric_revision(
        semantic, _AffineDualModel(offset=0.5)
    )
    with pytest.raises(ValueError, match="Numeric revision"):
        plan.prepare(model, stale_revision, dataset.normalizer)

    counterfeit_normalizer = TrainOnlyNormalizer(
        dataset.normalizer.mean + 1.0,
        dataset.normalizer.scale,
        dataset.normalizer.provenance,
        epsilon=dataset.normalizer.epsilon,
    )
    with pytest.raises(ValueError, match="Normalizer provenance"):
        plan.prepare(
            model,
            energy_equilibrium_numeric_revision(semantic, model),
            counterfeit_normalizer,
        )

    for bad_model in (
        _AffineDualModel(in_size=3, out_size=2),
        _AffineDualModel(in_size=4, out_size=1),
    ):
        with pytest.raises(ValueError, match="4→2"):
            plan.prepare(
                bad_model,
                energy_equilibrium_numeric_revision(semantic, bad_model),
                dataset.normalizer,
            )


def test_prepared_binding_freezes_and_vectorizes_finite_prediction_and_evaluation():
    plan, semantic, dataset = _binding_plan()
    model = _AffineDualModel()
    prepared = plan.prepare(
        model,
        energy_equilibrium_numeric_revision(semantic, model),
        dataset.normalizer,
    )
    conserved = jnp.asarray(
        ((1.0, 0.0, 0.0, 2.0), (2.0, 0.1, -0.1, 3.0)),
        dtype=jnp.float64,
    )

    dual = prepared.predict_dual(conserved)
    result = prepared.evaluate(
        jnp.asarray((2.0, 3.0)),
        jnp.zeros((2, 2), dtype=jnp.float64),
        conserved,
    )

    assert isinstance(prepared, PreparedLearnedEnergyEquilibriumBinding)
    assert isinstance(prepared, NonTrainableState)
    assert isinstance(prepared.model, FrozenModel)
    assert plan.population_convention == "weight_absorbed_total_energy_sum"
    assert dual.shape == (2, 2)
    assert result.populations.shape == (
        2,
        plan.equilibrium_plan.quadrature.population_count,
    )
    assert bool(jnp.all(jnp.isfinite(dual)))
    assert bool(jnp.all(jnp.isfinite(result.populations)))
    assert bool(jnp.all(result.populations > 0.0))
    np.testing.assert_allclose(result.evidence.recovered_total_energy, (2.0, 3.0))
    np.testing.assert_allclose(result.evidence.recovered_flux, 0.0, atol=1e-12)
