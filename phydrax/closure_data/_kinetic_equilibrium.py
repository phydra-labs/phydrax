#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._model import AbstractArrayModel, FrozenModel
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.discrete_velocity._energy_equilibrium import (
    EnergyEquilibriumResult,
    PositiveEnergyEquilibriumPlan,
)
from ._dataset import (
    ChunkedClosureDatasetManifest,
    ClosureSample,
    LeakageSafePartition,
    TrainOnlyNormalizer,
)
from ._state import FlowStateSchema


_ENERGY_POPULATION_CONVENTION = "weight_absorbed_total_energy_sum"


def energy_equilibrium_numeric_revision(
    semantic_provenance: SemanticProvenance,
    model: AbstractArrayModel,
    /,
) -> NumericRevision:
    """Bind one model's dynamic array leaves to declared equilibrium semantics."""

    if not isinstance(semantic_provenance, SemanticProvenance):
        raise TypeError("semantic_provenance must be SemanticProvenance.")
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("model must be an AbstractArrayModel.")
    leaves = tuple(leaf for leaf in jax.tree.leaves(model) if eqx.is_array(leaf))
    if not leaves:
        raise ValueError("Energy-equilibrium models require dynamic array leaves.")
    return NumericRevision(semantic_provenance, {"model_array_leaves": leaves})


class EnergyEquilibriumTrainingPair(StrictModule, NonTrainableState):
    """One key-aligned conserved state and oracle energy-equilibrium dual."""

    conserved: ClosureSample
    oracle_dual: ClosureSample
    schema_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    oracle_plan_id: str = eqx.field(static=True)
    pair_id: str = eqx.field(static=True)

    def __init__(
        self,
        conserved: ClosureSample,
        oracle_dual: ClosureSample,
        /,
        *,
        quadrature_id: str,
        material_id: str,
        oracle_plan_id: str,
    ):
        if not isinstance(conserved, ClosureSample):
            raise TypeError("conserved must be a ClosureSample.")
        if not isinstance(oracle_dual, ClosureSample):
            raise TypeError("oracle_dual must be a ClosureSample.")
        quadrature, material, oracle = tuple(
            str(value).strip() for value in (quadrature_id, material_id, oracle_plan_id)
        )
        if not quadrature or not material or not oracle:
            raise ValueError("Energy-equilibrium pair identities must be non-empty.")
        if conserved.values.shape != (4,):
            raise ValueError("A conserved-state sample must have shape (4,).")
        if oracle_dual.values.shape != (2,):
            raise ValueError("An oracle-dual sample must have shape (2,).")
        if conserved.key.sample_id != oracle_dual.key.sample_id:
            raise ValueError("Conserved and oracle-dual samples must share one key.")
        if conserved.schema_id != oracle_dual.schema_id:
            raise ValueError("Conserved and oracle-dual schema identities must match.")
        if not jnp.issubdtype(conserved.values.dtype, jnp.floating) or not jnp.issubdtype(
            oracle_dual.values.dtype, jnp.floating
        ):
            raise TypeError(
                "Energy-equilibrium training values must be real floating arrays."
            )
        if np.any(~np.isfinite(np.asarray(conserved.values))) or np.any(
            ~np.isfinite(np.asarray(oracle_dual.values))
        ):
            raise ValueError("Energy-equilibrium training values must be finite.")
        self.conserved = conserved
        self.oracle_dual = oracle_dual
        self.schema_id = conserved.schema_id
        self.quadrature_id = quadrature
        self.material_id = material
        self.oracle_plan_id = oracle
        self.pair_id = canonical_fingerprint(
            {
                "kind": "energy-equilibrium-training-pair",
                "conserved_sample": conserved.sample_id,
                "oracle_dual_sample": oracle_dual.sample_id,
                "key": conserved.key.sample_id,
                "schema": conserved.schema_id,
                "quadrature": quadrature,
                "material": material,
                "oracle_plan": oracle,
            }
        )


class PreparedEnergyEquilibriumDataset(StrictModule, NonTrainableState):
    """Leakage-safe aligned pairs with an authoritative train-only normalizer."""

    pairs: tuple[EnergyEquilibriumTrainingPair, ...]
    train_pairs: tuple[EnergyEquilibriumTrainingPair, ...]
    validation_pairs: tuple[EnergyEquilibriumTrainingPair, ...]
    test_pairs: tuple[EnergyEquilibriumTrainingPair, ...]
    manifest: ChunkedClosureDatasetManifest
    partition: LeakageSafePartition
    normalizer: TrainOnlyNormalizer
    schema_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    oracle_plan_id: str = eqx.field(static=True)
    preparation_id: str = eqx.field(static=True)

    def __init__(
        self,
        pairs: tuple[EnergyEquilibriumTrainingPair, ...],
        manifest: ChunkedClosureDatasetManifest,
        partition: LeakageSafePartition,
        normalizer: TrainOnlyNormalizer,
        /,
    ):
        values = tuple(pairs)
        if not values or any(
            not isinstance(value, EnergyEquilibriumTrainingPair) for value in values
        ):
            raise ValueError("Energy-equilibrium preparation requires aligned pairs.")
        if not isinstance(manifest, ChunkedClosureDatasetManifest):
            raise TypeError("manifest must be a ChunkedClosureDatasetManifest.")
        if not isinstance(partition, LeakageSafePartition):
            raise TypeError("partition must be a LeakageSafePartition.")
        if not isinstance(normalizer, TrainOnlyNormalizer):
            raise TypeError("normalizer must be a TrainOnlyNormalizer.")
        if len({value.pair_id for value in values}) != len(values) or len(
            {value.conserved.key.sample_id for value in values}
        ) != len(values):
            raise ValueError(
                "Energy-equilibrium pairs must have unique identities and keys."
            )
        identities = {
            (
                value.schema_id,
                value.quadrature_id,
                value.material_id,
                value.oracle_plan_id,
            )
            for value in values
        }
        if len(identities) != 1:
            raise ValueError("Energy-equilibrium pair identities must match exactly.")
        schema, quadrature, material, oracle = next(iter(identities))
        if manifest.schema_id != schema:
            raise ValueError("Pair and manifest schema identities do not match.")

        pair_coordinates = {
            (
                value.conserved.key.case_id,
                value.conserved.key.trajectory_id,
                value.conserved.key.realization_id,
                value.conserved.key.time_block_id,
                value.conserved.key.time_index,
            )
            for value in values
        }
        manifest_coordinates = {
            (
                extent.case_id,
                extent.trajectory_id,
                extent.realization_id,
                extent.time_block_id,
                index,
            )
            for extent in manifest.extents
            for index in range(extent.sample_count)
        }
        if pair_coordinates != manifest_coordinates:
            raise ValueError("Pair keys do not exactly cover the dataset manifest.")

        sample_ids = tuple(value.conserved.sample_id for value in values)
        assignment_ids = tuple(value.sample_id for value in partition.assignments)
        if len(assignment_ids) != len(sample_ids) or set(assignment_ids) != set(
            sample_ids
        ):
            raise ValueError(
                "Partition assignments must exactly match the conserved samples."
            )
        for value in values:
            assignment = partition.assignment_for(value.conserved.sample_id)
            expected_group = value.conserved.key.group_key(partition.plan.level)
            if assignment.group_key != expected_group:
                raise ValueError(
                    "Partition group identity does not match the conserved sample key."
                )

        split_pairs = {
            split: tuple(
                value
                for value in values
                if partition.assignment_for(value.conserved.sample_id).split == split
            )
            for split in ("train", "validation", "test")
        }
        training = split_pairs["train"]
        if not training:
            raise ValueError(
                "Energy-equilibrium preparation requires a nonempty training split."
            )
        training_samples = tuple(value.conserved for value in training)
        training_assignments = tuple(
            partition.assignment_for(value.sample_id) for value in training_samples
        )
        provenance = normalizer.provenance
        authoritative_normalizer = TrainOnlyNormalizer.fit(
            tuple(value.conserved for value in values),
            partition,
            feature_name=provenance.feature_name,
            epsilon=normalizer.epsilon,
        )
        if (
            provenance.partition_id != partition.partition_id
            or provenance.schema_id != schema
            or provenance.training_sample_ids
            != tuple(value.sample_id for value in training_samples)
            or provenance.training_assignment_ids
            != tuple(value.assignment_id for value in training_assignments)
            or normalizer.mean.shape != (4,)
            or normalizer.scale.shape != (4,)
            or normalizer.normalizer_id != authoritative_normalizer.normalizer_id
        ):
            raise ValueError(
                "Normalizer provenance must identify the authoritative conserved "
                "training samples."
            )

        self.pairs = values
        self.train_pairs = training
        self.validation_pairs = split_pairs["validation"]
        self.test_pairs = split_pairs["test"]
        self.manifest = manifest
        self.partition = partition
        self.normalizer = normalizer
        self.schema_id = schema
        self.quadrature_id = quadrature
        self.material_id = material
        self.oracle_plan_id = oracle
        self.preparation_id = canonical_fingerprint(
            {
                "kind": "prepared-energy-equilibrium-dataset",
                "pairs": [value.pair_id for value in values],
                "manifest": manifest.manifest_id,
                "partition": partition.partition_id,
                "normalizer": normalizer.normalizer_id,
                "schema": schema,
                "quadrature": quadrature,
                "material": material,
                "oracle_plan": oracle,
                "splits": {
                    split: [value.pair_id for value in split_pairs[split]]
                    for split in ("train", "validation", "test")
                },
            }
        )


def prepare_energy_equilibrium_dataset(
    pairs: tuple[EnergyEquilibriumTrainingPair, ...],
    manifest: ChunkedClosureDatasetManifest,
    partition: LeakageSafePartition,
    normalizer: TrainOnlyNormalizer,
    /,
) -> PreparedEnergyEquilibriumDataset:
    """Validate and bind aligned oracle pairs to one manifest and partition."""

    return PreparedEnergyEquilibriumDataset(pairs, manifest, partition, normalizer)


class LearnedEnergyEquilibriumBindingPlan(StrictModule, NonTrainableState):
    """Semantic deployment ABI for a learned positive-energy dual predictor."""

    equilibrium_plan: PositiveEnergyEquilibriumPlan
    schema: FlowStateSchema
    input_component_names: tuple[str, ...] = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    dataset_preparation_id: str = eqx.field(static=True)
    semantic_provenance: SemanticProvenance
    population_convention: str = eqx.field(static=True)
    normalizer_id: str = eqx.field(static=True)
    normalizer_provenance_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium_plan: PositiveEnergyEquilibriumPlan,
        schema: FlowStateSchema,
        dataset: PreparedEnergyEquilibriumDataset,
        semantic_provenance: SemanticProvenance,
        /,
        *,
        input_component_names: tuple[str, ...],
        material_id: str,
    ):
        if not isinstance(equilibrium_plan, PositiveEnergyEquilibriumPlan):
            raise TypeError("equilibrium_plan must be a PositiveEnergyEquilibriumPlan.")
        if not isinstance(schema, FlowStateSchema):
            raise TypeError("schema must be a FlowStateSchema.")
        if not isinstance(dataset, PreparedEnergyEquilibriumDataset):
            raise TypeError("dataset must be a PreparedEnergyEquilibriumDataset.")
        if not isinstance(semantic_provenance, SemanticProvenance):
            raise TypeError("semantic_provenance must be SemanticProvenance.")
        components = tuple(str(value).strip() for value in input_component_names)
        material = str(material_id).strip()
        if schema.component_count != 4 or components != schema.component_names:
            raise ValueError(
                "The binding input ABI must be the ordered four-component flow schema."
            )
        if not material or material != dataset.material_id:
            raise ValueError("Binding and dataset material identities do not match.")
        if dataset.schema_id != schema.schema_id:
            raise ValueError("Binding and dataset schema identities do not match.")
        if dataset.quadrature_id != equilibrium_plan.quadrature.quadrature_id:
            raise ValueError("Binding and dataset quadrature identities do not match.")
        if dataset.oracle_plan_id != equilibrium_plan.plan_id:
            raise ValueError("Binding and dataset oracle-plan identities do not match.")
        if equilibrium_plan.population_convention != _ENERGY_POPULATION_CONVENTION:
            raise ValueError("Energy-equilibrium population convention is unsupported.")
        self.equilibrium_plan = equilibrium_plan
        self.schema = schema
        self.input_component_names = components
        self.material_id = material
        self.dataset_preparation_id = dataset.preparation_id
        self.semantic_provenance = semantic_provenance
        self.population_convention = _ENERGY_POPULATION_CONVENTION
        self.normalizer_id = dataset.normalizer.normalizer_id
        self.normalizer_provenance_id = dataset.normalizer.provenance.provenance_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "learned-energy-equilibrium-binding-plan",
                "equilibrium_plan": equilibrium_plan.plan_id,
                "schema": schema.schema_id,
                "input_components": list(components),
                "material": material,
                "dataset_preparation": dataset.preparation_id,
                "semantic_provenance": semantic_provenance.semantic_id,
                "population_convention": self.population_convention,
                "normalizer": self.normalizer_id,
                "normalizer_provenance": self.normalizer_provenance_id,
            }
        )

    def prepare(
        self,
        model: AbstractArrayModel,
        numeric_revision: NumericRevision,
        normalizer: TrainOnlyNormalizer,
        /,
    ) -> PreparedLearnedEnergyEquilibriumBinding:
        """Validate numeric identity and freeze one deterministic 4→2 model."""

        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        if model.in_size != 4 or model.out_size != 2:
            raise ValueError("Energy-equilibrium models must have sizes 4→2.")
        if not isinstance(numeric_revision, NumericRevision):
            raise TypeError("numeric_revision must be a NumericRevision.")
        if not isinstance(normalizer, TrainOnlyNormalizer):
            raise TypeError("normalizer must be a TrainOnlyNormalizer.")
        expected_revision = energy_equilibrium_numeric_revision(
            self.semantic_provenance, model
        )
        if (
            numeric_revision.semantic_id != self.semantic_provenance.semantic_id
            or numeric_revision.revision_id != expected_revision.revision_id
        ):
            raise ValueError(
                "Numeric revision does not match the binding semantics and model."
            )
        if (
            normalizer.normalizer_id != self.normalizer_id
            or normalizer.provenance.provenance_id != self.normalizer_provenance_id
            or normalizer.provenance.schema_id != self.schema.schema_id
            or normalizer.mean.shape != (4,)
            or normalizer.scale.shape != (4,)
        ):
            raise ValueError("Normalizer provenance does not match the binding plan.")
        return PreparedLearnedEnergyEquilibriumBinding(
            FrozenModel(model), numeric_revision, normalizer, self
        )


class PreparedLearnedEnergyEquilibriumBinding(StrictModule, NonTrainableState):
    """Frozen, vectorized learned-dual deployment for one equilibrium plan."""

    model: FrozenModel
    numeric_revision: NumericRevision
    normalizer: TrainOnlyNormalizer
    plan: LearnedEnergyEquilibriumBindingPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: FrozenModel,
        numeric_revision: NumericRevision,
        normalizer: TrainOnlyNormalizer,
        plan: LearnedEnergyEquilibriumBindingPlan,
        /,
    ):
        if not isinstance(model, FrozenModel):
            raise TypeError("model must be a FrozenModel.")
        if not isinstance(numeric_revision, NumericRevision):
            raise TypeError("numeric_revision must be a NumericRevision.")
        if not isinstance(normalizer, TrainOnlyNormalizer):
            raise TypeError("normalizer must be a TrainOnlyNormalizer.")
        if not isinstance(plan, LearnedEnergyEquilibriumBindingPlan):
            raise TypeError("plan must be a LearnedEnergyEquilibriumBindingPlan.")
        self.model = model
        self.numeric_revision = numeric_revision
        self.normalizer = normalizer
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-learned-energy-equilibrium-binding",
                "plan": plan.plan_id,
                "numeric_revision": numeric_revision.revision_id,
                "normalizer": normalizer.normalizer_id,
                "population_convention": plan.population_convention,
            }
        )

    def predict_dual(self, conserved: ArrayLike, /) -> Array:
        """Predict oracle coordinates pointwise over arbitrary leading axes."""

        values = self.plan.schema.validate(
            conserved, owner="Energy-equilibrium conserved state"
        )
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("Energy-equilibrium conserved state must be real floating.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Energy-equilibrium conserved state must be finite.",
        )
        normalized = self.normalizer.normalize(values)
        flat = normalized.reshape((-1, 4))
        prediction = jax.vmap(lambda point: self.model(point, key=None))(flat)
        if prediction.shape != (flat.shape[0], 2):
            raise ValueError(
                "Energy-equilibrium model output must have trailing shape (2,)."
            )
        dual = prediction.reshape(values.shape[:-1] + (2,))
        return eqx.error_if(
            dual,
            jnp.any(~jnp.isfinite(dual)),
            "Energy-equilibrium model prediction must be finite.",
        )

    def evaluate(
        self,
        total_energy: ArrayLike,
        target_flux: ArrayLike,
        conserved: ArrayLike,
        /,
    ) -> EnergyEquilibriumResult:
        """Evaluate positive populations from the learned dual without hidden inputs."""

        dual = self.predict_dual(conserved)
        return self.plan.equilibrium_plan.evaluate(total_energy, target_flux, dual)


__all__ = [
    "energy_equilibrium_numeric_revision",
    "EnergyEquilibriumTrainingPair",
    "LearnedEnergyEquilibriumBindingPlan",
    "PreparedEnergyEquilibriumDataset",
    "PreparedLearnedEnergyEquilibriumBinding",
    "prepare_energy_equilibrium_dataset",
]
