#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .. import ein
from .._fingerprint import canonical_fingerprint
from .._identity import NumericRevision, SemanticProvenance
from .._model import AbstractArrayModel, FrozenModel
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.discrete_velocity._energy_equilibrium import (
    EnergyEquilibriumResult,
    PositiveEnergyEquilibriumPlan,
)
from ..equations._materials import IdealGasMaterial
from ._dataset import (
    ChunkedClosureDatasetManifest,
    ClosureSample,
    LeakageSafePartition,
    TrainOnlyNormalizer,
)
from ._state import FlowStateSchema


_ENERGY_POPULATION_CONVENTION = "weight_absorbed_total_energy_sum"


def energy_equilibrium_numeric_revision(
    semantic_provenance: SemanticProvenance | str,
    model: AbstractArrayModel,
    /,
) -> NumericRevision:
    """Bind one model's dynamic array leaves to declared equilibrium semantics."""

    if isinstance(semantic_provenance, SemanticProvenance):
        semantic = semantic_provenance.semantic_id
    elif isinstance(semantic_provenance, str):
        semantic = semantic_provenance.strip()
    else:
        raise TypeError("semantic_provenance must be SemanticProvenance or str.")
    if not semantic:
        raise ValueError("semantic_provenance must identify non-empty semantics.")
    if not isinstance(model, AbstractArrayModel):
        raise TypeError("model must be an AbstractArrayModel.")
    leaves = tuple(leaf for leaf in jax.tree.leaves(model) if eqx.is_array(leaf))
    if not leaves:
        raise ValueError("Energy-equilibrium models require dynamic array leaves.")
    return NumericRevision(semantic, {"model_array_leaves": leaves})


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


class EnergyEquilibriumSupportEnvelope(StrictModule, NonTrainableState):
    """Closed primitive-state support certified for one learned dual model."""

    rho_bounds: tuple[float, float] = eqx.field(static=True)
    u_x_bounds: tuple[float, float] = eqx.field(static=True)
    u_y_bounds: tuple[float, float] = eqx.field(static=True)
    temperature_bounds: tuple[float, float] = eqx.field(static=True)
    maximum_mach: float = eqx.field(static=True)
    minimum_hull_margin: float = eqx.field(static=True)
    minimum_particle_equilibrium_margin: float = eqx.field(static=True)
    primitive_component_names: tuple[str, ...] = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    normalizer_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    equilibrium_plan_id: str = eqx.field(static=True)
    training_preparation_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        rho_bounds: tuple[float, float],
        u_x_bounds: tuple[float, float],
        u_y_bounds: tuple[float, float],
        temperature_bounds: tuple[float, float],
        maximum_mach: float,
        minimum_hull_margin: float,
        minimum_particle_equilibrium_margin: float,
        schema_id: str,
        material_id: str,
        normalizer_id: str,
        quadrature_id: str,
        equilibrium_plan_id: str,
        training_preparation_id: str,
    ):
        bounds = tuple(
            tuple(float(endpoint) for endpoint in value)
            for value in (
                rho_bounds,
                u_x_bounds,
                u_y_bounds,
                temperature_bounds,
            )
        )
        if any(
            len(value) != 2
            or not all(np.isfinite(endpoint) for endpoint in value)
            or value[0] >= value[1]
            for value in bounds
        ):
            raise ValueError("Primitive-state support bounds must be finite intervals.")
        if bounds[0][0] <= 0.0 or bounds[3][0] <= 0.0:
            raise ValueError("Density and temperature support must be positive.")
        mach = float(maximum_mach)
        hull = float(minimum_hull_margin)
        particle = float(minimum_particle_equilibrium_margin)
        if (
            not np.isfinite(mach)
            or mach <= 0.0
            or not np.isfinite(hull)
            or hull < 0.0
            or not np.isfinite(particle)
            or particle < 0.0
        ):
            raise ValueError("Energy-equilibrium support margins are invalid.")
        identifiers = tuple(
            str(value).strip()
            for value in (
                schema_id,
                material_id,
                normalizer_id,
                quadrature_id,
                equilibrium_plan_id,
                training_preparation_id,
            )
        )
        if any(not value for value in identifiers):
            raise ValueError("Energy-equilibrium support identities must be non-empty.")
        self.rho_bounds = bounds[0]
        self.u_x_bounds = bounds[1]
        self.u_y_bounds = bounds[2]
        self.temperature_bounds = bounds[3]
        self.maximum_mach = mach
        self.minimum_hull_margin = hull
        self.minimum_particle_equilibrium_margin = particle
        self.primitive_component_names = ("rho", "u_x", "u_y", "temperature")
        (
            self.schema_id,
            self.material_id,
            self.normalizer_id,
            self.quadrature_id,
            self.equilibrium_plan_id,
            self.training_preparation_id,
        ) = identifiers
        self.support_id = canonical_fingerprint(
            {
                "kind": "energy-equilibrium-primitive-support",
                "primitive_components": list(self.primitive_component_names),
                "bounds": {
                    "rho": list(self.rho_bounds),
                    "u_x": list(self.u_x_bounds),
                    "u_y": list(self.u_y_bounds),
                    "temperature": list(self.temperature_bounds),
                },
                "maximum_mach": mach,
                "minimum_hull_margin": hull,
                "minimum_particle_equilibrium_margin": particle,
                "schema": self.schema_id,
                "material": self.material_id,
                "normalizer": self.normalizer_id,
                "quadrature": self.quadrature_id,
                "equilibrium_plan": self.equilibrium_plan_id,
                "training_preparation": self.training_preparation_id,
            }
        )


class EnergyEquilibriumSupportEvidence(StrictModule):
    """Per-lane primitive, realizability, and learned-model support evidence."""

    primitive_state: Array
    rho_margin: Array
    u_x_margin: Array
    u_y_margin: Array
    temperature_margin: Array
    mach: Array
    mach_margin: Array
    hull_margin: Array
    minimum_particle_equilibrium_population: Array
    particle_equilibrium_margin: Array
    finite: Array
    primitive_supported: Array
    model_output_finite: Array
    successful: Array
    support_id: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    material_id: str = eqx.field(static=True)
    quadrature_id: str = eqx.field(static=True)
    normalizer_id: str = eqx.field(static=True)
    equilibrium_plan_id: str = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    training_preparation_id: str = eqx.field(static=True)
    parent_artifact_id: str | None = eqx.field(static=True)


def _closed_interval_margin(value: Array, bounds: tuple[float, float], /) -> Array:
    lower, upper = bounds
    return jnp.minimum(value - lower, upper - value)


class LearnedEnergyEquilibriumBindingPlan(StrictModule, NonTrainableState):
    """Semantic deployment ABI for a learned positive-energy dual predictor."""

    equilibrium_plan: PositiveEnergyEquilibriumPlan
    schema: FlowStateSchema
    material: IdealGasMaterial
    normalizer: TrainOnlyNormalizer
    support: EnergyEquilibriumSupportEnvelope
    particle_moment_matrix: Array
    particle_moment_lift: Array
    input_component_names: tuple[str, ...] = eqx.field(static=True)
    semantic_id: str = eqx.field(static=True)
    training_preparation_id: str = eqx.field(static=True)
    parent_artifact_id: str | None = eqx.field(static=True)
    population_convention: str = eqx.field(static=True)
    normalizer_id: str = eqx.field(static=True)
    normalizer_provenance_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium_plan: PositiveEnergyEquilibriumPlan,
        schema: FlowStateSchema,
        material: IdealGasMaterial,
        normalizer: TrainOnlyNormalizer,
        support: EnergyEquilibriumSupportEnvelope,
        /,
        *,
        input_component_names: tuple[str, ...],
        semantic_id: str,
        training_preparation_id: str,
        parent_artifact_id: str | None = None,
    ):
        if not isinstance(equilibrium_plan, PositiveEnergyEquilibriumPlan):
            raise TypeError("equilibrium_plan must be a PositiveEnergyEquilibriumPlan.")
        if not isinstance(schema, FlowStateSchema):
            raise TypeError("schema must be a FlowStateSchema.")
        if not isinstance(material, IdealGasMaterial):
            raise TypeError("material must be an IdealGasMaterial.")
        if not isinstance(normalizer, TrainOnlyNormalizer):
            raise TypeError("normalizer must be a TrainOnlyNormalizer.")
        if not isinstance(support, EnergyEquilibriumSupportEnvelope):
            raise TypeError("support must be an EnergyEquilibriumSupportEnvelope.")
        components = tuple(str(value).strip() for value in input_component_names)
        semantic = str(semantic_id).strip()
        training = str(training_preparation_id).strip()
        parent = None if parent_artifact_id is None else str(parent_artifact_id).strip()
        if schema.component_count != 4 or components != schema.component_names:
            raise ValueError(
                "The binding input ABI must be the ordered four-component flow schema."
            )
        if not semantic or not training or parent == "":
            raise ValueError("Binding semantic and training identities are invalid.")
        quadrature = equilibrium_plan.quadrature
        if equilibrium_plan.population_convention != _ENERGY_POPULATION_CONVENTION:
            raise ValueError("Energy-equilibrium population convention is unsupported.")
        if (
            normalizer.provenance.schema_id != schema.schema_id
            or normalizer.mean.shape != (4,)
            or normalizer.scale.shape != (4,)
        ):
            raise ValueError("Normalizer provenance does not match the binding schema.")
        if support.minimum_hull_margin <= equilibrium_plan.interior_tolerance:
            raise ValueError(
                "Support hull margin must exceed the equilibrium interior tolerance."
            )
        if (
            support.schema_id != schema.schema_id
            or support.material_id != material.material_id
            or support.normalizer_id != normalizer.normalizer_id
            or support.quadrature_id != quadrature.quadrature_id
            or support.equilibrium_plan_id != equilibrium_plan.plan_id
            or (parent is None and support.training_preparation_id != training)
        ):
            raise ValueError("Support identities do not match the binding plan.")

        velocities = np.asarray(quadrature.velocities)
        moment_matrix = np.stack(
            (
                np.ones((quadrature.population_count,), dtype=velocities.dtype),
                velocities[:, 0],
                velocities[:, 1],
                velocities[:, 0] ** 2,
                velocities[:, 0] * velocities[:, 1],
                velocities[:, 1] ** 2,
            ),
            axis=0,
        )
        moment_lift = np.linalg.solve(moment_matrix @ moment_matrix.T, moment_matrix).T
        self.equilibrium_plan = equilibrium_plan
        self.schema = schema
        self.material = material
        self.normalizer = normalizer
        self.support = support
        self.particle_moment_matrix = jnp.asarray(
            moment_matrix, dtype=quadrature.velocities.dtype
        )
        self.particle_moment_lift = jnp.asarray(
            moment_lift, dtype=quadrature.velocities.dtype
        )
        self.input_component_names = components
        self.semantic_id = semantic
        self.training_preparation_id = training
        self.parent_artifact_id = parent
        self.population_convention = _ENERGY_POPULATION_CONVENTION
        self.normalizer_id = normalizer.normalizer_id
        self.normalizer_provenance_id = normalizer.provenance.provenance_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "learned-energy-equilibrium-binding-plan",
                "equilibrium_plan": equilibrium_plan.plan_id,
                "schema": schema.schema_id,
                "input_components": list(components),
                "material": material.material_id,
                "normalizer": normalizer.normalizer_id,
                "normalizer_provenance": normalizer.provenance.provenance_id,
                "support": support.support_id,
                "semantic": semantic,
                "training_preparation": training,
                "parent_artifact": parent,
                "population_convention": self.population_convention,
            }
        )

    def _particle_equilibrium_minimum(
        self,
        density: Array,
        momentum: Array,
        velocity: Array,
        pressure: Array,
        /,
    ) -> Array:
        quadrature = self.equilibrium_plan.quadrature
        velocities = quadrature.velocities
        temperature_0 = jnp.asarray(
            quadrature.reference_temperature, dtype=velocities.dtype
        )
        theta = pressure / density
        projected_velocity = ein.contract("...d,qd->...q", velocity, velocities)
        velocity_square = jnp.sum(velocity**2, axis=-1)
        speed_square = jnp.sum(velocities**2, axis=-1)
        particle_raw = (
            quadrature.weights
            * density[..., None]
            * (
                1.0
                + projected_velocity / temperature_0
                + (
                    projected_velocity**2
                    - temperature_0 * velocity_square[..., None]
                    + (theta - temperature_0)[..., None]
                    * (speed_square - 2.0 * temperature_0)
                )
                / (2.0 * temperature_0**2)
            )
        )
        target_moments = jnp.stack(
            (
                density,
                momentum[..., 0],
                momentum[..., 1],
                momentum[..., 0] ** 2 / density + pressure,
                momentum[..., 0] * momentum[..., 1] / density,
                momentum[..., 1] ** 2 / density + pressure,
            ),
            axis=-1,
        )
        recovered_moments = ein.contract(
            "mq,...q->...m", self.particle_moment_matrix, particle_raw
        )
        particle_equilibrium = particle_raw + ein.contract(
            "qm,...m->...q",
            self.particle_moment_lift,
            target_moments - recovered_moments,
        )
        return jnp.min(particle_equilibrium, axis=-1)

    def _support_quantities(
        self, conserved: Array, /
    ) -> tuple[
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
        Array,
    ]:
        finite_input = jnp.all(jnp.isfinite(conserved), axis=-1)
        raw_density = conserved[..., 0]
        valid_density = finite_input & (raw_density > 0.0)
        density = jnp.where(valid_density, raw_density, 1.0)
        momentum = jnp.where(jnp.isfinite(conserved[..., 1:3]), conserved[..., 1:3], 0.0)
        total_energy = jnp.where(
            jnp.isfinite(conserved[..., -1]), conserved[..., -1], 1.0
        )
        velocity = momentum / density[..., None]
        kinetic_energy = 0.5 * jnp.sum(momentum * velocity, axis=-1)
        specific_internal_energy = (total_energy - kinetic_energy) / density
        pressure = self.material.pressure(density, specific_internal_energy)
        temperature = self.material.temperature(density, pressure)
        thermodynamic = (
            valid_density
            & jnp.isfinite(pressure)
            & jnp.isfinite(temperature)
            & self.material.admissible(raw_density, pressure)
            & (temperature > 0.0)
            & (total_energy > 0.0)
        )
        safe_pressure = jnp.where(thermodynamic, pressure, 1.0)
        safe_total_energy = jnp.where(thermodynamic, total_energy, 1.0)
        safe_momentum = jnp.where(thermodynamic[..., None], momentum, 0.0)
        safe_velocity = jnp.where(thermodynamic[..., None], velocity, 0.0)
        sound_speed = self.material.sound_speed(density, safe_pressure)
        mach = jnp.sqrt(jnp.sum(safe_velocity**2, axis=-1)) / sound_speed
        primitive = jnp.concatenate(
            (
                raw_density[..., None],
                velocity,
                temperature[..., None],
            ),
            axis=-1,
        )
        rho_margin = _closed_interval_margin(raw_density, self.support.rho_bounds)
        u_x_margin = _closed_interval_margin(velocity[..., 0], self.support.u_x_bounds)
        u_y_margin = _closed_interval_margin(velocity[..., 1], self.support.u_y_bounds)
        temperature_margin = _closed_interval_margin(
            temperature, self.support.temperature_bounds
        )
        mach_margin = self.support.maximum_mach - mach

        normalized_target_flux = (
            (safe_total_energy + safe_pressure)[..., None]
            * safe_velocity
            / safe_total_energy[..., None]
        )
        raw_hull_margin = jnp.min(
            self.equilibrium_plan.halfspace_offsets
            - ein.contract(
                "hd,...d->...h",
                self.equilibrium_plan.halfspace_normals,
                normalized_target_flux,
            ),
            axis=-1,
        )
        hull_margin = raw_hull_margin
        minimum_particle = self._particle_equilibrium_minimum(
            density, safe_momentum, safe_velocity, safe_pressure
        )
        particle_margin = minimum_particle
        finite = (
            thermodynamic
            & jnp.all(jnp.isfinite(primitive), axis=-1)
            & jnp.isfinite(mach)
            & jnp.isfinite(raw_hull_margin)
            & jnp.isfinite(minimum_particle)
        )
        primitive_supported = (
            finite
            & (rho_margin >= 0.0)
            & (u_x_margin >= 0.0)
            & (u_y_margin >= 0.0)
            & (temperature_margin >= 0.0)
            & (mach_margin >= 0.0)
            & (hull_margin >= self.support.minimum_hull_margin)
            & (particle_margin >= self.support.minimum_particle_equilibrium_margin)
        )
        return (
            primitive,
            rho_margin,
            u_x_margin,
            u_y_margin,
            temperature_margin,
            mach,
            mach_margin,
            hull_margin,
            minimum_particle,
            particle_margin,
            finite,
            primitive_supported,
        )

    def predict_dual_with_evidence(
        self,
        model: AbstractArrayModel,
        conserved: ArrayLike,
        /,
    ) -> tuple[Array, EnergyEquilibriumSupportEvidence]:
        """Evaluate an explicit model safely while retaining per-lane support evidence."""

        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        if model.in_size != 4 or model.out_size != 2:
            raise ValueError("Energy-equilibrium models must have sizes 4→2.")
        values = self.schema.validate(
            conserved, owner="Energy-equilibrium conserved state"
        )
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("Energy-equilibrium conserved state must be real floating.")
        (
            primitive,
            rho_margin,
            u_x_margin,
            u_y_margin,
            temperature_margin,
            mach,
            mach_margin,
            hull_margin,
            minimum_particle,
            particle_margin,
            finite,
            primitive_supported,
        ) = self._support_quantities(values)
        safe_values = jnp.where(
            primitive_supported[..., None],
            values,
            jnp.asarray(self.normalizer.mean, dtype=values.dtype),
        )
        normalized = self.normalizer.normalize(safe_values)
        flat = normalized.reshape((-1, 4))
        prediction = jax.vmap(lambda point: model(point, key=None))(flat)
        if prediction.shape != (flat.shape[0], 2):
            raise ValueError(
                "Energy-equilibrium model output must have trailing shape (2,)."
            )
        if not jnp.issubdtype(prediction.dtype, jnp.floating):
            raise TypeError("Energy-equilibrium model output must be real floating.")
        prediction = prediction.reshape(values.shape[:-1] + (2,))
        model_output_finite = jnp.all(jnp.isfinite(prediction), axis=-1)
        successful = primitive_supported & model_output_finite
        dual = jnp.where(successful[..., None], prediction, jnp.zeros_like(prediction))
        evidence = EnergyEquilibriumSupportEvidence(
            primitive_state=primitive,
            rho_margin=rho_margin,
            u_x_margin=u_x_margin,
            u_y_margin=u_y_margin,
            temperature_margin=temperature_margin,
            mach=mach,
            mach_margin=mach_margin,
            hull_margin=hull_margin,
            minimum_particle_equilibrium_population=minimum_particle,
            particle_equilibrium_margin=particle_margin,
            finite=finite,
            primitive_supported=primitive_supported,
            model_output_finite=model_output_finite,
            successful=successful,
            support_id=self.support.support_id,
            normalizer_id=self.normalizer.normalizer_id,
            schema_id=self.schema.schema_id,
            material_id=self.material.material_id,
            quadrature_id=self.equilibrium_plan.quadrature.quadrature_id,
            equilibrium_plan_id=self.equilibrium_plan.plan_id,
            semantic_id=self.semantic_id,
            training_preparation_id=self.training_preparation_id,
            parent_artifact_id=self.parent_artifact_id,
        )
        return dual, evidence

    def prepare(
        self,
        model: AbstractArrayModel,
        numeric_revision: NumericRevision,
        /,
    ) -> PreparedLearnedEnergyEquilibriumBinding:
        """Validate numeric identity and freeze one deterministic 4→2 model."""

        if not isinstance(model, AbstractArrayModel):
            raise TypeError("model must be an AbstractArrayModel.")
        if model.in_size != 4 or model.out_size != 2:
            raise ValueError("Energy-equilibrium models must have sizes 4→2.")
        if not isinstance(numeric_revision, NumericRevision):
            raise TypeError("numeric_revision must be a NumericRevision.")
        expected_revision = energy_equilibrium_numeric_revision(self.semantic_id, model)
        if (
            numeric_revision.semantic_id != self.semantic_id
            or numeric_revision.revision_id != expected_revision.revision_id
        ):
            raise ValueError(
                "Numeric revision does not match the binding semantics and model."
            )
        return PreparedLearnedEnergyEquilibriumBinding(
            FrozenModel(model), numeric_revision, self
        )


class PreparedLearnedEnergyEquilibriumBinding(StrictModule, NonTrainableState):
    """Frozen, support-aware learned-dual deployment for one equilibrium plan."""

    model: FrozenModel
    numeric_revision: NumericRevision
    plan: LearnedEnergyEquilibriumBindingPlan
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        model: FrozenModel,
        numeric_revision: NumericRevision,
        plan: LearnedEnergyEquilibriumBindingPlan,
        /,
    ):
        if not isinstance(model, FrozenModel):
            raise TypeError("model must be a FrozenModel.")
        if not isinstance(numeric_revision, NumericRevision):
            raise TypeError("numeric_revision must be a NumericRevision.")
        if not isinstance(plan, LearnedEnergyEquilibriumBindingPlan):
            raise TypeError("plan must be a LearnedEnergyEquilibriumBindingPlan.")
        expected_revision = energy_equilibrium_numeric_revision(
            plan.semantic_id, model.as_trainable()
        )
        if (
            model.in_size != 4
            or model.out_size != 2
            or numeric_revision.semantic_id != plan.semantic_id
            or numeric_revision.revision_id != expected_revision.revision_id
        ):
            raise ValueError("Frozen model revision does not match the binding plan.")
        self.model = model
        self.numeric_revision = numeric_revision
        self.plan = plan
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-learned-energy-equilibrium-binding",
                "plan": plan.plan_id,
                "numeric_revision": numeric_revision.revision_id,
                "normalizer": plan.normalizer.normalizer_id,
                "support": plan.support.support_id,
                "population_convention": plan.population_convention,
            }
        )

    def predict_dual_with_evidence(
        self, conserved: ArrayLike, /
    ) -> tuple[Array, EnergyEquilibriumSupportEvidence]:
        """Predict duals with explicit per-lane support and model-finiteness evidence."""

        return self.plan.predict_dual_with_evidence(self.model, conserved)

    def predict_dual(self, conserved: ArrayLike, /) -> Array:
        """Predict duals strictly, refusing any lane outside declared support."""

        dual, evidence = self.predict_dual_with_evidence(conserved)
        return eqx.error_if(
            dual,
            jnp.any(~evidence.successful),
            "Energy-equilibrium prediction is outside declared support.",
        )

    def evaluate(
        self,
        total_energy: ArrayLike,
        target_flux: ArrayLike,
        conserved: ArrayLike,
        /,
    ) -> EnergyEquilibriumResult:
        """Evaluate positive populations from a strictly supported learned dual."""

        dual = self.predict_dual(conserved)
        return self.plan.equilibrium_plan.evaluate(total_energy, target_flux, dual)


__all__ = [
    "energy_equilibrium_numeric_revision",
    "EnergyEquilibriumSupportEnvelope",
    "EnergyEquilibriumSupportEvidence",
    "EnergyEquilibriumTrainingPair",
    "LearnedEnergyEquilibriumBindingPlan",
    "PreparedEnergyEquilibriumDataset",
    "PreparedLearnedEnergyEquilibriumBinding",
    "prepare_energy_equilibrium_dataset",
]
