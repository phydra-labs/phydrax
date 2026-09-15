#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._array_archive import ArrayArchiveCorruptionError
from .._model import AbstractArrayModel, FrozenModel
from ..discretization.discrete_velocity._energy_equilibrium import (
    PositiveEnergyEquilibriumPlan,
)
from ..discretization.discrete_velocity._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
)
from ..equations._materials import IdealGasMaterial
from ..ml.artifacts import MLArtifactManifest, read_ml_artifact, save_ml_artifact
from ._dataset import NormalizerProvenance, TrainOnlyNormalizer
from ._kinetic_equilibrium import (
    energy_equilibrium_numeric_revision,
    EnergyEquilibriumSupportEnvelope,
    LearnedEnergyEquilibriumBindingPlan,
    PreparedLearnedEnergyEquilibriumBinding,
)
from ._state import FlowStateSchema


_FEATURE_KIND = "learned-energy-equilibrium-features"
_TARGET_KIND = "learned-energy-equilibrium-target"
_PROVENANCE_KIND = "learned-energy-equilibrium-artifact"
_FEATURE_FIELDS = {
    "kind",
    "flow_schema",
    "input_component_names",
    "normalizer",
    "support",
}
_TARGET_FIELDS = {
    "kind",
    "output_component_names",
    "material",
    "equilibrium_plan",
    "population_convention",
}
_PROVENANCE_FIELDS = {
    "kind",
    "schema_id",
    "material_id",
    "quadrature_id",
    "equilibrium_plan_id",
    "normalizer_id",
    "normalizer_provenance_id",
    "support_id",
    "semantic_id",
    "training_preparation_id",
    "parent_artifact_id",
    "binding_plan_id",
    "numeric_revision_id",
    "prepared_binding_id",
}


@dataclass(frozen=True, slots=True)
class LearnedEnergyEquilibriumArtifact:
    """A restored frozen binding and its checksum-validated ML manifest."""

    binding: PreparedLearnedEnergyEquilibriumBinding
    manifest: MLArtifactManifest
    artifact_id: str


def _schema_record(schema: FlowStateSchema, /) -> dict[str, Any]:
    return {
        "component_names": list(schema.component_names),
        "component_units": list(schema.component_units),
        "reference_scales": list(schema.reference_scales),
        "density_name": schema.density_name,
        "velocity_names": list(schema.velocity_names),
        "species_names": list(schema.species_names),
        "total_energy_name": schema.total_energy_name,
        "enthalpy_name": schema.enthalpy_name,
        "component_axis": schema.component_axis,
        "schema_id": schema.schema_id,
    }


def _normalizer_record(normalizer: TrainOnlyNormalizer, /) -> dict[str, Any]:
    provenance = normalizer.provenance
    mean = np.asarray(normalizer.mean)
    scale = np.asarray(normalizer.scale)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "mean_dtype": mean.dtype.str,
        "scale_dtype": scale.dtype.str,
        "epsilon": normalizer.epsilon,
        "partition_id": provenance.partition_id,
        "training_assignment_ids": list(provenance.training_assignment_ids),
        "training_sample_ids": list(provenance.training_sample_ids),
        "feature_name": provenance.feature_name,
        "schema_id": provenance.schema_id,
        "provenance_id": provenance.provenance_id,
        "normalizer_id": normalizer.normalizer_id,
    }


def _support_record(support: EnergyEquilibriumSupportEnvelope, /) -> dict[str, Any]:
    return {
        "rho_bounds": list(support.rho_bounds),
        "u_x_bounds": list(support.u_x_bounds),
        "u_y_bounds": list(support.u_y_bounds),
        "temperature_bounds": list(support.temperature_bounds),
        "maximum_mach": support.maximum_mach,
        "minimum_hull_margin": support.minimum_hull_margin,
        "minimum_particle_equilibrium_margin": (
            support.minimum_particle_equilibrium_margin
        ),
        "primitive_component_names": list(support.primitive_component_names),
        "schema_id": support.schema_id,
        "material_id": support.material_id,
        "normalizer_id": support.normalizer_id,
        "quadrature_id": support.quadrature_id,
        "equilibrium_plan_id": support.equilibrium_plan_id,
        "training_preparation_id": support.training_preparation_id,
        "support_id": support.support_id,
    }


def _material_record(material: IdealGasMaterial, /) -> dict[str, Any]:
    return {
        "kind": "ideal-gas-material",
        "gamma": material.gamma,
        "gas_constant": material.gas_constant,
        "density_floor": material.density_floor,
        "pressure_floor": material.pressure_floor,
        "material_id": material.material_id,
    }


def _quadrature_record(
    quadrature: CertifiedDiscreteVelocityQuadrature, /
) -> dict[str, Any]:
    velocities = np.asarray(quadrature.velocities)
    weights = np.asarray(quadrature.weights)
    return {
        "name": quadrature.name,
        "velocities": velocities.tolist(),
        "weights": weights.tolist(),
        "dtype": velocities.dtype.str,
        "reference_temperature": quadrature.reference_temperature,
        "certified_degree": quadrature.certification.maximum_degree,
        "transport_kind": quadrature.transport_kind,
        "tolerance": quadrature.certification.tolerance,
        "quadrature_id": quadrature.quadrature_id,
    }


def _equilibrium_plan_record(plan: PositiveEnergyEquilibriumPlan, /) -> dict[str, Any]:
    return {
        "quadrature": _quadrature_record(plan.quadrature),
        "maximum_iterations": plan.maximum_iterations,
        "residual_tolerance": plan.residual_tolerance,
        "interior_tolerance": plan.interior_tolerance,
        "damping": plan.damping,
        "plan_id": plan.plan_id,
    }


def write_learned_energy_equilibrium_artifact(
    path: str | Path,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    /,
    *,
    licenses: Sequence[str] = (),
) -> Path:
    """Write one portable, pickle-free frozen learned-equilibrium deployment."""

    if not isinstance(binding, PreparedLearnedEnergyEquilibriumBinding):
        raise TypeError("binding must be a PreparedLearnedEnergyEquilibriumBinding.")
    destination = Path(path)
    if destination.suffix != ".phxml":
        raise ValueError("Learned energy-equilibrium artifacts require .phxml paths.")
    plan = binding.plan
    model = binding.model.as_trainable()
    expected_revision = energy_equilibrium_numeric_revision(plan.semantic_id, model)
    if (
        expected_revision.revision_id != binding.numeric_revision.revision_id
        or binding.numeric_revision.semantic_id != plan.semantic_id
    ):
        raise ValueError("Binding numeric revision is inconsistent with its model.")
    validated_schema = _schema_from_record(_schema_record(plan.schema))
    validated_material = _material_from_record(_material_record(plan.material))
    validated_normalizer = _normalizer_from_record(_normalizer_record(plan.normalizer))
    validated_equilibrium = _equilibrium_plan_from_record(
        _equilibrium_plan_record(plan.equilibrium_plan)
    )
    validated_support = _support_from_record(_support_record(plan.support))
    validated_plan = LearnedEnergyEquilibriumBindingPlan(
        validated_equilibrium,
        validated_schema,
        validated_material,
        validated_normalizer,
        validated_support,
        input_component_names=plan.input_component_names,
        semantic_id=plan.semantic_id,
        training_preparation_id=plan.training_preparation_id,
        parent_artifact_id=plan.parent_artifact_id,
    )
    validated_binding = validated_plan.prepare(model, expected_revision)
    if (
        validated_plan.plan_id != plan.plan_id
        or validated_binding.prepared_id != binding.prepared_id
    ):
        raise ValueError("Binding identities are inconsistent with owned dependencies.")
    feature_schema = {
        "kind": _FEATURE_KIND,
        "flow_schema": _schema_record(plan.schema),
        "input_component_names": list(plan.input_component_names),
        "normalizer": _normalizer_record(plan.normalizer),
        "support": _support_record(plan.support),
    }
    target_schema = {
        "kind": _TARGET_KIND,
        "output_component_names": ["energy_dual_x", "energy_dual_y"],
        "material": _material_record(plan.material),
        "equilibrium_plan": _equilibrium_plan_record(plan.equilibrium_plan),
        "population_convention": plan.population_convention,
    }
    provenance = {
        "kind": _PROVENANCE_KIND,
        "schema_id": plan.schema.schema_id,
        "material_id": plan.material.material_id,
        "quadrature_id": plan.equilibrium_plan.quadrature.quadrature_id,
        "equilibrium_plan_id": plan.equilibrium_plan.plan_id,
        "normalizer_id": plan.normalizer.normalizer_id,
        "normalizer_provenance_id": plan.normalizer.provenance.provenance_id,
        "support_id": plan.support.support_id,
        "semantic_id": plan.semantic_id,
        "training_preparation_id": plan.training_preparation_id,
        "parent_artifact_id": plan.parent_artifact_id,
        "binding_plan_id": plan.plan_id,
        "numeric_revision_id": binding.numeric_revision.revision_id,
        "prepared_binding_id": binding.prepared_id,
    }
    return save_ml_artifact(
        destination,
        model,
        feature_schema=feature_schema,
        target_schema=target_schema,
        provenance=provenance,
        licenses=licenses,
    )


def _mapping(
    value: Any,
    fields: set[str],
    owner: str,
    /,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ArrayArchiveCorruptionError(f"{owner} fields are invalid.")
    return value


def _identifier(value: Any, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ArrayArchiveCorruptionError(f"{owner} identity is invalid.")
    return value


def _optional_identifier(value: Any, owner: str, /) -> str | None:
    if value is None:
        return None
    return _identifier(value, owner)


def _string_tuple(value: Any, owner: str, /) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ArrayArchiveCorruptionError(f"{owner} must be a string list.")
    return tuple(value)


def _float_pair(value: Any, owner: str, /) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ArrayArchiveCorruptionError(f"{owner} must be a two-value interval.")
    endpoints = tuple(float(item) for item in value)
    if not all(np.isfinite(item) for item in endpoints):
        raise ArrayArchiveCorruptionError(f"{owner} interval is nonfinite.")
    return endpoints


def _schema_from_record(value: Any, /) -> FlowStateSchema:
    fields = {
        "component_names",
        "component_units",
        "reference_scales",
        "density_name",
        "velocity_names",
        "species_names",
        "total_energy_name",
        "enthalpy_name",
        "component_axis",
        "schema_id",
    }
    record = _mapping(value, fields, "Flow-state schema")
    scales = record["reference_scales"]
    if not isinstance(scales, list):
        raise ArrayArchiveCorruptionError("Flow-state reference scales are invalid.")
    schema = FlowStateSchema(
        _string_tuple(record["component_names"], "Flow-state components"),
        _string_tuple(record["component_units"], "Flow-state units"),
        tuple(float(item) for item in scales),
        density_name=record["density_name"],
        velocity_names=_string_tuple(record["velocity_names"], "Velocity components"),
        species_names=_string_tuple(record["species_names"], "Species components"),
        total_energy_name=record["total_energy_name"],
        enthalpy_name=record["enthalpy_name"],
        component_axis=int(record["component_axis"]),
    )
    if schema.schema_id != _identifier(record["schema_id"], "Flow-state schema"):
        raise ArrayArchiveCorruptionError("Flow-state schema identity is inconsistent.")
    return schema


def _normalizer_from_record(value: Any, /) -> TrainOnlyNormalizer:
    fields = {
        "mean",
        "scale",
        "mean_dtype",
        "scale_dtype",
        "epsilon",
        "partition_id",
        "training_assignment_ids",
        "training_sample_ids",
        "feature_name",
        "schema_id",
        "provenance_id",
        "normalizer_id",
    }
    record = _mapping(value, fields, "Train-only normalizer")
    mean_dtype = np.dtype(_identifier(record["mean_dtype"], "Normalizer mean dtype"))
    scale_dtype = np.dtype(_identifier(record["scale_dtype"], "Normalizer scale dtype"))
    mean = np.asarray(record["mean"], dtype=mean_dtype)
    scale = np.asarray(record["scale"], dtype=scale_dtype)
    provenance = NormalizerProvenance(
        partition_id=_identifier(record["partition_id"], "Normalizer partition"),
        training_assignment_ids=_string_tuple(
            record["training_assignment_ids"], "Normalizer training assignments"
        ),
        training_sample_ids=_string_tuple(
            record["training_sample_ids"], "Normalizer training samples"
        ),
        feature_name=_identifier(record["feature_name"], "Normalizer feature"),
        schema_id=_identifier(record["schema_id"], "Normalizer schema"),
    )
    normalizer = TrainOnlyNormalizer(
        mean,
        scale,
        provenance,
        epsilon=float(record["epsilon"]),
    )
    if provenance.provenance_id != _identifier(
        record["provenance_id"], "Normalizer provenance"
    ) or normalizer.normalizer_id != _identifier(record["normalizer_id"], "Normalizer"):
        raise ArrayArchiveCorruptionError("Normalizer identity is inconsistent.")
    return normalizer


def _material_from_record(value: Any, /) -> IdealGasMaterial:
    fields = {
        "kind",
        "gamma",
        "gas_constant",
        "density_floor",
        "pressure_floor",
        "material_id",
    }
    record = _mapping(value, fields, "Ideal-gas material")
    if record["kind"] != "ideal-gas-material":
        raise ArrayArchiveCorruptionError("Energy-equilibrium material kind is invalid.")
    material = IdealGasMaterial(
        float(record["gamma"]),
        float(record["gas_constant"]),
        density_floor=float(record["density_floor"]),
        pressure_floor=float(record["pressure_floor"]),
    )
    if material.material_id != _identifier(record["material_id"], "Material"):
        raise ArrayArchiveCorruptionError("Material identity is inconsistent.")
    return material


def _quadrature_from_record(value: Any, /) -> CertifiedDiscreteVelocityQuadrature:
    fields = {
        "name",
        "velocities",
        "weights",
        "dtype",
        "reference_temperature",
        "certified_degree",
        "transport_kind",
        "tolerance",
        "quadrature_id",
    }
    record = _mapping(value, fields, "Discrete-velocity quadrature")
    dtype = np.dtype(_identifier(record["dtype"], "Quadrature dtype"))
    quadrature = CertifiedDiscreteVelocityQuadrature(
        _identifier(record["name"], "Quadrature name"),
        np.asarray(record["velocities"], dtype=dtype),
        np.asarray(record["weights"], dtype=dtype),
        reference_temperature=float(record["reference_temperature"]),
        certified_degree=int(record["certified_degree"]),
        transport_kind=record["transport_kind"],
        tolerance=float(record["tolerance"]),
    )
    if quadrature.quadrature_id != _identifier(record["quadrature_id"], "Quadrature"):
        raise ArrayArchiveCorruptionError("Quadrature identity is inconsistent.")
    return quadrature


def _equilibrium_plan_from_record(value: Any, /) -> PositiveEnergyEquilibriumPlan:
    fields = {
        "quadrature",
        "maximum_iterations",
        "residual_tolerance",
        "interior_tolerance",
        "damping",
        "plan_id",
    }
    record = _mapping(value, fields, "Energy-equilibrium plan")
    plan = PositiveEnergyEquilibriumPlan(
        _quadrature_from_record(record["quadrature"]),
        maximum_iterations=int(record["maximum_iterations"]),
        residual_tolerance=float(record["residual_tolerance"]),
        interior_tolerance=float(record["interior_tolerance"]),
        damping=float(record["damping"]),
    )
    if plan.plan_id != _identifier(record["plan_id"], "Energy-equilibrium plan"):
        raise ArrayArchiveCorruptionError(
            "Energy-equilibrium plan identity is inconsistent."
        )
    return plan


def _support_from_record(value: Any, /) -> EnergyEquilibriumSupportEnvelope:
    fields = {
        "rho_bounds",
        "u_x_bounds",
        "u_y_bounds",
        "temperature_bounds",
        "maximum_mach",
        "minimum_hull_margin",
        "minimum_particle_equilibrium_margin",
        "primitive_component_names",
        "schema_id",
        "material_id",
        "normalizer_id",
        "quadrature_id",
        "equilibrium_plan_id",
        "training_preparation_id",
        "support_id",
    }
    record = _mapping(value, fields, "Energy-equilibrium support")
    primitive_names = _string_tuple(
        record["primitive_component_names"], "Support primitive components"
    )
    if primitive_names != ("rho", "u_x", "u_y", "temperature"):
        raise ArrayArchiveCorruptionError("Support primitive component ABI is invalid.")
    support = EnergyEquilibriumSupportEnvelope(
        rho_bounds=_float_pair(record["rho_bounds"], "Density support"),
        u_x_bounds=_float_pair(record["u_x_bounds"], "X-velocity support"),
        u_y_bounds=_float_pair(record["u_y_bounds"], "Y-velocity support"),
        temperature_bounds=_float_pair(
            record["temperature_bounds"], "Temperature support"
        ),
        maximum_mach=float(record["maximum_mach"]),
        minimum_hull_margin=float(record["minimum_hull_margin"]),
        minimum_particle_equilibrium_margin=float(
            record["minimum_particle_equilibrium_margin"]
        ),
        schema_id=_identifier(record["schema_id"], "Support schema"),
        material_id=_identifier(record["material_id"], "Support material"),
        normalizer_id=_identifier(record["normalizer_id"], "Support normalizer"),
        quadrature_id=_identifier(record["quadrature_id"], "Support quadrature"),
        equilibrium_plan_id=_identifier(
            record["equilibrium_plan_id"], "Support equilibrium plan"
        ),
        training_preparation_id=_identifier(
            record["training_preparation_id"], "Support training preparation"
        ),
    )
    if support.support_id != _identifier(record["support_id"], "Support"):
        raise ArrayArchiveCorruptionError("Support identity is inconsistent.")
    return support


def read_learned_energy_equilibrium_artifact(
    path: str | Path,
    /,
) -> LearnedEnergyEquilibriumArtifact:
    """Restore and cross-check a complete frozen learned-equilibrium deployment."""

    source = Path(path)
    if source.suffix != ".phxml":
        raise ValueError("Learned energy-equilibrium artifacts require .phxml paths.")
    artifact = read_ml_artifact(source)
    feature = _mapping(
        artifact.manifest.feature_schema,
        _FEATURE_FIELDS,
        "Learned energy-equilibrium feature schema",
    )
    target = _mapping(
        artifact.manifest.target_schema,
        _TARGET_FIELDS,
        "Learned energy-equilibrium target schema",
    )
    provenance = _mapping(
        artifact.manifest.provenance,
        _PROVENANCE_FIELDS,
        "Learned energy-equilibrium provenance",
    )
    if (
        feature["kind"] != _FEATURE_KIND
        or target["kind"] != _TARGET_KIND
        or provenance["kind"] != _PROVENANCE_KIND
        or _string_tuple(
            target["output_component_names"], "Energy-dual output components"
        )
        != ("energy_dual_x", "energy_dual_y")
    ):
        raise ArrayArchiveCorruptionError(
            "Archive is not a learned energy-equilibrium deployment."
        )

    schema = _schema_from_record(feature["flow_schema"])
    input_components = _string_tuple(
        feature["input_component_names"], "Energy-equilibrium input components"
    )
    normalizer = _normalizer_from_record(feature["normalizer"])
    support = _support_from_record(feature["support"])
    material = _material_from_record(target["material"])
    equilibrium_plan = _equilibrium_plan_from_record(target["equilibrium_plan"])
    semantic_id = _identifier(provenance["semantic_id"], "Semantic")
    training_preparation_id = _identifier(
        provenance["training_preparation_id"], "Training preparation"
    )
    parent_artifact_id = _optional_identifier(
        provenance["parent_artifact_id"], "Parent artifact"
    )
    if target["population_convention"] != "weight_absorbed_total_energy_sum":
        raise ArrayArchiveCorruptionError(
            "Energy-equilibrium population convention is inconsistent."
        )

    identity_checks = (
        (schema.schema_id, provenance["schema_id"], "schema"),
        (material.material_id, provenance["material_id"], "material"),
        (
            equilibrium_plan.quadrature.quadrature_id,
            provenance["quadrature_id"],
            "quadrature",
        ),
        (
            equilibrium_plan.plan_id,
            provenance["equilibrium_plan_id"],
            "equilibrium plan",
        ),
        (normalizer.normalizer_id, provenance["normalizer_id"], "normalizer"),
        (
            normalizer.provenance.provenance_id,
            provenance["normalizer_provenance_id"],
            "normalizer provenance",
        ),
        (support.support_id, provenance["support_id"], "support"),
    )
    if any(
        actual != _identifier(declared, owner)
        for actual, declared, owner in identity_checks
    ):
        raise ArrayArchiveCorruptionError(
            "Learned energy-equilibrium identities are inconsistent."
        )
    if normalizer.provenance.schema_id != schema.schema_id:
        raise ArrayArchiveCorruptionError(
            "Normalizer and flow-state schema identities do not match."
        )
    if not isinstance(artifact.model, AbstractArrayModel) or isinstance(
        artifact.model, FrozenModel
    ):
        raise ArrayArchiveCorruptionError(
            "Learned energy-equilibrium model payload is invalid."
        )

    plan = LearnedEnergyEquilibriumBindingPlan(
        equilibrium_plan,
        schema,
        material,
        normalizer,
        support,
        input_component_names=input_components,
        semantic_id=semantic_id,
        training_preparation_id=training_preparation_id,
        parent_artifact_id=parent_artifact_id,
    )
    if plan.plan_id != _identifier(provenance["binding_plan_id"], "Binding plan"):
        raise ArrayArchiveCorruptionError("Binding-plan identity is inconsistent.")
    numeric_revision = energy_equilibrium_numeric_revision(semantic_id, artifact.model)
    if numeric_revision.revision_id != _identifier(
        provenance["numeric_revision_id"], "Numeric revision"
    ):
        raise ArrayArchiveCorruptionError("Numeric revision is inconsistent.")
    binding = plan.prepare(artifact.model, numeric_revision)
    if binding.prepared_id != _identifier(
        provenance["prepared_binding_id"], "Prepared binding"
    ):
        raise ArrayArchiveCorruptionError("Prepared-binding identity is inconsistent.")
    return LearnedEnergyEquilibriumArtifact(
        binding, artifact.manifest, binding.prepared_id
    )


__all__ = [
    "LearnedEnergyEquilibriumArtifact",
    "read_learned_energy_equilibrium_artifact",
    "write_learned_energy_equilibrium_artifact",
]
