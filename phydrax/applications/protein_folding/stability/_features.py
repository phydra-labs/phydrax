#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Transparent, source-bound residue-environment mutation features."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....qualification import ReferenceArtifactManifest, ScientificCampaign
from ..interchange._megascale import (
    parse_mutation_code,
    ProteinStabilityCohort,
    ProteinStabilityMeasurement,
)


_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
_AMINO_ACID_SET = frozenset(_AMINO_ACIDS)
_SECONDARY_CLASSES = ("H", "E", "L")
_CONTACT_CLASSES = ("aliphatic", "aromatic", "polar", "positive", "negative")
_RIGHTS_KEYS = frozenset(("commercial_use", "redistribution", "training_use", "export"))


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _admitted_manifests(
    values: Sequence[ReferenceArtifactManifest],
    requested_use: Mapping[str, bool],
    name: str,
    /,
) -> tuple[tuple[ReferenceArtifactManifest, ...], tuple[str, ...]]:
    if not isinstance(values, Sequence) or isinstance(values, str) or not values:
        raise TypeError(f"{name} must contain admitted ReferenceArtifactManifest values.")
    manifests = tuple(values)
    if any(not isinstance(value, ReferenceArtifactManifest) for value in manifests):
        raise TypeError(f"{name} must contain admitted ReferenceArtifactManifest values.")
    if not isinstance(requested_use, Mapping) or set(requested_use) != _RIGHTS_KEYS:
        raise ValueError("Declare all four feature requested-use rights explicitly.")
    for manifest in manifests:
        manifest.require_rights(**dict(requested_use))
    by_id = {manifest.manifest_id: manifest for manifest in manifests}
    if len(by_id) != len(manifests):
        raise ValueError(f"{name} must contain unique admitted manifests.")
    ordered = tuple(by_id[key] for key in sorted(by_id))
    return ordered, tuple(manifest.manifest_id for manifest in ordered)


def _finite(value: float, name: str, /) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _role_case_ids(campaign: ScientificCampaign, role_name: str, /) -> tuple[str, ...]:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    for role in campaign.roles:
        if role.name == role_name:
            return role.case_ids
    return ()


class AminoAcidScalarDefinition(StrictModule, NonTrainableState):
    """Caller-supplied volume, charge, and hydropathy definitions for all residues."""

    volume: Array
    charge: Array
    hydropathy: Array
    residue_order: tuple[str, ...] = eqx.field(static=True)
    source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    definition_id: str = eqx.field(static=True)

    def __init__(
        self,
        volume: Mapping[str, float],
        charge: Mapping[str, float],
        hydropathy: Mapping[str, float],
        /,
        *,
        source_manifests: Sequence[ReferenceArtifactManifest],
        requested_use: Mapping[str, bool],
    ):
        for values, name in (
            (volume, "volume"),
            (charge, "charge"),
            (hydropathy, "hydropathy"),
        ):
            if not isinstance(values, Mapping) or set(values) != _AMINO_ACID_SET:
                raise ValueError(
                    f"{name} must define every canonical amino acid exactly."
                )
        manifests, source_ids = _admitted_manifests(
            source_manifests, requested_use, "scalar source manifests"
        )
        volume_array = np.asarray([volume[key] for key in _AMINO_ACIDS], dtype=float)
        charge_array = np.asarray([charge[key] for key in _AMINO_ACIDS], dtype=float)
        hydropathy_array = np.asarray(
            [hydropathy[key] for key in _AMINO_ACIDS], dtype=float
        )
        if not all(
            np.all(np.isfinite(array))
            for array in (volume_array, charge_array, hydropathy_array)
        ):
            raise ValueError("Amino-acid scalar definitions must be finite.")
        self.volume = jnp.asarray(volume_array)
        self.charge = jnp.asarray(charge_array)
        self.hydropathy = jnp.asarray(hydropathy_array)
        self.residue_order = _AMINO_ACIDS
        self.source_manifests = manifests
        self.source_manifest_ids = tuple(sorted(source_ids))
        self.definition_id = canonical_fingerprint(
            {
                "kind": "amino-acid-scalar-definition",
                "residue_order": list(_AMINO_ACIDS),
                "volume": [float(value).hex() for value in volume_array],
                "charge": [float(value).hex() for value in charge_array],
                "hydropathy": [float(value).hex() for value in hydropathy_array],
                "source_manifest_ids": list(self.source_manifest_ids),
            }
        )

    def differences(self, wild_type: str, mutant: str, /) -> tuple[float, float, float]:
        if wild_type not in _AMINO_ACID_SET or mutant not in _AMINO_ACID_SET:
            raise ValueError("Residue identities must be canonical amino acids.")
        wild_index = self.residue_order.index(wild_type)
        mutant_index = self.residue_order.index(mutant)
        return (
            float(self.volume[mutant_index] - self.volume[wild_index]),
            float(self.charge[mutant_index] - self.charge[wild_index]),
            float(self.hydropathy[mutant_index] - self.hydropathy[wild_index]),
        )


class ProteinResidueEnvironment(StrictModule, NonTrainableState):
    """One admitted structure/hypothesis mapping at one construct residue."""

    residue_position: int = eqx.field(static=True)
    secondary_structure: str = eqx.field(static=True)
    relative_solvent_exposure: float = eqx.field(static=True)
    neighborhood_density: float = eqx.field(static=True)
    contact_counts: tuple[tuple[str, float], ...] = eqx.field(static=True)
    phi_radians: float = eqx.field(static=True)
    psi_radians: float = eqx.field(static=True)
    hypothesis_id: str = eqx.field(static=True)
    residue_mapping_id: str = eqx.field(static=True)
    source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    source_manifest_ids: tuple[str, ...] = eqx.field(static=True)
    environment_id: str = eqx.field(static=True)

    def __init__(
        self,
        residue_position: int,
        secondary_structure: str,
        relative_solvent_exposure: float,
        neighborhood_density: float,
        contact_counts: Mapping[str, float],
        phi_radians: float,
        psi_radians: float,
        /,
        *,
        hypothesis_id: str,
        residue_mapping_id: str,
        source_manifests: Sequence[ReferenceArtifactManifest],
        requested_use: Mapping[str, bool],
    ):
        if isinstance(residue_position, bool) or not isinstance(residue_position, int):
            raise TypeError("residue_position must be a one-based integer.")
        if residue_position < 1:
            raise ValueError("residue_position must be positive.")
        if secondary_structure not in _SECONDARY_CLASSES:
            raise ValueError("secondary_structure must be H, E, or L.")
        exposure = _finite(relative_solvent_exposure, "relative solvent exposure")
        density = _finite(neighborhood_density, "neighborhood density")
        if not 0.0 <= exposure <= 1.0 or density < 0.0:
            raise ValueError(
                "Relative exposure must be in [0,1] and density non-negative."
            )
        if not isinstance(contact_counts, Mapping) or set(contact_counts) != set(
            _CONTACT_CLASSES
        ):
            raise ValueError("contact_counts must cover the five coarse classes exactly.")
        contacts = tuple(
            (name, _finite(contact_counts[name], f"{name} contact count"))
            for name in _CONTACT_CLASSES
        )
        if any(value < 0.0 for _, value in contacts):
            raise ValueError("Contact counts must be non-negative.")
        phi = _finite(phi_radians, "phi_radians")
        psi = _finite(psi_radians, "psi_radians")
        if not -math.pi <= phi <= math.pi or not -math.pi <= psi <= math.pi:
            raise ValueError("Backbone torsions must use radians in [-pi, pi].")
        manifests, source_ids = _admitted_manifests(
            source_manifests, requested_use, "environment source manifests"
        )
        self.residue_position = residue_position
        self.secondary_structure = secondary_structure
        self.relative_solvent_exposure = exposure
        self.neighborhood_density = density
        self.contact_counts = contacts
        self.phi_radians = phi
        self.psi_radians = psi
        self.hypothesis_id = _identifier(hypothesis_id, "hypothesis_id")
        self.residue_mapping_id = _identifier(residue_mapping_id, "residue_mapping_id")
        self.source_manifests = manifests
        self.source_manifest_ids = tuple(sorted(source_ids))
        self.environment_id = canonical_fingerprint(
            {
                "kind": "protein-residue-environment",
                "position": residue_position,
                "secondary_structure": secondary_structure,
                "relative_solvent_exposure": exposure.hex(),
                "neighborhood_density": density.hex(),
                "contact_counts": [(name, value.hex()) for name, value in contacts],
                "phi_radians": phi.hex(),
                "psi_radians": psi.hex(),
                "hypothesis_id": self.hypothesis_id,
                "residue_mapping_id": self.residue_mapping_id,
                "source_manifest_ids": list(self.source_manifest_ids),
            }
        )


class ProteinMutationFeatures(StrictModule, NonTrainableState):
    """Frozen named feature layout for one exact WT construct and substitution."""

    values: Array
    measurement_id: str = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    domain_family_id: str = eqx.field(static=True)
    background_id: str = eqx.field(static=True)
    wt_sequence: str = eqx.field(static=True)
    mutation_code: str = eqx.field(static=True)
    source_hypothesis_id: str = eqx.field(static=True)
    source_measurement_record_id: str = eqx.field(static=True)
    assay_channel: str = eqx.field(static=True)
    condition_id: str = eqx.field(static=True)
    observable: str = eqx.field(static=True)
    sign_convention: str = eqx.field(static=True)
    shared_wt_id: str = eqx.field(static=True)
    residue_mapping_id: str = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)
    feature_definition_id: str = eqx.field(static=True)
    source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    preprocessing_source_ids: tuple[str, ...] = eqx.field(static=True)
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        measurement_id: str,
        domain_id: str,
        domain_family_id: str,
        background_id: str,
        wt_sequence: str,
        mutation_code: str,
        source_hypothesis_id: str,
        source_measurement_record_id: str,
        assay_channel: str,
        condition_id: str,
        observable: str,
        sign_convention: str,
        shared_wt_id: str,
        residue_mapping_id: str,
        feature_names: Sequence[str],
        feature_definition_id: str,
        source_measurement_manifest_id: str,
        source_manifests: Sequence[ReferenceArtifactManifest],
        requested_use: Mapping[str, bool],
    ):
        array = np.asarray(values, dtype=float)
        names = tuple(_identifier(value, "feature name") for value in feature_names)
        if array.ndim != 1 or array.shape != (len(names),) or not names:
            raise ValueError("Feature values must align with one non-empty named layout.")
        if not np.all(np.isfinite(array)) or len(set(names)) != len(names):
            raise ValueError("Feature values must be finite and feature names unique.")
        manifests, feature_source_ids = _admitted_manifests(
            source_manifests, requested_use, "feature source manifests"
        )
        measurement_source_id = _identifier(
            source_measurement_manifest_id, "source measurement manifest ID"
        )
        source_ids = tuple(sorted({measurement_source_id, *feature_source_ids}))
        for value, name in (
            (measurement_id, "measurement_id"),
            (domain_id, "domain_id"),
            (domain_family_id, "domain_family_id"),
            (background_id, "background_id"),
            (source_measurement_record_id, "source_measurement_record_id"),
            (assay_channel, "assay_channel"),
            (condition_id, "condition_id"),
            (observable, "observable"),
            (sign_convention, "sign_convention"),
            (shared_wt_id, "shared_wt_id"),
            (source_hypothesis_id, "source_hypothesis_id"),
            (residue_mapping_id, "residue_mapping_id"),
            (feature_definition_id, "feature_definition_id"),
        ):
            _identifier(value, name)
        if (
            not isinstance(wt_sequence, str)
            or not wt_sequence
            or set(wt_sequence) - _AMINO_ACID_SET
        ):
            raise ValueError("wt_sequence must contain uppercase canonical amino acids.")
        substitutions = parse_mutation_code(mutation_code)
        if len(substitutions) != 1:
            raise ValueError("ProteinMutationFeatures require one point substitution.")
        wild_type, position, _ = substitutions[0]
        if position > len(wt_sequence) or wt_sequence[position - 1] != wild_type:
            raise ValueError("Mutation code does not match the exact WT sequence.")
        self.values = jnp.asarray(array)
        self.measurement_id = measurement_id
        self.domain_id = domain_id
        self.domain_family_id = domain_family_id
        self.background_id = background_id
        self.wt_sequence = wt_sequence
        self.mutation_code = mutation_code
        self.source_measurement_record_id = source_measurement_record_id
        self.assay_channel = assay_channel
        self.condition_id = condition_id
        self.observable = observable
        self.sign_convention = sign_convention
        self.shared_wt_id = shared_wt_id
        self.source_hypothesis_id = source_hypothesis_id
        self.residue_mapping_id = residue_mapping_id
        self.feature_names = names
        self.feature_definition_id = feature_definition_id
        self.source_manifests = manifests
        self.preprocessing_source_ids = tuple(sorted(source_ids))
        self.feature_id = canonical_fingerprint(
            {
                "kind": "protein-mutation-features",
                "measurement_id": measurement_id,
                "domain_id": domain_id,
                "family_id": domain_family_id,
                "background_id": background_id,
                "wt_sequence": wt_sequence,
                "mutation_code": mutation_code,
                "source_measurement_record_id": source_measurement_record_id,
                "assay_channel": assay_channel,
                "condition_id": condition_id,
                "observable": observable,
                "sign_convention": sign_convention,
                "shared_wt_id": shared_wt_id,
                "source_hypothesis_id": source_hypothesis_id,
                "residue_mapping_id": residue_mapping_id,
                "feature_names": list(names),
                "values": [float(value).hex() for value in array],
                "feature_definition_id": feature_definition_id,
                "preprocessing_source_ids": list(self.preprocessing_source_ids),
            }
        )


class ProteinFeatureTransform(StrictModule, NonTrainableState):
    """Calibration-only feature normalization with explicit constant columns."""

    mean: Array
    scale: Array
    feature_names: tuple[str, ...] = eqx.field(static=True)
    feature_definition_id: str = eqx.field(static=True)
    fit_case_ids: tuple[str, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)
    cohort_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    constant_feature_names: tuple[str, ...] = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        scale: ArrayLike,
        /,
        *,
        feature_names: Sequence[str],
        feature_definition_id: str,
        fit_case_ids: Sequence[str],
        source_ids: Sequence[str],
        campaign_id: str,
        cohort_id: str,
        source_id: str,
        constant_feature_names: Sequence[str] = (),
    ):
        mean_array = np.asarray(mean, dtype=float)
        scale_array = np.asarray(scale, dtype=float)
        names = tuple(feature_names)
        if (
            mean_array.shape != (len(names),)
            or scale_array.shape != mean_array.shape
            or not names
            or not np.all(np.isfinite(mean_array))
            or not np.all(np.isfinite(scale_array))
            or np.any(scale_array <= 0.0)
        ):
            raise ValueError(
                "Feature transform arrays must be finite, positive, and aligned."
            )
        fit_ids = tuple(
            sorted(_identifier(value, "fit case ID") for value in fit_case_ids)
        )
        sources = tuple(
            sorted(_identifier(value, "transform source ID") for value in source_ids)
        )
        constants = tuple(sorted(constant_feature_names))
        if not fit_ids or not sources or not set(constants).issubset(names):
            raise ValueError(
                "Transform requires fit/source IDs and valid constant features."
            )
        self.mean = jnp.asarray(mean_array)
        self.scale = jnp.asarray(scale_array)
        self.feature_names = names
        self.feature_definition_id = _identifier(
            feature_definition_id, "feature_definition_id"
        )
        self.fit_case_ids = fit_ids
        self.source_ids = sources
        self.campaign_id = _identifier(campaign_id, "campaign_id")
        self.cohort_id = _identifier(cohort_id, "cohort_id")
        self.source_id = _identifier(source_id, "source_id")
        self.constant_feature_names = constants
        self.transform_id = canonical_fingerprint(
            {
                "kind": "protein-feature-transform",
                "mean": [float(value).hex() for value in mean_array],
                "scale": [float(value).hex() for value in scale_array],
                "feature_names": list(names),
                "feature_definition_id": self.feature_definition_id,
                "fit_case_ids": list(fit_ids),
                "source_ids": list(sources),
                "campaign_id": self.campaign_id,
                "cohort_id": self.cohort_id,
                "source_id": self.source_id,
                "constant_feature_names": list(constants),
            }
        )

    def transform(self, features: ProteinMutationFeatures, /) -> Array:
        if not isinstance(features, ProteinMutationFeatures):
            raise TypeError("features must be ProteinMutationFeatures.")
        if (
            features.feature_names != self.feature_names
            or features.feature_definition_id != self.feature_definition_id
        ):
            raise ValueError("Feature layout/definition does not match the transform.")
        return (features.values - self.mean) / self.scale


def protein_mutation_features(
    measurement: ProteinStabilityMeasurement,
    environment: ProteinResidueEnvironment,
    scalars: AminoAcidScalarDefinition,
    /,
    *,
    feature_definition_id: str,
    requested_use: Mapping[str, bool],
) -> ProteinMutationFeatures:
    """Build the pinned V1 feature layout for one exact single substitution."""
    if not isinstance(measurement, ProteinStabilityMeasurement):
        raise TypeError("measurement must be ProteinStabilityMeasurement.")
    if not isinstance(environment, ProteinResidueEnvironment):
        raise TypeError("environment must be ProteinResidueEnvironment.")
    if not isinstance(scalars, AminoAcidScalarDefinition):
        raise TypeError("scalars must be AminoAcidScalarDefinition.")
    substitutions = parse_mutation_code(measurement.mutation_code)
    if len(substitutions) != 1:
        raise ValueError("V1 mutation features require exactly one point substitution.")
    wild_type, position, mutant = substitutions[0]
    if environment.residue_position != position:
        raise ValueError("Environment position does not match the mutation position.")
    length = len(measurement.sequence)
    if measurement.sequence[position - 1] != wild_type:
        raise ValueError("Mutation WT identity does not match the admitted construct.")
    wt_one_hot = tuple(float(wild_type == residue) for residue in _AMINO_ACIDS)
    mutant_one_hot = tuple(float(mutant == residue) for residue in _AMINO_ACIDS)
    secondary = tuple(
        float(environment.secondary_structure == value) for value in _SECONDARY_CLASSES
    )
    contacts = tuple(dict(environment.contact_counts)[name] for name in _CONTACT_CLASSES)
    differences = scalars.differences(wild_type, mutant)
    position_fraction = (position - 1) / max(length - 1, 1)
    terminal_distance_fraction = min(position - 1, length - position) / max(length - 1, 1)
    names = (
        *(f"wt:{residue}" for residue in _AMINO_ACIDS),
        *(f"mutant:{residue}" for residue in _AMINO_ACIDS),
        "sequence-position-fraction",
        "terminal-distance-fraction",
        *(f"secondary:{value}" for value in _SECONDARY_CLASSES),
        "relative-solvent-exposure",
        "neighborhood-density",
        *(f"contacts:{value}" for value in _CONTACT_CLASSES),
        "sin-phi",
        "cos-phi",
        "sin-psi",
        "cos-psi",
        "delta-volume",
        "delta-charge",
        "delta-hydropathy",
    )
    values = (
        *wt_one_hot,
        *mutant_one_hot,
        position_fraction,
        terminal_distance_fraction,
        *secondary,
        environment.relative_solvent_exposure,
        environment.neighborhood_density,
        *contacts,
        math.sin(environment.phi_radians),
        math.cos(environment.phi_radians),
        math.sin(environment.psi_radians),
        math.cos(environment.psi_radians),
        *differences,
    )
    manifests_by_id = {
        manifest.manifest_id: manifest
        for manifest in (*environment.source_manifests, *scalars.source_manifests)
    }
    source_manifests = tuple(manifests_by_id[key] for key in sorted(manifests_by_id))
    return ProteinMutationFeatures(
        values,
        measurement_id=measurement.measurement_id,
        domain_id=measurement.domain_id,
        domain_family_id=measurement.domain_family_id,
        background_id=measurement.background_id,
        wt_sequence=measurement.sequence,
        mutation_code=measurement.mutation_code,
        source_measurement_record_id=measurement.record_id,
        assay_channel=measurement.assay_channel,
        condition_id=measurement.condition_id,
        observable=measurement.observable,
        sign_convention=measurement.sign_convention,
        shared_wt_id=measurement.shared_wt_id,
        source_hypothesis_id=environment.hypothesis_id,
        residue_mapping_id=environment.residue_mapping_id,
        feature_names=names,
        feature_definition_id=_identifier(feature_definition_id, "feature_definition_id"),
        source_measurement_manifest_id=measurement.source_manifest_id,
        source_manifests=source_manifests,
        requested_use=requested_use,
    )


def fit_protein_feature_transform(
    features: Sequence[ProteinMutationFeatures],
    cohort: ProteinStabilityCohort,
    /,
) -> ProteinFeatureTransform:
    """Fit exact calibration-only normalization with rechecked training rights."""
    if not isinstance(cohort, ProteinStabilityCohort):
        raise TypeError("cohort must be a ProteinStabilityCohort.")
    values = tuple(features)
    if not values or any(
        not isinstance(item, ProteinMutationFeatures) for item in values
    ):
        raise TypeError("features must contain ProteinMutationFeatures.")
    by_id = {item.measurement_id: item for item in values}
    if len(by_id) != len(values):
        raise ValueError("Feature measurement IDs must be unique.")
    measurement_by_id = {item.measurement_id: item for item in cohort.measurements}
    unknown = set(by_id) - set(measurement_by_id)
    if unknown:
        raise ValueError("Every transform feature must identify a cohort measurement.")
    calibration_ids = _role_case_ids(cohort.campaign, "calibration")
    eligible_ids = tuple(
        case_id
        for case_id in calibration_ids
        if measurement_by_id[case_id].mutation_order == 1
    )
    if not eligible_ids:
        raise ValueError(
            "At least one calibration single mutant requires exact features."
        )
    missing = set(eligible_ids) - set(by_id)
    if missing:
        raise ValueError(
            "Features must exactly cover every eligible calibration single mutant."
        )
    selected = tuple(by_id[case_id] for case_id in eligible_ids)
    for feature in selected:
        measurement = measurement_by_id[feature.measurement_id]
        if (
            feature.source_measurement_record_id != measurement.record_id
            or feature.domain_id != measurement.domain_id
            or feature.domain_family_id != measurement.domain_family_id
            or feature.background_id != measurement.background_id
            or feature.wt_sequence != measurement.sequence
            or feature.mutation_code != measurement.mutation_code
            or feature.assay_channel != measurement.assay_channel
            or feature.condition_id != measurement.condition_id
            or feature.observable != measurement.observable
            or feature.sign_convention != measurement.sign_convention
            or feature.shared_wt_id != measurement.shared_wt_id
        ):
            raise ValueError(
                "Calibration feature does not bind its exact cohort measurement."
            )
        if measurement.source_manifest is None:
            raise ValueError(
                "Calibration preprocessing lacks its admitted measurement manifest."
            )
        measurement.source_manifest.require_rights(training_use=True)
        for manifest in feature.source_manifests:
            manifest.require_rights(training_use=True)
        expected_sources = {
            measurement.source_manifest.manifest_id,
            *(manifest.manifest_id for manifest in feature.source_manifests),
        }
        if set(feature.preprocessing_source_ids) != expected_sources:
            raise ValueError("Calibration feature preprocessing sources are not exact.")
    names = selected[0].feature_names
    definition = selected[0].feature_definition_id
    if any(
        item.feature_names != names or item.feature_definition_id != definition
        for item in selected
    ):
        raise ValueError("Calibration features must share one exact named definition.")
    matrix = np.stack([np.asarray(item.values) for item in selected])
    mean = np.mean(matrix, axis=0)
    empirical_scale = np.std(matrix, axis=0, ddof=0)
    constant = empirical_scale == 0.0
    scale = np.where(constant, 1.0, empirical_scale)
    source_ids = tuple(
        sorted({value for item in selected for value in item.preprocessing_source_ids})
    )
    return ProteinFeatureTransform(
        mean,
        scale,
        feature_names=names,
        feature_definition_id=definition,
        fit_case_ids=eligible_ids,
        source_ids=source_ids,
        campaign_id=cohort.campaign.campaign_id,
        cohort_id=cohort.cohort_id,
        source_id=cohort.source_id,
        constant_feature_names=tuple(
            name for name, is_constant in zip(names, constant, strict=True) if is_constant
        ),
    )


class DoubleMutationFeatures(StrictModule, NonTrainableState):
    """Symmetric low-order pair features for one explicit double-mutant case."""

    values: Array
    case_id: str = eqx.field(static=True)
    first_feature_id: str = eqx.field(static=True)
    second_feature_id: str = eqx.field(static=True)
    pair_unit_id: str = eqx.field(static=True)
    feature_names: tuple[str, ...] = eqx.field(static=True)
    feature_definition_id: str = eqx.field(static=True)
    source_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)

    def __init__(
        self,
        first: ProteinMutationFeatures,
        second: ProteinMutationFeatures,
        /,
        *,
        case_id: str,
        pair_unit_id: str,
        pair_distance: float,
        pair_context_id: str,
    ):
        if not isinstance(first, ProteinMutationFeatures) or not isinstance(
            second, ProteinMutationFeatures
        ):
            raise TypeError("Double features require two ProteinMutationFeatures.")
        if (
            first.feature_names != second.feature_names
            or first.feature_definition_id != second.feature_definition_id
            or first.background_id != second.background_id
            or first.domain_family_id != second.domain_family_id
            or first.wt_sequence != second.wt_sequence
            or first.source_hypothesis_id != second.source_hypothesis_id
            or first.assay_channel != second.assay_channel
            or first.condition_id != second.condition_id
            or first.observable != second.observable
            or first.sign_convention != second.sign_convention
            or first.shared_wt_id != second.shared_wt_id
        ):
            raise ValueError(
                "Double-feature components must share one exact background, "
                "structure, condition, observation law, and feature layout."
            )
        first_position = parse_mutation_code(first.mutation_code)[0][1]
        second_position = parse_mutation_code(second.mutation_code)[0][1]
        if first_position == second_position:
            raise ValueError("Double-feature components must mutate distinct positions.")
        distance = _finite(pair_distance, "pair_distance")
        if distance <= 0.0:
            raise ValueError("pair_distance must be positive.")
        lower, upper = sorted((first, second), key=lambda value: value.mutation_code)
        sum_values = lower.values + upper.values
        difference_values = jnp.abs(lower.values - upper.values)
        names = (
            *(f"sum:{name}" for name in lower.feature_names),
            *(f"absolute-difference:{name}" for name in lower.feature_names),
            "pair-distance",
        )
        self.values = jnp.concatenate(
            (sum_values, difference_values, jnp.asarray([distance]))
        )
        self.case_id = _identifier(case_id, "double-mutant case ID")
        self.first_feature_id = lower.feature_id
        self.second_feature_id = upper.feature_id
        self.pair_unit_id = _identifier(pair_unit_id, "pair_unit_id")
        self.feature_names = names
        manifests_by_id = {
            manifest.manifest_id: manifest
            for feature in (lower, upper)
            for manifest in feature.source_manifests
        }
        self.source_manifests = tuple(
            manifests_by_id[key] for key in sorted(manifests_by_id)
        )
        self.feature_definition_id = canonical_fingerprint(
            {
                "kind": "protein-double-mutation-feature-definition",
                "single_definition_id": lower.feature_definition_id,
                "pair_context_id": _identifier(pair_context_id, "pair_context_id"),
                "layout": list(names),
            }
        )


__all__ = [
    "AminoAcidScalarDefinition",
    "DoubleMutationFeatures",
    "ProteinFeatureTransform",
    "ProteinMutationFeatures",
    "ProteinResidueEnvironment",
    "fit_protein_feature_transform",
    "protein_mutation_features",
]
