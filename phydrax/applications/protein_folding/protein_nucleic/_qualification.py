# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exact protein-nucleic mechanics and later affinity claim inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....qualification import (
    QualificationEvidence,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificClaimProfile,
)
from ....units import conversion_factor, UnitDefinition
from ...nucleic_acid_biophysics._construct import NucleicAcidConstruct
from .._construct import ProteinConstruct


_MANDATORY_CLAIM_STAGES = frozenset(
    (
        "source-admission",
        "measurement-calibration",
        "parameter-identifiability",
        "predictive-calibration",
        "locked-prediction",
    )
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


def _role_case_ids(campaign: ScientificCampaign, role_name: str, /) -> tuple[str, ...]:
    return next(role.case_ids for role in campaign.roles if role.name == role_name)


def _validate_locked_mechanics_campaign(
    campaign: ScientificCampaign,
    observations: ProteinNucleicMechanicalObservations,
    /,
) -> None:
    if not isinstance(campaign, ScientificCampaign):
        raise TypeError("campaign must be a ScientificCampaign.")
    if set(_role_case_ids(campaign, "locked_evaluation")) != set(observations.case_ids):
        raise ValueError(
            "Campaign locked-evaluation cases must exactly match mechanics rows."
        )
    cases = {case.case_id: case for case in campaign.cases}
    for case_id, independent_unit_id, preparation_id, condition_id in zip(
        observations.case_ids,
        observations.independent_unit_ids,
        observations.preparation_ids,
        observations.condition_ids,
        strict=True,
    ):
        case = cases[case_id]
        if (
            case.independent_unit_id != independent_unit_id
            or case.construct_id != observations.complex_construct_id
            or case.preparation_id != preparation_id
            or case.condition_id != condition_id
            or observations.source.manifest_id not in case.source_manifest_ids
        ):
            raise ValueError(
                "Campaign case identity does not match protein-nucleic observations."
            )


@dataclass(frozen=True, slots=True, init=False)
class ProteinNucleicMechanicalObservations:
    """One exact complex and mechanical observable across independent preparations."""

    protein_construct_id: str
    nucleic_construct_id: str
    complex_construct_id: str
    case_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    preparation_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    observable_kind: Literal["complex-distance", "complex-force", "contact-probability"]
    values: Array
    standard_errors: Array
    unit: UnitDefinition
    source: ReferenceArtifactManifest
    observation_id: str

    def __init__(
        self,
        protein: ProteinConstruct,
        nucleic_acid: NucleicAcidConstruct,
        case_ids: tuple[str, ...],
        independent_unit_ids: tuple[str, ...],
        preparation_ids: tuple[str, ...],
        condition_ids: tuple[str, ...],
        /,
        *,
        observable_kind: Literal[
            "complex-distance", "complex-force", "contact-probability"
        ],
        values: ArrayLike,
        standard_errors: ArrayLike,
        unit: UnitDefinition,
        source: ReferenceArtifactManifest,
    ):
        if not isinstance(protein, ProteinConstruct) or not isinstance(
            nucleic_acid, NucleicAcidConstruct
        ):
            raise TypeError("Exact protein and nucleic-acid constructs are required.")
        cases = tuple(case_ids)
        independent = tuple(independent_unit_ids)
        preparations = tuple(preparation_ids)
        conditions = tuple(condition_ids)
        n = len(cases)
        if (
            not n
            or len(set(cases)) != n
            or any(len(items) != n for items in (independent, preparations, conditions))
            or any(
                not value or value != value.strip()
                for value in (*cases, *independent, *preparations, *conditions)
            )
        ):
            raise ValueError(
                "Complex observations need unique cases and aligned scientific grouping."
            )
        if observable_kind not in (
            "complex-distance",
            "complex-force",
            "contact-probability",
        ):
            raise ValueError("Unsupported protein-nucleic mechanical observable.")
        observed = np.asarray(values, dtype=float)
        errors = np.asarray(standard_errors, dtype=float)
        if (
            observed.shape != (n,)
            or errors.shape != (n,)
            or not np.all(np.isfinite(observed))
            or not np.all(np.isfinite(errors))
            or np.any(errors <= 0.0)
        ):
            raise ValueError(
                "Complex observations require finite rows and positive uncertainty."
            )
        if observable_kind == "contact-probability" and np.any(
            (observed < 0.0) | (observed > 1.0)
        ):
            raise ValueError("Contact probabilities must lie in [0, 1].")
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError("Complex observations require a ReferenceArtifactManifest.")
        source.require_rights()
        source.require_uncertainty()
        complex_construct_id = canonical_fingerprint(
            {
                "kind": "protein-nucleic-complex",
                "protein": protein.fingerprint(),
                "nucleic_acid": nucleic_acid.fingerprint(),
            }
        )
        object.__setattr__(self, "protein_construct_id", protein.fingerprint())
        object.__setattr__(self, "nucleic_construct_id", nucleic_acid.fingerprint())
        object.__setattr__(self, "complex_construct_id", complex_construct_id)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "independent_unit_ids", independent)
        object.__setattr__(self, "preparation_ids", preparations)
        object.__setattr__(self, "condition_ids", conditions)
        object.__setattr__(self, "observable_kind", observable_kind)
        object.__setattr__(self, "values", jnp.asarray(observed))
        object.__setattr__(self, "standard_errors", jnp.asarray(errors))
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "observation_id",
            canonical_fingerprint(
                {
                    "kind": "protein-nucleic-mechanical-observations",
                    "protein": protein.fingerprint(),
                    "nucleic_acid": nucleic_acid.fingerprint(),
                    "complex": complex_construct_id,
                    "cases": cases,
                    "independent_units": independent,
                    "preparations": preparations,
                    "conditions": conditions,
                    "observable_kind": observable_kind,
                    "values": array_tree_fingerprint(observed),
                    "standard_errors": array_tree_fingerprint(errors),
                    "unit": unit.unit_id,
                    "source": source.manifest_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class ProteinNucleicModelFit:
    """Content-bound fitted parameters, artifacts, execution, and campaign roles."""

    campaign: ScientificCampaign
    model_id: str
    calibration_case_ids: tuple[str, ...]
    model_selection_case_ids: tuple[str, ...]
    source_manifest_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    preparation_ids: tuple[str, ...]
    parameter_artifact_ids: tuple[str, ...]
    fitted_parameter_id: str
    prediction_code_manifest_id: str
    fit_execution_evidence_id: str
    fit_integrity_ids: tuple[str, ...]
    fit_id: str

    def __init__(
        self,
        model_id: str,
        campaign: ScientificCampaign,
        fitted_parameters: ArrayLike,
        parameter_artifacts: Sequence[ReferenceArtifactManifest],
        prediction_code: ReferenceArtifactManifest,
        fit_execution_evidence: QualificationEvidence,
        /,
    ):
        if not isinstance(campaign, ScientificCampaign):
            raise TypeError("campaign must be a ScientificCampaign.")
        model = _identifier(model_id, "model_id")
        calibration = _role_case_ids(campaign, "calibration")
        selection = _role_case_ids(campaign, "model_selection")
        cases = {case.case_id: case for case in campaign.cases}
        lineage = tuple(cases[case_id] for case_id in (*calibration, *selection))
        sources = tuple(
            sorted(
                {source_id for case in lineage for source_id in case.source_manifest_ids}
            )
        )
        units = tuple(sorted({case.independent_unit_id for case in lineage}))
        preparations = tuple(sorted({case.preparation_id for case in lineage}))
        parameters = np.asarray(fitted_parameters, dtype=float)
        if not parameters.size or not np.all(np.isfinite(parameters)):
            raise ValueError(
                "Fitted protein-nucleic parameters must be finite and nonempty."
            )
        parameter_id = canonical_fingerprint(
            {
                "kind": "protein-nucleic-fitted-parameters",
                "values": array_tree_fingerprint(parameters),
            }
        )
        artifacts = tuple(parameter_artifacts)
        if not artifacts or any(
            not isinstance(item, ReferenceArtifactManifest) for item in artifacts
        ):
            raise TypeError("parameter_artifacts must contain admitted manifests.")
        artifact_ids = tuple(sorted(item.manifest_id for item in artifacts))
        if not set(sources) <= set(artifact_ids):
            raise ValueError(
                "Parameter/source artifacts must cover every fit campaign source."
            )
        for artifact in artifacts:
            artifact.require_rights(training_use=True)
        if not isinstance(prediction_code, ReferenceArtifactManifest):
            raise TypeError("prediction_code must be a ReferenceArtifactManifest.")
        prediction_code.require_rights()
        if not isinstance(fit_execution_evidence, QualificationEvidence):
            raise TypeError("fit_execution_evidence must be QualificationEvidence.")
        required_subjects = {
            campaign.campaign_id,
            model,
            parameter_id,
            prediction_code.manifest_id,
            *sources,
        }
        if (
            fit_execution_evidence.outcome != "passed"
            or "fit-execution" not in fit_execution_evidence.criteria_ids
            or not required_subjects <= set(fit_execution_evidence.subject_ids)
        ):
            raise ValueError(
                "Fit execution evidence must bind campaign, model, parameters, code, and sources."
            )
        integrity = (
            fit_execution_evidence.evidence_id,
            parameter_id,
            prediction_code.manifest_id,
            *artifact_ids,
        )
        for name, value in (
            ("campaign", campaign),
            ("model_id", model),
            ("calibration_case_ids", calibration),
            ("model_selection_case_ids", selection),
            ("source_manifest_ids", sources),
            ("independent_unit_ids", units),
            ("preparation_ids", preparations),
            ("parameter_artifact_ids", artifact_ids),
            ("fitted_parameter_id", parameter_id),
            ("prediction_code_manifest_id", prediction_code.manifest_id),
            ("fit_execution_evidence_id", fit_execution_evidence.evidence_id),
            ("fit_integrity_ids", tuple(sorted(set(integrity)))),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "fit_id",
            canonical_fingerprint(
                {
                    "kind": "protein-nucleic-model-fit",
                    "campaign": campaign.campaign_id,
                    "model": model,
                    "calibration_cases": calibration,
                    "model_selection_cases": selection,
                    "sources": sources,
                    "parameter_artifacts": artifact_ids,
                    "fitted_parameters": parameter_id,
                    "prediction_code": prediction_code.manifest_id,
                    "fit_execution_evidence": fit_execution_evidence.evidence_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True, init=False)
class ProteinNucleicAffinityInputs:
    """Bound/unbound free energies with explicit covariance and source lineage."""

    case_ids: tuple[str, ...]
    preparation_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    standard_state_ids: tuple[str, ...]
    predicted_binding_free_energy: Array
    prediction_standard_error: Array | None
    component_covariance: Array | None
    components_conditionally_independent: bool
    observed_binding_free_energy: Array
    observation_standard_error: Array
    energy_unit: UnitDefinition
    bound_sampling_reference: ReferenceArtifactManifest | None
    unbound_sampling_reference: ReferenceArtifactManifest | None
    binding_measurement_reference: ReferenceArtifactManifest | None
    shared_sampling_lineage_ids: tuple[str, ...]
    fit_source_manifest_ids: tuple[str, ...]
    fit_independent_unit_ids: tuple[str, ...]
    fit_preparation_ids: tuple[str, ...]
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    fit_integrity_ids: tuple[str, ...]
    model_id: str
    fit_id: str
    prediction_id: str
    observation_id: str
    input_id: str

    def __init__(
        self,
        condition_ids: tuple[str, ...],
        independent_unit_ids: tuple[str, ...],
        standard_state_ids: tuple[str, ...],
        /,
        case_ids: tuple[str, ...],
        preparation_ids: tuple[str, ...],
        fit: ProteinNucleicModelFit,
        *,
        bound_free_energy: ArrayLike,
        unbound_protein_free_energy: ArrayLike,
        unbound_nucleic_free_energy: ArrayLike,
        standard_state_correction: ArrayLike,
        bound_standard_error: ArrayLike,
        unbound_protein_standard_error: ArrayLike,
        unbound_nucleic_standard_error: ArrayLike,
        standard_state_standard_error: ArrayLike | None,
        component_covariance: ArrayLike | None,
        components_conditionally_independent: bool,
        observed_binding_free_energy: ArrayLike,
        observation_standard_error: ArrayLike,
        energy_unit: UnitDefinition,
        bound_sampling_reference: ReferenceArtifactManifest | None,
        unbound_sampling_reference: ReferenceArtifactManifest | None,
        binding_measurement_reference: ReferenceArtifactManifest | None,
        shared_sampling_lineage_ids: tuple[str, ...] = (),
    ):
        conditions = tuple(condition_ids)
        independent = tuple(independent_unit_ids)
        standard_states = tuple(standard_state_ids)
        cases = tuple(case_ids)
        preparations = tuple(preparation_ids)
        if not isinstance(fit, ProteinNucleicModelFit):
            raise TypeError("fit must be a ProteinNucleicModelFit.")
        shared_lineage = tuple(
            sorted(
                _identifier(value, "shared_sampling_lineage_id")
                for value in shared_sampling_lineage_ids
            )
        )
        if len(set(shared_lineage)) != len(shared_lineage):
            raise ValueError("Shared sampling lineage IDs must be unique.")
        n = len(conditions)
        if (
            not n
            or len(set(conditions)) != n
            or len(set(cases)) != n
            or any(
                len(items) != n
                for items in (independent, standard_states, cases, preparations)
            )
            or any(
                not value or value != value.strip()
                for value in (
                    *conditions,
                    *independent,
                    *standard_states,
                    *cases,
                    *preparations,
                )
            )
        ):
            raise ValueError(
                "Affinity rows require exact condition, independent-unit, and standard-state IDs."
            )
        energy_components = tuple(
            np.asarray(value, dtype=float)
            for value in (
                bound_free_energy,
                unbound_protein_free_energy,
                unbound_nucleic_free_energy,
                standard_state_correction,
            )
        )
        component_errors = tuple(
            np.asarray(value, dtype=float)
            for value in (
                bound_standard_error,
                unbound_protein_standard_error,
                unbound_nucleic_standard_error,
            )
        )
        observed = np.asarray(observed_binding_free_energy, dtype=float)
        observation_error = np.asarray(observation_standard_error, dtype=float)
        if any(
            array.shape != (n,) or not np.all(np.isfinite(array))
            for array in (
                *energy_components,
                *component_errors,
                observed,
                observation_error,
            )
        ):
            raise ValueError(
                "Every affinity quantity must be a finite vector over conditions."
            )
        if any(np.any(array <= 0.0) for array in (*component_errors, observation_error)):
            raise ValueError(
                "Sampling and binding-measurement standard errors must be positive."
            )
        if not isinstance(components_conditionally_independent, bool):
            raise TypeError("components_conditionally_independent must be a bool.")
        correction_error = (
            None
            if standard_state_standard_error is None
            else np.asarray(standard_state_standard_error, dtype=float)
        )
        if correction_error is not None and (
            correction_error.shape != (n,)
            or not np.all(np.isfinite(correction_error))
            or np.any(correction_error <= 0.0)
        ):
            raise ValueError(
                "Standard-state correction uncertainty must be positive and aligned."
            )
        covariance = (
            None
            if component_covariance is None
            else np.asarray(component_covariance, dtype=float)
        )
        if covariance is not None:
            if components_conditionally_independent:
                raise ValueError(
                    "Provide joint covariance or conditional independence, not both."
                )
            if (
                covariance.shape != (n, 4, 4)
                or not np.all(np.isfinite(covariance))
                or not np.allclose(
                    covariance,
                    np.swapaxes(covariance, -1, -2),
                    rtol=1.0e-8,
                    atol=1.0e-12,
                )
            ):
                raise ValueError(
                    "Affinity component covariance must be finite, symmetric, and shaped (condition, 4, 4)."
                )
            covariance_scale = np.maximum(np.max(np.abs(covariance), axis=(-2, -1)), 1.0)
            if np.any(
                np.linalg.eigvalsh(covariance) < -1.0e-10 * covariance_scale[:, None]
            ):
                raise ValueError(
                    "Affinity component covariance must be positive semidefinite."
                )
            expected_variances = np.stack(component_errors, axis=-1) ** 2
            diagonal = np.diagonal(covariance, axis1=-2, axis2=-1)
            if not np.allclose(
                diagonal[:, :3],
                expected_variances,
                rtol=1.0e-6,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "Joint covariance diagonal must match component standard errors."
                )
            if np.any(diagonal[:, 3] <= 0.0):
                raise ValueError(
                    "Joint covariance must quantify standard-state correction uncertainty."
                )
            if correction_error is not None and not np.allclose(
                diagonal[:, 3],
                correction_error**2,
                rtol=1.0e-6,
                atol=1.0e-12,
            ):
                raise ValueError(
                    "Joint covariance disagrees with standard-state correction uncertainty."
                )
            contrast = np.asarray([1.0, -1.0, -1.0, 1.0])
            prediction_variance = np.einsum("i,nij,j->n", contrast, covariance, contrast)
            prediction_error = np.sqrt(np.maximum(prediction_variance, 0.0))
        elif components_conditionally_independent and correction_error is not None:
            prediction_error = np.sqrt(
                sum(array**2 for array in component_errors) + correction_error**2
            )
        else:
            prediction_error = None
        for reference in (
            bound_sampling_reference,
            unbound_sampling_reference,
            binding_measurement_reference,
        ):
            if reference is not None and not isinstance(
                reference, ReferenceArtifactManifest
            ):
                raise TypeError(
                    "Affinity references must be ReferenceArtifactManifest or None."
                )
        if set(_role_case_ids(fit.campaign, "locked_evaluation")) != set(cases):
            raise ValueError(
                "Affinity observation cases must exactly equal the locked campaign role."
            )
        campaign_cases = {case.case_id: case for case in fit.campaign.cases}
        for case_id, independent_unit, preparation, condition in zip(
            cases, independent, preparations, conditions, strict=True
        ):
            campaign_case = campaign_cases[case_id]
            if (
                campaign_case.independent_unit_id != independent_unit
                or campaign_case.preparation_id != preparation
                or campaign_case.condition_id != condition
                or (
                    binding_measurement_reference is not None
                    and binding_measurement_reference.manifest_id
                    not in campaign_case.source_manifest_ids
                )
            ):
                raise ValueError(
                    "Locked affinity observation identity does not match the campaign."
                )
        predicted = (
            energy_components[0]
            - energy_components[1]
            - energy_components[2]
            + energy_components[3]
        )
        prediction_id = canonical_fingerprint(
            {
                "kind": "protein-nucleic-affinity-prediction",
                "campaign": fit.campaign.campaign_id,
                "model": fit.model_id,
                "fit": fit.fit_id,
                "cases": cases,
                "conditions": conditions,
                "standard_states": standard_states,
                "energy_components": tuple(
                    array_tree_fingerprint(array) for array in energy_components
                ),
                "component_errors": tuple(
                    array_tree_fingerprint(array) for array in component_errors
                ),
                "component_covariance": (
                    None if covariance is None else array_tree_fingerprint(covariance)
                ),
                "bound_source": (
                    None
                    if bound_sampling_reference is None
                    else bound_sampling_reference.manifest_id
                ),
                "unbound_source": (
                    None
                    if unbound_sampling_reference is None
                    else unbound_sampling_reference.manifest_id
                ),
            }
        )
        observation_id = canonical_fingerprint(
            {
                "kind": "locked-protein-nucleic-affinity-observation",
                "campaign": fit.campaign.campaign_id,
                "cases": cases,
                "independent_units": independent,
                "preparations": preparations,
                "conditions": conditions,
                "standard_states": standard_states,
                "observed": array_tree_fingerprint(observed),
                "standard_errors": array_tree_fingerprint(observation_error),
                "source": (
                    None
                    if binding_measurement_reference is None
                    else binding_measurement_reference.manifest_id
                ),
                "unit": energy_unit.unit_id,
            }
        )
        object.__setattr__(self, "condition_ids", conditions)
        object.__setattr__(self, "case_ids", cases)
        object.__setattr__(self, "preparation_ids", preparations)
        object.__setattr__(self, "independent_unit_ids", independent)
        object.__setattr__(self, "standard_state_ids", standard_states)
        object.__setattr__(self, "predicted_binding_free_energy", jnp.asarray(predicted))
        object.__setattr__(
            self,
            "prediction_standard_error",
            None if prediction_error is None else jnp.asarray(prediction_error),
        )
        object.__setattr__(
            self,
            "component_covariance",
            None if covariance is None else jnp.asarray(covariance),
        )
        object.__setattr__(
            self,
            "components_conditionally_independent",
            components_conditionally_independent,
        )
        object.__setattr__(self, "observed_binding_free_energy", jnp.asarray(observed))
        object.__setattr__(
            self, "observation_standard_error", jnp.asarray(observation_error)
        )
        object.__setattr__(self, "energy_unit", energy_unit)
        object.__setattr__(self, "bound_sampling_reference", bound_sampling_reference)
        object.__setattr__(self, "unbound_sampling_reference", unbound_sampling_reference)
        object.__setattr__(
            self, "binding_measurement_reference", binding_measurement_reference
        )
        object.__setattr__(self, "shared_sampling_lineage_ids", shared_lineage)
        object.__setattr__(self, "fit_source_manifest_ids", fit.source_manifest_ids)
        object.__setattr__(self, "fit_independent_unit_ids", fit.independent_unit_ids)
        object.__setattr__(self, "fit_preparation_ids", fit.preparation_ids)
        object.__setattr__(self, "campaign_id", fit.campaign.campaign_id)
        object.__setattr__(self, "campaign_criteria_ids", fit.campaign.criteria_ids)
        object.__setattr__(self, "model_id", fit.model_id)
        object.__setattr__(self, "fit_id", fit.fit_id)
        object.__setattr__(self, "fit_integrity_ids", fit.fit_integrity_ids)
        object.__setattr__(self, "prediction_id", prediction_id)
        object.__setattr__(self, "observation_id", observation_id)
        object.__setattr__(
            self,
            "input_id",
            canonical_fingerprint(
                {
                    "kind": "protein-nucleic-affinity-inputs",
                    "prediction": prediction_id,
                    "observation": observation_id,
                    "campaign": fit.campaign.campaign_id,
                    "model": fit.model_id,
                    "fit": fit.fit_id,
                    "cases": cases,
                    "preparations": preparations,
                    "conditions": conditions,
                    "independent_units": independent,
                    "standard_states": standard_states,
                    "energy_components": tuple(
                        array_tree_fingerprint(array) for array in energy_components
                    ),
                    "component_errors": tuple(
                        array_tree_fingerprint(array) for array in component_errors
                    ),
                    "standard_state_standard_error": (
                        None
                        if correction_error is None
                        else array_tree_fingerprint(correction_error)
                    ),
                    "component_covariance": (
                        None if covariance is None else array_tree_fingerprint(covariance)
                    ),
                    "components_conditionally_independent": (
                        components_conditionally_independent
                    ),
                    "observed": array_tree_fingerprint(observed),
                    "observation_standard_error": array_tree_fingerprint(
                        observation_error
                    ),
                    "unit": energy_unit.unit_id,
                    "references": tuple(
                        None if item is None else item.manifest_id
                        for item in (
                            bound_sampling_reference,
                            unbound_sampling_reference,
                            binding_measurement_reference,
                        )
                    ),
                    "shared_sampling_lineage_ids": shared_lineage,
                }
            ),
        )

    @property
    def missing_prerequisites(self) -> tuple[str, ...]:
        missing = []
        if self.prediction_standard_error is None:
            if self.components_conditionally_independent:
                missing.append("standard-state-correction-uncertainty")
            else:
                missing.append(
                    "affinity-component-joint-covariance-or-conditional-independence"
                )
        for label, reference in (
            ("bound-standard-state-sampling", self.bound_sampling_reference),
            ("unbound-standard-state-sampling", self.unbound_sampling_reference),
        ):
            if (
                reference is not None
                and reference.manifest_id not in self.fit_source_manifest_ids
            ):
                missing.append(f"{label}:not-in-fit-lineage")
        if set(self.independent_unit_ids) & set(self.fit_independent_unit_ids):
            missing.append("affinity-independent-unit-leakage")
        if set(self.preparation_ids) & set(self.fit_preparation_ids):
            missing.append("affinity-preparation-leakage")
        references = (
            ("bound-standard-state-sampling", self.bound_sampling_reference),
            ("unbound-standard-state-sampling", self.unbound_sampling_reference),
            ("independent-binding-measurement", self.binding_measurement_reference),
        )
        for name, reference in references:
            if reference is None:
                missing.append(name)
                continue
            if reference.uncertainty is None:
                missing.append(f"{name}:unquantified-uncertainty")
            missing.extend(
                f"{name}:rights:{reason}"
                for reason in reference.rights_refusal_reasons(training_use=True)
            )
        bound = self.bound_sampling_reference
        unbound = self.unbound_sampling_reference
        measurement = self.binding_measurement_reference
        if bound is not None and unbound is not None:
            if bound.manifest_id == unbound.manifest_id:
                missing.append("bound-unbound-sampling-reference-reuse")
            shared = set(self.shared_sampling_lineage_ids)
            actual_shared = set(bound.lineage_ids) & set(unbound.lineage_ids)
            if not shared <= actual_shared:
                missing.append("declared-shared-sampling-lineage-not-common")
            if actual_shared - shared:
                missing.append("undeclared-shared-sampling-lineage")
        if measurement is not None:
            measurement_lineage = {measurement.manifest_id, *measurement.lineage_ids}
            for label, sampling in (
                ("bound", bound),
                ("unbound", unbound),
            ):
                if sampling is None:
                    continue
                sampling_lineage = {sampling.manifest_id, *sampling.lineage_ids}
                if measurement_lineage & sampling_lineage:
                    missing.append(
                        f"independent-binding-measurement:{label}-lineage-overlap"
                    )
        return tuple(sorted(set(missing)))


@dataclass(frozen=True, slots=True, init=False)
class ProteinNucleicMechanicsPrediction:
    """Frozen mechanics prediction bound to a content-addressed model fit."""

    values: Array
    standard_errors: Array
    model_id: str
    fit_id: str
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    observation_id: str
    case_ids: tuple[str, ...]
    fit_integrity_ids: tuple[str, ...]
    fit_independent_unit_ids: tuple[str, ...]
    fit_preparation_ids: tuple[str, ...]
    prediction_id: str

    def __init__(
        self,
        observations: ProteinNucleicMechanicalObservations,
        values: ArrayLike,
        standard_errors: ArrayLike,
        fit: ProteinNucleicModelFit,
        /,
        *,
        unit: UnitDefinition,
    ):
        if not isinstance(observations, ProteinNucleicMechanicalObservations):
            raise TypeError("observations must be ProteinNucleicMechanicalObservations.")
        if not isinstance(fit, ProteinNucleicModelFit):
            raise TypeError("fit must be a ProteinNucleicModelFit.")
        _validate_locked_mechanics_campaign(fit.campaign, observations)
        predicted = np.asarray(values, dtype=float)
        errors = np.asarray(standard_errors, dtype=float)
        if (
            predicted.shape != observations.values.shape
            or errors.shape != predicted.shape
            or not np.all(np.isfinite(errors))
            or np.any(errors <= 0.0)
        ):
            raise ValueError(
                "Mechanical prediction and positive uncertainty must match every row."
            )
        factor = float(conversion_factor(unit, observations.unit))
        predicted = predicted * factor
        errors = errors * factor
        for name, value in (
            ("values", jnp.asarray(predicted)),
            ("standard_errors", jnp.asarray(errors)),
            ("model_id", fit.model_id),
            ("fit_id", fit.fit_id),
            ("campaign_id", fit.campaign.campaign_id),
            ("campaign_criteria_ids", fit.campaign.criteria_ids),
            ("fit_integrity_ids", fit.fit_integrity_ids),
            ("observation_id", observations.observation_id),
            ("case_ids", observations.case_ids),
            ("fit_independent_unit_ids", fit.independent_unit_ids),
            ("fit_preparation_ids", fit.preparation_ids),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "prediction_id",
            canonical_fingerprint(
                {
                    "kind": "locked-protein-nucleic-mechanics-prediction",
                    "values": array_tree_fingerprint(predicted),
                    "fit_integrity_ids": fit.fit_integrity_ids,
                    "standard_errors": array_tree_fingerprint(errors),
                    "unit": observations.unit.unit_id,
                    "model": fit.model_id,
                    "fit": fit.fit_id,
                    "campaign": fit.campaign.campaign_id,
                    "observation": observations.observation_id,
                    "cases": observations.case_ids,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class ProteinNucleicQualificationAssessment:
    lane: Literal["complex-mechanics", "binding-affinity"]
    metric_values: tuple[tuple[str, float], ...]
    standardized_residuals: Array
    campaign_id: str
    campaign_criteria_ids: tuple[str, ...]
    model_id: str
    fit_id: str
    prediction_id: str
    qualification_evidence_ids: tuple[str, ...]
    capability_name: str
    observable_ids: tuple[str, ...]
    condition_domain_ids: tuple[str, ...]
    support_attributes: tuple[tuple[str, str], ...]
    missing_prerequisites: tuple[str, ...]
    failed_checks: tuple[str, ...]
    assessment_id: str

    @property
    def status(self) -> str:
        if self.failed_checks:
            return "failed"
        if self.missing_prerequisites:
            return "inconclusive"
        return "ready-for-claim-evaluation"

    def evaluate_claim(
        self,
        claim: ScientificClaimProfile,
        stage_evidence: Sequence[QualificationEvidence],
        /,
        *,
        build_id: str,
        environment_id: str,
        backend: str,
        topology: str,
        precision: str,
        reduction: str,
        replay_id: str,
        raw_artifact_ids: Sequence[str],
        reviewer_id: str,
        issued_at: int,
        expires_at: int,
    ) -> QualificationEvidence:
        if not isinstance(claim, ScientificClaimProfile):
            raise TypeError("claim must be ScientificClaimProfile.")
        if (
            claim.capability_name != self.capability_name
            or claim.observable_ids != self.observable_ids
            or claim.condition_domain_ids != self.condition_domain_ids
            or claim.support.attributes != self.support_attributes
        ):
            raise ValueError(
                "Scientific claim scope must exactly match the protein-nucleic assessment."
            )
        if claim.campaign_id != self.campaign_id:
            raise ValueError(
                "Scientific claim profile must bind the prediction campaign."
            )
        if tuple(claim.frozen_criteria_ids) != self.campaign_criteria_ids:
            raise ValueError(
                "Scientific claim criteria must exactly match the prediction campaign."
            )
        if not _MANDATORY_CLAIM_STAGES <= set(claim.required_stage_ids):
            raise ValueError(
                "Protein-nucleic claim profile omits mandatory scientific stages."
            )
        supplied_evidence_ids = {
            evidence.evidence_id
            for evidence in stage_evidence
            if isinstance(evidence, QualificationEvidence)
        }
        if not set(self.qualification_evidence_ids) <= supplied_evidence_ids:
            raise ValueError(
                "Claim evaluation must retain prediction qualification evidence."
            )
        issues = self.failed_checks or self.missing_prerequisites
        if issues:
            return QualificationEvidence(
                "scientific",
                "failed" if self.failed_checks else "inconclusive",
                (
                    claim.claim_id,
                    claim.campaign_id,
                    claim.support.support_tuple_id,
                    *self.qualification_evidence_ids,
                ),
                build_id=build_id,
                environment_id=environment_id,
                backend=backend,
                topology=topology,
                precision=precision,
                reduction=reduction,
                replay_id=replay_id,
                criteria_ids=(f"protein-nucleic-{self.lane}-domain-readiness", *issues),
                raw_artifact_ids=raw_artifact_ids,
                reviewer_id=reviewer_id,
                issued_at=issued_at,
                expires_at=expires_at,
                reason=";".join(issues),
                requalification_triggers=claim.invalidation_triggers,
            )
        metrics = dict(self.metric_values)
        model_stages = {
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        }
        bound_stage_evidence = tuple(
            evidence
            for evidence in stage_evidence
            if not model_stages.intersection(evidence.criteria_ids)
            or evidence.evidence_id in self.qualification_evidence_ids
        )
        return claim.evaluate(
            metrics,
            bound_stage_evidence,
            metric_units={name: "1" for name in metrics},
            metric_aggregations={name: "independent_unit_macro" for name in metrics},
            build_id=build_id,
            environment_id=environment_id,
            backend=backend,
            topology=topology,
            precision=precision,
            reduction=reduction,
            replay_id=replay_id,
            raw_artifact_ids=raw_artifact_ids,
            reviewer_id=reviewer_id,
            issued_at=issued_at,
            expires_at=expires_at,
        )


def _macro_rms(residuals: np.ndarray, independent_unit_ids: tuple[str, ...]) -> float:
    identities = np.asarray(independent_unit_ids)
    scores = [
        float(np.sqrt(np.mean(np.square(residuals[identities == identity]))))
        for identity in dict.fromkeys(independent_unit_ids)
    ]
    return float(np.mean(scores))


def _prediction_evidence_status(
    evidence: Sequence[QualificationEvidence],
    campaign_id: str,
    model_id: str,
    fit_id: str,
    prediction_id: str,
    fit_integrity_ids: tuple[str, ...],
    /,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if any(not isinstance(item, QualificationEvidence) for item in evidence):
        raise TypeError("prediction_evidence must contain QualificationEvidence.")
    missing = []
    failed = []
    accepted = []
    for stage_id, include_prediction in (
        ("parameter-identifiability", False),
        ("predictive-calibration", False),
        ("locked-prediction", True),
    ):
        subjects = {campaign_id, model_id, fit_id, *fit_integrity_ids}
        if include_prediction:
            subjects.add(prediction_id)
        matching = tuple(
            item
            for item in evidence
            if item.evidence_kind == "scientific"
            and stage_id in item.criteria_ids
            and subjects <= set(item.subject_ids)
        )
        accepted.extend(item.evidence_id for item in matching)
        if any(item.outcome == "failed" for item in matching):
            failed.append(stage_id)
        elif not matching or any(item.outcome == "inconclusive" for item in matching):
            missing.append(stage_id)
    return tuple(missing), tuple(failed), tuple(sorted(set(accepted)))


def assess_protein_nucleic_mechanics(
    observations: ProteinNucleicMechanicalObservations,
    prediction: ProteinNucleicMechanicsPrediction,
    /,
    *,
    mapping_reference: ReferenceArtifactManifest | None,
    maximum_standardized_rms: float,
    prediction_evidence: Sequence[QualificationEvidence],
) -> ProteinNucleicQualificationAssessment:
    if not isinstance(observations, ProteinNucleicMechanicalObservations):
        raise TypeError("observations must be ProteinNucleicMechanicalObservations.")
    if not isinstance(prediction, ProteinNucleicMechanicsPrediction):
        raise TypeError("prediction must be a ProteinNucleicMechanicsPrediction record.")
    if (
        prediction.observation_id != observations.observation_id
        or prediction.case_ids != observations.case_ids
    ):
        raise ValueError("Mechanical prediction targets do not match observations.")
    if not math.isfinite(maximum_standardized_rms) or maximum_standardized_rms <= 0.0:
        raise ValueError("maximum_standardized_rms must be finite and positive.")
    predicted = np.asarray(prediction.values)
    combined_errors = np.sqrt(
        np.asarray(observations.standard_errors) ** 2
        + np.asarray(prediction.standard_errors) ** 2
    )
    residuals = (predicted - np.asarray(observations.values)) / combined_errors
    metric = _macro_rms(residuals, observations.independent_unit_ids)
    missing = []
    if mapping_reference is None:
        missing.append("protein-nucleic-coordinate-mapping")
    elif not isinstance(mapping_reference, ReferenceArtifactManifest):
        raise TypeError("mapping_reference must be a manifest or None.")
    elif mapping_reference.uncertainty is None:
        missing.append("protein-nucleic-coordinate-mapping:unquantified-uncertainty")
    else:
        mapping_reference.require_rights()
    failed = []
    evidence_missing, evidence_failed, evidence_ids = _prediction_evidence_status(
        prediction_evidence,
        prediction.campaign_id,
        prediction.model_id,
        prediction.fit_id,
        prediction.prediction_id,
        prediction.fit_integrity_ids,
    )
    missing.extend(evidence_missing)
    failed.extend(evidence_failed)
    if not np.all(np.isfinite(predicted)):
        failed.append("nonfinite-complex-mechanics-prediction")
    if set(prediction.fit_independent_unit_ids) & set(observations.independent_unit_ids):
        failed.append("independent-unit-leakage")
    if set(prediction.fit_preparation_ids) & set(observations.preparation_ids):
        failed.append("preparation-leakage")
    if metric > maximum_standardized_rms:
        failed.append("complex-mechanics-macro-standardized-rms")
    metrics = (("complex-mechanics-macro-standardized-rms", metric),)
    missing_tuple = tuple(sorted(set(missing)))
    failed_tuple = tuple(sorted(set(failed)))
    capability_name = "protein-nucleic-mechanics"
    observable_ids = (observations.observation_id,)
    condition_domain_ids = tuple(sorted(set(observations.condition_ids)))
    support_attributes = tuple(
        sorted(
            {
                "complex_construct_id": observations.complex_construct_id,
                "fit_id": prediction.fit_id,
                "model_id": prediction.model_id,
                "observable_kind": observations.observable_kind,
                "observation_id": observations.observation_id,
                "prediction_id": prediction.prediction_id,
                "unit_id": observations.unit.unit_id,
            }.items()
        )
    )
    assessment_id = canonical_fingerprint(
        {
            "kind": "protein-nucleic-mechanics-assessment",
            "observations": observations.observation_id,
            "prediction": prediction.prediction_id,
            "capability": capability_name,
            "observable_ids": observable_ids,
            "condition_domain_ids": condition_domain_ids,
            "support_attributes": support_attributes,
            "campaign_criteria_ids": prediction.campaign_criteria_ids,
            "metrics": metrics,
            "missing": missing_tuple,
            "qualification_evidence": evidence_ids,
            "failed": failed_tuple,
        }
    )
    return ProteinNucleicQualificationAssessment(
        "complex-mechanics",
        metrics,
        jnp.asarray(residuals),
        prediction.campaign_id,
        prediction.campaign_criteria_ids,
        prediction.model_id,
        prediction.fit_id,
        prediction.prediction_id,
        evidence_ids,
        capability_name,
        observable_ids,
        condition_domain_ids,
        support_attributes,
        missing_tuple,
        failed_tuple,
        assessment_id,
    )


def assess_protein_nucleic_affinity(
    inputs: ProteinNucleicAffinityInputs,
    /,
    *,
    maximum_standardized_rms: float,
    prediction_evidence: Sequence[QualificationEvidence],
) -> ProteinNucleicQualificationAssessment:
    if not isinstance(inputs, ProteinNucleicAffinityInputs):
        raise TypeError("inputs must be ProteinNucleicAffinityInputs.")
    if not math.isfinite(maximum_standardized_rms) or maximum_standardized_rms <= 0.0:
        raise ValueError("maximum_standardized_rms must be finite and positive.")
    prediction_error = (
        np.zeros_like(np.asarray(inputs.observation_standard_error))
        if inputs.prediction_standard_error is None
        else np.asarray(inputs.prediction_standard_error)
    )
    combined = np.sqrt(
        prediction_error**2 + np.asarray(inputs.observation_standard_error) ** 2
    )
    residuals = (
        np.asarray(inputs.predicted_binding_free_energy)
        - np.asarray(inputs.observed_binding_free_energy)
    ) / combined
    metric = _macro_rms(residuals, inputs.independent_unit_ids)
    missing = list(inputs.missing_prerequisites)
    failed = []
    evidence_missing, evidence_failed, evidence_ids = _prediction_evidence_status(
        prediction_evidence,
        inputs.campaign_id,
        inputs.model_id,
        inputs.fit_id,
        inputs.prediction_id,
        inputs.fit_integrity_ids,
    )
    missing.extend(evidence_missing)
    failed.extend(evidence_failed)
    if metric > maximum_standardized_rms:
        failed.append("binding-affinity-macro-standardized-rms")
    missing_tuple = tuple(sorted(set(missing)))
    failed_tuple = tuple(sorted(set(failed)))
    metrics = (("binding-affinity-macro-standardized-rms", metric),)
    capability_name = "protein-nucleic-affinity"
    observable_ids = (inputs.observation_id,)
    condition_domain_ids = tuple(sorted(set(inputs.condition_ids)))
    support_attributes = tuple(
        sorted(
            {
                "fit_id": inputs.fit_id,
                "model_id": inputs.model_id,
                "observation_id": inputs.observation_id,
                "prediction_id": inputs.prediction_id,
                "standard_state_scope_id": canonical_fingerprint(
                    tuple(sorted(set(inputs.standard_state_ids)))
                ),
                "unit_id": inputs.energy_unit.unit_id,
            }.items()
        )
    )
    assessment_id = canonical_fingerprint(
        {
            "kind": "protein-nucleic-affinity-assessment",
            "inputs": inputs.input_id,
            "campaign": inputs.campaign_id,
            "campaign_criteria_ids": inputs.campaign_criteria_ids,
            "model": inputs.model_id,
            "fit": inputs.fit_id,
            "prediction": inputs.prediction_id,
            "observation": inputs.observation_id,
            "capability": capability_name,
            "observable_ids": observable_ids,
            "condition_domain_ids": condition_domain_ids,
            "support_attributes": support_attributes,
            "qualification_evidence": evidence_ids,
            "metrics": metrics,
            "missing": missing_tuple,
            "failed": failed_tuple,
        }
    )
    return ProteinNucleicQualificationAssessment(
        "binding-affinity",
        metrics,
        jnp.asarray(residuals),
        inputs.campaign_id,
        inputs.campaign_criteria_ids,
        inputs.model_id,
        inputs.fit_id,
        inputs.prediction_id,
        evidence_ids,
        capability_name,
        observable_ids,
        condition_domain_ids,
        support_attributes,
        missing_tuple,
        failed_tuple,
        assessment_id,
    )


__all__ = [
    "ProteinNucleicAffinityInputs",
    "ProteinNucleicMechanicalObservations",
    "ProteinNucleicMechanicsPrediction",
    "ProteinNucleicModelFit",
    "ProteinNucleicQualificationAssessment",
    "assess_protein_nucleic_affinity",
    "assess_protein_nucleic_mechanics",
]
