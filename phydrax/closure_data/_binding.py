#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume._closure import ConservativeFaceClosurePlan
from ..discretization.spectral._coordinates import HermitianSpectralCoordinates
from ..discretization.spectral._dealias import PreparedDealiasingPlan
from ..discretization.spectral._incompressible import PeriodicLerayProjector
from ._state import FlowStateSchema


ClosureDeploymentKind = Literal["conservative_face", "spectral_drift"]
SpectralEnergyPolicy = Literal["nonincreasing", "diagnostic"]


class LearnedClosureBindingPlan(StrictModule, NonTrainableState):
    """Artifact- and schema-bound learned predictor with an explicit deployment ABI."""

    predictor: Callable = eqx.field(static=True)
    deployment_kind: ClosureDeploymentKind = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    input_component_names: tuple[str, ...] = eqx.field(static=True)
    output_component_names: tuple[str, ...] = eqx.field(static=True)
    model_artifact_id: str = eqx.field(static=True)
    normalizer_provenance_id: str = eqx.field(static=True)
    differentiability: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        predictor: Callable,
        /,
        *,
        deployment_kind: ClosureDeploymentKind,
        schema_id: str,
        input_component_names: tuple[str, ...],
        output_component_names: tuple[str, ...],
        model_artifact_id: str,
        normalizer_provenance_id: str,
        differentiability: str = "smooth_discrete",
    ):
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        deployment = str(deployment_kind).strip()
        schema = str(schema_id).strip()
        inputs = tuple(str(value).strip() for value in input_component_names)
        outputs = tuple(str(value).strip() for value in output_component_names)
        artifact = str(model_artifact_id).strip()
        normalizer = str(normalizer_provenance_id).strip()
        differentiability_ = str(differentiability).strip()
        if (
            deployment not in ("conservative_face", "spectral_drift")
            or not schema
            or not inputs
            or not outputs
            or any(not value for value in (*inputs, *outputs))
            or len(set(inputs)) != len(inputs)
            or len(set(outputs)) != len(outputs)
            or not artifact
            or not normalizer
            or differentiability_
            not in ("smooth_discrete", "branchwise", "smooth_surrogate")
        ):
            raise ValueError("Learned closure binding metadata is invalid.")
        self.predictor = predictor
        self.deployment_kind = deployment
        self.schema_id = schema
        self.input_component_names = inputs
        self.output_component_names = outputs
        self.model_artifact_id = artifact
        self.normalizer_provenance_id = normalizer
        self.differentiability = differentiability_
        self.binding_id = canonical_fingerprint(
            {
                "kind": "learned-closure-binding-plan",
                "deployment_kind": deployment,
                "schema": schema,
                "input_components": list(inputs),
                "output_components": list(outputs),
                "model_artifact": artifact,
                "normalizer_provenance": normalizer,
                "differentiability": differentiability_,
            }
        )

    def bind_conservative_faces(
        self,
        schema: FlowStateSchema,
        /,
        *,
        consistency_tolerance: float = 1e-10,
    ) -> ConservativeFaceClosurePlan:
        self._validate_schema(schema)
        if self.deployment_kind != "conservative_face":
            raise ValueError(
                "Only conservative_face bindings can enter a face flux plan."
            )
        if self.output_component_names != schema.component_names:
            raise ValueError(
                "A conservative face closure must output every conservative component in schema order."
            )
        return ConservativeFaceClosurePlan(
            self.predictor,
            closure_id=self.binding_id,
            consistency_tolerance=consistency_tolerance,
            differentiability=self.differentiability,
        )

    def bind_spectral_drift(
        self,
        schema: FlowStateSchema,
        projector: PeriodicLerayProjector,
        hermitian_coordinates: HermitianSpectralCoordinates,
        dealiasing: PreparedDealiasingPlan,
        /,
        *,
        energy_policy: SpectralEnergyPolicy = "nonincreasing",
        evidence_tolerance: float = 1e-10,
    ) -> PreparedSpectralDriftHook:
        self._validate_schema(schema)
        if self.deployment_kind != "spectral_drift":
            raise ValueError("Only spectral_drift bindings can enter a spectral hook.")
        if self.output_component_names != schema.velocity_names:
            raise ValueError(
                "A spectral drift binding must output the declared velocity components in order."
            )
        return PreparedSpectralDriftHook(
            self.predictor,
            projector,
            hermitian_coordinates,
            dealiasing,
            binding_id=self.binding_id,
            energy_policy=energy_policy,
            evidence_tolerance=evidence_tolerance,
        )

    def _validate_schema(self, schema: FlowStateSchema) -> None:
        if not isinstance(schema, FlowStateSchema):
            raise TypeError("schema must be a FlowStateSchema.")
        if schema.schema_id != self.schema_id:
            raise ValueError(
                "Learned closure binding schema identity does not match deployment."
            )
        if any(
            value not in schema.component_names for value in self.input_component_names
        ):
            raise ValueError(
                "Learned closure input components are absent from the schema."
            )
        if any(
            value not in schema.component_names for value in self.output_component_names
        ):
            raise ValueError(
                "Learned closure output components are absent from the schema."
            )


class SpectralDriftEvidence(StrictModule, NonTrainableState):
    raw_energy_rate: Array
    constrained_energy_rate: Array
    divergence_norm: Array
    hermitian_defect: Array
    nonfinite_count: Array
    valid: Array
    binding_id: str = eqx.field(static=True)
    projector_id: str = eqx.field(static=True)
    hermitian_coordinate_id: str = eqx.field(static=True)
    dealiasing_id: str = eqx.field(static=True)
    dealiasing_kind: str = eqx.field(static=True)
    dealiasing_exact: bool = eqx.field(static=True)
    energy_policy: SpectralEnergyPolicy = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        raw_energy_rate: ArrayLike,
        constrained_energy_rate: ArrayLike,
        divergence_norm: ArrayLike,
        hermitian_defect: ArrayLike,
        nonfinite_count: ArrayLike,
        valid: ArrayLike,
        binding_id: str,
        projector_id: str,
        hermitian_coordinate_id: str,
        dealiasing_id: str,
        dealiasing_kind: str,
        dealiasing_exact: bool,
        energy_policy: SpectralEnergyPolicy,
    ):
        self.raw_energy_rate = jnp.asarray(raw_energy_rate)
        self.constrained_energy_rate = jnp.asarray(constrained_energy_rate)
        self.divergence_norm = jnp.asarray(divergence_norm)
        self.hermitian_defect = jnp.asarray(hermitian_defect)
        self.nonfinite_count = jnp.asarray(nonfinite_count)
        self.valid = jnp.asarray(valid, dtype=bool)
        self.binding_id = str(binding_id)
        self.projector_id = str(projector_id)
        self.hermitian_coordinate_id = str(hermitian_coordinate_id)
        self.dealiasing_id = str(dealiasing_id)
        self.dealiasing_kind = str(dealiasing_kind)
        self.dealiasing_exact = bool(dealiasing_exact)
        self.energy_policy = energy_policy
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "spectral-drift-evidence",
                "binding": self.binding_id,
                "projector": self.projector_id,
                "hermitian_coordinates": self.hermitian_coordinate_id,
                "dealiasing": self.dealiasing_id,
                "dealiasing_kind": self.dealiasing_kind,
                "dealiasing_exact": self.dealiasing_exact,
                "energy_policy": energy_policy,
            }
        )


class SpectralFallbackArtifact(StrictModule, NonTrainableState):
    """Typed record of the explicit zero-drift fallback selected by evidence."""

    used: Array
    reason_code: Array
    binding_id: str = eqx.field(static=True)
    fallback_kind: str = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(self, *, used: ArrayLike, reason_code: ArrayLike, binding_id: str):
        binding = str(binding_id).strip()
        if not binding:
            raise ValueError("binding_id must be non-empty.")
        self.used = jnp.asarray(used, dtype=bool)
        self.reason_code = jnp.asarray(reason_code, dtype=jnp.int32)
        self.binding_id = binding
        self.fallback_kind = "zero_spectral_drift"
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "spectral-drift-fallback-artifact",
                "binding": binding,
                "fallback_kind": self.fallback_kind,
                "reason_codes": {
                    "0": "not_used",
                    "1": "nonfinite_prediction",
                    "2": "constraint_evidence_failed",
                    "3": "energy_contract_failed",
                },
            }
        )


class SpectralDriftResult(StrictModule, NonTrainableState):
    drift: Array
    evidence: SpectralDriftEvidence
    fallback: SpectralFallbackArtifact

    def __init__(
        self,
        drift: ArrayLike,
        evidence: SpectralDriftEvidence,
        fallback: SpectralFallbackArtifact,
        /,
    ):
        if not isinstance(evidence, SpectralDriftEvidence):
            raise TypeError("evidence must be SpectralDriftEvidence.")
        if not isinstance(fallback, SpectralFallbackArtifact):
            raise TypeError("fallback must be SpectralFallbackArtifact.")
        self.drift = jnp.asarray(drift)
        self.evidence = evidence
        self.fallback = fallback


class PreparedSpectralDriftHook(StrictModule, NonTrainableState):
    """Dealiased, solenoidal, Hermitian learned drift with energy evidence."""

    predictor: Callable = eqx.field(static=True)
    projector: PeriodicLerayProjector
    hermitian_coordinates: HermitianSpectralCoordinates
    dealiasing: PreparedDealiasingPlan
    binding_id: str = eqx.field(static=True)
    energy_policy: SpectralEnergyPolicy = eqx.field(static=True)
    evidence_tolerance: float = eqx.field(static=True)
    hook_id: str = eqx.field(static=True)

    def __init__(
        self,
        predictor: Callable,
        projector: PeriodicLerayProjector,
        hermitian_coordinates: HermitianSpectralCoordinates,
        dealiasing: PreparedDealiasingPlan,
        /,
        *,
        binding_id: str,
        energy_policy: SpectralEnergyPolicy,
        evidence_tolerance: float,
    ):
        if not callable(predictor):
            raise TypeError("predictor must be callable.")
        if not isinstance(projector, PeriodicLerayProjector):
            raise TypeError("projector must be a PeriodicLerayProjector.")
        if not isinstance(hermitian_coordinates, HermitianSpectralCoordinates):
            raise TypeError("hermitian_coordinates must be HermitianSpectralCoordinates.")
        if not isinstance(dealiasing, PreparedDealiasingPlan):
            raise TypeError("dealiasing must be a PreparedDealiasingPlan.")
        binding = str(binding_id).strip()
        policy = str(energy_policy).strip()
        tolerance = float(evidence_tolerance)
        if (
            not binding
            or policy not in ("nonincreasing", "diagnostic")
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or projector.state_shape != hermitian_coordinates.state_shape
            or dealiasing.retained.prepared_id != projector.discretization.prepared_id
            or hermitian_coordinates.discretization.prepared_id
            != projector.discretization.prepared_id
        ):
            raise ValueError("Spectral drift hook contracts are incompatible.")
        self.predictor = predictor
        self.projector = projector
        self.hermitian_coordinates = hermitian_coordinates
        self.dealiasing = dealiasing
        self.binding_id = binding
        self.energy_policy = policy
        self.evidence_tolerance = tolerance
        self.hook_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-drift-hook",
                "binding": binding,
                "projector": projector.projector_id,
                "hermitian_coordinates": hermitian_coordinates.coordinate_id,
                "dealiasing": dealiasing.prepared_id,
                "energy_policy": policy,
                "evidence_tolerance": tolerance,
            }
        )

    def apply(self, state: ArrayLike, args: Any = None, /) -> SpectralDriftResult:
        value = self.projector.validate_state(state, owner="Spectral closure state")
        value = self.hermitian_coordinates.validate_state(value)
        input_divergence = self.projector.divergence_norm(value)
        input_hermitian = self.hermitian_coordinates.reality_defect(value)
        value = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value))
            | (input_divergence > self.evidence_tolerance)
            | (input_hermitian > self.evidence_tolerance),
            "Spectral closure input violates finiteness, projection, or Hermitian contracts.",
        )
        raw = jnp.asarray(self.predictor(value, args))
        if raw.shape != value.shape:
            raise ValueError("Spectral closure predictor output must match state shape.")
        if not jnp.issubdtype(raw.dtype, jnp.complexfloating):
            raise TypeError(
                "Spectral closure predictor output must be complex modal data."
            )
        raw_nonfinite = jnp.sum(~jnp.isfinite(raw))
        raw_energy_rate = _energy_rate(value, raw)
        constrained = self.dealiasing.filter(raw)
        constrained = self.projector.project(constrained)
        constrained = self.hermitian_coordinates.project(constrained)
        constrained = self.projector.project(constrained)
        if self.energy_policy == "nonincreasing":
            rate = _energy_rate(value, constrained)
            state_norm_squared = jnp.real(jnp.vdot(value, value))
            safe_norm = jnp.where(state_norm_squared > 0.0, state_norm_squared, 1.0)
            multiplier = jnp.where(rate > 0.0, rate / safe_norm, 0.0)
            constrained = constrained - multiplier.astype(constrained.dtype) * value
            constrained = self.projector.project(
                self.hermitian_coordinates.project(constrained)
            )
        constrained_energy_rate = _energy_rate(value, constrained)
        divergence = self.projector.divergence_norm(constrained)
        hermitian = self.hermitian_coordinates.reality_defect(constrained)
        constraint_failed = (
            (divergence > self.evidence_tolerance)
            | (hermitian > self.evidence_tolerance)
            | jnp.any(~jnp.isfinite(constrained))
        )
        energy_failed = (self.energy_policy == "nonincreasing") & (
            constrained_energy_rate > self.evidence_tolerance
        )
        nonfinite_failed = raw_nonfinite > 0
        fallback_used = nonfinite_failed | constraint_failed | energy_failed
        reason_code = jnp.where(
            nonfinite_failed,
            1,
            jnp.where(constraint_failed, 2, jnp.where(energy_failed, 3, 0)),
        )
        drift = jnp.where(fallback_used, jnp.zeros_like(constrained), constrained)
        evidence = SpectralDriftEvidence(
            raw_energy_rate=raw_energy_rate,
            constrained_energy_rate=constrained_energy_rate,
            divergence_norm=divergence,
            hermitian_defect=hermitian,
            nonfinite_count=raw_nonfinite,
            valid=~fallback_used,
            binding_id=self.binding_id,
            projector_id=self.projector.projector_id,
            hermitian_coordinate_id=self.hermitian_coordinates.coordinate_id,
            dealiasing_id=self.dealiasing.prepared_id,
            dealiasing_kind=self.dealiasing.report.kind,
            dealiasing_exact=self.dealiasing.report.exact,
            energy_policy=self.energy_policy,
        )
        fallback = SpectralFallbackArtifact(
            used=fallback_used,
            reason_code=reason_code,
            binding_id=self.binding_id,
        )
        return SpectralDriftResult(drift, evidence, fallback)

    def __call__(self, state: ArrayLike, args: Any = None, /) -> Array:
        return self.apply(state, args).drift


def _energy_rate(state: Array, drift: Array) -> Array:
    return jnp.real(jnp.vdot(state, drift))


__all__ = [
    "ClosureDeploymentKind",
    "LearnedClosureBindingPlan",
    "PreparedSpectralDriftHook",
    "SpectralDriftEvidence",
    "SpectralDriftResult",
    "SpectralEnergyPolicy",
    "SpectralFallbackArtifact",
]
