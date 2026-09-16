#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._chemical_mechanism import PreparedChemicalMechanism
from ...qualification import ReferenceArtifactManifest


class LearnedChemicalFallbackReason(IntEnum):
    NONE = 0
    OUT_OF_SUPPORT = 1
    UNCERTAIN = 2
    NONFINITE_MODEL = 3
    NEGATIVE_SPECIES = 4
    INVARIANT_FAILURE = 5
    EXACT_FAILURE = 6


class LearnedChemicalFeatureSchema(StrictModule, NonTrainableState):
    feature_names: tuple[str, ...] = eqx.field(static=True)
    feature_units: tuple[str, ...] = eqx.field(static=True)
    lower: Array
    upper: Array
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        feature_names: tuple[str, ...],
        feature_units: tuple[str, ...],
        lower: ArrayLike,
        upper: ArrayLike,
        /,
    ):
        names = tuple(str(value).strip() for value in feature_names)
        units = tuple(str(value).strip() for value in feature_units)
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        if (
            not names
            or len(names) != len(units)
            or len(set(names)) != len(names)
            or any(not value for value in (*names, *units))
            or lower_.shape != (len(names),)
            or upper_.shape != lower_.shape
            or np.any(~np.isfinite(lower_))
            or np.any(~np.isfinite(upper_))
            or np.any(upper_ <= lower_)
        ):
            raise ValueError("Learned chemistry feature schema is invalid.")
        self.feature_names = names
        self.feature_units = units
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.schema_id = canonical_fingerprint(
            {
                "kind": "learned-chemical-feature-schema",
                "names": names,
                "units": units,
                "lower": array_tree_fingerprint(lower_),
                "upper": array_tree_fingerprint(upper_),
            }
        )

    def evaluate(self, features: ArrayLike, /) -> tuple[Array, Array]:
        value = jnp.asarray(features, dtype=self.lower.dtype)
        if value.shape[-1:] != self.lower.shape:
            raise ValueError("Learned chemistry features do not match the schema.")
        scale = self.upper - self.lower
        normalized = 2.0 * (value - self.lower) / scale - 1.0
        margin = jnp.min(
            jnp.minimum(value - self.lower, self.upper - value) / scale,
            axis=-1,
        )
        supported = (
            jnp.all(jnp.isfinite(value), axis=-1)
            & jnp.all(value >= self.lower, axis=-1)
            & jnp.all(value <= self.upper, axis=-1)
        )
        return normalized, jnp.where(supported, margin, -jnp.abs(margin))


class LearnedChemicalTransitionResult(StrictModule):
    candidate_concentrations: Array
    accepted_concentrations: Array
    learned_concentrations: Array
    exact_concentrations: Array
    uncertainty: Array
    support_margin: Array
    fallback_used: Array
    fallback_reason: Array
    element_residual: Array
    charge_residual: Array
    derivative_valid: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class LearnedChemicalTransitionPlan(StrictModule, NonTrainableState):
    mechanism: PreparedChemicalMechanism
    feature_schema: LearnedChemicalFeatureSchema
    model: Callable = eqx.field(static=True)
    uncertainty_model: Callable = eqx.field(static=True)
    model_manifest: ReferenceArtifactManifest = eqx.field(static=True)
    training_manifests: tuple[ReferenceArtifactManifest, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    maximum_uncertainty: float = eqx.field(static=True)
    exact_subcycles: int = eqx.field(static=True)
    exact_iterations: int = eqx.field(static=True)
    invariant_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mechanism: PreparedChemicalMechanism,
        feature_schema: LearnedChemicalFeatureSchema,
        model: Callable[[Array], ArrayLike],
        uncertainty_model: Callable[[Array], ArrayLike],
        model_manifest: ReferenceArtifactManifest,
        training_manifests: tuple[ReferenceArtifactManifest, ...],
        /,
        *,
        model_id: str,
        maximum_uncertainty: float,
        exact_subcycles: int = 8,
        exact_iterations: int = 4,
        invariant_tolerance: float = 1.0e-9,
        commercial_use: bool = False,
        export: bool = False,
    ):
        if not isinstance(mechanism, PreparedChemicalMechanism):
            raise TypeError("mechanism must be PreparedChemicalMechanism.")
        if not isinstance(feature_schema, LearnedChemicalFeatureSchema):
            raise TypeError("feature_schema must be LearnedChemicalFeatureSchema.")
        expected_features = mechanism.schema.species_count + 3
        if len(feature_schema.feature_names) != expected_features:
            raise ValueError(
                "Learned chemistry features must be log concentrations, temperature, pressure, and step."
            )
        if not callable(model) or not callable(uncertainty_model):
            raise TypeError(
                "Learned chemistry model and uncertainty model must be callable."
            )
        manifests = tuple(training_manifests)
        if (
            not isinstance(model_manifest, ReferenceArtifactManifest)
            or not manifests
            or any(
                not isinstance(value, ReferenceArtifactManifest) for value in manifests
            )
        ):
            raise TypeError("Learned chemistry requires model and training manifests.")
        model_manifest.require_rights(commercial_use=commercial_use, export=export)
        for manifest in manifests:
            manifest.require_rights(
                commercial_use=commercial_use,
                training_use=True,
                export=export,
            )
        identifier = str(model_id).strip()
        uncertainty = float(maximum_uncertainty)
        subcycles, iterations = int(exact_subcycles), int(exact_iterations)
        tolerance = float(invariant_tolerance)
        if (
            not identifier
            or not isfinite(uncertainty)
            or uncertainty < 0.0
            or subcycles < 1
            or iterations < 1
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Learned chemistry identity or numerical policy is invalid.")
        self.mechanism = mechanism
        self.feature_schema = feature_schema
        self.model = model
        self.uncertainty_model = uncertainty_model
        self.model_manifest = model_manifest
        self.training_manifests = manifests
        self.model_id = identifier
        self.maximum_uncertainty = uncertainty
        self.exact_subcycles = subcycles
        self.exact_iterations = iterations
        self.invariant_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "learned-chemical-transition-with-exact-fallback",
                "mechanism": mechanism.mechanism_id,
                "features": feature_schema.schema_id,
                "model_manifest": model_manifest.manifest_id,
                "training_manifests": [value.manifest_id for value in manifests],
                "model_id": identifier,
                "maximum_uncertainty": uncertainty,
                "exact_subcycles": subcycles,
                "exact_iterations": iterations,
                "invariant_tolerance": tolerance,
            }
        )

    def _features(self, concentrations, temperature, pressure, step):
        tiny = jnp.finfo(concentrations.dtype).tiny
        return jnp.concatenate(
            (
                jnp.log(jnp.maximum(concentrations, tiny)),
                temperature[..., None],
                pressure[..., None],
                jnp.log(jnp.maximum(step, tiny))[..., None],
            ),
            axis=-1,
        )

    def _exact(self, concentrations, temperature, pressure, step, runtime):
        substep = step / self.exact_subcycles
        state = concentrations
        successful = jnp.all(jnp.isfinite(state) & (state >= 0.0), axis=-1)
        for _ in range(self.exact_subcycles):
            initial = self.mechanism.evaluate(
                state, temperature, pressure, runtime=runtime
            )
            candidate = state + substep * initial.species_amount_rate
            stage_success = initial.successful
            for _ in range(self.exact_iterations):
                safe = jnp.where(
                    jnp.all(candidate >= 0.0, axis=-1)[..., None],
                    candidate,
                    state,
                )
                evaluated = self.mechanism.evaluate(
                    safe, temperature, pressure, runtime=runtime
                )
                candidate = state + 0.5 * substep * (
                    initial.species_amount_rate + evaluated.species_amount_rate
                )
                stage_success = stage_success & evaluated.successful
            positive = jnp.all(candidate >= 0.0, axis=-1)
            successful = successful & stage_success & positive
            state = jnp.where(successful[..., None], candidate, state)
        return state, successful

    def advance(
        self,
        concentrations: ArrayLike,
        temperature: ArrayLike,
        pressure: ArrayLike,
        step_size: ArrayLike,
        runtime: Any = None,
        /,
    ) -> LearnedChemicalTransitionResult:
        concentration = jnp.asarray(concentrations)
        temperature_, pressure_, step = jnp.broadcast_arrays(
            jnp.asarray(temperature, dtype=concentration.dtype),
            jnp.asarray(pressure, dtype=concentration.dtype),
            jnp.asarray(step_size, dtype=concentration.dtype),
        )
        if concentration.shape[-1:] != (self.mechanism.schema.species_count,):
            raise ValueError("Learned chemistry concentrations have invalid shape.")
        if temperature_.shape != concentration.shape[:-1]:
            raise ValueError("Learned chemistry thermodynamic batch shape is invalid.")
        features = self._features(concentration, temperature_, pressure_, step)
        normalized, support_margin = self.feature_schema.evaluate(features)
        extent = jnp.asarray(self.model(normalized), dtype=concentration.dtype)
        uncertainty = jnp.asarray(
            self.uncertainty_model(normalized), dtype=concentration.dtype
        )
        if extent.shape != concentration.shape[:-1] + (self.mechanism.reaction_count,):
            raise ValueError("Learned model must return one extent per reaction.")
        if uncertainty.shape != concentration.shape[:-1]:
            raise ValueError("Learned uncertainty must match the batch shape.")
        learned = concentration + contract(
            "...r,rs->...s",
            extent,
            self.mechanism.net_stoichiometry.astype(concentration.dtype),
            backend="jax",
        )
        amount_change = learned - concentration
        element_residual = self.mechanism.schema.element_amount(amount_change)
        charge_residual = self.mechanism.schema.charge_amount(amount_change)
        scale = jnp.maximum(jnp.max(jnp.abs(concentration), axis=-1), 1.0)
        finite_model = jnp.all(jnp.isfinite(learned), axis=-1) & jnp.isfinite(uncertainty)
        positive = jnp.all(learned >= 0.0, axis=-1)
        invariant = (
            jnp.max(jnp.abs(element_residual), axis=-1)
            <= self.invariant_tolerance * scale
        ) & (jnp.abs(charge_residual) <= self.invariant_tolerance * scale)
        supported = support_margin >= 0.0
        certain = uncertainty <= self.maximum_uncertainty
        use_learned = supported & certain & finite_model & positive & invariant

        def exact_transition(_):
            return self._exact(concentration, temperature_, pressure_, step, runtime)

        exact, exact_success = jax.lax.cond(
            jnp.all(use_learned),
            lambda _: (concentration, jnp.ones(concentration.shape[:-1], dtype=bool)),
            exact_transition,
            operand=None,
        )
        candidate = jnp.where(use_learned[..., None], learned, exact)
        fallback = ~use_learned
        reason = jnp.where(
            ~supported,
            int(LearnedChemicalFallbackReason.OUT_OF_SUPPORT),
            jnp.where(
                ~certain,
                int(LearnedChemicalFallbackReason.UNCERTAIN),
                jnp.where(
                    ~finite_model,
                    int(LearnedChemicalFallbackReason.NONFINITE_MODEL),
                    jnp.where(
                        ~positive,
                        int(LearnedChemicalFallbackReason.NEGATIVE_SPECIES),
                        jnp.where(
                            ~invariant,
                            int(LearnedChemicalFallbackReason.INVARIANT_FAILURE),
                            int(LearnedChemicalFallbackReason.NONE),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        reason = jnp.where(
            fallback & ~exact_success,
            int(LearnedChemicalFallbackReason.EXACT_FAILURE),
            reason,
        ).astype(jnp.int32)
        successful = jnp.where(use_learned, True, exact_success) & jnp.all(
            jnp.isfinite(candidate), axis=-1
        )
        accepted = jnp.where(successful[..., None], candidate, concentration)
        boundary_distance = jnp.minimum(
            support_margin,
            jnp.abs(self.maximum_uncertainty - uncertainty),
        )
        derivative_valid = successful & (
            boundary_distance > 32.0 * jnp.finfo(concentration.dtype).eps
        )
        return LearnedChemicalTransitionResult(
            candidate,
            accepted,
            learned,
            exact,
            uncertainty,
            support_margin,
            fallback,
            reason,
            element_residual,
            charge_residual,
            derivative_valid,
            successful,
            self.plan_id,
        )


__all__ = [
    "LearnedChemicalFallbackReason",
    "LearnedChemicalFeatureSchema",
    "LearnedChemicalTransitionPlan",
    "LearnedChemicalTransitionResult",
]
