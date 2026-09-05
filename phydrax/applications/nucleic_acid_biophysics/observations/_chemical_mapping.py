# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Processed chemical mapping with native covariance likelihood and optimization."""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .... import linalg as la
from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ....ein import contract
from ....observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    CorrelatedGaussianPlan,
    LinearObservationPlan,
    TheoryVector,
)
from ....optim import least_squares, LevenbergMarquardt
from ....qualification import ReferenceArtifactManifest
from ....units import conversion_factor, SECOND, UnitDefinition
from .._construct import NucleicAcidConstruct, NucleotideKey


@dataclass(frozen=True, slots=True)
class ChemicalMappingCondition:
    condition_id: str
    annotations: tuple[str, ...]
    exposure: float | None
    exposure_unit: UnitDefinition = SECOND

    def __post_init__(self):
        conversion_factor(self.exposure_unit, SECOND)
        if (
            not self.condition_id
            or not isinstance(self.annotations, tuple)
            or any(not item for item in self.annotations)
        ):
            raise ValueError(
                "A condition requires explicit identity and immutable annotations."
            )
        if self.exposure is not None and (
            not np.isfinite(self.exposure) or self.exposure <= 0
        ):
            raise ValueError(
                "Exposure must be positive, or None when genuinely unreported."
            )


class ChemicalMappingObservation(StrictModule):
    """One construct/condition/replicate. Negative corrected reactivities are valid.

    covariance_lower, when supplied, is the covariance Cholesky of OBSERVED rows
    in their original relative order, not a precision or a factor sliced from a
    larger covariance. This preserves correlations without imputing missing data.
    Standard deviations alone imply a declared diagonal-noise approximation.
    """

    construct: NucleicAcidConstruct = eqx.field(static=True)
    nucleotide_keys: tuple[NucleotideKey, ...] = eqx.field(static=True)
    reagent: str = eqx.field(static=True)
    condition: ChemicalMappingCondition = eqx.field(static=True)
    replicate_id: str = eqx.field(static=True)
    preprocessing: tuple[str, ...] = eqx.field(static=True)
    source: ReferenceArtifactManifest
    reactivity: Array
    standard_deviation: Array
    observed: Array
    observed_indices: Array
    covariance: CholeskyCovarianceAction
    likelihood: CorrelatedGaussianPlan
    observation_id: str = eqx.field(static=True)
    covariance_semantics: str = eqx.field(static=True)

    def __init__(
        self,
        construct,
        nucleotide_keys,
        reactivity,
        standard_deviation,
        *,
        reagent,
        condition,
        replicate_id,
        preprocessing,
        source,
        observed=None,
        covariance_lower=None,
        requested_use=None,
    ):
        if not isinstance(source, ReferenceArtifactManifest):
            raise TypeError(
                "Chemical mapping requires source rights and uncertainty provenance."
            )
        source.require_rights(**({} if requested_use is None else requested_use))
        keys = tuple(nucleotide_keys)
        values, errors = (
            np.asarray(reactivity, float),
            np.asarray(standard_deviation, float),
        )
        mask = (
            np.ones(values.shape, bool)
            if observed is None
            else np.asarray(observed, bool)
        )
        if (
            values.ndim != 1
            or errors.shape != values.shape
            or mask.shape != values.shape
            or len(keys) != values.size
        ):
            raise ValueError("Measurement, uncertainty, mapping and mask must align.")
        if (
            not np.any(mask)
            or np.any(~np.isfinite(values[mask]))
            or np.any(~np.isfinite(errors[mask]))
            or np.any(errors[mask] <= 0)
        ):
            raise ValueError(
                "Observed reactivity requires finite values and positive measured SD."
            )
        if len(set(keys)) != len(keys) or any(
            key not in construct.nucleotide_keys for key in keys
        ):
            raise ValueError(
                "Assay nucleotide mapping must be unique and within its construct."
            )
        if (
            not reagent
            or not replicate_id
            or not isinstance(condition, ChemicalMappingCondition)
        ):
            raise ValueError(
                "Reagent, condition and replicate identity must be explicit."
            )
        self.construct, self.nucleotide_keys = construct, keys
        self.reagent, self.condition, self.replicate_id = reagent, condition, replicate_id
        self.preprocessing, self.source = tuple(preprocessing), source
        self.reactivity, self.standard_deviation, self.observed = (
            jnp.asarray(values),
            jnp.asarray(errors),
            jnp.asarray(mask),
        )
        selected = np.flatnonzero(mask)
        self.observed_indices = jnp.asarray(selected, dtype=jnp.int64)
        layout = CoordinateLayout(
            tuple(
                f"{replicate_id}:{keys[i].strand_id}:{keys[i].position}" for i in selected
            )
        )
        lower = (
            np.diag(errors[selected])
            if covariance_lower is None
            else np.asarray(covariance_lower, float)
        )
        if lower.shape != (selected.size, selected.size) or not np.allclose(
            np.sum(lower**2, axis=1), errors[selected] ** 2, rtol=1e-7, atol=0
        ):
            raise ValueError(
                "Observed covariance diagonal must agree with supplied measured SD."
            )
        self.covariance = CholeskyCovarianceAction(lower, layout)
        self.likelihood = CorrelatedGaussianPlan(
            values[selected],
            LinearObservationPlan(np.eye(selected.size), layout, layout),
            self.covariance,
        )
        self.covariance_semantics = (
            "observed-full-covariance"
            if covariance_lower is not None
            else "diagonal-SD-approximation"
        )
        self.observation_id = canonical_fingerprint(
            {
                "construct": construct.fingerprint(),
                "mapping": [(k.strand_id, k.position) for k in keys],
                "reagent": reagent,
                "condition": condition.condition_id,
                "annotations": condition.annotations,
                "exposure": condition.exposure,
                "exposure_unit": condition.exposure_unit.unit_id,
                "replicate": replicate_id,
                "preprocessing": self.preprocessing,
                "source": source.manifest_id,
                "likelihood": self.likelihood.plan_id,
            }
        )

    def score(self, prediction):
        values = jnp.asarray(prediction)
        if values.shape != self.reactivity.shape:
            raise ValueError("Prediction must retain original observation support.")
        return self.likelihood.evaluate(
            TheoryVector(
                values[self.observed_indices], self.covariance.layout, self.observation_id
            )
        )

    def residual(self, prediction):
        return self.covariance.whiten(
            (jnp.asarray(prediction) - self.reactivity)[self.observed_indices]
        )


class ChemicalMappingFit(StrictModule):
    optimization: object
    predictions: tuple
    scores: tuple
    design_rank: Array
    singular_values: Array
    identifiable: Array
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)


class AccessibilityReactivityModel(StrictModule):
    """Affine processed-reactivity law with explicit nuisance sharing.

    Each condition/replicate has an explicitly selected baseline group, a shared
    accessibility slope and declared numerical condition effects. Accessibility
    is a hypothesis feature in [0,1], NOT pairing inferred from reactivity. The
    model is a calibrated observation approximation, not chemical-rate kinetics.
    It fits one reagent/preprocessing profile at a time. Missing measurements
    never become zero accessibility, paired labels or extra fit equations.
    """

    observations: tuple[ChemicalMappingObservation, ...]
    design: tuple
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        observations,
        accessibility,
        *,
        baseline_groups,
        condition_features,
        condition_names=(),
    ):
        observations, accessibility, groups = (
            tuple(observations),
            tuple(accessibility),
            tuple(baseline_groups),
        )
        features = np.asarray(condition_features, float)
        if (
            not observations
            or len(observations) != len(accessibility)
            or len(groups) != len(observations)
            or features.shape != (len(observations), len(condition_names))
            or not np.all(np.isfinite(features))
            or any(not label for label in groups)
        ):
            raise ValueError(
                "Hypothesis features, conditions and baseline groups must align with observations."
            )
        profiles = {(obs.reagent, obs.preprocessing) for obs in observations}
        if len(profiles) != 1:
            raise ValueError(
                "Distinct reagents/preprocessing need separately named calibration models."
            )
        labels = tuple(dict.fromkeys(groups))
        names = (
            tuple(f"baseline:{label}" for label in labels)
            + ("accessibility-slope",)
            + tuple(f"condition:{name}" for name in condition_names)
        )
        if len(set(names)) != len(names):
            raise ValueError("Condition feature names must be unique.")
        matrices = []
        for obs, signal, group, feature in zip(
            observations, accessibility, groups, features, strict=True
        ):
            signal = np.asarray(signal, float)
            if (
                signal.shape != obs.reactivity.shape
                or np.any(~np.isfinite(signal))
                or np.any((signal < 0) | (signal > 1))
            ):
                raise ValueError(
                    "Structural accessibility must explicitly cover every selected measurement row in [0,1]."
                )
            matrix = np.zeros((signal.size, len(names)))
            matrix[:, labels.index(group)] = 1
            matrix[:, len(labels)] = signal
            matrix[:, len(labels) + 1 :] = feature
            matrices.append(jnp.asarray(matrix))
        self.observations, self.design, self.parameter_names = (
            observations,
            tuple(matrices),
            names,
        )
        self.model_id = canonical_fingerprint(
            {
                "kind": "affine-accessibility-reactivity",
                "observations": [o.observation_id for o in observations],
                "groups": groups,
                "condition_names": condition_names,
                "design": [np.asarray(m).tolist() for m in matrices],
            }
        )

    def predict(self, parameters):
        return tuple(contract("ij,j->i", matrix, parameters) for matrix in self.design)

    def fit(self, initial_parameters=None, *, termination=None, requested_use=None):
        use = {} if requested_use is None else dict(requested_use)
        use["training_use"] = True
        for observation in self.observations:
            observation.source.require_rights(**use)
        initial = (
            jnp.zeros((len(self.parameter_names),))
            if initial_parameters is None
            else jnp.asarray(initial_parameters)
        )
        if initial.shape != (len(self.parameter_names),):
            raise ValueError("Initial observation-model parameters have incorrect shape.")

        def residual(parameters, args):
            del args
            return tuple(
                obs.residual(prediction)
                for obs, prediction in zip(
                    self.observations, self.predict(parameters), strict=True
                )
            )

        result = least_squares(
            residual, initial, method=LevenbergMarquardt(), termination=termination
        )
        predictions = self.predict(result.parameters)
        whitened = jnp.concatenate(
            tuple(
                jnp.stack(
                    tuple(
                        obs.covariance.whiten(matrix[obs.observed_indices, i])
                        for i in range(matrix.shape[1])
                    ),
                    axis=-1,
                )
                for obs, matrix in zip(self.observations, self.design, strict=True)
            ),
            axis=0,
        )
        factor = la.factorize(
            la.DenseLinearOperator(whitened), la.FactorizationPolicy("svd")
        )
        rank = factor.rank()
        return ChemicalMappingFit(
            result,
            predictions,
            tuple(
                obs.score(prediction)
                for obs, prediction in zip(self.observations, predictions, strict=True)
            ),
            rank,
            factor.singular_values(),
            rank == len(self.parameter_names),
            self.parameter_names,
            self.model_id,
            tuple(obs.source.manifest_id for obs in self.observations),
        )


__all__ = [
    "ChemicalMappingCondition",
    "ChemicalMappingObservation",
    "ChemicalMappingFit",
    "AccessibilityReactivityModel",
]
