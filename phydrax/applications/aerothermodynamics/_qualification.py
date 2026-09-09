#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...qualification import ReferenceArtifactManifest
from ._contracts import AerothermodynamicSupportTuple


class AerothermodynamicValidationCase(StrictModule, NonTrainableState):
    reference: ReferenceArtifactManifest
    reference_values: Array
    uncertainties: Array
    observable_names: tuple[str, ...] = eqx.field(static=True)
    name: str = eqx.field(static=True)
    domain: str = eqx.field(static=True)
    case_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        domain: str,
        reference: ReferenceArtifactManifest,
        observable_names: Sequence[str],
        reference_values: ArrayLike,
        uncertainties: ArrayLike,
        /,
    ):
        name_ = str(name)
        domain_ = str(domain)
        observables = tuple(str(value) for value in observable_names)
        values = np.asarray(reference_values, dtype=float)
        uncertainty = np.asarray(uncertainties, dtype=float)
        if (
            not name_
            or not domain_
            or not isinstance(reference, ReferenceArtifactManifest)
            or not reference.commercial_use_permitted
            or not observables
            or len(set(observables)) != len(observables)
            or values.shape != (len(observables),)
            or uncertainty.shape != values.shape
            or np.any(~np.isfinite(values))
            or np.any(~np.isfinite(uncertainty))
            or np.any(uncertainty <= 0.0)
        ):
            raise ValueError(
                "Validation case identity, rights, values, or uncertainty are invalid."
            )
        self.name = name_
        self.domain = domain_
        self.reference = reference
        self.observable_names = observables
        self.reference_values = jnp.asarray(values)
        self.uncertainties = jnp.asarray(uncertainty)
        self.case_id = canonical_fingerprint(
            {
                "kind": "aerothermodynamic-validation-case",
                "name": name_,
                "domain": domain_,
                "reference": reference.manifest_id,
                "observables": observables,
                "values": array_tree_fingerprint(self.reference_values),
                "uncertainties": array_tree_fingerprint(self.uncertainties),
            }
        )


class ValidationCaseEvidence(StrictModule):
    predictions: Array
    normalized_errors: Array
    maximum_normalized_error: Array
    finite: Array
    successful: Array
    case_id: str = eqx.field(static=True)


class ValidationCampaignEvidence(StrictModule):
    cases: tuple[ValidationCaseEvidence, ...]
    maximum_normalized_error: Array
    finite: Array
    successful: Array
    campaign_id: str = eqx.field(static=True)


class AerothermodynamicValidationCampaignPlan(StrictModule, NonTrainableState):
    support: AerothermodynamicSupportTuple
    cases: tuple[AerothermodynamicValidationCase, ...]
    acceptance_sigma: float = eqx.field(static=True)
    campaign_id: str = eqx.field(static=True)

    def __init__(
        self,
        support: AerothermodynamicSupportTuple,
        cases: Sequence[AerothermodynamicValidationCase],
        /,
        *,
        acceptance_sigma: float = 3.0,
    ):
        cases_ = tuple(cases)
        sigma = float(acceptance_sigma)
        if (
            not isinstance(support, AerothermodynamicSupportTuple)
            or not cases_
            or any(
                not isinstance(value, AerothermodynamicValidationCase) for value in cases_
            )
            or len({value.case_id for value in cases_}) != len(cases_)
            or not np.isfinite(sigma)
            or sigma <= 0.0
        ):
            raise ValueError(
                "Validation campaign support, cases, or threshold are invalid."
            )
        self.support = support
        self.cases = cases_
        self.acceptance_sigma = sigma
        self.campaign_id = canonical_fingerprint(
            {
                "kind": "aerothermodynamic-validation-campaign",
                "support": support.support_id,
                "cases": tuple(value.case_id for value in cases_),
                "acceptance_sigma": sigma,
            }
        )

    def evaluate(
        self, predictions: Mapping[str, Mapping[str, ArrayLike]], /
    ) -> ValidationCampaignEvidence:
        if tuple(predictions) != tuple(value.name for value in self.cases):
            raise ValueError(
                "Validation predictions must follow exact campaign case order."
            )
        evidence = []
        for case in self.cases:
            case_predictions = predictions[case.name]
            if tuple(case_predictions) != case.observable_names:
                raise ValueError("Validation observables must follow exact case order.")
            prediction = jnp.stack(
                tuple(
                    jnp.asarray(case_predictions[name]) for name in case.observable_names
                )
            )
            normalized = jnp.abs(prediction - case.reference_values) / case.uncertainties
            maximum = jnp.max(normalized)
            finite = jnp.all(jnp.isfinite(prediction)) & jnp.all(jnp.isfinite(normalized))
            evidence.append(
                ValidationCaseEvidence(
                    prediction,
                    normalized,
                    maximum,
                    finite,
                    finite & (maximum <= self.acceptance_sigma),
                    case.case_id,
                )
            )
        maximum = jnp.max(
            jnp.stack(tuple(value.maximum_normalized_error for value in evidence))
        )
        finite = jnp.all(jnp.stack(tuple(value.finite for value in evidence)))
        successful = finite & jnp.all(
            jnp.stack(tuple(value.successful for value in evidence))
        )
        return ValidationCampaignEvidence(
            tuple(evidence), maximum, finite, successful, self.campaign_id
        )


__all__ = [
    "AerothermodynamicValidationCampaignPlan",
    "AerothermodynamicValidationCase",
    "ValidationCampaignEvidence",
    "ValidationCaseEvidence",
]
