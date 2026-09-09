#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...uq import CrossGradientPrior, ParameterSpace, PosteriorProblem


class IndependentModalityTerm(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    log_likelihood_fn: Callable[[object], Array] = eqx.field(static=True)
    predict_fn: Callable[[object], object] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)
    likelihood_id: str = eqx.field(static=True)
    prediction_id: str = eqx.field(static=True)
    term_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        log_likelihood: Callable[[object], Array],
        predict: Callable[[object], object],
        observation_id: str,
        /,
        *,
        likelihood_id: str,
        prediction_id: str,
    ):
        name_, observation = str(name).strip(), str(observation_id).strip()
        likelihood_identity = str(likelihood_id).strip()
        prediction_identity = str(prediction_id).strip()
        if (
            not name_
            or not observation
            or not likelihood_identity
            or not prediction_identity
            or not callable(log_likelihood)
            or not callable(predict)
        ):
            raise ValueError(
                "Modality term name/callables and their explicit identities are required."
            )
        self.name, self.log_likelihood_fn, self.predict_fn = (
            name_,
            log_likelihood,
            predict,
        )
        self.observation_id = observation
        self.likelihood_id = likelihood_identity
        self.prediction_id = prediction_identity
        self.term_id = canonical_fingerprint(
            {
                "kind": "independent-modality-term",
                "name": name_,
                "observation": observation,
                "likelihood": likelihood_identity,
                "prediction": prediction_identity,
            }
        )

    def log_likelihood(self, parameters: object, /) -> Array:
        value = jnp.asarray(self.log_likelihood_fn(parameters))
        if value.shape != () or jnp.iscomplexobj(value):
            raise ValueError("Modality log likelihood must be a real scalar.")
        return value

    def predict(self, parameters: object, /) -> object:
        return self.predict_fn(parameters)


class StructuralCrossGradientCoupling(StrictModule, NonTrainableState):
    first_field: Callable[[object], Array] = eqx.field(static=True)
    second_field: Callable[[object], Array] = eqx.field(static=True)
    prior: CrossGradientPrior
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        first_field: Callable[[object], Array],
        second_field: Callable[[object], Array],
        prior: CrossGradientPrior,
        /,
        *,
        first_name: str,
        second_name: str,
        first_map_id: str,
        second_map_id: str,
        prior_id: str,
    ):
        if (
            not callable(first_field)
            or not callable(second_field)
            or not isinstance(prior, CrossGradientPrior)
        ):
            raise TypeError(
                "Structural coupling requires two field maps and CrossGradientPrior."
            )
        names = (str(first_name).strip(), str(second_name).strip())
        if any(not value for value in names) or names[0] == names[1]:
            raise ValueError("Structural field names must be distinct and nonempty.")
        identities = tuple(
            str(value).strip() for value in (first_map_id, second_map_id, prior_id)
        )
        if any(not value for value in identities):
            raise ValueError("Structural field-map and prior identities are required.")
        self.first_field, self.second_field, self.prior = first_field, second_field, prior
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "structural-cross-gradient-coupling",
                "fields": names,
                "field_maps": identities[:2],
                "prior": identities[2],
            }
        )

    def log_density(self, parameters: object, /) -> Array:
        return self.prior.log_prob(
            self.first_field(parameters), self.second_field(parameters)
        )


class PetrophysicalDiscrepancyCoupling(StrictModule, NonTrainableState):
    predicted_property: Callable[[object], Array] = eqx.field(static=True)
    physical_property: Callable[[object], Array] = eqx.field(static=True)
    scale: Array
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        predicted_property: Callable[[object], Array],
        physical_property: Callable[[object], Array],
        scale: Array,
        /,
        *,
        relationship_id: str,
    ):
        if not callable(predicted_property) or not callable(physical_property):
            raise TypeError(
                "Petrophysical coupling requires predicted and physical property maps."
            )
        scale_ = jnp.asarray(scale)
        scale_ = eqx.error_if(
            scale_,
            jnp.any(~jnp.isfinite(scale_)) | jnp.any(scale_ <= 0),
            "Petrophysical discrepancy scale must be positive finite.",
        )
        relationship = str(relationship_id).strip()
        if not relationship:
            raise ValueError("Petrophysical relationship identity is required.")
        self.predicted_property, self.physical_property = (
            predicted_property,
            physical_property,
        )
        self.scale = scale_
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "petrophysical-discrepancy",
                "relationship": relationship,
                "scale": scale_,
            }
        )

    def log_density(self, parameters: object, /) -> Array:
        predicted = self.predicted_property(parameters)
        physical = self.physical_property(parameters)
        residual = (physical - predicted) / self.scale
        return jnp.sum(
            -0.5 * residual**2 - jnp.log(self.scale) - 0.5 * jnp.log(2 * jnp.pi)
        )


class SharedInterfaceCoupling(StrictModule, NonTrainableState):
    first_level_set: Callable[[object], Array] = eqx.field(static=True)
    second_level_set: Callable[[object], Array] = eqx.field(static=True)
    scale: Array
    coupling_id: str = eqx.field(static=True)

    def __init__(
        self,
        first_level_set: Callable[[object], Array],
        second_level_set: Callable[[object], Array],
        scale: Array,
        /,
        *,
        first_map_id: str,
        second_map_id: str,
    ):
        if not callable(first_level_set) or not callable(second_level_set):
            raise TypeError("Shared interface requires two level-set maps.")
        scale_ = jnp.asarray(scale)
        self.scale = eqx.error_if(
            scale_,
            jnp.any(~jnp.isfinite(scale_)) | jnp.any(scale_ <= 0),
            "Shared-interface scale must be positive finite.",
        )
        map_ids = tuple(str(value).strip() for value in (first_map_id, second_map_id))
        if any(not value for value in map_ids) or map_ids[0] == map_ids[1]:
            raise ValueError(
                "Shared-interface map identities must be distinct and nonempty."
            )
        self.first_level_set, self.second_level_set = first_level_set, second_level_set
        self.coupling_id = canonical_fingerprint(
            {
                "kind": "shared-interface-coupling",
                "scale": scale_,
                "field_maps": map_ids,
            }
        )

    def log_density(self, parameters: object, /) -> Array:
        residual = (
            self.first_level_set(parameters) - self.second_level_set(parameters)
        ) / self.scale
        return -0.5 * jnp.real(jnp.vdot(residual, residual))


class MultimodalJointInferencePlan(StrictModule, NonTrainableState):
    parameter_space: ParameterSpace
    modalities: tuple[IndependentModalityTerm, ...]
    couplings: tuple[
        StructuralCrossGradientCoupling
        | PetrophysicalDiscrepancyCoupling
        | SharedInterfaceCoupling,
        ...,
    ]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_space: ParameterSpace,
        modalities: Sequence[IndependentModalityTerm],
        couplings: Sequence[
            StructuralCrossGradientCoupling
            | PetrophysicalDiscrepancyCoupling
            | SharedInterfaceCoupling
        ] = (),
        /,
    ):
        terms = tuple(modalities)
        couplings_ = tuple(couplings)
        if (
            not isinstance(parameter_space, ParameterSpace)
            or not terms
            or any(not isinstance(value, IndependentModalityTerm) for value in terms)
        ):
            raise TypeError("Joint inference requires ParameterSpace and modality terms.")
        if len({value.name for value in terms}) != len(terms) or len(
            {value.observation_id for value in terms}
        ) != len(terms):
            raise ValueError(
                "Independent modality names and observation identities must be "
                "unique; correlated data belong in one term."
            )
        allowed = (
            StructuralCrossGradientCoupling,
            PetrophysicalDiscrepancyCoupling,
            SharedInterfaceCoupling,
        )
        if any(not isinstance(value, allowed) for value in couplings_):
            raise TypeError("Unknown multimodal coupling type.")
        self.parameter_space, self.modalities, self.couplings = (
            parameter_space,
            terms,
            couplings_,
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multimodal-joint-inference",
                "modalities": [value.term_id for value in terms],
                "couplings": [value.coupling_id for value in couplings_],
            }
        )

    def log_likelihood(self, parameters: object, /) -> Array:
        terms = tuple(value.log_likelihood(parameters) for value in self.modalities)
        couplings = tuple(value.log_density(parameters) for value in self.couplings)
        return jnp.sum(jnp.stack((*terms, *couplings)))

    def predictions(self, parameters: object, /) -> tuple[object, ...]:
        return tuple(value.predict(parameters) for value in self.modalities)

    def posterior(self) -> PosteriorProblem:
        def likelihood(parameters):
            return self.log_likelihood(parameters)

        def prediction(parameters):
            return self.predictions(parameters)

        return PosteriorProblem(
            self.parameter_space,
            likelihood,
            predict=prediction,
        )


__all__ = [
    "IndependentModalityTerm",
    "MultimodalJointInferencePlan",
    "PetrophysicalDiscrepancyCoupling",
    "SharedInterfaceCoupling",
    "StructuralCrossGradientCoupling",
]
