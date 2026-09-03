#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Prepared multimodal likelihoods for cardiovascular inverse problems."""

from __future__ import annotations

from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._likelihoods import AbstractLikelihood, GaussianLikelihood
from ...._probability import AbstractProbabilityLaw
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....linalg import (
    DenseLinearOperator,
    FactorizationPolicy,
    factorize,
    OperatorProperties,
)
from ....observation import (
    CholeskyCovarianceAction,
    CoordinateLayout,
    PrecisionCovarianceAction,
)
from ..observations._metadata import ObservationRecord


CovarianceAction = PrecisionCovarianceAction | CholeskyCovarianceAction


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace.")
    return value


def _optional_text(value: str | None, name: str, /) -> str | None:
    return None if value is None else _text(value, name)


def _covariance_action(value: object, /) -> bool:
    return isinstance(value, (PrecisionCovarianceAction, CholeskyCovarianceAction))


class ModalityObservation(StrictModule, NonTrainableState):
    """One immutable masked observation vector with normalized metadata identity."""

    values: Array
    valid_mask: Array
    record_id: str = eqx.field(static=True)
    modality: str = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    frame_id: str | None = eqx.field(static=True)
    timebase_id: str | None = eqx.field(static=True)
    asset_id: str | None = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)

    def __init__(
        self,
        record_id: str,
        modality: str,
        values: ArrayLike,
        valid_mask: ArrayLike,
        quantity: str,
        unit: str,
        /,
        *,
        frame_id: str | None = None,
        timebase_id: str | None = None,
        asset_id: str | None = None,
    ):
        record = _text(record_id, "record_id")
        modality_ = _text(modality, "modality")
        quantity_ = _text(quantity, "quantity")
        unit_ = _text(unit, "unit")
        frame = _optional_text(frame_id, "frame_id")
        timebase = _optional_text(timebase_id, "timebase_id")
        asset = _optional_text(asset_id, "asset_id")
        values_ = jax.lax.stop_gradient(jnp.asarray(values))
        mask = jax.lax.stop_gradient(jnp.asarray(valid_mask, dtype=bool))
        if values_.shape != mask.shape or values_.size == 0:
            raise ValueError(
                "Observation values and valid_mask must share a non-empty shape."
            )
        if jnp.issubdtype(values_.dtype, jnp.complexfloating):
            raise TypeError(
                "Cardiovascular personalization observations must be real-valued."
            )
        values_ = values_.astype(jnp.result_type(values_, jnp.float32))
        if not bool(jnp.any(mask)):
            raise ValueError(
                "An observation channel must contain at least one valid sample."
            )
        if bool(jnp.any(mask & ~jnp.isfinite(values_))):
            raise ValueError("Valid observation samples must be finite.")
        shape = tuple(int(size) for size in values_.shape)
        self.values = values_
        self.valid_mask = mask
        self.record_id = record
        self.modality = modality_
        self.quantity = quantity_
        self.unit = unit_
        self.frame_id = frame
        self.timebase_id = timebase
        self.asset_id = asset
        self.value_shape = shape
        self.observation_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-modality-observation",
                "record_id": record,
                "modality": modality_,
                "quantity": quantity_,
                "unit": unit_,
                "frame_id": frame,
                "timebase_id": timebase,
                "asset_id": asset,
                "values": array_tree_fingerprint(values_),
                "valid_mask": array_tree_fingerprint(mask),
            }
        )

    @classmethod
    def from_record(cls, record: ObservationRecord, /) -> "ModalityObservation":
        if not isinstance(record, ObservationRecord):
            raise TypeError("record must be an ObservationRecord.")
        return cls(
            record.record_id,
            record.modality,
            record.values,
            record.valid_mask,
            record.quantity,
            record.unit,
            frame_id=record.frame_id,
            timebase_id=record.timebase_id,
            asset_id=record.asset_id,
        )

    @property
    def size(self) -> int:
        return prod(self.value_shape)


class ReferenceGauge(StrictModule, NonTrainableState):
    """Remove one observed reference sample by differencing and dropping it."""

    reference_index: int = eqx.field(static=True)
    gauge_id: str = eqx.field(static=True)

    def __init__(self, reference_index: int, /):
        index = int(reference_index)
        if index < 0:
            raise ValueError("reference_index must be non-negative.")
        self.reference_index = index
        self.gauge_id = canonical_fingerprint(
            {"kind": "cardiovascular-reference-gauge", "reference_index": index}
        )


class LinearNuisanceModel(StrictModule, NonTrainableState):
    """Additive nuisance basis with a native probability-law prior."""

    basis: Array
    prior: AbstractProbabilityLaw
    nuisance_count: int = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(self, basis: ArrayLike, prior: AbstractProbabilityLaw, /):
        matrix = jax.lax.stop_gradient(jnp.asarray(basis, dtype=float))
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("Nuisance basis must be a non-empty matrix.")
        if bool(jnp.any(~jnp.isfinite(matrix))):
            raise ValueError("Nuisance basis must be finite.")
        if not isinstance(prior, AbstractProbabilityLaw):
            raise TypeError("prior must implement AbstractProbabilityLaw.")
        count = int(matrix.shape[1])
        prior_shape = tuple(prior.batch_shape) + tuple(prior.event_shape)
        if prior_shape and prior_shape != (count,):
            raise ValueError("Nuisance prior shape must match the nuisance basis width.")
        if prior.density_measure_kind != "lebesgue":
            raise ValueError("Nuisance parameters require a Lebesgue-density prior.")
        self.basis = matrix
        self.prior = prior
        self.nuisance_count = count
        self.model_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-linear-nuisance",
                "basis": array_tree_fingerprint(matrix),
                "prior_type": type(prior).__qualname__,
                "prior_arrays": array_tree_fingerprint(prior),
            }
        )

    def correction(self, values: ArrayLike, /) -> Array:
        nuisance = jnp.asarray(values, dtype=self.basis.dtype)
        if nuisance.shape != (self.nuisance_count,):
            raise ValueError(f"Nuisance values must have shape {(self.nuisance_count,)}.")
        return contract("ik,k->i", self.basis, nuisance)

    def log_prior(self, values: ArrayLike, /) -> Array:
        nuisance = jnp.asarray(values, dtype=self.basis.dtype)
        if nuisance.shape != (self.nuisance_count,):
            raise ValueError(f"Nuisance values must have shape {(self.nuisance_count,)}.")
        valid = jnp.all(self.prior.contains(nuisance))
        return jnp.where(valid, jnp.sum(self.prior.log_prob(nuisance)), -jnp.inf)


class GaussianModelDiscrepancy(StrictModule, NonTrainableState):
    """Declared additive model bias and low-rank Gaussian discrepancy covariance."""

    mean: Array
    covariance_factor: Array
    discrepancy_id: str = eqx.field(static=True)

    def __init__(
        self,
        mean: ArrayLike,
        covariance_factor: ArrayLike | None = None,
        /,
    ):
        mean_ = jax.lax.stop_gradient(jnp.asarray(mean, dtype=float).reshape(-1))
        factor = (
            jnp.zeros((mean_.size, 0), dtype=mean_.dtype)
            if covariance_factor is None
            else jax.lax.stop_gradient(jnp.asarray(covariance_factor, dtype=mean_.dtype))
        )
        if factor.ndim != 2 or factor.shape[0] != mean_.size:
            raise ValueError(
                "Discrepancy covariance_factor must have one row per sample."
            )
        if bool(jnp.any(~jnp.isfinite(mean_))) or bool(jnp.any(~jnp.isfinite(factor))):
            raise ValueError("Model discrepancy inputs must be finite.")
        self.mean = mean_
        self.covariance_factor = factor
        self.discrepancy_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-gaussian-discrepancy",
                "mean": array_tree_fingerprint(mean_),
                "covariance_factor": array_tree_fingerprint(factor),
            }
        )

    @property
    def size(self) -> int:
        return int(self.mean.size)

    @property
    def stochastic(self) -> bool:
        return int(self.covariance_factor.shape[1]) > 0

    @property
    def covariance(self) -> Array:
        return contract("ik,jk->ij", self.covariance_factor, self.covariance_factor)


def _selection_and_gauge(
    observation: ModalityObservation,
    gauge: ReferenceGauge | None,
    /,
) -> tuple[Array, Array]:
    active_host = np.flatnonzero(np.asarray(observation.valid_mask).reshape(-1))
    active = jnp.asarray(active_host, dtype=jnp.int32)
    count = int(active_host.size)
    if gauge is None:
        return active, jnp.eye(count, dtype=observation.values.dtype)
    if not isinstance(gauge, ReferenceGauge):
        raise TypeError("gauge must be a ReferenceGauge or None.")
    if gauge.reference_index >= observation.size:
        raise ValueError("Gauge reference_index lies outside the observation.")
    positions = np.flatnonzero(active_host == gauge.reference_index)
    if positions.size != 1:
        raise ValueError(
            "Gauge reference sample must be valid under the observation mask."
        )
    if count < 2:
        raise ValueError("A reference gauge requires at least two valid samples.")
    reference_position = int(positions[0])
    rows = []
    for position in range(count):
        if position != reference_position:
            row = np.zeros(count, dtype=np.asarray(observation.values).dtype)
            row[position] = 1.0
            row[reference_position] = -1.0
            rows.append(row)
    return active, jnp.asarray(np.stack(rows))


class ModalityLikelihoodEvaluation(StrictModule):
    """One channel score and fail-closed numerical evidence."""

    residual: Array
    log_likelihood: Array
    log_nuisance_prior: Array
    finite: Array
    nuisance_supported: Array
    successful: Array
    channel_id: str = eqx.field(static=True)


class ModalityLikelihoodChannel(StrictModule, NonTrainableState):
    """Prepared masked/gauged channel using an existing likelihood or covariance action."""

    observation: ModalityObservation
    likelihood: AbstractLikelihood | None
    covariance: CovarianceAction | None
    gauge: ReferenceGauge | None
    nuisance: LinearNuisanceModel | None
    discrepancy: GaussianModelDiscrepancy
    active_indices: Array
    gauge_matrix: Array
    gauged_target: Array
    output_size: int = eqx.field(static=True)
    channel_id: str = eqx.field(static=True)

    def __init__(
        self,
        observation: ModalityObservation,
        /,
        *,
        likelihood: AbstractLikelihood | None = None,
        covariance: CovarianceAction | None = None,
        gauge: ReferenceGauge | None = None,
        nuisance: LinearNuisanceModel | None = None,
        discrepancy: GaussianModelDiscrepancy | None = None,
        covariance_includes_discrepancy: bool = False,
    ):
        if not isinstance(observation, ModalityObservation):
            raise TypeError("observation must be a ModalityObservation.")
        if (likelihood is None) == (covariance is None):
            raise ValueError(
                "Provide exactly one existing likelihood or covariance action."
            )
        if likelihood is not None and not isinstance(likelihood, AbstractLikelihood):
            raise TypeError("likelihood must implement AbstractLikelihood.")
        if covariance is not None and not _covariance_action(covariance):
            raise TypeError("covariance must be a native covariance action.")
        if nuisance is not None and not isinstance(nuisance, LinearNuisanceModel):
            raise TypeError("nuisance must be a LinearNuisanceModel or None.")
        discrepancy_ = (
            GaussianModelDiscrepancy(
                jnp.zeros(observation.size, dtype=observation.values.dtype)
            )
            if discrepancy is None
            else discrepancy
        )
        if not isinstance(discrepancy_, GaussianModelDiscrepancy):
            raise TypeError("discrepancy must be a GaussianModelDiscrepancy or None.")
        if discrepancy_.size != observation.size:
            raise ValueError("Model discrepancy size must match the observation size.")
        if nuisance is not None and nuisance.basis.shape[0] != observation.size:
            raise ValueError("Nuisance basis row count must match the observation size.")
        active, gauge_matrix = _selection_and_gauge(observation, gauge)
        output_size = int(gauge_matrix.shape[0])
        if covariance is not None and covariance.layout.size != output_size:
            raise ValueError(
                "Covariance layout size must match masked/gauged observations."
            )
        if gauge is not None and isinstance(likelihood, GaussianLikelihood):
            raise ValueError(
                "A gauged Gaussian must use correlated_gaussian() so the gauge "
                "projection is applied to the complete covariance."
            )
        effective_likelihood = likelihood
        if isinstance(likelihood, GaussianLikelihood):
            scale = jnp.asarray(likelihood.scale)
            if scale.shape == observation.value_shape or scale.shape == (
                observation.size,
            ):
                base_scale = scale.reshape(-1)[active]
            else:
                base_scale = jnp.broadcast_to(scale, (output_size,))
            effective_likelihood = GaussianLikelihood(base_scale)
        if discrepancy_.stochastic and covariance is None:
            if not isinstance(effective_likelihood, GaussianLikelihood):
                raise ValueError(
                    "Stochastic discrepancy requires GaussianLikelihood or correlated_gaussian()."
                )
            discrepancy_covariance = discrepancy_.covariance[
                active[:, None], active[None, :]
            ]
            off_diagonal = discrepancy_covariance - jnp.diag(
                jnp.diag(discrepancy_covariance)
            )
            if bool(jnp.any(jnp.abs(off_diagonal) > 1.0e-10)):
                raise ValueError(
                    "Correlated low-rank discrepancy requires correlated_gaussian(); "
                    "an elementwise likelihood cannot discard off-diagonal covariance."
                )
            variance = jnp.diag(discrepancy_covariance)
            effective_likelihood = GaussianLikelihood(
                jnp.sqrt(effective_likelihood.scale**2 + variance)
            )
        if (
            discrepancy_.stochastic
            and covariance is not None
            and not covariance_includes_discrepancy
        ):
            raise ValueError(
                "Correlated covariance must explicitly include stochastic model discrepancy."
            )
        target = contract("oi,i->o", gauge_matrix, observation.values.reshape(-1)[active])
        self.observation = observation
        self.likelihood = effective_likelihood
        self.covariance = covariance
        self.gauge = gauge
        self.nuisance = nuisance
        self.discrepancy = discrepancy_
        self.active_indices = active
        self.gauge_matrix = gauge_matrix
        self.gauged_target = target
        self.output_size = output_size
        self.channel_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-modality-likelihood",
                "observation": observation.observation_id,
                "likelihood_type": None
                if effective_likelihood is None
                else type(effective_likelihood).__qualname__,
                "likelihood_arrays": None
                if effective_likelihood is None
                else array_tree_fingerprint(effective_likelihood),
                "covariance": None if covariance is None else covariance.action_id,
                "gauge": None if gauge is None else gauge.gauge_id,
                "nuisance": None if nuisance is None else nuisance.model_id,
                "discrepancy": discrepancy_.discrepancy_id,
            }
        )

    @classmethod
    def correlated_gaussian(
        cls,
        observation: ModalityObservation,
        measurement_covariance: ArrayLike,
        /,
        *,
        gauge: ReferenceGauge | None = None,
        nuisance: LinearNuisanceModel | None = None,
        discrepancy: GaussianModelDiscrepancy | None = None,
    ) -> "ModalityLikelihoodChannel":
        """Prepare total covariance with the native factorization substrate."""

        if not isinstance(observation, ModalityObservation):
            raise TypeError("observation must be a ModalityObservation.")
        discrepancy_ = (
            GaussianModelDiscrepancy(
                jnp.zeros(observation.size, dtype=observation.values.dtype)
            )
            if discrepancy is None
            else discrepancy
        )
        if not isinstance(discrepancy_, GaussianModelDiscrepancy):
            raise TypeError("discrepancy must be a GaussianModelDiscrepancy or None.")
        if discrepancy_.size != observation.size:
            raise ValueError("Model discrepancy size must match the observation size.")
        measurement = jnp.asarray(measurement_covariance, dtype=observation.values.dtype)
        if measurement.shape != (observation.size, observation.size):
            raise ValueError(
                "measurement_covariance must cover the complete observation."
            )
        if bool(jnp.any(~jnp.isfinite(measurement))) or bool(
            jnp.any(jnp.abs(measurement - measurement.T) > 1.0e-10)
        ):
            raise ValueError("Measurement covariance must be finite and symmetric.")
        active, gauge_matrix = _selection_and_gauge(observation, gauge)
        total = measurement + discrepancy_.covariance
        selected = total[active[:, None], active[None, :]]
        reduced = contract("ai,ij,bj->ab", gauge_matrix, selected, gauge_matrix)
        properties = OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={"self_adjoint": "construction", "positive_definite": "asserted"},
        )
        factorization = factorize(
            DenseLinearOperator(reduced, properties=properties),
            FactorizationPolicy("cholesky"),
        )
        inverse = factorization.materialize_inverse()
        if not bool(inverse.successful):
            raise ValueError("Total masked/gauged covariance is not positive definite.")
        labels = tuple(
            f"{observation.modality}:{index}" for index in range(reduced.shape[0])
        )
        layout = CoordinateLayout(labels)
        covariance = PrecisionCovarianceAction(
            inverse.value,
            factorization.log_abs_determinant(),
            layout,
        )
        return cls(
            observation,
            covariance=covariance,
            gauge=gauge,
            nuisance=nuisance,
            discrepancy=discrepancy_,
            covariance_includes_discrepancy=True,
        )

    def evaluate(
        self,
        prediction: ArrayLike,
        /,
        *,
        nuisance_values: ArrayLike | None = None,
    ) -> ModalityLikelihoodEvaluation:
        predicted = jnp.asarray(prediction, dtype=self.observation.values.dtype)
        if predicted.shape != self.observation.value_shape:
            raise ValueError(
                f"Prediction for {self.observation.modality!r} must have shape "
                f"{self.observation.value_shape}; got {predicted.shape}."
            )
        flattened = predicted.reshape(-1) + self.discrepancy.mean
        nuisance_prior = jnp.asarray(0.0, dtype=flattened.dtype)
        nuisance_supported = jnp.asarray(True)
        if self.nuisance is None:
            if nuisance_values is not None:
                raise ValueError(
                    "nuisance_values were supplied to a channel without nuisance."
                )
        else:
            if nuisance_values is None:
                raise ValueError("This channel requires nuisance_values.")
            correction = self.nuisance.correction(nuisance_values)
            flattened = flattened + correction
            nuisance_prior = self.nuisance.log_prior(nuisance_values)
            nuisance_supported = jnp.isfinite(nuisance_prior)
        location = contract("oi,i->o", self.gauge_matrix, flattened[self.active_indices])
        residual = self.gauged_target - location
        if self.likelihood is not None:
            log_likelihood = jnp.sum(
                self.likelihood.log_prob(location, self.gauged_target)
            )
        else:
            if self.covariance is None:
                raise RuntimeError(
                    "Prepared likelihood channel has no density implementation."
                )
            quadratic = self.covariance.quadratic(residual)
            size = jnp.asarray(self.output_size, dtype=residual.dtype)
            log_likelihood = -0.5 * (
                quadratic
                + self.covariance.logdet_covariance
                + size * jnp.log(jnp.asarray(2.0 * jnp.pi, dtype=residual.dtype))
            )
        finite = (
            jnp.all(jnp.isfinite(location))
            & jnp.all(jnp.isfinite(residual))
            & jnp.isfinite(log_likelihood)
            & jnp.isfinite(nuisance_prior)
        )
        successful = finite & nuisance_supported
        return ModalityLikelihoodEvaluation(
            residual=residual,
            log_likelihood=jnp.where(successful, log_likelihood, -jnp.inf),
            log_nuisance_prior=jnp.where(successful, nuisance_prior, -jnp.inf),
            finite=finite,
            nuisance_supported=nuisance_supported,
            successful=successful,
            channel_id=self.channel_id,
        )


class MultimodalLikelihoodResult(StrictModule):
    """Stable-order per-modality scores and aggregate acceptance evidence."""

    channel_results: tuple[ModalityLikelihoodEvaluation, ...]
    log_likelihood: Array
    log_nuisance_prior: Array
    log_density: Array
    finite: Array
    successful: Array
    runtime_id: str = eqx.field(static=True)


class MultimodalLikelihoodPlan(StrictModule, NonTrainableState):
    """Immutable modality composition; hold-out plans are explicit new identities."""

    channels: tuple[ModalityLikelihoodChannel, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(self, channels: Sequence[ModalityLikelihoodChannel], /):
        resolved = tuple(channels)
        if not resolved:
            raise ValueError("MultimodalLikelihoodPlan requires at least one channel.")
        if any(
            not isinstance(channel, ModalityLikelihoodChannel) for channel in resolved
        ):
            raise TypeError("Likelihood plans contain ModalityLikelihoodChannel values.")
        modalities = tuple(channel.observation.modality for channel in resolved)
        if len(modalities) != len(set(modalities)):
            raise ValueError("Each modality may occur at most once in a likelihood plan.")
        self.channels = resolved
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-multimodal-likelihood-plan",
                "channels": [channel.channel_id for channel in resolved],
            }
        )

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(channel.observation.modality for channel in self.channels)

    def held_out(self, modalities: Sequence[str], /) -> "MultimodalLikelihoodPlan":
        held = frozenset(_text(value, "held-out modality") for value in modalities)
        unknown = held - frozenset(self.modalities)
        if unknown:
            raise ValueError(f"Unknown held-out modalities: {sorted(unknown)}.")
        retained = tuple(
            channel
            for channel in self.channels
            if channel.observation.modality not in held
        )
        if not retained:
            raise ValueError(
                "A held-out likelihood plan must retain at least one modality."
            )
        return MultimodalLikelihoodPlan(retained)

    def prepare(self, /) -> "PreparedMultimodalLikelihood":
        return PreparedMultimodalLikelihood(self)


class PreparedMultimodalLikelihood(StrictModule, NonTrainableState):
    """Fixed-shape runtime for one multimodal likelihood plan."""

    plan: MultimodalLikelihoodPlan
    runtime_id: str = eqx.field(static=True)

    def __init__(self, plan: MultimodalLikelihoodPlan, /):
        if not isinstance(plan, MultimodalLikelihoodPlan):
            raise TypeError("plan must be a MultimodalLikelihoodPlan.")
        self.plan = plan
        self.runtime_id = canonical_fingerprint(
            {"kind": "prepared-cardiovascular-likelihood", "plan": plan.plan_id}
        )

    def evaluate(
        self,
        predictions: Sequence[ArrayLike],
        /,
        *,
        nuisance_values: Sequence[ArrayLike | None] | None = None,
    ) -> MultimodalLikelihoodResult:
        values = tuple(predictions)
        if len(values) != len(self.plan.channels):
            raise ValueError("Predictions must match likelihood channel count and order.")
        nuisances = (
            (None,) * len(values) if nuisance_values is None else tuple(nuisance_values)
        )
        if len(nuisances) != len(values):
            raise ValueError("nuisance_values must match likelihood channel count.")
        results = tuple(
            channel.evaluate(prediction, nuisance_values=nuisance)
            for channel, prediction, nuisance in zip(
                self.plan.channels, values, nuisances, strict=True
            )
        )
        log_likelihood = sum(
            (result.log_likelihood for result in results),
            start=jnp.asarray(0.0),
        )
        log_nuisance = sum(
            (result.log_nuisance_prior for result in results),
            start=jnp.asarray(0.0),
        )
        finite = jnp.all(jnp.stack(tuple(result.finite for result in results)))
        successful = jnp.all(jnp.stack(tuple(result.successful for result in results)))
        return MultimodalLikelihoodResult(
            channel_results=results,
            log_likelihood=jnp.where(successful, log_likelihood, -jnp.inf),
            log_nuisance_prior=jnp.where(successful, log_nuisance, -jnp.inf),
            log_density=jnp.where(successful, log_likelihood + log_nuisance, -jnp.inf),
            finite=finite,
            successful=successful,
            runtime_id=self.runtime_id,
        )


__all__ = [
    "GaussianModelDiscrepancy",
    "LinearNuisanceModel",
    "ModalityLikelihoodChannel",
    "ModalityLikelihoodEvaluation",
    "ModalityObservation",
    "MultimodalLikelihoodPlan",
    "MultimodalLikelihoodResult",
    "PreparedMultimodalLikelihood",
    "ReferenceGauge",
]
