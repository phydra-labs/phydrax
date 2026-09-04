#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-observation multimodal skeletal posterior assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....uq import (
    FixedObservationLikelihood,
    GaussianLikelihood,
    ParameterSpace,
    PosteriorProblem,
)
from .._quantities import (
    skeletal_muscle_quantity,
    SkeletalMuscleQuantitySpec,
)


class _StaticMaskProjection(StrictModule):
    prediction_fn: Callable[[PyTree[Any]], ArrayLike] = eqx.field(static=True)
    observation_shape: tuple[int, ...] = eqx.field(static=True)
    valid_mask: tuple[bool, ...] = eqx.field(static=True)
    active_indices: tuple[int, ...] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)

    def __init__(
        self,
        prediction: Callable[[PyTree[Any]], ArrayLike],
        observation_shape: tuple[int, ...],
        valid_mask: tuple[bool, ...],
        active_indices: tuple[int, ...],
        dtype: np.dtype,
        /,
    ):
        self.prediction_fn = prediction
        self.observation_shape = observation_shape
        self.valid_mask = valid_mask
        self.active_indices = active_indices
        self.dtype = dtype

    def __call__(self, parameters: PyTree[Any], /) -> Array:
        predicted = jnp.asarray(self.prediction_fn(parameters))
        if jnp.issubdtype(predicted.dtype, jnp.complexfloating):
            raise TypeError("Skeletal observation predictions must be real.")
        predicted = jnp.asarray(predicted, dtype=self.dtype)
        if predicted.shape != self.observation_shape:
            raise ValueError(
                "Skeletal observation prediction has shape "
                f"{predicted.shape}; expected {self.observation_shape}."
            )
        mask = jnp.asarray(self.valid_mask, dtype=bool).reshape(
            self.observation_shape
        )
        sanitized = jnp.where(
            mask,
            predicted,
            jnp.zeros((), dtype=predicted.dtype),
        )
        return jnp.take(
            sanitized.reshape((-1,)),
            jnp.asarray(self.active_indices, dtype=jnp.int32),
        )


class _ChannelPredictionMap(StrictModule):
    channel_ids: tuple[str, ...] = eqx.field(static=True)
    prediction_fns: tuple[Callable[[PyTree[Any]], ArrayLike], ...] = eqx.field(
        static=True
    )

    def __init__(
        self,
        channel_ids: tuple[str, ...],
        prediction_fns: tuple[Callable[[PyTree[Any]], ArrayLike], ...],
        /,
    ):
        self.channel_ids = channel_ids
        self.prediction_fns = prediction_fns

    def __call__(self, parameters: PyTree[Any], /) -> dict[str, ArrayLike]:
        return {
            channel_id: prediction(parameters)
            for channel_id, prediction in zip(
                self.channel_ids,
                self.prediction_fns,
                strict=True,
            )
        }


class SkeletalObservationChannel(StrictModule, NonTrainableState):
    channel_id: str = eqx.field(static=True)
    quantity_id: str = eqx.field(static=True)
    asset_id: str = eqx.field(static=True)
    values: Array
    standard_uncertainty: Array
    valid_mask: Array
    active_indices: tuple[int, ...] = eqx.field(static=True)
    channel_identity: str = eqx.field(static=True)

    def __init__(
        self,
        channel_id: str,
        quantity: str | SkeletalMuscleQuantitySpec,
        asset_id: str,
        values: ArrayLike,
        standard_uncertainty: ArrayLike,
        valid_mask: ArrayLike,
        /,
    ):
        identifiers = tuple(str(value).strip() for value in (channel_id, asset_id))
        if any(not value for value in identifiers):
            raise ValueError("Channel and asset IDs must be nonempty.")
        if isinstance(quantity, str):
            quantity_spec = skeletal_muscle_quantity(quantity)
        elif isinstance(quantity, SkeletalMuscleQuantitySpec):
            quantity_spec = skeletal_muscle_quantity(quantity.name)
            if quantity.quantity_id != quantity_spec.quantity_id:
                raise ValueError(
                    "quantity must match its registered canonical specification."
                )
        else:
            raise TypeError(
                "quantity must be a SkeletalMuscleQuantitySpec or registered name."
            )

        data = jnp.asarray(values)
        if jnp.issubdtype(data.dtype, jnp.complexfloating):
            raise TypeError("Observation values must be real.")
        if not jnp.issubdtype(data.dtype, jnp.floating):
            data = jnp.asarray(data, dtype=float)

        uncertainty_input = jnp.asarray(standard_uncertainty)
        if jnp.issubdtype(uncertainty_input.dtype, jnp.complexfloating):
            raise TypeError("Observation uncertainty must be real.")
        uncertainty = jnp.asarray(uncertainty_input, dtype=data.dtype)
        mask = jnp.asarray(valid_mask, dtype=bool)
        if (
            data.ndim == 0
            or uncertainty.shape not in ((), data.shape)
            or mask.shape != data.shape
        ):
            raise ValueError(
                "Observation data, uncertainty, and mask shapes are incompatible."
            )

        data_host = np.asarray(data)
        mask_host = np.asarray(mask)
        uncertainty_host = np.broadcast_to(np.asarray(uncertainty), data.shape)
        if not np.any(mask_host) or not (
            np.all(np.isfinite(data_host[mask_host]))
            and np.all(np.isfinite(uncertainty_host[mask_host]))
            and np.all(uncertainty_host[mask_host] > 0.0)
        ):
            raise ValueError(
                "Active observations require finite values and positive uncertainty."
            )

        sanitized_data = jnp.asarray(
            np.where(mask_host, data_host, 0.0),
            dtype=data.dtype,
        )
        sanitized_uncertainty = jnp.asarray(
            np.where(mask_host, uncertainty_host, 1.0),
            dtype=data.dtype,
        )
        active_indices = tuple(
            int(index) for index in np.flatnonzero(mask_host.reshape((-1,)))
        )

        self.channel_id, self.asset_id = identifiers
        self.quantity_id = quantity_spec.quantity_id
        self.values = sanitized_data
        self.standard_uncertainty = sanitized_uncertainty
        self.valid_mask = mask
        self.active_indices = active_indices
        self.channel_identity = canonical_fingerprint(
            {
                "kind": "skeletal-observation-channel",
                "channel_id": identifiers[0],
                "quantity_id": quantity_spec.quantity_id,
                "asset_id": identifiers[1],
                "values": array_tree_fingerprint(sanitized_data),
                "uncertainty": array_tree_fingerprint(
                    sanitized_uncertainty
                ),
                "valid_mask": array_tree_fingerprint(mask),
            }
        )

    def _likelihood_term(
        self,
        prediction: Callable[[PyTree[Any]], ArrayLike],
        /,
    ) -> FixedObservationLikelihood:
        mask_values = tuple(
            bool(value) for value in np.asarray(self.valid_mask).reshape((-1,))
        )
        projected_prediction = _StaticMaskProjection(
            prediction,
            tuple(self.values.shape),
            mask_values,
            self.active_indices,
            np.dtype(self.values.dtype),
        )
        indices = jnp.asarray(self.active_indices, dtype=jnp.int32)
        target = jnp.take(self.values.reshape((-1,)), indices)
        scale = jnp.take(self.standard_uncertainty.reshape((-1,)), indices)
        return FixedObservationLikelihood(
            projected_prediction,
            target,
            GaussianLikelihood(scale),
            label=self.channel_id,
        )


class SkeletalMultimodalLikelihoodPlan(StrictModule, NonTrainableState):
    channels: tuple[SkeletalObservationChannel, ...]
    channel_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, channels: Sequence[SkeletalObservationChannel], /):
        values = tuple(channels)
        if not values or any(
            not isinstance(value, SkeletalObservationChannel) for value in values
        ):
            raise TypeError(
                "channels must be a nonempty sequence of skeletal observations."
            )
        ids = tuple(value.channel_id for value in values)
        if len(set(ids)) != len(ids):
            raise ValueError("Observation channel IDs must be unique.")
        self.channels = values
        self.channel_ids = ids
        self.plan_id = canonical_fingerprint(
            {
                "kind": "skeletal-multimodal-fixed-observation-plan",
                "channels": tuple(value.channel_identity for value in values),
            }
        )

    def _prediction_functions(
        self,
        predictions: Mapping[str, Callable[[PyTree[Any]], ArrayLike]],
        /,
    ) -> tuple[Callable[[PyTree[Any]], ArrayLike], ...]:
        if not isinstance(predictions, Mapping):
            raise TypeError("predictions must be a channel-to-callable mapping.")
        missing = tuple(
            channel_id
            for channel_id in self.channel_ids
            if channel_id not in predictions
        )
        unexpected = tuple(
            channel_id
            for channel_id in predictions
            if channel_id not in self.channel_ids
        )
        if missing or unexpected:
            raise ValueError(
                "predictions must contain exactly the planned channel IDs; "
                f"missing={missing}, unexpected={unexpected}."
            )
        functions = tuple(predictions[channel_id] for channel_id in self.channel_ids)
        if any(not callable(function) for function in functions):
            raise TypeError("Every channel prediction must be callable.")
        return functions

    def likelihood_terms(
        self,
        predictions: Mapping[str, Callable[[PyTree[Any]], ArrayLike]],
        /,
    ) -> tuple[FixedObservationLikelihood, ...]:
        """Assemble core normalized likelihood terms in stable channel order."""
        functions = self._prediction_functions(predictions)
        return tuple(
            channel._likelihood_term(prediction)
            for channel, prediction in zip(
                self.channels,
                functions,
                strict=True,
            )
        )

    def posterior(
        self,
        parameter_space: ParameterSpace,
        predictions: Mapping[str, Callable[[PyTree[Any]], ArrayLike]],
        /,
    ) -> PosteriorProblem:
        """Assemble the core posterior that owns both declared prior and terms."""
        functions = self._prediction_functions(predictions)
        terms = tuple(
            channel._likelihood_term(prediction)
            for channel, prediction in zip(
                self.channels,
                functions,
                strict=True,
            )
        )
        return PosteriorProblem.from_terms(
            parameter_space,
            terms,
            predict=_ChannelPredictionMap(self.channel_ids, functions),
        )


__all__ = [
    "SkeletalMultimodalLikelihoodPlan",
    "SkeletalObservationChannel",
]
