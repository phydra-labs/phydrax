#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from phydrax.domain import DomainComponent, DomainFunction, PointSampling

from .._doc import DOC_KEY0
from .._exponential_family import BernoulliFamily, CategoricalFamily
from .._likelihoods import (
    CategoricalExponentialFamilyLikelihood,
    ScalarNaturalExponentialFamilyLikelihood,
)
from ..ml._schema import TargetSchema
from ..ml.metrics import accuracy_score, brier_score, log_loss
from ._likelihood import _AbstractSupervisedLikelihoodTerm
from ._supervised_dataset import SupervisedDatasetBatch


class SupervisedClassificationTerm(_AbstractSupervisedLikelihoodTerm):
    """Train canonical binary or multiclass logits on an empirical dataset."""

    target_schema: TargetSchema
    classification_kind: Literal["binary", "multiclass"] = eqx.field(static=True)
    class_count: int = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        component: DomainComponent,
        targets: ArrayLike,
        target_schema: TargetSchema,
        /,
        *,
        sampling: PointSampling,
        observation_operator: Callable[[DomainFunction], DomainFunction] | None = None,
        sample_mask: ArrayLike | None = None,
        sample_weight: ArrayLike | None = None,
        weight: ArrayLike = 1.0,
        reduction: Literal["mean", "sum"] = "mean",
        indices: ArrayLike | None = None,
        label: str | None = None,
    ):
        if not isinstance(target_schema, TargetSchema):
            raise TypeError("target_schema must be a TargetSchema.")
        kind = target_schema.kind
        if kind == "binary":
            class_count = 2
            likelihood = ScalarNaturalExponentialFamilyLikelihood(BernoulliFamily())
        elif kind == "multiclass":
            class_count = target_schema.num_classes
            if class_count < 2:
                raise ValueError(
                    "Multiclass classification requires class_labels to declare every "
                    "output class."
                )
            likelihood = CategoricalExponentialFamilyLikelihood(
                CategoricalFamily(class_count),
                prediction_coordinates="full_logits",
            )
        else:
            raise ValueError(
                "SupervisedClassificationTerm supports binary and multiclass "
                "TargetSchema kinds."
            )

        encoded = jnp.asarray(targets)
        if encoded.ndim == 2 and int(encoded.shape[1]) == 1:
            encoded = encoded[:, 0]
        if encoded.ndim != 1:
            raise ValueError("Classification targets must have shape (N,) or (N, 1).")
        if encoded.dtype != jnp.bool_ and not jnp.issubdtype(encoded.dtype, jnp.integer):
            raise TypeError("Classification targets must be integer or Boolean labels.")

        term_weight = jnp.asarray(weight, dtype=float)
        if term_weight.ndim != 0 or not bool(jnp.isfinite(term_weight)):
            raise ValueError("weight must be a finite scalar.")
        if bool(term_weight < 0.0):
            raise ValueError("Classification term weight must be nonnegative.")

        super().__init__(
            str(field),
            component,
            encoded,
            likelihood,
            sampling=sampling,
            observation_operator=observation_operator,
            sample_mask=sample_mask,
            sample_weight=sample_weight,
            weight=term_weight,
            reduction=reduction,
            indices=indices,
            label=label,
        )

        configured = (
            jnp.arange(self.domain.size, dtype=jnp.int32)
            if self.indices is None
            else self.indices
        )
        active_targets = self.values[configured]
        if bool(jnp.any(active_targets < 0)) or bool(
            jnp.any(active_targets >= class_count)
        ):
            raise ValueError(
                f"Configured classification targets must lie within [0, {class_count})."
            )

        self.target_schema = target_schema
        self.classification_kind = kind
        self.class_count = class_count

    def data_metrics(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        batch: SupervisedDatasetBatch | None = None,
        **kwargs: Any,
    ) -> dict[str, Array]:
        """Return classification diagnostics on the exact prepared mini-batch."""
        batch_value = self.sample(key=key) if batch is None else batch
        location, target = self.likelihood.align_observations(
            self._location(functions, batch_value, key=key, **kwargs),
            batch_value.target,
        )
        labels = jnp.asarray(target, dtype=jnp.int32)
        if self.classification_kind == "binary":
            logits = jnp.stack((jnp.zeros_like(location), location), axis=-1)
            prediction = (location >= 0.0).astype(jnp.int32)
            brier = brier_score(
                labels,
                location,
                sample_weight=batch_value.sample_weight,
                from_logits=True,
            )
        else:
            logits = jnp.asarray(location)
            prediction = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            probabilities = self.likelihood.class_probabilities(logits)
            brier = brier_score(
                labels,
                probabilities,
                sample_weight=batch_value.sample_weight,
            )

        negative_log_likelihood = log_loss(
            labels,
            logits,
            sample_weight=batch_value.sample_weight,
            from_logits=True,
        )
        accuracy = accuracy_score(
            labels,
            prediction,
            sample_weight=batch_value.sample_weight,
        )
        return {
            "data_negative_log_likelihood": jnp.asarray(
                negative_log_likelihood.value, dtype=float
            ).reshape(()),
            "data_accuracy": jnp.asarray(accuracy.value, dtype=float).reshape(()),
            "data_brier_score": jnp.asarray(brier.value, dtype=float).reshape(()),
            "data_effective_weight": jnp.asarray(
                negative_log_likelihood.effective_weight, dtype=float
            ).reshape(()),
            "data_valid": jnp.asarray(negative_log_likelihood.valid, dtype=bool).reshape(
                ()
            ),
            "data_status": jnp.asarray(
                negative_log_likelihood.status, dtype=jnp.int32
            ).reshape(()),
        }


__all__ = ["SupervisedClassificationTerm"]
