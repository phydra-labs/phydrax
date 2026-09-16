#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class UnbinnedDataSet(StrictModule, NonTrainableState):
    observations: Array
    weights: Array
    active: Array
    lower: Array
    upper: Array
    observable_names: tuple[str, ...] = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)

    def __init__(
        self,
        observations: ArrayLike,
        /,
        *,
        observable_names: Sequence[str],
        lower: ArrayLike,
        upper: ArrayLike,
        weights: ArrayLike | None = None,
        active: ArrayLike | None = None,
    ):
        values = np.asarray(observations, dtype=float)
        names = tuple(str(value).strip() for value in observable_names)
        lower_ = np.asarray(lower, dtype=float)
        upper_ = np.asarray(upper, dtype=float)
        if (
            values.ndim != 2
            or values.shape[0] < 1
            or values.shape[1] < 1
            or len(names) != values.shape[1]
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Observations require a non-empty event/observable matrix and unique names."
            )
        if (
            lower_.shape != (values.shape[1],)
            or upper_.shape != lower_.shape
            or np.any(~np.isfinite(lower_))
            or np.any(~np.isfinite(upper_))
            or np.any(lower_ >= upper_)
        ):
            raise ValueError("Observable bounds must be finite, aligned, and increasing.")
        weights_ = (
            np.ones((values.shape[0],), dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        )
        active_ = (
            np.ones((values.shape[0],), dtype=bool)
            if active is None
            else np.asarray(active, dtype=bool)
        )
        if weights_.shape != active_.shape or weights_.shape != (values.shape[0],):
            raise ValueError("Unbinned weights/activity must align with events.")
        in_domain = np.all((values >= lower_) & (values <= upper_), axis=1)
        if (
            np.any(~np.isfinite(values[active_]))
            or np.any(~np.isfinite(weights_[active_]))
            or np.any(weights_[active_] < 0.0)
            or np.any(active_ & ~in_domain)
        ):
            raise ValueError(
                "Active unbinned events must be finite, nonnegative-weighted, and in domain."
            )
        self.observations = jnp.asarray(values)
        self.weights = jnp.asarray(weights_)
        self.active = jnp.asarray(active_)
        self.lower = jnp.asarray(lower_)
        self.upper = jnp.asarray(upper_)
        self.observable_names = names
        self.dataset_id = canonical_fingerprint(
            {
                "kind": "hep-unbinned-dataset",
                "arrays": array_tree_fingerprint(
                    (values, weights_, active_, lower_, upper_)
                ),
                "observables": list(names),
            }
        )


class ExtendedMixtureModel(StrictModule, NonTrainableState):
    component_names: tuple[str, ...] = eqx.field(static=True)
    normalization_evidence_ids: tuple[str, ...] = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self, component_names: Sequence[str], normalization_evidence_ids: Sequence[str], /
    ):
        names = tuple(str(value).strip() for value in component_names)
        evidence = tuple(str(value).strip() for value in normalization_evidence_ids)
        if (
            not names
            or len(names) != len(evidence)
            or any(not value for value in names + evidence)
            or len(set(names)) != len(names)
        ):
            raise ValueError(
                "Mixture components require unique names and normalization evidence."
            )
        self.component_names = names
        self.normalization_evidence_ids = evidence
        self.model_id = canonical_fingerprint(
            {
                "kind": "hep-extended-mixture-model",
                "components": list(names),
                "normalization_evidence": list(evidence),
            }
        )


class UnbinnedLikelihoodEvaluation(StrictModule, NonTrainableState):
    event_log_intensity: Array
    extended_log_likelihood: Array
    finite: Array
    valid: Array
    model_id: str = eqx.field(static=True)
    dataset_id: str = eqx.field(static=True)


def evaluate_extended_mixture(
    model: ExtendedMixtureModel,
    dataset: UnbinnedDataSet,
    component_log_densities: ArrayLike,
    yields: ArrayLike,
    /,
) -> UnbinnedLikelihoodEvaluation:
    """Evaluate normalized component densities supplied by a qualified provider."""
    if not isinstance(model, ExtendedMixtureModel) or not isinstance(
        dataset, UnbinnedDataSet
    ):
        raise TypeError("model and dataset must use unbinned HEP statistics types.")
    log_densities = jnp.asarray(component_log_densities, dtype=dataset.observations.dtype)
    yields_ = jnp.asarray(yields, dtype=dataset.observations.dtype)
    expected_shape = (dataset.observations.shape[0], len(model.component_names))
    if log_densities.shape != expected_shape or yields_.shape != (
        len(model.component_names),
    ):
        raise ValueError(
            "Component density/yield support is incompatible with the model."
        )
    positive = jnp.all(yields_ > 0.0)
    log_intensity = jsp.special.logsumexp(
        jnp.log(jnp.maximum(yields_, jnp.finfo(yields_.dtype).tiny))[None, :]
        + log_densities,
        axis=1,
    )
    value = jnp.sum(
        jnp.where(dataset.active, dataset.weights * log_intensity, 0.0)
    ) - jnp.sum(yields_)
    finite = jnp.all(
        jnp.where(dataset.active[:, None], jnp.isfinite(log_densities), True)
    ) & jnp.isfinite(value)
    return UnbinnedLikelihoodEvaluation(
        log_intensity,
        value,
        finite,
        finite & positive,
        model.model_id,
        dataset.dataset_id,
    )


__all__ = [
    "ExtendedMixtureModel",
    "UnbinnedDataSet",
    "UnbinnedLikelihoodEvaluation",
    "evaluate_extended_mixture",
]
