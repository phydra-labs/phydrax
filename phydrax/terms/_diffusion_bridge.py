#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Mapping

import coordax as cx
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

import phydrax.ein as ein

from .._doc import DOC_KEY0
from .._strict import StrictModule
from .._term import AbstractScalarTerm
from ..domain import DomainFunction


class DiffusionBridgeControlDataset(StrictModule):
    """Finite Doob conditional control targets on represented support nodes."""

    times: Array
    states: Array
    reference_drift: Array
    controlled_drift_targets: Array
    weights: Array
    mask: Array
    bridge_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        states: ArrayLike,
        reference_drift: ArrayLike,
        controlled_drift_targets: ArrayLike,
        weights: ArrayLike,
        mask: ArrayLike,
        /,
        *,
        bridge_id: str,
    ):
        times_ = jnp.asarray(times)
        states_ = jnp.asarray(states)
        reference = jnp.asarray(reference_drift)
        targets = jnp.asarray(controlled_drift_targets)
        weights_ = jnp.asarray(weights)
        mask_ = jnp.asarray(mask, dtype=bool)
        if states_.shape != reference.shape or states_.shape != targets.shape:
            raise ValueError("state and drift target arrays must share one shape.")
        if (
            times_.shape != states_.shape[:-1]
            or weights_.shape != times_.shape
            or mask_.shape != times_.shape
        ):
            raise ValueError("times, weights, and mask must align state sample axes.")
        if not bridge_id:
            raise ValueError("bridge_id must be non-empty.")
        self.times = times_
        self.states = states_
        self.reference_drift = reference
        self.controlled_drift_targets = targets
        self.weights = weights_
        self.mask = mask_
        self.bridge_id = bridge_id


class DiffusionBridgeDriftTerm(AbstractScalarTerm):
    """Weighted supervised drift fit to one frozen prepared-chain target."""

    dataset: DiffusionBridgeControlDataset
    metric: Any
    field: str = eqx.field(static=True)
    label: str | None = eqx.field(static=True)

    def __init__(
        self,
        field: str,
        dataset: DiffusionBridgeControlDataset,
        /,
        *,
        metric: Any = None,
        label: str | None = None,
    ):
        if not field:
            raise ValueError("field must be non-empty.")
        if not isinstance(dataset, DiffusionBridgeControlDataset):
            raise TypeError("dataset must be DiffusionBridgeControlDataset.")
        if metric is not None and not callable(metric):
            raise TypeError("metric must be callable or None.")
        self.field = field
        self.dataset = dataset
        self.metric = metric
        self.label = label

    def loss(
        self,
        functions: Mapping[str, DomainFunction],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        iter_=None,
        **kwargs,
    ) -> Array:
        del iter_, kwargs
        field = functions[self.field]
        predicted = field.func(self.dataset.times, self.dataset.states, key=key)
        values = predicted.data if isinstance(predicted, cx.Field) else predicted
        residual = jnp.asarray(values) - self.dataset.controlled_drift_targets
        if self.metric is None:
            squared = ein.contract("...i,...i->...", residual, residual)
        else:
            metric = jnp.asarray(self.metric(self.dataset.states))
            squared = ein.contract("...i,...ij,...j->...", residual, metric, residual)
        weights = jnp.where(self.dataset.mask, self.dataset.weights, 0.0)
        normalizer = jnp.sum(weights)
        checked = eqx.error_if(
            normalizer,
            (~jnp.isfinite(normalizer)) | (normalizer <= 0.0),
            "Diffusion bridge drift dataset has zero or invalid active mass.",
        )
        return jnp.sum(weights * squared) / checked


__all__ = ["DiffusionBridgeControlDataset", "DiffusionBridgeDriftTerm"]
