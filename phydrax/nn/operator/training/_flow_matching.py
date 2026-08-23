#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._flow_matching_metric import AbstractFlowMatchingMetric
from ...._trainable import NonTrainableState
from ..data import FunctionSamples, OperatorOutputSpec
from ._physics import operator_hilbert_norm


class OperatorFlowMatchingMetric(AbstractFlowMatchingMetric, NonTrainableState):
    """Fixed-query quadrature-aware velocity error for one complete output field."""

    query: FunctionSamples
    output_spec: OperatorOutputSpec
    channel_metric: Array | None
    event_shape: tuple[int, ...] = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)

    def __init__(
        self,
        query: FunctionSamples,
        output_spec: OperatorOutputSpec,
        /,
        *,
        channel_metric: ArrayLike | None = None,
        metric_id: str | None = None,
    ):
        if not isinstance(query, FunctionSamples):
            raise TypeError("query must be FunctionSamples.")
        if not isinstance(output_spec, OperatorOutputSpec):
            raise TypeError("output_spec must be OperatorOutputSpec.")
        if query.geometry_case_shape:
            raise ValueError(
                "Operator flow matching requires one fixed query geometry per metric."
            )
        metric = None if channel_metric is None else jnp.asarray(channel_metric)
        channels = 1 if output_spec.channels == "scalar" else int(output_spec.channels)
        if metric is not None and metric.shape != (channels, channels):
            raise ValueError(
                f"channel_metric must have shape {(channels, channels)}; got {metric.shape}."
            )
        events = query.sample_shape + output_spec.channel_shape
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "operator-flow-matching-metric-v1",
                    "query": query.geometry_fingerprint(),
                    "channels": output_spec.channels,
                    "component_names": list(output_spec.component_names),
                    "channel_metric_shape": None
                    if metric is None
                    else list(metric.shape),
                }
            )
            if metric_id is None
            else str(metric_id)
        )
        if not resolved_id:
            raise ValueError("metric_id must be non-empty.")
        self.query = query
        self.output_spec = output_spec
        self.channel_metric = metric
        self.event_shape = events
        self.metric_id = resolved_id

    def __call__(
        self,
        state: Array,
        predicted_velocity: Array,
        target_velocity: Array,
        /,
    ) -> Array:
        if not (
            state.shape
            == predicted_velocity.shape
            == target_velocity.shape
            == self.event_shape
        ):
            raise ValueError(
                f"Operator flow-matching values must have fixed event shape "
                f"{self.event_shape}."
            )
        residual = predicted_velocity - target_velocity
        value = operator_hilbert_norm(
            residual,
            self.query,
            case_shape=(),
            channel_metric=self.channel_metric,
            squared=True,
            reduction="none",
        )
        return jnp.asarray(value, dtype=float).reshape(())


__all__ = ["OperatorFlowMatchingMetric"]
