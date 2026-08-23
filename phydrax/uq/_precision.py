#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp

from .._precision import (
    complex_precision_dtype,
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    real_precision_dtype_name,
)
from .._strict import StrictModule
from .._trainable import NonTrainableState


class PredictivePrecisionPolicy(StrictModule, NonTrainableState):
    """Predictive sample storage and summary arithmetic precision."""

    storage_dtype: str | None = eqx.field(static=True)
    summary_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        storage_dtype: Any | None = None,
        summary_dtype: Any | None = None,
    ):
        storage = (
            None if storage_dtype is None else real_precision_dtype_name(storage_dtype)
        )
        summary = (
            None if summary_dtype is None else real_precision_dtype_name(summary_dtype)
        )
        if storage in ("float16", "bfloat16") or summary in ("float16", "bfloat16"):
            raise ValueError(
                "Predictive precision initially supports float32/float64 only."
            )
        if (
            storage is not None
            and summary is not None
            and precision_itemsize(summary) < precision_itemsize(storage)
        ):
            raise ValueError(
                "Predictive summary precision cannot be narrower than storage."
            )
        request = PrecisionRequest(
            "predictive-uq",
            {"storage": storage, "accumulation": summary, "output": summary},
        )
        self.storage_dtype = storage
        self.summary_dtype = summary
        self.policy_id = request.request_id

    def storage(self, value: Any, /):
        array = jnp.asarray(value)
        if self.storage_dtype is None:
            return array
        target = (
            complex_precision_dtype(self.storage_dtype)
            if jnp.issubdtype(array.dtype, jnp.complexfloating)
            else self.storage_dtype
        )
        return array.astype(target)

    def summary(self, value: Any, /):
        array = jnp.asarray(value)
        if self.summary_dtype is None:
            return array
        target = (
            complex_precision_dtype(self.summary_dtype)
            if jnp.issubdtype(array.dtype, jnp.complexfloating)
            else self.summary_dtype
        )
        return array.astype(target)

    def evidence(self, sample_dtype: Any, /) -> PrecisionEvidenceEnvelope:
        sample = jnp.dtype(sample_dtype)
        sample_real = "float64" if sample.name in ("float64", "complex128") else "float32"
        storage_real = sample_real if self.storage_dtype is None else self.storage_dtype
        summary_real = storage_real if self.summary_dtype is None else self.summary_dtype
        complex_sample = jnp.issubdtype(sample, jnp.complexfloating)
        storage = (
            complex_precision_dtype(storage_real) if complex_sample else storage_real
        )
        summary = (
            complex_precision_dtype(summary_real) if complex_sample else summary_real
        )
        request = PrecisionRequest(
            "predictive-uq",
            {
                "storage": self.storage_dtype,
                "accumulation": self.summary_dtype,
                "output": self.summary_dtype,
            },
        )
        resolution = PrecisionResolution(
            request,
            "phydrax-predictive",
            {"storage": storage, "accumulation": summary, "output": summary},
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))


class ParticlePrecisionPolicy(StrictModule, NonTrainableState):
    """Particle state storage, weight statistics, decisions, and output precision."""

    state_storage_dtype: str | None = eqx.field(static=True)
    statistics_dtype: str = eqx.field(static=True)
    decision_dtype: str = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_storage_dtype: Any | None = None,
        statistics_dtype: Any = jnp.float64,
        decision_dtype: Any = jnp.float64,
        output_dtype: Any | None = None,
    ):
        state = (
            None
            if state_storage_dtype is None
            else real_precision_dtype_name(state_storage_dtype)
        )
        statistics = real_precision_dtype_name(statistics_dtype)
        decision = real_precision_dtype_name(decision_dtype)
        output = None if output_dtype is None else real_precision_dtype_name(output_dtype)
        if any(
            value in ("float16", "bfloat16")
            for value in (state, statistics, decision, output)
        ):
            raise ValueError(
                "Particle precision initially supports float32/float64 only."
            )
        if precision_itemsize(decision) < precision_itemsize(statistics):
            raise ValueError(
                "Particle decision precision cannot be narrower than statistics."
            )
        request = PrecisionRequest(
            "particle-filter",
            {
                "storage": state,
                "accumulation": statistics,
                "certification": decision,
                "output": output,
            },
        )
        self.state_storage_dtype = state
        self.statistics_dtype = statistics
        self.decision_dtype = decision
        self.output_dtype = output
        self.policy_id = request.request_id

    def evidence(self, state_dtype: Any, /) -> PrecisionEvidenceEnvelope:
        observed = jnp.dtype(state_dtype)
        complex_state = jnp.issubdtype(observed, jnp.complexfloating)
        if self.state_storage_dtype is not None:
            state = (
                complex_precision_dtype(self.state_storage_dtype)
                if complex_state
                else self.state_storage_dtype
            )
        elif jnp.issubdtype(observed, jnp.inexact):
            state = precision_dtype_name(observed)
        else:
            state = None
        output = (
            state
            if self.output_dtype is None
            else (
                complex_precision_dtype(self.output_dtype)
                if complex_state
                else self.output_dtype
            )
        )
        request = PrecisionRequest(
            "particle-filter",
            {
                "storage": self.state_storage_dtype,
                "accumulation": self.statistics_dtype,
                "certification": self.decision_dtype,
                "output": self.output_dtype,
            },
        )
        resolution = PrecisionResolution(
            request,
            "phydrax-particle-filter",
            {
                "storage": state,
                "accumulation": self.statistics_dtype,
                "certification": self.decision_dtype,
                "output": output,
            },
        )
        return PrecisionEvidenceEnvelope(resolution, dict(resolution.effective))

    def state(self, value: Any, /):
        array = jnp.asarray(value)
        if self.state_storage_dtype is None:
            return array
        target = (
            complex_precision_dtype(self.state_storage_dtype)
            if jnp.issubdtype(array.dtype, jnp.complexfloating)
            else self.state_storage_dtype
        )
        return array.astype(target)

    def statistics(self, value: Any, /):
        return jnp.asarray(value, dtype=self.statistics_dtype)

    def decision(self, value: Any, /):
        return jnp.asarray(value, dtype=self.decision_dtype)

    def output(self, value: Any, /):
        array = jnp.asarray(value)
        if self.output_dtype is None:
            return array
        target = (
            complex_precision_dtype(self.output_dtype)
            if jnp.issubdtype(array.dtype, jnp.complexfloating)
            else self.output_dtype
        )
        return array.astype(target)


__all__ = ["ParticlePrecisionPolicy", "PredictivePrecisionPolicy"]
