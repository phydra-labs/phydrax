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


class IntegrationPrecisionPolicy(StrictModule, NonTrainableState):
    """Evaluation, accumulation, adaptive-decision, and output precision."""

    evaluation_dtype: str | None = eqx.field(static=True)
    accumulation_dtype: str | None = eqx.field(static=True)
    decision_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        evaluation_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        decision_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ):
        evaluation = (
            None
            if evaluation_dtype is None
            else real_precision_dtype_name(evaluation_dtype)
        )
        accumulation = (
            None
            if accumulation_dtype is None
            else real_precision_dtype_name(accumulation_dtype)
        )
        decision = (
            None if decision_dtype is None else real_precision_dtype_name(decision_dtype)
        )
        output = None if output_dtype is None else precision_dtype_name(output_dtype)
        if (
            evaluation is not None
            and accumulation is not None
            and precision_itemsize(accumulation) < precision_itemsize(evaluation)
        ):
            raise ValueError(
                "Integration accumulation cannot be narrower than evaluation."
            )
        if (
            accumulation is not None
            and decision is not None
            and precision_itemsize(decision) < precision_itemsize(accumulation)
        ):
            raise ValueError(
                "Integration decision precision cannot be narrower than accumulation."
            )
        request = PrecisionRequest(
            "integration",
            {
                "compute": evaluation,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        self.evaluation_dtype = evaluation
        self.accumulation_dtype = accumulation
        self.decision_dtype = decision
        self.output_dtype = output
        self.policy_id = request.request_id

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "integration",
            {
                "compute": self.evaluation_dtype,
                "accumulation": self.accumulation_dtype,
                "certification": self.decision_dtype,
                "output": self.output_dtype,
            },
        )

    def evaluation(self, value: Any, /):
        array = jnp.asarray(value)
        if self.evaluation_dtype is None:
            return array
        target = (
            complex_precision_dtype(self.evaluation_dtype)
            if jnp.issubdtype(array.dtype, jnp.complexfloating)
            else self.evaluation_dtype
        )
        return array.astype(target)

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        if self.accumulation_dtype is None:
            return array
        target = (
            complex_precision_dtype(self.accumulation_dtype)
            if jnp.issubdtype(array.dtype, jnp.complexfloating)
            else self.accumulation_dtype
        )
        return array.astype(target)

    def decision(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.decision_dtype is None else array.astype(self.decision_dtype)

    def output(self, value: Any, /):
        array = jnp.asarray(value)
        if self.output_dtype is None:
            return array
        target = self.output_dtype
        if jnp.issubdtype(array.dtype, jnp.complexfloating) and target in (
            "float16",
            "bfloat16",
            "float32",
            "float64",
        ):
            target = complex_precision_dtype(real_precision_dtype_name(target))
        return array.astype(target)

    def evidence_for(
        self,
        value: Any,
        /,
        *,
        children: dict[str, PrecisionEvidenceEnvelope] | None = None,
    ) -> PrecisionEvidenceEnvelope:
        array = jnp.asarray(value)
        complex_value = jnp.issubdtype(array.dtype, jnp.complexfloating)
        observed_input = precision_dtype_name(array.dtype)

        def effective(value_: str | None, fallback: str, /) -> str:
            resolved = fallback if value_ is None else value_
            if complex_value and resolved in (
                "float16",
                "bfloat16",
                "float32",
                "float64",
            ):
                return complex_precision_dtype(resolved)
            return resolved

        compute = effective(self.evaluation_dtype, observed_input)
        accumulation = effective(self.accumulation_dtype, compute)
        decision = (
            real_precision_dtype_name(jnp.real(array).dtype)
            if self.decision_dtype is None
            else self.decision_dtype
        )
        output = effective(self.output_dtype, accumulation)
        request = self.request
        resolution = PrecisionResolution(
            request,
            "phydrax-integration",
            {
                "compute": compute,
                "accumulation": accumulation,
                "certification": decision,
                "output": output,
            },
        )
        return PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
            children={} if children is None else children,
        )


__all__ = ["IntegrationPrecisionPolicy"]
