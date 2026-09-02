#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ._precision import (
    complex_precision_dtype,
    precision_dtype_name,
    precision_itemsize,
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
    PrecisionResourceAssumptions,
    real_precision_dtype_name,
    ScalarPrecisionDType,
)
from ._strict import StrictModule
from ._trainable import NonTrainableState


_SUPPORTED_TEMPORAL_DTYPES = frozenset(("float32", "float64", "complex64", "complex128"))


def _real_component(value: Any, /) -> str:
    name = precision_dtype_name(value)
    if name == "complex64":
        return "float32"
    if name == "complex128":
        return "float64"
    return real_precision_dtype_name(name)


def _effective_dtype(
    requested: str | None,
    observed: ScalarPrecisionDType,
    /,
) -> ScalarPrecisionDType:
    if requested is None:
        return observed
    if observed in ("complex64", "complex128") and requested in (
        "float32",
        "float64",
    ):
        return complex_precision_dtype(requested)
    return precision_dtype_name(requested)


class TemporalPrecisionPolicy(StrictModule, NonTrainableState):
    """Coefficient, state, stage, reduction, decision, and persistence precision."""

    coefficient_dtype: str | None = eqx.field(static=True)
    state_dtype: str | None = eqx.field(static=True)
    stage_dtype: str | None = eqx.field(static=True)
    accumulation_dtype: str | None = eqx.field(static=True)
    residual_dtype: str | None = eqx.field(static=True)
    decision_dtype: str | None = eqx.field(static=True)
    checkpoint_dtype: str | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        coefficient_dtype: Any | None = None,
        state_dtype: Any | None = None,
        stage_dtype: Any | None = None,
        accumulation_dtype: Any | None = None,
        residual_dtype: Any | None = None,
        decision_dtype: Any | None = None,
        checkpoint_dtype: Any | None = None,
        output_dtype: Any | None = None,
    ):
        coefficient = (
            None
            if coefficient_dtype is None
            else real_precision_dtype_name(coefficient_dtype)
        )
        state = None if state_dtype is None else precision_dtype_name(state_dtype)
        stage = None if stage_dtype is None else precision_dtype_name(stage_dtype)
        accumulation = (
            None
            if accumulation_dtype is None
            else precision_dtype_name(accumulation_dtype)
        )
        residual = (
            None if residual_dtype is None else precision_dtype_name(residual_dtype)
        )
        decision = (
            None if decision_dtype is None else real_precision_dtype_name(decision_dtype)
        )
        checkpoint = (
            None if checkpoint_dtype is None else precision_dtype_name(checkpoint_dtype)
        )
        output = None if output_dtype is None else precision_dtype_name(output_dtype)
        requested = tuple(
            value
            for value in (
                coefficient,
                state,
                stage,
                accumulation,
                residual,
                decision,
                checkpoint,
                output,
            )
            if value is not None
        )
        if any(value not in _SUPPORTED_TEMPORAL_DTYPES for value in requested):
            raise ValueError("Temporal precision supports float32/64 and complex64/128.")
        if (
            accumulation is not None
            and stage is not None
            and precision_itemsize(_real_component(accumulation))
            < precision_itemsize(_real_component(stage))
        ):
            raise ValueError(
                "Temporal accumulation precision cannot be narrower than stages."
            )
        if (
            decision is not None
            and accumulation is not None
            and precision_itemsize(decision)
            < precision_itemsize(_real_component(accumulation))
        ):
            raise ValueError(
                "Temporal decision precision cannot be narrower than accumulation."
            )
        request = PrecisionRequest(
            "temporal",
            {
                "coefficient": coefficient,
                "storage": state,
                "compute": stage,
                "accumulation": accumulation,
                "residual": residual,
                "certification": decision,
                "checkpoint": checkpoint,
                "output": output,
            },
        )
        self.coefficient_dtype = coefficient
        self.state_dtype = state
        self.stage_dtype = stage
        self.accumulation_dtype = accumulation
        self.residual_dtype = residual
        self.decision_dtype = decision
        self.checkpoint_dtype = checkpoint
        self.output_dtype = output
        self.policy_id = request.request_id

    @property
    def request(self) -> PrecisionRequest:
        return PrecisionRequest(
            "temporal",
            {
                "coefficient": self.coefficient_dtype,
                "storage": self.state_dtype,
                "compute": self.stage_dtype,
                "accumulation": self.accumulation_dtype,
                "residual": self.residual_dtype,
                "certification": self.decision_dtype,
                "checkpoint": self.checkpoint_dtype,
                "output": self.output_dtype,
            },
        )

    def validate_state(self, state: Any, /) -> ScalarPrecisionDType:
        leaves = tuple(jnp.asarray(value) for value in jax.tree.leaves(state))
        if not leaves:
            raise ValueError("Temporal state must contain array leaves.")
        observed_values = tuple(precision_dtype_name(value.dtype) for value in leaves)
        real_values = {_real_component(value) for value in observed_values}
        if len(real_values) != 1:
            raise TypeError(
                "Temporal PyTree state leaves must share one real component dtype."
            )
        observed = (
            next(
                value for value in observed_values if value in ("complex64", "complex128")
            )
            if any(value in ("complex64", "complex128") for value in observed_values)
            else observed_values[0]
        )
        if self.state_dtype is not None and observed != self.state_dtype:
            raise TypeError(
                f"Temporal state dtype {observed} does not match {self.state_dtype}."
            )
        return observed

    def validate_implicit_state(self, state: Any, /) -> None:
        observed = self.validate_state(state)
        state_real = _real_component(observed)
        coefficient = (
            state_real if self.coefficient_dtype is None else self.coefficient_dtype
        )
        stage = _effective_dtype(self.stage_dtype, observed)
        residual = _effective_dtype(self.residual_dtype, observed)
        if coefficient != state_real or stage != observed or residual != observed:
            raise ValueError(
                "Native implicit temporal methods require coefficient, stage, and "
                "residual dtypes to match the stored state; accumulation and decision "
                "precision may be wider."
            )

    def validate_diffrax_state(
        self,
        state: Any,
        /,
        *,
        internal_precision: bool = False,
    ) -> None:
        observed = self.validate_state(state)
        state_real = _real_component(observed)
        if (
            observed != state_real
            and self.output_dtype is not None
            and _real_component(self.output_dtype) == self.output_dtype
        ):
            raise ValueError(
                "Complex Diffrax state requires a complex output dtype; explicit "
                "real/imaginary projection belongs outside the temporal backend."
            )
        coefficient = (
            state_real if self.coefficient_dtype is None else self.coefficient_dtype
        )
        if coefficient != state_real:
            raise ValueError(
                "Diffrax requires coefficient precision to match the real component "
                "of state precision because it promotes state against time."
            )
        if internal_precision:
            return
        stage = _effective_dtype(self.stage_dtype, observed)
        residual = _effective_dtype(self.residual_dtype, observed)
        accumulation = _effective_dtype(self.accumulation_dtype, stage)
        decision = (
            _real_component(accumulation)
            if self.decision_dtype is None
            else self.decision_dtype
        )
        checkpoint = _effective_dtype(self.checkpoint_dtype, observed)
        if (
            stage != observed
            or residual != observed
            or accumulation != observed
            or decision != state_real
            or checkpoint != observed
        ):
            raise ValueError(
                "Generic Diffrax solvers do not expose internal stage, residual, "
                "accumulation, decision, or checkpoint placement; only matching "
                "internal precision and an independent output dtype are supported."
            )

    def coefficient(self, value: Any, /):
        array = jnp.asarray(value)
        return (
            array
            if self.coefficient_dtype is None
            else array.astype(self.coefficient_dtype)
        )

    def stage(self, value: Any, /):
        array = jnp.asarray(value)
        if self.stage_dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(
            _effective_dtype(self.stage_dtype, precision_dtype_name(array.dtype))
        )

    def accumulation(self, value: Any, /):
        array = jnp.asarray(value)
        if self.accumulation_dtype is None or not jnp.issubdtype(
            array.dtype, jnp.inexact
        ):
            return array
        return array.astype(
            _effective_dtype(
                self.accumulation_dtype,
                precision_dtype_name(array.dtype),
            )
        )

    def residual(self, value: Any, /):
        array = jnp.asarray(value)
        if self.residual_dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(
            _effective_dtype(self.residual_dtype, precision_dtype_name(array.dtype))
        )

    def decision(self, value: Any, /):
        array = jnp.asarray(value)
        return array if self.decision_dtype is None else array.astype(self.decision_dtype)

    def checkpoint(self, value: Any, /):
        array = jnp.asarray(value)
        if self.checkpoint_dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
            return array
        return array.astype(
            _effective_dtype(
                self.checkpoint_dtype,
                precision_dtype_name(array.dtype),
            )
        )

    def output(self, value: Any, /):
        def cast(array_value):
            array = jnp.asarray(array_value)
            if self.output_dtype is None or not jnp.issubdtype(array.dtype, jnp.inexact):
                return array
            return array.astype(
                _effective_dtype(self.output_dtype, precision_dtype_name(array.dtype))
            )

        return jax.tree.map(cast, value)

    def evidence_for(
        self,
        state: Any,
        time: Any,
        /,
        *,
        children: dict[str, PrecisionEvidenceEnvelope] | None = None,
    ) -> PrecisionEvidenceEnvelope:
        state_observed = self.validate_state(state)
        real_precision_dtype_name(jnp.asarray(time).dtype)
        coefficient = (
            _real_component(state_observed)
            if self.coefficient_dtype is None
            else self.coefficient_dtype
        )
        storage = _effective_dtype(self.state_dtype, state_observed)
        compute = _effective_dtype(self.stage_dtype, storage)
        accumulation = _effective_dtype(self.accumulation_dtype, compute)
        residual = _effective_dtype(self.residual_dtype, compute)
        decision = (
            _real_component(accumulation)
            if self.decision_dtype is None
            else self.decision_dtype
        )
        checkpoint = _effective_dtype(self.checkpoint_dtype, storage)
        output = _effective_dtype(self.output_dtype, storage)
        request = self.request
        resolution = PrecisionResolution(
            request,
            "phydrax-temporal",
            {
                "coefficient": coefficient,
                "storage": storage,
                "compute": compute,
                "accumulation": accumulation,
                "residual": residual,
                "certification": decision,
                "checkpoint": checkpoint,
                "output": output,
            },
        )
        return PrecisionEvidenceEnvelope(
            resolution,
            dict(resolution.effective),
            children={} if children is None else children,
        )

    def resource_assumptions_for(
        self,
        state: Any,
        time: Any,
        /,
    ) -> PrecisionResourceAssumptions:
        evidence = self.evidence_for(state, time)
        return PrecisionResourceAssumptions("temporal", dict(evidence.observed))


__all__ = ["TemporalPrecisionPolicy"]
