# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ....._probability import AbstractProbabilityLaw
from ....._strict import StrictModule
from ...data import FunctionSamples, OperatorBatch
from ..conditioning._function_frame import (
    FunctionFrameEncoding,
    FunctionFrameReconstructor,
    FunctionProjectionPolicy,
    FunctionProjectionReport,
)


class FunctionFrameCoefficientFlowState(StrictModule):
    encoding: FunctionFrameEncoding
    coefficient_law: AbstractProbabilityLaw
    law_id: str = eqx.field(static=True)


class FunctionFrameProjectedLogProb(StrictModule):
    coefficients: Array
    log_prob: Array
    projection: FunctionProjectionReport
    reference_measure: str = eqx.field(static=True)


class ConditionalFunctionFrameFlowOperator(StrictModule):
    """Conditional normalized coefficient flow with arbitrary-query frame decoding."""

    encoder: FunctionFrameReconstructor
    coefficient_law_factory: Any
    field_space_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)

    def __init__(
        self,
        encoder: FunctionFrameReconstructor,
        coefficient_law_factory: Any,
        /,
        *,
        field_space_id: str,
    ):
        if not isinstance(encoder, FunctionFrameReconstructor):
            raise TypeError("encoder must be FunctionFrameReconstructor.")
        if not callable(coefficient_law_factory):
            raise TypeError("coefficient_law_factory must be callable.")
        if not field_space_id:
            raise ValueError("field_space_id must be nonempty.")
        self.encoder = encoder
        self.coefficient_law_factory = coefficient_law_factory
        self.field_space_id = str(field_space_id)
        self.reference_measure = "coefficient-space"

    def condition(self, batch: OperatorBatch, /) -> FunctionFrameCoefficientFlowState:
        encoding = self.encoder.encode_inputs(batch)
        law = self.coefficient_law_factory(encoding.fused_coefficients)
        if not isinstance(law, AbstractProbabilityLaw):
            raise TypeError("coefficient_law_factory must return AbstractProbabilityLaw.")
        expected = (self.encoder.latent_size,)
        if law.event_shape != expected:
            raise ValueError(f"Coefficient law event shape must be {expected}.")
        return FunctionFrameCoefficientFlowState(
            encoding,
            law,
            f"conditional-frame-flow:{self.field_space_id}",
        )

    def sample_coefficients(
        self,
        state: FunctionFrameCoefficientFlowState,
        key: Key[Array, ""],
        sample_shape: tuple[int, ...] = (),
    ) -> Array:
        return state.coefficient_law.sample(key, sample_shape)

    def log_prob_coefficients(
        self, state: FunctionFrameCoefficientFlowState, coefficients: ArrayLike, /
    ) -> Array:
        return state.coefficient_law.log_prob(coefficients)

    def decode(
        self,
        state: FunctionFrameCoefficientFlowState,
        coefficients: ArrayLike,
        query: FunctionSamples,
        /,
    ) -> Array:
        values = jnp.asarray(coefficients)
        expected_prefix = state.encoding.case_shape
        if values.shape[-1:] != (self.encoder.latent_size,):
            raise ValueError("Coefficient samples must end in the target frame rank.")
        sample_shape = values.shape[: values.ndim - 1 - len(expected_prefix)]
        if sample_shape:
            flattened = values.reshape(
                (-1,) + expected_prefix + (self.encoder.latent_size,)
            )
            decoded = jnp.stack(
                tuple(
                    self.encoder.target_frame.decode(
                        item,
                        query,
                        case_shape=expected_prefix,
                    )
                    for item in flattened
                ),
                axis=0,
            )
            return decoded.reshape(sample_shape + decoded.shape[1:])
        return self.encoder.target_frame.decode(
            values,
            query,
            case_shape=expected_prefix,
        )

    def sample_field(
        self,
        state: FunctionFrameCoefficientFlowState,
        key: Key[Array, ""],
        query: FunctionSamples,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[Array, Array]:
        coefficients = self.sample_coefficients(state, key, sample_shape)
        return coefficients, self.decode(state, coefficients, query)

    def project_and_log_prob(
        self,
        state: FunctionFrameCoefficientFlowState,
        samples: FunctionSamples,
        /,
        *,
        policy: FunctionProjectionPolicy | None = None,
    ) -> FunctionFrameProjectedLogProb:
        report = self.encoder.target_frame.project(
            samples,
            case_shape=state.encoding.case_shape,
            policy=policy,
        )
        coefficients = report.require_coefficients()
        return FunctionFrameProjectedLogProb(
            coefficients,
            state.coefficient_law.log_prob(coefficients),
            report,
            "coefficient-space",
        )

    def field_log_prob(self, *args: Any, **kwargs: Any) -> Array:
        del args, kwargs
        raise ValueError(
            "Function-frame decoding does not define sampled-field Lebesgue density; "
            "use log_prob_coefficients or explicit injective-map evidence."
        )


__all__ = [
    "ConditionalFunctionFrameFlowOperator",
    "FunctionFrameCoefficientFlowState",
    "FunctionFrameProjectedLogProb",
]
