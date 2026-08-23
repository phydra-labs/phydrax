#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule


PhysicalityStatus: TypeAlias = Literal["valid", "invalid", "unknown"]


class ApproximationAxis(StrictModule):
    name: str = eqx.field(static=True)
    value: Array
    parent_value: Array | None
    units: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        value: ArrayLike,
        /,
        *,
        parent_value: ArrayLike | None = None,
        units: str = "dimensionless",
    ):
        identifier = str(name)
        if not identifier:
            raise ValueError("Approximation-axis name must be non-empty.")
        self.name = identifier
        self.value = jnp.asarray(value)
        self.parent_value = None if parent_value is None else jnp.asarray(parent_value)
        self.units = str(units)


class ApproximationQuantity(StrictModule):
    name: str = eqx.field(static=True)
    value: Array
    threshold: Array
    units: str = eqx.field(static=True)
    norm_id: str = eqx.field(static=True)
    estimate_kind: str = eqx.field(static=True)
    confidence: Array
    valid: Array

    def __init__(
        self,
        name: str,
        value: ArrayLike,
        threshold: ArrayLike,
        /,
        *,
        units: str,
        norm_id: str,
        estimate_kind: Literal["bound", "estimate", "statistical"],
        confidence: ArrayLike = jnp.nan,
    ):
        if estimate_kind not in ("bound", "estimate", "statistical"):
            raise ValueError("Unknown approximation quantity kind.")
        self.name = str(name)
        self.value = jnp.asarray(value)
        self.threshold = jnp.asarray(threshold)
        self.units = str(units)
        self.norm_id = str(norm_id)
        self.estimate_kind = estimate_kind
        self.confidence = jnp.asarray(confidence)
        self.valid = (
            jnp.all(jnp.isfinite(self.value))
            & jnp.all(jnp.isfinite(self.threshold))
            & jnp.all(self.value <= self.threshold)
        )


class OpenSystemApproximationEvidence(StrictModule):
    axes: tuple[ApproximationAxis, ...]
    quantities: tuple[ApproximationQuantity, ...]
    valid: Array
    representation_id: str = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope | None = eqx.field(static=True)
    precision_policy_ids: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        representation_id: str,
        axes: Sequence[ApproximationAxis],
        quantities: Sequence[ApproximationQuantity],
        /,
        *,
        execution_valid: ArrayLike,
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
        precision_policy_ids: Sequence[str] = (),
    ):
        axes_ = tuple(axes)
        quantities_ = tuple(quantities)
        if not axes_ or not quantities_:
            raise ValueError(
                "Approximation evidence requires declared axes and quantities."
            )
        if precision_evidence is not None and not isinstance(
            precision_evidence, PrecisionEvidenceEnvelope
        ):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        self.representation_id = str(representation_id)
        self.axes = axes_
        self.quantities = quantities_
        self.valid = jnp.asarray(execution_valid, dtype=bool) & jnp.all(
            jnp.stack([quantity.valid for quantity in quantities_])
        )
        self.precision_evidence = precision_evidence
        self.precision_policy_ids = tuple(
            str(identifier) for identifier in precision_policy_ids
        )


class OpenSystemPhysicalityEvidence(StrictModule):
    trace_residual: Array
    hermiticity_residual: Array
    positivity_margin: Array
    channel_cp_margin: Array
    valid: Array
    status: PhysicalityStatus = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope | None = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        trace_residual: ArrayLike = jnp.nan,
        hermiticity_residual: ArrayLike = jnp.nan,
        positivity_margin: ArrayLike = jnp.nan,
        channel_cp_margin: ArrayLike = jnp.nan,
        status: PhysicalityStatus = "unknown",
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
    ):
        if precision_evidence is not None and not isinstance(
            precision_evidence,
            PrecisionEvidenceEnvelope,
        ):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        if status not in ("valid", "invalid", "unknown"):
            raise ValueError("Unknown physicality status.")
        self.trace_residual = jnp.asarray(trace_residual)
        self.hermiticity_residual = jnp.asarray(hermiticity_residual)
        self.positivity_margin = jnp.asarray(positivity_margin)
        self.channel_cp_margin = jnp.asarray(channel_cp_margin)
        self.status = status
        self.valid = jnp.asarray(status == "valid")
        self.precision_evidence = precision_evidence


class QuantumGeneratorAction(StrictModule):
    action_function: Callable[[Array, Array, Any], Array]
    representation_id: str = eqx.field(static=True)
    generator_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Callable[[Array, Array, Any], Array],
        /,
        *,
        representation_id: str,
        generator_id: str,
    ):
        if not callable(action):
            raise TypeError("Generator action must be callable.")
        self.action_function = action
        self.representation_id = str(representation_id)
        self.generator_id = str(generator_id)

    def __call__(self, time: Array, state: Array, args: Any = None, /) -> Array:
        result = jnp.asarray(self.action_function(time, state, args))
        if result.shape != state.shape:
            raise ValueError("Generator action must preserve the state shape.")
        return result


class QuantumObservablePlan(StrictModule):
    reducer: Callable[[Any], Array]
    observable_id: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)

    def __init__(
        self,
        reducer: Callable[[Any], Array],
        /,
        *,
        observable_id: str,
        exact: bool,
    ):
        if not callable(reducer):
            raise TypeError("Observable reducer must be callable.")
        self.reducer = reducer
        self.observable_id = str(observable_id)
        self.exact = bool(exact)

    def __call__(self, state: Any, /) -> Array:
        return jnp.asarray(self.reducer(state))


class OpenSystemRefinement(StrictModule):
    coarse_representation_id: str = eqx.field(static=True)
    fine_representation_id: str = eqx.field(static=True)
    axis: ApproximationAxis
    state_embedding: Callable[[Any], Any]

    def __init__(
        self,
        coarse_representation_id: str,
        fine_representation_id: str,
        axis: ApproximationAxis,
        state_embedding: Callable[[Any], Any],
        /,
    ):
        if not callable(state_embedding):
            raise TypeError("state_embedding must be callable.")
        self.coarse_representation_id = str(coarse_representation_id)
        self.fine_representation_id = str(fine_representation_id)
        self.axis = axis
        self.state_embedding = state_embedding

    def embed(self, state: Any, /) -> Any:
        return self.state_embedding(state)


class OpenSystemPromotionPolicy(StrictModule):
    required_axes: tuple[str, ...] = eqx.field(static=True)
    required_quantities: tuple[str, ...] = eqx.field(static=True)
    require_physicality: bool = eqx.field(static=True)
    require_precision: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        required_axes: Sequence[str],
        required_quantities: Sequence[str],
        /,
        *,
        require_physicality: bool,
        require_precision: bool = True,
        policy_id: str,
    ):
        self.required_axes = tuple(str(value) for value in required_axes)
        self.required_quantities = tuple(str(value) for value in required_quantities)
        self.require_physicality = bool(require_physicality)
        self.require_precision = bool(require_precision)
        self.policy_id = str(policy_id)


class OpenSystemPromotionDecision(StrictModule):
    promoted: Array
    missing_axes: tuple[str, ...] = eqx.field(static=True)
    missing_quantities: tuple[str, ...] = eqx.field(static=True)
    physicality_satisfied: Array
    archive_verified: Array
    capacity_available: Array
    precision_satisfied: Array
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        promoted: ArrayLike,
        missing_axes: Sequence[str],
        missing_quantities: Sequence[str],
        physicality_satisfied: ArrayLike,
        archive_verified: ArrayLike,
        capacity_available: ArrayLike,
        precision_satisfied: ArrayLike,
        /,
        *,
        policy_id: str,
    ):
        self.promoted = jnp.asarray(promoted, dtype=bool)
        self.missing_axes = tuple(missing_axes)
        self.missing_quantities = tuple(missing_quantities)
        self.physicality_satisfied = jnp.asarray(physicality_satisfied, dtype=bool)
        self.archive_verified = jnp.asarray(archive_verified, dtype=bool)
        self.capacity_available = jnp.asarray(capacity_available, dtype=bool)
        self.precision_satisfied = jnp.asarray(precision_satisfied, dtype=bool)
        self.policy_id = str(policy_id)


def evaluate_open_system_promotion(
    policy: OpenSystemPromotionPolicy,
    approximation: OpenSystemApproximationEvidence,
    physicality: OpenSystemPhysicalityEvidence,
    /,
    *,
    execution_success: ArrayLike,
    capacity_exhausted: ArrayLike,
    archive_verified: ArrayLike,
) -> OpenSystemPromotionDecision:
    axes = {axis.name for axis in approximation.axes}
    quantities = {quantity.name for quantity in approximation.quantities}
    missing_axes = tuple(name for name in policy.required_axes if name not in axes)
    missing_quantities = tuple(
        name for name in policy.required_quantities if name not in quantities
    )
    physicality_satisfied = (
        physicality.valid if policy.require_physicality else jnp.asarray(True)
    )
    precision_satisfied = (
        jnp.asarray(
            approximation.precision_evidence is not None
            and bool(approximation.precision_policy_ids)
        )
        if policy.require_precision
        else jnp.asarray(True)
    )
    capacity_available = ~jnp.asarray(capacity_exhausted, dtype=bool)
    promoted = (
        jnp.asarray(execution_success, dtype=bool)
        & approximation.valid
        & physicality_satisfied
        & capacity_available
        & precision_satisfied
        & jnp.asarray(archive_verified, dtype=bool)
        & (len(missing_axes) == 0)
        & (len(missing_quantities) == 0)
    )
    return OpenSystemPromotionDecision(
        promoted,
        missing_axes,
        missing_quantities,
        physicality_satisfied,
        archive_verified,
        capacity_available,
        precision_satisfied,
        policy_id=policy.policy_id,
    )


__all__ = [
    "ApproximationQuantity",
    "ApproximationAxis",
    "OpenSystemPromotionDecision",
    "OpenSystemPromotionPolicy",
    "evaluate_open_system_promotion",
    "OpenSystemApproximationEvidence",
    "OpenSystemPhysicalityEvidence",
    "OpenSystemRefinement",
    "PhysicalityStatus",
    "QuantumGeneratorAction",
    "QuantumObservablePlan",
]
