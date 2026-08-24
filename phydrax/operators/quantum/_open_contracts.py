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
        units_ = str(units)
        value_ = jnp.asarray(value)
        parent = None if parent_value is None else jnp.asarray(parent_value)
        if not identifier:
            raise ValueError("Approximation-axis name must be non-empty.")
        if not units_:
            raise ValueError("Approximation-axis units must be non-empty.")
        if value_.shape != () or not bool(jnp.isfinite(value_)):
            raise ValueError("Approximation-axis value must be one finite scalar.")
        if parent is not None and (
            parent.shape != () or not bool(jnp.isfinite(parent))
        ):
            raise ValueError(
                "Approximation-axis parent value must be one finite scalar or None."
            )
        self.name = identifier
        self.value = value_
        self.parent_value = parent
        self.units = units_


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
        identifier = str(name)
        units_ = str(units)
        norm = str(norm_id)
        if not identifier or not units_ or not norm:
            raise ValueError(
                "Approximation quantity name, units, and norm_id must be non-empty."
            )
        if estimate_kind not in ("bound", "estimate", "statistical"):
            raise ValueError("Unknown approximation quantity kind.")
        value_ = jnp.asarray(value)
        threshold_ = jnp.asarray(threshold)
        confidence_ = jnp.asarray(confidence)
        if (
            value_.shape != ()
            or threshold_.shape != ()
            or not bool(jnp.isfinite(value_))
            or not bool(jnp.isfinite(threshold_))
            or bool(value_ < 0.0)
            or bool(threshold_ < 0.0)
        ):
            raise ValueError(
                "Approximation values and thresholds must be finite non-negative scalars."
            )
        if estimate_kind == "statistical" and (
            confidence_.shape != ()
            or not bool(jnp.isfinite(confidence_))
            or not bool((confidence_ > 0.0) & (confidence_ < 1.0))
        ):
            raise ValueError(
                "Statistical approximation quantities require confidence in (0, 1)."
            )
        self.name = identifier
        self.value = value_
        self.threshold = threshold_
        self.units = units_
        self.norm_id = norm
        self.estimate_kind = estimate_kind
        self.confidence = confidence_
        self.valid = value_ <= threshold_


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
        identifier = str(representation_id)
        axes_ = tuple(axes)
        quantities_ = tuple(quantities)
        execution = jnp.asarray(execution_valid, dtype=bool)
        if not identifier:
            raise ValueError("representation_id must be non-empty.")
        if not axes_ or not quantities_:
            raise ValueError(
                "Approximation evidence requires declared axes and quantities."
            )
        if any(not isinstance(axis, ApproximationAxis) for axis in axes_):
            raise TypeError("axes must contain ApproximationAxis values.")
        if any(
            not isinstance(quantity, ApproximationQuantity)
            for quantity in quantities_
        ):
            raise TypeError(
                "quantities must contain ApproximationQuantity values."
            )
        axis_names = tuple(axis.name for axis in axes_)
        quantity_names = tuple(quantity.name for quantity in quantities_)
        if len(set(axis_names)) != len(axis_names):
            raise ValueError("Approximation-axis names must be unique.")
        if len(set(quantity_names)) != len(quantity_names):
            raise ValueError("Approximation-quantity names must be unique.")
        if execution.shape != ():
            raise ValueError("execution_valid must be one scalar Boolean.")
        if precision_evidence is not None and not isinstance(
            precision_evidence, PrecisionEvidenceEnvelope
        ):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        policy_ids = tuple(str(identifier) for identifier in precision_policy_ids)
        if any(not policy_id for policy_id in policy_ids) or len(
            set(policy_ids)
        ) != len(policy_ids):
            raise ValueError("Precision policy IDs must be unique and non-empty.")
        self.representation_id = identifier
        self.axes = axes_
        self.quantities = quantities_
        self.valid = execution & jnp.all(
            jnp.stack([quantity.valid for quantity in quantities_])
        )
        self.precision_evidence = precision_evidence
        self.precision_policy_ids = policy_ids


class OpenSystemPhysicalityEvidence(StrictModule):
    trace_residual: Array
    hermiticity_residual: Array
    positivity_margin: Array
    channel_cp_margin: Array
    trace_preservation_residual: Array
    closure_residual: Array
    trace_tolerance: Array
    hermiticity_tolerance: Array
    positivity_tolerance: Array
    channel_cp_tolerance: Array
    trace_preservation_tolerance: Array
    closure_tolerance: Array
    valid: Array
    status: PhysicalityStatus = eqx.field(static=True)
    certified_properties: tuple[str, ...] = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope | None = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        trace_residual: ArrayLike = jnp.nan,
        hermiticity_residual: ArrayLike = jnp.nan,
        positivity_margin: ArrayLike = jnp.nan,
        channel_cp_margin: ArrayLike = jnp.nan,
        trace_preservation_residual: ArrayLike = jnp.nan,
        closure_residual: ArrayLike = jnp.nan,
        trace_tolerance: ArrayLike = 1e-6,
        hermiticity_tolerance: ArrayLike = 1e-6,
        positivity_tolerance: ArrayLike = 1e-8,
        channel_cp_tolerance: ArrayLike = 1e-8,
        trace_preservation_tolerance: ArrayLike = 1e-6,
        closure_tolerance: ArrayLike = 1e-6,
        certified_properties: Sequence[str] = (),
        precision_evidence: PrecisionEvidenceEnvelope | None = None,
    ):
        if precision_evidence is not None and not isinstance(
            precision_evidence,
            PrecisionEvidenceEnvelope,
        ):
            raise TypeError(
                "precision_evidence must be PrecisionEvidenceEnvelope or None."
            )
        allowed = {
            "trace",
            "hermiticity",
            "positivity",
            "complete-positivity",
            "trace-preservation",
            "representation-closure",
        }
        properties = tuple(str(value) for value in certified_properties)
        if (
            any(value not in allowed for value in properties)
            or len(set(properties)) != len(properties)
        ):
            raise ValueError("Unknown or duplicate physicality property.")
        metrics = {
            "trace": jnp.asarray(trace_residual),
            "hermiticity": jnp.asarray(hermiticity_residual),
            "positivity": jnp.asarray(positivity_margin),
            "complete-positivity": jnp.asarray(channel_cp_margin),
            "trace-preservation": jnp.asarray(trace_preservation_residual),
            "representation-closure": jnp.asarray(closure_residual),
        }
        tolerances = {
            "trace": jnp.asarray(trace_tolerance),
            "hermiticity": jnp.asarray(hermiticity_tolerance),
            "positivity": jnp.asarray(positivity_tolerance),
            "complete-positivity": jnp.asarray(channel_cp_tolerance),
            "trace-preservation": jnp.asarray(trace_preservation_tolerance),
            "representation-closure": jnp.asarray(closure_tolerance),
        }
        for name, tolerance in tolerances.items():
            if (
                tolerance.shape != ()
                or not bool(jnp.isfinite(tolerance))
                or bool(tolerance < 0.0)
            ):
                raise ValueError(
                    f"Physicality tolerance {name!r} must be finite and non-negative."
                )
        for name in properties:
            metric = metrics[name]
            if metric.shape != () or not bool(jnp.isfinite(metric)):
                raise ValueError(
                    f"Certified physicality metric {name!r} must be finite and scalar."
                )
            if name not in ("positivity", "complete-positivity") and bool(
                metric < 0.0
            ):
                raise ValueError(
                    f"Physicality residual {name!r} must be non-negative."
                )
        checks = []
        for name in properties:
            metric = metrics[name]
            tolerance = tolerances[name]
            checks.append(
                metric >= -tolerance
                if name in ("positivity", "complete-positivity")
                else metric <= tolerance
            )
        valid = bool(properties) and bool(jnp.all(jnp.stack(checks)))
        self.trace_residual = metrics["trace"]
        self.hermiticity_residual = metrics["hermiticity"]
        self.positivity_margin = metrics["positivity"]
        self.channel_cp_margin = metrics["complete-positivity"]
        self.trace_preservation_residual = metrics["trace-preservation"]
        self.closure_residual = metrics["representation-closure"]
        self.trace_tolerance = tolerances["trace"]
        self.hermiticity_tolerance = tolerances["hermiticity"]
        self.positivity_tolerance = tolerances["positivity"]
        self.channel_cp_tolerance = tolerances["complete-positivity"]
        self.trace_preservation_tolerance = tolerances["trace-preservation"]
        self.closure_tolerance = tolerances["representation-closure"]
        self.certified_properties = properties
        self.status = "valid" if valid else ("invalid" if properties else "unknown")
        self.valid = jnp.asarray(valid)
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
    required_physicality: tuple[str, ...] = eqx.field(static=True)
    require_precision: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        required_axes: Sequence[str],
        required_quantities: Sequence[str],
        required_physicality: Sequence[str],
        /,
        *,
        require_precision: bool = True,
        policy_id: str,
    ):
        axes = tuple(str(value) for value in required_axes)
        quantities = tuple(str(value) for value in required_quantities)
        physicality = tuple(str(value) for value in required_physicality)
        identifier = str(policy_id)
        for label, values in (
            ("required_axes", axes),
            ("required_quantities", quantities),
            ("required_physicality", physicality),
        ):
            if any(not value for value in values) or len(set(values)) != len(values):
                raise ValueError(f"{label} must contain unique non-empty names.")
        if not identifier:
            raise ValueError("policy_id must be non-empty.")
        self.required_axes = axes
        self.required_quantities = quantities
        self.required_physicality = physicality
        self.require_precision = bool(require_precision)
        self.policy_id = identifier


class OpenSystemPromotionDecision(StrictModule):
    promoted: Array
    missing_axes: tuple[str, ...] = eqx.field(static=True)
    missing_quantities: tuple[str, ...] = eqx.field(static=True)
    missing_physicality: tuple[str, ...] = eqx.field(static=True)
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
        missing_physicality: Sequence[str],
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
        self.missing_physicality = tuple(missing_physicality)
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
    certified = set(physicality.certified_properties)
    missing_axes = tuple(name for name in policy.required_axes if name not in axes)
    missing_quantities = tuple(
        name for name in policy.required_quantities if name not in quantities
    )
    missing_physicality = tuple(
        name for name in policy.required_physicality if name not in certified
    )
    physicality_satisfied = physicality.valid & (len(missing_physicality) == 0)
    precision_satisfied = (
        jnp.asarray(
            approximation.precision_evidence is not None
            and bool(approximation.precision_policy_ids)
        )
        if policy.require_precision
        else jnp.asarray(True)
    )
    execution = jnp.asarray(execution_success, dtype=bool)
    exhausted = jnp.asarray(capacity_exhausted, dtype=bool)
    archived = jnp.asarray(archive_verified, dtype=bool)
    if execution.shape != () or exhausted.shape != () or archived.shape != ():
        raise ValueError("Promotion execution, capacity, and archive gates are scalar.")
    capacity_available = ~exhausted
    promoted = (
        execution
        & approximation.valid
        & physicality_satisfied
        & capacity_available
        & precision_satisfied
        & archived
        & (len(missing_axes) == 0)
        & (len(missing_quantities) == 0)
    )
    return OpenSystemPromotionDecision(
        promoted,
        missing_axes,
        missing_quantities,
        missing_physicality,
        physicality_satisfied,
        archived,
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
