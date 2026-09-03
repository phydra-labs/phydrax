#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._precision import PrecisionEvidenceEnvelope
from ..._strict import StrictModule
from ._precision import FiniteVolumePrecisionPolicy


if TYPE_CHECKING:
    from ...equations import ConvexEntropyPair


class FiniteVolumeEntropyDiagnostics(StrictModule):
    """Volume-weighted entropy evidence for one finite-volume residual."""

    pair_id: str = eqx.field(static=True)
    total_entropy: Array
    semidiscrete_entropy_rate: Array
    source_entropy_rate: Array
    convective_entropy_rate: Array
    admissible: Array
    precision_evidence: PrecisionEvidenceEnvelope

    def __init__(
        self,
        *,
        pair_id: str,
        total_entropy: ArrayLike,
        semidiscrete_entropy_rate: ArrayLike,
        source_entropy_rate: ArrayLike,
        convective_entropy_rate: ArrayLike,
        admissible: ArrayLike,
        precision_evidence: PrecisionEvidenceEnvelope,
    ):
        identifier = str(pair_id)
        if not identifier:
            raise ValueError("pair_id must be non-empty.")
        if not isinstance(precision_evidence, PrecisionEvidenceEnvelope):
            raise TypeError("precision_evidence must be a PrecisionEvidenceEnvelope.")
        total = jnp.asarray(total_entropy)
        semidiscrete = jnp.asarray(semidiscrete_entropy_rate)
        source = jnp.asarray(source_entropy_rate)
        convective = jnp.asarray(convective_entropy_rate)
        for value, name in (
            (total, "total_entropy"),
            (semidiscrete, "semidiscrete_entropy_rate"),
            (source, "source_entropy_rate"),
            (convective, "convective_entropy_rate"),
        ):
            if value.shape != ():
                raise ValueError(f"{name} must be scalar; got {value.shape}.")
        admissible_ = jnp.asarray(admissible, dtype=bool)
        if admissible_.shape != ():
            raise ValueError("admissible must be scalar.")
        self.pair_id = identifier
        self.total_entropy = total
        self.semidiscrete_entropy_rate = semidiscrete
        self.source_entropy_rate = source
        self.convective_entropy_rate = convective
        self.admissible = admissible_
        self.precision_evidence = precision_evidence


def _precision_policy(
    state: Array,
    precision: FiniteVolumePrecisionPolicy | None,
    /,
) -> FiniteVolumePrecisionPolicy:
    policy = (
        FiniteVolumePrecisionPolicy(jnp.dtype(state.dtype).name)
        if precision is None
        else precision
    )
    if not isinstance(policy, FiniteVolumePrecisionPolicy):
        raise TypeError("precision must be a FiniteVolumePrecisionPolicy or None.")
    return policy


def _validated_arrays(
    pair: "ConvexEntropyPair",
    state: ArrayLike,
    effective_volumes: ArrayLike,
    /,
    *rates: ArrayLike,
) -> tuple[Array, Array, tuple[Array, ...]]:
    from ...equations import ConvexEntropyPair

    if not isinstance(pair, ConvexEntropyPair):
        raise TypeError("pair must be a ConvexEntropyPair.")
    state_ = jnp.asarray(state)
    if state_.ndim < 1 or state_.shape[-1] != pair.component_count:
        raise ValueError(
            f"state must end in the entropy-pair component count; got {state_.shape}."
        )
    if not jnp.issubdtype(state_.dtype, jnp.floating):
        raise TypeError("state must use real floating-point coordinates.")
    volumes = jnp.asarray(effective_volumes)
    if volumes.shape != state_.shape[:-1]:
        raise ValueError(
            "effective_volumes must match the state leading cell shape; "
            f"got {volumes.shape} for {state_.shape}."
        )
    volumes = eqx.error_if(
        volumes,
        jnp.any(~jnp.isfinite(volumes) | (volumes < 0.0)),
        "effective_volumes must be finite and nonnegative.",
    )
    rate_arrays = tuple(jnp.asarray(rate) for rate in rates)
    if any(rate.shape != state_.shape for rate in rate_arrays):
        raise ValueError("finite-volume entropy rates must match the state shape.")
    return state_, volumes, rate_arrays


def _evaluate_finite_volume_entropy_diagnostics(
    pair: "ConvexEntropyPair",
    state: ArrayLike,
    effective_volumes: ArrayLike,
    convective_residual: ArrayLike,
    source_residual: ArrayLike,
    /,
    *,
    precision: FiniteVolumePrecisionPolicy | None = None,
) -> FiniteVolumeEntropyDiagnostics:
    state_, volumes, (convective, source) = _validated_arrays(
        pair,
        state,
        effective_volumes,
        convective_residual,
        source_residual,
    )
    policy = _precision_policy(state_, precision)
    policy.validate_state(state_)
    entropy_state = policy.flux(state_)
    entropy_variables = policy.reduction(pair.entropy_variables(entropy_state))
    convective_density = ein.contract(
        "...i,...i->...",
        entropy_variables,
        policy.reduction(convective),
    )
    source_density = ein.contract(
        "...i,...i->...",
        entropy_variables,
        policy.reduction(source),
    )
    volumes_ = policy.reduction(volumes)
    convective_rate = jnp.sum(
        policy.reduction(volumes_ * policy.reduction(convective_density))
    )
    source_rate = jnp.sum(policy.reduction(volumes_ * policy.reduction(source_density)))
    semidiscrete_rate = policy.reduction(convective_rate + source_rate)
    total_entropy = jnp.sum(
        policy.reduction(volumes_ * policy.reduction(pair.entropy(entropy_state)))
    )
    return FiniteVolumeEntropyDiagnostics(
        pair_id=pair.pair_id,
        total_entropy=policy.decision(total_entropy),
        semidiscrete_entropy_rate=policy.decision(semidiscrete_rate),
        source_entropy_rate=policy.decision(source_rate),
        convective_entropy_rate=policy.decision(convective_rate),
        admissible=jnp.all(pair.admissible(entropy_state)),
        precision_evidence=policy.evidence(),
    )


class FiniteVolumeEntropyProductionDiagnostics(StrictModule):
    """Content-form entropy balance including resolved viscous mechanisms."""

    pair_id: str = eqx.field(static=True)
    total_entropy: Array
    semidiscrete_entropy_rate: Array
    convective_entropy_rate: Array
    source_entropy_rate: Array
    geometric_entropy_rate: Array
    shear_entropy_production: Array
    bulk_entropy_production: Array
    thermal_entropy_production: Array
    admissible: Array
    precision_evidence: PrecisionEvidenceEnvelope


def evaluate_content_form_entropy_diagnostics(
    pair: "ConvexEntropyPair",
    state: ArrayLike,
    effective_volumes: ArrayLike,
    volume_rate: ArrayLike,
    convective_content_rate: ArrayLike,
    source_content_rate: ArrayLike,
    shear_content_rate: ArrayLike,
    bulk_content_rate: ArrayLike,
    thermal_content_rate: ArrayLike,
    /,
    *,
    precision: FiniteVolumePrecisionPolicy | None = None,
) -> FiniteVolumeEntropyProductionDiagnostics:
    """Evaluate ALE/cut/overset entropy rates from conservative content rates."""
    state_, volumes, rates = _validated_arrays(
        pair,
        state,
        effective_volumes,
        convective_content_rate,
        source_content_rate,
        shear_content_rate,
        bulk_content_rate,
        thermal_content_rate,
    )
    volume_rate_ = jnp.asarray(volume_rate)
    if volume_rate_.shape != volumes.shape:
        raise ValueError("volume_rate must match effective_volumes.")
    policy = _precision_policy(state_, precision)
    policy.validate_state(state_)
    entropy_state = policy.flux(state_)
    variables = policy.reduction(pair.entropy_variables(entropy_state))
    entropy = policy.reduction(pair.entropy(entropy_state))
    state_reduction = policy.reduction(entropy_state)
    volume_reduction = policy.reduction(volumes)

    def integrated(rate: Array, /) -> Array:
        return jnp.sum(
            policy.reduction(
                ein.contract(
                    "...i,...i->...",
                    variables,
                    policy.reduction(rate),
                    backend="jax",
                )
            )
        )

    convective, source, shear, bulk, thermal = rates
    convective_rate = integrated(convective)
    source_rate = integrated(source)
    shear_rate = integrated(shear)
    bulk_rate = integrated(bulk)
    thermal_rate = integrated(thermal)
    entropy_potential = entropy - ein.contract(
        "...i,...i->...", variables, state_reduction, backend="jax"
    )
    geometric_rate = jnp.sum(
        policy.reduction(entropy_potential * policy.reduction(volume_rate_))
    )
    semidiscrete = (
        convective_rate
        + source_rate
        + geometric_rate
        + shear_rate
        + bulk_rate
        + thermal_rate
    )
    total = jnp.sum(policy.reduction(volume_reduction * entropy))
    return FiniteVolumeEntropyProductionDiagnostics(
        pair_id=pair.pair_id,
        total_entropy=policy.decision(total),
        semidiscrete_entropy_rate=policy.decision(semidiscrete),
        convective_entropy_rate=policy.decision(convective_rate),
        source_entropy_rate=policy.decision(source_rate),
        geometric_entropy_rate=policy.decision(geometric_rate),
        shear_entropy_production=policy.decision(shear_rate),
        bulk_entropy_production=policy.decision(bulk_rate),
        thermal_entropy_production=policy.decision(thermal_rate),
        admissible=jnp.all(pair.admissible(entropy_state)),
        precision_evidence=policy.evidence(),
    )


def integrated_finite_volume_relative_entropy(
    pair: "ConvexEntropyPair",
    state: ArrayLike,
    reference: ArrayLike,
    effective_volumes: ArrayLike,
    /,
    *,
    precision: FiniteVolumePrecisionPolicy | None = None,
) -> Array:
    """Return volume-weighted relative entropy over finite-volume cells."""
    state_, volumes, (reference_,) = _validated_arrays(
        pair,
        state,
        effective_volumes,
        reference,
    )
    policy = _precision_policy(state_, precision)
    policy.validate_state(state_)
    policy.validate_state(reference_)
    relative = pair.relative_entropy(
        policy.flux(state_),
        policy.flux(reference_),
    )
    return policy.decision(
        jnp.sum(policy.reduction(policy.reduction(volumes) * policy.reduction(relative)))
    )


__all__ = [
    "FiniteVolumeEntropyDiagnostics",
    "FiniteVolumeEntropyProductionDiagnostics",
    "_evaluate_finite_volume_entropy_diagnostics",
    "evaluate_content_form_entropy_diagnostics",
    "integrated_finite_volume_relative_entropy",
]
