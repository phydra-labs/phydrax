#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class CoherentAmplitudePlan(StrictModule, NonTrainableState):
    component_names: tuple[str, ...] = eqx.field(static=True)
    phase_convention_id: str = eqx.field(static=True)
    normalization_evidence_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_names: Sequence[str],
        /,
        *,
        phase_convention_id: str,
        normalization_evidence_id: str,
    ):
        names = tuple(str(value).strip() for value in component_names)
        phase = str(phase_convention_id).strip()
        evidence = str(normalization_evidence_id).strip()
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
            or not phase
            or not evidence
        ):
            raise ValueError(
                "Amplitude components, phase convention, and normalization evidence are required."
            )
        self.component_names = names
        self.phase_convention_id = phase
        self.normalization_evidence_id = evidence
        self.plan_id = canonical_fingerprint(
            {
                "kind": "coherent-flavor-amplitude-plan",
                "components": list(names),
                "phase_convention": phase,
                "normalization_evidence": evidence,
            }
        )


class CoherentAmplitudeResult(StrictModule, NonTrainableState):
    total_amplitude: Array
    intensity: Array
    normalization: Array
    normalized_density: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def evaluate_coherent_amplitude(
    plan: CoherentAmplitudePlan,
    component_amplitudes: ArrayLike,
    complex_coefficients: ArrayLike,
    integration_weights: ArrayLike,
    /,
) -> CoherentAmplitudeResult:
    """Compose externally normalized line shapes with explicit coherent interference."""
    if not isinstance(plan, CoherentAmplitudePlan):
        raise TypeError("plan must be CoherentAmplitudePlan.")
    components = jnp.asarray(component_amplitudes)
    coefficients = jnp.asarray(complex_coefficients, dtype=components.dtype)
    integration = jnp.asarray(integration_weights, dtype=components.real.dtype)
    if (
        components.ndim != 2
        or components.shape[1] != len(plan.component_names)
        or coefficients.shape != (len(plan.component_names),)
        or integration.shape != (components.shape[0],)
    ):
        raise ValueError(
            "Amplitude components, coefficients, and integration weights do not align."
        )
    total = ein.contract("ec,c->e", components, coefficients)
    intensity = jnp.real(total * jnp.conj(total))
    normalization = jnp.sum(jnp.where(integration >= 0.0, integration * intensity, 0.0))
    density = intensity / jnp.maximum(normalization, jnp.finfo(intensity.dtype).tiny)
    finite = jnp.all(jnp.isfinite(total)) & jnp.isfinite(normalization)
    valid = finite & (normalization > 0.0) & jnp.all(integration >= 0.0)
    return CoherentAmplitudeResult(
        total,
        intensity,
        normalization,
        jnp.where(valid, density, jnp.nan),
        finite,
        valid,
        plan.plan_id,
    )


class AmplitudeInterference(StrictModule, NonTrainableState):
    matrix: Array
    fractions: Array
    sum_fraction: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def amplitude_interference_fractions(
    plan: CoherentAmplitudePlan,
    component_amplitudes: ArrayLike,
    complex_coefficients: ArrayLike,
    integration_weights: ArrayLike,
    /,
) -> AmplitudeInterference:
    if not isinstance(plan, CoherentAmplitudePlan):
        raise TypeError("plan must be CoherentAmplitudePlan.")
    components = jnp.asarray(component_amplitudes)
    coefficients = jnp.asarray(complex_coefficients, dtype=components.dtype)
    weights = jnp.asarray(integration_weights, dtype=components.real.dtype)
    weighted_components = components * coefficients[None, :]
    matrix = ein.contract(
        "e,ei,ej->ij", weights, weighted_components, jnp.conj(weighted_components)
    )
    matrix = jnp.real(matrix)
    total = jnp.sum(matrix)
    fractions = jnp.diag(matrix) / jnp.maximum(total, jnp.finfo(matrix.dtype).tiny)
    valid = jnp.all(jnp.isfinite(matrix)) & (total > 0.0) & jnp.all(weights >= 0.0)
    return AmplitudeInterference(
        matrix, fractions, jnp.sum(fractions), valid, plan.plan_id
    )


__all__ = [
    "AmplitudeInterference",
    "CoherentAmplitudePlan",
    "CoherentAmplitudeResult",
    "amplitude_interference_fractions",
    "evaluate_coherent_amplitude",
]
