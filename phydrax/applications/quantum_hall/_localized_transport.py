#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Calibrated localized-state Hall transport as a conservative rate network."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import DenseLinearOperator, DenseLU, LinearSolvePolicy, LinearSystem, solve


class LocalizedHallNetworkPlan(StrictModule, NonTrainableState):
    transition_rates: Array
    contact_injection: Array
    contact_extraction: Array
    detailed_balance_weights: Array
    residual_tolerance: float = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transition_rates: ArrayLike,
        contact_injection: ArrayLike,
        contact_extraction: ArrayLike,
        detailed_balance_weights: ArrayLike,
        model_id: str,
        /,
        *,
        residual_tolerance: float = 1.0e-10,
    ):
        rates = np.asarray(transition_rates, dtype=np.float64)
        injection = np.asarray(contact_injection, dtype=np.float64)
        extraction = np.asarray(contact_extraction, dtype=np.float64)
        weights = np.asarray(detailed_balance_weights, dtype=np.float64)
        identifier = str(model_id).strip()
        tolerance = float(residual_tolerance)
        if (
            rates.ndim != 2
            or rates.shape[0] != rates.shape[1]
            or injection.ndim != 2
            or injection.shape[1] != rates.shape[0]
            or extraction.shape != injection.shape
            or weights.shape != (rates.shape[0],)
            or np.any(~np.isfinite(rates))
            or np.any(~np.isfinite(injection))
            or np.any(~np.isfinite(extraction))
            or np.any(rates < 0.0)
            or np.any(injection < 0.0)
            or np.any(extraction < 0.0)
            or np.any(weights <= 0.0)
            or not identifier
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError(
                "Localized transport rates, contacts, weights, or identity are invalid."
            )
        np.fill_diagonal(rates, 0.0)
        self.transition_rates = jnp.asarray(rates)
        self.contact_injection = jnp.asarray(injection)
        self.contact_extraction = jnp.asarray(extraction)
        self.detailed_balance_weights = jnp.asarray(weights / np.sum(weights))
        self.residual_tolerance = tolerance
        self.model_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "localized-hall-network-plan",
                "arrays": array_tree_fingerprint(
                    {
                        "rates": rates,
                        "injection": injection,
                        "extraction": extraction,
                        "weights": weights,
                    }
                ),
                "model_id": identifier,
                "residual_tolerance": tolerance,
            }
        )


class LocalizedHallTransportResult(StrictModule, NonTrainableState):
    populations: Array
    contact_currents: Array
    stationary_residual: Array
    detailed_balance_residual: Array
    current_conservation_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def solve_localized_hall_transport(
    plan: LocalizedHallNetworkPlan,
    /,
) -> LocalizedHallTransportResult:
    if not isinstance(plan, LocalizedHallNetworkPlan):
        raise TypeError("plan must be LocalizedHallNetworkPlan.")
    rates = plan.transition_rates
    generator = rates.T - jnp.diag(jnp.sum(rates, axis=1))
    generator = generator - jnp.diag(jnp.sum(plan.contact_extraction, axis=0))
    source = jnp.sum(plan.contact_injection, axis=0)
    matrix = generator.at[-1, :].set(1.0)
    right = source.at[-1].set(1.0)
    populations = solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    ).value
    contact_currents = jnp.sum(
        plan.contact_injection - plan.contact_extraction * populations[None, :],
        axis=1,
    )
    residual = jnp.max(jnp.abs(generator @ populations + source))
    detailed = jnp.max(
        jnp.abs(
            rates * plan.detailed_balance_weights[:, None]
            - rates.T * plan.detailed_balance_weights[None, :]
        )
    )
    conservation = jnp.abs(jnp.sum(contact_currents))
    successful = (
        jnp.all(jnp.isfinite(populations))
        & jnp.all(populations >= -plan.residual_tolerance)
        & (residual <= plan.residual_tolerance)
        & (conservation <= plan.residual_tolerance)
    )
    return LocalizedHallTransportResult(
        populations,
        contact_currents,
        residual,
        detailed,
        conservation,
        successful,
        plan.plan_id,
        canonical_fingerprint(
            {"kind": "localized-hall-transport-result", "plan": plan.plan_id}
        ),
    )


__all__ = [
    "LocalizedHallNetworkPlan",
    "LocalizedHallTransportResult",
    "solve_localized_hall_transport",
]
