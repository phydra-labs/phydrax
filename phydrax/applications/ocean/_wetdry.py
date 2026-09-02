#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._hydrostatic import HydrostaticOceanState, PreparedHydrostaticOcean


class WetDryEpochPolicy(StrictModule, NonTrainableState):
    wet_depth: float = eqx.field(static=True)
    dry_depth: float = eqx.field(static=True)
    grazing_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        wet_depth: float,
        dry_depth: float,
        grazing_tolerance: float = 1.0e-10,
    ):
        wet = float(wet_depth)
        dry = float(dry_depth)
        tolerance = float(grazing_tolerance)
        if (
            not np.isfinite(wet)
            or not np.isfinite(dry)
            or not np.isfinite(tolerance)
            or wet <= dry
            or dry < 0.0
            or tolerance <= 0.0
        ):
            raise ValueError("Wet/dry epoch thresholds are invalid.")
        self.wet_depth = wet
        self.dry_depth = dry
        self.grazing_tolerance = tolerance
        self.policy_id = canonical_fingerprint(
            {
                "kind": "hydrostatic-wet-dry-epoch-policy",
                "wet_depth": wet,
                "dry_depth": dry,
                "grazing_tolerance": tolerance,
            }
        )


class WetDryTransitionEvidence(StrictModule):
    previous_wet: Array
    current_wet: Array
    activated: Array
    deactivated: Array
    changed: Array
    event_count: Array
    retained_dry_inventory: dict[str, Array]
    retained_dry_tke: Array
    grazing: Array
    topology_changed: Array
    derivative_available: Array
    finite: Array
    successful: Array
    event_id: str = eqx.field(static=True)


class HydrostaticWetDrySensitivityResult(StrictModule):
    state: HydrostaticOceanState
    eta_tangent: Array
    evidence: WetDryTransitionEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class HydrostaticWetDryEventPlan(StrictModule, NonTrainableState):
    policy: WetDryEpochPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(self, policy: WetDryEpochPolicy, /):
        if not isinstance(policy, WetDryEpochPolicy):
            raise TypeError("policy must be a WetDryEpochPolicy.")
        self.policy = policy
        self.plan_id = canonical_fingerprint(
            {"kind": "hydrostatic-wet-dry-event", "policy": policy.policy_id}
        )

    def transition(
        self,
        ocean: PreparedHydrostaticOcean,
        previous: HydrostaticOceanState,
        candidate: HydrostaticOceanState,
        /,
        *,
        eta_tangent: ArrayLike | None = None,
    ) -> HydrostaticWetDrySensitivityResult:
        if not isinstance(ocean, PreparedHydrostaticOcean):
            raise TypeError("ocean must be a PreparedHydrostaticOcean.")
        if not ocean.plan.wetting_and_drying:
            raise ValueError("Ocean plan has no wetting-and-drying event semantics.")
        previous_depth = ocean.geometry.rest_depth + previous.eta
        candidate_depth = ocean.geometry.rest_depth + candidate.eta
        previous_wet = previous_depth > self.policy.dry_depth
        current_wet = jnp.where(
            previous_wet,
            candidate_depth > self.policy.dry_depth,
            candidate_depth >= self.policy.wet_depth,
        )
        activated = ~previous_wet & current_wet
        deactivated = previous_wet & ~current_wet
        changed = activated | deactivated
        event_count = jnp.sum(changed.astype(jnp.int32))
        threshold_distance = jnp.minimum(
            jnp.abs(candidate_depth - self.policy.wet_depth),
            jnp.abs(candidate_depth - self.policy.dry_depth),
        )
        grazing = jnp.any(changed & (threshold_distance <= self.policy.grazing_tolerance))
        epoch = ocean.geometry.metric_epoch(candidate.eta)
        transports = (
            jnp.where(epoch.active_x_face, candidate.transports[0], 0.0),
            jnp.where(epoch.active_y_face, candidate.transports[1], 0.0),
        )
        state = HydrostaticOceanState(
            candidate.eta,
            transports,
            candidate.tracer_inventory,
            candidate.tke_inventory,
        )
        retained_inventory = {
            name: jnp.sum(
                jnp.where(
                    deactivated[..., None],
                    value,
                    0.0,
                ),
                axis=-1,
            )
            for name, value in candidate.tracer_inventory.items()
        }
        retained_tke = jnp.sum(
            jnp.where(deactivated[..., None], candidate.tke_inventory, 0.0), axis=-1
        )
        tangent = (
            jnp.zeros_like(candidate.eta)
            if eta_tangent is None
            else jnp.asarray(eta_tangent, dtype=candidate.eta.dtype)
        )
        if tangent.shape != candidate.eta.shape:
            raise ValueError("eta_tangent must match the free-surface shape.")
        derivative_available = (event_count <= 1) & ~grazing
        saltated_tangent = jnp.where(
            derivative_available & changed,
            0.0,
            tangent,
        )
        inventory_finite = (
            jnp.asarray(True)
            if not retained_inventory
            else jnp.all(
                jnp.stack(
                    tuple(
                        jnp.all(jnp.isfinite(value))
                        for value in retained_inventory.values()
                    )
                )
            )
        )
        finite = (
            epoch.finite
            & jnp.all(jnp.isfinite(candidate_depth))
            & jnp.all(jnp.isfinite(saltated_tangent))
            & inventory_finite
            & jnp.all(jnp.isfinite(retained_tke))
        )
        successful = finite & epoch.valid
        evidence = WetDryTransitionEvidence(
            previous_wet=previous_wet,
            current_wet=current_wet,
            activated=activated,
            deactivated=deactivated,
            changed=changed,
            event_count=event_count,
            retained_dry_inventory=retained_inventory,
            retained_dry_tke=retained_tke,
            grazing=grazing,
            topology_changed=jnp.any(changed),
            derivative_available=derivative_available,
            finite=finite,
            successful=successful,
            event_id=canonical_fingerprint(
                {
                    "kind": "hydrostatic-wet-dry-transition",
                    "plan": self.plan_id,
                    "geometry": ocean.geometry.geometry_id,
                    "route": "fixed-mask-or-isolated-saltation",
                }
            ),
        )
        return HydrostaticWetDrySensitivityResult(
            state=state,
            eta_tangent=saltated_tangent,
            evidence=evidence,
            successful=successful,
            plan_id=self.plan_id,
        )


__all__ = [
    "HydrostaticWetDryEventPlan",
    "HydrostaticWetDrySensitivityResult",
    "WetDryEpochPolicy",
    "WetDryTransitionEvidence",
]
