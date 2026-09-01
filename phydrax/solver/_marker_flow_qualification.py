#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class MarkerFlowQualificationProfile(StrictModule, NonTrainableState):
    family: str = eqx.field(static=True)
    minimum_order: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    stochastic_tolerance: float = eqx.field(static=True)
    require_stochastic: bool = eqx.field(static=True)
    require_contact: bool = eqx.field(static=True)
    require_interface: bool = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        family: str,
        /,
        *,
        minimum_order: float = 0.9,
        residual_tolerance: float = 1.0e-8,
        conservation_tolerance: float = 1.0e-8,
        stochastic_tolerance: float = 0.1,
        require_stochastic: bool = False,
        require_contact: bool = False,
        require_interface: bool = False,
    ):
        family_ = str(family)
        values = np.asarray(
            (
                minimum_order,
                residual_tolerance,
                conservation_tolerance,
                stochastic_tolerance,
            )
        )
        if not family_ or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Marker-flow qualification profile is invalid.")
        self.family = family_
        self.minimum_order = float(minimum_order)
        self.residual_tolerance = float(residual_tolerance)
        self.conservation_tolerance = float(conservation_tolerance)
        self.stochastic_tolerance = float(stochastic_tolerance)
        self.require_stochastic = bool(require_stochastic)
        self.require_contact = bool(require_contact)
        self.require_interface = bool(require_interface)
        self.profile_id = canonical_fingerprint(
            {
                "kind": "marker-flow-qualification-profile",
                "family": family_,
                "minimum_order": minimum_order,
                "residual_tolerance": residual_tolerance,
                "conservation_tolerance": conservation_tolerance,
                "stochastic_tolerance": stochastic_tolerance,
                "require_stochastic": require_stochastic,
                "require_contact": require_contact,
                "require_interface": require_interface,
            }
        )


class MarkerFlowQualificationEvidence(StrictModule):
    divergence_norm: Array
    marker_slip_norm: Array
    force_residual: Array
    torque_residual: Array
    work_residual: Array
    energy_residual: Array
    spatial_order: Array
    temporal_order: Array
    stochastic_covariance_error: Array
    interface_error: Array
    lubrication_error: Array
    contact_residual: Array
    replay_residual: Array
    finite: Array


class MarkerFlowQualificationResult(StrictModule):
    evidence: MarkerFlowQualificationEvidence
    residual_passed: Array
    conservation_passed: Array
    convergence_passed: Array
    stochastic_passed: Array
    contact_passed: Array
    interface_passed: Array
    successful: Array
    profile_id: str = eqx.field(static=True)


def observed_convergence_order(
    resolution: ArrayLike,
    error: ArrayLike,
    /,
) -> Array:
    spacing = jnp.asarray(resolution)
    values = jnp.asarray(error, dtype=spacing.dtype)
    if spacing.ndim != 1 or values.shape != spacing.shape or spacing.size < 2:
        raise ValueError("Convergence evidence needs at least two aligned levels.")
    log_spacing = jnp.log(spacing)
    log_error = jnp.log(values)
    centered_spacing = log_spacing - jnp.mean(log_spacing)
    centered_error = log_error - jnp.mean(log_error)
    denominator = jnp.sum(centered_spacing**2)
    return jnp.sum(centered_spacing * centered_error) / denominator


class MarkerFlowQualificationPlan(StrictModule, NonTrainableState):
    profile: MarkerFlowQualificationProfile
    plan_id: str = eqx.field(static=True)

    def __init__(self, profile: MarkerFlowQualificationProfile, /):
        if not isinstance(profile, MarkerFlowQualificationProfile):
            raise TypeError("profile must be MarkerFlowQualificationProfile.")
        self.profile = profile
        self.plan_id = canonical_fingerprint(
            {"kind": "marker-flow-qualification", "profile": profile.profile_id}
        )

    def evaluate(
        self,
        evidence: MarkerFlowQualificationEvidence,
        /,
    ) -> MarkerFlowQualificationResult:
        if not isinstance(evidence, MarkerFlowQualificationEvidence):
            raise TypeError("evidence must be MarkerFlowQualificationEvidence.")
        profile = self.profile
        residual = jnp.maximum(evidence.divergence_norm, evidence.marker_slip_norm)
        conservation = jnp.max(
            jnp.stack(
                (
                    jnp.max(jnp.abs(evidence.force_residual)),
                    jnp.max(jnp.abs(evidence.torque_residual)),
                    jnp.abs(evidence.work_residual),
                    jnp.abs(evidence.energy_residual),
                    jnp.abs(evidence.replay_residual),
                )
            )
        )
        residual_passed = residual <= profile.residual_tolerance
        conservation_passed = conservation <= profile.conservation_tolerance
        convergence_passed = (evidence.spatial_order >= profile.minimum_order) & (
            evidence.temporal_order >= profile.minimum_order
        )
        stochastic_passed = jnp.asarray(not profile.require_stochastic) | (
            evidence.stochastic_covariance_error <= profile.stochastic_tolerance
        )
        contact_passed = jnp.asarray(not profile.require_contact) | (
            jnp.maximum(evidence.contact_residual, evidence.lubrication_error)
            <= profile.residual_tolerance
        )
        interface_passed = jnp.asarray(not profile.require_interface) | (
            evidence.interface_error <= profile.residual_tolerance
        )
        successful = (
            evidence.finite
            & residual_passed
            & conservation_passed
            & convergence_passed
            & stochastic_passed
            & contact_passed
            & interface_passed
        )
        return MarkerFlowQualificationResult(
            evidence,
            residual_passed,
            conservation_passed,
            convergence_passed,
            stochastic_passed,
            contact_passed,
            interface_passed,
            successful,
            profile.profile_id,
        )


__all__ = [
    "MarkerFlowQualificationEvidence",
    "MarkerFlowQualificationPlan",
    "MarkerFlowQualificationProfile",
    "MarkerFlowQualificationResult",
    "observed_convergence_order",
]
