#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._dem import DEMDiagnostics, DEMRejectionReason
from ._qualification import AbstractParticleQualificationProfile


class DEMConstraintResiduals(StrictModule):
    net_internal_force: Array
    net_internal_torque: Array
    relative_energy_balance: Array
    negative_dissipation: Array
    friction_cone: Array
    maximum_overlap_fraction: Array
    wall_action_reaction: Array
    contact_history_continuity: Array


class DEMDifferentiabilityMargins(StrictModule):
    contact_activation: Array
    friction_switch: Array
    route_capacity_successful: Array


class DEMQualificationProfile(AbstractParticleQualificationProfile):
    internal_force_tolerance: float = eqx.field(static=True)
    internal_torque_tolerance: float = eqx.field(static=True)
    energy_balance_tolerance: float = eqx.field(static=True)
    dissipation_tolerance: float = eqx.field(static=True)
    friction_tolerance: float = eqx.field(static=True)
    maximum_overlap_fraction: float = eqx.field(static=True)
    boundary_tolerance: float = eqx.field(static=True)
    history_tolerance: float = eqx.field(static=True)
    profile_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        internal_force_tolerance: float = 1.0e-10,
        internal_torque_tolerance: float = 1.0e-10,
        energy_balance_tolerance: float = 1.0e-4,
        dissipation_tolerance: float = 1.0e-12,
        friction_tolerance: float = 1.0e-10,
        maximum_overlap_fraction: float = 0.1,
        boundary_tolerance: float = 1.0e-10,
        history_tolerance: float = 1.0e-12,
    ):
        values = tuple(
            float(value)
            for value in (
                internal_force_tolerance,
                internal_torque_tolerance,
                energy_balance_tolerance,
                dissipation_tolerance,
                friction_tolerance,
                maximum_overlap_fraction,
                boundary_tolerance,
                history_tolerance,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("DEM qualification tolerances must be finite and positive.")
        (
            self.internal_force_tolerance,
            self.internal_torque_tolerance,
            self.energy_balance_tolerance,
            self.dissipation_tolerance,
            self.friction_tolerance,
            self.maximum_overlap_fraction,
            self.boundary_tolerance,
            self.history_tolerance,
        ) = values
        self.profile_id = canonical_fingerprint(
            {"kind": "dem-qualification-profile", "tolerances": list(values)}
        )

    def constraints_satisfied(self, residuals: DEMConstraintResiduals, /) -> Array:
        if not isinstance(residuals, DEMConstraintResiduals):
            raise TypeError("residuals must be DEMConstraintResiduals.")
        return (
            (residuals.net_internal_force <= self.internal_force_tolerance)
            & (residuals.net_internal_torque <= self.internal_torque_tolerance)
            & (residuals.relative_energy_balance <= self.energy_balance_tolerance)
            & (residuals.negative_dissipation <= self.dissipation_tolerance)
            & (residuals.friction_cone <= self.friction_tolerance)
            & (residuals.maximum_overlap_fraction <= self.maximum_overlap_fraction)
            & (residuals.wall_action_reaction <= self.boundary_tolerance)
            & (residuals.contact_history_continuity <= self.history_tolerance)
        )


class DEMQualificationArtifact(StrictModule):
    profile_id: str = eqx.field(static=True)
    residuals: DEMConstraintResiduals
    execution_successful: Array
    constraints_satisfied: Array
    qualified: Array
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: DEMQualificationProfile,
        residuals: DEMConstraintResiduals,
        execution_successful: Array,
        /,
    ):
        if not isinstance(profile, DEMQualificationProfile):
            raise TypeError("profile must be a DEMQualificationProfile.")
        if not isinstance(residuals, DEMConstraintResiduals):
            raise TypeError("residuals must be DEMConstraintResiduals.")
        execution = jnp.asarray(execution_successful, dtype=bool)
        constraints = profile.constraints_satisfied(residuals)
        self.profile_id = profile.profile_id
        self.residuals = residuals
        self.execution_successful = execution
        self.constraints_satisfied = constraints
        self.qualified = execution & constraints
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "dem-qualification-artifact",
                "profile": profile.profile_id,
                "residual_schema": "dem-constraint-residuals:v1",
            }
        )


def dem_constraint_residuals(
    diagnostics: DEMDiagnostics,
    /,
) -> DEMConstraintResiduals:
    if not isinstance(diagnostics, DEMDiagnostics):
        raise TypeError("diagnostics must be DEMDiagnostics.")
    energy = diagnostics.energy
    return DEMConstraintResiduals(
        jnp.linalg.norm(diagnostics.net_internal_force),
        jnp.linalg.norm(diagnostics.net_internal_torque),
        jnp.abs(energy.last_relative_energy_residual),
        jnp.maximum(-energy.cumulative_contact_balance_loss, 0.0),
        jnp.maximum(diagnostics.maximum_friction_cone_defect, 0.0),
        diagnostics.maximum_overlap_fraction,
        jnp.abs(diagnostics.wall_action_reaction_defect),
        jnp.abs(diagnostics.contact_history_continuity_defect),
    )


def dem_differentiability_margins(
    diagnostics: DEMDiagnostics,
    /,
) -> DEMDifferentiabilityMargins:
    if not isinstance(diagnostics, DEMDiagnostics):
        raise TypeError("diagnostics must be DEMDiagnostics.")
    capacity_mask = int(
        DEMRejectionReason.CELL_CAPACITY
        | DEMRejectionReason.PAIR_CAPACITY
        | DEMRejectionReason.DOMAIN
        | DEMRejectionReason.PAIR_KEY
    )
    return DEMDifferentiabilityMargins(
        diagnostics.minimum_gap_margin,
        diagnostics.minimum_friction_switch_margin,
        (diagnostics.rejection_reasons & capacity_mask) == 0,
    )


def qualify_dem(
    diagnostics: DEMDiagnostics,
    profile: DEMQualificationProfile,
    /,
) -> DEMQualificationArtifact:
    return DEMQualificationArtifact(
        profile,
        dem_constraint_residuals(diagnostics),
        diagnostics.successful,
    )


__all__ = [
    "DEMConstraintResiduals",
    "DEMDifferentiabilityMargins",
    "DEMQualificationArtifact",
    "DEMQualificationProfile",
    "dem_constraint_residuals",
    "dem_differentiability_margins",
    "qualify_dem",
]
