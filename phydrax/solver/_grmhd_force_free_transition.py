#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations._force_free import (
    ForceFreeConstraintEvaluation,
    ForceFreeProjectionResult,
    GRForceFreeSystem,
)
from ..equations._relativistic_mhd import (
    IdealValenciaGRMHDSystem,
    ValenciaPrimitiveRecovery,
)
from ..metrix._adm_exchange import ADMGridGeometry


class GRMHDForceFreeHybridState(StrictModule):
    grmhd_conserved: Array
    force_free_state: Array
    force_free_mask: Array
    transition_energy_reservoir: Array


class GRMHDForceFreeTransitionLedger(StrictModule):
    entered_force_free: Array
    restored_grmhd: Array
    electromagnetic_energy_change: Array
    reservoir_energy_change: Array
    energy_balance_residual: Array
    accepted: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRMHDForceFreeTransitionResult(StrictModule):
    candidate: GRMHDForceFreeHybridState
    state: GRMHDForceFreeHybridState
    material_recovery: ValenciaPrimitiveRecovery
    support_recovery: ValenciaPrimitiveRecovery
    force_free_projection: ForceFreeProjectionResult
    force_free_constraints: ForceFreeConstraintEvaluation
    ledger: GRMHDForceFreeTransitionLedger
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class GRMHDForceFreeTransitionPlan(StrictModule, NonTrainableState):
    """Hysteretic, cellwise, non-blended GRMHD/force-free regime switching.

    Material variables become inactive, but remain stored, on entry. Restoration is
    permitted only from explicitly supplied conservative material support. An
    internal transition-energy reservoir makes constraint-projection changes
    explicit and exactly balanced instead of silently deleting field energy.
    """

    grmhd: IdealValenciaGRMHDSystem
    force_free: GRForceFreeSystem
    enter_magnetization: float = eqx.field(static=True)
    exit_magnetization: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grmhd: IdealValenciaGRMHDSystem,
        force_free: GRForceFreeSystem,
        /,
        *,
        enter_magnetization: float,
        exit_magnetization: float,
    ) -> None:
        if not isinstance(grmhd, IdealValenciaGRMHDSystem):
            raise TypeError("grmhd must be IdealValenciaGRMHDSystem.")
        if not isinstance(force_free, GRForceFreeSystem):
            raise TypeError("force_free must be GRForceFreeSystem.")
        if (
            grmhd.scale.scale_id != force_free.scale.scale_id
            or grmhd.convention.convention_id != force_free.convention.convention_id
        ):
            raise ValueError("GRMHD and force-free contracts differ.")
        enter = float(enter_magnetization)
        exit_ = float(exit_magnetization)
        if (
            not np.isfinite(enter)
            or not np.isfinite(exit_)
            or exit_ <= 0.0
            or enter <= exit_
        ):
            raise ValueError(
                "Force-free switching requires finite 0 < exit < enter magnetization."
            )
        self.grmhd = grmhd
        self.force_free = force_free
        self.enter_magnetization = enter
        self.exit_magnetization = exit_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "grmhd-force-free-transition",
                "grmhd": grmhd.system_id,
                "force_free": force_free.system_id,
                "enter_magnetization": enter,
                "exit_magnetization": exit_,
            }
        )

    def _ideal_force_free(
        self, recovery: ValenciaPrimitiveRecovery, geometry: ADMGridGeometry, /
    ) -> tuple[Array, ForceFreeProjectionResult]:
        primitive = recovery.primitive
        velocity = primitive[..., 1:4]
        magnetic = primitive[..., 5:8]
        orientation = (
            self.grmhd.convention.spacetime_orientation
            * self.grmhd.convention.future_time_orientation
        )
        electric_covector = (
            -orientation
            * geometry.sqrt_det_spatial_metric[..., None]
            * jnp.cross(velocity, magnetic)
            / float(self.grmhd.scale.speed_of_light)
        )
        electric = contract(
            "...ij,...j->...i",
            geometry.inverse_spatial_metric,
            electric_covector,
            backend="jax",
        )
        projection = self.force_free.project_constraints(electric, magnetic, geometry)
        state = jnp.concatenate(
            (
                projection.electric_field,
                magnetic,
                jnp.zeros(magnetic.shape[:-1] + (2,), dtype=magnetic.dtype),
            ),
            axis=-1,
        )
        return state, projection

    @staticmethod
    def _electromagnetic_energy(
        force_free_state: Array, geometry: ADMGridGeometry, /
    ) -> Array:
        electric = force_free_state[..., :3]
        magnetic = force_free_state[..., 3:6]
        return 0.5 * (
            contract(
                "...i,...ij,...j->...",
                electric,
                geometry.spatial_metric,
                electric,
                backend="jax",
            )
            + contract(
                "...i,...ij,...j->...",
                magnetic,
                geometry.spatial_metric,
                magnetic,
                backend="jax",
            )
        )

    def initialize(
        self,
        grmhd_conserved: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        force_free_mask: ArrayLike | None = None,
        composition: ArrayLike | None = None,
    ) -> GRMHDForceFreeHybridState:
        conserved = self.grmhd._state(
            grmhd_conserved, "Initial GRMHD/force-free conservative state"
        )
        if conserved.shape[:-1] != geometry.leading_shape:
            raise ValueError("Hybrid initial state must match ADM geometry.")
        mask = (
            jnp.zeros(geometry.leading_shape, dtype=bool)
            if force_free_mask is None
            else jnp.asarray(force_free_mask, dtype=bool)
        )
        if mask.shape != geometry.leading_shape:
            raise ValueError("Force-free mask must match ADM geometry.")
        recovery = self.grmhd.recover(conserved, geometry, composition)
        force_free, projection = self._ideal_force_free(recovery, geometry)
        valid = jnp.all(recovery.qualified) & jnp.all(projection.qualified | ~mask)
        conserved = eqx.error_if(
            conserved, ~valid, "Initial hybrid GRMHD/force-free state is invalid."
        )
        return GRMHDForceFreeHybridState(
            conserved,
            force_free,
            mask,
            jnp.zeros(geometry.leading_shape, dtype=conserved.dtype),
        )

    def transition(
        self,
        state: GRMHDForceFreeHybridState,
        geometry: ADMGridGeometry,
        /,
        *,
        material_support: ArrayLike | None = None,
        composition: ArrayLike | None = None,
    ) -> GRMHDForceFreeTransitionResult:
        if not isinstance(state, GRMHDForceFreeHybridState):
            raise TypeError("state must be GRMHDForceFreeHybridState.")
        material_recovery = self.grmhd.recover(
            state.grmhd_conserved, geometry, composition
        )
        if material_support is None:
            support = state.grmhd_conserved
            support_available = jnp.zeros(geometry.leading_shape, dtype=bool)
        else:
            support = self.grmhd._state(
                material_support, "Force-free restoration material support"
            )
            if support.shape[:-1] != geometry.leading_shape:
                raise ValueError("Material support must match ADM geometry.")
            support_available = jnp.ones(geometry.leading_shape, dtype=bool)
        support_recovery = self.grmhd.recover(support, geometry, composition)
        effective_magnetization = jnp.where(
            state.force_free_mask,
            support_recovery.magnetization,
            material_recovery.magnetization,
        )
        enter = ~state.force_free_mask & (
            effective_magnetization >= self.enter_magnetization
        )
        restore = (
            state.force_free_mask
            & support_available
            & (effective_magnetization <= self.exit_magnetization)
        )
        ideal_force_free, projection = self._ideal_force_free(material_recovery, geometry)
        existing_constraints = self.force_free.constraint_evaluation(
            state.force_free_state[..., :3],
            state.force_free_state[..., 3:6],
            geometry,
        )
        candidate_force_free = jnp.where(
            enter[..., None], ideal_force_free, state.force_free_state
        )
        candidate_grmhd = jnp.where(restore[..., None], support, state.grmhd_conserved)
        candidate_mask = (state.force_free_mask | enter) & ~restore
        existing_energy = self._electromagnetic_energy(state.force_free_state, geometry)
        safe_scale = jnp.maximum(
            projection.dominance_scale,
            jnp.finfo(ideal_force_free.dtype).tiny,
        )
        unprojected_electric = (
            projection.electric_field / safe_scale[..., None]
            + projection.parallel_correction
        )
        unprojected_ideal = ideal_force_free.at[..., :3].set(unprojected_electric)
        entry_energy = self._electromagnetic_energy(unprojected_ideal, geometry)
        entered_energy = self._electromagnetic_energy(ideal_force_free, geometry)
        support_force_free, _ = self._ideal_force_free(support_recovery, geometry)
        restored_energy = self._electromagnetic_energy(support_force_free, geometry)
        current_energy = jnp.where(enter, entry_energy, existing_energy)
        electromagnetic_change = jnp.where(
            enter,
            entered_energy - current_energy,
            jnp.where(restore, restored_energy - current_energy, 0.0),
        )
        reservoir_change = -electromagnetic_change
        reservoir_candidate = state.transition_energy_reservoir + reservoir_change
        energy_residual = electromagnetic_change + reservoir_change
        candidate = GRMHDForceFreeHybridState(
            candidate_grmhd,
            candidate_force_free,
            candidate_mask,
            reservoir_candidate,
        )
        finite = (
            jnp.all(material_recovery.finite)
            & jnp.all(support_recovery.finite | ~support_available)
            & jnp.all(projection.finite | ~enter)
            & jnp.all(existing_constraints.finite | ~state.force_free_mask | restore)
            & jnp.all(jnp.isfinite(reservoir_candidate))
        )
        physically_valid = (
            jnp.all(material_recovery.physically_valid | state.force_free_mask)
            & jnp.all(
                support_recovery.physically_valid
                | ~support_available
                | ~state.force_free_mask
            )
            & jnp.all(projection.physically_valid | ~enter)
            & jnp.all(
                existing_constraints.physically_valid | ~state.force_free_mask | restore
            )
        )
        qualified = (
            finite
            & physically_valid
            & jnp.all(material_recovery.qualified | state.force_free_mask)
            & jnp.all(support_recovery.qualified | ~restore)
            & jnp.all(projection.qualified | ~enter)
            & jnp.all(existing_constraints.qualified | ~candidate_mask | enter)
        )
        derivative = (
            qualified
            & ~jnp.any(enter | restore)
            & jnp.all(material_recovery.derivative_valid | state.force_free_mask)
            & jnp.all(existing_constraints.derivative_valid | ~candidate_mask)
        )
        accepted_state = GRMHDForceFreeHybridState(
            jnp.where(qualified, candidate.grmhd_conserved, state.grmhd_conserved),
            jnp.where(qualified, candidate.force_free_state, state.force_free_state),
            jnp.where(qualified, candidate.force_free_mask, state.force_free_mask),
            jnp.where(
                qualified,
                candidate.transition_energy_reservoir,
                state.transition_energy_reservoir,
            ),
        )
        zero_if_rejected = lambda value: jnp.where(
            qualified, value, jnp.zeros_like(value)
        )
        ledger = GRMHDForceFreeTransitionLedger(
            jnp.where(qualified, enter, jnp.zeros_like(enter)),
            jnp.where(qualified, restore, jnp.zeros_like(restore)),
            zero_if_rejected(electromagnetic_change),
            zero_if_rejected(reservoir_change),
            zero_if_rejected(energy_residual),
            qualified,
            finite,
            qualified,
            self.plan_id,
        )
        return GRMHDForceFreeTransitionResult(
            candidate,
            accepted_state,
            material_recovery,
            support_recovery,
            projection,
            existing_constraints,
            ledger,
            qualified,
            finite,
            physically_valid,
            qualified,
            derivative,
            self.plan_id,
        )


__all__ = [
    "GRMHDForceFreeHybridState",
    "GRMHDForceFreeTransitionLedger",
    "GRMHDForceFreeTransitionPlan",
    "GRMHDForceFreeTransitionResult",
]
