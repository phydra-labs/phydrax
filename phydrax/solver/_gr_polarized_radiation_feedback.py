#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la
from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import RelativityScaleContract
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ..metrix._spacetime_conventions import RelativityConvention


def polarized_propagation_matrix(
    absorption_stokes: ArrayLike, faraday_coefficients: ArrayLike, /
) -> Array:
    """Build the canonical (I,Q,U,V) absorption/Faraday propagation matrix."""

    absorption = jnp.asarray(absorption_stokes)
    faraday = jnp.asarray(faraday_coefficients, dtype=absorption.dtype)
    if absorption.shape[-1:] != (4,) or faraday.shape != absorption.shape[:-1] + (3,):
        raise ValueError("Polarized coefficients must end in four and three components.")
    alpha_i = absorption[..., 0]
    alpha_q, alpha_u, alpha_v = (
        absorption[..., 1],
        absorption[..., 2],
        absorption[..., 3],
    )
    rho_q, rho_u, rho_v = faraday[..., 0], faraday[..., 1], faraday[..., 2]
    matrix = jnp.zeros(absorption.shape[:-1] + (4, 4), dtype=absorption.dtype)
    matrix = matrix.at[..., 0, :].set(absorption)
    matrix = matrix.at[..., 1, 0].set(alpha_q)
    matrix = matrix.at[..., 2, 0].set(alpha_u)
    matrix = matrix.at[..., 3, 0].set(alpha_v)
    matrix = matrix.at[..., 1, 1].set(alpha_i)
    matrix = matrix.at[..., 2, 2].set(alpha_i)
    matrix = matrix.at[..., 3, 3].set(alpha_i)
    matrix = matrix.at[..., 1, 2].set(rho_v)
    matrix = matrix.at[..., 2, 1].set(-rho_v)
    matrix = matrix.at[..., 1, 3].set(-rho_u)
    matrix = matrix.at[..., 3, 1].set(rho_u)
    matrix = matrix.at[..., 2, 3].set(rho_q)
    return matrix.at[..., 3, 2].set(-rho_q)


class GRPolarizedRadiationFeedbackState(StrictModule):
    stokes: Array
    matter_energy_density: Array
    matter_momentum_covector: Array
    time: Array
    accepted_steps: Array


class GRPolarizedRadiationFeedbackLedger(StrictModule):
    radiation_energy_change: Array
    matter_energy_change: Array
    radiation_momentum_change: Array
    matter_momentum_change: Array
    energy_balance_residual: Array
    momentum_balance_residual: Array
    accepted: Array
    finite: Array
    qualified: Array
    plan_id: str = eqx.field(static=True)


class GRPolarizedRadiationFeedbackResult(StrictModule):
    candidate: GRPolarizedRadiationFeedbackState
    state: GRPolarizedRadiationFeedbackState
    stress_energy: StressEnergyProjection
    ledger: GRPolarizedRadiationFeedbackLedger
    action_converged: Array
    cone_residual: Array
    propagation_structure_residual: Array
    accepted: Array
    finite: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


class GRPolarizedRadiationFeedbackPlan(StrictModule, NonTrainableState):
    """Local polarized transfer with exact opposite material four-force feedback."""

    scale: RelativityScaleContract
    convention: RelativityConvention
    beam_weights: Array
    beam_count: int = eqx.field(static=True)
    cone_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale: RelativityScaleContract,
        convention: RelativityConvention,
        beam_weights: ArrayLike = (1.0,),
        /,
        *,
        cone_tolerance: float = 1.0e-9,
    ) -> None:
        if not isinstance(scale, RelativityScaleContract) or not isinstance(
            convention, RelativityConvention
        ):
            raise TypeError("Polarized feedback requires relativity contracts.")
        if convention.metric_signature != "mostly_plus":
            raise ValueError("Polarized GR feedback requires mostly-plus signature.")
        weights = np.asarray(beam_weights, dtype=np.float64)
        tolerance = float(cone_tolerance)
        if (
            weights.ndim != 1
            or weights.size == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Polarized feedback beam weights or tolerance are invalid.")
        normalized = weights / np.sum(weights)
        self.scale = scale
        self.convention = convention
        self.beam_weights = jnp.asarray(normalized)
        self.beam_count = weights.size
        self.cone_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gr-polarized-radiation-feedback",
                "scale": scale.scale_id,
                "convention": convention.convention_id,
                "beam_weights": array_tree_fingerprint(normalized),
                "cone_tolerance": tolerance,
            }
        )

    def _geometry(self, geometry: ADMGridGeometry, /) -> ADMGridGeometry:
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if (
            geometry.scale_id != self.scale.scale_id
            or geometry.convention_id != self.convention.convention_id
        ):
            raise ValueError("Polarized feedback and geometry contracts differ.")
        return geometry

    def _stokes_cone(self, stokes: Array, /) -> tuple[Array, Array]:
        polarized_norm = jnp.sqrt(jnp.sum(stokes[..., 1:] ** 2, axis=-1))
        residual = polarized_norm - stokes[..., 0]
        valid = (
            jnp.all(jnp.isfinite(stokes), axis=-1)
            & (stokes[..., 0] >= 0.0)
            & (residual <= self.cone_tolerance)
        )
        return residual, valid

    def initialize(
        self,
        stokes: ArrayLike,
        matter_energy_density: ArrayLike,
        matter_momentum_covector: ArrayLike,
        geometry: ADMGridGeometry,
        /,
        *,
        time: ArrayLike = 0.0,
    ) -> GRPolarizedRadiationFeedbackState:
        geometry = self._geometry(geometry)
        value = jnp.asarray(stokes)
        energy = jnp.asarray(matter_energy_density, dtype=value.dtype)
        momentum = jnp.asarray(matter_momentum_covector, dtype=value.dtype)
        expected = geometry.leading_shape + (self.beam_count, 4)
        if (
            value.shape != expected
            or energy.shape != geometry.leading_shape
            or momentum.shape != geometry.leading_shape + (3,)
        ):
            raise ValueError("Polarized feedback initial fields must match ADM geometry.")
        _, cone = self._stokes_cone(value)
        valid = (
            jnp.all(cone)
            & jnp.all(jnp.isfinite(energy))
            & jnp.all(energy >= 0.0)
            & jnp.all(jnp.isfinite(momentum))
        )
        value = eqx.error_if(
            value, ~valid, "Initial polarized radiation feedback state is invalid."
        )
        return GRPolarizedRadiationFeedbackState(
            value,
            energy,
            momentum,
            jnp.asarray(time, dtype=value.dtype).reshape(()),
            jnp.zeros((), dtype=jnp.int32),
        )

    def _directions(
        self, direction_vectors: ArrayLike, geometry: ADMGridGeometry, dtype, /
    ) -> tuple[Array, Array]:
        directions = jnp.asarray(direction_vectors, dtype=dtype)
        expected = geometry.leading_shape + (self.beam_count, 3)
        if directions.shape != expected:
            raise ValueError("Polarized beam directions must match ADM geometry.")
        norm_squared = contract(
            "...ni,...ij,...nj->...n",
            directions,
            geometry.spatial_metric,
            directions,
            backend="jax",
        )
        normalized = (
            directions
            / jnp.sqrt(jnp.maximum(norm_squared, jnp.finfo(dtype).tiny))[..., None]
        )
        covectors = contract(
            "...ij,...nj->...ni",
            geometry.spatial_metric,
            normalized,
            backend="jax",
        )
        return normalized, covectors

    def _projection(
        self,
        stokes: Array,
        direction_vectors: Array,
        direction_covectors: Array,
        geometry: ADMGridGeometry,
        qualified: Array,
        /,
    ) -> StressEnergyProjection:
        weights = self.beam_weights.astype(stokes.dtype)
        intensity = stokes[..., 0]
        energy = jnp.sum(weights * intensity, axis=-1)
        momentum = contract(
            "n,...n,...ni->...i",
            weights,
            intensity,
            direction_covectors,
            backend="jax",
        )
        stress = contract(
            "n,...n,...ni,...nj->...ij",
            weights,
            intensity,
            direction_covectors,
            direction_covectors,
            backend="jax",
        )
        projection_id = canonical_fingerprint(
            {
                "kind": "gr-polarized-feedback-stress-energy",
                "plan": self.plan_id,
                "geometry_lineage": geometry.geometry_lineage_id,
            }
        )
        del direction_vectors
        return StressEnergyProjection(
            energy,
            momentum,
            stress,
            geometry.active,
            qualified,
            jnp.max(jnp.abs(stress - jnp.swapaxes(stress, -1, -2)), axis=(-2, -1)),
            jnp.zeros_like(energy),
            snapshot_token=geometry.snapshot_token,
            geometry_lineage_id=geometry.geometry_lineage_id,
            convention_id=geometry.convention_id,
            scale_id=geometry.scale_id,
            topology_id=geometry.topology_id,
            projection_id=projection_id,
        )

    def advance(
        self,
        state: GRPolarizedRadiationFeedbackState,
        end_time: ArrayLike,
        emission_stokes: ArrayLike,
        propagation_matrix: ArrayLike,
        direction_vectors: ArrayLike,
        geometry: ADMGridGeometry,
        /,
    ) -> GRPolarizedRadiationFeedbackResult:
        if not isinstance(state, GRPolarizedRadiationFeedbackState):
            raise TypeError("state must be GRPolarizedRadiationFeedbackState.")
        geometry = self._geometry(geometry)
        end = jnp.asarray(end_time, dtype=state.stokes.dtype).reshape(())
        step = end - state.time
        end = eqx.error_if(
            end,
            ~jnp.isfinite(end) | (step <= 0.0),
            "Polarized feedback interval must be finite and increasing.",
        )
        emission = jnp.asarray(emission_stokes, dtype=state.stokes.dtype)
        propagation = jnp.asarray(propagation_matrix, dtype=state.stokes.dtype)
        if emission.shape != state.stokes.shape or propagation.shape != (
            state.stokes.shape[:-1] + (4, 4)
        ):
            raise ValueError("Polarized emission or propagation shape is invalid.")
        directions, direction_covectors = self._directions(
            direction_vectors, geometry, state.stokes.dtype
        )
        augmented = jnp.zeros(propagation.shape[:-2] + (5, 5), dtype=state.stokes.dtype)
        augmented = augmented.at[..., :4, :4].set(-propagation)
        augmented = augmented.at[..., :4, 4].set(emission)
        initial = jnp.concatenate(
            (
                state.stokes,
                jnp.ones(state.stokes.shape[:-1] + (1,), dtype=state.stokes.dtype),
            ),
            axis=-1,
        )
        flat_augmented = augmented.reshape((-1, 5, 5))
        flat_initial = initial.reshape((-1, 5))

        operator = la.DenseLinearOperator(
            flat_augmented, operator_id=f"{self.plan_id}:augmented-transfer"
        )
        actions = la.matrix_exponential_action(operator, flat_initial, step)
        stokes_candidate = actions.value[..., :4].reshape(state.stokes.shape)
        action_converged = actions.successful.reshape(state.stokes.shape[:-1])
        weights = self.beam_weights.astype(state.stokes.dtype)
        intensity_change = stokes_candidate[..., 0] - state.stokes[..., 0]
        radiation_energy_change = jnp.sum(weights * intensity_change, axis=-1)
        radiation_momentum_change = contract(
            "n,...n,...ni->...i",
            weights,
            intensity_change,
            direction_covectors,
            backend="jax",
        )
        matter_energy_candidate = state.matter_energy_density - radiation_energy_change
        matter_momentum_candidate = (
            state.matter_momentum_covector - radiation_momentum_change
        )
        candidate = GRPolarizedRadiationFeedbackState(
            stokes_candidate,
            matter_energy_candidate,
            matter_momentum_candidate,
            end,
            state.accepted_steps + jnp.asarray(1, dtype=jnp.int32),
        )
        cone_residual, cone_valid = self._stokes_cone(stokes_candidate)
        emission_residual, emission_cone = self._stokes_cone(emission)
        del emission_residual
        absorption = propagation[..., 0, :]
        faraday = jnp.stack(
            (
                propagation[..., 2, 3],
                -propagation[..., 1, 3],
                propagation[..., 1, 2],
            ),
            axis=-1,
        )
        reconstructed = polarized_propagation_matrix(absorption, faraday)
        structure_residual = jnp.max(jnp.abs(propagation - reconstructed), axis=(-2, -1))
        dichroism_norm = jnp.sqrt(jnp.sum(absorption[..., 1:] ** 2, axis=-1))
        finite = (
            geometry.all_active_valid
            & jnp.all(jnp.isfinite(emission))
            & jnp.all(jnp.isfinite(propagation))
            & jnp.all(jnp.isfinite(stokes_candidate))
            & jnp.all(jnp.isfinite(matter_energy_candidate))
            & jnp.all(jnp.isfinite(matter_momentum_candidate))
            & jnp.all(action_converged)
        )
        physically_valid = (
            finite
            & jnp.all(cone_valid)
            & jnp.all(emission_cone)
            & jnp.all(absorption[..., 0] >= dichroism_norm)
            & jnp.all(matter_energy_candidate >= 0.0)
        )
        qualified = physically_valid & jnp.all(structure_residual <= self.cone_tolerance)
        derivative = (
            qualified
            & jnp.all(cone_residual < -self.cone_tolerance)
            & jnp.all(absorption[..., 0] > dichroism_norm)
        )
        accepted_state = GRPolarizedRadiationFeedbackState(
            jnp.where(qualified, candidate.stokes, state.stokes),
            jnp.where(
                qualified,
                candidate.matter_energy_density,
                state.matter_energy_density,
            ),
            jnp.where(
                qualified,
                candidate.matter_momentum_covector,
                state.matter_momentum_covector,
            ),
            jnp.where(qualified, candidate.time, state.time),
            jnp.where(qualified, candidate.accepted_steps, state.accepted_steps),
        )
        matter_energy_change = matter_energy_candidate - state.matter_energy_density
        matter_momentum_change = (
            matter_momentum_candidate - state.matter_momentum_covector
        )
        energy_residual = radiation_energy_change + matter_energy_change
        momentum_residual = radiation_momentum_change + matter_momentum_change
        zero_if_rejected = lambda value: jnp.where(
            qualified, value, jnp.zeros_like(value)
        )
        ledger = GRPolarizedRadiationFeedbackLedger(
            zero_if_rejected(radiation_energy_change),
            zero_if_rejected(matter_energy_change),
            zero_if_rejected(radiation_momentum_change),
            zero_if_rejected(matter_momentum_change),
            zero_if_rejected(energy_residual),
            zero_if_rejected(momentum_residual),
            qualified,
            finite,
            qualified,
            self.plan_id,
        )
        accepted_cone = jnp.where(
            qualified[..., None] if qualified.ndim else qualified,
            cone_valid,
            self._stokes_cone(state.stokes)[1],
        )
        projection = self._projection(
            accepted_state.stokes,
            directions,
            direction_covectors,
            geometry,
            jnp.all(accepted_cone, axis=-1),
        )
        return GRPolarizedRadiationFeedbackResult(
            candidate,
            accepted_state,
            projection,
            ledger,
            action_converged,
            cone_residual,
            structure_residual,
            qualified,
            finite,
            physically_valid,
            qualified,
            derivative,
            self.plan_id,
        )


__all__ = [
    "GRPolarizedRadiationFeedbackLedger",
    "GRPolarizedRadiationFeedbackPlan",
    "GRPolarizedRadiationFeedbackResult",
    "GRPolarizedRadiationFeedbackState",
    "polarized_propagation_matrix",
]
