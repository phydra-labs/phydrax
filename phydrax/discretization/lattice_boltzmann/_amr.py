#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._collision import macroscopic_raw_moments, quadratic_equilibrium
from ._lattice import LatticeBoltzmannVelocitySet
from ._precision import LatticeBoltzmannPrecisionPolicy
from ._scaling import LatticeBoltzmannScaling


class LatticeBoltzmannAMRTransferEvidence(StrictModule):
    mass_defect: Array
    momentum_defect: Array
    minimum_population: Array
    successful: Array


class LatticeBoltzmannAMRTransferPlan(StrictModule, NonTrainableState):
    """Ratio-two moment-conservative transfer for blockwise on-lattice populations."""

    velocity_set: LatticeBoltzmannVelocitySet
    refinement_ratio: int = eqx.field(static=True)
    nonequilibrium_scale: float = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocity_set: LatticeBoltzmannVelocitySet,
        /,
        *,
        refinement_ratio: int = 2,
        nonequilibrium_scale: float = 1.0,
    ):
        if not isinstance(velocity_set, LatticeBoltzmannVelocitySet):
            raise TypeError("velocity_set must be LatticeBoltzmannVelocitySet.")
        ratio = int(refinement_ratio)
        scale = float(nonequilibrium_scale)
        if ratio != 2:
            raise ValueError("Initial LBM AMR supports refinement ratio two only.")
        if not np.isfinite(scale) or scale < 0.0:
            raise ValueError("nonequilibrium_scale must be finite and nonnegative.")
        self.velocity_set = velocity_set
        self.refinement_ratio = ratio
        self.nonequilibrium_scale = scale
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-amr-transfer",
                "lattice": velocity_set.lattice_id,
                "ratio": ratio,
                "nonequilibrium_scale": scale,
            }
        )

    def restrict(
        self, fine_populations: Array, /
    ) -> tuple[Array, LatticeBoltzmannAMRTransferEvidence]:
        fine = jnp.asarray(fine_populations)
        dimension = self.velocity_set.dimension
        if (
            fine.ndim != dimension + 1
            or fine.shape[-1] != self.velocity_set.population_count
        ):
            raise ValueError("Fine populations have incompatible LBM shape.")
        if any(size % self.refinement_ratio for size in fine.shape[:-1]):
            raise ValueError("Every fine spatial extent must be divisible by two.")
        coarse_shape = tuple(size // self.refinement_ratio for size in fine.shape[:-1])
        reshape = []
        for size in coarse_shape:
            reshape.extend((size, self.refinement_ratio))
        reshape.append(fine.shape[-1])
        blocked = fine.reshape(tuple(reshape))
        reduction_axes = tuple(range(1, 2 * dimension, 2))
        coarse = jnp.mean(blocked, axis=reduction_axes)
        evidence = self._evidence(fine, coarse, fine_to_coarse=True)
        return coarse, evidence

    def prolong(
        self, coarse_populations: Array, /
    ) -> tuple[Array, LatticeBoltzmannAMRTransferEvidence]:
        coarse = jnp.asarray(coarse_populations)
        dimension = self.velocity_set.dimension
        if (
            coarse.ndim != dimension + 1
            or coarse.shape[-1] != self.velocity_set.population_count
        ):
            raise ValueError("Coarse populations have incompatible LBM shape.")
        fine = coarse
        for axis in range(dimension):
            fine = jnp.repeat(fine, self.refinement_ratio, axis=axis)
        evidence = self._evidence(fine, coarse, fine_to_coarse=True)
        return fine, evidence

    def _evidence(
        self, fine: Array, coarse: Array, /, *, fine_to_coarse: bool
    ) -> LatticeBoltzmannAMRTransferEvidence:
        ratio_volume = float(self.refinement_ratio**self.velocity_set.dimension)
        fine_mass = jnp.sum(fine) / ratio_volume
        coarse_mass = jnp.sum(coarse)
        velocities = jnp.asarray(self.velocity_set.velocities, dtype=fine.dtype)
        fine_momentum = oe.contract("...q,qd->d", fine, velocities) / ratio_volume
        coarse_momentum = oe.contract("...q,qd->d", coarse, velocities)
        mass_defect = jnp.abs(fine_mass - coarse_mass)
        momentum_defect = jnp.sqrt(jnp.sum((fine_momentum - coarse_momentum) ** 2))
        minimum = jnp.minimum(jnp.min(fine), jnp.min(coarse))
        successful = (
            jnp.all(jnp.isfinite(fine))
            & jnp.all(jnp.isfinite(coarse))
            & (minimum >= 0.0)
            & (
                mass_defect
                <= 128.0
                * jnp.finfo(fine.dtype).eps
                * jnp.maximum(jnp.abs(coarse_mass), 1.0)
            )
            & (
                momentum_defect
                <= 128.0
                * jnp.finfo(fine.dtype).eps
                * jnp.maximum(jnp.sqrt(jnp.sum(coarse_momentum**2)), 1.0)
            )
        )
        return LatticeBoltzmannAMRTransferEvidence(
            mass_defect, momentum_defect, minimum, successful
        )


class LatticeBoltzmannAMRInterfaceEvidence(StrictModule):
    mass_defect: Array
    momentum_defect: Array
    nonequilibrium_mass_defect: Array
    nonequilibrium_momentum_defect: Array
    minimum_population: Array
    nonequilibrium_scale: Array
    temporal_interpolation_fraction: Array
    successful: Array


class PreparedLatticeBoltzmannAMRTransfer(StrictModule, NonTrainableState):
    """Collision-aware equilibrium/nonequilibrium ratio-two transfer."""

    transfer: LatticeBoltzmannAMRTransferPlan
    precision: LatticeBoltzmannPrecisionPolicy
    coarse_scaling: LatticeBoltzmannScaling
    fine_scaling: LatticeBoltzmannScaling
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: LatticeBoltzmannAMRTransferPlan,
        precision: LatticeBoltzmannPrecisionPolicy,
        coarse_scaling: LatticeBoltzmannScaling,
        fine_scaling: LatticeBoltzmannScaling,
        /,
    ):
        if not isinstance(transfer, LatticeBoltzmannAMRTransferPlan):
            raise TypeError("transfer must be LatticeBoltzmannAMRTransferPlan.")
        if transfer.nonequilibrium_scale <= 0.0:
            raise ValueError(
                "Collision-aware AMR requires positive nonequilibrium scaling."
            )
        if not isinstance(precision, LatticeBoltzmannPrecisionPolicy):
            raise TypeError("precision must be LatticeBoltzmannPrecisionPolicy.")
        if not isinstance(coarse_scaling, LatticeBoltzmannScaling) or not isinstance(
            fine_scaling, LatticeBoltzmannScaling
        ):
            raise TypeError(
                "coarse_scaling and fine_scaling must be LatticeBoltzmannScaling."
            )
        if not np.isclose(
            float(fine_scaling.cell_size) * transfer.refinement_ratio,
            float(coarse_scaling.cell_size),
        ) or not np.isclose(
            float(fine_scaling.time_step) * transfer.refinement_ratio,
            float(coarse_scaling.time_step),
        ):
            raise ValueError("Collision-aware AMR requires ratio-two acoustic scaling.")
        self.transfer = transfer
        self.precision = precision
        self.coarse_scaling = coarse_scaling
        self.fine_scaling = fine_scaling
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-collision-aware-lattice-boltzmann-amr-transfer",
                "transfer": transfer.transfer_id,
                "precision": precision.policy_id,
                "coarse_scaling": coarse_scaling.scaling_id,
                "fine_scaling": fine_scaling.scaling_id,
            }
        )

    def _scale(
        self,
        coarse_relaxation_rate: Array,
        fine_relaxation_rate: Array,
        /,
    ) -> Array:
        coarse = jnp.asarray(coarse_relaxation_rate)
        fine = jnp.asarray(fine_relaxation_rate, dtype=coarse.dtype)
        if coarse.shape != () or fine.shape != ():
            raise ValueError("AMR relaxation rates must be scalar.")
        valid = (
            jnp.isfinite(coarse)
            & (coarse > 0.0)
            & (coarse < 2.0)
            & jnp.isfinite(fine)
            & (fine > 0.0)
            & (fine < 2.0)
        )
        coarse_tau_offset = 1.0 / coarse - 0.5
        fine_tau_offset = 1.0 / fine - 0.5
        scale = self.transfer.nonequilibrium_scale * fine_tau_offset / coarse_tau_offset
        return eqx.error_if(
            scale,
            ~valid | ~jnp.isfinite(scale) | (scale <= 0.0),
            "AMR nonequilibrium scaling is invalid.",
        )

    def _repeat(self, value: Array, /) -> Array:
        result = value
        for axis in range(self.transfer.velocity_set.dimension):
            result = jnp.repeat(result, self.transfer.refinement_ratio, axis=axis)
        return result

    def _block_average(self, value: Array, /) -> Array:
        dimension = self.transfer.velocity_set.dimension
        coarse_shape = tuple(
            size // self.transfer.refinement_ratio for size in value.shape[:-1]
        )
        reshape = []
        for size in coarse_shape:
            reshape.extend((size, self.transfer.refinement_ratio))
        reshape.append(value.shape[-1])
        reduction_axes = tuple(range(1, 2 * dimension, 2))
        return jnp.mean(value.reshape(tuple(reshape)), axis=reduction_axes)

    def _nonequilibrium_defects(self, nonequilibrium: Array, /) -> tuple[Array, Array]:
        velocities = jnp.asarray(
            self.transfer.velocity_set.velocities,
            dtype=nonequilibrium.dtype,
        )
        return (
            jnp.max(jnp.abs(jnp.sum(nonequilibrium, axis=-1))),
            jnp.max(jnp.abs(oe.contract("...q,qd->...d", nonequilibrium, velocities))),
        )

    def prolong(
        self,
        coarse_populations: Array,
        coarse_relaxation_rate: Array,
        fine_relaxation_rate: Array,
        /,
    ) -> tuple[Array, LatticeBoltzmannAMRInterfaceEvidence]:
        coarse = jnp.asarray(coarse_populations)
        density, momentum = macroscopic_raw_moments(
            coarse,
            self.transfer.velocity_set,
            self.precision,
        )
        velocity = (
            momentum / jnp.maximum(density, jnp.finfo(coarse.dtype).tiny)[..., None]
        )
        equilibrium = quadratic_equilibrium(
            density,
            velocity,
            self.transfer.velocity_set,
            self.precision,
        )
        nonequilibrium = coarse - equilibrium
        scale = self._scale(coarse_relaxation_rate, fine_relaxation_rate)
        fine_equilibrium = self._repeat(equilibrium)
        fine_nonequilibrium = scale * self._repeat(nonequilibrium)
        fine = self.precision.population(fine_equilibrium + fine_nonequilibrium)
        restricted, transfer_evidence = self.transfer.restrict(fine)
        restricted_density, restricted_momentum = macroscopic_raw_moments(
            restricted,
            self.transfer.velocity_set,
            self.precision,
        )
        mass_defect = jnp.max(jnp.abs(restricted_density - density))
        momentum_defect = jnp.max(jnp.abs(restricted_momentum - momentum))
        neq_mass, neq_momentum = self._nonequilibrium_defects(nonequilibrium)
        tolerance = (
            256.0
            * jnp.finfo(fine.dtype).eps
            * jnp.maximum(
                jnp.maximum(jnp.max(jnp.abs(density)), jnp.max(jnp.abs(momentum))),
                1.0,
            )
        )
        successful = (
            transfer_evidence.successful
            & jnp.all(jnp.isfinite(fine))
            & (jnp.min(fine) >= 0.0)
            & (mass_defect <= tolerance)
            & (momentum_defect <= tolerance)
            & (neq_mass <= tolerance)
            & (neq_momentum <= tolerance)
        )
        evidence = LatticeBoltzmannAMRInterfaceEvidence(
            mass_defect,
            momentum_defect,
            neq_mass,
            neq_momentum,
            jnp.min(fine),
            scale,
            jnp.asarray(0.0, dtype=fine.dtype),
            successful,
        )
        return fine, evidence

    def restrict(
        self,
        fine_populations: Array,
        coarse_relaxation_rate: Array,
        fine_relaxation_rate: Array,
        /,
    ) -> tuple[Array, LatticeBoltzmannAMRInterfaceEvidence]:
        fine = jnp.asarray(fine_populations)
        density, momentum = macroscopic_raw_moments(
            fine,
            self.transfer.velocity_set,
            self.precision,
        )
        velocity = momentum / jnp.maximum(density, jnp.finfo(fine.dtype).tiny)[..., None]
        equilibrium = quadratic_equilibrium(
            density,
            velocity,
            self.transfer.velocity_set,
            self.precision,
        )
        nonequilibrium = fine - equilibrium
        scale = self._scale(coarse_relaxation_rate, fine_relaxation_rate)
        coarse_density = self._block_average(density[..., None])[..., 0]
        coarse_momentum = self._block_average(momentum)
        coarse_velocity = (
            coarse_momentum
            / jnp.maximum(coarse_density, jnp.finfo(fine.dtype).tiny)[..., None]
        )
        coarse_equilibrium = quadratic_equilibrium(
            coarse_density,
            coarse_velocity,
            self.transfer.velocity_set,
            self.precision,
        )
        coarse_nonequilibrium = self._block_average(nonequilibrium) / scale
        coarse = self.precision.population(coarse_equilibrium + coarse_nonequilibrium)
        recovered_density, recovered_momentum = macroscopic_raw_moments(
            coarse,
            self.transfer.velocity_set,
            self.precision,
        )
        mass_defect = jnp.max(jnp.abs(recovered_density - coarse_density))
        momentum_defect = jnp.max(jnp.abs(recovered_momentum - coarse_momentum))
        neq_mass, neq_momentum = self._nonequilibrium_defects(coarse_nonequilibrium)
        tolerance = (
            256.0
            * jnp.finfo(coarse.dtype).eps
            * jnp.maximum(
                jnp.maximum(
                    jnp.max(jnp.abs(coarse_density)),
                    jnp.max(jnp.abs(coarse_momentum)),
                ),
                1.0,
            )
        )
        successful = (
            jnp.all(jnp.isfinite(coarse))
            & (jnp.min(coarse) >= 0.0)
            & (mass_defect <= tolerance)
            & (momentum_defect <= tolerance)
            & (neq_mass <= tolerance)
            & (neq_momentum <= tolerance)
        )
        evidence = LatticeBoltzmannAMRInterfaceEvidence(
            mass_defect,
            momentum_defect,
            neq_mass,
            neq_momentum,
            jnp.min(coarse),
            scale,
            jnp.asarray(1.0, dtype=coarse.dtype),
            successful,
        )
        return coarse, evidence


class LatticeBoltzmannAMRTemporalInterfacePlan(StrictModule, NonTrainableState):
    """Fixed half-time interpolation for exactly two fine substeps."""

    interpolation_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self):
        self.interpolation_fraction = 0.5
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-amr-temporal-interface",
                "interpolation_fraction": 0.5,
            }
        )

    def interpolate(self, start: Array, end: Array, /) -> Array:
        start_ = jnp.asarray(start)
        end_ = jnp.asarray(end, dtype=start_.dtype)
        if start_.shape != end_.shape:
            raise ValueError("AMR temporal endpoints must have matching shapes.")
        fraction = jnp.asarray(self.interpolation_fraction, dtype=start_.dtype)
        return (1.0 - fraction) * start_ + fraction * end_


class LatticeBoltzmannAMRState(StrictModule):
    level_populations: tuple[Array, ...]
    active_masks: tuple[Array, ...]
    fine_subcycle_phase: Array

    def __init__(
        self,
        level_populations: Sequence[Array],
        active_masks: Sequence[Array],
        fine_subcycle_phase: Array | None = None,
    ):
        populations = tuple(jnp.asarray(value) for value in level_populations)
        masks = tuple(jnp.asarray(value, dtype=bool) for value in active_masks)
        if not populations or len(populations) != len(masks):
            raise ValueError(
                "AMR state requires matching nonempty level populations and masks."
            )
        for values, mask in zip(populations, masks, strict=True):
            if mask.shape != values.shape[:-1]:
                raise ValueError("AMR active masks must match level spatial shapes.")
        phase = jnp.asarray(
            0 if fine_subcycle_phase is None else fine_subcycle_phase,
            dtype=jnp.int32,
        )
        if phase.shape != ():
            raise ValueError("fine_subcycle_phase must be scalar.")
        self.level_populations = populations
        self.active_masks = masks
        self.fine_subcycle_phase = phase


class LatticeBoltzmannAMRAdvanceResult(StrictModule):
    state: LatticeBoltzmannAMRState
    transfer_evidence: LatticeBoltzmannAMRTransferEvidence
    successful: Array


class LatticeBoltzmannCollisionAwareAMRAdvanceResult(StrictModule):
    state: LatticeBoltzmannAMRState
    interface_evidence: LatticeBoltzmannAMRInterfaceEvidence
    successful: Array


class LatticeBoltzmannAMRPlan(StrictModule, NonTrainableState):
    """Fixed two-level acoustic subcycling with explicit transfer evidence."""

    transfer: LatticeBoltzmannAMRTransferPlan
    fine_substeps: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, transfer: LatticeBoltzmannAMRTransferPlan, /, *, fine_substeps: int = 2
    ):
        if not isinstance(transfer, LatticeBoltzmannAMRTransferPlan):
            raise TypeError("transfer must be LatticeBoltzmannAMRTransferPlan.")
        if int(fine_substeps) != 2:
            raise ValueError(
                "Initial ratio-two LBM AMR requires exactly two fine substeps."
            )
        self.transfer = transfer
        self.fine_substeps = 2
        self.plan_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-amr-plan",
                "transfer": transfer.transfer_id,
                "fine_substeps": 2,
            }
        )

    def advance_two_level(
        self,
        state: LatticeBoltzmannAMRState,
        coarse_step: Callable[[Array, Any], Array],
        fine_step: Callable[[Array, Any], Array],
        /,
        *,
        args: Any = None,
    ) -> LatticeBoltzmannAMRAdvanceResult:
        if (
            not isinstance(state, LatticeBoltzmannAMRState)
            or len(state.level_populations) != 2
        ):
            raise TypeError("advance_two_level requires a two-level LBM AMR state.")
        coarse_old, fine_old = state.level_populations
        coarse_active, fine_active = state.active_masks
        dimension = self.transfer.velocity_set.dimension
        expected_fine_shape = tuple(2 * size for size in coarse_old.shape[:-1])
        if fine_old.shape[:-1] != expected_fine_shape:
            raise ValueError("Fine AMR level must refine every coarse extent by two.")
        reshape = []
        for size in coarse_old.shape[:-1]:
            reshape.extend((size, 2))
        blocked_mask = fine_active.reshape(tuple(reshape))
        reduction_axes = tuple(range(1, 2 * dimension, 2))
        any_fine = jnp.any(blocked_mask, axis=reduction_axes)
        covered = jnp.all(blocked_mask, axis=reduction_axes)
        partial = any_fine != covered
        fine_active = eqx.error_if(
            fine_active,
            jnp.any(partial),
            "Fine AMR activity must cover complete ratio-two child blocks.",
        )
        coarse_active = eqx.error_if(
            coarse_active,
            jnp.any(covered & ~coarse_active),
            "Every refined coarse cell must be active.",
        )

        prolonged_old, _ = self.transfer.prolong(coarse_old)
        fine_candidate = jnp.where(fine_active[..., None], fine_old, prolonged_old)
        coarse_candidate = coarse_step(coarse_old, args)
        coarse_candidate = jnp.where(
            coarse_active[..., None], coarse_candidate, coarse_old
        )
        for _ in range(self.fine_substeps):
            stepped = fine_step(fine_candidate, args)
            fine_candidate = jnp.where(fine_active[..., None], stepped, fine_candidate)
        prolonged_candidate, _ = self.transfer.prolong(coarse_candidate)
        fine_for_restriction = jnp.where(
            fine_active[..., None], fine_candidate, prolonged_candidate
        )
        restricted, evidence = self.transfer.restrict(fine_for_restriction)
        corrected_coarse = jnp.where(covered[..., None], restricted, coarse_candidate)
        successful = (
            evidence.successful
            & jnp.all((~coarse_active[..., None]) | jnp.isfinite(corrected_coarse))
            & jnp.all((~fine_active[..., None]) | jnp.isfinite(fine_candidate))
        )
        accepted_coarse = jnp.where(successful, corrected_coarse, coarse_old)
        committed_fine = jnp.where(fine_active[..., None], fine_candidate, fine_old)
        accepted_fine = jnp.where(successful, committed_fine, fine_old)
        candidate_state = LatticeBoltzmannAMRState(
            (accepted_coarse, accepted_fine),
            state.active_masks,
            jnp.asarray(0, dtype=jnp.int32),
        )
        return LatticeBoltzmannAMRAdvanceResult(candidate_state, evidence, successful)

    def advance_two_level_collision_aware(
        self,
        state: LatticeBoltzmannAMRState,
        transfer: PreparedLatticeBoltzmannAMRTransfer,
        temporal: LatticeBoltzmannAMRTemporalInterfacePlan,
        coarse_step: Callable[[Array, Any], Array],
        fine_step: Callable[[Array, Array, Any], Array],
        coarse_relaxation_rate: Array,
        fine_relaxation_rate: Array,
        /,
        *,
        args: Any = None,
    ) -> LatticeBoltzmannCollisionAwareAMRAdvanceResult:
        if (
            not isinstance(state, LatticeBoltzmannAMRState)
            or len(state.level_populations) != 2
        ):
            raise TypeError("Collision-aware AMR requires a two-level state.")
        if not isinstance(transfer, PreparedLatticeBoltzmannAMRTransfer):
            raise TypeError("transfer must be PreparedLatticeBoltzmannAMRTransfer.")
        if not isinstance(temporal, LatticeBoltzmannAMRTemporalInterfacePlan):
            raise TypeError("temporal must be LatticeBoltzmannAMRTemporalInterfacePlan.")
        if transfer.transfer.transfer_id != self.transfer.transfer_id:
            raise ValueError("Prepared and scheduled AMR transfers do not match.")
        coarse_old, fine_old = state.level_populations
        coarse_active, fine_active = state.active_masks
        dimension = self.transfer.velocity_set.dimension
        expected_fine_shape = tuple(2 * size for size in coarse_old.shape[:-1])
        if fine_old.shape[:-1] != expected_fine_shape:
            raise ValueError("Fine AMR level must refine every coarse extent by two.")
        reshape = []
        for size in coarse_old.shape[:-1]:
            reshape.extend((size, 2))
        blocked_mask = fine_active.reshape(tuple(reshape))
        reduction_axes = tuple(range(1, 2 * dimension, 2))
        any_fine = jnp.any(blocked_mask, axis=reduction_axes)
        covered = jnp.all(blocked_mask, axis=reduction_axes)
        fine_active = eqx.error_if(
            fine_active,
            jnp.any(any_fine != covered),
            "Fine AMR activity must cover complete ratio-two child blocks.",
        )
        coarse_active = eqx.error_if(
            coarse_active,
            jnp.any(covered & ~coarse_active),
            "Every refined coarse cell must be active.",
        )
        prolonged_old, prolong_evidence = transfer.prolong(
            coarse_old,
            coarse_relaxation_rate,
            fine_relaxation_rate,
        )
        fine_candidate = jnp.where(fine_active[..., None], fine_old, prolonged_old)
        coarse_candidate = coarse_step(coarse_old, args)
        coarse_candidate = jnp.where(
            coarse_active[..., None], coarse_candidate, coarse_old
        )
        half_coarse = temporal.interpolate(coarse_old, coarse_candidate)
        first_fine = fine_step(fine_candidate, half_coarse, args)
        fine_candidate = jnp.where(fine_active[..., None], first_fine, fine_candidate)
        second_fine = fine_step(fine_candidate, coarse_candidate, args)
        fine_candidate = jnp.where(fine_active[..., None], second_fine, fine_candidate)
        prolonged_candidate, _ = transfer.prolong(
            coarse_candidate,
            coarse_relaxation_rate,
            fine_relaxation_rate,
        )
        fine_for_restriction = jnp.where(
            fine_active[..., None], fine_candidate, prolonged_candidate
        )
        restricted, restriction_evidence = transfer.restrict(
            fine_for_restriction,
            coarse_relaxation_rate,
            fine_relaxation_rate,
        )
        evidence = LatticeBoltzmannAMRInterfaceEvidence(
            restriction_evidence.mass_defect,
            restriction_evidence.momentum_defect,
            restriction_evidence.nonequilibrium_mass_defect,
            restriction_evidence.nonequilibrium_momentum_defect,
            restriction_evidence.minimum_population,
            restriction_evidence.nonequilibrium_scale,
            jnp.asarray(
                temporal.interpolation_fraction,
                dtype=restriction_evidence.minimum_population.dtype,
            ),
            prolong_evidence.successful & restriction_evidence.successful,
        )
        corrected_coarse = jnp.where(covered[..., None], restricted, coarse_candidate)
        successful = (
            evidence.successful
            & jnp.all((~coarse_active[..., None]) | jnp.isfinite(corrected_coarse))
            & jnp.all((~coarse_active[..., None]) | (corrected_coarse >= 0.0))
            & jnp.all((~fine_active[..., None]) | jnp.isfinite(fine_candidate))
            & jnp.all((~fine_active[..., None]) | (fine_candidate >= 0.0))
        )
        accepted_coarse = jnp.where(successful, corrected_coarse, coarse_old)
        committed_fine = jnp.where(fine_active[..., None], fine_candidate, fine_old)
        accepted_fine = jnp.where(successful, committed_fine, fine_old)
        next_state = LatticeBoltzmannAMRState(
            (accepted_coarse, accepted_fine),
            state.active_masks,
            jnp.asarray(0, dtype=jnp.int32),
        )
        return LatticeBoltzmannCollisionAwareAMRAdvanceResult(
            next_state,
            evidence,
            successful,
        )


__all__ = [
    "LatticeBoltzmannAMRAdvanceResult",
    "LatticeBoltzmannAMRInterfaceEvidence",
    "LatticeBoltzmannAMRPlan",
    "LatticeBoltzmannAMRState",
    "LatticeBoltzmannAMRTemporalInterfacePlan",
    "LatticeBoltzmannAMRTransferEvidence",
    "LatticeBoltzmannAMRTransferPlan",
    "LatticeBoltzmannCollisionAwareAMRAdvanceResult",
    "PreparedLatticeBoltzmannAMRTransfer",
]
