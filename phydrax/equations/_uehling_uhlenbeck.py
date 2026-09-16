#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-support Uehling--Uhlenbeck ``2 <-> 2`` collision operator."""

from __future__ import annotations

from enum import IntEnum
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class QuantumStatistics(IntEnum):
    """Occupation-number statistics used by one fixed species."""

    FERMI = -1
    CLASSICAL = 0
    BOSE = 1


class UUCollisionStatus(IntEnum):
    """Terminal status for a transactional collision update."""

    SUCCESS = 0
    INVALID_STATE = 1
    INVALID_STEP = 2
    SUBSTEP_CAPACITY_EXHAUSTED = 3
    OCCUPANCY_BOUND_FAILURE = 4
    INVARIANT_FAILURE = 5
    ENTROPY_FAILURE = 6


class UUCollisionEvidence(StrictModule):
    """Pointwise flux, invariant, H-theorem, and transaction evidence."""

    event_flux: Array
    detailed_balance_residual: Array
    charge_defect: Array
    four_momentum_defect: Array
    entropy_before: Array
    entropy_after: Array
    entropy_production: Array
    minimum_occupancy: Array
    maximum_fermi_occupancy: Array
    required_substeps: Array
    used_substeps: Array
    finite: Array
    occupancy_valid: Array
    invariants_valid: Array
    entropy_valid: Array
    successful: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class UUCollisionResult(StrictModule):
    """Candidate and atomically accepted dimensionless occupancies."""

    candidate_occupancy: Array
    accepted_occupancy: Array
    evidence: UUCollisionEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class UehlingUhlenbeckPlan(StrictModule, NonTrainableState):
    """One conservative fixed-capacity quantum ``2 <-> 2`` event stencil.

    Occupancies are dimensionless and have layout ``(species, spatial, momentum)``.
    Each active event names four ``(species, momentum)`` bins in incoming,
    incoming, outgoing, outgoing order. A single microreversible kernel multiplies
    both directions, so pointwise detailed balance is not weakened by independent
    forward and reverse tables. ``event_kernels`` use inverse ``time_unit_id`` and
    ``step_size`` uses that time unit. Phase-space weights are divided out when an
    event extent is scattered and multiplied back into every pointwise moment.
    """

    statistics: Array
    phase_space_weights: Array
    event_species: Array
    event_momenta: Array
    event_kernels: Array
    event_active: Array
    four_momenta: Array
    charges: Array
    maximum_substeps: int = eqx.field(static=True)
    substep_safety: float = eqx.field(static=True)
    invariant_tolerance: float = eqx.field(static=True)
    entropy_tolerance: float = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    momentum_count: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    charge_count: int = eqx.field(static=True)
    time_unit_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        statistics: ArrayLike,
        phase_space_weights: ArrayLike,
        event_species: ArrayLike,
        event_momenta: ArrayLike,
        event_kernels: ArrayLike,
        four_momenta: ArrayLike,
        charges: ArrayLike,
        /,
        *,
        time_unit_id: str,
        event_active: ArrayLike | None = None,
        maximum_substeps: int = 64,
        substep_safety: float = 0.8,
        invariant_tolerance: float = 1.0e-10,
        entropy_tolerance: float = 1.0e-10,
    ):
        statistics_ = np.asarray(statistics)
        weights = np.asarray(phase_space_weights, dtype=float)
        species = np.asarray(event_species)
        momenta = np.asarray(event_momenta)
        kernels = np.asarray(event_kernels, dtype=float)
        momenta4 = np.asarray(four_momenta, dtype=float)
        charges_ = np.asarray(charges, dtype=float)
        time_unit_id_ = str(time_unit_id).strip()
        if not time_unit_id_:
            raise ValueError("time_unit_id must be a nonempty explicit unit identity.")
        if statistics_.ndim != 1 or statistics_.size == 0:
            raise ValueError("statistics must contain one mark per species.")
        if not np.issubdtype(statistics_.dtype, np.integer):
            raise TypeError("statistics must use integer QuantumStatistics marks.")
        if np.any(~np.isin(statistics_, (-1, 0, 1))):
            raise ValueError("statistics entries must be FERMI, CLASSICAL, or BOSE.")
        species_count = int(statistics_.size)
        if (
            weights.ndim != 2
            or weights.shape[0] != species_count
            or weights.shape[1] == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "phase_space_weights must be finite positive species-by-momentum values."
            )
        momentum_count = int(weights.shape[1])
        if species.ndim != 2 or species.shape[1] != 4:
            raise ValueError("event_species must have shape (event_capacity, 4).")
        if not np.issubdtype(species.dtype, np.integer):
            raise TypeError("event_species must use integer indices.")
        if momenta.shape != species.shape or not np.issubdtype(momenta.dtype, np.integer):
            raise ValueError(
                "event_momenta must be integer indices matching event_species."
            )
        capacity = int(species.shape[0])
        if capacity == 0 or kernels.shape != (capacity,):
            raise ValueError(
                "event_kernels must have one value per nonempty event capacity."
            )
        active = (
            np.ones((capacity,), dtype=bool)
            if event_active is None
            else np.asarray(event_active)
        )
        if active.shape != (capacity,) or active.dtype != np.bool_:
            raise ValueError("event_active must be a Boolean event-capacity mask.")
        active_species = species[active]
        active_momenta = momenta[active]
        if (
            np.any(active_species < 0)
            or np.any(active_species >= species_count)
            or np.any(active_momenta < 0)
            or np.any(active_momenta >= momentum_count)
        ):
            raise ValueError("Active collision-event indices exceed fixed support.")
        if np.any(~active & ((species != 0).any(axis=1) | (momenta != 0).any(axis=1))):
            raise ValueError(
                "Inactive collision-event slots must use zero sentinel indices."
            )
        if (
            np.any(~np.isfinite(kernels))
            or np.any(kernels < 0.0)
            or np.any(~active & (kernels != 0.0))
        ):
            raise ValueError(
                "Event kernels must be finite, nonnegative, and zero when inactive."
            )
        if momenta4.shape != (species_count, momentum_count, 4) or np.any(
            ~np.isfinite(momenta4)
        ):
            raise ValueError(
                "four_momenta must have shape (species, momentum, 4) and be finite."
            )
        if (
            charges_.ndim != 2
            or charges_.shape[0] != species_count
            or np.any(~np.isfinite(charges_))
        ):
            raise ValueError("charges must be a finite species-by-charge array.")
        if isinstance(maximum_substeps, bool) or not isinstance(
            maximum_substeps, Integral
        ):
            raise TypeError("maximum_substeps must be an integer.")
        maximum = int(maximum_substeps)
        safety = float(substep_safety)
        invariant_tol = float(invariant_tolerance)
        entropy_tol = float(entropy_tolerance)
        if maximum <= 0:
            raise ValueError("maximum_substeps must be positive.")
        if not np.isfinite(safety) or not 0.0 < safety < 1.0:
            raise ValueError("substep_safety must lie in (0, 1).")
        if not np.isfinite(invariant_tol) or invariant_tol <= 0.0:
            raise ValueError("invariant_tolerance must be finite and positive.")
        if not np.isfinite(entropy_tol) or entropy_tol < 0.0:
            raise ValueError("entropy_tolerance must be finite and nonnegative.")
        stoichiometry = np.asarray((-1.0, -1.0, 1.0, 1.0))
        selected_four = momenta4[species, momenta]
        four_defect = np.sum(stoichiometry[None, :, None] * selected_four, axis=1)
        selected_charge = charges_[species]
        charge_defect = np.sum(stoichiometry[None, :, None] * selected_charge, axis=1)
        active_invariant_defect = max(
            float(np.max(np.abs(four_defect[active]), initial=0.0)),
            float(np.max(np.abs(charge_defect[active]), initial=0.0)),
        )
        if active_invariant_defect > invariant_tol:
            raise ValueError(
                "Active collision events violate declared charge or four-momentum conservation."
            )
        self.statistics = jax.lax.stop_gradient(jnp.asarray(statistics_, dtype=jnp.int8))
        self.phase_space_weights = jax.lax.stop_gradient(jnp.asarray(weights))
        self.event_species = jax.lax.stop_gradient(jnp.asarray(species, dtype=jnp.int32))
        self.event_momenta = jax.lax.stop_gradient(jnp.asarray(momenta, dtype=jnp.int32))
        self.event_kernels = jax.lax.stop_gradient(jnp.asarray(kernels))
        self.event_active = jax.lax.stop_gradient(jnp.asarray(active))
        self.four_momenta = jax.lax.stop_gradient(jnp.asarray(momenta4))
        self.charges = jax.lax.stop_gradient(jnp.asarray(charges_))
        self.maximum_substeps = maximum
        self.substep_safety = safety
        self.invariant_tolerance = invariant_tol
        self.entropy_tolerance = entropy_tol
        self.species_count = species_count
        self.momentum_count = momentum_count
        self.event_capacity = capacity
        self.charge_count = int(charges_.shape[1])
        self.time_unit_id = time_unit_id_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-support-uehling-uhlenbeck-2to2",
                "arrays": array_tree_fingerprint(
                    (
                        statistics_,
                        weights,
                        species,
                        momenta,
                        kernels,
                        active,
                        momenta4,
                        charges_,
                    )
                ),
                "time_unit_id": time_unit_id_,
                "maximum_substeps": maximum,
                "substep_safety": safety,
                "invariant_tolerance": invariant_tol,
                "entropy_tolerance": entropy_tol,
            }
        )

    @property
    def fixed_stencil_resources(self) -> dict[str, int]:
        """Return shape-stable resource counts without compiling the operator."""

        return {
            "species": self.species_count,
            "momentum_nodes": self.momentum_count,
            "event_capacity": self.event_capacity,
            "active_events": int(np.count_nonzero(np.asarray(self.event_active))),
            "maximum_substeps": self.maximum_substeps,
        }

    def _occupancy(self, occupancy: ArrayLike, /) -> Array:
        value = jnp.asarray(occupancy)
        if (
            value.ndim != 3
            or value.shape[0] != self.species_count
            or value.shape[2] != self.momentum_count
        ):
            raise ValueError(
                "occupancy must have fixed layout (species, spatial, momentum)."
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("occupancy must use a real floating dtype.")
        return value

    def admissible(self, occupancy: ArrayLike, /) -> Array:
        """Return one scalar bound/finite predicate for dimensionless occupancies."""

        value = self._occupancy(occupancy)
        marks = self.statistics[:, None, None]
        return (
            jnp.all(jnp.isfinite(value))
            & jnp.all(value >= 0.0)
            & jnp.all(
                jnp.where(marks == int(QuantumStatistics.FERMI), value <= 1.0, True)
            )
        )

    def event_flux(self, occupancy: ArrayLike, /) -> Array:
        """Evaluate the shared forward-minus-reverse event flux at every point."""

        value = self._occupancy(occupancy)
        legs = tuple(
            value[self.event_species[:, leg], :, self.event_momenta[:, leg]].T
            for leg in range(4)
        )
        marks = tuple(
            self.statistics[self.event_species[:, leg]][None, :] for leg in range(4)
        )
        factors = tuple(1.0 + marks[leg] * legs[leg] for leg in range(4))
        forward = legs[0] * legs[1] * factors[2] * factors[3]
        reverse = legs[2] * legs[3] * factors[0] * factors[1]
        return jnp.where(
            self.event_active[None, :],
            self.event_kernels[None, :] * (forward - reverse),
            0.0,
        )

    def detailed_balance_residual(self, occupancy: ArrayLike, /) -> Array:
        """Return absolute pointwise forward/reverse imbalance per event."""

        return jnp.abs(self.event_flux(occupancy))

    def collision_rate(self, occupancy: ArrayLike, /) -> Array:
        """Scatter conservative event extents to ``df/dt`` on fixed support."""

        value = self._occupancy(occupancy)
        flux = self.event_flux(value)
        flat = jnp.zeros(
            (value.shape[1], self.species_count * self.momentum_count),
            dtype=value.dtype,
        )
        bins = self.event_species * self.momentum_count + self.event_momenta
        weights = self.phase_space_weights[self.event_species, self.event_momenta]
        stoichiometry = jnp.asarray((-1.0, -1.0, 1.0, 1.0), dtype=value.dtype)
        for leg in range(4):
            flat = flat.at[:, bins[:, leg]].add(
                flux * stoichiometry[leg] / weights[:, leg][None, :]
            )
        return jnp.transpose(
            flat.reshape((value.shape[1], self.species_count, self.momentum_count)),
            (1, 0, 2),
        )

    def entropy(self, occupancy: ArrayLike, /) -> Array:
        """Return quantum kinetic entropy independently at each spatial point."""

        value = self._occupancy(occupancy)
        tiny = jnp.finfo(value.dtype).tiny
        f = jnp.maximum(value, tiny)
        marks = self.statistics[:, None, None]
        bose = (1.0 + value) * jnp.log1p(value) - value * jnp.log(f)
        vacancy = jnp.maximum(1.0 - value, tiny)
        fermi = -value * jnp.log(f) - (1.0 - value) * jnp.log(vacancy)
        classical = value - value * jnp.log(f)
        density = jnp.where(marks == 1, bose, jnp.where(marks == -1, fermi, classical))
        return ein.contract("sxp,sp->x", density, self.phase_space_weights)

    def conserved_moments(self, occupancy: ArrayLike, /) -> tuple[Array, Array]:
        """Return charge and contravariant four-momentum at every spatial point."""

        value = self._occupancy(occupancy)
        integrated = ein.contract("sxp,sp->xs", value, self.phase_space_weights)
        charge = ein.contract("xs,sc->xc", integrated, self.charges)
        four = ein.contract(
            "sxp,sp,spm->xm", value, self.phase_space_weights, self.four_momenta
        )
        return charge, four

    def _required_substeps(self, occupancy: Array, step: Array, /) -> Array:
        rate = self.collision_rate(occupancy)
        tiny = jnp.finfo(occupancy.dtype).tiny
        depletion = jnp.max(
            jnp.where(rate < 0.0, -rate / jnp.maximum(occupancy, tiny), 0.0)
        )
        fermi = self.statistics[:, None, None] == int(QuantumStatistics.FERMI)
        filling = jnp.max(
            jnp.where(
                fermi & (rate > 0.0),
                rate / jnp.maximum(1.0 - occupancy, tiny),
                0.0,
            )
        )
        stiffness = jnp.maximum(depletion, filling)
        raw = jnp.ceil(step * stiffness / self.substep_safety)
        return jnp.maximum(jnp.asarray(1, dtype=jnp.int32), raw.astype(jnp.int32))

    def advance(self, occupancy: ArrayLike, step_size: ArrayLike, /) -> UUCollisionResult:
        """Advance with fixed-capacity subcycling and all-or-nothing rollback."""

        initial = self._occupancy(occupancy)
        step = jnp.asarray(step_size, dtype=initial.dtype)
        if step.shape != ():
            raise ValueError("step_size must be scalar.")
        initial_valid = self.admissible(initial)
        step_valid = jnp.isfinite(step) & (step >= 0.0)
        required = jnp.where(step_valid, self._required_substeps(initial, step), 1)
        within_capacity = required <= self.maximum_substeps
        used = jnp.minimum(required, self.maximum_substeps)
        substep = jnp.where(used > 0, step / used.astype(initial.dtype), 0.0)

        def body(index, current):
            candidate = current + substep * self.collision_rate(current)
            return jnp.where(index < used, candidate, current)

        candidate = jax.lax.fori_loop(0, self.maximum_substeps, body, initial)
        finite = jnp.all(jnp.isfinite(candidate))
        occupancy_valid = self.admissible(candidate)
        charge_before, four_before = self.conserved_moments(initial)
        charge_after, four_after = self.conserved_moments(candidate)
        charge_defect = charge_after - charge_before
        four_defect = four_after - four_before
        invariant_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(charge_before), initial=0.0),
                jnp.max(jnp.abs(four_before)),
            ),
            1.0,
        )
        invariants_valid = (
            jnp.max(jnp.abs(charge_defect), initial=0.0)
            <= self.invariant_tolerance * invariant_scale
        ) & (jnp.max(jnp.abs(four_defect)) <= self.invariant_tolerance * invariant_scale)
        entropy_before = self.entropy(initial)
        entropy_after = self.entropy(candidate)
        entropy_production = entropy_after - entropy_before
        entropy_valid = jnp.all(
            entropy_production
            >= -self.entropy_tolerance * jnp.maximum(jnp.abs(entropy_before), 1.0)
        )
        successful = (
            initial_valid
            & step_valid
            & within_capacity
            & finite
            & occupancy_valid
            & invariants_valid
            & entropy_valid
        )
        accepted = jnp.where(successful, candidate, initial)
        fermi_values = jnp.where(
            self.statistics[:, None, None] == int(QuantumStatistics.FERMI),
            candidate,
            -jnp.inf,
        )
        status = jnp.where(
            successful,
            int(UUCollisionStatus.SUCCESS),
            jnp.where(
                ~initial_valid,
                int(UUCollisionStatus.INVALID_STATE),
                jnp.where(
                    ~step_valid,
                    int(UUCollisionStatus.INVALID_STEP),
                    jnp.where(
                        ~within_capacity,
                        int(UUCollisionStatus.SUBSTEP_CAPACITY_EXHAUSTED),
                        jnp.where(
                            ~finite | ~occupancy_valid,
                            int(UUCollisionStatus.OCCUPANCY_BOUND_FAILURE),
                            jnp.where(
                                ~invariants_valid,
                                int(UUCollisionStatus.INVARIANT_FAILURE),
                                int(UUCollisionStatus.ENTROPY_FAILURE),
                            ),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        evidence = UUCollisionEvidence(
            self.event_flux(initial),
            self.detailed_balance_residual(initial),
            charge_defect,
            four_defect,
            entropy_before,
            entropy_after,
            entropy_production,
            jnp.min(candidate),
            jnp.max(fermi_values),
            required,
            used,
            finite,
            occupancy_valid,
            invariants_valid,
            entropy_valid,
            successful,
            status,
            self.plan_id,
        )
        return UUCollisionResult(candidate, accepted, evidence, successful, self.plan_id)


__all__ = [
    "QuantumStatistics",
    "UUCollisionEvidence",
    "UUCollisionResult",
    "UUCollisionStatus",
    "UehlingUhlenbeckPlan",
]
