#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Relativistic fixed-support quantum dark-sector kinetics."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._uehling_uhlenbeck import (
    QuantumStatistics,
    UehlingUhlenbeckPlan,
    UUCollisionEvidence,
)
from ..relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from ._dark_sector_species import DarkSectorSpeciesPlan


class QuantumKineticStatus(IntEnum):
    """Terminal status for one quantum-kinetic transaction."""

    SUCCESS = 0
    STATE_MISMATCH = 1
    COLLISION_REFUSED = 2
    CONDENSATE_REQUIRED = 3
    CONDENSATE_UNQUALIFIED = 4
    NUMBER_CONSERVATION_FAILURE = 5
    NONFINITE = 6


def equilibrium_occupancy(
    energy: ArrayLike,
    temperature: ArrayLike,
    chemical_potential: ArrayLike,
    statistics: ArrayLike,
    units: RelativisticUnitContract,
    /,
) -> Array:
    """Evaluate classical, Bose--Einstein, or Fermi--Dirac occupancy.

    Energy and chemical potential use ``units.energy_unit``; temperature uses
    ``units.scale.temperature_unit`` and is converted with the contract's exact
    Boltzmann constant. ``statistics`` is broadcast over the leading species
    axis. Bose inputs at or below the chemical potential return ``inf`` rather
    than a clipped surrogate and therefore require an explicit condensate model.
    """

    energy_ = jnp.asarray(energy)
    if not isinstance(units, RelativisticUnitContract):
        raise TypeError("units must be a RelativisticUnitContract.")
    temperature_value = jnp.asarray(temperature, dtype=energy_.dtype)
    boltzmann = units.scale.boltzmann_constant
    temperature_ = (
        temperature_value
        * jnp.asarray(boltzmann.numerator, dtype=energy_.dtype)
        / jnp.asarray(boltzmann.denominator, dtype=energy_.dtype)
    )
    temperature_ = eqx.error_if(
        temperature_,
        jnp.any(~jnp.isfinite(temperature_) | (temperature_ <= 0.0)),
        "temperature must be finite and positive.",
    )
    chemical_ = jnp.asarray(chemical_potential, dtype=energy_.dtype)
    marks = jnp.asarray(statistics)
    if energy_.ndim < 1 or marks.ndim != 1 or energy_.shape[0] != marks.size:
        raise ValueError("energy and statistics must share a leading species axis.")
    if not jnp.issubdtype(marks.dtype, jnp.integer):
        raise TypeError("statistics must use integer marks.")
    trailing = (1,) * (energy_.ndim - 1)
    mark_values = marks.reshape((marks.size, *trailing))
    chemical_values = jnp.broadcast_to(chemical_, (marks.size,)).reshape(
        (marks.size, *trailing)
    )
    exponent = (energy_ - chemical_values) / temperature_
    classical = jnp.exp(-exponent)
    bose = 1.0 / jnp.expm1(exponent)
    fermi = jax.nn.sigmoid(-exponent)
    return jnp.where(
        mark_values == 1, bose, jnp.where(mark_values == -1, fermi, classical)
    )


class QuantumKineticState(StrictModule):
    """Dimensionless occupancy on fixed species/spatial/momentum support.

    The normal component has shape ``(species, spatial, momentum)``. Condensate
    number density is a separate optional ``(species, spatial)`` field and is never
    encoded as an oversized zero-momentum occupation. Inactive support is explicit
    and must contain exact zeros. ``time`` is the bound frame-stage coordinate time;
    collision substeps do not silently advance the geometry stage.
    """

    occupancy: Array
    condensate_density: Array | None
    species: tuple[DarkSectorSpeciesPlan, ...]
    statistics: Array
    spatial_active: Array
    momentum_active: Array
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    time: Array
    frame_token: Array
    species_plan_ids: tuple[str, ...] = eqx.field(static=True)
    statistics_marks: tuple[int, ...] = eqx.field(static=True)
    unit_contract_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    frame_scope: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        occupancy: ArrayLike,
        condensate_density: ArrayLike | None,
        /,
        *,
        species: Sequence[DarkSectorSpeciesPlan],
        statistics: ArrayLike,
        spatial_active: ArrayLike,
        momentum_active: ArrayLike,
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        time: ArrayLike | None = None,
    ):
        species_ = tuple(species)
        if not species_ or any(
            not isinstance(item, DarkSectorSpeciesPlan) for item in species_
        ):
            raise TypeError(
                "species must be a nonempty sequence of DarkSectorSpeciesPlan values."
            )
        if len({item.species_plan_id for item in species_}) != len(species_):
            raise ValueError("Quantum kinetic species identities must be unique.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be a RelativisticUnitContract.")
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame must be a LocalRelativisticFramePlan.")
        if frame.units.contract_id != units.contract_id:
            raise ValueError("Quantum kinetic frame and unit contract disagree.")
        mass_unit_id = units.scale.dimensional_scale.mass_unit.unit_id
        energy_unit_id = units.energy_unit.unit_id
        if any(
            item.mass_unit != mass_unit_id or item.energy_unit != energy_unit_id
            for item in species_
        ):
            raise ValueError(
                "Quantum kinetic species must use the exact relativistic mass "
                "and energy unit IDs."
            )
        value = np.asarray(occupancy)
        marks = np.asarray(statistics)
        space_mask = np.asarray(spatial_active)
        momentum_mask = np.asarray(momentum_active)
        if value.ndim != 3 or value.shape[0] != len(species_):
            raise ValueError("occupancy must have shape (species, spatial, momentum).")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError("occupancy must use a real floating dtype.")
        if marks.shape != (len(species_),) or not np.issubdtype(marks.dtype, np.integer):
            raise ValueError("statistics must contain one integer mark per species.")
        if np.any(~np.isin(marks, (-1, 0, 1))):
            raise ValueError("statistics entries must be FERMI, CLASSICAL, or BOSE.")
        if space_mask.shape != (value.shape[1],) or space_mask.dtype != np.bool_:
            raise ValueError("spatial_active must be a Boolean spatial-support mask.")
        if momentum_mask.shape != (value.shape[2],) or momentum_mask.dtype != np.bool_:
            raise ValueError("momentum_active must be a Boolean momentum-support mask.")
        frame_lane_shape = frame.geometry.leading_shape
        if frame_lane_shape == ():
            frame_scope = "homogeneous-frame-broadcast"
        elif frame_lane_shape == (value.shape[1],):
            frame_scope = "one-frame-lane-per-spatial-cell"
            frame_active = np.asarray(frame.geometry.active)
            if np.any(space_mask & ~frame_active):
                raise ValueError(
                    "Active quantum spatial cells require active local-frame lanes."
                )
        else:
            raise ValueError(
                "The local frame must be scalar or have one lane per spatial cell."
            )
        support_mask = space_mask[None, :, None] & momentum_mask[None, None, :]
        fermi_mask = marks[:, None, None] == int(QuantumStatistics.FERMI)
        if (
            np.any(~np.isfinite(value))
            or np.any(value < 0.0)
            or np.any(fermi_mask & (value > 1.0))
            or np.any(~support_mask & (value != 0.0))
        ):
            raise ValueError(
                "occupancy violates finite, support, positivity, or Fermi bounds."
            )
        condensate = None
        if condensate_density is not None:
            condensate_host = np.asarray(condensate_density, dtype=value.dtype)
            if (
                condensate_host.shape != value.shape[:2]
                or np.any(~np.isfinite(condensate_host))
                or np.any(condensate_host < 0.0)
                or np.any(~space_mask[None, :] & (condensate_host != 0.0))
                or np.any(
                    (marks[:, None] != int(QuantumStatistics.BOSE))
                    & (condensate_host != 0.0)
                )
            ):
                raise ValueError(
                    "condensate_density must be finite, nonnegative, active-support, and Bose-only."
                )
            condensate = jnp.asarray(condensate_host)
        frame_time = np.asarray(frame.time, dtype=value.dtype)
        if frame_time.size == 0 or np.any(~np.isfinite(frame_time)):
            raise ValueError("The local frame must carry finite coordinate time.")
        if not np.all(frame_time == frame_time.reshape((-1,))[0]):
            raise ValueError("Quantum kinetic frame lanes must share one stage time.")
        time_ = (
            np.asarray(frame_time.reshape((-1,))[0], dtype=value.dtype)
            if time is None
            else np.asarray(time, dtype=value.dtype)
        )
        if time_.shape != () or not np.isfinite(time_) or not np.all(frame_time == time_):
            raise ValueError("time must equal the exact local-frame stage time.")
        ids = tuple(item.species_plan_id for item in species_)
        frame_realization_id = frame.realization_id()
        support_id = canonical_fingerprint(
            {
                "kind": "quantum-dark-kinetic-support",
                "species": list(ids),
                "statistics": array_tree_fingerprint(marks),
                "shape": list(value.shape),
                "spatial_active": array_tree_fingerprint(space_mask),
                "momentum_active": array_tree_fingerprint(momentum_mask),
                "unit_contract": units.contract_id,
                "frame": frame.frame_id,
                "frame_realization": frame_realization_id,
                "frame_scope": frame_scope,
                "condensate_field": condensate_density is not None,
            }
        )
        self.occupancy = jnp.asarray(value)
        self.condensate_density = condensate
        self.species = species_
        self.statistics = jax.lax.stop_gradient(jnp.asarray(marks, dtype=jnp.int8))
        self.spatial_active = jax.lax.stop_gradient(jnp.asarray(space_mask))
        self.momentum_active = jax.lax.stop_gradient(jnp.asarray(momentum_mask))
        self.units = units
        self.frame = frame
        self.time = jnp.asarray(time_)
        self.frame_token = jax.lax.stop_gradient(jnp.asarray(frame.frame_token))
        self.statistics_marks = tuple(int(value) for value in marks)
        self.species_plan_ids = ids
        self.unit_contract_id = units.contract_id
        self.frame_id = frame.frame_id
        self.support_id = support_id
        self.frame_realization_id = frame_realization_id
        self.frame_scope = frame_scope


class CondensateCouplingPlan(StrictModule, NonTrainableState):
    """Qualified number-conserving Bose condensate/normal relaxation."""

    species_index: int = eqx.field(static=True)
    momentum_index: int = eqx.field(static=True)
    critical_occupancy: float = eqx.field(static=True)
    relaxation_rate: float = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)

    def __init__(
        self,
        species_index: int,
        momentum_index: int,
        critical_occupancy: float,
        relaxation_rate: float,
        /,
        *,
        qualification_id: str,
        model_id: str,
    ):
        species_index_ = int(species_index)
        momentum_index_ = int(momentum_index)
        critical = float(critical_occupancy)
        rate = float(relaxation_rate)
        qualification = str(qualification_id).strip()
        model = str(model_id).strip()
        if species_index_ < 0 or momentum_index_ < 0:
            raise ValueError("Condensate support indices must be nonnegative.")
        if not np.isfinite(critical) or critical <= 0.0:
            raise ValueError("critical_occupancy must be finite and positive.")
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError("relaxation_rate must be finite and positive.")
        if not qualification or not model:
            raise ValueError(
                "Condensate coupling requires qualification and model identities."
            )
        self.species_index = species_index_
        self.momentum_index = momentum_index_
        self.critical_occupancy = critical
        self.relaxation_rate = rate
        self.qualification_id = qualification
        self.model_id = model

    def apply(
        self,
        occupancy: Array,
        condensate_density: Array,
        phase_space_weights: Array,
        step_size: Array,
        /,
    ) -> tuple[Array, Array, Array]:
        """Relax the selected normal bin while preserving total particle number."""

        species = self.species_index
        momentum = self.momentum_index
        weight = phase_space_weights[species, momentum]
        normal_before = weight * occupancy[species, :, momentum]
        condensate_before = condensate_density[species]
        total = normal_before + condensate_before
        normal_equilibrium = jnp.minimum(total, weight * self.critical_occupancy)
        factor = jnp.exp(-self.relaxation_rate * step_size)
        normal_after = normal_equilibrium + factor * (normal_before - normal_equilibrium)
        condensate_after = total - normal_after
        occupancy_after = occupancy.at[species, :, momentum].set(normal_after / weight)
        density_after = condensate_density.at[species].set(condensate_after)
        return occupancy_after, density_after, condensate_after - condensate_before


class QuantumKineticEvidence(StrictModule):
    """Collision, condensate, and transactional evidence for one step."""

    collision: UUCollisionEvidence
    condensate_transfer: Array
    number_defect: Array
    condensate_required: Array
    finite: Array
    state_compatible: Array
    successful: Array
    status: Array
    qualification_id: str | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class QuantumKineticResult(StrictModule):
    """Proposed and accepted quantum kinetic states."""

    candidate_state: QuantumKineticState
    accepted_state: QuantumKineticState
    evidence: QuantumKineticEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


class QuantumDarkKineticsPlan(StrictModule, NonTrainableState):
    """Relativistic quantum collision and qualified-condensate transaction."""

    species: tuple[DarkSectorSpeciesPlan, ...]
    units: RelativisticUnitContract
    frame: LocalRelativisticFramePlan
    collision: UehlingUhlenbeckPlan
    condensate_coupling: CondensateCouplingPlan | None
    condensation_threshold: float | None = eqx.field(static=True)
    species_plan_ids: tuple[str, ...] = eqx.field(static=True)
    statistics_marks: tuple[int, ...] = eqx.field(static=True)
    charge_names: tuple[str, ...] = eqx.field(static=True)
    frame_realization_id: str = eqx.field(static=True)
    frame_lane_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: Sequence[DarkSectorSpeciesPlan],
        units: RelativisticUnitContract,
        frame: LocalRelativisticFramePlan,
        collision: UehlingUhlenbeckPlan,
        /,
        *,
        condensate_coupling: CondensateCouplingPlan | None = None,
        condensation_threshold: float | None = None,
    ):
        species_ = tuple(species)
        if not species_ or any(
            not isinstance(item, DarkSectorSpeciesPlan) for item in species_
        ):
            raise TypeError(
                "species must be a nonempty sequence of DarkSectorSpeciesPlan values."
            )
        if len({item.species_plan_id for item in species_}) != len(species_):
            raise ValueError("Quantum kinetic species identities must be unique.")
        if not isinstance(units, RelativisticUnitContract):
            raise TypeError("units must be a RelativisticUnitContract.")
        if (
            not isinstance(frame, LocalRelativisticFramePlan)
            or frame.units.contract_id != units.contract_id
        ):
            raise ValueError(
                "frame must use the exact supplied relativistic unit contract."
            )
        if not isinstance(collision, UehlingUhlenbeckPlan):
            raise TypeError("collision must be a UehlingUhlenbeckPlan.")
        if collision.species_count != len(species_):
            raise ValueError(
                "Collision species support disagrees with species identities."
            )
        if collision.time_unit_id != units.scale.dimensional_scale.time_unit.unit_id:
            raise ValueError(
                "Collision kernels and steps must use the relativistic scale time unit."
            )
        mass_unit_id = units.scale.dimensional_scale.mass_unit.unit_id
        energy_unit_id = units.energy_unit.unit_id
        if any(
            item.mass_unit != mass_unit_id or item.energy_unit != energy_unit_id
            for item in species_
        ):
            raise ValueError(
                "Quantum kinetic species must declare the exact relativistic mass "
                "and energy unit IDs."
            )
        charge_names = species_[0].charge_names
        if any(item.charge_names != charge_names for item in species_):
            raise ValueError("Quantum kinetic species charge axes must agree exactly.")
        declared_charges = np.stack(
            tuple(np.asarray(item.charges) for item in species_), axis=0
        )
        if collision.charge_count != len(charge_names) or not np.array_equal(
            np.asarray(collision.charges), declared_charges
        ):
            raise ValueError(
                "Collision conserved charges disagree with the species contracts."
            )
        rest_masses = jnp.asarray(tuple(item.mass for item in species_))[:, None]
        if not bool(
            np.all(
                np.asarray(
                    units.mass_shell_admissible(
                        collision.four_momenta,
                        rest_masses,
                        relative_tolerance=1.0e-10,
                    )
                )
            )
        ):
            raise ValueError(
                "Collision four-momentum support violates the species mass shells."
            )
        if condensate_coupling is not None:
            if not isinstance(condensate_coupling, CondensateCouplingPlan):
                raise TypeError(
                    "condensate_coupling must be CondensateCouplingPlan or None."
                )
            if (
                condensate_coupling.species_index >= len(species_)
                or condensate_coupling.momentum_index >= collision.momentum_count
                or int(collision.statistics[condensate_coupling.species_index])
                != int(QuantumStatistics.BOSE)
            ):
                raise ValueError("Condensate coupling must select a supported Bose bin.")
        threshold = (
            None if condensation_threshold is None else float(condensation_threshold)
        )
        if threshold is not None and (not np.isfinite(threshold) or threshold <= 0.0):
            raise ValueError(
                "condensation_threshold must be finite and positive or None."
            )
        ids = tuple(item.species_plan_id for item in species_)
        frame_realization_id = frame.realization_id()
        self.species = species_
        self.units = units
        self.frame = frame
        self.collision = collision
        self.condensate_coupling = condensate_coupling
        self.condensation_threshold = threshold
        self.statistics_marks = tuple(
            int(value) for value in np.asarray(collision.statistics)
        )
        self.species_plan_ids = ids
        self.charge_names = charge_names
        self.frame_realization_id = frame_realization_id
        self.frame_lane_shape = frame.geometry.leading_shape
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-quantum-dark-kinetics",
                "species": list(ids),
                "unit_contract": units.contract_id,
                "frame": frame.frame_id,
                "collision": collision.plan_id,
                "frame_realization": frame_realization_id,
                "frame_lane_shape": list(frame.geometry.leading_shape),
                "condensate_coupling": (
                    None
                    if condensate_coupling is None
                    else {
                        "model": condensate_coupling.model_id,
                        "qualification": condensate_coupling.qualification_id,
                    }
                ),
                "condensation_threshold": threshold,
            }
        )

    def initialize(
        self,
        occupancy: ArrayLike,
        /,
        *,
        spatial_active: ArrayLike,
        momentum_active: ArrayLike,
        condensate_density: ArrayLike | None = None,
        time: ArrayLike | None = None,
    ) -> QuantumKineticState:
        """Validate and bind a state to this exact compiled support."""

        state = QuantumKineticState(
            occupancy,
            condensate_density,
            species=self.species,
            statistics=self.collision.statistics,
            spatial_active=spatial_active,
            momentum_active=momentum_active,
            units=self.units,
            frame=self.frame,
            time=time,
        )
        if state.occupancy.shape[2] != self.collision.momentum_count:
            raise ValueError("State momentum support disagrees with collision support.")
        active_event_momenta = np.asarray(self.collision.event_momenta)[
            np.asarray(self.collision.event_active)
        ]
        if np.any(~np.asarray(state.momentum_active)[active_event_momenta]):
            raise ValueError(
                "Active collision events reference inactive momentum support."
            )
        return state

    def _state_with(
        self,
        state: QuantumKineticState,
        occupancy: Array,
        condensate_density: Array | None,
        time: Array,
        /,
    ) -> QuantumKineticState:
        return eqx.tree_at(
            lambda value: (value.occupancy, value.condensate_density, value.time),
            state,
            (occupancy, condensate_density, time),
        )

    def _number(self, occupancy: Array, condensate: Array | None, /) -> Array:
        normal = ein.contract("sxp,sp->", occupancy, self.collision.phase_space_weights)
        return normal if condensate is None else normal + jnp.sum(condensate)

    def advance(
        self, state: QuantumKineticState, step_size: ArrayLike, /
    ) -> QuantumKineticResult:
        """Advance collisions and optional condensate coupling with atomic rollback."""

        if not isinstance(state, QuantumKineticState):
            raise TypeError("state must be a QuantumKineticState.")
        compatible = (
            state.species_plan_ids == self.species_plan_ids
            and state.statistics_marks == self.statistics_marks
            and state.unit_contract_id == self.units.contract_id
            and state.frame_id == self.frame.frame_id
            and state.frame_realization_id == self.frame_realization_id
            and state.occupancy.shape[2] == self.collision.momentum_count
        )
        if not compatible:
            raise ValueError("Quantum kinetic state does not belong to this plan.")
        step = jnp.asarray(step_size, dtype=state.occupancy.dtype)
        collision_result = self.collision.advance(state.occupancy, step)
        occupancy = collision_result.candidate_occupancy
        condensate = state.condensate_density
        transfer = jnp.zeros(state.occupancy.shape[:2], dtype=state.occupancy.dtype)
        coupling_qualified = jnp.asarray(True)
        if self.condensate_coupling is not None:
            if condensate is None:
                coupling_qualified = jnp.asarray(False)
            else:
                occupancy, condensate, selected_transfer = self.condensate_coupling.apply(
                    occupancy,
                    condensate,
                    self.collision.phase_space_weights,
                    step,
                )
                transfer = transfer.at[self.condensate_coupling.species_index].set(
                    selected_transfer
                )
        bose = self.collision.statistics[:, None, None] == int(QuantumStatistics.BOSE)
        threshold_exceeded = jnp.asarray(False)
        if self.condensation_threshold is not None:
            threshold_exceeded = jnp.any(
                jnp.where(bose, occupancy > self.condensation_threshold, False)
            )
        condensate_required = threshold_exceeded & (
            (self.condensate_coupling is None) | (state.condensate_density is None)
        )
        number_before = self._number(state.occupancy, state.condensate_density)
        number_after = self._number(occupancy, condensate)
        number_defect = number_after - number_before
        number_valid = jnp.abs(
            number_defect
        ) <= self.collision.invariant_tolerance * jnp.maximum(jnp.abs(number_before), 1.0)
        finite = jnp.all(jnp.isfinite(occupancy)) & (
            jnp.asarray(True) if condensate is None else jnp.all(jnp.isfinite(condensate))
        )
        successful = (
            collision_result.successful
            & coupling_qualified
            & ~condensate_required
            & number_valid
            & finite
        )
        candidate = self._state_with(state, occupancy, condensate, state.time)
        accepted = jax.lax.cond(
            successful, lambda _: candidate, lambda _: state, operand=None
        )
        status = jnp.where(
            successful,
            int(QuantumKineticStatus.SUCCESS),
            jnp.where(
                ~collision_result.successful,
                int(QuantumKineticStatus.COLLISION_REFUSED),
                jnp.where(
                    condensate_required,
                    int(QuantumKineticStatus.CONDENSATE_REQUIRED),
                    jnp.where(
                        ~coupling_qualified,
                        int(QuantumKineticStatus.CONDENSATE_UNQUALIFIED),
                        jnp.where(
                            ~number_valid,
                            int(QuantumKineticStatus.NUMBER_CONSERVATION_FAILURE),
                            int(QuantumKineticStatus.NONFINITE),
                        ),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        evidence = QuantumKineticEvidence(
            collision_result.evidence,
            transfer,
            number_defect,
            condensate_required,
            finite,
            jnp.asarray(compatible),
            successful,
            status,
            None
            if self.condensate_coupling is None
            else self.condensate_coupling.qualification_id,
            self.plan_id,
        )
        return QuantumKineticResult(
            candidate, accepted, evidence, successful, self.plan_id
        )


__all__ = [
    "CondensateCouplingPlan",
    "QuantumDarkKineticsPlan",
    "QuantumKineticEvidence",
    "QuantumKineticResult",
    "QuantumKineticState",
    "QuantumKineticStatus",
    "equilibrium_occupancy",
]
