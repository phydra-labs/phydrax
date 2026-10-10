"""Physical step acceptance, separate from solver trial admissibility.

The nonlinear solver refuses an unphysical Newton trial on its own; the checks
here judge a finished candidate step and record why it was or was not accepted.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import IntEnum
from typing import Final, TYPE_CHECKING

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...typing import parse
from ._bundles import FuelModelBundle
from ._coolant import PrescribedWallTemperature, SingleChannelCoolant
from ._coupled_coolant import CoupledChannelCoolant
from ._identity import FUEL_GAS_SPECIES
from ._inventory import FuelGasLedger
from ._layout import CONTACT_PRESSURE_SCALE_PA, FuelSourceSlot, SOURCE_SLOTS
from ._state import FuelRodState


if TYPE_CHECKING:
    from ._problem import FuelRodCandidateResult, PreparedFuelRod


class FuelStepStatus(IntEnum):
    """Outcome code of one attempted step.

    ``ACCEPTED`` is zero. A refused step reports its highest-priority failed
    check: solver and coupling failures first, then local roots, physical
    admissibility, admitted range, energy and gas balance, contact, and the
    increment limits.
    """

    ACCEPTED = 0
    NONLINEAR_FAILURE = 1
    LOCAL_ROOT_FAILURE = 2
    PHYSICAL_FAILURE = 3
    TERMINAL_DOMAIN_EXIT = 4
    ENERGY_IMBALANCE = 5
    GAS_IMBALANCE = 6
    CONTACT_FAILURE = 7
    INCREMENT_LIMIT = 8
    COUPLING_FAILURE = 9
    JOINT_RESIDUAL_FAILURE = 10


# Failure statuses, highest priority first. When several checks fail, the step
# reports the first failing entry: a failed solve outranks every physical check
# because its candidate carries no physical meaning.
_FAILURE_PRIORITY: Final[tuple[FuelStepStatus, ...]] = (
    FuelStepStatus.NONLINEAR_FAILURE,
    FuelStepStatus.COUPLING_FAILURE,
    FuelStepStatus.JOINT_RESIDUAL_FAILURE,
    FuelStepStatus.LOCAL_ROOT_FAILURE,
    FuelStepStatus.PHYSICAL_FAILURE,
    FuelStepStatus.TERMINAL_DOMAIN_EXIT,
    FuelStepStatus.ENERGY_IMBALANCE,
    FuelStepStatus.GAS_IMBALANCE,
    FuelStepStatus.CONTACT_FAILURE,
    FuelStepStatus.INCREMENT_LIMIT,
)

# Accepted contact pressure may undershoot zero by solver roundoff [Pa].
CONTACT_PRESSURE_ROUNDOFF_PA: Final[float] = 1.0e-3
# Largest accepted normalized friction-row residual.
FRICTION_RESIDUAL_TOLERANCE: Final[float] = 1.0e-8
# Deposited energy below this floor [J] is judged by an absolute tolerance.
ENERGY_TOLERANCE_FLOOR_J: Final[float] = 1.0

# Krypton and xenon, the fission gases whose release fraction is limited.
_FISSION_GAS_SPECIES: Final[slice] = slice(
    FUEL_GAS_SPECIES.index("Kr"), FUEL_GAS_SPECIES.index("Xe") + 1
)


class FuelAcceptancePolicy(StrictModule):
    """Limits a converged candidate step must satisfy to be accepted.

    Increment limits bound one step's largest change of temperature [K],
    burnup [MWd/kgHM], Kr+Xe release fraction, and creep log strain. The energy
    balance is relative to the deposited energy (at least 1 J); the gas balance
    is absolute [mol]; ``contact_tolerance`` [m] bounds gap penetration and
    gap-pressure complementarity.
    """

    temperature_increment: float = eqx.field(static=True)
    burnup_increment: float = eqx.field(static=True)
    release_fraction_increment: float = eqx.field(static=True)
    creep_increment: float = eqx.field(static=True)
    relative_energy_tolerance: float = eqx.field(static=True)
    absolute_gas_tolerance: float = eqx.field(static=True)
    contact_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        temperature_increment: float = 100.0,
        burnup_increment: float = 0.1,
        release_fraction_increment: float = 0.02,
        creep_increment: float = 1e-4,
        relative_energy_tolerance: float = 1e-6,
        absolute_gas_tolerance: float = 1e-10,
        contact_tolerance: float = 1e-8,
    ) -> None:
        values = (
            temperature_increment,
            burnup_increment,
            release_fraction_increment,
            creep_increment,
            relative_energy_tolerance,
            absolute_gas_tolerance,
            contact_tolerance,
        )
        if any(not 0 < value < float("inf") for value in values):
            raise ValueError("Acceptance limits must be finite positive scalars.")
        (
            self.temperature_increment,
            self.burnup_increment,
            self.release_fraction_increment,
            self.creep_increment,
            self.relative_energy_tolerance,
            self.absolute_gas_tolerance,
            self.contact_tolerance,
        ) = values
        self.policy_id = canonical_fingerprint(
            {"kind": "fuel-physical-step-acceptance", "limits": values}
        )


def _coolant_source_id(
    coolant: PrescribedWallTemperature | SingleChannelCoolant | CoupledChannelCoolant,
    /,
) -> str:
    if isinstance(coolant, PrescribedWallTemperature):
        return "prescribed-wall-table:" + coolant.table_id
    transfer = (
        coolant.wall_transfer if isinstance(coolant, CoupledChannelCoolant) else coolant
    )
    return transfer.nucleate_boiling_validity.source_id


def source_ids(
    bundle: FuelModelBundle,
    coolant: PrescribedWallTemperature | SingleChannelCoolant | CoupledChannelCoolant,
    /,
) -> tuple[str, ...]:
    """Source identifier of every ``SOURCE_SLOTS`` entry, in slot order."""
    identifiers = {
        "fuel_heat": bundle.fuel_heat.source_id,
        "fuel_expansion": bundle.fuel_expansion.source_id,
        "fuel_conductivity": bundle.fuel_conductivity.source_id,
        "fuel_elasticity": bundle.fuel_elasticity.source_id,
        "fuel_emissivity": bundle.fuel_emissivity.source_id,
        "densification": bundle.densification.source_id,
        "solid_swelling": bundle.solid_swelling.source_id,
        "relocation": bundle.relocation.source_id,
        "fuel_creep": bundle.fuel_creep.source_id,
        "radial_power": bundle.radial_power.source_id,
        "fission_gas": bundle.fission_gas.source_id,
        "helium": bundle.helium.source_id,
        "relocation_history": bundle.relocation_history.source_id,
        "cladding_heat": bundle.cladding_heat.source_id,
        "cladding_conductivity": bundle.cladding_conductivity.source_id,
        "cladding_expansion": bundle.cladding_expansion.source_id,
        "cladding_elasticity": bundle.cladding_elasticity.source_id,
        "cladding_growth": bundle.cladding_growth.source_id,
        "cladding_creep": bundle.cladding_creep.source_id,
        "cladding_plasticity": bundle.cladding_plasticity.source_id,
        "gap": bundle.gap.source_id,
        "corrosion": bundle.corrosion.source_id,
        "hydrogen": bundle.hydrogen.source_id,
        "oxide_conductivity": bundle.oxide_conductivity.source_id,
        "bore_emissivity": bundle.bore_emissivity.source_id,
        # One slot covers both plenum-temperature closures.
        "plenum_temperature": bundle.plenum_temperature.source_id
        + ";"
        + bundle.plenum_temperature.prescribed_source_id,
        "coolant": _coolant_source_id(coolant),
    }
    return tuple(identifiers[slot] for slot in SOURCE_SLOTS)


class FuelSourceSupport(StrictModule):
    """Whether each source law is inside its validity range, one entry per slot.

    ``supported`` has ``slots`` as its last axis, in ``SOURCE_SLOTS`` order;
    ``source_ids`` names the source of every slot.
    """

    supported: Array
    slots: tuple[FuelSourceSlot, ...] = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)

    def of(self, slot: FuelSourceSlot, /) -> Array:
        """Support of one slot's source law."""
        return self.supported[..., self.slots.index(parse(slot, FuelSourceSlot, "slot"))]

    def all(self) -> Array:
        """Whether every source law is inside its validity range."""
        return jnp.all(self.supported, axis=-1)


class FuelNonlinearEvidence(StrictModule):
    """Thermomechanical Newton solve of the candidate step."""

    status: Array
    residual: Array
    iterations: Array
    counts_complete: Array


class FuelLinearEvidence(StrictModule):
    """Final Newton linear solve; unavailable rank is -1, condition NaN."""

    status: Array
    rank: Array
    condition_estimate: Array
    residual: Array
    converged: Array
    rank_available: Array
    condition_available: Array


class FuelCouplingEvidence(StrictModule):
    """Gas/thermomechanical coupling iteration and its accumulated work.

    ``joint_residual`` is the norm of the live coupled residual at the endpoint
    in scaled solver coordinates: thermomechanics with the gas re-evaluated at
    the endpoint plus the hydrostatic-stress closure, i.e. the system the
    implicit derivative linearizes. ``gas_admissible`` is physical
    admissibility of that endpoint gas.
    """

    status: Array
    iterations: Array
    evaluations: Array
    joint_residual: Array
    gas_admissible: Array
    thermomechanical_residual_evaluations: Array
    thermomechanical_iterations: Array
    gas_block_evaluations: Array


class FuelSolverEvidence(StrictModule):
    """Thermomechanical, final linear, and gas-coupling solve evidence of a step."""

    nonlinear: FuelNonlinearEvidence
    linear: FuelLinearEvidence
    coupling: FuelCouplingEvidence
    local_roots_converged: Array


class FuelPhysicalEvidence(StrictModule):
    """Physical admissibility, contact, and increments of the candidate step.

    ``minimum_gap`` is in m, ``complementarity_error`` is |gap * pressure| per
    the contact-pressure scale [m], and ``increments`` are those of
    ``physical_increments``.
    """

    physical: Array
    in_range: Array
    minimum_gap: Array
    complementarity_error: Array
    friction_error: Array
    increments: Array
    accommodation_extrapolated: Array
    accommodation_source_comparison: Array


class FuelBalanceEvidence(StrictModule):
    """Energy [J] and gas inventory balance errors of the candidate step."""

    energy_error: Array
    gas_error: Array


class FuelStepEvidence(StrictModule):
    """Evidence of one attempted step, whether accepted or refused.

    ``status`` is a ``FuelStepStatus`` code. ``terminal_domain_exit`` marks a
    converged, physical candidate outside its admitted range (every source
    range under ``strict-source-support``); the run stops instead of retrying.
    """

    step_index: Array
    time: Array
    time_step: Array
    accepted: Array
    terminal_domain_exit: Array
    status: Array
    solver: FuelSolverEvidence
    physics: FuelPhysicalEvidence
    balances: FuelBalanceEvidence
    source_support: FuelSourceSupport
    prepared_id: str = eqx.field(static=True)
    bundle_id: str = eqx.field(static=True)
    history_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    coupling_evaluation_budget: int = eqx.field(static=True)
    coupling_tolerance: float = eqx.field(static=True)


def _fission_gas_release_fraction(gas: FuelGasLedger, /) -> Array:
    """Released over produced krypton and xenon; zero when nothing was produced.

    Only exactly zero production is replaced; NaN or negative production
    propagates into the fraction.
    """
    produced = jnp.sum(gas.produced_moles[_FISSION_GAS_SPECIES])
    return jnp.sum(gas.released_moles[_FISSION_GAS_SPECIES]) / jnp.where(
        produced == 0, 1.0, produced
    )


def physical_increments(previous: FuelRodState, candidate: FuelRodState, /) -> Array:
    """Increments of temperature [K], burnup [MWd/kgHM], gas release, and creep."""
    temperature = jnp.maximum(
        jnp.max(
            jnp.abs(candidate.fields.temperature_fuel - previous.fields.temperature_fuel)
        ),
        jnp.max(
            jnp.abs(
                candidate.fields.temperature_cladding
                - previous.fields.temperature_cladding
            )
        ),
    )
    burnup = jnp.max(
        candidate.exposure.burnup_mwd_per_kghm - previous.exposure.burnup_mwd_per_kghm
    )
    release_fraction = jnp.abs(
        _fission_gas_release_fraction(candidate.gas)
        - _fission_gas_release_fraction(previous.gas)
    )
    creep = jnp.maximum(
        jnp.max(
            jnp.abs(
                candidate.fuel_sites.mechanics.creep_log_strain
                - previous.fuel_sites.mechanics.creep_log_strain
            )
        ),
        jnp.max(
            jnp.abs(
                candidate.cladding_sites.inelastic_log_strain
                - previous.cladding_sites.inelastic_log_strain
            )
        ),
    )
    return jnp.stack((temperature, burnup, release_fraction, creep))


def _step_status(passed: Mapping[FuelStepStatus, Array], /) -> Array:
    """``ACCEPTED``, or the highest-priority failure status whose check failed."""
    status = jnp.asarray(FuelStepStatus.ACCEPTED, dtype=jnp.int32)
    # Apply the lowest priority first so that every higher-priority failure
    # overwrites it.
    for failure in reversed(_FAILURE_PRIORITY):
        status = jnp.where(passed[failure], status, jnp.asarray(failure, dtype=jnp.int32))
    return status


def fuel_step_evidence(
    prepared: PreparedFuelRod,
    result: FuelRodCandidateResult,
    previous: FuelRodState,
    time: Array,
    dt: Array,
    step_index: Array,
    /,
    *,
    parameters: FuelModelBundle | None = None,
    policy: FuelAcceptancePolicy | None = None,
) -> FuelStepEvidence:
    bundle = prepared.admission.bundle if parameters is None else parameters
    policy = FuelAcceptancePolicy() if policy is None else policy
    evaluation = prepared.candidate_evaluation(result)
    candidate = evaluation.candidate
    thermomechanical = result.thermomechanical_result
    coupling = result.coupling_result
    verified_gas = result.verified_gas
    coupling_budget = prepared.coupling_termination.maximum_evaluations
    if coupling_budget is None:
        raise RuntimeError("Fuel coupling requires an explicit evaluation budget.")
    joint_norm = jnp.linalg.norm(result.joint_residual)
    joint_converged = jnp.all(jnp.isfinite(result.joint_residual)) & (
        joint_norm <= prepared.coupling_tolerance
    )
    verified_gas_roots_converged = (
        jnp.all(verified_gas.gas.diagnostics.vacancy_root_converged)
        & jnp.all(verified_gas.gas.diagnostics.hbs_root_converged)
        & jnp.all(verified_gas.helium.linear_solve_successful)
        & jnp.all(verified_gas.species.linear_solve_successful)
    )
    local_converged = evaluation.local_converged & verified_gas_roots_converged
    physical = evaluation.physical & result.coupling_gas_admissible
    in_range = (
        evaluation.in_range
        & jnp.all(verified_gas.gas.in_range)
        & jnp.all(verified_gas.helium.in_range)
        & jnp.all(verified_gas.species.in_range)
    )
    gap = evaluation.contact_gap
    contact_pressure = candidate.fields.contact_pressure[
        prepared.geometry.fuel_interface.interface_index
    ]
    # Contact complementarity |gap * pressure| [m Pa] is reported per the
    # solver's contact-pressure scale, i.e. as a gap [m].
    complementarity = jnp.max(jnp.abs(gap * contact_pressure)) / CONTACT_PRESSURE_SCALE_PA
    friction_error = jnp.max(jnp.abs(evaluation.friction_residual))
    contact_admissible = (
        jnp.all(gap >= -policy.contact_tolerance)
        & jnp.all(contact_pressure >= -CONTACT_PRESSURE_ROUNDOFF_PA)
        & (complementarity <= policy.contact_tolerance)
        & (friction_error <= FRICTION_RESIDUAL_TOLERANCE)
    )
    deposited = (
        candidate.energy.deposited_fuel
        + candidate.energy.deposited_cladding
        + candidate.energy.deposited_direct_coolant
    )
    energy_balanced = jnp.abs(
        evaluation.energy_error
    ) <= policy.relative_energy_tolerance * jnp.maximum(
        jnp.abs(deposited), ENERGY_TOLERANCE_FLOOR_J
    )
    gas_balanced = evaluation.gas_error <= policy.absolute_gas_tolerance
    increments = physical_increments(previous, candidate)
    increment_limits = jnp.asarray(
        (
            policy.temperature_increment,
            policy.burnup_increment,
            policy.release_fraction_increment,
            policy.creep_increment,
        ),
        dtype=increments.dtype,
    )
    status = _step_status(
        {
            FuelStepStatus.NONLINEAR_FAILURE: thermomechanical.successful,
            FuelStepStatus.COUPLING_FAILURE: coupling.successful,
            FuelStepStatus.JOINT_RESIDUAL_FAILURE: joint_converged,
            FuelStepStatus.LOCAL_ROOT_FAILURE: local_converged,
            FuelStepStatus.PHYSICAL_FAILURE: physical,
            FuelStepStatus.TERMINAL_DOMAIN_EXIT: in_range,
            FuelStepStatus.ENERGY_IMBALANCE: energy_balanced,
            FuelStepStatus.GAS_IMBALANCE: gas_balanced,
            FuelStepStatus.CONTACT_FAILURE: contact_admissible,
            FuelStepStatus.INCREMENT_LIMIT: jnp.all(increments <= increment_limits),
        }
    )
    terminal = (
        thermomechanical.successful
        & coupling.successful
        & local_converged
        & physical
        & ~in_range
    )
    diagnostics = thermomechanical.diagnostics
    return FuelStepEvidence(
        step_index=jnp.asarray(step_index, dtype=jnp.int64),
        time=time,
        time_step=dt,
        accepted=status == FuelStepStatus.ACCEPTED,
        terminal_domain_exit=terminal,
        status=status,
        solver=FuelSolverEvidence(
            nonlinear=FuelNonlinearEvidence(
                status=thermomechanical.status,
                residual=diagnostics.final_residual_norm,
                iterations=diagnostics.iterations,
                counts_complete=jnp.asarray(diagnostics.counts_complete, dtype=jnp.bool_),
            ),
            # Unavailable linear evidence is rank -1 and condition NaN; an
            # infinite measured condition estimate is available evidence of a
            # singular system.
            linear=FuelLinearEvidence(
                status=diagnostics.final_linear_status,
                rank=diagnostics.final_linear_rank,
                condition_estimate=diagnostics.final_linear_condition_estimate,
                residual=diagnostics.final_linear_residual_norm,
                converged=diagnostics.final_linear_converged,
                rank_available=diagnostics.final_linear_rank >= 0,
                condition_available=~jnp.isnan(
                    diagnostics.final_linear_condition_estimate
                ),
            ),
            coupling=FuelCouplingEvidence(
                status=coupling.status,
                iterations=coupling.diagnostics.iterations,
                evaluations=coupling.diagnostics.residual_evaluations,
                joint_residual=joint_norm,
                gas_admissible=result.coupling_gas_admissible,
                thermomechanical_residual_evaluations=result.thermomechanical_residual_evaluations,
                thermomechanical_iterations=result.thermomechanical_iterations,
                gas_block_evaluations=result.gas_block_evaluations,
            ),
            local_roots_converged=local_converged,
        ),
        physics=FuelPhysicalEvidence(
            physical=physical,
            in_range=in_range,
            minimum_gap=jnp.min(gap),
            complementarity_error=complementarity,
            friction_error=friction_error,
            increments=increments,
            accommodation_extrapolated=jnp.any(evaluation.accommodation_extrapolated),
            accommodation_source_comparison=jnp.all(
                evaluation.accommodation_source_comparison
            ),
        ),
        balances=FuelBalanceEvidence(
            energy_error=evaluation.energy_error, gas_error=evaluation.gas_error
        ),
        source_support=FuelSourceSupport(
            supported=evaluation.source_support,
            slots=SOURCE_SLOTS,
            source_ids=source_ids(bundle, prepared.admission.coolant),
        ),
        prepared_id=prepared.prepared_id,
        bundle_id=bundle.bundle_id,
        history_id=prepared.admission.history.history_id,
        policy_id=policy.policy_id,
        coupling_evaluation_budget=coupling_budget,
        coupling_tolerance=prepared.coupling_tolerance,
    )
