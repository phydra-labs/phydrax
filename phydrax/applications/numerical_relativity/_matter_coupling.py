#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection


RelativisticMatterKind: TypeAlias = Literal["grhd", "grmhd", "grrmhd"]


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value)
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


def _scalar(value: ArrayLike, role: str, /, *, dtype: Any | None = None) -> Array:
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != ():
        raise ValueError(f"{role} must be scalar.")
    return result


def _vector3(value: ArrayLike, role: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape != (3,):
        raise ValueError(f"{role} must have shape (3,).")
    return result


def _static_bound(value: float, role: str, /) -> float:
    bound = float(value)
    if not isfinite(bound) or bound < 0.0:
        raise ValueError(f"{role} must be finite and non-negative.")
    return bound


class MatterCouplingPolicy(StrictModule, NonTrainableState):
    """Fail-closed limits for one fixed-topology Z4c/matter evolution."""

    source_consistency_tolerance: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    maximum_floor_rest_mass: float = eqx.field(static=True)
    maximum_floor_energy: float = eqx.field(static=True)
    maximum_floor_momentum: float = eqx.field(static=True)
    maximum_consecutive_failures: int = eqx.field(static=True)
    require_derivative_valid: bool = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        source_consistency_tolerance: float = 1.0e-10,
        constraint_tolerance: float = 1.0e-8,
        conservation_tolerance: float = 1.0e-10,
        maximum_floor_rest_mass: float = 0.0,
        maximum_floor_energy: float = 0.0,
        maximum_floor_momentum: float = 0.0,
        maximum_consecutive_failures: int = 1,
        require_derivative_valid: bool = True,
    ):
        source = _static_bound(
            source_consistency_tolerance, "source_consistency_tolerance"
        )
        constraint = _static_bound(constraint_tolerance, "constraint_tolerance")
        conservation = _static_bound(conservation_tolerance, "conservation_tolerance")
        floor_mass = _static_bound(maximum_floor_rest_mass, "maximum_floor_rest_mass")
        floor_energy = _static_bound(maximum_floor_energy, "maximum_floor_energy")
        floor_momentum = _static_bound(maximum_floor_momentum, "maximum_floor_momentum")
        failures = int(maximum_consecutive_failures)
        if failures < 1:
            raise ValueError("maximum_consecutive_failures must be positive.")
        derivative = bool(require_derivative_valid)
        self.source_consistency_tolerance = source
        self.constraint_tolerance = constraint
        self.conservation_tolerance = conservation
        self.maximum_floor_rest_mass = floor_mass
        self.maximum_floor_energy = floor_energy
        self.maximum_floor_momentum = floor_momentum
        self.maximum_consecutive_failures = failures
        self.require_derivative_valid = derivative
        self.policy_id = canonical_fingerprint(
            {
                "kind": "z4c-relativistic-matter-coupling-policy",
                "source_consistency_tolerance": source,
                "constraint_tolerance": constraint,
                "conservation_tolerance": conservation,
                "maximum_floor_rest_mass": floor_mass,
                "maximum_floor_energy": floor_energy,
                "maximum_floor_momentum": floor_momentum,
                "maximum_consecutive_failures": failures,
                "require_derivative_valid": derivative,
            }
        )


class CoupledStageAddress(StrictModule, NonTrainableState):
    """One SSPRK stage address on an immutable spatial topology."""

    step_start_time: Array
    stage_time: Array
    step_end_time: Array
    step_id: Array
    stage_id: Array
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        step_start_time: ArrayLike,
        stage_time: ArrayLike,
        step_end_time: ArrayLike,
        step_id: ArrayLike,
        stage_id: ArrayLike,
        /,
        *,
        topology_id: str,
    ):
        start = _scalar(step_start_time, "step_start_time")
        self.step_start_time = start
        self.stage_time = _scalar(stage_time, "stage_time", dtype=start.dtype)
        self.step_end_time = _scalar(step_end_time, "step_end_time", dtype=start.dtype)
        self.step_id = _scalar(step_id, "step_id", dtype=jnp.int32)
        self.stage_id = _scalar(stage_id, "stage_id", dtype=jnp.int32)
        self.topology_id = _identifier(topology_id, "topology_id")

    @property
    def step_size(self) -> Array:
        return self.step_end_time - self.step_start_time

    def matches(self, other: CoupledStageAddress, /) -> Array:
        if not isinstance(other, CoupledStageAddress):
            raise TypeError("Stage identity comparison requires CoupledStageAddress.")
        topology = jnp.asarray(self.topology_id == other.topology_id)
        return (
            topology
            & (self.step_id == other.step_id)
            & (self.stage_id == other.stage_id)
            & (self.step_start_time == other.step_start_time)
            & (self.stage_time == other.stage_time)
            & (self.step_end_time == other.step_end_time)
        )


class SourceExchangeLedger(StrictModule):
    """Integrated matter source seen by geometry and its exchange mismatch."""

    energy: Array
    momentum: Array
    energy_defect: Array
    momentum_defect: Array

    def __init__(
        self,
        energy: ArrayLike,
        momentum: ArrayLike,
        energy_defect: ArrayLike = 0.0,
        momentum_defect: ArrayLike = (0.0, 0.0, 0.0),
        /,
    ):
        energy_ = _scalar(energy, "source energy")
        self.energy = energy_
        self.momentum = _vector3(momentum, "source momentum").astype(energy_.dtype)
        self.energy_defect = _scalar(
            energy_defect, "source energy_defect", dtype=energy_.dtype
        )
        self.momentum_defect = _vector3(momentum_defect, "source momentum_defect").astype(
            energy_.dtype
        )

    @property
    def finite(self) -> Array:
        return (
            jnp.isfinite(self.energy)
            & jnp.all(jnp.isfinite(self.momentum))
            & jnp.isfinite(self.energy_defect)
            & jnp.all(jnp.isfinite(self.momentum_defect))
        )

    @property
    def maximum_defect(self) -> Array:
        return jnp.maximum(
            jnp.abs(self.energy_defect), jnp.max(jnp.abs(self.momentum_defect))
        )


class ConstraintLedger(StrictModule):
    """Z4c Hamiltonian, momentum, and damping-constraint infinity norms."""

    hamiltonian_linf: Array
    momentum_linf: Array
    z4_linf: Array

    def __init__(
        self,
        hamiltonian_linf: ArrayLike,
        momentum_linf: ArrayLike,
        z4_linf: ArrayLike,
        /,
    ):
        hamiltonian = _scalar(hamiltonian_linf, "hamiltonian_linf")
        self.hamiltonian_linf = hamiltonian
        self.momentum_linf = _vector3(momentum_linf, "momentum_linf").astype(
            hamiltonian.dtype
        )
        self.z4_linf = _scalar(z4_linf, "z4_linf", dtype=hamiltonian.dtype)

    @property
    def finite(self) -> Array:
        return (
            jnp.isfinite(self.hamiltonian_linf)
            & jnp.all(jnp.isfinite(self.momentum_linf))
            & jnp.isfinite(self.z4_linf)
        )

    @property
    def maximum(self) -> Array:
        return jnp.maximum(
            jnp.maximum(jnp.abs(self.hamiltonian_linf), jnp.abs(self.z4_linf)),
            jnp.max(jnp.abs(self.momentum_linf)),
        )


class ConservationLedger(StrictModule):
    """Finite-volume matter balance defects for one candidate stage."""

    rest_mass_defect: Array
    energy_defect: Array
    momentum_defect: Array
    magnetic_divergence_linf: Array

    def __init__(
        self,
        rest_mass_defect: ArrayLike,
        energy_defect: ArrayLike,
        momentum_defect: ArrayLike,
        magnetic_divergence_linf: ArrayLike = 0.0,
        /,
    ):
        mass = _scalar(rest_mass_defect, "rest_mass_defect")
        self.rest_mass_defect = mass
        self.energy_defect = _scalar(energy_defect, "energy_defect", dtype=mass.dtype)
        self.momentum_defect = _vector3(momentum_defect, "momentum_defect").astype(
            mass.dtype
        )
        self.magnetic_divergence_linf = _scalar(
            magnetic_divergence_linf,
            "magnetic_divergence_linf",
            dtype=mass.dtype,
        )

    @property
    def finite(self) -> Array:
        return (
            jnp.isfinite(self.rest_mass_defect)
            & jnp.isfinite(self.energy_defect)
            & jnp.all(jnp.isfinite(self.momentum_defect))
            & jnp.isfinite(self.magnetic_divergence_linf)
        )

    @property
    def maximum(self) -> Array:
        return jnp.max(
            jnp.concatenate(
                (
                    jnp.abs(self.rest_mass_defect)[None],
                    jnp.abs(self.energy_defect)[None],
                    jnp.abs(self.momentum_defect),
                    jnp.abs(self.magnetic_divergence_linf)[None],
                )
            )
        )


class FloorLedger(StrictModule):
    """Explicit atmosphere/floor intervention; never hidden conservation."""

    rest_mass_added: Array
    energy_added: Array
    momentum_added: Array
    cell_count: Array

    def __init__(
        self,
        rest_mass_added: ArrayLike,
        energy_added: ArrayLike,
        momentum_added: ArrayLike,
        cell_count: ArrayLike,
        /,
    ):
        mass = _scalar(rest_mass_added, "floor rest_mass_added")
        self.rest_mass_added = mass
        self.energy_added = _scalar(energy_added, "floor energy_added", dtype=mass.dtype)
        self.momentum_added = _vector3(momentum_added, "floor momentum_added").astype(
            mass.dtype
        )
        self.cell_count = _scalar(cell_count, "floor cell_count", dtype=jnp.int32)

    @property
    def finite(self) -> Array:
        return (
            jnp.isfinite(self.rest_mass_added)
            & jnp.isfinite(self.energy_added)
            & jnp.all(jnp.isfinite(self.momentum_added))
        )

    @property
    def physically_valid(self) -> Array:
        return (
            (self.rest_mass_added >= 0.0)
            & (self.energy_added >= 0.0)
            & (self.cell_count >= 0)
        )


class HorizonFluxLedger(StrictModule):
    """Signed matter flux through a declared horizon world tube."""

    rest_mass: Array
    energy: Array
    momentum: Array
    angular_momentum: Array
    magnetic_flux: Array

    def __init__(
        self,
        rest_mass: ArrayLike,
        energy: ArrayLike,
        momentum: ArrayLike,
        angular_momentum: ArrayLike,
        magnetic_flux: ArrayLike = 0.0,
        /,
    ):
        mass = _scalar(rest_mass, "horizon rest_mass")
        self.rest_mass = mass
        self.energy = _scalar(energy, "horizon energy", dtype=mass.dtype)
        self.momentum = _vector3(momentum, "horizon momentum").astype(mass.dtype)
        self.angular_momentum = _vector3(
            angular_momentum, "horizon angular_momentum"
        ).astype(mass.dtype)
        self.magnetic_flux = _scalar(
            magnetic_flux, "horizon magnetic_flux", dtype=mass.dtype
        )

    @property
    def finite(self) -> Array:
        return (
            jnp.isfinite(self.rest_mass)
            & jnp.isfinite(self.energy)
            & jnp.all(jnp.isfinite(self.momentum))
            & jnp.all(jnp.isfinite(self.angular_momentum))
            & jnp.isfinite(self.magnetic_flux)
        )


class CoupledStepLedgers(StrictModule):
    """All auditable exchange and correction ledgers for one proposal."""

    source: SourceExchangeLedger
    constraint: ConstraintLedger
    conservation: ConservationLedger
    floor: FloorLedger
    horizon_flux: HorizonFluxLedger

    def __init__(
        self,
        source: SourceExchangeLedger,
        constraint: ConstraintLedger,
        conservation: ConservationLedger,
        floor: FloorLedger,
        horizon_flux: HorizonFluxLedger,
        /,
    ):
        if not isinstance(source, SourceExchangeLedger):
            raise TypeError("source must be SourceExchangeLedger.")
        if not isinstance(constraint, ConstraintLedger):
            raise TypeError("constraint must be ConstraintLedger.")
        if not isinstance(conservation, ConservationLedger):
            raise TypeError("conservation must be ConservationLedger.")
        if not isinstance(floor, FloorLedger):
            raise TypeError("floor must be FloorLedger.")
        if not isinstance(horizon_flux, HorizonFluxLedger):
            raise TypeError("horizon_flux must be HorizonFluxLedger.")
        self.source = source
        self.constraint = constraint
        self.conservation = conservation
        self.floor = floor
        self.horizon_flux = horizon_flux

    @classmethod
    def combine(cls, ledgers: tuple[CoupledStepLedgers, ...], /) -> CoupledStepLedgers:
        """Combine fixed stage contributions into one macro-step ledger."""
        values = tuple(ledgers)
        if not values or any(not isinstance(value, cls) for value in values):
            raise TypeError("ledgers must be a non-empty tuple of CoupledStepLedgers.")
        result = values[0]
        for value in values[1:]:
            left_source, right_source = result.source, value.source
            left_constraint, right_constraint = result.constraint, value.constraint
            left_conservation, right_conservation = (
                result.conservation,
                value.conservation,
            )
            left_floor, right_floor = result.floor, value.floor
            left_horizon, right_horizon = result.horizon_flux, value.horizon_flux
            result = cls(
                SourceExchangeLedger(
                    left_source.energy + right_source.energy,
                    left_source.momentum + right_source.momentum,
                    jnp.maximum(
                        jnp.abs(left_source.energy_defect),
                        jnp.abs(right_source.energy_defect),
                    ),
                    jnp.maximum(
                        jnp.abs(left_source.momentum_defect),
                        jnp.abs(right_source.momentum_defect),
                    ),
                ),
                ConstraintLedger(
                    jnp.maximum(
                        jnp.abs(left_constraint.hamiltonian_linf),
                        jnp.abs(right_constraint.hamiltonian_linf),
                    ),
                    jnp.maximum(
                        jnp.abs(left_constraint.momentum_linf),
                        jnp.abs(right_constraint.momentum_linf),
                    ),
                    jnp.maximum(
                        jnp.abs(left_constraint.z4_linf),
                        jnp.abs(right_constraint.z4_linf),
                    ),
                ),
                ConservationLedger(
                    left_conservation.rest_mass_defect
                    + right_conservation.rest_mass_defect,
                    left_conservation.energy_defect + right_conservation.energy_defect,
                    left_conservation.momentum_defect
                    + right_conservation.momentum_defect,
                    jnp.maximum(
                        jnp.abs(left_conservation.magnetic_divergence_linf),
                        jnp.abs(right_conservation.magnetic_divergence_linf),
                    ),
                ),
                FloorLedger(
                    left_floor.rest_mass_added + right_floor.rest_mass_added,
                    left_floor.energy_added + right_floor.energy_added,
                    left_floor.momentum_added + right_floor.momentum_added,
                    left_floor.cell_count + right_floor.cell_count,
                ),
                HorizonFluxLedger(
                    left_horizon.rest_mass + right_horizon.rest_mass,
                    left_horizon.energy + right_horizon.energy,
                    left_horizon.momentum + right_horizon.momentum,
                    left_horizon.angular_momentum + right_horizon.angular_momentum,
                    left_horizon.magnetic_flux + right_horizon.magnetic_flux,
                ),
            )
        return result

    @property
    def finite(self) -> Array:
        return (
            self.source.finite
            & self.constraint.finite
            & self.conservation.finite
            & self.floor.finite
            & self.horizon_flux.finite
        )

    def within_step_limits(self, policy: MatterCouplingPolicy, /) -> Array:
        if not isinstance(policy, MatterCouplingPolicy):
            raise TypeError("policy must be MatterCouplingPolicy.")
        return (
            self.finite
            & self.floor.physically_valid
            & (self.source.maximum_defect <= policy.source_consistency_tolerance)
            & (self.constraint.maximum <= policy.constraint_tolerance)
            & (self.conservation.maximum <= policy.conservation_tolerance)
        )


class CoupledBudget(StrictModule, NonTrainableState):
    """Accepted cumulative budget; rejected proposal ledgers never enter it."""

    source_energy: Array
    source_momentum: Array
    maximum_source_defect: Array
    maximum_constraint: Array
    conservation_rest_mass: Array
    conservation_energy: Array
    conservation_momentum: Array
    maximum_conservation_defect: Array
    maximum_magnetic_divergence: Array
    floor_rest_mass: Array
    floor_energy: Array
    floor_momentum: Array
    floor_cell_count: Array
    horizon_rest_mass: Array
    horizon_energy: Array
    horizon_momentum: Array
    horizon_angular_momentum: Array
    horizon_magnetic_flux: Array

    def __init__(
        self,
        source_energy: ArrayLike,
        source_momentum: ArrayLike,
        maximum_source_defect: ArrayLike,
        maximum_constraint: ArrayLike,
        conservation_rest_mass: ArrayLike,
        conservation_energy: ArrayLike,
        conservation_momentum: ArrayLike,
        maximum_conservation_defect: ArrayLike,
        maximum_magnetic_divergence: ArrayLike,
        floor_rest_mass: ArrayLike,
        floor_energy: ArrayLike,
        floor_momentum: ArrayLike,
        floor_cell_count: ArrayLike,
        horizon_rest_mass: ArrayLike,
        horizon_energy: ArrayLike,
        horizon_momentum: ArrayLike,
        horizon_angular_momentum: ArrayLike,
        horizon_magnetic_flux: ArrayLike,
        /,
    ):
        source = _scalar(source_energy, "budget source_energy")
        dtype = source.dtype
        self.source_energy = source
        self.source_momentum = _vector3(source_momentum, "budget source_momentum").astype(
            dtype
        )
        self.maximum_source_defect = _scalar(
            maximum_source_defect, "budget maximum_source_defect", dtype=dtype
        )
        self.maximum_constraint = _scalar(
            maximum_constraint, "budget maximum_constraint", dtype=dtype
        )
        self.conservation_rest_mass = _scalar(
            conservation_rest_mass, "budget conservation_rest_mass", dtype=dtype
        )
        self.conservation_energy = _scalar(
            conservation_energy, "budget conservation_energy", dtype=dtype
        )
        self.conservation_momentum = _vector3(
            conservation_momentum, "budget conservation_momentum"
        ).astype(dtype)
        self.maximum_conservation_defect = _scalar(
            maximum_conservation_defect,
            "budget maximum_conservation_defect",
            dtype=dtype,
        )
        self.maximum_magnetic_divergence = _scalar(
            maximum_magnetic_divergence,
            "budget maximum_magnetic_divergence",
            dtype=dtype,
        )
        self.floor_rest_mass = _scalar(
            floor_rest_mass, "budget floor_rest_mass", dtype=dtype
        )
        self.floor_energy = _scalar(floor_energy, "budget floor_energy", dtype=dtype)
        self.floor_momentum = _vector3(floor_momentum, "budget floor_momentum").astype(
            dtype
        )
        self.floor_cell_count = _scalar(
            floor_cell_count, "budget floor_cell_count", dtype=jnp.int32
        )
        self.horizon_rest_mass = _scalar(
            horizon_rest_mass, "budget horizon_rest_mass", dtype=dtype
        )
        self.horizon_energy = _scalar(
            horizon_energy, "budget horizon_energy", dtype=dtype
        )
        self.horizon_momentum = _vector3(
            horizon_momentum, "budget horizon_momentum"
        ).astype(dtype)
        self.horizon_angular_momentum = _vector3(
            horizon_angular_momentum, "budget horizon_angular_momentum"
        ).astype(dtype)
        self.horizon_magnetic_flux = _scalar(
            horizon_magnetic_flux, "budget horizon_magnetic_flux", dtype=dtype
        )

    @classmethod
    def zeros(cls, *, dtype: Any = jnp.float64) -> CoupledBudget:
        zero = jnp.asarray(0.0, dtype=dtype)
        vector = jnp.zeros((3,), dtype=zero.dtype)
        count = jnp.asarray(0, dtype=jnp.int32)
        return cls(
            zero,
            vector,
            zero,
            zero,
            zero,
            zero,
            vector,
            zero,
            zero,
            zero,
            zero,
            vector,
            count,
            zero,
            zero,
            vector,
            vector,
            zero,
        )

    def accumulate(self, ledgers: CoupledStepLedgers, /) -> CoupledBudget:
        if not isinstance(ledgers, CoupledStepLedgers):
            raise TypeError("ledgers must be CoupledStepLedgers.")
        source = ledgers.source
        constraint = ledgers.constraint
        conservation = ledgers.conservation
        floor = ledgers.floor
        horizon = ledgers.horizon_flux
        return CoupledBudget(
            self.source_energy + source.energy,
            self.source_momentum + source.momentum,
            jnp.maximum(self.maximum_source_defect, source.maximum_defect),
            jnp.maximum(self.maximum_constraint, constraint.maximum),
            self.conservation_rest_mass + conservation.rest_mass_defect,
            self.conservation_energy + conservation.energy_defect,
            self.conservation_momentum + conservation.momentum_defect,
            jnp.maximum(self.maximum_conservation_defect, conservation.maximum),
            jnp.maximum(
                self.maximum_magnetic_divergence,
                jnp.abs(conservation.magnetic_divergence_linf),
            ),
            self.floor_rest_mass + floor.rest_mass_added,
            self.floor_energy + floor.energy_added,
            self.floor_momentum + floor.momentum_added,
            self.floor_cell_count + floor.cell_count,
            self.horizon_rest_mass + horizon.rest_mass,
            self.horizon_energy + horizon.energy,
            self.horizon_momentum + horizon.momentum,
            self.horizon_angular_momentum + horizon.angular_momentum,
            self.horizon_magnetic_flux + horizon.magnetic_flux,
        )

    def within_limits(self, policy: MatterCouplingPolicy, /) -> Array:
        if not isinstance(policy, MatterCouplingPolicy):
            raise TypeError("policy must be MatterCouplingPolicy.")
        conservation_maximum = jnp.max(
            jnp.concatenate(
                (
                    jnp.abs(self.conservation_rest_mass)[None],
                    jnp.abs(self.conservation_energy)[None],
                    jnp.abs(self.conservation_momentum),
                    jnp.abs(self.maximum_conservation_defect)[None],
                    jnp.abs(self.maximum_magnetic_divergence)[None],
                )
            )
        )
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (
                        self.source_energy,
                        self.source_momentum,
                        self.maximum_source_defect,
                        self.maximum_constraint,
                        self.conservation_rest_mass,
                        self.conservation_energy,
                        self.conservation_momentum,
                        self.maximum_conservation_defect,
                        self.maximum_magnetic_divergence,
                        self.floor_rest_mass,
                        self.floor_energy,
                        self.floor_momentum,
                        self.floor_cell_count,
                        self.horizon_rest_mass,
                        self.horizon_energy,
                        self.horizon_momentum,
                        self.horizon_angular_momentum,
                        self.horizon_magnetic_flux,
                    )
                )
            )
        )
        return (
            finite
            & (self.maximum_constraint <= policy.constraint_tolerance)
            & (self.maximum_source_defect <= policy.source_consistency_tolerance)
            & (conservation_maximum <= policy.conservation_tolerance)
            & (self.floor_rest_mass <= policy.maximum_floor_rest_mass)
            & (self.floor_energy <= policy.maximum_floor_energy)
            & (jnp.max(jnp.abs(self.floor_momentum)) <= policy.maximum_floor_momentum)
        )


class CoupledParticipantStatus(StrictModule):
    """Independent numerical and scientific evidence for one stage proposal."""

    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array

    def __init__(
        self,
        status: ArrayLike,
        finite: ArrayLike,
        converged: ArrayLike,
        physically_valid: ArrayLike,
        qualified: ArrayLike,
        derivative_valid: ArrayLike,
        /,
    ):
        self.status = _scalar(status, "participant status", dtype=jnp.int32)
        self.finite = _scalar(finite, "participant finite", dtype=jnp.bool_)
        self.converged = _scalar(converged, "participant converged", dtype=jnp.bool_)
        self.physically_valid = _scalar(
            physically_valid, "participant physically_valid", dtype=jnp.bool_
        )
        self.qualified = _scalar(qualified, "participant qualified", dtype=jnp.bool_)
        self.derivative_valid = _scalar(
            derivative_valid, "participant derivative_valid", dtype=jnp.bool_
        )

    @property
    def forward_successful(self) -> Array:
        return (
            (self.status == 0)
            & self.finite
            & self.converged
            & self.physically_valid
            & self.qualified
        )


class Z4cStageProposal(StrictModule):
    """One Z4c candidate formed from the matching matter projection."""

    candidate: Any
    geometry: ADMGridGeometry
    address: CoupledStageAddress
    source: SourceExchangeLedger
    constraint: ConstraintLedger
    evidence: CoupledParticipantStatus

    def __init__(
        self,
        candidate: Any,
        geometry: ADMGridGeometry,
        address: CoupledStageAddress,
        source: SourceExchangeLedger,
        constraint: ConstraintLedger,
        evidence: CoupledParticipantStatus,
        /,
    ):
        if not isinstance(geometry, ADMGridGeometry):
            raise TypeError("geometry must be ADMGridGeometry.")
        if not isinstance(address, CoupledStageAddress):
            raise TypeError("address must be CoupledStageAddress.")
        if not isinstance(source, SourceExchangeLedger):
            raise TypeError("source must be SourceExchangeLedger.")
        if not isinstance(constraint, ConstraintLedger):
            raise TypeError("constraint must be ConstraintLedger.")
        if not isinstance(evidence, CoupledParticipantStatus):
            raise TypeError("evidence must be CoupledParticipantStatus.")
        self.candidate = candidate
        self.geometry = geometry
        self.address = address
        self.source = source
        self.constraint = constraint
        self.evidence = evidence


class MatterStageProposal(StrictModule):
    """One GRHD, GRMHD, or GRRMHD candidate on matching ADM geometry."""

    candidate: Any
    stress_energy: StressEnergyProjection
    address: CoupledStageAddress
    conservation: ConservationLedger
    floor: FloorLedger
    horizon_flux: HorizonFluxLedger
    evidence: CoupledParticipantStatus
    matter_kind: RelativisticMatterKind = eqx.field(static=True)

    def __init__(
        self,
        candidate: Any,
        stress_energy: StressEnergyProjection,
        address: CoupledStageAddress,
        conservation: ConservationLedger,
        floor: FloorLedger,
        horizon_flux: HorizonFluxLedger,
        evidence: CoupledParticipantStatus,
        /,
        *,
        matter_kind: RelativisticMatterKind,
    ):
        if not isinstance(stress_energy, StressEnergyProjection):
            raise TypeError("stress_energy must be StressEnergyProjection.")
        if not isinstance(address, CoupledStageAddress):
            raise TypeError("address must be CoupledStageAddress.")
        if not isinstance(conservation, ConservationLedger):
            raise TypeError("conservation must be ConservationLedger.")
        if not isinstance(floor, FloorLedger):
            raise TypeError("floor must be FloorLedger.")
        if not isinstance(horizon_flux, HorizonFluxLedger):
            raise TypeError("horizon_flux must be HorizonFluxLedger.")
        if not isinstance(evidence, CoupledParticipantStatus):
            raise TypeError("evidence must be CoupledParticipantStatus.")
        if matter_kind not in ("grhd", "grmhd", "grrmhd"):
            raise ValueError("matter_kind must be 'grhd', 'grmhd', or 'grrmhd'.")
        self.candidate = candidate
        self.stress_energy = stress_energy
        self.address = address
        self.conservation = conservation
        self.floor = floor
        self.horizon_flux = horizon_flux
        self.evidence = evidence
        self.matter_kind = matter_kind
