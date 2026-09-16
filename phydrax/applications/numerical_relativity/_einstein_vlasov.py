#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stage-exact particle coupling to the canonical fixed-grid Z4c owner."""

from __future__ import annotations

from collections.abc import Callable
from enum import IntFlag
from math import isfinite
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._relativistic_stress_transfer import (
    RelativisticParticleState,
    RelativisticStressDepositPlan,
    RelativisticStressDepositResult,
)
from ...metrix._adm_exchange import ADMGridGeometry, StressEnergyProjection
from ..relativistic_scattering._unit_contract import LocalRelativisticFramePlan
from ._boundaries import AbstractZ4cBoundary, PeriodicBoundary, Z4cBoundaryEvidence
from ._derivatives import FourthOrderDerivatives
from ._enforcement import Z4cAlgebraicEnforcement, Z4cEnforcementEvidence
from ._gauge import AbstractZ4cGauge
from ._grid import FixedGridGeometry
from ._state import Z4cState
from ._z4c import (
    evaluate_z4c_rhs,
    z4c_adm_geometry,
    z4c_snapshot_token,
    Z4cConstraintEvidence,
    Z4cRHSEvaluation,
    Z4cSystem,
)


RelativisticFrameProvider: TypeAlias = Callable[
    [ADMGridGeometry, Array, Array], LocalRelativisticFramePlan
]


class EinsteinVlasovStatus(IntFlag):
    SUCCESS = 0
    NONFINITE = 1
    SOURCE_INVALID = 2
    CONSTRAINT_EXCEEDED = 4
    MASS_SHELL_EXCEEDED = 8
    ENERGY_CONDITION_EXCEEDED = 16
    RESOURCE_EXCEEDED = 32
    BOUNDARY_FAILURE = 64
    ENFORCEMENT_FAILURE = 128
    STRONG_FIELD_REFUSED = 256
    TIME_IDENTITY_MISMATCH = 512
    DERIVATIVE_INVALID = 1024
    INITIAL_DATA_NOT_SOLVED = 2048
    TERMINAL = 4096
    STEP_REJECTED = 8192


class EinsteinVlasovConstraintSolveEvidence(StrictModule):
    """Evidence produced by an actual Hamiltonian/momentum constraint solve."""

    hamiltonian_before: Array
    momentum_before: Array
    hamiltonian_after: Array
    momentum_after: Array
    iterations: Array
    converged: Array
    finite: Array
    solver_id: str = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian_before: ArrayLike,
        momentum_before: ArrayLike,
        hamiltonian_after: ArrayLike,
        momentum_after: ArrayLike,
        iterations: ArrayLike,
        converged: ArrayLike,
        finite: ArrayLike,
        /,
        *,
        solver_id: str,
    ):
        scalar = tuple(
            jnp.asarray(value) for value in (hamiltonian_before, hamiltonian_after)
        )
        vectors = tuple(jnp.asarray(value) for value in (momentum_before, momentum_after))
        iteration = jnp.asarray(iterations, dtype=jnp.int32)
        if any(value.shape != () for value in scalar) or any(
            value.shape != (3,) for value in vectors
        ):
            raise ValueError(
                "Constraint-solve residuals must be one scalar and one 3-vector."
            )
        if iteration.shape != ():
            raise ValueError("Constraint-solve iterations must be scalar.")
        identifier = str(solver_id).strip()
        if not identifier:
            raise ValueError("solver_id must be non-empty.")
        self.hamiltonian_before, self.hamiltonian_after = scalar
        self.momentum_before, self.momentum_after = vectors
        self.iterations = iteration
        self.converged = jnp.asarray(converged, dtype=bool).reshape(())
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.solver_id = identifier


class EinsteinVlasovConstraintSolveResult(StrictModule):
    state: Z4cState
    evidence: EinsteinVlasovConstraintSolveEvidence


EinsteinVlasovConstraintSolver: TypeAlias = Callable[
    [Z4cState, RelativisticParticleState, StressEnergyProjection, ADMGridGeometry],
    EinsteinVlasovConstraintSolveResult,
]


class EinsteinVlasovInitialDataEvidence(StrictModule):
    solve: EinsteinVlasovConstraintSolveEvidence
    constraints: Z4cConstraintEvidence
    source_valid: Array
    mass_shell_valid: Array
    finite: Array
    admitted: Array
    evidence_id: str = eqx.field(static=True)


class EinsteinVlasovInitialDataResult(StrictModule):
    state: "EinsteinVlasovMatterState"
    deposit: RelativisticStressDepositResult
    evidence: EinsteinVlasovInitialDataEvidence


class EinsteinVlasovMatterState(StrictModule):
    """One authoritative accepted geometry/particle synchronization point."""

    z4c: Z4cState
    particles: RelativisticParticleState
    time: Array
    accepted_steps: Array
    rejected_steps: Array
    consecutive_failures: Array
    terminal: Array
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        z4c: Z4cState,
        particles: RelativisticParticleState,
        time: ArrayLike,
        accepted_steps: ArrayLike,
        rejected_steps: ArrayLike,
        consecutive_failures: ArrayLike,
        terminal: ArrayLike,
        /,
        *,
        runtime_id: str,
    ):
        if not isinstance(z4c, Z4cState):
            raise TypeError("z4c must be a Z4cState.")
        if not isinstance(particles, RelativisticParticleState):
            raise TypeError("particles must be a RelativisticParticleState.")
        time_ = jnp.asarray(time)
        counters = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (accepted_steps, rejected_steps, consecutive_failures)
        )
        if time_.shape != () or any(value.shape != () for value in counters):
            raise ValueError("Einstein-Vlasov time and counters must be scalar.")
        identifier = str(runtime_id).strip()
        if not identifier:
            raise ValueError("runtime_id must be non-empty.")
        self.z4c = z4c
        self.particles = particles
        self.time = time_
        self.accepted_steps, self.rejected_steps, self.consecutive_failures = counters
        self.terminal = jnp.asarray(terminal, dtype=bool).reshape(())
        self.runtime_id = identifier


class EinsteinVlasovGeodesicEvaluation(StrictModule):
    position_rate: Array
    covariant_momentum_rate: Array
    particle_energy: Array
    support_complete: Array
    finite: Array
    successful: Array


class EinsteinVlasovStageEvidence(StrictModule):
    snapshot_token: Array
    source_exchange_defect: Array
    hamiltonian_linf: Array
    momentum_linf: Array
    mass_shell_linf: Array
    dominant_energy_violation: Array
    finite: Array
    source_valid: Array
    support_complete: Array
    boundary_valid: Array
    derivative_valid: Array
    successful: Array


class EinsteinVlasovStepEvidence(StrictModule):
    stages: tuple[EinsteinVlasovStageEvidence, ...]
    source_exchange_defect: Array
    adm_constraint_linf: Array
    mass_shell_linf: Array
    dominant_energy_violation: Array
    metric_condition_number: Array
    maximum_extrinsic_curvature: Array
    finite: Array
    resource_valid: Array
    strong_field_supported: Array
    derivative_valid: Array
    qualified: Array


class EinsteinVlasovMatterResult(StrictModule):
    """Complete proposal and all-or-nothing geometry/particle commit."""

    source: EinsteinVlasovMatterState
    candidate: EinsteinVlasovMatterState
    accepted: EinsteinVlasovMatterState
    start_geometry: ADMGridGeometry
    endpoint_geometry: ADMGridGeometry
    start_stress: RelativisticStressDepositResult
    endpoint_stress: RelativisticStressDepositResult
    constraints: Z4cConstraintEvidence
    boundary: Z4cBoundaryEvidence
    enforcement: Z4cEnforcementEvidence
    evidence: EinsteinVlasovStepEvidence
    status: Array
    successful: Array
    runtime_id: str = eqx.field(static=True)


class _EinsteinVlasovStage(StrictModule):
    geometry: ADMGridGeometry
    frame: LocalRelativisticFramePlan
    deposit: RelativisticStressDepositResult
    z4c: Z4cRHSEvaluation
    z4c_rate: Z4cState
    geodesic: EinsteinVlasovGeodesicEvaluation
    boundary: Z4cBoundaryEvidence
    evidence: EinsteinVlasovStageEvidence


def _select_tree(condition: Array, proposed, current):
    if jax.tree.structure(proposed) != jax.tree.structure(current):
        raise ValueError(
            "Atomic Einstein-Vlasov alternatives must have one tree structure."
        )
    return jax.tree.map(
        lambda new, old: jnp.where(condition, new, old) if eqx.is_array(new) else new,
        proposed,
        current,
    )


def _maximum_momentum_constraint(constraints: Z4cConstraintEvidence, /) -> Array:
    return jnp.max(jnp.abs(constraints.momentum), initial=0.0)


def _dominant_energy_violation(
    projection: StressEnergyProjection, geometry: ADMGridGeometry, /
) -> Array:
    momentum_square = ein.contract(
        "...ij,...i,...j->...",
        geometry.inverse_spatial_metric,
        projection.momentum_covector,
        projection.momentum_covector,
        backend="jax",
    )
    violation = jnp.maximum(momentum_square - projection.energy_density**2, 0.0)
    violation = jnp.maximum(violation, jnp.maximum(-projection.energy_density, 0.0))
    return jnp.max(jnp.where(projection.active, violation, 0.0), initial=0.0)


def _source_exchange_defect(projection: StressEnergyProjection, /) -> Array:
    return jnp.max(
        jnp.where(projection.active, projection.conservation_defect, 0.0), initial=0.0
    )


def _metric_payload(
    derivatives: FourthOrderDerivatives, geometry: ADMGridGeometry, /
) -> Array:
    beta = jnp.moveaxis(geometry.beta_contravariant, -1, 0)
    inverse = jnp.moveaxis(geometry.inverse_spatial_metric, (-2, -1), (0, 1))
    alpha_gradient = jnp.moveaxis(derivatives.gradient(geometry.alpha), 0, -1)
    beta_gradient = jnp.moveaxis(derivatives.gradient(beta), (0, 1), (-2, -1))
    inverse_gradient = jnp.moveaxis(
        derivatives.gradient(inverse), (0, 1, 2), (-3, -2, -1)
    )
    return jnp.concatenate(
        (
            geometry.alpha[..., None],
            geometry.beta_contravariant,
            geometry.inverse_spatial_metric.reshape(geometry.leading_shape + (9,)),
            alpha_gradient,
            beta_gradient.reshape(geometry.leading_shape + (9,)),
            inverse_gradient.reshape(geometry.leading_shape + (27,)),
        ),
        axis=-1,
    )


def adm_geodesic_rates(
    covariant_momenta: ArrayLike,
    particle_energy: ArrayLike,
    gathered_metric_payload: ArrayLike,
    active_mask: ArrayLike,
    /,
) -> EinsteinVlasovGeodesicEvaluation:
    """Coordinate-time ADM Hamiltonian rates for covariant spatial momentum."""

    momentum = jnp.asarray(covariant_momenta)
    energy = jnp.asarray(particle_energy, dtype=momentum.dtype)
    payload = jnp.asarray(gathered_metric_payload, dtype=momentum.dtype)
    active = jnp.asarray(active_mask, dtype=bool)
    capacity = momentum.shape[0]
    if momentum.shape != (capacity, 3) or energy.shape != (capacity,):
        raise ValueError(
            "Particle momentum/energy shapes must be (capacity,3)/(capacity,)."
        )
    if payload.shape != (capacity, 52) or active.shape != (capacity,):
        raise ValueError("Gathered ADM payload must have shape (capacity,52).")
    alpha = payload[:, 0]
    beta = payload[:, 1:4]
    inverse = payload[:, 4:13].reshape((capacity, 3, 3))
    alpha_gradient = payload[:, 13:16]
    beta_gradient = payload[:, 16:25].reshape((capacity, 3, 3))
    inverse_gradient = payload[:, 25:52].reshape((capacity, 3, 3, 3))
    safe_energy = jnp.where(energy > 0.0, energy, 1.0)
    raised = ein.contract("...ij,...j->...i", inverse, momentum, backend="jax")
    position_rate = alpha[:, None] * raised / safe_energy[:, None] - beta
    momentum_rate = (
        -energy[:, None] * alpha_gradient
        - 0.5
        * alpha[:, None]
        / safe_energy[:, None]
        * ein.contract(
            "...ijk,...j,...k->...i",
            inverse_gradient,
            momentum,
            momentum,
            backend="jax",
        )
        + ein.contract("...ij,...j->...i", beta_gradient, momentum, backend="jax")
    )
    finite = (
        jnp.all(jnp.isfinite(position_rate) | ~active[:, None])
        & jnp.all(jnp.isfinite(momentum_rate) | ~active[:, None])
        & jnp.all(jnp.where(active, jnp.isfinite(energy) & (energy > 0.0), True))
    )
    position_rate = jnp.where(active[:, None], position_rate, 0.0)
    momentum_rate = jnp.where(active[:, None], momentum_rate, 0.0)
    return EinsteinVlasovGeodesicEvaluation(
        position_rate,
        momentum_rate,
        energy,
        jnp.asarray(True),
        finite,
        finite,
    )


class EinsteinVlasovMatterPlan(StrictModule, NonTrainableState):
    """Research-profile, fixed-capacity midpoint coupling of particles and Z4c."""

    system: Z4cSystem
    grid: FixedGridGeometry
    derivatives: FourthOrderDerivatives
    gauge: AbstractZ4cGauge
    boundary: AbstractZ4cBoundary
    enforcement: Z4cAlgebraicEnforcement
    stress: RelativisticStressDepositPlan
    frame_provider: RelativisticFrameProvider = eqx.field(static=True)
    frame_provider_id: str = eqx.field(static=True)
    time_step: float = eqx.field(static=True)
    source_tolerance: float = eqx.field(static=True)
    mass_shell_tolerance: float = eqx.field(static=True)
    energy_condition_tolerance: float = eqx.field(static=True)
    minimum_lapse: float = eqx.field(static=True)
    maximum_metric_condition_number: float = eqx.field(static=True)
    maximum_extrinsic_curvature: float = eqx.field(static=True)
    maximum_consecutive_failures: int = eqx.field(static=True)
    require_derivative_valid: bool = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: Z4cSystem,
        grid: FixedGridGeometry,
        derivatives: FourthOrderDerivatives,
        gauge: AbstractZ4cGauge,
        boundary: AbstractZ4cBoundary,
        enforcement: Z4cAlgebraicEnforcement,
        stress: RelativisticStressDepositPlan,
        frame_provider: RelativisticFrameProvider,
        /,
        *,
        frame_provider_id: str,
        time_step: float,
        source_tolerance: float = 1.0e-8,
        mass_shell_tolerance: float = 1.0e-8,
        energy_condition_tolerance: float = 1.0e-10,
        minimum_lapse: float = 1.0e-4,
        maximum_metric_condition_number: float = 1.0e6,
        maximum_extrinsic_curvature: float = 1.0e3,
        maximum_consecutive_failures: int = 1,
        require_derivative_valid: bool = False,
    ):
        if not isinstance(system, Z4cSystem) or not isinstance(grid, FixedGridGeometry):
            raise TypeError(
                "Einstein-Vlasov requires canonical Z4c system and grid owners."
            )
        if not isinstance(derivatives, FourthOrderDerivatives):
            raise TypeError("derivatives must be FourthOrderDerivatives.")
        if not isinstance(gauge, AbstractZ4cGauge):
            raise TypeError("gauge must implement AbstractZ4cGauge.")
        if not isinstance(boundary, AbstractZ4cBoundary):
            raise TypeError("boundary must implement AbstractZ4cBoundary.")
        if not isinstance(enforcement, Z4cAlgebraicEnforcement):
            raise TypeError("enforcement must be Z4cAlgebraicEnforcement.")
        if not isinstance(stress, RelativisticStressDepositPlan):
            raise TypeError("stress must be RelativisticStressDepositPlan.")
        if not callable(frame_provider):
            raise TypeError("frame_provider must be callable.")
        if (
            stress.topology_id != grid.grid_id
            or stress.transfer.target_shape != grid.shape
            or stress.units.scale.scale_id != system.scale.scale_id
            or stress.units.convention.convention_id != system.convention.convention_id
        ):
            raise ValueError(
                "Stress transfer, Z4c grid, units, and convention must be exact."
            )
        frame_id = str(frame_provider_id).strip()
        if not frame_id:
            raise ValueError("frame_provider_id must be non-empty.")
        if derivatives.grid_shape != grid.shape or derivatives.spacing != grid.spacing:
            raise ValueError("Derivative and fixed-grid identities differ.")
        if grid.periodic != isinstance(boundary, PeriodicBoundary):
            raise ValueError("Periodic topology and boundary owner disagree.")
        values = tuple(
            float(value)
            for value in (
                time_step,
                source_tolerance,
                mass_shell_tolerance,
                energy_condition_tolerance,
                minimum_lapse,
                maximum_metric_condition_number,
                maximum_extrinsic_curvature,
            )
        )
        failures = int(maximum_consecutive_failures)
        if not isinstance(require_derivative_valid, bool):
            raise TypeError("require_derivative_valid must be Boolean.")
        if (
            any(not isfinite(value) or value < 0.0 for value in values)
            or values[0] <= 0.0
            or values[4] <= 0.0
            or values[5] <= 1.0
            or values[6] <= 0.0
            or failures < 1
        ):
            raise ValueError("Einstein-Vlasov tolerances/resource bounds are invalid.")
        if values[0] / min(grid.spacing) > 0.25:
            raise ValueError("Einstein-Vlasov step exceeds the fixed-grid Courant limit.")
        self.system = system
        self.grid = grid
        self.derivatives = derivatives
        self.gauge = gauge
        self.boundary = boundary
        self.enforcement = enforcement
        self.stress = stress
        self.frame_provider = frame_provider
        self.frame_provider_id = frame_id
        (
            self.time_step,
            self.source_tolerance,
            self.mass_shell_tolerance,
            self.energy_condition_tolerance,
            self.minimum_lapse,
            self.maximum_metric_condition_number,
            self.maximum_extrinsic_curvature,
        ) = values
        self.maximum_consecutive_failures = failures
        self.require_derivative_valid = require_derivative_valid
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-z4c-einstein-vlasov-midpoint",
                "z4c": system.system_id,
                "grid": grid.grid_id,
                "derivatives": derivatives.derivative_id,
                "gauge": gauge.gauge_id,
                "boundary": boundary.boundary_id,
                "enforcement": enforcement.enforcement_id,
                "stress": stress.plan_id,
                "frame_provider": frame_id,
                "time_step": values[0],
                "source_tolerance": values[1],
                "mass_shell_tolerance": values[2],
                "energy_condition_tolerance": values[3],
                "minimum_lapse": values[4],
                "maximum_metric_condition_number": values[5],
                "maximum_extrinsic_curvature": values[6],
                "maximum_consecutive_failures": failures,
                "require_derivative_valid": require_derivative_valid,
            }
        )

    def _frame(
        self, geometry: ADMGridGeometry, time: Array, scale_factor: Array, /
    ) -> LocalRelativisticFramePlan:
        frame = self.frame_provider(geometry, time, scale_factor)
        if not isinstance(frame, LocalRelativisticFramePlan):
            raise TypeError("frame_provider must return LocalRelativisticFramePlan.")
        if frame.geometry is not geometry:
            raise ValueError(
                "Frame provider must retain the exact stage ADM geometry object."
            )
        return frame

    def _geodesic(
        self,
        particles: RelativisticParticleState,
        frame: LocalRelativisticFramePlan,
        deposit: RelativisticStressDepositResult,
        /,
    ) -> EinsteinVlasovGeodesicEvaluation:
        gathered = self.stress.gather(
            particles, frame, _metric_payload(self.derivatives, frame.geometry)
        )
        rates = adm_geodesic_rates(
            particles.covariant_momenta,
            deposit.particle_energy,
            gathered.values,
            particles.active_mask,
        )
        support = deposit.support_complete & gathered.support_complete
        successful = rates.successful & gathered.successful & support
        return EinsteinVlasovGeodesicEvaluation(
            rates.position_rate,
            rates.covariant_momentum_rate,
            rates.particle_energy,
            support,
            rates.finite & gathered.finite,
            successful,
        )

    def _stage(
        self,
        z4c: Z4cState,
        particles: RelativisticParticleState,
        time: Array,
        token: Array,
        /,
    ) -> _EinsteinVlasovStage:
        bounded = self.boundary.apply_state(time, z4c, self.grid)
        geometry = z4c_adm_geometry(
            self.system, self.grid, bounded.state, snapshot_token=token
        )
        frame = self._frame(geometry, time, particles.scale_factor)
        deposit = self.stress.deposit(particles, frame)
        evaluation = evaluate_z4c_rhs(
            self.system,
            self.grid,
            self.derivatives,
            self.gauge,
            bounded.state,
            snapshot_token=token,
            stress_energy=deposit.projection,
        )
        boundary_rate = self.boundary.apply_rates(
            time,
            bounded.state,
            evaluation.rates,
            self.grid,
            self.derivatives,
        )
        mass_shell = jnp.max(
            jnp.where(particles.active_mask, jnp.abs(deposit.mass_shell_defect), 0.0),
            initial=0.0,
        )
        dominant = _dominant_energy_violation(deposit.projection, geometry)
        exchange = _source_exchange_defect(deposit.projection)
        finite = (
            evaluation.finite
            & deposit.successful
            & boundary_rate.evidence.finite
            & jnp.all(jnp.isfinite(particles.positions) | ~particles.active_mask[:, None])
        )
        source_valid = evaluation.source_valid & deposit.projection.compatible_with(
            geometry
        )
        derivative_valid = (
            evaluation.derivative_valid
            & jnp.all(frame.tetrad.derivative_valid)
            & deposit.support_complete
        )
        geodesic = self._geodesic(particles, frame, deposit)
        successful = (
            finite
            & source_valid
            & deposit.support_complete
            & boundary_rate.evidence.successful
            & geodesic.successful
        )
        evidence = EinsteinVlasovStageEvidence(
            token,
            exchange,
            jnp.max(jnp.abs(evaluation.constraints.hamiltonian), initial=0.0),
            _maximum_momentum_constraint(evaluation.constraints),
            mass_shell,
            dominant,
            finite & geodesic.finite,
            source_valid,
            deposit.support_complete,
            boundary_rate.evidence.successful,
            derivative_valid,
            successful,
        )
        return _EinsteinVlasovStage(
            geometry,
            frame,
            deposit,
            evaluation,
            boundary_rate.state,
            geodesic,
            boundary_rate.evidence,
            evidence,
        )

    def _advance_particles(
        self,
        source: RelativisticParticleState,
        stage: _EinsteinVlasovStage,
        target_frame: LocalRelativisticFramePlan,
        step: Array,
        time: Array,
        /,
    ) -> RelativisticParticleState:
        return self.stress.replace_covariant_dynamics(
            source,
            target_frame,
            source.positions + step * stage.geodesic.position_rate,
            source.covariant_momenta + step * stage.geodesic.covariant_momentum_rate,
            time,
            source.scale_factor,
        )

    def admit_initial_data(
        self,
        z4c: Z4cState,
        particles: RelativisticParticleState,
        solver: EinsteinVlasovConstraintSolver,
        /,
    ) -> EinsteinVlasovInitialDataResult:
        """Solve and independently verify sourced Hamiltonian/momentum constraints."""

        if not isinstance(z4c, Z4cState) or z4c.grid_id != self.grid.grid_id:
            raise ValueError("Initial Z4c state does not belong to the plan grid.")
        if not isinstance(particles, RelativisticParticleState):
            raise TypeError("particles must be RelativisticParticleState.")
        if particles.topology_id != self.grid.grid_id:
            raise ValueError("Particle and Z4c topology identities differ.")
        if not callable(solver):
            raise TypeError("solver must be callable and return solve evidence.")
        time = jnp.asarray(particles.time, dtype=z4c.values.dtype)
        before_geometry = z4c_adm_geometry(
            self.system, self.grid, z4c, snapshot_token=jnp.int32(0)
        )
        before_frame = self._frame(before_geometry, time, particles.scale_factor)
        before_deposit = self.stress.deposit(particles, before_frame)
        solved = solver(z4c, particles, before_deposit.projection, before_geometry)
        if not isinstance(solved, EinsteinVlasovConstraintSolveResult):
            raise TypeError(
                "Initial-data solver must return EinsteinVlasovConstraintSolveResult."
            )
        bounded = self.boundary.apply_state(time, solved.state, self.grid)
        enforced = self.enforcement.apply(bounded.state)
        geometry = z4c_adm_geometry(
            self.system, self.grid, enforced.state, snapshot_token=jnp.int32(1)
        )
        frame = self._frame(geometry, time, particles.scale_factor)
        admitted_particles = self.stress.replace_covariant_dynamics(
            particles,
            frame,
            particles.positions,
            particles.covariant_momenta,
            time,
            particles.scale_factor,
        )
        deposit = self.stress.deposit(admitted_particles, frame)
        evaluated = evaluate_z4c_rhs(
            self.system,
            self.grid,
            self.derivatives,
            self.gauge,
            enforced.state,
            snapshot_token=jnp.int32(1),
            stress_energy=deposit.projection,
        )
        mass_shell = jnp.max(
            jnp.where(
                admitted_particles.active_mask,
                jnp.abs(deposit.mass_shell_defect),
                0.0,
            ),
            initial=0.0,
        )
        finite = (
            solved.evidence.finite
            & deposit.successful
            & evaluated.finite
            & bounded.evidence.finite
            & enforced.evidence.finite
        )
        admitted = (
            solved.evidence.converged
            & evaluated.constraints.within_tolerance
            & evaluated.source_valid
            & (mass_shell <= self.mass_shell_tolerance)
            & bounded.evidence.successful
            & enforced.evidence.successful
            & finite
        )
        state = EinsteinVlasovMatterState(
            enforced.state,
            admitted_particles,
            time,
            0,
            0,
            0,
            False,
            runtime_id=self.runtime_id,
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "einstein-vlasov-initial-data-admission",
                "runtime": self.runtime_id,
                "solver": solved.evidence.solver_id,
            }
        )
        evidence = EinsteinVlasovInitialDataEvidence(
            solved.evidence,
            evaluated.constraints,
            evaluated.source_valid,
            mass_shell <= self.mass_shell_tolerance,
            finite,
            admitted,
            evidence_id,
        )
        return EinsteinVlasovInitialDataResult(state, deposit, evidence)

    def initialize(
        self,
        z4c: Z4cState,
        particles: RelativisticParticleState,
        solver: EinsteinVlasovConstraintSolver,
        /,
    ) -> EinsteinVlasovMatterState:
        result = self.admit_initial_data(z4c, particles, solver)
        if not bool(result.evidence.admitted):
            raise ValueError(
                "Einstein-Vlasov initial data lack a successful sourced constraint solve."
            )
        return result.state

    def advance(self, state: EinsteinVlasovMatterState, /) -> EinsteinVlasovMatterResult:
        """Propose and atomically accept or roll back one coupled midpoint step."""

        if not isinstance(state, EinsteinVlasovMatterState):
            raise TypeError("state must be EinsteinVlasovMatterState.")
        if state.runtime_id != self.runtime_id:
            raise ValueError("Einstein-Vlasov state belongs to another runtime.")
        step = jnp.asarray(self.time_step, dtype=state.time.dtype)
        index = state.accepted_steps
        start_token = z4c_snapshot_token(index, jnp.int32(1))
        middle_token = z4c_snapshot_token(index, jnp.int32(2))
        endpoint_token = z4c_snapshot_token(index, jnp.int32(7))
        start_geometry = z4c_adm_geometry(
            self.system, self.grid, state.z4c, snapshot_token=start_token
        )
        start_frame = self._frame(
            start_geometry, state.time, state.particles.scale_factor
        )
        working_particles = self.stress.replace_covariant_dynamics(
            state.particles,
            start_frame,
            state.particles.positions,
            state.particles.covariant_momenta,
            state.time,
            state.particles.scale_factor,
        )
        start = self._stage(state.z4c, working_particles, state.time, start_token)
        middle_time = state.time + 0.5 * step
        middle_z4c_raw = state.z4c.with_values(
            state.z4c.values + 0.5 * step * start.z4c_rate.values
        )
        middle_boundary = self.boundary.apply_state(
            middle_time, middle_z4c_raw, self.grid
        )
        middle_enforcement = self.enforcement.apply(middle_boundary.state)
        middle_geometry = z4c_adm_geometry(
            self.system,
            self.grid,
            middle_enforcement.state,
            snapshot_token=middle_token,
        )
        middle_frame = self._frame(
            middle_geometry, middle_time, working_particles.scale_factor
        )
        middle_particles = self._advance_particles(
            working_particles, start, middle_frame, 0.5 * step, middle_time
        )
        middle = self._stage(
            middle_enforcement.state,
            middle_particles,
            middle_time,
            middle_token,
        )
        endpoint_time = state.time + step
        endpoint_raw = state.z4c.with_values(
            state.z4c.values + step * middle.z4c_rate.values
        )
        endpoint_boundary = self.boundary.apply_state(
            endpoint_time, endpoint_raw, self.grid
        )
        endpoint_enforcement = self.enforcement.apply(endpoint_boundary.state)
        endpoint_geometry = z4c_adm_geometry(
            self.system,
            self.grid,
            endpoint_enforcement.state,
            snapshot_token=endpoint_token,
        )
        endpoint_frame = self._frame(
            endpoint_geometry, endpoint_time, working_particles.scale_factor
        )
        endpoint_particles = self._advance_particles(
            working_particles, middle, endpoint_frame, step, endpoint_time
        )
        endpoint = self._stage(
            endpoint_enforcement.state,
            endpoint_particles,
            endpoint_time,
            endpoint_token,
        )
        eigenvalues = jnp.linalg.eigvalsh(endpoint.geometry.spatial_metric)
        minimum = jnp.min(eigenvalues)
        maximum = jnp.max(eigenvalues)
        condition = maximum / jnp.where(minimum > 0.0, minimum, 1.0)
        extrinsic = jnp.max(jnp.abs(endpoint.geometry.extrinsic_curvature), initial=0.0)
        source_exchange = jnp.max(
            jnp.stack(
                (
                    start.evidence.source_exchange_defect,
                    middle.evidence.source_exchange_defect,
                    endpoint.evidence.source_exchange_defect,
                )
            )
        )
        mass_shell = jnp.max(
            jnp.stack(
                (
                    start.evidence.mass_shell_linf,
                    middle.evidence.mass_shell_linf,
                    endpoint.evidence.mass_shell_linf,
                )
            )
        )
        dominant = jnp.max(
            jnp.stack(
                (
                    start.evidence.dominant_energy_violation,
                    middle.evidence.dominant_energy_violation,
                    endpoint.evidence.dominant_energy_violation,
                )
            )
        )
        finite = (
            start.evidence.finite
            & middle.evidence.finite
            & endpoint.evidence.finite
            & jnp.all(jnp.isfinite(endpoint_enforcement.state.values))
        )
        resource_valid = (
            start.evidence.support_complete
            & middle.evidence.support_complete
            & endpoint.evidence.support_complete
        )
        strong_field_supported = (
            jnp.all(endpoint.geometry.alpha >= self.minimum_lapse)
            & (condition <= self.maximum_metric_condition_number)
            & (extrinsic <= self.maximum_extrinsic_curvature)
        )
        derivative_valid = (
            start.evidence.derivative_valid
            & middle.evidence.derivative_valid
            & endpoint.evidence.derivative_valid
        )
        constraints = endpoint.z4c.constraints
        qualified = (
            start.evidence.successful
            & middle.evidence.successful
            & endpoint.evidence.successful
            & constraints.within_tolerance
            & (source_exchange <= self.source_tolerance)
            & (mass_shell <= self.mass_shell_tolerance)
            & (dominant <= self.energy_condition_tolerance)
            & resource_valid
            & strong_field_supported
            & finite
        )
        evidence = EinsteinVlasovStepEvidence(
            (start.evidence, middle.evidence, endpoint.evidence),
            source_exchange,
            constraints.maximum_norm,
            mass_shell,
            dominant,
            condition,
            extrinsic,
            finite,
            resource_valid,
            strong_field_supported,
            derivative_valid,
            qualified,
        )
        status = jnp.asarray(int(EinsteinVlasovStatus.SUCCESS), dtype=jnp.int32)

        def add_status(current, predicate, flag):
            return jnp.where(
                predicate,
                current,
                jnp.bitwise_or(current, jnp.int32(int(flag))),
            )

        status = add_status(status, finite, EinsteinVlasovStatus.NONFINITE)
        status = add_status(
            status,
            start.evidence.source_valid
            & middle.evidence.source_valid
            & endpoint.evidence.source_valid
            & (source_exchange <= self.source_tolerance),
            EinsteinVlasovStatus.SOURCE_INVALID,
        )
        status = add_status(
            status, constraints.within_tolerance, EinsteinVlasovStatus.CONSTRAINT_EXCEEDED
        )
        status = add_status(
            status,
            mass_shell <= self.mass_shell_tolerance,
            EinsteinVlasovStatus.MASS_SHELL_EXCEEDED,
        )
        status = add_status(
            status,
            dominant <= self.energy_condition_tolerance,
            EinsteinVlasovStatus.ENERGY_CONDITION_EXCEEDED,
        )
        status = add_status(
            status, resource_valid, EinsteinVlasovStatus.RESOURCE_EXCEEDED
        )
        status = add_status(
            status,
            middle_boundary.evidence.successful & endpoint_boundary.evidence.successful,
            EinsteinVlasovStatus.BOUNDARY_FAILURE,
        )
        status = add_status(
            status,
            middle_enforcement.evidence.successful
            & endpoint_enforcement.evidence.successful,
            EinsteinVlasovStatus.ENFORCEMENT_FAILURE,
        )
        status = add_status(
            status, strong_field_supported, EinsteinVlasovStatus.STRONG_FIELD_REFUSED
        )
        time_identity = (state.time == state.particles.time) & (
            endpoint_time == endpoint_particles.time
        )
        status = add_status(
            status, time_identity, EinsteinVlasovStatus.TIME_IDENTITY_MISMATCH
        )
        status = add_status(
            status,
            derivative_valid | ~jnp.asarray(self.require_derivative_valid),
            EinsteinVlasovStatus.DERIVATIVE_INVALID,
        )
        status = jnp.where(
            state.terminal,
            jnp.bitwise_or(status, jnp.int32(int(EinsteinVlasovStatus.TERMINAL))),
            status,
        )
        successful = (status == 0) & qualified & ~state.terminal
        status = jnp.where(
            successful,
            status,
            jnp.bitwise_or(status, jnp.int32(int(EinsteinVlasovStatus.STEP_REJECTED))),
        )
        candidate = EinsteinVlasovMatterState(
            endpoint_enforcement.state,
            endpoint_particles,
            endpoint_time,
            state.accepted_steps + 1,
            state.rejected_steps,
            0,
            False,
            runtime_id=self.runtime_id,
        )
        next_failures = state.consecutive_failures + 1
        rejected = EinsteinVlasovMatterState(
            state.z4c,
            state.particles,
            state.time,
            state.accepted_steps,
            state.rejected_steps + 1,
            next_failures,
            state.terminal | (next_failures >= self.maximum_consecutive_failures),
            runtime_id=self.runtime_id,
        )
        accepted = _select_tree(successful, candidate, rejected)
        return EinsteinVlasovMatterResult(
            state,
            candidate,
            accepted,
            start.geometry,
            endpoint.geometry,
            start.deposit,
            endpoint.deposit,
            constraints,
            endpoint_boundary.evidence,
            endpoint_enforcement.evidence,
            evidence,
            status,
            successful,
            self.runtime_id,
        )


__all__ = [
    "EinsteinVlasovConstraintSolveEvidence",
    "EinsteinVlasovConstraintSolveResult",
    "EinsteinVlasovGeodesicEvaluation",
    "EinsteinVlasovInitialDataEvidence",
    "EinsteinVlasovInitialDataResult",
    "EinsteinVlasovMatterPlan",
    "EinsteinVlasovMatterResult",
    "EinsteinVlasovMatterState",
    "EinsteinVlasovStageEvidence",
    "EinsteinVlasovStatus",
    "EinsteinVlasovStepEvidence",
    "RelativisticFrameProvider",
    "adm_geodesic_rates",
]
