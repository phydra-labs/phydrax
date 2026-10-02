# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Scientific bindings and immutable committed projector-QMC records."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import finite_real_scalar
from ..operators.quantum.lattice._column import QuantumLatticeColumnOperator
from ..operators.quantum.lattice._guide import GuidedQuantumColumnOperator, QuantumGuide
from ..typing import (
    as_array,
    Bool,
    Complex128,
    Dim,
    Float64,
    Identifier,
    Int32,
    Int64,
    parse,
    PRNGKey,
    Scalar,
    Scope,
    UInt32,
)
from ..units import DIMENSIONLESS, ENERGY, UnitDefinition


SpawnPolicy: TypeAlias = Literal["exact", "sampled", "semistochastic"]
CompressionPolicy: TypeAlias = Literal["none", "threshold"]
ControllerPolicy: TypeAlias = Literal["fixed-shift", "double-log"]


class ReplicaDim(Dim):
    pass


class SupportDim(Dim):
    pass


class KeyWordDim(Dim):
    pass


class InitialDim(Dim):
    pass


class TrialDim(Dim):
    pass


class HistoryDim(Dim):
    pass


class PairDim(Dim):
    pass


class ObservableDim(Dim):
    pass


class ProjectorMonteCarloStatus(IntEnum):
    SUCCESS = 0
    INVALID_STATE = 1
    HISTORY_EXHAUSTED = 2
    EXTINCTION = 3
    ATTEMPT_LIMIT = 4
    WORK_LIMIT = 5
    OPERATOR_FAILURE = 6
    GUIDE_FAILURE = 7
    INTERMEDIATE_GROUP_OVERFLOW = 8
    NONFINITE_PROPAGATION = 9
    FINAL_SUPPORT_OVERFLOW = 10
    NONFINITE_CONTROLLER = 11
    OBSERVATION_FAILURE = 12
    STEP_COUNTER_EXHAUSTED = 13


class ProjectorMonteCarloProblem(StrictModule):
    """Fixed physical input; dt is scaled imaginary time in inverse energy units."""

    __strict_contract__ = True
    hamiltonian: QuantumLatticeColumnOperator
    initial_keys: UInt32[InitialDim, KeyWordDim]
    initial_coefficients: Complex128[InitialDim]
    initial_active: Bool[InitialDim]
    trial_keys: UInt32[TrialDim, KeyWordDim]
    trial_coefficients: Complex128[TrialDim]
    trial_active: Bool[TrialDim]
    guide: QuantumGuide | None
    observables: tuple[QuantumLatticeColumnOperator, ...]
    observable_units: tuple[UnitDefinition, ...] = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    energy_unit: UnitDefinition = eqx.field(static=True)
    inverse_energy_unit: UnitDefinition = eqx.field(static=True)
    provenance_id: Identifier = eqx.field(static=True)
    problem_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        hamiltonian: QuantumLatticeColumnOperator,
        initial_keys: ArrayLike,
        initial_coefficients: ArrayLike,
        /,
        *,
        dt: float,
        energy_unit: UnitDefinition,
        inverse_energy_unit: UnitDefinition,
        provenance_id: str,
        initial_active: ArrayLike | None = None,
        trial_keys: ArrayLike | None = None,
        trial_coefficients: ArrayLike | None = None,
        trial_active: ArrayLike | None = None,
        guide: QuantumGuide | None = None,
        observables: tuple[QuantumLatticeColumnOperator, ...] = (),
        observable_units: Sequence[UnitDefinition] = (),
    ) -> None:
        if not jax.config.x64_enabled:
            raise ValueError("Projector Monte Carlo requires jax_enable_x64.")
        if not isinstance(hamiltonian, QuantumLatticeColumnOperator):
            raise TypeError("hamiltonian must be the native original column operator.")
        if not hamiltonian.self_adjoint:
            raise ValueError("The original Hamiltonian must certify self-adjointness.")
        if not isinstance(energy_unit, UnitDefinition) or not isinstance(
            inverse_energy_unit, UnitDefinition
        ):
            raise TypeError(
                "Explicit native energy and inverse-energy units are required."
            )
        if energy_unit.dimension not in (ENERGY, DIMENSIONLESS):
            raise ValueError(
                "Hamiltonian units must have energy or explicit dimensionless dimension."
            )
        if (
            energy_unit.reference_system_id != inverse_energy_unit.reference_system_id
            or not energy_unit.dimension.multiply(
                inverse_energy_unit.dimension
            ).is_dimensionless
            or energy_unit.scale_to_reference * inverse_energy_unit.scale_to_reference
            != 1
        ):
            raise ValueError(
                "dt units must be the reciprocal of the declared energy unit."
            )
        dt = finite_real_scalar(dt, "dt")
        if dt <= 0:
            raise ValueError("dt must be finite and positive.")
        provenance_id = parse(provenance_id, Identifier, "provenance_id")
        if guide is not None:
            if not isinstance(guide, QuantumGuide):
                raise TypeError("guide must be a frozen QuantumGuide.")
            if guide.domain_id != hamiltonian.domain.domain_id:
                raise ValueError("Guide and Hamiltonian domains differ.")
        if not isinstance(observables, tuple):
            raise TypeError("Requested observables must be a fixed tuple.")
        units = tuple(observable_units)
        if len(units) != len(observables):
            raise ValueError(
                "Every requested physical observable requires its explicit unit."
            )
        if any(not isinstance(unit, UnitDefinition) for unit in units):
            raise TypeError("Observable units must be native UnitDefinition values.")
        for observable in observables:
            if not isinstance(observable, QuantumLatticeColumnOperator):
                raise TypeError("Observables must be native physical column operators.")
            if observable.domain.domain_id != hamiltonian.domain.domain_id:
                raise ValueError("Observable and Hamiltonian domains differ.")
        scope = Scope()
        keys = parse(
            jnp.asarray(initial_keys),
            UInt32[InitialDim, KeyWordDim],
            "initial_keys",
            scope=scope,
        )
        coefficients = as_array(
            initial_coefficients,
            Complex128[InitialDim],
            "initial_coefficients",
            scope=scope,
        )
        active = as_array(
            jnp.ones(coefficients.shape, dtype=jnp.bool_)
            if initial_active is None
            else initial_active,
            Bool[InitialDim],
            "initial_active",
            scope=scope,
        )
        if (trial_keys is None) != (trial_coefficients is None):
            raise ValueError("Trial keys and coefficients must be supplied together.")
        tkeys = parse(
            keys if trial_keys is None else jnp.asarray(trial_keys),
            UInt32[TrialDim, KeyWordDim],
            "trial_keys",
            scope=scope,
        )
        tcoeff = as_array(
            coefficients if trial_coefficients is None else trial_coefficients,
            Complex128[TrialDim],
            "trial_coefficients",
            scope=scope,
        )
        tactive = as_array(
            active
            if trial_active is None and trial_keys is None
            else jnp.ones(tcoeff.shape, dtype=jnp.bool_)
            if trial_active is None
            else trial_active,
            Bool[TrialDim],
            "trial_active",
            scope=scope,
        )
        if keys.shape[1] != hamiltonian.domain.codec.word_count:
            raise ValueError("Packed word count does not match the domain.")
        (
            self.hamiltonian,
            self.initial_keys,
            self.initial_coefficients,
            self.initial_active,
        ) = hamiltonian, keys, coefficients, active
        self.trial_keys, self.trial_coefficients, self.trial_active = (
            tkeys,
            tcoeff,
            tactive,
        )
        self.guide, self.observables, self.dt = guide, observables, dt
        self.observable_units = units
        self.energy_unit, self.inverse_energy_unit, self.provenance_id = (
            energy_unit,
            inverse_energy_unit,
            provenance_id,
        )
        self.problem_id = canonical_fingerprint(
            {
                "kind": "projector-physical-problem",
                "operator": hamiltonian.operator_id,
                "domain": hamiltonian.domain.domain_id,
                "guide": "unguided" if guide is None else guide.guide_id,
                "observables": tuple(item.operator_id for item in observables),
                "observable_units": tuple(unit.unit_id for unit in units),
                "dt": dt,
                "energy_unit": energy_unit.unit_id,
                "inverse_energy_unit": inverse_energy_unit.unit_id,
                "provenance": provenance_id,
            }
        )


class ProjectorMonteCarloPlan(StrictModule):
    __strict_contract__ = True
    replicas: int = eqx.field(static=True)
    support_capacity: int = eqx.field(static=True)
    group_capacity: int = eqx.field(static=True)
    event_capacity: int = eqx.field(static=True)
    attempt_capacity: int = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    maximum_retained_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    spawn_policy: SpawnPolicy = eqx.field(static=True)
    compression: CompressionPolicy = eqx.field(static=True)
    controller: ControllerPolicy = eqx.field(static=True)
    boost: float = eqx.field(static=True)
    relative_threshold: float = eqx.field(static=True)
    absolute_threshold: float = eqx.field(static=True)
    theta: float = eqx.field(static=True)
    initial_shift: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    restoring: float = eqx.field(static=True)
    target_population: float = eqx.field(static=True)
    policy_id: Identifier = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        *,
        replicas: int,
        support_capacity: int,
        group_capacity: int,
        event_capacity: int,
        attempt_capacity: int,
        source_capacity: int,
        history_capacity: int,
        maximum_retained_bytes: int,
        maximum_workspace_bytes: int,
        spawn_policy: SpawnPolicy = "exact",
        compression: CompressionPolicy = "none",
        controller: ControllerPolicy = "fixed-shift",
        boost: float = 1.0,
        relative_threshold: float = 1.0,
        absolute_threshold: float = float("inf"),
        theta: float = 1.0,
        initial_shift: float = 0.0,
        damping: float = 0.05,
        restoring: float = 0.001,
        target_population: float = 1.0,
    ) -> None:
        capacities = {
            "replicas": replicas,
            "support_capacity": support_capacity,
            "group_capacity": group_capacity,
            "event_capacity": event_capacity,
            "attempt_capacity": attempt_capacity,
            "source_capacity": source_capacity,
            "history_capacity": history_capacity,
            "maximum_retained_bytes": maximum_retained_bytes,
            "maximum_workspace_bytes": maximum_workspace_bytes,
        }
        for name, value in capacities.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value < 1 or (not name.endswith("_bytes") and value > 2**31 - 1):
                raise ValueError(
                    f"{name} must be positive; runtime capacities must fit the native work index."
                )
        if group_capacity < support_capacity:
            raise ValueError("Require G >= S.")
        if group_capacity + event_capacity > 2**31 - 1:
            raise ValueError(
                "Combined retention/event grouping must fit the native work index."
            )
        spawn = parse(spawn_policy, SpawnPolicy, "spawn_policy")
        compression_ = parse(compression, CompressionPolicy, "compression")
        controller_ = parse(controller, ControllerPolicy, "controller")
        if (
            not isfinite(boost)
            or boost <= 0
            or not isfinite(relative_threshold)
            or relative_threshold < 0
            or absolute_threshold < 0
            or absolute_threshold != absolute_threshold
        ):
            raise ValueError("Spawn thresholds and boost are invalid.")
        if (
            not isfinite(theta)
            or theta <= 0
            or not isfinite(initial_shift)
            or not isfinite(damping)
            or damping < 0
            or not isfinite(restoring)
            or restoring < 0
            or not isfinite(target_population)
            or target_population <= 0
        ):
            raise ValueError("Compression/controller parameters are invalid.")
        self.replicas, self.support_capacity, self.group_capacity, self.event_capacity = (
            replicas,
            support_capacity,
            group_capacity,
            event_capacity,
        )
        self.attempt_capacity, self.source_capacity, self.history_capacity = (
            attempt_capacity,
            source_capacity,
            history_capacity,
        )
        self.maximum_retained_bytes, self.maximum_workspace_bytes = (
            maximum_retained_bytes,
            maximum_workspace_bytes,
        )
        self.spawn_policy, self.compression, self.controller = (
            spawn,
            compression_,
            controller_,
        )
        self.boost, self.relative_threshold, self.absolute_threshold, self.theta = (
            boost,
            relative_threshold,
            absolute_threshold,
            theta,
        )
        self.initial_shift, self.damping, self.restoring, self.target_population = (
            initial_shift,
            damping,
            restoring,
            target_population,
        )
        self.policy_id = canonical_fingerprint(
            {
                "spawn": spawn,
                "compression": compression_,
                "controller": controller_,
                "boost": boost,
                "relative": relative_threshold,
                "absolute": "infinity"
                if absolute_threshold == float("inf")
                else absolute_threshold,
                "theta": theta,
                "initial_shift": initial_shift,
                "damping": damping,
                "restoring": restoring,
                "target": target_population,
                "precision": "complex128",
                "reduction": "compensated",
                "count": "physical-raw-route-bound",
            }
        )
        self.plan_id = canonical_fingerprint(
            {"policy": self.policy_id, "capacities": capacities}
        )


class PreparedProjectorMonteCarlo(StrictModule):
    __strict_contract__ = True
    problem: ProjectorMonteCarloProblem
    plan: ProjectorMonteCarloPlan = eqx.field(static=True)
    original_operator: QuantumLatticeColumnOperator
    operator: QuantumLatticeColumnOperator | GuidedQuantumColumnOperator
    guide: QuantumGuide | None
    initial_keys: UInt32[SupportDim, KeyWordDim]
    initial_coefficients: Complex128[SupportDim]
    initial_active: Bool[SupportDim]
    trial_keys: UInt32[TrialDim, KeyWordDim]
    trial_coefficients: Complex128[TrialDim]
    trial_active: Bool[TrialDim]
    observables: tuple[QuantumLatticeColumnOperator, ...]
    observable_units: tuple[UnitDefinition, ...] = eqx.field(static=True)
    pair_ids: tuple[tuple[int, int], ...] = eqx.field(static=True)
    scientific_id: Identifier = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)
    retained_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class ProjectorMonteCarloHistory(StrictModule):
    __strict_contract__ = True
    applied_shifts: Float64[ReplicaDim, HistoryDim]
    populations: Float64[ReplicaDim, HistoryDim]
    projected_numerator: Complex128[ReplicaDim, HistoryDim]
    projected_denominator: Complex128[ReplicaDim, HistoryDim]
    pair_numerators: Complex128[PairDim, HistoryDim, ObservableDim]
    pair_denominators: Complex128[PairDim, HistoryDim]
    pre_annihilation_norm: Float64[ReplicaDim, HistoryDim]
    post_annihilation_norm: Float64[ReplicaDim, HistoryDim]
    valid: Bool[HistoryDim]
    count: Int64[Scalar]
    scientific_id: Identifier = eqx.field(static=True)
    domain_id: Identifier = eqx.field(static=True)
    operator_id: Identifier = eqx.field(static=True)
    guide_id: Identifier = eqx.field(static=True)
    metric_id: Identifier = eqx.field(static=True)


class ProjectorMonteCarloState(StrictModule):
    __strict_contract__ = True
    support_keys: UInt32[ReplicaDim, SupportDim, KeyWordDim]
    coefficients: Complex128[ReplicaDim, SupportDim]
    active: Bool[ReplicaDim, SupportDim]
    shifts: Float64[ReplicaDim]
    populations: Float64[ReplicaDim]
    step: Int64[Scalar]
    root_key: PRNGKey
    history: ProjectorMonteCarloHistory
    scientific_id: Identifier = eqx.field(static=True)
    prepared_id: Identifier = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)
    domain_id: Identifier = eqx.field(static=True)
    codec_id: Identifier = eqx.field(static=True)
    operator_id: Identifier = eqx.field(static=True)
    guide_id: Identifier = eqx.field(static=True)
    metric_id: Identifier = eqx.field(static=True)


class ProjectorMonteCarloEvidence(StrictModule):
    __strict_contract__ = True
    replica_status: Int32[ReplicaDim]
    required_groups: Int64[ReplicaDim]
    required_support: Int64[ReplicaDim]
    maximum_attempts: Int64[ReplicaDim]
    event_count: Int64[ReplicaDim]
    exact_sources: Int64[ReplicaDim]
    sampled_sources: Int64[ReplicaDim]
    pre_annihilation_norm: Float64[ReplicaDim]
    post_annihilation_norm: Float64[ReplicaDim]
    group_capacity: int = eqx.field(static=True)
    support_capacity: int = eqx.field(static=True)
    attempt_capacity: int = eqx.field(static=True)
    raw_route_bound: int = eqx.field(static=True)


class ProjectorMonteCarloStepResult(StrictModule):
    __strict_contract__ = True
    state: ProjectorMonteCarloState
    status: Int32[Scalar]
    accepted: Bool[Scalar]
    evidence: ProjectorMonteCarloEvidence


class ProjectorMonteCarloResult(StrictModule):
    __strict_contract__ = True
    state: ProjectorMonteCarloState
    status: Int32[Scalar]
    evidence: ProjectorMonteCarloEvidence
    requested_steps: Int64[Scalar]
    accepted_steps: Int64[Scalar]
