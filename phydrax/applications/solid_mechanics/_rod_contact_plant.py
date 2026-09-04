#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_tree import ArrayPyTreeSchema
from ..._fingerprint import canonical_fingerprint
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantStepContext,
)
from ..contact._cone import ContactConeSolverPlan
from ..contact._rod_capsule import ReducedRodCapsuleContactParticipant
from ..contact._rod_contact_lifecycle import (
    CompositeContactResponse,
    CompositeContactResult,
    prepare_composite_contact_block,
    PreparedRodContactSearch,
    RodContactCCDPlan,
    RodContactCCDResult,
    RodContactManifoldState,
    RodContactManifoldTransition,
    RodContactSearchFailure,
    RodContactSearchResult,
    RodContactWitnessBatch,
)
from ._rod_loads import RodLoadLedger
from ._rod_plant import (
    ReducedRodPassiveActuatorState,
    ReducedRodPassiveSensorState,
    ReducedRodPlantParameters,
)
from ._rod_reduced_dynamics import (
    PreparedReducedRodDynamics,
    ReducedRodDynamicsEvaluation,
    ReducedRodMaterialControl,
    ReducedRodMaterialState,
)
from ._rod_reduced_integrators import (
    integrate_reduced_rod_step,
    ReducedRodIntegrationState,
    ReducedRodIntegratorPolicy,
    ReducedRodStepResult,
)
from ._rod_reduction import ReducedRodState


RodContactCapabilityId: TypeAlias = Literal[
    "fixed-base-circular-capsule-plane-self-frictionless",
    "fixed-base-circular-capsule-plane-self-isotropic-coulomb",
]
FRICTIONLESS_ROD_CONTACT_CAPABILITY: RodContactCapabilityId = (
    "fixed-base-circular-capsule-plane-self-frictionless"
)
ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY: RodContactCapabilityId = (
    "fixed-base-circular-capsule-plane-self-isotropic-coulomb"
)


class ReducedRodContactPlantStatus(IntEnum):
    """Deterministic first failing stage of one atomic contact transaction."""

    SUCCESS = 0
    FREE_DYNAMICS_FAILED = 1
    SEARCH_FAILED = 2
    SEARCH_CAPACITY_EXCEEDED = 3
    CCD_FAILED = 4
    CCD_SAFE_PREFIX_ONLY = 5
    EVENT_SEARCH_FAILED = 6
    MANIFOLD_TRANSITION_FAILED = 7
    MASS_RESPONSE_FAILED = 8
    RESPONSE_SOLVE_FAILED = 9
    CORRECTED_INTEGRATION_FAILED = 10
    CORRECTION_CCD_FAILED = 11
    FINAL_SEARCH_FAILED = 12
    FINAL_MANIFOLD_FAILED = 13
    FINAL_GAP_INVALID = 14
    ENERGY_INVALID = 15
    CONSERVATION_INVALID = 16


class ReducedRodContactPlantState(StrictModule):
    """Complete mechanics, material, actuator, contact-history, and sensor payload."""

    reduced_state: ReducedRodState
    material_state: ReducedRodMaterialState
    actuator_state: ReducedRodPassiveActuatorState
    contact_state: RodContactManifoldState
    sensor_state: ReducedRodPassiveSensorState

    def __init__(
        self,
        reduced_state: ReducedRodState,
        material_state: ReducedRodMaterialState,
        actuator_state: ReducedRodPassiveActuatorState,
        contact_state: RodContactManifoldState,
        sensor_state: ReducedRodPassiveSensorState,
        /,
    ):
        if not isinstance(reduced_state, ReducedRodState):
            raise TypeError("reduced_state must be ReducedRodState.")
        if not isinstance(material_state, ReducedRodMaterialState):
            raise TypeError("material_state must be ReducedRodMaterialState.")
        if not isinstance(actuator_state, ReducedRodPassiveActuatorState):
            raise TypeError("actuator_state must be ReducedRodPassiveActuatorState.")
        if not isinstance(contact_state, RodContactManifoldState):
            raise TypeError("contact_state must be RodContactManifoldState.")
        if not isinstance(sensor_state, ReducedRodPassiveSensorState):
            raise TypeError("sensor_state must be ReducedRodPassiveSensorState.")
        self.reduced_state = reduced_state
        self.material_state = material_state
        self.actuator_state = actuator_state
        self.contact_state = contact_state
        self.sensor_state = sensor_state


class ReducedRodContactEnergyEvidence(StrictModule):
    source_mechanical_energy: Array
    free_mechanical_energy: Array
    final_mechanical_energy: Array
    contact_energy_change: Array
    friction_dissipation: Array
    scale: Array
    finite: Array
    contact_nonenergizing: Array
    friction_dissipative: Array
    valid: Array


class ReducedRodContactConservationEvidence(StrictModule):
    impulse_response_residual: Array
    impulse_response_scale: Array
    response_duality_residual: Array
    response_duality_scale: Array
    finite: Array
    impulse_response_valid: Array
    duality_valid: Array
    valid: Array


class ReducedRodContactPlantResetEvidence(StrictModule):
    evaluation: ReducedRodDynamicsEvaluation
    search_result: RodContactSearchResult
    manifold_transition: RodContactManifoldTransition
    minimum_gap: Array
    finite: Array
    valid: Array
    status: Array
    plant_id: str = eqx.field(static=True)


class ReducedRodContactPlantStepEvidence(StrictModule):
    """Every staged result retained even when the complete payload rolls back."""

    free_step: ReducedRodStepResult
    swept_ccd: RodContactCCDResult
    event_search: RodContactSearchResult
    event_manifold: RodContactManifoldTransition
    response: CompositeContactResult
    corrected_evaluation: ReducedRodDynamicsEvaluation
    correction_ccd: RodContactCCDResult
    final_search: RodContactSearchResult
    candidate_final_manifold: RodContactManifoldTransition
    accepted_final_manifold: RodContactManifoldTransition
    final_minimum_gap: Array
    position_correction_iterations: Array
    position_fixed_point_residual: Array
    position_correction_norm: Array
    position_correction_successful: Array
    energy: ReducedRodContactEnergyEvidence
    conservation: ReducedRodContactConservationEvidence
    full_interval_covered: Array
    finite: Array
    valid: Array
    status: Array
    backend_status: Array
    capability_id: RodContactCapabilityId = eqx.field(static=True)
    plant_id: str = eqx.field(static=True)


def _route_values(value: ArrayLike, capacity: int, name: str, /) -> Array:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.inexact) or np.iscomplexobj(array):
        raise TypeError(f"{name} must be a real inexact scalar or route vector.")
    if array.shape == ():
        array = np.full((capacity,), float(array), dtype=array.dtype)
    if array.shape != (capacity,) or np.any(~np.isfinite(array)):
        raise ValueError(f"{name} must be finite with scalar or contact-capacity shape.")
    return jnp.asarray(array)


def _minimum_gap(search: RodContactSearchResult, /) -> Array:
    valid = search.witnesses.valid
    return jnp.where(
        jnp.any(valid),
        jnp.min(jnp.where(valid, search.witnesses.physical_gap, jnp.inf)),
        jnp.asarray(0.0, dtype=search.witnesses.physical_gap.dtype),
    )


def _remap_speculative_impulse(
    previous_witnesses: RodContactWitnessBatch,
    previous_impulse: Array,
    current_witnesses: RodContactWitnessBatch,
    fallback: Array,
    /,
) -> Array:
    warm_start = jnp.asarray(fallback)
    previous_keys = np.asarray(previous_witnesses.route_keys)
    previous_valid = np.asarray(previous_witnesses.valid)
    current_keys = np.asarray(current_witnesses.route_keys)
    current_valid = np.asarray(current_witnesses.valid)
    for current_index in np.flatnonzero(current_valid).tolist():
        matches = np.flatnonzero(
            previous_valid & (previous_keys == current_keys[current_index])
        )
        if matches.size != 1:
            continue
        previous_index = int(matches[0])
        world_tangent = (
            previous_witnesses.tangent_basis[previous_index]
            @ previous_impulse[previous_index, 1:]
        )
        current_tangent = current_witnesses.tangent_basis[current_index].T @ world_tangent
        warm_start = warm_start.at[current_index, 0].set(
            previous_impulse[previous_index, 0]
        )
        warm_start = warm_start.at[current_index, 1:].set(current_tangent)
    return warm_start


def _first_failure_status(
    stages: tuple[tuple[Array, ReducedRodContactPlantStatus], ...], /
) -> Array:
    status = jnp.asarray(ReducedRodContactPlantStatus.SUCCESS, dtype=jnp.int32)
    for successful, failure in stages:
        status = jnp.where(
            (status == int(ReducedRodContactPlantStatus.SUCCESS)) & ~successful,
            jnp.asarray(int(failure), dtype=jnp.int32),
            status,
        )
    return status


class PreparedReducedRodContactPlant(AbstractDiscretePlant, NonTrainableState):
    """Atomic fixed-base capsule-rod mechanics and nonsmooth contact plant.

    A transaction stages a free reduced step, one swept canonical candidate
    epoch, certified CCD, persistent-manifold transition, matrix-free composite
    cone response, event-corrected integration, and final geometric/mechanical
    certificates.  A certified prefix is diagnostic only: it is never committed
    as a shortened step.  PlantCore selects the complete payload, clock, index,
    and PRNG key with the single final success mask.
    """

    dynamics: PreparedReducedRodDynamics
    policy: ReducedRodIntegratorPolicy
    participant: ReducedRodCapsuleContactParticipant
    search: PreparedRodContactSearch
    ccd: RodContactCCDPlan
    cone_solver: ContactConeSolverPlan
    native_loads: RodLoadLedger | None
    material_control: ReducedRodMaterialControl
    dynamic_friction: Array
    static_friction: Array
    compliance: Array
    material_revision: int = eqx.field(static=True)
    retention_steps: int = eqx.field(static=True)
    speculative_relaxation: float = eqx.field(static=True)
    maximum_speculative_iterations: int = eqx.field(static=True)
    gap_tolerance: float = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    conservation_tolerance: float = eqx.field(static=True)
    capability_id: RodContactCapabilityId = eqx.field(static=True)
    initial_state: ReducedRodContactPlantState
    default_parameters: ReducedRodPlantParameters
    state_schema: ArrayPyTreeSchema
    control_schema: None
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: ReducedRodContactPlantState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    plant_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedReducedRodDynamics,
        policy: ReducedRodIntegratorPolicy,
        participant: ReducedRodCapsuleContactParticipant,
        search: PreparedRodContactSearch,
        ccd: RodContactCCDPlan,
        /,
        *,
        dynamic_friction: ArrayLike = 0.0,
        static_friction: ArrayLike | None = None,
        compliance: ArrayLike = 0.0,
        cone_solver: ContactConeSolverPlan | None = None,
        material_revision: int = 0,
        retention_steps: int = 2,
        maximum_speculative_iterations: int = 16,
        gap_tolerance: float = 1.0e-6,
        speculative_relaxation: float = 0.5,
        energy_tolerance: float = 1.0e-6,
        conservation_tolerance: float = 1.0e-6,
        initial_reduced_state: ReducedRodState | None = None,
        initial_material_state: ReducedRodMaterialState | None = None,
        native_loads: RodLoadLedger | None = None,
        capability_id: RodContactCapabilityId | None = None,
    ):
        if not isinstance(dynamics, PreparedReducedRodDynamics):
            raise TypeError("dynamics must be PreparedReducedRodDynamics.")
        from ._rod_reduced_integrators import (
            ReducedRodImplicitMidpoint,
            ReducedRodSemiImplicitVelocityEuler,
        )

        if not isinstance(
            policy, (ReducedRodSemiImplicitVelocityEuler, ReducedRodImplicitMidpoint)
        ):
            raise TypeError("policy must select one reduced rod integrator route.")
        if not isinstance(participant, ReducedRodCapsuleContactParticipant):
            raise TypeError("participant must be ReducedRodCapsuleContactParticipant.")
        if not isinstance(search, PreparedRodContactSearch):
            raise TypeError("search must be PreparedRodContactSearch.")
        if not isinstance(ccd, RodContactCCDPlan):
            raise TypeError("ccd must be RodContactCCDPlan.")
        solver = ContactConeSolverPlan() if cone_solver is None else cone_solver
        if not isinstance(solver, ContactConeSolverPlan):
            raise TypeError("cone_solver must be ContactConeSolverPlan or None.")
        if dynamics.reduction.prepared_id != participant.reduced.prepared_id:
            raise ValueError("Dynamics and contact participant must share one reduction.")
        if search.surface_plan.topology_id != participant.surface_plan.topology_id:
            raise ValueError("Search and participant must share one capsule topology.")
        if dynamics.reduction.plan.dimension != 3:
            raise ValueError("Reduced rod contact requires a spatial reduction.")
        if native_loads is not None:
            if not isinstance(native_loads, RodLoadLedger):
                raise TypeError("native_loads must be RodLoadLedger or None.")
            native_loads._validate_rod(dynamics.reduction.rod)
        revision = int(material_revision)
        retention = int(retention_steps)
        if revision < 0 or retention < 0:
            raise ValueError("material_revision and retention_steps must be nonnegative.")
        speculative_iterations = int(maximum_speculative_iterations)
        if speculative_iterations <= 0:
            raise ValueError("maximum_speculative_iterations must be positive.")
        relaxation = float(speculative_relaxation)
        if not isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
            raise ValueError("speculative_relaxation must lie in (0, 1].")
        tolerances = tuple(
            float(value)
            for value in (gap_tolerance, energy_tolerance, conservation_tolerance)
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Contact plant tolerances must be finite and nonnegative.")
        capacity = search.plan.total_capacity
        dynamic = _route_values(dynamic_friction, capacity, "dynamic_friction")
        static = (
            dynamic
            if static_friction is None
            else _route_values(static_friction, capacity, "static_friction")
        )
        if np.any(np.asarray(dynamic) < 0.0) or np.any(
            np.asarray(static) < np.asarray(dynamic)
        ):
            raise ValueError("Friction requires static >= dynamic >= 0.")
        compliance_ = np.asarray(compliance)
        if (
            not np.issubdtype(compliance_.dtype, np.inexact)
            or np.iscomplexobj(compliance_)
            or np.any(~np.isfinite(compliance_))
            or np.any(compliance_ < 0.0)
            or compliance_.shape not in ((), (3,), (capacity, 3))
        ):
            raise ValueError(
                "compliance must be a finite nonnegative scalar, local vector, or contact-local array."
            )
        derived_capability: RodContactCapabilityId = (
            FRICTIONLESS_ROD_CONTACT_CAPABILITY
            if not np.any(np.asarray(static) != 0.0)
            else ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY
        )
        selected_capability = (
            derived_capability if capability_id is None else capability_id
        )
        if selected_capability not in (
            FRICTIONLESS_ROD_CONTACT_CAPABILITY,
            ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY,
        ):
            raise ValueError("Unsupported rod contact capability_id.")
        if selected_capability != derived_capability:
            raise ValueError("capability_id does not match the prepared friction law.")
        reduced = (
            dynamics.reduction.initialize_state()
            if initial_reduced_state is None
            else initial_reduced_state
        )
        dynamics.reduction.validate_state(reduced)
        material = (
            dynamics.initialize_material_state()
            if initial_material_state is None
            else initial_material_state
        )
        if not isinstance(material, ReducedRodMaterialState):
            raise TypeError(
                "initial_material_state must be ReducedRodMaterialState or None."
            )
        template = dynamics.initialize_material_state()
        if (
            material.stretch_shear_history.shape != template.stretch_shear_history.shape
            or material.stretch_shear_history.dtype
            != template.stretch_shear_history.dtype
            or material.bend_twist_history.shape != template.bend_twist_history.shape
            or material.bend_twist_history.dtype != template.bend_twist_history.dtype
        ):
            raise ValueError("Initial material history does not match prepared dynamics.")
        dtype = reduced.values.dtype
        empty = jnp.zeros((0,), dtype=dtype)
        manifold = RodContactManifoldState.empty(capacity, dtype=np.dtype(dtype))
        initial = ReducedRodContactPlantState(
            reduced,
            material,
            ReducedRodPassiveActuatorState(empty),
            manifold,
            ReducedRodPassiveSensorState(empty),
        )
        parameters = ReducedRodPlantParameters(empty)
        material_control = dynamics.initialize_material_control()
        state_schema = ArrayPyTreeSchema.from_tree(initial, case_ndim=0)
        parameter_schema = ArrayPyTreeSchema.from_tree(parameters, case_ndim=0)
        semantic = SemanticProvenance(
            {
                "kind": "prepared-atomic-reduced-rod-contact-plant",
                "profile": "fixed-base-spatial-constant-circular-capsule-plane-self",
                "capability_id": selected_capability,
                "integrator_route": policy.route,
                "search_route": int(search.plan.route),
                "ccd": "conservative-advancement-no-shortening",
                "response": "matrix-free-composite-true-dual-coulomb",
                "speculative_correction": "bounded-velocity-consistent-contact-iteration",
                "state_schema": state_schema.content_id,
                "parameter_schema": parameter_schema.content_id,
                "speculative_relaxation": relaxation.hex(),
            }
        )
        numeric = NumericRevision(
            semantic,
            {
                "dynamics_id": dynamics.dynamics_id,
                "participant_id": participant.participant_id,
                "search_id": search.prepared_id,
                "ccd_id": ccd.plan_id,
                "cone_solver_id": solver.plan_id,
                "material_control": material_control,
                "material_revision": revision,
                "dynamic_friction": dynamic,
                "static_friction": static,
                "compliance": compliance_,
                "maximum_speculative_iterations": speculative_iterations,
                "initial_state": initial,
                "native_load_ledger_id": (
                    None if native_loads is None else native_loads.ledger_id
                ),
            },
        )
        signature = ExecutableSignature(
            shapes={
                "reduced_state": reduced.values.shape,
                "stretch_shear_history": material.stretch_shear_history.shape,
                "bend_twist_history": material.bend_twist_history.shape,
                "manifold_keys": manifold.route_keys.shape,
                "manifold_impulse": manifold.impulse.shape,
                "manifold_slip": manifold.slip.shape,
                "friction": dynamic.shape,
                "compliance": compliance_.shape,
                "parameters": (0,),
            },
            dtypes={
                "mechanics": reduced.values.dtype,
                "history": manifold.impulse.dtype,
                "keys": manifold.route_keys.dtype,
            },
            space_ids={
                "configuration_tangent": dynamics.reduction.coefficient_space.space_id,
                "effort_cotangent": dynamics.reduction.reduced_effort_space.space_id,
                "contact_velocity": participant.contact_velocity_space.space_id,
            },
            topology_ids={
                "native_rod": dynamics.reduction.rod.prepared_id,
                "reduction": dynamics.reduction.prepared_id,
                "capsules": participant.geometry.prepared_id,
                "contact_search": search.prepared_id,
            },
            capacities={
                "coordinates": dynamics.reduction.plan.coordinate_count,
                "nodes": dynamics.reduction.rod.plan.node_count,
                "segments": dynamics.reduction.rod.plan.segment_count,
                "self_contact_routes": search.plan.capacity,
                "plane_contact_routes": search.plan.plane_capacity,
                "manifold_routes": capacity,
                "ccd_iterations": ccd.maximum_iterations,
                "cone_iterations": solver.maximum_iterations,
                "speculative_contact_iterations": speculative_iterations,
            },
            algorithm_facts={
                "integrator": policy.route,
                "integrator_policy_id": policy.policy_id,
                "search_route": int(search.plan.route),
                "ccd_plan_id": ccd.plan_id,
                "cone_solver_id": solver.plan_id,
                "speculative_contact_iterations": speculative_iterations,
                "capability_id": selected_capability,
                "time_policy": "full-requested-interval-or-reject",
                "speculative_relaxation": relaxation,
            },
        )
        plant_id = canonical_fingerprint(
            {
                "kind": "prepared-atomic-reduced-rod-contact-plant",
                "semantic": semantic.semantic_id,
                "numeric": numeric.revision_id,
                "execution": signature.signature_id,
            }
        )
        self.dynamics = dynamics
        self.policy = policy
        self.participant = participant
        self.search = search
        self.ccd = ccd
        self.cone_solver = solver
        self.native_loads = native_loads
        self.material_control = material_control
        self.dynamic_friction = dynamic
        self.static_friction = static
        self.compliance = jnp.asarray(compliance_, dtype=dtype)
        self.material_revision = revision
        self.retention_steps = retention
        self.maximum_speculative_iterations = speculative_iterations
        self.gap_tolerance, self.energy_tolerance, self.conservation_tolerance = (
            tolerances
        )
        self.capability_id = selected_capability
        self.speculative_relaxation = relaxation
        self.initial_state = initial
        self.default_parameters = parameters
        self.state_schema = state_schema
        self.control_schema = None
        self.parameter_schema = parameter_schema
        self.reset_fallback = initial
        self.semantic_provenance = semantic
        self.numeric_revision = numeric
        self.execution_signature = signature
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.plant_id = plant_id

    def bind_parameters(
        self, values: ReducedRodPlantParameters | None = None, /
    ) -> PlantParameters:
        resolved = self.default_parameters if values is None else values
        if not isinstance(resolved, ReducedRodPlantParameters):
            raise TypeError("values must be ReducedRodPlantParameters or None.")
        self.parameter_schema.validate(resolved)
        return PlantParameters(
            resolved, self.parameter_schema.schema_id, self.numeric_revision
        )

    def propose_reset(
        self,
        keys: Array,
        parameters: Any,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del keys
        if case_shape != ():
            raise ValueError("PreparedReducedRodContactPlant has scalar case_ndim=0.")
        if not isinstance(parameters, ReducedRodPlantParameters):
            raise TypeError("parameters must be ReducedRodPlantParameters.")
        evaluation = self.dynamics.evaluate(
            self.initial_state.reduced_state,
            material_state=self.initial_state.material_state,
            material_control=self.material_control,
            time=initial_time,
            step_size=jnp.asarray(
                1.0, dtype=self.initial_state.reduced_state.values.dtype
            ),
            native_loads=self.native_loads,
        )
        positions = self.participant.positions(
            self.initial_state.reduced_state.coefficients
        )
        search = self.search.search(positions)
        transition = self.initial_state.contact_state.update(
            search.witnesses,
            material_revision=self.material_revision,
            retention_steps=self.retention_steps,
        )
        minimum_gap = _minimum_gap(search)
        gap_valid = minimum_gap >= -self.gap_tolerance
        finite = (
            evaluation.finite
            & search.evidence.finite
            & transition.finite
            & jnp.isfinite(minimum_gap)
        )
        valid = (
            evaluation.valid
            & search.successful
            & transition.successful
            & gap_valid
            & finite
        )
        status = _first_failure_status(
            (
                (evaluation.valid, ReducedRodContactPlantStatus.FREE_DYNAMICS_FAILED),
                (
                    search.evidence.failure
                    != int(RodContactSearchFailure.CAPACITY_OVERFLOW),
                    ReducedRodContactPlantStatus.SEARCH_CAPACITY_EXCEEDED,
                ),
                (search.successful, ReducedRodContactPlantStatus.SEARCH_FAILED),
                (
                    transition.successful,
                    ReducedRodContactPlantStatus.MANIFOLD_TRANSITION_FAILED,
                ),
                (gap_valid, ReducedRodContactPlantStatus.FINAL_GAP_INVALID),
                (finite, ReducedRodContactPlantStatus.CORRECTED_INTEGRATION_FAILED),
            )
        )
        candidate = ReducedRodContactPlantState(
            self.initial_state.reduced_state,
            self.initial_state.material_state,
            self.initial_state.actuator_state,
            transition.state,
            self.initial_state.sensor_state,
        )
        evidence = ReducedRodContactPlantResetEvidence(
            evaluation,
            search,
            transition,
            minimum_gap,
            finite,
            valid,
            status,
            self.plant_id,
        )
        return PlantProposal(
            candidate,
            candidate,
            jnp.asarray(True),
            valid,
            status,
            search.evidence.failure,
            evidence,
        )

    def propose_step(
        self,
        context: PlantStepContext,
        source: Any,
        commands: Any,
        parameters: Any,
        keys: Array,
        /,
    ) -> PlantProposal:
        del keys
        if commands is not None:
            raise TypeError(
                "PreparedReducedRodContactPlant is passive and accepts no commands."
            )
        if not isinstance(source, ReducedRodContactPlantState):
            raise TypeError("source must be ReducedRodContactPlantState.")
        if not isinstance(parameters, ReducedRodPlantParameters):
            raise TypeError("parameters must be ReducedRodPlantParameters.")
        integration_source = ReducedRodIntegrationState(
            source.reduced_state,
            source.material_state,
            context.source_time,
            context.step_index,
        )
        free_step = integrate_reduced_rod_step(
            self.dynamics,
            self.policy,
            integration_source,
            context.duration,
            material_control=self.material_control,
            native_loads=self.native_loads,
        )
        free_reduced = free_step.candidate_state.reduced_state
        start_positions = self.participant.positions(source.reduced_state.coefficients)
        free_positions = self.participant.positions(free_reduced.coefficients)
        swept_ccd = self.ccd.evaluate(self.search, start_positions, free_positions)
        full_interval_covered = (
            swept_ccd.evidence.full_step_safe | swept_ccd.evidence.impact_detected
        )
        impact_fraction = jnp.where(
            swept_ccd.evidence.impact_detected & jnp.isfinite(swept_ccd.impact_fraction),
            jnp.clip(swept_ccd.impact_fraction, 0.0, 1.0),
            jnp.asarray(1.0, dtype=context.duration.dtype),
        )
        source_q = source.reduced_state.coefficients
        free_q = free_reduced.coefficients
        event_q = jnp.where(
            swept_ccd.evidence.impact_detected,
            source_q + impact_fraction * (free_q - source_q),
            free_q,
        )
        event_positions = self.participant.positions(event_q)
        event_search = self.search.search(
            event_positions,
            end_positions=free_positions,
        )
        event_manifold = source.contact_state.update(
            event_search.witnesses,
            material_revision=self.material_revision,
            retention_steps=self.retention_steps,
        )
        remaining = jnp.where(
            swept_ccd.evidence.impact_detected,
            (1.0 - impact_fraction) * context.duration,
            jnp.asarray(0.0, dtype=context.duration.dtype),
        )
        zero_effort = self.dynamics.reduction.reduced_effort_space.zeros()
        inverse_mass = self.dynamics.inverse_mass(event_q, zero_effort)
        block = prepare_composite_contact_block(
            self.participant,
            event_q,
            free_reduced.coefficient_velocities,
            inverse_mass.inverse_mass_operator,
            event_manifold.witnesses,
        )
        normal_bias = event_manifold.witnesses.physical_gap / jnp.maximum(
            remaining,
            jnp.finfo(context.duration.dtype).eps,
        )
        response = CompositeContactResponse(
            (block,),
            event_manifold.witnesses,
            dynamic_friction=self.dynamic_friction,
            static_friction=self.static_friction,
            compliance=self.compliance,
            normal_bias=normal_bias,
            solver=self.cone_solver,
        ).solve(initial_impulse=event_manifold.warm_start)
        post_velocity = response.post_velocities[0]
        uncorrected_q = event_q + remaining * post_velocity
        uncorrected_positions = self.participant.positions(uncorrected_q)
        correction_ccd = self.ccd.evaluate(
            self.search,
            event_positions,
            uncorrected_positions,
            supported_initial_plane_route_keys=event_manifold.witnesses.route_keys[
                event_manifold.witnesses.valid
            ],
        )
        correction_covered = correction_ccd.evidence.full_step_safe
        corrected_q = uncorrected_q
        position_correction_iterations = 0
        position_correction_norm = jnp.asarray(0.0, dtype=corrected_q.dtype)
        fixed_point_residual = jnp.asarray(0.0, dtype=corrected_q.dtype)
        authoritative_response_manifold = event_manifold
        safe_remaining = jnp.maximum(
            remaining,
            jnp.finfo(context.duration.dtype).eps,
        )
        speculative_tolerance = self.gap_tolerance + float(
            64.0
            * jnp.finfo(corrected_q.dtype).eps
            * jnp.maximum(1.0, jnp.sqrt(jnp.sum(corrected_q * corrected_q)))
        )
        for correction_index in range(self.maximum_speculative_iterations):
            final_positions = self.participant.positions(corrected_q)
            final_search = self.search.search(final_positions)
            final_minimum_gap = _minimum_gap(final_search)
            if (
                float(final_minimum_gap) >= -self.gap_tolerance
                and position_correction_iterations == 0
            ):
                break
            speculative_manifold = source.contact_state.update(
                final_search.witnesses,
                material_revision=self.material_revision,
                retention_steps=self.retention_steps,
            )
            speculative_inverse_mass = self.dynamics.inverse_mass(
                corrected_q, zero_effort
            )
            speculative_block = prepare_composite_contact_block(
                self.participant,
                corrected_q,
                free_reduced.coefficient_velocities,
                speculative_inverse_mass.inverse_mass_operator,
                speculative_manifold.witnesses,
            )
            source_offset_velocity = (event_q - corrected_q) / safe_remaining
            source_offset_contact_velocity = speculative_block.velocity_operator.mv(
                source_offset_velocity
            )
            speculative_bias = (
                speculative_manifold.witnesses.physical_gap / safe_remaining
                + source_offset_contact_velocity[:, 0]
            )
            speculative_response = CompositeContactResponse(
                (speculative_block,),
                speculative_manifold.witnesses,
                dynamic_friction=self.dynamic_friction,
                static_friction=self.static_friction,
                compliance=self.compliance,
                normal_bias=speculative_bias,
                solver=self.cone_solver,
            ).solve(
                initial_impulse=_remap_speculative_impulse(
                    authoritative_response_manifold.witnesses,
                    response.impulse,
                    speculative_manifold.witnesses,
                    speculative_manifold.warm_start,
                )
            )
            candidate_velocity = speculative_response.post_velocities[0]
            candidate_q = event_q + remaining * candidate_velocity
            fixed_point_residual = jnp.sqrt(jnp.sum((candidate_q - corrected_q) ** 2))
            position_correction_norm = position_correction_norm + fixed_point_residual
            position_correction_iterations = correction_index + 1
            response = speculative_response
            inverse_mass = speculative_inverse_mass
            post_velocity = candidate_velocity
            authoritative_response_manifold = speculative_manifold
            if not bool(speculative_response.successful):
                break
            if (
                float(final_minimum_gap) >= -self.gap_tolerance
                and float(fixed_point_residual) <= speculative_tolerance
            ):
                break
            corrected_q = corrected_q + self.speculative_relaxation * (
                candidate_q - corrected_q
            )
        final_positions = self.participant.positions(corrected_q)
        final_search = self.search.search(final_positions)
        final_minimum_gap = _minimum_gap(final_search)
        position_correction_successful = (
            response.successful
            & (final_minimum_gap >= -self.gap_tolerance)
            & (fixed_point_residual <= speculative_tolerance)
        )
        correction_ccd = self.ccd.evaluate(
            self.search,
            event_positions,
            final_positions,
            supported_initial_plane_route_keys=event_manifold.witnesses.route_keys[
                event_manifold.witnesses.valid
            ],
        )
        correction_covered = correction_ccd.evidence.full_step_safe
        corrected_reduced = ReducedRodState(corrected_q, post_velocity)
        corrected_evaluation = self.dynamics.evaluate(
            corrected_reduced,
            source_state=source.reduced_state,
            material_state=source.material_state,
            material_control=self.material_control,
            time=context.target_time,
            step_size=context.duration,
            native_loads=self.native_loads,
        )
        response_manifold = authoritative_response_manifold
        candidate_event_contact = response_manifold.state.record_response(
            response.route_keys,
            response.candidate_impulse,
            response.sticking,
            response.slip_velocity,
            step_size=remaining,
        )
        accepted_event_contact = response_manifold.state.commit(
            response_manifold.witnesses,
            response,
            step_size=remaining,
        )
        candidate_final_manifold = candidate_event_contact.update(
            final_search.witnesses,
            material_revision=self.material_revision,
            retention_steps=self.retention_steps,
        )
        accepted_final_manifold = accepted_event_contact.update(
            final_search.witnesses,
            material_revision=self.material_revision,
            retention_steps=self.retention_steps,
        )
        candidate = ReducedRodContactPlantState(
            corrected_reduced,
            corrected_evaluation.candidate_material_state,
            source.actuator_state,
            candidate_final_manifold.state,
            source.sensor_state,
        )
        accepted_payload = ReducedRodContactPlantState(
            corrected_reduced,
            corrected_evaluation.candidate_material_state,
            source.actuator_state,
            accepted_final_manifold.state,
            source.sensor_state,
        )
        final_minimum_gap = _minimum_gap(final_search)
        gap_valid = final_minimum_gap >= -self.gap_tolerance
        energy = self._energy_evidence(source, free_step, corrected_evaluation, response)
        conservation = self._conservation_evidence(inverse_mass, response)
        mass_valid = inverse_mass.solve_evidence.valid
        status = _first_failure_status(
            (
                (free_step.successful, ReducedRodContactPlantStatus.FREE_DYNAMICS_FAILED),
                (
                    swept_ccd.search.evidence.failure
                    != int(RodContactSearchFailure.CAPACITY_OVERFLOW),
                    ReducedRodContactPlantStatus.SEARCH_CAPACITY_EXCEEDED,
                ),
                (swept_ccd.search.successful, ReducedRodContactPlantStatus.SEARCH_FAILED),
                (swept_ccd.successful, ReducedRodContactPlantStatus.CCD_FAILED),
                (
                    full_interval_covered,
                    ReducedRodContactPlantStatus.CCD_SAFE_PREFIX_ONLY,
                ),
                (
                    event_search.successful,
                    ReducedRodContactPlantStatus.EVENT_SEARCH_FAILED,
                ),
                (
                    event_manifold.successful,
                    ReducedRodContactPlantStatus.MANIFOLD_TRANSITION_FAILED,
                ),
                (mass_valid, ReducedRodContactPlantStatus.MASS_RESPONSE_FAILED),
                (response.successful, ReducedRodContactPlantStatus.RESPONSE_SOLVE_FAILED),
                (
                    position_correction_successful,
                    ReducedRodContactPlantStatus.CORRECTED_INTEGRATION_FAILED,
                ),
                (
                    corrected_evaluation.valid,
                    ReducedRodContactPlantStatus.CORRECTED_INTEGRATION_FAILED,
                ),
                (correction_covered, ReducedRodContactPlantStatus.CORRECTION_CCD_FAILED),
                (
                    final_search.successful,
                    ReducedRodContactPlantStatus.FINAL_SEARCH_FAILED,
                ),
                (
                    candidate_final_manifold.successful
                    & accepted_final_manifold.successful,
                    ReducedRodContactPlantStatus.FINAL_MANIFOLD_FAILED,
                ),
                (gap_valid, ReducedRodContactPlantStatus.FINAL_GAP_INVALID),
                (energy.valid, ReducedRodContactPlantStatus.ENERGY_INVALID),
                (conservation.valid, ReducedRodContactPlantStatus.CONSERVATION_INVALID),
            )
        )
        finite = (
            free_step.evidence.finite
            & swept_ccd.evidence.finite
            & event_search.evidence.finite
            & event_manifold.finite
            & inverse_mass.solve_evidence.finite
            & response.evidence.finite
            & corrected_evaluation.finite
            & position_correction_successful
            & correction_ccd.evidence.finite
            & final_search.evidence.finite
            & candidate_final_manifold.finite
            & accepted_final_manifold.finite
            & jnp.isfinite(final_minimum_gap)
            & energy.finite
            & conservation.finite
        )
        successful = (status == int(ReducedRodContactPlantStatus.SUCCESS)) & finite
        backend_status = jnp.where(
            response.successful,
            free_step.backend_status,
            response.evidence.iterations,
        ).astype(jnp.int32)
        evidence = ReducedRodContactPlantStepEvidence(
            free_step,
            swept_ccd,
            event_search,
            event_manifold,
            response,
            corrected_evaluation,
            correction_ccd,
            final_search,
            candidate_final_manifold,
            accepted_final_manifold,
            final_minimum_gap,
            jnp.asarray(position_correction_iterations, dtype=jnp.int32),
            fixed_point_residual,
            position_correction_norm,
            position_correction_successful,
            energy,
            conservation,
            full_interval_covered,
            finite,
            successful,
            status,
            backend_status,
            self.capability_id,
            self.plant_id,
        )
        return PlantProposal(
            candidate,
            accepted_payload,
            jnp.asarray(True),
            successful,
            status,
            backend_status,
            evidence,
        )

    def _energy_evidence(
        self,
        source: ReducedRodContactPlantState,
        free_step: ReducedRodStepResult,
        corrected: ReducedRodDynamicsEvaluation,
        response: CompositeContactResult,
        /,
    ) -> ReducedRodContactEnergyEvidence:
        source_energy = (
            free_step.evidence.source_evaluation.energy.total_mechanical_energy
        )
        free_energy = (
            free_step.evidence.candidate_evaluation.energy.total_mechanical_energy
        )
        final_energy = corrected.energy.total_mechanical_energy
        contact_change = final_energy - free_energy
        friction = -jnp.sum(response.impulse[:, 1:] * response.slip_velocity)
        scale = jnp.maximum(
            1.0,
            jnp.maximum(
                jnp.abs(source_energy),
                jnp.maximum(jnp.abs(free_energy), jnp.abs(final_energy)),
            ),
        )
        tolerance = self.energy_tolerance * scale
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        source_energy,
                        free_energy,
                        final_energy,
                        contact_change,
                        friction,
                        scale,
                    )
                )
            )
        )
        nonenergizing = contact_change <= tolerance
        friction_dissipative = friction >= -tolerance
        valid = finite & nonenergizing & friction_dissipative
        del source
        return ReducedRodContactEnergyEvidence(
            source_energy,
            free_energy,
            final_energy,
            contact_change,
            friction,
            scale,
            finite,
            nonenergizing,
            friction_dissipative,
            valid,
        )

    def _conservation_evidence(
        self,
        inverse_mass: Any,
        response: CompositeContactResult,
        /,
    ) -> ReducedRodContactConservationEvidence:
        impulse = response.generalized_impulses[0]
        update = response.velocity_updates[0]
        reconstructed = inverse_mass.mass.operator.mv(update)
        residual = jnp.sqrt(jnp.sum((reconstructed - impulse) ** 2))
        impulse_norm = jnp.sqrt(jnp.sum(impulse**2))
        scale = jnp.maximum(1.0, impulse_norm)
        duality_residual = response.evidence.duality_residual
        duality_scale = response.evidence.duality_scale
        finite = jnp.all(
            jnp.isfinite(jnp.stack((residual, scale, duality_residual, duality_scale)))
        )
        response_valid = residual <= self.conservation_tolerance * scale
        duality_valid = response.evidence.duality_valid & (
            duality_residual
            <= self.conservation_tolerance * jnp.maximum(1.0, duality_scale)
        )
        return ReducedRodContactConservationEvidence(
            residual,
            scale,
            duality_residual,
            duality_scale,
            finite,
            response_valid,
            duality_valid,
            finite & response_valid & duality_valid,
        )


def prepare_reduced_rod_contact_plant(
    dynamics: PreparedReducedRodDynamics,
    policy: ReducedRodIntegratorPolicy,
    participant: ReducedRodCapsuleContactParticipant,
    search: PreparedRodContactSearch,
    ccd: RodContactCCDPlan,
    /,
    **kwargs: Any,
) -> PreparedReducedRodContactPlant:
    """Prepare the atomic fixed-base finite-radius rod contact profile."""

    return PreparedReducedRodContactPlant(
        dynamics, policy, participant, search, ccd, **kwargs
    )


__all__ = [
    "FRICTIONLESS_ROD_CONTACT_CAPABILITY",
    "ISOTROPIC_COULOMB_ROD_CONTACT_CAPABILITY",
    "PreparedReducedRodContactPlant",
    "ReducedRodContactConservationEvidence",
    "ReducedRodContactEnergyEvidence",
    "ReducedRodContactPlantResetEvidence",
    "ReducedRodContactPlantState",
    "ReducedRodContactPlantStatus",
    "ReducedRodContactPlantStepEvidence",
    "RodContactCapabilityId",
    "prepare_reduced_rod_contact_plant",
]
