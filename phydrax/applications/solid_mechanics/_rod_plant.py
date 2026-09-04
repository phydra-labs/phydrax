#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
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
    PlantRuntimeState,
    PlantStepContext,
    SecondOrderDifferentialSystem,
)
from ...linalg import AbstractLinearOperator
from ._rod_loads import RodLoadLedger
from ._rod_materials import (
    PreparedKelvinVoigtRodMaterial,
    PreparedLinearElasticRodMaterial,
)
from ._rod_reduced_dynamics import (
    PreparedReducedRodDynamics,
    ReducedRodDynamicsEvaluation,
    ReducedRodMassResult,
    ReducedRodMaterialControl,
    ReducedRodMaterialState,
    ReducedRodSolveEvidence,
)
from ._rod_reduced_integrators import (
    integrate_reduced_rod_step,
    ReducedRodImplicitMidpoint,
    ReducedRodIntegrationState,
    ReducedRodIntegratorPolicy,
    ReducedRodSemiImplicitVelocityEuler,
    ReducedRodStepResult,
    ReducedRodStepStatus,
)
from ._rod_reduction import ReducedRodState


def _empty_state(value: ArrayLike, owner: str, /) -> Array:
    array = jnp.asarray(value)
    if array.shape != (0,):
        raise ValueError(f"{owner} must have shape (0,) for the passive plant.")
    if not jnp.issubdtype(array.dtype, jnp.inexact) or jnp.iscomplexobj(array):
        raise TypeError(f"{owner} must use one real inexact dtype.")
    return array


class ReducedRodPassiveActuatorState(StrictModule):
    """Typed zero-width actuator state for the strictly passive profile."""

    values: Array

    def __init__(self, values: ArrayLike, /):
        self.values = _empty_state(values, "Passive actuator state")


class ReducedRodPassiveContactState(StrictModule):
    """Typed zero-width contact state for the contact-free passive profile."""

    values: Array

    def __init__(self, values: ArrayLike, /):
        self.values = _empty_state(values, "Passive contact state")


class ReducedRodPassiveSensorState(StrictModule):
    """Typed zero-width sensor state; observations are pure reconstructions."""

    values: Array

    def __init__(self, values: ArrayLike, /):
        self.values = _empty_state(values, "Passive sensor state")


class ReducedRodPlantState(StrictModule):
    """Complete passive reduced-rod payload owned by one plant transaction."""

    reduced_state: ReducedRodState
    material_state: ReducedRodMaterialState
    actuator_state: ReducedRodPassiveActuatorState
    contact_state: ReducedRodPassiveContactState
    sensor_state: ReducedRodPassiveSensorState

    def __init__(
        self,
        reduced_state: ReducedRodState,
        material_state: ReducedRodMaterialState,
        actuator_state: ReducedRodPassiveActuatorState,
        contact_state: ReducedRodPassiveContactState,
        sensor_state: ReducedRodPassiveSensorState,
        /,
    ):
        if not isinstance(reduced_state, ReducedRodState):
            raise TypeError("reduced_state must be ReducedRodState.")
        if not isinstance(material_state, ReducedRodMaterialState):
            raise TypeError("material_state must be ReducedRodMaterialState.")
        if not isinstance(actuator_state, ReducedRodPassiveActuatorState):
            raise TypeError("actuator_state must be ReducedRodPassiveActuatorState.")
        if not isinstance(contact_state, ReducedRodPassiveContactState):
            raise TypeError("contact_state must be ReducedRodPassiveContactState.")
        if not isinstance(sensor_state, ReducedRodPassiveSensorState):
            raise TypeError("sensor_state must be ReducedRodPassiveSensorState.")
        self.reduced_state = reduced_state
        self.material_state = material_state
        self.actuator_state = actuator_state
        self.contact_state = contact_state
        self.sensor_state = sensor_state


class ReducedRodPlantParameters(StrictModule):
    """Typed zero-width parameters for one fully prepared passive revision."""

    values: Array

    def __init__(self, values: ArrayLike, /):
        self.values = _empty_state(values, "Passive plant parameters")


class ReducedRodPlantResetEvidence(StrictModule):
    """Mechanics certification retained for one reset proposal."""

    evaluation: ReducedRodDynamicsEvaluation
    finite: Array
    valid: Array
    status: Array
    plant_id: str = eqx.field(static=True)


class ReducedRodPlantEvidence(StrictModule):
    """Full candidate/accepted integration transaction retained by the plant."""

    integration_result: ReducedRodStepResult
    finite: Array
    valid: Array
    plant_id: str = eqx.field(static=True)


class ReducedRodMassResponseRevision(StrictModule):
    """Certified inverse-mass response bound to one accepted mechanical state."""

    configuration: Array
    free_velocity: Array
    time: Array
    step_index: Array
    inverse_mass_operator: AbstractLinearOperator
    mass: ReducedRodMassResult
    solve_evidence: ReducedRodSolveEvidence
    finite: Array
    valid: Array
    plant_id: str = eqx.field(static=True)
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)
    revision_id: str = eqx.field(static=True)

    def apply_impulse(self, impulse: ArrayLike, /) -> Array:
        """Map one true reduced tangent covector to a tangent velocity increment."""
        return self.inverse_mass_operator.mv(impulse)

    def velocity_after_impulse(self, impulse: ArrayLike, /) -> Array:
        """Return the free velocity plus the certified impulse response."""
        return self.free_velocity + self.apply_impulse(impulse)

    def is_current(self, state: PlantRuntimeState, /) -> Array:
        """Check the complete mechanical revision and reject foreign identities."""
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("state must be PlantRuntimeState.")
        if (
            state.semantic_provenance_id != self.semantic_provenance_id
            or state.numeric_revision_id != self.numeric_revision_id
            or state.state_schema_id != self.state_schema_id
            or state.execution_signature_id != self.execution_signature_id
        ):
            raise ValueError(
                "Mass response and runtime state use different plant identities."
            )
        payload = state.payload
        if not isinstance(payload, ReducedRodPlantState):
            raise TypeError("Runtime payload must be ReducedRodPlantState.")
        return (
            (state.time == self.time)
            & (state.step_index == self.step_index)
            & jnp.all(payload.reduced_state.coefficients == self.configuration)
            & jnp.all(payload.reduced_state.coefficient_velocities == self.free_velocity)
        )


class ReducedRodDifferentialResidual(StrictModule, NonTrainableState):
    """Smooth stateless contact-free second-order residual adapter."""

    dynamics: PreparedReducedRodDynamics
    material_state: ReducedRodMaterialState
    material_control: ReducedRodMaterialControl
    native_loads: RodLoadLedger | None
    adapter_id: str = eqx.field(static=True)

    def __call__(
        self,
        time: Array,
        configuration: Array,
        velocity: Array,
        acceleration: Array,
        arguments: Any,
        /,
    ) -> Array:
        if arguments is not None:
            raise TypeError(
                "Reduced rod differential residual has no runtime arguments; "
                "prepare a new numeric plant revision instead."
            )
        state = ReducedRodState(configuration, velocity)
        result = self.dynamics.inverse_dynamics(
            state,
            acceleration,
            material_state=self.material_state,
            material_control=self.material_control,
            time=time,
            step_size=jnp.asarray(1.0, dtype=configuration.dtype),
            native_loads=self.native_loads,
        )
        return self.dynamics.reduction.reduced_effort_space.validate(result.residual)


class PreparedReducedRodPlant(AbstractDiscretePlant, NonTrainableState):
    """Passive fixed-base reduced rod with transactional PlantCore stepping."""

    dynamics: PreparedReducedRodDynamics
    policy: ReducedRodIntegratorPolicy
    native_loads: RodLoadLedger | None
    material_control: ReducedRodMaterialControl
    initial_state: ReducedRodPlantState
    default_parameters: ReducedRodPlantParameters
    state_schema: ArrayPyTreeSchema
    control_schema: None
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: ReducedRodPlantState
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
        /,
        *,
        initial_reduced_state: ReducedRodState | None = None,
        initial_material_state: ReducedRodMaterialState | None = None,
        native_loads: RodLoadLedger | None = None,
    ):
        if not isinstance(dynamics, PreparedReducedRodDynamics):
            raise TypeError("dynamics must be PreparedReducedRodDynamics.")

        if not isinstance(
            policy, (ReducedRodSemiImplicitVelocityEuler, ReducedRodImplicitMidpoint)
        ):
            raise TypeError("policy must select one reduced rod integrator route.")
        if native_loads is not None:
            if not isinstance(native_loads, RodLoadLedger):
                raise TypeError("native_loads must be RodLoadLedger or None.")
            native_loads._validate_rod(dynamics.reduction.rod)
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
        material_template = dynamics.initialize_material_state()
        for name, value, template in (
            (
                "stretch_shear_history",
                material.stretch_shear_history,
                material_template.stretch_shear_history,
            ),
            (
                "bend_twist_history",
                material.bend_twist_history,
                material_template.bend_twist_history,
            ),
        ):
            if value.shape != template.shape or value.dtype != template.dtype:
                raise ValueError(f"Initial {name} does not match the prepared material.")
        if (
            not bool(jnp.all(jnp.isfinite(reduced.values)))
            or not bool(jnp.all(jnp.isfinite(material.stretch_shear_history)))
            or not bool(jnp.all(jnp.isfinite(material.bend_twist_history)))
        ):
            raise ValueError("The prepared reset fallback must be finite.")
        dtype = reduced.values.dtype
        empty = jnp.zeros((0,), dtype=dtype)
        initial = ReducedRodPlantState(
            reduced,
            material,
            ReducedRodPassiveActuatorState(empty),
            ReducedRodPassiveContactState(empty),
            ReducedRodPassiveSensorState(empty),
        )
        material_control = dynamics.initialize_material_control()
        parameters = ReducedRodPlantParameters(empty)
        state_schema = ArrayPyTreeSchema.from_tree(initial, case_ndim=0)
        parameter_schema = ArrayPyTreeSchema.from_tree(parameters, case_ndim=0)
        loads_semantics = (
            None
            if native_loads is None
            else {
                "source_ids": native_loads.source_ids,
                "channel_names": native_loads.channel_names,
                "force_frame": native_loads.force_frame,
                "moment_frame": native_loads.moment_frame,
                "force_unit": native_loads.force_unit,
                "moment_unit": native_loads.moment_unit,
            }
        )
        semantic = SemanticProvenance(
            {
                "kind": "prepared-passive-reduced-rod-plant",
                "dimension": dynamics.reduction.plan.dimension,
                "base_policy": dynamics.reduction.plan.base_policy,
                "coordinate_count": dynamics.reduction.plan.coordinate_count,
                "integrator_route": policy.route,
                "mass_solver": dynamics.plan.solver,
                "stretch_shear_material": (
                    f"{type(dynamics.stretch_shear_material).__module__}."
                    f"{type(dynamics.stretch_shear_material).__qualname__}"
                ),
                "bend_twist_material": (
                    f"{type(dynamics.bend_twist_material).__module__}."
                    f"{type(dynamics.bend_twist_material).__qualname__}"
                ),
                "actuation": "passive-zero-width",
                "contact": "none-zero-width",
                "sensor_state": "zero-width",
                "native_loads": loads_semantics,
                "state_schema": state_schema.content_id,
                "parameter_schema": parameter_schema.content_id,
                "control_schema": None,
            }
        )
        numeric = NumericRevision(
            semantic,
            {
                "dynamics_id": dynamics.dynamics_id,
                "native_load_ledger_id": (
                    None if native_loads is None else native_loads.ledger_id
                ),
                "material_control": material_control,
                "initial_state": initial,
            },
        )
        stretch_sites = dynamics.stretch_shear_material.workset.site_count
        bend_sites = dynamics.bend_twist_material.workset.site_count
        signature = ExecutableSignature(
            shapes={
                "reduced_state": reduced.values.shape,
                "stretch_shear_history": material.stretch_shear_history.shape,
                "bend_twist_history": material.bend_twist_history.shape,
                "stretch_intrinsic_strain": material_control.stretch_shear_control.intrinsic_strain.shape,
                "stretch_intrinsic_strain_rate": material_control.stretch_shear_control.intrinsic_strain_rate.shape,
                "stretch_stiffness": material_control.stretch_shear_control.stiffness.shape,
                "stretch_stiffness_rate": material_control.stretch_shear_control.stiffness_rate.shape,
                "bend_intrinsic_strain": material_control.bend_twist_control.intrinsic_strain.shape,
                "bend_intrinsic_strain_rate": material_control.bend_twist_control.intrinsic_strain_rate.shape,
                "bend_stiffness": material_control.bend_twist_control.stiffness.shape,
                "bend_stiffness_rate": material_control.bend_twist_control.stiffness_rate.shape,
                "passive_actuator_state": (0,),
                "passive_contact_state": (0,),
                "passive_sensor_state": (0,),
                "passive_parameters": (0,),
            },
            dtypes={
                "state": reduced.values.dtype,
                "material": material.stretch_shear_history.dtype,
                "parameters": parameters.values.dtype,
            },
            space_ids={
                "configuration_tangent": dynamics.reduction.coefficient_space.space_id,
                "velocity_tangent": dynamics.reduction.coefficient_space.space_id,
                "effort_cotangent": dynamics.reduction.reduced_effort_space.space_id,
                "native_velocity": dynamics.reduction.native_velocity_space.space_id,
                "native_effort": dynamics.reduction.native_effort_space.space_id,
            },
            topology_ids={
                "native_rod": dynamics.reduction.rod.prepared_id,
                "reduction": dynamics.reduction.prepared_id,
                "basis": dynamics.reduction.basis.prepared_id,
            },
            capacities={
                "coordinates": dynamics.reduction.plan.coordinate_count,
                "nodes": dynamics.reduction.rod.plan.node_count,
                "segments": dynamics.reduction.rod.plan.segment_count,
                "stretch_shear_sites": stretch_sites,
                "bend_twist_sites": bend_sites,
            },
            algorithm_facts={
                "integrator_route": policy.route,
                "integrator_policy_id": policy.policy_id,
                "mass_solver": dynamics.plan.solver,
                "contact": "none",
                "actuation": "passive",
            },
        )
        plant_id = canonical_fingerprint(
            {
                "kind": "prepared-passive-reduced-rod-plant",
                "semantic": semantic.semantic_id,
                "numeric_revision": numeric.revision_id,
                "execution_signature": signature.signature_id,
                "dynamics": dynamics.dynamics_id,
                "policy": policy.policy_id,
            }
        )
        self.dynamics = dynamics
        self.policy = policy
        self.native_loads = native_loads
        self.material_control = material_control
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
        """Bind schema-exact passive parameters to this numeric revision."""
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
            raise ValueError("PreparedReducedRodPlant has scalar case_ndim=0.")
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
        valid = evaluation.valid
        status = jnp.where(
            valid,
            jnp.asarray(ReducedRodStepStatus.SUCCESS, dtype=jnp.int32),
            jnp.asarray(ReducedRodStepStatus.SOURCE_INVALID, dtype=jnp.int32),
        )
        evidence = ReducedRodPlantResetEvidence(
            evaluation, evaluation.finite, valid, status, self.plant_id
        )
        return PlantProposal(
            self.initial_state,
            self.initial_state,
            jnp.asarray(True),
            valid,
            status,
            jnp.asarray(0, dtype=jnp.int32),
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
            raise TypeError("PreparedReducedRodPlant is passive and accepts no commands.")
        if not isinstance(source, ReducedRodPlantState):
            raise TypeError("source must be ReducedRodPlantState.")
        if not isinstance(parameters, ReducedRodPlantParameters):
            raise TypeError("parameters must be ReducedRodPlantParameters.")
        integration_source = ReducedRodIntegrationState(
            source.reduced_state,
            source.material_state,
            context.source_time,
            context.step_index,
        )
        result = integrate_reduced_rod_step(
            self.dynamics,
            self.policy,
            integration_source,
            context.duration,
            material_control=self.material_control,
            native_loads=self.native_loads,
        )
        candidate = self._payload_from_integration(source, result.candidate_state)
        accepted = self._payload_from_integration(source, result.accepted_state)
        evidence = ReducedRodPlantEvidence(
            result, result.evidence.finite, result.successful, self.plant_id
        )
        return PlantProposal(
            candidate,
            accepted,
            result.attempted,
            result.successful,
            result.status,
            result.backend_status,
            evidence,
        )

    @staticmethod
    def _payload_from_integration(
        source: ReducedRodPlantState,
        integration: ReducedRodIntegrationState,
        /,
    ) -> ReducedRodPlantState:
        return ReducedRodPlantState(
            integration.reduced_state,
            integration.material_state,
            source.actuator_state,
            source.contact_state,
            source.sensor_state,
        )

    def mass_response(
        self, state: PlantRuntimeState, /
    ) -> ReducedRodMassResponseRevision:
        """Prepare the certified accepted-q inverse mass without route fallback."""
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("state must be PlantRuntimeState.")
        if (
            state.semantic_provenance_id != self.semantic_provenance.semantic_id
            or state.numeric_revision_id != self.numeric_revision.revision_id
            or state.state_schema_id != self.state_schema.schema_id
            or state.execution_signature_id != self.execution_signature.signature_id
        ):
            raise ValueError("Runtime state identities do not belong to this plant.")
        if self.state_schema.validate(state.payload) != ():
            raise ValueError("Reduced rod mass response requires one scalar plant case.")
        payload = state.payload
        if not isinstance(payload, ReducedRodPlantState):
            raise TypeError("Runtime payload must be ReducedRodPlantState.")
        effort_zero = self.dynamics.reduction.reduced_effort_space.zeros()
        inverse = self.dynamics.inverse_mass(
            payload.reduced_state.coefficients, effort_zero
        )
        finite = inverse.solve_evidence.finite & jnp.all(
            jnp.isfinite(payload.reduced_state.values)
        )
        valid = inverse.solve_evidence.valid & finite
        revision_id = canonical_fingerprint(
            {
                "kind": "reduced-rod-mass-response-revision",
                "plant": self.plant_id,
                "numeric_revision": self.numeric_revision.revision_id,
                "inverse_operator": inverse.inverse_mass_operator.operator_id,
            }
        )
        return ReducedRodMassResponseRevision(
            payload.reduced_state.coefficients,
            payload.reduced_state.coefficient_velocities,
            state.time,
            state.step_index,
            inverse.inverse_mass_operator,
            inverse.mass,
            inverse.solve_evidence,
            finite,
            valid,
            self.plant_id,
            self.semantic_provenance.semantic_id,
            self.numeric_revision.revision_id,
            self.state_schema.schema_id,
            self.execution_signature.signature_id,
            revision_id,
        )

    def as_second_order_differential_system(self, /) -> SecondOrderDifferentialSystem:
        """Expose only the prepared smooth, stateless, contact-free law."""
        supported = (
            PreparedLinearElasticRodMaterial,
            PreparedKelvinVoigtRodMaterial,
        )
        materials = (
            self.dynamics.stretch_shear_material,
            self.dynamics.bend_twist_material,
        )
        if not all(isinstance(material, supported) for material in materials):
            raise TypeError(
                "Differential adaptation requires smooth linear or Kelvin-Voigt laws."
            )
        if any(
            material.history_size != 0 or material.control_size != 0
            for material in materials
        ):
            raise ValueError(
                "Differential adaptation requires stateless, uncontrolled material laws."
            )
        controls = (
            self.material_control.stretch_shear_control,
            self.material_control.bend_twist_control,
        )
        if any(
            control.intrinsic_owner_id is not None
            or control.stiffness_owner_id is not None
            for control in controls
        ):
            raise ValueError(
                "Differential adaptation rejects actuator-owned constitutive controls."
            )
        residual = ReducedRodDifferentialResidual(
            self.dynamics,
            self.dynamics.initialize_material_state(),
            self.material_control,
            self.native_loads,
            canonical_fingerprint(
                {
                    "kind": "smooth-stateless-contact-free-reduced-rod-residual",
                    "plant": self.plant_id,
                    "numeric_revision": self.numeric_revision.revision_id,
                }
            ),
        )
        count = self.dynamics.reduction.plan.coordinate_count
        return SecondOrderDifferentialSystem(
            residual,
            state_shape=(count,),
            system_id=residual.adapter_id,
        )


def prepare_reduced_rod_plant(
    dynamics: PreparedReducedRodDynamics,
    policy: ReducedRodIntegratorPolicy,
    /,
    **kwargs: Any,
) -> PreparedReducedRodPlant:
    """Prepare the strictly passive transactional reduced-rod plant."""
    return PreparedReducedRodPlant(dynamics, policy, **kwargs)


def reduced_rod_differential_system(
    plant: PreparedReducedRodPlant, /
) -> SecondOrderDifferentialSystem:
    """Return the explicitly restricted smooth stateless differential adapter."""
    if not isinstance(plant, PreparedReducedRodPlant):
        raise TypeError("plant must be PreparedReducedRodPlant.")
    return plant.as_second_order_differential_system()


__all__ = [
    "prepare_reduced_rod_plant",
    "reduced_rod_differential_system",
    "PreparedReducedRodPlant",
    "ReducedRodDifferentialResidual",
    "ReducedRodMassResponseRevision",
    "ReducedRodPassiveActuatorState",
    "ReducedRodPassiveContactState",
    "ReducedRodPassiveSensorState",
    "ReducedRodPlantParameters",
    "ReducedRodPlantResetEvidence",
    "ReducedRodPlantState",
    "ReducedRodPlantEvidence",
]
