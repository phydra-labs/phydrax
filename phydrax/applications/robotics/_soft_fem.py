#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._array_tree import ArrayPyTreeSchema
from ..._fingerprint import canonical_fingerprint
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...dynamics._layout import StateLayout
from ...dynamics._plant import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantRuntimeState,
    PlantStepContext,
)
from ...dynamics._plant_codec import ControlVectorCodec, PlantStateVectorCodec
from ...equations._finite_element_material import MaterialState, MaterialTransaction
from ...linalg import ArraySpace
from ..solid_mechanics._fem_dynamics import (
    FiniteElementDynamicsPlan,
    FiniteElementDynamicsResult,
    FiniteElementDynamicsState,
    prepare_finite_element_dynamics_step,
    solve_finite_element_dynamics_step,
)
from ._backend import (
    ROBOTICS_OPERATIONS,
    RoboticsBackendProfile,
    RoboticsOperationCapability,
    RoboticsOperationStatus,
)


FEM_LINEAR_ELASTICITY_CAPABILITY_ID = "phydrax.soft-fem.constitutive.linear-elasticity.v1"
FEM_HYPERELASTICITY_CAPABILITY_ID = "phydrax.soft-fem.constitutive.hyperelasticity.v1"
FEM_VISCOELASTICITY_CAPABILITY_ID = "phydrax.soft-fem.constitutive.viscoelasticity.v1"
FEM_PRESSURE_ACTUATION_CAPABILITY_ID = "phydrax.soft-fem.actuation.region-pressure.v1"
FEM_FIBER_ACTUATION_CAPABILITY_ID = "phydrax.soft-fem.actuation.region-fiber.v1"
FEM_BODY_FORCE_ACTUATION_CAPABILITY_ID = "phydrax.soft-fem.actuation.region-body-force.v1"
FEM_REGION_DISPLACEMENT_SENSOR_CAPABILITY_ID = (
    "phydrax.soft-fem.sensor.region-displacement.v1"
)
FEM_REGION_FORCE_SENSOR_CAPABILITY_ID = "phydrax.soft-fem.sensor.region-force.v1"
FEM_EXACT_STATE_CODEC_CAPABILITY_ID = "phydrax.soft-fem.codec.complete-state-exact.v1"
FEM_EXACT_CONTROL_CODEC_CAPABILITY_ID = "phydrax.soft-fem.codec.control-exact.v1"
FEM_ATOMIC_REPLAY_CAPABILITY_ID = "phydrax.soft-fem.transaction.atomic-replay.v1"

FEM_REMESH_CAPABILITY_ID = "phydrax.soft-fem.topology.remesh.v1"
FEM_FRACTURE_CAPABILITY_ID = "phydrax.soft-fem.topology.fracture.v1"
FEM_CONTACT_CAPABILITY_ID = "phydrax.soft-fem.contact.v1"

FEMConstitutiveCapability: TypeAlias = Literal[
    "phydrax.soft-fem.constitutive.linear-elasticity.v1",
    "phydrax.soft-fem.constitutive.hyperelasticity.v1",
    "phydrax.soft-fem.constitutive.viscoelasticity.v1",
]
_CONSTITUTIVE_CAPABILITIES = frozenset(
    (
        FEM_LINEAR_ELASTICITY_CAPABILITY_ID,
        FEM_HYPERELASTICITY_CAPABILITY_ID,
        FEM_VISCOELASTICITY_CAPABILITY_ID,
    )
)
_UNSUPPORTED_CAPABILITIES = frozenset(
    (FEM_REMESH_CAPABILITY_ID, FEM_FRACTURE_CAPABILITY_ID, FEM_CONTACT_CAPABILITY_ID)
)


def _identifier(value: str, owner: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{owner} must be non-empty.")
    return identifier


def _identifiers(values: Sequence[str], owner: str, /) -> tuple[str, ...]:
    result = tuple(_identifier(value, owner) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{owner} values must be unique.")
    return result


def _common_dtype(arrays: Sequence[Array], owner: str, /) -> np.dtype:
    if not arrays:
        raise ValueError(f"{owner} requires at least one array.")
    dtype = np.dtype(arrays[0].dtype)
    if dtype.kind not in "fc" or any(np.dtype(value.dtype) != dtype for value in arrays):
        raise TypeError(f"{owner} arrays must have one common inexact dtype.")
    return dtype


def _unsupported(operation: str, reason: str, /) -> RoboticsOperationCapability:
    return RoboticsOperationCapability(
        operation,  # type: ignore[arg-type]
        supported=False,
        implementation="FEMSoftPlant",
        reason=reason,
    )


def _profile(
    dtype: np.dtype, differentiable: bool, devices: Sequence[str], /
) -> RoboticsBackendProfile:
    supported = {
        "step": "FEMSoftPlant.propose_step",
        "sensors": "FEMSoftPlant.observe",
    }
    operations = []
    for operation in ROBOTICS_OPERATIONS:
        if operation in supported:
            operations.append(
                RoboticsOperationCapability(
                    operation,
                    supported=True,
                    implementation=supported[operation],
                    devices=devices,
                    dtypes=(dtype.name,),
                    differentiability="conditional" if differentiable else "none",
                    solvers=("implicit-newmark",) if operation == "step" else (),
                )
            )
        else:
            reason = (
                "contact is not part of the fixed-mesh contact-free FEM profile"
                if operation == "contact"
                else "the FEM soft profile exposes no callable for this operation"
            )
            operations.append(_unsupported(operation, reason))
    return RoboticsBackendProfile(
        backend="phydrax-soft-fem",
        implementation="FEMSoftPlant",
        operations=tuple(operations),
    )


class FEMSoftLoadLayout(StrictModule, NonTrainableState):
    """Static names and exact array routes for native FEM load coefficients."""

    pressure_region_ids: tuple[str, ...] = eqx.field(static=True)
    fiber_channel_ids: tuple[str, ...] = eqx.field(static=True)
    fiber_region_ids: tuple[str, ...] = eqx.field(static=True)
    body_force_region_ids: tuple[str, ...] = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        pressure_region_ids: Sequence[str] = (),
        fiber_routes: Sequence[tuple[str, str]] = (),
        body_force_region_ids: Sequence[str] = (),
        spatial_dimension: int,
    ):
        pressure = _identifiers(pressure_region_ids, "pressure region ID")
        routes = tuple(
            (
                _identifier(channel, "fiber channel ID"),
                _identifier(region, "fiber region ID"),
            )
            for channel, region in fiber_routes
        )
        channels = tuple(channel for channel, _ in routes)
        if len(set(channels)) != len(channels):
            raise ValueError("Fiber channel IDs must be unique.")
        body = _identifiers(body_force_region_ids, "body-force region ID")
        dimension = int(spatial_dimension)
        if dimension not in (2, 3):
            raise ValueError("FEM soft spatial_dimension must be two or three.")
        if not pressure and not routes and not body:
            raise ValueError(
                "A FEM soft load layout must expose at least one command route."
            )
        self.pressure_region_ids = pressure
        self.fiber_channel_ids = channels
        self.fiber_region_ids = tuple(region for _, region in routes)
        self.body_force_region_ids = body
        self.spatial_dimension = dimension
        self.layout_id = canonical_fingerprint(
            {
                "kind": "fem-soft-load-layout",
                "pressure_regions": list(pressure),
                "fiber_routes": [list(route) for route in routes],
                "body_force_regions": list(body),
                "spatial_dimension": dimension,
            }
        )

    def zero_command(self, dtype: Any, /) -> FEMSoftCommand:
        dtype_ = np.dtype(dtype)
        if dtype_.kind not in "fc":
            raise TypeError("FEM soft commands require an inexact dtype.")
        return FEMSoftCommand(
            jnp.zeros((len(self.pressure_region_ids),), dtype=dtype_),
            jnp.zeros((len(self.fiber_channel_ids),), dtype=dtype_),
            jnp.zeros(
                (len(self.body_force_region_ids), self.spatial_dimension), dtype=dtype_
            ),
        )


class FEMSoftCommand(StrictModule, NonTrainableState):
    """Complete pressure, fiber-tension, and body-force command arrays."""

    pressure: Array
    fiber_tension: Array
    body_force: Array

    def __init__(
        self,
        pressure: ArrayLike,
        fiber_tension: ArrayLike,
        body_force: ArrayLike,
        /,
    ):
        pressure_ = jnp.asarray(pressure)
        fiber_ = jnp.asarray(fiber_tension)
        body_ = jnp.asarray(body_force)
        _common_dtype((pressure_, fiber_, body_), "FEM soft command")
        self.pressure = pressure_
        self.fiber_tension = fiber_
        self.body_force = body_


class FEMSoftLoads(StrictModule, NonTrainableState):
    """One command bound to named FEM boundary and volume regions."""

    command: FEMSoftCommand
    layout: FEMSoftLoadLayout

    def __init__(self, command: FEMSoftCommand, layout: FEMSoftLoadLayout, /):
        if not isinstance(command, FEMSoftCommand):
            raise TypeError("command must be FEMSoftCommand.")
        if not isinstance(layout, FEMSoftLoadLayout):
            raise TypeError("layout must be FEMSoftLoadLayout.")
        expected = (
            (len(layout.pressure_region_ids),),
            (len(layout.fiber_channel_ids),),
            (len(layout.body_force_region_ids), layout.spatial_dimension),
        )
        observed = (
            command.pressure.shape,
            command.fiber_tension.shape,
            command.body_force.shape,
        )
        if observed != expected:
            raise ValueError(
                f"FEM soft command shapes {observed} do not match load layout {expected}."
            )
        self.command = command
        self.layout = layout

    def pressure(self, region_id: str, /) -> Array:
        try:
            index = self.layout.pressure_region_ids.index(str(region_id))
        except ValueError as error:
            raise KeyError(f"Unknown pressure region {region_id!r}.") from error
        return self.command.pressure[index]

    def fiber_tension(self, channel_id: str, /) -> Array:
        try:
            index = self.layout.fiber_channel_ids.index(str(channel_id))
        except ValueError as error:
            raise KeyError(f"Unknown fiber channel {channel_id!r}.") from error
        return self.command.fiber_tension[index]

    def body_force(self, region_id: str, /) -> Array:
        try:
            index = self.layout.body_force_region_ids.index(str(region_id))
        except ValueError as error:
            raise KeyError(f"Unknown body-force region {region_id!r}.") from error
        return self.command.body_force[index]


class FEMSoftParameters(StrictModule, NonTrainableState):
    """Complete numeric parameter PyTree consumed by the prepared FEM form."""

    values: Any

    def __init__(self, values: Any = (), /):
        self.values = values


class FEMSoftStepArguments(StrictModule, NonTrainableState):
    """Native FEM user arguments retaining loads, parameters, and route identity."""

    loads: FEMSoftLoads
    parameters: Any


class FEMSoftState(StrictModule, NonTrainableState):
    """Complete evolving soft-body state without duplicated runtime counters."""

    displacement: Array
    velocity: Array
    acceleration: Array
    material_state: tuple[Array, ...]
    region_force: Array

    def __init__(
        self,
        displacement: ArrayLike,
        velocity: ArrayLike,
        acceleration: ArrayLike,
        material_state: Sequence[ArrayLike],
        region_force: ArrayLike,
        /,
    ):
        displacement_ = jnp.asarray(displacement)
        velocity_ = jnp.asarray(velocity)
        acceleration_ = jnp.asarray(acceleration)
        materials = tuple(jnp.asarray(value) for value in material_state)
        force = jnp.asarray(region_force)
        if (
            displacement_.ndim != 2
            or displacement_.shape != velocity_.shape
            or displacement_.shape != acceleration_.shape
        ):
            raise ValueError(
                "FEM soft displacement, velocity, and acceleration must share "
                "a (node, spatial-component) shape."
            )
        _common_dtype(
            (displacement_, velocity_, acceleration_, *materials, force),
            "FEM soft state",
        )
        self.displacement = displacement_
        self.velocity = velocity_
        self.acceleration = acceleration_
        self.material_state = materials
        self.region_force = force


class FEMSoftSensorLayout(StrictModule, NonTrainableState):
    """Named node regions for mean displacement and named force channels."""

    displacement_region_ids: tuple[str, ...] = eqx.field(static=True)
    displacement_region_nodes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    force_region_ids: tuple[str, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        displacement_regions: Mapping[str, Sequence[int]]
        | Sequence[tuple[str, Sequence[int]]] = (),
        force_region_ids: Sequence[str] = (),
    ):
        records = (
            tuple(sorted(displacement_regions.items()))
            if isinstance(displacement_regions, Mapping)
            else tuple(displacement_regions)
        )
        names = _identifiers(tuple(name for name, _ in records), "displacement region ID")
        nodes = tuple(tuple(int(index) for index in indices) for _, indices in records)
        if any(not region for region in nodes):
            raise ValueError(
                "Every displacement sensor region requires at least one node."
            )
        if any(index < 0 for region in nodes for index in region):
            raise ValueError("Displacement sensor node indices must be nonnegative.")
        if any(len(set(region)) != len(region) for region in nodes):
            raise ValueError("Displacement sensor regions must not repeat node indices.")
        forces = _identifiers(force_region_ids, "force region ID")
        if not names and not forces:
            raise ValueError("A FEM soft sensor layout must expose at least one sensor.")
        self.displacement_region_ids = names
        self.displacement_region_nodes = nodes
        self.force_region_ids = forces
        self.layout_id = canonical_fingerprint(
            {
                "kind": "fem-soft-sensor-layout",
                "displacement_regions": [
                    [name, list(region)]
                    for name, region in zip(names, nodes, strict=True)
                ],
                "force_regions": list(forces),
            }
        )


class FEMSoftObservation(StrictModule, NonTrainableState):
    """Current region displacement and force observations."""

    displacement: Array
    force: Array
    finite: Array
    successful: Array
    status: Array


class FEMSoftResetEvidence(StrictModule, NonTrainableState):
    initial_state: FEMSoftState
    observation: FEMSoftObservation
    capability_ids: tuple[str, ...] = eqx.field(static=True)


class FEMSoftStepEvidence(StrictModule, NonTrainableState):
    dynamics: FiniteElementDynamicsResult
    loads: FEMSoftLoads
    candidate_observation: FEMSoftObservation
    accepted_observation: FEMSoftObservation
    force_finite: Array


class FEMSoftCapabilityManifest(StrictModule, NonTrainableState):
    """Closed, content-addressed capability set for one fixed-mesh FEM plant."""

    capability_ids: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(self, capability_ids: Sequence[str], /):
        capabilities = tuple(sorted(_identifiers(capability_ids, "FEM capability ID")))
        self.capability_ids = capabilities
        self.manifest_id = canonical_fingerprint(
            {"kind": "fem-soft-capability-manifest", "capabilities": list(capabilities)}
        )

    def supports(self, capability_id: str, /) -> bool:
        return str(capability_id) in self.capability_ids

    def require(self, capability_ids: Sequence[str], /) -> None:
        requested = _identifiers(capability_ids, "required FEM capability ID")
        missing = tuple(value for value in requested if value not in self.capability_ids)
        if missing:
            explicitly_unsupported = tuple(
                value for value in missing if value in _UNSUPPORTED_CAPABILITIES
            )
            if explicitly_unsupported:
                raise ValueError(
                    "FEM soft fixed-mesh profile rejects unsupported remeshing, fracture, "
                    f"or contact capabilities: {explicitly_unsupported}."
                )
            raise ValueError(f"FEM soft capabilities are unavailable: {missing}.")


class FEMSoftPlant(AbstractDiscretePlant, NonTrainableState):
    """Fixed-mesh, contact-free transient FEM as an atomic discrete plant."""

    dynamics_plan: FiniteElementDynamicsPlan
    initial_dynamics_state: FiniteElementDynamicsState
    material_templates: tuple[MaterialState, ...]
    load_layout: FEMSoftLoadLayout
    sensor_layout: FEMSoftSensorLayout
    region_force_evaluator: Callable | None
    parameters: PlantParameters
    capabilities: FEMSoftCapabilityManifest
    profile: RoboticsBackendProfile
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: FEMSoftState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    state_codec: PlantStateVectorCodec
    control_codec: ControlVectorCodec
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    constitutive_capability_id: str = eqx.field(static=True)
    region_force_evaluator_id: str | None = eqx.field(static=True)
    initial_step_offset: int = eqx.field(static=True)
    initial_state_version_offset: int = eqx.field(static=True)

    def __init__(
        self,
        dynamics_plan: FiniteElementDynamicsPlan,
        initial_state: FiniteElementDynamicsState,
        parameters: FEMSoftParameters,
        load_layout: FEMSoftLoadLayout,
        sensor_layout: FEMSoftSensorLayout,
        /,
        *,
        constitutive_capability_id: FEMConstitutiveCapability,
        initial_region_force: ArrayLike | None = None,
        region_force_evaluator: Callable | None = None,
        region_force_evaluator_id: str | None = None,
    ):
        if not isinstance(dynamics_plan, FiniteElementDynamicsPlan):
            raise TypeError("dynamics_plan must be FiniteElementDynamicsPlan.")
        if not isinstance(initial_state, FiniteElementDynamicsState):
            raise TypeError("initial_state must be FiniteElementDynamicsState.")
        if not isinstance(parameters, FEMSoftParameters):
            raise TypeError("parameters must be FEMSoftParameters.")
        if not isinstance(load_layout, FEMSoftLoadLayout):
            raise TypeError("load_layout must be FEMSoftLoadLayout.")
        if not isinstance(sensor_layout, FEMSoftSensorLayout):
            raise TypeError("sensor_layout must be FEMSoftSensorLayout.")
        constitutive = str(constitutive_capability_id)
        if constitutive not in _CONSTITUTIVE_CAPABILITIES:
            raise ValueError("Unknown FEM soft constitutive capability ID.")
        if load_layout.spatial_dimension != initial_state.displacement.shape[-1]:
            raise ValueError("Load spatial dimension does not match FEM displacement.")
        dynamics_plan.problem.state_space.validate(initial_state.displacement)
        dynamics_plan.problem.state_space.validate(initial_state.velocity)
        dynamics_plan.problem.state_space.validate(initial_state.acceleration)
        if any(
            index >= initial_state.displacement.shape[0]
            for region in sensor_layout.displacement_region_nodes
            for index in region
        ):
            raise ValueError("A displacement sensor node lies outside the FEM state.")
        if (region_force_evaluator is None) != (region_force_evaluator_id is None):
            raise ValueError(
                "region_force_evaluator and its explicit identity must be supplied together."
            )
        if sensor_layout.force_region_ids and region_force_evaluator is None:
            raise ValueError("Force sensors require an explicit region_force_evaluator.")
        evaluator_id = (
            None
            if region_force_evaluator_id is None
            else _identifier(region_force_evaluator_id, "region force evaluator ID")
        )
        initial_step_offset = int(np.asarray(initial_state.step))
        initial_state_version_offset = int(np.asarray(initial_state.state_version))

        templates = (
            () if initial_state.materials is None else initial_state.materials.states
        )
        material_values = tuple(template.committed for template in templates)
        dtype = _common_dtype(
            (
                initial_state.displacement,
                initial_state.velocity,
                initial_state.acceleration,
                *material_values,
            ),
            "FEM soft initial state",
        )
        expected_force_shape = (
            len(sensor_layout.force_region_ids),
            load_layout.spatial_dimension,
        )
        if initial_region_force is None:
            if sensor_layout.force_region_ids:
                raise ValueError(
                    "initial_region_force is required when force sensors are configured."
                )
            force = jnp.zeros(expected_force_shape, dtype=dtype)
        else:
            force = jnp.asarray(initial_region_force)
            if force.shape != expected_force_shape:
                raise ValueError(
                    f"initial_region_force must have shape {expected_force_shape}."
                )
            if np.dtype(force.dtype) != dtype:
                raise TypeError("initial_region_force dtype must match the FEM state.")
        fallback = FEMSoftState(
            initial_state.displacement,
            initial_state.velocity,
            initial_state.acceleration,
            material_values,
            force,
        )
        zero_command = load_layout.zero_command(dtype)
        FEMSoftLoads(zero_command, load_layout)
        state_schema = ArrayPyTreeSchema.from_tree(fallback, case_ndim=0)
        control_schema = ArrayPyTreeSchema.from_tree(zero_command, case_ndim=0)
        parameter_schema = ArrayPyTreeSchema.from_tree(parameters, case_ndim=0)

        capability_ids = [
            constitutive,
            FEM_EXACT_STATE_CODEC_CAPABILITY_ID,
            FEM_EXACT_CONTROL_CODEC_CAPABILITY_ID,
            FEM_ATOMIC_REPLAY_CAPABILITY_ID,
        ]
        if sensor_layout.displacement_region_ids:
            capability_ids.append(FEM_REGION_DISPLACEMENT_SENSOR_CAPABILITY_ID)
        if load_layout.pressure_region_ids:
            capability_ids.append(FEM_PRESSURE_ACTUATION_CAPABILITY_ID)
        if load_layout.fiber_channel_ids:
            capability_ids.append(FEM_FIBER_ACTUATION_CAPABILITY_ID)
        if load_layout.body_force_region_ids:
            capability_ids.append(FEM_BODY_FORCE_ACTUATION_CAPABILITY_ID)
        if sensor_layout.force_region_ids:
            capability_ids.append(FEM_REGION_FORCE_SENSOR_CAPABILITY_ID)
        capabilities = FEMSoftCapabilityManifest(capability_ids)
        semantic = SemanticProvenance(
            {
                "kind": "fixed-mesh-contact-free-soft-fem-plant",
                "dynamics_plan_id": dynamics_plan.plan_id,
                "compilation_id": dynamics_plan.problem.compilation_id,
                "material_layout_id": (
                    None
                    if initial_state.materials is None
                    else initial_state.materials.layout_id
                ),
                "material_sites": [
                    [template.site_id.key, template.model_id] for template in templates
                ],
                "constitutive_capability_id": constitutive,
                "load_layout_id": load_layout.layout_id,
                "sensor_layout_id": sensor_layout.layout_id,
                "region_force_evaluator_id": evaluator_id,
                "capability_manifest_id": capabilities.manifest_id,
                "unsupported": sorted(_UNSUPPORTED_CAPABILITIES),
            }
        )
        numeric = NumericRevision(
            semantic,
            {
                "initial_state": fallback,
                "parameters": parameters,
                "mass_coefficient": dynamics_plan.mass_coefficient,
                "damping_coefficient": dynamics_plan.damping_coefficient,
                "initial_material_versions": tuple(
                    template.state_version for template in templates
                ),
                "initial_step_offset": initial_step_offset,
                "initial_state_version_offset": initial_state_version_offset,
            },
        )
        signature = ExecutableSignature(
            shapes=tuple(
                (f"state:{leaf.path}", leaf.shape) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.shape) for leaf in control_schema.leaves
            )
            + tuple(
                (f"parameter:{leaf.path}", leaf.shape) for leaf in parameter_schema.leaves
            ),
            dtypes=tuple(
                (f"state:{leaf.path}", leaf.dtype) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.dtype) for leaf in control_schema.leaves
            )
            + tuple(
                (f"parameter:{leaf.path}", leaf.dtype) for leaf in parameter_schema.leaves
            ),
            space_ids={
                "fem_state": dynamics_plan.problem.state_space.space_id,
                "state_schema": state_schema.schema_id,
                "control_schema": control_schema.schema_id,
                "parameter_schema": parameter_schema.schema_id,
            },
            capacities={
                "nodes": initial_state.displacement.shape[0],
                "spatial_dimension": load_layout.spatial_dimension,
                "pressure_regions": len(load_layout.pressure_region_ids),
                "fiber_channels": len(load_layout.fiber_channel_ids),
                "body_force_regions": len(load_layout.body_force_region_ids),
                "displacement_sensors": len(sensor_layout.displacement_region_ids),
                "force_sensors": len(sensor_layout.force_region_ids),
            },
            algorithm_facts={
                "integrator": dynamics_plan.method.method_id,
                "capabilities": capabilities.capability_ids,
                "fixed_mesh": True,
                "contact": False,
            },
            backend_facts={"implementation": "native-phydrax-fem"},
        )
        state_size = sum(prod(leaf.shape) for leaf in state_schema.leaves)
        state_codec = PlantStateVectorCodec(
            state_schema,
            StateLayout(
                (state_size,),
                axes=("coordinate",),
                local_space=ArraySpace((state_size,), dtype=dtype),
                tangent_space=ArraySpace((state_size,), dtype=dtype),
            ),
            fallback,
            semantic_provenance=semantic,
            numeric_revision=numeric,
            executable_signature=signature,
        )
        control_codec = ControlVectorCodec(
            control_schema,
            semantic_provenance=semantic,
            numeric_revision=numeric,
            executable_signature=signature,
        )

        self.dynamics_plan = dynamics_plan
        self.initial_dynamics_state = initial_state
        self.material_templates = templates
        self.load_layout = load_layout
        self.sensor_layout = sensor_layout
        self.region_force_evaluator = region_force_evaluator
        self.parameters = PlantParameters(parameters, parameter_schema.schema_id, numeric)
        self.capabilities = capabilities
        devices = tuple(
            sorted({device.platform for device in initial_state.displacement.devices()})
        )
        self.profile = _profile(
            dtype, dynamics_plan.derivative_policy is not None, devices
        )
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = fallback
        self.semantic_provenance = semantic
        self.numeric_revision = numeric
        self.execution_signature = signature
        self.state_codec = state_codec
        self.control_codec = control_codec
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.constitutive_capability_id = constitutive
        self.region_force_evaluator_id = evaluator_id
        self.initial_step_offset = initial_step_offset
        self.initial_state_version_offset = initial_state_version_offset

    def require_capabilities(self, capability_ids: Sequence[str], /) -> None:
        self.capabilities.require(capability_ids)

    def _payload(self, payload: FEMSoftState, /) -> FEMSoftState:
        if not isinstance(payload, FEMSoftState):
            raise TypeError("FEM soft plant payload must be FEMSoftState.")
        self.state_schema.validate(payload)
        return payload

    def _runtime_payload(self, state: PlantRuntimeState, /) -> FEMSoftState:
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("state must be PlantRuntimeState.")
        observed = (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        )
        expected = (
            self.semantic_provenance.semantic_id,
            self.numeric_revision.revision_id,
            self.state_schema.schema_id,
            self.execution_signature.signature_id,
        )
        if observed != expected:
            raise ValueError("Runtime state belongs to a different FEM soft plant.")
        return self._payload(state.payload)

    def _materials(
        self, values: tuple[Array, ...], step: Array, /
    ) -> MaterialTransaction | None:
        if not self.material_templates:
            return None
        step_host = np.asarray(step)
        if step_host.shape != () or not np.issubdtype(step_host.dtype, np.integer):
            raise TypeError("FEM material reconstruction requires a scalar integer step.")
        version_offset = int(step_host)
        states = tuple(
            MaterialState(
                template.site_id,
                template.model_id,
                value,
                state_version=template.state_version + version_offset,
            )
            for template, value in zip(self.material_templates, values, strict=True)
        )
        return MaterialTransaction(states)

    def _dynamics_state(
        self, payload: FEMSoftState, time: Array, step: Array, /
    ) -> FiniteElementDynamicsState:
        return FiniteElementDynamicsState(
            payload.displacement,
            payload.velocity,
            payload.acceleration,
            time=time,
            step=step + self.initial_step_offset,
            state_version=step + self.initial_state_version_offset,
            materials=self._materials(payload.material_state, step),
        )

    def _soft_state(
        self,
        state: FiniteElementDynamicsState,
        region_force: Array,
        /,
    ) -> FEMSoftState:
        materials = (
            ()
            if state.materials is None
            else tuple(material.trial for material in state.materials.states)
        )
        return FEMSoftState(
            state.displacement,
            state.velocity,
            state.acceleration,
            materials,
            region_force,
        )

    def _observe_payload(self, payload: FEMSoftState, /) -> FEMSoftObservation:
        if self.sensor_layout.displacement_region_nodes:
            displacement = jnp.stack(
                tuple(
                    jnp.mean(
                        payload.displacement[jnp.asarray(nodes, dtype=jnp.int32)], axis=0
                    )
                    for nodes in self.sensor_layout.displacement_region_nodes
                )
            )
        else:
            displacement = jnp.zeros(
                (0, self.load_layout.spatial_dimension), dtype=payload.displacement.dtype
            )
        finite = jnp.all(jnp.isfinite(displacement)) & jnp.all(
            jnp.isfinite(payload.region_force)
        )
        status = jnp.where(
            finite,
            int(RoboticsOperationStatus.SUCCESS),
            int(RoboticsOperationStatus.NONFINITE),
        ).astype(jnp.int32)
        return FEMSoftObservation(
            displacement, payload.region_force, finite, finite, status
        )

    def observe(self, state: PlantRuntimeState, /) -> FEMSoftObservation:
        """Observe current named regions without recomputing hidden plant state."""
        return self._observe_payload(self._runtime_payload(state))

    def propose_reset(
        self,
        keys: Array,
        parameters: FEMSoftParameters,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del keys, initial_time
        if case_shape:
            raise ValueError("FEMSoftPlant supports one unbatched FEM instance.")
        self.parameter_schema.validate(parameters)
        payload = self.reset_fallback
        successful = jnp.asarray(True)
        status = jnp.asarray(int(RoboticsOperationStatus.SUCCESS), dtype=jnp.int32)
        observation = self._observe_payload(payload)
        evidence = FEMSoftResetEvidence(
            payload, observation, self.capabilities.capability_ids
        )
        return PlantProposal(
            payload, payload, successful, successful, status, status, evidence
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
        """Solve one native FEM transaction and retain raw and rollback payloads."""
        del keys
        source = self._payload(source)
        self.control_schema.validate(commands)
        self.parameter_schema.validate(parameters)
        loads = FEMSoftLoads(commands, self.load_layout)
        user_args = FEMSoftStepArguments(loads, parameters.values)
        accepted_dynamics = self._dynamics_state(
            source, context.source_time, context.step_index
        )
        result = solve_finite_element_dynamics_step(
            prepare_finite_element_dynamics_step(
                self.dynamics_plan,
                accepted_dynamics,
                context.duration,
                args=user_args,
            )
        )
        if self.region_force_evaluator is None:
            candidate_force = source.region_force
        else:
            candidate_force = jnp.asarray(
                self.region_force_evaluator(
                    result.candidate.state, loads, parameters.values
                )
            )
            if candidate_force.shape != source.region_force.shape:
                raise ValueError(
                    "region_force_evaluator output shape changed from the prepared sensor layout."
                )
            if candidate_force.dtype != source.region_force.dtype:
                raise TypeError(
                    "region_force_evaluator output dtype changed from the prepared FEM state."
                )
        force_finite = jnp.all(jnp.isfinite(candidate_force))
        successful = result.accepted & force_finite
        candidate = self._soft_state(result.candidate.state, candidate_force)
        committed_candidate = self._soft_state(result.accepted_state, candidate_force)
        accepted = jax.tree.map(
            lambda candidate_leaf, source_leaf: jnp.where(
                successful, candidate_leaf, source_leaf
            ),
            committed_candidate,
            source,
        )
        status = jnp.where(
            ~result.nonlinear.successful,
            int(RoboticsOperationStatus.PROVIDER_FAILURE),
            jnp.where(
                ~result.candidate.admissibility.admissible,
                int(RoboticsOperationStatus.INVALID_STATE),
                jnp.where(
                    force_finite,
                    int(RoboticsOperationStatus.SUCCESS),
                    int(RoboticsOperationStatus.NONFINITE),
                ),
            ),
        ).astype(jnp.int32)
        candidate_observation = self._observe_payload(candidate)
        accepted_observation = self._observe_payload(accepted)
        evidence = FEMSoftStepEvidence(
            result,
            loads,
            candidate_observation,
            accepted_observation,
            force_finite,
        )
        return PlantProposal(
            candidate,
            accepted,
            jnp.asarray(True),
            successful,
            status,
            result.nonlinear.status,
            evidence,
        )


__all__ = [
    "FEM_ATOMIC_REPLAY_CAPABILITY_ID",
    "FEM_BODY_FORCE_ACTUATION_CAPABILITY_ID",
    "FEM_CONTACT_CAPABILITY_ID",
    "FEM_EXACT_CONTROL_CODEC_CAPABILITY_ID",
    "FEM_EXACT_STATE_CODEC_CAPABILITY_ID",
    "FEM_FIBER_ACTUATION_CAPABILITY_ID",
    "FEM_FRACTURE_CAPABILITY_ID",
    "FEM_HYPERELASTICITY_CAPABILITY_ID",
    "FEM_LINEAR_ELASTICITY_CAPABILITY_ID",
    "FEM_PRESSURE_ACTUATION_CAPABILITY_ID",
    "FEM_REGION_DISPLACEMENT_SENSOR_CAPABILITY_ID",
    "FEM_REGION_FORCE_SENSOR_CAPABILITY_ID",
    "FEM_REMESH_CAPABILITY_ID",
    "FEM_VISCOELASTICITY_CAPABILITY_ID",
    "FEMConstitutiveCapability",
    "FEMSoftCapabilityManifest",
    "FEMSoftCommand",
    "FEMSoftLoadLayout",
    "FEMSoftLoads",
    "FEMSoftObservation",
    "FEMSoftParameters",
    "FEMSoftPlant",
    "FEMSoftResetEvidence",
    "FEMSoftSensorLayout",
    "FEMSoftState",
    "FEMSoftStepArguments",
    "FEMSoftStepEvidence",
]
