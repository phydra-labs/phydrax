#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._array_tree import ArrayPyTreeSchema
from ..._fingerprint import canonical_fingerprint
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends._types import BackendUnavailableError
from ...discretization.mpm import (
    MPMGridState,
    MPMParticleState,
    MPMRunStatus,
    MPMRuntimeState,
    MPMStepResult,
    PreparedMPMDynamics,
)
from ...dynamics._plant import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantRuntimeState,
    PlantStepContext,
)
from ...equations._material_point import (
    CompiledMaterialPointProblem,
    MaterialPointArguments,
)
from ._backend import (
    ROBOTICS_OPERATIONS,
    RoboticsBackendProfile,
    RoboticsOperationCapability,
    RoboticsOperationEvidence,
    RoboticsOperationStatus,
)


_MPM_BACKEND = "native-mpm"
_INVALID_RUNTIME_METADATA = 1000
_MPM_FEATURES = frozenset(
    {
        "body-force-command",
        "fixed-body-force",
        "fixed-topology",
        "particle-region-observation",
        "grid-surface-observation",
        "contact",
        "amr",
        "topology-change",
    }
)


def _positive_integer(value: Any, owner: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{owner} must be an integer.")
    value_ = int(value)
    if value_ <= 0:
        raise ValueError(f"{owner} must be positive.")
    return value_


def _case_ndim(value: Any, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError("case_ndim must be an integer.")
    value_ = int(value)
    if value_ < 0:
        raise ValueError("case_ndim must be nonnegative.")
    return value_


def _broadcast_tree(tree: Any, case_shape: tuple[int, ...], /) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, case_shape + leaf.shape), tree
    )


def _apply_casewise(operation: Callable[..., Any], case_rank: int, *values: Any) -> Any:
    applied = operation
    for _ in range(case_rank):
        applied = jax.vmap(applied)
    return applied(*values)


def _empty_grid(dynamics: PreparedMPMDynamics, dtype: Any, /) -> MPMGridState:
    scalar_shape = (dynamics.nodal_fields.field_count,) + dynamics.splat.target_shape
    vector_shape = scalar_shape + (dynamics.dimension,)
    scalar = jnp.zeros(scalar_shape, dtype=dtype)
    vector = jnp.zeros(vector_shape, dtype=dtype)
    return MPMGridState(
        scalar,
        vector,
        vector,
        vector,
        vector,
        vector,
        jnp.zeros(scalar_shape, dtype=bool),
    )


def _runtime_at_reset_time(state: MPMRuntimeState, time: Array, /) -> MPMRuntimeState:
    dtype = state.particles.position.dtype
    return MPMRuntimeState(
        state.particles,
        jnp.asarray(time, dtype=dtype).reshape(()),
        jnp.zeros((), dtype=jnp.int32),
        jnp.asarray(int(MPMRunStatus.SUCCESS), dtype=jnp.int32),
        state.topology_generation,
        state.assignment_input,
        state.material_slots,
        state.body_ids,
        state.velocity_field_slots,
        state.storage_state,
        state.lifecycle_state,
    )


def _validate_initial_runtime(
    dynamics: PreparedMPMDynamics, state: MPMRuntimeState, /
) -> None:
    particles = state.particles
    if not isinstance(particles, MPMParticleState):
        raise TypeError("Initial MPM runtime must own MPMParticleState.")
    count = dynamics.particles.capacity
    dimension = dynamics.dimension
    vector_shape = (count, dimension)
    tensor_shape = (count, dimension, dimension)
    scalar_shape = (count,)
    expected_shapes = (
        ("position", particles.position.shape, vector_shape),
        ("velocity", particles.velocity.shape, vector_shape),
        ("deformation_gradient", particles.deformation_gradient.shape, tensor_shape),
        ("affine_velocity", particles.affine_velocity.shape, tensor_shape),
        ("reference_volume", particles.reference_volume.shape, scalar_shape),
        ("first_piola", particles.first_piola.shape, tensor_shape),
        (
            "reference_energy_density",
            particles.reference_energy_density.shape,
            scalar_shape,
        ),
        ("maximum_wave_speed", particles.maximum_wave_speed.shape, scalar_shape),
        (
            "material_state",
            particles.material_state.shape,
            scalar_shape + tuple(dynamics.material.state_shape),
        ),
    )
    for name, observed, expected in expected_shapes:
        if observed != expected:
            raise ValueError(
                f"Initial MPM {name} shape {observed} does not match {expected}."
            )
    for name, value in (
        ("time", state.time),
        ("accepted_step", state.accepted_step),
        ("last_status", state.last_status),
        ("topology_generation", state.topology_generation),
    ):
        if value.shape != ():
            raise ValueError(f"Initial MPM {name} must be scalar.")
    active = dynamics.particles.active_mask
    valid_fields = (state.velocity_field_slots >= 0) & (
        state.velocity_field_slots < dynamics.nodal_fields.field_count
    )
    if not bool(np.asarray(jnp.all((~active) | valid_fields))):
        raise ValueError("Initial active particles contain invalid nodal-field slots.")
    if not bool(np.asarray(jnp.all((~active) | (state.material_slots >= 0)))):
        raise ValueError("Initial active particles contain invalid material slots.")
    if not bool(np.asarray(jnp.all((~active) | (state.body_ids >= 0)))):
        raise ValueError("Initial active particles contain invalid body IDs.")
    if (dynamics.active_blocks is None) != (state.storage_state is None):
        raise ValueError(
            "Initial MPM sparse storage ownership does not match the prepared dynamics."
        )


def _common_device(tree: Any, /) -> tuple[str, str]:
    exact: tuple[str, ...] | None = None
    platform: str | None = None
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        if not isinstance(leaf, jax.Array):
            raise TypeError(f"MPM state leaf {index} must be a JAX array.")
        devices = tuple(
            sorted(f"{device.platform}:{device.id}" for device in leaf.devices())
        )
        if len(devices) != 1:
            raise ValueError("MPM plant arrays must each reside on one JAX device.")
        leaf_platform = next(iter(leaf.devices())).platform
        if exact is None:
            exact = devices
            platform = leaf_platform
        elif devices != exact:
            raise ValueError("All MPM plant arrays must share one exact JAX device.")
    if exact is None or platform is None:
        raise ValueError("MPM plant state must contain array leaves.")
    return platform, exact[0]


def _unsupported(operation: str, reason: str, /) -> RoboticsOperationCapability:
    return RoboticsOperationCapability(
        operation,  # type: ignore[arg-type]
        supported=False,
        implementation="MPMSoftPlant",
        reason=reason,
    )


def _profile(device: str, dtype: str, /) -> RoboticsBackendProfile:
    no_callable = "the MPM soft profile exposes no callable for this operation"
    operations = []
    for operation in ROBOTICS_OPERATIONS:
        if operation == "step":
            operations.append(
                RoboticsOperationCapability(
                    "step",
                    supported=True,
                    implementation="MPMSoftPlant.step",
                    devices=(device,),
                    dtypes=(dtype,),
                    differentiability="conditional",
                    solvers=("explicit-mpm",),
                )
            )
        elif operation == "sensors":
            operations.append(
                RoboticsOperationCapability(
                    "sensors",
                    supported=True,
                    implementation="MPMSoftPlant.observe",
                    devices=(device,),
                    dtypes=(dtype,),
                    differentiability="conditional",
                )
            )
        elif operation == "contact":
            operations.append(
                _unsupported(
                    operation,
                    "contact is not bound by this fixed-topology contact-free profile",
                )
            )
        else:
            operations.append(_unsupported(operation, no_callable))
    return RoboticsBackendProfile(
        backend=_MPM_BACKEND,
        implementation="MPMSoftPlant",
        operations=tuple(operations),
    )


class MPMSoftState(StrictModule, NonTrainableState):
    """Complete accepted MPM particles/material history and last nodal grid."""

    runtime: MPMRuntimeState
    grid: MPMGridState


class MPMSoftCommand(StrictModule):
    """Dynamic arguments routed only to the prepared MPM body-acceleration hook."""

    external_arguments: Any


class MPMSoftParameters(StrictModule):
    """Dynamic parameters routed only to the prepared MPM constitutive plan."""

    material_parameters: Any


class MPMSoftFeatureManifest(StrictModule, NonTrainableState):
    """Closed capability set for one prepared MPM soft-robot instance."""

    supported: tuple[str, ...] = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(self, *, commanded_body_force: bool, fixed_body_force: bool):
        supported = [
            "fixed-topology",
            "particle-region-observation",
            "grid-surface-observation",
        ]
        if commanded_body_force:
            supported.append("body-force-command")
        if fixed_body_force:
            supported.append("fixed-body-force")
        self.supported = tuple(sorted(supported))
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "mpm-soft-capability-manifest",
                "capabilities": self.supported,
            }
        )

    def supports(self, feature: str, /) -> bool:
        feature_ = str(feature).strip().lower()
        if feature_ not in _MPM_FEATURES:
            raise ValueError(f"Unknown MPM soft feature {feature_!r}.")
        return feature_ in self.supported

    def require(self, features: Sequence[str], /) -> MPMSoftFeatureManifest:
        for feature in features:
            feature_ = str(feature).strip().lower()
            if feature_ not in _MPM_FEATURES:
                raise ValueError(f"Unknown MPM soft feature {feature_!r}.")
            if feature_ not in self.supported:
                reason = {
                    "contact": "the MPM soft profile is explicitly contact-free",
                    "amr": "no adaptive-grid transaction is bound",
                    "topology-change": "particle lifecycle/topology mutation is not bound",
                    "body-force-command": "the prepared acceleration has no array command input",
                    "fixed-body-force": "the prepared problem has no fixed body acceleration",
                }.get(feature_, "the requested feature is not bound")
                raise BackendUnavailableError(
                    _MPM_BACKEND,
                    feature_,
                    "the closed MPM soft feature manifest",
                    reason,
                )
        return self


class MPMSoftResolutionRequirement(StrictModule, NonTrainableState):
    """Optional exact consumer requirements for particle/grid preparation."""

    particle_capacity: int | None = eqx.field(static=True)
    grid_shape: tuple[int, ...] | None = eqx.field(static=True)
    field_count: int | None = eqx.field(static=True)
    evidence_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        particle_capacity: int | None = None,
        grid_shape: Sequence[int] | None = None,
        field_count: int | None = None,
        evidence_id: str | None = None,
    ):
        self.particle_capacity = (
            None
            if particle_capacity is None
            else _positive_integer(particle_capacity, "particle_capacity")
        )
        if grid_shape is None:
            grid_shape_ = None
        else:
            grid_shape_ = tuple(
                _positive_integer(size, "grid_shape dimension") for size in grid_shape
            )
            if not grid_shape_:
                raise ValueError("grid_shape must be non-empty when supplied.")
        self.grid_shape = grid_shape_
        self.field_count = (
            None if field_count is None else _positive_integer(field_count, "field_count")
        )
        if evidence_id is not None and not str(evidence_id).strip():
            raise ValueError("evidence_id must be non-empty when supplied.")
        self.evidence_id = None if evidence_id is None else str(evidence_id).strip()


class MPMSoftResolutionEvidence(StrictModule, NonTrainableState):
    """Content-bound particle and background-grid resolution evidence."""

    particle_capacity: int = eqx.field(static=True)
    active_particle_count: int = eqx.field(static=True)
    grid_shape: tuple[int, ...] = eqx.field(static=True)
    grid_node_count: int = eqx.field(static=True)
    field_count: int = eqx.field(static=True)
    route_count: int = eqx.field(static=True)
    minimum_spacing: float = eqx.field(static=True)
    particle_prepared_id: str = eqx.field(static=True)
    grid_prepared_id: str = eqx.field(static=True)
    preparation_evidence_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedMPMDynamics, /):
        if not isinstance(dynamics, PreparedMPMDynamics):
            raise TypeError("dynamics must be PreparedMPMDynamics.")
        particle_capacity = int(dynamics.particles.capacity)
        active_particle_count = int(
            np.sum(np.asarray(dynamics.particles.active_mask, dtype=np.int64))
        )
        grid_shape = tuple(int(size) for size in dynamics.splat.target_shape)
        grid_node_count = int(np.prod(grid_shape))
        field_count = int(dynamics.nodal_fields.field_count)
        route_count = int(dynamics.splat.route_count)
        minimum_spacing = float(dynamics.minimum_spacing)
        particle_prepared_id = dynamics.particles.prepared_id
        grid_prepared_id = dynamics.splat.prepared_id
        preparation_evidence_id = dynamics.resource_evidence.evidence_id
        payload = {
            "kind": "mpm-soft-resolution",
            "particle_capacity": particle_capacity,
            "active_particle_count": active_particle_count,
            "grid_shape": grid_shape,
            "grid_node_count": grid_node_count,
            "field_count": field_count,
            "route_count": route_count,
            "minimum_spacing": minimum_spacing,
            "particle_prepared_id": particle_prepared_id,
            "grid_prepared_id": grid_prepared_id,
            "preparation_evidence_id": preparation_evidence_id,
        }
        self.particle_capacity = particle_capacity
        self.active_particle_count = active_particle_count
        self.grid_shape = grid_shape
        self.grid_node_count = grid_node_count
        self.field_count = field_count
        self.route_count = route_count
        self.minimum_spacing = minimum_spacing
        self.particle_prepared_id = particle_prepared_id
        self.grid_prepared_id = grid_prepared_id
        self.preparation_evidence_id = preparation_evidence_id
        self.evidence_id = canonical_fingerprint(payload)

    def require(
        self, requirement: MPMSoftResolutionRequirement, /
    ) -> MPMSoftResolutionEvidence:
        if not isinstance(requirement, MPMSoftResolutionRequirement):
            raise TypeError("requirement must be MPMSoftResolutionRequirement.")
        checks = (
            ("particle capacity", requirement.particle_capacity, self.particle_capacity),
            ("grid shape", requirement.grid_shape, self.grid_shape),
            ("field count", requirement.field_count, self.field_count),
            ("resolution evidence ID", requirement.evidence_id, self.evidence_id),
        )
        for name, expected, observed in checks:
            if expected is not None and expected != observed:
                raise ValueError(
                    f"MPM soft {name} requirement mismatch: expected {expected!r}, "
                    f"prepared {observed!r}."
                )
        return self


class MPMSoftResetEvidence(StrictModule, NonTrainableState):
    resolution: MPMSoftResolutionEvidence
    runtime_time_aligned: Array


class MPMSoftStepEvidence(StrictModule, NonTrainableState):
    """Native MPM transaction and adapter routing evidence for one plant step."""

    native: MPMStepResult
    resolution: MPMSoftResolutionEvidence
    runtime_metadata_aligned: Array
    command_routed: Array
    operation: RoboticsOperationEvidence


class MPMSoftObservationRequest(StrictModule, NonTrainableState):
    """Fixed-shape particle-region and/or oriented grid-surface selection."""

    particle_mask: Array | None
    grid_mask: Array | None
    surface_normals: Array | None

    def __init__(
        self,
        *,
        particle_mask: Any | None = None,
        grid_mask: Any | None = None,
        surface_normals: Any | None = None,
    ):
        if particle_mask is None and grid_mask is None:
            raise ValueError("An MPM observation must select a region or surface.")
        if (grid_mask is None) != (surface_normals is None):
            raise ValueError("grid_mask and surface_normals must be supplied together.")
        particle_mask_ = None if particle_mask is None else jnp.asarray(particle_mask)
        grid_mask_ = None if grid_mask is None else jnp.asarray(grid_mask)
        normals_ = None if surface_normals is None else jnp.asarray(surface_normals)
        if particle_mask_ is not None and particle_mask_.dtype != jnp.bool_:
            raise TypeError("particle_mask must have boolean dtype.")
        if grid_mask_ is not None and grid_mask_.dtype != jnp.bool_:
            raise TypeError("grid_mask must have boolean dtype.")
        self.particle_mask = particle_mask_
        self.grid_mask = grid_mask_
        self.surface_normals = normals_


class MPMParticleRegionObservation(StrictModule, NonTrainableState):
    selected: Array
    position: Array
    velocity: Array
    deformation_gradient: Array
    material_state: Array
    mass: Array
    momentum: Array
    center_of_mass: Array
    material_energy: Array


class MPMGridSurfaceObservation(StrictModule, NonTrainableState):
    selected: Array
    active_node_count: Array
    mass: Array
    momentum: Array
    internal_force: Array
    external_force: Array
    normal_force: Array


class MPMSoftObservation(StrictModule, NonTrainableState):
    """Provenance-bound current particle-region/grid-surface observation."""

    region: MPMParticleRegionObservation | None
    surface: MPMGridSurfaceObservation | None
    evidence: RoboticsOperationEvidence
    semantic_provenance_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    state_schema_id: str = eqx.field(static=True)
    execution_signature_id: str = eqx.field(static=True)

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def successful(self) -> Array:
        return self.evidence.successful


class MPMSoftPlant(AbstractDiscretePlant, NonTrainableState):
    """Fixed-topology, contact-free soft plant over native explicit MPM."""

    compiled: CompiledMaterialPointProblem
    initial_state: MPMSoftState
    parameters: PlantParameters
    control_template: MPMSoftCommand | None
    fixed_external_arguments: Any
    profile: RoboticsBackendProfile
    features: MPMSoftFeatureManifest
    resolution: MPMSoftResolutionEvidence
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema | None
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: MPMSoftState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    device: str = eqx.field(static=True)
    exact_device: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    case_ndim: int = eqx.field(static=True)

    def __init__(
        self,
        compiled: CompiledMaterialPointProblem,
        initial_runtime: MPMRuntimeState,
        arguments: MaterialPointArguments,
        /,
        *,
        case_ndim: int = 0,
        required_resolution: MPMSoftResolutionRequirement | None = None,
        required_features: Sequence[str] = (),
    ):
        if not isinstance(compiled, CompiledMaterialPointProblem):
            raise TypeError("compiled must be CompiledMaterialPointProblem.")
        if not isinstance(initial_runtime, MPMRuntimeState):
            raise TypeError("initial_runtime must be MPMRuntimeState.")
        if not isinstance(arguments, MaterialPointArguments):
            raise TypeError("arguments must be MaterialPointArguments.")
        dynamics = compiled.dynamics
        if dynamics.contact is not None:
            raise BackendUnavailableError(
                _MPM_BACKEND,
                "contact",
                "a contact-free MPMSoftPlant",
                "the prepared MPM dynamics contains a contact plan",
            )
        if initial_runtime.lifecycle_state is not None:
            raise BackendUnavailableError(
                _MPM_BACKEND,
                "topology-change",
                "fixed particle ownership",
                "particle lifecycle state is not bound by MPMSoftPlant",
            )
        if bool(np.asarray(initial_runtime.topology_generation != 0)):
            raise BackendUnavailableError(
                _MPM_BACKEND,
                "topology-change",
                "topology generation zero",
                "the initial MPM runtime has mutated topology",
            )
        _validate_initial_runtime(dynamics, initial_runtime)
        if (
            arguments.external_arguments is not None
            and dynamics.external_acceleration is None
        ):
            raise ValueError(
                "external_arguments are unbound when no external acceleration is prepared."
            )

        case_ndim_ = _case_ndim(case_ndim)
        dtype = np.dtype(initial_runtime.particles.position.dtype).name
        initial_grid = _empty_grid(dynamics, initial_runtime.particles.position.dtype)
        initial_state = MPMSoftState(initial_runtime, initial_grid)
        device, exact_device = _common_device(initial_state)
        probe_shape = (1,) * case_ndim_
        state_probe = _broadcast_tree(initial_state, probe_shape)
        state_schema = ArrayPyTreeSchema.from_tree(state_probe, case_ndim=case_ndim_)
        if not bool(np.asarray(jnp.all(state_schema.finite_mask(state_probe)))):
            raise ValueError("Initial MPM soft state must be completely finite.")

        parameter_template = MPMSoftParameters(arguments.material_parameters)
        parameter_schema = ArrayPyTreeSchema.from_tree(parameter_template, case_ndim=0)
        external_leaves = jax.tree_util.tree_leaves(arguments.external_arguments)
        commanded_body_force = dynamics.external_acceleration is not None and bool(
            external_leaves
        )
        if external_leaves and dynamics.external_acceleration is None:
            raise ValueError("External command leaves require a prepared acceleration.")
        if any(
            not isinstance(leaf, (jax.Array, np.ndarray, np.generic))
            for leaf in external_leaves
        ):
            raise TypeError("External MPM commands must be an array PyTree.")
        control_template = (
            MPMSoftCommand(arguments.external_arguments) if commanded_body_force else None
        )
        if control_template is None:
            control_schema = None
            fixed_external_arguments = arguments.external_arguments
        else:
            control_probe = _broadcast_tree(control_template, probe_shape)
            control_schema = ArrayPyTreeSchema.from_tree(
                control_probe, case_ndim=case_ndim_
            )
            fixed_external_arguments = None

        features = MPMSoftFeatureManifest(
            commanded_body_force=commanded_body_force,
            fixed_body_force=(
                dynamics.external_acceleration is not None and not commanded_body_force
            ),
        )
        features.require(required_features)
        resolution = MPMSoftResolutionEvidence(dynamics)
        if required_resolution is not None:
            resolution.require(required_resolution)

        semantic = SemanticProvenance(
            {
                "kind": "mpm-soft-plant",
                "compiled_problem": compiled.compilation_id,
                "case_ndim": case_ndim_,
                "features": features.supported,
                "capability_manifest": features.manifest_id,
                "state_contract": "particles-material-history-and-last-grid",
                "rollback": "casewise-native-accepted-payload",
                "contact": "rejected",
                "amr": "rejected",
                "topology_change": "rejected",
            },
            resource_ids={
                "compiled_problem": compiled.compilation_id,
                "prepared_dynamics": dynamics.prepared_id,
                "particle_support": dynamics.particles.prepared_id,
                "particle_grid_transfer": dynamics.splat.prepared_id,
                "resolution": resolution.evidence_id,
            },
        )
        numeric = NumericRevision(
            semantic,
            {
                "initial_state": initial_state,
                "parameters": parameter_template,
                "control": control_template,
                "fixed_external_arguments": fixed_external_arguments,
            },
        )
        parameters = PlantParameters(
            parameter_template, parameter_schema.schema_id, numeric
        )
        shapes = tuple(
            (f"state:{leaf.path}", leaf.shape) for leaf in state_schema.leaves
        ) + tuple(
            (f"parameter:{leaf.path}", leaf.shape) for leaf in parameter_schema.leaves
        )
        dtypes = tuple(
            (f"state:{leaf.path}", leaf.dtype) for leaf in state_schema.leaves
        ) + tuple(
            (f"parameter:{leaf.path}", leaf.dtype) for leaf in parameter_schema.leaves
        )
        spaces: dict[str, str] = {
            "state": state_schema.schema_id,
            "parameters": parameter_schema.schema_id,
        }
        if control_schema is not None:
            shapes += tuple(
                (f"control:{leaf.path}", leaf.shape) for leaf in control_schema.leaves
            )
            dtypes += tuple(
                (f"control:{leaf.path}", leaf.dtype) for leaf in control_schema.leaves
            )
            spaces["control"] = control_schema.schema_id
        execution = ExecutableSignature(
            shapes=shapes,
            dtypes=dtypes,
            space_ids=spaces,
            topology_ids={
                "particle_support": dynamics.particles.prepared_id,
                "background_grid": dynamics.splat.prepared_id,
                "particle_domain": dynamics.particle_domain.plan_id,
            },
            capacities={
                "particle_capacity": resolution.particle_capacity,
                "active_particle_count": resolution.active_particle_count,
                "grid_node_count": resolution.grid_node_count,
                "nodal_field_count": resolution.field_count,
                "route_count": resolution.route_count,
            },
            algorithm_facts={
                "compiled_problem": compiled.compilation_id,
                "prepared_dynamics": dynamics.prepared_id,
                "resolution": resolution.evidence_id,
                "features": features.supported,
            },
            backend_facts={
                "backend": _MPM_BACKEND,
                "device": exact_device,
                "dtype": dtype,
            },
        )

        self.compiled = compiled
        self.initial_state = initial_state
        self.parameters = parameters
        self.control_template = control_template
        self.fixed_external_arguments = fixed_external_arguments
        self.profile = _profile(device, dtype)
        self.features = features
        self.resolution = resolution
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = initial_state
        self.semantic_provenance = semantic
        self.numeric_revision = numeric
        self.execution_signature = execution
        self.require_finite_state = True
        self.require_finite_controls = True
        self.require_finite_parameters = True
        self.device = device
        self.exact_device = exact_device
        self.dtype = dtype
        self.case_ndim = case_ndim_

    @property
    def dynamics(self) -> PreparedMPMDynamics:
        return self.compiled.dynamics

    def require_resolution(
        self, requirement: MPMSoftResolutionRequirement, /
    ) -> MPMSoftResolutionEvidence:
        return self.resolution.require(requirement)

    def require_features(self, features: Sequence[str], /) -> MPMSoftFeatureManifest:
        return self.features.require(features)

    def _state(self, state: PlantRuntimeState, /) -> tuple[MPMSoftState, tuple[int, ...]]:
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
            raise ValueError("Runtime state belongs to a different MPM soft plant.")
        if not isinstance(state.payload, MPMSoftState):
            raise TypeError("MPM soft payload must be MPMSoftState.")
        case_shape = self.state_schema.validate(state.payload)
        if state.time.shape != case_shape or state.step_index.shape != case_shape:
            raise ValueError("Runtime metadata does not match MPM soft case axes.")
        aligned = (
            (state.payload.runtime.time == state.time)
            & (state.payload.runtime.accepted_step == state.step_index)
            & (state.payload.runtime.topology_generation == 0)
        )
        if not bool(np.asarray(jnp.all(aligned))):
            raise ValueError("MPM runtime counters are inconsistent with plant metadata.")
        return state.payload, case_shape

    def propose_reset(
        self,
        keys: Array,
        parameters: MPMSoftParameters,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Array,
    ) -> PlantProposal:
        del keys, parameters

        def reset_one(time: Array) -> MPMSoftState:
            return MPMSoftState(
                _runtime_at_reset_time(self.initial_state.runtime, time),
                self.initial_state.grid,
            )

        payload = _apply_casewise(reset_one, len(case_shape), initial_time)
        attempted = jnp.ones(case_shape, dtype=bool)
        aligned = payload.runtime.time == initial_time
        successful = attempted & aligned
        status = jnp.where(aligned, 0, _INVALID_RUNTIME_METADATA).astype(jnp.int32)
        evidence = MPMSoftResetEvidence(self.resolution, aligned)
        return PlantProposal(
            payload,
            payload,
            attempted,
            successful,
            status,
            status,
            evidence,
        )

    def propose_step(
        self,
        context: PlantStepContext,
        source: MPMSoftState,
        commands: MPMSoftCommand | None,
        parameters: MPMSoftParameters,
        keys: Array,
        /,
    ) -> PlantProposal:
        del keys
        if not isinstance(source, MPMSoftState):
            raise TypeError("source must be MPMSoftState.")
        if not isinstance(parameters, MPMSoftParameters):
            raise TypeError("parameters must be MPMSoftParameters.")
        case_shape = self.state_schema.validate(source)
        parameter_cases = _broadcast_tree(parameters, case_shape)
        if self.control_schema is None:
            if commands is not None:
                raise ValueError("This MPM soft plant is autonomous.")

            def step_one(
                runtime: MPMRuntimeState,
                values: MPMSoftParameters,
                duration: Array,
            ) -> MPMStepResult:
                return self.dynamics.step_detailed(
                    runtime,
                    duration,
                    MaterialPointArguments(
                        values.material_parameters, self.fixed_external_arguments
                    ),
                )

            detail = _apply_casewise(
                step_one,
                len(case_shape),
                source.runtime,
                parameter_cases,
                context.duration,
            )
            command_routed = jnp.zeros(case_shape, dtype=bool)
        else:
            if not isinstance(commands, MPMSoftCommand):
                raise TypeError("commands must be MPMSoftCommand.")

            def step_one(
                runtime: MPMRuntimeState,
                values: MPMSoftParameters,
                command: MPMSoftCommand,
                duration: Array,
            ) -> MPMStepResult:
                return self.dynamics.step_detailed(
                    runtime,
                    duration,
                    MaterialPointArguments(
                        values.material_parameters, command.external_arguments
                    ),
                )

            detail = _apply_casewise(
                step_one,
                len(case_shape),
                source.runtime,
                parameter_cases,
                commands,
                context.duration,
            )
            command_routed = jnp.ones(case_shape, dtype=bool)

        aligned = (
            (source.runtime.time == context.source_time)
            & (source.runtime.accepted_step == context.step_index)
            & (source.runtime.topology_generation == 0)
        )
        native_success = jnp.asarray(detail.successful, dtype=bool)
        candidate_metadata_aligned = (
            (detail.candidate_state.time == context.target_time)
            & (detail.candidate_state.accepted_step == context.step_index + 1)
            & (detail.candidate_state.topology_generation == 0)
        )
        transaction_aligned = aligned & ((~native_success) | candidate_metadata_aligned)
        successful = transaction_aligned & native_success
        native_status = jnp.asarray(detail.candidate_state.last_status, dtype=jnp.int32)
        status = jnp.where(
            transaction_aligned, native_status, _INVALID_RUNTIME_METADATA
        ).astype(jnp.int32)
        candidate_payload = MPMSoftState(detail.candidate_state, detail.grid)
        accepted_payload = MPMSoftState(detail.accepted_state, detail.grid)
        candidate_finite = self.state_schema.finite_mask(candidate_payload)
        operation_status = jnp.where(
            ~transaction_aligned,
            int(RoboticsOperationStatus.INVALID_STATE),
            jnp.where(
                successful,
                int(RoboticsOperationStatus.SUCCESS),
                jnp.where(
                    native_status == int(MPMRunStatus.NONFINITE_STATE),
                    int(RoboticsOperationStatus.NONFINITE),
                    int(RoboticsOperationStatus.PROVIDER_FAILURE),
                ),
            ),
        ).astype(jnp.int32)
        operation = RoboticsOperationEvidence(
            status=operation_status,
            finite=candidate_finite,
            backend=_MPM_BACKEND,
            operation="step",
            implementation="MPMSoftPlant.propose_step/PreparedMPMDynamics.step_detailed",
            device=self.device,
            dtype=self.dtype,
            detail=(
                "commands route only through MaterialPointArguments.external_arguments; "
                "native candidate/accepted payloads and diagnostics are retained"
            ),
        )
        evidence = MPMSoftStepEvidence(
            detail,
            self.resolution,
            transaction_aligned,
            command_routed,
            operation,
        )
        return PlantProposal(
            candidate_payload,
            accepted_payload,
            aligned,
            successful,
            status,
            native_status,
            evidence,
        )

    def observe(
        self, state: PlantRuntimeState, request: MPMSoftObservationRequest, /
    ) -> MPMSoftObservation:
        payload, case_shape = self._state(state)
        if not isinstance(request, MPMSoftObservationRequest):
            raise TypeError("request must be MPMSoftObservationRequest.")
        case_rank = len(case_shape)
        particles = payload.runtime.particles
        active_particles = self.dynamics.particles.active_mask
        particle_count = self.resolution.particle_capacity
        dimension = self.dynamics.dimension

        region = None
        if request.particle_mask is not None:
            if request.particle_mask.shape != (particle_count,):
                raise ValueError("particle_mask must match particle capacity.")
            selected_base = active_particles & request.particle_mask
            selected = jnp.broadcast_to(selected_base, case_shape + (particle_count,))
            selected_vector = selected[..., None]
            selected_tensor = selected[..., None, None]
            position = jnp.where(selected_vector, particles.position, 0.0)
            velocity = jnp.where(selected_vector, particles.velocity, 0.0)
            deformation = jnp.where(selected_tensor, particles.deformation_gradient, 0.0)
            history_axes = particles.material_state.ndim - case_rank - 1
            selected_history = jnp.reshape(selected, selected.shape + (1,) * history_axes)
            material_state = jnp.where(selected_history, particles.material_state, 0.0)
            masses = jnp.asarray(
                self.dynamics.particles.safe_masses,
                dtype=particles.position.dtype,
            )
            weighted_mass = jnp.where(selected, masses, 0.0)
            mass = jnp.sum(weighted_mass, axis=-1)
            momentum = jnp.sum(weighted_mass[..., None] * velocity, axis=-2)
            first_moment = jnp.sum(weighted_mass[..., None] * position, axis=-2)
            center = first_moment / jnp.where(mass[..., None] > 0.0, mass[..., None], 1.0)
            material_energy = jnp.sum(
                jnp.where(
                    selected,
                    particles.reference_volume * particles.reference_energy_density,
                    0.0,
                ),
                axis=-1,
            )
            region = MPMParticleRegionObservation(
                selected,
                position,
                velocity,
                deformation,
                material_state,
                mass,
                momentum,
                center,
                material_energy,
            )

        surface = None
        if request.grid_mask is not None and request.surface_normals is not None:
            expected_mask = self.resolution.grid_shape
            expected_normals = expected_mask + (dimension,)
            if request.grid_mask.shape != expected_mask:
                raise ValueError("grid_mask must match the prepared background grid.")
            if request.surface_normals.shape != expected_normals:
                raise ValueError(
                    "surface_normals must match the grid with one vector component axis."
                )
            selected_norms = jnp.linalg.norm(request.surface_normals, axis=-1)
            normals_valid = jnp.all(
                (~request.grid_mask)
                | (jnp.isfinite(selected_norms) & (selected_norms > 0.0))
            )
            if not bool(np.asarray(normals_valid)):
                raise ValueError("Selected surface normals must be finite and nonzero.")
            field_count = self.resolution.field_count
            mask_shape = (1,) * case_rank + (1,) + expected_mask
            selected_grid = jnp.reshape(request.grid_mask, mask_shape)
            selected_grid = jnp.broadcast_to(
                selected_grid, case_shape + (field_count,) + expected_mask
            )
            selected = selected_grid & payload.grid.active
            grid_axes = tuple(range(case_rank, case_rank + 1 + len(expected_mask)))
            mass = jnp.sum(jnp.where(selected, payload.grid.mass, 0.0), axis=grid_axes)
            momentum = jnp.sum(
                jnp.where(selected[..., None], payload.grid.momentum, 0.0),
                axis=grid_axes,
            )
            internal = jnp.sum(
                jnp.where(selected[..., None], payload.grid.internal_force, 0.0),
                axis=grid_axes,
            )
            external = jnp.sum(
                jnp.where(selected[..., None], payload.grid.external_force, 0.0),
                axis=grid_axes,
            )
            normals_shape = (1,) * case_rank + (1,) + expected_normals
            normals = jnp.reshape(request.surface_normals, normals_shape)
            force = payload.grid.internal_force + payload.grid.external_force
            normal_force = jnp.sum(
                jnp.where(selected, jnp.sum(force * normals, axis=-1), 0.0),
                axis=grid_axes,
            )
            surface = MPMGridSurfaceObservation(
                selected,
                jnp.sum(selected, axis=grid_axes, dtype=jnp.int32),
                mass,
                momentum,
                internal,
                external,
                normal_force,
            )

        observed = (region, surface)
        finite = jnp.ones(case_shape, dtype=bool)
        for leaf in jax.tree_util.tree_leaves(observed):
            axes = tuple(range(case_rank, leaf.ndim))
            finite = finite & jnp.all(jnp.isfinite(leaf), axis=axes)
        status = jnp.where(
            finite,
            int(RoboticsOperationStatus.SUCCESS),
            int(RoboticsOperationStatus.NONFINITE),
        ).astype(jnp.int32)
        evidence = RoboticsOperationEvidence(
            status=status,
            finite=finite,
            backend=_MPM_BACKEND,
            operation="sensors",
            implementation="MPMSoftPlant.observe",
            device=self.device,
            dtype=self.dtype,
            detail=(
                "particle regions use fixed particle ownership and grid surfaces use "
                "the last accepted nodal grid"
            ),
        )
        return MPMSoftObservation(
            region,
            surface,
            evidence,
            self.semantic_provenance.semantic_id,
            self.numeric_revision.revision_id,
            self.state_schema.schema_id,
            self.execution_signature.signature_id,
        )


__all__ = [
    "MPMGridSurfaceObservation",
    "MPMParticleRegionObservation",
    "MPMSoftCommand",
    "MPMSoftFeatureManifest",
    "MPMSoftObservation",
    "MPMSoftObservationRequest",
    "MPMSoftParameters",
    "MPMSoftPlant",
    "MPMSoftResetEvidence",
    "MPMSoftResolutionEvidence",
    "MPMSoftResolutionRequirement",
    "MPMSoftState",
    "MPMSoftStepEvidence",
]
