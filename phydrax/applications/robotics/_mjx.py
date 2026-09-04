#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import core as jax_core

from ..._array_tree import ArrayPyTreeSchema
from ..._identity import ExecutableSignature, NumericRevision, SemanticProvenance
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends._availability import import_backend_module, probe_backend
from ...backends._types import (
    BackendAvailability,
    BackendCapabilities,
    BackendUnavailableError,
)
from ...dynamics._plant import (
    AbstractDiscretePlant,
    PlantParameters,
    PlantProposal,
    PlantRuntimeState,
    PlantStepContext,
)
from ._backend import (
    ObservationFreshness,
    ROBOTICS_OPERATIONS,
    RoboticsBackendProfile,
    RoboticsIndexEntry,
    RoboticsOperationCapability,
    RoboticsOperationEvidence,
    RoboticsOperationStatus,
    RoboticsProjection,
    RoboticsProjectionMap,
    RoboticsProjectionProvenance,
)


MJXProviderRelease: TypeAlias = tuple[int, int, int]

_MJX_PROVIDER_MINOR = (3, 12)
_MJX_JAX_DEVICES = ("cpu", "gpu", "tpu")
_MJX_DTYPES = ("float32", "float64")
_MJX_JAX_SOLVERS = ("cg", "newton")
_MJX_JAX_CONTACT_FEATURES = (
    "box-box",
    "box-mesh",
    "capsule-box",
    "capsule-capsule",
    "capsule-cylinder",
    "capsule-ellipsoid",
    "capsule-mesh",
    "cylinder-cylinder",
    "ellipsoid-cylinder",
    "ellipsoid-ellipsoid",
    "hfield-box",
    "hfield-capsule",
    "hfield-mesh",
    "hfield-sphere",
    "mesh-mesh",
    "plane-box",
    "plane-capsule",
    "plane-cylinder",
    "plane-ellipsoid",
    "plane-mesh",
    "plane-sphere",
    "sphere-box",
    "sphere-capsule",
    "sphere-cylinder",
    "sphere-ellipsoid",
    "sphere-mesh",
    "sphere-sphere",
)


def _capability(
    operation: str,
    implementation: str,
    devices: Sequence[str],
    dtypes: Sequence[str],
    /,
    *,
    differentiability: Literal["none", "conditional", "guaranteed"] = "none",
    solvers: Sequence[str] = (),
    contact_features: Sequence[str] = (),
) -> RoboticsOperationCapability:
    return RoboticsOperationCapability(
        operation,  # type: ignore[arg-type]
        supported=True,
        implementation=implementation,
        devices=devices,
        dtypes=dtypes,
        differentiability=differentiability,
        solvers=solvers,
        contact_features=contact_features,
    )


def _unsupported_capability(
    operation: str,
    implementation: str,
    reason: str,
    /,
) -> RoboticsOperationCapability:
    return RoboticsOperationCapability(
        operation,  # type: ignore[arg-type]
        supported=False,
        implementation=implementation,
        reason=reason,
    )


def _jax_profile(
    solvers: Sequence[str],
    contact_features: Sequence[str],
    devices: Sequence[str],
    dtypes: Sequence[str],
    /,
) -> RoboticsBackendProfile:
    no_callable = "the adapter exposes no callable for this operation"
    return RoboticsBackendProfile(
        backend="mjx-jax",
        implementation="MJXAdapter",
        operations=(
            _unsupported_capability("forward-kinematics", "MJXAdapter", no_callable),
            _unsupported_capability("smooth-dynamics", "MJXAdapter", no_callable),
            _unsupported_capability("contact", "MJXAdapter", no_callable),
            _capability(
                "step",
                "MJXAdapter.step",
                devices,
                dtypes,
                differentiability="conditional",
                solvers=solvers,
                contact_features=contact_features,
            ),
            _capability(
                "sensors",
                "MJXAdapter.observe/MJXAdapter.refresh",
                devices,
                dtypes,
                differentiability="conditional",
            ),
            _unsupported_capability("model-batching", "MJXAdapter", no_callable),
            _unsupported_capability("jit", "MJXAdapter", no_callable),
            _unsupported_capability("vmap", "MJXAdapter", no_callable),
            _unsupported_capability("jvp", "MJXAdapter", no_callable),
            _unsupported_capability("vjp", "MJXAdapter", no_callable),
        ),
    )


MJX_JAX_PROFILE = _jax_profile(
    _MJX_JAX_SOLVERS,
    _MJX_JAX_CONTACT_FEATURES,
    _MJX_JAX_DEVICES,
    _MJX_DTYPES,
)

_MJX_WARP_NO_CALLABLE = (
    "no MJX-Warp adapter callable is implemented; MJX-Warp remains a distinct "
    "non-differentiable provider"
)
MJX_WARP_PROFILE = RoboticsBackendProfile(
    backend="mjx-warp",
    implementation="mjx-warp",
    operations=tuple(
        _unsupported_capability(
            operation,
            "mjx-warp",
            (
                "MJX-Warp does not support automatic differentiation"
                if operation in ("jvp", "vjp")
                else _MJX_WARP_NO_CALLABLE
            ),
        )
        for operation in ROBOTICS_OPERATIONS
    ),
)

MJX_JAX_BACKEND_CAPABILITIES = BackendCapabilities(
    backend="mjx-jax",
    problem_kinds=("robotics.step", "robotics.sensors"),
    execution="device",
    host_only=False,
    supports_matrix_free=False,
    supports_assembled=False,
    coordinate_dtypes=_MJX_DTYPES,
    supports_plan_prepare_solve_refresh=True,
)


def _provider_release(version: str, /) -> MJXProviderRelease | None:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(version))
    if match is None:
        return None
    groups = match.groups()
    return int(groups[0]), int(groups[1]), int(groups[2])


def _provider_pair_reason(versions: Sequence[tuple[str, str]], /) -> str | None:
    version_by_name = dict(versions)
    if "mujoco" not in version_by_name or "mujoco-mjx" not in version_by_name:
        return "both mujoco and mujoco-mjx distributions must provide version evidence"
    mujoco_release = _provider_release(version_by_name["mujoco"])
    mjx_release = _provider_release(version_by_name["mujoco-mjx"])
    if mujoco_release is None or mjx_release is None:
        return "provider versions must begin with a major.minor.micro release"
    if (
        mujoco_release[:2] != _MJX_PROVIDER_MINOR
        or mjx_release[:2] != _MJX_PROVIDER_MINOR
    ):
        return "only the qualified MuJoCo/MJX 3.12 minor is supported"
    if mujoco_release != mjx_release:
        return "mujoco and mujoco-mjx base releases must match exactly"
    return None


def mjx_availability() -> BackendAvailability:
    """Probe the optional, release-matched MuJoCo/MJX provider pair."""
    availability = probe_backend(
        MJX_JAX_BACKEND_CAPABILITIES,
        module="mujoco.mjx",
        requirement="install matching mujoco and mujoco-mjx 3.12.x providers",
        distributions=("mujoco", "mujoco-mjx"),
    )
    if not availability.available:
        return availability
    reason = _provider_pair_reason(availability.versions)
    if reason is None:
        return availability
    return BackendAvailability(
        capabilities=availability.capabilities,
        available=False,
        requirement=availability.requirement,
        reason=reason,
        versions=availability.versions,
    )


class MJXObservationRequest(StrictModule, NonTrainableState):
    """Static selection of canonical fields in one MJX observation."""

    qpos: bool = eqx.field(static=True)
    qvel: bool = eqx.field(static=True)
    control: bool = eqx.field(static=True)
    sensors: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        qpos: bool = True,
        qvel: bool = True,
        control: bool = True,
        sensors: bool = True,
    ):
        selected = (bool(qpos), bool(qvel), bool(control), bool(sensors))
        if not any(selected):
            raise ValueError("An MJX observation must select at least one field.")
        self.qpos, self.qvel, self.control, self.sensors = selected


class MJXPreparedModelManifest(StrictModule, NonTrainableState):
    """Closed set of provider features found in one accepted host model."""

    integrator: str = eqx.field(static=True)
    solver: str = eqx.field(static=True)
    cone: str = eqx.field(static=True)
    jacobian: str = eqx.field(static=True)
    joint_types: tuple[str, ...] = eqx.field(static=True)
    geom_types: tuple[str, ...] = eqx.field(static=True)
    contact_features: tuple[str, ...] = eqx.field(static=True)
    actuator_bias_types: tuple[str, ...] = eqx.field(static=True)
    actuator_dynamics_types: tuple[str, ...] = eqx.field(static=True)
    actuator_gain_types: tuple[str, ...] = eqx.field(static=True)
    actuator_transmission_types: tuple[str, ...] = eqx.field(static=True)
    equality_types: tuple[str, ...] = eqx.field(static=True)
    sensor_types: tuple[str, ...] = eqx.field(static=True)
    tendon_wrap_types: tuple[str, ...] = eqx.field(static=True)
    enabled_features: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        integrator: str,
        solver: str,
        cone: str,
        jacobian: str,
        joint_types: Sequence[str],
        geom_types: Sequence[str],
        contact_features: Sequence[str],
        actuator_bias_types: Sequence[str],
        actuator_dynamics_types: Sequence[str],
        actuator_gain_types: Sequence[str],
        actuator_transmission_types: Sequence[str],
        equality_types: Sequence[str],
        sensor_types: Sequence[str],
        tendon_wrap_types: Sequence[str],
        enabled_features: Sequence[str],
    ):
        self.integrator = str(integrator)
        self.solver = str(solver)
        self.cone = str(cone)
        self.jacobian = str(jacobian)
        self.joint_types = tuple(joint_types)
        self.geom_types = tuple(geom_types)
        self.contact_features = tuple(contact_features)
        self.actuator_bias_types = tuple(actuator_bias_types)
        self.actuator_dynamics_types = tuple(actuator_dynamics_types)
        self.actuator_gain_types = tuple(actuator_gain_types)
        self.actuator_transmission_types = tuple(actuator_transmission_types)
        self.equality_types = tuple(equality_types)
        self.sensor_types = tuple(sensor_types)
        self.tendon_wrap_types = tuple(tendon_wrap_types)
        self.enabled_features = tuple(enabled_features)


class MJXState(StrictModule, NonTrainableState):
    """Complete opaque ``mjx.Data`` payload with derived-field epochs."""

    opaque: Any
    epoch: Any
    sensor_epoch: Any

    def __init__(self, opaque: Any, epoch: Any, sensor_epoch: Any, /):
        self.opaque = opaque
        self.epoch = jnp.asarray(epoch, dtype=jnp.int32)
        self.sensor_epoch = jnp.asarray(sensor_epoch, dtype=jnp.int32)


class MJXObservation(StrictModule, NonTrainableState):
    """One provenance-bound observation and its casewise validity evidence."""

    projection: RoboticsProjection
    request: MJXObservationRequest
    evidence: RoboticsOperationEvidence
    freshness: ObservationFreshness = eqx.field(static=True, default="state-current")

    @property
    def status(self) -> Any:
        return self.evidence.status

    @property
    def successful(self) -> Any:
        return self.evidence.successful


class MJXRefreshResult(StrictModule, NonTrainableState):
    """Candidate and casewise accepted refresh of one plant runtime state."""

    candidate_state: PlantRuntimeState
    accepted_state: PlantRuntimeState
    attempted: Any
    observation: MJXObservation
    evidence: RoboticsOperationEvidence

    @property
    def status(self) -> Any:
        return self.evidence.status

    @property
    def successful(self) -> Any:
        return self.evidence.successful


class MJXMuscleProjectionPlan(StrictModule, NonTrainableState):
    """Static selection of named or all compiled MuJoCo muscle actuators."""

    names: tuple[str, ...] | None = eqx.field(static=True)

    def __init__(self, names: Sequence[str] | None = None, /):
        if names is None:
            self.names = None
            return
        selected = tuple(str(name).strip() for name in names)
        if not selected or any(not name for name in selected):
            raise ValueError("names must contain at least one non-empty actuator name.")
        if len(set(selected)) != len(selected):
            raise ValueError("names must contain unique actuator names.")
        self.names = selected

    def prepare(self, adapter: MJXAdapter, /) -> MJXPreparedMuscleProjection:
        return MJXPreparedMuscleProjection(adapter, self)


class MJXMuscleSnapshot(StrictModule, NonTrainableState):
    """Fresh provider-native muscle state and raw signed-force projections.

    Length is the MuJoCo actuator transmission length in m, velocity is its
    extension rate in m/s, and raw force is ``mjData.actuator_force`` in N.
    Per MuJoCo's muscle convention, pulling/tensile force is negative.
    """

    activation: RoboticsProjection
    length_m: RoboticsProjection
    velocity_m_per_s: RoboticsProjection
    raw_force_N: RoboticsProjection
    evidence: RoboticsOperationEvidence
    names: tuple[str, ...] = eqx.field(static=True)
    force_owner: str = eqx.field(static=True, default="provider-native")
    raw_force_sign: str = eqx.field(static=True, default="negative-is-pulling-tension")
    geometry_authority: str = eqx.field(
        static=True, default="mujoco-compiled-transmission"
    )

    @property
    def freshness(self) -> Any:
        return self.length_m.freshness

    @property
    def successful(self) -> Any:
        return self.evidence.successful


class MJXPreparedMuscleProjection(StrictModule, NonTrainableState):
    """Prepared fixed-shape gather/scatter for compiled built-in muscles."""

    adapter: MJXAdapter
    activation_map: RoboticsProjectionMap
    length_map: RoboticsProjectionMap
    velocity_map: RoboticsProjectionMap
    raw_force_map: RoboticsProjectionMap
    names: tuple[str, ...] = eqx.field(static=True)
    actuator_indices: tuple[int, ...] = eqx.field(static=True)
    activation_indices: tuple[int, ...] = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(self, adapter: MJXAdapter, plan: MJXMuscleProjectionPlan, /):
        if not isinstance(adapter, MJXAdapter):
            raise TypeError("adapter must be MJXAdapter.")
        if not isinstance(plan, MJXMuscleProjectionPlan):
            raise TypeError("plan must be MJXMuscleProjectionPlan.")
        available = adapter.muscle_actuator_names
        if not available:
            raise ValueError("The prepared MJX model has no compiled built-in muscles.")
        names = available if plan.names is None else plan.names
        missing = tuple(name for name in names if name not in available)
        if missing:
            raise ValueError(
                "Requested names are not compiled built-in MuJoCo muscles: "
                + ", ".join(missing)
            )
        positions = tuple(available.index(name) for name in names)
        actuator_indices = tuple(
            adapter.muscle_actuator_indices[position] for position in positions
        )
        activation_indices = tuple(
            adapter.muscle_activation_indices[position] for position in positions
        )

        def projection_map(kind):
            return RoboticsProjectionMap(
                kind,
                len(names),
                tuple(
                    RoboticsIndexEntry(name, index, index + 1)
                    for index, name in enumerate(names)
                ),
                adapter.provenance,
            )

        self.adapter = adapter
        self.activation_map = projection_map("activation")
        self.length_map = projection_map("length")
        self.velocity_map = projection_map("velocity")
        self.raw_force_map = projection_map("raw-force")
        self.names = names
        self.actuator_indices = actuator_indices
        self.activation_indices = activation_indices
        digest = hashlib.sha256()
        digest.update(adapter.provenance.model.encode("utf-8"))
        digest.update(b"\x00mjx-muscle-projection-v1\x00")
        for name in names:
            digest.update(name.encode("utf-8"))
            digest.update(b"\x00")
        self.prepared_id = f"mjx-muscle-projection-sha256:{digest.hexdigest()}"

    @property
    def muscle_count(self) -> int:
        return len(self.names)

    def scatter_control(
        self,
        complete_control: Any | RoboticsProjection,
        independent_excitation: Any,
        /,
    ) -> RoboticsProjection:
        """Overwrite selected muscle controls in an otherwise complete control.

        ``complete_control`` is mandatory so non-muscle actuator controls are
        never silently zeroed or dropped. ``independent_excitation`` is dimensionless
        in [0, 1] and is independent of D1's common excitation scale.
        """

        state_epoch = None
        sample_epoch = None
        if isinstance(complete_control, RoboticsProjection):
            if complete_control.index_map.kind != self.adapter.control_map.kind:
                raise ValueError("Complete control projection must have kind 'control'.")
            if complete_control.provenance != self.adapter.provenance:
                raise ValueError("Control provenance does not match this MJX model.")
            if (
                complete_control.index_map.size != self.adapter.control_map.size
                or complete_control.index_map.name_to_range
                != self.adapter.control_map.name_to_range
            ):
                raise ValueError("Control layout does not match this MJX model.")
            base_values = complete_control.values
            state_epoch = complete_control.state_epoch
            sample_epoch = complete_control.sample_epoch
        else:
            base_values = complete_control
        base = jnp.asarray(
            base_values, dtype=self.adapter.reset_fallback.opaque.ctrl.dtype
        )
        values = jnp.asarray(independent_excitation, dtype=base.dtype)
        expected_base = (self.adapter.control_map.size,)
        if base.shape[-1:] != expected_base:
            raise ValueError(
                "complete_control must end in the complete model control size "
                f"{expected_base[0]}."
            )
        expected_excitation = base.shape[:-1] + (self.muscle_count,)
        if values.shape != expected_excitation:
            raise ValueError(
                f"independent_excitation must have complete shape {expected_excitation}."
            )
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) | (values < 0.0) | (values > 1.0)),
            "MuJoCo independent muscle excitation must be finite and lie in [0, 1].",
        )
        indices = jnp.asarray(self.actuator_indices, dtype=jnp.int32)
        scattered = base.at[..., indices].set(values)
        return RoboticsProjection(
            scattered,
            self.adapter.control_map,
            state_epoch=state_epoch,
            sample_epoch=sample_epoch,
        )

    def snapshot(
        self,
        state: PlantRuntimeState | None = None,
        /,
    ) -> MJXMuscleSnapshot:
        """Gather activation and forward-derived provider muscle quantities."""

        if state is None:
            resolved = self.adapter.reset_fallback
        else:
            _, resolved, _ = self.adapter._state(state)
        data = resolved.opaque
        actuator_indices = jnp.asarray(self.actuator_indices, dtype=jnp.int32)
        activation_indices = jnp.asarray(self.activation_indices, dtype=jnp.int32)
        activation = RoboticsProjection(
            data.act[..., activation_indices], self.activation_map
        )
        epoch_arguments = {
            "state_epoch": resolved.epoch,
            "sample_epoch": resolved.sensor_epoch,
        }
        length = RoboticsProjection(
            data.actuator_length[..., actuator_indices],
            self.length_map,
            **epoch_arguments,
        )
        velocity = RoboticsProjection(
            data._impl.actuator_velocity[..., actuator_indices],
            self.velocity_map,
            **epoch_arguments,
        )
        raw_force = RoboticsProjection(
            data.actuator_force[..., actuator_indices],
            self.raw_force_map,
            **epoch_arguments,
        )
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        activation.values,
                        length.values,
                        velocity.values,
                        raw_force.values,
                    ),
                    axis=-1,
                )
            ),
            axis=(-2, -1),
        )
        fresh = length.freshness
        status = jnp.where(
            fresh,
            jnp.where(
                finite,
                int(RoboticsOperationStatus.SUCCESS),
                int(RoboticsOperationStatus.NONFINITE),
            ),
            int(RoboticsOperationStatus.INVALID_STATE),
        ).astype(jnp.int32)
        evidence = RoboticsOperationEvidence(
            status=status,
            finite=finite,
            backend="mjx-jax",
            operation="sensors",
            implementation="MJXPreparedMuscleProjection.snapshot",
            device=self.adapter.device,
            dtype=self.adapter.dtype,
            detail=(
                "activation is state-current; length, extension velocity, and raw "
                "signed actuator force require explicit mjx.forward freshness"
            ),
        )
        return MJXMuscleSnapshot(
            activation,
            length,
            velocity,
            raw_force,
            evidence,
            self.names,
        )


class MJXAdapter(AbstractDiscretePlant, NonTrainableState):
    """Prepared MJX-JAX model implementing the complete-state plant lifecycle."""

    model: Any
    parameters: PlantParameters
    qpos_map: RoboticsProjectionMap
    qvel_map: RoboticsProjectionMap
    control_map: RoboticsProjectionMap
    observation_map: RoboticsProjectionMap
    profile: RoboticsBackendProfile
    feature_manifest: MJXPreparedModelManifest
    state_schema: ArrayPyTreeSchema
    control_schema: ArrayPyTreeSchema
    parameter_schema: ArrayPyTreeSchema
    reset_fallback: MJXState
    semantic_provenance: SemanticProvenance
    numeric_revision: NumericRevision
    execution_signature: ExecutableSignature
    provenance: RoboticsProjectionProvenance
    muscle_actuator_names: tuple[str, ...] = eqx.field(static=True)
    muscle_actuator_indices: tuple[int, ...] = eqx.field(static=True)
    muscle_activation_indices: tuple[int, ...] = eqx.field(static=True)
    require_finite_state: bool = eqx.field(static=True)
    require_finite_controls: bool = eqx.field(static=True)
    require_finite_parameters: bool = eqx.field(static=True)
    device: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    _prepared_devices: tuple[str, ...] = eqx.field(static=True)
    _mjx: Any = eqx.field(static=True)

    def __init__(
        self,
        *,
        model: Any,
        data: Any,
        qpos_map: RoboticsProjectionMap,
        qvel_map: RoboticsProjectionMap,
        control_map: RoboticsProjectionMap,
        observation_map: RoboticsProjectionMap,
        feature_manifest: MJXPreparedModelManifest,
        provenance: RoboticsProjectionProvenance,
        muscle_actuator_names: tuple[str, ...],
        muscle_actuator_indices: tuple[int, ...],
        muscle_activation_indices: tuple[int, ...],
        semantic_provenance: SemanticProvenance,
        numeric_revision: NumericRevision,
        device: str,
        dtype: Any,
        case_ndim: int,
        mjx_module: Any,
    ):
        if isinstance(case_ndim, bool) or not isinstance(case_ndim, (int, np.integer)):
            raise TypeError("case_ndim must be an integer.")
        case_ndim_ = int(case_ndim)
        if case_ndim_ < 0:
            raise ValueError("case_ndim must be nonnegative.")
        if not isinstance(semantic_provenance, SemanticProvenance):
            raise TypeError("semantic_provenance must be SemanticProvenance.")
        if not isinstance(numeric_revision, NumericRevision):
            raise TypeError("numeric_revision must be NumericRevision.")
        if numeric_revision.semantic_id != semantic_provenance.semantic_id:
            raise ValueError("MJX numeric revision belongs to different semantics.")

        device_ = str(device).lower()
        dtype_ = np.dtype(dtype).name
        prepared_devices = _common_devices(data)
        data_devices = data.qpos.devices()
        if len(data_devices) != 1:
            raise ValueError("MJX adapter state must reside on exactly one JAX device.")
        data_device = next(iter(data_devices))
        epoch = jnp.zeros((), dtype=jnp.int32, device=data_device)
        initial_state = MJXState(data, epoch, epoch)
        probe_case_shape = (1,) * case_ndim_
        state_probe = _broadcast_tree(initial_state, probe_case_shape)
        state_schema = ArrayPyTreeSchema.from_tree(state_probe, case_ndim=case_ndim_)
        if not bool(np.asarray(jnp.all(state_schema.finite_mask(state_probe)))):
            raise ValueError("Initial mjx.forward state must be completely finite.")
        control_probe = RoboticsProjection(
            jnp.broadcast_to(data.ctrl, probe_case_shape + data.ctrl.shape),
            control_map,
            state_epoch=jnp.zeros(probe_case_shape, dtype=jnp.int32, device=data_device),
            sample_epoch=jnp.zeros(probe_case_shape, dtype=jnp.int32, device=data_device),
        )
        control_schema = ArrayPyTreeSchema.from_tree(control_probe, case_ndim=case_ndim_)
        parameter_schema = ArrayPyTreeSchema.from_tree((), case_ndim=0)
        parameters = PlantParameters((), parameter_schema.schema_id, numeric_revision)
        execution_signature = ExecutableSignature(
            shapes=tuple(
                (f"state:{leaf.path}", leaf.shape) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.shape) for leaf in control_schema.leaves
            ),
            dtypes=tuple(
                (f"state:{leaf.path}", leaf.dtype) for leaf in state_schema.leaves
            )
            + tuple(
                (f"control:{leaf.path}", leaf.dtype) for leaf in control_schema.leaves
            ),
            space_ids={
                "state": state_schema.schema_id,
                "control": control_schema.schema_id,
                "parameters": parameter_schema.schema_id,
            },
            capacities={
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "nsensordata": int(model.nsensordata),
            },
            algorithm_facts={"feature_manifest": feature_manifest},
            backend_facts={
                "device": device_,
                "implementation": "jax",
                "provider": provenance.provider,
            },
        )

        self.model = model
        self.parameters = parameters
        self.qpos_map = qpos_map
        self.qvel_map = qvel_map
        self.control_map = control_map
        self.observation_map = observation_map
        self.profile = _jax_profile(
            (feature_manifest.solver,),
            feature_manifest.contact_features,
            (device_,),
            (dtype_,),
        )
        self.feature_manifest = feature_manifest
        self.state_schema = state_schema
        self.control_schema = control_schema
        self.parameter_schema = parameter_schema
        self.reset_fallback = initial_state
        self.semantic_provenance = semantic_provenance
        self.numeric_revision = numeric_revision
        self.execution_signature = execution_signature
        self.provenance = provenance
        if not (
            len(muscle_actuator_names)
            == len(muscle_actuator_indices)
            == len(muscle_activation_indices)
        ):
            raise ValueError("Prepared MJX muscle manifest arrays must have equal size.")
        self.muscle_actuator_names = tuple(muscle_actuator_names)
        self.muscle_actuator_indices = tuple(muscle_actuator_indices)
        self.muscle_activation_indices = tuple(muscle_activation_indices)
        self.require_finite_state = True
        self.require_finite_controls = False
        self.require_finite_parameters = True
        self.device = device_
        self.dtype = dtype_
        self._prepared_devices = prepared_devices
        self._mjx = mjx_module

    def _payload(self, state: MJXState, /) -> tuple[MJXState, tuple[int, ...]]:
        if not isinstance(state, MJXState):
            raise TypeError("MJX plant payload must be MJXState.")
        if not isinstance(state.opaque, self._mjx.Data):
            raise TypeError("MJX state must retain a complete mjx.Data PyTree.")
        if state.opaque.impl != self._mjx.Impl.JAX:
            raise TypeError("MJX state must retain the prepared JAX implementation.")
        case_shape = self.state_schema.validate(state)
        _validate_devices(state, self._prepared_devices)
        return state, case_shape

    def _state(
        self, state: PlantRuntimeState, /
    ) -> tuple[PlantRuntimeState, MJXState, tuple[int, ...]]:
        if not isinstance(state, PlantRuntimeState):
            raise TypeError("state must be PlantRuntimeState.")
        observed_ids = (
            state.semantic_provenance_id,
            state.numeric_revision_id,
            state.state_schema_id,
            state.execution_signature_id,
        )
        expected_ids = (
            self.semantic_provenance.semantic_id,
            self.numeric_revision.revision_id,
            self.state_schema.schema_id,
            self.execution_signature.signature_id,
        )
        if observed_ids != expected_ids:
            raise ValueError(
                "Plant runtime state belongs to a different prepared MJX plant."
            )
        payload, case_shape = self._payload(state.payload)
        if state.time.shape != case_shape or state.step_index.shape != case_shape:
            raise ValueError("Plant runtime metadata must match the MJX case axes.")
        if np.dtype(state.time.dtype).kind not in "biufc":
            raise TypeError("Plant runtime time must have a numeric dtype.")
        key = jnp.asarray(state.key)
        if jax.dtypes.issubdtype(key.dtype, jax.dtypes.prng_key):
            if key.shape != case_shape:
                raise ValueError(
                    "Plant runtime typed PRNG keys must match the MJX case axes."
                )
        else:
            if np.dtype(key.dtype) != np.dtype(jnp.uint32):
                raise TypeError("Plant runtime legacy PRNG keys must have uint32 dtype.")
            if key.shape != case_shape + (2,):
                raise ValueError(
                    "Plant runtime legacy PRNG keys must end in one size-two key axis."
                )
        checked_time = eqx.error_if(
            state.time,
            jnp.any(~jnp.isfinite(state.time)),
            "Plant runtime time must be finite.",
        )
        checked_step = eqx.error_if(
            state.step_index,
            jnp.any(state.step_index < 0),
            "Plant runtime step index must be nonnegative.",
        )
        checked = PlantRuntimeState(
            payload, checked_time, checked_step, key, *expected_ids
        )
        return checked, payload, case_shape

    def qpos(self, state: PlantRuntimeState, /) -> RoboticsProjection:
        _, payload, _ = self._state(state)
        return RoboticsProjection(payload.opaque.qpos, self.qpos_map)

    def qvel(self, state: PlantRuntimeState, /) -> RoboticsProjection:
        _, payload, _ = self._state(state)
        return RoboticsProjection(payload.opaque.qvel, self.qvel_map)

    def control(self, state: PlantRuntimeState, /) -> RoboticsProjection:
        _, payload, _ = self._state(state)
        return RoboticsProjection(
            payload.opaque.ctrl,
            self.control_map,
            state_epoch=payload.epoch,
            sample_epoch=payload.epoch,
        )

    def prepare_muscle_projection(
        self, names: Sequence[str] | None = None, /
    ) -> MJXPreparedMuscleProjection:
        """Prepare fixed gathers for named or all compiled built-in muscles."""

        return MJXMuscleProjectionPlan(names).prepare(self)

    def observe(
        self,
        state: PlantRuntimeState,
        request: MJXObservationRequest | None = None,
        /,
    ) -> MJXObservation:
        """Project requested fields with freshness derived from payload epochs."""
        _, payload, _ = self._state(state)
        request_ = MJXObservationRequest() if request is None else request
        if not isinstance(request_, MJXObservationRequest):
            raise TypeError("request must be MJXObservationRequest.")
        values, index_map = _observation_projection(
            payload.opaque,
            request_,
            self.qpos_map,
            self.qvel_map,
            self.control_map,
            self.observation_map,
        )
        sample_epoch = payload.sensor_epoch if request_.sensors else payload.epoch
        projection = RoboticsProjection(
            values,
            index_map,
            state_epoch=payload.epoch,
            sample_epoch=sample_epoch,
        )
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        status = jnp.where(
            projection.freshness,
            jnp.where(
                finite,
                int(RoboticsOperationStatus.SUCCESS),
                int(RoboticsOperationStatus.NONFINITE),
            ),
            int(RoboticsOperationStatus.INVALID_STATE),
        ).astype(jnp.int32)
        evidence = RoboticsOperationEvidence(
            status=status,
            finite=finite,
            backend="mjx-jax",
            operation="sensors",
            implementation="MJXAdapter.observe",
            device=self.device,
            dtype=self.dtype,
            detail="sensor samples are current only when sensor_epoch equals state epoch",
        )
        return MJXObservation(projection, request_, evidence)

    def propose_reset(
        self,
        keys: Any,
        parameters: Any,
        /,
        *,
        case_shape: tuple[int, ...],
        initial_time: Any,
    ) -> PlantProposal:
        del keys, parameters, initial_time
        payload = _broadcast_tree(self.reset_fallback, case_shape)
        successful = jnp.ones(case_shape, dtype=bool)
        status = jnp.zeros(case_shape, dtype=jnp.int32)
        return PlantProposal(payload, payload, successful, successful, status, status, ())

    def propose_step(
        self,
        context: PlantStepContext,
        source: MJXState,
        commands: Any,
        parameters: Any,
        keys: Any,
        /,
    ) -> PlantProposal:
        """Advance complete MJX payloads, leaving derived sensor fields stale."""
        del context, parameters, keys
        source, case_shape = self._payload(source)
        if not isinstance(commands, RoboticsProjection):
            raise TypeError("commands must be a control-kind RoboticsProjection.")
        if commands.index_map.identity != self.control_map.identity:
            raise ValueError("Control projection map identity does not match this plant.")
        if commands.state_epoch is None or commands.sample_epoch is None:
            raise ValueError("MJX control projections must be bound to a state epoch.")

        current = (commands.state_epoch == source.epoch) & (
            commands.sample_epoch == source.epoch
        )
        safe_control = jnp.where(current[..., None], commands.values, source.opaque.ctrl)
        stepped_source = source.opaque.replace(ctrl=safe_control)
        candidate_data = _apply_casewise(
            self._mjx.step, self.model, stepped_source, len(case_shape)
        )
        candidate_payload = MJXState(
            candidate_data, source.epoch + 1, source.sensor_epoch
        )
        self._payload(candidate_payload)
        finite = self.state_schema.finite_mask(candidate_payload)
        attempted = current
        successful = attempted & finite
        status = jnp.where(
            ~current,
            int(RoboticsOperationStatus.INVALID_STATE),
            jnp.where(
                finite,
                int(RoboticsOperationStatus.SUCCESS),
                int(RoboticsOperationStatus.NONFINITE),
            ),
        ).astype(jnp.int32)
        evidence = RoboticsOperationEvidence(
            status=status,
            finite=finite,
            backend="mjx-jax",
            operation="step",
            implementation="MJXAdapter.propose_step",
            device=self.device,
            dtype=self.dtype,
            detail=(
                "controls must match the source epoch and every complete candidate "
                "MJX payload case must remain finite"
            ),
        )
        return PlantProposal(
            candidate_payload,
            candidate_payload,
            attempted,
            successful,
            status,
            status,
            evidence,
        )

    def refresh(
        self,
        state: PlantRuntimeState,
        request: MJXObservationRequest | None = None,
        /,
    ) -> MJXRefreshResult:
        """Run ``mjx.forward`` and retain failed complete cases transactionally."""
        source_state, source, case_shape = self._state(state)
        candidate_data = _apply_casewise(
            self._mjx.forward, self.model, source.opaque, len(case_shape)
        )
        candidate_payload = MJXState(candidate_data, source.epoch, source.epoch)
        self._payload(candidate_payload)
        finite = self.state_schema.finite_mask(candidate_payload)
        accepted_payload = self.state_schema.select_cases(
            finite, candidate_payload, source
        )
        ids = (
            source_state.semantic_provenance_id,
            source_state.numeric_revision_id,
            source_state.state_schema_id,
            source_state.execution_signature_id,
        )
        candidate_state = PlantRuntimeState(
            candidate_payload,
            source_state.time,
            source_state.step_index,
            source_state.key,
            *ids,
        )
        accepted_state = PlantRuntimeState(
            accepted_payload,
            source_state.time,
            source_state.step_index,
            source_state.key,
            *ids,
        )
        observation = self.observe(accepted_state, request)
        attempted = jnp.ones(case_shape, dtype=bool)
        status = jnp.where(
            finite,
            int(RoboticsOperationStatus.SUCCESS),
            int(RoboticsOperationStatus.NONFINITE),
        ).astype(jnp.int32)
        evidence = RoboticsOperationEvidence(
            status=status,
            finite=finite,
            backend="mjx-jax",
            operation="sensors",
            implementation="MJXAdapter.refresh/mjx.forward",
            device=self.device,
            dtype=self.dtype,
            detail=(
                "each forwarded complete payload case is accepted and made fresh "
                "only when all of its arrays are finite"
            ),
        )
        return MJXRefreshResult(
            candidate_state, accepted_state, attempted, observation, evidence
        )


def _array_devices(array: jax.Array, /) -> tuple[str, ...]:
    return tuple(sorted(f"{device.platform}:{device.id}" for device in array.devices()))


def _common_devices(tree: Any, /) -> tuple[str, ...]:
    expected: tuple[str, ...] | None = None
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        if not isinstance(leaf, jax.Array):
            raise TypeError(f"Canonical MJX array leaf {index} is not a JAX array.")
        observed = _array_devices(leaf)
        if expected is None:
            expected = observed
        elif observed != expected:
            raise ValueError("Canonical MJX arrays must share one exact device set.")
    if not expected:
        raise ValueError("Canonical MJX state must contain array leaves.")
    return expected


def _validate_devices(tree: Any, expected: tuple[str, ...], /) -> None:
    for index, leaf in enumerate(jax.tree_util.tree_leaves(tree)):
        if not isinstance(leaf, jax.Array):
            raise TypeError(f"MJX state leaf {index} is not a canonical JAX array.")
        if isinstance(leaf, jax_core.Tracer):
            continue
        if _array_devices(leaf) != expected:
            raise ValueError(f"MJX state leaf {index} is on a non-prepared device set.")


def _broadcast_tree(tree: Any, case_shape: tuple[int, ...], /) -> Any:
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(leaf, case_shape + leaf.shape), tree
    )


def _apply_casewise(
    operation: Callable[[Any, Any], Any],
    model: Any,
    data: Any,
    case_rank: int,
    /,
) -> Any:
    applied: Callable[[Any], Any] = lambda one: operation(model, one)
    for _ in range(case_rank):
        applied = jax.vmap(applied)
    return applied(data)


def _unsupported_model(reason: str, /) -> BackendUnavailableError:
    return BackendUnavailableError(
        "mjx-jax",
        "robotics.step",
        "the closed MJX 3.12 prepared-model feature manifest",
        reason,
    )


def _enum_feature(
    value: Any,
    allowed: Sequence[tuple[int, str]],
    category: str,
    /,
) -> str:
    value_ = int(value)
    for code, name in allowed:
        if value_ == code:
            return name
    raise _unsupported_model(f"unsupported {category} value {value_}")


def _enum_features(
    values: Any,
    allowed: Sequence[tuple[int, str]],
    category: str,
    /,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                _enum_feature(value, allowed, category)
                for value in np.asarray(values).reshape(-1)
            }
        )
    )


def _contact_feature_name(
    first: str,
    second: str,
    order: dict[str, int],
    /,
) -> str:
    return "-".join(sorted((first, second), key=order.__getitem__))


def _prepare_feature_manifest(mujoco: Any, model: Any, /) -> MJXPreparedModelManifest:
    integrators = (
        (int(mujoco.mjtIntegrator.mjINT_EULER), "euler"),
        (int(mujoco.mjtIntegrator.mjINT_RK4), "rk4"),
        (int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST), "implicitfast"),
    )
    solvers = (
        (int(mujoco.mjtSolver.mjSOL_CG), "cg"),
        (int(mujoco.mjtSolver.mjSOL_NEWTON), "newton"),
    )
    cones = (
        (int(mujoco.mjtCone.mjCONE_PYRAMIDAL), "pyramidal"),
        (int(mujoco.mjtCone.mjCONE_ELLIPTIC), "elliptic"),
    )
    jacobians = (
        (int(mujoco.mjtJacobian.mjJAC_DENSE), "dense"),
        (int(mujoco.mjtJacobian.mjJAC_SPARSE), "sparse"),
        (int(mujoco.mjtJacobian.mjJAC_AUTO), "auto"),
    )
    joints = (
        (int(mujoco.mjtJoint.mjJNT_FREE), "free"),
        (int(mujoco.mjtJoint.mjJNT_BALL), "ball"),
        (int(mujoco.mjtJoint.mjJNT_SLIDE), "slide"),
        (int(mujoco.mjtJoint.mjJNT_HINGE), "hinge"),
    )
    geoms = (
        (int(mujoco.mjtGeom.mjGEOM_PLANE), "plane"),
        (int(mujoco.mjtGeom.mjGEOM_HFIELD), "hfield"),
        (int(mujoco.mjtGeom.mjGEOM_SPHERE), "sphere"),
        (int(mujoco.mjtGeom.mjGEOM_CAPSULE), "capsule"),
        (int(mujoco.mjtGeom.mjGEOM_ELLIPSOID), "ellipsoid"),
        (int(mujoco.mjtGeom.mjGEOM_CYLINDER), "cylinder"),
        (int(mujoco.mjtGeom.mjGEOM_BOX), "box"),
        (int(mujoco.mjtGeom.mjGEOM_MESH), "mesh"),
    )
    biases = (
        (int(mujoco.mjtBias.mjBIAS_NONE), "none"),
        (int(mujoco.mjtBias.mjBIAS_AFFINE), "affine"),
        (int(mujoco.mjtBias.mjBIAS_MUSCLE), "muscle"),
    )
    dynamics = (
        (int(mujoco.mjtDyn.mjDYN_NONE), "none"),
        (int(mujoco.mjtDyn.mjDYN_INTEGRATOR), "integrator"),
        (int(mujoco.mjtDyn.mjDYN_FILTER), "filter"),
        (int(mujoco.mjtDyn.mjDYN_FILTEREXACT), "filterexact"),
        (int(mujoco.mjtDyn.mjDYN_MUSCLE), "muscle"),
    )
    gains = (
        (int(mujoco.mjtGain.mjGAIN_FIXED), "fixed"),
        (int(mujoco.mjtGain.mjGAIN_AFFINE), "affine"),
        (int(mujoco.mjtGain.mjGAIN_MUSCLE), "muscle"),
    )
    transmissions = (
        (int(mujoco.mjtTrn.mjTRN_JOINT), "joint"),
        (int(mujoco.mjtTrn.mjTRN_JOINTINPARENT), "joint-in-parent"),
        (int(mujoco.mjtTrn.mjTRN_TENDON), "tendon"),
        (int(mujoco.mjtTrn.mjTRN_SITE), "site"),
    )
    equalities = (
        (int(mujoco.mjtEq.mjEQ_CONNECT), "connect"),
        (int(mujoco.mjtEq.mjEQ_WELD), "weld"),
        (int(mujoco.mjtEq.mjEQ_JOINT), "joint"),
        (int(mujoco.mjtEq.mjEQ_TENDON), "tendon"),
    )
    wraps = (
        (int(mujoco.mjtWrap.mjWRAP_JOINT), "joint"),
        (int(mujoco.mjtWrap.mjWRAP_PULLEY), "pulley"),
        (int(mujoco.mjtWrap.mjWRAP_SITE), "site"),
        (int(mujoco.mjtWrap.mjWRAP_SPHERE), "sphere"),
        (int(mujoco.mjtWrap.mjWRAP_CYLINDER), "cylinder"),
    )
    sensors = (
        (int(mujoco.mjtSensor.mjSENS_MAGNETOMETER), "magnetometer"),
        (int(mujoco.mjtSensor.mjSENS_CAMPROJECTION), "camera-projection"),
        (int(mujoco.mjtSensor.mjSENS_RANGEFINDER), "rangefinder"),
        (int(mujoco.mjtSensor.mjSENS_JOINTPOS), "joint-position"),
        (int(mujoco.mjtSensor.mjSENS_TENDONPOS), "tendon-position"),
        (int(mujoco.mjtSensor.mjSENS_ACTUATORPOS), "actuator-position"),
        (int(mujoco.mjtSensor.mjSENS_BALLQUAT), "ball-quaternion"),
        (int(mujoco.mjtSensor.mjSENS_FRAMEPOS), "frame-position"),
        (int(mujoco.mjtSensor.mjSENS_FRAMEXAXIS), "frame-x-axis"),
        (int(mujoco.mjtSensor.mjSENS_FRAMEYAXIS), "frame-y-axis"),
        (int(mujoco.mjtSensor.mjSENS_FRAMEZAXIS), "frame-z-axis"),
        (int(mujoco.mjtSensor.mjSENS_FRAMEQUAT), "frame-quaternion"),
        (int(mujoco.mjtSensor.mjSENS_SUBTREECOM), "subtree-com"),
        (int(mujoco.mjtSensor.mjSENS_CLOCK), "clock"),
        (int(mujoco.mjtSensor.mjSENS_VELOCIMETER), "velocimeter"),
        (int(mujoco.mjtSensor.mjSENS_GYRO), "gyro"),
        (int(mujoco.mjtSensor.mjSENS_JOINTVEL), "joint-velocity"),
        (int(mujoco.mjtSensor.mjSENS_TENDONVEL), "tendon-velocity"),
        (int(mujoco.mjtSensor.mjSENS_ACTUATORVEL), "actuator-velocity"),
        (int(mujoco.mjtSensor.mjSENS_BALLANGVEL), "ball-angular-velocity"),
        (int(mujoco.mjtSensor.mjSENS_FRAMELINVEL), "frame-linear-velocity"),
        (int(mujoco.mjtSensor.mjSENS_FRAMEANGVEL), "frame-angular-velocity"),
        (int(mujoco.mjtSensor.mjSENS_SUBTREELINVEL), "subtree-linear-velocity"),
        (int(mujoco.mjtSensor.mjSENS_SUBTREEANGMOM), "subtree-angular-momentum"),
        (int(mujoco.mjtSensor.mjSENS_TOUCH), "touch"),
        (int(mujoco.mjtSensor.mjSENS_CONTACT), "contact"),
        (int(mujoco.mjtSensor.mjSENS_ACCELEROMETER), "accelerometer"),
        (int(mujoco.mjtSensor.mjSENS_FORCE), "force"),
        (int(mujoco.mjtSensor.mjSENS_TORQUE), "torque"),
        (int(mujoco.mjtSensor.mjSENS_ACTUATORFRC), "actuator-force"),
        (int(mujoco.mjtSensor.mjSENS_JOINTACTFRC), "joint-actuator-force"),
        (int(mujoco.mjtSensor.mjSENS_TENDONACTFRC), "tendon-actuator-force"),
        (int(mujoco.mjtSensor.mjSENS_FRAMELINACC), "frame-linear-acceleration"),
        (int(mujoco.mjtSensor.mjSENS_FRAMEANGACC), "frame-angular-acceleration"),
    )
    enabled = (
        (int(mujoco.mjtEnableBit.mjENBL_INVDISCRETE), "inverse-discrete"),
        (int(mujoco.mjtEnableBit.mjENBL_SLEEP), "sleep"),
    )

    integrator = _enum_feature(model.opt.integrator, integrators, "integrator")
    solver = _enum_feature(model.opt.solver, solvers, "solver")
    cone = _enum_feature(model.opt.cone, cones, "friction cone")
    jacobian = _enum_feature(model.opt.jacobian, jacobians, "Jacobian mode")
    if int(model.nflex):
        raise _unsupported_model("flexible bodies are not supported by MJX-JAX")
    if int(model.npluginstate):
        raise _unsupported_model("plugin state is outside the closed feature manifest")
    if integrator == "implicitfast" and (
        float(model.opt.density) > 0.0
        or float(model.opt.viscosity) > 0.0
        or np.any(np.asarray(model.opt.wind) != 0.0)
    ):
        raise _unsupported_model("implicitfast with fluid drag is not supported")

    enabled_features = tuple(
        name for bit, name in enabled if int(model.opt.enableflags) & bit
    )
    allowed_enable_mask = sum(bit for bit, _ in enabled)
    unknown_enable_mask = int(model.opt.enableflags) & ~allowed_enable_mask
    if unknown_enable_mask:
        raise _unsupported_model(
            f"unsupported enabled-feature bitmask {unknown_enable_mask}"
        )

    geom_names = tuple(
        _enum_feature(value, geoms, "geometry") for value in model.geom_type
    )
    geom_order = {name: code for code, name in geoms}
    contact_features: set[str] = set()
    for first in range(int(model.ngeom)):
        for second in range(first + 1, int(model.ngeom)):
            can_collide = (
                int(model.geom_contype[first]) & int(model.geom_conaffinity[second])
            ) or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
            if can_collide:
                contact_features.add(
                    _contact_feature_name(
                        geom_names[first], geom_names[second], geom_order
                    )
                )
    for pair_index in range(int(model.npair)):
        first = int(model.pair_geom1[pair_index])
        second = int(model.pair_geom2[pair_index])
        contact_features.add(
            _contact_feature_name(geom_names[first], geom_names[second], geom_order)
        )
    unsupported_contacts = contact_features.difference(_MJX_JAX_CONTACT_FEATURES)
    if unsupported_contacts:
        raise _unsupported_model(
            "unsupported collision pairs: " + ", ".join(sorted(unsupported_contacts))
        )

    no_margin = {"mesh", "hfield"}
    for geom_index, geom_name in enumerate(geom_names):
        if geom_name in no_margin and float(model.geom_margin[geom_index]) != 0.0:
            raise _unsupported_model(
                f"{geom_name} margin/gap is not supported by MJX-JAX"
            )
    for pair_index in range(int(model.npair)):
        first_name = geom_names[int(model.pair_geom1[pair_index])]
        second_name = geom_names[int(model.pair_geom2[pair_index])]
        if (
            no_margin.intersection((first_name, second_name))
            and float(model.pair_margin[pair_index]) != 0.0
        ):
            raise _unsupported_model(
                f"{first_name}-{second_name} margin/gap is not supported"
            )
    if cone == "elliptic" and np.any(np.asarray(model.geom_condim) == 1):
        raise _unsupported_model("elliptic contacts with condim=1 are not supported")

    contact_sensor = int(mujoco.mjtSensor.mjSENS_CONTACT)
    contact_sensor_mask = np.asarray(model.sensor_type) == contact_sensor
    if np.any(contact_sensor_mask):
        object_types = set(
            np.concatenate(
                (
                    np.asarray(model.sensor_objtype)[contact_sensor_mask],
                    np.asarray(model.sensor_reftype)[contact_sensor_mask],
                )
            ).tolist()
        )
        if int(mujoco.mjtObj.mjOBJ_SITE) in set(
            np.asarray(model.sensor_objtype)[contact_sensor_mask].tolist()
        ):
            raise _unsupported_model("contact sensors with site matching are unsupported")
        if int(mujoco.mjtObj.mjOBJ_BODY) in object_types:
            raise _unsupported_model("contact sensors with body matching are unsupported")
        if int(mujoco.mjtObj.mjOBJ_XBODY) in object_types:
            raise _unsupported_model(
                "contact sensors with subtree matching are unsupported"
            )
        if np.any(np.asarray(model.sensor_intprm)[contact_sensor_mask, 1] == 3):
            raise _unsupported_model(
                "contact sensors with net-force reduction are unsupported"
            )

    return MJXPreparedModelManifest(
        integrator=integrator,
        solver=solver,
        cone=cone,
        jacobian=jacobian,
        joint_types=_enum_features(model.jnt_type, joints, "joint"),
        geom_types=tuple(sorted(set(geom_names))),
        contact_features=tuple(sorted(contact_features)),
        actuator_bias_types=_enum_features(
            model.actuator_biastype, biases, "actuator bias"
        ),
        actuator_dynamics_types=_enum_features(
            model.actuator_dyntype, dynamics, "actuator dynamics"
        ),
        actuator_gain_types=_enum_features(
            model.actuator_gaintype, gains, "actuator gain"
        ),
        actuator_transmission_types=_enum_features(
            model.actuator_trntype, transmissions, "actuator transmission"
        ),
        equality_types=_enum_features(model.eq_type, equalities, "equality"),
        sensor_types=_enum_features(model.sensor_type, sensors, "sensor"),
        tendon_wrap_types=_enum_features(model.wrap_type, wraps, "tendon wrap"),
        enabled_features=enabled_features,
    )


def _plant_identities(
    mujoco: Any,
    model: Any,
    versions: Sequence[tuple[str, str]],
    feature_manifest: MJXPreparedModelManifest,
    /,
) -> tuple[RoboticsProjectionProvenance, SemanticProvenance, NumericRevision]:
    model_buffer = np.empty(int(mujoco.mj_sizeModel(model)), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=model_buffer)
    digest = hashlib.sha256(model_buffer.tobytes()).hexdigest()
    version_by_name = dict(versions)
    provenance = RoboticsProjectionProvenance(
        model=f"mujoco-mjb-sha256:{digest}",
        compiler=f"mujoco:{version_by_name['mujoco']}",
        provider=f"mujoco-mjx:{version_by_name['mujoco-mjx']}",
        asset=f"compiled-assets-sha256:{digest}",
        unit_system="MuJoCo SI base units",
        frame_convention="MuJoCo world and body-local frames",
    )
    semantic = SemanticProvenance(
        {
            "kind": "mjx-complete-state-discrete-plant",
            "feature_manifest": feature_manifest,
            "unit_system": provenance.unit_system,
            "frame_convention": provenance.frame_convention,
        },
        resource_ids={
            "asset": provenance.asset,
            "compiler": provenance.compiler,
            "provider": provenance.provider,
        },
    )
    numeric = NumericRevision(semantic, {"compiled_mjb": model_buffer})
    return provenance, semantic, numeric


def _object_name(
    mujoco: Any,
    model: Any,
    object_type: Any,
    index: int,
    fallback: str,
    /,
) -> str:
    name = mujoco.mj_id2name(model, object_type, index)
    return fallback if name is None or not str(name) else str(name)


def _muscle_actuator_manifest(
    mujoco: Any, model: Any, /
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[int, ...]]:
    gain = np.asarray(model.actuator_gaintype)
    bias = np.asarray(model.actuator_biastype)
    dynamics = np.asarray(model.actuator_dyntype)
    muscle_mask = (
        (gain == int(mujoco.mjtGain.mjGAIN_MUSCLE))
        & (bias == int(mujoco.mjtBias.mjBIAS_MUSCLE))
        & (dynamics == int(mujoco.mjtDyn.mjDYN_MUSCLE))
    )
    actuator_indices = tuple(int(index) for index in np.flatnonzero(muscle_mask).tolist())
    activation_addresses = np.asarray(model.actuator_actadr)
    activation_counts = np.asarray(model.actuator_actnum)
    length_ranges = np.asarray(model.actuator_lengthrange)
    names: list[str] = []
    activation_indices: list[int] = []
    for actuator_index in actuator_indices:
        if (
            int(activation_counts[actuator_index]) != 1
            or int(activation_addresses[actuator_index]) < 0
        ):
            raise _unsupported_model(
                "a built-in muscle actuator must own exactly one activation state"
            )
        length_range = length_ranges[actuator_index]
        if (
            length_range.shape != (2,)
            or not np.all(np.isfinite(length_range))
            or not float(length_range[1]) > float(length_range[0])
        ):
            raise _unsupported_model(
                "a built-in muscle actuator requires a finite increasing length range"
            )
        names.append(
            _object_name(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_index,
                f"actuator-{actuator_index}",
            )
        )
        activation_indices.append(int(activation_addresses[actuator_index]))
    if len(set(names)) != len(names):
        raise _unsupported_model("compiled muscle actuator names must be unique")
    if len(set(activation_indices)) != len(activation_indices):
        raise _unsupported_model(
            "compiled built-in muscles must own distinct activation states"
        )
    return tuple(names), actuator_indices, tuple(activation_indices)


def _validate_muscle_data_fields(data: Any, actuator_count: int, /) -> None:
    expected = (actuator_count,)
    fields = (
        ("actuator_length", data.actuator_length),
        ("actuator_velocity", data._impl.actuator_velocity),
        ("actuator_force", data.actuator_force),
    )
    for name, field in fields:
        if field.shape != expected:
            raise TypeError(
                f"Canonical MJX Data {name} must have intrinsic shape {expected}."
            )


def _joint_map(
    mujoco: Any,
    model: Any,
    provenance: RoboticsProjectionProvenance,
    /,
    *,
    kind: Literal["qpos", "qvel"],
) -> RoboticsProjectionMap:
    size = int(model.nq if kind == "qpos" else model.nv)
    addresses = model.jnt_qposadr if kind == "qpos" else model.jnt_dofadr
    entries: list[RoboticsIndexEntry] = []
    for joint_index in range(int(model.njnt)):
        start = int(addresses[joint_index])
        stop = (
            int(addresses[joint_index + 1]) if joint_index + 1 < int(model.njnt) else size
        )
        name = _object_name(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_index,
            f"joint-{joint_index}",
        )
        entries.append(RoboticsIndexEntry(name, start, stop))
    return RoboticsProjectionMap(kind, size, entries, provenance)


def _control_map(
    mujoco: Any,
    model: Any,
    provenance: RoboticsProjectionProvenance,
    /,
) -> RoboticsProjectionMap:
    entries = tuple(
        RoboticsIndexEntry(
            _object_name(
                mujoco,
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                index,
                f"actuator-{index}",
            ),
            index,
            index + 1,
        )
        for index in range(int(model.nu))
    )
    return RoboticsProjectionMap("control", int(model.nu), entries, provenance)


def _full_observation_map(
    mujoco: Any,
    model: Any,
    qpos_map: RoboticsProjectionMap,
    qvel_map: RoboticsProjectionMap,
    control_map: RoboticsProjectionMap,
    provenance: RoboticsProjectionProvenance,
    /,
) -> RoboticsProjectionMap:
    entries: list[RoboticsIndexEntry] = []
    offset = 0
    for prefix, index_map in (
        ("qpos", qpos_map),
        ("qvel", qvel_map),
        ("control", control_map),
    ):
        entries.extend(
            RoboticsIndexEntry(
                f"{prefix}/{entry.name}",
                offset + entry.start,
                offset + entry.stop,
            )
            for entry in index_map.entries
        )
        offset += index_map.size
    for sensor_index in range(int(model.nsensor)):
        start = offset + int(model.sensor_adr[sensor_index])
        stop = start + int(model.sensor_dim[sensor_index])
        name = _object_name(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_index,
            f"sensor-{sensor_index}",
        )
        entries.append(RoboticsIndexEntry(f"sensor/{name}", start, stop))
    size = offset + int(model.nsensordata)
    return RoboticsProjectionMap("observation", size, entries, provenance)


def _observation_projection(
    data: Any,
    request: MJXObservationRequest,
    qpos_map: RoboticsProjectionMap,
    qvel_map: RoboticsProjectionMap,
    control_map: RoboticsProjectionMap,
    full_map: RoboticsProjectionMap,
    /,
) -> tuple[Any, RoboticsProjectionMap]:
    arrays: list[Any] = []
    entries: list[RoboticsIndexEntry] = []
    offset = 0
    source = (
        (request.qpos, "qpos", data.qpos, qpos_map.entries),
        (request.qvel, "qvel", data.qvel, qvel_map.entries),
        (request.control, "control", data.ctrl, control_map.entries),
    )
    for selected, prefix, array, source_entries in source:
        if selected:
            arrays.append(array)
            entries.extend(
                RoboticsIndexEntry(
                    f"{prefix}/{entry.name}",
                    offset + entry.start,
                    offset + entry.stop,
                )
                for entry in source_entries
            )
            offset += int(array.shape[-1])
    if request.sensors:
        arrays.append(data.sensordata)
        sensor_entries = tuple(
            entry for entry in full_map.entries if entry.name.startswith("sensor/")
        )
        sensor_base = qpos_map.size + qvel_map.size + control_map.size
        entries.extend(
            RoboticsIndexEntry(
                entry.name,
                offset + entry.start - sensor_base,
                offset + entry.stop - sensor_base,
            )
            for entry in sensor_entries
        )
        offset += int(data.sensordata.shape[-1])
    values = arrays[0] if len(arrays) == 1 else jnp.concatenate(arrays, axis=-1)
    index_map = RoboticsProjectionMap("observation", offset, entries, full_map.provenance)
    return values, index_map


def _validate_provider_api(mjx: Any, /) -> None:
    callables = (
        ("put_model", mjx.put_model),
        ("make_data", mjx.make_data),
        ("step", mjx.step),
        ("forward", mjx.forward),
    )
    missing = tuple(name for name, function in callables if not callable(function))
    if missing:
        raise BackendUnavailableError(
            "mjx-jax",
            "robotics.step",
            "the qualified MJX 3.12 callable API",
            "missing required callables: " + ", ".join(missing),
        )


def prepare_mjx_adapter(
    model: Any,
    /,
    *,
    device: Any | None = None,
    case_ndim: int = 0,
) -> MJXAdapter:
    """Prepare one MJX-JAX complete-state discrete plant."""
    availability = mjx_availability()
    mujoco = import_backend_module(availability, "robotics.step", "mujoco")
    mjx = import_backend_module(availability, "robotics.step", "mujoco.mjx")
    _validate_provider_api(mjx)
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be an already compiled mujoco.MjModel.")

    feature_manifest = _prepare_feature_manifest(mujoco, model)
    (
        muscle_actuator_names,
        muscle_actuator_indices,
        muscle_activation_indices,
    ) = _muscle_actuator_manifest(mujoco, model)
    provenance, semantic, numeric = _plant_identities(
        mujoco, model, availability.versions, feature_manifest
    )
    device_model = mjx.put_model(model, device=device, impl="jax")
    canonical = mjx.make_data(device_model, device=device, impl="jax")
    if not isinstance(canonical, mjx.Data) or canonical.impl != mjx.Impl.JAX:
        raise TypeError("make_data(model) must return a complete MJX-JAX Data PyTree.")
    opaque = mjx.forward(device_model, canonical)
    if not isinstance(opaque, mjx.Data) or opaque.impl != mjx.Impl.JAX:
        raise TypeError("forward(model, data) must return a complete MJX-JAX Data.")
    if muscle_actuator_names:
        _validate_muscle_data_fields(opaque, int(model.nu))

    qpos_map = _joint_map(mujoco, model, provenance, kind="qpos")
    qvel_map = _joint_map(mujoco, model, provenance, kind="qvel")
    control_map = _control_map(mujoco, model, provenance)
    observation_map = _full_observation_map(
        mujoco, model, qpos_map, qvel_map, control_map, provenance
    )
    devices = opaque.qpos.devices()
    if len(devices) != 1:
        raise ValueError("MJX adapter state must reside on exactly one JAX device.")
    device_name = next(iter(devices)).platform
    return MJXAdapter(
        model=device_model,
        data=opaque,
        qpos_map=qpos_map,
        qvel_map=qvel_map,
        control_map=control_map,
        observation_map=observation_map,
        feature_manifest=feature_manifest,
        provenance=provenance,
        muscle_actuator_names=muscle_actuator_names,
        muscle_actuator_indices=muscle_actuator_indices,
        muscle_activation_indices=muscle_activation_indices,
        semantic_provenance=semantic,
        numeric_revision=numeric,
        device=device_name,
        dtype=opaque.qpos.dtype,
        case_ndim=case_ndim,
        mjx_module=mjx,
    )


def prepare_mjx_muscle_projection(
    adapter: MJXAdapter,
    names: Sequence[str] | None = None,
    /,
) -> MJXPreparedMuscleProjection:
    """Prepare fixed gathers/scatters for named or all built-in muscles."""

    if not isinstance(adapter, MJXAdapter):
        raise TypeError("adapter must be MJXAdapter.")
    return MJXMuscleProjectionPlan(names).prepare(adapter)


__all__ = [
    "MJXAdapter",
    "MJX_JAX_BACKEND_CAPABILITIES",
    "MJX_JAX_PROFILE",
    "MJXObservation",
    "MJXMuscleProjectionPlan",
    "MJXMuscleSnapshot",
    "MJXObservationRequest",
    "MJXPreparedModelManifest",
    "MJXPreparedMuscleProjection",
    "MJXRefreshResult",
    "MJXState",
    "MJX_WARP_PROFILE",
    "mjx_availability",
    "prepare_mjx_adapter",
    "prepare_mjx_muscle_projection",
]
