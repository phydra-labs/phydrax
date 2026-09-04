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

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends._availability import import_backend_module, probe_backend
from ...backends._types import (
    BackendAvailability,
    BackendCapabilities,
    BackendUnavailableError,
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
MJXStepObservationMode: TypeAlias = Literal["none", "pre", "post", "both"]

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
        dtypes=_MJX_DTYPES,
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
    solvers: Sequence[str], contact_features: Sequence[str], /
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
                _MJX_JAX_DEVICES,
                differentiability="conditional",
                solvers=solvers,
                contact_features=contact_features,
            ),
            _capability(
                "sensors",
                "MJXAdapter.observe/MJXAdapter.refresh",
                _MJX_JAX_DEVICES,
                differentiability="conditional",
            ),
            _unsupported_capability("model-batching", "MJXAdapter", no_callable),
            _unsupported_capability("jit", "MJXAdapter", no_callable),
            _unsupported_capability("vmap", "MJXAdapter", no_callable),
            _unsupported_capability("jvp", "MJXAdapter", no_callable),
            _unsupported_capability("vjp", "MJXAdapter", no_callable),
        ),
    )


MJX_JAX_PROFILE = _jax_profile(_MJX_JAX_SOLVERS, _MJX_JAX_CONTACT_FEATURES)

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
    if mujoco_release[:2] != _MJX_PROVIDER_MINOR or mjx_release[:2] != _MJX_PROVIDER_MINOR:
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


class MJXArrayLeafSpec(StrictModule, NonTrainableState):
    """Intrinsic contract for one canonical ``mjx.Data`` array leaf."""

    shape: tuple[int, ...] = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    devices: tuple[str, ...] = eqx.field(static=True)
    initial_finite: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: Any,
        devices: Sequence[str],
        initial_finite: bool,
    ):
        self.shape = tuple(int(size) for size in shape)
        self.dtype = str(dtype)
        self.devices = tuple(devices)
        self.initial_finite = bool(initial_finite)


class MJXDataSchema(StrictModule, NonTrainableState):
    """Complete intrinsic PyTree contract independent of leading case axes."""

    treedef: Any = eqx.field(static=True)
    leaves: tuple[MJXArrayLeafSpec, ...] = eqx.field(static=True)
    qpos_shape: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        treedef: Any,
        leaves: Sequence[MJXArrayLeafSpec],
        qpos_shape: Sequence[int],
    ):
        self.treedef = treedef
        self.leaves = tuple(leaves)
        self.qpos_shape = tuple(int(size) for size in qpos_shape)

    def validate(self, data: Any, /) -> tuple[int, ...]:
        leaves, treedef = jax.tree_util.tree_flatten(data)
        if treedef != self.treedef or len(leaves) != len(self.leaves):
            raise TypeError("MJX Data must retain the canonical complete PyTree structure.")
        qpos_shape = tuple(int(size) for size in data.qpos.shape)
        intrinsic_rank = len(self.qpos_shape)
        if len(qpos_shape) < intrinsic_rank or (
            intrinsic_rank and qpos_shape[-intrinsic_rank:] != self.qpos_shape
        ):
            raise ValueError(
                "MJX Data qpos does not end in the canonical intrinsic shape "
                f"{self.qpos_shape}; got {qpos_shape}."
            )
        case_shape = qpos_shape[: len(qpos_shape) - intrinsic_rank]
        for index, (leaf, spec) in enumerate(zip(leaves, self.leaves, strict=True)):
            if not isinstance(leaf, jax.Array):
                raise TypeError(f"MJX Data leaf {index} is not a canonical JAX array.")
            expected_shape = case_shape + spec.shape
            if tuple(int(size) for size in leaf.shape) != expected_shape:
                raise ValueError(
                    f"MJX Data leaf {index} must have case/intrinsic shape "
                    f"{expected_shape}; got {leaf.shape}."
                )
            if str(leaf.dtype) != spec.dtype:
                raise TypeError(
                    f"MJX Data leaf {index} must have dtype {spec.dtype}; "
                    f"got {leaf.dtype}."
                )
            if (
                not isinstance(leaf, jax.core.Tracer)
                and _array_devices(leaf) != spec.devices
            ):
                raise ValueError(
                    f"MJX Data leaf {index} is on a noncanonical device set."
                )
        return case_shape


class MJXState(StrictModule, NonTrainableState):
    """Adapter-owned complete ``mjx.Data`` plus derived-field epochs."""

    opaque: Any
    epoch: Any
    sensor_epoch: Any
    rollback_source: Any
    provenance: RoboticsProjectionProvenance = eqx.field(static=True)
    _owner: object = eqx.field(static=True)

    def __init__(
        self,
        opaque: Any,
        epoch: Any,
        sensor_epoch: Any,
        rollback_source: Any,
        provenance: RoboticsProjectionProvenance,
        owner: object,
        /,
    ):
        self.opaque = opaque
        self.epoch = jnp.asarray(epoch, dtype=jnp.int32)
        self.sensor_epoch = jnp.asarray(sensor_epoch, dtype=jnp.int32)
        self.rollback_source = jnp.asarray(rollback_source, dtype=jnp.bool_)
        self.provenance = provenance
        self._owner = owner


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


class MJXStepResult(StrictModule, NonTrainableState):
    """Candidate and casewise fail-closed accepted states from one step."""

    candidate_state: MJXState
    accepted_state: MJXState
    evidence: RoboticsOperationEvidence
    pre_step_observation: MJXObservation | None
    post_step_observation: MJXObservation | None
    observation_mode: MJXStepObservationMode = eqx.field(static=True)

    @property
    def status(self) -> Any:
        return self.evidence.status

    @property
    def successful(self) -> Any:
        return self.evidence.successful

    @property
    def state(self) -> MJXState:
        return self.accepted_state

    @property
    def rolled_back(self) -> Any:
        return ~self.evidence.successful


class MJXRefreshResult(StrictModule, NonTrainableState):
    """Candidate and accepted refreshed states with a requested observation."""

    candidate_state: MJXState
    accepted_state: MJXState
    observation: MJXObservation
    evidence: RoboticsOperationEvidence
    rollback_source_refreshed: Any

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
    raw_force_sign: str = eqx.field(
        static=True, default="negative-is-pulling-tension"
    )
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

    def __init__(
        self, adapter: MJXAdapter, plan: MJXMuscleProjectionPlan, /
    ):
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

        if isinstance(complete_control, RoboticsProjection):
            if complete_control.index_map.kind != self.adapter.control_map.kind:
                raise ValueError(
                    "Complete control projection must have kind 'control'."
                )
            if complete_control.provenance != self.adapter.provenance:
                raise ValueError("Control provenance does not match this MJX model.")
            if (
                complete_control.index_map.size != self.adapter.control_map.size
                or complete_control.index_map.name_to_range
                != self.adapter.control_map.name_to_range
            ):
                raise ValueError("Control layout does not match this MJX model.")
            base_values = complete_control.values
        else:
            base_values = complete_control
        base = jnp.asarray(
            base_values, dtype=self.adapter.initial_state.opaque.ctrl.dtype
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
                "independent_excitation must have complete shape "
                f"{expected_excitation}."
            )
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values) | (values < 0.0) | (values > 1.0)),
            "MuJoCo independent muscle excitation must be finite and lie in [0, 1].",
        )
        indices = jnp.asarray(self.actuator_indices, dtype=jnp.int32)
        scattered = base.at[..., indices].set(values)
        return RoboticsProjection(scattered, self.adapter.control_map)

    def snapshot(self, state: MJXState | None = None, /) -> MJXMuscleSnapshot:
        """Gather activation and forward-derived provider muscle quantities."""

        resolved, _ = self.adapter._state(state)
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





class MJXAdapter(StrictModule, NonTrainableState):
    """Prepared MJX-JAX model and its closed integrity boundary."""

    model: Any
    initial_state: MJXState
    qpos_map: RoboticsProjectionMap
    qvel_map: RoboticsProjectionMap
    control_map: RoboticsProjectionMap
    observation_map: RoboticsProjectionMap
    profile: RoboticsBackendProfile
    feature_manifest: MJXPreparedModelManifest
    data_schema: MJXDataSchema
    provenance: RoboticsProjectionProvenance
    muscle_actuator_names: tuple[str, ...] = eqx.field(static=True)
    muscle_actuator_indices: tuple[int, ...] = eqx.field(static=True)
    muscle_activation_indices: tuple[int, ...] = eqx.field(static=True)
    device: str = eqx.field(static=True)
    dtype: str = eqx.field(static=True)
    _mjx: Any = eqx.field(static=True)
    _owner: object = eqx.field(static=True)

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
        data_schema: MJXDataSchema,
        provenance: RoboticsProjectionProvenance,
        muscle_actuator_names: tuple[str, ...],
        muscle_actuator_indices: tuple[int, ...],
        muscle_activation_indices: tuple[int, ...],
        device: str,
        dtype: str,
        mjx_module: Any,
        owner: object,
    ):
        case_shape = data_schema.validate(data)
        epoch = jnp.zeros(case_shape, dtype=jnp.int32)
        self.model = model
        self.initial_state = MJXState(
            data, epoch, epoch, jnp.zeros(case_shape, dtype=jnp.bool_), provenance, owner
        )
        self.qpos_map = qpos_map
        self.qvel_map = qvel_map
        self.control_map = control_map
        self.observation_map = observation_map
        self.profile = _jax_profile(
            (feature_manifest.solver,), feature_manifest.contact_features
        )
        self.feature_manifest = feature_manifest
        self.data_schema = data_schema
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
        self.device = str(device).lower()
        self.dtype = np.dtype(dtype).name
        self._mjx = mjx_module
        self._owner = owner

    def _state(
        self, state: MJXState | None, /
    ) -> tuple[MJXState, tuple[int, ...]]:
        resolved = self.initial_state if state is None else state
        if not isinstance(resolved, MJXState):
            raise TypeError("state must be MJXState.")
        if resolved._owner is not self._owner:
            raise ValueError("MJX state belongs to a different prepared adapter.")
        if resolved.provenance != self.provenance:
            raise ValueError("MJX state provenance does not match this adapter.")
        if not isinstance(resolved.opaque, self._mjx.Data):
            raise TypeError("MJX state must retain a complete mjx.Data PyTree.")
        if resolved.opaque.impl != self._mjx.Impl.JAX:
            raise TypeError("MJX state must retain the prepared JAX implementation.")
        case_shape = self.data_schema.validate(resolved.opaque)
        if resolved.rollback_source.shape != case_shape:
            raise ValueError(
                "MJX rollback-source evidence must have exactly the data case axes."
            )
        if resolved.epoch.shape != case_shape or resolved.sensor_epoch.shape != case_shape:
            raise ValueError("MJX state epochs must have exactly the data case axes.")
        return resolved, case_shape

    def qpos(self, state: MJXState | None = None, /) -> RoboticsProjection:
        resolved, _ = self._state(state)
        return RoboticsProjection(resolved.opaque.qpos, self.qpos_map)

    def qvel(self, state: MJXState | None = None, /) -> RoboticsProjection:
        resolved, _ = self._state(state)
        return RoboticsProjection(resolved.opaque.qvel, self.qvel_map)

    def control(self, state: MJXState | None = None, /) -> RoboticsProjection:
        resolved, _ = self._state(state)
        return RoboticsProjection(resolved.opaque.ctrl, self.control_map)

    def prepare_muscle_projection(
        self, names: Sequence[str] | None = None, /
    ) -> MJXPreparedMuscleProjection:
        """Prepare fixed gathers for named or all compiled built-in muscles."""

        return MJXMuscleProjectionPlan(names).prepare(self)
    def observe(
        self,
        state: MJXState | None = None,
        request: MJXObservationRequest | None = None,
        /,
    ) -> MJXObservation:
        """Project requested fields with freshness derived from state epochs."""
        resolved, _ = self._state(state)
        request_ = MJXObservationRequest() if request is None else request
        if not isinstance(request_, MJXObservationRequest):
            raise TypeError("request must be MJXObservationRequest.")
        values, index_map = _observation_projection(
            resolved.opaque,
            request_,
            self.qpos_map,
            self.qvel_map,
            self.control_map,
            self.observation_map,
        )
        sample_epoch = resolved.sensor_epoch if request_.sensors else resolved.epoch
        projection = RoboticsProjection(
            values,
            index_map,
            state_epoch=resolved.epoch,
            sample_epoch=sample_epoch,
        )
        finite = jnp.all(jnp.isfinite(values), axis=-1)
        fresh = projection.freshness
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
            implementation="MJXAdapter.observe",
            device=self.device,
            dtype=self.dtype,
            detail="sensor samples are current only when sensor_epoch equals state epoch",
        )
        return MJXObservation(projection, request_, evidence)

    def _requested_observation(
        self,
        refresh: MJXRefreshResult,
        freshness: ObservationFreshness,
        /,
    ) -> MJXObservation:
        observation = refresh.observation
        refresh_evidence = refresh.evidence
        observation_evidence = observation.evidence
        evidence = RoboticsOperationEvidence(
            status=jnp.where(
                refresh_evidence.successful,
                observation_evidence.status,
                refresh_evidence.status,
            ).astype(jnp.int32),
            finite=refresh_evidence.finite & observation_evidence.finite,
            backend="mjx-jax",
            operation="sensors",
            implementation="MJXAdapter.step/refresh-observation",
            device=self.device,
            dtype=self.dtype,
            detail=(
                "the requested refresh and observation must both succeed "
                "casewise"
            ),
        )
        return MJXObservation(
            observation.projection,
            observation.request,
            evidence,
            freshness,
        )

    def step(
        self,
        state: MJXState | None = None,
        control: Any | RoboticsProjection | None = None,
        /,
        *,
        observations: MJXStepObservationMode = "none",
        observation_request: MJXObservationRequest | None = None,
    ) -> MJXStepResult:
        """Advance state with optional explicitly refreshed pre/post observations."""
        transaction_state, case_shape = self._state(state)
        if observations not in ("none", "pre", "post", "both"):
            raise ValueError("observations must be 'none', 'pre', 'post', or 'both'.")
        if observation_request is not None and not isinstance(
            observation_request, MJXObservationRequest
        ):
            raise TypeError(
                "observation_request must be MJXObservationRequest or None."
            )

        source_state = transaction_state
        if control is not None:
            if isinstance(control, RoboticsProjection):
                if control.index_map.kind != self.control_map.kind:
                    raise ValueError("Control projection must have kind 'control'.")
                if control.provenance != self.provenance:
                    raise ValueError(
                        "Control projection provenance does not match this adapter."
                    )
                if (
                    control.index_map.size != self.control_map.size
                    or control.index_map.name_to_range
                    != self.control_map.name_to_range
                ):
                    raise ValueError(
                        "Control projection layout does not match this adapter."
                    )
                control_values = control.values
            else:
                control_values = control
            control_array = jnp.asarray(
                control_values, dtype=transaction_state.opaque.ctrl.dtype
            )
            if control_array.shape != transaction_state.opaque.ctrl.shape:
                raise ValueError(
                    "Control must have complete shape "
                    f"{transaction_state.opaque.ctrl.shape}; got {control_array.shape}."
                )
            source_state = MJXState(
                transaction_state.opaque.replace(ctrl=control_array),
                transaction_state.epoch,
                transaction_state.sensor_epoch,
                transaction_state.rollback_source,
                self.provenance,
                self._owner,
            )

        pre_step_observation = None
        if observations in ("pre", "both"):
            pre_refresh = self.refresh(source_state, observation_request)
            source_state = pre_refresh.accepted_state
            pre_step_observation = self._requested_observation(
                pre_refresh,
                "pre-step",
            )
            case_shape = self.data_schema.validate(source_state.opaque)

        source = source_state.opaque
        candidate = _apply_casewise(
            self._mjx.step, self.model, source, len(case_shape)
        )
        self.data_schema.validate(candidate)
        finite = _finite_dynamic_state(candidate, self.data_schema, case_shape)
        accepted = _select_complete_state(finite, source, candidate, case_shape)
        candidate_epoch = source_state.epoch + 1
        accepted_epoch = jnp.where(
            finite, candidate_epoch, source_state.epoch
        ).astype(jnp.int32)
        candidate_state = MJXState(
            candidate,
            candidate_epoch,
            source_state.sensor_epoch,
            ~finite,
            self.provenance,
            self._owner,
        )
        step_accepted_state = MJXState(
            accepted,
            accepted_epoch,
            source_state.sensor_epoch,
            ~finite,
            self.provenance,
            self._owner,
        )
        status = jnp.where(
            finite,
            int(RoboticsOperationStatus.SUCCESS),
            int(RoboticsOperationStatus.NONFINITE),
        ).astype(jnp.int32)
        evidence = RoboticsOperationEvidence(
            status=status,
            finite=finite,
            backend="mjx-jax",
            operation="step",
            implementation="MJXAdapter.step",
            device=self.device,
            dtype=self.dtype,
            detail=(
                "each complete candidate mjx.Data case is accepted only when every "
                "dynamic floating leaf in that case is finite"
            ),
        )
        if observations in ("pre", "both"):
            pre_status = pre_step_observation.evidence.status
            pre_successful = pre_step_observation.successful
            evidence = RoboticsOperationEvidence(
                status=jnp.where(pre_successful, evidence.status, pre_status),
                finite=pre_step_observation.evidence.finite & evidence.finite,
                backend="mjx-jax",
                operation="step",
                implementation="MJXAdapter.step",
                device=self.device,
                dtype=self.dtype,
                detail=(
                    "the requested pre-step refresh and observation and the "
                    "complete step candidate must all succeed casewise"
                ),
            )

        successful = evidence.successful
        accepted_state = MJXState(
            _select_complete_state(
                successful,
                transaction_state.opaque,
                step_accepted_state.opaque,
                case_shape,
            ),
            jnp.where(
                successful,
                step_accepted_state.epoch,
                transaction_state.epoch,
            ),
            jnp.where(
                successful,
                step_accepted_state.sensor_epoch,
                transaction_state.sensor_epoch,
            ),
            jnp.where(
                successful,
                step_accepted_state.rollback_source,
                jnp.ones(case_shape, dtype=jnp.bool_),
            ),
            self.provenance,
            self._owner,
        )

        post_step_observation = None
        if observations in ("post", "both"):
            post_refresh = self.refresh(accepted_state, observation_request)
            post_step_observation = self._requested_observation(
                post_refresh,
                "post-step-refreshed",
            )
            post_status = post_step_observation.evidence.status
            evidence = RoboticsOperationEvidence(
                status=jnp.where(evidence.successful, post_status, evidence.status),
                finite=evidence.finite & post_step_observation.evidence.finite,
                backend="mjx-jax",
                operation="step",
                implementation="MJXAdapter.step",
                device=self.device,
                dtype=self.dtype,
                detail=(
                    "the complete step candidate and its requested post-step "
                    "refresh and observation must all succeed casewise"
                ),
            )
            successful = evidence.successful
            post_state = post_refresh.accepted_state
            accepted_state = MJXState(
                _select_complete_state(
                    successful,
                    transaction_state.opaque,
                    post_state.opaque,
                    case_shape,
                ),
                jnp.where(successful, post_state.epoch, transaction_state.epoch),
                jnp.where(
                    successful,
                    post_state.sensor_epoch,
                    transaction_state.sensor_epoch,
                ),
                jnp.where(
                    successful,
                    post_state.rollback_source,
                    jnp.ones(case_shape, dtype=jnp.bool_),
                ),
                self.provenance,
                self._owner,
            )
        return MJXStepResult(
            candidate_state,
            accepted_state,
            evidence,
            pre_step_observation,
            post_step_observation,
            observations,
        )

    def refresh(
        self,
        state: MJXState | None = None,
        request: MJXObservationRequest | None = None,
        /,
    ) -> MJXRefreshResult:
        """Run ``mjx.forward`` and accept fresh derived fields casewise."""
        source_state, case_shape = self._state(state)
        candidate = _apply_casewise(
            self._mjx.forward, self.model, source_state.opaque, len(case_shape)
        )
        self.data_schema.validate(candidate)
        finite = _finite_dynamic_state(candidate, self.data_schema, case_shape)
        accepted = _select_complete_state(
            finite, source_state.opaque, candidate, case_shape
        )
        candidate_state = MJXState(
            candidate,
            source_state.epoch,
            source_state.epoch,
            jnp.zeros(case_shape, dtype=jnp.bool_),
            self.provenance,
            self._owner,
        )
        accepted_sensor_epoch = jnp.where(
            finite, source_state.epoch, source_state.sensor_epoch
        ).astype(jnp.int32)
        accepted_state = MJXState(
            accepted,
            source_state.epoch,
            accepted_sensor_epoch,
            jnp.where(
                finite,
                jnp.zeros(case_shape, dtype=jnp.bool_),
                source_state.rollback_source,
            ),
            self.provenance,
            self._owner,
        )
        observation = self.observe(accepted_state, request)
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
                "each complete forwarded state is accepted and its sensor epoch is "
                "advanced only when every floating leaf in that case is finite"
            ),
        )
        return MJXRefreshResult(
            candidate_state,
            accepted_state,
            observation,
            evidence,
            finite & source_state.rollback_source,
        )


def _array_devices(array: jax.Array, /) -> tuple[str, ...]:
    return tuple(
        sorted(f"{device.platform}:{device.id}" for device in array.devices())
    )

def _data_schema(data: Any, /) -> MJXDataSchema:
    leaves, treedef = jax.tree_util.tree_flatten(data)
    specs: list[MJXArrayLeafSpec] = []
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, jax.Array):
            raise TypeError(f"Canonical MJX Data leaf {index} is not a JAX array.")
        initial_finite = True
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            initial_finite = bool(np.asarray(jnp.all(jnp.isfinite(leaf))))
        if not initial_finite:
            raise ValueError(
                f"Canonical make_data(model) leaf {index} is initially nonfinite."
            )
        specs.append(
            MJXArrayLeafSpec(
                shape=leaf.shape,
                dtype=leaf.dtype,
                devices=_array_devices(leaf),
                initial_finite=initial_finite,
            )
        )
    return MJXDataSchema(
        treedef=treedef,
        leaves=specs,
        qpos_shape=data.qpos.shape,
    )


def _finite_dynamic_state(
    data: Any,
    schema: MJXDataSchema,
    case_shape: tuple[int, ...],
    /,
) -> Any:
    finite = jnp.ones(case_shape, dtype=jnp.bool_)
    leaves = jax.tree_util.tree_leaves(data)
    for leaf, spec in zip(leaves, schema.leaves, strict=True):
        if jnp.issubdtype(leaf.dtype, jnp.inexact):
            intrinsic_axes = tuple(
                range(len(case_shape), len(case_shape) + len(spec.shape))
            )
            leaf_finite = jnp.isfinite(leaf)
            if intrinsic_axes:
                leaf_finite = jnp.all(leaf_finite, axis=intrinsic_axes)
            finite = finite & leaf_finite
    return finite


def _select_complete_state(
    finite: Any,
    source: Any,
    candidate: Any,
    case_shape: tuple[int, ...],
    /,
) -> Any:
    def select(source_leaf: Any, candidate_leaf: Any, /) -> Any:
        intrinsic_rank = candidate_leaf.ndim - len(case_shape)
        mask = jnp.reshape(finite, case_shape + (1,) * intrinsic_rank)
        return jnp.where(mask, candidate_leaf, source_leaf)

    return jax.tree_util.tree_map(select, source, candidate)


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
    if (
        integrator == "implicitfast"
        and (
            float(model.opt.density) > 0.0
            or float(model.opt.viscosity) > 0.0
            or np.any(np.asarray(model.opt.wind) != 0.0)
        )
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
            ) or (
                int(model.geom_contype[second]) & int(model.geom_conaffinity[first])
            )
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
            _contact_feature_name(
                geom_names[first], geom_names[second], geom_order
            )
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


def _projection_provenance(
    mujoco: Any,
    model: Any,
    versions: Sequence[tuple[str, str]],
    /,
) -> RoboticsProjectionProvenance:
    model_buffer = np.empty(int(mujoco.mj_sizeModel(model)), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=model_buffer)
    digest = hashlib.sha256(model_buffer.tobytes()).hexdigest()
    version_by_name = dict(versions)
    return RoboticsProjectionProvenance(
        model=f"mujoco-mjb-sha256:{digest}",
        compiler=f"mujoco:{version_by_name['mujoco']}",
        provider=f"mujoco-mjx:{version_by_name['mujoco-mjx']}",
        asset=f"compiled-assets-sha256:{digest}",
        unit_system="MuJoCo SI base units",
        frame_convention="MuJoCo world and body-local frames",
    )


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
    actuator_indices = tuple(
        int(index) for index in np.flatnonzero(muscle_mask).tolist()
    )
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
            int(addresses[joint_index + 1])
            if joint_index + 1 < int(model.njnt)
            else size
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
        sensor_base = (
            qpos_map.size + qvel_map.size + control_map.size
        )
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
    index_map = RoboticsProjectionMap(
        "observation", offset, entries, full_map.provenance
    )
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
) -> MJXAdapter:
    """Validate a host model before one transfer into an owned MJX-JAX state."""
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
    provenance = _projection_provenance(mujoco, model, availability.versions)
    device_model = mjx.put_model(model, device=device, impl="jax")
    canonical = mjx.make_data(device_model, device=device, impl="jax")
    if not isinstance(canonical, mjx.Data) or canonical.impl != mjx.Impl.JAX:
        raise TypeError("make_data(model) must return a complete MJX-JAX Data PyTree.")
    opaque = mjx.forward(device_model, canonical)
    data_schema = _data_schema(opaque)
    data_schema.validate(opaque)
    if muscle_actuator_names:
        _validate_muscle_data_fields(opaque, int(model.nu))
    if not bool(np.asarray(_finite_dynamic_state(opaque, data_schema, ()))):
        raise ValueError("Initial mjx.forward state must be completely finite.")

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
    owner = object()
    return MJXAdapter(
        model=device_model,
        data=opaque,
        qpos_map=qpos_map,
        qvel_map=qvel_map,
        control_map=control_map,
        observation_map=observation_map,
        feature_manifest=feature_manifest,
        data_schema=data_schema,
        provenance=provenance,
        muscle_actuator_names=muscle_actuator_names,
        muscle_actuator_indices=muscle_actuator_indices,
        muscle_activation_indices=muscle_activation_indices,
        device=device_name,
        dtype=opaque.qpos.dtype,
        mjx_module=mjx,
        owner=owner,
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
    "MJXArrayLeafSpec",
    "MJXDataSchema",
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
    "MJXStepResult",
    "MJXStepObservationMode",
    "MJX_WARP_PROFILE",
    "mjx_availability",
    "prepare_mjx_adapter",
    "prepare_mjx_muscle_projection",
]
