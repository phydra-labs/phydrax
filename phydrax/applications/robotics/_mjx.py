#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
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
    ROBOTICS_OPERATIONS,
    RoboticsBackendProfile,
    RoboticsIndexEntry,
    RoboticsOperationCapability,
    RoboticsOperationEvidence,
    RoboticsOperationStatus,
    RoboticsProjection,
    RoboticsProjectionMap,
)


MJXStepObservationRequest: TypeAlias = Literal[
    "none", "pre-step", "post-step-refreshed", "both"
]

_MJX_JAX_DEVICES = ("cpu", "gpu", "tpu")
_MJX_WARP_DEVICES = ("cpu", "gpu")
_MJX_DTYPES = ("float32", "float64")
_MJX_JAX_SOLVER_EXCLUSIONS = ("pgs", "noslip")
_MJX_JAX_CONTACT_EXCLUSIONS = (
    "sdf",
    "sphere-cylinder",
    "box-cylinder",
    "mesh-cylinder",
    "hfield-cylinder",
    "box-ellipsoid",
    "mesh-ellipsoid",
    "hfield-ellipsoid",
    "elliptic-condim-1",
)


def _capability(
    operation: str,
    implementation: str,
    devices: Sequence[str],
    /,
    *,
    differentiability: Literal["none", "conditional", "guaranteed"] = "none",
    solver_exclusions: Sequence[str] = (),
    contact_exclusions: Sequence[str] = (),
) -> RoboticsOperationCapability:
    return RoboticsOperationCapability(
        operation,  # type: ignore[arg-type]
        supported=True,
        implementation=implementation,
        devices=devices,
        dtypes=_MJX_DTYPES,
        differentiability=differentiability,
        solver_exclusions=solver_exclusions,
        contact_exclusions=contact_exclusions,
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


MJX_JAX_PROFILE = RoboticsBackendProfile(
    backend="mjx-jax",
    implementation="mjx-jax",
    operations=(
        _capability(
            "forward-kinematics",
            "mjx.kinematics",
            _MJX_JAX_DEVICES,
            differentiability="conditional",
        ),
        _capability(
            "smooth-dynamics",
            "mjx.forward",
            _MJX_JAX_DEVICES,
            differentiability="conditional",
            solver_exclusions=_MJX_JAX_SOLVER_EXCLUSIONS,
        ),
        _capability(
            "contact",
            "mjx.collision",
            _MJX_JAX_DEVICES,
            contact_exclusions=_MJX_JAX_CONTACT_EXCLUSIONS,
        ),
        _capability(
            "step",
            "mjx.step",
            _MJX_JAX_DEVICES,
            differentiability="conditional",
            solver_exclusions=_MJX_JAX_SOLVER_EXCLUSIONS,
            contact_exclusions=_MJX_JAX_CONTACT_EXCLUSIONS,
        ),
        _capability(
            "sensors",
            "mjx.forward",
            _MJX_JAX_DEVICES,
            differentiability="conditional",
            contact_exclusions=_MJX_JAX_CONTACT_EXCLUSIONS,
        ),
        _capability("model-batching", "jax PyTree batching", _MJX_JAX_DEVICES),
        _capability("jit", "jax.jit", _MJX_JAX_DEVICES),
        _capability("vmap", "jax.vmap", _MJX_JAX_DEVICES),
        _capability(
            "jvp",
            "jax.jvp",
            _MJX_JAX_DEVICES,
            differentiability="conditional",
            solver_exclusions=_MJX_JAX_SOLVER_EXCLUSIONS,
            contact_exclusions=_MJX_JAX_CONTACT_EXCLUSIONS,
        ),
        _capability(
            "vjp",
            "jax.vjp",
            _MJX_JAX_DEVICES,
            differentiability="conditional",
            solver_exclusions=_MJX_JAX_SOLVER_EXCLUSIONS,
            contact_exclusions=_MJX_JAX_CONTACT_EXCLUSIONS,
        ),
    ),
)

MJX_WARP_PROFILE = RoboticsBackendProfile(
    backend="mjx-warp",
    implementation="mjx-warp",
    operations=(
        _capability("forward-kinematics", "mjx-warp", _MJX_WARP_DEVICES),
        _capability(
            "smooth-dynamics",
            "mjx-warp",
            _MJX_WARP_DEVICES,
            solver_exclusions=_MJX_JAX_SOLVER_EXCLUSIONS,
        ),
        _capability("contact", "mjx-warp", _MJX_WARP_DEVICES),
        _capability(
            "step",
            "mjx-warp",
            _MJX_WARP_DEVICES,
            solver_exclusions=_MJX_JAX_SOLVER_EXCLUSIONS,
        ),
        _capability("sensors", "mjx-warp", _MJX_WARP_DEVICES),
        _capability("model-batching", "mjx-warp", _MJX_WARP_DEVICES),
        _capability("jit", "mjx-warp JAX FFI", _MJX_WARP_DEVICES),
        _capability("vmap", "mjx-warp JAX FFI", _MJX_WARP_DEVICES),
        _unsupported_capability(
            "jvp",
            "mjx-warp",
            "MJX-Warp does not support automatic differentiation",
        ),
        _unsupported_capability(
            "vjp",
            "mjx-warp",
            "MJX-Warp does not support automatic differentiation",
        ),
    ),
)

MJX_JAX_BACKEND_CAPABILITIES = BackendCapabilities(
    backend="mjx-jax",
    problem_kinds=tuple(f"robotics.{operation}" for operation in ROBOTICS_OPERATIONS),
    execution="device",
    host_only=False,
    supports_matrix_free=False,
    supports_assembled=False,
    coordinate_dtypes=_MJX_DTYPES,
    supports_plan_prepare_solve_refresh=False,
)


def mjx_availability() -> BackendAvailability:
    """Probe the optional MuJoCo/MJX provider through the shared backend boundary."""
    return probe_backend(
        MJX_JAX_BACKEND_CAPABILITIES,
        module="mujoco.mjx",
        requirement="install the optional mujoco-mjx provider",
        distributions=("mujoco", "mujoco-mjx"),
    )


class MJXState(StrictModule, NonTrainableState):
    """Adapter-owned opaque, complete ``mjx.Data`` PyTree."""

    opaque: Any
    _owner: object = eqx.field(static=True)

    def __init__(self, opaque: Any, owner: object, /):
        self.opaque = opaque
        self._owner = owner


class MJXStepResult(StrictModule, NonTrainableState):
    """Fail-closed state and explicitly timed observations from one MJX step."""

    state: MJXState
    pre_step_observation: RoboticsProjection | None
    post_step_observation: RoboticsProjection | None
    evidence: RoboticsOperationEvidence


class MJXAdapter(StrictModule, NonTrainableState):
    """Prepared MJX-JAX model plus adapter-owned initial foreign state."""

    model: Any
    initial_state: MJXState
    qpos_map: RoboticsProjectionMap
    qvel_map: RoboticsProjectionMap
    control_map: RoboticsProjectionMap
    observation_map: RoboticsProjectionMap
    profile: RoboticsBackendProfile
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
        device: str,
        dtype: str,
        mjx_module: Any,
        owner: object,
    ):
        self.model = model
        self.initial_state = MJXState(data, owner)
        self.qpos_map = qpos_map
        self.qvel_map = qvel_map
        self.control_map = control_map
        self.observation_map = observation_map
        self.profile = MJX_JAX_PROFILE
        self.device = str(device).lower()
        self.dtype = np.dtype(dtype).name
        self._mjx = mjx_module
        self._owner = owner

    def _data(self, state: MJXState | None, /) -> Any:
        resolved = self.initial_state if state is None else state
        if not isinstance(resolved, MJXState):
            raise TypeError("state must be MJXState.")
        if resolved._owner is not self._owner:
            raise ValueError("MJX state belongs to a different prepared adapter.")
        if not isinstance(resolved.opaque, self._mjx.Data):
            raise TypeError("MJX state must retain a complete mjx.Data PyTree.")
        return resolved.opaque

    def qpos(self, state: MJXState | None = None, /) -> RoboticsProjection:
        """Project canonical generalized positions without copying foreign state."""
        data = self._data(state)
        return RoboticsProjection(data.qpos, self.qpos_map)

    def qvel(self, state: MJXState | None = None, /) -> RoboticsProjection:
        """Project canonical generalized velocities without copying foreign state."""
        data = self._data(state)
        return RoboticsProjection(data.qvel, self.qvel_map)

    def control(self, state: MJXState | None = None, /) -> RoboticsProjection:
        """Project canonical controls without copying foreign state."""
        data = self._data(state)
        return RoboticsProjection(data.ctrl, self.control_map)

    def observation(
        self,
        state: MJXState | None = None,
        /,
        *,
        freshness: Literal[
            "state-current", "pre-step", "post-step-refreshed"
        ] = "state-current",
    ) -> RoboticsProjection:
        """Project qpos, qvel, and sensor data with explicit freshness evidence."""
        data = self._data(state)
        values = jnp.concatenate((data.qpos, data.qvel, data.sensordata), axis=-1)
        return RoboticsProjection(values, self.observation_map, freshness=freshness)

    def step(
        self,
        state: MJXState | None = None,
        control: Any | RoboticsProjection | None = None,
        /,
        *,
        observations: MJXStepObservationRequest = "none",
    ) -> MJXStepResult:
        """Advance complete foreign state using only public MJX simulation APIs."""
        if observations not in (
            "none",
            "pre-step",
            "post-step-refreshed",
            "both",
        ):
            raise ValueError(f"Unknown MJX observation request {observations!r}.")
        source = self._data(state)
        stepped_source = source
        if control is not None:
            if isinstance(control, RoboticsProjection):
                if control.index_map != self.control_map:
                    raise ValueError("Control projection map does not match this adapter.")
                control_values = control.values
            else:
                control_values = control
            control_array = jnp.asarray(control_values, dtype=source.ctrl.dtype)
            if control_array.shape != source.ctrl.shape:
                raise ValueError(
                    f"Control must have complete shape {source.ctrl.shape}; "
                    f"got {control_array.shape}."
                )
            stepped_source = source.replace(ctrl=control_array)

        pre_observation = None
        if observations in ("pre-step", "both"):
            pre_observation = self._observation_from_data(
                stepped_source, freshness="pre-step"
            )

        candidate = self._mjx.step(self.model, stepped_source)
        post_requested = observations in ("post-step-refreshed", "both")
        if post_requested:
            candidate = self._mjx.forward(self.model, candidate)

        finite = _finite_dynamic_state(candidate)
        accepted = _select_complete_state(finite, source, candidate)
        accepted_state = MJXState(accepted, self._owner)
        post_observation = None
        if post_requested:
            post_observation = self._observation_from_data(
                accepted, freshness="post-step-refreshed"
            )
        status = jnp.where(
            finite,
            int(RoboticsOperationStatus.SUCCESS),
            int(RoboticsOperationStatus.NONFINITE),
        ).astype(jnp.int32)
        freshness = ()
        if observations == "pre-step":
            freshness = ("pre-step",)
        elif observations == "post-step-refreshed":
            freshness = ("post-step-refreshed",)
        elif observations == "both":
            freshness = ("pre-step", "post-step-refreshed")
        evidence = RoboticsOperationEvidence(
            status=status,
            finite=finite,
            backend="mjx-jax",
            operation="step",
            implementation="mjx.step",
            device=self.device,
            dtype=self.dtype,
            observation_freshness=freshness,
            detail=(
                "the complete candidate mjx.Data is accepted only when every "
                "dynamic floating leaf is finite"
            ),
        )
        return MJXStepResult(
            accepted_state,
            pre_observation,
            post_observation,
            evidence,
        )

    def _observation_from_data(
        self,
        data: Any,
        /,
        *,
        freshness: Literal["pre-step", "post-step-refreshed"],
    ) -> RoboticsProjection:
        values = jnp.concatenate((data.qpos, data.qvel, data.sensordata), axis=-1)
        return RoboticsProjection(values, self.observation_map, freshness=freshness)


def _finite_dynamic_state(data: Any, /) -> Any:
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(data):
        if eqx.is_inexact_array(leaf) and not isinstance(leaf, np.ndarray):
            finite = finite & jnp.all(jnp.isfinite(leaf))
    return finite


def _select_complete_state(finite: Any, source: Any, candidate: Any, /) -> Any:
    def select(source_leaf: Any, candidate_leaf: Any, /) -> Any:
        if eqx.is_array(candidate_leaf) and not isinstance(candidate_leaf, np.ndarray):
            return jnp.where(finite, candidate_leaf, source_leaf)
        return source_leaf

    return jax.tree_util.tree_map(select, source, candidate)


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


def _joint_map(
    mujoco: Any,
    model: Any,
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
    return RoboticsProjectionMap(kind, size, entries)


def _control_map(mujoco: Any, model: Any, /) -> RoboticsProjectionMap:
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
    return RoboticsProjectionMap("control", int(model.nu), entries)


def _observation_map(
    mujoco: Any,
    model: Any,
    qpos_map: RoboticsProjectionMap,
    qvel_map: RoboticsProjectionMap,
    /,
) -> RoboticsProjectionMap:
    entries: list[RoboticsIndexEntry] = []
    for prefix, index_map, offset in (
        ("qpos", qpos_map, 0),
        ("qvel", qvel_map, qpos_map.size),
    ):
        entries.extend(
            RoboticsIndexEntry(
                f"{prefix}/{entry.name}",
                offset + entry.start,
                offset + entry.stop,
            )
            for entry in index_map.entries
        )
    sensor_offset = qpos_map.size + qvel_map.size
    for sensor_index in range(int(model.nsensor)):
        start = sensor_offset + int(model.sensor_adr[sensor_index])
        stop = start + int(model.sensor_dim[sensor_index])
        name = _object_name(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_index,
            f"sensor-{sensor_index}",
        )
        entries.append(RoboticsIndexEntry(f"sensor/{name}", start, stop))
    size = sensor_offset + int(model.nsensordata)
    return RoboticsProjectionMap("observation", size, entries)


def _validate_complete_data(data: Any, mjx: Any, model: Any, /) -> None:
    if not isinstance(data, mjx.Data):
        raise TypeError("data must be a complete mjx.Data instance.")
    if data.impl != mjx.Impl.JAX:
        raise BackendUnavailableError(
            "mjx-jax",
            "robotics.step",
            "an MJX-JAX Data PyTree",
            "MJX-Warp is a distinct non-differentiable backend profile",
        )
    expected = (
        ("qpos", data.qpos.shape[-1], int(model.nq)),
        ("qvel", data.qvel.shape[-1], int(model.nv)),
        ("ctrl", data.ctrl.shape[-1], int(model.nu)),
        ("sensordata", data.sensordata.shape[-1], int(model.nsensordata)),
    )
    for name, observed, required in expected:
        if observed != required:
            raise ValueError(
                f"MJX Data {name} final axis must have size {required}; got {observed}."
            )


def prepare_mjx_adapter(
    model: Any,
    /,
    *,
    data: Any | None = None,
    device: Any | None = None,
) -> MJXAdapter:
    """Host factory for one compiled ``MjModel`` and complete MJX-JAX state."""
    availability = mjx_availability()
    mujoco = import_backend_module(
        availability, "robotics.step", "mujoco"
    )
    mjx = import_backend_module(
        availability, "robotics.step", "mujoco.mjx"
    )
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be an already compiled mujoco.MjModel.")

    device_model = mjx.put_model(model, device=device, impl="jax")
    opaque = (
        mjx.make_data(device_model, device=device, impl="jax")
        if data is None
        else data
    )
    _validate_complete_data(opaque, mjx, model)

    qpos_map = _joint_map(mujoco, model, kind="qpos")
    qvel_map = _joint_map(mujoco, model, kind="qvel")
    control_map = _control_map(mujoco, model)
    observation_map = _observation_map(mujoco, model, qpos_map, qvel_map)
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
        device=device_name,
        dtype=opaque.qpos.dtype,
        mjx_module=mjx,
        owner=owner,
    )


__all__ = [
    "MJXAdapter",
    "MJX_JAX_BACKEND_CAPABILITIES",
    "MJX_JAX_PROFILE",
    "MJX_WARP_PROFILE",
    "MJXState",
    "MJXStepObservationRequest",
    "MJXStepResult",
    "mjx_availability",
    "prepare_mjx_adapter",
]
