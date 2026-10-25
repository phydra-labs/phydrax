#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only adaptation of a compiled MuJoCo model into a native robot model.

`adapt_mujoco_model` accepts an already compiled public ``mujoco.MjModel``,
inventories every compiled feature, snapshots the admitted numerical data and
lowers it through the shared `lower_native_robot_tree` owner. It never parses
XML, never retains the provider model, callbacks or plugins, and the returned
native model executes without MuJoCo.

Closed compatibility profile (everything else is refused, never ignored):

* one physical fixed base: the only child of the world body has no joint and
  positive compiled mass; every other body descends from it, has positive
  mass and at most one hinge or slide joint (zero joints lower to a native
  fixed edge). A massless world anchor is not given an artificial mass.
* compiled COM inertias: body ``i`` becomes the native COM-centered body whose
  axes are the compiled principal inertial frame ``body_iquat``, with the
  principal moments ``body_inertia`` as its diagonal tensor. No parallel-axis
  shift is applied because MuJoCo already compiles inertia about the COM.
* references: native lifted coordinates are ``q = qpos - qpos0``; the native
  reference pose is the compiled ``qpos0`` pose from ``body_pos``/``body_quat``.
* stateless motors: joint transmission, no dynamics, fixed gain, no bias,
  symmetric or absent force range. Native ``gear = gain * gear`` and
  ``effort_limit = |gear| * forcerange`` reproduce MuJoCo's order
  ``clip(ctrl) -> gain -> clip(force) -> gear``.
* executed viscous joint damping ``dof_damping`` unless the damper is disabled.
* body-origin and site frames retained exactly in the COM body frames.
* joint position/velocity, frame pose/velocity and clock sensors as evidence.

Joint limits and contacts are MuJoCo regularized constraint laws. They are
never treated as equivalent native laws: a model that executes them is refused
unless the caller explicitly requests a declared transformation
(`MujocoJointStopTransformation`, `MujocoNativeContactTransformation`) or an
explicit exclusion (`MujocoContactExclusion`), each recorded as an
interpretation-changing loss whose waiver rationale is the caller's.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Literal

import equinox as eqx
import numpy as np
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._physical import DimensionalScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...backends._availability import import_backend_module, probe_backend
from ...backends._types import BackendAvailability, BackendCapabilities
from ...discretization.contact._articulated_geometry import ArticulatedPrimitiveKind
from ...discretization.particle._reduced_articulation import (
    PreparedReducedArticulation,
)
from ...discretization.particle._rigid_joint_laws import (
    ScalarJointDampingPlan,
    ScalarJointMotorPlan,
)
from ...discretization.particle._rigid_unilateral import ReducedJointStopPlan
from ...interchange._report import (
    _AdapterLossCategory,
    AdapterCapability,
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
    AdapterWaiver,
    negotiate_adapter,
)
from ...typing import parse
from ._native_model import (
    AbstractRobotSourceEvidence,
    lower_native_robot_tree,
    NativeTreeJoint,
    RobotAdaptation,
    RobotCollisionGeometryPlan,
    RobotContactMaterialTable,
    RobotFramePlan,
    RobotNameIDMap,
)


type MujocoJointKind = Literal["fixed", "hinge", "prismatic"]
type MujocoSensorStage = Literal["position", "velocity", "clock"]
type MujocoGeomRole = Literal["primitive", "plane", "excluded-contact", "non-colliding"]
type MujocoContactMode = Literal[
    "no-compiled-contact", "excluded", "native-inelastic-coulomb-hard-contact"
]
type MujocoJointLimitMode = Literal["none", "disabled-in-source", "native-hard-stops"]
type _FloatVectorLike = ArrayLike | Sequence[float]

_SOURCE_FORMAT = "MuJoCo compiled mjModel"
_TARGET_FORMAT = "phydrax-native-robot-model"
_IDENTITY_QUATERNION = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

MUJOCO_HOST_CAPABILITIES = BackendCapabilities(
    backend="mujoco-host",
    problem_kinds=("robotics.model-adaptation",),
    execution="host",
    host_only=True,
    supports_matrix_free=False,
    supports_assembled=False,
    coordinate_dtypes=("float64",),
    supports_plan_prepare_solve_refresh=False,
)

_CAPABILITY_SEMANTICS = (
    "robot.connected-oriented-tree",
    "robot.inertial-rigid-links",
    "robot.fixed-hinge-prismatic-kinematics",
    "robot.reference-kinematics",
    "robot.si-dimensional-contract",
    "robot.stable-int64-name-maps",
    "robot.source-index-maps",
    "robot.fixed-world-root",
    "robot.actuation-mapping",
    "robot.executed-joint-damping",
    "robot.unilateral-joint-stops",
    "robot.collision-geometry",
    "robot.sensor-selection",
)
_CAPABILITIES = {
    semantic: AdapterCapability(semantic) for semantic in _CAPABILITY_SEMANTICS
}
# Source laws that the native model never preserves. Losses replacing or
# dropping them affect these capabilities, which the native target does not
# advertise; they are optional requirements, so a declared waiver is honored
# while every executed native capability stays a required semantic.
_LIMIT_LAW_SEMANTIC = "mujoco.regularized-joint-limit-law"
_CONTACT_LAW_SEMANTIC = "mujoco.soft-contact-law"
_LIMIT_LAW_CAPABILITY_ID = AdapterCapability(_LIMIT_LAW_SEMANTIC).capability_id
_CONTACT_LAW_CAPABILITY_ID = AdapterCapability(_CONTACT_LAW_SEMANTIC).capability_id
_NAME_CAPABILITY_ID = _CAPABILITIES["robot.stable-int64-name-maps"].capability_id


def mujoco_host_availability() -> BackendAvailability:
    """Probe the optional host MuJoCo provider used only for adaptation."""

    return probe_backend(
        MUJOCO_HOST_CAPABILITIES,
        module="mujoco",
        requirement="install the optional mujoco provider to adapt compiled models",
        distributions=("mujoco",),
    )


class MujocoModelAdaptationError(AdapterError):
    """Fail-closed compiled-model refusal with optional report and evidence."""

    report: AdapterReport | None
    evidence: MujocoModelEvidence | None

    def __init__(
        self,
        status: AdapterStatus,
        message: str,
        /,
        *,
        report: AdapterReport | None = None,
        evidence: MujocoModelEvidence | None = None,
    ) -> None:
        self.report = report
        self.evidence = evidence
        super().__init__(status, message)


def _refuse(message: str, /) -> MujocoModelAdaptationError:
    return MujocoModelAdaptationError(
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, message
    )


class MujocoJointStopTransformation(StrictModule, NonTrainableState):
    """Declared replacement of MuJoCo regularized limits by native hard stops.

    MuJoCo limits are soft constraints shaped by ``solref``/``solimp``; native
    stops are coupled hard unilateral stops on the lifted coordinates
    ``[range_low - qpos0, range_high - qpos0]``. The two laws are different;
    ``rationale`` is the caller's waiver of that interpretation change.
    """

    rationale: str = eqx.field(static=True)

    def __init__(self, rationale: str, /) -> None:
        rationale_ = str(rationale).strip()
        if not rationale_:
            raise ValueError("A joint-stop transformation requires a rationale.")
        self.rationale = rationale_


class MujocoContactExclusion(StrictModule, NonTrainableState):
    """Declared removal of every MuJoCo contact from the native model."""

    rationale: str = eqx.field(static=True)

    def __init__(self, rationale: str, /) -> None:
        rationale_ = str(rationale).strip()
        if not rationale_:
            raise ValueError("A contact exclusion requires a rationale.")
        self.rationale = rationale_


class MujocoNativeContactTransformation(StrictModule, NonTrainableState):
    """Declared replacement of MuJoCo contact by the native hard-contact law.

    The compiled geometry and MuJoCo's exact pair filtering are retained; the
    regularized soft-contact law, ``condim``, ``solref``/``solimp``, margins,
    gaps and friction mixing are *not* mapped. Every native primitive takes
    its material ID from ``geom_material_ids`` (by compiled geom name) or
    ``default_material_id``; ``materials`` must cover every pair that can meet.
    """

    materials: RobotContactMaterialTable
    geom_material_ids: tuple[tuple[str, int], ...] = eqx.field(static=True)
    default_material_id: int | None = eqx.field(static=True)
    rationale: str = eqx.field(static=True)

    def __init__(
        self,
        materials: RobotContactMaterialTable,
        rationale: str,
        /,
        *,
        geom_material_ids: Mapping[str, int] | None = None,
        default_material_id: int | None = None,
    ) -> None:
        if not isinstance(materials, RobotContactMaterialTable):
            raise TypeError("materials must be a RobotContactMaterialTable.")
        rationale_ = str(rationale).strip()
        if not rationale_:
            raise ValueError("A native contact transformation requires a rationale.")
        entries = tuple(
            sorted(
                (str(name), int(material))
                for name, material in dict(geom_material_ids or {}).items()
            )
        )
        if any(not name for name, _ in entries):
            raise ValueError("Geom material names must be non-empty.")
        self.materials = materials
        self.geom_material_ids = entries
        self.default_material_id = (
            None if default_material_id is None else int(default_material_id)
        )
        self.rationale = rationale_

    def material_for(self, geom_name: str, /) -> int:
        for name, material in self.geom_material_ids:
            if name == geom_name:
                return material
        if self.default_material_id is None:
            raise _refuse(f"Native contact geom {geom_name!r} has no declared material.")
        return self.default_material_id


type MujocoContactPolicy = MujocoContactExclusion | MujocoNativeContactTransformation


class MujocoBodyMap(StrictModule, NonTrainableState):
    """One compiled body and its native body."""

    name: str = eqx.field(static=True)
    mujoco_body_id: int = eqx.field(static=True)
    native_body_id: int = eqx.field(static=True)
    mass: float = eqx.field(static=True)
    principal_inertia: tuple[float, float, float] = eqx.field(static=True)
    com_in_body: tuple[float, float, float] = eqx.field(static=True)
    inertial_quaternion: tuple[float, float, float, float] = eqx.field(static=True)
    reference_com_position: tuple[float, float, float] = eqx.field(static=True)
    reference_com_orientation: tuple[float, float, float, float] = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        mujoco_body_id: int,
        native_body_id: int,
        mass: float,
        principal_inertia: _FloatVectorLike,
        com_in_body: _FloatVectorLike,
        inertial_quaternion: _FloatVectorLike,
        reference_com_position: _FloatVectorLike,
        reference_com_orientation: _FloatVectorLike,
        /,
    ) -> None:
        self.name = str(name)
        self.mujoco_body_id = int(mujoco_body_id)
        self.native_body_id = int(native_body_id)
        self.mass = float(mass)
        self.principal_inertia = _vector3(principal_inertia, "principal_inertia")
        self.com_in_body = _vector3(com_in_body, "com_in_body")
        self.inertial_quaternion = _quaternion4(
            inertial_quaternion, "inertial_quaternion"
        )
        self.reference_com_position = _vector3(
            reference_com_position, "reference_com_position"
        )
        self.reference_com_orientation = _quaternion4(
            reference_com_orientation, "reference_com_orientation"
        )


class MujocoJointMap(StrictModule, NonTrainableState):
    """One native tree edge and its compiled joint, if any.

    Fixed edges of jointless bodies have ``mujoco_joint_id == -1`` and no
    coordinate addresses. ``qpos0`` is the compiled reference coordinate
    subtracted to obtain the native lifted coordinate.
    """

    name: str = eqx.field(static=True)
    mujoco_joint_id: int = eqx.field(static=True)
    native_joint_id: int = eqx.field(static=True)
    kind: MujocoJointKind = eqx.field(static=True)
    parent_body: str = eqx.field(static=True)
    child_body: str = eqx.field(static=True)
    qpos_address: int = eqx.field(static=True)
    dof_address: int = eqx.field(static=True)
    qpos0: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    limited: bool = eqx.field(static=True)
    range: tuple[float, float] = eqx.field(static=True)
    anchor: tuple[float, float, float] = eqx.field(static=True)
    axis: tuple[float, float, float] = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        mujoco_joint_id: int,
        native_joint_id: int,
        kind: MujocoJointKind,
        parent_body: str,
        child_body: str,
        qpos_address: int,
        dof_address: int,
        qpos0: float,
        damping: float,
        limited: bool,
        range_: _FloatVectorLike,
        anchor: _FloatVectorLike,
        axis: _FloatVectorLike,
        /,
    ) -> None:
        self.name = str(name)
        self.mujoco_joint_id = int(mujoco_joint_id)
        self.native_joint_id = int(native_joint_id)
        self.kind = parse(kind, MujocoJointKind, "kind")
        self.parent_body = str(parent_body)
        self.child_body = str(child_body)
        self.qpos_address = int(qpos_address)
        self.dof_address = int(dof_address)
        self.qpos0 = float(qpos0)
        self.damping = float(damping)
        self.limited = bool(limited)
        self.range = _pair(range_, "range")
        self.anchor = _vector3(anchor, "anchor")
        self.axis = _vector3(axis, "axis")


class MujocoActuatorMap(StrictModule, NonTrainableState):
    """One compiled actuator and the native motor row it became."""

    name: str = eqx.field(static=True)
    mujoco_actuator_id: int = eqx.field(static=True)
    native_actuator_id: int = eqx.field(static=True)
    joint: str = eqx.field(static=True)
    gain: float = eqx.field(static=True)
    source_gear: float = eqx.field(static=True)
    native_gear: float = eqx.field(static=True)
    command_range: tuple[float, float] = eqx.field(static=True)
    force_range: tuple[float, float] = eqx.field(static=True)
    native_effort_limit: float = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        mujoco_actuator_id: int,
        native_actuator_id: int,
        joint: str,
        gain: float,
        source_gear: float,
        native_gear: float,
        command_range: _FloatVectorLike,
        force_range: _FloatVectorLike,
        native_effort_limit: float,
        /,
    ) -> None:
        self.name = str(name)
        self.mujoco_actuator_id = int(mujoco_actuator_id)
        self.native_actuator_id = int(native_actuator_id)
        self.joint = str(joint)
        self.gain = float(gain)
        self.source_gear = float(source_gear)
        self.native_gear = float(native_gear)
        self.command_range = _pair(command_range, "command_range")
        self.force_range = _pair(force_range, "force_range")
        self.native_effort_limit = float(native_effort_limit)


class MujocoGeomMap(StrictModule, NonTrainableState):
    """One compiled geom and its native role.

    ``role`` is ``"primitive"`` or ``"plane"`` for lowered native contact
    geometry, ``"excluded-contact"`` for collidable geometry removed by an
    explicit exclusion, and ``"non-colliding"`` for geometry that takes part in
    no compiled collision pair (its mass is already in the compiled inertia).
    """

    name: str = eqx.field(static=True)
    mujoco_geom_id: int = eqx.field(static=True)
    mujoco_type: str = eqx.field(static=True)
    body: str = eqx.field(static=True)
    role: MujocoGeomRole = eqx.field(static=True)
    native_name: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        mujoco_geom_id: int,
        mujoco_type: str,
        body: str,
        role: MujocoGeomRole,
        native_name: str,
        /,
    ) -> None:
        self.name = str(name)
        self.mujoco_geom_id = int(mujoco_geom_id)
        self.mujoco_type = str(mujoco_type)
        self.body = str(body)
        self.role = parse(role, MujocoGeomRole, "role")
        self.native_name = str(native_name)


class MujocoSensorMap(StrictModule, NonTrainableState):
    """One admitted compiled sensor: quantity, native target, units and stage."""

    name: str = eqx.field(static=True)
    mujoco_sensor_id: int = eqx.field(static=True)
    quantity: str = eqx.field(static=True)
    target: str = eqx.field(static=True)
    units: str = eqx.field(static=True)
    stage: MujocoSensorStage = eqx.field(static=True)
    address: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        mujoco_sensor_id: int,
        quantity: str,
        target: str,
        units: str,
        stage: MujocoSensorStage,
        address: int,
        dimension: int,
        /,
    ) -> None:
        self.name = str(name)
        self.mujoco_sensor_id = int(mujoco_sensor_id)
        self.quantity = str(quantity)
        self.target = str(target)
        self.units = str(units)
        self.stage = parse(stage, MujocoSensorStage, "stage")
        self.address = int(address)
        self.dimension = int(dimension)


class MujocoModelEvidence(AbstractRobotSourceEvidence):
    """Exact compiled-source identities, maps and executed source options.

    Options that the native model does not own (timestep, integrator, gravity,
    enabled/disabled flags) are retained here so a comparison can decide
    whether its tier is valid; they are not silently executed by the model.
    """

    mujoco_version: str = eqx.field(static=True)
    model_sha256: str = eqx.field(static=True)
    root_body: str = eqx.field(static=True)
    timestep: float = eqx.field(static=True)
    integrator: str = eqx.field(static=True)
    gravity: tuple[float, float, float] = eqx.field(static=True)
    disabled_flags: tuple[str, ...] = eqx.field(static=True)
    enabled_flags: tuple[str, ...] = eqx.field(static=True)
    contact_mode: MujocoContactMode = eqx.field(static=True)
    joint_limit_mode: MujocoJointLimitMode = eqx.field(static=True)
    damping_executed: bool = eqx.field(static=True)
    control_clamped: bool = eqx.field(static=True)
    bodies: tuple[MujocoBodyMap, ...] = eqx.field(static=True)
    joints: tuple[MujocoJointMap, ...] = eqx.field(static=True)
    actuators: tuple[MujocoActuatorMap, ...] = eqx.field(static=True)
    geoms: tuple[MujocoGeomMap, ...] = eqx.field(static=True)
    sensors: tuple[MujocoSensorMap, ...] = eqx.field(static=True)
    frames: tuple[tuple[str, str, int], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    loss_paths: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        mujoco_version: str,
        model_sha256: str,
        root_body: str,
        timestep: float,
        integrator: str,
        gravity: _FloatVectorLike,
        disabled_flags: Sequence[str],
        enabled_flags: Sequence[str],
        contact_mode: MujocoContactMode,
        joint_limit_mode: MujocoJointLimitMode,
        damping_executed: bool,
        control_clamped: bool,
        bodies: Sequence[MujocoBodyMap],
        joints: Sequence[MujocoJointMap],
        actuators: Sequence[MujocoActuatorMap],
        geoms: Sequence[MujocoGeomMap],
        sensors: Sequence[MujocoSensorMap],
        frames: Sequence[tuple[str, str, int]],
        loss_paths: Sequence[str],
        source_id: str,
        target_id: str,
        /,
    ) -> None:
        self.mujoco_version = str(mujoco_version)
        self.model_sha256 = str(model_sha256)
        self.root_body = str(root_body)
        self.timestep = float(timestep)
        self.integrator = str(integrator)
        self.gravity = _vector3(gravity, "gravity")
        self.disabled_flags = tuple(sorted(str(item) for item in disabled_flags))
        self.enabled_flags = tuple(sorted(str(item) for item in enabled_flags))
        self.contact_mode = parse(contact_mode, MujocoContactMode, "contact_mode")
        self.joint_limit_mode = parse(
            joint_limit_mode, MujocoJointLimitMode, "joint_limit_mode"
        )
        self.damping_executed = bool(damping_executed)
        self.control_clamped = bool(control_clamped)
        self.bodies = tuple(bodies)
        self.joints = tuple(joints)
        self.actuators = tuple(actuators)
        self.geoms = tuple(geoms)
        self.sensors = tuple(sensors)
        self.frames = tuple((str(a), str(b), int(c)) for a, b, c in frames)
        self.loss_paths = tuple(str(item) for item in loss_paths)
        self.source_id = str(source_id)
        self.target_id = str(target_id)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "mujoco-compiled-model-evidence",
                "mujoco_version": self.mujoco_version,
                "model_sha256": self.model_sha256,
                "root_body": self.root_body,
                "timestep": np.asarray(self.timestep, dtype=np.float64),
                "integrator": self.integrator,
                "gravity": np.asarray(self.gravity, dtype=np.float64),
                "disabled_flags": list(self.disabled_flags),
                "enabled_flags": list(self.enabled_flags),
                "contact_mode": self.contact_mode,
                "joint_limit_mode": self.joint_limit_mode,
                "damping_executed": self.damping_executed,
                "control_clamped": self.control_clamped,
                "bodies": [_record(item) for item in self.bodies],
                "joints": [_record(item) for item in self.joints],
                "actuators": [_record(item) for item in self.actuators],
                "geoms": [_record(item) for item in self.geoms],
                "sensors": [_record(item) for item in self.sensors],
                "frames": [list(item) for item in self.frames],
                "loss_paths": list(self.loss_paths),
                "source_id": self.source_id,
                "target_id": self.target_id,
            }
        )

    def coordinate_map(
        self, articulation: PreparedReducedArticulation, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return MuJoCo ``qpos``/``qvel`` addresses and ``qpos0`` per native DOF.

        Native configuration ``q[k] = qpos[qpos_address[k]] - qpos0[k]`` and
        native velocity ``v[k] = qvel[dof_address[k]]``.
        """

        if not isinstance(articulation, PreparedReducedArticulation):
            raise TypeError("articulation must be a PreparedReducedArticulation.")
        by_id = {joint.native_joint_id: joint for joint in self.joints}
        ordered_ids = np.asarray(articulation.joint_ids)[
            np.asarray(articulation.dof_joint_indices)
        ]
        selected = [by_id[int(identifier)] for identifier in ordered_ids]
        if any(joint.kind == "fixed" for joint in selected):
            raise ValueError("The articulation does not match this source evidence.")
        return (
            np.asarray([joint.qpos_address for joint in selected], dtype=np.int64),
            np.asarray([joint.dof_address for joint in selected], dtype=np.int64),
            np.asarray([joint.qpos0 for joint in selected], dtype=np.float64),
        )

    def frame_name(self, kind: Literal["body", "site"], source_name: str, /) -> str:
        """Return the native frame name of a compiled body origin or site."""

        for frame, kind_, _ in self.frames:
            if kind_ == kind and frame == _frame_name(kind, source_name):
                return frame
        raise KeyError((kind, source_name))


def _host_floats(value: _FloatVectorLike, size: int, name: str, /) -> np.ndarray:
    """Convert one rank-1 host vector of exactly ``size`` floats (no reshaping)."""

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or np.any(np.isnan(array)):
        raise ValueError(f"{name} must be {size} non-NaN values of shape ({size},).")
    return array


def _pair(value: _FloatVectorLike, name: str, /) -> tuple[float, float]:
    array = _host_floats(value, 2, name)
    return float(array[0]), float(array[1])


def _vector3(value: _FloatVectorLike, name: str, /) -> tuple[float, float, float]:
    array = _host_floats(value, 3, name)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return float(array[0]), float(array[1]), float(array[2])


def _quaternion4(
    value: _FloatVectorLike, name: str, /
) -> tuple[float, float, float, float]:
    array = _host_floats(value, 4, name)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return float(array[0]), float(array[1]), float(array[2]), float(array[3])


def _record(item: StrictModule, /) -> dict[str, object]:
    """Canonical evidence payload; floats are content-addressed float64 arrays.

    Unbounded ranges and limits (``±inf``) therefore keep their exact IEEE
    identity, as in the native plans, instead of non-JSON float literals.
    """

    payload: dict[str, object] = {}
    for name in type(item).__dataclass_fields__:
        if name.startswith("_"):
            continue
        value = getattr(item, name)
        if isinstance(value, float) or (
            isinstance(value, tuple)
            and value
            and all(isinstance(entry, float) for entry in value)
        ):
            payload[name] = np.asarray(value, dtype=np.float64)
        elif isinstance(value, tuple):
            payload[name] = list(value)
        else:
            payload[name] = value
    return payload


def _frame_name(kind: str, source_name: str, /) -> str:
    return f"{kind}:{source_name}"


def _quaternion_multiply(left: np.ndarray, right: np.ndarray, /) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _quaternion_conjugate(value: np.ndarray, /) -> np.ndarray:
    return np.asarray([value[0], -value[1], -value[2], -value[3]], dtype=np.float64)


def _unit_quaternion(value: np.ndarray, /) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not norm > 0.0 or not np.isfinite(norm):
        raise _refuse("The compiled model contains a degenerate quaternion.")
    return quaternion / norm


def _rotation(value: np.ndarray, /) -> np.ndarray:
    w, x, y, z = _unit_quaternion(value)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _enum_name(enum: type, value: int, prefix: str, /) -> str:
    for name in dir(enum):
        if name.startswith(prefix) and int(getattr(enum, name)) == int(value):
            return name[len(prefix) :].lower()
    return f"unknown-{int(value)}"


def _flag_names(enum: type, mask: int, prefix: str, /) -> tuple[str, ...]:
    names: list[str] = []
    for name in dir(enum):
        if not name.startswith(prefix):
            continue
        bit = int(getattr(enum, name))
        if bit and mask & bit:
            names.append(name[len(prefix) :].lower())
    return tuple(sorted(names))


def _object_names(
    mujoco: ModuleType, model: Any, object_type: object, count: int, prefix: str, /
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    names: list[str] = []
    synthesized: list[int] = []
    for index in range(count):
        name = mujoco.mj_id2name(model, object_type, index)
        if name is None or not str(name):
            names.append(f"{prefix}/{index}")
            synthesized.append(index)
        else:
            names.append(str(name))
    if len(set(names)) != len(names):
        raise _refuse(f"Compiled {prefix} names are not unique after naming.")
    return tuple(names), tuple(synthesized)


def _loss(
    path: str,
    category: _AdapterLossCategory,
    rationale: str,
    /,
    *,
    changes_interpretation: bool,
    affected: Sequence[str],
) -> AdapterLoss:
    return AdapterLoss(
        path,
        "import",
        category,
        rationale,
        changes_interpretation=changes_interpretation,
        affected_capability_ids=affected,
    )


def _nonzero(value: ArrayLike, /) -> bool:
    array = np.asarray(value, dtype=np.float64)
    return bool(array.size) and bool(np.any(array != 0.0))


def _refuse_unsupported_inventory(mujoco: ModuleType, model: Any, /) -> None:
    counts = (
        ("nflex", "flexible bodies"),
        ("ntendon", "tendons"),
        ("neq", "equality constraints"),
        ("nmocap", "mocap bodies"),
        ("nplugin", "plugins"),
        ("npluginstate", "plugin state"),
        ("na", "actuator activation state"),
        ("nhistory", "actuator/sensor history buffers"),
        ("ngravcomp", "gravity compensation"),
    )
    for field, label in counts:
        if int(getattr(model, field, 0)):
            raise _refuse(f"Compiled {label} are outside the native profile.")
    option = model.opt
    if (
        float(option.density) != 0.0
        or float(option.viscosity) != 0.0
        or _nonzero(option.wind)
    ):
        raise _refuse("Fluid forces (density, viscosity, wind) are not native.")
    if int(getattr(option, "disableactuator", 0)):
        raise _refuse("Actuator-group disabling is outside the native profile.")
    sleep = getattr(mujoco.mjtEnableBit, "mjENBL_SLEEP", None)
    if sleep is not None and int(option.enableflags) & int(sleep):
        raise _refuse("Body sleeping changes dynamics and is not native.")
    for field, label in (
        ("body_gravcomp", "body gravity compensation"),
        ("jnt_stiffness", "passive joint springs"),
        ("jnt_stiffnesspoly", "polynomial passive joint springs"),
        ("jnt_actgravcomp", "actuator gravity compensation"),
        ("jnt_actfrclimited", "joint-level actuator force limits"),
        ("dof_armature", "joint armature"),
        ("dof_dampingpoly", "polynomial joint damping"),
        ("actuator_armature", "actuator armature"),
        ("actuator_damping", "actuator damping"),
        ("actuator_dampingpoly", "polynomial actuator damping"),
        ("actuator_delay", "actuator delay"),
        ("actuator_history", "actuator history"),
        ("actuator_plugin", "actuator plugins"),
        ("body_plugin", "body plugins"),
        ("geom_plugin", "geom plugins"),
    ):
        if not hasattr(model, field):
            continue
        if field.endswith("plugin"):
            if np.any(np.asarray(getattr(model, field), dtype=np.int64) >= 0):
                raise _refuse(f"Compiled {label} are outside the native profile.")
        elif _nonzero(getattr(model, field)):
            raise _refuse(f"Compiled {label} are outside the native profile.")


@dataclass(frozen=True, slots=True)
class _GeomPairAdmission:
    """Host-only predicate reproducing MuJoCo's static geom-pair admission.

    Storage is per-geom (``O(G)``) plus the sparse compiled ``<exclude>`` and
    ``<pair>`` overrides; no pair table is materialized. The predicate is used
    only during adaptation and is not retained by the native model, which
    receives exclusively the resulting sparse excluded body pairs.
    """

    enabled: bool
    filterparent: bool
    geom_body: np.ndarray
    geom_weld: np.ndarray
    geom_parent_weld: np.ndarray
    geom_plane: np.ndarray
    contype: np.ndarray
    conaffinity: np.ndarray
    excluded_signatures: frozenset[int]
    explicit_pairs: frozenset[tuple[int, int]]

    @classmethod
    def from_model(
        cls, mujoco: ModuleType, model: Any, filterparent: bool, enabled: bool, /
    ) -> _GeomPairAdmission:
        weld = np.asarray(model.body_weldid, dtype=np.int64)
        parent = np.asarray(model.body_parentid, dtype=np.int64)
        geom_body = np.asarray(model.geom_bodyid, dtype=np.int64)
        geom_weld = weld[geom_body]
        return cls(
            enabled,
            filterparent,
            geom_body,
            geom_weld,
            weld[parent[geom_weld]],
            np.asarray(model.geom_type, dtype=np.int64)
            == int(mujoco.mjtGeom.mjGEOM_PLANE),
            np.asarray(model.geom_contype, dtype=np.int64),
            np.asarray(model.geom_conaffinity, dtype=np.int64),
            frozenset(
                int(item) for item in np.asarray(model.exclude_signature, dtype=np.int64)
            ),
            frozenset(
                (
                    min(int(first), int(second)),
                    max(int(first), int(second)),
                )
                for first, second in zip(
                    np.asarray(model.pair_geom1, dtype=np.int64),
                    np.asarray(model.pair_geom2, dtype=np.int64),
                    strict=True,
                )
            ),
        )

    def admitted(self, first: int, second: int, /) -> bool:
        if not self.enabled:
            return False
        low, high = (first, second) if first < second else (second, first)
        if (low, high) in self.explicit_pairs:
            return True
        weld_low = int(self.geom_weld[low])
        weld_high = int(self.geom_weld[high])
        if weld_low == weld_high or (self.geom_plane[low] and self.geom_plane[high]):
            return False
        if (
            self.filterparent
            and weld_low != 0
            and weld_high != 0
            and (
                weld_low == int(self.geom_parent_weld[high])
                or weld_high == int(self.geom_parent_weld[low])
            )
        ):
            return False
        body_low, body_high = sorted(
            (int(self.geom_body[low]), int(self.geom_body[high]))
        )
        if (body_low << 16) + body_high in self.excluded_signatures:
            return False
        return bool(
            (self.contype[low] & self.conaffinity[high])
            or (self.contype[high] & self.conaffinity[low])
        )

    def colliding_geoms(self, maximum_pair_checks: int, /) -> tuple[int, ...]:
        """Geoms in at least one admitted pair, within a declared check budget.

        The scan costs ``G(G-1)/2`` predicate evaluations and ``O(G)`` memory;
        it is refused before it starts when that exceeds the budget.
        """

        if not self.enabled:
            return ()
        count = int(self.geom_body.shape[0])
        checks = count * (count - 1) // 2
        if checks > maximum_pair_checks:
            raise _refuse(
                f"Collision-pair inventory needs {checks} predicate checks, above "
                f"maximum_geom_pair_checks={maximum_pair_checks}."
            )
        colliding = np.zeros((count,), dtype=np.bool_)
        for first in range(count):
            for second in range(first + 1, count):
                if (not colliding[first] or not colliding[second]) and self.admitted(
                    first, second
                ):
                    colliding[first] = colliding[second] = True
        return tuple(int(item) for item in np.flatnonzero(colliding))


def _model_digest(mujoco: ModuleType, model: Any, /) -> str:
    buffer = np.empty(int(mujoco.mj_sizeModel(model)), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


@dataclass(frozen=True, slots=True)
class _SourceOptions:
    """Executed compiled option flags that change the admitted mechanics."""

    disabled: tuple[str, ...]
    enabled: tuple[str, ...]
    limits_executed: bool
    contact_enabled: bool
    damping_executed: bool
    clamp_control: bool
    filterparent: bool
    gravity: np.ndarray
    timestep: float
    integrator: str


def _source_options(mujoco: ModuleType, model: Any, /) -> _SourceOptions:
    option = model.opt
    disable = int(option.disableflags)

    def is_disabled(name: str) -> bool:
        return bool(disable & int(getattr(mujoco.mjtDisableBit, f"mjDSBL_{name}")))

    constraints = not is_disabled("CONSTRAINT")
    gravity = (
        np.zeros((3,), dtype=np.float64)
        if is_disabled("GRAVITY")
        else np.asarray(option.gravity, dtype=np.float64)
    )
    if (
        constraints
        and not is_disabled("FRICTIONLOSS")
        and _nonzero(model.dof_frictionloss)
    ):
        raise _refuse("Dry joint friction loss is outside the native profile.")
    if int(model.nu) and is_disabled("ACTUATION"):
        raise _refuse("Disabled actuation with declared actuators is not mapped.")
    return _SourceOptions(
        _flag_names(mujoco.mjtDisableBit, disable, "mjDSBL_"),
        _flag_names(mujoco.mjtEnableBit, int(option.enableflags), "mjENBL_"),
        constraints and not is_disabled("LIMIT"),
        constraints and not is_disabled("CONTACT"),
        not is_disabled("DAMPER"),
        not is_disabled("CLAMPCTRL"),
        not is_disabled("FILTERPARENT"),
        gravity,
        float(option.timestep),
        _enum_name(mujoco.mjtIntegrator, int(option.integrator), "mjINT_"),
    )


@dataclass(frozen=True, slots=True)
class _SourceNames:
    """Compiled object names by source index, with synthesized-name losses."""

    bodies: tuple[str, ...]
    joints: tuple[str, ...]
    actuators: tuple[str, ...]
    geoms: tuple[str, ...]
    sites: tuple[str, ...]
    sensors: tuple[str, ...]
    losses: tuple[AdapterLoss, ...]


def _source_names(mujoco: ModuleType, model: Any, /) -> _SourceNames:
    objects = mujoco.mjtObj
    inventory = (
        ("bodies", objects.mjOBJ_BODY, int(model.nbody), "body"),
        ("joints", objects.mjOBJ_JOINT, int(model.njnt), "joint"),
        ("actuators", objects.mjOBJ_ACTUATOR, int(model.nu), "actuator"),
        ("geoms", objects.mjOBJ_GEOM, int(model.ngeom), "geom"),
        ("sites", objects.mjOBJ_SITE, int(model.nsite), "site"),
        ("sensors", objects.mjOBJ_SENSOR, int(model.nsensor), "sensor"),
    )
    resolved = [
        _object_names(mujoco, model, object_type, count, prefix)
        for _, object_type, count, prefix in inventory
    ]
    losses = tuple(
        _loss(
            f"/{plural}/{index}/name",
            "synthesized",
            f"unnamed compiled {plural[:-1]} {index} received a positional native name",
            changes_interpretation=False,
            affected=(_NAME_CAPABILITY_ID,),
        )
        for (plural, _, _, _), (_, synthesized) in zip(inventory, resolved, strict=True)
        for index in synthesized
        if not (plural == "bodies" and index == 0)
    )
    return _SourceNames(
        resolved[0][0],
        resolved[1][0],
        resolved[2][0],
        resolved[3][0],
        resolved[4][0],
        resolved[5][0],
        losses,
    )


@dataclass(frozen=True, slots=True)
class _BodyFrames:
    """Physical fixed base, qpos0 body frames and COM-centered native bodies.

    Per-compiled-body arrays are indexed by MuJoCo body ID (row 0 is the
    world); native arrays are in ``names`` ID order.
    """

    root: int
    parent: np.ndarray
    joint_count: np.ndarray
    position: np.ndarray
    orientation: np.ndarray
    inertial_quaternion: np.ndarray
    com_in_body: np.ndarray
    names: RobotNameIDMap
    native_id: dict[int, int]
    masses: np.ndarray
    inertia: np.ndarray
    reference_position: np.ndarray
    reference_orientation: np.ndarray
    maps: tuple[MujocoBodyMap, ...]

    def to_com_point(self, body: int, point: ArrayLike, /) -> np.ndarray:
        """A point of compiled body frame ``body`` in its native COM frame."""

        return _rotation(self.inertial_quaternion[body]).T @ (
            np.asarray(point, dtype=np.float64) - self.com_in_body[body]
        )

    def to_com_orientation(self, body: int, quaternion: ArrayLike, /) -> np.ndarray:
        """An orientation in compiled body frame ``body`` relative to COM axes."""

        return _unit_quaternion(
            _quaternion_multiply(
                _quaternion_conjugate(self.inertial_quaternion[body]),
                np.asarray(quaternion, dtype=np.float64),
            )
        )


def _fixed_base(
    mujoco: ModuleType, model: Any, names: _SourceNames, /
) -> tuple[int, np.ndarray, np.ndarray]:
    """Refuse any topology outside one physical fixed base of scalar joints."""

    parent = np.asarray(model.body_parentid, dtype=np.int64)
    world_children = [body for body in range(1, int(model.nbody)) if parent[body] == 0]
    if len(world_children) != 1:
        raise _refuse(
            "Exactly one compiled body must attach to the world as the fixed base."
        )
    root = world_children[0]
    joint_count = np.asarray(model.body_jntnum, dtype=np.int64)
    if int(joint_count[root]) != 0:
        raise _refuse(
            "The world child must be welded (no joint); floating bases are refused."
        )
    masses = np.asarray(model.body_mass, dtype=np.float64)
    for body in range(1, int(model.nbody)):
        if not masses[body] > 0.0:
            raise _refuse(
                f"Body {names.bodies[body]!r} has no positive compiled mass; "
                "no artificial mass is invented."
            )
        if int(joint_count[body]) > 1:
            raise _refuse(
                f"Body {names.bodies[body]!r} has multiple joints; only one "
                "scalar joint per body is admitted."
            )
    scalar = (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
    joint_type = np.asarray(model.jnt_type, dtype=np.int64)
    for joint in range(int(model.njnt)):
        if int(joint_type[joint]) not in scalar:
            raise _refuse(f"Joint {names.joints[joint]!r} is not a hinge or slide joint.")
    return root, parent, joint_count


def _body_frames(
    model: Any,
    names: _SourceNames,
    root: int,
    parent: np.ndarray,
    joint_count: np.ndarray,
    /,
) -> _BodyFrames:
    """Compose qpos0 frames and lower compiled inertials without shifting them."""

    count = int(model.nbody)
    position = np.zeros((count, 3), dtype=np.float64)
    orientation = np.tile(_IDENTITY_QUATERNION, (count, 1))
    inertial_quaternion = np.tile(_IDENTITY_QUATERNION, (count, 1))
    com_in_body = np.zeros((count, 3), dtype=np.float64)
    for body in range(1, count):
        rotation = _rotation(orientation[parent[body]])
        position[body] = position[parent[body]] + rotation @ np.asarray(
            model.body_pos[body], dtype=np.float64
        )
        orientation[body] = _unit_quaternion(
            _quaternion_multiply(
                orientation[parent[body]],
                np.asarray(model.body_quat[body], dtype=np.float64),
            )
        )
        inertial_quaternion[body] = _unit_quaternion(
            np.asarray(model.body_iquat[body], dtype=np.float64)
        )
        com_in_body[body] = np.asarray(model.body_ipos[body], dtype=np.float64)
    body_names = RobotNameIDMap(tuple(sorted(names.bodies[1:])))
    native_id = {
        body: int(body_names.id_for_name(names.bodies[body])) for body in range(1, count)
    }
    native_count = body_names.count
    masses = np.empty((native_count,), dtype=np.float64)
    inertia = np.zeros((native_count, 3, 3), dtype=np.float64)
    reference_position = np.empty((native_count, 3), dtype=np.float64)
    reference_orientation = np.empty((native_count, 4), dtype=np.float64)
    compiled_mass = np.asarray(model.body_mass, dtype=np.float64)
    maps: list[MujocoBodyMap] = []
    for body in range(1, count):
        index = native_id[body]
        com = position[body] + _rotation(orientation[body]) @ com_in_body[body]
        axes = _unit_quaternion(
            _quaternion_multiply(orientation[body], inertial_quaternion[body])
        )
        principal = np.asarray(model.body_inertia[body], dtype=np.float64)
        masses[index] = compiled_mass[body]
        inertia[index] = np.diag(principal)
        reference_position[index] = com
        reference_orientation[index] = axes
        maps.append(
            MujocoBodyMap(
                names.bodies[body],
                body,
                index,
                float(compiled_mass[body]),
                principal,
                com_in_body[body],
                inertial_quaternion[body],
                com,
                axes,
            )
        )
    return _BodyFrames(
        root,
        parent,
        joint_count,
        position,
        orientation,
        inertial_quaternion,
        com_in_body,
        body_names,
        native_id,
        masses,
        inertia,
        reference_position,
        reference_orientation,
        tuple(sorted(maps, key=lambda item: item.native_body_id)),
    )


@dataclass(frozen=True, slots=True)
class _TreeJoints:
    """Native tree edges in compiled traversal order and their source maps."""

    edges: tuple[NativeTreeJoint, ...]
    names: RobotNameIDMap
    maps: tuple[MujocoJointMap, ...]
    by_source: dict[int, MujocoJointMap]


def _tree_edge(
    mujoco: ModuleType,
    model: Any,
    names: _SourceNames,
    frames: _BodyFrames,
    body: int,
    /,
) -> tuple[NativeTreeJoint, int]:
    """One parent-to-``body`` edge in world zero-configuration coordinates."""

    parent = names.bodies[int(frames.parent[body])]
    child = names.bodies[body]
    if int(frames.joint_count[body]) == 0:
        return NativeTreeJoint(f"weld/{child}", "fixed", parent, child), -1
    source = int(model.body_jntadr[body])
    hinge = int(model.jnt_type[source]) == int(mujoco.mjtJoint.mjJNT_HINGE)
    rotation = _rotation(frames.orientation[body])
    anchor = frames.position[body] + rotation @ np.asarray(
        model.jnt_pos[source], dtype=np.float64
    )
    axis = rotation @ np.asarray(model.jnt_axis[source], dtype=np.float64)
    edge = NativeTreeJoint(
        names.joints[source],
        "hinge" if hinge else "prismatic",
        parent,
        child,
        anchor=anchor,
        axis=axis / np.linalg.norm(axis),
    )
    return edge, source


def _joint_map(
    model: Any, edge: NativeTreeJoint, source: int, joint_names: RobotNameIDMap, /
) -> MujocoJointMap:
    native = int(joint_names.id_for_name(edge.name))
    if source < 0:
        return MujocoJointMap(
            edge.name,
            -1,
            native,
            "fixed",
            edge.parent,
            edge.child,
            -1,
            -1,
            0.0,
            0.0,
            False,
            (0.0, 0.0),
            edge.anchor,
            edge.axis,
        )
    qpos_address = int(model.jnt_qposadr[source])
    dof_address = int(model.jnt_dofadr[source])
    return MujocoJointMap(
        edge.name,
        source,
        native,
        edge.kind,
        edge.parent,
        edge.child,
        qpos_address,
        dof_address,
        float(model.qpos0[qpos_address]),
        float(model.dof_damping[dof_address]),
        bool(model.jnt_limited[source]),
        np.asarray(model.jnt_range[source], dtype=np.float64),
        edge.anchor,
        edge.axis,
    )


def _tree_joints(
    mujoco: ModuleType, model: Any, names: _SourceNames, frames: _BodyFrames, /
) -> _TreeJoints:
    edges: list[NativeTreeJoint] = []
    sources: list[int] = []
    for body in range(1, int(model.nbody)):
        if body == frames.root:
            continue
        edge, source = _tree_edge(mujoco, model, names, frames, body)
        edges.append(edge)
        sources.append(source)
    edge_names = [edge.name for edge in edges]
    if len(set(edge_names)) != len(edge_names):
        raise _refuse("Synthesized fixed-edge names collide with compiled joint names.")
    joint_names = RobotNameIDMap(tuple(sorted(edge_names)))
    maps = tuple(
        _joint_map(model, edge, source, joint_names)
        for edge, source in zip(edges, sources, strict=True)
    )
    return _TreeJoints(
        tuple(edges),
        joint_names,
        maps,
        {joint.mujoco_joint_id: joint for joint in maps if joint.kind != "fixed"},
    )


@dataclass(frozen=True, slots=True)
class _JointLaws:
    """Executed damping and the declared treatment of compiled limits."""

    damping: ScalarJointDampingPlan | None
    stops: ReducedJointStopPlan | None
    limit_mode: MujocoJointLimitMode
    losses: tuple[AdapterLoss, ...]
    waivers: tuple[AdapterWaiver, ...]


def _joint_laws(
    model: Any,
    tree: _TreeJoints,
    options: _SourceOptions,
    joint_stops: MujocoJointStopTransformation | None,
    /,
) -> _JointLaws:
    scalar = [joint for joint in tree.maps if joint.kind != "fixed"]
    damped = [joint for joint in scalar if joint.damping != 0.0]
    damping = (
        ScalarJointDampingPlan(
            np.asarray([joint.native_joint_id for joint in damped], dtype=np.int64),
            np.asarray([joint.damping for joint in damped], dtype=np.float64),
        )
        if damped and options.damping_executed
        else None
    )
    limited = [joint for joint in scalar if joint.limited]
    if not limited:
        return _JointLaws(damping, None, "none", (), ())
    if not options.limits_executed:
        losses = tuple(
            _loss(
                f"/joints/{joint.mujoco_joint_id}/range",
                "dropped",
                "the compiled limit is disabled by the source option flags and is "
                "retained only as evidence",
                changes_interpretation=False,
                affected=(_LIMIT_LAW_CAPABILITY_ID,),
            )
            for joint in limited
        )
        return _JointLaws(damping, None, "disabled-in-source", losses, ())
    if joint_stops is None:
        raise _refuse(
            "Executed MuJoCo joint limits are regularized constraints; pass an "
            "explicit MujocoJointStopTransformation to lower them to native hard "
            "stops."
        )
    margins = np.asarray(model.jnt_margin, dtype=np.float64)
    if _nonzero(margins[[joint.mujoco_joint_id for joint in limited]]):
        raise _refuse("Joint-limit margins have no native stop equivalent.")
    stops = ReducedJointStopPlan(
        np.asarray([joint.native_joint_id for joint in limited], dtype=np.int64),
        np.asarray([joint.range[0] - joint.qpos0 for joint in limited], dtype=np.float64),
        np.asarray([joint.range[1] - joint.qpos0 for joint in limited], dtype=np.float64),
    )
    loss = _loss(
        "/joints/limits",
        "transformed",
        "MuJoCo solref/solimp-regularized joint limits were replaced by native "
        "coupled hard unilateral stops on lifted coordinates",
        changes_interpretation=True,
        affected=(_LIMIT_LAW_CAPABILITY_ID,),
    )
    waiver = AdapterWaiver(loss, joint_stops.rationale)
    return _JointLaws(damping, stops, "native-hard-stops", (loss,), (waiver,))


@dataclass(frozen=True, slots=True)
class _Actuation:
    """Stateless joint motors in native actuator-ID order."""

    names: RobotNameIDMap | None
    rows: tuple[MujocoActuatorMap, ...]
    motors: ScalarJointMotorPlan | None


def _actuator_row(
    mujoco: ModuleType,
    model: Any,
    actuator: int,
    label: str,
    native_id: int,
    tree: _TreeJoints,
    clamp_control: bool,
    /,
) -> MujocoActuatorMap:
    """Refuse a stateful/biased actuator or lower one stateless joint motor."""

    affine_bias = int(mujoco.mjtBias.mjBIAS_AFFINE)
    if int(model.actuator_trntype[actuator]) != int(mujoco.mjtTrn.mjTRN_JOINT):
        raise _refuse(f"Actuator {label!r} does not use a joint transmission.")
    if int(model.actuator_dyntype[actuator]) != int(mujoco.mjtDyn.mjDYN_NONE):
        raise _refuse(f"Actuator {label!r} has stateful activation dynamics.")
    gain_parameters = np.asarray(model.actuator_gainprm[actuator], dtype=np.float64)
    if int(model.actuator_gaintype[actuator]) != int(
        mujoco.mjtGain.mjGAIN_FIXED
    ) or _nonzero(gain_parameters[1:]):
        raise _refuse(f"Actuator {label!r} has a state-dependent gain.")
    bias_type = int(model.actuator_biastype[actuator])
    if bias_type not in (int(mujoco.mjtBias.mjBIAS_NONE), affine_bias) or (
        bias_type == affine_bias and _nonzero(model.actuator_biasprm[actuator])
    ):
        raise _refuse(f"Actuator {label!r} has a nonzero bias (servo) law.")
    gear_vector = np.asarray(model.actuator_gear[actuator], dtype=np.float64)
    if _nonzero(gear_vector[1:]):
        raise _refuse(f"Actuator {label!r} has a non-scalar joint gear.")
    source_joint = int(model.actuator_trnid[actuator, 0])
    if source_joint not in tree.by_source:
        raise _refuse(f"Actuator {label!r} drives an unmapped joint.")
    gain = float(gain_parameters[0])
    gear = float(gear_vector[0])
    if gain == 0.0 or gear == 0.0:
        raise _refuse(f"Actuator {label!r} has a zero gain or gear.")
    unbounded = np.asarray([-np.inf, np.inf], dtype=np.float64)
    ctrl_range = (
        np.asarray(model.actuator_ctrlrange[actuator], dtype=np.float64)
        if clamp_control and bool(model.actuator_ctrllimited[actuator])
        else unbounded
    )
    force_range = unbounded
    effort_limit = np.inf
    if bool(model.actuator_forcelimited[actuator]):
        force_range = np.asarray(model.actuator_forcerange[actuator], dtype=np.float64)
        if force_range[0] != -force_range[1] or force_range[1] < 0.0:
            raise _refuse(
                f"Actuator {label!r} has an asymmetric force range, which the "
                "native symmetric effort clamp cannot represent."
            )
        effort_limit = abs(gear) * float(force_range[1])
    return MujocoActuatorMap(
        label,
        actuator,
        native_id,
        tree.by_source[source_joint].name,
        gain,
        gear,
        gain * gear,
        ctrl_range,
        force_range,
        effort_limit,
    )


def _actuation(
    mujoco: ModuleType,
    model: Any,
    names: _SourceNames,
    tree: _TreeJoints,
    options: _SourceOptions,
    /,
) -> _Actuation:
    if not int(model.nu):
        return _Actuation(None, (), None)
    actuator_names = RobotNameIDMap(tuple(sorted(names.actuators)))
    rows = sorted(
        (
            _actuator_row(
                mujoco,
                model,
                actuator,
                names.actuators[actuator],
                int(actuator_names.id_for_name(names.actuators[actuator])),
                tree,
                options.clamp_control,
            )
            for actuator in range(int(model.nu))
        ),
        key=lambda row: row.native_actuator_id,
    )
    motors = ScalarJointMotorPlan(
        np.asarray(
            [int(tree.names.id_for_name(row.joint)) for row in rows], dtype=np.int64
        ),
        np.asarray([row.native_gear for row in rows], dtype=np.float64),
        effort_limit=np.asarray(
            [row.native_effort_limit for row in rows], dtype=np.float64
        ),
        command_lower=np.asarray(
            [row.command_range[0] for row in rows], dtype=np.float64
        ),
        command_upper=np.asarray(
            [row.command_range[1] for row in rows], dtype=np.float64
        ),
    )
    return _Actuation(actuator_names, tuple(rows), motors)


def _frame_plan(
    model: Any, names: _SourceNames, frames: _BodyFrames, /
) -> tuple[RobotFramePlan, tuple[tuple[str, str, int], ...]]:
    """Every body origin and site, exactly in native COM body frames."""

    entries: list[tuple[str, str, int, np.ndarray, np.ndarray]] = [
        (
            _frame_name("body", names.bodies[body]),
            "body",
            body,
            frames.to_com_point(body, np.zeros((3,), dtype=np.float64)),
            frames.to_com_orientation(body, _IDENTITY_QUATERNION),
        )
        for body in range(1, int(model.nbody))
    ]
    site_body = np.asarray(model.site_bodyid, dtype=np.int64)
    for site in range(int(model.nsite)):
        body = int(site_body[site])
        if body == 0:
            raise _refuse(
                f"World site {names.sites[site]!r} is not attached to a native body."
            )
        entries.append(
            (
                _frame_name("site", names.sites[site]),
                "site",
                body,
                frames.to_com_point(body, model.site_pos[site]),
                frames.to_com_orientation(body, model.site_quat[site]),
            )
        )
    entries.sort(key=lambda entry: entry[0])
    plan = RobotFramePlan(
        RobotNameIDMap(tuple(entry[0] for entry in entries)),
        np.asarray([frames.native_id[entry[2]] for entry in entries], dtype=np.int64),
        np.asarray([entry[3] for entry in entries], dtype=np.float64).reshape((-1, 3)),
        np.asarray([entry[4] for entry in entries], dtype=np.float64).reshape((-1, 4)),
    )
    records = tuple((entry[0], entry[1], frames.native_id[entry[2]]) for entry in entries)
    return plan, records


@dataclass(frozen=True, slots=True)
class _ContactLowering:
    """Declared treatment of compiled contact and every geom's native role."""

    mode: MujocoContactMode
    plan: RobotCollisionGeometryPlan | None
    geoms: tuple[MujocoGeomMap, ...]
    losses: tuple[AdapterLoss, ...]
    waivers: tuple[AdapterWaiver, ...]


def _contact_lowering(
    mujoco: ModuleType,
    model: Any,
    names: _SourceNames,
    frames: _BodyFrames,
    options: _SourceOptions,
    contact: MujocoContactPolicy | None,
    maximum_geom_pair_checks: int,
    /,
) -> _ContactLowering:
    admission = _GeomPairAdmission.from_model(
        mujoco, model, options.filterparent, options.contact_enabled
    )
    colliding = admission.colliding_geoms(maximum_geom_pair_checks)
    geom_types = tuple(
        _enum_name(mujoco.mjtGeom, int(value), "mjGEOM_")
        for value in np.asarray(model.geom_type, dtype=np.int64)
    )
    roles: dict[int, tuple[MujocoGeomRole, str]] = {
        geom: ("non-colliding", "") for geom in range(int(model.ngeom))
    }
    mode: MujocoContactMode = "no-compiled-contact"
    plan: RobotCollisionGeometryPlan | None = None
    losses: tuple[AdapterLoss, ...] = ()
    waivers: tuple[AdapterWaiver, ...] = ()
    if colliding and contact is None:
        raise _refuse(
            "The compiled model executes MuJoCo contact; pass an explicit "
            "MujocoNativeContactTransformation or MujocoContactExclusion."
        )
    if colliding and isinstance(contact, MujocoContactExclusion):
        mode = "excluded"
        for geom in colliding:
            roles[geom] = ("excluded-contact", "")
        loss = _loss(
            "/contact",
            "dropped",
            "every compiled MuJoCo contact pair was removed from the native model",
            changes_interpretation=True,
            affected=(_CONTACT_LAW_CAPABILITY_ID,),
        )
        losses, waivers = (loss,), (AdapterWaiver(loss, contact.rationale),)
    elif colliding and isinstance(contact, MujocoNativeContactTransformation):
        mode = "native-inelastic-coulomb-hard-contact"
        plan = _native_collision_plan(
            model, contact, admission, colliding, names, geom_types, frames, roles
        )
        loss = _loss(
            "/contact",
            "transformed",
            "the MuJoCo soft-contact law (condim, solref/solimp, margin/gap, friction "
            "mixing, priority) was replaced by the native position-level inelastic "
            "hard-contact law with declared isotropic Coulomb materials; compiled "
            "geometry and pair filtering were retained",
            changes_interpretation=True,
            affected=(_CONTACT_LAW_CAPABILITY_ID,),
        )
        losses, waivers = (loss,), (AdapterWaiver(loss, contact.rationale),)
    geom_body = np.asarray(model.geom_bodyid, dtype=np.int64)
    geoms = tuple(
        MujocoGeomMap(
            names.geoms[geom],
            geom,
            geom_types[geom],
            names.bodies[int(geom_body[geom])] if int(geom_body[geom]) else "world",
            roles[geom][0],
            roles[geom][1],
        )
        for geom in range(int(model.ngeom))
    )
    return _ContactLowering(mode, plan, geoms, losses, waivers)


type _PrimitiveRow = tuple[
    str, ArticulatedPrimitiveKind, int, np.ndarray, np.ndarray, float, int
]
type _PlaneRow = tuple[str, np.ndarray, float, int]


def _lower_geoms(
    model: Any,
    policy: MujocoNativeContactTransformation,
    colliding: Sequence[int],
    names: _SourceNames,
    geom_types: Sequence[str],
    frames: _BodyFrames,
    roles: dict[int, tuple[MujocoGeomRole, str]],
    /,
) -> tuple[list[_PrimitiveRow], list[_PlaneRow], dict[int, int]]:
    """Colliding spheres/capsules in COM frames and world-static planes."""

    primitives: list[_PrimitiveRow] = []
    planes: list[_PlaneRow] = []
    owner: dict[int, int] = {}
    weld = np.asarray(model.body_weldid, dtype=np.int64)
    geom_body = np.asarray(model.geom_bodyid, dtype=np.int64)
    for geom in colliding:
        kind = geom_types[geom]
        name = names.geoms[geom]
        body = int(geom_body[geom])
        material = policy.material_for(name)
        position = np.asarray(model.geom_pos[geom], dtype=np.float64)
        quaternion = np.asarray(model.geom_quat[geom], dtype=np.float64)
        size = np.asarray(model.geom_size[geom], dtype=np.float64)
        if kind == "plane":
            if int(weld[body]) != 0:
                raise _refuse(f"Plane {name!r} is not world-static.")
            rotation = _rotation(frames.orientation[body])
            world_position = frames.position[body] + rotation @ position
            normal = rotation @ _rotation(quaternion)[:, 2]
            normal = normal / np.linalg.norm(normal)
            planes.append((name, normal, float(normal @ world_position), material))
            roles[geom] = ("plane", name)
            owner[geom] = frames.native_id[frames.root]
            continue
        if body == 0:
            raise _refuse(
                f"World {kind} {name!r} has no native body; attach static "
                "primitives to the fixed base."
            )
        if kind == "sphere":
            first = second = frames.to_com_point(body, position)
            primitive_kind = ArticulatedPrimitiveKind.SPHERE
        elif kind == "capsule":
            half = _rotation(quaternion)[:, 2] * float(size[1])
            first = frames.to_com_point(body, position - half)
            second = frames.to_com_point(body, position + half)
            primitive_kind = ArticulatedPrimitiveKind.CAPSULE
        else:
            raise _refuse(f"Colliding {kind} geom {name!r} has no native primitive.")
        primitives.append(
            (
                name,
                primitive_kind,
                frames.native_id[body],
                first,
                second,
                float(size[0]),
                material,
            )
        )
        roles[geom] = ("primitive", name)
        owner[geom] = frames.native_id[body]
    return primitives, planes, owner


def _excluded_body_pairs(
    admission: _GeomPairAdmission,
    geom_types: Sequence[str],
    owner: Mapping[int, int],
    /,
) -> list[tuple[int, int]]:
    """Native body pairs MuJoCo never admits; mixed admission is refused.

    Native candidates are every geometry pair between distinct native bodies
    (planes belong to the root) except excluded body pairs, so MuJoCo's
    admission must be all-or-none per native body pair. Each predicate is
    evaluated once on the fly, within the budget already checked over all
    compiled geoms; only excluded body pairs are kept.
    """

    by_body: dict[int, list[int]] = {}
    for geom, body in owner.items():
        by_body.setdefault(body, []).append(geom)
    excluded: list[tuple[int, int]] = []
    bodies = sorted(by_body)
    for index, left in enumerate(bodies):
        for right in bodies[index + 1 :]:
            admitted = False
            refused = False
            for first in by_body[left]:
                for second in by_body[right]:
                    if geom_types[first] == geom_types[second] == "plane":
                        continue
                    if admission.admitted(first, second):
                        admitted = True
                    else:
                        refused = True
                    if admitted and refused:
                        raise _refuse(
                            "MuJoCo geom-level pair filtering between bodies "
                            f"{left} and {right} is not representable by native "
                            "body-pair exclusions."
                        )
            if not admitted:
                excluded.append((left, right))
    return excluded


def _native_collision_plan(
    model: Any,
    policy: MujocoNativeContactTransformation,
    admission: _GeomPairAdmission,
    colliding: Sequence[int],
    names: _SourceNames,
    geom_types: Sequence[str],
    frames: _BodyFrames,
    roles: dict[int, tuple[MujocoGeomRole, str]],
    /,
) -> RobotCollisionGeometryPlan:
    """Lower colliding spheres/capsules/planes with exact body-pair filtering."""

    primitives, planes, owner = _lower_geoms(
        model, policy, colliding, names, geom_types, frames, roles
    )
    excluded = _excluded_body_pairs(admission, geom_types, owner)
    primitives.sort(key=lambda item: item[0])
    planes.sort(key=lambda item: item[0])
    return RobotCollisionGeometryPlan(
        RobotNameIDMap(tuple(item[0] for item in primitives)),
        tuple(item[1] for item in primitives),
        np.asarray([item[2] for item in primitives], dtype=np.int64),
        np.asarray([item[3] for item in primitives], dtype=np.float64).reshape((-1, 3)),
        np.asarray([item[4] for item in primitives], dtype=np.float64).reshape((-1, 3)),
        np.asarray([item[5] for item in primitives], dtype=np.float64),
        np.asarray([item[6] for item in primitives], dtype=np.int64),
        materials=policy.materials,
        plane_names=RobotNameIDMap(tuple(item[0] for item in planes)),
        plane_normals=np.asarray([item[1] for item in planes], dtype=np.float64).reshape(
            (-1, 3)
        ),
        plane_offsets=np.asarray([item[2] for item in planes], dtype=np.float64),
        plane_material_ids=np.asarray([item[3] for item in planes], dtype=np.int64),
        excluded_body_pairs=np.asarray(excluded, dtype=np.int64).reshape((-1, 2)),
    )


def _refuse_sensor_processing(model: Any, index: int, label: str, /) -> None:
    for field in (
        "sensor_noise",
        "sensor_cutoff",
        "sensor_delay",
        "sensor_interval",
        "sensor_history",
    ):
        if hasattr(model, field) and _nonzero(
            np.asarray(getattr(model, field), dtype=np.float64)[index]
        ):
            raise _refuse(
                f"Sensor {label!r} has noise, filtering, delay, interval or "
                "history, which are not native observations."
            )
    if hasattr(model, "sensor_plugin") and int(model.sensor_plugin[index]) >= 0:
        raise _refuse(f"Sensor {label!r} is a plugin sensor.")


def _frame_sensor_target(
    mujoco: ModuleType, model: Any, index: int, label: str, names: _SourceNames, /
) -> str:
    objects = mujoco.mjtObj
    if (
        int(model.sensor_reftype[index]) != int(objects.mjOBJ_UNKNOWN)
        and int(model.sensor_refid[index]) >= 0
    ):
        raise _refuse(f"Sensor {label!r} uses a non-world reference frame.")
    object_type = int(model.sensor_objtype[index])
    object_id = int(model.sensor_objid[index])
    if object_type == int(objects.mjOBJ_SITE):
        return _frame_name("site", names.sites[object_id])
    if object_type == int(objects.mjOBJ_XBODY) and object_id > 0:
        return _frame_name("body", names.bodies[object_id])
    if object_type == int(objects.mjOBJ_BODY) and object_id > 0:
        return f"com:{names.bodies[object_id]}"
    raise _refuse(f"Sensor {label!r} observes an unmapped frame object.")


def _sensor_maps(
    mujoco: ModuleType, model: Any, names: _SourceNames, tree: _TreeJoints, /
) -> tuple[MujocoSensorMap, ...]:
    sensor = mujoco.mjtSensor
    frame_quantities: dict[int, tuple[str, str, MujocoSensorStage]] = {
        int(sensor.mjSENS_FRAMEPOS): ("frame-position", "m", "position"),
        int(sensor.mjSENS_FRAMEQUAT): ("frame-quaternion", "1", "position"),
        int(sensor.mjSENS_FRAMELINVEL): ("frame-linear-velocity", "m/s", "velocity"),
        int(sensor.mjSENS_FRAMEANGVEL): ("frame-angular-velocity", "rad/s", "velocity"),
    }
    joint_quantities: dict[int, tuple[str, str, MujocoSensorStage]] = {
        int(sensor.mjSENS_JOINTPOS): ("joint-position", "", "position"),
        int(sensor.mjSENS_JOINTVEL): ("joint-velocity", "/s", "velocity"),
    }
    maps: list[MujocoSensorMap] = []
    for index in range(int(model.nsensor)):
        label = names.sensors[index]
        _refuse_sensor_processing(model, index, label)
        kind = int(model.sensor_type[index])
        address = int(model.sensor_adr[index])
        dimension = int(model.sensor_dim[index])
        stage: MujocoSensorStage
        if kind in joint_quantities:
            object_id = int(model.sensor_objid[index])
            if object_id not in tree.by_source:
                raise _refuse(f"Sensor {label!r} observes an unmapped joint.")
            joint = tree.by_source[object_id]
            quantity, rate, stage = joint_quantities[kind]
            units = ("m" if joint.kind == "prismatic" else "rad") + rate
            target = f"joint:{joint.name}"
        elif kind == int(sensor.mjSENS_CLOCK):
            quantity, target, units, dimension = "clock", "time", "s", 1
            stage = "clock"
        elif kind in frame_quantities:
            quantity, units, stage = frame_quantities[kind]
            target = _frame_sensor_target(mujoco, model, index, label, names)
        else:
            raise _refuse(
                f"Sensor {label!r} of type "
                f"{_enum_name(sensor, kind, 'mjSENS_')!r} has no native observation."
            )
        maps.append(
            MujocoSensorMap(
                label, index, quantity, target, units, stage, address, dimension
            )
        )
    return tuple(maps)


def _adapter_report(
    source_id: str,
    target_id: str,
    version: str,
    contact_mode: MujocoContactMode,
    semantics: Sequence[str],
    source_laws: Sequence[str],
    losses: Sequence[AdapterLoss],
    waivers: Sequence[AdapterWaiver],
    /,
) -> AdapterReport:
    """Negotiate executed native semantics; replaced source laws are optional."""

    requirements = tuple(AdapterRequirement(semantic) for semantic in semantics) + tuple(
        AdapterRequirement(
            law,
            required=False,
            rationale="the source law is not preserved; the native model executes "
            "its own declared law",
        )
        for law in source_laws
    )
    capabilities = tuple(_CAPABILITIES[semantic] for semantic in semantics)
    negotiation = negotiate_adapter(
        requirements, capabilities, losses=losses, waivers=waivers
    )
    return AdapterReport(
        negotiation.status,
        _SOURCE_FORMAT,
        _TARGET_FORMAT,
        source_id=source_id,
        target_id=target_id,
        stage="host-import",
        source_profile=AdapterFormatProfile(
            _SOURCE_FORMAT,
            qualifiers={
                "execution": "host-only",
                "provider": f"mujoco:{version}",
                "units": "MuJoCo SI base units",
            },
        ),
        target_profile=AdapterFormatProfile(
            _TARGET_FORMAT,
            qualifiers={
                "base": "fixed",
                "body-frame": "center-of-mass",
                "coordinates": "lifted q = qpos - qpos0",
                "contact-law": contact_mode,
            },
        ),
        coordinate_mapping=(
            "compiled body_pos/body_quat at qpos0 -> zero-configuration body frames",
            "compiled body_ipos/body_iquat -> COM-centered native body pose and axes",
            "compiled principal body_inertia -> diagonal COM tensor in inertial axes",
            "compiled jnt_pos/jnt_axis -> world zero-configuration anchor and axis",
            "qpos - qpos0 -> native lifted coordinate; qvel -> native velocity",
            "gain*gear, |gear|*forcerange, ctrlrange -> native motor transmission",
            "body origins and sites -> frames in native COM body coordinates",
        ),
        preserved_fields=(
            "body, joint, actuator, geom, site and sensor names and source indices",
            "compiled masses, COM positions, inertial frames and principal inertias",
            "hinge/slide kinematics and qpos0 reference offsets",
            "stateless joint-motor gain, gear, control and force clamps",
            "executed viscous joint damping",
            "site and body-origin frames",
        ),
        assumptions=(
            "compiled MuJoCo quantities are SI (m, kg, s, rad)",
            "the compiled model is the single authority; no XML is parsed",
            "timestep, integrator and gravity are retained as source evidence "
            "and are runtime choices of any native execution",
        ),
        losses=losses,
        requirements=requirements,
        capabilities=capabilities,
        waivers=waivers,
    )


def _executed_semantics(
    actuation: _Actuation,
    laws: _JointLaws,
    contact: _ContactLowering,
    sensors: Sequence[MujocoSensorMap],
    /,
) -> tuple[list[str], list[str]]:
    """Required executed native semantics and replaced optional source laws."""

    semantics = [
        "robot.connected-oriented-tree",
        "robot.inertial-rigid-links",
        "robot.fixed-hinge-prismatic-kinematics",
        "robot.reference-kinematics",
        "robot.si-dimensional-contract",
        "robot.stable-int64-name-maps",
        "robot.source-index-maps",
        "robot.fixed-world-root",
    ]
    optional_entries = (
        (actuation.motors is not None, "robot.actuation-mapping"),
        (laws.damping is not None, "robot.executed-joint-damping"),
        (laws.stops is not None, "robot.unilateral-joint-stops"),
        (contact.plan is not None, "robot.collision-geometry"),
        (bool(sensors), "robot.sensor-selection"),
    )
    semantics.extend(semantic for present, semantic in optional_entries if present)
    source_laws: list[str] = []
    if laws.limit_mode != "none":
        source_laws.append(_LIMIT_LAW_SEMANTIC)
    if contact.mode != "no-compiled-contact":
        source_laws.append(_CONTACT_LAW_SEMANTIC)
    return semantics, source_laws


def _validated_mujoco(
    model: Any,
    contact: MujocoContactPolicy | None,
    joint_stops: MujocoJointStopTransformation | None,
    maximum_geom_pair_checks: int,
    /,
) -> tuple[ModuleType, str]:
    """Import the probed provider and validate every adaptation argument."""

    if (
        isinstance(maximum_geom_pair_checks, bool)
        or not isinstance(maximum_geom_pair_checks, int)
        or maximum_geom_pair_checks < 0
    ):
        raise ValueError("maximum_geom_pair_checks must be a nonnegative integer.")
    availability = mujoco_host_availability()
    if not availability.available:
        raise MujocoModelAdaptationError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE, availability.reason
        )
    mujoco = import_backend_module(availability, "robotics.model-adaptation", "mujoco")
    if not isinstance(model, mujoco.MjModel):
        raise TypeError("model must be an already compiled mujoco.MjModel.")
    if contact is not None and not isinstance(
        contact, (MujocoContactExclusion, MujocoNativeContactTransformation)
    ):
        raise TypeError("contact must be a declared MuJoCo contact policy.")
    if joint_stops is not None and not isinstance(
        joint_stops, MujocoJointStopTransformation
    ):
        raise TypeError("joint_stops must be a MujocoJointStopTransformation.")
    version = dict(availability.versions).get("mujoco", str(mujoco.__version__))
    return mujoco, version


def adapt_mujoco_model(
    model: Any,
    /,
    *,
    contact: MujocoContactPolicy | None = None,
    joint_stops: MujocoJointStopTransformation | None = None,
    waivers: Sequence[AdapterWaiver] = (),
    maximum_geom_pair_checks: int = 1 << 22,
) -> RobotAdaptation[MujocoModelEvidence]:
    """Adapt one compiled ``mujoco.MjModel`` to a provider-free native model.

    Refuses every compiled feature outside the closed profile with
    `MujocoModelAdaptationError`. Executed MuJoCo limits require
    ``joint_stops``; any compiled collision pair requires ``contact``. Each
    declared transformation or exclusion is an interpretation-changing loss of
    the optional source-law semantic, waived by its own rationale; caller
    ``waivers`` may accept any remaining interpretation-changing loss.

    Reproducing MuJoCo's static pair filtering is a temporary host scan of at
    most ``G(G-1)/2`` pair predicates with ``O(G)`` memory for ``G`` compiled
    geoms; a model whose scan exceeds ``maximum_geom_pair_checks`` is refused
    before scanning. No pair table is materialized or retained: the native
    model receives only sparse excluded body pairs.
    """

    mujoco, version = _validated_mujoco(
        model, contact, joint_stops, maximum_geom_pair_checks
    )
    _refuse_unsupported_inventory(mujoco, model)
    options = _source_options(mujoco, model)
    names = _source_names(mujoco, model)
    root, parent, joint_count = _fixed_base(mujoco, model, names)
    frames = _body_frames(model, names, root, parent, joint_count)
    tree = _tree_joints(mujoco, model, names, frames)
    laws = _joint_laws(model, tree, options, joint_stops)
    actuation = _actuation(mujoco, model, names, tree, options)
    frame_plan, frame_records = _frame_plan(model, names, frames)
    lowering = _contact_lowering(
        mujoco, model, names, frames, options, contact, maximum_geom_pair_checks
    )
    sensors = _sensor_maps(mujoco, model, names, tree)
    native = lower_native_robot_tree(
        "mujoco",
        frames.names,
        frames.masses,
        frames.inertia,
        frames.reference_position,
        frames.reference_orientation,
        names.bodies[root],
        tree.edges,
        DimensionalScaleContract.si(),
        frame_plan,
        actuator_names=actuation.names,
        motors=actuation.motors,
        damping=laws.damping,
        stops=laws.stops,
        collision=lowering.plan,
    )
    digest = _model_digest(mujoco, model)
    source_id = f"mujoco-mjb-sha256:{digest}"
    losses = names.losses + laws.losses + lowering.losses
    applied_waivers = tuple(waivers) + laws.waivers + lowering.waivers
    evidence = MujocoModelEvidence(
        version,
        digest,
        names.bodies[root],
        options.timestep,
        options.integrator,
        options.gravity,
        options.disabled,
        options.enabled,
        lowering.mode,
        laws.limit_mode,
        options.damping_executed,
        options.clamp_control,
        frames.maps,
        sorted(tree.maps, key=lambda item: item.native_joint_id),
        actuation.rows,
        lowering.geoms,
        sensors,
        frame_records,
        tuple(loss.path for loss in losses),
        source_id,
        native.model_id,
    )
    semantics, source_laws = _executed_semantics(actuation, laws, lowering, sensors)
    report = _adapter_report(
        source_id,
        native.model_id,
        version,
        lowering.mode,
        semantics,
        source_laws,
        losses,
        applied_waivers,
    )
    if not report.negotiation.valid or not report.valid:
        raise MujocoModelAdaptationError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "The compiled model has an unwaived interpretation-changing loss or a "
            "stale waiver.",
            report=report,
            evidence=evidence,
        )
    return RobotAdaptation(native, report, evidence)


__all__ = [
    "MUJOCO_HOST_CAPABILITIES",
    "MujocoActuatorMap",
    "MujocoBodyMap",
    "MujocoContactExclusion",
    "MujocoContactMode",
    "MujocoContactPolicy",
    "MujocoGeomMap",
    "MujocoGeomRole",
    "MujocoJointKind",
    "MujocoJointLimitMode",
    "MujocoJointMap",
    "MujocoJointStopTransformation",
    "MujocoModelAdaptationError",
    "MujocoModelEvidence",
    "MujocoNativeContactTransformation",
    "MujocoSensorMap",
    "MujocoSensorStage",
    "adapt_mujoco_model",
    "mujoco_host_availability",
]
