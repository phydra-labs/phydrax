#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Secure host-only adaptation of the supported URDF 1.0 robot subset."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from math import cos, inf, isfinite, isnan, sin
from pathlib import Path
from typing import assert_never, cast, Literal, Never

import equinox as eqx
import numpy as np

from ..._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    read_bounded_resource,
    ResourceLimits,
    ResourceManifest,
    ResourceReadError,
)
from ..._fingerprint import canonical_fingerprint
from ..._physical import DimensionalScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.contact._articulated_geometry import ArticulatedPrimitiveKind
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
    NativeTreeJointKind,
    RobotAdaptation,
    RobotCollisionGeometryPlan,
    RobotContactMaterialTable,
    RobotFramePlan,
    RobotNameIDMap,
)


_DEFAULT_MAX_BYTES = 4 * 1024 * 1024
_DEFAULT_MAX_DEPTH = 64
_DEFAULT_MAX_NODES = 100_000
_DEFAULT_MAX_ATTRIBUTES = 500_000
_DEFAULT_MAX_LOSSES = 4_096
_FLOAT_TOKEN = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII)
_UNSAFE_DECLARATION = re.compile(
    r"<!\s*(?:DOCTYPE|ENTITY|ELEMENT|ATTLIST|NOTATION)|<!\[CDATA\[",
    re.IGNORECASE,
)
_PROCESSING_INSTRUCTION = re.compile(r"<\?(?!xml(?:\s|\?>))", re.IGNORECASE)
_XML_DECLARATION = re.compile(
    r"<\?xml\s+version\s*=\s*(['\"])1\.0\1"
    r"(?:\s+encoding\s*=\s*(['\"])UTF-8\2)?"
    r"(?:\s+standalone\s*=\s*(['\"])(?:yes|no)\3)?\s*\?>",
    re.IGNORECASE,
)
_SUPPORTED_JOINT_KINDS = frozenset(("fixed", "revolute", "continuous", "prismatic"))
_URDFRootPolicy = Literal["fixed_world", "reject_unpinned"]
type URDFJointStopPolicy = Literal["evidence-only", "native-unilateral-stops"]
_URDF_CAPABILITIES = (
    AdapterCapability("robot.connected-oriented-tree"),
    AdapterCapability("robot.inertial-rigid-links"),
    AdapterCapability("robot.fixed-hinge-prismatic-kinematics"),
    AdapterCapability("robot.reference-kinematics"),
    AdapterCapability("robot.joint-limit-evidence"),
    AdapterCapability("robot.executed-joint-damping"),
    AdapterCapability("robot.si-dimensional-contract"),
    AdapterCapability("robot.stable-int64-name-maps"),
    AdapterCapability("robot.fixed-world-root"),
)
_URDF_CAPABILITY_IDS = {
    capability.semantic_id: capability.capability_id for capability in _URDF_CAPABILITIES
}
_EXTENSION_CAPABILITY_ID = AdapterCapability(
    "robot.source-extension-semantics"
).capability_id
_VISUAL_CAPABILITY_ID = AdapterCapability("robot.visual-description").capability_id
_COLLISION_CAPABILITY = AdapterCapability("robot.collision-geometry")
_ACTUATION_CAPABILITY = AdapterCapability("robot.actuation-mapping")
_STOP_CAPABILITY = AdapterCapability("robot.unilateral-joint-stops")
_COLLISION_CAPABILITY_ID = _COLLISION_CAPABILITY.capability_id
_ACTUATION_CAPABILITY_ID = _ACTUATION_CAPABILITY.capability_id
_FRICTION_CAPABILITY_ID = AdapterCapability("robot.joint-friction").capability_id
_IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0)


class URDFLinkEvidence(StrictModule, NonTrainableState):
    """Exact link-frame information retained beside COM-centered native bodies."""

    name: str = eqx.field(static=True)
    body_id: int = eqx.field(static=True)
    mass_kg: float = eqx.field(static=True)
    com_in_link_frame_m: tuple[float, float, float] = eqx.field(static=True)
    link_frame_in_body_m: tuple[float, float, float] = eqx.field(static=True)
    inertial_frame_rpy_rad: tuple[float, float, float] = eqx.field(static=True)
    inertia_in_inertial_frame_kg_m2: tuple[float, ...] = eqx.field(static=True)
    inertia_in_body_frame_kg_m2: tuple[float, ...] = eqx.field(static=True)
    reference_link_position_m: tuple[float, float, float] = eqx.field(static=True)
    reference_body_position_m: tuple[float, float, float] = eqx.field(static=True)
    reference_orientation_wxyz: tuple[float, float, float, float] = eqx.field(static=True)


class URDFJointEvidence(StrictModule, NonTrainableState):
    """URDF joint data, including limits and damping not owned by tree topology."""

    name: str = eqx.field(static=True)
    joint_id: int = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    parent_link: str = eqx.field(static=True)
    parent_body_id: int = eqx.field(static=True)
    child_link: str = eqx.field(static=True)
    child_body_id: int = eqx.field(static=True)
    origin_xyz_m: tuple[float, float, float] = eqx.field(static=True)
    origin_rpy_rad: tuple[float, float, float] = eqx.field(static=True)
    reference_anchor_m: tuple[float, float, float] = eqx.field(static=True)
    axis_in_joint_frame: tuple[float, float, float] = eqx.field(static=True)
    reference_axis: tuple[float, float, float] = eqx.field(static=True)
    lower_limit: float | None = eqx.field(static=True)
    upper_limit: float | None = eqx.field(static=True)
    effort_limit: float | None = eqx.field(static=True)
    velocity_limit: float | None = eqx.field(static=True)
    damping: float | None = eqx.field(static=True)


class URDFCollisionEvidence(StrictModule, NonTrainableState):
    """One URDF collision sphere lowered into a native body-attached sphere."""

    primitive_name: str = eqx.field(static=True)
    primitive_id: int = eqx.field(static=True)
    link: str = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    source_name: str | None = eqx.field(static=True)
    radius_m: float = eqx.field(static=True)
    center_in_link_frame_m: tuple[float, float, float] = eqx.field(static=True)
    origin_rpy_rad: tuple[float, float, float] = eqx.field(static=True)
    center_in_body_frame_m: tuple[float, float, float] = eqx.field(static=True)
    material_id: int = eqx.field(static=True)


class URDFJointMotor(StrictModule, NonTrainableState):
    """Explicit caller mapping of one stateless motor onto a URDF joint.

    URDF transmissions are not interpreted. The motor's effort clamp is the
    joint's URDF ``<limit effort>``; a zero effort limit lowers to an
    explicitly disabled motor. ``gear`` is signed and command bounds may be
    infinite.
    """

    name: str = eqx.field(static=True)
    joint: str = eqx.field(static=True)
    gear: float = eqx.field(static=True)
    command_lower: float = eqx.field(static=True)
    command_upper: float = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        joint: str,
        /,
        *,
        gear: float = 1.0,
        command_lower: float = -inf,
        command_upper: float = inf,
    ) -> None:
        name_ = str(name)
        joint_ = str(joint)
        if not name_ or not joint_:
            raise ValueError("URDF motor and joint names must be non-empty.")
        gear_ = float(gear)
        lower = float(command_lower)
        upper = float(command_upper)
        if not isfinite(gear_) or gear_ == 0.0:
            raise ValueError("URDF motor gear must be finite and nonzero.")
        if isnan(lower) or isnan(upper) or not lower < upper:
            raise ValueError("URDF motor command_lower must lie below command_upper.")
        self.name = name_
        self.joint = joint_
        self.gear = gear_
        self.command_lower = lower
        self.command_upper = upper


class URDFStaticPlane(StrictModule, NonTrainableState):
    """A world-fixed plane ``n . x = offset`` with free space on ``n . x > offset``."""

    name: str = eqx.field(static=True)
    normal: tuple[float, float, float] = eqx.field(static=True)
    offset: float = eqx.field(static=True)
    material_id: int = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        normal: Sequence[float],
        offset: float,
        material_id: int,
        /,
    ) -> None:
        name_ = str(name)
        values = tuple(float(value) for value in normal)
        if not name_ or len(values) != 3:
            raise ValueError("URDF static planes need a name and a 3-vector normal.")
        self.name = name_
        self.normal = (values[0], values[1], values[2])
        self.offset = float(offset)
        self.material_id = int(material_id)


class URDFCollisionPolicy(StrictModule, NonTrainableState):
    """Explicit native contact declaration for URDF collision lowering.

    URDF has no contact law, so the caller supplies the native Coulomb
    material table, one material per link that owns a lowered sphere, any
    static planes, and the excluded link pairs. Only URDF ``<sphere>``
    collision geometry is lowered; every other shape is an unwaivable loss
    once collision lowering is requested.
    """

    materials: RobotContactMaterialTable
    planes: tuple[URDFStaticPlane, ...]
    link_materials: tuple[tuple[str, int], ...] = eqx.field(static=True)
    excluded_link_pairs: tuple[tuple[str, str], ...] = eqx.field(static=True)
    exclude_jointed_links: bool = eqx.field(static=True)

    def __init__(
        self,
        materials: RobotContactMaterialTable,
        link_materials: Mapping[str, int],
        /,
        *,
        planes: Sequence[URDFStaticPlane] = (),
        excluded_link_pairs: Sequence[tuple[str, str]] = (),
        exclude_jointed_links: bool = False,
    ) -> None:
        if not isinstance(materials, RobotContactMaterialTable):
            raise TypeError("materials must be a RobotContactMaterialTable.")
        planes_ = tuple(planes)
        if not all(isinstance(plane, URDFStaticPlane) for plane in planes_):
            raise TypeError("planes must contain URDFStaticPlane values.")
        if len({plane.name for plane in planes_}) != len(planes_):
            raise ValueError("URDF static plane names must be unique.")
        pairs = tuple(
            sorted(
                (min(str(first), str(second)), max(str(first), str(second)))
                for first, second in excluded_link_pairs
            )
        )
        if len(set(pairs)) != len(pairs) or any(
            first == second for first, second in pairs
        ):
            raise ValueError("excluded_link_pairs must be unique pairs of distinct links.")
        self.materials = materials
        self.planes = tuple(sorted(planes_, key=lambda plane: plane.name))
        self.link_materials = tuple(
            sorted((str(link), int(material)) for link, material in link_materials.items())
        )
        self.excluded_link_pairs = pairs
        self.exclude_jointed_links = bool(exclude_jointed_links)


class URDFFormatEvidence(AbstractRobotSourceEvidence):
    """Auditable format, security, topology, mapping, and frame evidence."""

    robot_name: str = eqx.field(static=True)
    format_name: str = eqx.field(static=True)
    format_version: str = eqx.field(static=True)
    source_path: str | None = eqx.field(static=True)
    source_size_bytes: int = eqx.field(static=True)
    source_bytes: bytes = eqx.field(static=True)
    source_content_sha256: str = eqx.field(static=True)
    resource_manifest: ResourceManifest = eqx.field(static=True)
    root_policy: _URDFRootPolicy = eqx.field(static=True)
    joint_stop_policy: URDFJointStopPolicy = eqx.field(static=True)
    root_link: str = eqx.field(static=True)
    link_count: int = eqx.field(static=True)
    joint_count: int = eqx.field(static=True)
    link_ids: RobotNameIDMap
    joint_ids: RobotNameIDMap
    links: tuple[URDFLinkEvidence, ...]
    joints: tuple[URDFJointEvidence, ...]
    collisions: tuple[URDFCollisionEvidence, ...]
    motors: tuple[URDFJointMotor, ...]
    execution_policy: tuple[str, ...] = eqx.field(static=True)
    loss_paths: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        robot_name: str,
        resource: BoundedResource,
        root_policy: _URDFRootPolicy,
        joint_stop_policy: URDFJointStopPolicy,
        root_link: str,
        link_ids: RobotNameIDMap,
        joint_ids: RobotNameIDMap,
        links: Sequence[URDFLinkEvidence],
        joints: Sequence[URDFJointEvidence],
        collisions: Sequence[URDFCollisionEvidence],
        motors: Sequence[URDFJointMotor],
        loss_paths: Sequence[str],
        source_id: str,
        target_id: str,
        /,
    ) -> None:
        links_ = tuple(links)
        joints_ = tuple(joints)
        collisions_ = tuple(collisions)
        motors_ = tuple(motors)
        manifest = resource.manifest
        policies = (
            "host-only parsing",
            "network access disabled",
            "DTD and entity declarations disabled",
            "xacro and include expansion disabled",
            "plugin execution disabled",
            "URDF joint damping executed as native viscous damping",
            f"URDF position limits lowered as {joint_stop_policy}",
            "motors only from explicit caller joint mappings",
            "collision spheres only under an explicit native contact policy",
            f"root interpretation explicitly selected as {root_policy}",
        )
        paths = tuple(loss_paths)
        self.robot_name = robot_name
        self.format_name = "URDF"
        self.format_version = "1.0"
        self.source_path = manifest.source_path
        self.source_size_bytes = manifest.size_bytes
        self.source_bytes = resource.data
        self.source_content_sha256 = manifest.content_sha256
        self.resource_manifest = manifest
        self.root_policy = root_policy
        self.joint_stop_policy = joint_stop_policy
        self.root_link = root_link
        self.link_count = len(links_)
        self.joint_count = len(joints_)
        self.link_ids = link_ids
        self.joint_ids = joint_ids
        self.links = links_
        self.joints = joints_
        self.collisions = collisions_
        self.motors = motors_
        self.execution_policy = policies
        self.loss_paths = paths
        self.source_id = source_id
        self.target_id = target_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "urdf-adapter-evidence",
                "format": "URDF 1.0",
                "robot_name": robot_name,
                "source_path": manifest.source_path,
                "source_size_bytes": manifest.size_bytes,
                "source_content_sha256": manifest.content_sha256,
                "resource_manifest": manifest.manifest_id,
                "root_policy": root_policy,
                "joint_stop_policy": joint_stop_policy,
                "root_link": root_link,
                "link_mapping": link_ids.mapping_id,
                "joint_mapping": joint_ids.mapping_id,
                "links": [_link_evidence_payload(item) for item in links_],
                "joints": [_joint_evidence_payload(item) for item in joints_],
                "collisions": [
                    _collision_evidence_payload(item) for item in collisions_
                ],
                "motors": [_motor_evidence_payload(item) for item in motors_],
                "execution_policy": list(policies),
                "loss_paths": list(paths),
                "source_id": source_id,
                "target_id": target_id,
            }
        )


class URDFImportError(AdapterError):
    """Fail-closed URDF error with an optional canonical report and evidence."""

    report: AdapterReport | None
    evidence: URDFFormatEvidence | None

    def __init__(
        self,
        status: AdapterStatus,
        message: str,
        /,
        *,
        report: AdapterReport | None = None,
        evidence: URDFFormatEvidence | None = None,
    ) -> None:
        self.report = report
        self.evidence = evidence
        super().__init__(status, message)


@dataclass(frozen=True, slots=True)
class _CollisionRecord:
    index: int
    source_name: str | None
    xyz: np.ndarray
    rpy: np.ndarray
    radius: float


@dataclass(frozen=True, slots=True)
class _LinkRecord:
    name: str
    mass: float
    inertial_xyz: np.ndarray
    inertial_rpy: np.ndarray
    inertia_inertial: np.ndarray
    inertia_body: np.ndarray
    collisions: tuple[_CollisionRecord, ...]


@dataclass(frozen=True, slots=True)
class _JointRecord:
    name: str
    kind: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    damping: float | None


@dataclass(frozen=True, slots=True)
class _ParsedURDF:
    robot_name: str
    links: tuple[_LinkRecord, ...]
    joints: tuple[_JointRecord, ...]
    root_link: str
    traversal: tuple[_JointRecord, ...]
    losses: tuple[AdapterLoss, ...]


@dataclass(slots=True)
class _LossAccumulator:
    maximum: int
    items: list[AdapterLoss]

    def __init__(self, maximum: int, /) -> None:
        self.maximum = int(maximum)
        self.items = []

    def append(self, loss: AdapterLoss, /) -> None:
        if len(self.items) >= self.maximum:
            _fail(
                AdapterStatus.MALFORMED_SOURCE,
                f"URDF semantic losses exceed the configured {self.maximum}-loss limit.",
            )
        self.items.append(loss)


def _fail(status: AdapterStatus, message: str, /) -> Never:
    raise URDFImportError(status, message)


def _pointer(value: str, /) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _tag_name(element: ET.Element, /) -> str:
    tag = element.tag
    return tag if isinstance(tag, str) else "non-element"


def _check_leaf_text(element: ET.Element, path: str, /) -> None:
    if element.text is not None and element.text.strip():
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path} cannot contain text content.")
    for child in element:
        if child.tail is not None and child.tail.strip():
            _fail(AdapterStatus.MALFORMED_SOURCE, f"{path} cannot contain mixed text.")


def _float(token: str, path: str, /) -> float:
    token_ = str(token).strip()
    if _FLOAT_TOKEN.fullmatch(token_) is None:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path} must be a finite decimal number.")
    value = float(token_)
    if not isfinite(value):
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path} must be finite.")
    return value


def _vector(token: str, size: int, path: str, /) -> np.ndarray:
    fields = str(token).split()
    if len(fields) != size:
        _fail(
            AdapterStatus.MALFORMED_SOURCE, f"{path} must contain exactly {size} numbers."
        )
    return np.asarray(
        [_float(field, f"{path}[{index}]") for index, field in enumerate(fields)]
    )


def _required_attribute(element: ET.Element, name: str, path: str, /) -> str:
    if name not in element.attrib:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path}/@{name} is required.")
    value = element.attrib[name]
    if not value or value != value.strip():
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"{path}/@{name} must be non-empty and unpadded.",
        )
    return value


def _loss(
    path: str,
    rationale: str,
    /,
    *,
    changes_interpretation: bool,
    category: _AdapterLossCategory = "unsupported",
    affected_capability_ids: Sequence[str] = (_EXTENSION_CAPABILITY_ID,),
) -> AdapterLoss:
    return AdapterLoss(
        path,
        "import",
        category,
        rationale,
        changes_interpretation=changes_interpretation,
        affected_capability_ids=affected_capability_ids,
    )


def _attribute_losses(
    element: ET.Element,
    allowed: frozenset[str],
    path: str,
    losses: _LossAccumulator,
    /,
) -> None:
    for name in sorted(set(element.attrib) - allowed):
        losses.append(
            _loss(
                f"{path}/@{_pointer(name)}",
                "This attribute is outside the supported URDF subset and is not lowered.",
                changes_interpretation=True,
            )
        )


def _unknown_child_losses(
    element: ET.Element,
    allowed: frozenset[str],
    path: str,
    losses: _LossAccumulator,
    /,
) -> None:
    counts: dict[str, int] = {}
    for child in element:
        tag = _tag_name(child)
        if tag in allowed:
            continue
        index = counts.get(tag, 0)
        counts[tag] = index + 1
        losses.append(
            _loss(
                f"{path}/extensions/{_pointer(tag)}/{index}",
                "This extension subtree is retained only as declared format loss; it is never executed or expanded.",
                changes_interpretation=True,
            )
        )


def _children(element: ET.Element, tag: str, /) -> tuple[ET.Element, ...]:
    return tuple(child for child in element if _tag_name(child) == tag)


def _single_child(
    element: ET.Element,
    tag: str,
    path: str,
    /,
    *,
    required: bool,
) -> ET.Element | None:
    matches = _children(element, tag)
    if len(matches) > 1:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path}/{tag} may occur at most once.")
    if required and not matches:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path}/{tag} is required.")
    return None if not matches else matches[0]


def _parse_origin(
    owner: ET.Element,
    owner_path: str,
    losses: _LossAccumulator,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    origin = _single_child(owner, "origin", owner_path, required=False)
    if origin is None:
        return np.zeros((3,), dtype=np.float64), np.zeros((3,), dtype=np.float64)
    path = f"{owner_path}/origin"
    _attribute_losses(origin, frozenset(("xyz", "rpy")), path, losses)
    _unknown_child_losses(origin, frozenset(), path, losses)
    _check_leaf_text(origin, path)
    xyz = (
        np.zeros((3,), dtype=np.float64)
        if "xyz" not in origin.attrib
        else _vector(origin.attrib["xyz"], 3, f"{path}/@xyz")
    )
    rpy = (
        np.zeros((3,), dtype=np.float64)
        if "rpy" not in origin.attrib
        else _vector(origin.attrib["rpy"], 3, f"{path}/@rpy")
    )
    return xyz, rpy


def _rotation_from_rpy(rpy: np.ndarray, /) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = cos(roll), sin(roll)
    cp, sp = cos(pitch), sin(pitch)
    cy, sy = cos(yaw), sin(yaw)
    return np.asarray(
        (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        ),
        dtype=np.float64,
    )


def _quaternion_from_rotation(rotation: np.ndarray, /) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(1.0 + trace)
        quaternion = np.asarray(
            (
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            )
        )
    else:
        diagonal = int(np.argmax(np.diag(rotation)))
        if diagonal == 0:
            scale = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif diagonal == 1:
            scale = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.asarray(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.asarray(
                (
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0.0 else -quaternion


def _parse_collision(
    element: ET.Element,
    path: str,
    index: int,
    losses: _LossAccumulator,
    requested: bool,
    /,
) -> _CollisionRecord | None:
    if not requested:
        losses.append(
            _loss(
                path,
                "Collision geometry is not lowered because no native collision "
                "policy was supplied.",
                changes_interpretation=True,
                category="dropped",
                affected_capability_ids=(_COLLISION_CAPABILITY_ID,),
            )
        )
        return None
    _check_leaf_text(element, path)
    _attribute_losses(element, frozenset(("name",)), path, losses)
    _unknown_child_losses(element, frozenset(("origin", "geometry")), path, losses)
    xyz, rpy = _parse_origin(element, path, losses)
    geometry = _single_child(element, "geometry", path, required=True)
    if geometry is None:
        raise RuntimeError("A required URDF collision geometry was not returned.")
    geometry_path = f"{path}/geometry"
    _check_leaf_text(geometry, geometry_path)
    _attribute_losses(geometry, frozenset(), geometry_path, losses)
    shapes = tuple(geometry)
    if len(shapes) != 1:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"{geometry_path} must contain exactly one shape.",
        )
    shape = shapes[0]
    tag = _tag_name(shape)
    if tag != "sphere":
        losses.append(
            _loss(
                f"{geometry_path}/{_pointer(tag)}",
                f"URDF {tag} collision geometry has no exact primitive in the "
                "native sphere/capsule/plane profile and is not approximated.",
                changes_interpretation=True,
                affected_capability_ids=(_COLLISION_CAPABILITY_ID,),
            )
        )
        return None
    sphere_path = f"{geometry_path}/sphere"
    _attribute_losses(shape, frozenset(("radius",)), sphere_path, losses)
    _unknown_child_losses(shape, frozenset(), sphere_path, losses)
    _check_leaf_text(shape, sphere_path)
    radius = _float(
        _required_attribute(shape, "radius", sphere_path), f"{sphere_path}/@radius"
    )
    if radius <= 0.0:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{sphere_path}/@radius must be positive.")
    return _CollisionRecord(index, element.attrib.get("name"), xyz, rpy, radius)


def _parse_link(
    element: ET.Element, losses: _LossAccumulator, collision_requested: bool, /
) -> _LinkRecord:
    name = _required_attribute(element, "name", "/robot/link")
    path = f"/robot/links/{_pointer(name)}"
    _check_leaf_text(element, path)
    _attribute_losses(element, frozenset(("name",)), path, losses)
    _unknown_child_losses(
        element, frozenset(("inertial", "visual", "collision")), path, losses
    )
    inertial = _single_child(element, "inertial", path, required=True)
    assert inertial is not None
    inertial_path = f"{path}/inertial"
    _check_leaf_text(inertial, inertial_path)
    _attribute_losses(inertial, frozenset(), inertial_path, losses)
    _unknown_child_losses(
        inertial, frozenset(("origin", "mass", "inertia")), inertial_path, losses
    )
    inertial_xyz, inertial_rpy = _parse_origin(inertial, inertial_path, losses)
    mass_element = _single_child(inertial, "mass", inertial_path, required=True)
    inertia_element = _single_child(inertial, "inertia", inertial_path, required=True)
    assert mass_element is not None and inertia_element is not None
    _attribute_losses(
        mass_element, frozenset(("value",)), f"{inertial_path}/mass", losses
    )
    _unknown_child_losses(mass_element, frozenset(), f"{inertial_path}/mass", losses)
    _check_leaf_text(mass_element, f"{inertial_path}/mass")
    mass = _float(
        _required_attribute(mass_element, "value", f"{inertial_path}/mass"),
        f"{inertial_path}/mass/@value",
    )
    if mass <= 0.0:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{inertial_path}/mass must be positive.")
    tensor_names = ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
    _attribute_losses(
        inertia_element, frozenset(tensor_names), f"{inertial_path}/inertia", losses
    )
    _unknown_child_losses(
        inertia_element, frozenset(), f"{inertial_path}/inertia", losses
    )
    _check_leaf_text(inertia_element, f"{inertial_path}/inertia")
    tensor_values = {
        field: _float(
            _required_attribute(inertia_element, field, f"{inertial_path}/inertia"),
            f"{inertial_path}/inertia/@{field}",
        )
        for field in tensor_names
    }
    inertia_inertial = np.asarray(
        (
            (tensor_values["ixx"], tensor_values["ixy"], tensor_values["ixz"]),
            (tensor_values["ixy"], tensor_values["iyy"], tensor_values["iyz"]),
            (tensor_values["ixz"], tensor_values["iyz"], tensor_values["izz"]),
        ),
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(inertia_inertial)
    scale = float(np.max(np.abs(inertia_inertial)))
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if np.any(eigenvalues <= 0.0):
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"{inertial_path}/inertia must be symmetric positive definite.",
        )
    if eigenvalues[-1] > eigenvalues[0] + eigenvalues[1] + tolerance:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"{inertial_path}/inertia violates the physical principal-moment triangle inequality.",
        )
    inertial_rotation = _rotation_from_rpy(inertial_rpy)
    inertia_body = inertial_rotation @ inertia_inertial @ inertial_rotation.T
    for index, visual in enumerate(_children(element, "visual")):
        del visual
        losses.append(
            _loss(
                f"{path}/visual/{index}",
                "Visual geometry and material are intentionally not lowered into the native dynamics plans.",
                changes_interpretation=False,
                category="dropped",
                affected_capability_ids=(_VISUAL_CAPABILITY_ID,),
            )
        )
    collisions = tuple(
        _parse_collision(
            collision,
            f"{path}/collision/{index}",
            index,
            losses,
            collision_requested,
        )
        for index, collision in enumerate(_children(element, "collision"))
    )
    return _LinkRecord(
        name,
        mass,
        inertial_xyz,
        inertial_rpy,
        inertia_inertial,
        inertia_body,
        tuple(record for record in collisions if record is not None),
    )


def _parse_limit(
    joint: ET.Element,
    path: str,
    kind: str,
    losses: _LossAccumulator,
    /,
) -> tuple[float | None, float | None, float | None, float | None]:
    limit = _single_child(joint, "limit", path, required=kind != "fixed")
    if kind == "fixed":
        if limit is not None:
            losses.append(
                _loss(
                    f"{path}/limit",
                    "A fixed-joint limit has no native interpretation and is not lowered.",
                    changes_interpretation=True,
                    affected_capability_ids=(
                        _URDF_CAPABILITY_IDS["robot.joint-limit-evidence"],
                    ),
                )
            )
        return None, None, None, None
    assert limit is not None
    limit_path = f"{path}/limit"
    allowed = frozenset(("lower", "upper", "effort", "velocity"))
    _attribute_losses(limit, allowed, limit_path, losses)
    _unknown_child_losses(limit, frozenset(), limit_path, losses)
    _check_leaf_text(limit, limit_path)
    effort = _float(
        _required_attribute(limit, "effort", limit_path), f"{limit_path}/@effort"
    )
    velocity = _float(
        _required_attribute(limit, "velocity", limit_path), f"{limit_path}/@velocity"
    )
    if effort < 0.0 or velocity < 0.0:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"{limit_path} effort and velocity limits must be nonnegative.",
        )
    if kind == "continuous":
        for bound in ("lower", "upper"):
            if bound in limit.attrib:
                _float(limit.attrib[bound], f"{limit_path}/@{bound}")
                losses.append(
                    _loss(
                        f"{limit_path}/@{bound}",
                        "Continuous-joint position bounds contradict the unbounded "
                        "native coordinate and are not lowered.",
                        changes_interpretation=True,
                        affected_capability_ids=(
                            _URDF_CAPABILITY_IDS["robot.joint-limit-evidence"],
                        ),
                    )
                )
        return None, None, effort, velocity
    lower = _float(
        _required_attribute(limit, "lower", limit_path), f"{limit_path}/@lower"
    )
    upper = _float(
        _required_attribute(limit, "upper", limit_path), f"{limit_path}/@upper"
    )
    if lower >= upper:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{limit_path} lower must be below upper.")
    return lower, upper, effort, velocity


def _parse_damping(
    joint: ET.Element,
    path: str,
    kind: str,
    losses: _LossAccumulator,
    /,
) -> float | None:
    dynamics = _single_child(joint, "dynamics", path, required=False)
    if dynamics is None:
        return None
    dynamics_path = f"{path}/dynamics"
    if kind == "fixed":
        losses.append(
            _loss(
                dynamics_path,
                "Fixed-joint dynamics has no free coordinate and is not lowered.",
                changes_interpretation=True,
                affected_capability_ids=(
                    _URDF_CAPABILITY_IDS["robot.executed-joint-damping"],
                ),
            )
        )
        return None
    _attribute_losses(dynamics, frozenset(("damping", "friction")), dynamics_path, losses)
    _unknown_child_losses(dynamics, frozenset(), dynamics_path, losses)
    _check_leaf_text(dynamics, dynamics_path)
    damping = None
    if "damping" in dynamics.attrib:
        damping = _float(dynamics.attrib["damping"], f"{dynamics_path}/@damping")
        if damping < 0.0:
            _fail(
                AdapterStatus.MALFORMED_SOURCE,
                f"{dynamics_path}/@damping must be nonnegative.",
            )
    if "friction" in dynamics.attrib:
        friction = _float(dynamics.attrib["friction"], f"{dynamics_path}/@friction")
        if friction < 0.0:
            _fail(
                AdapterStatus.MALFORMED_SOURCE,
                f"{dynamics_path}/@friction must be nonnegative.",
            )
        losses.append(
            _loss(
                f"{dynamics_path}/@friction",
                "URDF joint friction is not represented by the native articulation target.",
                changes_interpretation=True,
                category="dropped",
                affected_capability_ids=(_FRICTION_CAPABILITY_ID,),
            )
        )
    return damping


def _parse_joint(element: ET.Element, losses: _LossAccumulator, /) -> _JointRecord:
    name = _required_attribute(element, "name", "/robot/joint")
    path = f"/robot/joints/{_pointer(name)}"
    _check_leaf_text(element, path)
    kind = _required_attribute(element, "type", path)
    if kind not in _SUPPORTED_JOINT_KINDS:
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            f"{path} has unsupported joint type {kind!r}.",
        )
    _attribute_losses(element, frozenset(("name", "type")), path, losses)
    _unknown_child_losses(
        element,
        frozenset(("origin", "parent", "child", "axis", "limit", "dynamics")),
        path,
        losses,
    )
    parent_element = _single_child(element, "parent", path, required=True)
    child_element = _single_child(element, "child", path, required=True)
    assert parent_element is not None and child_element is not None
    _attribute_losses(parent_element, frozenset(("link",)), f"{path}/parent", losses)
    _attribute_losses(child_element, frozenset(("link",)), f"{path}/child", losses)
    _unknown_child_losses(parent_element, frozenset(), f"{path}/parent", losses)
    _unknown_child_losses(child_element, frozenset(), f"{path}/child", losses)
    _check_leaf_text(parent_element, f"{path}/parent")
    _check_leaf_text(child_element, f"{path}/child")
    parent = _required_attribute(parent_element, "link", f"{path}/parent")
    child = _required_attribute(child_element, "link", f"{path}/child")
    if parent == child:
        _fail(AdapterStatus.INCONSISTENT_SOURCE, f"{path} cannot join a link to itself.")
    origin_xyz, origin_rpy = _parse_origin(element, path, losses)
    axis_element = _single_child(element, "axis", path, required=False)
    if kind == "fixed":
        if axis_element is not None:
            losses.append(
                _loss(
                    f"{path}/axis",
                    "A fixed-joint axis has no native interpretation and is not lowered.",
                    changes_interpretation=True,
                    affected_capability_ids=(
                        _URDF_CAPABILITY_IDS["robot.fixed-hinge-prismatic-kinematics"],
                    ),
                )
            )
        axis = np.asarray((1.0, 0.0, 0.0))
    else:
        if axis_element is None:
            axis = np.asarray((1.0, 0.0, 0.0))
        else:
            axis_path = f"{path}/axis"
            _attribute_losses(axis_element, frozenset(("xyz",)), axis_path, losses)
            _unknown_child_losses(axis_element, frozenset(), axis_path, losses)
            _check_leaf_text(axis_element, axis_path)
            axis = _vector(
                axis_element.attrib.get("xyz", "1 0 0"), 3, f"{axis_path}/@xyz"
            )
            norm = float(np.linalg.norm(axis))
            if norm <= np.finfo(np.float64).eps:
                _fail(AdapterStatus.MALFORMED_SOURCE, f"{axis_path} must be nonzero.")
            axis = axis / norm
    lower, upper, effort, velocity = _parse_limit(element, path, kind, losses)
    damping = _parse_damping(element, path, kind, losses)
    return _JointRecord(
        name,
        kind,
        parent,
        child,
        origin_xyz,
        origin_rpy,
        axis,
        lower,
        upper,
        effort,
        velocity,
        damping,
    )


def _parse_tree(
    root: ET.Element, max_losses: int, collision_requested: bool, /
) -> _ParsedURDF:
    if _tag_name(root) != "robot":
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF 1.0 input must have an unnamespaced <robot> document element.",
        )
    robot_name = _required_attribute(root, "name", "/robot")
    _check_leaf_text(root, "/robot")
    losses = _LossAccumulator(max_losses)
    _attribute_losses(root, frozenset(("name", "version")), "/robot", losses)
    if "version" in root.attrib and root.attrib["version"] != "1.0":
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Only the URDF 1.0 format version is supported.",
        )
    _unknown_child_losses(
        root, frozenset(("link", "joint", "material", "transmission")), "/robot", losses
    )
    link_elements = _children(root, "link")
    joint_elements = _children(root, "joint")
    if not link_elements:
        _fail(
            AdapterStatus.MALFORMED_SOURCE, "URDF robot must contain at least one link."
        )
    links = tuple(
        _parse_link(element, losses, collision_requested) for element in link_elements
    )
    joints = tuple(_parse_joint(element, losses) for element in joint_elements)
    link_names = tuple(record.name for record in links)
    joint_names = tuple(record.name for record in joints)
    if len(set(link_names)) != len(link_names):
        _fail(AdapterStatus.INCONSISTENT_SOURCE, "URDF link names must be unique.")
    if len(set(joint_names)) != len(joint_names):
        _fail(AdapterStatus.INCONSISTENT_SOURCE, "URDF joint names must be unique.")
    known_links = frozenset(link_names)
    incoming: dict[str, _JointRecord] = {}
    adjacency: dict[str, list[_JointRecord]] = {name: [] for name in link_names}
    for joint in joints:
        if joint.parent not in known_links or joint.child not in known_links:
            _fail(
                AdapterStatus.INCONSISTENT_SOURCE,
                f"Joint {joint.name!r} references a missing parent or child link.",
            )
        if joint.child in incoming:
            _fail(
                AdapterStatus.INCONSISTENT_SOURCE,
                f"Link {joint.child!r} has more than one parent joint.",
            )
        incoming[joint.child] = joint
        adjacency[joint.parent].append(joint)
    resolved: set[str] = set()
    for name in link_names:
        chain: set[str] = set()
        current = name
        while current in incoming and current not in resolved:
            if current in chain:
                _fail(
                    AdapterStatus.INCONSISTENT_SOURCE,
                    "URDF joint graph must be acyclic.",
                )
            chain.add(current)
            current = incoming[current].parent
        resolved.update(chain)
    roots = tuple(sorted(known_links - frozenset(incoming)))
    if len(roots) != 1:
        _fail(
            AdapterStatus.INCONSISTENT_SOURCE,
            "URDF joint graph must contain exactly one root link.",
        )
    root_name = roots[0]
    traversal: list[_JointRecord] = []
    visited = {root_name}
    queue = deque((root_name,))
    while queue:
        parent = queue.popleft()
        outgoing = sorted(adjacency[parent], key=lambda item: (item.child, item.name))
        for joint in outgoing:
            if joint.child in visited:
                _fail(
                    AdapterStatus.INCONSISTENT_SOURCE, "URDF joint graph must be a tree."
                )
            visited.add(joint.child)
            traversal.append(joint)
            queue.append(joint.child)
    if visited != known_links or len(joints) != len(links) - 1:
        _fail(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Every URDF link must belong to one connected tree.",
        )
    for index, material in enumerate(_children(root, "material")):
        del material
        losses.append(
            _loss(
                f"/robot/material/{index}",
                "Robot-level visual material declarations are not lowered into dynamics plans.",
                changes_interpretation=False,
                category="dropped",
                affected_capability_ids=(_VISUAL_CAPABILITY_ID,),
            )
        )
    for index, transmission in enumerate(_children(root, "transmission")):
        del transmission
        losses.append(
            _loss(
                f"/robot/transmission/{index}",
                "URDF transmissions and actuator mappings are not represented by the native articulation target.",
                changes_interpretation=True,
                category="dropped",
                affected_capability_ids=(_ACTUATION_CAPABILITY_ID,),
            )
        )
    return _ParsedURDF(
        robot_name,
        tuple(sorted(links, key=lambda item: item.name)),
        tuple(sorted(joints, key=lambda item: item.name)),
        root_name,
        tuple(traversal),
        tuple(sorted(losses.items, key=lambda item: (item.path, item.loss_id))),
    )


def _validated_xml(resource: BoundedResource, /) -> tuple[ET.Element, str, int, int, int]:
    try:
        text = resource.data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE,
            "URDF sources must be strict UTF-8.",
        ) from error
    if _UNSAFE_DECLARATION.search(text) is not None:
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF DTD, entity, notation, and CDATA declarations are disabled.",
        )
    if _PROCESSING_INSTRUCTION.search(text) is not None:
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF processing instructions are disabled.",
        )
    stripped = text.lstrip("\ufeff \t\r\n")
    if stripped.startswith("<?xml"):
        declaration_end = stripped.find("?>")
        if (
            declaration_end < 0
            or _XML_DECLARATION.fullmatch(stripped[: declaration_end + 2]) is None
        ):
            _fail(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Only a standard UTF-8 XML 1.0 declaration is accepted.",
            )
    limits = resource.manifest.limits
    parser = ET.XMLPullParser(events=("start", "end"))
    root: ET.Element | None = None
    current_depth = 0
    observed_depth = 0
    nodes = 0
    attributes = 0

    def consume_events() -> None:
        nonlocal root, current_depth, observed_depth, nodes, attributes
        events = cast(Iterator[tuple[str, ET.Element]], parser.read_events())
        for event, element in events:
            if event == "start":
                if root is None:
                    root = element
                current_depth += 1
                observed_depth = max(observed_depth, current_depth)
                nodes += 1
                attributes += len(element.attrib)
                if observed_depth > limits.max_depth:
                    _fail(
                        AdapterStatus.MALFORMED_SOURCE,
                        f"URDF XML exceeds the configured {limits.max_depth}-level depth limit.",
                    )
                if nodes > limits.max_nodes:
                    _fail(
                        AdapterStatus.MALFORMED_SOURCE,
                        f"URDF XML exceeds the configured {limits.max_nodes}-node limit.",
                    )
                if attributes > limits.max_attributes:
                    _fail(
                        AdapterStatus.MALFORMED_SOURCE,
                        f"URDF XML exceeds the configured {limits.max_attributes}-attribute limit.",
                    )
                _reject_external_execution(element)
            else:
                current_depth -= 1

    try:
        for offset in range(0, len(text), 64 * 1024):
            parser.feed(text[offset : offset + 64 * 1024])
            consume_events()
        parser.close()
        consume_events()
    except ET.ParseError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE, f"Malformed URDF XML: {error}."
        ) from error
    if root is None or current_depth != 0:
        _fail(AdapterStatus.MALFORMED_SOURCE, "URDF XML must contain one complete root.")
    return root, text, observed_depth, nodes, attributes


def _reject_external_execution(element: ET.Element, /) -> None:
    tag = _tag_name(element)
    local_tag = tag.rsplit("}", 1)[-1].lower()
    if local_tag in ("include", "plugin"):
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF include and plugin elements are disabled.",
        )
    if any(
        "://" in value or value.startswith(("//", "\\\\"))
        for value in element.attrib.values()
    ):
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF network and external resource references are disabled.",
        )


def _resource_limits(
    max_bytes: int,
    max_depth: int,
    max_nodes: int,
    max_attributes: int,
    max_losses: int,
    /,
) -> ResourceLimits:
    return ResourceLimits(
        max_bytes,
        max_depth,
        max_nodes,
        max_attributes,
        max_losses,
    )


def _resource_failure(error: ResourceReadError, /) -> Never:
    status = (
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC
        if error.reason == "policy"
        else AdapterStatus.INCONSISTENT_SOURCE
        if error.reason == "inconsistent"
        else AdapterStatus.MALFORMED_SOURCE
    )
    raise URDFImportError(status, str(error)) from error


def _root_policy(value: str, /) -> _URDFRootPolicy:
    value_ = str(value).strip()
    if value_ == "fixed_world":
        return "fixed_world"
    if value_ == "reject_unpinned":
        return "reject_unpinned"
    raise ValueError("root_policy must be 'fixed_world' or 'reject_unpinned'.")


def _world_link_frames(
    parsed: _ParsedURDF,
    /,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {
        parsed.root_link: (np.zeros((3,), dtype=np.float64), np.eye(3, dtype=np.float64))
    }
    for joint in parsed.traversal:
        parent_position, parent_rotation = frames[joint.parent]
        child_position = parent_position + parent_rotation @ joint.origin_xyz
        child_rotation = parent_rotation @ _rotation_from_rpy(joint.origin_rpy)
        frames[joint.child] = (child_position, child_rotation)
    return frames


def _link_evidence_payload(item: URDFLinkEvidence, /) -> dict[str, object]:
    return {
        "name": item.name,
        "body_id": item.body_id,
        "mass_kg": item.mass_kg,
        "com_in_link_frame_m": list(item.com_in_link_frame_m),
        "link_frame_in_body_m": list(item.link_frame_in_body_m),
        "inertial_frame_rpy_rad": list(item.inertial_frame_rpy_rad),
        "inertia_in_inertial_frame_kg_m2": list(item.inertia_in_inertial_frame_kg_m2),
        "inertia_in_body_frame_kg_m2": list(item.inertia_in_body_frame_kg_m2),
        "reference_link_position_m": list(item.reference_link_position_m),
        "reference_body_position_m": list(item.reference_body_position_m),
        "reference_orientation_wxyz": list(item.reference_orientation_wxyz),
    }


def _joint_evidence_payload(item: URDFJointEvidence, /) -> dict[str, object]:
    return {
        "name": item.name,
        "joint_id": item.joint_id,
        "kind": item.kind,
        "parent_link": item.parent_link,
        "parent_body_id": item.parent_body_id,
        "child_link": item.child_link,
        "child_body_id": item.child_body_id,
        "origin_xyz_m": list(item.origin_xyz_m),
        "origin_rpy_rad": list(item.origin_rpy_rad),
        "reference_anchor_m": list(item.reference_anchor_m),
        "axis_in_joint_frame": list(item.axis_in_joint_frame),
        "reference_axis": list(item.reference_axis),
        "lower_limit": item.lower_limit,
        "upper_limit": item.upper_limit,
        "effort_limit": item.effort_limit,
        "velocity_limit": item.velocity_limit,
        "damping": item.damping,
    }


def _bound_payload(value: float, /) -> float | str:
    # Canonical JSON admits only finite numbers; an absent bound is the explicit
    # token for the corresponding infinite side, never a NaN/Infinity literal.
    if isfinite(value):
        return value
    return "unbounded-above" if value > 0.0 else "unbounded-below"


def _motor_evidence_payload(item: URDFJointMotor, /) -> dict[str, object]:
    return {
        "name": item.name,
        "joint": item.joint,
        "gear": item.gear,
        "command_lower": _bound_payload(item.command_lower),
        "command_upper": _bound_payload(item.command_upper),
    }


def _collision_evidence_payload(item: URDFCollisionEvidence, /) -> dict[str, object]:
    return {
        "primitive_name": item.primitive_name,
        "primitive_id": item.primitive_id,
        "link": item.link,
        "source_index": item.source_index,
        "source_name": item.source_name,
        "radius_m": item.radius_m,
        "center_in_link_frame_m": list(item.center_in_link_frame_m),
        "origin_rpy_rad": list(item.origin_rpy_rad),
        "center_in_body_frame_m": list(item.center_in_body_frame_m),
        "material_id": item.material_id,
    }


def _tuple3(value: np.ndarray, /) -> tuple[float, float, float]:
    return float(value[0]), float(value[1]), float(value[2])


def _tuple4(value: np.ndarray, /) -> tuple[float, float, float, float]:
    return float(value[0]), float(value[1]), float(value[2]), float(value[3])


def _waivers_for_losses(
    losses: tuple[AdapterLoss, ...],
    waivers: Sequence[AdapterWaiver],
    waived_loss_paths: Sequence[str],
    /,
) -> tuple[AdapterWaiver, ...]:
    supplied = tuple(waivers)
    if not all(isinstance(item, AdapterWaiver) for item in supplied):
        raise TypeError("waivers must contain AdapterWaiver values.")
    paths = tuple(str(path) for path in waived_loss_paths)
    if any(not path for path in paths) or len(set(paths)) != len(paths):
        raise ValueError("waived_loss_paths must contain unique non-empty paths.")
    by_path = {loss.path: loss for loss in losses}
    generated = tuple(
        AdapterWaiver(
            by_path[path],
            f"Caller explicitly accepted the interpretation change at {path}.",
        )
        for path in paths
        if path in by_path
    )
    unknown = tuple(path for path in paths if path not in by_path)
    if unknown:
        _fail(
            AdapterStatus.INCONSISTENT_SOURCE,
            f"URDF loss waiver paths do not name declared losses: {unknown!r}.",
        )
    combined = supplied + generated
    if len({item.loss_id for item in combined}) != len(combined):
        raise ValueError("Each URDF loss may be waived at most once.")
    return combined


_NATIVE_JOINT_KINDS: dict[str, NativeTreeJointKind] = {
    "fixed": "fixed",
    "revolute": "hinge",
    "continuous": "hinge",
    "prismatic": "prismatic",
}


def _motor_components(
    motors: Sequence[URDFJointMotor],
    parsed: _ParsedURDF,
    joint_ids: RobotNameIDMap,
    /,
) -> tuple[RobotNameIDMap | None, ScalarJointMotorPlan | None, tuple[URDFJointMotor, ...]]:
    supplied = tuple(motors)
    if not all(isinstance(motor, URDFJointMotor) for motor in supplied):
        raise TypeError("motors must contain URDFJointMotor values.")
    if not supplied:
        return None, None, ()
    ordered = tuple(sorted(supplied, key=lambda motor: motor.name))
    names = RobotNameIDMap(tuple(motor.name for motor in ordered))
    by_name = {joint.name: joint for joint in parsed.joints}
    efforts: list[float] = []
    for motor in ordered:
        joint = by_name.get(motor.joint)
        if joint is None or joint.kind == "fixed" or joint.effort is None:
            raise ValueError(
                f"URDF motor {motor.name!r} must map to a revolute, continuous, or "
                "prismatic joint."
            )
        efforts.append(joint.effort)
    plan = ScalarJointMotorPlan(
        np.asarray([joint_ids.id_for_name(motor.joint) for motor in ordered], np.int64),
        np.asarray([motor.gear for motor in ordered], dtype=np.float64),
        effort_limit=np.asarray(efforts, dtype=np.float64),
        command_lower=np.asarray(
            [motor.command_lower for motor in ordered], dtype=np.float64
        ),
        command_upper=np.asarray(
            [motor.command_upper for motor in ordered], dtype=np.float64
        ),
    )
    return names, plan, ordered


def _damping_plan(
    parsed: _ParsedURDF, joint_ids: RobotNameIDMap, /
) -> ScalarJointDampingPlan | None:
    damped = tuple(
        joint
        for joint in parsed.joints
        if joint.kind != "fixed" and joint.damping is not None
    )
    if not damped:
        return None
    return ScalarJointDampingPlan(
        np.asarray([joint_ids.id_for_name(joint.name) for joint in damped], np.int64),
        np.asarray([joint.damping for joint in damped], dtype=np.float64),
    )


def _stop_plan(
    parsed: _ParsedURDF,
    joint_ids: RobotNameIDMap,
    policy: URDFJointStopPolicy,
    /,
) -> ReducedJointStopPlan | None:
    match policy:
        case "evidence-only":
            return None
        case "native-unilateral-stops":
            limited = tuple(
                joint
                for joint in parsed.joints
                if joint.lower is not None and joint.upper is not None
            )
            if not limited:
                return None
            return ReducedJointStopPlan(
                np.asarray(
                    [joint_ids.id_for_name(joint.name) for joint in limited], np.int64
                ),
                np.asarray([joint.lower for joint in limited], dtype=np.float64),
                np.asarray([joint.upper for joint in limited], dtype=np.float64),
            )
        case unknown:
            assert_never(unknown)


def _collision_components(
    policy: URDFCollisionPolicy | None,
    parsed: _ParsedURDF,
    link_ids: RobotNameIDMap,
    /,
) -> tuple[RobotCollisionGeometryPlan | None, tuple[URDFCollisionEvidence, ...]]:
    if policy is None:
        return None, ()
    known = frozenset(link_ids.names)
    materials = dict(policy.link_materials)
    referenced = set(materials) | {
        link for pair in policy.excluded_link_pairs for link in pair
    }
    unknown = sorted(referenced - known)
    if unknown:
        raise ValueError(f"URDF collision policy names unknown links {unknown!r}.")
    records = sorted(
        (
            (f"{link.name}/collision/{collision.index}", link, collision)
            for link in parsed.links
            for collision in link.collisions
        ),
        key=lambda item: item[0],
    )
    missing = sorted({link.name for _, link, _ in records} - set(materials))
    if missing:
        raise ValueError(f"URDF collision policy has no material for links {missing!r}.")
    if not records:
        # Unsupported shapes are already unwaivable losses; with no lowered
        # sphere there is no body-attached geometry to execute.
        return None, ()
    names = RobotNameIDMap(tuple(name for name, _, _ in records))
    centers = np.asarray(
        [collision.xyz - link.inertial_xyz for _, link, collision in records],
        dtype=np.float64,
    ).reshape((-1, 3))
    excluded = set(policy.excluded_link_pairs)
    if policy.exclude_jointed_links:
        excluded |= {
            (min(joint.parent, joint.child), max(joint.parent, joint.child))
            for joint in parsed.joints
        }
    pairs = sorted(excluded)
    planes = policy.planes
    plan = RobotCollisionGeometryPlan(
        names,
        tuple(ArticulatedPrimitiveKind.SPHERE for _ in records),
        np.asarray([link_ids.id_for_name(link.name) for _, link, _ in records], np.int64),
        centers,
        centers,
        np.asarray([collision.radius for _, _, collision in records], np.float64),
        np.asarray([materials[link.name] for _, link, _ in records], np.int64),
        materials=policy.materials,
        plane_names=RobotNameIDMap(tuple(plane.name for plane in planes)),
        plane_normals=np.asarray([plane.normal for plane in planes]).reshape((-1, 3)),
        plane_offsets=np.asarray([plane.offset for plane in planes], np.float64),
        plane_material_ids=np.asarray(
            [plane.material_id for plane in planes], np.int64
        ),
        excluded_body_pairs=np.asarray(
            [
                [link_ids.id_for_name(first), link_ids.id_for_name(second)]
                for first, second in pairs
            ],
            dtype=np.int64,
        ).reshape((-1, 2)),
    )
    evidence = tuple(
        URDFCollisionEvidence(
            name,
            int(names.id_for_name(name)),
            link.name,
            collision.index,
            collision.source_name,
            collision.radius,
            _tuple3(collision.xyz),
            _tuple3(collision.rpy),
            _tuple3(collision.xyz - link.inertial_xyz),
            materials[link.name],
        )
        for name, link, collision in records
    )
    return plan, evidence


def _requirements_and_capabilities(
    *, motors: bool, stops: bool, collision: bool
) -> tuple[tuple[AdapterRequirement, ...], tuple[AdapterCapability, ...]]:
    capabilities = list(_URDF_CAPABILITIES)
    if motors:
        capabilities.append(_ACTUATION_CAPABILITY)
    if stops:
        capabilities.append(_STOP_CAPABILITY)
    if collision:
        capabilities.append(_COLLISION_CAPABILITY)
    requirements = tuple(
        AdapterRequirement(capability.semantic_id) for capability in capabilities
    )
    return requirements, tuple(capabilities)


def _adapt(
    parsed: _ParsedURDF,
    source_id: str,
    resource: BoundedResource,
    root_policy: _URDFRootPolicy,
    joint_stop_policy: URDFJointStopPolicy,
    motors: Sequence[URDFJointMotor],
    collision_policy: URDFCollisionPolicy | None,
    waivers: Sequence[AdapterWaiver],
    waived_loss_paths: Sequence[str],
    /,
) -> RobotAdaptation[URDFFormatEvidence]:
    link_ids = RobotNameIDMap(tuple(link.name for link in parsed.links))
    joint_ids = RobotNameIDMap(tuple(joint.name for joint in parsed.joints))
    link_id = {name: int(identifier) for name, identifier in link_ids.name_to_id}
    joint_id = {name: int(identifier) for name, identifier in joint_ids.name_to_id}
    frames = _world_link_frames(parsed)
    count = len(parsed.links)
    masses = np.asarray([link.mass for link in parsed.links], dtype=np.float64)
    inertias = np.stack([link.inertia_body for link in parsed.links], axis=0)
    positions = np.empty((count, 3), dtype=np.float64)
    orientations = np.empty((count, 4), dtype=np.float64)
    link_evidence: list[URDFLinkEvidence] = []
    for index, link in enumerate(parsed.links):
        link_position, link_rotation = frames[link.name]
        body_position = link_position + link_rotation @ link.inertial_xyz
        quaternion = _quaternion_from_rotation(link_rotation)
        positions[index] = body_position
        orientations[index] = quaternion
        link_evidence.append(
            URDFLinkEvidence(
                link.name,
                link_id[link.name],
                link.mass,
                _tuple3(link.inertial_xyz),
                _tuple3(-link.inertial_xyz),
                _tuple3(link.inertial_rpy),
                tuple(float(value) for value in link.inertia_inertial.reshape((-1,))),
                tuple(float(value) for value in link.inertia_body.reshape((-1,))),
                _tuple3(link_position),
                _tuple3(body_position),
                _tuple4(quaternion),
            )
        )
    joint_evidence: list[URDFJointEvidence] = []
    for joint in parsed.joints:
        anchor, joint_rotation = frames[joint.child]
        axis = joint_rotation @ joint.axis
        joint_evidence.append(
            URDFJointEvidence(
                joint.name,
                joint_id[joint.name],
                joint.kind,
                joint.parent,
                link_id[joint.parent],
                joint.child,
                link_id[joint.child],
                _tuple3(joint.origin_xyz),
                _tuple3(joint.origin_rpy),
                _tuple3(anchor),
                _tuple3(joint.axis),
                _tuple3(axis),
                joint.lower,
                joint.upper,
                joint.effort,
                joint.velocity,
                joint.damping,
            )
        )
    tree_joints = tuple(
        NativeTreeJoint(
            joint.name,
            _NATIVE_JOINT_KINDS[joint.kind],
            joint.parent,
            joint.child,
            anchor=_tuple3(frames[joint.child][0]),
            axis=_tuple3(frames[joint.child][1] @ joint.axis),
        )
        for joint in parsed.traversal
    )
    # URDF body axes are the link axes, so the link frame sits at minus the
    # inertial origin in the COM-centered body frame.
    frame_plan = RobotFramePlan(
        link_ids,
        np.asarray(link_ids.ids, dtype=np.int64),
        np.stack([-link.inertial_xyz for link in parsed.links], axis=0),
        np.tile(np.asarray(_IDENTITY_ORIENTATION, dtype=np.float64), (count, 1)),
    )
    actuator_names, motor_plan, mapped_motors = _motor_components(
        motors, parsed, joint_ids
    )
    collision, collision_evidence = _collision_components(
        collision_policy, parsed, link_ids
    )
    model = lower_native_robot_tree(
        parsed.robot_name,
        link_ids,
        masses,
        inertias,
        positions,
        orientations,
        parsed.root_link,
        tree_joints,
        DimensionalScaleContract.si(),
        frame_plan,
        actuator_names=actuator_names,
        motors=motor_plan,
        damping=_damping_plan(parsed, joint_ids),
        stops=_stop_plan(parsed, joint_ids, joint_stop_policy),
        collision=collision,
    )
    target_id = model.model_id
    applied_waivers = _waivers_for_losses(parsed.losses, waivers, waived_loss_paths)
    evidence = URDFFormatEvidence(
        parsed.robot_name,
        resource,
        root_policy,
        joint_stop_policy,
        parsed.root_link,
        link_ids,
        joint_ids,
        link_evidence,
        joint_evidence,
        collision_evidence,
        mapped_motors,
        tuple(loss.path for loss in parsed.losses),
        source_id,
        target_id,
    )
    requirements, capabilities = _requirements_and_capabilities(
        motors=motor_plan is not None,
        stops=joint_stop_policy == "native-unilateral-stops",
        collision=collision_policy is not None,
    )
    negotiation = negotiate_adapter(
        requirements, capabilities, losses=parsed.losses, waivers=applied_waivers
    )
    target_qualifiers = {
        "base": "fixed",
        "body-frame": "center-of-mass",
        "coordinates": "SI",
        "joint-stops": joint_stop_policy,
    }
    if collision is not None:
        target_qualifiers["contact-law"] = collision.contact_law
    report = AdapterReport(
        negotiation.status,
        "URDF 1.0",
        "phydrax-native-robot-model",
        source_id=source_id,
        target_id=target_id,
        stage="host-import",
        source_profile=AdapterFormatProfile(
            "URDF 1.0",
            qualifiers={
                "execution": "host-only",
                "length": "m",
                "mass": "kg",
                "time": "s",
            },
        ),
        target_profile=AdapterFormatProfile(
            "phydrax-native-robot-model", qualifiers=target_qualifiers
        ),
        coordinate_mapping=(
            "URDF parent-link joint origin -> zero-configuration world anchor",
            "URDF joint-frame axis -> zero-configuration world reference axis",
            "URDF joint position -> native lifted coordinate (zero at the URDF origin)",
            "URDF link origin -> COM-centered body position with the link frame "
            "retained as a named native frame",
            "URDF inertial-frame tensor -> body-frame tensor by R_link_inertial I R_link_inertial^T",
            "URDF roll-pitch-yaw -> scalar-first unit quaternion",
            "URDF collision sphere origin -> COM body-frame center (origin - inertial origin)",
        ),
        preserved_fields=(
            "robot and link/joint names",
            "oriented connected tree topology",
            "link masses and COM positions",
            "link inertial-frame rotations and tensors",
            "link frames",
            "fixed, revolute, continuous, and prismatic joint kinematics",
            "joint position, effort, and velocity limits",
            "joint viscous damping (executed)",
            "zero-configuration reference kinematics",
        ),
        assumptions=(
            "URDF linear dimensions are meters",
            "URDF masses are kilograms",
            "URDF angles are radians",
            "URDF inertia components are kg*m^2 about the link COM",
            "URDF damping is linear viscous joint damping",
            "the caller explicitly selected fixed_world root attachment",
            f"the caller explicitly selected {joint_stop_policy} for position limits",
            "motors and contact materials are explicit caller declarations, not URDF semantics",
            "external macros, includes, resources, networks, and plugins are rejected",
        ),
        losses=parsed.losses,
        requirements=requirements,
        capabilities=capabilities,
        waivers=applied_waivers,
    )
    if not negotiation.valid:
        raise URDFImportError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF input contains an unwaived or unwaivable semantic loss or a stale waiver.",
            report=report,
            evidence=evidence,
        )
    if not report.valid:
        raise URDFImportError(
            report.status,
            "URDF adapter negotiation did not produce a valid conversion.",
            report=report,
            evidence=evidence,
        )
    return RobotAdaptation(model, report, evidence)


def _adapt_resource(
    resource: BoundedResource,
    root_policy: _URDFRootPolicy,
    joint_stop_policy: URDFJointStopPolicy,
    motors: Sequence[URDFJointMotor],
    collision: URDFCollisionPolicy | None,
    waivers: Sequence[AdapterWaiver],
    waived_loss_paths: Sequence[str],
    /,
) -> RobotAdaptation[URDFFormatEvidence]:
    if collision is not None and not isinstance(collision, URDFCollisionPolicy):
        raise TypeError("collision must be a URDFCollisionPolicy or None.")
    root, _, depth, nodes, attributes = _validated_xml(resource)
    parsed = _parse_tree(root, resource.manifest.limits.max_losses, collision is not None)
    try:
        accounted = account_bounded_resource(
            resource,
            depth=depth,
            nodes=nodes,
            attributes=attributes,
            losses=len(parsed.losses),
        )
    except ResourceReadError as error:
        _resource_failure(error)
    if root_policy == "reject_unpinned":
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF does not encode a world attachment for its root link; reject_unpinned forbids synthesizing one.",
        )
    source_id = canonical_fingerprint(
        {
            "kind": "urdf-source",
            "format": "URDF 1.0",
            "size_bytes": accounted.manifest.size_bytes,
            "content_sha256": accounted.manifest.content_sha256,
        }
    )
    return _adapt(
        parsed,
        source_id,
        accounted,
        root_policy,
        joint_stop_policy,
        motors,
        collision,
        waivers,
        waived_loss_paths,
    )


def parse_urdf_text(
    text: str,
    /,
    *,
    root_policy: _URDFRootPolicy,
    joint_stops: URDFJointStopPolicy = "evidence-only",
    motors: Sequence[URDFJointMotor] = (),
    collision: URDFCollisionPolicy | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_nodes: int = _DEFAULT_MAX_NODES,
    max_attributes: int = _DEFAULT_MAX_ATTRIBUTES,
    max_losses: int = _DEFAULT_MAX_LOSSES,
    waivers: Sequence[AdapterWaiver] = (),
    waived_loss_paths: Sequence[str] = (),
) -> RobotAdaptation[URDFFormatEvidence]:
    """Lower bounded URDF text into a native robot model.

    ``root_policy`` is the required root interpretation. URDF damping is
    executed; position limits become native unilateral stops only under
    ``joint_stops="native-unilateral-stops"``; motors exist only for explicit
    ``motors`` mappings; sphere collisions are lowered only under an explicit
    ``collision`` policy.
    """

    policy = _root_policy(root_policy)
    stop_policy = parse(joint_stops, URDFJointStopPolicy, "joint_stops")
    if not isinstance(text, str):
        raise TypeError("URDF text must be a string.")
    try:
        data = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE,
            "URDF text must be valid Unicode encodable as UTF-8.",
        ) from error
    limits = _resource_limits(max_bytes, max_depth, max_nodes, max_attributes, max_losses)
    try:
        resource = bounded_resource_from_bytes(data, limits=limits)
    except ResourceReadError as error:
        _resource_failure(error)
    return _adapt_resource(
        resource, policy, stop_policy, motors, collision, waivers, waived_loss_paths
    )


def parse_urdf_file(
    path: str | Path,
    /,
    *,
    allowed_root: str | Path,
    root_policy: _URDFRootPolicy,
    joint_stops: URDFJointStopPolicy = "evidence-only",
    motors: Sequence[URDFJointMotor] = (),
    collision: URDFCollisionPolicy | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    max_depth: int = _DEFAULT_MAX_DEPTH,
    max_nodes: int = _DEFAULT_MAX_NODES,
    max_attributes: int = _DEFAULT_MAX_ATTRIBUTES,
    max_losses: int = _DEFAULT_MAX_LOSSES,
    waivers: Sequence[AdapterWaiver] = (),
    waived_loss_paths: Sequence[str] = (),
) -> RobotAdaptation[URDFFormatEvidence]:
    """Descriptor-read one bounded UTF-8 URDF and lower it like `parse_urdf_text`."""

    policy = _root_policy(root_policy)
    stop_policy = parse(joint_stops, URDFJointStopPolicy, "joint_stops")
    limits = _resource_limits(max_bytes, max_depth, max_nodes, max_attributes, max_losses)
    try:
        resource = read_bounded_resource(
            path,
            trusted_root=allowed_root,
            limits=limits,
        )
    except ResourceReadError as error:
        _resource_failure(error)
    return _adapt_resource(
        resource, policy, stop_policy, motors, collision, waivers, waived_loss_paths
    )


__all__ = [
    "URDFCollisionEvidence",
    "URDFCollisionPolicy",
    "URDFFormatEvidence",
    "URDFImportError",
    "URDFJointEvidence",
    "URDFJointMotor",
    "URDFJointStopPolicy",
    "URDFLinkEvidence",
    "URDFStaticPlane",
    "parse_urdf_file",
    "parse_urdf_text",
]
