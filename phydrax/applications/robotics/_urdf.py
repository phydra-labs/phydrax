#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Secure host-only adaptation of the supported URDF 1.0 robot subset."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from math import cos, isfinite, sin
from pathlib import Path
from typing import Never

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import DimensionalScaleContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.particle._core import ParticleSetPlan
from ...discretization.particle._reduced_articulation import ReducedArticulationPlan
from ...discretization.particle._rigid_body import (
    RigidBodyKinematics,
    RigidBodySetPlan,
)
from ...discretization.particle._rigid_joints import (
    FixedJointSetPlan,
    HingeJointSetPlan,
    PrismaticJointSetPlan,
    RigidJointGraphPlan,
)
from ...interchange._report import (
    AdapterCapability,
    AdapterError,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterNegotiationResult,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
    AdapterWaiver,
    negotiate_adapter,
)


_DEFAULT_MAX_BYTES = 4 * 1024 * 1024
_FLOAT_TOKEN = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", re.ASCII
)
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


class RobotNameIDMap(StrictModule, NonTrainableState):
    """Deterministic bijection between sorted URDF names and signed 64-bit IDs."""

    names: tuple[str, ...] = eqx.field(static=True)
    ids: tuple[int, ...] = eqx.field(static=True)
    mapping_id: str = eqx.field(static=True)

    def __init__(self, names: Sequence[str], /):
        names_ = tuple(str(name) for name in names)
        if (
            any(not name for name in names_)
            or tuple(sorted(names_)) != names_
            or len(set(names_)) != len(names_)
        ):
            raise ValueError("Robot name maps require unique, non-empty, sorted names.")
        ids_host = np.arange(len(names_), dtype=np.int64)
        self.names = names_
        self.ids = tuple(int(identifier) for identifier in ids_host)
        self.mapping_id = canonical_fingerprint(
            {
                "kind": "robot-name-id-map",
                "entries": [[name, int(identifier)] for name, identifier in zip(names_, ids_host)],
                "integer_dtype": "int64",
            }
        )

    @property
    def name_to_id(self) -> tuple[tuple[str, np.int64], ...]:
        return tuple(
            (name, np.int64(self.ids[index])) for index, name in enumerate(self.names)
        )

    @property
    def id_to_name(self) -> tuple[tuple[np.int64, str], ...]:
        return tuple((identifier, name) for name, identifier in self.name_to_id)

    def id_for_name(self, name: str, /) -> np.int64:
        name_ = str(name)
        if name_ not in self.names:
            raise KeyError(name_)
        return np.int64(np.asarray(self.ids, dtype=np.int64)[self.names.index(name_)])

    def name_for_id(self, identifier: int, /) -> str:
        identifier_ = int(identifier)
        if identifier_ < 0 or identifier_ >= len(self.names):
            raise KeyError(identifier_)
        return self.names[identifier_]


class URDFLinkEvidence(StrictModule, NonTrainableState):
    """Exact link-frame information retained beside COM-centred native bodies."""

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
    reference_orientation_wxyz: tuple[float, float, float, float] = eqx.field(
        static=True
    )


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


class URDFFormatEvidence(StrictModule, NonTrainableState):
    """Auditable format, security, topology, mapping, and frame evidence."""

    robot_name: str = eqx.field(static=True)
    format_name: str = eqx.field(static=True)
    format_version: str = eqx.field(static=True)
    source_path: str | None = eqx.field(static=True)
    source_size_bytes: int = eqx.field(static=True)
    root_link: str = eqx.field(static=True)
    link_count: int = eqx.field(static=True)
    joint_count: int = eqx.field(static=True)
    link_ids: RobotNameIDMap
    joint_ids: RobotNameIDMap
    links: tuple[URDFLinkEvidence, ...]
    joints: tuple[URDFJointEvidence, ...]
    execution_policy: tuple[str, ...] = eqx.field(static=True)
    loss_paths: tuple[str, ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        robot_name: str,
        source_path: str | None,
        source_size_bytes: int,
        root_link: str,
        link_ids: RobotNameIDMap,
        joint_ids: RobotNameIDMap,
        links: Sequence[URDFLinkEvidence],
        joints: Sequence[URDFJointEvidence],
        loss_paths: Sequence[str],
        source_id: str,
        target_id: str,
        /,
    ):
        links_ = tuple(links)
        joints_ = tuple(joints)
        policies = (
            "host-only parsing",
            "network access disabled",
            "DTD and entity declarations disabled",
            "xacro and include expansion disabled",
            "plugin execution disabled",
        )
        paths = tuple(loss_paths)
        self.robot_name = robot_name
        self.format_name = "URDF"
        self.format_version = "1.0"
        self.source_path = source_path
        self.source_size_bytes = int(source_size_bytes)
        self.root_link = root_link
        self.link_count = len(links_)
        self.joint_count = len(joints_)
        self.link_ids = link_ids
        self.joint_ids = joint_ids
        self.links = links_
        self.joints = joints_
        self.execution_policy = policies
        self.loss_paths = paths
        self.source_id = source_id
        self.target_id = target_id
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "urdf-adapter-evidence",
                "format": "URDF 1.0",
                "robot_name": robot_name,
                "source_path": source_path,
                "source_size_bytes": int(source_size_bytes),
                "root_link": root_link,
                "link_mapping": link_ids.mapping_id,
                "joint_mapping": joint_ids.mapping_id,
                "links": [_link_evidence_payload(item) for item in links_],
                "joints": [_joint_evidence_payload(item) for item in joints_],
                "execution_policy": list(policies),
                "loss_paths": list(paths),
                "source_id": source_id,
                "target_id": target_id,
            }
        )


class RobotAdaptation(StrictModule, NonTrainableState):
    """Final native robot plans, reference kinematics, and adapter evidence."""

    particles: ParticleSetPlan
    bodies: RigidBodySetPlan
    joints: RigidJointGraphPlan
    articulation: ReducedArticulationPlan
    reference: RigidBodyKinematics
    dimensions: DimensionalScaleContract
    link_ids: RobotNameIDMap
    joint_ids: RobotNameIDMap
    report: AdapterReport
    evidence: URDFFormatEvidence
    target_id: str = eqx.field(static=True)

    @property
    def particle_plan(self) -> ParticleSetPlan:
        return self.particles

    @property
    def body_plan(self) -> RigidBodySetPlan:
        return self.bodies

    @property
    def joint_plan(self) -> RigidJointGraphPlan:
        return self.joints

    @property
    def articulation_plan(self) -> ReducedArticulationPlan:
        return self.articulation

    @property
    def reference_kinematics(self) -> RigidBodyKinematics:
        return self.reference

    @property
    def dimensional_contract(self) -> DimensionalScaleContract:
        return self.dimensions

    @property
    def link_name_to_id(self) -> tuple[tuple[str, np.int64], ...]:
        return self.link_ids.name_to_id

    @property
    def link_id_to_name(self) -> tuple[tuple[np.int64, str], ...]:
        return self.link_ids.id_to_name

    @property
    def joint_name_to_id(self) -> tuple[tuple[str, np.int64], ...]:
        return self.joint_ids.name_to_id

    @property
    def joint_id_to_name(self) -> tuple[tuple[np.int64, str], ...]:
        return self.joint_ids.id_to_name

    @property
    def negotiation(self) -> AdapterNegotiationResult:
        return self.report.negotiation



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
    ):
        self.report = report
        self.evidence = evidence
        super().__init__(status, message)


@dataclass(frozen=True, slots=True)
class _LinkRecord:
    name: str
    mass: float
    inertial_xyz: np.ndarray
    inertial_rpy: np.ndarray
    inertia_inertial: np.ndarray
    inertia_body: np.ndarray


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
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path} must contain exactly {size} numbers.")
    return np.asarray([_float(field, f"{path}[{index}]") for index, field in enumerate(fields)])


def _required_attribute(element: ET.Element, name: str, path: str, /) -> str:
    if name not in element.attrib:
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path}/@{name} is required.")
    value = element.attrib[name]
    if not value or value != value.strip():
        _fail(AdapterStatus.MALFORMED_SOURCE, f"{path}/@{name} must be non-empty and unpadded.")
    return value


def _loss(
    path: str,
    rationale: str,
    /,
    *,
    changes_interpretation: bool,
    category: str = "unsupported",
) -> AdapterLoss:
    return AdapterLoss(
        path,
        "import",
        category,
        rationale,
        changes_interpretation=changes_interpretation,
    )


def _attribute_losses(
    element: ET.Element,
    allowed: frozenset[str],
    path: str,
    losses: list[AdapterLoss],
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
    losses: list[AdapterLoss],
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
    losses: list[AdapterLoss],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    origin = _single_child(owner, "origin", owner_path, required=False)
    if origin is None:
        return np.zeros((3,), dtype=float), np.zeros((3,), dtype=float)
    path = f"{owner_path}/origin"
    _attribute_losses(origin, frozenset(("xyz", "rpy")), path, losses)
    _unknown_child_losses(origin, frozenset(), path, losses)
    _check_leaf_text(origin, path)
    xyz = (
        np.zeros((3,), dtype=float)
        if "xyz" not in origin.attrib
        else _vector(origin.attrib["xyz"], 3, f"{path}/@xyz")
    )
    rpy = (
        np.zeros((3,), dtype=float)
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
        dtype=float,
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
            scale = 2.0 * np.sqrt(
                1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]
            )
            quaternion = np.asarray(
                (
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                )
            )
        elif diagonal == 1:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]
            )
            quaternion = np.asarray(
                (
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                )
            )
        else:
            scale = 2.0 * np.sqrt(
                1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]
            )
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


def _parse_link(element: ET.Element, losses: list[AdapterLoss], /) -> _LinkRecord:
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
            _required_attribute(
                inertia_element, field, f"{inertial_path}/inertia"
            ),
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
        dtype=float,
    )
    eigenvalues = np.linalg.eigvalsh(inertia_inertial)
    scale = float(np.max(np.abs(inertia_inertial)))
    tolerance = 64.0 * np.finfo(float).eps * scale
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
            )
        )
    for index, collision in enumerate(_children(element, "collision")):
        del collision
        losses.append(
            _loss(
                f"{path}/collision/{index}",
                "Collision geometry is not represented by the articulation target.",
                changes_interpretation=True,
                category="dropped",
            )
        )
    return _LinkRecord(
        name,
        mass,
        inertial_xyz,
        inertial_rpy,
        inertia_inertial,
        inertia_body,
    )


def _parse_limit(
    joint: ET.Element,
    path: str,
    kind: str,
    losses: list[AdapterLoss],
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
    losses: list[AdapterLoss],
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
            )
        )
        return None
    _attribute_losses(
        dynamics, frozenset(("damping", "friction")), dynamics_path, losses
    )
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
            )
        )
    return damping


def _parse_joint(element: ET.Element, losses: list[AdapterLoss], /) -> _JointRecord:
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
            if norm <= np.finfo(float).eps:
                _fail(AdapterStatus.MALFORMED_SOURCE, f"{axis_path} must be nonzero.")
            axis = axis / norm
    lower, upper, effort, velocity = _parse_limit(
        element, path, kind, losses
    )
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


def _parse_tree(root: ET.Element, /) -> _ParsedURDF:
    if _tag_name(root) != "robot":
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF 1.0 input must have an unnamespaced <robot> document element.",
        )
    robot_name = _required_attribute(root, "name", "/robot")
    _check_leaf_text(root, "/robot")
    losses: list[AdapterLoss] = []
    _attribute_losses(root, frozenset(("name", "version")), "/robot", losses)
    if "version" in root.attrib and root.attrib["version"] != "1.0":
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Only the URDF 1.0 format version is supported.",
        )
    _unknown_child_losses(root, frozenset(("link", "joint", "material", "transmission")), "/robot", losses)
    link_elements = _children(root, "link")
    joint_elements = _children(root, "joint")
    if not link_elements:
        _fail(AdapterStatus.MALFORMED_SOURCE, "URDF robot must contain at least one link.")
    links = tuple(_parse_link(element, losses) for element in link_elements)
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
                _fail(AdapterStatus.INCONSISTENT_SOURCE, "URDF joint graph must be a tree.")
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
            )
        )
    return _ParsedURDF(
        robot_name,
        tuple(sorted(links, key=lambda item: item.name)),
        tuple(sorted(joints, key=lambda item: item.name)),
        root_name,
        tuple(traversal),
        tuple(sorted(losses, key=lambda item: (item.path, item.loss_id))),
    )


def _validated_xml(text: str, max_bytes: int, /) -> tuple[ET.Element, int]:
    if not isinstance(text, str):
        raise TypeError("URDF text must be a string.")
    maximum = int(max_bytes)
    if maximum <= 0:
        raise ValueError("max_bytes must be positive.")
    try:
        encoded = text.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE, "URDF text must be valid Unicode encodable as UTF-8."
        ) from error
    if len(encoded) > maximum:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"URDF source exceeds the configured {maximum}-byte size limit.",
        )
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
        if declaration_end < 0 or _XML_DECLARATION.fullmatch(stripped[: declaration_end + 2]) is None:
            _fail(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Only a standard UTF-8 XML 1.0 declaration is accepted.",
            )
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE, f"Malformed URDF XML: {error}."
        ) from error
    return root, len(encoded)


def _resolved_file(path: str | Path, allowed_root: str | Path, /) -> Path:
    source_text = str(path)
    root_text = str(allowed_root)
    if (
        "://" in source_text
        or source_text.startswith(("//", "\\\\"))
        or "://" in root_text
        or root_text.startswith(("//", "\\\\"))
    ):
        _fail(AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, "Network URDF locations are disabled.")
    try:
        root = Path(allowed_root).expanduser().resolve(strict=True)
    except OSError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE, "The URDF allowed root does not exist."
        ) from error
    if not root.is_dir():
        _fail(AdapterStatus.MALFORMED_SOURCE, "The URDF allowed root must be a directory.")
    candidate = Path(path).expanduser()
    candidate = candidate if candidate.is_absolute() else root / candidate
    try:
        normalized = candidate.resolve(strict=True)
    except OSError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE, "The requested URDF file does not exist."
        ) from error
    if not normalized.is_relative_to(root):
        _fail(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "The normalized URDF path escapes the configured allowed root.",
        )
    if not normalized.is_file():
        _fail(AdapterStatus.MALFORMED_SOURCE, "The normalized URDF path is not a file.")
    return normalized


def _world_link_frames(
    parsed: _ParsedURDF, /,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    frames: dict[str, tuple[np.ndarray, np.ndarray]] = {
        parsed.root_link: (np.zeros((3,), dtype=float), np.eye(3, dtype=float))
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


def _adapt(
    parsed: _ParsedURDF,
    source_id: str,
    source_path: str | None,
    source_size_bytes: int,
    waivers: Sequence[AdapterWaiver],
    waived_loss_paths: Sequence[str],
    /,
) -> RobotAdaptation:
    link_ids = RobotNameIDMap(tuple(link.name for link in parsed.links))
    joint_ids = RobotNameIDMap(tuple(joint.name for joint in parsed.joints))
    link_id = {name: int(identifier) for name, identifier in link_ids.name_to_id}
    joint_id = {
        name: int(identifier) for name, identifier in joint_ids.name_to_id
    }
    frames = _world_link_frames(parsed)
    count = len(parsed.links)
    masses = np.asarray([link.mass for link in parsed.links], dtype=float)
    inertias = np.stack([link.inertia_body for link in parsed.links], axis=0)
    positions = np.empty((count, 3), dtype=float)
    orientations = np.empty((count, 4), dtype=float)
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
    particles = ParticleSetPlan(
        np.asarray([link_id[link.name] for link in parsed.links], dtype=np.int64),
        masses,
        ambient_dimension=3,
        name=f"{parsed.robot_name}-links",
        domain_labels=("rigid_body", "robot_link"),
        coordinate_dtype="float64",
    )
    fixed_mask = np.asarray(
        [link.name == parsed.root_link for link in parsed.links], dtype=bool
    )
    bodies = RigidBodySetPlan(
        np.zeros((count,), dtype=np.int64),
        inertias,
        fixed_mask=fixed_mask,
        name=f"{parsed.robot_name}-rigid-bodies",
    )
    reference = RigidBodyKinematics(
        jnp.asarray(positions),
        jnp.zeros_like(jnp.asarray(positions)),
        jnp.asarray(orientations),
        jnp.zeros((count, 3), dtype=jnp.asarray(positions).dtype),
    )
    anchors: dict[str, np.ndarray] = {}
    axes: dict[str, np.ndarray] = {}
    joint_evidence: list[URDFJointEvidence] = []
    for joint in parsed.joints:
        anchor, joint_rotation = frames[joint.child]
        axis = joint_rotation @ joint.axis
        anchors[joint.name] = anchor
        axes[joint.name] = axis
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
    fixed_records = tuple(joint for joint in parsed.joints if joint.kind == "fixed")
    hinge_records = tuple(
        joint for joint in parsed.joints if joint.kind in ("revolute", "continuous")
    )
    prismatic_records = tuple(
        joint for joint in parsed.joints if joint.kind == "prismatic"
    )
    fixed_plan = (
        None
        if not fixed_records
        else FixedJointSetPlan(
            np.asarray([joint_id[joint.name] for joint in fixed_records], dtype=np.int64),
            np.asarray([link_id[joint.parent] for joint in fixed_records], dtype=np.int64),
            np.asarray([link_id[joint.child] for joint in fixed_records], dtype=np.int64),
        )
    )
    hinge_plan = (
        None
        if not hinge_records
        else HingeJointSetPlan(
            np.asarray([joint_id[joint.name] for joint in hinge_records], dtype=np.int64),
            np.asarray([link_id[joint.parent] for joint in hinge_records], dtype=np.int64),
            np.asarray([link_id[joint.child] for joint in hinge_records], dtype=np.int64),
            np.stack([anchors[joint.name] for joint in hinge_records]),
            np.stack([axes[joint.name] for joint in hinge_records]),
        )
    )
    prismatic_plan = (
        None
        if not prismatic_records
        else PrismaticJointSetPlan(
            np.asarray([joint_id[joint.name] for joint in prismatic_records], dtype=np.int64),
            np.asarray([link_id[joint.parent] for joint in prismatic_records], dtype=np.int64),
            np.asarray([link_id[joint.child] for joint in prismatic_records], dtype=np.int64),
            np.stack([anchors[joint.name] for joint in prismatic_records]),
            np.stack([axes[joint.name] for joint in prismatic_records]),
        )
    )
    joints = RigidJointGraphPlan(
        fixed=fixed_plan, hinge=hinge_plan, prismatic=prismatic_plan
    )
    articulation = ReducedArticulationPlan(
        link_id[parsed.root_link],
        np.asarray([joint_id[joint.name] for joint in parsed.traversal], dtype=np.int64),
        np.asarray([link_id[joint.parent] for joint in parsed.traversal], dtype=np.int64),
        np.asarray([link_id[joint.child] for joint in parsed.traversal], dtype=np.int64),
    )
    dimensions = DimensionalScaleContract.si()
    target_id = canonical_fingerprint(
        {
            "kind": "native-robot-articulation-adaptation",
            "particles": particles.plan_id,
            "bodies": bodies.plan_id,
            "joints": joints.plan_id,
            "articulation": articulation.plan_id,
            "reference": array_tree_fingerprint(
                {
                    "position": positions,
                    "orientation": orientations,
                    "velocity": np.zeros_like(positions),
                    "angular_velocity": np.zeros_like(positions),
                }
            ),
            "dimensions": dimensions.scale_id,
            "link_mapping": link_ids.mapping_id,
            "joint_mapping": joint_ids.mapping_id,
        }
    )
    applied_waivers = _waivers_for_losses(
        parsed.losses, waivers, waived_loss_paths
    )
    evidence = URDFFormatEvidence(
        parsed.robot_name,
        source_path,
        source_size_bytes,
        parsed.root_link,
        link_ids,
        joint_ids,
        link_evidence,
        joint_evidence,
        tuple(loss.path for loss in parsed.losses),
        source_id,
        target_id,
    )
    requirements = (
        AdapterRequirement("robot.connected-oriented-tree"),
        AdapterRequirement("robot.inertial-rigid-links"),
        AdapterRequirement("robot.fixed-hinge-prismatic-kinematics"),
        AdapterRequirement("robot.reference-kinematics"),
        AdapterRequirement("robot.joint-limit-evidence"),
        AdapterRequirement("robot.joint-damping-evidence"),
        AdapterRequirement("robot.si-dimensional-contract"),
        AdapterRequirement("robot.stable-int64-name-maps"),
    )
    capabilities = (
        AdapterCapability("robot.connected-oriented-tree"),
        AdapterCapability("robot.inertial-rigid-links"),
        AdapterCapability("robot.fixed-hinge-prismatic-kinematics"),
        AdapterCapability("robot.reference-kinematics"),
        AdapterCapability("robot.joint-limit-evidence"),
        AdapterCapability("robot.joint-damping-evidence"),
        AdapterCapability("robot.si-dimensional-contract"),
        AdapterCapability("robot.stable-int64-name-maps"),
    )
    negotiation = negotiate_adapter(
        requirements, capabilities, losses=parsed.losses, waivers=applied_waivers
    )
    report = AdapterReport(
        negotiation.status,
        "URDF 1.0",
        "phydrax-native-robot-articulation",
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
            "phydrax-native-robot-articulation",
            qualifiers={
                "base": "fixed",
                "body-frame": "center-of-mass",
                "coordinates": "SI",
            },
        ),
        coordinate_mapping=(
            "URDF parent-link joint origin -> zero-configuration world anchor",
            "URDF joint-frame axis -> zero-configuration world reference axis",
            "URDF link origin -> COM-centred body position with link-frame offset retained as evidence",
            "URDF inertial-frame tensor -> body-frame tensor by R_link_inertial I R_link_inertial^T",
            "URDF roll-pitch-yaw -> scalar-first unit quaternion",
        ),
        preserved_fields=(
            "robot and link/joint names",
            "oriented connected tree topology",
            "link masses and COM positions",
            "link inertial-frame rotations and tensors",
            "fixed, revolute, continuous, and prismatic joint kinematics",
            "joint position, effort, and velocity limits",
            "joint damping",
            "zero-configuration reference kinematics",
        ),
        assumptions=(
            "URDF linear dimensions are metres",
            "URDF masses are kilograms",
            "URDF angles are radians",
            "URDF inertia components are kg*m^2 about the link COM",
            "the unique root link is fixed to the world",
            "external macros, includes, resources, networks, and plugins are not evaluated",
        ),
        losses=parsed.losses,
        requirements=requirements,
        capabilities=capabilities,
        waivers=applied_waivers,
    )
    if not negotiation.valid:
        raise URDFImportError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "URDF input contains unwaived interpretation-changing semantics.",
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
    return RobotAdaptation(
        particles,
        bodies,
        joints,
        articulation,
        reference,
        dimensions,
        link_ids,
        joint_ids,
        report,
        evidence,
        target_id,
    )


def parse_urdf_text(
    text: str,
    /,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    waivers: Sequence[AdapterWaiver] = (),
    waived_loss_paths: Sequence[str] = (),
) -> RobotAdaptation:
    """Parse bounded URDF text without resolving any external resource."""

    root, source_size = _validated_xml(text, max_bytes)
    parsed = _parse_tree(root)
    source_id = canonical_fingerprint(
        {"kind": "urdf-source", "format": "URDF 1.0", "utf8_text": text}
    )
    return _adapt(
        parsed,
        source_id,
        None,
        source_size,
        waivers,
        waived_loss_paths,
    )


def parse_urdf_file(
    path: str | Path,
    /,
    *,
    allowed_root: str | Path,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    waivers: Sequence[AdapterWaiver] = (),
    waived_loss_paths: Sequence[str] = (),
) -> RobotAdaptation:
    """Resolve and parse one bounded UTF-8 URDF beneath an explicit allowed root."""

    maximum = int(max_bytes)
    if maximum <= 0:
        raise ValueError("max_bytes must be positive.")
    source = _resolved_file(path, allowed_root)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE,
            "The normalized URDF file could not be inspected.",
        ) from error
    if size > maximum:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"URDF source exceeds the configured {maximum}-byte size limit.",
        )
    try:
        with source.open("rb") as stream:
            data = stream.read(maximum + 1)
    except OSError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE,
            "The normalized URDF file could not be read.",
        ) from error
    if len(data) > maximum:
        _fail(
            AdapterStatus.MALFORMED_SOURCE,
            f"URDF source exceeds the configured {maximum}-byte size limit.",
        )
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise URDFImportError(
            AdapterStatus.MALFORMED_SOURCE, "URDF files must be strict UTF-8."
        ) from error
    root, source_size = _validated_xml(text, maximum)
    parsed = _parse_tree(root)
    source_id = canonical_fingerprint(
        {"kind": "urdf-source", "format": "URDF 1.0", "utf8_text": text}
    )
    return _adapt(
        parsed,
        source_id,
        str(source),
        source_size,
        waivers,
        waived_loss_paths,
    )


__all__ = [
    "RobotAdaptation",
    "RobotNameIDMap",
    "URDFFormatEvidence",
    "URDFImportError",
    "URDFJointEvidence",
    "URDFLinkEvidence",
    "parse_urdf_file",
    "parse_urdf_text",
]
