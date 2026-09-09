#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded ROS log profiles lowered into typed measurement collections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import new as new_digest
from importlib import import_module, util
from pathlib import Path
from typing import Any

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..geometry import FrameTransformGraph, FrameTransformTimeline, RigidFrame
from ..qualification import ReferenceArtifactManifest
from ..units import METER, ONE, SECOND
from ._asset import (
    AcquisitionIdentity,
    DataOrigin,
    DataStage,
    DerivationRecord,
    MeasurementAsset,
)
from ._collection import MeasurementCollection, MeasurementRole, MeasurementRoleAssignment
from ._field import QuantityField, SamplingSemantics, SpatialSamplingKind
from ._quantity import QuantitySpec, ValueKind, ValueLayout
from ._support import IndexSampleSupport, PointSampleSupport, RaySampleSupport
from ._time import SampleTimeAxis


class RosMessageKind(StrEnum):
    IMAGE = "image"
    CAMERA_INFO = "camera_info"
    POINT_CLOUD = "point_cloud"
    LASER_SCAN = "laser_scan"
    IMU = "imu"
    ODOMETRY = "odometry"
    JOINT_STATE = "joint_state"
    TRANSFORM = "transform"


@dataclass(frozen=True, slots=True)
class RosTopicProfile:
    topic: str
    message_type: str
    kind: RosMessageKind
    frame_id: str


@dataclass(frozen=True, slots=True)
class RosMessageRecord:
    topic: str
    kind: RosMessageKind
    sensor_time: float
    bag_time: float
    sequence: int
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RosbagImportResult:
    collection: MeasurementCollection
    frame_graph: FrameTransformGraph | None
    bag_time_ordered: bool
    result_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "result_id",
            canonical_fingerprint(
                {
                    "kind": "rosbag-import-result",
                    "collection": self.collection.content_id,
                    "frame_graph": None
                    if self.frame_graph is None
                    else self.frame_graph.graph_id,
                    "bag_time_ordered": self.bag_time_ordered,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RosbagImportPlan:
    profiles: tuple[RosTopicProfile, ...]
    maximum_messages: int = 1_000_000
    maximum_decoded_bytes: int = 4_000_000_000
    maximum_source_bytes: int = 8_000_000_000
    plan_id: str = field(init=False)

    def __post_init__(self) -> None:
        profiles = tuple(self.profiles)
        if not profiles or len({value.topic for value in profiles}) != len(profiles):
            raise ValueError("ROS topic profiles must be nonempty and unique by topic.")
        if (
            self.maximum_messages < 1
            or self.maximum_decoded_bytes < 1
            or self.maximum_source_bytes < 1
        ):
            raise ValueError("ROS resource limits must be positive.")
        object.__setattr__(self, "profiles", profiles)
        object.__setattr__(
            self,
            "plan_id",
            canonical_fingerprint(
                {
                    "kind": "rosbag-import-plan",
                    "profiles": [
                        (
                            value.topic,
                            value.message_type,
                            value.kind.value,
                            value.frame_id,
                        )
                        for value in profiles
                    ],
                    "maximum_messages": self.maximum_messages,
                    "maximum_decoded_bytes": self.maximum_decoded_bytes,
                    "maximum_source_bytes": self.maximum_source_bytes,
                }
            ),
        )

    def read(
        self,
        path: str | Path,
        reference: ReferenceArtifactManifest,
        /,
        *,
        campaign_id: str,
    ) -> RosbagImportResult:
        if util.find_spec("rosbags") is None:
            raise ImportError("ROS bag admission requires the optional 'rosbag' extra.")
        reference.require_rights()
        if reference.size_bytes > self.maximum_source_bytes:
            raise MemoryError("ROS bag exceeds maximum_source_bytes.")
        source = Path(path).resolve()
        _verify(source, reference)
        backend = import_module("rosbags.highlevel")
        records = []
        profiles = {value.topic: value for value in self.profiles}
        with backend.AnyReader([source]) as reader:
            connections = [
                value
                for value in reader.connections
                if value.topic in profiles
                and value.msgtype == profiles[value.topic].message_type
            ]
            for sequence, (connection, timestamp, raw) in enumerate(
                reader.messages(connections=connections)
            ):
                if sequence >= self.maximum_messages:
                    raise MemoryError("ROS bag exceeds maximum_messages.")
                message = reader.deserialize(raw, connection.msgtype)
                records.append(
                    _normalize_message(
                        profiles[connection.topic], message, timestamp * 1.0e-9, sequence
                    )
                )
        return self.lower(tuple(records), reference, campaign_id=campaign_id)

    def lower(
        self,
        records: tuple[RosMessageRecord, ...],
        reference: ReferenceArtifactManifest,
        /,
        *,
        campaign_id: str,
    ) -> RosbagImportResult:
        profiles = {value.topic: value for value in self.profiles}
        if any(
            record.topic not in profiles or profiles[record.topic].kind is not record.kind
            for record in records
        ):
            raise ValueError(
                "Every normalized ROS record must match a declared topic profile."
            )
        if len(records) > self.maximum_messages:
            raise MemoryError("ROS records exceed maximum_messages.")
        decoded = sum(
            sum(
                np.asarray(value).nbytes
                for value in record.payload.values()
                if isinstance(value, (np.ndarray, list, tuple))
            )
            for record in records
        )
        if decoded > self.maximum_decoded_bytes:
            raise MemoryError("ROS records exceed maximum_decoded_bytes.")
        assets = []
        transforms: dict[tuple[str, str], list[tuple[float, RigidFrame]]] = {}
        bag_times = []
        for record in records:
            bag_times.append(record.bag_time)
            if record.kind is RosMessageKind.TRANSFORM:
                edge = (
                    str(record.payload["source_frame"]),
                    str(record.payload["target_frame"]),
                )
                transforms.setdefault(edge, []).append(
                    (
                        record.sensor_time,
                        RigidFrame(
                            record.payload["rotation"], record.payload["translation"]
                        ),
                    )
                )
                continue
            assets.extend(_lower_record(record, reference, campaign_id))
        if not assets:
            raise ValueError("ROS bag produced no supported measurement assets.")
        roles = tuple(
            MeasurementRoleAssignment(value.asset_id, MeasurementRole.OBSERVATION)
            for value in assets
        )
        collection = MeasurementCollection(
            campaign_id,
            campaign_id,
            tuple(assets),
            roles,
            metadata={"rosbag_plan_id": self.plan_id},
        )
        timelines = []
        for (source, target), samples in transforms.items():
            samples.sort(key=lambda value: value[0])
            if len(samples) < 2:
                continue
            axis = SampleTimeAxis(
                f"{source}-to-{target}",
                np.asarray([value[0] for value in samples]),
                SECOND,
            )
            timelines.append(
                FrameTransformTimeline(
                    source,
                    target,
                    axis,
                    tuple(value[1] for value in samples),
                    f"{campaign_id}:tf",
                )
            )
        graph = None if not timelines else FrameTransformGraph(tuple(timelines))
        ordered = all(
            right >= left
            for left, right in zip(bag_times[:-1], bag_times[1:], strict=True)
        )
        return RosbagImportResult(collection, graph, ordered)


def _verify(path: Path, reference: ReferenceArtifactManifest) -> None:
    if path.is_file():
        members = (path,)
        root = path.parent
    elif path.is_dir():
        members = tuple(sorted(value for value in path.rglob("*") if value.is_file()))
        root = path
    else:
        raise ValueError("ROS bag source does not exist.")
    size = sum(value.stat().st_size for value in members)
    if size != reference.size_bytes:
        raise ValueError("ROS bag size disagrees with the reference manifest.")
    digest = new_digest(reference.checksum_algorithm)
    for member in members:
        relative = str(member.relative_to(root)).encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with member.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
    if digest.hexdigest() != reference.checksum:
        raise ValueError("ROS bag checksum disagrees with the reference manifest.")


def _stamp(message: Any) -> float:
    return float(message.header.stamp.sec) + 1.0e-9 * float(message.header.stamp.nanosec)


def _normalize_message(
    profile: RosTopicProfile, message: Any, bag_time: float, sequence: int
) -> RosMessageRecord:
    kind = profile.kind
    payload: dict[str, Any]
    if kind is RosMessageKind.IMAGE:
        data = np.frombuffer(message.data, dtype=np.uint8).reshape(
            (message.height, message.step)
        )
        payload = {"values": data[:, : message.width], "encoding": message.encoding}
    elif kind is RosMessageKind.CAMERA_INFO:
        payload = {
            "intrinsic": np.asarray(message.k).reshape((3, 3)),
            "distortion": np.asarray(message.d),
        }
    elif kind is RosMessageKind.POINT_CLOUD:
        fields = {field.name: field for field in message.fields}
        if any(name not in fields for name in ("x", "y", "z")):
            raise ValueError("PointCloud2 profile requires x, y, and z fields.")
        endian = ">" if message.is_bigendian else "<"
        dtype = np.dtype(
            {
                "names": ("x", "y", "z"),
                "formats": (f"{endian}f4",) * 3,
                "offsets": tuple(fields[name].offset for name in ("x", "y", "z")),
                "itemsize": message.point_step,
            }
        )
        structured = np.ndarray(
            shape=(message.height, message.width),
            dtype=dtype,
            buffer=message.data,
            strides=(message.row_step, message.point_step),
        )
        payload = {
            "points": np.stack(
                (structured["x"], structured["y"], structured["z"]), axis=-1
            ).reshape((-1, 3))
        }
    elif kind is RosMessageKind.LASER_SCAN:
        ranges = np.asarray(message.ranges, dtype=float)
        angles = message.angle_min + message.angle_increment * np.arange(ranges.size)
        payload = {
            "ranges": ranges,
            "angles": angles,
            "range_min": message.range_min,
            "range_max": message.range_max,
        }
    elif kind is RosMessageKind.IMU:
        payload = {
            "angular_velocity": np.asarray(
                (
                    message.angular_velocity.x,
                    message.angular_velocity.y,
                    message.angular_velocity.z,
                )
            ),
            "linear_acceleration": np.asarray(
                (
                    message.linear_acceleration.x,
                    message.linear_acceleration.y,
                    message.linear_acceleration.z,
                )
            ),
        }
    elif kind is RosMessageKind.JOINT_STATE:
        payload = {
            "names": tuple(message.name),
            "positions": np.asarray(message.position),
            "velocities": np.asarray(message.velocity),
        }
    elif kind is RosMessageKind.ODOMETRY:
        quaternion = np.asarray(
            (
                message.pose.pose.orientation.w,
                message.pose.pose.orientation.x,
                message.pose.pose.orientation.y,
                message.pose.pose.orientation.z,
            )
        )
        payload = {
            "position": np.asarray(
                (
                    message.pose.pose.position.x,
                    message.pose.pose.position.y,
                    message.pose.pose.position.z,
                )
            ),
            "quaternion": quaternion,
        }
    elif kind is RosMessageKind.TRANSFORM:
        transform = message.transform
        q = np.asarray(
            (
                transform.rotation.w,
                transform.rotation.x,
                transform.rotation.y,
                transform.rotation.z,
            )
        )
        rotation = _quaternion_rotation(q)
        payload = {
            "source_frame": message.child_frame_id,
            "target_frame": message.header.frame_id,
            "rotation": rotation,
            "translation": np.asarray(
                (
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                )
            ),
        }
    else:
        raise ValueError(
            "PointCloud2 requires an explicit normalized point payload in this bounded profile."
        )
    return RosMessageRecord(
        profile.topic, kind, _stamp(message), bag_time, sequence, payload
    )


def _quaternion_rotation(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


def _lower_record(
    record: RosMessageRecord, reference: ReferenceArtifactManifest, campaign: str
) -> list[MeasurementAsset]:
    acquisition = AcquisitionIdentity(
        f"{campaign}:{record.topic}:{record.sequence}",
        record.topic,
        record.kind.value,
        "ros-topic",
        clock_id="ros-header-clock",
    )
    derivation = DerivationRecord(
        DataOrigin.EXTERNAL,
        DataStage.CALIBRATED,
        transformation_id=canonical_fingerprint(
            {"kind": "ros-lowering", "topic": record.topic, "sequence": record.sequence}
        ),
    )
    result = []
    if record.kind is RosMessageKind.LASER_SCAN:
        ranges = np.asarray(record.payload["ranges"])
        angles = np.asarray(record.payload["angles"])
        directions = np.stack(
            (np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1
        )
        contract = SpatialCoordinateContract(
            METER, coordinate_system="cartesian", reference_frame="sensor"
        )
        support = RaySampleSupport(
            np.zeros_like(directions),
            directions,
            tuple(
                f"{record.topic}:{record.sequence}:{index}"
                for index in range(ranges.size)
            ),
            contract,
            active_mask=np.isfinite(ranges),
            near=np.full(ranges.shape, record.payload["range_min"]),
            far=np.full(ranges.shape, record.payload["range_max"]),
        )
        result.append(
            _asset(
                f"{campaign}:{record.topic}:{record.sequence}:range",
                ranges,
                support,
                "range",
                METER,
                ValueLayout.scalar(),
                acquisition,
                derivation,
                reference,
            )
        )
    elif record.kind is RosMessageKind.POINT_CLOUD:
        points = np.asarray(record.payload["points"], dtype=float)
        contract = SpatialCoordinateContract(
            METER, coordinate_system="cartesian", reference_frame="sensor"
        )
        valid = np.all(np.isfinite(points), axis=-1)
        support = PointSampleSupport(
            points,
            tuple(
                f"{record.topic}:{record.sequence}:{index}"
                for index in range(points.shape[0])
            ),
            contract,
            active_mask=valid,
        )
        result.append(
            _asset(
                f"{campaign}:{record.topic}:{record.sequence}:points",
                points,
                support,
                "position",
                METER,
                ValueLayout(
                    ValueKind.VECTOR,
                    (3,),
                    ("x", "y", "z"),
                    "sensor",
                ),
                acquisition,
                derivation,
                reference,
            )
        )
    else:
        for name, values in record.payload.items():
            if not isinstance(values, np.ndarray):
                continue
            array = np.asarray(values)
            support = IndexSampleSupport(
                array.shape[:-1]
                if array.ndim > 1 and array.shape[-1] in (2, 3, 4, 9)
                else array.shape,
                tuple(
                    f"axis-{index}"
                    for index in range(
                        array.ndim
                        - (1 if array.ndim > 1 and array.shape[-1] in (2, 3, 4, 9) else 0)
                    )
                ),
                frame_id="sensor",
            )
            component_shape = array.shape[len(support.sample_shape) :]
            layout = (
                ValueLayout.scalar()
                if not component_shape
                else ValueLayout(
                    ValueKind.VECTOR, component_shape, component_frame_id="sensor"
                )
            )
            result.append(
                _asset(
                    f"{campaign}:{record.topic}:{record.sequence}:{name}",
                    array,
                    support,
                    name.replace("_", "-"),
                    ONE,
                    layout,
                    acquisition,
                    derivation,
                    reference,
                )
            )
    return result


def _asset(
    asset_id: str,
    values: np.ndarray,
    support,
    name: str,
    unit,
    layout: ValueLayout,
    acquisition: AcquisitionIdentity,
    derivation: DerivationRecord,
    reference: ReferenceArtifactManifest,
) -> MeasurementAsset:
    field = QuantityField(
        f"{asset_id}.field",
        QuantitySpec("ros", name, name, unit, f"ros.{name}"),
        layout,
        support,
        SamplingSemantics(SpatialSamplingKind.POINT),
        values,
        np.all(
            np.isfinite(values), axis=tuple(range(len(support.sample_shape), values.ndim))
        )
        if values.ndim > len(support.sample_shape)
        else np.isfinite(values),
    )
    return MeasurementAsset.from_single_reference(
        asset_id, field, reference, derivation, acquisition=acquisition
    )


__all__ = [
    "RosMessageKind",
    "RosMessageRecord",
    "RosTopicProfile",
    "RosbagImportPlan",
    "RosbagImportResult",
]
