#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded ROS log profiles lowered into typed measurement collections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import new as new_digest
from importlib import import_module, util
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np

from .._fingerprint import canonical_fingerprint
from .._physical import SpatialCoordinateContract
from ..geometry import FrameTransformGraph, FrameTransformTimeline, RigidFrame
from ..qualification import ReferenceArtifactManifest
from ..units import derived_unit, METER, ONE, RADIAN, SECOND
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


_ANGULAR_VELOCITY = derived_unit("rad/s", ((RADIAN, 1), (SECOND, -1)))
_ACCELERATION = derived_unit("m/s2", ((METER, 1), (SECOND, -2)))


_IMAGE_ENCODINGS = {
    "mono8": (np.dtype(np.uint8), ("intensity",)),
    "8UC1": (np.dtype(np.uint8), ("intensity",)),
    "mono16": (np.dtype(np.uint16), ("intensity",)),
    "16UC1": (np.dtype(np.uint16), ("intensity",)),
    "32FC1": (np.dtype(np.float32), ("intensity",)),
    "64FC1": (np.dtype(np.float64), ("intensity",)),
    "rgb8": (np.dtype(np.uint8), ("red", "green", "blue")),
    "bgr8": (np.dtype(np.uint8), ("blue", "green", "red")),
    "rgba8": (np.dtype(np.uint8), ("red", "green", "blue", "alpha")),
    "bgra8": (np.dtype(np.uint8), ("blue", "green", "red", "alpha")),
    "8UC3": (np.dtype(np.uint8), ("channel-0", "channel-1", "channel-2")),
    "8UC4": (
        np.dtype(np.uint8),
        ("channel-0", "channel-1", "channel-2", "channel-3"),
    ),
}


def _decode_image(message: Any, /) -> tuple[np.ndarray, tuple[str, ...]]:
    encoding = str(message.encoding)
    if encoding not in _IMAGE_ENCODINGS:
        raise ValueError(f"Unsupported ROS image encoding {encoding!r}.")
    base_dtype, labels = _IMAGE_ENCODINGS[encoding]
    byte_order = ">" if bool(message.is_bigendian) else "<"
    dtype = base_dtype.newbyteorder(byte_order)
    height, width, step = int(message.height), int(message.width), int(message.step)
    channels = len(labels)
    row_bytes = width * channels * dtype.itemsize
    payload = memoryview(message.data)
    if height < 1 or width < 1 or step < row_bytes or len(payload) < height * step:
        raise ValueError(
            "ROS image dimensions, step, and payload length are inconsistent."
        )
    shape = (height, width) if channels == 1 else (height, width, channels)
    strides = (
        (step, dtype.itemsize)
        if channels == 1
        else (step, channels * dtype.itemsize, dtype.itemsize)
    )
    values = np.ndarray(shape, dtype=dtype, buffer=payload, strides=strides).copy()
    if encoding == "bgr8":
        values = values[..., ::-1]
        labels = ("red", "green", "blue")
    elif encoding == "bgra8":
        values = values[..., (2, 1, 0, 3)]
        labels = ("red", "green", "blue", "alpha")
    return values.astype(np.float64), labels


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

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value and value == value.strip()
            for value in (self.topic, self.message_type, self.frame_id)
        ):
            raise ValueError(
                "ROS topic, message_type, and frame_id must be canonical nonempty text."
            )
        if not isinstance(self.kind, RosMessageKind):
            raise TypeError("kind must be RosMessageKind.")


@dataclass(frozen=True, slots=True)
class RosMessageRecord:
    topic: str
    kind: RosMessageKind
    sensor_time: float
    bag_time: float
    sequence: int
    payload: dict[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.topic, str)
            or not self.topic
            or self.topic != self.topic.strip()
        ):
            raise ValueError("topic must be canonical nonempty text.")
        if not isinstance(self.kind, RosMessageKind):
            raise TypeError("kind must be RosMessageKind.")
        if not np.isfinite(self.sensor_time) or not np.isfinite(self.bag_time):
            raise ValueError("ROS sensor_time and bag_time must be finite.")
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, Integral)
            or self.sequence < 0
        ):
            raise ValueError("sequence must be a nonnegative integer.")
        if not isinstance(self.payload, dict):
            raise TypeError("payload must be a dictionary.")


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
            assets.extend(
                _lower_record(record, profiles[record.topic], reference, campaign_id)
            )
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
        values, component_labels = _decode_image(message)
        payload = {
            "values": values,
            "encoding": str(message.encoding),
            "component_labels": component_labels,
        }
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
        ranges = np.asarray(message.ranges, dtype=np.float64)
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
    norm = np.linalg.norm(quaternion)
    if not np.isfinite(norm) or norm <= np.finfo(np.float64).tiny:
        raise ValueError("ROS transform quaternion must have finite nonzero norm.")
    w, x, y, z = quaternion / norm
    return np.asarray(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )


def _payload_contract(
    record: RosMessageRecord,
    name: str,
    values: np.ndarray,
    frame_id: str,
    /,
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...], ValueLayout, object]:
    array = np.asarray(values)
    if record.kind is RosMessageKind.IMAGE:
        if name != "values" or array.ndim not in (2, 3):
            raise ValueError("Normalized ROS image payload has an invalid shape.")
        if array.ndim == 2:
            return (
                array.astype(np.float64),
                array.shape,
                ("row", "column"),
                ValueLayout.scalar(),
                ONE,
            )
        labels = tuple(record.payload["component_labels"])
        return (
            array.astype(np.float64),
            array.shape[:2],
            ("row", "column"),
            ValueLayout(
                ValueKind.VECTOR,
                (array.shape[-1],),
                labels,
                f"{frame_id}:image-components",
            ),
            ONE,
        )
    vector_units = {
        (RosMessageKind.IMU, "angular_velocity"): _ANGULAR_VELOCITY,
        (RosMessageKind.IMU, "linear_acceleration"): _ACCELERATION,
        (RosMessageKind.ODOMETRY, "position"): METER,
        (RosMessageKind.ODOMETRY, "quaternion"): ONE,
    }
    if (record.kind, name) in vector_units:
        labels = ("w", "x", "y", "z") if name == "quaternion" else ("x", "y", "z")
        if array.shape != (len(labels),):
            raise ValueError(f"ROS {name} must have shape ({len(labels)},).")
        return (
            array.reshape((1, len(labels))),
            (1,),
            ("observation",),
            ValueLayout(ValueKind.VECTOR, (len(labels),), labels, frame_id),
            vector_units[(record.kind, name)],
        )
    if record.kind is RosMessageKind.CAMERA_INFO and name == "intrinsic":
        if array.shape != (3, 3):
            raise ValueError("ROS camera intrinsic must have shape (3, 3).")
        return (
            array.reshape((1, 3, 3)),
            (1,),
            ("observation",),
            ValueLayout(ValueKind.GENERAL_TENSOR, (3, 3), component_frame_id=frame_id),
            ONE,
        )
    if record.kind is RosMessageKind.CAMERA_INFO and name == "distortion":
        if array.ndim != 1 or array.size < 1:
            raise ValueError("ROS camera distortion must be a nonempty vector.")
        return (
            array.reshape((1, array.size)),
            (1,),
            ("observation",),
            ValueLayout(
                ValueKind.VECTOR,
                (array.size,),
                component_frame_id=frame_id,
            ),
            ONE,
        )
    if array.ndim != 1 or array.size < 1:
        raise ValueError(f"ROS {name} requires a nonempty rank-one scalar payload.")
    return array, array.shape, (name,), ValueLayout.scalar(), ONE


def _lower_record(
    record: RosMessageRecord,
    profile: RosTopicProfile,
    reference: ReferenceArtifactManifest,
    campaign: str,
) -> list[MeasurementAsset]:
    sensor_clock_id = f"ros-header:{profile.topic}"
    bag_clock_id = "ros-bag-clock"
    acquisition = AcquisitionIdentity(
        f"{campaign}:{record.topic}:{record.sequence}",
        record.topic,
        record.kind.value,
        "ros-topic",
        clock_id=sensor_clock_id,
    )
    derivation = DerivationRecord(
        DataOrigin.EXTERNAL,
        DataStage.CALIBRATED,
        transformation_id=canonical_fingerprint(
            {"kind": "ros-lowering", "topic": record.topic, "sequence": record.sequence}
        ),
    )
    metadata = {
        "ros_sensor_time_seconds": record.sensor_time,
        "ros_bag_time_seconds": record.bag_time,
        "ros_sensor_clock_id": sensor_clock_id,
        "ros_bag_clock_id": bag_clock_id,
        "ros_frame_id": profile.frame_id,
    }
    result = []
    if record.kind is RosMessageKind.LASER_SCAN:
        ranges = np.asarray(record.payload["ranges"])
        angles = np.asarray(record.payload["angles"])
        directions = np.stack(
            (np.cos(angles), np.sin(angles), np.zeros_like(angles)), axis=-1
        )
        contract = SpatialCoordinateContract(
            METER,
            coordinate_system="cartesian",
            reference_frame=profile.frame_id,
        )
        support = RaySampleSupport(
            np.zeros_like(directions),
            directions,
            tuple(
                f"{record.topic}:{record.sequence}:{index}"
                for index in range(ranges.size)
            ),
            contract,
            sample_times=np.full(ranges.shape, record.sensor_time),
            time_unit=SECOND,
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
                metadata,
            )
        )
    elif record.kind is RosMessageKind.POINT_CLOUD:
        points = np.asarray(record.payload["points"], dtype=np.float64)
        contract = SpatialCoordinateContract(
            METER,
            coordinate_system="cartesian",
            reference_frame=profile.frame_id,
        )
        valid = np.all(np.isfinite(points), axis=-1)
        support = PointSampleSupport(
            points,
            tuple(
                f"{record.topic}:{record.sequence}:{index}"
                for index in range(points.shape[0])
            ),
            contract,
            sample_times=np.full((points.shape[0],), record.sensor_time),
            time_unit=SECOND,
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
                    profile.frame_id,
                ),
                acquisition,
                derivation,
                reference,
                metadata,
            )
        )
    else:
        for name, values in record.payload.items():
            if not isinstance(values, np.ndarray):
                continue
            array, sample_shape, axis_labels, layout, unit = _payload_contract(
                record,
                name,
                values,
                profile.frame_id,
            )
            support = IndexSampleSupport(
                sample_shape,
                axis_labels,
                frame_id=profile.frame_id,
            )
            result.append(
                _asset(
                    f"{campaign}:{record.topic}:{record.sequence}:{name}",
                    array,
                    support,
                    name.replace("_", "-"),
                    unit,
                    layout,
                    acquisition,
                    derivation,
                    reference,
                    metadata,
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
    metadata: dict[str, object],
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
        asset_id,
        field,
        reference,
        derivation,
        acquisition=acquisition,
        metadata=metadata,
    )


__all__ = [
    "RosMessageKind",
    "RosMessageRecord",
    "RosTopicProfile",
    "RosbagImportPlan",
    "RosbagImportResult",
]
