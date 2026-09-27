#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded HDF5 adapter for the pinned upcoming openPMD LaserEnvelope draft."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import BinaryIO, TYPE_CHECKING

import h5py
import numpy as np

from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    ResourceLimits,
    ResourceReadError,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._publication import publish_bytes
from ._openpmd_base import (
    attribute_text,
    BoundedHDF5Buffer,
    ELECTRIC_FIELD_DIMENSION,
    LENGTH_DIMENSION,
    numeric_attribute,
    OPENPMD_LASER_ENVELOPE_REVISION,
    OpenPMDDimension,
    OpenPMDHDF5Inventory,
    OpenPMDIterationTime,
    OpenPMDUnit,
    OpenPMDUnsupportedError,
    preflight_hdf5,
    read_grid_unit_si,
    read_iteration,
    read_record_unit,
    read_series_root,
    required_text,
    scalar_attribute,
    TIME_DIMENSION,
    write_grid_units,
    write_iteration,
    write_record_unit,
    write_series_root,
)
from ._report import AdapterReport, AdapterStatus


if TYPE_CHECKING:
    from ..geometry.analytic._operations import RigidFrame
    from ..optics.wave._envelope import PulseEnvelopeField
    from ..optics.wave._fields import PlaneFieldSpace


_REVISION = OPENPMD_LASER_ENVELOPE_REVISION
_ASSUMPTION = f"LaserEnvelope draft pinned at {_REVISION.source}"
_RECORD_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def _axis_dimensions(labels: tuple[str, str, str], /) -> tuple[OpenPMDDimension, ...]:
    return tuple(TIME_DIMENSION if label == "t" else LENGTH_DIMENSION for label in labels)


@dataclass(frozen=True, slots=True)
class OpenPMDLaserEnvelopeProfile:
    """One exact record selection and normalized polarization for the pinned draft."""

    record_name: str = "laserEnvelope"
    iteration: int = 0
    axis_labels: tuple[str, str, str] = ("x", "y", "t")
    polarization: tuple[complex, complex] = (1.0 + 0.0j, 0.0 + 0.0j)
    profile_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = str(self.record_name).strip()
        if not name or _RECORD_NAME.fullmatch(name) is None:
            raise ValueError(
                "record_name must contain only letters, digits, and underscores."
            )
        if isinstance(self.iteration, bool) or not isinstance(self.iteration, Integral):
            raise TypeError("iteration must be an integer.")
        iteration = int(self.iteration)
        if iteration < 0:
            raise ValueError("iteration must be nonnegative.")
        labels = tuple(str(value).strip() for value in self.axis_labels)
        if len(labels) != 3 or set(labels) != {"x", "y", "t"}:
            raise ValueError("axis_labels must be one permutation of ('x', 'y', 't').")
        polarization = np.asarray(self.polarization, dtype=np.complex128)
        if polarization.shape != (2,) or np.any(~np.isfinite(polarization)):
            raise ValueError("polarization must be a finite complex two-vector.")
        norm = float(np.sqrt(np.sum(np.abs(polarization) ** 2)))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("polarization must be nonzero.")
        polarization = polarization / norm
        resolved = (complex(polarization[0]), complex(polarization[1]))
        object.__setattr__(self, "record_name", name)
        object.__setattr__(self, "iteration", iteration)
        object.__setattr__(self, "axis_labels", labels)
        object.__setattr__(self, "polarization", resolved)
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "openpmd-laser-envelope-profile",
                    "standard_commit": _REVISION.source,
                    "record_name": name,
                    "iteration": iteration,
                    "axis_labels": list(labels),
                    "polarization": [[value.real, value.imag] for value in resolved],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OpenPMDLaserEnvelopeImportPolicy:
    """Decoded-payload and metadata tolerances for one bounded HDF5 import."""

    profile: OpenPMDLaserEnvelopeProfile = field(
        default_factory=OpenPMDLaserEnvelopeProfile
    )
    maximum_decoded_bytes: int = 256 * 1024 * 1024
    metadata_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if not isinstance(self.profile, OpenPMDLaserEnvelopeProfile):
            raise TypeError("profile must be an OpenPMDLaserEnvelopeProfile.")
        if isinstance(self.maximum_decoded_bytes, bool) or not isinstance(
            self.maximum_decoded_bytes, Integral
        ):
            raise TypeError("maximum_decoded_bytes must be an integer.")
        maximum = int(self.maximum_decoded_bytes)
        tolerance = float(self.metadata_tolerance)
        if maximum <= 0:
            raise ValueError("maximum_decoded_bytes must be positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("metadata_tolerance must be finite and nonnegative.")
        object.__setattr__(self, "maximum_decoded_bytes", maximum)
        object.__setattr__(self, "metadata_tolerance", tolerance)


@dataclass(frozen=True, slots=True)
class OpenPMDLaserEnvelopeImportResult:
    field: PulseEnvelopeField
    resource: BoundedResource
    report: AdapterReport
    profile: OpenPMDLaserEnvelopeProfile


@dataclass(frozen=True, slots=True)
class OpenPMDLaserEnvelopeExportResult:
    path: Path
    resource: BoundedResource
    report: AdapterReport
    profile: OpenPMDLaserEnvelopeProfile


class OpenPMDLaserEnvelopeError(ValueError):
    """Rejected LaserEnvelope conversion with an auditable adapter report."""

    status: AdapterStatus
    report: AdapterReport

    def __init__(self, message: str, report: AdapterReport, /) -> None:
        self.status = report.status
        self.report = report
        super().__init__(str(message))


def _failure(
    status: AdapterStatus,
    source_id: str,
    message: str,
    /,
    *,
    target_format: str = "PulseEnvelopeField",
) -> OpenPMDLaserEnvelopeError:
    failure_id = canonical_fingerprint(
        {
            "kind": "openpmd-laser-envelope-failure",
            "source": source_id,
            "status": int(status),
            "message": str(message),
        }
    )
    report = AdapterReport(
        status,
        _REVISION.format_id,
        target_format,
        source_id=source_id,
        target_id=failure_id,
        assumptions=(_ASSUMPTION,),
    )
    return OpenPMDLaserEnvelopeError(message, report)


def _axis_labels(attributes: h5py.AttributeManager, /) -> tuple[str, str, str]:
    if "axisLabels" not in attributes:
        raise ValueError("Missing required openPMD attribute axisLabels.")
    values = np.asarray(attributes["axisLabels"])
    if values.shape != (3,):
        raise ValueError("axisLabels must contain exactly three labels.")
    labels = (
        attribute_text(values[0], "axisLabels"),
        attribute_text(values[1], "axisLabels"),
        attribute_text(values[2], "axisLabels"),
    )
    if set(labels) != {"x", "y", "t"}:
        raise OpenPMDUnsupportedError(
            "Only Cartesian temporal axes ('x', 'y', 't') are supported."
        )
    return labels


def _select_envelope(
    handle: h5py.File,
    inventory: OpenPMDHDF5Inventory,
    resource: BoundedResource,
    policy: OpenPMDLaserEnvelopeImportPolicy,
    /,
) -> h5py.Dataset:
    selected_path = f"data/{policy.profile.iteration}/meshes/{policy.profile.record_name}"
    if selected_path not in inventory.datasets:
        if selected_path in handle:
            raise OpenPMDUnsupportedError(
                "Vector/group LaserEnvelope records are unsupported."
            )
        raise ValueError("The selected LaserEnvelope record is absent.")
    selected = inventory.datasets[selected_path]
    # Reserve the detached canonical complex128 array in addition to HDF5's
    # fixed-width decoded buffer before reading any dataset payload.
    inventory.require_budget(
        resource.manifest.limits,
        policy.maximum_decoded_bytes,
        selected.size * np.dtype(np.complex128).itemsize,
    )
    return selected


def _validate_series_metadata(
    handle: h5py.File,
    dataset: h5py.Dataset,
    policy: OpenPMDLaserEnvelopeImportPolicy,
    /,
) -> tuple[
    tuple[str, str, str],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    OpenPMDUnit,
    float,
    np.ndarray,
]:
    root = read_series_root(handle, _REVISION)
    if root.meshes_path is None:
        raise ValueError("Missing required openPMD attribute meshesPath.")
    if root.meshes_path != "meshes/":
        raise OpenPMDUnsupportedError("Only the canonical meshesPath is supported.")
    if root.iteration_encoding != "groupBased":
        raise OpenPMDUnsupportedError("Only groupBased openPMD HDF5 is supported.")
    _, time = read_iteration(handle, policy.profile.iteration)
    if time.time != 0.0 or time.dt != 0.0:
        raise ValueError("The temporal-envelope profile requires zero iteration time/dt.")
    attributes = dataset.attrs
    geometry = required_text(attributes, "geometry")
    if geometry != "cartesian":
        raise OpenPMDUnsupportedError(
            "Only Cartesian LaserEnvelope geometry is supported."
        )
    if "geometryParameters" in attributes:
        parameters = attribute_text(
            attributes["geometryParameters"], "geometryParameters"
        )
        if parameters:
            raise ValueError("Cartesian LaserEnvelope geometry takes no parameters.")
    if required_text(attributes, "envelopeField") != "electric_field":
        raise OpenPMDUnsupportedError(
            "Normalized vector-potential envelopes are unsupported."
        )
    labels = _axis_labels(attributes)
    if labels != policy.profile.axis_labels:
        raise OpenPMDUnsupportedError(
            "LaserEnvelope axisLabels do not match the selected profile."
        )
    spacing = numeric_attribute(attributes, "gridSpacing", (3,))
    offset = numeric_attribute(attributes, "gridGlobalOffset", (3,))
    if np.any(spacing <= 0.0):
        raise ValueError("LaserEnvelope grid spacing must be positive.")
    grid_unit = read_grid_unit_si(
        attributes, _REVISION, _axis_dimensions(labels), policy.metadata_tolerance
    )
    position = numeric_attribute(attributes, "position", (3,))
    if np.any(position < 0.0) or np.any(position >= 1.0):
        raise ValueError("LaserEnvelope component positions must lie in [0, 1).")
    # The scalar envelope record is its own component.
    unit = read_record_unit(
        attributes, attributes, ELECTRIC_FIELD_DIMENSION, policy.metadata_tolerance
    )
    time_offset = scalar_attribute(attributes, "timeOffset")
    angular_frequency = scalar_attribute(attributes, "angularFrequency")
    if time_offset != 0.0 or angular_frequency <= 0.0:
        raise ValueError("LaserEnvelope time offset or carrier is invalid.")
    polarization = np.asarray(attributes.get("polarization"))
    if polarization.shape != (2,) or not np.issubdtype(
        polarization.dtype, np.complexfloating
    ):
        raise ValueError("LaserEnvelope polarization must be a complex two-vector.")
    polarization = np.asarray(polarization, dtype=np.complex128)
    norm = float(np.sqrt(np.sum(np.abs(polarization) ** 2)))
    if np.any(~np.isfinite(polarization)) or not np.isclose(
        norm, 1.0, rtol=0.0, atol=policy.metadata_tolerance
    ):
        raise ValueError("LaserEnvelope polarization must be finite and normalized.")
    expected_polarization = np.asarray(policy.profile.polarization)
    if not np.allclose(
        polarization,
        expected_polarization,
        rtol=0.0,
        atol=policy.metadata_tolerance,
    ):
        raise OpenPMDUnsupportedError(
            "LaserEnvelope polarization does not match the selected profile."
        )
    return (
        labels,
        spacing * grid_unit,
        offset * grid_unit,
        position,
        unit,
        angular_frequency,
        polarization,
    )


def _decode_openpmd_laser(
    resource: BoundedResource,
    policy: OpenPMDLaserEnvelopeImportPolicy,
    frame: RigidFrame,
    longitudinal_coordinate: float,
    /,
) -> OpenPMDLaserEnvelopeImportResult:
    from ..discretization._axis import TensorGridPlan, UniformAxisSpec
    from ..geometry.analytic._operations import RigidFrame
    from ..optics.wave._envelope import PulseEnvelopeField
    from ..optics.wave._fields import PlaneFieldSpace
    from ..optics.wave._pulse_time import PulseTimeSpace

    if not isinstance(frame, RigidFrame) or frame.dimension != 3:
        raise ValueError("frame must be a three-dimensional RigidFrame.")
    with h5py.File(BytesIO(resource.data), "r") as handle:
        inventory = preflight_hdf5(handle, resource.manifest.limits)
        dataset = _select_envelope(handle, inventory, resource, policy)
        if dataset.ndim != 3 or any(size < 2 for size in dataset.shape):
            raise ValueError(
                "LaserEnvelope payload must have three axes of at least two points."
            )
        if not np.issubdtype(dataset.dtype, np.complexfloating):
            raise TypeError("LaserEnvelope payload must use fixed-width complex storage.")
        dataset_shape = tuple(dataset.shape)
        metadata = _validate_series_metadata(handle, dataset, policy)
        labels, spacing, offset, position, unit, carrier, polarization = metadata
        # The complete HDF5 tree, selected metadata, dtype, shape, and decoded-byte
        # budget have been checked before this sole payload allocation.
        payload = np.array(dataset[()], dtype=np.complex128, copy=True)
    if np.any(~np.isfinite(payload)):
        raise ValueError("LaserEnvelope payload must contain only finite values.")
    canonical_order = tuple(labels.index(name) for name in ("x", "y", "t"))
    scalar = unit.to_si(np.transpose(payload, canonical_order))
    counts = {label: dataset_shape[index] for index, label in enumerate(labels)}
    coordinate0 = {
        label: offset[index] + position[index] * spacing[index]
        for index, label in enumerate(labels)
    }
    coordinate1 = {
        label: coordinate0[label] + (counts[label] - 1) * spacing[labels.index(label)]
        for label in labels
    }
    plane_grid = TensorGridPlan(
        (UniformAxisSpec(counts["x"]), UniformAxisSpec(counts["y"])),
        axis_names=("x", "y"),
    ).prepare(
        np.asarray(
            [
                [coordinate0["x"], coordinate0["y"]],
                [coordinate1["x"], coordinate1["y"]],
            ]
        )
    )
    time_grid = TensorGridPlan(
        (UniformAxisSpec(counts["t"]),), axis_names=("time",)
    ).prepare(np.asarray([[coordinate0["t"]], [coordinate1["t"]]]))
    plane_space = PlaneFieldSpace(plane_grid, frame, "finite-window")
    time_space = PulseTimeSpace(time_grid, topology="finite-window")
    values = scalar[..., None] * polarization
    field_value = PulseEnvelopeField(
        plane_space,
        time_space,
        values,
        carrier,
        longitudinal_coordinate,
        polarization="tangential",
    )
    target_id = canonical_fingerprint(
        {
            "kind": "openpmd-laser-envelope-import",
            "profile": policy.profile.profile_id,
            "plane_space": plane_space.space_id,
            "time_space": time_space.space_id,
            "carrier": carrier,
            "longitudinal_coordinate": float(field_value.longitudinal_coordinate),
            "values": array_tree_fingerprint(values),
        }
    )
    accounted = account_bounded_resource(
        resource,
        depth=inventory.maximum_depth,
        nodes=inventory.object_count + payload.size,
        attributes=inventory.attribute_count,
        losses=0,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        _REVISION.format_id,
        "PulseEnvelopeField",
        source_id=accounted.manifest.manifest_id,
        target_id=target_id,
        coordinate_mapping=(
            "openPMD Cartesian x,y,t axes -> plane-local x,y and PulseTimeSpace time",
            "gridSpacing/gridGlobalOffset/gridUnitSI/position -> SI point coordinates",
            "scalar envelope times normalized polarization -> tangential envelope",
        ),
        preserved_fields=(
            "complex electric-field envelope",
            "carrier angular frequency",
            "normalized complex polarization",
            "Cartesian temporal grid",
        ),
        assumptions=(
            _ASSUMPTION,
            "frame and longitudinal coordinate are caller-supplied embedding metadata",
        ),
    )
    return OpenPMDLaserEnvelopeImportResult(
        field_value, accounted, report, policy.profile
    )


def read_openpmd_laser_envelope_hdf5(
    resource: BoundedResource,
    policy: OpenPMDLaserEnvelopeImportPolicy,
    /,
    *,
    frame: RigidFrame,
    longitudinal_coordinate: float,
) -> OpenPMDLaserEnvelopeImportResult:
    """Read one bounded HDF5 image after complete resource and metadata preflight."""
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(policy, OpenPMDLaserEnvelopeImportPolicy):
        raise TypeError("policy must be an OpenPMDLaserEnvelopeImportPolicy.")
    try:
        return _decode_openpmd_laser(resource, policy, frame, longitudinal_coordinate)
    except OpenPMDLaserEnvelopeError:
        raise
    except ResourceReadError as error:
        raise _failure(
            AdapterStatus.INCONSISTENT_SOURCE,
            resource.manifest.manifest_id,
            str(error),
        ) from error
    except OpenPMDUnsupportedError as error:
        raise _failure(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            resource.manifest.manifest_id,
            str(error),
        ) from error
    except (OSError, KeyError, UnicodeError, TypeError, ValueError) as error:
        raise _failure(
            AdapterStatus.MALFORMED_SOURCE, resource.manifest.manifest_id, str(error)
        ) from error


def _finite_uniform_axis(
    space: PlaneFieldSpace, index: int, /
) -> tuple[np.ndarray, float]:
    axis = space.grid.axes[index]
    if (
        axis.basis != "uniform"
        or axis.periodic
        or axis.primary_entity != "point"
        or not axis.lower_endpoint_included
        or not axis.upper_endpoint_included
    ):
        raise ValueError("openPMD export requires finite uniform point axes.")
    nodes = np.asarray(axis.nodes, dtype=np.float64)
    if nodes.size < 2:
        raise ValueError("openPMD export axes require at least two points.")
    differences = np.diff(nodes)
    spacing = float(differences[0])
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.allclose(differences, spacing, rtol=1.0e-10, atol=0.0)
    ):
        raise ValueError("Nonuniform axes are unsupported by this openPMD profile.")
    return nodes, spacing


def _scalar_envelope_for_export(
    field_value: PulseEnvelopeField,
    profile: OpenPMDLaserEnvelopeProfile,
    tolerance: float,
    /,
) -> np.ndarray:
    values = np.asarray(field_value.values)
    if np.any(~np.isfinite(values)):
        raise ValueError("Pulse envelope values must be finite.")
    if field_value.polarization == "scalar":
        return np.asarray(values, dtype=np.complex128)
    polarization = np.asarray(profile.polarization, dtype=np.complex128)
    scalar = np.sum(np.conj(polarization) * values, axis=-1)
    residual = values - scalar[..., None] * polarization
    denominator = max(float(np.linalg.norm(values)), np.finfo(np.float64).tiny)
    relative = float(np.linalg.norm(residual) / denominator)
    if not np.isfinite(relative) or relative > tolerance:
        raise ValueError(
            "Tangential envelope is not factorizable by the profile polarization."
        )
    return np.asarray(scalar, dtype=np.complex128)


def _write_payload(
    stream: BinaryIO,
    payload: np.ndarray,
    field_value: PulseEnvelopeField,
    profile: OpenPMDLaserEnvelopeProfile,
    spacings: dict[str, float],
    offsets: dict[str, float],
    /,
) -> None:
    with h5py.File(stream, "w") as handle:
        write_series_root(handle, _REVISION, meshes_path="meshes/", particles_path=None)
        iteration = write_iteration(
            handle, profile.iteration, OpenPMDIterationTime(0.0, 0.0, 1.0)
        )
        meshes = iteration.require_group("meshes")
        order = tuple(("x", "y", "t").index(label) for label in profile.axis_labels)
        stored = np.transpose(payload, order)
        dataset = meshes.create_dataset(profile.record_name, data=stored)
        labels = profile.axis_labels
        dataset.attrs["geometry"] = np.bytes_("cartesian")
        dataset.attrs["axisLabels"] = np.asarray(labels, dtype="S1")
        dataset.attrs["gridSpacing"] = np.asarray(
            [spacings[label] for label in labels], dtype=np.float64
        )
        dataset.attrs["gridGlobalOffset"] = np.asarray(
            [offsets[label] for label in labels], dtype=np.float64
        )
        write_grid_units(dataset.attrs, _REVISION, _axis_dimensions(labels))
        dataset.attrs["position"] = np.zeros(3, dtype=np.float64)
        write_record_unit(
            dataset.attrs, dataset.attrs, OpenPMDUnit(1.0, ELECTRIC_FIELD_DIMENSION)
        )
        dataset.attrs["timeOffset"] = np.float64(0.0)
        dataset.attrs["envelopeField"] = np.bytes_("electric_field")
        dataset.attrs["angularFrequency"] = np.float64(
            field_value.carrier_angular_frequency
        )
        dataset.attrs["polarization"] = np.asarray(
            profile.polarization, dtype=np.complex128
        )


def write_openpmd_laser_envelope_hdf5(
    path: str | Path,
    field_value: PulseEnvelopeField,
    profile: OpenPMDLaserEnvelopeProfile,
    /,
    *,
    limits: ResourceLimits,
    factorization_tolerance: float = 1.0e-6,
) -> OpenPMDLaserEnvelopeExportResult:
    """Encode and exclusively publish one bounded group-based HDF5 iteration."""
    from ..optics.wave._envelope import PulseEnvelopeField

    if not isinstance(field_value, PulseEnvelopeField):
        raise TypeError("field_value must be a PulseEnvelopeField.")
    if not isinstance(profile, OpenPMDLaserEnvelopeProfile):
        raise TypeError("profile must be an OpenPMDLaserEnvelopeProfile.")
    if not isinstance(limits, ResourceLimits):
        raise TypeError("limits must be ResourceLimits.")
    tolerance = float(factorization_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("factorization_tolerance must be finite and nonnegative.")
    if (
        field_value.plane_space.topology != "finite-window"
        or field_value.time_space.topology != "finite-window"
    ):
        raise ValueError(
            "This openPMD profile exports only finite-window envelope grids."
        )
    frame = field_value.plane_space.frame
    if not np.array_equal(np.asarray(frame.rotation), np.eye(3)) or not np.array_equal(
        np.asarray(frame.translation), np.zeros(3)
    ):
        raise ValueError("This openPMD profile cannot encode a nonidentity plane frame.")
    if float(field_value.longitudinal_coordinate) != 0.0:
        raise ValueError(
            "This openPMD profile cannot encode a nonzero longitudinal coordinate."
        )
    x_nodes, dx = _finite_uniform_axis(field_value.plane_space, 0)
    y_nodes, dy = _finite_uniform_axis(field_value.plane_space, 1)
    time_axis = field_value.time_space.temporal_grid.axes[0]
    if (
        time_axis.basis != "uniform"
        or time_axis.periodic
        or time_axis.primary_entity != "point"
    ):
        raise ValueError("openPMD export requires finite uniform point pulse time.")
    t_nodes = np.asarray(time_axis.nodes, dtype=np.float64)
    if t_nodes.size < 2:
        raise ValueError("openPMD export time requires at least two points.")
    dt = float(t_nodes[1] - t_nodes[0])
    if not np.allclose(np.diff(t_nodes), dt, rtol=1.0e-10, atol=0.0):
        raise ValueError("Nonuniform pulse time is unsupported by this openPMD profile.")
    scalar = _scalar_envelope_for_export(field_value, profile, tolerance)
    if scalar.nbytes > limits.max_bytes or scalar.size > limits.max_nodes:
        raise ResourceReadError("limit", "LaserEnvelope payload exceeds resource limits.")
    buffer = BoundedHDF5Buffer(limits.max_bytes)
    _write_payload(
        buffer,
        scalar,
        field_value,
        profile,
        {"x": dx, "y": dy, "t": dt},
        {"x": float(x_nodes[0]), "y": float(y_nodes[0]), "t": float(t_nodes[0])},
    )
    data = buffer.getvalue()
    destination = Path(path)
    resource = bounded_resource_from_bytes(
        data, limits=limits, source_path=str(destination)
    )
    publish_bytes(
        destination,
        data,
        maximum_bytes=limits.max_bytes,
        mode="exclusive",
    )
    source_id = canonical_fingerprint(
        {
            "kind": "pulse-envelope-openpmd-export",
            "plane_space": field_value.plane_space.space_id,
            "time_space": field_value.time_space.space_id,
            "carrier": float(field_value.carrier_angular_frequency),
            "values": array_tree_fingerprint(field_value.values),
            "longitudinal_coordinate": float(field_value.longitudinal_coordinate),
            "polarization": field_value.polarization,
            "profile": profile.profile_id,
        }
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        "PulseEnvelopeField",
        _REVISION.format_id,
        source_id=source_id,
        target_id=resource.manifest.manifest_id,
        coordinate_mapping=(
            "plane-local x,y and PulseTimeSpace time -> openPMD Cartesian x,y,t axes",
            "SI point coordinates -> gridSpacing/gridGlobalOffset with unitSI=1",
            "tangential envelope -> scalar envelope times normalized polarization",
        ),
        preserved_fields=(
            "complex electric-field envelope",
            "carrier angular frequency",
            "normalized complex polarization",
            "Cartesian temporal grid",
            "canonical identity frame and zero longitudinal coordinate",
        ),
        assumptions=(
            _ASSUMPTION,
            "canonical identity frame and zero longitudinal coordinate are implicit",
        ),
    )
    return OpenPMDLaserEnvelopeExportResult(destination, resource, report, profile)


__all__ = [
    "OpenPMDLaserEnvelopeError",
    "OpenPMDLaserEnvelopeExportResult",
    "OpenPMDLaserEnvelopeImportPolicy",
    "OpenPMDLaserEnvelopeImportResult",
    "OpenPMDLaserEnvelopeProfile",
    "read_openpmd_laser_envelope_hdf5",
    "write_openpmd_laser_envelope_hdf5",
]
