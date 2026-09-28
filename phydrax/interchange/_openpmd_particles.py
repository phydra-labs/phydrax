#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded HDF5 adapter for openPMD 1.1.0 particle-species tracks.

One species is followed through an increasing sequence of group-based
iterations. Particles are matched across iterations by their openPMD ``id``
alone and emitted as ``ChargedTrajectory`` lanes in ascending identity order,
so the result is independent of the storage order inside every iteration.
Record values are converted through the record ``unitSI`` to SI and from SI to
the units of the bound ``ElectromagneticScaleContract``; each record's
``unitDimension`` must equal the scale's openPMD dimension of its quantity.
Per-particle quantities follow the ED-PIC ``macroWeighted``/``weightingPower``
convention, which this profile requires on every record it reads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
from jax.typing import ArrayLike

from .._external_resource import (
    account_bounded_resource,
    bounded_resource_from_bytes,
    BoundedResource,
    ResourceLimits,
    ResourceReadError,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._physical import ElectromagneticScaleContract
from .._publication import publish_bytes
from ._openpmd_base import (
    BoundedHDF5Buffer,
    component_shape,
    component_values,
    identity_values,
    OPENPMD_PARTICLE_REVISION,
    OpenPMDDimension,
    OpenPMDHDF5Inventory,
    OpenPMDIterationTime,
    OpenPMDSeriesRoot,
    OpenPMDUnit,
    OpenPMDUnsupportedError,
    preflight_hdf5,
    read_iteration,
    read_record_unit,
    read_series_root,
    scalar_attribute,
    weighting_metadata,
    write_component,
    write_iteration,
    write_particle_metadata,
    write_record_unit,
    write_series_root,
)
from ._report import AdapterLoss, AdapterReport, AdapterStatus


if TYPE_CHECKING:
    from ..electromagnetics._trajectory_radiation import ChargedTrajectory


_REVISION = OPENPMD_PARTICLE_REVISION
_ASSUMPTIONS = (
    "openPMD 1.1.0 group-based particle records",
    "ED-PIC macroWeighted/weightingPower declared on every read record",
    "particles are matched across iterations by openPMD id only",
)
_NAME = re.compile(r"^\w+$", re.ASCII)
_AXES = ("x", "y", "z")
_DIMENSIONLESS: OpenPMDDimension = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_PARTICLES_PATH = "particles/"
# Scale quantity of every read record; ``None`` marks a dimensionless record.
_RECORD_QUANTITIES: dict[str, str | None] = {
    "position": "length",
    "positionOffset": "length",
    "momentum": "momentum",
    "weighting": None,
    "charge": "charge",
    "mass": "mass",
    "id": None,
}
_VECTOR_RECORDS = ("position", "positionOffset", "momentum")
# Host scratch per stored particle of one iteration: nine vector components,
# weighting, charge, mass, identity, sort order, and two lane-index words.
_SCRATCH_WORDS = 16
# Canonical output per lane sample: positions and proper velocities (six
# float64) plus the activity flag; per lane: charge, multiplicity, mass (three
# float64) and two uint32 identity words.
_SAMPLE_BYTES = 6 * 8 + 1
_LANE_BYTES = 3 * 8 + 2 * 4


class OpenPMDParticleTrackError(ValueError):
    """Rejected particle-track conversion with an auditable adapter report."""

    status: AdapterStatus
    report: AdapterReport

    def __init__(self, message: str, report: AdapterReport, /) -> None:
        self.status = report.status
        self.report = report
        super().__init__(str(message))


class _TrackInconsistencyError(ValueError):
    """Well-formed records whose particle tracks contradict each other."""


def _record_name(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    text = value.strip()
    if _NAME.fullmatch(text) is None:
        raise ValueError(f"{name} must contain only letters, digits, and underscores.")
    return text


def _iteration_indices(values: object, name: str, /) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of integers.")
    indices: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{name} must contain integers.")
        index = int(value)
        if index < 0 or index >= 2**64:
            raise ValueError(f"{name} must contain unsigned 64-bit integers.")
        indices.append(index)
    if not indices:
        raise ValueError(f"{name} must be nonempty.")
    if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
        raise ValueError(f"{name} must be strictly increasing.")
    return tuple(indices)


def _identity_values(values: object, /) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("identities must be a sequence of integers.")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in values
    ):
        raise TypeError("identities must contain integers.")
    identities = sorted(int(value) for value in values)
    if not identities:
        raise ValueError("identities must be nonempty.")
    if identities[0] < 0 or identities[-1] >= 2**64:
        raise ValueError("identities must be unsigned 64-bit integers.")
    if any(later == earlier for earlier, later in zip(identities, identities[1:])):
        raise ValueError("identities must be unique.")
    return tuple(identities)


@dataclass(frozen=True, slots=True)
class OpenPMDParticleTrackSelection:
    """One species, iteration range, and optional identity subset to follow.

    ``iterations=None`` follows every iteration of the series in increasing
    order; explicit iterations must be strictly increasing. ``identities=None``
    follows every particle and requires the same identity set in every selected
    iteration; explicit ``uint64`` identities must each occur in every selected
    iteration. Lanes are always ordered by ascending identity.
    """

    species: str
    iterations: Sequence[int] | None = None
    identities: Sequence[int] | None = None
    selection_id: str = field(init=False)

    def __post_init__(self) -> None:
        species = _record_name(self.species, "species")
        iterations = (
            None
            if self.iterations is None
            else _iteration_indices(self.iterations, "iterations")
        )
        identities = (
            None if self.identities is None else _identity_values(self.identities)
        )
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "identities", identities)
        object.__setattr__(
            self,
            "selection_id",
            canonical_fingerprint(
                {
                    "kind": "openpmd-particle-track-selection",
                    "standard": _REVISION.version,
                    "species": species,
                    "iterations": None if iterations is None else list(iterations),
                    "identities": None if identities is None else list(identities),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class OpenPMDParticleTrackImportPolicy:
    """Decoded-payload budget and metadata tolerance of one bounded import."""

    selection: OpenPMDParticleTrackSelection
    maximum_decoded_bytes: int = 256 * 1024 * 1024
    metadata_tolerance: float = 1.0e-10

    def __post_init__(self) -> None:
        if not isinstance(self.selection, OpenPMDParticleTrackSelection):
            raise TypeError("selection must be an OpenPMDParticleTrackSelection.")
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
class OpenPMDParticleTrackImportResult:
    """Imported lanes in scale units with the per-lane rest masses.

    ``masses[P]`` are single-particle rest masses in the scale's mass unit;
    ``ChargedTrajectory`` carries charges but not masses.
    """

    trajectory: ChargedTrajectory
    masses: np.ndarray
    iterations: tuple[int, ...]
    resource: BoundedResource
    report: AdapterReport
    selection: OpenPMDParticleTrackSelection


@dataclass(frozen=True, slots=True)
class OpenPMDParticleTrackExportResult:
    path: Path
    iterations: tuple[int, ...]
    resource: BoundedResource
    report: AdapterReport


# Structural scan ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Component:
    item: h5py.Dataset | h5py.Group
    unit: OpenPMDUnit
    count: int


@dataclass(frozen=True, slots=True)
class _Record:
    components: tuple[_Component, ...]
    macro_weighted: bool
    weighting_power: float
    time_offset: float


@dataclass(frozen=True, slots=True)
class _SpeciesIteration:
    iteration: int
    time: OpenPMDIterationTime
    count: int
    records: dict[str, _Record]


def _component_count(item: h5py.Dataset | h5py.Group, path: str, /) -> int:
    """Return the particle count of one dataset or constant record component."""
    shape = component_shape(item, path)
    if len(shape) != 1:
        raise ValueError(f"Particle record component {path} must be one-dimensional.")
    return shape[0]


def _scan_record(
    species: h5py.Group,
    name: str,
    dimension: OpenPMDDimension,
    tolerance: float,
    /,
) -> _Record:
    if name not in species:
        if name == "id":
            raise OpenPMDUnsupportedError(
                "Particle identities are missing; tracks require the id record."
            )
        raise ValueError(f"Missing required particle record {name}.")
    record = species[name]
    path = f"{species.name}/{name}"
    items: tuple[tuple[h5py.Dataset | h5py.Group, str], ...]
    if name in _VECTOR_RECORDS:
        if not isinstance(record, h5py.Group) or "value" in record.attrs:
            raise ValueError(f"Particle record {name} must be a vector record group.")
        missing = [axis for axis in _AXES if axis not in record]
        if missing:
            raise OpenPMDUnsupportedError(
                f"Particle record {name} lacks Cartesian components {missing}; "
                "only three-dimensional positions and momenta are supported."
            )
        items = tuple((record[axis], f"{path}/{axis}") for axis in _AXES)
    else:
        items = ((record, path),)
    components = tuple(
        _Component(
            item,
            read_record_unit(record.attrs, item.attrs, dimension, tolerance),
            _component_count(item, item_path),
        )
        for item, item_path in items
    )
    macro, power = weighting_metadata(record.attrs, path)
    return _Record(components, macro, power, scalar_attribute(record.attrs, "timeOffset"))


def _scan_iteration(
    handle: h5py.File,
    root: OpenPMDSeriesRoot,
    iteration: int,
    species_name: str,
    dimensions: dict[str, OpenPMDDimension],
    tolerance: float,
    /,
) -> _SpeciesIteration:
    group, time = read_iteration(handle, iteration)
    path = f"{root.particles_path}{species_name}"
    if path not in group or not isinstance(group[path], h5py.Group):
        raise ValueError(
            f"Species {species_name} is absent from openPMD iteration {iteration}."
        )
    species = group[path]
    records = {
        name: _scan_record(species, name, dimensions[name], tolerance)
        for name in _RECORD_QUANTITIES
    }
    counts = {
        component.count for record in records.values() for component in record.components
    }
    if len(counts) != 1:
        raise ValueError(
            f"Particle records of iteration {iteration} are truncated: component "
            f"extents {sorted(counts)} disagree."
        )
    (count,) = counts
    if count == 0:
        raise ValueError(f"Species {species_name} is empty at iteration {iteration}.")
    identity = records["id"].components[0]
    if identity.unit.unit_si != 1.0:
        raise ValueError("Particle id records must carry unitSI = 1.")
    offsets = np.asarray(
        [records[name].time_offset for name in _VECTOR_RECORDS], dtype=np.float64
    )
    if not np.allclose(offsets, offsets[0], rtol=tolerance, atol=0.0):
        raise OpenPMDUnsupportedError(
            "Staggered position and momentum timeOffset values are unsupported."
        )
    return _SpeciesIteration(iteration, time, count, records)


def _selected_iterations(
    handle: h5py.File, selection: OpenPMDParticleTrackSelection, /
) -> tuple[int, ...]:
    if selection.iterations is not None:
        return tuple(selection.iterations)
    if "data" not in handle or not isinstance(handle["data"], h5py.Group):
        raise ValueError("The openPMD series holds no iterations.")
    names = tuple(handle["data"])
    if not all(name.isdecimal() for name in names):
        raise ValueError("openPMD iteration group names must be unsigned integers.")
    return _iteration_indices(sorted(int(name) for name in names), "iterations")


# Payload decoding -----------------------------------------------------------------


def _component_values(component: _Component, /) -> np.ndarray:
    return component_values(component.item, (component.count,))


def _record_si(record: _Record, weighting: np.ndarray, /) -> np.ndarray:
    """Per-particle SI values ``[N]`` (scalar) or ``[N, 3]`` (vector)."""
    values = np.stack(
        [
            component.unit.to_si(_component_values(component))
            for component in record.components
        ],
        axis=-1,
    )
    if record.macro_weighted:
        values = values / (weighting**record.weighting_power)[:, None]
    return values if values.shape[1] == 3 else values[:, 0]


def _identities(component: _Component, /) -> np.ndarray:
    return identity_values(component.item, (component.count,))


@dataclass(frozen=True, slots=True)
class _DecodedIteration:
    positions: np.ndarray
    momenta: np.ndarray
    charges: np.ndarray
    masses: np.ndarray
    weights: np.ndarray


def _decode_iteration(
    scanned: _SpeciesIteration, lanes: np.ndarray | None, complete: bool, /
) -> tuple[np.ndarray, _DecodedIteration]:
    """Decode one iteration and gather it onto ascending-identity lanes.

    ``lanes=None`` adopts the complete identity set of this iteration. With
    ``complete`` the iteration must hold exactly ``lanes``; otherwise it must
    hold at least ``lanes``.
    """
    records = scanned.records
    identities = _identities(records["id"].components[0])
    order = np.argsort(identities, kind="stable")
    stored = identities[order]
    if np.any(stored[1:] == stored[:-1]):
        raise _TrackInconsistencyError(
            f"Particle ids repeat within iteration {scanned.iteration}."
        )
    selected = stored if lanes is None else lanes
    slot = np.minimum(np.searchsorted(stored, selected), stored.shape[0] - 1)
    present = stored[slot] == selected
    if not np.all(present):
        raise _TrackInconsistencyError(
            f"Particle ids {selected[~present][:8].tolist()} are missing from "
            f"iteration {scanned.iteration}."
        )
    if complete and stored.shape[0] != selected.shape[0]:
        extra = np.setdiff1d(stored, selected, assume_unique=True)
        raise _TrackInconsistencyError(
            f"Particle ids {extra[:8].tolist()} of iteration {scanned.iteration} "
            "are missing from other selected iterations."
        )
    index = order[slot]
    weights = _record_si(records["weighting"], np.ones(scanned.count))
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("Particle weighting must be finite and positive.")
    positions = _record_si(records["position"], weights) + _record_si(
        records["positionOffset"], weights
    )
    decoded = _DecodedIteration(
        positions[index],
        _record_si(records["momentum"], weights)[index],
        _record_si(records["charge"], weights)[index],
        _record_si(records["mass"], weights)[index],
        weights[index],
    )
    return selected, decoded


@dataclass(frozen=True, slots=True)
class _Tracks:
    iterations: tuple[int, ...]
    identities: np.ndarray
    times: np.ndarray
    positions: np.ndarray
    proper_velocities: np.ndarray
    charges: np.ndarray
    masses: np.ndarray
    multiplicities: np.ndarray


def _scale_dimension(values: tuple[float, ...], /) -> OpenPMDDimension:
    if len(values) != 7:
        raise ValueError("Scale unit dimensions must hold seven openPMD powers.")
    return (
        values[0],
        values[1],
        values[2],
        values[3],
        values[4],
        values[5],
        values[6],
    )


def _record_dimensions(
    units: dict[str, tuple[float, tuple[float, ...]]], /
) -> dict[str, OpenPMDDimension]:
    return {
        name: _DIMENSIONLESS if quantity is None else _scale_dimension(units[quantity][1])
        for name, quantity in _RECORD_QUANTITIES.items()
    }


def _sample_times(
    scanned: tuple[_SpeciesIteration, ...], time_unit_si: float, /
) -> np.ndarray:
    """Record times ``(time + timeOffset) timeUnitSI`` in the scale time unit."""
    times = np.asarray(
        [
            (item.time.time + item.records["position"].time_offset)
            * item.time.time_unit_si
            for item in scanned
        ],
        dtype=np.float64,
    ) / np.float64(time_unit_si)
    if np.any(~np.isfinite(times)):
        raise ValueError("Particle record times must be finite.")
    if np.any(np.diff(times) <= 0.0):
        raise _TrackInconsistencyError(
            "Particle record times are nonmonotonic across the selected iterations."
        )
    return times


def _lane_bound(
    scanned: tuple[_SpeciesIteration, ...], selection: OpenPMDParticleTrackSelection, /
) -> int:
    if selection.identities is not None:
        return len(selection.identities)
    counts = {item.count for item in scanned}
    if len(counts) != 1:
        raise _TrackInconsistencyError(
            f"Particle counts {sorted(counts)} differ across the selected "
            "iterations, so some particle ids are missing."
        )
    return scanned[0].count


def _require_invariant(
    name: str,
    reference: np.ndarray,
    values: np.ndarray,
    tolerance: float,
    iteration: int,
    /,
) -> None:
    if not np.allclose(values, reference, rtol=tolerance, atol=0.0):
        raise _TrackInconsistencyError(
            f"Per-particle {name} changes at iteration {iteration}; a track carries "
            "one charge, mass, and weighting."
        )


def _read_tracks(
    handle: h5py.File,
    inventory: OpenPMDHDF5Inventory,
    resource: BoundedResource,
    policy: OpenPMDParticleTrackImportPolicy,
    units: dict[str, tuple[float, tuple[float, ...]]],
    /,
) -> _Tracks:
    selection = policy.selection
    tolerance = policy.metadata_tolerance
    root = read_series_root(handle, _REVISION)
    if root.particles_path is None:
        raise ValueError("Missing required openPMD attribute particlesPath.")
    if root.iteration_encoding != "groupBased":
        raise OpenPMDUnsupportedError("Only groupBased openPMD HDF5 is supported.")
    iterations = _selected_iterations(handle, selection)
    dimensions = _record_dimensions(units)
    scanned = tuple(
        _scan_iteration(handle, root, iteration, selection.species, dimensions, tolerance)
        for iteration in iterations
    )
    times = _sample_times(scanned, units["time"][0])
    lanes = _lane_bound(scanned, selection)
    largest = max(item.count for item in scanned)
    # Reserve the canonical lanes and one iteration's host scratch in addition
    # to every stored payload before any dataset payload is read.
    inventory.require_budget(
        resource.manifest.limits,
        policy.maximum_decoded_bytes,
        len(scanned) * (lanes * _SAMPLE_BYTES + 8)
        + lanes * _LANE_BYTES
        + largest * _SCRATCH_WORDS * 8,
    )
    identities = (
        None
        if selection.identities is None
        else np.asarray(selection.identities, dtype=np.uint64)
    )
    positions = np.empty((len(scanned), lanes, 3), dtype=np.float64)
    velocities = np.empty((len(scanned), lanes, 3), dtype=np.float64)
    reference: _DecodedIteration | None = None
    for sample, item in enumerate(scanned):
        identities, decoded = _decode_iteration(
            item, identities, selection.identities is None
        )
        if reference is None:
            if np.any(~np.isfinite(decoded.masses)) or np.any(decoded.masses <= 0.0):
                raise ValueError("Particle masses must be finite and positive.")
            if np.any(~np.isfinite(decoded.charges)):
                raise ValueError("Particle charges must be finite.")
            reference = decoded
        else:
            _require_invariant(
                "charge", reference.charges, decoded.charges, tolerance, item.iteration
            )
            _require_invariant(
                "mass", reference.masses, decoded.masses, tolerance, item.iteration
            )
            _require_invariant(
                "weighting", reference.weights, decoded.weights, tolerance, item.iteration
            )
        if np.any(~np.isfinite(decoded.positions)) or np.any(
            ~np.isfinite(decoded.momenta)
        ):
            raise ValueError(
                f"Particle positions and momenta of iteration {item.iteration} "
                "must be finite."
            )
        positions[sample] = decoded.positions
        velocities[sample] = decoded.momenta / reference.masses[:, None]
    if reference is None or identities is None:
        raise RuntimeError("A nonempty iteration selection decoded no particles.")
    return _Tracks(
        iterations,
        identities,
        times,
        positions / np.float64(units["length"][0]),
        velocities / np.float64(units["velocity"][0]),
        reference.charges / np.float64(units["charge"][0]),
        reference.masses / np.float64(units["mass"][0]),
        reference.weights,
    )


def _failure(
    status: AdapterStatus,
    source_id: str,
    message: str,
    /,
) -> OpenPMDParticleTrackError:
    failure_id = canonical_fingerprint(
        {
            "kind": "openpmd-particle-track-failure",
            "source": source_id,
            "status": int(status),
            "message": str(message),
        }
    )
    report = AdapterReport(
        status,
        _REVISION.format_id,
        "ChargedTrajectory",
        source_id=source_id,
        target_id=failure_id,
        assumptions=_ASSUMPTIONS,
    )
    return OpenPMDParticleTrackError(message, report)


def _identity_words(identities: np.ndarray, /) -> tuple[np.ndarray, np.ndarray]:
    return (
        (identities >> np.uint64(32)).astype(np.uint32),
        (identities & np.uint64(0xFFFFFFFF)).astype(np.uint32),
    )


def _decode_openpmd_tracks(
    resource: BoundedResource,
    policy: OpenPMDParticleTrackImportPolicy,
    scale: ElectromagneticScaleContract,
    units: dict[str, tuple[float, tuple[float, ...]]],
    /,
) -> OpenPMDParticleTrackImportResult:
    from ..electromagnetics._trajectory_radiation import ChargedTrajectory

    with h5py.File(BytesIO(resource.data), "r") as handle:
        inventory = preflight_hdf5(handle, resource.manifest.limits)
        tracks = _read_tracks(handle, inventory, resource, policy, units)
    samples, lanes = tracks.positions.shape[:2]
    trajectory = ChargedTrajectory(
        tracks.times,
        tracks.positions,
        tracks.proper_velocities,
        tracks.charges,
        tracks.multiplicities,
        np.ones((samples, lanes), dtype=np.bool_),
        _identity_words(tracks.identities),
    )
    target_id = canonical_fingerprint(
        {
            "kind": "openpmd-particle-track-import",
            "selection": policy.selection.selection_id,
            "scale": scale.scale_id,
            "iterations": list(tracks.iterations),
            "tracks": array_tree_fingerprint(
                (
                    tracks.identities,
                    tracks.times,
                    tracks.positions,
                    tracks.proper_velocities,
                    tracks.charges,
                    tracks.masses,
                    tracks.multiplicities,
                )
            ),
        }
    )
    accounted = account_bounded_resource(
        resource,
        depth=inventory.maximum_depth,
        nodes=inventory.object_count + samples * lanes * 6 + lanes * 4,
        attributes=inventory.attribute_count,
        losses=0,
    )
    report = AdapterReport(
        AdapterStatus.LOSSLESS,
        _REVISION.format_id,
        "ChargedTrajectory",
        source_id=accounted.manifest.manifest_id,
        target_id=target_id,
        coordinate_mapping=(
            "openPMD iteration (time + record timeOffset) * timeUnitSI -> scale time",
            "position + positionOffset, each times unitSI -> scale length",
            "momentum / mass (per particle via macroWeighted, weightingPower) -> "
            "proper velocity in scale units",
            "openPMD uint64 id -> (id >> 32, id & 0xffffffff) lanes in ascending order",
        ),
        preserved_fields=(
            "particle identity",
            "position",
            "momentum",
            "charge",
            "rest mass",
            "weighting as lane multiplicity",
            "record times",
        ),
        assumptions=(
            *_ASSUMPTIONS,
            "species records other than id, position, positionOffset, momentum, "
            "weighting, charge, and mass are not read",
        ),
    )
    return OpenPMDParticleTrackImportResult(
        trajectory,
        tracks.masses,
        tracks.iterations,
        accounted,
        report,
        policy.selection,
    )


def read_openpmd_particle_tracks_hdf5(
    resource: BoundedResource,
    policy: OpenPMDParticleTrackImportPolicy,
    /,
    *,
    scale: ElectromagneticScaleContract,
) -> OpenPMDParticleTrackImportResult:
    """Read species tracks from one bounded group-based openPMD HDF5 image.

    The complete HDF5 tree, every selected record's structure and units, the
    record times, and the decoded-byte budget are validated before any
    particle payload is read. Refusals raise ``OpenPMDParticleTrackError``
    carrying an invalid ``AdapterReport``: missing ids, repeated ids, and
    nonmonotonic times are ``INCONSISTENT_SOURCE``; truncated records and
    unit-dimension mismatches are ``MALFORMED_SOURCE``; resource overflow is
    ``INCONSISTENT_SOURCE`` with a ``maximum_decoded_bytes`` or limit message.
    """
    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if not isinstance(policy, OpenPMDParticleTrackImportPolicy):
        raise TypeError("policy must be an OpenPMDParticleTrackImportPolicy.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    units = scale.unit_si_map()
    source_id = resource.manifest.manifest_id
    try:
        return _decode_openpmd_tracks(resource, policy, scale, units)
    except ResourceReadError as error:
        raise _failure(
            AdapterStatus.INCONSISTENT_SOURCE, source_id, str(error)
        ) from error
    except _TrackInconsistencyError as error:
        raise _failure(
            AdapterStatus.INCONSISTENT_SOURCE, source_id, str(error)
        ) from error
    except OpenPMDUnsupportedError as error:
        raise _failure(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC, source_id, str(error)
        ) from error
    except (OSError, KeyError, UnicodeError, TypeError, ValueError) as error:
        raise _failure(AdapterStatus.MALFORMED_SOURCE, source_id, str(error)) from error


# Export -----------------------------------------------------------------------------


def _write_species(
    species: h5py.Group,
    identities: np.ndarray,
    positions: np.ndarray,
    momenta: np.ndarray,
    scalars: dict[str, tuple[np.ndarray, OpenPMDUnit, float]],
    length: OpenPMDUnit,
    momentum: OpenPMDUnit,
    /,
) -> None:
    for name, values, unit, power in (
        ("position", positions, length, 0.0),
        ("positionOffset", np.zeros_like(positions), length, 0.0),
        ("momentum", momenta, momentum, 1.0),
    ):
        record = species.create_group(name)
        for axis, axis_name in enumerate(_AXES):
            component = write_component(record, axis_name, values[:, axis])
            write_record_unit(record.attrs, component.attrs, unit)
        write_particle_metadata(record.attrs, 0, power, 0.0)
    identity = species.create_dataset("id", data=identities)
    write_record_unit(identity.attrs, identity.attrs, OpenPMDUnit(1.0, _DIMENSIONLESS))
    write_particle_metadata(identity.attrs, 0, 0.0, 0.0)
    for name, (values, unit, power) in scalars.items():
        # A scalar record is its own component.
        component = write_component(species, name, values)
        write_record_unit(component.attrs, component.attrs, unit)
        write_particle_metadata(
            component.attrs, 1 if name == "weighting" else 0, power, 0.0
        )


def _export_arrays(
    trajectory: ChargedTrajectory, masses: ArrayLike, /
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    times = np.asarray(trajectory.times, dtype=np.float64)
    if np.any(times != times[:, :1]):
        raise ValueError(
            "openPMD iterations share one time; per-lane times must be identical."
        )
    shared = times[:, 0]
    if np.any(~np.isfinite(shared)) or np.any(np.diff(shared) <= 0.0):
        raise ValueError("Trajectory times must be finite and strictly increasing.")
    if not np.all(np.asarray(trajectory.active)):
        raise ValueError(
            "Inactive samples cannot be exported: tracks require every particle "
            "in every iteration."
        )
    identities = (
        np.asarray(trajectory.id_hi, dtype=np.uint64) << np.uint64(32)
    ) | np.asarray(trajectory.id_lo, dtype=np.uint64)
    if np.unique(identities).shape[0] != identities.shape[0]:
        raise ValueError("Trajectory lane identities must be unique.")
    mass = np.asarray(masses, dtype=np.float64)
    if mass.shape != (trajectory.particle_count,):
        raise ValueError("masses must hold one rest mass per trajectory lane.")
    if np.any(~np.isfinite(mass)) or np.any(mass <= 0.0):
        raise ValueError("masses must be finite and positive.")
    return (
        shared,
        identities,
        np.asarray(trajectory.positions, dtype=np.float64),
        np.asarray(trajectory.proper_velocities, dtype=np.float64) * mass[None, :, None],
        np.asarray(trajectory.charges, dtype=np.float64),
        mass,
        np.asarray(trajectory.multiplicities, dtype=np.float64),
    )


def write_openpmd_particle_tracks_hdf5(
    path: str | Path,
    trajectory: ChargedTrajectory,
    masses: ArrayLike,
    /,
    *,
    scale: ElectromagneticScaleContract,
    species: str,
    limits: ResourceLimits,
    iterations: Sequence[int] | None = None,
) -> OpenPMDParticleTrackExportResult:
    """Encode and exclusively publish trajectory lanes as one openPMD species.

    Sample ``k`` becomes iteration ``iterations[k]`` (default ``k``) with time
    in the scale time unit. Records are stored in the scale's units with
    ``unitSI``/``unitDimension`` from ``scale.unit_si_map()``; ``momentum`` is
    ``masses * proper_velocities`` per particle. Lanes must share their times
    and be active at every sample. ``proper_accelerations`` have no openPMD
    particle record and are reported as a declared loss.
    """
    from ..electromagnetics._trajectory_radiation import ChargedTrajectory

    if not isinstance(trajectory, ChargedTrajectory):
        raise TypeError("trajectory must be a ChargedTrajectory.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be an ElectromagneticScaleContract.")
    if not isinstance(limits, ResourceLimits):
        raise TypeError("limits must be ResourceLimits.")
    species_name = _record_name(species, "species")
    samples, lanes = trajectory.sample_count, trajectory.particle_count
    indices = (
        tuple(range(samples))
        if iterations is None
        else _iteration_indices(iterations, "iterations")
    )
    if len(indices) != samples:
        raise ValueError("iterations must name one iteration per trajectory sample.")
    times, identities, positions, momenta, charges, mass, weights = _export_arrays(
        trajectory, masses
    )
    stored_words = samples * lanes * 7
    if stored_words * 8 > limits.max_bytes or stored_words > limits.max_nodes:
        raise ResourceReadError("limit", "Particle tracks exceed resource limits.")
    units = scale.unit_si_map()

    def unit(quantity: str) -> OpenPMDUnit:
        factor, dimension = units[quantity]
        return OpenPMDUnit(factor, _scale_dimension(dimension))

    scalars = {
        "weighting": (weights, OpenPMDUnit(1.0, _DIMENSIONLESS), 1.0),
        "charge": (charges, unit("charge"), 1.0),
        "mass": (mass, unit("mass"), 1.0),
    }
    buffer = BoundedHDF5Buffer(limits.max_bytes)
    with h5py.File(buffer, "w") as handle:
        write_series_root(
            handle, _REVISION, meshes_path=None, particles_path=_PARTICLES_PATH
        )
        for sample, iteration in enumerate(indices):
            step = times[sample] - times[sample - 1] if sample else 0.0
            group = write_iteration(
                handle,
                iteration,
                OpenPMDIterationTime(times[sample], step, units["time"][0]),
            )
            _write_species(
                group.create_group(f"{_PARTICLES_PATH}{species_name}"),
                identities,
                positions[sample],
                momenta[sample],
                scalars,
                unit("length"),
                unit("momentum"),
            )
    data = buffer.getvalue()
    destination = Path(path)
    resource = bounded_resource_from_bytes(
        data, limits=limits, source_path=str(destination)
    )
    publish_bytes(destination, data, maximum_bytes=limits.max_bytes, mode="exclusive")
    losses = (
        ()
        if trajectory.proper_accelerations is None
        else (
            AdapterLoss(
                "proper_accelerations",
                "export",
                "dropped",
                "openPMD 1.1.0 particle records carry no acceleration record.",
                changes_interpretation=False,
            ),
        )
    )
    source_id = canonical_fingerprint(
        {
            "kind": "charged-trajectory-openpmd-export",
            "scale": scale.scale_id,
            "species": species_name,
            "iterations": list(indices),
            "tracks": array_tree_fingerprint(
                (identities, times, positions, momenta, charges, mass, weights)
            ),
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS if losses else AdapterStatus.LOSSLESS,
        "ChargedTrajectory",
        _REVISION.format_id,
        source_id=source_id,
        target_id=resource.manifest.manifest_id,
        coordinate_mapping=(
            "shared lane times -> iteration time with timeUnitSI of the scale",
            "positions -> position with zero constant positionOffset",
            "masses * proper_velocities -> per-particle momentum",
            "(id_hi, id_lo) -> uint64 id",
        ),
        preserved_fields=(
            "particle identity",
            "position",
            "momentum",
            "charge",
            "rest mass",
            "lane multiplicity as weighting",
            "sample times",
        ),
        assumptions=_ASSUMPTIONS[:1],
        losses=losses,
    )
    return OpenPMDParticleTrackExportResult(destination, indices, resource, report)


__all__ = [
    "OpenPMDParticleTrackError",
    "OpenPMDParticleTrackExportResult",
    "OpenPMDParticleTrackImportPolicy",
    "OpenPMDParticleTrackImportResult",
    "OpenPMDParticleTrackSelection",
    "read_openpmd_particle_tracks_hdf5",
    "write_openpmd_particle_tracks_hdf5",
]
