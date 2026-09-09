#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._report import (
    AdapterCapability,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)
from ._resource import BoundedResource, read_bounded_resource, ResourceLimits
from ._time_reference import TimeReferenceContract


class GeodeticFormatDependencyError(RuntimeError):
    """The optional RINEX decoder is unavailable."""


class SINEXPositionSolution(StrictModule, NonTrainableState):
    stations: tuple[str, ...] = eqx.field(static=True)
    point_codes: tuple[str, ...] = eqx.field(static=True)
    solution_ids: tuple[str, ...] = eqx.field(static=True)
    reference_epochs: tuple[str, ...] = eqx.field(static=True)
    ecef_m: Array
    standard_deviation_m: Array
    frame_id: str = eqx.field(static=True)
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    solution_id: str = eqx.field(static=True)


class RINEXObservationData(StrictModule, NonTrainableState):
    times_tai_seconds: Array
    satellites: tuple[str, ...] = eqx.field(static=True)
    observables: tuple[str, ...] = eqx.field(static=True)
    values: Array
    valid: Array
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    observation_id: str = eqx.field(static=True)


def read_sinex_positions(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    frame_id: str,
) -> SINEXPositionSolution:
    frame = str(frame_id).strip()
    if not frame:
        raise ValueError("SINEX position import requires an explicit frame identity.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    try:
        text = resource.data.decode("ascii", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("SINEX must be ASCII.") from error
    if not text.startswith("%=SNX"):
        raise ValueError("SINEX file lacks its required %=SNX header.")
    active = False
    grouped: dict[tuple[str, str, str, str], dict[str, tuple[float, float]]] = {}
    for line in text.splitlines():
        if line.startswith("+SOLUTION/ESTIMATE"):
            active = True
            continue
        if line.startswith("-SOLUTION/ESTIMATE"):
            active = False
            continue
        if not active or not line.strip() or line.startswith("*"):
            continue
        fields = line.split()
        if len(fields) < 10 or fields[1] not in ("STAX", "STAY", "STAZ"):
            continue
        if fields[6] != "m":
            raise ValueError("SINEX station coordinates must use metres.")
        key = (fields[2], fields[3], fields[4], fields[5])
        if key not in grouped and 6 * (len(grouped) + 1) > limits.max_nodes:
            raise ValueError("SINEX station solutions exceed the decoded-node limit.")
        component = fields[1][-1]
        if component in grouped.setdefault(key, {}):
            raise ValueError("SINEX station component is duplicated.")
        value, sigma = float(fields[8]), float(fields[9])
        if not np.isfinite(value) or not np.isfinite(sigma) or sigma < 0:
            raise ValueError("SINEX station estimates and sigmas must be finite.")
        grouped[key][component] = (value, sigma)
    complete = sorted(
        (key, value) for key, value in grouped.items() if set(value) == {"X", "Y", "Z"}
    )
    if not complete or len(complete) != len(grouped):
        raise ValueError("SINEX profile requires complete X/Y/Z station solutions.")
    stations, points, solutions, epochs = zip(*(key for key, _ in complete), strict=True)
    positions = np.asarray(
        [[values[axis][0] for axis in "XYZ"] for _, values in complete]
    )
    deviations = np.asarray(
        [[values[axis][1] for axis in "XYZ"] for _, values in complete]
    )
    identity = canonical_fingerprint(
        {
            "kind": "sinex-position-solution",
            "resource": resource.manifest.content_sha256,
            "frame_id": frame,
            "keys": [key for key, _ in complete],
            "positions_m": positions,
            "standard_deviation_m": deviations,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "SINEX",
        "SINEXPositionSolution",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "SINEX", qualifiers={"profile": "solution-estimate-position"}
        ),
        preserved_fields=(
            "station, point, solution, and reference-epoch identifiers",
            "ECEF position estimates and standard deviations in metres",
            "exact source bytes",
        ),
        assumptions=(
            "The caller binds the SINEX solution to the supplied reference-frame identity.",
        ),
        losses=(
            AdapterLoss(
                "non-position-solution-blocks",
                "import",
                "dropped",
                "Only complete STAX/STAY/STAZ solution estimates are normalized; all source bytes remain retained.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(
            AdapterCapability("geodetic-position-solution"),
            AdapterCapability("measurement-uncertainty"),
        ),
    )
    return SINEXPositionSolution(
        tuple(stations),
        tuple(points),
        tuple(solutions),
        tuple(epochs),
        jnp.asarray(positions),
        jnp.asarray(deviations),
        frame,
        resource,
        report,
        identity,
    )


def read_rinex_observations(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    time_reference: TimeReferenceContract,
    observables: Sequence[str],
) -> RINEXObservationData:
    if (
        time_reference.epoch_nominal_seconds is None
        or time_reference.scale == "source-relative"
    ):
        raise ValueError(
            "RINEX timestamps require an explicit absolute source-time contract and nominal epoch."
        )
    requested = tuple(str(value).strip() for value in observables)
    if (
        not requested
        or any(not value for value in requested)
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("RINEX observable names must be explicit, nonempty, and unique.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    try:
        georinex = cast(Any, import_module("georinex"))
    except ImportError as error:
        raise GeodeticFormatDependencyError(
            "RINEX decoding requires the optional georinex runtime."
        ) from error
    with TemporaryDirectory(prefix="phydrax-rinex-") as temporary:
        source = Path(temporary) / "observations.rnx"
        source.write_bytes(resource.data)
        dataset = georinex.load(source)
        if "time" not in dataset.coords or "sv" not in dataset.coords:
            raise ValueError("RINEX dataset lacks time or satellite coordinates.")
        if any(name not in dataset for name in requested):
            raise ValueError("Requested RINEX observable is absent.")
        selected = dataset[list(requested)]
        selected_arrays = tuple(selected[name] for name in requested)
        if any(tuple(value.dims) != ("time", "sv") for value in selected_arrays):
            raise ValueError("RINEX observable dimensions must be exactly (time, sv).")
        decoded_nodes = (
            int(selected.coords["time"].size)
            + int(selected.coords["sv"].size)
            + sum(int(value.size) for value in selected_arrays)
        )
        attribute_count = (
            len(selected.attrs)
            + sum(len(value.attrs) for value in selected_arrays)
            + sum(len(value.attrs) for value in selected.coords.values())
        )
        if (
            decoded_nodes > limits.max_nodes
            or attribute_count > limits.max_attributes
            or any(
                not np.issubdtype(value.dtype, np.number)
                or np.issubdtype(value.dtype, np.complexfloating)
                for value in selected_arrays
            )
        ):
            raise ValueError(
                "Selected RINEX observables exceed resource limits or are not real numeric."
            )
        dataset = selected.load()
    time_ns = np.asarray(dataset.time.values).astype("datetime64[ns]").astype(np.int64)
    nominal_source = time_ns.astype(np.float64) * 1e-9
    recorded = nominal_source - time_reference.epoch_nominal_seconds
    times_tai = np.asarray(time_reference.to_tai(recorded))
    satellites = tuple(str(value) for value in np.asarray(dataset.sv.values))
    arrays = []
    for name in requested:
        variable = dataset[name]
        arrays.append(np.asarray(variable.values, dtype=float))
    values = np.stack(arrays, axis=-1)
    valid = np.isfinite(values)
    data = np.where(valid, values, 0.0)
    identity = canonical_fingerprint(
        {
            "kind": "rinex-observation-data",
            "resource": resource.manifest.content_sha256,
            "time_contract": time_reference.time_id,
            "times_tai_seconds": times_tai,
            "satellites": satellites,
            "observables": requested,
            "values": data,
            "valid": valid,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "RINEX",
        "RINEXObservationData",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "RINEX",
            qualifiers={
                "profile": "selected-observables",
                "time_scale": time_reference.scale,
                "observables": ",".join(requested),
            },
        ),
        preserved_fields=(
            "selected time/satellite/observable values and validity",
            "absolute TAI timestamps through the supplied source-time contract",
            "exact source bytes",
        ),
        assumptions=(
            "The caller binds decoded nominal timestamps to the supplied source-time contract.",
        ),
        losses=(
            AdapterLoss(
                "unselected-observables-and-headers",
                "import",
                "dropped",
                "Only requested observables are normalized; all source bytes remain retained.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(
            AdapterCapability("gnss-observations"),
            AdapterCapability("absolute-time"),
            AdapterCapability("sample-validity"),
        ),
    )
    return RINEXObservationData(
        jnp.asarray(times_tai),
        satellites,
        requested,
        jnp.asarray(data),
        jnp.asarray(valid),
        resource,
        report,
        identity,
    )


__all__ = [
    "GeodeticFormatDependencyError",
    "RINEXObservationData",
    "SINEXPositionSolution",
    "read_rinex_observations",
    "read_sinex_positions",
]
