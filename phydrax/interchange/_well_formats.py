#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

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
from ..units import conversion_factor, METER, UnitDefinition
from ._borehole import BoreholeTrajectory
from ._report import (
    AdapterCapability,
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
)
from ._resource import BoundedResource, read_bounded_resource, ResourceLimits


class WellFormatDependencyError(RuntimeError):
    """The optional LAS decoder is unavailable."""


class QualifiedWellLog(StrictModule, NonTrainableState):
    measured_depth_m: Array
    values: Array
    valid: Array
    mnemonic: str = eqx.field(static=True)
    value_unit_label: str = eqx.field(static=True)
    null_value: float = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    resource: BoundedResource = eqx.field(static=True)
    report: AdapterReport = eqx.field(static=True)
    log_id: str = eqx.field(static=True)


def read_las_curve(
    path: str | Path,
    /,
    *,
    trusted_root: str | Path,
    limits: ResourceLimits,
    trajectory: BoreholeTrajectory,
    mnemonic: str,
    measured_depth_unit: UnitDefinition,
    expected_index_unit_label: str,
) -> QualifiedWellLog:
    if not isinstance(trajectory, BoreholeTrajectory):
        raise TypeError("LAS curve requires a qualified BoreholeTrajectory.")
    curve_name = str(mnemonic).strip().upper()
    expected_depth_unit = str(expected_index_unit_label).strip().upper()
    if not curve_name or not expected_depth_unit:
        raise ValueError("LAS curve mnemonic and expected index-unit label are required.")
    resource = read_bounded_resource(path, trusted_root=trusted_root, limits=limits)
    try:
        lasio = cast(Any, import_module("lasio"))
    except ImportError as error:
        raise WellFormatDependencyError(
            "LAS decoding requires the optional lasio runtime."
        ) from error
    with TemporaryDirectory(prefix="phydrax-las-") as temporary:
        source = Path(temporary) / "well.las"
        source.write_bytes(resource.data)
        las = lasio.read(source)
    if 3 * len(las.index) > limits.max_nodes or len(las.curves) > limits.max_attributes:
        raise ValueError("LAS selected curve exceeds decoded resource limits.")
    if curve_name not in tuple(str(name).upper() for name in las.keys()):
        raise ValueError("Requested LAS curve is absent.")
    depth_name = str(las.index_unit).strip().upper()
    if not depth_name or depth_name != expected_depth_unit:
        raise ValueError("LAS measured-depth index unit is absent or unexpected.")
    factor = float(conversion_factor(measured_depth_unit, METER))
    depth = np.asarray(las.index, dtype=float) * factor
    values = np.asarray(las[curve_name], dtype=float)
    if depth.ndim != 1 or values.shape != depth.shape or depth.size == 0:
        raise ValueError("LAS index and selected curve must be matching vectors.")
    if np.any(~np.isfinite(depth)) or np.any(np.diff(depth) <= 0):
        raise ValueError("LAS measured depth must be finite and strictly increasing.")
    stations = np.asarray(trajectory.measured_depth_m)
    if depth[0] < stations[0] or depth[-1] > stations[-1]:
        raise ValueError("LAS curve lies outside the qualified borehole trajectory.")
    null = float(las.well.NULL.value)
    valid = np.isfinite(values) & (values != null)
    data = np.where(valid, values, 0.0)
    curve = las.curves[curve_name]
    unit_label = str(curve.unit).strip()
    identity = canonical_fingerprint(
        {
            "kind": "qualified-las-curve",
            "resource": resource.manifest.content_sha256,
            "trajectory": trajectory.trajectory_id,
            "mnemonic": curve_name,
            "value_unit_label": unit_label,
            "depth_unit_label": depth_name,
            "depth_m": depth,
            "values": data,
            "valid": valid,
        }
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "LAS",
        "QualifiedWellLog",
        source_id=resource.manifest.content_sha256,
        target_id=identity,
        source_profile=AdapterFormatProfile(
            "LAS", qualifiers={"profile": "2-or-3-selected-curve"}
        ),
        preserved_fields=(
            "measured depth",
            "selected curve values and null mask",
            "curve mnemonic and unit label",
            "exact source bytes",
        ),
        assumptions=(
            "Caller explicitly binds the matched LAS depth-unit label to the supplied UnitDefinition.",
            "Curve value unit remains an uninterpreted external label until a modality adapter qualifies it.",
        ),
        losses=(
            AdapterLoss(
                "unselected-curves-and-headers",
                "import",
                "dropped",
                "Only the requested curve is materialized; exact source bytes remain retained.",
                changes_interpretation=False,
            ),
        ),
        capabilities=(
            AdapterCapability("borehole-log"),
            AdapterCapability("explicit-null-mask"),
        ),
    )
    return QualifiedWellLog(
        jnp.asarray(depth),
        jnp.asarray(data),
        jnp.asarray(valid),
        curve_name,
        unit_label,
        null,
        trajectory.trajectory_id,
        resource,
        report,
        identity,
    )


__all__ = ["QualifiedWellLog", "WellFormatDependencyError", "read_las_curve"]
