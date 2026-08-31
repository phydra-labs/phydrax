#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Literal

import numpy as np
import scipy.io

from ..imaging import DenseDisplacementField2D
from ..piv import PhysicalPIVResult2D
from ._piv_field import field_columns, field_from_columns
from ._report import (
    AdapterError,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    require_lossless,
)


PIVlabStage = Literal["original", "filtered", "smoothed"]
PIVlabYAxis = Literal["down", "up"]
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"


def read_pivlab(
    path: str | Path,
    /,
    *,
    geometry_id: str | None = None,
    y_axis: PIVlabYAxis,
    stage: PIVlabStage = "original",
    delta_t: float | None = None,
) -> tuple[
    tuple[DenseDisplacementField2D | PhysicalPIVResult2D, ...],
    AdapterReport,
]:
    """Read supported PIVlab MAT or HDF5 variable layouts without MATLAB execution."""
    source = Path(path)
    if y_axis not in ("down", "up"):
        raise ValueError("y_axis must explicitly be 'down' or 'up'.")
    if stage not in ("original", "filtered", "smoothed"):
        raise ValueError("Unknown PIVlab stage.")
    hdf5 = source.read_bytes()[:8] == _HDF5_SIGNATURE
    variables = _read_hdf5_variables(source) if hdf5 else _read_mat_variables(source)
    x_frames = _frames(variables, "x")
    y_frames = _frames(variables, "y")
    if stage == "original":
        u_name, v_name, type_name = "u_original", "v_original", "typevector_original"
    elif stage == "filtered":
        u_name, v_name, type_name = "u_filtered", "v_filtered", "typevector_filtered"
    else:
        u_name, v_name, type_name = "u_smoothed", "v_smoothed", "typevector_filtered"
    u_frames = _frames(variables, u_name)
    v_frames = _frames(variables, v_name)
    type_frames = _frames(variables, type_name)
    count = len(x_frames)
    if any(
        len(values) != count for values in (y_frames, u_frames, v_frames, type_frames)
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "PIVlab variables contain inconsistent frame counts.",
        )
    units = _text(variables, "units")
    unit_mode = _unit_mode(units)
    if unit_mode == "physical-velocity":
        if delta_t is None:
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "PIVlab physical velocity requires explicit delta_t to preserve displacement and velocity separately.",
            )
        time = float(delta_t)
        if not np.isfinite(time) or time <= 0.0:
            raise ValueError("delta_t must be finite and positive.")
    else:
        time = 1.0
    if unit_mode == "pixel-displacement" and (
        geometry_id is None or not str(geometry_id).strip()
    ):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Pixel-space PIVlab data require geometry_id.",
        )
    source_id = _file_id(source, "pivlab-hdf5" if hdf5 else "pivlab-mat")
    fields: list[DenseDisplacementField2D | PhysicalPIVResult2D] = []
    for index, (x, y, u, v, typevector) in enumerate(
        zip(x_frames, y_frames, u_frames, v_frames, type_frames, strict=True)
    ):
        x_, y_, u_, v_, type_ = (np.asarray(value) for value in (x, y, u, v, typevector))
        if (
            x_.ndim != 2
            or any(value.shape != x_.shape for value in (y_, u_, v_, type_))
            or np.iscomplexobj(x_)
            or np.iscomplexobj(y_)
            or np.iscomplexobj(u_)
            or np.iscomplexobj(v_)
        ):
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "PIVlab frames must contain same-shaped real two-dimensional arrays.",
            )
        if not np.issubdtype(type_.dtype, np.number) or np.any(type_ != np.floor(type_)):
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                "PIVlab typevector arrays must contain integer categories.",
            )
        valid = np.isin(type_.astype(int), (1, 3)) & np.isfinite(u_) & np.isfinite(v_)
        frame_source_id = f"{source_id}:frame:{index}"
        if unit_mode == "pixel-displacement":
            if y_axis == "down":
                row = y_
                dr = v_
            else:
                row = np.nanmin(y_) + np.nanmax(y_) - y_
                dr = -v_
            fields.append(
                field_from_columns(
                    row,
                    x_,
                    dr,
                    u_,
                    valid,
                    geometry_id=str(geometry_id),
                    source_id=frame_source_id,
                )
            )
        else:
            if y_axis == "down":
                physical_y = np.nanmin(y_) + np.nanmax(y_) - y_
                physical_v = -v_
            else:
                physical_y = y_
                physical_v = v_
            positions, vectors, validity = _physical_grid(
                x_,
                physical_y,
                u_,
                physical_v,
                valid,
            )
            if unit_mode == "physical-displacement":
                displacement = vectors
                velocity = vectors
                time_unit = "frame"
            else:
                velocity = vectors
                displacement = vectors * time
                time_unit = "s"
            fields.append(
                PhysicalPIVResult2D(
                    positions,
                    displacement,
                    velocity,
                    validity,
                    frame_source_id,
                    f"pivlab-physical:{source_id}",
                    "m",
                    time_unit,
                )
            )
    losses = [
        AdapterLoss(
            "typevector",
            "import",
            "dropped",
            "PIVlab typevector categories were reduced to authoritative native validity.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "session_and_algorithm_state",
            "import",
            "unsupported",
            "PIVlab field exports do not completely encode image, pass, validation, and calibration provenance.",
            changes_interpretation=True,
        ),
    ]
    if stage == "smoothed":
        losses.append(
            AdapterLoss(
                "smoothed_vectors",
                "import",
                "transformed",
                "PIVlab smoothed vectors are imported while rejected samples remain "
                "invalid; smoothing provenance is unavailable.",
                changes_interpretation=True,
            )
        )
    native_ids = tuple(
        field.field_id
        if isinstance(field, DenseDisplacementField2D)
        else field.source_field_id
        for field in fields
    )
    target_id = hashlib.sha256("|".join(native_ids).encode("utf-8")).hexdigest()
    target_format = (
        "DenseDisplacementField2D-sequence"
        if unit_mode == "pixel-displacement"
        else "PhysicalPIVResult2D-sequence"
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        "PIVlab-HDF5" if hdf5 else "PIVlab-MAT",
        target_format,
        source_id=source_id,
        target_id=target_id,
        coordinate_mapping=(
            (
                "PIVlab x -> column_right"
                if unit_mode == "pixel-displacement"
                else "PIVlab x -> physical x"
            ),
            (
                "PIVlab y -> row_down"
                if unit_mode == "pixel-displacement" and y_axis == "down"
                else "reflected PIVlab y -> row_down"
                if unit_mode == "pixel-displacement"
                else "reflected PIVlab y -> physical y"
                if y_axis == "down"
                else "PIVlab y -> physical y"
            ),
            (
                "PIVlab (u,v) -> image displacement (column_right,row_down)"
                if unit_mode == "pixel-displacement"
                else "PIVlab (u,v) -> right-handed physical (x,y)"
            ),
        ),
        preserved_fields=("x", "y", u_name, v_name, type_name),
        assumptions=(f"units={units}", f"stage={stage}", f"y_axis={y_axis}"),
        losses=losses,
    )
    return tuple(fields), report


def write_pivlab(
    path: str | Path,
    fields: (
        DenseDisplacementField2D
        | PhysicalPIVResult2D
        | tuple[DenseDisplacementField2D | PhysicalPIVResult2D, ...]
    ),
    /,
    *,
    y_axis: PIVlabYAxis,
    hdf5: bool | None = None,
    lossless: bool = False,
) -> AdapterReport:
    """Write the documented PIVlab field-variable layout with an explicit loss report."""
    if y_axis not in ("down", "up"):
        raise ValueError("y_axis must explicitly be 'down' or 'up'.")
    fields_ = (
        (fields,)
        if isinstance(fields, (DenseDisplacementField2D, PhysicalPIVResult2D))
        else tuple(fields)
    )
    if not fields_:
        raise ValueError("fields must contain at least one native PIV field.")
    pixel_fields = all(isinstance(field, DenseDisplacementField2D) for field in fields_)
    physical_fields = all(isinstance(field, PhysicalPIVResult2D) for field in fields_)
    if not (pixel_fields or physical_fields):
        raise TypeError(
            "PIVlab export cannot mix pixel-displacement and physical-velocity fields."
        )
    destination = Path(path)
    use_hdf5 = (
        destination.suffix.lower() in (".h5", ".hdf5") if hdf5 is None else bool(hdf5)
    )
    x_values: list[np.ndarray] = []
    y_values: list[np.ndarray] = []
    u_values: list[np.ndarray] = []
    v_values: list[np.ndarray] = []
    type_values: list[np.ndarray] = []
    source_ids: list[str] = []
    for field in fields_:
        if isinstance(field, DenseDisplacementField2D):
            row, column, dr, dc, valid = field_columns(field)
            x_values.append(column)
            if y_axis == "down":
                y_values.append(row)
                v_values.append(np.where(valid, dr, np.nan))
            else:
                y_values.append(np.min(row) + np.max(row) - row)
                v_values.append(np.where(valid, -dr, np.nan))
            u_values.append(np.where(valid, dc, np.nan))
            source_ids.append(field.field_id)
        else:
            x, y, velocity, valid = _physical_columns(field)
            x_values.append(x)
            if y_axis == "down":
                y_values.append(np.min(y) + np.max(y) - y)
                v_values.append(np.where(valid, -velocity[..., 1], np.nan))
            else:
                y_values.append(y)
                v_values.append(np.where(valid, velocity[..., 1], np.nan))
            u_values.append(np.where(valid, velocity[..., 0], np.nan))
            source_ids.append(field.source_field_id)
        type_values.append(valid.astype(np.int32))
    variables = {
        "x": x_values,
        "y": y_values,
        "u_original": u_values,
        "v_original": v_values,
        "typevector_original": type_values,
        "u_filtered": u_values,
        "v_filtered": v_values,
        "typevector_filtered": type_values,
        "u_smoothed": u_values,
        "v_smoothed": v_values,
    }
    units = "[px] respectively [px/frame]" if pixel_fields else "[m] respectively [m/s]"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if use_hdf5:
        _write_hdf5_variables(destination, variables, units=units)
    else:
        _write_mat_variables(destination, variables, units=units)
    target_id = _file_id(destination, "pivlab-hdf5" if use_hdf5 else "pivlab-mat")
    source_id = hashlib.sha256("|".join(source_ids).encode("utf-8")).hexdigest()
    losses = (
        AdapterLoss(
            "field_identity_and_provenance",
            "export",
            "dropped",
            "PIVlab field variables cannot encode native geometry, field identity, and provenance.",
            changes_interpretation=True,
        ),
        AdapterLoss(
            "typevector",
            "export",
            "synthesized",
            "Native validity was encoded as PIVlab type 1 or 0 without inventing "
            "validation or interpolation categories.",
            changes_interpretation=False,
        ),
        AdapterLoss(
            "filtered_and_smoothed_stages",
            "export",
            "synthesized",
            "The same unmodified field was used for original, filtered, and smoothed variables.",
            changes_interpretation=False,
        ),
    )
    report = AdapterReport(
        AdapterStatus.DECLARED_LOSS,
        (
            "DenseDisplacementField2D-sequence"
            if pixel_fields
            else "PhysicalPIVResult2D-sequence"
        ),
        "PIVlab-HDF5" if use_hdf5 else "PIVlab-MAT",
        source_id=source_id,
        target_id=target_id,
        coordinate_mapping=(
            ("column_right -> PIVlab x" if pixel_fields else "physical x -> PIVlab x"),
            (
                "row_down -> PIVlab y"
                if pixel_fields and y_axis == "down"
                else "reflected row_down -> PIVlab y"
                if pixel_fields
                else "reflected physical y -> PIVlab y_down"
                if y_axis == "down"
                else "physical y -> PIVlab y_up"
            ),
            (
                "image displacement -> PIVlab (u,v)"
                if pixel_fields
                else "physical velocity -> PIVlab (u,v)"
            ),
        ),
        preserved_fields=(
            ("positions_rc", "displacement_rc", "valid")
            if pixel_fields
            else ("positions_xy", "velocity_xy", "valid")
        ),
        assumptions=(f"units={units}", f"y_axis={y_axis}"),
        losses=losses,
    )
    if lossless:
        require_lossless(report)
    return report


def _read_mat_variables(path: Path, /) -> dict[str, Any]:
    loaded = scipy.io.loadmat(path, squeeze_me=False, struct_as_record=True)
    return {name: value for name, value in loaded.items() if not name.startswith("__")}


def _read_hdf5_variables(path: Path, /) -> dict[str, Any]:
    h5py = _h5py()
    values: dict[str, Any] = {}
    with h5py.File(path, "r") as archive:
        required = {"x", "y", "u_original", "v_original", "typevector_original"}
        if not required.issubset(archive.keys()):
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "HDF5 file is not a supported PIVlab field-variable layout.",
            )
        for name in archive.keys():
            item = archive[name]
            if not isinstance(item, h5py.Dataset):
                raise AdapterError(
                    AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                    "PIVlab HDF5 groups and MATLAB object-reference cells are unsupported.",
                )
            if h5py.check_dtype(ref=item.dtype) is not None:
                raise AdapterError(
                    AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                    "PIVlab HDF5 object references require MATLAB execution and are unsupported.",
                )
            if item.dtype.hasobject and not (
                name == "units" and h5py.check_string_dtype(item.dtype) is not None
            ):
                raise AdapterError(
                    AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                    "PIVlab HDF5 object values are unsupported.",
                )
            values[name] = item[()]
    return values


def _write_mat_variables(
    path: Path,
    variables: dict[str, list[np.ndarray]],
    /,
    *,
    units: str,
) -> None:
    count = len(variables["x"])
    output: dict[str, Any] = {}
    for name, frames in variables.items():
        if count == 1:
            output[name] = frames[0]
        else:
            cells = np.empty((count, 1), dtype=object)
            for index, value in enumerate(frames):
                cells[index, 0] = value
            output[name] = cells
    output.update(
        {
            "calxy": 1.0,
            "calu": 1.0,
            "calv": 1.0,
            "units": units,
            "information": "Native validity is encoded as typevector 1 or 0.",
        }
    )
    scipy.io.savemat(path, output, do_compression=False, oned_as="column")


def _write_hdf5_variables(
    path: Path,
    variables: dict[str, list[np.ndarray]],
    /,
    *,
    units: str,
) -> None:
    h5py = _h5py()
    with h5py.File(path, "w") as archive:
        for name, frames in variables.items():
            archive.create_dataset(name, data=np.stack(frames, axis=0))
        archive.create_dataset("calxy", data=1.0)
        archive.create_dataset("calu", data=1.0)
        archive.create_dataset("calv", data=1.0)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        archive.create_dataset("units", data=units, dtype=string_dtype)


def _frames(variables: dict[str, Any], name: str, /) -> tuple[np.ndarray, ...]:
    if name not in variables:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            f"PIVlab source does not contain required variable {name!r}.",
        )
    value = np.asarray(variables[name])
    if value.dtype.hasobject:
        frames = tuple(np.asarray(item) for item in value.reshape((-1,), order="F"))
        if not frames or any(frame.size == 0 for frame in frames):
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                f"PIVlab variable {name!r} contains empty or unsupported cells.",
            )
        return frames
    if value.ndim == 2:
        return (value,)
    if value.ndim == 3:
        return tuple(value[index] for index in range(value.shape[0]))
    raise AdapterError(
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
        f"PIVlab variable {name!r} has an unsupported rank.",
    )


def _text(variables: dict[str, Any], name: str, /) -> str:
    if name not in variables:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            f"PIVlab source omits required {name!r} metadata.",
        )
    value = np.asarray(variables[name])
    if value.dtype.kind in ("U", "S"):
        items = tuple(
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in value.reshape((-1,))
        )
        return "".join(items).strip()
    if value.ndim == 0 and isinstance(value.item(), (str, bytes)):
        item = value.item()
        return item.decode("utf-8") if isinstance(item, bytes) else item
    raise AdapterError(
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
        f"PIVlab metadata {name!r} is not a supported text value.",
    )


def _physical_columns(
    field: PhysicalPIVResult2D,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(field.positions_xy)
    velocity = np.asarray(field.velocity_xy)
    valid = np.asarray(field.valid, dtype=bool)
    if (
        positions.ndim != 3
        or positions.shape[-1] != 2
        or velocity.shape != positions.shape
        or valid.shape != positions.shape[:-1]
    ):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "PhysicalPIVResult2D arrays have inconsistent shapes.",
        )
    x = positions[..., 0]
    y = positions[..., 1]
    if not (
        np.allclose(x, np.broadcast_to(x[0, :], valid.shape), rtol=0.0, atol=0.0)
        and np.allclose(
            y, np.broadcast_to(y[:, 0, None], valid.shape), rtol=0.0, atol=0.0
        )
    ):
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "PIVlab export requires a rectilinear physical grid.",
        )
    if np.any(valid & ~np.all(np.isfinite(velocity), axis=-1)):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Every valid physical PIV velocity must be finite.",
        )
    return x, y, velocity, valid


def _unit_mode(units: str, /) -> str:
    normalized = " ".join(units.split())
    if normalized == "[px] respectively [px/frame]":
        return "pixel-displacement"
    if normalized == "[m] respectively [m/frame]":
        return "physical-displacement"
    if normalized == "[m] respectively [m/s]":
        return "physical-velocity"
    raise AdapterError(
        AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
        f"Unsupported or ambiguous PIVlab units {units!r}.",
    )


def _physical_grid(
    x,
    y,
    vector_x,
    vector_y,
    valid,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_ = np.asarray(x, dtype=float).reshape((-1,))
    y_ = np.asarray(y, dtype=float).reshape((-1,))
    vx = np.asarray(vector_x, dtype=float).reshape((-1,))
    vy = np.asarray(vector_y, dtype=float).reshape((-1,))
    valid_ = np.asarray(valid, dtype=bool).reshape((-1,))
    if np.any(valid_ & (~np.isfinite(vx) | ~np.isfinite(vy))):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Every valid physical PIVlab vector must be finite.",
        )
    _, x_first = np.unique(x_, return_index=True)
    _, y_first = np.unique(y_, return_index=True)
    x_axis = x_[np.sort(x_first)]
    y_axis = y_[np.sort(y_first)]
    if x_axis.size * y_axis.size != x_.size:
        raise AdapterError(
            AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            "Physical PIVlab positions must define a complete rectilinear grid.",
        )
    x_matches = x_[:, None] == x_axis[None, :]
    y_matches = y_[:, None] == y_axis[None, :]
    if not np.all(np.any(x_matches, axis=1)) or not np.all(np.any(y_matches, axis=1)):
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Physical PIVlab positions do not map exactly to their coordinate axes.",
        )
    ix = np.argmax(x_matches, axis=1)
    iy = np.argmax(y_matches, axis=1)
    flat = iy * x_axis.size + ix
    if np.unique(flat).size != x_.size:
        raise AdapterError(
            AdapterStatus.INCONSISTENT_SOURCE,
            "Physical PIVlab positions contain duplicates.",
        )
    shape = (y_axis.size, x_axis.size)
    positions = np.empty(shape + (2,), dtype=float)
    positions[..., 0] = x_axis[None, :]
    positions[..., 1] = y_axis[:, None]
    vectors = np.zeros(shape + (2,), dtype=np.result_type(vx, vy, float))
    validity = np.zeros(shape, dtype=bool)
    vectors[iy, ix, 0] = np.where(valid_, vx, 0.0)
    vectors[iy, ix, 1] = np.where(valid_, vy, 0.0)
    validity[iy, ix] = valid_
    return positions, vectors, validity


def _h5py():
    if importlib.util.find_spec("h5py") is None:
        raise AdapterError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "PIVlab HDF5 interoperability requires optional dependency 'h5py'.",
        )
    return importlib.import_module("h5py")


def _file_id(path: Path, format_name: str, /) -> str:
    return f"{format_name}:sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


__all__ = [
    "PIVlabStage",
    "PIVlabYAxis",
    "read_pivlab",
    "write_pivlab",
]
