#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..spatial._frame import MICROMETRE, MILLIMETRE, NANOMETRE, SpatialUnit
from ..spatial._imaging import ImageAxis, ImagePyramidLevel, ImagePyramidMetadata


class NGFFMetadataError(ValueError):
    """Raised when host metadata does not satisfy OME-NGFF multiscale semantics."""


_UNIT_BY_NGFF_NAME: dict[str, SpatialUnit] = {
    "nanometer": NANOMETRE,
    "nanometre": NANOMETRE,
    "nm": NANOMETRE,
    "micrometer": MICROMETRE,
    "micrometre": MICROMETRE,
    "micron": MICROMETRE,
    "µm": MICROMETRE,
    "um": MICROMETRE,
    "millimeter": MILLIMETRE,
    "millimetre": MILLIMETRE,
    "mm": MILLIMETRE,
}

_NGFF_NAME_BY_UNIT = {
    NANOMETRE: "nanometer",
    MICROMETRE: "micrometer",
    MILLIMETRE: "millimeter",
}


def is_ome_ngff_metadata(attributes: Any, /) -> bool:
    """Return whether host attributes declare an OME-NGFF multiscale object.

    A plain Zarr group or array is intentionally false even if it has shapes,
    chunks, or arbitrary user attributes.
    """
    if not isinstance(attributes, Mapping):
        return False
    multiscales = attributes.get("multiscales")
    if not isinstance(multiscales, Sequence) or isinstance(multiscales, (str, bytes)):
        return False
    if len(multiscales) < 1:
        return False
    return all(
        isinstance(entry, Mapping)
        and isinstance(entry.get("axes"), Sequence)
        and not isinstance(entry.get("axes"), (str, bytes))
        and isinstance(entry.get("datasets"), Sequence)
        and not isinstance(entry.get("datasets"), (str, bytes))
        for entry in multiscales
    )


def _axis(axis: Any, /) -> ImageAxis:
    if not isinstance(axis, Mapping):
        raise NGFFMetadataError(
            "OME-NGFF axes must be typed mappings; untyped generic Zarr axes are ambiguous."
        )
    name = axis.get("name")
    axis_type = axis.get("type")
    if not isinstance(name, str) or not name:
        raise NGFFMetadataError("Every OME-NGFF axis needs a non-empty name.")
    if axis_type not in ("space", "channel", "time"):
        raise NGFFMetadataError(f"OME-NGFF axis {name!r} has an unsupported type.")
    unit_name = axis.get("unit")
    if axis_type == "space":
        if unit_name is None:
            unit = None
        elif isinstance(unit_name, str) and unit_name.lower() in _UNIT_BY_NGFF_NAME:
            unit = _UNIT_BY_NGFF_NAME[unit_name.lower()]
        else:
            raise NGFFMetadataError(
                f"Spatial axis {name!r} uses an unsupported or non-physical unit."
            )
    elif axis_type == "time":
        if unit_name is None:
            unit = None
        elif isinstance(unit_name, str) and unit_name:
            unit = unit_name
        else:
            raise NGFFMetadataError(
                f"Temporal axis {name!r} uses an invalid unit declaration."
            )
    else:
        if unit_name is not None:
            raise NGFFMetadataError("OME-NGFF channel axes cannot declare a unit.")
        unit = None
    return ImageAxis(name, axis_type, unit)


def _transform_sequence(
    transforms: Any,
    dimension: int,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.ones((dimension,), dtype=float)
    translation = np.zeros((dimension,), dtype=float)
    if transforms is None:
        return scale, translation
    if not isinstance(transforms, Sequence) or isinstance(transforms, (str, bytes)):
        raise NGFFMetadataError("coordinateTransformations must be a sequence.")
    for transform in transforms:
        if not isinstance(transform, Mapping):
            raise NGFFMetadataError("Every coordinate transformation must be a mapping.")
        kind = transform.get("type")
        if kind == "scale":
            value = np.asarray(transform.get("scale"), dtype=float)
            if (
                value.shape != (dimension,)
                or np.any(~np.isfinite(value))
                or np.any(value <= 0.0)
            ):
                raise NGFFMetadataError(
                    "OME-NGFF scale transformations must be positive finite axis vectors."
                )
            translation = value * translation
            scale = value * scale
        elif kind == "translation":
            value = np.asarray(transform.get("translation"), dtype=float)
            if value.shape != (dimension,) or np.any(~np.isfinite(value)):
                raise NGFFMetadataError(
                    "OME-NGFF translations must be finite axis vectors."
                )
            translation = translation + value
        else:
            raise NGFFMetadataError(
                "Only OME-NGFF scale and translation transformations can be represented "
                "by ImagePyramidMetadata."
            )
    return scale, translation


def _composed_transforms(
    dataset_transforms: Any,
    global_transforms: Any,
    dimension: int,
    /,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    dataset_scale, dataset_translation = _transform_sequence(
        dataset_transforms, dimension
    )
    global_scale, global_translation = _transform_sequence(global_transforms, dimension)
    scale = global_scale * dataset_scale
    translation = global_scale * dataset_translation + global_translation
    return tuple(scale.tolist()), tuple(translation.tolist())


def ome_ngff_to_image_pyramid(
    attributes: Mapping[str, Any],
    array_shapes: Mapping[str, Sequence[int]],
    chunk_shapes: Mapping[str, Sequence[int]],
    /,
    *,
    multiscale_index: int = 0,
) -> ImagePyramidMetadata:
    """Convert validated OME-NGFF host metadata to native pyramid metadata.

    Array and chunk shapes are supplied by the storage layer because they are Zarr
    array properties, not OME-NGFF metadata. Their presence alone is never treated
    as evidence that a generic Zarr object is OME-NGFF.
    """
    if not is_ome_ngff_metadata(attributes):
        raise NGFFMetadataError(
            "Attributes do not declare OME-NGFF multiscales; generic Zarr is not NGFF."
        )
    index = int(multiscale_index)
    multiscales = attributes["multiscales"]
    if index < 0 or index >= len(multiscales):
        raise NGFFMetadataError("multiscale_index is out of range.")
    entry = multiscales[index]
    axes = tuple(_axis(axis) for axis in entry["axes"])
    dimension = len(axes)
    datasets = entry["datasets"]
    if len(datasets) < 1:
        raise NGFFMetadataError("OME-NGFF multiscales require at least one dataset.")
    global_transforms = entry.get("coordinateTransformations")
    levels: list[ImagePyramidLevel] = []
    for dataset in datasets:
        if not isinstance(dataset, Mapping):
            raise NGFFMetadataError("Every OME-NGFF dataset must be a mapping.")
        path = dataset.get("path")
        if not isinstance(path, str) or not path:
            raise NGFFMetadataError("Every OME-NGFF dataset requires a non-empty path.")
        if path not in array_shapes or path not in chunk_shapes:
            raise NGFFMetadataError(
                f"Storage shape and chunk metadata are required for dataset {path!r}."
            )
        scale, translation = _composed_transforms(
            dataset.get("coordinateTransformations"),
            global_transforms,
            dimension,
        )
        levels.append(
            ImagePyramidLevel(
                path,
                array_shapes[path],
                chunk_shapes[path],
                scale,
                translation,
            )
        )
    version = entry.get("version")
    if not isinstance(version, str) or not version:
        raise NGFFMetadataError("OME-NGFF multiscales require an explicit version.")
    name = entry.get("name", "")
    if not isinstance(name, str):
        raise NGFFMetadataError("OME-NGFF multiscale name must be a string.")
    return ImagePyramidMetadata(
        name,
        axes,
        levels,
        ngff_version=version,
    )


def image_pyramid_to_ome_ngff(metadata: ImagePyramidMetadata, /) -> dict[str, Any]:
    """Serialize native pyramid metadata as one canonical OME-NGFF multiscale entry."""
    if not isinstance(metadata, ImagePyramidMetadata):
        raise TypeError("metadata must be ImagePyramidMetadata.")
    axes = []
    for axis in metadata.axes:
        item: dict[str, Any] = {"name": axis.name, "type": axis.axis_type}
        if isinstance(axis.unit, SpatialUnit):
            if axis.unit not in _NGFF_NAME_BY_UNIT:
                raise NGFFMetadataError(
                    f"Spatial unit {axis.unit.name!r} has no canonical OME-NGFF encoding."
                )
            item["unit"] = _NGFF_NAME_BY_UNIT[axis.unit]
        elif isinstance(axis.unit, str):
            item["unit"] = axis.unit
        axes.append(item)
    datasets = []
    for level in metadata.levels:
        transforms: list[dict[str, Any]] = [{"type": "scale", "scale": list(level.scale)}]
        if any(value != 0.0 for value in level.translation):
            transforms.append(
                {"type": "translation", "translation": list(level.translation)}
            )
        datasets.append(
            {
                "path": level.path,
                "coordinateTransformations": transforms,
            }
        )
    entry: dict[str, Any] = {
        "version": metadata.ngff_version,
        "axes": axes,
        "datasets": datasets,
    }
    if metadata.name:
        entry["name"] = metadata.name
    return {"multiscales": [entry]}


__all__ = [
    "NGFFMetadataError",
    "image_pyramid_to_ome_ngff",
    "is_ome_ngff_metadata",
    "ome_ngff_to_image_pyramid",
]
