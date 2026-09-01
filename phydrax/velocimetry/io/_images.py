#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import equinox as eqx
import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...interchange import AdapterError, AdapterLoss, AdapterReport, AdapterStatus
from ..imaging import ImageGeometry2D, ImagePair2D


ImageLoader = Callable[[Path], Any]


class LazyImageSequence2D(StrictModule, NonTrainableState):
    """Host-side image paths materialized only when a frame or pair is requested."""

    geometry: ImageGeometry2D | None
    paths: tuple[Path, ...] = eqx.field(static=True)
    times: tuple[float, ...] = eqx.field(static=True)
    loader: ImageLoader | None = eqx.field(static=True)
    mask_loader: ImageLoader | None = eqx.field(static=True)
    array_name: str | None = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        paths: Sequence[str | Path],
        /,
        *,
        times: Sequence[float] | None = None,
        geometry: ImageGeometry2D | None = None,
        loader: ImageLoader | None = None,
        mask_loader: ImageLoader | None = None,
        array_name: str | None = None,
        dtype: Any = np.float32,
        loader_id: str | None = None,
        source_id: str | None = None,
    ):
        paths_ = tuple(Path(path) for path in paths)
        if not paths_:
            raise ValueError("Lazy image sequences require at least one path.")
        times_ = (
            tuple(float(index) for index in range(len(paths_)))
            if times is None
            else tuple(float(value) for value in times)
        )
        if len(times_) != len(paths_) or not np.all(np.isfinite(times_)):
            raise ValueError("times must contain one finite value per image path.")
        if any(right <= left for left, right in zip(times_[:-1], times_[1:])):
            raise ValueError("Lazy image times must be strictly increasing.")
        if geometry is not None and not isinstance(geometry, ImageGeometry2D):
            raise TypeError("geometry must be ImageGeometry2D or None.")
        dtype_ = np.dtype(dtype)
        if not np.issubdtype(dtype_, np.floating):
            raise TypeError("Lazy image materialization requires a real floating dtype.")
        if loader is not None and (loader_id is None or not str(loader_id).strip()):
            raise ValueError("Custom image loaders require a stable loader_id.")
        array_name_ = None if array_name is None else str(array_name).strip()
        if array_name is not None and not array_name_:
            raise ValueError("array_name must be non-empty or None.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "lazy-image-sequence-2d",
                    "paths": [str(path) for path in paths_],
                    "times": list(times_),
                    "geometry_id": None if geometry is None else geometry.geometry_id,
                    "loader_id": "native" if loader is None else str(loader_id),
                    "array_name": array_name_,
                    "dtype": dtype_.str,
                }
            )
            if source_id is None
            else str(source_id).strip()
        )
        if not identifier:
            raise ValueError("source_id must be non-empty.")
        self.geometry = geometry
        self.paths = paths_
        self.times = times_
        self.loader = loader
        self.mask_loader = mask_loader
        self.array_name = array_name_
        self.dtype = dtype_
        self.source_id = identifier

    def read(self, index: int, /) -> np.ndarray:
        """Materialize one scalar image without loading neighboring frames."""
        index_ = int(index)
        if index_ < 0 or index_ >= len(self.paths):
            raise IndexError("Lazy image index is outside sequence capacity.")
        path = self.paths[index_]
        value = (
            _read_native_image(path, array_name=self.array_name)
            if self.loader is None
            else self.loader(path)
        )
        image = np.asarray(value)
        if image.ndim != 2:
            raise AdapterError(
                AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                "Image ingestion requires a scalar two-dimensional image; select "
                "a channel explicitly in a custom loader.",
            )
        if np.iscomplexobj(image) or not np.all(np.isfinite(image)):
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                "Image values must be finite and real-valued.",
            )
        if self.geometry is not None and tuple(image.shape) != self.geometry.image_shape:
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                "Image shape does not match the declared ImageGeometry2D.",
            )
        return image.astype(self.dtype, copy=False)

    def pair(
        self,
        first_index: int,
        second_index: int,
        /,
    ) -> tuple[ImagePair2D, AdapterReport]:
        """Materialize exactly two frames as a native image pair plus conversion report."""
        first_index_ = int(first_index)
        second_index_ = int(second_index)
        if first_index_ >= second_index_:
            raise ValueError("Image pair indices must be strictly increasing.")
        first = self.read(first_index_)
        second = self.read(second_index_)
        if first.shape != second.shape:
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                "Image pair frames have inconsistent shapes.",
            )
        geometry = (
            ImageGeometry2D(first.shape) if self.geometry is None else self.geometry
        )
        first_mask = self._read_mask(first_index_, first.shape)
        second_mask = self._read_mask(second_index_, second.shape)
        pair_id = canonical_fingerprint(
            {
                "source_id": self.source_id,
                "indices": [first_index_, second_index_],
                "geometry_id": geometry.geometry_id,
            }
        )
        pair = ImagePair2D(
            first,
            second,
            geometry,
            first_mask=first_mask,
            second_mask=second_mask,
            delta_t=self.times[second_index_] - self.times[first_index_],
            pair_id=pair_id,
            provenance=(self.source_id,),
        )
        loss = AdapterLoss(
            "intensity.dtype",
            "import",
            "transformed",
            f"Source intensities were represented as {self.dtype.str} native floating values.",
            changes_interpretation=False,
        )
        report = AdapterReport(
            AdapterStatus.DECLARED_LOSS,
            "image-files",
            "ImagePair2D",
            source_id=self.source_id,
            target_id=pair.pair_id,
            coordinate_mapping=(
                "array axis 0 -> row_down",
                "array axis 1 -> column_right",
            ),
            preserved_fields=("first", "second", "first_mask", "second_mask", "delta_t"),
            assumptions=("integer samples identify pixel cells",),
            losses=(loss,),
        )
        return pair, report

    def _read_mask(self, index: int, shape: tuple[int, ...], /) -> np.ndarray | None:
        if self.mask_loader is None:
            return None
        mask = np.asarray(self.mask_loader(self.paths[index]), dtype=bool)
        if mask.shape != shape:
            raise AdapterError(
                AdapterStatus.INCONSISTENT_SOURCE,
                "An image support mask has a shape inconsistent with its image.",
            )
        return mask


def _read_native_image(path: Path, /, *, array_name: str | None) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False, mmap_mode="r")
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=False)
        names = tuple(archive.files)
        if array_name is None:
            if len(names) != 1:
                archive.close()
                raise AdapterError(
                    AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                    "NPZ image sources with multiple arrays require array_name.",
                )
            selected = names[0]
        else:
            selected = array_name
            if selected not in names:
                archive.close()
                raise AdapterError(
                    AdapterStatus.MALFORMED_SOURCE,
                    f"NPZ image source does not contain array {selected!r}.",
                )
        value = np.array(archive[selected], copy=True)
        archive.close()
        return value
    if importlib.util.find_spec("imageio") is None:
        raise AdapterError(
            AdapterStatus.OPTIONAL_DEPENDENCY_UNAVAILABLE,
            "Reading non-NumPy image files requires optional dependency 'imageio'.",
        )
    imageio = importlib.import_module("imageio.v3")
    return np.asarray(imageio.imread(path))


__all__ = ["ImageLoader", "LazyImageSequence2D"]
