#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Lazy NIfTI/MGZ interchange with exact spatial and rights contracts."""

from __future__ import annotations

from collections.abc import Callable
from hashlib import new as new_digest
from importlib import import_module, util
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .._physical import SpatialCoordinateContract
from ..qualification import ReferenceArtifactManifest
from ..units import METER, MICROMETER, MILLIMETER, MILLISECOND, SECOND
from ._core import (
    DeidentificationEvidence,
    ImageAcquisitionIdentity,
    ImageAxisConvention,
    ImageIndexAffine,
    ImageTimeAxis,
    ImageValueLayout,
    MedicalImageAsset,
    VoxelReference,
)


@runtime_checkable
class _SpatialHeader(Protocol):
    def get_qform(self, *, coded: bool) -> tuple[np.ndarray, int]: ...
    def get_sform(self, *, coded: bool) -> tuple[np.ndarray, int]: ...
    def get_xyzt_units(self) -> tuple[str, str]: ...
    def set_xyzt_units(self, xyz: str, t: str | None = None) -> None: ...
    def get_zooms(self) -> tuple[float, ...]: ...
    def set_zooms(self, zooms: tuple[float, ...]) -> None: ...


@runtime_checkable
class _Image(Protocol):
    affine: np.ndarray
    shape: tuple[int, ...]
    header: _SpatialHeader
    dataobj: object


@runtime_checkable
class _Nibabel(Protocol):
    Nifti1Image: Callable[[np.ndarray, np.ndarray], _Image]

    def load(self, filename: str, /) -> _Image: ...

    def save(self, image: _Image, filename: str, /) -> None: ...


_LENGTH_UNITS = {"mm": MILLIMETER, "micron": MICROMETER, "meter": METER}
_TIME_UNITS = {"sec": SECOND, "msec": MILLISECOND}


def _backend() -> _Nibabel:
    if util.find_spec("nibabel") is None:
        raise ImportError(
            "NIfTI/MGZ interchange requires the optional 'imaging-nifti' extra."
        )
    backend = import_module("nibabel")
    if not isinstance(backend, _Nibabel):
        raise ImportError("The nibabel binding must expose load and save.")
    return backend


def _form(value: tuple[np.ndarray, int], name: str, /) -> np.ndarray | None:
    matrix, code = value
    if int(code) == 0:
        return None
    result = np.asarray(matrix, dtype=float)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} is not a finite homogeneous affine.")
    return result


def _require_use(reference: ReferenceArtifactManifest, intended_use: str, /) -> None:
    requested = {
        "research": {},
        "commercial": {"commercial_use": True},
        "training": {"training_use": True},
        "redistribution": {"redistribution": True},
        "export": {"export": True},
    }
    if intended_use not in requested:
        raise ValueError(
            "intended_use must be research, commercial, training, redistribution, or export."
        )
    reference.require_rights(**requested[intended_use])


def _verify_reference(path: Path, reference: ReferenceArtifactManifest, /) -> None:
    if path.stat().st_size != reference.size_bytes:
        raise ValueError("Image source size does not match its reference manifest.")
    digest = new_digest(reference.checksum_algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != reference.checksum:
        raise ValueError("Image source checksum does not match its reference manifest.")


class NibabelImageProvider:
    """Read normalized medical-image assets without importing nibabel at package import."""

    def read(
        self,
        path: str | Path,
        layout: ImageValueLayout,
        deidentification: DeidentificationEvidence,
        reference: ReferenceArtifactManifest,
        /,
        *,
        asset_id: str,
        modality: str,
        reference_frame: str,
        intended_use: str = "research",
        valid_mask=None,
        time_axis: ImageTimeAxis | None = None,
        acquisition: ImageAcquisitionIdentity | None = None,
        conflict_tolerance: float = 1.0e-5,
    ) -> MedicalImageAsset:
        _require_use(reference, intended_use)
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        _verify_reference(source, reference)
        backend = _backend()
        image = backend.load(str(source))
        if not isinstance(image, _Image):
            raise TypeError("nibabel returned an unsupported image object.")
        spatial_unit, temporal_unit = image.header.get_xyzt_units()
        if spatial_unit not in _LENGTH_UNITS:
            raise ValueError(f"Unsupported image spatial unit {spatial_unit!r}.")
        unit = _LENGTH_UNITS[spatial_unit]
        contract = SpatialCoordinateContract(
            unit,
            coordinate_system="cartesian-ras",
            reference_frame=reference_frame,
        )
        qform = _form(image.header.get_qform(coded=True), "qform")
        sform = _form(image.header.get_sform(coded=True), "sform")
        if qform is None and sform is None:
            fallback = np.asarray(image.affine, dtype=float)
            if fallback.shape != (4, 4) or not np.all(np.isfinite(fallback)):
                raise ValueError("Image has no valid qform, sform, or fallback affine.")
            qform = fallback
        affine = ImageIndexAffine.from_qform_sform(
            qform=qform,
            sform=sform,
            source_frame_id=f"{asset_id}:voxel-index",
            coordinate_contract=contract,
            axis_convention=ImageAxisConvention.RAS,
            voxel_reference=VoxelReference.CENTER,
            conflict_tolerance=conflict_tolerance,
        )
        values = np.asanyarray(image.dataobj)
        if time_axis is not None and time_axis.sample_count > 1:
            if temporal_unit not in _TIME_UNITS:
                raise ValueError(f"Unsupported image temporal unit {temporal_unit!r}.")
            header_time_unit = _TIME_UNITS[temporal_unit]
            interval = time_axis.interval_in(header_time_unit)
            zooms = image.header.get_zooms()
            if (
                interval is None
                or len(zooms) < 4
                or not np.isclose(zooms[3], interval, rtol=1.0e-8, atol=0.0)
            ):
                raise ValueError(
                    "Image header timing and supplied ImageTimeAxis disagree."
                )
        return MedicalImageAsset(
            asset_id,
            modality,
            values,
            affine,
            layout,
            deidentification,
            reference,
            time_axis,
            valid_mask,
            acquisition,
            {"provider": "nibabel"},
            intended_use,
        )

    def write(self, asset: MedicalImageAsset, path: str | Path, /) -> Path:
        if not isinstance(asset, MedicalImageAsset):
            raise TypeError("asset must be MedicalImageAsset.")
        asset.reference.require_rights(export=True)
        target = Path(path).resolve()
        if target.suffix not in (".nii", ".gz") or (
            target.suffix == ".gz" and not target.name.endswith(".nii.gz")
        ):
            raise ValueError("NIfTI export path must end in .nii or .nii.gz.")
        target.parent.mkdir(parents=True, exist_ok=True)
        backend = _backend()
        affine = asset.spatial_affine.to_convention(ImageAxisConvention.RAS)
        image = backend.Nifti1Image(np.asarray(asset.values), np.asarray(affine.matrix))
        spatial_unit = {
            MILLIMETER.unit_id: "mm",
            MICROMETER.unit_id: "micron",
            METER.unit_id: "meter",
        }.get(affine.coordinate_contract.length_unit.unit_id)
        if spatial_unit is None:
            raise ValueError(
                "NIfTI export supports meter, millimeter, or micrometer length units."
            )
        temporal_unit = None
        if asset.time_axis is not None:
            if not asset.time_axis.is_uniform:
                raise ValueError("NIfTI export requires a uniform image time axis.")
            temporal_unit = {
                SECOND.unit_id: "sec",
                MILLISECOND.unit_id: "msec",
            }.get(asset.time_axis.time_unit.unit_id)
            if temporal_unit is None:
                raise ValueError("NIfTI export supports second or millisecond time axes.")
            if asset.time_axis.sample_count > 1:
                interval = asset.time_axis.interval_in(asset.time_axis.time_unit)
                if interval is None:
                    raise RuntimeError("Uniform time axis has no interval.")
                zooms = list(image.header.get_zooms())
                if len(zooms) < 4:
                    raise ValueError("Timed NIfTI values require a fourth sample axis.")
                zooms[3] = interval
                image.header.set_zooms(tuple(zooms))
        image.header.set_xyzt_units(spatial_unit, temporal_unit)
        backend.save(image, str(target))
        restored = backend.load(str(target))
        if not isinstance(restored, _Image):
            raise TypeError("nibabel returned an unsupported image after export.")
        if not np.array_equal(np.asanyarray(restored.dataobj), np.asarray(asset.values)):
            raise RuntimeError("NIfTI export did not preserve image values.")
        if not np.allclose(np.asarray(restored.affine), np.asarray(affine.matrix)):
            raise RuntimeError("NIfTI export did not preserve the spatial affine.")
        return target


__all__ = ["NibabelImageProvider"]
