#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Strict research-only CT-number calibration to material fields."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..measurement import (
    DataStage,
    DerivationRecord,
    IndependentStandardUncertainty,
    ValueKind,
)
from ..qualification import ReferenceArtifactManifest
from ..units import conversion_factor, KILOGRAM_PER_CUBIC_METER, ONE
from ._asset import ImageFieldSpec
from ._core import MedicalImageAsset


def _readonly(value, name: str, /) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use object dtype.")
    array.setflags(write=False)
    return array


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result or result != value:
        raise ValueError(f"{name} must be non-empty canonical text.")
    return result


@dataclass(frozen=True, slots=True)
class HUCalibrationAnchor:
    """One exact HU anchor with density and ordered material fractions."""

    hu: float
    density_kg_m3: float
    material_fractions: np.ndarray
    anchor_id: str = field(init=False)

    def __post_init__(self) -> None:
        hu = float(self.hu)
        density = float(self.density_kg_m3)
        fractions = _readonly(self.material_fractions, "material_fractions")
        if not np.isfinite(hu):
            raise ValueError("hu must be finite.")
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("density_kg_m3 must be finite and strictly positive.")
        if fractions.ndim != 1 or fractions.size < 1:
            raise ValueError("material_fractions must be a nonempty rank-one array.")
        if np.any(~np.isfinite(fractions)) or np.any(fractions < 0.0):
            raise ValueError("material_fractions must be finite and non-negative.")
        tolerance = 64.0 * np.finfo(fractions.dtype).eps
        if not np.isclose(np.sum(fractions), 1.0, atol=tolerance, rtol=0.0):
            raise ValueError("material_fractions must sum to one.")
        object.__setattr__(self, "hu", hu)
        object.__setattr__(self, "density_kg_m3", density)
        object.__setattr__(self, "material_fractions", fractions)
        object.__setattr__(
            self,
            "anchor_id",
            canonical_fingerprint(
                {
                    "kind": "hu-calibration-anchor",
                    "hu": hu,
                    "density_kg_m3": density,
                    "material_fractions": array_tree_fingerprint(fractions),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class HUToMaterialCalibration:
    """Piecewise-linear HU calibration with a closed, non-extrapolating support."""

    calibration_name: str
    material_ids: tuple[str, ...]
    anchors: tuple[HUCalibrationAnchor, ...]
    reference: ReferenceArtifactManifest
    calibration_id: str = field(init=False)

    def __post_init__(self) -> None:
        name = _identifier(self.calibration_name, "calibration_name")
        materials = tuple(
            _identifier(material_id, "material_id") for material_id in self.material_ids
        )
        anchors = tuple(self.anchors)
        if not materials or len(materials) != len(set(materials)):
            raise ValueError("material_ids must be nonempty and unique.")
        if len(anchors) < 2 or any(
            not isinstance(anchor, HUCalibrationAnchor) for anchor in anchors
        ):
            raise TypeError(
                "anchors must contain at least two HUCalibrationAnchor values."
            )
        if not isinstance(self.reference, ReferenceArtifactManifest):
            raise TypeError("reference must be ReferenceArtifactManifest.")
        reference_id = self.reference.require_rights()
        if any(
            anchor.material_fractions.shape != (len(materials),) for anchor in anchors
        ):
            raise ValueError("Every anchor must contain one fraction per material_id.")
        hu = np.asarray([anchor.hu for anchor in anchors])
        if np.any(np.diff(hu) <= 0.0):
            raise ValueError("Anchor HU values must be strictly increasing.")
        object.__setattr__(self, "calibration_name", name)
        object.__setattr__(self, "material_ids", materials)
        object.__setattr__(self, "anchors", anchors)
        object.__setattr__(
            self,
            "calibration_id",
            canonical_fingerprint(
                {
                    "kind": "hu-to-material-calibration",
                    "name": name,
                    "materials": list(materials),
                    "anchors": [anchor.anchor_id for anchor in anchors],
                    "reference": reference_id,
                }
            ),
        )

    @property
    def hu_support(self) -> tuple[float, float]:
        return self.anchors[0].hu, self.anchors[-1].hu


@dataclass(frozen=True, slots=True)
class HUToMaterialResult:
    """Density and ordered fraction images on the source image support."""

    density: MedicalImageAsset
    material_fractions: MedicalImageAsset
    calibration_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.density, MedicalImageAsset) or not isinstance(
            self.material_fractions, MedicalImageAsset
        ):
            raise TypeError(
                "density and material_fractions must be MedicalImageAsset values."
            )
        if self.density.support.support_id != self.material_fractions.support.support_id:
            raise ValueError("Calibrated density and fractions must share one support.")
        object.__setattr__(
            self, "calibration_id", _identifier(self.calibration_id, "calibration_id")
        )

    @property
    def support(self):
        return self.density.support


def _calibrated_values(
    values: np.ndarray,
    valid: np.ndarray,
    calibration: HUToMaterialCalibration,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hu = np.asarray([anchor.hu for anchor in calibration.anchors], dtype=np.float64)
    density = np.asarray(
        [anchor.density_kg_m3 for anchor in calibration.anchors], dtype=np.float64
    )
    fractions = np.stack(
        [anchor.material_fractions for anchor in calibration.anchors], axis=0
    )
    selected = values[valid]
    if np.any(~np.isfinite(selected)):
        raise ValueError("Valid CT-number samples must be finite.")
    if np.any((selected < hu[0]) | (selected > hu[-1])):
        raise ValueError(
            "CT-number samples lie outside the closed calibration support; "
            "clamping and extrapolation are not permitted."
        )
    interval = np.searchsorted(hu, selected, side="right") - 1
    interval = np.where(selected == hu[-1], hu.size - 2, interval)
    alpha = (selected - hu[interval]) / (hu[interval + 1] - hu[interval])
    density_values = np.zeros(values.shape, dtype=np.result_type(values, density))
    fraction_values = np.zeros(
        values.shape + (len(calibration.material_ids),),
        dtype=np.result_type(values, fractions),
    )
    density_values[valid] = (1.0 - alpha) * density[interval] + alpha * density[
        interval + 1
    ]
    fraction_values[valid] = (1.0 - alpha[:, None]) * fractions[interval] + alpha[
        :, None
    ] * fractions[interval + 1]
    density_slopes = (density[interval + 1] - density[interval]) / (
        hu[interval + 1] - hu[interval]
    )
    return density_values, fraction_values, density_slopes


def apply_hu_calibration(
    asset: MedicalImageAsset,
    calibration: HUToMaterialCalibration,
    /,
) -> HUToMaterialResult:
    """Apply a strict HU calibration without clamping or extrapolation."""

    if not isinstance(asset, MedicalImageAsset):
        raise TypeError("asset must be MedicalImageAsset.")
    if not isinstance(calibration, HUToMaterialCalibration):
        raise TypeError("calibration must be HUToMaterialCalibration.")
    if asset.quantity.quantity_kind != "ct_number":
        raise ValueError("HU calibration requires quantity_kind='ct_number'.")
    if asset.layout.kind is not ValueKind.REAL_SCALAR or asset.layout.component_shape:
        raise ValueError("HU calibration requires real scalar CT-number samples.")
    hu_scale = float(conversion_factor(asset.quantity.unit, ONE))
    values = np.asarray(asset.values, dtype=np.float64) * hu_scale
    valid = np.asarray(asset.valid_mask, dtype=np.bool_)
    density_values, fraction_values, density_slopes = _calibrated_values(
        values, valid, calibration
    )

    density_uncertainty = None
    if asset.uncertainty is not None:
        uncertainty_scale = float(conversion_factor(asset.uncertainty.unit, ONE))
        source_uncertainty = np.broadcast_to(
            np.asarray(asset.uncertainty.values, dtype=np.float64) * uncertainty_scale,
            values.shape,
        )
        density_standard = np.zeros(values.shape, dtype=density_values.dtype)
        density_standard[valid] = np.abs(density_slopes) * source_uncertainty[valid]
        density_uncertainty = IndependentStandardUncertainty(
            density_standard, KILOGRAM_PER_CUBIC_METER
        )

    derivation = DerivationRecord(
        asset.derivation.origin,
        DataStage.CALIBRATED,
        parent_ids=(asset.content_id,),
        transformation_id=f"hu-to-material:{calibration.calibration_id}",
        calibration_ids=(calibration.calibration_id,),
    )
    metadata = dict(asset.metadata)
    metadata["hu_calibration_id"] = calibration.calibration_id
    metadata["hu_calibration_reference_id"] = calibration.reference.manifest_id
    metadata["material_fraction_uncertainty"] = (
        "not-represented-shared-hu-covariance"
        if asset.uncertainty is not None
        else "unquantified"
    )
    references = (
        asset.references
        if calibration.reference.manifest_id
        in {value.manifest_id for value in asset.references}
        else asset.references + (calibration.reference,)
    )
    density_spec = ImageFieldSpec.named(
        "mass-density",
        KILOGRAM_PER_CUBIC_METER,
        ValueKind.REAL_SCALAR,
        quantity_kind="mass_density",
        compatibility_key="imaging.mass_density",
    )
    fraction_spec = ImageFieldSpec.named(
        "material-fraction",
        ONE,
        ValueKind.PROBABILITY,
        (len(calibration.material_ids),),
        component_labels=calibration.material_ids,
        quantity_kind="material_fraction",
        compatibility_key="imaging.material_fraction",
    )
    common = {
        "time_axis": asset.time_axis,
        "valid_mask": valid,
        "acquisition": asset.acquisition,
        "metadata": metadata,
        "intended_use": asset.intended_use,
    }
    density_asset = MedicalImageAsset(
        f"{asset.asset_id}:density:{calibration.calibration_id[:12]}",
        asset.modality,
        density_values,
        asset.spatial_affine,
        density_spec,
        asset.deidentification,
        references,
        derivation,
        uncertainty=density_uncertainty,
        quality_flags=asset.quality_flags,
        **common,
    )
    fraction_asset = MedicalImageAsset(
        f"{asset.asset_id}:material-fractions:{calibration.calibration_id[:12]}",
        asset.modality,
        fraction_values,
        asset.spatial_affine,
        fraction_spec,
        asset.deidentification,
        references,
        derivation,
        uncertainty=None,
        quality_flags=asset.quality_flags,
        **common,
    )
    if (
        density_asset.support.support_id != asset.support.support_id
        or fraction_asset.support.support_id != asset.support.support_id
    ):
        raise RuntimeError("HU calibration changed the medical-image support.")
    return HUToMaterialResult(density_asset, fraction_asset, calibration.calibration_id)


__all__ = [
    "HUCalibrationAnchor",
    "HUToMaterialCalibration",
    "HUToMaterialResult",
    "apply_hu_calibration",
]
