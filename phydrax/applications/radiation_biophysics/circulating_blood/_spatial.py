#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit spatial dose-rate mixtures for circulating-blood compartments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real

import numpy as np

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....imaging import LabelVolume, MedicalImageAsset
from ....measurement import (
    QualityFlag,
    QuantitySpec,
    SpatialSamplingKind,
    ValueKind,
)
from ....units import conversion_factor, GRAY_PER_SECOND
from ._dose import _require_dose_rate_quantity, DoseRateInterval
from ._model import _identifier


@dataclass(frozen=True, slots=True)
class PreparedSpatialCompartmentMixture:
    """Compartment dose rates reduced from one exact dose/label lattice."""

    compartment_ids: tuple[str, ...]
    voxel_weights: np.ndarray
    valid_mask: np.ndarray
    dose_rates_gy_per_s: np.ndarray
    standard_uncertainties_gy_per_s: np.ndarray | None
    quantity: QuantitySpec
    quality_flags: tuple[QualityFlag, ...]
    dose_asset_id: str
    label_volume_id: str
    mixture_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "mixture_id",
            canonical_fingerprint(
                {
                    "kind": "prepared-spatial-circulating-blood-mixture",
                    "compartments": list(self.compartment_ids),
                    "voxel_weights": array_tree_fingerprint(self.voxel_weights),
                    "valid": array_tree_fingerprint(self.valid_mask),
                    "dose_rates_gy_per_s": array_tree_fingerprint(
                        self.dose_rates_gy_per_s
                    ),
                    "standard_uncertainties_gy_per_s": None
                    if self.standard_uncertainties_gy_per_s is None
                    else array_tree_fingerprint(self.standard_uncertainties_gy_per_s),
                    "quantity": self.quantity.quantity_id,
                    "quality_flags": [value.flag_id for value in self.quality_flags],
                    "dose_asset": self.dose_asset_id,
                    "label_volume": self.label_volume_id,
                }
            ),
        )

    def interval(self, start_s: float, end_s: float, /) -> DoseRateInterval:
        """Create one constant schedule interval without inventing uncertainty."""

        return DoseRateInterval(
            start_s,
            end_s,
            self.quantity,
            self.dose_rates_gy_per_s,
            self.standard_uncertainties_gy_per_s,
        )


def _compartment_ids(values: Sequence[str], /) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError("compartment_ids must be a sequence, not one string.")
    identifiers = tuple(_identifier(value, "compartment_id") for value in values)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("compartment_ids must be non-empty and unique.")
    return identifiers


def _weight_row(
    values: Mapping[str, Real],
    compartments: tuple[str, ...],
    label_id: str,
    /,
) -> np.ndarray:
    if not isinstance(values, Mapping):
        raise TypeError(f"Mixture weights for label {label_id!r} must be a mapping.")
    if set(values) != set(compartments):
        raise ValueError(
            f"Mixture weights for label {label_id!r} must explicitly name every "
            "compartment and no others."
        )
    row_values: list[float] = []
    for compartment in compartments:
        value = values[compartment]
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("Spatial mixture weights must be real numbers.")
        row_values.append(float(value))
    row = np.asarray(row_values, dtype=float)
    tolerance = 128.0 * np.finfo(row.dtype).eps
    if np.any(~np.isfinite(row)) or np.any(row < 0.0):
        raise ValueError("Spatial mixture weights must be finite and nonnegative.")
    if not np.isclose(np.sum(row), 1.0, atol=tolerance, rtol=0.0):
        raise ValueError("Spatial mixture weights must be normalized to one per label.")
    row.setflags(write=False)
    return row


def prepare_spatial_compartment_mixture(
    dose_rate: MedicalImageAsset,
    labels: LabelVolume,
    compartment_ids: Sequence[str],
    label_weights: Mapping[str, Mapping[str, Real]],
    /,
) -> PreparedSpatialCompartmentMixture:
    """Reduce a scalar dose-rate image by an explicit label-to-compartment mixture.

    Dose and labels must have exactly the same content-addressed lattice support. No
    registration, interpolation, affine tolerance, label inference, or DVH sampling is
    performed. Each ontology label carries a complete nonnegative normalized mixture.
    """

    if not isinstance(dose_rate, MedicalImageAsset):
        raise TypeError("dose_rate must be a MedicalImageAsset.")
    if not isinstance(labels, LabelVolume):
        raise TypeError("labels must be a LabelVolume.")
    compartments = _compartment_ids(compartment_ids)
    _require_dose_rate_quantity(dose_rate.quantity)
    if dose_rate.time_axis is not None or dose_rate.values.ndim != 3:
        raise ValueError("Spatial dose rate must be one static three-dimensional image.")
    if dose_rate.layout.kind is not ValueKind.REAL_SCALAR:
        raise ValueError("Spatial dose rate must use a real-scalar image layout.")
    if dose_rate.sampling.spatial_kind is not SpatialSamplingKind.CELL_AVERAGE:
        raise ValueError(
            "Spatial compartment mixtures require cell-average dose-rate samples."
        )
    if dose_rate.support.support_id != labels.asset.support.support_id:
        raise ValueError(
            "Dose-rate and label images must share the exact spatial affine and lattice "
            "support; implicit registration or resampling is refused."
        )
    if not isinstance(label_weights, Mapping):
        raise TypeError("label_weights must be a mapping.")
    ontology_ids = tuple(value.label_id for value in labels.ontology.labels)
    if set(label_weights) != set(ontology_ids):
        raise ValueError(
            "label_weights must explicitly cover every ontology label and no others."
        )
    rows = {
        label.label_id: _weight_row(
            label_weights[label.label_id], compartments, label.label_id
        )
        for label in labels.ontology.labels
    }
    values = np.asarray(dose_rate.values, dtype=float)
    valid = np.asarray(dose_rate.valid_mask, dtype=bool) & np.asarray(
        labels.asset.valid_mask, dtype=bool
    )
    if not np.any(valid):
        raise ValueError("Dose/label support has no jointly valid voxels.")
    if np.any(~np.isfinite(values[valid])) or np.any(values[valid] < 0.0):
        raise ValueError("Valid spatial dose-rate values must be finite and nonnegative.")
    voxel_weights = np.zeros(values.shape + (len(compartments),), dtype=float)
    label_values = np.asarray(labels.asset.values)
    for label in labels.ontology.labels:
        voxel_weights[label_values == label.value] = rows[label.label_id]
    weighted_support = voxel_weights * valid[..., None]
    denominators = np.sum(weighted_support, axis=(0, 1, 2))
    if np.any(denominators <= 0.0):
        missing = tuple(
            compartments[index] for index in np.flatnonzero(denominators <= 0.0).tolist()
        )
        raise ValueError(
            f"Every compartment needs positive spatial mixture support; missing {missing}."
        )
    normalized = weighted_support / denominators
    safe_values = np.where(valid, values, 0.0)
    dose_rates = np.sum(normalized * safe_values[..., None], axis=(0, 1, 2))
    if dose_rate.uncertainty is None:
        uncertainty = None
    else:
        scale = float(conversion_factor(dose_rate.uncertainty.unit, GRAY_PER_SECOND))
        voxel_uncertainty = np.asarray(dose_rate.uncertainty.values, dtype=float) * scale
        uncertainty = np.sqrt(
            np.sum(
                np.square(normalized * voxel_uncertainty[..., None]),
                axis=(0, 1, 2),
            )
        )
        uncertainty.setflags(write=False)
    voxel_weights.setflags(write=False)
    valid = np.array(valid, copy=True)
    valid.setflags(write=False)
    dose_rates.setflags(write=False)
    return PreparedSpatialCompartmentMixture(
        compartments,
        voxel_weights,
        valid,
        dose_rates,
        uncertainty,
        dose_rate.quantity,
        tuple(dose_rate.quality_flags),
        dose_rate.asset_id,
        labels.label_volume_id,
    )


__all__ = [
    "PreparedSpatialCompartmentMixture",
    "prepare_spatial_compartment_mixture",
]
