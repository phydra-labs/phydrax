#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ..foundation import BiospecimenLineage
from ._frame import SpatialCoordinates, SpatialFrame


@dataclass(frozen=True, slots=True)
class SpatialSampleRecord:
    """Host provenance for one assayed tissue section or imaging field.

    Identifiers and biological lineage stay outside numerical PyTrees. The packed
    numeric assay refers to records only through an integer ``sample_index`` leaf.
    """

    sample_id: str
    biospecimen_id: str
    donor_id: str
    section_id: str
    frame: SpatialFrame
    lineage: BiospecimenLineage

    def __post_init__(self):
        for name in ("sample_id", "biospecimen_id", "donor_id", "section_id"):
            value = object.__getattribute__(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if not isinstance(self.frame, SpatialFrame):
            raise TypeError("frame must be a SpatialFrame.")
        if not isinstance(self.lineage, BiospecimenLineage):
            raise TypeError("lineage must be a BiospecimenLineage.")


class SpatialAssayData(StrictModule):
    """Packed differentiable measurements and nondifferentiable sample topology."""

    coordinates: SpatialCoordinates
    values: Array
    valid_spots: Array
    sample_index: Array
    spot_weights: Array
    feature_count: int = eqx.field(static=True)

    def __init__(
        self,
        coordinates: SpatialCoordinates,
        values: Any,
        sample_index: Any,
        /,
        *,
        valid_spots: Any | None = None,
        spot_weights: Any | None = None,
    ):
        if not isinstance(coordinates, SpatialCoordinates):
            raise TypeError("coordinates must be SpatialCoordinates.")
        if coordinates.values.ndim != 2:
            raise ValueError(
                "Spatial assay coordinates must have shape (spot, coordinate)."
            )
        values_ = jnp.asarray(values, dtype=float)
        if values_.ndim == 1:
            values_ = values_[:, None]
        if values_.ndim != 2 or int(values_.shape[0]) != int(coordinates.values.shape[0]):
            raise ValueError("values must have shape (spot, feature).")
        spots = int(values_.shape[0])
        if spots < 1 or int(values_.shape[1]) < 1:
            raise ValueError("Spatial assays require at least one spot and feature.")
        sample_index_ = jnp.asarray(sample_index, dtype=jnp.int32)
        if sample_index_.shape != (spots,):
            raise ValueError(f"sample_index must have shape {(spots,)}.")
        valid_ = (
            jnp.ones((spots,), dtype=bool)
            if valid_spots is None
            else jnp.asarray(valid_spots, dtype=bool)
        )
        if valid_.shape != (spots,):
            raise ValueError(f"valid_spots must have shape {(spots,)}.")
        weights_ = (
            valid_.astype(values_.dtype)
            if spot_weights is None
            else jnp.asarray(spot_weights, dtype=values_.dtype)
        )
        if weights_.shape != (spots,):
            raise ValueError(f"spot_weights must have shape {(spots,)}.")
        host_values = np.asarray(values_)
        host_coordinates = np.asarray(coordinates.values)
        host_index = np.asarray(sample_index_)
        host_valid = np.asarray(valid_)
        host_weights = np.asarray(weights_)
        if np.any(~np.isfinite(host_values[host_valid])):
            raise ValueError("Valid spatial assay values must be finite.")
        if np.any(~np.isfinite(host_coordinates[host_valid])):
            raise ValueError("Valid spatial assay coordinates must be finite.")
        if np.any(~np.isfinite(host_weights)) or np.any(host_weights < 0.0):
            raise ValueError("spot_weights must be finite and non-negative.")
        if np.any(host_valid & (host_weights <= 0.0)):
            raise ValueError("Every valid spot must have positive sampling weight.")
        if np.any(host_index < 0):
            raise ValueError("sample_index entries must be non-negative.")
        if np.any(~host_valid & (host_weights != 0.0)):
            raise ValueError("Invalid spots must carry zero spot weight.")
        self.coordinates = coordinates
        self.values = values_
        self.valid_spots = valid_
        self.sample_index = sample_index_
        self.spot_weights = weights_
        self.feature_count = int(values_.shape[1])


@dataclass(frozen=True, slots=True)
class SpatialAssay:
    """Host-linked packed spatial assay supporting unequal section densities."""

    records: tuple[SpatialSampleRecord, ...]
    data: SpatialAssayData

    def __init__(
        self,
        records: Sequence[SpatialSampleRecord],
        data: SpatialAssayData,
        /,
    ):
        records_ = tuple(records)
        if not records_ or any(
            not isinstance(record, SpatialSampleRecord) for record in records_
        ):
            raise TypeError(
                "records must be a non-empty sequence of SpatialSampleRecord."
            )
        if len({record.sample_id for record in records_}) != len(records_):
            raise ValueError("Spatial sample identifiers must be unique.")
        if not isinstance(data, SpatialAssayData):
            raise TypeError("data must be SpatialAssayData.")
        sample_index = np.asarray(data.sample_index)
        valid = np.asarray(data.valid_spots)
        if np.any(sample_index >= len(records_)):
            raise ValueError("sample_index refers to a missing spatial sample record.")
        populated = np.bincount(sample_index[valid], minlength=len(records_))
        if np.any(populated == 0):
            raise ValueError(
                "Every spatial sample record must own at least one valid spot."
            )
        coordinate_code = np.asarray(data.coordinates.frame_code)
        for record in records_:
            if not np.array_equal(
                coordinate_code, np.asarray(record.frame.code, dtype=np.uint8)
            ):
                raise ValueError(
                    "Packed SpatialAssayData must use the shared frame of every sample record."
                )
        object.__setattr__(self, "records", records_)
        object.__setattr__(self, "data", data)

    @property
    def sample_count(self) -> int:
        return len(self.records)

    @property
    def donor_count(self) -> int:
        return len({record.donor_id for record in self.records})

    @property
    def section_count(self) -> int:
        return len({record.section_id for record in self.records})

    def donor_index(self) -> Array:
        """Return a packed spot-to-donor relation without placing donor strings in JAX."""
        donors = tuple(dict.fromkeys(record.donor_id for record in self.records))
        sample_to_donor = jnp.asarray(
            [donors.index(record.donor_id) for record in self.records], dtype=jnp.int32
        )
        return sample_to_donor[self.data.sample_index]

    def section_index(self) -> Array:
        """Return the packed spot-to-section relation used for graph isolation."""
        sections = tuple(
            dict.fromkeys((record.donor_id, record.section_id) for record in self.records)
        )
        sample_to_section = jnp.asarray(
            [
                sections.index((record.donor_id, record.section_id))
                for record in self.records
            ],
            dtype=jnp.int32,
        )
        return sample_to_section[self.data.sample_index]


__all__ = ["SpatialAssay", "SpatialAssayData", "SpatialSampleRecord"]
