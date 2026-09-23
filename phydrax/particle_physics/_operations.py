#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..measurement._operations import (
    DataQualityAnnotation,
    ExposureRecord,
    OperationalCoordinate,
    ResolvedConditionSnapshot,
)
from ..units import ENERGY, LENGTH, UnitDefinition


def _identifier(value: str, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


class BeamConditionSnapshot(StrictModule, NonTrainableState):
    species_pdg_ids: tuple[int, int] = eqx.field(static=True)
    beam_energies: tuple[float, float] = eqx.field(static=True)
    bunch_intensities: tuple[float, float] = eqx.field(static=True)
    crossing_angle: float = eqx.field(static=True)
    beam_spot_mean: Array
    beam_spot_covariance: Array
    energy_unit: UnitDefinition = eqx.field(static=True)
    length_unit: UnitDefinition = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        species_pdg_ids: tuple[int, int],
        beam_energies: tuple[float, float],
        bunch_intensities: tuple[float, float],
        crossing_angle: float,
        beam_spot_mean,
        beam_spot_covariance,
        energy_unit: UnitDefinition,
        length_unit: UnitDefinition,
        source_id: str,
    ):
        int32 = np.iinfo(np.int32)
        if len(species_pdg_ids) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < int32.min
            or value > int32.max
            for value in species_pdg_ids
        ):
            raise ValueError("Beam species must contain two signed-int32 PDG identities.")
        species = tuple(species_pdg_ids)
        energies = tuple(float(value) for value in beam_energies)
        intensities = tuple(float(value) for value in bunch_intensities)
        angle = float(crossing_angle)
        mean_host = np.asarray(beam_spot_mean, dtype=np.float64)
        covariance_host = np.asarray(beam_spot_covariance, dtype=np.float64)
        mean = jnp.asarray(mean_host)
        covariance = jnp.asarray(covariance_host)
        if len(species) != 2 or len(energies) != 2 or len(intensities) != 2:
            raise ValueError("Beam snapshots require two beam entries.")
        if any(
            not math.isfinite(value) or value <= 0.0 for value in energies + intensities
        ) or not math.isfinite(angle):
            raise ValueError(
                "Beam energies/intensities must be positive finite and angle finite."
            )
        if (
            mean_host.shape != (4,)
            or covariance_host.shape != (4, 4)
            or np.any(~np.isfinite(mean_host))
            or np.any(~np.isfinite(covariance_host))
        ):
            raise ValueError(
                "Beam-spot mean/covariance must be finite four-dimensional values."
            )
        if not np.allclose(covariance_host, covariance_host.T):
            raise ValueError("Beam-spot covariance must be symmetric.")
        tolerance = (
            512.0
            * np.finfo(covariance_host.dtype).eps
            * max(1.0, float(np.linalg.norm(covariance_host, ord=2)))
        )
        if np.min(np.linalg.eigvalsh(covariance_host)) < -tolerance:
            raise ValueError("Beam-spot covariance must be positive semidefinite.")
        if not isinstance(energy_unit, UnitDefinition) or energy_unit.dimension != ENERGY:
            raise ValueError("energy_unit must have physical energy dimension.")
        if not isinstance(length_unit, UnitDefinition) or length_unit.dimension != LENGTH:
            raise ValueError("length_unit must have physical length dimension.")
        self.species_pdg_ids = species
        self.beam_energies = energies
        self.bunch_intensities = intensities
        self.crossing_angle = angle
        self.beam_spot_mean = mean
        self.beam_spot_covariance = covariance
        self.energy_unit = energy_unit
        self.length_unit = length_unit
        self.source_id = _identifier(source_id, "Source ID")
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "beam-condition-snapshot",
                "species": list(species),
                "energies": list(energies),
                "intensities": list(intensities),
                "crossing_angle": angle,
                "mean": mean.tolist(),
                "covariance": covariance.tolist(),
                "units": [self.energy_unit.unit_id, self.length_unit.unit_id],
                "source": self.source_id,
            }
        )


class HEPRunContext(StrictModule, NonTrainableState):
    coordinate: OperationalCoordinate
    beam: BeamConditionSnapshot
    conditions: ResolvedConditionSnapshot
    data_quality: DataQualityAnnotation
    exposures: tuple[ExposureRecord, ...]
    campaign_id: str = eqx.field(static=True)
    stream_id: str = eqx.field(static=True)
    context_id: str = eqx.field(static=True)

    def __init__(
        self,
        coordinate: OperationalCoordinate,
        beam: BeamConditionSnapshot,
        conditions: ResolvedConditionSnapshot,
        data_quality: DataQualityAnnotation,
        exposures: Sequence[ExposureRecord],
        /,
        *,
        campaign_id: str,
        stream_id: str,
    ):
        if not isinstance(coordinate, OperationalCoordinate):
            raise TypeError("coordinate must be OperationalCoordinate.")
        if not isinstance(beam, BeamConditionSnapshot):
            raise TypeError("beam must be BeamConditionSnapshot.")
        if (
            not isinstance(conditions, ResolvedConditionSnapshot)
            or conditions.coordinate.coordinate_id != coordinate.coordinate_id
        ):
            raise ValueError("conditions must resolve the exact run coordinate.")
        if not isinstance(
            data_quality, DataQualityAnnotation
        ) or not data_quality.interval.contains(coordinate):
            raise ValueError("data_quality must contain the run coordinate.")
        exposures_ = tuple(exposures)
        if not exposures_ or any(
            not isinstance(value, ExposureRecord) for value in exposures_
        ):
            raise TypeError("exposures must contain typed non-empty records.")
        if any(not value.interval.contains(coordinate) for value in exposures_):
            raise ValueError("Every exposure must contain the run coordinate.")
        exposure_ids = tuple(value.exposure_id for value in exposures_)
        if len(set(exposure_ids)) != len(exposure_ids):
            raise ValueError("Run exposures must be unique.")
        self.coordinate = coordinate
        self.beam = beam
        self.conditions = conditions
        self.data_quality = data_quality
        self.exposures = tuple(sorted(exposures_, key=lambda value: value.exposure_id))
        self.campaign_id = _identifier(campaign_id, "Campaign ID")
        self.stream_id = _identifier(stream_id, "Stream ID")
        self.context_id = canonical_fingerprint(
            {
                "kind": "hep-run-context",
                "coordinate": coordinate.coordinate_id,
                "beam": beam.snapshot_id,
                "conditions": conditions.snapshot_id,
                "data_quality": data_quality.annotation_id,
                "exposures": list(exposure_ids),
                "campaign": self.campaign_id,
                "stream": self.stream_id,
            }
        )

    @property
    def scientifically_admitted(self) -> bool:
        return self.conditions.successful and self.data_quality.certified


class ProcessNormalization(StrictModule, NonTrainableState):
    process_id: str = eqx.field(static=True)
    attempted_count: int = eqx.field(static=True)
    generated_count: int = eqx.field(static=True)
    accepted_count: int = eqx.field(static=True)
    positive_count: int = eqx.field(static=True)
    negative_count: int = eqx.field(static=True)
    sum_weights: float = eqx.field(static=True)
    sum_absolute_weights: float = eqx.field(static=True)
    sum_squared_weights: float = eqx.field(static=True)
    cross_section: float = eqx.field(static=True)
    cross_section_uncertainty: float = eqx.field(static=True)
    cross_section_unit_id: str = eqx.field(static=True)
    provider_id: str = eqx.field(static=True)
    normalization_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        process_id: str,
        attempted_count: int,
        generated_count: int,
        accepted_count: int,
        positive_count: int,
        negative_count: int,
        sum_weights: float,
        sum_absolute_weights: float,
        sum_squared_weights: float,
        cross_section: float,
        cross_section_uncertainty: float,
        cross_section_unit_id: str,
        provider_id: str,
    ):
        counts = tuple(
            map(
                int,
                (
                    attempted_count,
                    generated_count,
                    accepted_count,
                    positive_count,
                    negative_count,
                ),
            )
        )
        values = tuple(
            map(
                float,
                (
                    sum_weights,
                    sum_absolute_weights,
                    sum_squared_weights,
                    cross_section,
                    cross_section_uncertainty,
                ),
            )
        )
        if (
            any(value < 0 for value in counts)
            or not counts[0] >= counts[1] >= counts[2]
            or counts[3] + counts[4] > counts[1]
        ):
            raise ValueError("Process normalization counts are inconsistent.")
        if (
            any(not math.isfinite(value) for value in values)
            or values[1] < abs(values[0])
            or values[2] < 0.0
            or values[3] < 0.0
            or values[4] < 0.0
        ):
            raise ValueError("Process normalization values are invalid.")
        self.process_id = _identifier(process_id, "Process ID")
        (
            self.attempted_count,
            self.generated_count,
            self.accepted_count,
            self.positive_count,
            self.negative_count,
        ) = counts
        (
            self.sum_weights,
            self.sum_absolute_weights,
            self.sum_squared_weights,
            self.cross_section,
            self.cross_section_uncertainty,
        ) = values
        self.cross_section_unit_id = _identifier(
            cross_section_unit_id, "Cross-section unit ID"
        )
        self.provider_id = _identifier(provider_id, "Provider ID")
        self.normalization_id = canonical_fingerprint(
            {
                "kind": "hep-process-normalization",
                "process": self.process_id,
                "counts": list(counts),
                "values": list(values),
                "unit": self.cross_section_unit_id,
                "provider": self.provider_id,
            }
        )


def merge_process_normalizations(
    values: Sequence[ProcessNormalization], /
) -> ProcessNormalization:
    """Deterministically merge disjoint shards of one exact generator process."""
    records = tuple(values)
    if not records or any(
        not isinstance(value, ProcessNormalization) for value in records
    ):
        raise TypeError("values must contain ProcessNormalization records.")
    reference = records[0]
    if any(
        (value.process_id, value.cross_section_unit_id, value.provider_id)
        != (reference.process_id, reference.cross_section_unit_id, reference.provider_id)
        for value in records[1:]
    ):
        raise ValueError("Only identical process/provider/unit shards can be merged.")
    inverse_variances = tuple(
        0.0
        if value.cross_section_uncertainty == 0.0
        else 1.0 / value.cross_section_uncertainty**2
        for value in records
    )
    total_precision = sum(inverse_variances)
    if total_precision > 0.0:
        cross_section = (
            sum(
                value.cross_section * weight
                for value, weight in zip(records, inverse_variances, strict=True)
            )
            / total_precision
        )
        uncertainty = math.sqrt(1.0 / total_precision)
    else:
        cross_sections = {value.cross_section for value in records}
        if len(cross_sections) != 1:
            raise ValueError(
                "Exact cross sections disagree across zero-uncertainty shards."
            )
        cross_section = reference.cross_section
        uncertainty = 0.0
    return ProcessNormalization(
        process_id=reference.process_id,
        attempted_count=sum(value.attempted_count for value in records),
        generated_count=sum(value.generated_count for value in records),
        accepted_count=sum(value.accepted_count for value in records),
        positive_count=sum(value.positive_count for value in records),
        negative_count=sum(value.negative_count for value in records),
        sum_weights=sum(value.sum_weights for value in records),
        sum_absolute_weights=sum(value.sum_absolute_weights for value in records),
        sum_squared_weights=sum(value.sum_squared_weights for value in records),
        cross_section=cross_section,
        cross_section_uncertainty=uncertainty,
        cross_section_unit_id=reference.cross_section_unit_id,
        provider_id=reference.provider_id,
    )


__all__ = [
    "BeamConditionSnapshot",
    "HEPRunContext",
    "ProcessNormalization",
    "merge_process_normalizations",
]
