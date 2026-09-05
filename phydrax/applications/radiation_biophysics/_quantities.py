#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Initial-lesion yield denominators and independent-history sampling uncertainty."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ...units import conversion_factor, derived_unit, JOULE, KILOGRAM, ONE, UnitDefinition
from ._clusters import RadiationClusters
from ._interactions import _nonnegative, PrimaryHistoryKey
from ._lesions import InitialLesionLedger


GRAY = derived_unit("Gy", ((JOULE, 1), (KILOGRAM, -1)))
PER_GRAY = derived_unit("Gy^-1", ((GRAY, -1),))
PER_JOULE = derived_unit("J^-1", ((JOULE, -1),))


@dataclass(frozen=True, slots=True)
class HistoryExposure:
    """Whole scored-mass deposition, not merely energy inside DNA target spheres.

    Base pairs count duplex pairs, not nucleotides, once for the scored ensemble.
    A zero-dose history is retained in sampling uncertainty. The supplied dose
    uncertainty is a one-standard-deviation value in Gy; None means unknown.
    """

    history: PrimaryHistoryKey
    deposited_energy: float
    energy_unit: UnitDefinition
    mass: float
    mass_unit: UnitDefinition
    base_pairs: int
    molecule_count: int
    dose_standard_error_gy: float | None = None

    def __post_init__(self):
        _nonnegative(self.deposited_energy, "scored deposited energy")
        if not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Scored mass must be finite and positive.")
        conversion_factor(self.energy_unit, JOULE)
        conversion_factor(self.mass_unit, KILOGRAM)
        if type(self.base_pairs) is not int or self.base_pairs <= 0:
            raise ValueError("Scored duplex base-pair count must be positive.")
        if type(self.molecule_count) is not int or self.molecule_count <= 0:
            raise ValueError("Scored molecule count must be positive.")
        if self.dose_standard_error_gy is not None:
            _nonnegative(self.dose_standard_error_gy, "dose uncertainty")

    @property
    def mass_kg(self) -> float:
        return self.mass * float(conversion_factor(self.mass_unit, KILOGRAM))

    @property
    def dose_gy(self) -> float:
        return (
            self.deposited_energy
            * float(conversion_factor(self.energy_unit, JOULE))
            / self.mass_kg
        )


@dataclass(frozen=True, slots=True)
class RadiationYield:
    observable: str
    convention: str
    value: float
    unit: UnitDefinition
    history_sampling_standard_error: float | None
    normalization_standard_error: float | None
    total_count: int
    history_count: int
    denominator: float
    history_counts: tuple[tuple[PrimaryHistoryKey, int], ...]


def radiation_yield(
    lesions: InitialLesionLedger,
    clusters: RadiationClusters,
    exposures: tuple[HistoryExposure, ...],
    *,
    observable: str,
    convention: str,
) -> RadiationYield:
    """Ratio-of-sums yield with independent-primary delta-method standard error.

    Supported conventions: per-primary, per-Gy, per-Gy-per-Mbp,
    per-Gy-per-molecule, per-Gy-per-kg. Per-Gy means the explicitly scored
    ensemble; it is not an implicit single-plasmid normalization. Normalization
    uncertainty assumes independent supplied dose errors and exact declared mass,
    molecule and base-pair denominators. Unknown errors stay unknown.
    """
    if clusters.realization_id != lesions.realization_id:
        raise ValueError("Cluster and lesion realizations do not match.")
    if observable not in (
        "lesions",
        "direct",
        "indirect",
        "SSB",
        "SSB-cluster",
        "DSB",
        "base-damage",
    ):
        raise ValueError("Unknown initial-lesion yield observable.")
    if convention not in (
        "per-primary",
        "per-Gy",
        "per-Gy-per-Mbp",
        "per-Gy-per-molecule",
        "per-Gy-per-kg",
    ):
        raise ValueError("Unknown yield normalization convention.")
    by_history = {item.history: item for item in exposures}
    if not by_history or len(by_history) != len(exposures):
        raise ValueError(
            "Every independent primary exposure must be provided exactly once."
        )
    if any(history.source_id != lesions.candidates.source_id for history in by_history):
        raise ValueError("Scored exposures must belong to the lesion source artifact.")
    if len({history.fraction_id for history in by_history}) != 1:
        raise ValueError(
            "Dose fractions cannot be pooled without a separate biological protocol."
        )
    counts = {history: 0 for history in by_history}
    if observable in ("lesions", "direct", "indirect"):
        for lesion in lesions.lesions:
            if lesion.history not in counts:
                raise ValueError("Lesion primary has no scored exposure.")
            counts[lesion.history] += int(
                observable == "lesions" or observable in lesion.causes
            )
    else:
        for cluster in clusters.clusters:
            if cluster.history not in counts:
                raise ValueError("Cluster primary has no scored exposure.")
            counts[cluster.history] += int(cluster.classification == observable)
    histories = sorted(by_history)
    weights = []
    denominators = []
    for history in histories:
        item = by_history[history]
        weight = (
            item.base_pairs / 1e6
            if convention == "per-Gy-per-Mbp"
            else item.molecule_count
            if convention == "per-Gy-per-molecule"
            else item.mass_kg
            if convention == "per-Gy-per-kg"
            else 1.0
        )
        weights.append(weight)
        denominators.append(1.0 if convention == "per-primary" else item.dose_gy * weight)
    denominator = math.fsum(denominators)
    if denominator <= 0:
        raise ValueError("Yield normalization requires positive total exposure.")
    total = sum(counts.values())
    value = total / denominator
    n = len(histories)
    sampling_error = (
        math.sqrt(
            n
            / (n - 1)
            * math.fsum(
                (counts[history] - value * den) ** 2
                for history, den in zip(histories, denominators, strict=True)
            )
        )
        / denominator
        if n > 1
        else None
    )
    if convention == "per-primary":
        normalization_error = 0.0  # Exact declared number of independent primaries.
    elif all(
        by_history[history].dose_standard_error_gy is not None for history in histories
    ):
        denominator_error = math.sqrt(
            math.fsum(
                (by_history[history].dose_standard_error_gy * weight) ** 2
                for history, weight in zip(histories, weights, strict=True)
            )
        )
        normalization_error = abs(value) * denominator_error / denominator
    else:
        normalization_error = None
    unit = (
        ONE
        if convention == "per-primary"
        else PER_JOULE
        if convention == "per-Gy-per-kg"
        else PER_GRAY
    )
    return RadiationYield(
        observable,
        convention,
        value,
        unit,
        sampling_error,
        normalization_error,
        total,
        n,
        denominator,
        tuple((history, counts[history]) for history in histories),
    )
