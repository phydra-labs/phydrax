#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum, IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from numpy.typing import DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ._spectrum import SpectrumBatch


class AcquisitionKind(IntEnum):
    """Mass-spectrometry acquisition strategy."""

    FULL_SCAN = 0
    DATA_DEPENDENT = 1
    DATA_INDEPENDENT = 2
    PARALLEL_REACTION_MONITORING = 3
    SELECTED_REACTION_MONITORING = 4


class MassAnalyzer(IntEnum):
    """Analyzer class needed to interpret acquisition resolution."""

    UNKNOWN = 0
    QUADRUPOLE = 1
    TIME_OF_FLIGHT = 2
    ORBITRAP = 3
    ION_TRAP = 4
    FOURIER_TRANSFORM_ION_CYCLOTRON = 5


class DissociationMethod(IntEnum):
    """Precursor dissociation method."""

    NONE = 0
    CID = 1
    HCD = 2
    ETD = 3
    ECD = 4
    UVPD = 5


class AcquisitionStatus(IntEnum):
    """Status of a bounded acquisition query."""

    SUCCESS = 0
    MISSING_SPECTRUM = 1
    NONFINITE_QUERY = 2


class AcquisitionEvidence(IntFlag):
    """Evidence retained by a bounded acquisition query."""

    NONE = 0
    SPECTRUM_PRESENT = 1
    PRECURSOR_PRESENT = 2
    CHIMERIC_PRECURSOR = 4
    ION_MOBILITY_PRESENT = 8


_LOOKUP_CONTRACT = BioinformaticsMethodContract(
    "bounded spectrum lookup",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.ALMOST_EVERYWHERE,
    OutputKind.STRUCTURED,
    conditioning_statement="Exact equality lookup over a fixed scan capacity.",
    truncation_statement="No truncation; absent scan identifiers produce an observable failure status.",
    capacity_semantics="Runtime is bounded by scan, point, and precursor capacities.",
    assumptions=("Active scan identifiers are unique.",),
    nondifferentiable_outputs=("status", "evidence", "scan_index", "precursor_charge"),
)


class AcquisitionMetadata(StrictModule):
    """Static semantic identity of one acquisition method."""

    acquisition_kind: AcquisitionKind = eqx.field(static=True)
    analyzer: MassAnalyzer = eqx.field(static=True)
    dissociation: DissociationMethod = eqx.field(static=True)
    resolution_at_reference: float = eqx.field(static=True)
    reference_mass_to_charge: float = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        acquisition_kind: AcquisitionKind,
        /,
        *,
        analyzer: MassAnalyzer = MassAnalyzer.UNKNOWN,
        dissociation: DissociationMethod = DissociationMethod.NONE,
        resolution_at_reference: float = 0.0,
        reference_mass_to_charge: float = 200.0,
    ):
        kind = AcquisitionKind(acquisition_kind)
        mass_analyzer = MassAnalyzer(analyzer)
        fragmentation = DissociationMethod(dissociation)
        resolution = float(resolution_at_reference)
        reference = float(reference_mass_to_charge)
        if not np.isfinite(resolution) or resolution < 0.0:
            raise ValueError("resolution_at_reference must be finite and nonnegative.")
        if not np.isfinite(reference) or reference <= 0.0:
            raise ValueError("reference_mass_to_charge must be finite and positive.")
        self.acquisition_kind = kind
        self.analyzer = mass_analyzer
        self.dissociation = fragmentation
        self.resolution_at_reference = resolution
        self.reference_mass_to_charge = reference
        self.method_id = canonical_fingerprint(
            {
                "kind": "spectrometry-acquisition",
                "acquisition_kind": int(kind),
                "analyzer": int(mass_analyzer),
                "dissociation": int(fragmentation),
                "resolution_at_reference": resolution,
                "reference_mass_to_charge": reference,
            }
        )


class PrecursorBatch(StrictModule):
    """Fixed scan-by-precursor capacity metadata, including chimeric scans."""

    mass_to_charge: Array
    charge: Array
    isolation_lower: Array
    isolation_upper: Array
    collision_energy: Array
    active_mask: Array
    scan_capacity: int = eqx.field(static=True)
    precursor_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        mass_to_charge: ArrayLike,
        charge: ArrayLike,
        isolation_lower: ArrayLike,
        isolation_upper: ArrayLike,
        collision_energy: ArrayLike,
        active_mask: ArrayLike,
        /,
    ):
        mz = np.asarray(mass_to_charge)
        charges = np.asarray(charge)
        lower = np.asarray(isolation_lower)
        upper = np.asarray(isolation_upper)
        energy = np.asarray(collision_energy)
        mask = np.asarray(active_mask, dtype=bool)
        if mz.ndim != 2 or mz.size == 0 or mz.shape[1] == 0:
            raise ValueError(
                "mass_to_charge must have positive scan and precursor capacities."
            )
        if any(
            value.shape != mz.shape for value in (charges, lower, upper, energy, mask)
        ):
            raise ValueError("All precursor arrays must have the same shape.")
        if not np.issubdtype(charges.dtype, np.integer):
            raise TypeError("charge must contain integers; zero denotes unknown charge.")
        dtype = np.result_type(
            mz.dtype, lower.dtype, upper.dtype, energy.dtype, np.float32
        )
        mz = mz.astype(dtype, copy=False)
        lower = lower.astype(dtype, copy=False)
        upper = upper.astype(dtype, copy=False)
        energy = energy.astype(dtype, copy=False)
        for row in mask:
            count = int(np.count_nonzero(row))
            if not np.all(row[:count]) or np.any(row[count:]):
                raise ValueError("Each precursor mask row must be a left-prefix mask.")
        if np.any(~np.isfinite(mz[mask])) or np.any(mz[mask] <= 0.0):
            raise ValueError(
                "Active precursor mass-to-charge values must be finite and positive."
            )
        if np.any(~np.isfinite(lower[mask])) or np.any(~np.isfinite(upper[mask])):
            raise ValueError("Active isolation bounds must be finite.")
        if np.any(lower[mask] < 0.0) or np.any(upper[mask] < 0.0):
            raise ValueError("Isolation offsets must be nonnegative.")
        if np.any(~np.isfinite(energy[mask])) or np.any(energy[mask] < 0.0):
            raise ValueError("Active collision energies must be finite and nonnegative.")
        if np.any(np.abs(charges[mask]) > 64):
            raise ValueError("Active precursor charge magnitude cannot exceed 64.")
        for value in (mz, charges, lower, upper, energy):
            if np.any(value[~mask] != 0):
                raise ValueError("Inactive precursor entries must be zero padding.")
        self.mass_to_charge = jnp.asarray(mz)
        self.charge = jnp.asarray(charges, dtype=jnp.int32)
        self.isolation_lower = jnp.asarray(lower)
        self.isolation_upper = jnp.asarray(upper)
        self.collision_energy = jnp.asarray(energy)
        self.active_mask = jnp.asarray(mask)
        self.scan_capacity = int(mz.shape[0])
        self.precursor_capacity = int(mz.shape[1])

    @classmethod
    def empty(
        cls,
        scan_capacity: int,
        precursor_capacity: int = 1,
        /,
        *,
        dtype: DTypeLike = float,
    ) -> PrecursorBatch:
        scans = int(scan_capacity)
        precursors = int(precursor_capacity)
        if scans < 1 or precursors < 1:
            raise ValueError("scan_capacity and precursor_capacity must be positive.")
        shape = (scans, precursors)
        return cls(
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=np.int32),
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=dtype),
            np.zeros(shape, dtype=bool),
        )


class AcquisitionRun(StrictModule):
    """Native bounded spectra, precursors, and acquisition semantics."""

    spectra: SpectrumBatch
    precursors: PrecursorBatch
    metadata: AcquisitionMetadata = eqx.field(static=True)
    run_id: Array

    def __init__(
        self,
        spectra: SpectrumBatch,
        precursors: PrecursorBatch,
        metadata: AcquisitionMetadata,
        /,
        *,
        run_id: int | ArrayLike = 0,
    ):
        if not isinstance(spectra, SpectrumBatch):
            raise TypeError("spectra must be a SpectrumBatch.")
        if not isinstance(precursors, PrecursorBatch):
            raise TypeError("precursors must be a PrecursorBatch.")
        if not isinstance(metadata, AcquisitionMetadata):
            raise TypeError("metadata must be AcquisitionMetadata.")
        if spectra.scan_capacity != precursors.scan_capacity:
            raise ValueError("Spectra and precursor scan capacities must match.")
        precursor_scan_mask = np.any(np.asarray(precursors.active_mask), axis=1)
        if np.any(precursor_scan_mask & ~np.asarray(spectra.scan_mask)):
            raise ValueError("Precursors cannot refer to inactive scans.")
        if np.any(precursor_scan_mask & (np.asarray(spectra.ms_levels) < 2)):
            raise ValueError("Precursor metadata requires MS level 2 or greater.")
        identifier = np.asarray(run_id)
        if identifier.shape != () or not np.issubdtype(identifier.dtype, np.integer):
            raise TypeError("run_id must be an integer scalar.")
        self.spectra = spectra
        self.precursors = precursors
        self.metadata = metadata
        self.run_id = jnp.asarray(identifier, dtype=jnp.int64)


class SpectrumLookupResult(StrictModule):
    """One bounded lookup payload with explicit missing and chimeric evidence."""

    mass_to_charge: Array
    intensity: Array
    active_mask: Array
    ion_mobility: Array
    ion_mobility_mask: Array
    scan_id: Array
    scan_index: Array
    ms_level: Array
    retention_time: Array
    precursor_mass_to_charge: Array
    precursor_charge: Array
    precursor_isolation_lower: Array
    precursor_isolation_upper: Array
    precursor_collision_energy: Array
    precursor_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def lookup_spectrum(
    run: AcquisitionRun,
    scan_id: int | ArrayLike,
    /,
) -> SpectrumLookupResult:
    """Look up a scan without data-dependent allocation or silent substitution."""
    if not isinstance(run, AcquisitionRun):
        raise TypeError("run must be an AcquisitionRun.")
    query = jnp.asarray(scan_id)
    if query.shape != () or not jnp.issubdtype(query.dtype, jnp.integer):
        raise TypeError("scan_id must be an integer scalar.")
    matches = run.spectra.scan_mask & (run.spectra.scan_ids == query)
    found = jnp.any(matches)
    index = jnp.argmax(matches.astype(jnp.int32))
    safe_index = jnp.where(found, index, 0)
    point_mask = run.spectra.point_mask[safe_index] & found
    precursor_mask = run.precursors.active_mask[safe_index] & found
    mobility_mask = run.spectra.ion_mobility_mask[safe_index] & found
    precursor_count = jnp.sum(precursor_mask, dtype=jnp.int32)
    evidence = jnp.where(
        found,
        jnp.asarray(int(AcquisitionEvidence.SPECTRUM_PRESENT), dtype=jnp.uint32),
        jnp.asarray(int(AcquisitionEvidence.NONE), dtype=jnp.uint32),
    )
    evidence = evidence | jnp.where(
        precursor_count > 0,
        jnp.asarray(int(AcquisitionEvidence.PRECURSOR_PRESENT), dtype=jnp.uint32),
        jnp.asarray(0, dtype=jnp.uint32),
    )
    evidence = evidence | jnp.where(
        precursor_count > 1,
        jnp.asarray(int(AcquisitionEvidence.CHIMERIC_PRECURSOR), dtype=jnp.uint32),
        jnp.asarray(0, dtype=jnp.uint32),
    )
    evidence = evidence | jnp.where(
        jnp.any(mobility_mask),
        jnp.asarray(int(AcquisitionEvidence.ION_MOBILITY_PRESENT), dtype=jnp.uint32),
        jnp.asarray(0, dtype=jnp.uint32),
    )
    zeros = jnp.zeros_like(run.spectra.mass_to_charge[safe_index])
    precursor_zeros = jnp.zeros_like(run.precursors.mass_to_charge[safe_index])
    return SpectrumLookupResult(
        mass_to_charge=jnp.where(
            point_mask, run.spectra.mass_to_charge[safe_index], zeros
        ),
        intensity=jnp.where(point_mask, run.spectra.intensity[safe_index], zeros),
        active_mask=point_mask,
        ion_mobility=jnp.where(
            mobility_mask, run.spectra.ion_mobility[safe_index], zeros
        ),
        ion_mobility_mask=mobility_mask,
        scan_id=jnp.where(found, run.spectra.scan_ids[safe_index], -1),
        scan_index=jnp.where(found, safe_index, -1).astype(jnp.int32),
        ms_level=jnp.where(found, run.spectra.ms_levels[safe_index], 0),
        retention_time=jnp.where(found, run.spectra.retention_time[safe_index], 0.0),
        precursor_mass_to_charge=jnp.where(
            precursor_mask,
            run.precursors.mass_to_charge[safe_index],
            precursor_zeros,
        ),
        precursor_charge=jnp.where(precursor_mask, run.precursors.charge[safe_index], 0),
        precursor_isolation_lower=jnp.where(
            precursor_mask,
            run.precursors.isolation_lower[safe_index],
            precursor_zeros,
        ),
        precursor_isolation_upper=jnp.where(
            precursor_mask,
            run.precursors.isolation_upper[safe_index],
            precursor_zeros,
        ),
        precursor_collision_energy=jnp.where(
            precursor_mask,
            run.precursors.collision_energy[safe_index],
            precursor_zeros,
        ),
        precursor_mask=precursor_mask,
        valid=found,
        status=jnp.where(
            found,
            int(AcquisitionStatus.SUCCESS),
            int(AcquisitionStatus.MISSING_SPECTRUM),
        ).astype(jnp.int32),
        evidence=evidence,
        method_contract=_LOOKUP_CONTRACT,
    )


__all__ = [
    "AcquisitionEvidence",
    "AcquisitionKind",
    "AcquisitionMetadata",
    "AcquisitionRun",
    "AcquisitionStatus",
    "DissociationMethod",
    "MassAnalyzer",
    "PrecursorBatch",
    "SpectrumLookupResult",
    "lookup_spectrum",
]
