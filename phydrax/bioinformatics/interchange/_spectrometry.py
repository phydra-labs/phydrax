#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterable, Mapping
from enum import IntEnum, IntFlag
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax._strict import StrictModule
from phydrax.bioinformatics.foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from phydrax.bioinformatics.spectrometry._acquisition import (
    AcquisitionKind,
    AcquisitionMetadata,
    AcquisitionRun,
    PrecursorBatch,
)
from phydrax.bioinformatics.spectrometry._spectrum import (
    IonMobilityUnit,
    IonPolarity,
    MassSpectrum,
    SpectrometryUnits,
    SpectrumBatch,
    SpectrumRepresentation,
)


class MzMLReadStatus(IntEnum):
    """Status of lowering a host mzML-like read record."""

    SUCCESS = 0
    POINT_CAPACITY_EXCEEDED = 1
    PRECURSOR_CAPACITY_EXCEEDED = 2
    SCAN_CAPACITY_EXCEEDED = 3
    MISSING_SPECTRAL_ARRAY = 4
    ARRAY_SHAPE_MISMATCH = 5
    MIXED_RUN_SEMANTICS = 6
    NONFINITE = 7


class MzMLReadEvidence(IntFlag):
    """mzML read metadata preserved by lowering."""

    NONE = 0
    SPECTRAL_ARRAYS = 1
    PRECURSOR = 2
    CHIMERIC_PRECURSOR = 4
    ION_MOBILITY = 8
    ACQUISITION_METADATA = 16


_RECORD_CONTRACT = BioinformaticsMethodContract(
    "mzML-like record read lowering",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Recognized mzML CV-style read fields are copied into native bounded arrays "
        "with explicit units and acquisition defaults."
    ),
    truncation_statement=(
        "Point and precursor capacities are preflighted; overflow returns failure "
        "and never truncates a record."
    ),
    capacity_semantics="Point and precursor capacities are fixed by the lowering plan.",
    assumptions=("Input mappings use Pyteomics-compatible mzML read keys.",),
    nondifferentiable_outputs=("all outputs",),
)

_RUN_CONTRACT = BioinformaticsMethodContract(
    "mzML read run lowering",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement=(
        "Record order is retained and one homogeneous native run is assembled after "
        "every capacity and semantics check passes."
    ),
    truncation_statement="Scan, point, and precursor overflow return failure without a partial run.",
    capacity_semantics="Run storage is exactly scan_capacity × point_capacity with a fixed precursor axis.",
    assumptions=(
        "One native run has a common representation, polarity, and unit contract.",
    ),
    nondifferentiable_outputs=("all outputs",),
)


class MzMLLoweringPlan(StrictModule):
    """Explicit fixed capacities and defaults for mzML read lowering."""

    units: SpectrometryUnits = eqx.field(static=True)
    acquisition: AcquisitionMetadata = eqx.field(static=True)
    default_representation: SpectrumRepresentation = eqx.field(static=True)
    default_polarity: IonPolarity = eqx.field(static=True)
    scan_capacity: int = eqx.field(static=True)
    point_capacity: int = eqx.field(static=True)
    precursor_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        scan_capacity: int,
        point_capacity: int,
        precursor_capacity: int = 1,
        units: SpectrometryUnits | None = None,
        acquisition: AcquisitionMetadata | None = None,
        default_representation: SpectrumRepresentation = SpectrumRepresentation.CENTROID,
        default_polarity: IonPolarity = IonPolarity.UNKNOWN,
    ):
        scans = int(scan_capacity)
        points = int(point_capacity)
        precursors = int(precursor_capacity)
        if scans < 1 or points < 1 or precursors < 1:
            raise ValueError("All mzML lowering capacities must be positive.")
        resolved_units = SpectrometryUnits() if units is None else units
        resolved_acquisition = (
            AcquisitionMetadata(AcquisitionKind.FULL_SCAN)
            if acquisition is None
            else acquisition
        )
        if not isinstance(resolved_units, SpectrometryUnits):
            raise TypeError("units must be SpectrometryUnits.")
        if not isinstance(resolved_acquisition, AcquisitionMetadata):
            raise TypeError("acquisition must be AcquisitionMetadata.")
        self.units = resolved_units
        self.acquisition = resolved_acquisition
        self.default_representation = SpectrumRepresentation(default_representation)
        self.default_polarity = IonPolarity(default_polarity)
        self.scan_capacity = scans
        self.point_capacity = points
        self.precursor_capacity = precursors


class MzMLRecordReadResult(StrictModule):
    """One native spectrum and precursor row from a host mzML-like record."""

    spectrum: MassSpectrum
    precursors: PrecursorBatch
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


class MzMLRunReadResult(StrictModule):
    """One native acquisition run lowered from read records."""

    run: AcquisitionRun
    record_valid_mask: Array
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)


def _flag(record: Mapping[str, Any], key: str, /) -> bool:
    return key in record and record[key] is not False


def _representation(
    record: Mapping[str, Any], default: SpectrumRepresentation, /
) -> SpectrumRepresentation:
    if _flag(record, "profile spectrum"):
        return SpectrumRepresentation.PROFILE
    if _flag(record, "centroid spectrum"):
        return SpectrumRepresentation.CENTROID
    return default


def _polarity(record: Mapping[str, Any], default: IonPolarity, /) -> IonPolarity:
    positive = _flag(record, "positive scan")
    negative = _flag(record, "negative scan")
    if positive and negative:
        raise ValueError("An mzML record cannot be both positive and negative polarity.")
    if positive:
        return IonPolarity.POSITIVE
    if negative:
        return IonPolarity.NEGATIVE
    return default


def _scan_entry(record: Mapping[str, Any], /) -> Mapping[str, Any]:
    scan_list = record.get("scanList", {})
    if not isinstance(scan_list, Mapping):
        return {}
    scans = scan_list.get("scan", ())
    if not isinstance(scans, (list, tuple)) or not scans:
        return {}
    first = scans[0]
    return first if isinstance(first, Mapping) else {}


def _retention_time(record: Mapping[str, Any], /) -> float:
    scan = _scan_entry(record)
    return float(scan.get("scan start time", record.get("scan start time", 0.0)))


def _mobility_array(
    record: Mapping[str, Any],
    point_count: int,
    units: SpectrometryUnits,
    /,
) -> np.ndarray | None:
    keys_by_unit = {
        IonMobilityUnit.DRIFT_TIME_MILLISECOND: (
            "ion mobility drift time array",
            "mean ion mobility drift time array",
        ),
        IonMobilityUnit.INVERSE_REDUCED_MOBILITY: (
            "mean inverse reduced ion mobility array",
            "inverse reduced ion mobility array",
        ),
        IonMobilityUnit.COMPENSATION_VOLT: ("compensation voltage array",),
        IonMobilityUnit.NONE: (),
    }
    for key in keys_by_unit[units.ion_mobility]:
        if key in record:
            values = np.asarray(record[key])
            if values.shape != (point_count,):
                raise ValueError(
                    "Ion-mobility and spectral arrays must have equal length."
                )
            return values
    scan = _scan_entry(record)
    scalar_keys = {
        IonMobilityUnit.DRIFT_TIME_MILLISECOND: "ion mobility drift time",
        IonMobilityUnit.INVERSE_REDUCED_MOBILITY: "inverse reduced ion mobility",
        IonMobilityUnit.COMPENSATION_VOLT: "compensation voltage",
        IonMobilityUnit.NONE: "",
    }
    scalar_key = scalar_keys[units.ion_mobility]
    if scalar_key and scalar_key in scan:
        return np.full((point_count,), float(scan[scalar_key]))
    return None


def _precursor_rows(
    record: Mapping[str, Any], /
) -> list[tuple[float, int, float, float, float]]:
    precursor_list = record.get("precursorList", {})
    if not isinstance(precursor_list, Mapping):
        return []
    precursor_items = precursor_list.get("precursor", ())
    if not isinstance(precursor_items, (list, tuple)):
        return []
    rows: list[tuple[float, int, float, float, float]] = []
    for precursor in precursor_items:
        if not isinstance(precursor, Mapping):
            continue
        isolation = precursor.get("isolationWindow", {})
        if not isinstance(isolation, Mapping):
            isolation = {}
        selected_list = precursor.get("selectedIonList", {})
        if not isinstance(selected_list, Mapping):
            selected_list = {}
        selected_ions = selected_list.get("selectedIon", ())
        if not isinstance(selected_ions, (list, tuple)):
            selected_ions = ()
        activation = precursor.get("activation", {})
        if not isinstance(activation, Mapping):
            activation = {}
        for ion in selected_ions:
            if not isinstance(ion, Mapping) or "selected ion m/z" not in ion:
                continue
            rows.append(
                (
                    float(ion["selected ion m/z"]),
                    int(ion.get("charge state", ion.get("possible charge state", 0))),
                    float(isolation.get("isolation window lower offset", 0.0)),
                    float(isolation.get("isolation window upper offset", 0.0)),
                    float(activation.get("collision energy", 0.0)),
                )
            )
    return rows


def _empty_record_result(
    plan: MzMLLoweringPlan,
    status: MzMLReadStatus,
    /,
    *,
    representation: SpectrumRepresentation | None = None,
    polarity: IonPolarity | None = None,
) -> MzMLRecordReadResult:
    spectrum = MassSpectrum(
        np.zeros((plan.point_capacity,), dtype=float),
        np.zeros((plan.point_capacity,), dtype=float),
        active_mask=np.zeros((plan.point_capacity,), dtype=bool),
        scan_id=-1,
        representation=plan.default_representation
        if representation is None
        else representation,
        polarity=plan.default_polarity if polarity is None else polarity,
        units=plan.units,
    )
    return MzMLRecordReadResult(
        spectrum=spectrum,
        precursors=PrecursorBatch.empty(1, plan.precursor_capacity),
        valid=jnp.asarray(False),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence=jnp.asarray(int(MzMLReadEvidence.NONE), dtype=jnp.uint32),
        method_contract=_RECORD_CONTRACT,
    )


def lower_mzml_record(
    record: Mapping[str, Any],
    plan: MzMLLoweringPlan,
    /,
) -> MzMLRecordReadResult:
    """Lower a host mzML-like read record without retaining the host mapping."""
    if not isinstance(record, Mapping):
        raise TypeError("record must be a host mapping.")
    if not isinstance(plan, MzMLLoweringPlan):
        raise TypeError("plan must be MzMLLoweringPlan.")
    representation = _representation(record, plan.default_representation)
    polarity = _polarity(record, plan.default_polarity)
    if "m/z array" not in record or "intensity array" not in record:
        return _empty_record_result(
            plan,
            MzMLReadStatus.MISSING_SPECTRAL_ARRAY,
            representation=representation,
            polarity=polarity,
        )
    mz = np.asarray(record["m/z array"])
    signal = np.asarray(record["intensity array"])
    if mz.ndim != 1 or signal.shape != mz.shape:
        return _empty_record_result(
            plan,
            MzMLReadStatus.ARRAY_SHAPE_MISMATCH,
            representation=representation,
            polarity=polarity,
        )
    if mz.size > plan.point_capacity:
        return _empty_record_result(
            plan,
            MzMLReadStatus.POINT_CAPACITY_EXCEEDED,
            representation=representation,
            polarity=polarity,
        )
    precursor_rows = _precursor_rows(record)
    if len(precursor_rows) > plan.precursor_capacity:
        return _empty_record_result(
            plan,
            MzMLReadStatus.PRECURSOR_CAPACITY_EXCEEDED,
            representation=representation,
            polarity=polarity,
        )
    mobility = _mobility_array(record, int(mz.size), plan.units)
    retention_time = _retention_time(record)
    precursor_values = np.asarray(precursor_rows, dtype=float)
    finite = (
        np.all(np.isfinite(mz))
        and np.all(np.isfinite(signal))
        and (mobility is None or np.all(np.isfinite(mobility)))
        and np.isfinite(retention_time)
        and np.all(np.isfinite(precursor_values))
    )
    if not finite:
        return _empty_record_result(
            plan,
            MzMLReadStatus.NONFINITE,
            representation=representation,
            polarity=polarity,
        )
    dtype = np.result_type(mz.dtype, signal.dtype, np.float32)
    padded_mz = np.zeros((plan.point_capacity,), dtype=dtype)
    padded_signal = np.zeros((plan.point_capacity,), dtype=dtype)
    point_mask = np.zeros((plan.point_capacity,), dtype=bool)
    padded_mz[: mz.size] = mz
    padded_signal[: signal.size] = signal
    point_mask[: mz.size] = True
    if mobility is None:
        padded_mobility = None
        mobility_mask = None
    else:
        padded_mobility = np.zeros((plan.point_capacity,), dtype=dtype)
        padded_mobility[: mz.size] = mobility
        mobility_mask = point_mask.copy()
    scan_id = int(record.get("index", record.get("scan index", -1)))
    level = int(record.get("ms level", 1))
    spectrum = MassSpectrum(
        padded_mz,
        padded_signal,
        active_mask=point_mask,
        ion_mobility=padded_mobility,
        ion_mobility_mask=mobility_mask,
        scan_id=scan_id,
        retention_time=retention_time,
        representation=representation,
        polarity=polarity,
        ms_level=level,
        units=plan.units,
    )
    shape = (1, plan.precursor_capacity)
    precursor_mz = np.zeros(shape, dtype=dtype)
    charge = np.zeros(shape, dtype=np.int32)
    lower = np.zeros(shape, dtype=dtype)
    upper = np.zeros(shape, dtype=dtype)
    energy = np.zeros(shape, dtype=dtype)
    precursor_mask = np.zeros(shape, dtype=bool)
    for index, (value, state, low, high, collision) in enumerate(precursor_rows):
        precursor_mz[0, index] = value
        charge[0, index] = state
        lower[0, index] = low
        upper[0, index] = high
        energy[0, index] = collision
        precursor_mask[0, index] = True
    precursors = PrecursorBatch(
        precursor_mz,
        charge,
        lower,
        upper,
        energy,
        precursor_mask,
    )
    evidence = jnp.asarray(int(MzMLReadEvidence.SPECTRAL_ARRAYS), dtype=jnp.uint32)
    evidence = evidence | jnp.where(
        len(precursor_rows) > 0, int(MzMLReadEvidence.PRECURSOR), 0
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        len(precursor_rows) > 1,
        int(MzMLReadEvidence.CHIMERIC_PRECURSOR),
        0,
    ).astype(jnp.uint32)
    evidence = evidence | jnp.where(
        mobility is not None, int(MzMLReadEvidence.ION_MOBILITY), 0
    ).astype(jnp.uint32)
    evidence = evidence | jnp.asarray(
        int(MzMLReadEvidence.ACQUISITION_METADATA), dtype=jnp.uint32
    )
    return MzMLRecordReadResult(
        spectrum=spectrum,
        precursors=precursors,
        valid=jnp.asarray(True),
        status=jnp.asarray(int(MzMLReadStatus.SUCCESS), dtype=jnp.int32),
        evidence=evidence,
        method_contract=_RECORD_CONTRACT,
    )


def _empty_run(
    plan: MzMLLoweringPlan,
    status: MzMLReadStatus,
    /,
) -> MzMLRunReadResult:
    shape = (plan.scan_capacity, plan.point_capacity)
    spectra = SpectrumBatch(
        np.zeros(shape, dtype=float),
        np.zeros(shape, dtype=float),
        point_mask=np.zeros(shape, dtype=bool),
        scan_mask=np.zeros((plan.scan_capacity,), dtype=bool),
        scan_ids=np.zeros((plan.scan_capacity,), dtype=np.int64),
        ms_levels=np.zeros((plan.scan_capacity,), dtype=np.int32),
        retention_time=np.zeros((plan.scan_capacity,), dtype=float),
        representation=plan.default_representation,
        polarity=plan.default_polarity,
        units=plan.units,
    )
    run = AcquisitionRun(
        spectra,
        PrecursorBatch.empty(plan.scan_capacity, plan.precursor_capacity),
        plan.acquisition,
    )
    return MzMLRunReadResult(
        run=run,
        record_valid_mask=jnp.zeros((plan.scan_capacity,), dtype=bool),
        valid=jnp.asarray(False),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        evidence=jnp.asarray(int(MzMLReadEvidence.NONE), dtype=jnp.uint32),
        method_contract=_RUN_CONTRACT,
    )


def lower_mzml_records(
    records: Iterable[Mapping[str, Any]],
    plan: MzMLLoweringPlan,
    /,
) -> MzMLRunReadResult:
    """Lower ordered mzML-like read records to one homogeneous native run."""
    if not isinstance(plan, MzMLLoweringPlan):
        raise TypeError("plan must be MzMLLoweringPlan.")
    items = tuple(records)
    if len(items) > plan.scan_capacity:
        return _empty_run(plan, MzMLReadStatus.SCAN_CAPACITY_EXCEEDED)
    lowered = tuple(lower_mzml_record(record, plan) for record in items)
    for result in lowered:
        if not bool(np.asarray(result.valid)):
            return _empty_run(plan, MzMLReadStatus(int(np.asarray(result.status))))
    representation = (
        plan.default_representation if not lowered else lowered[0].spectrum.representation
    )
    polarity = plan.default_polarity if not lowered else lowered[0].spectrum.polarity
    if any(
        result.spectrum.representation != representation
        or result.spectrum.polarity != polarity
        for result in lowered
    ):
        return _empty_run(plan, MzMLReadStatus.MIXED_RUN_SEMANTICS)
    scan_shape = (plan.scan_capacity, plan.point_capacity)
    mz = np.zeros(scan_shape, dtype=float)
    signal = np.zeros(scan_shape, dtype=float)
    point_mask = np.zeros(scan_shape, dtype=bool)
    mobility = np.zeros(scan_shape, dtype=float)
    mobility_mask = np.zeros(scan_shape, dtype=bool)
    scan_mask = np.zeros((plan.scan_capacity,), dtype=bool)
    scan_ids = np.zeros((plan.scan_capacity,), dtype=np.int64)
    levels = np.zeros((plan.scan_capacity,), dtype=np.int32)
    times = np.zeros((plan.scan_capacity,), dtype=float)
    precursor_shape = (plan.scan_capacity, plan.precursor_capacity)
    precursor_mz = np.zeros(precursor_shape, dtype=float)
    charge = np.zeros(precursor_shape, dtype=np.int32)
    lower = np.zeros(precursor_shape, dtype=float)
    upper = np.zeros(precursor_shape, dtype=float)
    energy = np.zeros(precursor_shape, dtype=float)
    precursor_mask = np.zeros(precursor_shape, dtype=bool)
    evidence = 0
    for index, result in enumerate(lowered):
        spectrum = result.spectrum
        mz[index] = np.asarray(spectrum.mass_to_charge)
        signal[index] = np.asarray(spectrum.intensity)
        point_mask[index] = np.asarray(spectrum.active_mask)
        mobility[index] = np.asarray(spectrum.ion_mobility)
        mobility_mask[index] = np.asarray(spectrum.ion_mobility_mask)
        scan_mask[index] = True
        scan_ids[index] = int(np.asarray(spectrum.scan_id))
        levels[index] = spectrum.ms_level
        times[index] = float(np.asarray(spectrum.retention_time))
        precursor_mz[index] = np.asarray(result.precursors.mass_to_charge[0])
        charge[index] = np.asarray(result.precursors.charge[0])
        lower[index] = np.asarray(result.precursors.isolation_lower[0])
        upper[index] = np.asarray(result.precursors.isolation_upper[0])
        energy[index] = np.asarray(result.precursors.collision_energy[0])
        precursor_mask[index] = np.asarray(result.precursors.active_mask[0])
        evidence |= int(np.asarray(result.evidence))
    spectra = SpectrumBatch(
        mz,
        signal,
        point_mask=point_mask,
        scan_mask=scan_mask,
        scan_ids=scan_ids,
        ms_levels=levels,
        retention_time=times,
        ion_mobility=mobility
        if plan.units.ion_mobility != IonMobilityUnit.NONE
        else None,
        ion_mobility_mask=mobility_mask
        if plan.units.ion_mobility != IonMobilityUnit.NONE
        else None,
        representation=representation,
        polarity=polarity,
        units=plan.units,
    )
    precursors = PrecursorBatch(
        precursor_mz,
        charge,
        lower,
        upper,
        energy,
        precursor_mask,
    )
    run = AcquisitionRun(spectra, precursors, plan.acquisition)
    record_valid = np.zeros((plan.scan_capacity,), dtype=bool)
    record_valid[: len(items)] = True
    return MzMLRunReadResult(
        run=run,
        record_valid_mask=jnp.asarray(record_valid),
        valid=jnp.asarray(True),
        status=jnp.asarray(int(MzMLReadStatus.SUCCESS), dtype=jnp.int32),
        evidence=jnp.asarray(evidence, dtype=jnp.uint32),
        method_contract=_RUN_CONTRACT,
    )


def read_pyteomics_mzml(
    source: str | Path,
    plan: MzMLLoweringPlan,
    /,
) -> MzMLRunReadResult:
    """Read mzML through the optional Pyteomics reader and lower immediately."""
    if importlib.util.find_spec("pyteomics") is None:
        raise ModuleNotFoundError(
            "read_pyteomics_mzml requires the bioinformatics-spectrometry extra."
        )
    mzml_module = importlib.import_module("pyteomics.mzml")
    with mzml_module.MzML(str(source)) as reader:
        records = tuple(reader)
    return lower_mzml_records(records, plan)


__all__ = [
    "MzMLLoweringPlan",
    "MzMLReadEvidence",
    "MzMLReadStatus",
    "MzMLRecordReadResult",
    "MzMLRunReadResult",
    "lower_mzml_record",
    "lower_mzml_records",
    "read_pyteomics_mzml",
]
