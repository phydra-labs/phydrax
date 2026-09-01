#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule


class SpectrumRepresentation(IntEnum):
    """Sampling semantics of a mass spectrum."""

    PROFILE = 0
    CENTROID = 1


class IonPolarity(IntEnum):
    """Ionization polarity retained from the acquisition."""

    UNKNOWN = 0
    POSITIVE = 1
    NEGATIVE = -1


class MassToChargeUnit(IntEnum):
    """Supported mass-to-charge coordinate units."""

    MZ = 0
    THOMSON = 1


class IntensityUnit(IntEnum):
    """Supported detector-intensity units."""

    COUNTS = 0
    ARBITRARY = 1
    IONS_PER_SECOND = 2


class TimeUnit(IntEnum):
    """Supported chromatographic time units."""

    SECOND = 0
    MINUTE = 1


class IonMobilityUnit(IntEnum):
    """Supported ion-mobility coordinates."""

    NONE = 0
    DRIFT_TIME_MILLISECOND = 1
    INVERSE_REDUCED_MOBILITY = 2
    COMPENSATION_VOLT = 3


class SpectrometryUnits(StrictModule):
    """Static unit identity for one native spectrometry payload."""

    mass_to_charge: MassToChargeUnit = eqx.field(static=True)
    intensity: IntensityUnit = eqx.field(static=True)
    time: TimeUnit = eqx.field(static=True)
    ion_mobility: IonMobilityUnit = eqx.field(static=True)
    units_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        mass_to_charge: MassToChargeUnit = MassToChargeUnit.MZ,
        intensity: IntensityUnit = IntensityUnit.COUNTS,
        time: TimeUnit = TimeUnit.SECOND,
        ion_mobility: IonMobilityUnit = IonMobilityUnit.NONE,
    ):
        mz_unit = MassToChargeUnit(mass_to_charge)
        intensity_unit = IntensityUnit(intensity)
        time_unit = TimeUnit(time)
        mobility_unit = IonMobilityUnit(ion_mobility)
        self.mass_to_charge = mz_unit
        self.intensity = intensity_unit
        self.time = time_unit
        self.ion_mobility = mobility_unit
        self.units_id = canonical_fingerprint(
            {
                "kind": "spectrometry-units",
                "mass_to_charge": int(mz_unit),
                "intensity": int(intensity_unit),
                "time": int(time_unit),
                "ion_mobility": int(mobility_unit),
            }
        )

    def compatible_with(
        self,
        other: SpectrometryUnits,
        /,
        *,
        require_intensity: bool = False,
        require_mobility: bool = False,
    ) -> bool:
        if not isinstance(other, SpectrometryUnits):
            return False
        return (
            self.mass_to_charge == other.mass_to_charge
            and self.time == other.time
            and (not require_intensity or self.intensity == other.intensity)
            and (not require_mobility or self.ion_mobility == other.ion_mobility)
        )


def _prefix_mask(mask: np.ndarray, /) -> bool:
    count = int(np.count_nonzero(mask))
    return bool(np.all(mask[:count]) and not np.any(mask[count:]))


def _validate_axis(
    coordinate: ArrayLike,
    intensity: ArrayLike,
    active_mask: ArrayLike | None,
    /,
    *,
    rank: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinate_host = np.asarray(coordinate)
    intensity_host = np.asarray(intensity)
    if coordinate_host.ndim != rank or coordinate_host.shape != intensity_host.shape:
        raise ValueError(
            "coordinate and intensity must have the same requested-rank shape."
        )
    if coordinate_host.size == 0 or coordinate_host.shape[-1] == 0:
        raise ValueError("A bounded spectrum axis must have positive capacity.")
    dtype = np.result_type(coordinate_host.dtype, intensity_host.dtype, np.float32)
    coordinate_host = coordinate_host.astype(dtype, copy=False)
    intensity_host = intensity_host.astype(dtype, copy=False)
    mask = (
        np.ones(coordinate_host.shape, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    if mask.shape != coordinate_host.shape:
        raise ValueError("active_mask must match the coordinate shape.")
    rows = mask.reshape((-1, mask.shape[-1]))
    if any(not _prefix_mask(row) for row in rows):
        raise ValueError("Each active_mask row must be a left-prefix mask.")
    if np.any(~np.isfinite(coordinate_host[mask])):
        raise ValueError("Active coordinates must be finite.")
    if np.any(~np.isfinite(intensity_host[mask])) or np.any(intensity_host[mask] < 0.0):
        raise ValueError("Active intensities must be finite and nonnegative.")
    if np.any(coordinate_host[mask] <= 0.0):
        raise ValueError("Active mass-to-charge coordinates must be positive.")
    if np.any(coordinate_host[~mask] != 0.0) or np.any(intensity_host[~mask] != 0.0):
        raise ValueError("Inactive bounded entries must be zero padding.")
    safe_coordinate = np.where(mask, coordinate_host, 0.0)
    sorted_coordinate = np.sort(np.where(mask, coordinate_host, np.inf), axis=-1)
    observed_coordinate = np.where(mask, coordinate_host, np.inf)
    if np.any(sorted_coordinate != observed_coordinate):
        raise ValueError("Active mass-to-charge coordinates must be sorted.")
    return safe_coordinate, np.where(mask, intensity_host, 0.0), mask


def _validate_mobility(
    ion_mobility: ArrayLike | None,
    ion_mobility_mask: ArrayLike | None,
    active_mask: np.ndarray,
    units: SpectrometryUnits,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    if ion_mobility is None:
        mobility = np.zeros(active_mask.shape, dtype=float)
        mobility_mask = np.zeros(active_mask.shape, dtype=bool)
    else:
        mobility = np.asarray(ion_mobility)
        if mobility.shape != active_mask.shape:
            raise ValueError("ion_mobility must match the spectrum coordinate shape.")
        mobility = mobility.astype(float, copy=False)
        mobility_mask = (
            active_mask.copy()
            if ion_mobility_mask is None
            else np.asarray(ion_mobility_mask, dtype=bool)
        )
        if mobility_mask.shape != active_mask.shape:
            raise ValueError("ion_mobility_mask must match the spectrum shape.")
    if np.any(mobility_mask & ~active_mask):
        raise ValueError("Ion-mobility entries must refer to active spectral points.")
    if np.any(~np.isfinite(mobility[mobility_mask])):
        raise ValueError("Active ion-mobility coordinates must be finite.")
    if np.any(mobility[~mobility_mask] != 0.0):
        raise ValueError("Inactive ion-mobility entries must be zero padding.")
    if units.ion_mobility == IonMobilityUnit.NONE and np.any(mobility_mask):
        raise ValueError("Ion-mobility values require a non-NONE mobility unit.")
    return np.where(mobility_mask, mobility, 0.0), mobility_mask


class MassSpectrum(StrictModule):
    """One fixed-capacity profile or centroid mass spectrum."""

    mass_to_charge: Array
    intensity: Array
    active_mask: Array
    ion_mobility: Array
    ion_mobility_mask: Array
    scan_id: Array
    retention_time: Array
    representation: SpectrumRepresentation = eqx.field(static=True)
    polarity: IonPolarity = eqx.field(static=True)
    ms_level: int = eqx.field(static=True)
    units: SpectrometryUnits = eqx.field(static=True)
    point_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        mass_to_charge: ArrayLike,
        intensity: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        ion_mobility: ArrayLike | None = None,
        ion_mobility_mask: ArrayLike | None = None,
        scan_id: int | ArrayLike = -1,
        retention_time: float | ArrayLike = 0.0,
        representation: SpectrumRepresentation = SpectrumRepresentation.CENTROID,
        polarity: IonPolarity = IonPolarity.UNKNOWN,
        ms_level: int = 1,
        units: SpectrometryUnits | None = None,
    ):
        resolved_units = SpectrometryUnits() if units is None else units
        if not isinstance(resolved_units, SpectrometryUnits):
            raise TypeError("units must be SpectrometryUnits.")
        mz, signal, mask = _validate_axis(mass_to_charge, intensity, active_mask, rank=1)
        mobility, mobility_mask = _validate_mobility(
            ion_mobility, ion_mobility_mask, mask, resolved_units
        )
        scan = np.asarray(scan_id)
        time = np.asarray(retention_time, dtype=mz.dtype)
        if scan.shape != () or not np.issubdtype(scan.dtype, np.integer):
            raise TypeError("scan_id must be an integer scalar.")
        if time.shape != () or not np.isfinite(time) or float(time) < 0.0:
            raise ValueError("retention_time must be a finite nonnegative scalar.")
        level = int(ms_level)
        if level < 1:
            raise ValueError("ms_level must be positive.")
        self.mass_to_charge = jnp.asarray(mz)
        self.intensity = jnp.asarray(signal)
        self.active_mask = jnp.asarray(mask)
        self.ion_mobility = jnp.asarray(mobility, dtype=mz.dtype)
        self.ion_mobility_mask = jnp.asarray(mobility_mask)
        self.scan_id = jnp.asarray(scan, dtype=jnp.int64)
        self.retention_time = jnp.asarray(time)
        self.representation = SpectrumRepresentation(representation)
        self.polarity = IonPolarity(polarity)
        self.ms_level = level
        self.units = resolved_units
        self.point_capacity = int(mz.shape[0])

    @property
    def point_count(self) -> Array:
        return jnp.sum(self.active_mask, dtype=jnp.int32)


class SpectrumBatch(StrictModule):
    """A fixed scan-by-point capacity mass-spectrometry run payload."""

    mass_to_charge: Array
    intensity: Array
    point_mask: Array
    scan_mask: Array
    scan_ids: Array
    ms_levels: Array
    retention_time: Array
    ion_mobility: Array
    ion_mobility_mask: Array
    representation: SpectrumRepresentation = eqx.field(static=True)
    polarity: IonPolarity = eqx.field(static=True)
    units: SpectrometryUnits = eqx.field(static=True)
    scan_capacity: int = eqx.field(static=True)
    point_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        mass_to_charge: ArrayLike,
        intensity: ArrayLike,
        /,
        *,
        point_mask: ArrayLike | None = None,
        scan_mask: ArrayLike | None = None,
        scan_ids: ArrayLike | None = None,
        ms_levels: ArrayLike | None = None,
        retention_time: ArrayLike | None = None,
        ion_mobility: ArrayLike | None = None,
        ion_mobility_mask: ArrayLike | None = None,
        representation: SpectrumRepresentation = SpectrumRepresentation.CENTROID,
        polarity: IonPolarity = IonPolarity.UNKNOWN,
        units: SpectrometryUnits | None = None,
    ):
        resolved_units = SpectrometryUnits() if units is None else units
        if not isinstance(resolved_units, SpectrometryUnits):
            raise TypeError("units must be SpectrometryUnits.")
        mz, signal, peaks = _validate_axis(mass_to_charge, intensity, point_mask, rank=2)
        scans, points = mz.shape
        active_scans = (
            np.any(peaks, axis=1)
            if scan_mask is None
            else np.asarray(scan_mask, dtype=bool)
        )
        if active_scans.shape != (scans,):
            raise ValueError("scan_mask must have shape (scan_capacity,).")
        if not _prefix_mask(active_scans):
            raise ValueError("scan_mask must be a left-prefix mask.")
        if np.any(peaks[~active_scans]):
            raise ValueError("Inactive scans cannot contain active spectral points.")
        ids = (
            np.arange(scans, dtype=np.int64) if scan_ids is None else np.asarray(scan_ids)
        )
        if ids.shape != (scans,) or not np.issubdtype(ids.dtype, np.integer):
            raise TypeError("scan_ids must be an integer scan-capacity vector.")
        levels = (
            np.ones((scans,), dtype=np.int32)
            if ms_levels is None
            else np.asarray(ms_levels)
        )
        if levels.shape != (scans,) or not np.issubdtype(levels.dtype, np.integer):
            raise TypeError("ms_levels must be an integer scan-capacity vector.")
        if np.any(levels[active_scans] < 1):
            raise ValueError("Active scans require positive ms_levels.")
        times = (
            np.zeros((scans,), dtype=mz.dtype)
            if retention_time is None
            else np.asarray(retention_time, dtype=mz.dtype)
        )
        if times.shape != (scans,):
            raise ValueError("retention_time must be a scan-capacity vector.")
        if np.any(~np.isfinite(times[active_scans])) or np.any(times[active_scans] < 0.0):
            raise ValueError("Active retention times must be finite and nonnegative.")
        if np.any(np.diff(times[active_scans]) < 0.0):
            raise ValueError("Active retention times must be nondecreasing.")
        mobility, mobility_mask = _validate_mobility(
            ion_mobility, ion_mobility_mask, peaks, resolved_units
        )
        self.mass_to_charge = jnp.asarray(mz)
        self.intensity = jnp.asarray(signal)
        self.point_mask = jnp.asarray(peaks)
        self.scan_mask = jnp.asarray(active_scans)
        self.scan_ids = jnp.asarray(np.where(active_scans, ids, -1), dtype=jnp.int64)
        self.ms_levels = jnp.asarray(np.where(active_scans, levels, 0), dtype=jnp.int32)
        self.retention_time = jnp.asarray(np.where(active_scans, times, 0.0))
        self.ion_mobility = jnp.asarray(mobility, dtype=mz.dtype)
        self.ion_mobility_mask = jnp.asarray(mobility_mask)
        self.representation = SpectrumRepresentation(representation)
        self.polarity = IonPolarity(polarity)
        self.units = resolved_units
        self.scan_capacity = scans
        self.point_capacity = points

    @property
    def scan_count(self) -> Array:
        return jnp.sum(self.scan_mask, dtype=jnp.int32)


class Chromatogram(StrictModule):
    """One fixed-capacity chromatogram with explicit transition identity."""

    time: Array
    intensity: Array
    active_mask: Array
    precursor_mass_to_charge: Array
    product_mass_to_charge: Array
    has_precursor: Array
    has_product: Array
    polarity: IonPolarity = eqx.field(static=True)
    units: SpectrometryUnits = eqx.field(static=True)
    point_capacity: int = eqx.field(static=True)

    def __init__(
        self,
        time: ArrayLike,
        intensity: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
        precursor_mass_to_charge: float | ArrayLike | None = None,
        product_mass_to_charge: float | ArrayLike | None = None,
        polarity: IonPolarity = IonPolarity.UNKNOWN,
        units: SpectrometryUnits | None = None,
    ):
        resolved_units = SpectrometryUnits() if units is None else units
        if not isinstance(resolved_units, SpectrometryUnits):
            raise TypeError("units must be SpectrometryUnits.")
        time_host = np.asarray(time)
        signal = np.asarray(intensity)
        if time_host.ndim != 1 or time_host.shape != signal.shape or time_host.size == 0:
            raise ValueError("time and intensity must be equal non-empty vectors.")
        dtype = np.result_type(time_host.dtype, signal.dtype, np.float32)
        time_host = time_host.astype(dtype, copy=False)
        signal = signal.astype(dtype, copy=False)
        mask = (
            np.ones(time_host.shape, dtype=bool)
            if active_mask is None
            else np.asarray(active_mask, dtype=bool)
        )
        if mask.shape != time_host.shape or not _prefix_mask(mask):
            raise ValueError("active_mask must be a left-prefix time vector.")
        if np.any(~np.isfinite(time_host[mask])) or np.any(time_host[mask] < 0.0):
            raise ValueError("Active times must be finite and nonnegative.")
        if np.any(np.diff(time_host[mask]) < 0.0):
            raise ValueError("Active chromatogram times must be nondecreasing.")
        if np.any(~np.isfinite(signal[mask])) or np.any(signal[mask] < 0.0):
            raise ValueError(
                "Active chromatogram intensities must be finite and nonnegative."
            )
        if np.any(time_host[~mask] != 0.0) or np.any(signal[~mask] != 0.0):
            raise ValueError("Inactive chromatogram entries must be zero padding.")

        def transition(value: float | ArrayLike | None) -> tuple[np.ndarray, np.ndarray]:
            if value is None:
                return np.asarray(0.0, dtype=dtype), np.asarray(False)
            scalar = np.asarray(value, dtype=dtype)
            if scalar.shape != () or not np.isfinite(scalar) or float(scalar) <= 0.0:
                raise ValueError("Transition mass-to-charge must be finite and positive.")
            return scalar, np.asarray(True)

        precursor, has_precursor = transition(precursor_mass_to_charge)
        product, has_product = transition(product_mass_to_charge)
        if bool(has_product) and not bool(has_precursor):
            raise ValueError("A product transition requires precursor mass-to-charge.")
        self.time = jnp.asarray(np.where(mask, time_host, 0.0))
        self.intensity = jnp.asarray(np.where(mask, signal, 0.0))
        self.active_mask = jnp.asarray(mask)
        self.precursor_mass_to_charge = jnp.asarray(precursor)
        self.product_mass_to_charge = jnp.asarray(product)
        self.has_precursor = jnp.asarray(has_precursor)
        self.has_product = jnp.asarray(has_product)
        self.polarity = IonPolarity(polarity)
        self.units = resolved_units
        self.point_capacity = int(time_host.shape[0])


__all__ = [
    "Chromatogram",
    "IntensityUnit",
    "IonMobilityUnit",
    "IonPolarity",
    "MassSpectrum",
    "MassToChargeUnit",
    "SpectrometryUnits",
    "SpectrumBatch",
    "SpectrumRepresentation",
    "TimeUnit",
]
