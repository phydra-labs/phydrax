#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import conversion_factor, HERTZ, UnitDefinition
from ._gr_products import StokesImage
from ._photometry import ObservationDataProvenance


class InterferometryStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_VISIBILITY = 1
    ZERO_AMPLITUDE = 2


def interferometry_status_message(status: InterferometryStatus | int, /) -> str:
    code = InterferometryStatus(int(status))
    if code is InterferometryStatus.SUCCESS:
        return "success"
    if code is InterferometryStatus.NONFINITE_VISIBILITY:
        return "selected visibility is non-finite"
    return "selected visibility amplitude is zero or below tolerance"


def _station_ids(values: Sequence[str], /) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if (
        len(result) < 2
        or len(set(result)) != len(result)
        or any(not value for value in result)
    ):
        raise ValueError("station_ids must contain at least two unique non-empty names.")
    return result


def _integer_matrix(
    value: ArrayLike, columns: int, name: str, /, *, allow_empty: bool
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1:] != (columns,):
        raise ValueError(f"{name} must have shape (count, {columns}).")
    if not allow_empty and raw.shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")
    integers = raw.astype(np.int64)
    if not np.array_equal(raw, integers):
        raise ValueError(f"{name} must contain integer indices.")
    return integers


class VisibilitySampling(StrictModule, NonTrainableState):
    """Fixed station-baseline and Fourier-plane support for direct visibility data."""

    uv_coordinates: Array
    station_pairs: Array
    frequencies: Array
    frequency_unit: UnitDefinition
    station_ids: tuple[str, ...] = eqx.field(static=True)
    uv_unit: str = eqx.field(static=True)
    visibility_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        uv_coordinates: ArrayLike,
        station_pairs: ArrayLike,
        frequencies: ArrayLike,
        station_ids: Sequence[str],
        /,
        *,
        frequency_unit: UnitDefinition = HERTZ,
        uv_unit: str = "wavelength",
    ):
        if not isinstance(frequency_unit, UnitDefinition):
            raise TypeError("frequency_unit must be UnitDefinition.")
        conversion_factor(frequency_unit, HERTZ)
        uv_unit_ = str(uv_unit).strip()
        if uv_unit_ != "wavelength":
            raise ValueError("Direct image visibilities require uv_unit='wavelength'.")
        stations = _station_ids(station_ids)
        uv = np.asarray(uv_coordinates, dtype=float)
        if uv.ndim != 2 or uv.shape[1:] != (2,) or uv.shape[0] == 0:
            raise ValueError("uv_coordinates must have shape (visibility, 2).")
        pairs = _integer_matrix(station_pairs, 2, "station_pairs", allow_empty=False)
        frequency = np.asarray(frequencies, dtype=float)
        if frequency.shape == ():
            frequency = np.full((uv.shape[0],), frequency)
        if pairs.shape[0] != uv.shape[0] or frequency.shape != (uv.shape[0],):
            raise ValueError("UV coordinates, station pairs, and frequencies must align.")
        if (
            np.any(~np.isfinite(uv))
            or np.any(~np.isfinite(frequency))
            or np.any(frequency <= 0.0)
            or np.any(pairs < 0)
            or np.any(pairs >= len(stations))
            or np.any(pairs[:, 0] == pairs[:, 1])
        ):
            raise ValueError(
                "Visibility support must be finite, positive-frequency cross-station data."
            )
        self.uv_coordinates = jnp.asarray(uv)
        self.station_pairs = jnp.asarray(pairs, dtype=jnp.int32)
        self.frequencies = jnp.asarray(frequency, dtype=self.uv_coordinates.dtype)
        self.frequency_unit = frequency_unit
        self.station_ids = stations
        self.uv_unit = uv_unit_
        self.visibility_count = uv.shape[0]
        self.topology_id = canonical_fingerprint(
            {
                "kind": "visibility-sampling",
                "arrays": array_tree_fingerprint(
                    (self.uv_coordinates, self.station_pairs, self.frequencies)
                ),
                "stations": list(stations),
                "frequency_unit": frequency_unit.unit_id,
                "uv_unit": uv_unit_,
                "fourier_kernel": "exp(-2pi*i*(u*l+v*m))",
            }
        )


class StokesVisibilityData(StrictModule, NonTrainableState):
    """Complex ``(I,Q,U,V)`` visibilities tied to one fixed sampling topology."""

    visibilities: Array
    sampling: VisibilitySampling
    visibility_unit: UnitDefinition
    provenance: ObservationDataProvenance
    finite: Array
    parent_product_ids: tuple[str, ...] = eqx.field(static=True)
    content_id: str = eqx.field(static=True)

    def __init__(
        self,
        visibilities: ArrayLike,
        sampling: VisibilitySampling,
        visibility_unit: UnitDefinition,
        provenance: ObservationDataProvenance,
        /,
        *,
        parent_product_ids: Sequence[str] = (),
    ):
        if not isinstance(sampling, VisibilitySampling):
            raise TypeError("sampling must be VisibilitySampling.")
        if not isinstance(visibility_unit, UnitDefinition):
            raise TypeError("visibility_unit must be UnitDefinition.")
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        parents = tuple(str(value).strip() for value in parent_product_ids)
        if any(not value for value in parents):
            raise ValueError("parent_product_ids must contain non-empty identities.")
        raw = np.asarray(visibilities)
        expected = (4, sampling.visibility_count)
        if raw.shape != expected or not np.issubdtype(raw.dtype, np.number):
            raise ValueError(f"Stokes visibilities must have numeric shape {expected}.")
        values = raw.astype(np.result_type(raw.dtype, np.complex64), copy=False)
        finite = np.all(np.isfinite(values))
        self.visibilities = jnp.asarray(values)
        self.sampling = sampling
        self.visibility_unit = visibility_unit
        self.provenance = provenance
        self.finite = jnp.asarray(finite)
        self.parent_product_ids = parents
        self.content_id = canonical_fingerprint(
            {
                "kind": "stokes-visibility-data",
                "sampling": sampling.topology_id,
                "visibilities": array_tree_fingerprint(self.visibilities),
                "visibility_unit": visibility_unit.unit_id,
                "provenance": provenance.provenance_id,
                "parents": list(parents),
            }
        )

    @property
    def total_intensity(self) -> Array:
        return self.visibilities[0]


def direct_stokes_visibilities(
    image: StokesImage, sampling: VisibilitySampling, /
) -> StokesVisibilityData:
    """Evaluate the signed direct Fourier sum of a physical Stokes image."""

    if not isinstance(image, StokesImage) or not isinstance(sampling, VisibilitySampling):
        raise TypeError(
            "Direct visibility evaluation needs StokesImage and VisibilitySampling."
        )
    frequency_factor = float(
        conversion_factor(sampling.frequency_unit, image.frequency_unit)
    )
    frequency = np.asarray(sampling.frequencies) * frequency_factor
    image_frequency = float(np.asarray(image.frequency))
    tolerance = 64.0 * np.finfo(frequency.dtype).eps * max(1.0, abs(image_frequency))
    if np.any(np.abs(frequency - image_frequency) > tolerance):
        raise ValueError("Visibility frequencies do not match the monochromatic image.")

    coordinates = image.screen.coordinates_in_radians.reshape((-1, 2))
    uv_phase = contract("bd,pd->bp", sampling.uv_coordinates, coordinates)
    kernel = jnp.exp(-2j * jnp.pi * uv_phase)
    mask = image.valid_mask.reshape((-1,))
    solid_angle = jnp.where(mask, image.screen.solid_angle.reshape((-1,)), 0.0)
    weighted_stokes = image.stokes.reshape((4, -1)) * solid_angle[None, :]
    visibility = contract("sp,bp->sb", weighted_stokes, kernel)
    visibility = visibility * image.integration_unit_factor
    return StokesVisibilityData(
        visibility,
        sampling,
        image.flux_density_unit,
        image.provenance,
        parent_product_ids=(image.content_id,),
    )


def apply_station_gains(
    data: StokesVisibilityData, station_gains: ArrayLike, /
) -> StokesVisibilityData:
    """Apply scalar antenna gains ``g_i conjugate(g_j)`` to every Stokes product."""

    if not isinstance(data, StokesVisibilityData):
        raise TypeError("data must be StokesVisibilityData.")
    raw = np.asarray(station_gains)
    if raw.shape != (len(data.sampling.station_ids),) or not np.issubdtype(
        raw.dtype, np.number
    ):
        raise ValueError("station_gains must provide one numeric gain per station.")
    gains = jnp.asarray(raw.astype(np.result_type(raw.dtype, np.complex64), copy=False))
    pairs = data.sampling.station_pairs
    baseline_gain = gains[pairs[:, 0]] * jnp.conj(gains[pairs[:, 1]])
    return StokesVisibilityData(
        data.visibilities * baseline_gain[None, :],
        data.sampling,
        data.visibility_unit,
        data.provenance,
        parent_product_ids=(*data.parent_product_ids, data.content_id),
    )


class PolarizationVisibilityProducts(StrictModule, NonTrainableState):
    complex_linear_polarization: Array
    fractional_linear_polarization: Array
    fractional_circular_polarization: Array
    electric_vector_position_angle: Array
    circular_correlations: Array
    linear_correlations: Array
    finite: Array
    fractional_defined: Array
    angle_defined: Array
    status: Array
    zero_tolerance: float = eqx.field(static=True)
    source_visibility_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)


def polarization_visibility_products(
    data: StokesVisibilityData, /, *, zero_tolerance: float = 0.0
) -> PolarizationVisibilityProducts:
    """Form circular/linear feed correlations and safe fractional polarization."""

    if not isinstance(data, StokesVisibilityData):
        raise TypeError("data must be StokesVisibilityData.")
    tolerance = float(zero_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative.")
    intensity, q_value, u_value, circular = data.visibilities
    linear = q_value + 1j * u_value
    finite = (
        jnp.isfinite(intensity.real)
        & jnp.isfinite(intensity.imag)
        & jnp.isfinite(q_value.real)
        & jnp.isfinite(q_value.imag)
        & jnp.isfinite(u_value.real)
        & jnp.isfinite(u_value.imag)
        & jnp.isfinite(circular.real)
        & jnp.isfinite(circular.imag)
    )
    fractional_defined = finite & (jnp.abs(intensity) > tolerance)
    angle_defined = finite & (jnp.abs(linear) > tolerance)
    safe_intensity = jnp.where(fractional_defined, intensity, 1.0 + 0.0j)
    fractional_linear = jnp.where(fractional_defined, linear / safe_intensity, 0.0 + 0.0j)
    fractional_circular = jnp.where(
        fractional_defined, circular / safe_intensity, 0.0 + 0.0j
    )
    angle = jnp.where(angle_defined, 0.5 * jnp.angle(linear), 0.0)
    circular_correlations = jnp.stack(
        (intensity + circular, intensity - circular, linear, q_value - 1j * u_value)
    )
    linear_correlations = jnp.stack(
        (
            intensity + q_value,
            intensity - q_value,
            u_value + 1j * circular,
            u_value - 1j * circular,
        )
    )
    status = jnp.where(
        ~finite,
        int(InterferometryStatus.NONFINITE_VISIBILITY),
        jnp.where(
            fractional_defined & angle_defined,
            int(InterferometryStatus.SUCCESS),
            int(InterferometryStatus.ZERO_AMPLITUDE),
        ),
    ).astype(jnp.int32)
    result = PolarizationVisibilityProducts(
        linear,
        fractional_linear,
        fractional_circular,
        angle,
        circular_correlations,
        linear_correlations,
        finite,
        fractional_defined,
        angle_defined,
        status,
        tolerance,
        data.content_id,
        canonical_fingerprint(
            {
                "kind": "polarization-visibility-products",
                "source": data.content_id,
                "zero_tolerance": tolerance,
                "convention": {
                    "circular": ["I+V", "I-V", "Q+iU", "Q-iU"],
                    "linear": ["I+Q", "I-Q", "U+iV", "U-iV"],
                },
            }
        ),
    )
    return result


class ClosureTopology(StrictModule, NonTrainableState):
    """Gain-invariant triangle and quadrangle routes on fixed visibility samples.

    Phase rows multiply three oriented baselines; ``phase_conjugated`` reverses
    stored orientation. Amplitude rows mean ``|V0 V1| / |V2 V3|``. Construction
    proves station-gain cancellation rather than trusting caller labels.
    """

    phase_baseline_indices: Array
    phase_conjugated: Array
    amplitude_baseline_indices: Array
    visibility_topology_id: str = eqx.field(static=True)
    phase_count: int = eqx.field(static=True)
    amplitude_count: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        sampling: VisibilitySampling,
        phase_baseline_indices: ArrayLike,
        amplitude_baseline_indices: ArrayLike,
        /,
        *,
        phase_conjugated: ArrayLike | None = None,
    ):
        if not isinstance(sampling, VisibilitySampling):
            raise TypeError("sampling must be VisibilitySampling.")
        phase = _integer_matrix(
            phase_baseline_indices, 3, "phase_baseline_indices", allow_empty=True
        )
        amplitude = _integer_matrix(
            amplitude_baseline_indices,
            4,
            "amplitude_baseline_indices",
            allow_empty=True,
        )
        if phase.shape[0] == 0 and amplitude.shape[0] == 0:
            raise ValueError("ClosureTopology needs at least one closure row.")
        if phase_conjugated is None:
            conjugated = np.zeros(phase.shape, dtype=bool)
        else:
            conjugated = np.asarray(phase_conjugated)
            if conjugated.dtype != np.dtype(bool) or conjugated.shape != phase.shape:
                raise ValueError(
                    "phase_conjugated must be boolean and match phase indices."
                )
        if (
            np.any(phase < 0)
            or np.any(phase >= sampling.visibility_count)
            or np.any(amplitude < 0)
            or np.any(amplitude >= sampling.visibility_count)
        ):
            raise ValueError(
                "Closure baseline indices are outside the visibility support."
            )
        pairs = np.asarray(sampling.station_pairs)
        station_count = len(sampling.station_ids)
        for indices, reversals in zip(phase, conjugated, strict=True):
            balance = np.zeros((station_count,), dtype=np.int64)
            for index, reversal in zip(indices, reversals, strict=True):
                first, second = pairs[index]
                if reversal:
                    first, second = second, first
                balance[first] += 1
                balance[second] -= 1
            if np.any(balance != 0):
                raise ValueError(
                    "Every phase row must be an oriented closed station cycle."
                )
        for indices in amplitude:
            balance = np.zeros((station_count,), dtype=np.int64)
            for position, index in enumerate(indices):
                sign = 1 if position < 2 else -1
                first, second = pairs[index]
                balance[first] += sign
                balance[second] += sign
            if np.any(balance != 0):
                raise ValueError(
                    "Every amplitude row must cancel all station gain magnitudes."
                )
        self.phase_baseline_indices = jnp.asarray(phase, dtype=jnp.int32)
        self.phase_conjugated = jnp.asarray(conjugated)
        self.amplitude_baseline_indices = jnp.asarray(amplitude, dtype=jnp.int32)
        self.visibility_topology_id = sampling.topology_id
        self.phase_count = phase.shape[0]
        self.amplitude_count = amplitude.shape[0]
        self.topology_id = canonical_fingerprint(
            {
                "kind": "interferometric-closure-topology",
                "visibility_topology": sampling.topology_id,
                "routes": array_tree_fingerprint(
                    (
                        self.phase_baseline_indices,
                        self.phase_conjugated,
                        self.amplitude_baseline_indices,
                    )
                ),
                "amplitude_convention": "abs(V0*V1)/abs(V2*V3)",
            }
        )

    @classmethod
    def from_station_cycles(
        cls,
        sampling: VisibilitySampling,
        /,
        *,
        phase_cycles: Sequence[tuple[str, str, str]] = (),
        amplitude_cycles: Sequence[tuple[str, str, str, str]] = (),
    ) -> ClosureTopology:
        """Resolve named station cycles when each baseline pair has one sample."""

        if not isinstance(sampling, VisibilitySampling):
            raise TypeError("sampling must be VisibilitySampling.")
        station_index = {name: index for index, name in enumerate(sampling.station_ids)}
        directed: dict[tuple[int, int], tuple[int, bool]] = {}
        for index, pair in enumerate(np.asarray(sampling.station_pairs)):
            first, second = int(pair[0]), int(pair[1])
            if (first, second) in directed or (second, first) in directed:
                raise ValueError(
                    "Named station cycles require one visibility sample per baseline pair."
                )
            directed[(first, second)] = (index, False)
            directed[(second, first)] = (index, True)

        def edge(first_name: str, second_name: str) -> tuple[int, bool]:
            if first_name not in station_index or second_name not in station_index:
                raise ValueError("Closure cycle references an unknown station.")
            key = (station_index[first_name], station_index[second_name])
            if key not in directed:
                raise ValueError("Closure cycle references an unsampled baseline.")
            return directed[key]

        phase_indices: list[list[int]] = []
        phase_reversals: list[list[bool]] = []
        for cycle in phase_cycles:
            if len(cycle) != 3 or len(set(cycle)) != 3:
                raise ValueError("Phase cycles require three distinct station names.")
            routes = (
                edge(cycle[0], cycle[1]),
                edge(cycle[1], cycle[2]),
                edge(cycle[2], cycle[0]),
            )
            phase_indices.append([route[0] for route in routes])
            phase_reversals.append([route[1] for route in routes])
        amplitude_indices: list[list[int]] = []
        for cycle in amplitude_cycles:
            if len(cycle) != 4 or len(set(cycle)) != 4:
                raise ValueError("Amplitude cycles require four distinct station names.")
            routes = (
                edge(cycle[0], cycle[1]),
                edge(cycle[2], cycle[3]),
                edge(cycle[0], cycle[2]),
                edge(cycle[1], cycle[3]),
            )
            amplitude_indices.append([route[0] for route in routes])
        return cls(
            sampling,
            np.asarray(phase_indices, dtype=np.int64).reshape((-1, 3)),
            np.asarray(amplitude_indices, dtype=np.int64).reshape((-1, 4)),
            phase_conjugated=np.asarray(phase_reversals, dtype=bool).reshape((-1, 3)),
        )


class ClosureProducts(StrictModule, NonTrainableState):
    bispectrum: Array
    closure_phase: Array
    closure_amplitude: Array
    log_closure_amplitude: Array
    phase_finite: Array
    amplitude_finite: Array
    phase_physically_valid: Array
    amplitude_physically_valid: Array
    phase_derivative_valid: Array
    amplitude_derivative_valid: Array
    phase_status: Array
    amplitude_status: Array
    zero_tolerance: float = eqx.field(static=True)
    stokes_component: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    source_visibility_id: str = eqx.field(static=True)
    content_id: str = eqx.field(static=True)


def closure_products(
    data: StokesVisibilityData,
    topology: ClosureTopology,
    /,
    *,
    stokes_component: str = "I",
    zero_tolerance: float = 0.0,
) -> ClosureProducts:
    """Compute bispectra, closure phases, and safe (log-)closure amplitudes."""

    if not isinstance(data, StokesVisibilityData) or not isinstance(
        topology, ClosureTopology
    ):
        raise TypeError(
            "Closure evaluation needs StokesVisibilityData and ClosureTopology."
        )
    if data.sampling.topology_id != topology.visibility_topology_id:
        raise ValueError("Closure and visibility topology identities disagree.")
    component = str(stokes_component).strip()
    if component not in ("I", "Q", "U", "V"):
        raise ValueError("stokes_component must be one of I, Q, U, V.")
    tolerance = float(zero_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("zero_tolerance must be finite and non-negative.")
    visibility = data.visibilities[("I", "Q", "U", "V").index(component)]

    phase_values = visibility[topology.phase_baseline_indices]
    phase_values = jnp.where(
        topology.phase_conjugated, jnp.conj(phase_values), phase_values
    )
    phase_finite = jnp.all(
        jnp.isfinite(phase_values.real) & jnp.isfinite(phase_values.imag), axis=1
    )
    phase_nonzero = jnp.all(jnp.abs(phase_values) > tolerance, axis=1)
    phase_valid = phase_finite & phase_nonzero
    raw_bispectrum = jnp.prod(phase_values, axis=1)
    bispectrum = jnp.where(phase_finite, raw_bispectrum, 0.0 + 0.0j)
    phase = jnp.where(phase_valid, jnp.angle(raw_bispectrum), 0.0)
    phase_derivative_valid = phase_valid & ~(
        (raw_bispectrum.real < 0.0) & (jnp.abs(raw_bispectrum.imag) <= tolerance)
    )
    phase_status = jnp.where(
        ~phase_finite,
        int(InterferometryStatus.NONFINITE_VISIBILITY),
        jnp.where(
            phase_nonzero,
            int(InterferometryStatus.SUCCESS),
            int(InterferometryStatus.ZERO_AMPLITUDE),
        ),
    ).astype(jnp.int32)

    amplitude_values = visibility[topology.amplitude_baseline_indices]
    amplitude_finite = jnp.all(
        jnp.isfinite(amplitude_values.real) & jnp.isfinite(amplitude_values.imag),
        axis=1,
    )
    magnitudes = jnp.abs(amplitude_values)
    amplitude_nonzero = jnp.all(magnitudes > tolerance, axis=1)
    amplitude_valid = amplitude_finite & amplitude_nonzero
    safe_magnitudes = jnp.where(amplitude_valid[:, None], magnitudes, 1.0)
    log_amplitude = (
        jnp.log(safe_magnitudes[:, 0])
        + jnp.log(safe_magnitudes[:, 1])
        - jnp.log(safe_magnitudes[:, 2])
        - jnp.log(safe_magnitudes[:, 3])
    )
    amplitude = jnp.where(amplitude_valid, jnp.exp(log_amplitude), 0.0)
    log_amplitude = jnp.where(amplitude_valid, log_amplitude, 0.0)
    amplitude_status = jnp.where(
        ~amplitude_finite,
        int(InterferometryStatus.NONFINITE_VISIBILITY),
        jnp.where(
            amplitude_nonzero,
            int(InterferometryStatus.SUCCESS),
            int(InterferometryStatus.ZERO_AMPLITUDE),
        ),
    ).astype(jnp.int32)
    return ClosureProducts(
        bispectrum,
        phase,
        amplitude,
        log_amplitude,
        phase_finite,
        amplitude_finite,
        phase_valid,
        amplitude_valid,
        phase_derivative_valid,
        amplitude_valid,
        phase_status,
        amplitude_status,
        tolerance,
        component,
        topology.topology_id,
        data.content_id,
        canonical_fingerprint(
            {
                "kind": "interferometric-closure-products",
                "source": data.content_id,
                "topology": topology.topology_id,
                "stokes_component": component,
                "zero_tolerance": tolerance,
                "visibility_unit": data.visibility_unit.unit_id,
                "provenance": data.provenance.provenance_id,
            }
        ),
    )


__all__ = [
    "ClosureProducts",
    "ClosureTopology",
    "InterferometryStatus",
    "PolarizationVisibilityProducts",
    "StokesVisibilityData",
    "VisibilitySampling",
    "apply_station_gains",
    "closure_products",
    "direct_stokes_visibilities",
    "interferometry_status_message",
    "polarization_visibility_products",
]
