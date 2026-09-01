#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._exponential_family import PoissonFamily
from ..._fingerprint import canonical_fingerprint
from ..._likelihoods import ScalarNaturalExponentialFamilyLikelihood
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import DifferentiationContract
from ._observation_status import AstrophysicsObservationStatus


_PLANCK_CONSTANT = 6.62607015e-34
_SPEED_OF_LIGHT = 299792458.0


class ObservationDataProvenance(StrictModule, NonTrainableState):
    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    checksum: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    differentiation: DifferentiationContract
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        producer: str,
        producer_version: str,
        source_id: str,
        checksum: str,
        license_id: str,
        differentiation: DifferentiationContract | str,
    ):
        values = tuple(
            str(value).strip()
            for value in (producer, producer_version, source_id, checksum, license_id)
        )
        if any(not value for value in values):
            raise ValueError("Observation provenance fields must be non-empty.")
        differentiation_ = (
            DifferentiationContract.from_label(differentiation)
            if isinstance(differentiation, str)
            else differentiation
        )
        if not isinstance(differentiation_, DifferentiationContract):
            raise TypeError("Unknown observation differentiation contract.")
        (
            self.producer,
            self.producer_version,
            self.source_id,
            self.checksum,
            self.license_id,
        ) = values
        self.differentiation = differentiation_
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "observation-data-provenance",
                "producer": values[0],
                "producer_version": values[1],
                "source_id": values[2],
                "checksum": values[3],
                "license_id": values[4],
                "differentiation": differentiation_.contract_id,
            }
        )

    @classmethod
    def native(cls, source_id: str, /) -> ObservationDataProvenance:
        return cls(
            producer="phydrax",
            producer_version="native",
            source_id=source_id,
            checksum="content-fingerprinted",
            license_id="Phydrax-native",
            differentiation=DifferentiationContract.native(),
        )


class PhotonCountingBandpass(StrictModule, NonTrainableState):
    wavelength: Array
    throughput: Array
    photon_weights: Array
    provenance: ObservationDataProvenance
    band_id: str = eqx.field(static=True)
    response_id: str = eqx.field(static=True)

    def __init__(
        self,
        wavelength: ArrayLike,
        throughput: ArrayLike,
        provenance: ObservationDataProvenance,
        /,
        *,
        band_id: str,
    ):
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        identifier = str(band_id).strip()
        if not identifier:
            raise ValueError("band_id must be non-empty.")
        wavelength_host = np.asarray(wavelength, dtype=float)
        throughput_host = np.asarray(throughput, dtype=float)
        if (
            wavelength_host.ndim != 1
            or wavelength_host.size < 2
            or throughput_host.shape != wavelength_host.shape
        ):
            raise ValueError(
                "Bandpass wavelength and throughput must be matching vectors."
            )
        if (
            np.any(~np.isfinite(wavelength_host))
            or np.any(~np.isfinite(throughput_host))
            or np.any(wavelength_host <= 0.0)
            or np.any(np.diff(wavelength_host) <= 0.0)
            or np.any(throughput_host < 0.0)
        ):
            raise ValueError(
                "Bandpass nodes must be positive, increasing, finite, and non-negative."
            )
        trapezoid = np.empty_like(wavelength_host)
        trapezoid[0] = 0.5 * (wavelength_host[1] - wavelength_host[0])
        trapezoid[-1] = 0.5 * (wavelength_host[-1] - wavelength_host[-2])
        trapezoid[1:-1] = 0.5 * (wavelength_host[2:] - wavelength_host[:-2])
        weights = (
            trapezoid
            * throughput_host
            * wavelength_host
            / (_PLANCK_CONSTANT * _SPEED_OF_LIGHT)
        )
        self.wavelength = jnp.asarray(wavelength_host)
        self.throughput = jnp.asarray(throughput_host)
        self.photon_weights = jnp.asarray(weights)
        self.provenance = provenance
        self.band_id = identifier
        self.response_id = canonical_fingerprint(
            {
                "kind": "photon-counting-bandpass",
                "band_id": identifier,
                "wavelength": wavelength_host.tolist(),
                "throughput": throughput_host.tolist(),
                "provenance": provenance.provenance_id,
            }
        )

    def photon_rate(self, spectral_flux_density: ArrayLike, /) -> Array:
        flux = jnp.asarray(spectral_flux_density)
        if flux.shape[-1:] != self.wavelength.shape:
            raise ValueError("Spectral flux must match the bandpass wavelength axis.")
        return contract("...l,l->...", flux, self.photon_weights)


class TransitPhotometryResult(StrictModule):
    relative_flux: Array
    photon_rate: Array
    expected_counts: Array
    log_expected_counts: Array
    poisson_supported: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class TransitPhotometryPlan(StrictModule, NonTrainableState):
    bandpasses: tuple[PhotonCountingBandpass, ...]
    packed_weights: Array
    band_index: Array
    exposure_time: Array
    collecting_area: Array
    background_rate: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bandpasses: tuple[PhotonCountingBandpass, ...],
        band_index: ArrayLike,
        exposure_time: ArrayLike,
        /,
        *,
        collecting_area: ArrayLike,
        background_rate: ArrayLike = 0.0,
    ):
        bands = tuple(bandpasses)
        if not bands or any(
            not isinstance(band, PhotonCountingBandpass) for band in bands
        ):
            raise ValueError("bandpasses must contain PhotonCountingBandpass objects.")
        identifiers = tuple(band.band_id for band in bands)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Transit band IDs must be unique.")
        reference = np.asarray(bands[0].wavelength)
        if any(
            not np.array_equal(np.asarray(band.wavelength), reference)
            for band in bands[1:]
        ):
            raise ValueError("Transit bandpasses must share one wavelength grid.")
        indices = np.asarray(band_index)
        exposure = np.asarray(exposure_time, dtype=float)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise TypeError("band_index must be a rank-one integer array.")
        if exposure.shape != indices.shape:
            raise ValueError("exposure_time must match band_index.")
        if np.any(indices < 0) or np.any(indices >= len(bands)):
            raise ValueError("band_index contains an out-of-range band.")
        if np.any(~np.isfinite(exposure)) or np.any(exposure < 0.0):
            raise ValueError("exposure_time must be finite and non-negative.")
        area = jnp.asarray(collecting_area).reshape(())
        background = jnp.asarray(background_rate)
        if background.shape not in ((), indices.shape):
            raise ValueError("background_rate must be scalar or measurement-shaped.")
        self.bandpasses = bands
        self.packed_weights = jnp.stack(tuple(band.photon_weights for band in bands))
        self.band_index = jnp.asarray(indices, dtype=jnp.int32)
        self.exposure_time = jnp.asarray(exposure)
        self.collecting_area = area
        self.background_rate = jnp.broadcast_to(background, indices.shape)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "transit-photometry-plan",
                "bands": [band.response_id for band in bands],
                "band_index": indices.tolist(),
                "num_measurements": int(indices.size),
            }
        )

    def evaluate(
        self,
        relative_flux: ArrayLike,
        source_spectral_flux_density: ArrayLike,
        /,
    ) -> TransitPhotometryResult:
        relative = jnp.asarray(relative_flux)
        spectra = jnp.asarray(source_spectral_flux_density)
        measurements = int(self.band_index.size)
        wavelength_count = int(self.packed_weights.shape[1])
        if relative.shape != (measurements,):
            raise ValueError("relative_flux must have the measurement shape.")
        if spectra.shape not in ((wavelength_count,), (measurements, wavelength_count)):
            raise ValueError(
                "Source spectra must be wavelength- or measurement-by-wavelength shaped."
            )
        spectra = jnp.broadcast_to(spectra, (measurements, wavelength_count))
        rates_by_band = contract("nl,bl->nb", spectra, self.packed_weights)
        selected_rate = rates_by_band[jnp.arange(measurements), self.band_index]
        finite = (
            jnp.all(jnp.isfinite(relative))
            & jnp.all(jnp.isfinite(spectra))
            & jnp.isfinite(self.collecting_area)
            & jnp.all(jnp.isfinite(self.background_rate))
        )
        physical = (
            finite
            & jnp.all((relative >= 0.0) & (relative <= 1.0))
            & jnp.all(spectra >= 0.0)
            & (self.collecting_area > 0.0)
            & jnp.all(self.background_rate >= 0.0)
            & jnp.all(selected_rate >= 0.0)
        )
        expected = (
            relative * selected_rate * self.collecting_area + self.background_rate
        ) * self.exposure_time
        expected = jnp.where(physical, expected, 0.0)
        poisson_supported = physical & (expected > 0.0)
        log_expected = jnp.log(jnp.where(poisson_supported, expected, 1.0))
        status = jnp.where(
            ~finite,
            int(AstrophysicsObservationStatus.NONFINITE_INPUT),
            jnp.where(
                physical,
                int(AstrophysicsObservationStatus.SUCCESS),
                int(AstrophysicsObservationStatus.NONPHYSICAL_MODEL),
            ),
        ).astype(jnp.int32)
        return TransitPhotometryResult(
            jnp.where(physical, relative, 1.0),
            jnp.where(physical, selected_rate, 0.0),
            expected,
            log_expected,
            poisson_supported,
            jnp.broadcast_to(physical, expected.shape),
            jnp.broadcast_to(status, expected.shape),
            self.plan_id,
        )


def transit_poisson_likelihood() -> ScalarNaturalExponentialFamilyLikelihood:
    """Return the existing normalized scalar Poisson likelihood contract."""

    return ScalarNaturalExponentialFamilyLikelihood(PoissonFamily())


def transit_poisson_log_prob(
    result: TransitPhotometryResult,
    observed_counts: ArrayLike,
    /,
    *,
    mask: ArrayLike | None = None,
) -> Array:
    """Compose valid positive transit means with the existing Poisson law."""

    if not isinstance(result, TransitPhotometryResult):
        raise TypeError("result must be a TransitPhotometryResult.")
    counts = jnp.asarray(observed_counts)
    if counts.shape != result.expected_counts.shape:
        raise ValueError("observed_counts must match expected_counts.")
    active = (
        jnp.ones_like(result.poisson_supported)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if active.shape != counts.shape:
        raise ValueError("mask must match observed_counts.")
    terms = transit_poisson_likelihood().log_prob(
        result.log_expected_counts,
        counts,
    )
    valid = ~active | result.poisson_supported
    return jnp.where(jnp.all(valid), jnp.sum(jnp.where(active, terms, 0.0)), -jnp.inf)


__all__ = [
    "ObservationDataProvenance",
    "PhotonCountingBandpass",
    "TransitPhotometryPlan",
    "TransitPhotometryResult",
    "transit_poisson_likelihood",
    "transit_poisson_log_prob",
]
