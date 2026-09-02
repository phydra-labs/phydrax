#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import (
    CoordinateLayout,
    LinearObservationPlan,
    ScientificArtifactEnvelope,
    TheoryVector,
)
from ._cmb import CmbSpectrumTable


class CmbIngressEvidence(StrictModule):
    low_multipoles_zero: Array
    diagonal_nonnegative: Array
    pairwise_covariance_valid: Array
    finite: Array
    successful: Array


class CmbIngressPlan(StrictModule, NonTrainableState):
    tolerance: float = eqx.field(static=True)

    def __init__(self, *, tolerance: float = 1.0e-10):
        value = float(tolerance)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("CMB ingress tolerance must be finite and positive.")
        self.tolerance = value

    def validate(self, table: CmbSpectrumTable, /) -> CmbIngressEvidence:
        values = table.spectra
        low = table.multipoles < 2
        low_zero = jnp.all(
            jnp.where(low[None, :, None, None], jnp.abs(values), 0.0) <= self.tolerance
        )
        diagonal = jnp.diagonal(values, axis1=-2, axis2=-1)
        diagonal_nonnegative = jnp.all(diagonal >= -self.tolerance)
        products = diagonal[..., :, None] * diagonal[..., None, :]
        pairwise = jnp.all(values**2 <= products + self.tolerance)
        finite = jnp.all(jnp.isfinite(values))
        successful = low_zero & diagonal_nonnegative & pairwise & finite
        return CmbIngressEvidence(
            low_zero, diagonal_nonnegative, pairwise, finite, successful
        )


class CmbSkyMapProduct(StrictModule):
    iqu: Array
    pixelization: str = eqx.field(static=True)
    nside: int = eqx.field(static=True)
    coordinate_frame: str = eqx.field(static=True)
    convention: str = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope
    map_id: str = eqx.field(static=True)


class HarmonicSkySynthesisPlan(StrictModule, NonTrainableState):
    synthesis_matrix: Array
    harmonic_cholesky: Array
    nside: int = eqx.field(static=True)
    lmax: int = eqx.field(static=True)
    pixelization: str = eqx.field(static=True)
    artifact: ScientificArtifactEnvelope
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        synthesis_matrix: ArrayLike,
        harmonic_cholesky: ArrayLike,
        /,
        *,
        nside: int,
        lmax: int,
        pixelization: str,
        artifact: ScientificArtifactEnvelope,
    ):
        matrix = jax.lax.stop_gradient(jnp.asarray(synthesis_matrix))
        cholesky = jax.lax.stop_gradient(
            jnp.asarray(harmonic_cholesky, dtype=matrix.dtype)
        )
        nside_ = int(nside)
        lmax_ = int(lmax)
        pixelization_ = str(pixelization).strip()
        pixel_count = 12 * nside_**2
        if (
            nside_ <= 0
            or lmax_ < 2
            or not pixelization_
            or matrix.ndim != 2
            or matrix.shape[0] != 3 * pixel_count
            or cholesky.ndim != 2
            or cholesky.shape[0] != cholesky.shape[1]
            or matrix.shape[1] != cholesky.shape[0]
        ):
            raise ValueError("CMB harmonic synthesis shapes or geometry are invalid.")
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(matrix)) | jnp.any(~jnp.isfinite(cholesky)),
            "CMB harmonic synthesis arrays must be finite.",
        )
        self.synthesis_matrix = matrix
        self.harmonic_cholesky = cholesky
        self.nside = nside_
        self.lmax = lmax_
        self.pixelization = pixelization_
        self.artifact = artifact
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cmb-harmonic-synthesis",
                "nside": nside_,
                "lmax": lmax_,
                "pixelization": pixelization_,
                "artifact": artifact.artifact_id,
                "arrays": array_tree_fingerprint((matrix, cholesky)),
            }
        )

    def realize(self, key: Array, /) -> CmbSkyMapProduct:
        normal = jax.random.normal(
            key, (self.harmonic_cholesky.shape[0],), dtype=self.synthesis_matrix.dtype
        )
        coefficients = contract("ij,j->i", self.harmonic_cholesky, normal)
        flat_map = contract("pi,i->p", self.synthesis_matrix, coefficients)
        iqu = flat_map.reshape((12 * self.nside**2, 3))
        return CmbSkyMapProduct(
            iqu,
            self.pixelization,
            self.nside,
            "celestial",
            "IAU",
            self.artifact,
            canonical_fingerprint(
                {"kind": "cmb-sky-map", "plan": self.plan_id, "shape": list(iqu.shape)}
            ),
        )


class CmbBeamProduct(StrictModule, NonTrainableState):
    full_width_half_max_radians: float = eqx.field(static=True)
    channel: str = eqx.field(static=True)
    beam_id: str = eqx.field(static=True)

    def __init__(self, full_width_half_max_radians: float, channel: str, /):
        width = float(full_width_half_max_radians)
        channel_ = str(channel).strip()
        if not np.isfinite(width) or width <= 0.0 or not channel_:
            raise ValueError("CMB beam parameters are invalid.")
        self.full_width_half_max_radians = width
        self.channel = channel_
        self.beam_id = canonical_fingerprint(
            {"kind": "circular-gaussian-beam", "fwhm": width, "channel": channel_}
        )


class CmbPointingProduct(StrictModule, NonTrainableState):
    pixel_indices: Array
    polarization_angles: Array
    flags: Array
    detector_indices: Array
    pixel_count: int = eqx.field(static=True)
    pointing_id: str = eqx.field(static=True)

    def __init__(
        self,
        pixel_indices: ArrayLike,
        polarization_angles: ArrayLike,
        flags: ArrayLike,
        detector_indices: ArrayLike,
        /,
        *,
        pixel_count: int,
    ):
        pixels = jax.lax.stop_gradient(jnp.asarray(pixel_indices, dtype=jnp.int32))
        angles = jax.lax.stop_gradient(jnp.asarray(polarization_angles))
        flags_ = jax.lax.stop_gradient(jnp.asarray(flags, dtype=bool))
        detectors = jax.lax.stop_gradient(jnp.asarray(detector_indices, dtype=jnp.int32))
        count = int(pixel_count)
        if (
            pixels.ndim != 1
            or angles.shape != pixels.shape
            or flags_.shape != pixels.shape
            or detectors.shape != pixels.shape
            or count <= 0
        ):
            raise ValueError("CMB pointing arrays are invalid.")
        pixels = eqx.error_if(
            pixels,
            jnp.any(pixels < 0)
            | jnp.any(pixels >= count)
            | jnp.any(~jnp.isfinite(angles)),
            "CMB pointing samples must be in range and finite.",
        )
        self.pixel_indices = pixels
        self.polarization_angles = angles
        self.flags = flags_
        self.detector_indices = detectors
        self.pixel_count = count
        self.pointing_id = canonical_fingerprint(
            {
                "kind": "cmb-pointing",
                "pixel_count": count,
                "arrays": array_tree_fingerprint((pixels, angles, flags_, detectors)),
            }
        )


class CmbTodProduct(StrictModule):
    samples: Array
    pointing: CmbPointingProduct
    noise_sigma: Array
    sample_interval: Array
    tod_id: str = eqx.field(static=True)


class CmbTodSimulationPlan(StrictModule, NonTrainableState):
    pointing: CmbPointingProduct
    beam: CmbBeamProduct
    net_microkelvin_sqrt_second: float = eqx.field(static=True)
    sample_interval_seconds: float = eqx.field(static=True)
    gain: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pointing: CmbPointingProduct,
        beam: CmbBeamProduct,
        /,
        *,
        net_microkelvin_sqrt_second: float = 50.0,
        sample_interval_seconds: float = 0.1,
        gain: float = 1.0,
    ):
        net = float(net_microkelvin_sqrt_second)
        interval = float(sample_interval_seconds)
        gain_ = float(gain)
        if (
            not np.isfinite(net)
            or net <= 0.0
            or not np.isfinite(interval)
            or interval <= 0.0
            or not np.isfinite(gain_)
        ):
            raise ValueError("CMB TOD policy is invalid.")
        self.pointing = pointing
        self.beam = beam
        self.net_microkelvin_sqrt_second = net
        self.sample_interval_seconds = interval
        self.gain = gain_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cmb-white-noise-tod",
                "pointing": pointing.pointing_id,
                "beam": beam.beam_id,
                "net": net,
                "sample_interval": interval,
                "gain": gain_,
            }
        )

    def simulate(self, sky: CmbSkyMapProduct, key: Array, /) -> CmbTodProduct:
        if sky.iqu.shape != (self.pointing.pixel_count, 3):
            raise ValueError("Sky map and pointing pixelization disagree.")
        selected = sky.iqu[self.pointing.pixel_indices]
        angle = self.pointing.polarization_angles
        signal = self.gain * (
            selected[:, 0]
            + selected[:, 1] * jnp.cos(2.0 * angle)
            + selected[:, 2] * jnp.sin(2.0 * angle)
        )
        sigma = jnp.asarray(
            self.net_microkelvin_sqrt_second / np.sqrt(self.sample_interval_seconds),
            dtype=signal.dtype,
        )
        noise = sigma * jax.random.normal(key, signal.shape, dtype=signal.dtype)
        samples = jnp.where(self.pointing.flags, 0.0, signal + noise)
        return CmbTodProduct(
            samples,
            self.pointing,
            sigma,
            jnp.asarray(self.sample_interval_seconds, dtype=signal.dtype),
            canonical_fingerprint(
                {"kind": "cmb-tod", "plan": self.plan_id, "shape": list(samples.shape)}
            ),
        )


class CmbMapmakingEvidence(StrictModule):
    hit_count: Array
    determinant: Array
    selected_pixels: Array
    finite: Array
    successful: Array


class CmbMapmakingResult(StrictModule):
    map: Array
    evidence: CmbMapmakingEvidence
    successful: Array


class CmbMapmakingPlan(StrictModule, NonTrainableState):
    pixel_count: int = eqx.field(static=True)
    determinant_tolerance: float = eqx.field(static=True)

    def __init__(self, pixel_count: int, /, *, determinant_tolerance: float = 1.0e-10):
        count = int(pixel_count)
        tolerance = float(determinant_tolerance)
        if count <= 0 or not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("CMB mapmaking policy is invalid.")
        self.pixel_count = count
        self.determinant_tolerance = tolerance

    @staticmethod
    def _solve_symmetric(
        matrix: Array, rhs: Array, tolerance: float
    ) -> tuple[Array, Array]:
        a, b, c = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
        d, e, f = matrix[..., 1, 1], matrix[..., 1, 2], matrix[..., 2, 2]
        determinant = a * (d * f - e**2) - b * (b * f - c * e) + c * (b * e - c * d)
        adjugate = jnp.stack(
            (
                jnp.stack((d * f - e**2, c * e - b * f, b * e - c * d), axis=-1),
                jnp.stack((c * e - b * f, a * f - c**2, b * c - a * e), axis=-1),
                jnp.stack((b * e - c * d, b * c - a * e, a * d - b**2), axis=-1),
            ),
            axis=-2,
        )
        safe = jnp.where(jnp.abs(determinant) > tolerance, determinant, 1.0)
        solution = contract("...ij,...j->...i", adjugate, rhs) / safe[..., None]
        return solution, determinant

    def solve(self, tod: CmbTodProduct, /) -> CmbMapmakingResult:
        pointing = tod.pointing
        angle = pointing.polarization_angles
        design = jnp.stack(
            (jnp.ones_like(angle), jnp.cos(2.0 * angle), jnp.sin(2.0 * angle)),
            axis=-1,
        )
        weight = jnp.where(pointing.flags, 0.0, 1.0 / tod.noise_sigma**2)
        normal_samples = weight[:, None, None] * design[:, :, None] * design[:, None, :]
        rhs_samples = weight[:, None] * design * tod.samples[:, None]
        normal = (
            jnp.zeros((self.pixel_count, 3, 3), dtype=tod.samples.dtype)
            .at[pointing.pixel_indices]
            .add(normal_samples)
        )
        rhs = (
            jnp.zeros((self.pixel_count, 3), dtype=tod.samples.dtype)
            .at[pointing.pixel_indices]
            .add(rhs_samples)
        )
        hits = (
            jnp.zeros((self.pixel_count,), dtype=jnp.int32)
            .at[pointing.pixel_indices]
            .add((~pointing.flags).astype(jnp.int32))
        )
        map_, determinant = self._solve_symmetric(normal, rhs, self.determinant_tolerance)
        selected = jnp.abs(determinant) > self.determinant_tolerance
        map_ = jnp.where(selected[:, None], map_, 0.0)
        finite = jnp.all(jnp.isfinite(map_))
        successful = finite & jnp.all(selected)
        evidence = CmbMapmakingEvidence(hits, determinant, selected, finite, successful)
        return CmbMapmakingResult(map_, evidence, successful)


class CmbBandpowerHandoff(StrictModule):
    theory: TheoryVector
    observation: LinearObservationPlan
    binned: TheoryVector
    handoff_id: str = eqx.field(static=True)

    def __init__(
        self,
        raw_values: ArrayLike,
        raw_layout: CoordinateLayout,
        binning_matrix: ArrayLike,
        binned_layout: CoordinateLayout,
        parent_product_id: str,
        /,
    ):
        theory = TheoryVector(raw_values, raw_layout, parent_product_id)
        observation = LinearObservationPlan(binning_matrix, raw_layout, binned_layout)
        binned = observation.apply(theory)
        self.theory = theory
        self.observation = observation
        self.binned = binned
        self.handoff_id = canonical_fingerprint(
            {
                "kind": "cmb-bandpower-handoff",
                "parent": parent_product_id,
                "observation": observation.plan_id,
            }
        )


__all__ = [
    "CmbBandpowerHandoff",
    "CmbBeamProduct",
    "CmbIngressEvidence",
    "CmbIngressPlan",
    "CmbMapmakingEvidence",
    "CmbMapmakingPlan",
    "CmbMapmakingResult",
    "CmbPointingProduct",
    "CmbSkyMapProduct",
    "CmbTodProduct",
    "CmbTodSimulationPlan",
    "HarmonicSkySynthesisPlan",
]
