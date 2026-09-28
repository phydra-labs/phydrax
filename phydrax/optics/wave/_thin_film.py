#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-layer ambient-film-substrate interference by exact Airy summation."""

from __future__ import annotations

import math
from enum import IntFlag
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import fixed_field, parameter_field
from ...ein import contract
from ...typing import Bool, Complex, Dim, Float, Identifier, Int32, VariadicDim
from ..geometric._interface import _fresnel_amplitudes
from ..materials._refractive_index import _passive_square_root


class _WavelengthDim(Dim, minimum=1):
    """Vacuum-wavelength samples of one interference plan."""


class _IntervalDim(Dim):
    """Adjacent vacuum-wavelength sample intervals."""


class _SampleDims(VariadicDim):
    """Broadcast film-sample axes of one evaluation."""


# Natural (unpolarized) light is an equal incoherent mixture of s and p.
_NATURAL_LIGHT_WEIGHTS = (0.5, 0.5)


class ThinFilmInterferenceStatus(IntFlag):
    """Fail-closed per-sample status bits of one thin-film evaluation."""

    SUCCESS = 0
    INVALID_THICKNESS = 1
    INVALID_INCIDENCE = 2
    SPECTRAL_UNDERSAMPLED = 4
    ENERGY_RESIDUAL_EXCEEDED = 8
    NONFINITE = 16


class ThinFilmInterferenceEvidence(StrictModule):
    """Spectral-sampling, energy, regime, and validity evidence per film sample.

    ``round_trip_phase`` is ``4 pi d Re(n_1 cos theta_1) / lambda``. Over each
    adjacent wavelength interval the exact phase advance ``|Delta phi|`` gives
    ``2 pi / |Delta phi|`` samples per interference fringe; for a nondispersive
    film this is ``lambda^2 / (2 n_1 d cos theta_1 Delta lambda)``, i.e. the
    fringe period ``Delta lambda_fringe ~ lambda^2 / (2 n_1 d cos theta_1)``
    divided by the sample spacing. A single-wavelength plan has no intervals and
    no spectral-quadrature claim.
    """

    __strict_contract__ = True

    round_trip_phase: Float[_SampleDims, _WavelengthDim]
    interval_samples_per_fringe: Float[_SampleDims, _IntervalDim]
    minimum_samples_per_fringe: Float[_SampleDims]
    spectral_sampling_resolved: Bool[_SampleDims]
    maximum_energy_residual: Float[_SampleDims]
    energy_conserved: Bool[_SampleDims]
    film_evanescent: Bool[_SampleDims, _WavelengthDim]
    substrate_evanescent: Bool[_SampleDims, _WavelengthDim]
    finite: Bool[_SampleDims]
    accepted: Bool[_SampleDims]
    status: Int32[_SampleDims]
    required_samples_per_fringe: float | None = eqx.field(static=True)
    energy_tolerance: float = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)


class ThinFilmInterferenceResult(StrictModule):
    """Complex Airy amplitudes, flux coefficients, and interference evidence.

    The final polarization axis is ordered ``(s, p)`` with the amplitude and p
    basis conventions of
    :class:`phydrax.optics.geometric.RefractiveInterfaceResult`. Transmittance
    is the normal Poynting flux entering the substrate divided by the incident
    flux, so a lossless ambient and film give ``R + T = 1`` for every passive
    substrate. Every rejected sample carries NaN spectral outputs; finite
    quality diagnostics remain available in :attr:`evidence`.
    """

    __strict_contract__ = True

    reflection_amplitudes: Complex[_SampleDims, _WavelengthDim, Literal[2]]
    transmission_amplitudes: Complex[_SampleDims, _WavelengthDim, Literal[2]]
    reflectance: Float[_SampleDims, _WavelengthDim, Literal[2]]
    transmittance: Float[_SampleDims, _WavelengthDim, Literal[2]]
    unpolarized_reflectance: Float[_SampleDims, _WavelengthDim]
    unpolarized_transmittance: Float[_SampleDims, _WavelengthDim]
    energy_residual: Float[_SampleDims, _WavelengthDim, Literal[2]]
    film_cosine: Complex[_SampleDims, _WavelengthDim]
    substrate_cosine: Complex[_SampleDims, _WavelengthDim]
    evidence: ThinFilmInterferenceEvidence

    @property
    def status(self) -> Array:
        return self.evidence.status

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted


def _wavelength_grid(value: ArrayLike, /) -> np.ndarray:
    grid = np.asarray(value)
    if grid.ndim != 1 or grid.shape[0] < 1:
        raise ValueError("wavelengths must be a non-empty rank-one array.")
    if not np.issubdtype(grid.dtype, np.floating):
        raise TypeError("wavelengths must be real floating vacuum wavelengths in m.")
    if not np.all(np.isfinite(grid)) or not np.all(grid > 0.0):
        raise ValueError("wavelengths must be finite and positive.")
    if not np.all(np.diff(grid) > 0.0):
        raise ValueError("wavelengths must be strictly increasing.")
    return grid


def _real_spectral_index(value: ArrayLike, count: int, name: str, /) -> np.ndarray:
    index = np.asarray(value)
    if not np.issubdtype(index.dtype, np.floating):
        if np.issubdtype(index.dtype, np.integer):
            index = index.astype(np.float64)
        else:
            raise TypeError(f"{name} must be a real refractive index.")
    if index.shape not in ((), (count,)):
        raise ValueError(f"{name} must be a scalar or one value per wavelength.")
    if not np.all(np.isfinite(index)) or not np.all(index > 0.0):
        raise ValueError(f"{name} must be finite and positive.")
    return np.broadcast_to(index, (count,))


def _passive_spectral_index(value: ArrayLike, count: int, /) -> np.ndarray:
    index = np.asarray(value)
    if not np.issubdtype(index.dtype, np.number) or np.issubdtype(index.dtype, np.bool_):
        raise TypeError("substrate_index must be a numeric refractive index.")
    if index.shape not in ((), (count,)):
        raise ValueError("substrate_index must be a scalar or one value per wavelength.")
    real = np.real(index)
    imaginary = np.imag(index)
    if not np.all(np.isfinite(real) & np.isfinite(imaginary)):
        raise ValueError("substrate_index must be finite.")
    if not np.all((imaginary > 0.0) | ((imaginary == 0.0) & (real > 0.0))):
        raise ValueError(
            "substrate_index must be passive in the exp(-i omega t) convention: "
            "Im(n) > 0, or Im(n) = 0 and Re(n) > 0."
        )
    return np.broadcast_to(index, (count,))


def _optional_positive(value: float | None, name: str, /) -> float | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def _real_input(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.integer):
        return array.astype(jnp.float32)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        raise TypeError(f"{name} must be real numeric data.")
    return array


class ThinFilmInterferencePlan(StrictModule):
    """One homogeneous film between semi-infinite ambient and substrate media.

    Wavelengths are strictly increasing vacuum wavelengths in meters. The
    ambient and film indices are real and positive (lossless); the substrate
    index may be complex and must be passive in the ``exp(-i omega t)``
    convention (``Im(n) >= 0``). Each index is a scalar or one value per
    wavelength, so dispersive laws enter as samples, for example from
    :func:`phydrax.optics.materials.evaluate_refractive_index` at
    ``omega = 2 pi c / lambda`` after the caller checks its ``accepted`` flags.
    The film and substrate indices are inferable parameters; wavelengths and the
    ambient index are fixed.

    Evaluation sums the multiple-beam (Airy) series exactly::

        r = (r01 + r12 e^{2 i beta}) / (1 + r01 r12 e^{2 i beta})
        t = t01 t12 e^{i beta} / (1 + r01 r12 e^{2 i beta})
        beta = 2 pi d n_1 cos(theta_1) / lambda

    with complex cosines on the passive branch, which covers frustrated total
    internal reflection in the film and evanescent or absorbing substrates.
    ``required_samples_per_fringe`` (default 4) is the spectral-quadrature
    admission; ``None`` records the evidence without enforcing it (discrete
    lines). ``energy_tolerance`` bounds ``|R + T - 1|``; ``None`` uses
    ``256 eps`` of the evaluation dtype.
    """

    __strict_contract__ = True

    wavelengths: Float[_WavelengthDim] = fixed_field()
    ambient_index: Float[_WavelengthDim] = fixed_field()
    film_index: Float[_WavelengthDim] = parameter_field()
    substrate_index: Complex[_WavelengthDim] = parameter_field()
    required_samples_per_fringe: float | None = eqx.field(static=True)
    energy_tolerance: float | None = eqx.field(static=True)
    plan_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        wavelengths: ArrayLike,
        ambient_index: ArrayLike,
        film_index: ArrayLike,
        substrate_index: ArrayLike,
        /,
        *,
        required_samples_per_fringe: float | None = 4.0,
        energy_tolerance: float | None = None,
    ) -> None:
        grid = _wavelength_grid(wavelengths)
        count = grid.shape[0]
        ambient = _real_spectral_index(ambient_index, count, "ambient_index")
        film = _real_spectral_index(film_index, count, "film_index")
        substrate = _passive_spectral_index(substrate_index, count)
        samples = _optional_positive(
            required_samples_per_fringe, "required_samples_per_fringe"
        )
        tolerance = _optional_positive(energy_tolerance, "energy_tolerance")
        dtype = np.result_type(
            grid.dtype, ambient.dtype, film.dtype, np.real(substrate).dtype
        )
        complex_dtype = np.result_type(dtype, np.complex64)
        self.wavelengths = jnp.asarray(grid, dtype=dtype)
        self.ambient_index = jnp.asarray(ambient, dtype=dtype)
        self.film_index = jnp.asarray(film, dtype=dtype)
        self.substrate_index = jnp.asarray(substrate, dtype=complex_dtype)
        self.required_samples_per_fringe = samples
        self.energy_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "thin-film-interference-plan",
                "wavelengths": grid.astype(dtype),
                "dtype": np.dtype(dtype).name,
                "required_samples_per_fringe": None
                if samples is None
                else float(samples).hex(),
                "energy_tolerance": None if tolerance is None else float(tolerance).hex(),
            }
        )

    def evaluate(
        self, thickness: ArrayLike, incidence_cosine: ArrayLike, /
    ) -> ThinFilmInterferenceResult:
        """Evaluate film samples of physical thickness ``d`` (m) at one incidence.

        ``incidence_cosine`` is ``cos(theta_0)`` in the ambient medium, in
        ``(0, 1]``. Thickness and incidence broadcast to the sample shape ``B``;
        spectral outputs have shape ``B + (W,)`` and, per polarization,
        ``B + (W, 2)``. The evaluation is differentiable in thickness,
        incidence and the parameter indices.
        """
        thickness_ = _real_input(thickness, "thickness")
        cosine_ = _real_input(incidence_cosine, "incidence_cosine")
        batch_shape = jnp.broadcast_shapes(thickness_.shape, cosine_.shape)
        dtype = jnp.result_type(self.wavelengths.dtype, thickness_.dtype, cosine_.dtype)
        complex_dtype = jnp.result_type(dtype, jnp.complex64)
        thickness_ = jnp.broadcast_to(thickness_.astype(dtype), batch_shape)
        cosine_ = jnp.broadcast_to(cosine_.astype(dtype), batch_shape)
        thickness_valid = jnp.isfinite(thickness_) & (thickness_ >= 0.0)
        incidence_valid = jnp.isfinite(cosine_) & (cosine_ > 0.0) & (cosine_ <= 1.0)
        inputs_valid = thickness_valid & incidence_valid
        depth = jnp.where(thickness_valid, thickness_, 0.0)[..., None]
        cosine0 = jnp.where(incidence_valid, cosine_, 1.0)[..., None]

        wavelengths = self.wavelengths.astype(dtype)
        ambient = self.ambient_index.astype(dtype)
        film = self.film_index.astype(dtype)
        substrate = self.substrate_index.astype(complex_dtype)
        # Conserved tangential wavenumber (n_0 sin theta_0)^2 in vacuum units.
        tangential = ambient * ambient * (1.0 - cosine0 * cosine0)
        normal0 = (ambient * cosine0).astype(complex_dtype)
        normal1 = _passive_square_root((film * film - tangential).astype(complex_dtype))
        normal2 = _passive_square_root(substrate * substrate - tangential)
        ambient_c = ambient.astype(complex_dtype)
        film_c = film.astype(complex_dtype)
        cosine0_c = jnp.broadcast_to(cosine0, normal0.shape).astype(complex_dtype)
        cosine1 = normal1 / film_c
        cosine2 = normal2 / substrate
        r01, t01 = _fresnel_amplitudes(ambient_c, film_c, cosine0_c, cosine1)
        r12, t12 = _fresnel_amplitudes(film_c, substrate, cosine1, cosine2)

        phase = (2.0 * jnp.pi / wavelengths) * normal1 * depth
        single_pass = jnp.exp(1j * phase)[..., None]
        round_trip = single_pass * single_pass
        denominator = 1.0 + r01 * r12 * round_trip
        reflection = (r01 + r12 * round_trip) / denominator
        transmission = t01 * t12 * single_pass / denominator
        flux_ratio = (
            jnp.stack(
                (jnp.real(normal2), jnp.real(jnp.conj(substrate) * cosine2)), axis=-1
            )
            / jnp.real(normal0)[..., None]
        )
        reflectance = jnp.real(reflection * jnp.conj(reflection)).astype(dtype)
        transmittance = (
            flux_ratio * jnp.real(transmission * jnp.conj(transmission))
        ).astype(dtype)
        weights = jnp.asarray(_NATURAL_LIGHT_WEIGHTS, dtype=dtype)
        unpolarized_reflectance = contract("...p,p->...", reflectance, weights)
        unpolarized_transmittance = contract("...p,p->...", transmittance, weights)
        energy_residual = jnp.abs(reflectance + transmittance - 1.0)

        round_trip_phase = 2.0 * jnp.real(phase).astype(dtype)
        phase_step = jnp.abs(jnp.diff(round_trip_phase, axis=-1))
        interval_samples = jnp.where(
            phase_step > 0.0,
            2.0 * jnp.pi / jnp.where(phase_step > 0.0, phase_step, 1.0),
            jnp.inf,
        )
        minimum_samples = jnp.min(interval_samples, axis=-1, initial=jnp.inf)
        resolved = (
            jnp.ones(batch_shape, dtype=jnp.bool_)
            if self.required_samples_per_fringe is None
            else minimum_samples >= self.required_samples_per_fringe
        )
        tolerance = (
            256.0 * float(jnp.finfo(dtype).eps)
            if self.energy_tolerance is None
            else self.energy_tolerance
        )
        maximum_residual = jnp.max(energy_residual, axis=(-2, -1))
        conserved = maximum_residual <= tolerance
        finite = (
            jnp.all(jnp.isfinite(reflection), axis=(-2, -1))
            & jnp.all(jnp.isfinite(transmission), axis=(-2, -1))
            & jnp.all(jnp.isfinite(transmittance), axis=(-2, -1))
            & jnp.all(jnp.isfinite(round_trip_phase), axis=-1)
        )
        status = _status(thickness_valid, incidence_valid, resolved, conserved, finite)
        accepted = status == int(ThinFilmInterferenceStatus.SUCCESS)

        def masked(values: Array, samples: Array, /) -> Array:
            trailing = (1,) * (values.ndim - len(batch_shape))
            mask = samples.reshape(batch_shape + trailing)
            return jnp.where(mask, values, jnp.nan)

        evidence = ThinFilmInterferenceEvidence(
            round_trip_phase=masked(round_trip_phase, inputs_valid),
            interval_samples_per_fringe=masked(interval_samples, inputs_valid),
            minimum_samples_per_fringe=masked(minimum_samples, inputs_valid),
            spectral_sampling_resolved=resolved & inputs_valid,
            maximum_energy_residual=masked(maximum_residual, inputs_valid),
            energy_conserved=conserved & inputs_valid,
            film_evanescent=(jnp.real(normal1) <= 0.0) & inputs_valid[..., None],
            substrate_evanescent=(jnp.real(normal2) <= 0.0) & inputs_valid[..., None],
            finite=finite & inputs_valid,
            accepted=accepted,
            status=status,
            required_samples_per_fringe=self.required_samples_per_fringe,
            energy_tolerance=tolerance,
            plan_id=self.plan_id,
        )
        return ThinFilmInterferenceResult(
            reflection_amplitudes=masked(reflection, accepted),
            transmission_amplitudes=masked(transmission, accepted),
            reflectance=masked(reflectance, accepted),
            transmittance=masked(transmittance, accepted),
            unpolarized_reflectance=masked(unpolarized_reflectance, accepted),
            unpolarized_transmittance=masked(unpolarized_transmittance, accepted),
            energy_residual=masked(energy_residual, accepted),
            film_cosine=masked(cosine1, accepted),
            substrate_cosine=masked(cosine2, accepted),
            evidence=evidence,
        )


def _status(
    thickness_valid: Array,
    incidence_valid: Array,
    resolved: Array,
    conserved: Array,
    finite: Array,
    /,
) -> Array:
    inputs_valid = thickness_valid & incidence_valid
    bits = (
        (~thickness_valid, ThinFilmInterferenceStatus.INVALID_THICKNESS),
        (~incidence_valid, ThinFilmInterferenceStatus.INVALID_INCIDENCE),
        (inputs_valid & ~resolved, ThinFilmInterferenceStatus.SPECTRAL_UNDERSAMPLED),
        (
            inputs_valid & finite & ~conserved,
            ThinFilmInterferenceStatus.ENERGY_RESIDUAL_EXCEEDED,
        ),
        (inputs_valid & ~finite, ThinFilmInterferenceStatus.NONFINITE),
    )
    status = jnp.zeros(thickness_valid.shape, dtype=jnp.int32)
    for condition, flag in bits:
        status = status | jnp.where(condition, int(flag), 0).astype(jnp.int32)
    return status


__all__ = [
    "ThinFilmInterferenceEvidence",
    "ThinFilmInterferencePlan",
    "ThinFilmInterferenceResult",
    "ThinFilmInterferenceStatus",
]
