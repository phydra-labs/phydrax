#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._fields import PlaneFieldSpace


class PhaseScreenStatus(IntEnum):
    SUCCESS = 0
    NONFINITE = 1
    HERMITIAN_DEFECT = 2
    PARSEVAL_DEFECT = 3


class VonKarmanPhaseScreenPlan(StrictModule, NonTrainableState):
    """A fixed periodic support and physical von Kármán phase PSD."""

    space: PlaneFieldSpace
    fried_parameter: Array
    outer_scale: Array
    inner_scale: Array
    remove_piston: bool = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        fried_parameter: ArrayLike,
        outer_scale: ArrayLike,
        /,
        *,
        inner_scale: ArrayLike = 0.0,
        remove_piston: bool = True,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        if space.topology != "periodic-cell":
            raise ValueError("Phase screens require a periodic-cell field space.")
        fried = jnp.asarray(fried_parameter, dtype=float)
        outer = jnp.asarray(outer_scale, dtype=float)
        inner = jnp.asarray(inner_scale, dtype=float)
        if fried.shape != () or outer.shape != () or inner.shape != ():
            raise ValueError("Atmospheric length scales must be scalar.")
        fried = eqx.error_if(
            fried,
            (~jnp.isfinite(fried)) | (fried <= 0.0),
            "fried_parameter must be finite and positive.",
        )
        outer = eqx.error_if(
            outer,
            (~jnp.isfinite(outer)) | (outer <= 0.0),
            "outer_scale must be finite and positive.",
        )
        inner = eqx.error_if(
            inner,
            (~jnp.isfinite(inner)) | (inner < 0.0),
            "inner_scale must be finite and nonnegative.",
        )
        self.space = space
        self.fried_parameter = fried
        self.outer_scale = outer
        self.inner_scale = inner
        self.remove_piston = bool(remove_piston)

    def prepare(self, /) -> "PreparedVonKarmanPhaseScreen":
        lengths = jnp.stack(
            tuple(
                axis.bounds[1] - axis.bounds[0]
                for axis in self.space.grid.structured_axes
            )
        )
        shape_array = jnp.asarray(self.space.shape, dtype=lengths.dtype)
        spacings = lengths / shape_array
        frequency_axes = tuple(
            jnp.fft.fftfreq(count, d=spacing)
            for count, spacing in zip(self.space.shape, spacings, strict=True)
        )
        frequency_mesh = jnp.meshgrid(*frequency_axes, indexing="ij")
        spatial_frequencies = jnp.stack(frequency_mesh, axis=-1)
        radial_squared = jnp.sum(spatial_frequencies * spatial_frequencies, axis=-1)
        outer_frequency = 1.0 / self.outer_scale
        safe_inner_scale = jnp.where(self.inner_scale > 0.0, self.inner_scale, 1.0)
        inner_frequency = jnp.where(
            self.inner_scale > 0.0,
            5.92 / (2.0 * jnp.pi * safe_inner_scale),
            jnp.inf,
        )
        power_spectral_density = (
            0.023
            * self.fried_parameter ** (-5.0 / 3.0)
            * (radial_squared + outer_frequency**2) ** (-11.0 / 6.0)
            * jnp.exp(-radial_squared / inner_frequency**2)
        )
        supported = jnp.ones(self.space.shape, dtype=bool)
        for axis, count in enumerate(self.space.shape):
            if count % 2 == 0:
                axis_indices = jnp.arange(count)
                reshape = [1, 1]
                reshape[axis] = count
                supported = supported & (
                    axis_indices.reshape(tuple(reshape)) != count // 2
                )
        if self.remove_piston:
            supported = supported.at[0, 0].set(False)
        power_spectral_density = jnp.where(supported, power_spectral_density, 0.0)
        cell_area = spacings[0] * spacings[1]
        spectral_filter = jnp.sqrt(power_spectral_density / cell_area)
        domain_area = lengths[0] * lengths[1]
        predicted_variance = jnp.sum(power_spectral_density) / domain_area
        return PreparedVonKarmanPhaseScreen(
            self,
            spatial_frequencies,
            frequency_axes,
            power_spectral_density,
            spectral_filter,
            supported,
            spacings,
            lengths,
            predicted_variance,
        )


class PreparedVonKarmanPhaseScreen(StrictModule, NonTrainableState):
    plan: VonKarmanPhaseScreenPlan
    spatial_frequencies: Array
    frequency_axes: tuple[Array, Array]
    power_spectral_density: Array
    spectral_filter: Array
    supported_modes: Array
    spacings: Array
    lengths: Array
    predicted_variance: Array

    def sample(self, key: Array, /) -> "PhaseScreenSample":
        return sample_von_karman_phase_screen(self, key)


class PhaseScreenEvidence(StrictModule, NonTrainableState):
    predicted_variance: Array
    realized_variance: Array
    piston: Array
    hermitian_error: Array
    parseval_relative_error: Array
    finite: Array
    valid: Array
    status: Array


class PhaseScreenSample(StrictModule):
    prepared: PreparedVonKarmanPhaseScreen
    phase: Array
    spectral_coefficients: Array
    power_scale: Array
    evidence: PhaseScreenEvidence

    @property
    def valid(self) -> Array:
        return self.evidence.valid

    def advect(
        self,
        time: ArrayLike,
        velocity: ArrayLike,
        /,
    ) -> "PhaseScreenSample":
        return frozen_flow_phase_screen(self, time, velocity)


class AtmosphericLayer(StrictModule, NonTrainableState):
    screen: VonKarmanPhaseScreenPlan
    altitude: Array
    velocity: Array
    strength_fraction: Array
    layer_id: str = eqx.field(static=True)

    def __init__(
        self,
        screen: VonKarmanPhaseScreenPlan,
        altitude: ArrayLike,
        velocity: ArrayLike,
        strength_fraction: ArrayLike,
        /,
        *,
        layer_id: str,
    ):
        if not isinstance(screen, VonKarmanPhaseScreenPlan):
            raise TypeError("screen must be a VonKarmanPhaseScreenPlan.")
        altitude_ = jnp.asarray(altitude, dtype=float)
        velocity_ = jnp.asarray(velocity, dtype=float)
        strength = jnp.asarray(strength_fraction, dtype=float)
        if altitude_.shape != () or velocity_.shape != (2,) or strength.shape != ():
            raise ValueError(
                "Layer altitude/strength are scalar and velocity has shape (2,)."
            )
        altitude_ = eqx.error_if(
            altitude_,
            (~jnp.isfinite(altitude_)) | (altitude_ < 0.0),
            "Layer altitude must be finite and nonnegative.",
        )
        velocity_ = eqx.error_if(
            velocity_,
            jnp.any(~jnp.isfinite(velocity_)),
            "Layer velocity must be finite.",
        )
        strength = eqx.error_if(
            strength,
            (~jnp.isfinite(strength)) | (strength <= 0.0),
            "Layer strength_fraction must be finite and positive.",
        )
        identifier = str(layer_id)
        if not identifier:
            raise ValueError("layer_id must be non-empty.")
        self.screen = screen
        self.altitude = altitude_
        self.velocity = velocity_
        self.strength_fraction = strength
        self.layer_id = identifier


class LayeredAtmosphere(StrictModule, NonTrainableState):
    layers: tuple[AtmosphericLayer, ...]

    def __init__(self, layers: Sequence[AtmosphericLayer], /):
        layers_ = tuple(layers)
        if not layers_ or not all(
            isinstance(layer, AtmosphericLayer) for layer in layers_
        ):
            raise TypeError("layers must contain one or more AtmosphericLayer records.")
        reference_space = layers_[0].screen.space.space_id
        if any(layer.screen.space.space_id != reference_space for layer in layers_):
            raise ValueError("All atmospheric layers must share one periodic support.")
        strengths = np.asarray(
            [layer.strength_fraction for layer in layers_],
            dtype=float,
        )
        if not np.isclose(np.sum(strengths), 1.0, rtol=1e-10, atol=1e-12):
            raise ValueError("Atmospheric strength fractions must sum to one.")
        identifiers = tuple(layer.layer_id for layer in layers_)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Atmospheric layer IDs must be unique.")
        self.layers = layers_

    def prepare(self, /) -> "PreparedLayeredAtmosphere":
        return PreparedLayeredAtmosphere(
            self,
            tuple(
                PreparedAtmosphericLayer(layer, layer.screen.prepare())
                for layer in self.layers
            ),
        )


class PreparedAtmosphericLayer(StrictModule, NonTrainableState):
    layer: AtmosphericLayer
    screen: PreparedVonKarmanPhaseScreen


class PreparedLayeredAtmosphere(StrictModule, NonTrainableState):
    plan: LayeredAtmosphere
    layers: tuple[PreparedAtmosphericLayer, ...]

    def sample(self, key: Array, /) -> "LayeredAtmosphereRealization":
        return sample_layered_atmosphere(self, key)


class LayerPhaseScreen(StrictModule):
    layer: PreparedAtmosphericLayer
    screen: PhaseScreenSample


class LayeredAtmosphereRealization(StrictModule):
    prepared: PreparedLayeredAtmosphere
    layers: tuple[LayerPhaseScreen, ...]

    @property
    def phase(self) -> Array:
        return sum(
            (layer.screen.phase for layer in self.layers),
            jnp.zeros_like(self.layers[0].screen.phase),
        )

    @property
    def valid(self) -> Array:
        return jnp.all(jnp.stack(tuple(layer.screen.valid for layer in self.layers)))

    def advect(self, time: ArrayLike, /) -> "LayeredAtmosphereRealization":
        return frozen_flow_layered_atmosphere(self, time)


def _hermitian_error(coefficients: Array, /) -> Array:
    first_indices = (-jnp.arange(coefficients.shape[0])) % coefficients.shape[0]
    second_indices = (-jnp.arange(coefficients.shape[1])) % coefficients.shape[1]
    partners = jnp.conj(coefficients[first_indices[:, None], second_indices[None, :]])
    scale = jnp.maximum(jnp.max(jnp.abs(coefficients)), 1.0)
    return jnp.max(jnp.abs(coefficients - partners)) / scale


def _screen_from_coefficients(
    prepared: PreparedVonKarmanPhaseScreen,
    coefficients: Array,
    /,
    *,
    power_scale: ArrayLike = 1.0,
) -> PhaseScreenSample:
    complex_phase = jnp.fft.ifft2(coefficients, norm="ortho")
    phase = jnp.real(complex_phase)
    piston = jnp.mean(phase)
    realized_variance = jnp.mean((phase - piston) ** 2)
    hermitian_error = _hermitian_error(coefficients)
    spectral_power = jnp.mean(jnp.abs(coefficients) ** 2)
    spatial_power = jnp.mean(jnp.abs(complex_phase) ** 2)
    parseval_error = jnp.abs(spectral_power - spatial_power) / jnp.maximum(
        spectral_power, jnp.finfo(spatial_power.dtype).tiny
    )
    tolerance = 256.0 * jnp.finfo(phase.dtype).eps
    finite = (
        jnp.all(jnp.isfinite(phase))
        & jnp.all(jnp.isfinite(coefficients))
        & jnp.isfinite(realized_variance)
    )
    hermitian = hermitian_error <= tolerance
    parseval = parseval_error <= tolerance
    valid = finite & hermitian & parseval
    status = jnp.where(
        ~finite,
        int(PhaseScreenStatus.NONFINITE),
        jnp.where(
            ~hermitian,
            int(PhaseScreenStatus.HERMITIAN_DEFECT),
            jnp.where(
                ~parseval,
                int(PhaseScreenStatus.PARSEVAL_DEFECT),
                int(PhaseScreenStatus.SUCCESS),
            ),
        ),
    ).astype(jnp.int32)
    evidence = PhaseScreenEvidence(
        prepared.predicted_variance * jnp.asarray(power_scale),
        realized_variance,
        piston,
        hermitian_error,
        parseval_error,
        finite,
        valid,
        status,
    )
    return PhaseScreenSample(
        prepared,
        phase,
        coefficients,
        jnp.asarray(power_scale, dtype=phase.dtype),
        evidence,
    )


def _sample_spectral_coefficients(
    prepared: PreparedVonKarmanPhaseScreen,
    key: Array,
    /,
) -> Array:
    white = jax.random.normal(
        key,
        prepared.plan.space.shape,
        dtype=prepared.power_spectral_density.dtype,
    )
    white_spectrum = jnp.fft.fft2(white, norm="ortho")
    coefficients = white_spectrum * prepared.spectral_filter
    return jnp.where(prepared.supported_modes, coefficients, 0.0)


def sample_von_karman_phase_screen(
    prepared: PreparedVonKarmanPhaseScreen,
    key: Array,
    /,
) -> PhaseScreenSample:
    """Draw a reproducible real phase screen from its prepared physical PSD."""
    if not isinstance(prepared, PreparedVonKarmanPhaseScreen):
        raise TypeError("prepared must be a PreparedVonKarmanPhaseScreen.")
    coefficients = _sample_spectral_coefficients(prepared, key)
    return _screen_from_coefficients(prepared, coefficients)


def frozen_flow_phase_screen(
    sample: PhaseScreenSample,
    time: ArrayLike,
    velocity: ArrayLike,
    /,
) -> PhaseScreenSample:
    """Translate a realization exactly on its retained Fourier support."""
    if not isinstance(sample, PhaseScreenSample):
        raise TypeError("sample must be a PhaseScreenSample.")
    time_ = jnp.asarray(time, dtype=sample.phase.dtype)
    velocity_ = jnp.asarray(velocity, dtype=sample.phase.dtype)
    if time_.shape != () or velocity_.shape != (2,):
        raise ValueError("time must be scalar and velocity must have shape (2,).")
    frequency = contract("...i,i->...", sample.prepared.spatial_frequencies, velocity_)
    phase_factor = jnp.exp(-2j * jnp.pi * time_ * frequency)
    coefficients = sample.spectral_coefficients * phase_factor
    return _screen_from_coefficients(
        sample.prepared,
        coefficients,
        power_scale=sample.power_scale,
    )


def sample_layered_atmosphere(
    prepared: PreparedLayeredAtmosphere,
    key: Array,
    /,
) -> LayeredAtmosphereRealization:
    """Sample independent layers, scaling their phase PSD by strength fraction."""
    if not isinstance(prepared, PreparedLayeredAtmosphere):
        raise TypeError("prepared must be a PreparedLayeredAtmosphere.")
    keys = jax.random.split(key, len(prepared.layers))
    realizations = []
    for layer, layer_key in zip(prepared.layers, keys, strict=True):
        coefficients = _sample_spectral_coefficients(layer.screen, layer_key)
        scale = jnp.sqrt(layer.layer.strength_fraction)
        realizations.append(
            LayerPhaseScreen(
                layer,
                _screen_from_coefficients(
                    layer.screen,
                    scale * coefficients,
                    power_scale=layer.layer.strength_fraction,
                ),
            )
        )
    return LayeredAtmosphereRealization(prepared, tuple(realizations))


def frozen_flow_layered_atmosphere(
    realization: LayeredAtmosphereRealization,
    time: ArrayLike,
    /,
) -> LayeredAtmosphereRealization:
    """Advance every immutable atmospheric layer by its exact frozen-flow phase."""
    if not isinstance(realization, LayeredAtmosphereRealization):
        raise TypeError("realization must be a LayeredAtmosphereRealization.")
    layers = tuple(
        LayerPhaseScreen(
            layer.layer,
            frozen_flow_phase_screen(
                layer.screen,
                time,
                layer.layer.layer.velocity,
            ),
        )
        for layer in realization.layers
    )
    return LayeredAtmosphereRealization(realization.prepared, layers)


__all__ = [
    "AtmosphericLayer",
    "LayerPhaseScreen",
    "LayeredAtmosphere",
    "LayeredAtmosphereRealization",
    "PhaseScreenEvidence",
    "PhaseScreenSample",
    "PhaseScreenStatus",
    "PreparedAtmosphericLayer",
    "PreparedLayeredAtmosphere",
    "PreparedVonKarmanPhaseScreen",
    "VonKarmanPhaseScreenPlan",
    "frozen_flow_layered_atmosphere",
    "frozen_flow_phase_screen",
    "sample_layered_atmosphere",
    "sample_von_karman_phase_screen",
]
