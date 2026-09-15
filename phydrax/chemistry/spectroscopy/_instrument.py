#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared spectroscopy instruments separated from raw forward responses."""

from __future__ import annotations

from enum import StrEnum
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...observation import CoordinateLayout, TheoryVector
from ...signal import ConvolutionMethod, convolve
from ._profile import SpectralProfilePlan
from ._response import (
    SpectralResponseProduct,
    SpectralResponseRepresentation,
)


class SpectralInstrumentKind(StrEnum):
    LINE_PROFILE = "line-profile"
    STATIONARY_KERNEL = "stationary-kernel"


class SpectralInstrumentPlan(StrictModule, NonTrainableState):
    """Static instrument route with bounded channel and spectral capacities."""

    kind: SpectralInstrumentKind = eqx.field(static=True)
    source_profile_id: str = eqx.field(static=True)
    channel_capacity: int = eqx.field(static=True)
    coordinate_capacity: int = eqx.field(static=True)
    kernel_capacity: int = eqx.field(static=True)
    convolution_method: ConvolutionMethod = eqx.field(static=True)
    area_tolerance: float = eqx.field(static=True)
    profile: SpectralProfilePlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: SpectralInstrumentKind,
        source_profile_id: str,
        channel_capacity: int,
        coordinate_capacity: int,
        /,
        *,
        kernel_capacity: int = 0,
        convolution_method: ConvolutionMethod = "direct",
        area_tolerance: float = 5.0e-3,
        profile: SpectralProfilePlan | None = None,
    ):
        profile_id = str(source_profile_id).strip()
        channels = int(channel_capacity)
        coordinates = int(coordinate_capacity)
        kernels = int(kernel_capacity)
        tolerance = float(area_tolerance)
        if (
            not isinstance(kind, SpectralInstrumentKind)
            or not profile_id
            or channels <= 0
            or coordinates <= 0
            or kernels < 0
            or convolution_method not in ("direct", "fft")
            or not isfinite(tolerance)
            or tolerance <= 0.0
        ):
            raise ValueError("Spectral instrument plan parameters are invalid.")
        if kind is SpectralInstrumentKind.LINE_PROFILE:
            if not isinstance(profile, SpectralProfilePlan) or kernels != 0:
                raise ValueError(
                    "Line instruments require exactly one SpectralProfilePlan."
                )
            if coordinates != profile.grid_size:
                raise ValueError(
                    "Line instrument coordinate capacity must equal profile grid size."
                )
        elif profile is not None or kernels <= 0 or coordinates < 2:
            raise ValueError(
                "Stationary instruments require at least two coordinates and a positive kernel capacity only."
            )
        self.kind = kind
        self.source_profile_id = profile_id
        self.channel_capacity = channels
        self.coordinate_capacity = coordinates
        self.kernel_capacity = kernels
        self.convolution_method = convolution_method
        self.area_tolerance = tolerance
        self.profile = profile
        self.plan_id = canonical_fingerprint(
            {
                "kind": "spectral-instrument-plan",
                "instrument_kind": kind.value,
                "source_profile": profile_id,
                "channel_capacity": channels,
                "coordinate_capacity": coordinates,
                "kernel_capacity": kernels,
                "convolution_method": convolution_method,
                "area_tolerance": tolerance,
                "profile": None if profile is None else profile.plan_id,
            }
        )


class PreparedSpectralInstrument(StrictModule, NonTrainableState):
    """Numeric stationary kernel prepared against an immutable instrument plan."""

    plan: SpectralInstrumentPlan
    normalized_kernel: Array
    kernel_normalization_residual: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: SpectralInstrumentPlan,
        normalized_kernel: ArrayLike,
        kernel_normalization_residual: ArrayLike,
        /,
    ):
        kernel = jnp.asarray(normalized_kernel)
        residual = jnp.asarray(kernel_normalization_residual, dtype=kernel.dtype).reshape(
            ()
        )
        expected = (
            (0,)
            if plan.kind is SpectralInstrumentKind.LINE_PROFILE
            else (plan.kernel_capacity,)
        )
        if kernel.shape != expected:
            raise ValueError("Prepared kernel does not match instrument route capacity.")
        self.plan = plan
        self.normalized_kernel = kernel
        self.kernel_normalization_residual = residual
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-spectral-instrument",
                "plan": plan.plan_id,
                "kernel": array_tree_fingerprint(np.asarray(kernel)),
                "kernel_normalization_residual": float(residual),
            }
        )


class SpectralInstrumentEvidence(StrictModule, NonTrainableState):
    kernel_normalization_residual: Array
    area_residual: Array
    finite_window_loss: Array
    successful: Array
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel_normalization_residual: ArrayLike,
        area_residual: ArrayLike,
        finite_window_loss: ArrayLike,
        successful: ArrayLike,
        /,
    ):
        residuals = jnp.asarray(
            [kernel_normalization_residual, area_residual, finite_window_loss],
            dtype=float,
        ).reshape((3,))
        if bool(jnp.any(~jnp.isfinite(residuals))) or bool(jnp.any(residuals < 0.0)):
            raise ValueError("Instrument residuals must be finite and non-negative.")
        self.kernel_normalization_residual = residuals[0]
        self.area_residual = residuals[1]
        self.finite_window_loss = residuals[2]
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "spectral-instrument-evidence",
                "residuals": array_tree_fingerprint(np.asarray(residuals)),
                "successful": bool(self.successful),
            }
        )


class SpectralInstrumentResult(StrictModule, NonTrainableState):
    """Convolved spectrum retaining its non-substitutable raw parent."""

    raw_response: SpectralResponseProduct
    coordinates: Array
    convolved_values: Array
    theory: TheoryVector
    evidence: SpectralInstrumentEvidence
    prepared_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        raw_response: SpectralResponseProduct,
        coordinates: ArrayLike,
        convolved_values: ArrayLike,
        theory: TheoryVector,
        evidence: SpectralInstrumentEvidence,
        prepared_id: str,
        /,
    ):
        coordinate = jnp.asarray(coordinates)
        values = jnp.asarray(convolved_values)
        if values.shape != (len(raw_response.channels), coordinate.size):
            raise ValueError("Instrument output channels and coordinates must align.")
        self.raw_response = raw_response
        self.coordinates = coordinate
        self.convolved_values = values
        self.theory = theory
        self.evidence = evidence
        self.prepared_id = str(prepared_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "spectral-instrument-result",
                "raw": raw_response.product_id,
                "prepared": self.prepared_id,
                "theory": theory.product_id,
                "evidence": evidence.evidence_id,
            }
        )


def plan_spectral_instrument(
    kind: SpectralInstrumentKind,
    source_profile_id: str,
    channel_capacity: int,
    coordinate_capacity: int,
    /,
    **kwargs: object,
) -> SpectralInstrumentPlan:
    return SpectralInstrumentPlan(
        kind,
        source_profile_id,
        channel_capacity,
        coordinate_capacity,
        **kwargs,
    )


def prepare_spectral_instrument(
    plan: SpectralInstrumentPlan,
    kernel: ArrayLike | None = None,
    /,
) -> PreparedSpectralInstrument:
    if plan.kind is SpectralInstrumentKind.LINE_PROFILE:
        if kernel is not None:
            raise ValueError("A line-profile instrument does not accept a second kernel.")
        return PreparedSpectralInstrument(plan, jnp.empty((0,), dtype=float), 0.0)
    if kernel is None:
        raise ValueError("A stationary instrument requires explicit resolution taps.")
    taps = jnp.asarray(kernel)
    if taps.shape != (plan.kernel_capacity,):
        raise ValueError("Resolution taps do not match kernel capacity.")
    if bool(jnp.any(~jnp.isfinite(taps))) or bool(jnp.any(taps < 0.0)):
        raise ValueError("Resolution taps must be finite and non-negative.")
    total = jnp.sum(taps)
    if bool(total <= 0.0):
        raise ValueError("Resolution taps must have positive total weight.")
    normalized = taps / total
    return PreparedSpectralInstrument(plan, normalized, jnp.abs(total - 1.0))


def refresh_spectral_instrument(
    prepared: PreparedSpectralInstrument,
    kernel: ArrayLike,
    /,
) -> PreparedSpectralInstrument:
    if prepared.plan.kind is not SpectralInstrumentKind.STATIONARY_KERNEL:
        raise ValueError(
            "Only a stationary-kernel instrument has refreshable numeric taps."
        )
    return prepare_spectral_instrument(prepared.plan, kernel)


def _integrated_density(coordinates: Array, values: Array) -> Array:
    return jnp.trapezoid(values, coordinates, axis=-1)


def apply_spectral_instrument(
    prepared: PreparedSpectralInstrument,
    response: SpectralResponseProduct,
    /,
) -> SpectralInstrumentResult:
    plan = prepared.plan
    if response.source_profile_id != plan.source_profile_id:
        raise ValueError("Raw response profile does not match the instrument plan.")
    if len(response.channels) > plan.channel_capacity:
        raise ValueError("Raw response exceeds the planned channel capacity.")
    if not bool(response.evidence.successful):
        raise ValueError("An unsuccessful raw response cannot enter an instrument.")

    if plan.kind is SpectralInstrumentKind.LINE_PROFILE:
        if response.representation is not SpectralResponseRepresentation.LINES:
            raise TypeError(
                "A line-profile instrument requires an integrated-line response."
            )
        if plan.profile is None:
            raise RuntimeError("Line-profile instrument invariant failed.")
        profiles = [
            plan.profile.evaluate(
                np.asarray(response.coordinates)[np.asarray(response.active)],
                np.asarray(response.values[channel])[np.asarray(response.active)],
            )
            for channel in range(len(response.channels))
        ]
        coordinates = jnp.asarray(profiles[0].grid)
        values = jnp.stack([profile.intensity_density for profile in profiles])
        expected = jnp.sum(
            jnp.where(response.active[None, :], response.values, 0.0), axis=-1
        )
        integrated = _integrated_density(coordinates, values)
        successful_profiles = all(bool(profile.successful) for profile in profiles)
        kernel_residual = jnp.asarray(0.0, dtype=values.dtype)
    else:
        if response.representation is not SpectralResponseRepresentation.DENSITY:
            raise TypeError(
                "A stationary instrument requires a gridded density response."
            )
        if response.coordinates.size != plan.coordinate_capacity:
            raise ValueError("Raw grid does not match the planned coordinate capacity.")
        spacing = jnp.diff(response.coordinates)
        if bool(jnp.any(jnp.abs(spacing - spacing[0]) > 1.0e-10 * jnp.abs(spacing[0]))):
            raise ValueError("Stationary resolution requires a uniformly spaced grid.")
        coordinates = response.coordinates
        values = convolve(
            response.values,
            prepared.normalized_kernel,
            axis=-1,
            mode="same",
            method=plan.convolution_method,
        )
        expected = _integrated_density(coordinates, response.values)
        integrated = _integrated_density(coordinates, values)
        successful_profiles = True
        kernel_residual = prepared.kernel_normalization_residual

    denominator = jnp.maximum(
        jnp.abs(expected), jnp.asarray(jnp.finfo(values.dtype).tiny, dtype=values.dtype)
    )
    per_channel_loss = jnp.abs(expected - integrated) / denominator
    area_residual = jnp.max(per_channel_loss)
    finite_window_loss = jnp.max(jnp.maximum(expected - integrated, 0.0) / denominator)
    successful = (
        successful_profiles
        and bool(jnp.all(jnp.isfinite(values)))
        and bool(area_residual <= plan.area_tolerance)
    )
    evidence = SpectralInstrumentEvidence(
        kernel_residual,
        area_residual,
        finite_window_loss,
        successful,
    )
    labels = tuple(
        f"{channel}@{float(coordinate):.17g}"
        for channel in response.channels
        for coordinate in np.asarray(coordinates)
    )
    layout = CoordinateLayout(labels)
    theory = TheoryVector(
        values.reshape((-1,)),
        layout,
        canonical_fingerprint(
            {
                "kind": "spectral-theory-vector",
                "raw": response.product_id,
                "instrument": prepared.prepared_id,
            }
        ),
    )
    return SpectralInstrumentResult(
        response,
        coordinates,
        values,
        theory,
        evidence,
        prepared.prepared_id,
    )


__all__ = [
    "PreparedSpectralInstrument",
    "SpectralInstrumentEvidence",
    "SpectralInstrumentKind",
    "SpectralInstrumentPlan",
    "SpectralInstrumentResult",
    "apply_spectral_instrument",
    "plan_spectral_instrument",
    "prepare_spectral_instrument",
    "refresh_spectral_instrument",
]
