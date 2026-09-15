#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from math import erf, prod, sqrt
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._fields import (
    _angular_frequency,
    _complex_field_values,
    _longitudinal_coordinate,
    PlaneFieldSpace,
)
from ._nonlinear_response import AnalyticPulseField
from ._pulse_time import PulseTimeSpace


PulseEnvelopePolarization = Literal["scalar", "tangential"]


class PulseEnvelopeField(StrictModule):
    """Complex electric-field envelope sampled over one plane and pulse-time space.

    The physical carrier-resolved field follows ``exp(-1j * omega * t)``. Values
    contain only the envelope; the carrier is represented once by
    ``carrier_angular_frequency``. A tangential field stores components in the
    plane frame's ``(u, v)`` basis.
    """

    plane_space: PlaneFieldSpace
    time_space: PulseTimeSpace
    values: Array
    carrier_angular_frequency: Array
    longitudinal_coordinate: Array
    polarization: PulseEnvelopePolarization = eqx.field(static=True)

    def __init__(
        self,
        plane_space: PlaneFieldSpace,
        time_space: PulseTimeSpace,
        values: ArrayLike,
        carrier_angular_frequency: ArrayLike,
        longitudinal_coordinate: ArrayLike,
        /,
        *,
        polarization: PulseEnvelopePolarization = "scalar",
    ):
        if not isinstance(plane_space, PlaneFieldSpace):
            raise TypeError("plane_space must be a PlaneFieldSpace.")
        if not isinstance(time_space, PulseTimeSpace):
            raise TypeError("time_space must be a PulseTimeSpace.")
        if polarization not in ("scalar", "tangential"):
            raise ValueError("polarization must be 'scalar' or 'tangential'.")
        expected = (
            plane_space.shape + time_space.shape
            if polarization == "scalar"
            else plane_space.shape + time_space.shape + (2,)
        )
        self.plane_space = plane_space
        self.time_space = time_space
        self.values = _complex_field_values("values", values, expected)
        self.carrier_angular_frequency = _angular_frequency(carrier_angular_frequency)
        self.longitudinal_coordinate = _longitudinal_coordinate(longitudinal_coordinate)
        self.polarization = polarization


class PulseEnvelopeBridgeStatus(IntFlag):
    """Fail-closed disposition of an exact carrier-bin shift."""

    SUCCESS = 0
    INCOMPATIBLE_FIELD = 1
    NONPOSITIVE_OR_NYQUIST_BAND = 2
    NONFINITE = 4


class PulseEnvelopeBridgePlan(StrictModule, NonTrainableState):
    """Static policy for an exact periodic envelope/carrier representation shift."""

    time_space: PulseTimeSpace
    carrier_angular_frequency: Array
    carrier_grid_tolerance: float = eqx.field(static=True)
    spectral_support_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_space: PulseTimeSpace,
        carrier_angular_frequency: ArrayLike,
        /,
        *,
        carrier_grid_tolerance: float = 1.0e-10,
        spectral_support_tolerance: float = 1.0e-12,
    ):
        if not isinstance(time_space, PulseTimeSpace):
            raise TypeError("time_space must be a PulseTimeSpace.")
        if time_space.topology != "periodic-cell":
            raise ValueError("Exact envelope bridging requires periodic-cell pulse time.")
        grid_tolerance = float(carrier_grid_tolerance)
        support_tolerance = float(spectral_support_tolerance)
        if not np.isfinite(grid_tolerance) or grid_tolerance < 0.0:
            raise ValueError("carrier_grid_tolerance must be finite and nonnegative.")
        if (
            not np.isfinite(support_tolerance)
            or support_tolerance < 0.0
            or support_tolerance > 1.0
        ):
            raise ValueError("spectral_support_tolerance must lie in [0, 1].")
        carrier = _angular_frequency(carrier_angular_frequency)
        self.time_space = time_space
        self.carrier_angular_frequency = carrier
        self.carrier_grid_tolerance = grid_tolerance
        self.spectral_support_tolerance = support_tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pulse-envelope-bridge-plan",
                "time_space": time_space.space_id,
                "carrier_angular_frequency": float(carrier).hex(),
                "carrier_grid_tolerance": grid_tolerance.hex(),
                "spectral_support_tolerance": support_tolerance.hex(),
            }
        )


class PreparedPulseEnvelopeBridge(StrictModule, NonTrainableState):
    """Resolved carrier mode and admissible fixed spectral bands."""

    plan: PulseEnvelopeBridgePlan
    carrier_angular_frequency: Array
    angular_frequency_offsets: Array
    envelope_supported_mask: Array
    analytic_supported_mask: Array
    carrier_mode: int = eqx.field(static=True)
    carrier_grid_alignment_error: float = eqx.field(static=True)
    nyquist_angular_frequency: float = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PulseEnvelopeBridgeEvidence(StrictModule):
    """Band, carrier-alignment, and finiteness evidence for one bridge."""

    total_spectral_energy: Array
    supported_spectral_energy: Array
    rejected_spectral_fraction: Array
    carrier_grid_alignment_error: Array
    nyquist_angular_frequency: Array
    finite: Array
    accepted: Array
    status: Array


class PulseEnvelopeBridgeResult(StrictModule):
    """Bridged field and fail-closed exactness evidence."""

    field: PulseEnvelopeField | AnalyticPulseField
    evidence: PulseEnvelopeBridgeEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.accepted

    @property
    def status(self) -> Array:
        return self.evidence.status


def prepare_pulse_envelope_bridge(
    plan: PulseEnvelopeBridgePlan,
    /,
) -> PreparedPulseEnvelopeBridge:
    """Resolve an exactly grid-aligned carrier shift without allocating payloads."""
    if not isinstance(plan, PulseEnvelopeBridgePlan):
        raise TypeError("plan must be a PulseEnvelopeBridgePlan.")
    count = plan.time_space.size
    spacing = plan.time_space.sample_spacing
    angular_bin = 2.0 * np.pi / (count * spacing)
    supplied = float(plan.carrier_angular_frequency)
    carrier_mode = int(np.rint(supplied / angular_bin))
    aligned = carrier_mode * angular_bin
    alignment_error = abs(supplied - aligned)
    coordinate_dtype = np.asarray(plan.time_space.coordinates).dtype
    numerical_tolerance = (
        64.0 * np.finfo(coordinate_dtype).eps * max(1.0, abs(supplied), abs(aligned))
    )
    if alignment_error > max(plan.carrier_grid_tolerance, numerical_tolerance):
        raise ValueError(
            "carrier_angular_frequency is not aligned to a temporal FFT bin."
        )
    maximum_positive_mode = (count - 1) // 2
    if carrier_mode <= 0 or carrier_mode > maximum_positive_mode:
        raise ValueError(
            "The carrier must occupy a strictly positive non-Nyquist FFT bin."
        )
    signed_modes = np.rint(np.fft.fftfreq(count) * count).astype(np.int64)
    shifted_modes = signed_modes + carrier_mode
    envelope_supported = (shifted_modes > 0) & (shifted_modes <= maximum_positive_mode)
    analytic_supported = (signed_modes > 0) & (signed_modes <= maximum_positive_mode)
    offsets = jnp.asarray(
        signed_modes * angular_bin, dtype=plan.carrier_angular_frequency.dtype
    )
    nyquist = float(np.pi / spacing)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-pulse-envelope-bridge",
            "plan": plan.plan_id,
            "carrier_mode": carrier_mode,
            "aligned_carrier": aligned.hex(),
            "envelope_supported_modes": np.flatnonzero(envelope_supported).tolist(),
        }
    )
    return PreparedPulseEnvelopeBridge(
        plan,
        jnp.asarray(aligned, dtype=plan.carrier_angular_frequency.dtype),
        offsets,
        jnp.asarray(envelope_supported),
        jnp.asarray(analytic_supported),
        carrier_mode,
        alignment_error,
        nyquist,
        prepared_id,
    )


def _bridge_evidence(
    prepared: PreparedPulseEnvelopeBridge,
    values: Array,
    supported_mask: Array,
    compatible: Array,
    /,
) -> PulseEnvelopeBridgeEvidence:
    spectrum = jnp.fft.ifft(values, axis=2, norm="ortho")
    mask = supported_mask.reshape((1, 1, supported_mask.size) + (1,) * (values.ndim - 3))
    energy = jnp.abs(spectrum) ** 2
    total = jnp.sum(energy)
    supported = jnp.sum(jnp.where(mask, energy, 0.0))
    rejected = jnp.sum(jnp.where(mask, 0.0, energy))
    rejected_fraction = jnp.where(total > 0.0, rejected / total, 0.0)
    finite = (
        jnp.all(jnp.isfinite(jnp.real(values)))
        & jnp.all(jnp.isfinite(jnp.imag(values)))
        & jnp.isfinite(total)
        & jnp.isfinite(rejected_fraction)
    )
    support_ok = rejected_fraction <= prepared.plan.spectral_support_tolerance
    status = jnp.where(
        compatible,
        int(PulseEnvelopeBridgeStatus.SUCCESS),
        int(PulseEnvelopeBridgeStatus.INCOMPATIBLE_FIELD),
    ).astype(jnp.int32)
    status = status | jnp.where(
        support_ok,
        int(PulseEnvelopeBridgeStatus.SUCCESS),
        int(PulseEnvelopeBridgeStatus.NONPOSITIVE_OR_NYQUIST_BAND),
    ).astype(jnp.int32)
    status = status | jnp.where(
        finite,
        int(PulseEnvelopeBridgeStatus.SUCCESS),
        int(PulseEnvelopeBridgeStatus.NONFINITE),
    ).astype(jnp.int32)
    accepted = status == int(PulseEnvelopeBridgeStatus.SUCCESS)
    return PulseEnvelopeBridgeEvidence(
        total,
        supported,
        rejected_fraction,
        jnp.asarray(
            prepared.carrier_grid_alignment_error,
            dtype=values.real.dtype,
        ),
        jnp.asarray(prepared.nyquist_angular_frequency, dtype=values.real.dtype),
        finite,
        accepted,
        status,
    )


def envelope_to_analytic_field(
    prepared: PreparedPulseEnvelopeBridge,
    field: PulseEnvelopeField,
    /,
) -> PulseEnvelopeBridgeResult:
    """Shift an envelope onto its absolute analytic band with exact grid phases."""
    if not isinstance(prepared, PreparedPulseEnvelopeBridge):
        raise TypeError("prepared must be a PreparedPulseEnvelopeBridge.")
    if not isinstance(field, PulseEnvelopeField):
        raise TypeError("field must be a PulseEnvelopeField.")
    compatible = (
        field.time_space.space_id == prepared.plan.time_space.space_id
        and field.carrier_angular_frequency == prepared.plan.carrier_angular_frequency
    )
    phase_shape = (1, 1, field.time_space.size) + (1,) * (field.values.ndim - 3)
    phase = jnp.exp(
        -1j * prepared.carrier_angular_frequency * field.time_space.coordinates
    ).reshape(phase_shape)
    values = field.values * phase
    analytic = AnalyticPulseField(
        field.plane_space,
        field.time_space,
        values,
        prepared.carrier_angular_frequency,
        field.longitudinal_coordinate,
        polarization=field.polarization,
    )
    evidence = _bridge_evidence(
        prepared,
        field.values,
        prepared.envelope_supported_mask,
        jnp.asarray(compatible),
    )
    return PulseEnvelopeBridgeResult(analytic, evidence, prepared.prepared_id)


def analytic_field_to_envelope(
    prepared: PreparedPulseEnvelopeBridge,
    field: AnalyticPulseField,
    /,
) -> PulseEnvelopeBridgeResult:
    """Demodulate an analytic field by the prepared exact carrier-bin shift."""
    if not isinstance(prepared, PreparedPulseEnvelopeBridge):
        raise TypeError("prepared must be a PreparedPulseEnvelopeBridge.")
    if not isinstance(field, AnalyticPulseField):
        raise TypeError("field must be an AnalyticPulseField.")
    compatible = (
        field.time_space.space_id == prepared.plan.time_space.space_id
        and field.angular_frequency == prepared.carrier_angular_frequency
    )
    phase_shape = (1, 1, field.time_space.size) + (1,) * (field.values.ndim - 3)
    phase = jnp.exp(
        1j * prepared.carrier_angular_frequency * field.time_space.coordinates
    ).reshape(phase_shape)
    values = field.values * phase
    envelope = PulseEnvelopeField(
        field.space,
        field.time_space,
        values,
        prepared.carrier_angular_frequency,
        field.longitudinal_coordinate,
        polarization=field.polarization,
    )
    evidence = _bridge_evidence(
        prepared,
        field.values,
        prepared.analytic_supported_mask,
        jnp.asarray(compatible),
    )
    return PulseEnvelopeBridgeResult(envelope, evidence, prepared.prepared_id)


class GaussianPulseEnvelopeStatus(IntFlag):
    """Support disposition of one sampled Gaussian envelope."""

    SUCCESS = 0
    BOUNDARY_SUPPORT_LIMIT = 1
    SPECTRAL_EDGE_LIMIT = 2
    NONFINITE = 4


class GaussianPulseEnvelopePlan(StrictModule, NonTrainableState):
    """Physical Gaussian electric-envelope parameters and sampling policy.

    Widths are RMS widths of the intensity, not of the field amplitude. ``chirp``
    is the temporal quadratic phase coefficient in rad/s².
    """

    plane_space: PlaneFieldSpace
    time_space: PulseTimeSpace
    carrier_angular_frequency: Array
    peak_amplitude: Array
    transverse_center: Array
    transverse_rms_width: Array
    temporal_center: Array
    temporal_rms_duration: Array
    carrier_phase: Array
    chirp: Array
    jones_vector: Array | None
    polarization: PulseEnvelopePolarization = eqx.field(static=True)
    boundary_tolerance: float = eqx.field(static=True)
    spectral_edge_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plane_space: PlaneFieldSpace,
        time_space: PulseTimeSpace,
        carrier_angular_frequency: ArrayLike,
        /,
        *,
        peak_amplitude: ArrayLike,
        transverse_center: ArrayLike,
        transverse_rms_width: ArrayLike,
        temporal_center: ArrayLike,
        temporal_rms_duration: ArrayLike,
        carrier_phase: ArrayLike = 0.0,
        chirp: ArrayLike = 0.0,
        polarization: PulseEnvelopePolarization = "scalar",
        jones_vector: ArrayLike | None = None,
        boundary_tolerance: float = 1.0e-6,
        spectral_edge_tolerance: float = 1.0e-8,
    ):
        if not isinstance(plane_space, PlaneFieldSpace):
            raise TypeError("plane_space must be a PlaneFieldSpace.")
        if not isinstance(time_space, PulseTimeSpace):
            raise TypeError("time_space must be a PulseTimeSpace.")
        if polarization not in ("scalar", "tangential"):
            raise ValueError("polarization must be 'scalar' or 'tangential'.")

        def real_array(name: str, value: ArrayLike, shape: tuple[int, ...]) -> Array:
            supplied = jnp.asarray(value)
            if supplied.shape != shape:
                raise ValueError(f"{name} must have shape {shape}; got {supplied.shape}.")
            if jnp.iscomplexobj(supplied) or not jnp.issubdtype(
                supplied.dtype, jnp.number
            ):
                raise TypeError(f"{name} must be real numeric data.")
            result = supplied.astype(jnp.result_type(supplied.dtype, jnp.float32))
            return eqx.error_if(
                result,
                jnp.any(~jnp.isfinite(result)),
                f"{name} must contain finite values.",
            )

        amplitude = real_array("peak_amplitude", peak_amplitude, ())
        amplitude = eqx.error_if(
            amplitude, amplitude <= 0.0, "peak_amplitude must be strictly positive."
        )
        center = real_array("transverse_center", transverse_center, (2,))
        widths = real_array("transverse_rms_width", transverse_rms_width, (2,))
        widths = eqx.error_if(
            widths,
            jnp.any(widths <= 0.0),
            "transverse_rms_width must be strictly positive.",
        )
        time_center = real_array("temporal_center", temporal_center, ())
        duration = real_array("temporal_rms_duration", temporal_rms_duration, ())
        duration = eqx.error_if(
            duration,
            duration <= 0.0,
            "temporal_rms_duration must be strictly positive.",
        )
        phase = real_array("carrier_phase", carrier_phase, ())
        chirp_ = real_array("chirp", chirp, ())
        boundary = float(boundary_tolerance)
        spectral = float(spectral_edge_tolerance)
        if not np.isfinite(boundary) or not 0.0 <= boundary <= 1.0:
            raise ValueError("boundary_tolerance must lie in [0, 1].")
        if not np.isfinite(spectral) or not 0.0 <= spectral <= 1.0:
            raise ValueError("spectral_edge_tolerance must lie in [0, 1].")
        if polarization == "scalar":
            if jones_vector is not None:
                raise ValueError("Scalar envelopes do not accept jones_vector.")
            jones = None
        else:
            if jones_vector is None:
                raise ValueError("Tangential envelopes require jones_vector.")
            candidate = jnp.asarray(jones_vector)
            if candidate.shape != (2,) or not jnp.issubdtype(candidate.dtype, jnp.number):
                raise ValueError("jones_vector must be a numeric two-vector.")
            jones = candidate.astype(jnp.result_type(candidate.dtype, jnp.complex64))
            norm = jnp.sqrt(jnp.sum(jnp.abs(jones) ** 2))
            jones = eqx.error_if(
                jones,
                jnp.any(~jnp.isfinite(jnp.real(jones)))
                | jnp.any(~jnp.isfinite(jnp.imag(jones)))
                | (~jnp.isfinite(norm))
                | (norm <= 0.0),
                "jones_vector must be finite and nonzero.",
            )
            jones = jones / norm
        carrier = _angular_frequency(carrier_angular_frequency)
        self.plane_space = plane_space
        self.time_space = time_space
        self.carrier_angular_frequency = carrier
        self.peak_amplitude = amplitude
        self.transverse_center = center
        self.transverse_rms_width = widths
        self.temporal_center = time_center
        self.temporal_rms_duration = duration
        self.carrier_phase = phase
        self.chirp = chirp_
        self.polarization = polarization
        self.jones_vector = jones
        self.boundary_tolerance = boundary
        self.spectral_edge_tolerance = spectral
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gaussian-pulse-envelope-plan",
                "plane_space": plane_space.space_id,
                "time_space": time_space.space_id,
                "carrier": float(carrier).hex(),
                "peak_amplitude": float(amplitude).hex(),
                "transverse_center": array_tree_fingerprint(center),
                "transverse_rms_width": array_tree_fingerprint(widths),
                "temporal_center": float(time_center).hex(),
                "temporal_rms_duration": float(duration).hex(),
                "carrier_phase": float(phase).hex(),
                "chirp": float(chirp_).hex(),
                "polarization": polarization,
                "jones_vector": None if jones is None else array_tree_fingerprint(jones),
                "boundary_tolerance": boundary.hex(),
                "spectral_edge_tolerance": spectral.hex(),
            }
        )


class PreparedGaussianPulseEnvelope(StrictModule, NonTrainableState):
    """Prepared fixed-shape Gaussian geometry and support bounds."""

    plan: GaussianPulseEnvelopePlan
    normalized_transverse_squared: Array
    temporal_displacement: Array
    boundary_omitted_fraction: Array
    spectral_edge_mask: Array
    workspace_complex_elements: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class GaussianPulseEnvelopeEvidence(StrictModule):
    """Analytic boundary support and sampled spectral-edge evidence."""

    captured_support_fraction: Array
    boundary_omitted_fraction: Array
    spectral_edge_fraction: Array
    finite: Array
    accepted: Array
    status: Array


class GaussianPulseEnvelopeResult(StrictModule):
    """Sampled Gaussian pulse envelope and its support evidence."""

    field: PulseEnvelopeField
    evidence: GaussianPulseEnvelopeEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.accepted


def _gaussian_interval_fraction(
    lower: float, upper: float, center: float, rms: float
) -> float:
    scale = sqrt(2.0) * rms
    return 0.5 * (erf((upper - center) / scale) - erf((lower - center) / scale))


def prepare_gaussian_pulse_envelope(
    plan: GaussianPulseEnvelopePlan,
    /,
) -> PreparedGaussianPulseEnvelope:
    """Resolve one fixed-shape Gaussian sampler and analytic boundary support."""
    if not isinstance(plan, GaussianPulseEnvelopePlan):
        raise TypeError("plan must be a GaussianPulseEnvelopePlan.")
    transverse = plan.plane_space.transverse_coordinates
    normalized_transverse = jnp.sum(
        ((transverse - plan.transverse_center) / plan.transverse_rms_width) ** 2,
        axis=-1,
    )
    temporal_displacement = plan.time_space.coordinates - plan.temporal_center
    bounds = [axis.bounds for axis in plan.plane_space.grid.axes]
    time_bounds = plan.time_space.temporal_grid.axes[0].bounds
    if any(item is None for item in bounds) or time_bounds is None:
        raise ValueError("Gaussian support evidence requires finite axis bounds.")
    fractions = []
    for axis_bounds, center, width in zip(
        bounds,
        np.asarray(plan.transverse_center),
        np.asarray(plan.transverse_rms_width),
        strict=True,
    ):
        resolved = np.asarray(axis_bounds, dtype=float)
        fractions.append(
            _gaussian_interval_fraction(
                float(resolved[0]), float(resolved[1]), float(center), float(width)
            )
        )
    resolved_time_bounds = np.asarray(time_bounds, dtype=float)
    fractions.append(
        _gaussian_interval_fraction(
            float(resolved_time_bounds[0]),
            float(resolved_time_bounds[1]),
            float(plan.temporal_center),
            float(plan.temporal_rms_duration),
        )
    )
    captured = float(np.clip(prod(fractions), 0.0, 1.0))
    mode_axes = tuple(
        np.abs(np.fft.fftfreq(size) * size)
        for size in plan.plane_space.shape + plan.time_space.shape
    )
    edge_axes = tuple(
        modes >= 0.9 * max(1.0, float(size // 2))
        for modes, size in zip(
            mode_axes,
            plan.plane_space.shape + plan.time_space.shape,
            strict=True,
        )
    )
    edge_mask = (
        edge_axes[0][:, None, None]
        | edge_axes[1][None, :, None]
        | edge_axes[2][None, None, :]
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-gaussian-pulse-envelope",
            "plan": plan.plan_id,
            "shape": list(plan.plane_space.shape + plan.time_space.shape),
            "captured_support_fraction": captured.hex(),
        }
    )
    return PreparedGaussianPulseEnvelope(
        plan,
        normalized_transverse,
        temporal_displacement,
        jnp.asarray(1.0 - captured, dtype=transverse.dtype),
        jnp.asarray(edge_mask),
        prod(plan.plane_space.shape + plan.time_space.shape),
        prepared_id,
    )


def sample_gaussian_pulse_envelope(
    prepared: PreparedGaussianPulseEnvelope,
    /,
) -> GaussianPulseEnvelopeResult:
    """Sample the prepared Gaussian without carrier-resolved oscillation."""
    if not isinstance(prepared, PreparedGaussianPulseEnvelope):
        raise TypeError("prepared must be a PreparedGaussianPulseEnvelope.")
    plan = prepared.plan
    normalized_time_squared = (
        prepared.temporal_displacement / plan.temporal_rms_duration
    ) ** 2
    magnitude = plan.peak_amplitude * jnp.exp(
        -0.25
        * (
            prepared.normalized_transverse_squared[..., None]
            + normalized_time_squared[None, None, :]
        )
    )
    phase = jnp.exp(
        1j * (plan.carrier_phase + 0.5 * plan.chirp * prepared.temporal_displacement**2)
    )
    scalar = magnitude * phase[None, None, :]
    values = (
        scalar if plan.polarization == "scalar" else scalar[..., None] * plan.jones_vector
    )
    spatial_spectrum = jnp.fft.fftn(values, axes=(0, 1), norm="ortho")
    spectrum = jnp.fft.ifft(spatial_spectrum, axis=2, norm="ortho")
    edge_mask = prepared.spectral_edge_mask.reshape(
        prepared.spectral_edge_mask.shape + (1,) * (values.ndim - 3)
    )
    energy = jnp.abs(spectrum) ** 2
    total = jnp.sum(energy)
    edge = jnp.sum(jnp.where(edge_mask, energy, 0.0))
    edge_fraction = jnp.where(total > 0.0, edge / total, 0.0)
    finite = (
        jnp.all(jnp.isfinite(jnp.real(values)))
        & jnp.all(jnp.isfinite(jnp.imag(values)))
        & jnp.isfinite(edge_fraction)
        & jnp.isfinite(prepared.boundary_omitted_fraction)
    )
    boundary_ok = prepared.boundary_omitted_fraction <= plan.boundary_tolerance
    edge_ok = edge_fraction <= plan.spectral_edge_tolerance
    status = jnp.where(
        boundary_ok,
        int(GaussianPulseEnvelopeStatus.SUCCESS),
        int(GaussianPulseEnvelopeStatus.BOUNDARY_SUPPORT_LIMIT),
    ).astype(jnp.int32)
    status = status | jnp.where(
        edge_ok,
        int(GaussianPulseEnvelopeStatus.SUCCESS),
        int(GaussianPulseEnvelopeStatus.SPECTRAL_EDGE_LIMIT),
    ).astype(jnp.int32)
    status = status | jnp.where(
        finite,
        int(GaussianPulseEnvelopeStatus.SUCCESS),
        int(GaussianPulseEnvelopeStatus.NONFINITE),
    ).astype(jnp.int32)
    accepted = status == int(GaussianPulseEnvelopeStatus.SUCCESS)
    field = PulseEnvelopeField(
        plan.plane_space,
        plan.time_space,
        values,
        plan.carrier_angular_frequency,
        0.0,
        polarization=plan.polarization,
    )
    evidence = GaussianPulseEnvelopeEvidence(
        1.0 - prepared.boundary_omitted_fraction,
        prepared.boundary_omitted_fraction,
        edge_fraction,
        finite,
        accepted,
        status,
    )
    return GaussianPulseEnvelopeResult(field, evidence, prepared.prepared_id)


__all__ = [
    "GaussianPulseEnvelopeEvidence",
    "GaussianPulseEnvelopePlan",
    "GaussianPulseEnvelopeResult",
    "GaussianPulseEnvelopeStatus",
    "PreparedGaussianPulseEnvelope",
    "PreparedPulseEnvelopeBridge",
    "PulseEnvelopeBridgeEvidence",
    "PulseEnvelopeBridgePlan",
    "PulseEnvelopeBridgeResult",
    "PulseEnvelopeBridgeStatus",
    "PulseEnvelopeField",
    "PulseEnvelopePolarization",
    "analytic_field_to_envelope",
    "envelope_to_analytic_field",
    "prepare_gaussian_pulse_envelope",
    "prepare_pulse_envelope_bridge",
    "sample_gaussian_pulse_envelope",
]
