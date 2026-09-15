#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Scalar axisymmetric carrier-resolved unidirectional propagation."""

from __future__ import annotations

from enum import IntEnum
from math import prod
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._checkpointed_scan import checkpointed_scan
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.spectral._cylindrical_hankel import (
    CylindricalHankelEvidence,
    PreparedCylindricalHankel,
)
from ..materials._refractive_index import (
    AbstractRefractiveIndexLaw,
    evaluate_refractive_index,
)
from ._fields import (
    _angular_frequency,
    _complex_field_values,
    _longitudinal_coordinate,
)
from ._nonlinear_response import (
    _VACUUM_PERMITTIVITY,
    AbstractCarrierResolvedResponse,
    CarrierResolvedResponseEvaluation,
    PreparedCarrierResolvedResponse,
)
from ._pulse_time import PulseTimeSpace
from ._unidirectional import _forward_square_root, _relative_error, _uniform_spacing


class CylindricalUnidirectionalPropagationStatus(IntEnum):
    """Fail-closed disposition of one fixed-shape axisymmetric propagation."""

    SUCCESS = 0
    NONFINITE_INPUT = 1
    INVALID_DISTANCE = 2
    INCOMPATIBLE_FREQUENCY = 3
    NUMERICAL_FAILURE = 4
    ANALYTIC_SIGNAL_DEFECT = 5
    SPECTRAL_EDGE_LIMIT = 6
    NONLINEAR_BANDWIDTH_LIMIT = 7
    REFINEMENT_LIMIT = 8
    UNIDIRECTIONAL_LIMIT = 9
    MATERIAL_RESPONSE_FAILURE = 10
    HANKEL_CERTIFICATION_FAILURE = 11
    RADIAL_BOUNDARY_LIMIT = 12
    RADIAL_HIGH_MODE_LIMIT = 13
    LONGITUDINAL_CUTOFF_LIMIT = 14
    RESOURCE_LIMIT = 15


class CylindricalAnalyticPulseField(StrictModule):
    """Scalar m=0 analytic electric field on a prepared finite-radius grid."""

    hankel: PreparedCylindricalHankel
    time_space: PulseTimeSpace
    values: Array
    angular_frequency: Array
    longitudinal_coordinate: Array

    def __init__(
        self,
        hankel: PreparedCylindricalHankel,
        time_space: PulseTimeSpace,
        values: ArrayLike,
        angular_frequency: ArrayLike,
        longitudinal_coordinate: ArrayLike,
        /,
    ):
        if not isinstance(hankel, PreparedCylindricalHankel):
            raise TypeError("hankel must be PreparedCylindricalHankel.")
        if hankel.plan.order != 0:
            raise ValueError("Cylindrical optical propagation supports only m=0.")
        if not isinstance(time_space, PulseTimeSpace):
            raise TypeError("time_space must be a PulseTimeSpace.")
        if time_space.topology != "periodic-cell":
            raise ValueError("Cylindrical analytic pulses require periodic-cell time.")
        self.hankel = hankel
        self.time_space = time_space
        self.values = _complex_field_values(
            "values", values, (hankel.plan.radial_count, time_space.size)
        )
        self.angular_frequency = _angular_frequency(angular_frequency)
        self.longitudinal_coordinate = _longitudinal_coordinate(longitudinal_coordinate)

    @property
    def radial_coordinates(self) -> Array:
        return self.hankel.radial_coordinates

    @property
    def radial_area_weights(self) -> Array:
        return self.hankel.area_weights

    @property
    def temporal_coordinates(self) -> Array:
        return self.time_space.coordinates


class CylindricalUnidirectionalPropagationEvidence(StrictModule):
    """Radial-measure, transform, bandwidth, and one-way validity evidence."""

    spectral_edge_fraction: Array
    analytic_signal_defect: Array
    hermitian_reconstruction_defect: Array
    nonlinear_rejected_fraction: Array
    nonlinear_bandwidth_margin: Array
    fixed_step_refinement_error: Array
    backward_wave_estimate: Array
    unidirectional_applicability_margin: Array
    dispersion_extrapolated_fraction: Array
    radial_boundary_fraction: Array
    radial_high_mode_fraction: Array
    longitudinal_cutoff_fraction: Array
    input_radial_measure: Array
    output_radial_measure: Array
    radial_measure_relative_change: Array
    hankel: CylindricalHankelEvidence
    workspace_bytes: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)


class CylindricalUnidirectionalPropagationResult(StrictModule):
    """Axisymmetric propagated pulse, response ledger, evidence, and status."""

    field: CylindricalAnalyticPulseField
    response_evaluation: CarrierResolvedResponseEvaluation
    evidence: CylindricalUnidirectionalPropagationEvidence
    finite: Array
    status: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class CylindricalUnidirectionalPropagationPlan(StrictModule, NonTrainableState):
    """Static m=0 finite-radius topology and fail-closed accuracy policy."""

    hankel: PreparedCylindricalHankel
    time_space: PulseTimeSpace
    angular_frequency: Array
    step_count: int = eqx.field(static=True)
    dealias_fraction: float = eqx.field(static=True)
    edge_guard_fraction: float = eqx.field(static=True)
    maximum_spectral_edge_fraction: float = eqx.field(static=True)
    maximum_analytic_signal_defect: float = eqx.field(static=True)
    maximum_hermitian_reconstruction_defect: float = eqx.field(static=True)
    maximum_nonlinear_rejected_fraction: float = eqx.field(static=True)
    maximum_refinement_error: float = eqx.field(static=True)
    maximum_backward_wave_estimate: float = eqx.field(static=True)
    maximum_radial_boundary_fraction: float = eqx.field(static=True)
    maximum_radial_high_mode_fraction: float = eqx.field(static=True)
    maximum_longitudinal_cutoff_fraction: float = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        hankel: PreparedCylindricalHankel,
        time_space: PulseTimeSpace,
        angular_frequency: ArrayLike,
        /,
        *,
        step_count: int,
        dealias_fraction: float = 0.5,
        edge_guard_fraction: float = 0.1,
        maximum_spectral_edge_fraction: float = 1.0e-6,
        maximum_analytic_signal_defect: float = 1.0e-11,
        maximum_hermitian_reconstruction_defect: float = 1.0e-11,
        maximum_nonlinear_rejected_fraction: float = 1.0e-6,
        maximum_refinement_error: float = 1.0e-5,
        maximum_backward_wave_estimate: float = 1.0e-3,
        maximum_radial_boundary_fraction: float = 1.0e-6,
        maximum_radial_high_mode_fraction: float = 1.0e-6,
        maximum_longitudinal_cutoff_fraction: float = 0.0,
        maximum_workspace_bytes: int = 1 << 30,
    ):
        if not isinstance(hankel, PreparedCylindricalHankel):
            raise TypeError("hankel must be PreparedCylindricalHankel.")
        if hankel.plan.order != 0:
            raise ValueError("Cylindrical optical propagation supports only m=0.")
        if not isinstance(time_space, PulseTimeSpace):
            raise TypeError("time_space must be a PulseTimeSpace.")
        if time_space.topology != "periodic-cell":
            raise ValueError("Cylindrical propagation requires periodic-cell pulse time.")
        steps = int(step_count)
        if steps < 2 or steps % 2 != 0:
            raise ValueError("step_count must be an even integer of at least two.")

        def fraction(name: str, value: float, *, upper: float = 1.0) -> float:
            resolved = float(value)
            if not np.isfinite(resolved) or resolved <= 0.0 or resolved > upper:
                raise ValueError(f"{name} must lie in (0, {upper}].")
            return resolved

        def limit(name: str, value: float) -> float:
            resolved = float(value)
            if not np.isfinite(resolved) or resolved < 0.0 or resolved > 1.0:
                raise ValueError(f"{name} must lie in [0, 1].")
            return resolved

        def nonnegative(name: str, value: float) -> float:
            resolved = float(value)
            if not np.isfinite(resolved) or resolved < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            return resolved

        if isinstance(maximum_workspace_bytes, bool) or not isinstance(
            maximum_workspace_bytes, Integral
        ):
            raise TypeError("maximum_workspace_bytes must be an integer.")
        workspace_limit = int(maximum_workspace_bytes)
        if workspace_limit <= 0:
            raise ValueError("maximum_workspace_bytes must be strictly positive.")
        frequency = _angular_frequency(angular_frequency)
        self.hankel = hankel
        self.time_space = time_space
        self.angular_frequency = frequency
        self.step_count = steps
        self.dealias_fraction = fraction("dealias_fraction", dealias_fraction)
        self.edge_guard_fraction = fraction(
            "edge_guard_fraction", edge_guard_fraction, upper=0.5
        )
        self.maximum_spectral_edge_fraction = limit(
            "maximum_spectral_edge_fraction", maximum_spectral_edge_fraction
        )
        self.maximum_analytic_signal_defect = limit(
            "maximum_analytic_signal_defect", maximum_analytic_signal_defect
        )
        self.maximum_hermitian_reconstruction_defect = limit(
            "maximum_hermitian_reconstruction_defect",
            maximum_hermitian_reconstruction_defect,
        )
        self.maximum_nonlinear_rejected_fraction = limit(
            "maximum_nonlinear_rejected_fraction",
            maximum_nonlinear_rejected_fraction,
        )
        self.maximum_refinement_error = nonnegative(
            "maximum_refinement_error", maximum_refinement_error
        )
        self.maximum_backward_wave_estimate = nonnegative(
            "maximum_backward_wave_estimate", maximum_backward_wave_estimate
        )
        self.maximum_radial_boundary_fraction = limit(
            "maximum_radial_boundary_fraction", maximum_radial_boundary_fraction
        )
        self.maximum_radial_high_mode_fraction = limit(
            "maximum_radial_high_mode_fraction", maximum_radial_high_mode_fraction
        )
        self.maximum_longitudinal_cutoff_fraction = limit(
            "maximum_longitudinal_cutoff_fraction",
            maximum_longitudinal_cutoff_fraction,
        )
        self.maximum_workspace_bytes = workspace_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cylindrical-unidirectional-propagation",
                "hankel": hankel.prepared_id,
                "time_space": time_space.space_id,
                "angular_frequency": float(np.asarray(frequency)),
                "step_count": steps,
                "dealias_fraction": self.dealias_fraction,
                "edge_guard_fraction": self.edge_guard_fraction,
                "maximum_spectral_edge_fraction": self.maximum_spectral_edge_fraction,
                "maximum_analytic_signal_defect": self.maximum_analytic_signal_defect,
                "maximum_hermitian_reconstruction_defect": self.maximum_hermitian_reconstruction_defect,
                "maximum_nonlinear_rejected_fraction": self.maximum_nonlinear_rejected_fraction,
                "maximum_refinement_error": self.maximum_refinement_error,
                "maximum_backward_wave_estimate": self.maximum_backward_wave_estimate,
                "maximum_radial_boundary_fraction": self.maximum_radial_boundary_fraction,
                "maximum_radial_high_mode_fraction": self.maximum_radial_high_mode_fraction,
                "maximum_longitudinal_cutoff_fraction": self.maximum_longitudinal_cutoff_fraction,
                "maximum_workspace_bytes": workspace_limit,
            }
        )

    def prepare(
        self, dispersion: AbstractRefractiveIndexLaw, /
    ) -> "PreparedCylindricalUnidirectionalPropagation":
        return prepare_cylindrical_unidirectional_propagation(self, dispersion)


class PreparedCylindricalUnidirectionalPropagation(StrictModule, NonTrainableState):
    """Resolved physical Hankel pair, dispersion branch, masks, and resources."""

    plan: CylindricalUnidirectionalPropagationPlan
    absolute_angular_frequencies: Array
    angular_frequency_offsets: Array
    refractive_indices: Array
    medium_wavenumbers: Array
    longitudinal_wavenumbers: Array
    linear_generator: Array
    nonlinear_source_factors: Array
    free_current_source_factors: Array
    positive_frequency_mask: Array
    dealias_mask: Array
    spectral_edge_mask: Array
    radial_high_mode_mask: Array
    longitudinal_cutoff_mask: Array
    dispersion_within_validity: Array
    dispersion_extrapolated: Array
    dispersion_status: Array
    minimum_longitudinal_wavenumber_fraction: Array
    workspace_complex_elements: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    resource_admissible: bool = eqx.field(static=True)
    hankel_admissible: bool = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def execute(
        self,
        field: CylindricalAnalyticPulseField,
        response: AbstractCarrierResolvedResponse | PreparedCarrierResolvedResponse,
        distance: ArrayLike,
        /,
    ) -> CylindricalUnidirectionalPropagationResult:
        return propagate_cylindrical_unidirectional(self, field, response, distance)


def prepare_cylindrical_unidirectional_propagation(
    plan: CylindricalUnidirectionalPropagationPlan,
    dispersion: AbstractRefractiveIndexLaw,
    /,
) -> PreparedCylindricalUnidirectionalPropagation:
    """Prepare one m=0 finite-radius one-way carrier-resolved propagation."""
    if not isinstance(plan, CylindricalUnidirectionalPropagationPlan):
        raise TypeError("plan must be CylindricalUnidirectionalPropagationPlan.")
    if not isinstance(dispersion, AbstractRefractiveIndexLaw):
        raise TypeError("dispersion must be an AbstractRefractiveIndexLaw.")
    temporal_spacing = _uniform_spacing(plan.time_space.coordinates, "temporal axis")
    radial_count = plan.hankel.plan.radial_count
    temporal_size = plan.time_space.size
    if temporal_size < 8:
        raise ValueError("Nonlinear propagation requires at least eight time samples.")
    complex_elements = 28 * prod((radial_count, temporal_size))
    workspace_bytes = 16 * complex_elements
    resource_admissible = workspace_bytes <= plan.maximum_workspace_bytes
    hankel_admissible = bool(np.asarray(plan.hankel.evidence.successful))
    dtype = plan.angular_frequency.dtype
    omega = jnp.asarray(
        2.0 * np.pi * np.fft.fftfreq(temporal_size, d=temporal_spacing),
        dtype=dtype,
    )
    positive = omega > 0.0
    first_positive = jnp.min(jnp.where(positive, omega, jnp.inf))
    material_omega = jnp.where(positive, omega, first_positive)
    evaluation = evaluate_refractive_index(dispersion, material_omega)
    active_accepted = jnp.where(positive, evaluation.accepted, True)
    refractive_indices = jnp.where(positive, evaluation.refractive_index, 0.0)
    refractive_indices = eqx.error_if(
        refractive_indices,
        jnp.any(~active_accepted),
        "The dispersion law rejected at least one active positive frequency.",
    )
    medium_wavenumbers = jnp.where(
        positive,
        refractive_indices * omega / dispersion.reference_wave_speed,
        0.0,
    )
    medium_wavenumbers = eqx.error_if(
        medium_wavenumbers,
        jnp.any(
            positive
            & (
                ~jnp.isfinite(jnp.real(medium_wavenumbers))
                | ~jnp.isfinite(jnp.imag(medium_wavenumbers))
                | (jnp.real(medium_wavenumbers) <= 0.0)
                | (jnp.imag(medium_wavenumbers) < 0.0)
            )
        ),
        "Active medium wavenumbers must have Re(k)>0 and Im(k)>=0.",
    )
    transverse = plan.hankel.transverse_angular_wavenumbers.astype(dtype)
    active = positive[None, :]
    squared_difference = medium_wavenumbers[None, :] ** 2 - transverse[:, None] ** 2
    longitudinal = _forward_square_root(squared_difference)
    longitudinal = jnp.where(active, longitudinal, 0.0)
    singular = active & (jnp.abs(longitudinal) <= jnp.finfo(dtype).eps)
    safe_longitudinal = jnp.where(active & ~singular, longitudinal, 1.0)
    linear_generator = jnp.where(active & ~singular, 1j * longitudinal, 0.0)
    omega2 = omega[None, :]
    polarization_source = jnp.where(
        active & ~singular,
        1j
        * (omega2 / dispersion.reference_wave_speed) ** 2
        / (2.0 * jnp.asarray(_VACUUM_PERMITTIVITY, dtype=dtype) * safe_longitudinal),
        0.0,
    )
    current_source = jnp.where(
        active & ~singular,
        -omega2
        / (
            2.0
            * jnp.asarray(_VACUUM_PERMITTIVITY, dtype=dtype)
            * dispersion.reference_wave_speed**2
            * safe_longitudinal
        ),
        0.0,
    )
    mode = jnp.arange(1, radial_count + 1, dtype=dtype)
    maximum_mode = float(radial_count)
    nyquist_omega = jnp.asarray(np.pi / temporal_spacing, dtype=dtype)
    radial_dealias = mode[:, None] <= plan.dealias_fraction * maximum_mode
    temporal_dealias = active & (omega2 <= plan.dealias_fraction * nyquist_omega)
    dealias_mask = radial_dealias & temporal_dealias
    radial_high_mode = mode[:, None] >= (1.0 - plan.edge_guard_fraction) * maximum_mode
    temporal_edge = active & (
        (omega2 <= plan.edge_guard_fraction * nyquist_omega)
        | (omega2 >= (1.0 - plan.edge_guard_fraction) * nyquist_omega)
    )
    spectral_edge_mask = active & (radial_high_mode | temporal_edge)
    cutoff = active & ((jnp.real(squared_difference) <= 0.0) | singular)
    active_longitudinal = jnp.where(active & ~singular, jnp.abs(longitudinal), jnp.inf)
    active_medium = jnp.where(active, jnp.abs(medium_wavenumbers)[None, :], 1.0)
    minimum_fraction = jnp.min(active_longitudinal / active_medium)
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-cylindrical-unidirectional-propagation",
            "plan": plan.plan_id,
            "law": dispersion.law_id,
            "provenance": dispersion.provenance.provenance_id,
            "shape": [radial_count, temporal_size],
            "workspace_complex_elements": complex_elements,
            "workspace_bytes": workspace_bytes,
            "resource_admissible": resource_admissible,
            "hankel_admissible": hankel_admissible,
        }
    )
    return PreparedCylindricalUnidirectionalPropagation(
        plan,
        omega,
        omega - plan.angular_frequency,
        refractive_indices,
        medium_wavenumbers,
        longitudinal,
        linear_generator,
        polarization_source,
        current_source,
        positive,
        dealias_mask,
        spectral_edge_mask,
        radial_high_mode & active,
        cutoff,
        jnp.where(positive, evaluation.within_validity, True),
        jnp.where(positive, evaluation.extrapolated, False),
        jnp.where(positive, evaluation.status, 0).astype(jnp.int32),
        minimum_fraction,
        workspace_complex_elements=complex_elements,
        workspace_bytes=workspace_bytes,
        resource_admissible=resource_admissible,
        hankel_admissible=hankel_admissible,
        law_id=dispersion.law_id,
        provenance_id=dispersion.provenance.provenance_id,
        prepared_id=prepared_id,
    )


def _to_spectrum(
    prepared: PreparedCylindricalUnidirectionalPropagation, values: Array, /
) -> Array:
    radial = prepared.plan.hankel.forward(values, axis=0)
    return jnp.fft.ifft(radial, axis=1, norm="ortho")


def _from_spectrum(
    prepared: PreparedCylindricalUnidirectionalPropagation, values: Array, /
) -> Array:
    temporal = jnp.fft.fft(values, axis=1, norm="ortho")
    return prepared.plan.hankel.inverse(temporal, axis=0)


def _weighted_spectral_fraction(
    prepared: PreparedCylindricalUnidirectionalPropagation,
    values: Array,
    mask: Array,
    /,
) -> Array:
    weights = prepared.plan.hankel.spectral_weights[:, None]
    energy = weights * jnp.abs(values) ** 2
    total = jnp.sum(energy)
    selected = jnp.sum(jnp.where(mask, energy, 0.0))
    safe_total = jnp.where(total > 0.0, total, 1.0)
    return jnp.where(total > 0.0, selected / safe_total, 0.0)


def _radial_measure(
    prepared: PreparedCylindricalUnidirectionalPropagation, values: Array, /
) -> Array:
    weights = (
        prepared.plan.hankel.area_weights[:, None]
        * prepared.plan.time_space.weights[None, :]
    )
    return jnp.sum(weights * jnp.abs(values) ** 2)


def _radial_boundary_fraction(
    prepared: PreparedCylindricalUnidirectionalPropagation, values: Array, /
) -> Array:
    radius = prepared.plan.hankel.radial_coordinates
    threshold = (1.0 - prepared.plan.edge_guard_fraction) * radius[-1]
    mask = radius[:, None] >= threshold
    weights = (
        prepared.plan.hankel.area_weights[:, None]
        * prepared.plan.time_space.weights[None, :]
    )
    energy = weights * jnp.abs(values) ** 2
    total = jnp.sum(energy)
    selected = jnp.sum(jnp.where(mask, energy, 0.0))
    safe_total = jnp.where(total > 0.0, total, 1.0)
    return jnp.where(total > 0.0, selected / safe_total, 0.0)


def _prepare_response(
    prepared: PreparedCylindricalUnidirectionalPropagation,
    response: AbstractCarrierResolvedResponse | PreparedCarrierResolvedResponse,
    /,
) -> PreparedCarrierResolvedResponse:
    field_shape = (
        prepared.plan.hankel.plan.radial_count,
        prepared.plan.time_space.size,
    )
    if isinstance(response, AbstractCarrierResolvedResponse):
        resolved = response.prepare(
            prepared.plan.time_space,
            prepared.positive_frequency_mask,
            field_shape,
            temporal_axis=1,
        )
    elif isinstance(response, PreparedCarrierResolvedResponse):
        resolved = response
    else:
        raise TypeError(
            "response must implement AbstractCarrierResolvedResponse or "
            "PreparedCarrierResolvedResponse."
        )
    if resolved.field_kind != "scalar":
        raise TypeError("Cylindrical m=0 propagation requires a scalar response.")
    if resolved.time_space.space_id != prepared.plan.time_space.space_id:
        raise ValueError("Prepared response and propagation use different pulse time.")
    if resolved.field_shape != field_shape or resolved.temporal_axis != 1:
        raise ValueError("Prepared response field geometry is incompatible.")
    return resolved


def _nonlinear_rate(
    prepared: PreparedCylindricalUnidirectionalPropagation,
    spectral_field: Array,
    response: PreparedCarrierResolvedResponse,
    /,
) -> tuple[Array, Array, Array, Array]:
    input_rejected = _weighted_spectral_fraction(
        prepared, spectral_field, ~prepared.dealias_mask
    )
    filtered_field = jnp.where(prepared.dealias_mask, spectral_field, 0.0)
    physical_field = _from_spectrum(prepared, filtered_field)
    evaluation = response.evaluate(physical_field)
    spectral_polarization = _to_spectrum(
        prepared, evaluation.analytic_nonlinear_polarization
    )
    spectral_current = _to_spectrum(prepared, evaluation.analytic_free_current)
    response_active = jnp.any(
        evaluation.analytic_nonlinear_polarization != 0.0
    ) | jnp.any(evaluation.analytic_free_current != 0.0)
    input_rejected = jnp.where(response_active, input_rejected, 0.0)
    rejected = jnp.maximum(
        input_rejected,
        jnp.maximum(
            _weighted_spectral_fraction(
                prepared, spectral_polarization, ~prepared.dealias_mask
            ),
            _weighted_spectral_fraction(
                prepared, spectral_current, ~prepared.dealias_mask
            ),
        ),
    )
    spectral_polarization = jnp.where(prepared.dealias_mask, spectral_polarization, 0.0)
    spectral_current = jnp.where(prepared.dealias_mask, spectral_current, 0.0)
    rate = (
        prepared.nonlinear_source_factors * spectral_polarization
        + prepared.free_current_source_factors * spectral_current
    )
    field_scale = jnp.max(jnp.abs(physical_field))
    polarization_scale = jnp.max(jnp.abs(evaluation.analytic_nonlinear_polarization))
    current_scale = jnp.max(jnp.abs(evaluation.analytic_free_current))
    minimum_frequency = jnp.min(
        jnp.where(
            prepared.positive_frequency_mask,
            prepared.absolute_angular_frequencies,
            jnp.inf,
        )
    )
    equivalent_polarization = polarization_scale + current_scale / minimum_frequency
    safe_field_scale = jnp.where(field_scale > 0.0, field_scale, 1.0)
    coupling = jnp.where(
        field_scale > 0.0,
        equivalent_polarization
        / (
            jnp.asarray(_VACUUM_PERMITTIVITY, dtype=physical_field.real.dtype)
            * safe_field_scale
        ),
        0.0,
    )
    return rate, rejected, 0.5 * coupling, evaluation.successful


def _interaction_picture_solve(
    prepared: PreparedCylindricalUnidirectionalPropagation,
    initial_spectrum: Array,
    response: PreparedCarrierResolvedResponse,
    distance: Array,
    step_count: int,
    /,
) -> tuple[Array, Array, Array, Array]:
    step = distance / float(step_count)
    generator = prepared.linear_generator
    half_linear = jnp.exp(0.5 * step * generator)
    full_linear = half_linear * half_linear
    inverse_half = jnp.exp(-0.5 * step * generator)
    inverse_full = inverse_half * inverse_half
    zero = jnp.asarray(0.0, dtype=initial_spectrum.real.dtype)
    valid = jnp.asarray(True)

    def scan_step(carry, _):
        state, maximum_rejected, maximum_backward, response_successful = carry
        k1, rejected1, backward1, valid1 = _nonlinear_rate(prepared, state, response)
        state2 = half_linear * (state + 0.5 * step * k1)
        raw2, rejected2, backward2, valid2 = _nonlinear_rate(prepared, state2, response)
        k2 = inverse_half * raw2
        state3 = half_linear * (state + 0.5 * step * k2)
        raw3, rejected3, backward3, valid3 = _nonlinear_rate(prepared, state3, response)
        k3 = inverse_half * raw3
        state4 = full_linear * (state + step * k3)
        raw4, rejected4, backward4, valid4 = _nonlinear_rate(prepared, state4, response)
        k4 = inverse_full * raw4
        next_state = full_linear * (state + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)
        rejected = jnp.maximum(
            jnp.maximum(rejected1, rejected2), jnp.maximum(rejected3, rejected4)
        )
        backward = jnp.maximum(
            jnp.maximum(backward1, backward2), jnp.maximum(backward3, backward4)
        )
        return (
            next_state,
            jnp.maximum(maximum_rejected, rejected),
            jnp.maximum(maximum_backward, backward),
            response_successful & valid1 & valid2 & valid3 & valid4,
        ), None

    final, _ = checkpointed_scan(
        scan_step,
        (initial_spectrum, zero, zero, valid),
        jnp.arange(step_count, dtype=jnp.int32),
        length=step_count,
        mode="step",
    )
    return final


def _hermitian_reconstruction_defect(values: Array, /) -> Array:
    physical = jnp.real(values)
    spectrum = jnp.fft.ifft(physical, axis=1, norm="ortho")
    indices = (-jnp.arange(spectrum.shape[1], dtype=jnp.int32)) % spectrum.shape[1]
    partner = jnp.conj(jnp.take(spectrum, indices, axis=1))
    return _relative_error(spectrum, partner)


def propagate_cylindrical_unidirectional(
    prepared: PreparedCylindricalUnidirectionalPropagation,
    field: CylindricalAnalyticPulseField,
    response: AbstractCarrierResolvedResponse | PreparedCarrierResolvedResponse,
    distance: ArrayLike,
    /,
) -> CylindricalUnidirectionalPropagationResult:
    """Execute paired fixed-step RK4 with the prepared physical Hankel pair."""
    if not isinstance(prepared, PreparedCylindricalUnidirectionalPropagation):
        raise TypeError("prepared must be PreparedCylindricalUnidirectionalPropagation.")
    if not isinstance(field, CylindricalAnalyticPulseField):
        raise TypeError("field must be CylindricalAnalyticPulseField.")
    plan = prepared.plan
    if field.hankel.prepared_id != plan.hankel.prepared_id:
        raise ValueError("field and plan must use the same prepared Hankel pair.")
    if field.time_space.space_id != plan.time_space.space_id:
        raise ValueError("field and plan must use the same PulseTimeSpace.")
    resolved_response = _prepare_response(prepared, response)
    distance_array = jnp.asarray(distance, dtype=field.longitudinal_coordinate.dtype)
    if distance_array.shape != ():
        raise ValueError("distance must be scalar.")
    distance_valid = jnp.isfinite(distance_array) & (distance_array >= 0.0)
    safe_distance = jnp.where(distance_valid, distance_array, 0.0)
    frequency_compatible = field.angular_frequency == plan.angular_frequency
    finite_input = (
        jnp.all(jnp.isfinite(jnp.real(field.values)))
        & jnp.all(jnp.isfinite(jnp.imag(field.values)))
        & jnp.isfinite(field.angular_frequency)
        & jnp.isfinite(field.longitudinal_coordinate)
    )
    initial_spectrum = _to_spectrum(prepared, field.values)
    execution_admissible = prepared.resource_admissible and prepared.hankel_admissible
    if execution_admissible:
        fine, nonlinear_rejected, backward_estimate, fine_response_successful = (
            _interaction_picture_solve(
                prepared,
                initial_spectrum,
                resolved_response,
                safe_distance,
                plan.step_count,
            )
        )
        coarse, _, _, coarse_response_successful = _interaction_picture_solve(
            prepared,
            initial_spectrum,
            resolved_response,
            safe_distance,
            plan.step_count // 2,
        )
        transformed_output = _from_spectrum(prepared, fine)
        output_values = jnp.where(safe_distance == 0.0, field.values, transformed_output)
        output_distance = safe_distance
    else:
        fine = initial_spectrum
        coarse = initial_spectrum
        nonlinear_rejected = jnp.asarray(0.0, dtype=field.values.real.dtype)
        backward_estimate = jnp.asarray(0.0, dtype=field.values.real.dtype)
        fine_response_successful = jnp.asarray(True)
        coarse_response_successful = jnp.asarray(True)
        output_values = field.values
        output_distance = jnp.asarray(0.0, dtype=safe_distance.dtype)
    output = CylindricalAnalyticPulseField(
        field.hankel,
        field.time_space,
        output_values,
        field.angular_frequency,
        field.longitudinal_coordinate + output_distance,
    )
    response_evaluation = resolved_response.evaluate(output_values)
    edge_fraction = jnp.maximum(
        _weighted_spectral_fraction(
            prepared, initial_spectrum, prepared.spectral_edge_mask
        ),
        _weighted_spectral_fraction(prepared, fine, prepared.spectral_edge_mask),
    )
    nonpositive = ~prepared.positive_frequency_mask[None, :]
    analytic_defect = jnp.maximum(
        _weighted_spectral_fraction(prepared, initial_spectrum, nonpositive),
        _weighted_spectral_fraction(prepared, fine, nonpositive),
    )
    high_mode_fraction = jnp.maximum(
        _weighted_spectral_fraction(
            prepared, initial_spectrum, prepared.radial_high_mode_mask
        ),
        _weighted_spectral_fraction(prepared, fine, prepared.radial_high_mode_mask),
    )
    cutoff_fraction = jnp.maximum(
        _weighted_spectral_fraction(
            prepared, initial_spectrum, prepared.longitudinal_cutoff_mask
        ),
        _weighted_spectral_fraction(prepared, fine, prepared.longitudinal_cutoff_mask),
    )
    boundary_fraction = jnp.maximum(
        _radial_boundary_fraction(prepared, field.values),
        _radial_boundary_fraction(prepared, output_values),
    )
    input_measure = _radial_measure(prepared, field.values)
    output_measure = _radial_measure(prepared, output_values)
    safe_measure = jnp.where(input_measure > 0.0, input_measure, 1.0)
    measure_change = jnp.where(
        input_measure > 0.0,
        jnp.abs(output_measure - input_measure) / safe_measure,
        jnp.abs(output_measure - input_measure),
    )
    refinement_error = _relative_error(fine, coarse)
    hermitian_defect = _hermitian_reconstruction_defect(output_values)
    extrapolated_fraction = jnp.sum(
        prepared.dispersion_extrapolated.astype(output_values.real.dtype)
    ) / jnp.sum(prepared.positive_frequency_mask)
    evidence = CylindricalUnidirectionalPropagationEvidence(
        edge_fraction,
        analytic_defect,
        hermitian_defect,
        nonlinear_rejected,
        plan.maximum_nonlinear_rejected_fraction - nonlinear_rejected,
        refinement_error,
        backward_estimate,
        plan.maximum_backward_wave_estimate - backward_estimate,
        extrapolated_fraction,
        boundary_fraction,
        high_mode_fraction,
        cutoff_fraction,
        input_measure,
        output_measure,
        measure_change,
        plan.hankel.evidence,
        workspace_bytes=prepared.workspace_bytes,
        maximum_workspace_bytes=plan.maximum_workspace_bytes,
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.real(output_values)))
        & jnp.all(jnp.isfinite(jnp.imag(output_values)))
        & response_evaluation.finite
        & jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        edge_fraction,
                        analytic_defect,
                        hermitian_defect,
                        nonlinear_rejected,
                        refinement_error,
                        backward_estimate,
                        extrapolated_fraction,
                        boundary_fraction,
                        high_mode_fraction,
                        cutoff_fraction,
                        input_measure,
                        output_measure,
                        measure_change,
                    )
                )
            )
        )
    )
    status = jnp.asarray(
        CylindricalUnidirectionalPropagationStatus.SUCCESS, dtype=jnp.int32
    )

    def update(
        condition: Array | bool, code: CylindricalUnidirectionalPropagationStatus
    ) -> None:
        nonlocal status
        status = jnp.where(
            (status == int(CylindricalUnidirectionalPropagationStatus.SUCCESS))
            & condition,
            int(code),
            status,
        ).astype(jnp.int32)

    update(~finite_input, CylindricalUnidirectionalPropagationStatus.NONFINITE_INPUT)
    update(~distance_valid, CylindricalUnidirectionalPropagationStatus.INVALID_DISTANCE)
    update(
        ~frequency_compatible,
        CylindricalUnidirectionalPropagationStatus.INCOMPATIBLE_FREQUENCY,
    )
    update(
        not prepared.resource_admissible,
        CylindricalUnidirectionalPropagationStatus.RESOURCE_LIMIT,
    )
    update(
        not prepared.hankel_admissible,
        CylindricalUnidirectionalPropagationStatus.HANKEL_CERTIFICATION_FAILURE,
    )
    update(~finite, CylindricalUnidirectionalPropagationStatus.NUMERICAL_FAILURE)
    update(
        ~fine_response_successful
        | ~coarse_response_successful
        | ~response_evaluation.successful,
        CylindricalUnidirectionalPropagationStatus.MATERIAL_RESPONSE_FAILURE,
    )
    update(
        (analytic_defect > plan.maximum_analytic_signal_defect)
        | (hermitian_defect > plan.maximum_hermitian_reconstruction_defect),
        CylindricalUnidirectionalPropagationStatus.ANALYTIC_SIGNAL_DEFECT,
    )
    update(
        edge_fraction > plan.maximum_spectral_edge_fraction,
        CylindricalUnidirectionalPropagationStatus.SPECTRAL_EDGE_LIMIT,
    )
    update(
        boundary_fraction > plan.maximum_radial_boundary_fraction,
        CylindricalUnidirectionalPropagationStatus.RADIAL_BOUNDARY_LIMIT,
    )
    update(
        high_mode_fraction > plan.maximum_radial_high_mode_fraction,
        CylindricalUnidirectionalPropagationStatus.RADIAL_HIGH_MODE_LIMIT,
    )
    update(
        cutoff_fraction > plan.maximum_longitudinal_cutoff_fraction,
        CylindricalUnidirectionalPropagationStatus.LONGITUDINAL_CUTOFF_LIMIT,
    )
    update(
        nonlinear_rejected > plan.maximum_nonlinear_rejected_fraction,
        CylindricalUnidirectionalPropagationStatus.NONLINEAR_BANDWIDTH_LIMIT,
    )
    update(
        refinement_error > plan.maximum_refinement_error,
        CylindricalUnidirectionalPropagationStatus.REFINEMENT_LIMIT,
    )
    update(
        backward_estimate > plan.maximum_backward_wave_estimate,
        CylindricalUnidirectionalPropagationStatus.UNIDIRECTIONAL_LIMIT,
    )
    successful = status == int(CylindricalUnidirectionalPropagationStatus.SUCCESS)
    return CylindricalUnidirectionalPropagationResult(
        output,
        response_evaluation,
        evidence,
        finite,
        status,
        successful,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "CylindricalAnalyticPulseField",
    "CylindricalUnidirectionalPropagationEvidence",
    "CylindricalUnidirectionalPropagationPlan",
    "CylindricalUnidirectionalPropagationResult",
    "CylindricalUnidirectionalPropagationStatus",
    "PreparedCylindricalUnidirectionalPropagation",
    "prepare_cylindrical_unidirectional_propagation",
    "propagate_cylindrical_unidirectional",
]
