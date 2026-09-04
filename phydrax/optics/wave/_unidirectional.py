#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._numerics._checkpointed_scan import checkpointed_scan
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import PreparedTensorGrid
from ..materials._refractive_index import (
    AbstractRefractiveIndexLaw,
    evaluate_refractive_index,
)
from ._fields import _angular_frequency, PlaneFieldSpace
from ._nonlinear_response import (
    _VACUUM_PERMITTIVITY,
    AnalyticPulseField,
    AnalyticPulsePolarization,
    instantaneous_nonlinear_polarization,
    InstantaneousScalarSusceptibility,
    OrientedTensorSusceptibility,
)


class UnidirectionalPropagationStatus(IntEnum):
    """JAX-compatible disposition of one fixed-shape propagation."""

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


class UnidirectionalApproximationEvidence(StrictModule):
    """Observable validity evidence for the returned one-way spectral field."""

    spectral_edge_fraction: Array
    analytic_signal_defect: Array
    hermitian_reconstruction_defect: Array
    nonlinear_rejected_fraction: Array
    nonlinear_bandwidth_margin: Array
    fixed_step_refinement_error: Array
    backward_wave_estimate: Array
    unidirectional_applicability_margin: Array
    dispersion_extrapolated_fraction: Array


class UnidirectionalPropagationResult(StrictModule):
    """Propagated pulse, approximation evidence, and explicit status."""

    field: AnalyticPulseField
    evidence: UnidirectionalApproximationEvidence
    finite: Array
    status: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class UnidirectionalPropagationPlan(StrictModule, NonTrainableState):
    """Static Cartesian topology and accuracy policy for one pulse propagation.

    The temporal representation is carrier-resolved rather than an envelope. For
    samples ``t_n``, ``ifft(values, axis=time, norm='ortho')`` is the spectrum of
    the ``exp(-1j * omega * t)`` analytic field. Only strictly positive absolute
    FFT frequencies are physical. DC, the even-grid Nyquist bin, and negative
    bins are inactive and must contain negligible energy.
    """

    space: PlaneFieldSpace
    temporal_grid: PreparedTensorGrid
    angular_frequency: Array
    polarization: AnalyticPulsePolarization = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    dealias_fraction: float = eqx.field(static=True)
    edge_guard_fraction: float = eqx.field(static=True)
    maximum_spectral_edge_fraction: float = eqx.field(static=True)
    maximum_analytic_signal_defect: float = eqx.field(static=True)
    maximum_hermitian_reconstruction_defect: float = eqx.field(static=True)
    maximum_nonlinear_rejected_fraction: float = eqx.field(static=True)
    maximum_refinement_error: float = eqx.field(static=True)
    maximum_backward_wave_estimate: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        temporal_grid: PreparedTensorGrid,
        angular_frequency: ArrayLike,
        /,
        *,
        polarization: AnalyticPulsePolarization = "scalar",
        step_count: int,
        dealias_fraction: float = 0.5,
        edge_guard_fraction: float = 0.1,
        maximum_spectral_edge_fraction: float = 1.0e-6,
        maximum_analytic_signal_defect: float = 1.0e-11,
        maximum_hermitian_reconstruction_defect: float = 1.0e-11,
        maximum_nonlinear_rejected_fraction: float = 1.0e-6,
        maximum_refinement_error: float = 1.0e-5,
        maximum_backward_wave_estimate: float = 1.0e-3,
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        if not isinstance(temporal_grid, PreparedTensorGrid):
            raise TypeError("temporal_grid must be a PreparedTensorGrid.")
        if len(temporal_grid.shape) != 1:
            raise ValueError("temporal_grid must be exactly one-dimensional.")
        temporal_axis = temporal_grid.axes[0]
        if (
            temporal_axis.basis != "fourier"
            or not temporal_axis.periodic
            or temporal_axis.primary_entity != "point"
        ):
            raise ValueError(
                "temporal_grid must be a periodic point-primary Fourier grid."
            )
        if polarization not in ("scalar", "tangential"):
            raise ValueError("polarization must be 'scalar' or 'tangential'.")
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

        dealias = fraction("dealias_fraction", dealias_fraction)
        edge_guard = fraction("edge_guard_fraction", edge_guard_fraction, upper=0.5)
        edge_limit = limit(
            "maximum_spectral_edge_fraction", maximum_spectral_edge_fraction
        )
        analytic_limit = limit(
            "maximum_analytic_signal_defect", maximum_analytic_signal_defect
        )
        hermitian_limit = limit(
            "maximum_hermitian_reconstruction_defect",
            maximum_hermitian_reconstruction_defect,
        )
        nonlinear_limit = limit(
            "maximum_nonlinear_rejected_fraction",
            maximum_nonlinear_rejected_fraction,
        )
        refinement_limit = nonnegative(
            "maximum_refinement_error", maximum_refinement_error
        )
        backward_limit = nonnegative(
            "maximum_backward_wave_estimate", maximum_backward_wave_estimate
        )
        frequency = _angular_frequency(angular_frequency)
        self.space = space
        self.temporal_grid = temporal_grid
        self.angular_frequency = frequency
        self.polarization = polarization
        self.step_count = steps
        self.dealias_fraction = dealias
        self.edge_guard_fraction = edge_guard
        self.maximum_spectral_edge_fraction = edge_limit
        self.maximum_analytic_signal_defect = analytic_limit
        self.maximum_hermitian_reconstruction_defect = hermitian_limit
        self.maximum_nonlinear_rejected_fraction = nonlinear_limit
        self.maximum_refinement_error = refinement_limit
        self.maximum_backward_wave_estimate = backward_limit
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cartesian-unidirectional-propagation",
                "space": space.space_id,
                "temporal_grid": temporal_grid.prepared_id,
                "angular_frequency": float(frequency),
                "polarization": polarization,
                "step_count": steps,
                "dealias_fraction": dealias,
                "edge_guard_fraction": edge_guard,
                "maximum_spectral_edge_fraction": edge_limit,
                "maximum_analytic_signal_defect": analytic_limit,
                "maximum_hermitian_reconstruction_defect": hermitian_limit,
                "maximum_nonlinear_rejected_fraction": nonlinear_limit,
                "maximum_refinement_error": refinement_limit,
                "maximum_backward_wave_estimate": backward_limit,
            }
        )

    def prepare(
        self, dispersion: AbstractRefractiveIndexLaw, /
    ) -> "PreparedUnidirectionalPropagation":
        return prepare_unidirectional_propagation(self, dispersion)


class PreparedUnidirectionalPropagation(StrictModule, NonTrainableState):
    """Resolved dispersion, spectral geometry, masks, and resource evidence."""

    plan: UnidirectionalPropagationPlan
    absolute_angular_frequencies: Array
    angular_frequency_offsets: Array
    transverse_angular_frequencies: Array
    refractive_indices: Array
    medium_wavenumbers: Array
    longitudinal_wavenumbers: Array
    linear_generator: Array
    nonlinear_source_factors: Array
    positive_frequency_mask: Array
    dealias_mask: Array
    spectral_edge_mask: Array
    dispersion_within_validity: Array
    dispersion_extrapolated: Array
    dispersion_status: Array
    minimum_longitudinal_wavenumber_fraction: Array
    workspace_complex_elements: int = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def execute(
        self,
        field: AnalyticPulseField,
        susceptibility: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
        distance: ArrayLike,
        /,
    ) -> UnidirectionalPropagationResult:
        return propagate_unidirectional(self, field, susceptibility, distance)


def _uniform_spacing(nodes: Array, name: str, /) -> float:
    host = np.asarray(nodes, dtype=float)
    if host.ndim != 1 or host.size < 4:
        raise ValueError(f"{name} requires at least four samples.")
    differences = np.diff(host)
    spacing = float(differences[0])
    tolerance = 64.0 * np.finfo(host.dtype).eps * max(1.0, abs(spacing))
    if (
        not np.isfinite(spacing)
        or spacing <= 0.0
        or not np.all(np.isfinite(differences))
        or not np.allclose(differences, spacing, rtol=1.0e-10, atol=tolerance)
    ):
        raise ValueError(f"{name} must have finite, increasing, uniform samples.")
    return spacing


def _forward_square_root(value: Array, /) -> Array:
    root = jnp.sqrt(value.astype(jnp.result_type(value.dtype, jnp.complex64)))
    wrong_branch = (jnp.imag(root) < 0.0) | (
        (jnp.imag(root) == 0.0) & (jnp.real(root) < 0.0)
    )
    return jnp.where(wrong_branch, -root, root)


def prepare_unidirectional_propagation(
    plan: UnidirectionalPropagationPlan,
    dispersion: AbstractRefractiveIndexLaw,
    /,
) -> PreparedUnidirectionalPropagation:
    """Resolve one admitted isotropic dispersion law on the immutable FFT grid."""
    if not isinstance(plan, UnidirectionalPropagationPlan):
        raise TypeError("plan must be a UnidirectionalPropagationPlan.")
    if not isinstance(dispersion, AbstractRefractiveIndexLaw):
        raise TypeError("dispersion must be an AbstractRefractiveIndexLaw.")
    spacing0 = _uniform_spacing(plan.space.coordinate_axes[0], "plane axis 0")
    spacing1 = _uniform_spacing(plan.space.coordinate_axes[1], "plane axis 1")
    temporal_spacing = _uniform_spacing(plan.temporal_grid.axes[0].nodes, "temporal axis")
    n0, n1 = plan.space.shape
    temporal_size = plan.temporal_grid.shape[0]
    if temporal_size < 8:
        raise ValueError("Nonlinear propagation requires at least eight time samples.")
    dtype = plan.angular_frequency.dtype
    omega = jnp.asarray(
        2.0 * np.pi * np.fft.fftfreq(temporal_size, d=temporal_spacing),
        dtype=dtype,
    )
    positive = omega > 0.0
    if not bool(np.any(np.asarray(positive))):
        raise ValueError("The temporal FFT grid has no strictly positive frequencies.")

    # The material API intentionally rejects non-positive omega. Feed inactive
    # lanes the first active positive frequency, then mask every resulting datum.
    # No inactive material value participates in execution or evidence.
    first_positive_frequency = jnp.min(jnp.where(positive, omega, jnp.inf))
    material_omega = jnp.where(positive, omega, first_positive_frequency)
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
        "Active medium wavenumbers must be finite with Re(k)>0 and Im(k)>=0.",
    )

    k0 = jnp.asarray(
        2.0 * np.pi * np.fft.fftfreq(n0, d=spacing0),
        dtype=dtype,
    )
    k1 = jnp.asarray(
        2.0 * np.pi * np.fft.fftfreq(n1, d=spacing1),
        dtype=dtype,
    )
    transverse0, transverse1 = jnp.meshgrid(k0, k1, indexing="ij")
    transverse = jnp.stack((transverse0, transverse1), axis=-1)
    transverse_squared = transverse0[..., None] ** 2 + transverse1[..., None] ** 2
    squared_medium = medium_wavenumbers[None, None, :] ** 2
    longitudinal = _forward_square_root(squared_medium - transverse_squared)
    active3 = positive[None, None, :]
    longitudinal = jnp.where(active3, longitudinal, 0.0)
    singular = active3 & (jnp.abs(longitudinal) <= jnp.finfo(dtype).eps)
    longitudinal = eqx.error_if(
        longitudinal,
        jnp.any(singular),
        "The resolved grid contains a longitudinal cutoff singularity.",
    )
    linear_generator = jnp.where(active3, 1j * longitudinal, 0.0)
    omega3 = omega[None, None, :]
    safe_longitudinal = jnp.where(active3, longitudinal, 1.0)
    source = jnp.where(
        active3,
        1j
        * (omega3 / dispersion.reference_wave_speed) ** 2
        / (2.0 * jnp.asarray(_VACUUM_PERMITTIVITY, dtype=dtype) * safe_longitudinal),
        0.0,
    )

    mode0 = jnp.abs(jnp.fft.fftfreq(n0) * n0)
    mode1 = jnp.abs(jnp.fft.fftfreq(n1) * n1)
    maximum_mode0 = max(1.0, float(n0 // 2))
    maximum_mode1 = max(1.0, float(n1 // 2))
    nyquist_omega = jnp.asarray(np.pi / temporal_spacing, dtype=dtype)
    spatial_dealias = (mode0[:, None, None] <= plan.dealias_fraction * maximum_mode0) & (
        mode1[None, :, None] <= plan.dealias_fraction * maximum_mode1
    )
    temporal_dealias = positive[None, None, :] & (
        omega3 <= plan.dealias_fraction * nyquist_omega
    )
    dealias_mask = spatial_dealias & temporal_dealias

    spatial_edge = (
        mode0[:, None, None] >= (1.0 - plan.edge_guard_fraction) * maximum_mode0
    ) | (mode1[None, :, None] >= (1.0 - plan.edge_guard_fraction) * maximum_mode1)
    temporal_edge = active3 & (
        (omega3 <= plan.edge_guard_fraction * nyquist_omega)
        | (omega3 >= (1.0 - plan.edge_guard_fraction) * nyquist_omega)
    )
    edge_mask = active3 & (spatial_edge | temporal_edge)
    active_longitudinal = jnp.where(active3, jnp.abs(longitudinal), jnp.inf)
    active_medium = jnp.where(active3, jnp.abs(medium_wavenumbers)[None, None, :], 1.0)
    minimum_fraction = jnp.min(active_longitudinal / active_medium)
    components = 1 if plan.polarization == "scalar" else 2
    complex_elements = 24 * prod((n0, n1, temporal_size, components))
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-cartesian-unidirectional-propagation",
            "plan": plan.plan_id,
            "law": dispersion.law_id,
            "provenance": dispersion.provenance.provenance_id,
            "shape": [n0, n1, temporal_size, components],
            "workspace_complex_elements": complex_elements,
        }
    )
    return PreparedUnidirectionalPropagation(
        plan=plan,
        absolute_angular_frequencies=omega,
        angular_frequency_offsets=omega - plan.angular_frequency,
        transverse_angular_frequencies=transverse,
        refractive_indices=refractive_indices,
        medium_wavenumbers=medium_wavenumbers,
        longitudinal_wavenumbers=longitudinal,
        linear_generator=linear_generator,
        nonlinear_source_factors=source,
        positive_frequency_mask=positive,
        dealias_mask=dealias_mask,
        spectral_edge_mask=edge_mask,
        dispersion_within_validity=jnp.where(positive, evaluation.within_validity, True),
        dispersion_extrapolated=jnp.where(positive, evaluation.extrapolated, False),
        dispersion_status=jnp.where(positive, evaluation.status, 0).astype(jnp.int32),
        minimum_longitudinal_wavenumber_fraction=minimum_fraction,
        workspace_complex_elements=complex_elements,
        law_id=dispersion.law_id,
        provenance_id=dispersion.provenance.provenance_id,
        prepared_id=prepared_id,
    )


def _to_spectrum(values: Array, /) -> Array:
    spatial = jnp.fft.fftn(values, axes=(0, 1), norm="ortho")
    return jnp.fft.ifft(spatial, axis=2, norm="ortho")


def _from_spectrum(values: Array, /) -> Array:
    temporal = jnp.fft.fft(values, axis=2, norm="ortho")
    return jnp.fft.ifftn(temporal, axes=(0, 1), norm="ortho")


def _component_multiplier(multiplier: Array, values: Array, /) -> Array:
    return multiplier if values.ndim == 3 else multiplier[..., None]


def _masked_fraction(values: Array, mask: Array, /) -> Array:
    resolved_mask = mask if values.ndim == 3 else mask[..., None]
    energy = jnp.abs(values) ** 2
    total = jnp.sum(energy)
    selected = jnp.sum(jnp.where(resolved_mask, energy, 0.0))
    safe_total = jnp.where(total > 0.0, total, 1.0)
    return jnp.where(total > 0.0, selected / safe_total, 0.0)


def _relative_error(candidate: Array, reference: Array, /) -> Array:
    numerator = jnp.sqrt(jnp.sum(jnp.abs(candidate - reference) ** 2))
    denominator = jnp.sqrt(jnp.sum(jnp.abs(candidate) ** 2))
    safe_denominator = jnp.where(denominator > 0.0, denominator, 1.0)
    return jnp.where(denominator > 0.0, numerator / safe_denominator, numerator)


def _analytic_polarization(
    prepared: PreparedUnidirectionalPropagation,
    values: Array,
    susceptibility: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
    /,
) -> Array:
    if prepared.plan.polarization == "scalar":
        if not isinstance(susceptibility, InstantaneousScalarSusceptibility):
            raise TypeError(
                "Scalar propagation requires InstantaneousScalarSusceptibility."
            )
        return instantaneous_nonlinear_polarization(
            susceptibility,
            values,
            prepared.positive_frequency_mask,
            temporal_axis=2,
        )
    if not isinstance(susceptibility, OrientedTensorSusceptibility):
        raise TypeError("Tangential propagation requires OrientedTensorSusceptibility.")
    basis = prepared.plan.space.transverse_basis.astype(values.real.dtype)
    lab_field = contract("ic,...c->...i", basis, values)
    lab_polarization = instantaneous_nonlinear_polarization(
        susceptibility,
        lab_field,
        prepared.positive_frequency_mask,
        temporal_axis=2,
    )
    return contract("ic,...i->...c", basis, lab_polarization)


def _nonlinear_rate(
    prepared: PreparedUnidirectionalPropagation,
    spectral_field: Array,
    susceptibility: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
    /,
) -> tuple[Array, Array, Array]:
    mask = (
        prepared.dealias_mask
        if spectral_field.ndim == 3
        else prepared.dealias_mask[..., None]
    )
    input_rejected_fraction = _masked_fraction(spectral_field, ~prepared.dealias_mask)
    if isinstance(susceptibility, InstantaneousScalarSusceptibility):
        response_active = (susceptibility.second_order != 0.0) | (
            susceptibility.third_order != 0.0
        )
    elif isinstance(susceptibility, OrientedTensorSusceptibility):
        response_active = jnp.any(susceptibility.second_order != 0.0) | jnp.any(
            susceptibility.third_order != 0.0
        )
    else:
        raise TypeError("Unsupported instantaneous susceptibility.")
    input_rejected_fraction = jnp.where(response_active, input_rejected_fraction, 0.0)
    nonlinear_spectrum = jnp.where(mask, spectral_field, 0.0)
    field = _from_spectrum(nonlinear_spectrum)
    polarization = _analytic_polarization(prepared, field, susceptibility)
    spectral_polarization = _to_spectrum(polarization)
    polarization_rejected_fraction = _masked_fraction(
        spectral_polarization, ~prepared.dealias_mask
    )
    rejected_fraction = jnp.maximum(
        input_rejected_fraction, polarization_rejected_fraction
    )
    filtered_polarization = jnp.where(mask, spectral_polarization, 0.0)
    source = _component_multiplier(
        prepared.nonlinear_source_factors, filtered_polarization
    )
    rate = source * filtered_polarization
    field_scale = jnp.max(jnp.abs(field))
    polarization_scale = jnp.max(jnp.abs(polarization))
    safe_field_scale = jnp.where(field_scale > 0.0, field_scale, 1.0)
    coupling = jnp.where(
        field_scale > 0.0,
        polarization_scale
        / (jnp.asarray(_VACUUM_PERMITTIVITY, dtype=field.real.dtype) * safe_field_scale),
        0.0,
    )
    return rate, rejected_fraction, 0.5 * coupling


def _interaction_picture_solve(
    prepared: PreparedUnidirectionalPropagation,
    initial_spectrum: Array,
    susceptibility: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
    distance: Array,
    step_count: int,
    /,
) -> tuple[Array, Array, Array]:
    step = distance / float(step_count)
    generator = _component_multiplier(prepared.linear_generator, initial_spectrum)
    half_linear = jnp.exp(0.5 * step * generator)
    full_linear = half_linear * half_linear
    inverse_half = jnp.exp(-0.5 * step * generator)
    inverse_full = inverse_half * inverse_half
    zero = jnp.asarray(0.0, dtype=initial_spectrum.real.dtype)

    def scan_step(carry, _):
        state, maximum_rejected, maximum_backward = carry
        k1, rejected1, backward1 = _nonlinear_rate(prepared, state, susceptibility)
        state2 = half_linear * (state + 0.5 * step * k1)
        raw2, rejected2, backward2 = _nonlinear_rate(prepared, state2, susceptibility)
        k2 = inverse_half * raw2
        state3 = half_linear * (state + 0.5 * step * k2)
        raw3, rejected3, backward3 = _nonlinear_rate(prepared, state3, susceptibility)
        k3 = inverse_half * raw3
        state4 = full_linear * (state + step * k3)
        raw4, rejected4, backward4 = _nonlinear_rate(prepared, state4, susceptibility)
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
        ), None

    final, _ = checkpointed_scan(
        scan_step,
        (initial_spectrum, zero, zero),
        jnp.arange(step_count, dtype=jnp.int32),
        length=step_count,
        mode="step",
    )
    return final


def _hermitian_reconstruction_defect(values: Array, /) -> Array:
    physical = jnp.real(values)
    spectrum = jnp.fft.ifft(physical, axis=2, norm="ortho")
    indices = (-jnp.arange(spectrum.shape[2], dtype=jnp.int32)) % spectrum.shape[2]
    partner = jnp.conj(jnp.take(spectrum, indices, axis=2))
    return _relative_error(spectrum, partner)


def propagate_unidirectional(
    prepared: PreparedUnidirectionalPropagation,
    field: AnalyticPulseField,
    susceptibility: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
    distance: ArrayLike,
    /,
) -> UnidirectionalPropagationResult:
    """Execute paired fixed-step interaction-picture RK4 propagation.

    The evolved spectrum obeys
    ``dE/dz = 1j*kz*E + 1j*(omega/c)^2*P_NL/(2*epsilon_0*kz)``.
    Spatial forward transforms use ``fft`` while the temporal forward transform
    uses ``ifft``; all are unitary. The fine solution uses the plan step count and
    a paired half-count solve supplies an observable refinement error.
    """
    if not isinstance(prepared, PreparedUnidirectionalPropagation):
        raise TypeError("prepared must be PreparedUnidirectionalPropagation.")
    if not isinstance(field, AnalyticPulseField):
        raise TypeError("field must be an AnalyticPulseField.")
    plan = prepared.plan
    if field.space.space_id != plan.space.space_id:
        raise ValueError("field and plan must use the same PlaneFieldSpace.")
    if field.temporal_grid.prepared_id != plan.temporal_grid.prepared_id:
        raise ValueError("field and plan must use the same temporal grid.")
    if field.polarization != plan.polarization:
        raise ValueError("field and plan polarization policies differ.")
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
    initial_spectrum = _to_spectrum(field.values)
    fine, nonlinear_rejected, backward_estimate = _interaction_picture_solve(
        prepared,
        initial_spectrum,
        susceptibility,
        safe_distance,
        plan.step_count,
    )
    coarse, _, _ = _interaction_picture_solve(
        prepared,
        initial_spectrum,
        susceptibility,
        safe_distance,
        plan.step_count // 2,
    )
    output_values = _from_spectrum(fine)
    output = AnalyticPulseField(
        field.space,
        field.temporal_grid,
        output_values,
        field.angular_frequency,
        field.longitudinal_coordinate + safe_distance,
        polarization=field.polarization,
    )
    edge_fraction = jnp.maximum(
        _masked_fraction(initial_spectrum, prepared.spectral_edge_mask),
        _masked_fraction(fine, prepared.spectral_edge_mask),
    )
    nonpositive = ~prepared.positive_frequency_mask[None, None, :]
    analytic_defect = jnp.maximum(
        _masked_fraction(initial_spectrum, nonpositive),
        _masked_fraction(fine, nonpositive),
    )
    refinement_error = _relative_error(fine, coarse)
    hermitian_defect = _hermitian_reconstruction_defect(output_values)
    extrapolated_fraction = jnp.sum(
        prepared.dispersion_extrapolated.astype(output_values.real.dtype)
    ) / jnp.sum(prepared.positive_frequency_mask)
    evidence = UnidirectionalApproximationEvidence(
        spectral_edge_fraction=edge_fraction,
        analytic_signal_defect=analytic_defect,
        hermitian_reconstruction_defect=hermitian_defect,
        nonlinear_rejected_fraction=nonlinear_rejected,
        nonlinear_bandwidth_margin=(
            plan.maximum_nonlinear_rejected_fraction - nonlinear_rejected
        ),
        fixed_step_refinement_error=refinement_error,
        backward_wave_estimate=backward_estimate,
        unidirectional_applicability_margin=(
            plan.maximum_backward_wave_estimate - backward_estimate
        ),
        dispersion_extrapolated_fraction=extrapolated_fraction,
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.real(output_values)))
        & jnp.all(jnp.isfinite(jnp.imag(output_values)))
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
                    )
                )
            )
        )
    )
    status = jnp.asarray(UnidirectionalPropagationStatus.SUCCESS, dtype=jnp.int32)

    def update(condition: Array, code: UnidirectionalPropagationStatus) -> None:
        nonlocal status
        status = jnp.where(
            (status == int(UnidirectionalPropagationStatus.SUCCESS)) & condition,
            int(code),
            status,
        ).astype(jnp.int32)

    update(~finite_input, UnidirectionalPropagationStatus.NONFINITE_INPUT)
    update(~distance_valid, UnidirectionalPropagationStatus.INVALID_DISTANCE)
    update(~frequency_compatible, UnidirectionalPropagationStatus.INCOMPATIBLE_FREQUENCY)
    update(~finite, UnidirectionalPropagationStatus.NUMERICAL_FAILURE)
    update(
        (analytic_defect > plan.maximum_analytic_signal_defect)
        | (hermitian_defect > plan.maximum_hermitian_reconstruction_defect),
        UnidirectionalPropagationStatus.ANALYTIC_SIGNAL_DEFECT,
    )
    update(
        edge_fraction > plan.maximum_spectral_edge_fraction,
        UnidirectionalPropagationStatus.SPECTRAL_EDGE_LIMIT,
    )
    update(
        nonlinear_rejected > plan.maximum_nonlinear_rejected_fraction,
        UnidirectionalPropagationStatus.NONLINEAR_BANDWIDTH_LIMIT,
    )
    update(
        refinement_error > plan.maximum_refinement_error,
        UnidirectionalPropagationStatus.REFINEMENT_LIMIT,
    )
    update(
        backward_estimate > plan.maximum_backward_wave_estimate,
        UnidirectionalPropagationStatus.UNIDIRECTIONAL_LIMIT,
    )
    successful = status == int(UnidirectionalPropagationStatus.SUCCESS)
    return UnidirectionalPropagationResult(
        field=output,
        evidence=evidence,
        finite=finite,
        status=status,
        successful=successful,
        prepared_id=prepared.prepared_id,
    )


__all__ = [
    "PreparedUnidirectionalPropagation",
    "UnidirectionalApproximationEvidence",
    "UnidirectionalPropagationPlan",
    "UnidirectionalPropagationResult",
    "UnidirectionalPropagationStatus",
    "prepare_unidirectional_propagation",
    "propagate_unidirectional",
]
