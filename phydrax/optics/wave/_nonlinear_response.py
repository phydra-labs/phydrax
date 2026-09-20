#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from enum import IntEnum
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import RigidFrame
from ._fields import (
    _angular_frequency,
    _complex_field_values,
    _longitudinal_coordinate,
    PlaneFieldSpace,
)
from ._pulse_time import PulseTimeSpace


AnalyticPulsePolarization = Literal["scalar", "tangential"]
CarrierResolvedFieldKind = Literal["scalar", "lab-vector"]

# CODATA 2018. Susceptibilities below are electric SI susceptibilities, so the
# physical nonlinear polarization is epsilon_0 times the contracted response.
_VACUUM_PERMITTIVITY = 8.854_187_812_8e-12


def _real_finite_array(name: str, value: ArrayLike, shape: tuple[int, ...], /) -> Array:
    array = jnp.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if jnp.iscomplexobj(array) or not jnp.issubdtype(array.dtype, jnp.number):
        raise TypeError(f"{name} must be a real numeric array.")
    result = array.astype(jnp.result_type(array.dtype, jnp.float32))
    return eqx.error_if(
        result,
        jnp.any(~jnp.isfinite(result)),
        f"{name} must contain only finite values.",
    )


def _nonnegative_finite_scalar(name: str, value: ArrayLike, /) -> Array:
    result = _real_finite_array(name, value, ())
    return eqx.error_if(result, result < 0.0, f"{name} must be nonnegative.")


def _positive_finite_scalar(name: str, value: ArrayLike, /) -> Array:
    result = _real_finite_array(name, value, ())
    return eqx.error_if(result, result <= 0.0, f"{name} must be strictly positive.")


def _identifier(name: str, value: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{name} must be non-empty.")
    return identifier


class CarrierResolvedResponseStatus(IntEnum):
    """JAX-compatible disposition of one finite-pulse material evaluation."""

    SUCCESS = 0
    NONFINITE = 1
    ELECTRON_BOUNDS = 2


class PulseResponseLedger(StrictModule):
    """Local work ledger over one causal pulse window.

    Every array has the field shape with its temporal axis removed. Optical work
    is positive when energy passes from the field into the response. Material
    work is the sum of the explicitly represented terminal and dissipative
    channels; ``energy_closure_defect`` is reported rather than silently forced
    to zero.
    """

    optical_work_density: Array
    material_work_density: Array
    polarization_work_density: Array
    free_current_work_density: Array
    ionization_potential_energy_density: Array
    collisional_energy_density: Array
    raman_oscillator_energy_density: Array
    raman_dissipated_energy_density: Array
    terminal_polarization_energy_density: Array
    terminal_kinetic_energy_density: Array
    energy_closure_defect: Array


class CarrierResolvedResponseEvidence(StrictModule):
    """Observable initialization, endpoint, bound, and finiteness evidence."""

    initial_state_norm: Array
    endpoint_state_norm: Array
    finite_pulse_endpoint_residual: Array
    minimum_electron_density: Array
    maximum_electron_density: Array
    neutral_number_density: Array
    electron_bound_violation: Array
    initialized_at_window_entrance: Array
    finite: Array
    successful: Array


class CarrierResolvedResponseEvaluation(StrictModule):
    """Analytic sources, physical material history, ledger, and validity."""

    analytic_nonlinear_polarization: Array
    analytic_free_current: Array
    physical_state: Array
    ledger: PulseResponseLedger
    evidence: CarrierResolvedResponseEvidence
    finite: Array
    status: Array
    successful: Array
    response_id: str = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class AbstractCarrierResolvedResponse(StrictModule):
    """Plan contract for a fixed-shape carrier-resolved material response."""

    @property
    @abc.abstractmethod
    def response_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def provenance_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def field_kind(self) -> CarrierResolvedFieldKind:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare(
        self,
        time_space: PulseTimeSpace,
        positive_frequency_mask: ArrayLike,
        field_shape: tuple[int, ...],
        /,
        *,
        temporal_axis: int,
    ) -> "PreparedCarrierResolvedResponse":
        raise NotImplementedError


class PreparedCarrierResolvedResponse(StrictModule, NonTrainableState):
    """Prepared fixed-shape response evaluated from the pulse-window entrance."""

    __strict_abstract__ = True

    @property
    @abc.abstractmethod
    def response_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def provenance_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def field_kind(self) -> CarrierResolvedFieldKind:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def time_space(self) -> PulseTimeSpace:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def positive_frequency_mask(self) -> Array:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def field_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def temporal_axis(self) -> int:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def prepared_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(
        self, analytic_electric_field: ArrayLike, /
    ) -> CarrierResolvedResponseEvaluation:
        raise NotImplementedError


def _prepared_geometry(
    time_space: PulseTimeSpace,
    positive_frequency_mask: ArrayLike,
    field_shape: tuple[int, ...],
    temporal_axis: int,
    /,
) -> tuple[Array, tuple[int, ...], int]:
    if not isinstance(time_space, PulseTimeSpace):
        raise TypeError("time_space must be a PulseTimeSpace.")
    shape = tuple(field_shape)
    if not shape or any(size <= 0 for size in shape):
        raise ValueError("field_shape must contain only positive dimensions.")
    axis = int(temporal_axis) % len(shape)
    if shape[axis] != time_space.size:
        raise ValueError("field_shape temporal axis must match time_space.")
    mask = jnp.asarray(positive_frequency_mask, dtype=jnp.bool_)
    if mask.shape != time_space.shape:
        raise ValueError("positive_frequency_mask must match time_space.")
    mask = eqx.error_if(
        mask,
        ~jnp.any(mask),
        "positive_frequency_mask must contain an active bin.",
    )
    return mask, shape, axis


def _prepared_response_id(
    response_id: str,
    time_space: PulseTimeSpace,
    field_shape: tuple[int, ...],
    temporal_axis: int,
    /,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "prepared-carrier-resolved-response",
            "response": response_id,
            "time_space": time_space.space_id,
            "field_shape": list(field_shape),
            "temporal_axis": temporal_axis,
        }
    )


class InstantaneousScalarSusceptibility(AbstractCarrierResolvedResponse):
    """Lossless instantaneous scalar electric susceptibility through third order.

    ``second_order`` and ``third_order`` are the SI electric susceptibilities
    chi(2) and chi(3). For real electric field ``E`` the returned physical
    polarization is ``epsilon_0 * (chi2 * E**2 + chi3 * E**3)``. There is no
    delayed response, ionization, plasma state, or implicit linear material law.
    """

    second_order: Array
    third_order: Array

    def __init__(
        self,
        second_order: ArrayLike = 0.0,
        third_order: ArrayLike = 0.0,
        /,
    ):
        self.second_order = _real_finite_array("second_order", second_order, ())
        self.third_order = _real_finite_array("third_order", third_order, ())

    @property
    def response_id(self) -> str:
        return "instantaneous-scalar-susceptibility"

    @property
    def provenance_id(self) -> str:
        return "explicit-si-susceptibility"

    @property
    def field_kind(self) -> CarrierResolvedFieldKind:
        return "scalar"

    def physical_polarization(self, electric_field: ArrayLike, /) -> Array:
        """Evaluate the real instantaneous nonlinear polarization in SI units."""
        field = jnp.asarray(electric_field)
        if jnp.iscomplexobj(field) or not jnp.issubdtype(field.dtype, jnp.number):
            raise TypeError("physical scalar electric_field must be real numeric.")
        field = field.astype(jnp.result_type(field.dtype, jnp.float32))
        return jnp.asarray(_VACUUM_PERMITTIVITY, dtype=field.dtype) * (
            self.second_order * field**2 + self.third_order * field**3
        )

    def prepare(
        self,
        time_space: PulseTimeSpace,
        positive_frequency_mask: ArrayLike,
        field_shape: tuple[int, ...],
        /,
        *,
        temporal_axis: int,
    ) -> PreparedCarrierResolvedResponse:
        return _PreparedInstantaneousResponse(
            self,
            time_space,
            positive_frequency_mask,
            field_shape,
            temporal_axis=temporal_axis,
        )


class OrientedTensorSusceptibility(AbstractCarrierResolvedResponse):
    """Crystal-frame instantaneous chi(2)/chi(3) rotated into the lab frame.

    ``crystal_frame.rotation`` maps crystal components to lab components. The
    frame translation is intentionally irrelevant to this homogeneous local
    response. Tensors use output-first crystal indices ``chi2[i,j,k]`` and
    ``chi3[i,j,k,l]``; no symmetry is inserted or inferred.
    """

    second_order: Array
    third_order: Array
    crystal_frame: RigidFrame

    def __init__(
        self,
        second_order: ArrayLike,
        third_order: ArrayLike,
        crystal_frame: RigidFrame,
        /,
    ):
        if not isinstance(crystal_frame, RigidFrame) or crystal_frame.dimension != 3:
            raise ValueError("crystal_frame must be a three-dimensional RigidFrame.")
        self.second_order = _real_finite_array("second_order", second_order, (3, 3, 3))
        self.third_order = _real_finite_array("third_order", third_order, (3, 3, 3, 3))
        self.crystal_frame = crystal_frame

    @property
    def response_id(self) -> str:
        return "oriented-tensor-susceptibility"

    @property
    def provenance_id(self) -> str:
        return "explicit-si-susceptibility"

    @property
    def field_kind(self) -> CarrierResolvedFieldKind:
        return "lab-vector"

    def physical_polarization(self, electric_field: ArrayLike, /) -> Array:
        """Evaluate physical polarization for lab-frame real three-vectors."""
        field = jnp.asarray(electric_field)
        if field.ndim < 1 or field.shape[-1] != 3:
            raise ValueError("electric_field must have trailing shape (3,).")
        if jnp.iscomplexobj(field) or not jnp.issubdtype(field.dtype, jnp.number):
            raise TypeError("physical vector electric_field must be real numeric.")
        field = field.astype(jnp.result_type(field.dtype, jnp.float32))
        rotation = self.crystal_frame.rotation.astype(field.dtype)
        crystal_field = contract("ij,...i->...j", rotation, field)
        quadratic = contract(
            "ijk,...j,...k->...i",
            self.second_order,
            crystal_field,
            crystal_field,
        )
        cubic = contract(
            "ijkl,...j,...k,...l->...i",
            self.third_order,
            crystal_field,
            crystal_field,
            crystal_field,
        )
        crystal_polarization = jnp.asarray(_VACUUM_PERMITTIVITY, dtype=field.dtype) * (
            quadratic + cubic
        )
        return contract("ij,...j->...i", rotation, crystal_polarization)

    def prepare(
        self,
        time_space: PulseTimeSpace,
        positive_frequency_mask: ArrayLike,
        field_shape: tuple[int, ...],
        /,
        *,
        temporal_axis: int,
    ) -> PreparedCarrierResolvedResponse:
        if not field_shape or field_shape[-1] != 3:
            raise ValueError(
                "Oriented tensor response requires a trailing lab-vector axis."
            )
        return _PreparedInstantaneousResponse(
            self,
            time_space,
            positive_frequency_mask,
            field_shape,
            temporal_axis=temporal_axis,
        )


class AnalyticPulseField(StrictModule):
    """Carrier-resolved positive-frequency electric field on a plane and time space.

    The pulse-time topology must be a periodic Fourier cell. Values are the
    analytic electric field itself, not a slowly varying envelope. With the
    package phasor convention, a positive-frequency component varies as
    ``exp(-1j * omega * t)`` and the real field is ``real(values)``. The unitary
    forward temporal transform is therefore ``ifft(values, norm='ortho')``.
    DC and the even-grid Nyquist bin are outside the represented analytic band.

    A scalar pulse has shape ``space.shape + time_space.shape``. A tangential
    pulse appends a two-component ``(u, v)`` axis. Independent cases use ``vmap``;
    this value deliberately has no leading batch axes and owns no duplicate grid
    coordinates.
    """

    space: PlaneFieldSpace
    time_space: PulseTimeSpace
    values: Array
    angular_frequency: Array
    longitudinal_coordinate: Array
    polarization: AnalyticPulsePolarization = eqx.field(static=True)

    def __init__(
        self,
        space: PlaneFieldSpace,
        time_space: PulseTimeSpace,
        values: ArrayLike,
        angular_frequency: ArrayLike,
        longitudinal_coordinate: ArrayLike,
        /,
        *,
        polarization: AnalyticPulsePolarization = "scalar",
    ):
        if not isinstance(space, PlaneFieldSpace):
            raise TypeError("space must be a PlaneFieldSpace.")
        if not isinstance(time_space, PulseTimeSpace):
            raise TypeError("time_space must be a PulseTimeSpace.")
        if time_space.topology != "periodic-cell":
            raise ValueError("An analytic pulse requires periodic-cell pulse time.")
        if polarization not in ("scalar", "tangential"):
            raise ValueError("polarization must be 'scalar' or 'tangential'.")
        expected_shape = (
            space.shape + time_space.shape
            if polarization == "scalar"
            else space.shape + time_space.shape + (2,)
        )
        self.space = space
        self.time_space = time_space
        self.values = _complex_field_values("values", values, expected_shape)
        self.angular_frequency = _angular_frequency(angular_frequency)
        self.longitudinal_coordinate = _longitudinal_coordinate(longitudinal_coordinate)
        self.polarization = polarization

    @property
    def temporal_coordinates(self) -> Array:
        return self.time_space.coordinates

    @property
    def temporal_weights(self) -> Array:
        return self.time_space.weights

    @property
    def temporal_size(self) -> int:
        return self.time_space.size

    @property
    def polarization_components(self) -> int:
        return 1 if self.polarization == "scalar" else 2


def _project_physical_to_analytic(
    physical_polarization: Array,
    positive_frequency_mask: Array,
    temporal_axis: int,
    /,
) -> Array:
    axis = int(temporal_axis) % physical_polarization.ndim
    mask = jnp.asarray(positive_frequency_mask, dtype=jnp.bool_)
    if mask.shape != (physical_polarization.shape[axis],):
        raise ValueError("positive_frequency_mask must match the temporal-axis length.")
    mask_shape = [1] * physical_polarization.ndim
    mask_shape[axis] = mask.size
    resolved_mask = mask.reshape(tuple(mask_shape))
    physical_spectrum = jnp.fft.ifft(physical_polarization, axis=axis, norm="ortho")
    analytic_spectrum = jnp.where(resolved_mask, 2.0 * physical_spectrum, 0.0)
    return jnp.fft.fft(analytic_spectrum, axis=axis, norm="ortho")


def _interval_work(
    physical_field: Array,
    physical_polarization: Array,
    physical_current: Array,
    temporal_axis: int,
    time_step: float,
    /,
    *,
    sum_last_component: bool,
) -> tuple[Array, Array]:
    axis = int(temporal_axis)
    earlier = [slice(None)] * physical_field.ndim
    later = [slice(None)] * physical_field.ndim
    earlier[axis] = slice(None, -1)
    later[axis] = slice(1, None)
    e0 = physical_field[tuple(earlier)]
    e1 = physical_field[tuple(later)]
    p0 = physical_polarization[tuple(earlier)]
    p1 = physical_polarization[tuple(later)]
    j0 = physical_current[tuple(earlier)]
    j1 = physical_current[tuple(later)]
    polarization = jnp.sum(0.5 * (e0 + e1) * (p1 - p0), axis=axis)
    current = float(time_step) * jnp.sum(0.5 * (e0 * j0 + e1 * j1), axis=axis)
    if sum_last_component:
        polarization = jnp.sum(polarization, axis=-1)
        current = jnp.sum(current, axis=-1)
    return polarization, current


def _response_evaluation(
    *,
    physical_field: Array,
    physical_polarization: Array,
    physical_current: Array,
    physical_state: Array,
    positive_frequency_mask: Array,
    temporal_axis: int,
    time_step: float,
    neutral_number_density: Array,
    electron_density: Array,
    ionization_potential_energy_density: Array,
    collisional_energy_density: Array,
    raman_oscillator_energy_density: Array,
    raman_dissipated_energy_density: Array,
    terminal_polarization_energy_density: Array,
    terminal_kinetic_energy_density: Array,
    response_id: str,
    provenance_id: str,
    prepared_id: str,
    sum_last_component: bool = False,
) -> CarrierResolvedResponseEvaluation:
    axis = int(temporal_axis)
    polarization_work, current_work = _interval_work(
        physical_field,
        physical_polarization,
        physical_current,
        axis,
        time_step,
        sum_last_component=sum_last_component,
    )
    optical_work = polarization_work + current_work
    material_work = (
        ionization_potential_energy_density
        + collisional_energy_density
        + raman_oscillator_energy_density
        + raman_dissipated_energy_density
        + terminal_polarization_energy_density
        + terminal_kinetic_energy_density
    )
    ledger = PulseResponseLedger(
        optical_work,
        material_work,
        polarization_work,
        current_work,
        ionization_potential_energy_density,
        collisional_energy_density,
        raman_oscillator_energy_density,
        raman_dissipated_energy_density,
        terminal_polarization_energy_density,
        terminal_kinetic_energy_density,
        optical_work - material_work,
    )
    initial_state = jnp.take(physical_state, 0, axis=axis)
    endpoint_state = jnp.take(physical_state, physical_field.shape[axis] - 1, axis=axis)
    initial_norm = jnp.sqrt(jnp.sum(jnp.abs(initial_state) ** 2))
    endpoint_norm = jnp.sqrt(jnp.sum(jnp.abs(endpoint_state) ** 2))
    state_scale = jnp.sqrt(jnp.sum(jnp.abs(physical_state) ** 2))
    safe_scale = jnp.where(state_scale > 0.0, state_scale, 1.0)
    endpoint_residual = jnp.where(
        state_scale > 0.0, endpoint_norm / safe_scale, endpoint_norm
    )
    minimum_electron = jnp.min(electron_density)
    maximum_electron = jnp.max(electron_density)
    bound_violation = jnp.maximum(
        jnp.maximum(-minimum_electron, 0.0),
        jnp.maximum(maximum_electron - neutral_number_density, 0.0),
    )
    initialized = initial_norm == 0.0
    finite = (
        jnp.all(jnp.isfinite(physical_field))
        & jnp.all(jnp.isfinite(physical_polarization))
        & jnp.all(jnp.isfinite(physical_current))
        & jnp.all(jnp.isfinite(physical_state))
        & jnp.all(jnp.isfinite(electron_density))
        & jnp.all(
            jnp.isfinite(
                jnp.stack(
                    (
                        jnp.max(jnp.abs(optical_work)),
                        jnp.max(jnp.abs(material_work)),
                        endpoint_residual,
                        bound_violation,
                    )
                )
            )
        )
    )
    bounds_ok = bound_violation == 0.0
    successful = finite & bounds_ok & initialized
    status = jnp.where(
        ~finite | ~initialized,
        int(CarrierResolvedResponseStatus.NONFINITE),
        jnp.where(
            ~bounds_ok,
            int(CarrierResolvedResponseStatus.ELECTRON_BOUNDS),
            int(CarrierResolvedResponseStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    evidence = CarrierResolvedResponseEvidence(
        initial_norm,
        endpoint_norm,
        endpoint_residual,
        minimum_electron,
        maximum_electron,
        neutral_number_density,
        bound_violation,
        initialized,
        finite,
        successful,
    )
    return CarrierResolvedResponseEvaluation(
        _project_physical_to_analytic(
            physical_polarization, positive_frequency_mask, axis
        ),
        _project_physical_to_analytic(physical_current, positive_frequency_mask, axis),
        physical_state,
        ledger,
        evidence,
        finite,
        status,
        successful,
        response_id=response_id,
        provenance_id=provenance_id,
        prepared_id=prepared_id,
    )


class _PreparedInstantaneousResponse(PreparedCarrierResolvedResponse):
    response: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility
    _time_space: PulseTimeSpace
    _positive_frequency_mask: Array
    _field_shape: tuple[int, ...] = eqx.field(static=True)
    _temporal_axis: int = eqx.field(static=True)
    _prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        response: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
        time_space: PulseTimeSpace,
        positive_frequency_mask: ArrayLike,
        field_shape: tuple[int, ...],
        /,
        *,
        temporal_axis: int,
    ):
        mask, shape, axis = _prepared_geometry(
            time_space, positive_frequency_mask, field_shape, temporal_axis
        )
        self.response = response
        self._time_space = time_space
        self._positive_frequency_mask = mask
        self._field_shape = shape
        self._temporal_axis = axis
        self._prepared_id = _prepared_response_id(
            response.response_id, time_space, shape, axis
        )

    @property
    def response_id(self) -> str:
        return self.response.response_id

    @property
    def provenance_id(self) -> str:
        return self.response.provenance_id

    @property
    def field_kind(self) -> CarrierResolvedFieldKind:
        return self.response.field_kind

    @property
    def time_space(self) -> PulseTimeSpace:
        return self._time_space

    @property
    def positive_frequency_mask(self) -> Array:
        return self._positive_frequency_mask

    @property
    def field_shape(self) -> tuple[int, ...]:
        return self._field_shape

    @property
    def temporal_axis(self) -> int:
        return self._temporal_axis

    @property
    def prepared_id(self) -> str:
        return self._prepared_id

    def evaluate(
        self, analytic_electric_field: ArrayLike, /
    ) -> CarrierResolvedResponseEvaluation:
        analytic = jnp.asarray(analytic_electric_field)
        if analytic.shape != self.field_shape:
            raise ValueError(
                "analytic_electric_field does not match prepared field_shape."
            )
        if not jnp.iscomplexobj(analytic):
            raise TypeError("analytic_electric_field must be complex.")
        physical_field = jnp.real(analytic)
        physical_polarization = self.response.physical_polarization(physical_field)
        zero = jnp.zeros_like(physical_field)
        state = jnp.zeros(self.field_shape + (0,), dtype=physical_field.dtype)
        ledger_shape = tuple(
            size
            for index, size in enumerate(self.field_shape)
            if index != self.temporal_axis
        )
        if self.field_kind == "lab-vector":
            ledger_shape = ledger_shape[:-1]
        ledger_zero = jnp.zeros(ledger_shape, dtype=physical_field.dtype)
        if isinstance(self.response, InstantaneousScalarSusceptibility):
            entrance_field = jnp.take(physical_field, 0, axis=self.temporal_axis)
            endpoint_field = jnp.take(
                physical_field,
                self.field_shape[self.temporal_axis] - 1,
                axis=self.temporal_axis,
            )
            epsilon = jnp.asarray(_VACUUM_PERMITTIVITY, dtype=physical_field.dtype)
            terminal_polarization_energy = epsilon * (
                (2.0 / 3.0)
                * self.response.second_order
                * (endpoint_field**3 - entrance_field**3)
                + (3.0 / 4.0)
                * self.response.third_order
                * (endpoint_field**4 - entrance_field**4)
            )
        else:
            terminal_polarization_energy = ledger_zero
        return _response_evaluation(
            physical_field=physical_field,
            physical_polarization=physical_polarization,
            physical_current=zero,
            physical_state=state,
            positive_frequency_mask=self.positive_frequency_mask,
            temporal_axis=self.temporal_axis,
            time_step=self.time_space.sample_spacing,
            neutral_number_density=jnp.asarray(0.0, dtype=physical_field.dtype),
            electron_density=zero,
            ionization_potential_energy_density=ledger_zero,
            collisional_energy_density=ledger_zero,
            raman_oscillator_energy_density=ledger_zero,
            raman_dissipated_energy_density=ledger_zero,
            terminal_polarization_energy_density=terminal_polarization_energy,
            terminal_kinetic_energy_density=ledger_zero,
            response_id=self.response_id,
            provenance_id=self.provenance_id,
            prepared_id=self.prepared_id,
            sum_last_component=self.field_kind == "lab-vector",
        )


def instantaneous_nonlinear_polarization(
    susceptibility: InstantaneousScalarSusceptibility | OrientedTensorSusceptibility,
    analytic_electric_field: ArrayLike,
    positive_frequency_mask: ArrayLike,
    /,
    *,
    temporal_axis: int = -1,
) -> Array:
    """Return the exact represented-band analytic part of an instantaneous response.

    The real field is reconstructed as ``real(E_plus)``. Its physical nonlinear
    polarization is evaluated before a unitary temporal transform projects onto
    strictly positive ``exp(-1j*omega*t)`` bins. Positive bins are doubled; DC,
    non-positive bins, and the even-grid Nyquist bin are zero. This construction
    retains both sum- and difference-frequency mixing and fixes the chi factors
    without a rotating-wave or envelope approximation.
    """
    analytic = jnp.asarray(analytic_electric_field)
    if not jnp.iscomplexobj(analytic):
        raise TypeError("analytic_electric_field must be complex.")
    axis = int(temporal_axis) % analytic.ndim
    if isinstance(susceptibility, InstantaneousScalarSusceptibility):
        physical = susceptibility.physical_polarization(jnp.real(analytic))
    elif isinstance(susceptibility, OrientedTensorSusceptibility):
        if analytic.shape[-1] != 3 or axis == analytic.ndim - 1:
            raise ValueError(
                "Tensor response requires lab-vector fields with trailing shape (3,) and a distinct temporal axis."
            )
        physical = susceptibility.physical_polarization(jnp.real(analytic))
    else:
        raise TypeError(
            "susceptibility must be InstantaneousScalarSusceptibility or OrientedTensorSusceptibility."
        )
    return _project_physical_to_analytic(
        physical,
        jnp.asarray(positive_frequency_mask, dtype=jnp.bool_),
        axis,
    )


def prepare_carrier_resolved_response(
    response: AbstractCarrierResolvedResponse,
    time_space: PulseTimeSpace,
    positive_frequency_mask: ArrayLike,
    field_shape: tuple[int, ...],
    /,
    *,
    temporal_axis: int,
) -> PreparedCarrierResolvedResponse:
    """Prepare one response against an immutable pulse grid and exact field shape."""
    if not isinstance(response, AbstractCarrierResolvedResponse):
        raise TypeError("response must implement AbstractCarrierResolvedResponse.")
    return response.prepare(
        time_space,
        positive_frequency_mask,
        field_shape,
        temporal_axis=temporal_axis,
    )


__all__ = [
    "AbstractCarrierResolvedResponse",
    "AnalyticPulseField",
    "CarrierResolvedResponseEvaluation",
    "CarrierResolvedResponseEvidence",
    "CarrierResolvedResponseStatus",
    "InstantaneousScalarSusceptibility",
    "OrientedTensorSusceptibility",
    "PreparedCarrierResolvedResponse",
    "PulseResponseLedger",
    "instantaneous_nonlinear_polarization",
    "prepare_carrier_resolved_response",
]
