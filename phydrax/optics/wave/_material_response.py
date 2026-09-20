#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Causal finite-pulse Raman, ionization-rate, and Drude response plans."""

from __future__ import annotations

from math import prod
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._nonlinear_response import (
    _identifier,
    _nonnegative_finite_scalar,
    _positive_finite_scalar,
    _prepared_geometry,
    _prepared_response_id,
    _real_finite_array,
    _response_evaluation,
    _VACUUM_PERMITTIVITY,
    AbstractCarrierResolvedResponse,
    CarrierResolvedFieldKind,
    CarrierResolvedResponseEvaluation,
    PreparedCarrierResolvedResponse,
)
from ._pulse_time import PulseTimeSpace


# CODATA 2018 values. They are overridable only through an explicit Drude plan.
_ELEMENTARY_CHARGE = 1.602_176_634e-19
_ELECTRON_MASS = 9.109_383_701_5e-31


def _workspace_limit(value: int, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("maximum_workspace_bytes must be an integer.")
    result = int(value)
    if result <= 0:
        raise ValueError("maximum_workspace_bytes must be strictly positive.")
    return result


def _preflight_workspace(
    field_shape: tuple[int, ...], factor: int, maximum_workspace_bytes: int, /
) -> tuple[int, int]:
    elements = factor * prod(field_shape)
    # Physical response histories use real values, but use sixteen bytes per
    # element so float64 execution and one simultaneously retained complex source
    # are both covered before any history is materialized.
    workspace_bytes = 16 * elements
    if workspace_bytes > maximum_workspace_bytes:
        raise ValueError("Carrier-resolved response exceeds maximum_workspace_bytes.")
    return elements, workspace_bytes


def _without_temporal_axis(shape: tuple[int, ...], axis: int, /) -> tuple[int, ...]:
    return tuple(size for index, size in enumerate(shape) if index != axis)


def _time_integral(values: Array, temporal_axis: int, time_step: float, /) -> Array:
    axis = int(temporal_axis)
    earlier = [slice(None)] * values.ndim
    later = [slice(None)] * values.ndim
    earlier[axis] = slice(None, -1)
    later[axis] = slice(1, None)
    return float(time_step) * jnp.sum(
        0.5 * (values[tuple(earlier)] + values[tuple(later)]), axis=axis
    )


def _oscillator_transition(
    angular_frequency: Array, damping_rate: Array, time_step: float, dtype, /
) -> tuple[Array, Array, Array, Array]:
    omega = angular_frequency.astype(dtype)
    damping = damping_rate.astype(dtype)
    step = jnp.asarray(time_step, dtype=dtype)
    argument = (damping * damping - omega * omega) * step * step
    positive_root = jnp.sqrt(jnp.maximum(argument, 0.0))
    negative_root = jnp.sqrt(jnp.maximum(-argument, 0.0))
    positive_denominator = jnp.where(positive_root > 0.0, positive_root, 1.0)
    negative_denominator = jnp.where(negative_root > 0.0, negative_root, 1.0)
    positive_ratio = jnp.sinh(positive_root) / positive_denominator
    negative_ratio = jnp.sin(negative_root) / negative_denominator
    series = 1.0 + argument / 6.0 + argument * argument / 120.0
    ratio = jnp.where(
        jnp.abs(argument) < 1.0e-8,
        series,
        jnp.where(argument >= 0.0, positive_ratio, negative_ratio),
    )
    cosine = jnp.where(argument >= 0.0, jnp.cosh(positive_root), jnp.cos(negative_root))
    scaled_sine = step * ratio
    decay = jnp.exp(-damping * step)
    return (
        decay * (cosine + damping * scaled_sine),
        decay * scaled_sine,
        -decay * omega * omega * scaled_sine,
        decay * (cosine - damping * scaled_sine),
    )


class DelayedRamanResponsePlan(AbstractCarrierResolvedResponse):
    """Declared scalar delayed chi(3) oscillator over one finite pulse window.

    The physical oscillator obeys ``Q'' + 2 gamma Q' + Omega_R^2 Q =
    Omega_R^2 E^2`` with exactly zero displacement and velocity at the first
    pulse sample. ``P_R = epsilon_0 * delayed_third_order * E * Q``. Each time
    interval uses the exact damped-oscillator transition for a held left-endpoint
    drive, so causality never wraps across the Fourier cell.
    """

    delayed_third_order: Array
    oscillator_angular_frequency: Array
    damping_rate: Array
    maximum_workspace_bytes: int = eqx.field(static=True)
    _provenance_id: str = eqx.field(static=True)
    _response_id: str = eqx.field(static=True)

    def __init__(
        self,
        delayed_third_order: ArrayLike,
        oscillator_angular_frequency: ArrayLike,
        damping_rate: ArrayLike,
        /,
        *,
        provenance_id: str,
        maximum_workspace_bytes: int = 1 << 30,
    ):
        strength = _real_finite_array("delayed_third_order", delayed_third_order, ())
        frequency = _positive_finite_scalar(
            "oscillator_angular_frequency", oscillator_angular_frequency
        )
        damping = _nonnegative_finite_scalar("damping_rate", damping_rate)
        provenance = _identifier("provenance_id", provenance_id)
        workspace = _workspace_limit(maximum_workspace_bytes)
        self.delayed_third_order = strength
        self.oscillator_angular_frequency = frequency
        self.damping_rate = damping
        self.maximum_workspace_bytes = workspace
        self._provenance_id = provenance
        self._response_id = canonical_fingerprint(
            {
                "kind": "delayed-raman-response",
                "delayed_third_order": float(np.asarray(strength)),
                "oscillator_angular_frequency": float(np.asarray(frequency)),
                "damping_rate": float(np.asarray(damping)),
                "provenance": provenance,
                "maximum_workspace_bytes": workspace,
            }
        )

    @property
    def response_id(self) -> str:
        return self._response_id

    @property
    def provenance_id(self) -> str:
        return self._provenance_id

    @property
    def field_kind(self) -> CarrierResolvedFieldKind:
        return "scalar"

    def prepare(
        self,
        time_space: PulseTimeSpace,
        positive_frequency_mask: ArrayLike,
        field_shape: tuple[int, ...],
        /,
        *,
        temporal_axis: int,
    ) -> "PreparedDelayedRamanResponse":
        mask, shape, axis = _prepared_geometry(
            time_space, positive_frequency_mask, field_shape, temporal_axis
        )
        elements, workspace_bytes = _preflight_workspace(
            shape, 8, self.maximum_workspace_bytes
        )
        return PreparedDelayedRamanResponse(
            self,
            time_space,
            mask,
            field_shape=shape,
            temporal_axis=axis,
            workspace_real_elements=elements,
            workspace_bytes=workspace_bytes,
            prepared_id=_prepared_response_id(self.response_id, time_space, shape, axis),
        )


class PreparedDelayedRamanResponse(PreparedCarrierResolvedResponse):
    """Prepared exact-step causal Raman oscillator.

    Evaluation state channels are displacement ``Q`` then velocity ``Q'``.
    """

    plan: DelayedRamanResponsePlan
    _time_space: PulseTimeSpace
    _positive_frequency_mask: Array
    _field_shape: tuple[int, ...] = eqx.field(static=True)
    _temporal_axis: int = eqx.field(static=True)
    workspace_real_elements: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    _prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DelayedRamanResponsePlan,
        time_space: PulseTimeSpace,
        positive_frequency_mask: Array,
        /,
        *,
        field_shape: tuple[int, ...],
        temporal_axis: int,
        workspace_real_elements: int,
        workspace_bytes: int,
        prepared_id: str,
    ):
        self.plan = plan
        self._time_space = time_space
        self._positive_frequency_mask = positive_frequency_mask
        self._field_shape = field_shape
        self._temporal_axis = temporal_axis
        self.workspace_real_elements = workspace_real_elements
        self.workspace_bytes = workspace_bytes
        self._prepared_id = prepared_id

    @property
    def response_id(self) -> str:
        return self.plan.response_id

    @property
    def provenance_id(self) -> str:
        return self.plan.provenance_id

    @property
    def field_kind(self) -> CarrierResolvedFieldKind:
        return "scalar"

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
        physical = jnp.real(analytic)
        moved = jnp.moveaxis(physical, self.temporal_axis, 0)
        drive = moved * moved
        dtype = physical.dtype
        transition = _oscillator_transition(
            self.plan.oscillator_angular_frequency,
            self.plan.damping_rate,
            self.time_space.sample_spacing,
            dtype,
        )
        initial = (
            jnp.zeros(moved.shape[1:], dtype=dtype),
            jnp.zeros(moved.shape[1:], dtype=dtype),
        )

        def advance(state, forcing):
            displacement, velocity = state
            next_displacement = (
                transition[0] * displacement
                + transition[1] * velocity
                + (1.0 - transition[0]) * forcing
            )
            next_velocity = (
                transition[2] * displacement
                + transition[3] * velocity
                - transition[2] * forcing
            )
            next_state = (next_displacement, next_velocity)
            return next_state, next_state

        _, history = jax.lax.scan(advance, initial, drive[:-1])
        displacement_moved = jnp.concatenate((initial[0][None], history[0]), axis=0)
        velocity_moved = jnp.concatenate((initial[1][None], history[1]), axis=0)
        displacement = jnp.moveaxis(displacement_moved, 0, self.temporal_axis)
        velocity = jnp.moveaxis(velocity_moved, 0, self.temporal_axis)
        epsilon = jnp.asarray(_VACUUM_PERMITTIVITY, dtype=dtype)
        strength = self.plan.delayed_third_order.astype(dtype)
        active = strength != 0.0
        state_moved = jnp.stack(
            (
                jnp.where(active, displacement_moved, 0.0),
                jnp.where(active, velocity_moved, 0.0),
            ),
            axis=-1,
        )
        state = jnp.moveaxis(state_moved, 0, self.temporal_axis)
        polarization = epsilon * strength * physical * displacement
        zero = jnp.zeros_like(physical)
        oscillator_scale = (
            epsilon
            * jnp.abs(strength)
            / (2.0 * self.plan.oscillator_angular_frequency.astype(dtype) ** 2)
        )
        endpoint_q = jnp.take(
            displacement, physical.shape[self.temporal_axis] - 1, axis=self.temporal_axis
        )
        endpoint_v = jnp.take(
            velocity, physical.shape[self.temporal_axis] - 1, axis=self.temporal_axis
        )
        oscillator_energy = oscillator_scale * (
            endpoint_v * endpoint_v
            + self.plan.oscillator_angular_frequency.astype(dtype) ** 2
            * endpoint_q
            * endpoint_q
        )
        dissipated_power = (
            2.0
            * epsilon
            * jnp.abs(strength)
            * self.plan.damping_rate.astype(dtype)
            / self.plan.oscillator_angular_frequency.astype(dtype) ** 2
            * velocity
            * velocity
        )
        dissipated_energy = _time_integral(
            dissipated_power, self.temporal_axis, self.time_space.sample_spacing
        )
        ledger_zero = jnp.zeros(
            _without_temporal_axis(self.field_shape, self.temporal_axis), dtype=dtype
        )
        return _response_evaluation(
            physical_field=physical,
            physical_polarization=polarization,
            physical_current=zero,
            physical_state=state,
            positive_frequency_mask=self.positive_frequency_mask,
            temporal_axis=self.temporal_axis,
            time_step=self.time_space.sample_spacing,
            neutral_number_density=jnp.asarray(0.0, dtype=dtype),
            electron_density=zero,
            ionization_potential_energy_density=ledger_zero,
            collisional_energy_density=ledger_zero,
            raman_oscillator_energy_density=oscillator_energy,
            raman_dissipated_energy_density=dissipated_energy,
            terminal_polarization_energy_density=ledger_zero,
            terminal_kinetic_energy_density=ledger_zero,
            response_id=self.response_id,
            provenance_id=self.provenance_id,
            prepared_id=self.prepared_id,
        )


class MultiphotonIonizationRatePlan(StrictModule, NonTrainableState):
    """Declared power-law multiphoton ionization rate.

    ``rate_coefficient * abs(E)**(2 * photon_order)`` has units of inverse time.
    The coefficient and order are explicit user declarations; no atomic species,
    band-gap, tunneling, or cycle-averaged calibration is inferred.
    """

    rate_coefficient: Array
    photon_order: int = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    rate_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate_coefficient: ArrayLike,
        photon_order: int,
        /,
        *,
        provenance_id: str,
    ):
        coefficient = _nonnegative_finite_scalar("rate_coefficient", rate_coefficient)
        if isinstance(photon_order, bool) or not isinstance(photon_order, Integral):
            raise TypeError("photon_order must be an integer.")
        order = int(photon_order)
        if order < 2:
            raise ValueError("photon_order must be at least two.")
        provenance = _identifier("provenance_id", provenance_id)
        self.rate_coefficient = coefficient
        self.photon_order = order
        self.provenance_id = provenance
        self.rate_id = canonical_fingerprint(
            {
                "kind": "declared-multiphoton-ionization-rate",
                "rate_coefficient": float(np.asarray(coefficient)),
                "photon_order": order,
                "provenance": provenance,
            }
        )

    def rate(self, physical_electric_field: ArrayLike, /) -> Array:
        field = jnp.asarray(physical_electric_field)
        if jnp.iscomplexobj(field) or not jnp.issubdtype(field.dtype, jnp.number):
            raise TypeError("physical_electric_field must be real numeric.")
        field = field.astype(jnp.result_type(field.dtype, jnp.float32))
        return self.rate_coefficient.astype(field.dtype) * jnp.abs(field) ** (
            2 * self.photon_order
        )


class DrudePlasmaEvaluation(StrictModule):
    """Causal conduction current and its explicit material-energy channels."""

    physical_free_current: Array
    collisional_energy_density: Array
    terminal_kinetic_energy_density: Array
    finite: Array
    prepared_id: str = eqx.field(static=True)


class DrudePlasmaResponsePlan(StrictModule, NonTrainableState):
    """Declared causal Drude ADE component for a supplied electron history."""

    collision_angular_frequency: Array
    electron_charge_magnitude: Array
    electron_mass: Array
    maximum_workspace_bytes: int = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        collision_angular_frequency: ArrayLike,
        /,
        *,
        provenance_id: str,
        electron_charge_magnitude: ArrayLike = _ELEMENTARY_CHARGE,
        electron_mass: ArrayLike = _ELECTRON_MASS,
        maximum_workspace_bytes: int = 1 << 30,
    ):
        collision = _nonnegative_finite_scalar(
            "collision_angular_frequency", collision_angular_frequency
        )
        charge = _positive_finite_scalar(
            "electron_charge_magnitude", electron_charge_magnitude
        )
        mass = _positive_finite_scalar("electron_mass", electron_mass)
        provenance = _identifier("provenance_id", provenance_id)
        workspace = _workspace_limit(maximum_workspace_bytes)
        self.collision_angular_frequency = collision
        self.electron_charge_magnitude = charge
        self.electron_mass = mass
        self.maximum_workspace_bytes = workspace
        self.provenance_id = provenance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "drude-plasma-response",
                "collision_angular_frequency": float(np.asarray(collision)),
                "electron_charge_magnitude": float(np.asarray(charge)),
                "electron_mass": float(np.asarray(mass)),
                "provenance": provenance,
                "maximum_workspace_bytes": workspace,
            }
        )

    def prepare(
        self,
        time_space: PulseTimeSpace,
        field_shape: tuple[int, ...],
        /,
        *,
        temporal_axis: int,
    ) -> "PreparedDrudePlasmaResponse":
        dummy_mask = jnp.ones(time_space.shape, dtype=jnp.bool_)
        _, shape, axis = _prepared_geometry(
            time_space, dummy_mask, field_shape, temporal_axis
        )
        elements, workspace_bytes = _preflight_workspace(
            shape, 5, self.maximum_workspace_bytes
        )
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-drude-plasma-response",
                "plan": self.plan_id,
                "time_space": time_space.space_id,
                "field_shape": list(shape),
                "temporal_axis": axis,
            }
        )
        return PreparedDrudePlasmaResponse(
            self,
            time_space,
            field_shape=shape,
            temporal_axis=axis,
            workspace_real_elements=elements,
            workspace_bytes=workspace_bytes,
            prepared_id=prepared_id,
        )


class PreparedDrudePlasmaResponse(StrictModule, NonTrainableState):
    """Prepared exact held-source Drude current update."""

    plan: DrudePlasmaResponsePlan
    time_space: PulseTimeSpace
    field_shape: tuple[int, ...] = eqx.field(static=True)
    temporal_axis: int = eqx.field(static=True)
    workspace_real_elements: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: DrudePlasmaResponsePlan,
        time_space: PulseTimeSpace,
        /,
        *,
        field_shape: tuple[int, ...],
        temporal_axis: int,
        workspace_real_elements: int,
        workspace_bytes: int,
        prepared_id: str,
    ):
        self.plan = plan
        self.time_space = time_space
        self.field_shape = field_shape
        self.temporal_axis = temporal_axis
        self.workspace_real_elements = workspace_real_elements
        self.workspace_bytes = workspace_bytes
        self.prepared_id = prepared_id

    def evaluate(
        self,
        physical_electric_field: ArrayLike,
        electron_number_density: ArrayLike,
        /,
    ) -> DrudePlasmaEvaluation:
        field = jnp.asarray(physical_electric_field)
        density = jnp.asarray(electron_number_density)
        if field.shape != self.field_shape or density.shape != self.field_shape:
            raise ValueError(
                "field and electron density must match prepared field_shape."
            )
        if jnp.iscomplexobj(field) or jnp.iscomplexobj(density):
            raise TypeError("Drude field and electron density must be real.")
        dtype = jnp.result_type(field.dtype, density.dtype, jnp.float32)
        field = field.astype(dtype)
        density = density.astype(dtype)
        moved_field = jnp.moveaxis(field, self.temporal_axis, 0)
        moved_density = jnp.moveaxis(density, self.temporal_axis, 0)
        step = jnp.asarray(self.time_space.sample_spacing, dtype=dtype)
        collision = self.plan.collision_angular_frequency.astype(dtype)
        decay = jnp.exp(-collision * step)
        safe_collision = jnp.where(collision > 0.0, collision, 1.0)
        damped_scale = -jnp.expm1(-collision * step) / safe_collision
        collision_scale = jnp.where(collision > 0.0, damped_scale, step)
        acceleration_scale = self.plan.electron_charge_magnitude.astype(
            dtype
        ) ** 2 / self.plan.electron_mass.astype(dtype)
        initial = jnp.zeros(moved_field.shape[1:], dtype=dtype)

        def advance(current, inputs):
            interval_field, interval_density = inputs
            forcing = acceleration_scale * interval_density * interval_field
            next_current = decay * current + collision_scale * forcing
            return next_current, next_current

        _, history = jax.lax.scan(
            advance, initial, (moved_field[:-1], moved_density[:-1])
        )
        moved_current = jnp.concatenate((initial[None], history), axis=0)
        current = jnp.moveaxis(moved_current, 0, self.temporal_axis)
        charge_squared = self.plan.electron_charge_magnitude.astype(dtype) ** 2
        mass = self.plan.electron_mass.astype(dtype)
        safe_density = jnp.where(density > 0.0, density, 1.0)
        kinetic_history = jnp.where(
            density > 0.0,
            mass * current * current / (2.0 * charge_squared * safe_density),
            0.0,
        )
        collisional_power = 2.0 * collision * kinetic_history
        collisional_energy = _time_integral(
            collisional_power, self.temporal_axis, self.time_space.sample_spacing
        )
        terminal_kinetic = jnp.take(
            kinetic_history,
            self.field_shape[self.temporal_axis] - 1,
            axis=self.temporal_axis,
        )
        finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(collisional_energy))
            & jnp.all(jnp.isfinite(terminal_kinetic))
            & jnp.all(density >= 0.0)
        )
        return DrudePlasmaEvaluation(
            current,
            collisional_energy,
            terminal_kinetic,
            finite,
            prepared_id=self.prepared_id,
        )


class IonizingDrudeResponsePlan(AbstractCarrierResolvedResponse):
    """Composite scalar multiphoton depletion plus causal Drude ADE response.

    ``neutral_number_density`` is in inverse cubic meters and
    ``ionization_potential`` is the energy in joules required per new electron.
    """

    ionization: MultiphotonIonizationRatePlan
    drude: DrudePlasmaResponsePlan
    neutral_number_density: Array
    ionization_potential: Array
    maximum_workspace_bytes: int = eqx.field(static=True)
    _response_id: str = eqx.field(static=True)

    def __init__(
        self,
        ionization: MultiphotonIonizationRatePlan,
        drude: DrudePlasmaResponsePlan,
        neutral_number_density: ArrayLike,
        ionization_potential: ArrayLike,
        /,
        *,
        maximum_workspace_bytes: int = 1 << 30,
    ):
        if not isinstance(ionization, MultiphotonIonizationRatePlan):
            raise TypeError("ionization must be MultiphotonIonizationRatePlan.")
        if not isinstance(drude, DrudePlasmaResponsePlan):
            raise TypeError("drude must be DrudePlasmaResponsePlan.")
        density = _nonnegative_finite_scalar(
            "neutral_number_density", neutral_number_density
        )
        potential = _nonnegative_finite_scalar(
            "ionization_potential", ionization_potential
        )
        workspace = _workspace_limit(maximum_workspace_bytes)
        self.ionization = ionization
        self.drude = drude
        self.neutral_number_density = density
        self.ionization_potential = potential
        self.maximum_workspace_bytes = workspace
        self._response_id = canonical_fingerprint(
            {
                "kind": "ionizing-drude-response",
                "ionization": ionization.rate_id,
                "drude": drude.plan_id,
                "neutral_number_density": float(np.asarray(density)),
                "ionization_potential": float(np.asarray(potential)),
                "maximum_workspace_bytes": workspace,
            }
        )

    @property
    def response_id(self) -> str:
        return self._response_id

    @property
    def provenance_id(self) -> str:
        return canonical_fingerprint(
            {
                "ionization": self.ionization.provenance_id,
                "drude": self.drude.provenance_id,
            }
        )

    @property
    def field_kind(self) -> CarrierResolvedFieldKind:
        return "scalar"

    def prepare(
        self,
        time_space: PulseTimeSpace,
        positive_frequency_mask: ArrayLike,
        field_shape: tuple[int, ...],
        /,
        *,
        temporal_axis: int,
    ) -> "PreparedIonizingDrudeResponse":
        mask, shape, axis = _prepared_geometry(
            time_space, positive_frequency_mask, field_shape, temporal_axis
        )
        elements, workspace_bytes = _preflight_workspace(
            shape, 12, self.maximum_workspace_bytes
        )
        prepared_drude = self.drude.prepare(time_space, shape, temporal_axis=axis)
        return PreparedIonizingDrudeResponse(
            self,
            prepared_drude,
            time_space,
            mask,
            field_shape=shape,
            temporal_axis=axis,
            workspace_real_elements=elements,
            workspace_bytes=workspace_bytes,
            prepared_id=_prepared_response_id(self.response_id, time_space, shape, axis),
        )


class PreparedIonizingDrudeResponse(PreparedCarrierResolvedResponse):
    """Prepared exact-neutral-depletion and Drude-current pulse response.

    Evaluation state channels are electron number density then conductive Drude
    current density. The returned analytic free current additionally contains
    the ionization-potential loss current.
    """

    plan: IonizingDrudeResponsePlan
    drude: PreparedDrudePlasmaResponse
    _time_space: PulseTimeSpace
    _positive_frequency_mask: Array
    _field_shape: tuple[int, ...] = eqx.field(static=True)
    _temporal_axis: int = eqx.field(static=True)
    workspace_real_elements: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    _prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: IonizingDrudeResponsePlan,
        drude: PreparedDrudePlasmaResponse,
        time_space: PulseTimeSpace,
        positive_frequency_mask: Array,
        /,
        *,
        field_shape: tuple[int, ...],
        temporal_axis: int,
        workspace_real_elements: int,
        workspace_bytes: int,
        prepared_id: str,
    ):
        self.plan = plan
        self.drude = drude
        self._time_space = time_space
        self._positive_frequency_mask = positive_frequency_mask
        self._field_shape = field_shape
        self._temporal_axis = temporal_axis
        self.workspace_real_elements = workspace_real_elements
        self.workspace_bytes = workspace_bytes
        self._prepared_id = prepared_id

    @property
    def response_id(self) -> str:
        return self.plan.response_id

    @property
    def provenance_id(self) -> str:
        return self.plan.provenance_id

    @property
    def field_kind(self) -> CarrierResolvedFieldKind:
        return "scalar"

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
        physical = jnp.real(analytic)
        rate = self.plan.ionization.rate(physical)
        moved_rate = jnp.moveaxis(rate, self.temporal_axis, 0)
        dtype = physical.dtype
        neutral_total = self.plan.neutral_number_density.astype(dtype)
        step = jnp.asarray(self.time_space.sample_spacing, dtype=dtype)
        initial_neutral = jnp.full(moved_rate.shape[1:], neutral_total, dtype=dtype)

        def deplete(neutral, interval_rate):
            next_neutral = neutral * jnp.exp(-interval_rate * step)
            return next_neutral, next_neutral

        _, neutral_history = jax.lax.scan(deplete, initial_neutral, moved_rate[:-1])
        moved_neutral = jnp.concatenate((initial_neutral[None], neutral_history), axis=0)
        electron = neutral_total - jnp.moveaxis(moved_neutral, 0, self.temporal_axis)
        drude_evaluation = self.drude.evaluate(physical, electron)
        ionization_density_rate = rate * (neutral_total - electron)
        potential = self.plan.ionization_potential.astype(dtype)
        safe_field = jnp.where(physical != 0.0, physical, 1.0)
        ionization_current = jnp.where(
            physical != 0.0,
            potential * ionization_density_rate / safe_field,
            0.0,
        )
        total_current = drude_evaluation.physical_free_current + ionization_current
        zero = jnp.zeros_like(physical)
        state = jnp.stack((electron, drude_evaluation.physical_free_current), axis=-1)
        endpoint_electron = jnp.take(
            electron, self.field_shape[self.temporal_axis] - 1, axis=self.temporal_axis
        )
        ionization_energy = potential * endpoint_electron
        ledger_zero = jnp.zeros(
            _without_temporal_axis(self.field_shape, self.temporal_axis), dtype=dtype
        )
        return _response_evaluation(
            physical_field=physical,
            physical_polarization=zero,
            physical_current=total_current,
            physical_state=state,
            positive_frequency_mask=self.positive_frequency_mask,
            temporal_axis=self.temporal_axis,
            time_step=self.time_space.sample_spacing,
            neutral_number_density=neutral_total,
            electron_density=electron,
            ionization_potential_energy_density=ionization_energy,
            collisional_energy_density=drude_evaluation.collisional_energy_density,
            raman_oscillator_energy_density=ledger_zero,
            raman_dissipated_energy_density=ledger_zero,
            terminal_polarization_energy_density=ledger_zero,
            terminal_kinetic_energy_density=drude_evaluation.terminal_kinetic_energy_density,
            response_id=self.response_id,
            provenance_id=self.provenance_id,
            prepared_id=self.prepared_id,
        )


__all__ = [
    "DelayedRamanResponsePlan",
    "DrudePlasmaEvaluation",
    "DrudePlasmaResponsePlan",
    "IonizingDrudeResponsePlan",
    "MultiphotonIonizationRatePlan",
    "PreparedDelayedRamanResponse",
    "PreparedDrudePlasmaResponse",
    "PreparedIonizingDrudeResponse",
]
