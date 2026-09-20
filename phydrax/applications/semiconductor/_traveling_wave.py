#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reduced longitudinal semiconductor traveling-wave laser dynamics.

This is an explicit carrier-pair rate model coupled to power-normalized forward
and backward envelopes. It is not a drift-diffusion discretization and does not
claim to recover electrostatic, current-continuity, or quasi-Fermi fields. A
uniform characteristic grid advances exactly one cell per fixed time step.
"""

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._numerics._checkpointed_scan import checkpointed_scan
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._optical_response import (
    CarrierOpticalResponsePlan,
    evaluate_semiconductor_optical_response,
    LinearizedCarrierOpticalResponsePlan,
    TabulatedCarrierOpticalResponsePlan,
)
from ._quantities import ELEMENTARY_CHARGE_SI


_REDUCED_PLANCK_CONSTANT = 1.054_571_817e-34


class TravelingWaveLaserStatus(IntFlag):
    """Fail-closed threshold or transient disposition."""

    SUCCESS = 0
    NONFINITE = 1
    OPTICAL_RESPONSE_REJECTED = 2
    CARRIER_OUTSIDE_BOUNDS = 4
    INPUT_REJECTED = 8
    LEDGER_DEFECT = 16
    THRESHOLD_NOT_BRACKETED = 32
    NO_CLOSED_CAVITY = 64
    THRESHOLD_MODE_NOT_CONVERGED = 128
    GRATING_UNITARITY_DEFECT = 256
    THRESHOLD_MODE_NOT_ISOLATED = 512


class TravelingWaveSemiconductorLaserState(StrictModule):
    """Reduced state: neutral pair density and sqrt(W) traveling amplitudes."""

    carrier_pair_density: Array
    forward_field: Array
    backward_field: Array

    def __init__(
        self,
        carrier_pair_density: ArrayLike,
        forward_field: ArrayLike,
        backward_field: ArrayLike,
        /,
    ):
        density = jnp.asarray(carrier_pair_density)
        forward = jnp.asarray(forward_field)
        backward = jnp.asarray(backward_field)
        if (
            density.ndim != 1
            or forward.shape != density.shape
            or backward.shape != density.shape
        ):
            raise ValueError(
                "State density and both fields must be equal one-dimensional vectors."
            )
        if jnp.iscomplexobj(density):
            raise TypeError("carrier_pair_density must be real.")
        dtype = jnp.result_type(forward.dtype, backward.dtype, jnp.complex64)
        self.carrier_pair_density = density
        self.forward_field = forward.astype(dtype)
        self.backward_field = backward.astype(dtype)


class TravelingWaveLaserInput(StrictModule):
    """Fixed-step drive: total injection current and lattice temperature.

    Current is amperes delivered to the active region. It may be scalar or have
    shape ``(step_count,)``. Temperature may be scalar, ``(section_count,)``, or
    ``(step_count, section_count)``. The prepared injection weights distribute
    charge-converted carrier pairs across physical section volumes.
    """

    injection_current: Array
    lattice_temperature: Array

    def __init__(
        self,
        injection_current: ArrayLike,
        lattice_temperature: ArrayLike,
        /,
    ):
        current = jnp.asarray(injection_current)
        temperature = jnp.asarray(lattice_temperature)
        if jnp.iscomplexobj(current) or jnp.iscomplexobj(temperature):
            raise TypeError("Laser current and lattice temperature must be real.")
        self.injection_current = current
        self.lattice_temperature = temperature


class TravelingWaveLaserNoisePlan(StrictModule, NonTrainableState):
    """Explicit replayable additive increments, separate from deterministic physics."""

    field_amplitude_standard_deviation: Array
    carrier_density_standard_deviation: Array
    provenance: str = eqx.field(static=True)
    noise_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_amplitude_standard_deviation: ArrayLike,
        carrier_density_standard_deviation: ArrayLike,
        /,
        *,
        provenance: str,
        noise_id: str | None = None,
    ):
        field = np.asarray(field_amplitude_standard_deviation)
        carrier = np.asarray(carrier_density_standard_deviation)
        if (
            field.shape != ()
            or carrier.shape != ()
            or np.iscomplexobj(field)
            or np.iscomplexobj(carrier)
            or not np.isfinite(field)
            or not np.isfinite(carrier)
            or field < 0.0
            or carrier < 0.0
        ):
            raise ValueError(
                "Noise standard deviations must be finite nonnegative scalars."
            )
        provenance_ = provenance.strip()
        if not provenance_:
            raise ValueError("Noise provenance must be explicit nonempty text.")
        self.field_amplitude_standard_deviation = jnp.asarray(field)
        self.carrier_density_standard_deviation = jnp.asarray(carrier)
        self.provenance = provenance_
        generated = canonical_fingerprint(
            {
                "kind": "traveling-wave-laser-noise-plan",
                "field_amplitude_standard_deviation": float(field).hex(),
                "carrier_density_standard_deviation": float(carrier).hex(),
                "provenance": provenance_,
            }
        )
        identifier = generated if noise_id is None else str(noise_id)
        if not identifier:
            raise ValueError("noise_id must be non-empty.")
        self.noise_id = identifier

    def realize(
        self,
        prepared: "PreparedTravelingWaveSemiconductorLaser",
        key: Array,
        /,
    ) -> "TravelingWaveLaserNoiseRealization":
        return realize_traveling_wave_laser_noise(self, prepared, key)


class TravelingWaveLaserNoiseRealization(StrictModule, NonTrainableState):
    """A fixed-shape forcing tape; identical tapes replay bit-for-bit."""

    carrier_density_increment: Array
    field_amplitude_increment: Array
    noise_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)


class TravelingWaveSemiconductorLaserPlan(StrictModule, NonTrainableState):
    """Static reduced longitudinal cavity, carrier rates, and resource policy.

    Facets are complex *amplitude* reflections. Their squared magnitudes are
    power reflectivities and must not exceed one. Per-section reciprocal
    ``coupling`` and real ``detuning`` define an exact local unitary grating
    rotation before each characteristic shift. A/B/C coefficients define
    ``R(N)=A*N+B*N**2+C*N**3`` in carrier pairs m^-3 s^-1. One time step is
    fixed to ``dz/group_velocity``; this exact characteristic shift avoids a
    hidden CFL branch and gives a closed photon ledger.
    """

    z_grid: Array
    optical_response: CarrierOpticalResponsePlan
    carrier_angular_frequency: Array
    group_velocity: Array
    active_area: Array
    recombination_a: Array
    recombination_b: Array
    recombination_c: Array
    injection_efficiency: Array
    injection_weights: Array
    gain_compression: Array
    linewidth_enhancement_factor: Array
    detuning: Array
    coupling: Array
    left_facet_amplitude_reflection: Array
    right_facet_amplitude_reflection: Array
    carrier_density_bounds: Array
    step_count: int = eqx.field(static=True)
    maximum_sections: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_threshold_map_applications: int = eqx.field(static=True)
    ledger_tolerance: float = eqx.field(static=True)
    grating_tolerance: float = eqx.field(static=True)
    has_distributed_grating: bool = eqx.field(static=True)
    grating_provenance: str = eqx.field(static=True)
    grating_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        z_grid: ArrayLike,
        optical_response: CarrierOpticalResponsePlan,
        carrier_angular_frequency: ArrayLike,
        /,
        *,
        group_velocity: ArrayLike,
        active_area: ArrayLike,
        recombination_a: ArrayLike,
        recombination_b: ArrayLike,
        recombination_c: ArrayLike,
        left_facet_amplitude_reflection: ArrayLike,
        right_facet_amplitude_reflection: ArrayLike,
        carrier_density_bounds: ArrayLike,
        step_count: int,
        injection_efficiency: ArrayLike = 1.0,
        injection_weights: ArrayLike | None = None,
        gain_compression: ArrayLike = 0.0,
        linewidth_enhancement_factor: ArrayLike = 0.0,
        detuning: ArrayLike | None = None,
        coupling: ArrayLike | None = None,
        grating_provenance: str | None = None,
        grating_tolerance: float = 1.0e-12,
        maximum_sections: int = 100_000,
        maximum_steps: int = 1_000_000,
        maximum_workspace_bytes: int = 1 << 30,
        maximum_threshold_map_applications: int = 1_000_000,
        ledger_tolerance: float = 1.0e-9,
        plan_id: str | None = None,
    ):
        if not isinstance(
            optical_response,
            (LinearizedCarrierOpticalResponsePlan, TabulatedCarrierOpticalResponsePlan),
        ):
            raise TypeError("optical_response must be a carrier optical-response plan.")
        z_input = np.asarray(z_grid)
        if (
            z_input.ndim != 1
            or z_input.size < 2
            or np.iscomplexobj(z_input)
            or np.any(~np.isfinite(z_input))
            or np.any(np.diff(z_input) <= 0.0)
        ):
            raise ValueError("z_grid must be a finite increasing real vector.")
        z = np.asarray(z_input, dtype=np.float64)
        widths = np.diff(z)
        tolerance = 64.0 * np.finfo(widths.dtype).eps * max(1.0, abs(float(widths[0])))
        if not np.allclose(widths, widths[0], rtol=1.0e-12, atol=tolerance):
            raise ValueError("Traveling-wave characteristic cells must be uniform.")
        sections = z.size - 1
        count = int(step_count)
        section_limit = int(maximum_sections)
        step_limit = int(maximum_steps)
        workspace_limit = int(maximum_workspace_bytes)
        if count < 1 or section_limit < 1 or step_limit < 1 or workspace_limit < 1:
            raise ValueError("Laser step and resource bounds must be positive integers.")
        if sections > section_limit or count > step_limit:
            raise ValueError("Laser topology exceeds declared fixed resource bounds.")
        detuning_host = (
            np.zeros((sections,), dtype=np.float64)
            if detuning is None
            else np.asarray(detuning)
        )
        coupling_host = (
            np.zeros((sections,), dtype=np.complex128)
            if coupling is None
            else np.asarray(coupling)
        )
        if (
            detuning_host.shape != (sections,)
            or np.iscomplexobj(detuning_host)
            or np.any(~np.isfinite(detuning_host))
        ):
            raise ValueError("detuning must be one finite real value per section.")
        if coupling_host.shape != (sections,) or np.any(
            ~np.isfinite(np.real(coupling_host)) | ~np.isfinite(np.imag(coupling_host))
        ):
            raise ValueError("coupling must be one finite complex value per section.")
        has_grating = bool(np.any(coupling_host != 0.0))
        if grating_provenance is None:
            if has_grating:
                raise ValueError(
                    "Nonzero distributed coupling requires explicit grating_provenance."
                )
            grating_provenance_ = "explicit zero-coupling no-grating regime"
        else:
            grating_provenance_ = str(grating_provenance).strip()
            if not grating_provenance_:
                raise ValueError("grating_provenance must be nonempty.")
        grating_tolerance_ = float(grating_tolerance)
        if not np.isfinite(grating_tolerance_) or grating_tolerance_ < 0.0:
            raise ValueError("grating_tolerance must be finite and nonnegative.")
        grating_id = canonical_fingerprint(
            {
                "kind": "traveling-wave-distributed-grating",
                "detuning": array_tree_fingerprint(detuning_host),
                "coupling": array_tree_fingerprint(coupling_host),
                "provenance": grating_provenance_,
            }
        )

        def scalar(value, name, *, positive=False, nonnegative=False):
            host = np.asarray(value)
            if host.shape != () or np.iscomplexobj(host) or not np.isfinite(host):
                raise ValueError(f"{name} must be one finite real scalar.")
            if positive and host <= 0.0:
                raise ValueError(f"{name} must be positive.")
            if nonnegative and host < 0.0:
                raise ValueError(f"{name} must be nonnegative.")
            return jnp.asarray(host)

        frequency = scalar(
            carrier_angular_frequency, "carrier_angular_frequency", positive=True
        )
        velocity = scalar(group_velocity, "group_velocity", positive=True)
        area = scalar(active_area, "active_area", positive=True)
        coefficient_a = scalar(recombination_a, "recombination_a", nonnegative=True)
        coefficient_b = scalar(recombination_b, "recombination_b", nonnegative=True)
        coefficient_c = scalar(recombination_c, "recombination_c", nonnegative=True)
        efficiency = scalar(injection_efficiency, "injection_efficiency", positive=True)
        if float(efficiency) > 1.0:
            raise ValueError("injection_efficiency must lie in (0, 1].")
        compression = scalar(gain_compression, "gain_compression", nonnegative=True)
        linewidth = scalar(
            linewidth_enhancement_factor,
            "linewidth_enhancement_factor",
            nonnegative=True,
        )
        bounds = np.asarray(carrier_density_bounds)
        if (
            bounds.shape != (2,)
            or np.iscomplexobj(bounds)
            or np.any(~np.isfinite(bounds))
            or bounds[0] < 0.0
            or bounds[1] <= bounds[0]
        ):
            raise ValueError("carrier_density_bounds must be increasing and nonnegative.")
        if bounds[0] < float(optical_response.density_range[0]) or bounds[1] > float(
            optical_response.density_range[1]
        ):
            raise ValueError(
                "Laser carrier bounds must lie within optical-response support."
            )
        if not (
            float(optical_response.angular_frequency_range[0])
            <= float(frequency)
            <= float(optical_response.angular_frequency_range[1])
        ):
            raise ValueError(
                "carrier_angular_frequency lies outside optical-response support."
            )

        reflections = tuple(
            np.asarray(value)
            for value in (
                left_facet_amplitude_reflection,
                right_facet_amplitude_reflection,
            )
        )
        if any(
            value.shape != ()
            or not np.isfinite(np.real(value))
            or not np.isfinite(np.imag(value))
            or abs(complex(value)) > 1.0
            for value in reflections
        ):
            raise ValueError(
                "Facet amplitude reflections must be finite with magnitude <= 1."
            )
        volumes = widths * float(area)
        if injection_weights is None:
            weights = volumes / np.sum(volumes)
        else:
            weights = np.asarray(injection_weights)
            if (
                weights.shape != (sections,)
                or np.iscomplexobj(weights)
                or np.any(~np.isfinite(weights))
                or np.any(weights < 0.0)
                or not np.isclose(np.sum(weights), 1.0, rtol=1.0e-12, atol=1.0e-14)
            ):
                raise ValueError(
                    "injection_weights must be a nonnegative section vector summing to one."
                )
        ledger = float(ledger_tolerance)
        if not np.isfinite(ledger) or ledger < 0.0:
            raise ValueError("ledger_tolerance must be finite and nonnegative.")
        self.z_grid = jnp.asarray(z)
        self.optical_response = optical_response
        self.carrier_angular_frequency = frequency
        self.group_velocity = velocity
        self.active_area = area
        self.recombination_a = coefficient_a
        self.recombination_b = coefficient_b
        self.recombination_c = coefficient_c
        self.injection_efficiency = efficiency
        self.injection_weights = jnp.asarray(weights)
        self.gain_compression = compression
        self.linewidth_enhancement_factor = linewidth
        complex_dtype = jnp.result_type(
            reflections[0].dtype,
            reflections[1].dtype,
            coupling_host.dtype,
            jnp.complex64,
        )
        real_dtype = jnp.real(jnp.zeros((), dtype=complex_dtype)).dtype
        self.detuning = jnp.asarray(detuning_host, dtype=real_dtype)
        self.coupling = jnp.asarray(coupling_host, dtype=complex_dtype)
        self.left_facet_amplitude_reflection = jnp.asarray(
            reflections[0], dtype=complex_dtype
        )
        self.right_facet_amplitude_reflection = jnp.asarray(
            reflections[1], dtype=complex_dtype
        )
        self.carrier_density_bounds = jnp.asarray(bounds)
        self.step_count = count
        self.maximum_sections = section_limit
        self.maximum_steps = step_limit
        self.maximum_workspace_bytes = workspace_limit
        threshold_application_limit = int(maximum_threshold_map_applications)
        if threshold_application_limit < 1:
            raise ValueError(
                "maximum_threshold_map_applications must be a positive integer."
            )
        self.maximum_threshold_map_applications = threshold_application_limit
        self.ledger_tolerance = ledger
        self.grating_tolerance = grating_tolerance_
        self.has_distributed_grating = has_grating
        self.grating_provenance = grating_provenance_
        self.grating_id = grating_id
        generated = canonical_fingerprint(
            {
                "kind": "traveling-wave-semiconductor-laser-plan",
                "z_grid": array_tree_fingerprint(z),
                "optical_response": optical_response.model_id,
                "carrier_angular_frequency": float(frequency).hex(),
                "group_velocity": float(velocity).hex(),
                "active_area": float(area).hex(),
                "recombination": [
                    float(coefficient_a).hex(),
                    float(coefficient_b).hex(),
                    float(coefficient_c).hex(),
                ],
                "injection_efficiency": float(efficiency).hex(),
                "injection_weights": array_tree_fingerprint(weights),
                "gain_compression": float(compression).hex(),
                "linewidth_enhancement_factor": float(linewidth).hex(),
                "grating_id": grating_id,
                "grating_tolerance": grating_tolerance_.hex(),
                "facet_reflections": [
                    [
                        float(np.real(reflections[0])).hex(),
                        float(np.imag(reflections[0])).hex(),
                    ],
                    [
                        float(np.real(reflections[1])).hex(),
                        float(np.imag(reflections[1])).hex(),
                    ],
                ],
                "carrier_density_bounds": array_tree_fingerprint(bounds),
                "step_count": count,
                "ledger_tolerance": ledger.hex(),
                "maximum_sections": section_limit,
                "maximum_steps": step_limit,
                "maximum_workspace_bytes": workspace_limit,
                "maximum_threshold_map_applications": threshold_application_limit,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.plan_id = identifier

    def prepare(self, /) -> "PreparedTravelingWaveSemiconductorLaser":
        return prepare_traveling_wave_semiconductor_laser(self)


class PreparedTravelingWaveSemiconductorLaser(StrictModule, NonTrainableState):
    """Fixed characteristic geometry and preflighted retained/workspace shapes."""

    plan: TravelingWaveSemiconductorLaserPlan
    section_widths: Array
    section_volumes: Array
    time_step: Array
    times: Array
    grating_step: Array
    maximum_grating_unitarity_error: Array
    left_power_reflectivity: Array
    right_power_reflectivity: Array
    section_count: int = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    retained_result_bytes: int = eqx.field(static=True)
    grating_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class TravelingWaveLaserLedger(StrictModule):
    """Carrier-pair and photon inventories, all as dimensionless counts."""

    initial_carrier_pairs: Array
    final_carrier_pairs: Array
    injected_carrier_pairs: Array
    recombined_carrier_pairs: Array
    stimulated_carrier_pairs: Array
    stochastic_carrier_pairs: Array
    carrier_balance_residual: Array
    carrier_relative_residual: Array
    initial_photons: Array
    final_photons: Array
    stimulated_photons: Array
    internally_lost_photons: Array
    output_photons: Array
    stochastic_photons: Array
    photon_balance_residual: Array
    photon_relative_residual: Array


class TravelingWaveLaserEvidence(StrictModule):
    """Runtime support, conservation, determinism, status, and fixed resources."""

    finite: Array
    optical_response_accepted: Array
    carrier_within_bounds: Array
    inputs_accepted: Array
    charge_neutrality_error: Array
    carrier_ledger_error: Array
    photon_ledger_error: Array
    maximum_grating_unitarity_error: Array
    grating_unitary: Array
    threshold_modal_residual: Array
    threshold_modal_gap: Array
    threshold_mode_converged: Array
    threshold_mode_isolated: Array
    stochastic: Array
    successful: Array
    status: Array
    section_count: int = eqx.field(static=True)
    step_count: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    retained_result_bytes: int = eqx.field(static=True)
    threshold_map_applications: int = eqx.field(static=True)
    threshold_mode_iteration_count: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    grating_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class TravelingWaveLaserTransientResult(StrictModule):
    """Fixed-shape deterministic or explicit-noise-replay transient."""

    times: Array
    carrier_pair_density: Array
    forward_field: Array
    backward_field: Array
    left_output_power: Array
    right_output_power: Array
    final_state: TravelingWaveSemiconductorLaserState
    ledger: TravelingWaveLaserLedger
    evidence: TravelingWaveLaserEvidence
    noise_realization_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful

    @property
    def status(self) -> Array:
        return self.evidence.status


class TravelingWaveLaserThresholdResult(StrictModule):
    """Fixed-carrier threshold, dominant optical mode, and convergence evidence."""

    threshold_carrier_pair_density: Array
    threshold_injection_current: Array
    threshold_injection_pair_rate: Array
    modal_power_gain: Array
    internal_loss: Array
    mirror_loss: Array
    round_trip_log_power_residual: Array
    threshold_mode_forward: Array
    threshold_mode_backward: Array
    modal_eigenvalue: Array
    modal_residual: Array
    modal_gap: Array
    iteration_count: int = eqx.field(static=True)
    evidence: TravelingWaveLaserEvidence

    @property
    def successful(self) -> Array:
        return self.evidence.successful

    @property
    def status(self) -> Array:
        return self.evidence.status


def prepare_traveling_wave_semiconductor_laser(
    plan: TravelingWaveSemiconductorLaserPlan,
    /,
) -> PreparedTravelingWaveSemiconductorLaser:
    """Preflight every fixed result and live workspace shape."""

    if not isinstance(plan, TravelingWaveSemiconductorLaserPlan):
        raise TypeError("plan must be a TravelingWaveSemiconductorLaserPlan.")
    widths = jnp.diff(plan.z_grid)
    volumes = widths * plan.active_area
    time_step = widths[0] / plan.group_velocity
    times = jnp.arange(plan.step_count + 1, dtype=widths.dtype) * time_step
    sections = widths.size
    grating_norm = jnp.sqrt(
        plan.detuning * plan.detuning + jnp.real(plan.coupling * jnp.conj(plan.coupling))
    )
    grating_argument = grating_norm * widths
    sine_over_norm = widths * jnp.sinc(grating_argument / jnp.pi)
    cosine = jnp.cos(grating_argument)
    grating_step = jnp.stack(
        (
            jnp.stack(
                (
                    cosine + 1j * plan.detuning * sine_over_norm,
                    1j * plan.coupling * sine_over_norm,
                ),
                axis=-1,
            ),
            jnp.stack(
                (
                    1j * jnp.conj(plan.coupling) * sine_over_norm,
                    cosine - 1j * plan.detuning * sine_over_norm,
                ),
                axis=-1,
            ),
        ),
        axis=-2,
    )
    column0_norm = (
        jnp.abs(grating_step[:, 0, 0]) ** 2 + jnp.abs(grating_step[:, 1, 0]) ** 2
    )
    column1_norm = (
        jnp.abs(grating_step[:, 0, 1]) ** 2 + jnp.abs(grating_step[:, 1, 1]) ** 2
    )
    column_overlap = (
        jnp.conj(grating_step[:, 0, 0]) * grating_step[:, 0, 1]
        + jnp.conj(grating_step[:, 1, 0]) * grating_step[:, 1, 1]
    )
    maximum_grating_unitarity_error = jnp.max(
        jnp.maximum(
            jnp.maximum(jnp.abs(column0_norm - 1.0), jnp.abs(column1_norm - 1.0)),
            jnp.abs(column_overlap),
        )
    )
    steps = plan.step_count
    # Three retained state histories, two outputs, response masks, and scan ledgers.
    retained_real_elements = (steps + 1) * sections + 2 * steps + steps * sections
    retained_complex_elements = 2 * (steps + 1) * sections
    retained_bytes = retained_real_elements * 8 + retained_complex_elements * 16
    # Maximum of transient live state and two-vector threshold subspace work.
    workspace_bytes = sections * (12 * 8 + 24 * 16) + 32 * 8
    if retained_bytes + workspace_bytes > plan.maximum_workspace_bytes:
        raise ValueError("Laser transient shapes exceed maximum_workspace_bytes.")
    left_power = jnp.real(
        plan.left_facet_amplitude_reflection
        * jnp.conj(plan.left_facet_amplitude_reflection)
    )
    right_power = jnp.real(
        plan.right_facet_amplitude_reflection
        * jnp.conj(plan.right_facet_amplitude_reflection)
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-traveling-wave-semiconductor-laser",
            "plan": plan.plan_id,
            "section_count": sections,
            "step_count": steps,
            "workspace_bytes": workspace_bytes,
            "retained_result_bytes": retained_bytes,
        }
    )
    return PreparedTravelingWaveSemiconductorLaser(
        plan,
        widths,
        volumes,
        time_step,
        times,
        grating_step,
        maximum_grating_unitarity_error,
        left_power,
        right_power,
        sections,
        steps,
        workspace_bytes,
        retained_bytes,
        plan.grating_id,
        prepared_id,
    )


def _threshold_evidence(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    *,
    finite: Array,
    response_accepted: Array,
    carrier_within_bounds: Array,
    bracketed: Array,
    closed_cavity: Array,
    modal_residual: Array,
    modal_gap: Array,
    mode_converged: Array,
    mode_isolated: Array,
    map_applications: int,
    mode_iteration_count: int,
) -> TravelingWaveLaserEvidence:
    grating_unitary = (
        prepared.maximum_grating_unitarity_error <= prepared.plan.grating_tolerance
    )
    status = (
        jnp.where(finite, 0, int(TravelingWaveLaserStatus.NONFINITE))
        | jnp.where(
            response_accepted,
            0,
            int(TravelingWaveLaserStatus.OPTICAL_RESPONSE_REJECTED),
        )
        | jnp.where(
            carrier_within_bounds,
            0,
            int(TravelingWaveLaserStatus.CARRIER_OUTSIDE_BOUNDS),
        )
        | jnp.where(bracketed, 0, int(TravelingWaveLaserStatus.THRESHOLD_NOT_BRACKETED))
        | jnp.where(closed_cavity, 0, int(TravelingWaveLaserStatus.NO_CLOSED_CAVITY))
        | jnp.where(
            mode_converged,
            0,
            int(TravelingWaveLaserStatus.THRESHOLD_MODE_NOT_CONVERGED),
        )
        | jnp.where(
            mode_isolated,
            0,
            int(TravelingWaveLaserStatus.THRESHOLD_MODE_NOT_ISOLATED),
        )
        | jnp.where(
            grating_unitary,
            0,
            int(TravelingWaveLaserStatus.GRATING_UNITARITY_DEFECT),
        )
    ).astype(jnp.int32)
    successful = status == int(TravelingWaveLaserStatus.SUCCESS)
    evidence_id = canonical_fingerprint(
        {
            "kind": "traveling-wave-laser-threshold-evidence",
            "prepared": prepared.prepared_id,
            "map_applications": map_applications,
            "mode_iteration_count": mode_iteration_count,
        }
    )
    zero = jnp.asarray(0.0)
    return TravelingWaveLaserEvidence(
        finite,
        response_accepted,
        carrier_within_bounds,
        jnp.asarray(True),
        zero,
        zero,
        zero,
        prepared.maximum_grating_unitarity_error,
        grating_unitary,
        modal_residual,
        modal_gap,
        mode_converged,
        mode_isolated,
        jnp.asarray(False),
        successful,
        status,
        prepared.section_count,
        prepared.step_count,
        prepared.workspace_bytes,
        prepared.retained_result_bytes,
        map_applications,
        mode_iteration_count,
        prepared.prepared_id,
        prepared.grating_id,
        evidence_id,
    )


def _complex_norm(value: Array, /) -> Array:
    squared = jnp.real(contract("n,n->", jnp.conj(value), value))
    return jnp.sqrt(jnp.maximum(squared, 0.0))


def _normalize_mode(value: Array, fallback: Array, /) -> Array:
    norm = _complex_norm(value)
    fallback_norm = _complex_norm(fallback)
    epsilon = jnp.finfo(jnp.real(value).dtype).tiny
    safe_norm = jnp.where(norm > epsilon, norm, 1.0)
    safe_fallback_norm = jnp.where(fallback_norm > epsilon, fallback_norm, 1.0)
    return jnp.where(
        norm > epsilon,
        value / safe_norm,
        fallback / safe_fallback_norm,
    )


def _apply_frozen_optical_map(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    amplitude_factor: Array,
    mode: Array,
    /,
) -> Array:
    count = prepared.section_count
    forward = amplitude_factor * mode[:count]
    backward = amplitude_factor * mode[count:]
    grating_forward = (
        prepared.grating_step[:, 0, 0] * forward
        + prepared.grating_step[:, 0, 1] * backward
    )
    grating_backward = (
        prepared.grating_step[:, 1, 0] * forward
        + prepared.grating_step[:, 1, 1] * backward
    )
    next_forward = jnp.concatenate(
        (
            (prepared.plan.left_facet_amplitude_reflection * grating_backward[0])[None],
            grating_forward[:-1],
        )
    )
    next_backward = jnp.concatenate(
        (
            grating_backward[1:],
            (prepared.plan.right_facet_amplitude_reflection * grating_forward[-1])[None],
        )
    )
    return jnp.concatenate((next_forward, next_backward))


def _dominant_frozen_optical_mode(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    density: Array,
    temperature: Array,
    mode_iterations: int,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    count = prepared.section_count
    response = evaluate_semiconductor_optical_response(
        prepared.plan.optical_response,
        jnp.broadcast_to(density, (count,)),
        temperature,
        prepared.plan.carrier_angular_frequency,
    )
    gain = response.modal_power_gain
    phase_per_length = (
        response.propagation_constant_shift
        - 0.5 * prepared.plan.linewidth_enhancement_factor * gain
    )
    amplitude_factor = jnp.exp(
        (0.5 * (gain - response.internal_loss) + 1j * phase_per_length)
        * prepared.section_widths
    )
    dimension = 2 * count
    indices = jnp.arange(dimension, dtype=jnp.real(amplitude_factor).dtype)
    first = jnp.ones((dimension,), dtype=amplitude_factor.dtype)
    second = jnp.exp(2j * jnp.pi * indices / dimension).astype(amplitude_factor.dtype)
    first = _normalize_mode(first, first)
    second = second - first * contract("n,n->", jnp.conj(first), second)
    second = _normalize_mode(second, jnp.roll(first, 1))

    def iteration(_, basis):
        q0, q1 = basis
        image0 = _apply_frozen_optical_map(prepared, amplitude_factor, q0)
        image1 = _apply_frozen_optical_map(prepared, amplitude_factor, q1)
        next0 = _normalize_mode(image0, q0)
        orthogonal1 = image1 - next0 * contract("n,n->", jnp.conj(next0), image1)
        fallback1 = q1 - next0 * contract("n,n->", jnp.conj(next0), q1)
        next1 = _normalize_mode(orthogonal1, fallback1)
        return next0, next1

    q0, q1 = jax.lax.fori_loop(0, mode_iterations, iteration, (first, second))
    image0 = _apply_frozen_optical_map(prepared, amplitude_factor, q0)
    image1 = _apply_frozen_optical_map(prepared, amplitude_factor, q1)
    projected00 = contract("n,n->", jnp.conj(q0), image0)
    projected01 = contract("n,n->", jnp.conj(q0), image1)
    projected10 = contract("n,n->", jnp.conj(q1), image0)
    projected11 = contract("n,n->", jnp.conj(q1), image1)
    discriminant = jnp.sqrt(
        (projected00 - projected11) ** 2 + 4.0 * projected01 * projected10
    )
    eigenvalue_plus = 0.5 * (projected00 + projected11 + discriminant)
    eigenvalue_minus = 0.5 * (projected00 + projected11 - discriminant)
    plus_dominant = jnp.abs(eigenvalue_plus) >= jnp.abs(eigenvalue_minus)
    eigenvalue = jnp.where(plus_dominant, eigenvalue_plus, eigenvalue_minus)
    secondary = jnp.where(plus_dominant, eigenvalue_minus, eigenvalue_plus)

    def projected_coefficients(value, fallback):
        candidate0 = jnp.stack((projected01, value - projected00))
        candidate1 = jnp.stack((value - projected11, projected10))
        use_first = _complex_norm(candidate0) >= _complex_norm(candidate1)
        selected = jnp.where(use_first, candidate0, candidate1)
        return _normalize_mode(selected, fallback)

    coefficients = projected_coefficients(
        eigenvalue,
        jnp.asarray((1.0, 0.0), dtype=eigenvalue.dtype),
    )
    secondary_coefficients = projected_coefficients(
        secondary,
        jnp.asarray((0.0, 1.0), dtype=secondary.dtype),
    )
    mode = _normalize_mode(coefficients[0] * q0 + coefficients[1] * q1, q0)
    secondary_mode = _normalize_mode(
        secondary_coefficients[0] * q0 + secondary_coefficients[1] * q1,
        q1,
    )
    image = _apply_frozen_optical_map(prepared, amplitude_factor, mode)
    secondary_image = _apply_frozen_optical_map(
        prepared, amplitude_factor, secondary_mode
    )
    eigenvalue = contract("n,n->", jnp.conj(mode), image)
    secondary = contract("n,n->", jnp.conj(secondary_mode), secondary_image)
    absolute_residual = _complex_norm(image - eigenvalue * mode)
    secondary_absolute_residual = _complex_norm(
        secondary_image - secondary * secondary_mode
    )
    modulus = jnp.abs(eigenvalue)
    safe_modulus = jnp.maximum(
        modulus,
        jnp.finfo(jnp.real(modulus).dtype).tiny,
    )
    residual = absolute_residual / safe_modulus
    gap = (
        jnp.maximum(
            modulus
            - jnp.abs(secondary)
            - absolute_residual
            - secondary_absolute_residual,
            0.0,
        )
        / safe_modulus
    )
    return (
        modulus,
        eigenvalue,
        mode[:count],
        mode[count:],
        residual,
        gap,
        jnp.all(response.successful),
    )


def _fabry_perot_threshold_mode(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    response,
    /,
) -> tuple[Array, Array, Array, Array]:
    gain = response.modal_power_gain
    phase_per_length = (
        response.propagation_constant_shift
        - 0.5 * prepared.plan.linewidth_enhancement_factor * gain
    )
    amplitude_factor = jnp.exp(
        (0.5 * (gain - response.internal_loss) + 1j * phase_per_length)
        * prepared.section_widths
    )
    forward_factor = amplitude_factor * prepared.grating_step[:, 0, 0]
    backward_factor = amplitude_factor * prepared.grating_step[:, 1, 1]
    cycle_factor = (
        prepared.plan.left_facet_amplitude_reflection
        * prepared.plan.right_facet_amplitude_reflection
        * jnp.prod(forward_factor)
        * jnp.prod(backward_factor)
    )
    eigenvalue = jnp.exp(jnp.log(cycle_factor) / (2.0 * prepared.section_count))
    forward = [jnp.asarray(1.0 + 0.0j, dtype=amplitude_factor.dtype)]
    for index in range(1, prepared.section_count):
        forward.append(forward_factor[index - 1] * forward[-1] / eigenvalue)
    backward = [
        prepared.plan.right_facet_amplitude_reflection
        * forward_factor[-1]
        * forward[-1]
        / eigenvalue
    ]
    for index in range(prepared.section_count - 2, -1, -1):
        backward.append(backward_factor[index + 1] * backward[-1] / eigenvalue)
    backward.reverse()
    mode = _normalize_mode(
        jnp.concatenate((jnp.stack(tuple(forward)), jnp.stack(tuple(backward)))),
        jnp.ones((2 * prepared.section_count,), dtype=amplitude_factor.dtype),
    )
    image = _apply_frozen_optical_map(prepared, amplitude_factor, mode)
    residual = _complex_norm(image - eigenvalue * mode) / jnp.maximum(
        jnp.abs(eigenvalue),
        jnp.finfo(jnp.real(eigenvalue).dtype).tiny,
    )
    return (
        mode[: prepared.section_count],
        mode[prepared.section_count :],
        eigenvalue,
        residual,
    )


def solve_traveling_wave_laser_threshold(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    lattice_temperature: ArrayLike,
    /,
    *,
    carrier_density_bounds: ArrayLike | None = None,
    iteration_count: int = 96,
    mode_iteration_count: int = 128,
    modal_residual_tolerance: float = 1.0e-8,
    minimum_modal_gap: float = 1.0e-5,
) -> TravelingWaveLaserThresholdResult:
    """Solve a fixed-carrier FP or distributed-grating modal threshold.

    Zero coupling uses the exact Fabry-Perot round-trip power equation.
    Distributed coupling uses bounded two-vector subspace iteration on the
    matrix-free frozen optical step and bisects its dominant eigenvalue modulus
    to one. A grating threshold is accepted only for a converged isolated mode.
    """

    if not isinstance(prepared, PreparedTravelingWaveSemiconductorLaser):
        raise TypeError("prepared must be a PreparedTravelingWaveSemiconductorLaser.")
    iterations = int(iteration_count)
    mode_iterations = int(mode_iteration_count)
    residual_tolerance = float(modal_residual_tolerance)
    modal_gap_limit = float(minimum_modal_gap)
    if iterations < 1 or mode_iterations < 2:
        raise ValueError("Threshold and mode iteration counts must be positive.")
    if (
        not np.isfinite(residual_tolerance)
        or residual_tolerance < 0.0
        or not np.isfinite(modal_gap_limit)
        or not 0.0 <= modal_gap_limit <= 1.0
    ):
        raise ValueError(
            "Modal residual tolerance must be nonnegative and gap must lie in [0, 1]."
        )
    if carrier_density_bounds is None:
        bounds = prepared.plan.carrier_density_bounds
    else:
        host = np.asarray(carrier_density_bounds)
        if (
            host.shape != (2,)
            or np.iscomplexobj(host)
            or np.any(~np.isfinite(host))
            or host[0] < float(prepared.plan.carrier_density_bounds[0])
            or host[1] > float(prepared.plan.carrier_density_bounds[1])
            or host[1] <= host[0]
        ):
            raise ValueError("Threshold bounds must lie inside laser carrier bounds.")
        bounds = jnp.asarray(host)
    temperature = jnp.asarray(lattice_temperature)
    if temperature.shape == ():
        temperature = jnp.broadcast_to(temperature, (prepared.section_count,))
    if temperature.shape != (prepared.section_count,):
        raise ValueError("lattice_temperature must be scalar or one value per section.")

    if prepared.plan.has_distributed_grating:
        map_applications = (iterations + 3) * (2 * mode_iterations + 4)
        used_mode_iterations = mode_iterations
        if map_applications > prepared.plan.maximum_threshold_map_applications:
            raise ValueError(
                "Distributed-grating threshold exceeds maximum_threshold_map_applications."
            )

        def mode_at(density):
            return _dominant_frozen_optical_mode(
                prepared, density, temperature, mode_iterations
            )

        lower_mode = mode_at(bounds[0])
        upper_mode = mode_at(bounds[1])
        lower_modulus = lower_mode[0]
        upper_modulus = upper_mode[0]
        bracketed = (
            jnp.isfinite(lower_modulus)
            & jnp.isfinite(upper_modulus)
            & (lower_modulus <= 1.0)
            & (upper_modulus >= 1.0)
        )

        def body(_, state):
            lower, upper = state
            middle = 0.5 * (lower + upper)
            modulus = mode_at(middle)[0]
            return jax.lax.cond(
                modulus >= 1.0,
                lambda: (lower, middle),
                lambda: (middle, upper),
            )

        lower, upper = jax.lax.fori_loop(0, iterations, body, (bounds[0], bounds[1]))
        threshold = 0.5 * (lower + upper)
        (
            modulus,
            eigenvalue,
            mode_forward,
            mode_backward,
            modal_residual,
            modal_gap,
            final_response_accepted,
        ) = mode_at(threshold)
        response = evaluate_semiconductor_optical_response(
            prepared.plan.optical_response,
            jnp.broadcast_to(threshold, (prepared.section_count,)),
            temperature,
            prepared.plan.carrier_angular_frequency,
        )
        response_accepted = (
            lower_mode[6]
            & upper_mode[6]
            & final_response_accepted
            & jnp.all(response.successful)
        )
        round_trip_residual = 2.0 * jnp.log(modulus)
        length = prepared.plan.z_grid[-1] - prepared.plan.z_grid[0]
        mirror_loss = (
            contract(
                "n,n->",
                prepared.section_widths,
                response.modal_power_gain - response.internal_loss,
            )
            / length
        )
        mode_converged = jnp.isfinite(modal_residual) & (
            modal_residual <= residual_tolerance
        )
        mode_isolated = jnp.isfinite(modal_gap) & (modal_gap >= modal_gap_limit)
        closed_cavity = jnp.asarray(True)
    else:
        map_applications = 0
        used_mode_iterations = 0
        reflectivity_product = (
            prepared.left_power_reflectivity * prepared.right_power_reflectivity
        )
        safe_product = jnp.where(reflectivity_product > 0.0, reflectivity_product, 1.0)

        def margin(density):
            response_ = evaluate_semiconductor_optical_response(
                prepared.plan.optical_response,
                jnp.broadcast_to(density, (prepared.section_count,)),
                temperature,
                prepared.plan.carrier_angular_frequency,
            )
            local_net_gain = response_.modal_power_gain - response_.internal_loss
            value = 2.0 * contract(
                "n,n->", prepared.section_widths, local_net_gain
            ) + jnp.log(safe_product)
            return value, response_

        lower_value, lower_response = margin(bounds[0])
        upper_value, upper_response = margin(bounds[1])
        bracketed = (
            jnp.isfinite(lower_value)
            & jnp.isfinite(upper_value)
            & (lower_value <= 0.0)
            & (upper_value >= 0.0)
            & (reflectivity_product > 0.0)
        )

        def body(_, state):
            lower, upper = state
            middle = 0.5 * (lower + upper)
            value, _ = margin(middle)
            return jax.lax.cond(
                value >= 0.0,
                lambda: (lower, middle),
                lambda: (middle, upper),
            )

        lower, upper = jax.lax.fori_loop(0, iterations, body, (bounds[0], bounds[1]))
        threshold = 0.5 * (lower + upper)
        round_trip_residual, response = margin(threshold)
        (
            mode_forward,
            mode_backward,
            eigenvalue,
            modal_residual,
        ) = _fabry_perot_threshold_mode(prepared, response)
        modulus = jnp.abs(eigenvalue)
        modal_gap = jnp.asarray(jnp.inf, dtype=modulus.dtype)
        response_accepted = (
            jnp.all(lower_response.successful)
            & jnp.all(upper_response.successful)
            & jnp.all(response.successful)
        )
        length = prepared.plan.z_grid[-1] - prepared.plan.z_grid[0]
        mirror_loss = -jnp.log(safe_product) / (2.0 * length)
        mode_converged = jnp.isfinite(modal_residual) & (
            modal_residual <= residual_tolerance
        )
        mode_isolated = jnp.asarray(True)
        closed_cavity = reflectivity_product > 0.0

    recombination_rate = (
        prepared.plan.recombination_a * threshold
        + prepared.plan.recombination_b * threshold**2
        + prepared.plan.recombination_c * threshold**3
    )
    active_volume = jnp.sum(prepared.section_volumes)
    threshold_pairs_per_second = active_volume * recombination_rate
    threshold_current = (
        ELEMENTARY_CHARGE_SI
        * threshold_pairs_per_second
        / prepared.plan.injection_efficiency
    )
    finite = (
        jnp.isfinite(threshold)
        & jnp.isfinite(threshold_current)
        & jnp.isfinite(round_trip_residual)
        & jnp.isfinite(modulus)
        & jnp.all(jnp.isfinite(jnp.real(mode_forward)))
        & jnp.all(jnp.isfinite(jnp.imag(mode_forward)))
        & jnp.all(jnp.isfinite(jnp.real(mode_backward)))
        & jnp.all(jnp.isfinite(jnp.imag(mode_backward)))
        & jnp.all(jnp.isfinite(response.modal_power_gain))
    )
    in_bounds = (threshold >= bounds[0]) & (threshold <= bounds[1])
    evidence = _threshold_evidence(
        prepared,
        finite=finite,
        response_accepted=response_accepted,
        carrier_within_bounds=in_bounds,
        bracketed=bracketed,
        closed_cavity=closed_cavity,
        modal_residual=modal_residual,
        modal_gap=modal_gap,
        mode_converged=mode_converged,
        mode_isolated=mode_isolated,
        map_applications=map_applications,
        mode_iteration_count=used_mode_iterations,
    )
    admitted = lambda value: jnp.where(evidence.successful, value, jnp.nan)
    return TravelingWaveLaserThresholdResult(
        admitted(threshold),
        admitted(threshold_current),
        admitted(threshold_pairs_per_second),
        admitted(response.modal_power_gain),
        admitted(response.internal_loss),
        admitted(mirror_loss),
        admitted(round_trip_residual),
        admitted(mode_forward),
        admitted(mode_backward),
        admitted(eigenvalue),
        admitted(modal_residual),
        admitted(modal_gap),
        iterations,
        evidence,
    )


def realize_traveling_wave_laser_noise(
    plan: TravelingWaveLaserNoisePlan,
    prepared: PreparedTravelingWaveSemiconductorLaser,
    key: Array,
    /,
) -> TravelingWaveLaserNoiseRealization:
    """Draw one explicit forcing tape from one explicit JAX key."""

    if not isinstance(plan, TravelingWaveLaserNoisePlan):
        raise TypeError("plan must be a TravelingWaveLaserNoisePlan.")
    if not isinstance(prepared, PreparedTravelingWaveSemiconductorLaser):
        raise TypeError("prepared must be a PreparedTravelingWaveSemiconductorLaser.")
    carrier_key, field_key = jax.random.split(key)
    carrier = plan.carrier_density_standard_deviation * jax.random.normal(
        carrier_key,
        (prepared.step_count, prepared.section_count),
        dtype=prepared.plan.carrier_density_bounds.dtype,
    )
    real = jax.random.normal(
        field_key,
        (prepared.step_count, prepared.section_count, 2, 2),
        dtype=prepared.plan.carrier_density_bounds.dtype,
    )
    field = (
        plan.field_amplitude_standard_deviation
        * (real[..., 0] + 1j * real[..., 1])
        / jnp.sqrt(2.0)
    )
    key_data = np.asarray(jax.random.key_data(key))
    realization_id = canonical_fingerprint(
        {
            "kind": "traveling-wave-laser-noise-realization",
            "noise": plan.noise_id,
            "prepared": prepared.prepared_id,
            "key": array_tree_fingerprint(key_data),
        }
    )
    return TravelingWaveLaserNoiseRealization(
        carrier,
        field,
        plan.noise_id,
        prepared.prepared_id,
        plan.provenance,
        realization_id,
    )


def _drive_arrays(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    inputs: TravelingWaveLaserInput,
    /,
) -> tuple[Array, Array, Array]:
    steps, sections = prepared.step_count, prepared.section_count
    current = inputs.injection_current
    if current.shape == ():
        current = jnp.broadcast_to(current, (steps,))
    if current.shape != (steps,):
        raise ValueError("injection_current must be scalar or have shape (step_count,).")
    temperature = inputs.lattice_temperature
    if temperature.shape == ():
        temperature = jnp.broadcast_to(temperature, (steps, sections))
    elif temperature.shape == (sections,):
        temperature = jnp.broadcast_to(temperature, (steps, sections))
    elif temperature.shape == (steps, 1):
        temperature = jnp.broadcast_to(temperature, (steps, sections))
    if temperature.shape != (steps, sections):
        raise ValueError(
            "lattice_temperature must be scalar, section-shaped, or step-by-section."
        )
    injection_source = (
        prepared.plan.injection_efficiency
        * current[:, None]
        * prepared.plan.injection_weights[None, :]
        / (ELEMENTARY_CHARGE_SI * prepared.section_volumes[None, :])
    )
    accepted = (
        jnp.all(jnp.isfinite(current))
        & jnp.all(current >= 0.0)
        & jnp.all(jnp.isfinite(temperature))
        & jnp.all(temperature > 0.0)
    )
    return injection_source, temperature, accepted


def _exprel(argument: Array, /) -> Array:
    small = jnp.abs(argument) < 1.0e-5
    safe = jnp.where(small, 1.0, argument)
    regular = jnp.expm1(argument) / safe
    series = 1.0 + argument / 2.0 + argument**2 / 6.0 + argument**3 / 24.0
    return jnp.where(small, series, regular)


def _simulate(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    initial_state: TravelingWaveSemiconductorLaserState,
    inputs: TravelingWaveLaserInput,
    noise: TravelingWaveLaserNoiseRealization | None,
    /,
) -> TravelingWaveLaserTransientResult:
    if initial_state.carrier_pair_density.shape != (prepared.section_count,):
        raise ValueError("initial_state does not match the prepared section count.")
    injection_source, temperature, inputs_accepted = _drive_arrays(prepared, inputs)
    if noise is None:
        carrier_noise = jnp.zeros(
            (prepared.step_count, prepared.section_count),
            dtype=initial_state.carrier_pair_density.dtype,
        )
        field_noise = jnp.zeros(
            (prepared.step_count, prepared.section_count, 2),
            dtype=initial_state.forward_field.dtype,
        )
        realization_id = "deterministic-no-noise"
        stochastic = False
    else:
        if not isinstance(noise, TravelingWaveLaserNoiseRealization):
            raise TypeError("noise must be a TravelingWaveLaserNoiseRealization or None.")
        if noise.prepared_id != prepared.prepared_id:
            raise ValueError("Noise realization was prepared for a different laser.")
        expected_carrier = (prepared.step_count, prepared.section_count)
        expected_field = (prepared.step_count, prepared.section_count, 2)
        if (
            noise.carrier_density_increment.shape != expected_carrier
            or noise.field_amplitude_increment.shape != expected_field
        ):
            raise ValueError("Noise realization has incompatible fixed shapes.")
        carrier_noise = noise.carrier_density_increment
        field_noise = noise.field_amplitude_increment
        realization_id = noise.realization_id
        stochastic = True

    plan = prepared.plan
    dt = prepared.time_step
    omega = plan.carrier_angular_frequency
    photon_energy = _REDUCED_PLANCK_CONSTANT * omega
    zero = jnp.asarray(0.0, dtype=initial_state.carrier_pair_density.dtype)
    initial_totals = (zero, zero, zero, zero, zero, zero, zero)
    initial_carry = (
        initial_state.carrier_pair_density,
        initial_state.forward_field,
        initial_state.backward_field,
        initial_totals,
    )

    def step(carry, forcing):
        density, forward, backward, totals = carry
        injection, local_temperature, density_noise, amplitude_noise = forcing
        response = evaluate_semiconductor_optical_response(
            plan.optical_response,
            density,
            local_temperature,
            omega,
        )
        forward_power = jnp.real(forward * jnp.conj(forward))
        backward_power = jnp.real(backward * jnp.conj(backward))
        total_power = forward_power + backward_power
        material_gain = response.modal_power_gain
        effective_gain = jnp.where(
            material_gain > 0.0,
            material_gain / (1.0 + plan.gain_compression * total_power),
            material_gain,
        )
        net_temporal_rate = plan.group_velocity * (
            effective_gain - response.internal_loss
        )
        phase_rate = plan.group_velocity * (
            response.propagation_constant_shift
            - 0.5 * plan.linewidth_enhancement_factor * effective_gain
        )
        factor = jnp.exp(0.5 * net_temporal_rate * dt + 1j * phase_rate * dt)
        local_forward = forward * factor
        local_backward = backward * factor
        grating_forward = (
            prepared.grating_step[:, 0, 0] * local_forward
            + prepared.grating_step[:, 0, 1] * local_backward
        )
        grating_backward = (
            prepared.grating_step[:, 1, 0] * local_forward
            + prepared.grating_step[:, 1, 1] * local_backward
        )
        integrated_power_time = total_power * dt * _exprel(net_temporal_rate * dt)
        stimulated_by_section = (
            prepared.section_widths
            * effective_gain
            * integrated_power_time
            / photon_energy
        )
        internal_loss_by_section = (
            prepared.section_widths
            * response.internal_loss
            * integrated_power_time
            / photon_energy
        )
        recombination_rate = (
            plan.recombination_a * density
            + plan.recombination_b * density**2
            + plan.recombination_c * density**3
        )
        injected_by_section = prepared.section_volumes * injection * dt
        recombined_by_section = prepared.section_volumes * recombination_rate * dt
        deterministic_density = (
            density
            + dt * (injection - recombination_rate)
            - (stimulated_by_section / prepared.section_volumes)
        )
        noisy_density = deterministic_density + density_noise
        stochastic_carrier_by_section = prepared.section_volumes * density_noise
        noisy_forward = grating_forward + amplitude_noise[:, 0]
        noisy_backward = grating_backward + amplitude_noise[:, 1]
        local_photon_inventory = (
            prepared.section_widths
            * (
                jnp.real(grating_forward * jnp.conj(grating_forward))
                + jnp.real(grating_backward * jnp.conj(grating_backward))
            )
            / (photon_energy * plan.group_velocity)
        )
        noisy_photon_inventory = (
            prepared.section_widths
            * (
                jnp.real(noisy_forward * jnp.conj(noisy_forward))
                + jnp.real(noisy_backward * jnp.conj(noisy_backward))
            )
            / (photon_energy * plan.group_velocity)
        )
        stochastic_photons = jnp.sum(noisy_photon_inventory - local_photon_inventory)

        left_output_power = (1.0 - prepared.left_power_reflectivity) * jnp.real(
            noisy_backward[0] * jnp.conj(noisy_backward[0])
        )
        right_output_power = (1.0 - prepared.right_power_reflectivity) * jnp.real(
            noisy_forward[-1] * jnp.conj(noisy_forward[-1])
        )
        output_photons = dt * (left_output_power + right_output_power) / photon_energy
        next_forward = jnp.concatenate(
            (
                (plan.left_facet_amplitude_reflection * noisy_backward[0])[None],
                noisy_forward[:-1],
            )
        )
        next_backward = jnp.concatenate(
            (
                noisy_backward[1:],
                (plan.right_facet_amplitude_reflection * noisy_forward[-1])[None],
            )
        )
        increments = (
            jnp.sum(injected_by_section),
            jnp.sum(recombined_by_section),
            jnp.sum(stimulated_by_section),
            jnp.sum(internal_loss_by_section),
            output_photons,
            jnp.sum(stochastic_carrier_by_section),
            stochastic_photons,
        )
        next_totals = tuple(a + b for a, b in zip(totals, increments, strict=True))
        output = (
            noisy_density,
            next_forward,
            next_backward,
            left_output_power,
            right_output_power,
            jnp.all(response.successful),
        )
        return (noisy_density, next_forward, next_backward, next_totals), output

    final_carry, outputs = checkpointed_scan(
        step,
        initial_carry,
        (injection_source, temperature, carrier_noise, field_noise),
        length=prepared.step_count,
        mode="full",
    )
    final_density, final_forward, final_backward, totals = final_carry
    (
        density_history,
        forward_history,
        backward_history,
        left_output,
        right_output,
        response_success,
    ) = outputs
    density_trajectory = jnp.concatenate(
        (initial_state.carrier_pair_density[None, :], density_history), axis=0
    )
    forward_trajectory = jnp.concatenate(
        (initial_state.forward_field[None, :], forward_history), axis=0
    )
    backward_trajectory = jnp.concatenate(
        (initial_state.backward_field[None, :], backward_history), axis=0
    )
    (
        injected,
        recombined,
        stimulated,
        internal_loss,
        output_photons,
        stochastic_carriers,
        stochastic_photons,
    ) = totals
    initial_carriers = contract(
        "n,n->", prepared.section_volumes, initial_state.carrier_pair_density
    )
    final_carriers = contract("n,n->", prepared.section_volumes, final_density)
    initial_photons = contract(
        "n,n->",
        prepared.section_widths,
        jnp.real(initial_state.forward_field * jnp.conj(initial_state.forward_field))
        + jnp.real(initial_state.backward_field * jnp.conj(initial_state.backward_field)),
    ) / (photon_energy * plan.group_velocity)
    final_photons = contract(
        "n,n->",
        prepared.section_widths,
        jnp.real(final_forward * jnp.conj(final_forward))
        + jnp.real(final_backward * jnp.conj(final_backward)),
    ) / (photon_energy * plan.group_velocity)
    carrier_residual = (
        final_carriers
        - initial_carriers
        - injected
        + recombined
        + stimulated
        - stochastic_carriers
    )
    photon_residual = (
        final_photons
        - initial_photons
        - stimulated
        + internal_loss
        + output_photons
        - stochastic_photons
    )
    carrier_scale = jnp.maximum(
        jnp.max(
            jnp.abs(
                jnp.stack(
                    (
                        initial_carriers,
                        final_carriers,
                        injected,
                        recombined,
                        stimulated,
                        stochastic_carriers,
                    )
                )
            )
        ),
        1.0,
    )
    photon_scale = jnp.maximum(
        jnp.max(
            jnp.abs(
                jnp.stack(
                    (
                        initial_photons,
                        final_photons,
                        stimulated,
                        internal_loss,
                        output_photons,
                        stochastic_photons,
                    )
                )
            )
        ),
        1.0,
    )
    carrier_relative = jnp.abs(carrier_residual) / carrier_scale
    photon_relative = jnp.abs(photon_residual) / photon_scale
    ledger = TravelingWaveLaserLedger(
        initial_carriers,
        final_carriers,
        injected,
        recombined,
        stimulated,
        stochastic_carriers,
        carrier_residual,
        carrier_relative,
        initial_photons,
        final_photons,
        stimulated,
        internal_loss,
        output_photons,
        stochastic_photons,
        photon_residual,
        photon_relative,
    )
    finite = jnp.all(
        jnp.stack(
            tuple(
                jnp.all(jnp.isfinite(value))
                for value in (
                    density_trajectory,
                    jnp.real(forward_trajectory),
                    jnp.imag(forward_trajectory),
                    jnp.real(backward_trajectory),
                    jnp.imag(backward_trajectory),
                    left_output,
                    right_output,
                )
            )
        )
    )
    response_accepted = jnp.all(response_success)
    carrier_bounds = jnp.all(
        (density_trajectory >= plan.carrier_density_bounds[0])
        & (density_trajectory <= plan.carrier_density_bounds[1])
    )
    ledger_closed = (carrier_relative <= plan.ledger_tolerance) & (
        photon_relative <= plan.ledger_tolerance
    )
    grating_unitary = prepared.maximum_grating_unitarity_error <= plan.grating_tolerance
    status = (
        jnp.where(finite, 0, int(TravelingWaveLaserStatus.NONFINITE))
        | jnp.where(
            response_accepted,
            0,
            int(TravelingWaveLaserStatus.OPTICAL_RESPONSE_REJECTED),
        )
        | jnp.where(
            carrier_bounds,
            0,
            int(TravelingWaveLaserStatus.CARRIER_OUTSIDE_BOUNDS),
        )
        | jnp.where(inputs_accepted, 0, int(TravelingWaveLaserStatus.INPUT_REJECTED))
        | jnp.where(ledger_closed, 0, int(TravelingWaveLaserStatus.LEDGER_DEFECT))
        | jnp.where(
            grating_unitary,
            0,
            int(TravelingWaveLaserStatus.GRATING_UNITARITY_DEFECT),
        )
    ).astype(jnp.int32)
    successful = status == int(TravelingWaveLaserStatus.SUCCESS)
    evidence_id = canonical_fingerprint(
        {
            "kind": "traveling-wave-laser-transient-evidence",
            "prepared": prepared.prepared_id,
            "noise_realization": realization_id,
        }
    )
    evidence = TravelingWaveLaserEvidence(
        finite,
        response_accepted,
        carrier_bounds,
        inputs_accepted,
        jnp.asarray(0.0, dtype=density_trajectory.dtype),
        carrier_relative,
        photon_relative,
        prepared.maximum_grating_unitarity_error,
        grating_unitary,
        jnp.asarray(0.0, dtype=density_trajectory.dtype),
        jnp.asarray(jnp.inf, dtype=density_trajectory.dtype),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(stochastic),
        successful,
        status,
        prepared.section_count,
        prepared.step_count,
        prepared.workspace_bytes,
        prepared.retained_result_bytes,
        0,
        0,
        prepared.prepared_id,
        prepared.grating_id,
        evidence_id,
    )
    final_state = TravelingWaveSemiconductorLaserState(
        final_density, final_forward, final_backward
    )
    return TravelingWaveLaserTransientResult(
        prepared.times,
        density_trajectory,
        forward_trajectory,
        backward_trajectory,
        left_output,
        right_output,
        final_state,
        ledger,
        evidence,
        realization_id,
    )


def simulate_traveling_wave_laser(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    initial_state: TravelingWaveSemiconductorLaserState,
    inputs: TravelingWaveLaserInput,
    /,
    *,
    noise: TravelingWaveLaserNoiseRealization | None = None,
) -> TravelingWaveLaserTransientResult:
    """Run deterministically, or replay an already-realized explicit noise tape.

    This API never creates a random key. New stochastic forcing must first be
    created with :func:`realize_traveling_wave_laser_noise`.
    """

    if not isinstance(prepared, PreparedTravelingWaveSemiconductorLaser):
        raise TypeError("prepared must be a PreparedTravelingWaveSemiconductorLaser.")
    if not isinstance(initial_state, TravelingWaveSemiconductorLaserState):
        raise TypeError("initial_state must be a TravelingWaveSemiconductorLaserState.")
    if not isinstance(inputs, TravelingWaveLaserInput):
        raise TypeError("inputs must be a TravelingWaveLaserInput.")
    return _simulate(prepared, initial_state, inputs, noise)


def simulate_stochastic_traveling_wave_laser(
    prepared: PreparedTravelingWaveSemiconductorLaser,
    initial_state: TravelingWaveSemiconductorLaserState,
    inputs: TravelingWaveLaserInput,
    noise: TravelingWaveLaserNoiseRealization,
    /,
) -> TravelingWaveLaserTransientResult:
    """Replay one explicit stochastic forcing tape; no key is accepted here."""

    if not isinstance(noise, TravelingWaveLaserNoiseRealization):
        raise TypeError("noise must be a TravelingWaveLaserNoiseRealization.")
    return simulate_traveling_wave_laser(prepared, initial_state, inputs, noise=noise)


__all__ = [
    "PreparedTravelingWaveSemiconductorLaser",
    "TravelingWaveLaserEvidence",
    "TravelingWaveLaserInput",
    "TravelingWaveLaserLedger",
    "TravelingWaveLaserNoisePlan",
    "TravelingWaveLaserNoiseRealization",
    "TravelingWaveLaserStatus",
    "TravelingWaveLaserThresholdResult",
    "TravelingWaveLaserTransientResult",
    "TravelingWaveSemiconductorLaserPlan",
    "TravelingWaveSemiconductorLaserState",
    "prepare_traveling_wave_semiconductor_laser",
    "realize_traveling_wave_laser_noise",
    "simulate_stochastic_traveling_wave_laser",
    "simulate_traveling_wave_laser",
    "solve_traveling_wave_laser_threshold",
]
