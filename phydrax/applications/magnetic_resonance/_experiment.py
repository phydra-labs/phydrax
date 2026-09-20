#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed pulses, exact density evolution, FID, FFT, and instrument transform."""

from __future__ import annotations

import math
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...control._parameterization import PiecewiseConstantControlParameterization
from ...dynamics._grid import TimeGrid
from ...linalg import HermitianSpectrum, MaterializationPolicy
from ...solver._local_hamiltonian import (
    FixedGridLocalHamiltonian,
    materialize_local_hamiltonian,
)
from ...solver._quantum_control import (
    assemble_fixed_grid_local_hamiltonian,
    LinearQuantumControlTransfer,
    QuantumControlLine,
    QuantumControlSchedule,
    QuantumControlScheduleResult,
    sample_quantum_control_schedule,
)
from ._spin_system import PreparedMagneticResonanceSystem, spin_operators


class FixedPulseSequence(StrictModule):
    """Piecewise-constant laboratory magnetic field on fixed intervals."""

    time_grid_s: Array
    field_t: Array
    sequence_id: str = eqx.field(static=True)

    def __init__(
        self,
        time_grid_s: ArrayLike,
        field_t: ArrayLike,
        /,
        *,
        sequence_id: str = "fixed-pulse-sequence",
    ):
        times = np.asarray(time_grid_s, dtype=np.float64)
        field = np.asarray(field_t, dtype=np.float64)
        if times.ndim != 1 or times.size < 2 or np.any(~np.isfinite(times)):
            raise ValueError(
                "time_grid_s must be a finite rank-one array of length >= 2."
            )
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("time_grid_s must be strictly increasing.")
        expected = (times.size - 1, 3)
        if field.shape != expected or np.any(~np.isfinite(field)):
            raise ValueError(f"field_t must be finite with shape {expected}.")
        identifier = str(sequence_id)
        if not identifier:
            raise ValueError("sequence_id must be nonempty.")
        self.time_grid_s = jnp.asarray(times)
        self.field_t = jnp.asarray(field)
        self.sequence_id = identifier


class PreparedPulseSequence(StrictModule):
    sequence: FixedPulseSequence
    control_schedule: QuantumControlSchedule
    sampled_controls: QuantumControlScheduleResult
    fixed_hamiltonian: FixedGridLocalHamiltonian


def prepare_pulse_sequence(
    prepared: PreparedMagneticResonanceSystem,
    sequence: FixedPulseSequence,
    /,
) -> PreparedPulseSequence:
    """Compile lab-field samples through the canonical quantum-control schedule."""

    if not isinstance(prepared, PreparedMagneticResonanceSystem):
        raise TypeError("prepared must be a PreparedMagneticResonanceSystem.")
    if not isinstance(sequence, FixedPulseSequence):
        raise TypeError("sequence must be a FixedPulseSequence.")
    grid = TimeGrid(sequence.time_grid_s, time_id=f"{sequence.sequence_id}:time")
    parameterization = PiecewiseConstantControlParameterization(
        grid,
        (),
        parameterization_id=f"{sequence.sequence_id}:piecewise-constant",
    )
    lines = tuple(
        QuantumControlLine(
            parameterization,
            sequence.field_t[:, axis],
            support_start=sequence.time_grid_s[0],
            support_stop=sequence.time_grid_s[-1],
            line_id=f"{sequence.sequence_id}:lab-axis:{axis}",
        )
        for axis in range(3)
    )
    transfer = np.zeros((3, len(prepared.drive_terms)), dtype=np.float64)
    for site_index in range(len(prepared.system.sites)):
        for axis in range(3):
            transfer[axis, 3 * site_index + axis] = 1.0
    schedule = QuantumControlSchedule(
        lines,
        LinearQuantumControlTransfer(
            transfer,
            transfer_id=f"{sequence.sequence_id}:lab-field-transfer",
        ),
        schedule_id=f"{sequence.sequence_id}:quantum-control",
    )
    sampled = sample_quantum_control_schedule(schedule, sequence.time_grid_s)
    fixed = assemble_fixed_grid_local_hamiltonian(
        prepared.hamiltonian,
        prepared.drive_terms,
        sampled,
        hbar=1.0,
    )
    return PreparedPulseSequence(sequence, schedule, sampled, fixed)


class ExactEvolutionEvidence(StrictModule):
    trace_residuals: Array
    hermiticity_residuals: Array
    minimum_eigenvalues: Array
    unitarity_residuals: Array
    finite: Array
    valid: Array


class ExactDensityEvolutionResult(StrictModule):
    time_grid_s: Array
    density_matrices: Array
    evidence: ExactEvolutionEvidence


def _validated_density(
    prepared: PreparedMagneticResonanceSystem,
    density_matrix: ArrayLike,
    /,
) -> Array:
    host = np.asarray(density_matrix, dtype=np.complex128)
    dimension = prepared.layout.dimension
    if host.shape != (dimension, dimension) or np.any(~np.isfinite(host)):
        raise ValueError(
            f"density_matrix must be finite with shape ({dimension}, {dimension})."
        )
    hermiticity = float(np.max(np.abs(host - np.conj(host.T))))
    trace = complex(np.trace(host))
    minimum = float(np.min(np.linalg.eigvalsh(0.5 * (host + np.conj(host.T)))))
    if hermiticity > 1.0e-9 or abs(trace - 1.0) > 1.0e-9 or minimum < -1.0e-9:
        raise ValueError(
            "density_matrix must be Hermitian, positive semidefinite, and unit trace."
        )
    return jnp.asarray(host, dtype=prepared.dense_hamiltonian_rad_s.dtype)


def _density_evidence(densities: Array, unitarity: Array, /) -> ExactEvolutionEvidence:
    trace_residuals = jnp.abs(jnp.trace(densities, axis1=-2, axis2=-1) - 1.0)
    hermiticity = jnp.max(
        jnp.abs(densities - jnp.conj(jnp.swapaxes(densities, -1, -2))), axis=(-2, -1)
    )
    minimum = HermitianSpectrum(densities, tolerance=1.0e-8).minimum_eigenvalue
    finite = jnp.all(jnp.isfinite(densities)) & jnp.all(jnp.isfinite(unitarity))
    valid = (
        finite
        & jnp.all(trace_residuals <= 1.0e-8)
        & jnp.all(hermiticity <= 1.0e-8)
        & jnp.all(minimum >= -1.0e-8)
        & jnp.all(unitarity <= 1.0e-8)
    )
    return ExactEvolutionEvidence(
        trace_residuals,
        hermiticity,
        minimum,
        unitarity,
        finite,
        valid,
    )


def evolve_density_exact(
    prepared: PreparedMagneticResonanceSystem,
    pulse_sequence: PreparedPulseSequence,
    initial_density: ArrayLike,
    /,
) -> ExactDensityEvolutionResult:
    """Apply an exact dense exponential for every fixed total-Hamiltonian interval."""

    if not isinstance(prepared, PreparedMagneticResonanceSystem):
        raise TypeError("prepared must be a PreparedMagneticResonanceSystem.")
    if not isinstance(pulse_sequence, PreparedPulseSequence):
        raise TypeError("pulse_sequence must be a PreparedPulseSequence.")
    if (
        pulse_sequence.fixed_hamiltonian.hamiltonian.layout.layout_id
        != prepared.layout.layout_id
    ):
        raise ValueError("pulse_sequence and prepared system use different layouts.")
    density = _validated_density(prepared, initial_density)
    policy = MaterializationPolicy(
        max_entries=prepared.system.resource_policy.maximum_density_elements,
        max_bytes=16 * prepared.system.resource_policy.maximum_density_elements,
    )
    densities = [density]
    unitarity = []
    identity = jnp.eye(prepared.layout.dimension, dtype=density.dtype)
    intervals = np.diff(np.asarray(pulse_sequence.sequence.time_grid_s))
    for interval, coefficients in zip(
        intervals,
        pulse_sequence.fixed_hamiltonian.coefficients,
        strict=True,
    ):
        generator = materialize_local_hamiltonian(
            pulse_sequence.fixed_hamiltonian.hamiltonian,
            coefficients,
            policy=policy,
        )
        unitary = jsp.linalg.expm(-1j * float(interval) * generator)
        density = unitary @ density @ jnp.conj(unitary.T)
        densities.append(density)
        unitarity.append(jnp.max(jnp.abs(unitary @ jnp.conj(unitary.T) - identity)))
    stacked = jnp.stack(tuple(densities))
    residuals = jnp.stack(tuple(unitarity))
    return ExactDensityEvolutionResult(
        pulse_sequence.sequence.time_grid_s,
        stacked,
        _density_evidence(stacked, residuals),
    )


class AcquisitionPlan(StrictModule):
    """Uniform exact FID acquisition using a fixed lowering-operator receiver."""

    dwell_time_s: float = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    receiver_phase_rad: float = eqx.field(static=True)
    site_weights: tuple[complex, ...] = eqx.field(static=True)

    def __init__(
        self,
        dwell_time_s: float,
        sample_count: int,
        /,
        *,
        receiver_phase_rad: float = 0.0,
        site_weights: Sequence[complex] = (),
    ):
        dwell = float(dwell_time_s)
        count = int(sample_count)
        phase = float(receiver_phase_rad)
        weights = tuple(complex(value) for value in site_weights)
        if not math.isfinite(dwell) or dwell <= 0.0:
            raise ValueError("dwell_time_s must be finite and positive.")
        if count < 2:
            raise ValueError("sample_count must be at least two.")
        if not math.isfinite(phase) or any(
            not math.isfinite(value.real) or not math.isfinite(value.imag)
            for value in weights
        ):
            raise ValueError("Receiver phase and weights must be finite.")
        self.dwell_time_s = dwell
        self.sample_count = count
        self.receiver_phase_rad = phase
        self.site_weights = weights


class FIDResult(StrictModule):
    times_s: Array
    signal: Array
    trace_residuals: Array
    hermiticity_residuals: Array
    step_unitarity_residual: Array
    finite: Array
    valid: Array


class SpectrumResult(StrictModule):
    frequency_hz: Array
    angular_frequency_rad_s: Array
    amplitude: Array
    finite: Array


class InstrumentPlan(StrictModule):
    gain: float = eqx.field(static=True)
    receiver_phase_rad: float = eqx.field(static=True)
    frequency_offset_hz: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        gain: float = 1.0,
        receiver_phase_rad: float = 0.0,
        frequency_offset_hz: float = 0.0,
    ):
        gain_ = float(gain)
        phase = float(receiver_phase_rad)
        offset = float(frequency_offset_hz)
        if (
            not all(math.isfinite(value) for value in (gain_, phase, offset))
            or gain_ <= 0.0
        ):
            raise ValueError("Instrument gain must be positive and all settings finite.")
        self.gain = gain_
        self.receiver_phase_rad = phase
        self.frequency_offset_hz = offset


class InstrumentResult(StrictModule):
    bare_fid: FIDResult
    detected_fid: Array
    spectrum: SpectrumResult
    finite: Array
    valid: Array


def _embed_single_site(
    local: Array,
    site_index: int,
    dimensions: tuple[int, ...],
    /,
) -> Array:
    result = jnp.asarray([[1.0]], dtype=local.dtype)
    for index, dimension in enumerate(dimensions):
        factor = local if index == site_index else jnp.eye(dimension, dtype=local.dtype)
        result = jnp.kron(result, factor)
    return result


def receiver_operator(
    prepared: PreparedMagneticResonanceSystem,
    site_weights: Sequence[complex] = (),
    /,
) -> Array:
    """Materialize the convention-fixed weighted ``Ix - i Iy`` receiver."""

    if not isinstance(prepared, PreparedMagneticResonanceSystem):
        raise TypeError("prepared must be a PreparedMagneticResonanceSystem.")
    weights = tuple(complex(value) for value in site_weights)
    if not weights:
        weights = (1.0 + 0.0j,) * len(prepared.system.sites)
    if len(weights) != len(prepared.system.sites):
        raise ValueError("site_weights must have one entry per spin site.")
    operator = jnp.zeros_like(prepared.dense_hamiltonian_rad_s)
    for index, (site, weight) in enumerate(
        zip(prepared.system.sites, weights, strict=True)
    ):
        spin = spin_operators(site.isotope.spin)
        lowering = spin.x - 1j * spin.y
        operator = operator + weight * _embed_single_site(
            lowering,
            index,
            prepared.layout.local_dimensions,
        )
    return operator


def acquire_fid(
    prepared: PreparedMagneticResonanceSystem,
    initial_density: ArrayLike,
    plan: AcquisitionPlan,
    /,
) -> FIDResult:
    """Acquire a uniform exact FID under the complete static Hamiltonian."""

    if not isinstance(plan, AcquisitionPlan):
        raise TypeError("plan must be an AcquisitionPlan.")
    density = _validated_density(prepared, initial_density)
    receiver = receiver_operator(prepared, plan.site_weights)
    receiver = jnp.exp(-1j * plan.receiver_phase_rad) * receiver
    unitary = jsp.linalg.expm(-1j * plan.dwell_time_s * prepared.dense_hamiltonian_rad_s)
    identity = jnp.eye(prepared.layout.dimension, dtype=density.dtype)
    unitary_residual = jnp.max(jnp.abs(unitary @ jnp.conj(unitary.T) - identity))
    signal = []
    traces = []
    hermiticity = []
    for _ in range(plan.sample_count):
        signal.append(jnp.trace(density @ receiver))
        traces.append(jnp.abs(jnp.trace(density) - 1.0))
        hermiticity.append(jnp.max(jnp.abs(density - jnp.conj(density.T))))
        density = unitary @ density @ jnp.conj(unitary.T)
    signal_array = jnp.stack(tuple(signal))
    traces_array = jnp.stack(tuple(traces))
    hermiticity_array = jnp.stack(tuple(hermiticity))
    finite = (
        jnp.all(jnp.isfinite(signal_array))
        & jnp.all(jnp.isfinite(traces_array))
        & jnp.all(jnp.isfinite(hermiticity_array))
        & jnp.isfinite(unitary_residual)
    )
    valid = (
        finite
        & jnp.all(traces_array <= 1.0e-8)
        & jnp.all(hermiticity_array <= 1.0e-8)
        & (unitary_residual <= 1.0e-8)
    )
    return FIDResult(
        plan.dwell_time_s * jnp.arange(plan.sample_count),
        signal_array,
        traces_array,
        hermiticity_array,
        unitary_residual,
        finite,
        valid,
    )


def fid_spectrum(fid: FIDResult, /) -> SpectrumResult:
    """Apply the convention-fixed unwindowed complex FFT to a uniform FID."""

    if not isinstance(fid, FIDResult):
        raise TypeError("fid must be an FIDResult.")
    dwell = float(np.asarray(fid.times_s[1] - fid.times_s[0]))
    frequency = jnp.fft.fftshift(jnp.fft.fftfreq(fid.signal.size, d=dwell))
    amplitude = dwell * jnp.fft.fftshift(jnp.fft.fft(fid.signal))
    finite = jnp.all(jnp.isfinite(frequency)) & jnp.all(jnp.isfinite(amplitude))
    return SpectrumResult(frequency, 2.0 * math.pi * frequency, amplitude, finite)


def apply_instrument(
    fid: FIDResult,
    plan: InstrumentPlan,
    /,
) -> InstrumentResult:
    """Apply explicit gain/phase/reference offset without altering the bare FID."""

    if not isinstance(fid, FIDResult):
        raise TypeError("fid must be an FIDResult.")
    if not isinstance(plan, InstrumentPlan):
        raise TypeError("plan must be an InstrumentPlan.")
    detected = plan.gain * jnp.exp(-1j * plan.receiver_phase_rad) * fid.signal
    detected_fid = FIDResult(
        fid.times_s,
        detected,
        fid.trace_residuals,
        fid.hermiticity_residuals,
        fid.step_unitarity_residual,
        fid.finite & jnp.all(jnp.isfinite(detected)),
        fid.valid & jnp.all(jnp.isfinite(detected)),
    )
    base = fid_spectrum(detected_fid)
    spectrum = SpectrumResult(
        base.frequency_hz - plan.frequency_offset_hz,
        base.angular_frequency_rad_s - 2.0 * math.pi * plan.frequency_offset_hz,
        base.amplitude,
        base.finite,
    )
    finite = detected_fid.finite & spectrum.finite
    return InstrumentResult(fid, detected, spectrum, finite, fid.valid & finite)


__all__ = [
    "AcquisitionPlan",
    "ExactDensityEvolutionResult",
    "ExactEvolutionEvidence",
    "FIDResult",
    "FixedPulseSequence",
    "InstrumentPlan",
    "InstrumentResult",
    "PreparedPulseSequence",
    "SpectrumResult",
    "acquire_fid",
    "apply_instrument",
    "evolve_density_exact",
    "fid_spectrum",
    "prepare_pulse_sequence",
    "receiver_operator",
]
