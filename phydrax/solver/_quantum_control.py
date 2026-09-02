#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.core as jax_core
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..control._parameterization import AbstractControlParameterization
from ._local_hamiltonian import (
    FixedGridLocalHamiltonian,
    LocalHamiltonian,
    LocalHamiltonianTerm,
)


def _finite_scalar(value: ArrayLike, name: str, /) -> Array:
    result = jnp.asarray(value)
    if result.shape != () or jnp.issubdtype(result.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be one real scalar.")
    invalid = ~jnp.isfinite(result)
    if isinstance(result, jax_core.Tracer):
        return eqx.error_if(result, invalid, f"{name} must be finite.")
    if bool(invalid):
        raise ValueError(f"{name} must be finite.")
    return result


class QuantumCarrier(StrictModule):
    """Angular carrier phase and whole-waveform delay for one real control line."""

    angular_rate: Array
    phase: Array
    delay: Array

    def __init__(
        self,
        *,
        angular_rate: ArrayLike = 0.0,
        phase: ArrayLike = 0.0,
        delay: ArrayLike = 0.0,
    ):
        self.angular_rate = _finite_scalar(angular_rate, "angular_rate")
        self.phase = _finite_scalar(phase, "phase")
        self.delay = _finite_scalar(delay, "delay")


class QuantumControlLine(StrictModule):
    """One scalar I/Q envelope and carrier with explicit compact support."""

    parameterization: AbstractControlParameterization
    in_phase_coefficients: Array
    quadrature_coefficients: Array | None
    carrier: QuantumCarrier
    support_start: Array
    support_stop: Array
    finite: Array
    valid: Array
    line_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameterization: AbstractControlParameterization,
        in_phase_coefficients: ArrayLike,
        /,
        *,
        quadrature_coefficients: ArrayLike | None = None,
        carrier: QuantumCarrier | None = None,
        support_start: ArrayLike,
        support_stop: ArrayLike,
        line_id: str | None = None,
    ):
        if not isinstance(parameterization, AbstractControlParameterization):
            raise TypeError(
                "parameterization must be an AbstractControlParameterization."
            )
        if parameterization.control_shape != ():
            raise ValueError("Quantum control lines require scalar control_shape=().")
        in_phase = jnp.asarray(in_phase_coefficients)
        if in_phase.shape != parameterization.parameter_shape:
            raise ValueError(
                "in_phase_coefficients must match parameterization.parameter_shape."
            )
        if jnp.issubdtype(in_phase.dtype, jnp.complexfloating):
            raise TypeError("Control coefficients must be real.")
        if quadrature_coefficients is None:
            quadrature = None
        else:
            quadrature = jnp.asarray(quadrature_coefficients)
            if quadrature.shape != parameterization.parameter_shape:
                raise ValueError(
                    "quadrature_coefficients must match parameterization.parameter_shape."
                )
            if jnp.issubdtype(quadrature.dtype, jnp.complexfloating):
                raise TypeError("Control coefficients must be real.")
        carrier_ = QuantumCarrier() if carrier is None else carrier
        if not isinstance(carrier_, QuantumCarrier):
            raise TypeError("carrier must be a QuantumCarrier or None.")
        start = _finite_scalar(support_start, "support_start")
        stop = _finite_scalar(support_stop, "support_stop")
        finite = (
            jnp.all(jnp.isfinite(in_phase)) & jnp.isfinite(start) & jnp.isfinite(stop)
        )
        if quadrature is not None:
            finite = finite & jnp.all(jnp.isfinite(quadrature))
        valid = finite & (stop > start)
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "quantum-control-line",
                    "parameterization": parameterization.parameterization_id,
                    "parameter_shape": list(parameterization.parameter_shape),
                    "quadrature": quadrature is not None,
                    "dtype": str(in_phase.dtype),
                }
            )
            if line_id is None
            else str(line_id)
        )
        if not identifier:
            raise ValueError("line_id must be nonempty.")
        self.parameterization = parameterization
        self.in_phase_coefficients = in_phase
        self.quadrature_coefficients = quadrature
        self.carrier = carrier_
        self.support_start = start
        self.support_stop = stop
        self.finite = finite
        self.valid = valid
        self.line_id = identifier


class LinearQuantumControlTransfer(StrictModule):
    """Real line-to-Hamiltonian-term actuation and crosstalk matrix."""

    matrix: Array
    finite: Array
    valid: Array
    line_count: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        matrix: ArrayLike,
        /,
        *,
        transfer_id: str | None = None,
    ):
        value = jnp.asarray(matrix)
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
            raise ValueError("matrix must be one nonempty line-by-term matrix.")
        if jnp.issubdtype(value.dtype, jnp.complexfloating):
            raise TypeError("Control transfer matrices must be real.")
        if not jnp.issubdtype(value.dtype, jnp.inexact):
            value = value.astype(float)
        finite = jnp.all(jnp.isfinite(value))
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "linear-quantum-control-transfer",
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                }
            )
            if transfer_id is None
            else str(transfer_id)
        )
        if not identifier:
            raise ValueError("transfer_id must be nonempty.")
        self.matrix = value
        self.finite = finite
        self.valid = finite
        self.line_count = int(value.shape[0])
        self.term_count = int(value.shape[1])
        self.transfer_id = identifier


class QuantumControlSchedule(StrictModule):
    """Fixed control lines and their transfer into ordered drive terms."""

    lines: tuple[QuantumControlLine, ...]
    transfer: LinearQuantumControlTransfer
    finite: Array
    valid: Array
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        lines: Sequence[QuantumControlLine],
        transfer: LinearQuantumControlTransfer,
        /,
        *,
        schedule_id: str | None = None,
    ):
        selected = tuple(lines)
        if not selected or not all(
            isinstance(line, QuantumControlLine) for line in selected
        ):
            raise ValueError("lines must contain at least one QuantumControlLine.")
        if not isinstance(transfer, LinearQuantumControlTransfer):
            raise TypeError("transfer must be a LinearQuantumControlTransfer.")
        if len(selected) != transfer.line_count:
            raise ValueError("Control line count must match the transfer matrix.")
        finite = (
            jnp.all(jnp.stack(tuple(line.finite for line in selected))) & transfer.finite
        )
        valid = (
            jnp.all(jnp.stack(tuple(line.valid for line in selected))) & transfer.valid
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "quantum-control-schedule",
                    "lines": [line.line_id for line in selected],
                    "transfer": transfer.transfer_id,
                }
            )
            if schedule_id is None
            else str(schedule_id)
        )
        if not identifier:
            raise ValueError("schedule_id must be nonempty.")
        self.lines = selected
        self.transfer = transfer
        self.finite = finite
        self.valid = valid
        self.schedule_id = identifier


class QuantumControlScheduleDiagnostics(StrictModule):
    """Sampling and transfer evidence for one fixed time grid."""

    positive_intervals: Array
    lines_valid: Array
    transfer_valid: Array
    finite: Array
    valid: Array


class QuantumControlScheduleResult(StrictModule):
    """Sampled real line values and ordered Hamiltonian-term coefficients."""

    time_grid: Array
    sample_times: Array
    line_values: Array
    term_coefficients: Array
    diagnostics: QuantumControlScheduleDiagnostics
    schedule_id: str = eqx.field(static=True)


def sample_quantum_control_schedule(
    schedule: QuantumControlSchedule,
    time_grid: ArrayLike,
    /,
) -> QuantumControlScheduleResult:
    """Sample every control line at interval midpoints and apply crosstalk."""

    if not isinstance(schedule, QuantumControlSchedule):
        raise TypeError("schedule must be a QuantumControlSchedule.")
    times = jnp.asarray(time_grid)
    if times.ndim != 1 or times.shape[0] < 2:
        raise ValueError("time_grid must have shape (interval_count + 1,).")
    if jnp.issubdtype(times.dtype, jnp.complexfloating):
        raise TypeError("time_grid must be real.")
    times = times.astype(jnp.result_type(times, float))
    intervals = jnp.diff(times)
    positive = jnp.all(intervals > 0.0)
    sample_times = 0.5 * (times[:-1] + times[1:])
    line_values: list[Array] = []
    for line in schedule.lines:
        shifted = sample_times - line.carrier.delay
        query = jnp.clip(shifted, line.support_start, line.support_stop)
        in_phase = line.parameterization.sample(line.in_phase_coefficients, query)
        quadrature = (
            jnp.zeros_like(in_phase)
            if line.quadrature_coefficients is None
            else line.parameterization.sample(line.quadrature_coefficients, query)
        )
        if jnp.issubdtype(in_phase.dtype, jnp.complexfloating) or jnp.issubdtype(
            quadrature.dtype, jnp.complexfloating
        ):
            raise TypeError("Sampled I/Q envelopes must be real.")
        angle = line.carrier.angular_rate * shifted + line.carrier.phase
        active = (shifted >= line.support_start) & (shifted <= line.support_stop)
        value = jnp.where(
            active,
            in_phase * jnp.cos(angle) + quadrature * jnp.sin(angle),
            0.0,
        )
        line_values.append(value)
    lines = jnp.stack(line_values, axis=-1)
    coefficients = lines @ schedule.transfer.matrix
    finite = (
        jnp.all(jnp.isfinite(times))
        & jnp.all(jnp.isfinite(lines))
        & jnp.all(jnp.isfinite(coefficients))
    )
    diagnostics = QuantumControlScheduleDiagnostics(
        positive,
        jnp.all(jnp.stack(tuple(line.valid for line in schedule.lines))),
        schedule.transfer.valid,
        finite,
        schedule.valid & positive & finite,
    )
    return QuantumControlScheduleResult(
        times,
        sample_times,
        lines,
        coefficients,
        diagnostics,
        schedule.schedule_id,
    )


def assemble_fixed_grid_local_hamiltonian(
    drift: LocalHamiltonian,
    drive_terms: Sequence[LocalHamiltonianTerm],
    controls: QuantumControlScheduleResult,
    /,
    *,
    hbar: ArrayLike = 1.0,
) -> FixedGridLocalHamiltonian:
    """Combine constant drift and sampled drive coefficients without special cases."""

    if not isinstance(drift, LocalHamiltonian):
        raise TypeError("drift must be a LocalHamiltonian.")
    drives = tuple(drive_terms)
    if not drives or not all(isinstance(term, LocalHamiltonianTerm) for term in drives):
        raise ValueError("drive_terms must contain LocalHamiltonianTerm values.")
    if not isinstance(controls, QuantumControlScheduleResult):
        raise TypeError("controls must be a QuantumControlScheduleResult.")
    if controls.term_coefficients.shape[1] != len(drives):
        raise ValueError("Control term coefficients must align with drive_terms.")
    combined = LocalHamiltonian(
        drift.layout,
        drift.terms + drives,
        hamiltonian_id=canonical_fingerprint(
            {
                "kind": "driven-local-hamiltonian",
                "drift": drift.hamiltonian_id,
                "drives": [term.term_id for term in drives],
                "controls": controls.schedule_id,
            }
        ),
    )
    interval_count = controls.term_coefficients.shape[0]
    drift_coefficients = jnp.ones(
        (interval_count, len(drift.terms)),
        dtype=controls.term_coefficients.dtype,
    )
    coefficients = jnp.concatenate(
        (drift_coefficients, controls.term_coefficients),
        axis=1,
    )
    return FixedGridLocalHamiltonian(
        combined,
        controls.time_grid,
        coefficients,
        hbar=hbar,
        source_valid=controls.diagnostics.valid,
        schedule_id=canonical_fingerprint(
            {
                "kind": "assembled-driven-local-hamiltonian",
                "hamiltonian": combined.hamiltonian_id,
                "controls": controls.schedule_id,
                "grid_shape": list(controls.time_grid.shape),
            }
        ),
    )


__all__ = [
    "LinearQuantumControlTransfer",
    "QuantumCarrier",
    "QuantumControlLine",
    "QuantumControlSchedule",
    "QuantumControlScheduleDiagnostics",
    "QuantumControlScheduleResult",
    "assemble_fixed_grid_local_hamiltonian",
    "sample_quantum_control_schedule",
]
