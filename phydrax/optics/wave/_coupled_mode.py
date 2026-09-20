#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Passive power-normalized bidirectional coupled-mode propagation.

The envelopes use the ``exp(-i omega t)`` convention. ``E+`` and ``E-`` are
power-normalized amplitudes in sqrt(W), ordered left-to-right as
``[E+(z), E-(z)]``.  Each interval is piecewise constant and is converted to an
exact two-port scattering map before composition; exponentially growing
transfer matrices are never cascaded.
"""

from __future__ import annotations

from enum import IntFlag
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class BidirectionalCoupledModeStatus(IntFlag):
    """Fail-closed disposition of a passive coupled-mode solve."""

    SUCCESS = 0
    NONFINITE = 1
    RECIPROCITY_DEFECT = 2
    PASSIVITY_DEFECT = 4
    POWER_DEFECT = 8
    SINGULAR_COMPOSITION = 16


class CoupledModeBoundary(StrictModule):
    """Explicit incoming/outgoing amplitudes at the ordered left/right ports."""

    left_incoming: Array
    right_incoming: Array
    left_outgoing: Array
    right_outgoing: Array


class BidirectionalCoupledModeEvidence(StrictModule):
    """Reciprocity, passivity, section power, and resource evidence."""

    input_power: Array
    output_power: Array
    absorbed_power: Array
    section_absorbed_power: Array
    power_balance_residual: Array
    reciprocity_error: Array
    maximum_local_reciprocity_error: Array
    passivity_excess: Array
    maximum_local_passivity_excess: Array
    minimum_composition_denominator: Array
    finite: Array
    successful: Array
    status: Array
    section_count: int = eqx.field(static=True)
    workspace_complex_elements: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class BidirectionalCoupledModeResult(StrictModule):
    """Boundary solution and all longitudinal interface amplitudes."""

    boundary: CoupledModeBoundary
    forward_amplitude: Array
    backward_amplitude: Array
    scattering_matrix: Array
    evidence: BidirectionalCoupledModeEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.evidence.successful

    @property
    def status(self) -> Array:
        return self.evidence.status


class BidirectionalCoupledModePlan(StrictModule, NonTrainableState):
    """Piecewise-constant reciprocal passive coupled-mode topology.

    ``z_grid`` has one more entry than each section coefficient. ``attenuation``
    is a nonnegative *power* attenuation in m^-1. A negative attenuation is gain
    and is rejected at construction rather than silently entering this passive
    solver. ``coupling`` may be complex; its conjugate appears in the reverse
    equation so the local map remains reciprocal.
    """

    z_grid: Array
    carrier_angular_frequency: Array
    detuning: Array
    coupling: Array
    attenuation: Array
    reference_propagation_constant: Array
    reciprocity_tolerance: float = eqx.field(static=True)
    passivity_tolerance: float = eqx.field(static=True)
    power_tolerance: float = eqx.field(static=True)
    singularity_tolerance: float = eqx.field(static=True)
    maximum_sections: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        z_grid: ArrayLike,
        carrier_angular_frequency: ArrayLike,
        detuning: ArrayLike,
        coupling: ArrayLike,
        attenuation: ArrayLike,
        /,
        *,
        reference_propagation_constant: ArrayLike = 0.0,
        reciprocity_tolerance: float = 1.0e-10,
        passivity_tolerance: float = 1.0e-10,
        power_tolerance: float = 1.0e-10,
        singularity_tolerance: float = 1.0e-14,
        maximum_sections: int = 100_000,
        maximum_workspace_bytes: int = 1 << 30,
        plan_id: str | None = None,
    ):
        z_host = np.asarray(z_grid)
        detuning_host = np.asarray(detuning)
        coupling_host = np.asarray(coupling)
        attenuation_host = np.asarray(attenuation)
        if (
            z_host.ndim != 1
            or z_host.size < 2
            or not np.issubdtype(z_host.dtype, np.number)
            or np.iscomplexobj(z_host)
            or np.any(~np.isfinite(z_host))
            or np.any(np.diff(z_host) <= 0.0)
        ):
            raise ValueError("z_grid must be a finite strictly increasing real vector.")
        section_count = z_host.size - 1
        if detuning_host.shape != (section_count,):
            raise ValueError("detuning must have one scalar per z interval.")
        if coupling_host.shape != (section_count,):
            raise ValueError("coupling must have one scalar per z interval.")
        if attenuation_host.shape != (section_count,):
            raise ValueError("attenuation must have one scalar per z interval.")
        if (
            np.iscomplexobj(detuning_host)
            or np.iscomplexobj(attenuation_host)
            or np.any(~np.isfinite(detuning_host))
            or np.any(~np.isfinite(attenuation_host))
            or np.any(attenuation_host < 0.0)
        ):
            raise ValueError(
                "detuning must be finite real data and attenuation must be finite, "
                "real, and nonnegative; passive coupled mode rejects gain."
            )
        if np.any(~np.isfinite(np.real(coupling_host))) or np.any(
            ~np.isfinite(np.imag(coupling_host))
        ):
            raise ValueError("coupling must be finite complex data.")
        frequency_host = np.asarray(carrier_angular_frequency)
        reference_host = np.asarray(reference_propagation_constant)
        if (
            frequency_host.shape != ()
            or np.iscomplexobj(frequency_host)
            or not np.isfinite(frequency_host)
            or frequency_host <= 0.0
        ):
            raise ValueError("carrier_angular_frequency must be finite and positive.")
        if (
            reference_host.shape != ()
            or np.iscomplexobj(reference_host)
            or not np.isfinite(reference_host)
            or reference_host < 0.0
        ):
            raise ValueError(
                "reference_propagation_constant must be finite and nonnegative."
            )

        tolerances = (
            float(reciprocity_tolerance),
            float(passivity_tolerance),
            float(power_tolerance),
            float(singularity_tolerance),
        )
        if any(not np.isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError("Coupled-mode tolerances must be finite and nonnegative.")
        sections = int(maximum_sections)
        workspace = int(maximum_workspace_bytes)
        if sections < 1 or workspace < 1:
            raise ValueError("Coupled-mode resource bounds must be positive integers.")
        if section_count > sections:
            raise ValueError("z_grid exceeds maximum_sections.")
        complex_dtype = jnp.result_type(coupling_host.dtype, jnp.complex64)
        real_dtype = jnp.empty((), dtype=complex_dtype).real.dtype
        self.z_grid = jnp.asarray(z_host, dtype=real_dtype)
        self.carrier_angular_frequency = jnp.asarray(frequency_host, dtype=real_dtype)
        self.detuning = jnp.asarray(detuning_host, dtype=real_dtype)
        self.coupling = jnp.asarray(coupling_host, dtype=complex_dtype)
        self.attenuation = jnp.asarray(attenuation_host, dtype=real_dtype)
        self.reference_propagation_constant = jnp.asarray(
            reference_host, dtype=real_dtype
        )
        self.reciprocity_tolerance = tolerances[0]
        self.passivity_tolerance = tolerances[1]
        self.power_tolerance = tolerances[2]
        self.singularity_tolerance = tolerances[3]
        self.maximum_sections = sections
        self.maximum_workspace_bytes = workspace
        generated = canonical_fingerprint(
            {
                "kind": "bidirectional-coupled-mode-plan",
                "z_grid": array_tree_fingerprint(z_host),
                "carrier_angular_frequency": float(frequency_host).hex(),
                "detuning": array_tree_fingerprint(detuning_host),
                "coupling": array_tree_fingerprint(coupling_host),
                "attenuation": array_tree_fingerprint(attenuation_host),
                "reference_propagation_constant": float(reference_host).hex(),
                "reciprocity_tolerance": tolerances[0].hex(),
                "passivity_tolerance": tolerances[1].hex(),
                "power_tolerance": tolerances[2].hex(),
                "singularity_tolerance": tolerances[3].hex(),
                "maximum_sections": sections,
                "maximum_workspace_bytes": workspace,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.plan_id = identifier

    def prepare(self, /) -> "PreparedBidirectionalCoupledMode":
        return prepare_bidirectional_coupled_mode(self)


class PreparedBidirectionalCoupledMode(StrictModule, NonTrainableState):
    """Exact local maps and stable prefix/suffix scattering compositions."""

    plan: BidirectionalCoupledModePlan
    section_lengths: Array
    section_scattering: Array
    prefix_scattering: Array
    suffix_scattering: Array
    scattering_matrix: Array
    local_reciprocity_error: Array
    local_passivity_excess: Array
    composition_denominators: Array
    section_count: int = eqx.field(static=True)
    workspace_complex_elements: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def execute(
        self,
        /,
        *,
        left_incoming: ArrayLike = 0.0j,
        right_incoming: ArrayLike = 0.0j,
    ) -> BidirectionalCoupledModeResult:
        return solve_bidirectional_coupled_mode(
            self,
            left_incoming=left_incoming,
            right_incoming=right_incoming,
        )


def _identity_scattering(dtype) -> Array:
    return jnp.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=dtype)


def _cascade(left: Array, right: Array, /) -> tuple[Array, Array]:
    """Redheffer composition for a scalar two-port left of ``right``."""

    denominator = 1.0 - left[1, 1] * right[0, 0]
    safe = jnp.where(jnp.abs(denominator) > 0.0, denominator, 1.0 + 0.0j)
    combined = jnp.stack(
        (
            jnp.stack(
                (
                    left[0, 0] + left[0, 1] * right[0, 0] * left[1, 0] / safe,
                    left[0, 1] * right[0, 1] / safe,
                )
            ),
            jnp.stack(
                (
                    right[1, 0] * left[1, 0] / safe,
                    right[1, 1] + right[1, 0] * left[1, 1] * right[0, 1] / safe,
                )
            ),
        )
    )
    return combined, jnp.abs(denominator)


def _passivity_excess(scattering: Array, /) -> Array:
    h00 = jnp.real(jnp.conj(scattering[0, 0]) * scattering[0, 0]) + jnp.real(
        jnp.conj(scattering[1, 0]) * scattering[1, 0]
    )
    h11 = jnp.real(jnp.conj(scattering[0, 1]) * scattering[0, 1]) + jnp.real(
        jnp.conj(scattering[1, 1]) * scattering[1, 1]
    )
    h01 = (
        jnp.conj(scattering[0, 0]) * scattering[0, 1]
        + jnp.conj(scattering[1, 0]) * scattering[1, 1]
    )
    largest = 0.5 * (
        h00 + h11 + jnp.sqrt((h00 - h11) ** 2 + 4.0 * jnp.real(h01 * jnp.conj(h01)))
    )
    return jnp.maximum(largest - 1.0, 0.0)


def _local_scattering(
    length: Array,
    detuning: Array,
    coupling: Array,
    attenuation: Array,
    reference_propagation_constant: Array,
    /,
) -> Array:
    """Exact constant-section exponential, scaled before transfer-to-scattering."""

    dtype = jnp.result_type(coupling.dtype, jnp.complex64)
    diagonal = (
        -0.5 * attenuation + 1j * (detuning + reference_propagation_constant)
    ).astype(dtype)
    gamma = jnp.sqrt(diagonal * diagonal + coupling * jnp.conj(coupling))
    argument = gamma * length
    small = jnp.abs(argument) <= 32.0 * jnp.finfo(jnp.real(argument).dtype).eps
    scale = jnp.abs(jnp.real(argument))
    positive = jnp.exp(argument - scale)
    negative = jnp.exp(-argument - scale)
    cosine_scaled = 0.5 * (positive + negative)
    sine_scaled = 0.5 * (positive - negative)
    safe_gamma = jnp.where(small, 1.0 + 0.0j, gamma)
    hyperbolic_over_gamma = sine_scaled / safe_gamma
    series_h = length * (1.0 + argument * argument / 6.0 + argument**4 / 120.0)
    series_c = 1.0 + argument * argument / 2.0 + argument**4 / 24.0
    hyperbolic_over_gamma = jnp.where(small, series_h, hyperbolic_over_gamma)
    cosine_scaled = jnp.where(small, series_c, cosine_scaled)
    denominator = cosine_scaled - diagonal * hyperbolic_over_gamma
    safe_denominator = jnp.where(jnp.abs(denominator) > 0.0, denominator, 1.0 + 0.0j)
    transmission = jnp.exp(-scale) / safe_denominator
    return jnp.stack(
        (
            jnp.stack(
                (
                    1j * jnp.conj(coupling) * hyperbolic_over_gamma / safe_denominator,
                    transmission,
                )
            ),
            jnp.stack(
                (
                    transmission,
                    1j * coupling * hyperbolic_over_gamma / safe_denominator,
                )
            ),
        )
    )


def prepare_bidirectional_coupled_mode(
    plan: BidirectionalCoupledModePlan,
    /,
) -> PreparedBidirectionalCoupledMode:
    """Prepare exact section maps and stable two-sided compositions."""

    if not isinstance(plan, BidirectionalCoupledModePlan):
        raise TypeError("plan must be a BidirectionalCoupledModePlan.")
    lengths = jnp.diff(plan.z_grid)
    local = jnp.stack(
        tuple(
            _local_scattering(
                lengths[index],
                plan.detuning[index],
                plan.coupling[index],
                plan.attenuation[index],
                plan.reference_propagation_constant,
            )
            for index in range(lengths.size)
        )
    )
    identity = _identity_scattering(local.dtype)
    prefixes = [identity]
    denominators = []
    for section in local:
        combined, denominator = _cascade(prefixes[-1], section)
        prefixes.append(combined)
        denominators.append(denominator)
    suffixes = [identity]
    for section in reversed(local):
        combined, denominator = _cascade(section, suffixes[-1])
        suffixes.append(combined)
        denominators.append(denominator)
    suffixes.reverse()
    prefix = jnp.stack(tuple(prefixes))
    suffix = jnp.stack(tuple(suffixes))
    composition_denominators = jnp.stack(tuple(denominators))
    local_reciprocity = jnp.abs(local[:, 0, 1] - local[:, 1, 0])
    local_passivity = jnp.stack(tuple(_passivity_excess(value) for value in local))
    section_count = lengths.size
    workspace_complex_elements = int(
        prod(local.shape) + prod(prefix.shape) + prod(suffix.shape)
    )
    workspace_bytes = workspace_complex_elements * np.dtype(local.dtype).itemsize
    if workspace_bytes > plan.maximum_workspace_bytes:
        raise ValueError("Prepared coupled-mode maps exceed maximum_workspace_bytes.")
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-bidirectional-coupled-mode",
            "plan": plan.plan_id,
            "section_count": section_count,
            "workspace_complex_elements": workspace_complex_elements,
            "workspace_bytes": workspace_bytes,
        }
    )
    return PreparedBidirectionalCoupledMode(
        plan,
        lengths,
        local,
        prefix,
        suffix,
        prefix[-1],
        local_reciprocity,
        local_passivity,
        composition_denominators,
        section_count,
        workspace_complex_elements,
        workspace_bytes,
        prepared_id,
    )


def solve_bidirectional_coupled_mode(
    prepared: PreparedBidirectionalCoupledMode,
    /,
    *,
    left_incoming: ArrayLike = 0.0j,
    right_incoming: ArrayLike = 0.0j,
) -> BidirectionalCoupledModeResult:
    """Solve the explicit left/right incoming boundary-value problem."""

    if not isinstance(prepared, PreparedBidirectionalCoupledMode):
        raise TypeError("prepared must be a PreparedBidirectionalCoupledMode.")
    dtype = prepared.scattering_matrix.dtype
    left = jnp.asarray(left_incoming, dtype=dtype)
    right = jnp.asarray(right_incoming, dtype=dtype)
    if left.shape != right.shape:
        raise ValueError("left_incoming and right_incoming must have the same shape.")
    if left.ndim > 1:
        raise ValueError(
            "Boundary amplitudes must be scalars or one-dimensional batches."
        )
    scattering = prepared.scattering_matrix
    left_out = scattering[0, 0] * left + scattering[0, 1] * right
    right_out = scattering[1, 0] * left + scattering[1, 1] * right

    forward = []
    backward = []
    junction_denominators = []
    for index in range(prepared.section_count + 1):
        prefix = prepared.prefix_scattering[index]
        suffix = prepared.suffix_scattering[index]
        denominator = 1.0 - suffix[0, 0] * prefix[1, 1]
        safe = jnp.where(jnp.abs(denominator) > 0.0, denominator, 1.0 + 0.0j)
        backward_here = (suffix[0, 0] * prefix[1, 0] * left + suffix[0, 1] * right) / safe
        forward_here = prefix[1, 0] * left + prefix[1, 1] * backward_here
        forward.append(forward_here)
        backward.append(backward_here)
        junction_denominators.append(jnp.abs(denominator))
    forward_amplitude = jnp.stack(tuple(forward))
    backward_amplitude = jnp.stack(tuple(backward))
    junction_denominator = jnp.stack(tuple(junction_denominators))

    forward_power = jnp.real(forward_amplitude * jnp.conj(forward_amplitude))
    backward_power = jnp.real(backward_amplitude * jnp.conj(backward_amplitude))
    flux = forward_power - backward_power
    section_absorbed = flux[:-1] - flux[1:]
    input_power = jnp.real(left * jnp.conj(left) + right * jnp.conj(right))
    output_power = jnp.real(
        left_out * jnp.conj(left_out) + right_out * jnp.conj(right_out)
    )
    absorbed_power = jnp.sum(section_absorbed, axis=0)
    scale = jnp.maximum(input_power, jnp.asarray(1.0, dtype=input_power.dtype))
    power_residual = jnp.abs(input_power - output_power - absorbed_power) / scale
    reciprocity = jnp.abs(scattering[0, 1] - scattering[1, 0])
    passivity = _passivity_excess(scattering)
    maximum_local_reciprocity = jnp.max(prepared.local_reciprocity_error)
    maximum_local_passivity = jnp.max(prepared.local_passivity_excess)
    minimum_denominator = jnp.min(
        jnp.concatenate((prepared.composition_denominators, junction_denominator))
    )
    finite = (
        jnp.all(jnp.isfinite(jnp.real(forward_amplitude)))
        & jnp.all(jnp.isfinite(jnp.imag(forward_amplitude)))
        & jnp.all(jnp.isfinite(jnp.real(backward_amplitude)))
        & jnp.all(jnp.isfinite(jnp.imag(backward_amplitude)))
        & jnp.all(jnp.isfinite(section_absorbed))
        & jnp.all(jnp.isfinite(power_residual))
    )
    reciprocal = (reciprocity <= prepared.plan.reciprocity_tolerance) & (
        maximum_local_reciprocity <= prepared.plan.reciprocity_tolerance
    )
    passive = (
        (passivity <= prepared.plan.passivity_tolerance)
        & (maximum_local_passivity <= prepared.plan.passivity_tolerance)
        & jnp.all(section_absorbed >= -prepared.plan.power_tolerance * scale)
    )
    power_closed = jnp.all(power_residual <= prepared.plan.power_tolerance)
    nonsingular = minimum_denominator > prepared.plan.singularity_tolerance
    status = (
        jnp.where(finite, 0, int(BidirectionalCoupledModeStatus.NONFINITE))
        | jnp.where(reciprocal, 0, int(BidirectionalCoupledModeStatus.RECIPROCITY_DEFECT))
        | jnp.where(passive, 0, int(BidirectionalCoupledModeStatus.PASSIVITY_DEFECT))
        | jnp.where(power_closed, 0, int(BidirectionalCoupledModeStatus.POWER_DEFECT))
        | jnp.where(
            nonsingular, 0, int(BidirectionalCoupledModeStatus.SINGULAR_COMPOSITION)
        )
    ).astype(jnp.int32)
    successful = status == int(BidirectionalCoupledModeStatus.SUCCESS)
    boundary = CoupledModeBoundary(left, right, left_out, right_out)
    evidence_id = canonical_fingerprint(
        {
            "kind": "bidirectional-coupled-mode-evidence",
            "prepared": prepared.prepared_id,
            "section_count": prepared.section_count,
        }
    )
    evidence = BidirectionalCoupledModeEvidence(
        input_power,
        output_power,
        absorbed_power,
        section_absorbed,
        power_residual,
        reciprocity,
        maximum_local_reciprocity,
        passivity,
        maximum_local_passivity,
        minimum_denominator,
        finite,
        successful,
        status,
        prepared.section_count,
        prepared.workspace_complex_elements,
        evidence_id,
    )
    return BidirectionalCoupledModeResult(
        boundary,
        forward_amplitude,
        backward_amplitude,
        scattering,
        evidence,
        prepared.prepared_id,
    )


__all__ = [
    "BidirectionalCoupledModeEvidence",
    "BidirectionalCoupledModePlan",
    "BidirectionalCoupledModeResult",
    "BidirectionalCoupledModeStatus",
    "CoupledModeBoundary",
    "PreparedBidirectionalCoupledMode",
    "prepare_bidirectional_coupled_mode",
    "solve_bidirectional_coupled_mode",
]
