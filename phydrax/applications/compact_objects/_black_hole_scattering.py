#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    solve as solve_linear,
)
from ._perturbation import SeparatedMode
from ._radial_perturbation import (
    _integrate_schwarzschild_horizon_riccati,
    _schwarzschild_infinity_series,
    RadialAsymptoticEvidence,
    schwarzschild_radial_asymptotics,
    SchwarzschildRadialPlan,
)


class SuperradianceStatus(IntEnum):
    """Classification of one real-frequency horizon channel."""

    NONSUPERRADIANT = 0
    SUPERRADIANT = 1
    THRESHOLD = 2
    INCONSISTENT = 3


class BlackHoleScatteringStatus(IntEnum):
    """Terminal numerical/physical status of a scattering evaluation."""

    SUCCESS = 0
    NONFINITE = 1
    INVALID_FREQUENCY = 2
    ZERO_INCIDENT_FLUX = 3
    WRONSKIAN_NOT_CONVERGED = 4
    FLUX_NOT_CONSERVED = 5
    INCONSISTENT_SUPERRADIANCE = 6


class ScatteringQualificationEvidence(StrictModule, NonTrainableState):
    """Explicit external qualification bound to one radial scattering source."""

    accepted: Array
    mode_id: str = eqx.field(static=True)
    radial_source_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: SeparatedMode,
        accepted: bool,
        /,
        *,
        radial_source_id: str,
        source_id: str,
    ):
        if not isinstance(mode, SeparatedMode):
            raise TypeError("mode must be a SeparatedMode.")
        radial = str(radial_source_id)
        source = str(source_id)
        if not radial or not source:
            raise ValueError("Qualification radial/source identifiers must be non-empty.")
        accepted_ = bool(accepted)
        self.accepted = jnp.asarray(accepted_)
        self.mode_id = mode.mode_id
        self.radial_source_id = radial
        self.source_id = source
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "black-hole-scattering-qualification",
                "accepted": accepted_,
                "radial_source": radial,
                "mode": mode.mode_id,
                "source": source,
            }
        )


class BlackHoleScatteringPlan(StrictModule, NonTrainableState):
    """Flux ledger for independently solved real-frequency radial amplitudes.

    The initial convention is the massless-scalar Killing-energy normalization
    ``F_in=omega |I|^2``, ``F_ref=omega |R|^2`` and
    ``F_H=(omega-m Omega_H) |T|^2``.  It deliberately does not accept a QNM
    result: real-frequency scattering and complex-frequency poles have distinct
    boundary data and qualification routes.
    """

    mode: SeparatedMode
    horizon_angular_velocity: Array
    flux_tolerance: float = eqx.field(static=True)
    threshold_tolerance: float = eqx.field(static=True)
    flux_normalization: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        mode: SeparatedMode,
        horizon_angular_velocity: ArrayLike,
        flux_tolerance: float,
        threshold_tolerance: float,
        /,
        *,
        flux_normalization: str = "scalar-killing-energy-unit-incoming-at-infinity",
    ):
        if not isinstance(mode, SeparatedMode):
            raise TypeError("mode must be a SeparatedMode.")
        if mode.family != "scattering":
            raise ValueError(
                "BlackHoleScatteringPlan requires a mode with family='scattering'."
            )
        omega_h = jnp.asarray(horizon_angular_velocity)
        if omega_h.shape != () or not jnp.issubdtype(omega_h.dtype, jnp.floating):
            raise TypeError("horizon_angular_velocity must be one real floating scalar.")
        tolerances = float(flux_tolerance), float(threshold_tolerance)
        if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError("Scattering tolerances must be finite and positive.")
        normalization = str(flux_normalization)
        if normalization != "scalar-killing-energy-unit-incoming-at-infinity":
            raise ValueError(
                "The initial scattering closure supports only the declared "
                "massless-scalar Killing-energy normalization."
            )
        if mode.spin_weight != 0:
            raise ValueError(
                "The scalar Killing-energy amplitude normalization requires spin_weight=0."
            )
        if not bool(jnp.isfinite(omega_h)):
            raise ValueError("horizon_angular_velocity must be finite.")
        self.mode = mode
        self.horizon_angular_velocity = omega_h
        self.flux_tolerance, self.threshold_tolerance = tolerances
        self.flux_normalization = normalization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "black-hole-scattering-plan",
                "mode": mode.mode_id,
                "horizon_angular_velocity": float(omega_h),
                "flux_tolerance": self.flux_tolerance,
                "threshold_tolerance": self.threshold_tolerance,
                "flux_normalization": normalization,
            }
        )


class BlackHoleScatteringResult(StrictModule):
    """Amplitudes, signed fluxes, and conservation evidence for one real mode."""

    angular_frequency: Array
    horizon_wave_number: Array
    incident_amplitude: Array
    reflected_amplitude: Array
    horizon_amplitude: Array
    incident_flux: Array
    reflected_flux: Array
    horizon_flux: Array
    graybody_factor: Array
    amplification_factor: Array
    corotation_slope: Array
    wronskian_residual: Array
    flux_residual: Array
    superradiant: Array
    superradiance_status: Array
    status: Array
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    mode_id: str = eqx.field(static=True)
    radial_source_id: str = eqx.field(static=True)
    qualification_id: str = eqx.field(static=True)
    flux_normalization: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        """Whether the numerical and physical scattering solve is accepted."""

        return self.finite & self.converged & self.physically_valid


def _real_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or not jnp.issubdtype(scalar.dtype, jnp.floating):
        raise TypeError(f"{name} must be one real floating scalar.")
    return scalar


def _complex_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if scalar.shape != () or not jnp.issubdtype(scalar.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be one complex floating scalar.")
    return scalar


def solve_black_hole_scattering(
    plan: BlackHoleScatteringPlan,
    angular_frequency: ArrayLike,
    incident_amplitude: ArrayLike,
    reflected_amplitude: ArrayLike,
    horizon_amplitude: ArrayLike,
    radial_wronskian_residual: ArrayLike,
    corotation_slope: ArrayLike,
    /,
    *,
    radial_source_id: str,
    qualification: ScatteringQualificationEvidence | None = None,
) -> BlackHoleScatteringResult:
    """Close the real-frequency flux ledger for independent radial data.

    ``radial_wronskian_residual`` is the residual reported by the independent
    radial solution.  ``flux_residual`` is recomputed from the asymptotic
    amplitudes here, so an apparently small radial Wronskian cannot hide an
    inconsistent boundary normalization.

    ``corotation_slope`` is the independently resolved derivative
    ``d(graybody_factor)/d(angular_frequency)`` along this fixed mode and
    background branch.  It is retained for the paired thermal limit at the
    superradiant threshold; this function never manufactures it from one lane.
    """

    if not isinstance(plan, BlackHoleScatteringPlan):
        raise TypeError("plan must be a BlackHoleScatteringPlan.")
    if qualification is not None and not isinstance(
        qualification, ScatteringQualificationEvidence
    ):
        raise TypeError("qualification must be ScatteringQualificationEvidence or None.")
    source = str(radial_source_id)
    if not source:
        raise ValueError("radial_source_id must be non-empty.")

    frequency = _real_scalar(angular_frequency, "angular_frequency")
    incident = _complex_scalar(incident_amplitude, "incident_amplitude")
    reflected = _complex_scalar(reflected_amplitude, "reflected_amplitude")
    horizon = _complex_scalar(horizon_amplitude, "horizon_amplitude")
    wronskian = _complex_scalar(radial_wronskian_residual, "radial_wronskian_residual")
    slope = _real_scalar(corotation_slope, "corotation_slope")

    horizon_wave_number = frequency - plan.mode.m * plan.horizon_angular_velocity
    incident_flux = frequency * jnp.real(incident * jnp.conj(incident))
    reflected_flux = frequency * jnp.real(reflected * jnp.conj(reflected))
    horizon_flux = horizon_wave_number * jnp.real(horizon * jnp.conj(horizon))
    tiny = jnp.finfo(frequency.dtype).tiny
    safe_incident_flux = jnp.where(incident_flux > tiny, incident_flux, 1.0)
    graybody_factor = horizon_flux / safe_incident_flux
    amplification_factor = reflected_flux / safe_incident_flux - 1.0
    flux_residual = incident_flux - reflected_flux - horizon_flux

    finite = jnp.all(
        jnp.isfinite(
            jnp.asarray(
                (
                    frequency,
                    horizon_wave_number,
                    incident,
                    reflected,
                    horizon,
                    wronskian,
                    incident_flux,
                    reflected_flux,
                    horizon_flux,
                    graybody_factor,
                    amplification_factor,
                    slope,
                    flux_residual,
                )
            )
        )
    )
    positive_frequency = frequency > 0.0
    nonzero_incident = incident_flux > tiny
    flux_scale = jnp.maximum(
        jnp.maximum(jnp.abs(incident_flux), jnp.abs(reflected_flux)),
        jnp.maximum(jnp.abs(horizon_flux), tiny),
    )
    wronskian_scale = jnp.maximum(2.0 * flux_scale, tiny)
    wronskian_converged = jnp.abs(wronskian) <= (plan.flux_tolerance * wronskian_scale)
    flux_converged = jnp.abs(flux_residual) <= plan.flux_tolerance * flux_scale
    converged = finite & wronskian_converged & flux_converged

    below_threshold = horizon_wave_number < -plan.threshold_tolerance
    above_threshold = horizon_wave_number > plan.threshold_tolerance
    at_threshold = ~(below_threshold | above_threshold)
    amplified = reflected_flux > incident_flux
    negative_absorption = graybody_factor < 0.0
    superradiant = below_threshold & amplified & negative_absorption
    regime_consistent = (
        at_threshold
        | (below_threshold & superradiant)
        | (above_threshold & (~amplified) & (graybody_factor >= 0.0))
    )
    superradiance_status = jnp.where(
        at_threshold,
        int(SuperradianceStatus.THRESHOLD),
        jnp.where(
            superradiant,
            int(SuperradianceStatus.SUPERRADIANT),
            jnp.where(
                regime_consistent,
                int(SuperradianceStatus.NONSUPERRADIANT),
                int(SuperradianceStatus.INCONSISTENT),
            ),
        ),
    ).astype(jnp.int32)
    physically_valid = finite & positive_frequency & nonzero_incident & regime_consistent

    qualified_by_source = jnp.asarray(False)
    qualification_id = ""
    if qualification is not None:
        qualification_id = qualification.evidence_id
        qualified_by_source = (
            qualification.accepted
            & jnp.asarray(qualification.mode_id == plan.mode.mode_id)
            & jnp.asarray(qualification.radial_source_id == source)
        )
    qualified = converged & physically_valid & qualified_by_source
    derivative_valid = (
        converged
        & physically_valid
        & (jnp.abs(horizon_wave_number) > plan.threshold_tolerance)
        & jnp.isfinite(slope)
    )

    status = jnp.asarray(int(BlackHoleScatteringStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        finite,
        status,
        int(BlackHoleScatteringStatus.NONFINITE),
    )
    status = jnp.where(
        finite & (~positive_frequency),
        int(BlackHoleScatteringStatus.INVALID_FREQUENCY),
        status,
    )
    status = jnp.where(
        finite & positive_frequency & (~nonzero_incident),
        int(BlackHoleScatteringStatus.ZERO_INCIDENT_FLUX),
        status,
    )
    status = jnp.where(
        finite & positive_frequency & nonzero_incident & (~wronskian_converged),
        int(BlackHoleScatteringStatus.WRONSKIAN_NOT_CONVERGED),
        status,
    )
    status = jnp.where(
        finite
        & positive_frequency
        & nonzero_incident
        & wronskian_converged
        & (~flux_converged),
        int(BlackHoleScatteringStatus.FLUX_NOT_CONSERVED),
        status,
    )
    status = jnp.where(
        finite
        & positive_frequency
        & nonzero_incident
        & wronskian_converged
        & flux_converged
        & (~regime_consistent),
        int(BlackHoleScatteringStatus.INCONSISTENT_SUPERRADIANCE),
        status,
    ).astype(jnp.int32)

    return BlackHoleScatteringResult(
        frequency,
        horizon_wave_number,
        incident,
        reflected,
        horizon,
        incident_flux,
        reflected_flux,
        horizon_flux,
        graybody_factor,
        amplification_factor,
        slope,
        wronskian,
        flux_residual,
        superradiant,
        superradiance_status,
        status,
        finite,
        converged,
        physically_valid,
        qualified,
        derivative_valid,
        plan.mode.mode_id,
        source,
        qualification_id,
        plan.flux_normalization,
        plan.plan_id,
    )


class SchwarzschildScatteringStatus(IntEnum):
    """Terminal status of the computed scalar Schwarzschild scattering solve."""

    SUCCESS = 0
    INVALID_FREQUENCY = 1
    NONFINITE = 2
    DECOMPOSITION_FAILED = 3
    INCIDENT_AMPLITUDE_UNRESOLVED = 4
    BASIS_FLUX_UNRESOLVED = 5
    ASYMPTOTIC_UNRESOLVED = 6
    STEP_REFINEMENT_UNRESOLVED = 7
    WRONSKIAN_NOT_CONSERVED = 8
    FLUX_NOT_CONSERVED = 9
    LOW_FREQUENCY_CONTROL_FAILED = 10
    SCATTERING_LEDGER_UNRESOLVED = 11
    SLOPE_REFINEMENT_UNRESOLVED = 12


class SchwarzschildScatteringSolvePlan(StrictModule, NonTrainableState):
    """Bounded one-sided scalar Schwarzschild scattering solve.

    The coarse and refined plans share one physical domain and asymptotic order.
    Only the fixed RK4 substep count changes, so their difference is direct
    integration-resolution evidence rather than a change of radial problem.
    """

    radial_plan: SchwarzschildRadialPlan
    refined_radial_plan: SchwarzschildRadialPlan
    flux_plan: BlackHoleScatteringPlan
    decomposition_policy: LinearSolvePolicy
    frequency_step: float = eqx.field(static=True)
    decomposition_tolerance: float = eqx.field(static=True)
    refinement_tolerance: float = eqx.field(static=True)
    slope_refinement_tolerance: float = eqx.field(static=True)
    absolute_flux_tolerance: float = eqx.field(static=True)
    incident_amplitude_tolerance: float = eqx.field(static=True)
    low_frequency_maximum: float = eqx.field(static=True)
    low_frequency_relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        radial_plan: SchwarzschildRadialPlan,
        flux_plan: BlackHoleScatteringPlan,
        decomposition_policy: LinearSolvePolicy,
        /,
        *,
        refined_integration_substeps: int,
        frequency_step: float,
        decomposition_tolerance: float,
        refinement_tolerance: float,
        slope_refinement_tolerance: float,
        absolute_flux_tolerance: float,
        incident_amplitude_tolerance: float,
        low_frequency_maximum: float,
        low_frequency_relative_tolerance: float,
    ):
        if not isinstance(radial_plan, SchwarzschildRadialPlan):
            raise TypeError("radial_plan must be a SchwarzschildRadialPlan.")
        if not isinstance(flux_plan, BlackHoleScatteringPlan):
            raise TypeError("flux_plan must be a BlackHoleScatteringPlan.")
        if not isinstance(decomposition_policy, LinearSolvePolicy):
            raise TypeError("decomposition_policy must be a LinearSolvePolicy.")
        mode = radial_plan.mode
        if (
            mode.family != "scattering"
            or mode.spin_weight != 0
            or mode.sector != "scalar"
        ):
            raise ValueError(
                "Schwarzschild scattering requires family='scattering', spin_weight=0, and sector='scalar'."
            )
        if (
            radial_plan.boundary.horizon != "ingoing"
            or radial_plan.boundary.infinity != "outgoing"
        ):
            raise ValueError(
                "One-sided Schwarzschild scattering requires an ingoing horizon "
                "and the outgoing infinity basis convention."
            )
        if flux_plan.mode.mode_id != mode.mode_id:
            raise ValueError("Radial and flux plan mode identities must match.")
        if float(np.asarray(flux_plan.horizon_angular_velocity)) != 0.0:
            raise ValueError("A Schwarzschild scattering flux plan requires Omega_H=0.")
        if isinstance(refined_integration_substeps, bool) or not isinstance(
            refined_integration_substeps, Integral
        ):
            raise TypeError("refined_integration_substeps must be an exact integer.")
        refined_substeps = int(refined_integration_substeps)
        if refined_substeps <= radial_plan.integration_substeps:
            raise ValueError(
                "refined_integration_substeps must exceed the radial plan substeps."
            )
        values = (
            frequency_step,
            decomposition_tolerance,
            refinement_tolerance,
            slope_refinement_tolerance,
            absolute_flux_tolerance,
            incident_amplitude_tolerance,
            low_frequency_maximum,
            low_frequency_relative_tolerance,
        )
        tolerances = tuple(float(value) for value in values)
        if any(not math.isfinite(value) or value <= 0.0 for value in tolerances):
            raise ValueError(
                "Scattering frequency/control steps and tolerances must be finite and positive."
            )
        nodes = np.asarray(radial_plan.radial_nodes)
        inner = float(nodes[0])
        outer = float(nodes[-1])
        match_fraction = float((nodes[radial_plan.match_index] - inner) / (outer - inner))
        refined = SchwarzschildRadialPlan(
            mode,
            float(np.asarray(radial_plan.mass)),
            node_count=radial_plan.node_count,
            inner_radius=inner,
            outer_radius=outer,
            match_fraction=match_fraction,
            boundary=radial_plan.boundary,
            residual_tolerance=radial_plan.residual_tolerance,
            matching_tolerance=radial_plan.matching_tolerance,
            asymptotic_tolerance=radial_plan.asymptotic_tolerance,
            wave_number_tolerance=radial_plan.wave_number_tolerance,
            maximum_dimension=radial_plan.node_count,
            integration_substeps=refined_substeps,
            infinity_asymptotic_order=radial_plan.infinity_asymptotic_order,
        )
        (
            self.frequency_step,
            self.decomposition_tolerance,
            self.refinement_tolerance,
            self.slope_refinement_tolerance,
            self.absolute_flux_tolerance,
            self.incident_amplitude_tolerance,
            self.low_frequency_maximum,
            self.low_frequency_relative_tolerance,
        ) = tolerances
        self.radial_plan = radial_plan
        self.refined_radial_plan = refined
        self.flux_plan = flux_plan
        self.decomposition_policy = decomposition_policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "schwarzschild-scalar-scattering-solve-plan",
                "radial_plan": radial_plan.plan_id,
                "refined_radial_plan": refined.plan_id,
                "flux_plan": flux_plan.plan_id,
                "decomposition_method": decomposition_policy.method.name,
                "decomposition_relative_tolerance": (
                    decomposition_policy.tolerance.relative
                ),
                "decomposition_absolute_tolerance": (
                    decomposition_policy.tolerance.absolute
                ),
                "frequency_step": self.frequency_step,
                "decomposition_tolerance": self.decomposition_tolerance,
                "refinement_tolerance": self.refinement_tolerance,
                "slope_refinement_tolerance": self.slope_refinement_tolerance,
                "absolute_flux_tolerance": self.absolute_flux_tolerance,
                "incident_amplitude_tolerance": self.incident_amplitude_tolerance,
                "low_frequency_maximum": self.low_frequency_maximum,
                "low_frequency_relative_tolerance": (
                    self.low_frequency_relative_tolerance
                ),
            }
        )


class _SchwarzschildAmplitudeEvaluation(StrictModule):
    angular_frequency: Array
    source_incident_amplitude: Array
    source_reflected_amplitude: Array
    reflected_amplitude: Array
    horizon_amplitude: Array
    infinity_incident_unit_flux: Array
    horizon_unit_flux: Array
    decomposition_residual: Array
    decomposition_condition: Array
    wronskian_residual: Array
    asymptotic: RadialAsymptoticEvidence
    decomposition: LinearSolveResult
    finite: Array
    decomposition_valid: Array
    incident_valid: Array
    basis_flux_valid: Array
    asymptotic_valid: Array
    qualified: Array
    radial_plan_id: str = eqx.field(static=True)


class _SchwarzschildAmplitudePair(StrictModule):
    coarse: _SchwarzschildAmplitudeEvaluation
    refined: _SchwarzschildAmplitudeEvaluation
    reflection_error: Array
    horizon_error: Array
    graybody_error: Array
    threshold: Array
    finite: Array
    resolved: Array


class SchwarzschildScatteringEvidence(StrictModule):
    """Decomposition, asymptotic, refinement, and control evidence."""

    source_incident_amplitude: Array
    source_reflected_amplitude: Array
    infinity_incident_unit_flux: Array
    horizon_unit_flux: Array
    decomposition_residual: Array
    decomposition_condition: Array
    reflection_refinement_error: Array
    horizon_refinement_error: Array
    graybody_refinement_error: Array
    refinement_threshold: Array
    neighboring_frequencies: Array
    neighboring_graybody_factors: Array
    neighboring_resolved: Array
    coarse_dimensionless_corotation_slope: Array
    dimensionless_slope_refinement_error: Array
    neighboring_ledger_valid: Array
    dimensionless_slope_refinement_threshold: Array
    slope_refinement_valid: Array
    low_frequency_reference: Array
    low_frequency_relative_error: Array
    low_frequency_control_active: Array
    low_frequency_control_valid: Array
    source_finite: Array
    basis_flux_valid: Array
    source_converged: Array
    scattering_ledger_valid: Array
    absolute_wronskian_valid: Array
    absolute_flux_valid: Array
    radial_plan_id: str = eqx.field(static=True)
    refined_radial_plan_id: str = eqx.field(static=True)
    decomposition_plan_id: str = eqx.field(static=True)


class SchwarzschildScatteringResult(StrictModule):
    """Computed scalar Schwarzschild amplitudes and signed flux product."""

    scattering: BlackHoleScatteringResult
    evidence: SchwarzschildScatteringEvidence
    asymptotic: RadialAsymptoticEvidence
    finite: Array
    converged: Array
    physically_valid: Array
    qualified: Array
    derivative_valid: Array
    status: Array
    mode_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    @property
    def angular_frequency(self) -> Array:
        return self.scattering.angular_frequency

    @property
    def incident_amplitude(self) -> Array:
        return self.scattering.incident_amplitude

    @property
    def reflected_amplitude(self) -> Array:
        return self.scattering.reflected_amplitude

    @property
    def horizon_amplitude(self) -> Array:
        return self.scattering.horizon_amplitude

    @property
    def incident_flux(self) -> Array:
        return self.scattering.incident_flux

    @property
    def reflected_flux(self) -> Array:
        return self.scattering.reflected_flux

    @property
    def horizon_flux(self) -> Array:
        return self.scattering.horizon_flux

    @property
    def graybody_factor(self) -> Array:
        return self.scattering.graybody_factor

    @property
    def corotation_slope(self) -> Array:
        return self.scattering.corotation_slope

    @property
    def wronskian_residual(self) -> Array:
        return self.scattering.wronskian_residual

    @property
    def flux_residual(self) -> Array:
        return self.scattering.flux_residual

    @property
    def successful(self) -> Array:
        return self.finite & self.converged & self.physically_valid


def _schwarzschild_amplitudes(
    plan: SchwarzschildScatteringSolvePlan,
    radial_plan: SchwarzschildRadialPlan,
    angular_frequency: Array,
    /,
) -> _SchwarzschildAmplitudeEvaluation:
    omega = angular_frequency.astype(jnp.result_type(angular_frequency, radial_plan.mass))
    complex_omega = omega.astype(jnp.result_type(omega, 1.0j))
    states = _integrate_schwarzschild_horizon_riccati(
        radial_plan,
        complex_omega,
        -1,
    )
    outer_logarithm = states[-1, 0]
    outer_q = states[-1, 1]
    scaled_wave = jnp.exp(1.0j * jnp.imag(outer_logarithm))
    right_hand_side = jnp.stack((scaled_wave, outer_q * scaled_wave))
    outgoing_value, outgoing_q = _schwarzschild_infinity_series(
        radial_plan,
        complex_omega,
        1,
    )
    incoming_value, incoming_q = _schwarzschild_infinity_series(
        radial_plan,
        complex_omega,
        -1,
    )
    matrix = jnp.stack(
        (
            jnp.stack((outgoing_value, incoming_value)),
            jnp.stack(
                (
                    outgoing_q * outgoing_value,
                    incoming_q * incoming_value,
                )
            ),
        )
    )
    decomposition = solve_linear(
        LinearSystem(
            DenseLinearOperator(
                matrix,
                operator_id=f"{plan.plan_id}:infinity-wave-decomposition",
            ),
            problem_id=f"{plan.plan_id}:infinity-wave-decomposition",
        ),
        right_hand_side,
        policy=plan.decomposition_policy,
    )
    source_reflected_scaled, source_incident_scaled = decomposition.value
    source_scale = jnp.exp(jnp.real(outer_logarithm))
    source_reflected = source_scale * source_reflected_scaled
    source_incident = source_scale * source_incident_scaled
    safe_incident = jnp.where(
        jnp.abs(source_incident_scaled) > 0.0,
        source_incident_scaled,
        jnp.ones_like(source_incident_scaled),
    )
    outgoing_derivative = outgoing_q * outgoing_value
    incoming_derivative = incoming_q * incoming_value
    outgoing_wronskian = jnp.conj(
        outgoing_value
    ) * outgoing_derivative - outgoing_value * jnp.conj(outgoing_derivative)
    incoming_wronskian = jnp.conj(
        incoming_value
    ) * incoming_derivative - incoming_value * jnp.conj(incoming_derivative)
    infinity_outgoing_unit_flux = 0.5 * jnp.imag(outgoing_wronskian)
    infinity_incident_unit_flux = -0.5 * jnp.imag(incoming_wronskian)
    initial_wave = jnp.exp(states[0, 0])
    initial_derivative = states[0, 1] * initial_wave
    initial_wronskian = jnp.conj(
        initial_wave
    ) * initial_derivative - initial_wave * jnp.conj(initial_derivative)
    horizon_unit_flux = -0.5 * jnp.imag(initial_wronskian)
    flux_tiny = jnp.finfo(omega.dtype).tiny
    basis_flux_valid = (
        (infinity_outgoing_unit_flux > flux_tiny)
        & (infinity_incident_unit_flux > flux_tiny)
        & (horizon_unit_flux > flux_tiny)
    )
    safe_outgoing_flux = jnp.where(
        infinity_outgoing_unit_flux > flux_tiny,
        infinity_outgoing_unit_flux,
        1.0,
    )
    safe_incident_flux = jnp.where(
        infinity_incident_unit_flux > flux_tiny,
        infinity_incident_unit_flux,
        1.0,
    )
    safe_horizon_flux = jnp.where(
        horizon_unit_flux > flux_tiny,
        horizon_unit_flux,
        1.0,
    )
    reflected = (
        source_reflected_scaled
        / safe_incident
        * jnp.sqrt(safe_outgoing_flux / safe_incident_flux)
    )
    horizon_coefficient = jnp.exp(-jnp.real(outer_logarithm)) / safe_incident
    horizon = horizon_coefficient * jnp.sqrt(safe_horizon_flux / safe_incident_flux)

    reconstructed = contract("ij,j->i", matrix, decomposition.value)
    residual_vector = reconstructed - right_hand_side
    decomposition_scale = jnp.maximum(
        jnp.maximum(jnp.max(jnp.abs(reconstructed)), jnp.max(jnp.abs(right_hand_side))),
        jnp.finfo(omega.dtype).tiny,
    )
    decomposition_residual = jnp.max(jnp.abs(residual_vector)) / decomposition_scale

    field_flux_normalization = jnp.sqrt(omega / safe_incident_flux)
    outer_wave = scaled_wave / safe_incident * field_flux_normalization
    outer_derivative = outer_q * outer_wave
    outer_wronskian = jnp.conj(outer_wave) * outer_derivative - outer_wave * jnp.conj(
        outer_derivative
    )
    inner_wave = initial_wave * horizon_coefficient * field_flux_normalization
    inner_derivative = states[0, 1] * inner_wave
    inner_wronskian = jnp.conj(inner_wave) * inner_derivative - inner_wave * jnp.conj(
        inner_derivative
    )
    wronskian_residual = outer_wronskian - inner_wronskian
    asymptotic = schwarzschild_radial_asymptotics(radial_plan, complex_omega)
    finite = (
        jnp.all(jnp.isfinite(jnp.real(states)))
        & jnp.all(jnp.isfinite(jnp.imag(states)))
        & jnp.isfinite(jnp.real(source_incident))
        & jnp.isfinite(jnp.imag(source_incident))
        & jnp.isfinite(jnp.real(source_reflected))
        & jnp.isfinite(jnp.imag(source_reflected))
        & jnp.isfinite(jnp.real(reflected))
        & jnp.isfinite(jnp.imag(reflected))
        & jnp.isfinite(jnp.real(horizon))
        & jnp.isfinite(jnp.imag(horizon))
        & jnp.isfinite(infinity_incident_unit_flux)
        & jnp.isfinite(horizon_unit_flux)
        & jnp.isfinite(decomposition_residual)
        & jnp.isfinite(jnp.real(wronskian_residual))
        & jnp.isfinite(jnp.imag(wronskian_residual))
    )
    decomposition_valid = (
        decomposition.successful
        & decomposition.diagnostics.finite
        & (decomposition_residual <= plan.decomposition_tolerance)
    )
    incident_valid = jnp.abs(source_incident) > plan.incident_amplitude_tolerance
    asymptotic_valid = asymptotic.qualified
    qualified = (
        finite
        & decomposition_valid
        & incident_valid
        & basis_flux_valid
        & asymptotic_valid
    )
    return _SchwarzschildAmplitudeEvaluation(
        omega,
        source_incident,
        source_reflected,
        reflected,
        horizon,
        infinity_incident_unit_flux,
        horizon_unit_flux,
        decomposition_residual,
        decomposition.diagnostics.condition_estimate,
        wronskian_residual,
        asymptotic,
        decomposition,
        finite,
        decomposition_valid,
        incident_valid,
        basis_flux_valid,
        asymptotic_valid,
        qualified,
        radial_plan.plan_id,
    )


def _schwarzschild_amplitude_pair(
    plan: SchwarzschildScatteringSolvePlan,
    angular_frequency: Array,
    /,
) -> _SchwarzschildAmplitudePair:
    coarse = _schwarzschild_amplitudes(plan, plan.radial_plan, angular_frequency)
    refined = _schwarzschild_amplitudes(
        plan,
        plan.refined_radial_plan,
        angular_frequency,
    )
    reflection_error = jnp.abs(refined.reflected_amplitude - coarse.reflected_amplitude)
    horizon_error = jnp.abs(refined.horizon_amplitude - coarse.horizon_amplitude)
    coarse_graybody = jnp.real(
        coarse.horizon_amplitude * jnp.conj(coarse.horizon_amplitude)
    )
    refined_graybody = jnp.real(
        refined.horizon_amplitude * jnp.conj(refined.horizon_amplitude)
    )
    graybody_error = jnp.abs(refined_graybody - coarse_graybody)
    scale = jnp.maximum(
        jnp.maximum(
            jnp.abs(refined.reflected_amplitude),
            jnp.abs(refined.horizon_amplitude),
        ),
        1.0,
    )
    threshold = plan.refinement_tolerance * scale
    finite = coarse.finite & refined.finite
    resolved = (
        finite
        & coarse.qualified
        & refined.qualified
        & (reflection_error <= threshold)
        & (horizon_error <= threshold)
        & (graybody_error <= plan.refinement_tolerance)
    )
    return _SchwarzschildAmplitudePair(
        coarse,
        refined,
        reflection_error,
        horizon_error,
        graybody_error,
        threshold,
        finite,
        resolved,
    )


def solve_schwarzschild_scattering(
    plan: SchwarzschildScatteringSolvePlan,
    angular_frequency: ArrayLike,
    /,
) -> SchwarzschildScatteringResult:
    """Compute scalar Schwarzschild scattering and a step-refined local slope."""

    if not isinstance(plan, SchwarzschildScatteringSolvePlan):
        raise TypeError("plan must be a SchwarzschildScatteringSolvePlan.")
    requested_frequency = _real_scalar(angular_frequency, "angular_frequency")
    frequency_finite = jnp.isfinite(requested_frequency)
    frequency_valid = frequency_finite & (requested_frequency > plan.frequency_step)
    frequency = jnp.where(
        frequency_valid,
        requested_frequency,
        jnp.asarray(2.0 * plan.frequency_step, dtype=requested_frequency.dtype),
    )
    half_step = 0.5 * plan.frequency_step
    minus_frequency = frequency - plan.frequency_step
    minus_half_frequency = frequency - half_step
    plus_half_frequency = frequency + half_step
    plus_frequency = frequency + plan.frequency_step

    central = _schwarzschild_amplitude_pair(plan, frequency)
    minus = _schwarzschild_amplitude_pair(plan, minus_frequency)
    minus_half = _schwarzschild_amplitude_pair(plan, minus_half_frequency)
    plus_half = _schwarzschild_amplitude_pair(plan, plus_half_frequency)
    plus = _schwarzschild_amplitude_pair(plan, plus_frequency)
    pairs = (central, minus, minus_half, plus_half, plus)
    neighboring_pairs = (minus, minus_half, plus_half, plus)

    def graybody(pair: _SchwarzschildAmplitudePair) -> Array:
        amplitude = pair.refined.horizon_amplitude
        return jnp.real(amplitude * jnp.conj(amplitude))

    def amplitude_ledger_valid(
        evaluation: _SchwarzschildAmplitudeEvaluation,
    ) -> Array:
        ledger = solve_black_hole_scattering(
            plan.flux_plan,
            evaluation.angular_frequency,
            jnp.asarray(1.0 + 0.0j, dtype=evaluation.reflected_amplitude.dtype),
            evaluation.reflected_amplitude,
            evaluation.horizon_amplitude,
            evaluation.wronskian_residual,
            jnp.asarray(0.0, dtype=evaluation.angular_frequency.dtype),
            radial_source_id=f"{plan.plan_id}:amplitude-ledger",
        )
        return (
            evaluation.qualified
            & ledger.converged
            & ledger.physically_valid
            & (jnp.abs(ledger.wronskian_residual) <= plan.absolute_flux_tolerance)
            & (jnp.abs(ledger.flux_residual) <= plan.absolute_flux_tolerance)
        )

    pair_ledger_valid = jnp.asarray(
        tuple(
            amplitude_ledger_valid(pair.coarse) & amplitude_ledger_valid(pair.refined)
            for pair in pairs
        )
    )
    neighboring_ledger_valid = pair_ledger_valid[1:]

    central_graybody = graybody(central)
    minus_graybody = graybody(minus)
    minus_half_graybody = graybody(minus_half)
    plus_half_graybody = graybody(plus_half)
    plus_graybody = graybody(plus)
    coarse_corotation_slope = (plus_graybody - minus_graybody) / (
        2.0 * plan.frequency_step
    )
    corotation_slope = (plus_half_graybody - minus_half_graybody) / plan.frequency_step
    mass = plan.radial_plan.mass
    coarse_dimensionless_slope = coarse_corotation_slope / mass
    dimensionless_slope = corotation_slope / mass
    slope_refinement_error = jnp.abs(dimensionless_slope - coarse_dimensionless_slope)
    slope_refinement_threshold = plan.slope_refinement_tolerance * jnp.maximum(
        jnp.abs(dimensionless_slope),
        1.0,
    )
    neighboring_resolved = jnp.asarray(tuple(pair.resolved for pair in neighboring_pairs))
    slope_refinement_valid = (
        jnp.all(neighboring_resolved)
        & jnp.all(neighboring_ledger_valid)
        & jnp.isfinite(coarse_dimensionless_slope)
        & jnp.isfinite(dimensionless_slope)
        & (slope_refinement_error <= slope_refinement_threshold)
    )

    source_id = plan.plan_id
    scattering = solve_black_hole_scattering(
        plan.flux_plan,
        requested_frequency,
        jnp.asarray(1.0 + 0.0j, dtype=central.refined.reflected_amplitude.dtype),
        central.refined.reflected_amplitude,
        central.refined.horizon_amplitude,
        central.refined.wronskian_residual,
        corotation_slope,
        radial_source_id=source_id,
    )
    absolute_wronskian_valid = (
        jnp.abs(scattering.wronskian_residual) <= plan.absolute_flux_tolerance
    )
    absolute_flux_valid = (
        jnp.abs(scattering.flux_residual) <= plan.absolute_flux_tolerance
    )
    source_finite = jnp.all(jnp.asarray(tuple(pair.finite for pair in pairs)))
    radial_refinement_valid = central.resolved & jnp.all(neighboring_resolved)
    scattering_ledger_valid = (
        scattering.converged & scattering.physically_valid & jnp.all(pair_ledger_valid)
    )
    source_converged = (
        radial_refinement_valid
        & slope_refinement_valid
        & scattering_ledger_valid
        & absolute_wronskian_valid
        & absolute_flux_valid
    )
    dimensionless_frequency = plan.radial_plan.mass * frequency
    low_frequency_control_active = (
        frequency_valid
        & (plan.radial_plan.mode.ell == 0)
        & (dimensionless_frequency <= plan.low_frequency_maximum)
    )
    low_frequency_reference = 16.0 * dimensionless_frequency**2
    low_frequency_relative_error = jnp.abs(
        central_graybody - low_frequency_reference
    ) / jnp.maximum(
        low_frequency_reference,
        jnp.finfo(frequency.dtype).tiny,
    )
    low_frequency_control_valid = (~low_frequency_control_active) | (
        low_frequency_relative_error <= plan.low_frequency_relative_tolerance
    )
    finite = (
        frequency_finite
        & scattering.finite
        & source_finite
        & jnp.isfinite(coarse_dimensionless_slope)
        & jnp.isfinite(dimensionless_slope)
        & jnp.isfinite(slope_refinement_error)
        & jnp.isfinite(low_frequency_relative_error)
    )
    converged = finite & frequency_valid & source_converged
    physically_valid = frequency_valid & scattering.physically_valid
    qualified = converged & physically_valid & low_frequency_control_valid
    derivative_valid = qualified & scattering.derivative_valid & slope_refinement_valid
    decomposition_valid = jnp.all(
        jnp.asarray(
            tuple(
                pair.coarse.decomposition_valid & pair.refined.decomposition_valid
                for pair in pairs
            )
        )
    )
    incident_valid = jnp.all(
        jnp.asarray(
            tuple(
                pair.coarse.incident_valid & pair.refined.incident_valid for pair in pairs
            )
        )
    )
    asymptotic_valid = jnp.all(
        jnp.asarray(
            tuple(
                pair.coarse.asymptotic_valid & pair.refined.asymptotic_valid
                for pair in pairs
            )
        )
    )
    basis_flux_valid = jnp.all(
        jnp.asarray(
            tuple(
                pair.coarse.basis_flux_valid & pair.refined.basis_flux_valid
                for pair in pairs
            )
        )
    )

    status = jnp.asarray(int(SchwarzschildScatteringStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        ~frequency_finite,
        int(SchwarzschildScatteringStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~frequency_valid,
        int(SchwarzschildScatteringStatus.INVALID_FREQUENCY),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~finite,
        int(SchwarzschildScatteringStatus.NONFINITE),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~decomposition_valid,
        int(SchwarzschildScatteringStatus.DECOMPOSITION_FAILED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~incident_valid,
        int(SchwarzschildScatteringStatus.INCIDENT_AMPLITUDE_UNRESOLVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~basis_flux_valid,
        int(SchwarzschildScatteringStatus.BASIS_FLUX_UNRESOLVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~asymptotic_valid,
        int(SchwarzschildScatteringStatus.ASYMPTOTIC_UNRESOLVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~radial_refinement_valid,
        int(SchwarzschildScatteringStatus.STEP_REFINEMENT_UNRESOLVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS))
        & ~absolute_wronskian_valid,
        int(SchwarzschildScatteringStatus.WRONSKIAN_NOT_CONSERVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~absolute_flux_valid,
        int(SchwarzschildScatteringStatus.FLUX_NOT_CONSERVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~scattering_ledger_valid,
        int(SchwarzschildScatteringStatus.SCATTERING_LEDGER_UNRESOLVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS)) & ~slope_refinement_valid,
        int(SchwarzschildScatteringStatus.SLOPE_REFINEMENT_UNRESOLVED),
        status,
    )
    status = jnp.where(
        (status == int(SchwarzschildScatteringStatus.SUCCESS))
        & ~low_frequency_control_valid,
        int(SchwarzschildScatteringStatus.LOW_FREQUENCY_CONTROL_FAILED),
        status,
    ).astype(jnp.int32)

    evidence = SchwarzschildScatteringEvidence(
        central.refined.source_incident_amplitude,
        central.refined.source_reflected_amplitude,
        central.refined.infinity_incident_unit_flux,
        central.refined.horizon_unit_flux,
        central.refined.decomposition_residual,
        central.refined.decomposition_condition,
        central.reflection_error,
        central.horizon_error,
        central.graybody_error,
        central.threshold,
        jnp.asarray(
            (
                minus_frequency,
                minus_half_frequency,
                plus_half_frequency,
                plus_frequency,
            )
        ),
        jnp.asarray(
            (
                minus_graybody,
                minus_half_graybody,
                plus_half_graybody,
                plus_graybody,
            )
        ),
        neighboring_resolved,
        coarse_dimensionless_slope,
        slope_refinement_error,
        neighboring_ledger_valid,
        slope_refinement_threshold,
        slope_refinement_valid,
        low_frequency_reference,
        low_frequency_relative_error,
        jnp.asarray(low_frequency_control_active),
        low_frequency_control_valid,
        source_finite,
        basis_flux_valid,
        source_converged,
        scattering_ledger_valid,
        absolute_wronskian_valid,
        absolute_flux_valid,
        plan.radial_plan.plan_id,
        plan.refined_radial_plan.plan_id,
        central.refined.decomposition.provenance.plan_id,
    )
    return SchwarzschildScatteringResult(
        scattering,
        evidence,
        central.refined.asymptotic,
        finite,
        converged,
        physically_valid,
        qualified,
        derivative_valid,
        status,
        plan.radial_plan.mode.mode_id,
        source_id,
        plan.plan_id,
    )


__all__ = [
    "BlackHoleScatteringPlan",
    "BlackHoleScatteringResult",
    "BlackHoleScatteringStatus",
    "SchwarzschildScatteringEvidence",
    "SchwarzschildScatteringResult",
    "SchwarzschildScatteringSolvePlan",
    "SchwarzschildScatteringStatus",
    "ScatteringQualificationEvidence",
    "SuperradianceStatus",
    "solve_black_hole_scattering",
    "solve_schwarzschild_scattering",
]
