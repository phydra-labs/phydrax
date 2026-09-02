#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._background import FLRWBackground
from ._closure import ScientificArtifactEnvelope
from ._cmb import CmbSpectrumTable
from ._parity import ParityProfile
from ._products import (
    CosmologyProductProvenance,
    LinearTransferDescriptor,
    LinearTransferTable,
    ThermodynamicsHistory,
)


class NativeThermodynamicsResult(StrictModule):
    scale_factors: Array
    ionization_fraction: Array
    baryon_temperature: Array
    opacity_derivative: Array
    optical_depth: Array
    visibility: Array
    finite: Array
    successful: Array


class ThermodynamicsRateTable(StrictModule, NonTrainableState):
    scale_factors: Array
    recombination_rate: Array
    ionization_rate: Array
    compton_rate: Array
    photon_temperature: Array
    artifact: ScientificArtifactEnvelope
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factors: ArrayLike,
        recombination_rate: ArrayLike,
        ionization_rate: ArrayLike,
        compton_rate: ArrayLike,
        photon_temperature: ArrayLike,
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        scale = jax.lax.stop_gradient(jnp.asarray(scale_factors))
        values = tuple(
            jax.lax.stop_gradient(jnp.asarray(value, dtype=scale.dtype))
            for value in (
                recombination_rate,
                ionization_rate,
                compton_rate,
                photon_temperature,
            )
        )
        if (
            scale.ndim != 1
            or scale.size < 3
            or any(value.shape != scale.shape for value in values)
        ):
            raise ValueError(
                "Thermodynamics rate arrays must share an increasing scale-factor grid."
            )
        scale = eqx.error_if(
            scale,
            jnp.any(~jnp.isfinite(scale))
            | jnp.any(scale <= 0.0)
            | jnp.any(jnp.diff(scale) <= 0.0)
            | jnp.any(~jnp.isfinite(jnp.stack(values)))
            | jnp.any(jnp.stack(values) < 0.0),
            "Thermodynamics rate table must be finite and non-negative.",
        )
        self.scale_factors = scale
        (
            self.recombination_rate,
            self.ionization_rate,
            self.compton_rate,
            self.photon_temperature,
        ) = values
        self.artifact = artifact
        self.table_id = canonical_fingerprint(
            {
                "kind": "native-thermodynamics-rate-table",
                "artifact": artifact.artifact_id,
                "arrays": array_tree_fingerprint((scale, *values)),
            }
        )

    def evaluate(self, scale_factor: Array, /) -> tuple[Array, Array, Array, Array]:
        return tuple(
            jnp.interp(scale_factor, self.scale_factors, value)
            for value in (
                self.recombination_rate,
                self.ionization_rate,
                self.compton_rate,
                self.photon_temperature,
            )
        )


class NativeThermodynamicsPlan(StrictModule, NonTrainableState):
    rate_table: ThermodynamicsRateTable
    hydrogen_number_density_today: float = eqx.field(static=True)
    thomson_cross_section: float = eqx.field(static=True)
    speed_of_light: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        rate_table: ThermodynamicsRateTable,
        /,
        *,
        hydrogen_number_density_today: float,
        thomson_cross_section: float,
        speed_of_light: float,
    ):
        density = float(hydrogen_number_density_today)
        sigma = float(thomson_cross_section)
        speed = float(speed_of_light)
        if any(
            not np.isfinite(value) or value <= 0.0 for value in (density, sigma, speed)
        ):
            raise ValueError(
                "Thermodynamics physical constants must be finite and positive."
            )
        self.rate_table = rate_table
        self.hydrogen_number_density_today = density
        self.thomson_cross_section = sigma
        self.speed_of_light = speed
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-recombination-thermodynamics",
                "rate_table": rate_table.table_id,
                "hydrogen_number_density_today": density,
                "thomson_cross_section": sigma,
                "speed_of_light": speed,
            }
        )

    def solve(
        self,
        hubble_values: ArrayLike,
        /,
        *,
        initial_ionization_fraction: float = 1.0,
        initial_baryon_temperature: float | None = None,
    ) -> NativeThermodynamicsResult:
        scale = self.rate_table.scale_factors
        hubble = jnp.asarray(hubble_values, dtype=scale.dtype)
        if hubble.shape != scale.shape:
            raise ValueError("Thermodynamics Hubble values must match scale factors.")
        initial_temperature = (
            self.rate_table.photon_temperature[0]
            if initial_baryon_temperature is None
            else jnp.asarray(initial_baryon_temperature, dtype=scale.dtype)
        )
        initial = jnp.stack(
            (
                jnp.asarray(initial_ionization_fraction, dtype=scale.dtype),
                initial_temperature,
            )
        )

        def step(state, interval):
            start, end = interval
            midpoint = 0.5 * (start + end)
            delta = end - start
            hubble_mid = jnp.interp(midpoint, scale, hubble)
            recombination, ionization, compton, photon_temperature = (
                self.rate_table.evaluate(midpoint)
            )

            def rate(value):
                electron, temperature = value
                density = self.hydrogen_number_density_today / midpoint**3
                electron_rate = (
                    ionization * (1.0 - electron) - recombination * density * electron**2
                ) / (midpoint * hubble_mid)
                temperature_rate = -2.0 * temperature / midpoint + compton * electron * (
                    photon_temperature - temperature
                ) / (midpoint * hubble_mid)
                return jnp.stack((electron_rate, temperature_rate))

            first = rate(state)
            midpoint_state = state + 0.5 * delta * first
            candidate = state + delta * rate(midpoint_state)
            candidate = candidate.at[0].set(jnp.clip(candidate[0], 0.0, 1.0))
            candidate = candidate.at[1].set(jnp.maximum(candidate[1], 0.0))
            return candidate, candidate

        intervals = jnp.stack((scale[:-1], scale[1:]), axis=-1)
        _, history = jax.lax.scan(step, initial, intervals)
        states = jnp.concatenate((initial[None, :], history), axis=0)
        electron = states[:, 0]
        temperature = states[:, 1]
        density = self.hydrogen_number_density_today / scale**3
        opacity = (
            self.speed_of_light
            * self.thomson_cross_section
            * density
            * electron
            / (scale * hubble)
        )
        reverse_increment = 0.5 * (opacity[:-1] + opacity[1:]) * jnp.diff(scale)
        optical_depth = jnp.concatenate(
            (
                jnp.cumsum(reverse_increment[::-1])[::-1],
                jnp.zeros((1,), dtype=scale.dtype),
            )
        )
        visibility = opacity * jnp.exp(-optical_depth)
        finite = jnp.all(
            jnp.isfinite(
                jnp.stack((electron, temperature, opacity, optical_depth, visibility))
            )
        )
        return NativeThermodynamicsResult(
            scale,
            electron,
            temperature,
            opacity,
            optical_depth,
            visibility,
            finite,
            finite,
        )


class ScalarHierarchyLayout(StrictModule, NonTrainableState):
    photon_order: int = eqx.field(static=True)
    polarization_order: int = eqx.field(static=True)
    relic_order: int = eqx.field(static=True)
    component_names: tuple[str, ...] = eqx.field(static=True)
    state_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        photon_order: int = 16,
        polarization_order: int = 16,
        relic_order: int = 16,
    ):
        orders = tuple(
            int(value) for value in (photon_order, polarization_order, relic_order)
        )
        if any(value < 2 for value in orders):
            raise ValueError("Scalar hierarchy orders must be at least two.")
        names = (
            "metric_h",
            "metric_eta",
            "delta_cdm",
            "delta_baryon",
            "theta_baryon",
            *(f"photon_F_{ell}" for ell in range(orders[0] + 1)),
            *(f"photon_G_{ell}" for ell in range(orders[1] + 1)),
            *(f"massless_relic_F_{ell}" for ell in range(orders[2] + 1)),
        )
        self.photon_order, self.polarization_order, self.relic_order = orders
        self.component_names = names
        self.state_size = len(names)
        self.layout_id = canonical_fingerprint(
            {
                "kind": "scalar-hierarchy-layout",
                "orders": list(orders),
                "components": list(names),
            }
        )


class ApproximationTransitionPolicy(StrictModule, NonTrainableState):
    tight_coupling_exit: float = eqx.field(static=True)
    radiation_streaming_entry: float = eqx.field(static=True)
    overlap_tolerance: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        tight_coupling_exit: float,
        radiation_streaming_entry: float,
        overlap_tolerance: float,
    ):
        values = tuple(
            float(value)
            for value in (
                tight_coupling_exit,
                radiation_streaming_entry,
                overlap_tolerance,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Approximation transition policy is invalid.")
        (
            self.tight_coupling_exit,
            self.radiation_streaming_entry,
            self.overlap_tolerance,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "boltzmann-approximation-transitions",
                "tight_coupling_exit": values[0],
                "radiation_streaming_entry": values[1],
                "overlap_tolerance": values[2],
            }
        )

    def phases(self, scale_factors: ArrayLike, /) -> Array:
        scale = jnp.asarray(scale_factors)
        return jnp.where(
            scale < self.tight_coupling_exit,
            0,
            jnp.where(scale < self.radiation_streaming_entry, 1, 2),
        )


class ScalarEvolutionOperatorTable(StrictModule, NonTrainableState):
    scale_factors: Array
    wavenumbers: Array
    matrices: Array
    source_vectors: Array
    layout: ScalarHierarchyLayout
    artifact: ScientificArtifactEnvelope
    table_id: str = eqx.field(static=True)

    def __init__(
        self,
        scale_factors: ArrayLike,
        wavenumbers: ArrayLike,
        matrices: ArrayLike,
        source_vectors: ArrayLike,
        layout: ScalarHierarchyLayout,
        artifact: ScientificArtifactEnvelope,
        /,
    ):
        scale = jax.lax.stop_gradient(jnp.asarray(scale_factors))
        k = jax.lax.stop_gradient(jnp.asarray(wavenumbers, dtype=scale.dtype))
        matrix = jax.lax.stop_gradient(jnp.asarray(matrices, dtype=scale.dtype))
        source = jax.lax.stop_gradient(jnp.asarray(source_vectors, dtype=scale.dtype))
        expected_matrix = (scale.size, k.size, layout.state_size, layout.state_size)
        expected_source = (scale.size, k.size, layout.state_size)
        if (
            scale.ndim != 1
            or k.ndim != 1
            or matrix.shape != expected_matrix
            or source.shape != expected_source
            or scale.size < 2
            or k.size < 1
        ):
            raise ValueError("Scalar evolution operator table shapes are invalid.")
        matrix = eqx.error_if(
            matrix,
            jnp.any(~jnp.isfinite(scale))
            | jnp.any(jnp.diff(scale) <= 0.0)
            | jnp.any(~jnp.isfinite(k))
            | jnp.any(k <= 0.0)
            | jnp.any(~jnp.isfinite(matrix))
            | jnp.any(~jnp.isfinite(source)),
            "Scalar evolution table must be finite with ordered coordinates.",
        )
        self.scale_factors = scale
        self.wavenumbers = k
        self.matrices = matrix
        self.source_vectors = source
        self.layout = layout
        self.artifact = artifact
        self.table_id = canonical_fingerprint(
            {
                "kind": "scalar-evolution-operator-table",
                "layout": layout.layout_id,
                "artifact": artifact.artifact_id,
                "shape": list(matrix.shape),
            }
        )


class ScalarTransferResult(StrictModule):
    scale_factors: Array
    wavenumbers: Array
    states: Array
    transition_phases: Array
    finite: Array
    successful: Array


class RestrictedScalarTransferPlan(StrictModule, NonTrainableState):
    operators: ScalarEvolutionOperatorTable
    transitions: ApproximationTransitionPolicy
    profile: ParityProfile
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        operators: ScalarEvolutionOperatorTable,
        transitions: ApproximationTransitionPolicy,
        profile: ParityProfile,
        /,
    ):
        if (
            profile.geometry != "flat-FLRW"
            or "scalar-adiabatic" not in profile.approximations
        ):
            raise ValueError(
                "Restricted scalar transfer requires a flat scalar-adiabatic profile."
            )
        self.operators = operators
        self.transitions = transitions
        self.profile = profile
        self.plan_id = canonical_fingerprint(
            {
                "kind": "restricted-native-scalar-transfer",
                "operators": operators.table_id,
                "transitions": transitions.policy_id,
                "profile": profile.profile_id,
            }
        )

    def solve(self, initial_states: ArrayLike, /) -> ScalarTransferResult:
        initial = jnp.asarray(initial_states, dtype=self.operators.matrices.dtype)
        expected = (self.operators.wavenumbers.size, self.operators.layout.state_size)
        if initial.shape != expected:
            raise ValueError(f"Initial scalar hierarchy must have shape {expected}.")
        scale = self.operators.scale_factors
        matrix = self.operators.matrices
        source = self.operators.source_vectors

        def step(state, index):
            delta = scale[index + 1] - scale[index]
            matrix_mid = 0.5 * (matrix[index] + matrix[index + 1])
            source_mid = 0.5 * (source[index] + source[index + 1])
            first = contract("kij,kj->ki", matrix[index], state) + source[index]
            midpoint = state + 0.5 * delta * first
            candidate = state + delta * (
                contract("kij,kj->ki", matrix_mid, midpoint) + source_mid
            )
            return candidate, candidate

        _, history = jax.lax.scan(step, initial, jnp.arange(scale.size - 1))
        states = jnp.concatenate((initial[None, ...], history), axis=0)
        phases = self.transitions.phases(scale)
        finite = jnp.all(jnp.isfinite(states))
        return ScalarTransferResult(
            scale, self.operators.wavenumbers, states, phases, finite, finite
        )


class ScalarEinsteinBoltzmannEvidence(StrictModule):
    einstein_constraint_residual: Array
    tight_coupling_overlap_error: Array
    hierarchy_tail_amplitude: Array
    line_of_sight_quadrature_error: Array
    transition_schedule_valid: Array
    finite: Array
    successful: Array


class ScalarEinsteinBoltzmannResult(StrictModule):
    transfer: ScalarTransferResult
    temperature_source: Array
    polarization_source: Array
    temperature_transfer: Array
    polarization_transfer: Array
    transfer_table: LinearTransferTable
    cmb_spectra: CmbSpectrumTable
    evidence: ScalarEinsteinBoltzmannEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


def _line_of_sight_transfers(
    delta_time: Array,
    temperature_source: Array,
    polarization_source: Array,
    radial: Array,
    /,
) -> tuple[Array, Array, Array, Array]:
    trapezoid_weights = jnp.concatenate(
        (
            delta_time[:1] / 2.0,
            (delta_time[:-1] + delta_time[1:]) / 2.0,
            delta_time[-1:] / 2.0,
        )
    )
    left_weights = jnp.concatenate((delta_time, jnp.zeros((1,), dtype=delta_time.dtype)))
    temperature_transfer = contract(
        "t,tk,lkt->lk", trapezoid_weights, temperature_source, radial
    )
    polarization_transfer = contract(
        "t,tk,lkt->lk", trapezoid_weights, polarization_source, radial
    )
    left_temperature = contract("t,tk,lkt->lk", left_weights, temperature_source, radial)
    left_polarization = contract(
        "t,tk,lkt->lk", left_weights, polarization_source, radial
    )
    temperature_error = jnp.max(jnp.abs(temperature_transfer - left_temperature))
    polarization_error = jnp.max(jnp.abs(polarization_transfer - left_polarization))
    error = jnp.maximum(temperature_error, polarization_error)
    finite = jnp.isfinite(temperature_error) & jnp.isfinite(polarization_error)
    return temperature_transfer, polarization_transfer, error, finite


class ScalarEinsteinBoltzmannPlan(StrictModule, NonTrainableState):
    """Compile the bounded native flat scalar synchronous-gauge hierarchy."""

    background: FLRWBackground
    thermodynamics: ThermodynamicsHistory
    wavenumbers: Array
    layout: ScalarHierarchyLayout
    transitions: ApproximationTransitionPolicy
    multipoles: Array
    multipole_values: tuple[int, ...] = eqx.field(static=True)
    baryon_matter_fraction: float = eqx.field(static=True)
    constraint_tolerance: float = eqx.field(static=True)
    overlap_tolerance: float = eqx.field(static=True)
    tail_tolerance: float = eqx.field(static=True)
    line_of_sight_quadrature_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        background: FLRWBackground,
        thermodynamics: ThermodynamicsHistory,
        wavenumbers: ArrayLike,
        layout: ScalarHierarchyLayout,
        transitions: ApproximationTransitionPolicy,
        multipoles: ArrayLike,
        /,
        *,
        baryon_matter_fraction: float = 0.158,
        constraint_tolerance: float = 1.0e2,
        overlap_tolerance: float | None = None,
        tail_tolerance: float = 1.0,
        line_of_sight_quadrature_tolerance: float = 1.0e-2,
    ):
        if not isinstance(background, FLRWBackground):
            raise TypeError("background must be an FLRWBackground.")
        if not isinstance(thermodynamics, ThermodynamicsHistory):
            raise TypeError(
                "thermodynamics must be a provenance-bearing ThermodynamicsHistory."
            )
        if not isinstance(layout, ScalarHierarchyLayout):
            raise TypeError("layout must be a ScalarHierarchyLayout.")
        if not isinstance(transitions, ApproximationTransitionPolicy):
            raise TypeError("transitions must be an ApproximationTransitionPolicy.")
        if background.scale.scale_id != thermodynamics.scale.scale_id:
            raise ValueError("Background and thermodynamics scale identities disagree.")
        if float(np.asarray(background.curvature_density)) != 0.0:
            raise ValueError("Native scalar Einstein-Boltzmann execution is flat-FLRW.")
        k = jnp.asarray(wavenumbers, dtype=thermodynamics.scale_factors.dtype)
        ell_host = np.asarray(multipoles, dtype=int).reshape((-1,))
        baryon = float(baryon_matter_fraction)
        constraint = float(constraint_tolerance)
        overlap = (
            transitions.overlap_tolerance
            if overlap_tolerance is None
            else float(overlap_tolerance)
        )
        tail = float(tail_tolerance)
        line_of_sight = float(line_of_sight_quadrature_tolerance)
        if (
            k.ndim != 1
            or k.size < 2
            or bool(jnp.any(~jnp.isfinite(k)))
            or bool(jnp.any(k <= 0.0))
            or bool(jnp.any(jnp.diff(k) <= 0.0))
            or ell_host.size < 1
            or np.any(ell_host < 2)
            or np.any(np.diff(ell_host) <= 0)
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in (constraint, overlap, tail, line_of_sight)
            )
            or not 0.0 < baryon < 1.0
        ):
            raise ValueError(
                "Scalar Einstein-Boltzmann coordinates or policy are invalid."
            )
        self.background = background
        self.thermodynamics = thermodynamics
        self.wavenumbers = k
        self.layout = layout
        self.transitions = transitions
        self.multipoles = jnp.asarray(ell_host, dtype=jnp.int32)
        self.multipole_values = tuple(int(value) for value in ell_host)
        self.baryon_matter_fraction = baryon
        self.constraint_tolerance = constraint
        self.overlap_tolerance = overlap
        self.tail_tolerance = tail
        self.line_of_sight_quadrature_tolerance = line_of_sight
        self.plan_id = canonical_fingerprint(
            {
                "kind": "native-scalar-einstein-boltzmann",
                "background": background.model_form_id,
                "thermodynamics": thermodynamics.provenance.provenance_id,
                "layout": layout.layout_id,
                "transitions": transitions.policy_id,
                "wavenumbers": np.asarray(k).tolist(),
                "multipoles": ell_host.tolist(),
                "baryon_matter_fraction": baryon,
                "constraint_tolerance": constraint,
                "overlap_tolerance": overlap,
                "tail_tolerance": tail,
                "line_of_sight_quadrature_tolerance": line_of_sight,
            }
        )

    def prepare(self, /) -> "PreparedScalarEinsteinBoltzmann":
        return PreparedScalarEinsteinBoltzmann(self)


class PreparedScalarEinsteinBoltzmann(StrictModule):
    __hash__ = object.__hash__

    plan: ScalarEinsteinBoltzmannPlan
    conformal_times: Array
    transition_phases: Array
    radial: "FlatRadialKernelPlan"
    provenance: CosmologyProductProvenance
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: ScalarEinsteinBoltzmannPlan, /):
        scale = plan.thermodynamics.scale_factors
        inverse_conformal_rate = 1.0 / (scale**2 * plan.background.hubble(scale))
        increments = (
            0.5
            * (inverse_conformal_rate[:-1] + inverse_conformal_rate[1:])
            * jnp.diff(scale)
        )
        conformal = jnp.concatenate(
            (jnp.zeros((1,), dtype=scale.dtype), jnp.cumsum(increments))
        )
        phases = plan.transitions.phases(scale)
        maximum_ell = int(np.max(np.asarray(plan.multipoles)))
        provenance = CosmologyProductProvenance(
            producer="phydrax-native",
            producer_version="scalar-einstein-boltzmann",
            model_form_id=plan.background.model_form_id,
            request_id=plan.plan_id,
            numerical_policy_id=canonical_fingerprint(
                {
                    "kind": "fixed-scalar-hierarchy",
                    "layout": plan.layout.layout_id,
                    "transitions": plan.transitions.policy_id,
                    "constraint_tolerance": plan.constraint_tolerance,
                    "overlap_tolerance": plan.overlap_tolerance,
                    "tail_tolerance": plan.tail_tolerance,
                    "line_of_sight_quadrature_tolerance": (
                        plan.line_of_sight_quadrature_tolerance
                    ),
                }
            ),
            physics_policy_id="flat-synchronous-scalar-adiabatic-photon-polarization-massless-relic",
            scale_id=plan.background.scale.scale_id,
            source_kind="native",
            differentiation=plan.thermodynamics.provenance.differentiation,
            parent_product_ids=(plan.thermodynamics.provenance.provenance_id,),
        )
        self.plan = plan
        self.conformal_times = conformal
        self.transition_phases = phases
        self.radial = FlatRadialKernelPlan(maximum_ell)
        self.provenance = provenance
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-native-scalar-einstein-boltzmann",
                "plan": plan.plan_id,
                "provenance": provenance.provenance_id,
            }
        )

    def _indices(self, /) -> tuple[int, int, int]:
        photon = 5
        polarization = photon + self.plan.layout.photon_order + 1
        relic = polarization + self.plan.layout.polarization_order + 1
        return photon, polarization, relic

    def _initial_states(self, /) -> Array:
        plan = self.plan
        photon, polarization, relic = self._indices()
        initial_scale = plan.thermodynamics.scale_factors[0]
        radiation = jnp.maximum(plan.background.radiation_density, 1.0e-30)
        conformal = initial_scale / (
            plan.background.hubble_constant * jnp.sqrt(radiation)
        )

        def one_mode(k):
            x = k * conformal
            h = x * x
            state = jnp.zeros((plan.layout.state_size,), dtype=k.dtype)
            state = state.at[0].set(h)
            state = state.at[1].set(1.0 - h / 12.0)
            state = state.at[2].set(-0.5 * h)
            state = state.at[3].set(-0.5 * h)
            state = state.at[4].set(-k * x * x / 18.0)
            state = state.at[photon].set(-2.0 * h / 3.0)
            state = state.at[photon + 1].set(-x * h / 18.0)
            state = state.at[photon + 2].set(h / 30.0)
            state = state.at[polarization + 2].set(h / 120.0)
            state = state.at[relic].set(-2.0 * h / 3.0)
            state = state.at[relic + 1].set(-x * h / 18.0)
            state = state.at[relic + 2].set(h / 15.0)
            return state

        return jax.vmap(one_mode)(plan.wavenumbers)

    def _rate(self, scale_factor: Array, states: Array, /) -> Array:
        plan = self.plan
        layout = plan.layout
        photon, polarization, relic = self._indices()
        hubble = plan.background.hubble(scale_factor)
        conformal_hubble = scale_factor * hubble
        denominator = scale_factor**2 * hubble
        matter_fraction = plan.background.matter_fraction(scale_factor)
        radiation_fraction = plan.background.radiation_fraction(scale_factor)
        opacity = jnp.interp(
            scale_factor,
            plan.thermodynamics.scale_factors,
            plan.thermodynamics.opacity_derivative,
        )

        def one_mode(k, state):
            rate = jnp.zeros_like(state)
            cold_baryon = (1.0 - plan.baryon_matter_fraction) * state[
                2
            ] + plan.baryon_matter_fraction * state[3]
            radiation_density = 0.5 * (state[photon] + state[relic])
            hdot = 2.0 * k * k * state[1] / jnp.maximum(
                conformal_hubble, 1.0e-30
            ) - 3.0 * conformal_hubble * (
                matter_fraction * cold_baryon + radiation_fraction * radiation_density
            )
            etadot = (
                1.5
                * conformal_hubble**2
                * (
                    matter_fraction * plan.baryon_matter_fraction * state[4]
                    + radiation_fraction * k * (state[photon + 1] + state[relic + 1])
                )
                / jnp.maximum(k * k, 1.0e-30)
            )
            rate = rate.at[0].set(hdot)
            rate = rate.at[1].set(etadot)
            rate = rate.at[2].set(-0.5 * hdot)
            rate = rate.at[3].set(-state[4] - 0.5 * hdot)
            sound_speed_squared = (
                5.0e-5
                * jnp.interp(
                    scale_factor,
                    plan.thermodynamics.scale_factors,
                    plan.thermodynamics.baryon_temperature,
                )
                / jnp.maximum(plan.thermodynamics.baryon_temperature[0], 1.0e-30)
            )
            photon_velocity = 0.75 * k * state[photon + 1]
            rate = rate.at[4].set(
                -conformal_hubble * state[4]
                + sound_speed_squared * k * k * state[3]
                + opacity * (photon_velocity - state[4])
            )
            rate = rate.at[photon].set(-k * state[photon + 1] - 2.0 * hdot / 3.0)
            rate = rate.at[photon + 1].set(
                k * (state[photon] - 2.0 * state[photon + 2]) / 3.0
                - opacity
                * (state[photon + 1] - 4.0 * state[4] / jnp.maximum(3.0 * k, 1.0e-30))
            )
            quadrupole = (
                state[photon + 2] + state[polarization] + state[polarization + 2]
            ) / 8.0
            for ell in range(2, layout.photon_order + 1):
                lower = state[photon + ell - 1]
                upper = (
                    state[photon + ell + 1]
                    if ell < layout.photon_order
                    else (
                        (2.0 * ell + 1.0)
                        * state[photon + ell]
                        / jnp.maximum(k * self.conformal_times[-1], 1.0)
                        - lower
                    )
                )
                rate = rate.at[photon + ell].set(
                    k * (ell * lower - (ell + 1.0) * upper) / (2.0 * ell + 1.0)
                    - opacity * (state[photon + ell] - (quadrupole if ell == 2 else 0.0))
                )
            rate = rate.at[polarization].set(
                -k * state[polarization + 1]
                - opacity * (state[polarization] - 4.0 * quadrupole)
            )
            rate = rate.at[polarization + 1].set(
                k * (state[polarization] - 2.0 * state[polarization + 2]) / 3.0
                - opacity * state[polarization + 1]
            )
            for ell in range(2, layout.polarization_order + 1):
                lower = state[polarization + ell - 1]
                upper = (
                    state[polarization + ell + 1]
                    if ell < layout.polarization_order
                    else (
                        (2.0 * ell + 1.0)
                        * state[polarization + ell]
                        / jnp.maximum(k * self.conformal_times[-1], 1.0)
                        - lower
                    )
                )
                rate = rate.at[polarization + ell].set(
                    k * (ell * lower - (ell + 1.0) * upper) / (2.0 * ell + 1.0)
                    - opacity
                    * (state[polarization + ell] - (quadrupole if ell == 2 else 0.0))
                )
            rate = rate.at[relic].set(-k * state[relic + 1] - 2.0 * hdot / 3.0)
            rate = rate.at[relic + 1].set(
                k * (state[relic] - 2.0 * state[relic + 2]) / 3.0
            )
            for ell in range(2, layout.relic_order + 1):
                lower = state[relic + ell - 1]
                upper = (
                    state[relic + ell + 1]
                    if ell < layout.relic_order
                    else (
                        (2.0 * ell + 1.0)
                        * state[relic + ell]
                        / jnp.maximum(k * self.conformal_times[-1], 1.0)
                        - lower
                    )
                )
                rate = rate.at[relic + ell].set(
                    k * (ell * lower - (ell + 1.0) * upper) / (2.0 * ell + 1.0)
                )
            return rate / denominator

        return jax.vmap(one_mode)(plan.wavenumbers, states)

    def solve(self, primordial_power: ArrayLike, /) -> ScalarEinsteinBoltzmannResult:
        plan = self.plan
        scale = plan.thermodynamics.scale_factors
        primordial = jnp.asarray(primordial_power, dtype=scale.dtype)
        if primordial.shape != plan.wavenumbers.shape:
            raise ValueError(
                "Primordial scalar power must match the prepared wavenumber grid."
            )
        initial = self._initial_states()

        def step(state, index):
            start = scale[index]
            end = scale[index + 1]
            delta = end - start
            first = self._rate(start, state)
            midpoint = state + 0.5 * delta * first
            candidate = state + delta * self._rate(0.5 * (start + end), midpoint)
            return candidate, candidate

        _, history = jax.lax.scan(step, initial, jnp.arange(scale.size - 1))
        states = jnp.concatenate((initial[None, ...], history), axis=0)
        finite_states = jnp.all(jnp.isfinite(states))
        transfer = ScalarTransferResult(
            scale,
            plan.wavenumbers,
            states,
            self.transition_phases,
            finite_states,
            finite_states,
        )
        rates = jax.vmap(
            lambda a, state: self._rate(a, state),
            in_axes=(0, 0),
        )(scale, states)
        photon, polarization, relic = self._indices()
        visibility = plan.thermodynamics.visibility[:, None]
        opacity = plan.thermodynamics.opacity_derivative[:, None]
        optical_increment = (
            opacity
            * jnp.concatenate((jnp.diff(scale), jnp.zeros((1,), dtype=scale.dtype)))[
                :, None
            ]
        )
        attenuation = jnp.exp(-jnp.cumsum(optical_increment[::-1], axis=0)[::-1])
        temperature_source_tk = (
            visibility
            * (
                0.25 * states[:, :, photon]
                + states[:, :, 1]
                + 0.75 * states[:, :, photon + 2]
            )
            + 0.5 * attenuation * rates[:, :, 0]
        )
        polarization_source_tk = (
            visibility
            * (
                states[:, :, photon + 2]
                + states[:, :, polarization]
                + states[:, :, polarization + 2]
            )
            / 8.0
        )
        argument = (
            plan.wavenumbers[:, None]
            * (self.conformal_times[-1] - self.conformal_times)[None, :]
        )
        radial = self.radial.evaluate(argument)[plan.multipoles]
        delta_time = jnp.diff(self.conformal_times)
        (
            temperature_transfer,
            polarization_transfer,
            los_error,
            los_finite,
        ) = _line_of_sight_transfers(
            delta_time,
            temperature_source_tk,
            polarization_source_tk,
            radial,
        )
        delta_log_k = jnp.diff(jnp.log(plan.wavenumbers))
        k_weights = jnp.concatenate(
            (
                delta_log_k[:1] / 2.0,
                (delta_log_k[:-1] + delta_log_k[1:]) / 2.0,
                delta_log_k[-1:] / 2.0,
            )
        )
        tt = (
            4.0
            * jnp.pi
            * contract(
                "k,k,lk,lk->l",
                k_weights,
                primordial,
                temperature_transfer,
                temperature_transfer,
            )
        )
        te = (
            4.0
            * jnp.pi
            * contract(
                "k,k,lk,lk->l",
                k_weights,
                primordial,
                temperature_transfer,
                polarization_transfer,
            )
        )
        ee = (
            4.0
            * jnp.pi
            * contract(
                "k,k,lk,lk->l",
                k_weights,
                primordial,
                polarization_transfer,
                polarization_transfer,
            )
        )
        spectra_values = jnp.zeros((1, plan.multipoles.size, 4, 4), dtype=states.dtype)
        spectra_values = spectra_values.at[0, :, 0, 0].set(tt)
        spectra_values = spectra_values.at[0, :, 0, 1].set(te)
        spectra_values = spectra_values.at[0, :, 1, 0].set(te)
        spectra_values = spectra_values.at[0, :, 1, 1].set(ee)
        descriptor = LinearTransferDescriptor(
            ("cold_baryon", "total_matter"),
            gauge="synchronous",
            normalization="unit-primordial-curvature",
        )
        cold_baryon = (1.0 - plan.baryon_matter_fraction) * states[
            :, :, 2
        ] + plan.baryon_matter_fraction * states[:, :, 3]
        total_matter = plan.background.matter_fraction(scale)[
            :, None
        ] * cold_baryon + plan.background.radiation_fraction(scale)[:, None] * 0.5 * (
            states[:, :, photon] + states[:, :, relic]
        )
        transfer_table = LinearTransferTable(
            scale,
            plan.wavenumbers,
            jnp.stack((cold_baryon, total_matter), axis=0),
            descriptor,
            plan.background.scale,
            self.provenance,
            plan.thermodynamics.realization,
        )
        cmb_spectra = CmbSpectrumTable(
            plan.multipole_values,
            spectra_values,
            ("scalar",),
            self.provenance,
            plan.thermodynamics.realization,
            lensing_state="unlensed",
            nonlinear_source_id="none",
        )
        hubble_conformal = scale * plan.background.hubble(scale)
        radiation = 0.5 * (states[:, :, photon] + states[:, :, relic])
        constraint = (
            plan.wavenumbers[None, :] ** 2 * states[:, :, 1]
            - 0.5 * hubble_conformal[:, None] * rates[:, :, 0]
            - 1.5
            * hubble_conformal[:, None] ** 2
            * (
                plan.background.matter_fraction(scale)[:, None] * cold_baryon
                + plan.background.radiation_fraction(scale)[:, None] * radiation
            )
        )
        constraint_residual = jnp.max(jnp.abs(constraint)) / jnp.maximum(
            jnp.max(jnp.abs(plan.wavenumbers[None, :] ** 2 * states[:, :, 1])),
            1.0,
        )
        tight_relation = states[:, :, photon + 1] - 4.0 * states[:, :, 4] / (
            3.0 * plan.wavenumbers[None, :]
        )
        overlap_error = jnp.max(
            jnp.where(
                self.transition_phases[:, None] <= 1,
                jnp.abs(tight_relation),
                0.0,
            )
        )
        tail_amplitude = jnp.max(
            jnp.maximum(
                jnp.abs(states[:, :, photon + plan.layout.photon_order]),
                jnp.maximum(
                    jnp.abs(states[:, :, polarization + plan.layout.polarization_order]),
                    jnp.abs(states[:, :, relic + plan.layout.relic_order]),
                ),
            )
        )
        finite = (
            finite_states
            & jnp.all(jnp.isfinite(temperature_transfer))
            & jnp.all(jnp.isfinite(polarization_transfer))
            & jnp.all(jnp.isfinite(spectra_values))
            & los_finite
        )
        schedule_valid = jnp.all(self.transition_phases == plan.transitions.phases(scale))
        successful = (
            finite
            & schedule_valid
            & (constraint_residual <= plan.constraint_tolerance)
            & (overlap_error <= plan.overlap_tolerance)
            & (tail_amplitude <= plan.tail_tolerance)
            & (los_error <= plan.line_of_sight_quadrature_tolerance)
        )
        evidence = ScalarEinsteinBoltzmannEvidence(
            constraint_residual,
            overlap_error,
            tail_amplitude,
            los_error,
            schedule_valid,
            finite,
            successful,
        )
        return ScalarEinsteinBoltzmannResult(
            transfer=transfer,
            temperature_source=jnp.swapaxes(temperature_source_tk, 0, 1),
            polarization_source=jnp.swapaxes(polarization_source_tk, 0, 1),
            temperature_transfer=temperature_transfer,
            polarization_transfer=polarization_transfer,
            transfer_table=transfer_table,
            cmb_spectra=cmb_spectra,
            evidence=evidence,
            successful=successful,
            plan_id=self.prepared_id,
        )


class FlatRadialKernelPlan(StrictModule, NonTrainableState):
    maximum_multipole: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, maximum_multipole: int, /):
        maximum = int(maximum_multipole)
        if maximum < 2:
            raise ValueError("Maximum radial multipole must be at least two.")
        self.maximum_multipole = maximum
        self.plan_id = canonical_fingerprint(
            {"kind": "flat-spherical-bessel-kernels", "maximum_multipole": maximum}
        )

    def evaluate(self, argument: ArrayLike, /) -> Array:
        x = jnp.asarray(argument)
        safe = jnp.where(jnp.abs(x) > 1.0e-8, x, 1.0)
        j0 = jnp.where(jnp.abs(x) > 1.0e-8, jnp.sin(x) / safe, 1.0 - x**2 / 6.0)
        j1 = jnp.where(
            jnp.abs(x) > 1.0e-5,
            jnp.sin(x) / safe**2 - jnp.cos(x) / safe,
            x / 3.0 - x**3 / 30.0,
        )
        values = [j0, j1]
        previous, current = j0, j1
        for ell in range(1, self.maximum_multipole):
            following = (2.0 * ell + 1.0) / safe * current - previous
            following = jnp.where(jnp.abs(x) > 1.0e-5, following, 0.0)
            values.append(following)
            previous, current = current, following
        return jnp.stack(values, axis=0)


class LineOfSightSpectraResult(StrictModule):
    multipoles: Array
    transfer: Array
    spectra: Array
    finite: Array
    successful: Array


class LineOfSightSpectraPlan(StrictModule, NonTrainableState):
    radial: FlatRadialKernelPlan
    multipoles: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, radial: FlatRadialKernelPlan, multipoles: ArrayLike, /):
        ell = np.asarray(multipoles, dtype=int).reshape((-1,))
        if ell.size < 1 or np.any(ell < 2) or np.any(ell > radial.maximum_multipole):
            raise ValueError("Line-of-sight multipoles are invalid.")
        self.radial = radial
        self.multipoles = jnp.asarray(ell)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "flat-line-of-sight-spectra",
                "radial": radial.plan_id,
                "multipoles": ell.tolist(),
            }
        )

    def project(
        self,
        conformal_times: ArrayLike,
        wavenumbers: ArrayLike,
        source_values: ArrayLike,
        primordial_power: ArrayLike,
        /,
    ) -> LineOfSightSpectraResult:
        time = jnp.asarray(conformal_times)
        k = jnp.asarray(wavenumbers, dtype=time.dtype)
        source = jnp.asarray(source_values, dtype=time.dtype)
        primordial = jnp.asarray(primordial_power, dtype=time.dtype)
        if source.shape != (k.size, time.size) or primordial.shape != k.shape:
            raise ValueError("Line-of-sight source/power shapes are invalid.")
        distance = time[-1] - time
        argument = k[:, None] * distance[None, :]
        radial = self.radial.evaluate(argument)[self.multipoles]
        delta_time = jnp.diff(time)
        weights = jnp.concatenate(
            (
                delta_time[:1] / 2.0,
                (delta_time[:-1] + delta_time[1:]) / 2.0,
                delta_time[-1:] / 2.0,
            )
        )
        transfer = contract("t,kt,lkt->lk", weights, source, radial)
        log_k = jnp.log(k)
        delta_log = jnp.diff(log_k)
        k_weights = jnp.concatenate(
            (
                delta_log[:1] / 2.0,
                (delta_log[:-1] + delta_log[1:]) / 2.0,
                delta_log[-1:] / 2.0,
            )
        )
        spectra = (
            4.0
            * jnp.pi
            * contract("k,k,lk,lk->l", k_weights, primordial, transfer, transfer)
        )
        finite = jnp.all(jnp.isfinite(transfer)) & jnp.all(jnp.isfinite(spectra))
        return LineOfSightSpectraResult(
            self.multipoles, transfer, spectra, finite, finite
        )


__all__ = [
    "ApproximationTransitionPolicy",
    "FlatRadialKernelPlan",
    "LineOfSightSpectraPlan",
    "LineOfSightSpectraResult",
    "NativeThermodynamicsPlan",
    "NativeThermodynamicsResult",
    "PreparedScalarEinsteinBoltzmann",
    "RestrictedScalarTransferPlan",
    "ScalarEvolutionOperatorTable",
    "ScalarHierarchyLayout",
    "ScalarEinsteinBoltzmannEvidence",
    "ScalarEinsteinBoltzmannPlan",
    "ScalarEinsteinBoltzmannResult",
    "ScalarTransferResult",
    "ThermodynamicsRateTable",
]
