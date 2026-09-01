#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._closure import ScientificArtifactEnvelope
from ._parity import ParityProfile


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
    "RestrictedScalarTransferPlan",
    "ScalarEvolutionOperatorTable",
    "ScalarHierarchyLayout",
    "ScalarTransferResult",
    "ThermodynamicsRateTable",
]
