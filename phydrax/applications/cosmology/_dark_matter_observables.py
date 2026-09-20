#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native, fixed-shape analysis products for dark-matter realizations.

The products in this module analyze states owned by the wave, particle, SIDM, and
mixed-matter runtimes.  They do not introduce an alternate dynamics state or an
alternate Fourier/particle transfer implementation.  Spectra delegate to
:class:`PeriodicFourierShellPlan`, particle fields delegate to
:class:`PreparedParticleGridSplat`, and observation/radial-shell/lensing handoffs
return typed products composed from their existing owners.
"""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ScientificArtifactEnvelope
from ...discretization.particle import ParticleDiscretization
from ...discretization.spectral import PeriodicFourierShellPlan
from ...discretization.splatting import PreparedParticleGridSplat
from ...observation import LinearObservationPlan, TheoryVector
from ._halo_finder import FoFFinderResult, PeriodicFoFFinderPlan
from ._mixed_matter import MixedCosmologyDiagnostics
from ._nonlinear_closure import LensingPlanePlan, LightConePlan, LightConeResult
from ._particles import CosmologicalParticleState
from ._sidm import SIDMCollisionDiagnostics
from ._sidm_gravothermal import GravothermalSIDMDiagnostics
from ._sidm_weighted import (
    WeightedSIDMCollisionDiagnostics,
    WeightedSIDMPacketState,
)
from ._simulation_products import ParticleSimulationSnapshot
from ._wave_dark_matter import PreparedPeriodicWaveDarkMatter, WaveDarkMatterState


DarkMatterDensityConvention: TypeAlias = Literal[
    "additive-field", "total-density-contrast"
]


def _identifier(value: str, name: str, /) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be a non-empty identifier.")
    return normalized


def _real_array(value: ArrayLike, name: str, /) -> Array:
    array = jnp.asarray(value)
    if jnp.issubdtype(array.dtype, jnp.complexfloating):
        raise TypeError(f"{name} must be real.")
    return jnp.asarray(array, dtype=jnp.result_type(array.dtype, jnp.float32))


def _scalar_flag(value: ArrayLike, name: str, /) -> Array:
    flag = jnp.asarray(value, dtype=jnp.bool_)
    if flag.shape != ():
        raise ValueError(f"{name} must be scalar.")
    return flag


def _wrapped_angle(value: Array, /) -> Array:
    return jnp.arctan2(jnp.sin(value), jnp.cos(value))


class DarkMatterSpectrumProduct(StrictModule):
    """One native shell-reduced scalar or vector field spectrum."""

    wavenumbers: Array
    power: Array
    mode_counts: Array
    valid_shells: Array
    imaginary_residual: Array
    integrated_power: Array
    finite: Array
    successful: Array
    valid_shell_indices: tuple[int, ...] = eqx.field(static=True)
    observable_name: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    source_product_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def as_theory_vector(self, /) -> TheoryVector:
        indices = jnp.asarray(self.valid_shell_indices, dtype=jnp.int32)
        labels = tuple(
            f"{self.observable_name}:k-shell:{index}"
            for index in self.valid_shell_indices
        )
        from ...observation import CoordinateLayout

        return TheoryVector(
            self.power[indices],
            CoordinateLayout(labels),
            self.product_id,
        )


def _field_spectrum(
    shells: PeriodicFourierShellPlan,
    fields: tuple[Array, ...],
    /,
    *,
    observable_name: str,
    coordinate_convention: str,
    source_product_ids: tuple[str, ...],
    plan_id: str,
    valid_shell_indices: tuple[int, ...],
) -> DarkMatterSpectrumProduct:
    if not fields:
        raise ValueError("A field spectrum requires at least one scalar component.")
    statistics = tuple(shells.auto_power(shells.transform(field)) for field in fields)
    first = statistics[0]
    power = sum(
        (value.shell_values for value in statistics),
        jnp.zeros_like(first.shell_values),
    )
    integrated = sum(
        (value.total_weighted_value for value in statistics),
        jnp.zeros_like(first.total_weighted_value),
    )
    imaginary = jnp.max(
        jnp.stack(tuple(value.imaginary_residual for value in statistics))
    )
    finite = jnp.all(jnp.stack(tuple(value.finite for value in statistics))) & jnp.all(
        jnp.isfinite(power)
    )
    successful = finite & jnp.all(
        jnp.stack(tuple(value.successful for value in statistics))
    )
    product_id = canonical_fingerprint(
        {
            "kind": "dark-matter-field-spectrum",
            "observable": observable_name,
            "coordinate_convention": coordinate_convention,
            "sources": list(source_product_ids),
            "plan": plan_id,
        }
    )
    return DarkMatterSpectrumProduct(
        first.representative_wavenumbers,
        power,
        first.weighted_mode_count,
        first.valid_shells,
        imaginary,
        integrated,
        finite,
        successful,
        valid_shell_indices,
        observable_name,
        coordinate_convention,
        source_product_ids,
        plan_id,
        product_id,
    )


class RadialCoreProfileProduct(StrictModule):
    """Fixed-bin core profile with explicit center-selection evidence."""

    center: Array
    radial_edges: Array
    representative_radii: Array
    shell_density: Array
    enclosed_mass: Array
    sample_count: Array
    valid_shells: Array
    core_radius: Array
    peak_value: Array
    center_unique: Array
    core_identified: Array
    finite: Array
    successful: Array
    profile_kind: str = eqx.field(static=True)
    coordinate_convention: str = eqx.field(static=True)
    topology_differentiable: bool = eqx.field(static=True, default=False)


class VortexCirculationProduct(StrictModule):
    """Integer plaquette winding and quantized physical circulation."""

    axis_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    winding_number: Array
    circulation: Array
    supported_plaquette: Array
    total_absolute_winding: Array
    topology_present: Array
    finite: Array
    successful: Array
    circulation_convention: str = eqx.field(static=True)
    topology_differentiable: bool = eqx.field(static=True, default=False)


class WaveDarkMatterObservableProduct(StrictModule):
    """Density, peculiar mass current, phase, spectra, core, and vortices."""

    density: Array
    peculiar_mass_current: Array
    phase: Array
    phase_valid: Array
    density_spectrum: DarkMatterSpectrumProduct
    current_spectrum: DarkMatterSpectrumProduct
    phase_spectrum: DarkMatterSpectrumProduct
    core_profile: RadialCoreProfileProduct
    vortices: VortexCirculationProduct
    scale_factor: Array
    finite: Array
    successful: Array
    source_product_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class WaveDarkMatterObservablePlan(StrictModule, NonTrainableState):
    """Analysis bound to one prepared periodic wave realization and shell geometry."""

    wave: PreparedPeriodicWaveDarkMatter
    shells: PeriodicFourierShellPlan
    radial_edges: Array
    relative_phase_amplitude_floor: float = eqx.field(static=True)
    box_lengths: tuple[float, ...] = eqx.field(static=True)
    axis_pairs: tuple[tuple[int, int], ...] = eqx.field(static=True)
    valid_shell_indices: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        wave: PreparedPeriodicWaveDarkMatter,
        shells: PeriodicFourierShellPlan,
        radial_edges: ArrayLike,
        /,
        *,
        relative_phase_amplitude_floor: float = 1.0e-10,
    ):
        if not isinstance(wave, PreparedPeriodicWaveDarkMatter):
            raise TypeError("wave must be PreparedPeriodicWaveDarkMatter.")
        if not isinstance(shells, PeriodicFourierShellPlan):
            raise TypeError("shells must be PeriodicFourierShellPlan.")
        if shells.source_shape != wave.discretization.physical_shape:
            raise ValueError("Wave and Fourier-shell physical shapes disagree.")
        lengths = tuple(
            float(np.asarray(axis.length)) for axis in wave.discretization.axes
        )
        if len(lengths) != len(shells.box_lengths) or not np.allclose(
            np.asarray(lengths), np.asarray(shells.box_lengths)
        ):
            raise ValueError("Wave and Fourier-shell periodic boxes disagree.")
        edges_host = np.asarray(radial_edges, dtype=np.float64).reshape((-1,))
        maximum_radius = 0.5 * math.sqrt(sum(length * length for length in lengths))
        if (
            edges_host.size < 2
            or np.any(~np.isfinite(edges_host))
            or edges_host[0] != 0.0
            or np.any(np.diff(edges_host) <= 0.0)
            or edges_host[-1] > maximum_radius * (1.0 + 1.0e-12)
        ):
            raise ValueError(
                "Wave radial edges must start at zero, increase, and remain inside the periodic half-box radius."
            )
        floor = float(relative_phase_amplitude_floor)
        if not math.isfinite(floor) or not 0.0 < floor < 1.0:
            raise ValueError("relative_phase_amplitude_floor must lie in (0, 1).")
        rank = len(lengths)
        pairs = (
            ((0, 1),) if rank == 2 else (((0, 1), (1, 2), (2, 0)) if rank == 3 else ())
        )
        valid_shell_indices = tuple(np.flatnonzero(np.asarray(shells.valid_shells)))
        self.wave = wave
        self.shells = shells
        self.radial_edges = jax.lax.stop_gradient(jnp.asarray(edges_host))
        self.relative_phase_amplitude_floor = floor
        self.box_lengths = lengths
        self.axis_pairs = pairs
        self.valid_shell_indices = valid_shell_indices
        self.plan_id = canonical_fingerprint(
            {
                "kind": "wave-dark-matter-observable-plan",
                "wave": wave.prepared_id,
                "shells": shells.plan_id,
                "radial_edges": edges_host.tolist(),
                "relative_phase_amplitude_floor": floor,
            }
        )

    def _derivatives(self, psi: Array, /) -> tuple[Array, ...]:
        coefficients = self.wave.discretization.project(psi)
        return tuple(
            self.wave.discretization.reconstruct(
                self.wave.discretization.modal_derivative(coefficients, axis=axis),
                real_output=False,
            )
            for axis in range(len(self.box_lengths))
        )

    def _core_profile(self, density: Array, /) -> RadialCoreProfileProduct:
        flat_index = jnp.argmax(density.reshape((-1,)))
        center_index = jnp.asarray(jnp.unravel_index(flat_index, density.shape))
        coordinates = tuple(axis.nodes for axis in self.wave.discretization.axes)
        center = jnp.stack(
            tuple(
                coordinates[axis][center_index[axis]] for axis in range(len(coordinates))
            )
        )
        squared = jnp.zeros(density.shape, dtype=density.dtype)
        for axis, (nodes, length) in enumerate(
            zip(coordinates, self.box_lengths, strict=True)
        ):
            delta = jnp.abs(nodes - center[axis])
            periodic = jnp.minimum(delta, length - delta)
            shape = [1] * density.ndim
            shape[axis] = nodes.size
            squared = squared + periodic.reshape(tuple(shape)) ** 2
        radius = jnp.sqrt(squared)
        bin_count = self.radial_edges.size - 1
        indices = jnp.searchsorted(self.radial_edges, radius, side="right") - 1
        indices = jnp.where(
            jnp.isclose(radius, self.radial_edges[-1]),
            bin_count - 1,
            indices,
        )
        valid = (indices >= 0) & (indices < bin_count)
        safe = jnp.where(valid, indices, 0).reshape((-1,))
        flat_valid = valid.reshape((-1,))
        counts = (
            jnp.zeros((bin_count,), dtype=density.dtype)
            .at[safe]
            .add(flat_valid.astype(density.dtype))
        )
        shell_sum = (
            jnp.zeros((bin_count,), dtype=density.dtype)
            .at[safe]
            .add(jnp.where(flat_valid, density.reshape((-1,)), 0.0))
        )
        shell_density = shell_sum / jnp.maximum(counts, 1.0)
        shell_mass = shell_sum * self.wave.cell_volume
        enclosed = jnp.cumsum(shell_mass)
        valid_shells = counts > 0.0
        peak = jnp.max(density)
        half_peak = 0.5 * peak
        shell_peak_index = jnp.argmax(jnp.where(valid_shells, shell_density, -jnp.inf))
        first_valid_index = jnp.argmax(valid_shells.astype(jnp.int32))
        outward = jnp.arange(bin_count) > shell_peak_index
        crossed = valid_shells & outward & (shell_density <= half_peak)
        core_identified = jnp.any(crossed)
        core_index = jnp.argmax(crossed.astype(jnp.int32))
        radii = 0.5 * (self.radial_edges[:-1] + self.radial_edges[1:])
        core_radius = jnp.where(core_identified, radii[core_index], jnp.nan)
        center_unique = (jnp.sum(density == peak) == 1) & (
            shell_peak_index == first_valid_index
        )
        finite = (
            jnp.all(jnp.isfinite(shell_density))
            & jnp.all(jnp.isfinite(enclosed))
            & jnp.isfinite(peak)
        )
        return RadialCoreProfileProduct(
            center,
            self.radial_edges,
            radii,
            shell_density,
            enclosed,
            counts.astype(jnp.int32),
            valid_shells,
            core_radius,
            peak,
            center_unique,
            core_identified,
            finite,
            finite & center_unique & core_identified,
            "wave-density-peak-centered",
            "flat-periodic-comoving-cartesian",
        )

    def _vortices(
        self,
        psi: Array,
        phase: Array,
        phase_valid: Array,
        scale_factor: Array,
        /,
    ) -> VortexCirculationProduct:
        if not self.axis_pairs:
            shape = (0,) + psi.shape
            empty_integer = jnp.zeros(shape, dtype=jnp.int32)
            empty_real = jnp.zeros(shape, dtype=psi.real.dtype)
            empty_mask = jnp.zeros(shape, dtype=jnp.bool_)
            return VortexCirculationProduct(
                self.axis_pairs,
                empty_integer,
                empty_real,
                empty_mask,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
                jnp.asarray(True),
                jnp.asarray(False),
                "vortex winding requires a two- or three-dimensional periodic grid",
            )
        forward = tuple(
            _wrapped_angle(jnp.roll(phase, -1, axis=axis) - phase)
            for axis in range(phase.ndim)
        )
        windings = []
        supports = []
        for first, second in self.axis_pairs:
            loop = (
                forward[first]
                + jnp.roll(forward[second], -1, axis=first)
                - jnp.roll(forward[first], -1, axis=second)
                - forward[second]
            )
            support = (
                phase_valid
                & jnp.roll(phase_valid, -1, axis=first)
                & jnp.roll(phase_valid, -1, axis=second)
                & jnp.roll(jnp.roll(phase_valid, -1, axis=first), -1, axis=second)
            )
            winding = jnp.where(
                support,
                jnp.rint(loop / (2.0 * jnp.pi)).astype(jnp.int32),
                0,
            )
            windings.append(winding)
            supports.append(support)
        winding = jnp.stack(tuple(windings))
        supported = jnp.stack(tuple(supports))
        quantum = (
            2.0
            * jnp.pi
            * self.wave.reduced_planck_constant
            / (self.wave.boson_mass * scale_factor)
        )
        circulation = quantum * winding.astype(psi.real.dtype)
        total = jnp.sum(jnp.abs(winding))
        finite = jnp.all(jnp.isfinite(circulation))
        return VortexCirculationProduct(
            self.axis_pairs,
            winding,
            circulation,
            supported,
            total,
            total > 0,
            finite,
            finite & jnp.any(supported),
            "physical-peculiar-velocity circulation around comoving plaquettes",
        )

    def evaluate(
        self,
        state: WaveDarkMatterState,
        /,
        *,
        source_product_id: str,
    ) -> WaveDarkMatterObservableProduct:
        source = _identifier(source_product_id, "source_product_id")
        if not isinstance(state, WaveDarkMatterState):
            raise TypeError("state must be WaveDarkMatterState.")
        psi = jnp.asarray(state.psi)
        if psi.shape != self.shells.source_shape:
            raise ValueError("Wave state does not match the observable grid.")
        density = self.wave.density(state)
        derivatives = self._derivatives(psi)
        current = jnp.stack(
            tuple(
                self.wave.reduced_planck_constant
                * jnp.imag(jnp.conj(psi) * derivative)
                / state.scale_factor
                for derivative in derivatives
            ),
            axis=-1,
        )
        amplitude_squared = jnp.abs(psi) ** 2
        threshold = self.relative_phase_amplitude_floor * jnp.max(amplitude_squared)
        phase_valid = amplitude_squared > threshold
        phase = jnp.where(phase_valid, jnp.angle(psi), 0.0)
        density_spectrum = _field_spectrum(
            self.shells,
            (density,),
            valid_shell_indices=self.valid_shell_indices,
            observable_name="wave-comoving-mass-density",
            coordinate_convention="flat-periodic-comoving-cartesian",
            source_product_ids=(source,),
            plan_id=self.plan_id,
        )
        current_spectrum = _field_spectrum(
            self.shells,
            tuple(current[..., axis] for axis in range(current.shape[-1])),
            valid_shell_indices=self.valid_shell_indices,
            observable_name="wave-physical-peculiar-mass-current",
            coordinate_convention="flat-periodic-comoving-cartesian",
            source_product_ids=(source,),
            plan_id=self.plan_id,
        )
        phase_spectrum = _field_spectrum(
            self.shells,
            (phase,),
            valid_shell_indices=self.valid_shell_indices,
            observable_name="wave-principal-phase",
            coordinate_convention="principal-angle-on-flat-periodic-grid",
            source_product_ids=(source,),
            plan_id=self.plan_id,
        )
        core = self._core_profile(density)
        vortices = self._vortices(psi, phase, phase_valid, state.scale_factor)
        finite = (
            jnp.all(jnp.isfinite(density))
            & jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(phase))
            & density_spectrum.finite
            & current_spectrum.finite
            & phase_spectrum.finite
            & core.finite
            & vortices.finite
        )
        successful = (
            finite
            & density_spectrum.successful
            & current_spectrum.successful
            & phase_spectrum.successful
            & core.successful
            & vortices.successful
        )
        product_id = canonical_fingerprint(
            {
                "kind": "wave-dark-matter-observable-product",
                "source": source,
                "plan": self.plan_id,
            }
        )
        return WaveDarkMatterObservableProduct(
            density,
            current,
            phase,
            phase_valid,
            density_spectrum,
            current_spectrum,
            phase_spectrum,
            core,
            vortices,
            state.scale_factor,
            finite,
            successful,
            source,
            self.plan_id,
            product_id,
        )


class WeightedParticleStatistics(StrictModule):
    """Population moments and Kish effective sample evidence."""

    mean: Array
    covariance: Array
    normalized_weights: Array
    total_weight: Array
    effective_sample_size: Array
    effective_sample_fraction: Array
    active_count: Array
    finite: Array
    successful: Array


def weighted_particle_statistics(
    values: ArrayLike,
    weights: ArrayLike,
    active_mask: ArrayLike,
    /,
) -> WeightedParticleStatistics:
    samples = _real_array(values, "values")
    if samples.ndim == 1:
        samples = samples[:, None]
    if samples.ndim != 2:
        raise ValueError(
            "Weighted particle values must have shape (particle, component)."
        )
    weight = _real_array(weights, "weights").reshape((-1,))
    active = jnp.asarray(active_mask, dtype=jnp.bool_).reshape((-1,))
    if weight.shape != active.shape or samples.shape[0] != weight.size:
        raise ValueError("Weighted particle values, weights, and mask must align.")
    sample_finite = jnp.all(jnp.isfinite(samples), axis=-1)
    weight_valid = jnp.isfinite(weight) & (weight >= 0.0)
    invalid = active & (~weight_valid | ~sample_finite)
    usable = active & weight_valid & sample_finite & (weight > 0.0)
    safe_samples = jnp.where(usable[:, None], samples, 0.0)
    safe_weight = jnp.where(usable, weight, 0.0)
    total = jnp.sum(safe_weight)
    normalizer = jnp.where(total > 0.0, total, 1.0)
    normalized = safe_weight / normalizer
    mean = ein.contract("n,ni->i", normalized, safe_samples)
    centered = safe_samples - mean
    covariance = ein.contract(
        "n,ni,nj->ij",
        normalized,
        centered,
        centered,
    )
    squared_sum = jnp.sum(normalized * normalized)
    effective = 1.0 / jnp.maximum(
        squared_sum,
        jnp.finfo(normalized.dtype).tiny,
    )
    count = jnp.sum(usable.astype(jnp.int32))
    fraction = effective / jnp.maximum(count, 1)
    finite = (
        ~jnp.any(invalid)
        & jnp.isfinite(total)
        & jnp.all(jnp.isfinite(mean))
        & jnp.all(jnp.isfinite(covariance))
        & jnp.isfinite(effective)
    )
    return WeightedParticleStatistics(
        mean,
        covariance,
        normalized,
        total,
        effective,
        fraction,
        count,
        finite,
        finite & (total > 0.0) & (count > 0),
    )


class WeightedSIDMPacketObservableProduct(StrictModule):
    """Multiplicity- and mass-weighted packet moments with identity evidence."""

    multiplicity_statistics: WeightedParticleStatistics
    mass_statistics: WeightedParticleStatistics
    represented_particle_count: Array
    gravitational_mass: Array
    multiplicity_effective_sample_size: Array
    mass_effective_sample_size: Array
    mass_relation_valid: Array
    stable_identity_valid: Array
    finite: Array
    successful: Array


def weighted_sidm_packet_observables(
    state: WeightedSIDMPacketState,
    /,
) -> WeightedSIDMPacketObservableProduct:
    if not isinstance(state, WeightedSIDMPacketState):
        raise TypeError("state must be WeightedSIDMPacketState.")
    capacity = state.positions.shape[0]
    if (
        state.positions.ndim != 2
        or state.canonical_momenta.shape != state.positions.shape
        or any(
            value.shape != (capacity,)
            for value in (
                state.microscopic_masses,
                state.weights,
                state.gravitational_masses,
                state.active_mask,
                state.packet_ids,
                state.parent_packet_ids,
                state.lineage_depth,
            )
        )
    ):
        raise ValueError("Weighted SIDM packet arrays do not share one capacity.")
    active = state.active_mask
    safe_mass = jnp.where(active, state.gravitational_masses, 1.0)
    velocity = state.canonical_momenta / (safe_mass[:, None] * state.scale_factor)
    multiplicity = weighted_particle_statistics(velocity, state.weights, active)
    mass = weighted_particle_statistics(velocity, state.gravitational_masses, active)
    represented = jnp.sum(jnp.where(active, state.weights, 0.0))
    gravitational = jnp.sum(jnp.where(active, state.gravitational_masses, 0.0))
    expected_mass = state.microscopic_masses * state.weights
    tolerance = (
        64.0
        * jnp.finfo(state.positions.dtype).eps
        * jnp.maximum(jnp.abs(expected_mass), 1.0)
    )
    mass_relation = jnp.all(
        ~active
        | (
            (state.microscopic_masses > 0.0)
            & (state.weights > 0.0)
            & (state.gravitational_masses > 0.0)
            & (jnp.abs(state.gravitational_masses - expected_mass) <= tolerance)
        )
    )
    order = jnp.lexsort((state.packet_ids, ~active))
    ordered_active = active[order]
    ordered_ids = state.packet_ids[order]
    duplicate = (
        ordered_active[:-1] & ordered_active[1:] & (ordered_ids[:-1] == ordered_ids[1:])
    )
    identity_valid = jnp.all(~duplicate) & jnp.all(~active | (state.packet_ids >= 0))
    finite = (
        multiplicity.finite
        & mass.finite
        & jnp.isfinite(represented)
        & jnp.isfinite(gravitational)
        & jnp.isfinite(state.scale_factor)
        & (state.scale_factor > 0.0)
        & jnp.all(jnp.isfinite(state.positions) | ~active[:, None])
        & jnp.all(jnp.isfinite(state.canonical_momenta) | ~active[:, None])
    )
    successful = (
        finite
        & multiplicity.successful
        & mass.successful
        & mass_relation
        & identity_valid
    )
    return WeightedSIDMPacketObservableProduct(
        multiplicity,
        mass,
        represented,
        gravitational,
        multiplicity.effective_sample_size,
        mass.effective_sample_size,
        mass_relation,
        identity_valid,
        finite,
        successful,
    )


class ParticleAngularMomentProduct(StrictModule):
    center: Array
    mean_specific_angular_momentum: Array
    second_angular_moment: Array
    radial_specific_angular_momentum: Array
    radial_effective_sample_size: Array
    finite: Array
    successful: Array
    convention: str = eqx.field(static=True)


class ParticleGravothermalProduct(StrictModule):
    radial_edges: Array
    representative_radii: Array
    shell_density: Array
    radial_velocity_dispersion: Array
    tangential_velocity_dispersion: Array
    anisotropy: Array
    anisotropy_identified: Array
    temperature_proxy: Array
    radial_heat_flux: Array
    effective_sample_size: Array
    valid_shells: Array
    finite: Array
    successful: Array
    temperature_convention: str = eqx.field(static=True)
    heat_flux_convention: str = eqx.field(static=True)


class ParticleDarkMatterObservableProduct(StrictModule):
    density: Array
    mass_content: Array
    bulk_velocity: Array
    velocity_dispersion_tensor: Array
    populated_cells: Array
    global_statistics: WeightedParticleStatistics
    global_anisotropy: Array
    global_anisotropy_identified: Array
    core_profile: RadialCoreProfileProduct
    angular_moments: ParticleAngularMomentProduct
    gravothermal: ParticleGravothermalProduct
    scale_factor: Array
    finite: Array
    successful: Array
    source_product_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


def _minimum_image(
    positions: Array,
    center: Array,
    box_lengths: tuple[float, ...],
    /,
) -> Array:
    lengths = jnp.asarray(box_lengths, dtype=positions.dtype)
    return jnp.mod(positions - center + 0.5 * lengths, lengths) - 0.5 * lengths


class ParticleDarkMatterObservablePlan(StrictModule, NonTrainableState):
    """Mass-conservative particle fields and weighted halo-frame diagnostics."""

    transfer: PreparedParticleGridSplat
    center: Array
    radial_edges: Array
    box_lengths: tuple[float, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedParticleGridSplat,
        center: ArrayLike,
        radial_edges: ArrayLike,
        /,
    ):
        if not isinstance(transfer, PreparedParticleGridSplat):
            raise TypeError("transfer must be PreparedParticleGridSplat.")
        dimension = transfer.particles.ambient_dimension
        if dimension != 3:
            raise ValueError(
                "Particle dark-matter anisotropy and gravothermal diagnostics require three dimensions."
            )
        center_host = np.asarray(center, dtype=np.float64).reshape((-1,))
        edges_host = np.asarray(radial_edges, dtype=np.float64).reshape((-1,))
        lengths = tuple(upper - lower for lower, upper in transfer.axis_bounds)
        if (
            center_host.shape != (dimension,)
            or np.any(~np.isfinite(center_host))
            or any(length <= 0.0 or not math.isfinite(length) for length in lengths)
        ):
            raise ValueError("Particle observable center or periodic box is invalid.")
        if (
            edges_host.size < 2
            or edges_host[0] != 0.0
            or np.any(~np.isfinite(edges_host))
            or np.any(np.diff(edges_host) <= 0.0)
            or edges_host[-1] > 0.5 * min(lengths) * (1.0 + 1.0e-12)
        ):
            raise ValueError(
                "Particle radial edges must start at zero, increase, and fit the periodic half-box."
            )
        self.transfer = transfer
        self.center = jax.lax.stop_gradient(jnp.asarray(center_host))
        self.radial_edges = jax.lax.stop_gradient(jnp.asarray(edges_host))
        self.box_lengths = lengths
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-dark-matter-observable-plan",
                "transfer": transfer.prepared_id,
                "center": center_host.tolist(),
                "radial_edges": edges_host.tolist(),
            }
        )

    def _radial_geometry(self, positions: Array, /) -> tuple[Array, Array, Array]:
        relative = _minimum_image(positions, self.center, self.box_lengths)
        radius = jnp.sqrt(ein.contract("ni,ni->n", relative, relative))
        safe_radius = jnp.where(radius > 0.0, radius, 1.0)
        radial_unit = jnp.where(
            (radius > 0.0)[:, None], relative / safe_radius[:, None], 0.0
        )
        bin_count = self.radial_edges.size - 1
        indices = jnp.searchsorted(self.radial_edges, radius, side="right") - 1
        indices = jnp.where(
            jnp.isclose(radius, self.radial_edges[-1]),
            bin_count - 1,
            indices,
        )
        valid = (indices >= 0) & (indices < bin_count)
        return radius, radial_unit, jnp.where(valid, indices, 0)

    def _radial_products(
        self,
        positions: Array,
        velocities: Array,
        effective_mass: Array,
        active: Array,
        scale_factor: Array,
        statistics: WeightedParticleStatistics,
        /,
    ) -> tuple[
        RadialCoreProfileProduct,
        ParticleAngularMomentProduct,
        ParticleGravothermalProduct,
        Array,
        Array,
    ]:
        safe_positions = jnp.where(active[:, None], positions, self.center)
        safe_velocities = jnp.where(active[:, None], velocities, 0.0)
        radius, radial_unit, indices = self._radial_geometry(safe_positions)
        bin_count = self.radial_edges.size - 1
        inside = active & (radius <= self.radial_edges[-1])
        weight = jnp.where(inside, effective_mass, 0.0)
        counts = (
            jnp.zeros((bin_count,), dtype=effective_mass.dtype)
            .at[indices]
            .add(inside.astype(effective_mass.dtype))
        )
        shell_mass = (
            jnp.zeros((bin_count,), dtype=effective_mass.dtype).at[indices].add(weight)
        )
        dimension = positions.shape[-1]
        coefficient = math.pi ** (0.5 * dimension) / math.gamma(0.5 * dimension + 1.0)
        shell_volume = coefficient * (
            self.radial_edges[1:] ** dimension - self.radial_edges[:-1] ** dimension
        )
        shell_density = shell_mass / shell_volume
        enclosed = jnp.cumsum(shell_mass)
        radii = 0.5 * (self.radial_edges[:-1] + self.radial_edges[1:])
        valid_shells = shell_mass > 0.0
        peak_index = jnp.argmax(jnp.where(valid_shells, shell_density, -jnp.inf))
        peak = shell_density[peak_index]
        first_valid_index = jnp.argmax(valid_shells.astype(jnp.int32))
        outward = jnp.arange(bin_count) > peak_index
        crossed = valid_shells & outward & (shell_density <= 0.5 * peak)
        core_identified = jnp.any(crossed)
        core_index = jnp.argmax(crossed.astype(jnp.int32))
        core_radius = jnp.where(core_identified, radii[core_index], jnp.nan)
        center_unique = peak_index == first_valid_index
        core_finite = (
            jnp.all(jnp.isfinite(shell_density))
            & jnp.all(jnp.isfinite(enclosed))
            & jnp.isfinite(peak)
        )
        core = RadialCoreProfileProduct(
            self.center,
            self.radial_edges,
            radii,
            shell_density,
            enclosed,
            counts.astype(jnp.int32),
            valid_shells,
            core_radius,
            peak,
            center_unique,
            core_identified,
            core_finite,
            core_finite & core_identified,
            "particle-fixed-center",
            "flat-periodic-comoving-cartesian",
        )

        peculiar = safe_velocities - statistics.mean
        radial_velocity = ein.contract("ni,ni->n", peculiar, radial_unit)
        speed_squared = ein.contract("ni,ni->n", peculiar, peculiar)
        tangential_squared = jnp.maximum(speed_squared - radial_velocity**2, 0.0)
        safe_shell_mass = jnp.maximum(shell_mass, jnp.finfo(effective_mass.dtype).tiny)
        radial_mean = (
            jnp.zeros((bin_count,), dtype=velocities.dtype)
            .at[indices]
            .add(weight * radial_velocity)
            / safe_shell_mass
        )
        radial_variance = (
            jnp.zeros((bin_count,), dtype=velocities.dtype)
            .at[indices]
            .add(weight * (radial_velocity - radial_mean[indices]) ** 2)
            / safe_shell_mass
        )
        tangential_variance = (
            jnp.zeros((bin_count,), dtype=velocities.dtype)
            .at[indices]
            .add(weight * tangential_squared)
            / safe_shell_mass
        )
        anisotropy = jnp.where(
            radial_variance > 0.0,
            1.0 - tangential_variance / (2.0 * radial_variance),
            jnp.nan,
        )
        anisotropy_identified = valid_shells & (radial_variance > 0.0)
        temperature = (radial_variance + tangential_variance) / dimension
        specific_energy = 0.5 * speed_squared
        heat_flux = (
            jnp.zeros((bin_count,), dtype=velocities.dtype)
            .at[indices]
            .add(weight * (radial_velocity - radial_mean[indices]) * specific_energy)
            / shell_volume
        )
        shell_squared_weight = (
            jnp.zeros((bin_count,), dtype=effective_mass.dtype)
            .at[indices]
            .add(weight * weight)
        )
        shell_ess = shell_mass**2 / jnp.maximum(
            shell_squared_weight, jnp.finfo(effective_mass.dtype).tiny
        )
        gravothermal_finite = (
            jnp.all(jnp.isfinite(shell_density))
            & jnp.all(jnp.where(valid_shells, jnp.isfinite(radial_variance), True))
            & jnp.all(jnp.where(valid_shells, jnp.isfinite(tangential_variance), True))
            & jnp.all(jnp.where(anisotropy_identified, jnp.isfinite(anisotropy), True))
            & jnp.all(jnp.where(valid_shells, jnp.isfinite(temperature), True))
            & jnp.all(jnp.where(valid_shells, jnp.isfinite(heat_flux), True))
            & jnp.all(jnp.isfinite(shell_ess))
        )
        gravothermal = ParticleGravothermalProduct(
            self.radial_edges,
            radii,
            shell_density,
            radial_variance,
            tangential_variance,
            anisotropy,
            anisotropy_identified,
            temperature,
            heat_flux,
            shell_ess,
            valid_shells,
            gravothermal_finite,
            gravothermal_finite & jnp.any(valid_shells),
            "one-dimensional physical peculiar velocity variance",
            "radial peculiar kinetic-energy flux per comoving shell volume",
        )

        relative = _minimum_image(safe_positions, self.center, self.box_lengths)
        physical_relative = scale_factor * relative
        angular_value = jnp.cross(physical_relative, peculiar)
        angular_value = jnp.where(active[:, None], angular_value, 0.0)
        normalized = statistics.normalized_weights
        mean_angular = ein.contract("n,ni->i", normalized, angular_value)
        second_angular = ein.contract(
            "n,ni,nj->ij", normalized, angular_value, angular_value
        )
        radial_angular = (
            jnp.zeros((bin_count, 3), dtype=positions.dtype)
            .at[indices]
            .add(weight[:, None] * angular_value)
            / safe_shell_mass[:, None]
        )
        angular_finite = (
            jnp.all(jnp.isfinite(mean_angular))
            & jnp.all(jnp.isfinite(second_angular))
            & jnp.all(
                jnp.where(valid_shells[:, None], jnp.isfinite(radial_angular), True)
            )
        )
        angular = ParticleAngularMomentProduct(
            self.center,
            mean_angular,
            second_angular,
            radial_angular,
            shell_ess,
            angular_finite,
            angular_finite,
            "physical-specific-angular-momentum=a*x_comoving cross u_peculiar",
        )
        radial_global = ein.contract("n,n->", normalized, radial_velocity**2)
        tangential_global = ein.contract("n,n->", normalized, tangential_squared)
        global_anisotropy_identified = radial_global > 0.0
        global_anisotropy = jnp.where(
            global_anisotropy_identified,
            1.0 - tangential_global / (2.0 * radial_global),
            jnp.nan,
        )
        return (
            core,
            angular,
            gravothermal,
            global_anisotropy,
            global_anisotropy_identified,
        )

    def evaluate(
        self,
        state: CosmologicalParticleState,
        /,
        *,
        statistical_weights: ArrayLike | None = None,
        source_product_id: str,
    ) -> ParticleDarkMatterObservableProduct:
        if not isinstance(state, CosmologicalParticleState):
            raise TypeError("state must be CosmologicalParticleState.")
        source = _identifier(source_product_id, "source_product_id")
        particles = self.transfer.particles
        if (
            state.positions.shape
            != (
                particles.capacity,
                particles.ambient_dimension,
            )
            or state.canonical_momenta.shape != state.positions.shape
        ):
            raise ValueError(
                "Particle state does not match the prepared transfer support."
            )
        active = particles.active_mask
        statistical = (
            jnp.ones((particles.capacity,), dtype=state.positions.dtype)
            if statistical_weights is None
            else _real_array(statistical_weights, "statistical_weights").reshape((-1,))
        )
        if statistical.shape != (particles.capacity,):
            raise ValueError("statistical_weights must match particle capacity.")
        invalid_weights = active & ((~jnp.isfinite(statistical)) | (statistical < 0.0))
        effective_mass = jnp.where(
            active,
            particles.safe_masses.astype(state.positions.dtype) * statistical,
            0.0,
        )
        velocities = state.canonical_momenta / (
            particles.safe_masses[:, None].astype(state.positions.dtype)
            * state.scale_factor
        )
        transfer_state = self.transfer.build(state.positions, active_mask=active)
        mass = self.transfer.deposit_content(transfer_state, effective_mass)
        momentum = self.transfer.deposit_content(
            transfer_state, effective_mass[:, None] * velocities
        )
        velocity_outer = ein.contract("ni,nj->nij", velocities, velocities)
        second = self.transfer.deposit_content(
            transfer_state, effective_mass[:, None, None] * velocity_outer
        )
        mass_content = mass.content
        populated = mass_content > 0.0
        safe_content = jnp.where(populated, mass_content, 1.0)
        bulk = momentum.content / safe_content[..., None]
        raw_second = second.content / safe_content[..., None, None]
        bulk_outer = ein.contract("...i,...j->...ij", bulk, bulk)
        dispersion = raw_second - bulk_outer
        bulk = jnp.where(populated[..., None], bulk, 0.0)
        dispersion = jnp.where(populated[..., None, None], dispersion, 0.0)
        statistics = weighted_particle_statistics(velocities, effective_mass, active)
        (
            core,
            angular,
            gravothermal,
            anisotropy,
            anisotropy_identified,
        ) = self._radial_products(
            state.positions,
            velocities,
            effective_mass,
            active,
            state.scale_factor,
            statistics,
        )
        finite = (
            ~jnp.any(invalid_weights)
            & mass.successful
            & momentum.successful
            & second.successful
            & statistics.finite
            & core.finite
            & angular.finite
            & gravothermal.finite
            & jnp.all(jnp.isfinite(dispersion))
            & jnp.isfinite(state.scale_factor)
        )
        successful = (
            finite
            & statistics.successful
            & core.successful
            & angular.successful
            & gravothermal.successful
        )
        product_id = canonical_fingerprint(
            {
                "kind": "particle-dark-matter-observable-product",
                "source": source,
                "plan": self.plan_id,
            }
        )
        return ParticleDarkMatterObservableProduct(
            mass.density,
            mass_content,
            bulk,
            dispersion,
            populated,
            statistics,
            anisotropy,
            anisotropy_identified,
            core,
            angular,
            gravothermal,
            state.scale_factor,
            finite,
            successful,
            source,
            self.plan_id,
            product_id,
        )


class SIDMCollisionObservableProduct(StrictModule):
    expected_event_count: Array
    accepted_event_count: Array
    collision_rate: Array
    maximum_pair_probability: Array
    maximum_particle_probability: Array
    minimum_knudsen_number: Array
    collisionless: Array
    momentum_defect_norm: Array
    kinetic_energy_defect: Array
    probability_valid: Array
    conservative: Array
    finite: Array
    successful: Array


def sidm_collision_observables(
    diagnostics: SIDMCollisionDiagnostics,
    /,
) -> SIDMCollisionObservableProduct:
    if not isinstance(diagnostics, SIDMCollisionDiagnostics):
        raise TypeError("diagnostics must be SIDMCollisionDiagnostics.")
    expected = jnp.sum(diagnostics.pair_probability)
    accepted = diagnostics.event_count
    time = diagnostics.physical_time_step
    rate = jnp.where(time > 0.0, accepted.astype(time.dtype) / time, 0.0)
    maximum_pair = jnp.max(diagnostics.pair_probability, initial=0.0)
    maximum_particle = jnp.max(diagnostics.particle_aggregate_probability, initial=0.0)
    resolved_knudsen = jnp.isfinite(diagnostics.knudsen_number) & (
        diagnostics.knudsen_number >= 0.0
    )
    collisionless = (expected == 0.0) & (maximum_pair == 0.0)
    has_knudsen = jnp.any(resolved_knudsen)
    infinite_knudsen = collisionless & jnp.any(jnp.isposinf(diagnostics.knudsen_number))
    minimum_knudsen = jnp.where(
        has_knudsen,
        jnp.min(
            jnp.where(resolved_knudsen, diagnostics.knudsen_number, jnp.inf),
            initial=jnp.inf,
        ),
        jnp.where(infinite_knudsen, jnp.inf, jnp.nan),
    )
    knudsen_evidence_valid = jnp.isfinite(minimum_knudsen) | (
        collisionless & jnp.isposinf(minimum_knudsen)
    )
    momentum_defect = jnp.sqrt(
        jnp.sum(diagnostics.total_momentum_defect * diagnostics.total_momentum_defect)
    )
    finite = (
        diagnostics.finite
        & jnp.isfinite(expected)
        & jnp.isfinite(rate)
        & jnp.isfinite(maximum_pair)
        & jnp.isfinite(maximum_particle)
        & knudsen_evidence_valid
        & jnp.isfinite(momentum_defect)
        & jnp.isfinite(diagnostics.total_kinetic_energy_defect)
    )
    return SIDMCollisionObservableProduct(
        expected,
        accepted,
        rate,
        maximum_pair,
        maximum_particle,
        minimum_knudsen,
        collisionless,
        momentum_defect,
        diagnostics.total_kinetic_energy_defect,
        diagnostics.probability_valid & diagnostics.aggregate_probability_valid,
        diagnostics.conservative,
        finite,
        finite & diagnostics.successful,
    )


class WeightedSIDMCollisionObservableProduct(StrictModule):
    expected_event_count: Array
    accepted_event_count: Array
    exchanged_weight: Array
    effective_exchange_event_count: Array
    maximum_pair_probability: Array
    maximum_particle_probability: Array
    minimum_mean_free_path_physical: Array
    collisionless: Array
    mass_defect: Array
    momentum_defect_norm: Array
    kinetic_energy_defect: Array
    mass_relation_valid: Array
    conservative: Array
    finite: Array
    successful: Array


def weighted_sidm_collision_observables(
    diagnostics: WeightedSIDMCollisionDiagnostics,
    /,
) -> WeightedSIDMCollisionObservableProduct:
    if not isinstance(diagnostics, WeightedSIDMCollisionDiagnostics):
        raise TypeError("diagnostics must be WeightedSIDMCollisionDiagnostics.")
    expected = jnp.sum(diagnostics.pair_probability)
    exchanged = jnp.where(
        diagnostics.accepted_pairs,
        diagnostics.exchanged_weight,
        0.0,
    )
    total_exchanged = jnp.sum(exchanged)
    effective = total_exchanged**2 / jnp.maximum(
        jnp.sum(exchanged * exchanged),
        jnp.finfo(exchanged.dtype).tiny,
    )
    resolved_path = jnp.isfinite(diagnostics.mean_free_path_physical) & (
        diagnostics.mean_free_path_physical >= 0.0
    )
    collisionless = expected == 0.0
    has_resolved_path = jnp.any(resolved_path)
    infinite_path = collisionless & jnp.any(
        jnp.isposinf(diagnostics.mean_free_path_physical)
    )
    minimum_path = jnp.where(
        has_resolved_path,
        jnp.min(
            jnp.where(
                resolved_path,
                diagnostics.mean_free_path_physical,
                jnp.inf,
            ),
            initial=jnp.inf,
        ),
        jnp.where(infinite_path, jnp.inf, jnp.nan),
    )
    path_evidence_valid = jnp.isfinite(minimum_path) | (
        collisionless & jnp.isposinf(minimum_path)
    )
    momentum_defect = jnp.sqrt(
        jnp.sum(diagnostics.momentum_defect * diagnostics.momentum_defect)
    )
    finite = (
        diagnostics.finite
        & jnp.isfinite(expected)
        & jnp.isfinite(total_exchanged)
        & jnp.isfinite(effective)
        & path_evidence_valid
        & jnp.isfinite(diagnostics.mass_defect)
        & jnp.isfinite(momentum_defect)
        & jnp.isfinite(diagnostics.kinetic_energy_defect)
    )
    return WeightedSIDMCollisionObservableProduct(
        expected,
        diagnostics.event_count,
        total_exchanged,
        effective,
        jnp.max(diagnostics.pair_probability, initial=0.0),
        jnp.max(diagnostics.particle_aggregate_probability, initial=0.0),
        minimum_path,
        collisionless,
        diagnostics.mass_defect,
        momentum_defect,
        diagnostics.kinetic_energy_defect,
        diagnostics.mass_relation_valid,
        diagnostics.conservative,
        finite,
        finite & diagnostics.successful,
    )


class GravothermalSIDMObservableProduct(StrictModule):
    """Scalar release evidence composed from the gravothermal owner's diagnostics."""

    diagnostics: GravothermalSIDMDiagnostics
    maximum_absolute_hydrostatic_residual: Array
    relative_energy_balance_defect: Array
    all_shells_regime_supported: Array
    finite: Array
    successful: Array


def gravothermal_sidm_observables(
    diagnostics: GravothermalSIDMDiagnostics,
    /,
) -> GravothermalSIDMObservableProduct:
    if not isinstance(diagnostics, GravothermalSIDMDiagnostics):
        raise TypeError("diagnostics must be GravothermalSIDMDiagnostics.")
    hydrostatic = jnp.max(jnp.abs(diagnostics.hydrostatic_residual), initial=0.0)
    energy_scale = jnp.maximum(
        jnp.maximum(
            jnp.abs(diagnostics.thermal_energy_before),
            jnp.abs(diagnostics.thermal_energy_after),
        ),
        1.0,
    )
    relative_energy = jnp.abs(diagnostics.energy_balance_defect) / energy_scale
    regime = jnp.all(diagnostics.regime_supported)
    finite = (
        diagnostics.finite & jnp.isfinite(hydrostatic) & jnp.isfinite(relative_energy)
    )
    return GravothermalSIDMObservableProduct(
        diagnostics,
        hydrostatic,
        relative_energy,
        regime,
        finite,
        finite & diagnostics.successful & regime,
    )


class SIDMAngularMomentProduct(StrictModule):
    orders: Array
    legendre_moments: Array
    total_weight: Array
    effective_sample_size: Array
    active_count: Array
    finite: Array
    successful: Array
    convention: str = eqx.field(static=True)


def sidm_angular_moments(
    scattering_cosine: ArrayLike,
    weights: ArrayLike,
    active_mask: ArrayLike,
    /,
    *,
    maximum_order: int = 4,
) -> SIDMAngularMomentProduct:
    cosine = _real_array(scattering_cosine, "scattering_cosine").reshape((-1,))
    weight = _real_array(weights, "weights").reshape((-1,))
    active = jnp.asarray(active_mask, dtype=jnp.bool_).reshape((-1,))
    order = int(maximum_order)
    if cosine.shape != weight.shape or cosine.shape != active.shape:
        raise ValueError("SIDM angles, weights, and mask must align.")
    if order < 0:
        raise ValueError("maximum_order must be nonnegative.")
    cosine_valid = jnp.isfinite(cosine) & (jnp.abs(cosine) <= 1.0)
    weight_valid = jnp.isfinite(weight) & (weight >= 0.0)
    invalid = active & (~cosine_valid | ~weight_valid)
    usable = active & cosine_valid & weight_valid & (weight > 0.0)
    safe_cosine = jnp.where(usable, cosine, 0.0)
    safe_weight = jnp.where(usable, weight, 0.0)
    total = jnp.sum(safe_weight)
    normalized = safe_weight / jnp.where(total > 0.0, total, 1.0)
    polynomials = [jnp.ones_like(safe_cosine)]
    if order >= 1:
        polynomials.append(safe_cosine)
    for degree in range(2, order + 1):
        polynomials.append(
            (
                (2 * degree - 1) * safe_cosine * polynomials[-1]
                - (degree - 1) * polynomials[-2]
            )
            / degree
        )
    moments = jnp.stack(
        tuple(ein.contract("n,n->", normalized, polynomial) for polynomial in polynomials)
    )
    squared = jnp.sum(normalized * normalized)
    effective = 1.0 / jnp.maximum(squared, jnp.finfo(normalized.dtype).tiny)
    count = jnp.sum(usable.astype(jnp.int32))
    finite = ~jnp.any(invalid) & jnp.all(jnp.isfinite(moments)) & jnp.isfinite(effective)
    return SIDMAngularMomentProduct(
        jnp.arange(order + 1, dtype=jnp.int32),
        moments,
        total,
        effective,
        count,
        finite,
        finite & (total > 0.0),
        "normalized Legendre moments of physical center-of-mass scattering cosine",
    )


class MixedComponentSpectrumProduct(StrictModule):
    component_names: tuple[str, ...] = eqx.field(static=True)
    pair_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    scale_factor: Array
    wavenumbers: Array
    auto_power: Array
    cross_power: Array
    direct_total_power: Array
    reconstructed_total_power: Array
    closure_residual: Array
    maximum_imaginary_residual: Array
    mode_counts: Array
    valid_shells: Array
    finite: Array
    successful: Array
    normalization: DarkMatterDensityConvention = eqx.field(static=True)
    source_product_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class MixedComponentSpectrumPlan(StrictModule, NonTrainableState):
    """Auto/cross spectra whose algebra closes against the direct total field."""

    shells: PeriodicFourierShellPlan
    component_names: tuple[str, ...] = eqx.field(static=True)
    pair_indices: tuple[tuple[int, int], ...] = eqx.field(static=True)
    normalization: DarkMatterDensityConvention = eqx.field(static=True)
    closure_absolute_tolerance: float = eqx.field(static=True)
    closure_relative_tolerance: float = eqx.field(static=True)
    imaginary_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        shells: PeriodicFourierShellPlan,
        component_names: tuple[str, ...],
        /,
        *,
        normalization: DarkMatterDensityConvention = "total-density-contrast",
        closure_absolute_tolerance: float = 1.0e-10,
        closure_relative_tolerance: float = 1.0e-8,
        imaginary_tolerance: float = 1.0e-10,
    ):
        if not isinstance(shells, PeriodicFourierShellPlan):
            raise TypeError("shells must be PeriodicFourierShellPlan.")
        names = tuple(_identifier(value, "component name") for value in component_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("Mixed spectra require at least two unique components.")
        if normalization not in ("additive-field", "total-density-contrast"):
            raise ValueError("Unknown mixed-spectrum normalization.")
        absolute = float(closure_absolute_tolerance)
        relative = float(closure_relative_tolerance)
        imaginary = float(imaginary_tolerance)
        if (
            not math.isfinite(absolute)
            or absolute < 0.0
            or not math.isfinite(relative)
            or relative < 0.0
            or not math.isfinite(imaginary)
            or imaginary < 0.0
        ):
            raise ValueError("Mixed-spectrum closure tolerances must be nonnegative.")
        pairs = tuple(
            (left, right)
            for left in range(len(names))
            for right in range(left + 1, len(names))
        )
        self.shells = shells
        self.component_names = names
        self.pair_indices = pairs
        self.normalization = normalization
        self.closure_absolute_tolerance = absolute
        self.closure_relative_tolerance = relative
        self.imaginary_tolerance = imaginary
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mixed-component-spectrum-plan",
                "shells": shells.plan_id,
                "components": list(names),
                "normalization": normalization,
                "closure_absolute_tolerance": absolute,
                "closure_relative_tolerance": relative,
                "imaginary_tolerance": imaginary,
            }
        )

    def evaluate(
        self,
        component_fields: ArrayLike,
        scale_factor: ArrayLike,
        /,
        *,
        source_product_ids: tuple[str, ...],
    ) -> MixedComponentSpectrumProduct:
        fields = _real_array(component_fields, "component_fields")
        epoch = jnp.asarray(scale_factor, dtype=fields.dtype).reshape(())
        expected = (len(self.component_names),) + self.shells.source_shape
        if fields.shape != expected:
            raise ValueError(f"component_fields must have shape {expected}.")
        sources = tuple(
            _identifier(value, "source_product_id") for value in source_product_ids
        )
        if len(sources) != len(self.component_names):
            raise ValueError("One source product ID is required per mixed component.")
        total = jnp.sum(fields, axis=0)
        if self.normalization == "total-density-contrast":
            mean = jnp.mean(total)
            mean = eqx.error_if(
                mean,
                ~jnp.isfinite(mean) | (mean <= 0.0),
                "Total mixed density mean must be finite and positive.",
            )
            component_axes = tuple(range(1, fields.ndim))
            normalized = (
                fields - jnp.mean(fields, axis=component_axes, keepdims=True)
            ) / mean
        else:
            normalized = fields
        transformed = tuple(
            self.shells.transform(normalized[index])
            for index in range(len(self.component_names))
        )
        autos = tuple(self.shells.auto_power(value) for value in transformed)
        crosses = tuple(
            self.shells.cross_power(transformed[left], transformed[right])
            for left, right in self.pair_indices
        )
        total_statistic = self.shells.auto_power(
            self.shells.transform(jnp.sum(normalized, axis=0))
        )
        auto_power = jnp.stack(tuple(value.shell_values for value in autos))
        cross_power = jnp.stack(tuple(value.shell_values for value in crosses))
        reconstructed = jnp.sum(auto_power, axis=0) + 2.0 * jnp.sum(cross_power, axis=0)
        residual_by_shell = jnp.abs(total_statistic.shell_values - reconstructed)
        residual = jnp.max(
            jnp.where(
                total_statistic.valid_shells,
                residual_by_shell,
                0.0,
            ),
            initial=0.0,
        )
        scale = jnp.max(
            jnp.where(
                total_statistic.valid_shells,
                jnp.abs(total_statistic.shell_values),
                0.0,
            ),
            initial=0.0,
        )
        tolerance = (
            self.closure_absolute_tolerance + self.closure_relative_tolerance * scale
        )
        all_statistics = (*autos, *crosses, total_statistic)
        maximum_imaginary = jnp.max(
            jnp.stack(tuple(value.imaginary_residual for value in all_statistics)),
            initial=0.0,
        )
        finite = (
            jnp.all(jnp.stack(tuple(value.finite for value in all_statistics)))
            & jnp.isfinite(residual)
            & jnp.isfinite(maximum_imaginary)
            & jnp.isfinite(epoch)
            & (epoch > 0.0)
        )
        successful = (
            finite
            & jnp.all(jnp.stack(tuple(value.successful for value in all_statistics)))
            & (residual <= tolerance)
            & (maximum_imaginary <= self.imaginary_tolerance)
        )
        product_id = canonical_fingerprint(
            {
                "kind": "mixed-component-spectrum-product",
                "plan": self.plan_id,
                "sources": list(sources),
            }
        )
        return MixedComponentSpectrumProduct(
            self.component_names,
            self.pair_indices,
            epoch,
            total_statistic.representative_wavenumbers,
            auto_power,
            cross_power,
            total_statistic.shell_values,
            reconstructed,
            residual,
            maximum_imaginary,
            total_statistic.weighted_mode_count,
            total_statistic.valid_shells,
            finite,
            successful,
            self.normalization,
            sources,
            self.plan_id,
            product_id,
        )


class ComponentForceWorkLedger(StrictModule):
    component_names: tuple[str, ...] = eqx.field(static=True)
    interval_start_scale_factor: Array
    interval_end_scale_factor: Array
    component_force: Array
    system_force: Array
    force_closure_residual: Array
    component_work: Array
    system_work: Array
    work_closure_residual: Array
    interval_valid: Array
    finite: Array
    successful: Array
    force_time_level: str = eqx.field(static=True)
    displacement_time_level: str = eqx.field(static=True)
    force_unit: str = eqx.field(static=True)
    work_unit: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ComponentForceWorkLedgerPlan(StrictModule, NonTrainableState):
    """Fixed-schedule component force/work accounting without inferred time levels."""

    component_names: tuple[str, ...] = eqx.field(static=True)
    force_time_level: str = eqx.field(static=True)
    displacement_time_level: str = eqx.field(static=True)
    force_unit: str = eqx.field(static=True)
    work_unit: str = eqx.field(static=True)
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_names: tuple[str, ...],
        /,
        *,
        force_time_level: str,
        displacement_time_level: str,
        force_unit: str,
        work_unit: str,
        absolute_tolerance: float = 1.0e-10,
        relative_tolerance: float = 1.0e-8,
    ):
        names = tuple(_identifier(value, "component name") for value in component_names)
        if len(names) < 2 or len(set(names)) != len(names):
            raise ValueError("A mixed force/work ledger requires unique components.")
        force_level = _identifier(force_time_level, "force_time_level")
        displacement_level = _identifier(
            displacement_time_level, "displacement_time_level"
        )
        force_unit_ = _identifier(force_unit, "force_unit")
        work_unit_ = _identifier(work_unit, "work_unit")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not math.isfinite(absolute)
            or absolute < 0.0
            or not math.isfinite(relative)
            or relative < 0.0
        ):
            raise ValueError("Ledger closure tolerances must be nonnegative.")
        self.component_names = names
        self.force_time_level = force_level
        self.displacement_time_level = displacement_level
        self.force_unit = force_unit_
        self.work_unit = work_unit_
        self.absolute_tolerance = absolute
        self.relative_tolerance = relative
        self.plan_id = canonical_fingerprint(
            {
                "kind": "component-force-work-ledger-plan",
                "components": list(names),
                "force_time_level": force_level,
                "displacement_time_level": displacement_level,
                "force_unit": force_unit_,
                "work_unit": work_unit_,
                "absolute_tolerance": absolute,
                "relative_tolerance": relative,
            }
        )

    def evaluate(
        self,
        interval_start_scale_factor: ArrayLike,
        interval_end_scale_factor: ArrayLike,
        component_force: ArrayLike,
        component_displacement: ArrayLike,
        system_force: ArrayLike,
        system_work: ArrayLike,
        /,
    ) -> ComponentForceWorkLedger:
        start = _real_array(
            interval_start_scale_factor, "interval_start_scale_factor"
        ).reshape((-1,))
        end = _real_array(interval_end_scale_factor, "interval_end_scale_factor").reshape(
            (-1,)
        )
        force = _real_array(component_force, "component_force")
        displacement = _real_array(component_displacement, "component_displacement")
        system_force_ = _real_array(system_force, "system_force")
        system_work_ = _real_array(system_work, "system_work").reshape((-1,))
        interval_count = start.size
        expected_prefix = (interval_count, len(self.component_names))
        if (
            end.shape != start.shape
            or force.ndim < 3
            or force.shape[:2] != expected_prefix
            or displacement.shape != force.shape
            or system_force_.shape != (interval_count,) + force.shape[2:]
            or system_work_.shape != (interval_count,)
        ):
            raise ValueError("Component force/work ledger array shapes disagree.")
        component_work = jnp.sum(force * displacement, axis=tuple(range(2, force.ndim)))
        force_sum = jnp.sum(force, axis=1)
        force_residual = jnp.sqrt(
            jnp.sum(
                (force_sum - system_force_) ** 2,
                axis=tuple(range(1, system_force_.ndim)),
            )
        )
        reconstructed_work = jnp.sum(component_work, axis=1)
        work_residual = jnp.abs(reconstructed_work - system_work_)
        interval_valid = jnp.isfinite(start) & jnp.isfinite(end) & (end > start)
        finite = (
            jnp.all(interval_valid)
            & jnp.all(jnp.isfinite(force))
            & jnp.all(jnp.isfinite(displacement))
            & jnp.all(jnp.isfinite(system_force_))
            & jnp.all(jnp.isfinite(system_work_))
            & jnp.all(jnp.isfinite(force_residual))
            & jnp.all(jnp.isfinite(work_residual))
        )
        force_scale = jnp.sqrt(
            jnp.sum(
                system_force_ * system_force_, axis=tuple(range(1, system_force_.ndim))
            )
        )
        force_tolerance = self.absolute_tolerance + self.relative_tolerance * force_scale
        work_tolerance = self.absolute_tolerance + self.relative_tolerance * jnp.abs(
            system_work_
        )
        successful = (
            finite
            & jnp.all(force_residual <= force_tolerance)
            & jnp.all(work_residual <= work_tolerance)
        )
        return ComponentForceWorkLedger(
            self.component_names,
            start,
            end,
            force,
            system_force_,
            force_residual,
            component_work,
            system_work_,
            work_residual,
            interval_valid,
            finite,
            successful,
            self.force_time_level,
            self.displacement_time_level,
            self.force_unit,
            self.work_unit,
            self.plan_id,
        )


def component_force_work_ledger_from_mixed_diagnostics(
    plan: ComponentForceWorkLedgerPlan,
    diagnostics: MixedCosmologyDiagnostics,
    /,
) -> ComponentForceWorkLedger:
    """Compose the runtime's independently recorded force and gravity-work ledgers."""

    if not isinstance(plan, ComponentForceWorkLedgerPlan):
        raise TypeError("plan must be ComponentForceWorkLedgerPlan.")
    if not isinstance(diagnostics, MixedCosmologyDiagnostics):
        raise TypeError("diagnostics must be MixedCosmologyDiagnostics.")
    if tuple(diagnostics.component_names) != plan.component_names:
        raise ValueError("Mixed runtime and observable component orders disagree.")
    start = diagnostics.start_scale_factor
    end = diagnostics.end_scale_factor
    force = diagnostics.component_force
    system_force = diagnostics.total_force
    component_work = diagnostics.component_gravity_work
    system_work = diagnostics.total_gravity_work
    if (
        force.ndim < 3
        or force.shape[:2] != (start.size, len(plan.component_names))
        or system_force.shape != (start.size,) + force.shape[2:]
        or component_work.shape != (start.size, len(plan.component_names))
        or system_work.shape != (start.size,)
    ):
        raise ValueError("Mixed runtime force/work diagnostics have invalid shapes.")
    force_residual = jnp.sqrt(
        jnp.sum(
            (jnp.sum(force, axis=1) - system_force) ** 2,
            axis=tuple(range(1, system_force.ndim)),
        )
    )
    work_residual = jnp.abs(jnp.sum(component_work, axis=1) - system_work)
    interval_valid = (
        diagnostics.attempted & jnp.isfinite(start) & jnp.isfinite(end) & (end > start)
    )
    finite = (
        jnp.all(jnp.where(interval_valid[..., None, None], jnp.isfinite(force), True))
        & jnp.all(jnp.where(interval_valid[..., None], jnp.isfinite(system_force), True))
        & jnp.all(
            jnp.where(interval_valid[..., None], jnp.isfinite(component_work), True)
        )
        & jnp.all(jnp.where(interval_valid, jnp.isfinite(system_work), True))
        & jnp.all(jnp.where(interval_valid, jnp.isfinite(force_residual), True))
        & jnp.all(jnp.where(interval_valid, jnp.isfinite(work_residual), True))
    )
    force_scale = jnp.sqrt(
        jnp.sum(system_force * system_force, axis=tuple(range(1, system_force.ndim)))
    )
    force_tolerance = plan.absolute_tolerance + plan.relative_tolerance * force_scale
    work_tolerance = plan.absolute_tolerance + plan.relative_tolerance * jnp.abs(
        system_work
    )
    closed = jnp.all(
        ~interval_valid
        | ((force_residual <= force_tolerance) & (work_residual <= work_tolerance))
    )
    return ComponentForceWorkLedger(
        plan.component_names,
        start,
        end,
        force,
        system_force,
        jnp.where(interval_valid, force_residual, 0.0),
        component_work,
        system_work,
        jnp.where(interval_valid, work_residual, 0.0),
        interval_valid,
        finite,
        finite & closed & diagnostics.completed,
        plan.force_time_level,
        plan.displacement_time_level,
        plan.force_unit,
        plan.work_unit,
        plan.plan_id,
    )


class DarkMatterSpatialContract(StrictModule, NonTrainableState):
    """Typed Euclidean geometry, frame, unit, scale, and coordinate-time identity."""

    box_lengths: tuple[float, ...] = eqx.field(static=True)
    axis_names: tuple[str, ...] = eqx.field(static=True)
    geometry_kind: Literal["flat-periodic-cartesian", "flat-sky-cartesian"] = eqx.field(
        static=True
    )
    frame_id: str = eqx.field(static=True)
    physics_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    coordinate_time_level: str = eqx.field(static=True)
    length_unit_id: str = eqx.field(static=True)
    length_coordinate_kind: Literal["comoving", "physical"] = eqx.field(static=True)
    contract_id: str = eqx.field(static=True)

    def __init__(
        self,
        box_lengths: tuple[float, ...],
        /,
        *,
        axis_names: tuple[str, ...],
        geometry_kind: Literal["flat-periodic-cartesian", "flat-sky-cartesian"],
        frame_id: str,
        physics_id: str,
        scale_id: str,
        coordinate_time_level: str,
        length_unit_id: str,
        length_coordinate_kind: Literal["comoving", "physical"],
    ):
        lengths = tuple(float(value) for value in box_lengths)
        if len(lengths) not in (2, 3) or any(
            not math.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError("Dark-matter spatial box geometry is invalid.")
        axes = tuple(_identifier(value, "axis_name") for value in axis_names)
        if len(axes) != len(lengths) or len(set(axes)) != len(axes):
            raise ValueError("Spatial axis names must be unique and match box rank.")
        if geometry_kind not in (
            "flat-periodic-cartesian",
            "flat-sky-cartesian",
        ):
            raise ValueError("Unknown dark-matter spatial geometry kind.")
        if length_coordinate_kind not in ("comoving", "physical"):
            raise ValueError("Unknown dark-matter length coordinate kind.")
        self.box_lengths = lengths
        self.axis_names = axes
        self.geometry_kind = geometry_kind
        self.frame_id = _identifier(frame_id, "frame_id")
        self.physics_id = _identifier(physics_id, "physics_id")
        self.scale_id = _identifier(scale_id, "scale_id")
        self.coordinate_time_level = _identifier(
            coordinate_time_level, "coordinate_time_level"
        )
        self.length_unit_id = _identifier(length_unit_id, "length_unit_id")
        self.length_coordinate_kind = length_coordinate_kind
        self.contract_id = canonical_fingerprint(
            {
                "kind": "dark-matter-spatial-contract",
                "box_lengths": list(lengths),
                "axis_names": list(axes),
                "geometry_kind": geometry_kind,
                "frame_id": self.frame_id,
                "physics_id": self.physics_id,
                "scale_id": self.scale_id,
                "coordinate_time_level": self.coordinate_time_level,
                "length_unit_id": self.length_unit_id,
                "length_coordinate_kind": length_coordinate_kind,
            }
        )


class DarkMatterHaloCompositionProduct(StrictModule, NonTrainableState):
    catalog: FoFFinderResult
    snapshot: ParticleSimulationSnapshot
    spatial: DarkMatterSpatialContract
    finite: Array
    successful: Array
    owner_plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class DarkMatterPeriodicRadialShellProduct(StrictModule, NonTrainableState):
    selection: LightConeResult
    selected_active: Array
    selected_count: Array
    selected_indices: Array
    selected_stable_ids: Array
    snapshot: ParticleSimulationSnapshot
    spatial: DarkMatterSpatialContract
    observer_position: Array
    finite: Array
    successful: Array
    owner_plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


class DarkMatterSurfaceDensityProduct(StrictModule, NonTrainableState):
    surface_density: Array
    critical_density: Array
    spatial: DarkMatterSpatialContract
    artifact: ScientificArtifactEnvelope
    finite: Array
    successful: Array
    density_unit_id: str = eqx.field(static=True)
    density_coordinate_kind: Literal["comoving", "physical"] = eqx.field(static=True)
    source_product_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)

    def __init__(
        self,
        surface_density: ArrayLike,
        critical_density: ArrayLike,
        spatial: DarkMatterSpatialContract,
        artifact: ScientificArtifactEnvelope,
        /,
        *,
        density_unit_id: str,
        density_coordinate_kind: Literal["comoving", "physical"],
        source_product_id: str,
    ):
        if not isinstance(spatial, DarkMatterSpatialContract):
            raise TypeError("spatial must be DarkMatterSpatialContract.")
        if len(spatial.box_lengths) != 2:
            raise ValueError(
                "Lensing surface density requires a two-dimensional contract."
            )
        if not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be ScientificArtifactEnvelope.")
        if artifact.status != "complete":
            raise ValueError("Lensing input artifact must be complete.")
        if (
            density_coordinate_kind not in ("comoving", "physical")
            or density_coordinate_kind != spatial.length_coordinate_kind
        ):
            raise ValueError("Surface-density and spatial coordinate kinds must agree.")
        density = jax.lax.stop_gradient(jnp.asarray(surface_density))
        critical = jax.lax.stop_gradient(
            jnp.asarray(critical_density, dtype=density.dtype).reshape(())
        )
        if density.ndim != 2:
            raise ValueError("Surface density must be a two-dimensional field.")
        finite = (
            jnp.all(jnp.isfinite(density)) & jnp.isfinite(critical) & (critical > 0.0)
        )
        unit = _identifier(density_unit_id, "density_unit_id")
        source = _identifier(source_product_id, "source_product_id")
        self.surface_density = density
        self.critical_density = critical
        self.spatial = spatial
        self.artifact = artifact
        self.finite = finite
        self.successful = finite
        self.density_unit_id = unit
        self.density_coordinate_kind = density_coordinate_kind
        self.source_product_id = source
        self.product_id = canonical_fingerprint(
            {
                "kind": "dark-matter-surface-density-product",
                "source": source,
                "artifact": artifact.artifact_id,
                "spatial": spatial.contract_id,
                "density_unit_id": unit,
                "density_coordinate_kind": density_coordinate_kind,
                "surface_density": array_tree_fingerprint(np.asarray(density)),
                "critical_density": array_tree_fingerprint(np.asarray(critical)),
            }
        )


class DarkMatterLensingCompositionProduct(StrictModule, NonTrainableState):
    convergence: Array
    first_shear: Array
    second_shear: Array
    source: DarkMatterSurfaceDensityProduct
    finite: Array
    successful: Array
    owner_plan_id: str = eqx.field(static=True)
    product_id: str = eqx.field(static=True)


def find_dark_matter_halos(
    plan: PeriodicFoFFinderPlan,
    snapshot: ParticleSimulationSnapshot,
    particles: ParticleDiscretization,
    spatial: DarkMatterSpatialContract,
    /,
) -> DarkMatterHaloCompositionProduct:
    """Delegate halo finding after exact snapshot/geometry identity admission."""

    if not isinstance(plan, PeriodicFoFFinderPlan):
        raise TypeError("plan must be PeriodicFoFFinderPlan.")
    if not isinstance(snapshot, ParticleSimulationSnapshot):
        raise TypeError("snapshot must be ParticleSimulationSnapshot.")
    if not isinstance(particles, ParticleDiscretization):
        raise TypeError("particles must be ParticleDiscretization.")
    if not isinstance(spatial, DarkMatterSpatialContract):
        raise TypeError("spatial must be DarkMatterSpatialContract.")
    active_host = np.asarray(snapshot.active_mask, dtype=np.bool_)
    mass_host = np.asarray(snapshot.macro_masses)
    weight_host = np.asarray(snapshot.packet_weights)
    active_properties_valid = bool(
        np.all(np.isfinite(mass_host[active_host]))
        and np.all(mass_host[active_host] > 0.0)
        and np.all(np.isfinite(weight_host[active_host]))
        and np.all(weight_host[active_host] > 0.0)
    )
    if (
        particles.ambient_dimension != 3
        or len(spatial.box_lengths) != 3
        or spatial.axis_names != ("x", "y", "z")
        or spatial.geometry_kind != "flat-periodic-cartesian"
        or not active_properties_valid
        or snapshot.support_id != particles.prepared_id
        or snapshot.physics_id != spatial.physics_id
        or spatial.length_coordinate_kind != "comoving"
        or snapshot.scale_id != spatial.scale_id
        or snapshot.coordinate_time_level != spatial.coordinate_time_level
        or not np.allclose(
            np.asarray(plan.box_size),
            np.asarray(spatial.box_lengths),
            rtol=0.0,
            atol=0.0,
        )
        or not np.array_equal(
            np.asarray(snapshot.stable_ids),
            np.asarray(particles.particle_ids),
        )
    ):
        raise ValueError(
            "Halo owner, particle snapshot, support, box, scale, or time identity disagrees."
        )
    active = snapshot.active_mask
    safe_mass = jnp.where(active, snapshot.macro_masses, 1.0)
    velocities = snapshot.canonical_momenta / (safe_mass[:, None] * snapshot.scale_factor)
    catalog = plan.find(
        snapshot.stable_ids,
        snapshot.positions,
        velocities,
        snapshot.macro_masses,
        active,
    )
    finite = (
        snapshot.evidence.status.successful
        & catalog.finite
        & jnp.all(jnp.isfinite(velocities) | ~active[:, None])
    )
    successful = finite & catalog.successful
    product_id = canonical_fingerprint(
        {
            "kind": "dark-matter-halo-composition",
            "snapshot": snapshot.snapshot_id,
            "spatial": spatial.contract_id,
            "owner_plan": plan.plan_id,
        }
    )
    return DarkMatterHaloCompositionProduct(
        catalog,
        snapshot,
        spatial,
        finite,
        successful,
        plan.plan_id,
        product_id,
    )


def apply_dark_matter_observation(
    product: TheoryVector,
    observation: LinearObservationPlan,
    /,
) -> TheoryVector:
    """Delegate a dark-matter theory vector to the native observation owner."""

    if not isinstance(product, TheoryVector):
        raise TypeError("product must be a TheoryVector.")
    if not isinstance(observation, LinearObservationPlan):
        raise TypeError("observation must be LinearObservationPlan.")
    return observation.apply(product)


def select_dark_matter_periodic_radial_shells(
    plan: LightConePlan,
    snapshot: ParticleSimulationSnapshot,
    spatial: DarkMatterSpatialContract,
    observer_position: ArrayLike,
    /,
) -> DarkMatterPeriodicRadialShellProduct:
    """Select one-time periodic radial shells without claiming a past light cone."""

    if not isinstance(plan, LightConePlan):
        raise TypeError("plan must be LightConePlan.")
    if not isinstance(snapshot, ParticleSimulationSnapshot):
        raise TypeError("snapshot must be ParticleSimulationSnapshot.")
    if not isinstance(spatial, DarkMatterSpatialContract):
        raise TypeError("spatial must be DarkMatterSpatialContract.")
    radii_host = np.asarray(plan.shell_radii, dtype=np.float64).reshape((-1,))
    observer = jax.lax.stop_gradient(
        jnp.asarray(
            observer_position,
            dtype=snapshot.positions.dtype,
        ).reshape((-1,))
    )
    if (
        observer.shape != (3,)
        or radii_host.size == 0
        or np.any(~np.isfinite(radii_host))
        or np.any(radii_host <= 0.0)
        or np.any(np.diff(radii_host) <= 0.0)
        or plan.capacity <= 0
        or plan.capacity > snapshot.stable_ids.size
        or len(spatial.box_lengths) != 3
        or spatial.axis_names != ("x", "y", "z")
        or spatial.geometry_kind != "flat-periodic-cartesian"
        or snapshot.physics_id != spatial.physics_id
        or spatial.length_coordinate_kind != "comoving"
        or snapshot.scale_id != spatial.scale_id
        or snapshot.coordinate_time_level != spatial.coordinate_time_level
        or not bool(np.all(np.isfinite(np.asarray(observer))))
        or radii_host[-1] > 0.5 * min(spatial.box_lengths)
    ):
        raise ValueError(
            "Periodic radial-shell owner/source/frame/time/unit support is inconsistent."
        )
    lengths = jnp.asarray(
        spatial.box_lengths,
        dtype=snapshot.positions.dtype,
    )
    relative = (
        jnp.mod(snapshot.positions - observer + 0.5 * lengths, lengths) - 0.5 * lengths
    )
    outside = jnp.asarray(
        (2.0 * radii_host[-1], 0.0, 0.0),
        dtype=relative.dtype,
    )
    safe_relative = jnp.where(
        snapshot.active_mask[:, None],
        relative,
        outside,
    )
    source_radius = jnp.sqrt(jnp.sum(safe_relative * safe_relative, axis=-1))
    source_shell = jnp.searchsorted(plan.shell_radii, source_radius)
    source_selected = source_shell < plan.shell_radii.size
    source_order = jnp.argsort(~source_selected)
    selected_indices = source_order[: plan.capacity]
    selected_stable_ids = snapshot.stable_ids[selected_indices]
    selected_active = source_selected[selected_indices]
    selected_count = jnp.sum(selected_active.astype(jnp.int32))
    selection = plan.select(safe_relative)
    owner_plan_id = canonical_fingerprint(
        {
            "kind": "bound-dark-matter-periodic-radial-shell-owner",
            "shell_radii": array_tree_fingerprint(radii_host),
            "capacity": plan.capacity,
            "spatial": spatial.contract_id,
        }
    )
    finite = (
        snapshot.evidence.status.successful
        & selection.valid
        & jnp.all(jnp.isfinite(selection.positions))
        & jnp.all(jnp.isfinite(relative) | ~snapshot.active_mask[:, None])
    )
    successful = finite & ~selection.overflow
    product_id = canonical_fingerprint(
        {
            "kind": "dark-matter-periodic-radial-shell-composition",
            "snapshot": snapshot.snapshot_id,
            "spatial": spatial.contract_id,
            "owner_plan": owner_plan_id,
            "observer": array_tree_fingerprint(np.asarray(observer)),
            "selected_indices": array_tree_fingerprint(np.asarray(selected_indices)),
            "selected_stable_ids": array_tree_fingerprint(
                np.asarray(selected_stable_ids)
            ),
        }
    )
    return DarkMatterPeriodicRadialShellProduct(
        selection,
        selected_active,
        selected_count,
        selected_indices,
        selected_stable_ids,
        snapshot,
        spatial,
        observer,
        finite,
        successful,
        owner_plan_id,
        product_id,
    )


def project_dark_matter_lensing_plane(
    plan: LensingPlanePlan,
    source: DarkMatterSurfaceDensityProduct,
    /,
) -> DarkMatterLensingCompositionProduct:
    """Delegate convergence/shear with bound density and projection contracts."""

    if not isinstance(plan, LensingPlanePlan):
        raise TypeError("plan must be LensingPlanePlan.")
    if not isinstance(source, DarkMatterSurfaceDensityProduct):
        raise TypeError("source must be DarkMatterSurfaceDensityProduct.")
    pixel_scale = jnp.asarray(plan.pixel_scale).reshape(())
    if not bool(np.isfinite(np.asarray(pixel_scale))) or not bool(
        np.asarray(pixel_scale) > 0.0
    ):
        raise ValueError("Lensing pixel scale must be finite and positive.")
    pixel_host = np.asarray(pixel_scale)
    numeric_dtype = np.dtype(np.result_type(pixel_host.dtype, np.float32))
    raster_extent = (
        np.asarray(source.surface_density.shape, dtype=numeric_dtype) * pixel_host
    )
    box_extent = np.asarray(
        source.spatial.box_lengths,
        dtype=numeric_dtype,
    )
    tolerance = 64.0 * np.finfo(numeric_dtype).eps * np.maximum(np.abs(box_extent), 1.0)
    if (
        source.spatial.axis_names != ("y", "x")
        or source.spatial.geometry_kind != "flat-sky-cartesian"
        or np.any(np.abs(raster_extent - box_extent) > tolerance)
    ):
        raise ValueError(
            "Lensing raster shape/pixel scale does not match the bound y/x box."
        )
    convergence, first, second = plan.convergence_and_shear(
        source.surface_density,
        source.critical_density,
    )
    owner_plan_id = canonical_fingerprint(
        {
            "kind": "bound-dark-matter-lensing-owner",
            "pixel_scale": array_tree_fingerprint(np.asarray(pixel_scale)),
            "spatial": source.spatial.contract_id,
            "density_unit_id": source.density_unit_id,
            "density_coordinate_kind": source.density_coordinate_kind,
        }
    )
    finite = (
        source.finite
        & jnp.all(jnp.isfinite(convergence))
        & jnp.all(jnp.isfinite(first))
        & jnp.all(jnp.isfinite(second))
    )
    product_id = canonical_fingerprint(
        {
            "kind": "dark-matter-lensing-composition",
            "source": source.product_id,
            "owner_plan": owner_plan_id,
        }
    )
    return DarkMatterLensingCompositionProduct(
        convergence,
        first,
        second,
        source,
        finite,
        finite & source.successful,
        owner_plan_id,
        product_id,
    )


__all__ = [
    "ComponentForceWorkLedger",
    "ComponentForceWorkLedgerPlan",
    "DarkMatterDensityConvention",
    "DarkMatterHaloCompositionProduct",
    "DarkMatterLensingCompositionProduct",
    "DarkMatterPeriodicRadialShellProduct",
    "DarkMatterSpatialContract",
    "DarkMatterSurfaceDensityProduct",
    "DarkMatterSpectrumProduct",
    "GravothermalSIDMObservableProduct",
    "MixedComponentSpectrumPlan",
    "MixedComponentSpectrumProduct",
    "ParticleAngularMomentProduct",
    "ParticleDarkMatterObservablePlan",
    "ParticleDarkMatterObservableProduct",
    "ParticleGravothermalProduct",
    "RadialCoreProfileProduct",
    "SIDMAngularMomentProduct",
    "SIDMCollisionObservableProduct",
    "VortexCirculationProduct",
    "WaveDarkMatterObservablePlan",
    "WaveDarkMatterObservableProduct",
    "WeightedParticleStatistics",
    "WeightedSIDMCollisionObservableProduct",
    "WeightedSIDMPacketObservableProduct",
    "apply_dark_matter_observation",
    "component_force_work_ledger_from_mixed_diagnostics",
    "find_dark_matter_halos",
    "gravothermal_sidm_observables",
    "project_dark_matter_lensing_plane",
    "select_dark_matter_periodic_radial_shells",
    "sidm_angular_moments",
    "sidm_collision_observables",
    "weighted_particle_statistics",
    "weighted_sidm_collision_observables",
    "weighted_sidm_packet_observables",
]
