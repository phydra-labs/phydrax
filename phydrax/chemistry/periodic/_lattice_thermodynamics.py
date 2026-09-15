#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stable-mode harmonic thermodynamics and interior-only quasiharmonic analysis."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomisticUnitSystem


class HarmonicThermodynamicsResult(StrictModule, NonTrainableState):
    temperatures: Array
    free_energy: Array
    internal_energy: Array
    entropy: Array
    heat_capacity: Array
    zero_point_energy: Array
    thermodynamic_identity_residual: Array
    minimum_dimensionless_frequency: Array
    maximum_dimensionless_frequency: Array
    asymptotic_mode_count: Array
    successful: Array
    unit_system_id: str = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures,
        free,
        internal,
        entropy,
        heat,
        zero,
        identity,
        x_min,
        x_max,
        asymptotic_count,
        successful,
        unit_system_id,
        /,
    ):
        self.temperatures = jnp.asarray(temperatures)
        dtype = self.temperatures.dtype
        self.free_energy = jnp.asarray(free, dtype=dtype)
        self.internal_energy = jnp.asarray(internal, dtype=dtype)
        self.entropy = jnp.asarray(entropy, dtype=dtype)
        self.heat_capacity = jnp.asarray(heat, dtype=dtype)
        self.zero_point_energy = jnp.asarray(zero, dtype=dtype).reshape(())
        self.thermodynamic_identity_residual = jnp.asarray(identity, dtype=dtype).reshape(
            ()
        )
        self.minimum_dimensionless_frequency = jnp.asarray(x_min, dtype=dtype).reshape(())
        self.maximum_dimensionless_frequency = jnp.asarray(x_max, dtype=dtype).reshape(())
        self.asymptotic_mode_count = jnp.asarray(
            asymptotic_count, dtype=jnp.int32
        ).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.unit_system_id = str(unit_system_id)
        self.normalization = "per-primitive-cell"
        self.result_id = canonical_fingerprint(
            {
                "kind": "harmonic-thermodynamics-result",
                "unit_system": self.unit_system_id,
                "normalization": self.normalization,
                "arrays": array_tree_fingerprint(
                    {
                        "temperatures": np.asarray(self.temperatures),
                        "free": np.asarray(self.free_energy),
                        "internal": np.asarray(self.internal_energy),
                        "entropy": np.asarray(self.entropy),
                        "heat": np.asarray(self.heat_capacity),
                        "zpe": np.asarray(self.zero_point_energy),
                    }
                ),
            }
        )


class HarmonicThermodynamicsPlan(StrictModule, NonTrainableState):
    """Thermodynamic integration over a Γ-excluding stable reciprocal mesh."""

    qpoint_weights: Array
    temperatures: Array
    units: AtomisticUnitSystem
    maximum_scalar_evaluations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        qpoint_weights: ArrayLike,
        temperatures: ArrayLike,
        units: AtomisticUnitSystem,
        /,
        *,
        maximum_scalar_evaluations: int = 2_000_000_000,
    ):
        weights = np.asarray(qpoint_weights, dtype=float)
        temperature = np.asarray(temperatures, dtype=float)
        if (
            weights.ndim != 1
            or weights.size == 0
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "Thermodynamic q-point weights must be finite and strictly positive."
            )
        if not np.isclose(np.sum(weights), 1.0, atol=1.0e-12):
            raise ValueError("Thermodynamic q-point weights must sum to one.")
        if (
            temperature.ndim != 1
            or temperature.size == 0
            or np.any(~np.isfinite(temperature))
            or np.any(temperature <= 0.0)
            or np.any(np.diff(temperature) <= 0.0)
        ):
            raise ValueError(
                "Temperatures must be finite, positive, and strictly increasing."
            )
        if not isinstance(units, AtomisticUnitSystem):
            raise TypeError("units must be AtomisticUnitSystem.")
        if int(maximum_scalar_evaluations) <= 0:
            raise ValueError("maximum_scalar_evaluations must be positive.")
        self.qpoint_weights = jnp.asarray(weights)
        self.temperatures = jnp.asarray(temperature)
        self.units = units
        self.maximum_scalar_evaluations = int(maximum_scalar_evaluations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "harmonic-thermodynamics-plan",
                "units": units.unit_system_id,
                "maximum_scalar_evaluations": self.maximum_scalar_evaluations,
                "arrays": array_tree_fingerprint(
                    {"weights": weights, "temperatures": temperature}
                ),
            }
        )

    def evaluate(self, angular_frequencies: ArrayLike, /) -> HarmonicThermodynamicsResult:
        frequency = jnp.asarray(angular_frequencies, dtype=self.qpoint_weights.dtype)
        if frequency.ndim != 2 or frequency.shape[0] != self.qpoint_weights.shape[0]:
            raise ValueError("Angular frequencies must have shape (Q, branches).")
        if int(frequency.size * self.temperatures.size) > self.maximum_scalar_evaluations:
            raise ValueError(
                "Harmonic thermodynamic scalar-evaluation capacity exceeded."
            )
        if bool(jnp.any(~jnp.isfinite(frequency))) or bool(jnp.any(frequency <= 0.0)):
            raise ValueError(
                "Thermodynamics requires strictly positive stable modes on a Γ-excluding mesh."
            )
        hbar = self.units.reduced_planck_constant
        boltzmann = self.units.boltzmann_constant
        energy = hbar * frequency
        x = energy[None, :, :] / (boltzmann * self.temperatures[:, None, None])
        occupation = jnp.where(x > 50.0, jnp.exp(-x), 1.0 / jnp.expm1(x))
        logarithm = jnp.log(-jnp.expm1(-x))
        weighted = self.qpoint_weights[None, :, None]
        zero = 0.5 * jnp.sum(self.qpoint_weights[:, None] * energy)
        internal = jnp.sum(
            weighted * energy[None, :, :] * (0.5 + occupation), axis=(1, 2)
        )
        free = zero + boltzmann * self.temperatures * jnp.sum(
            weighted * logarithm, axis=(1, 2)
        )
        entropy = (internal - free) / self.temperatures
        heat_kernel = jnp.where(
            x > 50.0, x**2 * jnp.exp(-x), x**2 * occupation * (1.0 + occupation)
        )
        heat = boltzmann * jnp.sum(weighted * heat_kernel, axis=(1, 2))
        identity = jnp.max(
            jnp.abs(free - (internal - self.temperatures * entropy)), initial=0.0
        )
        successful = (
            jnp.all(jnp.isfinite(free))
            & jnp.all(jnp.isfinite(internal))
            & jnp.all(jnp.isfinite(entropy))
            & jnp.all(jnp.isfinite(heat))
        )
        return HarmonicThermodynamicsResult(
            self.temperatures,
            free,
            internal,
            entropy,
            heat,
            zero,
            identity,
            jnp.min(x),
            jnp.max(x),
            jnp.sum(x > 50.0),
            successful,
            self.units.unit_system_id,
        )


class QuasiHarmonicResult(StrictModule, NonTrainableState):
    temperatures: Array
    volumes: Array
    raw_free_energies: Array
    equilibrium_volumes: Array
    minimum_free_energies: Array
    bulk_curvature: Array
    volumetric_thermal_expansion: Array
    bracket_indices: Array
    interpolation_residuals: Array
    successful: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        temperatures,
        volumes,
        raw_free,
        equilibrium,
        minimum_free,
        curvature,
        expansion,
        brackets,
        interpolation,
        successful,
        /,
    ):
        self.temperatures = jnp.asarray(temperatures)
        self.volumes = jnp.asarray(volumes, dtype=self.temperatures.dtype)
        self.raw_free_energies = jnp.asarray(raw_free, dtype=self.temperatures.dtype)
        self.equilibrium_volumes = jnp.asarray(equilibrium, dtype=self.temperatures.dtype)
        self.minimum_free_energies = jnp.asarray(
            minimum_free, dtype=self.temperatures.dtype
        )
        self.bulk_curvature = jnp.asarray(curvature, dtype=self.temperatures.dtype)
        self.volumetric_thermal_expansion = jnp.asarray(
            expansion, dtype=self.temperatures.dtype
        )
        self.bracket_indices = jnp.asarray(brackets, dtype=jnp.int32)
        self.interpolation_residuals = jnp.asarray(
            interpolation, dtype=self.temperatures.dtype
        )
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.result_id = canonical_fingerprint(
            {
                "kind": "quasi-harmonic-result",
                "arrays": array_tree_fingerprint(
                    {
                        "temperatures": np.asarray(self.temperatures),
                        "volumes": np.asarray(self.volumes),
                        "raw_free": np.asarray(self.raw_free_energies),
                        "equilibrium": np.asarray(self.equilibrium_volumes),
                        "curvature": np.asarray(self.bulk_curvature),
                        "brackets": np.asarray(self.bracket_indices),
                    }
                ),
            }
        )


class QuasiHarmonicPlan(StrictModule, NonTrainableState):
    """Fixed-phase isotropic QHA with a strict interior three-volume bracket."""

    volumes: Array
    static_energies: Array
    frequencies_by_volume: Array
    thermodynamics: HarmonicThermodynamicsPlan
    minimum_boundary_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        volumes: ArrayLike,
        static_energies: ArrayLike,
        frequencies_by_volume: ArrayLike,
        thermodynamics: HarmonicThermodynamicsPlan,
        /,
        *,
        minimum_boundary_margin: float = 0.0,
    ):
        volume = np.asarray(volumes, dtype=float)
        static = np.asarray(static_energies, dtype=float)
        frequency = np.asarray(frequencies_by_volume, dtype=float)
        if not isinstance(thermodynamics, HarmonicThermodynamicsPlan):
            raise TypeError("thermodynamics must be HarmonicThermodynamicsPlan.")
        if (
            volume.ndim != 1
            or volume.size < 5
            or static.shape != volume.shape
            or frequency.ndim != 3
            or frequency.shape[0] != volume.size
            or frequency.shape[1] != thermodynamics.qpoint_weights.size
        ):
            raise ValueError(
                "QHA requires V>=5 volumes, static energies, and aligned (V,Q,B) frequencies."
            )
        if (
            np.any(~np.isfinite(volume))
            or np.any(~np.isfinite(static))
            or np.any(~np.isfinite(frequency))
            or np.any(frequency <= 0.0)
            or np.any(volume <= 0.0)
            or np.any(np.diff(volume) <= 0.0)
        ):
            raise ValueError(
                "QHA inputs must be finite, stable, positive, and volume ordered."
            )
        margin = float(minimum_boundary_margin)
        if not isfinite(margin) or margin < 0.0:
            raise ValueError("minimum_boundary_margin must be finite and nonnegative.")
        scalar_work = int(
            volume.size
            * frequency.shape[1]
            * frequency.shape[2]
            * thermodynamics.temperatures.size
        )
        if scalar_work > thermodynamics.maximum_scalar_evaluations:
            raise ValueError("QHA scalar-evaluation capacity exceeded.")
        self.volumes = jnp.asarray(volume)
        self.static_energies = jnp.asarray(static)
        self.frequencies_by_volume = jnp.asarray(frequency)
        self.thermodynamics = thermodynamics
        self.minimum_boundary_margin = margin
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasi-harmonic-plan",
                "thermodynamics": thermodynamics.plan_id,
                "minimum_boundary_margin": margin,
                "arrays": array_tree_fingerprint(
                    {"volumes": volume, "static": static, "frequencies": frequency}
                ),
            }
        )

    def evaluate(self, /) -> QuasiHarmonicResult:
        volume = np.asarray(self.volumes)
        static = np.asarray(self.static_energies)
        temperatures = np.asarray(self.thermodynamics.temperatures)
        rows = []
        for index in range(volume.size):
            vibrational = self.thermodynamics.evaluate(self.frequencies_by_volume[index])
            if not bool(vibrational.successful):
                raise ValueError("A QHA volume has unsuccessful harmonic thermodynamics.")
            rows.append(static[index] + np.asarray(vibrational.free_energy))
        raw = np.stack(rows, axis=1)  # (T,V)
        equilibrium = []
        minimum_free = []
        curvature = []
        brackets = []
        interpolation = []
        for row in raw:
            center = int(np.argmin(row))
            if center == 0 or center == volume.size - 1:
                raise ValueError(
                    "QHA free-energy minimum is not strictly interior; extrapolation is refused."
                )
            x = volume[center - 1 : center + 2]
            y = row[center - 1 : center + 2]
            coefficients = np.polyfit(x, y, 2)
            second = 2.0 * coefficients[0]
            if not np.isfinite(second) or second <= 0.0:
                raise ValueError("QHA interior bracket does not have positive curvature.")
            candidate = -coefficients[1] / second
            lower = x[0] + self.minimum_boundary_margin
            upper = x[-1] - self.minimum_boundary_margin
            if not lower < candidate < upper:
                raise ValueError(
                    "QHA interpolated minimum lies outside the admitted interior bracket."
                )
            fitted = np.polyval(coefficients, x)
            equilibrium.append(candidate)
            minimum_free.append(np.polyval(coefficients, candidate))
            curvature.append(second * candidate)
            brackets.append((center - 1, center, center + 1))
            interpolation.append(float(np.max(np.abs(fitted - y), initial=0.0)))
        equilibrium_ = np.asarray(equilibrium)
        edge_order = 2 if temperatures.size >= 3 else 1
        expansion = (
            np.gradient(equilibrium_, temperatures, edge_order=edge_order) / equilibrium_
        )
        successful = np.all(np.isfinite(raw)) and np.all(np.isfinite(expansion))
        return QuasiHarmonicResult(
            temperatures,
            volume,
            raw,
            equilibrium_,
            minimum_free,
            curvature,
            expansion,
            brackets,
            interpolation,
            successful,
        )


class LatticeConvergenceReport(StrictModule, NonTrainableState):
    """Paired raw estimates and deterministic discretization deltas."""

    coarse: Array
    refined: Array
    absolute_delta: Array
    relative_delta: Array
    successful: Array
    refinement_axis: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse: ArrayLike,
        refined: ArrayLike,
        /,
        *,
        refinement_axis: str,
        absolute_tolerance: float,
        relative_tolerance: float,
    ):
        first = np.asarray(coarse)
        second = np.asarray(refined)
        if (
            first.shape != second.shape
            or np.any(~np.isfinite(first))
            or np.any(~np.isfinite(second))
        ):
            raise ValueError(
                "Convergence estimates must have equal shape and finite values."
            )
        absolute = np.abs(second - first)
        relative = absolute / np.maximum(np.abs(second), np.finfo(float).tiny)
        abs_tol = float(absolute_tolerance)
        rel_tol = float(relative_tolerance)
        if any(not isfinite(value) or value < 0.0 for value in (abs_tol, rel_tol)):
            raise ValueError("Convergence tolerances must be finite and nonnegative.")
        self.coarse = jnp.asarray(first)
        self.refined = jnp.asarray(second)
        self.absolute_delta = jnp.asarray(absolute)
        self.relative_delta = jnp.asarray(relative)
        self.successful = jnp.asarray(
            np.all((absolute <= abs_tol) | (relative <= rel_tol))
        )
        self.refinement_axis = str(refinement_axis)
        if not self.refinement_axis:
            raise ValueError("refinement_axis must be non-empty.")
        self.report_id = canonical_fingerprint(
            {
                "kind": "lattice-convergence-report",
                "axis": self.refinement_axis,
                "absolute_tolerance": abs_tol,
                "relative_tolerance": rel_tol,
                "arrays": array_tree_fingerprint({"coarse": first, "refined": second}),
            }
        )


__all__ = [
    "HarmonicThermodynamicsPlan",
    "HarmonicThermodynamicsResult",
    "LatticeConvergenceReport",
    "QuasiHarmonicPlan",
    "QuasiHarmonicResult",
]
