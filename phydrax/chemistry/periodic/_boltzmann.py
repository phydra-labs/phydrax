#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Constant-positive-scalar-relaxation periodic Boltzmann thermoelectrics.

The relaxation time is caller-supplied physical data in seconds.  It is never
inferred from a Kubo linewidth or a numerical Green-function broadening.
Energies are joules, band velocities are Cartesian metres per second, the cell
volume is cubic metres, and normalized k weights represent one primitive cell.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._reciprocal import ReciprocalMeshPlan
from ...ein import contract
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    HermitianSpectrum,
    LinearSolvePolicy,
    LinearSystem,
    solve,
)
from ...units import conversion_factor, derived_unit, JOULE, METER, SECOND
from ._observables import PeriodicVelocityResult
from ._spectrum import PeriodicSpectrumResult


_BOLTZMANN_CONSTANT_SI = 1.380649e-23
_ELECTRON_CHARGE_SI = -1.602176634e-19
_VELOCITY_UNIT = derived_unit("m/s", ((METER, 1), (SECOND, -1)))


def _positive_scalar(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _relative_symmetry_residual(value: Array, /) -> Array:
    scale = jnp.maximum(jnp.max(jnp.abs(value), initial=0.0), jnp.finfo(value.dtype).tiny)
    return jnp.max(jnp.abs(value - value.T), initial=0.0) / scale


class ConstantRelaxationTime(StrictModule, NonTrainableState):
    """One physical positive scalar elastic relaxation-time assumption."""

    seconds: float = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)

    def __init__(self, seconds: float, /, *, mechanism_id: str):
        relaxation = _positive_scalar(seconds, "relaxation time")
        mechanism = str(mechanism_id).strip()
        if not mechanism:
            raise ValueError("A physical relaxation mechanism identity is required.")
        self.seconds = relaxation
        self.mechanism_id = mechanism


class PeriodicBoltzmannPlan(StrictModule, NonTrainableState):
    """Fixed independent-particle data for zero-field thermoelectric moments."""

    energies_joule: Array
    band_velocities_m_per_s: Array
    k_weights: Array
    relaxation_time: ConstantRelaxationTime
    chemical_potential_joule: float = eqx.field(static=True)
    temperature_kelvin: float = eqx.field(static=True)
    cell_volume_m3: float = eqx.field(static=True)
    spin_degeneracy: int = eqx.field(static=True)
    rank_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies_joule: ArrayLike,
        band_velocities_m_per_s: ArrayLike,
        k_weights: ArrayLike,
        relaxation_time: ConstantRelaxationTime,
        /,
        *,
        chemical_potential_joule: float,
        temperature_kelvin: float,
        cell_volume_m3: float,
        spin_degeneracy: int = 1,
        rank_tolerance: float = 1.0e-12,
    ):
        energies = np.asarray(energies_joule)
        velocities = np.asarray(band_velocities_m_per_s)
        weights = np.asarray(k_weights, dtype=float)
        chemical = float(chemical_potential_joule)
        temperature = _positive_scalar(temperature_kelvin, "temperature_kelvin")
        volume = _positive_scalar(cell_volume_m3, "cell_volume_m3")
        rank_tolerance_ = _positive_scalar(rank_tolerance, "rank_tolerance")
        if not isinstance(relaxation_time, ConstantRelaxationTime):
            raise TypeError("relaxation_time must be ConstantRelaxationTime.")
        if isinstance(spin_degeneracy, bool) or not isinstance(
            spin_degeneracy, (int, np.integer)
        ):
            raise TypeError("spin_degeneracy must be a positive integer.")
        degeneracy = int(spin_degeneracy)
        if degeneracy <= 0:
            raise ValueError("spin_degeneracy must be a positive integer.")
        if (
            energies.ndim != 2
            or velocities.ndim != 3
            or velocities.shape[:2] != energies.shape
            or velocities.shape[2] not in (1, 2, 3)
            or weights.shape != (energies.shape[0],)
            or np.any(~np.isfinite(energies))
            or np.any(~np.isfinite(velocities))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
            or not np.isfinite(chemical)
        ):
            raise ValueError(
                "Boltzmann energies, physical band velocities, k weights, and thermodynamic state are invalid."
            )
        self.energies_joule = jnp.asarray(energies)
        self.band_velocities_m_per_s = jnp.asarray(velocities)
        self.k_weights = jnp.asarray(weights)
        self.relaxation_time = relaxation_time
        self.chemical_potential_joule = chemical
        self.temperature_kelvin = temperature
        self.cell_volume_m3 = volume
        self.spin_degeneracy = degeneracy
        self.rank_tolerance = rank_tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-constant-scalar-tau-boltzmann-plan",
                "chemical_potential_joule": chemical,
                "temperature_kelvin": temperature,
                "cell_volume_m3": volume,
                "spin_degeneracy": degeneracy,
                "rank_tolerance": rank_tolerance_,
                "relaxation_seconds": relaxation_time.seconds,
                "relaxation_mechanism_id": relaxation_time.mechanism_id,
                "arrays": array_tree_fingerprint(
                    {
                        "energies_joule": energies,
                        "band_velocities_m_per_s": velocities,
                        "k_weights": weights,
                    }
                ),
            }
        )

    @property
    def dimension(self) -> int:
        return int(self.band_velocities_m_per_s.shape[-1])

    @classmethod
    def from_periodic_results(
        cls,
        spectrum: PeriodicSpectrumResult,
        velocity: PeriodicVelocityResult,
        mesh: ReciprocalMeshPlan,
        relaxation_time: ConstantRelaxationTime,
        /,
        *,
        chemical_potential: float,
        temperature_kelvin: float,
        cell_volume_m3: float,
        spin_degeneracy: int = 1,
        rank_tolerance: float = 1.0e-12,
    ) -> "PeriodicBoltzmannPlan":
        """Bind canonical spectrum/velocity/mesh results to the SI transport plan."""

        if not isinstance(spectrum, PeriodicSpectrumResult):
            raise TypeError("spectrum must be PeriodicSpectrumResult.")
        if not isinstance(velocity, PeriodicVelocityResult):
            raise TypeError("velocity must be PeriodicVelocityResult.")
        if not isinstance(mesh, ReciprocalMeshPlan):
            raise TypeError("mesh must be ReciprocalMeshPlan.")
        if (
            spectrum.support_id != mesh.mesh_id
            or spectrum.cell_id != mesh.cell_id
            or velocity.spectrum_id != spectrum.result_id
            or velocity.cell_id != spectrum.cell_id
        ):
            raise ValueError(
                "Periodic Boltzmann spectrum, velocity, and reciprocal mesh do not match."
            )
        if (
            mesh.rank != 3
            or mesh.cell.ambient_dimension != 3
            or not mesh.cell.fully_periodic
        ):
            raise ValueError(
                "Bulk thermoelectrics require a fully periodic rank-three cell and physical volume."
            )
        if (
            velocity.derivative_basis != "cartesian-wavevector-m-per-s"
            or not bool(spectrum.successful)
            or not bool(velocity.successful)
        ):
            raise ValueError(
                "Periodic Boltzmann transport requires successful physical Cartesian velocities."
            )
        energy_scale = float(conversion_factor(spectrum.energy_unit, JOULE))
        velocity_scale = float(conversion_factor(velocity.velocity_unit, _VELOCITY_UNIT))
        return cls(
            spectrum.energies * energy_scale,
            velocity.band_velocities * velocity_scale,
            mesh.weights,
            relaxation_time,
            chemical_potential_joule=float(chemical_potential) * energy_scale,
            temperature_kelvin=temperature_kelvin,
            cell_volume_m3=cell_volume_m3,
            spin_degeneracy=spin_degeneracy,
            rank_tolerance=rank_tolerance,
        )

    def evaluate(self, /) -> "BoltzmannThermoelectricResult":
        """Evaluate L0/L1/L2 and full-rank open-circuit thermoelectrics."""

        temperature = self.temperature_kelvin
        occupations = jax.nn.sigmoid(
            (self.chemical_potential_joule - self.energies_joule)
            / (_BOLTZMANN_CONSTANT_SI * temperature)
        )
        minus_derivative = (
            occupations * (1.0 - occupations) / (_BOLTZMANN_CONSTANT_SI * temperature)
        )
        offset = self.energies_joule - self.chemical_potential_joule
        base = (
            self.spin_degeneracy
            * self.relaxation_time.seconds
            / self.cell_volume_m3
            * self.k_weights[:, None]
            * minus_derivative
        )
        outer_velocity = contract(
            "kna,knb->knab",
            self.band_velocities_m_per_s,
            self.band_velocities_m_per_s,
            backend="jax",
        )
        moment_zero = contract("kn,knab->ab", base, outer_velocity, backend="jax")
        moment_one = contract(
            "kn,kn,knab->ab", base, offset, outer_velocity, backend="jax"
        )
        moment_two = contract(
            "kn,kn,knab->ab", base, offset**2, outer_velocity, backend="jax"
        )
        spectrum = HermitianSpectrum(moment_zero, tolerance=self.rank_tolerance)
        magnitude = jnp.max(jnp.abs(spectrum.eigenvalues), initial=0.0)
        threshold = self.rank_tolerance * magnitude
        rank = jnp.sum(spectrum.eigenvalues > threshold)
        if int(rank) < self.dimension:
            raise ValueError(
                "Boltzmann electrical transport moment is rank deficient on the sampled velocity support."
            )
        solved = solve(
            LinearSystem(
                DenseLinearOperator(moment_zero),
                problem_id="boltzmann-open-circuit-thermoelectric",
            ),
            moment_one,
            policy=LinearSolvePolicy(
                DenseLU(),
                failure=FailurePolicy("error"),
            ),
        )
        reduced_first = solved.value
        conductivity = _ELECTRON_CHARGE_SI**2 * moment_zero
        seebeck = reduced_first / (_ELECTRON_CHARGE_SI * temperature)
        peltier = temperature * seebeck
        thermal = (moment_two - moment_one @ reduced_first) / temperature
        thermal = 0.5 * (thermal + thermal.T)
        sigma_spectrum = HermitianSpectrum(conductivity, tolerance=self.rank_tolerance)
        thermal_spectrum = HermitianSpectrum(thermal, tolerance=self.rank_tolerance)
        moment_symmetry = jnp.max(
            jnp.stack(
                tuple(
                    _relative_symmetry_residual(value)
                    for value in (moment_zero, moment_one, moment_two)
                )
            )
        )
        onsager_scale = jnp.maximum(
            jnp.max(jnp.abs(peltier), initial=0.0),
            jnp.finfo(peltier.dtype).tiny,
        )
        onsager = (
            jnp.max(jnp.abs(peltier - temperature * seebeck), initial=0.0) / onsager_scale
        )
        finite = jnp.all(
            jnp.stack(
                tuple(
                    jnp.all(jnp.isfinite(value))
                    for value in (
                        moment_zero,
                        moment_one,
                        moment_two,
                        conductivity,
                        seebeck,
                        peltier,
                        thermal,
                    )
                )
            )
        )
        conductivity_scale = jnp.maximum(
            jnp.max(jnp.abs(sigma_spectrum.eigenvalues), initial=0.0),
            jnp.finfo(sigma_spectrum.eigenvalues.dtype).tiny,
        )
        thermal_scale = jnp.maximum(
            jnp.maximum(
                jnp.max(jnp.abs(thermal_spectrum.eigenvalues), initial=0.0),
                jnp.max(jnp.abs(moment_two / temperature), initial=0.0),
            ),
            jnp.finfo(thermal_spectrum.eigenvalues.dtype).tiny,
        )
        passive = (
            sigma_spectrum.valid
            & thermal_spectrum.valid
            & (
                sigma_spectrum.minimum_eigenvalue
                >= -self.rank_tolerance * conductivity_scale
            )
            & (
                thermal_spectrum.minimum_eigenvalue
                >= -self.rank_tolerance * thermal_scale
            )
        )
        evidence = BoltzmannTransportEvidence(
            rank,
            moment_symmetry,
            onsager,
            sigma_spectrum.minimum_eigenvalue,
            thermal_spectrum.minimum_eigenvalue,
            finite,
            passive,
            self.relaxation_time.mechanism_id,
            False,
        )
        return BoltzmannThermoelectricResult(
            occupations,
            moment_zero,
            moment_one,
            moment_two,
            conductivity,
            seebeck,
            peltier,
            thermal,
            evidence,
            finite
            & passive
            & jnp.all(solved.successful)
            & (moment_symmetry <= 10.0 * self.rank_tolerance)
            & (onsager <= self.rank_tolerance),
            self.plan_id,
        )


class BoltzmannTransportEvidence(StrictModule, NonTrainableState):
    """Rank, passivity, reciprocity, and assumption evidence."""

    electrical_rank: Array
    moment_symmetry_relative_residual: Array
    kelvin_onsager_relative_residual: Array
    minimum_conductivity_eigenvalue: Array
    minimum_thermal_conductivity_eigenvalue: Array
    finite: Array
    passive: Array
    relaxation_mechanism_id: str = eqx.field(static=True)
    relaxation_inferred_from_linewidth: bool = eqx.field(static=True)


class BoltzmannThermoelectricResult(StrictModule, NonTrainableState):
    """Zero-field, constant-tau thermoelectric tensors in SI units."""

    occupations: Array
    moment_zero: Array
    moment_one_joule: Array
    moment_two_joule2: Array
    electrical_conductivity_siemens_per_m: Array
    seebeck_volt_per_kelvin: Array
    peltier_volt: Array
    electronic_thermal_conductivity_watt_per_m_kelvin: Array
    evidence: BoltzmannTransportEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


__all__ = [
    "BoltzmannThermoelectricResult",
    "BoltzmannTransportEvidence",
    "ConstantRelaxationTime",
    "PeriodicBoltzmannPlan",
]
