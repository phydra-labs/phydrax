#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent-particle periodic Kubo transitions and optical conductivity.

Energies are explicit joules, velocities are physical Cartesian meters per
second, normalized k weights represent one primitive cell, and cell volume is
cubic meters.  The regular finite-frequency response and zero-frequency Drude
distribution are never conflated.  A linewidth broadens only named interband
transitions; it is neither a relaxation time nor a finite ballistic DC law.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.operators.periodic import PeriodicSpectrumResult

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._reciprocal import ReciprocalMeshPlan
from ...ein import contract
from ...linalg import HermitianSpectrum
from ...units import conversion_factor, derived_unit, JOULE, METER, SECOND
from ._observables import PeriodicVelocityResult


_BOLTZMANN_CONSTANT_SI = 1.380649e-23
_ELECTRON_CHARGE_MAGNITUDE_SI = 1.602176634e-19
_HBAR_SI = 1.0545718176461565e-34
_VELOCITY_UNIT = derived_unit("m/s", ((METER, 1), (SECOND, -1)))


def _adjoint(value: Array, /) -> Array:
    return jnp.swapaxes(jnp.conj(value), -1, -2)


def _positive_scalar(value: float, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return result


def _fermi(energy: Array, chemical_potential: float, temperature: float, /) -> Array:
    argument = (chemical_potential - energy) / (_BOLTZMANN_CONSTANT_SI * temperature)
    return jax.nn.sigmoid(argument)


class KuboDiamagneticSumRule(StrictModule, NonTrainableState):
    """Independent diamagnetic spectral weight used to test the f-sum rule."""

    spectral_weight: Array
    source_id: str = eqx.field(static=True)

    def __init__(self, spectral_weight: ArrayLike, /, *, source_id: str):
        weight = np.asarray(spectral_weight)
        source = str(source_id).strip()
        if (
            weight.ndim != 2
            or weight.shape[0] != weight.shape[1]
            or np.any(~np.isfinite(weight))
            or not np.allclose(weight, weight.T.conj(), atol=1.0e-12, rtol=1.0e-12)
            or not source
        ):
            raise ValueError(
                "Diamagnetic spectral weight must be one finite Hermitian tensor with a source identity."
            )
        self.spectral_weight = jnp.asarray(weight)
        self.source_id = source


class KuboLinewidth(StrictModule, NonTrainableState):
    """Named physical interband energy linewidth, explicitly not numerical eta."""

    energy_width_joule: float = eqx.field(static=True)
    mechanism_id: str = eqx.field(static=True)

    def __init__(self, energy_width_joule: float, /, *, mechanism_id: str):
        width = _positive_scalar(energy_width_joule, "energy_width_joule")
        mechanism = str(mechanism_id).strip()
        if not mechanism:
            raise ValueError("A physical linewidth mechanism identity is required.")
        self.energy_width_joule = width
        self.mechanism_id = mechanism


class PeriodicKuboPlan(StrictModule, NonTrainableState):
    """Fixed independent-particle data for charge-current Kubo response."""

    energies_joule: Array
    velocity_matrices_m_per_s: Array
    k_weights: Array
    degeneracy_mask: Array
    diamagnetic_sum_rule: KuboDiamagneticSumRule
    chemical_potential_joule: float = eqx.field(static=True)
    temperature_kelvin: float = eqx.field(static=True)
    cell_volume_m3: float = eqx.field(static=True)
    spin_degeneracy: int = eqx.field(static=True)
    degeneracy_tolerance_joule: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        energies_joule: ArrayLike,
        velocity_matrices_m_per_s: ArrayLike,
        k_weights: ArrayLike,
        diamagnetic_sum_rule: KuboDiamagneticSumRule,
        /,
        *,
        chemical_potential_joule: float,
        temperature_kelvin: float,
        cell_volume_m3: float,
        spin_degeneracy: int = 1,
        degeneracy_tolerance_joule: float = 1.0e-12 * _ELECTRON_CHARGE_MAGNITUDE_SI,
        degeneracy_mask: ArrayLike | None = None,
    ):
        energies = np.asarray(energies_joule)
        velocities = np.asarray(velocity_matrices_m_per_s)
        weights = np.asarray(k_weights, dtype=np.float64)
        mask = (
            np.abs(energies[:, :, None] - energies[:, None, :])
            <= float(degeneracy_tolerance_joule)
            if degeneracy_mask is None
            else np.asarray(degeneracy_mask)
        )
        chemical = float(chemical_potential_joule)
        temperature = _positive_scalar(temperature_kelvin, "temperature_kelvin")
        volume = _positive_scalar(cell_volume_m3, "cell_volume_m3")
        tolerance = _positive_scalar(
            degeneracy_tolerance_joule, "degeneracy_tolerance_joule"
        )
        if isinstance(spin_degeneracy, bool) or not isinstance(
            spin_degeneracy, (int, np.integer)
        ):
            raise TypeError("spin_degeneracy must be a positive integer.")
        degeneracy = int(spin_degeneracy)
        if degeneracy <= 0:
            raise ValueError("spin_degeneracy must be a positive integer.")
        if (
            energies.ndim != 2
            or velocities.ndim != 4
            or velocities.shape[:3]
            != (energies.shape[0], energies.shape[1], energies.shape[1])
            or velocities.shape[3] not in (1, 2, 3)
            or weights.shape != (energies.shape[0],)
            or np.any(~np.isfinite(energies))
            or np.any(~np.isfinite(velocities))
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or not np.isclose(np.sum(weights), 1.0, atol=1.0e-12)
            or not np.isfinite(chemical)
            or mask.shape != (energies.shape[0], energies.shape[1], energies.shape[1])
            or mask.dtype.kind != "b"
            or not np.all(mask == mask.swapaxes(1, 2))
            or not np.all(np.diagonal(mask, axis1=1, axis2=2))
        ):
            raise ValueError(
                "Kubo energies, physical velocity matrices, k weights, and thermodynamic state are invalid."
            )
        if np.any(np.diff(energies, axis=1) < -tolerance):
            raise ValueError("Kubo band energies must be nondecreasing at every k point.")
        hermiticity = np.max(
            np.abs(velocities - velocities.swapaxes(1, 2).conj()), initial=0.0
        )
        velocity_scale = max(float(np.max(np.abs(velocities), initial=0.0)), 1.0)
        if hermiticity > 1.0e-10 * velocity_scale:
            raise ValueError("Every Cartesian band-velocity matrix must be Hermitian.")
        if not isinstance(diamagnetic_sum_rule, KuboDiamagneticSumRule):
            raise TypeError("diamagnetic_sum_rule must be KuboDiamagneticSumRule.")
        dimension = velocities.shape[-1]
        if diamagnetic_sum_rule.spectral_weight.shape != (dimension, dimension):
            raise ValueError("Diamagnetic and velocity Cartesian dimensions must agree.")
        self.energies_joule = jnp.asarray(energies)
        self.velocity_matrices_m_per_s = jnp.asarray(velocities)
        self.k_weights = jnp.asarray(weights)
        self.degeneracy_mask = jnp.asarray(mask)
        self.diamagnetic_sum_rule = diamagnetic_sum_rule
        self.chemical_potential_joule = chemical
        self.temperature_kelvin = temperature
        self.cell_volume_m3 = volume
        self.spin_degeneracy = degeneracy
        self.degeneracy_tolerance_joule = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-independent-particle-kubo-plan",
                "chemical_potential_joule": chemical,
                "temperature_kelvin": temperature,
                "cell_volume_m3": volume,
                "spin_degeneracy": degeneracy,
                "degeneracy_tolerance_joule": tolerance,
                "diamagnetic_source_id": diamagnetic_sum_rule.source_id,
                "arrays": array_tree_fingerprint(
                    {
                        "energies_joule": energies,
                        "velocity_matrices_m_per_s": velocities,
                        "k_weights": weights,
                        "degeneracy_mask": mask,
                    }
                ),
            }
        )

    @property
    def dimension(self) -> int:
        return self.velocity_matrices_m_per_s.shape[-1]

    @classmethod
    def from_periodic_results(
        cls,
        spectrum: PeriodicSpectrumResult,
        velocity: PeriodicVelocityResult,
        mesh: ReciprocalMeshPlan,
        diamagnetic_sum_rule: KuboDiamagneticSumRule,
        /,
        *,
        chemical_potential: float,
        temperature_kelvin: float,
        cell_volume_m3: float,
        spin_degeneracy: int = 1,
        degeneracy_tolerance_joule: float = 1.0e-12 * _ELECTRON_CHARGE_MAGNITUDE_SI,
    ) -> "PeriodicKuboPlan":
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
                "Periodic Kubo spectrum, velocity, and reciprocal mesh do not match."
            )
        if (
            mesh.rank != 3
            or mesh.cell.ambient_dimension != 3
            or not mesh.cell.fully_periodic
        ):
            raise ValueError(
                "Bulk conductivity requires a fully periodic rank-three cell and physical volume."
            )
        if (
            velocity.derivative_basis != "cartesian-wavevector-m-per-s"
            or not bool(spectrum.successful)
            or not bool(velocity.successful)
        ):
            raise ValueError(
                "Periodic Kubo requires successful physical Cartesian velocities."
            )
        energy_scale = float(conversion_factor(spectrum.energy_unit, JOULE))
        velocity_scale = float(conversion_factor(velocity.velocity_unit, _VELOCITY_UNIT))
        return cls(
            spectrum.energies * energy_scale,
            velocity.velocity_matrices * velocity_scale,
            mesh.weights,
            diamagnetic_sum_rule,
            chemical_potential_joule=float(chemical_potential) * energy_scale,
            temperature_kelvin=temperature_kelvin,
            cell_volume_m3=cell_volume_m3,
            spin_degeneracy=spin_degeneracy,
            degeneracy_tolerance_joule=degeneracy_tolerance_joule,
            degeneracy_mask=velocity.degeneracy_mask,
        )

    def raw_transitions(self, /) -> "KuboRawTransitions":
        """Return positive-energy raw transitions and a separate Drude weight."""

        energies = self.energies_joule
        velocities = self.velocity_matrices_m_per_s
        occupations = _fermi(
            energies, self.chemical_potential_joule, self.temperature_kelvin
        )
        delta = energies[:, None, :] - energies[:, :, None]
        occupation_difference = occupations[:, :, None] - occupations[:, None, :]
        bands = energies.shape[1]
        upper = jnp.triu(jnp.ones((bands, bands), dtype=jnp.bool_), k=1)
        active = upper[None, :, :] & (delta > self.degeneracy_tolerance_joule)
        safe_delta = jnp.where(active, delta, 1.0)
        transition_factor = jnp.where(
            active,
            self.spin_degeneracy
            * self.k_weights[:, None, None]
            * occupation_difference
            / safe_delta,
            0.0,
        )
        velocity_products = contract(
            "knma,knmb->knmab", velocities, jnp.conj(velocities), backend="jax"
        )

        derivative = (
            occupations
            * (1.0 - occupations)
            / (_BOLTZMANN_CONSTANT_SI * self.temperature_kelvin)
        )
        degenerate = self.degeneracy_mask
        cluster_factor = (
            self.spin_degeneracy
            * self.k_weights[:, None, None]
            * 0.5
            * (derivative[:, :, None] + derivative[:, None, :])
            * degenerate
        )
        drude_weight = (
            np.pi
            * _ELECTRON_CHARGE_MAGNITUDE_SI**2
            / self.cell_volume_m3
            * jnp.real(
                contract(
                    "knm,knmab->ab", cluster_factor, velocity_products, backend="jax"
                )
            )
        )
        regular_weight = (
            np.pi
            * _ELECTRON_CHARGE_MAGNITUDE_SI**2
            / self.cell_volume_m3
            * jnp.real(
                contract(
                    "knm,knmab->ab", transition_factor, velocity_products, backend="jax"
                )
            )
        )
        observed = 0.5 * drude_weight + regular_weight
        expected = self.diamagnetic_sum_rule.spectral_weight
        scale = jnp.maximum(
            jnp.max(jnp.abs(expected), initial=0.0),
            jnp.finfo(expected.real.dtype).tiny,
        )
        f_sum_residual = jnp.max(jnp.abs(observed - expected), initial=0.0) / scale
        regular_spectrum = HermitianSpectrum(regular_weight, tolerance=1.0e-10)
        drude_spectrum = HermitianSpectrum(drude_weight, tolerance=1.0e-10)
        return KuboRawTransitions(
            energies,
            occupations,
            delta,
            occupation_difference,
            active,
            transition_factor,
            velocity_products,
            drude_weight,
            regular_weight,
            observed,
            expected,
            f_sum_residual,
            jnp.minimum(
                regular_spectrum.minimum_eigenvalue,
                drude_spectrum.minimum_eigenvalue,
            ),
            regular_spectrum.valid & drude_spectrum.valid,
            self.plan_id,
            self.diamagnetic_sum_rule.source_id,
        )


class KuboRawTransitions(StrictModule, NonTrainableState):
    """Unbroadened positive-energy transitions plus a zero-frequency Drude weight."""

    energies_joule: Array
    occupations: Array
    transition_energies_joule: Array
    occupation_differences: Array
    active: Array
    transition_factors: Array
    velocity_products: Array
    drude_weight: Array
    regular_spectral_weight: Array
    observed_f_sum_weight: Array
    expected_f_sum_weight: Array
    f_sum_relative_residual: Array
    minimum_spectral_weight_eigenvalue: Array
    passive: Array
    plan_id: str = eqx.field(static=True)
    diamagnetic_source_id: str = eqx.field(static=True)
    drude_representation: str = eqx.field(
        static=True,
        default="zero-frequency delta distribution; never linewidth-broadened",
    )


class KuboResponseEvidence(StrictModule, NonTrainableState):
    """Passivity and sum-rule evidence for regular finite-frequency response."""

    minimum_dissipative_eigenvalue: Array
    f_sum_relative_residual: Array
    finite: Array
    passive: Array
    sum_rule_satisfied: Array
    linewidth_mechanism_id: str = eqx.field(static=True)
    includes_drude: bool = eqx.field(static=True)
    infers_relaxation_time: bool = eqx.field(static=True)


class FiniteFrequencyKuboResponse(StrictModule, NonTrainableState):
    """Regular optical conductivity on strictly positive angular frequencies."""

    angular_frequencies_rad_per_s: Array
    regular_conductivity_siemens_per_m: Array
    raw: KuboRawTransitions
    evidence: KuboResponseEvidence
    successful: Array
    plan_id: str = eqx.field(static=True)


def finite_frequency_kubo_response(
    plan: PeriodicKuboPlan,
    angular_frequencies_rad_per_s: ArrayLike,
    linewidth: KuboLinewidth,
    /,
    *,
    f_sum_tolerance: float = 1.0e-8,
    passivity_tolerance: float = 1.0e-10,
) -> FiniteFrequencyKuboResponse:
    """Broaden only the regular interband response with a named linewidth."""

    if not isinstance(plan, PeriodicKuboPlan):
        raise TypeError("plan must be PeriodicKuboPlan.")
    if not isinstance(linewidth, KuboLinewidth):
        raise TypeError("linewidth must be KuboLinewidth.")
    sum_tolerance = _positive_scalar(f_sum_tolerance, "f_sum_tolerance")
    passive_tolerance = _positive_scalar(passivity_tolerance, "passivity_tolerance")
    frequencies = np.asarray(angular_frequencies_rad_per_s, dtype=np.float64)
    if (
        frequencies.ndim != 1
        or frequencies.size == 0
        or np.any(~np.isfinite(frequencies))
        or np.any(frequencies <= 0.0)
        or np.any(np.diff(frequencies) <= 0.0)
    ):
        raise ValueError(
            "Finite-frequency Kubo response requires strictly increasing positive angular frequencies."
        )
    raw = plan.raw_transitions()
    omega = jnp.asarray(frequencies)
    transition_frequency = raw.transition_energies_joule / _HBAR_SI
    rate = linewidth.energy_width_joule / _HBAR_SI
    resonant = 1.0 / (
        omega[:, None, None, None] - transition_frequency[None, :, :, :] + 1j * rate
    )
    antiresonant = 1.0 / (
        omega[:, None, None, None] + transition_frequency[None, :, :, :] + 1j * rate
    )
    weighted = raw.transition_factors[None, :, :, :, None, None]
    product = raw.velocity_products[None, :, :, :, :, :]
    conductivity = (
        1j
        * _ELECTRON_CHARGE_MAGNITUDE_SI**2
        / plan.cell_volume_m3
        * jnp.sum(
            weighted
            * (
                product * resonant[..., None, None]
                + jnp.swapaxes(product, -1, -2) * antiresonant[..., None, None]
            ),
            axis=(1, 2, 3),
        )
    )
    dissipative = 0.5 * (conductivity + _adjoint(conductivity))
    spectra = HermitianSpectrum(dissipative, tolerance=passive_tolerance)
    scale = jnp.maximum(
        jnp.max(jnp.abs(spectra.eigenvalues), axis=-1),
        jnp.finfo(spectra.eigenvalues.dtype).tiny,
    )
    passive = jnp.all(spectra.minimum_eigenvalue >= -passive_tolerance * scale) & jnp.all(
        spectra.valid
    )
    finite = jnp.all(jnp.isfinite(conductivity))
    sum_rule = raw.f_sum_relative_residual <= sum_tolerance
    evidence = KuboResponseEvidence(
        jnp.min(spectra.minimum_eigenvalue),
        raw.f_sum_relative_residual,
        finite,
        passive,
        sum_rule,
        linewidth.mechanism_id,
        False,
        False,
    )
    return FiniteFrequencyKuboResponse(
        omega,
        conductivity,
        raw,
        evidence,
        finite & passive & sum_rule,
        plan.plan_id,
    )


class ConservedCollinearSpinEvidence(StrictModule, NonTrainableState):
    """Evidence that a supplied band-basis Sz closes under the Hamiltonian."""

    commutator_relative_residual: Array
    hermiticity_relative_residual: Array
    finite: Array
    conserved: Array
    tolerance: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)


def conserved_collinear_spin_evidence(
    energies_joule: ArrayLike,
    spin_z_matrices: ArrayLike,
    /,
    *,
    source_id: str,
    tolerance: float = 1.0e-10,
) -> ConservedCollinearSpinEvidence:
    """Test ``[diag(E_k), Sz_k]=0`` before admitting collinear spin response."""

    tolerance_ = _positive_scalar(tolerance, "tolerance")
    energies = jnp.asarray(energies_joule)
    spin = jnp.asarray(spin_z_matrices)
    source = str(source_id).strip()
    if (
        energies.ndim != 2
        or spin.shape != (energies.shape[0], energies.shape[1], energies.shape[1])
        or not source
    ):
        raise ValueError(
            "Spin evidence requires aligned band energies, Sz matrices, and source identity."
        )
    commutator = (energies[:, :, None] - energies[:, None, :]) * spin
    energy_scale = jnp.maximum(
        jnp.max(jnp.abs(energies), initial=0.0),
        jnp.finfo(energies.real.dtype).tiny,
    )
    spin_scale = jnp.maximum(
        jnp.max(jnp.abs(spin), initial=0.0),
        jnp.finfo(spin.real.dtype).tiny,
    )
    commutator_residual = jnp.max(jnp.abs(commutator), initial=0.0) / (
        energy_scale * spin_scale
    )
    hermiticity_residual = (
        jnp.max(jnp.abs(spin - _adjoint(spin)), initial=0.0) / spin_scale
    )
    finite = jnp.all(jnp.isfinite(energies)) & jnp.all(jnp.isfinite(spin))
    conserved = (
        finite
        & (commutator_residual <= tolerance_)
        & (hermiticity_residual <= tolerance_)
    )
    return ConservedCollinearSpinEvidence(
        commutator_residual,
        hermiticity_residual,
        finite,
        conserved,
        tolerance_,
        source,
    )


def require_conserved_collinear_spin(evidence: ConservedCollinearSpinEvidence, /) -> None:
    """Refuse spin response unless the supplied collinear generator closes."""

    if not isinstance(evidence, ConservedCollinearSpinEvidence):
        raise TypeError("evidence must be ConservedCollinearSpinEvidence.")
    if not bool(evidence.conserved):
        raise ValueError("Collinear spin response requires [H,Sz]=0 closure evidence.")


__all__ = [
    "ConservedCollinearSpinEvidence",
    "FiniteFrequencyKuboResponse",
    "KuboDiamagneticSumRule",
    "KuboLinewidth",
    "KuboRawTransitions",
    "KuboResponseEvidence",
    "PeriodicKuboPlan",
    "conserved_collinear_spin_evidence",
    "finite_frequency_kubo_response",
    "require_conserved_collinear_spin",
]
