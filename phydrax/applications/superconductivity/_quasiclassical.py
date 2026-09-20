#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


_REDUCED_PLANCK = 1.054571817e-34


def _a_derivative(value, delta, omega, speed):
    return (delta - 2.0 * omega * value - jnp.conj(delta) * value**2) / (
        _REDUCED_PLANCK * speed
    )


def _b_derivative(value, delta, omega, speed):
    return -(jnp.conj(delta) - 2.0 * omega * value - delta * value**2) / (
        _REDUCED_PLANCK * speed
    )


class FermiSurfacePlan(StrictModule, NonTrainableState):
    velocities: Array
    weights: Array
    form_factors: Array
    channel_labels: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        velocities: ArrayLike,
        weights: ArrayLike,
        form_factors: ArrayLike,
        channel_labels: tuple[str, ...],
        /,
    ):
        velocity = np.asarray(velocities, dtype=np.float64)
        weight = np.asarray(weights, dtype=np.float64)
        factors = np.asarray(form_factors, dtype=np.complex128)
        labels = tuple(str(value).strip() for value in channel_labels)
        if (
            velocity.ndim != 2
            or velocity.shape[1] != 2
            or weight.shape != (velocity.shape[0],)
            or factors.shape != (len(labels), velocity.shape[0])
            or not labels
            or len(set(labels)) != len(labels)
            or any(not value for value in labels)
            or np.any(~np.isfinite(velocity))
            or np.any(np.linalg.norm(velocity, axis=-1) <= 0.0)
            or np.any(~np.isfinite(weight))
            or np.any(weight <= 0.0)
            or not np.isclose(np.sum(weight), 1.0)
            or np.any(~np.isfinite(factors))
        ):
            raise ValueError(
                "Fermi-surface velocities, weights, or channels are invalid."
            )
        self.velocities = jnp.asarray(velocity)
        self.weights = jnp.asarray(weight)
        self.form_factors = jnp.asarray(factors)
        self.channel_labels = labels
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasiclassical-fermi-surface",
                "velocities": array_tree_fingerprint(velocity),
                "weights": array_tree_fingerprint(weight),
                "form_factors": array_tree_fingerprint(factors),
                "channels": labels,
            }
        )

    @property
    def trajectory_count(self) -> int:
        return self.velocities.shape[0]

    @property
    def channel_count(self) -> int:
        return len(self.channel_labels)


class MatsubaraQuadraturePlan(StrictModule, NonTrainableState):
    temperature_energy: float = eqx.field(static=True)
    frequencies: Array
    plan_id: str = eqx.field(static=True)

    def __init__(self, temperature_energy: float, frequency_count: int, /):
        temperature = float(temperature_energy)
        count = int(frequency_count)
        if not isfinite(temperature) or temperature <= 0.0 or count < 1:
            raise ValueError("Matsubara temperature and frequency count are invalid.")
        frequencies = (2.0 * np.arange(count) + 1.0) * np.pi * temperature
        self.temperature_energy = temperature
        self.frequencies = jnp.asarray(frequencies)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fermionic-matsubara-quadrature",
                "temperature_energy": temperature,
                "frequency_count": count,
            }
        )

    @property
    def frequency_count(self) -> int:
        return self.frequencies.size


class RiccatiTrajectoryEvidence(StrictModule):
    normalization_residual: Array
    minimum_denominator: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class RiccatiTrajectoryResult(StrictModule):
    coherence_a: Array
    coherence_b: Array
    normal_green: Array
    anomalous_green: Array
    conjugate_anomalous_green: Array
    evidence: RiccatiTrajectoryEvidence
    plan_id: str = eqx.field(static=True)


class RiccatiTrajectoryPlan(StrictModule, NonTrainableState):
    fermi_surface: FermiSurfacePlan
    segment_lengths: Array
    vector_potential_coupling: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fermi_surface: FermiSurfacePlan,
        segment_lengths: ArrayLike,
        /,
        *,
        vector_potential_coupling: float = 0.0,
        tolerance: float = 1.0e-9,
    ):
        if not isinstance(fermi_surface, FermiSurfacePlan):
            raise TypeError("fermi_surface must be FermiSurfacePlan.")
        lengths = np.asarray(segment_lengths, dtype=np.float64)
        coupling, tolerance_ = float(vector_potential_coupling), float(tolerance)
        if (
            lengths.ndim != 2
            or lengths.shape[0] != fermi_surface.trajectory_count
            or lengths.shape[1] < 1
            or np.any(~np.isfinite(lengths))
            or np.any(lengths <= 0.0)
            or not isfinite(coupling)
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Riccati trajectories or numerical controls are invalid.")
        self.fermi_surface = fermi_surface
        self.segment_lengths = jnp.asarray(lengths)
        self.vector_potential_coupling = coupling
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "specular-quasiclassical-riccati-trajectories",
                "fermi_surface": fermi_surface.plan_id,
                "segment_lengths": array_tree_fingerprint(lengths),
                "vector_potential_coupling": coupling,
                "tolerance": tolerance_,
            }
        )

    @property
    def segment_count(self) -> int:
        return self.segment_lengths.shape[1]

    @staticmethod
    def _bulk_coherence(gap, frequency):
        omega = jnp.sqrt(frequency**2 + jnp.abs(gap) ** 2)
        denominator = frequency + omega
        return gap / denominator, jnp.conj(gap) / denominator

    def evaluate(
        self,
        matsubara: MatsubaraQuadraturePlan,
        gap: ArrayLike,
        vector_potential_shift: ArrayLike = 0.0,
        /,
    ) -> RiccatiTrajectoryResult:
        if not isinstance(matsubara, MatsubaraQuadraturePlan):
            raise TypeError("matsubara must be MatsubaraQuadraturePlan.")
        gap_ = jnp.asarray(gap)
        expected = (self.fermi_surface.trajectory_count, self.segment_count)
        if gap_.shape != expected or not jnp.iscomplexobj(gap_):
            raise ValueError("Riccati gap must have shape (trajectory, segment).")
        shift = jnp.broadcast_to(
            jnp.asarray(vector_potential_shift, dtype=jnp.real(gap_).dtype), expected
        )
        speed = jnp.linalg.norm(self.fermi_surface.velocities, axis=-1)
        a_values = []
        b_values = []
        for frequency in matsubara.frequencies:
            effective = frequency + 1.0j * self.vector_potential_coupling * shift
            a0, _ = self._bulk_coherence(gap_[:, 0], effective[:, 0])
            a_segments = [a0]
            current = a0
            for segment in range(1, self.segment_count):
                ds = self.segment_lengths[:, segment - 1]
                delta = gap_[:, segment - 1]
                omega = effective[:, segment - 1]

                k1 = _a_derivative(current, delta, omega, speed)
                k2 = _a_derivative(current + 0.5 * ds * k1, delta, omega, speed)
                k3 = _a_derivative(current + 0.5 * ds * k2, delta, omega, speed)
                k4 = _a_derivative(current + ds * k3, delta, omega, speed)
                current = current + ds / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                a_segments.append(current)
            _, b0 = self._bulk_coherence(gap_[:, -1], effective[:, -1])
            b_segments = [b0]
            current = b0
            for segment in range(self.segment_count - 2, -1, -1):
                ds = self.segment_lengths[:, segment + 1]
                delta = gap_[:, segment + 1]
                omega = effective[:, segment + 1]

                k1 = _b_derivative(current, delta, omega, speed)
                k2 = _b_derivative(current + 0.5 * ds * k1, delta, omega, speed)
                k3 = _b_derivative(current + 0.5 * ds * k2, delta, omega, speed)
                k4 = _b_derivative(current + ds * k3, delta, omega, speed)
                current = current + ds / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                b_segments.append(current)
            a_values.append(jnp.stack(tuple(a_segments), axis=-1))
            b_values.append(jnp.stack(tuple(reversed(b_segments)), axis=-1))
        a = jnp.stack(tuple(a_values))
        b = jnp.stack(tuple(b_values))
        denominator = 1.0 + a * b
        normal = (1.0 - a * b) / denominator
        anomalous = 2.0 * a / denominator
        conjugate = 2.0 * b / denominator
        normalization = normal**2 + anomalous * conjugate - 1.0
        residual = jnp.max(jnp.abs(normalization), initial=0.0)
        minimum = jnp.min(jnp.abs(denominator))
        finite = (
            jnp.all(jnp.isfinite(a))
            & jnp.all(jnp.isfinite(b))
            & jnp.all(jnp.isfinite(normal))
        )
        successful = finite & (minimum > self.tolerance) & (residual <= self.tolerance)
        evidence = RiccatiTrajectoryEvidence(
            residual, minimum, finite, successful, self.plan_id
        )
        return RiccatiTrajectoryResult(
            a, b, normal, anomalous, conjugate, evidence, self.plan_id
        )


class QuasiclassicalEquilibriumEvidence(StrictModule):
    gap_residual: Array
    normalization_residual: Array
    free_energy_finite: Array
    converged: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class QuasiclassicalEquilibriumResult(StrictModule):
    channel_amplitudes: Array
    trajectory_gap: Array
    propagator: RiccatiTrajectoryResult
    current_density: Array
    free_energy: Array
    evidence: QuasiclassicalEquilibriumEvidence
    branch_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class QuasiclassicalSuperconductivityPlan(StrictModule, NonTrainableState):
    trajectories: RiccatiTrajectoryPlan
    matsubara: MatsubaraQuadraturePlan
    coupling_matrix: Array
    damping: float = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        trajectories: RiccatiTrajectoryPlan,
        matsubara: MatsubaraQuadraturePlan,
        coupling_matrix: ArrayLike,
        /,
        *,
        damping: float = 0.5,
        iterations: int = 128,
        tolerance: float = 1.0e-8,
    ):
        if not isinstance(trajectories, RiccatiTrajectoryPlan) or not isinstance(
            matsubara, MatsubaraQuadraturePlan
        ):
            raise TypeError(
                "Quasiclassical plan requires trajectory and Matsubara plans."
            )
        coupling = np.asarray(coupling_matrix, dtype=np.float64)
        count = trajectories.fermi_surface.channel_count
        damping_, tolerance_ = float(damping), float(tolerance)
        iterations_ = int(iterations)
        if (
            coupling.shape != (count, count)
            or np.any(~np.isfinite(coupling))
            or not np.allclose(coupling, coupling.T)
            or np.min(np.linalg.eigvalsh(coupling)) <= 0.0
            or not 0.0 < damping_ <= 1.0
            or iterations_ < 1
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
        ):
            raise ValueError("Quasiclassical coupling or nonlinear policy is invalid.")
        self.trajectories = trajectories
        self.matsubara = matsubara
        self.coupling_matrix = jnp.asarray(coupling)
        self.damping = damping_
        self.iterations = iterations_
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "equilibrium-quasiclassical-superconductivity",
                "trajectories": trajectories.plan_id,
                "matsubara": matsubara.plan_id,
                "coupling": array_tree_fingerprint(coupling),
                "damping": damping_,
                "iterations": iterations_,
                "tolerance": tolerance_,
            }
        )

    def _gap(self, amplitudes):
        trajectory = contract(
            "a,ak->k",
            amplitudes,
            self.trajectories.fermi_surface.form_factors,
            backend="jax",
        )
        return jnp.broadcast_to(
            trajectory[:, None],
            (
                self.trajectories.fermi_surface.trajectory_count,
                self.trajectories.segment_count,
            ),
        )

    def _mapping(self, amplitudes):
        gap = self._gap(amplitudes)
        propagator = self.trajectories.evaluate(self.matsubara, gap)
        segment_average = jnp.mean(propagator.anomalous_green, axis=-1)
        projected = (
            2.0
            * jnp.pi
            * self.matsubara.temperature_energy
            * contract(
                "ak,nk,k->a",
                jnp.conj(self.trajectories.fermi_surface.form_factors),
                segment_average,
                self.trajectories.fermi_surface.weights,
                backend="jax",
            )
        )
        return self.coupling_matrix @ projected, propagator

    def solve(
        self, initial_channel_amplitudes: ArrayLike, /
    ) -> QuasiclassicalEquilibriumResult:
        amplitudes = jnp.asarray(initial_channel_amplitudes)
        count = self.trajectories.fermi_surface.channel_count
        if amplitudes.shape != (count,) or not jnp.iscomplexobj(amplitudes):
            raise ValueError(
                "Quasiclassical initial amplitudes must be complex channel data."
            )
        anchor_phase = jnp.angle(amplitudes[0])
        amplitudes = amplitudes * jnp.exp(-1.0j * anchor_phase)
        for _ in range(self.iterations):
            mapped, _ = self._mapping(amplitudes)
            phase = jnp.angle(mapped[0])
            mapped = mapped * jnp.exp(-1.0j * phase)
            amplitudes = (1.0 - self.damping) * amplitudes + self.damping * mapped
        mapped, propagator = self._mapping(amplitudes)
        residual = jnp.linalg.norm(mapped - amplitudes)
        gap = self._gap(amplitudes)
        current = contract(
            "k,ki,nks->si",
            self.trajectories.fermi_surface.weights,
            self.trajectories.fermi_surface.velocities,
            jnp.imag(propagator.normal_green),
            backend="jax",
        )
        inverse_coupling = jnp.linalg.solve(self.coupling_matrix, amplitudes)
        condensation = jnp.real(jnp.vdot(amplitudes, inverse_coupling))
        quasiparticle = (
            -2.0
            * self.matsubara.temperature_energy
            * jnp.sum(
                self.trajectories.fermi_surface.weights[None, :, None]
                * (jnp.real(propagator.normal_green) - 1.0)
            )
        )
        free_energy = condensation + quasiparticle
        converged = residual <= self.tolerance * jnp.maximum(
            jnp.linalg.norm(amplitudes), 1.0
        )
        successful = (
            propagator.evidence.successful & converged & jnp.isfinite(free_energy)
        )
        branch = canonical_fingerprint(
            {
                "kind": "quasiclassical-equilibrium-branch",
                "plan": self.plan_id,
                "channel_count": count,
            }
        )
        evidence = QuasiclassicalEquilibriumEvidence(
            residual,
            propagator.evidence.normalization_residual,
            jnp.isfinite(free_energy),
            converged,
            successful,
            self.plan_id,
        )
        return QuasiclassicalEquilibriumResult(
            amplitudes,
            gap,
            propagator,
            current,
            free_energy,
            evidence,
            branch,
            self.plan_id,
        )


class RetardedSpectroscopyResult(StrictModule):
    energies: Array
    density_of_states: Array
    broadening: Array
    finite: Array
    causal: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class RetardedSpectroscopyPlan(StrictModule, NonTrainableState):
    equilibrium_plan_id: str = eqx.field(static=True)
    energies: Array
    broadening: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        equilibrium: QuasiclassicalSuperconductivityPlan,
        energies: ArrayLike,
        /,
        *,
        broadening: float,
    ):
        if not isinstance(equilibrium, QuasiclassicalSuperconductivityPlan):
            raise TypeError("equilibrium must be QuasiclassicalSuperconductivityPlan.")
        energy = np.asarray(energies, dtype=np.float64)
        broadening_ = float(broadening)
        if (
            energy.ndim != 1
            or energy.size < 1
            or np.any(~np.isfinite(energy))
            or not isfinite(broadening_)
            or broadening_ <= 0.0
        ):
            raise ValueError("Retarded energies or broadening are invalid.")
        self.equilibrium_plan_id = equilibrium.plan_id
        self.energies = jnp.asarray(energy)
        self.broadening = broadening_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quasiclassical-retarded-spectroscopy",
                "equilibrium": equilibrium.plan_id,
                "energies": array_tree_fingerprint(energy),
                "broadening": broadening_,
            }
        )

    def evaluate(
        self, equilibrium: QuasiclassicalEquilibriumResult, /
    ) -> RetardedSpectroscopyResult:
        if equilibrium.plan_id != self.equilibrium_plan_id or not bool(
            equilibrium.evidence.successful
        ):
            raise ValueError(
                "Retarded spectroscopy requires its converged equilibrium branch."
            )
        gap = jnp.mean(jnp.abs(equilibrium.trajectory_gap))
        z = self.energies + 1.0j * self.broadening
        root = jnp.sqrt(z**2 - gap**2)
        root = jnp.where(jnp.imag(root) < 0.0, -root, root)
        green = z / root
        density = jnp.maximum(jnp.real(green), 0.0)
        finite = jnp.all(jnp.isfinite(density))
        causal = jnp.all(jnp.imag(root) >= 0.0)
        successful = finite & causal
        return RetardedSpectroscopyResult(
            self.energies,
            density,
            jnp.asarray(self.broadening, dtype=density.dtype),
            finite,
            causal,
            successful,
            self.plan_id,
        )


__all__ = [
    "FermiSurfacePlan",
    "MatsubaraQuadraturePlan",
    "QuasiclassicalEquilibriumEvidence",
    "QuasiclassicalEquilibriumResult",
    "QuasiclassicalSuperconductivityPlan",
    "RetardedSpectroscopyPlan",
    "RetardedSpectroscopyResult",
    "RiccatiTrajectoryEvidence",
    "RiccatiTrajectoryPlan",
    "RiccatiTrajectoryResult",
]
