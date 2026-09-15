#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Supercell force constants, nonanalytic phonons, QHA, and RTA transport."""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...ein import contract
from ...linalg import DenseLinearOperator, OperatorProperties
from ...linalg.eigen import DenseEigh, Eigenproblem, eigensolve, EigenSolvePolicy


ForceEvaluator = Callable[[Array], Array]


class SupercellForceConstantResult(StrictModule, NonTrainableState):
    raw_force_constants: Array
    force_constants: Array
    symmetry_residual: Array
    acoustic_sum_rule_residual: Array
    displacement: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self, raw, corrected, symmetry, acoustic, displacement, successful, plan_id, /
    ):
        raw_ = jnp.asarray(raw)
        corrected_ = jnp.asarray(corrected, dtype=raw_.dtype)
        if (
            raw_.ndim != 4
            or raw_.shape != corrected_.shape
            or raw_.shape[1] != 3
            or raw_.shape[3] != 3
            or raw_.shape[0] != raw_.shape[2]
        ):
            raise ValueError("Supercell force constants must have shape (atom,3,atom,3).")
        self.raw_force_constants = raw_
        self.force_constants = corrected_
        self.symmetry_residual = jnp.asarray(symmetry, dtype=raw_.dtype).reshape(())
        self.acoustic_sum_rule_residual = jnp.asarray(acoustic, dtype=raw_.dtype).reshape(
            ()
        )
        self.displacement = jnp.asarray(displacement, dtype=raw_.dtype).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "supercell-force-constant-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "raw": np.asarray(raw_),
                        "corrected": np.asarray(corrected_),
                        "symmetry": np.asarray(self.symmetry_residual),
                        "acoustic": np.asarray(self.acoustic_sum_rule_residual),
                    }
                ),
            }
        )


class SupercellForceConstantPlan(StrictModule, NonTrainableState):
    force_evaluator: ForceEvaluator = eqx.field(static=True)
    equilibrium_positions: Array
    displacement: float = eqx.field(static=True)
    enforce_acoustic_sum_rule: bool = eqx.field(static=True)
    symmetry_tolerance: float = eqx.field(static=True)
    acoustic_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        force_evaluator: ForceEvaluator,
        equilibrium_positions: ArrayLike,
        /,
        *,
        displacement: float = 1.0e-3,
        enforce_acoustic_sum_rule: bool = True,
        symmetry_tolerance: float = 1.0e-6,
        acoustic_tolerance: float = 1.0e-8,
    ):
        if not callable(force_evaluator):
            raise TypeError("force_evaluator must be callable.")
        equilibrium = jnp.asarray(equilibrium_positions)
        step = float(displacement)
        symmetry = float(symmetry_tolerance)
        acoustic = float(acoustic_tolerance)
        if (
            equilibrium.ndim != 2
            or equilibrium.shape[1] != 3
            or any(
                not isfinite(value) or value <= 0.0
                for value in (step, symmetry, acoustic)
            )
        ):
            raise ValueError(
                "Supercell equilibrium, displacement, or tolerances are invalid."
            )
        self.force_evaluator = force_evaluator
        self.equilibrium_positions = equilibrium
        self.displacement = step
        self.enforce_acoustic_sum_rule = bool(enforce_acoustic_sum_rule)
        self.symmetry_tolerance = symmetry
        self.acoustic_tolerance = acoustic
        self.plan_id = canonical_fingerprint(
            {
                "kind": "supercell-force-constant-plan",
                "displacement": step,
                "enforce_acoustic_sum_rule": self.enforce_acoustic_sum_rule,
                "symmetry_tolerance": symmetry,
                "acoustic_tolerance": acoustic,
                "equilibrium": array_tree_fingerprint(np.asarray(equilibrium)),
            }
        )

    def evaluate(self, /) -> SupercellForceConstantResult:
        equilibrium = np.asarray(self.equilibrium_positions)
        dimension = equilibrium.size
        columns = []
        finite = True
        for coordinate in range(dimension):
            shift = np.zeros((dimension,), dtype=equilibrium.dtype)
            shift[coordinate] = self.displacement
            shift = shift.reshape(equilibrium.shape)
            plus = np.asarray(self.force_evaluator(equilibrium + shift))
            minus = np.asarray(self.force_evaluator(equilibrium - shift))
            if plus.shape != equilibrium.shape or minus.shape != equilibrium.shape:
                raise ValueError("Force evaluator changed supercell shape.")
            finite = finite and np.all(np.isfinite(plus)) and np.all(np.isfinite(minus))
            columns.append(-(plus - minus).reshape((-1,)) / (2.0 * self.displacement))
        raw_matrix = np.stack(columns, axis=1)
        symmetric = 0.5 * (raw_matrix + raw_matrix.T)
        symmetry_residual = float(np.max(np.abs(raw_matrix - raw_matrix.T), initial=0.0))
        if self.enforce_acoustic_sum_rule:
            atom_count = equilibrium.shape[0]
            translations = np.zeros((dimension, 3), dtype=equilibrium.dtype)
            for axis in range(3):
                translations[axis::3, axis] = 1.0 / np.sqrt(atom_count)
            projector = np.eye(dimension) - translations @ translations.T
            corrected = projector @ symmetric @ projector
        else:
            corrected = symmetric
        reshaped = corrected.reshape((equilibrium.shape[0], 3, equilibrium.shape[0], 3))
        acoustic_residual = float(np.max(np.abs(np.sum(reshaped, axis=2)), initial=0.0))
        successful = (
            finite
            and symmetry_residual <= self.symmetry_tolerance
            and acoustic_residual <= self.acoustic_tolerance
        )
        return SupercellForceConstantResult(
            raw_matrix.reshape(reshaped.shape),
            reshaped,
            symmetry_residual,
            acoustic_residual,
            self.displacement,
            successful,
            self.plan_id,
        )


class PhononDispersionResult(StrictModule, NonTrainableState):
    qpoints: Array
    frequencies: Array
    eigenvectors: Array
    dynamical_matrices: Array
    imaginary_mask: Array
    hermiticity_residual: Array
    acoustic_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        qpoints,
        frequencies,
        eigenvectors,
        dynamical,
        imaginary,
        hermiticity,
        acoustic,
        successful,
        plan_id,
        /,
    ):
        points = jnp.asarray(qpoints)
        frequency = jnp.asarray(frequencies, dtype=points.dtype)
        vectors = jnp.asarray(eigenvectors)
        matrices = jnp.asarray(dynamical)
        branches = (
            frequency.shape[1]
            if frequency.ndim == 2 and frequency.shape[0] == points.shape[0]
            else -1
        )
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or vectors.shape != (points.shape[0], branches, branches)
            or matrices.shape != vectors.shape
        ):
            raise ValueError(
                "Phonon q points, frequencies, eigenvectors, and matrices do not align."
            )
        self.qpoints = points
        self.frequencies = frequency
        self.eigenvectors = vectors
        self.dynamical_matrices = matrices
        self.imaginary_mask = jnp.asarray(imaginary, dtype=bool)
        self.hermiticity_residual = jnp.asarray(hermiticity, dtype=points.dtype).reshape(
            ()
        )
        self.acoustic_residual = jnp.asarray(acoustic, dtype=points.dtype).reshape(())
        self.successful = jnp.asarray(successful, dtype=bool).reshape(())
        self.plan_id = str(plan_id)
        self.result_id = canonical_fingerprint(
            {
                "kind": "phonon-dispersion-result",
                "plan": self.plan_id,
                "successful": bool(self.successful),
                "arrays": array_tree_fingerprint(
                    {
                        "qpoints": np.asarray(points),
                        "frequencies": np.asarray(frequency),
                        "eigenvectors": np.asarray(vectors),
                        "dynamical": np.asarray(matrices),
                    }
                ),
            }
        )


class PeriodicPhononPlan(StrictModule, NonTrainableState):
    translations: Array
    force_constant_blocks: Array
    masses: Array
    cell_volume: float = eqx.field(static=True)
    born_effective_charges: Array | None
    dielectric_tensor: Array | None
    nonanalytic_direction: Array | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        translations: ArrayLike,
        force_constant_blocks: ArrayLike,
        masses: ArrayLike,
        cell_volume: float,
        /,
        *,
        born_effective_charges: ArrayLike | None = None,
        dielectric_tensor: ArrayLike | None = None,
        nonanalytic_direction: ArrayLike | None = None,
    ):
        translations_ = jnp.asarray(translations)
        force = jnp.asarray(force_constant_blocks)
        masses_ = jnp.asarray(masses, dtype=force.real.dtype)
        volume = float(cell_volume)
        atoms = int(masses_.size)
        born = (
            None
            if born_effective_charges is None
            else jnp.asarray(born_effective_charges, dtype=force.real.dtype)
        )
        dielectric = (
            None
            if dielectric_tensor is None
            else jnp.asarray(dielectric_tensor, dtype=force.real.dtype)
        )
        direction = (
            None
            if nonanalytic_direction is None
            else jnp.asarray(nonanalytic_direction, dtype=force.real.dtype)
        )
        if (
            translations_.ndim != 2
            or translations_.shape[1] != 3
            or force.shape != (translations_.shape[0], atoms, 3, atoms, 3)
            or bool(jnp.any(masses_ <= 0.0))
            or not isfinite(volume)
            or volume <= 0.0
        ):
            raise ValueError(
                "Phonon translations, force constants, masses, or volume are invalid."
            )
        if not ((born is None) == (dielectric is None) == (direction is None)):
            raise ValueError(
                "LO--TO correction requires Born charges, dielectric, and direction together."
            )
        if born is not None:
            if dielectric is None or direction is None:
                raise RuntimeError("LO--TO optional tensor presence invariant failed.")
            if (
                born.shape != (atoms, 3, 3)
                or dielectric.shape != (3, 3)
                or direction.shape != (3,)
                or bool(jnp.linalg.norm(direction) == 0.0)
            ):
                raise ValueError("LO--TO tensors or direction have invalid shapes.")
        self.translations = translations_
        self.force_constant_blocks = force
        self.masses = masses_
        self.cell_volume = volume
        self.born_effective_charges = born
        self.dielectric_tensor = dielectric
        self.nonanalytic_direction = direction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-phonon-plan",
                "cell_volume": volume,
                "arrays": array_tree_fingerprint(
                    {
                        "translations": np.asarray(translations_),
                        "force_constants": np.asarray(force),
                        "masses": np.asarray(masses_),
                        "born": None if born is None else np.asarray(born),
                        "dielectric": None
                        if dielectric is None
                        else np.asarray(dielectric),
                        "direction": None if direction is None else np.asarray(direction),
                    }
                ),
            }
        )

    def evaluate(self, qpoints: ArrayLike, /) -> PhononDispersionResult:
        points = jnp.asarray(qpoints, dtype=self.force_constant_blocks.real.dtype)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Phonon qpoints must have shape (Q,3).")
        root_mass = jnp.repeat(jnp.sqrt(self.masses), 3)
        matrices = []
        frequencies = []
        eigenvectors = []
        maximum_hermiticity = jnp.asarray(0.0, dtype=points.dtype)
        for point in points:
            phase = jnp.exp(2.0j * jnp.pi * (self.translations @ point))
            matrix = contract("r,riajb->iajb", phase, self.force_constant_blocks).reshape(
                (root_mass.size, root_mass.size)
            )
            matrix = matrix / root_mass[:, None] / root_mass[None, :]
            if self.born_effective_charges is not None and bool(
                jnp.linalg.norm(point) <= 1.0e-10
            ):
                direction = self.nonanalytic_direction / jnp.linalg.norm(
                    self.nonanalytic_direction
                )
                charge_projection = contract(
                    "iab,b->ia", self.born_effective_charges, direction
                ).reshape((-1,))
                denominator = direction @ self.dielectric_tensor @ direction
                matrix = (
                    matrix
                    + (4.0 * jnp.pi / self.cell_volume)
                    * jnp.outer(
                        charge_projection / root_mass, charge_projection / root_mass
                    )
                    / denominator
                )
            hermiticity = jnp.max(jnp.abs(matrix - jnp.conj(matrix.T)), initial=0.0)
            maximum_hermiticity = jnp.maximum(maximum_hermiticity, hermiticity)
            matrix = 0.5 * (matrix + jnp.conj(matrix.T))
            solved = eigensolve(
                Eigenproblem(
                    DenseLinearOperator(
                        matrix,
                        properties=OperatorProperties(
                            self_adjoint=True, evidence={"self_adjoint": "construction"}
                        ),
                    )
                ),
                policy=EigenSolvePolicy(
                    DenseEigh(), count=int(matrix.shape[0]), which="smallest-algebraic"
                ),
            )
            values = solved.eigenvalues.real
            frequency = jnp.sign(values) * jnp.sqrt(jnp.abs(values))
            matrices.append(matrix)
            frequencies.append(frequency)
            eigenvectors.append(solved.eigenvectors)
        frequency_values = jnp.stack(tuple(frequencies))
        gamma_indices = np.flatnonzero(
            np.linalg.norm(np.asarray(points), axis=1) <= 1.0e-10
        )
        acoustic = (
            jnp.max(jnp.abs(frequency_values[int(gamma_indices[0]), :3]), initial=0.0)
            if gamma_indices.size and self.born_effective_charges is None
            else jnp.asarray(0.0, dtype=points.dtype)
        )
        successful = (maximum_hermiticity <= 1.0e-8) & jnp.all(
            jnp.isfinite(frequency_values)
        )
        return PhononDispersionResult(
            points,
            frequency_values,
            jnp.stack(tuple(eigenvectors)),
            jnp.stack(tuple(matrices)),
            frequency_values < 0.0,
            maximum_hermiticity,
            acoustic,
            successful,
            self.plan_id,
        )


class LatticeThermodynamicsResult(StrictModule, NonTrainableState):
    free_energy: Array
    internal_energy: Array
    entropy: Array
    heat_capacity: Array
    zero_point_energy: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def lattice_thermodynamics(
    frequencies: ArrayLike,
    qpoint_weights: ArrayLike,
    temperature: float,
    boltzmann_constant: float,
    hbar: float,
    /,
) -> LatticeThermodynamicsResult:
    frequency = jnp.asarray(frequencies)
    weights = jnp.asarray(qpoint_weights, dtype=frequency.dtype)
    temperature_ = float(temperature)
    boltzmann = float(boltzmann_constant)
    hbar_ = float(hbar)
    if (
        frequency.ndim != 2
        or weights.shape != (frequency.shape[0],)
        or bool(jnp.any(frequency <= 0.0))
        or bool(jnp.any(weights < 0.0))
        or bool(jnp.any(~jnp.isfinite(frequency)))
        or bool(jnp.any(~jnp.isfinite(weights)))
        or not np.isclose(float(jnp.sum(weights)), 1.0, atol=1.0e-12)
        or any(
            not isfinite(value) or value <= 0.0
            for value in (temperature_, boltzmann, hbar_)
        )
    ):
        raise ValueError(
            "Lattice frequencies, weights, temperature, or constants are invalid."
        )
    energy = hbar_ * frequency
    thermal = boltzmann * temperature_
    x = energy / thermal
    occupation = 1.0 / jnp.expm1(x)
    zero = 0.5 * jnp.sum(weights[:, None] * energy)
    internal = jnp.sum(weights[:, None] * energy * (0.5 + occupation))
    free = zero + thermal * jnp.sum(weights[:, None] * jnp.log1p(-jnp.exp(-x)))
    entropy = (internal - free) / temperature_
    decay = jnp.exp(-x)
    heat = boltzmann * jnp.sum(weights[:, None] * x**2 * decay / (1.0 - decay) ** 2)
    successful = jnp.all(jnp.isfinite(jnp.asarray((free, internal, entropy, heat, zero))))
    return LatticeThermodynamicsResult(
        free,
        internal,
        entropy,
        heat,
        zero,
        successful,
        canonical_fingerprint(
            {
                "kind": "lattice-thermodynamics-result",
                "temperature": temperature_,
                "boltzmann_constant": boltzmann,
                "hbar": hbar_,
                "arrays": array_tree_fingerprint(
                    {
                        "frequencies": np.asarray(frequency),
                        "weights": np.asarray(weights),
                        "free_energy": np.asarray(free),
                        "internal_energy": np.asarray(internal),
                        "entropy": np.asarray(entropy),
                        "heat_capacity": np.asarray(heat),
                        "zero_point_energy": np.asarray(zero),
                    }
                ),
            }
        ),
    )


class QuasiHarmonicResult(StrictModule, NonTrainableState):
    temperatures: Array
    equilibrium_volumes: Array
    free_energies: Array
    thermal_expansion: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def quasi_harmonic_thermodynamics(
    volumes: ArrayLike,
    static_energies: ArrayLike,
    frequencies: ArrayLike,
    qpoint_weights: ArrayLike,
    temperatures: ArrayLike,
    boltzmann_constant: float,
    hbar: float,
    /,
) -> QuasiHarmonicResult:
    volume = np.asarray(volumes, dtype=float)
    static = np.asarray(static_energies, dtype=float)
    frequency = np.asarray(frequencies, dtype=float)
    temperatures_ = np.asarray(temperatures, dtype=float)
    if (
        volume.ndim != 1
        or static.shape != volume.shape
        or frequency.ndim != 3
        or frequency.shape[0] != volume.size
        or temperatures_.ndim != 1
        or volume.size < 2
        or temperatures_.size < 2
        or np.any(~np.isfinite(volume))
        or np.any(~np.isfinite(static))
        or np.any(~np.isfinite(frequency))
        or np.any(frequency <= 0.0)
        or np.any(~np.isfinite(temperatures_))
        or np.any(temperatures_ <= 0.0)
        or np.any(np.diff(volume) <= 0.0)
    ):
        raise ValueError(
            "QHA volumes, energies, frequencies, or temperatures are invalid."
        )
    free_rows = []
    equilibrium = []
    for temperature in temperatures_:
        totals = []
        for index in range(volume.size):
            vibrational = lattice_thermodynamics(
                frequency[index],
                qpoint_weights,
                float(temperature),
                boltzmann_constant,
                hbar,
            )
            totals.append(static[index] + float(vibrational.free_energy))
        totals = np.asarray(totals)
        minimum = int(np.argmin(totals))
        if 0 < minimum < volume.size - 1:
            coefficients = np.polyfit(
                volume[minimum - 1 : minimum + 2], totals[minimum - 1 : minimum + 2], 2
            )
            candidate = -coefficients[1] / (2.0 * coefficients[0])
            equilibrium.append(
                float(np.clip(candidate, volume[minimum - 1], volume[minimum + 1]))
            )
        else:
            equilibrium.append(float(volume[minimum]))
        free_rows.append(totals)
    equilibrium = np.asarray(equilibrium)
    expansion = np.gradient(equilibrium, temperatures_, edge_order=1) / equilibrium
    successful = np.all(np.isfinite(free_rows)) and np.all(np.isfinite(expansion))
    return QuasiHarmonicResult(
        jnp.asarray(temperatures_),
        jnp.asarray(equilibrium),
        jnp.asarray(free_rows),
        jnp.asarray(expansion),
        jnp.asarray(successful),
        canonical_fingerprint(
            {
                "kind": "quasi-harmonic-result",
                "boltzmann_constant": float(boltzmann_constant),
                "hbar": float(hbar),
                "arrays": array_tree_fingerprint(
                    {
                        "volumes": volume,
                        "static_energies": static,
                        "frequencies": frequency,
                        "qpoint_weights": np.asarray(qpoint_weights),
                        "temperatures": temperatures_,
                        "equilibrium_volumes": equilibrium,
                        "free_energies": np.asarray(free_rows),
                        "thermal_expansion": expansion,
                    }
                ),
            }
        ),
    )


class AnharmonicTransportResult(StrictModule, NonTrainableState):
    scattering_rates: Array
    lifetimes: Array
    thermal_conductivity: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def anharmonic_rta_transport(
    frequencies: ArrayLike,
    group_velocities: ArrayLike,
    heat_capacities: ArrayLike,
    cubic_vertices: ArrayLike,
    volume: float,
    broadening: float,
    /,
) -> AnharmonicTransportResult:
    frequency = jnp.asarray(frequencies)
    velocity = jnp.asarray(group_velocities, dtype=frequency.dtype)
    heat = jnp.asarray(heat_capacities, dtype=frequency.dtype)
    vertices = jnp.asarray(cubic_vertices)
    modes = int(frequency.size)
    volume_ = float(volume)
    width = float(broadening)
    if (
        frequency.shape != (modes,)
        or velocity.shape != (modes, 3)
        or heat.shape != (modes,)
        or vertices.shape != (modes, modes, modes)
        or bool(jnp.any(frequency <= 0.0))
        or bool(jnp.any(heat < 0.0))
        or bool(jnp.any(~jnp.isfinite(frequency)))
        or bool(jnp.any(~jnp.isfinite(velocity)))
        or bool(jnp.any(~jnp.isfinite(heat)))
        or bool(jnp.any(~jnp.isfinite(vertices)))
        or not isfinite(volume_)
        or volume_ <= 0.0
        or not isfinite(width)
        or width <= 0.0
    ):
        raise ValueError(
            "Anharmonic transport mode arrays, volume, or broadening are invalid."
        )
    difference = (
        frequency[:, None, None] - frequency[None, :, None] - frequency[None, None, :]
    )
    delta = jnp.exp(-0.5 * (difference / width) ** 2) / (width * jnp.sqrt(2.0 * jnp.pi))
    rates = 2.0 * jnp.pi * jnp.sum(jnp.abs(vertices) ** 2 * delta, axis=(1, 2))
    lifetimes = 1.0 / jnp.maximum(2.0 * rates, jnp.finfo(frequency.dtype).tiny)
    conductivity = (
        contract("m,ma,mb,m->ab", heat, velocity, velocity, lifetimes) / volume_
    )
    successful = jnp.all(jnp.isfinite(rates)) & jnp.all(jnp.isfinite(conductivity))
    return AnharmonicTransportResult(
        rates,
        lifetimes,
        conductivity,
        successful,
        canonical_fingerprint(
            {
                "kind": "anharmonic-rta-transport-result",
                "volume": volume_,
                "broadening": width,
                "arrays": array_tree_fingerprint(
                    {
                        "frequencies": np.asarray(frequency),
                        "velocities": np.asarray(velocity),
                        "heat_capacities": np.asarray(heat),
                        "cubic_vertices": np.asarray(vertices),
                        "scattering_rates": np.asarray(rates),
                        "lifetimes": np.asarray(lifetimes),
                        "thermal_conductivity": np.asarray(conductivity),
                    }
                ),
            }
        ),
    )


__all__ = [
    "AnharmonicTransportResult",
    "LatticeThermodynamicsResult",
    "PeriodicPhononPlan",
    "PhononDispersionResult",
    "QuasiHarmonicResult",
    "SupercellForceConstantPlan",
    "SupercellForceConstantResult",
    "anharmonic_rta_transport",
    "lattice_thermodynamics",
    "quasi_harmonic_thermodynamics",
]
