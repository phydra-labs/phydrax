#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Projected fractional quantum Hall systems on the Haldane sphere."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from fractions import Fraction
from math import isfinite, pi, sqrt
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._limit_study import (
    run_scientific_limit_study,
    ScientificLimitAxis,
    ScientificLimitDatum,
    ScientificLimitStudyPlan,
    ScientificLimitStudyResult,
    ScientificLimitVariation,
)
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...applications.fuzzy_space import (
    FuzzySphereManyBodyPlan,
    prepare_fuzzy_sphere_many_body,
    PreparedFuzzySphereManyBody,
)
from ...linalg.eigen import (
    Eigenproblem,
    eigensolve,
    EigenSolvePolicy,
    RestartedLanczos,
)
from ...tensor_network import su2_wigner_3j, su2_wigner_6j
from ...units import ENERGY, UnitDefinition
from ._identity import MonopoleLandauLevel


_HBAR_SI = 1.054_571_817e-34
_ELEMENTARY_CHARGE_SI = 1.602_176_634e-19
_EPSILON_ZERO_SI = 8.854_187_8128e-12
_ELECTRON_MASS_SI = 9.109_383_7139e-31

HallStatistics: TypeAlias = Literal["fermion", "boson"]
QuantumHallGapKind: TypeAlias = Literal["neutral", "charge", "transport"]


class QuantumHallEnergyScale(StrictModule, NonTrainableState):
    """One declared energy unit with an exact SI conversion for a Hall campaign."""

    unit: UnitDefinition
    joules_per_unit: float = eqx.field(static=True)
    label: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        unit: UnitDefinition,
        joules_per_unit: float,
        label: str,
        /,
    ):
        value = float(joules_per_unit)
        label_ = str(label).strip()
        if not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition.")
        if unit.dimension != ENERGY or not isfinite(value) or value <= 0.0 or not label_:
            raise ValueError("Quantum Hall energy scale is invalid.")
        self.unit = unit
        self.joules_per_unit = value
        self.label = label_
        self.scale_id = canonical_fingerprint(
            {
                "kind": "quantum-hall-energy-scale",
                "unit": unit.unit_id,
                "joules_per_unit": value,
                "label": label_,
            }
        )


class QuantumHallMaterialPlan(StrictModule, NonTrainableState):
    """SI material inputs and derived magnetic/Coulomb scales."""

    magnetic_field_tesla: float = eqx.field(static=True)
    carrier_density_per_square_meter: float = eqx.field(static=True)
    effective_mass_in_electron_masses: float = eqx.field(static=True)
    relative_permittivity: float = eqx.field(static=True)
    charge_magnitude_in_elementary_charges: float = eqx.field(static=True)
    magnetic_length_meter: float = eqx.field(static=True)
    cyclotron_energy_joule: float = eqx.field(static=True)
    coulomb_energy_joule: float = eqx.field(static=True)
    landau_level_mixing: float = eqx.field(static=True)
    filling_factor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        magnetic_field_tesla: float,
        carrier_density_per_square_meter: float,
        effective_mass_in_electron_masses: float,
        relative_permittivity: float,
        /,
        *,
        charge_magnitude_in_elementary_charges: float = 1.0,
    ):
        field = float(magnetic_field_tesla)
        density = float(carrier_density_per_square_meter)
        mass_ratio = float(effective_mass_in_electron_masses)
        permittivity = float(relative_permittivity)
        charge_ratio = float(charge_magnitude_in_elementary_charges)
        if any(
            not isfinite(value) or value <= 0.0
            for value in (field, density, mass_ratio, permittivity, charge_ratio)
        ):
            raise ValueError(
                "Quantum Hall material inputs must be positive finite SI values."
            )
        charge = charge_ratio * _ELEMENTARY_CHARGE_SI
        mass = mass_ratio * _ELECTRON_MASS_SI
        magnetic_length = sqrt(_HBAR_SI / (charge * field))
        cyclotron = _HBAR_SI * charge * field / mass
        coulomb = charge**2 / (
            4.0 * pi * _EPSILON_ZERO_SI * permittivity * magnetic_length
        )
        filling = density * 2.0 * pi * magnetic_length**2
        self.magnetic_field_tesla = field
        self.carrier_density_per_square_meter = density
        self.effective_mass_in_electron_masses = mass_ratio
        self.relative_permittivity = permittivity
        self.charge_magnitude_in_elementary_charges = charge_ratio
        self.magnetic_length_meter = magnetic_length
        self.cyclotron_energy_joule = cyclotron
        self.coulomb_energy_joule = coulomb
        self.landau_level_mixing = coulomb / cyclotron
        self.filling_factor = filling
        self.plan_id = canonical_fingerprint(
            {
                "kind": "quantum-hall-material-plan",
                "magnetic_field_tesla": field,
                "carrier_density_per_square_meter": density,
                "effective_mass_in_electron_masses": mass_ratio,
                "relative_permittivity": permittivity,
                "charge_magnitude_in_elementary_charges": charge_ratio,
            }
        )


class HaldaneSpherePlan(StrictModule, NonTrainableState):
    particle_count: int = eqx.field(static=True)
    manifold: MonopoleLandauLevel
    statistics: HallStatistics = eqx.field(static=True)
    filling: tuple[int, int] | None = eqx.field(static=True)
    shift: int | None = eqx.field(static=True)
    flux_offset: int = eqx.field(static=True)
    energy_scale: QuantumHallEnergyScale
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_count: int,
        manifold: MonopoleLandauLevel,
        statistics: HallStatistics,
        energy_scale: QuantumHallEnergyScale,
        /,
        *,
        filling: Fraction | None = None,
        shift: int | None = None,
        flux_offset: int = 0,
    ):
        particles = int(particle_count)
        statistics_ = str(statistics)
        offset = int(flux_offset)
        if not isinstance(manifold, MonopoleLandauLevel):
            raise TypeError("manifold must be MonopoleLandauLevel.")
        if not isinstance(energy_scale, QuantumHallEnergyScale):
            raise TypeError("energy_scale must be QuantumHallEnergyScale.")
        if particles < 2 or statistics_ not in ("fermion", "boson"):
            raise ValueError("Haldane sphere particles or statistics are invalid.")
        if statistics_ == "fermion" and particles > manifold.orbital_count:
            raise ValueError("Fermion particle count exceeds the sphere orbital count.")
        if (filling is None) != (shift is None):
            raise ValueError("filling and shift must be supplied together.")
        filling_record = None
        if filling is not None:
            if not isinstance(filling, Fraction) or filling <= 0:
                raise TypeError("filling must be a positive fractions.Fraction.")
            expected = Fraction(particles, 1) / filling - int(shift) + offset
            if (
                expected.denominator != 1
                or expected.numerator != manifold.twice_monopole_strength
            ):
                raise ValueError(
                    "Physical flux does not satisfy the declared filling, shift, and offset."
                )
            filling_record = (filling.numerator, filling.denominator)
        self.particle_count = particles
        self.manifold = manifold
        self.statistics = statistics_  # type: ignore[assignment]
        self.filling = filling_record
        self.shift = None if shift is None else int(shift)
        self.flux_offset = offset
        self.energy_scale = energy_scale
        self.plan_id = canonical_fingerprint(
            {
                "kind": "haldane-sphere-plan",
                "particle_count": particles,
                "manifold": manifold.manifold_id,
                "statistics": statistics_,
                "filling": filling_record,
                "shift": self.shift,
                "flux_offset": offset,
                "energy_scale": energy_scale.scale_id,
            }
        )

    @property
    def twice_monopole_flux(self) -> int:
        return self.manifold.twice_monopole_strength

    @property
    def landau_level(self) -> int:
        return self.manifold.landau_level

    @property
    def twice_orbital_spin(self) -> int:
        return self.manifold.twice_orbital_spin

    @property
    def orbital_count(self) -> int:
        return self.manifold.orbital_count


class HaldanePseudopotentialPlan(StrictModule, NonTrainableState):
    sphere: HaldaneSpherePlan
    relative_channels: tuple[tuple[int, float], ...] = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sphere: HaldaneSpherePlan,
        relative_channels: Mapping[int, float],
        source_id: str,
        /,
    ):
        if not isinstance(sphere, HaldaneSpherePlan):
            raise TypeError("sphere must be HaldaneSpherePlan.")
        channels = tuple(
            sorted(
                (int(index), float(value)) for index, value in relative_channels.items()
            )
        )
        source = str(source_id).strip()
        allowed = tuple(
            value
            for value in range(sphere.twice_orbital_spin + 1)
            if value % 2 == (1 if sphere.statistics == "fermion" else 0)
        )
        if (
            tuple(index for index, _ in channels) != allowed
            or any(not isfinite(value) for _, value in channels)
            or not source
        ):
            raise ValueError(
                "Pseudopotentials must cover every statistics-compatible relative channel."
            )
        self.sphere = sphere
        self.relative_channels = channels
        self.source_id = source
        self.plan_id = canonical_fingerprint(
            {
                "kind": "haldane-pseudopotential-plan",
                "sphere": sphere.plan_id,
                "channels": channels,
                "source": source,
            }
        )

    def pair_spin_channels(self, /) -> dict[int, float]:
        twice_spin = self.sphere.twice_orbital_spin
        return {
            2 * (twice_spin - relative): value
            for relative, value in self.relative_channels
        }


def coulomb_haldane_pseudopotentials(
    sphere: HaldaneSpherePlan,
    /,
) -> HaldanePseudopotentialPlan:
    """Exact chord-distance Coulomb pseudopotentials for one monopole LL."""

    if not isinstance(sphere, HaldaneSpherePlan):
        raise TypeError("sphere must be HaldaneSpherePlan.")
    flux = sphere.twice_monopole_flux
    twice_spin = sphere.twice_orbital_spin
    radius = sqrt(flux / 2.0)
    channels: dict[int, float] = {}
    for relative in range(twice_spin + 1):
        if relative % 2 != (1 if sphere.statistics == "fermion" else 0):
            continue
        total = twice_spin - relative
        twice_total = 2 * total
        terms = []
        for multipole in range(twice_spin + 1):
            three_j = su2_wigner_3j(
                twice_spin,
                2 * multipole,
                twice_spin,
                -flux,
                0,
                flux,
            )
            if three_j == 0.0:
                continue
            terms.append(
                su2_wigner_6j(
                    twice_total,
                    twice_spin,
                    twice_spin,
                    2 * multipole,
                    twice_spin,
                    twice_spin,
                )
                * three_j**2
            )
        value = ((-1.0) ** (flux + total)) * (twice_spin + 1) ** 2 * sum(terms) / radius
        channels[relative] = float(value)
    return HaldanePseudopotentialPlan(
        sphere,
        channels,
        f"exact-chord-coulomb:n={sphere.landau_level}",
    )


class PreparedHaldaneSphereHamiltonian(StrictModule, NonTrainableState):
    sphere: HaldaneSpherePlan
    pseudopotentials: HaldanePseudopotentialPlan
    twice_projection: int = eqx.field(static=True)
    many_body: PreparedFuzzySphereManyBody
    prepared_id: str = eqx.field(static=True)


def prepare_haldane_sphere_hamiltonian(
    pseudopotentials: HaldanePseudopotentialPlan,
    /,
    *,
    twice_projection: int = 0,
    maximum_basis_dimension: int = 100_000,
    maximum_nonzero_routes: int = 10_000_000,
    maximum_table_bytes: int = 64 * 1024 * 1024,
) -> PreparedHaldaneSphereHamiltonian:
    if not isinstance(pseudopotentials, HaldanePseudopotentialPlan):
        raise TypeError("pseudopotentials must be HaldanePseudopotentialPlan.")
    sphere = pseudopotentials.sphere
    projection = int(twice_projection)
    prepared = prepare_fuzzy_sphere_many_body(
        FuzzySphereManyBodyPlan(
            sphere.twice_orbital_spin,
            sphere.particle_count,
            sphere.statistics,
            pseudopotentials.pair_spin_channels(),
            twice_projection=projection,
            maximum_basis_dimension=maximum_basis_dimension,
            maximum_nonzero_routes=maximum_nonzero_routes,
            maximum_table_bytes=maximum_table_bytes,
        )
    )
    return PreparedHaldaneSphereHamiltonian(
        sphere,
        pseudopotentials,
        projection,
        prepared,
        canonical_fingerprint(
            {
                "kind": "prepared-haldane-sphere-hamiltonian",
                "sphere": sphere.plan_id,
                "pseudopotentials": pseudopotentials.plan_id,
                "twice_projection": projection,
                "many_body": prepared.prepared_id,
            }
        ),
    )


class HaldaneSphereSpectrumResult(StrictModule, NonTrainableState):
    energies: Array
    eigenvectors: Array
    residual_norms: Array
    relative_residuals: Array
    successful: Array
    prepared: PreparedHaldaneSphereHamiltonian
    result_id: str = eqx.field(static=True)


class HaldaneSphereSpectrumPlan(StrictModule, NonTrainableState):
    prepared: PreparedHaldaneSphereHamiltonian
    eigenpair_count: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    random_seed: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prepared: PreparedHaldaneSphereHamiltonian,
        eigenpair_count: int,
        /,
        *,
        maximum_steps: int = 200,
        random_seed: int = 0,
    ):
        if not isinstance(prepared, PreparedHaldaneSphereHamiltonian):
            raise TypeError("prepared must be PreparedHaldaneSphereHamiltonian.")
        count = int(eigenpair_count)
        steps = int(maximum_steps)
        seed = int(random_seed)
        dimension = prepared.many_body.dimension
        if count < 1 or count >= dimension or steps < 1 or seed < 0:
            raise ValueError("Sphere eigensolve count, steps, or seed is invalid.")
        self.prepared = prepared
        self.eigenpair_count = count
        self.maximum_steps = steps
        self.random_seed = seed
        self.plan_id = canonical_fingerprint(
            {
                "kind": "haldane-sphere-spectrum-plan",
                "prepared": prepared.prepared_id,
                "eigenpair_count": count,
                "maximum_steps": steps,
                "random_seed": seed,
            }
        )

    def evaluate(self, /) -> HaldaneSphereSpectrumResult:
        dimension = self.prepared.many_body.dimension
        subspace = min(dimension, max(2 * self.eigenpair_count + 8, 12))
        restart = min(max(self.eigenpair_count + 1, 2), subspace - 1)
        solved = eigensolve(
            Eigenproblem(
                self.prepared.many_body.operator,
                problem_id=self.plan_id,
            ),
            policy=EigenSolvePolicy(
                RestartedLanczos(
                    subspace_dimension=subspace,
                    restart_dimension=restart,
                ),
                count=self.eigenpair_count,
                which="smallest-algebraic",
                max_steps=self.maximum_steps,
                key=jnp.asarray((0, self.random_seed), dtype=jnp.uint32),
            ),
        )
        result_id = canonical_fingerprint(
            {
                "kind": "haldane-sphere-spectrum-result",
                "plan": self.plan_id,
                "arrays": array_tree_fingerprint(
                    {
                        "energies": np.asarray(solved.eigenvalues),
                        "residual_norms": np.asarray(solved.residual_norms),
                    }
                ),
            }
        )
        return HaldaneSphereSpectrumResult(
            solved.eigenvalues.real,
            solved.eigenvectors,
            solved.residual_norms,
            solved.relative_residuals,
            jnp.all(solved.successful),
            self.prepared,
            result_id,
        )


class QuantumHallGapResult(StrictModule, NonTrainableState):
    kind: QuantumHallGapKind = eqx.field(static=True)
    gap: Array
    component_energies: Array
    successful: Array
    result_id: str = eqx.field(static=True)


def neutral_gap(spectrum: HaldaneSphereSpectrumResult, /) -> QuantumHallGapResult:
    if not isinstance(spectrum, HaldaneSphereSpectrumResult):
        raise TypeError("spectrum must be HaldaneSphereSpectrumResult.")
    if spectrum.energies.shape[0] < 2:
        raise ValueError("Neutral gaps require at least two eigenpairs.")
    energies = spectrum.energies[:2]
    gap = energies[1] - energies[0]
    successful = spectrum.successful & jnp.isfinite(gap) & (gap >= 0.0)
    return QuantumHallGapResult(
        "neutral",
        gap,
        energies,
        successful,
        canonical_fingerprint(
            {
                "kind": "quantum-hall-neutral-gap",
                "spectrum": spectrum.result_id,
                "energies": array_tree_fingerprint(np.asarray(energies)),
            }
        ),
    )


def charge_gap(
    lower_flux_ground_energy: float,
    center_ground_energy: float,
    upper_flux_ground_energy: float,
    /,
) -> QuantumHallGapResult:
    components = np.asarray(
        (lower_flux_ground_energy, center_ground_energy, upper_flux_ground_energy),
        dtype=np.float64,
    )
    if np.any(~np.isfinite(components)):
        raise ValueError("Charge-gap component energies must be finite.")
    gap = components[0] + components[2] - 2.0 * components[1]
    return QuantumHallGapResult(
        "charge",
        jnp.asarray(gap),
        jnp.asarray(components),
        jnp.asarray(gap >= 0.0),
        canonical_fingerprint(
            {
                "kind": "quantum-hall-charge-gap",
                "components": array_tree_fingerprint(components),
            }
        ),
    )


def run_quantum_hall_finite_size_study(
    inverse_particle_counts: Sequence[float],
    gaps: Sequence[float],
    standard_errors: Sequence[float],
    /,
) -> ScientificLimitStudyResult:
    coordinates = tuple(float(value) for value in inverse_particle_counts)
    values = tuple(float(value) for value in gaps)
    errors = tuple(float(value) for value in standard_errors)
    if (
        len(coordinates) < 4
        or len(coordinates) != len(values)
        or len(values) != len(errors)
    ):
        raise ValueError("Quantum Hall finite-size studies require four aligned points.")
    axis = ScientificLimitAxis(
        "inverse-particle-count",
        0.0,
        minimum_span=max(coordinates) - min(coordinates),
    )
    plan = ScientificLimitStudyPlan(
        (axis,),
        (
            ScientificLimitVariation(
                "linear",
                {"inverse-particle-count": 1},
                minimum_points=3,
            ),
            ScientificLimitVariation(
                "quadratic",
                {"inverse-particle-count": 2},
                minimum_points=4,
            ),
        ),
    )
    data = tuple(
        ScientificLimitDatum(
            f"size-{index}",
            {"inverse-particle-count": coordinate},
            value,
            error,
        )
        for index, (coordinate, value, error) in enumerate(
            zip(coordinates, values, errors, strict=True)
        )
    )
    return run_scientific_limit_study(plan, data)


__all__ = [
    "HaldanePseudopotentialPlan",
    "HaldaneSpherePlan",
    "HaldaneSphereSpectrumPlan",
    "HaldaneSphereSpectrumResult",
    "PreparedHaldaneSphereHamiltonian",
    "QuantumHallEnergyScale",
    "QuantumHallGapKind",
    "QuantumHallGapResult",
    "QuantumHallMaterialPlan",
    "charge_gap",
    "coulomb_haldane_pseudopotentials",
    "neutral_gap",
    "prepare_haldane_sphere_hamiltonian",
    "run_quantum_hall_finite_size_study",
]
