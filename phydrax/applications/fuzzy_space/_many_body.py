#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Many-body Haldane-sphere sectors, matrix-free pseudopotentials, and MPS lowering."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

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
from ...linalg import (
    AbstractLinearOperator,
    ArraySpace,
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ...operators.quantum import AbelianGroup, FermionModeOrder
from ...operators.quantum.lattice import (
    AbstractSectorBasis,
    FixedAbelianChargeBasis,
    FixedBosonNumberBasis,
    FixedCardinalityFermionBasis,
    SectorBasisResourcePolicy,
)
from ...tensor_network import (
    MatrixProductOperator,
    MatrixProductState,
    su2_clebsch_gordan,
    su2_fusion,
)


FuzzyManyBodyStatistics: TypeAlias = Literal["boson", "fermion"]


def _apply_annihilation(
    occupation: tuple[int, ...],
    orbital: int,
    statistics: str,
    /,
) -> tuple[tuple[int, ...], complex] | None:
    count = occupation[orbital]
    if count == 0:
        return None
    values = list(occupation)
    values[orbital] -= 1
    coefficient: complex
    if statistics == "boson":
        coefficient = math.sqrt(count)
    else:
        coefficient = -1.0 if sum(occupation[:orbital]) % 2 else 1.0
    return tuple(values), coefficient


def _apply_creation(
    occupation: tuple[int, ...],
    orbital: int,
    statistics: str,
    /,
) -> tuple[tuple[int, ...], complex] | None:
    count = occupation[orbital]
    if statistics == "fermion" and count:
        return None
    values = list(occupation)
    values[orbital] += 1
    coefficient: complex
    if statistics == "boson":
        coefficient = math.sqrt(count + 1)
    else:
        coefficient = -1.0 if sum(occupation[:orbital]) % 2 else 1.0
    return tuple(values), coefficient


def _apply_normal_ordered(
    occupation: tuple[int, ...],
    creators: Sequence[int],
    annihilators: Sequence[int],
    statistics: str,
    /,
) -> tuple[tuple[int, ...], complex] | None:
    state = occupation
    coefficient = 1.0 + 0.0j
    for orbital in annihilators:
        result = _apply_annihilation(state, int(orbital), statistics)
        if result is None:
            return None
        state, factor = result
        coefficient *= factor
    for orbital in creators:
        result = _apply_creation(state, int(orbital), statistics)
        if result is None:
            return None
        state, factor = result
        coefficient *= factor
    return state, coefficient


@dataclass(frozen=True, slots=True)
class FuzzyThreeBodyTerm:
    """One explicit normal-ordered three-body orbital interaction."""

    creators: tuple[int, int, int]
    annihilators: tuple[int, int, int]
    coefficient: complex
    term_id: str

    def __init__(
        self,
        creators: Sequence[int],
        annihilators: Sequence[int],
        coefficient: complex,
        /,
    ):
        creators_ = tuple(creators)
        annihilators_ = tuple(annihilators)
        coefficient_ = complex(coefficient)
        if len(creators_) != 3 or len(annihilators_) != 3:
            raise ValueError("Three-body terms require three creators and annihilators.")
        if not np.isfinite(coefficient_):
            raise ValueError("Three-body coefficients must be finite.")
        content = {
            "kind": "fuzzy-three-body-term",
            "creators": creators_,
            "annihilators": annihilators_,
            "coefficient": (coefficient_.real, coefficient_.imag),
        }
        object.__setattr__(self, "creators", creators_)  # type: ignore[arg-type]
        object.__setattr__(self, "annihilators", annihilators_)  # type: ignore[arg-type]
        object.__setattr__(self, "coefficient", coefficient_)
        object.__setattr__(self, "term_id", canonical_fingerprint(content))


class FuzzySphereManyBodyPlan(StrictModule):
    """Fixed-N bosonic or fermionic lowest-Landau-level sphere model."""

    twice_monopole_flux: int = eqx.field(static=True)
    particle_count: int = eqx.field(static=True)
    statistics: FuzzyManyBodyStatistics = eqx.field(static=True)
    twice_projection: int | None = eqx.field(static=True)
    pseudopotentials: tuple[tuple[int, float], ...] = eqx.field(static=True)
    three_body_terms: tuple[FuzzyThreeBodyTerm, ...] = eqx.field(static=True)
    maximum_basis_dimension: int = eqx.field(static=True)
    maximum_nonzero_routes: int = eqx.field(static=True)
    maximum_table_bytes: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        twice_monopole_flux: int,
        particle_count: int,
        statistics: FuzzyManyBodyStatistics,
        pseudopotentials: Mapping[int, float],
        /,
        *,
        twice_projection: int | None = None,
        three_body_terms: Sequence[FuzzyThreeBodyTerm] = (),
        maximum_basis_dimension: int = 100_000,
        maximum_nonzero_routes: int = 10_000_000,
        maximum_table_bytes: int = 64 * 1024 * 1024,
        tolerance: float = 1e-12,
    ):
        flux = int(twice_monopole_flux)
        particles = int(particle_count)
        projection = None if twice_projection is None else int(twice_projection)
        statistics_ = str(statistics)
        potentials = tuple(
            sorted((int(spin), float(value)) for spin, value in pseudopotentials.items())
        )
        terms = tuple(three_body_terms)
        maximum_basis = int(maximum_basis_dimension)
        maximum_routes = int(maximum_nonzero_routes)
        maximum_table = int(maximum_table_bytes)
        tolerance_ = float(tolerance)
        if flux < 1 or particles < 2 or statistics_ not in ("boson", "fermion"):
            raise ValueError(
                "Fuzzy many-body flux, particles, or statistics are invalid."
            )
        if statistics_ == "fermion" and particles > flux + 1:
            raise ValueError("Fermion particle count exceeds the orbital count.")
        if projection is not None and (
            abs(projection) > particles * flux or (projection - particles * flux) % 2 != 0
        ):
            raise ValueError("twice_projection is incompatible with the sphere sector.")
        allowed = tuple(
            total
            for total in su2_fusion(flux, flux)
            if (((2 * flux - total) // 2) % 2 == 0) == (statistics_ == "boson")
        )
        if tuple(spin for spin, _ in potentials) != allowed:
            raise ValueError(
                "Pseudopotentials must cover every statistics-allowed pair spin."
            )
        if any(not isinstance(value, FuzzyThreeBodyTerm) for value in terms):
            raise TypeError("three_body_terms must contain FuzzyThreeBodyTerm values.")
        if any(
            orbital < 0 or orbital > flux
            for term in terms
            for orbital in (*term.creators, *term.annihilators)
        ):
            raise ValueError("A three-body term references an unavailable orbital.")
        if (
            maximum_basis < 1
            or maximum_routes < 1
            or maximum_table < 1
            or tolerance_ <= 0.0
        ):
            raise ValueError("Fuzzy many-body resource controls must be positive.")
        content = {
            "kind": "fuzzy-sphere-many-body-plan",
            "twice_monopole_flux": flux,
            "particle_count": particles,
            "statistics": statistics_,
            "twice_projection": projection,
            "pseudopotentials": potentials,
            "three_body_terms": [value.term_id for value in terms],
            "maximum_basis_dimension": maximum_basis,
            "maximum_nonzero_routes": maximum_routes,
            "maximum_table_bytes": maximum_table,
            "tolerance": tolerance_,
        }
        self.twice_monopole_flux = flux
        self.particle_count = particles
        self.twice_projection = projection
        self.statistics = statistics_  # type: ignore[assignment]
        self.pseudopotentials = potentials
        self.three_body_terms = terms
        self.maximum_basis_dimension = maximum_basis
        self.maximum_nonzero_routes = maximum_routes
        self.maximum_table_bytes = maximum_table
        self.tolerance = tolerance_
        self.plan_id = canonical_fingerprint(content)

    @property
    def orbital_count(self) -> int:
        return self.twice_monopole_flux + 1


class FuzzyManyBodyEvidence(StrictModule):
    hermiticity_residual: Array
    maximum_particle_number_residual: Array
    maximum_twice_projection_residual: Array
    nonzero_routes: int = eqx.field(static=True)
    finite: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


class _FuzzyManyBodyLinearOperator(AbstractLinearOperator):
    rows: Array
    columns: Array
    values: Array
    dimension: int = eqx.field(static=True)

    def __init__(
        self,
        rows: ArrayLike,
        columns: ArrayLike,
        values: ArrayLike,
        dimension: int,
        /,
        *,
        self_adjoint: bool,
    ):
        rows_ = jnp.asarray(rows, dtype=jnp.int32)
        columns_ = jnp.asarray(columns, dtype=jnp.int32)
        values_ = jnp.asarray(values, dtype=jnp.complex128)
        dimension_ = int(dimension)
        if (
            rows_.shape != columns_.shape
            or rows_.shape != values_.shape
            or rows_.ndim != 1
            or dimension_ < 1
        ):
            raise ValueError("Fuzzy sparse routes and dimension are incompatible.")
        space = ArraySpace(
            (dimension_,),
            dtype=np.complex128,
            space_id=f"fuzzy-many-body:{dimension_}",
        )
        self.rows = rows_
        self.columns = columns_
        self.values = values_
        self.dimension = dimension_
        self.source = space
        self.target = space
        self.properties = OperatorProperties(
            self_adjoint=self_adjoint,
            evidence={"self_adjoint": "verified"} if self_adjoint else None,
        )
        self.capabilities = OperatorCapabilities(
            transpose=True,
            adjoint=True,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "fuzzy-many-body-linear-operator",
                "dimension": dimension_,
                "rows": array_tree_fingerprint(np.asarray(rows_)),
                "columns": array_tree_fingerprint(np.asarray(columns_)),
                "values": array_tree_fingerprint(np.asarray(values_)),
            }
        )

    def mv(self, vector: ArrayLike, /) -> Array:
        value = self.source.validate(vector)
        return self.target.validate(
            jax.ops.segment_sum(
                self.values * value[self.columns],
                self.rows,
                num_segments=self.dimension,
            )
        )

    def transpose_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(
            jax.ops.segment_sum(
                self.values * value[self.rows],
                self.columns,
                num_segments=self.dimension,
            )
        )

    def adjoint_mv(self, vector: ArrayLike, /) -> Array:
        value = self.target.validate(vector)
        return self.source.validate(
            jax.ops.segment_sum(
                jnp.conj(self.values) * value[self.rows],
                self.columns,
                num_segments=self.dimension,
            )
        )

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "Fuzzy many-body operators intentionally forbid dense materialization."
        )


def _route_hermiticity_residual(
    transitions: Mapping[tuple[int, int], complex], /
) -> float:
    norm_squared = sum(abs(value) ** 2 for value in transitions.values())
    residual_squared = sum(
        abs(value - np.conj(transitions.get((column, row), 0.0j))) ** 2
        for (row, column), value in transitions.items()
    )
    return float(math.sqrt(residual_squared / max(1.0, norm_squared)))


def _direct_sector_basis(plan: FuzzySphereManyBodyPlan, /) -> AbstractSectorBasis:
    resources = SectorBasisResourcePolicy(
        maximum_dimension=plan.maximum_basis_dimension,
        maximum_table_bytes=plan.maximum_table_bytes,
    )
    projections = tuple(
        -plan.twice_monopole_flux + 2 * index for index in range(plan.orbital_count)
    )
    if plan.twice_projection is None:
        if plan.statistics == "fermion":
            return FixedCardinalityFermionBasis(
                FermionModeOrder(
                    tuple(f"orbital-{index}" for index in range(plan.orbital_count))
                ),
                plan.particle_count,
                resources=resources,
            )
        return FixedBosonNumberBasis(
            tuple(f"orbital-{index}" for index in range(plan.orbital_count)),
            (plan.particle_count + 1,) * plan.orbital_count,
            plan.particle_count,
            resources=resources,
        )
    if plan.statistics == "fermion":
        local = tuple(((0, 0), (1, projection)) for projection in projections)
    else:
        local = tuple(
            tuple(
                (occupation, occupation * projection)
                for occupation in range(plan.particle_count + 1)
            )
            for projection in projections
        )
    return FixedAbelianChargeBasis(
        tuple(f"orbital-{index}" for index in range(plan.orbital_count)),
        local,
        ("particle-number", "twice-projection"),
        (plan.particle_count, plan.twice_projection),
        AbelianGroup((None, None)),
        resources=resources,
    )


class PreparedFuzzySphereManyBody(StrictModule):
    plan: FuzzySphereManyBodyPlan = eqx.field(static=True)
    basis: AbstractSectorBasis
    occupations: Array
    rows: Array
    columns: Array
    values: Array
    twice_projections: Array
    operator: _FuzzyManyBodyLinearOperator
    evidence: FuzzyManyBodyEvidence
    prepared_id: str = eqx.field(static=True)

    @property
    def dimension(self) -> int:
        return self.occupations.shape[0]

    def mv(self, vector: ArrayLike, /) -> Array:
        return self.operator.mv(vector)

    def dense(self, /, *, maximum_elements: int = 4_000_000) -> Array:
        if self.dimension * self.dimension > int(maximum_elements):
            raise ValueError("Dense fuzzy Hamiltonian exceeds maximum_elements.")
        matrix = jnp.zeros((self.dimension, self.dimension), dtype=self.values.dtype)
        return matrix.at[self.rows, self.columns].add(self.values)


def _pair_interaction(plan: FuzzySphereManyBodyPlan, /) -> np.ndarray:
    orbitals = plan.orbital_count
    interaction = np.zeros((orbitals, orbitals, orbitals, orbitals), dtype=np.complex128)
    for total, potential in plan.pseudopotentials:
        coefficients = np.asarray(
            su2_clebsch_gordan(
                plan.twice_monopole_flux,
                plan.twice_monopole_flux,
                total,
            )
        )
        interaction += potential * np.asarray(
            ein.contract(
                "abm,cdm->abcd",
                coefficients,
                np.conj(coefficients),
            )
        )
    return interaction


def prepare_fuzzy_sphere_many_body(
    plan: FuzzySphereManyBodyPlan,
    /,
) -> PreparedFuzzySphereManyBody:
    """Compile the fixed-N Fock sector and coalesced matrix-free routes."""

    if not isinstance(plan, FuzzySphereManyBodyPlan):
        raise TypeError("plan must be FuzzySphereManyBodyPlan.")
    basis = _direct_sector_basis(plan)
    occupation_array = np.asarray(
        jax.vmap(basis._coordinate_unchecked)(
            jnp.arange(basis.dimension, dtype=jnp.int64)
        ),
        dtype=np.int32,
    )
    occupations = tuple(tuple(int(value) for value in row) for row in occupation_array)
    rank = {value: index for index, value in enumerate(occupations)}
    pair = _pair_interaction(plan)
    transitions: dict[tuple[int, int], complex] = {}
    nonzero_pair = np.argwhere(np.abs(pair) > plan.tolerance)
    for source, occupation in enumerate(occupations):
        for first, second, third, fourth in nonzero_pair:
            applied = _apply_normal_ordered(
                occupation,
                (int(second), int(first)),
                (int(third), int(fourth)),
                plan.statistics,
            )
            if applied is None:
                continue
            target_occupation, factor = applied
            target = rank.get(target_occupation)
            if target is None:
                raise ValueError("A pair interaction leaves the declared charge sector.")
            coefficient = 0.5 * pair[first, second, third, fourth] * factor
            transitions[target, source] = (
                transitions.get((target, source), 0.0j) + coefficient
            )
        for term in plan.three_body_terms:
            applied = _apply_normal_ordered(
                occupation,
                tuple(reversed(term.creators)),
                term.annihilators,
                plan.statistics,
            )
            if applied is None:
                continue
            target_occupation, factor = applied
            target = rank.get(target_occupation)
            if target is None:
                raise ValueError("A three-body term leaves the declared charge sector.")
            transitions[target, source] = (
                transitions.get((target, source), 0.0j) + term.coefficient * factor
            )
    transitions = {
        key: value for key, value in transitions.items() if abs(value) > plan.tolerance
    }
    if len(transitions) > plan.maximum_nonzero_routes:
        raise ValueError("Fuzzy many-body routes exceed maximum_nonzero_routes.")
    ordered = tuple(sorted(transitions.items()))
    rows = np.asarray([key[0] for key, _ in ordered], dtype=np.int32)
    columns = np.asarray([key[1] for key, _ in ordered], dtype=np.int32)
    values = np.asarray([value for _, value in ordered], dtype=np.complex128)
    hermiticity = _route_hermiticity_residual(transitions)
    particle_residual = 0
    projections = np.asarray(
        [
            sum(
                occupation[index] * (-plan.twice_monopole_flux + 2 * index)
                for index in range(plan.orbital_count)
            )
            for occupation in occupations
        ],
        dtype=np.int32,
    )
    projection_residual = 0
    for row, column in zip(rows, columns, strict=True):
        particle_residual = max(
            particle_residual,
            abs(sum(occupations[row]) - sum(occupations[column])),
        )
        projection_residual = max(
            projection_residual,
            abs(int(projections[row]) - int(projections[column])),
        )
    finite = bool(np.all(np.isfinite(values)))
    accepted = (
        finite
        and hermiticity <= 100.0 * plan.tolerance
        and particle_residual == 0
        and projection_residual == 0
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "fuzzy-many-body-evidence",
            "plan": plan.plan_id,
            "occupations": array_tree_fingerprint(
                np.asarray(occupations, dtype=np.int32)
            ),
            "rows": array_tree_fingerprint(rows),
            "columns": array_tree_fingerprint(columns),
            "values": array_tree_fingerprint(values),
            "hermiticity": hermiticity,
        }
    )
    evidence = FuzzyManyBodyEvidence(
        hermiticity_residual=jnp.asarray(hermiticity),
        maximum_particle_number_residual=jnp.asarray(particle_residual),
        maximum_twice_projection_residual=jnp.asarray(projection_residual),
        nonzero_routes=len(transitions),
        finite=jnp.asarray(finite),
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )
    operator = _FuzzyManyBodyLinearOperator(
        rows,
        columns,
        values,
        basis.dimension,
        self_adjoint=accepted,
    )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-fuzzy-sphere-many-body",
            "plan": plan.plan_id,
            "evidence": evidence_id,
        }
    )
    return PreparedFuzzySphereManyBody(
        plan=plan,
        basis=basis,
        occupations=jnp.asarray(occupations, dtype=jnp.int32),
        rows=jnp.asarray(rows),
        columns=jnp.asarray(columns),
        values=jnp.asarray(values),
        twice_projections=jnp.asarray(projections),
        operator=operator,
        evidence=evidence,
        prepared_id=prepared_id,
    )


class FuzzyManyBodyObservables(StrictModule):
    occupations: Array
    one_body_density: Array
    orbital_entanglement_spectrum: Array
    orbital_entanglement_entropy: Array
    total_twice_projection: Array
    norm_residual: Array
    evidence_id: str = eqx.field(static=True)


def evaluate_fuzzy_many_body_observables(
    prepared: PreparedFuzzySphereManyBody,
    state: ArrayLike,
    /,
    *,
    orbital_cut: int,
) -> FuzzyManyBodyObservables:
    if not isinstance(prepared, PreparedFuzzySphereManyBody):
        raise TypeError("prepared must be PreparedFuzzySphereManyBody.")
    vector = np.asarray(state, dtype=np.complex128)
    if vector.shape != (prepared.dimension,):
        raise ValueError("Fuzzy many-body state has the wrong shape.")
    norm = float(np.vdot(vector, vector).real)
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("Fuzzy many-body state must have finite positive norm.")
    vector = vector / math.sqrt(norm)
    occupations = np.asarray(prepared.occupations)
    mean_occupations = np.sum(np.abs(vector[:, None]) ** 2 * occupations, axis=0)
    density = np.zeros(
        (prepared.plan.orbital_count, prepared.plan.orbital_count),
        dtype=np.complex128,
    )
    rank = {tuple(value): index for index, value in enumerate(occupations.tolist())}
    for source, occupation_array in enumerate(occupations):
        occupation = tuple(occupation_array)
        for annihilator in range(prepared.plan.orbital_count):
            removed = _apply_annihilation(
                occupation,
                annihilator,
                prepared.plan.statistics,
            )
            if removed is None:
                continue
            intermediate, first = removed
            for creator in range(prepared.plan.orbital_count):
                created = _apply_creation(intermediate, creator, prepared.plan.statistics)
                if created is None:
                    continue
                target, second = created
                target_index = rank.get(target)
                if target_index is None:
                    continue
                density[creator, annihilator] += (
                    np.conj(vector[target_index]) * first * second * vector[source]
                )
    cut = int(orbital_cut)
    if cut <= 0 or cut >= prepared.plan.orbital_count:
        raise ValueError("orbital_cut must split the orbital roster.")
    left_states = tuple(sorted({tuple(value[:cut]) for value in occupations.tolist()}))
    right_states = tuple(sorted({tuple(value[cut:]) for value in occupations.tolist()}))
    left_rank = {value: index for index, value in enumerate(left_states)}
    right_rank = {value: index for index, value in enumerate(right_states)}
    amplitudes = np.zeros((len(left_states), len(right_states)), dtype=np.complex128)
    for coefficient, occupation in zip(vector, occupations.tolist(), strict=True):
        amplitudes[
            left_rank[tuple(occupation[:cut])], right_rank[tuple(occupation[cut:])]
        ] = coefficient
    singular_values = np.linalg.svd(amplitudes, compute_uv=False)
    probabilities = singular_values**2
    probabilities = probabilities[probabilities > prepared.plan.tolerance]
    entanglement = -np.log(probabilities)
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    total_projection = float(
        np.sum(np.abs(vector) ** 2 * np.asarray(prepared.twice_projections))
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "fuzzy-many-body-observables",
            "prepared": prepared.prepared_id,
            "state": array_tree_fingerprint(vector),
            "orbital_cut": cut,
        }
    )
    return FuzzyManyBodyObservables(
        occupations=jnp.asarray(mean_occupations),
        one_body_density=jnp.asarray(density),
        orbital_entanglement_spectrum=jnp.asarray(entanglement),
        orbital_entanglement_entropy=jnp.asarray(entropy),
        total_twice_projection=jnp.asarray(total_projection),
        norm_residual=jnp.asarray(abs(norm - 1.0)),
        evidence_id=evidence_id,
    )


class FuzzyGeometricPhaseEvidence(StrictModule):
    berry_phase: Array
    fubini_study_distances: Array
    minimum_overlap: Array
    closed_path_residual: Array
    accepted: Array
    evidence_id: str = eqx.field(static=True)


def fuzzy_geometric_phase(
    state_path: ArrayLike,
    /,
    *,
    minimum_overlap: float = 1e-8,
) -> FuzzyGeometricPhaseEvidence:
    states = np.asarray(state_path, dtype=np.complex128)
    if states.ndim != 2 or states.shape[0] < 3:
        raise ValueError("A geometric phase path requires at least three state vectors.")
    norms = np.linalg.norm(states, axis=1)
    if np.any(norms == 0.0) or not np.all(np.isfinite(states)):
        raise ValueError("Geometric phase states must be finite and nonzero.")
    states = states / norms[:, None]
    overlaps = np.asarray(
        [
            np.vdot(states[index], states[index + 1])
            for index in range(states.shape[0] - 1)
        ]
        + [np.vdot(states[-1], states[0])]
    )
    minimum = float(np.min(np.abs(overlaps)))
    phase = float(-np.angle(np.prod(overlaps / np.abs(overlaps))))
    distances = np.arccos(np.clip(np.abs(overlaps), 0.0, 1.0))
    closure = float(
        np.linalg.norm(
            states[0]
            - states[-1]
            * np.vdot(states[-1], states[0])
            / abs(np.vdot(states[-1], states[0]))
        )
    )
    accepted = minimum >= float(minimum_overlap)
    evidence_id = canonical_fingerprint(
        {
            "kind": "fuzzy-geometric-phase-evidence",
            "states": array_tree_fingerprint(states),
            "minimum_overlap": float(minimum_overlap),
        }
    )
    return FuzzyGeometricPhaseEvidence(
        berry_phase=jnp.asarray(phase),
        fubini_study_distances=jnp.asarray(distances),
        minimum_overlap=jnp.asarray(minimum),
        closed_path_residual=jnp.asarray(closure),
        accepted=jnp.asarray(accepted),
        evidence_id=evidence_id,
    )


def _tensor_train_state(
    dense: np.ndarray,
    dimensions: tuple[int, ...],
    maximum_bond_dimension: int,
    /,
) -> tuple[tuple[np.ndarray, ...], float]:
    tensor = dense.reshape(dimensions)
    tensors: list[np.ndarray] = []
    left_dimension = 1
    discarded = 0.0
    for site, physical in enumerate(dimensions[:-1]):
        matrix = tensor.reshape(left_dimension * physical, -1)
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        keep = min(maximum_bond_dimension, singular.size)
        discarded += float(np.sum(singular[keep:] ** 2))
        tensors.append(left[:, :keep].reshape(left_dimension, physical, keep))
        tensor = singular[:keep, None] * right[:keep]
        left_dimension = keep
    tensors.append(tensor.reshape(left_dimension, dimensions[-1], 1))
    return tuple(tensors), discarded


def lower_fuzzy_state_to_mps(
    prepared: PreparedFuzzySphereManyBody,
    state: ArrayLike,
    /,
    *,
    maximum_bond_dimension: int,
) -> tuple[MatrixProductState, float]:
    """Lower fixed-N amplitudes to an orbital-chain MPS by bounded TT-SVD."""

    vector = np.asarray(state, dtype=np.complex128)
    if vector.shape != (prepared.dimension,):
        raise ValueError("Fuzzy state has the wrong shape.")
    local_dimension = (
        2 if prepared.plan.statistics == "fermion" else prepared.plan.particle_count + 1
    )
    dimensions = (local_dimension,) * prepared.plan.orbital_count
    dense = np.zeros((math.prod(dimensions),), dtype=np.complex128)
    for coefficient, occupation in zip(
        vector, np.asarray(prepared.occupations), strict=True
    ):
        dense[np.ravel_multi_index(tuple(occupation), dimensions)] = coefficient
    tensors, discarded = _tensor_train_state(
        dense,
        dimensions,
        int(maximum_bond_dimension),
    )
    return MatrixProductState(tensors), discarded


def lower_fuzzy_hamiltonian_to_mpo(
    prepared: PreparedFuzzySphereManyBody,
    /,
    *,
    maximum_bond_dimension: int,
    maximum_dense_elements: int = 4_000_000,
) -> tuple[MatrixProductOperator, float]:
    """Lower the bounded many-body Hamiltonian to an orbital MPO by TT-SVD."""

    local_dimension = (
        2 if prepared.plan.statistics == "fermion" else prepared.plan.particle_count + 1
    )
    dimensions = (local_dimension,) * prepared.plan.orbital_count
    dense_dimension = math.prod(dimensions)
    if dense_dimension * dense_dimension > int(maximum_dense_elements):
        raise ValueError("Dense orbital embedding exceeds maximum_dense_elements.")
    full = np.zeros((dense_dimension, dense_dimension), dtype=np.complex128)
    occupation_table = np.asarray(prepared.occupations)
    indices = np.asarray(
        [np.ravel_multi_index(tuple(value), dimensions) for value in occupation_table]
    )
    reduced = np.asarray(prepared.dense(maximum_elements=maximum_dense_elements))
    full[np.ix_(indices, indices)] = reduced
    interleaved = full.reshape(dimensions + dimensions)
    permutation = tuple(
        value
        for pair in zip(
            range(len(dimensions)),
            range(len(dimensions), 2 * len(dimensions)),
            strict=True,
        )
        for value in pair
    )
    tensor = np.transpose(interleaved, permutation).reshape(
        tuple(value * value for value in dimensions)
    )
    tensors, discarded = _tensor_train_state(
        tensor.reshape(-1),
        tuple(value * value for value in dimensions),
        int(maximum_bond_dimension),
    )
    mpo_tensors = tuple(
        value.reshape(value.shape[0], dimension, dimension, value.shape[-1])
        for value, dimension in zip(tensors, dimensions, strict=True)
    )
    return MatrixProductOperator(mpo_tensors), discarded


class FuzzyOperatorStateEvidence(StrictModule):
    observed_counting: tuple[int, ...] = eqx.field(static=True)
    expected_counting: tuple[int, ...] = eqx.field(static=True)
    maximum_level_deviation: int = eqx.field(static=True)
    finite_size_sequences: Array
    matched: Array
    claim: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def assess_fuzzy_operator_state_correspondence(
    entanglement_levels: ArrayLike,
    expected_counting: Sequence[int],
    /,
    *,
    level_tolerance: float = 1e-6,
) -> FuzzyOperatorStateEvidence:
    levels = np.sort(np.asarray(entanglement_levels, dtype=np.float64))
    expected = tuple(expected_counting)
    if (
        levels.ndim != 1
        or not levels.size
        or not expected
        or any(value < 1 for value in expected)
    ):
        raise ValueError("Operator-state evidence inputs are invalid.")
    groups: list[int] = []
    start = 0
    for index in range(1, levels.size + 1):
        if index == levels.size or abs(levels[index] - levels[start]) > level_tolerance:
            groups.append(index - start)
            start = index
    observed = tuple(groups[: len(expected)])
    deviation = max(
        (
            abs(left - right)
            for left, right in itertools.zip_longest(observed, expected, fillvalue=0)
        ),
        default=0,
    )
    matched = observed == expected
    evidence_id = canonical_fingerprint(
        {
            "kind": "fuzzy-operator-state-evidence",
            "levels": array_tree_fingerprint(levels),
            "expected_counting": expected,
            "level_tolerance": float(level_tolerance),
        }
    )
    return FuzzyOperatorStateEvidence(
        observed_counting=observed,
        expected_counting=expected,
        maximum_level_deviation=deviation,
        finite_size_sequences=jnp.asarray(levels),
        matched=jnp.asarray(matched),
        claim="finite-size-counting-evidence-not-cft-identification",
        evidence_id=evidence_id,
    )


def run_fuzzy_continuum_study(
    inverse_fluxes: Sequence[float],
    values: Sequence[float],
    standard_errors: Sequence[float],
    /,
) -> ScientificLimitStudyResult:
    points = tuple(float(value) for value in inverse_fluxes)
    observations = tuple(float(value) for value in values)
    errors = tuple(float(value) for value in standard_errors)
    if len(points) < 4 or len(points) != len(observations) or len(points) != len(errors):
        raise ValueError("Fuzzy continuum studies require four aligned points.")
    axis = ScientificLimitAxis(
        "inverse-flux", 0.0, minimum_span=max(points) - min(points)
    )
    plan = ScientificLimitStudyPlan(
        (axis,),
        (
            ScientificLimitVariation("linear", {"inverse-flux": 1}, minimum_points=3),
            ScientificLimitVariation("quadratic", {"inverse-flux": 2}, minimum_points=4),
        ),
    )
    data = tuple(
        ScientificLimitDatum(
            f"flux-{index}",
            {"inverse-flux": coordinate},
            value,
            error,
        )
        for index, (coordinate, value, error) in enumerate(
            zip(points, observations, errors, strict=True)
        )
    )
    return run_scientific_limit_study(plan, data)


__all__ = [
    "FuzzyGeometricPhaseEvidence",
    "FuzzyManyBodyEvidence",
    "FuzzyManyBodyObservables",
    "FuzzyManyBodyStatistics",
    "FuzzyOperatorStateEvidence",
    "FuzzySphereManyBodyPlan",
    "FuzzyThreeBodyTerm",
    "PreparedFuzzySphereManyBody",
    "assess_fuzzy_operator_state_correspondence",
    "evaluate_fuzzy_many_body_observables",
    "fuzzy_geometric_phase",
    "lower_fuzzy_hamiltonian_to_mpo",
    "lower_fuzzy_state_to_mps",
    "prepare_fuzzy_sphere_many_body",
    "run_fuzzy_continuum_study",
]
