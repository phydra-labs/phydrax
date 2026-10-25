"""Native point-charge electrostatics and the term-owned reciprocal mesh.

References come from ``tests/_support/atomistic_long_range_reference.py``
(independent NumPy sums), exact Madelung constants, alpha independence of a
converged Ewald split and finite differences.
"""

import itertools
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._model._ports import ValuePort
from phydrax.atomistic._conditioning import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
)
from phydrax.atomistic._electrostatics import (
    DampedShiftedForceCoulombPotential,
    DirectCoulombPotential,
    estimate_ewald_splitting,
    estimate_pme_splitting,
    EwaldReferencePotential,
    ParticleMeshEwaldPotential,
    ScreenedCoulombPotential,
    ScreenedTruncation,
)
from phydrax.atomistic._neighborhood_bundle import AtomisticNeighborhoodRole
from phydrax.atomistic._reciprocal_mesh import (
    ReciprocalBudget,
    ReciprocalMeshPlan,
    ReciprocalModeSetPlan,
)
from phydrax.discretization import cardinal_bspline_periodic_symbol
from phydrax.units import CHARGE, ELEMENTARY_CHARGE
from tests._support.atomistic_bundles import (
    bundle_state,
    prepare_program,
    role_capacity,
    search_capacity,
)
from tests._support.atomistic_long_range_reference import (
    central_difference,
    converged_ewald_energy,
    CSCL_MADELUNG,
    dispersion_lattice_energy,
    dsf_energy,
    NACL_MADELUNG,
    slab_ewald2d_energy,
    structure_factor,
    yukawa_energy,
)


TRICLINIC = np.asarray([[6.0, 0.0, 0.0], [1.0, 6.5, 0.0], [0.5, 1.0, 7.0]])


def _system(charges: Any, *, cell: Any = None) -> Any:
    count = len(charges)
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(count, dtype=np.int64),
        np.ones((count,), dtype=np.int64),
        np.ones((count,), dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.zeros((count,), dtype=np.int64),
        charges=np.asarray(charges, dtype=np.float64),
        cell=cell,
    ).prepare()


def _resources(count: int) -> dict[str, Any]:
    # Explicit bundle of the former dense neighborhood: every unordered pair for
    # complete requirements, and one cell-list source wide enough to hold every
    # directed candidate route of every image for bounded requirements.
    return {
        "search": phx.atomistic.AtomisticCellListImageSearch(
            maximum_candidate_slots=1 << 16, deformation_margin=0.05
        ),
        "skin": 0.5,
        "search_capacities": (
            search_capacity(27 * count * count, particles_per_cell=count, images=27),
        ),
        # Storage includes skin candidates, which can have repeated images even
        # when the physical cutoff is within the minimum-image radius.
        "role_capacities": (role_capacity(27 * count * count, 27 * count),),
        "maximum_pairs": count * (count - 1) // 2,
    }


def _bound(
    term: Any,
    charges: Any,
    positions: Any,
    *,
    cell: Any = None,
    cell_vectors: Array | None = None,
) -> Any:
    system = _system(charges, cell=cell)
    definition = phx.atomistic.AtomisticPotentialProgram([term])
    bounded = tuple(
        requirement.bounded_cutoff()
        for requirement in definition.requirements.neighborhoods
        if not requirement.complete
    )
    frame = None
    if cell is None and bounded:
        coordinates = np.asarray(positions, dtype=np.float64)
        padding = max(bounded) + 0.5
        frame = phx.discretization.PeriodicCell(
            np.diag(np.ptp(coordinates, axis=0) + 2.0 * padding),
            origin=np.min(coordinates, axis=0) - padding,
            periodic_axes=(False, False, False),
        )
    program = prepare_program(
        definition, system, frame=frame, **_resources(system.capacity)
    )
    position = jnp.asarray(positions, dtype=jnp.float64)
    state = bundle_state(program, position, cell_vectors=cell_vectors)
    return program, state, position


def _energy(term: Any, charges: Any, positions: Any, *, cell: Any = None) -> float:
    program, state, position = _bound(term, charges, positions, cell=cell)
    evaluation = program.evaluate(position, state)
    assert bool(evaluation.successful)
    return float(evaluation.energy)


def _term_evaluation(
    term: Any,
    charges: Any,
    positions: Any,
    *,
    cell: Any = None,
    cell_vectors: Array | None = None,
    **kwargs: Any,
) -> Any:
    # The bundle state is bound at the exact evaluated point and cell.
    program, state, position = _bound(
        term, charges, positions, cell=cell, cell_vectors=cell_vectors
    )
    return program.terms[0].energy(
        program.context(position, state, cell_vectors=cell_vectors, **kwargs)
    )


def _configuration(
    seed: int, vectors: np.ndarray, count: int, minimum: float
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shifts = np.asarray(list(itertools.product((-1, 0, 1), repeat=3))) @ vectors
    points: list[np.ndarray] = []
    while len(points) < count:
        candidate = rng.uniform(0.0, 1.0, 3) @ vectors
        if all(
            np.min(np.linalg.norm(candidate - point + shifts, axis=1)) > minimum
            for point in points
        ):
            points.append(candidate)
    return np.asarray(points)


def _neutral(seed: int, count: int) -> np.ndarray:
    values = np.random.default_rng(seed).normal(size=count)
    return values - values.mean()


def _charge_port() -> ValuePort:
    return ValuePort(
        "test.atom-partial-charge",
        event_shape=(),
        component_ids=("charge",),
        representation="atom-partial-charge",
        dimensions=(CHARGE,),
        normalization_id="identity",
    )


# --- independent periodic references ---------------------------------------


def test_ewald_reproduces_rock_salt_madelung_energy() -> None:
    cell = phx.discretization.PeriodicCell(6.0 * np.eye(3))
    base = np.asarray(list(itertools.product((0.0, 3.0), repeat=3)))
    charges = [
        1.0 if int(round(np.sum(point) / 3.0)) % 2 == 0 else -1.0 for point in base
    ]
    energy = _energy(EwaldReferencePotential(1.6, 2.9, 16), charges, base, cell=cell)
    np.testing.assert_allclose(energy, -4.0 * NACL_MADELUNG / 3.0, rtol=1.0e-8)


def test_ewald_reproduces_caesium_chloride_madelung_energy() -> None:
    cell = phx.discretization.PeriodicCell(6.0 * np.eye(3))
    positions = np.asarray([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0]])
    energy = _energy(
        EwaldReferencePotential(1.6, 2.9, 16), [1.0, -1.0], positions, cell=cell
    )
    np.testing.assert_allclose(
        energy, -CSCL_MADELUNG / (3.0 * math.sqrt(3.0)), rtol=1.0e-8
    )


def test_ewald_rotated_primitive_triclinic_rock_salt() -> None:
    angle = 0.37
    rotation = np.asarray(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle) * 0.6, math.cos(angle) * 0.6, -0.8],
            [math.sin(angle) * 0.8, math.cos(angle) * 0.8, 0.6],
        ]
    )
    primitive = np.asarray([[0.0, 3.0, 3.0], [3.0, 0.0, 3.0], [3.0, 3.0, 0.0]])
    vectors = primitive @ rotation.T
    positions = np.asarray([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]) @ rotation.T
    cell = phx.discretization.PeriodicCell(vectors)
    energy = _energy(
        EwaldReferencePotential(2.2, 2.05, 16), [1.0, -1.0], positions, cell=cell
    )
    np.testing.assert_allclose(energy, -NACL_MADELUNG / 3.0, rtol=1.0e-7)


def test_ewald_triclinic_matches_independent_converged_reference() -> None:
    positions = _configuration(3, TRICLINIC, 6, 0.9)
    charges = _neutral(5, 6)
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    reference, history = converged_ewald_energy(
        positions, charges, TRICLINIC, 1.2, maximum_shells=16
    )
    assert len(history) >= 2
    native = _energy(EwaldReferencePotential(1.6, 2.8, 20), charges, positions, cell=cell)
    other = _energy(EwaldReferencePotential(1.4, 2.8, 20), charges, positions, cell=cell)
    np.testing.assert_allclose(native, reference, rtol=1.0e-8, atol=1.0e-9)
    np.testing.assert_allclose(other, native, rtol=1.0e-6, atol=1.0e-7)


def test_uniform_background_is_alpha_independent_and_neutrality_refuses() -> None:
    positions = _configuration(7, TRICLINIC, 4, 1.0)
    charges = np.asarray([1.0, -0.5, 0.25, 0.75])
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    reference, _ = converged_ewald_energy(
        positions, charges, TRICLINIC, 1.2, background=True, maximum_shells=16
    )
    energies = [
        _energy(
            EwaldReferencePotential(alpha, 2.8, 20, neutrality="uniform-background"),
            charges,
            positions,
            cell=cell,
        )
        for alpha in (1.4, 1.6)
    ]
    np.testing.assert_allclose(energies[1], reference, rtol=1.0e-8, atol=1.0e-9)
    np.testing.assert_allclose(energies[0], energies[1], rtol=1.0e-6, atol=1.0e-7)
    refused = _term_evaluation(
        EwaldReferencePotential(1.6, 2.8, 8), charges, positions, cell=cell
    )
    (evidence,) = refused.evidence
    assert not bool(refused.successful)
    assert bool(jnp.isnan(refused.energy))
    assert not bool(evidence.neutral)
    np.testing.assert_allclose(evidence.total_charge, 1.5, atol=1.0e-12)
    assert bool(evidence.charges_successful) and bool(evidence.cell_successful)
    assert not refused.atom_energy_complete


def test_vacuum_sphere_boundary_adds_dipole_term() -> None:
    positions = _configuration(11, TRICLINIC, 4, 1.0)
    charges = _neutral(13, 4)
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    tin = _energy(EwaldReferencePotential(1.6, 2.8, 12), charges, positions, cell=cell)
    vacuum = _energy(
        EwaldReferencePotential(1.6, 2.8, 12, boundary="vacuum-sphere"),
        charges,
        positions,
        cell=cell,
    )
    dipole = np.asarray(charges) @ positions
    volume = abs(np.linalg.det(TRICLINIC))
    np.testing.assert_allclose(
        vacuum - tin, 2.0 * math.pi * dipole @ dipole / (3.0 * volume), rtol=1.0e-10
    )
    with pytest.raises(ValueError, match="require-neutral"):
        EwaldReferencePotential(
            1.6, 2.8, 12, boundary="vacuum-sphere", neutrality="uniform-background"
        )


# --- slab ------------------------------------------------------------------


def _slab_cell(height: float) -> Any:
    return phx.discretization.PeriodicCell(
        np.asarray([[6.0, 0.0, 0.0], [1.5, 6.0, 0.0], [0.0, 0.0, height]]),
        periodic_axes=(True, True, False),
    )


def _slab_positions() -> np.ndarray:
    rng = np.random.default_rng(17)
    planar = rng.uniform(0.0, 1.0, (4, 2)) @ np.asarray([[6.0, 0.0], [1.5, 6.0]])
    heights = np.asarray([1.0, 1.8, 2.7, 3.5])
    return np.column_stack((planar, heights))


def _slab_term(height: float, **kwargs: Any) -> EwaldReferencePotential:
    axial = math.ceil(13.2 * height / (2.0 * math.pi)) + 1
    return EwaldReferencePotential(
        1.2, 2.8, (14, 14, axial), boundary="slab-yeh-berkowitz", **kwargs
    )


def test_slab_correction_converges_to_two_dimensional_ewald() -> None:
    positions = _slab_positions()
    charges = np.asarray([1.0, -1.0, 1.0, -1.0])
    planar = np.asarray([[6.0, 0.0], [1.5, 6.0]])
    reference = slab_ewald2d_energy(
        positions, charges, planar, 1.0, real_shells=4, reciprocal_shells=13
    )
    check = slab_ewald2d_energy(
        positions, charges, planar, 0.9, real_shells=4, reciprocal_shells=13
    )
    np.testing.assert_allclose(check, reference, rtol=1.0e-7)
    errors = [
        abs(
            _energy(_slab_term(height), charges, positions, cell=_slab_cell(height))
            - reference
        )
        for height in (8.0, 14.0)
    ]
    assert errors[1] < errors[0]
    assert errors[1] < 1.0e-3 * abs(reference)


def test_charged_slab_is_translation_invariant_along_the_normal() -> None:
    positions = _slab_positions()[:3]
    charges = [1.0, -1.0, 0.5]
    cell = _slab_cell(12.0)
    term = _slab_term(12.0, neutrality="uniform-background")
    shifted = positions + np.asarray([0.0, 0.0, 0.7])
    np.testing.assert_allclose(
        _energy(term, charges, shifted, cell=cell),
        _energy(term, charges, positions, cell=cell),
        rtol=1.0e-9,
    )


def test_invalid_periodicity_refuses_instead_of_zero_correction() -> None:
    positions = _slab_positions()
    charges = [1.0, -1.0, 1.0, -1.0]
    with pytest.raises(ValueError, match="exactly two periodic"):
        _bound(
            EwaldReferencePotential(1.2, 2.8, 8, boundary="slab-yeh-berkowitz"),
            charges,
            positions,
            cell=phx.discretization.PeriodicCell(8.0 * np.eye(3)),
        )
    with pytest.raises(ValueError, match="fully periodic"):
        _bound(
            EwaldReferencePotential(1.2, 2.8, 8), charges, positions, cell=_slab_cell(9.0)
        )
    with pytest.raises(ValueError, match="exactly two periodic"):
        _bound(
            EwaldReferencePotential(1.2, 2.8, 8, boundary="slab-yeh-berkowitz"),
            charges,
            positions,
            cell=phx.discretization.PeriodicCell(
                8.0 * np.eye(3), periodic_axes=(True, False, False)
            ),
        )
    with pytest.raises(ValueError):
        _bound(DirectCoulombPotential(), charges, positions, cell=_slab_cell(9.0))


# --- PME -------------------------------------------------------------------


def _pme_case() -> tuple[np.ndarray, np.ndarray, Any, float]:
    positions = _configuration(23, TRICLINIC, 6, 0.9)
    charges = _neutral(29, 6)
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    reference = _energy(
        EwaldReferencePotential(1.0, 2.8, 14), charges, positions, cell=cell
    )
    return positions, charges, cell, reference


def test_pme_refines_with_grid_and_spline_degree() -> None:
    positions, charges, cell, reference = _pme_case()

    def error(shape: int, degree: int, **kwargs: Any) -> float:
        term = ParticleMeshEwaldPotential(
            1.0, 2.8, (shape, shape, shape), spline_degree=degree, **kwargs
        )
        return abs(_energy(term, charges, positions, cell=cell) - reference)

    by_grid = [error(shape, 3) for shape in (12, 24, 48)]
    assert by_grid[0] > by_grid[1] > by_grid[2]
    assert by_grid[2] < 1.0e-4 * max(1.0, abs(reference))
    by_degree = [error(20, degree) for degree in (1, 3, 5)]
    assert by_degree[0] > by_degree[1] > by_degree[2]
    for modulus in ("discrete-bspline-modulus", "continuous-sinc-window"):
        for nyquist in ("symmetrized", "excluded"):
            assert error(48, 4, modulus=modulus, nyquist=nyquist) < 1.0e-4 * max(
                1.0, abs(reference)
            )


def test_pme_evidence_reports_mesh_and_components() -> None:
    positions, charges, cell, _ = _pme_case()
    evaluation = _term_evaluation(
        ParticleMeshEwaldPotential(1.0, 2.8, (16, 18, 20), spline_degree=5),
        charges,
        positions,
        cell=cell,
    )
    (evidence,) = evaluation.evidence
    assert bool(evaluation.successful)
    assert "degree-5" in evidence.method and "nyquist-symmetrized" in evidence.method
    total = (
        evidence.real_energy
        + evidence.reciprocal_energy
        + evidence.self_energy
        + evidence.exception_energy
        + evidence.background_energy
        + evidence.boundary_energy
    )
    np.testing.assert_allclose(evaluation.energy, total, rtol=1.0e-12)
    np.testing.assert_allclose(evidence.volume, abs(np.linalg.det(TRICLINIC)))
    assert not evaluation.atom_energy_complete


# --- coverage, identity and resources ---------------------------------------


@pytest.mark.parametrize(
    "term",
    [
        EwaldReferencePotential(1.0, 2.5, 6, reciprocal_cutoff=7.0),
        ParticleMeshEwaldPotential(1.0, 2.5, (16, 16, 16), reciprocal_cutoff=8.0),
    ],
)
def test_declared_mode_coverage_fails_under_cell_expansion(term: Any) -> None:
    cell = phx.discretization.PeriodicCell(6.0 * np.eye(3), maximum_condition_number=2.0)
    positions = np.asarray([[1.0, 1.0, 1.0], [3.0, 1.2, 1.4]])
    charges = [1.0, -1.0]
    accepted = _term_evaluation(term, charges, positions, cell=cell)
    assert bool(accepted.successful)
    assert bool(accepted.evidence[0].mode_coverage_successful)
    expanded = _term_evaluation(
        term,
        charges,
        positions * 1.1,
        cell=cell,
        cell_vectors=jnp.asarray(6.6 * np.eye(3)),
    )
    (evidence,) = expanded.evidence
    assert not bool(evidence.mode_coverage_successful)
    assert bool(evidence.cell_successful)
    assert float(evidence.support_radius) < float(accepted.evidence[0].support_radius)
    assert not bool(expanded.successful)


def test_prepared_artifacts_refuse_a_stale_context() -> None:
    cell = phx.discretization.PeriodicCell(6.0 * np.eye(3))
    positions = np.asarray([[1.0, 1.0, 1.0], [3.0, 1.2, 1.4]])
    term = ParticleMeshEwaldPotential(1.0, 2.5, (8, 8, 8))
    program_a, state_a, position = _bound(term, [1.0, -1.0], positions, cell=cell)
    program_b, _, _ = _bound(term, [0.5, -0.5], positions, cell=cell)
    evaluation = program_b.terms[0].energy(program_a.context(position, state_a))
    assert not bool(evaluation.successful)
    assert np.isnan(float(evaluation.energy))


def test_reciprocal_resource_admission_refuses() -> None:
    cell = phx.discretization.PeriodicCell(6.0 * np.eye(3))
    positions = np.asarray([[1.0, 1.0, 1.0], [3.0, 1.2, 1.4]])
    with pytest.raises(ValueError, match="grid_points"):
        _bound(
            ParticleMeshEwaldPotential(
                1.0, 2.5, (16, 16, 16), budget=ReciprocalBudget(maximum_grid_points=1000)
            ),
            [1.0, -1.0],
            positions,
            cell=cell,
        )
    with pytest.raises(ValueError, match="modes"):
        _bound(
            EwaldReferencePotential(
                1.0, 2.5, 6, budget=ReciprocalBudget(maximum_modes=50)
            ),
            [1.0, -1.0],
            positions,
            cell=cell,
        )


def test_explicit_image_real_space_requests_image_pair_sum_role() -> None:
    term = EwaldReferencePotential(
        1.0, 4.0, 6, real_space="explicit-image", name="small-cell-ewald"
    )
    (requirement,) = term.requirements.neighborhoods
    assert requirement.role is AtomisticNeighborhoodRole.IMAGE_PAIR_SUM
    assert requirement.owner == "small-cell-ewald"
    assert requirement.cutoff == 4.0
    small = _system([1.0, -1.0], cell=phx.discretization.PeriodicCell(5.0 * np.eye(3)))
    term.prepare(small)
    with pytest.raises(ValueError, match="unique-image"):
        EwaldReferencePotential(1.0, 4.0, 6).prepare(small)


def test_charge_field_dependency_identity_and_unit() -> None:
    units = phx.atomistic.AtomisticUnitSystem.reduced()
    reference = AtomisticFieldReference(
        "charge-model",
        AtomisticObservable.CHARGES,
        _charge_port(),
        level=AtomisticConditionLevel.ATOM,
        unit=units.charge_unit,
    )
    for term in (
        DirectCoulombPotential(charge_field=reference),
        DampedShiftedForceCoulombPotential(0.2, 4.0, charge_field=reference),
        EwaldReferencePotential(1.0, 2.5, 4, charge_field=reference),
        ParticleMeshEwaldPotential(1.0, 2.5, (8, 8, 8), charge_field=reference),
    ):
        (dependency,) = term.requirements.field_dependencies
        assert dependency.reference_id == reference.reference_id
    assert (
        DirectCoulombPotential(charge_field=reference).term_id
        != DirectCoulombPotential().term_id
    )
    with pytest.raises(ValueError, match="CHARGES"):
        DirectCoulombPotential(
            charge_field=AtomisticFieldReference(
                "charge-model",
                AtomisticObservable.DIPOLE,
                _charge_port(),
                level=AtomisticConditionLevel.ATOM,
                unit=units.charge_unit,
            )
        )
    foreign = AtomisticFieldReference(
        "charge-model",
        AtomisticObservable.CHARGES,
        _charge_port(),
        level=AtomisticConditionLevel.ATOM,
        unit=ELEMENTARY_CHARGE,
    )
    with pytest.raises(ValueError, match="unit"):
        DirectCoulombPotential(charge_field=foreign).prepare(_system([1.0, -1.0]))


def test_splitting_estimates_are_uncertified_heuristics() -> None:
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    estimate = estimate_ewald_splitting(cell, 2.8, 1.0e-8)
    assert not estimate.certified and estimate.estimator == "gaussian-tail-heuristic"
    assert estimate.reciprocal_extent is not None
    positions = _configuration(3, TRICLINIC, 6, 0.9)
    charges = _neutral(5, 6)
    reference, _ = converged_ewald_energy(
        positions, charges, TRICLINIC, 1.2, maximum_shells=16
    )
    energy = _energy(
        EwaldReferencePotential(
            estimate.alpha,
            2.8,
            estimate.reciprocal_extent,
            reciprocal_cutoff=estimate.reciprocal_cutoff,
        ),
        charges,
        positions,
        cell=cell,
    )
    np.testing.assert_allclose(energy, reference, rtol=1.0e-5, atol=1.0e-6)
    mesh = estimate_pme_splitting(cell, 2.8, 1.0e-6, spline_degree=5)
    assert mesh.grid_shape is not None and not mesh.certified
    assert all(size % 2 == 0 for size in mesh.grid_shape)


# --- derivatives -------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    [
        EwaldReferencePotential(1.2, 2.8, 12),
        ParticleMeshEwaldPotential(1.2, 2.8, (24, 24, 24), spline_degree=4),
    ],
)
def test_forces_strain_and_charge_derivatives_match_finite_differences(
    term: Any,
) -> None:
    positions = _configuration(31, TRICLINIC, 4, 1.0)
    charges = _neutral(37, 4)
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    program, state, position = _bound(term, charges, positions, cell=cell)

    def energy_at(value: np.ndarray) -> float:
        # A fresh bundle state certified at the displaced point.
        displaced = jnp.asarray(value)
        return float(program.energy(displaced, bundle_state(program, displaced))[0])

    evaluation = program.evaluate(position, state, compute_stress=True)
    assert bool(evaluation.successful)
    np.testing.assert_allclose(
        -np.asarray(evaluation.forces),
        central_difference(energy_at, positions, 1.0e-5),
        rtol=1.0e-6,
        atol=1.0e-8,
    )

    def strained(strain: np.ndarray) -> float:
        deformation = np.eye(3) + strain
        deformed = jnp.asarray(positions @ deformation.T)
        vectors = jnp.asarray(TRICLINIC @ deformation.T)
        return float(
            program.energy(
                deformed,
                bundle_state(program, deformed, cell_vectors=vectors),
                cell_vectors=vectors,
            )[0]
        )

    np.testing.assert_allclose(
        np.asarray(evaluation.strain_derivative),
        central_difference(strained, np.zeros((3, 3)), 1.0e-6),
        rtol=1.0e-5,
        atol=1.0e-7,
    )
    prepared = program.terms[0]
    context = program.context(position, state)

    def charge_energy(values: Array) -> Array:
        rebound = eqx.tree_at(lambda value: value.system.plan.charges, prepared, values)
        return rebound.energy(context).energy

    gradient = jax.grad(charge_energy)(jnp.asarray(charges))
    # require-neutral defines a charge-conserving manifold; varying one charge
    # alone is deliberately refused, so compare tangent directions e_i - e_last.
    tangent = np.vstack(
        (
            np.eye(charges.size - 1, dtype=np.float64),
            -np.ones((1, charges.size - 1), dtype=np.float64),
        )
    )
    reference = central_difference(
        lambda offset: _energy(term, charges + tangent @ offset, positions, cell=cell),
        np.zeros(charges.size - 1, dtype=np.float64),
        1.0e-6,
    )
    np.testing.assert_allclose(
        np.asarray(gradient[:-1] - gradient[-1]), reference, rtol=1.0e-6, atol=1.0e-8
    )


def test_pme_hessian_vector_product_matches_gradient_differences() -> None:
    positions = _configuration(41, TRICLINIC, 4, 1.0)
    charges = _neutral(43, 4)
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    program, state, position = _bound(
        ParticleMeshEwaldPotential(1.2, 2.8, (24, 24, 24), spline_degree=5),
        charges,
        positions,
        cell=cell,
    )

    def gradient(value: Array, at: Any) -> Array:
        # ``at`` is the bundle state certified at the primal of ``value``.
        return jax.grad(lambda x: program.energy(x, at)[0])(value)

    def gradient_at(value: Array) -> Array:
        return gradient(value, bundle_state(program, value))

    direction = jnp.asarray(np.random.default_rng(47).normal(size=positions.shape))
    _, action = jax.jvp(lambda value: gradient(value, state), (position,), (direction,))
    step = 1.0e-5
    expected = (
        gradient_at(position + step * direction)
        - gradient_at(position - step * direction)
    ) / (2.0 * step)
    assert bool(jnp.all(jnp.isfinite(action)))
    np.testing.assert_allclose(action, expected, rtol=1.0e-5, atol=1.0e-7)


# --- finite-range pair methods ---------------------------------------------------


def test_dsf_matches_reference_with_complete_atom_energies() -> None:
    positions = np.random.default_rng(53).uniform(0.0, 5.0, (5, 3))
    charges = np.asarray([0.8, -0.3, -0.6, 0.4, -0.3])
    finite = _term_evaluation(
        DampedShiftedForceCoulombPotential(0.25, 8.0), charges, positions
    )
    np.testing.assert_allclose(
        finite.energy, dsf_energy(positions, charges, 0.25, 8.0), rtol=1.0e-12
    )
    np.testing.assert_allclose(jnp.sum(finite.atom_energy), finite.energy, rtol=1.0e-12)
    assert finite.atom_energy_complete
    vectors = 10.0 * np.eye(3)
    periodic_positions = np.random.default_rng(59).uniform(0.0, 10.0, (5, 3))
    periodic = _energy(
        DampedShiftedForceCoulombPotential(0.3, 4.5),
        charges,
        periodic_positions,
        cell=phx.discretization.PeriodicCell(vectors),
    )
    np.testing.assert_allclose(
        periodic,
        dsf_energy(periodic_positions, charges, 0.3, 4.5, vectors=vectors),
        rtol=1.0e-12,
    )


def test_dsf_energy_and_force_vanish_continuously_at_cutoff() -> None:
    cutoff = 3.0
    positions = np.asarray([[0.0, 0.0, 0.0], [cutoff - 1.0e-7, 0.0, 0.0]])
    program, state, position = _bound(
        DampedShiftedForceCoulombPotential(0.4, cutoff), [1.0, -1.0], positions
    )
    evaluation = program.evaluate(position, state)
    (evidence,) = program.terms[0].energy(program.context(position, state)).evidence
    np.testing.assert_allclose(evidence.pair_energy, 0.0, atol=1.0e-12)
    np.testing.assert_allclose(evaluation.forces, 0.0, atol=1.0e-6)


@pytest.mark.parametrize("truncation", ["hard", "shifted-potential", "shifted-force"])
def test_yukawa_screened_coulomb_matches_reference(
    truncation: ScreenedTruncation,
) -> None:
    positions = np.random.default_rng(61).uniform(0.0, 4.0, (4, 3))
    charges = np.asarray([1.0, -0.4, 0.7, -1.3])
    energy = _energy(
        ScreenedCoulombPotential(0.7, 6.0, truncation=truncation), charges, positions
    )
    np.testing.assert_allclose(
        energy, yukawa_energy(positions, charges, 0.7, 6.0, truncation), rtol=1.0e-12
    )


# --- reciprocal mesh owner -----------------------------------------------------------


def _mesh_case(
    shape: tuple[int, int, int], **kwargs: Any
) -> tuple[Any, Any, Any, np.ndarray]:
    positions = _configuration(67, TRICLINIC, 5, 0.8)
    system = _system(_neutral(71, 5), cell=phx.discretization.PeriodicCell(TRICLINIC))
    mesh = ReciprocalMeshPlan(shape, **kwargs).prepare(
        system.cell, system.particle_discretization()
    )
    geometry = mesh.geometry(jnp.asarray(TRICLINIC))
    routes = mesh.build(jnp.asarray(positions), geometry)
    return mesh, geometry, routes, positions


def _smooth_influence(wave: Array, squared: Array) -> Array:
    return jnp.exp(-squared / 6.0) * (1.0 + 0.3 * wave[..., 0] ** 2)


@pytest.mark.parametrize("nyquist", ["symmetrized", "excluded"])
def test_mesh_half_spectrum_equals_full_spectrum(nyquist: str) -> None:
    shape = (8, 9, 10)
    mesh, geometry, routes, _ = _mesh_case(shape, spline_degree=3, nyquist=nyquist)
    values = jnp.asarray(_neutral(73, 5))[:, None] + 0.1
    half = mesh.quadratic_energy(
        routes, values, geometry, _smooth_influence, zero_mode="include"
    )
    assert bool(half.successful)
    content = np.asarray(mesh.splat.deposit_content(routes.state, values).content[..., 0])
    full = np.fft.fftn(content)
    symbols = [cardinal_bspline_periodic_symbol(3, count) for count in shape]
    reciprocal = 2.0 * np.pi * np.linalg.inv(TRICLINIC).T
    total = 0.0
    for index in itertools.product(*(range(count) for count in shape)):
        choices = []
        nyquist_mode = False
        for axis, (j, count) in enumerate(zip(index, shape, strict=True)):
            if count % 2 == 0 and 2 * j == count:
                nyquist_mode = True
                choices.append((-count // 2, count // 2))
            else:
                choices.append((j if j <= count // 2 else j - count,))
        if nyquist_mode and nyquist == "excluded":
            continue
        influences = []
        for mode in itertools.product(*choices):
            wave = np.asarray(mode, dtype=np.float64) @ reciprocal
            influences.append(
                math.exp(-float(wave @ wave) / 6.0) * (1.0 + 0.3 * wave[0] ** 2)
            )
        window = symbols[0][index[0]] * symbols[1][index[1]] * symbols[2][index[2]]
        total += float(np.mean(influences)) * abs(full[index]) ** 2 / window**2
    np.testing.assert_allclose(half.value, total, rtol=1.0e-11)
    resources = mesh.resources
    if nyquist == "excluded":
        assert resources.excluded_modes > 0 and resources.symmetrized_modes == 0
    else:
        assert resources.excluded_modes == 0 and resources.symmetrized_modes > 0


def test_mesh_channel_chunks_densities_and_cross_forms_are_consistent() -> None:
    values = jnp.asarray(np.random.default_rng(79).normal(size=(5, 3)))
    weights = jnp.asarray([1.0, -0.5, 2.0])
    results = []
    for chunk in (1, 2, 4):
        mesh, geometry, routes, _ = _mesh_case((10, 10, 12), channel_chunk=chunk)
        results.append(
            mesh.quadratic_energy(
                routes,
                values,
                geometry,
                _smooth_influence,
                zero_mode="exclude",
                channel_weights=weights,
            ).value
        )
    np.testing.assert_allclose(results[1:], results[0], rtol=1.0e-12)
    density = mesh.transform(routes, values)
    resolved = mesh.resolve_influence(geometry, _smooth_influence)
    cross = mesh.cross_quadratic(
        density, density, resolved, zero_mode="exclude", channel_weights=weights
    )
    assert bool(cross.successful)
    np.testing.assert_allclose(cross.value, results[0], rtol=1.0e-12)
    left = mesh.transform(routes, values[:, :1])
    right = mesh.transform(routes, values[:, 1:2])
    mixed = mesh.cross_quadratic(left, right, resolved, zero_mode="exclude").value
    plus = mesh.quadratic_energy(
        routes,
        values[:, :1] + values[:, 1:2],
        geometry,
        _smooth_influence,
        zero_mode="exclude",
    ).value
    minus = mesh.quadratic_energy(
        routes,
        values[:, :1] - values[:, 1:2],
        geometry,
        _smooth_influence,
        zero_mode="exclude",
    ).value
    np.testing.assert_allclose(mixed, 0.25 * (plus - minus), rtol=1.0e-10)


def test_mesh_retained_density_bytes_are_admitted() -> None:
    mesh, _, routes, _ = _mesh_case(
        (8, 8, 8), budget=ReciprocalBudget(maximum_retained_bytes=100_000)
    )
    mesh.transform(routes, jnp.ones((5, 8)))
    with pytest.raises(ValueError, match="retained_bytes"):
        mesh.transform(routes, jnp.ones((5, 64)))


def test_mesh_structure_factor_improves_with_assignment_degree() -> None:
    errors = []
    for degree in (1, 3, 5):
        mesh, geometry, routes, positions = _mesh_case((24, 24, 24), spline_degree=degree)
        values = jnp.asarray(_neutral(83, 5))[:, None]
        density = mesh.transform(routes, values)
        selector = np.zeros(mesh.spectral_shape)
        selector[1, 2, 1] = 1.0
        power = mesh.cross_quadratic(
            density, density, jnp.asarray(selector), zero_mode="include"
        ).value / float(mesh.hermitian_weights[1, 2, 1])
        wave = np.asarray(geometry.wavevectors[1, 2, 1])
        exact = abs(structure_factor(positions, np.asarray(values[:, 0]), wave)) ** 2
        errors.append(abs(float(power) - exact) / exact)
        # Even-order centered B-splines have nonnegative alias coefficients.
        # Bound the omitted Fourier images independently, including the tail.
        order = degree + 1
        alias_bounds = []
        for ratio in np.asarray([1.0, 2.0, 1.0], dtype=np.float64) / 24.0:
            aliases = sum(
                abs(ratio / (ratio + image)) ** order for image in range(-64, 65) if image
            )
            tail = 2.0 * ratio**order / ((order - 1) * (64.0 - ratio) ** (order - 1))
            alias_bounds.append(aliases + tail)
        amplitude_bound = (
            2.0
            * np.sum(np.abs(np.asarray(values[:, 0])))
            * (1.0 - np.prod(1.0 / (1.0 + np.asarray(alias_bounds))))
        )
        power_bound = 2.0 * math.sqrt(exact) * amplitude_bound + amplitude_bound**2
        assert abs(float(power) - exact) <= power_bound + 1.0e-12
        if degree == 5:
            assert power_bound / exact < 1.0e-3
    assert errors[0] > errors[1] > errors[2]
    assert errors[2] < 1.0e-3


def test_mode_set_chunking_is_exact() -> None:
    positions = jnp.asarray(_configuration(89, TRICLINIC, 5, 0.8))
    values = jnp.asarray(np.random.default_rng(97).normal(size=(5, 2)))
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    results = []
    for chunk in (7, 4096):
        modes = ReciprocalModeSetPlan((4, 5, 3), mode_chunk=chunk).prepare(cell, 5)
        geometry = modes.geometry(jnp.asarray(TRICLINIC))
        results.append(
            modes.quadratic_energy(
                positions, values, geometry, _smooth_influence, zero_mode="exclude"
            ).value
        )
    np.testing.assert_allclose(results[1], results[0], rtol=1.0e-12)


def _reference_half_space(extents: tuple[int, int, int]) -> list[tuple[int, ...]]:
    """Lexicographic box modes whose first nonzero component is positive."""
    box = itertools.product(*(range(-extent, extent + 1) for extent in extents))
    return [mode for mode in box if next((m for m in mode if m), 0) > 0]


@pytest.mark.parametrize(
    ("extents", "zero_mode", "chunk"),
    [((1, 2, 3), True, 7), ((2, 1, 1), False, 4096), ((20, 20, 20), False, 128)],
    ids=["zero-mode-padded", "single-chunk", "multi-block-padded"],
)
def test_mode_set_stores_the_canonical_half_space_with_exact_padding(
    extents: tuple[int, int, int], zero_mode: bool, chunk: int
) -> None:
    plan = ReciprocalModeSetPlan(extents, include_zero_mode=zero_mode, mode_chunk=chunk)
    prepared = plan.prepare_lattice_family(3)
    rows = ([(0, 0, 0)] if zero_mode else []) + _reference_half_space(extents)
    stored = len(rows)
    width = min(chunk, stored)
    padded = -(-stored // width) * width
    modes = np.zeros((padded, 3), dtype=np.float64)
    modes[:stored] = rows
    weights = np.zeros((padded,), dtype=np.float64)
    weights[:stored] = 2.0
    weights[0] = 1.0 if zero_mode else 2.0
    zero = np.zeros((padded,), dtype=np.bool_)
    zero[0] = zero_mode
    np.testing.assert_array_equal(np.asarray(prepared.modes, dtype=np.float64), modes)
    np.testing.assert_array_equal(
        np.asarray(prepared.hermitian_weights, dtype=np.float64), weights
    )
    np.testing.assert_array_equal(np.asarray(prepared.zero_mode), zero)
    assert prepared.chunk_count == padded // width
    assert prepared.resources.channel_chunk == width
    assert prepared.resources.stored_modes == padded
    assert prepared.resources.supported_modes == stored
    prepared.validate()


def test_mode_set_validation_refuses_a_zero_flag_in_padding() -> None:
    plan = ReciprocalModeSetPlan((1, 2, 3), include_zero_mode=True, mode_chunk=7)
    prepared = plan.prepare_lattice_family(3)
    flagged_padding = eqx.tree_at(
        lambda value: value.zero_mode, prepared, prepared.zero_mode.at[-1].set(True)
    )
    with pytest.raises(ValueError):
        flagged_padding.validate()


def test_mode_set_validation_refuses_reordered_modes() -> None:
    prepared = ReciprocalModeSetPlan((1, 2, 3)).prepare_lattice_family(3)
    reordered = eqx.tree_at(lambda value: value.modes, prepared, prepared.modes[::-1])
    with pytest.raises(ValueError):
        reordered.validate()


def test_mode_set_validation_refuses_host_array_leaves() -> None:
    prepared = ReciprocalModeSetPlan((1, 2, 3)).prepare_lattice_family(3)
    host_backed = eqx.tree_at(
        lambda value: value.hermitian_weights,
        prepared,
        np.asarray(prepared.hermitian_weights),
    )
    with pytest.raises(TypeError):
        host_backed.validate()


@pytest.mark.parametrize("limit", [55, 56], ids=["one-row-short", "exact-capacity"])
def test_mode_set_budget_includes_chunk_padding(limit: int) -> None:
    plan = ReciprocalModeSetPlan(
        (1, 2, 3),
        include_zero_mode=True,
        mode_chunk=7,
        budget=ReciprocalBudget(maximum_modes=limit),
    )
    if limit == 55:
        with pytest.raises(ValueError):
            plan.prepare_lattice_family(2)
    else:
        prepared = plan.prepare_lattice_family(2)
        assert prepared.resources.supported_modes == 53
        assert prepared.resources.stored_modes == 56


def test_lattice_family_mode_set_matches_concrete_binding_and_batches() -> None:
    plan = ReciprocalModeSetPlan((4, 5, 3), include_zero_mode=True, mode_chunk=16)
    family = plan.prepare_lattice_family(5)
    concrete = plan.prepare(phx.discretization.PeriodicCell(TRICLINIC), 5)
    assert family.cell is None and family.scope == "lattice-family"
    assert concrete.scope == "concrete-cell"
    assert family.prepared_id != concrete.prepared_id
    assert family.resources.evidence_id == concrete.resources.evidence_id
    assert plan.prepare_lattice_family(5).prepared_id == family.prepared_id
    positions = jnp.asarray(_configuration(103, TRICLINIC, 5, 0.8))
    values = jnp.asarray(np.random.default_rng(107).normal(size=(5, 2)))
    expected = concrete.quadratic_energy(
        positions,
        values,
        concrete.geometry(jnp.asarray(TRICLINIC)),
        _smooth_influence,
        zero_mode="include",
    )
    geometry = family.geometry(jnp.asarray(TRICLINIC))
    np.testing.assert_allclose(
        geometry.volume, abs(np.linalg.det(TRICLINIC)), rtol=1.0e-12
    )
    observed = family.quadratic_energy(
        positions, values, geometry, _smooth_influence, zero_mode="include"
    )
    assert bool(observed.successful)
    np.testing.assert_allclose(observed.value, expected.value, rtol=1.0e-12)
    shear = np.asarray([[1.0, 0.2, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.5]])
    strained = jnp.asarray([TRICLINIC, TRICLINIC @ (np.eye(3) + 0.05 * shear).T])

    def case(vectors: Array) -> tuple[Array, Array, Array]:
        current = family.geometry(vectors)
        result = family.quadratic_energy(
            positions, values, current, _smooth_influence, zero_mode="exclude"
        )
        return result.value, current.support_radius, result.successful

    batched, support, successful = jax.vmap(case)(strained)
    assert bool(jnp.all(successful))
    for index in range(2):
        single = case(strained[index])
        np.testing.assert_allclose(batched[index], single[0], rtol=1.0e-12)
        np.testing.assert_allclose(support[index], single[1], rtol=1.0e-12)
    cell_gradient = jax.grad(lambda vectors: case(vectors)[0])(strained[0])
    assert bool(jnp.all(jnp.isfinite(cell_gradient)))


def test_lattice_family_reports_failed_lattice_solve() -> None:
    family = ReciprocalModeSetPlan((2, 2, 2)).prepare_lattice_family(2)
    singular = jnp.asarray([[4.0, 0.0, 0.0], [8.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
    geometry = family.geometry(singular)
    assert not bool(geometry.successful)
    result = family.quadratic_energy(
        jnp.zeros((2, 3)),
        jnp.ones((2, 1)),
        geometry,
        _smooth_influence,
        zero_mode="exclude",
    )
    assert not bool(result.successful)
    with pytest.raises(TypeError, match="prepare_lattice_family"):
        missing: Any = None
        ReciprocalModeSetPlan((2, 2, 2)).prepare(missing, 2)
    with pytest.raises(ValueError, match=r"\(3, 3\)"):
        family.geometry(jnp.eye(2))


# --- LJ-PME on the shared mesh -------------------------------------------------------


def test_lennard_jones_pme_matches_direct_lattice_dispersion() -> None:
    vectors = 6.0 * np.eye(3)
    cell = phx.discretization.PeriodicCell(vectors)
    positions = _configuration(101, vectors, 4, 1.5)
    charges = [0.0] * 4
    reference = dispersion_lattice_energy(
        positions, np.full((4, 4), 0.1), vectors, shells=10
    )
    errors = []
    for shape in (16, 32):
        term = phx.atomistic.LennardJonesPMEPotential(
            np.asarray([[0.1]]), 1.0, 2.9, (shape,) * 3
        )
        program, state, position = _bound(term, charges, positions, cell=cell)
        evaluation = program.evaluate(position, state)
        assert bool(evaluation.successful)
        np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=1.0e-10)
        errors.append(abs(float(evaluation.energy) - reference))
    assert errors[1] < errors[0]
    assert errors[1] < 2.0e-2 * abs(reference)


def test_reciprocal_mesh_layout_shared_and_foreign_material_refused() -> None:
    cell = phx.discretization.PeriodicCell(TRICLINIC)
    plan = ReciprocalMeshPlan((8, 8, 8), spline_degree=3)

    def sources(ids: Any) -> Any:
        return phx.discretization.ParticleSetPlan(
            np.asarray(ids, dtype=np.int64), np.ones((5,)), ambient_dimension=3
        ).prepare()

    first = plan.prepare(cell, sources((0, 1, 2, 3, 4)))
    second = plan.prepare(cell, sources((10, 3, 7, 1, 4)))
    # One executable layout, two material bindings.
    assert jax.tree.structure(first) == jax.tree.structure(second)
    for mesh in (first, second):
        with pytest.raises(ValueError, match="indexed-support schema"):
            mesh.splat.physical_particles()
    assert first.prepared_id != second.prepared_id
    positions = jnp.asarray(_configuration(67, TRICLINIC, 5, 0.8))
    values = jnp.asarray(_neutral(71, 5))[:, None]
    geometry = second.geometry(jnp.asarray(TRICLINIC))
    own = second.build(positions, geometry)
    foreign = first.build(positions, geometry)
    accepted = second.quadratic_energy(
        own, values, geometry, _smooth_influence, zero_mode="exclude"
    )
    refused = second.quadratic_energy(
        foreign, values, geometry, _smooth_influence, zero_mode="exclude"
    )
    assert bool(accepted.successful)
    assert not bool(refused.successful)
    assert not bool(refused.routes_successful)
    influence = second.resolve_influence(geometry, _smooth_influence)
    own_density = second.transform(own, values)
    foreign_density = first.transform(foreign, values)
    assert bool(
        second.cross_quadratic(
            own_density, own_density, influence, zero_mode="exclude"
        ).successful
    )
    assert not bool(
        second.cross_quadratic(
            foreign_density, own_density, influence, zero_mode="exclude"
        ).successful
    )
