"""FourierD3: signed reference factors, modified CN, radial kernel and mesh sum.

References come from ``tests/_support/d3_dispersion_reference.py``: an
independently formed direct lattice sum of the same definition (modified CN,
exact normalized-Gaussian C6), oscillatory radial quadrature and finite
differences. Tables are invented synthetic fixtures.
"""

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.atomistic._d3_parameters import (
    D3BJDamping,
    D3ReferenceTables,
    D3TableProvenance,
    FourierC6Decomposition,
    FourierD3CoordinationDefinition,
)
from phydrax.atomistic._dispersion_d3 import D3Routes
from phydrax.atomistic._dispersion_fourier_d3 import (
    damped_dispersion_fourier_kernel,
    FourierD3Potential,
    modified_count,
    SERIES_CROSSOVER,
    SHAPE6_SERIES,
    SHAPE8_SERIES,
)
from phydrax.atomistic._reciprocal_mesh import ReciprocalBudget
from phydrax.units import ANGSTROM, BOHR, HARTREE
from tests._support.d3_dispersion_reference import (
    BOHR_ANGSTROM,
    central_difference,
    coordination_numbers as reference_coordination,
    fourier_d3_direct_energy,
    image_routes,
    modified_count as reference_count,
    radial_fourier_transform,
    rank_error_curve,
    single_species_block,
    strained,
    synthetic_tables,
    SyntheticD3Tables,
)


UNITS = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
VECTORS = 5.0 * np.eye(3)
TRICLINIC = np.asarray([[5.0, 0.0, 0.0], [0.6, 4.8, 0.0], [0.4, 0.7, 5.2]])
POSITIONS = np.asarray([[0.3, 0.4, 0.2], [1.9, 1.1, 1.6], [3.4, 3.1, 3.6]])
NUMBERS = (6, 6, 1)
ROWS = np.asarray([1, 1, 0])
DAMPING = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}
CUTOFF = 4.0
EPSILON = 1.0e-6
MARGIN = 0.5


def _provenance() -> D3TableProvenance:
    return D3TableProvenance(
        "phydrax synthetic D3 test fixture",
        kind="synthetic-fixture",
        license_id="synthetic-test-data",
        use_admitted=True,
        redistribution_admitted=True,
        admission="invented values; qualifies no element",
    )


def _tables(data: SyntheticD3Tables | None = None) -> D3ReferenceTables:
    values = synthetic_tables() if data is None else data
    return D3ReferenceTables(
        values["species"],
        covalent_radii=values["covalent_radii"],
        r4r2=values["r4r2"],
        c6_reference=values["c6_reference"],
        reference_coordination=values["reference_coordination"],
        reference_valid=values["reference_valid"],
        length_unit=BOHR,
        energy_unit=HARTREE,
        covalent_radius_scale=4.0 / 3.0,
        provenance=_provenance(),
    )


def _single_species_tables() -> D3ReferenceTables:
    block = single_species_block()
    return _tables(
        {
            "species": (6,),
            "covalent_radii": np.asarray([1.9]),
            "r4r2": np.asarray([3.1]),
            "c6_reference": block.reshape(1, 1, 4, 4),
            "reference_coordination": np.asarray([[0.0, 1.0, 2.0, 3.0]]),
            "reference_valid": np.ones((1, 4), dtype=np.bool_),
        }
    )


def _damping() -> D3BJDamping:
    return D3BJDamping(
        a1=DAMPING["a1"],
        a2=DAMPING["a2"],
        s8=DAMPING["s8"],
        s6=DAMPING["s6"],
        functional="synthetic PBE-like BJ parameters",
    )


TABLES = _tables()
FULL = FourierC6Decomposition(TABLES, (1, 6), tolerance=1.0e-12)


def _system(
    numbers: tuple[int, ...] = NUMBERS,
    *,
    vectors: np.ndarray = VECTORS,
    cell: Any = "periodic",
    active: list[bool] | None = None,
) -> Any:
    count = len(numbers)
    box = phx.discretization.PeriodicCell(vectors) if cell == "periodic" else cell
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(count),
        np.asarray(numbers),
        np.asarray([1.0] * count),
        UNITS,
        cell=box,
        active_mask=None if active is None else np.asarray(active, dtype=np.bool_),
    ).prepare()


def _term(
    grid: tuple[int, int, int] = (32, 32, 32),
    *,
    degree: int = 5,
    decomposition: FourierC6Decomposition = FULL,
    tables: D3ReferenceTables = TABLES,
    **options: Any,
) -> FourierD3Potential:
    return FourierD3Potential(
        tables,
        decomposition,
        _damping(),
        grid,
        coordination_cutoff=CUTOFF,
        spline_degree=degree,
        **options,
    )


def _routes(
    positions: jax.Array,
    vectors: jax.Array,
    route_set: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> D3Routes:
    receivers, sources, shifts = route_set
    displacement = (
        positions[receivers]
        - positions[sources]
        + jnp.asarray(shifts, dtype=positions.dtype) @ vectors
    )
    counted = jax.lax.stop_gradient(
        jnp.sum(displacement * displacement, axis=-1) < CUTOFF * CUTOFF
    )
    return D3Routes(
        jnp.asarray(receivers),
        jnp.asarray(sources),
        displacement,
        counted,
        jnp.asarray(True),
        directed_multiplicity=1,
    )


def _evaluator(
    prepared: Any, positions: np.ndarray = POSITIONS, vectors: np.ndarray = VECTORS
) -> Any:
    route_set = image_routes(positions, vectors, CUTOFF + MARGIN)

    def evaluate(x: jax.Array, h: jax.Array) -> Any:
        return prepared.evaluate(x, h, _routes(x, h, route_set))

    return evaluate


def _direct(
    positions: np.ndarray = POSITIONS,
    vectors: np.ndarray = VECTORS,
    *,
    shells: int | None = None,
    radius: float = 50.0,
    width: float = 8.0,
    rows: np.ndarray = ROWS,
) -> float:
    return fourier_d3_direct_energy(
        positions,
        vectors,
        rows,
        synthetic_tables(),
        DAMPING,
        coordination_cutoff=CUTOFF,
        epsilon_bohr2=EPSILON,
        shells=shells,
        radius=None if shells is not None else radius,
        width=None if shells is not None else width,
    )


# --- signed reference decomposition -------------------------------------------


def test_signed_factors_reconstruct_the_reference_block_with_negative_eigenvalues() -> (
    None
):
    tables = _single_species_tables()
    decomposition = FourierC6Decomposition(tables, (6,), tolerance=1.0e-12)
    block = single_species_block()
    reference = np.linalg.eigvalsh(block)
    assert decomposition.rank == 4
    assert decomposition.negative_eigenvalues == int(np.count_nonzero(reference < 0.0))
    assert decomposition.negative_eigenvalues >= 2
    np.testing.assert_allclose(
        np.sort(np.asarray(decomposition.eigenvalues)), np.sort(reference), rtol=1.0e-12
    )
    factors = np.asarray(decomposition.factors)[0]
    approx = (factors * np.asarray(decomposition.eigenvalues)) @ factors.T
    np.testing.assert_allclose(approx, block, rtol=1.0e-12)
    assert decomposition.achieved_error < 1.0e-12
    assert decomposition.reconstruction_error(tables) == decomposition.achieved_error
    decomposition.validate_restored(tables)
    tampered = eqx.tree_at(
        lambda value: value.eigenvalues, decomposition, jnp.abs(decomposition.eigenvalues)
    )
    with pytest.raises(ValueError, match="achieved error"):
        tampered.validate_restored(tables)


def test_unpopulated_slots_are_exact_zero_factors_and_errors_are_of_stored_factors() -> (
    None
):
    decomposition = FULL
    factors = np.asarray(decomposition.factors)
    assert decomposition.species == (1, 6)
    np.testing.assert_array_equal(factors[0, 2], 0.0)
    assert decomposition.rank == 5
    assert decomposition.reconstruction_error(TABLES) == decomposition.achieved_error


def test_rank_selection_scans_a_nonmonotone_signed_error_curve() -> None:
    tables = _single_species_tables()
    curve = rank_error_curve(single_species_block())
    assert curve[1] > curve[0]
    loose = FourierC6Decomposition(tables, (6,), tolerance=0.5)
    np.testing.assert_allclose(loose.rank_errors, curve, rtol=1.0e-10, atol=1.0e-14)
    assert loose.rank == 1 and loose.meets_tolerance
    assert loose.rank_errors[1] > loose.tolerance
    tighter = FourierC6Decomposition(tables, (6,), tolerance=0.1)
    assert tighter.rank == 3
    assert tighter.negative_eigenvalues == 1
    assert tighter.achieved_error == pytest.approx(curve[2], rel=1.0e-10)


def test_capped_rank_that_misses_the_tolerance_is_refused() -> None:
    tables = _single_species_tables()
    capped = FourierC6Decomposition(tables, (6,), tolerance=1.0e-6, max_rank=2)
    assert not capped.meets_tolerance
    assert capped.rank == 2 and capped.max_rank == 2
    with pytest.raises(ValueError, match="misses tolerance"):
        capped.require_admitted()
    with pytest.raises(ValueError, match="misses tolerance"):
        FourierD3Potential(
            tables,
            capped,
            _damping(),
            (16, 16, 16),
            coordination_cutoff=CUTOFF,
        )
    with pytest.raises(ValueError, match="maximum_block_size"):
        FourierC6Decomposition(TABLES, (1, 6), tolerance=1.0e-6, maximum_block_size=5)


# --- radial kernel and counting function -----------------------------------------


def test_series_coefficients_match_the_inspected_source_constants() -> None:
    np.testing.assert_allclose(
        SHAPE6_SERIES[:7],
        [1.0, 0.0, -1.0 / 3.0, 0.125, -1.0 / 60.0, 0.0, 1.0 / 5040.0],
        rtol=1.0e-15,
        atol=1.0e-17,
    )
    np.testing.assert_allclose(
        SHAPE8_SERIES[:7],
        [
            -0.54119610014619668,
            0.0,
            0.090199350024366076,
            0.0,
            -0.010888024707303151,
            1.0 / 360.0,
            -0.00025923868350721731,
        ],
        rtol=1.0e-14,
        atol=1.0e-17,
    )


def test_radial_kernel_matches_independent_quadrature() -> None:
    radius, product, s6, s8 = 6.0, 6.2, 1.0, 0.7875

    def potential(r: float) -> float:
        return s6 / (r**6 + radius**6) + 3.0 * s8 * product / (r**8 + radius**8)

    zero = radial_fourier_transform(potential, 0.0)
    for wavenumber in (0.0, 0.02, 0.1, 0.16, 0.17, 0.3, 0.6, 1.2):
        value = float(
            damped_dispersion_fourier_kernel(
                jnp.asarray(wavenumber),
                jnp.asarray(radius),
                jnp.asarray(s6),
                jnp.asarray(s8),
                jnp.asarray(product),
            )
        )
        np.testing.assert_allclose(
            value,
            radial_fourier_transform(potential, wavenumber),
            rtol=1.0e-8,
            atol=1.0e-11 * abs(zero),
        )


def test_kernel_series_and_closed_form_join_at_roundoff() -> None:
    radius = jnp.asarray(5.0)

    def kernel(k: jax.Array) -> jax.Array:
        return damped_dispersion_fourier_kernel(
            k, radius, jnp.asarray(1.0), jnp.asarray(0.8), jnp.asarray(7.0)
        )

    join = SERIES_CROSSOVER / 5.0
    below, above = (
        jnp.asarray(join * (1.0 - 1.0e-13)),
        jnp.asarray(join * (1.0 + 1.0e-13)),
    )
    np.testing.assert_allclose(float(kernel(below)), float(kernel(above)), rtol=1.0e-12)
    np.testing.assert_allclose(
        float(jax.grad(kernel)(below)), float(jax.grad(kernel)(above)), rtol=1.0e-10
    )
    np.testing.assert_allclose(
        float(jax.grad(jax.grad(kernel))(below)),
        float(jax.grad(jax.grad(kernel))(above)),
        rtol=1.0e-8,
    )
    assert np.isfinite(float(jax.grad(kernel)(jnp.asarray(0.0))))


def test_modified_count_matches_the_source_definition_and_its_regularity() -> None:
    covalent, cutoff, scale = 3.8, CUTOFF / BOHR_ANGSTROM, 4.0 / 3.0

    def count(r: jax.Array, epsilon: float = EPSILON) -> jax.Array:
        return modified_count(r, jnp.asarray(covalent), cutoff, scale, 16.0, epsilon)

    for distance in (2.0, 3.8, 4.9, 5.4, 6.3, 7.0, 7.5):
        np.testing.assert_allclose(
            float(count(jnp.asarray(distance))),
            reference_count(distance, covalent, cutoff, EPSILON),
            rtol=1.0e-12,
            atol=1.0e-300,
        )
    transition = 0.5 * (covalent / scale + cutoff)
    first = jax.grad(count)
    second = jax.grad(first)
    left, right = jnp.asarray(transition - 1.0e-7), jnp.asarray(transition + 1.0e-7)
    np.testing.assert_allclose(float(first(left)), float(first(right)), rtol=1.0e-5)
    u = covalent / transition - 1.0
    sigma = 1.0 / (1.0 + math.exp(-16.0 * u))
    predicted = sigma * (1.0 - sigma) * u * 2.0 / ((cutoff - transition) ** 2 + EPSILON)
    np.testing.assert_allclose(
        float(second(right) - second(left)), predicted, rtol=1.0e-4
    )
    large = 10.0
    inside = reference_count(cutoff * (1.0 - 1.0e-13), covalent, cutoff, large)
    steep = 16.0 + (cutoff - transition) ** 2 / large
    expected = 1.0 / (1.0 + math.exp(-steep * (covalent / cutoff - 1.0)))
    np.testing.assert_allclose(inside, expected, rtol=1.0e-6)
    assert inside > 0.0
    assert reference_count(cutoff, covalent, cutoff, large) == 0.0


def test_epsilon_is_a_dimensionful_area_converted_explicitly() -> None:
    bohr = FourierD3CoordinationDefinition()
    angstrom = FourierD3CoordinationDefinition(
        epsilon=1.0e-6, epsilon_length_unit=ANGSTROM
    )
    assert bohr.epsilon_in(BOHR) == pytest.approx(1.0e-6)
    assert angstrom.epsilon_in(BOHR) == pytest.approx(
        1.0e-6 / BOHR_ANGSTROM**2, rel=1.0e-12
    )
    assert bohr.definition_id != angstrom.definition_id


def test_regularity_reports_the_transition_and_the_nonzero_cutoff_step() -> None:
    prepared = _term((16, 16, 16), degree=3).prepare(_system())
    cutoff = CUTOFF / BOHR_ANGSTROM
    radii = synthetic_tables()["covalent_radii"]
    logs = []
    jumps = []
    for a in range(2):
        for b in range(2):
            covalent = radii[a] + radii[b]
            transition = 0.5 * (covalent * 0.75 + cutoff)
            steep = 16.0 + (cutoff - transition) ** 2 / EPSILON
            logs.append(steep * (covalent / cutoff - 1.0))
            jumps.append(2.0 / ((cutoff - transition) ** 2 + EPSILON))
    regularity = prepared.regularity
    np.testing.assert_allclose(regularity.log_cutoff_step, max(logs), rtol=1.0e-12)
    assert math.isfinite(regularity.log_cutoff_step) and regularity.log_cutoff_step < 0.0
    np.testing.assert_allclose(
        regularity.transition_curvature_jump, max(jumps), rtol=1.0e-12
    )
    assert regularity.mesh_coordinate_continuity == 2


# --- coordination admission, species and resources ----------------------------------


def test_small_cell_modified_coordination_counts_self_images() -> None:
    vectors = 3.0 * np.eye(3)
    positions = np.asarray([[0.2, 0.3, 0.1], [1.6, 1.4, 1.5], [2.5, 0.4, 2.2]])
    prepared = _term((8, 8, 8), degree=3).prepare(_system(vectors=vectors))
    route_set = image_routes(positions, vectors, CUTOFF + MARGIN)
    x, h = jnp.asarray(positions), jnp.asarray(vectors)
    numbers, status, geometry = prepared.coordination_numbers(_routes(x, h, route_set))
    cutoff = CUTOFF / BOHR_ANGSTROM
    expected = reference_coordination(
        positions / BOHR_ANGSTROM,
        vectors / BOHR_ANGSTROM,
        ROWS,
        synthetic_tables()["covalent_radii"],
        cutoff,
        lambda r, covalent: reference_count(r, covalent, cutoff, EPSILON),
    )
    assert bool(status) and bool(geometry)
    assert np.any(route_set[0] == route_set[1])
    np.testing.assert_allclose(np.asarray(numbers), expected, rtol=1.0e-12)
    with pytest.raises(ValueError, match="full directed"):
        directed = _routes(x, h, route_set)
        prepared.coordination_numbers(
            D3Routes(
                directed.receivers,
                directed.sources,
                directed.displacement,
                directed.counted,
                directed.successful,
                directed_multiplicity=2,
            )
        )


def test_species_coverage_geometry_and_padding_admission() -> None:
    term = _term((16, 16, 16), degree=3)
    with pytest.raises(ValueError, match=r"cover active atomic numbers \[8\]"):
        term.prepare(_system((6, 6, 8)))
    carbon = FourierC6Decomposition(TABLES, (6,), tolerance=1.0e-12)
    with pytest.raises(ValueError, match="decomposition does not cover"):
        _term((16, 16, 16), degree=3, decomposition=carbon).prepare(_system())
    with pytest.raises(ValueError, match="fully periodic"):
        term.prepare(_system(cell=None))
    with pytest.raises(ValueError, match="fully periodic"):
        term.prepare(
            _system(
                cell=phx.discretization.PeriodicCell(
                    VECTORS, periodic_axes=(True, True, False)
                )
            )
        )
    full = term.prepare(_system())
    padded = term.prepare(_system((6, 6, 1, 0), active=[True, True, True, False]))
    positions = np.vstack([POSITIONS, [[4.0, 0.5, 2.5]]])
    route_set = image_routes(POSITIONS, VECTORS, CUTOFF + MARGIN)
    x, h = jnp.asarray(positions), jnp.asarray(VECTORS)
    result = padded.evaluate(x, h, _routes(x, h, route_set))
    reference = _evaluator(full)(jnp.asarray(POSITIONS), h)
    assert bool(result.successful)
    np.testing.assert_allclose(
        float(result.energy), float(reference.energy), rtol=1.0e-12
    )
    coefficients = padded.coefficients(result.evidence.coordination_numbers)
    np.testing.assert_array_equal(np.asarray(coefficients[3]), 0.0)


def test_rank_chunks_and_residency_refusals() -> None:
    small = _term((16, 16, 16), degree=3, rank_chunk=2).prepare(_system())
    whole = _term((16, 16, 16), degree=3).prepare(_system())
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)
    chunked = _evaluator(small)(x, h)
    resident = _evaluator(whole)(x, h)
    np.testing.assert_allclose(
        float(chunked.energy), float(resident.energy), rtol=1.0e-12
    )
    resources = small.resources
    spectral = 16 * 16 * 9
    assert resources.padded_rank == 6 and resources.rank_chunks == 3
    assert resources.density_bytes == 2 * 2 * spectral * 16
    assert resources.coefficient_bytes == 2 * 3 * 6 * 8
    assert whole.resources.density_bytes == 2 * 5 * spectral * 16
    budget = ReciprocalBudget(maximum_retained_bytes=100 * spectral)
    with pytest.raises(ValueError, match="retained_bytes"):
        _term((16, 16, 16), degree=3, nyquist="excluded", budget=budget).prepare(
            _system()
        )
    admitted = _term(
        (16, 16, 16), degree=3, nyquist="excluded", budget=budget, rank_chunk=1
    ).prepare(_system())
    assert admitted.resources.retained_bytes <= 100 * spectral
    with pytest.raises(ValueError, match="exceeds the decomposition rank"):
        _term((16, 16, 16), rank_chunk=6)
    with pytest.raises(ValueError, match="positive"):
        _term((16, 16, 16), rank_chunk=0)


# --- same-definition lattice sums, forces and virial ---------------------------------


def test_mesh_energy_matches_the_same_definition_direct_lattice_sum() -> None:
    prepared = _term((32, 32, 32), degree=5).prepare(_system())
    result = _evaluator(prepared)(jnp.asarray(POSITIONS), jnp.asarray(VECTORS))
    assert bool(result.successful)
    reference = _direct()
    # A second, independent partition (radius, width) whose neglected
    # reciprocal-lattice terms are bounded far below 1e-12 relative.
    converged = _direct(radius=40.0, width=7.0)
    assert abs(reference - converged) < 1.0e-12 * abs(reference)
    np.testing.assert_allclose(float(result.energy), reference, rtol=1.0e-5)
    evidence = result.evidence
    assert float(evidence.self_energy) > 0.0
    assert evidence.rank == 5 and evidence.coefficient_residual < 1.0e-12


def test_triclinic_mesh_energy_matches_the_direct_lattice_sum() -> None:
    positions = POSITIONS @ np.linalg.inv(VECTORS) @ TRICLINIC
    prepared = _term((32, 32, 36), degree=5).prepare(_system(vectors=TRICLINIC))
    result = _evaluator(prepared, positions, TRICLINIC)(
        jnp.asarray(positions), jnp.asarray(TRICLINIC)
    )
    np.testing.assert_allclose(
        float(result.energy), _direct(positions, TRICLINIC), rtol=1.0e-5
    )


def test_mesh_refinement_and_translation_artifacts_decrease() -> None:
    reference = _direct()
    errors, drifts = [], []
    shift = np.asarray([0.37, -0.21, 0.13])
    for size in (10, 20):
        prepared = _term((size, size, size), degree=3).prepare(_system())
        evaluate = _evaluator(prepared)
        base = float(evaluate(jnp.asarray(POSITIONS), jnp.asarray(VECTORS)).energy)
        moved = float(
            _evaluator(prepared, POSITIONS + shift)(
                jnp.asarray(POSITIONS + shift), jnp.asarray(VECTORS)
            ).energy
        )
        errors.append(abs(base - reference))
        drifts.append(abs(moved - base))
    assert errors[1] < errors[0]
    assert drifts[1] < drifts[0]
    assert drifts[1] < 1.0e-3 * abs(reference)
    anchored = _term(
        (10, 10, 10), degree=3, frame="anchor-source", anchor_index=0
    ).prepare(_system())
    base = float(
        _evaluator(anchored)(jnp.asarray(POSITIONS), jnp.asarray(VECTORS)).energy
    )
    moved = float(
        _evaluator(anchored, POSITIONS + shift)(
            jnp.asarray(POSITIONS + shift), jnp.asarray(VECTORS)
        ).energy
    )
    np.testing.assert_allclose(moved, base, rtol=1.0e-12)


def test_total_frozen_cn_and_self_force_terms() -> None:
    prepared = _term((32, 32, 32), degree=5).prepare(_system())
    route_set = image_routes(POSITIONS, VECTORS, CUTOFF + MARGIN)
    h = jnp.asarray(VECTORS)
    x = jnp.asarray(POSITIONS)

    def coordination(y: jax.Array) -> jax.Array:
        return prepared.coordination_numbers(_routes(y, h, route_set))[0]

    def at(y: jax.Array, cn: jax.Array) -> jax.Array:
        parts = prepared.parts(y, h, cn)
        return parts.reciprocal_energy + parts.self_energy

    total = jax.grad(lambda y: prepared.evaluate(y, h, _routes(y, h, route_set)).energy)(
        x
    )
    numbers = coordination(x)
    frozen = jax.grad(lambda y: at(y, numbers))(x)
    sensitivity = jax.grad(lambda cn: at(x, cn))(numbers)
    _, pullback = jax.vjp(coordination, x)
    (chain,) = pullback(sensitivity)
    np.testing.assert_allclose(
        np.asarray(total), np.asarray(frozen + chain), rtol=1.0e-10, atol=1.0e-12
    )
    assert float(jnp.max(jnp.abs(chain))) > 1.0e-6 * float(jnp.max(jnp.abs(total)))
    self_sensitivity = jax.grad(lambda cn: prepared.parts(x, h, cn).self_energy)(numbers)
    (self_chain,) = pullback(self_sensitivity)
    assert float(jnp.max(jnp.abs(self_chain))) > 1.0e-3 * float(jnp.max(jnp.abs(chain)))
    expected = central_difference(lambda y: _direct(y, shells=6), POSITIONS, 1.0e-5)
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(
        np.asarray(total), expected, rtol=1.0e-4, atol=1.0e-4 * scale
    )
    # The cell-origin mesh frame is translation invariant only to mesh accuracy.
    np.testing.assert_allclose(
        np.asarray(jnp.sum(total, axis=0)), 0.0, atol=1.0e-4 * scale
    )


def test_finite_strain_virial_matches_differences() -> None:
    prepared = _term((32, 32, 32), degree=5).prepare(_system(vectors=TRICLINIC))
    positions = POSITIONS @ np.linalg.inv(VECTORS) @ TRICLINIC
    evaluate = _evaluator(prepared, positions, TRICLINIC)

    def energy(strain: jax.Array) -> jax.Array:
        deformation = jnp.eye(3) + strain
        return evaluate(
            jnp.asarray(positions) @ deformation.T, jnp.asarray(TRICLINIC) @ deformation.T
        ).energy

    gradient = np.asarray(jax.grad(energy)(jnp.zeros((3, 3))))
    own = central_difference(
        lambda e: float(energy(jnp.asarray(e.reshape(3, 3)))), np.zeros(9), 1.0e-6
    ).reshape(3, 3)
    np.testing.assert_allclose(gradient, own, rtol=1.0e-6, atol=1.0e-9)
    # Every count switch and route admission is far from its threshold, so the
    # finite-difference nodes (|strain| = 1e-6, distances move < 1e-5 angstrom)
    # see the same counted routes and the same route set as the base point.
    receivers, sources, shifts = image_routes(positions, TRICLINIC, CUTOFF + 2.0 * MARGIN)
    distances = np.linalg.norm(
        positions[receivers] - positions[sources] + shifts @ TRICLINIC, axis=1
    )
    assert np.min(np.abs(distances - CUTOFF)) > 1.0e-4
    assert np.min(np.abs(distances - (CUTOFF + MARGIN))) > 1.0e-4

    def stress(radius: float, width: float) -> np.ndarray:
        # The full infinite lattice (partitioned near sum plus continuum tail):
        # a fixed shell set would drop the long-range volume response of the
        # r^-6 tail, about 7e-4 of the diagonal for |n_i| <= 6.
        return central_difference(
            lambda e: _direct(
                strained(positions, e.reshape(3, 3)),
                strained(TRICLINIC, e.reshape(3, 3)),
                radius=radius,
                width=width,
            ),
            np.zeros(9),
            1.0e-6,
        ).reshape(3, 3)

    reference = stress(50.0, 8.0)
    scale = float(np.max(np.abs(reference)))
    # An independent partition agrees to finite-difference roundoff.
    np.testing.assert_allclose(
        stress(40.0, 7.0), reference, rtol=0.0, atol=1.0e-8 * scale
    )
    np.testing.assert_allclose(gradient, reference, rtol=1.0e-4, atol=1.0e-4 * scale)


def test_damping_parameter_derivatives_reach_kernel_and_self_energy() -> None:
    prepared = _term((24, 24, 24), degree=5).prepare(_system())
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)

    route_set = image_routes(POSITIONS, VECTORS, CUTOFF + MARGIN)
    # Differentiate the float damping leaves; ``damping_identity`` (uint32
    # IdentityWords) stays the prepared identity and is not a parameter.

    def with_damping(values: tuple[jax.Array, ...]) -> jax.Array:
        term = eqx.tree_at(
            lambda value: (
                value.plan.damping.a1,
                value.plan.damping.a2,
                value.plan.damping.s8,
            ),
            prepared,
            values,
        )
        return term.evaluate(x, h, _routes(x, h, route_set)).energy

    damping = prepared.plan.damping
    base = (damping.a1, damping.a2, damping.s8)
    gradient = jax.grad(with_damping)(base)
    for index, derivative in enumerate(gradient):
        step = 1.0e-6

        def shifted(sign: float) -> float:
            values = tuple(
                value + sign * step if slot == index else value
                for slot, value in enumerate(base)
            )
            return float(with_damping(values))

        expected = (shifted(1.0) - shifted(-1.0)) / (2.0 * step)
        np.testing.assert_allclose(float(derivative), expected, rtol=1.0e-6)
