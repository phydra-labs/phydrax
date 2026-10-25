"""Conventional two-body D3(BJ): tables, counting, energies and derivatives.

References come from ``tests/_support/d3_dispersion_reference.py`` (independent
NumPy lattice sums, explicit interpolation loops, finite differences). Tables
are invented synthetic fixtures: they exercise the definition, never the
fidelity of any element.
"""

import hashlib
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
    D3CoordinationPolicy,
    D3InterpolationRule,
    D3ReferenceTables,
    D3TableProvenance,
)
from phydrax.atomistic._dispersion_d3 import (
    D3BJPotential,
    D3DispersionCutoff,
    D3Routes,
    s5_switch,
)
from phydrax.units import BOHR, HARTREE
from tests._support.d3_dispersion_reference import (
    BOHR_ANGSTROM,
    canonical_half,
    central_difference,
    conventional_count,
    d3_energy,
    image_routes,
    s5_switch as reference_s5,
    strained,
    synthetic_tables,
    SyntheticD3Tables,
)


UNITS = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
VECTORS = np.asarray([[3.6, 0.0, 0.0], [0.4, 3.4, 0.0], [0.3, 0.5, 3.8]])
POSITIONS = np.asarray([[0.2, 0.1, 0.3], [1.7, 1.4, 1.2], [2.9, 2.6, 2.95]])
NUMBERS = (6, 6, 1)
ROWS = np.asarray([1, 1, 0])
DAMPING = {"s6": 1.0, "s8": 0.7875, "a1": 0.4289, "a2": 4.4407}
COORDINATION_CUTOFF = 5.0
DISPERSION_CUTOFF = 8.0
MARGIN = 0.5


def _provenance(*, use: bool = True) -> D3TableProvenance:
    return D3TableProvenance(
        "phydrax synthetic D3 test fixture",
        kind="synthetic-fixture",
        license_id="synthetic-test-data",
        use_admitted=use,
        redistribution_admitted=True,
        admission="invented values; qualifies no element",
    )


def _tables(
    data: SyntheticD3Tables | None = None, *, use: bool = True
) -> D3ReferenceTables:
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
        provenance=_provenance(use=use),
    )


def _damping(**overrides: float) -> D3BJDamping:
    values = {**DAMPING, **overrides}
    return D3BJDamping(
        a1=values["a1"],
        a2=values["a2"],
        s8=values["s8"],
        s6=values["s6"],
        functional="synthetic PBE-like BJ parameters",
    )


def _system(
    numbers: tuple[int, ...] = NUMBERS,
    *,
    cell: Any = "periodic",
    active: list[bool] | None = None,
) -> Any:
    count = len(numbers)
    box = phx.discretization.PeriodicCell(VECTORS) if cell == "periodic" else cell
    return phx.atomistic.AtomisticSystemPlan(
        np.arange(count),
        np.asarray(numbers),
        np.asarray([1.0] * count),
        UNITS,
        cell=box,
        active_mask=None if active is None else np.asarray(active, dtype=np.bool_),
    ).prepare()


def _term(
    interpolation: D3InterpolationRule = "normalized-gaussian",
    *,
    switch: tuple[float, float] | None = None,
    tables: D3ReferenceTables | None = None,
    damping: D3BJDamping | None = None,
    dispersion_cutoff: float = DISPERSION_CUTOFF,
    route_chunk: int = 64,
) -> D3BJPotential:
    cutoff = (
        D3DispersionCutoff(dispersion_cutoff)
        if switch is None
        else D3DispersionCutoff(
            dispersion_cutoff, switch_on=switch[0], switch_off=switch[1]
        )
    )
    return D3BJPotential(
        _tables() if tables is None else tables,
        _damping() if damping is None else damping,
        D3CoordinationPolicy(interpolation),
        cutoff,
        coordination_cutoff=COORDINATION_CUTOFF,
        route_chunk=route_chunk,
    )


def _route_set(
    positions: np.ndarray, vectors: np.ndarray | None, radius: float, *, half: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    routes = image_routes(positions, vectors, radius + MARGIN)
    return canonical_half(*routes) if half else routes


def _routes(
    positions: jax.Array,
    vectors: jax.Array,
    route_set: tuple[np.ndarray, np.ndarray, np.ndarray],
    radius: float,
    multiplicity: int,
) -> D3Routes:
    receivers, sources, shifts = route_set
    displacement = (
        positions[receivers]
        - positions[sources]
        + jnp.asarray(shifts, dtype=positions.dtype) @ vectors
    )
    counted = jax.lax.stop_gradient(
        jnp.sum(displacement * displacement, axis=-1) < radius * radius
    )
    return D3Routes(
        jnp.asarray(receivers),
        jnp.asarray(sources),
        displacement,
        counted,
        jnp.asarray(True),
        directed_multiplicity=multiplicity,
    )


def _energy_function(
    prepared: Any,
    positions: np.ndarray = POSITIONS,
    vectors: np.ndarray | None = VECTORS,
    *,
    half: bool = True,
) -> Any:
    coordination = _route_set(positions, vectors, COORDINATION_CUTOFF, half=False)
    dispersion = _route_set(
        positions, vectors, prepared.plan.dispersion.cutoff, half=half
    )

    def evaluate(x: jax.Array, h: jax.Array, term: Any = prepared) -> Any:
        return term.evaluate(
            _routes(x, h, coordination, COORDINATION_CUTOFF, 1),
            _routes(x, h, dispersion, term.plan.dispersion.cutoff, 2 if half else 1),
        )

    return evaluate


def _reference(
    positions: np.ndarray = POSITIONS,
    vectors: np.ndarray | None = VECTORS,
    *,
    tables: SyntheticD3Tables | None = None,
    damping: dict | None = None,
    switch: tuple[float, float] | None = None,
    threshold: float | None = None,
    dispersion_cutoff: float = DISPERSION_CUTOFF,
    rows: np.ndarray = ROWS,
) -> float:
    return d3_energy(
        positions,
        vectors,
        rows,
        synthetic_tables() if tables is None else tables,
        DAMPING if damping is None else damping,
        coordination_cutoff=COORDINATION_CUTOFF,
        dispersion_cutoff=dispersion_cutoff,
        switch=switch,
        threshold=threshold,
    )


# --- authenticated tables -----------------------------------------------------


def test_tables_refuse_asymmetric_incomplete_or_mislabeled_reference_blocks() -> None:
    base = synthetic_tables()
    asymmetric = base.copy()
    asymmetric["c6_reference"] = base["c6_reference"].copy()
    asymmetric["c6_reference"][0, 1, 0, 2] += 1.0
    with pytest.raises(ValueError, match="symmetric"):
        _tables(asymmetric)
    hole = base.copy()
    hole["c6_reference"] = base["c6_reference"].copy()
    hole["c6_reference"][1, 1, 0, 2] = hole["c6_reference"][1, 1, 2, 0] = 0.0
    with pytest.raises(ValueError, match="positive C6"):
        _tables(hole)
    stray = base.copy()
    stray["c6_reference"] = base["c6_reference"].copy()
    stray["c6_reference"][0, 0, 2, 2] = 1.0
    with pytest.raises(ValueError, match="zero C6"):
        _tables(stray)
    sentinel = base.copy()
    sentinel["reference_coordination"] = base["reference_coordination"].copy()
    sentinel["reference_coordination"][0, 2] = -1.0
    with pytest.raises(ValueError, match="zero coordination"):
        _tables(sentinel)
    with pytest.raises(ValueError, match="digests"):
        D3TableProvenance(
            "Grimme dftd3",
            kind="grimme-dftd3-reference",
            license_id="GPL-1.0-or-later",
            use_admitted=True,
            redistribution_admitted=False,
            admission="caller",
        )


def test_tables_map_species_and_refuse_uncovered_elements() -> None:
    tables = _tables()
    np.testing.assert_array_equal(tables.rows(np.asarray([6, 1, 6])), [1, 0, 1])
    assert tables.uncovered(np.asarray([1, 8, 6, 26])) == (8, 26)
    with pytest.raises(ValueError, match=r"\[8\]"):
        tables.rows(np.asarray([1, 8]))
    expected = np.full((119,), -1, dtype=np.int32)
    expected[1], expected[6] = 0, 1
    np.testing.assert_array_equal(np.asarray(tables.species_lookup), expected)


def test_restored_tables_are_revalidated_against_their_identity() -> None:
    tables = _tables()
    tables.validate_restored()
    scaled = eqx.tree_at(
        lambda value: value.c6_reference, tables, tables.c6_reference * 1.01
    )
    with pytest.raises(ValueError, match="identity"):
        scaled.validate_restored()
    broken = np.asarray(tables.c6_reference).copy()
    broken[0, 1, 0, 0] += 1.0
    tampered = eqx.tree_at(lambda value: value.c6_reference, tables, jnp.asarray(broken))
    with pytest.raises(ValueError, match="symmetric"):
        tampered.validate_restored()


def test_unadmitted_table_rights_refuse_preparation() -> None:
    term = _term(tables=_tables(use=False))
    with pytest.raises(ValueError, match="use right"):
        term.prepare(_system())


def _fortran_values(name: str, values: np.ndarray) -> str:
    lines = [f"      data {name} /"]
    chunks = [values[index : index + 5] for index in range(0, values.size, 5)]
    for number, chunk in enumerate(chunks):
        text = ", ".join(f"{value:.8f}" for value in chunk)
        tail = " /" if number == len(chunks) - 1 else ","
        lines.append(f"     . {text}{tail}")
    return "\n".join(lines)


def _grimme_sources(
    tables: SyntheticD3Tables, *, conflict: bool = False
) -> tuple[str, str]:
    species = list(tables["species"]) + [8]
    rcov = np.full(94, 2.0)
    r2r4 = np.full(94, 1.5)
    for row, number in enumerate(tables["species"]):
        rcov[number - 1] = tables["covalent_radii"][row]
        r2r4[number - 1] = tables["r4r2"][row]
    dftd3 = "\n".join(
        [
            "      program synthetic",
            "c synthetic fixture, not Grimme data",
            "!      data rcov/",
            "!     .  9.9, 9.9",
            _fortran_values("r2r4", r2r4),
            _fortran_values("rcov", rcov),
            "      end",
        ]
    )
    slots = {1: [0.0, 0.9], 6: [0.0, 1.1, 2.4], 8: [0.0]}
    records = []
    for i, z_i in enumerate(species):
        for j, z_j in enumerate(species[: i + 1]):
            for p, cn_p in enumerate(slots[z_i]):
                for q, cn_q in enumerate(slots[z_j]):
                    if z_i in tables["species"] and z_j in tables["species"]:
                        a = tables["species"].index(z_i)
                        b = tables["species"].index(z_j)
                        value = tables["c6_reference"][a, b, p, q]
                    else:
                        value = 7.5
                    if conflict and z_i == 8 and z_j == 6 and q == 1:
                        cn_q = 1.2
                    records.append((value, z_i + 100 * p, z_j + 100 * q, cn_p, cn_q))
    body = []
    for index, record in enumerate(records):
        text = ",".join(f"{value:.8E}".replace("E", "D") for value in record)
        body.append(("     . " if index == 0 else "     .,") + text)
    pars = "\n".join(
        [
            f"      real*8 pars({5 * len(records):8d})",
            f"      nlines={len(records):8d}",
            f"      pars(      1:{5 * len(records):7d})=(/",
            *body,
            "     ./)",
        ]
    )
    return dftd3, pars


def _grimme_provenance(
    dftd3: str, pars: str, *, digest: str | None = None
) -> D3TableProvenance:
    return D3TableProvenance(
        "synthetic Fortran-layout fixture",
        kind="grimme-dftd3-reference",
        license_id="synthetic-test-data",
        use_admitted=True,
        redistribution_admitted=False,
        admission="test fixture in Grimme's file layout",
        digests=(
            ("dftd3.f", digest or hashlib.sha256(dftd3.encode()).hexdigest()),
            ("pars.f", hashlib.sha256(pars.encode()).hexdigest()),
        ),
    )


def test_grimme_fortran_layout_is_parsed_with_verified_digests(tmp_path: Any) -> None:
    data = synthetic_tables()
    dftd3, pars = _grimme_sources(data)
    (tmp_path / "dftd3.f").write_text(dftd3)
    (tmp_path / "pars.f").write_text(pars)
    tables = D3ReferenceTables.from_grimme_fortran(
        tmp_path / "dftd3.f",
        tmp_path / "pars.f",
        species=(1, 6),
        length_unit=BOHR,
        energy_unit=HARTREE,
        provenance=_grimme_provenance(dftd3, pars),
    )
    assert tables.covalent_radius_scale == pytest.approx(4.0 / 3.0)
    np.testing.assert_array_equal(
        np.asarray(tables.covalent_radii), data["covalent_radii"]
    )
    np.testing.assert_array_equal(np.asarray(tables.r4r2), data["r4r2"])
    np.testing.assert_array_equal(
        np.asarray(tables.reference_valid)[:, :3], data["reference_valid"]
    )
    assert not np.any(np.asarray(tables.reference_valid)[:, 3:])
    np.testing.assert_array_equal(
        np.asarray(tables.c6_reference)[:, :, :3, :3], data["c6_reference"]
    )
    np.testing.assert_array_equal(
        np.asarray(tables.reference_coordination)[:, :3], data["reference_coordination"]
    )
    with pytest.raises(ValueError, match="SHA-256"):
        D3ReferenceTables.from_grimme_fortran(
            tmp_path / "dftd3.f",
            tmp_path / "pars.f",
            species=(1, 6),
            length_unit=BOHR,
            energy_unit=HARTREE,
            provenance=_grimme_provenance(dftd3, pars, digest="0" * 64),
        )
    dftd3_bad, pars_bad = _grimme_sources(data, conflict=True)
    (tmp_path / "pars.f").write_text(pars_bad)
    with pytest.raises(ValueError, match="differs"):
        D3ReferenceTables.from_grimme_fortran(
            tmp_path / "dftd3.f",
            tmp_path / "pars.f",
            species=(1, 6),
            length_unit=BOHR,
            energy_unit=HARTREE,
            provenance=_grimme_provenance(dftd3_bad, pars_bad),
        )


# --- species coverage, padding and route admission ---------------------------


def test_uncovered_active_species_refuse_and_inactive_slots_are_padding() -> None:
    term = _term()
    with pytest.raises(ValueError, match=r"cover active atomic numbers \[8\]"):
        term.prepare(_system((6, 6, 1, 8)))
    prepared = term.prepare(_system((6, 6, 1, 0), active=[True, True, True, False]))
    positions = np.vstack([POSITIONS, [[1.0, 2.5, 0.4]]])
    result = _energy_function(prepared, positions)(
        jnp.asarray(positions), jnp.asarray(VECTORS)
    )
    assert not bool(result.successful)
    assert not bool(result.evidence.endpoints_successful)
    padded = _term().prepare(_system((6, 6, 1, 0), active=[True, True, True, False]))
    coordination = _route_set(POSITIONS, VECTORS, COORDINATION_CUTOFF, half=False)
    dispersion = _route_set(POSITIONS, VECTORS, DISPERSION_CUTOFF, half=True)
    x = jnp.asarray(positions)
    h = jnp.asarray(VECTORS)
    clean = padded.evaluate(
        _routes(x, h, coordination, COORDINATION_CUTOFF, 1),
        _routes(x, h, dispersion, DISPERSION_CUTOFF, 2),
    )
    assert bool(clean.successful)
    np.testing.assert_allclose(float(clean.energy), _reference(), rtol=1.0e-11)
    assert float(clean.atom_energy[3]) == 0.0
    assert float(clean.evidence.coordination_numbers[3]) == 0.0


def test_active_non_element_particles_and_coincident_atoms_refuse() -> None:
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray([0, 1]),
        np.asarray([6, 0]),
        np.asarray([1.0, 1.0]),
        UNITS,
        element_mask=np.asarray([True, False]),
    ).prepare()
    with pytest.raises(ValueError, match="non-element"):
        _term().prepare(system)
    prepared = _term().prepare(_system((6, 6), cell=None))
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    result = _energy_function(prepared, positions, None)(
        jnp.asarray(positions), jnp.zeros((3, 3))
    )
    assert not bool(result.evidence.geometry_successful)
    assert not bool(result.successful)


def test_coordination_requires_full_directed_routes() -> None:
    prepared = _term().prepare(_system())
    half = _route_set(POSITIONS, VECTORS, COORDINATION_CUTOFF, half=True)
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)
    with pytest.raises(ValueError, match="full directed"):
        prepared.coordination_numbers(_routes(x, h, half, COORDINATION_CUTOFF, 2))


# --- independent energies and counting -----------------------------------------


@pytest.mark.parametrize(
    "interpolation", ["normalized-gaussian", "nvalchemi-relative-threshold"]
)
@pytest.mark.parametrize("switch", [None, (6.5, 8.0)])
def test_lattice_energy_matches_independent_image_sum(
    interpolation: D3InterpolationRule, switch: tuple[float, float] | None
) -> None:
    prepared = _term(interpolation, switch=switch).prepare(_system())
    result = _energy_function(prepared)(jnp.asarray(POSITIONS), jnp.asarray(VECTORS))
    threshold = 12.0 if interpolation == "nvalchemi-relative-threshold" else None
    assert bool(result.successful)
    np.testing.assert_allclose(
        float(result.energy),
        _reference(switch=switch, threshold=threshold),
        rtol=1.0e-11,
    )
    np.testing.assert_allclose(
        float(jnp.sum(result.atom_energy)), float(result.energy), rtol=1.0e-13
    )


def test_threshold_rule_is_a_distinct_definition() -> None:
    gaussian = _term("normalized-gaussian").prepare(_system())
    thresholded = _term("nvalchemi-relative-threshold").prepare(_system())
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)
    smooth = float(_energy_function(gaussian)(x, h).energy)
    stepped = float(_energy_function(thresholded)(x, h).energy)
    np.testing.assert_allclose(
        smooth - stepped,
        _reference() - _reference(threshold=12.0),
        rtol=1.0e-6,
        atol=1.0e-12 * abs(smooth),
    )
    np.testing.assert_allclose(stepped, _reference(threshold=12.0), rtol=1.0e-11)
    assert gaussian.regularity.interpolation == "smooth"
    assert thresholded.regularity.interpolation == "threshold-steps"
    assert thresholded.regularity.threshold_relative_weight == pytest.approx(
        math.exp(-12.0)
    )
    assert gaussian.term_id != thresholded.plan.term_id


def test_half_and_directed_dispersion_routes_count_self_images_identically() -> None:
    prepared = _term().prepare(_system())
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)
    half = _energy_function(prepared, half=True)(x, h)
    directed = _energy_function(prepared, half=False)(x, h)
    np.testing.assert_allclose(float(half.energy), float(directed.energy), rtol=1.0e-13)
    receivers, sources, shifts = _route_set(
        POSITIONS, VECTORS, DISPERSION_CUTOFF, half=True
    )
    self_images = receivers == sources
    assert np.count_nonzero(self_images) > 0
    repeated = {}
    for t, s in zip(receivers, sources):
        repeated[(t, s)] = repeated.get((t, s), 0) + 1
    assert max(repeated.values()) > 1
    trimmed = tuple(value[~self_images] for value in (receivers, sources, shifts))
    coordination = _route_set(POSITIONS, VECTORS, COORDINATION_CUTOFF, half=False)
    without = prepared.evaluate(
        _routes(x, h, coordination, COORDINATION_CUTOFF, 1),
        _routes(x, h, trimmed, DISPERSION_CUTOFF, 2),
    )
    assert abs(float(without.energy) - float(half.energy)) > 1.0e-6 * abs(
        float(half.energy)
    )


def test_large_cutoff_small_cell_coordination_matches_independent_count() -> None:
    prepared = _term().prepare(_system())
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)
    routes = _routes(
        x,
        h,
        _route_set(POSITIONS, VECTORS, COORDINATION_CUTOFF, half=False),
        COORDINATION_CUTOFF,
        1,
    )
    numbers, status, geometry = prepared.coordination_numbers(routes)
    data = synthetic_tables()
    receivers, sources, shifts = image_routes(POSITIONS, VECTORS, COORDINATION_CUTOFF)
    expected = np.zeros(3)
    for t, s, shift in zip(receivers, sources, shifts):
        distance = np.linalg.norm(POSITIONS[t] - POSITIONS[s] + shift @ VECTORS)
        covalent = data["covalent_radii"][ROWS[t]] + data["covalent_radii"][ROWS[s]]
        expected[t] += conventional_count(distance / BOHR_ANGSTROM, covalent)
    assert bool(status) and bool(geometry)
    assert np.any(receivers == sources)
    np.testing.assert_allclose(np.asarray(numbers), expected, rtol=1.0e-13)


def test_finite_dimer_energy_and_hard_and_switched_cutoff_joins() -> None:
    hard = _term(dispersion_cutoff=6.0).prepare(_system((6, 6), cell=None))
    switched = _term(dispersion_cutoff=6.0, switch=(5.5, 6.0)).prepare(
        _system((6, 6), cell=None)
    )
    rows = np.asarray([1, 1])

    def dimer(term: Any, distance: float) -> Any:
        positions = np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        return _energy_function(term, positions, None)(
            jnp.asarray(positions), jnp.zeros((3, 3))
        )

    inside, outside = dimer(hard, 6.0 - 1.0e-9), dimer(hard, 6.0 + 1.0e-9)
    reference = _reference(
        np.asarray([[0.0, 0.0, 0.0], [6.0 - 1.0e-9, 0.0, 0.0]]),
        None,
        dispersion_cutoff=6.0,
        rows=rows,
    )
    np.testing.assert_allclose(float(inside.energy), reference, rtol=1.0e-11)
    assert float(outside.energy) == 0.0
    assert abs(float(inside.energy)) > 1.0e-6
    assert hard.regularity.dispersion_cutoff == "value-step"
    assert hard.regularity.dispersion_cutoff_step >= abs(float(inside.energy)) * 0.999
    near = dimer(switched, 6.0 - 1.0e-6)
    assert abs(float(near.energy)) < 1.0e-15
    assert switched.regularity.dispersion_cutoff == "c2-s5-switch"

    evaluate = _energy_function(
        switched, np.asarray([[0.0, 0.0, 0.0], [5.75, 0.0, 0.0]]), None
    )

    def switched_energy(distance: jax.Array) -> jax.Array:
        positions = jnp.zeros((2, 3)).at[1, 0].set(distance)
        return evaluate(positions, jnp.zeros((3, 3))).energy

    curvature = jax.grad(jax.grad(switched_energy))
    third = jax.grad(curvature)
    scale = abs(float(curvature(jnp.asarray(5.5 - 1.0e-3))))
    assert scale > 0.0
    # C2 at a join means equal one-sided limits of E''. Each limit is taken by
    # first-order Taylor extrapolation from distance delta, leaving
    # delta^2 / 2 * |E''''| (about 1e-14 here) instead of the delta * |E'''|
    # (about 6e-8 at 5.5, where the S5 switch is C2 but not C3) of comparing
    # E''(join - delta) with E''(join + delta) directly.
    delta = 1.0e-7
    for join in (5.5, 6.0):
        below, above = jnp.asarray(join - delta), jnp.asarray(join + delta)
        left = float(curvature(below)) + delta * float(third(below))
        right = float(curvature(above)) - delta * float(third(above))
        np.testing.assert_allclose(left, right, atol=1.0e-8 * scale)


def test_s5_switch_matches_reference_with_c2_joins() -> None:
    for distance in (4.0, 5.0, 5.3, 5.71, 6.0, 6.4):
        np.testing.assert_allclose(
            float(s5_switch(jnp.asarray(distance), 5.0, 6.0)),
            reference_s5(distance, 5.0, 6.0),
            rtol=1.0e-14,
            atol=1.0e-15,
        )
    first = jax.grad(lambda r: s5_switch(r, 5.0, 6.0))
    second = jax.grad(first)
    for join in (5.0, 6.0):
        assert abs(float(first(jnp.asarray(join)))) < 1.0e-12
        assert abs(float(second(jnp.asarray(join)))) < 1.0e-12


def test_coordination_cutoff_step_is_reported_and_observed() -> None:
    prepared = _term().prepare(_system())
    data = synthetic_tables()
    cutoff = COORDINATION_CUTOFF / BOHR_ANGSTROM
    expected = max(
        conventional_count(cutoff, data["covalent_radii"][a] + data["covalent_radii"][b])
        for a in range(2)
        for b in range(2)
    )
    np.testing.assert_allclose(
        prepared.regularity.coordination_cutoff_step, expected, rtol=1.0e-12
    )
    dimer = _term().prepare(_system((6, 6), cell=None))

    def coordination(distance: float) -> float:
        positions = np.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        x = jnp.asarray(positions)
        routes = _routes(
            x,
            jnp.zeros((3, 3)),
            _route_set(positions, None, COORDINATION_CUTOFF, half=False),
            COORDINATION_CUTOFF,
            1,
        )
        return float(dimer.coordination_numbers(routes)[0][0])

    step = coordination(COORDINATION_CUTOFF - 1.0e-9) - coordination(
        COORDINATION_CUTOFF + 1.0e-9
    )
    np.testing.assert_allclose(
        step,
        conventional_count(cutoff, 2.0 * data["covalent_radii"][1]),
        rtol=1.0e-6,
    )


# --- derivatives -----------------------------------------------------------------


def _strain_energy(prepared: Any) -> Any:
    evaluate = _energy_function(prepared)

    def energy(strain: jax.Array) -> jax.Array:
        deformation = jnp.eye(3) + strain
        x = jnp.asarray(POSITIONS) @ deformation.T
        h = jnp.asarray(VECTORS) @ deformation.T
        return evaluate(x, h).energy

    return energy


@pytest.mark.parametrize("switch", [None, (6.5, 8.0)])
def test_forces_and_strain_response_match_reference_differences(
    switch: tuple[float, float] | None,
) -> None:
    prepared = _term(switch=switch).prepare(_system())
    evaluate = _energy_function(prepared)
    gradient = jax.grad(lambda x: evaluate(x, jnp.asarray(VECTORS)).energy)(
        jnp.asarray(POSITIONS)
    )
    expected = central_difference(
        lambda x: _reference(x, switch=switch), POSITIONS, 1.0e-5
    )
    np.testing.assert_allclose(np.asarray(gradient), expected, rtol=1.0e-6, atol=1.0e-9)
    strain = jax.grad(_strain_energy(prepared))(jnp.zeros((3, 3)))
    expected_strain = central_difference(
        lambda e: _reference(
            strained(POSITIONS, e.reshape(3, 3)),
            strained(VECTORS, e.reshape(3, 3)),
            switch=switch,
        ),
        np.zeros(9),
        1.0e-6,
    ).reshape(3, 3)
    np.testing.assert_allclose(
        np.asarray(strain), expected_strain, rtol=1.0e-6, atol=1.0e-9
    )
    np.testing.assert_allclose(np.asarray(jnp.sum(gradient, axis=0)), 0.0, atol=1.0e-10)


def test_damping_and_table_coefficient_derivatives_are_connected() -> None:
    prepared = _term().prepare(_system())
    evaluate = _energy_function(prepared)
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)

    # Differentiate the four float damping leaves; ``damping_identity`` (uint32
    # IdentityWords) stays the prepared identity and is not a parameter.
    names = ("a1", "a2", "s8", "s6")

    def with_damping(values: tuple[jax.Array, ...]) -> jax.Array:
        term = eqx.tree_at(
            lambda value: (
                value.plan.damping.a1,
                value.plan.damping.a2,
                value.plan.damping.s8,
                value.plan.damping.s6,
            ),
            prepared,
            values,
        )
        return evaluate(x, h, term).energy

    damping = prepared.plan.damping
    gradient = jax.grad(with_damping)((damping.a1, damping.a2, damping.s8, damping.s6))
    for name, derivative in zip(names, gradient, strict=True):
        step = 1.0e-6
        expected = (
            _reference(damping={**DAMPING, name: DAMPING[name] + step})
            - _reference(damping={**DAMPING, name: DAMPING[name] - step})
        ) / (2.0 * step)
        np.testing.assert_allclose(float(derivative), expected, rtol=1.0e-6)

    def with_c6(table: jax.Array) -> jax.Array:
        term = eqx.tree_at(lambda value: value.plan.tables.c6_reference, prepared, table)
        return evaluate(x, h, term).energy

    table_gradient = np.asarray(jax.grad(with_c6)(prepared.plan.tables.c6_reference))
    data = synthetic_tables()
    for index in ((0, 1, 1, 2), (1, 1, 0, 1), (0, 0, 1, 1)):
        a, b, p, q = index
        # C6 interpolation and energy are linear in this table. Evaluating the
        # independent reference on a basis direction avoids cancellation of a
        # tiny coefficient response against the complete interaction energy.
        direction = np.zeros_like(data["c6_reference"])
        direction[a, b, p, q] = 1.0
        if (a, p) != (b, q):
            direction[b, a, q, p] = 1.0
        replaced = data.copy()
        replaced["c6_reference"] = direction
        expected = _reference(tables=replaced)
        actual = table_gradient[a, b, p, q] + (
            table_gradient[b, a, q, p] if (a, p) != (b, q) else 0.0
        )
        np.testing.assert_allclose(actual, expected, rtol=1.0e-6, atol=1.0e-12)


def test_mixed_second_derivatives_inside_the_smooth_domain() -> None:
    prepared = _term(switch=(6.5, 8.0)).prepare(_system())
    evaluate = _energy_function(prepared)
    h = jnp.asarray(VECTORS)
    force = jax.grad(lambda x: evaluate(x, h).energy)
    direction = jnp.asarray(np.random.default_rng(3).normal(size=POSITIONS.shape))
    x = jnp.asarray(POSITIONS)
    _, hvp = jax.jvp(force, (x,), (direction,))
    step = 1.0e-5
    expected = (force(x + step * direction) - force(x - step * direction)) / (2.0 * step)
    np.testing.assert_allclose(
        np.asarray(hvp), np.asarray(expected), rtol=1.0e-6, atol=1.0e-9
    )

    def position_gradient(a1: jax.Array) -> jax.Array:
        damping = eqx.tree_at(lambda value: value.a1, prepared.plan.damping, a1)
        term = eqx.tree_at(lambda value: value.plan.damping, prepared, damping)
        return jax.grad(lambda y: evaluate(y, h, term).energy)(x)

    a1 = prepared.plan.damping.a1
    mixed = jax.jacfwd(position_gradient)(a1)
    expected_mixed = (
        position_gradient(a1 + 1.0e-6) - position_gradient(a1 - 1.0e-6)
    ) / 2.0e-6
    np.testing.assert_allclose(
        np.asarray(mixed), np.asarray(expected_mixed), rtol=1.0e-6, atol=1.0e-9
    )

    strain_energy = _strain_energy(prepared)
    cell_gradient = jax.grad(strain_energy)
    tangent = jnp.asarray(np.random.default_rng(5).normal(size=(3, 3))) * 0.1
    _, cell_hvp = jax.jvp(cell_gradient, (jnp.zeros((3, 3)),), (tangent,))
    expected_cell = (
        cell_gradient(1.0e-6 * tangent) - cell_gradient(-1.0e-6 * tangent)
    ) / 2.0e-6
    np.testing.assert_allclose(
        np.asarray(cell_hvp), np.asarray(expected_cell), rtol=1.0e-5, atol=1.0e-8
    )


def test_route_chunking_does_not_change_the_definition() -> None:
    x, h = jnp.asarray(POSITIONS), jnp.asarray(VECTORS)
    small = _energy_function(_term(route_chunk=7).prepare(_system()))(x, h)
    large = _energy_function(_term(route_chunk=100000).prepare(_system()))(x, h)
    np.testing.assert_allclose(float(small.energy), float(large.energy), rtol=1.0e-14)
