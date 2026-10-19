# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""White-box mapped-density integration and whole-map correspondence contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    RestrictedCellGeometryElement,
)
from phydrax.discretization._cell_geometry_transfer import plc_nested_approximation
from phydrax.discretization._cell_geometry_transition_types import (
    NestedReferenceWitnesses,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._coordinate_enclosure import (
    evaluate,
    multiply,
    Polynomial,
)
from phydrax.discretization._mapped_geometry_measure import (
    affine_nested_geometry_pair_bound,
    certified_sqrt_polynomial_integral_state,
    certify_nested_geometry_pairs,
    polynomial_nested_geometry_pair_bound,
    reuse_embedded_polynomial_area,
)
from phydrax.discretization._nested_reference_types import NestedReferencePair
from phydrax.discretization._reference_cell import reference_cell_topology
from phydrax.geometry._embedding_state import MeshCertificateLimits
from tests._support.cell_meshes import kuhn_cube
from tests._support.coordinate_references import coordinate_reference_budget


# Preserve the unit-scale solver accuracy allocation (2**20 binary64 eps) with
# its stricter decimal threshold; the integer/Fraction formulas are the oracle.
_SOLVE_ACCURACY = 1e-10
# This is the requested relative moment enclosure, not an empirical tolerance.
_MOMENT_RELATIVE_ERROR = 1e-11


@pytest.mark.parametrize(
    ("kind", "expected"), (("triangle", (409, 400)), ("quadrilateral", (1059, 400)))
)
def test_high_degree_sqrt_density_preserves_weighted_reference_moments(
    kind: str,
    expected: tuple[int, int],
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    root: Polynomial = {
        (0, 0): Fraction(1),
        (3, 0): Fraction(1, 10),
        (0, 3): Fraction(1, 10),
    }
    weight: Polynomial = {(0, 0): Fraction(1), (1, 0): Fraction(1), (0, 1): Fraction(2)}
    ledger = coordinate_budget(1_000_000, 16 * 1024**2)
    with ledger.activate():
        value, error = certified_sqrt_polynomial_integral_state(
            multiply(root, root, budget=ledger),
            weight,
            kind,
            Fraction(1, 10**11),
            _MOMENT_RELATIVE_ERROR,
            10000,
            32,
            [0, 1_000_000],
            budget=ledger,
        )
    assert abs(Fraction(value) - Fraction(*expected)) <= Fraction(error)
    assert error <= _MOMENT_RELATIVE_ERROR * float(Fraction(*expected))


@pytest.mark.parametrize("increment", ((1, 10**11), (1, 2)))
def test_area_reuse_encloses_changed_gram_or_integrates_it_fresh(
    increment: tuple[int, int],
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    delta = Fraction(*increment)
    root: Polynomial = {(0, 0): Fraction(2), (1, 0): delta}
    reference: tuple[Polynomial, Fraction, float, float] = (
        {(0, 0): Fraction(4)},
        Fraction(4),
        1.0,
        0.0,
    )
    ledger = coordinate_budget(1_000_000, 16 * 1024**2)
    with ledger.activate():
        (value, error), prepared = reuse_embedded_polynomial_area(
            multiply(root, root, budget=ledger),
            "triangle",
            [reference],
            Fraction(1, 10**10),
            _SOLVE_ACCURACY,
            10000,
            32,
            [0, 1_000_000],
            budget=ledger,
        )
    assert abs(Fraction(value) - (Fraction(1) + delta / 6)) <= Fraction(error)
    assert error <= max(_SOLVE_ACCURACY, _SOLVE_ACCURACY * (value - error))
    if delta < Fraction(1, 1000):
        assert value == 1.0
        assert error > 0.0
        assert prepared is None
    else:
        assert value > 1.08
        assert prepared is not None


def test_polynomial_square_preserves_cancelled_consumer_values_under_work_cap(
    budget: CoordinateEnclosureBudget,
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    polynomial: Polynomial = {
        (a, b): Fraction((-1) ** (a + b) * (a + 1), 2 ** (a + b + 3))
        for a in range(11)
        for b in range(11 - a)
    }
    ledger = coordinate_budget(2800, 16 * 1024**2)
    with ledger.activate():
        squared = multiply(polynomial, polynomial, budget=ledger)
    for point in (
        (Fraction(0), Fraction(0)),
        (Fraction(1, 3), Fraction(-2, 5)),
        (Fraction(7, 6), Fraction(2, 3)),
    ):
        assert (
            evaluate(squared, point, budget=budget)
            == evaluate(polynomial, point, budget=budget) ** 2
        )
    cancellation: Polynomial = {
        (0, 0): Fraction(1),
        (1, 0): Fraction(-2),
        (2, 0): Fraction(1),
    }
    squared = multiply(cancellation, cancellation, budget=budget)
    assert evaluate(squared, (Fraction(1), Fraction(0)), budget=budget) == 0
    assert evaluate(squared, (Fraction(3), Fraction(0)), budget=budget) == 16


@pytest.mark.parametrize("coarsening", (False, True))
def test_affine_correspondence_bounds_corner_difference(
    coarsening: bool, coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget]
) -> None:

    source = kuhn_cube(1, orientation_swap=(0, 1))
    coordinates = np.asarray(source.coordinates).copy()
    coordinates[0, 0] += 0.125
    target = source.with_coordinates(coordinates, numeric_version="perturbed-map")
    ids = np.concatenate([np.asarray(block.global_ids) for block in source.blocks])
    reference = np.concatenate((np.zeros((1, 3)), np.eye(3)), axis=0)
    witness = NestedReferenceWitnesses(
        ids,
        ids,
        np.broadcast_to(reference, (ids.size, 4, 3)).copy(),
    )
    empty = NestedReferenceWitnesses(
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros((0, 4, 3), dtype=np.float64),
    )
    ledger = coordinate_budget(100_000, 16 * 1024**2)
    with ledger.activate():
        bounds = plc_nested_approximation(
            source,
            CellGeometrySpec.affine(source),
            target,
            CellGeometrySpec.affine(target),
            empty if coarsening else witness,
            witness if coarsening else empty,
            budget=ledger,
        )
    assert bounds == ((0.0, 0.125) if coarsening else (0.125, 0.0))
    assert ledger.work_units > 0


def test_constant_embedded_area_work_is_independent_of_prior_candidates(
    coordinate_budget: Callable[[int, int], CoordinateEnclosureBudget],
) -> None:

    def measure(candidate_count: int) -> tuple[tuple[float, float], int]:
        ledger = coordinate_budget(1_000_000, 64 * 1024**2)
        work = [0, 1_000_000]
        with ledger.activate():
            _, reference = reuse_embedded_polynomial_area(
                {(0, 0): Fraction(1), (2, 0): Fraction(1, 4)},
                "triangle",
                [],
                Fraction(1, 1000),
                0.0,
                1000,
                32,
                work,
                budget=ledger,
            )
            if reference is None:
                raise RuntimeError(
                    "The positive varying Gram must earn its reference proof."
                )
            starting_work = ledger.work_units
            result, _ = reuse_embedded_polynomial_area(
                {(0, 0): Fraction(2)},
                "triangle",
                [reference] * candidate_count,
                Fraction(1, 10**10),
                0.0,
                1000,
                32,
                work,
                budget=ledger,
            )
        return result, ledger.work_units - starting_work

    empty, empty_work = measure(0)
    crowded, crowded_work = measure(100)
    assert crowded == empty
    assert crowded_work == empty_work > 0
    value, error = crowded
    assert abs(value - np.sqrt(2.0) / 2.0) <= error


@pytest.mark.parametrize("kind", ("interval", "triangle", "tetrahedron"))
@pytest.mark.parametrize(
    ("defect", "accepted"),
    ((0.0, True), (2.0**-49, True), (2.0**-40, False)),
    ids=("exact", "within-rounding-bound", "different-physical-map"),
)
def test_affine_nested_corner_proof_matches_polynomial_bound_and_status(
    kind: str, defect: float, accepted: bool
) -> None:

    element = coordinate_lagrange_element(kind, 1)
    nodes = np.asarray(element.reference_nodes, dtype=np.float64)
    corners = np.asarray(reference_cell_topology(kind).vertices, dtype=np.float64)
    dimension = corners.shape[1]
    matrix = 0.5 * np.eye(dimension, dtype=np.float64)
    if dimension > 1:
        matrix[0, 1] = 0.125
    offset = np.full((dimension,), 0.125, dtype=np.float64)
    action = np.diag(np.arange(1, dimension + 1, dtype=np.float64))
    origin = np.ones((dimension,), dtype=np.float64)

    def physical(reference: np.ndarray, *, fine: bool) -> np.ndarray:
        mapped = reference @ matrix.T + offset if fine else reference
        values = mapped @ action.T + origin
        if fine:
            values[:, 0] += defect * (1.0 + 2.0 * reference[:, 0])
        return values

    row = np.arange(corners.shape[0], dtype=np.int32)[None]
    coarse_mesh = CellMesh(
        physical(corners, fine=False),
        (CellBlock("cells", kind, row),),
    )
    fine_mesh = CellMesh(
        physical(corners, fine=True),
        (CellBlock("cells", kind, row),),
    )
    route = np.arange(nodes.shape[0], dtype=np.int32)[None]
    coarse_values = physical(nodes, fine=False)
    fine_values = physical(nodes, fine=True)
    coarse_geometry = CellGeometrySpec(
        {"cells": element}, {"cells": route}, coarse_values
    )
    fine_geometry = CellGeometrySpec({"cells": element}, {"cells": route}, fine_values)
    polynomial_geometry = CellGeometrySpec(
        {
            "cells": RestrictedCellGeometryElement(
                element,
                kind,
                np.eye(dimension, dtype=np.float64),
                np.zeros((dimension,), dtype=np.float64),
            )
        },
        {"cells": route},
        fine_values,
    )
    pair = NestedReferencePair(0, 0, False, matrix, offset)
    with coordinate_reference_budget() as ledger:
        affine_bound = affine_nested_geometry_pair_bound(
            element, fine_values, element, coarse_values, pair, budget=ledger
        )
        polynomial_bound = polynomial_nested_geometry_pair_bound(
            element, fine_values, element, coarse_values, pair, budget=ledger
        )
        assert affine_bound is not None and affine_bound == polynomial_bound
        if accepted:
            assert (
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    fine_geometry,
                    (pair,),
                    budget=ledger,
                )
                == polynomial_bound
            )
            assert (
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    polynomial_geometry,
                    (pair,),
                    budget=ledger,
                )
                == polynomial_bound
            )
        else:
            with pytest.raises(ValueError, match="nested_geometry.physical_map"):
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    fine_geometry,
                    (pair,),
                    budget=ledger,
                )
            with pytest.raises(ValueError, match="nested_geometry.physical_map"):
                certify_nested_geometry_pairs(
                    coarse_mesh,
                    coarse_geometry,
                    fine_mesh,
                    polynomial_geometry,
                    (pair,),
                    budget=ledger,
                )


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with coordinate_reference_budget() as ledger:
        yield ledger


@pytest.fixture
def coordinate_budget() -> Iterator[Callable[[int, int], CoordinateEnclosureBudget]]:
    with ExitStack() as resources:

        def create(work: int, memory: int) -> CoordinateEnclosureBudget:
            return resources.enter_context(
                coordinate_reference_budget(
                    MeshCertificateLimits(
                        maximum_work_units=work,
                        maximum_scratch_bytes=memory,
                    )
                )
            )

        yield create
