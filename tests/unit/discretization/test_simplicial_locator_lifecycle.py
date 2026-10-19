#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import (
    CellLocationStatus,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    PreparedSimplicialCellLocator,
    SimplicialLocationPolicy,
)
from phydrax.discretization.fem import prepare_finite_element_cell_map
from tests._support.cell_meshes import unit_square


@pytest.fixture
def locator() -> PreparedSimplicialCellLocator:
    mesh = unit_square()
    prepared = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    ).prepare()
    return PreparedSimplicialCellLocator(
        prepare_finite_element_cell_map(prepared, 0),
        prepared.default_runtime.coordinates,
        SimplicialLocationPolicy(2, 8, 1),
    )


def test_locator_phases_preserve_lowest_cell_facet_tie_and_all_candidates(
    locator: PreparedSimplicialCellLocator,
) -> None:
    points = jnp.asarray(((0.75, 0.25), (0.5, 0.5)), dtype=jnp.float64)
    result = locator.locate(points)
    np.testing.assert_array_equal(result.cell_ids, (0, 0))
    np.testing.assert_array_equal(result.status, (CellLocationStatus.LOCATED,) * 2)
    np.testing.assert_array_equal(result.candidate_count, (1, 2))
    np.testing.assert_array_equal(result.candidates_complete, (True, True))
    np.testing.assert_array_equal(result.successful, (True, True))
    # Affine inversion residual is bounded by the caller-declared admission tolerance.
    np.testing.assert_allclose(
        result.reference_coordinates,
        ((0.5, 0.25), (0.0, 0.5)),
        rtol=0.0,
        atol=locator.policy.residual_tolerance,
    )
    assert np.all(
        np.asarray(result.geometry_residual) <= locator.policy.residual_tolerance
    )
    candidates = np.asarray(locator.support_cells(points))
    np.testing.assert_array_equal(candidates[1][candidates[1] >= 0], (0, 1))


def test_locator_nonfinite_and_outside_keep_distinct_status_evidence(
    locator: PreparedSimplicialCellLocator,
) -> None:
    result = locator.locate(jnp.asarray(((2.0, 2.0), (np.nan, 0.5)), dtype=jnp.float64))
    np.testing.assert_array_equal(result.cell_ids, (-1, -1))
    np.testing.assert_array_equal(
        result.status, (CellLocationStatus.OUTSIDE, CellLocationStatus.NONFINITE)
    )
    np.testing.assert_array_equal(result.inside, (False, False))
    np.testing.assert_array_equal(result.candidates_complete, (True, False))


def test_locator_mask_limits_containing_candidates_without_changing_search_evidence(
    locator: PreparedSimplicialCellLocator,
) -> None:
    result = locator.locate(
        jnp.asarray(((0.5, 0.5),), dtype=jnp.float64),
        cell_mask=jnp.asarray((False, True), dtype=jnp.bool_),
    )
    np.testing.assert_array_equal(result.cell_ids, (1,))
    np.testing.assert_array_equal(result.candidate_count, (1,))
    np.testing.assert_array_equal(result.candidates_complete, (True,))
    np.testing.assert_array_equal(result.status, (CellLocationStatus.LOCATED,))


def test_reference_capabilities_preserve_simplex_weights_and_boundary_tolerance(
    locator: PreparedSimplicialCellLocator,
) -> None:
    reference = jnp.asarray(((0.25, 0.5), (0.5, 0.5), (0.75, 0.5)), dtype=jnp.float64)
    np.testing.assert_array_equal(
        locator.contains_reference(reference), (True, True, False)
    )
    np.testing.assert_array_equal(
        locator.reference_weights(reference),
        ((0.25, 0.25, 0.5), (0.0, 0.5, 0.5), (-0.25, 0.75, 0.5)),
    )
