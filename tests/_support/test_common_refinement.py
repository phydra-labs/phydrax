"""Contract and deliberate-fault checks for common-refinement test support."""

from __future__ import annotations

import equinox as eqx
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.geometry import CommonRefinementPolicy, prepare_common_refinement
from tests._support.common_refinement import (
    assert_complete_moments,
    quad_mesh,
    unit_cube_moments,
)


@pytest.mark.parametrize("dimension", (2, 3), ids=("square", "cube"))
def test_unit_domain_moment_reference_matches_separable_integrals(dimension: int) -> None:
    first, second = unit_cube_moments(dimension)
    # Independent separable integrals: integral(x)=1/2, integral(x^2)=1/3,
    # and integral(x*y)=1/4 on a unit box.
    np.testing.assert_array_equal(first, np.full(dimension, 0.5, dtype=np.float64))
    expected = np.asarray(
        [
            [1.0 / 3.0 if row == column else 0.25 for column in range(dimension)]
            for row in range(dimension)
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(second, expected)


@pytest.mark.meshcore
@pytest.mark.skipif(
    not meshcore_available(), reason="native phydrax-meshcore library is not available"
)
@pytest.mark.parametrize("moment", ("measure", "first", "second"))
def test_complete_moment_assertion_rejects_corrupted_scientific_output(
    moment: str,
) -> None:
    mesh = quad_mesh(2)
    refinement = prepare_common_refinement(
        mesh, mesh, policy=CommonRefinementPolicy(second_moments=True)
    )
    assert_complete_moments(refinement, 2)
    match moment:
        case "measure":
            corrupted = eqx.tree_at(
                lambda value: value.volumes, refinement, 2.0 * refinement.volumes
            )
        case "first":
            corrupted = eqx.tree_at(
                lambda value: value.first_moments,
                refinement,
                2.0 * refinement.first_moments,
            )
        case "second":
            assert refinement.second_moments is not None
            corrupted = eqx.tree_at(
                lambda value: value.second_moments,
                refinement,
                2.0 * refinement.second_moments,
            )
        case _:
            raise ValueError(f"moment must name a supported fault; got {moment}.")
    with pytest.raises(AssertionError):
        assert_complete_moments(corrupted, 2)
