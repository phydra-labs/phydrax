"""Curving-budget references bound independently evaluated quadratic maps."""

import numpy as np
import pytest

from tests._support.cad_curving import (
    circular_p2_fidelity_bound,
    icosphere,
    sphere_fidelity_bound,
    triangle_lattice,
)


@pytest.mark.parametrize("half_angle", [np.pi / 6.0, np.pi / 4.0])
def test_circle_budget_covers_quadratic_arc_radial_error(half_angle: float) -> None:
    # Evaluate the quadratic interpolant at the extrema t=+-1/sqrt(2).
    # A Euclidean norm uses at most a few rounded binary64 operations; 16 eps
    # bounds their accumulated absolute comparison error at unit scale.
    comparison_roundoff = 16.0 * np.finfo(np.float64).eps
    t = np.asarray((-(2.0**-0.5), 2.0**-0.5), dtype=np.float64)
    points = np.column_stack(
        (1.0 - (1.0 - np.cos(half_angle)) * t**2, np.sin(half_angle) * t)
    )
    error = np.max(np.abs(np.linalg.norm(points, axis=1) - 1.0))
    budget = circular_p2_fidelity_bound(half_angle)
    assert error > 0.0
    assert error <= budget / 2.0 + comparison_roundoff
    assert budget < 1.0


def test_sphere_budget_covers_independent_quadratic_triangle_interpolation() -> None:
    mesh = icosphere(0)
    corners = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    reference = triangle_lattice(8)
    barycentric = np.column_stack((1.0 - reference.sum(axis=1), reference))
    values = barycentric * (2.0 * barycentric - 1.0)
    interpolated = values[None] @ corners
    for first, second in ((0, 1), (1, 2), (2, 0)):
        midpoint = (corners[:, first] + corners[:, second]) / 2.0
        midpoint /= np.linalg.norm(midpoint, axis=1, keepdims=True)
        weights = 4.0 * barycentric[:, first] * barycentric[:, second]
        interpolated += weights[None, :, None] * midpoint[:, None, :]
    error = np.max(np.abs(np.linalg.norm(interpolated, axis=2) - 1.0))
    budget = sphere_fidelity_bound(mesh)
    assert error > 0.0
    assert error <= budget
    assert budget < 1.0
