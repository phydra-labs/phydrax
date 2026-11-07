from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization import CellGeometrySpec, CellMesh
from tools.meshing_qualification.scenarios.reconstruction import (
    _actual_target_triangles,
    _input_to_envelope_upper_bound,
    _point_triangle_distances,
)


def test_continuous_enclosure_uses_declared_threshold_exactly() -> None:
    source = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    target = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]])
    assert _input_to_envelope_upper_bound(source, target, 1.0) == 1.0
    with pytest.raises(RuntimeError, match="inconclusive"):
        _input_to_envelope_upper_bound(source, target, np.nextafter(1.0, 0.0))


def test_displaced_envelope_cannot_borrow_a_producer_certificate() -> None:
    source = np.array([[[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.2, 0.0]]])
    target = source + np.array([0.0, 0.0, 1.0])
    with pytest.raises(RuntimeError, match="inconclusive"):
        _input_to_envelope_upper_bound(source, target, 0.55)


def test_sample_point_islands_do_not_certify_the_intervening_source() -> None:
    source = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    samples = np.array([[i / 16, j / 16, 0.0] for i in range(17) for j in range(17 - i)])
    # Tiny disjoint triangles contain every old diagnostic sample but leave
    # most of the source uncovered at the requested distance.
    target = np.stack(
        (samples, samples + [1.0e-5, 0.0, 0.0], samples + [0.0, 1.0e-5, 0.0]),
        axis=1,
    )
    assert np.max(_point_triangle_distances(samples, target)) < 1.0e-12
    with pytest.raises(RuntimeError, match="inconclusive"):
        _input_to_envelope_upper_bound(source, target, 0.005)


def test_empty_or_nonfinite_triangle_soup_is_not_a_proof() -> None:
    source = np.zeros((1, 3, 3))
    with pytest.raises(ValueError, match="finite nonempty"):
        _input_to_envelope_upper_bound(source, np.empty((0, 3, 3)), 0.5)
    source[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite nonempty"):
        _input_to_envelope_upper_bound(source, np.zeros((1, 3, 3)), 0.5)


def test_enclosure_uses_actual_target_boundary_vertices() -> None:
    points = np.asarray(
        ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0), (0.0, 0.0, 2.0))
    )
    mesh = CellMesh.from_tetrahedra(points, np.asarray(((0, 1, 2, 3),), dtype=np.int32))
    target = _actual_target_triangles(mesh, CellGeometrySpec.affine(mesh))
    assert target.shape == (4, 3, 3)
    source = np.asarray([((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0))])
    with pytest.raises(RuntimeError, match="inconclusive"):
        _input_to_envelope_upper_bound(source, target, 0.55)
