# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hydroelastic pressure extraction on a binary64 reference tetrahedron."""

from __future__ import annotations

import numpy as np
import pytest

from phydrax.discretization import (
    CellBlock,
    CellMesh,
    HydroelasticPatchExtractionPlan,
    HydroelasticPressureFieldPlan,
    HydroelasticPressureFieldState,
)


# Vertices and half-plane crossings are representable in binary64. Fewer than 32 rounded
# operations produce each interpolation, centroid, area, or normal component;
# 64 eps covers those chains and the independently rounded analytic references.
_PATCH_ATOL = 64.0 * np.finfo(np.float64).eps


def test_pressure_patch_preserves_half_plane_measure_and_interpolation() -> None:
    mesh = CellMesh(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        (
            CellBlock(
                "tetrahedron", "tetrahedron", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
            ),
        ),
    )
    plus = HydroelasticPressureFieldPlan(mesh)
    minus = HydroelasticPressureFieldPlan(mesh)
    prepared = HydroelasticPatchExtractionPlan(
        maximum_overlap_pairs=1, maximum_polytope_edges=12, patch_capacity=4
    ).prepare(plus, minus)
    plus_state = HydroelasticPressureFieldState(np.asarray((0.0, 1.0, 1.0, 1.0)))
    minus_state = HydroelasticPressureFieldState(np.full((4,), 0.5, dtype=np.float64))
    extraction = prepared.evaluate(plus_state, minus_state)
    patch, evidence = extraction.patch, extraction.evidence
    valid = np.asarray(patch.valid)

    assert bool(evidence.successful)
    assert bool(evidence.complete) and bool(evidence.derivative_valid)
    assert int(evidence.triangle_count) == 1
    assert int(evidence.overflow_count) == 0
    assert evidence.plus_field_id == plus.field_id
    assert evidence.minus_field_id == minus.field_id
    np.testing.assert_allclose(
        np.asarray(patch.quadrature_point)[valid],
        np.full((1, 3), 1.0 / 6.0),
        rtol=0.0,
        atol=_PATCH_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(patch.pressure)[valid], (0.5,), rtol=0.0, atol=_PATCH_ATOL
    )
    np.testing.assert_allclose(
        evidence.total_area, np.sqrt(3.0) / 8.0, rtol=0.0, atol=_PATCH_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(patch.normal)[valid],
        np.full((1, 3), 1.0 / np.sqrt(3.0)),
        rtol=0.0,
        atol=_PATCH_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(patch.plus_interpolation)[valid],
        ((0.5, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),),
        rtol=0.0,
        atol=_PATCH_ATOL,
    )
    scaled = prepared.evaluate(
        HydroelasticPressureFieldState(2.0 * plus_state.pressure),
        HydroelasticPressureFieldState(2.0 * minus_state.pressure),
    )
    assert scaled.patch.patch_id == patch.patch_id
    np.testing.assert_array_equal(scaled.patch.valid, patch.valid)
    np.testing.assert_allclose(
        scaled.patch.quadrature_weight,
        patch.quadrature_weight,
        rtol=0.0,
        atol=_PATCH_ATOL,
    )
    np.testing.assert_allclose(
        scaled.patch.pressure, 2.0 * patch.pressure, rtol=0.0, atol=_PATCH_ATOL
    )


def test_pressure_field_boundary_refuses_untyped_meshes() -> None:
    with pytest.raises(TypeError, match="mesh"):
        HydroelasticPressureFieldPlan(None)  # ty: ignore[invalid-argument-type]
