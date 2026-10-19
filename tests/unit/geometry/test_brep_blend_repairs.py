"""Analytic blend regressions for rational junctions and native meridian gauges."""

import math

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
from jax import Array

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import (
    brep_cylinder,
    BRepModel,
    BRepTessellationPolicy,
    prepare_brep_query,
)
from phydrax.geometry.brep._differentiable import _FixedTopologyBRepKernel
from phydrax.geometry.brep._fillet import brep_fillet, BRepBlendResult
from phydrax.geometry.brep._patches import SurfaceIsoparametricCurve, TorusPatch
from tests._support.brep_feature_sources import unnormalized_wedge


# Empirical `test_rational_wedge_fillet_certifies_mixed_native_profile_junctions`: matched max |delta|=1.42109e-14; joint allowance fraction=7.10543e-05, margin=99.9929%.
_WEDGE_FILLET_VOLUME_ATOL = 2e-10
# Empirical `test_bottom_circular_fillet_uses_canonical_native_meridian_window`: matched max |delta|=7.10543e-15; joint allowance fraction=3.55271e-05, margin=99.9964%.
_CIRCULAR_FILLET_VOLUME_ATOL = 2e-10
# Empirical `test_bottom_fillet_radius_map_preserves_shifted_phase_and_seam_evidence`: matched max |delta|=4.03199e-10; joint allowance fraction=0.002016, margin=99.7984%.
_CIRCULAR_FILLET_RADIUS_DERIVATIVE_ATOL = 2e-7


_LAZY = BRepTessellationPolicy(realize=False)
_CONTRACT = SpatialCoordinateContract.si()


def test_rational_wedge_fillet_certifies_mixed_native_profile_junctions() -> None:
    source = unnormalized_wedge()
    geometry = source.require_geometry()
    edge = next(
        index
        for index, (first, last) in enumerate(geometry.edge_vertices)
        if first != last
        and np.array_equal(np.asarray(geometry.vertex_points[first])[:2], (0.0, 0.0))
        and np.array_equal(np.asarray(geometry.vertex_points[last])[:2], (0.0, 0.0))
    )
    result = brep_fillet(source, (source.edge_ids[edge],), 5 / 16, tessellation=_LAZY)
    assert result.status == "certified", result.reason
    model = result.model
    assert model is not None
    with mpmath.workdps(80):
        alpha = mpmath.acos(mpmath.mpf(3) / 5)
        removed = mpmath.mpf(5 / 16) ** 2 * 4 * (2 - (mpmath.pi - alpha) / 2)
        expected = float(56 - removed)
    assert float(prepare_brep_query(model).measures.solid_volumes[0]) == pytest.approx(
        expected, abs=_WEDGE_FILLET_VOLUME_ATOL
    )
    assert np.all(np.asarray(result.g1_angle_bounds)[:, 1] < 1e-6)
    assert model.require_geometry().edge_vertex_gap_bounds is not None


def _bottom_fillet() -> tuple[BRepModel, BRepBlendResult]:
    source = brep_cylinder(2.0, 3.0, coordinate_contract=_CONTRACT, tessellation=_LAZY)
    geometry = source.require_geometry()
    edge = next(
        index
        for index, (first, last) in enumerate(geometry.edge_vertices)
        if first == last and float(geometry.vertex_points[first, 2]) == 0.0
    )
    result = brep_fillet(source, (source.edge_ids[edge],), 0.25, tessellation=_LAZY)
    assert result.status == "certified", result.reason
    return source, result


def test_bottom_circular_fillet_uses_canonical_native_meridian_window() -> None:
    source, result = _bottom_fillet()
    model = result.model
    assert model is not None
    patch = model.patches[-1]
    assert isinstance(patch, TorusPatch)
    assert float(model.parameter_bounds[-1, 0, 1]) >= 0.0
    geometry = model.require_geometry()
    seam = next(
        curve
        for curve in geometry.curves
        if isinstance(curve, SurfaceIsoparametricCurve)
        and curve.surface is patch
        and curve.fixed_axis == 0
    )
    assert seam.parameter_domain == tuple(
        float(value) for value in np.asarray(model.parameter_bounds[-1, :, 1])
    )
    old = prepare_brep_query(source).measures
    new = prepare_brep_query(model).measures
    radius = 0.25
    removed = math.pi * (
        (2 - math.pi / 2) * 2 * radius**2 + (math.pi / 2 - 5 / 3) * radius**3
    )
    assert float(new.solid_volumes[0]) == pytest.approx(
        float(old.solid_volumes[0]) - removed, abs=_CIRCULAR_FILLET_VOLUME_ATOL
    )
    assert np.all(np.asarray(result.g1_angle_bounds)[:, 1] < 1e-6)


def test_bottom_fillet_radius_map_preserves_shifted_phase_and_seam_evidence() -> None:
    _, result = _bottom_fillet()
    source = result.source
    assert source is not None
    compiled = source.compile(coordinate_unit=_CONTRACT.length_unit)
    kernel = compiled.kernel
    assert isinstance(kernel, _FixedTopologyBRepKernel)
    index = compiled.schema.index(compiled.schema.parameter_ids[0])

    def volume(radius: Array) -> Array:
        return kernel.measure(compiled.state.replace_at(index, radius))

    step = 2.0**-15
    values = []
    for radius in (0.25 - step, 0.25, 0.25 + step):
        state = compiled.state.replace_at(index, jnp.asarray(radius, dtype=jnp.float64))
        realized = kernel.realize(state)
        assert float(realized.parameter_bounds[-1, 0, 1]) >= 0.0
        assert float(realized.seam_residual) <= float(realized.seam_bound)
        values.append(float(kernel.measure(state)))
    automatic = float(jax.grad(volume)(jnp.asarray(0.25, dtype=jnp.float64)))
    difference = (values[2] - values[0]) / (2 * step)
    assert automatic == pytest.approx(
        difference, abs=_CIRCULAR_FILLET_RADIUS_DERIVATIVE_ATOL
    )
