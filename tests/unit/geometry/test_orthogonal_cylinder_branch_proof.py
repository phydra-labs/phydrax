"""Live source-branch proof behavior for orthogonal cylinder intersections."""

from __future__ import annotations

import gc
import math
import weakref
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
from phydrax.geometry._meshing_domain import (
    MeshingDomain,
    MeshingDomainCurve,
    MeshingSurfacePatch,
    PatchCurveUse,
    SourcePreparation,
)
from phydrax.geometry.brep._constructors import brep_box
from phydrax.geometry.brep._intersection import (
    intersect_surface_regions,
    IntersectionCurvePointRoot,
    TrimRootEndpoint,
)
from phydrax.geometry.brep._intersection_curve import (
    CurveTrimSegment,
    IntersectionCurve,
    IntersectionPCurve,
    SurfaceRegion,
)
from phydrax.geometry.brep._patches import CylinderPatch, ExactLineCurve


def _branch() -> IntersectionCurve:
    box = np.asarray(((0.0, -2.0), (math.tau, 2.0)), dtype=np.float64)
    large = CylinderPatch((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), 1.0)
    small = CylinderPatch((0, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 0), 0.5)
    result = intersect_surface_regions(
        SurfaceRegion(large, box), SurfaceRegion(small, box)
    )
    assert result.complete and len(result.curves) == 2
    return result.curves[0]


def _preparation() -> SourcePreparation:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    return SourcePreparation(MeshingDomain.from_brep(model))


def _bounds(
    curve: IntersectionCurve, preparation: SourcePreparation
) -> tuple[np.ndarray, ...]:
    first, last = 0.1, 0.2
    pcurve = IntersectionPCurve(curve, "first")
    derivative = preparation.curve_derivative_bounds(pcurve, first, last)
    return (
        curve.parameter_enclosures(first, last),
        curve.bounding_box(first, last),
        preparation.curve_box(pcurve, first, last),
        *derivative,
    )


def test_rebuilt_branch_reproves_and_preserves_bounds() -> None:
    curve = _branch()
    preparation = _preparation()
    first = _bounds(curve, preparation)
    second = _bounds(curve, preparation)
    fresh = _bounds(curve, _preparation())
    rebuilt = jax.tree_util.tree_map(lambda leaf: leaf, curve)
    assert rebuilt is not curve
    independent = _bounds(rebuilt, preparation)
    for original, repeated, cold, restored in zip(
        first, second, fresh, independent, strict=True
    ):
        np.testing.assert_array_equal(original, repeated)
        np.testing.assert_array_equal(original, cold)
        np.testing.assert_array_equal(original, restored)
    pcurve = IntersectionPCurve(curve, "first")
    np.testing.assert_array_equal(first[2], pcurve.bounding_box(0.1, 0.2))
    for prepared, direct in zip(
        first[3:], pcurve.derivative_bounds(0.1, 0.2), strict=True
    ):
        np.testing.assert_array_equal(prepared, direct)

    consumer_box = preparation.curve_box(pcurve, 0.1, 0.2)
    consumer_box[:] = 0.0
    np.testing.assert_array_equal(preparation.curve_box(pcurve, 0.1, 0.2), first[2])

    bad_axes = rebuilt.chart_axes.at[0].set((int(rebuilt.chart_axes[0]) + 1) % 4)
    corrupt = eqx.tree_at(lambda value: value.chart_axes, rebuilt, bad_axes)
    with pytest.raises(ValueError, match="no isolated source sign"):
        preparation.curve_box(IntersectionPCurve(corrupt, "first"), 0.1, 0.2)


def test_branch_proof_does_not_keep_curve_alive() -> None:
    curve = _branch()
    preparation = _preparation()
    _bounds(curve, preparation)
    reference = weakref.ref(curve)
    del curve, preparation
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize("reversed", (False, True))
def test_prepared_rooted_trim_preserves_boxes_and_derivatives(reversed: bool) -> None:
    curve = _branch()
    pcurve = IntersectionPCurve(curve, "first", first=0.0, last=1.0)
    first = TrimRootEndpoint(IntersectionCurvePointRoot(curve, 0.25), "first")
    last = TrimRootEndpoint(IntersectionCurvePointRoot(curve, 0.75), "first")
    segment = CurveTrimSegment(
        pcurve, 0.25, 0.75, reversed=reversed, first_root=first, last_root=last
    )
    nested = CurveTrimSegment(segment, 0.1, 0.9, reversed=True)
    preparation = _preparation()
    for carrier in (segment, nested):
        expected_box = carrier.bounding_box(0.0, 1.0)
        expected_derivatives = carrier.derivative_bounds(0.0, 1.0)
        for prepared in (preparation, _preparation()):
            np.testing.assert_array_equal(
                prepared.curve_box(carrier, 0.0, 1.0), expected_box
            )
            for actual, expected in zip(
                prepared.curve_derivative_bounds(carrier, 0.0, 1.0),
                expected_derivatives,
                strict=True,
            ):
                np.testing.assert_array_equal(actual, expected)


def test_trim_partition_uses_prepared_intersection_pcurve() -> None:
    curve = _branch()
    pcurve = IntersectionPCurve(curve, "first", first=0.0, last=1.0)
    segment = CurveTrimSegment(pcurve, 0.25, 0.75)
    endpoints = np.asarray(segment.evaluate(jnp.asarray((0.0, 1.0), dtype=jnp.float64)))
    start = (
        Fraction.from_float(float(endpoints[0, 0])),
        Fraction.from_float(float(endpoints[0, 1])),
    )
    end = (
        Fraction.from_float(float(endpoints[1, 0])),
        Fraction.from_float(float(endpoints[1, 1])),
    )
    dx, dy = end[0] - start[0], end[1] - start[1]
    origin = (
        Fraction.from_float(float(start[0] - dy)),
        Fraction.from_float(float(start[1] + dx)),
    )
    uses = (
        PatchCurveUse(0, segment, 0.0, 1.0),
        PatchCurveUse(1, ExactLineCurve(end, origin), 0.0, 1.0),
        PatchCurveUse(2, ExactLineCurve(origin, start), 0.0, 1.0),
    )
    domain = MeshingDomain(
        (MeshingSurfacePatch(curve.first.patch, (uses,)),),
        (MeshingDomainCurve(0, 1), MeshingDomainCurve(1, 2), MeshingDomainCurve(2, 0)),
        3,
        source_id="orthogonal-cylinder-trim",
        source_revision="orthogonal-cylinder-trim-source",
        tolerance=1.0e-12,
    )
    source = MeshingDomainBoundarySource(domain, (0,))
    triangles = ((start, end, origin),)
    preparation = SourcePreparation(domain)
    expected = source.trim_partition_reference(0, triangles)
    assert expected[0] and expected[1]
    assert (
        source.trim_partition_reference(0, triangles, preparation=preparation) == expected
    )
    assert (
        source.trim_partition_reference(0, triangles, preparation=preparation) == expected
    )
    assert (
        source.trim_partition_reference(
            0, triangles, preparation=SourcePreparation(domain)
        )
        == expected
    )
