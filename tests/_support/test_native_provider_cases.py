"""Scientific contracts of the native provider fixture builders."""

import jax
import numpy as np
import pytest

from phydrax.geometry import ParameterId
from phydrax.meshing import SizeControlStrength, UniformSizeControl
from tests._support.native_provider_cases import (
    curve_spec,
    grid,
    implicit_case,
    plate,
    tee,
)


@pytest.mark.parametrize("count", [5, 9], ids=["five", "nine"])
def test_grid_keeps_three_bounded_physical_coordinate_axes(count: int) -> None:
    prepared = grid(count)
    axes = jax.device_get(
        tuple(axis.point_coordinates for axis in prepared.structured_axes)
    )
    assert len(axes) == 3
    for values in axes:
        host = np.asarray(values)
        assert host.shape == (count,)
        np.testing.assert_array_equal(host[[0, -1]], (-1.4, 1.4))
        assert np.all(np.diff(host) > 0.0)


def test_implicit_case_binds_the_sphere_radius_and_surface_scope() -> None:
    geometry, scope, specification = implicit_case(size_strength=SizeControlStrength.SOFT)
    radius = geometry.schema.index(ParameterId("sphere", "radius"))
    assert float(jax.device_get(geometry.state.values[radius])) == 0.75
    assert specification.scope is scope
    assert (scope.source_id, scope.source_revision, scope.entity_dimension) == (
        "sphere",
        "r1",
        2,
    )
    assert specification.target.ambient_dimension == 3
    assert specification.target.topological_dimension == 2


def test_plate_keeps_the_square_hole_and_embedded_crack() -> None:
    source, specification = plate()
    vertices, offsets = jax.device_get(
        (source.region.vertices, source.region.loop_offsets)
    )
    np.testing.assert_array_equal(offsets, (0, 4, 8))
    np.testing.assert_array_equal(
        np.asarray(vertices)[4:], ((0.8, 0.8), (0.8, 1.2), (1.2, 1.2), (1.2, 0.8))
    )
    assert source.embedded is not None
    np.testing.assert_array_equal(
        jax.device_get(source.embedded.vertices), ((0.2, 0.2), (0.5, 1.7))
    )
    assert specification.scope.source_id == source.source_id == "plate"


def test_tee_curve_spec_keeps_independent_edges_and_requested_size() -> None:
    segments, scope = tee()
    np.testing.assert_array_equal(
        jax.device_get(segments.vertices),
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (-1.0, 0.0)),
    )
    control = UniformSizeControl(scope, 0.25, strength=SizeControlStrength.SOFT)
    specification = curve_spec(scope, (), size_controls=(control,))
    assert specification.scope is scope
    assert len(specification.size_controls) == 1
    assert specification.size_controls[0] is control
    assert specification.junctions == ()
    assert specification.target.cell_families.required == ("interval",)
