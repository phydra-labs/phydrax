"""Linked native trim motion, with independent rectangular-prism oracles."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import SpatialCoordinateContract
from phydrax.geometry.brep import (
    brep_box,
    BRepGeometry,
    BRepModel,
    BRepTessellationPolicy,
    LineCurve,
    PlanePatch,
)
from phydrax.geometry.brep._differentiable import (
    _FixedTopologyBRepKernel,
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
    FixedTopologyBRepSource,
)
from phydrax.geometry.brep._patches import AbstractSurfacePatch
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations
from phydrax.geometry.design._schema import (
    DesignState,
    ParameterBinding,
    ParameterCollector,
    ParameterId,
)
from phydrax.units import METER


# Empirical `test_linked_height_moves_native_trims_with_exact_volume_derivative`: matched max |delta|=1.45537e-11; joint allowance fraction=0.00121271, margin=99.8787%.
_LINKED_HEIGHT_FINITE_DIFFERENCE_RTOL = 2e-9
# Empirical `test_linked_height_moves_native_trims_with_exact_volume_derivative`: matched max |delta|=1.77636e-15; joint allowance fraction=0.000136643, margin=99.9863%.
_LINKED_HEIGHT_VOLUME_DERIVATIVE_RTOL = 2e-12
# Empirical `test_linked_height_moves_native_trims_with_exact_volume_derivative`: matched max |delta|=7.10543e-15; joint allowance fraction=0.000142117, margin=99.9858%.
_LINKED_HEIGHT_VOLUME_RTOL = 2e-12
# Empirical `test_linked_height_moves_native_trims_with_exact_volume_derivative`: matched max |delta|=7.10543e-15; joint allowance fraction=0.000142117, margin=99.9858%.
_LINKED_HEIGHT_VOLUME_ATOL = 2e-12
# Empirical `test_linked_height_moves_native_trims_with_exact_volume_derivative`: matched max |delta|=7.10543e-15; joint allowance fraction=6.76739e-05, margin=99.9932%.
_LINKED_HEIGHT_SURFACE_AREA_RTOL = 2e-12


_WIDTH = 2.0
_DEPTH = 3.0
_HEIGHT = 4.0
_HEIGHT_ID = ParameterId("linked-box", "height")


class _BoxHeightMap(AbstractBRepParameterMap):
    broken_trim: bool = eqx.field(static=True)

    def __init__(self, *, broken_trim: bool = False) -> None:
        self.broken_trim = broken_trim

    def compile(
        self, context: ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        geometry = model.geometry
        assert geometry is not None
        assert all(isinstance(patch, PlanePatch) for patch in model.patches)
        assert all(isinstance(curve, LineCurve) for curve in geometry.curves)
        assert all(isinstance(curve, LineCurve) for curve in geometry.pcurves)
        # This is host preparation, not a predicate on a traced design value.
        side_faces = tuple(
            face
            for face, patch in enumerate(model.patches)
            if isinstance(patch, PlanePatch) and np.asarray(patch.second_axis)[2] != 0.0
        )
        side_coedges = tuple(
            coedge
            for face in side_faces
            for loop in geometry.face_loops[face]
            for coedge in loop
        )
        vertical_edges = tuple(
            edge
            for edge, (first, last) in enumerate(geometry.edge_vertices)
            if np.asarray(geometry.vertex_points)[first, 2]
            != np.asarray(geometry.vertex_points)[last, 2]
        )
        height = context.bind(
            _HEIGHT_ID,
            jnp.asarray(_HEIGHT, dtype=jnp.float64),
            role="length",
            unit=METER,
            physical_scale=_HEIGHT,
            bounds=(0.0, None),
        )
        return _BoundBoxHeightMap(
            height=height,
            patches=model.patches,
            geometry=geometry,
            parameter_bounds=model.parameter_bounds,
            side_faces=side_faces,
            side_coedges=side_coedges,
            vertical_edges=vertical_edges,
            broken_trim=self.broken_trim,
        )


class _BoundBoxHeightMap(AbstractBoundBRepParameterMap):
    height: ParameterBinding = eqx.field(static=True)
    patches: tuple[AbstractSurfacePatch, ...]
    geometry: BRepGeometry
    parameter_bounds: Array
    side_faces: tuple[int, ...] = eqx.field(static=True)
    side_coedges: tuple[int, ...] = eqx.field(static=True)
    vertical_edges: tuple[int, ...] = eqx.field(static=True)
    broken_trim: bool = eqx.field(static=True)

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        ratio = self.height.read(state) / _HEIGHT
        spatial_scale = jnp.asarray([1.0, 1.0, ratio], dtype=jnp.float64)
        chart_scale = jnp.asarray([1.0, ratio], dtype=jnp.float64)
        patches: list[AbstractSurfacePatch] = []
        for patch in self.patches:
            assert isinstance(patch, PlanePatch)
            patches.append(
                eqx.tree_at(
                    lambda value: value.origin, patch, patch.origin * spatial_scale
                )
            )
        curves: list[LineCurve] = []
        for curve in self.geometry.curves:
            assert isinstance(curve, LineCurve)
            curves.append(
                eqx.tree_at(
                    lambda value: value.origin, curve, curve.origin * spatial_scale
                )
            )
        pcurves = list(self.geometry.pcurves)
        if not self.broken_trim:
            for coedge in self.side_coedges:
                pcurve = pcurves[coedge]
                assert isinstance(pcurve, LineCurve)
                pcurves[coedge] = eqx.tree_at(
                    lambda value: value.origin, pcurve, pcurve.origin * chart_scale
                )
        ranges = self.geometry.edge_ranges
        for edge in self.vertical_edges:
            ranges = ranges.at[edge].set(ranges[edge] * ratio)
        geometry = eqx.tree_at(
            lambda value: (
                value.vertex_points,
                value.curves,
                value.edge_ranges,
                value.pcurves,
            ),
            self.geometry,
            (
                self.geometry.vertex_points * spatial_scale,
                tuple(curves),
                ranges,
                tuple(pcurves),
            ),
        )
        bounds = self.parameter_bounds
        for face in self.side_faces:
            bounds = bounds.at[face, :, 1].set(bounds[face, :, 1] * ratio)
        return BRepParameterGeometry(
            patches=tuple(patches), geometry=geometry, parameter_bounds=bounds
        )


def _box() -> BRepModel:
    return brep_box(
        (0.0, 0.0, 0.0),
        (_WIDTH, _DEPTH, _HEIGHT),
        coordinate_contract=SpatialCoordinateContract(METER),
        tessellation=BRepTessellationPolicy(linear_deflection=0.25),
        preparations=TrimRequestPreparations(),
    )


def test_linked_height_moves_native_trims_with_exact_volume_derivative() -> None:
    model = _box()
    geometry = FixedTopologyBRepSource(model, parameter_map=_BoxHeightMap()).compile(
        coordinate_unit=METER
    )
    kernel = geometry.kernel
    assert isinstance(kernel, _FixedTopologyBRepKernel)
    assert geometry.schema.parameter_ids == (_HEIGHT_ID,)
    index = geometry.schema.index(_HEIGHT_ID)
    base_area = _WIDTH * _DEPTH
    step = 2.0**-12

    def volume(height: Array) -> Array:
        return kernel.measure(geometry.state.replace_at(index, height))

    volumes: list[float] = []
    for height in (_HEIGHT - step, _HEIGHT, _HEIGHT + step):
        state = geometry.state.replace_at(index, jnp.asarray(height, dtype=jnp.float64))
        realized = kernel.realize(state)
        residual = float(realized.seam_residual)
        bound = float(realized.seam_bound)
        # F-041: evidence is checked separately at every independently realized
        # finite-difference state, not inferred from an unchanged primal mesh.
        assert np.isfinite(residual) and residual >= 0.0
        assert np.isfinite(bound) and bound >= 0.0
        assert residual <= bound
        native = realized.geometry
        assert native is not None
        assert model.geometry is not None
        assert native.edge_vertices == model.geometry.edge_vertices
        assert native.coedge_edges == model.geometry.coedge_edges
        assert native.coedge_senses == model.geometry.coedge_senses
        assert native.face_loops == model.geometry.face_loops
        assert native.shell_orientations == model.geometry.shell_orientations
        np.testing.assert_array_equal(realized.faces, model.mesh_faces)
        assert np.max(np.asarray(realized.vertices)[:, 2]) == pytest.approx(height)
        assert np.max(np.asarray(native.vertex_points)[:, 2]) == pytest.approx(height)
        for face, patch in enumerate(realized.patches):
            assert isinstance(patch, PlanePatch)
            if np.asarray(patch.second_axis)[2] != 0.0:
                assert float(realized.parameter_bounds[face, 1, 1]) == pytest.approx(
                    height
                )
                loop = native.face_loops[face][0]
                trim_heights = [
                    float(
                        native.pcurves[coedge].evaluate(
                            native.edge_ranges[native.coedge_edges[coedge], 0]
                        )[1]
                    )
                    for coedge in loop
                ]
                assert max(trim_heights) == pytest.approx(height)
        value = float(kernel.measure(state))
        volumes.append(value)
        assert value == pytest.approx(
            base_area * height,
            rel=_LINKED_HEIGHT_VOLUME_RTOL,
            abs=_LINKED_HEIGHT_VOLUME_ATOL,
        )
        expected_surface_area = 2.0 * (base_area + (_WIDTH + _DEPTH) * height)
        assert float(kernel.boundary_measure(state)) == pytest.approx(
            expected_surface_area, rel=_LINKED_HEIGHT_SURFACE_AREA_RTOL
        )
        points = jnp.asarray(
            [
                [_WIDTH / 2, _DEPTH / 2, height - step / 4],
                [_WIDTH / 2, _DEPTH / 2, height + step / 4],
            ],
            dtype=jnp.float64,
        )
        np.testing.assert_array_equal(kernel.contains(state, points), [True, False])
    finite_difference = (volumes[2] - volumes[0]) / (2.0 * step)
    automatic = float(jax.grad(volume)(jnp.asarray(_HEIGHT, dtype=jnp.float64)))
    assert finite_difference == pytest.approx(
        base_area, rel=_LINKED_HEIGHT_FINITE_DIFFERENCE_RTOL
    )
    assert automatic == pytest.approx(
        base_area, rel=_LINKED_HEIGHT_VOLUME_DERIVATIVE_RTOL
    )
    assert automatic == pytest.approx(
        finite_difference, rel=_LINKED_HEIGHT_FINITE_DIFFERENCE_RTOL
    )


def test_broken_linked_trim_refuses_with_f041_seam_evidence() -> None:
    geometry = FixedTopologyBRepSource(
        _box(), parameter_map=_BoxHeightMap(broken_trim=True)
    ).compile(coordinate_unit=METER)
    kernel = geometry.kernel
    assert isinstance(kernel, _FixedTopologyBRepKernel)
    index = geometry.schema.index(_HEIGHT_ID)
    state = geometry.state.replace_at(
        index, jnp.asarray(_HEIGHT + 0.25, dtype=jnp.float64)
    )
    realized = kernel.realize(state)
    residual = float(realized.seam_residual)
    bound = float(realized.seam_bound)
    assert np.isfinite(residual) and np.isfinite(bound)
    assert residual > bound
    with pytest.raises(RuntimeError, match="seam_residual.*bound"):
        kernel.measure(state).block_until_ready()
