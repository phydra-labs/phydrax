# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""FE geometry compilation follows numerical buckets, not source identities."""

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import numpy as np
import pytest
from jax import Array

from phydrax.discretization import (
    CellBlock,
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementPrecisionPolicy,
    form_element,
    lagrange_element,
    PolyhedralBlock,
)
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    coordinate_lagrange_element,
    coordinate_tabulation,
)
from phydrax.discretization.fem._generic import (
    _prepare_block_geometry,
    FiniteElementBlockGeometry,
)
from phydrax.integration import CoordinateMeasurePlan
from tools.compile_counter import CompileCounter


def _mesh(name: str, epoch: str, cell_count: int, /) -> CellMesh:
    coordinates = np.asarray(
        ((1.0, -0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.5, 0.5)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 4), (1, 2, 4), (2, 3, 4)), dtype=np.int32)
    return CellMesh(
        coordinates,
        (
            CellBlock(
                name,
                "triangle",
                cells[:cell_count],
                global_ids=np.arange(cell_count, dtype=np.int64) + int(epoch) * 10,
            ),
        ),
        vertex_global_ids=np.arange(5, dtype=np.int64) + int(epoch) * 100,
        numeric_version=epoch,
    )


def _assert_bits(left: Array, right: Array, /) -> None:
    left_host, right_host = np.asarray(left), np.asarray(right)
    assert left_host.shape == right_host.shape
    assert left_host.dtype == right_host.dtype
    np.testing.assert_array_equal(
        np.ascontiguousarray(left_host).view(np.uint8),
        np.ascontiguousarray(right_host).view(np.uint8),
    )


def _assert_geometry_bits(
    left: FiniteElementBlockGeometry, right: FiniteElementBlockGeometry, /
) -> None:
    for left_array, right_array in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        _assert_bits(left_array, right_array)


@pytest.mark.parametrize("cell_count", (1, 3), ids=("one-cell", "three-cells"))
@pytest.mark.parametrize(
    "kind", ("scalar", "circulation", "axisymmetric"), ids=("h1", "piola", "rz")
)
def test_block_geometry_reuses_bucket_across_source_metadata(
    cell_count: int, kind: Literal["scalar", "circulation", "axisymmetric"]
) -> None:
    first_mesh = _mesh("source-cells", "0", cell_count)
    next_mesh = _mesh("published-cells", "2", cell_count)
    element = (
        form_element("triangle", 1, 1, proxy="circulation")
        if kind == "circulation"
        else lagrange_element("triangle", 1)
    )
    measure = (
        CoordinateMeasurePlan("axisymmetric-rz", radial_axis=0, frame="current")
        if kind == "axisymmetric"
        else None
    )
    policy = FiniteElementPrecisionPolicy()
    first_block, next_block = first_mesh.blocks[0], next_mesh.blocks[0]
    identities = (
        first_mesh.mesh_id,
        next_mesh.mesh_id,
        first_block.block_id,
        next_block.block_id,
        element.element_id,
    )
    assert first_mesh.mesh_id != next_mesh.mesh_id
    assert first_block.block_id != next_block.block_id
    assert first_mesh.numeric_version != next_mesh.numeric_version
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _prepare_block_geometry(
                first_mesh,
                first_block,
                element,
                precision_policy=policy,
                coordinate_measure=measure,
            )
        )
    with CompileCounter(capture_stack=True) as warm:
        repeated = jax.block_until_ready(
            _prepare_block_geometry(
                next_mesh,
                next_block,
                element,
                precision_policy=policy,
                coordinate_measure=measure,
            )
        )
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    _assert_geometry_bits(first, repeated)
    assert first.block_name == "source-cells"
    assert repeated.block_name == "published-cells"
    assert first.coordinate_measure is repeated.coordinate_measure is measure
    assert identities == (
        first_mesh.mesh_id,
        next_mesh.mesh_id,
        first_block.block_id,
        next_block.block_id,
        element.element_id,
    )
    np.testing.assert_array_equal(next_block.global_ids, np.arange(cell_count) + 20)
    np.testing.assert_array_equal(next_mesh.vertex_global_ids, np.arange(5) + 200)
    assert first_mesh.storage is next_mesh.storage is None


@pytest.mark.parametrize("depth", (1, 2), ids=("single-action", "nested-actions"))
@pytest.mark.parametrize("kind", ("scalar", "circulation"), ids=("h1", "piola"))
def test_barycentric_geometry_ids_do_not_specialize_numerical_bucket(
    depth: int, kind: Literal["scalar", "circulation"]
) -> None:
    mesh = _mesh("source", "0", 1)
    element = (
        form_element("triangle", 1, 1, proxy="circulation")
        if kind == "circulation"
        else lagrange_element("triangle", 1)
    )
    root = coordinate_lagrange_element("triangle", 1)
    first_source = BarycentricCellGeometryElement(root, np.eye(3, dtype=np.float64))
    next_source = BarycentricCellGeometryElement(
        root, np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=np.float64)
    )
    if depth == 2:
        first_source = BarycentricCellGeometryElement(
            first_source, np.eye(3, dtype=np.float64)
        )
        next_source = BarycentricCellGeometryElement(
            next_source, np.eye(3, dtype=np.float64)
        )
    identities = (first_source.element_id, next_source.element_id)
    assert identities[0] != identities[1]
    first_structure = jax.tree.structure(first_source)
    next_structure = jax.tree.structure(next_source)
    first_arrays = tuple(
        np.asarray(leaf).copy() for leaf in jax.tree.leaves(first_source)
    )
    next_arrays = tuple(np.asarray(leaf).copy() for leaf in jax.tree.leaves(next_source))
    # P maps basis columns cyclically; inverse routes retain the same physical map.
    next_routes = jax.device_put(np.asarray(((4, 0, 1),), dtype=np.int32))
    points = jax.device_put(np.asarray(((0.25, 0.25), (0.5, 0.125)), dtype=np.float64))
    weights = jax.device_put(np.asarray((0.125, 0.375), dtype=np.float64))
    policy = FiniteElementPrecisionPolicy()
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _prepare_block_geometry(
                mesh,
                mesh.blocks[0],
                element,
                coordinate_element=first_source,
                reference_points=points,
                reference_weights=weights,
                precision_policy=policy,
            )
        )
    with CompileCounter(capture_stack=True) as warm:
        repeated = jax.block_until_ready(
            _prepare_block_geometry(
                mesh,
                mesh.blocks[0],
                element,
                coordinate_element=next_source,
                geometry_dofs=next_routes,
                reference_points=points,
                reference_weights=weights,
                precision_policy=policy,
            )
        )
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    _assert_geometry_bits(first, repeated)
    assert (first_source.element_id, next_source.element_id) == identities
    assert jax.tree.structure(first_source) == first_structure
    assert jax.tree.structure(next_source) == next_structure
    for source, arrays in ((first_source, first_arrays), (next_source, next_arrays)):
        for current, original in zip(jax.tree.leaves(source), arrays, strict=True):
            np.testing.assert_array_equal(
                np.ascontiguousarray(current).view(np.uint8),
                np.ascontiguousarray(original).view(np.uint8),
            )


def test_barycentric_projection_retains_noncommuting_action_order() -> None:
    mesh = _mesh("source", "0", 1)
    root = coordinate_lagrange_element("triangle", 1)
    inner = np.asarray(((1, 0, 0), (0.25, 0.75, 0), (0, 0, 1)), dtype=np.float64)
    outer = np.asarray(((0, 1, 0), (0, 0, 1), (1, 0, 0)), dtype=np.float64)
    source = BarycentricCellGeometryElement(
        BarycentricCellGeometryElement(root, inner), outer
    )
    points_host = np.asarray(((0.25, 0.25), (0.5, 0.125)), dtype=np.float64)
    points = jax.device_put(points_host)
    weights = jax.device_put(np.asarray((0.125, 0.375), dtype=np.float64))
    projection = coordinate_tabulation(source)
    values, gradients = projection.tabulate(points)
    source_values, source_gradients = source.tabulate(points)
    _assert_bits(values, source_values)
    _assert_bits(gradients, source_gradients)
    cardinal = np.column_stack((1.0 - points_host.sum(axis=1), points_host))
    expected = cardinal @ outer @ inner
    np.testing.assert_array_equal(values, expected)
    assert not np.array_equal(expected, cardinal @ inner @ outer)
    geometry = jax.block_until_ready(
        _prepare_block_geometry(
            mesh,
            mesh.blocks[0],
            lagrange_element("triangle", 1),
            coordinate_element=source,
            reference_points=points,
            reference_weights=weights,
            precision_policy=FiniteElementPrecisionPolicy(),
        )
    )
    np.testing.assert_array_equal(
        geometry.physical_points, (((1.5, 0.125), (1.53125, 0.25)),)
    )
    np.testing.assert_array_equal(geometry.physical_weights, ((0.046875, 0.140625),))
    np.testing.assert_array_equal(geometry.measure, (0.1875,))


@pytest.mark.parametrize(
    "points",
    (
        np.asarray((0.25, 0.25), dtype=np.float64),
        np.asarray(((0.25,),), dtype=np.float64),
    ),
    ids=("rank", "dimension"),
)
def test_barycentric_projection_retains_reference_layout_refusal(
    points: np.ndarray,
) -> None:
    source = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1), np.eye(3, dtype=np.float64)
    )
    for tabulation in (source, coordinate_tabulation(source)):
        with pytest.raises(ValueError, match="FiniteElementSpec.points must have shape"):
            tabulation.tabulate(points)


def test_block_geometry_keeps_coordinates_routes_and_quadrature_dynamic() -> None:
    mesh = _mesh("cells", "0", 3)
    element = lagrange_element("triangle", 1)
    coordinate_element = coordinate_lagrange_element("triangle", 1)
    policy = FiniteElementPrecisionPolicy()
    points = jax.device_put(np.asarray(((0.25, 0.25), (0.5, 0.125)), dtype=np.float64))
    weights = jax.device_put(np.asarray((0.125, 0.375), dtype=np.float64))
    next_coordinates = jax.device_put(np.asarray(mesh.coordinates) * 2.0)
    next_routes = jax.device_put(np.asarray(mesh.blocks[0].vertices)[::-1].copy())
    next_points = jax.device_put(np.asarray(points)[::-1].copy())
    next_weights = jax.device_put(np.asarray(weights)[::-1].copy() * 2.0)
    jax.clear_caches()
    first = jax.block_until_ready(
        _prepare_block_geometry(
            mesh,
            mesh.blocks[0],
            element,
            coordinate_element=coordinate_element,
            reference_points=points,
            reference_weights=weights,
            precision_policy=policy,
        )
    )
    with CompileCounter(capture_stack=True) as warm:
        changed = jax.block_until_ready(
            _prepare_block_geometry(
                mesh,
                mesh.blocks[0],
                element,
                coordinates=next_coordinates,
                geometry_dofs=next_routes,
                coordinate_element=coordinate_element,
                reference_points=next_points,
                reference_weights=next_weights,
                precision_policy=policy,
            )
        )
    assert warm.count == 0, [event.to_record() for event in warm.events]
    # Dyadic points and affine charts yield binary-representable reference values.
    np.testing.assert_array_equal(
        first.basis_values, ((0.5, 0.25, 0.25), (0.375, 0.5, 0.125))
    )
    np.testing.assert_array_equal(
        first.reference_gradients,
        np.tile(np.asarray(((-1.0, -1.0), (1.0, 0.0), (0.0, 1.0))), (2, 1, 1)),
    )
    np.testing.assert_array_equal(
        first.physical_points,
        (
            ((1.375, 0.125), (1.5625, 0.0625)),
            ((1.875, 0.375), (1.9375, 0.5625)),
            ((1.625, 0.875), (1.4375, 0.9375)),
        ),
    )
    gradients = np.asarray(
        (
            ((-1.0, -1.0), (1.0, -1.0), (0.0, 2.0)),
            ((1.0, -1.0), (1.0, 1.0), (-2.0, 0.0)),
            ((1.0, 1.0), (-1.0, 1.0), (0.0, -2.0)),
        ),
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        first.physical_gradients, np.repeat(gradients[:, None], 2, axis=1)
    )
    np.testing.assert_array_equal(
        first.physical_weights, np.tile(np.asarray((0.0625, 0.1875)), (3, 1))
    )
    np.testing.assert_array_equal(first.measure, (0.25, 0.25, 0.25))
    _assert_bits(changed.reference_points, first.reference_points[::-1])
    _assert_bits(changed.reference_weights, first.reference_weights[::-1] * 2.0)
    _assert_bits(changed.basis_values, first.basis_values[::-1])
    _assert_bits(changed.reference_gradients, first.reference_gradients[::-1])
    _assert_bits(changed.physical_points, first.physical_points[::-1, ::-1] * 2.0)
    _assert_bits(changed.physical_gradients, first.physical_gradients[::-1, ::-1] * 0.5)
    _assert_bits(changed.physical_weights, first.physical_weights[::-1, ::-1] * 8.0)
    _assert_bits(changed.measure, first.measure[::-1] * 8.0)


def test_public_geometry_consumers_reuse_distinct_source_bucket() -> None:
    element = lagrange_element("triangle", 1)
    field = FiniteElementFieldSpec("u", element)
    first = FiniteElementPlan(_mesh("first", "0", 3), field).prepare()
    second = FiniteElementPlan(_mesh("second", "2", 3), field).prepare()
    points = jax.device_put(np.asarray(((0.25, 0.25),), dtype=np.float64))
    weights = jax.device_put(np.asarray((0.5,), dtype=np.float64))
    jax.clear_caches()
    expected = jax.block_until_ready(
        first.evaluate_geometry("u", first.default_runtime.coordinates)
    )
    expected_block = jax.block_until_ready(
        first.evaluate_block_geometry(
            "u", 0, first.default_runtime.coordinates, points, weights
        )
    )
    with CompileCounter(capture_stack=True) as warm:
        repeated = jax.block_until_ready(
            second.evaluate_geometry("u", second.default_runtime.coordinates)
        )
        repeated_block = jax.block_until_ready(
            second.evaluate_block_geometry(
                "u", 0, second.default_runtime.coordinates, points, weights
            )
        )
    assert warm.count == 0, [event.to_record() for event in warm.events]
    _assert_geometry_bits(expected[0], repeated[0])
    _assert_geometry_bits(expected_block, repeated_block)
    assert repeated[0].block_name == repeated_block.block_name == "second"


@pytest.mark.parametrize("invalid", ("collapsed", "nonfinite"))
def test_warm_block_geometry_preserves_metric_refusal(invalid: str) -> None:
    mesh = _mesh("cells", "0", 1)
    element = lagrange_element("triangle", 1)
    policy = FiniteElementPrecisionPolicy()
    coordinates = np.asarray(mesh.coordinates).copy()
    if invalid == "collapsed":
        coordinates[4] = coordinates[0]
    else:
        coordinates[4, 0] = np.nan
    invalid_coordinates = jax.device_put(coordinates)
    jax.block_until_ready(
        _prepare_block_geometry(mesh, mesh.blocks[0], element, precision_policy=policy)
    )
    with pytest.raises(
        eqx.EquinoxRuntimeError, match="positive finite metric determinant"
    ):
        jax.block_until_ready(
            _prepare_block_geometry(
                mesh,
                mesh.blocks[0],
                element,
                coordinates=invalid_coordinates,
                precision_policy=policy,
            )
        )


def test_block_geometry_preserves_semantic_guard_order() -> None:
    mesh = _mesh("cells", "0", 1)
    element = lagrange_element("triangle", 1)
    policy = FiniteElementPrecisionPolicy()
    polyhedral = PolyhedralBlock(
        "polyhedral", np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    points = jax.device_put(np.asarray(((0.25, 0.25),), dtype=np.float64))
    with pytest.raises(ValueError, match="points and weights together"):
        _prepare_block_geometry(
            mesh, polyhedral, element, reference_points=points, precision_policy=policy
        )
    with pytest.raises(TypeError, match="fixed-cell block"):
        _prepare_block_geometry(mesh, polyhedral, element, precision_policy=policy)
    with pytest.raises(ValueError, match="requires H1 coordinate elements"):
        _prepare_block_geometry(
            mesh,
            mesh.blocks[0],
            element,
            coordinate_element=form_element("triangle", 1, 1, proxy="circulation"),
            precision_policy=policy,
        )


def test_warm_rotational_geometry_rejects_negative_nodes_with_positive_quadrature() -> (
    None
):
    mesh = CellMesh(
        np.asarray(((0.0,), (2.0,)), dtype=np.float64),
        (CellBlock("radial", "interval", np.asarray(((0, 1),), dtype=np.int32)),),
    )
    element = lagrange_element("interval", 1)
    policy = FiniteElementPrecisionPolicy()
    measure = CoordinateMeasurePlan("spherical-radial", radial_axis=0)
    negative_nodes = jax.device_put(np.asarray(((-0.01,), (2.0,)), dtype=np.float64))
    jax.clear_caches()
    admitted = jax.block_until_ready(
        _prepare_block_geometry(
            mesh,
            mesh.blocks[0],
            element,
            precision_policy=policy,
            coordinate_measure=measure,
        )
    )
    with CompileCounter() as warm:
        refused = jax.block_until_ready(
            _prepare_block_geometry(
                mesh,
                mesh.blocks[0],
                element,
                coordinates=negative_nodes,
                precision_policy=policy,
                coordinate_measure=measure,
            )
        )
    assert warm.count == 0
    assert np.all(np.asarray(admitted.physical_weights) > 0.0)
    assert np.all(np.asarray(refused.physical_points)[..., 0] > 0.0)
    assert np.all(np.isnan(np.asarray(refused.physical_weights)))
    assert np.all(np.isnan(np.asarray(refused.measure)))
    assert refused.coordinate_measure is measure
    assert admitted.reference_points.shape == (3, 1)
