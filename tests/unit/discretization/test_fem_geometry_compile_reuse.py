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
    IntegrationDomain,
    lagrange_element,
    PolyhedralBlock,
)
from phydrax.discretization._cell_geometry import (
    BarycentricCellGeometryElement,
    coordinate_lagrange_element,
    coordinate_tabulation,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._local_variational import PreparedLocalRegion
from phydrax.discretization.fem._generic import (
    _prepare_block_geometry,
    FiniteElementBlockGeometry,
)
from phydrax.equations import CompiledFiniteElementProblem
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


def test_composed_tabulation_compiles_once_with_dynamic_chart_coordinates() -> None:
    import jax.numpy as jnp

    from phydrax.discretization._cell_geometry import (
        PolynomialComposedCellGeometryElement,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    source = coordinate_lagrange_element("quadrilateral", 2)
    chart = coordinate_lagrange_element("quadrilateral", 1)
    corners = np.asarray(chart.reference_nodes, dtype=np.int64)
    with coordinate_reference_budget() as budget:
        first = PolynomialComposedCellGeometryElement(
            source, chart, corners, np.full_like(corners, 2), budget=budget
        )
        second = PolynomialComposedCellGeometryElement(
            source,
            chart,
            2 * corners + np.asarray((1, 1)),
            np.full_like(corners, 4),
            budget=budget,
        )
        assert first.degree == second.degree == 2
        assert len(budget.source_composition_degree_cache) == 1
        mixed_corners = corners.copy()
        mixed_corners[:, 0] += mixed_corners[:, 1]
        mixed = PolynomialComposedCellGeometryElement(
            source, chart, mixed_corners, np.full_like(corners, 4), budget=budget
        )
        assert mixed.degree == 4
        assert len(budget.source_composition_degree_cache) == 2
    assert first.element_id != second.element_id
    points = jnp.asarray(((0.125, 0.25), (0.75, 0.625)), dtype=jnp.float64)
    jax.clear_caches()
    with CompileCounter() as cold:
        first_values, first_gradients = jax.block_until_ready(first.tabulate(points))
    with CompileCounter() as warm:
        second_values, second_gradients = jax.block_until_ready(second.tabulate(points))
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    for values, gradients, offset in (
        (first_values, first_gradients, 0.0),
        (second_values, second_gradients, 0.25),
    ):
        expected_values, expected_gradients = source.tabulate(0.5 * points + offset)
        np.testing.assert_allclose(values, expected_values, rtol=1e-14, atol=1e-14)
        np.testing.assert_allclose(
            gradients, 0.5 * expected_gradients, rtol=1e-14, atol=1e-14
        )


@pytest.mark.parametrize("proxy", ("circulation", "flux"))
def test_composed_form_projection_reuses_numerical_chart_bucket(
    proxy: Literal["circulation", "flux"],
) -> None:
    import jax.numpy as jnp

    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.discretization.fem._cell_map import PreparedFiniteElementCellMap
    from tests._support.coordinate_references import coordinate_reference_budget

    source = coordinate_lagrange_element("quadrilateral", 2)
    chart = coordinate_lagrange_element("quadrilateral", 1)
    corners = np.asarray(chart.reference_nodes, dtype=np.int64)
    vertices = np.asarray(reference_cell_topology("quadrilateral").vertices)
    field = FiniteElementFieldSpec(
        "v", form_element(
            "quadrilateral", 1, 1, family="tensor-trimmed", proxy=proxy,
            twist="twisted" if proxy == "flux" else "untwisted",
        )
    )
    spaces = []
    with coordinate_reference_budget() as budget:
        for offset in (0, 1):
            element = PolynomialComposedCellGeometryElement(
                source, chart, 2 * corners + offset,
                np.full_like(corners, 4), budget=budget,
            )
            mesh = CellMesh(
                0.5 * vertices + 0.25 * offset,
                (CellBlock("cells", "quadrilateral", np.arange(4)[None, :]),),
            )
            geometry = CellGeometrySpec(
                {"cells": element},
                {"cells": np.arange(source.local_dof_count)[None, :]},
                source.reference_nodes,
                budget=budget,
            )
            spaces.append(
                FiniteElementPlan(
                    mesh, field, coordinate_spec=geometry, budget=budget
                ).prepare(budget=budget)
            )
    maps = tuple(PreparedFiniteElementCellMap(space, 0) for space in spaces)
    assert maps[0].cell_map_id != maps[1].cell_map_id
    assert maps[0].coordinate_element.element_id != maps[1].coordinate_element.element_id
    assert jax.tree.structure(maps[0].numerical) == jax.tree.structure(maps[1].numerical)

    def constant_vector(points: Array, vector: object) -> Array:
        assert isinstance(vector, Array)
        return jnp.broadcast_to(vector, points.shape)

    vector = jnp.asarray((2.0, 3.0), dtype=jnp.float64)
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(spaces[0].project("v", constant_vector, args=vector))
    with CompileCounter(capture_stack=True) as warm:
        second = jax.block_until_ready(spaces[1].project("v", constant_vector, args=vector))
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    _assert_bits(first, second)


def test_mapped_fv_and_local_regions_reuse_nested_composed_tabulation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import (
        _cell_geometry as geometry_owner,
        UnstructuredFiniteVolumePlan,
    )
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.discretization.fem._local_provider import (
        FiniteElementGeometryActions,
        FiniteElementLocalProvider,
    )
    from tests._support.coordinate_references import coordinate_reference_budget

    source = coordinate_lagrange_element("quadrilateral", 2)
    chart = coordinate_lagrange_element("quadrilateral", 1)
    corners = np.asarray(chart.reference_nodes, dtype=np.int64)
    vertices = np.asarray(reference_cell_topology("quadrilateral").vertices)
    inputs = []
    spaces = []
    with coordinate_reference_budget() as budget:
        for offset in (0, 1):
            inner = PolynomialComposedCellGeometryElement(
                source, chart, 2 * corners + offset,
                np.full_like(corners, 4), budget=budget,
            )
            element = PolynomialComposedCellGeometryElement(
                inner, chart, corners, np.full_like(corners, 2), budget=budget,
            )
            mesh = CellMesh(
                0.25 * vertices + 0.25 * offset,
                (CellBlock("cells", "quadrilateral", np.arange(4)[None, :]),),
            )
            geometry = CellGeometrySpec(
                {"cells": element},
                {"cells": np.arange(source.local_dof_count)[None, :]},
                source.reference_nodes,
                budget=budget,
            )
            inputs.append((UnstructuredFiniteVolumePlan.from_cell_mesh(mesh), geometry))
            spaces.append(
                FiniteElementPlan(
                    mesh, FiniteElementFieldSpec("u", lagrange_element("quadrilateral", 1)),
                    coordinate_spec=geometry, budget=budget,
                ).prepare(budget=budget)
            )
    jax.clear_caches()
    with coordinate_reference_budget() as budget:
        with CompileCounter() as cold:
            first = inputs[0][0].prepare(cell_geometry=inputs[0][1], budget=budget)
        with CompileCounter(capture_stack=True) as warm:
            second = inputs[1][0].prepare(cell_geometry=inputs[1][1], budget=budget)
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    assert first.geometry_id != second.geometry_id
    np.testing.assert_allclose(first.cell_volumes, (1 / 16,), rtol=0, atol=2e-14)
    _assert_bits(first.cell_volumes, second.cell_volumes)
    np.testing.assert_allclose(second.face_measures, 0.25, rtol=0, atol=2e-14)
    np.testing.assert_allclose(
        np.asarray(second.cell_centers) - np.asarray(first.cell_centers),
        ((0.25, 0.25),), rtol=0, atol=2e-14,
    )
    with coordinate_reference_budget() as budget:
        with CompileCounter() as local_cold:
            first_regions = FiniteElementLocalProvider(spaces[0]).prepare_local_regions(
                spaces[0].cell_domain, budget=budget, field_names=("u",),
                maximum_derivative_order=2, kernel_mode="dense",
            )
        with CompileCounter(capture_stack=True) as local_warm:
            second_regions = FiniteElementLocalProvider(spaces[1]).prepare_local_regions(
                spaces[1].cell_domain, budget=budget, field_names=("u",),
                maximum_derivative_order=2, kernel_mode="dense",
            )
    assert local_cold.count > 0
    assert local_warm.count == 0, [event.to_record() for event in local_warm.events]
    # Compare each changed numerical input with the preceding full-element
    # execution, rather than requiring translated maps to round identically.
    with monkeypatch.context() as patch, coordinate_reference_budget() as budget:
        patch.setattr(geometry_owner, "coordinate_tabulation", lambda element: element)
        for actual, (plan, geometry) in zip((first, second), inputs, strict=True):
            oracle = plan.prepare(cell_geometry=geometry, budget=budget)
            for name in ("cell_volumes", "face_measures", "cell_centers", "face_centers"):
                _assert_bits(getattr(actual, name), getattr(oracle, name))
        for actual_regions, space in zip((first_regions, second_regions), spaces, strict=True):
            oracle_regions = FiniteElementLocalProvider(space).prepare_local_regions(
                space.cell_domain, budget=budget, field_names=("u",),
                maximum_derivative_order=2, kernel_mode="dense",
            )
            for actual_region, oracle_region in zip(actual_regions, oracle_regions, strict=True):
                actual_geometry, oracle_geometry = actual_region.geometry_actions, oracle_region.geometry_actions
                assert isinstance(actual_geometry, FiniteElementGeometryActions)
                assert isinstance(oracle_geometry, FiniteElementGeometryActions)
                for name in ("coordinate_basis", "coordinate_gradients", "coordinate_hessians"):
                    _assert_bits(getattr(actual_geometry, name), getattr(oracle_geometry, name))


@pytest.mark.parametrize("split_blocks", (False, True))
@pytest.mark.parametrize("degree", (1, 2))
def test_compiler_consumes_the_original_native_local_budget_at_r_and_r_minus_one(
    split_blocks: bool, degree: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.discretization.fem import _local_provider as provider
    from phydrax.equations import (
        compile_finite_element_problem,
        DiffusionAction,
        FiniteElementForm,
        MassAction,
        SourceAction,
    )
    from phydrax.meshing import MeshingLimits
    from tests._support.coordinate_references import native_meshing_request

    mesh = _mesh("source", "0", 3)
    if split_blocks:
        cells = np.asarray(mesh.blocks[0].vertices)
        mesh = CellMesh(
            mesh.coordinates,
            (
                CellBlock("first", "triangle", cells[:1], global_ids=np.asarray((0,))),
                CellBlock("second", "triangle", cells[1:], global_ids=np.asarray((1, 2))),
            ),
        )
    space = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", degree))
    ).prepare()
    form = FiniteElementForm(
        "explicit-native-local-consumer", "u",
        (MassAction("u", 1.0), DiffusionAction("u", 1.0), SourceAction("u", 1.0)),
    )
    standalone = compile_finite_element_problem(form, space)
    original = provider.FiniteElementLocalProvider.prepare_local_regions
    preparations = []

    def observe(
        self: provider.FiniteElementLocalProvider, domain: IntegrationDomain, /, *,
        field_names: tuple[str, ...], maximum_derivative_order: int, kernel_mode: str,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[PreparedLocalRegion, ...]:
        preparations.append(
            (self.discretization, domain, {"maximum_derivative_order": maximum_derivative_order, "budget": budget})
        )
        return original(
            self, domain, field_names=field_names, maximum_derivative_order=maximum_derivative_order,
            kernel_mode=kernel_mode, budget=budget,
        )

    monkeypatch.setattr(provider.FiniteElementLocalProvider, "prepare_local_regions", observe)

    def no_replacement_ticket(*args: object, **kwargs: object) -> None:
        pytest.fail("An explicit native consumer created a replacement HostBatchTicket")

    monkeypatch.setattr(provider, "HostBatchTicket", no_replacement_ticket)

    def prepare(maximum_work: int) -> tuple[CompiledFiniteElementProblem, int, Array]:
        limits = MeshingLimits(maximum_work_units=maximum_work)
        with native_meshing_request(limits) as (scope, budget, _):
            with budget.bind_ticket(scope.ticket_view()):
                start = len(preparations)
                compiled = compile_finite_element_problem(form, space, budget=budget)
                work = maximum_work - int(scope.remaining().remaining_work_units)
                current = preparations[start:]
                # Mass earns order zero; diffusion must earn order one, which
                # source can reuse. A fresh compilation must earn both again.
                assert [entry[2]["maximum_derivative_order"] for entry in current] == [0, 1]
                assert all(entry[0] is space for entry in current)
                assert all(entry[1] is space.cell_domain for entry in current)
                assert all(entry[2]["budget"] is budget for entry in current)
        assert all(
            not isinstance(value, CoordinateEnclosureBudget)
            for value in jax.tree.leaves(compiled)
        )
        # Compiled numerics remain usable after the preparation ledger settles.
        residual = jax.block_until_ready(compiled.residual(compiled.state_space.zeros()))
        return compiled, work, residual

    _, requirement, reference = prepare(100_000)
    assert 0 < requirement < 100_000
    compiled, consumed, actual = prepare(requirement)
    assert consumed == requirement
    assert compiled.compilation_id == standalone.compilation_id
    _assert_bits(actual, reference)
    _assert_bits(
        actual,
        jax.block_until_ready(standalone.residual(standalone.state_space.zeros())),
    )
    with pytest.raises(CoordinateEnclosureResourceError) as refused:
        prepare(requirement - 1)
    assert refused.value.resource == "coefficient_work"
    assert refused.value.limit == requirement - 1
    assert refused.value.requested == requirement


def test_mixed_prism_tetrahedron_diffusion_uses_paid_actual_local_widths() -> None:
    import jax.numpy as jnp

    from phydrax.discretization._cell_geometry import CellGeometrySpec
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.equations import (
        compile_finite_element_problem,
        DiffusionAction,
        FiniteElementForm,
    )
    from phydrax.meshing import MeshingLimits
    from tests._support.coordinate_references import native_meshing_request

    blocks, vertices, elements, routes, source_points = [], [], {}, {}, []
    vertex_offset = source_offset = 0
    for row, kind in enumerate(("prism", "tetrahedron")):
        shift = np.asarray((2.0 * row, 0.0, 0.0))
        corners = np.asarray(reference_cell_topology(kind).vertices) + shift
        element = coordinate_lagrange_element(kind, 2)
        nodes = np.asarray(element.reference_nodes) + shift

        def physical(points: np.ndarray) -> np.ndarray:
            result = points.copy()
            result[:, 2] += 0.125 * points[:, 0] * points[:, 1]
            return result

        vertices.append(physical(corners))
        source_points.append(physical(nodes))
        blocks.append(CellBlock(
            kind, kind, np.arange(vertex_offset, vertex_offset + len(corners))[None, :],
            global_ids=np.asarray((row,), dtype=np.int64),
        ))
        elements[kind] = element
        routes[kind] = np.arange(source_offset, source_offset + len(nodes))[None, :]
        vertex_offset += len(corners)
        source_offset += len(nodes)
    mesh = CellMesh(np.concatenate(vertices), tuple(blocks))
    geometry = CellGeometrySpec(elements, routes, np.concatenate(source_points))
    field = FiniteElementFieldSpec(
        "u", {kind: lagrange_element(kind, 1) for kind in ("prism", "tetrahedron")}
    )
    space = FiniteElementPlan(mesh, field, coordinate_spec=geometry).prepare()
    with native_meshing_request(MeshingLimits()) as (scope, budget, _):
        with budget.bind_ticket(scope.ticket_view()):
            problem = compile_finite_element_problem(
                FiniteElementForm("mixed-paid-diffusion", "u", (DiffusionAction("u", 1.0),)),
                space, budget=budget,
            )
    size = space.dof_maps[0].global_dof_count
    actual = np.asarray(jax.jacfwd(problem.residual)(jnp.zeros((size,), dtype=jnp.float64)))
    expected = np.zeros((size, size), dtype=np.float64)
    from phydrax.discretization.fem._reference_rules import degree_aware_reference_rule

    for block_index, kind in enumerate(("prism", "tetrahedron")):
        # Independent analytic map: (x,y,z) -> (x,y,z + xy/8).
        # Its determinant is exactly one and no prepared metric enters this oracle.
        points, weights = degree_aware_reference_rule(kind, 2)
        points_ = np.asarray(points)
        jacobian = np.broadcast_to(np.eye(3), (len(points_), 3, 3)).copy()
        jacobian[:, 2, 0] = 0.125 * points_[:, 1]
        jacobian[:, 2, 1] = 0.125 * (points_[:, 0] + 2.0 * block_index)
        _, reference_gradients = lagrange_element(kind, 1).tabulate(points)
        gradients = np.einsum(
            "qir,qrd->qid", np.asarray(reference_gradients), np.linalg.inv(jacobian)
        )
        local = np.einsum("q,qir,qjr->ij", np.asarray(weights), gradients, gradients)[None]
        dofs = space.dof_maps[0].cell_dofs[block_index]
        for indices, values in zip(np.asarray(dofs), local, strict=True):
            expected[np.ix_(indices, indices)] += values
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    assert np.all(np.isfinite(actual))


@pytest.mark.parametrize("change", ("none", "gradient", "hessian", "signed-zero", "gradient-dtype"))
def test_coordinate_table_coalescing_preserves_complete_reference_bundle(
    change: str,
) -> None:
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.discretization.fem._local_provider import (
        _coordinate_table_bank,
        _tabulation_hessians,
    )

    element = coordinate_lagrange_element("triangle", 2)
    tabulation = coordinate_tabulation(element)
    points = jax.device_put(np.asarray(element.reference_nodes))
    values, gradients = tabulation.tabulate(points)
    hessians = _tabulation_hessians(tabulation, points)
    originals = (values, gradients, hessians)
    changed = [np.array(table) for table in originals]
    if change == "gradient":
        changed[1].flat[0] += 0.125
    elif change == "hessian":
        changed[2].flat[0] += 0.125
    elif change == "signed-zero":
        flat = changed[0].reshape(-1)
        position = int(np.flatnonzero(flat == 0.0)[0])
        flat[position] = -flat[position]
    elif change == "gradient-dtype":
        changed[1] = changed[1].astype(np.float32)
    bundles = tuple(
        (original, jax.device_put(other), original)
        for original, other in zip(originals, changed, strict=True)
    )
    counts = (2, 3, 1)

    def merge(
        limit: int, memory: int = 1 << 24,
    ) -> tuple[tuple[np.ndarray, ...], np.ndarray | None, int, int]:
        with HostBatchTicket(1_000_000, bytes_upper=1 << 24, queries_upper=0) as ticket:
            with CoordinateEnclosureBudget(
                ticket, maximum_work_units=limit, maximum_memory_bytes=memory
            ).activate() as budget:
                result, rows = _coordinate_table_bank(bundles, counts, budget=budget)
                return result, rows, budget.work_units, budget.peak_bytes_upper

    if change == "gradient-dtype":
        with pytest.raises(ValueError, match="matching shapes and dtypes"):
            merge(1_000_000)
        return
    result, rows, required, required_memory = merge(1_000_000)
    if change == "none":
        assert rows is None
    else:
        np.testing.assert_array_equal(rows, (0, 0, 1, 1, 1, 0))
    for actual, tables in zip(result, bundles, strict=True):
        if change == "none":
            expected = np.asarray(tables[0])
            assert actual.shape == expected.shape
        else:
            expected = np.stack((np.asarray(tables[0]), np.asarray(tables[1])))
            assert actual.shape[0] == 2
        bits = np.dtype((np.void, actual.dtype.itemsize))
        np.testing.assert_array_equal(actual.view(bits), expected.view(bits))
    admitted, admitted_rows, admitted_work, _ = merge(required)
    np.testing.assert_array_equal(admitted_rows, rows)
    assert admitted_work == required
    for actual, repeated in zip(result, admitted, strict=True):
        bits = np.dtype((np.void, actual.dtype.itemsize))
        np.testing.assert_array_equal(actual.view(bits), repeated.view(bits))
    with pytest.raises(CoordinateEnclosureResourceError) as refusal:
        merge(required - 1)
    assert refusal.value.resource == "coefficient_work"
    _, _, _, admitted_memory = merge(1_000_000, required_memory)
    assert admitted_memory == required_memory
    with pytest.raises(CoordinateEnclosureResourceError):
        merge(1_000_000, required_memory - 1)


def test_heterogeneous_composed_regions_use_indexed_actual_coordinate_bank() -> None:
    import jax.numpy as jnp

    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        PolynomialComposedCellGeometryElement,
    )
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.discretization._reference_cell import reference_cell_topology
    from phydrax.discretization.fem import _local_provider as provider
    from phydrax.equations import (
        compile_finite_element_problem,
        DiffusionAction,
        FiniteElementForm,
    )
    from phydrax.meshing import MeshingLimits
    from tests._support.coordinate_references import (
        coordinate_reference_budget,
        native_meshing_request,
    )

    source = coordinate_lagrange_element("quadrilateral", 2)
    chart = coordinate_lagrange_element("quadrilateral", 1)
    corners = np.asarray(chart.reference_nodes, dtype=np.int64)
    vertices = np.asarray(reference_cell_topology("quadrilateral").vertices)
    elements = {}
    with coordinate_reference_budget() as budget:
        for offset in (0, 1):
            inner = PolynomialComposedCellGeometryElement(
                source, chart, 2 * corners + offset,
                np.full_like(corners, 4), budget=budget,
            )
            elements[str(offset)] = PolynomialComposedCellGeometryElement(
                inner, chart, corners, np.full_like(corners, 2), budget=budget,
            )
        mesh = CellMesh(
            np.concatenate(tuple(0.25 * vertices + 0.25 * offset for offset in (0, 1))),
            tuple(
                CellBlock(
                    str(offset), "quadrilateral", (np.arange(4) + 4 * offset)[None, :],
                    global_ids=np.asarray((offset,), dtype=np.int64),
                )
                for offset in (0, 1)
            ),
        )
        geometry = CellGeometrySpec(
            elements,
            {str(offset): np.arange(source.local_dof_count)[None, :] for offset in (0, 1)},
            source.reference_nodes, budget=budget,
        )
        space = FiniteElementPlan(
            mesh, FiniteElementFieldSpec("u", lagrange_element("quadrilateral", 1)),
            coordinate_spec=geometry, budget=budget,
        ).prepare(budget=budget)
        regions = provider.FiniteElementLocalProvider(space).prepare_local_regions(
            space.cell_domain, budget=budget, field_names=("u",),
            maximum_derivative_order=2, kernel_mode="dense",
        )
        assert len(regions) == 1
        banked = regions[0].geometry_actions
        assert isinstance(banked, provider.FiniteElementGeometryActions)
        np.testing.assert_array_equal(banked.coordinate_table_rows, (0, 1))
        assert banked.coordinate_basis.shape[0] == 2
        assert banked.entity_count == 2
        selector = np.asarray(banked.coordinate_table_rows)
        dense = provider.FiniteElementGeometryActions(
            banked.runtime_layout_id, banked.domain_kind,
            np.asarray(banked.coordinate_basis)[selector],
            np.asarray(banked.coordinate_gradients)[selector],
            banked.coordinate_gathers, banked.reference_weights,
            coordinate_hessians=(
                np.asarray(banked.coordinate_hessians)[selector]
                if banked.coordinate_hessians.size else None
            ),
            coordinate_measure=banked.coordinate_measure,
        )
        runtime = space.default_runtime
        for name in ("points", "physical_weights", "jacobian", "inverse_jacobian", "inverse_hessian"):
            np.testing.assert_allclose(
                np.asarray(getattr(banked.realize(runtime), name)),
                np.asarray(getattr(dense.realize(runtime), name)),
                rtol=1e-13, atol=1e-14,
            )
        tangent = jnp.full_like(runtime.coordinates, 0.125)

        def points_for(
            actions: provider.FiniteElementGeometryActions, coordinates: Array,
        ) -> Array:
            moved = eqx.tree_at(lambda value: value.coordinates, runtime, coordinates)
            return actions.realize(moved).points

        _, banked_jvp = jax.jvp(
            lambda coordinates: points_for(banked, coordinates),
            (runtime.coordinates,), (tangent,),
        )
        _, dense_jvp = jax.jvp(
            lambda coordinates: points_for(dense, coordinates),
            (runtime.coordinates,), (tangent,),
        )
        np.testing.assert_allclose(banked_jvp, dense_jvp, rtol=1e-13, atol=1e-14)

    form = FiniteElementForm("heterogeneous-chart-diffusion", "u", (DiffusionAction("u", 1.0),))

    def compile_with_limit(work: int) -> tuple[CompiledFiniteElementProblem, int]:
        with native_meshing_request(MeshingLimits(maximum_work_units=work)) as (scope, budget, _):
            with budget.bind_ticket(scope.ticket_view()):
                problem = compile_finite_element_problem(form, space, budget=budget)
                used = work - int(scope.remaining().remaining_work_units)
        return problem, used

    problem, required = compile_with_limit(1_000_000)
    admitted, used = compile_with_limit(required)
    assert used == required
    size = space.dof_maps[0].global_dof_count
    zeros = jnp.zeros((size,), dtype=jnp.float64)
    actual = np.asarray(jax.jacfwd(problem.residual)(zeros))
    np.testing.assert_array_equal(
        np.asarray(jax.jacfwd(admitted.residual)(zeros)), actual
    )
    expected = np.zeros_like(actual)
    for dofs, measured in zip(
        space.dof_maps[0].cell_dofs,
        space.evaluate_geometry("u", space.default_runtime.coordinates),
        strict=True,
    ):
        gradients = np.asarray(measured.physical_gradients)
        local = np.einsum(
            "cq,cqir,cqjr->cij", np.asarray(measured.physical_weights), gradients, gradients
        )
        for indices, values in zip(np.asarray(dofs), local, strict=True):
            expected[np.ix_(indices, indices)] += values
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    with pytest.raises(CoordinateEnclosureResourceError):
        compile_with_limit(required - 1)


@pytest.mark.parametrize("with_hessians", (False, True))
def test_indexed_coordinate_stream_reuses_callable_with_dynamic_bank_leaves(
    with_hessians: bool,
) -> None:
    import jax.numpy as jnp

    from phydrax.discretization.fem._local_provider import _realize_coordinate_bank

    basis = jnp.asarray((((0.75, 0.25),), ((0.25, 0.75),)))
    gradients = jnp.asarray(((((-1.0,), (1.0,)),), (((-0.5,), (0.5,)),)))
    hessians = (
        jnp.ones(gradients.shape + (1,), dtype=gradients.dtype)
        if with_hessians else jnp.empty((0,), dtype=gradients.dtype)
    )
    rows = jnp.asarray((0, 0, 1, 1, 1, 0), dtype=jnp.int32)
    coordinates = jnp.arange(12.0).reshape(6, 2, 1)
    jax.clear_caches()
    with CompileCounter() as cold:
        first = jax.block_until_ready(
            _realize_coordinate_bank(basis, gradients, hessians, rows, coordinates)
        )
    changed_basis = basis + 0.125
    changed_rows = rows[::-1]
    with CompileCounter(capture_stack=True) as warm:
        changed = jax.block_until_ready(
            _realize_coordinate_bank(
                changed_basis, gradients, hessians, changed_rows, coordinates
            )
        )
    assert cold.count > 0
    assert warm.count == 0, [event.to_record() for event in warm.events]
    expected = np.einsum(
        "cqi,cid->cqd", np.asarray(changed_basis)[np.asarray(changed_rows)],
        np.asarray(coordinates),
    )
    np.testing.assert_allclose(changed[0], expected, rtol=1e-14, atol=1e-14)
    assert not np.array_equal(first[0], changed[0])
    _, tangent = jax.jvp(
        lambda tables: _realize_coordinate_bank(
            tables, gradients, hessians, rows, coordinates
        )[0],
        (basis,), (jnp.ones_like(basis),),
    )
    np.testing.assert_allclose(
        tangent, np.sum(np.asarray(coordinates), axis=1)[:, None, :],
        rtol=1e-14, atol=1e-14,
    )


def test_p2_diffusion_source_share_paid_regions_without_changing_exact_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Independent barycentric moments, not another compiled FE action oracle."""
    import jax.numpy as jnp

    from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
    from phydrax.discretization.fem import _local_provider as provider
    from phydrax.equations import (
        compile_finite_element_problem,
        DiffusionAction,
        FiniteElementForm,
        SourceAction,
    )
    from phydrax.meshing import MeshingLimits
    from tests._support.coordinate_references import native_meshing_request

    vertices = np.concatenate((np.zeros((1, 3)), np.eye(3), np.full((1, 3), 0.25)))
    cells = np.asarray(((4, 1, 2, 3), (0, 4, 2, 3), (0, 1, 4, 3), (0, 1, 2, 4)))
    mesh = CellMesh(
        vertices,
        tuple(
            CellBlock(str(index), "tetrahedron", cell[None], global_ids=np.asarray((index,)))
            for index, cell in enumerate(cells)
        ),
    )
    space = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("tetrahedron", 2))
    ).prepare()
    form = FiniteElementForm(
        "p2-shared-local-regions", "u",
        (DiffusionAction("u", 1.0), SourceAction("u", -6.0)),
    )
    original = provider.FiniteElementLocalProvider.prepare_local_regions
    calls = []

    def observe(
        self: provider.FiniteElementLocalProvider, domain: IntegrationDomain, /, *,
        field_names: tuple[str, ...], maximum_derivative_order: int, kernel_mode: str,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[PreparedLocalRegion, ...]:
        calls.append((domain, {"maximum_derivative_order": maximum_derivative_order, "budget": budget}))
        return original(
            self, domain, field_names=field_names, maximum_derivative_order=maximum_derivative_order,
            kernel_mode=kernel_mode, budget=budget,
        )

    monkeypatch.setattr(provider.FiniteElementLocalProvider, "prepare_local_regions", observe)

    def compile_with_limit(limit: int) -> tuple[CompiledFiniteElementProblem, int]:
        with native_meshing_request(MeshingLimits(maximum_work_units=limit)) as (scope, budget, _):
            with budget.bind_ticket(scope.ticket_view()):
                start = len(calls)
                problem = compile_finite_element_problem(form, space, budget=budget)
                assert len(calls) == start + 1
                assert calls[-1][0] is space.cell_domain
                assert calls[-1][1]["budget"] is budget
                assert calls[-1][1]["maximum_derivative_order"] == 1
                used = limit - int(scope.remaining().remaining_work_units)
        return problem, used

    problem, required = compile_with_limit(1_000_000)
    admitted, used = compile_with_limit(required)
    assert used == required
    with pytest.raises(CoordinateEnclosureResourceError):
        compile_with_limit(required - 1)

    dofs = space.dof_maps[0]
    size = dofs.global_dof_count
    expected_matrix = np.zeros((size, size))
    expected_rhs = np.zeros(size)
    points = np.asarray(dofs.dof_coordinates)
    for cell, routes in zip(cells, dofs.cell_dofs, strict=True):
        indices = np.asarray(routes)[0]
        corners = vertices[cell]
        affine = np.column_stack((np.ones(4), corners))
        inverse = np.linalg.inv(affine)
        gradients = inverse[1:].T
        barycentric = np.column_stack((np.ones(indices.size), points[indices])) @ inverse
        labels = np.rint(2 * barycentric).astype(int)
        volume = abs(np.linalg.det(corners[1:] - corners[0])) / 6
        # grad(phi) = sum_i lambda_i * coefficients[:, i, :].
        coefficients = np.zeros((indices.size, 4, 3))
        integrals = np.empty(indices.size)
        for row, label in enumerate(labels):
            support = np.flatnonzero(label)
            if support.size == 1:
                vertex = support[0]
                coefficients[row] = -gradients[vertex]
                coefficients[row, vertex] += 4 * gradients[vertex]
                integrals[row] = -volume / 20
            else:
                assert support.size == 2 and np.all(label[support] == 1)
                first, second = support
                coefficients[row, first] = 4 * gradients[second]
                coefficients[row, second] = 4 * gradients[first]
                integrals[row] = volume / 5
        sums = coefficients.sum(axis=1)
        local = volume / 20 * (
            sums @ sums.T + np.einsum("aik,bik->ab", coefficients, coefficients)
        )
        expected_matrix[np.ix_(indices, indices)] += local
        expected_rhs[indices] += -6 * integrals

    zero = jnp.zeros(size, dtype=jnp.float64)
    actual_matrix = np.asarray(jax.jacfwd(problem.residual)(zero))
    actual_rhs = -np.asarray(problem.residual(zero))
    np.testing.assert_allclose(actual_matrix, expected_matrix, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(actual_rhs, expected_rhs, rtol=1e-12, atol=1e-13)
    _assert_bits(admitted.residual(zero), problem.residual(zero))
    exact = np.sum(points**2, axis=1)
    boundary = np.asarray(dofs.boundary_dof_mask)
    free = np.flatnonzero(~boundary)
    assert free.size > 0
    solved = np.linalg.solve(
        actual_matrix[np.ix_(free, free)],
        actual_rhs[free] - actual_matrix[np.ix_(free, np.flatnonzero(boundary))] @ exact[boundary],
    )
    np.testing.assert_allclose(solved, exact[free], rtol=1e-12, atol=1e-13)


def test_local_region_reuse_keeps_domain_field_order_and_kernel_contracts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization.fem import _local_provider as provider
    from phydrax.equations import (
        coefficient,
        DiffusionAction,
        FiniteElementForm,
        SourceAction,
    )
    from phydrax.equations.fem import compile_workset_program, lower_finite_element_form
    from tests._support.coordinate_references import coordinate_reference_budget

    space = FiniteElementPlan(
        _mesh("source", "0", 3),
        tuple(
            FiniteElementFieldSpec(name, lagrange_element("triangle", degree))
            for name, degree in (("u", 2), ("v", 1))
        ),
    ).prepare()
    domain = space.cell_domain
    reversed_domain = eqx.tree_at(
        lambda value: (value.entity_indices, value.owner_cells),
        domain, (domain.entity_indices[::-1], domain.owner_cells[::-1]),
    )
    assert reversed_domain.domain_id == domain.domain_id
    coefficients = {
        name: coefficient(
            np.ones(field.layout.size), location="dof",
            support_id=field.support_id, field_space_id=field.field_space_id,
        )
        for name, field in zip(("u", "v"), space.field_spaces, strict=True)
    }
    form = FiniteElementForm(
        "local-owner-contracts", ("u", "v"),
        (
            DiffusionAction("u", 1.0),
            SourceAction("u", 1.0),
            SourceAction("u", 1.0, domain=reversed_domain, action_id="reversed"),
            SourceAction("v", 1.0, action_id="other-field"),
            SourceAction("u", coefficients["v"], action_id="uv"),
            SourceAction("v", coefficients["u"], action_id="vu"),
        ),
    )
    original = provider.FiniteElementLocalProvider.prepare_local_regions
    calls = []

    def observe(
        self: provider.FiniteElementLocalProvider, selected: IntegrationDomain, /, *,
        field_names: tuple[str, ...], maximum_derivative_order: int, kernel_mode: str,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[PreparedLocalRegion, ...]:
        calls.append((selected, {"field_names": field_names, "budget": budget, "kernel_mode": kernel_mode}))
        return original(
            self, selected, field_names=field_names, maximum_derivative_order=maximum_derivative_order,
            kernel_mode=kernel_mode, budget=budget,
        )

    monkeypatch.setattr(provider.FiniteElementLocalProvider, "prepare_local_regions", observe)
    ir = lower_finite_element_form(form, space)
    for mode in ("auto", "dense", "partial"):
        with coordinate_reference_budget() as budget:
            start = len(calls)
            compile_workset_program(ir, form, space, local_kernel=mode, budget=budget)
            current = calls[start:]
            assert len(current) == 5
            assert [entry[1]["field_names"] for entry in current] == [
                ("u",), ("u",), ("v",), ("u", "v"), ("v", "u"),
            ]
            assert current[0][0] is domain
            assert current[1][0] is reversed_domain
            assert all(entry[1]["budget"] is budget for entry in current)
            expected_mode = "dense" if mode == "auto" else mode
            assert all(entry[1]["kernel_mode"] == expected_mode for entry in current)
    # Capability negotiation remains mandatory, even after an equal-space
    # compilation has earned dense regions.
    with coordinate_reference_budget() as budget, pytest.raises(ValueError):
        compile_workset_program(ir, form, space, local_kernel="not-a-kernel", budget=budget)
