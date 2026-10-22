"""Finite-element reuse, portable tabulation, and stored-input validation."""

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.discretization import (
    discontinuous_element,
    FiniteElementDofMap,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    FiniteElementSpec,
    form_element,
    lagrange_element,
)
from phydrax.discretization._integration_domain import IntegrationDomain
from phydrax.discretization._local_variational import PreparedLocalRegion
from phydrax.discretization.fem._assembly import (
    build_finite_element_dof_routes,
    FormDofLayout,
    lifted_finite_element_dof_layout,
)
from phydrax.discretization.fem._generic import FiniteElementRuntimeData
from phydrax.discretization.fem._reference_types import FiniteElementTabulator
from tests._support.cell_meshes import unit_square
from tools.compile_counter import CompileCounter


# Empirical `test_host_local_selection_preserves_routes_and_dynamic_geometry`: matched max |delta|=2.22045e-16; joint allowance fraction=0.0222045, margin=97.7796%.
_TRANSLATED_LOCAL_POINTS_ATOL = 1e-14


@pytest.mark.parametrize(
    "cell,degree",
    [("triangle", 3), ("quadrilateral", 2), ("prism", 2), ("pyramid", 2)],
    ids=["simplex", "tensor", "prism", "pyramid"],
)
def test_portable_tabulation_retains_numerical_source_leaves(
    cell: str, degree: int
) -> None:
    element = lagrange_element(cell, degree)
    tabulator = element.tabulator
    assert isinstance(tabulator, FiniteElementTabulator)
    leaves = jax.tree_util.tree_leaves(tabulator)
    assert any(isinstance(leaf, jax.Array) for leaf in leaves)
    points = element.reference_nodes
    expected_values, expected_gradients = tabulator.source.tabulate(points)
    values, gradients = element.tabulate(points)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(gradients, expected_gradients)


def test_portable_tabulator_refuses_undeclared_source() -> None:
    with pytest.raises(TypeError, match="source"):
        FiniteElementTabulator(object())  # ty: ignore[invalid-argument-type]


def test_portable_tabulator_refuses_unknown_method() -> None:
    source = lagrange_element("triangle", 1)
    with pytest.raises(TypeError, match="FiniteElementTabulator.method"):
        FiniteElementTabulator(source, "missing")


@pytest.mark.parametrize(
    "cell,degree",
    [
        ("triangle", 1),
        ("triangle", 3),
        ("quadrilateral", 2),
        ("prism", 2),
        ("pyramid", 2),
    ],
    ids=["linear", "simplex", "tensor", "prism", "pyramid"],
)
def test_stored_element_round_trip_preserves_identity_and_values(
    cell: str, degree: int
) -> None:
    element = lagrange_element(cell, degree)
    restored = FiniteElementSpec.from_stored_fields(element)
    assert restored.element_id == element.element_id
    assert jax.tree_util.tree_structure(restored) == jax.tree_util.tree_structure(element)
    expected_values, expected_gradients = element.tabulate(element.reference_nodes)
    values, gradients = restored.tabulate(element.reference_nodes)
    np.testing.assert_array_equal(values, expected_values)
    np.testing.assert_array_equal(gradients, expected_gradients)


def test_stored_element_revalidates_corrupted_reference_nodes() -> None:
    element = lagrange_element("triangle", 1)
    corrupted = eqx.tree_at(
        lambda value: value.reference_nodes,
        element,
        element.reference_nodes.at[0, 0].set(jnp.nan),
    )
    with pytest.raises(ValueError, match="reference_nodes"):
        FiniteElementSpec.from_stored_fields(corrupted)


@pytest.mark.parametrize(
    "mode", ["transfer", "variational"], ids=["transfer", "variational"]
)
def test_reusable_preparation_checks_coordinate_bits(
    mode: Literal["transfer", "variational"],
) -> None:
    mesh = unit_square()
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    plan = FiniteElementPlan(mesh, field)
    prepared = plan.prepare_transfer() if mode == "transfer" else plan.prepare()
    prepared.require_exact_plan(plan)
    geometry = eqx.tree_at(
        lambda value: value.coordinates,
        plan.coordinate_spec,
        plan.coordinate_spec.coordinates.at[0, 0].set(-0.0),
    )
    changed = FiniteElementPlan(mesh, field, coordinate_spec=geometry)
    with pytest.raises(ValueError, match="coordinate_map"):
        prepared.require_exact_plan(changed)


@pytest.mark.parametrize(
    "mode", ["transfer", "variational"], ids=["transfer", "variational"]
)
def test_reusable_preparation_checks_ordered_field_roles(
    mode: Literal["transfer", "variational"],
) -> None:
    mesh = unit_square()
    element = lagrange_element("triangle", 1)
    fields = (FiniteElementFieldSpec("u", element), FiniteElementFieldSpec("v", element))
    plan = FiniteElementPlan(mesh, fields)
    prepared = plan.prepare_transfer() if mode == "transfer" else plan.prepare()
    changed = FiniteElementPlan(mesh, fields[::-1])
    with pytest.raises(ValueError, match=r"\.plan must preserve"):
        prepared.require_exact_plan(changed)


def test_stored_plan_round_trip_preserves_plan_identity() -> None:
    mesh = unit_square()
    plan = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    )
    restored = FiniteElementPlan.from_stored_fields(plan)
    assert restored.plan_id == plan.plan_id
    assert restored.mesh is plan.mesh
    assert restored.fields == plan.fields


def test_stored_plan_revalidates_corrupted_mesh_geometry() -> None:
    mesh = unit_square()
    plan = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    )
    corrupted = eqx.tree_at(
        lambda value: value.mesh.coordinates,
        plan,
        plan.mesh.coordinates.at[0, 0].set(jnp.nan),
    )
    with pytest.raises(ValueError, match="positive finite metric determinant"):
        FiniteElementPlan.from_stored_fields(corrupted)


def _host_owner_compiles(counter: CompileCounter, owners: tuple[str, ...]) -> int:
    """Attribute compiler events to preparation, not nested numerical tabulation."""
    count = 0
    for event in counter.events:
        package_frames = [
            frame
            for frame in event.first_party_stack
            if frame.startswith("phydrax/")
            and not frame.startswith(
                ("phydrax/_strict.py:", "phydrax/_typing_signature.py:")
            )
        ]
        if package_frames:
            path, _, function = package_frames[-1].rsplit(":", 2)
            count += f"{path}:{function}" in owners
    return count


@pytest.mark.parametrize(
    "form_degree,proxy", [(1, "circulation"), (2, "flux")], ids=["hcurl", "hdiv"]
)
def test_shared_entity_moment_cache_reproduces_uncached_transforms(
    form_degree: int, proxy: Literal["circulation", "flux"]
) -> None:
    from phydrax.discretization.fem._form_basis_types import EntityMomentCache

    basis = form_element(
        "hexahedron",
        form_degree,
        2,
        family="tensor-trimmed",
        proxy=proxy,
        twist="untwisted",
    ).form_basis
    assert basis is not None
    rng = np.random.default_rng(7)
    cache = EntityMomentCache()
    orderings = [tuple(range(8))] + [
        tuple(int(value) for value in rng.permutation(64)[:8]) for _ in range(12)
    ]
    for vertices in orderings:
        np.testing.assert_array_equal(
            np.asarray(basis.entity_permutation_matrix(vertices, cache=cache)),
            np.asarray(basis.entity_permutation_matrix(vertices)),
        )
    # Traces are prepared once per local entity, not once per vertex ordering.
    assert len(cache.entities) <= 12 + 6 + 1


@pytest.mark.parametrize("family", ("nodal", "cell", "form"))
def test_host_dof_routes_do_not_compile_or_reorder(
    family: Literal["nodal", "cell", "form"],
) -> None:
    mesh = unit_square(2, cell_global_ids=np.arange(8, dtype=np.int64)[::-1])
    match family:
        case "nodal":
            element = lagrange_element("triangle", 1)
        case "cell":
            element = discontinuous_element("triangle", 1)
        case "form":
            element = form_element("triangle", 1, 1, proxy="circulation")
    resolved = (element,)
    layout = lifted_finite_element_dof_layout(mesh, resolved, ())
    if isinstance(layout, FormDofLayout):
        expected = layout.canonical_routes[0]
    elif family == "cell":
        expected = np.arange(24, dtype=np.int32).reshape((8, 3))
    else:
        expected = np.asarray(mesh.blocks[0].vertices)

    jax.clear_caches()
    with CompileCounter(capture_stack=True) as counter:
        routes, signs, relations = build_finite_element_dof_routes(mesh, resolved, layout)
        jax.block_until_ready((routes, signs, relations))

    assert counter.count == 0
    np.testing.assert_array_equal(routes[0], expected)
    np.testing.assert_array_equal(relations[0].source_indices, expected)
    np.testing.assert_array_equal(
        np.asarray(signs[0]).view(np.uint64),
        np.ones(expected.shape, dtype=np.float64).view(np.uint64),
    )
    assert routes[0].dtype == np.dtype(np.int32)
    assert signs[0].dtype == np.dtype(np.float64)


@pytest.mark.parametrize("family", ("nodal", "form"))
def test_host_dof_transforms_preserve_bits_and_identity_without_compiles(
    family: Literal["nodal", "form"],
) -> None:
    mesh = unit_square(2, cell_global_ids=np.arange(8, dtype=np.int64)[::-1])
    element = (
        lagrange_element("triangle", 2)
        if family == "nodal"
        else form_element("triangle", 1, 1, proxy="circulation")
    )
    layout = lifted_finite_element_dof_layout(mesh, (element,), ())
    expected = (
        layout.canonical_transforms[0]
        if isinstance(layout, FormDofLayout)
        else np.broadcast_to(
            np.eye(element.local_dof_count, dtype=np.float64),
            (8, element.local_dof_count, element.local_dof_count),
        )
    )
    jax.clear_caches()
    with CompileCounter(capture_stack=True) as counter:
        dofs = FiniteElementDofMap(mesh, (element,))
        jax.block_until_ready(dofs)
    assert (
        _host_owner_compiles(
            counter,
            (
                "phydrax/discretization/fem/_generic.py:__init__",
                "phydrax/discretization/fem/_generic.py:<genexpr>",
                "phydrax/discretization/fem/_assembly.py:build_finite_element_dof_routes",
                "phydrax/discretization/fem/_assembly.py:build_finite_element_dof_coordinates",
            ),
        )
        == 0
    )
    np.testing.assert_array_equal(
        np.asarray(dofs.cell_transforms[0]).view(np.uint64),
        np.asarray(expected).view(np.uint64),
    )
    rebuilt = FiniteElementDofMap(mesh, (element,))
    assert rebuilt.dof_map_id == dofs.dof_map_id
    assert jax.tree_util.tree_structure(rebuilt) == jax.tree_util.tree_structure(dofs)


@eqx.filter_jit
def _local_runtime_points(
    region: PreparedLocalRegion, runtime: FiniteElementRuntimeData
) -> jax.Array:
    return region.geometry_actions.realize(runtime).points


@pytest.mark.parametrize("family", ("nodal", "form"))
def test_host_local_selection_preserves_routes_and_dynamic_geometry(
    family: Literal["nodal", "form"],
) -> None:
    mesh = unit_square(2)
    element = (
        lagrange_element("triangle", 2)
        if family == "nodal"
        else form_element("triangle", 0, 1, proxy="scalar")
    )
    prepared = FiniteElementPlan(
        mesh,
        (FiniteElementFieldSpec("u", element), FiniteElementFieldSpec("v", element)),
    ).prepare()
    domain = IntegrationDomain(
        "cell",
        np.asarray((7, 1, 4), dtype=np.int32),
        prepared.cell_domain.support_id,
        prepared.cell_domain.entity_set_id,
        owner_local_entities=np.asarray((31, 11, 21), dtype=np.int32),
        selection_id="host-subset",
    )
    jax.clear_caches()
    with CompileCounter(capture_stack=True) as counter:
        region = prepared.prepare_local_regions(
            domain,
            field_names=("v", "u"),
            maximum_derivative_order=1,
            kernel_mode="dense",
        )[0]
        jax.block_until_ready(region)
    assert (
        _host_owner_compiles(
            counter,
            ("phydrax/discretization/fem/_local_provider.py:_prepare_local_regions",),
        )
        == 0
    )
    selected = np.asarray((1, 4, 7), dtype=np.int32)
    assert region.field_names == ("v", "u")
    np.testing.assert_array_equal(region.entity_indices, selected)
    np.testing.assert_array_equal(region.owner_local_entities, (11, 21, 31))
    for name, gathers in zip(region.field_names, region.field_gathers, strict=True):
        dofs = prepared.dof_maps[prepared.field_index(name)]
        np.testing.assert_array_equal(gathers, np.asarray(dofs.cell_dofs[0])[selected])
    ordered_domain = IntegrationDomain(
        "cell",
        selected,
        domain.support_id,
        domain.entity_set_id,
        owner_local_entities=np.asarray((11, 21, 31), dtype=np.int32),
        selection_id=domain.selection_id,
    )
    replay = prepared.prepare_local_regions(
        ordered_domain,
        field_names=("v", "u"),
        maximum_derivative_order=1,
        kernel_mode="dense",
    )[0]
    assert replay.region_id == region.region_id
    for first, second in zip(
        jax.tree_util.tree_leaves(region),
        jax.tree_util.tree_leaves(replay),
        strict=True,
    ):
        if isinstance(first, jax.Array):
            assert isinstance(second, jax.Array)
            assert np.asarray(first).tobytes() == np.asarray(second).tobytes()

    runtime = prepared.default_runtime
    moved = eqx.tree_at(
        lambda value: value.coordinates, runtime, runtime.coordinates + 0.125
    )
    with pxconfig.context(
        pxconfig.NumericsConfig(transfer_guard_device_to_host="disallow")
    ):
        points = _local_runtime_points(region, runtime)
        translated = _local_runtime_points(region, moved)
        jax.block_until_ready((points, translated))
    np.testing.assert_allclose(
        translated, np.asarray(points) + 0.125, rtol=0, atol=_TRANSLATED_LOCAL_POINTS_ATOL
    )


def test_multiblock_local_geometry_retains_ordered_routes_and_affine_moments() -> None:
    from phydrax.discretization import CellBlock, CellMesh
    from phydrax.discretization.fem._local_provider import FiniteElementGeometryActions

    original = unit_square(2)
    original_block = original.blocks[0]
    blocks = tuple(
        CellBlock(
            f"cohort-{index}",
            "triangle",
            np.asarray(original_block.vertices)[start:stop],
            global_ids=np.asarray(original_block.global_ids)[start:stop],
        )
        for index, (start, stop) in enumerate(((0, 1), (1, 4), (4, 8)))
    )
    mesh = CellMesh(original.coordinates, blocks)
    prepared = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 2))
    ).prepare()
    regions = prepared.prepare_local_regions(
        prepared.cell_domain,
        field_names=("u",),
        maximum_derivative_order=2,
        kernel_mode="dense",
    )
    assert len(regions) == 1
    region = regions[0]
    np.testing.assert_array_equal(region.entity_indices, np.arange(8, dtype=np.int32))
    np.testing.assert_array_equal(
        region.field_gathers[0], np.concatenate(prepared.dof_maps[0].cell_dofs)
    )
    geometry = region.geometry_actions
    assert isinstance(geometry, FiniteElementGeometryActions)
    runtime = prepared.default_runtime
    metric = geometry.realize(runtime)
    corners = np.asarray(mesh.coordinates)[
        np.concatenate(tuple(block.vertices for block in blocks))
    ]
    jacobian = np.swapaxes(corners[:, 1:] - corners[:, :1], -1, -2)
    measure = np.abs(np.linalg.det(jacobian))
    assert geometry.coordinate_basis.ndim == 2
    assert geometry.coordinate_table_rows.size == 0
    expected_points = corners[:, None, 0, :] + np.asarray(geometry.coordinate_basis)[
        None, :, 1:
    ] @ np.swapaxes(jacobian, -1, -2)
    # Affine evaluation and the 2D determinant each have at most eight operations.
    epsilon = np.finfo(np.float64).eps
    gamma = 8 * epsilon / (1 - 8 * epsilon)
    np.testing.assert_allclose(metric.points, expected_points, rtol=gamma, atol=0.0)
    np.testing.assert_allclose(
        np.sum(metric.physical_weights, axis=1), measure / 2, rtol=gamma, atol=0.0
    )
    translated = FiniteElementRuntimeData(
        mesh,
        runtime.coordinates + jnp.asarray((0.125, 0.25), dtype=jnp.float64),
        numeric_version="translated-workset",
    )
    np.testing.assert_allclose(
        geometry.realize(translated).points,
        np.asarray(metric.points) + np.asarray((0.125, 0.25), dtype=np.float64),
        rtol=gamma,
        atol=0.0,
    )
