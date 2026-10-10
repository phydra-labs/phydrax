#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _mixed_mesh() -> Any:
    coordinates = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (0.0, 0.0, 2.0),
        )
    )
    prism = phx.discretization.CellBlock(
        "prisms",
        "prism",
        np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32),
        global_ids=np.asarray((101,), dtype=np.int64),
    )
    tetrahedron = phx.discretization.CellBlock(
        "tetrahedra",
        "tetrahedron",
        np.asarray(((6, 3, 5, 4),), dtype=np.int32),
        global_ids=np.asarray((303,), dtype=np.int64),
    )
    return phx.discretization.CellMesh(
        coordinates,
        (prism, tetrahedron),
        numeric_version="mixed-r1",
    )


def _field(*, degree: Any = 1, component_shape: Any = ()) -> Any:
    return phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.lagrange_element("prism", degree),
            "tetrahedra": phx.discretization.lagrange_element("tetrahedron", degree),
        },
        component_shape=component_shape,
    )


def _discretization() -> Any:
    return phx.discretization.FiniteElementPlan(_mixed_mesh(), _field()).prepare()


def test_mixed_prism_contracts() -> None:
    discretization = _discretization()
    mesh = discretization.mesh
    dof_map = discretization.dof_maps[0]

    assert isinstance(mesh.connectivity, phx.discretization.PolyhedralConnectivity)
    assert dof_map.association == "vertex"
    assert dof_map.global_dof_count == 7
    prism_routes = np.asarray(dof_map.cell_dofs[0])[0]
    tetrahedron_routes = np.asarray(dof_map.cell_dofs[1])[0]
    np.testing.assert_array_equal(prism_routes[[3, 4, 5]], (3, 4, 5))
    np.testing.assert_array_equal(tetrahedron_routes[[1, 3, 2]], (3, 4, 5))

    connectivity = mesh.connectivity
    shared_face = int(np.flatnonzero(np.asarray(connectivity.face_neighbor) >= 0)[0])
    start, stop = np.asarray(connectivity.face_vertex_offsets)[
        shared_face : shared_face + 2
    ]
    np.testing.assert_array_equal(
        np.sort(np.asarray(connectivity.face_vertex_values)[start:stop]),
        (3, 4, 5),
    )
    discretization = _discretization()
    mesh = discretization.mesh
    cells = mesh.entity_set(3)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        3,
        cells.entity_set_id,
        np.asarray((303,), dtype=np.int64),
    )
    selection = phx.meshing.resolve_mesh_scope(mesh, scope)
    domain = discretization.integration_domain("cell", selection)
    functional = phx.equations.FiniteElementFunctional(
        "selected-tetrahedron-volume",
        "u",
        lambda values, gradients, points, context: jnp.ones(values.shape[:2]),
        domain=domain,
    )

    np.testing.assert_array_equal(domain.entity_indices, (1,))
    np.testing.assert_allclose(
        functional.evaluate(discretization, jnp.zeros((7,))),
        1.0 / 6.0,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_mixed_quadratic_h1_reproduces_physical_quadratic_in_both_cells() -> None:
    mesh = _mixed_mesh()
    space = phx.discretization.FiniteElementPlan(mesh, _field(degree=2)).prepare()
    coefficients = np.sum(np.asarray(space.dof_maps[0].dof_coordinates) ** 2, axis=1)
    first_route = np.asarray(space.dof_maps[0].cell_dofs[0])[0]
    second_route = np.asarray(space.dof_maps[0].cell_dofs[1])[0]
    first_basis = np.asarray(
        space.elements[0][0].tabulate(np.asarray([[0.2, 0.3, 0.4]], dtype=np.float64))[0]
    )
    second_basis = np.asarray(
        space.elements[0][1].tabulate(np.asarray([[0.2, 0.3, 0.1]], dtype=np.float64))[0]
    )
    np.testing.assert_allclose(
        first_basis @ coefficients[first_route], 0.2**2 + 0.3**2 + 0.4**2, atol=1e-12
    )
    np.testing.assert_allclose(
        second_basis @ coefficients[second_route], 0.1**2 + 0.3**2 + 1.4**2, atol=1e-12
    )


def test_periodic_mixed_prism_tet_p2_retains_common_nodal_traces() -> None:
    D = phx.discretization
    lifted = _mixed_mesh()
    generator = np.eye(4, dtype=np.float64)
    generator[:2, :2] = ((0.0, -1.0), (1.0, 0.0))
    periodic = D.PeriodicMeshTopology(
        lifted,
        D.PeriodicIsometryGroup(generator[None]),
        np.asarray((0, 1, 1, 3, 4, 4, 6), dtype=np.int32),
        np.asarray((0, 0, 1, 0, 0, 1, 0), dtype=np.int32)[:, None],
    )
    mesh = D.CellMesh(
        lifted.coordinates,
        lifted.blocks,
        vertex_global_ids=lifted.vertex_global_ids,
        periodic_topology=periodic,
    )
    space = D.FiniteElementPlan(mesh, _field(degree=2)).prepare()
    # The independent physical polynomial is invariant under the authored
    # rotation and must survive shared prism/tet and quotient nodal traces.
    coefficients = np.sum(np.asarray(space.dof_maps[0].dof_coordinates) ** 2, axis=1)
    prism_route = np.asarray(space.dof_maps[0].cell_dofs[0])[0]
    tet_route = np.asarray(space.dof_maps[0].cell_dofs[1])[0]
    prism_values = np.asarray(
        space.elements[0][0].tabulate(np.asarray(((0.2, 0.3, 0.4),), dtype=np.float64))[0]
    )
    tet_values = np.asarray(
        space.elements[0][1].tabulate(np.asarray(((0.2, 0.3, 0.1),), dtype=np.float64))[0]
    )
    np.testing.assert_allclose(
        prism_values @ coefficients[prism_route], 0.2**2 + 0.3**2 + 0.4**2, atol=1.0e-12
    )
    np.testing.assert_allclose(
        tet_values @ coefficients[tet_route], 0.1**2 + 0.3**2 + 1.4**2, atol=1.0e-12
    )


@pytest.mark.parametrize("degree", (2, 4))
def test_periodic_pyramid_hex_common_quad_and_triangular_traces(degree: int) -> None:
    from phydrax.discretization._reference_cell import reference_cell_topology

    D = phx.discretization
    points = np.concatenate(
        (
            np.asarray(reference_cell_topology("hexahedron").vertices, dtype=np.float64),
            np.asarray(((0.0, 0.0, 2.0),), dtype=np.float64),
        )
    )
    blocks = (
        D.CellBlock(
            "hex",
            "hexahedron",
            np.arange(8, dtype=np.int32)[None],
            global_ids=np.asarray((10,), dtype=np.int64),
        ),
        D.CellBlock(
            "pyramid",
            "pyramid",
            np.asarray(((4, 5, 6, 7, 8),), dtype=np.int32),
            global_ids=np.asarray((11,), dtype=np.int64),
        ),
    )
    lifted = D.CellMesh(points, blocks)
    generator = np.eye(4, dtype=np.float64)
    generator[:2, :2] = ((0.0, -1.0), (1.0, 0.0))
    periodic = D.PeriodicMeshTopology(
        lifted,
        D.PeriodicIsometryGroup(generator[None]),
        np.asarray((0, 1, 2, 1, 4, 5, 6, 5, 8), dtype=np.int32),
        np.asarray((0, 0, 0, 1, 0, 0, 0, 1, 0), dtype=np.int32)[:, None],
    )
    mesh = D.CellMesh(points, blocks, periodic_topology=periodic)
    field = D.FiniteElementFieldSpec(
        "u", {block.name: D.lagrange_element(block.cell_kind, degree) for block in blocks}
    )
    space = D.FiniteElementPlan(mesh, field).prepare()
    coefficients = np.sum(np.asarray(space.dof_maps[0].dof_coordinates) ** 2, axis=1)
    hex_route = np.asarray(space.dof_maps[0].cell_dofs[0])[0]
    pyramid_route = np.asarray(space.dof_maps[0].cell_dofs[1])[0]
    hex_values = np.asarray(
        space.elements[0][0].tabulate(np.asarray(((0.2, 0.3, 0.4),), dtype=np.float64))[0]
    )
    pyramid_values = np.asarray(
        space.elements[0][1].tabulate(np.asarray(((0.4, 0.3, 0.2),), dtype=np.float64))[0]
    )
    np.testing.assert_allclose(
        hex_values @ coefficients[hex_route], 0.2**2 + 0.3**2 + 0.4**2, atol=1.0e-10
    )
    np.testing.assert_allclose(
        pyramid_values @ coefficients[pyramid_route],
        0.3**2 + 0.2**2 + 1.2**2,
        atol=1.0e-10,
    )


def test_mixed_dg_constant_density_integrates_actual_region_volume() -> None:
    mesh = _mixed_mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.discontinuous_element(block.cell_kind, 0)
            for block in mesh.blocks
        },
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    density = phx.equations.FiniteElementFunctional(
        "mixed-dg-density",
        "u",
        lambda values, gradients, points, context: values**2,
    )
    np.testing.assert_allclose(
        density.evaluate(space, jnp.ones((2,), dtype=jnp.float64)), 2.0 / 3.0, atol=1e-12
    )


def test_mixed_h1_components_preserve_distinct_polynomial_values() -> None:
    space = phx.discretization.FiniteElementPlan(
        _mixed_mesh(), _field(degree=2, component_shape=(2,))
    ).prepare()
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    coefficients = np.column_stack((np.sum(points, axis=1), np.sum(points**2, axis=1)))
    first_route = np.asarray(space.dof_maps[0].cell_dofs[0])[0]
    second_route = np.asarray(space.dof_maps[0].cell_dofs[1])[0]
    first_basis = np.asarray(
        space.elements[0][0].tabulate(np.asarray([[0.2, 0.3, 0.4]], dtype=np.float64))[0]
    )
    second_basis = np.asarray(
        space.elements[0][1].tabulate(np.asarray([[0.2, 0.3, 0.1]], dtype=np.float64))[0]
    )
    np.testing.assert_allclose(
        first_basis @ coefficients[first_route], [[0.9, 0.29]], atol=1e-12
    )
    np.testing.assert_allclose(
        second_basis @ coefficients[second_route], [[1.8, 2.06]], atol=1e-12
    )


@pytest.mark.parametrize(
    "degree,twist,proxy", ((1, "untwisted", "circulation"), (2, "twisted", "flux"))
)
def test_mixed_canonical_forms_reconstruct_one_physical_vector_across_interface(
    degree: int,
    twist: Any,
    proxy: Any,
) -> None:
    from phydrax.discretization._cell_geometry import (
        _require_scalar_coordinate_element,
    )
    from phydrax.linalg import (
        DenseLinearOperator,
        FactorizationPolicy,
        factorize,
        RHSLayout,
        SmallLinearSolvePlan,
        solve_small_linear,
    )

    mesh = _mixed_mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.form_element(
                block.cell_kind, degree, 2, family="trimmed", twist=twist, proxy=proxy
            )
            for block in mesh.blocks
        },
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    physical = np.asarray((1.0, 2.0, -0.5), dtype=np.float64)
    coefficients = np.full(space.dof_maps[0].global_dof_count, np.nan)
    small = SmallLinearSolvePlan(3)
    runtime = np.asarray(space.default_runtime.coordinates)
    for element, coordinate, routes, transforms, geometry_routes in zip(
        space.elements[0],
        space.coordinate_elements,
        space.dof_maps[0].cell_dofs,
        space.dof_maps[0].cell_transforms,
        space.coordinate_dofs,
        strict=True,
    ):
        coordinate = _require_scalar_coordinate_element(
            coordinate, "Mixed canonical form fixture"
        )
        basis = element.form_basis
        assert basis is not None
        gradients = np.asarray(coordinate.tabulate(basis.functional_points)[1])
        for route, transform, geometry_route in zip(
            np.asarray(routes),
            np.asarray(transforms),
            np.asarray(geometry_routes),
            strict=True,
        ):
            jacobian = np.einsum("qna,nd->qda", gradients, runtime[geometry_route])
            if degree == 1:
                components = np.einsum("qda,d->qa", jacobian, physical)
            else:
                solved = solve_small_linear(
                    small, jacobian, np.broadcast_to(physical, (len(jacobian), 3))
                )
                assert np.all(np.asarray(solved.successful))
                reference = np.asarray(solved.value) * np.linalg.det(jacobian)[:, None]
                components = np.column_stack(
                    (reference[:, 2], -reference[:, 1], reference[:, 0])
                )
            moments = basis.interpolate(components)
            factor = factorize(
                DenseLinearOperator(jnp.asarray(transform)), FactorizationPolicy("svd")
            )
            local = np.asarray(
                factor.solve(moments[:, None], rhs_layout=RHSLayout((1,))).value
            )[:, 0]
            owned = np.isfinite(coefficients[route])
            np.testing.assert_allclose(
                coefficients[route[owned]], local[owned], atol=2e-11, rtol=2e-11
            )
            coefficients[route] = local
        sample = np.asarray(((0.2, 0.3, 0.4),), dtype=np.float64)
        values = np.asarray(element.tabulate(sample)[0])[0]
        gradients = np.asarray(coordinate.tabulate(sample)[1])
        for route, transform, geometry_route in zip(
            np.asarray(routes),
            np.asarray(transforms),
            np.asarray(geometry_routes),
            strict=True,
        ):
            jacobian = np.einsum("qna,nd->qda", gradients, runtime[geometry_route])[0]
            reference = values.T @ (transform @ coefficients[route])
            if degree == 1:
                solved = solve_small_linear(small, jacobian.T, reference)
                assert np.all(np.asarray(solved.successful))
                reconstructed = np.asarray(solved.value)
            else:
                reconstructed = jacobian @ reference / np.linalg.det(jacobian)
            np.testing.assert_allclose(reconstructed, physical, atol=2e-11, rtol=2e-11)


def test_mixed_raw_vector_element_without_canonical_owner_is_refused() -> None:
    from phydrax.exterior._form_type import FormType, FormValueSpec

    mesh = _mixed_mesh()
    value = FormValueSpec(FormType(3, 1, twist="untwisted"), proxy="components")
    elements = {}
    for block in mesh.blocks:
        scalar = phx.discretization.lagrange_element(block.cell_kind, 1)
        elements[block.name] = phx.discretization.FiniteElementSpec(
            "raw-vector",
            block.cell_kind,
            1,
            scalar.reference_nodes,
            scalar.entity_dofs,
            value_spec=value,
            tabulator=scalar.tabulator,
            tabulator_id=scalar.tabulator_id,
        )
    with pytest.raises(ValueError):
        phx.discretization.FiniteElementPlan(
            mesh, phx.discretization.FiniteElementFieldSpec("u", elements)
        )


def test_prism_form_family_without_owning_reference_contract_is_refused() -> None:
    with pytest.raises(ValueError):
        phx.discretization.form_element("prism", 1, 2, family="full", proxy="circulation")
