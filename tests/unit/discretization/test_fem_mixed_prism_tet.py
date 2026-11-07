#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.fem._form_elements import FormProxy, FormTwist


def _mixed_mesh(tetrahedron_kind: str = "tetrahedron") -> Any:
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
        tetrahedron_kind,
        np.asarray(((6, 3, 5, 4),), dtype=np.int32),
        global_ids=np.asarray((303,), dtype=np.int64),
    )
    return phx.discretization.CellMesh(
        coordinates,
        (prism, tetrahedron),
        numeric_version="mixed-r1",
    )


def _field(
    *, degree: Any = 1, component_shape: Any = (), tetrahedron_kind: str = "tetrahedron"
) -> Any:
    return phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.lagrange_element("prism", degree),
            "tetrahedra": phx.discretization.lagrange_element(tetrahedron_kind, degree),
        },
        component_shape=component_shape,
    )


def _assert_form_mass_assembly(
    mesh: Any, elements: Any, dofs: Any, coefficients: np.ndarray
) -> None:
    from phydrax.discretization.fem._generic import (
        _assemble_local_operator,
        _local_mass_tensor,
        _prepare_block_geometry,
    )
    from phydrax.discretization.fem._precision import FiniteElementPrecisionPolicy

    local_masses = tuple(
        _local_mass_tensor(
            _prepare_block_geometry(
                mesh, block, element, precision_policy=FiniteElementPrecisionPolicy()
            )
        )
        for block, element in zip(mesh.blocks, elements, strict=True)
    )
    mass = _assemble_local_operator(
        dofs, local_masses, "mixed-form-mass", positive_definite=True
    )
    expected_energy = 0.0
    for index, local_mass in enumerate(local_masses):
        for cell in range(local_mass.shape[0]):
            route = np.asarray(dofs.cell_dofs[index][cell])
            transform = np.asarray(dofs.cell_transforms[index][cell])
            local = transform @ coefficients[route]
            expected_energy += float(local @ np.asarray(local_mass[cell]) @ local)
    assembled_energy = float(
        coefficients @ np.asarray(mass.mv(jnp.asarray(coefficients)))
    )
    assert expected_energy > 0.0
    np.testing.assert_allclose(assembled_energy, expected_energy, rtol=1e-10, atol=1e-10)


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


def test_mixed_quadratic_trace_and_assembly() -> None:
    mesh = _mixed_mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh, _field(degree=2)
    ).prepare()
    dofs = discretization.dof_maps[0]
    prism, tetrahedron = (np.asarray(route[0]) for route in dofs.cell_dofs)
    assert dofs.global_dof_count == 22
    shared = np.intersect1d(prism, tetrahedron)
    assert shared.size == 6
    np.testing.assert_allclose(np.asarray(dofs.dof_coordinates)[shared, 2], 1.0)
    functional = phx.equations.FiniteElementFunctional(
        "linear-integral", "u", lambda values, gradients, points, context: values
    )
    coefficients = jnp.sum(dofs.dof_coordinates, axis=-1)
    np.testing.assert_allclose(
        functional.evaluate(discretization, coefficients),
        7.0 / 8.0,
        rtol=1e-11,
        atol=1e-11,
    )


def test_mixed_simplex_prism_quadratic_trace_and_forms() -> None:
    mesh = _mixed_mesh("simplex:3")
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.lagrange_element("prism", 2),
            "tetrahedra": phx.discretization.lagrange_element("simplex:3", 2),
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    prism, simplex = (
        np.asarray(route[0]) for route in discretization.dof_maps[0].cell_dofs
    )
    assert discretization.dof_maps[0].global_dof_count == 22
    assert np.intersect1d(prism, simplex).size == 6
    for degree, twist, proxy in ((1, "untwisted", "circulation"), (2, "twisted", "flux")):
        form = phx.discretization.FiniteElementFieldSpec(
            "form",
            {
                "prisms": phx.discretization.form_element(
                    "prism", degree, 2, family="prism-trimmed", twist=twist, proxy=proxy
                ),
                "tetrahedra": phx.discretization.form_element(
                    "simplex:3", degree, 2, family="trimmed", twist=twist, proxy=proxy
                ),
            },
        )
        prepared = phx.discretization.FiniteElementPlan(mesh, form).prepare()
        values = jnp.sin(
            jnp.arange(prepared.dof_maps[0].global_dof_count, dtype=jnp.float64) + 0.17
        )
        assert float(values @ prepared.mass.mv(values)) > 0.0


@pytest.mark.parametrize("explicit", (False, True))
def test_mixed_simplex_prism_jit_diffusion_preserves_prepared_rules(
    explicit: bool,
) -> None:
    mesh = _mixed_mesh("simplex:3")
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.lagrange_element("prism", 2),
            "tetrahedra": phx.discretization.lagrange_element("simplex:3", 2),
        },
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    rules = (
        {
            "prisms": phx.integration.ReferencePrismRule(
                phx.integration.GaussLegendreRule(4)
            ),
            "tetrahedra": phx.integration.CubatureRule("simplex", 5, dimension=3),
        }
        if explicit
        else {}
    )
    action = phx.equations.DiffusionAction("u", 1.0, rules=rules)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("mixed-simplex-rule", "u", (action,)), space
    )
    values = jnp.linspace(-0.4, 0.8, space.dof_maps[0].global_dof_count)
    eager = problem.full_residual(values, None)
    compiled = jax.jit(lambda state: problem.full_residual(state, None))(values)
    np.testing.assert_allclose(compiled, eager, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("degree", "twist", "proxy"),
    (
        (1, "untwisted", "circulation"),
        (2, "twisted", "flux"),
    ),
)
def test_chunked_piola_values_and_derivatives_match_full_batch(
    monkeypatch: pytest.MonkeyPatch,
    degree: int,
    twist: FormTwist,
    proxy: FormProxy,
) -> None:
    from phydrax.discretization.fem import _generic
    from phydrax.exterior import map_reference_values

    element = phx.discretization.form_element(
        "prism",
        degree,
        2,
        family="prism-trimmed",
        twist=twist,
        proxy=proxy,
    )
    points = jnp.asarray(((0.2, 0.1, 0.3), (0.1, 0.3, 0.7)), dtype=jnp.float64)
    values = element.tabulate(points)[0]
    jacobians = jnp.broadcast_to(
        jnp.asarray(((1.1, 0.2, 0.0), (0.0, 0.9, 0.1), (0.0, 0.0, 1.2))),
        (7, len(points), 3, 3),
    )
    monkeypatch.setattr(_generic, "_MAPPED_GRADIENT_ELEMENT_BUDGET", 30)

    def full(jacobian: Any) -> Any:
        return map_reference_values(
            values[None], element.value_spec, jacobian[:, :, None]
        )

    def chunked(jacobian: Any) -> Any:
        return _generic._row_chunked_map(full, jacobian)

    np.testing.assert_allclose(
        chunked(jacobians), full(jacobians), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        jax.grad(lambda value: jnp.sum(chunked(value)))(jacobians),
        jax.grad(lambda value: jnp.sum(full(value)))(jacobians),
        rtol=1e-12,
        atol=1e-12,
    )


def test_mixed_quadratic_quad_trace() -> None:
    base = _mixed_mesh()
    coordinates = np.concatenate(
        (np.asarray(base.coordinates), np.asarray(((0, -1, 0), (0, -1, 1)))),
        axis=0,
    )
    prisms = phx.discretization.CellBlock(
        "prisms",
        "prism",
        np.asarray(((0, 1, 2, 3, 4, 5), (1, 0, 7, 4, 3, 8)), dtype=np.int32),
        global_ids=np.asarray((101, 102), dtype=np.int64),
    )
    mesh = phx.discretization.CellMesh(coordinates, (prisms, base.blocks[1]))
    discretization = phx.discretization.FiniteElementPlan(
        mesh, _field(degree=2)
    ).prepare()
    routes = np.asarray(discretization.dof_maps[0].cell_dofs[0])
    shared = np.intersect1d(routes[0], routes[1])
    assert shared.size == 9
    np.testing.assert_allclose(
        np.asarray(discretization.dof_maps[0].dof_coordinates)[shared, 1], 0.0
    )


def test_mixed_discontinuous_and_incompatible_traces() -> None:
    mesh = _mixed_mesh()
    discontinuous = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.discontinuous_element("prism", 1),
            "tetrahedra": phx.discretization.discontinuous_element("tetrahedron", 1),
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, discontinuous).prepare()
    assert discretization.dof_maps[0].global_dof_count == 10
    mismatch = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            "prisms": phx.discretization.lagrange_element("prism", 2),
            "tetrahedra": phx.discretization.lagrange_element("tetrahedron", 1),
        },
    )
    with pytest.raises(ValueError, match="incompatible nodal coordinates"):
        phx.discretization.FiniteElementPlan(mesh, mismatch)


def test_mixed_curved_geometry_requires_matching_facet_charts() -> None:
    mesh = _mixed_mesh()
    field = _field(degree=2)
    affine = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    dofs = affine.dof_maps[0]
    points = np.asarray(dofs.dof_coordinates).copy()
    points[:, 2] += 0.05 * points[:, 0] * points[:, 1]
    elements = dict(zip(("prisms", "tetrahedra"), field.resolve(mesh), strict=True))
    routes = dict(zip(("prisms", "tetrahedra"), dofs.cell_dofs, strict=True))
    geometry = phx.discretization.CellGeometrySpec(elements, routes, points)
    curved = phx.discretization.FiniteElementPlan(
        mesh, field, coordinate_spec=geometry
    ).prepare()
    np.testing.assert_allclose(curved.dof_maps[0].dof_coordinates, points, atol=1e-11)
    broken_routes = {name: np.asarray(route).copy() for name, route in routes.items()}
    shared = np.intersect1d(broken_routes["prisms"], broken_routes["tetrahedra"])[0]
    target = broken_routes["tetrahedra"]
    target[target == shared] = points.shape[0]
    equivalent = phx.discretization.CellGeometrySpec(
        elements, broken_routes, np.concatenate((points, points[shared][None]), axis=0)
    )
    phx.discretization.FiniteElementPlan(mesh, field, coordinate_spec=equivalent)
    displaced = points[shared].copy()
    displaced[2] += 0.01
    broken = phx.discretization.CellGeometrySpec(
        elements, broken_routes, np.concatenate((points, displaced[None]), axis=0)
    )
    with pytest.raises(ValueError, match="charts disagree on a shared facet"):
        phx.discretization.FiniteElementPlan(mesh, field, coordinate_spec=broken)


@pytest.mark.parametrize(
    ("degree", "twist", "proxy"),
    ((1, "untwisted", "circulation"), (2, "twisted", "flux")),
)
def test_mixed_form_plan_assembles_and_rejects_identity_mismatch(
    degree: int, twist: FormTwist, proxy: FormProxy
) -> None:
    mesh = _mixed_mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "circulation",
        {
            "prisms": phx.discretization.form_element(
                "prism", degree, 2, family="prism-trimmed", twist=twist, proxy=proxy
            ),
            "tetrahedra": phx.discretization.form_element(
                "tetrahedron", degree, 2, family="trimmed", twist=twist, proxy=proxy
            ),
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    dofs = discretization.dof_maps[0]
    coefficients = jnp.sin(jnp.arange(dofs.global_dof_count, dtype=jnp.float64) + 0.17)
    assert float(coefficients @ discretization.mass.mv(coefficients)) > 0.0
    inconsistent = phx.discretization.FiniteElementFieldSpec(
        "circulation",
        {
            "prisms": phx.discretization.form_element(
                "prism",
                1,
                2,
                family="prism-trimmed",
                twist="untwisted",
                proxy="circulation",
            ),
            "tetrahedra": phx.discretization.form_element(
                "tetrahedron",
                1,
                2,
                family="trimmed",
                twist="twisted",
                proxy="circulation",
            ),
        },
    )
    with pytest.raises(ValueError, match="scientific form identity"):
        phx.discretization.FiniteElementPlan(mesh, inconsistent)


@pytest.mark.parametrize(
    ("degree", "twist", "proxy"),
    ((1, "untwisted", "circulation"), (2, "twisted", "flux")),
)
def test_mixed_form_entity_routing_matches_triangular_trace(
    degree: int, twist: FormTwist, proxy: FormProxy
) -> None:
    mesh = _mixed_mesh()
    elements = (
        phx.discretization.form_element(
            "prism", degree, 2, family="prism-trimmed", twist=twist, proxy=proxy
        ),
        phx.discretization.form_element(
            "tetrahedron", degree, 2, family="trimmed", twist=twist, proxy=proxy
        ),
    )
    dofs = phx.discretization.FiniteElementDofMap(mesh, elements)
    from phydrax.discretization.fem._de_rham import _exact_form_cell_transform

    for index, block in enumerate(mesh.blocks):
        exact = _exact_form_cell_transform(
            mesh, block, elements[index], 0, np.asarray(mesh.vertex_global_ids)
        )
        execution = np.asarray(dofs.cell_transforms[index][0])
        np.testing.assert_array_equal(np.asarray(exact, dtype=execution.dtype), execution)
    coefficients = np.arange(dofs.global_dof_count, dtype=np.float64) / 17.0
    prism_points = jnp.asarray(((0.2, 0.2, 1.0), (0.6, 0.1, 1.0)))
    tetra_points = jnp.asarray(((0.6, 0.2, 0.2), (0.3, 0.1, 0.6)))
    values = []
    for index, points in enumerate((prism_points, tetra_points)):
        routes = np.asarray(dofs.cell_dofs[index][0])
        transform = np.asarray(dofs.cell_transforms[index][0])
        reference = np.asarray(elements[index].tabulate(points)[0])
        values.append(np.einsum("qic,i->qc", reference, transform @ coefficients[routes]))
    vertices = np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[1].vertices[0])]
    jacobian = (vertices[1:] - vertices[0]).T
    if degree == 1:
        tetra_physical = values[1] @ np.linalg.inv(jacobian)
        np.testing.assert_allclose(values[0][:, :2], tetra_physical[:, :2], atol=1e-10)
    else:
        tetra_physical = values[1] @ jacobian.T / abs(np.linalg.det(jacobian))
        np.testing.assert_allclose(values[0][:, 2], tetra_physical[:, 2], atol=1e-10)
    _assert_form_mass_assembly(mesh, elements, dofs, coefficients)


@pytest.mark.parametrize(
    ("degree", "twist", "proxy"),
    ((1, "untwisted", "circulation"), (2, "twisted", "flux")),
)
def test_mixed_form_entity_routing_matches_quadrilateral_trace(
    degree: int, twist: FormTwist, proxy: FormProxy
) -> None:
    base = _mixed_mesh()
    coordinates = np.concatenate(
        (np.asarray(base.coordinates), np.asarray(((0, -1, 0), (0, -1, 1)))),
        axis=0,
    )
    prisms = phx.discretization.CellBlock(
        "prisms",
        "prism",
        np.asarray(((0, 1, 2, 3, 4, 5), (1, 0, 7, 4, 3, 8)), dtype=np.int32),
        global_ids=np.asarray((101, 102), dtype=np.int64),
    )
    mesh = phx.discretization.CellMesh(coordinates, (prisms, base.blocks[1]))
    elements = (
        phx.discretization.form_element(
            "prism", degree, 2, family="prism-trimmed", twist=twist, proxy=proxy
        ),
        phx.discretization.form_element(
            "tetrahedron", degree, 2, family="trimmed", twist=twist, proxy=proxy
        ),
    )
    dofs = phx.discretization.FiniteElementDofMap(mesh, elements)
    coefficients = np.arange(dofs.global_dof_count, dtype=np.float64) / 19.0
    values = []
    for cell, point in enumerate(((0.35, 0.0, 0.4), (0.65, 0.0, 0.4))):
        route = np.asarray(dofs.cell_dofs[0][cell])
        transform = np.asarray(dofs.cell_transforms[0][cell])
        reference = np.asarray(elements[0].tabulate(jnp.asarray((point,)))[0])
        values.append(np.einsum("qic,i->qc", reference, transform @ coefficients[route]))
    vertices = coordinates[np.asarray(prisms.vertices[1])]
    jacobian = np.stack(
        (vertices[1] - vertices[0], vertices[2] - vertices[0], vertices[3] - vertices[0]),
        axis=1,
    )
    if degree == 1:
        physical = values[1] @ np.linalg.inv(jacobian)
        np.testing.assert_allclose(values[0][:, (0, 2)], physical[:, (0, 2)], atol=1e-10)
    else:
        physical = values[1] @ jacobian.T / abs(np.linalg.det(jacobian))
        np.testing.assert_allclose(values[0][:, 1], physical[:, 1], atol=1e-10)
    _assert_form_mass_assembly(mesh, elements, dofs, coefficients)


def test_prism_native_partition_transfer_refine_coarsen(tmp_path: Path) -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
        CellGeometryTransitionEvidence,
    )
    from phydrax.discretization._nested_reference_types import (
        ReferenceCellMapWitnesses,
        ReferenceCellPartition,
    )
    from phydrax.discretization._reference_cell import reference_cell_topology

    corners = tuple(
        tuple(Fraction(int(value)) for value in point)
        for point in reference_cell_topology("prism").vertices
    )
    children = tuple(
        tuple((x, y, z / 2 + offset) for x, y, z in corners)
        for offset in (Fraction(0), Fraction(1, 2))
    )
    coarse = phx.discretization.CellMesh(
        np.asarray(corners, dtype=float),
        (
            phx.discretization.CellBlock(
                "prisms", "prism", ((0, 1, 2, 3, 4, 5),), global_ids=(41,)
            ),
        ),
    )
    fine_points = np.asarray(
        tuple((x, y, z) for z in (0.0, 0.5, 1.0) for x, y in ((0, 0), (1, 0), (0, 1))),
        dtype=float,
    )
    fine = phx.discretization.CellMesh(
        fine_points,
        (
            phx.discretization.CellBlock(
                "prisms",
                "prism",
                ((0, 1, 2, 3, 4, 5), (3, 4, 5, 6, 7, 8)),
                global_ids=(0, 1),
            ),
        ),
    )
    coarse_geometry = phx.discretization.CellGeometrySpec.affine(coarse)
    fine_geometry = phx.discretization.CellGeometrySpec.affine(fine)
    witness = ReferenceCellMapWitnesses(
        np.asarray((0, 1)),
        np.asarray((41, 41)),
        "prism",
        "prism",
        np.asarray(children, dtype=float),
        children,
    )
    refinement = CellGeometryTransition(
        coarse,
        coarse_geometry,
        fine,
        fine_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", reference_map_count=2
        ),
        reference_map=witness,
    )
    # Persist the actual endpoint meshes, geometry and exact reference maps,
    # then reconstruct and independently reprove them in a fresh process.
    import subprocess
    import sys

    from phydrax.meshing._model_codec import (
        model_archive_content_id,
        MODEL_ARCHIVE_LIMITS,
        write_model_artifact,
    )

    path = tmp_path / "actual-prism-transition.phx"
    write_model_artifact(path, refinement, limits=MODEL_ARCHIVE_LIMITS)
    content_id = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from phydrax.discretization._cell_geometry_transition_types import CellGeometryTransition
from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value
t = read_model_artifact(sys.argv[1], expected_type=CellGeometryTransition,
    restore=restore_model_archive_value, expected_content_id=sys.argv[2], limits=MODEL_ARCHIVE_LIMITS)
assert t.transition_id == sys.argv[3]
assert t.source_mesh.mesh_id == sys.argv[4] and t.target_mesh.mesh_id == sys.argv[5]
""",
            str(path),
            content_id,
            refinement.transition_id,
            coarse.mesh_id,
            fine.mesh_id,
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=900,
    )
    coarsening = CellGeometryTransition(
        fine,
        fine_geometry,
        coarse,
        coarse_geometry,
        evidence=CellGeometryTransitionEvidence(
            "parent_restoration", coarsening_map_count=2
        ),
        coarsening_partition=ReferenceCellPartition((witness,)),
    )
    source = phx.discretization.FiniteElementDeRhamComplex(
        coarse, family="prism-trimmed", order=1, coordinate_spec=coarse_geometry
    )
    refined = phx.discretization.FiniteElementDeRhamComplex(
        fine, family="prism-trimmed", order=1, coordinate_spec=fine_geometry
    )
    prolong_result = source.transfer(refined, geometry_transition=refinement)
    restrict_result = refined.transfer(source, geometry_transition=coarsening)
    prolong = prolong_result.complex_map
    restrict = restrict_result.complex_map
    assert len(restrict_result.support_report()) == 4
    for degree in range(4):
        size = source.hilbert_complex().space(degree).size
        values = jnp.sin(jnp.arange(size, dtype=jnp.float64) + 0.37)
        restored = restrict.map(degree).mv(prolong.map(degree).mv(values))
        np.testing.assert_allclose(restored, values, rtol=1e-10, atol=1e-10)

    from phydrax.discretization.fem._topology_transfer import nested_nodal_transfer

    fields = (
        phx.discretization.FiniteElementFieldSpec(
            "temperature", phx.discretization.lagrange_element("prism", 2)
        ),
        phx.discretization.FiniteElementFieldSpec(
            "density", phx.discretization.discontinuous_element("prism", 1)
        ),
    )
    coarse_fields = phx.discretization.FiniteElementPlan(
        coarse, fields, coordinate_spec=coarse_geometry
    ).prepare()
    fine_fields = phx.discretization.FiniteElementPlan(
        fine, fields, coordinate_spec=fine_geometry
    ).prepare()
    for name, semantics, index in (
        ("temperature", "interpolate", 0),
        ("density", "density", 1),
    ):
        forward = nested_nodal_transfer(
            coarse_fields, fine_fields, refinement, name, semantics=semantics
        )
        backward = nested_nodal_transfer(
            fine_fields, coarse_fields, coarsening, name, semantics=semantics
        )
        values = 1.0 + jnp.sum(coarse_fields.dof_maps[index].dof_coordinates, axis=-1)
        restored = backward.apply(forward.apply(values))
        np.testing.assert_allclose(restored, values, rtol=1e-10, atol=1e-10)
        assert forward.conservative == (semantics == "density")
        assert backward.conservative == (semantics == "density")
        distributed = forward.distributed(
            np.zeros((forward.source_size,), dtype=np.int32),
            np.zeros((forward.target_size,), dtype=np.int32),
            1,
        )
        assert distributed.source_size == forward.source_size
        assert distributed.target_size == forward.target_size
        report = backward.support_report()
        assert np.all(report.target_supported)
        assert (
            0
            <= report.rank_lower_bound
            <= report.rank_upper_bound
            <= backward.target_size
        )
        if report.exact_rank is not None:
            assert report.rank_lower_bound == report.rank_upper_bound == report.exact_rank
    density_restriction = nested_nodal_transfer(
        fine_fields, coarse_fields, coarsening, "density", semantics="density"
    )
    fine_density = jnp.sin(jnp.arange(fine_fields.dof_maps[1].global_dof_count) + 0.41)
    coarse_density = density_restriction.apply(fine_density)
    content = phx.equations.FiniteElementFunctional(
        "density-content", "density", lambda values, gradients, points, context: values
    )
    np.testing.assert_allclose(
        content.evaluate(fine_fields, fine_density),
        content.evaluate(coarse_fields, coarse_density),
        rtol=1e-10,
        atol=1e-10,
    )
    scoped = nested_nodal_transfer(
        fine_fields,
        coarse_fields,
        coarsening,
        "density",
        semantics="density",
        source_cell_classes=np.asarray((2, 2)),
        target_cell_classes=np.asarray((2,)),
    )
    np.testing.assert_allclose(scoped.apply(fine_density), coarse_density, atol=1e-10)
    with pytest.raises(ValueError, match="crosses a declared region"):
        nested_nodal_transfer(
            fine_fields,
            coarse_fields,
            coarsening,
            "density",
            semantics="density",
            source_cell_classes=np.asarray((2, 3)),
            target_cell_classes=np.asarray((2,)),
        )
    intensive = nested_nodal_transfer(
        fine_fields, coarse_fields, coarsening, "density", semantics="intensive"
    )
    assert not intensive.conservative
    np.testing.assert_allclose(intensive.apply(fine_density), coarse_density, atol=1e-10)
    with pytest.raises(ValueError, match="semantics"):
        nested_nodal_transfer(
            coarse_fields,
            fine_fields,
            refinement,
            "temperature",
            semantics="damage-history",  # ty: ignore[invalid-argument-type]
        )
    with pytest.raises(ValueError):
        nested_nodal_transfer(
            coarse_fields, fine_fields, coarsening, "temperature", semantics="interpolate"
        )


@pytest.mark.parametrize("tetrahedron_kind", ("tetrahedron", "simplex:3"))
def test_mixed_curved_coordinate_restrictions_survive_reopen(
    tetrahedron_kind: str,
) -> None:
    from phydrax.discretization._cell_geometry_transition_types import (
        CellGeometryTransition,
        CellGeometryTransitionEvidence,
    )
    from phydrax.discretization.fem._topology_transfer import nested_nodal_transfer
    from phydrax.meshing._mixed_adaptation import adapt_mixed_mesh
    from phydrax.meshing._topology_edit import assemble_topology_edit

    def density(mesh: Any) -> Any:
        return phx.discretization.FiniteElementFieldSpec(
            "density",
            {
                block.name: phx.discretization.discontinuous_element(block.cell_kind, 1)
                for block in mesh.blocks
            },
        )

    def map_count(maps: Any) -> int:
        return sum(len(record.target_cell_ids) for record in maps)

    def classes(mesh: Any, hierarchy: Any) -> np.ndarray:
        by_id = {
            int(cell.cell_id): int(cell.cell_class) for cell in hierarchy.active_cells
        }
        return np.asarray(
            [
                by_id[int(identifier)]
                for block in mesh.blocks
                for identifier in np.asarray(block.global_ids)
            ]
        )

    def region_values(discretization: Any, cell_classes: np.ndarray) -> jnp.ndarray:
        values = np.zeros((discretization.dof_maps[1].global_dof_count,))
        offset = 0
        for block, routes in zip(
            discretization.mesh.blocks, discretization.dof_maps[1].cell_dofs, strict=True
        ):
            for row in range(block.cell_count):
                values[np.asarray(routes[row])] = cell_classes[offset + row]
            offset += block.cell_count
        return jnp.asarray(values)

    mesh = _mixed_mesh(tetrahedron_kind)
    field = _field(degree=2, tetrahedron_kind=tetrahedron_kind)
    initial = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    dofs = initial.dof_maps[0]
    coordinates = np.asarray(dofs.dof_coordinates).copy()
    coordinates[:, 2] += 0.05 * coordinates[:, 0] * coordinates[:, 1]
    geometry = phx.discretization.CellGeometrySpec(
        dict(zip(("prisms", "tetrahedra"), field.resolve(mesh), strict=True)),
        dict(zip(("prisms", "tetrahedra"), dofs.cell_dofs, strict=True)),
        coordinates,
    )
    refinement = adapt_mixed_mesh(
        mesh,
        refine_cell_ids=np.asarray((101,), dtype=np.int64),
        source_geometry=geometry,
        cell_classes=np.asarray((2, 3), dtype=np.int64),
    )
    fine, _, _ = assemble_topology_edit(
        mesh, refinement.edit, numeric_version="curved-fine"
    )
    assert refinement.target_geometry is not None
    fine_fields = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, 2)
            for block in fine.blocks
        },
    )
    fine_discretization = phx.discretization.FiniteElementPlan(
        fine, (fine_fields, density(fine)), coordinate_spec=refinement.target_geometry
    ).prepare()
    fine_ids = np.concatenate([np.asarray(block.global_ids) for block in fine.blocks])
    coarsening = adapt_mixed_mesh(
        fine,
        coarsen_cell_ids=fine_ids,
        hierarchy=refinement.hierarchy,
        source_geometry=refinement.target_geometry,
    )
    restored, _, _ = assemble_topology_edit(
        fine, coarsening.edit, numeric_version="curved-restored"
    )
    assert coarsening.target_geometry is not None
    restored_fields = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(block.cell_kind, 2)
            for block in restored.blocks
        },
    )
    restored_discretization = phx.discretization.FiniteElementPlan(
        restored,
        (restored_fields, density(restored)),
        coordinate_spec=coarsening.target_geometry,
    ).prepare()
    coarse_discretization = phx.discretization.FiniteElementPlan(
        mesh, (field, density(mesh)), coordinate_spec=geometry
    ).prepare()
    refine_transition = CellGeometryTransition(
        mesh,
        geometry,
        fine,
        refinement.target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "nested_restriction", reference_map_count=map_count(refinement.reference_maps)
        ),
        reference_map=refinement.reference_maps,
    )
    assert coarsening.coarsening_partition is not None
    coarsen_transition = CellGeometryTransition(
        fine,
        refinement.target_geometry,
        restored,
        coarsening.target_geometry,
        evidence=CellGeometryTransitionEvidence(
            "parent_restoration",
            reference_map_count=map_count(coarsening.reference_maps),
            coarsening_map_count=map_count(coarsening.coarsening_partition.maps),
        ),
        reference_map=coarsening.reference_maps or None,
        coarsening_partition=coarsening.coarsening_partition,
    )
    coarse_classes = np.asarray((2, 3), dtype=np.int64)
    fine_classes = classes(fine, refinement.hierarchy)
    restored_classes = classes(restored, coarsening.hierarchy)
    for name, semantics, index in (("u", "interpolate", 0), ("density", "density", 1)):
        forward = nested_nodal_transfer(
            coarse_discretization,
            fine_discretization,
            refine_transition,
            name,
            semantics=semantics,
            **(
                {
                    "source_cell_classes": coarse_classes,
                    "target_cell_classes": fine_classes,
                }
                if index == 1
                else {}
            ),
        )
        backward = nested_nodal_transfer(
            fine_discretization,
            restored_discretization,
            coarsen_transition,
            name,
            semantics=semantics,
            **(
                {
                    "source_cell_classes": fine_classes,
                    "target_cell_classes": restored_classes,
                }
                if index == 1
                else {}
            ),
        )
        values = (
            1.0 + jnp.sum(coarse_discretization.dof_maps[index].dof_coordinates, axis=-1)
            if index == 0
            else region_values(coarse_discretization, coarse_classes)
        )
        fine_values = forward.apply(values)
        recovered = backward.apply(fine_values)
        expected = (
            1.0
            + jnp.sum(restored_discretization.dof_maps[index].dof_coordinates, axis=-1)
            if index == 0
            else region_values(restored_discretization, restored_classes)
        )
        np.testing.assert_allclose(recovered, expected, rtol=1e-9, atol=1e-9)
        if index == 1:
            for region in (2, 3):
                inventories = []
                for discretization, cell_classes, field_values in (
                    (coarse_discretization, coarse_classes, values),
                    (fine_discretization, fine_classes, fine_values),
                    (restored_discretization, restored_classes, recovered),
                ):
                    entities = discretization.mesh.topology.entity_sets[3]
                    selection = phx.discretization.EntitySelection(
                        entities, jnp.asarray(cell_classes == region)
                    )
                    functional = phx.equations.FiniteElementFunctional(
                        f"region-{region}-content",
                        "density",
                        lambda values, gradients, points, context: values,
                        domain=discretization.integration_domain("cell", selection),
                    )
                    inventories.append(
                        float(functional.evaluate(discretization, field_values))
                    )
                np.testing.assert_allclose(
                    inventories, inventories[0], rtol=1e-12, atol=1e-12
                )

    for twist, degree in (("untwisted", 1), ("twisted", 2)):
        coarse_complex = phx.discretization.FiniteElementDeRhamComplex(
            mesh, family="trimmed", order=2, twist=twist, coordinate_spec=geometry
        )
        fine_complex = phx.discretization.FiniteElementDeRhamComplex(
            fine,
            family="trimmed",
            order=2,
            twist=twist,
            coordinate_spec=refinement.target_geometry,
        )
        restored_complex = phx.discretization.FiniteElementDeRhamComplex(
            restored,
            family="trimmed",
            order=2,
            twist=twist,
            coordinate_spec=coarsening.target_geometry,
        )
        forward = coarse_complex.transfer(
            fine_complex, geometry_transition=refine_transition
        )
        backward = fine_complex.transfer(
            restored_complex, geometry_transition=coarsen_transition
        )
        assert len(forward.commuting_evidence) == len(backward.commuting_evidence) == 3
        size = coarse_complex.discretization.dof_maps[degree].global_dof_count
        coefficients = jnp.sin(jnp.arange(size, dtype=jnp.float64) + 0.31)
        recovered = backward.complex_map.map(degree).mv(
            forward.complex_map.map(degree).mv(coefficients)
        )
        restored_cells = {
            int(identifier): (block_index, row)
            for block_index, block in enumerate(restored.blocks)
            for row, identifier in enumerate(np.asarray(block.global_ids))
        }
        original_dofs = coarse_complex.discretization.dof_maps[degree]
        restored_dofs = restored_complex.discretization.dof_maps[degree]
        for block_index, block in enumerate(mesh.blocks):
            for row, identifier in enumerate(np.asarray(block.global_ids)):
                target_block, target_row = restored_cells[int(identifier)]
                original_vertices = np.asarray(mesh.vertex_global_ids)[
                    np.asarray(block.vertices[row])
                ]
                target_vertices = np.asarray(restored.vertex_global_ids)[
                    np.asarray(restored.blocks[target_block].vertices[target_row])
                ]
                np.testing.assert_array_equal(original_vertices, target_vertices)
                original_local = np.asarray(
                    original_dofs.cell_transforms[block_index][row]
                ) @ np.asarray(
                    coefficients[np.asarray(original_dofs.cell_dofs[block_index][row])]
                )
                restored_local = np.asarray(
                    restored_dofs.cell_transforms[target_block][target_row]
                ) @ np.asarray(
                    recovered[
                        np.asarray(restored_dofs.cell_dofs[target_block][target_row])
                    ]
                )
                np.testing.assert_allclose(
                    restored_local, original_local, rtol=1e-9, atol=1e-9
                )


def test_circulation_basis_reuse_preserves_actual_mixed_edge_tabulations() -> None:
    from copy import copy
    from types import SimpleNamespace

    from phydrax.discretization.fem._generic import _degree_aware_reference_rule
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _layer_circulation_reference_basis,
        _layer_field_plan,
    )

    mesh = _mixed_mesh("simplex:3")
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    field = next(
        item for item in _layer_field_plan(SimpleNamespace(mesh=mesh, geometry=geometry)).fields
        if item.name == "Hcurl"
    )
    parameters = np.asarray(_degree_aware_reference_rule("interval", 8)[0]).reshape(-1)
    cache: dict[tuple[int, tuple[int, ...], str, bytes], tuple[Any, np.ndarray]] = {}
    for element in field.resolve(mesh):
        vertices = np.asarray(phx.discretization.reference_cell_topology(element.cell_kind).vertices)
        edges = phx.discretization.reference_cell_topology(element.cell_kind).entities[1]
        for first, last in edges:
            start = vertices[first]
            points = start + parameters[:, None] * (vertices[last] - start)
            expected = np.asarray(element.tabulate(points)[0], dtype=np.float64)
            actual = _layer_circulation_reference_basis(element, points, cache)
            np.testing.assert_array_equal(actual.view(np.uint64), expected.view(np.uint64))
            assert _layer_circulation_reference_basis(element, points.copy(), cache) is actual
        duplicate = copy(element)
        start = vertices[edges[0][0]]
        points = start + parameters[:, None] * (vertices[edges[0][1]] - start)
        reversed_points = points[::-1].copy()
        reversed_basis = _layer_circulation_reference_basis(element, reversed_points, cache)
        expected_reversed = np.asarray(element.tabulate(reversed_points)[0], dtype=np.float64)
        np.testing.assert_array_equal(reversed_basis.view(np.uint64), expected_reversed.view(np.uint64))
        assert _layer_circulation_reference_basis(duplicate, points, cache) is not (
            _layer_circulation_reference_basis(element, points, cache)
        )


@pytest.mark.parametrize("tetrahedron_kind", ("tetrahedron", "simplex:3"))
@pytest.mark.parametrize("curved", (False, True))
def test_named_layer_form_layouts_preserve_actual_mixed_dofs(
    tetrahedron_kind: str,
    curved: bool,
) -> None:
    from copy import copy
    from types import SimpleNamespace

    from phydrax.geometry._mesh_certificates import _same_tree_content
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _layer_field_plan,
        _layer_form_fields,
    )

    mesh = _mixed_mesh(tetrahedron_kind)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    if curved:
        coordinate_field = _field(degree=2, tetrahedron_kind=tetrahedron_kind)
        coordinates_space = phx.discretization.FiniteElementPlan(
            mesh, coordinate_field
        ).prepare()
        coordinate_dofs = coordinates_space.dof_maps[0]
        points = np.asarray(coordinate_dofs.dof_coordinates).copy()
        points[:, 2] += 0.05 * points[:, 0] * points[:, 1]
        geometry = phx.discretization.CellGeometrySpec(
            dict(
                zip(("prisms", "tetrahedra"), coordinate_field.resolve(mesh), strict=True)
            ),
            dict(zip(("prisms", "tetrahedra"), coordinate_dofs.cell_dofs, strict=True)),
            points,
        )
    result = SimpleNamespace(mesh=mesh, geometry=geometry)
    plan = _layer_field_plan(result)
    reordered = phx.discretization.FiniteElementPlan(
        mesh, tuple(reversed(plan.fields)), coordinate_spec=geometry
    ).prepare()
    declarations = _layer_form_fields(result, reordered)
    assert tuple(item.name for item in declarations) == ("Hcurl", "Hdiv")
    for spec in declarations:
        name = spec.name
        index = tuple(item.name for item in reordered.field_spaces).index(name)
        original_dofs = reordered.dof_maps[index]
        original_elements = reordered.elements[index]
        assert spec.field_spec_id == next(
            field.field_spec_id for field in plan.fields if field.name == name
        )
        active = phx.discretization.FiniteElementPlan(
            mesh, (spec,), coordinate_spec=geometry
        ).prepare()
        assert _same_tree_content(active.dof_maps[0], original_dofs)
        structural = phx.discretization.FiniteElementDofMap(
            mesh,
            spec.resolve(mesh),
            component_shape=spec.component_shape,
            coordinate_spec=geometry,
        )
        assert _same_tree_content(structural, original_dofs)
        assert tuple(e.element_id for e in active.elements[0]) == tuple(
            e.element_id for e in original_elements
        )
        assert active.default_runtime.geometry_layout_id == geometry.geometry_layout_id
    assert len(declarations) == 2
    altered = copy(geometry)
    object.__setattr__(altered, "coordinates", np.asarray(geometry.coordinates) + 0.125)
    with pytest.raises(ValueError, match="actual mesh and geometry"):
        _layer_form_fields(SimpleNamespace(mesh=mesh, geometry=altered), reordered)


@pytest.mark.parametrize("tetrahedron_kind", ("tetrahedron", "simplex:3"))
@pytest.mark.parametrize("twist", ("untwisted", "twisted"))
@pytest.mark.parametrize("degree", (1, 2, 3))
def test_operation_local_reference_gradients_preserve_actual_form_orientation(
    tetrahedron_kind: str,
    twist: FormTwist,
    degree: int,
) -> None:
    from phydrax.discretization.fem._de_rham import _exact_form_cell_transform
    from phydrax.discretization.fem._generic import _form_cell_orientation_sign

    mesh = _mixed_mesh(tetrahedron_kind)
    vertex_ids = np.asarray(mesh.vertex_global_ids, dtype=np.int64)
    gradient_cache: dict[str, np.ndarray] = {}
    transform_cache: dict[Any, Any] = {}
    for block in mesh.blocks:
        element = phx.discretization.form_element(
            block.cell_kind,
            degree,
            2,
            family="prism-trimmed" if block.cell_kind == "prism" else "trimmed",
            twist=twist,
            proxy="components",
        )
        for row, cell in enumerate(np.asarray(block.vertices, dtype=np.int64)):
            identifiers = tuple(vertex_ids[cell].tolist())
            direct = _form_cell_orientation_sign(mesh, block, element, cell, identifiers)
            reused = _form_cell_orientation_sign(
                mesh, block, element, cell, identifiers, gradient_cache
            )
            assert reused == direct
            uncached = _exact_form_cell_transform(mesh, block, element, row, vertex_ids)
            cached = _exact_form_cell_transform(
                mesh, block, element, row, vertex_ids, transform_cache, gradient_cache
            )
            assert cached == uncached
    assert set(gradient_cache) == (
        {block.cell_kind for block in mesh.blocks} if twist == "twisted" else set()
    )
