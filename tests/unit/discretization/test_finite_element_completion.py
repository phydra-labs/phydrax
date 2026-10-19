#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


# Empirical `test_resolved_affine_geometry_preserves_measure_and_linear_gradient`: matched max |delta|=8.27181e-25; joint allowance fraction=0.000165436, margin=99.9835%.
_AFFINE_MEASURE_RTOL = 1e-12
# Empirical `test_resolved_affine_geometry_preserves_measure_and_linear_gradient`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_AFFINE_LINEAR_GRADIENT_RTOL = 1e-12
# Empirical `test_resolved_affine_geometry_preserves_measure_and_linear_gradient`: matched max |delta|=0; joint allowance fraction=0, margin=100%.
_AFFINE_LINEAR_GRADIENT_ATOL = 1e-12


def _square_mesh() -> Any:
    vertices = jnp.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]
    )
    cells = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    return phx.discretization.CellMesh.from_triangles(
        vertices,
        cells,
        vertex_global_ids=jnp.asarray([10, 20, 30, 40, 50]),
        cell_global_ids=jnp.asarray([100, 200, 300, 400]),
    )


def _scalar_discretization(*, precision_policy: Any = None) -> Any:
    mesh = _square_mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 1),
    )
    return phx.discretization.FiniteElementPlan(
        mesh,
        field,
        precision_policy=precision_policy,
    ).prepare()


@pytest.mark.parametrize("dimension", (2, 3), ids=("triangle", "tetrahedron"))
def test_resolved_affine_geometry_preserves_measure_and_linear_gradient(
    dimension: int,
) -> None:
    jacobian = np.eye(dimension, dtype=np.float64)
    jacobian[0, 1] = 1.0
    jacobian[1, 1] = 1e-8
    coordinates = np.concatenate(
        (np.zeros((1, dimension), dtype=np.float64), jacobian.T), axis=0
    )
    mesh = phx.discretization.CellMesh.from_simplices(
        coordinates,
        np.arange(dimension + 1, dtype=np.int32)[None, :],
        dimension=dimension,
    )
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.form_element(
            f"simplex:{dimension}", 0, 1, family="trimmed", proxy="scalar"
        ),
    )
    space = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    geometry = space.block_geometries[0][0]
    expected_measure = 1e-8 / (2.0 if dimension == 2 else 6.0)
    np.testing.assert_allclose(
        geometry.measure, expected_measure, rtol=_AFFINE_MEASURE_RTOL, atol=0.0
    )
    dof_map = space.dof_maps[0]
    nodal_x = np.asarray(dof_map.dof_coordinates)[:, 0]
    local_x = nodal_x[np.asarray(dof_map.cell_dofs[0])]
    physical_gradients = np.asarray(geometry.physical_gradients)
    gradient = np.sum(local_x[:, None, :, None] * physical_gradients, axis=-2)
    expected_gradient = np.zeros_like(gradient)
    expected_gradient[..., 0] = 1.0
    np.testing.assert_allclose(
        gradient,
        expected_gradient,
        rtol=_AFFINE_LINEAR_GRADIENT_RTOL,
        atol=_AFFINE_LINEAR_GRADIENT_ATOL,
    )


def test_runtime_geometry_precision_identity_and_projection_are_operational() -> None:
    precision = phx.discretization.FiniteElementPrecisionPolicy(
        geometry_dtype="float64",
        evaluation_dtype="float64",
        accumulation_dtype="float64",
        output_dtype="float64",
    )
    discretization = _scalar_discretization(precision_policy=precision)
    form = phx.equations.FiniteElementForm(
        "shape-diffusion",
        "u",
        (phx.equations.DiffusionAction("u"),),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            accumulation="compensated",
        ),
    )
    state = jnp.asarray([0.0, 1.0, 2.0, 1.0, 1.0])
    direction = jnp.zeros_like(discretization.mesh.coordinates).at[1, 0].set(0.1)

    def residual_at(coordinates: Any) -> Any:
        runtime = discretization.prepare_runtime(
            coordinates,
            numeric_version="shape",
        )
        context = phx.equations.FiniteElementExecutionContext(runtime)
        return compiled.full_residual(state, context)

    _, tangent = jax.jvp(
        residual_at,
        (discretization.mesh.coordinates,),
        (direction,),
    )
    projected = discretization.project(
        "u",
        lambda points, args: points[:, 0] + points[:, 1],
    )
    trace_points, trace_values = discretization.trace("u", projected)

    assert jnp.linalg.norm(tangent) > 0.0
    assert projected.shape == (5,)
    assert trace_points.shape == (4, 2)
    assert trace_values.shape == (4,)
    assert discretization.precision_evidence is not None
    assert jnp.array_equal(
        discretization.mesh.topology.entity_sets[0].entity_ids,
        jnp.asarray([10, 20, 30, 40, 50]),
    )


def test_finite_element_completion_scenario_1() -> None:
    mesh = _square_mesh()
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec("u", element, component_shape=(2,)),
            phx.discretization.FiniteElementFieldSpec("p", element),
        ),
    ).prepare()
    u_constraint = phx.discretization.dirichlet_constraint(discretization, "u")
    p_constraint = phx.discretization.dirichlet_constraint(discretization, "p")
    form = phx.equations.FiniteElementForm(
        "mixed",
        ("u", "p"),
        (
            phx.equations.DiffusionAction("u", action_id="u-diffusion"),
            phx.equations.DiffusionAction("p", action_id="p-diffusion"),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraints={"u": u_constraint, "p": p_constraint},
        dirichlet_values_by_field={
            "u": lambda points: jnp.stack((points[:, 0], points[:, 1]), axis=-1),
            "p": lambda points: points[:, 0] + points[:, 1],
        },
    )
    system, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(system, right_hand_side)
    displacement, pressure = compiled.expand(result.value)

    assert jnp.all(result.successful)
    assert displacement.shape == (5, 2)
    assert pressure.shape == (5,)
    # ty: ignore[unresolved-attribute]
    assert compiled.state_space.names == ("u", "p")
    discretization = _scalar_discretization()
    cells = discretization.mesh.topology.entity_sets[2]
    selection = phx.discretization.EntitySelection(
        cells,
        jnp.asarray([True, False, False, False]),
    )
    domain = discretization.integration_domain("cell", selection)
    source = phx.equations.coefficient(
        jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        location="cell",
        support_id=discretization.support.support_id,
        entity_set_id=cells.entity_set_id,
    )
    term = phx.equations.SourceAction(
        "u",
        source,
        domain=domain,
        rules={"triangles": phx.integration.ReferenceTriangleRule()},
    )
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("selected-source", "u", (term,)),
        discretization,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free"
        ),
    )
    residual = compiled.full_residual(jnp.zeros((5,)))

    assert jnp.allclose(jnp.sum(residual), -0.25)
    assert domain.selection_id == selection.selection_id
    discretization = _scalar_discretization()
    state = jnp.asarray([0.0, 1.0, 2.0, 1.0, 1.0])
    functional = phx.variational.Functional(
        "energy",
        (
            phx.variational.LocalIntegralTerm(
                "dirichlet-energy",
                region="body",
                fields=(phx.variational.FieldJetSpec("u", gradient=True),),
                density=lambda fields, geometry, context: (
                    0.5 * jnp.sum(fields["u"].gradient ** 2, axis=-1)
                ),
                density_id="dirichlet-energy",
            ),
        ),
        variable_fields=("u",),
    )
    compiled_energy = phx.equations.compile_finite_element_functional(
        functional,
        discretization,
        fields={"u": "u"},
        regions={"body": None},
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free"
        ),
    )
    assert jnp.allclose(
        compiled_energy.full_residual(state),
        discretization.stiffness.mv(state),
        atol=1.0e-12,
    )

    dg = phx.discretization.FiniteElementPlan(
        _square_mesh(),
        phx.discretization.FiniteElementFieldSpec(
            "w", phx.discretization.discontinuous_element("triangle")
        ),
    ).prepare()
    flux = phx.equations.InteriorFacetAction(
        "w",
        ("w",),
        lambda plus, minus, points, weights, normal, context: (
            plus[0] - minus[0],
            minus[0] - plus[0],
        ),
        action_id="jump-flux",
    )
    compiled_dg = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("dg-jump", "w", (flux,)),
        dg,
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free"
        ),
    )
    dg_residual = compiled_dg.full_residual(jnp.asarray([1.0, 0.0, 0.0, 0.0]))

    assert jnp.allclose(jnp.sum(dg_residual), 0.0)
    assert jnp.linalg.norm(dg_residual) > 0.0


def test_finite_element_completion_scenario_2() -> None:
    base = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = phx.discretization.CellMesh.from_triangles(
        base,
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    )
    coordinate_values = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, -0.1], [0.5, 0.5], [0.0, 0.5]]
    )
    coordinate_spec = phx.discretization.CellGeometrySpec(
        {"triangles": phx.discretization.lagrange_element("triangle", 2)},
        {"triangles": jnp.asarray([[0, 1, 2, 3, 4, 5]], dtype=jnp.int32)},
        coordinate_values,
    )
    curved = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
        coordinate_spec=coordinate_spec,
    ).prepare()
    rt = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "q",
            phx.discretization.form_element(
                "triangle", 1, 1, twist="twisted", proxy="flux"
            ),
        ),
    ).prepare()
    ne = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "e",
            phx.discretization.form_element(
                "triangle", 1, 1, twist="untwisted", proxy="circulation"
            ),
        ),
    ).prepare()

    elimination = phx.linalg.LocalEliminationPlan(3, jnp.asarray([1, 2]))
    local_matrix = jnp.asarray([[[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]]])
    condensed = elimination.condense(local_matrix, jnp.asarray([[1.0, 0.0, 1.0]]))
    reconstructed = elimination.reconstruct(jnp.asarray([[1.0, 1.0]]), condensed)
    hdg = phx.discretization.HDGCondensationPlan(
        phx.discretization.HDGTraceSpace(mesh), 1
    )

    assert curved.default_runtime.coordinates.shape == (6, 2)
    assert rt.field_spaces[0].conformity == "Hdiv"
    assert ne.field_spaces[0].conformity == "Hcurl"
    assert jnp.allclose(reconstructed, 1.0)
    assert hdg.local_trace_dof_count == 3
    mesh = _square_mesh()
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "q",
            phx.discretization.form_element(
                "triangle", 1, 1, twist="twisted", proxy="flux"
            ),
        ),
    ).prepare()
    rule = phx.integration.ReferenceTriangleRule(phx.integration.GaussLegendreRule(4))
    functional = phx.equations.FiniteElementFunctional(
        "hdiv-norm",
        "q",
        lambda values, gradients, points, context: jnp.sum(values**2, axis=-1),
        rules={"triangles": rule},
    )
    state = jnp.linspace(
        -0.4,
        1.1,
        discretization.dof_maps[0].global_dof_count,
    )
    data = rule.materialize()
    reconstructed = discretization.reconstruct(
        "q",
        state,
        "triangles",
        data.points,
    )
    geometry = discretization.evaluate_block_geometry(
        "q",
        0,
        discretization.default_runtime.coordinates,
        data.points,
        data.weights,
    )
    expected = jnp.sum(jnp.sum(reconstructed**2, axis=-1) * geometry.physical_weights)

    assert jnp.allclose(functional.evaluate(discretization, state), expected)


def test_solver_material_checkpoint_and_distributed_contracts(tmp_path: Any) -> None:
    discretization = _scalar_discretization()
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "stiffness",
            "u",
            (phx.equations.DiffusionAction("u"),),
        ),
        discretization,
    )
    dae = compiled.as_dae_system()
    second_order = compiled.as_second_order_system()
    eigenproblem = compiled.as_generalized_eigenproblem()

    state = phx.equations.MaterialState(
        phx.equations.MaterialSiteId("history"),
        "history",
        jnp.zeros((2, 3)),
    )
    transaction = phx.equations.MaterialTransaction((state,))
    committed = transaction.with_trials({"history": jnp.ones((2, 3))}).commit()
    checkpoint = phx.solver.FiniteElementCheckpoint(
        discretization.prepared_id,
        compiled.compilation_id,
        0.0,
        1,
        (jnp.ones((5,)),),
        materials=committed,
    )
    path = tmp_path / "finite-element.npz"
    phx.solver.write_finite_element_checkpoint(path, checkpoint)
    restored = phx.solver.read_finite_element_checkpoint(
        path,
        prepared_id=discretization.prepared_id,
        compilation_id=compiled.compilation_id,
    )

    halo = phx.discretization.FiniteElementHaloPlan(
        jnp.asarray([[0, 2], [1, 3]], dtype=jnp.int32)
    )
    values = jnp.asarray([1.0, 2.0, 3.0, 4.0])

    assert jnp.allclose(
        dae(jnp.asarray(0.0), jnp.ones((5,)), jnp.zeros((5,)), None),
        0.0,
    )
    assert jnp.allclose(
        second_order.evaluate(
            jnp.asarray(0.0),
            jnp.ones((5,)),
            jnp.zeros((5,)),
            jnp.zeros((5,)),
            None,
        ),
        0.0,
    )
    assert eigenproblem.problem_id
    assert restored.checkpoint_id == checkpoint.checkpoint_id
    assert restored.materials is not None
    assert restored.materials.states[0].state_version == 1
    assert jnp.allclose(halo.sum_contributions(values), jnp.asarray([4.0, 6.0, 4.0, 6.0]))
    assert jnp.allclose(halo.average_replicas(values), jnp.asarray([2.0, 3.0, 2.0, 3.0]))


def test_adaptivity_embedding_partition_and_io_contracts(tmp_path: Any) -> None:
    discretization = _scalar_discretization()
    estimate = phx.discretization.residual_jump_estimate(
        jnp.ones((4,)),
        jnp.full((4,), 0.25),
        jnp.asarray([0.5]),
        jnp.asarray([jnp.sqrt(2.0)]),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.asarray([1], dtype=jnp.int32),
    )
    embedded = phx.discretization.EmbeddedQuadrature(
        discretization.cell_domain,
        jnp.zeros((4, 2, 2)),
        jnp.full((4, 2), 0.125),
        jnp.ones((4, 2), dtype="bool"),
        classification_version="level-set-0",
    )
    enrichment = phx.discretization.FiniteElementEnrichment(
        lambda points: jnp.stack((jnp.ones(points.shape[:-1]), points[..., 0]), axis=-1),
        jnp.asarray([True, False, True, False]),
        2,
        enrichment_id="partition-of-unity",
    )
    partition = phx.discretization.PartitionedFiniteElementDofMap(
        discretization.dof_maps[0],
        jnp.arange(5),
        jnp.asarray([True, True, True, False, False]),
        multiplicity=jnp.asarray([1.0, 1.0, 1.0, 2.0, 2.0]),
    )
    constraint = phx.discretization.affine_dof_constraint(
        discretization,
        "u",
        jnp.eye(5),
    )
    distributed_constraint = phx.discretization.DistributedFiniteElementConstraint(
        constraint,
        partition,
    )
    output_path = tmp_path / "field.vtu"
    phx.discretization.write_finite_element_field(
        output_path,
        discretization,
        "u",
        jnp.arange(5.0),
    )

    assert estimate.global_estimate > 0.0
    assert embedded.quadrature_id
    assert enrichment.evaluate(jnp.asarray([[0.25, 0.5]])).shape == (1, 2)
    assert partition.global_inner(jnp.ones(5), jnp.ones(5)) == 3.0
    assert distributed_constraint.partition_id == partition.partition_id
    assert output_path.exists()


@pytest.mark.parametrize("transfer_only", (False, True))
def test_multifield_preparation_resolves_coordinates_once(
    monkeypatch: pytest.MonkeyPatch, transfer_only: bool
) -> None:
    mesh = _square_mesh()
    D = phx.discretization
    geometry = D.CellGeometrySpec.affine(mesh)
    fields = (
        D.FiniteElementFieldSpec("u", D.lagrange_element("triangle", 1)),
        D.FiniteElementFieldSpec("v", D.lagrange_element("triangle", 2)),
    )
    expected = tuple(
        D.FiniteElementPlan(mesh, field, coordinate_spec=geometry).prepare()
        for field in fields
    )
    plan = D.FiniteElementPlan(mesh, fields, coordinate_spec=geometry)
    resolve = D.CellGeometrySpec.resolve
    calls = 0

    def observe(
        spec: phx.discretization.CellGeometrySpec,
        carrier: phx.discretization.CellMesh,
        **kwargs: Any,
    ) -> Any:
        nonlocal calls
        if spec is geometry:
            calls += 1
        return resolve(spec, carrier, **kwargs)

    monkeypatch.setattr(D.CellGeometrySpec, "resolve", observe)
    prepared = plan.prepare_transfer() if transfer_only else plan.prepare()
    assert calls == 1
    for index, reference in enumerate(expected):
        assert prepared.dof_maps[index].dof_map_id == reference.dof_maps[0].dof_map_id
        assert (
            prepared.field_spaces[index].field_space_id
            == reference.field_spaces[0].field_space_id
        )
        np.testing.assert_array_equal(
            prepared.dof_maps[index].dof_coordinates,
            reference.dof_maps[0].dof_coordinates,
        )
        if isinstance(prepared, phx.discretization.FiniteElementDiscretization):
            for actual_block, expected_block in zip(
                prepared.block_geometries[index],
                reference.block_geometries[0],
                strict=True,
            ):
                for actual, wanted in zip(
                    jax.tree_util.tree_leaves(actual_block),
                    jax.tree_util.tree_leaves(expected_block),
                    strict=True,
                ):
                    np.testing.assert_array_equal(actual, wanted)


def test_coordinate_preparation_refuses_foreign_immutable_inputs() -> None:
    from phydrax.discretization.fem._assembly import _PreparedCoordinateMaps

    mesh = _square_mesh()
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    prepared = _PreparedCoordinateMaps.prepare(mesh, geometry, budget=None)
    with pytest.raises(ValueError, match="coordinate_preparation.binding"):
        prepared.require(_square_mesh(), geometry)
    with pytest.raises(ValueError, match="coordinate_preparation.binding"):
        prepared.require(mesh, phx.discretization.CellGeometrySpec.affine(mesh))


@pytest.mark.parametrize("proxy", ("scalar", "circulation", "flux"))
@pytest.mark.parametrize("retained_action", (False, True))
def test_prepared_geometry_padding_retains_bitwise_cell_tables(
    proxy: Literal["scalar", "circulation", "flux"],
    retained_action: bool,
) -> None:
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
    )
    from phydrax.discretization.fem._generic import _prepare_block_geometry

    D = phx.discretization
    square = _square_mesh()
    mesh = D.CellMesh.from_triangles(
        np.asarray(square.coordinates),
        np.asarray(square.blocks[0].vertices)[:3],
    )
    element = D.form_element(
        "triangle",
        0 if proxy == "scalar" else 1,
        1,
        family="trimmed",
        proxy=proxy,
        twist="twisted" if proxy == "flux" else "untwisted",
    )
    precision = D.FiniteElementPrecisionPolicy()
    coordinate_element = (
        BarycentricCellGeometryElement(
            coordinate_lagrange_element("triangle", 1),
            np.asarray(
                ((1.0, 0.0, 0.0), (0.25, 0.75, 0.0), (0.125, 0.375, 0.5)),
                dtype=np.float64,
            ),
        )
        if retained_action
        else None
    )
    coordinates = (
        np.asarray(mesh.coordinates) * np.asarray((1.1, 1.3), dtype=np.float64)
        + np.asarray((0.123, 0.234), dtype=np.float64)
        if retained_action
        else mesh.coordinates
    )
    reference = _prepare_block_geometry(
        mesh,
        mesh.blocks[0],
        element,
        precision_policy=precision,
        coordinate_element=coordinate_element,
        coordinates=coordinates,
    )
    bucketed = _prepare_block_geometry(
        mesh,
        mesh.blocks[0],
        element,
        precision_policy=precision,
        coordinate_element=coordinate_element,
        coordinates=coordinates,
        _preparation_bucket=True,
    )
    assert bucketed.physical_points.shape[0] == 3
    for actual, expected in zip(
        jax.tree_util.tree_leaves(bucketed),
        jax.tree_util.tree_leaves(reference),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


@eqx.filter_jit
def _ungrouped_local_tensors(
    geometries: tuple[Any, ...], *, mass: bool
) -> tuple[jax.Array, ...]:
    from phydrax.discretization.fem._assembly import (
        local_mass_tensor,
        local_stiffness_tensor,
    )

    return tuple(
        local_mass_tensor(block) if mass else local_stiffness_tensor(block)
        for block in geometries
    )


@pytest.mark.parametrize("mass", (False, True))
@pytest.mark.parametrize("proxy", ("scalar", "circulation", "flux"))
def test_cell_worksets_preserve_interleaved_block_coefficients(
    mass: bool, proxy: Literal["scalar", "circulation", "flux"]
) -> None:
    from phydrax.discretization.fem._assembly import (
        _transformed_local_values,
        geometry_local_tensors,
    )

    D = phx.discretization
    mesh = _square_mesh()
    element = D.form_element(
        "triangle",
        0 if proxy == "scalar" else 1,
        1,
        family="trimmed",
        proxy=proxy,
        twist="twisted" if proxy == "flux" else "untwisted",
    )
    space = D.FiniteElementPlan(mesh, D.FiniteElementFieldSpec("u", element)).prepare()
    geometry = space.block_geometries[0][0]
    cellwise_basis = geometry.basis_values.ndim != 2
    blocks = tuple(
        eqx.tree_at(
            lambda value: (
                value.physical_weights,
                value.physical_gradients,
                value.physical_points,
                value.measure,
                *((value.basis_values,) if cellwise_basis else ()),
            ),
            geometry,
            (
                geometry.physical_weights[start:stop],
                geometry.physical_gradients[start:stop],
                geometry.physical_points[start:stop],
                geometry.measure[start:stop],
                *((geometry.basis_values[start:stop],) if cellwise_basis else ()),
            ),
        )
        for start, stop in ((0, 1), (1, 4), (0, 2))
    )
    local = geometry_local_tensors(blocks, mass=mass)
    reference = _ungrouped_local_tensors(blocks, mass=mass)
    for block, actual, expected in zip(blocks, local, reference, strict=True):
        if proxy == "scalar" and mass:
            # P1 simplex moment integrals do not share tensor quadrature.
            analytic = (
                np.asarray(block.measure)[:, None, None]
                * (np.ones((3, 3), dtype=np.float64) + np.eye(3, dtype=np.float64))
                / 12.0
            )
            # Two products per quadrature term and its reduction, plus scaling.
            operations = 3 * block.physical_weights.shape[1] + 2
            epsilon = np.finfo(np.float64).eps
            gamma = operations * epsilon / (1.0 - operations * epsilon)
            np.testing.assert_allclose(actual, analytic, rtol=gamma, atol=0.0)
            np.testing.assert_allclose(expected, analytic, rtol=gamma, atol=0.0)
        else:
            np.testing.assert_array_equal(actual, expected)
    width = local[0].shape[-1]
    transforms = tuple(
        jnp.broadcast_to(jnp.eye(width, dtype=jnp.float64)[:, ::-1], values.shape)
        for values in local
    )
    transformed = _transformed_local_values(transforms, local)
    for actual, values, transform in zip(transformed, local, transforms, strict=True):
        # Independently apply each signed/permuted local coefficient map.
        expected = (
            np.swapaxes(np.asarray(transform), -1, -2)
            @ np.asarray(values)
            @ np.asarray(transform)
        )
        np.testing.assert_array_equal(actual, expected)
