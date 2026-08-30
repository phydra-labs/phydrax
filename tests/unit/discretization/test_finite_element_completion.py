#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def _square_mesh():
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


def _scalar_discretization(*, precision_policy=None):
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


def test_runtime_geometry_precision_identity_and_projection_are_operational():
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

    def residual_at(coordinates):
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


def test_component_and_mixed_block_spaces_solve_through_native_linalg():
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
    assert compiled.state_space.names == ("u", "p")


def test_domains_rules_and_entity_coefficients_select_exact_cells():
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


def test_energy_custom_residual_and_interior_flux_share_one_compiler():
    discretization = _scalar_discretization()
    state = jnp.asarray([0.0, 1.0, 2.0, 1.0, 1.0])
    energy = phx.equations.CellEnergyAction(
        "u",
        lambda values, gradients, points, context: 0.5 * jnp.sum(gradients**2, axis=-1),
        action_id="dirichlet-energy",
    )
    compiled_energy = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("energy", "u", (energy,)),
        discretization,
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
        lambda plus, minus, points, weights, normal, context: (
            plus - minus,
            minus - plus,
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


def test_curved_compatible_local_and_hdg_spaces_are_executable():
    base = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mesh = phx.discretization.CellMesh.from_triangles(
        base,
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    )
    coordinate_values = jnp.asarray(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, -0.1], [0.5, 0.5], [0.0, 0.5]]
    )
    coordinate_spec = phx.discretization.FiniteElementCoordinateSpec(
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
            "q", phx.discretization.raviart_thomas_element("triangle")
        ),
    ).prepare()
    ne = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "e", phx.discretization.nedelec_element("triangle")
        ),
    ).prepare()

    elimination = phx.discretization.FiniteElementLocalEliminationPlan(
        3, jnp.asarray([1, 2])
    )
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


def test_solver_material_checkpoint_and_distributed_contracts(tmp_path):
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

    state = phx.equations.FiniteElementMaterialState("history", jnp.zeros((2, 3)))
    transaction = phx.equations.FiniteElementMaterialTransaction((state,))
    committed = transaction.with_trials({"history": jnp.ones((2, 3))}).commit()
    checkpoint = phx.solver.FiniteElementCheckpoint(
        discretization.prepared_id,
        compiled.compilation_id,
        0.0,
        1,
        (jnp.ones((5,)),),
        material_states=committed.states,
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
    assert restored.material_states[0].state_version == 1
    assert jnp.allclose(halo.sum_contributions(values), jnp.asarray([4.0, 6.0, 4.0, 6.0]))
    assert jnp.allclose(halo.average_replicas(values), jnp.asarray([2.0, 3.0, 2.0, 3.0]))


def test_adaptivity_embedding_partition_and_io_contracts(tmp_path):
    discretization = _scalar_discretization()
    refinement = phx.discretization.FiniteElementRefinementMap(
        "coarse",
        "fine",
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        jnp.asarray([0, 1, 2, 3, 4], dtype=jnp.int32),
    )
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
        jnp.ones((4, 2), dtype=bool),
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

    assert refinement.refinement_id
    assert estimate.global_estimate > 0.0
    assert embedded.quadrature_id
    assert enrichment.evaluate(jnp.asarray([[0.25, 0.5]])).shape == (1, 2)
    assert partition.global_inner(jnp.ones(5), jnp.ones(5)) == 3.0
    assert distributed_constraint.partition_id == partition.partition_id
    assert output_path.exists()
