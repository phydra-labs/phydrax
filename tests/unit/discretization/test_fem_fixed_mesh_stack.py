#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = jnp.asarray([[0, 1, 3], [1, 2, 3]], dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(
        vertices, cells, cell_global_ids=jnp.asarray([10, 20])
    )


def test_action_ir_and_packed_facet_routes_are_explicit():
    mesh = _mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.discontinuous_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.fem.sipg_poisson_form(
        "u",
        1.0,
        phx.equations.fem.SIPGPenaltyPolicy(12.0),
        discretization.cell_domain,
        discretization.interior_facet_domain,
        (),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    facet = tuple(
        workset
        for workset in compiled._workset_program.worksets
        if workset.signature.region_kind == "interior-facet"
    )[0]

    assert compiled._action_ir.actions
    assert facet.action_indices.shape == (1,)
    assert facet.owner_local_entities.shape == facet.entity_indices.shape
    assert facet.neighbour_local_entities.shape == facet.entity_indices.shape
    assert int(facet.owner_permutations[0]) == -int(facet.neighbour_permutations[0])
    assert dict(facet.neighbour_gathers)["u"].shape == dict(facet.gathers)["u"].shape


def test_rectangular_element_tensor_has_exact_sparse_and_transpose_actions():
    matrices = jnp.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    inputs = jnp.asarray([[0, 1]], dtype=jnp.int32)
    outputs = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    operator = phx.equations.fem.ElementTensorOperator(matrices, inputs, outputs, 2, 3)
    value = jnp.asarray([2.0, -1.0])
    covector = jnp.asarray([1.0, 2.0, -1.0])

    assert jnp.allclose(operator.mv(value), matrices[0] @ value)
    assert jnp.allclose(operator.transpose_mv(covector), matrices[0].T @ covector)
    assert jnp.allclose(operator.as_sparse_coordinate().mv(value), operator.mv(value))


def test_conforming_p3_routes_and_tensor_partial_action():
    mesh = _mesh()
    element = phx.discretization.lagrange_element("triangle", 3)
    discretization = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("u", element)
    ).prepare()
    projected = discretization.project(
        "u", lambda points, args: points[:, 0] ** 2 + points[:, 1] ** 2
    )

    family = phx.discretization.fem.ReferenceNodalFamily("quadrilateral", 1)
    tabulation = phx.discretization.fem.TensorProductTabulation(
        family, jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 1.0])
    )
    plan = phx.discretization.fem.SumFactorizationPlan(tabulation)
    tensor = phx.equations.fem.TensorProductPartialAssemblyOperator(
        plan,
        jnp.ones((1, 2, 2)),
        jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32),
        4,
        action_kind="mass",
    )

    assert discretization.dof_maps[0].association == "entity"
    assert projected.shape == (16,)
    assert jnp.allclose(tensor.mv(jnp.arange(1.0, 5.0)), jnp.arange(1.0, 5.0))


def test_block_graph_preconditioner_data_and_auxiliary_validity():
    mesh = _mesh()
    p1 = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        mesh,
        (
            phx.discretization.FiniteElementFieldSpec(
                "u",
                phx.discretization.lagrange_element("triangle", 2),
                component_shape=(2,),
            ),
            phx.discretization.FiniteElementFieldSpec("p", p1),
        ),
    ).prepare()
    compiled = phx.equations.compile_finite_element_problem(
        phx.equations.fem.stokes_form("u", "p"), discretization
    )
    zero = compiled.state_space.zeros()
    block = compiled.block_linearization_operator(zero)

    assert compiled.block_dependency_graph() == ((True, True), (True, False))
    assert tuple(value.shape for value in block.mv(zero)) == ((9, 2), (4,))

    scalar = phx.discretization.FiniteElementPlan(
        mesh, phx.discretization.FiniteElementFieldSpec("s", p1)
    ).prepare()

    def auxiliary(state, context):
        return phx.equations.fem.FiniteElementAuxiliaryEvaluation(
            state + 1.0, successful=True, admissible=True
        )

    form = phx.equations.FiniteElementForm(
        "auxiliary",
        "s",
        (phx.equations.DiffusionAction("s"),),
        auxiliary_evaluator=auxiliary,
        auxiliary_id="state-plus-one",
    )
    scalar_compiled = phx.equations.compile_finite_element_problem(form, scalar)
    _, evaluated = scalar_compiled.residual_with_auxiliary(jnp.zeros((4,)))
    data = scalar_compiled.preconditioner_data()

    assert evaluated.valid
    assert jnp.allclose(evaluated.trial_state, 1.0)
    assert data.diagonal.shape == (4,)


def test_observation_restart_and_result_roundtrip(tmp_path):
    space = phx.linalg.ArraySpace((4,))
    observation = phx.equations.fem.CoordinateObservation(
        space, jnp.asarray([1, 3]), weights=jnp.asarray([2.0, 1.0])
    )
    state = jnp.arange(4.0)
    test_value = jnp.asarray([3.0, 4.0])
    assert jnp.allclose(
        jnp.vdot(observation.evaluate(state), test_value),
        jnp.vdot(state, observation.transpose(test_value)),
    )

    accepted = phx.solver.FiniteElementAcceptedState(
        (state,), 0.5, 3, "topology", "prepared", "compiled"
    )
    manifest = phx.solver.FiniteElementRestartManifest(
        accepted,
        auxiliary_state=(("history", jnp.asarray([2.0])),),
        integrator_state=(("previous", jnp.asarray([0.25])),),
    )
    restart_path = tmp_path / "restart.npz"
    phx.solver.write_finite_element_restart(restart_path, manifest)
    restored = phx.solver.read_finite_element_restart(restart_path)
    assert restored.manifest_id == manifest.manifest_id

    diagnostics = phx.solver.FiniteElementSolveDiagnostics(True, 1.0e-12)
    result = phx.solver.FiniteElementResult(
        ("u",), (state,), 0.5, "prepared", "compiled", diagnostics
    )
    result_path = tmp_path / "result.npz"
    phx.solver.write_finite_element_result(result_path, result)
    loaded = phx.solver.read_finite_element_result(result_path)
    assert loaded.result_id == result.result_id


def test_contact_search_and_cut_quadrature_are_deterministic():
    contact = phx.applications.contact.ContactSearchPlan(
        jnp.asarray([[[0.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 1.0]]]),
        jnp.asarray([10, 20]),
    )
    pairs = contact.search(jnp.asarray([[0.5, 0.1]]), jnp.asarray([3]))
    assert int(pairs.master_ids[0]) == 10

    mesh = _mesh()
    fracture = phx.applications.fracture
    crack = fracture.CrackGeometry([0.0, 0.5], [1.0, 0.5])
    enrichment = fracture.classify_crack_cells(mesh, crack)
    quadrature = fracture.cut_cell_quadrature(mesh, crack)
    layout = fracture.FixedMeshEnrichmentLayout(mesh, enrichment)

    assert jnp.allclose(jnp.sum(quadrature.weights), 1.0)
    assert layout.enriched_dofs.size == enrichment.active_vertex_ids.size
