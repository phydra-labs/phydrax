#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import opt_einsum as oe
import pytest

import phydrax as phx


def _tri_mesh():
    vertices = jnp.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.5, 0.5]])
    cells = jnp.asarray(
        [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
        dtype=jnp.int32,
    )
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def test_sum_factorized_gradient_contracts_every_nodal_axis():
    family = phx.discretization.fem.ReferenceNodalFamily("quadrilateral", 2)
    tabulation = phx.discretization.fem.TensorProductTabulation(
        family,
        jnp.asarray([0.2, 0.8]),
        jnp.asarray([0.3, 0.7]),
    )
    plan = phx.discretization.fem.SumFactorizationPlan(tabulation)
    coefficients = jnp.zeros((3, 3)).at[2, 1].set(1.0)
    gradient = plan.gradient(coefficients)
    points = jnp.stack(
        jnp.meshgrid(
            jnp.asarray([0.2, 0.8]),
            jnp.asarray([0.3, 0.7]),
            indexing="ij",
        ),
        axis=-1,
    ).reshape((-1, 2))
    _, dense_gradient = family.tabulate(points)
    expected = oe.contract("qia,i->qa", dense_gradient, coefficients.reshape((-1,)))

    assert gradient.shape == (2, 2, 2)
    assert jnp.linalg.norm(gradient) > 0.0
    assert jnp.allclose(gradient.reshape((-1, 2)), expected)


def test_smoothed_elasticity_defaults_to_budgeted_matrix_free_operator():
    mesh = _tri_mesh()
    smoothing = phx.discretization.fem.smoothing
    plan = smoothing.SmoothedElasticityPlan(
        "ES", mesh, smoothing.plane_stress_matrix(1.0, 0.3)
    )
    operator = plan.operator(mesh.coordinates)
    displacement = jnp.arange(10.0).reshape((5, 2))

    assert operator.mv(displacement).shape == displacement.shape
    assert operator.diagonal().shape == (10,)
    with pytest.raises(ValueError, match="entry budget"):
        operator.materialize(max_entries=10)
    assert operator.materialize(max_entries=100).shape == (10, 10)


def test_rejected_schedule_stage_restores_committed_state():
    law = phx.solver.TimeLaw.constant(1.0)

    def solve(state, start, end, time_law, args):
        return phx.solver.ScheduleStepResult(
            state=state + 10.0,
            accepted=jnp.asarray(False),
            diagnostics=jnp.asarray(1.0),
        )

    stage = phx.solver.SolveStage("reject", 0.0, 1.0, law, solve)
    final, results = phx.solver.SolveSchedule((stage,)).run(jnp.asarray(2.0))

    assert jnp.allclose(final, 2.0)
    assert jnp.allclose(results[0].state, 2.0)


def test_material_evaluate_uses_implicit_root_derivative():
    material = phx.equations.fem.LocalImplicitMaterial(
        lambda state, target: state**2 - target,
        lambda state, target: phx.equations.ConstitutiveResponse(state, state),
        state_shape=(1,),
        model_id="implicit-square-root",
    )
    initial = jnp.asarray([1.0])
    response, tangent = jax.jvp(
        lambda target: material.evaluate(initial, target).response,
        (jnp.asarray([4.0]),),
        (jnp.asarray([1.0]),),
    )

    assert jnp.allclose(response, 2.0)
    assert jnp.allclose(tangent, 0.25, atol=1.0e-8)


def test_smoothing_certificate_checks_full_affine_identity():
    mesh = _tri_mesh()
    smoothing = phx.discretization.fem.smoothing
    plan = smoothing.SmoothedElasticityPlan(
        "ES", mesh, smoothing.plane_stress_matrix(1.0, 0.3)
    )
    operator = plan.operator(mesh.coordinates)
    matrix = operator.materialize(max_entries=100)
    constrained = jnp.asarray(
        [True, True, False, True, False, False, False, False, False, False]
    )
    evidence = smoothing.certify_smoothing_operator(
        plan.layout,
        plan.geometry(mesh.coordinates),
        mesh.coordinates,
        matrix,
        constrained,
        1.0,
        0,
    )

    assert jnp.max(evidence.affine_reproduction_defect) < 1.0e-12
    assert jnp.max(evidence.closure_defect) < 1.0e-12


def test_ir_is_explicitly_lowered_but_not_advertised_as_executor():
    mesh = _tri_mesh()
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    form = phx.equations.WeakForm(
        "poisson",
        "u",
        (phx.equations.DiffusionTerm("u"),),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    action_ir = phx.equations.fem.lower_weak_form(form, discretization)

    assert action_ir.ir_id
    assert "action_ir" not in compiled.__dataclass_fields__
    assert "darcy_form" not in phx.equations.fem.__all__
    assert "sipg_poisson_ir" not in phx.equations.fem.__all__
