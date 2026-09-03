#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import opt_einsum as oe
import pytest

import phydrax as phx


def _space(*, component_shape=()):
    coordinates = jnp.asarray(
        tuple((float(i), float(j)) for j in range(3) for i in range(3))
    )
    cells = (
        (0, 1, 4, 3),
        (1, 2, 5, 4),
        (3, 4, 7, 6),
        (4, 5, 8, 7),
    )
    mesh = phx.discretization.CellMesh.from_polygons(coordinates, cells)
    field = phx.discretization.ExplicitPolygonH1FieldSpec(
        "u", component_shape=component_shape
    )
    return phx.discretization.ExplicitPolygonH1Plan(mesh, field).prepare()


def test_auto_capability_selects_dense_matrix_free_and_affine_patch_solves():
    space = _space()
    constraint = phx.discretization.explicit_polygon_h1_dirichlet_constraint(space, "u")
    form = phx.equations.FiniteElementForm(
        "affine-polygon",
        "u",
        (
            phx.equations.DiffusionAction("u"),
            phx.equations.SourceAction("u", 0.0),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=constraint,
        dirichlet_values=lambda points: points[..., 0] + 2.0 * points[..., 1],
    )
    problem, right_hand_side = compiled.linear_system()
    result = phx.linalg.solve(problem, right_hand_side)
    solution = compiled.expand(result.value)
    expected = space.mesh.coordinates[:, 0] + 2.0 * space.mesh.coordinates[:, 1]

    assert jnp.all(result.successful)
    assert jnp.allclose(solution, expected, atol=1e-10, rtol=1e-10)
    assert all(
        workset.signature.local_kernel == "dense"
        and workset.signature.operator_realization == "matrix_free"
        and workset.signature.provider_selection_id is not None
        for workset in compiled._workset_program.worksets
    )


def test_tensor_diffusion_uses_prepared_polygon_capability():
    space = _space()
    state = jnp.linspace(-0.4, 0.7, space.dof_map.global_dof_count)
    scalar = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "scalar-polygon-diffusion",
            "u",
            (phx.equations.DiffusionAction("u", 2.0),),
        ),
        space,
    )
    tensor = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "tensor-polygon-diffusion",
            "u",
            (
                phx.equations.TensorDiffusionAction(
                    "u", 2.0 * jnp.eye(2), action_id="polygon-tensor-diffusion"
                ),
            ),
        ),
        space,
    )

    assert jnp.allclose(
        tensor.full_residual(state, None),
        scalar.full_residual(state, None),
        atol=1e-10,
        rtol=1e-10,
    )
    assert all(
        workset.signature.provider_selection_id is not None
        for workset in tensor._workset_program.worksets
    )


def test_cell_actions_and_exact_transposes_use_the_same_basis():
    space = _space(component_shape=(2,))
    region = space.prepare_local_regions(
        space.cell_domain,
        field_names=("u",),
        maximum_derivative_order=1,
        kernel_mode="dense",
    )[0]
    reference = region.reference_actions[0]
    runtime = space.default_runtime
    local = jnp.arange(region.entity_indices.size * 4 * 2, dtype=float).reshape(
        (region.entity_indices.size, 4, 2)
    )
    value_dual = jnp.linspace(
        -0.3, 0.8, region.entity_indices.size * reference.point_count * 2
    ).reshape((region.entity_indices.size, reference.point_count, 2))
    gradient_dual = jnp.linspace(
        -0.5, 0.7, region.entity_indices.size * reference.point_count * 2 * 2
    ).reshape((region.entity_indices.size, reference.point_count, 2, 2))

    values = reference.interpolate(runtime, local)
    value_transpose = reference.interpolate_transpose(runtime, value_dual)
    gradients = reference.reference_gradient(runtime, local)
    gradient_transpose = reference.reference_gradient_transpose(runtime, gradient_dual)

    assert jnp.allclose(
        oe.contract("cqk,cqk->", values, value_dual),
        oe.contract("cik,cik->", local, value_transpose),
        atol=1e-10,
    )
    assert jnp.allclose(
        oe.contract("cqkd,cqkd->", gradients, gradient_dual),
        oe.contract("cik,cik->", local, gradient_transpose),
        atol=1e-10,
    )


def test_boundary_load_and_functional_use_exterior_and_cell_worksets():
    space = _space()
    boundary_form = phx.equations.FiniteElementForm(
        "polygon-boundary",
        "u",
        (phx.equations.BoundaryLoadAction("u", 1.0),),
    )
    boundary = phx.equations.compile_finite_element_problem(boundary_form, space)
    load = -boundary.full_residual(jnp.zeros((9,)), None)

    functional = phx.variational.Functional(
        "polygon-integral",
        (
            phx.variational.LocalIntegralTerm(
                "integral",
                region="body",
                fields=(phx.variational.FieldJetSpec("u", value=True),),
                density=lambda fields, geometry, context: fields["u"].value,
                density_id="explicit-polygon-integral",
            ),
        ),
        variable_fields=("u",),
    )
    compiled_functional = phx.equations.compile_finite_element_functional(
        functional,
        space,
        fields={"u": "u"},
        regions={"body": None},
    )

    assert jnp.allclose(jnp.sum(load), 8.0, atol=1e-10)
    assert all(
        workset.signature.region_kind == "exterior-facet"
        for workset in boundary._workset_program.worksets
    )
    assert jnp.allclose(compiled_functional.potential(jnp.ones((9,))), 4.0, atol=1e-10)


def test_vector_constraint_selects_components_and_unoffered_modes_fail():
    space = _space(component_shape=(2,))
    constraint = phx.discretization.explicit_polygon_h1_dirichlet_constraint(
        space, "u", components=(0,)
    )
    lift = constraint.lift(
        lambda points: jnp.stack((points[:, 0], points[:, 1]), axis=-1)
    )
    assert jnp.allclose(lift[:, 1], 0.0)
    assert jnp.count_nonzero(lift[:, 0]) > 0

    form = phx.equations.FiniteElementForm(
        "unsupported-sparse",
        "u",
        (phx.equations.DiffusionAction("u"),),
    )
    with pytest.raises(ValueError, match="does not offer|requested local variational"):
        phx.equations.compile_finite_element_problem(
            form,
            space,
            execution_policy=phx.equations.FiniteElementExecutionPolicy(
                realization="sparse", local_kernel="dense"
            ),
        )
