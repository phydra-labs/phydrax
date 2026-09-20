#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.amr._composite import CompositeAMRCellLayout
from phydrax.discretization.finite_volume._amr_diffusion import (
    composite_amr_multigrid_builder,
    CompositeAMRDiffusionPlan,
)


def _hierarchy(*, periodic=False):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    return phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(0, (4,), 2, halo_width=1),
            phx.discretization.BlockLevelPlan(1, (2,), 8, halo_width=1),
        ),
    )


def _topology(*, periodic=False, tagged_cell=2):
    hierarchy = _hierarchy(periodic=periodic)
    compiler = phx.discretization.BlockTopologyCompiler(hierarchy)
    initial = compiler.initial_topology()
    tags = jnp.zeros((2, 4), dtype="bool")
    tags = tags.at[tagged_cell // 4, tagged_cell % 4].set(True)
    result = compiler.compile(initial, (tags,))
    assert result.status.successful
    return result.topology


def _operator(*, periodic=False, dirichlet=False, coefficient=1.0):
    layout = CompositeAMRCellLayout(_topology(periodic=periodic), dtype=jnp.float64)
    boundaries = None
    if dirichlet:
        boundaries = {"x": ("dirichlet", "dirichlet")}
    plan = CompositeAMRDiffusionPlan(layout, boundaries=boundaries)
    return plan.prepare(coefficient)


def _vector(layout, *, phase=0.0):
    coordinates = jnp.sin(
        jnp.arange(layout.space.size, dtype=layout.dtype) * 0.37 + phase
    )
    return layout.space.unflatten(coordinates)


def _assert_tree_allclose(left, right, *, rtol=1.0e-12, atol=1.0e-12):
    for left_leaf, right_leaf in zip(
        jax.tree.leaves(left), jax.tree.leaves(right), strict=True
    ):
        np.testing.assert_allclose(left_leaf, right_leaf, rtol=rtol, atol=atol)


def _norm(space, value):
    return jnp.sqrt(jnp.real(space.inner(value, value)))


def test_composite_layout_uses_leaf_volumes_and_positive_dummy_weights():
    layout = CompositeAMRCellLayout(_topology(), dtype=jnp.float64)
    leaf = np.asarray(layout.flat_leaf_mask)
    measures = np.asarray(layout.cell_measures)

    assert layout.physical_cell_count == 9
    assert np.count_nonzero(~leaf) > 0
    np.testing.assert_allclose(measures[leaf].sum(), 1.0, rtol=0.0, atol=1.0e-14)
    assert np.all(measures[~leaf] > 0.0)
    assert layout.topology_fingerprint
    assert layout.layout_id


def test_composite_operator_is_linear_and_masked_identity_is_decoupled():
    operator = _operator(dirichlet=True)
    layout = operator.layout
    left = _vector(layout, phase=0.1)
    right = _vector(layout, phase=0.7)
    alpha, beta = 1.7, -0.4
    combination = jax.tree.map(lambda x, y: alpha * x + beta * y, left, right)
    expected = jax.tree.map(
        lambda x, y: alpha * x + beta * y,
        operator.mv(left),
        operator.mv(right),
    )

    _assert_tree_allclose(operator.mv(combination), expected)

    masked_coordinates = jnp.where(
        layout.flat_leaf_mask[:, None],
        0.0,
        jnp.arange(layout.cell_count, dtype=layout.dtype)[:, None] + 1.0,
    )
    masked = layout.unflatten_cells(masked_coordinates)
    image = layout.flatten_cells(operator.mv(masked))
    np.testing.assert_allclose(
        image[~layout.flat_leaf_mask], masked_coordinates[~layout.flat_leaf_mask]
    )
    np.testing.assert_allclose(image[layout.flat_leaf_mask], 0.0, atol=0.0)
    active_coordinates = jnp.where(
        layout.flat_leaf_mask[:, None],
        jnp.arange(layout.cell_count, dtype=layout.dtype)[:, None] + 1.0,
        0.0,
    )
    active = layout.unflatten_cells(active_coordinates)
    active_image = layout.flatten_cells(operator.mv(active))
    np.testing.assert_allclose(active_image[~layout.flat_leaf_mask], 0.0, atol=0.0)

    bad_rhs = layout.unflatten_cells(masked_coordinates)
    with pytest.raises(ValueError, match="must be zero on inactive and covered"):
        layout.require_zero_masked(bad_rhs)


def test_energy_weighted_adjoint_and_euclidean_transpose_are_exact_routes():
    operator = _operator(periodic=True, coefficient=2.5)
    layout = operator.layout
    left = _vector(layout, phase=0.2)
    right = _vector(layout, phase=0.9)
    left_image = operator.mv(left)
    right_image = operator.mv(right)

    assert float(operator.energy(left)) >= -1.0e-13
    np.testing.assert_allclose(
        layout.space.inner(left, right_image),
        layout.space.inner(left_image, right),
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    _assert_tree_allclose(operator.adjoint_mv(right), right_image)
    assert bool(jnp.any(operator.plan.routes.edge_periodic))

    policy = phx.linalg.MaterializationPolicy(max_entries=10_000, max_bytes=1_000_000)
    matrix = phx.linalg.materialize(operator, policy)
    right_coordinates = layout.space.flatten(right)
    transpose_coordinates = layout.space.flatten(operator.transpose_mv(right))
    np.testing.assert_allclose(
        transpose_coordinates,
        matrix.T @ right_coordinates,
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    metric = jnp.concatenate(
        tuple(weight.reshape((-1,)) for weight in layout.pairing_weights)
    )
    np.testing.assert_allclose(
        metric[:, None] * matrix,
        matrix.T * metric[None, :],
        rtol=2.0e-12,
        atol=2.0e-12,
    )
    diagonal = phx.linalg.assemble_diagonal(operator)
    np.testing.assert_allclose(
        diagonal,
        jnp.diagonal(matrix),
        rtol=2.0e-12,
        atol=2.0e-12,
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_periodic_and_neumann_constant_kernel_compatibility_and_zero_mean_gauge(
    periodic,
):
    operator = _operator(periodic=periodic)
    layout = operator.layout
    constant = layout.space.unflatten(layout.constant_mode_coordinates()[:, 0])
    residual = layout.space.flatten(operator.mv(constant))
    np.testing.assert_allclose(residual, 0.0, atol=2.0e-13)
    assert operator.has_constant_nullspace
    assert operator.constant_nullspace() is not None

    incompatible = layout.zero_masked(
        layout.space.unflatten(jnp.ones(layout.space.size, dtype=layout.dtype))
    )
    assert float(jnp.abs(operator.compatibility_defect(incompatible))) > 0.0
    with pytest.raises(ValueError, match="incompatible with the constant nullspace"):
        operator.require_compatible_rhs(incompatible)
    compatible = operator.project_compatible_rhs(incompatible)
    np.testing.assert_allclose(
        operator.compatibility_defect(compatible), 0.0, atol=2.0e-14
    )
    if not periodic:
        neumann_lift = operator.boundary_rhs_lift({"x": (1.0, -1.0)})
        assert float(_norm(layout.space, neumann_lift)) > 0.0
        np.testing.assert_allclose(
            operator.compatibility_defect(neumann_lift), 0.0, atol=2.0e-14
        )

    gauged = operator.zero_mean_gauge(_vector(layout, phase=0.4))
    np.testing.assert_allclose(layout.integral(gauged), 0.0, atol=2.0e-14)
    system = operator.linear_system(compatibility="project", gauge="project")
    assert system.nullspace_policy is not None
    assert system.nullspace_policy.right is system.nullspace_policy.left


def test_dirichlet_operator_is_definite_and_boundary_data_is_only_rhs_lift():
    operator = _operator(dirichlet=True, coefficient=3.0)
    layout = operator.layout
    policy = phx.linalg.MaterializationPolicy(max_entries=10_000, max_bytes=1_000_000)
    matrix = phx.linalg.materialize(operator, policy)
    metric = jnp.concatenate(
        tuple(weight.reshape((-1,)) for weight in layout.pairing_weights)
    )
    square_root = jnp.sqrt(metric)
    hilbert_matrix = square_root[:, None] * matrix / square_root[None, :]
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (hilbert_matrix + hilbert_matrix.T))

    assert operator.properties.certifies("positive_definite")
    assert float(jnp.min(eigenvalues)) > 0.0
    assert not operator.has_constant_nullspace
    zero = layout.space.zeros()
    _assert_tree_allclose(operator.mv(zero), zero, atol=0.0)

    lift = operator.boundary_rhs_lift({"x": (1.25, -0.5)})
    assert float(_norm(layout.space, lift)) > 0.0
    assert operator.operator_id == operator.plan.prepare(3.0).operator_id


def test_harmonic_mortar_weights_and_integrated_interface_flux_cancel():
    topology = _topology()
    layout = CompositeAMRCellLayout(topology, dtype=jnp.float64)
    coefficients = tuple(
        jnp.where(mask, 1.0 + level, jnp.nan)
        for level, mask in enumerate(layout.leaf_mask)
    )
    operator = CompositeAMRDiffusionPlan(layout).prepare(coefficients)
    routes = operator.plan.routes
    flat_coefficient = jnp.concatenate(
        tuple(value.reshape((-1,)) for value in operator.coefficient)
    )
    expected = routes.edge_area / (
        routes.edge_left_distance / flat_coefficient[routes.edge_left]
        + routes.edge_right_distance / flat_coefficient[routes.edge_right]
    )
    np.testing.assert_allclose(operator.edge_weights, expected, rtol=1.0e-14)
    assert bool(jnp.any(routes.edge_level_jump))

    left_flux, right_flux = operator.interface_flux_contributions(
        _vector(layout, phase=0.3)
    )
    np.testing.assert_allclose(left_flux + right_flux, 0.0, atol=0.0)
    assert float(jnp.max(jnp.abs(left_flux))) > 0.0
    assert operator.plan.route_fingerprint == routes.route_id
    assert operator.plan.precision_fingerprint == operator.plan.precision.policy_id
    assert operator.coefficient_fingerprint
    assert operator.numeric_fingerprint


def test_composite_topology_mismatch_is_refused():
    first = _topology(tagged_cell=2)
    second = _topology(tagged_cell=5)
    layout = CompositeAMRCellLayout(first, dtype=jnp.float64)
    plan = CompositeAMRDiffusionPlan(layout)

    with pytest.raises(ValueError, match="exact fixed topology epoch"):
        layout.require_topology(second)
    with pytest.raises(ValueError, match="exact fixed topology epoch"):
        plan.require_topology(second)


def test_dirichlet_composite_operator_solves_with_ordinary_krylov():
    operator = _operator(dirichlet=True)
    rhs = operator.prepare_rhs(1.0, boundary_data={"x": (0.0, 1.0)})
    result = phx.linalg.solve(
        operator.linear_system(),
        rhs,
        policy=phx.linalg.LinearSolvePolicy(
            phx.linalg.ConjugateGradient(),
            tolerance=phx.linalg.TolerancePolicy(
                relative=1.0e-10,
                absolute=1.0e-12,
                max_steps=200,
            ),
        ),
    )
    residual = jax.tree.map(
        lambda target, image: target - image,
        rhs,
        operator.mv(result.value),
    )

    assert bool(result.successful)
    assert float(_norm(operator.source, residual)) < 1.0e-8 * float(
        _norm(operator.source, rhs)
    )


def test_fv_side_native_v_cycle_contracts_composite_residual():
    operator = _operator(dirichlet=True)
    fine_size = operator.source.size
    coarse_size = (fine_size + 1) // 2
    coarse_space = phx.linalg.ArraySpace((coarse_size,), dtype=jnp.float64)
    prolongation_matrix = jnp.zeros((fine_size, coarse_size), dtype=jnp.float64)
    prolongation_matrix = prolongation_matrix.at[
        jnp.arange(fine_size), jnp.arange(fine_size) // 2
    ].set(1.0)
    prolongation = phx.linalg.DenseLinearOperator(
        prolongation_matrix,
        source=coarse_space,
        target=operator.source,
    )
    restriction = phx.linalg.adjoint(prolongation)
    materialization = phx.linalg.MaterializationPolicy(
        max_entries=100_000,
        max_bytes=2_000_000,
    )
    fine_matrix = phx.linalg.materialize(operator, materialization)
    restriction_matrix = phx.linalg.materialize(restriction, materialization)
    coarse_operator = phx.linalg.DenseLinearOperator(
        restriction_matrix @ fine_matrix @ prolongation_matrix,
        source=coarse_space,
        target=coarse_space,
    )
    builder = composite_amr_multigrid_builder(
        (operator, coarse_operator),
        (
            phx.linalg.JacobiPreconditionerBuilder(relaxation=2.0 / 3.0),
            phx.linalg.DenseInversePreconditionerBuilder(),
        ),
        (restriction,),
        (prolongation,),
        coarse_operator_source="direct",
    )
    cycle = builder.prepare(operator, materialization=materialization)
    rhs = operator.prepare_rhs(1.0)
    correction = cycle.apply(rhs)
    remaining = jax.tree.map(
        lambda target, image: target - image,
        rhs,
        operator.mv(correction),
    )

    assert isinstance(cycle, phx.linalg.MultigridPreconditioner)
    assert float(_norm(operator.source, remaining)) < float(_norm(operator.source, rhs))
