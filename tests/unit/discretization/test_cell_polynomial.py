#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization import (
    PreparedUnstructuredFiniteVolumeCoupling,
    UnstructuredFiniteVolumeCouplingPlan,
)


def _quadrilateral_grid(nx=4, ny=4):
    logical = np.asarray([(i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)])
    x = logical[:, 0]
    y = logical[:, 1]
    vertices = np.stack((x + 0.15 * x * y, y + 0.08 * x * (1.0 - y)), axis=-1)
    quadrilaterals = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            quadrilaterals.append((lower_left, lower_right, upper_right, upper_left))
    return vertices, np.asarray(quadrilaterals, dtype=np.int32)


def _tetrahedral_grid(resolution):
    vertices = np.asarray(
        [
            (i / resolution, j / resolution, k / resolution)
            for k in range(resolution + 1)
            for j in range(resolution + 1)
            for i in range(resolution + 1)
        ]
    )

    def vertex(i, j, k):
        return k * (resolution + 1) ** 2 + j * (resolution + 1) + i

    tetrahedra = []
    for k in range(resolution):
        for j in range(resolution):
            for i in range(resolution):
                lower = vertex(i, j, k)
                x = vertex(i + 1, j, k)
                y = vertex(i, j + 1, k)
                xy = vertex(i + 1, j + 1, k)
                z = vertex(i, j, k + 1)
                xz = vertex(i + 1, j, k + 1)
                yz = vertex(i, j + 1, k + 1)
                upper = vertex(i + 1, j + 1, k + 1)
                tetrahedra.extend(
                    (
                        (lower, x, xy, upper),
                        (lower, xy, y, upper),
                        (lower, y, yz, upper),
                        (lower, yz, z, upper),
                        (lower, z, xz, upper),
                        (lower, xz, x, upper),
                    )
                )
    return vertices, np.asarray(tetrahedra, dtype=np.int32)


def _cell_averages(discretization, function):
    values = function(discretization.cell_quadrature_points)
    return (
        jnp.sum(discretization.cell_quadrature_weights * values, axis=1)
        / discretization.cell_volumes
    )


def _quadratic(points):
    x = points[..., 0]
    y = points[..., 1]
    return 0.7 + 0.3 * x - 0.2 * y + 0.4 * x**2 - 0.25 * x * y + 0.15 * y**2


def _cubic_3d(points):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    return (
        0.6
        + 0.2 * x
        - 0.3 * y
        + 0.1 * z
        + 0.4 * x**2
        - 0.15 * x * y
        + 0.2 * y * z
        - 0.1 * z**2
        + 0.25 * x**3
        - 0.2 * x * y * z
        + 0.12 * y**2 * z
        + 0.08 * z**3
    )


def _cubic_advection_residual(points, velocity):
    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]
    derivative_x = 0.2 + 0.8 * x - 0.15 * y + 0.75 * x**2 - 0.2 * y * z
    derivative_y = -0.3 - 0.15 * x + 0.2 * z - 0.2 * x * z + 0.24 * y * z
    derivative_z = 0.1 + 0.2 * y - 0.2 * z - 0.2 * x * y + 0.12 * y**2 + 0.24 * z**2
    return -(
        velocity[0] * derivative_x
        + velocity[1] * derivative_y
        + velocity[2] * derivative_z
    )


def _scalar_system(velocity):
    speed = jnp.asarray(velocity)
    return phx.equations.ScalarConservationSystem(
        speed.size,
        lambda state, axis, args: speed[axis] * state,
        lambda left, right, axis, args: jnp.full(
            left.shape[:-1], jnp.abs(speed[axis]), dtype=left.dtype
        ),
        system_id="unstructured-polynomial-advection",
    )


def _coupling_mesh_plan(
    nx=4,
    ny=4,
    *,
    x_offset=0.0,
    component_names=None,
):
    vertices = np.asarray(
        [(x_offset + i / nx, j / ny) for j in range(ny + 1) for i in range(nx + 1)]
    )
    quadrilaterals = []
    for j in range(ny):
        for i in range(nx):
            lower_left = j * (nx + 1) + i
            lower_right = lower_left + 1
            upper_left = lower_left + nx + 1
            upper_right = upper_left + 1
            quadrilaterals.append((lower_left, lower_right, upper_right, upper_left))
    options = {} if component_names is None else {"component_names": component_names}
    return phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=np.asarray(quadrilaterals, dtype=np.int32),
        **options,
    )


def _compile_scalar_coupling(
    discretization,
    coupling: UnstructuredFiniteVolumeCouplingPlan | None = None,
    *,
    reconstruction=None,
    interface_solver=None,
):
    system = _scalar_system((0.4, -0.15))
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        (
            phx.discretization.PiecewiseConstantReconstruction()
            if reconstruction is None
            else reconstruction
        ),
        (
            phx.discretization.RusanovFluxPlan()
            if interface_solver is None
            else interface_solver
        ),
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "unstructured-coupling",
        "state",
        system,
        boundaries,
    )
    return phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        coupling=coupling,
    )


def _two_material_system():
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.IdealGasMaterial(1.4),
        phx.equations.StiffenedGasMaterial(4.4, 2.0, 1.0),
    )
    return phx.equations.TwoMaterialVOFSystem(2, eos=eos)


def _compile_two_material_coupling(
    discretization,
    coupling: UnstructuredFiniteVolumeCouplingPlan | None = None,
    *,
    reconstruction=None,
):
    system = _two_material_system()
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        (
            phx.discretization.PiecewiseConstantReconstruction()
            if reconstruction is None
            else reconstruction
        ),
        phx.discretization.RusanovFluxPlan(),
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "two-material-vof-coupling",
        "state",
        system,
        boundaries,
    )
    return phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        coupling=coupling,
    )


def _coupling_hierarchy(coarse, *, nx=4, ny=4, x_offset=0.0):
    fine = _coupling_mesh_plan(2 * nx, 2 * ny, x_offset=x_offset).prepare()
    parent = np.asarray(
        [(j // 2) * nx + i // 2 for j in range(2 * ny) for i in range(2 * nx)],
        dtype=np.int32,
    )
    prolongation = phx.discretization.UnstructuredConservativeRemapPlan(
        coarse,
        fine,
        np.arange(fine.cell_count + 1, dtype=np.int32),
        parent,
        fine.cell_volumes,
        method="coupling-test-prolongation",
        provenance="analytic-uniform-refinement",
    )
    children = np.asarray(
        [
            fine_j * (2 * nx) + fine_i
            for coarse_j in range(ny)
            for coarse_i in range(nx)
            for fine_j in (2 * coarse_j, 2 * coarse_j + 1)
            for fine_i in (2 * coarse_i, 2 * coarse_i + 1)
        ],
        dtype=np.int32,
    )
    restriction = phx.discretization.UnstructuredConservativeRemapPlan(
        fine,
        coarse,
        np.arange(0, 4 * coarse.cell_count + 1, 4, dtype=np.int32),
        children,
        fine.cell_volumes[children],
        method="coupling-test-restriction",
        provenance="analytic-uniform-refinement",
    )
    return phx.discretization.UnstructuredAMRHierarchyPlan(
        coarse,
        fine,
        prolongation,
        restriction,
    )


def _current_coupling_artifacts(base_plan, discretization):
    motion = phx.discretization.FixedConnectivityMotionPlan(
        base_plan,
        lambda time, vertices, args: vertices,
        mapping_id="stationary-current-geometry",
    )
    embedded_boundary = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: jnp.ones((points.shape[0],)),
        field_id="stationary-full-fluid",
        body_tag=7,
    )
    embedded_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {7: phx.discretization.SlipWallBoundary()}
    )
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    vof = phx.discretization.UnstructuredVOFPlan(discretization, gradient)
    amr = _coupling_hierarchy(discretization)
    cell_count = discretization.cell_count
    overset = phx.discretization.UnstructuredOversetPlan(
        discretization,
        discretization,
        np.arange(cell_count, dtype=np.int32),
        np.arange(cell_count + 1, dtype=np.int32),
        np.arange(cell_count, dtype=np.int32),
        discretization.cell_volumes,
    )
    sliding = phx.discretization.PeriodicSlidingInterfacePlan(
        np.asarray((0.0, 0.5, 1.0)),
        np.asarray((0.0, 0.25, 0.75, 1.0)),
        1.0,
        interface_id="current-periodic-seam",
    )
    return (
        motion,
        embedded_boundary,
        embedded_boundaries,
        vof,
        amr,
        overset,
        sliding,
    )


def _stationary_embedded_coupling(
    discretization,
    *,
    level_set=None,
    field_id="stationary-cut",
    body_tag=7,
    stabilization_policy=None,
):
    level_set_ = (
        (lambda points, args: points[:, 0] - 0.43) if level_set is None else level_set
    )
    embedded_boundary = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        level_set_,
        field_id=field_id,
        body_tag=body_tag,
        stabilization_policy=stabilization_policy,
    )
    embedded_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {body_tag: phx.discretization.SlipWallBoundary()}
    )
    coupling = UnstructuredFiniteVolumeCouplingPlan(
        embedded_boundary=embedded_boundary,
        embedded_boundaries=embedded_boundaries,
    )
    return coupling, embedded_boundary, embedded_boundaries


def test_unstructured_coupling_types_are_exported_from_discretization_root():
    assert (
        UnstructuredFiniteVolumeCouplingPlan
        is phx.discretization.finite_volume.UnstructuredFiniteVolumeCouplingPlan
    )
    assert (
        PreparedUnstructuredFiniteVolumeCoupling
        is phx.discretization.finite_volume.PreparedUnstructuredFiniteVolumeCoupling
    )
    assert (
        phx.discretization.UnstructuredEmbeddedBoundarySet
        is phx.discretization.finite_volume.UnstructuredEmbeddedBoundarySet
    )


@pytest.mark.parametrize(
    "reconstruction_plan",
    (
        pytest.param(
            phx.discretization.CellPolynomialReconstructionPlan(1),
            id="cell-polynomial",
        ),
        pytest.param(
            phx.discretization.UnstructuredWENOZReconstructionPlan(2, limiter="none"),
            id="weno-z",
        ),
    ),
)
def test_moving_unstructured_rejects_static_high_order_at_compile_and_dynamics(
    reconstruction_plan,
):
    base_plan = _coupling_mesh_plan()
    discretization = base_plan.prepare()
    reconstruction = reconstruction_plan.prepare(discretization)
    static_compiled = _compile_scalar_coupling(
        discretization,
        reconstruction=reconstruction,
    )
    assert static_compiled.dynamics.coupling.motion is None
    assert static_compiled.method.reconstruction is reconstruction
    motion = phx.discretization.FixedConnectivityMotionPlan(
        base_plan,
        lambda time, vertices, args: vertices,
        mapping_id="high-order-rejection-motion",
    )
    coupling = UnstructuredFiniteVolumeCouplingPlan(motion=motion)
    prepared_coupling = coupling.prepare(discretization)

    with pytest.raises(ValueError) as compile_error:
        _compile_scalar_coupling(
            discretization,
            coupling,
            reconstruction=reconstruction,
        )
    assert prepared_coupling.prepared_id in str(compile_error.value)
    assert static_compiled.method.method_id in str(compile_error.value)
    assert type(reconstruction).__name__ in str(compile_error.value)

    with pytest.raises(ValueError) as dynamics_error:
        phx.discretization.PreparedUnstructuredFiniteVolumeDynamics(
            static_compiled.problem.system,
            discretization,
            static_compiled.method,
            static_compiled.problem.boundaries,
            coupling=prepared_coupling,
        )
    assert prepared_coupling.prepared_id in str(dynamics_error.value)
    assert static_compiled.method.method_id in str(dynamics_error.value)
    assert type(reconstruction).__name__ in str(dynamics_error.value)


def test_moving_unstructured_piecewise_constant_compiles():
    base_plan = _coupling_mesh_plan()
    discretization = base_plan.prepare()
    motion = phx.discretization.FixedConnectivityMotionPlan(
        base_plan,
        lambda time, vertices, args: vertices,
        mapping_id="piecewise-constant-motion",
    )
    compiled = _compile_scalar_coupling(
        discretization,
        UnstructuredFiniteVolumeCouplingPlan(motion=motion),
    )

    assert compiled.dynamics.coupling.motion is motion
    assert (
        type(compiled.method.reconstruction)
        is phx.discretization.PiecewiseConstantReconstruction
    )


def test_two_material_vof_rejects_rusanov_without_vof_and_high_order_stage_path():
    system = _two_material_system()
    discretization = _coupling_mesh_plan(component_names=system.component_names).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    coupling = UnstructuredFiniteVolumeCouplingPlan(
        vof=phx.discretization.UnstructuredVOFPlan(discretization, gradient)
    )

    with pytest.raises(ValueError, match="requires prepared unstructured VOF coupling"):
        _compile_two_material_coupling(discretization)
    with pytest.raises(ValueError, match="PiecewiseConstantReconstruction"):
        _compile_two_material_coupling(
            discretization,
            coupling,
            reconstruction=gradient,
        )


def test_two_material_vof_valid_compile_preserves_geometry_and_identities():
    system = _two_material_system()
    discretization = _coupling_mesh_plan(component_names=system.component_names).prepare()
    gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )
    first_vof = phx.discretization.UnstructuredVOFPlan(discretization, gradient)
    second_vof = phx.discretization.UnstructuredVOFPlan(
        discretization,
        gradient,
        bisection_iterations=61,
    )
    first_plan = UnstructuredFiniteVolumeCouplingPlan(vof=first_vof)
    second_plan = UnstructuredFiniteVolumeCouplingPlan(vof=second_vof)

    first = _compile_two_material_coupling(discretization, first_plan)
    second = _compile_two_material_coupling(discretization, second_plan)
    prepared = first.dynamics.coupling

    assert prepared.vof is first_vof
    assert prepared.topology_id == first_vof.discretization.topology_id
    assert prepared.geometry_id == first_vof.discretization.geometry_id
    assert prepared.discretization_id == first_vof.discretization.prepared_id
    assert prepared.discretization_id == first_vof.gradient.discretization.prepared_id
    assert (
        type(first.method.reconstruction)
        is phx.discretization.PiecewiseConstantReconstruction
    )
    assert first_plan.plan_id != second_plan.plan_id
    assert prepared.prepared_id != second.dynamics.coupling.prepared_id
    assert first.dynamics.dynamics_id != second.dynamics.dynamics_id
    assert first.compilation_id != second.compilation_id


def test_two_material_vof_rejects_equal_size_moved_geometry_and_stale_gradient():
    system = _two_material_system()
    current = _coupling_mesh_plan(component_names=system.component_names).prepare()
    moved = _coupling_mesh_plan(
        x_offset=0.125,
        component_names=system.component_names,
    ).prepare()
    moved_gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(moved)
    moved_vof = phx.discretization.UnstructuredVOFPlan(moved, moved_gradient)

    assert moved.cell_count == current.cell_count
    assert moved.topology_id == current.topology_id
    assert moved.geometry_id != current.geometry_id
    assert moved.prepared_id != current.prepared_id
    with pytest.raises(ValueError, match="VOF plan belongs to stale"):
        _compile_two_material_coupling(
            current,
            UnstructuredFiniteVolumeCouplingPlan(vof=moved_vof),
        )

    current_gradient = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        current
    )
    current_vof = phx.discretization.UnstructuredVOFPlan(
        current,
        current_gradient,
    )
    stale_gradient_vof = eqx.tree_at(
        lambda candidate: candidate.gradient,
        current_vof,
        moved_gradient,
    )
    with pytest.raises(ValueError, match="VOF gradient belongs to stale"):
        _compile_two_material_coupling(
            current,
            UnstructuredFiniteVolumeCouplingPlan(vof=stale_gradient_vof),
        )


def test_cell_polynomial_is_k_exact_on_mapped_quadrilaterals():
    vertices, quadrilaterals = _quadrilateral_grid()
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=quadrilaterals
    ).prepare()
    reconstruction = phx.discretization.CellPolynomialReconstructionPlan(2).prepare(
        discretization
    )
    state = _cell_averages(discretization, _quadratic)

    routes = jnp.arange(discretization.cell_count, dtype=jnp.int32)
    basis = reconstruction.basis_values(routes, discretization.cell_quadrature_points)
    basis_average = (
        jnp.sum(discretization.cell_quadrature_weights[..., None] * basis, axis=1)
        / discretization.cell_volumes[:, None]
    )
    np.testing.assert_allclose(basis_average, 0.0, atol=2e-12)

    left, right = reconstruction.reconstruct_at(
        state, discretization.face_quadrature_points
    )
    exact = _quadratic(discretization.face_quadrature_points)
    np.testing.assert_allclose(left, exact, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(right, exact, rtol=2e-10, atol=2e-10)
    left_jit, right_jit = eqx.filter_jit(reconstruction.reconstruct_at)(
        state, discretization.face_quadrature_points
    )
    np.testing.assert_allclose(left_jit, exact, rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(right_jit, exact, rtol=2e-10, atol=2e-10)
    gradient = jax.grad(
        lambda values: jnp.sum(
            reconstruction.reconstruct_at(values, discretization.face_quadrature_points)[
                0
            ]
        )
    )(state)
    assert jnp.all(jnp.isfinite(gradient))
    assert reconstruction.report.minimum_rank == reconstruction.basis.feature_count


def test_degree_one_cell_polynomial_is_affine_exact_on_tetrahedra():
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0),
            (-1.0, 0.0, 0.0),
            (0.0, -1.0, 0.0),
            (0.0, 0.0, -1.0),
        )
    )
    tetrahedra = np.asarray(
        (
            (0, 1, 2, 3),
            (1, 2, 3, 4),
            (0, 2, 3, 5),
            (0, 1, 3, 6),
            (0, 1, 2, 7),
        )
    )
    system = _scalar_system((0.2, -0.1, 0.3))
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        tetrahedra=tetrahedra,
        component_names=system.component_names,
    ).prepare()
    reconstruction = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )

    def affine(points):
        return 0.4 + 0.2 * points[..., 0] - 0.3 * points[..., 1] + 0.5 * points[..., 2]

    state = _cell_averages(discretization, affine)
    left, right = reconstruction.reconstruct_at(
        state, discretization.face_quadrature_points
    )
    exact = affine(discretization.face_quadrature_points)
    np.testing.assert_allclose(left, exact, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(right, exact, rtol=2e-11, atol=2e-11)

    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        reconstruction, phx.discretization.RusanovFluxPlan()
    )
    problem = phx.equations.ConservationProblemIR(
        "tetrahedral-affine-advection", "state", system, boundaries
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    expected_residual = -(0.2 * 0.2 + (-0.1) * (-0.3) + 0.3 * 0.5)
    np.testing.assert_allclose(
        compiled(jnp.asarray(0.0), state[:, None]),
        expected_residual,
        rtol=2e-10,
        atol=2e-10,
    )


def test_degree_three_tetrahedral_advection_residual_is_cubic_exact_under_refinement():
    velocity = np.asarray((0.3, -0.2, 0.25))
    maximum_errors = []
    for resolution in (2, 3):
        vertices, tetrahedra = _tetrahedral_grid(resolution)
        system = _scalar_system(velocity)
        discretization = phx.discretization.UnstructuredFiniteVolumePlan(
            vertices,
            tetrahedra=tetrahedra,
            component_names=system.component_names,
        ).prepare()
        reconstruction = phx.discretization.CellPolynomialReconstructionPlan(
            3, oversampling=4
        ).prepare(discretization)
        assert reconstruction.report.degree == 3
        assert reconstruction.report.minimum_rank == reconstruction.basis.feature_count
        state = _cell_averages(discretization, _cubic_3d)

        left, right = reconstruction.reconstruct_at(
            state, discretization.face_quadrature_points
        )
        exact_faces = _cubic_3d(discretization.face_quadrature_points)
        np.testing.assert_allclose(left, exact_faces, rtol=2e-9, atol=2e-9)
        interior = np.asarray(discretization.neighbour_cells) >= 0
        np.testing.assert_allclose(
            np.asarray(right)[interior],
            np.asarray(exact_faces)[interior],
            rtol=2e-9,
            atol=2e-9,
        )

        boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
            discretization.boundary_patch_names,
            {
                name: phx.discretization.ExtrapolationBoundary()
                for name in discretization.boundary_patch_names
            },
        )
        method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            reconstruction, phx.discretization.RusanovFluxPlan()
        )
        problem = phx.equations.ConservationProblemIR(
            f"tetrahedral-cubic-advection-{resolution}",
            "state",
            system,
            boundaries,
        )
        compiled = phx.equations.compile_conservation_problem(
            problem, discretization, method
        )
        observed = compiled(jnp.asarray(0.0), state[:, None])[:, 0]
        expected = _cell_averages(
            discretization,
            lambda points: _cubic_advection_residual(points, velocity),
        )
        error = float(jnp.max(jnp.abs(observed - expected)))
        maximum_errors.append(error)
        np.testing.assert_allclose(observed, expected, rtol=2e-8, atol=2e-8)

    assert max(maximum_errors) < 2e-8


def test_cell_polynomial_rejects_rank_deficient_components():
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        quadrilaterals=np.asarray(((0, 1, 2, 3),)),
    ).prepare()
    with pytest.raises(ValueError, match="fewer cells than features"):
        phx.discretization.CellPolynomialReconstructionPlan(1).prepare(discretization)


def test_unstructured_weno_z_has_exact_smooth_limit_and_jittable_gradient():
    vertices, quadrilaterals = _quadrilateral_grid()
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=quadrilaterals
    ).prepare()
    reconstruction = phx.discretization.UnstructuredWENOZReconstructionPlan(
        2, limiter="none"
    ).prepare(discretization)

    def affine(points):
        return 0.8 + 0.25 * points[..., 0] - 0.15 * points[..., 1]

    state = _cell_averages(discretization, affine)
    left, right = reconstruction.reconstruct_at(
        state, discretization.face_quadrature_points
    )
    exact = affine(discretization.face_quadrature_points)
    np.testing.assert_allclose(left, exact, rtol=5e-10, atol=5e-10)
    np.testing.assert_allclose(right, exact, rtol=5e-10, atol=5e-10)
    left_jit, right_jit = eqx.filter_jit(reconstruction.reconstruct_at)(
        state, discretization.face_quadrature_points
    )
    np.testing.assert_allclose(left_jit, exact, rtol=5e-10, atol=5e-10)
    np.testing.assert_allclose(right_jit, exact, rtol=5e-10, atol=5e-10)
    gradient = jax.grad(
        lambda values: jnp.sum(
            reconstruction.reconstruct_at(values, discretization.face_quadrature_points)[
                0
            ]
        )
    )(state)
    assert jnp.all(jnp.isfinite(gradient))


def test_unstructured_weno_sector_stencils_remain_compact_under_refinement_and_rotation():
    maximum_depths = []
    for cells, angle in ((8, 0.0), (16, 0.0), (8, 0.63), (16, 0.63)):
        vertices, quadrilaterals = _quadrilateral_grid(cells, cells)
        rotation = np.asarray(
            (
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            )
        )
        vertices = vertices @ rotation.T
        discretization = phx.discretization.UnstructuredFiniteVolumePlan(
            vertices, quadrilaterals=quadrilaterals
        ).prepare()
        reconstruction = phx.discretization.UnstructuredWENOZReconstructionPlan().prepare(
            discretization
        )
        depths = [
            int(sector.report.maximum_stencil_depth) for sector in reconstruction.sectors
        ]
        maximum_depths.append(max(depths))
        central_cell = int(
            jnp.argmin(
                jnp.linalg.norm(
                    discretization.cell_centers
                    - jnp.mean(discretization.cell_centers, axis=0),
                    axis=-1,
                )
            )
        )
        sector_stencils = {
            tuple(
                np.asarray(sector.stencil_cells[central_cell])[
                    np.asarray(sector.stencil_valid[central_cell])
                ].tolist()
            )
            for sector in reconstruction.sectors
        }
        assert len(sector_stencils) >= 3
    assert max(maximum_depths) <= 2
    assert maximum_depths[1] <= maximum_depths[0] + 1
    assert maximum_depths[3] <= maximum_depths[2] + 1


def test_unstructured_weno_smooth_trace_error_converges_on_refined_mapped_grids():
    errors = []

    def smooth(points):
        return jnp.exp(0.3 * points[..., 0] + 0.2 * points[..., 1])

    for cells in (6, 12, 24):
        vertices, quadrilaterals = _quadrilateral_grid(cells, cells)
        discretization = phx.discretization.UnstructuredFiniteVolumePlan(
            vertices, quadrilaterals=quadrilaterals
        ).prepare()
        reconstruction = phx.discretization.UnstructuredWENOZReconstructionPlan().prepare(
            discretization
        )
        state = _cell_averages(discretization, smooth)
        left, _ = reconstruction.reconstruct_at(
            state, discretization.face_quadrature_points
        )
        exact = smooth(discretization.face_quadrature_points)
        owner = discretization.owner_cells
        neighbour = discretization.neighbour_cells
        centers = discretization.cell_centers
        margin = 1.5 / cells
        owner_interior = jnp.all(
            (centers[owner] > margin) & (centers[owner] < 1.0 - margin),
            axis=-1,
        )
        safe_neighbour = jnp.maximum(neighbour, 0)
        neighbour_interior = (neighbour >= 0) & jnp.all(
            (centers[safe_neighbour] > margin) & (centers[safe_neighbour] < 1.0 - margin),
            axis=-1,
        )
        interior = owner_interior & neighbour_interior
        weights = discretization.face_quadrature_weights[interior]
        defect = left[interior] - exact[interior]
        errors.append(float(jnp.sqrt(jnp.sum(weights * defect**2) / jnp.sum(weights))))
    rates = [
        np.log(errors[index] / errors[index + 1]) / np.log(2.0)
        for index in range(len(errors) - 1)
    ]
    assert errors[2] < errors[1] < errors[0]
    assert min(rates) > 1.7


def test_unstructured_weno_extrema_limiter_preserves_cell_average_bounds():
    vertices, quadrilaterals = _quadrilateral_grid(6, 4)
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices, quadrilaterals=quadrilaterals
    ).prepare()
    reconstruction = phx.discretization.UnstructuredWENOZReconstructionPlan().prepare(
        discretization
    )
    state = jnp.where(discretization.cell_centers[:, 0] < 0.55, 1.0, 0.0)
    left, right = reconstruction.reconstruct_at(
        state, discretization.face_quadrature_points
    )
    assert jnp.min(left) >= -1e-12
    assert jnp.max(left) <= 1.0 + 1e-12
    assert jnp.min(right) >= -1e-12
    assert jnp.max(right) <= 1.0 + 1e-12
    constant = jnp.ones_like(state)
    constant_left, constant_right = reconstruction.reconstruct_at(
        constant, discretization.face_quadrature_points
    )
    np.testing.assert_allclose(constant_left, 1.0)
    np.testing.assert_allclose(constant_right, 1.0)

    def traces(values):
        return reconstruction.reconstruct_at(
            values, discretization.face_quadrature_points
        )

    reverse_gradient = jax.grad(
        lambda values: sum(jnp.sum(trace) for trace in traces(values))
    )(constant)
    direction = jnp.linspace(-0.5, 0.5, constant.size)
    _, tangent = jax.jvp(traces, (constant,), (direction,))
    assert jnp.all(jnp.isfinite(reverse_gradient))
    assert all(jnp.all(jnp.isfinite(value)) for value in tangent)


def test_unstructured_weno_flux_quadrature_recovers_affine_advection_residual():
    vertices, quadrilaterals = _quadrilateral_grid()
    system = _scalar_system((0.6, -0.25))
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
        component_names=system.component_names,
    ).prepare()
    reconstruction = phx.discretization.UnstructuredWENOZReconstructionPlan(
        2, limiter="none"
    ).prepare(discretization)
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        reconstruction, phx.discretization.RusanovFluxPlan()
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "unstructured-affine-advection",
        "state",
        system,
        boundaries,
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)

    def affine(points):
        return 0.8 + 0.25 * points[..., 0] - 0.15 * points[..., 1]

    state = _cell_averages(discretization, affine)[:, None]
    residual, diagnostics = compiled.residual_with_diagnostics(jnp.asarray(0.0), state)
    flux, _ = compiled.face_fluxes(jnp.asarray(0.0), state)
    balance_terms = np.asarray(discretization.cell_volumes[:, None] * residual)
    integrated = np.asarray(flux * discretization.face_measures[:, None])
    boundary_terms = np.where(
        np.asarray(discretization.neighbour_cells < 0)[:, None],
        integrated,
        0.0,
    )
    expected_defect = math.fsum(
        balance_terms[:, 0].tolist() + boundary_terms[:, 0].tolist()
    )
    assert float(diagnostics.conservation_defect[0]) == expected_defect
    expected = -(0.6 * 0.25 + (-0.25) * (-0.15))
    np.testing.assert_allclose(residual, expected, rtol=2e-9, atol=2e-9)
    np.testing.assert_allclose(
        eqx.filter_jit(compiled)(jnp.asarray(0.0), state),
        expected,
        rtol=2e-9,
        atol=2e-9,
    )
    np.testing.assert_allclose(diagnostics.conservation_defect, 0.0, atol=2e-12)
    assert compiled.stable_step(state) > 0.0


def test_unstructured_weno_uses_shared_positivity_runtime():
    vertices, quadrilaterals = _quadrilateral_grid()
    system = phx.equations.EulerSystem(2)
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
        component_names=system.component_names,
    ).prepare()
    reconstruction = phx.discretization.UnstructuredWENOZReconstructionPlan().prepare(
        discretization
    )
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        reconstruction, phx.discretization.HLLCFluxPlan()
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "unstructured-runtime",
        "state",
        system,
        boundaries,
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem, discretization, method
    ).dynamics
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.1, -0.05, 1.0)),
        discretization.state_shape,
    )
    state = system.primitive_to_conserved(primitive)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics, phx.discretization.FluxPositivityPlan()
    )
    initial = runtime.initialize_state(state, 0.0, 1e-3)
    np.testing.assert_allclose(
        initial.content_state.conservative_content,
        state * discretization.cell_volumes[:, None],
        rtol=2e-11,
        atol=2e-11,
    )
    result = eqx.filter_jit(runtime.advance)(initial)
    assert result.accepted
    assert result.positivity.limited_state_valid
    assert result.accepted_flux_integrals.units == "content"
    assert len(result.accepted_flux_integrals.blocks) == 1
    assert result.accepted_flux_integrals.blocks[0].flux_integral.shape == (
        discretization.face_measures.size,
        system.component_count,
    )
    np.testing.assert_allclose(
        result.runtime_state.cell_average(),
        state,
        rtol=2e-11,
        atol=2e-11,
    )
    np.testing.assert_allclose(
        result.runtime_state.content_state.conservative_content,
        state * discretization.cell_volumes[:, None],
        rtol=2e-11,
        atol=2e-11,
    )


def test_unstructured_coupling_none_is_the_canonical_empty_compilation():
    discretization = _coupling_mesh_plan().prepare()
    compiled_default = _compile_scalar_coupling(discretization)
    coupling_type = phx.discretization.finite_volume.UnstructuredFiniteVolumeCouplingPlan
    compiled_empty = _compile_scalar_coupling(
        discretization,
        coupling_type(),
    )

    assert compiled_default.dynamics.coupling is not compiled_empty.dynamics.coupling
    assert (
        compiled_default.dynamics.coupling.prepared_id
        == compiled_empty.dynamics.coupling.prepared_id
    )
    assert compiled_default.dynamics.dynamics_id == compiled_empty.dynamics.dynamics_id
    assert compiled_default.compilation_id == compiled_empty.compilation_id
    assert compiled_default.dynamics.coupling.topology_event_capacity == 0
    assert compiled_default.dynamics.coupling.topology_event_policy == "disabled"


def test_unstructured_coupling_components_and_event_policies_change_identities():
    base_plan = _coupling_mesh_plan()
    discretization = base_plan.prepare()
    (
        motion,
        embedded_boundary,
        embedded_boundaries,
        vof,
        amr,
        overset,
        sliding,
    ) = _current_coupling_artifacts(base_plan, discretization)
    coupling_type = phx.discretization.finite_volume.UnstructuredFiniteVolumeCouplingPlan
    plans = (
        coupling_type(),
        coupling_type(motion=motion),
        coupling_type(
            embedded_boundary=embedded_boundary,
            embedded_boundaries=embedded_boundaries,
        ),
        coupling_type(vof=vof),
        coupling_type(amr=amr),
        coupling_type(overset=overset),
        coupling_type(
            motion=motion,
            sliding=sliding,
            topology_event_capacity=1,
            topology_event_policy="accepted_step",
        ),
        coupling_type(
            topology_event_capacity=1,
            topology_event_policy="accepted_step",
        ),
        coupling_type(
            topology_event_capacity=2,
            topology_event_policy="accepted_step",
        ),
    )

    plan_ids = [plan.plan_id for plan in plans]
    duplicate_ids = {
        identifier: [index for index, value in enumerate(plan_ids) if value == identifier]
        for identifier in set(plan_ids)
        if plan_ids.count(identifier) > 1
    }
    assert not duplicate_ids, duplicate_ids
    compilable_plans = plans[:3] + (plans[5],) + plans[7:]
    compilations = tuple(
        _compile_scalar_coupling(discretization, plan) for plan in compilable_plans
    )
    assert len({compiled.dynamics.dynamics_id for compiled in compilations}) == len(
        compilable_plans
    )
    assert len({compiled.compilation_id for compiled in compilations}) == len(
        compilable_plans
    )
    with pytest.raises(TypeError, match="TwoMaterialVOFSystem"):
        _compile_scalar_coupling(discretization, plans[3])
    with pytest.raises(ValueError, match="PreparedUnstructuredAMRRuntime"):
        _compile_scalar_coupling(discretization, plans[4])
    with pytest.raises(ValueError, match="Sliding coupling requires"):
        _compile_scalar_coupling(discretization, plans[6])

    with pytest.raises(ValueError, match="stage-bound moved receptor artifacts"):
        _compile_scalar_coupling(
            discretization,
            coupling_type(motion=motion, overset=overset),
        )
    combined = coupling_type(
        overset=overset,
        topology_event_capacity=4,
        topology_event_policy="accepted_step",
    )
    prepared = _compile_scalar_coupling(discretization, combined).dynamics.coupling
    assert prepared.motion is None
    assert prepared.embedded_boundary is None
    assert prepared.vof is None
    assert prepared.amr is None
    assert prepared.overset is overset
    assert prepared.sliding is None
    assert prepared.plan_id == combined.plan_id
    assert prepared.topology_event_capacity == 4
    assert prepared.topology_event_policy == "accepted_step"


def test_stationary_embedded_boundary_compilation_prepares_certified_metrics():
    discretization = _coupling_mesh_plan().prepare()
    coupling, embedded_boundary, embedded_boundaries = _stationary_embedded_coupling(
        discretization
    )

    compiled = _compile_scalar_coupling(discretization, coupling)
    prepared = compiled.dynamics.coupling
    static_compiled = _compile_scalar_coupling(discretization)

    assert prepared.embedded_boundary is embedded_boundary
    assert prepared.embedded_boundaries is embedded_boundaries
    assert (
        prepared.embedded_stabilization_policy is embedded_boundary.stabilization_policy
    )
    assert prepared.embedded_metrics.prepared_id == discretization.prepared_id
    assert prepared.embedded_metrics.topology_id == discretization.topology_id
    assert prepared.embedded_metrics.geometry_id == discretization.geometry_id
    assert prepared.embedded_metrics.field_id == embedded_boundary.field_id
    assert prepared.embedded_metrics.body_tag == embedded_boundary.body_tag
    assert prepared.embedded_metrics.stabilization_policy_id == (
        embedded_boundary.stabilization_policy.policy_id
    )
    assert bool(np.asarray(prepared.embedded_metrics.evidence.passed))
    assert int(np.asarray(prepared.embedded_metrics.evidence.status)) == int(
        phx.discretization.EmbeddedBoundaryStatus.SUCCESS
    )
    assert prepared.cut_boundary_id == embedded_boundaries.boundary_set_id
    assert prepared.prepared_id != static_compiled.dynamics.coupling.prepared_id
    assert compiled.dynamics.dynamics_id != static_compiled.dynamics.dynamics_id
    assert compiled.compilation_id != static_compiled.compilation_id


def test_embedded_boundary_metric_policy_and_body_identities_reach_compilation():
    discretization = _coupling_mesh_plan().prepare()
    base, _, _ = _stationary_embedded_coupling(
        discretization,
        field_id="identity-cut",
    )
    metric_changed, _, _ = _stationary_embedded_coupling(
        discretization,
        level_set=lambda points, args: points[:, 0] + 0.17 * points[:, 1] - 0.43,
        field_id="identity-cut",
    )
    changed_policy = phx.discretization.EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.2,
        maximum_recipients=4,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )
    policy_changed, _, _ = _stationary_embedded_coupling(
        discretization,
        field_id="identity-cut",
        stabilization_policy=changed_policy,
    )
    body_changed, _, _ = _stationary_embedded_coupling(
        discretization,
        field_id="identity-cut",
        body_tag=8,
    )

    assert base.plan_id == metric_changed.plan_id
    compiled = tuple(
        _compile_scalar_coupling(discretization, plan)
        for plan in (base, metric_changed, policy_changed, body_changed)
    )
    prepared = tuple(value.dynamics.coupling for value in compiled)

    assert len({value.embedded_metrics.metrics_id for value in prepared}) == len(prepared)
    assert len({value.prepared_id for value in prepared}) == len(prepared)
    assert len({value.dynamics.dynamics_id for value in compiled}) == len(compiled)
    assert len({value.compilation_id for value in compiled}) == len(compiled)
    assert (
        prepared[0].embedded_stabilization_policy.policy_id
        != prepared[2].embedded_stabilization_policy.policy_id
    )
    assert prepared[0].cut_boundary_id != prepared[3].cut_boundary_id


@pytest.mark.parametrize(
    "component",
    ("vof", "amr", "overset", "sliding", "motion", "topology_events"),
)
def test_embedded_boundary_compilation_rejects_coupled_subsystems_with_ids(
    component,
):
    base_plan = _coupling_mesh_plan()
    discretization = base_plan.prepare()
    (
        motion,
        _,
        _,
        vof,
        amr,
        overset,
        sliding,
    ) = _current_coupling_artifacts(base_plan, discretization)
    _, embedded_boundary, embedded_boundaries = _stationary_embedded_coupling(
        discretization
    )
    options = {
        "embedded_boundary": embedded_boundary,
        "embedded_boundaries": embedded_boundaries,
    }
    conflicting_components = {
        "motion": motion,
        "vof": vof,
        "amr": amr,
        "overset": overset,
    }
    if component == "sliding":
        options.update(
            motion=motion,
            sliding=sliding,
            topology_event_capacity=1,
            topology_event_policy="accepted_step",
        )
    elif component == "topology_events":
        options.update(
            topology_event_capacity=1,
            topology_event_policy="accepted_step",
        )
    else:
        options[component] = conflicting_components[component]
    coupling = UnstructuredFiniteVolumeCouplingPlan(**options)
    conflicting_id = (
        coupling.topology_event_id
        if component == "topology_events"
        else getattr(coupling, component).plan_id
    )

    assert conflicting_id is not None

    with pytest.raises(ValueError) as compile_error:
        _compile_scalar_coupling(discretization, coupling)

    message = str(compile_error.value)
    if component == "amr":
        assert conflicting_id in message
        assert "PreparedUnstructuredAMRRuntime" in message
    elif component == "sliding":
        assert "Sliding coupling requires" in message
    else:
        assert embedded_boundary.plan_id in message
        assert conflicting_id in message
        assert f"{component}=" in message


def test_embedded_boundary_compilation_rejects_high_order_reconstruction():
    discretization = _coupling_mesh_plan().prepare()
    coupling, _, _ = _stationary_embedded_coupling(discretization)
    reconstruction = phx.discretization.CellPolynomialReconstructionPlan(1).prepare(
        discretization
    )

    with pytest.raises(
        ValueError,
        match="embedded-boundary.*PiecewiseConstantReconstruction",
    ):
        _compile_scalar_coupling(
            discretization,
            coupling,
            reconstruction=reconstruction,
        )


def test_embedded_boundary_compilation_requires_complete_supported_body_policies():
    discretization = _coupling_mesh_plan().prepare()
    embedded_boundary = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.43,
        field_id="required-cut-policy",
        body_tag=7,
    )
    cut_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {7: phx.discretization.SlipWallBoundary()}
    )

    with pytest.raises(ValueError, match="requires both"):
        UnstructuredFiniteVolumeCouplingPlan(
            embedded_boundary=embedded_boundary,
        )
    with pytest.raises(ValueError, match="requires both"):
        UnstructuredFiniteVolumeCouplingPlan(
            embedded_boundaries=cut_boundaries,
        )
    with pytest.raises(TypeError, match="SlipWallBoundary"):
        phx.discretization.UnstructuredEmbeddedBoundarySet(
            {7: phx.discretization.ExtrapolationBoundary()}
        )

    wrong_body = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {8: phx.discretization.SlipWallBoundary()}
    )
    with pytest.raises(ValueError, match="no cut-boundary policy.*7"):
        _compile_scalar_coupling(
            discretization,
            UnstructuredFiniteVolumeCouplingPlan(
                embedded_boundary=embedded_boundary,
                embedded_boundaries=wrong_body,
            ),
        )


def test_embedded_boundary_compilation_rejects_extra_cut_boundary_policies():
    discretization = _coupling_mesh_plan().prepare()
    embedded_boundary = phx.discretization.EmbeddedBoundaryPlan(
        discretization,
        lambda points, args: points[:, 0] - 0.43,
        field_id="extra-cut-policy",
        body_tag=7,
    )
    cut_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {
            7: phx.discretization.SlipWallBoundary(),
            8: phx.discretization.SlipWallBoundary(),
        }
    )
    coupling = UnstructuredFiniteVolumeCouplingPlan(
        embedded_boundary=embedded_boundary,
        embedded_boundaries=cut_boundaries,
    )

    with pytest.raises(ValueError) as compile_error:
        _compile_scalar_coupling(discretization, coupling)

    message = str(compile_error.value)
    assert "metric_body_tags={7}" in message
    assert "cut_boundary_policy_body_tags={7, 8}" in message
    assert "extra cut-boundary policy body tag(s): 8" in message


def test_embedded_boundary_compilation_rejects_failed_and_stale_metrics(
    monkeypatch,
):
    vertices, quadrilaterals = _quadrilateral_grid()
    discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices,
        quadrilaterals=quadrilaterals,
    ).prepare()
    zero_tolerance = phx.discretization.EmbeddedBoundaryStabilizationPolicy(
        minimum_volume_fraction=0.05,
        maximum_recipients=3,
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    failed_coupling, _, _ = _stationary_embedded_coupling(
        discretization,
        level_set=lambda points, args: points[:, 0] + 0.31 * points[:, 1] - 0.51,
        field_id="failed-evidence",
        stabilization_policy=zero_tolerance,
    )
    failed_metrics = failed_coupling.embedded_boundary.prepare()
    failed_metrics = eqx.tree_at(
        lambda candidate: candidate.evidence.passed,
        failed_metrics,
        jnp.asarray(False),
    )
    failed_metrics = eqx.tree_at(
        lambda candidate: candidate.evidence.status,
        failed_metrics,
        jnp.asarray(int(phx.discretization.EmbeddedBoundaryStatus.FAILED)),
    )
    monkeypatch.setattr(
        phx.discretization.EmbeddedBoundaryPlan,
        "prepare",
        lambda self, args=None: failed_metrics,
    )
    with pytest.raises(ValueError, match="SUCCESS metric evidence"):
        _compile_scalar_coupling(discretization, failed_coupling)

    current_coupling, _, _ = _stationary_embedded_coupling(
        discretization,
        field_id="stale-metrics",
    )
    stale_discretization = phx.discretization.UnstructuredFiniteVolumePlan(
        vertices + np.asarray((0.125, 0.0)),
        quadrilaterals=quadrilaterals,
    ).prepare()
    stale_metrics = phx.discretization.EmbeddedBoundaryPlan(
        stale_discretization,
        lambda points, args: points[:, 0] - 0.43,
        field_id="stale-metrics",
        body_tag=7,
    ).prepare()
    monkeypatch.setattr(
        phx.discretization.EmbeddedBoundaryPlan,
        "prepare",
        lambda self, args=None: stale_metrics,
    )
    with pytest.raises(ValueError, match="(metrics belong to stale|field/body identity)"):
        _compile_scalar_coupling(discretization, current_coupling)


def test_embedded_boundary_compilation_rejects_three_dimensions_and_viscous_physics():
    tetrahedral = phx.discretization.UnstructuredFiniteVolumePlan(
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        tetrahedra=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    ).prepare()
    with pytest.raises(ValueError, match="2-D polygons"):
        phx.discretization.EmbeddedBoundaryPlan(
            tetrahedral,
            lambda points, args: points[:, 0] - 0.5,
            field_id="three-dimensional-cut",
        )

    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2),
        2,
    )
    discretization = _coupling_mesh_plan(component_names=system.component_names).prepare()
    coupling, _, _ = _stationary_embedded_coupling(discretization)
    method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: phx.discretization.ExtrapolationBoundary()
            for name in discretization.boundary_patch_names
        },
    )
    problem = phx.equations.ConservationProblemIR(
        "viscous-embedded-rejection",
        "state",
        system,
        boundaries,
    )
    with pytest.raises(ValueError, match="Viscous embedded-boundary"):
        phx.equations.compile_conservation_problem(
            problem,
            discretization,
            method,
            coupling=coupling,
        )


def test_unstructured_coupling_rejects_invalid_event_combinations():
    coupling_type = phx.discretization.finite_volume.UnstructuredFiniteVolumeCouplingPlan
    sliding = phx.discretization.PeriodicSlidingInterfacePlan(
        np.asarray((0.0, 1.0)),
        np.asarray((0.0, 0.4, 1.0)),
        1.0,
        interface_id="missing-motion",
    )
    with pytest.raises(ValueError, match="zero event capacity"):
        coupling_type(topology_event_capacity=1)
    with pytest.raises(ValueError, match="positive event capacity"):
        coupling_type(topology_event_policy="accepted_step")
    with pytest.raises(ValueError, match="motion identity"):
        coupling_type(
            sliding=sliding,
            topology_event_capacity=1,
            topology_event_policy="accepted_step",
        )


def test_unstructured_coupling_rejects_stale_and_mismatched_artifacts():
    base_plan = _coupling_mesh_plan()
    discretization = base_plan.prepare()
    coupling_type = phx.discretization.finite_volume.UnstructuredFiniteVolumeCouplingPlan

    different_topology_plan = _coupling_mesh_plan(nx=3, ny=4)
    different_topology_motion = phx.discretization.FixedConnectivityMotionPlan(
        different_topology_plan,
        lambda time, vertices, args: vertices,
        mapping_id="different-topology",
    )
    with pytest.raises(ValueError, match="different unstructured topology"):
        _compile_scalar_coupling(
            discretization,
            coupling_type(motion=different_topology_motion),
        )

    stale_plan = _coupling_mesh_plan(x_offset=0.125)
    stale = stale_plan.prepare()
    stale_embedded = phx.discretization.EmbeddedBoundaryPlan(
        stale,
        lambda points, args: jnp.ones((points.shape[0],)),
        field_id="stale-embedded-geometry",
        body_tag=7,
    )
    stale_cut_boundaries = phx.discretization.UnstructuredEmbeddedBoundarySet(
        {7: phx.discretization.SlipWallBoundary()}
    )
    with pytest.raises(ValueError, match="Embedded-boundary plan.*stale"):
        _compile_scalar_coupling(
            discretization,
            coupling_type(
                embedded_boundary=stale_embedded,
                embedded_boundaries=stale_cut_boundaries,
            ),
        )

    stale_amr = _coupling_hierarchy(
        stale,
        x_offset=0.125,
    )
    with pytest.raises(ValueError, match="PreparedUnstructuredAMRRuntime"):
        _compile_scalar_coupling(
            discretization,
            coupling_type(amr=stale_amr),
        )

    cell_count = stale.cell_count
    stale_overset = phx.discretization.UnstructuredOversetPlan(
        stale,
        stale,
        np.arange(cell_count, dtype=np.int32),
        np.arange(cell_count + 1, dtype=np.int32),
        np.arange(cell_count, dtype=np.int32),
        stale.cell_volumes,
    )
    with pytest.raises(ValueError, match="Overset receptor.*stale"):
        _compile_scalar_coupling(
            discretization,
            coupling_type(overset=stale_overset),
        )

    stale_donor_overset = phx.discretization.UnstructuredOversetPlan(
        stale,
        discretization,
        np.arange(cell_count, dtype=np.int32),
        np.arange(cell_count + 1, dtype=np.int32),
        np.arange(cell_count, dtype=np.int32),
        discretization.cell_volumes,
    )
    with pytest.raises(ValueError, match="Overset donor.*stale"):
        _compile_scalar_coupling(
            discretization,
            coupling_type(overset=stale_donor_overset),
        )


def test_unstructured_coupling_preserves_hllc_system_validation():
    discretization = _coupling_mesh_plan().prepare()
    with pytest.raises(ValueError, match="Euler-compatible"):
        _compile_scalar_coupling(
            discretization,
            interface_solver=phx.discretization.HLLCFluxPlan(),
        )
