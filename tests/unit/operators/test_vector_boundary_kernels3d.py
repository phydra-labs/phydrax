import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.operators.integral.layer_potential._elasticity3d import (
    ElasticityLayerKernel3D,
    ElasticitySingleLayerDP0Policy3D,
    evaluate_elasticity_layer_3d,
    prepare_elasticity_single_layer_dp0_3d,
)
from phydrax.operators.integral.layer_potential._stokes3d import (
    evaluate_stokes_layer_3d,
    evaluate_stokes_pressure_3d,
    prepare_stokes_single_layer_dp0_3d,
    StokesLayerKernel3D,
    StokesLayerPotential3D,
    StokesSingleLayerDP0Policy3D,
)
from phydrax.solver._elasticity_boundary import (
    solve_elasticity_interior_displacement_dirichlet_3d,
)
from phydrax.solver._stokes_boundary import (
    solve_stokes_interior_velocity_dirichlet_3d,
)


_VERTICES = jnp.asarray(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)
_FACES = jnp.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=jnp.int32)


def _region():
    return phx.geometry.MeshRegion(_VERTICES, _FACES)


@pytest.fixture(scope="module")
def elasticity_prepared():
    return prepare_elasticity_single_layer_dp0_3d(
        _region(),
        shear_modulus=2.0,
        poisson_ratio=0.25,
        policy=ElasticitySingleLayerDP0Policy3D(
            regular_order=3,
            singular_order=3,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
            max_face_count=8,
            max_matrix_bytes=4096,
            max_preparation_workspace_bytes=1024 * 1024,
        ),
    )


@pytest.fixture(scope="module")
def stokes_prepared():
    return prepare_stokes_single_layer_dp0_3d(
        _region(),
        viscosity=2.0,
        policy=StokesSingleLayerDP0Policy3D(
            regular_order=3,
            singular_order=3,
            absolute_tolerance=1.0,
            relative_tolerance=1.0,
            max_face_count=8,
            max_matrix_bytes=4096,
            max_preparation_workspace_bytes=1024 * 1024,
        ),
    )


def test_kelvin_kernel_matches_point_source_and_reciprocity():
    mu = 2.0
    nu = 0.25
    kernel = ElasticityLayerKernel3D(mu, nu)
    source = jnp.asarray([0.0, 0.0, 0.0])
    target = jnp.asarray([2.0, 0.0, 0.0])
    radius = 2.0
    prefactor = 1.0 / (16.0 * jnp.pi * mu * (1.0 - nu) * radius)
    expected = prefactor * jnp.diag(
        jnp.asarray([4.0 - 4.0 * nu, 3.0 - 4.0 * nu, 3.0 - 4.0 * nu])
    )

    assert jnp.allclose(kernel.value(target, source), expected, rtol=2.0e-6, atol=1.0e-7)
    force = jnp.asarray([0.3, -0.4, 0.2])
    assert jnp.allclose(kernel.apply_point_force(target, source, force), expected @ force)

    x = jnp.asarray([0.4, -0.2, 1.3])
    y = jnp.asarray([-0.7, 0.5, 0.1])
    assert jnp.allclose(kernel.value(x, y), kernel.value(y, x).T, rtol=1.0e-6)
    traction = kernel.source_traction(x, y, jnp.asarray([0.0, 0.0, 1.0]))
    assert traction.shape == (3, 3)
    assert jnp.all(jnp.isfinite(traction))
    assert kernel.contract.ambient_dimension == 3
    assert "dynamic" in " ".join(kernel.contract.non_goals)


def test_stokeslet_matches_point_source_pressure_and_reciprocity():
    viscosity = 2.0
    kernel = StokesLayerKernel3D(viscosity)
    source = jnp.asarray([0.0, 0.0, 0.0])
    target = jnp.asarray([2.0, 0.0, 0.0])
    expected = jnp.diag(jnp.asarray([1.0, 0.5, 0.5])) / (8.0 * jnp.pi * viscosity)
    expected_pressure = jnp.asarray([1.0, 0.0, 0.0]) / (16.0 * jnp.pi)

    assert jnp.allclose(kernel.value(target, source), expected, rtol=2.0e-6, atol=1.0e-7)
    assert jnp.allclose(
        kernel.pressure_vector(target, source), expected_pressure, rtol=2.0e-6
    )
    force = jnp.asarray([0.3, -0.4, 0.2])
    velocity, pressure = kernel.apply_point_force(target, source, force)
    assert jnp.allclose(velocity, expected @ force)
    assert jnp.allclose(pressure, expected_pressure @ force)

    x = jnp.asarray([0.4, -0.2, 1.3])
    y = jnp.asarray([-0.7, 0.5, 0.1])
    assert jnp.allclose(kernel.value(x, y), kernel.value(y, x).T, rtol=1.0e-6)
    stresslet = kernel.source_traction(x, y, jnp.asarray([0.0, 1.0, 0.0]))
    assert stresslet.shape == (3, 3)
    assert jnp.all(jnp.isfinite(stresslet))
    assert kernel.contract.ambient_dimension == 3
    assert "Navier-Stokes" in " ".join(kernel.contract.non_goals)


def test_elasticity_prepared_reciprocity_transpose_modes_and_field(elasticity_prepared):
    prepared = elasticity_prepared
    weak = prepared.weak_operator.matrix
    assert prepared.assembly_report.pair_counts == (4, 12, 0, 0)
    assert sum(prepared.assembly_report.pair_counts) == prepared.face_count**2
    assert jnp.allclose(weak, weak.T, rtol=2.0e-6, atol=2.0e-7)
    assert bool(prepared.assembly_report.accuracy_supported)
    assert not prepared.assembly_report.continuum_discretization_error_estimated

    x = jnp.linspace(-0.4, 0.7, 3 * prepared.face_count)
    y = jnp.linspace(0.8, -0.2, 3 * prepared.face_count)
    action = prepared.strong_operator.mv(x)
    transpose = prepared.strong_operator.transpose_mv(y)
    assert jnp.allclose(y @ action, x @ transpose, rtol=2.0e-6, atol=2.0e-7)

    metadata = prepared.nullspace
    assert metadata.rigid_displacement_modes.shape == (3 * prepared.face_count, 6)
    assert metadata.force_torque_functionals.shape == (6, 3 * prepared.face_count)
    assert metadata.rigid_mode_dimension == 6
    assert metadata.pressure_nullspace_dimension == 0
    translations = metadata.rigid_displacement_modes[:, :3].reshape(
        (prepared.face_count, 3, 3)
    )
    assert jnp.allclose(translations, jnp.broadcast_to(jnp.eye(3), translations.shape))
    constant_traction = jnp.tile(jnp.asarray([0.3, -0.2, 0.4]), prepared.face_count)
    force_torque = metadata.force_torque_functionals @ constant_traction
    assert jnp.allclose(
        force_torque[:3],
        jnp.sum(prepared.face_areas) * jnp.asarray([0.3, -0.2, 0.4]),
        atol=2.0e-6,
    )
    assert jnp.allclose(force_torque[3:], jnp.zeros((3,)), atol=2.0e-6)

    potential = prepared.potential(jnp.ones((prepared.face_count, 3)))
    value, target_report = evaluate_elasticity_layer_3d(
        potential, jnp.asarray([2.0, 2.0, 2.0]), target_side="exterior"
    )
    assert bool(target_report.pde_membership_valid)
    assert value.shape == (3,)
    assert jnp.all(jnp.isfinite(value))


def test_stokes_prepared_reciprocity_transpose_constraints_and_field(stokes_prepared):
    prepared = stokes_prepared
    weak = prepared.weak_operator.matrix
    assert prepared.assembly_report.pair_counts == (4, 12, 0, 0)
    assert sum(prepared.assembly_report.pair_counts) == prepared.face_count**2
    assert jnp.allclose(weak, weak.T, rtol=2.0e-6, atol=2.0e-7)
    assert bool(prepared.assembly_report.accuracy_supported)
    assert not prepared.assembly_report.continuum_discretization_error_estimated

    x = jnp.linspace(-0.4, 0.7, 3 * prepared.face_count)
    y = jnp.linspace(0.8, -0.2, 3 * prepared.face_count)
    action = prepared.strong_operator.mv(x)
    transpose = prepared.strong_operator.transpose_mv(y)
    assert jnp.allclose(y @ action, x @ transpose, rtol=2.0e-6, atol=2.0e-7)

    metadata = prepared.nullspace
    assert metadata.rigid_velocity_modes.shape == (3 * prepared.face_count, 6)
    assert metadata.force_torque_functionals.shape == (6, 3 * prepared.face_count)
    assert metadata.single_layer_density_null_vector.shape == (3 * prepared.face_count,)
    assert metadata.pressure_nullspace_dimension == 1
    assert jnp.allclose(
        weak @ metadata.single_layer_density_null_vector,
        jnp.zeros((3 * prepared.face_count,)),
        atol=2.0e-6,
    )
    assert jnp.allclose(
        metadata.boundary_flux_functional @ metadata.rigid_velocity_modes,
        jnp.zeros((6,)),
        atol=2.0e-6,
    )
    constant_force = jnp.tile(jnp.asarray([-0.2, 0.5, 0.1]), prepared.face_count)
    force_torque = metadata.force_torque_functionals @ constant_force
    assert jnp.allclose(
        force_torque[:3],
        jnp.sum(prepared.face_areas) * jnp.asarray([-0.2, 0.5, 0.1]),
        atol=2.0e-6,
    )
    assert jnp.allclose(force_torque[3:], jnp.zeros((3,)), atol=2.0e-6)

    potential = prepared.potential(jnp.ones((prepared.face_count, 3)))
    velocity, target_report = evaluate_stokes_layer_3d(
        potential, jnp.asarray([2.0, 2.0, 2.0]), target_side="exterior"
    )
    pressure, pressure_report = evaluate_stokes_pressure_3d(
        potential, jnp.asarray([2.0, 2.0, 2.0]), target_side="exterior"
    )
    assert bool(target_report.pde_membership_valid)
    assert pressure_report.report_id == target_report.report_id
    assert velocity.shape == (3,)
    assert jnp.all(jnp.isfinite(velocity))
    assert jnp.isfinite(pressure)


def test_named_elasticity_and_stokes_dirichlet_solves(
    elasticity_prepared, stokes_prepared
):
    elastic_trace = jnp.broadcast_to(
        jnp.asarray([0.2, -0.1, 0.3]), (elasticity_prepared.face_count, 3)
    )
    elastic = solve_elasticity_interior_displacement_dirichlet_3d(
        elasticity_prepared, elastic_trace
    )
    assert bool(elastic.valid)
    assert elastic.traction_density.shape == elastic_trace.shape
    assert jnp.isfinite(elastic.boundary_residual_norm)
    elastic_value, _ = evaluate_elasticity_layer_3d(
        elastic.potential, jnp.asarray([2.0, 2.0, 2.0]), target_side="exterior"
    )
    assert jnp.all(jnp.isfinite(elastic_value))

    stokes_trace = jnp.broadcast_to(
        jnp.asarray([0.1, -0.2, 0.05]), (stokes_prepared.face_count, 3)
    )
    stokes = solve_stokes_interior_velocity_dirichlet_3d(
        stokes_prepared, stokes_trace, flux_tolerance=1.0e-6
    )
    assert bool(stokes.valid)
    assert stokes.force_density.shape == stokes_trace.shape
    assert jnp.abs(stokes.boundary_flux) < 1.0e-6
    assert jnp.abs(stokes.density_gauge_residual) < 1.0e-5
    stokes_value, _ = evaluate_stokes_layer_3d(
        stokes.potential, jnp.asarray([2.0, 2.0, 2.0]), target_side="exterior"
    )
    assert jnp.all(jnp.isfinite(stokes_value))


def test_pressure_unsupported_and_displacement_discontinuity_contract(stokes_prepared):
    double_layer = StokesLayerPotential3D(
        stokes_prepared.panelization,
        viscosity=2.0,
        kind="double",
    )
    with pytest.raises(phx.linalg.LinearCapabilityError, match="double layer"):
        double_layer.pressure(jnp.asarray([2.0, 2.0, 2.0]))

    prepared = phx.operators.prepare_displacement_discontinuity_3d(
        _VERTICES[:3],
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        shear_modulus=2.0,
        poisson_ratio=0.25,
    )
    assert prepared.space.vertex_count == 3
    assert prepared.space.face_count == 1
    assert prepared.space.crack_front_edges.shape == (3, 2)
    assert prepared.evidence.conforming_p1
    assert not prepared.evidence.dp0_hypersingular_supported
    assert prepared.evidence.minimum_face_area > 0.0
    assert prepared.evidence.maximum_symmetry_defect == 0.0

    rigid_jump = jnp.broadcast_to(
        jnp.asarray([0.3, -0.2, 0.4]),
        (prepared.space.vertex_count, 3),
    )
    assert jnp.allclose(prepared.traction(rigid_jump), 0.0, atol=1.0e-12)
    jump = jnp.asarray([[0.1, 0.2, -0.3], [-0.4, 0.2, 0.5], [0.3, -0.1, 0.2]])
    traction = prepared.traction(jump)
    assert traction.shape == jump.shape
    assert jnp.all(jnp.isfinite(traction))
    assert jnp.vdot(jump, traction) >= -1.0e-12

    with pytest.raises(ValueError, match="open sheet"):
        phx.operators.prepare_displacement_discontinuity_3d(
            _VERTICES,
            _FACES,
            shear_modulus=2.0,
            poisson_ratio=0.25,
        )
