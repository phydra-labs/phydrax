import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from phydrax.geometry import MeshRegion
from phydrax.operators.integral.layer_potential._free_surface_green3d import (
    FreeSurfaceGreenPolicy3D,
    prepare_free_surface_green_3d,
    solve_finite_depth_dispersion_3d,
)
from phydrax.operators.integral.layer_potential._free_surface_hydrodynamics3d import (
    FreeSurfaceHydrodynamicsPolicy3D,
    prepare_free_surface_hydrodynamics_3d,
    prepare_hydrostatic_properties_3d,
)
from phydrax.operators.integral.layer_potential._galerkin3d import (
    LaplaceSingleLayerDP0GalerkinPolicy3D,
)
from phydrax.solver._potential_flow_hydrodynamics import (
    solve_potential_flow_hydrodynamics_3d,
)


def _region(mesh, *, feature_id):
    return MeshRegion(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces, dtype=np.int32),
        feature_id=feature_id,
    )


def _submerged_sphere():
    mesh = trimesh.creation.icosphere(subdivisions=0, radius=0.5)
    mesh.apply_translation((0.0, 0.0, -2.0))
    return _region(mesh, feature_id="submerged-octahedral-sphere")


def _fast_policy():
    return FreeSurfaceHydrodynamicsPolicy3D(
        green=FreeSurfaceGreenPolicy3D(
            radial_order_per_interval=8,
            angular_order=8,
            cutoff_clearance_factor=3.0,
            minimum_cutoff_root_ratio=2.0,
            maximum_wavenumber=100.0,
        ),
        galerkin=LaplaceSingleLayerDP0GalerkinPolicy3D(
            regular_order=3,
            singular_order=3,
            near_order=3,
            near_ratio=1.5,
            near_max_depth=1,
            absolute_tolerance=2.0e-2,
            relative_tolerance=1.0e-1,
            target_block_size=8,
            source_block_size=8,
        ),
        max_faces=32,
        max_dense_entries=2 * 32 * 32,
        max_resident_bytes=32 * 1024 * 1024,
    )


@pytest.fixture(scope="module")
def submerged_problem():
    prepared = prepare_free_surface_hydrodynamics_3d(
        _submerged_sphere(),
        2.0,
        gravity=9.81,
        frame_id="tank-z-up",
        unit_system_id="si-water",
        policy=_fast_policy(),
    )
    result = solve_potential_flow_hydrodynamics_3d(
        prepared,
        fluid_density=1000.0,
        incident_headings=(0.0, 0.5 * np.pi),
        reciprocity_tolerance=0.35,
        radiated_power_tolerance=1.0e-6,
    )
    return prepared, result


def test_finite_depth_dispersion_root_has_residual_and_provenance():
    root = solve_finite_depth_dispersion_3d(
        1.7,
        9.81,
        3.5,
        tolerance=1.0e-13,
        frame_id="tank-z-up",
        unit_system_id="si-water",
    )
    residual = root.wavenumber * jnp.tanh(root.wavenumber * 3.5) - 1.7**2 / 9.81

    assert bool(root.converged)
    assert abs(float(residual)) <= 3.0e-13
    assert float(root.residual) <= 3.0e-13
    assert root.frame_id == "tank-z-up"
    assert root.unit_system_id == "si-water"
    assert root.ambient_dimension == 3
    assert root.non_goals == ("capillary dispersion", "current-modified dispersion")


def test_finite_depth_green_is_reciprocal_and_retains_tail_evidence():
    green = prepare_free_surface_green_3d(
        1.7,
        9.81,
        minimum_clearance=1.0,
        depth=5.0,
        frame_id="tank-z-up",
        unit_system_id="si-water",
        policy=FreeSurfaceGreenPolicy3D(
            radial_order_per_interval=8,
            angular_order=8,
            cutoff_clearance_factor=3.0,
            maximum_wavenumber=100.0,
        ),
    )
    left = jnp.asarray((0.0, 0.0, -1.5))
    right = jnp.asarray((0.7, -0.2, -2.0))
    forward = green.value(left, right)
    reverse = green.value(right, left)

    assert jnp.all(jnp.isfinite(jnp.asarray((forward, reverse))))
    assert jnp.allclose(forward, reverse, rtol=1.0e-12, atol=1.0e-12)
    assert float(green.dispersion.residual) <= 3.0e-13
    assert float(green.errors.spectral_tail_envelope_bound) > 0.0
    assert (
        float(green.errors.spectral_tail_envelope_bound)
        <= green.errors.maximum_spectral_tail_bound
    )
    assert green.errors.wavenumber_cutoff > green.wavenumber
    assert not green.errors.continuum_discretization_error_estimated
    assert not green.errors.quadrature_convergence_estimated


def test_submerged_sphere_symmetry_reciprocity_and_radiated_power(submerged_problem):
    prepared, result = submerged_problem
    diagonal_a = jnp.diag(result.added_mass)
    diagonal_b = jnp.diag(result.radiation_damping)

    assert prepared.mode_names[:3] == (
        "body-0:surge",
        "body-0:sway",
        "body-0:heave",
    )
    assert jnp.allclose(diagonal_a[0], diagonal_a[1], rtol=8.0e-2, atol=1.0e-8)
    assert jnp.allclose(diagonal_b[0], diagonal_b[1], rtol=8.0e-2, atol=1.0e-8)
    assert jnp.array_equal(result.added_mass, result.added_mass.T)
    assert jnp.array_equal(result.radiation_damping, result.radiation_damping.T)
    assert float(result.added_mass_reciprocity_defect) <= 0.35
    assert float(result.damping_reciprocity_defect) <= 0.35
    assert bool(result.radiated_power_nonnegative)
    assert float(result.radiated_power(jnp.ones((6,), dtype=jnp.complex128))) >= -1.0e-6
    assert result.excitation_loads.shape == (6, 2)
    assert jnp.all(jnp.isfinite(result.excitation_loads))


def test_operators_have_exact_forward_transpose_and_density_semantics(submerged_problem):
    prepared, result = submerged_problem
    x = jnp.linspace(0.1, 0.8, prepared.face_count).astype(jnp.complex128)
    y = (0.4 - 0.2j) * jnp.linspace(1.0, 2.0, prepared.face_count)

    for operator in (prepared.boundary_operator, prepared.trace_operator):
        assert jnp.allclose(operator.mv(x), operator.matrix @ x)
        assert jnp.allclose(operator.transpose_mv(y), operator.matrix.T @ y)
        assert jnp.allclose(y @ operator.mv(x), x @ operator.transpose_mv(y))

    assert jnp.allclose(prepared.potential_trace(x), prepared.trace_operator.matrix @ x)
    assert "unweighted" in prepared.density_semantics
    assert result.density_semantics == prepared.density_semantics
    assert result.radiation_density.shape == (prepared.face_count, 6)
    assert prepared.frame_id == result.frame_id == "tank-z-up"
    assert prepared.unit_system_id == result.unit_system_id == "si-water"
    assert prepared.coordinate_convention == "right-handed-cartesian-z-up"
    assert result.coordinate_convention == prepared.coordinate_convention
    assert prepared.time_convention == "exp(-i*omega*t)"
    assert prepared.normal_convention == "body-to-fluid"


def test_resource_and_error_evidence_is_explicit(submerged_problem):
    prepared, result = submerged_problem
    report = prepared.assembly_report

    assert report.boundary_operator_bytes == prepared.boundary_operator.matrix.nbytes
    assert report.trace_operator_bytes == prepared.trace_operator.matrix.nbytes
    assert 0 < report.resident_bytes <= report.maximum_resident_bytes
    assert (
        0
        < report.preparation_workspace_bytes
        <= report.maximum_preparation_workspace_bytes
    )
    assert prepared.green.resources.within_policy
    assert prepared.green.errors.spectral_tail_envelope_bound > 0.0
    assert not report.continuum_discretization_error_estimated
    assert not report.collocation_error_estimated
    assert result.resource_evidence[0] == report.resident_bytes
    assert "no continuum collocation" in result.error_evidence[-1]


def test_transversal_cube_hydrostatics_and_waterline_invalidity():
    cube = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    hydrostatics = prepare_hydrostatic_properties_3d(
        _region(cube, feature_id="surface-piercing-cube"),
        fluid_density=1000.0,
        gravity=10.0,
        frame_id="tank-z-up",
        unit_system_id="si-water",
    )

    assert jnp.allclose(hydrostatics.displaced_volume, 4.0, rtol=1.0e-12)
    assert jnp.allclose(hydrostatics.center_of_buoyancy, jnp.asarray((0.0, 0.0, -0.5)))
    assert jnp.allclose(hydrostatics.waterplane_area, 4.0, rtol=1.0e-12)
    assert jnp.allclose(hydrostatics.waterplane_centroid, jnp.zeros((3,)))
    assert jnp.allclose(hydrostatics.restoring_matrix[2, 2], 40_000.0)
    assert hydrostatics.waterline_loop_count == 1
    assert hydrostatics.frame_id == "tank-z-up"
    assert hydrostatics.unit_system_id == "si-water"

    touching = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    touching.apply_translation((0.0, 0.0, 0.5))
    with pytest.raises(ValueError, match="Waterline vertices"):
        prepare_hydrostatic_properties_3d(
            _region(touching, feature_id="invalid-waterline-cube")
        )
    with pytest.raises(ValueError, match="strictly submerged"):
        prepare_free_surface_hydrodynamics_3d(
            _region(cube, feature_id="invalid-radiation-waterline-cube"),
            2.0,
            policy=_fast_policy(),
        )
