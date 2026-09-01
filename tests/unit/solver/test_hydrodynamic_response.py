import jax.numpy as jnp
import pytest

from phydrax.linalg import LinearSolveStatus
from phydrax.solver._hydrodynamic_response import (
    HydrodynamicResponseStatus,
    solve_hydrodynamic_response_3d,
    WetSurfaceModalGeneralizedForceMap3D,
)
from phydrax.solver._potential_flow_hydrodynamics import (
    PotentialFlowHydrodynamicsResult3D,
)


def _hydrodynamics(
    added_mass,
    radiation_damping,
    *,
    frame_id="tank-z-up",
    unit_system_id="si-water",
    geometry_id="prepared-wet-body",
):
    added = jnp.asarray(added_mass, dtype=jnp.float64)
    damping = jnp.asarray(radiation_damping, dtype=jnp.float64)
    size = int(added.shape[0])
    names = tuple(f"body-0:mode-{index}" for index in range(size))
    face_columns = jnp.zeros((1, size), dtype=jnp.complex128)
    empty_faces = jnp.zeros((1, 0), dtype=jnp.complex128)
    return PotentialFlowHydrodynamicsResult3D(
        radiation_density=face_columns,
        radiation_potential_trace=face_columns,
        radiation_integrals=jnp.zeros((size, size), dtype=jnp.complex128),
        added_mass=added,
        radiation_damping=damping,
        diffraction_density=empty_faces,
        diffraction_potential_trace=empty_faces,
        excitation_loads=jnp.zeros((size, 0), dtype=jnp.complex128),
        radiation_linear_results=(),
        diffraction_linear_results=(),
        assembly_report=None,
        fluid_density=jnp.asarray(1025.0),
        added_mass_reciprocity_defect=jnp.asarray(0.0),
        damping_reciprocity_defect=jnp.asarray(0.0),
        minimum_radiated_power_eigenvalue=0.5 * jnp.min(jnp.linalg.eigvalsh(damping)),
        radiated_power_nonnegative=jnp.asarray(True),
        valid=jnp.asarray(True),
        incident_headings=(),
        mode_names=names,
        ambient_dimension=3,
        coordinate_convention="right-handed-cartesian-z-up",
        pde_id="zero-speed-linear-free-surface-potential-flow",
        geometry_id=geometry_id,
        formulation_id="unit-generalized-velocity-radiation-and-fixed-body-diffraction",
        provider_id="synthetic-already-prepared-hydrodynamic-coefficients",
        precision_id="float64-complex128",
        frame_id=frame_id,
        unit_system_id=unit_system_id,
        time_convention="exp(-i*omega*t)",
        normal_convention="body-to-fluid",
        density_semantics="unweighted DP0 source strength",
        resource_evidence=(0, 0, 0),
        error_evidence=("synthetic exact coefficient fixture",),
        non_goals=("continuum certification",),
    )


def _solve_kwargs(size):
    return {
        "angular_frequency": 2.0,
        "coefficient_frequency_id": "omega=2-rad-per-second",
        "excitation_id": "unit-wave-loads",
        "physical_mass_inertia": jnp.eye(size),
        "external_damping": jnp.zeros((size, size)),
        "hydrostatic_restoring": 20.0 * jnp.eye(size),
        "mooring_restoring": jnp.zeros((size, size)),
        "incident_excitation": jnp.ones((size, 1), dtype=jnp.complex128),
        "frame_id": "tank-z-up",
        "unit_system_id": "si-water",
        "reference_point_id": "body-reference-set-A",
        "hydrodynamic_reference_point_id": "body-reference-set-A",
    }


def test_rigid_rao_matrix_equation_exact_residual_and_operator_duals():
    hydrodynamics = _hydrodynamics(
        jnp.diag(jnp.asarray((1.0, 0.5))),
        jnp.diag(jnp.asarray((0.6, 0.3))),
    )
    excitation = jnp.asarray(
        ((1.0 + 0.5j, -0.2j), (0.4 - 0.1j, 1.5)),
        dtype=jnp.complex128,
    )
    result = solve_hydrodynamic_response_3d(
        hydrodynamics,
        angular_frequency=2.0,
        coefficient_frequency_id="omega=2-rad-per-second",
        excitation_id="two-unit-wave-headings",
        physical_mass_inertia=jnp.diag(jnp.asarray((3.0, 4.0))),
        external_damping=jnp.diag(jnp.asarray((0.4, 0.2))),
        hydrostatic_restoring=jnp.diag(jnp.asarray((30.0, 40.0))),
        mooring_restoring=jnp.diag(jnp.asarray((2.0, 4.0))),
        incident_excitation=excitation,
        frame_id="tank-z-up",
        unit_system_id="si-water",
        reference_point_id="body-reference-set-A",
        hydrodynamic_reference_point_id="body-reference-set-A",
    )

    expected_dynamic = (
        jnp.diag(jnp.asarray((32.0, 44.0)))
        - 2.0**2 * jnp.diag(jnp.asarray((4.0, 4.5)))
        - 2.0j * jnp.diag(jnp.asarray((1.0, 0.5)))
    )
    exact_residual = (
        result.dynamic_operator.matrix @ result.displacement_response
        - result.generalized_excitation
    )
    probe = jnp.asarray((0.3 + 0.1j, -0.4j))

    assert bool(result.successful)
    assert result.status == int(HydrodynamicResponseStatus.SUCCESS)
    assert result.rigid_body_rao.shape == (2, 2)
    assert result.modal_response.shape == (0, 2)
    assert jnp.allclose(result.dynamic_operator.matrix, expected_dynamic)
    assert jnp.allclose(expected_dynamic @ result.rigid_body_rao, excitation)
    assert jnp.array_equal(result.residual, exact_residual)
    assert jnp.all(result.residual_norm <= result.residual_threshold)
    assert jnp.allclose(result.apply_dynamic(probe), expected_dynamic @ probe)
    assert jnp.allclose(result.apply_dynamic_transpose(probe), expected_dynamic.T @ probe)
    assert jnp.allclose(
        result.apply_dynamic_adjoint(probe), jnp.conj(expected_dynamic.T) @ probe
    )
    assert result.time_convention == "exp(-i*omega*t)"
    assert "-i*omega*(C_dry+B)" in result.formulation_id
    assert result.solver_provider_id.startswith("phydrax.linalg.DenseLU")


def test_checked_wet_force_map_builds_bounded_modal_hydroelastic_response():
    hydrodynamics = _hydrodynamics(
        jnp.diag(jnp.asarray((0.8, 0.4))),
        jnp.diag(jnp.asarray((0.5, 0.2))),
    )
    force_map = WetSurfaceModalGeneralizedForceMap3D(
        jnp.asarray(((0.5, -0.25j),)),
        source_mode_names=hydrodynamics.mode_names,
        modal_names=("dry-mode-0",),
        geometry_id=hydrodynamics.geometry_id,
        mapping_id="checked-wet-to-mode-map",
        frame_id="tank-z-up",
        unit_system_id="si-water",
        reference_point_id="body-reference-set-A",
    )
    excitation = jnp.asarray(((1.0 + 0.2j,), (0.3 - 0.1j,)))
    result = solve_hydrodynamic_response_3d(
        hydrodynamics,
        angular_frequency=1.5,
        coefficient_frequency_id="omega=1.5-rad-per-second",
        excitation_id="unit-wave-modal-case",
        physical_mass_inertia=jnp.diag(jnp.asarray((2.0, 3.0))),
        external_damping=jnp.diag(jnp.asarray((0.2, 0.3))),
        hydrostatic_restoring=jnp.diag(jnp.asarray((12.0, 15.0))),
        mooring_restoring=jnp.diag(jnp.asarray((1.0, 2.0))),
        incident_excitation=excitation,
        frame_id="tank-z-up",
        unit_system_id="si-water",
        reference_point_id="body-reference-set-A",
        hydrodynamic_reference_point_id="body-reference-set-A",
        structural_modal_mass=jnp.asarray(((1.2,),)),
        structural_modal_stiffness=jnp.asarray(((18.0,),)),
        structural_modal_damping=jnp.asarray(((0.15,),)),
        modal_force_map=force_map,
    )
    modal_probe = jnp.asarray((0.7 - 0.4j,))

    assert force_map.rank == 1
    assert jnp.allclose(
        force_map.mv(excitation[:, 0]), force_map.matrix @ excitation[:, 0]
    )
    assert jnp.allclose(
        force_map.transpose_mv(modal_probe), force_map.matrix.T @ modal_probe
    )
    assert jnp.allclose(
        force_map.adjoint_mv(modal_probe), jnp.conj(force_map.matrix.T) @ modal_probe
    )
    assert result.displacement_response.shape == (3, 1)
    assert result.rigid_body_rao.shape == (2, 1)
    assert result.modal_response.shape == (1, 1)
    assert result.modal_names == ("dry-mode-0",)
    assert jnp.allclose(result.generalized_excitation[2:], force_map.matrix @ excitation)
    assert jnp.allclose(
        result.dynamic_operator.matrix @ result.displacement_response,
        result.generalized_excitation,
    )
    assert bool(result.successful)
    assert bool(result.passive)
    assert not result.continuum_certified


def test_frame_unit_reference_and_modal_geometry_mismatches_fail_before_solve():
    hydrodynamics = _hydrodynamics(jnp.asarray(((0.2,),)), jnp.asarray(((0.1,),)))

    frame_kwargs = _solve_kwargs(1)
    frame_kwargs["frame_id"] = "earth-fixed-z-up"
    with pytest.raises(ValueError, match="frame IDs"):
        solve_hydrodynamic_response_3d(hydrodynamics, **frame_kwargs)

    unit_kwargs = _solve_kwargs(1)
    unit_kwargs["unit_system_id"] = "imperial-water"
    with pytest.raises(ValueError, match="unit-system IDs"):
        solve_hydrodynamic_response_3d(hydrodynamics, **unit_kwargs)

    reference_kwargs = _solve_kwargs(1)
    reference_kwargs["hydrodynamic_reference_point_id"] = "body-reference-set-B"
    with pytest.raises(ValueError, match="reference-point IDs"):
        solve_hydrodynamic_response_3d(hydrodynamics, **reference_kwargs)

    wrong_geometry_map = WetSurfaceModalGeneralizedForceMap3D(
        jnp.asarray(((1.0,),)),
        source_mode_names=hydrodynamics.mode_names,
        modal_names=("dry-mode-0",),
        geometry_id="other-wet-body",
        mapping_id="wrong-geometry-map",
        frame_id="tank-z-up",
        unit_system_id="si-water",
        reference_point_id="body-reference-set-A",
    )
    modal_kwargs = _solve_kwargs(1)
    modal_kwargs.update(
        structural_modal_mass=jnp.asarray(((1.0,),)),
        structural_modal_stiffness=jnp.asarray(((2.0,),)),
        structural_modal_damping=jnp.asarray(((0.1,),)),
        modal_force_map=wrong_geometry_map,
    )
    with pytest.raises(ValueError, match="geometry IDs"):
        solve_hydrodynamic_response_3d(hydrodynamics, **modal_kwargs)


def test_singular_dynamic_stiffness_returns_checked_failure_status():
    hydrodynamics = _hydrodynamics(jnp.asarray(((0.0,),)), jnp.asarray(((0.0,),)))
    result = solve_hydrodynamic_response_3d(
        hydrodynamics,
        angular_frequency=1.0,
        coefficient_frequency_id="omega=1-rad-per-second",
        excitation_id="singular-load",
        physical_mass_inertia=jnp.asarray(((1.0,),)),
        external_damping=jnp.asarray(((0.0,),)),
        hydrostatic_restoring=jnp.asarray(((1.0,),)),
        mooring_restoring=jnp.asarray(((0.0,),)),
        incident_excitation=jnp.asarray((1.0 + 0.0j,)),
        frame_id="tank-z-up",
        unit_system_id="si-water",
        reference_point_id="body-reference-set-A",
        hydrodynamic_reference_point_id="body-reference-set-A",
    )

    assert not bool(result.successful)
    assert result.status == int(HydrodynamicResponseStatus.SINGULAR_DYNAMICS)
    assert result.linear_statuses[0] == int(LinearSolveStatus.SINGULAR)
    assert not bool(result.linear_results[0].successful)
    assert not bool(result.residual_accepted)


def test_passivity_energy_identity_and_nonpassive_input_rejection():
    hydrodynamics = _hydrodynamics(jnp.asarray(((0.5,),)), jnp.asarray(((0.6,),)))
    result = solve_hydrodynamic_response_3d(
        hydrodynamics,
        angular_frequency=2.0,
        coefficient_frequency_id="omega=2-rad-per-second",
        excitation_id="energy-load",
        physical_mass_inertia=jnp.asarray(((2.0,),)),
        external_damping=jnp.asarray(((0.4,),)),
        hydrostatic_restoring=jnp.asarray(((20.0,),)),
        mooring_restoring=jnp.asarray(((0.0,),)),
        incident_excitation=jnp.asarray((1.0 + 0.3j,)),
        frame_id="tank-z-up",
        unit_system_id="si-water",
        reference_point_id="body-reference-set-A",
        hydrodynamic_reference_point_id="body-reference-set-A",
    )

    assert bool(result.passive)
    assert float(result.minimum_total_damping_eigenvalue) == pytest.approx(1.0)
    assert jnp.all(result.average_radiated_power >= 0.0)
    assert jnp.all(result.average_external_dissipation >= 0.0)
    assert jnp.allclose(
        result.average_total_dissipation,
        result.average_radiated_power + result.average_external_dissipation,
    )
    assert jnp.allclose(
        result.average_incident_power,
        result.average_total_dissipation,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert jnp.allclose(result.average_power_balance_residual, 0.0, atol=1.0e-12)

    bad_kwargs = _solve_kwargs(1)
    bad_kwargs["external_damping"] = jnp.asarray(((-0.1,),))
    with pytest.raises(ValueError, match="passive envelope"):
        solve_hydrodynamic_response_3d(hydrodynamics, **bad_kwargs)


def test_periodic_fluid_kernel_and_continuum_certification_are_explicit_non_goals():
    hydrodynamics = _hydrodynamics(jnp.asarray(((0.2,),)), jnp.asarray(((0.1,),)))
    result = solve_hydrodynamic_response_3d(hydrodynamics, **_solve_kwargs(1))

    assert "periodic fluid-kernel synthesis" in result.non_goals
    assert (
        "unprepared repeated-cell or Bloch hydrodynamic coefficients" in result.non_goals
    )
    assert not result.continuum_certified
    assert "no continuum" in result.error_evidence[-1]
    assert result.coefficient_frequency_id == "omega=2-rad-per-second"
    assert result.reference_point_id == "body-reference-set-A"
    assert result.frame_id == hydrodynamics.frame_id
    assert result.unit_system_id == hydrodynamics.unit_system_id
