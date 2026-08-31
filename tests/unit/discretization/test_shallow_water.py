#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _compiled_1d(bathymetry, reconstruction, *, source=None):
    bed = jnp.asarray(bathymetry)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(int(bed.size), periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    system = phx.equations.ShallowWaterSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    method = phx.discretization.FiniteVolumeMethodPlan(
        reconstruction,
        phx.discretization.ShallowWaterHydrostaticHLLPlan(),
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    problem = phx.equations.ConservationProblemIR(
        "shallow-water",
        "state",
        system,
        boundaries,
        source=source,
        source_id=None if source is None else source.source_id,
    )
    compiled = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        bathymetry=bed,
    )
    return compiled, bed


def test_shallow_water_system_defines_exact_dry_state():
    system = phx.equations.ShallowWaterSystem(2)
    dry = jnp.asarray((0.0, 0.0, 0.0))
    invalid_dry = jnp.asarray((0.0, 1.0, 0.0))

    np.testing.assert_array_equal(system.conserved_to_primitive(dry), dry)
    np.testing.assert_array_equal(system.physical_flux(dry, 0), dry)
    assert bool(system.admissible(dry))
    assert not bool(system.admissible(invalid_dry))
    assert not bool(system.admissible(jnp.asarray((1.0, jnp.nan, 0.0))))


def test_shallow_water_normal_bounds_are_rotation_covariant():
    system = phx.equations.ShallowWaterSystem(2)
    left = jnp.asarray(((1.0, 0.3, -0.4),))
    right = jnp.asarray(((0.8, -0.1, 0.2),))
    normal = jnp.asarray(((0.6, 0.8),))

    lower, upper = system.normal_signal_bounds(left, right, normal)
    left_velocity = jnp.sum(system.velocity(left) * normal, axis=-1)
    right_velocity = jnp.sum(system.velocity(right) * normal, axis=-1)
    expected_lower = jnp.minimum(
        left_velocity - jnp.sqrt(system.gravity * left[..., 0]),
        right_velocity - jnp.sqrt(system.gravity * right[..., 0]),
    )
    expected_upper = jnp.maximum(
        left_velocity + jnp.sqrt(system.gravity * left[..., 0]),
        right_velocity + jnp.sqrt(system.gravity * right[..., 0]),
    )

    np.testing.assert_allclose(lower, expected_lower)
    np.testing.assert_allclose(upper, expected_upper)


def test_hydrostatic_face_balances_lake_at_rest_step():
    system = phx.equations.ShallowWaterSystem()
    result = phx.discretization.ShallowWaterHydrostaticHLLPlan().face_contribution(
        system,
        jnp.asarray(((0.9, 0.0),)),
        jnp.asarray(((0.7, 0.0),)),
        jnp.asarray((0.1,)),
        jnp.asarray((0.3,)),
        0,
    )

    np.testing.assert_allclose(result.normal_flux[..., 0], 0.0)
    np.testing.assert_allclose(result.left_flux[..., 1], 0.5 * 9.81 * 0.9**2)
    np.testing.assert_allclose(result.right_flux[..., 1], 0.5 * 9.81 * 0.7**2)
    np.testing.assert_array_equal(result.left_correction[..., 0], 0.0)
    np.testing.assert_array_equal(result.right_correction[..., 0], 0.0)


@pytest.mark.parametrize(
    "reconstruction",
    [
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.MUSCLReconstruction(),
    ],
)
def test_compiled_wet_dry_lake_has_zero_residual(reconstruction):
    bathymetry = jnp.asarray((0.1, 0.2, 0.4, 1.2, 1.3, 0.4, 0.2, 0.1))
    compiled, bed = _compiled_1d(bathymetry, reconstruction)
    state = jnp.stack((jnp.maximum(1.0 - bed, 0.0), jnp.zeros_like(bed)), axis=-1)

    residual, diagnostics = compiled.residual_with_diagnostics(0.0, state)

    np.testing.assert_allclose(residual, 0.0, atol=2e-13)
    np.testing.assert_allclose(diagnostics.conservation_defect, 0.0, atol=2e-13)
    assert jnp.all(jnp.isfinite(diagnostics.bed_source_integral))


def test_runtime_preserves_wet_dry_lake_and_records_sided_integrals():
    bathymetry = jnp.asarray((0.1, 0.2, 0.4, 1.2, 1.3, 0.4, 0.2, 0.1))
    compiled, bed = _compiled_1d(bathymetry, phx.discretization.MUSCLReconstruction())
    state = jnp.stack((jnp.maximum(1.0 - bed, 0.0), jnp.zeros_like(bed)), axis=-1)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    runtime_state = runtime.initialize_state(state, 0.0, 0.005)

    result = runtime.advance(runtime_state)

    assert bool(result.accepted)
    np.testing.assert_allclose(result.runtime_state.cell_average(), state, atol=2e-13)
    assert result.shallow_water_integrals is not None
    assert result.shallow_water_integrals.bed_id == compiled.dynamics.bathymetry.bed_id
    assert all(
        jnp.all(integral[..., 0] == 0.0)
        for integral in result.shallow_water_integrals.left_correction_integrals
    )


def test_runtime_preserves_mass_and_positivity_for_dry_dam_break():
    bathymetry = jnp.zeros((64,))
    compiled, _ = _compiled_1d(bathymetry, phx.discretization.MUSCLReconstruction())
    depth = jnp.where(jnp.arange(64) < 32, 1.0, 0.0)
    state = jnp.stack((depth, jnp.zeros_like(depth)), axis=-1)
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.25),
    )
    step = compiled.stable_step(state, cfl=0.25)
    result = runtime.advance(runtime.initialize_state(state, 0.0, step))
    updated = result.runtime_state.cell_average()

    assert bool(result.accepted)
    assert jnp.all(jnp.isfinite(updated))
    assert jnp.all(updated[..., 0] >= 0.0)
    assert jnp.all(jnp.where(updated[..., 0] == 0.0, updated[..., 1] == 0.0, True))
    np.testing.assert_allclose(
        jnp.sum(updated[..., 0]), jnp.sum(state[..., 0]), atol=2e-12
    )


def test_bathymetry_requires_balanced_method_and_balanced_method_requires_bed():
    bed = jnp.zeros((8,))
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(8, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray(((0.0,), (1.0,))))
    system = phx.equations.ShallowWaterSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x",))
    problem = phx.equations.ConservationProblemIR(
        "shallow-water", "state", system, boundaries
    )
    ordinary = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.HLLFluxPlan(),
    )
    balanced = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.ShallowWaterHydrostaticHLLPlan(),
    )

    with pytest.raises(ValueError, match="Bathymetry requires"):
        phx.equations.compile_conservation_problem(
            problem, discretization, ordinary, bathymetry=bed
        )
    with pytest.raises(ValueError, match="requires bathymetry"):
        phx.equations.compile_conservation_problem(problem, discretization, balanced)


def test_coriolis_source_has_zero_mass_and_known_rotation():
    source = phx.equations.ShallowWaterCoriolisSource(2.0, beta=0.5, meridional_axis=1)
    state = jnp.asarray(((1.0, 3.0, 4.0),))
    coordinates = jnp.asarray(((0.0, 2.0),))

    rate = source(jnp.asarray(0.0), state, coordinates)

    np.testing.assert_allclose(rate, jnp.asarray(((0.0, 12.0, -9.0),)))
    np.testing.assert_allclose(source.stable_step(coordinates), jnp.sqrt(3.0) / 3.0)


def test_shallow_water_observables_include_bed_surface_and_energy():
    compiled, bed = _compiled_1d(
        jnp.asarray((0.1, 0.3)),
        phx.discretization.PiecewiseConstantReconstruction(),
    )
    state = jnp.asarray(((0.9, 0.0), (0.7, 0.0)))

    observables = compiled.dynamics.shallow_water_observables(state)

    np.testing.assert_allclose(observables.surface, 1.0)
    np.testing.assert_allclose(observables.bathymetry, bed)
    np.testing.assert_allclose(observables.velocity, 0.0)
    assert observables.bed_id == compiled.dynamics.bathymetry.bed_id


def test_output_snapshot_stores_shallow_water_observables(tmp_path):
    h5py = pytest.importorskip("h5py")
    compiled, _ = _compiled_1d(
        jnp.asarray((0.1, 0.3)),
        phx.discretization.PiecewiseConstantReconstruction(),
    )
    state = jnp.asarray(((0.9, 0.0), (0.7, 0.0)))
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    runtime_state = runtime.initialize_state(state, 0.0, 0.01)
    observables = compiled.dynamics.shallow_water_observables(state)
    target = tmp_path / "shallow-water.h5"
    output = phx.solver.FiniteVolumeOutputPlan(target, compiled.discretization)

    index = output.write_snapshot(
        compiled.discretization,
        runtime_state,
        shallow_water=observables,
    )

    assert index == 0
    with h5py.File(target, "r") as handle:
        group = handle["steps"]["00000000"]["shallow_water"]
        np.testing.assert_allclose(group["surface"], 1.0)
        np.testing.assert_allclose(group["bathymetry"], observables.bathymetry)
        assert group.attrs["bed_id"] == observables.bed_id
