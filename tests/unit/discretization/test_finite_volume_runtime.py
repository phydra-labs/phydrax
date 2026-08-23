#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _runtime(cells=64, *, cfl=0.4, retries=4):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells),), axis_names=("x",)
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.SupersonicOutflowBoundary(),
        phx.discretization.SupersonicOutflowBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "runtime-sod",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.HLLCFluxPlan(),
        positivity=phx.discretization.ConvexStateLimiterPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method
    )
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        compiled.dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(
            cfl=cfl, maximum_retries=retries, reduction_factor=0.5
        ),
    )
    x = grid.structured_axes[0].interval_centers
    primitive = jnp.stack(
        (
            jnp.where(x < 0.5, 1.0, 0.125),
            jnp.zeros_like(x),
            jnp.where(x < 0.5, 1.0, 0.1),
        ),
        axis=-1,
    )
    return runtime, system.primitive_to_conserved(primitive)


def test_einfeldt_fallback_is_consistent_and_has_finite_bounds():
    system = phx.equations.EulerSystem()
    state = system.primitive_to_conserved(jnp.asarray([[1.0, 0.2, 1.0]]))
    result = phx.discretization.EinfeldtHLLFluxPlan().face_flux(
        system, state, state, 0
    )
    np.testing.assert_allclose(
        result.normal_flux, system.physical_flux(state, 0), rtol=1e-12
    )
    assert jnp.all(jnp.isfinite(result.max_speed))


def test_global_flux_blending_preserves_conservation_and_admissibility():
    system = phx.equations.EulerSystem()
    fallback = system.primitive_to_conserved(
        jnp.asarray([[1.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    )
    high = fallback.at[0, 0].set(-0.1)
    high = high.at[1].add(fallback[0] - high[0])
    result = phx.discretization.FluxPositivityPlan().limit_candidate(
        system, high, fallback
    )

    assert result.report.activated
    assert jnp.all(system.admissible(result.state))
    np.testing.assert_allclose(
        jnp.sum(result.state, axis=0), jnp.sum(high, axis=0), atol=2e-10
    )


def test_runtime_accepts_admissible_step_and_advances_state_atomically():
    runtime, state = _runtime()
    initial = phx.solver.FiniteVolumeRuntimeState(state, 0.0, 0.002)
    result = runtime.advance(initial)

    assert result.accepted
    assert result.runtime_state.accepted_step == 1
    assert result.runtime_state.time > initial.time
    assert jnp.all(runtime.dynamics.system.admissible(result.runtime_state.conservative_state))
    assert jnp.all(jnp.isfinite(result.runtime_state.conservative_state))


def test_runtime_rejects_invalid_initial_state_without_mutation():
    runtime, state = _runtime(retries=1)
    invalid = state.at[0, 0].set(-1.0)
    initial = phx.solver.FiniteVolumeRuntimeState(invalid, 0.0, 0.01)
    result = runtime.advance(initial)

    assert not result.accepted
    assert result.runtime_state.last_status == int(
        phx.solver.FiniteVolumeRunStatus.INVALID_INITIAL_STATE
    )
    np.testing.assert_allclose(result.runtime_state.conservative_state, invalid)
    np.testing.assert_allclose(result.runtime_state.time, initial.time)


def test_runtime_step_is_jittable_and_status_is_bounded():
    runtime, state = _runtime(cells=32)
    initial = phx.solver.FiniteVolumeRuntimeState(state, 0.0, 0.001)
    result = eqx.filter_jit(runtime.advance)(initial)

    assert result.runtime_state.last_status in (
        int(phx.solver.FiniteVolumeRunStatus.SUCCESS),
        int(phx.solver.FiniteVolumeRunStatus.RECOVERED_REJECTION),
    )
    assert result.retries <= runtime.policy.maximum_retries


def test_face_local_positivity_blending_preserves_shared_flux_conservation():
    cells = 6
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    primitive = jnp.broadcast_to(jnp.asarray([1.0, 0.0, 1.0]), (cells, 3))
    state = system.primitive_to_conserved(primitive)
    low_flux = jnp.zeros((cells + 1, 3))
    high_flux = low_flux.at[3, 0].set(100.0)
    result = phx.discretization.FluxPositivityPlan().limit_face_fluxes(
        system,
        state,
        (high_flux,),
        (low_flux,),
        jnp.zeros_like(state),
        jnp.asarray(0.1),
        discretization,
    )

    assert result.report.activated
    assert jnp.all(system.admissible(result.state))
    assert result.face_blend_factors[0][3] < 1.0
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes[..., None] * result.state, axis=0),
        jnp.sum(discretization.cell_volumes[..., None] * state, axis=0),
        atol=2e-11,
    )


def test_runtime_exposes_time_averaged_accepted_integrated_fluxes():
    runtime, state = _runtime(cells=24)
    result = runtime.advance(
        phx.solver.FiniteVolumeRuntimeState(state, 0.0, 0.001)
    )

    assert result.accepted
    assert len(result.accepted_integrated_fluxes) == 1
    assert result.accepted_integrated_fluxes[0].shape == (25, 3)
    assert jnp.all(jnp.isfinite(result.accepted_integrated_fluxes[0]))
