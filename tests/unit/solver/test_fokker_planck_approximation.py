import diffrax as dfx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx
from phydrax.solver._fokker_planck_approximation import (
    ParticleFokkerPlanckPlan,
    solve_particle_fokker_planck,
    WeakObservable,
)
from phydrax.solver._spde_truncation import (
    prepare_spde_approximation,
    solve_spde_approximation,
    SPDEApproximationFamily,
    SPDEApproximationLevel,
)
from phydrax.stochastic._path_ensemble import (
    prepare_stochastic_path_ensemble,
    solve_stochastic_path_ensemble,
    StochasticPathEnsemblePlan,
)


def _brownian_problem(initial_value=0.0):
    return phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([initial_value]),
        t0=0.0,
        t1=0.1,
        wiener_terms=(
            phx.solver.WienerTerm(
                "brownian",
                lambda time, state, args: jnp.ones((1, 1)),
                (1,),
                structure="additive",
            ),
        ),
    )


def _path_ensemble_stop_condition(time, state, args):
    del time, args
    return state[0] > 100.0


def _ensemble_plan(path_count=8):
    return StochasticPathEnsemblePlan(
        phx.dynamics.TimeGrid(jnp.asarray([0.0, 0.05, 0.1]), time_id="weak-grid"),
        path_count=path_count,
        dt0=0.01,
        max_steps=64,
    )


def _small_spde():
    axis = phx.discretization.FourierAxisSpec(4).materialize(0.0, 1.0)
    discretization = phx.discretization.TensorSpectralDiscretization.from_axes((axis,))
    basis = phx.stochastic.SpatialNoiseBasis.from_spectrum(discretization, 0.02, rank=1)
    return phx.solver.semidiscretize_reaction_diffusion(
        jnp.zeros(discretization.state_shape),
        discretization,
        t0=0.0,
        t1=0.1,
        kappa=0.1,
        noise_basis=basis,
    )


def test_path_ensemble_has_fixed_output_grid_replay_and_backend_evidence():
    problem = _brownian_problem()
    plan = _ensemble_plan()
    first = solve_stochastic_path_ensemble(
        prepare_stochastic_path_ensemble(problem, plan, key=jr.key(7))
    )
    replay = solve_stochastic_path_ensemble(
        prepare_stochastic_path_ensemble(problem, plan, key=jr.key(7))
    )

    assert first.states.shape == (8, 3, 1)
    assert first.path_valid.shape == (8,)
    assert first.accepted_steps.shape == (8,)
    assert first.temporal_evidence is not None
    assert first.realization_id == replay.realization_id
    assert jnp.array_equal(first.states, replay.states)


def test_path_ensemble_identity_includes_initial_state_content_and_absence():
    problem = _brownian_problem()
    plan = _ensemble_plan(path_count=2)
    absent = prepare_stochastic_path_ensemble(problem, plan, key=jr.key(11))
    explicit_default = prepare_stochastic_path_ensemble(
        problem,
        plan,
        realization=absent.realization,
        initial_states=jnp.broadcast_to(
            problem.initial_state,
            (plan.path_count,) + problem.initial_state.shape,
        ),
    )
    absent_replay = prepare_stochastic_path_ensemble(
        _brownian_problem(),
        plan,
        realization=absent.realization,
    )
    changed_default = prepare_stochastic_path_ensemble(
        _brownian_problem(1.0),
        plan,
        realization=absent.realization,
    )
    first = prepare_stochastic_path_ensemble(
        problem,
        plan,
        realization=absent.realization,
        initial_states=jnp.asarray([[1.0], [2.0]], dtype=problem.initial_state.dtype),
    )
    replay = prepare_stochastic_path_ensemble(
        problem,
        plan,
        realization=absent.realization,
        initial_states=jnp.asarray([[1.0], [2.0]], dtype=problem.initial_state.dtype),
    )
    changed = prepare_stochastic_path_ensemble(
        problem,
        plan,
        realization=absent.realization,
        initial_states=jnp.asarray([[1.0], [3.0]], dtype=problem.initial_state.dtype),
    )

    assert first.prepared_id == replay.prepared_id
    assert first.prepared_id != changed.prepared_id
    assert absent.prepared_id != explicit_default.prepared_id
    assert absent.prepared_id == absent_replay.prepared_id
    assert absent.prepared_id != changed_default.prepared_id

    first_result = solve_stochastic_path_ensemble(first)
    replay_result = solve_stochastic_path_ensemble(replay)
    changed_result = solve_stochastic_path_ensemble(changed)
    assert first_result.result_id == replay_result.result_id
    assert first_result.result_id != changed_result.result_id


def test_path_ensemble_default_identity_binds_all_solve_settings():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.05, 0.1]), time_id="configuration-grid"
    )

    def configured_plan(
        *,
        time_grid=grid,
        path_count=2,
        dt0=0.01,
        solver=None,
        stepsize_controller=None,
        adjoint=None,
        event=None,
        event_id=None,
        max_steps=64,
        rtol=1.0e-6,
        atol=1.0e-8,
        wiener_tolerance=1.0e-3,
        levy_area="brownian",
        dense=False,
        throw=False,
    ):
        return StochasticPathEnsemblePlan(
            time_grid,
            path_count=path_count,
            dt0=dt0,
            solver=solver,
            stepsize_controller=stepsize_controller,
            adjoint=adjoint,
            event=event,
            event_id=event_id,
            max_steps=max_steps,
            rtol=rtol,
            atol=atol,
            wiener_tolerance=wiener_tolerance,
            levy_area=levy_area,
            dense=dense,
            throw=throw,
        )

    first = configured_plan(
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    replay = configured_plan(
        solver=dfx.Euler(),
        stepsize_controller=dfx.ConstantStepSize(),
        adjoint=dfx.RecursiveCheckpointAdjoint(),
    )
    assert first.plan_id == replay.plan_id
    assert first.configuration_id == replay.configuration_id

    changed = (
        configured_plan(
            dt0=0.02,
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
        ),
        configured_plan(
            solver=dfx.EulerHeun(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.PIDController(rtol=1.0e-6, atol=1.0e-8),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.ImplicitAdjoint(),
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            event=dfx.Event(_path_ensemble_stop_condition),
            event_id="path-ensemble-event",
        ),
        configured_plan(
            time_grid=phx.dynamics.TimeGrid(
                jnp.asarray([0.0, 0.04, 0.1]), time_id="configuration-grid"
            ),
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
        ),
        configured_plan(
            path_count=3,
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            max_steps=65,
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            rtol=2.0e-6,
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            atol=2.0e-8,
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            wiener_tolerance=2.0e-3,
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            levy_area="space_time",
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            dense=True,
        ),
        configured_plan(
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            throw=True,
        ),
    )
    identifiers = {first.plan_id, *(plan.plan_id for plan in changed)}
    assert len(identifiers) == len(changed) + 1


def test_path_ensemble_requires_identity_for_opaque_execution_objects():
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.05, 0.1]), time_id="opaque-event-grid"
    )
    event = dfx.Event(_path_ensemble_stop_condition)
    with pytest.raises(ValueError, match="event_id"):
        StochasticPathEnsemblePlan(grid, path_count=2, dt0=0.01, event=event)

    first = StochasticPathEnsemblePlan(
        grid,
        path_count=2,
        dt0=0.01,
        event=event,
        event_id="path-ensemble-event:v1",
    )
    replay = StochasticPathEnsemblePlan(
        grid,
        path_count=2,
        dt0=jnp.asarray(0.01),
        event=dfx.Event(_path_ensemble_stop_condition),
        event_id="path-ensemble-event:v1",
    )
    changed = StochasticPathEnsemblePlan(
        grid,
        path_count=2,
        dt0=0.01,
        event=event,
        event_id="path-ensemble-event:v2",
    )
    assert first.plan_id == replay.plan_id
    assert first.plan_id != changed.plan_id


def test_path_ensemble_execution_identity_propagates_to_prepared_and_result():
    problem = _brownian_problem()
    grid = phx.dynamics.TimeGrid(
        jnp.asarray([0.0, 0.05, 0.1]), time_id="execution-identity-grid"
    )

    def plan(dt0):
        return StochasticPathEnsemblePlan(
            grid,
            path_count=2,
            dt0=dt0,
            solver=dfx.Euler(),
            stepsize_controller=dfx.ConstantStepSize(),
            adjoint=dfx.RecursiveCheckpointAdjoint(),
            max_steps=64,
        )

    first_plan = plan(0.01)
    replay_plan = plan(jnp.asarray(0.01))
    changed_plan = plan(0.02)
    initial_states = jnp.asarray([[1.0], [2.0]], dtype=problem.initial_state.dtype)
    first = prepare_stochastic_path_ensemble(
        problem,
        first_plan,
        key=jr.key(19),
        initial_states=initial_states,
    )
    replay = prepare_stochastic_path_ensemble(
        problem,
        replay_plan,
        realization=first.realization,
        initial_states=jnp.array(initial_states),
    )
    changed = prepare_stochastic_path_ensemble(
        problem,
        changed_plan,
        realization=first.realization,
        initial_states=jnp.array(initial_states),
    )

    assert first_plan.plan_id == replay_plan.plan_id
    assert first_plan.plan_id != changed_plan.plan_id
    assert first.prepared_id == replay.prepared_id
    assert first.prepared_id != changed.prepared_id

    first_result = solve_stochastic_path_ensemble(first)
    replay_result = solve_stochastic_path_ensemble(replay)
    changed_result = solve_stochastic_path_ensemble(changed)
    assert first_result.result_id == replay_result.result_id
    assert first_result.result_id != changed_result.result_id
    assert first_result.temporal_evidence.configuration_id == first_plan.configuration_id


def test_path_ensemble_rejects_realization_capacity_mismatch():
    problem = _brownian_problem()
    wrong = phx.stochastic.WienerRealization.independent(
        jr.key(2),
        problem.noise_shape,
        support=(0.0, 0.1),
        sample_shape=(3,),
        noise_id=problem.noise_id,
    )
    with pytest.raises(ValueError, match="path_count"):
        prepare_stochastic_path_ensemble(
            problem,
            _ensemble_plan(path_count=4),
            realization=wrong,
        )


def test_particle_fokker_planck_returns_normalized_empirical_weak_laws():
    plan = ParticleFokkerPlanckPlan(
        _ensemble_plan(path_count=16),
        (
            WeakObservable(lambda state: state[0], observable_id="mean"),
            WeakObservable(lambda state: state[0] ** 2, observable_id="second"),
        ),
        0.95,
        jnp.asarray([0.0, 0.05, 0.1]),
    )
    result = solve_particle_fokker_planck(_brownian_problem(), plan, key=jr.key(9))

    assert result.approximation_kind == "particle-weak-law"
    assert len(result.laws) == 3
    assert result.observable_means.shape == (2, 3)
    assert result.weak_residuals.shape == (2, 2)
    assert jnp.all(jnp.isfinite(result.sampling_errors))


def test_finite_spde_family_reports_coupled_cauchy_and_tail_evidence():
    spde = _small_spde()
    levels = (
        SPDEApproximationLevel(
            spde,
            _ensemble_plan().time_grid,
            lambda values: values,
            (4, 1),
            4.0,
            level_id="coarse",
        ),
        SPDEApproximationLevel(
            spde,
            _ensemble_plan().time_grid,
            lambda values: values,
            (4, 1),
            8.0,
            level_id="replay-fine",
        ),
    )
    family = SPDEApproximationFamily(
        levels,
        "time",
        1,
        "shared-spde-noise",
        tail_envelope=jnp.asarray([0.2, 0.1]),
    )
    result = solve_spde_approximation(
        prepare_spde_approximation(
            family,
            ensemble_plan=_ensemble_plan(path_count=4),
            key=jr.key(11),
        )
    )

    assert result.approximation_kind == "finite-time-refinement"
    assert result.cauchy_differences.shape == (1,)
    assert jnp.allclose(result.cauchy_differences, 0.0)
    assert jnp.array_equal(result.tail_bounds, jnp.asarray([0.2, 0.1]))
