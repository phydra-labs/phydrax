#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax


jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


_INITIAL = jnp.asarray((-0.2, 0.1, 0.3, -0.1), dtype=jnp.float64)


def _mesh():
    vertices = jnp.asarray(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        dtype=jnp.float64,
    )
    cells = jnp.asarray(((0, 1, 3), (1, 2, 3)), dtype=jnp.int32)
    return phx.discretization.CellMesh.from_triangles(vertices, cells)


def _model(*, gradient_coefficient=1.0):
    return phx.applications.phase_field.BinaryPhaseFieldModel(
        phx.equations.BinaryThermodynamicParameters(
            1.0,
            gradient_coefficient,
        )
    )


def _allen_cahn(*, termination=None):
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(),
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    return phx.applications.phase_field.AllenCahnFEMPlan(
        _model(),
        1.0,
        termination=termination,
    ).prepare(discretization, "eta")


def _cahn_hilliard():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(),
        (
            phx.discretization.FiniteElementFieldSpec("c", element),
            phx.discretization.FiniteElementFieldSpec("mu", element),
        ),
    ).prepare()
    return phx.applications.phase_field.CahnHilliardFEMPlan(_model(), 1.0).prepare(
        discretization, "c", "mu"
    )


def test_allen_cahn_nonuniform_step_closes_energy_dissipation_ledger():
    method = _allen_cahn()
    initial = method.initialize(_INITIAL)

    result = method.step_detailed(
        jnp.asarray(0),
        jnp.asarray(0.0),
        initial,
        jnp.asarray(0.01),
    )

    assert bool(result.successful)
    assert bool(result.evidence.energy_stable)
    assert result.evidence.energy_after < result.evidence.energy_before
    assert result.evidence.dissipation > 0.0
    assert result.evidence.energy_balance_defect <= result.evidence.energy_tolerance


def test_cahn_hilliard_nonuniform_steps_preserve_reference_mass_and_energy_ledger():
    method = _cahn_hilliard()
    initial = method.initialize(_INITIAL)
    state = initial
    energies = [initial.energy]

    for step_index in range(3):
        result = method.step_detailed(
            jnp.asarray(step_index),
            jnp.asarray(0.005 * step_index),
            state,
            jnp.asarray(0.005),
        )
        assert bool(result.successful)
        assert bool(result.evidence.mass_conserved)
        assert bool(result.evidence.energy_stable)
        assert result.evidence.energy_balance_defect <= result.evidence.energy_tolerance
        state = result.accepted_state
        energies.append(state.energy)

    assert jnp.all(jnp.diff(jnp.stack(energies)) < 0.0)
    assert jnp.abs(state.mass - initial.reference_mass) <= (
        method.plan.acceptance.mass_tolerance(
            initial.reference_mass,
            method.domain_measure,
        )
    )


def test_phase_field_preparation_rejects_underresolved_interface():
    element = phx.discretization.lagrange_element("triangle", 1)
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(),
        phx.discretization.FiniteElementFieldSpec("eta", element),
    ).prepare()
    with pytest.raises(ValueError, match="underresolved"):
        phx.applications.phase_field.AllenCahnFEMPlan(
            _model(gradient_coefficient=0.02),
            1.0,
        ).prepare(discretization, "eta")


def test_failed_allen_cahn_attempt_preserves_exact_accepted_state():
    method = _allen_cahn(
        termination=phx.nonlinear.NonlinearTermination(
            absolute_residual=0.0,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=1,
        )
    )
    initial = method.initialize(_INITIAL)

    result = method.step_detailed(
        jnp.asarray(0),
        jnp.asarray(0.0),
        initial,
        jnp.asarray(0.05),
    )

    assert not bool(result.successful)
    assert not bool(result.evidence.nonlinear_successful)
    for accepted, previous in zip(
        jax.tree.leaves(result.accepted_state),
        jax.tree.leaves(initial),
        strict=True,
    ):
        np.testing.assert_array_equal(accepted, previous)


def test_phase_field_production_checkpoint_restart_matches_uninterrupted(
    tmp_path,
):
    method = _allen_cahn()
    initial = method.initialize(_INITIAL)
    case = method.production_case("allen-cahn-restart", initial)
    plan = method.production_run_plan(
        step_size=0.01,
        end_time=0.03,
        maximum_steps=3,
        checkpoint_interval=1,
        segment_steps=1,
        retry_policy=phx.solver.RobustRetryPolicy(maximum_retries=0),
    )
    policy = phx.solver.CheckpointGenerationPolicy(3)

    complete_store = phx.solver.DurableCheckpointStore(
        tmp_path / "complete",
        case.manifest,
        policy,
    )
    complete_runtime = phx.solver.PreparedProductionRun(
        case.manifest,
        plan,
        complete_store,
    )
    complete = complete_runtime.run(complete_runtime.initial_state(case.initial_state))

    interrupted_path = tmp_path / "interrupted"
    interrupted_store = phx.solver.DurableCheckpointStore(
        interrupted_path,
        case.manifest,
        policy,
    )
    interrupted_runtime = phx.solver.PreparedProductionRun(
        case.manifest,
        plan,
        interrupted_store,
    )
    after_one, attempted = interrupted_runtime.step(
        interrupted_runtime.initial_state(case.initial_state)
    )
    assert bool(attempted.successful)
    assert int(after_one.step_index) == 1

    resumed_store = phx.solver.DurableCheckpointStore(
        interrupted_path,
        case.manifest,
        policy,
    )
    resumed_runtime = phx.solver.PreparedProductionRun(
        case.manifest,
        plan,
        resumed_store,
    )
    template = resumed_runtime.initial_state(case.initial_state)
    resumed = resumed_runtime.run(resumed_runtime.resume(template))

    assert bool(complete.successful)
    assert bool(resumed.successful)
    assert int(resumed.state.step_index) == int(complete.state.step_index)
    np.testing.assert_array_equal(resumed.state.time, complete.state.time)
    for restarted, uninterrupted in zip(
        jax.tree.leaves(resumed.state.accepted_state),
        jax.tree.leaves(complete.state.accepted_state),
        strict=True,
    ):
        np.testing.assert_array_equal(restarted, uninterrupted)
