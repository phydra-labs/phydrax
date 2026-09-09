# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Source weak forms, fixed-law trajectories and atomic prepared-state acceptance."""

from pathlib import Path
from shutil import copyfile

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._trainable import combine_trainable, partition_trainable
from phydrax.applications.skeletal_muscle.continuum import (
    Almonacid2024Control,
    Almonacid2024InputHistory,
    Almonacid2024MuscleAponeurosisPlan,
    almonacid_2024_repository_case,
)
from phydrax.linalg import TolerancePolicy
from phydrax.nonlinear import NonlinearTermination


INPUTS = Path(__file__).resolve().parents[2] / "fixtures/flexodeal_0698e3d"


@pytest.fixture(scope="module")
def model():
    with jax.enable_x64(True):
        plan, parameters, _, _, _ = almonacid_2024_repository_case(INPUTS, refinement=0)
        yield plan.prepare(parameters)


@pytest.fixture(scope="module")
def candidate(model):
    control = Almonacid2024Control(0.01, 0.0, 0.0, source_id=model.plan.control_source_id)
    result = eqx.filter_jit(lambda p, c: p.propose(c))(model, control)
    assert bool(result.successful)
    return result


def test_total_degree_one_weak_constraint_is_cell_local(model):
    # A slope in one cell must not leak into its neighbour, and the exact four
    # test moments are those of 1,x,y,z, not tensor Q1's eight nodal modes.
    state = model.state
    dilation = state.dilation_coefficients.at[0, 1].set(0.06)
    control = Almonacid2024Control(0.01, 0.0, 0.0, source_id=model.plan.control_source_id)
    residual, _ = model.residual(
        state.displacement_m, state.pressure_coefficients_Pa, dilation, control
    )
    volumes = np.sum(model.geometry.weights_m3, axis=1)
    expected = np.zeros_like(residual[1])
    expected[0] = -0.06 * volumes[0] * np.array((0.5, 1 / 3, 0.25, 0.25))
    np.testing.assert_allclose(residual[1], expected, atol=2e-18, rtol=3e-12)


def test_uniform_translation_has_source_backward_euler_total_inertia(model):
    state = model.state
    translation = jnp.array((0.0002, -0.0001, 0.0003))
    u = jnp.broadcast_to(translation, state.displacement_m.shape)
    dt = 0.012
    control = Almonacid2024Control(dt, 0.0, 0.0, source_id=model.plan.control_source_id)
    residual, _ = model.residual(
        u, state.pressure_coefficients_Pa, state.dilation_coefficients, control
    )
    mass = float(model.parameters.density_kg_per_m3) * float(
        jnp.sum(model.geometry.weights_m3)
    )
    # Internal forces (including source aponeurosis rest prestress) have zero
    # sum by partition of unity, leaving exactly the momentum increment.
    np.testing.assert_allclose(
        np.sum(residual[0], axis=0),
        mass * np.asarray(translation) / dt**2,
        rtol=2e-10,
        atol=1e-10,
    )
    velocity = jnp.broadcast_to(translation / dt, state.velocity_m_per_s.shape)
    moving = eqx.tree_at(lambda x: x.state.velocity_m_per_s, model, velocity)
    residual, _ = moving.residual(
        u, state.pressure_coefficients_Pa, state.dilation_coefficients, control
    )
    np.testing.assert_allclose(np.sum(residual[0], axis=0), np.zeros(3), atol=1e-10)


def test_history_rejects_undefined_pre_start_and_holds_last_value():
    with jax.enable_x64(True):
        history = Almonacid2024InputHistory(
            ((0.1, 0.0), (0.3, 1.0)), ((0.1, 0.0), (0.2, 0.02)), source_id="history"
        )
        assert not bool(history.sample(0.0).successful)
        np.testing.assert_allclose(history.sample(0.2).activation, 0.5)
        np.testing.assert_allclose(history.sample(0.4).engineering_strain, 0.02)


def test_repository_control_preserves_finite_source_overshoot(model):
    _, _, history, _, _ = almonacid_2024_repository_case(INPUTS, refinement=0)
    control = history.sample(0.85)
    assert float(control.activation) > 1.0
    state = model.state
    response = model.material_response(
        state.displacement_m,
        state.pressure_coefficients_Pa,
        state.dilation_coefficients,
        control.activation,
        0.01,
    )
    assert bool(jnp.all(response.admissible))


def test_commit_rolls_back_the_whole_preparation_and_rejects_stale_state(
    model, candidate
):
    rejected = eqx.filter_jit(lambda c, p: c.commit(p, accept=False))(candidate, model)
    assert bool(eqx.tree_equal(rejected, model, typematch=True))
    advanced = candidate.commit(model)
    assert float(advanced.state.time_s) == 0.01
    assert int(advanced.state.accepted_steps) == 1
    np.testing.assert_allclose(
        advanced.state.reference_velocity_gradient_per_s,
        (advanced.state.deformation_gradient - model.state.deformation_gradient) / 0.01,
        atol=1e-13,
    )
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(lambda c, p: c.commit(p))(candidate, advanced)


@pytest.mark.parametrize(
    "field",
    [
        lambda p: p.parameters.density_kg_per_m3,
        lambda p: p.parameters.aponeurosis.maximum_fiber_stress_Pa,
        lambda p: p.geometry.gradients,
        lambda p: p.geometry.weights_m3,
        lambda p: p.geometry.traces.weights_m2,
        lambda p: p.geometry.free_dofs,
    ],
    ids=[
        "density",
        "law",
        "volume-gradients",
        "volume-quadrature",
        "face-quadrature",
        "dof-topology",
    ],
)
def test_commit_rejects_changed_numeric_origin(model, candidate, field):
    previous = field(model)
    changed = eqx.tree_at(field, model, previous.at[...].add(jnp.ones_like(previous)))
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(lambda c, p: c.commit(p))(candidate, changed)


def test_candidate_rejects_changed_nested_solver_policy(model, candidate):
    method = eqx.tree_at(
        lambda m: m.linear_policy.tolerance,
        model.plan.method,
        TolerancePolicy(relative=1e-9, absolute=1e-12, max_steps=2048),
    )
    plan = Almonacid2024MuscleAponeurosisPlan(
        model.plan.geometry,
        control_source_id=model.plan.control_source_id,
        dynamic=model.plan.dynamic,
        pulling_face_id=model.plan.pulling_face_id,
        stress_scale_Pa=model.plan.stress_scale_Pa,
        method=method,
        termination=model.plan.termination,
    )
    assert plan.plan_id != model.plan.plan_id
    foreign = eqx.tree_at(lambda p: p.plan, model, plan)
    with pytest.raises(ValueError):
        candidate.commit(foreign)


def test_failed_plan_policy_cannot_advance_even_when_accept_requested(model):
    plan = Almonacid2024MuscleAponeurosisPlan(
        model.plan.geometry,
        control_source_id=model.plan.control_source_id,
        dynamic=model.plan.dynamic,
        pulling_face_id=model.plan.pulling_face_id,
        stress_scale_Pa=model.plan.stress_scale_Pa,
        method=model.plan.method,
        termination=NonlinearTermination(maximum_steps=1, maximum_evaluations=1),
    )
    limited = plan.prepare(model.parameters)
    assert plan.plan_id != model.plan.plan_id
    control = Almonacid2024Control(0.01, 0.2, 0.001, source_id=plan.control_source_id)
    failed = eqx.filter_jit(lambda p, c: p.propose(c))(limited, control)
    assert not bool(failed.nonlinear_result.successful)
    assert not bool(failed.successful)
    assert bool(
        eqx.tree_equal(failed.commit(limited, accept=True), limited, typematch=True)
    )


def test_repository_loader_rejects_same_shape_input_tampering(tmp_path):
    for name in (
        "manifest.json",
        "parameters.prm",
        "control_points_activation.dat",
        "control_points_strain.dat",
    ):
        copyfile(INPUTS / name, tmp_path / name)
    table = np.loadtxt(tmp_path / "control_points_activation.dat")
    table[-1, 1] += 0.001
    np.savetxt(tmp_path / "control_points_activation.dat", table)
    with pytest.raises(ValueError, match="content mismatch"):
        almonacid_2024_repository_case(tmp_path, refinement=0)


def test_content_bound_foreign_history_is_rejected_before_solving(model):
    _, _, history, _, _ = almonacid_2024_repository_case(INPUTS, refinement=0)
    activation = np.column_stack((history.activation_time_s, history.activation))
    strain = np.column_stack((history.strain_time_s, history.engineering_strain))
    activation[-1, 1] += 0.001
    foreign = Almonacid2024InputHistory(
        activation, strain, source_id=history.source_provenance
    )
    with pytest.raises(ValueError, match="foreign input source"):
        model.propose(foreign.sample(0.01))


@pytest.mark.parametrize(
    "field",
    [
        lambda p: p.parameters.density_kg_per_m3,
        lambda p: p.parameters.muscle.maximum_fiber_stress_Pa,
    ],
    ids=["density", "material-law"],
)
def test_accepted_trajectory_rejects_parameter_changes(model, candidate, field):
    advanced = candidate.commit(model)
    changed = eqx.tree_at(field, advanced, field(advanced) * 1.01)
    control = Almonacid2024Control(0.02, 0.0, 0.0, source_id=model.plan.control_source_id)
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(lambda p, c: p.propose(c))(changed, control)
    with pytest.raises(eqx.EquinoxRuntimeError):
        eqx.filter_jit(lambda p: p.quadrature_fields(0.01))(changed)


def test_first_step_energy_uses_the_trainable_law_not_an_old_cache(model):
    # A prestrained initial state makes law ownership observable even before the
    # first accepted step (the default stress-free reference has zero energy).
    s = model.state
    u = model._displacement(
        jnp.zeros_like(s.displacement_m[model.geometry.free_dofs]), jnp.asarray(0.0002)
    )
    response = model.material_response(
        u, s.pressure_coefficients_Pa, s.dilation_coefficients, 0.0, 0.01
    )
    energy = jnp.sum(response.passive_energy_density_J_per_m3 * model.geometry.weights_m3)
    initial = eqx.tree_at(
        lambda p: (
            p.state.displacement_m,
            p.state.deformation_gradient,
            p.state.passive_energy_J,
        ),
        model,
        (u, model.deformation(u), energy),
    )
    varied = eqx.tree_at(
        lambda p: p.parameters.aponeurosis.maximum_base_stress_Pa,
        initial,
        initial.parameters.aponeurosis.maximum_base_stress_Pa * 2.0,
    )
    control = Almonacid2024Control(
        0.01, 0.0, 0.0002, source_id=model.plan.control_source_id
    )
    residual, response = varied.residual(
        u, s.pressure_coefficients_Pa, s.dilation_coefficients, control
    )
    new_energy = jnp.sum(
        response.passive_energy_density_J_per_m3 * model.geometry.weights_m3
    )
    assert float(new_energy) > float(energy)
    diagnostics = varied._diagnostics(
        u,
        s.velocity_m_per_s,
        s.pressure_coefficients_Pa,
        s.dilation_coefficients,
        control,
        residual,
        response,
    )
    np.testing.assert_allclose(diagnostics.work_energy_residual_J, 0.0, atol=1e-14)


@pytest.mark.parametrize(
    "history_values", [False, True], ids=["control", "interpolated-history"]
)
def test_optimizer_changes_control_response_but_not_the_clock(model, history_values):
    if history_values:
        inputs = Almonacid2024InputHistory(
            ((0.1, 0.0), (0.3, 0.0)), ((0.1, 0.0), (0.3, 0.0)), source_id="optimization"
        )
        sample = lambda x: x.sample(0.2)
        learning_rate = 1.0
    else:
        inputs = Almonacid2024Control(
            0.2, 0.0, 0.0, source_id=model.plan.control_source_id
        )
        sample = lambda x: x
        learning_rate = 0.5
    trainable, fixed = partition_trainable(inputs)

    def objective(values):
        control = sample(combine_trainable(values, fixed))
        return (control.activation - 0.2) ** 2 + (
            control.engineering_strain - 0.0002
        ) ** 2

    gradient = jax.grad(objective)(trainable)
    updated = jax.tree_util.tree_map(
        lambda x, g: x - learning_rate * g, trainable, gradient
    )
    optimized = sample(combine_trainable(updated, fixed))
    assert float(objective(updated)) < float(objective(trainable)) * 1e-12
    np.testing.assert_array_equal(optimized.time_s, sample(inputs).time_s)
    if history_values:
        updated_history = combine_trainable(updated, fixed)
        np.testing.assert_array_equal(
            updated_history.activation_time_s, inputs.activation_time_s
        )
        np.testing.assert_array_equal(updated_history.strain_time_s, inputs.strain_time_s)

    def stress(control):
        s = model.state
        u = model._displacement(
            jnp.zeros_like(s.displacement_m[model.geometry.free_dofs]),
            control.engineering_strain,
        )
        return model.material_response(
            u,
            s.pressure_coefficients_Pa,
            s.dilation_coefficients,
            control.activation,
            control.time_s,
        ).first_piola_Pa

    target = Almonacid2024Control(
        0.2, 0.2, 0.0002, source_id=model.plan.control_source_id
    )
    np.testing.assert_allclose(stress(optimized), stress(target), rtol=1e-12, atol=1e-9)


def test_fixed_parameter_rollout_retains_implicit_sensitivity(model):
    def reaction(log_density):
        prepared = eqx.tree_at(
            lambda p: p.parameters.density_kg_per_m3,
            model,
            model.parameters.density_kg_per_m3 * jnp.exp(log_density),
        )
        successful = jnp.asarray(True)
        for step in (1, 2):
            control = Almonacid2024Control(
                step * 0.01, 0.0, 0.0, source_id=model.plan.control_source_id
            )
            candidate = prepared.propose(control)
            successful = successful & candidate.successful
            prepared = candidate.commit(prepared)
        return jnp.where(successful, candidate.diagnostics.reaction_pulling_N[0], jnp.nan)

    zero = jnp.asarray(0.0)
    _, tangent = eqx.filter_jit(lambda x: jax.jvp(reaction, (x,), (jnp.ones_like(x),)))(
        zero
    )
    evaluate = eqx.filter_jit(reaction)
    epsilon = 1e-3
    difference = (evaluate(zero + epsilon) - evaluate(zero - epsilon)) / (2 * epsilon)
    assert abs(float(difference)) > 1e-10
    np.testing.assert_allclose(tangent, difference, rtol=5e-3, atol=1e-7)
