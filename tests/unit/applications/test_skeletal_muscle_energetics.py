#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.skeletal_muscle.energetics import (
    integrate_metabolic_energy_joule,
    UchidaUmberger2010Parameters,
    UchidaUmberger2010Plan,
)


def _plan(mass=(0.5, 1.0)):
    return UchidaUmberger2010Plan(
        UchidaUmberger2010Parameters(
            jnp.asarray(mass),
            jnp.asarray((0.5, 0.7)),
            jnp.asarray((0.1, 0.12)),
            jnp.asarray((10.0, 10.0)),
        ),
        ("muscle-a", "muscle-b"),
    )


def _scalar_plan(*, minimum_heat_rate=1.0):
    return UchidaUmberger2010Plan(
        UchidaUmberger2010Parameters(
            jnp.asarray((1.0,)),
            jnp.asarray((0.5,)),
            jnp.asarray((0.1,)),
            jnp.asarray((10.0,)),
            minimum_heat_rate_W_per_kg=minimum_heat_rate,
        ),
        ("muscle",),
    )


def test_zero_activity_uses_declared_heat_floor_and_mass_scaling():
    plan = _plan()
    zeros = jnp.zeros(2)
    result = plan.evaluate(
        zeros,
        zeros,
        zeros,
        jnp.ones(2),
        jnp.asarray((0.1, 0.12)),
        zeros,
    )

    assert bool(result.evidence.successful)
    np.testing.assert_allclose(result.heat_rate_W_per_kg, 1.0)
    np.testing.assert_allclose(result.muscle_metabolic_power_W, (0.5, 1.0))
    assert bool(jnp.all(result.evidence.heat_floor_active))


def test_zero_excitation_has_a_finite_gradient_with_nonzero_activation():
    plan = _scalar_plan()

    def power(excitation):
        return plan.evaluate(
            jnp.asarray((excitation,)),
            jnp.asarray((0.4,)),
            jnp.zeros(1),
            jnp.asarray((0.9,)),
            jnp.asarray((0.095,)),
            jnp.asarray((-0.01,)),
        ).total_muscle_metabolic_power_W

    assert jnp.isfinite(jax.grad(power)(jnp.asarray(0.0)))
    result = plan.evaluate(
        jnp.zeros(1),
        jnp.asarray((0.4,)),
        jnp.zeros(1),
        jnp.asarray((0.9,)),
        jnp.asarray((0.095,)),
        jnp.asarray((-0.01,)),
    )
    assert not bool(result.evidence.branch_smooth)


def test_shortening_work_and_derived_lengthening_correction_are_explicit():
    plan = _plan()
    excitation = jnp.asarray((0.8, 0.8))
    activation = jnp.asarray((0.7, 0.7))
    force = jnp.asarray((100.0, 0.0))
    force_length = jnp.asarray((1.0, 1.0))
    length = jnp.asarray((0.1, 0.12))
    velocity = jnp.asarray((-0.01, 0.5))
    unloaded = plan.evaluate(
        excitation,
        activation,
        force,
        force_length,
        length,
        velocity,
    )
    heat_without_work = (
        unloaded.activation_maintenance_heat_W_per_kg
        + unloaded.shortening_lengthening_heat_W_per_kg
    )
    force = force.at[1].set(
        plan.parameters.muscle_mass_kg[1]
        * (heat_without_work[1] + 10.0)
        / velocity[1]
    )
    result = plan.evaluate(
        excitation,
        activation,
        force,
        force_length,
        length,
        velocity,
    )
    raw_total = heat_without_work[1] + result.mechanical_work_W_per_kg[1]

    assert result.mechanical_work_W_per_kg[0] > 0.0
    assert result.mechanical_work_W_per_kg[1] < 0.0
    assert raw_total < 0.0
    assert bool(result.evidence.negative_power_correction_active[1])
    assert bool(jnp.all(result.muscle_metabolic_power_W >= 0.0))
    assert result.muscle_metabolic_power_W[1] == 0.0
    corrected_energy = integrate_metabolic_energy_joule(
        jnp.asarray((0.0, 1.0)),
        jnp.stack(
            (result.muscle_metabolic_power_W, result.muscle_metabolic_power_W)
        ),
    )
    assert corrected_energy[1] == 0.0


def test_energy_integral_jit_and_local_parameter_derivative():
    plan = _plan()
    excitation = jnp.asarray((0.75, 0.65))
    activation = jnp.asarray((0.6, 0.55))
    force = jnp.asarray((80.0, 90.0))
    force_length = jnp.asarray((0.9, 0.85))
    length = jnp.asarray((0.095, 0.115))
    velocity = jnp.asarray((-0.01, -0.015))
    result = eqx.filter_jit(plan.evaluate)(
        excitation, activation, force, force_length, length, velocity
    )
    assert bool(result.evidence.successful)

    time = jnp.asarray((0.0, 0.5, 1.0))
    trace = jnp.stack(
        (result.muscle_metabolic_power_W,) * 3,
        axis=0,
    )
    energy = integrate_metabolic_energy_joule(time, trace)
    np.testing.assert_allclose(energy, result.muscle_metabolic_power_W)

    derivative = jax.grad(
        lambda value: jnp.sum(
            eqx.tree_at(
                lambda model: model.parameters.aerobic_factor,
                plan,
                value,
            ).evaluate(
                excitation,
                activation,
                force,
                force_length,
                length,
                velocity,
            ).muscle_metabolic_power_W
        )
    )(plan.parameters.aerobic_factor)
    assert jnp.isfinite(derivative)


@pytest.mark.parametrize(
    ("time", "power", "message"),
    (
        ((0.0, np.nan), ((1.0,), (1.0,)), "finite and strictly increasing"),
        ((0.0, 0.0), ((1.0,), (1.0,)), "finite and strictly increasing"),
        ((0.0, 1.0), ((1.0,), (np.inf,)), "finite and non-negative"),
        ((0.0, 1.0), ((1.0,), (-1.0,)), "finite and non-negative"),
    ),
)
def test_energy_integral_rejects_invalid_physical_inputs(time, power, message):
    compiled_integral = eqx.filter_jit(integrate_metabolic_energy_joule)
    with pytest.raises((ValueError, RuntimeError), match=message):
        invalid = compiled_integral(jnp.asarray(time), jnp.asarray(power))
        jax.block_until_ready(invalid)


def test_branch_smooth_rejects_every_piecewise_surface():
    plan = _scalar_plan()

    def evaluate(
        *,
        excitation=0.8,
        activation=0.7,
        force=10.0,
        force_length=0.9,
        length=0.095,
        velocity=-0.01,
        selected_plan=plan,
    ):
        return selected_plan.evaluate(
            jnp.asarray((excitation,)),
            jnp.asarray((activation,)),
            jnp.asarray((force,)),
            jnp.asarray((force_length,)),
            jnp.asarray((length,)),
            jnp.asarray((velocity,)),
        )

    baseline = evaluate()
    assert bool(baseline.evidence.branch_smooth)

    on_activity_switch = evaluate(excitation=0.7, activation=0.7)
    on_length_switch = evaluate(length=0.1)
    on_velocity_switch = evaluate(velocity=0.0)
    on_slow_rate_cap = evaluate(velocity=-0.4)
    on_recruitment_boundary = evaluate(excitation=0.0, activation=0.4)
    on_force_clamp = evaluate(force=0.0)

    unloaded = evaluate(force=0.0, velocity=1.0)
    raw_boundary_force = (
        unloaded.activation_maintenance_heat_W_per_kg[0]
        + unloaded.shortening_lengthening_heat_W_per_kg[0]
    )
    on_negative_power_boundary = evaluate(
        force=raw_boundary_force,
        velocity=1.0,
    )

    heat_before_floor = (
        baseline.activation_maintenance_heat_W_per_kg[0]
        + baseline.shortening_lengthening_heat_W_per_kg[0]
    )
    floor_plan = _scalar_plan(minimum_heat_rate=heat_before_floor)
    on_heat_floor = evaluate(selected_plan=floor_plan)

    surfaces = (
        on_activity_switch,
        on_length_switch,
        on_velocity_switch,
        on_slow_rate_cap,
        on_recruitment_boundary,
        on_force_clamp,
        on_negative_power_boundary,
        on_heat_floor,
    )
    assert all(not bool(result.evidence.branch_smooth) for result in surfaces)


def test_invalid_physical_input_fails_evidence_without_fabricated_success():
    plan = _plan()
    result = plan.evaluate(
        jnp.asarray((1.2, 0.5)),
        jnp.asarray((0.5, 0.5)),
        jnp.asarray((10.0, 10.0)),
        jnp.ones(2),
        jnp.asarray((0.1, 0.12)),
        jnp.zeros(2),
    )
    assert not bool(result.evidence.successful)
    assert not bool(result.evidence.inputs_admissible)
