#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._trainable import partition_trainable
from phydrax.applications.skeletal_muscle.cellular import (
    ShortenCellState,
    ShortenFastTwitchModel,
    ShortenIntegrationPlan,
    ShortenPulseProtocol,
)


_OPEN_COR_FAST_TRACE = {
    "vS": np.array(
        [
            -79.974,
            -70.56984893234592,
            -65.70913559330221,
            -60.78146659943978,
            -50.85026953450061,
            -16.708012321459826,
            30.95461429513052,
            34.243156102644186,
            29.068672923369597,
            23.57911324817365,
            18.850029745397745,
        ]
    ),
    "vT": np.array(
        [
            -80.2,
            -79.61131460404869,
            -78.31889340369032,
            -76.6957615003432,
            -74.50525447267584,
            -70.24867907985126,
            -60.522923881167905,
            -48.54234923766906,
            -34.540547388150806,
            -14.372310683780894,
            6.195523220276491,
        ]
    ),
    "Ca_1": np.array(
        [
            0.1,
            0.2429379353792191,
            0.2955065541906228,
            0.3539320121406062,
            0.42105582260196767,
            0.5025792472471213,
            0.6177056990291135,
            0.8183843821635061,
            1.2168809621401484,
            2.1980420224751724,
            5.254755884007505,
        ]
    ),
    "Ca_2": np.array(
        [
            0.1,
            0.15634130294752863,
            0.1899304617885617,
            0.21645863304654478,
            0.23781586122958662,
            0.25558566826599305,
            0.2713194202402873,
            0.28723076564259564,
            0.3072660627068752,
            0.341045508490308,
            0.4218805896020381,
        ]
    ),
    "A_2": np.array(
        [
            0.23,
            0.23027145724221798,
            0.23051068011035883,
            0.23072998972707892,
            0.23093147657653515,
            0.2311153703402217,
            0.23128160565639028,
            0.2314301159009796,
            0.23156089727517648,
            0.23167404936823247,
            0.23176986440800512,
        ]
    ),
}


def test_source_identity_layouts_and_state_count_resolution():
    model = ShortenFastTwitchModel()

    assert model.source_revision == "637da9ef28f7992e40fe79947364a51a38ec818c"
    assert model.source_sha256 == (
        "e14e2aeffeb7b935017414a5ef53c06e43ed6b5fd4d7a92f07e0518b48b413c1"
    )
    assert model.source_license == (
        "Creative Commons Attribution 3.0 Unported (CC BY 3.0)"
    )
    assert model.state_layout.count == 56
    assert model.parameter_layout.count == 99
    assert model.constant_layout.count == 105
    assert model.algebraic_layout.count == 71
    assert model.state_layout.index("P_C_SR") == 55
    assert model.algebraic_layout.index("I_HH") == 32
    assert model.constant_layout.index("V_SR2") == 104
    for layout in (
        model.state_layout,
        model.parameter_layout,
        model.constant_layout,
        model.algebraic_layout,
    ):
        assert len(layout.names) == len(layout.units) == len(layout.source_symbols)
        assert len(set(layout.names)) == layout.count
        assert all(symbol.count("/") == 1 for symbol in layout.source_symbols)


def test_source_initial_rhs_and_current_reference_values():
    model = ShortenFastTwitchModel()
    state = model.initialize(dtype=jnp.float64)
    evaluation = model.evaluate(0.0, state)
    rates = evaluation.state_rate_per_ms

    expected_rates = {
        "vS": 142.22126573539276,
        "vT": -1.1303020276710498,
        "n": 1.4034769892546626e-7,
        "h_K": -4.522340661201656e-9,
        "m": 4.100762903031739e-4,
        "h": 7.162937553712532e-6,
        "S": 9.810020857262693e-8,
        "n_t": -5.510692327142293e-5,
        "h_K_t": 2.5415541998198326e-5,
        "m_t": -9.13400068151359e-3,
        "h_t": 1.7540295074031881e-3,
        "S_t": 3.76951668345105e-7,
        "A_1": -0.030599999999999995,
        "A_2": 0.0029999999999999953,
        "P": 2.4005899999999997e-5,
    }
    for name, expected in expected_rates.items():
        np.testing.assert_allclose(
            rates[model.state_layout.index(name)], expected, rtol=2e-11, atol=2e-11
        )

    expected_algebraics = {
        "I_T": 1.5066666666666606,
        "I_HH": 150.0,
        "I_Cl": 4.1054172489320715,
        "I_IR": 1.1083423178755314,
        "I_DR": 3.9299144639919512e-7,
        "I_Na": -1.5742434721521985,
        "I_NaK": 2.6325511102937238,
        "I_ionic_s": -143.72793240205942,
        "I_Cl_t": 0.3608557572244595,
        "I_IR_t": 0.9789025374472027,
        "I_DR_t": 1.551916436039707e-7,
        "I_Na_t": -0.15781307233069097,
        "I_NaK_t": 0.26224553902732267,
        "I_ionic_t": 1.4441909165599374,
        "T_0": 106.47,
        "k_C": 0.0187829203289675,
        "k_Cm": 0.5323985740693232,
    }
    for name, expected in expected_algebraics.items():
        np.testing.assert_allclose(
            evaluation.algebraic_value(name), expected, rtol=2e-11, atol=2e-11
        )


def test_opencor_source_trajectory_current_calcium_and_tension_agreement():
    model = ShortenFastTwitchModel()
    grid = np.linspace(0.0, 1.0, 11)
    trajectory = ShortenIntegrationPlan(
        model,
        grid,
        relative_tolerance=2.0e-8,
        absolute_tolerance=2.0e-10,
    ).prepare().integrate()

    assert bool(jnp.all(trajectory.successful))
    for name, reference in _OPEN_COR_FAST_TRACE.items():
        observed = trajectory.states[:, model.state_layout.index(name)]
        np.testing.assert_allclose(observed, reference, rtol=2e-4, atol=2e-5)

    final = model.evaluate(grid[-1], trajectory.states[-1])
    assert final.tension_driver_uM == trajectory.states[
        -1, model.state_layout.index("A_2")
    ]
    assert final.force_bearing_crossbridge_uM == final.tension_driver_uM
    assert final.cytosolic_calcium_uM.shape == (2,)


def test_source_pulse_alignment_and_endpoint_convention():
    protocol = ShortenPulseProtocol()
    times = jnp.asarray([-1.0e-6, 0.0, 0.499999, 0.5, 49.999, 50.0, 50.5, 400.0, 400.5])
    expected = jnp.asarray([0.0, 150.0, 150.0, 0.0, 0.0, 150.0, 0.0, 150.0, 0.0])
    np.testing.assert_array_equal(protocol.current(times), expected)
    np.testing.assert_allclose(
        protocol.event_times_ms(),
        np.stack((np.arange(9) * 50.0, np.arange(9) * 50.0 + 0.5), axis=-1).reshape(-1),
    )

    with pytest.raises(ValueError, match="pin every stimulus"):
        ShortenIntegrationPlan(ShortenFastTwitchModel(), [0.0, 1.0])
    ShortenIntegrationPlan(ShortenFastTwitchModel(), [0.0, 0.5, 1.0])


def test_exact_gates_semigroup_and_stiffness_evidence():
    model = ShortenFastTwitchModel()
    state = model.initialize()
    full = model.exact_gate_update(0.75, state, 0.02)
    half = model.exact_gate_update(0.75, state, 0.01)
    refined = model.exact_gate_update(0.75, half, 0.01)
    np.testing.assert_allclose(full[8:18], refined[8:18], rtol=2e-6, atol=2e-8)
    np.testing.assert_array_equal(full[:8], state[:8])
    np.testing.assert_array_equal(full[18:], state[18:])

    kinetics = model.evaluate(0.75, state, stimulus_current_uA_per_cm2=0.0)
    fastest_source_time_ms = 1.0 / model.parameters[
        model.parameter_layout.index("k_Lm")
    ]
    stiffness_ratio = (
        jnp.max(kinetics.gate_time_constant_ms) / fastest_source_time_ms
    )
    assert float(stiffness_ratio) > 1.0e6


def test_failed_step_trajectory_pairs_rolled_back_time_and_values():
    prepared = ShortenIntegrationPlan(
        ShortenFastTwitchModel(), [0.0, 0.5]
    ).prepare()
    initial = prepared.initialize()
    misaligned = ShortenCellState(0.1, initial.values)
    trajectory = prepared.integrate(misaligned)

    assert not bool(trajectory.successful[0])
    np.testing.assert_array_equal(trajectory.times_ms, [0.1, 0.1])
    np.testing.assert_array_equal(
        trajectory.states,
        np.stack((misaligned.values, misaligned.values)),
    )


def test_integration_schedule_is_fixed_and_identity_is_content_complete():
    model = ShortenFastTwitchModel()
    plan = ShortenIntegrationPlan(model, [0.0, 0.5, 1.0])
    trainable, fixed = partition_trainable(plan)

    trainable_leaves = jax.tree.leaves(trainable)
    assert len(trainable_leaves) == 1
    np.testing.assert_array_equal(trainable_leaves[0], model.parameters)
    assert trainable.schedule is None
    assert fixed.schedule is plan.schedule

    protocol_trainable, protocol_fixed = partition_trainable(plan.protocol)
    assert not jax.tree.leaves(protocol_trainable)
    assert protocol_fixed is plan.protocol

    same = ShortenIntegrationPlan(model, [0.0, 0.5, 1.0])
    different_grid = ShortenIntegrationPlan(model, [0.0, 0.5, 1.5])
    different_timing = ShortenIntegrationPlan(
        model,
        [0.0, 0.5, 1.0],
        protocol=ShortenPulseProtocol(period_ms=60.0),
    )
    different_amplitude = ShortenIntegrationPlan(
        model,
        [0.0, 0.5, 1.0],
        protocol=ShortenPulseProtocol(amplitude_uA_per_cm2=125.0),
    )
    assert same.plan_id == plan.plan_id
    assert different_grid.plan_id != plan.plan_id
    assert different_timing.plan_id != plan.plan_id
    assert different_amplitude.plan_id != plan.plan_id


def test_rhs_is_jittable_vectorized_and_forward_differentiable():
    model = ShortenFastTwitchModel()
    state = model.initialize(dtype=jnp.float64)
    compiled = eqx.filter_jit(lambda configured, value: configured.rhs(0.75, value))
    observed = compiled(model, state)
    assert observed.shape == (56,)
    assert bool(jnp.all(jnp.isfinite(observed)))

    batch = jnp.stack((state, state.at[0].add(0.01), state.at[1].add(-0.01)))
    vectorized = jax.vmap(lambda value: model.rhs(0.75, value))(batch)
    assert vectorized.shape == (3, 56)

    direction = jnp.linspace(-1.0e-5, 1.0e-5, 56, dtype=state.dtype)
    primal, tangent = jax.jvp(
        lambda value: model.rhs(0.75, value), (state,), (direction,)
    )
    assert primal.shape == tangent.shape == (56,)
    assert bool(jnp.all(jnp.isfinite(tangent)))
    removable_gate_singularities = (
        state.at[model.state_layout.index("vS")].set(-46.0)
        .at[model.state_layout.index("vT")]
        .set(-40.0)
    )
    _, singular_tangent = jax.jvp(
        lambda value: model.rhs(0.75, value),
        (removable_gate_singularities,),
        (direction,),
    )
    zero_voltage = (
        state.at[model.state_layout.index("vS")].set(0.0)
        .at[model.state_layout.index("vT")]
        .set(0.0)
    )
    zero_voltage_rhs = model.rhs(0.75, zero_voltage)
    assert bool(jnp.all(jnp.isfinite(singular_tangent)))
    assert bool(jnp.all(jnp.isfinite(zero_voltage_rhs)))


    parameter_direction = jnp.zeros_like(model.parameters).at[39].set(1.0e-4)
    tangent_model = eqx.tree_at(
        lambda configured: configured.parameters, model, parameter_direction
    )
    _, parameter_tangent = jax.jvp(
        lambda configured: configured.rhs(0.75, state),
        (model,),
        (tangent_model,),
    )
    assert bool(jnp.all(jnp.isfinite(parameter_tangent)))
    assert float(jnp.linalg.norm(parameter_tangent)) > 0.0


def test_batched_layout_and_admissibility_contract():
    model = ShortenFastTwitchModel()
    state = model.initialize((4,))
    evaluation = model.evaluate(jnp.arange(4) * 0.1, state)
    assert evaluation.state_rate_per_ms.shape == (4, 56)
    assert evaluation.algebraic.shape == (4, 71)
    assert evaluation.sarcolemmal_current_uA_per_cm2.shape == (4, 5)
    assert bool(jnp.all(evaluation.valid))

    invalid = state.at[2, model.state_layout.index("K_i")].set(0.0)
    validity = model.admissible(invalid)
    np.testing.assert_array_equal(validity, [True, True, False, True])
