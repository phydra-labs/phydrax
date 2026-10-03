#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Native method owners bound as transactional partitioned-coupling participants."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.solver._balance_law_composition import AdditiveIMEXTableau


cpl = phx.solver.coupling
_KEY_IMPL = "threefry2x32"


def _space(name: str, /) -> phx.linalg.ArraySpace:
    return phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id=name)


def _port(
    port_id: str,
    direction: phx.solver.coupling.CouplingDirection,
    space: phx.linalg.AbstractVectorSpace,
    /,
    *,
    waveform_plan: phx.solver.coupling.CouplingWaveformPlan | None = None,
) -> phx.solver.coupling.CouplingPort:
    return cpl.CouplingPort(
        port_id, direction, space, waveform_plan=waveform_plan, reference_scale=1.0
    )


def _decay(time: Array, state: Array, forcing: Array) -> Array:
    del time
    return forcing - 2.0 * state


def _forced_decay_participant(
    substeps: int,
    /,
    *,
    subsystem_id: str = "decay",
    randomness: phx.solver.coupling.MethodParticipantRandomness = "none",
    noise: float = 0.0,
) -> phx.solver.coupling.FixedStepCouplingParticipant:
    """SSPRK(3,3) owner of y' = u - 2y with a frozen window input u."""
    space = _space(f"{subsystem_id}-scalar")
    plan = cpl.CouplingWaveformPlan(
        substeps + 1,
        1,
        tuple(np.linspace(0.0, 1.0, substeps + 1)),
        plan_id=f"{subsystem_id}-native-nodes",
    )

    def bind(
        window: phx.solver.coupling.CouplingWindow,
        views: tuple[Any, ...],
        model_state: Array,
        key: Array | None,
        args: None,
    ) -> phx.solver.coupling.MethodWindowBinding:
        del window, args
        forcing = views[0]
        if key is not None:
            forcing = forcing + noise * jax.random.normal(key, (1,), dtype=jnp.float64)
        return cpl.MethodWindowBinding(forcing, model_state + 1)

    return cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(_decay),
        bind,
        lambda state, args: (state, state),
        subsystem_id=subsystem_id,
        substeps=substeps,
        input_ports=(_port(f"{subsystem_id}/u", "input", space),),
        output_ports=(
            _port(f"{subsystem_id}/y", "output", space),
            _port(f"{subsystem_id}/y-nodes", "output", space, waveform_plan=plan),
        ),
        randomness=randomness,
        key_impl=_KEY_IMPL if randomness == "carried-key" else None,
    )


def _explicit() -> phx.solver.coupling.ExplicitCouplingPolicy:
    return cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi"))


def test_fixed_step_participant_runs_native_substeps_at_the_owner_order() -> None:
    errors = []
    for substeps in (4, 8):
        participant = _forced_decay_participant(substeps)
        state = participant.initial_state(jnp.zeros(1), model_state=jnp.asarray(0))
        result = participant.advance_window(
            cpl.CouplingWindow(0, 0.0, 1.0), state, (jnp.ones(1),), None
        )
        exact = 0.5 * (1.0 - np.exp(-2.0))
        errors.append(abs(float(result.outputs[0][0]) - exact))
        nodes = result.outputs[1].values[:, 0]
        assert bool(result.successful)
        assert int(result.work) == 3 * substeps
        assert int(result.iterations) == substeps
        assert int(result.candidate_state.model_state) == substeps
        assert float(nodes[0]) == 0.0
        assert float(nodes[-1]) == float(result.outputs[0][0])
        np.testing.assert_allclose(
            nodes, 0.5 * (1.0 - np.exp(-2.0 * np.linspace(0, 1, substeps + 1))), atol=4e-3
        )
    # SSPRK(3,3) is third order: halving the native step divides the error by 8,
    # approached from above on these pre-asymptotic steps (h = 1/4, 1/8).
    assert 7.0 < errors[0] / errors[1] < 11.0


def test_conservation_imex_participant_spends_a_window_amount_at_uniform_rate() -> None:
    tableau = AdditiveIMEXTableau(
        jnp.asarray(((0.0,),)),
        jnp.asarray(((1.0,),)),
        jnp.asarray((1.0,)),
        jnp.asarray((1.0,)),
    )

    def implicit_solver(
        provisional: Array, time: Array, coefficient: Array, args: Array
    ) -> phx.solver.ImplicitConservationStageResult:
        del time, args
        return phx.solver.ImplicitConservationStageResult(
            provisional / (1.0 + 3.0 * coefficient),
            jnp.asarray(True),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(0.0),
        )

    method = phx.solver.ConservationIMEXMethod(
        tableau,
        lambda time, state, rate: rate,
        lambda time, state, rate: -3.0 * state,
        implicit_solver,
        method_id="imex-euler-decay",
    )
    field = phx.discretization.DiscreteFieldSpace(
        "reservoir",
        "reservoir-support",
        phx.discretization.EntityDofLayout("reservoir/cells", 1, 1),
        _space("reservoir-values"),
        representation="cell_integral",
    )
    amount_port = cpl.CouplingPort(
        "reservoir/amount",
        "input",
        field.vector_space,
        field_space=field,
        quantity=cpl.CouplingQuantity("mass", phx.units.KILOGRAM),
        measurement=cpl.CouplingMeasurement.extensive(
            field.vector_space, field.support_id, provenance_id="reservoir-cell"
        ),
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )
    participant = cpl.FixedStepCouplingParticipant(
        method,
        lambda window, views, model, key, args: cpl.MethodWindowBinding(
            views[0] / window.size, model
        ),
        lambda state, args: (),
        subsystem_id="reservoir",
        substeps=5,
        input_ports=(amount_port,),
    )
    state = participant.initial_state(jnp.asarray((1.0,)))

    result = participant.advance_window(
        cpl.CouplingWindow(0, 0.0, 0.5), state, (jnp.asarray((2.0,)),), None
    )

    # IMEX Euler, the amount spent at rate 2 / 0.5 over five 0.1 substeps: the
    # implicit stage solves s = y / (1 + 0.3) and y <- y + 0.1 * (4 - 3 s).
    reference = 1.0
    for _ in range(5):
        stage = reference / (1.0 + 0.3)
        reference = reference + 0.1 * (4.0 - 3.0 * stage)
    assert bool(result.successful)
    np.testing.assert_allclose(result.candidate_state.native, [reference], rtol=1e-14)
    assert int(result.work) == 5


def _implicit_cycle(
    participant: phx.solver.coupling.FixedStepCouplingParticipant,
) -> tuple[phx.solver.coupling.CouplingGraph, phx.solver.coupling.ImplicitCouplingPolicy]:
    space = participant.input_ports[0].space

    def half(
        window: phx.solver.coupling.CouplingWindow,
        state: Array,
        inputs: tuple[Array, ...],
        args: None,
    ) -> phx.solver.coupling.CouplingSubsystemResult:
        del window, args
        return cpl.CouplingSubsystemResult(
            state, (0.5 * inputs[0] + 1.0,), successful=True, status=0
        )

    partner = cpl.CallableCouplingSubsystem(
        half,
        subsystem_id="partner",
        input_ports=(_port("partner/in", "input", space),),
        output_ports=(_port("partner/out", "output", space),),
        capabilities=cpl.CouplingSubsystemCapabilities(
            jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
        ),
    )
    decay_output = participant.output_ports[0].port_id
    graph = cpl.CouplingGraph(
        (participant, partner),
        (
            cpl.CouplingExchange("to-partner", decay_output, "partner/in"),
            cpl.CouplingExchange(
                "to-decay", "partner/out", participant.input_ports[0].port_id
            ),
        ),
    )
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-10, relative_residual=0.0, maximum_steps=60
        ),
        (
            cpl.CouplingTolerance("partner/in", absolute=1e-8),
            cpl.CouplingTolerance(participant.input_ports[0].port_id, absolute=1e-8),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )
    return graph, policy


def test_implicit_iterates_replay_the_checkpoint_and_advance_it_once() -> None:
    participant = _forced_decay_participant(
        3, subsystem_id="noisy", randomness="carried-key", noise=0.05
    )
    key = jax.random.key(11)
    state = participant.initial_state(jnp.zeros(1), model_state=jnp.asarray(0), key=key)
    graph, policy = _implicit_cycle(participant)
    prepared = cpl.prepare_coupling(
        graph, (state, jnp.zeros(1)), (jnp.zeros(1), jnp.zeros(1)), policy=policy
    )
    index = prepared.reference_state.subsystem_ids.index("noisy")

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 0.25)

    accepted = result.accepted_state.participant_states[index]
    assert bool(result.successful)
    assert int(result.diagnostics.participant_evaluations[index]) > 2
    assert int(accepted.accepted_windows) == 1
    assert int(accepted.model_state) == 3
    np.testing.assert_array_equal(
        accepted.key_data, jax.random.key_data(jax.random.split(key)[1])
    )
    replay = cpl.advance_coupling_window(prepared, prepared.reference_state, 0.25)
    assert bool(eqx.tree_equal(replay.accepted_state, result.accepted_state))


def test_implicit_fixed_point_window_counts_the_work_of_every_iterate() -> None:
    participant = _forced_decay_participant(3)
    graph, policy = _implicit_cycle(participant)
    prepared = cpl.prepare_coupling(
        graph,
        (
            participant.initial_state(jnp.zeros(1), model_state=jnp.asarray(0)),
            jnp.zeros(1),
        ),
        (jnp.zeros(1), jnp.zeros(1)),
        policy=policy,
    )
    index = prepared.reference_state.subsystem_ids.index("decay")

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 0.25)

    diagnostics = result.diagnostics
    evaluations = int(diagnostics.nonlinear_residual_evaluations) + 1
    assert bool(result.successful)
    assert evaluations > 2
    assert int(diagnostics.participant_evaluations[index]) == evaluations
    # Every interface iterate and the final re-evaluation run three SSPRK(3,3)
    # substeps of work three; superseded iterates keep their spent work.
    assert int(diagnostics.participant_work[index]) == 9 * evaluations
    assert diagnostics.counts_complete


def test_gauss_seidel_fixed_point_window_balances_its_spent_window_amount() -> None:
    field = phx.discretization.DiscreteFieldSpace(
        "tank",
        "tank-support",
        phx.discretization.EntityDofLayout("tank/cells", 1, 1),
        _space("tank-values"),
        representation="cell_integral",
    )

    def amount_port(port_id: str, direction: Any) -> phx.solver.coupling.CouplingPort:
        return cpl.CouplingPort(
            port_id,
            direction,
            field.vector_space,
            field_space=field,
            quantity=cpl.CouplingQuantity("mass", phx.units.KILOGRAM),
            measurement=cpl.CouplingMeasurement.extensive(
                field.vector_space, field.support_id, provenance_id="tank-cell"
            ),
            temporal_kind="interval_integral",
            reference_scale=1.0,
        )

    level_space = _space("tank-level")
    capabilities = cpl.CouplingSubsystemCapabilities(
        jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
    )
    # The source spends a level-dependent amount over the window; the tank
    # accumulates exactly the amount it receives and reports its new level.
    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (window.size * (0.5 * inputs[0] + 1.0),), successful=True, status=0
        ),
        subsystem_id="source",
        input_ports=(_port("source/level", "input", level_space),),
        output_ports=(amount_port("source/amount", "output"),),
        capabilities=capabilities,
    )
    tank = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state + inputs[0], (state + inputs[0],), successful=True, status=0
        ),
        subsystem_id="tank",
        input_ports=(amount_port("tank/amount", "input"),),
        output_ports=(_port("tank/level", "output", level_space),),
        capabilities=capabilities,
    )
    graph = cpl.CouplingGraph(
        (source, tank),
        (
            cpl.CouplingExchange(
                "spent",
                "source/amount",
                "tank/amount",
                temporal=cpl.CouplingTemporalConversion("window-integral"),
            ),
            cpl.CouplingExchange("level", "tank/level", "source/level"),
        ),
    )
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=1e-10,
            relative_residual=0.0,
            absolute_step=0.0,
            relative_step=0.0,
            maximum_steps=80,
        ),
        (
            cpl.CouplingTolerance("tank/amount", absolute=1e-8),
            cpl.CouplingTolerance("source/level", absolute=1e-8),
        ),
        fixed_point_sweep=cpl.CouplingSweep(
            "gauss-seidel", subsystem_order=("source", "tank")
        ),
    )
    prepared = cpl.prepare_coupling(
        graph, (jnp.zeros(1), jnp.ones(1)), (jnp.zeros(1), jnp.ones(1)), policy=policy
    )
    row = prepared.reference_state.budget_row_ids.index("spent")
    index = prepared.reference_state.subsystem_ids.index("tank")

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    # The tank consumes, in the same sweep, the amount the source just spent.
    debit, credit = np.asarray(result.accepted_exchange_budget[row])
    level = np.asarray(result.accepted_state.participant_states[index])
    assert int(result.status) == int(cpl.CouplingStatus.SUCCESS)
    assert bool(result.successful)
    assert debit == -credit
    np.testing.assert_allclose(credit, level[0] - 1.0, rtol=1e-14)
    np.testing.assert_allclose(level, [4.0], rtol=1e-9)


def test_external_randomness_is_refused_by_replaying_routes() -> None:
    participant = _forced_decay_participant(2, randomness="external")
    state = participant.initial_state(jnp.zeros(1), model_state=jnp.asarray(0))
    graph, policy = _implicit_cycle(participant)

    assert not participant.capabilities.deterministic_replay
    with pytest.raises(ValueError, match="deterministic replay"):
        cpl.prepare_coupling(
            graph, (state, jnp.zeros(1)), (jnp.zeros(1), jnp.zeros(1)), policy=policy
        )
    explicit = cpl.prepare_coupling(
        graph, (state, jnp.zeros(1)), (jnp.zeros(1), jnp.zeros(1)), policy=_explicit()
    )
    plan = cpl.AdaptiveCouplingRolloutPlan(
        4,
        phx.solver.FixedCapacitySegmentPolicy(4, 4),
        cpl.AdaptiveCouplingWindowPolicy(
            0.25,
            0.05,
            0.5,
            absolute_tolerance=1.0,
            relative_tolerance=0.0,
            maximum_attempts=4,
        ),
        phx.solver.HybridReplayPolicy(0),
    )
    with pytest.raises(ValueError, match="deterministic replay"):
        cpl.rollout_adaptive_coupling(explicit, explicit.reference_state, 1.0, plan)


def _dae_participant(
    *, adaptive: bool = True
) -> phx.solver.coupling.DAECouplingParticipant:
    system = phx.dynamics.DifferentialAlgebraicSystem(
        lambda time, state, rate, rate_constant: rate + rate_constant * state,
        state_shape=(1,),
        structure=phx.dynamics.DAEStructure(("differential",)),
        system_id="coupled-decay-dae",
    )
    problem = phx.solver.DifferentialAlgebraicProblem(
        system, jnp.asarray((1.0,)), args=jnp.asarray(1.0), problem_id="coupled-decay"
    )
    policy = phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        adaptive=(
            phx.solver.DAEAdaptivePolicy(
                relative_tolerance=1e-7,
                absolute_tolerance=1e-10,
                maximum_accepted_steps=256,
                maximum_attempts=512,
            )
            if adaptive
            else None
        ),
        failure="status",
    )
    template = phx.dynamics.TimeGrid(jnp.asarray((0.0, 1.0)), time_id="window-template")
    space = _space("dae-scalar")
    return cpl.DAECouplingParticipant(
        phx.solver.prepare_dae(problem, template, policy=policy),
        lambda window, views, model, key, args: cpl.MethodWindowBinding(
            views[0][0], model
        ),
        lambda state, rate, args: (state,),
        subsystem_id="dae",
        input_ports=(_port("dae/rate-constant", "input", space),),
        output_ports=(_port("dae/state", "output", space),),
    )


def test_dae_participant_resumes_its_exact_continuation_across_windows() -> None:
    participant = _dae_participant()
    state = participant.initial_state()
    rate_constant = (jnp.asarray((1.0,)),)
    step = eqx.filter_jit(participant.advance_window)

    first = step(cpl.CouplingWindow(0, 0.0, 0.1), state, rate_constant, None)
    second = step(
        cpl.CouplingWindow(1, 0.1, 0.2), first.candidate_state, rate_constant, None
    )

    assert bool(first.successful) and bool(second.successful)
    assert bool(second.candidate_state.native.started)
    assert int(second.candidate_state.accepted_windows) == 2
    assert int(second.work) > 0 and int(second.iterations) > 0
    np.testing.assert_allclose(first.outputs[0], [np.exp(-0.1)], rtol=1e-5)
    np.testing.assert_allclose(second.outputs[0], [np.exp(-0.2)], rtol=1e-5)
    with pytest.raises(ValueError, match="rollback would otherwise be incomplete"):
        _dae_participant(adaptive=False)


def test_steady_response_is_instantaneous_and_refuses_window_amounts() -> None:
    space = _space("steady-scalar")

    def respond(
        inputs: tuple[Array, ...], model_state: None, key: None, args: None
    ) -> phx.solver.coupling.SteadyResponse:
        del key, args
        return cpl.SteadyResponse(
            (2.0 * inputs[0],),
            model_state,
            jnp.asarray(True),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
        )

    steady = cpl.SteadyResponseCouplingParticipant(
        respond,
        subsystem_id="steady",
        input_ports=(_port("steady/in", "input", space),),
        output_ports=(_port("steady/out", "output", space),),
        initial_response=(jnp.zeros(1),),
    )
    result = steady.advance_window(
        cpl.CouplingWindow(0, 0.0, 5.0), steady.initial_state(), (jnp.ones(1) * 3,), None
    )

    np.testing.assert_allclose(result.outputs[0], [6.0])
    assert float(result.error_estimate.error_norm) == 0.0
    assert int(result.candidate_state.accepted_windows) == 1
    field = phx.discretization.DiscreteFieldSpace(
        "steady-amount",
        "steady-support",
        phx.discretization.EntityDofLayout("steady/cells", 1, 1),
        _space("steady-amount-values"),
        representation="cell_integral",
    )
    amount = cpl.CouplingPort(
        "steady/amount",
        "output",
        field.vector_space,
        field_space=field,
        quantity=cpl.CouplingQuantity("mass", phx.units.KILOGRAM),
        measurement=cpl.CouplingMeasurement.extensive(
            field.vector_space, field.support_id, provenance_id="steady-cell"
        ),
        temporal_kind="interval_integral",
        reference_scale=1.0,
    )
    with pytest.raises(ValueError, match="no temporal accumulation"):
        cpl.SteadyResponseCouplingParticipant(
            respond,
            subsystem_id="steady-amount",
            input_ports=(_port("steady-amount/in", "input", space),),
            output_ports=(amount,),
            initial_response=(jnp.zeros(1),),
        )


def _scheduled_counter(time: Array, state: Array, size: Array, index: Array) -> Any:
    """Native step that, like production schedules, refuses a foreign step index."""
    del time, size
    matches = index == state[0].astype(jnp.int32)
    advanced = jnp.where(matches, state + 1.0, state)
    return phx.solver.FixedStepResult(
        advanced,
        advanced,
        matches,
        jnp.zeros((), dtype=state.dtype),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.zeros((), dtype=state.dtype),
    )


def test_fixed_step_participant_continues_the_native_step_index_across_windows() -> None:
    space = _space("counter-scalar")
    participant = cpl.FixedStepCouplingParticipant(
        phx.solver.CallableFixedStepMethod(
            lambda index, time, state, size, args: _scheduled_counter(
                time, state, size, index
            ),
            "scheduled-counter",
        ),
        lambda window, views, model, key, args: cpl.MethodWindowBinding(None, model),
        lambda state, args: (state,),
        subsystem_id="counter",
        substeps=2,
        output_ports=(_port("counter/steps", "output", space),),
    )
    # The native continuation has already taken three owner steps.
    accepted = participant.initial_state(jnp.asarray((3.0,)), step_index=3)

    for window in range(3):
        bounds = cpl.CouplingWindow(window, float(window), window + 1.0)
        first = participant.advance_window(bounds, accepted, (), None)
        replay = participant.advance_window(bounds, accepted, (), None)
        assert bool(first.successful)
        assert bool(eqx.tree_equal(first, replay))
        accepted = first.candidate_state

    np.testing.assert_array_equal(accepted.native, [9.0])
    assert int(accepted.native_steps) == 9
    assert int(accepted.accepted_windows) == 3
    stale = participant.initial_state(jnp.asarray((3.0,)))
    rejected = participant.advance_window(
        cpl.CouplingWindow(0, 0.0, 1.0), stale, (), None
    )
    assert not bool(rejected.successful)
    with pytest.raises(ValueError, match="integer scalar"):
        participant.initial_state(jnp.asarray((3.0,)), step_index=1.5)


def _doubling_response(
    inputs: tuple[Array, ...], model_state: Array, key: None, args: None
) -> phx.solver.coupling.SteadyResponse:
    del key, args
    return cpl.SteadyResponse(
        (2.0 * inputs[0],),
        model_state + 1,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(1, dtype=jnp.int32),
    )


def test_steady_waveform_response_answers_at_every_shared_plan_node() -> None:
    space = _space("steady-waveform-scalar")
    plan = cpl.CouplingWaveformPlan(3, 1, (0.0, 0.5, 1.0), plan_id="steady-nodes")
    steady = cpl.SteadyResponseCouplingParticipant(
        _doubling_response,
        subsystem_id="steady-waveform",
        input_ports=(_port("steady-waveform/in", "input", space, waveform_plan=plan),),
        output_ports=(_port("steady-waveform/out", "output", space, waveform_plan=plan),),
        initial_response=(jnp.zeros(1),),
    )
    signal = cpl.CouplingWaveform(
        plan.initial_grid(), jnp.asarray([[1.0], [4.0], [-2.0]]), space
    )

    result = steady.advance_window(
        cpl.CouplingWindow(0, 0.0, 1.0),
        steady.initial_state(model_state=jnp.asarray(0)),
        (signal,),
        None,
    )

    np.testing.assert_array_equal(result.outputs[0].values, [[2.0], [8.0], [-4.0]])
    assert int(result.candidate_state.model_state) == 3
    np.testing.assert_array_equal(result.candidate_state.native[0], [-4.0])
    with pytest.raises(ValueError, match="share one waveform plan"):
        cpl.SteadyResponseCouplingParticipant(
            _doubling_response,
            subsystem_id="steady-mixed",
            input_ports=(_port("steady-mixed/in", "input", space, waveform_plan=plan),),
            output_ports=(_port("steady-mixed/out", "output", space),),
            initial_response=(jnp.zeros(1),),
        )


def test_steady_checkpoint_publishes_its_accepted_response_on_restart() -> None:
    space = _space("steady-restart-scalar")
    steady = cpl.SteadyResponseCouplingParticipant(
        _doubling_response,
        subsystem_id="steady-restart",
        input_ports=(_port("steady-restart/in", "input", space),),
        output_ports=(_port("steady-restart/out", "output", space),),
        initial_response=(jnp.asarray((-1.0,)),),
    )
    initial = steady.initial_state(model_state=jnp.asarray(0))

    accepted = steady.advance_window(
        cpl.CouplingWindow(0, 0.0, 1.0), initial, (jnp.asarray((3.0,)),), None
    ).candidate_state

    np.testing.assert_array_equal(steady.initial_outputs(initial, None)[0], [-1.0])
    # A restart from the accepted checkpoint observes the accepted 2 x 3 response.
    np.testing.assert_array_equal(steady.initial_outputs(accepted, None)[0], [6.0])


def test_lowering_derives_initial_exchanges_from_participant_checkpoints() -> None:
    participant = _forced_decay_participant(2, subsystem_id="lowered")
    graph, _ = _implicit_cycle(participant)
    declaration = cpl.PartitionedCouplingDeclaration(
        graph.subsystems, graph.exchanges, _explicit()
    )
    states = {
        "lowered": participant.initial_state(
            jnp.asarray((0.75,)), model_state=jnp.asarray(0)
        ),
        "partner": jnp.zeros(1),
    }

    problem = cpl.lower_partitioned_coupling(
        declaration,
        states,
        t0=0.0,
        t1=1.0,
        window_size=0.5,
        exchange_values={"to-decay": jnp.asarray((1.0,))},
    )
    solution = cpl.solve_coupling(problem)

    values = dict(
        zip(
            (e.exchange_id for e in graph.exchanges), problem.exchange_values, strict=True
        )
    )
    np.testing.assert_array_equal(values["to-partner"], [0.75])
    assert bool(solution.successful)
    assert float(solution.final_state.time) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="cannot be overridden"):
        cpl.lower_partitioned_coupling(
            declaration,
            states,
            t0=0.0,
            t1=1.0,
            window_size=0.5,
            exchange_values={"to-decay": jnp.ones(1), "to-partner": jnp.ones(1)},
        )
    with pytest.raises(ValueError, match="explicit initial values"):
        cpl.lower_partitioned_coupling(
            declaration, states, t0=0.0, t1=1.0, window_size=0.5
        )


def _ramp_fed_rollout(
    feed_estimate: phx.solver.coupling.CouplingWindowErrorEstimate,
) -> tuple[
    phx.solver.coupling.PreparedCoupling, phx.solver.coupling.AdaptiveCouplingRolloutPlan
]:
    """SSPRK(3,3) ramp y' = u fed a constant rate by a feed reporting `feed_estimate`."""
    space = _space("ramp-scalar")

    def bind(
        window: phx.solver.coupling.CouplingWindow,
        views: tuple[Any, ...],
        model_state: None,
        key: None,
        args: None,
    ) -> phx.solver.coupling.MethodWindowBinding:
        del window, key, args
        return cpl.MethodWindowBinding(views[0], model_state)

    ramp = cpl.FixedStepCouplingParticipant(
        phx.solver.SSPRK33FixedStepMethod(lambda time, state, rate: rate),
        bind,
        lambda state, args: (state,),
        subsystem_id="ramp",
        substeps=3,
        input_ports=(_port("ramp/rate", "input", space),),
        output_ports=(_port("ramp/value", "output", space),),
        # A window-size-sensitive local error: the squared increment per window.
        estimate_error=lambda start, end, args: cpl.CouplingWindowErrorEstimate(
            jnp.sum((end - start) ** 2), 1.0, 1, True
        ),
    )
    feed = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state,
            (jnp.ones(1),),
            successful=True,
            status=0,
            error_estimate=feed_estimate,
        ),
        subsystem_id="feed",
        input_ports=(_port("feed/in", "input", space),),
        output_ports=(_port("feed/out", "output", space),),
        capabilities=cpl.CouplingSubsystemCapabilities(
            jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
        ),
    )
    graph = cpl.CouplingGraph(
        (ramp, feed),
        (
            cpl.CouplingExchange("rate", "feed/out", "ramp/rate"),
            cpl.CouplingExchange("value", "ramp/value", "feed/in"),
        ),
    )
    prepared = cpl.prepare_coupling(
        graph,
        (ramp.initial_state(jnp.zeros(1)), jnp.zeros(1)),
        (jnp.ones(1), jnp.zeros(1)),
        policy=cpl.ExplicitCouplingPolicy(
            cpl.CouplingSweep("gauss-seidel", subsystem_order=("feed", "ramp"))
        ),
    )
    plan = cpl.AdaptiveCouplingRolloutPlan(
        16,
        phx.solver.FixedCapacitySegmentPolicy(16, 8),
        cpl.AdaptiveCouplingWindowPolicy(
            0.8,
            0.01,
            0.8,
            absolute_tolerance=0.04,
            relative_tolerance=0.0,
            maximum_attempts=8,
        ),
        phx.solver.HybridReplayPolicy(0),
    )
    return prepared, plan


def test_adaptive_rollout_evidence_includes_rejected_attempt_work() -> None:
    # A constant feed has no temporal truncation error; adaptive acceptance needs
    # a reliable estimate from every participant.
    prepared, plan = _ramp_fed_rollout(cpl.CouplingWindowErrorEstimate(0.0, 1.0, 1, True))

    solution = cpl.rollout_adaptive_coupling(
        prepared, prepared.reference_state, 1.0, plan
    )

    index = prepared.reference_state.subsystem_ids.index("ramp")
    attempts = int(solution.attempted_windows)
    assert bool(solution.successful)
    assert int(solution.rejected_attempts) >= 1
    assert attempts == int(solution.accepted_windows) + int(solution.rejected_attempts)
    # Every executed attempt runs three SSPRK(3,3) substeps of work three.
    assert int(solution.participant_work[index]) == 9 * attempts
    assert int(solution.participant_evaluations[index]) == attempts
    assert float(solution.maximum_rejected_error_ratio) > 1.0
    assert solution.counts_complete
    final = solution.final_state.participant_states[index]
    np.testing.assert_allclose(final.native, [1.0], rtol=1e-12)
    assert int(final.accepted_windows) == int(solution.accepted_windows)


def test_adaptive_rollout_refuses_a_window_without_a_reliable_error_estimate() -> None:
    prepared, plan = _ramp_fed_rollout(
        cpl.CouplingWindowErrorEstimate(0.0, 1.0, 1, False)
    )

    solution = cpl.rollout_adaptive_coupling(
        prepared, prepared.reference_state, 1.0, plan
    )

    # The window itself succeeds; only its acceptance evidence is missing.
    assert int(solution.terminal_status) == int(
        cpl.CouplingStatus.UNRELIABLE_ERROR_ESTIMATE
    )
    assert not bool(solution.successful)
    assert int(solution.accepted_windows) == 0
    assert int(solution.attempted_windows) == 1
    assert int(solution.rejected_attempts) == 1
    assert bool(eqx.tree_equal(solution.final_state, prepared.reference_state))


def _refusing_counter(refused_index: int, /) -> phx.solver.CallableFixedStepMethod:
    """Unit counter that refuses one native step index and reports that index."""

    def step(index: Array, time: Array, state: Array, size: Array, args: None) -> Any:
        del time, size, args
        successful = index != refused_index
        advanced = state + 1.0
        return phx.solver.FixedStepResult(
            advanced,
            jnp.where(successful, advanced, state),
            successful,
            jnp.zeros((), dtype=state.dtype),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.zeros((), dtype=state.dtype),
            evidence=jnp.asarray(index, dtype=jnp.int32),
        )

    return phx.solver.CallableFixedStepMethod(step, f"refusing-counter-{refused_index}")


def test_participant_evidence_survives_substeps_rollback_and_window_rollout() -> None:
    space = _space("evidence-counter")
    participant = cpl.FixedStepCouplingParticipant(
        _refusing_counter(4),
        lambda window, views, model, key, args: cpl.MethodWindowBinding(None, model),
        lambda state, args: (state,),
        subsystem_id="counter",
        substeps=3,
        output_ports=(_port("counter/out", "output", space),),
    )
    accepted = participant.initial_state(jnp.zeros(1))
    first = participant.advance_window(
        cpl.CouplingWindow(0, 0.0, 1.0), accepted, (), None
    )
    evidence = first.evidence
    assert isinstance(evidence, cpl.FixedStepParticipantEvidence)
    np.testing.assert_array_equal(evidence.executed, [True, True, True])
    np.testing.assert_array_equal(evidence.successful, [True, True, True])
    # An owner without a declared reduction keeps the bounded substep stack.
    np.testing.assert_array_equal(evidence.method, [0, 1, 2])

    second = participant.advance_window(
        cpl.CouplingWindow(1, 1.0, 2.0), first.candidate_state, (), None
    )
    assert not bool(second.successful)
    # The refusing substep's evidence survives; no owner work follows it.
    np.testing.assert_array_equal(second.evidence.executed, [True, True, False])
    np.testing.assert_array_equal(second.evidence.successful, [True, False, False])
    np.testing.assert_array_equal(second.evidence.method[:2], [3, 4])

    monitor = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (), successful=True, status=0
        ),
        subsystem_id="monitor",
        input_ports=(_port("monitor/in", "input", space),),
        capabilities=cpl.CouplingSubsystemCapabilities(
            jit=True, differentiable=True, deterministic_replay=True, fixed_topology=True
        ),
    )
    graph = cpl.CouplingGraph(
        (participant, monitor),
        (cpl.CouplingExchange("observe", "counter/out", "monitor/in"),),
    )
    prepared = cpl.prepare_coupling(
        graph, (accepted, jnp.zeros(1)), (jnp.zeros(1),), policy=_explicit()
    )
    index = prepared.reference_state.subsystem_ids.index("counter")
    window = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)
    assert window.participant_evidence[1 - index] is None
    np.testing.assert_array_equal(window.participant_evidence[index].method, [0, 1, 2])

    solution = cpl.CouplingRolloutPlan().rollout(
        prepared, window_count=3, window_size=1.0
    )
    assert not bool(solution.successful)
    record = solution.participant_evidence
    assert record is not None
    assert int(record.accepted_step) == 0
    assert int(record.refused_step) == 1
    np.testing.assert_array_equal(record.accepted[index].method, [0, 1, 2])
    np.testing.assert_array_equal(record.refused[index].successful, [True, False, False])
    # The refused window rolled back: the checkpoint still resumes at step 3.
    final = solution.final_state.participant_states[index]
    np.testing.assert_array_equal(final.native, [3.0])
    assert int(final.native_steps) == 3
