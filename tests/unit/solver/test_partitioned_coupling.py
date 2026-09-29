#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


cpl = phx.solver.coupling


def _capabilities(*, differentiable: Any = True, waveform: Any = False) -> Any:
    return cpl.CouplingSubsystemCapabilities(
        jit=True,
        differentiable=differentiable,
        deterministic_replay=True,
        fixed_topology=True,
        supports_endpoint=not waveform,
        supports_waveform=waveform,
    )


def _linear_graph(
    *, fail_b: Any = False, parameterized: Any = False, count_state: Any = False
) -> Any:
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="coupled-scalar")
    a_input = cpl.CouplingPort("a-input", "input", space, reference_scale=1.0)
    a_output = cpl.CouplingPort("a-output", "output", space, reference_scale=1.0)
    b_input = cpl.CouplingPort("b-input", "input", space, reference_scale=1.0)
    b_output = cpl.CouplingPort("b-output", "output", space, reference_scale=1.0)

    def advance_a(window: Any, state: Any, inputs: Any, args: Any) -> Any:
        del window, args
        value = 0.5 * inputs[0]
        candidate = state + 1.0 if count_state else value
        return cpl.CouplingSubsystemResult(candidate, (value,), successful=True, status=0)

    def advance_b(window: Any, state: Any, inputs: Any, args: Any) -> Any:
        forcing = args if parameterized else jnp.asarray(1.0, dtype=inputs[0].dtype)
        value = 0.5 * (inputs[0] + forcing)
        candidate = state + 1.0 if count_state else value
        successful = jnp.asarray(True)
        if fail_b:
            successful = window.index == 0
        return cpl.CouplingSubsystemResult(
            candidate,
            (value,),
            successful=successful,
            status=jnp.where(successful, 0, 17),
        )

    subsystem_a = cpl.CallableCouplingSubsystem(
        advance_a,
        subsystem_id="a",
        input_ports=(a_input,),
        output_ports=(a_output,),
        capabilities=_capabilities(),
    )
    subsystem_b = cpl.CallableCouplingSubsystem(
        advance_b,
        subsystem_id="b",
        input_ports=(b_input,),
        output_ports=(b_output,),
        capabilities=_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (subsystem_b, subsystem_a),
        (
            cpl.CouplingExchange("a-to-b", "a-output", "b-input"),
            cpl.CouplingExchange("b-to-a", "b-output", "a-input"),
        ),
    )
    states = (jnp.zeros((1,), dtype=jnp.float64),) * 2
    values = (jnp.zeros((1,), dtype=jnp.float64),) * 2
    return graph, states, values


def _implicit_policy(*, maximum_steps: Any = 40, absolute: Any = 1e-10) -> Any:
    return cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(
            acceleration=phx.nonlinear.AndersonAcceleration(history=4)
        ),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=absolute,
            relative_residual=0.0,
            maximum_steps=maximum_steps,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-9),
            cpl.CouplingTolerance("b-input", absolute=1e-9),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )


def test_partitioned_coupling_scenario_1() -> None:
    graph, states, values = _linear_graph()
    reordered = cpl.CouplingGraph(
        tuple(reversed(graph.subsystems)),
        tuple(reversed(graph.exchanges)),
    )

    assert graph.graph_id == reordered.graph_id
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=_implicit_policy(),
    )
    assert prepared.report.subsystem_ids == ("a", "b")
    assert prepared.report.exchange_ids == ("a-to-b", "b-to-a")
    assert len(prepared.stages) == 1
    assert prepared.stages[0].cyclic
    assert prepared.report.resources.interface_size == 2
    first = phx.linalg.ArraySpace((1,), space_id="first")
    second = phx.linalg.ArraySpace((1,), space_id="second")
    output = cpl.CouplingPort("output", "output", first, reference_scale=1.0)
    input_ = cpl.CouplingPort("input", "input", second, reference_scale=1.0)
    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (state,), successful=True, status=0
        ),
        subsystem_id="source",
        output_ports=(output,),
        capabilities=_capabilities(),
    )
    target = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (), successful=True, status=0
        ),
        subsystem_id="target",
        input_ports=(input_,),
        capabilities=_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (source, target),
        (cpl.CouplingExchange("bad", "output", "input"),),
    )
    with pytest.raises(ValueError, match="vector-space identity"):
        cpl.prepare_coupling(
            graph,
            (jnp.zeros(1), jnp.zeros(1)),
            (jnp.zeros(1),),
            policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        )
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert bool(result.successful)
    assert not bool(result.converged)
    assert float(result.accepted_state.time) == pytest.approx(1.0)
    assert jnp.allclose(result.accepted_state.exchange_values[0], 0.0)
    assert jnp.allclose(result.accepted_state.exchange_values[1], 0.5)
    assert jnp.allclose(
        result.diagnostics.exchange_residual_norms,
        jnp.asarray([0.0, 0.5]),
    )
    graph, states, values = _linear_graph()
    policy = cpl.ExplicitCouplingPolicy(
        cpl.CouplingSweep("gauss-seidel", subsystem_order=("b", "a"))
    )
    prepared = cpl.prepare_coupling(graph, states, values, policy=policy)

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert bool(result.successful)
    assert jnp.allclose(result.accepted_state.exchange_values[0], 0.25)
    assert jnp.allclose(result.accepted_state.exchange_values[1], 0.5)
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(graph, states, values, policy=_implicit_policy())

    step = eqx.filter_jit(cpl.advance_coupling_window)
    result = step(prepared, prepared.reference_state, 1.0, None)

    assert bool(result.successful)
    assert bool(result.converged)
    assert jnp.allclose(
        result.accepted_state.exchange_values[0], jnp.asarray([1.0 / 3.0]), atol=1e-8
    )
    assert jnp.allclose(
        result.accepted_state.exchange_values[1], jnp.asarray([2.0 / 3.0]), atol=1e-8
    )
    assert jnp.all(result.diagnostics.exchange_certified)
    graph, states, values = _linear_graph(count_state=True)
    prepared = cpl.prepare_coupling(graph, states, values, policy=_implicit_policy())

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert bool(result.successful)
    assert all(
        jnp.allclose(state, 1.0) for state in result.accepted_state.participant_states
    )
    assert int(result.diagnostics.participant_evaluations[0]) > 1


def test_partitioned_coupling_scenario_2() -> None:
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=_implicit_policy(maximum_steps=1, absolute=1e-30),
    )

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert not bool(result.successful)
    assert not bool(result.converged)
    assert int(result.status) == int(cpl.CouplingStatus.WORK_EXHAUSTED)
    assert float(result.accepted_state.time) == pytest.approx(0.0)
    assert jnp.allclose(result.accepted_state.exchange_values[0], 0.0)
    assert not jnp.allclose(result.candidate_state.exchange_values[1], 0.0)
    graph, states, values = _linear_graph(fail_b=True)
    problem = cpl.CouplingProblem(
        graph,
        states,
        values,
        cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        t0=0.0,
        t1=3.0,
        window_size=1.0,
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
    )

    solution = cpl.solve_coupling(
        problem,
        rollout=cpl.CouplingRolloutPlan(retention="trajectory"),
    )

    assert not bool(solution.successful)
    assert float(solution.final_state.time) == pytest.approx(1.0)
    assert solution.retained_valid.tolist() == [True, True, False, False]
    assert int(solution.statuses[1]) == int(cpl.CouplingStatus.PARTICIPANT_FAILURE)
    assert jnp.all(solution.participant_evaluations[2] == 0)
    graph, states, values = _linear_graph()
    policy = cpl.ImplicitCouplingPolicy(
        phx.nonlinear.FixedPointIteration(),
        phx.nonlinear.NonlinearTermination(
            absolute_residual=0.4,
            relative_residual=0.0,
            maximum_steps=10,
        ),
        (
            cpl.CouplingTolerance("a-input", absolute=1e-12),
            cpl.CouplingTolerance("b-input", absolute=1e-12),
        ),
        fixed_point_sweep=cpl.CouplingSweep("jacobi"),
    )
    prepared = cpl.prepare_coupling(graph, states, values, policy=policy)

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    assert not bool(result.successful)
    assert int(result.status) == int(cpl.CouplingStatus.CERTIFICATION_FAILURE)
    assert float(result.accepted_state.time) == pytest.approx(0.0)


def test_partitioned_coupling_scenario_3() -> None:
    graph, states, values = _linear_graph()
    prepared = cpl.prepare_coupling(
        graph,
        states,
        values,
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
    )

    refreshed_graph, _, _ = _linear_graph(parameterized=True)
    refreshed = cpl.refresh_coupling(prepared, refreshed_graph, args=jnp.asarray(1.0))
    result = cpl.advance_coupling_window(
        refreshed,
        refreshed.reference_state,
        1.0,
        jnp.asarray(2.0),
    )

    assert refreshed.plan_id == prepared.plan_id
    assert int(refreshed.numeric_version) == 1
    assert jnp.allclose(result.accepted_state.exchange_values[1], 1.0)
    source_space = _field_space("source")
    target_space = _field_space("target")
    matrix = jnp.asarray([[1.0, 0.0, 0.0], [0.25, 0.5, 0.25], [0.0, 0.0, 1.0]])
    forward = phx.linalg.DenseLinearOperator(
        matrix,
        source=source_space.vector_space,
        target=target_space.vector_space,
    )
    adjoint = phx.linalg.DenseLinearOperator(
        matrix.T,
        source=target_space.vector_space,
        target=source_space.vector_space,
    )
    transfer = phx.discretization.FieldTransfer(
        source_space,
        target_space,
        forward,
        dual_pullback_operator=adjoint,
        hilbert_adjoint_operator=adjoint,
        properties=phx.discretization.TransferProperties(
            constant_preserving=True,
            adjoint_paired=True,
            exact_on=("constants",),
        ),
    )
    source_output = cpl.CouplingPort(
        "source-output",
        "output",
        source_space.vector_space,
        field_space=source_space,
        reference_scale=1.0,
    )
    target_input = cpl.CouplingPort(
        "target-input",
        "input",
        target_space.vector_space,
        field_space=target_space,
        reference_scale=1.0,
    )
    target_output = cpl.CouplingPort(
        "target-output",
        "output",
        target_space.vector_space,
        field_space=target_space,
        reference_scale=1.0,
    )
    source_input = cpl.CouplingPort(
        "source-input",
        "input",
        source_space.vector_space,
        field_space=source_space,
        reference_scale=1.0,
    )
    source_value = jnp.asarray([1.0, 2.0, 3.0])
    target_value = jnp.asarray([2.0, -1.0, 0.5])

    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (source_value,), successful=True, status=0
        ),
        subsystem_id="source",
        input_ports=(source_input,),
        output_ports=(source_output,),
        capabilities=_capabilities(),
    )
    target = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (target_value,), successful=True, status=0
        ),
        subsystem_id="target",
        input_ports=(target_input,),
        output_ports=(target_output,),
        capabilities=_capabilities(),
    )
    graph = cpl.CouplingGraph(
        (source, target),
        (
            cpl.CouplingExchange(
                "forward",
                "source-output",
                "target-input",
                transfer=transfer,
                requirement=cpl.CouplingTransferRequirement(constant_preserving=True),
            ),
            cpl.CouplingExchange(
                "adjoint",
                "target-output",
                "source-input",
                transfer=transfer,
                use_adjoint=True,
                requirement=cpl.CouplingTransferRequirement(adjoint_paired=True),
            ),
        ),
    )
    prepared = cpl.prepare_coupling(
        graph,
        (jnp.zeros(1), jnp.zeros(1)),
        (jnp.zeros(3), jnp.zeros(3)),
        policy=cpl.ExplicitCouplingPolicy(cpl.CouplingSweep("jacobi")),
        differentiation=cpl.CouplingDifferentiationPolicy("algorithmic"),
    )
    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    mapped_source = result.accepted_state.exchange_values[0]
    mapped_target = result.accepted_state.exchange_values[1]
    assert jnp.allclose(mapped_target, matrix @ source_value)
    assert jnp.allclose(mapped_source, matrix.T @ target_value)
    assert jnp.vdot(mapped_target, target_value) == pytest.approx(
        float(jnp.vdot(source_value, mapped_source))
    )


def _field_space(name: Any) -> Any:
    topology = phx.discretization.TensorTopology(("x",), (3,))
    support = phx.discretization.DiscreteSupport(topology, 1, f"{name}-support")
    layout = phx.discretization.TensorDofLayout(("x",), (3,))
    vector_space = phx.linalg.ArraySpace((3,), space_id=f"{name}-vectors")
    return phx.discretization.DiscreteFieldSpace(
        name,
        support.support_id,
        layout,
        vector_space,
        representation="point_value",
    )


def test_component_inventories_keep_separate_ledger_rows() -> None:
    area = np.asarray([1.0, 3.0])
    measure = phx.discretization.DiscreteMeasure(
        "surface-area", "vector-surface", "vector-cells", jnp.asarray(area)
    )
    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64, space_id="vector-cells")
    field = phx.discretization.DiscreteFieldSpace(
        "surface-momentum",
        "vector-surface",
        phx.discretization.EntityDofLayout("vector-cells", 2, 2, component_shape=(2,)),
        space,
        representation="cell_average",
    )
    unit = phx.units.derived_unit(
        "kg/(m s)",
        ((phx.units.KILOGRAM, 1), (phx.units.METER, -1), (phx.units.SECOND, -1)),
    )
    measurement = cpl.CouplingMeasurement.from_measure(
        measure,
        space,
        phx.units.derived_unit("m²", ((phx.units.METER, 2),)),
        component_ids=("x", "y"),
    )
    quantity = cpl.CouplingQuantity("momentum_per_area", unit)
    # Entity-major (entity, component) amounts per unit window length.
    rate = np.asarray([1.0, -2.0, 3.0, 5.0])

    def port(port_id: Any, direction: Any, frame: Any) -> Any:
        return cpl.CouplingPort(
            port_id,
            direction,
            space,
            field_space=field,
            quantity=quantity,
            measurement=measurement,
            temporal_kind="interval_integral",
            frame=frame,
            reference_scale=1.0,
        )

    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (jnp.asarray(rate) * window.size,), successful=True, status=0
        ),
        subsystem_id="source",
        output_ports=(port("out", "output", "cartesian-xy"),),
        capabilities=_capabilities(),
    )
    target = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state + inputs[0], (), successful=True, status=0
        ),
        subsystem_id="target",
        input_ports=(port("in", "input", "cartesian-xy"),),
        capabilities=_capabilities(),
    )
    exchange = cpl.CouplingExchange(
        "ex", "out", "in", temporal=cpl.CouplingTemporalConversion("window-integral")
    )
    zero = jnp.zeros((4,), dtype=jnp.float64)
    prepared = cpl.prepare_coupling(
        cpl.CouplingGraph((source, target), (exchange,)),
        (zero, zero),
        (zero,),
        policy=cpl.ExplicitCouplingPolicy(
            cpl.CouplingSweep("gauss-seidel", subsystem_order=("source", "target"))
        ),
    )

    first = cpl.advance_coupling_window(prepared, prepared.reference_state, 2.0)
    second = cpl.advance_coupling_window(prepared, first.accepted_state, 0.5)

    components = area @ rate.reshape(2, 2)
    expected = np.stack((-components, components), axis=-1)
    assert prepared.reference_state.budget_row_ids == ("ex[x]", "ex[y]")
    assert second.accepted_state.budget_row_ids == ("ex[x]", "ex[y]")
    assert bool(first.successful) and bool(second.successful)
    np.testing.assert_allclose(first.accepted_exchange_budget, 2.0 * expected)
    np.testing.assert_allclose(
        second.accepted_state.cumulative_exchange_budget, 2.5 * expected
    )
    with pytest.raises(ValueError, match="declared component frame"):
        port("scalar-in", "input", "scalar")


def _mass_transfer(amount: Any, sweep: Any) -> Any:
    """A tank spends `amount` kg per window; its reference scale is that amount."""
    space = phx.linalg.ArraySpace((1,), dtype=jnp.float64, space_id="cell-mass")
    field = phx.discretization.DiscreteFieldSpace(
        "mass",
        "tank",
        phx.discretization.EntityDofLayout("tank/cells", 1, 1),
        space,
        representation="cell_integral",
    )
    measurement = cpl.CouplingMeasurement.extensive(space, "tank", provenance_id="cell")

    def port(port_id: Any, direction: Any) -> Any:
        return cpl.CouplingPort(
            port_id,
            direction,
            space,
            field_space=field,
            measurement=measurement,
            quantity=cpl.CouplingQuantity("mass", phx.units.KILOGRAM),
            temporal_kind="interval_integral",
            reference_scale=amount,
        )

    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state - amount,
            (jnp.full((1,), amount, dtype=jnp.float64),),
            successful=True,
            status=0,
        ),
        subsystem_id="source",
        output_ports=(port("out", "output"),),
        capabilities=_capabilities(),
    )
    sink = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state + inputs[0], (), successful=True, status=0
        ),
        subsystem_id="sink",
        input_ports=(port("in", "input"),),
        capabilities=_capabilities(),
    )
    exchange = cpl.CouplingExchange(
        "mass", "out", "in", temporal=cpl.CouplingTemporalConversion("window-integral")
    )
    zero = jnp.zeros((1,), dtype=jnp.float64)
    return cpl.prepare_coupling(
        cpl.CouplingGraph((source, sink), (exchange,)),
        (zero, zero),
        (zero,),
        policy=cpl.ExplicitCouplingPolicy(sweep),
    )


@pytest.mark.parametrize("amount", [1.0, 1.0e-6, 1.0e-15], ids=["kg", "mg", "fg"])
def test_ledger_certification_scales_with_the_declared_reference_amount(
    amount: Any,
) -> None:
    # A Jacobi sweep consumes the stale zero exchange, so the whole amount is lost.
    lossy = _mass_transfer(amount, cpl.CouplingSweep("jacobi"))
    balanced = _mass_transfer(
        amount, cpl.CouplingSweep("gauss-seidel", subsystem_order=("source", "sink"))
    )

    lost = cpl.advance_coupling_window(lossy, lossy.reference_state, 1.0)
    kept = cpl.advance_coupling_window(balanced, balanced.reference_state, 1.0)

    assert not bool(lost.successful)
    assert int(lost.status) == int(cpl.CouplingStatus.CERTIFICATION_FAILURE)
    assert bool(kept.successful)
    np.testing.assert_allclose(kept.accepted_exchange_budget, [[-amount, amount]])


def _density_to_cell_integral(conservative: Any) -> Any:
    area = jnp.asarray([1.0, 3.0])
    source_space = phx.linalg.ArraySpace((2,), dtype=jnp.float64, space_id="density")
    source_field = phx.discretization.DiscreteFieldSpace(
        "surface-heat",
        "source",
        phx.discretization.EntityDofLayout("source/cells", 2, 2),
        source_space,
        representation="cell_average",
    )
    target_space = phx.linalg.ArraySpace((2,), dtype=jnp.float64, space_id="content")
    target_field = phx.discretization.DiscreteFieldSpace(
        "surface-content",
        "target",
        phx.discretization.EntityDofLayout("target/cells", 2, 2),
        target_space,
        representation="cell_integral",
    )
    square_meter = phx.units.derived_unit("m²", ((phx.units.METER, 2),))
    output = cpl.CouplingPort(
        "out",
        "output",
        source_space,
        field_space=source_field,
        measurement=cpl.CouplingMeasurement.from_measure(
            phx.discretization.DiscreteMeasure(
                "surface-area", "source", "source/cells", area
            ),
            source_space,
            square_meter,
        ),
        quantity=cpl.CouplingQuantity("enthalpy", phx.units.JOULE_PER_SQUARE_METER),
        reference_scale=1.0,
    )
    input_ = cpl.CouplingPort(
        "in",
        "input",
        target_space,
        field_space=target_field,
        measurement=cpl.CouplingMeasurement.extensive(
            target_space, "target", provenance_id="target-cell"
        ),
        quantity=cpl.CouplingQuantity("enthalpy", phx.units.JOULE),
        reference_scale=1.0,
    )
    # Conservative: cell content = area x density; otherwise densities are copied.
    matrix = jnp.diag(area) if conservative else jnp.eye(2)
    transfer = phx.discretization.FieldTransfer(
        source_field,
        target_field,
        phx.linalg.DenseLinearOperator(matrix, source=source_space, target=target_space),
        dual_pullback_operator=phx.linalg.DenseLinearOperator(
            matrix.T, source=target_space, target=source_space
        ),
        properties=phx.discretization.TransferProperties(
            conservative=conservative, positivity_preserving=True
        ),
    )
    source = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            state, (jnp.asarray([2.0, 3.0]),), successful=True, status=0
        ),
        subsystem_id="source",
        output_ports=(output,),
        capabilities=_capabilities(),
    )
    sink = cpl.CallableCouplingSubsystem(
        lambda window, state, inputs, args: cpl.CouplingSubsystemResult(
            inputs[0], (), successful=True, status=0
        ),
        subsystem_id="sink",
        input_ports=(input_,),
        capabilities=_capabilities(),
    )
    exchange = cpl.CouplingExchange(
        "energy",
        "out",
        "in",
        transfer=transfer,
        requirement=cpl.CouplingTransferRequirement(
            conservative=conservative, positivity_preserving=True
        ),
    )
    zero = jnp.zeros((2,), dtype=jnp.float64)
    return cpl.prepare_coupling(
        cpl.CouplingGraph((source, sink), (exchange,)),
        (zero, zero),
        (zero,),
        policy=cpl.ExplicitCouplingPolicy(
            cpl.CouplingSweep("gauss-seidel", subsystem_order=("source", "sink"))
        ),
    )


def test_density_reaches_extensive_storage_only_through_a_certified_transfer() -> None:
    prepared = _density_to_cell_integral(True)

    result = cpl.advance_coupling_window(prepared, prepared.reference_state, 1.0)

    # J/m² densities 2 and 3 over cells of 1 m² and 3 m² hold 2 J and 9 J.
    assert bool(result.successful)
    state = result.accepted_state
    np.testing.assert_allclose(
        state.participant_states[state.subsystem_ids.index("sink")], [2.0, 9.0]
    )
    with pytest.raises(ValueError, match="certified conservative transfer"):
        _density_to_cell_integral(False)
