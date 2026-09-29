from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.solver._dae_events import _DAEEventRootArguments, _DAEEventRootResidual
from phydrax.solver._dae_initialization import _scaled_space
from tests._support.differentiation import assert_hilbert_adjoint_duality


la = phx.linalg
dyn = phx.dynamics

# Heat/flux cell: differential temperature with an algebraic constitutive flux.
_CAPACITY = np.asarray([2.0, 3.0])
_CONDUCTANCE = 1.0e-3
_TEMPERATURE_SCALE = np.asarray([300.0, 50.0])
_TEMPERATURE_RATE_SCALE = 10.0
_FLUX_SCALE = 5.0
_BALANCE_SCALE = 20.0
_LAW_SCALE = 0.5
_INITIAL = np.asarray([0.0, 0.0, 300.0, 40.0])
_MATERIALIZATION = la.MaterializationPolicy(max_entries=1_024, max_bytes=65_536)


def _balance(time: Array, jet: Any, forcing: Array) -> Array:
    del time, forcing
    return jnp.asarray(_CAPACITY) * jet.value("temperature", 1) + jet.value("flux")


def _law(time: Array, jet: Any, forcing: Array) -> Array:
    temperature = jet.value("temperature")
    return jet.value("flux") - _CONDUCTANCE * temperature**2 + forcing * jnp.sin(time)


def _heat_compilation() -> dyn.ReducedDAECompilation:
    component = dyn.DAEComponent(
        "cell",
        (
            dyn.DAEVariableBlock(
                "temperature",
                (2,),
                1,
                state_scale=jnp.asarray(_TEMPERATURE_SCALE),
                rate_scale=_TEMPERATURE_RATE_SCALE,
            ),
            dyn.DAEVariableBlock("flux", (2,), 0, state_scale=_FLUX_SCALE),
        ),
        (
            dyn.DAEEquationBlock(
                "balance",
                _balance,
                (
                    dyn.DAEDerivativeIncidence("temperature", 1),
                    dyn.DAEDerivativeIncidence("flux", 0),
                ),
                residual_scale=_BALANCE_SCALE,
                residual_semantic_id="cell-energy-balance",
                residual_numeric_id="cell-energy-balance-capacity",
            ),
            dyn.DAEEquationBlock(
                "law",
                _law,
                (
                    dyn.DAEDerivativeIncidence("flux", 0),
                    dyn.DAEDerivativeIncidence("temperature", 0),
                ),
                residual_scale=_LAW_SCALE,
                residual_semantic_id="cell-radiative-law",
                residual_numeric_id="cell-radiative-law-conductance",
            ),
        ),
    )
    return dyn.compile_acausal_dae(
        dyn.AcausalDAESource((component,)),
        dyn.DAEStructuralPolicy(0, 0),
        args=jnp.asarray(0.3),
    )


def _heat_linearization(
    adapter: phx.solver.DAECoordinateAdapter,
) -> phx.solver.AutonomousDAEBlockLinearization:
    variables = adapter.variable_space
    equations = adapter.equation_space
    flux = variables.spaces[variables.names.index("cell.flux")]
    temperature = variables.spaces[variables.names.index("cell.temperature")]
    balance = equations.spaces[equations.names.index("cell.balance")]
    law = equations.spaces[equations.names.index("cell.law")]

    def linearization(
        time: Array, state: Any, rate: Any, forcing: Array
    ) -> phx.solver.DAEBlockJacobian:
        del time, rate, forcing
        values = dict(zip(variables.names, state, strict=True))
        state_jacobian = la.assemble_block_operator(
            (
                (
                    ("cell.balance",),
                    ("cell.flux",),
                    la.DenseLinearOperator(jnp.eye(2), source=flux, target=balance),
                ),
                (
                    ("cell.law",),
                    ("cell.flux",),
                    la.DenseLinearOperator(jnp.eye(2), source=flux, target=law),
                ),
                (
                    ("cell.law",),
                    ("cell.temperature",),
                    la.DenseLinearOperator(
                        jnp.diag(-2.0 * _CONDUCTANCE * values["cell.temperature"]),
                        source=temperature,
                        target=law,
                    ),
                ),
            ),
            source=variables,
            target=equations,
        )
        rate_jacobian = la.assemble_block_operator(
            (
                (
                    ("cell.balance",),
                    ("cell.temperature",),
                    la.DenseLinearOperator(
                        jnp.diag(jnp.asarray(_CAPACITY)),
                        source=temperature,
                        target=balance,
                    ),
                ),
            ),
            source=variables,
            target=equations,
        )
        return phx.solver.DAEBlockJacobian(state_jacobian, rate_jacobian)

    return linearization


def _oscillator_compilation() -> dyn.ReducedDAECompilation:
    def newton(time: Array, jet: Any, args: Any) -> Array:
        del time, args
        return 2.0 * jet.value("position", 2) + jet.value("force")

    def spring(time: Array, jet: Any, args: Any) -> Array:
        del time, args
        return jet.value("force") - 3.0 * jet.value("position")

    component = dyn.DAEComponent(
        "mass",
        (
            dyn.DAEVariableBlock("position", (), 2, state_scale=4.0, rate_scale=0.25),
            dyn.DAEVariableBlock("force", (), 0, state_scale=7.0),
        ),
        (
            dyn.DAEEquationBlock(
                "newton",
                newton,
                (
                    dyn.DAEDerivativeIncidence("position", 2),
                    dyn.DAEDerivativeIncidence("force", 0),
                ),
                residual_scale=9.0,
                residual_semantic_id="mass-newton",
                residual_numeric_id="mass-newton-mass",
            ),
            dyn.DAEEquationBlock(
                "spring",
                spring,
                (
                    dyn.DAEDerivativeIncidence("force", 0),
                    dyn.DAEDerivativeIncidence("position", 0),
                ),
                residual_scale=0.125,
                residual_semantic_id="mass-spring",
                residual_numeric_id="mass-spring-stiffness",
            ),
        ),
    )
    return dyn.compile_acausal_dae(
        dyn.AcausalDAESource((component,)), dyn.DAEStructuralPolicy(0, 0)
    )


def _oscillator_linearization(
    adapter: phx.solver.DAECoordinateAdapter,
) -> phx.solver.AutonomousDAEBlockLinearization:
    variables = adapter.variable_space
    equations = adapter.equation_space
    force = variables.spaces[variables.names.index("mass.force")]
    position = variables.spaces[variables.names.index("mass.position")]
    newton = equations.spaces[equations.names.index("mass.newton")]
    spring = equations.spaces[equations.names.index("mass.spring")]
    assert isinstance(position, la.BlockSpace)
    displacement, velocity = position.spaces

    def linearization(
        time: Array, state: Any, rate: Any, args: Any
    ) -> phx.solver.DAEBlockJacobian:
        del time, state, rate, args
        state_jacobian = la.assemble_block_operator(
            (
                (
                    ("mass.newton",),
                    ("mass.force",),
                    la.DenseLinearOperator(jnp.eye(1), source=force, target=newton),
                ),
                (
                    ("mass.spring",),
                    ("mass.force",),
                    la.DenseLinearOperator(jnp.eye(1), source=force, target=spring),
                ),
                (
                    ("mass.spring",),
                    ("mass.position", "0"),
                    la.DenseLinearOperator(
                        -3.0 * jnp.eye(1), source=displacement, target=spring
                    ),
                ),
            ),
            source=variables,
            target=equations,
        )
        rate_jacobian = la.assemble_block_operator(
            (
                (
                    ("mass.newton",),
                    ("mass.position", "1"),
                    la.DenseLinearOperator(
                        2.0 * jnp.eye(1), source=velocity, target=newton
                    ),
                ),
            ),
            source=variables,
            target=equations,
        )
        return phx.solver.DAEBlockJacobian(state_jacobian, rate_jacobian)

    return linearization


def _named_policy(
    adapter: phx.solver.DAECoordinateAdapter,
    *,
    adaptive: phx.solver.DAEAdaptivePolicy | None = None,
) -> phx.solver.DAESolvePolicy:
    """Stage LDU over named role blocks and an exact named consistency correction."""
    stage = la.SubspaceCorrectionTerm(
        *adapter.correction_transfers("stage", adapter.role_groups("stage")),
        la.BlockFactorizationPreconditionerBuilder(
            la.DenseInversePreconditionerBuilder(),
            la.DenseInversePreconditionerBuilder(),
            "ldu",
        ),
    )
    initialization = la.SubspaceCorrectionTerm(
        *adapter.correction_transfers(
            "initialization", adapter.role_groups("initialization")
        ),
        la.DenseInversePreconditionerBuilder(),
    )

    def method(term: la.SubspaceCorrectionTerm) -> phx.nonlinear.NewtonKrylov:
        return phx.nonlinear.NewtonKrylov(
            linear_policy=la.LinearSolvePolicy(
                la.FGMRES(restart=4),
                tolerance=la.TolerancePolicy(relative=1e-13, absolute=1e-15, max_steps=4),
                preconditioning=la.PreconditioningPolicy(
                    la.AdditiveSubspaceCorrectionBuilder((term,))
                ),
                materialization=_MATERIALIZATION,
            )
        )

    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_method=method(stage),
        initialization_method=method(initialization),
        nonlinear_termination=_termination(),
        initialization_termination=_termination(),
        adaptive=adaptive,
    )


def _termination() -> phx.nonlinear.NonlinearTermination:
    return phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-12, relative_residual=0.0, maximum_steps=20
    )


def _unpreconditioned_method() -> phx.nonlinear.NewtonKrylov:
    return phx.nonlinear.NewtonKrylov(
        linear_policy=la.LinearSolvePolicy(
            la.GMRES(restart=4),
            tolerance=la.TolerancePolicy(relative=1e-13, absolute=1e-15, max_steps=8),
        )
    )


def _native_heat_system() -> dyn.DifferentialAlgebraicSystem:
    """Independent flat declaration in the compiled (flux, temperature) order."""

    def residual(time: Array, state: Array, rate: Array, forcing: Array) -> Array:
        flux, temperature = state[:2], state[2:]
        balance = jnp.asarray(_CAPACITY) * rate[2:] + flux
        law = flux - _CONDUCTANCE * temperature**2 + forcing * jnp.sin(time)
        return jnp.concatenate((balance, law))

    return dyn.DifferentialAlgebraicSystem(
        residual,
        state_shape=(4,),
        structure=dyn.DAEStructure(
            ("algebraic", "algebraic", "differential", "differential"),
            equation_roles=("differential", "differential", "algebraic", "algebraic"),
        ),
        state_scale=jnp.asarray((_FLUX_SCALE, _FLUX_SCALE, *_TEMPERATURE_SCALE)),
        state_rate_scale=jnp.asarray((1.0, 1.0) + (_TEMPERATURE_RATE_SCALE,) * 2),
        residual_scale=jnp.asarray((_BALANCE_SCALE,) * 2 + (_LAW_SCALE,) * 2),
        system_id="native-heat-flux-cell",
    )


def _materialized(operator: la.AbstractLinearOperator) -> np.ndarray:
    return np.asarray(la.materialize(operator, _MATERIALIZATION))


def test_named_blocks_preserve_native_scales_and_distinct_roles() -> None:
    adapter = phx.solver.DAECoordinateAdapter(_heat_compilation())
    state = jnp.asarray([1.0, 2.0, 3.0, 4.0])
    roles = {coordinate.path: coordinate.role for coordinate in adapter.variables}
    row_roles = {coordinate.path: coordinate.role for coordinate in adapter.rows}

    flux_scale, temperature_scale = adapter.scale_view("state")
    ((balance_scale, law_scale),) = adapter.scale_view("residual")

    assert roles == {("cell.flux",): "algebraic", ("cell.temperature",): "differential"}
    assert row_roles == {
        ("equations", "cell.balance"): "differential",
        ("equations", "cell.law"): "algebraic",
    }
    assert adapter.system.structure.variable_roles != (
        adapter.system.structure.equation_roles
    )
    np.testing.assert_allclose(temperature_scale, _TEMPERATURE_SCALE)
    np.testing.assert_allclose(flux_scale, [_FLUX_SCALE] * 2)
    np.testing.assert_allclose(adapter.scale_view("rate")[1], [10.0, 10.0])
    np.testing.assert_allclose(balance_scale, [_BALANCE_SCALE] * 2)
    np.testing.assert_allclose(law_scale, [_LAW_SCALE] * 2)
    np.testing.assert_array_equal(adapter.native_state(adapter.state_view(state)), state)
    np.testing.assert_array_equal(adapter.native_rows(adapter.row_view(state)), state)


@pytest.mark.parametrize(
    ("compilation", "linearization", "initial"),
    [
        pytest.param(
            _heat_compilation, _heat_linearization, _INITIAL, id="heat-flux-cell"
        ),
        pytest.param(
            _oscillator_compilation,
            _oscillator_linearization,
            np.asarray([0.0, 1.0, -0.5]),
            id="second-order-kinematics",
        ),
    ],
)
def test_named_root_setups_equal_native_root_jacobians(
    compilation: Any, linearization: Any, initial: np.ndarray
) -> None:
    adapter = phx.solver.DAECoordinateAdapter(compilation())
    bound = adapter.bind_linearization(linearization(adapter))
    system = bound.system
    problem = phx.solver.DifferentialAlgebraicProblem(
        bound,
        jnp.asarray(initial),
        initial_state_rate=jnp.asarray(np.linspace(0.5, -0.5, initial.size)),
        args=jnp.asarray(0.3),
        initialization="structural",
    )
    prepared = phx.solver.prepare_dae(
        problem,
        dyn.TimeGrid(jnp.asarray([0.0, 0.1]), time_id="named-root-maps"),
        policy=_named_policy(adapter),
    )
    stage_problem = prepared.stage_problem
    stage_arguments = prepared.stage_solve.args
    increment = jnp.asarray(np.linspace(0.02, -0.03, initial.size))
    source, target = stage_problem.state_space, stage_problem.residual_space
    assert isinstance(source, la.ArraySpace) and isinstance(target, la.ArraySpace)
    stage_setup = system.stage_linear_setup
    tangent_setup = system.tangent_linear_setup
    adjoint_setup = system.adjoint_linear_setup
    initialization_setup = system.initialization_linear_setup
    assert stage_setup is not None and tangent_setup is not None
    assert adjoint_setup is not None and initialization_setup is not None
    covector = jnp.asarray(np.linspace(1.0, -2.0, initial.size))

    stage = stage_setup(increment, stage_arguments, source, target)
    tangent = tangent_setup(increment, stage_arguments, source, target)
    adjoint = adjoint_setup(increment, stage_arguments, target, source)
    reference = np.asarray(
        jax.jacfwd(lambda value: stage_problem.residual(value, stage_arguments))(
            increment
        )
    )
    consistency = prepared.initialization.nonlinear_problem
    consistency_solve = prepared.initialization.nonlinear_solve
    assert consistency is not None and consistency_solve is not None
    consistency_source = consistency.state_space
    consistency_target = consistency.residual_space
    assert isinstance(consistency_source, la.ArraySpace)
    assert isinstance(consistency_target, la.ArraySpace)
    consistency_arguments = consistency_solve.args
    unknown = consistency_solve.state
    initialization = initialization_setup(
        unknown, consistency_arguments, consistency_source, consistency_target
    )
    initialization_reference = np.asarray(
        jax.jacfwd(lambda value: consistency.residual(value, consistency_arguments))(
            unknown
        )
    )

    # The block setup is evaluated at physical_state(z) and scaled exactly once.
    np.testing.assert_allclose(_materialized(stage), reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_materialized(tangent), reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        _materialized(adjoint), reference.T, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        stage.transpose_mv(covector), reference.T @ covector, rtol=1e-12, atol=1e-12
    )
    assert_hilbert_adjoint_duality(stage, increment, covector, rtol=1e-12, atol=1e-12)
    assert not np.allclose(stage.adjoint_mv(covector), stage.transpose_mv(covector))
    np.testing.assert_allclose(
        _materialized(initialization),
        initialization_reference,
        rtol=1e-12,
        atol=1e-12,
    )
    with pytest.raises(TypeError, match="initialization root arguments"):
        stage_setup(unknown, consistency_arguments, source, target)


def test_named_event_setup_borders_the_stage_with_time_and_guard() -> None:
    adapter = phx.solver.DAECoordinateAdapter(_heat_compilation())
    system = adapter.bind_linearization(_heat_linearization(adapter)).system
    size = system.state_size

    def guard(time: Array, state: Array, forcing: Array) -> Array:
        return state[2] - 250.0 + forcing * time

    history = jnp.asarray(
        [_INITIAL + offset * np.asarray([1.0, -1.0, -2.0, 0.5]) for offset in range(5)]
    )
    arguments = _DAEEventRootArguments(
        history,
        jnp.zeros_like(history),
        jnp.asarray([0.2, 0.1, 0.0, -0.1, -0.2]),
        jnp.asarray(2, dtype=jnp.int32),
        jnp.asarray(0.3),
        jnp.asarray(True),
        jnp.zeros((size + 1,)),
        jnp.asarray(0.2),
        jnp.asarray(0.4),
        guard,
        jnp.asarray(1e-4),
    )
    residual = _DAEEventRootResidual(system, None, guard, 1e-4, (size,), size)
    augmented = jnp.asarray([0.01, -0.02, -0.5, 0.25, 0.27])
    source = _scaled_space(
        (size + 1,),
        jnp.float64,
        jnp.concatenate((system.state_scale, jnp.ones((1,)))),
        space_id="named-event-increment",
    )
    target = _scaled_space(
        (size + 1,),
        jnp.float64,
        jnp.ones((size + 1,)),
        space_id="named-event-residual",
    )

    event_setup = system.event_linear_setup
    assert event_setup is not None
    setup = event_setup(augmented, arguments, source, target)
    reference = np.asarray(
        jax.jacfwd(lambda value: residual(value, arguments))(augmented)
    )

    assert isinstance(setup, la.MappedBlockLinearOperator)
    assert setup.row_map.names == ("equations", "guard")
    assert setup.column_map.names == ("increment", "time")
    np.testing.assert_allclose(_materialized(setup), reference, rtol=1e-10, atol=1e-10)


def test_named_block_preconditioned_bdf_matches_native_array_dae() -> None:
    compilation = _heat_compilation()
    adapter = phx.solver.DAECoordinateAdapter(compilation)
    bound = adapter.bind_linearization(_heat_linearization(adapter))
    grid = dyn.TimeGrid(jnp.linspace(0.0, 1.0, 11), time_id="named-heat-grid")
    named = phx.solver.prepare_dae(
        phx.solver.DifferentialAlgebraicProblem(
            bound,
            jnp.asarray(_INITIAL),
            args=jnp.asarray(0.3),
            initialization="structural",
        ),
        grid,
        policy=_named_policy(adapter),
    )
    native = phx.solver.prepare_dae(
        phx.solver.DifferentialAlgebraicProblem(
            _native_heat_system(), jnp.asarray(_INITIAL), args=jnp.asarray(0.3)
        ),
        grid,
        policy=phx.solver.DAESolvePolicy(
            method=phx.solver.BDFMethod(2),
            nonlinear_method=_unpreconditioned_method(),
            initialization_method=_unpreconditioned_method(),
            nonlinear_termination=_termination(),
            initialization_termination=_termination(),
        ),
    )

    def terminal(prepared: phx.solver.PreparedDAESolve, forcing: Array) -> Array:
        return phx.solver.solve_dae(prepared, args=forcing).states[-1, 2]

    named_solution = phx.solver.solve_dae(named)
    native_solution = phx.solver.solve_dae(native)
    history = named_solution.attempt_history
    attempts = int(history.count)
    forcing = jnp.asarray(0.3)
    named_tangent = jax.jvp(
        lambda value: terminal(named, value), (forcing,), (jnp.asarray(1.0),)
    )[1]
    native_tangent = jax.jvp(
        lambda value: terminal(native, value), (forcing,), (jnp.asarray(1.0),)
    )[1]

    assert named_solution.successful and native_solution.successful
    np.testing.assert_allclose(
        named_solution.states, native_solution.states, rtol=1e-10, atol=1e-9
    )
    # The named LDU is the exact inverse of the named stage setup, and that setup
    # equals the native Jacobian: every Krylov solve takes one iteration.
    np.testing.assert_array_equal(
        history.linear_iterations[:attempts], history.linear_solves[:attempts]
    )
    assert int(jnp.sum(history.linear_solves[:attempts])) > 0
    np.testing.assert_allclose(named_tangent, native_tangent, rtol=1e-7, atol=1e-9)


def test_named_setup_continues_accepted_bdf_history() -> None:
    adapter = phx.solver.DAECoordinateAdapter(_heat_compilation())
    problem = phx.solver.DifferentialAlgebraicProblem(
        adapter.bind_linearization(_heat_linearization(adapter)),
        jnp.asarray(_INITIAL),
        args=jnp.asarray(0.3),
        initialization="structural",
    )
    policy = _named_policy(
        adapter,
        adaptive=phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8,
            maximum_accepted_steps=128,
            maximum_attempts=256,
        ),
    )
    full = phx.solver.solve_dae(
        problem,
        dyn.TimeGrid(jnp.asarray([0.0, 0.2, 0.4]), time_id="named-full"),
        policy=policy,
    )
    first = phx.solver.solve_dae(
        problem,
        dyn.TimeGrid(jnp.asarray([0.0, 0.2]), time_id="named-first"),
        policy=policy,
    )
    second = phx.solver.solve_dae(
        phx.solver.prepare_dae(
            problem,
            dyn.TimeGrid(jnp.asarray([0.2, 0.4]), time_id="named-second"),
            policy=policy,
        ),
        continuation=first.continuation,
    )
    first_count = int(first.step_history.count)
    second_count = int(second.step_history.count)
    full_count = int(full.step_history.count)

    assert full.successful & first.successful & second.successful
    assert second.initialization.nonlinear_result is None
    np.testing.assert_allclose(
        jnp.concatenate((first.states, second.states[1:])),
        full.states,
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(
        jnp.concatenate(
            (
                first.step_history.accepted_times[:first_count],
                second.step_history.accepted_times[:second_count],
            )
        ),
        full.step_history.accepted_times[:full_count],
    )


def _index_two_source() -> dyn.AcausalDAESource:
    def dynamics(time: Array, jet: Any, args: Any) -> Array:
        del time, args
        return jet.value("position", 1) - jet.value("multiplier")

    def constraint(time: Array, jet: Any, args: Any) -> Array:
        del args
        return jet.value("position") - jnp.sin(time)

    return dyn.AcausalDAESource(
        (
            dyn.DAEComponent(
                "rail",
                (
                    dyn.DAEVariableBlock("position", (), 1),
                    dyn.DAEVariableBlock("multiplier", (), 0),
                ),
                (
                    dyn.DAEEquationBlock(
                        "dynamics",
                        dynamics,
                        (
                            dyn.DAEDerivativeIncidence("position", 1),
                            dyn.DAEDerivativeIncidence("multiplier", 0),
                        ),
                        residual_semantic_id="rail-dynamics",
                        residual_numeric_id="rail-dynamics-unit",
                    ),
                    dyn.DAEEquationBlock(
                        "constraint",
                        constraint,
                        (dyn.DAEDerivativeIncidence("position", 0),),
                        residual_semantic_id="rail-position-constraint",
                        residual_numeric_id="rail-position-constraint-sine",
                    ),
                ),
            ),
        )
    )


def test_initialization_frees_the_state_and_rate_of_one_block() -> None:
    # The flux is fixed in state and rate; the temperature state and rate are both
    # free: four unknowns for four residual rows, two columns per temperature path.
    fixed = np.asarray([True, True, False, False])
    spec = phx.solver.DAEInitializationSpec.from_masks(fixed, fixed)
    adapter = phx.solver.DAECoordinateAdapter(_heat_compilation(), initialization=spec)
    columns = adapter.root_columns("initialization")
    assert columns.names == ("state", "rate") and columns.size == 4
    bound = adapter.bind_linearization(_heat_linearization(adapter))
    # The named consistency correction pairs balance rows with the free rates and
    # law rows with the free states of the same temperature path.
    prepared = phx.solver.prepare_dae(
        phx.solver.DifferentialAlgebraicProblem(
            bound,
            jnp.asarray(_INITIAL),
            initial_state_rate=jnp.zeros(4),
            args=jnp.asarray(0.3),
            initialization=spec,
        ),
        dyn.TimeGrid(jnp.asarray([0.0, 0.1]), time_id="state-and-rate-free"),
        policy=_named_policy(adapter),
    )
    consistency = prepared.initialization.nonlinear_problem
    solve = prepared.initialization.nonlinear_solve
    setup = bound.system.initialization_linear_setup
    assert consistency is not None and solve is not None and setup is not None
    source, target = consistency.state_space, consistency.residual_space
    assert isinstance(source, la.ArraySpace) and isinstance(target, la.ArraySpace)
    unknown = solve.state + jnp.asarray([0.5, -0.25, 1.0, 2.0])
    reference = np.asarray(
        jax.jacfwd(lambda value: consistency.residual(value, solve.args))(unknown)
    )
    np.testing.assert_allclose(
        _materialized(setup(unknown, solve.args, source, target)),
        reference,
        rtol=1e-12,
        atol=1e-12,
    )


def test_unreduced_higher_index_and_unadmitted_systems_are_refused() -> None:
    source = _index_two_source()
    analysis = dyn.analyze_dae_structure(source, dyn.DAEStructuralPolicy(0, 0))
    adapter = phx.solver.DAECoordinateAdapter(_heat_compilation())

    assert analysis.status == "differentiation-capacity-exceeded"
    assert max(analysis.differentiation_counts) == 1
    with pytest.raises(ValueError, match="differentiation-capacity-exceeded"):
        dyn.compile_acausal_dae(source, dyn.DAEStructuralPolicy(0, 0))
    with pytest.raises(TypeError, match="ReducedDAECompilation"):
        phx.solver.DAECoordinateAdapter(_native_heat_system())  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="block-aligned"):
        phx.solver.DAECoordinateAdapter(
            _heat_compilation(),
            initialization=phx.solver.DAEInitializationSpec.from_masks(
                np.asarray([True, False, True, True]),
                np.asarray([False, True, False, False]),
            ),
        )
    with pytest.raises(ValueError, match="different number of columns"):
        adapter.correction_transfers(
            "stage",
            (
                (
                    "mismatched",
                    (("equations", "cell.balance"),),
                    (("increment", "cell.flux"), ("increment", "cell.temperature")),
                ),
            ),
        )
