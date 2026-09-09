import jax
import jax.numpy as jnp
import pytest

from phydrax.dynamics import CallableInputPolicy, HeldInputPolicy, InputLayout
from phydrax.dynamics._dae_structural import (
    AcausalDAESource,
    analyze_dae_structure,
    compile_acausal_dae,
    DAEComponent,
    DAEDerivativeIncidence,
    DAEEquationBlock,
    DAEStructuralPolicy,
    DAEVariableBlock,
)
from phydrax.solver import DifferentialAlgebraicProblem


def _pendulum_like_source():
    variables = (
        DAEVariableBlock("q", (), 2, state_scale=1.0, rate_scale=1.0),
        DAEVariableBlock("lambda", (), 0, state_scale=1.0, rate_scale=1.0),
    )
    equations = (
        DAEEquationBlock(
            "dynamics",
            lambda time, jet, args: jet.value("q", 2) + jet.value("lambda"),
            (
                DAEDerivativeIncidence("q", 2),
                DAEDerivativeIncidence("lambda", 0),
            ),
        ),
        DAEEquationBlock(
            "constraint",
            lambda time, jet, args: jet.value("q") ** 2 - 1.0,
            (DAEDerivativeIncidence("q", 0),),
        ),
    )
    return AcausalDAESource((DAEComponent("body", variables, equations),))


def test_index_three_original_constraint_audit_detects_inconsistent_state():
    source = _pendulum_like_source()
    policy = DAEStructuralPolicy(2, 1)
    compilation = compile_acausal_dae(source, policy)
    state = jnp.asarray([0.0, 1.0, 0.0])
    rate = jnp.zeros(3)
    residual = jax.jit(compilation.system.residual)(0.0, state, rate, None)
    assert jnp.allclose(residual, 0)
    assert jnp.allclose(compilation.residual_audit(0.0, state, rate), 0)
    # Differentiating a holonomic constraint does not establish its initial
    # invariant: a stationary off-manifold state must fail the original audit.
    inconsistent = state.at[1].set(1.1)
    assert jnp.allclose(compilation.system.residual(0.0, inconsistent, rate, None), 0)
    assert jnp.max(jnp.abs(compilation.residual_audit(0.0, inconsistent, rate))) > 0.2


def test_index_one_lowering_preserves_physical_flow_and_state_derivative():
    component = DAEComponent(
        "decay",
        (DAEVariableBlock("a_flow", (), 0), DAEVariableBlock("z_state", (), 1)),
        (
            DAEEquationBlock(
                "balance",
                lambda time, jet, args: jet.value("z_state", 1) - jet.value("a_flow"),
                (DAEDerivativeIncidence("z_state", 1), DAEDerivativeIncidence("a_flow")),
            ),
            DAEEquationBlock(
                "constitutive",
                lambda time, jet, args: jet.value("a_flow") + jet.value("z_state"),
                (DAEDerivativeIncidence("a_flow"), DAEDerivativeIncidence("z_state")),
            ),
        ),
    )
    compilation = compile_acausal_dae(
        AcausalDAESource((component,)), DAEStructuralPolicy(1, 0, tearing="none")
    )
    # x=1, flow=-1, dx/dt=-1 is a physical jet of dx/dt=-x.
    state, rate = jnp.asarray([-1.0, 1.0]), jnp.asarray([0.0, -1.0])
    assert jnp.allclose(jax.jit(compilation.system.residual)(0.0, state, rate, None), 0)
    assert jnp.allclose(compilation.residual_audit(0.0, state, rate), 0)


def test_structural_failure_names_unmatched_variables_and_capacity():
    component = DAEComponent(
        "singular",
        (DAEVariableBlock("x"), DAEVariableBlock("y")),
        (
            DAEEquationBlock(
                "one",
                lambda time, jet, args: jet.value("x", 1),
                (DAEDerivativeIncidence("x", 1),),
            ),
        ),
    )
    analysis = analyze_dae_structure(
        AcausalDAESource((component,)),
        DAEStructuralPolicy(1, 0),
    )
    assert not analysis.successful
    assert analysis.unmatched_variables == ("singular.y",)

    with pytest.raises(ValueError, match="differentiation-capacity"):
        compile_acausal_dae(_pendulum_like_source(), DAEStructuralPolicy(1, 1))


def test_missing_declared_jvp_incidence_fails_compilation():
    component = DAEComponent(
        "bad",
        (DAEVariableBlock("x", (), 1),),
        (
            DAEEquationBlock(
                "equation",
                lambda time, jet, args: jet.value("x") + jet.value("x", 1),
                (DAEDerivativeIncidence("x", 1),),
            ),
        ),
    )
    with pytest.raises(ValueError, match="undeclared JVP incidence"):
        compile_acausal_dae(
            AcausalDAESource((component,)),
            DAEStructuralPolicy(1, 0),
        )


def test_input_aware_structural_dae_propagates_independent_scales_and_jits():
    layout = InputLayout(
        (2,),
        axes=("input",),
        component_names=("bias", "gain"),
        roles="forcing",
    )
    component = DAEComponent(
        "driven",
        (
            DAEVariableBlock(
                "x",
                state_scale=2.0,
                rate_scale=3.0,
            ),
        ),
        (
            DAEEquationBlock(
                "balance",
                lambda time, jet, inputs, args: (
                    jet.value("x", 1) - inputs[1] * jet.value("x") - inputs[0]
                ),
                (
                    DAEDerivativeIncidence("x", 0),
                    DAEDerivativeIncidence("x", 1),
                ),
                residual_scale=5.0,
            ),
        ),
    )
    source = AcausalDAESource((component,), input_layout=layout)
    compilation = compile_acausal_dae(source, DAEStructuralPolicy(0, 0))

    assert compilation.system.input_layout is layout
    assert jnp.array_equal(compilation.system.state_scale, jnp.asarray([2.0]))
    assert jnp.array_equal(compilation.system.state_rate_scale, jnp.asarray([3.0]))
    assert jnp.array_equal(compilation.system.residual_scale, jnp.asarray([5.0]))

    residual = jax.jit(
        lambda values: compilation.system.evaluate(
            0.0,
            jnp.asarray([2.0]),
            jnp.asarray([7.0]),
            None,
            inputs=values,
        )
    )(jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(residual, jnp.asarray([2.0]))
    assert jnp.allclose(
        compilation.residual_audit(
            0.0,
            jnp.asarray([2.0]),
            jnp.asarray([7.0]),
            jnp.asarray([1.0, 2.0]),
        ),
        residual,
    )

    callable_policy = CallableInputPolicy(
        lambda time, state, args: jnp.asarray([args, 2.0]),
        input_layout=layout,
        policy_id="structural-callable",
    )
    callable_problem = DifferentialAlgebraicProblem(
        compilation.system,
        jnp.asarray([2.0]),
        input_policy=callable_policy,
        args=1.0,
        problem_id="structural-callable-problem",
    )
    assert callable_problem.input_policy is callable_policy

    held_policy = HeldInputPolicy(
        jnp.asarray([0.0, 1.0, 2.0]),
        jnp.asarray([[1.0, 2.0], [3.0, 4.0]]),
        input_layout=layout,
        policy_id="structural-held",
    )
    held_problem = DifferentialAlgebraicProblem(
        compilation.system,
        jnp.asarray([2.0]),
        input_policy=held_policy,
        problem_id="structural-held-problem",
    )
    assert held_problem.input_policy is held_policy


def test_structural_input_and_residual_scale_shape_mismatches_fail_preparation():
    layout = InputLayout(
        (2,),
        component_names=("bias", "gain"),
        roles="forcing",
    )
    component = DAEComponent(
        "vector",
        (
            DAEVariableBlock(
                "x",
                (2,),
                0,
                state_scale=jnp.ones((2,)),
                rate_scale=jnp.ones((2,)),
            ),
        ),
        (
            DAEEquationBlock(
                "balance",
                lambda time, jet, inputs, args: jet.value("x"),
                (DAEDerivativeIncidence("x", 0),),
                residual_scale=jnp.ones((1,)),
            ),
        ),
    )
    source = AcausalDAESource((component,), input_layout=layout)
    with pytest.raises(ValueError, match="inputs must have shape"):
        compile_acausal_dae(
            source,
            DAEStructuralPolicy(0, 0),
            inputs=jnp.ones((1,)),
        )
    with pytest.raises(ValueError, match="residual_scale"):
        compile_acausal_dae(source, DAEStructuralPolicy(0, 0))

    reversed_layout = InputLayout(
        (2,),
        component_names=("gain", "bias"),
        roles="forcing",
    )
    mismatched = CallableInputPolicy(
        lambda time, state, args: jnp.ones((2,)),
        input_layout=reversed_layout,
        policy_id="structural-mismatched",
    )
    valid_component = DAEComponent(
        "scalar",
        (DAEVariableBlock("x"),),
        (
            DAEEquationBlock(
                "balance",
                lambda time, jet, inputs, args: jet.value("x", 1),
                (DAEDerivativeIncidence("x", 1),),
            ),
        ),
    )
    valid = compile_acausal_dae(
        AcausalDAESource((valid_component,), input_layout=layout),
        DAEStructuralPolicy(0, 0),
    )
    with pytest.raises(ValueError, match="layout must exactly match"):
        DifferentialAlgebraicProblem(
            valid.system,
            jnp.zeros((1,)),
            input_policy=mismatched,
            problem_id="structural-layout-mismatch",
        )
