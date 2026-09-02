import jax
import jax.numpy as jnp
import pytest

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


def _pendulum_like_source():
    variables = (
        DAEVariableBlock("q", (), 2, 1.0),
        DAEVariableBlock("lambda", (), 0, 1.0),
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


def test_declared_pantelides_reduction_reports_conditional_index_three():
    source = _pendulum_like_source()
    policy = DAEStructuralPolicy(2, 1)
    analysis = analyze_dae_structure(source, policy)
    assert analysis.successful
    assert analysis.structural_index == 3
    compilation = compile_acausal_dae(source, policy)
    state = jnp.asarray([0.0, 1.0, 0.0])
    rate = jnp.asarray([0.0, 0.0, -1.0])
    residual = jax.jit(compilation.system.residual)(0.0, state, rate, None)
    assert residual.shape == state.shape
    assert compilation.residual_audit(0.0, state, rate).shape == (2,)


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
