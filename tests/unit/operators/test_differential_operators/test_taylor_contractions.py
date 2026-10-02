from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._differentiation import (
    DerivativeRegularity,
    DerivativeSurface,
    GradientLevel,
    RegularityPolicy,
)
from phydrax.operators.differential._jet import jet_series_normalized, jet_terms
from phydrax.operators.differential._taylor_contracts import (
    TaylorContractionPolicy,
    TaylorContractionRequest,
    TaylorContractionResources,
    TaylorContractionStrategy,
)
from phydrax.operators.differential._taylor_execution import (
    evaluate_taylor_contractions,
    TaylorContractionResult,
)
from phydrax.operators.differential._taylor_planning import (
    plan_taylor_contractions,
    TaylorContractionPlan,
    TaylorContractionRecipe,
)


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


def _mixed_polynomial(value: Array) -> Array:
    return value[0] ** 2 * value[1] + value[0] ** 5


def _mixed_seven(value: Array) -> Array:
    return value[0] ** 3 * value[1] ** 4


def _basis() -> dict[str, tuple[Array, ...]]:
    return {
        "x": (jnp.asarray([1.0, 0.0], dtype=jnp.float32),),
        "y": (jnp.asarray([0.0, 1.0], dtype=jnp.float32),),
    }


def test_normalized_arbitrary_series_preserves_derivative_scaled_helpers() -> None:
    def square(value: Array) -> Array:
        return value**2

    point = jnp.asarray(2.0, dtype=jnp.float32)
    one = jnp.asarray(1.0, dtype=jnp.float32)
    primal, normalized = jet_series_normalized(square, (point,), ((one, one),))
    _, derivative_scaled = jet_terms(square, point, one, order=2)
    np.testing.assert_allclose(primal, 4.0, rtol=2e-5)
    np.testing.assert_allclose(jnp.stack(normalized), [4.0, 5.0], rtol=2e-5)
    np.testing.assert_allclose(jnp.stack(derivative_scaled), [4.0, 2.0], rtol=2e-5)


@pytest.mark.parametrize(
    "wrong_shape", [False, True], ids=["empty-series", "coefficient-layout"]
)
def test_normalized_series_refuses_invalid_layout(wrong_shape: bool) -> None:
    def square(value: Array) -> Array:
        return value**2

    point = jnp.asarray(2.0, dtype=jnp.float32)
    terms = (jnp.ones(2, dtype=jnp.float32),) if wrong_shape else ()
    with pytest.raises(
        ValueError, match="shape and dtype" if wrong_shape else "same positive order"
    ):
        jet_series_normalized(square, (point,), (terms,))


@pytest.mark.parametrize("strategy", ["linear", "single", "prime"])
def test_mixed_seven_normalization_against_polynomial_reference(
    strategy: TaylorContractionStrategy,
) -> None:
    request = TaylorContractionRequest(("x", "y"), (3, 4))
    plan = plan_taylor_contractions(
        (request,), policy=TaylorContractionPolicy(strategy=strategy)
    )
    result = evaluate_taylor_contractions(
        _mixed_seven, (jnp.zeros(2, dtype=jnp.float32),), _basis(), plan
    )
    np.testing.assert_allclose(result.values, [144.0], rtol=3e-4, atol=2e-3)
    assert bool(result.derivative_valid[0])


@pytest.mark.parametrize("strategy", ["linear", "single", "prime"])
def test_exact_contraction_excludes_kdv_contamination(
    strategy: TaylorContractionStrategy,
) -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    point = jnp.asarray([0.3, -0.4], dtype=jnp.float32)
    plan = plan_taylor_contractions(
        (request,), policy=TaylorContractionPolicy(strategy=strategy)
    )
    result = evaluate_taylor_contractions(_mixed_polynomial, (point,), _basis(), plan)
    np.testing.assert_allclose(result.values, [2.0], rtol=4e-4, atol=1e-4)


def test_colliding_recipe_cannot_enter_execution() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    with pytest.raises(ValueError, match="colliding"):
        TaylorContractionRecipe(
            request, "single", resources=TaylorContractionResources(), degrees=(1, 3)
        )


def test_output_layout_identity_preserves_order_and_shared_resource_admission() -> None:
    second = TaylorContractionRequest(("v",), (2,), request_id="second")
    fourth = TaylorContractionRequest(("v",), (4,), request_id="fourth")
    policy = TaylorContractionPolicy(
        resources=TaylorContractionResources(
            max_linear_terms=1, max_logical_buffer_elements=64
        )
    )
    plan = plan_taylor_contractions((fourth, second), policy=policy)
    reordered = plan_taylor_contractions((second, fourth), policy=policy)
    assert plan.plan_id != reordered.plan_id
    point = jnp.asarray(2.0, dtype=jnp.float32)

    def quartic(value: Array) -> Array:
        return value**4

    directions = {"v": (jnp.asarray(3.0, dtype=jnp.float32),)}
    result = evaluate_taylor_contractions(quartic, (point,), directions, plan)
    reverse = evaluate_taylor_contractions(quartic, (point,), directions, reordered)
    np.testing.assert_allclose(result.values, [24.0 * 81.0, 12.0 * 4.0 * 9.0], rtol=3e-5)
    np.testing.assert_allclose(reverse.values, result.values[::-1], rtol=3e-5)
    np.testing.assert_allclose(result.primal, 16.0, rtol=3e-5)


def test_request_identity_uses_direction_semantics_not_spelling_order() -> None:
    same = TaylorContractionRequest(("y", "x"), (1, 2))
    canonical = TaylorContractionRequest(("x", "y"), (2, 1))
    assert same.request_id == canonical.request_id
    other = TaylorContractionRequest(("a", "b"), (2, 1))
    assert other.request_id != canonical.request_id


def test_auto_ranks_linear_work_below_higher_order_curve() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    plan = plan_taylor_contractions((request,))
    # Five third-order curves cost 45 order-squared units; order seven costs 49.
    assert plan.recipes[0].strategy == "linear"
    result = evaluate_taylor_contractions(
        _mixed_polynomial, (jnp.asarray([0.3, -0.4], dtype=jnp.float32),), _basis(), plan
    )
    np.testing.assert_allclose(result.values, [2.0], rtol=4e-4, atol=1e-4)


def test_auto_sharing_selection_is_independent_of_output_traversal() -> None:
    mixed = TaylorContractionRequest(("x", "y"), (2, 1), request_id="mixed")
    derivative = TaylorContractionRequest(("x",), (3,), request_id="third")
    # Shared third-order coordinate curve saves one linear evaluation. Request
    # traversal cannot change admission under this union-schedule limit.
    policy = TaylorContractionPolicy(
        resources=TaylorContractionResources(
            max_linear_terms=6, max_logical_buffer_elements=110
        )
    )
    first = plan_taylor_contractions((mixed, derivative), policy=policy)
    second = plan_taylor_contractions((derivative, mixed), policy=policy)
    point = jnp.asarray([0.3, -0.4], dtype=jnp.float32)
    a = evaluate_taylor_contractions(_mixed_polynomial, (point,), _basis(), first)
    b = evaluate_taylor_contractions(_mixed_polynomial, (point,), _basis(), second)
    np.testing.assert_allclose(a.values, [2.0, 60.0 * 0.3**2], rtol=4e-4, atol=1e-4)
    np.testing.assert_allclose(b.values, a.values[::-1], rtol=4e-4, atol=1e-4)


@pytest.mark.parametrize("strategy", ["linear", "single"])
def test_multivariate_arguments_complex_outputs_and_parameter_direction_gradients(
    strategy: TaylorContractionStrategy,
) -> None:
    request = TaylorContractionRequest(("left", "right"), (2, 1))
    plan = plan_taylor_contractions(
        (request,), policy=TaylorContractionPolicy(strategy=strategy)
    )
    first = jnp.asarray(0.2, dtype=jnp.float32)
    second = jnp.asarray(-0.3, dtype=jnp.float32)
    zero = jnp.asarray(0.0, dtype=jnp.float32)

    def evaluate(scale: Array, left: Array, right: Array) -> Array:
        def polynomial(x: Array, y: Array) -> Array:
            base = (scale * x**2 * y).astype(jnp.complex64)
            return jnp.stack((base, jnp.asarray(2j, dtype=jnp.complex64) * base))

        return evaluate_taylor_contractions(
            polynomial,
            (first, second),
            {"left": (left, zero), "right": (zero, right)},
            plan,
        ).values[0]

    scale = jnp.asarray(4.0, dtype=jnp.float32)
    left = jnp.asarray(2.0, dtype=jnp.float32)
    right = jnp.asarray(3.0, dtype=jnp.float32)
    np.testing.assert_allclose(evaluate(scale, left, right), [96.0, 192j], rtol=4e-5)

    def real_value(scale: Array, left: Array, right: Array) -> Array:
        return jnp.real(evaluate(scale, left, right)[0])

    gradients = jax.grad(real_value, argnums=(0, 1, 2))(scale, left, right)
    np.testing.assert_allclose(jnp.stack(gradients), [24.0, 96.0, 32.0], rtol=8e-5)

    def imaginary_value(scale: Array, left: Array, right: Array) -> Array:
        return jnp.imag(evaluate(scale, left, right)[1])

    imaginary_gradients = jax.grad(imaginary_value, argnums=(0, 1, 2))(scale, left, right)
    np.testing.assert_allclose(
        jnp.stack(imaginary_gradients), [48.0, 192.0, 64.0], rtol=8e-5
    )
    compiled = jax.jit(evaluate)
    np.testing.assert_allclose(compiled(scale, left, right), [96.0, 192j], rtol=4e-5)
    np.testing.assert_allclose(
        compiled(scale + 1.0, left, right), [120.0, 240j], rtol=4e-5
    )


def test_declared_polynomial_curved_execution_uses_composed_regularity() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    regularity = DerivativeRegularity.smooth(
        degree_bound=3, conditions=("declared-support",)
    )
    plan = plan_taylor_contractions(
        (request,),
        regularity=regularity,
        policy=TaylorContractionPolicy(strategy="single"),
    )

    def cubic(value: Array) -> Array:
        return value[0] ** 2 * value[1]

    result = evaluate_taylor_contractions(
        cubic,
        (jnp.asarray([0.3, -0.4], dtype=jnp.float32),),
        _basis(),
        plan,
    )
    np.testing.assert_allclose(result.values, [2.0], rtol=4e-4, atol=1e-4)
    assert bool(result.derivative_valid[0])
    assert result.evidence[0].supported
    assert result.evidence[0].request.order == 7
    assert result.evidence[0].conditions == ("declared-support",)


def test_global_polynomial_high_order_contraction_is_supported_zero() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 2))
    plan = plan_taylor_contractions(
        (request,),
        regularity=DerivativeRegularity.smooth(
            degree_bound=3, conditions=("declared-support",)
        ),
        policy=TaylorContractionPolicy(strategy="single"),
    )

    def cubic(value: Array) -> Array:
        return value[0] ** 2 * value[1]

    result = evaluate_taylor_contractions(
        cubic, (jnp.asarray([0.3, -0.4], dtype=jnp.float32),), _basis(), plan
    )
    np.testing.assert_allclose(result.values, [0.0], atol=1e-6)
    assert bool(result.derivative_valid[0])
    assert plan.requested_admissions[0].request.order == 4
    assert (
        plan.requested_admissions[0].level(DerivativeSurface.INPUT)
        is GradientLevel.SMOOTH
    )
    assert result.evidence[0].request.order == plan.required_regularity_order
    assert result.evidence[0].level(DerivativeSurface.INPUT) is GradientLevel.SMOOTH
    assert result.evidence[0].conditions == ("declared-support",)


def test_piecewise_polynomial_refuses_degree_degenerate_requested_order() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 2))
    with pytest.raises(ValueError, match="regularity"):
        plan_taylor_contractions(
            (request,),
            regularity=DerivativeRegularity.piecewise_polynomial(
                continuity=2, degree_bound=3
            ),
            regularity_policy=RegularityPolicy(allow_almost_everywhere=True),
            policy=TaylorContractionPolicy(strategy="linear"),
        )


def test_regularity_admission_selects_admitted_execution_order() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    plan = plan_taylor_contractions(
        (request,), regularity=DerivativeRegularity.piecewise_smooth(continuity=3)
    )
    assert plan.required_regularity_order == 3
    result = evaluate_taylor_contractions(
        _mixed_polynomial, (jnp.zeros(2, dtype=jnp.float32),), _basis(), plan
    )
    np.testing.assert_allclose(result.values, [2.0], rtol=4e-4, atol=1e-4)
    assert bool(result.derivative_valid[0])


@pytest.mark.parametrize(
    "strategy,continuity",
    [("single", 3), ("linear", 2)],
    ids=["executed-order", "requested-order"],
)
def test_regularity_refuses_unsupported_derivative_order(
    strategy: TaylorContractionStrategy, continuity: int
) -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    with pytest.raises(ValueError, match="admitted"):
        plan_taylor_contractions(
            (request,),
            regularity=DerivativeRegularity.piecewise_smooth(continuity=continuity),
            policy=TaylorContractionPolicy(strategy=strategy),
        )


@pytest.mark.parametrize("conditions", [(), ("declared-support",)])
def test_almost_everywhere_permission_preserves_source_conditions(
    conditions: tuple[str, ...],
) -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    plan = plan_taylor_contractions(
        (request,),
        regularity=DerivativeRegularity.piecewise_smooth(
            continuity=2, conditions=conditions
        ),
        regularity_policy=RegularityPolicy(allow_almost_everywhere=True),
        policy=TaylorContractionPolicy(strategy="linear"),
    )
    result = evaluate_taylor_contractions(
        _mixed_polynomial, (jnp.zeros(2, dtype=jnp.float32),), _basis(), plan
    )
    np.testing.assert_allclose(result.values, [2.0], rtol=4e-4, atol=1e-4)
    assert bool(result.derivative_valid[0])
    assert plan.requested_admissions[0].request.order == 3
    assert result.evidence[0].request.order == 3
    assert (
        result.evidence[0].level(DerivativeSurface.INPUT)
        is GradientLevel.ALMOST_EVERYWHERE
    )
    assert result.evidence[0].conditions == conditions
    assert plan.requested_admissions[0].conditions == conditions


@pytest.mark.parametrize(
    "binding",
    ["missing", "shape", "arguments"],
    ids=["direction-ids", "component-shape", "argument-count"],
)
def test_binding_layout_refusals(binding: str) -> None:
    plan = plan_taylor_contractions((TaylorContractionRequest(("x", "y"), (2, 1)),))
    point = jnp.zeros(2, dtype=jnp.float32)
    directions = _basis()
    if binding == "missing":
        directions.pop("y")
        message = "exactly match"
    elif binding == "shape":
        directions["x"] = (jnp.ones(3, dtype=jnp.float32),)
        message = "shapes"
    else:
        directions["x"] = (point, point)
        message = "component per primal"
    with pytest.raises(ValueError, match=message):
        evaluate_taylor_contractions(_mixed_polynomial, (point,), directions, plan)


def test_complex_primal_is_refused() -> None:
    plan = plan_taylor_contractions((TaylorContractionRequest(("x", "y"), (2, 1)),))
    with pytest.raises(TypeError, match="floating"):
        evaluate_taylor_contractions(
            _mixed_polynomial, (jnp.zeros(2, dtype=jnp.complex64),), _basis(), plan
        )


def test_complex_direction_is_refused_before_input_dtype_conversion() -> None:
    plan = plan_taylor_contractions((TaylorContractionRequest(("x", "y"), (2, 1)),))
    directions = _basis()
    directions["x"] = (jnp.asarray([1j, 0j], dtype=jnp.complex64),)
    with pytest.raises(TypeError, match="floating"):
        evaluate_taylor_contractions(
            _mixed_polynomial, (jnp.zeros(2, dtype=jnp.float32),), directions, plan
        )


def test_host_direction_dtype_binding_preserves_input_precision() -> None:
    plan = plan_taylor_contractions((TaylorContractionRequest(("v",), (2,)),))

    def square(value: Array) -> Array:
        return value**2

    result = evaluate_taylor_contractions(
        square,
        (jnp.asarray(2.0, dtype=jnp.float32),),
        {"v": (np.asarray(3.0, dtype=np.float64),)},
        plan,
    )
    np.testing.assert_allclose(result.values, [18.0], rtol=3e-5)
    assert result.values.dtype == jnp.dtype(jnp.float32)


def test_input_workset_resources_are_refused() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    plan = plan_taylor_contractions(
        (request,),
        policy=TaylorContractionPolicy(
            strategy="linear",
            resources=TaylorContractionResources(max_logical_buffer_elements=2),
        ),
    )
    with pytest.raises(ValueError, match="input worksets"):
        evaluate_taylor_contractions(
            _mixed_polynomial, (jnp.zeros(2, dtype=jnp.float32),), _basis(), plan
        )


def test_linear_preparation_resources_are_refused() -> None:
    with pytest.raises(ValueError, match="resources"):
        plan_taylor_contractions(
            (TaylorContractionRequest(("x", "y"), (2, 1)),),
            policy=TaylorContractionPolicy(
                strategy="linear",
                resources=TaylorContractionResources(max_linear_terms=2),
            ),
        )


def test_request_id_collision_is_refused() -> None:
    first = TaylorContractionRequest(("x",), (1,), request_id="collision")
    second = TaylorContractionRequest(("y",), (1,), request_id="collision")
    with pytest.raises(ValueError, match="request ID"):
        plan_taylor_contractions((first, second))


def test_duplicate_direction_identity_is_refused() -> None:
    with pytest.raises(ValueError, match="once"):
        TaylorContractionRequest(("x", "x"), (1, 1))


def test_plan_reconstruction_recognizes_stricter_resource_policy() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    recipe = TaylorContractionRecipe(
        request, "single", resources=TaylorContractionResources(), degrees=(2, 3)
    )
    with pytest.raises(ValueError, match="resources"):
        TaylorContractionPlan(
            (recipe,),
            policy=TaylorContractionPolicy(
                resources=TaylorContractionResources(max_order=3)
            ),
        )


def test_semantic_planning_key_is_reproducible() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    policy = TaylorContractionPolicy(strategy="prime")
    first = plan_taylor_contractions((request,), policy=policy, key=jax.random.key(0))
    second = plan_taylor_contractions((request,), policy=policy, key=jax.random.key(0))
    point = jnp.zeros(2, dtype=jnp.float32)
    a = evaluate_taylor_contractions(_mixed_polynomial, (point,), _basis(), first)
    b = evaluate_taylor_contractions(_mixed_polynomial, (point,), _basis(), second)
    np.testing.assert_allclose(a.values, b.values, rtol=0.0, atol=0.0)
    assert first.plan_id == second.plan_id


def test_legacy_planning_key_layout_is_refused() -> None:
    request = TaylorContractionRequest(("x", "y"), (2, 1))
    with pytest.raises(TypeError, match="typed JAX key"):
        plan_taylor_contractions((request,), key=jnp.zeros(2, dtype=jnp.uint32))


def test_output_workset_resources_are_refused_before_derivative_execution() -> None:
    request = TaylorContractionRequest(("v",), (2,))
    plan = plan_taylor_contractions(
        (request,),
        policy=TaylorContractionPolicy(
            resources=TaylorContractionResources(max_logical_buffer_elements=128)
        ),
    )

    def wide(value: Array) -> Array:
        return jnp.ones(64, dtype=jnp.float32) * value**2

    with pytest.raises(ValueError, match="output worksets"):
        evaluate_taylor_contractions(
            wide,
            (jnp.asarray(0.0, dtype=jnp.float32),),
            {"v": (jnp.asarray(1.0, dtype=jnp.float32),)},
            plan,
        )


def test_nonfinite_derivative_evidence_reaches_result() -> None:
    plan = plan_taylor_contractions((TaylorContractionRequest(("v",), (2,)),))

    def overflow(value: Array) -> Array:
        return jnp.exp(value)

    result = evaluate_taylor_contractions(
        overflow,
        (jnp.asarray(1000.0, dtype=jnp.float32),),
        {"v": (jnp.asarray(1.0, dtype=jnp.float32),)},
        plan,
    )
    assert not bool(result.finite[0])
    assert not bool(result.derivative_valid[0])


def test_workset_policy_identity_does_not_change_numerical_values() -> None:
    requests = (TaylorContractionRequest(("x", "y"), (2, 1)),)
    point = jnp.asarray([0.3, -0.4], dtype=jnp.float32)
    plans = tuple(
        plan_taylor_contractions(
            requests,
            policy=TaylorContractionPolicy(
                strategy="linear", resources=TaylorContractionResources(workset_size=size)
            ),
        )
        for size in (1, 3)
    )
    assert plans[0].plan_id != plans[1].plan_id
    for size, plan in zip((1, 3), plans, strict=True):
        output = evaluate_taylor_contractions(
            _mixed_polynomial, (point,), _basis(), plan
        ).values
        np.testing.assert_allclose(
            output, [2.0], rtol=3e-4, atol=1e-4, err_msg=f"workset_size={size}"
        )


@pytest.mark.parametrize(
    "leaf",
    ["values", "finite", "derivative_valid"],
    ids=["event-layout", "finite-dtype", "valid-extent"],
)
def test_result_constructor_refuses_inconsistent_numerical_contract(leaf: str) -> None:
    plan = plan_taylor_contractions((TaylorContractionRequest(("v",), (1,)),))
    primal = jnp.zeros(2, dtype=jnp.float32)
    values = jnp.zeros((1, 3) if leaf == "values" else (1, 2), dtype=jnp.float32)
    finite = jnp.ones(1, dtype=jnp.int32 if leaf == "finite" else jnp.bool_)
    valid = jnp.ones(2 if leaf == "derivative_valid" else 1, dtype=jnp.bool_)
    with pytest.raises(TypeError if leaf == "finite" else ValueError):
        TaylorContractionResult(primal, values, finite, valid, plan)


@pytest.mark.parametrize(
    ("dtype", "counts", "strategy"),
    [
        (jnp.float32, (35,), "single"),
        (jnp.float32, (34,), "single"),
        (jnp.float16, (8,), "single"),
        (jnp.float16, (20, 1), "linear"),
    ],
    ids=[
        "factorial-overflow",
        "float32-reciprocal-subnormal",
        "float16-reciprocal-subnormal",
        "signed-binomial-weights",
    ],
)
def test_precision_refuses_unrepresentable_normalization_before_execution(
    dtype: type,
    counts: tuple[int, ...],
    strategy: TaylorContractionStrategy,
) -> None:
    identifiers = tuple(f"direction-{index}" for index in range(len(counts)))
    plan = plan_taylor_contractions(
        (TaylorContractionRequest(identifiers, counts),),
        policy=TaylorContractionPolicy(strategy=strategy),
    )
    point = jnp.zeros(len(counts), dtype=dtype)
    directions = {
        identifier: (jax.nn.one_hot(index, len(counts), dtype=dtype),)
        for index, identifier in enumerate(identifiers)
    }
    with pytest.raises(ValueError, match="dtype.*range"):
        evaluate_taylor_contractions(jnp.sum, (point,), directions, plan)


def test_wide_precision_preserves_high_order_exponential_derivative() -> None:
    plan = plan_taylor_contractions((TaylorContractionRequest(("v",), (35,)),))
    point = jnp.asarray(0.0, dtype=jnp.float64)
    direction = jnp.asarray(1.0, dtype=jnp.float64)
    result = evaluate_taylor_contractions(jnp.exp, (point,), {"v": (direction,)}, plan)
    np.testing.assert_allclose(result.values, [1.0], rtol=2e-12)
