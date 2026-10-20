#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))


def _band_pattern(size: int, *, symmetric: bool = False) -> phx.sparse.SparsePattern:
    rows = []
    cols = []
    for row in range(size):
        for column in range(max(0, row - 1), min(size, row + 2)):
            rows.append(row)
            cols.append(column)
    return phx.sparse.SparsePattern.from_coo(
        np.asarray(rows, dtype=np.int32),
        np.asarray(cols, dtype=np.int32),
        (size, size),
        symmetric=symmetric,
    )


def _band_vector_function(value: Any, arguments: Any) -> Any:
    return jnp.asarray(
        [
            value[0] ** 2 + arguments[0] * value[1],
            value[0] * value[1] + value[1] ** 2 + value[2],
            value[1] + jnp.sin(value[2]) + value[2] * value[3],
            value[2] ** 2 + arguments[1] * value[3],
        ]
    )


def _band_scalar_function(value: Any, arguments: Any) -> Any:
    return (
        value[0] ** 2
        + arguments[0] * value[0] * value[1]
        + value[1] ** 2
        + value[1] * value[2]
        + jnp.sin(value[2])
        + arguments[1] * value[2] * value[3]
        + value[3] ** 2
    )


def test_sparse_contracts() -> None:
    first = phx.sparse.SparsePattern.from_coo(
        jnp.asarray([2, 0, 1, 0, 2, 0]),
        jnp.asarray([1, 0, 1, 2, 1, 2]),
        (3, 3),
        origin="structural",
    )
    second = phx.sparse.SparsePattern.from_coo(
        jnp.asarray([0, 0, 1, 2]),
        jnp.asarray([0, 2, 1, 1]),
        (3, 3),
        origin="structural",
    )

    assert np.array_equal(np.asarray(first.rows), np.asarray([0, 0, 1, 2]))
    assert np.array_equal(np.asarray(first.cols), np.asarray([0, 2, 1, 1]))
    assert first.nnz == 4
    assert first.shape == (3, 3)
    assert first.pattern_id == second.pattern_id
    assert first.to_dict() == second.to_dict()
    assert (
        phx.sparse.SparsePattern.from_dict(
            json.loads(json.dumps(first.to_dict()))
        ).pattern_id
        == first.pattern_id
    )

    corrupted = first.to_dict()
    corrupted["rows"] = [0, 1, 1, 2]
    with pytest.raises(ValueError, match="fingerprint"):
        phx.sparse.SparsePattern.from_dict(corrupted)
    invalid_schema = first.to_dict()
    invalid_schema["schema_version"] = 1.9
    with pytest.raises(ValueError, match="schema"):
        phx.sparse.SparsePattern.from_dict(invalid_schema)

    invalid_shape = first.to_dict()
    invalid_shape["shape"] = [3.0, 3]
    with pytest.raises(ValueError, match="integer dimensions"):
        phx.sparse.SparsePattern.from_dict(invalid_shape)
    with pytest.raises(ValueError, match="equal shape"):
        phx.sparse.SparsePattern.from_coo([0], [0, 1], (2, 2))
    with pytest.raises(TypeError, match="integer dtype"):
        phx.sparse.SparsePattern.from_coo(np.asarray([0.0]), [0], (2, 2))
    with pytest.raises(ValueError, match="rows lie outside"):
        phx.sparse.SparsePattern.from_coo([-1], [0], (2, 2))
    with pytest.raises(ValueError, match="columns lie outside"):
        phx.sparse.SparsePattern.from_coo([0], [2], (2, 2))
    with pytest.raises(ValueError, match="fit in int32"):
        phx.sparse.SparsePattern.from_coo(
            [0],
            [2**31],
            (1, 2**31 + 1),
        )
    with pytest.raises(ValueError, match="must be square"):
        phx.sparse.SparsePattern.from_coo([], [], (2, 3), symmetric=True)
    with pytest.raises(ValueError, match="transpose entry"):
        phx.sparse.SparsePattern.from_coo([0, 1], [0, 0], (2, 2), symmetric=True)
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    point = jnp.asarray([1.0, 2.0])
    pattern = phx.sparse.SparsePattern.from_coo([0, 1], [0, 1], (2, 2))
    plan = phx.sparse.compile_sparse_jacobian(
        lambda value, arguments: arguments["scale"][0] * value,
        point,
        source=space,
        target=space,
        sample_args={"scale": jnp.asarray([2.0])},
        structure=pattern,
        compiler="native",
    )
    with pytest.raises(ValueError, match="PyTree structure"):
        plan.coefficients(point, {"other": jnp.asarray([2.0])})
    with pytest.raises(ValueError, match="shape and dtype"):
        plan.coefficients(point, {"scale": jnp.asarray([2.0, 3.0])})
    with pytest.raises(ValueError, match="declared pattern"):
        phx.sparse.compile_sparse_jacobian(
            lambda value, _: value,
            point,
            source=space,
            target=space,
            compiler="native",
        )
    with pytest.raises(ValueError):
        phx.sparse.compile_sparse_hessian(
            lambda value, _: value,
            point,
            space=space,
            structure=phx.sparse.SparsePattern.from_coo(
                [0, 1], [0, 1], (2, 2), symmetric=True
            ),
            compiler="native",
        )
    complex_space = phx.linalg.ArraySpace((2,), dtype=jnp.complex128)
    with pytest.raises(ValueError):
        phx.sparse.compile_sparse_jacobian(
            lambda value, _: value,
            point.astype(jnp.complex128),
            source=complex_space,
            target=complex_space,
            structure=pattern,
            compiler="native",
        )
    source = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    point = jnp.asarray([0.2, -0.5])
    matrix = jnp.asarray([[2.0, 0.0], [1.0, -1.0], [0.0, 3.0]])
    jacobian_pattern = phx.sparse.SparsePattern.from_coo(
        [0, 1, 1, 2], [0, 0, 1, 1], (3, 2)
    )
    jacobian = phx.sparse.compile_sparse_jacobian(
        lambda value, _: matrix @ value,
        point,
        source=source,
        target=target,
        structure=jacobian_pattern,
        compiler="native",
    ).operator(point)
    observations = jnp.asarray([1.0, -2.0, 0.5])
    least_squares = phx.linalg.solve(
        phx.linalg.LeastSquaresProblem(jacobian),
        observations,
    )
    assert bool(least_squares.successful)
    assert jnp.allclose(
        least_squares.value,
        jnp.linalg.lstsq(matrix, observations, rcond=None)[0],
    )

    hessian_pattern = phx.sparse.SparsePattern.from_coo(
        [0, 0, 1, 1], [0, 1, 0, 1], (2, 2), symmetric=True
    )
    hessian_plan = phx.sparse.compile_sparse_hessian(
        lambda value, _: (
            2.0 * value[0] ** 2
            + value[0] * value[1]
            + value[1] ** 2
            + 0.25 * value[0] ** 4
        ),
        point,
        space=source,
        structure=hessian_pattern,
        compiler="native",
        contract=phx.sparse.SparseHessianContract("riesz"),
        properties=phx.linalg.OperatorProperties(
            self_adjoint=True,
            positive_definite=True,
            evidence={
                "self_adjoint": "asserted",
                "positive_definite": "asserted",
                "positive_semidefinite": "asserted",
            },
        ),
    )
    hessian = hessian_plan.operator(point)
    right_hand_side = jnp.asarray([1.0, -3.0])
    solved = phx.linalg.solve(phx.linalg.LinearSystem(hessian), right_hand_side)
    assert bool(solved.successful)
    assert jnp.allclose(
        solved.value,
        jnp.linalg.solve(hessian.as_dense(), right_hand_side),
    )
    prepared = phx.linalg.prepare(phx.linalg.LinearSystem(hessian))
    updated_point = jnp.asarray([0.8, -0.1])
    updated_hessian = hessian_plan.operator(updated_point)
    refreshed = phx.linalg.refresh(
        prepared,
        phx.linalg.LinearSystem(updated_hessian),
    )
    refreshed_result = phx.linalg.solve(refreshed, right_hand_side)
    assert refreshed.numeric_version == prepared.numeric_version + 1
    assert refreshed.plan.plan_id == prepared.plan.plan_id
    assert jnp.allclose(
        refreshed_result.value,
        jnp.linalg.solve(updated_hessian.as_dense(), right_hand_side),
    )


def test_sparse_derivatives_scenario_1() -> None:
    pattern = _band_pattern(5)
    first = phx.sparse.compile_sparse_jacobian(
        lambda value, _: value,
        jnp.ones((5,)),
        source=phx.linalg.ArraySpace((5,)),
        target=phx.linalg.ArraySpace((5,)),
        structure=pattern,
        compiler="native",
        mode="fwd",
    ).coloring
    second = phx.sparse.compile_sparse_jacobian(
        lambda value, _: value,
        jnp.ones((5,)),
        source=phx.linalg.ArraySpace((5,)),
        target=phx.linalg.ArraySpace((5,)),
        structure=pattern,
        compiler="native",
        mode="fwd",
    ).coloring

    assert first.num_colors == 3
    assert first.coloring_id == second.coloring_id
    assert np.array_equal(np.asarray(first.colors), np.asarray(second.colors))
    restored = phx.sparse.SparseColoring.from_dict(
        json.loads(json.dumps(first.to_dict()))
    )
    assert restored.coloring_id == first.coloring_id
    assert restored.pattern.pattern_id == pattern.pattern_id

    corrupted = first.to_dict()
    corrupted["gather_elements"][0] = pattern.target_size
    with pytest.raises(ValueError, match="compressed coordinate"):
        phx.sparse.SparseColoring.from_dict(corrupted)
    invalid_count = first.to_dict()
    invalid_count["num_colors"] = float(first.num_colors)
    with pytest.raises(ValueError, match="non-negative integer"):
        phx.sparse.SparseColoring.from_dict(invalid_count)
    invalid_color = first.to_dict()
    invalid_color["colors"][0] = 2**31
    with pytest.raises(ValueError, match="fit in int32"):
        phx.sparse.SparseColoring.from_dict(invalid_color)
    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4, 1.5])
    arguments = jnp.asarray([1.3, -0.8])
    runtime_arguments = jnp.asarray([-0.2, 2.1])
    pattern = _band_pattern(4)
    expected = jax.jacfwd(_band_vector_function)(point, runtime_arguments)

    for mode in ("fwd", "rev"):
        unchunked = phx.sparse.compile_sparse_jacobian(
            _band_vector_function,
            point,
            source=space,
            target=space,
            sample_args=arguments,
            structure=pattern,
            compiler="native",
            mode=mode,
        )
        chunked = phx.sparse.compile_sparse_jacobian(
            _band_vector_function,
            point,
            source=space,
            target=space,
            sample_args=arguments,
            structure=pattern,
            compiler="native",
            mode=mode,
            chunk_size=1,
        )
        coefficients = jax.jit(
            lambda value, dynamic: chunked.coefficients(value, dynamic)
        )(point, runtime_arguments)
        assert jnp.allclose(
            chunked.operator(point, runtime_arguments).as_dense(), expected
        )
        assert jnp.allclose(
            coefficients,
            unchunked.coefficients(point, runtime_arguments),
        )
        batched = jax.vmap(lambda value: chunked.coefficients(value, runtime_arguments))(
            jnp.stack((point, point + 0.1))
        )
        assert batched.shape == (2, pattern.nnz)
        point_gradient = jax.grad(
            lambda value: jnp.sum(chunked.coefficients(value, runtime_arguments))
        )(point)
        argument_gradient = jax.grad(
            lambda dynamic: jnp.sum(chunked.coefficients(point, dynamic))
        )(runtime_arguments)
        assert jnp.all(jnp.isfinite(point_gradient))
        assert jnp.all(jnp.isfinite(argument_gradient))
    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4, 1.5])
    arguments = jnp.asarray([1.3, -0.8])
    runtime_arguments = jnp.asarray([-0.2, 2.1])
    pattern = _band_pattern(4, symmetric=True)
    expected = jax.hessian(_band_scalar_function)(point, runtime_arguments)

    for mode in ("fwd_over_rev", "rev_over_fwd", "rev_over_rev"):
        plan = phx.sparse.compile_sparse_hessian(
            _band_scalar_function,
            point,
            space=space,
            sample_args=arguments,
            structure=pattern,
            compiler="native",
            mode=mode,
            chunk_size=2,
        )
        coefficients = jax.jit(lambda value, dynamic: plan.coefficients(value, dynamic))(
            point, runtime_arguments
        )
        assert coefficients.shape == (pattern.nnz,)
        assert jnp.allclose(plan.operator(point, runtime_arguments).as_dense(), expected)
        third_order = jax.grad(
            lambda value: jnp.sum(plan.coefficients(value, runtime_arguments))
        )(point)
        assert jnp.all(jnp.isfinite(third_order))
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    point = jnp.asarray([1.5, -0.5])
    pattern = phx.sparse.SparsePattern.from_coo([0, 1], [0, 1], (2, 2))
    plan = phx.sparse.compile_sparse_jacobian(
        lambda value, scale: scale * value**2,
        point,
        source=space,
        target=space,
        sample_args=np.float64(2.0),
        structure=pattern,
        compiler="native",
    )

    eager = plan.operator(point, 3.0).as_dense()
    jitted = jax.jit(lambda scale: plan.operator(point, scale).as_dense())(
        np.float64(4.0)
    )
    argument_gradient = jax.grad(lambda scale: jnp.sum(plan.coefficients(point, scale)))(
        jnp.asarray(4.0)
    )
    assert jnp.allclose(eager, jnp.diag(6.0 * point))
    assert jnp.allclose(jitted, jnp.diag(8.0 * point))
    assert jnp.isfinite(argument_gradient)
    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4, 1.5])
    arguments = jnp.asarray([1.3, -0.8])
    plan = phx.sparse.compile_sparse_jacobian(
        _band_vector_function,
        point,
        source=space,
        target=space,
        sample_args=arguments,
        compiler="auto",
    )
    expected = jax.jacfwd(_band_vector_function)(point, arguments)

    assert plan.pattern.origin == "structural"
    assert plan.coloring.compiler == "native"
    assert jnp.allclose(
        jax.jit(lambda value: plan.operator(value, arguments).as_dense())(point),
        expected,
    )


def test_numeric_derivative_refresh_preserves_operator_carry_under_jit() -> None:
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    point = jnp.asarray([1.5, -0.5], dtype=jnp.float64)
    plan = phx.sparse.compile_sparse_jacobian(
        lambda value, scale: scale * value**2,
        point,
        source=space,
        target=space,
        sample_args=jnp.asarray(2.0),
        structure=phx.sparse.SparsePattern.from_coo([0, 1], [0, 1], (2, 2)),
        compiler="native",
    )
    initial = plan.operator(point, jnp.asarray(2.0))

    @eqx.filter_jit
    def accumulated(
        runtime_plan: phx.sparse.SparseDerivativePlan,
        operator: phx.sparse.SparseCoordinateOperator,
        scales: jax.Array,
    ) -> jax.Array:
        def step(
            previous: phx.sparse.SparseCoordinateOperator, scale: jax.Array
        ) -> tuple[phx.sparse.SparseCoordinateOperator, jax.Array]:
            del previous
            refreshed = runtime_plan.operator(point, scale)
            return refreshed, refreshed.mv(jnp.ones_like(point))

        _, actions = jax.lax.scan(step, operator, scales)
        return jnp.sum(actions)

    scales = jnp.asarray([0.8, 1.1, 1.7], dtype=jnp.float64)
    value, gradient = jax.value_and_grad(
        lambda values: accumulated(plan, initial, values)
    )(scales)
    np.testing.assert_allclose(value, 2 * np.sum(point) * np.sum(scales), atol=1e-12)
    np.testing.assert_allclose(gradient, np.full(3, 2 * np.sum(point)), atol=1e-12)


def test_rectangular_pytree_jacobian_preserves_coordinate_semantics() -> None:
    source = phx.linalg.PyTreeSpace(
        {
            "field": jnp.zeros((2,), dtype=jnp.float64),
            "parameter": jnp.zeros((), dtype=jnp.float64),
        }
    )
    target = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    point = {
        "field": jnp.asarray([2.0, -1.0]),
        "parameter": jnp.asarray(0.5),
    }

    def function(value: Any, scale: Any) -> Any:
        return jnp.asarray(
            [
                scale * value["field"][0],
                value["field"][1] + value["parameter"] ** 2,
            ]
        )

    pattern = phx.sparse.SparsePattern.from_coo(
        [0, 1, 1],
        [0, 1, 2],
        (2, 3),
    )
    plan = phx.sparse.compile_sparse_jacobian(
        function,
        point,
        source=source,
        target=target,
        sample_args=jnp.asarray(3.0),
        structure=pattern,
        compiler="native",
    )
    dense_reference = jax.jacfwd(
        lambda coordinates: function(source.unflatten(coordinates), jnp.asarray(4.0))
    )(source.flatten(point))

    assert plan.pattern.shape == (target.size, source.size)
    assert jnp.allclose(
        plan.operator(point, jnp.asarray(4.0)).as_dense(), dense_reference
    )


def test_empty_and_dense_patterns_remain_valid() -> None:
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    point = jnp.asarray([1.0, 2.0, 3.0])
    empty = phx.sparse.SparsePattern.from_coo([], [], (3, 3))
    traces = []

    def zero_function(value: Any, _: Any) -> Any:
        traces.append(None)
        return jnp.zeros_like(value)

    empty_plan = phx.sparse.compile_sparse_jacobian(
        zero_function,
        point,
        source=space,
        target=space,
        structure=empty,
        compiler="native",
    )
    compile_traces = len(traces)
    assert empty_plan.num_colors == 0
    assert empty_plan.coefficients(point).shape == (0,)
    assert jnp.array_equal(empty_plan.operator(point).as_dense(), jnp.zeros((3, 3)))
    assert len(traces) == compile_traces
    restored_empty_coloring = phx.sparse.SparseColoring.from_dict(
        json.loads(json.dumps(empty_plan.coloring.to_dict()))
    )
    assert restored_empty_coloring.coloring_id == empty_plan.coloring.coloring_id

    rows, cols = np.indices((3, 3))
    dense = phx.sparse.SparsePattern.from_coo(
        rows.reshape(-1),
        cols.reshape(-1),
        (3, 3),
    )
    dense_plan = phx.sparse.compile_sparse_jacobian(
        lambda value, _: jnp.tanh(value) + jnp.sum(value),
        point,
        source=space,
        target=space,
        structure=dense,
        compiler="native",
    )
    assert dense_plan.num_colors == 3
    assert jnp.allclose(
        dense_plan.operator(point).as_dense(),
        jax.jacfwd(lambda value: jnp.tanh(value) + jnp.sum(value))(point),
    )


def test_sparse_derivatives_scenario_2() -> None:
    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4, 1.5])
    arguments = jnp.asarray([1.3, -0.8])
    pattern = _band_pattern(4, symmetric=True)
    expected = jax.hessian(_band_scalar_function)(point, arguments)

    native = phx.sparse.compile_sparse_hessian(
        _band_scalar_function,
        point,
        space=space,
        sample_args=arguments,
        structure=pattern,
        compiler="native",
    )
    compiled = phx.sparse.compile_sparse_hessian(
        _band_scalar_function,
        point,
        space=space,
        sample_args=arguments,
        structure=pattern,
        compiler="auto",
    )

    assert native.pattern.pattern_id == compiled.pattern.pattern_id
    assert jnp.allclose(native.operator(point).as_dense(), expected)
    assert jnp.allclose(compiled.operator(point).as_dense(), expected)
    assert compiled.num_colors <= native.num_colors
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import phydrax",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    public = set(phx.sparse.__all__)
    assert "compile_sparse_jacobian" in public
    assert "compile_sparse_hessian" in public
    assert "SparseDerivativePlan" in public
    assert not any(name.startswith("Asdex") for name in public)
    assert "compile_asdex_jacobian" not in public
    assert "compile_asdex_hessian" not in public


def test_structural_scalar_dot_respects_contracted_axis_permutations() -> None:
    source = phx.linalg.ArraySpace((2, 3), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((), dtype=jnp.float64)
    point = jnp.arange(6.0, dtype=jnp.float64).reshape((2, 3))
    constant = jnp.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=jnp.float64,
    )

    def contracted(value: Any, _: Any) -> Any:
        return jax.lax.dot_general(
            value,
            constant,
            dimension_numbers=(((0, 1), (1, 0)), ((), ())),
        )

    plan = phx.sparse.compile_sparse_jacobian(
        contracted,
        point,
        source=source,
        target=target,
        compiler="auto",
    )

    expected = jax.jacfwd(contracted)(point, None).reshape((1, -1))
    assert jnp.array_equal(plan.operator(point).as_dense(), expected)


def test_structural_dynamic_slice_clamps_resolved_starts() -> None:
    source = phx.linalg.ArraySpace((5,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    point = jnp.arange(5.0, dtype=jnp.float64)

    def sliced(value: Any, _: Any) -> Any:
        return jax.lax.dynamic_slice(value, (4,), (3,))

    plan = phx.sparse.compile_sparse_jacobian(
        sliced,
        point,
        source=source,
        target=target,
        compiler="auto",
    )

    expected = jax.jacfwd(sliced)(point, None)
    assert jnp.array_equal(plan.operator(point).as_dense(), expected)


def test_matrix_free_verification_detects_missing_structure() -> None:
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4])

    def function(value: Any, _: Any) -> Any:
        return jnp.asarray([value[0] + value[1], value[1] * value[2], value[2] ** 2])

    complete = phx.sparse.SparsePattern.from_coo([0, 0, 1, 1, 2], [0, 1, 1, 2, 2], (3, 3))
    missing = phx.sparse.SparsePattern.from_coo([0, 1, 1, 2], [0, 1, 2, 2], (3, 3))
    complete_plan = phx.sparse.compile_sparse_jacobian(
        function,
        point,
        source=space,
        target=space,
        structure=complete,
        compiler="native",
    )
    missing_plan = phx.sparse.compile_sparse_jacobian(
        function,
        point,
        source=space,
        target=space,
        structure=missing,
        compiler="native",
    )

    accepted = phx.sparse.verify_sparse_derivative(
        complete_plan,
        point,
        key=jax.random.key(0),
        num_probes=4,
    )
    rejected = phx.sparse.verify_sparse_derivative(
        missing_plan,
        point,
        key=jax.random.key(0),
        num_probes=4,
    )
    assert bool(accepted.passed)
    assert accepted.scope == "sample-point"
    assert not bool(rejected.passed)
    assert float(rejected.maximum_absolute_error) > 0.0


def _traced_pattern(plan: phx.sparse.SparseDerivativePlan, shape: tuple[int, int]) -> Any:
    relation = plan.coloring.pattern.relation
    pattern = np.zeros(shape, dtype=np.bool_)
    pattern[np.asarray(relation.target_indices), np.asarray(relation.source_indices)] = (
        True
    )
    return pattern


@jax.jit
def _take(values: Any, index: Any) -> Any:
    return values[index]


def test_structural_trace_isolates_repeated_calls_of_one_jitted_helper() -> None:
    # Both calls bind one shared inner jaxpr. The first call's static index must
    # not be reused as the second call's value-dependent index.
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64)

    def repeated(value: Any, _: Any) -> Any:
        permuted = _take(value, jnp.asarray([1, 2, 0], dtype=jnp.int32))
        selected = jnp.clip(jnp.floor(value), 0, 2).astype(jnp.int32)
        return permuted + _take(value, selected)

    plan = phx.sparse.compile_sparse_jacobian(
        repeated, jnp.zeros((3,)), source=space, target=space, compiler="auto"
    )
    pattern = _traced_pattern(plan, (3, 3))
    for point in ([0.2, 0.3, 0.4], [2.5, 1.5, 0.5], [1.2, 2.7, 1.9]):
        jacobian = np.asarray(jax.jacfwd(repeated)(jnp.asarray(point), None))
        assert not np.any((jacobian != 0) & ~pattern), point


def test_structural_trace_resolves_statically_passing_bounds_checks() -> None:
    # ``eqx.error_if`` routes the index through a cond on a statically false
    # predicate; the checked permutation stays exact instead of dense.
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64)

    def checked(value: Any, _: Any) -> Any:
        index = jnp.asarray([2, 0, 1], dtype=jnp.int32)
        index = eqx.error_if(index, (index < 0) | (index >= 3), "index out of range")
        return 2.0 * value[index]

    plan = phx.sparse.compile_sparse_jacobian(
        checked, jnp.zeros((3,)), source=space, target=space, compiler="auto"
    )
    expected = np.asarray(jax.jacfwd(checked)(jnp.ones((3,)), None)) != 0
    assert np.array_equal(_traced_pattern(plan, (3, 3)), expected)
    assert plan.coloring.num_colors == 1


@pytest.mark.parametrize("reverse", (False, True))
def test_structural_scan_tracks_constants_tuple_carry_and_stacked_outputs(
    reverse: bool,
) -> None:
    source = phx.linalg.ArraySpace((6,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((7,), dtype=jnp.float64)

    def scanned(value: jax.Array, _: None) -> jax.Array:
        gain = 0.2 + value[0]

        def step(
            carry: tuple[jax.Array, jax.Array], sample: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            current, count = carry
            updated = gain * current + sample
            return (updated, count + 1), updated[::-1]

        final, history = jax.lax.scan(
            step,
            (value[:2], jnp.asarray(0, dtype=jnp.int32)),
            value[2:].reshape((2, 2)),
            reverse=reverse,
        )
        return jnp.concatenate(
            (final[0], history.reshape((-1,)), final[1].astype(jnp.float64).reshape((1,)))
        )

    point = jnp.arange(1, 7, dtype=jnp.float64)
    plan = phx.sparse.compile_sparse_jacobian(
        scanned, point, source=source, target=target, compiler="auto"
    )
    expected = np.asarray(jax.jacfwd(scanned)(point, None)) != 0
    assert np.array_equal(_traced_pattern(plan, (7, 6)), expected)


def test_structural_scan_without_samples_threads_captured_parameters() -> None:
    source = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((4,), dtype=jnp.float64)

    def scanned(value: jax.Array, _: None) -> jax.Array:
        def step(carry: jax.Array, sample: None) -> tuple[jax.Array, jax.Array]:
            del sample
            updated = carry * value[0]
            return updated, updated * value[2]

        final, history = jax.lax.scan(step, value[1], None, length=3)
        return jnp.concatenate((final.reshape((1,)), history))

    point = jnp.asarray([1.3, 0.7, 2.0], dtype=jnp.float64)
    plan = phx.sparse.compile_sparse_jacobian(
        scanned, point, source=source, target=target, compiler="auto"
    )
    expected = np.asarray(jax.jacfwd(scanned)(point, None)) != 0
    assert np.array_equal(_traced_pattern(plan, (4, 3)), expected)


def test_structural_while_binds_condition_and_body_constants_in_their_own_lanes() -> None:
    source = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((1,), dtype=jnp.float64)

    def looped(value: jax.Array, _: None) -> jax.Array:
        def condition(carry: tuple[jax.Array, jax.Array]) -> jax.Array:
            return carry[0] < value[1]

        def body(
            carry: tuple[jax.Array, jax.Array],
        ) -> tuple[jax.Array, jax.Array]:
            return carry[0] + 1, carry[1] + value[0]

        _count, result = jax.lax.while_loop(
            condition, body, (jnp.asarray(0, dtype=jnp.int32), value[2])
        )
        return result.reshape((1,))

    point = jnp.asarray([3.0, 2.2, 4.0], dtype=jnp.float64)
    plan = phx.sparse.compile_sparse_jacobian(
        looped, point, source=source, target=target, compiler="auto"
    )
    expected = np.asarray(jax.jacfwd(looped)(point, None)) != 0
    assert np.array_equal(_traced_pattern(plan, (1, 3)), expected)


@pytest.mark.parametrize("kind", ("jit", "remat", "custom_jvp", "custom_vjp"))
def test_nested_structural_programs_preserve_attached_constant_values(kind: str) -> None:
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    weights = jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float64)

    def primal(value: jax.Array) -> jax.Array:
        return weights * value[::-1]

    if kind == "jit":
        transformed = jax.jit(primal)
    elif kind == "remat":
        transformed = jax.checkpoint(primal)
    elif kind == "custom_jvp":
        transformed = jax.custom_jvp(primal)

        def tangent_rule(
            primals: tuple[jax.Array], tangents: tuple[jax.Array]
        ) -> tuple[jax.Array, jax.Array]:
            return primal(primals[0]), primal(tangents[0])

        transformed.defjvp(tangent_rule)
    else:
        transformed = jax.custom_vjp(primal)

        def forward(value: jax.Array) -> tuple[jax.Array, None]:
            return primal(value), None

        def backward(residual: None, cotangent: jax.Array) -> tuple[jax.Array]:
            del residual
            return ((weights * cotangent)[::-1],)

        transformed.defvjp(forward, backward)

    def function(value: jax.Array, _: None) -> jax.Array:
        return transformed(value)

    point = jnp.asarray([0.5, 1.2, 2.3], dtype=jnp.float64)
    plan = phx.sparse.compile_sparse_jacobian(
        function,
        point,
        source=space,
        target=space,
        compiler="auto",
        mode="rev" if kind == "custom_vjp" else "fwd",
    )
    # The custom-VJP primal deliberately has no forward-mode entry point;
    # its independent linear reference has the same scientific Jacobian.
    expected = np.asarray(jax.jacfwd(primal)(point)) != 0
    assert np.array_equal(_traced_pattern(plan, (3, 3)), expected)


def test_structural_condition_unions_dynamic_branch_jacobians() -> None:
    source = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    weights = jnp.asarray([2.0, 3.0], dtype=jnp.float64)

    def branched(value: jax.Array, _: None) -> jax.Array:
        return jax.lax.cond(
            value[0] > 0.0,
            lambda operand: weights * operand[1:3],
            lambda operand: weights[::-1] * operand[2:4],
            value,
        )

    positive = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    negative = positive.at[0].set(-1.0)
    plan = phx.sparse.compile_sparse_jacobian(
        branched, positive, source=source, target=target, compiler="auto"
    )
    expected = (np.asarray(jax.jacfwd(branched)(positive, None)) != 0) | (
        np.asarray(jax.jacfwd(branched)(negative, None)) != 0
    )
    assert np.array_equal(_traced_pattern(plan, (2, 4)), expected)


@pytest.mark.parametrize("size", (0, 2), ids=("empty-constraints", "allocated-values"))
def test_structural_trace_handles_input_independent_allocation(size: int) -> None:
    source = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((3 + size,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4])

    def allocated(value: Any, _: Any) -> Any:
        return jnp.concatenate((value**2, jnp.empty((size,), dtype=value.dtype)))

    plan = phx.sparse.compile_sparse_jacobian(
        allocated, point, source=source, target=target, compiler="auto"
    )
    expected = jax.jacfwd(allocated)(point, None)
    assert np.array_equal(_traced_pattern(plan, (3 + size, 3)), expected != 0)
    assert jnp.array_equal(plan.operator(point).as_dense(), expected)


_STOCK_POINTS = (
    jnp.asarray([0.3, 0.7, 1.1, 1.9], dtype=jnp.float64),
    jnp.asarray([-0.4, 2.5, 0.0, 1.2], dtype=jnp.float64),
    jnp.asarray([1.0, 0.0, -2.0, 0.5], dtype=jnp.float64),
)
_STOCK_GRID = jnp.linspace(0.0, 2.0, 5, dtype=jnp.float64)
_DIAGONAL = np.eye(4, dtype=np.bool_)
_ANTIDIAGONAL = _DIAGONAL[::-1]
_MIRRORED = _DIAGONAL | _ANTIDIAGONAL


def _nested_ldexp_polynomial(value: Any, _: Any) -> Any:
    inner = jax.vmap(lambda a, b: jnp.polyval(jnp.stack([a, b]), jnp.ldexp(a, 2)))
    outer = jax.jit(jax.vmap(inner))
    return outer(value.reshape(2, 2), value[::-1].reshape(2, 2)).ravel()


_STOCK_JACOBIANS: dict[str, tuple[Any, Any]] = {
    "ldexp": (lambda v, _: jnp.ldexp(v, jnp.arange(4)), _DIAGONAL),
    "frexp-mantissa": (lambda v, _: jnp.frexp(v)[0][::-1], _ANTIDIAGONAL),
    "frexp-exponent": (lambda v, _: jnp.frexp(v)[1][::-1] * v, _DIAGONAL),
    "searchsorted-static": (
        lambda v, _: v[jnp.searchsorted(_STOCK_GRID, jnp.asarray([0.7, 1.2, 0.2, 1.4]))],
        np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 0, 1]]),
    ),
    "searchsorted-interp": (
        lambda v, _: jnp.interp(
            jnp.asarray([0.25, 1.6, 2.2, 2.9]), jnp.linspace(0.0, 3.0, 4), v
        ),
        np.asarray([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1]]),
    ),
    "searchsorted-batched-bounds": (
        lambda v, _: jax.vmap(
            lambda row, query: jnp.append(row, 0.0)[jnp.searchsorted(row, query)]
        )(v.reshape(2, 2), jnp.asarray([0.5, 1.5])),
        np.asarray([[1, 1, 0, 0], [0, 0, 1, 1]]),
    ),
    "nonzero-static": (
        lambda v, _: v[jnp.nonzero(jnp.asarray([False, True, False, True]), size=2)[0]],
        np.asarray([[0, 1, 0, 0], [0, 0, 0, 1]]),
    ),
    "nonzero-bounds": (
        lambda v, _: v[jnp.nonzero(v[:2] > 0.5, size=1)[0]],
        np.asarray([[1, 1, 0, 0]]),
    ),
    "complex-sign": (lambda v, _: jnp.real(jnp.sign(v + 1j * v[::-1])), _MIRRORED),
    "real-sign": (lambda v, _: jnp.sign(v) * v[::-1], _ANTIDIAGONAL),
    "polynomial": (
        lambda v, _: jnp.polyval(v[:2], v),
        np.asarray([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 1]]),
    ),
    "polynomial-broadcast": (
        lambda v, _: jax.lax.polynomial(
            v[:2].reshape(2, 1), [v[2:3].reshape(1, 1), v[3:4].reshape(1, 1)]
        ).ravel(),
        np.asarray([[1, 0, 1, 1], [0, 1, 1, 1]]),
    ),
    "nested-batch-jit": (_nested_ldexp_polynomial, _MIRRORED),
}


@pytest.mark.parametrize("case", tuple(_STOCK_JACOBIANS))
def test_structural_trace_stock_operations_match_analytic_jacobians(case: str) -> None:
    function, expected = _STOCK_JACOBIANS[case]
    expected = np.asarray(expected, dtype=np.bool_)
    source = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    target = phx.linalg.ArraySpace((expected.shape[0],), dtype=jnp.float64)

    plan = phx.sparse.compile_sparse_jacobian(
        function, _STOCK_POINTS[0], source=source, target=target, compiler="auto"
    )
    assert np.array_equal(_traced_pattern(plan, expected.shape), expected)
    for point in _STOCK_POINTS:
        jacobian = np.asarray(jax.jacfwd(function)(point, None))
        assert not np.any((jacobian != 0) & ~expected), point
        assert np.allclose(np.asarray(plan.operator(point).as_dense()), jacobian)


_STOCK_HESSIANS: dict[str, tuple[Any, Any]] = {
    "ldexp": (
        lambda v, _: jnp.sum(jnp.ldexp(v, jnp.arange(4)) ** 2 * v[::-1]),
        _MIRRORED,
    ),
    "frexp": (lambda v, _: jnp.sum(jnp.frexp(v)[0] * v**2), _DIAGONAL),
    "polynomial": (
        lambda v, _: jnp.sum(jnp.polyval(v[:2], v)),
        np.asarray([[1, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]),
    ),
    "complex-sign": (
        lambda v, _: jnp.sum(jnp.real(jnp.sign(v + 1j * v[::-1]))),
        _MIRRORED,
    ),
}


@pytest.mark.parametrize("case", tuple(_STOCK_HESSIANS))
def test_structural_trace_stock_operations_match_analytic_hessians(case: str) -> None:
    function, expected = _STOCK_HESSIANS[case]
    expected = np.asarray(expected, dtype=np.bool_)
    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)

    plan = phx.sparse.compile_sparse_hessian(
        function, _STOCK_POINTS[0], space=space, compiler="auto"
    )
    assert np.array_equal(_traced_pattern(plan, (4, 4)), expected)
    for point in _STOCK_POINTS:
        hessian = np.asarray(jax.hessian(function)(point, None))
        assert not np.any((hessian != 0) & ~expected), point
        assert np.allclose(np.asarray(plan.operator(point).as_dense()), hessian)


def test_structural_trace_rejects_unknown_high_level_operations() -> None:
    from jax._src.core import typeof
    from jax._src.numpy.ufuncs import Ldexp

    class ForeignLdexp(Ldexp):
        pass

    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    exponents = jnp.arange(4)

    def function(value: Any, _: Any) -> Any:
        return ForeignLdexp(typeof(value), typeof(exponents))(value, exponents)

    with pytest.raises(TypeError, match="high-level JAX operation .*ForeignLdexp"):
        phx.sparse.compile_sparse_jacobian(
            function, _STOCK_POINTS[0], source=space, target=space, compiler="auto"
        )
