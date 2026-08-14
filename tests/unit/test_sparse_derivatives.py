#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


jax.config.update("jax_enable_x64", True)


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


def _band_vector_function(value, arguments):
    return jnp.asarray(
        [
            value[0] ** 2 + arguments[0] * value[1],
            value[0] * value[1] + value[1] ** 2 + value[2],
            value[1] + jnp.sin(value[2]) + value[2] * value[3],
            value[2] ** 2 + arguments[1] * value[3],
        ]
    )


def _band_scalar_function(value, arguments):
    return (
        value[0] ** 2
        + arguments[0] * value[0] * value[1]
        + value[1] ** 2
        + value[1] * value[2]
        + jnp.sin(value[2])
        + arguments[1] * value[2] * value[3]
        + value[3] ** 2
    )


def test_sparse_pattern_canonicalization_identity_and_roundtrip():
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


def test_sparse_pattern_rejects_invalid_coordinates_and_symmetry():
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


def test_native_coloring_is_deterministic_valid_and_portable():
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


def test_native_jacobian_modes_chunking_jit_vmap_and_gradients():
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


def test_python_and_numpy_scalar_arguments_remain_dynamic():
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


def test_rectangular_pytree_jacobian_preserves_coordinate_semantics():
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

    def function(value, scale):
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


def test_empty_and_dense_patterns_remain_valid():
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    point = jnp.asarray([1.0, 2.0, 3.0])
    empty = phx.sparse.SparsePattern.from_coo([], [], (3, 3))
    traces = []

    def zero_function(value, _):
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


def test_native_hessian_modes_match_dense_and_remain_differentiable():
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
        assert plan.properties.self_adjoint
        assert plan.properties.evidence_for("self_adjoint") == "construction"


def test_asdex_compilation_normalizes_then_evaluates_natively(monkeypatch):
    space = phx.linalg.ArraySpace((4,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4, 1.5])
    arguments = jnp.asarray([1.3, -0.8])
    plan = phx.sparse.compile_sparse_jacobian(
        _band_vector_function,
        point,
        source=space,
        target=space,
        sample_args=arguments,
        compiler="asdex",
    )
    expected = plan.operator(point, arguments).as_dense()

    assert plan.pattern.origin == "asdex"
    assert plan.coloring.compiler == "asdex"
    assert all(
        not type(leaf).__module__.startswith("asdex") for leaf in jax.tree.leaves(plan)
    )
    for name in tuple(sys.modules):
        if name == "asdex" or name.startswith("asdex."):
            monkeypatch.delitem(sys.modules, name)
    assert jnp.allclose(
        jax.jit(lambda value: plan.operator(value, arguments).as_dense())(point),
        expected,
    )
    assert "asdex" not in sys.modules


def test_asdex_and_native_known_pattern_plans_agree():
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
        compiler="asdex",
    )

    assert native.pattern.pattern_id == compiled.pattern.pattern_id
    assert jnp.allclose(native.operator(point).as_dense(), expected)
    assert jnp.allclose(compiled.operator(point).as_dense(), expected)
    assert compiled.num_colors <= native.num_colors


def test_sparse_derivative_contract_rejections_are_explicit():
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
    with pytest.raises(ValueError, match="return a scalar"):
        phx.sparse.compile_sparse_hessian(
            lambda value, _: value,
            point,
            space=space,
            structure=phx.sparse.SparsePattern.from_coo(
                [0, 1], [0, 1], (2, 2), symmetric=True
            ),
            compiler="native",
        )
    non_euclidean = phx.linalg.ArraySpace(
        (2,),
        dtype=jnp.float64,
        pairing=phx.linalg.DiagonalPairing(jnp.asarray([2.0, 3.0])),
    )
    with pytest.raises(ValueError, match="Euclidean pairing"):
        phx.sparse.compile_sparse_hessian(
            lambda value, _: jnp.sum(value**2),
            point,
            space=non_euclidean,
            structure=phx.sparse.SparsePattern.from_coo(
                [0, 1], [0, 1], (2, 2), symmetric=True
            ),
            compiler="native",
        )
    complex_space = phx.linalg.ArraySpace((2,), dtype=jnp.complex128)
    with pytest.raises(TypeError, match="real floating-point coordinates"):
        phx.sparse.compile_sparse_jacobian(
            lambda value, _: value,
            point.astype(jnp.complex128),
            source=complex_space,
            target=complex_space,
            structure=pattern,
            compiler="native",
        )
    mixed_target = phx.linalg.ArraySpace((2,), dtype=jnp.float32)
    with pytest.raises(TypeError, match="same dtype"):
        phx.sparse.compile_sparse_jacobian(
            lambda value, _: value.astype(jnp.float32),
            point,
            source=space,
            target=mixed_target,
            structure=pattern,
            compiler="native",
        )


def test_matrix_free_verification_detects_missing_structure():
    space = phx.linalg.ArraySpace((3,), dtype=jnp.float64)
    point = jnp.asarray([0.7, -1.2, 0.4])

    def function(value, _):
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


def test_sparse_derivatives_participate_in_shared_linear_solves():
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


def test_import_boundary_and_provider_neutral_public_api():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import phydrax; assert 'asdex' not in sys.modules",
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
