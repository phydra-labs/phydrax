#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import optax

import phydrax as phx
from phydrax import linalg as la
from phydrax._numerics import (
    dequantize_mx,
    execute_precision_rewrite,
    MicroscalingFormat,
    PrecisionRewritePolicy,
    PrecisionRewriteRule,
    prepare_precision_rewrite,
    quantize_mx,
)
from phydrax.optim._state_compression import (
    compress_optimizer_state,
    decompress_optimizer_state,
    OptimizerStateCompressionPolicy,
    prepare_compressed_optimizer,
    prepare_optimizer_state_compression,
)


def _positive_properties():
    return la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
        },
    )


def test_batched_factorization_matrix_functions_estimators_and_inertia():
    matrices = jnp.asarray(
        [
            [[4.0, 1.0], [1.0, 3.0]],
            [[2.0, 0.25], [0.25, 1.5]],
        ]
    )
    operator = la.DenseLinearOperator(matrices, properties=_positive_properties())
    factor = la.factorize(
        operator,
        la.FactorizationPolicy(
            "cholesky",
            differentiation=la.DifferentiationPolicy("mathematical"),
        ),
    )
    shared_rhs = jnp.asarray([1.0, -2.0])
    solved = factor.solve(shared_rhs).value
    assert factor.batch_shape == (2,)
    assert factor.rank().shape == (2,)
    assert jnp.allclose(
        solved, jax.vmap(jnp.linalg.solve)(matrices, jnp.broadcast_to(shared_rhs, (2, 2)))
    )
    assert factor.log_abs_determinant().shape == (2,)

    action = la.matrix_exponential_action(operator, shared_rhs)
    reference = jax.vmap(lambda matrix: jsp.linalg.expm(matrix) @ shared_rhs)(matrices)
    assert action.value.shape == (2, 2)
    assert jnp.allclose(action.value, reference, rtol=1e-5, atol=1e-5)
    trace = la.stochastic_trace(operator, key=jr.key(2), num_probes=8)
    replay = la.stochastic_trace(operator, key=jr.key(2), num_probes=8)
    assert trace.samples.shape == (8, 2)
    assert jnp.array_equal(trace.samples, replay.samples)

    inertia = la.factorization_inertia(
        factor,
        la.InertiaPolicy(
            absolute_zero_tolerance=1e-6,
            source="bounded-dense",
            maximum_dense_dimension=2,
            materialization=la.MaterializationPolicy(
                max_entries=matrices.size,
                max_bytes=matrices.nbytes,
            ),
        ),
    )
    assert jnp.array_equal(inertia.positive, jnp.asarray([2, 2]))
    assert jnp.all(inertia.certified)
    assert jnp.all(inertia.zero_count_reliable)


def test_shared_pattern_sparse_factorization_batch_is_independent():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1, 0, 1], dtype=jnp.int32),
        jnp.asarray([0, 0, 1, 1], dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    coefficients = jnp.asarray([[4.0, 1.0, 1.0, 3.0], [2.0, 0.5, 0.5, 1.5]])
    operator = phx.sparse.SparseLinearMap(
        relation,
        coefficients,
        properties=_positive_properties(),
    )
    prepared = la.factorize_sparse(
        operator,
        la.SparseFactorizationPolicy("cholesky"),
    )
    result = prepared.solve(jnp.asarray([1.0, 2.0]))
    dense = operator.as_dense()
    expected = jax.vmap(jnp.linalg.solve)(
        dense,
        jnp.broadcast_to(jnp.asarray([1.0, 2.0]), (2, 2)),
    )
    assert prepared.batch_shape == (2,)
    assert result.status.shape == (2,)
    assert jnp.all(result.success)
    assert jnp.allclose(result.value, expected)


def test_sparse_derivatives_have_explicit_dtype_complex_and_hessian_semantics():
    source = la.ArraySpace((2,), dtype=jnp.float32)
    target = la.ArraySpace((2,), dtype=jnp.float64)
    diagonal = phx.sparse.EdgeRelation(
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        source_size=2,
        target_size=2,
    )
    plan = phx.sparse.compile_sparse_jacobian(
        lambda value, _: value.astype(jnp.float64) ** 2,
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
        source=source,
        target=target,
        structure=diagonal,
        compiler="native",
        mode="fwd",
        precision=phx.sparse.SparseDerivativePrecisionPolicy(
            coefficient=jnp.float64,
            accumulation=jnp.float64,
        ),
    )
    image = plan.operator(jnp.asarray([1.0, 2.0], dtype=jnp.float32)).mv(
        jnp.ones((2,), dtype=jnp.float32)
    )
    assert image.dtype == jnp.float64
    assert jnp.array_equal(image, jnp.asarray([2.0, 4.0], dtype=jnp.float64))

    hessian = phx.sparse.compile_sparse_hessian(
        lambda value, _: jnp.sum(value**2),
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        space=target,
        structure=diagonal,
        compiler="native",
        contract=phx.sparse.SparseHessianContract("bilinear"),
    )
    assert isinstance(hessian.target, la.DualSpace)
    assert jnp.array_equal(
        hessian.operator(jnp.asarray([1.0, 2.0], dtype=jnp.float64)).mv(
            jnp.ones((2,), dtype=jnp.float64)
        ),
        jnp.asarray([2.0, 2.0], dtype=jnp.float64),
    )


def test_mx_formats_rewrite_and_local_optimizer_compression_are_explicit():
    values = jnp.linspace(-3.0, 3.0, 35, dtype=jnp.float32)
    format_ = MicroscalingFormat("mxfp4-e2m1", axis=0)
    payload = quantize_mx(values, format_)
    restored = dequantize_mx(payload)
    assert restored.shape == values.shape
    assert payload.payload_bytes < values.nbytes
    assert int(payload.saturation_count) == 0

    function = lambda left, right: left @ right
    arguments = (jnp.ones((4, 3)), jnp.ones((3, 2)))
    rewrite = prepare_precision_rewrite(
        function,
        arguments,
        PrecisionRewritePolicy(
            (PrecisionRewriteRule("dot_general", "float32", "float32"),)
        ),
    )
    rewritten = execute_precision_rewrite(rewrite, *arguments)
    assert rewritten.dtype == jnp.float32
    assert jnp.array_equal(rewritten, function(*arguments))
    assert rewrite.original_fingerprint != ""

    state = {
        "count": jnp.asarray(3, dtype=jnp.int32),
        "moment": jnp.asarray([0.25, -0.5], dtype=jnp.float32),
    }
    roles = {"count": "exact", "moment": "first-moment"}
    plan = prepare_optimizer_state_compression(
        state,
        OptimizerStateCompressionPolicy("float16"),
        transformation_id="test-transform",
        leaf_roles=roles,
    )
    compressed = compress_optimizer_state(plan, state)
    decompressed = decompress_optimizer_state(plan, compressed)
    assert decompressed["count"] == state["count"]
    assert decompressed["moment"].dtype == state["moment"].dtype
    assert compressed.diagnostics[0].payload_bytes < state["moment"].nbytes

    prepared_optimizer = prepare_compressed_optimizer(
        optax.adam(1e-2),
        jnp.asarray([1.0, -1.0]),
        OptimizerStateCompressionPolicy("float16"),
        transformation_id="optax.adam:test",
    )
    compressed_state = prepared_optimizer.init(jnp.asarray([1.0, -1.0]))
    updates, compressed_state = prepared_optimizer.update(
        jnp.asarray([0.5, -0.25]),
        compressed_state,
        jnp.asarray([1.0, -1.0]),
    )
    assert updates.shape == (2,)
    assert compressed_state.plan_id == prepared_optimizer.plan.plan_id


def test_coordinate_tree_and_full_complex_training_checkpoint_round_trip(tmp_path):
    complex_space = la.ArraySpace((2,), dtype=jnp.complex64)
    coordinate_map = la.ComplexCartesianCoordinates(complex_space)
    template = {"complex": jnp.ones((2,), dtype=jnp.complex64)}
    prepared_coordinates = la.prepare_real_coordinate_tree(
        template,
        {"complex": coordinate_map},
    )
    coordinates = prepared_coordinates.to_real_coordinates(template)
    assert jax.tree.all(
        jax.tree.map(
            jnp.array_equal,
            prepared_coordinates.from_real_coordinates(coordinates),
            template,
        )
    )

    model = phx.nn.layers.ComplexLinear(in_size=2, out_size=2, key=jr.key(0))
    optimizer_state = {
        "first_real": jnp.asarray([0.1, 0.2]),
        "first_imag": jnp.asarray([-0.3, 0.4]),
        "second_real": jnp.asarray([0.5, 0.6]),
        "second_imag": jnp.asarray([0.7, 0.8]),
        "count": jnp.asarray(4, dtype=jnp.int32),
    }
    paths = {
        key: jax.tree_util.keystr((jax.tree_util.DictKey(key),))
        for key in optimizer_state
    }
    groups = (
        phx.export.ComplexOptimizerStateGroup(
            "first",
            "complex-vector",
            (paths["first_real"], paths["first_imag"]),
        ),
        phx.export.ComplexOptimizerStateGroup(
            "second",
            "cartesian-second-moment",
            (paths["second_real"], paths["second_imag"]),
        ),
        phx.export.ComplexOptimizerStateGroup(
            "count",
            "exact-discrete",
            (paths["count"],),
        ),
    )
    rng_state = {"train": jr.key(1), "eval": jr.key(2)}
    auxiliary = {"loss_scale": jnp.asarray(1024.0)}
    prepared = phx.export.prepare_complex_training_interchange(
        model,
        optimizer_state,
        rng_state,
        auxiliary,
        optimizer_groups=groups,
        training_id="complex-training-test",
    )
    state = phx.export.export_complex_training_state(
        prepared,
        model,
        optimizer_state,
        rng_state,
        auxiliary,
        step=7,
    )
    destination = tmp_path / "complex-training.phx"
    phx.export.write_complex_training_checkpoint(str(destination), state)
    loaded = phx.export.read_complex_training_checkpoint(str(destination), prepared)
    restored = phx.export.import_complex_training_state(prepared, loaded)
    assert restored.step == 7
    assert jax.tree.all(
        jax.tree.map(jnp.array_equal, restored.optimizer_state, optimizer_state)
    )
    assert jax.tree.all(
        jax.tree.map(
            lambda left, right: jnp.array_equal(jr.key_data(left), jr.key_data(right)),
            restored.rng_state,
            rng_state,
        )
    )
