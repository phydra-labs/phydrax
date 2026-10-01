#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import functools
import sys
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from equinox._ad import _ClosureConvert
from jax import Array

from phydrax._execution_pool import PoolExecutionSignature
from phydrax._identity import (
    ArtifactBindingIdentity,
    callable_payload,
    ExecutableSignature,
    execution_metadata_payload,
    NumericRevision,
    SemanticProvenance,
    strict_module_payload,
)
from phydrax._strict import StrictModule


class _AffineCallable(StrictModule):
    weight: Array
    enabled: bool = eqx.field(static=True)

    def __init__(self, weight: Any, /, *, enabled: bool = True) -> None:
        self.weight = jnp.asarray(weight, dtype=jnp.float32)
        self.enabled = bool(enabled)

    def __call__(self, value: Any) -> Any:
        return self.weight * value if self.enabled else value


class _ActivatedCallable(StrictModule):
    activation: object = eqx.field(static=True)

    def __init__(self, activation: Any, /) -> None:
        self.activation = activation

    def __call__(self, value: Any) -> Any:
        # ty: ignore[call-non-callable]
        return self.activation(value)


def _square(value: Any, *, scale: Any = 1.0) -> Any:
    return scale * value * value


def _cube(value: Any, *, scale: Any = 1.0) -> Any:
    return scale * value * value * value


_module_lambda = lambda value: value


def test_identity_scenario_1() -> None:
    payload = callable_payload(_square)

    assert callable_payload(_square) == payload
    assert (
        callable_payload(_cube)["semantic_content_id"] != (payload["semantic_content_id"])
    )
    nested = strict_module_payload(_ActivatedCallable(_square))
    assert nested == strict_module_payload(_ActivatedCallable(_square))
    assert (
        nested["semantic_content_id"]
        != (strict_module_payload(_ActivatedCallable(_cube))["semantic_content_id"])
    )
    for reader in (_weighted, _offset, _calls_weighted):
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            callable_payload(reader)
        with pytest.raises(TypeError, match="requires explicit semantic and numeric"):
            strict_module_payload(_ActivatedCallable(reader))
        payload = callable_payload(reader, semantic_id="law", numeric_id="weights")
        assert payload["numeric_content_id"] == "weights"
    first = lambda value: value + 1.0
    second = lambda value: value + 2.0

    for opaque in (first, second, _module_lambda):
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            callable_payload(opaque)
        with pytest.raises(TypeError, match="requires explicit semantic and numeric"):
            strict_module_payload(_ActivatedCallable(opaque))
    first_payload = callable_payload(first, semantic_id="shift", numeric_id="one")
    second_payload = callable_payload(second, semantic_id="shift", numeric_id="two")
    assert first_payload["numeric_content_id"] != second_payload["numeric_content_id"]


_GLOBAL_WEIGHTS = jnp.asarray([1.0, 2.0])
_GLOBAL_OFFSETS = (jnp.asarray([3], dtype=jnp.int32),)
_GLOBAL_SCALE = 2.0


def _weighted(value: Any) -> Any:
    return _GLOBAL_WEIGHTS * value


def _offset(value: Any) -> Any:
    return value + _GLOBAL_OFFSETS[0]


def _calls_weighted(value: Any) -> Any:
    return _weighted(value) + 1.0


def _scaled(value: Any) -> Any:
    return jnp.sin(value) * _GLOBAL_SCALE


def test_plain_function_identity_follows_scalar_global_values(monkeypatch: Any) -> None:
    payload = callable_payload(_scaled)
    assert callable_payload(_scaled) == payload

    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_SCALE", 3.0)

    assert (
        callable_payload(_scaled)["semantic_content_id"]
        != (payload["semantic_content_id"])
    )


def test_identity_scenario_2() -> None:
    opaque_callables = (
        functools.partial(_square, scale=2.0),
        _AffineCallable([1.0]).__call__,
        jnp.sin,
        _AffineCallable,
    )
    for opaque in opaque_callables:
        with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
            callable_payload(opaque)
    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        callable_payload(_square, semantic_id="square-law")
    first = _AffineCallable([1.0, 2.0])
    second = _AffineCallable([3.0, 4.0])

    first_payload = strict_module_payload(first)
    second_payload = strict_module_payload(second)

    assert first_payload["semantic_content_id"] == second_payload["semantic_content_id"]
    assert first_payload["numeric_content_id"] != second_payload["numeric_content_id"]
    assert callable_payload(first) == first_payload

    provenance = SemanticProvenance(
        first_payload["semantic_payload"],
        resource_ids={"mesh": "mesh-content-47", "law": "linear-law"},
    )
    first_revision = NumericRevision(provenance, first)
    second_revision = NumericRevision(provenance, second)

    assert first_revision.semantic_id == second_revision.semantic_id
    assert first_revision.content_id != second_revision.content_id
    assert first_revision.revision_id != second_revision.revision_id
    content = {"law": "linear-elastic", "state_space": "cartesian"}
    first = SemanticProvenance(content, resource_ids={"mesh": "mesh-a"})
    second = SemanticProvenance(content, resource_ids={"mesh": "mesh-b"})

    assert first.content_id == second.content_id
    assert first.semantic_id != second.semantic_id


def test_identity_scenario_3() -> None:
    with pytest.raises(TypeError, match="integer sequences"):
        ExecutableSignature(shapes={"state": jnp.asarray([2])})
    with pytest.raises(TypeError, match="not arrays"):
        ExecutableSignature(dtypes={"state": jnp.zeros((2,))})
    with pytest.raises(TypeError, match="cannot contain a numeric array"):
        ExecutableSignature(algorithm_facts={"coefficients": jnp.asarray([1.0, 2.0])})
    with pytest.raises(TypeError, match="cannot contain a numeric array"):
        ExecutableSignature(backend_facts={"device_state": jnp.asarray(1)})
    pool = PoolExecutionSignature(
        topology_id="pool-topology",
        method_id="pool-method",
        precision_id="float32",
        backend_id="cpu",
        shard_count=4,
        static_callables={"response": _AffineCallable([2.0])},
    )
    generic = ExecutableSignature(
        topology_ids={"pool": "pool-topology"},
        capacities={"shards": 4, "processes": 1},
        algorithm_facts={"method_id": "pool-method"},
        backend_facts={"backend_id": "cpu", "precision_id": "float32"},
        static_callables={"response": _AffineCallable([2.0])},
    )

    assert pool.signature_id == generic.signature_id
    assert pool.executable_signature.signature_id == generic.signature_id
    semantic = SemanticProvenance({"kind": "affine-response"})
    revision = NumericRevision(semantic, {"weight": jnp.asarray([1.0, 2.0])})
    signature = ExecutableSignature(shapes={"weight": (2,)})
    binding = ArtifactBindingIdentity(semantic, revision, signature)

    assert (binding.semantic_id, binding.numeric_revision_id) == (
        semantic.semantic_id,
        revision.revision_id,
    )
    assert binding.executable_signature_id == signature.signature_id
    assert ArtifactBindingIdentity.from_record(binding.to_record()) == binding
    assert (
        ArtifactBindingIdentity(
            semantic.semantic_id, revision.revision_id, signature.signature_id
        )
        == binding
    )
    retrained = NumericRevision(semantic, {"weight": jnp.asarray([1.0, 5.0])})
    assert (
        ArtifactBindingIdentity(semantic, retrained, signature).binding_id
        != binding.binding_id
    )
    foreign = NumericRevision(SemanticProvenance({"kind": "other"}), {})
    with pytest.raises(ValueError, match="another semantic provenance"):
        ArtifactBindingIdentity(semantic, foreign, signature)
    record = binding.to_record()
    with pytest.raises(ValueError, match="corrupt"):
        ArtifactBindingIdentity.from_record(
            {**record, "numeric_revision_id": retrained.revision_id}
        )
    with pytest.raises(ValueError, match="fields do not match"):
        ArtifactBindingIdentity.from_record(
            {key: value for key, value in record.items() if key != "binding_id"}
        )


def test_opaque_callable_requires_explicit_semantic_and_numeric_ids() -> None:
    offset = 2.0

    def closure(value: Any) -> Any:
        return value + offset

    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        callable_payload(closure)

    payload = callable_payload(
        closure,
        semantic_id="translation-law",
        numeric_id="translation-offset-two",
    )
    assert payload["semantic_content_id"] == "translation-law"
    assert payload["numeric_content_id"] == "translation-offset-two"


def test_static_held_weights_are_part_of_the_executable_signature() -> None:
    def signature(**callables: Any) -> Any:
        return ExecutableSignature(
            shapes={"x": (2,)}, dtypes={"x": "float32"}, static_callables=callables
        ).signature_id

    baseline = signature(response=_AffineCallable([1.0, 2.0]))
    assert signature(response=_AffineCallable([1.0, 2.0])) == baseline
    assert signature(response=_AffineCallable([1.0, 3.0])) != baseline
    assert signature(response=_ActivatedCallable(_square)) != signature(
        response=_ActivatedCallable(_cube)
    )
    assert signature() != baseline
    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        signature(response=lambda value: value)


def _program_identity(
    function: Any, value: Any, /, *, dynamic_captures: bool = False
) -> Any:
    return execution_metadata_payload(
        jax.make_jaxpr(function)(value), dynamic_captures=dynamic_captures
    )


def test_compiler_identity_tracks_program_semantics_not_variable_addresses() -> None:
    value = jnp.ones((2,), dtype=jnp.float64)
    baseline = _program_identity(lambda x: x + 2.0, value)
    assert _program_identity(lambda y: y + 2.0, value) == baseline
    assert _program_identity(lambda x: x * 2.0, value) != baseline
    assert _program_identity(lambda x: x + 3.0, value) != baseline
    assert (
        _program_identity(lambda x: x + 2.0, jnp.ones((3,), dtype=value.dtype))
        != baseline
    )
    assert _program_identity(lambda x: x - 2.0, value) != baseline
    first = lambda x: jax.lax.cond(x[0] > 0, lambda y: y + 2, lambda y: y * 2, x)
    second = lambda x: jax.lax.cond(x[0] > 0, lambda y: y + 2, lambda y: y * 3, x)
    assert _program_identity(first, value) != _program_identity(second, value)


def test_compiler_identity_separates_dynamic_captures_from_static_constants() -> None:
    value = jnp.ones((2,), dtype=jnp.float64)
    one = jnp.asarray([2.0, 3.0])
    two = jnp.asarray([4.0, 5.0])
    first = eqx.filter_closure_convert(lambda x: x + one, value)
    second = eqx.filter_closure_convert(lambda x: x + two, value)
    if not isinstance(first, _ClosureConvert) or not isinstance(second, _ClosureConvert):
        pytest.fail(
            "Captured native arrays must produce the owning Equinox closure-conversion type."
        )
    assert execution_metadata_payload(
        first.jaxpr, dynamic_captures=True
    ) == execution_metadata_payload(second.jaxpr, dynamic_captures=True)
    assert _program_identity(lambda x: x + one, value) != _program_identity(
        lambda x: x + two, value
    )
    assert bool(jnp.all(first(value) == jnp.asarray([3.0, 4.0])))
    assert bool(jnp.all(second(value) == jnp.asarray([5.0, 6.0])))
    first_nested = jax.jit(lambda x: x + one)
    second_nested = jax.jit(lambda x: x + two)
    assert _program_identity(first_nested, value) != _program_identity(
        second_nested, value
    )


def test_compiler_identity_tracks_primitive_parameters_and_custom_derivatives() -> None:
    value = jnp.asarray([1.0, 2.0, 3.0])
    assert _program_identity(
        lambda x: jax.lax.slice(x, (0,), (2,)), value
    ) != _program_identity(lambda x: jax.lax.slice(x, (1,), (3,)), value)

    def make_rule(scale: float) -> Any:
        function = jax.custom_jvp(lambda x: x * x)
        function.defjvp(
            lambda primals, tangents: (primals[0] ** 2, scale * primals[0] * tangents[0])
        )
        return function

    first, second = make_rule(2.0), make_rule(3.0)
    assert _program_identity(first, value) != _program_identity(second, value)
    assert bool(
        jnp.all(jax.jvp(first, (value,), (jnp.ones_like(value),))[1] == 2.0 * value)
    )
    assert bool(
        jnp.all(jax.jvp(second, (value,), (jnp.ones_like(value),))[1] == 3.0 * value)
    )


def test_compiler_identity_tracks_callbacks_and_refuses_unowned_effects() -> None:
    value = jnp.asarray([1.0, 2.0])

    def callback(offset: float) -> Any:
        def function(x: Any) -> Any:
            return jax.pure_callback(
                lambda y: y + offset, jax.ShapeDtypeStruct(x.shape, x.dtype), x
            )

        return function

    assert _program_identity(callback(2.0), value) != _program_identity(
        callback(3.0), value
    )
    assert bool(jnp.all(callback(2.0)(value) == jnp.asarray([3.0, 4.0])))

    def effectful(x: Any) -> Any:
        jax.debug.callback(lambda y: None, x, ordered=True)
        return x

    with pytest.raises(TypeError, match="Effectful compiler program"):
        _program_identity(effectful, value)


def test_compiler_metadata_tracks_tree_shape_weakness_and_refuses_placement() -> None:
    leaf = execution_metadata_payload(jax.tree.structure(0))
    assert execution_metadata_payload(jax.tree.structure(None)) != leaf
    assert execution_metadata_payload(
        jax.tree.structure((0, None))
    ) != execution_metadata_payload(jax.tree.structure([0, None]))
    shape = jax.ShapeDtypeStruct((2,), jnp.float64)
    assert execution_metadata_payload(shape) != execution_metadata_payload(
        jax.ShapeDtypeStruct((2,), jnp.float64, weak_type=True)
    )
    assert execution_metadata_payload(shape) != execution_metadata_payload(
        jax.ShapeDtypeStruct((2,), jnp.float32)
    )
    placed = jax.ShapeDtypeStruct(
        (2,), jnp.float64, sharding=jax.sharding.SingleDeviceSharding(jax.devices()[0])
    )
    with pytest.raises(TypeError, match="Unsupported compiler sharding"):
        execution_metadata_payload(placed)


def test_compiler_callback_identity_tracks_bound_global_array_values(
    monkeypatch: Any,
) -> None:
    value = jnp.asarray([1.0, 2.0])

    def host_callback(x: Any) -> Any:
        return x + _GLOBAL_WEIGHTS

    function = lambda x: jax.pure_callback(
        host_callback, jax.ShapeDtypeStruct(x.shape, x.dtype), x
    )
    baseline = _program_identity(function, value)
    monkeypatch.setattr(sys.modules[__name__], "_GLOBAL_WEIGHTS", jnp.asarray([4.0, 5.0]))
    assert _program_identity(function, value) != baseline
    assert bool(jnp.all(function(value) == jnp.asarray([5.0, 7.0])))


def test_compiler_identity_retains_recursive_native_error_callbacks() -> None:
    value = jnp.asarray([1.0, 2.0])
    function = lambda x: eqx.error_if(x, x[0] < 0, "negative input")
    program = jax.make_jaxpr(function)(value)
    baseline = execution_metadata_payload(program)
    assert execution_metadata_payload(program) == baseline
    assert (
        _program_identity(
            lambda x: eqx.error_if(x, x[0] < 0, "wrong scientific role"), value
        )
        != baseline
    )
    assert bool(jnp.all(function(value) == value))
    batch = jnp.stack((value, value + 2.0))
    checked = jax.vmap(function)
    assert _program_identity(checked, batch) != _program_identity(
        jax.vmap(lambda x: eqx.error_if(x, x[0] < 0, "wrong scientific role")), batch
    )
    assert bool(jnp.all(checked(batch) == batch))
    assert bool(jnp.all(jax.jvp(checked, (batch,), (jnp.ones_like(batch),))[1] == 1.0))


def test_compiler_identity_refuses_name_colliding_foreign_primitives() -> None:
    from jax.extend import core

    primitive = core.Primitive("add")
    primitive.def_impl(lambda x: x * 2)
    primitive.def_abstract_eval(lambda x: x)
    with pytest.raises(TypeError, match="Foreign compiler primitive"):
        _program_identity(lambda x: primitive.bind(x), jnp.asarray([1.0, 2.0]))


def test_compiler_identity_preserves_bound_custom_derivative_selectors() -> None:
    @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
    def function(scale: float, x: Any) -> Any:
        return x * x

    @function.defjvp
    def rule(scale: float, primals: Any, tangents: Any) -> Any:
        return primals[0] ** 2, scale * primals[0] * tangents[0]

    value = jnp.asarray([1.0, 2.0])
    first = lambda x: function(2.0, x) + function(3.0, x)
    second = lambda x: function(2.0, x) + function(4.0, x)
    assert _program_identity(first, value) != _program_identity(second, value)
    assert bool(
        jnp.all(jax.jvp(first, (value,), (jnp.ones_like(value),))[1] == 5.0 * value)
    )
    assert bool(
        jnp.all(jax.jvp(second, (value,), (jnp.ones_like(value),))[1] == 6.0 * value)
    )


def test_compiler_identity_tracks_native_stack_axis_semantics() -> None:
    value = jnp.asarray([1.0, 2.0])
    first = lambda x: jnp.stack((x, x + 3.0), axis=0)
    second = lambda x: jnp.stack((x, x + 3.0), axis=1)
    assert _program_identity(first, value) != _program_identity(second, value)
    assert bool(jnp.all(first(value) == jnp.asarray([[1.0, 2.0], [4.0, 5.0]])))
    assert bool(jnp.all(second(value) == jnp.asarray([[1.0, 4.0], [2.0, 5.0]])))


def test_compiler_identity_retains_batched_custom_derivative_axis_semantics() -> None:
    function = jax.custom_jvp(lambda x: jnp.sum(x) * x)
    function.defjvp(
        lambda ps, ts: (
            jnp.sum(ps[0]) * ps[0],
            jnp.sum(ps[0]) * ts[0] + jnp.sum(ts[0]) * ps[0],
        )
    )
    value = jnp.asarray([[0.0, 1.0], [2.0, 3.0]])
    rows = jax.vmap(function, in_axes=0)
    columns = jax.vmap(function, in_axes=1, out_axes=1)
    baseline = _program_identity(rows, value)
    assert _program_identity(rows, value) == baseline
    assert _program_identity(columns, value) != baseline
    assert bool(jnp.all(rows(value) == jnp.asarray([[0.0, 1.0], [10.0, 15.0]])))
    assert bool(jnp.all(columns(value) == jnp.asarray([[0.0, 4.0], [4.0, 12.0]])))


def test_compiler_tree_identity_retains_native_module_static_semantics() -> None:
    first = _AffineCallable([1.0, 2.0], enabled=True)
    same_static = _AffineCallable([3.0, 4.0], enabled=True)
    disabled = _AffineCallable([1.0, 2.0], enabled=False)
    baseline = execution_metadata_payload(jax.tree.structure(first))
    assert execution_metadata_payload(jax.tree.structure(same_static)) == baseline
    assert execution_metadata_payload(jax.tree.structure(disabled)) != baseline
    assert execution_metadata_payload(
        jax.tree.structure(_ActivatedCallable(_square))
    ) != execution_metadata_payload(jax.tree.structure(_ActivatedCallable(_cube)))


def test_compiler_identity_preserves_static_closure_owner_capture_rebinding() -> None:
    weights = jnp.asarray([2.0, 3.0])
    value = jnp.asarray([1.0, 2.0])
    bound = eqx.filter_closure_convert(lambda x: weights * x, value)
    if not isinstance(bound, _ClosureConvert):
        pytest.fail("Captured arrays require the native closure-conversion owner.")
    changed = eqx.tree_at(lambda owner: owner.consts[0], bound, jnp.asarray([4.0, 5.0]))
    other = eqx.tree_at(lambda owner: owner.consts[0], bound, jnp.asarray([6.0, 7.0]))

    @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
    def function(callback: Any, x: Any) -> Any:
        return x * x

    @function.defjvp
    def rule(callback: Any, primals: Any, tangents: Any) -> Any:
        return primals[0] ** 2, callback(tangents[0])

    first = lambda x: function(bound, x) + function(changed, x)
    second = lambda x: function(bound, x) + function(other, x)
    assert _program_identity(first, value) != _program_identity(second, value)
    assert bool(
        jnp.all(
            jax.jvp(first, (value,), (jnp.ones_like(value),))[1]
            == jnp.asarray([6.0, 8.0])
        )
    )
    assert bool(
        jnp.all(
            jax.jvp(second, (value,), (jnp.ones_like(value),))[1]
            == jnp.asarray([8.0, 10.0])
        )
    )


def test_compiler_identity_retains_native_partitioned_closure_bindings() -> None:
    weights = jnp.asarray([2.0, 3.0])
    value = jnp.asarray([1.0, 2.0])
    bound = eqx.filter_closure_convert(lambda x: weights * x, value)
    if not isinstance(bound, _ClosureConvert):
        pytest.fail("Captured arrays require the native closure-conversion owner.")
    changed = eqx.tree_at(lambda owner: owner.consts[0], bound, jnp.asarray([4.0, 5.0]))
    first_pair = eqx.partition(bound, eqx.is_array)
    second_pair = eqx.partition(changed, eqx.is_array)

    @functools.partial(jax.custom_jvp, nondiff_argnums=(0,))
    def function(parts: Any, x: Any) -> Any:
        return x * x

    @function.defjvp
    def rule(parts: Any, primals: Any, tangents: Any) -> Any:
        callback = eqx.combine(*parts)
        return primals[0] ** 2, callback(tangents[0])

    first = lambda x: function(first_pair, x)
    second = lambda x: function(second_pair, x)
    assert _program_identity(first, value) != _program_identity(second, value)
    assert bool(jnp.all(jax.jvp(first, (value,), (jnp.ones_like(value),))[1] == weights))
    assert bool(
        jnp.all(
            jax.jvp(second, (value,), (jnp.ones_like(value),))[1]
            == jnp.asarray([4.0, 5.0])
        )
    )
    trivial = eqx.filter_closure_convert(lambda x: 2.0 * x, value)
    trivial_pair = eqx.partition(trivial, eqx.is_array)
    third = lambda x: function(trivial_pair, x)
    assert _program_identity(third, value) != _program_identity(first, value)
    assert bool(jnp.all(jax.jvp(third, (value,), (jnp.ones_like(value),))[1] == 2.0))


def test_compiler_identity_retains_declared_derivative_refusal_without_executing_it() -> (
    None
):
    from collections.abc import Callable
    from typing import NoReturn

    from jax import Array

    def refusing_derivative(message: str) -> Callable[..., Array]:
        @jax.custom_jvp
        def function(*values: Array) -> Array:
            return jnp.sum(jnp.stack(values))

        def rule(primals: tuple[Array, ...], tangents: tuple[Any, ...]) -> NoReturn:
            raise RuntimeError(message)

        function.defjvp(rule, symbolic_zeros=True)
        return function

    first = refusing_derivative("Discrete policy has no derivative.")
    second = refusing_derivative("Source boundary has no derivative.")
    inputs = tuple(jnp.asarray(1.0, dtype=jnp.float64) for _ in range(21))
    first_program = jax.make_jaxpr(first)(*inputs)
    second_program = jax.make_jaxpr(second)(*inputs)
    assert execution_metadata_payload(first_program) != execution_metadata_payload(
        second_program
    )
    assert float(first(*inputs)) == 21.0
    with pytest.raises(RuntimeError):
        jax.jvp(first, inputs, tuple(jnp.ones_like(value) for value in inputs))
