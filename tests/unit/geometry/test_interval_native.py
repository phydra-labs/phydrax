"""Native straight-line interval programs reproduce the canonical interpreter bitwise."""

import math
from collections.abc import Callable
from dataclasses import replace
from typing import NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
import phydrax._meshcore as meshcore_owner
from phydrax._meshcore import meshcore_provider_installed, MeshcoreUnavailableError
from phydrax.geometry._interval_enclosure import (
    NativeIntervalProgram,
    prepare_interval_function,
    PreparedIntervalFunction,
)
from phydrax.geometry._interval_native import (
    MAXIMUM_REDUCED_LENGTH,
    REDUCE_SUM,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._query import _face_certificates, prepare_brep_query
from phydrax.geometry.brep._query_types import FaceCertificate
from tests._support.native_interval_reference import execute as reference_execute


type Bounds = tuple[np.ndarray, np.ndarray]
type Executor = Callable[[NativeIntervalProgram, np.ndarray, np.ndarray], Bounds]

# Only a genuinely absent provider skips; an installed but unloadable library fails.
_NATIVE = pytest.mark.skipif(
    not meshcore_provider_installed(), reason="phydrax-meshcore provider is not installed"
)


def _meshcore_execute(
    program: NativeIntervalProgram, lower: np.ndarray, upper: np.ndarray
) -> Bounds:
    return meshcore_owner.interval_program_execute(program, lower, upper)


_EXECUTORS = pytest.mark.parametrize(
    "executor",
    (
        pytest.param(reference_execute, id="reference"),
        pytest.param(
            _meshcore_execute, id="meshcore", marks=(pytest.mark.meshcore, _NATIVE)
        ),
    ),
)


def _same(first: np.ndarray, second: np.ndarray) -> bool:
    first, second = np.asarray(first, np.float64), np.asarray(second, np.float64)
    return first.shape == second.shape and bool(
        np.all(first.reshape(-1).view(np.int64) == second.reshape(-1).view(np.int64))
    )


def _interpreted(program: PreparedIntervalFunction) -> PreparedIntervalFunction:
    """The canonical owner interpreter of the same prepared program."""
    return replace(program, native=None)


def _stacked_components(values: Array) -> Array:
    return jnp.stack(
        (values[::-1] + values, jnp.sin(values), jnp.full_like(values, 0.25)), axis=1
    )


def _joined_components(values: Array) -> Array:
    matrix = values.reshape(2, 2)
    return jnp.concatenate((matrix, matrix.T - 0.5), axis=1)


@_EXECUTORS
@pytest.mark.parametrize(
    ("function", "dimension"),
    ((_stacked_components, 3), (_joined_components, 4)),
    ids=("stack-independent-regions", "concatenate-transposed-region"),
)
@pytest.mark.parametrize("batch", (1, 3))
def test_structural_joins_preserve_ordered_interval_values(
    executor: Executor, function: Callable[[Array], Array], dimension: int, batch: int
) -> None:
    program = prepare_interval_function(function, dimension, batch_capacity=batch)
    native = program.native
    assert native is not None
    interpreted = _interpreted(program)
    for lower, upper in _boxes(dimension, 73, batch):
        with np.errstate(invalid="ignore", over="ignore"):
            expected = interpreted.evaluate(lower, upper)
            actual = executor(native, lower, upper)
        assert _same(actual[0], expected[0]) and _same(actual[1], expected[1])


def _boxes(dimension: int, seed: int, rows: int = 1) -> list[Bounds]:
    generator = np.random.default_rng(seed)
    boxes = []
    for scale in (1.0e-12, 1.0e-3, 0.5, 4.0, 40.0):
        for _ in range(12):
            center = generator.normal(size=(rows, dimension)) * (1.0 + 3.0 * scale)
            radius = np.abs(generator.normal(size=(rows, dimension))) * scale
            boxes.append((center - radius, center + radius))
    point = generator.normal(size=(rows, dimension))
    boxes.append((point, point))  # degenerate box: proved-zero and point paths
    boxes.append((np.zeros((rows, dimension)), np.zeros((rows, dimension))))
    boxes.append((-np.zeros((rows, dimension)), np.zeros((rows, dimension))))
    boxes.append(
        (np.full((rows, dimension), -np.inf), np.full((rows, dimension), np.inf))
    )
    boxes.append(
        (np.full((rows, dimension), 1.0e300), np.full((rows, dimension), 1.7e308))
    )
    boxes.append(
        (np.full((rows, dimension), -7.0), np.full((rows, dimension), 7.0))
    )  # > 2 pi
    return boxes


def _assert_identical(
    program: PreparedIntervalFunction, executor: Executor, dimension: int, seed: int
) -> None:
    native = program.native
    assert native is not None
    owner = _interpreted(program)
    for lower, upper in _boxes(dimension, seed, program.batch_capacity):
        expected = owner.evaluate(lower, upper)
        actual = executor(native, lower, upper)
        assert _same(actual[0], expected[0]) and _same(actual[1], expected[1])


def _rotation(source: Array) -> Array:
    # Rotation-like trigonometric products, a three-term innermost sum and a
    # same-operand difference (proved zero on finite bounds).
    rotated = jnp.cos(source[0]) * source - jnp.sin(source[0]) * source[::-1]
    return jnp.sum(rotated * rotated) + (source[0] - source[0]) - source[1] * source[2]


@pytest.fixture(scope="module")
def cylinder_certificates() -> tuple[FaceCertificate, ...]:
    model = phx.geometry.brep.brep_cylinder(
        0.6,
        3.0,
        base_center=(-1.5, 0.0, 0.0),
        axis=(1.0, 0.0, 0.0),
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        tessellation=phx.geometry.brep.BRepTessellationPolicy(realize=False),
    )
    query = prepare_brep_query(model)
    with coordinate_source_request((query,)) as budget:
        return _face_certificates(query.faces, budget=budget)


@_EXECUTORS
@pytest.mark.parametrize("jet", ("value", "gradient", "hessian"))
def test_source_face_distance_jets_lower_to_bitwise_identical_programs(
    cylinder_certificates: tuple[FaceCertificate, ...], jet: str, executor: Executor
) -> None:
    for index, certificate in enumerate(cylinder_certificates):
        program = {
            "value": certificate.value,
            "gradient": certificate.gradient,
            "hessian": certificate.hessian,
        }[jet]
        _assert_identical(program, executor, 5, 97 * index + len(jet))


@_EXECUTORS
def test_trigonometric_and_reduction_edge_cases_match_the_interpreter(
    executor: Executor,
) -> None:
    program = prepare_interval_function(_rotation, 3, batch_capacity=1)
    native = program.native
    assert native is not None
    owner = _interpreted(program)
    special = (
        0.0,
        -0.0,
        1.0e-320,
        math.pi,
        -math.pi,
        0.5 * math.pi,
        2.0 * math.pi,
        1.0e16,
        -1.0e300,
        1.7976931348623157e308,
        math.inf,
        -math.inf,
    )
    for first in special:
        for second in special:
            lower = np.asarray([[min(first, second), first, -1.0]])
            upper = np.asarray(
                [[max(first, second), max(first, 2.0), second if second > -1.0 else -1.0]]
            )
            expected = owner.evaluate(lower, upper)
            actual = executor(native, lower, upper)
            assert _same(actual[0], expected[0]) and _same(actual[1], expected[1])


@_EXECUTORS
@pytest.mark.parametrize("capacity", (2, 5))
def test_batched_programs_match_the_interpreter(
    executor: Executor, capacity: int
) -> None:
    def program_function(source: Array) -> Array:
        # Broadcast operands feed a C-order innermost sum; transposed and
        # reversed operands feed indexed reads.
        outer = source[:, None] * jnp.sin(source)[None, :]
        skew = outer.T - outer
        return jnp.sum(outer, axis=-1) * source[::-1] + skew[0] + jnp.cos(source[1])

    program = prepare_interval_function(program_function, 3, batch_capacity=capacity)
    assert program.native is not None
    _assert_identical(program, executor, 3, 11 * capacity)


@_EXECUTORS
@pytest.mark.parametrize(
    "function",
    (
        lambda source: jnp.asarray([1.0, -2.0, 0.5]),
        lambda source: source,
        lambda source: source[::-1],
    ),
    ids=("constant-only", "identity", "reversed"),
)
def test_constant_identity_and_reversed_programs_preserve_interval_values(
    executor: Executor, function: Callable[[Array], Array]
) -> None:
    program = prepare_interval_function(function, 3, batch_capacity=2)
    _assert_identical(program, executor, 3, 5)


@pytest.mark.parametrize(
    "function",
    (
        lambda source: source[:2] / (1.0 + source[2] ** 2),  # div and integer_pow
        lambda source: jnp.sum(
            jnp.broadcast_to(source[0], (MAXIMUM_REDUCED_LENGTH + 1,))
        ),  # long sum
        lambda source: jnp.sum((source[:, None] * source[None, :]).T, axis=-1),
        lambda source: jax.lax.cond(source[0] > 0, jnp.sin, jnp.cos, source[1]),
        lambda source: jax.lax.fori_loop(
            0, 3, lambda _, value: value * source[0], source[1]
        ),
    ),
    ids=("division", "long-reduction", "noncontiguous-reduction", "conditional", "loop"),
)
def test_unsupported_native_programs_still_enclose_independent_point_values(
    function: Callable[[Array], Array],
) -> None:
    program = prepare_interval_function(function, 3, batch_capacity=1)
    lower = np.asarray([[-0.4, -0.2, 0.3]], dtype=np.float64)
    upper = np.asarray([[0.5, 0.6, 0.9]], dtype=np.float64)
    bounds = program.evaluate(lower, upper)
    probes = np.concatenate((lower, upper, (lower + upper) * 0.5), axis=0)
    values = np.asarray(jax.vmap(function)(jnp.asarray(probes)))
    assert np.all(values >= bounds[0])
    assert np.all(values <= bounds[1])


def test_retained_native_banks_refuse_mutation() -> None:
    program = prepare_interval_function(_rotation, 3, batch_capacity=1)
    native = program.native
    assert native is not None
    lower, upper = np.asarray([[0.25, -1.0, 2.0]]), np.asarray([[0.5, 1.0, 3.0]])
    expected = reference_execute(native, lower, upper)
    banks = (
        native.instructions,
        native.indices,
        native.parameters,
        native.template_lower,
        native.template_upper,
    )
    contents = tuple(bank.tobytes() for bank in banks)
    for bank in banks:
        for view in (bank, bank.view(), bank.reshape(-1)):
            with pytest.raises(ValueError):
                view.setflags(write=True)
            if view.size:
                with pytest.raises(ValueError, match="read-only"):
                    view.reshape(-1)[0] = 0
    assert tuple(bank.tobytes() for bank in banks) == contents
    actual = reference_execute(native, lower, upper)
    assert _same(actual[0], expected[0]) and _same(actual[1], expected[1])


def test_absent_provider_keeps_the_owner_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = prepare_interval_function(_rotation, 3, batch_capacity=2)
    assert program.native is not None
    lower, upper = _boxes(3, 3, 5)[30]
    expected = _interpreted(program).evaluate(lower, upper)

    def unavailable_kernel(*arguments: object) -> NoReturn:
        raise AssertionError("the native kernel ran without an installed provider")

    monkeypatch.setattr(meshcore_owner, "meshcore_provider_installed", lambda: False)
    monkeypatch.setattr(meshcore_owner, "interval_program_execute", unavailable_kernel)
    actual = program.evaluate(lower, upper)
    assert _same(actual[0], expected[0]) and _same(actual[1], expected[1])


def test_installed_provider_load_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = prepare_interval_function(_rotation, 3, batch_capacity=2)
    assert program.native is not None
    lower, upper = _boxes(3, 4, 2)[30]

    def refused_load() -> NoReturn:
        raise MeshcoreUnavailableError(
            "meshcore library implements another C ABI contract"
        )

    monkeypatch.setattr(meshcore_owner, "meshcore_provider_installed", lambda: True)
    monkeypatch.setattr(meshcore_owner, "load_meshcore", refused_load)
    with pytest.raises(MeshcoreUnavailableError, match="another C ABI contract"):
        program.evaluate(lower, upper)


@pytest.mark.meshcore
@_NATIVE
def test_prepared_evaluation_preserves_bounds_across_padded_chunks(
    cylinder_certificates: tuple[FaceCertificate, ...],
) -> None:
    programs = [
        program
        for certificate in cylinder_certificates
        for program in (certificate.value, certificate.gradient, certificate.hessian)
    ]
    programs.append(prepare_interval_function(_rotation, 3, batch_capacity=4))
    cases = []
    for index, program in enumerate(programs):
        # Seven rows exercise padded chunks of every capacity.
        boxes = _boxes(program.input_dimension, index, 7)[::9]
        owner = _interpreted(program)
        cases.append((program, [(box, owner.evaluate(*box)) for box in boxes]))
    for program, evaluations in cases:
        for (lower, upper), expected in evaluations:
            actual = program.evaluate(lower, upper)
            assert _same(actual[0], expected[0]) and _same(actual[1], expected[1])


@pytest.mark.meshcore
@_NATIVE
def test_invalid_native_programs_are_refused_before_any_write() -> None:
    def program_function(source: Array) -> Array:
        return jnp.sum(jnp.sin(source[::-1]) * source) + source[0]

    program = prepare_interval_function(program_function, 3, batch_capacity=1)
    native = program.native
    assert native is not None
    rows = native.instructions
    indexed = int(np.flatnonzero(rows[:, 4] >= 0)[0])
    reduction = int(np.flatnonzero(rows[:, 0] == REDUCE_SUM)[0])
    lower, upper = np.asarray([[0.25, -1.0, 2.0]]), np.asarray([[0.5, 1.0, 3.0]])
    inputs = (lower.copy(), upper.copy())
    templates = (native.template_lower.copy(), native.template_upper.copy())

    def mutated(row: int, column: int, value: int) -> NativeIntervalProgram:
        changed = np.array(rows)
        changed[row, column] = value
        return replace(native, instructions=changed)

    corrupted = (
        mutated(0, 0, 0),  # unknown opcode
        mutated(indexed, 3, int(rows[indexed, 1])),  # reads its own output region
        mutated(indexed, 4, native.indices.size),  # index bank overrun
        mutated(reduction, 6, native.parameters.size),  # parameter bank overrun
        mutated(len(rows) - 1, 2, 1 << 62),  # workspace overrun
        mutated(reduction, 7, MAXIMUM_REDUCED_LENGTH + 1),  # long reduction
    )
    for candidate in corrupted:
        with pytest.raises(ValueError, match="interval_program_execute"):
            meshcore_owner.interval_program_execute(candidate, lower, upper)
        assert _same(lower, inputs[0]) and _same(upper, inputs[1])
        assert _same(native.template_lower, templates[0])
        assert _same(native.template_upper, templates[1])
    expected = _interpreted(program).evaluate(lower, upper)
    actual = program.evaluate(lower, upper)
    assert _same(actual[0], expected[0]) and _same(actual[1], expected[1])


@pytest.mark.meshcore
@_NATIVE
def test_native_workspace_obeys_original_memory_allowance_and_retains_output() -> None:
    program = prepare_interval_function(_rotation, 3, batch_capacity=1)
    native = program.native
    assert native is not None
    lower = np.asarray([[0.25, -1.0, 2.0]], dtype=np.float64)
    upper = np.asarray([[0.5, 1.0, 3.0]], dtype=np.float64)
    expected = _interpreted(program).evaluate(lower, upper)

    def scope_with_bytes(capacity: int) -> meshcore_owner.NativeExecutionScope:
        return meshcore_owner.NativeExecutionScope(
            meshcore_owner.NativeExecutionLimits(0, 0, 0, capacity, math.inf)
        )

    with scope_with_bytes(65536) as observed:
        observed.observe_managed_admission()
        result = meshcore_owner.interval_program_execute(native, lower, upper)
    evidence = observed.evidence
    assert evidence is not None
    peak = evidence.memory_counters.peak_bytes
    assert peak >= native.template_lower.nbytes + native.template_upper.nbytes
    assert evidence.managed_admission_peak_bytes_upper == peak
    assert evidence.work_counters.construction_work == 0
    # These views retain their native allocation beyond the original scope.
    assert _same(result[0], expected[0]) and _same(result[1], expected[1])
    with scope_with_bytes(peak) as exact:
        exact.observe_managed_admission()
        bounded = meshcore_owner.interval_program_execute(native, lower, upper)
    assert _same(bounded[0], expected[0]) and _same(bounded[1], expected[1])
    with pytest.raises(meshcore_owner.MeshcoreError) as refusal:
        with scope_with_bytes(peak - 1) as insufficient:
            insufficient.observe_managed_admission()
            meshcore_owner.interval_program_execute(native, lower, upper)
    assert refusal.value.status is meshcore_owner.MeshcoreStatus.CAPACITY_EXCEEDED
    assert insufficient.evidence is not None
    assert insufficient.evidence.memory_counters.refusals == 1
    assert _same(result[0], expected[0]) and _same(result[1], expected[1])
