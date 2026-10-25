"""CPU acceptance coverage for planning, admission, and explicit execution requests."""

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax import config
from phydrax.execution import default_device
from phydrax.tensor_network import (
    ContractionExecutionPolicy,
    ContractionStructure,
    PreparedContraction,
    PreparedSlicedContraction,
)


tn = phx.tensor_network


@pytest.fixture(autouse=True)
def _cpu_execution() -> Iterator[None]:
    with (
        config.context(config.NumericsConfig(enable_x64=True)),
        default_device(jax.devices("cpu")[0]),
    ):
        yield


def _structure(
    inputs: Sequence[str], outputs: str, dimensions: Mapping[str, int]
) -> ContractionStructure:
    return tn.ContractionStructure(
        tuple(
            tn.ContractionOperand(
                f"operand-{index}",
                tuple(tn.ContractionLeg(label, dimensions[label]) for label in labels),
            )
            for index, labels in enumerate(inputs)
        ),
        tuple(outputs),
    )


def _chain() -> ContractionStructure:
    return _structure(("ij", "jk", "kl"), "il", dict(i=2, j=3, k=4, l=5))


@pytest.mark.parametrize(
    "path,error",
    [
        (42, TypeError),
        ((0, 1), TypeError),
        (((0, "1"), (0, 1)), ValueError),
        (((0, 1.0), (0, 1)), ValueError),
        (((False, 1), (0, 1)), ValueError),
        (((0, True), (0, 1)), ValueError),
        (((0, 0), (0, 1)), ValueError),
        (((-1, 0), (0, 1)), ValueError),
        (((), (0, 1)), ValueError),
        (((0, 3), (0, 1)), ValueError),
        (((0, 1), (0, 2)), ValueError),
        (((0, 1),), ValueError),
        ((), ValueError),
    ],
)
def test_imported_path_rejects_malformed_or_incomplete_live_indices(
    path: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        tn.plan_contraction(_chain(), path=path, dtype="float64")


@pytest.mark.parametrize(
    "options",
    [
        {"path": ((0, 1), (0, 1)), "optimizer": "greedy"},
        {"path": ((0, 1), (0, 1)), "path_planner": tn.ContractionPathPlanner()},
        {"optimizer": "optimal", "path_planner": tn.ContractionPathPlanner()},
    ],
)
def test_explicit_planning_sources_are_mutually_exclusive(
    options: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        tn.plan_contraction(_chain(), dtype="float64", **options)


def test_imported_path_identity_and_resource_cost_are_recomputed() -> None:
    left_path = ((0, 1), (0, 1))
    right_path = ((1, 2), (0, 1))
    left = tn.plan_contraction(_chain(), path=left_path, dtype="float64")
    same = tn.plan_contraction(_chain(), path=[[0, 1], [0, 1]], dtype="float64")
    right = tn.plan_contraction(_chain(), path=right_path, dtype="float64")
    assert left.path == left.schedule.path == left_path
    assert right.path == right.schedule.path == right_path
    assert left.optimizer == right.optimizer == "imported"
    assert left.plan_id == same.plan_id
    assert left.path_provenance == same.path_provenance
    assert left.plan_id != right.plan_id
    assert left.path_provenance != right.path_provenance
    # Independent live-set arithmetic: input sizes 6 + 12 + 20; first
    # temporaries are respectively 2*4 and 3*5 elements.
    assert left.cost.operand_elements == right.cost.operand_elements == 38
    assert left.cost.output_elements == right.cost.output_elements == 10
    assert left.cost.peak_live_elements == 46
    assert right.cost.peak_live_elements == 53
    assert left.cost.estimated_flops == 2 * 3 * 4 + 2 * 4 * 5
    assert right.cost.estimated_flops == 3 * 4 * 5 + 2 * 3 * 5
    assert left.cost.workspace_bytes == (38 + 46 + 10) * 8
    assert right.cost.workspace_bytes == (38 + 53 + 10) * 8
    assert left.cost.planner_search_bound == right.cost.planner_search_bound == 0

    arrays = tuple(
        np.arange(size, dtype=np.float64).reshape(shape) / 7
        for size, shape in ((6, (2, 3)), (12, (3, 4)), (20, (4, 5)))
    )
    expected = np.einsum("ij,jk,kl->il", *arrays)
    for plan in (left, right):
        np.testing.assert_allclose(
            tn.execute_contraction(tn.prepare_contraction(plan, arrays)).value, expected
        )


@pytest.mark.parametrize(
    "limit,value,match",
    [
        ("maximum_operand_elements", 37, "operand"),
        ("maximum_output_elements", 9, "output"),
        ("maximum_intermediate_elements", 9, "intermediate"),
        ("maximum_workspace_bytes", 751, "workspace"),
        ("maximum_flops", 63, "flops"),
        ("maximum_schedule_steps", 1, "schedule"),
    ],
)
def test_imported_paths_still_obey_resource_admission(
    limit: str, value: int, match: str
) -> None:
    with pytest.raises(MemoryError, match=match):
        tn.plan_contraction(
            _chain(),
            path=((0, 1), (0, 1)),
            dtype="float64",
            resources=tn.ContractionResourcePolicy(**{limit: value}),
        )


@pytest.mark.parametrize("provider", ["", "auto", "cuda", "CUTENSOR"])
def test_unknown_execution_provider_is_not_an_automatic_fallback(provider: str) -> None:
    with pytest.raises(ValueError):
        ContractionExecutionPolicy(provider)  # ty: ignore[invalid-argument-type]


def test_execution_provider_refuses_wrong_object_kind() -> None:
    with pytest.raises(TypeError):
        ContractionExecutionPolicy(None)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "field",
    [
        "maximum_workspace_bytes",
        "maximum_cache_bytes",
        "maximum_cache_entries",
        "maximum_batch_size",
        "maximum_derivative_bytes",
    ],
)
@pytest.mark.parametrize(
    "value,error",
    [(True, TypeError), (1.5, TypeError), (0, ValueError), (-1, ValueError)],
)
def test_execution_policy_requires_positive_integer_bounds(
    field: str, value: Any, error: type[Exception]
) -> None:
    with pytest.raises(error):
        tn.ContractionExecutionPolicy(**{field: value})


@pytest.mark.parametrize(
    "options,error",
    [
        ({"maximum_search_iterations": True}, TypeError),
        ({"maximum_search_iterations": 0}, ValueError),
        ({"maximum_workspace_bytes": 1.5}, TypeError),
        ({"maximum_workspace_bytes": 0}, ValueError),
        ({"seed": False}, TypeError),
        ({"seed": -1}, ValueError),
        ({"seed": 2**31}, ValueError),
        ({"maximum_planning_seconds": True}, TypeError),
        ({"maximum_planning_seconds": "2.0"}, TypeError),
        ({"maximum_planning_seconds": 0}, ValueError),
        ({"maximum_planning_seconds": float("inf")}, ValueError),
        ({"maximum_planning_seconds": float("nan")}, ValueError),
    ],
)
def test_optional_path_planner_validates_without_loading_a_provider(
    options: dict[str, Any], error: type[Exception]
) -> None:
    with pytest.raises(error):
        tn.ContractionPathPlanner(**options)


def test_policy_identity_binds_provider_and_resource_bounds() -> None:
    policy = tn.ContractionExecutionPolicy()
    assert policy.provider == "native"
    assert policy.policy_id == tn.ContractionExecutionPolicy("native").policy_id
    assert policy.policy_id != tn.ContractionExecutionPolicy("cutensor").policy_id
    assert (
        policy.policy_id
        != tn.ContractionExecutionPolicy(maximum_cache_entries=1).policy_id
    )
    assert (
        tn.ContractionPathPlanner().planner_id == tn.ContractionPathPlanner().planner_id
    )
    assert (
        tn.ContractionPathPlanner().planner_id
        != tn.ContractionPathPlanner(seed=1).planner_id
    )


def _matrix_prepared() -> PreparedContraction:
    structure = _structure(("ij", "jk"), "ki", dict(i=2, j=3, k=4))
    plan = tn.plan_contraction(structure, path=((0, 1),), dtype="float64")
    operands = (jnp.arange(6.0).reshape(2, 3), jnp.arange(12.0).reshape(3, 4))
    return tn.prepare_contraction(plan, operands)


def test_default_native_execution_evidence_and_refresh_preserve_executor() -> None:
    prepared = _matrix_prepared()
    executor = prepared.executor
    assert executor.policy.provider == "native"
    assert executor.external_workspace_bound_bytes == 0
    assert executor.external_cache_bound_bytes == 0
    assert executor.normalization_bytes == 0
    result = eqx.filter_jit(tn.execute_contraction)(prepared)
    assert result.evidence.provider == "native"
    assert result.evidence.execution_id == executor.lowering_id
    np.testing.assert_allclose(
        result.value,
        (np.asarray(prepared.operands[0]) @ np.asarray(prepared.operands[1])).T,
    )
    refreshed = tn.refresh_contraction(
        prepared, (prepared.operands[0] + 1, prepared.operands[1])
    )
    assert refreshed.executor is executor
    assert refreshed.plan is prepared.plan
    assert refreshed.prepared_id != prepared.prepared_id
    assert refreshed.operand_id != prepared.operand_id
    assert int(refreshed.numeric_version) == 1
    updated = tn.execute_contraction(refreshed)
    assert updated.evidence.execution_id == result.evidence.execution_id
    np.testing.assert_allclose(
        updated.value,
        ((np.asarray(prepared.operands[0]) + 1) @ np.asarray(prepared.operands[1])).T,
    )


@pytest.mark.parametrize("provider", ["cutensor", "cutensornet"])
def test_optional_provider_explicitly_refuses_cpu_without_transfer_or_fallback(
    provider: Literal["cutensor", "cutensornet"],
) -> None:
    prepared = _matrix_prepared()
    assert all(
        device.platform == "cpu"
        for value in prepared.operands
        for device in value.devices()
    )
    with pytest.raises(ValueError, match="CUDA-resident operands"):
        tn.prepare_contraction(
            prepared.plan,
            prepared.operands,
            execution=tn.ContractionExecutionPolicy(provider),
        )


@pytest.mark.parametrize("boundary", ["prepare", "refresh"])
def test_content_addressed_host_boundaries_refuse_traced_operands(
    boundary: Literal["prepare", "refresh"],
) -> None:
    prepared = _matrix_prepared()

    def traced(left: Array) -> Array:
        operands = (left, prepared.operands[1])
        if boundary == "prepare":
            return tn.prepare_contraction(prepared.plan, operands).operands[0]
        return tn.refresh_contraction(prepared, operands).operands[0]

    with pytest.raises(TypeError, match="cannot accept traced"):
        jax.make_jaxpr(traced)(prepared.operands[0])


def test_refresh_refuses_a_traced_numeric_version_even_with_constant_operands() -> None:
    prepared = _matrix_prepared()

    def traced(version: Array) -> Array:
        dynamic = eqx.tree_at(lambda item: item.numeric_version, prepared, version)
        return tn.refresh_contraction(dynamic, prepared.operands).numeric_version

    with pytest.raises(TypeError, match="cannot accept traced"):
        jax.make_jaxpr(traced)(jnp.asarray(0, dtype=jnp.int32))


def test_native_accumulation_is_actually_wide_not_just_output_cast() -> None:
    structure = _structure(("i", "i"), "", {"i": 3})
    precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype="float32",
        contraction_dtype="float32",
        accumulation_dtype="float64",
        output_dtype="float64",
    )
    plan = tn.plan_contraction(
        structure, path=((0, 1),), dtype="float32", precision=precision
    )
    left = np.asarray([2**25, 1, -(2**25)], dtype=np.float32)
    right = np.ones(3, dtype=np.float32)
    expected = np.sum(left.astype(np.float64) * right.astype(np.float64))
    assert expected == 1.0
    prepared = tn.prepare_contraction(plan, (left, right))
    for result in (
        tn.execute_contraction(prepared),
        eqx.filter_jit(tn.execute_contraction)(prepared),
    ):
        assert result.value.dtype == jnp.float64
        np.testing.assert_array_equal(result.value, expected)
    replay = tn.execute_schedule(
        plan.schedule, prepared.operands, accumulation_dtype="float64"
    )
    assert replay.dtype == jnp.float64
    np.testing.assert_array_equal(replay, expected)


def test_imported_nary_path_is_not_silently_split_by_native_replay() -> None:
    path = ((0, 1, 2),)
    plan = tn.plan_contraction(_chain(), path=path, dtype="float64")
    assert plan.path == plan.schedule.path == path
    assert len(plan.schedule.steps) == 1
    operands = (
        jnp.ones((2, 3), dtype=jnp.float64),
        jnp.ones((3, 4), dtype=jnp.float64),
        jnp.ones((4, 5), dtype=jnp.float64),
    )
    with pytest.raises(ValueError, match="unary/binary"):
        tn.prepare_contraction(plan, operands)
    with pytest.raises(ValueError, match="unary/binary"):
        tn.execute_schedule(plan.schedule, operands)


def test_slices_accumulate_before_the_single_final_output_cast() -> None:
    structure = _structure(("i", "i"), "", {"i": 3})
    precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype="float32",
        contraction_dtype="float32",
        accumulation_dtype="float64",
        output_dtype="float32",
    )
    plan = tn.plan_contraction(
        structure, path=((0, 1),), dtype="float32", precision=precision
    )
    prepared = tn.prepare_contraction(
        plan,
        (
            jnp.asarray([1e8, 1.0, -1e8], dtype=jnp.float32),
            jnp.ones((3,), dtype=jnp.float32),
        ),
    )
    sliced = tn.plan_sliced_contraction(plan, ("i",), batch_size=2)
    binding = tn.prepare_sliced_contraction(prepared, sliced)
    for mode in ("serial", "batched"):
        result = tn.execute_sliced_contraction(binding, mode=mode)
        assert result.value.dtype == jnp.float32
        assert result.checkpoint.scaled_sum.dtype == jnp.float64
        np.testing.assert_array_equal(result.value, np.asarray(1.0, dtype=np.float32))
    with pytest.raises(MemoryError, match="checkpoint"):
        tn.plan_sliced_contraction(
            plan,
            ("i",),
            resources=tn.SlicingResourcePolicy(maximum_checkpoint_bytes=4),
        )


@pytest.mark.parametrize("narrow_host_chunks", (False, True))
def test_empty_residual_preserves_declared_accumulation_through_placement(
    monkeypatch: pytest.MonkeyPatch, narrow_host_chunks: bool
) -> None:
    from phydrax.tensor_network import _placement

    structure = _structure(("s",), "", {"s": 3})
    precision = tn.TensorNetworkPrecisionPolicy(
        storage_dtype="float32",
        contraction_dtype="float32",
        accumulation_dtype="float64",
        output_dtype="float64",
    )
    plan = tn.plan_contraction(structure, path=(), dtype="float32", precision=precision)
    prepared = tn.prepare_contraction(
        plan, (jnp.asarray([1e8, 1.0, -1e8], dtype=jnp.float32),)
    )
    sliced = tn.plan_sliced_contraction(plan, ("s",), batch_size=2)
    assert not sliced.residual.schedule.steps
    binding = tn.prepare_sliced_contraction(prepared, sliced)
    local = tn.execute_slice_assignment(binding, jnp.asarray([1], dtype=jnp.int32))
    assert local.dtype == jnp.float64
    np.testing.assert_array_equal(local, np.asarray(1.0, dtype=np.float64))
    for mode in ("serial", "batched"):
        result = tn.execute_sliced_contraction(binding, mode=mode)
        assert result.value.dtype == jnp.float64
        np.testing.assert_array_equal(result.value, np.asarray(1.0, dtype=np.float64))
    if narrow_host_chunks:
        execute_local = _placement._execute_local_slices

        def storage_dtype_chunk(
            current: PreparedSlicedContraction, assignments: Array
        ) -> Array:
            # Exercise the host reduction boundary independently of local casts.
            return execute_local(current, assignments).astype(jnp.float32)

        monkeypatch.setattr(_placement, "_execute_local_slices", storage_dtype_chunk)
    mesh = tn.create_tensor_network_mesh((jax.devices("cpu")[0],))
    placement = tn.plan_slice_placement(sliced, mesh)
    distributed = tn.execute_distributed_slices(prepared, placement)
    assert distributed.aggregate is not None
    assert distributed.aggregate.dtype == jnp.float64
    assert bool(distributed.evidence.accepted)
    np.testing.assert_array_equal(
        distributed.aggregate, np.asarray(1.0, dtype=np.float64)
    )


def test_unique_native_paths_do_not_launch_search_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.tensor_network import _contraction

    def unexpected_worker(_method: str) -> NoReturn:
        raise AssertionError("A unique native contraction path launched a search worker.")

    monkeypatch.setattr(_contraction, "get_context", unexpected_worker)
    unary = _structure(("ii",), "", {"i": 3})
    binary = _structure(("ij", "jk"), "ki", {"i": 2, "j": 3, "k": 4})
    for optimizer in ("greedy", "optimal"):
        unary_plan = tn.plan_contraction(unary, dtype="float64", optimizer=optimizer)
        assert unary_plan.path == ()
        assert unary_plan.cost.planner_search_bound == 0
        matrix = jnp.arange(9, dtype=jnp.float64).reshape(3, 3)
        observed = tn.execute_contraction(tn.prepare_contraction(unary_plan, (matrix,)))
        np.testing.assert_array_equal(observed.value, np.trace(np.asarray(matrix)))
        binary_plan = tn.plan_contraction(binary, dtype="float64", optimizer=optimizer)
        assert binary_plan.path == ((0, 1),)
        assert binary_plan.cost.planner_search_bound == 0
        assert binary_plan.cost.estimated_flops == 2 * 2 * 3 * 4
    with pytest.raises(MemoryError):
        tn.plan_contraction(
            binary,
            dtype="float64",
            resources=tn.ContractionResourcePolicy(maximum_flops=47),
        )
    with pytest.raises(TimeoutError):
        tn.plan_contraction(
            binary,
            dtype="float64",
            planner=tn.ContractionPlannerPolicy(maximum_planning_seconds=1e-9),
        )
