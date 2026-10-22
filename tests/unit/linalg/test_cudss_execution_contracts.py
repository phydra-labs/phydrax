#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax.linalg import (
    CudssExecutionPolicy,
    CudssMemoryEvidence,
    CudssMemoryPolicy,
    sparse_provider_capabilities,
    SparseLDLT,
)
from phydrax.linalg.backends import _cudss_ffi


def _group(count: int, *, processes: bool = False) -> ExecutionGroupSpec:
    return ExecutionGroupSpec(
        "cudss-execution-contract",
        tuple(range(count)) if processes else (0,),
        tuple((index, 0) for index in range(count))
        if processes
        else tuple((0, index) for index in range(count)),
    )


def test_local_default_retains_explicit_unbounded_hybrid_scope() -> None:
    method = SparseLDLT(memory_mode="hybrid")
    assert method.execution.mode == "local"
    assert method.memory_policy.hybrid_device_limit_bytes is None
    assert not method.memory_policy.require_certified_bounds
    bounded_numeric = SparseLDLT(
        memory_mode="hybrid",
        memory_policy=CudssMemoryPolicy(hybrid_device_limit_bytes=2**35 + 17),
    )
    assert bounded_numeric.memory_policy.hybrid_device_limit_bytes == 2**35 + 17


@pytest.mark.parametrize(
    "kwargs", [{"host_limit_bytes": 1}, {"require_certified_bounds": True}]
)
def test_certified_all_phase_memory_refused_without_loading_provider(
    kwargs: dict,
) -> None:
    with pytest.raises(ValueError, match="cannot enforce certified all-phase"):
        SparseLDLT(memory_mode="hybrid", memory_policy=CudssMemoryPolicy(**kwargs))


@pytest.mark.parametrize("value", [0, -1, 2**63])
def test_hybrid_device_limit_range(value: int) -> None:
    with pytest.raises(ValueError, match="2\\*\\*63"):
        CudssMemoryPolicy(hybrid_device_limit_bytes=value)


@pytest.mark.parametrize("value", [True, 1.5, "1024"])
def test_hybrid_device_limit_type(value: object) -> None:
    with pytest.raises(TypeError, match="positive integer"):
        CudssMemoryPolicy(hybrid_device_limit_bytes=value)  # ty: ignore[invalid-argument-type]


def test_memory_mode_is_the_only_memory_selector() -> None:
    with pytest.raises(ValueError, match="requires memory_mode='hybrid'"):
        SparseLDLT(memory_policy=CudssMemoryPolicy(hybrid_device_limit_bytes=1024))
    with pytest.raises(ValueError, match="hybrid-plus-MG/MGMN"):
        SparseLDLT(
            memory_mode="hybrid", execution=CudssExecutionPolicy("mg", group=_group(2))
        )


@pytest.mark.parametrize("count", [2, 4, 8])
def test_mg_group_is_explicit_ordered_reservation(count: int) -> None:
    group = _group(count)
    method = SparseLDLT(execution=CudssExecutionPolicy("mg", group=group))
    assert method.execution.group is group
    assert method.execution.group.device_keys[0] == (0, 0)
    assert method.execution.group.device_count == count


@pytest.mark.parametrize("count", [1, 3, 16])
def test_mg_group_count_refusal(count: int) -> None:
    with pytest.raises(ValueError, match="ordered 2, 4, or 8"):
        CudssExecutionPolicy("mg", group=_group(count))


def test_mg_group_and_local_mode_are_not_interchangeable() -> None:
    with pytest.raises(ValueError, match="explicit execution group"):
        CudssExecutionPolicy("mg")
    with pytest.raises(ValueError, match="does not take"):
        CudssExecutionPolicy("local", group=_group(2))
    with pytest.raises(ValueError, match="one process"):
        CudssExecutionPolicy("mg", group=_group(2, processes=True))


def test_mgmn_requires_exact_site_communication_provenance() -> None:
    group = _group(2, processes=True)
    with pytest.raises(ValueError, match="site communication-library provenance"):
        CudssExecutionPolicy("mgmn", group=group)
    execution = CudssExecutionPolicy(
        "mgmn",
        group=group,
        communication_library_path="/site/lib/libcudss_commlayer_openmpi.so",
        communication_library_sha256="a" * 64,
        communication_build_identity="site-openmpi-pinned",
    )
    assert execution.mode == "mgmn"
    with pytest.raises(ValueError, match="only valid for cuDSS MGMN"):
        CudssExecutionPolicy("local", communication_build_identity="unbound")


def test_exact_memory_words_do_not_require_jax_x64_or_infer_host_sum() -> None:
    values = np.asarray(
        [[2**33 + 5, 2**40 + 17, 23, 29, 0, 0], [31, 2**39 + 7, 23, 29, 0, 0]],
        dtype=np.uint64,
    )
    words = np.stack(
        (values.astype(np.uint32), (values >> np.uint64(32)).astype(np.uint32)), axis=-1
    )
    words[:, 4:] = np.iinfo(np.uint32).max
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=False)):
        evidence = CudssMemoryEvidence(
            jnp.asarray(words), device_indices=(2, 3), group_id="nonzero-group"
        )
        observed = evidence.host_estimates()
        assert observed[0, 1] == 2**40 + 17
        assert observed[0, 4] is None
        resources = evidence.resource_evidence(caller_device_bytes=(11, 13))
        assert resources.per_device_peak_bytes == 2**40 + 28
        assert resources.per_host_peak_bytes is None
        assert resources.memory_basis == "provider_estimate"


def test_unknown_device_peak_stays_unknown() -> None:
    words = jnp.full((1, 6, 2), np.iinfo(np.uint32).max, dtype=jnp.uint32)
    evidence = CudssMemoryEvidence(words)
    assert (
        evidence.resource_evidence(caller_device_bytes=(1024,)).per_device_peak_bytes
        is None
    )
    with pytest.raises(ValueError, match="per device"):
        evidence.resource_evidence(caller_device_bytes=())


def test_native_config_preserves_group_order_and_full_width_limit() -> None:
    config = (5, 8, 1, 0, 0, 1, 2**35 + 17, 23, (5, 3))
    attributes = _cudss_ffi._attributes(config)
    assert attributes["hybrid_device_limit"] == 2**35 + 17
    assert attributes["group"] == 23
    np.testing.assert_array_equal(
        attributes["devices"], np.asarray([5, 3], dtype=np.int64)
    )


def test_expanded_matrix_batch_overflow_refused_before_array_allocation() -> None:
    offsets = jax.ShapeDtypeStruct((2**30 + 1,), jnp.int32)
    columns = jax.ShapeDtypeStruct((1,), jnp.int32)
    with pytest.raises(ValueError, match="overflow int32"):
        jax.eval_shape(
            lambda off, col: _cudss_ffi._expand_structure(off, col, 4),
            offsets,
            columns,
        )


def test_mg_matrix_batch_refused_before_provider_loading() -> None:
    with pytest.raises(ValueError, match="refuses matrix batches"):
        _cudss_ffi.analyze(
            jnp.ones((2, 1), dtype=jnp.float32),
            jnp.asarray([0, 1], dtype=jnp.int32),
            jnp.asarray([0], dtype=jnp.int32),
            partition=1,
            capacity=1,
            reordering="default",
            memory="device",
            group=1,
            devices=(0, 1),
        )


def test_mode_capabilities_do_not_promote_partial_inertia_or_batches() -> None:
    local = sparse_provider_capabilities("cudss")
    mg = sparse_provider_capabilities("cudss", execution_mode="mg")
    mgmn = sparse_provider_capabilities("cudss", execution_mode="mgmn")
    assert local.batched_shared_pattern and local.inertia
    assert not mg.batched_shared_pattern and not mg.inertia
    assert mg.requires_execution_group and not mg.requires_collective_lease
    assert mgmn.requires_execution_group and mgmn.requires_collective_lease
    assert not mgmn.batched_shared_pattern and not mgmn.inertia
    for capabilities in (local, mg, mgmn):
        assert capabilities.multiple_rhs
        assert not capabilities.reliable_zero_inertia
        assert not capabilities.certified_memory_bounds


def test_rank_local_estimate_does_not_claim_remote_group_capacity() -> None:
    evidence = CudssMemoryEvidence(
        jnp.zeros((1, 6, 2), dtype=jnp.uint32),
        device_indices=(0,),
        process_index=1,
        group_device_count=4,
    )
    with pytest.raises(ValueError, match="explicit all-rank resource evidence"):
        evidence.resource_evidence(caller_device_bytes=(1024,))


def test_release_uses_host_partition_without_touching_token_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An abstract ID has no reshape/device materialization route. Retirement
    # must need only the static owner even when a dispatched ID is inaccessible.
    token = _cudss_ffi.CudssToken(
        jax.ShapeDtypeStruct((1,), jnp.int32),  # ty: ignore[invalid-argument-type]
        jnp.ones((1,), dtype=jnp.float32),
        jnp.asarray([0, 1], dtype=jnp.int32),
        jnp.asarray([0], dtype=jnp.int32),
        jnp.zeros((1,), dtype=jnp.int32),
        jnp.zeros((2, 6, 2), dtype=jnp.uint32),
        "factorized",
        np.dtype(np.float32),
        1,
        1,
        1,
        (17, 8, 1, 0, 0, 0, 0, 23, (0, 1)),
    )
    retired: list[tuple[str, int]] = []

    def release(name: str, partition: int) -> int:
        retired.append((name, partition))
        return 3

    monkeypatch.setattr(_cudss_ffi, "release_cuda_partition", release)
    assert _cudss_ffi.release(token)
    assert retired == [("cudss", 17)]
