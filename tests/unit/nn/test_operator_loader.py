#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import shutil
import threading
import time
from dataclasses import replace
from typing import Any, cast

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint
from tools.operator_benchmarks.data_plane import run_data_plane_benchmark


def _dataset(
    *,
    cases=9,
    resolution=4,
    input_offset=0.0,
    target_offset=0.0,
    node_offset=0.0,
    weight_offset=0.0,
    masked=False,
    provenance=None,
):
    nodes = jnp.linspace(0.0, 1.0, resolution) + node_offset
    weights = jnp.full((resolution,), 1.0 / resolution)
    weights = weights.at[0].add(weight_offset)
    axis = phx.nn.operator.OperatorAxis(
        "x",
        nodes,
        quadrature_weights=weights,
    )
    values = jnp.arange(cases, dtype=float)[:, None] + nodes[None, :] + input_offset
    query_mask = jnp.arange(resolution) != resolution - 1 if masked else None
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"solution": 2.0 * values + target_offset},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
        query_mask=query_mask,
        provenance=provenance,
    )


def _callback_source(*, size=8, safe=False, fail_at=None, fingerprint="cases-v1"):
    metadata_reads = []
    case_reads = []
    reader_threads = []
    coordinates = jnp.arange(3.0)[:, None]
    weights = jnp.full((3,), 1.0 / 3.0)

    def samples(*, values=None):
        return phx.nn.operator.FunctionSamples(
            values=values,
            coordinates=coordinates,
            quadrature_weights=weights,
        )

    def metadata_reader(index):
        metadata_reads.append(index)
        geometry = samples()
        return phx.nn.operator.OperatorCaseMetadata(
            inputs={"state": geometry},
            queries={"query": geometry},
        )

    def case_reader(index, request):
        del request
        case_reads.append(index)
        reader_threads.append(threading.current_thread().name)
        if index == fail_at:
            raise RuntimeError(f"reader failed at case {index}")
        batch = phx.nn.operator.OperatorBatch(
            inputs={"state": samples(values=jnp.full((3,), float(index)))},
            queries={"query": samples()},
        )
        targets = phx.nn.operator.OperatorTargetBatch.from_arrays(
            {"solution": jnp.full((3,), 2.0 * index)},
            batch,
        )
        return phx.nn.operator.OperatorCase(batch, targets)

    source = phx.nn.operator.CallbackOperatorCaseSource(
        size,
        metadata_reader=metadata_reader,
        case_reader=case_reader,
        content_fingerprint=f"test:{fingerprint}",
        background_read_safe=safe,
        configuration={"adapter": "unit-test", "revision": 1},
    )
    return source, metadata_reads, case_reads, reader_threads


def _wait_until(predicate, *, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("Condition did not become true before the deadline.")
        time.sleep(0.005)


def test_array_fingerprint_golden_vector_is_stable():
    fingerprint = array_tree_fingerprint(
        {
            "x": jnp.arange(6, dtype=jnp.float32).reshape(2, 3),
            "y": jnp.asarray([True, False]),
        }
    )
    assert (
        fingerprint["sha256"]
        == "cfa2b27e3a5889ee3a170636ec835b3e7f7370be9770a4bf1165ab64ba95e735"
    )
    with pytest.raises(TypeError, match="object dtype"):
        array_tree_fingerprint(np.asarray([object()], dtype=object))


def test_dataset_and_loader_fingerprints_cover_content_but_not_prefetch():
    provenance = tuple(
        phx.nn.operator.OperatorCaseProvenance(
            f"case-{index}",
            identities={"trajectory": f"run-{index // 2}"},
            order={"time": float(index)},
        )
        for index in range(9)
    )
    baseline = _dataset(provenance=provenance)
    equivalent = _dataset(provenance=provenance)
    baseline_fingerprint = phx.nn.operator.training.operator_dataset_fingerprint(baseline)
    assert baseline.case_log_weights is not None
    assert baseline.case_mask is not None

    assert (
        phx.nn.operator.training.operator_dataset_fingerprint(equivalent)
        == baseline_fingerprint
    )
    assert (
        phx.nn.operator.training.operator_dataset_fingerprint(
            _dataset(input_offset=0.25, provenance=provenance)
        )
        != baseline_fingerprint
    )
    assert (
        phx.nn.operator.training.operator_dataset_fingerprint(
            _dataset(target_offset=0.25, provenance=provenance)
        )
        != baseline_fingerprint
    )
    assert (
        phx.nn.operator.training.operator_dataset_fingerprint(
            _dataset(node_offset=0.25, provenance=provenance)
        )
        != baseline_fingerprint
    )
    assert (
        phx.nn.operator.training.operator_dataset_fingerprint(
            _dataset(weight_offset=0.01, provenance=provenance)
        )
        != baseline_fingerprint
    )
    assert (
        phx.nn.operator.training.operator_dataset_fingerprint(
            _dataset(masked=True, provenance=provenance)
        )
        != baseline_fingerprint
    )

    explicit_equivalent = phx.nn.operator.training.OperatorDataset(
        baseline.batch,
        baseline.targets,
        baseline.provenance,
        case_log_weights=jnp.zeros((baseline.size,)),
        case_mask=jnp.ones((baseline.size,), dtype=bool),
    )
    changed_case_metadata = (
        replace(
            baseline,
            case_log_weights=baseline.case_log_weights.at[0].set(0.25),
        ),
        replace(
            baseline,
            case_mask=baseline.case_mask.at[0].set(False),
        ),
    )
    assert (
        phx.nn.operator.training.operator_dataset_fingerprint(explicit_equivalent)
        == baseline_fingerprint
    )
    for changed in changed_case_metadata:
        assert (
            phx.nn.operator.training.operator_dataset_fingerprint(changed)
            != baseline_fingerprint
        )

    changed_case_id = (replace(provenance[0], case_id="other-case"), *provenance[1:])
    changed_identity = (
        replace(provenance[0], identities={"trajectory": "other-run"}),
        *provenance[1:],
    )
    changed_order = (
        replace(provenance[0], order={"time": 99.0}),
        *provenance[1:],
    )
    for changed in (changed_case_id, changed_identity, changed_order):
        assert (
            phx.nn.operator.training.operator_dataset_fingerprint(
                phx.nn.operator.training.OperatorDataset(
                    baseline.batch, baseline.targets, changed
                )
            )
            != baseline_fingerprint
        )

    synchronous = phx.nn.operator.training.OperatorBatchLoader(
        baseline,
        batch_size=3,
        shuffle=True,
        seed=5,
        prefetch=0,
    )
    prefetched = phx.nn.operator.training.OperatorBatchLoader(
        equivalent,
        batch_size=3,
        shuffle=True,
        seed=5,
        prefetch=4,
    )
    changed_seed = phx.nn.operator.training.OperatorBatchLoader(
        equivalent,
        batch_size=3,
        shuffle=True,
        seed=6,
        prefetch=4,
    )

    assert synchronous.fingerprint == prefetched.fingerprint
    assert synchronous.fingerprint != changed_seed.fingerprint
    assert synchronous.fingerprint == synchronous.fingerprint
    identical_metadata = phx.nn.operator.training.OperatorBatchLoader(
        explicit_equivalent,
        batch_size=3,
        shuffle=True,
        seed=5,
        prefetch=0,
    )
    changed_metadata = tuple(
        phx.nn.operator.training.OperatorBatchLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            seed=5,
            prefetch=0,
        )
        for dataset in changed_case_metadata
    )

    assert synchronous.fingerprint == identical_metadata.fingerprint
    assert all(
        synchronous.fingerprint != loader.fingerprint for loader in changed_metadata
    )


def test_in_memory_loader_preserves_indexed_case_weights_and_masks():
    dataset = _dataset(cases=5)
    log_weights = jnp.asarray([-2.0, 0.5, 1.25, -0.75, 3.0])
    case_mask = jnp.asarray([True, False, True, True, False])
    weighted = phx.nn.operator.training.OperatorDataset(
        dataset.batch,
        dataset.targets,
        dataset.provenance,
        case_log_weights=log_weights,
        case_mask=case_mask,
    )
    loader = phx.nn.operator.training.OperatorBatchLoader(
        weighted,
        batch_size=3,
        shuffle=False,
        prefetch=0,
    )

    selected = loader.prepare_indices((3, 1, 4), epoch=2, batch_index=0)

    assert jnp.array_equal(
        selected.case_log_weights,
        log_weights[jnp.asarray((3, 1, 4))],
    )
    assert jnp.array_equal(
        selected.case_mask,
        case_mask[jnp.asarray((3, 1, 4))],
    )


def test_exact_resume_rejects_changed_case_weight_or_mask(tmp_path):
    dataset = _dataset(cases=5, resolution=8)
    assert dataset.case_log_weights is not None
    assert dataset.case_mask is not None
    checkpoint = tmp_path / "case-metadata-checkpoint"
    common: dict[str, Any] = {
        "epochs": 2,
        "batch_size": 2,
        "seed": 31,
        "checkpoint_every": 1,
        "checkpoint_path": checkpoint,
        "configuration": {"test_contract": "case-metadata-exact-resume"},
    }
    phx.nn.operator.training.fit_operator(
        _fit_model(),
        dataset,
        steps=1,
        **common,
    )

    changed_metadata = (
        replace(
            dataset,
            case_log_weights=dataset.case_log_weights.at[0].set(0.5),
        ),
        replace(
            dataset,
            case_mask=dataset.case_mask.at[0].set(False),
        ),
    )
    for changed in changed_metadata:
        with pytest.raises(ValueError, match="data contract mismatch"):
            phx.nn.operator.training.fit_operator(
                _fit_model(),
                changed,
                steps=2,
                resume=True,
                **common,
            )


def test_loader_fingerprint_and_public_epoch_plan_contract_are_stable():
    loader = phx.nn.operator.training.OperatorBatchLoader(
        _dataset(cases=6, resolution=4),
        batch_size=4,
        shuffle=True,
        seed=23,
        drop_last=False,
        prefetch=2,
        split="train",
    )
    positional = phx.nn.operator.training.OperatorEpochPlan(7, 3, True, 5, 2, False)
    keyword = phx.nn.operator.training.OperatorEpochPlan(
        source_size=7,
        batch_size=3,
        shuffle=True,
        seed=5,
        epoch=2,
        drop_last=False,
    )

    assert (
        loader.fingerprint
        == "a4f52f4c52fc978f134d7253f984518e5bf04e2edab1c8939749143b867bc395"
    )
    assert type(positional) is phx.nn.operator.training.OperatorEpochPlan
    assert positional == keyword
    assert tuple(positional) == tuple(keyword)


def test_callback_fingerprint_lookup_performs_no_hidden_reads():
    source, metadata_reads, case_reads, _ = _callback_source(size=100)
    loader = phx.nn.operator.training.OperatorBatchLoader(
        source,
        batch_size=7,
        shuffle=True,
        seed=8,
        prefetch=3,
    )

    first = loader.fingerprint
    second = loader.fingerprint

    assert first == second
    assert metadata_reads == []
    assert case_reads == []


def test_prefetch_preserves_order_and_has_bounded_read_ahead():
    source, _, case_reads, reader_threads = _callback_source(size=7, safe=True)
    loader = phx.nn.operator.training.OperatorBatchLoader(
        source,
        batch_size=1,
        shuffle=False,
        prefetch=2,
    )

    with loader.epoch(0) as batches:
        _wait_until(lambda: len(case_reads) == 2)
        assert len(case_reads) == 2
        first = next(batches)
        assert first.indices == (0,)
        _wait_until(lambda: len(case_reads) == 3)
        assert len(case_reads) == 3

    assert all(name.startswith("phydrax-operator-epoch-") for name in reader_threads)
    assert not any(
        thread.name.startswith("phydrax-operator-epoch-")
        for thread in threading.enumerate()
    )


def test_unsafe_callback_source_stays_synchronous():
    source, _, case_reads, reader_threads = _callback_source(size=4, safe=False)
    loader = phx.nn.operator.training.OperatorBatchLoader(
        source,
        batch_size=1,
        shuffle=False,
        prefetch=4,
    )
    main_thread = threading.current_thread().name

    assert loader.effective_prefetch == 0
    with loader.epoch(0) as batches:
        assert next(batches).indices == (0,)

    assert case_reads == [0]
    assert reader_threads == [main_thread]


def test_prefetch_matches_synchronous_batches_and_propagates_reader_errors():
    dataset = _dataset(cases=11)
    synchronous = phx.nn.operator.training.OperatorBatchLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=19,
        prefetch=0,
    )
    prefetched = phx.nn.operator.training.OperatorBatchLoader(
        dataset,
        batch_size=3,
        shuffle=True,
        seed=19,
        prefetch=3,
    )

    with synchronous.epoch(2) as sync_batches:
        sync_result = tuple(
            (batch.indices, batch.batch.input("state").values) for batch in sync_batches
        )
    with prefetched.epoch(2) as async_batches:
        async_result = tuple(
            (batch.indices, batch.batch.input("state").values) for batch in async_batches
        )

    assert tuple(indices for indices, _ in sync_result) == tuple(
        indices for indices, _ in async_result
    )
    assert all(
        sync_values is not None
        and async_values is not None
        and jnp.array_equal(sync_values, async_values)
        for (_, sync_values), (_, async_values) in zip(
            sync_result,
            async_result,
            strict=True,
        )
    )

    source, _, _, _ = _callback_source(size=4, safe=True, fail_at=1)
    failing = phx.nn.operator.training.OperatorBatchLoader(
        source,
        batch_size=1,
        shuffle=False,
        prefetch=2,
    )
    with pytest.raises(RuntimeError, match="reader failed at case 1"):
        with failing.epoch(0) as batches:
            tuple(batches)
    assert not any(
        thread.name.startswith("phydrax-operator-epoch-")
        for thread in threading.enumerate()
    )


def _logged_dataset_source(dataset, reads, *, fingerprint=None):
    backing = phx.nn.operator.InMemoryOperatorCaseSource(dataset)

    def metadata_reader(index):
        return backing.case_metadata(index)

    def case_reader(index, request):
        reads.append(index)
        return backing.read_case(index, request=request)

    return phx.nn.operator.CallbackOperatorCaseSource(
        dataset.size,
        metadata_reader=metadata_reader,
        case_reader=case_reader,
        content_fingerprint=(
            backing.content_fingerprint if fingerprint is None else fingerprint
        ),
        background_read_safe=True,
        configuration={"adapter": "logged-in-memory", "revision": 1},
    )


def _fit_model():
    return phx.nn.operator.architectures.FNO(
        in_channels="scalar",
        out_channels="scalar",
        width=4,
        depth=1,
        n_modes=(3,),
        key=jr.key(2),
    )


def test_lazy_fit_resumes_at_short_final_batch_and_rejects_source_before_reads(
    tmp_path,
):
    dataset = _dataset(cases=5, resolution=8)
    common: dict[str, Any] = {
        "epochs": 2,
        "batch_size": 2,
        "gradient_accumulation": 2,
        "seed": 17,
        "checkpoint_every": 1,
        "configuration": {"test_contract": "lazy-exact-resume-v2"},
    }

    uninterrupted_reads = []
    uninterrupted = phx.nn.operator.training.fit_operator(
        _fit_model(),
        _logged_dataset_source(dataset, uninterrupted_reads),
        steps=3,
        prefetch=0,
        **common,
    )

    checkpoint = tmp_path / "lazy-fit-checkpoint"
    resumed_reads = []
    resumable_source = _logged_dataset_source(dataset, resumed_reads)
    phx.nn.operator.training.fit_operator(
        _fit_model(),
        resumable_source,
        steps=1,
        prefetch=0,
        checkpoint_path=checkpoint,
        **common,
    )
    resumed_reads.clear()
    expected_first_indices = (
        phx.nn.operator.training.OperatorBatchLoader(
            resumable_source,
            batch_size=2,
            shuffle=True,
            seed=17,
            prefetch=3,
        )
        .epoch_plan(0)
        .batch(2)
    )

    resumed = phx.nn.operator.training.fit_operator(
        _fit_model(),
        resumable_source,
        steps=3,
        prefetch=3,
        checkpoint_path=checkpoint,
        resume=True,
        **common,
    )

    assert tuple(resumed_reads[: len(expected_first_indices)]) == expected_first_indices
    assert resumed.resumed_from_step == 1
    assert resumed.history == uninterrupted.history
    for expected, actual in zip(
        jax.tree_util.tree_leaves(uninterrupted.last_execution_model),
        jax.tree_util.tree_leaves(resumed.last_execution_model),
        strict=True,
    ):
        if isinstance(expected, jax.Array):
            assert jnp.array_equal(expected, actual)
    assert not any(
        thread.name.startswith("phydrax-operator-epoch-")
        for thread in threading.enumerate()
    )

    incompatible_reads = []
    incompatible_source = _logged_dataset_source(
        dataset,
        incompatible_reads,
        fingerprint="test:different-content",
    )
    with pytest.raises(ValueError, match="data contract mismatch"):
        phx.nn.operator.training.fit_operator(
            _fit_model(),
            incompatible_source,
            steps=4,
            prefetch=1,
            checkpoint_path=checkpoint,
            resume=True,
            **common,
        )
    assert incompatible_reads == []

    changed_order_reads = []
    with pytest.raises(ValueError, match="data contract mismatch"):
        phx.nn.operator.training.fit_operator(
            _fit_model(),
            _logged_dataset_source(dataset, changed_order_reads),
            steps=4,
            prefetch=1,
            checkpoint_path=checkpoint,
            resume=True,
            **cast(dict[str, Any], {**common, "seed": 18}),
        )
    assert changed_order_reads == []

    corrupt_checkpoint = tmp_path / "corrupt-lazy-fit-checkpoint"
    shutil.copytree(checkpoint, corrupt_checkpoint)
    corrupt_state = next(corrupt_checkpoint.glob("state-*.eqx"))
    corrupt_state.write_bytes(corrupt_state.read_bytes() + b"corrupt")
    corrupt_reads = []
    with pytest.raises(ValueError, match="checksum mismatch"):
        phx.nn.operator.training.fit_operator(
            _fit_model(),
            _logged_dataset_source(dataset, corrupt_reads),
            steps=4,
            prefetch=1,
            checkpoint_path=corrupt_checkpoint,
            resume=True,
            **common,
        )
    assert corrupt_reads == []

    manifest_path = checkpoint / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("version")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    old_format_reads = []
    with pytest.raises(ValueError, match="current canonical fields"):
        phx.nn.operator.training.fit_operator(
            _fit_model(),
            _logged_dataset_source(dataset, old_format_reads),
            steps=4,
            prefetch=1,
            checkpoint_path=checkpoint,
            resume=True,
            **common,
        )
    assert old_format_reads == []


def test_data_plane_benchmark_reports_correctness_and_identity_gates():
    result = run_data_plane_benchmark(
        cases=7,
        batch_size=3,
        resolution=4,
        prefetch=2,
        repetitions=1,
    )

    assert result.exact_order_match
    assert result.fingerprint_case_reads == 0
    assert result.resume_gate_case_reads == 0
    assert result.current_batch_device_bytes > 0
    assert result.sync_peak_host_bytes > 0
    assert result.prefetched_peak_host_bytes > 0
