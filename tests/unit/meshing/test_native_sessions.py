from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from types import TracebackType
from typing import assert_never, Literal, NoReturn

import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import tree_content_equal
from phydrax._meshcore import MeshcoreError, MeshcoreStatus, NativeParallelExecution
from phydrax.execution import ResourceRequest
from phydrax.meshing import (
    CellMeshingResult,
    MeshingExecutionPolicy,
    NativeExecutionRecord,
    NativeMeshingPhaseMeasurement,
    NativeMeshingPlan,
    NativeMeshingProvider,
)
from phydrax.meshing.providers import _gmsh as gmsh_owner
from tests._support.native_provider_cases import curve_plan, curve_spec, tee


@pytest.fixture(scope="module")
def native_plan() -> NativeMeshingPlan:
    segments, scope = tee()
    specification = curve_spec(
        scope,
        (phx.meshing.CurveJunction("hub", ((0, "start"), (1, "start"), (2, "start"))),),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.25, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
    )
    return curve_plan(segments, specification)


@pytest.fixture(scope="module")
def native_provider(native_plan: NativeMeshingPlan) -> NativeMeshingProvider:
    return NativeMeshingProvider(native_plan.options)


@pytest.fixture(scope="module")
def default_result(native_plan: NativeMeshingPlan) -> CellMeshingResult:
    return native_plan.execute()


@pytest.fixture
def lowered_requests(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, float]]:
    previous_init = NativeParallelExecution.__init__
    requested: list[tuple[int, float]] = []

    def record_policy(
        context: NativeParallelExecution, parallelism: int, maximum_seconds: float, /
    ) -> None:
        requested.append((parallelism, maximum_seconds))
        previous_init(context, parallelism, maximum_seconds)

    monkeypatch.setattr(NativeParallelExecution, "__init__", record_policy)
    return requested


def _counter_state(record: NativeExecutionRecord | None) -> tuple[object, ...] | None:
    if record is None:
        return None
    return (
        record.work_counters,
        record.memory_counters,
        record.status,
        record.externally_charged_work,
        record.externally_charged_geometry_queries,
        record.native_primitive_queries,
        record.host_storage_live_bytes_upper,
        record.host_storage_peak_bytes_upper,
        record.source_preparation_work_units,
        record.source_preparation_geometry_queries,
        record.owner_id,
        _counter_state(record.preparation_evidence),
        _counter_state(record.consumer_evidence),
    )


def _assert_result_equal(first: CellMeshingResult, second: CellMeshingResult) -> None:
    # Scientific fields and counter graphs are bitwise contracts; clocks are not.
    scientific_fields = tuple(
        field.name for field in fields(first) if field.name != "execution_evidence"
    )
    assert tree_content_equal(
        tuple(getattr(first, name) for name in scientific_fields),
        tuple(getattr(second, name) for name in scientific_fields),
    )
    assert first.execution_evidence is not None
    assert second.execution_evidence is not None
    first.execution_evidence.require_valid()
    second.execution_evidence.require_valid()
    assert tree_content_equal(
        _counter_state(first.execution_evidence),
        _counter_state(second.execution_evidence),
    )


def test_meshing_execution_policy_defaults_to_one_deterministic_thread() -> None:
    policy = MeshingExecutionPolicy()
    assert policy.parallelism == 1
    assert policy.deterministic
    assert policy.execution_mode is phx.meshing.MeshingExecutionMode.IN_PROCESS
    assert policy.timeout_seconds == 3600.0
    assert policy.maximum_transferred_bytes == 4_000_000_000
    assert policy.cleanup_workspace


@pytest.mark.parametrize(
    "threads", (1, 2, 7, 64), ids=("one", "two", "seven", "sixty-four")
)
def test_native_session_admits_deterministic_threads_with_bitwise_evidence(
    threads: int,
    native_provider: NativeMeshingProvider,
    native_plan: NativeMeshingPlan,
    default_result: CellMeshingResult,
    lowered_requests: list[tuple[int, float]],
) -> None:
    policy = MeshingExecutionPolicy(parallelism=threads)
    with native_provider.open_session(policy) as session:
        assert isinstance(session, phx.meshing.AbstractMeshingSession)
        assert isinstance(session, phx.meshing.NativeMeshingSession)
        assert not session.closed
        result = session.execute(native_plan)
        assert result.runtime.deterministic
        assert result.execution_evidence is not None
        assert (
            result.execution_evidence.preparation_evidence
            is native_plan.preparation_evidence
        )
        _assert_result_equal(result, default_result)
    assert session.closed
    assert lowered_requests == [(threads, policy.timeout_seconds)]


def test_native_provider_advertises_parallel_capability(
    native_provider: NativeMeshingProvider,
) -> None:
    assert phx.meshing.MeshingCapability.PARALLEL in native_provider.info().capabilities


def test_native_plan_default_execute_matches_explicit_default_session(
    native_provider: NativeMeshingProvider,
    native_plan: NativeMeshingPlan,
    default_result: CellMeshingResult,
) -> None:
    with native_provider.open_session() as session:
        _assert_result_equal(session.execute(native_plan), default_result)


def test_native_session_can_execute_multiple_plans_before_closing(
    native_provider: NativeMeshingProvider, native_plan: NativeMeshingPlan
) -> None:
    with native_provider.open_session(MeshingExecutionPolicy(parallelism=7)) as session:
        first = session.execute(native_plan)
        second = session.execute(native_plan)
        _assert_result_equal(first, second)
        assert not session.closed
    assert session.closed


def test_native_session_cpu_request_is_an_admission_ceiling(
    native_provider: NativeMeshingProvider,
) -> None:
    resources = ResourceRequest(cpu_cores=2, memory_bytes=1 << 30)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        native_provider.open_session(
            MeshingExecutionPolicy(parallelism=7), resources=resources
        )
    assert failure.value.category is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.value.evidence.provider_code == "cpu_ceiling_exceeded"
    assert dict(failure.value.evidence.requested)["parallelism"] == 7
    assert dict(failure.value.evidence.achieved)["cpu_cores"] == 2


def test_native_session_admits_the_cpu_ceiling_without_changing_results(
    native_provider: NativeMeshingProvider,
    native_plan: NativeMeshingPlan,
    default_result: CellMeshingResult,
) -> None:
    resources = ResourceRequest(cpu_cores=7, memory_bytes=1 << 30)
    with native_provider.open_session(
        MeshingExecutionPolicy(parallelism=7), resources=resources
    ) as session:
        _assert_result_equal(session.execute(native_plan), default_result)


def test_native_session_keeps_default_admission_under_a_larger_cpu_ceiling(
    lowered_requests: list[tuple[int, float]],
    native_provider: NativeMeshingProvider,
    native_plan: NativeMeshingPlan,
    default_result: CellMeshingResult,
) -> None:
    resources = ResourceRequest(cpu_cores=64, memory_bytes=1 << 30)
    with native_provider.open_session(resources=resources) as session:
        _assert_result_equal(session.execute(native_plan), default_result)
    assert lowered_requests == [(1, MeshingExecutionPolicy().timeout_seconds)]


@pytest.mark.parametrize(
    "mode",
    (
        phx.meshing.MeshingExecutionMode.SUBPROCESS,
        phx.meshing.MeshingExecutionMode.REMOTE,
    ),
    ids=("subprocess", "remote"),
)
def test_native_session_refuses_unsupported_execution_modes(
    mode: phx.meshing.MeshingExecutionMode, native_provider: NativeMeshingProvider
) -> None:
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        native_provider.open_session(MeshingExecutionPolicy(mode))
    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_CAPABILITY
    )
    assert failure.value.evidence.provider_code == "execution_mode_unavailable"


def test_native_session_context_is_single_use(
    native_provider: NativeMeshingProvider,
) -> None:
    session = native_provider.open_session()
    with session:
        with pytest.raises(RuntimeError, match="NativeMeshingSession.context"):
            with session:
                pytest.fail("repeated context entry was admitted")
        assert not session.closed
    assert session.closed
    with pytest.raises(RuntimeError, match="NativeMeshingSession.closed"):
        with session:
            pytest.fail("closed context entry was admitted")


def test_native_session_closed_execution_is_refused(
    native_provider: NativeMeshingProvider, native_plan: NativeMeshingPlan
) -> None:
    session = native_provider.open_session()
    session.close()
    session.close()
    assert session.closed
    with pytest.raises(RuntimeError, match="NativeMeshingSession.closed"):
        session.execute(native_plan)


@pytest.mark.parametrize("operation", ("enter", "execute", "close"))
def test_native_session_lifecycle_refuses_foreign_threads(
    operation: Literal["enter", "execute", "close"],
    native_provider: NativeMeshingProvider,
    native_plan: NativeMeshingPlan,
) -> None:
    with native_provider.open_session() as session:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(RuntimeError, match="NativeMeshingSession.thread"):
                match operation:
                    case "enter":
                        executor.submit(session.__enter__).result()
                    case "execute":
                        executor.submit(session.execute, native_plan).result()
                    case "close":
                        executor.submit(session.close).result()
                    case _:
                        assert_never(operation)
        assert not session.closed


def test_native_session_body_error_closes_without_replacing_the_error(
    native_provider: NativeMeshingProvider,
) -> None:
    error = RuntimeError("consumer.body must finish; got consumer body failure")
    with pytest.raises(RuntimeError, match="consumer body failure") as failure:
        with native_provider.open_session() as session:
            raise error
    assert failure.value is error
    assert session.closed


def test_native_session_execution_error_restores_policy_and_can_continue(
    native_provider: NativeMeshingProvider,
    native_plan: NativeMeshingPlan,
    default_result: CellMeshingResult,
) -> None:
    error = RuntimeError("consumer.phase must finish; got consumer phase failure")

    def fail_phase(_measurement: NativeMeshingPhaseMeasurement) -> NoReturn:
        raise error

    with native_provider.open_session(MeshingExecutionPolicy(parallelism=7)) as session:
        with pytest.raises(RuntimeError, match="consumer phase failure") as failure:
            session.execute(native_plan, record_phase=fail_phase)
        assert failure.value is error
        assert not session.closed
        _assert_result_equal(session.execute(native_plan), default_result)


def test_native_session_expired_timeout_preserves_resource_failure_evidence(
    native_provider: NativeMeshingProvider, native_plan: NativeMeshingPlan
) -> None:
    policy = MeshingExecutionPolicy(
        parallelism=7, timeout_seconds=np.finfo(np.float64).tiny
    )
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        with native_provider.open_session(policy) as session:
            session.execute(native_plan)
    assert failure.value.category is phx.meshing.MeshingFailureCategory.TIMED_OUT
    assert failure.value.evidence.provider_code == "timeout"
    assert session.closed


def test_native_session_policy_exit_timeout_retains_ended_scope_evidence(
    monkeypatch: pytest.MonkeyPatch,
    native_provider: NativeMeshingProvider,
    native_plan: NativeMeshingPlan,
    default_result: CellMeshingResult,
) -> None:
    previous_exit = NativeParallelExecution.__exit__

    def expire_after_exit(
        context: NativeParallelExecution,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        previous_exit(context, error_type, error, traceback)
        if error is None:
            raise MeshcoreError(
                MeshcoreStatus.TIMEOUT,
                "NativeParallelExecution.deadline must admit publication; got injected policy exit timeout",
            )

    monkeypatch.setattr(NativeParallelExecution, "__exit__", expire_after_exit)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        with native_provider.open_session() as session:
            session.execute(native_plan)
    assert failure.value.category is phx.meshing.MeshingFailureCategory.TIMED_OUT
    assert (
        failure.value.evidence.stage
        == phx.meshing.MeshingStageKind.CANONICALIZATION.value
    )
    (stage,) = failure.value.cut_stage_records
    assert stage.execution.status is MeshcoreStatus.OK
    assert default_result.execution_evidence is not None
    expected = default_result.execution_evidence.native_evidence()
    assert tree_content_equal(stage.execution.work_counters, expected.work_counters)
    assert tree_content_equal(stage.execution.memory_counters, expected.memory_counters)
    assert session.closed


@pytest.mark.parametrize("threads", (2, 7, 64), ids=("two", "seven", "sixty-four"))
def test_gmsh_refuses_deterministic_parallel_policy_before_provider_probe(
    threads: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse_provider_probe(name: str) -> NoReturn:
        pytest.fail(f"provider {name!r} was probed before policy admission")

    monkeypatch.setattr(gmsh_owner.util, "find_spec", refuse_provider_probe)
    policy = MeshingExecutionPolicy(parallelism=threads)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.GmshProvider().open_session(policy)
    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_CAPABILITY
    )
    assert failure.value.evidence.provider_code == "deterministic_parallelism_unavailable"
    assert dict(failure.value.evidence.requested)["parallelism"] == threads


def test_gmsh_nondeterministic_parallel_policy_reaches_provider_admission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable_provider(_name: str) -> None:
        return None

    monkeypatch.setattr(gmsh_owner.util, "find_spec", unavailable_provider)
    policy = MeshingExecutionPolicy(parallelism=7, deterministic=False)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.GmshProvider().open_session(policy)
    assert (
        failure.value.category is phx.meshing.MeshingFailureCategory.PROVIDER_UNAVAILABLE
    )
