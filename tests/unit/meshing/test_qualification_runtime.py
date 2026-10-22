# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import FrozenInstanceError
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, final

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from tests._support.cold_worker_programs import IMPORTED_MARKER_VARIABLE
from tools.compile_counter import CompileCounter
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import (
    CampaignProcessTimeout,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    prepare_cold_worker,
    QualificationFailure,
    REPORTING_GRACE_SECONDS,
    run_campaign_process,
    StageLedger,
)
from tools.meshing_qualification.scenarios.design import (
    _design_qualification_condition_matrix,
    _design_qualification_derivative_probe,
    _design_qualification_linearizations,
)


# Fixed empirical residual bounds from test_qualification_runtime.py
# at the pre-migration design-probe fixture; each comparison retains its strictness.
# Design-probe derivative versus finite difference for repeated identical numerical
# signatures.
_DESIGN_DIFFERENCE_RESIDUAL = 1e-9
# Design-probe JVP/VJP primal/dual dot-product error for the retained derivative
# fixture.
_DESIGN_DUALITY_RESIDUAL = 1e-12


@pytest.fixture
def affine_problem() -> phx.equations.CompiledFiniteElementProblem:
    points = np.asarray(
        [(x, y) for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)],
        dtype=np.float64,
    )
    cells = np.asarray(
        [
            triangle
            for lower in (0, 1, 3, 4)
            for triangle in (
                (lower, lower + 1, lower + 4),
                (lower, lower + 4, lower + 3),
            )
        ],
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("cells", "triangle", cells),)
    )
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "temperature", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()

    def affine(coordinates: jax.Array) -> jax.Array:
        return coordinates[..., 0] + coordinates[..., 1]

    return phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "compile-hygiene-affine-diffusion",
            "temperature",
            (phx.equations.DiffusionAction("temperature", 1.0),),
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(space, "temperature"),
        dirichlet_values=affine,
    )


def test_design_condition_kernel_reuses_capacity_bucket(
    affine_problem: phx.equations.CompiledFiniteElementProblem,
) -> None:
    basis = jax.device_put(np.eye(1, dtype=np.float64))
    record = {}
    ledger = StageLedger(record)
    first = ledger.run(
        "first",
        lambda: _design_qualification_condition_matrix(
            affine_problem, basis
        ).block_until_ready(),
    )
    second = ledger.run(
        "cached",
        lambda: _design_qualification_condition_matrix(
            affine_problem, basis
        ).block_until_ready(),
    )
    system, _ = affine_problem.linear_system()
    expected = jax.vmap(system.operator.mv)(basis).T

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, first)
    assert record["stage_measurements"]["first"]["eager_xla_compile_count"] == 1
    assert record["stage_measurements"]["cached"]["eager_xla_compile_count"] == 0


@final
class _DesignProbeEquation(phx.StrictModule):
    coefficients: jax.Array

    def __call__(self, radius: jax.Array) -> jax.Array:
        return self.coefficients * radius**2


@pytest.mark.strict_jax
def test_design_derivative_kernel_reuses_numerical_signature() -> None:
    # LF16's fixed-route probe has one scalar parameter and a capacity-three
    # output here. Changed coefficients are request data, not a static selector.
    parameter = jax.device_put(np.asarray(0.75, dtype=np.float64))
    direction = jax.device_put(np.asarray(0.7, dtype=np.float64))
    step = 1.0e-5
    coefficients = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
    changed_coefficients = np.asarray((4.0, -2.0, 0.5), dtype=np.float64)
    equation = _DesignProbeEquation(jax.device_put(coefficients))
    changed_equation = _DesignProbeEquation(jax.device_put(changed_coefficients))
    linearizations = (
        _design_qualification_linearizations(equation, equation, parameter)[0],
    )
    changed_linearizations = (
        _design_qualification_linearizations(
            changed_equation, changed_equation, parameter
        )[0],
    )
    lower = jax.device_put(coefficients * (0.75 - step * 0.7) ** 2)
    upper = jax.device_put(coefficients * (0.75 + step * 0.7) ** 2)
    changed_lower = jax.device_put(changed_coefficients * (0.75 - step * 0.7) ** 2)
    changed_upper = jax.device_put(changed_coefficients * (0.75 + step * 0.7) ** 2)
    record = {}
    ledger = StageLedger(record)
    with CompileCounter(stage="design.derivative-probe.capacity-3") as counter:
        first = ledger.run(
            "design.derivative.cold.capacity-3",
            lambda: _design_qualification_derivative_probe(
                linearizations, direction, lower, upper, step
            ),
        )
        repeated = ledger.run(
            "design.derivative.warm.capacity-3",
            lambda: _design_qualification_derivative_probe(
                changed_linearizations,
                direction,
                changed_lower,
                changed_upper,
                step,
            ),
        )
    assert first["finite_difference_error"] < _DESIGN_DIFFERENCE_RESIDUAL
    assert repeated["finite_difference_error"] < _DESIGN_DIFFERENCE_RESIDUAL
    assert first["JVP_VJP_duality_error"] < _DESIGN_DUALITY_RESIDUAL
    assert repeated["JVP_VJP_duality_error"] < _DESIGN_DUALITY_RESIDUAL
    measurements = record["stage_measurements"]
    assert (
        measurements["design.derivative.cold.capacity-3"]["eager_xla_compile_count"] == 1
    )
    assert (
        measurements["design.derivative.warm.capacity-3"]["eager_xla_compile_count"] == 0
    )
    assert counter.count == 1
    assert counter.events[0].signature is not None
    assert counter.events[0].stage == "design.derivative-probe.capacity-3"


def test_compile_counter_sees_two_compiles_inside_stage_ledger() -> None:
    record = {}
    ledger = StageLedger(record)

    @jax.jit
    def known_kernel(value: jax.Array) -> jax.Array:
        return value * 2

    inputs = (
        jnp.asarray([1.0], dtype=jnp.float64),
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
    )

    def compile_twice() -> None:
        for value in inputs:
            known_kernel(value).block_until_ready()

    parent = logging.getLogger("jax")
    sink = logging.NullHandler()
    parent.addHandler(sink)
    try:
        with CompileCounter(
            stage=lambda: record["active_phase"], capture_stack=True
        ) as counter:
            ledger.run("known_twice", compile_twice)
    finally:
        parent.removeHandler(sink)
    assert counter.count == 2
    assert (
        ledger.record["stage_measurements"]["known_twice"]["eager_xla_compile_count"] == 2
    )
    assert counter.seconds >= 0.0
    assert all("known_kernel" in event.kernel_name for event in counter.events)
    assert all(event.signature is not None for event in counter.events)
    assert all(event.stage == "known_twice" for event in counter.events)
    assert all(event.first_party_stack for event in counter.events)


def test_native_deadline_policy_preserves_parallelism_and_parent_deadline() -> None:
    from phydrax._meshcore import NativeParallelExecution

    default = NativeParallelExecution()
    with default:
        assert default.parallelism == 1
    outer = NativeParallelExecution(8, 10.0)
    with outer:
        inherited = NativeParallelExecution(None, 20.0)
        before = outer.remaining_seconds()
        with inherited:
            assert inherited.parallelism == 8
            assert 0.0 < inherited.remaining_seconds() <= before
            explicit = NativeParallelExecution(3, 30.0)
            before_inner = inherited.remaining_seconds()
            with explicit:
                assert explicit.parallelism == 3
                assert 0.0 < explicit.remaining_seconds() <= before_inner


def test_stage_ledger_lowers_original_deadline_into_host_coordinate_work() -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeExecutionScope,
    )
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from tools.meshing_qualification._records import FailureRecord

    scope = NativeExecutionScope(NativeExecutionLimits(1024, 0, 0, 4096, 60.0))
    record: dict[str, Any] = {}
    ledger = StageLedger(record, deadline=Deadline(perf_counter() + 0.5))
    ledger.run("completed_prefix", lambda: sleep(0.01))

    def host_loop() -> None:
        with scope:
            with scope.batch(1024, bytes_upper=4096) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=1024, maximum_memory_bytes=4096
                )
                sleep(0.51)
                budget.reserve(1)

    with pytest.raises(MeshcoreError) as caught:
        ledger.run("long_host_work", host_loop)
    error = caught.value
    assert error.status is MeshcoreStatus.TIMEOUT
    assert error.maximum_seconds is not None and 0.0 < error.maximum_seconds < 0.5
    assert error.elapsed_seconds is not None
    assert error.elapsed_seconds >= error.maximum_seconds
    failure = FailureRecord.from_exception(error, "long_host_work").to_record()
    assert failure["native_status"] == "TIMEOUT"
    assert failure["maximum_seconds"] == error.maximum_seconds
    assert failure["elapsed_seconds"] == error.elapsed_seconds
    assert record["active_phase"] == "long_host_work"
    assert record["stage_measurements"]["long_host_work"]["invocations"] == 1


def test_failure_record_omits_unobserved_native_time() -> None:
    from phydrax._meshcore import MeshcoreError, MeshcoreStatus
    from tools.meshing_qualification._records import FailureRecord

    ordinary = FailureRecord.from_exception(
        ValueError("unavailable"), "stage"
    ).to_record()
    assert ordinary == {
        "failure_phase": "stage",
        "failure_type": "ValueError",
        "failure": "unavailable",
    }
    for status in MeshcoreStatus:
        native = FailureRecord.from_exception(
            MeshcoreError(
                status, "fixture.native must report the refusal; got unavailable"
            ),
            "stage",
        ).to_record()
        assert native["native_status"] == status.name
        assert "elapsed_seconds" not in native
        assert "maximum_seconds" not in native


@pytest.mark.parametrize("expired", (False, True), ids=("ample", "expired"))
def test_cad_workflow_uses_original_shared_deadline(expired: bool) -> None:
    import json

    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.cad import run

    controls = QualificationControls(timeout=1e-9 if expired else 120.0)
    record: dict[str, Any] = run(controls)
    assert record["targets"]["wall_seconds"] == controls.timeout
    json.dumps(record)
    if expired:
        assert record["status"] == "timeout"
        assert record["failure_type"] == "TimeoutError"
        assert "qualification.deadline" in record["failure"]
    else:
        assert record["status"] == "passed", record
        assert record["target_error_reached"]
        assert record["physical_error"] <= controls.target_error


@pytest.mark.meshcore
def test_curved_cad_workflow_bounds_later_native_request_by_original_expiry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import cad

    controls = QualificationControls(cad_case="external-routine", timeout=10.0)
    observed = []

    def refuse_before_science(*args: Any, **kwargs: Any) -> None:
        # A later request keeps its original long request limit, but must
        # inherit the orchestrator's shorter unchanged absolute workflow clock.
        with NativeExecutionScope(NativeExecutionLimits(1, 1, 1, 65536, 120.0)) as scope:
            remaining = scope.remaining()
            assert 0.0 < remaining.remaining_seconds <= controls.timeout
            observed.append(remaining.remaining_seconds)
        raise ValueError(
            "intentional pre-science refusal after observing the actual clock"
        )

    monkeypatch.setattr(cad, "_cad_case_resolution", refuse_before_science)
    record = cad.run(controls)
    assert observed
    assert record["status"] == "failed"
    failure = record["failure"]
    assert isinstance(failure, str)
    assert "intentional pre-science refusal" in failure
    assert record["case_contract_completion"] == "incomplete"


def test_stage_ledger_records_real_compile_then_cache_reuse() -> None:
    record = {}
    ledger = StageLedger(record)

    @jax.jit
    def increment(value: jax.Array) -> jax.Array:
        return value + 1

    value = jnp.asarray([3.0], dtype=jnp.float64)
    first = ledger.run("first", lambda: increment(value).block_until_ready())
    second = ledger.run("cached", lambda: increment(value).block_until_ready())
    assert jnp.array_equal(first, second)
    assert record["stage_measurements"]["first"]["eager_xla_compile_count"] == 1
    assert record["stage_measurements"]["cached"]["eager_xla_compile_count"] == 0


def test_stage_ledger_accumulates_compile_evidence_and_failed_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS", "1")
    record = {}
    ledger = StageLedger(record)

    def compile_stage() -> int:
        logging.getLogger("jax._src.dispatch").warning(
            "Finished XLA compilation jit(example) in 0.125 sec"
        )
        return 7

    assert ledger.run("prepare", compile_stage) == 7
    assert ledger.run("prepare", compile_stage) == 7
    measurement = record["stage_measurements"]["prepare"]
    assert measurement["invocations"] == 2
    assert measurement["eager_xla_compile_count"] == 2
    assert measurement["eager_xla_compile_seconds"] == 0.25
    assert measurement["peak_resident_unit"] == "bytes"
    assert measurement["peak_resident_bytes"] > 0
    assert measurement["python_resource_accounting_seconds"] >= 0

    def failure() -> None:
        raise ValueError(
            "test.stage must preserve scientific refusal; got invalid evidence"
        )

    previous = sys.getprofile()
    with pytest.raises(ValueError, match="scientific refusal"):
        ledger.run("refusal", failure)
    assert sys.getprofile() is previous
    assert record["active_phase"] == "refusal"
    assert record["stage_measurements"]["refusal"]["invocations"] == 1


@pytest.mark.skipif(
    sys.platform != "linux", reason="post-exec high water is published only on Linux"
)
def test_child_peak_resident_bytes_excludes_launcher_high_water() -> None:
    # Raise this launcher's high water well above anything the child needs.
    floor = b"\x01" * (512 << 20)
    launcher_peak = peak_resident_bytes()
    del floor
    assert launcher_peak is not None
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tools.meshing_qualification._runtime import peak_resident_bytes; "
            "print(peak_resident_bytes())",
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    )
    child_peak = int(child.stdout)
    assert 0 < child_peak < launcher_peak - (256 << 20)


def test_campaign_timeout_kills_descendants_without_cli_receipt(tmp_path: Path) -> None:
    heartbeat = tmp_path / "heartbeat"
    descendant = (
        "from pathlib import Path; from time import sleep; "
        f"p=Path({str(heartbeat)!r}); "
        "exec('while True:\\n p.write_text(str(__import__(\"time\").monotonic_ns()))\\n sleep(0.01)')"
    )
    parent = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable,'-c',{descendant!r}]); "
        "time.sleep(60)"
    )
    started = perf_counter()
    with pytest.raises(CampaignProcessTimeout) as captured:
        run_campaign_process([sys.executable, "-c", parent], cwd=tmp_path, timeout=1.0)
    assert perf_counter() - started < 1.0 + REPORTING_GRACE_SECONDS + 5.0
    if sys.platform == "linux":
        assert captured.value.peak_resident_bytes is not None
    assert heartbeat.exists()
    stopped = heartbeat.read_text()
    sleep(0.1)
    assert heartbeat.read_text() == stopped


def test_launched_child_reports_timeout_phase_from_launch_origin() -> None:
    # The child's own deadline is an hour from its scenario start; only the
    # inherited launch origin can stop it, and it must still report its phase.
    child = (
        "import json, time\n"
        "from time import perf_counter\n"
        "from phydrax._meshcore import MeshcoreError\n"
        "from tools.meshing_qualification._records import record_failure\n"
        "from tools.meshing_qualification._runtime import Deadline, StageLedger\n"
        "record = {'status': 'failed'}\n"
        "ledger = StageLedger(record, deadline=Deadline(perf_counter() + 3600.0))\n"
        "try:\n"
        "    while True:\n"
        "        ledger.run('bounded_phase', lambda: time.sleep(0.05))\n"
        "except (TimeoutError, MeshcoreError) as failure:\n"
        "    record_failure(record, failure, record['active_phase'])\n"
        "print(json.dumps(record))\n"
    )
    timeout = 20.0
    started = perf_counter()
    completed = run_campaign_process(
        [sys.executable, "-c", child],
        cwd=Path(__file__).resolve().parents[3],
        timeout=timeout,
    )
    elapsed = perf_counter() - started
    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout)
    assert record["status"] == "timeout"
    assert record["failure_phase"] == "bounded_phase"
    assert record["active_phase"] == "bounded_phase"
    assert record["stage_measurements"]["bounded_phase"]["invocations"] > 0
    assert timeout <= elapsed < timeout + REPORTING_GRACE_SECONDS


def test_campaign_success_preserves_outputs(tmp_path: Path) -> None:
    result = run_campaign_process(
        [
            sys.executable,
            "-c",
            "import sys; print('record'); print('diagnostic',file=sys.stderr)",
        ],
        cwd=tmp_path,
        timeout=5.0,
    )
    assert result.returncode == 0
    assert result.stdout == "record\n"
    assert result.stderr == "diagnostic\n"
    assert result.deadline_expired is False


def test_environment_scrubbing_child_cannot_turn_grace_into_science(
    tmp_path: Path,
) -> None:
    # ``env -i`` drops the inherited launch deadline, so the child cannot stop
    # itself; it exits inside the reporting grace and must be reported late.
    result = run_campaign_process(
        ["env", "-i", sys.executable, "-c", "import time; time.sleep(2.5); print(1)"],
        cwd=tmp_path,
        timeout=1.0,
    )
    assert result.returncode == 0
    assert result.stdout == "1\n"
    assert result.deadline_expired is True


_COLD_PROGRAMS = "tests._support.cold_worker_programs"


def _process_alive(pid: int) -> bool:
    """Whether ``pid`` is a running (not exited or zombie) process."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] not in ("Z", "X")


def test_cold_worker_executes_one_bound_packet_after_publication(
    tmp_path: Path,
) -> None:
    record: dict[str, Any] = {}
    deadline = Deadline(perf_counter() + 120.0)
    archive = tmp_path / "archive.bin"
    with prepare_cold_worker(
        _COLD_PROGRAMS,
        deadline=deadline,
        ledger=StageLedger(record, deadline=deadline, cold_phase="cold"),
    ) as worker:
        # The worker process exists before the archive is durably written.
        assert _process_alive(worker.process_id)
        assert not archive.exists()
        archive.write_text("published", encoding="utf-8")
        result = worker.execute(archive, "content", {"mode": "echo", "repeats": 2})
    process = record["cold_process"]
    assert result == {
        "archive": "published",
        "content_id": "content",
        "controls": {"mode": "echo", "repeats": 2},
        "remaining_seconds": result["remaining_seconds"],
        "process_id": worker.process_id,
    }
    assert worker.process_id != os.getpid()
    assert process["status"] == "completed"
    assert process["module"] == _COLD_PROGRAMS
    assert process["process_id"] == process["session_id"] == worker.process_id
    assert process["parent_process_id"] == os.getpid()
    assert process["content_id"] == "content"
    assert (
        process["launched_monotonic"]
        <= process["bootstrap_started_monotonic"]
        <= process["import_completed_monotonic"]
        <= process["packet_received_monotonic"]
        <= process["execute_completed_monotonic"]
        <= process["result_returned_monotonic"]
    )
    assert process["packet_sent_monotonic"] <= process["packet_received_monotonic"]
    for name in (
        "import_seconds",
        "packet_wait_seconds",
        "read_seconds",
        "solve_seconds",
    ):
        assert process[name] >= 0.0
    expiry = time.clock_gettime(time.CLOCK_MONOTONIC) + (
        deadline.expires_at - perf_counter()
    )
    assert process["expires_monotonic"] == pytest.approx(expiry, abs=1.0e-3)
    # The packet does not reset the clock: the child measures the launch expiry.
    assert result["remaining_seconds"] <= 120.0 - (
        process["packet_received_monotonic"] - process["launched_monotonic"]
    )
    assert record["stage_measurements"]["cold"]["invocations"] == 1
    assert not _process_alive(worker.process_id)


def test_cold_worker_archive_failure_kills_prepared_session() -> None:
    record: dict[str, Any] = {}
    with pytest.raises(OSError, match="archive write failed"):
        with prepare_cold_worker(
            _COLD_PROGRAMS,
            deadline=Deadline(perf_counter() + 120.0),
            ledger=StageLedger(record),
        ) as worker:
            raise OSError("archive write failed")
    assert not _process_alive(worker.process_id)
    assert record["cold_process"]["status"] == "cancelled"
    assert record["cold_process"]["returncode"] == -signal.SIGKILL


def test_cold_worker_timeout_keeps_launch_expiry(tmp_path: Path) -> None:
    archive = tmp_path / "archive.bin"
    archive.write_text("published", encoding="utf-8")
    record: dict[str, Any] = {}
    started = perf_counter()
    with prepare_cold_worker(
        _COLD_PROGRAMS,
        deadline=Deadline(started + 15.0),
        ledger=StageLedger(record, cold_phase="cold"),
    ) as worker:
        sleep(5.0)
        with pytest.raises(ColdWorkerTimeout):
            worker.execute(archive, "content", {"mode": "sleep"})
    # Killed at the expiry measured from preparation, not 15 s after the packet.
    assert 15.0 <= perf_counter() - started < 17.0
    assert record["cold_process"]["status"] == "timeout"
    assert record["cold_process"]["returncode"] == -signal.SIGKILL
    assert not _process_alive(worker.process_id)


@pytest.mark.parametrize(
    ("module", "mode", "message"),
    [
        (_COLD_PROGRAMS, "fail", "must refuse; got fail"),
        (_COLD_PROGRAMS, "process_id", "must leave process_id to the runtime"),
        ("tests._support.no_such_cold_program", "echo", "ModuleNotFoundError"),
    ],
)
def test_cold_worker_refusal_retains_child_evidence(
    tmp_path: Path, module: str, mode: str, message: str
) -> None:
    archive = tmp_path / "archive.bin"
    archive.write_text("published", encoding="utf-8")
    record: dict[str, Any] = {}
    with prepare_cold_worker(
        module, deadline=Deadline(perf_counter() + 120.0), ledger=StageLedger(record)
    ) as worker:
        with pytest.raises(QualificationFailure, match="returncode must equal zero"):
            worker.execute(archive, "content", {"mode": mode})
    assert message in record["cold_process_refusal"]["stderr"]
    assert record["cold_process_refusal"]["stdout"] == ""
    assert record["cold_process"]["status"] == "refused"
    assert record["cold_process"]["process_id"] == worker.process_id


@pytest.mark.skipif(sys.platform != "linux", reason="parent-death binding is Linux")
def test_cold_worker_dies_with_killed_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = tmp_path / "imported"
    monkeypatch.setenv(IMPORTED_MARKER_VARIABLE, str(marker))
    launcher = (
        "import sys, time\n"
        "from time import perf_counter\n"
        "from tools.meshing_qualification._runtime import (\n"
        "    Deadline, prepare_cold_worker, StageLedger,\n"
        ")\n"
        "prepare_cold_worker(\n"
        "    sys.argv[1], deadline=Deadline(perf_counter() + 300.0), ledger=StageLedger({})\n"
        ")\n"
        "time.sleep(300.0)\n"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", launcher, _COLD_PROGRAMS],
        cwd=Path(__file__).resolve().parents[3],
    )
    try:
        waited = perf_counter() + 90.0
        while not marker.exists() and perf_counter() < waited:
            sleep(0.05)
        # The worker imported its program and now waits for a packet that the
        # SIGKILLed launcher can never send or cancel.
        pid = int(marker.read_text(encoding="utf-8"))
        assert _process_alive(pid)
    finally:
        process.kill()
        process.wait()
    waited = perf_counter() + 30.0
    while _process_alive(pid) and perf_counter() < waited:
        sleep(0.05)
    assert not _process_alive(pid)


def test_design_memory_gate_without_platform_peak_is_unassessed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import design

    # Platforms without a post-exec high-water source publish no peak.
    monkeypatch.setattr(design, "peak_resident_bytes", lambda: None)
    record: dict[str, Any] = {
        "status": "passed",
        "failed_positive_attempts": [],
        "active_phase": "continued_solve",
    }
    design._design_qualification_completion(
        QualificationControls(), [], record, perf_counter()
    )
    assert record["peak_process_resident_bytes"] is None
    assert record["memory_budget_status"] == "unassessed"
    assert record["status"] == "failed"
    assert record["mandatory_workflow_completion"] == "incomplete"
    assert record["failed_positive_attempts"][-1]["failure_phase"] == (
        "process_peak_memory_budget"
    )
    json.dumps(record)


def test_shared_wall_schedule_preserves_authored_identity() -> None:
    from tests._support.boundary_layer_walls import SCHEDULE
    from tools._meshing_cases import original_wall_layer_schedule

    original = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2)
    shared = original_wall_layer_schedule()
    assert original.schedule_id == shared.schedule_id == SCHEDULE.schedule_id
    assert np.array_equal(original.thicknesses, shared.thicknesses)
    assert np.array_equal(original.growth_rates, shared.growth_rates)


def test_clean_stage_and_legacy_benchmark_do_not_install_accounting_profiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_benchmarks import _measured_benchmark_case

    monkeypatch.delenv("PHYDRAX_MESHING_ACCOUNTING_DIAGNOSTICS", raising=False)
    previous = sys.getprofile()
    record = {}

    def clean_stage() -> None:
        assert sys.getprofile() is previous

    StageLedger(record).run("clean", clean_stage)
    clean = record["stage_measurements"]["clean"]
    assert clean["python_resource_accounting_seconds"] is None
    assert clean["accounting_diagnostics"] is False

    @jax.jit
    def known_kernel(value: jax.Array) -> jax.Array:
        return value + 3

    value = jnp.asarray([1.0], dtype=jnp.float64)

    def benchmark() -> dict[str, object]:
        assert sys.getprofile() is previous
        known_kernel(value).block_until_ready()
        return {
            "scientific_id": "retained",
            "stage_measurements": {"owner_diagnostic": {"retained": True}},
        }

    measured = _measured_benchmark_case("known", benchmark)
    assert measured["scientific_id"] == "retained"
    import_seconds = measured["import_seconds"]
    assert isinstance(import_seconds, float)
    assert import_seconds >= 0.0
    assert measured["peak_resident_unit"] == "bytes"
    measurements = measured["stage_measurements"]
    assert isinstance(measurements, dict)
    assert measurements["owner_diagnostic"] == {"retained": True}
    sample = measurements["benchmark_case.known"]
    assert sample["eager_xla_compile_count"] == 1
    assert sample["eager_xla_compile_seconds"] >= 0.0
    assert sample["python_resource_accounting_seconds"] is None
    assert (
        sample["python_resource_accounting_measurement"]
        == "unmeasured-clean-timing-no-accounting-profiler"
    )
    assert sample["peak_resident_bytes"] > 0


def test_distributed_launcher_preserves_canonical_failure_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    from collections.abc import Mapping, Sequence
    from time import perf_counter

    from phydrax.execution import LocalProcessLaunchPlan, LocalProcessResult
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import FailureRecord
    from tools.meshing_qualification._runtime import QualificationFailure
    from tools.meshing_qualification.scenarios import distributed

    failure = FailureRecord.from_exception(
        ValueError("Bisection.compatibility must admit refinement; got REJECT"),
        "owner_local_preparation_0",
    ).to_record()

    def launch(
        argv: Sequence[str],
        plan: LocalProcessLaunchPlan,
        *,
        environment: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[LocalProcessResult, ...]:
        del argv, plan, cwd
        assert environment is not None
        directory = Path(environment["PHYDRAX_MESHING_RECORD_DIRECTORY"])
        (directory / "process-0.json").write_text(
            json.dumps({"status": "failed", **failure}), encoding="utf-8"
        )
        return (LocalProcessResult(0, 1, b"", b""),)

    monkeypatch.setattr(distributed, "launch_local_processes", launch)
    args = QualificationControls(distributed_checkpoint_root=str(tmp_path / "checkpoint"))
    record = {}
    with pytest.raises(
        QualificationFailure,
        match="owner_local_preparation_0: ValueError: Bisection.compatibility",
    ):
        distributed._distributed_qualification_launch(
            args, record, perf_counter(), "producer"
        )
    assert record["producer_processes"][0]["failure_phase"] == "owner_local_preparation_0"
    assert record["producer_processes"][0]["failure_type"] == "ValueError"


def test_distributed_launcher_rejects_failed_rank_after_passed_record(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import json
    from collections.abc import Mapping, Sequence
    from time import perf_counter

    from phydrax.execution import LocalProcessLaunchPlan, LocalProcessResult
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._runtime import QualificationFailure
    from tools.meshing_qualification.scenarios import distributed

    def launch(
        argv: Sequence[str],
        plan: LocalProcessLaunchPlan,
        *,
        environment: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[LocalProcessResult, ...]:
        del argv, plan, cwd
        assert environment is not None
        directory = Path(environment["PHYDRAX_MESHING_RECORD_DIRECTORY"])
        (directory / "process-0.json").write_text(
            json.dumps({"status": "passed"}), encoding="utf-8"
        )
        return (LocalProcessResult(0, -6, b"", b"late collective abort"),)

    monkeypatch.setattr(distributed, "launch_local_processes", launch)
    args = QualificationControls(distributed_checkpoint_root=str(tmp_path / "checkpoint"))
    record = {}
    with pytest.raises(
        QualificationFailure, match="process 0 exited -6.*late collective abort"
    ):
        distributed._distributed_qualification_launch(
            args, record, perf_counter(), "producer"
        )
    assert record["producer_processes"] == [{"status": "passed"}]
    assert record["producer_process_results"][0]["returncode"] == -6
    assert record["producer_process_results"][0]["stderr"] == "late collective abort"


@pytest.mark.parametrize("record_mode", ("complete", "missing", "malformed"))
@pytest.mark.parametrize("late", (False, True))
def test_distributed_launcher_retains_real_rank_failures_before_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, record_mode: str, late: bool
) -> None:
    from collections.abc import Mapping, Sequence

    from phydrax.execution import (
        launch_local_processes,
        LocalProcessLaunchPlan,
        LocalProcessResult,
    )
    from tools.meshing_qualification import _runtime
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import record_failure
    from tools.meshing_qualification.scenarios import distributed

    # These subprocesses exercise launch/report transport only. Each actually
    # raises a rank-specific error and exits nonzero; none claims to run science.
    child = (
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "rank = int(os.environ['PHYDRAX_PROCESS_ID'])\n"
        "try:\n"
        "    raise ValueError(f'launcher transport failure on rank {rank}')\n"
        "except ValueError as failure:\n"
        "    record = {'status': 'failed', 'failure_phase': 'launcher_transport_probe',\n"
        "              'failure_type': type(failure).__name__, 'failure': str(failure),\n"
        "              'process_id': rank, 'pid': os.getpid(), 'science_executed': False}\n"
        "path = Path(os.environ['PHYDRAX_MESHING_RECORD_DIRECTORY']) / f'process-{rank}.json'\n"
        f"mode = {record_mode!r}\n"
        "if rank != 0 or mode != 'missing':\n"
        "    path.write_text('{incomplete' if rank == 0 and mode == 'malformed'\n"
        "                    else json.dumps(record), encoding='utf-8')\n"
        "print(f'late exit from rank {rank}', file=sys.stderr)\n"
        "sys.exit(rank + 1)\n"
    )
    args = QualificationControls(distributed_checkpoint_root=str(tmp_path / "checkpoint"))
    assert args.timeout == 120.0
    clock = [perf_counter()]

    def launch(
        argv: Sequence[str],
        plan: LocalProcessLaunchPlan,
        *,
        environment: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[LocalProcessResult, ...]:
        del argv
        try:
            return launch_local_processes(
                [sys.executable, "-c", child], plan, environment=environment, cwd=cwd
            )
        finally:
            if late:
                clock[0] += args.timeout + 1.0

    monkeypatch.setattr(distributed, "launch_local_processes", launch)
    monkeypatch.setattr(_runtime, "perf_counter", lambda: clock[0])
    record: dict[str, Any] = {}
    with pytest.raises(TimeoutError if late else QualificationFailure) as caught:
        distributed._distributed_qualification_launch(args, record, clock[0], "producer")
    record_failure(record, caught.value, record["active_phase"])
    assert record["status"] == ("timeout" if late else "failed")
    assert record["failure_phase"] == "producer_processes"
    assert record["failure_type"] == ("TimeoutError" if late else "QualificationFailure")
    directory = args.distributed_repository_root() / "producer-records"
    ranks = range(args.distributed_processes)
    expected = [
        json.loads((directory / f"process-{rank}.json").read_text(encoding="utf-8"))
        for rank in ranks
        if rank != 0 or record_mode == "complete"
    ]
    assert record["producer_processes"] == expected
    assert len({item["pid"] for item in expected}) == len(expected)
    assert all(item["science_executed"] is False for item in expected)
    exits = record["producer_process_results"]
    assert [item["process_id"] for item in exits] == list(ranks)
    assert [item["returncode"] for item in exits] == [rank + 1 for rank in ranks]
    assert [item["stderr"] for item in exits] == [
        f"late exit from rank {rank}\n" for rank in ranks
    ]
    assert [item["process_id"] for item in record["producer_process_record_errors"]] == (
        [] if record_mode == "complete" else [0]
    )
    json.dumps(record)


def _distributed_stage_boundary_rank_probe() -> dict[str, Any]:
    """Real two-rank boundary transport with explicit local fault injection."""
    from jax.experimental import multihost_utils

    from tools.meshing_qualification import _runtime
    from tools.meshing_qualification._records import record_failure
    from tools.meshing_qualification.scenarios import distributed

    rank = jax.process_index()
    records = {}
    modes = (
        "pre-deadline",
        "pre-resource",
        "native-entry-deadline",
        "native-entry-resource",
        "native-exit-deadline",
        "post-deadline",
        "measurement-resource",
    )
    for mode in modes:
        record: dict[str, Any] = {
            "status": "failed",
            "entered_callback": False,
            "entered_next_stage": False,
        }
        deadline = Deadline(
            perf_counter() + (-1.0 if rank == 1 and mode == "pre-deadline" else 120.0)
        )
        remaining = Deadline.remaining
        parallel = _runtime.NativeParallelExecution
        calls = 0

        def remaining_with_fault(value: Deadline) -> float:
            nonlocal calls
            calls += 1
            fault_call = {"native-entry-deadline": 2, "post-deadline": 3}.get(mode)
            if rank == 1 and calls == fault_call:
                raise TimeoutError(f"injected {mode} on rank {rank}")
            return remaining(value)

        def parallel_with_fault(parallelism: int | None, maximum_seconds: float) -> Any:
            if rank == 1 and mode == "native-entry-resource":
                raise MemoryError("injected native entry resource exhaustion on rank 1")
            return parallel(
                parallelism,
                0.0 if rank == 1 and mode == "native-exit-deadline" else maximum_seconds,
            )

        def resource_fault(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise MemoryError(f"injected {mode} on rank {rank}")

        def collective_callback() -> None:
            record["entered_callback"] = True
            multihost_utils.sync_global_devices(f"ledger-probe-{mode}-callback")

        with pytest.MonkeyPatch.context() as patch:
            patch.setenv(distributed._DISTRIBUTED_ROLE, "producer")
            patch.setattr(Deadline, "remaining", remaining_with_fault)
            patch.setattr(_runtime, "NativeParallelExecution", parallel_with_fault)
            if rank == 1 and mode == "pre-resource":
                patch.setattr(_runtime, "CompileCounter", resource_fault)
            if rank == 1 and mode == "measurement-resource":
                patch.setattr(_runtime, "peak_resident_bytes", resource_fault)
            try:
                distributed._distributed_stage_ledger(record, deadline=deadline).run(
                    "first_collective_stage", collective_callback
                )
                record["entered_next_stage"] = True
                distributed._distributed_stage_ledger(record, deadline=deadline).run(
                    "must_not_enter", collective_callback
                )
            except Exception as failure:
                record_failure(record, failure, record["active_phase"])
        # Completion of this barrier proves a local boundary fault did not leave
        # the other process inside the callback or the next stage's collective.
        multihost_utils.sync_global_devices(f"ledger-probe-{mode}-terminated")
        records[mode] = record
    return {"rank": rank, "records": records}


def test_distributed_stage_boundaries_terminate_all_real_ranks() -> None:
    import socket
    from concurrent.futures import ThreadPoolExecutor

    from tests._support.forced_devices import gloo_rank_environment, run_on_forced_devices

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        coordinator = f"127.0.0.1:{listener.getsockname()[1]}"

    def launch(rank: int) -> dict[str, Any]:
        return run_on_forced_devices(
            __name__,
            "_distributed_stage_boundary_rank_probe",
            (),
            devices=1,
            env=gloo_rank_environment(coordinator, rank),
            timeout_seconds=120.0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        owners = tuple(executor.map(launch, range(2)))
    assert [owner["rank"] for owner in owners] == [0, 1]
    for mode, first in owners[0]["records"].items():
        second = owners[1]["records"][mode]
        is_timeout = "deadline" in mode
        assert (
            first["status"] == second["status"] == ("timeout" if is_timeout else "failed")
        )
        assert not first["entered_next_stage"] and not second["entered_next_stage"]
        entered = mode in (
            "native-exit-deadline",
            "post-deadline",
            "measurement-resource",
        )
        assert first["entered_callback"] is second["entered_callback"] is entered
        assert second["stage_boundary_failures"]
        if mode == "native-exit-deadline":
            assert second["failure_type"] == "MeshcoreError"
            assert second["native_status"] == "TIMEOUT"
        elif is_timeout:
            assert second["failure_type"] == "TimeoutError"
        else:
            assert second["failure_type"] == "MemoryError"
        json.dumps(first)
        json.dumps(second)


def test_stage_ledger_does_not_admit_a_mixed_callback_exception() -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    entered = []

    @contextmanager
    def boundary(name: str) -> Iterator[None]:
        entered.append(name)
        yield

    def callback() -> None:
        raise ValueError("The mixed callback owns its own failure protocol.")

    record: dict[str, Any] = {}
    with pytest.raises(ValueError, match="owns its own failure protocol"):
        StageLedger(record, local_boundary=boundary).run("mixed", callback)
    assert entered == ["mixed.admission", "mixed.native-entry"]
    assert record["stage_measurements"]["mixed"]["invocations"] == 1


def test_concave_corpus_material_controls_preserve_single_fluid_source() -> None:
    from time import perf_counter

    from phydrax.meshing.qualification import layer_core_material_declaration
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import _corpus_profiles
    from tools.meshing_qualification.scenarios.hybrid import _hybrid_concave_publication

    profile = _corpus_profiles()["native-hybrid-lifecycle:concave-box-core-lifecycle"]
    defaults = QualificationControls()
    controls = QualificationControls.from_record(
        {**defaults.to_record(), **profile["controls"]}
    )
    assert defaults.hybrid_region_densities is None
    assert controls.hybrid_region_densities == (2.0,)
    assert controls.hybrid_material_site == "fluid-density"
    record = {
        "started": perf_counter(),
        "targets": {"wall_seconds": controls.timeout},
    }
    publication = _hybrid_concave_publication(controls, record)
    assert publication.source.region_ids == ("fluid",)
    np.testing.assert_array_equal(publication.source.core_region_map, (0,))
    np.testing.assert_array_equal(
        publication.source.layer_regions,
        np.zeros(publication.source.layer_regions.shape, dtype=np.int64),
    )
    material = record["material_declaration"]
    assert isinstance(material, dict)
    assert material["region_ids"] == ("fluid",)
    assert material["densities"] == (2.0,)
    assert material["site_id"] == "fluid-density"
    # Splitting geometric cell families does not author a second material.
    with pytest.raises(ValueError, match="material.densities"):
        layer_core_material_declaration(
            publication.generated,
            densities=(2.0, 3.0),
            site_id="two-material-density",
        )


def test_hybrid_high_order_publication_retains_native_source_request(
    tmp_path: Path,
) -> None:
    from time import perf_counter

    from phydrax.meshing.qualification import (
        prepare_hybrid_adaptation_policy,
        prepare_hybrid_high_order_map,
        qualify_hybrid_high_order_publication,
    )
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import _corpus_profiles
    from tools.meshing_qualification.scenarios.hybrid import _hybrid_concave_publication

    profile = _corpus_profiles()["native-hybrid-lifecycle:concave-box-core-lifecycle"]
    controls = QualificationControls.from_record(
        {**QualificationControls().to_record(), **profile["controls"]}
    )
    publication = _hybrid_concave_publication(
        controls,
        {"started": perf_counter(), "targets": {"wall_seconds": controls.timeout}},
    )
    generated = publication.generated
    prepared = prepare_hybrid_high_order_map(
        generated,
        degree=2,
        maximum_nodes=32 * controls.capacity,
        maximum_entries=1024 * controls.capacity,
    )
    assert prepared.execution_evidence is not None
    assert prepared.execution_evidence.preparation_evidence is None
    assert prepared.execution_evidence.history_evidence == (generated.execution_evidence,)
    assert prepared.lifting_evidence.work_units > 0
    assert (
        prepared.lifting_evidence.native_charged_work_units
        == prepared.lifting_evidence.work_units
    )
    accepted = qualify_hybrid_high_order_publication(
        publication.source, publication.specification, generated, prepared
    )
    assert accepted.certification is not None and accepted.certification.passed
    assert accepted.compliance.report_id == generated.compliance.report_id
    assert (
        accepted.compliance.specification_id == publication.specification.specification_id
    )
    binding = accepted.trace.binding
    assert binding is not None
    assert binding.source_id == publication.source.source_id
    assert binding.source_revision == publication.source.source_revision
    assert binding.geometry_layout_id == accepted.geometry.geometry_layout_id
    assert accepted.geometry.geometry_layout_id != generated.geometry.geometry_layout_id
    assert accepted.execution_evidence is not None
    accepted.execution_evidence.require_valid()
    assert accepted.execution_evidence.preparation_evidence is prepared.execution_evidence
    policy = prepare_hybrid_adaptation_policy(
        publication.source,
        publication.specification,
        accepted,
        limits=publication.specification.limits,
    )
    assert policy.association_transfer is not None
    from fractions import Fraction

    from phydrax.geometry._embedding_state import coordinate_certificate_request
    from phydrax.meshing._association import PlcBankContext
    from phydrax.meshing._association_composition import ComposedAssociationTransfer

    transfer = policy.association_transfer
    assert isinstance(transfer, ComposedAssociationTransfer)
    with coordinate_certificate_request(accepted.certification.request.limits) as budget:
        contexts: list[PlcBankContext] = []
        classes = transfer.classes(accepted, budget=budget, _prepared_contexts=contexts)
        protected = transfer.protected_edges(
            accepted, _classes=classes, _prepared_contexts=contexts, budget=budget
        )
        witnessed = refused_foreign = False
        for namespace, (plan, context) in enumerate(
            zip(transfer.plans, contexts, strict=True)
        ):
            rows = np.flatnonzero(
                (classes[1].namespace_dimensions[namespace] >= 1)
                & (classes[1].namespace_dimensions[namespace] < 3)
                & ~protected
            )
            for row in rows.tolist():
                vertices = context.mapped.entity_vertices[1][row]
                midpoint = np.sum(
                    np.asarray(accepted.mesh.coordinates)[vertices] * np.float64(0.5),
                    axis=0,
                )
                index = int(classes[1].namespace_indices[namespace][row])
                dimension = int(classes[1].namespace_dimensions[namespace][row])
                if plan.point_support(
                    dimension,
                    index,
                    midpoint,
                    [plan.maximum_support_queries],
                    budget=budget,
                ):
                    continue
                exact = context.mapped.edge_midpoint(row)
                assert all(isinstance(value, Fraction) for value in exact)
                assert transfer._exact_split_support(
                    context, plan, row, dimension, index, budget=budget
                )
                witnessed = True
                for foreign in context.facet_indices[:8].tolist():
                    if not transfer._exact_split_support(
                        context, plan, row, 2, int(foreign), budget=budget
                    ):
                        refused_foreign = True
                        break
                break
            if witnessed:
                break
        assert witnessed, (
            "the unchanged nonbinary source must admit an exact split rejected by its rounded point query"
        )
        assert refused_foreign, "an unrelated source facet must remain protected"
        from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
        from phydrax._model._structure import (
            _canonical_recipe_payload,
            model_structure_recipe,
        )
        from phydrax.discretization import ExactRationalCellGeometrySource
        from phydrax.lifecycle import write_meshing_source_closure
        from tests._support.forced_devices import run_on_forced_devices

        exact_source = accepted.geometry.exact_source
        assert isinstance(exact_source, ExactRationalCellGeometrySource)
        bank = accepted.geometry.source_coordinates(budget=budget)
        coefficients = canonical_fingerprint(
            tuple(
                tuple(
                    (rational.numerator, rational.denominator)
                    for value in row
                    for rational in (
                        rational_coefficient(value, role="fixture coordinate"),
                    )
                )
                for row in bank
            )
        )
        representatives = canonical_fingerprint(
            array_tree_fingerprint(accepted.geometry.coordinates)
        )
        roots = (accepted, exact_source.parent)
        recipe_bytes = len(_canonical_recipe_payload(model_structure_recipe(roots)))
        assert recipe_bytes <= 1_048_576
        receipt = write_meshing_source_closure(tmp_path / "hybrid-exact-source", roots)
        cold = run_on_forced_devices(
            "tests._support.discretization_process_scenarios",
            "cold_exact_rational_publication",
            (
                str(receipt.path),
                receipt.content_id,
                accepted.result_id,
                exact_source.source_id,
                coefficients,
                representatives,
            ),
            devices=1,
            env={},
            timeout_seconds=180,
        )
        assert all(
            cold[key]
            for key in (
                "result_identity",
                "source_identity",
                "parent_sharing",
                "exact_coefficients",
                "representative_bytes",
                "certificate_retained",
            )
        )
        assert cold["retained_bytes"] > 0
        assert cold["csr_array_bytes"] > 0
        import json

        (tmp_path / "publication-evidence.json").write_text(
            json.dumps(
                {
                    **cold,
                    "recipe_bytes": recipe_bytes,
                    "recipe_byte_cap": 1_048_576,
                    "archive_path": str(receipt.path),
                    "archive_content_id": receipt.content_id,
                    "archive_bytes": receipt.path.stat().st_size,
                    "source_coefficient_digest": coefficients,
                    "representative_digest": representatives,
                    "mesh_vertex_count": accepted.mesh.coordinates.shape[0],
                    "mesh_cell_count": sum(
                        block.cell_count for block in accepted.mesh.blocks
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )


@pytest.fixture
def high_order_plc() -> tuple[
    phx.discretization.CellMesh,
    phx.discretization.CellGeometrySpec,
    phx.geometry.PiecewiseLinearDomain,
]:
    from phydrax.meshing._curving import straight_geometry
    from tests._support.mesh_certificate_fixtures import prepared_tetrahedral_acceptance

    mesh, _, _, prepared = prepared_tetrahedral_acceptance()
    domain = prepared.request.domain
    assert isinstance(domain, phx.geometry.PiecewiseLinearDomain)
    return mesh, straight_geometry(mesh, 2), domain


def test_mapped_plc_support_records_exact_high_order_affine_proof(
    high_order_plc: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellGeometrySpec,
        phx.geometry.PiecewiseLinearDomain,
    ],
) -> None:
    from phydrax.geometry._embedding_state import coordinate_certificate_request
    from phydrax.meshing._plc_mapped_support import prepare_mapped_plc_support

    mesh, geometry, domain = high_order_plc
    limits = phx.geometry.MeshCertificateLimits()
    with coordinate_certificate_request(limits) as budget:
        support = prepare_mapped_plc_support(
            mesh,
            geometry,
            domain,
            np.asarray((0,), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            maximum_support_queries=limits.maximum_work_units,
            certificate_limits=limits,
            budget=budget,
        )
        element = support.cells[0].element
        assert isinstance(element, phx.discretization.FiniteElementSpec)
        assert element.degree == 2
        assert support.degree_one_exactness_work_units > 0
        assert support.prepared.validity.all_certified
        assert support.embedding.status == support.coverage.status == "certified"
        for certificate in (support.embedding, support.coverage):
            assert certificate.source_expression_work_units is not None
            assert certificate.source_expression_work_units > 0
            assert certificate.binding.limits_id == limits.limits_id
        edges = support.entity_vertices[1]
        triangles = np.asarray(domain.facets, dtype=np.int64)
        faces = support.entity_vertices[2]
        assert triangles.ndim == faces.ndim == 2
        facet = next(
            index
            for index in range(triangles.shape[0])
            if np.array_equal(np.sort(triangles[index, :]), np.sort(faces[0, :]))
        )
        work = [limits.maximum_work_units]
        before = budget.work_units
        assert support.entity_support(
            1,
            0,
            1,
            0,
            edge_vertices=edges,
            triangle_vertices=triangles,
            triangle_facets=np.arange(triangles.shape[0], dtype=np.int64),
            work=work,
        )[0]
        assert support.entity_support(
            2,
            0,
            2,
            facet,
            edge_vertices=edges,
            triangle_vertices=triangles,
            triangle_facets=np.arange(triangles.shape[0], dtype=np.int64),
            work=work,
        )[0]
        assert budget.work_units > before
        assert work[0] < limits.maximum_work_units


def test_mapped_plc_support_refuses_curved_high_order_corner_interpolant(
    high_order_plc: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellGeometrySpec,
        phx.geometry.PiecewiseLinearDomain,
    ],
) -> None:
    from phydrax.geometry._embedding_state import coordinate_certificate_request
    from phydrax.meshing._plc_mapped_support import prepare_mapped_plc_support

    mesh, geometry, domain = high_order_plc
    coordinates = np.asarray(geometry.coordinates).copy()
    midpoint = np.flatnonzero(np.all(coordinates == np.asarray((0.5, 0.0, 0.0)), axis=1))
    assert midpoint.size == 1
    coordinates[midpoint[0], 2] = -0.125
    curved = geometry.with_coordinates(coordinates)
    limits = phx.geometry.MeshCertificateLimits()
    with (
        coordinate_certificate_request(limits) as budget,
        pytest.raises(ValueError, match="MappedPlcSupport.degree_one_exactness"),
    ):
        prepare_mapped_plc_support(
            mesh,
            curved,
            domain,
            np.asarray((0,), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            maximum_support_queries=limits.maximum_work_units,
            certificate_limits=limits,
            budget=budget,
        )


def test_mapped_plc_support_preserves_unsupported_family_refusal(
    high_order_plc: tuple[
        phx.discretization.CellMesh,
        phx.discretization.CellGeometrySpec,
        phx.geometry.PiecewiseLinearDomain,
    ],
) -> None:
    from phydrax.geometry._embedding_state import coordinate_certificate_request
    from phydrax.meshing._plc_mapped_support import (
        _mapped_block_maps,
        _prepare_mapped_cell_ownership,
    )

    mesh, geometry, _ = high_order_plc
    block = phx.discretization.CellBlock(
        "unsupported-surface", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)
    )
    with coordinate_certificate_request(phx.geometry.MeshCertificateLimits()) as budget:
        prepared = _prepare_mapped_cell_ownership(mesh, geometry, budget=budget)
        with pytest.raises(ValueError, match="MappedPlcSupport.elements"):
            _mapped_block_maps(
                block,
                phx.discretization.lagrange_element("triangle", 2),
                np.arange(6, dtype=np.int64)[None, :],
                prepared,
                [],
                {},
                prepare_expressions=True,
                exact_power=True,
                shared_images=None,
                budget=budget,
            )


def test_lifecycle_host_worksets_do_not_walk_numerical_model_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_qualification.scenarios import distributed

    values = jnp.asarray((1.0, 2.0, 3.0))
    original_leaves = jax.tree_util.tree_leaves
    walks: list[object] = []

    def counted_leaves(value: Any, *args: Any, **kwargs: Any) -> Any:
        walks.append(value)
        return original_leaves(value, *args, **kwargs)

    monkeypatch.setattr(jax.tree_util, "tree_leaves", counted_leaves)
    warm: list[float] = []
    continuation = distributed._DistributedFEContinuation(0.0, 3, warm, values)
    integrals = {"area": 1.0}
    references = {"area": 1.0}
    inventory = distributed._DistributedInventories(integrals, references)
    assert walks == []
    assert continuation.warm is warm
    assert continuation.values is values
    assert inventory.integrals is integrals
    assert inventory.references is references
    with pytest.raises(FrozenInstanceError):
        setattr(continuation, "free", 4)


@pytest.fixture(scope="module")
def hybrid_lifting_admission(
    request: pytest.FixtureRequest,
) -> tuple[phx.meshing.CellMeshingResult, int, dict[str, int]]:
    from time import perf_counter

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.meshing.qualification import prepare_hybrid_high_order_map
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import _corpus_profiles
    from tools.meshing_qualification.scenarios.hybrid import _hybrid_concave_publication

    profile = _corpus_profiles()["native-hybrid-lifecycle:concave-box-core-lifecycle"]
    controls = QualificationControls.from_record(
        {**QualificationControls().to_record(), **profile["controls"]}
    )
    generated = _hybrid_concave_publication(
        controls,
        {"started": perf_counter(), "targets": {"wall_seconds": controls.timeout}},
    ).generated
    assert generated.certification is not None
    limits = generated.certification.request.limits
    bounds = {"storage": 0}
    with HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_distance_evaluations,
        bytes_upper=limits.maximum_scratch_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        admit = budget._admit_ticket

        def observe(work: int, storage_upper: int) -> None:
            bounds["storage"] = max(bounds["storage"], storage_upper)
            admit(work, storage_upper)

        patch = pytest.MonkeyPatch()
        patch.setattr(budget, "_admit_ticket", observe)
        request.addfinalizer(patch.undo)
        with budget.activate():
            budget.reserve(7, 19)
            budget.reserve_geometry_queries(3)
            prepared = prepare_hybrid_high_order_map(
                generated,
                degree=2,
                maximum_nodes=32 * controls.capacity,
                maximum_entries=1024 * controls.capacity,
                budget=budget,
            )
        bounds["work"] = budget.required_work_units
        bounds["query"] = ticket.geometry_queries
        bounds["completed_work"] = ticket.work_units
        assert prepared.lifting_evidence.work_units == ticket.work_units - 7
        assert prepared.lifting_evidence.geometry_queries == ticket.geometry_queries - 3
        assert prepared.execution_evidence is None
    assert bounds["query"] == 3
    return generated, controls.capacity, bounds


@pytest.mark.parametrize("resource", ("work", "storage"))
@pytest.mark.parametrize("short", (False, True), ids=("at-limit", "one-unit-short"))
def test_hybrid_lift_uses_remaining_original_ticket_allowance(
    hybrid_lifting_admission: tuple[phx.meshing.CellMeshingResult, int, dict[str, int]],
    resource: str,
    short: bool,
) -> None:
    from phydrax._resource_ticket import HostBatchEstimateError, HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.meshing.qualification import prepare_hybrid_high_order_map

    generated, capacity, measured = hybrid_lifting_admission
    bounds = {name: measured[name] for name in ("work", "query", "storage")}
    bounds[resource] -= int(short)
    with HostBatchTicket(
        bounds["work"],
        queries_upper=bounds["query"],
        bytes_upper=bounds["storage"],
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=bounds["work"],
            maximum_memory_bytes=bounds["storage"],
        )
        with budget.activate():
            budget.reserve(7, 19)
            budget.reserve_geometry_queries(3)
            if short:
                with pytest.raises(
                    (CoordinateEnclosureResourceError, HostBatchEstimateError)
                ):
                    prepare_hybrid_high_order_map(
                        generated,
                        degree=2,
                        maximum_nodes=32 * capacity,
                        maximum_entries=1024 * capacity,
                        budget=budget,
                    )
                assert ticket.work_units == budget.work_units
                assert ticket.work_units <= bounds["work"]
                assert ticket.geometry_queries <= bounds["query"]
            else:
                prepared = prepare_hybrid_high_order_map(
                    generated,
                    degree=2,
                    maximum_nodes=32 * capacity,
                    maximum_entries=1024 * capacity,
                    budget=budget,
                )
                assert (
                    ticket.work_units == budget.work_units == measured["completed_work"]
                )
                assert ticket.geometry_queries == measured["query"]
                assert prepared.lifting_evidence.work_units == ticket.work_units - 7
                assert (
                    prepared.lifting_evidence.geometry_queries
                    == ticket.geometry_queries - 3
                )
                assert prepared.execution_evidence is None
    if short and resource == "work":
        # Lifting fits this allowance alone; the completed prefix makes the
        # continuation refuse instead of acquiring a fresh allowance.
        with HostBatchTicket(
            bounds["work"],
            queries_upper=bounds["query"],
            bytes_upper=bounds["storage"],
        ) as separate:
            with CoordinateEnclosureBudget(
                separate,
                maximum_work_units=bounds["work"],
                maximum_memory_bytes=bounds["storage"],
            ).activate() as independent:
                prepare_hybrid_high_order_map(
                    generated,
                    degree=2,
                    maximum_nodes=32 * capacity,
                    maximum_entries=1024 * capacity,
                    budget=independent,
                )
                assert separate.work_units == independent.work_units > 0


@pytest.mark.parametrize(
    "status",
    tuple(
        status
        for status in phx.meshing.MeshAdaptationStatus
        if status is not phx.meshing.MeshAdaptationStatus.COMPLETE
    ),
)
def test_hybrid_transition_rejects_incomplete_requested_operations(
    monkeypatch: pytest.MonkeyPatch,
    status: phx.meshing.MeshAdaptationStatus,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification.scenarios.hybrid import (
        _hybrid_qualification_transition,
    )

    adaptation = SimpleNamespace(
        status=status, transition=object(), geometry_transition=object()
    )
    monkeypatch.setattr(
        phx.meshing, "prepare_mesh_adaptation", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        phx.meshing, "execute_mesh_adaptation", lambda prepared, **kwargs: adaptation
    )
    record = {
        "started": perf_counter(),
        "targets": {"wall_seconds": 120.0},
        "stages_seconds": {},
    }
    with pytest.raises(RuntimeError, match="did not complete every requested cell"):
        _hybrid_qualification_transition(
            None, None, None, None, None, None, record, "refine", False
        )


def test_quad_hex_qualification_refuses_empty_profile_selection() -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.native_quad_hex_meshing import run

    controls = QualificationControls(scenario="native-quad-hex", corpus_profile=())
    with pytest.raises(ValueError, match="quad_hex.corpus_profile"):
        run(controls)


@pytest.mark.meshcore
@pytest.mark.skipif(
    not phx.meshing.meshcore_available(),
    reason="native phydrax-meshcore library is not available",
)
def test_remap_resources_attribute_each_target_preparation_to_its_own_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Callable

    from tools.meshing_qualification._runtime import _measured, StageMeasurement
    from tools.meshing_qualification.scenarios import remap_p1

    produced: list[tuple[StageMeasurement, object]] = []

    def recorded[T](call: Callable[[], T], /) -> tuple[T, StageMeasurement]:
        value, stage = _measured(call)
        produced.append((stage, value))
        return value, stage

    monkeypatch.setattr(remap_p1, "_measured", recorded)
    record = remap_p1.qualify_remap_p1()
    resources, targets = record["resources"], record["target"]
    assert isinstance(resources, dict) and isinstance(targets, dict)
    values = {id(stage): value for stage, value in produced}
    # The finite-volume and L2 routes each prepare their own target; each
    # resource must carry the measurement of the stage that built that target.
    fv_target = values[id(resources["target_prepare"])]
    l2_target = values[id(resources["l2_target_prepare"])]
    assert isinstance(
        fv_target, phx.discretization.UnstructuredFiniteVolumeDiscretization
    )
    assert fv_target.prepared_id == targets["finite_volume"]["prepared_id"]
    assert isinstance(l2_target, phx.discretization.PreparedL2ProjectionTarget)
    assert (
        l2_target.discretization.prepared_id == targets["finite_element"]["prepared_id"]
    )


def test_family_physical_failure_has_its_own_epoch_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification.scenarios import family

    record = {"stages_seconds": {}}
    inputs: Any = SimpleNamespace(
        case={},
        args=SimpleNamespace(repeats=1),
        record=record,
        ledger=StageLedger(record),
    )
    result: Any = SimpleNamespace(mesh=SimpleNamespace(coordinates=np.zeros((1, 2))))
    owners: Any = SimpleNamespace(
        solver=(object(), object(), np.zeros((0,), dtype=np.int64)),
        field_spaces={}, fv=None,
    )
    monkeypatch.setattr(
        family,
        "_compiled_solver_stages",
        lambda problem, seed, repeats: (np.zeros((1,)), None, {}),
    )

    def fail_physical(space: Any, values: Any, case: Any) -> None:
        raise ValueError("independent physical quadrature failed")

    monkeypatch.setattr(family, "_family_qualification_physical", fail_physical)
    from tests._support.coordinate_references import coordinate_reference_budget

    with coordinate_reference_budget() as budget, pytest.raises(
        ValueError, match="independent physical quadrature failed"
    ):
        family._family_solve_epoch(
            inputs, result, None, {}, None, 0, owners, budget=budget
        )
    assert record["active_phase"] == "physical_error_epoch_0"
    measurements = record["stage_measurements"]
    assert measurements["solver_execution_epoch_0"]["invocations"] == 1
    assert measurements["physical_error_epoch_0"]["invocations"] == 1


@pytest.mark.parametrize(
    "family_case", ("transfinite-quad", "curved-quarter-extrusion-quad")
)
def test_family_epoch_reuses_explicit_actual_allfield_owners(
    monkeypatch: pytest.MonkeyPatch, family_case: str
) -> None:
    import jax.numpy as jnp

    import phydrax as phx
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import family

    source, request, options, case = family._family_qualification_fixture(
        QualificationControls(family_case=family_case, resolution=2)
    )
    result = (
        phx.meshing.NativeMeshingProvider(options)
        .plan(source, request, coordinate_contract=phx.SpatialCoordinateContract.si())
        .execute()
    )
    owners = family._family_allfield_spaces(result)
    space, problem, _ = family._family_qualification_solver(result, case)
    values = space.project("u", lambda points, args: jnp.sin(jnp.pi * points[:, 0]))
    fv = family._family_fv(result)

    def refuse_duplicate_preparation(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("An explicitly owned accepted epoch rebuilt its FE plans.")

    monkeypatch.setattr(family, "_family_allfield_spaces", refuse_duplicate_preparation)
    monkeypatch.setattr(family, "_family_fv", refuse_duplicate_preparation)
    fields = family._family_allfield_state(result, spaces=owners, fv=fv)
    inventory = family._family_allfield_inventory(result, fields, spaces=owners, fv=fv)
    if family_case == "curved-quarter-extrusion-quad":
        # Both accepted scalar/history columns are affine in the physical axial
        # coordinate.  Integrate them independently against the original
        # embedded geometry, not a square-Jacobian determinant.
        quadrature = phx.integration.ReferenceQuadrilateralRule(
            phx.integration.GaussLegendreRule(10)
        ).materialize()
        expected = np.zeros((2,), dtype=np.float64)
        for block_index in range(len(result.mesh.blocks)):
            geometry = owners["H1"].evaluate_block_geometry(
                "current",
                block_index,
                owners["H1"].default_runtime.coordinates,
                quadrature.points,
                quadrature.weights,
            )
            axial = np.asarray(geometry.physical_points)[..., 0]
            samples = np.stack((1.0 + axial, 0.5 + 0.125 * axial), axis=-1)
            expected += np.sum(
                np.asarray(geometry.physical_weights)[..., None] * samples,
                axis=(0, 1),
            )
        for kind in ("H1", "DG"):
            np.testing.assert_allclose(
                inventory[kind]["physical_content"], expected, rtol=1e-11, atol=1e-12
            )
    accepted = family._family_accepted_epoch(
        result, space, problem, values, fields, 0, None, field_spaces=owners, fv=fv
    )
    assert set(inventory) == {*owners, "material_content"}
    for kind, owner in owners.items():
        assert owner.mesh is result.mesh
        assert inventory[kind]["field_space_ids"] == [
            field.field_space_id for field in owner.field_spaces
        ]
        np.testing.assert_array_equal(
            accepted.fields[f"{kind}/current/coefficients"], fields[kind][:, 0]
        )
        np.testing.assert_array_equal(
            accepted.fields[f"{kind}/history/coefficients"], fields[kind][:, 1]
        )


def test_unit_box_carries_every_accepted_epoch_without_reseeding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import hybrid

    controls = QualificationControls(
        adaptation_rounds=2,
        timeout=120.0,
        hybrid_region_densities=(2.0,),
        hybrid_material_site="fluid-density",
    )
    record: dict[str, Any] = {
        "started": perf_counter(),
        "targets": {"wall_seconds": controls.timeout},
        "stages_seconds": {},
        "invariants": [],
    }
    results = [
        SimpleNamespace(
            result_id=f"result-{index}",
            geometry=object(),
            mesh=SimpleNamespace(
                topology_id="root" if index % 2 == 0 else f"refined-{index}",
                blocks=(
                    SimpleNamespace(
                        cell_kind="tetrahedron", global_ids=np.asarray((11, 12))
                    ),
                    SimpleNamespace(cell_kind="prism", global_ids=np.asarray((20,))),
                ),
            ),
        )
        for index in range(5)
    ]
    bank_names = (
        "fe/H1",
        "fe/DG",
        "fe/Hcurl",
        "fe/Hdiv",
        "fe/u",
        "fe/damage-history",
        "fe/accepted-epoch",
        "fv/density",
        "fv/accepted-epoch",
        "fe/original-u",
    )
    epochs = [
        SimpleNamespace(
            epoch_id=f"epoch-{index}",
            fields={name: np.asarray((index,)) for name in bank_names},
        )
        for index in range(5)
    ]
    limits = object()
    requests: list[Any] = []
    adaptations: list[Any] = []
    warm_calls: list[tuple[Any, Any, Any]] = []
    rejected: list[tuple[Any, Any]] = []
    scalar_epochs: list[tuple[int, Any]] = []
    archived: list[Any] = []
    restored: list[tuple[Any, Any]] = []
    warm_states = [object(), object()]
    preparations = [object() for _ in results]
    monkeypatch.setattr(
        hybrid,
        "_hybrid_scalar_request",
        lambda result, limits, previous, call: (call(None), preparations[0]),
    )

    monkeypatch.setattr(
        hybrid,
        "_hybrid_qualification_solver",
        lambda result, **kwargs: (None, None, None),
    )

    def solve_epoch(
        result: Any,
        space: Any,
        problem: Any,
        free: Any,
        carried: Any,
        args: Any,
        record: Any,
        epoch: int,
        name: str,
        *,
        limits: Any,
        current_preparation: Any,
    ) -> tuple[np.ndarray, float, Any]:
        assert current_preparation is preparations[epoch]
        scalar_epochs.append((epoch, carried))
        return np.asarray((epoch,), dtype=np.float64), 1.0e-12, current_preparation

    monkeypatch.setattr(hybrid, "_hybrid_qualification_epoch", solve_epoch)
    monkeypatch.setattr(
        phx.meshing, "MeshAdaptationPolicy", lambda *args, **kwargs: kwargs
    )
    monkeypatch.setattr(
        phx.meshing, "BRepAssociationTransfer", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        hybrid,
        "_hybrid_qualification_columns",
        lambda result: SimpleNamespace(columns_id=result.mesh.topology_id),
    )

    def mark(*args: Any, **kwargs: Any) -> Any:
        request = SimpleNamespace(args=args, **kwargs)
        requests.append(request)
        return request

    monkeypatch.setattr(phx.meshing, "MarkedMeshAdaptation", mark)

    def transition(
        result: Any,
        space: Any,
        values: Any,
        request: Any,
        policy: Any,
        source: Any,
        record: Any,
        label: str,
        coarsening: bool,
        *,
        coordinate_operands: Any = None,
    ) -> tuple[Any, None, None, None, np.ndarray, Any]:
        index = len(adaptations) + 1
        assert result is results[index - 1]
        assert coarsening is (index % 2 == 0)
        adaptation = SimpleNamespace(
            target=results[index],
            hierarchy=SimpleNamespace(
                siblings=SimpleNamespace(
                    all_child_ids=lambda: np.asarray((100, 101), dtype=np.int64)
                )
            ),
        )
        adaptations.append(adaptation)
        return (
            adaptation,
            None,
            None,
            None,
            np.asarray((index - 1.0,)),
            preparations[index],
        )

    monkeypatch.setattr(hybrid, "_hybrid_qualification_transition", transition)
    monkeypatch.setattr(
        hybrid,
        "_HybridAdaptationCycle",
        lambda refinement, coarsening: SimpleNamespace(
            refinement=refinement, coarsening=coarsening
        ),
    )

    def warm_cycle(
        args: Any,
        record: Any,
        accepted: Any,
        adaptation: Any,
        cycle: int,
        *,
        part_name: str,
        limits: Any,
        carried: Any,
        original_scalar_values: Any,
        original_scalar_preparations: Any,
        prepared_operands: Any = None,
        transition_operands: Any = (None, None),
    ) -> tuple[tuple[Any, ...], Any]:
        warm_calls.append((accepted, adaptation, carried))
        assert part_name == "unit-box-layer-core"
        assert original_scalar_preparations == tuple(
            preparations[2 * cycle : 2 * cycle + 3]
        )
        for offset, actual in enumerate(original_scalar_values):
            np.testing.assert_array_equal(actual, (2 * cycle + offset,))
        return tuple(epochs[2 * cycle : 2 * cycle + 3]), warm_states[cycle]

    monkeypatch.setattr(hybrid, "_hybrid_warm_cycle", warm_cycle)
    monkeypatch.setattr(
        hybrid,
        "_hybrid_qualification_rejection",
        lambda adaptation, values, policy, record, cycle: rejected.append(
            (adaptation, values)
        ),
    )
    monkeypatch.setattr(
        hybrid,
        "require_restored_layer_core_result_identity",
        lambda root, result: restored.append((root, result)),
    )
    monkeypatch.setattr(
        hybrid,
        "_hybrid_qualification_acceptance",
        lambda result, source: {"passed": True},
    )

    def archive(
        args: Any,
        record: Any,
        source: Any,
        specification: Any,
        values: Any,
    ) -> Any:
        archived.append(values)
        return SimpleNamespace(physical_error=1.0e-12)

    monkeypatch.setattr(hybrid, "_hybrid_unit_box_archive_continuation", archive)
    error = hybrid._hybrid_qualification_epochs(
        results[0], object(), object(), SimpleNamespace(limits=limits), controls, record
    )
    assert error == 1.0e-12
    assert archived == [tuple(epochs)]
    assert len(warm_calls) == 2
    assert warm_calls[0][0] is results[0]
    assert warm_calls[0][2] is None
    assert warm_calls[1][0] is results[2]
    assert warm_calls[1][2] is warm_states[0]
    for cycle, (_, adaptation, _) in enumerate(warm_calls):
        assert adaptation.refinement is adaptations[2 * cycle]
        assert adaptation.coarsening is adaptations[2 * cycle + 1]
    assert scalar_epochs[0] == (0, None)
    assert [epoch for epoch, _ in scalar_epochs] == list(range(5))
    assert all(carried is not None for _, carried in scalar_epochs[1:])
    for request in requests[::2]:
        np.testing.assert_array_equal(request.args[0], (11, 12))
    for request in requests[1::2]:
        np.testing.assert_array_equal(request.args[1], (100, 101))
    for cycle, (adaptation, values) in enumerate(rejected):
        assert adaptation is adaptations[2 * cycle]
        np.testing.assert_array_equal(values[0], (2 * cycle + 1,))
        assert values[1] is epochs[2 * cycle + 1].fields
    assert restored == [(results[0], results[2]), (results[2], results[4])]


@pytest.mark.parametrize("changed_epoch", (None, 0, 2, 4))
def test_unit_box_archive_retains_complete_history_and_original_deadline(
    monkeypatch: pytest.MonkeyPatch,
    changed_epoch: int | None,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import hybrid

    controls = QualificationControls(
        adaptation_rounds=2,
        capacity=20000,
        timeout=120.0,
        hybrid_region_densities=(2.0,),
        hybrid_material_site="fluid-density",
    )
    started = perf_counter()
    record: dict[str, Any] = {
        "started": started,
        "targets": {"wall_seconds": controls.timeout},
        "stages_seconds": {},
    }
    epochs: Any = tuple(SimpleNamespace(epoch_id=f"epoch-{index}") for index in range(5))
    restored = [{"epoch_id": epoch.epoch_id} for epoch in epochs]
    if changed_epoch is not None:
        restored[changed_epoch] = {"epoch_id": "substituted"}
    source: Any = object()
    specification: Any = object()
    sections = {
        "restored_epochs": restored,
        "allfields": {"fields": ["H1", "DG", "Hcurl", "Hdiv"]},
        "finite_volume": {"region_inventories": [2.0]},
        "history": {"epochs": [{"name": "coarsen"}]},
        "material_declaration": {"densities": [2.0], "site_id": "fluid-density"},
        "original_scalar": {"field": "original-u", "degree": 1},
    }
    archives: list[tuple[Any, ...]] = []
    writes: list[tuple[Path, Any]] = []
    consumers: list[Any] = []
    continued = SimpleNamespace(physical_error=1.0e-12)

    def lineage(*args: Any) -> dict[str, Any]:
        archives.append(args)
        return {"accepted_data": {"accepted_epochs": args[3]}}

    def write(path: Path, records: Any) -> Any:
        writes.append((path, records))
        return SimpleNamespace(path=path, content_id="authenticated-lineage")

    def consume(args: Any, continuation: Any, *, expected_rounds: int) -> Any:
        consumers.append(args)
        assert expected_rounds == 2
        assert continuation["started"] == started
        assert continuation["targets"] is record["targets"]
        assert args.timeout == 120.0
        assert args.capacity == 20000
        assert args.hybrid_region_densities == (2.0,)
        assert args.hybrid_material_site == "fluid-density"
        assert args.hybrid_archive == str(writes[0][0])
        assert args.hybrid_content_id == "authenticated-lineage"
        continuation.update(sections)
        return continued

    monkeypatch.setattr(hybrid, "layer_core_lineage_records", lineage)
    monkeypatch.setattr(hybrid, "write_meshing_source_closure", write)
    monkeypatch.setattr(hybrid, "_hybrid_qualification_archived_profile", consume)
    if changed_epoch is None:
        actual = hybrid._hybrid_unit_box_archive_continuation(
            controls, record, source, specification, epochs
        )
        assert actual is continued
        assert record["accepted_epochs"] is restored
        for section in (
            "allfields",
            "finite_volume",
            "history",
            "material_declaration",
            "original_scalar",
        ):
            assert record[section] is sections[section]
    else:
        with pytest.raises(RuntimeError, match="retain every warm accepted epoch"):
            hybrid._hybrid_unit_box_archive_continuation(
                controls, record, source, specification, epochs
            )
        assert "accepted_epochs" not in record
    assert len(archives) == len(writes) == len(consumers) == 1
    assert archives[0][0] is source
    assert archives[0][1] is specification
    assert archives[0][2].route == "layer_core"
    assert archives[0][3] is epochs
    assert "started" not in record["cold_continuation"]
    assert "active_phase" not in record["cold_continuation"]


def test_original_unit_box_scalar_rejects_wrong_actual_p1_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    from phydrax.meshing.qualification import _layer_continuation as continuation

    monkeypatch.setattr(
        continuation,
        "_original_scalar_space",
        lambda *args, **kwargs: SimpleNamespace(
            dof_maps=(SimpleNamespace(global_dof_count=3),)
        ),
    )
    result: Any = SimpleNamespace()
    with pytest.raises(ValueError, match="actual P1 layout"):
        continuation._original_scalar_physical(
            result,
            jnp.zeros((4,)),
            plans=continuation.LayerCoreLifecyclePlans(),
        )


@pytest.mark.parametrize("change", ("missing", "shape", "values"))
def test_original_unit_box_scalar_comparison_rejects_changed_bank(change: str) -> None:
    from types import SimpleNamespace

    from phydrax.meshing.qualification import _layer_continuation as continuation

    actual = jnp.asarray((0.0, 1.0, 2.0))
    reference = (
        {}
        if change == "missing"
        else {"fe/original-u": jnp.zeros((2,)) if change == "shape" else actual + 0.1}
    )
    state: Any = SimpleNamespace(original_scalar=actual)
    epoch: Any = SimpleNamespace(fields=reference)
    with pytest.raises(ValueError, match="original_scalar"):
        continuation._original_scalar_comparison(state, epoch)


@pytest.mark.meshcore
@pytest.mark.skipif(
    not phx.meshing.meshcore_available(),
    reason="native phydrax-meshcore library is not available",
)
def test_unit_box_original_p1_bank_is_archived_and_independently_cold_continued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios import hybrid

    controls = QualificationControls(
        hybrid_profile="unit-box",
        resolution=2,
        capacity=20000,
        adaptation_rounds=2,
        geometry_order=2,
        repeats=1,
        timeout=120.0,
        hybrid_region_densities=(2.0,),
        hybrid_material_site="fluid-density",
    )
    original_epoch = hybrid._hybrid_qualification_epoch
    original_write = hybrid.write_meshing_source_closure
    warm_values: list[jax.Array] = []
    archived_epochs: list[Any] = []

    def capture_epoch(*args: Any, **kwargs: Any) -> tuple[jax.Array, float, Any]:
        values, physical_error, preparation = original_epoch(*args, **kwargs)
        warm_values.append(values)
        return values, physical_error, preparation

    def capture_archive(path: Path, records: Any) -> Any:
        epochs = records["accepted_data"]["accepted_epochs"]
        assert len(epochs) == len(warm_values) == 5
        for index, epoch in enumerate(epochs):
            np.testing.assert_array_equal(
                epoch.fields["fe/original-u"], warm_values[index]
            )
            declaration = epoch.declarations[0]
            field = next(
                field
                for field in declaration.finite_element_fields
                if field.name == "original-u"
            )
            assert all(element.degree == 1 for element in field.elements)
            role = next(
                role
                for role in declaration.state_roles
                if role.entry_name == "fe/original-u"
            )
            assert role.shape == warm_values[index].shape
            assert role.field_name == "original-u"
            assert role.index_binding is not None
        archived_epochs.extend(epochs)
        return original_write(path, records)

    monkeypatch.setattr(hybrid, "_hybrid_qualification_epoch", capture_epoch)
    monkeypatch.setattr(hybrid, "write_meshing_source_closure", capture_archive)
    record = hybrid._hybrid_qualification_lifecycle(controls)
    assert record["status"] == "passed", record
    assert len(archived_epochs) == 5
    cold = record["cold_continuation"]
    assert isinstance(cold, dict)
    assert cold["continued_from"]["epoch_id"] == archived_epochs[-2].epoch_id
    assert cold["warm_continuation_defects"]["original-u"] <= 1.0e-9
    assert cold["restoration_defects"]["original-u"] <= 1.0e-9
    scalar = record["original_scalar"]
    assert isinstance(scalar, dict)
    assert scalar["field"] == "original-u"
    assert scalar["degree"] == 1
    assert scalar["dofs"][-1] == warm_values[-1].shape[0]
    assert scalar["accepted_bank_origin"] == "actual-archived-P1-state-transfer"
    assert scalar["solve_comparison_defect"] <= 1.0e-9
    assert scalar["physical"]["l2_error"] <= controls.target_error
    assert scalar["independent_physical"]["l2_error"] <= controls.target_error
    assert abs(scalar["physical"]["analytic_volume_defect"]) <= 1.0e-10
    independent = scalar["independent_solve"]
    assert independent["name"] == "original-u"
    assert independent["compiler"]["warm_samples_seconds"]
    capacities, targets = record["control_capacities"], record["targets"]
    assert isinstance(capacities, dict) and isinstance(targets, dict)
    assert capacities["entity_capacity"] == 20000
    assert targets["wall_seconds"] == 120.0


@pytest.mark.meshcore
def test_polyhedral_cold_verdict_requires_the_original_warm_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A real cold load at another native artifact path cannot reuse a warm verdict."""
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification._records import _corpus_profiles
    from tools.meshing_qualification._runtime import _runtime_identity
    from tools.meshing_qualification.scenarios import polyhedral

    runtime = _runtime_identity()
    native = runtime["meshcore_build"]
    assert isinstance(native, dict)
    independent_library = tmp_path / "independent-meshcore.so"
    independent_library.symlink_to(native["loaded_path"])
    original_cold = polyhedral._polyhedral_qualification_cold

    def independently_loaded_cold(
        root: Any,
        expected: dict[str, Any],
        kind: str,
        accepted_epochs: int,
        repeats: int,
        target_error: float,
        stages: dict[str, Any],
        deadline: Deadline,
        *,
        expected_build_id: str,
    ) -> dict[str, Any]:
        # Only the actual child's native load location changes. Its archive,
        # source controls, physical solves and process evidence remain real.
        with monkeypatch.context() as context:
            context.setenv("PHYDRAX_MESHCORE_LIBRARY", str(independent_library))
            return original_cold(
                root,
                expected,
                kind,
                accepted_epochs,
                repeats,
                target_error,
                stages,
                deadline,
                expected_build_id=expected_build_id,
            )

    monkeypatch.setattr(
        polyhedral, "_polyhedral_qualification_cold", independently_loaded_cold
    )
    controls = QualificationControls(
        scenario="native-polyhedral-lifecycle",
        polyhedral_source="polyhedral-material-L",
        **_corpus_profiles()["polyhedral-material-L"]["controls"],
    )
    record = polyhedral.qualify_native_polyhedral(controls)
    evidence_path = tmp_path / "polyhedral-cold-build-record.json"
    evidence_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    assert record["status"] == "failed", record
    assert "cold" in record, json.dumps(record, indent=2)
    cold = record["cold"]
    assert isinstance(cold, dict)
    restored = cold["fresh_interpreter"]
    assert restored["status"] == "passed"
    assert restored["process_id"] != os.getpid()
    assert cold["accepted_fields_bitwise_preserved"] is True
    assert cold["cold_build_matches_warm"] is False
    assert cold["status"] == "failed"
    assert "source/build drift" in cold["failure"]
    assert restored["runtime"]["build_id"] != cold["warm_build_id"]
    assert restored["runtime"]["meshcore_build"]["loaded_path"] == str(
        independent_library
    )


def test_polyhedral_operator_oracles_keep_independent_residual_and_raw_transpose() -> (
    None
):
    from tools.meshing_qualification.scenarios.polyhedral import (
        _polyhedral_qualification_operator_evidence,
    )

    matrix = np.asarray(
        ((3.0, 1.0, -2.0), (-0.5, 2.0, 0.25), (1.0, -3.0, 4.0)),
        dtype=np.float64,
    )
    operator = phx.linalg.DenseLinearOperator(jnp.asarray(matrix))
    solution = np.asarray((0.2, -0.3, 0.7), dtype=np.float64)
    for residual_offset in (0.0, 0.125):
        rhs = matrix @ solution + residual_offset
        record = _polyhedral_qualification_operator_evidence(
            operator, jnp.asarray(rhs), jnp.asarray(solution)
        )
        np.testing.assert_allclose(
            record["maximum_algebraic_residual"],
            np.max(np.abs(matrix @ solution - rhs)),
            rtol=0.0,
            atol=1e-14,
        )
        left = np.sin(np.arange(3, dtype=np.float64) + 0.3)
        right = np.cos(np.arange(3, dtype=np.float64) + 0.7)
        independent = abs(np.vdot(left, matrix @ right) - np.vdot(matrix.T @ left, right))
        np.testing.assert_allclose(
            record["raw_transpose_pairing_defect"],
            independent,
            rtol=0.0,
            atol=1e-14,
        )


@pytest.mark.parametrize("kind", ("H1", "DG"))
def test_family_embedded_scalar_content_integrates_both_actual_columns(kind: str) -> None:
    from tools.meshing_qualification.scenarios import family

    mesh = phx.discretization.CellMesh(
        np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 1)), dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "surface", "quadrilateral", np.asarray(((0, 1, 2, 3),))
            ),
        ),
    )
    template = family.family_transport_field_templates("quadrilateral", kind=kind)[0][1]
    field = phx.discretization.FiniteElementFieldSpec(
        "current",
        {"surface": template.elements[0]},
        component_shape=template.component_shape,
    )
    space = phx.discretization.FiniteElementPlan(
        mesh,
        field,
        coordinate_spec=phx.discretization.CellGeometrySpec.affine(mesh),
    ).prepare()
    current = space.project("current", lambda points, args: 1.0 + points[:, 0])
    history = space.project("current", lambda points, args: 0.5 + 0.125 * points[:, 0])
    actual = family._family_scalar_content(
        space, np.asarray(jnp.stack((current, history), axis=-1))
    )
    np.testing.assert_allclose(
        actual, np.sqrt(2.0) * np.asarray((1.5, 0.5625)), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("embedded", (False, True))
def test_family_initial_form_projection_preserves_authored_moments(
    embedded: bool,
) -> None:
    from types import SimpleNamespace

    from tools.meshing_qualification.scenarios import family

    points = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.float64)
    if embedded:
        points = np.column_stack((points, points[:, 1]))
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "quadrilateral", np.asarray(((0, 1, 2, 3),))
            ),
        ),
    )
    result = SimpleNamespace(
        mesh=mesh, geometry=phx.discretization.CellGeometrySpec.affine(mesh)
    )
    spaces = family._family_allfield_spaces(result)
    fields = family._family_allfield_state(result, spaces=spaces)
    for kind in ("Hcurl", "Hdiv"):
        for column, (name, vector) in enumerate(
            (
                ("current", (1.0, 0.0, 0.0) if embedded else (1.0, 1.0)),
                ("history", (0.125, 0.0, 0.0) if embedded else (0.125, 0.1)),
            )
        ):
            expected = spaces[kind].project(
                name,
                lambda locations, args: jnp.broadcast_to(
                    jnp.asarray(vector), locations.shape
                ),
            )
            np.testing.assert_array_equal(fields[kind][:, column], expected)


def _distributed_resource_launch_probe(
    monkeypatch: pytest.MonkeyPatch,
    args: QualificationControls,
    role: str,
    mutation: str | None = None,
) -> list[dict[str, Any]]:
    """Exercise real CLI parsing and record transport without claiming science."""
    from collections.abc import Mapping, Sequence

    from phydrax.execution import LocalProcessLaunchPlan, LocalProcessResult
    from tools.meshing_qualification import __main__ as cli
    from tools.meshing_qualification.scenarios import distributed

    captured: list[QualificationControls] = []

    class Parsed(Exception):
        pass

    def capture(controls: QualificationControls) -> dict[str, object]:
        captured.append(controls)
        raise Parsed

    def launch(
        argv: Sequence[str],
        plan: LocalProcessLaunchPlan,
        *,
        environment: Mapping[str, str] | None = None,
        cwd: str | None = None,
    ) -> tuple[LocalProcessResult, ...]:
        del plan, cwd
        assert environment is not None
        with monkeypatch.context() as patch:
            patch.setattr(sys, "argv", [argv[2], *argv[3:]])
            patch.setattr(distributed, "run", capture)
            with pytest.raises(Parsed):
                cli.main()
        child = captured[-1]
        expected = distributed._distributed_qualification_request(args)
        frozen = distributed._distributed_qualification_request(child)
        assert json.loads(environment[distributed._DISTRIBUTED_REQUEST]) == expected
        assert frozen == expected
        for name in (
            "maximum_work_units", "maximum_geometry_queries", "maximum_memory_bytes"
        ):
            value = expected[name]
            flag = "--" + name.replace("_", "-")
            if value is None:
                assert flag not in argv
                assert frozen[name] is None
            else:
                assert argv[argv.index(flag) + 1] == str(value)
        if mutation in (
            "maximum_work_units", "maximum_geometry_queries", "maximum_memory_bytes"
        ):
            frozen[mutation] = 1
        elif mutation == "missing_resource":
            del frozen["maximum_geometry_queries"]
        request_id = distributed.canonical_fingerprint(frozen)
        record: dict[str, Any] = {
            "status": "passed", "science_executed": False,
            "request": frozen, "request_id": request_id,
        }
        if mutation == "stale_identity":
            record["request_id"] = "not-the-current-request"
        elif mutation == "missing_request":
            del record["request"]
        elif mutation == "missing_identity":
            del record["request_id"]
        count = args.distributed_processes if role == "producer" else args.restart_parts
        directory = Path(environment["PHYDRAX_MESHING_RECORD_DIRECTORY"])
        for rank in range(count):
            (directory / f"process-{rank}.json").write_text(json.dumps(record))
        return tuple(LocalProcessResult(rank, 0, b"", b"") for rank in range(count))

    monkeypatch.setattr(distributed, "launch_local_processes", launch)
    return distributed._distributed_qualification_launch(args, {}, perf_counter(), role)


@pytest.mark.parametrize("role", ("producer", "consumer"))
@pytest.mark.parametrize("work", (None, 1_280_000))
@pytest.mark.parametrize("queries", (None, 2_560_000))
@pytest.mark.parametrize("memory_gib", (12, 16))
def test_distributed_resource_command_roundtrip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    role: str, work: int | None, queries: int | None, memory_gib: int,
) -> None:
    args = QualificationControls(
        timeout=180.0, maximum_work_units=work,
        maximum_geometry_queries=queries, maximum_memory_bytes=memory_gib * 1024**3,
        distributed_parts=4, distributed_processes=4, restart_parts=2,
        distributed_checkpoint_root=str(tmp_path / "checkpoint"),
    )
    records = _distributed_resource_launch_probe(monkeypatch, args, role)
    for record in records:
        assert record["science_executed"] is False
        assert record["request"]["maximum_work_units"] == work
        assert record["request"]["maximum_geometry_queries"] == queries
        assert record["request"]["maximum_memory_bytes"] == memory_gib * 1024**3
        assert record["request"]["timeout"] == 180.0


@pytest.mark.parametrize("role", ("producer", "consumer"))
@pytest.mark.parametrize(
    "mutation",
    (
        "maximum_work_units", "maximum_geometry_queries", "maximum_memory_bytes",
        "missing_resource", "stale_identity", "missing_request", "missing_identity",
    ),
)
def test_distributed_resource_returned_record_mismatch_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, role: str, mutation: str,
) -> None:
    args = QualificationControls(
        maximum_work_units=1_280_000, maximum_geometry_queries=2_560_000,
        maximum_memory_bytes=12 * 1024**3,
        distributed_parts=4, distributed_processes=4, restart_parts=2,
        distributed_checkpoint_root=str(tmp_path / "checkpoint"),
    )
    # Every rank lies consistently and recomputes its ID for changed controls:
    # agreement among ranks is not agreement with the authored parent request.
    with pytest.raises(QualificationFailure, match="exact parent request"):
        _distributed_resource_launch_probe(monkeypatch, args, role, mutation)


@pytest.mark.parametrize("role,count", (("producer", 4), ("consumer", 2)))
@pytest.mark.parametrize(
    "changed",
    (
        "maximum_work_units", "maximum_geometry_queries", "maximum_memory_bytes",
        "missing_resource", "missing_parent", "unchanged",
    ),
)
def test_distributed_resource_child_admission_binds_parent_before_science(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    role: str, count: int, changed: str,
) -> None:
    from tools.meshing_qualification.scenarios import distributed

    args = QualificationControls(
        maximum_work_units=1_280_000, maximum_geometry_queries=2_560_000,
        maximum_memory_bytes=12 * 1024**3,
        distributed_parts=4, distributed_processes=4, restart_parts=2,
        distributed_checkpoint_root=str(tmp_path),
    )
    parent = distributed._distributed_qualification_request(args)
    if changed == "missing_resource":
        del parent["maximum_geometry_queries"]
    elif changed not in ("missing_parent", "unchanged"):
        parent[changed] = 1
    monkeypatch.setenv(distributed._DISTRIBUTED_ROLE, role)
    monkeypatch.setattr(jax, "process_count", lambda: count)
    if changed == "missing_parent":
        monkeypatch.delenv(distributed._DISTRIBUTED_REQUEST, raising=False)
    else:
        monkeypatch.setenv(distributed._DISTRIBUTED_REQUEST, json.dumps(parent))
    if changed == "unchanged":
        topology = distributed._distributed_qualification_topology(args)
        assert topology["role"] == role
    else:
        with pytest.raises(ValueError, match=r"topology.request must"):
            distributed._distributed_qualification_topology(args)


@pytest.mark.parametrize(
    "name", ("maximum_work_units", "maximum_geometry_queries", "maximum_memory_bytes")
)
def test_distributed_resource_frozen_identity_binds_each_control(
    monkeypatch: pytest.MonkeyPatch, name: str,
) -> None:
    from dataclasses import replace

    from tools.meshing_qualification.scenarios import distributed

    args = QualificationControls(
        maximum_work_units=1_280_000, maximum_geometry_queries=2_560_000,
        maximum_memory_bytes=12 * 1024**3,
    )
    # Missing checkpoint root deliberately refuses before any scientific work.
    # Even a refused lifecycle must retain all actual authored resource controls.
    monkeypatch.setattr(distributed, "_runtime_identity", lambda: {})
    before = distributed._distributed_qualification_lifecycle(args)
    after = distributed._distributed_qualification_lifecycle(replace(args, **{name: 1}))
    assert before["status"] == after["status"] == "failed"
    assert before["request_id"] != after["request_id"]
    assert before["request"] == distributed._distributed_qualification_request(args)
    assert before["request_id"] == distributed.canonical_fingerprint(before["request"])


@pytest.mark.parametrize("work,queries", ((1, 7), (1_280_000, 2_560_000)))
def test_distributed_resource_source_uses_exact_authored_limits_and_refuses_overrun(
    work: int, queries: int,
) -> None:
    from phydrax._meshcore import MeshcoreError, NativeExecutionScope
    from tools.meshing_qualification.scenarios import distributed

    args = QualificationControls(
        timeout=180.0, maximum_work_units=work,
        maximum_geometry_queries=queries, maximum_memory_bytes=12 * 1024**3,
    )
    _, request, _ = distributed._distributed_qualification_source(args)
    assert request.limits.maximum_work_units == work
    assert request.limits.maximum_geometry_queries == queries
    assert request.limits.maximum_wall_seconds == 180.0
    # External group memory is not a substitute for the native scratch cap.
    assert request.limits.maximum_scratch_bytes == 4096 * args.capacity
    with pytest.raises(MeshcoreError, match="CAPACITY_EXCEEDED"):
        with NativeExecutionScope(request.limits.native_limits()) as scope:
            scope.batch(work + 1)
