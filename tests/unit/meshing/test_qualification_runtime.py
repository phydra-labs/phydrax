# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import logging
import subprocess
import sys
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
from tools.compile_counter import CompileCounter
from tools.meshing_qualification._runtime import (
    Deadline,
    peak_resident_bytes,
    run_campaign_process,
    StageLedger,
)
from tools.meshing_qualification.scenarios.design import (
    _design_qualification_condition_matrix,
    _design_qualification_derivative_probe,
)
from tools.meshing_qualification.scenarios.distributed import (
    _continued_distributed_fe_solve,
    _continued_distributed_fe_system,
)


# Fixed empirical allowances from test_qualification_runtime.py
# at the pre-migration nine-node affine continuation fixture; no meshing acceptance policy is relaxed.
# Nine-node continued FE solve for u=x+y on the unit square, with values from 0 through
# 2.
_AFFINE_CONTINUATION_ATOL = 1e-14


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
                equation, parameter, direction, lower, upper, step
            ),
        )
        repeated = ledger.run(
            "design.derivative.warm.capacity-3",
            lambda: _design_qualification_derivative_probe(
                changed_equation,
                parameter,
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
    assert "_design_qualification_derivative_values" in counter.events[0].kernel_name
    assert counter.events[0].signature is not None
    assert counter.events[0].stage == "design.derivative-probe.capacity-3"


def test_distributed_continuation_kernels_reuse_capacity_bucket(
    affine_problem: phx.equations.CompiledFiniteElementProblem,
) -> None:
    record = {}
    ledger = StageLedger(record)
    operator, rhs = ledger.run(
        "system",
        lambda: jax.block_until_ready(_continued_distributed_fe_system(affine_problem)),
    )
    ledger.run(
        "cached-system",
        lambda: jax.block_until_ready(_continued_distributed_fe_system(affine_problem)),
    )
    policy = phx.linalg.LinearSolvePolicy(
        tolerance=phx.linalg.TolerancePolicy(relative=1e-12, absolute=1e-13)
    )
    first, solved = ledger.run(
        "solve",
        lambda: jax.block_until_ready(
            _continued_distributed_fe_solve(affine_problem, operator, rhs, policy)
        ),
    )
    second, repeated = ledger.run(
        "cached-solve",
        lambda: jax.block_until_ready(
            _continued_distributed_fe_solve(affine_problem, operator, rhs, policy)
        ),
    )
    expected = np.asarray(
        [x + y for y in (0.0, 0.5, 1.0) for x in (0.0, 0.5, 1.0)],
        dtype=np.float64,
    )

    np.testing.assert_allclose(first, expected, rtol=0.0, atol=_AFFINE_CONTINUATION_ATOL)
    np.testing.assert_array_equal(second, first)
    assert np.all(np.asarray(solved.successful))
    assert np.all(np.asarray(repeated.successful))
    assert record["stage_measurements"]["system"]["eager_xla_compile_count"] == 1
    assert record["stage_measurements"]["cached-system"]["eager_xla_compile_count"] == 0
    assert record["stage_measurements"]["solve"]["eager_xla_compile_count"] == 1
    assert record["stage_measurements"]["cached-solve"]["eager_xla_compile_count"] == 0


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


def test_peak_resident_bytes_names_platform_units(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Usage:
        ru_maxrss = 123

    monkeypatch.setattr("resource.getrusage", lambda who: Usage())
    monkeypatch.setattr(sys, "platform", "linux")
    assert peak_resident_bytes() == 123 * 1024
    monkeypatch.setattr(sys, "platform", "darwin")
    assert peak_resident_bytes() == 123


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
        "time.sleep(30)"
    )
    with pytest.raises(subprocess.TimeoutExpired):
        run_campaign_process([sys.executable, "-c", parent], cwd=tmp_path, timeout=1.0)
    assert heartbeat.exists()
    stopped = heartbeat.read_text()
    sleep(0.1)
    assert heartbeat.read_text() == stopped


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
        distributed._distributed_qualification_launch(args, record, perf_counter())
    assert record["producer_processes"][0]["failure_phase"] == "owner_local_preparation_0"
    assert record["producer_processes"][0]["failure_type"] == "ValueError"


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
    assert defaults.hybrid_region_densities == (2.0, 3.0)
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
    assert (
        prepared.execution_evidence.preparation_evidence is generated.execution_evidence
    )
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
                budget=budget,
            )


def test_lifecycle_host_worksets_do_not_walk_numerical_model_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_qualification.scenarios import distributed

    original_leaves = jax.tree_util.tree_leaves
    walks: list[object] = []

    def counted_leaves(value: Any, *args: Any, **kwargs: Any) -> Any:
        walks.append(value)
        return original_leaves(value, *args, **kwargs)

    monkeypatch.setattr(jax.tree_util, "tree_leaves", counted_leaves)
    warm: list[float] = []
    continuation = distributed._DistributedFEContinuation(0.0, 3, warm)
    integrals = {"area": 1.0}
    references = {"area": 1.0}
    inventory = distributed._DistributedInventories(integrals, references)
    assert walks == []
    assert continuation.warm is warm
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


def test_quad_hex_qualification_refuses_empty_profile_selection() -> None:
    from tools.meshing_qualification._controls import QualificationControls
    from tools.meshing_qualification.scenarios.native_quad_hex_meshing import run

    controls = QualificationControls(scenario="native-quad-hex", corpus_profile=())
    with pytest.raises(ValueError, match="quad_hex.corpus_profile"):
        run(controls)
