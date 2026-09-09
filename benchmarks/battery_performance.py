#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Governed absolute battery workload measurements and identity-exact regressions."""

from __future__ import annotations

import argparse
import hashlib
import math
import statistics
import sys
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax
from jax.extend.core import ClosedJaxpr, Jaxpr

from benchmarks._comparison import (
    compare_performance as compare_distributions,
    PerformancePolicy,
)
from benchmarks._runtime import (
    compiler_evidence,
    DurationDistribution,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_synchronized,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax.qualification import (
    CampaignObservationRecord,
    CampaignStartRecord,
    QualificationEvidence,
    validate_qualification_causality,
)
from tools.battery_campaign_registry import metric_key
from tools.battery_campaign_resources import (
    BatteryResourcePlan,
    classify_resource,
    RESOURCE_METRICS,
)
from tools.battery_qualification import (
    BatteryCampaignSpec,
    campaign_attempt_record,
    campaign_input_identity,
    capture_runtime_identity,
    execution_source_build_id,
    load_campaign_spec,
    prepare_builtin_campaign,
    read_json_object,
    verify_campaign_harness,
    verify_campaign_preflight,
    write_json_immutable,
)


def _positive_count(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _percentile(values: Sequence[float], percentile: float, /) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered or any(not math.isfinite(value) or value < 0.0 for value in ordered):
        raise ValueError("Timing samples must be finite, nonnegative, and nonempty.")
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def performance_harness_source_id(root: Path, /) -> str:
    """Fingerprint the exact performance runner and shared timing runtime."""
    if not isinstance(root, Path):
        raise TypeError("Performance harness source root must be a pathlib.Path.")
    source_root = root.resolve()
    paths = (
        source_root / "benchmarks/battery_performance.py",
        source_root / "benchmarks/_runtime.py",
        source_root / "benchmarks/_comparison.py",
        source_root / "tools/battery_qualification.py",
        source_root / "tools/battery_campaign_registry.py",
        source_root / "tools/battery_campaign_resources.py",
        *sorted((source_root / "tools").glob("_battery_*.py")),
    )
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError(
            "Performance harness requires battery_performance.py and _runtime.py."
        )
    sources: list[dict[str, object]] = []
    for path in sorted(paths, key=lambda value: value.as_posix()):
        payload = path.read_bytes()
        sources.append(
            {
                "path": path.relative_to(source_root).as_posix(),
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return canonical_fingerprint(
        {
            "kind": "battery-performance-harness-source",
            "sources": sources,
        }
    )


def performance_workload_id(
    spec: BatteryCampaignSpec,
    sample_count: int,
    harness_source_id: str,
    /,
) -> str:
    """Bind operation, measurement plan, and exact harness source."""
    count = _positive_count(sample_count, "sample_count")
    if (
        type(harness_source_id) is not str
        or not harness_source_id
        or harness_source_id != harness_source_id.strip()
    ):
        raise ValueError("harness_source_id must be a canonical identifier.")
    return canonical_fingerprint(
        {
            "kind": "battery-performance-workload-measurement",
            "campaign_workload_id": spec.workload_id(),
            "harness_source_id": harness_source_id,
            "sample_count": count,
            "synchronization": "all-jax-output-leaves",
            "statistics": ["median", "linear-interpolated-p95"],
        }
    )


def _compiler_resource_record(compiled) -> dict[str, object]:
    unavailable_reason = None
    try:
        cost = compiled.cost_analysis()
        memory = compiled.memory_analysis()
    except (NotImplementedError, RuntimeError) as error:
        cost = None
        memory = None
        unavailable_reason = f"{type(error).__name__}:{error}"
    evidence = compiler_evidence(
        cost,
        memory,
        source="jax-compiled-executable",
        unavailable_reason=unavailable_reason,
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
        "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
        "source": evidence.source,
        "unavailable_reason": evidence.unavailable_reason,
    }


def _dense_array_count(closed: ClosedJaxpr, /) -> int:
    """Inspect all nested compiled computation graphs, not a source-text assertion."""
    seen: set[int] = set()

    def visit(value: object) -> int:
        if isinstance(value, ClosedJaxpr):
            return visit(value.jaxpr)
        if isinstance(value, Jaxpr):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            count = 0
            for variable in (*value.constvars, *value.invars):
                if isinstance(variable.aval, jax.core.ShapedArray):
                    shape = variable.aval.shape
                    count += int(
                        len(shape) >= 2 and shape[-1] == shape[-2] and shape[-1] > 64
                    )
            for equation in value.eqns:
                for variable in equation.outvars:
                    if isinstance(variable.aval, jax.core.ShapedArray):
                        shape = variable.aval.shape
                        count += int(
                            len(shape) >= 2 and shape[-1] == shape[-2] and shape[-1] > 64
                        )
                count += sum(visit(parameter) for parameter in equation.params.values())
            return count
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        if isinstance(value, Mapping):
            return sum(visit(item) for item in value.values())
        return 0

    return visit(closed)


def run_performance(
    spec: BatteryCampaignSpec,
    /,
    *,
    campaign_directory: Path,
    sample_count: int,
    resource_plan: BatteryResourcePlan,
    source_root: Path | None = None,
    distribution_manifest: Path | None = None,
) -> dict[str, object]:
    """Persist absolute outcomes; unavailable counters or bytes are inconclusive."""
    destination = campaign_directory / "qualification-artifacts"
    harness_root = Path(__file__).resolve().parents[1]
    execution_root = harness_root if source_root is None else source_root
    try:
        count = _positive_count(sample_count, "sample_count")
        BatteryResourcePlan.from_record(resource_plan.to_record())
        if (
            count != resource_plan.sample_count
            or resource_plan.workload_id != spec.workload_id()
        ):
            raise ValueError(
                "Resource plan does not bind this exact workload and measurement count."
            )
        verify_campaign_preflight(
            spec,
            campaign_directory,
            source_root=execution_root,
            distribution_manifest=distribution_manifest,
        )
        harness_source_id = performance_harness_source_id(harness_root)
        runtime = capture_runtime_identity(
            source_root=execution_root, distribution_manifest=distribution_manifest
        )
        started_at = time.time_ns()
        if (
            not spec.planned_schedule.not_before
            <= started_at
            < spec.planned_schedule.deadline
        ):
            raise ValueError("Resource campaign start lies outside the planned schedule.")
        for criterion in resource_plan.criteria:
            if (
                criterion.support_tuple_id != spec.candidate_support.support_tuple_id
                or not criterion.issued_at < started_at
                or not criterion.is_valid(started_at)
            ):
                raise ValueError(
                    "Resource criteria must be preapproved, current and model-exact."
                )
    except Exception as error:
        refusal = campaign_attempt_record(
            error,
            campaign_spec_id=spec.campaign_spec_id,
            resource_plan_id=resource_plan.plan_id,
        )
        write_json_immutable(destination / f"{refusal['attempt_id']}.json", refusal)
        return refusal

    campaign_id = canonical_fingerprint(
        {
            "campaign_spec_id": spec.campaign_spec_id,
            "resource_plan_id": resource_plan.plan_id,
        }
    )
    starts = tuple(
        CampaignStartRecord(
            campaign_spec_id=campaign_id,
            criterion_id=criterion.criterion_id,
            resolved_run_spec_id=spec.resolved_run_spec.spec_id,
            support_tuple_id=spec.candidate_support.support_tuple_id,
            started_at=started_at,
        )
        for criterion in resource_plan.criteria
    )
    for start in starts:
        write_json_immutable(
            destination / f"{start.start_record_id}.json", start.to_record()
        )

    values: dict[str, float | int | None] = {
        name: None for name, _, _, _ in RESOURCE_METRICS
    }
    reasons = {name: "native-or-runtime-measurement-not-exposed" for name in values}
    samples: list[float] = []
    compiler = None
    failure = None
    try:
        if tracemalloc.is_tracing():
            raise RuntimeError(
                "Preparation tracing requires an unowned tracemalloc session."
            )
        tracemalloc.start()
        try:
            builtin, values["prepare-seconds"] = measure_host(
                lambda: prepare_builtin_campaign(
                    spec.campaign_kind, spec.planned_schedule.sample_times_s
                )
            )
            _, values["host-python-peak-bytes"] = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        jitted = jax.jit(builtin.operation)
        compiled, compilation = measure_lower_and_compile(
            jitted.lower, lambda lowered: lowered.compile()
        )
        values["lower-seconds"] = compilation.lowering_seconds
        values["compile-seconds"] = compilation.compilation_seconds
        first_output, values["first-seconds"] = measure_synchronized(compiled)
        values["output-bytes"] = logical_array_bytes(first_output)
        if builtin.resources is not None:
            native = builtin.resources(first_output)
            if set(native) - set(values):
                raise ValueError("Native resource observer emitted undeclared metrics.")
            values.update(native)
        for _ in range(count):
            output, elapsed = measure_synchronized(compiled)
            samples.append(elapsed)
            values["output-bytes"] = max(
                values["output-bytes"], logical_array_bytes(output)
            )
            if builtin.resources is not None:
                for name, value in builtin.resources(output).items():
                    if value is not None:
                        values[name] = (
                            value if values[name] is None else max(values[name], value)
                        )
        values["warm-p95-seconds"] = _percentile(samples, 0.95)
        compiler = _compiler_resource_record(compiled)
        values["device-executable-bytes"] = compiler["estimated_device_memory_bytes"]
        values["scratch-bytes"] = compiler["temporary_bytes"]
        values["global-dense-arrays"] = _dense_array_count(
            jax.make_jaxpr(builtin.operation)()
        )
        if sys.platform in ("darwin", "linux"):
            import resource

            scale = 1 if sys.platform == "darwin" else 1024
            values["host-process-peak-bytes"] = (
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * scale
            )
        if performance_harness_source_id(harness_root) != harness_source_id:
            raise RuntimeError("Performance harness changed during measurement.")
        if execution_source_build_id(execution_root) != spec.build_id:
            raise RuntimeError("Execution source closure changed during measurement.")
        verify_campaign_harness(execution_root)
        if (
            capture_runtime_identity(
                source_root=execution_root, distribution_manifest=distribution_manifest
            ).environment_id
            != spec.environment_id
        ):
            raise RuntimeError("Imported package changed during performance execution.")
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}

    observed_at = time.time_ns()
    measurements = {
        metric_key(resource_plan.case_id, name): {
            "value": value,
            "unit": unit,
            "aggregation": aggregation,
            "unavailable_reason": reasons[name] if value is None else None,
        }
        for name, unit, aggregation, _ in RESOURCE_METRICS
        for value in (values[name],)
    }
    content = {
        "kind": "battery-resource-performance-record",
        "campaign_spec_id": spec.campaign_spec_id,
        "campaign_workload_id": spec.workload_id(),
        "workload_id": performance_workload_id(spec, count, harness_source_id),
        "harness_source_id": harness_source_id,
        "build_id": spec.build_id,
        "environment_id": runtime.environment_id,
        "backend_id": runtime.backend_id,
        "device_id": runtime.device_id,
        "precision_id": runtime.precision_id,
        "topology_id": runtime.topology_id,
        "discretization_id": spec.discretization_id,
        "parameter_id": spec.parameter_id,
        "model_selection_id": spec.model_selection_id,
        "runtime_identity": runtime.to_record(),
        "resource_plan": resource_plan.to_record(),
        "timing": {
            "host_preparation_seconds": values["prepare-seconds"],
            "lowering_seconds": values["lower-seconds"],
            "compilation_seconds": values["compile-seconds"],
            "first_call_seconds": values["first-seconds"],
            "warm_samples_seconds": samples,
            "warm_median_seconds": statistics.median(samples) if samples else None,
            "warm_p95_seconds": values["warm-p95-seconds"],
        },
        "metrics": measurements,
        "compiler_resources": compiler,
        "infrastructure_failures": [] if failure is None else [failure],
        "evidence_scope": "resource-only",
    }
    raw_id = canonical_fingerprint(content)
    raw = {**content, "raw_artifact_id": raw_id}
    write_json_immutable(destination / f"{raw_id}.json", raw)
    observations, evidence_records = [], []
    issued_at = time.time_ns()
    for criterion, start in zip(resource_plan.criteria, starts, strict=True):
        observation = CampaignObservationRecord(
            start_record_id=start.start_record_id,
            campaign_spec_id=campaign_id,
            criterion_id=criterion.criterion_id,
            resolved_run_spec_id=spec.resolved_run_spec.spec_id,
            support_tuple_id=spec.candidate_support.support_tuple_id,
            raw_artifact_ids=(raw_id,),
            observed_at=observed_at,
        )
        outcome, reason = classify_resource(criterion, values[criterion.metric])
        if failure is not None:
            outcome, reason = (
                "inconclusive",
                "post-start-performance-infrastructure-failure",
            )
        if (
            not criterion.is_valid(issued_at)
            or issued_at >= spec.planned_schedule.deadline
            or issued_at >= spec.resolved_run_spec.valid_until
        ):
            outcome, reason = "inconclusive", "resource-execution-window-expired"
        expiry = issued_at + spec.planned_schedule.evidence_validity_duration
        if outcome == "passed":
            expiry = min(
                expiry,
                spec.resolved_run_spec.valid_until,
                criterion.valid_until if criterion.valid_until is not None else expiry,
            )
        evidence = QualificationEvidence(
            "performance",
            outcome,
            (spec.candidate_profile.profile_id, spec.candidate_support.support_tuple_id),
            build_id=spec.build_id,
            environment_id=spec.environment_id,
            backend=spec.backend_id,
            topology=spec.topology_id,
            precision=spec.precision_id,
            reduction=criterion.aggregation,
            replay_id=spec.replay_id,
            criteria_ids=(criterion.criterion_id,),
            raw_artifact_ids=(raw_id,),
            campaign_start_record_ids=(start.start_record_id,),
            campaign_observation_record_ids=(observation.observation_record_id,),
            reviewer_id=spec.reviewer_id,
            issued_at=issued_at,
            expires_at=expiry,
            reason=reason,
        )
        validate_qualification_causality(criterion, start, observation, evidence)
        observations.append(observation.to_record())
        evidence_records.append(evidence.to_record())
        write_json_immutable(
            destination / f"{observation.observation_record_id}.json",
            observation.to_record(),
        )
        write_json_immutable(
            destination / f"{evidence.evidence_id}.json", evidence.to_record()
        )
    outcomes = {record["outcome"] for record in evidence_records}
    result = {
        "kind": "battery-absolute-performance-result",
        "performance": raw,
        "outcome": "failed"
        if "failed" in outcomes
        else "inconclusive"
        if "inconclusive" in outcomes
        else "passed",
        "campaign_start_records": [start.to_record() for start in starts],
        "campaign_observation_records": observations,
        "evidence_records": evidence_records,
    }
    write_json_immutable(destination / f"{canonical_fingerprint(result)}.json", result)
    return result


def _verified_performance_record(value: object, name: str, /) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    required = {
        "kind",
        "environment_id",
        "workload_id",
        "harness_source_id",
        "raw_artifact_id",
        "timing",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"{name} is missing fields: {', '.join(missing)}.")
    if value["kind"] != "battery-resource-performance-record":
        raise ValueError(f"{name} has an unsupported kind.")
    recorded_id = value["raw_artifact_id"]
    if type(recorded_id) is not str or not recorded_id:
        raise ValueError(f"{name} raw_artifact_id must be non-empty.")
    content = dict(value)
    del content["raw_artifact_id"]
    if canonical_fingerprint(content) != recorded_id:
        raise ValueError(f"{name} has an invalid content address.")
    return value


def compare_performance(
    current: Mapping[str, object], baseline: Mapping[str, object], /
) -> dict[str, object]:
    """Only same-workload/environment/harness raw samples enter shared regression statistics."""
    current_ = _verified_performance_record(current, "Current performance record")
    baseline_ = _verified_performance_record(baseline, "Baseline performance record")
    for name in ("harness_source_id", "environment_id", "workload_id"):
        if current_[name] != baseline_[name]:
            raise ValueError(f"Performance comparison requires matching {name}.")
    identity = canonical_fingerprint(
        {"current": current_["raw_artifact_id"], "baseline": baseline_["raw_artifact_id"]}
    )
    current_samples = DurationDistribution(
        tuple(current_["timing"]["warm_samples_seconds"])
    )
    baseline_samples = DurationDistribution(
        tuple(baseline_["timing"]["warm_samples_seconds"])
    )
    if not current_samples.count or not baseline_samples.count:
        statistics_record = {
            "regressed": None,
            "reason": "required timing samples unavailable",
        }
    else:
        statistics_record = compare_distributions(
            baseline_samples,
            current_samples,
            PerformancePolicy(
                "minimize", relative_tolerance=0.10, absolute_tolerance=0.001
            ),
            comparison_id=identity,
        ).to_dict()
    content = {
        "kind": "battery-performance-comparison",
        "environment_id": current_["environment_id"],
        "workload_id": current_["workload_id"],
        "baseline_record_id": baseline_["raw_artifact_id"],
        "current_record_id": current_["raw_artifact_id"],
        "statistics": statistics_record,
    }
    return {**content, "comparison_id": canonical_fingerprint(content)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure an ex-ante absolute battery resource campaign."
    )
    parser.add_argument("--campaign-spec", required=True, type=Path)
    parser.add_argument("--resource-plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--distribution-manifest", type=Path)
    parser.add_argument("--compare", type=Path)
    arguments = parser.parse_args(argv)
    try:
        spec = load_campaign_spec(arguments.campaign_spec)
        plan = BatteryResourcePlan.from_record(read_json_object(arguments.resource_plan))
    except Exception as error:
        result = campaign_attempt_record(
            error,
            campaign_spec=campaign_input_identity(arguments.campaign_spec),
            resource_plan=campaign_input_identity(arguments.resource_plan),
        )
    else:
        result = run_performance(
            spec,
            campaign_directory=arguments.campaign_spec.parent,
            sample_count=plan.sample_count,
            resource_plan=plan,
            source_root=arguments.source_root,
            distribution_manifest=arguments.distribution_manifest,
        )
    write_json_immutable(arguments.output, result)
    # The campaign outcome is already durable even if baseline comparison refuses.
    if arguments.compare is not None and "performance" in result:
        baseline = read_json_object(arguments.compare)
        comparison = compare_performance(
            result["performance"], baseline.get("performance", baseline)
        )
        write_json_immutable(arguments.output.with_suffix(".comparison.json"), comparison)
    return {"passed": 0, "failed": 1, "preflight-refused": 2, "inconclusive": 3}[
        result["outcome"]
    ]


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "compare_performance",
    "performance_harness_source_id",
    "performance_workload_id",
    "run_performance",
    "main",
]
