#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Execute selected energy scenarios and produce native qualification/lifecycle evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np

from examples.energy._artifacts import (
    archive_metrics,
    execution_identity,
    identity,
    json_bytes,
)
from examples.energy.building_dispatch import run_building_dispatch
from examples.energy.power_fault import run_power_fault
from phydrax import lifecycle
from phydrax.qualification import QualificationEvidence, QualificationMatrix


SCENARIOS = (
    "building-dispatch",
    "power-fault",
    "building-scientific",
    "energyplus",
    "radiance",
    "fmi",
    "helics",
    "opendss",
)
EXTERNAL = frozenset(("energyplus", "radiance", "fmi", "helics", "opendss"))


def artifact_record(artifact):
    return {
        "artifact_kind": artifact.artifact_kind,
        "content_digest": artifact.content_digest,
        "producer": artifact.producer,
        "producer_version": artifact.producer_version,
        "build_id": artifact.build_id,
        "license_id": artifact.license_id,
        "parent_artifact_ids": artifact.parent_artifact_ids,
        "resource_id": artifact.resource_id,
        "status": artifact.status,
        "failure_reason": artifact.failure_reason,
        "artifact_id": artifact.artifact_id,
    }


def run_fmi_reference(path, sha256, version, license_id):
    """The original accumulator specimen only; not a generic FMU acceptance claim."""
    from phydrax.interchange.fmi import FMICoSimulationSession

    path = Path(path).resolve()
    with FMICoSimulationSession(
        path,
        sha256=sha256,
        trusted_root=path.parent,
        license_id=license_id,
        expected_fmpy_version=version,
        start_values={"u": 3.0, "gain": 2.0, "event_time": 0.5, "stop_at_event": False},
    ) as session:
        if session.model.model_name != "PhydraxEnergyAccumulator":
            raise ValueError(
                "FMI reference scenario requires the authored PhydraxEnergyAccumulator specimen."
            )
        first = session.advance(0.25)
        before = session.get_values(("x", "event_done", "steps"))
        saved = session.save_state()
        event = session.advance(0.75)
        at_event = session.get_values(("x", "event_done", "steps"))
        session.restore_state(saved)
        restored = session.get_values(("x", "event_done", "steps"))
        repeated = session.advance(0.75)
        replayed = session.get_values(("x", "event_done", "steps"))
        final = session.advance(0.75)
        after = session.get_values(("x", "event_done", "steps"))
        session.free_state(saved)
        artifact = artifact_record(session.artifact)
    errors = [abs(before["x"] - 1.5), abs(at_event["x"] + 3.0), abs(after["x"] + 1.5)]
    passed = (
        max(errors) <= 1e-10
        and first.reached_time == 0.25
        and event.reached_time == 0.5
        and event.early_return
        and repeated.early_return
        and repeated.reached_time == 0.5
        and final.reached_time == 0.75
        and not before["event_done"]
        and at_event["event_done"]
        and restored == before
        and replayed == at_event
    )
    return {
        "passed": passed,
        "scope": "FMI2 authored dx/dt=gain*u, internal x->-x time event, native save/restore/replay",
        "before": before,
        "event": at_event,
        "restored": restored,
        "replayed": replayed,
        "after": after,
        "reached_times_s": [
            first.reached_time,
            event.reached_time,
            repeated.reached_time,
            final.reached_time,
        ],
        "maximum_analytic_error": max(errors),
        "absolute_tolerance": 1e-10,
        "FMU_sha256": sha256,
        "artifact": artifact,
    }


def run_helics_reference(version, license_id):
    from phydrax.interchange.helics import HelicsChannel, HelicsValueSession

    channel = HelicsChannel("energy-qualification/power", "double", "W")
    subscription = HelicsChannel("received", "double", "W", target=channel.name)
    with HelicsValueSession(
        "energy-producer",
        publications=(channel,),
        federate_count=2,
        license_id=license_id,
        expected_version=version,
    ) as producer:
        with HelicsValueSession(
            "energy-consumer",
            subscriptions=(subscription,),
            broker=producer.broker_address,
            license_id=license_id,
            expected_version=version,
        ) as consumer:
            producer.enter_execution_async()
            consumer.enter_execution_async()
            producer.complete_execution()
            consumer.complete_execution()
            producer.publish({channel.name: 125.0})
            producer.request_time_async(1.0)
            consumer.request_time_async(1.0)
            sent = producer.complete_time()
            received = consumer.complete_time()
            (sample,) = consumer.read_values()
            producer_artifact = artifact_record(producer.artifact)
            consumer_artifact = artifact_record(consumer.artifact)
    return {
        "passed": sample.has_value
        and sample.value == 125.0
        and 0 < sample.granted_time <= 1.0,
        "scope": "two real HELICS value federates, exact double/W delivery; no iterative convergence or rollback claim",
        "sample": asdict(sample),
        "requested_time_s": 1.0,
        "producer_granted_time_s": sent.granted_time,
        "consumer_granted_time_s": received.granted_time,
        "producer_artifact": producer_artifact,
        "consumer_artifact": consumer_artifact,
    }


def run_opendss_reference(version, license_id):
    from phydrax.interchange.energy_runtime import run_opendss

    commands = (
        "Clear",
        "New Circuit.qualification basekv=12.47 pu=1 phases=3 bus1=source",
        (
            "New Line.feed bus1=source bus2=load phases=3 r1=0.1 x1=0.2 "
            "r0=0.3 x0=0.4 c1=0 c0=0 length=1 units=km"
        ),
        "New Load.demand bus1=load phases=3 conn=wye kv=12.47 kw=100 kvar=25",
        "Set voltagebases=[12.47]",
        "CalcVoltageBases",
        "Solve",
    )
    result = run_opendss(commands, expected_version=version, license_id=license_id)
    # Raw source-terminal inward-positive kW/kvar -> supplied total W/var.
    supply = -1000.0 * np.asarray(result.total_power)
    losses = np.asarray(result.losses)
    residual = supply - np.asarray([100000.0, 25000.0]) - losses
    load_power = next(
        np.asarray(powers)
        for name, _, _, powers in result.element_powers
        if name.lower() == "load.demand"
    )
    demand = 1000.0 * np.sum(load_power, axis=0)
    error = float(np.max(np.abs(residual)))
    return {
        "passed": bool(result.converged)
        and error <= 1.0
        and bool(np.all(losses >= 0))
        and float(np.max(np.abs(demand - [100000.0, 25000.0]))) <= 1.0,
        "scope": "balanced three-phase OpenDSS 100kW/25kvar feeder; aggregate source-minus-load-minus-loss ledger",
        "commands": list(commands),
        "source_supply_W_var": supply.tolist(),
        "loss_W_var": losses.tolist(),
        "measured_load_W_var": demand.tolist(),
        "balance_W_var": residual.tolist(),
        "absolute_tolerance_W_var": 1.0,
        "node_names": list(result.node_names),
        "node_voltages_V": list(result.node_voltages),
        "engine_version": result.engine_version,
        "artifact": artifact_record(result.artifact),
    }


def run_scenario(name, args, execution):
    output = args.output / name
    if name == "building-dispatch":
        result = run_building_dispatch(
            output, epw_path=args.epw, intervals=args.intervals, execution=execution
        )
        return {**result["metrics"], **archive_metrics(result["archives"])}
    if name == "power-fault":
        result = run_power_fault(output, execution=execution)
        return {
            **result["metrics"],
            **archive_metrics(result["archives"]),
            "continuation": {
                **result["continuation_metrics"],
                **archive_metrics(result["continuation_archives"]),
            },
        }
    if name == "building-scientific":
        from tools.building_energy_benchmarks import run_native

        rows = run_native()
        return {
            "passed": all(row["passed"] for row in rows.values()),
            "scope": (
                "authored analytic RC, explicit air/ground boundary, native HVAC, "
                "identifiable calibration and held-out prediction"
            ),
            "rows": rows,
        }
    if name == "energyplus":
        from tools.building_energy_benchmarks import run_energyplus_reference

        row = run_energyplus_reference(
            args.energyplus, args.energyplus_version, args.energyplus_license
        )
        return {
            **row,
            "scope": (
                "authored weather-independent adiabatic 100W ideal-load steady reference; "
                "not transient equivalence"
            ),
            "executable_path": str(args.energyplus.resolve()),
            "declared_version": args.energyplus_version,
            "license_id": args.energyplus_license,
        }
    if name == "radiance":
        from tools.building_energy_benchmarks import run_radiance_reference

        row = run_radiance_reference(
            args.oconv,
            args.rtrace,
            str(args.raypath),
            args.radiance_version,
            args.radiance_license,
        )
        with args.oconv.open("rb") as source:
            oconv_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        resources = sorted(
            path.relative_to(args.raypath)
            for path in args.raypath.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        return {
            **row,
            "scope": "real uniform unit-radiance upper hemisphere, upward irradiance pi per RGB channel",
            "oconv_sha256": oconv_sha256,
            "oconv_path": str(args.oconv.resolve()),
            "rtrace_path": str(args.rtrace.resolve()),
            "raypath": str(args.raypath.resolve()),
            "radiance_resources_sha256": lifecycle.digest_paths(args.raypath, resources),
            "declared_version": args.radiance_version,
            "license_id": args.radiance_license,
        }
    if name == "fmi":
        return run_fmi_reference(
            args.fmu, args.fmu_sha256, args.fmpy_version, args.fmu_license
        )
    if name == "helics":
        return run_helics_reference(args.helics_version, args.helics_license)
    if name == "opendss":
        return run_opendss_reference(args.opendss_version, args.opendss_license)
    raise ValueError(name)


def qualify(args):
    """Return existing native evidence objects plus raw observed metric records."""
    selected = tuple(args.scenario or ("building-dispatch", "power-fault"))
    if len(set(selected)) != len(selected):
        raise ValueError("Each scenario may be selected once.")
    required = {
        "energyplus": ("energyplus", "energyplus_version", "energyplus_license"),
        "radiance": (
            "oconv",
            "rtrace",
            "raypath",
            "radiance_version",
            "radiance_license",
        ),
        "fmi": ("fmu", "fmu_sha256", "fmpy_version", "fmu_license"),
        "helics": ("helics_version", "helics_license"),
        "opendss": ("opendss_version", "opendss_license"),
    }
    for name in selected:
        missing = [key for key in required.get(name, ()) if not vars(args)[key]]
        if missing:
            raise ValueError(
                f"Selected {name} requires explicit "
                + ", ".join("--" + key.replace("_", "-") for key in missing)
            )
    execution = execution_identity()
    args.output.mkdir(parents=True, exist_ok=True)
    issued = int(time.time())
    rows, evidence, predicates = {}, [], {}
    for name in selected:
        started = time.perf_counter()
        try:
            row = run_scenario(name, args, execution)
            # A nonfinite physical metric is an execution failure, not JSON NaN evidence.
            json_bytes(row)
        except Exception as error:
            row = {
                "passed": False,
                "exception": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "scope": "selected scenario failed; no replacement or retry",
            }
        row["elapsed_seconds"] = time.perf_counter() - started
        raw = np.frombuffer(json_bytes(row), dtype=np.uint8)
        raw_id = identity(row)
        raw_manifest = lifecycle.ResultManifest(
            name + ":raw-metrics",
            execution["build_id"],
            (("metrics", "metrics", "UTF-8 JSON bytes"),),
            {"metrics": lifecycle.payload_digest(raw)},
        )
        raw_archive = lifecycle.create(
            args.output / f"{name}.metrics.zip",
            manifest=raw_manifest,
            arrays={"metrics": raw},
        )
        reopened = lifecycle.open(raw_archive.path)
        if not np.array_equal(raw, reopened.arrays["metrics"]):
            raise RuntimeError("Qualification metric archive failed exact reopen.")
        kind = "reference" if name in EXTERNAL else "scientific"
        criterion = name + ":predeclared-physical-acceptance"
        item = QualificationEvidence(
            kind,
            "passed" if row["passed"] else "failed",
            (name,),
            build_id=execution["build_id"],
            environment_id=execution["environment_id"],
            backend="external-host"
            if name in EXTERNAL
            else execution["environment"]["backend"],
            topology=name,
            precision=(
                "external-runtime-not-asserted"
                if name in EXTERNAL
                else "float64"
                if execution["environment"]["jax_enable_x64"]
                else "float32"
            ),
            reduction=row["scope"],
            replay_id=row.get("result_archive_id", raw_id),
            criteria_ids=(criterion,),
            raw_artifact_ids=(raw_archive.archive_id,),
            reviewer_id=args.reviewer,
            issued_at=issued,
            expires_at=issued + args.valid_for_seconds,
            reason="Observed scenario acceptance passed."
            if row["passed"]
            else "Observed execution/physical acceptance failed; inspect raw metrics.",
            requalification_triggers=(
                "source-change",
                "runtime-or-environment-change",
                "model-or-input-change",
            ),
        )
        predicates[name] = {
            "evidence_kind": kind,
            "subject_id": name,
            "build_id": execution["build_id"],
            "environment_id": execution["environment_id"],
            "criterion_id": criterion,
            "raw_artifact_id": raw_archive.archive_id,
        }
        evidence.append(item)
        rows[name] = {
            **row,
            "raw_archive_id": raw_archive.archive_id,
            "raw_archive_path": str(raw_archive.path),
        }
    matrix = QualificationMatrix(predicates)
    coverage = matrix.evaluate(tuple(evidence), at_time=issued)
    record = {
        "execution": execution,
        "selected_scenarios": list(selected),
        "raw_metrics": rows,
        "evidence": [item.to_record() for item in evidence],
        "matrix": matrix.to_record(),
        "coverage": coverage.to_record(),
    }
    payload = np.frombuffer(json_bytes(record), dtype=np.uint8)
    manifest = lifecycle.ResultManifest(
        "energy-qualification",
        identity(record),
        (("qualification", "qualification", "UTF-8 JSON bytes"),),
        {"qualification": lifecycle.payload_digest(payload)},
        evidence_ids=tuple(item.evidence_id for item in evidence),
    )
    archive = lifecycle.create(
        args.output / "qualification.zip",
        manifest=manifest,
        arrays={"qualification": payload},
    )
    lifecycle.open(archive.path)
    (args.output / "qualification.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    return {
        "record": record,
        "evidence": tuple(evidence),
        "matrix": matrix,
        "coverage": coverage,
        "archive": archive,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument(
        "--output", type=Path, default=Path("energy-results/qualification")
    )
    parser.add_argument("--reviewer", default="automated:energy-qualification")
    parser.add_argument("--valid-for-seconds", type=int, default=86400)
    parser.add_argument("--epw", type=Path)
    parser.add_argument("--intervals", type=int, choices=(2, 3, 4), default=4)
    for name in ("energyplus", "oconv", "rtrace", "raypath", "fmu"):
        parser.add_argument("--" + name, type=Path)
    for name in (
        "energyplus-version",
        "energyplus-license",
        "radiance-version",
        "radiance-license",
        "fmu-sha256",
        "fmpy-version",
        "fmu-license",
        "helics-version",
        "helics-license",
        "opendss-version",
        "opendss-license",
    ):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.valid_for_seconds <= 0:
        parser.error("--valid-for-seconds must be positive")
    result = qualify(args)
    print(json.dumps(result["record"], indent=2, allow_nan=False))
    if result["coverage"].outcome != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
