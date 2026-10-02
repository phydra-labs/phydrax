# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Isolated float64 Taylor contraction/gradient qualification.

Run with ``python -m tools.taylor_gradient_diagnostic --case all --deadline 120
--repeats 3 --output /tmp/taylor-gradient.json``. Each of twelve independent
case/phase children has a wall-clock deadline; each case is bounded by three
such deadlines. JSONL stage journals survive refusal, crash, and timeout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any


_CASES = ("pure", "mixed-linear", "mixed-single", "mixed-prime")
_PHASES = ("forward", "parameter", "direction")


def _emit(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
    print(line, flush=True)


def _child(case: str, phase: str, repeats: int, journal: Path) -> None:
    # Heavy imports intentionally occur only inside bounded children.
    base: dict[str, Any] = {"case": case, "phase": phase}

    def stage(name: str, **evidence: Any) -> None:
        _emit(journal, base | {"stage": name, **evidence})

    stage("imports")
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import Array

    from benchmarks._runtime import (
        capture_environment,
        compiler_evidence,
        logical_array_bytes,
        measure_host,
        measure_repeated,
        measure_synchronized,
    )
    from phydrax.operators.differential import (
        evaluate_taylor_contractions,
        plan_taylor_contractions,
        TaylorContractionPolicy,
        TaylorContractionRequest,
        TaylorContractionResources,
        TaylorContractionStrategy,
    )

    jax.config.update("jax_enable_x64", True)
    counts = (8,) if case == "pure" else (4, 4)
    strategy: TaylorContractionStrategy = "linear"
    if case == "mixed-single":
        strategy = "single"
    elif case == "mixed-prime":
        strategy = "prime"
    stage("planning")
    resources = TaylorContractionResources(workset_size=1)
    plan, planning_seconds = measure_host(
        lambda: plan_taylor_contractions(
            (
                TaylorContractionRequest(
                    tuple(f"d{i}" for i in range(len(counts))), counts
                ),
            ),
            policy=TaylorContractionPolicy(strategy=strategy, resources=resources),
        )
    )
    plan_evidence = {
        "plan_id": plan.plan_id,
        "requested_multiplicities": counts,
        "actual_strategies": [recipe.strategy for recipe in plan.recipes],
        "required_regularity_order": plan.required_regularity_order,
        "schedules": [
            {"coefficients": s.coefficients, "order": s.order} for s in plan.schedules
        ],
        "resources": {
            name: getattr(resources, name)
            for name in (
                "max_order",
                "max_certificate_states",
                "max_linear_terms",
                "max_candidates",
                "workset_size",
                "max_logical_buffer_elements",
            )
        },
        "planning_seconds": planning_seconds,
    }
    stage("inputs", plan=plan_evidence)
    theta = jnp.asarray(1.3, dtype=jnp.float64)
    point = jnp.asarray([0.1, -0.2], dtype=jnp.float64)
    coefficients = jnp.asarray([0.7, -0.2], dtype=jnp.float64)
    directions = jnp.asarray([[0.5, 0.3], [-0.4, 0.6]][: len(counts)], dtype=jnp.float64)

    def field(weight: Array, position: Array) -> Array:
        return weight * jnp.exp(jnp.dot(coefficients, position))

    def contraction(weight: Array, tangents: Array) -> Array:
        def bound(position: Array) -> Array:
            return field(weight, position)

        return evaluate_taylor_contractions(
            bound,
            (point,),
            {
                identifier: (tangents[i],)
                for i, identifier in enumerate(plan.direction_ids)
            },
            plan,
        ).values[0]

    operation = (
        contraction
        if phase == "forward"
        else jax.grad(contraction, argnums=0 if phase == "parameter" else 1)
    )
    arguments = (theta, directions)
    stage("binding")
    jitted, binding_seconds = measure_host(lambda: jax.jit(operation))
    stage("lowering", binding_seconds=binding_seconds)
    lowered, lowering_seconds = measure_host(lambda: jitted.lower(*arguments))
    stage("compilation", lowering_seconds=lowering_seconds)
    compiled, compilation_seconds = measure_host(lowered.compile)
    stage("first_execution", compilation_seconds=compilation_seconds)
    actual, first_seconds = measure_synchronized(lambda: compiled(*arguments))
    stage("warm_execution", first_synchronized_execution_seconds=first_seconds)
    _, warmed = measure_repeated(lambda: compiled(*arguments), warmup=1, repeats=repeats)
    stage("analytic_check", warmed=warmed.to_seconds_dict())
    # Independent host formula uses explicit products, never divisions by projections.
    a = np.asarray([0.7, -0.2], dtype=np.float64)
    x = np.asarray([0.1, -0.2], dtype=np.float64)
    vectors = np.asarray([[0.5, 0.3], [-0.4, 0.6]][: len(counts)], dtype=np.float64)
    projections = vectors @ a
    exponential = math.exp(float(a @ x))
    product = math.prod(float(p) ** m for p, m in zip(projections, counts, strict=True))
    forward_reference = 1.3 * exponential * product
    parameter_reference = exponential * product
    direction_reference = np.stack(
        [
            1.3
            * exponential
            * m
            * math.prod(
                float(p) ** (n - (i == j))
                for j, (p, n) in enumerate(zip(projections, counts, strict=True))
            )
            * a
            for i, m in enumerate(counts)
        ]
    )
    reference = {
        "forward": forward_reference,
        "parameter": parameter_reference,
        "direction": direction_reference,
    }[phase]
    host_actual = np.asarray(jax.device_get(actual))
    error = float(np.max(np.abs(host_actual - reference)))
    passed = bool(
        np.all(np.isfinite(host_actual))
        and np.allclose(host_actual, reference, rtol=1e-7, atol=1e-12)
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The active backend does not report every compiler estimate.",
    )
    working = (
        min(resources.workset_size, len(plan.schedules))
        * (plan.required_regularity_order + 1)
        * 3
        * 8
    )
    stage(
        "complete",
        status="passed" if passed else "analytic_failure",
        plan=plan_evidence,
        dtype=str(actual.dtype),
        actual=host_actual.tolist(),
        reference=np.asarray(reference).tolist(),
        maximum_absolute_error=error,
        relative_tolerance=1e-7,
        absolute_tolerance=1e-12,
        lowering_seconds=lowering_seconds,
        compilation_seconds=compilation_seconds,
        first_synchronized_execution_seconds=first_seconds,
        warmed=warmed.to_seconds_dict(),
        compiler=asdict(compiler),
        environment=capture_environment().to_dict(),
        logical_retained_bytes=logical_array_bytes(
            (arguments, point, coefficients, actual)
        ),
        logical_retained_scope="Dynamic inputs, captured field arrays and result; excludes host plan metadata.",
        logical_working_bytes_estimate=working,
        logical_working_scope="Curve input/output series only; excludes reverse-mode tape, field intermediates and compiler temporaries.",
    )


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all", *_CASES), default="all")
    parser.add_argument(
        "--deadline",
        type=float,
        default=120.0,
        help="Wall-clock seconds per isolated phase, including startup; case budget is three times this.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--child-phase", choices=_PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.deadline) or args.deadline <= 0 or args.repeats < 1:
        parser.error("deadline must be positive and finite; repeats must be positive")
    if args.child_phase:
        if args.case == "all":
            parser.error("a child requires one concrete case")
        try:
            _child(args.case, args.child_phase, args.repeats, args.output)
        except Exception as exc:
            _emit(
                args.output,
                {
                    "case": args.case,
                    "phase": args.child_phase,
                    "stage": "failed",
                    "status": "refused_or_failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
            raise
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "deadline_seconds_per_phase": args.deadline,
        "deadline_seconds_per_case": 3 * args.deadline,
        "repeats": args.repeats,
        "field": "theta * exp(a @ x)",
        "theta": 1.3,
        "x": [0.1, -0.2],
        "a": [0.7, -0.2],
        "directions": [[0.5, 0.3], [-0.4, 0.6]],
        "records": [],
    }
    root = Path(__file__).resolve().parents[1]
    for case in _CASES if args.case == "all" else (args.case,):
        for phase in _PHASES:
            journal = args.output.with_name(f"{args.output.name}.{case}.{phase}.jsonl")
            journal.write_text("", encoding="utf-8")
            started = time.monotonic()
            command = [
                sys.executable,
                "-m",
                "tools.taylor_gradient_diagnostic",
                "--case",
                case,
                "--child-phase",
                phase,
                "--repeats",
                str(args.repeats),
                "--output",
                str(journal.resolve()),
            ]
            timed_out = False
            with subprocess.Popen(command, cwd=root, start_new_session=True) as process:
                try:
                    process.wait(timeout=args.deadline)
                except subprocess.TimeoutExpired:
                    timed_out = True
                finally:
                    # Same process-group ownership pattern as tools._pytest_outcomes;
                    # its public runner is pytest-specific and cannot run this product.
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            events = _records(journal)
            last = events[-1] if events else {}
            record = {
                "case": case,
                "phase": phase,
                "elapsed_seconds": time.monotonic() - started,
                "journal": str(journal),
                "events": events,
                "returncode": process.returncode,
                "status": "timeout" if timed_out else last.get("status", "crashed"),
                "timeout_stage": last.get("stage", "startup") if timed_out else None,
            }
            evidence["records"].append(record)
            evidence["qualified"] = len(evidence["records"]) == (
                12 if args.case == "all" else 3
            ) and all(item["status"] == "passed" for item in evidence["records"])
            # Persist completed and partial evidence before proceeding to another child.
            args.output.write_text(
                json.dumps(evidence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )
    if not evidence["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
