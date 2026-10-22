#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Publish and independently continue the native design lifecycle's accepted state.

``produce archive --design-case CASE`` retains the registered case, all accepted
fields and its uninterrupted continued solution. ``consume archive content_id
produce.json`` binds that exact producer record and its authored work/wall limits.
``qualify archive produce.json --design-case CASE`` runs both phases under one
inherited whole-workflow clock; separate successful commands do not establish it.
Routine defaults remain 320000 work/120 seconds; larger and combined defaults are
480000 work/180 seconds. Explicit positive finite controls bind both interpreters.
Cold raw-control bootstrap opens one fresh native request before reading source
owners; restoration, reproof and continued FE consumers share its budget.
Deadline refusals retain original worker diagnostics without extending that clock.
Archive limits remain at their 257-member/16-level defaults. No RNG or
constitutive state is invented, and the accepted solution is never reseeded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import array_tree_fingerprint, ArrayArchiveLimits
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.lifecycle import (
    admit_meshing_source_model,
    read_meshing_source_closure,
    read_meshing_source_execution_controls,
    write_meshing_source_closure,
)
from phydrax.meshing import NativeDesignSourceState
from phydrax.meshing.qualification._design_oracle import (
    design_qualification_cases,
    DESIGN_QUALIFICATION_DEFAULT_CASE,
)
from phydrax.meshing.qualification._native_lifecycle import execute_native_request
from tools.meshing_qualification._controls import (
    authored_resource_envelope,
    DesignCase,
    qualification_timeout,
    QualificationControls,
)
from tools.meshing_qualification._physics import _compiled_solver_stages
from tools.meshing_qualification._records import _result_identity
from tools.meshing_qualification._runtime import (
    CampaignProcessTimeout,
    Deadline,
    qualification_failure,
    run_campaign_process,
)
from tools.meshing_qualification.scenarios.curved_native_transfer import LinearMarker
from tools.meshing_qualification.scenarios.design import (
    _design_qualification_array,
    _design_qualification_case,
    _design_qualification_derivatives,
    _design_qualification_lifecycle,
    _design_qualification_physical,
    _design_qualification_prepare,
    _design_qualification_source_acceptance,
    _design_qualification_transaction,
    _DesignRetainedSource,
)


# Archive identity of the learned marker model retained by the design source.
_MARKER_IDENTITY = (
    "tools.meshing_qualification.scenarios.curved_native_transfer:LinearMarker"
)


def original_controls(
    design_case: DesignCase = DESIGN_QUALIFICATION_DEFAULT_CASE,
    *,
    maximum_work_units: int | None = None,
    timeout: float | None = None,
) -> QualificationControls:
    """Resolve the registered defaults or explicit authored whole-request limits."""
    if design_case not in design_qualification_cases():
        raise ValueError(f"design_case must name a registered design case; got {design_case!r}.")
    defaults: dict[str, object] = {"maximum_work_units": 320_000, "timeout": 120.0}
    if design_case != "routine-sphere":
        envelope = authored_resource_envelope(
            "native-design-lifecycle", design_case, {}, larger=True
        )
        if envelope is None:
            raise RuntimeError("Registered design cases must have an authored envelope.")
        defaults = envelope["effective"]
    work = defaults["maximum_work_units"] if maximum_work_units is None else maximum_work_units
    if type(work) is not int or work <= 0:
        raise ValueError(f"maximum_work_units must be a positive integer; got {work!r}.")
    wall = defaults["timeout"] if timeout is None else timeout
    return QualificationControls(
        design_case=design_case,
        resolution=4,
        capacity=20000,
        timeout=qualification_timeout({"timeout": wall}, 120.0),
        maximum_work_units=work,
        repeats=3,
        target_error=0.05,
        adaptation_rounds=1,
        maximum_memory_bytes=8 * 1024**3,
        maximum_condition=1.0e12,
    )


def _identity(state: NativeDesignSourceState) -> dict[str, object]:
    return {
        "source_revision": state.source.source_revision,
        "specification_id": state.specification.specification_id,
        "limits_id": state.specification.limits.limits_id,
        "options_id": state.options.options_id,
        "plan_id": state.plan.plan_id,
        "initial_result_id": state.initial.result_id,
        "accepted_result_id": state.accepted.result_id,
        "accepted_state_id": state.accepted_state.accepted_id,
        "decision_id": state.decision.decision_id,
        "selected_candidate_id": state.adaptation.result_id,
        "learned_model_id": state.learned_proposer.model_id,
        "learned_feature_id": state.learned_features.feature_id,
        "learned_proposal_id": state.learned_transaction.projection.proposal.proposal_id,
        "derivative_evidence_id": state.derivative.evidence_id,
        "field_fingerprint": array_tree_fingerprint(state.accepted_state.fields),
        "source_fingerprint": array_tree_fingerprint(
            (state.geometry, state.source, state.plan.prepared)
        ),
        "source_field_id": state.source_fields.prepared_id,
        "target_field_id": state.target_fields.prepared_id,
    }


def produce(
    path: Path, *, design_case: DesignCase = DESIGN_QUALIFICATION_DEFAULT_CASE,
    maximum_work_units: int | None = None, timeout: float | None = None,
) -> dict[str, object]:
    """Execute the case once with its newly retained scientific owners.

    The record carries the exact uninterrupted continued solve of the accepted
    state; ``consume`` must reproduce it from the restored archive.
    """
    started = perf_counter()
    admit_meshing_source_model(_MARKER_IDENTITY, LinearMarker)
    controls = original_controls(
        design_case, maximum_work_units=maximum_work_units, timeout=timeout
    )
    deadline = Deadline(started + controls.timeout)
    retained: list[_DesignRetainedSource] = []
    campaign = _design_qualification_lifecycle(
        controls, retained_source_states=retained
    )
    if campaign["status"] != "passed" or len(retained) != 1:
        raise RuntimeError(
            json.dumps({"original_campaign": campaign, "retained_states": len(retained)})
        )
    deadline.remaining()
    state, continued = retained[0].state, retained[0].uninterrupted_continuation
    limits = ArrayArchiveLimits()
    receipt = write_meshing_source_closure(path, state, limits=limits)
    identity = _identity(state)
    continuation = {
        "fingerprint": array_tree_fingerprint(continued),
        "values": np.asarray(continued, dtype=np.float64).tolist(),
        "change_from_accepted": float(
            jnp.max(jnp.abs(continued - state.accepted_state.fields[0]))
        ),
    }
    elapsed = perf_counter() - started
    deadline.remaining()
    return {
        "status": "published",
        "path": str(receipt.path),
        "content_id": receipt.content_id,
        "design_case": design_case,
        "identity": identity,
        "uninterrupted_continuation": continuation,
        "original_campaign": campaign,
        "archive_limits": {
            "max_members": limits.max_members,
            "max_manifest_nesting": limits.max_manifest_nesting,
        },
        "elapsed_seconds": elapsed,
        "source_clock_seconds": controls.timeout,
        "maximum_work_units": controls.maximum_work_units,
        "interpreter_process": os.getpid(),
        "state_scope": "Actual original u, density2, affine advected history; no invented RNG/constitutive state",
    }


def _uninterrupted_reference(
    reference: object,
    state: NativeDesignSourceState,
    content_id: str,
) -> jnp.ndarray:
    """Bind the producer's uninterrupted continuation to the restored accepted state."""
    if not isinstance(reference, dict):
        raise RuntimeError(
            "_uninterrupted_reference.reference must be a published producer mapping; got another JSON value."
        )
    if reference.get("status") != "published" or reference.get("content_id") != content_id:
        raise RuntimeError(
            f"_uninterrupted_reference.reference must be the published record of {content_id!r}; got {reference.get('content_id')!r}."
        )
    restored = json.loads(json.dumps(_identity(state), sort_keys=True))
    if reference.get("identity") != restored:
        raise RuntimeError(
            "_uninterrupted_reference.identity must equal the restored source, decision and accepted-field fingerprint; got a different producer identity."
        )
    continuation = reference.get("uninterrupted_continuation")
    fields = ("fingerprint", "values", "change_from_accepted")
    if not isinstance(continuation, dict) or any(
        field not in continuation for field in fields
    ):
        raise RuntimeError(
            f"_uninterrupted_reference.uninterrupted_continuation must be a mapping with {fields!r}; got {continuation!r:.200}."
        )
    accepted = state.accepted_state.fields[0]
    try:
        array = np.asarray(continuation["values"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "_uninterrupted_reference.values must be a numeric array; got a malformed record."
        ) from error
    if array.shape != accepted.shape or not bool(np.all(np.isfinite(array))):
        raise RuntimeError(
            f"_uninterrupted_reference.values must be finite with the accepted-field shape {accepted.shape!r}; got shape {array.shape!r}."
        )
    values = jnp.asarray(array)
    fingerprint = json.loads(json.dumps(array_tree_fingerprint(values), sort_keys=True))
    if fingerprint != continuation["fingerprint"]:
        raise RuntimeError(
            "_uninterrupted_reference.values must exactly match the recorded continued-solve fingerprint; got different values."
        )
    change = float(jnp.max(jnp.abs(values - accepted)))
    if continuation["change_from_accepted"] != change:
        raise RuntimeError(
            f"_uninterrupted_reference.change_from_accepted must equal the bound values' change from the restored accepted field; got {continuation['change_from_accepted']!r}, computed {change!r}."
        )
    return values


def _require_fields(
    state: NativeDesignSourceState,
    prepared: phx.discretization.FiniteElementDiscretization,
) -> None:
    if prepared.prepared_id != state.target_fields.prepared_id:
        raise RuntimeError(
            f"_require_fields.prepared_id must equal state.target_fields.prepared_id; got {prepared.prepared_id!r}."
        )
    for value, space in zip(
        state.accepted_state.fields, prepared.field_spaces, strict=True
    ):
        if value.shape != space.vector_space.structure().shape or not bool(
            jnp.all(jnp.isfinite(value))
        ):
            raise RuntimeError(
                f"_require_fields.value must have the declared space shape and finite values; got {value!r}."
            )
    density_error = float(jnp.max(jnp.abs(state.accepted_state.fields[1] - 2.0)))
    expected_history = prepared.dof_maps[2].dof_coordinates[:, 1] + 0.5
    history_error = float(
        jnp.max(jnp.abs(state.accepted_state.fields[2] - expected_history))
    )
    if max(density_error, history_error) > 1.0e-10:
        raise RuntimeError(
            f"_require_fields.field_errors must not exceed 1e-10 for density and advected history; got {(density_error, history_error)!r}."
        )


def _design_consumer_reference(
    reference_path: Path,
    *,
    maximum_work_units: int | None = None,
    timeout: float | None = None,
) -> tuple[QualificationControls, dict[str, object]]:
    reference = json.loads(reference_path.read_text())
    if (
        not isinstance(reference, dict)
        or not isinstance(reference.get("design_case"), str)
        or reference["design_case"] not in design_qualification_cases()
    ):
        raise RuntimeError(
            "consume.reference must name the original registered design case; got a partial producer record."
        )
    controls = original_controls(
        reference["design_case"], maximum_work_units=maximum_work_units, timeout=timeout
    )
    if (
        type(reference.get("maximum_work_units")) is not int
        or isinstance(reference.get("source_clock_seconds"), bool)
        or reference.get("source_clock_seconds") != controls.timeout
        or reference.get("maximum_work_units") != controls.maximum_work_units
    ):
        raise RuntimeError(
            "consume.reference resource controls must equal the authored work/wall limits."
        )
    campaign = reference.get("original_campaign")
    expected_controls = {
        name: getattr(controls, name)
        for name in (
            "design_case", "resolution", "capacity", "adaptation_rounds", "repeats",
            "maximum_work_units", "timeout",
        )
    }
    expected_case = json.loads(json.dumps(_design_qualification_case(controls)))
    if (
        not isinstance(campaign, dict)
        or campaign.get("status") != "passed"
        or campaign.get("controls") != expected_controls
        or campaign.get("case") != expected_case
    ):
        raise RuntimeError(
            "consume.reference.original_campaign must retain the passed original case and controls."
        )
    return controls, reference


def _consume_design_request(
    path: Path, content_id: str, reference: dict[str, object],
    controls: QualificationControls,
    budget: CoordinateEnclosureBudget,
    carrier: list[phx.meshing.CellMeshingResult],
) -> dict[str, object]:
    """Restore source/state, prepare the FE owners, continue and requalify them.

    ``reference`` is the validated ``produce`` record. The cold continued solve from
    the restored accepted state must agree with the producer's uninterrupted
    continued solve; the accepted epoch is replayed from its archived solution.
    """
    started = perf_counter()
    admit_meshing_source_model(_MARKER_IDENTITY, LinearMarker)
    limits = ArrayArchiveLimits()
    state = read_meshing_source_closure(
        path, expected_content_id=content_id, limits=limits, budget=budget
    )
    if not isinstance(state, NativeDesignSourceState) or not isinstance(
        state.learned_proposer.model, LinearMarker
    ):
        raise TypeError(
            f"consume.state must be NativeDesignSourceState with a LinearMarker model; got {type(state).__name__!r}."
        )
    carrier.append(state.accepted)
    # The archive reader already renews this exact owner's source integrity,
    # including its fixed implicit preparation. Do not rediscover it a second time.
    uninterrupted = _uninterrupted_reference(reference, state, content_id)
    source_acceptance = _design_qualification_source_acceptance(
        state.initial, state.geometry
    )
    target_acceptance = _design_qualification_source_acceptance(
        state.accepted, state.geometry
    )
    scalar, all_fields, problem, fields = _design_qualification_prepare(state.accepted, budget=budget)
    if tuple(field.name for field in fields) != tuple(
        field.name for field in state.field_specs
    ):
        raise RuntimeError(
            f"consume.field_names must equal state.field_specs names; got {tuple(field.name for field in fields)!r}."
        )
    _require_fields(state, all_fields)
    preparation_seconds = perf_counter() - started
    if preparation_seconds >= controls.timeout:
        raise RuntimeError(
            f"consume.preparation_seconds must be less than controls.timeout; got {preparation_seconds!r}."
        )
    values, compiler, solve_stages = _compiled_solver_stages(
        problem, state.accepted_state.fields[0], controls.repeats
    )
    values = _design_qualification_array(values)
    defect = float(jnp.max(jnp.abs(values - uninterrupted)))
    if not defect <= 1.0e-10:
        raise RuntimeError(
            f"consume.defect must not exceed 1e-10 against the uninterrupted continued solve; got {defect!r}."
        )
    change = float(jnp.max(jnp.abs(values - state.accepted_state.fields[0])))
    physical = _design_qualification_physical(state.accepted, scalar, values, 10)
    if not (
        physical["field_L2_error"] <= controls.target_error
        and physical["field_L2_error"] < state.decision.baseline.bound
    ):
        raise RuntimeError(
            f"consume.field_L2_error must not exceed controls.target_error and be less than the baseline bound; got {physical['field_L2_error']!r}."
        )
    # The accepted epoch's transfer refreshed u with its archived epoch solution;
    # the reanalysis oracle independently re-solves the target PDE.
    executor, reanalysis, _, records = _design_qualification_transaction(
        state.initial,
        state.accepted,
        scalar,
        all_fields,
        problem,
        fields,
        state.accepted_state.fields[0],
    )
    actual = reanalysis(state.accepted.mesh, state.accepted_state.fields, None, None)
    state.decision.require_reanalysis(actual)
    # Replay the allfield transition/physical acceptance from the
    # accepted epoch; a decoded report or preserved constants alone is not proof.
    replay = executor.execute(
        state.initial_state,
        state.initial.mesh,
        state.adaptation,
        decision=state.decision,
        reanalysis=reanalysis,
        compiled_layout_id=problem.compilation_id,
    )
    if (
        not bool(replay.committed)
        or replay.receipt is None
        or not replay.receipt.published
    ):
        raise RuntimeError(replay.diagnostics)
    for original, renewed in zip(
        state.accepted_state.fields, replay.state.fields, strict=True
    ):
        if not float(jnp.max(jnp.abs(original - renewed))) <= 1.0e-10:
            raise RuntimeError(
                f"consume.fields must differ by at most 1e-10 after transition; got {(original, renewed)!r}."
            )
    source_scalar, _, source_problem, _ = _design_qualification_prepare(state.initial, budget=budget)
    base = _design_qualification_derivatives(
        state.geometry, state.plan, state.initial, source_scalar, source_problem, None
    )
    derivative_records: list[phx.meshing.FixedEpochDerivativeEvidence] = []
    derivative = _design_qualification_derivatives(
        state.geometry,
        state.plan,
        state.initial,
        source_scalar,
        source_problem,
        {
            "candidate": state.decision.require_selected(
                state.initial.result_id,
                state.adaptation.result_id,
                state.accepted.result_id,
            ),
            "adaptation": state.adaptation,
            "executor": executor,
            "proposal": "restored-selected",
            "reanalysis": reanalysis,
            "problem": problem,
            "scalar": scalar,
            "all_fields": all_fields,
            "values": state.accepted_state.fields[0],
            "staged": replay,
            "physical": actual,
            "learned_proposer": state.learned_proposer,
            "learned_features": state.learned_features,
            "proposal_transaction": state.learned_transaction,
        },
        base,
        retained_evidence=derivative_records,
    )
    if not derivative_records or (
        derivative_records[0].epoch_id != state.derivative.epoch_id
        or derivative_records[0].parameter_id != state.derivative.parameter_id
        or derivative_records[0].routes != state.derivative.routes
    ):
        raise RuntimeError(
            f"consume.derivative_records must retain the qualified epoch, parameter and routes; got {derivative_records!r}."
        )
    identity = _identity(state)
    restored_result = _result_identity(state.accepted)
    elapsed = perf_counter() - started
    if elapsed > controls.timeout:
        raise RuntimeError(
            f"consume.elapsed must not exceed controls.timeout during composite reanalysis; got {elapsed!r}."
        )
    return {
        "status": "passed",
        "identity": identity,
        "source_clock_seconds": controls.timeout,
        "maximum_work_units": controls.maximum_work_units,
        "restored_result": restored_result,
        "source_acceptance": source_acceptance,
        "target_acceptance": target_acceptance,
        "continued_physical": physical,
        "continued_solution_defect": defect,
        "continued_change_from_accepted": change,
        "allfield_replay_receipt": replay.receipt.receipt_id,
        "actual_reanalysis": records[-1],
        "composite_derivative": derivative,
        "compiler": compiler,
        "solve_stages_seconds": solve_stages,
        "preparation_seconds": preparation_seconds,
        "elapsed_seconds": elapsed,
        "archive_limits": {
            "max_members": limits.max_members,
            "max_manifest_nesting": limits.max_manifest_nesting,
        },
        "interpreter_process": os.getpid(),
        "source_field_id": source_scalar.prepared_id,
        "state_scope": "Original u/density/advected history; no provided RNG or constitutive state",
        "claim": "Source-backed cold/allfield/composite consumer only; no superiority or global qualification",
    }


def consume(
    path: Path, content_id: str, reference_path: Path,
    *, deadline: Deadline | None = None,
    maximum_work_units: int | None = None, timeout: float | None = None,
) -> dict[str, object]:
    """Restore, reprove and continue under the bound authored request allowance."""
    started = perf_counter()
    admit_meshing_source_model(_MARKER_IDENTITY, LinearMarker)
    scientific_controls, reference = _design_consumer_reference(
        reference_path, maximum_work_units=maximum_work_units, timeout=timeout
    )
    expiry = started + scientific_controls.timeout
    deadline = Deadline(expiry if deadline is None else min(expiry, deadline.expires_at))
    deadline.remaining()
    controls = read_meshing_source_execution_controls(
        path, expected_content_id=content_id
    )
    if (
        controls.limits.maximum_work_units != scientific_controls.maximum_work_units
        or controls.limits.maximum_wall_seconds != scientific_controls.timeout
    ):
        raise RuntimeError(
            "consume.archive resource controls must equal the authored work/wall limits."
        )
    carrier: list[phx.meshing.CellMeshingResult] = []
    renewed = execute_native_request(
        controls.execution_evidence,
        controls.limits,
        lambda budget: _consume_design_request(
            path, content_id, reference, scientific_controls, budget, carrier
        ),
        target=lambda _: carrier[0],
        failure_factory=qualification_failure,
        workflow_remaining_seconds=deadline.remaining(),
    )
    execution = renewed.result.execution_evidence
    if execution is None:
        raise RuntimeError("consume.request must retain its ended execution record.")
    renewed.value["native_execution"] = execution.to_record()
    renewed.value["native_execution_scope"] = (
        "fresh authored-cap reader/reproof/preparation/continued-design request"
    )
    deadline.remaining()
    return renewed.value


def qualify(
    path: Path,
    reference_path: Path,
    *,
    design_case: DesignCase = DESIGN_QUALIFICATION_DEFAULT_CASE,
    maximum_work_units: int | None = None,
    timeout: float | None = None,
) -> dict[str, object]:
    """Run real production and an independent interpreter under one whole clock."""
    started = perf_counter()
    controls = original_controls(
        design_case, maximum_work_units=maximum_work_units, timeout=timeout
    )
    deadline = Deadline(started + controls.timeout)
    path, reference_path = path.resolve(), reference_path.resolve()
    root = Path(__file__).resolve().parents[1]
    resource_arguments = [
        "--maximum-work-units", str(controls.maximum_work_units),
        "--timeout", str(controls.timeout),
    ]
    try:
        producer = run_campaign_process(
            [
                sys.executable, "-m", "tools.meshing_design_source", "produce",
                str(path), "--design-case", design_case,
                *resource_arguments,
            ],
            cwd=root,
            timeout=deadline.remaining(),
        )
    except CampaignProcessTimeout as error:
        error.add_note(
            f"Original producer stdout:\n{error.stdout}\nOriginal producer stderr:\n{error.stderr}"
        )
        raise
    if producer.deadline_expired:
        raise TimeoutError(
            "qualify.producer exhausted the original whole workflow clock; "
            f"original stdout:\n{producer.stdout}\noriginal stderr:\n{producer.stderr}"
        )
    if producer.returncode != 0:
        raise RuntimeError(
            f"qualify.producer refused with exit {producer.returncode}; "
            f"original producer diagnostics:\n{producer.stderr}"
        )
    published = json.loads(producer.stdout)
    deadline.remaining()
    reference_path.write_text(json.dumps(published, sort_keys=True, allow_nan=False))
    try:
        completed = run_campaign_process(
            [
                sys.executable, "-m", "tools.meshing_design_source", "consume",
                str(path), str(published["content_id"]), str(reference_path),
                *resource_arguments,
            ],
            cwd=root,
            timeout=deadline.remaining(),
        )
    except CampaignProcessTimeout as error:
        error.add_note(
            f"Original cold stdout:\n{error.stdout}\nOriginal cold stderr:\n{error.stderr}"
        )
        raise
    if completed.deadline_expired:
        raise TimeoutError(
            "qualify.cold exhausted the original whole workflow clock; "
            f"original stdout:\n{completed.stdout}\noriginal stderr:\n{completed.stderr}"
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"qualify.cold refused with exit {completed.returncode}; original consumer diagnostics:\n{completed.stderr}"
        )
    cold = json.loads(completed.stdout)
    elapsed = perf_counter() - started
    deadline.remaining()
    if cold.get("status") != "passed" or cold.get("identity") != published["identity"]:
        raise RuntimeError("qualify.cold must pass with the exact published accepted state.")
    if cold.get("interpreter_process") == published["interpreter_process"]:
        raise RuntimeError("qualify.cold must execute independently of the producer.")
    return {
        "status": "passed",
        "design_case": design_case,
        "producer": published,
        "cold": cold,
        "end_to_end_seconds": elapsed,
        "source_clock_seconds": controls.timeout,
        "maximum_work_units": controls.maximum_work_units,
        "producer_process": published["interpreter_process"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    publish = subparsers.add_parser("produce")
    publish.add_argument("path", type=Path)
    publish.add_argument("--design-case", choices=tuple(design_qualification_cases()), default=DESIGN_QUALIFICATION_DEFAULT_CASE)
    restore = subparsers.add_parser("consume")
    restore.add_argument("path", type=Path)
    restore.add_argument("content_id")
    restore.add_argument("reference", type=Path)
    whole = subparsers.add_parser("qualify")
    whole.add_argument("path", type=Path)
    whole.add_argument("reference", type=Path)
    whole.add_argument("--design-case", choices=tuple(design_qualification_cases()), default=DESIGN_QUALIFICATION_DEFAULT_CASE)
    for command in (publish, restore, whole):
        command.add_argument("--maximum-work-units", type=int)
        command.add_argument("--timeout", type=float)
    args = parser.parse_args()
    if args.operation == "produce":
        result = produce(
            args.path, design_case=args.design_case,
            maximum_work_units=args.maximum_work_units, timeout=args.timeout,
        )
    elif args.operation == "qualify":
        result = qualify(
            args.path, args.reference, design_case=args.design_case,
            maximum_work_units=args.maximum_work_units, timeout=args.timeout,
        )
    else:
        result = consume(
            args.path, args.content_id, args.reference,
            maximum_work_units=args.maximum_work_units, timeout=args.timeout,
        )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
