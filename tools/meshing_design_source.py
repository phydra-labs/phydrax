#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Original native design Source1 publication and genuine fresh-process reanalysis.

This is the original sphere/.75, res4/count9, [-1.4,1.4]^3, cap20000,
120-second, three-comparator case. Archive limits stay at their canonical
257-member/16-level defaults. The original source has no carried RNG or
constitutive history; its actual u/density/advected-history owners are retained.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import jax.numpy as jnp

import phydrax as phx

# Trusted original scientific model declaration, imported before artifact restore.
from examples.curved_native_transfer import LinearMarker
from phydrax._array_archive import ArrayArchiveLimits
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    register_native_design_source_artifacts,
    write_meshing_source_closure,
)
from phydrax.meshing._decision import NativeDesignSourceState
from tools.meshing_qualification import (
    _compiled_solver_stages,
    _design_qualification_array,
    _design_qualification_derivatives,
    _design_qualification_lifecycle,
    _design_qualification_physical,
    _design_qualification_prepare,
    _design_qualification_source_acceptance,
    _design_qualification_transaction,
    _result_identity,
)


def original_controls() -> argparse.Namespace:
    return argparse.Namespace(
        resolution=4,
        capacity=20000,
        timeout=120.0,
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


def produce(path: Path) -> dict[str, object]:
    """Execute the original case once with its newly retained scientific owners."""
    started = perf_counter()
    register_native_design_source_artifacts()
    controls = original_controls()
    states: list[NativeDesignSourceState] = []
    campaign = _design_qualification_lifecycle(controls, retained_source_states=states)
    if campaign["status"] != "passed" or len(states) != 1:
        raise RuntimeError(
            json.dumps({"original_campaign": campaign, "retained_states": len(states)})
        )
    remaining = controls.timeout - (perf_counter() - started)
    if remaining <= 0.0:
        raise RuntimeError(
            "Original source campaign clock expired before archive publication."
        )
    state = states[0]
    limits = ArrayArchiveLimits()
    receipt = write_meshing_source_closure(path, state, limits=limits)
    elapsed = perf_counter() - started
    if elapsed > controls.timeout:
        raise RuntimeError(
            "Original120 source clock expired during default-limit publication."
        )
    return {
        "status": "published",
        "path": str(receipt.path),
        "content_id": receipt.content_id,
        "identity": _identity(state),
        "original_campaign": campaign,
        "archive_limits": {
            "max_members": limits.max_members,
            "max_manifest_nesting": limits.max_manifest_nesting,
        },
        "elapsed_seconds": elapsed,
        "source_clock_seconds": controls.timeout,
        "state_scope": "Actual original u, density2, affine advected history; no invented RNG/constitutive state",
    }


def _require_fields(
    state: NativeDesignSourceState,
    prepared: phx.discretization.FiniteElementDiscretization,
) -> None:
    if prepared.prepared_id != state.target_fields.prepared_id:
        raise RuntimeError(
            "Cold FE preparation changed the original accepted field owner."
        )
    for value, space in zip(
        state.accepted_state.fields, prepared.field_spaces, strict=True
    ):
        if value.shape != space.vector_space.structure().shape or not bool(
            jnp.all(jnp.isfinite(value))
        ):
            raise RuntimeError(
                "Cold accepted field does not realize its original physical layout."
            )
    density_error = float(jnp.max(jnp.abs(state.accepted_state.fields[1] - 2.0)))
    expected_history = prepared.dof_maps[2].dof_coordinates[:, 1] + 0.5
    history_error = float(
        jnp.max(jnp.abs(state.accepted_state.fields[2] - expected_history))
    )
    if max(density_error, history_error) > 1.0e-10:
        raise RuntimeError(
            "Cold density or actual original advected history failed its unchanged physical gate."
        )


def consume(path: Path, content_id: str) -> dict[str, object]:
    """Restore exact source/state; reprepare, solve and requalify the real owners."""
    started = perf_counter()
    register_native_design_source_artifacts()
    controls = original_controls()
    limits = ArrayArchiveLimits()
    state = read_meshing_source_closure(
        path, expected_content_id=content_id, limits=limits
    )
    if not isinstance(state, NativeDesignSourceState) or not isinstance(
        state.learned_proposer.model, LinearMarker
    ):
        raise TypeError(
            "Source1 archive did not restore its exact original design/model declaration."
        )
    state.validate_source_integrity()
    source_acceptance = _design_qualification_source_acceptance(
        state.initial, state.geometry
    )
    target_acceptance = _design_qualification_source_acceptance(
        state.accepted, state.geometry
    )
    scalar, all_fields, problem, fields = _design_qualification_prepare(state.accepted)
    if tuple(field.name for field in fields) != tuple(
        field.name for field in state.field_specs
    ):
        raise RuntimeError(
            "Cold allfield declaration changed its original scientific names."
        )
    _require_fields(state, all_fields)
    preparation_seconds = perf_counter() - started
    if preparation_seconds >= controls.timeout:
        raise RuntimeError(
            "Original120 cold source clock expired before continued PDE solve."
        )
    values, compiler, solve_stages = _compiled_solver_stages(
        problem, state.accepted_state.fields[0], controls.repeats
    )
    values = _design_qualification_array(values)
    defect = float(jnp.max(jnp.abs(values - state.accepted_state.fields[0])))
    if defect > 1.0e-10:
        raise RuntimeError(
            "Cold independently continued PDE differs from the original accepted solution."
        )
    physical = _design_qualification_physical(state.accepted, scalar, values, 10)
    if (
        physical["field_L2_error"] > controls.target_error
        or physical["field_L2_error"] >= state.decision.baseline.bound
    ):
        raise RuntimeError(
            "Cold physical source objective did not retain the actual admitted improvement."
        )
    executor, reanalysis, _, records = _design_qualification_transaction(
        state.initial,
        state.accepted,
        scalar,
        all_fields,
        problem,
        fields,
        values,
    )
    actual = reanalysis(state.accepted.mesh, state.accepted_state.fields, None, None)
    state.decision.require_reanalysis(actual)
    # Replay the actual allfield transition/physical acceptance from the original
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
        if float(jnp.max(jnp.abs(original - renewed))) > 1.0e-10:
            raise RuntimeError(
                "Cold allfield transition violates the original declared-field error gate."
            )
    source_scalar, _, _, _ = _design_qualification_prepare(state.initial)
    base = _design_qualification_derivatives(
        state.geometry, state.plan, state.initial, None
    )
    derivative_records: list[phx.meshing.FixedEpochDerivativeEvidence] = []
    derivative = _design_qualification_derivatives(
        state.geometry,
        state.plan,
        state.initial,
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
            "values": values,
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
            "Cold composite derivative source/plan routes differ from the original qualification."
        )
    elapsed = perf_counter() - started
    if elapsed > controls.timeout:
        raise RuntimeError(
            "Original120 cold source clock expired during composite reanalysis."
        )
    return {
        "status": "passed",
        "identity": _identity(state),
        "restored_result": _result_identity(state.accepted),
        "source_acceptance": source_acceptance,
        "target_acceptance": target_acceptance,
        "continued_physical": physical,
        "recomputed_solution_defect": defect,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    publish = subparsers.add_parser("produce")
    publish.add_argument("path", type=Path)
    restore = subparsers.add_parser("consume")
    restore.add_argument("path", type=Path)
    restore.add_argument("content_id")
    args = parser.parse_args()
    result = (
        produce(args.path)
        if args.operation == "produce"
        else consume(args.path, args.content_id)
    )
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
