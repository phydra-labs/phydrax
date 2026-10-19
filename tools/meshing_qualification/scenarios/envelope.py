# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import json
import os
import resource
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, final

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import (
    array_tree_fingerprint,
    ArrayArchiveLimits,
    canonical_fingerprint,
    model_array_bytes,
)
from phydrax.geometry import SurfaceMetadata
from phydrax.lifecycle import (
    MeshingAcceptedEpoch,
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.meshing import (
    EnvelopeFeaturePermission,
    EnvelopeTopologyPermission,
    NativeMeshingPhaseMeasurement,
    RawTriangleSoup,
    SurfaceEnvelopePolicy,
)
from phydrax.meshing.qualification import (
    prepare_envelope_accepted_epoch,
    prepare_envelope_solver,
    transition_envelope_state,
    validate_envelope_acceptance,
)
from tools.meshing_qualification._controls import (
    capacity_meshing_limits,
    QualificationControls,
)
from tools.meshing_qualification._physics import (
    _compiled_solver_stages,
    _manufactured_source,
    _manufactured_value,
    _physical_error,
)
from tools.meshing_qualification._records import (
    _phase_recordings,
    _result_identity,
    record_failure,
)
from tools.meshing_qualification._runtime import (
    _measured,
    _runtime_identity,
    ColdWorkerTimeout,
    Deadline,
    peak_resident_bytes,
    qualification_failure,
    run_cold_worker,
    StageLedger,
)
from tools.meshing_qualification.scenarios.bisection import (
    _bisection_lineage,
    _bisection_p1_integral,
)
from tools.meshing_qualification.scenarios.triangulation import _simplex_measures


def _envelope_qualification_fixture(
    resolution: int,
    capacity: int,
) -> tuple[RawTriangleSoup, SurfaceEnvelopePolicy, dict[str, object]]:
    """Freeze the positive open-sheet source and explicit thickening semantics."""

    payload = {
        "vertices": ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)),
        "triangles": ((0, 1, 2),),
        "feature_edges": ((0, 1),),
        "coordinate_units": "meter",
        "region_id": "wrapped-solid",
        "interpretation": "sampled_unsigned_distance_sublevel",
        "offset": 0.263,
        "maximum_deviation": 0.55,
        "feature_permission": "allow_rounding",
        "topology_permission": "allow_change",
        "provenance": "PHYDRA synthetic open sheet; test_surface_envelope positive fixture",
    }
    digest = canonical_fingerprint(payload)
    metadata = SurfaceMetadata(
        source_id="qualification-open-envelope-sheet",
        source_revision=digest,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        provenance=(payload["provenance"],),
    )
    soup = RawTriangleSoup(
        np.asarray(payload["vertices"], dtype=np.float64),
        np.asarray(payload["triangles"], dtype=np.int64),
        metadata,
        feature_edges=np.asarray(payload["feature_edges"], dtype=np.int64),
    )
    policy = SurfaceEnvelopePolicy(
        offset=0.263,
        maximum_deviation=0.55,
        spacing=0.44 / resolution,
        certificate_spacing=0.2 / resolution,
        feature_permission=EnvelopeFeaturePermission.ALLOW_ROUNDING,
        topology_permission=EnvelopeTopologyPermission.ALLOW_CHANGE,
        max_samples=5 * capacity,
        max_work_units=1000 * capacity,
        max_vertices=capacity,
        max_triangles=capacity,
        max_volume_vertices=capacity,
        max_tetrahedra=5 * capacity,
    )
    return (
        soup,
        policy,
        {
            "name": "open-sheet-repaired-envelope",
            "source_digest": digest,
            "source_geometry_id": soup.soup_id,
            "source_id": metadata.source_id,
            "source_revision": metadata.source_revision,
            "payload": payload,
            "policy_id": policy.policy_id,
            "spacing": policy.spacing,
            "certificate_spacing": policy.certificate_spacing,
        },
    )


def _envelope_qualification_limits(
    capacity: int, timeout: float
) -> phx.meshing.MeshingLimits:
    return capacity_meshing_limits(
        capacity, timeout, cells=5, work_units=1000, cavity_cells=None
    )


@final
class _EnvelopePreparationStages(phx.StrictModule):
    record: dict[str, Any]
    deadline: Deadline

    def run[T](self, name: str, call: Callable[[], T]) -> T:
        """Retain the nested construction/request measurements at the tool edge."""

        phase = (
            self.record.get("active_phase", "wrap_surface_envelope")
            if name == "native_envelope_construction"
            else name
        )
        value, measured = StageLedger(self.record, deadline=self.deadline).run(
            phase, lambda: _measured(call)
        )
        self.record["stages_seconds"][name] = measured["wall_seconds"]
        return value


# Accepted volume identity binds the repaired source, certified mesh and controls.
type _AcceptedEnvelopeVolume = tuple[
    phx.meshing.NativeSurfaceEnvelopeSource,
    phx.meshing.CellMeshingResult,
    phx.meshing.VolumeMeshingSpec,
    phx.meshing.NativeMeshingOptions,
]


def _prepare_envelope_volume(
    args: QualificationControls,
    record: dict[str, Any],
    ledger: StageLedger,
    phases: list[NativeMeshingPhaseMeasurement],
    *,
    deadline: Deadline,
) -> _AcceptedEnvelopeVolume:
    """Repair the source, publish its strict volume, then independently accept it."""
    stages = record["stages_seconds"]
    fixture, measured = ledger.run(
        record.get("active_phase", "_envelope_qualification_fixture"),
        lambda: _measured(
            lambda: _envelope_qualification_fixture(args.resolution, args.capacity)
        ),
    )
    soup, policy, record["case"] = fixture
    stages["source_preparation"] = measured["wall_seconds"]

    prepared = ledger.run(
        "native_envelope_construction",
        lambda: phx.meshing.qualification.prepare_envelope_source(
            soup,
            policy,
            _envelope_qualification_limits(args.capacity, args.timeout),
            failure_factory=qualification_failure,
            record_phase=phases.append,
            stages=_EnvelopePreparationStages(record=record, deadline=deadline),
        ),
    )
    envelope, request, source_receipt = (
        prepared.envelope,
        prepared.request,
        prepared.receipt,
    )
    source, specification, options = (
        request.source,
        request.specification,
        request.options,
    )
    record["repair"] = {
        "envelope_id": envelope.envelope_id,
        "carrier_id": envelope.carrier_id,
        "evidence": asdict(envelope.evidence),
        "original_source_geometry_id": soup.soup_id,
        "repaired_model_id": envelope.repaired.model.model_id,
        "interpretation": "sampled_unsigned_distance_sublevel",
        "offset": policy.offset,
        "maximum_deviation": policy.maximum_deviation,
        "spacing": policy.spacing,
        "certificate_spacing": policy.certificate_spacing,
        "feature_permission": policy.feature_permission.value,
        "topology_permission": policy.topology_permission.value,
        "topology_comparison_semantics": "incidence-counts-not-homeomorphism",
    }
    record.update(
        {
            "source_binding_id": source.binding_id,
            "request_id": specification.specification_id,
            "limits_id": specification.limits.limits_id,
            "native_options_id": options.options_id,
        }
    )
    provider = phx.meshing.NativeMeshingProvider(options)

    plan, measured = ledger.run(
        "geometry_constraint_preparation",
        lambda: _measured(
            lambda: provider.plan(
                source,
                specification,
                coordinate_contract=phx.SpatialCoordinateContract.si(),
                record_phase=phases.append,
                source_preparation_evidence=source_receipt,
            )
        ),
    )
    stages["geometry_constraint_preparation"] = measured["wall_seconds"]
    record["prepared_id"] = plan.plan_id

    result, measured = ledger.run(
        "generation_publication_certification",
        lambda: _measured(lambda: plan.execute(record_phase=phases.append)),
    )
    stages["generation_publication_certification"] = measured["wall_seconds"]
    record["generation"] = _result_identity(result)

    acceptance, measured = ledger.run(
        "independent_acceptance",
        lambda: _measured(
            lambda: validate_envelope_acceptance(
                source,
                result,
                limits=specification.limits,
                failure_factory=qualification_failure,
                measure_cells=_simplex_measures,
            )
        ),
    )
    stages["independent_acceptance"] = measured["wall_seconds"]
    record["independent_acceptance"] = acceptance
    return source, result, specification, options


def _solve_envelope_epoch(
    result: phx.meshing.CellMeshingResult,
    previous_values: jax.Array | None,
    args: QualificationControls,
    record: dict[str, Any],
    ledger: StageLedger,
    started: float,
    epoch: int,
) -> tuple[
    phx.discretization.FiniteElementDiscretization,
    phx.equations.CompiledFiniteElementProblem,
    jax.Array,
    dict[str, Any],
]:
    """Solve from the carried field and record independent physical-error evidence."""
    stages = record["stages_seconds"]
    solver, measured = ledger.run(
        f"solver_prepare_epoch_{epoch}",
        lambda: _measured(
            lambda: prepare_envelope_solver(
                result,
                failure_factory=qualification_failure,
                source_coefficient=_manufactured_source,
                dirichlet_values=_manufactured_value,
            )
        ),
    )
    space, problem, free_dofs = solver
    stages[record["active_phase"]] = measured["wall_seconds"]
    seed = jax.device_put(
        np.zeros((free_dofs.size,), dtype=result.mesh.coordinates.dtype)
        if previous_values is None
        else np.asarray(previous_values)[np.asarray(free_dofs)]
    )

    values, compiler, timings = ledger.run(
        f"solver_execution_epoch_{epoch}",
        lambda: _compiled_solver_stages(problem, seed, args.repeats),
    )
    stages.update({f"{key}_epoch_{epoch}": value for key, value in timings.items()})
    error = _physical_error(result.mesh, values)
    if error <= args.target_error and record["time_to_target_error_seconds"] is None:
        record["time_to_target_error_seconds"] = perf_counter() - started
    epoch_record = {
        "epoch": epoch,
        "result": _result_identity(result),
        "l2_error": error,
        "dofs": space.dof_maps[0].global_dof_count,
        "free_dofs": free_dofs.size,
        "compiler": compiler,
        "retained_bytes": model_array_bytes(
            (result, space, problem, values, previous_values)
        ),
        "seed_origin": "zero"
        if previous_values is None
        else "actual-transferred-solver-state",
        "transferred_state_l2_error": (
            None
            if previous_values is None
            else _physical_error(result.mesh, previous_values)
        ),
    }
    record["epochs"].append(epoch_record)
    return space, problem, values, epoch_record


def _envelope_qualification_epochs(
    source: phx.meshing.NativeSurfaceEnvelopeSource | phx.meshing.NativePlcSource,
    result: phx.meshing.CellMeshingResult,
    specification: phx.meshing.VolumeMeshingSpec,
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    *,
    deadline: Deadline,
    generation_options: phx.meshing.NativeMeshingOptions | None = None,
    accept_result: Callable[[phx.meshing.CellMeshingResult], dict[str, object]]
    | None = None,
) -> None:
    """Continue the PDE with every accepted density/history field carried atomically."""

    _ledger = StageLedger(record, deadline=deadline)
    if accept_result is None:
        if not isinstance(source, phx.meshing.NativeSurfaceEnvelopeSource):
            raise ValueError(
                f"_envelope_qualification_epochs.source must be a NativeSurfaceEnvelopeSource with source acceptance; got {type(source).__name__!r}."
            )

        def accept_envelope(
            candidate: phx.meshing.CellMeshingResult,
        ) -> dict[str, object]:
            return validate_envelope_acceptance(
                source,
                candidate,
                limits=specification.limits,
                failure_factory=qualification_failure,
                measure_cells=_simplex_measures,
            )

        accept_result = accept_envelope
    previous_values = None
    previous_auxiliary = None
    accepted_epochs: list[MeshingAcceptedEpoch] = []
    accepted_transition = None
    stages = record["stages_seconds"]

    association_transfer, measured = _ledger.run(
        "source_association_transfer_preparation",
        lambda: _measured(
            lambda: phx.meshing.qualification.prepare_envelope_association_transfer(
                source, result, specification.limits
            )
        ),
    )
    stages[record["active_phase"]] = measured["wall_seconds"]
    record["association_transfer"] = {
        "transfer_id": association_transfer.transfer_id,
        "domain_id": association_transfer.domain.domain_id,
        "source_revision": association_transfer.source_revision,
        "source_edges": association_transfer.edge_vertices.shape[0],
        "source_triangles": association_transfer.triangle_vertices.shape[0],
        "retained_bytes": model_array_bytes(association_transfer),
    }
    for epoch in range(args.adaptation_rounds + 1):
        space, problem, values, epoch_record = _solve_envelope_epoch(
            result, previous_values, args, record, _ledger, started, epoch
        )
        if isinstance(source, phx.meshing.NativeSurfaceEnvelopeSource):
            if previous_auxiliary is None:
                previous_auxiliary = (
                    jnp.full(values.shape, 2.0, dtype=values.dtype),
                    space.dof_maps[0].dof_coordinates[:, 0] + 0.5,
                )
            accepted_epochs.append(
                _ledger.run(
                    f"accepted_epoch_declaration_{epoch}",
                    lambda: prepare_envelope_accepted_epoch(
                        result,
                        space,
                        values,
                        previous_auxiliary,
                        epoch,
                        part_name="envelope",
                        transition=accepted_transition,
                    ),
                )
            )
        if epoch == args.adaptation_rounds:
            break

        request = _ledger.run(
            f"adaptation_prepare_epoch_{epoch}",
            lambda: phx.meshing.MarkedMeshAdaptation(
                np.concatenate(
                    [
                        np.asarray(block.global_ids, dtype=np.int64)
                        for block in result.mesh.blocks
                    ]
                ),
                np.empty((0,), dtype=np.int64),
            ),
        )
        policy = phx.meshing.MeshAdaptationPolicy(
            phx.meshing.MeshAdaptationRoute.NATIVE_MIXED,
            limits=specification.limits,
            association_transfer=association_transfer,
            audit_policy=phx.meshing.CellMeshAuditPolicy(
                require_complete_association=True,
                watertight_boundary=phx.meshing.CellMeshAuditDisposition.REJECT,
            ),
        )
        prepared, measured = _ledger.run(
            record.get("active_phase", "prepare_mesh_adaptation"),
            lambda: _measured(
                lambda: phx.meshing.prepare_mesh_adaptation(
                    result, request, policy=policy
                )
            ),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]

        adaptation, measured = _ledger.run(
            f"adaptation_execute_epoch_{epoch}",
            lambda: _measured(lambda: phx.meshing.execute_mesh_adaptation(prepared)),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]
        if not adaptation.status.converged or adaptation.transition is None:
            raise qualification_failure(
                "_envelope_qualification_epochs",
                (
                    f"Positive source-volume adaptation was not accepted: {adaptation.status.value}."
                ),
                {"adaptation": adaptation},
            )

        transition, measured = _ledger.run(
            f"solver_state_transfer_epoch_{epoch}",
            lambda: _measured(
                lambda: transition_envelope_state(
                    result,
                    adaptation,
                    problem,
                    values,
                    epoch,
                    previous_auxiliary,
                    failure_factory=qualification_failure,
                    physical_error=_physical_error,
                )
            ),
        )
        previous_values, previous_auxiliary, transfer = transition
        stages[record["active_phase"]] = measured["wall_seconds"]
        before = _bisection_p1_integral(adaptation.source.mesh, values)
        after = _bisection_p1_integral(adaptation.target.mesh, previous_values)
        if abs(after - before) > 1.0e-10 * max(1.0, abs(before)):
            raise qualification_failure(
                "_envelope_qualification_epochs",
                ("Envelope solver-state P1 inventory changed in transfer."),
                {"after": after, "before": before},
            )
        epoch_record.update(
            {
                "transfer": transfer,
                "lineage": _bisection_lineage(adaptation),
                "state_inventory": {
                    "source": before,
                    "target": after,
                    "defect": after - before,
                },
                "adaptation_retained_bytes": model_array_bytes(adaptation),
            }
        )
        result = adaptation.target
        accepted_transition = adaptation

        acceptance, measured = _ledger.run(
            f"independent_acceptance_epoch_{epoch + 1}",
            lambda: _measured(lambda: accept_result(result)),
        )
        stages[record["active_phase"]] = measured["wall_seconds"]
        epoch_record["target_acceptance"] = acceptance
    if isinstance(source, phx.meshing.NativeSurfaceEnvelopeSource):
        if generation_options is None:
            raise ValueError(
                "envelope.checkpoint.generation_options must retain the authored native options; got None"
            )
        _envelope_qualification_cold_reanalysis(
            source,
            result,
            specification,
            generation_options,
            tuple(accepted_epochs),
            args,
            record,
            started,
            deadline=deadline,
        )


def _envelope_qualification_cold_worker(
    path: str,
    content_id: str,
    expected: tuple[str, str, str, str],
    repeats: int,
    target_error: float,
) -> dict[str, object]:
    """Fresh-process source certification and continued solve from archived fields."""

    restored = read_meshing_source_closure(
        Path(path), expected_content_id=content_id, limits=ArrayArchiveLimits()
    )
    source = restored["generation_source"]
    result = restored["accepted_target"]
    specification = restored["generation_specification"]
    history = restored["accepted_data"]["accepted_epochs"]
    fields = tuple(history[-1].fields[name] for name in ("u", "density", "history"))
    if (
        not isinstance(source, phx.meshing.NativeSurfaceEnvelopeSource)
        or not isinstance(result, phx.meshing.CellMeshingResult)
        or not isinstance(specification, phx.meshing.VolumeMeshingSpec)
        or (
            source.binding_id,
            result.result_id,
            specification.specification_id,
            array_tree_fingerprint(fields),
        )
        != expected
    ):
        raise qualification_failure(
            "_envelope_qualification_cold_worker",
            ("Cold envelope archive changed source, controls or accepted fields."),
            {
                "array_tree_fingerprint": array_tree_fingerprint,
                "expected": expected,
                "fields": fields,
                "result": result,
                "source": source,
                "specification": specification,
            },
        )
    acceptance = validate_envelope_acceptance(
        source,
        result,
        limits=specification.limits,
        failure_factory=qualification_failure,
        measure_cells=_simplex_measures,
    )
    space, problem, free = prepare_envelope_solver(
        result,
        failure_factory=qualification_failure,
        source_coefficient=_manufactured_source,
        dirichlet_values=_manufactured_value,
    )
    values, compiler, timings = _compiled_solver_stages(
        problem, jax.device_put(np.asarray(fields[0])[np.asarray(free)]), repeats
    )
    error = _physical_error(result.mesh, values)
    density_defect = float(np.max(np.abs(np.asarray(fields[1]) - 2.0)))
    history_defect = float(
        np.max(
            np.abs(
                np.asarray(fields[2])
                - (np.asarray(space.dof_maps[0].dof_coordinates)[:, 0] + 0.5)
            )
        )
    )
    if error > target_error or density_defect > 1.0e-12 or history_defect > 1.0e-12:
        raise qualification_failure(
            "_envelope_qualification_cold_worker",
            ("Cold envelope continued PDE or restored density/history oracle failed."),
            {
                "density_defect": density_defect,
                "error": error,
                "history_defect": history_defect,
                "target_error": target_error,
            },
        )
    return {
        "passed": True,
        "process_id": os.getpid(),
        "runtime": _runtime_identity(),
        "source_acceptance": acceptance,
        "restored_result": _result_identity(result),
        "field_names": ["u", "density", "history"],
        "accepted_fields_bitwise_preserved": True,
        "prepared_fe_caches_serialized": False,
        "continued_physical_L2_error": error,
        "density_max_defect": density_defect,
        "history_max_defect": history_defect,
        "compiler": compiler,
        "solver_seconds": timings,
    }


def _envelope_qualification_cold_reanalysis(
    source: phx.meshing.NativeSurfaceEnvelopeSource,
    result: phx.meshing.CellMeshingResult,
    specification: phx.meshing.VolumeMeshingSpec,
    generation_options: phx.meshing.NativeMeshingOptions,
    history: tuple[MeshingAcceptedEpoch, ...],
    args: QualificationControls,
    record: dict[str, Any],
    started: float,
    *,
    deadline: Deadline,
) -> None:

    _ledger = StageLedger(record, deadline=deadline)

    limits = ArrayArchiveLimits()
    fields = tuple(history[-1].fields[name] for name in ("u", "density", "history"))
    report = result.certification
    if report is None:
        raise ValueError(
            "envelope.checkpoint.certification must retain the accepted target certificate; got None"
        )
    records = {
        "certification_inputs": report.request,
        "report": report,
        "associations": result.associations,
        "generation_source": source,
        "generation_specification": specification,
        "generation_options": generation_options,
        "accepted_target": result,
        "accepted_data": {"accepted_epochs": history},
    }
    with TemporaryDirectory(prefix="phydrax-native-envelope-source-") as directory:
        receipt, timing = _ledger.run(
            "allfield_history_checkpoint_publication",
            lambda: _measured(
                lambda: write_meshing_source_closure(
                    Path(directory) / "envelope.checkpoint.zip", records, limits=limits
                )
            ),
        )
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        expected = (
            source.binding_id,
            result.result_id,
            specification.specification_id,
            array_tree_fingerprint(fields),
        )
        # The scenario module owns its worker command.
        remaining = deadline.expires_at - perf_counter()
        if remaining <= 0.0:
            raise phx.meshing.MeshingFailure(
                phx.meshing.MeshingFailureCategory.TIMED_OUT,
                "Envelope wall allowance expired before cold continued solve.",
                stage=phx.meshing.MeshingStageKind.CERTIFICATION.value,
            )

        cold_started = perf_counter()
        try:
            cold = run_cold_worker(
                "scenarios.envelope",
                (
                    str(receipt.path),
                    receipt.content_id,
                    json.dumps(expected),
                    str(args.repeats),
                    str(args.target_error),
                ),
                deadline=deadline,
                ledger=StageLedger(
                    record,
                    deadline=deadline,
                    cold_phase="fresh_process_source_allfield_history_continued_PDE",
                ),
            )
            timing = {"wall_seconds": perf_counter() - cold_started}
        except ColdWorkerTimeout as error:
            raise phx.meshing.MeshingFailure(
                phx.meshing.MeshingFailureCategory.TIMED_OUT,
                "Cold envelope process exhausted the wall allowance.",
                stage=phx.meshing.MeshingStageKind.CERTIFICATION.value,
                requested=(("remaining_wall_seconds", remaining),),
            ) from error
        record["stages_seconds"][record["active_phase"]] = timing["wall_seconds"]
        # Worker return-code evidence is retained by StageLedger.
        # Parsed and process-validated by run_cold_worker.
        if (
            not isinstance(cold, dict)
            or cold.get("passed") is not True
            or cold.get("process_id") == os.getpid()
        ):
            raise qualification_failure(
                "_envelope_qualification_cold_reanalysis",
                (
                    "Envelope cold certificate did not originate in an independent process."
                ),
                {"cold": cold},
            )
        record["checkpoint"] = {
            **cold,
            "content_id": receipt.content_id,
            "container_bytes": receipt.path.stat().st_size,
            "archive_limits": {
                "members": limits.max_members,
                "nesting": limits.max_manifest_nesting,
                "rank": limits.max_array_rank,
            },
            "source_authority_stripped": False,
        }


def _envelope_qualification_lifecycle(args: QualificationControls) -> dict[str, object]:
    """Bounded native repair → strict volume → transferred-state diffusion."""

    if (
        args.resolution < 4
        or args.capacity < 1
        or args.repeats < 1
        or args.adaptation_rounds < 1
        or not np.isfinite(args.timeout)
        or args.timeout <= 0.0
        or not np.isfinite(args.target_error)
        or args.target_error <= 0.0
    ):
        raise ValueError(
            f"_envelope_qualification_lifecycle.bounds must have resolution at least four, positive capacity, repeats, finite timeout and target error, and at least one adaptation round; got {(args.resolution, args.capacity, args.repeats, args.adaptation_rounds, args.timeout, args.target_error)!r}."
        )
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    phases: list[NativeMeshingPhaseMeasurement] = []
    record: dict[str, Any] = {
        "scenario": "native-envelope-lifecycle",
        "engine": "native",
        "status": "failed",
        "runtime": _runtime_identity(),
        "expected_admission": "positive",
        "adaptation_engine": "native-mixed-family-preserving-templates",
        "targets": {"l2_error": args.target_error, "wall_seconds": args.timeout},
        "control_capacities": {
            "resolution": args.resolution,
            "entity_capacity": args.capacity,
            "adaptation_rounds": args.adaptation_rounds,
            "warm_repeats": args.repeats,
        },
        "epochs": [],
        "stages_seconds": {},
        "time_to_target_error_seconds": None,
        "mandatory_workflow_completion": "incomplete",
        "release_evidence": "unassessed",
        "leadership_evidence": "unassessed",
        "active_phase": "source_preparation",
        "manufactured_solution": {
            "exact": "|x|^2",
            "forcing": -6.0,
            "diffusion": 1.0,
            "norm": "independent-exact-P1-quadratic-L2",
            "domain": "repaired-envelope-volume",
        },
        "nonclaims": [
            "No intended dirty-shell interior is inferred.",
            "Topology-change permission is not a homeomorphism theorem.",
            "This routine case is not the full mandatory W15 corpus.",
            "No release or performance leadership follows from this case.",
        ],
    }
    _ledger = StageLedger(record, deadline=deadline)
    try:
        source, result, specification, options = _prepare_envelope_volume(
            args, record, _ledger, phases, deadline=deadline
        )
        _envelope_qualification_epochs(
            source,
            result,
            specification,
            args,
            record,
            started,
            deadline=deadline,
            generation_options=options,
        )
        elapsed = perf_counter() - started
        final_error = record["epochs"][-1]["l2_error"]
        record.update(
            {
                "status": (
                    "timeout"
                    if elapsed > args.timeout
                    else "passed"
                    if final_error <= args.target_error
                    else "failed"
                ),
                "physical_error": final_error,
                "target_error_reached": final_error <= args.target_error,
                "independent_scientific_certification": "passed",
                "mandatory_workflow_completion": (
                    "complete"
                    if elapsed <= args.timeout and final_error <= args.target_error
                    else "incomplete"
                ),
                "implemented_admitted_workflow": "repaired-envelope-allfield-history-transfer-continued-PDE-cold",
            }
        )
    except phx.meshing.MeshingFailure as failure:
        record_failure(record, failure, record["active_phase"])
    except (RuntimeError, ValueError, TypeError, TimeoutError) as failure:
        record_failure(record, failure, record["active_phase"])
        if (
            record["active_phase"].startswith("adaptation_")
            or record["active_phase"] == "source_association_transfer_preparation"
        ):
            record["positive_prerequisite"] = (
                "The exact repaired-PLC source binding and canonical lineage transfer "
                "must satisfy the requested association/domain contract without source stripping."
            )
    finally:
        record["native_phases"] = _phase_recordings(phases)
        record["end_to_end_seconds"] = perf_counter() - started
        record["peak_process_resident_bytes"] = peak_resident_bytes(resource.RUSAGE_SELF)
        record["process_peak_scope"] = "process-lifetime"
        record["device_memory"] = [
            {
                "device_id": device.id,
                "platform": device.platform,
                "device_kind": device.device_kind,
                "status": "unavailable" if statistics is None else "backend-reported",
                "peak_bytes": None
                if statistics is None
                else statistics.get("peak_bytes_in_use"),
                "memory_statistics": statistics,
            }
            for device in jax.local_devices()
            for statistics in (device.memory_stats(),)
        ]
        record.pop("active_phase")
    return record


def worker(argv: tuple[str, ...]) -> dict[str, Any]:
    """Decode process arguments and run the scenario-owned cold consumer."""
    return _envelope_qualification_cold_worker(
        argv[0], argv[1], tuple(json.loads(argv[2])), int(argv[3]), float(argv[4])
    )


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] != "worker":
        raise ValueError(
            "qualification.subcommand must equal worker; got " + repr(sys.argv[1:])
        )
    print(json.dumps(worker(tuple(sys.argv[2:]))))


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return _envelope_qualification_lifecycle(args)
