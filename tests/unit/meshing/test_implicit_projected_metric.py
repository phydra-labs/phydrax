# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Fresh public torus witnesses, never saved meshes or full lifecycles.

``test_offset_torus_one_pass_diagnostic`` is the bounded smoke selector: the
registered coarsest carrier and one metric pass, without a convergence claim.
The other metric selectors deliberately retain the expensive finest-carrier,
16-pass registered qualification request and its original convergence gates.
"""
from __future__ import annotations

from time import perf_counter
from typing import Any, TYPE_CHECKING, TypedDict

import numpy as np
import pytest

import phydrax as phx
from tools.meshing_qualification._controls import (
    authored_resource_envelope,
    QualificationControls,
)
from tools.meshing_qualification._runtime import Deadline
from tools.meshing_qualification.scenarios import implicit


if TYPE_CHECKING:
    from jax import Array


class _TorusEpoch(TypedDict):
    args: QualificationControls
    source: phx.meshing.CellMeshingResult
    volume: phx.meshing.PreparedImplicitVolume
    solved: implicit._ImplicitSolvedEpoch
    adaptation: phx.meshing.MeshAdaptationResult
    record: dict[str, Any]
    deadline: Deadline


class _TorusQualificationEpoch(_TorusEpoch):
    prepared: implicit._ImplicitPreparedSource


class _TorusDiagnosticEpoch(_TorusEpoch):
    request: phx.meshing.VolumeMeshingSpec


def _controls() -> QualificationControls:
    case = "offset-ring-torus-anisotropic"
    envelope = authored_resource_envelope(
        "native-implicit-lifecycle", "native-implicit-lifecycle",
        {"implicit_case": case}, larger=False,
    )
    assert envelope is not None
    return QualificationControls.from_record({
        **QualificationControls().to_record(),
        "scenario": "native-implicit-lifecycle",
        "implicit_case": case,
        "adaptation_rounds": implicit.IMPLICIT_QUALIFICATION_CASES[case].minimum_adaptation_rounds,
        **envelope["effective"],
    })


@pytest.fixture(scope="module")
def torus_metric_epoch() -> _TorusQualificationEpoch:
    args = _controls()
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record: dict[str, Any] = {
        "stages_seconds": {}, "epochs": [], "time_to_target_error_seconds": None,
    }
    prepared = implicit._implicit_prepare_source(args, record, deadline=deadline)
    prepared = implicit._implicit_discover_source(
        args, prepared, record, deadline=deadline,
    )
    # Publish the actual finest request, without paying for the coarser ladder
    # or claiming that this bounded witness qualified the entire lifecycle.
    source, volume, _ = implicit._implicit_publish_level(
        prepared, prepared.request, record, [], deadline=deadline, suffix="",
    )
    solved = implicit._implicit_solve_epoch(
        prepared.geometry, source, prepared.request, prepared.quality,
        args, record, started, 0, source, None, deadline=deadline,
    )
    before = phx.array_tree_fingerprint((source, solved.values))
    adaptation = implicit._implicit_adapt_epoch(
        prepared.geometry, source, prepared.request, prepared.quality,
        args, record, 0, source, solved.row, prepared.control, deadline=deadline,
    )
    assert phx.array_tree_fingerprint((source, solved.values)) == before
    return {
        "args": args, "source": source, "volume": volume,
        "prepared": prepared, "solved": solved, "adaptation": adaptation,
        "record": record, "deadline": deadline,
    }


@pytest.fixture(scope="module")
def torus_one_pass_diagnostic() -> _TorusDiagnosticEpoch:
    """One coarsest-ladder publication/pass; not registered-profile completion."""
    args = _controls()
    started = perf_counter()
    deadline = Deadline(started + args.timeout)
    record: dict[str, Any] = {
        "stages_seconds": {}, "epochs": [], "time_to_target_error_seconds": None,
    }
    prepared = implicit._implicit_prepare_source(args, record, deadline=deadline)
    prepared = implicit._implicit_discover_source(args, prepared, record, deadline=deadline)
    request = prepared.ladder_requests[0]
    source, volume, _ = implicit._implicit_publish_level(
        prepared, request, record, [], deadline=deadline, suffix="_diagnostic",
    )
    solved = implicit._implicit_solve_epoch(
        prepared.geometry, source, request, prepared.quality, args, record,
        started, 0, source, None, deadline=deadline,
    )
    control = request.size_controls[0]
    assert isinstance(control, phx.meshing.UniformSizeControl)
    h = control.target_size / 2.0 ** (1.0 / 3.0)
    anisotropy = implicit._implicit_case(args).metric_anisotropy
    sizes = np.asarray((h, h, h / anisotropy))
    vertices = source.mesh.entity_set(0)
    scope = phx.meshing.MeshingScope(
        source.mesh.mesh_id, source.mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH, 0,
        vertices.entity_set_id, vertices.entity_ids,
    )
    metric = phx.meshing.MeshMetricField(
        scope,
        np.broadcast_to(np.diag(1.0 / sizes**2), (source.mesh.coordinates.shape[0], 3, 3)).copy(),
        minimum_size=float(sizes.min()), maximum_size=h, maximum_anisotropy=anisotropy,
    )
    assert source.certification is not None
    assert source.certification.request.fidelity_tolerance is not None
    boundary = source.certification.request.source
    assert isinstance(boundary, phx.geometry.ImplicitProjectionBoundarySource)
    policy = phx.meshing.MeshAdaptationPolicy(
        phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_3D,
        maximum_passes=1, relocation=True, limits=request.limits,
        association_transfer=phx.meshing.ImplicitAssociationTransfer(
            boundary.profile, maximum_queries=request.limits.maximum_geometry_queries,
        ),
        geometry_transition=phx.discretization.CellGeometryTransitionPolicy(
            reconstruction="source_realization",
            reconstruction_tolerance=2.0 * source.certification.request.fidelity_tolerance,
            maximum_evaluations=request.limits.maximum_work_units,
        ),
    )
    before = phx.array_tree_fingerprint((source, solved.values))
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source, phx.meshing.MetricMeshAdaptation(metric), policy=policy,
        ),
    )
    assert phx.array_tree_fingerprint((source, solved.values)) == before
    return {
        "args": args, "source": source, "volume": volume, "request": request,
        "solved": solved, "adaptation": adaptation, "record": record, "deadline": deadline,
    }


def test_offset_torus_one_pass_diagnostic(torus_one_pass_diagnostic: _TorusDiagnosticEpoch) -> None:
    """Exercise real projected publication and transport, not metric convergence."""
    witness = torus_one_pass_diagnostic
    source, adaptation, solved = witness["source"], witness["adaptation"], witness["solved"]
    assert witness["request"].protected_features[0].maximum_deviation == 0.02
    assert witness["volume"].profile.genus == 1
    assert adaptation.policy.maximum_passes == 1
    assert isinstance(adaptation.request, phx.meshing.MetricMeshAdaptation)
    assert adaptation.target.result_id != source.result_id
    assert adaptation.transition is not None
    chain = adaptation.transition.volume_material_chain
    assert chain is not None
    assert any(edit.curved for edit in chain.edits)
    before = phx.array_tree_fingerprint((source, solved.values))
    carried = implicit._implicit_transfer_epoch(
        source, adaptation, solved, None, 0, witness["args"], witness["record"],
        deadline=witness["deadline"],
    )
    transfer = solved.row["transfer"]
    assert transfer["field_names"] == ["u", "density", "history"]
    assert all(transfer["transport_accepted"])
    assert len(transfer["field_transfers"]) == 3
    assert all(item["passed"] for item in transfer["field_transfers"])
    assert transfer["rollback"]["passed"]
    assert transfer["rollback"]["gate"] == "forced-rejection"
    assert transfer["rollback"]["topology_id"] == source.mesh.topology_id
    assert phx.array_tree_fingerprint((source, solved.values)) == before
    mapped = transfer["mapped_material_reference"]
    assert mapped["density_content_defect"] <= 1e-12
    assert all(value <= 1e-12 for value in mapped["weak_defects"])
    assert all(
        value <= witness["args"].target_error
        for error in mapped["material_L2_errors"] for value in error["maximum_local_L2"]
    )
    _assert_material_inventory(adaptation, source, solved.values, carried.carried, carried.auxiliary)
    assert np.isfinite(transfer["state_L2_error"])


def test_offset_torus_public_metric_changes_certified_domain(torus_metric_epoch: _TorusQualificationEpoch) -> None:
    witness = torus_metric_epoch
    args, prepared = witness["args"], witness["prepared"]
    source, adaptation = witness["source"], witness["adaptation"]
    assert args.resolution == 4
    assert args.implicit_radius == 0.371
    assert args.implicit_fidelity == 0.02
    assert prepared.request.protected_features[0].maximum_deviation == 0.01
    assert args.maximum_work_units == 80_000_000
    assert args.timeout == 2400.0
    assert args.implicit_metric_passes == 16
    assert witness["volume"].profile.genus == 1
    assert witness["record"]["discovery"]["domain_probes_certified"]
    assert isinstance(adaptation.request, phx.meshing.MetricMeshAdaptation)
    assert adaptation.route is phx.meshing.MeshAdaptationRoute.NATIVE_METRIC_3D
    assert adaptation.status.converged
    assert adaptation.compliance.passed
    assert adaptation.transition is not None
    assert adaptation.target.result_id != source.result_id
    chain = adaptation.transition.volume_material_chain
    assert chain is not None
    assert any(edit.curved for edit in chain.edits)
    chain.require_bound(
        source.mesh, source.geometry, adaptation.target.mesh, adaptation.target.geometry,
    )
    metric = witness["solved"].row["metric"]
    assert metric["anisotropy"] == 2.0
    assert metric["tensor_anisotropy"] == 2.0
    assert metric["axis_sizes"][2] == metric["axis_sizes"][0] / 2.0
    certificate = adaptation.target.certification
    assert certificate is not None
    assert isinstance(certificate.request.source, phx.geometry.ImplicitProjectionBoundarySource)
    assert certificate.request.source.profile.genus == 1
    assert np.isfinite(witness["solved"].error)


def test_offset_torus_public_field_history_transition_and_rollback(
    torus_metric_epoch: _TorusQualificationEpoch,
) -> None:
    witness = torus_metric_epoch
    source, solved = witness["source"], witness["solved"]
    before = phx.array_tree_fingerprint((source, solved.values))
    transferred = implicit._implicit_transfer_epoch(
        source, witness["adaptation"], solved, None, 0, witness["args"],
        witness["record"], deadline=witness["deadline"],
    )
    transfer = solved.row["transfer"]
    assert transfer["field_names"] == ["u", "density", "history"]
    assert all(transfer["transport_accepted"])
    assert len(transfer["field_transfers"]) == 3
    assert all(item["passed"] for item in transfer["field_transfers"])
    rollback = transfer["rollback"]
    assert rollback["passed"]
    assert rollback["gate"] == "forced-rejection"
    assert rollback["topology_id"] == source.mesh.topology_id
    assert rollback["transfer_ids"] == [
        item["transfer_id"] for item in transfer["field_transfers"]
    ]
    assert phx.array_tree_fingerprint((source, solved.values)) == before
    assert transferred.current.result_id == witness["adaptation"].target.result_id
    assert np.all(np.isfinite(np.asarray(transferred.carried)))
    assert len(transferred.auxiliary) == 2
    assert all(np.all(np.isfinite(np.asarray(field))) for field in transferred.auxiliary)
    assert transfer["auxiliary_seed_origin"] == "authored-initial-fields"
    assert np.isfinite(transfer["state_L2_error"])
    assert solved.row["prepared_field_query"]["query_id"]
    _assert_material_inventory(
        witness["adaptation"], source, solved.values, transferred.carried, transferred.auxiliary,
    )
    mapped = transfer["mapped_material_reference"]
    assert mapped is not None
    assert mapped["density_content_defect"] <= 1e-12
    assert all(defect <= 1e-12 for defect in mapped["weak_defects"])
    assert mapped["material_error_tolerance"] == witness["args"].target_error
    assert all(
        value <= witness["args"].target_error
        for error in mapped["material_L2_errors"] for value in error["maximum_local_L2"]
    )


@pytest.mark.parametrize("parent", ("missing", "foreign"))
def test_offset_torus_field_transition_refuses_wrong_parent(
    torus_metric_epoch: _TorusQualificationEpoch, parent: str,
) -> None:
    import equinox as eqx

    from phydrax.meshing.qualification import transition_envelope_state
    from tools.meshing_qualification._runtime import qualification_failure

    witness = torus_metric_epoch
    source, solved = witness["source"], witness["solved"]
    adaptation = witness["adaptation"]
    before = phx.array_tree_fingerprint((source, solved.values))
    if parent == "missing":
        malformed = eqx.tree_at(lambda value: value.transition, adaptation, None)
    else:
        malformed = eqx.tree_at(
            lambda value: value.source, adaptation, adaptation.target,
        )
    with pytest.raises(ValueError):
        transition_envelope_state(
            source, malformed, solved.problem, solved.values, 0,
            projection_policy=phx.geometry.CommonRefinementPolicy(
                overlap_simplices=True,
                maximum_candidate_pairs=256 * witness["args"].capacity,
                maximum_accepted_pairs=256 * witness["args"].capacity,
                maximum_memory_bytes=witness["args"].maximum_memory_bytes,
            ),
            failure_factory=qualification_failure,
            material_error_tolerance=witness["args"].target_error,
        )
    assert phx.array_tree_fingerprint((source, solved.values)) == before


def _tetra_volumes(mesh: phx.discretization.CellMesh) -> np.ndarray:
    corners = np.asarray(mesh.coordinates)[
        np.concatenate([np.asarray(block.vertices) for block in mesh.blocks])
    ]
    determinants = np.linalg.det(corners[:, 1:] - corners[:, :1])
    assert np.all(determinants > 0.0)
    return determinants / 6.0


def _assert_material_inventory(
    adaptation: phx.meshing.MeshAdaptationResult,
    source: phx.meshing.CellMeshingResult,
    solved_values: Array,
    carried: Array,
    auxiliary: tuple[Array, Array],
) -> None:
    """Independently replay every carried role through the actual material chain.

    The NumPy reference re-derives pieces and local solves from the declared
    chain and the authored source values, never from production FE rows.
    """
    from phydrax.discretization.fem._topology_transfer import (
        SourceRealizationFieldSemantics,
    )
    from phydrax.meshing.qualification import envelope_transport_field_templates
    from phydrax.meshing.qualification._envelope import material_chain_reference

    transition = adaptation.transition
    assert transition is not None and transition.volume_material_chain is not None
    chain = transition.volume_material_chain
    fields = tuple(
        field for _, field in envelope_transport_field_templates("tetrahedron", solution_degree=1)
    )
    names = tuple(field.name for field in fields)
    semantics: dict[str, SourceRealizationFieldSemantics] = {
        "u": "intensive", "density": "conservative-density", "history": "intensive",
    }

    def vertex_values(result: phx.meshing.CellMeshingResult, values: tuple[Array, ...]) -> np.ndarray:
        space = phx.discretization.FiniteElementPlan(
            result.mesh, fields, coordinate_spec=result.geometry,
        ).prepare()
        cells = np.concatenate([np.asarray(block.vertices) for block in result.mesh.blocks])
        rows = np.zeros((result.mesh.coordinates.shape[0], len(values)))
        for index, field in enumerate(values):
            routes = np.concatenate([np.asarray(r) for r in space.dof_maps[index].cell_dofs])
            rows[cells.ravel(), index] = np.asarray(field)[routes].ravel()
        return rows

    coordinates = np.asarray(source.mesh.coordinates)
    authored = vertex_values(source, (solved_values,))[:, 0]
    initial = np.column_stack((authored, np.full_like(authored, 2.0), coordinates[:, 0] + 0.5))
    source_ids = np.asarray(source.mesh.vertex_global_ids).tolist()
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    limits = adaptation.policy.limits
    with HostBatchTicket(limits.maximum_work_units, bytes_upper=limits.maximum_scratch_bytes) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.activate():
            reference = material_chain_reference(
                chain,
                {
                    name: {vertex: initial[row, index:index + 1] for row, vertex in enumerate(source_ids)}
                    for index, name in enumerate(names)
                },
                semantics,
                budget=budget,
            )
    assert reference.coverage_ratio <= 1.0
    target = adaptation.target
    actual = vertex_values(target, (carried, *auxiliary))
    for index, name in enumerate(names):
        expected = np.asarray([
            reference.values[name][vertex][0]
            for vertex in np.asarray(target.mesh.vertex_global_ids).tolist()
        ])
        np.testing.assert_allclose(actual[:, index], expected, rtol=0.0, atol=1e-12)
    cells = np.concatenate([np.asarray(block.vertices) for block in target.mesh.blocks])
    np.testing.assert_allclose(
        _tetra_volumes(target.mesh) @ actual[cells, 1].mean(axis=1),
        2.0 * np.sum(_tetra_volumes(source.mesh)), rtol=1e-12, atol=0.0,
    )


def test_offset_torus_mapped_fv_preserves_actual_contents(torus_metric_epoch: _TorusQualificationEpoch) -> None:
    from phydrax.discretization import (
        prepare_unstructured_conservative_remap,
        UnstructuredFiniteVolumePlan,
    )

    witness = torus_metric_epoch
    adaptation = witness["adaptation"]
    assert adaptation.transition is not None
    chain = adaptation.transition.volume_material_chain
    assert chain is not None
    source = UnstructuredFiniteVolumePlan.from_cell_mesh(
        witness["source"].mesh,
    ).prepare(cell_geometry=witness["source"].geometry)
    cells = np.concatenate([
        np.asarray(block.vertices) for block in witness["source"].mesh.blocks
    ])
    centers = np.mean(np.asarray(witness["source"].mesh.coordinates)[cells], axis=1)
    values = np.column_stack((2.0 + centers[:, 0], 0.5 + centers[:, 2]))
    original = np.sum(np.asarray(source.cell_volumes)[:, None] * values, axis=0)
    before = phx.array_tree_fingerprint((witness["source"], values))
    policy = phx.geometry.CommonRefinementPolicy(
        overlap_simplices=True,
        maximum_candidate_pairs=256 * witness["args"].capacity,
        maximum_accepted_pairs=256 * witness["args"].capacity,
        maximum_memory_bytes=witness["args"].maximum_memory_bytes,
    )
    target = UnstructuredFiniteVolumePlan.from_cell_mesh(
        chain.target_mesh,
    ).prepare(cell_geometry=chain.target_geometry)
    remap = prepare_unstructured_conservative_remap(
        source, target, provenance=chain.chain_id, policy=policy,
        source_geometry=chain.source_geometry,
        target_geometry=chain.target_geometry,
        volume_material_chain=chain,
    )
    assert remap.plan is not None
    contents = np.asarray(source.cell_volumes)[:, None] * values
    next_values = np.asarray(remap.plan.apply(values))
    next_contents = np.asarray(remap.plan.apply_content(contents))
    np.testing.assert_allclose(
        np.asarray(target.cell_volumes)[:, None] * next_values,
        next_contents, rtol=0.0, atol=1e-12,
    )
    np.testing.assert_allclose(
        np.sum(next_contents, axis=0), original, rtol=0.0, atol=1e-12,
    )
    assert target.mesh.mesh_id == adaptation.target.mesh.mesh_id
    # Remaps must not mutate the accepted source or its authored payload.
    initial_values = np.column_stack((2.0 + centers[:, 0], 0.5 + centers[:, 2]))
    assert phx.array_tree_fingerprint((witness["source"], initial_values)) == before
