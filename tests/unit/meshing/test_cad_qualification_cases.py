#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Never

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

import phydrax as phx
from phydrax._array_archive import ArrayArchiveCorruptionError
from phydrax._fingerprint import canonical_fingerprint
from phydrax._meshcore import meshcore_available, NativeExecutionScope
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.fem._reference_types import FiniteElementSpec
from phydrax.meshing._curving import straight_geometry
from tests._support.coordinate_references import native_meshing_request
from tools.meshing_qualification._controls import (
    archive_limit_values,
    qualification_archive_limits,
    QualificationControls,
)
from tools.meshing_qualification._runtime import ColdControlPacket, Deadline
from tools.meshing_qualification.scenarios import cad as cad_owner
from tools.meshing_qualification.scenarios.cad import (
    _cad_interchange_policies,
    _cross_cylinder_volume,
    _cross_loop_length,
    _radial_hole_volume,
    _washer_volume,
    CAD_EXTERNAL_FIXTURE_PATH,
    load_cad_external_fixture,
    load_cad_quadric_fixture,
)


def test_cad_import_is_model_only_without_changing_public_defaults_or_limits() -> None:
    capacity = 20000
    contract = phx.SpatialCoordinateContract.si()
    imported, exported = _cad_interchange_policies(capacity, contract)
    assert imported.tessellation is not None
    assert not imported.tessellation.realize
    assert imported.pcurve_fit is None
    assert imported.limits.max_losses == 0
    assert imported.limits.max_bytes == 4096 * capacity
    assert imported.limits.max_nodes == 64 * capacity
    assert imported.limits.max_attributes == 512 * capacity
    assert imported.maximum_occurrences == capacity
    assert exported.maximum_entities == 64 * capacity
    default = phx.interchange.CadImportPolicy(
        contract, imported.limits, maximum_occurrences=capacity
    )
    assert default.tessellation is None
    assert default.policy_id != imported.policy_id


@pytest.mark.parametrize(
    "load_fixture",
    (
        pytest.param(load_cad_quadric_fixture, id="authored-quadric"),
        pytest.param(load_cad_external_fixture, id="external-nist"),
    ),
)
def test_loaded_cad_fixture_preserves_its_authored_record_identity(
    load_fixture: Callable[[], dict[str, Any]],
) -> None:
    fixture = load_fixture()
    body = {
        key: value
        for key, value in fixture.items()
        if key not in ("record_id", "file_sha256", "source_path")
    }
    assert canonical_fingerprint(body) == fixture["record_id"]


def test_external_fixture_is_the_unmodified_declared_nist_file() -> None:
    fixture = load_cad_external_fixture()
    raw = Path(fixture["source_path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == fixture["source_sha256"]
    assert raw.startswith(b"ISO-10303-21;")
    assert b"nist_ftc_11" in raw
    assert fixture["provenance"]["archive_sha256"] == (
        "8fa78429e6d8d9b0d7681d223b6aa9ec98c3772185c55b1a0e3679b21c181911"
    )


def test_tampered_external_descriptor_is_refused(tmp_path: Path) -> None:
    value = json.loads(CAD_EXTERNAL_FIXTURE_PATH.read_text())
    value["independent_reference"]["flat_z"] = 0.5
    path = tmp_path / CAD_EXTERNAL_FIXTURE_PATH.name
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="record fingerprint"):
        load_cad_external_fixture(path)


def test_quadric_oracle_matches_independent_limits() -> None:
    fixture = load_cad_quadric_fixture()
    radii = [float(operand["radius"]) for operand in fixture["operands"]]
    assert radii[1] < radii[0]
    # Equal radii give the Steinmetz bicylinder 16 R³ / 3.
    assert _cross_cylinder_volume(1.0, 1.0 - 1e-12) == pytest.approx(16.0 / 3.0, rel=1e-6)
    # A thin transverse cylinder crosses a nearly flat wall: length → 2πr.
    assert _cross_loop_length(1.0e6, 1.0) == pytest.approx(2.0 * math.pi, rel=1e-9)
    # Monte Carlo cross-check of the actual authored configuration.
    generator = np.random.default_rng(7)
    samples = generator.uniform(-1.0, 1.0, size=(400_000, 3)) * np.array(
        [radii[0], radii[1], radii[1]]
    )
    inside = (np.hypot(samples[:, 0], samples[:, 1]) <= radii[0]) & (
        np.hypot(samples[:, 1], samples[:, 2]) <= radii[1]
    )
    box = 8.0 * radii[0] * radii[1] ** 2
    estimate = box * float(np.mean(inside))
    assert _cross_cylinder_volume(*radii) == pytest.approx(estimate, rel=1e-2)


def test_external_oracle_matches_monte_carlo_profile() -> None:
    fixture = load_cad_external_fixture()
    reference: dict[str, Any] = fixture["independent_reference"]
    generator = np.random.default_rng(11)
    count = 400_000
    rho = np.sqrt(generator.uniform(16.0**2, 31.5**2, size=count))
    z = generator.uniform(-1.5, 1.5, size=count)
    top = np.full(count, 0.425)
    for bead in (reference["inner_bead"], reference["outer_bead"]):
        center, minor = bead["center_radius"], bead["minor_radius"]
        near = np.abs(rho - center) <= minor
        arc = np.sqrt(np.maximum(minor**2 - (rho - center) ** 2, 0.0))
        within = near & (
            (rho <= reference["flat_inner_radius"]) | (rho >= reference["flat_outer_radius"])
        )
        top = np.where(within, arc, top)
    shell = math.pi * (31.5**2 - 16.0**2) * 3.0
    estimate = shell * float(np.mean(z <= top))
    assert _washer_volume(reference) == pytest.approx(estimate, rel=1e-2)


def test_radial_hole_volume_respects_annular_chord_bounds() -> None:
    fixture = load_cad_external_fixture()
    reference = fixture["independent_reference"]
    hole = fixture["authored_hole"]
    radius = hole["radius"]
    inner, outer = reference["inner_radius"], reference["outer_radius"]
    assert reference["bottom_z"] < hole["center_z"] - radius
    assert hole["center_z"] + radius < 0.0
    cross_section = math.pi * radius**2
    lower = 2.0 * cross_section * (outer - inner)
    upper = 2.0 * cross_section * (
        math.sqrt(outer**2 - radius**2) - math.sqrt(inner**2 - radius**2)
    )
    # Annular chords grow away from the radial centerline; the wall thickness
    # gives a lower bound, not the upper bound of the removed volume.
    assert lower < _radial_hole_volume(reference, hole) < upper


def test_cad_boolean_borrows_one_source_stage_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = QualificationControls(
        scenario="native-cad-lifecycle", cad_case="quadric-routine"
    )
    contract = phx.SpatialCoordinateContract.si()
    tessellation = phx.geometry.brep.BRepTessellationPolicy(realize=False)
    first = phx.geometry.brep.brep_box(
        (0.0, 0.0, 0.0), (2.0, 2.0, 2.0),
        coordinate_contract=contract, tessellation=tessellation,
    )
    second = phx.geometry.brep.brep_box(
        (1.0, 1.0, 1.0), (3.0, 3.0, 3.0),
        coordinate_contract=contract, tessellation=tessellation,
    )
    limits = cad_owner._cad_native_limits(args)

    def unexpected_scope(*args: object, **kwargs: object) -> Never:
        raise AssertionError("A borrowed source budget must not create another scope.")

    monkeypatch.setattr(cad_owner, "NativeExecutionScope", unexpected_scope)
    with NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(
            limits.maximum_work_units,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
            cavity_upper=limits.maximum_cavity_cells,
        ) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with ticket.native_target(), budget.activate():
                before = ticket.remaining_work_units
                model, record = cad_owner._cad_boolean(
                    first, second, "intersection", args, coordinate_budget=budget
                )
                assert ticket.remaining_work_units < before
                assert model.topology.num_solids == 1
                assert record["execution_scope"] == "shared-source-stage"
                assert "execution" not in record


def _small_p2_carrier(degree: int = 2) -> phx.meshing.CellMeshingResult:
    points = np.asarray(
        ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (.25, .25, .25))
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 4), (0, 3, 1, 4), (0, 2, 3, 4), (1, 3, 2, 4)))
    )
    if degree == 1:
        geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    else:
        policy = phx.meshing.HighOrderCurvingPolicy(
            degree=degree, fidelity_tolerance=0.01
        )
        geometry = straight_geometry(
            mesh, degree,
            maximum_nodes=policy.maximum_geometry_nodes,
            maximum_entries=policy.maximum_tabulation_entries,
            maximum_bernstein_nodes=policy.validity.maximum_bernstein_nodes,
        )
    return phx.meshing.certify_cell_mesh(
        mesh,
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
    )


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore library is unavailable")
@pytest.mark.parametrize("large_metadata", (False, True), ids=("ordinary", "large-publication"))
def test_cad_p2_solve_retains_real_coefficients_and_continues_the_accepted_state(
    large_metadata: bool,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.lifecycle._meshing_sources import (
        meshing_source_owner_reconstruction_scope,
    )

    carrier = _small_p2_carrier()
    if large_metadata:
        carrier = phx.meshing.CellMeshingResult(
            carrier.mesh, carrier.geometry, carrier.coordinate_contract, carrier.audit,
            carrier.quality, carrier.compliance, carrier.trace, carrier.provider,
            carrier.runtime, carrier.derivative_mode,
            phx.SemanticProvenance({
                "kind": "large-physical-publication",
                "description": "x" * DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes,
            }),
            boundary=carrier.boundary, patches=carrier.patches,
        )
    args = QualificationControls(
        scenario="native-cad-lifecycle", cad_case="quadric-routine",
        resolution=4, capacity=20000, timeout=120., repeats=3, target_error=.05,
    )
    # This small algebraic fixture tests the physical owner, not CAD fidelity.
    case: Any = SimpleNamespace(manufactured_length=1., reference_volume=1. / 6.)
    deadline = Deadline(perf_counter() + args.timeout)
    with native_meshing_request(cad_owner._cad_native_limits(args)) as (scope, budget, _):
        with (
            budget.bind_ticket(scope.ticket_view()),
            meshing_source_owner_reconstruction_scope(budget=budget),
        ):
            accepted, continued, solve = cad_owner._cad_p2_solve(
                carrier, case, args, deadline=deadline, budget=budget
            )
            assert budget.work_units > 0
    assert accepted.carrier is carrier
    assert set(accepted.fields) == {
        "u", "boundary-mask", "boundary-values", "accepted-epoch"
    }
    declaration = accepted.declarations[0]
    assert isinstance(declaration, phx.lifecycle.MeshingFiniteElementFieldDeclaration)
    assert declaration.finite_element_fields[0].field_spec_id == (
        cad_owner._cad_p2_field().field_spec_id
    )
    assert declaration.finite_element_fields[0].elements[0].degree == 2
    assert declaration.maximum_derivative_orders["u"] == 1
    space, problem, free = cad_owner._cad_prepare_p2_solver(
        carrier, case.manufactured_length, declaration
    )
    assert free.size > 0
    assert np.any(np.asarray(accepted.fields["u"])[np.asarray(free)] != 0.)
    assert solve["compiler"]["solver"]["successful"]
    assert solve["continuation"]["compiler"]["solver"]["successful"]
    assert solve["continuation"]["seed_fingerprint"] == phx.array_tree_fingerprint(
        jnp.asarray(accepted.fields["u"])[free]
    )
    assert solve["continuation"]["solution_fingerprint"] == phx.array_tree_fingerprint(continued)
    np.testing.assert_array_equal(accepted.fields["boundary-values"], problem.lift)
    assert declaration.field_space_ids["u"] == space.field_spaces[0].field_space_id


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore library is unavailable")
def test_cad_p2_solver_refuses_a_p1_carrier() -> None:
    with pytest.raises(RuntimeError, match="actual P2"):
        cad_owner._cad_prepare_p2_solver(_small_p2_carrier(1), 1.)


@pytest.fixture(scope="module")
def cad_physical_checkpoint(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, str, dict[str, Any], QualificationControls]:
    """Produce the actual source→P2→solve→independent-cold path once, without mocks."""
    if not meshcore_available():
        pytest.skip("native meshcore library is unavailable")
    directory = tmp_path_factory.mktemp("cad-physical-checkpoint")
    published: list[tuple[Path, str]] = []
    original_write = cad_owner.write_meshing_source_closure

    def retain_archive(path: Path, records: Any, **kwargs: Any) -> Any:
        receipt = original_write(path, records, **kwargs)
        retained = directory / f"physical-{len(published)}.zip"
        shutil.copyfile(receipt.path, retained)
        published.append((retained, receipt.content_id))
        return receipt

    args = QualificationControls(
        scenario="native-cad-lifecycle", cad_case="quadric-routine", cad_format="step",
        resolution=4, capacity=20000, timeout=3600., target_error=.05,
        adaptation_rounds=1, repeats=3, maximum_work_units=1_000_000_000,
        maximum_recipe_bytes=8 * 1024**2,
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cad_owner, "write_meshing_source_closure", retain_archive)
        record = cad_owner.run(args)
    (directory / "qualification.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    assert record["status"] == "passed", record
    assert record["case_contract_completion"] == "complete", record
    assert len(published) == 1
    return *published[0], record, args


def test_cad_physical_archive_restores_the_actual_p2_epoch_bitwise(
    cad_physical_checkpoint: tuple[Path, str, dict[str, Any], QualificationControls],
) -> None:
    path, content_id, record, controls = cad_physical_checkpoint
    records = phx.lifecycle.read_meshing_source_closure(
        path, expected_content_id=content_id, limits=qualification_archive_limits(controls)
    )
    history = phx.lifecycle.validate_meshing_accepted_epochs(records)
    assert set(records["accepted_data"]) == {"accepted_epochs"}
    assert "generation_part" not in records
    assert "adaptation_results" not in records["accepted_data"]
    assert records["accepted_target"].result_id == history[-1].carrier.result_id
    assert isinstance(records["generation_source"], phx.meshing.NativeCadVolumeSource)
    assert all(
        isinstance(element, FiniteElementSpec) and element.degree == 2
        for element in history[-1].carrier.geometry.elements
    )
    section = record["resolutions"][0]
    checkpoint = section["physical_checkpoint"]
    cold = checkpoint["fresh_interpreter"]
    assert cold["process_id"] != os.getpid()
    assert checkpoint["accepted_fields_bitwise_preserved"] is True
    assert cold["accepted_field_fingerprint"] == phx.array_tree_fingerprint(history[-1].fields)
    assert cold["accepted_field_fingerprint"] == section["solve"]["accepted_field_fingerprint"]
    assert cold["continued_from_epoch_id"] == history[-1].epoch_id
    assert history[-1].carrier.certification is not None
    assert cold["original_certification_id"] == history[-1].carrier.certification.report_id
    assert cold["renewed_certification_id"]
    assert cold["source_certification_scientifically_preserved"] is True
    assert cold["continuation_seed_fingerprint"] == (
        section["solve"]["continuation"]["seed_fingerprint"]
    )
    assert cold["warm_cold_continuation_bitwise_equal"] is True
    assert cold["continued_solution_fingerprint"] == (
        section["solve"]["continuation"]["solution_fingerprint"]
    )
    assert checkpoint["container_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert not checkpoint["source_authority_stripped"]
    assert not checkpoint["prepared_fe_caches_serialized"]


@pytest.mark.parametrize("fault", ("stale-content-id", "partial-container"))
def test_cad_physical_archive_refuses_stale_or_partial_bytes(
    cad_physical_checkpoint: tuple[Path, str, dict[str, Any], QualificationControls],
    tmp_path: Path,
    fault: str,
) -> None:
    path, content_id, _, controls = cad_physical_checkpoint
    if fault == "stale-content-id":
        content_id = "0" * 64
    else:
        partial = tmp_path / "partial.zip"
        partial.write_bytes(path.read_bytes()[:-128])
        path = partial
    with pytest.raises(ArrayArchiveCorruptionError):
        phx.lifecycle.read_meshing_source_closure(
            path, expected_content_id=content_id, limits=qualification_archive_limits(controls)
        )


def test_cad_cold_refuses_a_foreign_authenticated_physical_epoch_before_solving(
    cad_physical_checkpoint: tuple[Path, str, dict[str, Any], QualificationControls],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, content_id, _, controls = cad_physical_checkpoint
    limits = qualification_archive_limits(controls)
    records = phx.lifecycle.read_meshing_source_closure(
        path, expected_content_id=content_id, limits=limits
    )
    expected = cad_owner._cad_checkpoint_identity(records)
    accepted = records["accepted_data"]["accepted_epochs"][0]
    fields = dict(accepted.fields)
    free_row = int(np.flatnonzero(~np.asarray(fields["boundary-mask"]))[0])
    fields["u"] = jnp.asarray(fields["u"]).at[free_row].add(.125)
    foreign = phx.lifecycle.MeshingAcceptedEpoch(
        accepted.carrier, accepted.declarations, fields
    )
    records = dict(records)
    records["accepted_data"] = {"accepted_epochs": (foreign,)}
    receipt = phx.lifecycle.write_meshing_source_closure(
        tmp_path / "foreign.zip", records, limits=limits
    )

    def unexpected_solve(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Foreign accepted epochs must refuse before a physical solve.")

    monkeypatch.setattr(cad_owner, "_compiled_solver_stages", unexpected_solve)
    packet = ColdControlPacket(
        archive_path=receipt.path,
        content_id=receipt.content_id,
        controls={
            "expected_identity_id": canonical_fingerprint(expected),
            "controls": controls.to_record(),
            "archive_limits": archive_limit_values(qualification_archive_limits(controls)),
        },
        deadline=Deadline(perf_counter() + controls.timeout),
    )
    with pytest.raises(ValueError):
        cad_owner.cold_worker(packet)


def test_cad_accepted_epoch_refuses_a_partial_physical_field_bank(
    cad_physical_checkpoint: tuple[Path, str, dict[str, Any], QualificationControls],
) -> None:
    path, content_id, _, controls = cad_physical_checkpoint
    records = phx.lifecycle.read_meshing_source_closure(
        path, expected_content_id=content_id, limits=qualification_archive_limits(controls)
    )
    accepted = records["accepted_data"]["accepted_epochs"][0]
    partial = {name: array for name, array in accepted.fields.items() if name != "u"}
    with pytest.raises(ValueError):
        phx.lifecycle.MeshingAcceptedEpoch(
            accepted.carrier, accepted.declarations, partial
        )


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore library is unavailable")
@pytest.mark.parametrize("on_surface", [False, True], ids=["off-surface", "on-surface"])
def test_straight_rejection_uses_fresh_resolved_face_distance(on_surface: bool) -> None:
    from phydrax.geometry._embedding_state import (
        coordinate_certificate_request,
        MeshCertificateLimits,
    )
    from phydrax.meshing._association import (
        associate_mesh_vertices,
        AssociationPropagationPolicy,
    )

    model = phx.geometry.brep.brep_cylinder(
        1.0,
        1.0,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        tessellation=phx.geometry.brep.BRepTessellationPolicy(realize=False),
    )
    projection = phx.geometry.brep.prepare_brep_projection(model)
    points = np.asarray(
        ((1.0, 0.0, 0.25), (0.0, 1.0, 0.25), (-1.0, 0.0, 0.25), (0.0, -1.0, 0.75)),
        dtype=np.float64,
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2, fidelity_tolerance=1e-3)
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        association = associate_mesh_vertices(
            mesh, projection, policy=AssociationPropagationPolicy(), budget=budget
        )
        straight = straight_geometry(mesh, 2)
        surface_points = np.asarray(straight.coordinates).copy()
        radii = np.hypot(surface_points[:, 0], surface_points[:, 1])
        nonzero = radii > 0
        surface_points[nonzero, :2] /= radii[nonzero, None]
        surface_points[~nonzero, :2] = np.asarray((1.0, 0.0), dtype=np.float64)
        surface = straight.with_coordinates(surface_points)
        if on_surface:
            with pytest.raises(RuntimeError, match="no resolved lower-distance witness"):
                cad_owner._cad_straight_rejection(
                    mesh, association, projection, surface, surface, policy, budget
                )
            return
        witness = cad_owner._cad_straight_rejection(
            mesh, association, projection, straight, surface, policy, budget
        )
    lower, upper = witness["distance_lower_bound"], witness["distance_upper_bound"]
    assert isinstance(lower, float) and isinstance(upper, float)
    assert math.isfinite(lower) and math.isfinite(upper) and lower <= upper
    assert witness["criterion"] == "constrained-node-distance"
    assert witness["bound_scope"] == "single-node-lower-bound-not-global-maximum"
    assert witness["entity_dimension"] == 2
    assert witness["face"] == witness["entity_index"]
    assert witness["query_status"] in (
        phx.geometry.brep.BRepProjectionStatus.UNIQUE,
        phx.geometry.brep.BRepProjectionStatus.SEAM,
    )
    point = np.asarray(witness["query_point"], dtype=np.float64)
    node = witness["node"]
    assert isinstance(node, int)
    np.testing.assert_array_equal(
        point, np.asarray(straight.coordinates)[node]
    )
    analytic_distance = abs(1.0 - math.hypot(float(point[0]), float(point[1])))
    assert witness["residual_tolerance"] == policy.residual_tolerance
    assert lower > policy.residual_tolerance
    assert lower <= analytic_distance + 2e-14
    assert analytic_distance <= upper + 2e-14
    assert witness["source_id"] == model.source_id
    assert witness["source_revision"] == model.source_revision
    assert witness["projection_id"] == projection.projection_id
    assert witness["query_id"] == projection.query.query_id
    assert witness["geometry_layout_id"] == straight.geometry_layout_id


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore library is unavailable")
@pytest.mark.parametrize(
    ("status", "unresolved"),
    (
        pytest.param(
            phx.geometry.brep.BRepProjectionStatus.UNIQUE,
            True,
            id="unresolved",
        ),
        pytest.param(
            phx.geometry.brep.BRepProjectionStatus.FAILED,
            False,
            id="failed-status",
        ),
    ),
)
def test_straight_rejection_refuses_unresolved_or_invalid_query_evidence(
    status: phx.geometry.brep.BRepProjectionStatus,
    unresolved: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry._embedding_state import (
        coordinate_certificate_request,
        MeshCertificateLimits,
    )
    from phydrax.meshing._association import (
        associate_mesh_vertices,
        AssociationPropagationPolicy,
    )

    model = phx.geometry.brep.brep_cylinder(
        1.0,
        1.0,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        tessellation=phx.geometry.brep.BRepTessellationPolicy(realize=False),
    )
    projection = phx.geometry.brep.prepare_brep_projection(model)
    points = np.asarray(
        ((1.0, 0.0, 0.25), (0.0, 1.0, 0.25), (-1.0, 0.0, 0.25), (0.0, -1.0, 0.75)),
        dtype=np.float64,
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        points, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2, fidelity_tolerance=1e-3)
    query_type = phx.geometry.brep.PreparedBRepQuery
    closest_point = query_type.closest_point
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        association = associate_mesh_vertices(
            mesh, projection, policy=AssociationPropagationPolicy(), budget=budget
        )
        straight = straight_geometry(mesh, 2)
        surface_points = np.asarray(straight.coordinates).copy()
        radii = np.hypot(surface_points[:, 0], surface_points[:, 1])
        nonzero = radii > 0
        surface_points[nonzero, :2] /= radii[nonzero, None]
        surface_points[~nonzero, :2] = np.asarray((1.0, 0.0), dtype=np.float64)
        surface = straight.with_coordinates(surface_points)

        def invalid_closest_point(
            query: phx.geometry.brep.PreparedBRepQuery,
            query_points: ArrayLike,
            /,
            *,
            faces: tuple[int, ...] | None = None,
            path: tuple[str, ...] | None = None,
            query_budget: phx.geometry.brep.BRepQueryBudget | None = None,
            preparations: cad_owner.TrimRequestPreparations | None = None,
            coordinate_budget: CoordinateEnclosureBudget | None = None,
        ) -> phx.geometry.brep.BRepSurfaceQueryResult:
            result = closest_point(
                query,
                query_points,
                faces=faces,
                path=path,
                budget=query_budget,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            )
            if coordinate_budget is not budget:
                return result
            return eqx.tree_at(
                lambda value: (value.status, value.unresolved),
                result,
                (
                    jnp.full_like(result.status, int(status)),
                    jnp.full_like(result.unresolved, unresolved),
                ),
            )

        monkeypatch.setattr(query_type, "closest_point", invalid_closest_point)
        with pytest.raises(RuntimeError, match="no resolved lower-distance witness"):
            cad_owner._cad_straight_rejection(
                mesh, association, projection, straight, surface, policy, budget
            )


@pytest.mark.skipif(not meshcore_available(), reason="native meshcore library is unavailable")
def test_straight_rejection_queries_the_assigned_placed_face_occurrence() -> None:
    from phydrax.geometry._embedding_state import (
        coordinate_certificate_request,
        MeshCertificateLimits,
    )
    from phydrax.meshing._association import (
        associate_mesh_vertices,
        AssociationPropagationPolicy,
    )
    from tests._support.cad_models import native_repeated_assembly

    model = native_repeated_assembly()
    geometry = model.geometry
    if geometry is None:
        raise ValueError("Repeated assembly fixture must retain its B-Rep geometry.")
    projection = phx.geometry.brep.prepare_brep_projection(model)
    query_occurrence = next(
        index
        for index, value in enumerate(projection.query.occurrences)
        if value.path == geometry.occurrences[1].path
    )
    coedges = geometry.face_loops[0][0]
    corner_indices = np.asarray(
        [
            geometry.edge_vertices[geometry.coedge_edges[coedge]][
                0 if geometry.coedge_senses[coedge] > 0 else 1
            ]
            for coedge in coedges
        ],
        dtype=np.int32,
    )
    corners = np.asarray(geometry.vertex_points, dtype=np.float64)[corner_indices]
    weights = np.asarray(
        (
            (0.25, 0.25, 0.25, 0.25),
            (0.55, 0.25, 0.10, 0.10),
            (0.10, 0.55, 0.25, 0.10),
        ),
        dtype=np.float64,
    )
    occurrence = geometry.occurrences[1]
    points = occurrence.place(weights @ corners)
    normal = np.cross(points[1] - points[0], points[2] - points[0])
    normal /= np.linalg.norm(normal)
    displaced = np.concatenate(
        (points + 0.1 * normal, points[:1] + 0.15 * normal), axis=0,
    )
    mesh = phx.discretization.CellMesh.from_tetrahedra(
        displaced, np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    )
    policy = phx.meshing.HighOrderCurvingPolicy(degree=2, fidelity_tolerance=1e-3)
    with coordinate_certificate_request(MeshCertificateLimits()) as budget:
        association = associate_mesh_vertices(
            mesh,
            projection,
            policy=AssociationPropagationPolicy(classification_tolerance=0.2),
            budget=budget,
        )
        assert association.source_entities.occurrence_paths() == (occurrence.path,) * 4
        straight = straight_geometry(mesh, 2)
        coordinates = np.asarray(straight.coordinates, dtype=np.float64)
        normal_distance = (coordinates - points[0]) @ normal
        curved = straight.with_coordinates(
            coordinates - normal_distance[:, None] * normal,
        )
        witness = cad_owner._cad_straight_rejection(
            mesh, association, projection, straight, curved, policy, budget
        )
    assert witness["bound_scope"] == "single-node-lower-bound-not-global-maximum"
    assert witness["face"] == witness["entity_index"] == 0
    assert witness["occurrence_path"] == list(occurrence.path)
    assert witness["query_occurrence"] == query_occurrence
    lower = witness["distance_lower_bound"]
    assert isinstance(lower, float)
    assert lower > policy.residual_tolerance
    node = witness["node"]
    assert isinstance(node, int)
    np.testing.assert_array_equal(
        witness["query_point"], np.asarray(straight.coordinates)[node]
    )
    assert witness["source_revision"] == model.source_revision
    assert witness["projection_id"] == projection.projection_id
    assert witness["query_id"] == projection.query.query_id
    assert witness["geometry_layout_id"] == straight.geometry_layout_id


def test_cad_cold_rejects_manifest_policy_drift_before_reading(tmp_path: Path) -> None:
    from time import perf_counter

    from phydrax._array_archive import ArrayArchiveLimits
    from tools.meshing_qualification._controls import (
        archive_limit_values,
        QualificationControls,
    )
    from tools.meshing_qualification._runtime import ColdControlPacket, Deadline
    from tools.meshing_qualification.scenarios.cad import cold_worker

    controls = QualificationControls(
        scenario="native-cad-lifecycle", maximum_recipe_bytes=8 * 1024**2
    )
    packet = ColdControlPacket(
        archive_path=tmp_path / "must-not-be-read.zip",
        content_id="unread-content",
        controls={
            "controls": controls.to_record(),
            "archive_limits": archive_limit_values(ArrayArchiveLimits()),
        },
        deadline=Deadline(perf_counter() + 60.0),
    )
    with pytest.raises(ValueError):
        cold_worker(packet)
