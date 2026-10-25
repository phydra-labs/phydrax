# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Independent moving-overset qualification oracles on hand-checkable inputs."""

from __future__ import annotations

import json
from copy import copy
from fractions import Fraction
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import (
    Deadline,
    qualification_failure,
    QualificationFailure,
)
from tools.meshing_qualification.scenarios.overset import (
    _body_areas,
    _donor_verdict,
    _DonorChart,
    _exact_orientation,
    _manufactured,
    _manufactured_l2_error,
    _measure_bound,
    _oracle_fringe_layers,
    _overset_ale_coordinates,
    _overset_archive_limits,
    _overset_composite_update,
    _overset_conservative_checkpoint,
    _overset_donor_oracle,
    _overset_part_term,
    _OVERSET_PROFILES,
    _overset_qualification_conservative,
    _overset_require_complete_cold_remap,
    _overset_workflow_obligations,
    _OversetColdRemapRefusal,
    _OversetComposite,
    _PartLayout,
    _ReceptorDonors,
    _solve_overset_composite,
    _triangles_touch_rectangle,
    _wall_rectangle,
)


def _rational_orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
    ax, ay = (Fraction(float(value)) for value in a)
    bx, by = (Fraction(float(value)) for value in b)
    cx, cy = (Fraction(float(value)) for value in c)
    exact = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
    return (exact > 0) - (exact < 0)


def test_exact_orientation_matches_rational_signs_near_degeneracy() -> None:
    generator = np.random.default_rng(7)
    base = generator.uniform(-3.0, 3.0, size=(64, 2))
    direction = generator.uniform(-1.0, 1.0, size=(64, 2))
    first = base
    second = base + direction
    # Points on the binary64 rounding of the carrier line plus one-ulp offsets.
    third = base + 0.375 * direction
    third[1::2] = np.nextafter(third[1::2], np.inf)
    signs = _exact_orientation(first, second, third)
    expected = [
        _rational_orientation(np.asarray(a), np.asarray(b), np.asarray(c))
        for a, b, c in zip(first, second, third, strict=True)
    ]
    assert signs.tolist() == expected
    assert _exact_orientation(
        np.asarray([[0.0, 0.0]]), np.asarray([[1.0, 1.0]]), np.asarray([[2.0, 2.0]])
    ).tolist() == [0]


def test_triangle_rectangle_intersection_is_closed_and_exact() -> None:
    rectangle = (0.0, 0.0, 1.0, 1.0)
    triangles = np.asarray(
        (
            ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0)),
            ((np.nextafter(1.0, 2.0), 0.0), (2.0, 0.0), (2.0, 1.0)),
            ((-5.0, -5.0), (10.0, -5.0), (-5.0, 10.0)),
            ((-1.0, 0.5), (2.0, 0.4), (2.0, 0.6)),
            ((0.5, 1.5), (2.0, 0.0), (2.0, 2.0)),
            ((0.25, 0.25), (0.5, 0.25), (0.25, 0.5)),
        ),
        dtype=np.float64,
    )
    assert _triangles_touch_rectangle(triangles, rectangle).tolist() == [
        True,
        False,
        True,
        True,
        True,
        True,
    ]


def test_wall_rectangle_reads_subdivided_axis_aligned_walls_only() -> None:
    points = np.asarray(
        ((0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
    )
    edges = np.asarray(((0, 1), (1, 2), (2, 3), (3, 4), (4, 0)), dtype=np.int64)
    assert _wall_rectangle(points, edges) == (0.0, 0.0, 1.0, 1.0)
    slanted = points.copy()
    slanted[3] = (1.0, 1.25)
    with pytest.raises(QualificationFailure):
        _wall_rectangle(slanted, edges)


def test_oracle_fringe_layers_follow_declared_layer_law() -> None:
    cells = np.asarray(((0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)), dtype=np.int64)
    hole = np.zeros(6, dtype=np.bool_)
    hole[0] = True
    cell_hole = np.any(hole[cells], axis=1)
    empty = np.zeros(6, dtype=np.bool_)
    assert _oracle_fringe_layers(cells, hole, cell_hole, empty, empty, 2).tolist() == [
        0,
        1,
        2,
        1,
        2,
        0,
    ]
    protected = empty.copy()
    protected[3] = True
    boundary = empty.copy()
    boundary[5] = True
    assert _oracle_fringe_layers(
        cells, hole, cell_hole, boundary, protected, 1
    ).tolist() == [0, 1, 0, 0, 0, 1]


def test_manufactured_l2_error_matches_degree_five_quadrature() -> None:
    points = np.asarray(((0.3, -0.2), (1.1, 0.4), (0.2, 0.9)), dtype=np.float64)
    cells = np.asarray(((0, 1, 2),), dtype=np.int64)
    nodal = _manufactured(points) + np.asarray((0.01, -0.02, 0.005))
    a1, b1, w1 = 0.0597158717897698, 0.4701420641051151, 0.1323941527885062
    a2, b2, w2 = 0.7974269853530873, 0.1012865073234563, 0.1259391805448271
    barycentric = [((1.0 / 3.0,) * 3, 0.225)]
    for a, b, weight in ((a1, b1, w1), (a2, b2, w2)):
        barycentric.extend(
            (coordinates, weight) for coordinates in ((a, b, b), (b, a, b), (b, b, a))
        )
    area = 0.5 * abs(
        (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
        - (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
    )
    integral = 0.0
    for coordinates, weight in barycentric:
        lam = np.asarray(coordinates)
        site = lam @ points
        integral += weight * (lam @ nodal - _manufactured(site[None, :])[0]) ** 2
    expected = float(np.sqrt(area * integral))
    assert _manufactured_l2_error(points, cells, nodal) == pytest.approx(
        expected, rel=1.0e-12
    )
    # The manufactured field satisfies -Laplace(u) = 1 by construction.
    assert _manufactured(np.asarray(((0.0, 0.0),)))[0] == 5.0


def test_body_areas_and_analytic_reaction_fluxes() -> None:
    routine = _OVERSET_PROFILES["routine"].bodies[0]
    assert _body_areas(routine) == {
        "domain": 20.0,
        "wall_flux": 0.25,
        "exterior_flux": None,
    }
    assert _body_areas(None) == {
        "domain": 36.0,
        "wall_flux": None,
        "exterior_flux": -36.0,
    }


def test_profiles_preserve_original_routine_schedule_and_cycle_others() -> None:
    routine = _OVERSET_PROFILES["routine"]
    assert routine.fringe_layers == 1
    assert routine.bodies[0].offset(0).tolist() == [0.55, 0.0]
    assert routine.bodies[0].offset(7).tolist() == [0.05, 0.0]
    for name in ("larger", "combined"):
        profile = _OVERSET_PROFILES[name]
        assert len(profile.bodies) == 2
        for body in profile.bodies:
            assert body.offset(2).tolist() == body.offset(0).tolist()
            assert np.allclose(body.offset(0) + body.offset(1), 0.0)
        assert json.loads(json.dumps(profile.record())) == profile.record()


def test_ale_motion_preserves_authored_size_not_only_angle_margin() -> None:
    points = np.asarray([(x, y) for y in (-3.0, 0.0, 3.0) for x in (-3.0, 0.0, 3.0)])
    cells = np.asarray(
        [
            triangle
            for row in range(2)
            for column in range(2)
            for triangle in (
                (3 * row + column, 3 * row + column + 1, 3 * row + column + 4),
                (3 * row + column, 3 * row + column + 4, 3 * row + column + 3),
            )
        ],
        dtype=np.int32,
    )
    mesh = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("cells", "triangle", cells),)
    )
    scope = phx.meshing.MeshingScope(
        "ale-test", "source", phx.meshing.MeshingEntityKind.GEOMETRY, 2, "fluid",
        np.asarray((0,), dtype=np.int64),
    )
    specification = phx.meshing.SurfaceMeshingSpec(
        phx.meshing.CellMeshingTarget(
            2, 2, phx.meshing.CellFamilyPolicy(required=("triangle",))
        ),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
        ),
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 3.0, maximum_size=4.5,
                strength=phx.meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=phx.meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        quality_target=phx.meshing.MeshQualityTarget(
            minimum_angle=np.radians(25.0), hard=True
        ),
    )
    moved, evidence = _overset_ale_coordinates(
        mesh, specification, np.asarray((1.0, 0.0)), 0.55
    )
    # The full amplitude meets the angle margin but exceeds the hard local size.
    assert evidence["halvings"] == 1
    assert evidence["size_compliance_issues"] == []
    assert evidence["deformed_minimum_angle"] > np.radians(35.0)
    connectivity = mesh.connectivity
    assert isinstance(connectivity, phx.discretization.PolygonalConnectivity)
    edges = np.asarray(connectivity.edges)
    assert np.max(np.linalg.norm(moved[edges[:, 1]] - moved[edges[:, 0]], axis=1)) <= 4.5
    boundary = np.unique(edges[np.asarray(connectivity.boundary_edges)])
    np.testing.assert_array_equal(moved[boundary], points[boundary])
    assert moved[4, 0] > points[4, 0]


def test_actual_native_ale_epoch_archive_retains_every_field_and_motion_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.discretization import _mapped_geometry_measure as measure_owner
    from phydrax.discretization._cell_geometry_transfer import (
        transition_displaced_cell_geometry,
    )
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle import _meshing_field_records as records_owner
    from phydrax.lifecycle._meshing_field_records import MeshingGeometryMotionTransition
    from phydrax.lifecycle._meshing_source_records import content_equal
    from phydrax.meshing.qualification import (
        _moving_overset as motion_owner,
        execute_native_consumer,
        execute_native_request,
        native_planar_part,
    )

    measure_calls: list[object] = []
    displacement_calls: list[int] = []
    original_measures = measure_owner.certified_cell_measures

    def measured(*args: Any, **kwargs: Any) -> Any:
        measure_calls.append(args[0])
        return original_measures(*args, **kwargs)

    def displaced(*args: Any, **kwargs: Any) -> Any:
        before_calls = len(measure_calls)
        result = transition_displaced_cell_geometry(*args, **kwargs)
        displacement_calls.append(len(measure_calls) - before_calls)
        return result

    monkeypatch.setattr(measure_owner, "certified_cell_measures", measured)
    monkeypatch.setattr(motion_owner, "transition_displaced_cell_geometry", displaced)
    monkeypatch.setattr(records_owner, "transition_displaced_cell_geometry", displaced)

    background, plan = native_planar_part("background", 1.0, size=0.7)
    moving, _ = native_planar_part("moving", 0.5, size=0.4)
    connectivity = phx.meshing.prepare_overset_connectivity(
        phx.meshing.MeshAssembly((background, moving)),
        (phx.meshing.OversetPartSpec("background"), phx.meshing.OversetPartSpec("moving")),
    )
    controls = QualificationControls()
    record: dict[str, Any] = {"stages_seconds": {}, "active_phase": "initial"}
    accepted = _overset_qualification_conservative(
        connectivity, plan, _OVERSET_PROFILES["routine"], controls, record, [],
        Deadline(perf_counter() + controls.timeout),
    )
    assert displacement_calls == [2]
    motion = accepted.motion
    assert motion.transition_kind is phx.meshing.MeshTransitionKind.MOTION
    assert motion.source_mesh_id == accepted.source.mesh.mesh_id
    assert motion.target.mesh.topology_id == accepted.source.mesh.topology_id
    assert motion.target.mesh.mesh_id != accepted.source.mesh.mesh_id
    assert motion.geometry_transition is not None
    assert motion.geometry_transition.geometry.exact_source is accepted.source.geometry.exact_source
    assert motion.geometry_transition.evidence.measure_exact
    assert motion.geometry_transition.evidence.continuity_residual == 0.0
    assert accepted.prepared_operands is not None
    operands = accepted.prepared_operands
    assert operands._measure_snapshots
    assert motion.target.execution_evidence is not None
    from phydrax.meshing._measurements import (
        validate_native_execution_interchange,
    )
    assert (
        validate_native_execution_interchange(
            motion.target.execution_evidence.to_record()
        )["managed_admission_peak_bytes_upper"]
        is not None
    )
    paid_receipt = motion.target.execution_evidence
    limits = _overset_archive_limits(controls.capacity)

    def prepare_archive(budget: CoordinateEnclosureBudget) -> tuple[dict[str, Any], dict[str, Any], Any]:
        for key, integral in operands._measure_snapshots:
            actual = budget.exact_cell_measure_cache[key]
            for current, expected in zip(actual, integral, strict=True):
                assert type(current) is float and type(expected) is float
                assert float.hex(current) == float.hex(expected)
        from phydrax.lifecycle._meshing_sources import (
            _source_owner_reconstruction_scope,
            meshing_source_owner_reconstruction_scope,
        )

        with meshing_source_owner_reconstruction_scope(
            budget=budget, archive_limits=limits
        ):
            before_calls = len(measure_calls)
            records, expected = _overset_conservative_checkpoint(accepted, plan, budget=budget)
            # Both finite-volume owners and their canonical reconstructions still
            # compute measures. Only the displacement's two reductions disappear:
            # original issuance + warm checkpoint uses two calls, not four.
            assert len(measure_calls) - before_calls == 4
            assert displacement_calls == [2, 0]
            request_scope = _source_owner_reconstruction_scope.get()
            assert request_scope is not None
            owner_keys = frozenset(request_scope.field_owners)
            publication = phx.lifecycle.write_meshing_source_closure(
                tmp_path / "actual-ale.zip", records, limits=limits
            )
            assert frozenset(request_scope.field_owners) == owner_keys
            return records, expected, publication

    prepared = execute_native_consumer(
        motion.target, plan.specification.limits,
        prepare_archive,
        failure_factory=qualification_failure,
        prepared_operands=accepted.prepared_operands,
        observe_managed_admission=True,
    )
    consumer_receipt = prepared.result.execution_evidence
    assert consumer_receipt is not None
    assert (
        validate_native_execution_interchange(consumer_receipt.to_record())[
            "managed_admission_peak_bytes_upper"
        ]
        is not None
    )
    records, expected, publication = prepared.value

    def cold_checkpoint(budget: CoordinateEnclosureBudget) -> dict[str, Any]:
        before_calls = len(measure_calls)
        before_displacements = len(displacement_calls)
        cold_records, cold_expected = _overset_conservative_checkpoint(
            accepted, plan, budget=budget
        )
        assert len(measure_calls) - before_calls == 6
        assert displacement_calls[before_displacements:] == [2]
        assert displacement_calls[0] + displacement_calls[-1] == 4
        assert cold_expected == expected
        assert content_equal(cold_records, records, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS)
        return cold_records

    cold = execute_native_consumer(
        motion.target, plan.specification.limits, cold_checkpoint,
        failure_factory=qualification_failure,
        observe_managed_admission=True,
    )
    cold_receipt = cold.result.execution_evidence
    assert cold_receipt is not None
    assert (
        validate_native_execution_interchange(cold_receipt.to_record())[
            "managed_admission_peak_bytes_upper"
        ]
        is not None
    )
    foreign_called: list[bool] = []
    with pytest.raises(ValueError):
        execute_native_consumer(
            accepted.source, plan.specification.limits,
            lambda budget: foreign_called.append(True),
            failure_factory=qualification_failure,
            prepared_operands=accepted.prepared_operands,
        )
    assert foreign_called == []
    def conflict_before_admission(budget: CoordinateEnclosureBudget) -> None:
        key, integral = operands._measure_snapshots[0]
        budget.reserve(1, 256)
        budget.exact_cell_measure_cache[key] = integral[0] + 1.0, integral[1]
        operands.require(
            motion.target.mesh, motion.target.geometry, paid_receipt, budget,
            coordinate_contract=motion.target.coordinate_contract,
        )

    with pytest.raises(ValueError, match="conflict with current exact cell measures"):
        execute_native_consumer(
            motion.target, plan.specification.limits, conflict_before_admission,
            failure_factory=qualification_failure,
        )

    def conflict_after_admission(budget: CoordinateEnclosureBudget) -> None:
        key, integral = operands._measure_snapshots[0]
        budget.reserve(1, 256)
        budget.exact_cell_measure_cache[key] = (
            (integral[0], -integral[1]) if integral[1] == 0.0
            else (integral[0] + 1.0, integral[1])
        )
        # A receiver must not silently replace the kernel's rounded bit policy.
        operands.require(
            motion.target.mesh, motion.target.geometry, paid_receipt, budget,
            coordinate_contract=motion.target.coordinate_contract,
        )

    with pytest.raises(ValueError, match="conflict with current exact cell measures"):
        execute_native_consumer(
            motion.target, plan.specification.limits, conflict_after_admission,
            failure_factory=qualification_failure, prepared_operands=operands,
        )
    # A caller's working dictionary is not a kernel issue, even if its key has
    # the exact compact form and it borrows a genuine live native allowance.
    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector
    from phydrax.meshing._native_preparation import native_preparation_ledger

    forged = CoordinateOperandCollector(capture_measures=True)
    with NativeExecutionScope(plan.specification.limits.native_limits()) as scope:
        forged.begin(scope, None, native_preparation_ledger(scope).overlap(scope, None))
        initial = scope.batch(0)
        initial.settle()
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=plan.specification.limits.maximum_work_units,
            maximum_memory_bytes=plan.specification.limits.maximum_scratch_bytes,
        )
        with budget.bind_ticket(scope.ticket_view()):
            budget.capture_coordinate_operands = True
            budget.capture_measure_operands = True
            key, integral = operands._measure_snapshots[0]
            budget.reserve(1, 256)
            budget.exact_cell_measure_cache[key] = integral[0] + 1.0, integral[1]
            forged.capture(
                motion.target.mesh, motion.target.geometry, budget, scope,
                coordinate_contract=motion.target.coordinate_contract,
            )
            assert forged._pending is not None
            assert forged._pending._measure_snapshots == ()
        forged.abort()
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._mapped_geometry_measure import certified_cell_measures

    # Real standalone arithmetic still cannot be relabeled as a paid native
    # earning phase merely by presenting a different live scope at capture.
    with HostBatchTicket(
        plan.specification.limits.maximum_work_units,
        bytes_upper=plan.specification.limits.maximum_scratch_bytes,
    ) as host_ticket:
        host_budget = CoordinateEnclosureBudget(
            host_ticket,
            maximum_work_units=plan.specification.limits.maximum_work_units,
            maximum_memory_bytes=plan.specification.limits.maximum_scratch_bytes,
        )
        host_budget.capture_coordinate_operands = True
        host_budget.capture_measure_operands = True
        certified_cell_measures(motion.target.mesh, motion.target.geometry, budget=host_budget)
        unpaid = CoordinateOperandCollector(capture_measures=True)
        with NativeExecutionScope(plan.specification.limits.native_limits()) as scope:
            unpaid.begin(scope, None, native_preparation_ledger(scope).overlap(scope, None))
            with pytest.raises(ValueError, match="actual native earning scope"):
                unpaid.capture(
                    motion.target.mesh, motion.target.geometry, host_budget, scope,
                    coordinate_contract=motion.target.coordinate_contract,
                )
            unpaid.abort()
    epochs = phx.lifecycle.validate_meshing_accepted_epochs(records)
    assert [epoch.epoch_id for epoch in epochs] == expected["epoch_ids"]
    assert isinstance(epochs[1].transition, MeshingGeometryMotionTransition)
    assert set(epochs[1].fields) == {
        "conservative:density", "conservative:materials", "conservative:history"
    }
    raw = phx.lifecycle.read_meshing_source_execution_controls(
        publication.path, expected_content_id=publication.content_id, limits=limits
    )
    def restore_and_replay(budget: CoordinateEnclosureBudget) -> dict[str, Any]:
        restored = phx.lifecycle.read_meshing_source_closure(
            publication.path, expected_content_id=publication.content_id,
            limits=limits, budget=budget,
        )
        old, new = phx.lifecycle.validate_meshing_accepted_epochs(restored)
        from phydrax.meshing._native_preparation import _require_native_completed_source

        assert new.carrier.execution_evidence is not None
        # Decoded evidence is historical data, never the warm process's issuer.
        with pytest.raises(ValueError, match="actual issued weak owner"):
            _require_native_completed_source(
                new.carrier, new.carrier.execution_evidence, budget=budget,
            )
        remap = phx.meshing.prepare_overset_conservative_remap(
            phx.meshing.MeshPart("background", old.carrier),
            phx.meshing.MeshPart("background", new.carrier),
            policy=phx.geometry.CommonRefinementPolicy(
                maximum_candidate_pairs=controls.overset_overlap_pairs,
                maximum_accepted_pairs=controls.overset_overlap_pairs,
                maximum_memory_bytes=4096 * controls.capacity,
                coverage_tolerance=1.0e-10,
            ),
            budget=budget,
        )
        remap_plan, overlap = _overset_require_complete_cold_remap(remap, budget=budget)
        assert overlap.source_gap_count == overlap.target_gap_count == 0
        for name, value in new.fields.items():
            np.testing.assert_allclose(
                remap_plan.apply(jnp.asarray(old.fields[name])), value,
                rtol=0.0, atol=1.0e-12,
            )
        return restored

    replay = execute_native_request(
        raw.execution_evidence,
        raw.limits,
        restore_and_replay,
        target=lambda restored: restored["accepted_target"],
        failure_factory=qualification_failure,
    )
    restored_epochs = phx.lifecycle.validate_meshing_accepted_epochs(replay.value)
    for old, restored in zip(epochs, restored_epochs, strict=True):
        assert old.epoch_id == restored.epoch_id
        for name, value in old.fields.items():
            np.testing.assert_array_equal(
                np.asarray(value).view(np.uint8),
                np.asarray(restored.fields[name]).view(np.uint8),
            )
    for attribute, value in (
        ("source_mesh_id", "foreign-source"),
        ("lineage", phx.meshing.MeshLineage("foreign", "foreign", motion.lineage.entities)),
    ):
        corrupted = copy(motion)
        object.__setattr__(corrupted, attribute, value)
        with pytest.raises(ValueError):
            MeshingGeometryMotionTransition(accepted.source, motion.target, corrupted)
    altered_geometry = copy(motion.geometry_transition)
    object.__setattr__(altered_geometry, "source_geometry_id", "foreign-geometry")
    corrupted = copy(motion)
    object.__setattr__(corrupted, "geometry_transition", altered_geometry)
    with pytest.raises(ValueError):
        MeshingGeometryMotionTransition(accepted.source, motion.target, corrupted)
    authority = accepted.source.geometry.exact_source
    assert isinstance(authority, phx.discretization.ExactPlcCellGeometrySource)
    stratified = int(np.flatnonzero(np.asarray(authority.vertex_strata) != 0)[0])
    altered = np.asarray(accepted.source.mesh.coordinates).copy()
    altered[stratified, 0] += 1.0e-6
    with pytest.raises(ValueError, match="stratified"):
        transition_displaced_cell_geometry(
            accepted.source.mesh, accepted.source.geometry,
            accepted.source.mesh.with_coordinates(altered, numeric_version="invalid-stratum"),
        )


def _complete_record(error: float) -> dict[str, Any]:
    poisson = {"passed": True, "physical_error": error}
    motion = {
        "independent_registration_oracle": {"exact": True},
        "registration": {"complete": True},
        "motion_refresh": {"refreshed": True},
        "transfer": {"oracles_passed": True},
        "rollback": {"atomic": True},
        "poisson": poisson,
    }
    return {
        "epochs": [
            {
                "epoch": 0,
                "independent_registration_oracle": {"exact": True},
                "registration": {"complete": True},
                "poisson": poisson,
            },
            {**motion, "epoch": 1},
            {**motion, "epoch": 2},
        ],
        "conservative_profile": {"certified": True, "refusals_atomic": True},
        "refused_overreach_motion": {"atomic": True},
        "restart": {
            "passed": True,
            "continued_composite_poisson": {"physical_error": error},
            "conservative": {"passed": True},
            "cold_request": {"reader_proof_solve_shared_budget": True},
        },
    }


def test_workflow_obligations_are_derived_from_recorded_evidence() -> None:
    args = QualificationControls(target_error=0.05, timeout=120.0)
    empty = _overset_workflow_obligations({}, args, 1.0)
    assert empty.pop("wall_budget") is True
    assert not any(empty.values())
    assert all(_overset_workflow_obligations(_complete_record(0.01), args, 1.0).values())
    missed = _overset_workflow_obligations(_complete_record(0.06), args, 1.0)
    assert missed["accepted_pde_update"] is False
    assert missed["cold_restart_ownership_continuation"] is False
    late = _overset_workflow_obligations(_complete_record(0.01), args, 121.0)
    assert late["wall_budget"] is False
    broken = _complete_record(0.01)
    broken["epochs"][1]["rollback"] = {"atomic": False}
    assert _overset_workflow_obligations(broken, args, 1.0)["atomic_refusals"] is False
    no_history = _complete_record(0.01)
    del no_history["restart"]["conservative"]
    assert not _overset_workflow_obligations(
        no_history, args, 1.0
    )["cold_restart_ownership_continuation"]
    renewed_budget = _complete_record(0.01)
    renewed_budget["restart"]["cold_request"] = {}
    assert not _overset_workflow_obligations(
        renewed_budget, args, 1.0
    )["cold_restart_ownership_continuation"]


def _donor_charts(*, fine_admissible: bool = True) -> dict[int, _DonorChart]:
    coarse = _DonorChart(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        np.asarray((10, 11), dtype=np.int64),
        np.asarray((True, True)),
    )
    fine = _DonorChart(
        np.asarray(((0.5, 0.0), (1.0, 0.0), (0.5, 0.5))),
        np.asarray(((0, 1, 2),), dtype=np.int64),
        np.asarray((20,), dtype=np.int64),
        np.asarray((fine_admissible,)),
    )
    return {1: coarse, 2: fine}


def _receptor(
    site: tuple[float, float] = (0.6, 0.2),
    *,
    status: int = 0,
    part: int = 2,
    cell: int = 20,
    candidates: int = 2,
    residual: float = 0.0,
    measure: float = 0.125,
) -> _ReceptorDonors:
    return _ReceptorDonors(
        np.asarray((site,), dtype=np.float64),
        np.asarray((status,), dtype=np.int32),
        np.asarray((part,), dtype=np.int32),
        np.asarray((cell,), dtype=np.int64),
        np.asarray((candidates,), dtype=np.int32),
        np.asarray((residual,), dtype=np.float64),
        np.asarray((measure,), dtype=np.float64),
    )


def test_donor_verdict_accepts_correct_finest_donor_within_measure_bound() -> None:
    verdict = _donor_verdict(_receptor(), _donor_charts())
    assert verdict["exact"] is True
    rounded = _donor_verdict(
        _receptor(measure=float(np.nextafter(0.125, 1.0))), _donor_charts()
    )
    assert rounded["exact"] is True
    assert 0.0 < rounded["maximum_measure_discrepancy_to_bound_ratio"] < 1.0
    assert _measure_bound(2.0, 0.125) < 1.0e-11


@pytest.mark.parametrize(
    ("receptor", "fine_admissible", "fault"),
    (
        (_receptor(candidates=1), True, "candidate_part_mismatches"),
        (_receptor(candidates=3), True, "candidate_part_mismatches"),
        (
            _receptor(measure=0.125 * (1.0 + 1.0e-6)),
            True,
            "measure_discrepancy_violations",
        ),
        (_receptor(part=1, cell=10, measure=0.5), True, "not_finest"),
        (_receptor(part=1, cell=11, measure=0.5), True, "chosen_not_containing"),
        (_receptor(part=1, cell=99, measure=0.5), True, "chosen_not_admissible"),
        (_receptor(), False, "chosen_not_admissible"),
        (_receptor(residual=1.0e-6), True, "residual_violations"),
        (_receptor(status=1, part=-1, cell=-1), True, "oracle_orphans"),
        (_receptor(status=1, part=-1, cell=-1), True, "registration_orphans"),
    ),
)
def test_donor_verdict_refuses_each_injected_fault(
    receptor: _ReceptorDonors, fine_admissible: bool, fault: str
) -> None:
    verdict = _donor_verdict(receptor, _donor_charts(fine_admissible=fine_admissible))
    assert verdict[fault] == 1
    assert verdict["exact"] is False


def test_donor_verdict_band_admits_owner_tolerance_either_way_only() -> None:
    # (0.5, 0.2) lies exactly on the fine cell's edge: the owner's tolerance may
    # locate it there or not, but it is certainly inside the coarse cell.
    for candidates in (1, 2):
        verdict = _donor_verdict(
            _receptor((0.5, 0.2), part=1, cell=10, candidates=candidates, measure=0.5),
            _donor_charts(),
        )
        assert verdict["exact"] is True
        assert verdict["band_pairs"] == 1
    missing = _donor_verdict(
        _receptor((0.5, 0.2), part=1, cell=10, candidates=0, measure=0.5),
        _donor_charts(),
    )
    assert missing["candidate_part_mismatches"] == 1
    assert missing["exact"] is False


@pytest.mark.parametrize("field", ("reference_tolerance", "residual_tolerance"))
@pytest.mark.parametrize("value", (1.0e-12, 1.0e-8, float("nan")))
def test_donor_oracle_refuses_changed_product_locator_controls(
    field: str, value: float,
) -> None:
    controls = {"reference_tolerance": 1.0e-10, "residual_tolerance": 1.0e-10}
    controls[field] = value
    geometry = SimpleNamespace(
        discretization=SimpleNamespace(
            block_geometries=((SimpleNamespace(reference_weights=np.ones(3)),),)
        ),
        cell_measures=np.ones(1, dtype=np.float64),
        locators=(SimpleNamespace(policy=SimpleNamespace(**controls)),),
    )
    connectivity = SimpleNamespace(
        assembly=SimpleNamespace(parts=(SimpleNamespace(name="part"),)),
        geometries=(geometry,),
        policy=SimpleNamespace(donor_priority="finest-donor"),
        specs=(SimpleNamespace(priority=0),),
    )
    layout = _PartLayout(
        np.zeros((3, 2)), np.asarray(((0, 1, 2),)), np.asarray((0,))
    )
    with pytest.raises(QualificationFailure, match="exact locator controls"):
        _overset_donor_oracle(connectivity, {"part": layout})  # ty: ignore[invalid-argument-type]  # Deliberately corrupt product-policy fixture.


@pytest.mark.parametrize(
    ("registration", "query"), (("previous", "current-query"), ("current", "other-query"))
)
def test_composite_actions_refuse_stale_source_binding_before_execution(
    registration: str, query: str,
) -> None:
    connectivity = SimpleNamespace(connectivity_id="current")
    solver = SimpleNamespace(
        rows={},
        composite=SimpleNamespace(coupling=SimpleNamespace(operator_id=query)),
        route=SimpleNamespace(
            connectivity=SimpleNamespace(connectivity_id=registration),
            transfer_id="current-query",
        ),
    )
    # Missing numerical operands are intentional: source mismatch must refuse
    # before a compiled action or carried coefficient is consumed.
    with pytest.raises(QualificationFailure, match="current registration and field query"):
        _overset_composite_update(
            connectivity, {}, solver, reused=True, warm_repeats=0  # ty: ignore[invalid-argument-type]  # Deliberately incomplete stale-registration fixture.
        )


def _composite_queries(values: dict[str, Array]) -> dict[str, Array]:
    return {"a": 0.5 * values["b"][:1], "b": 0.5 * values["a"][:1]}


def test_native_composite_solve_matches_independent_coupled_system() -> None:
    space = phx.linalg.ArraySpace((2,), dtype=jnp.float64)
    matrix = jnp.asarray(((2.0, -1.0), (-1.0, 2.0)))
    local = phx.linalg.FunctionLinearOperator(
        lambda value: matrix @ value, source=space, target=space,
    )
    identity = phx.linalg.FunctionLinearOperator(
        lambda value: value, source=space, target=space,
    )
    lifts = (jnp.asarray((0.25, 0.0)), jnp.asarray((0.5, 0.0)))
    output_space = phx.linalg.ArraySpace((1,), dtype=jnp.float64)
    coupling = phx.linalg.FunctionLinearOperator(
        _composite_queries,
        source=phx.linalg.PyTreeSpace(
            {name: space.structure() for name in ("a", "b")},
            space_id="test-donor-coefficients",
        ),
        target=phx.linalg.PyTreeSpace(
            {name: output_space.structure() for name in ("a", "b")},
            space_id="test-receptor-values",
        ),
        operator_id="test-composite-query",
    )
    composite = _OversetComposite(
        (local, local), (jnp.asarray((1.0, 0.0)), jnp.asarray((2.0, 0.0))),
        (identity, identity), lifts,
        (jnp.asarray((1,)), jnp.asarray((1,))),
        coupling, ("a", "b"), ("a", "b"), (0, 2, 4), "test-composite",
    )
    expected_matrix = np.asarray(
        ((2.0, -1.0, 0.0, 0.0), (0.0, 1.0, -0.5, 0.0),
         (0.0, 0.0, 2.0, -1.0), (-0.5, 0.0, 0.0, 1.0))
    )
    expected_rhs = np.asarray((1.0, 0.25, 2.0, 0.125))
    probe = jnp.arange(4, dtype=jnp.float64)
    np.testing.assert_allclose(composite.homogeneous(probe), expected_matrix @ probe)
    np.testing.assert_allclose(composite.right_hand_side(), expected_rhs)
    materialization = phx.linalg.MaterializationPolicy()
    prepared = tuple(
        _overset_part_term(
            composite, index, np.asarray(((0, 1, 2),)),
            np.asarray((0, 1)), materialization,
        )
        for index in range(2)
    )
    terms = tuple(item[0] for item in prepared)
    realized = eqx.tree_at(
        lambda value: value.operators, composite, tuple(item[3] for item in prepared)
    )
    np.testing.assert_allclose(realized.homogeneous(probe), composite.homogeneous(probe))
    preconditioner = phx.linalg.MultiplicativeSubspaceCorrectionBuilder(terms).prepare(
        realized.operator(), materialization=materialization,
    )
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=4),
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0, absolute=1.0e-11, max_steps=64,
        ),
        preconditioning=phx.linalg.PreconditioningPolicy(preconditioner, side="right"),
    )
    values, solved = _solve_overset_composite(realized, jnp.zeros(4), policy)
    assert bool(solved.successful)
    reference = np.linalg.solve(expected_matrix, expected_rhs)
    for index, name in enumerate(("a", "b")):
        np.testing.assert_allclose(
            values[name], reference[2 * index:2 * index + 2] + lifts[index],
            rtol=0.0, atol=1.0e-11,
        )
    # Reuse only symbolic structure; the numerical factors must follow the new
    # local block rather than carrying the accepted predecessor's coefficients.
    changed_matrix = matrix.at[0, 0].set(3.0)
    changed_operator = phx.linalg.FunctionLinearOperator(
        lambda value: changed_matrix @ value, source=space, target=space,
    )
    changed = _OversetComposite(
        (changed_operator, local), composite.loads,
        composite.prolongations, composite.lifts, composite.positions,
        composite.coupling, composite.donor_names, composite.names,
        composite.offsets, "test-composite-changed",
    )
    term, evidence, symbolic, _ = _overset_part_term(
        changed, 0, np.asarray(((0, 1, 2),)), np.asarray((0, 1)),
        materialization, previous=prepared[0][2],
    )
    assert evidence["reused_symbolic_plan"] is True
    assert evidence["factor_symbolic_work"] == 0
    assert symbolic.plan.plan_id == prepared[0][2].plan.plan_id
    assert isinstance(term.local_solver, phx.linalg.AbstractPreconditioner)
    np.testing.assert_allclose(
        term.local_solver.apply(jnp.asarray((1.0, 2.0))),
        np.linalg.solve(np.asarray(((3.0, -1.0), (0.0, 1.0))), np.asarray((1.0, 2.0))),
        rtol=0.0, atol=1.0e-12,
    )
    continuation_policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=4),
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0, absolute=1.0e-11, max_steps=64,
        ),
    )
    _, accepted = _solve_overset_composite(composite, solved.value, continuation_policy)
    assert bool(accepted.successful)
    # This route must solve a nonzero physical residual, not merely compare the
    # restored coefficients with a saved state or claim convergence from its ID.
    _, corrected = _solve_overset_composite(
        composite, solved.value + jnp.asarray((0.1, -0.2, 0.3, -0.4)),
        continuation_policy,
    )
    assert bool(corrected.successful)
    assert int(corrected.diagnostics.iterations) > 0
    np.testing.assert_allclose(corrected.value, reference, rtol=0.0, atol=1.0e-11)

def test_overset_update_measures_bounded_real_executions_and_refuses_runtime_compile() -> None:
    import jax

    from phydrax.meshing.qualification import native_planar_part
    from tools.meshing_qualification.scenarios.overset import (
        _OversetJitEntry,
        _OversetPartRows,
        _OversetSolver,
        _part_carrier,
    )

    parts = {
        name: native_planar_part(name, 1.0, size=1.0)[0]
        for name in ("a", "b")
    }
    connectivity = phx.meshing.prepare_overset_connectivity(
        phx.meshing.MeshAssembly(tuple(parts.values())),
        tuple(phx.meshing.OversetPartSpec(name) for name in parts),
    )
    point_count = np.asarray(_part_carrier(parts["a"]).mesh.coordinates).shape[0]
    spaces = {
        name: phx.linalg.ArraySpace((point_count,), dtype=jnp.float64)
        for name in parts
    }
    local = tuple(
        phx.linalg.FunctionLinearOperator(
            lambda value: value,
            source=spaces[name],
            target=spaces[name],
            operator_id=f"test-warm-{name}",
        )
        for name in ("a", "b")
    )
    identity = tuple(
        phx.linalg.FunctionLinearOperator(
            lambda value: value,
            source=spaces[name],
            target=spaces[name],
        )
        for name in ("a", "b")
    )
    output_space = phx.linalg.ArraySpace((1,), dtype=jnp.float64)
    coupling = phx.linalg.FunctionLinearOperator(
        _composite_queries,
        source=phx.linalg.PyTreeSpace(
            {name: spaces[name].structure() for name in ("a", "b")},
            space_id="test-warm-donors",
        ),
        target=phx.linalg.PyTreeSpace(
            {name: output_space.structure() for name in ("a", "b")},
            space_id="test-warm-receptors",
        ),
        operator_id="test-warm-query",
    )
    zeros = tuple(jnp.zeros((point_count,), dtype=jnp.float64) for _ in parts)
    composite = _OversetComposite(
        local,
        zeros,
        identity,
        zeros,
        (jnp.asarray((0,)), jnp.asarray((0,))),
        coupling,
        ("a", "b"),
        ("a", "b"),
        (0, point_count, 2 * point_count),
        "test-warm-composite",
    )
    seed = jnp.zeros((2 * point_count,), dtype=jnp.float64)
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.GMRES(restart=4),
        tolerance=phx.linalg.TolerancePolicy(
            relative=0.0, absolute=1.0e-11, max_steps=64,
        ),
    )
    entry = _solve_overset_composite
    assert isinstance(entry, _OversetJitEntry)
    compiled = entry.lower(composite, seed, policy).compile()
    calls: list[tuple[_OversetComposite, Array, phx.linalg.LinearSolvePolicy]] = []
    outputs: list[tuple[dict[str, Array], phx.linalg.LinearSolveResult]] = []

    def counted(
        comp: _OversetComposite, value: Array, solve_policy: phx.linalg.LinearSolvePolicy,
    ) -> tuple[dict[str, Array], phx.linalg.LinearSolveResult]:
        calls.append((comp, value, solve_policy))
        output = compiled(comp, value, solve_policy)
        outputs.append(output)
        return output

    route = SimpleNamespace(
        connectivity=connectivity,
        transfer_id="test-warm-query",
        donor_names=("a", "b"),
        apply=_composite_queries,
    )
    rows = {
        name: _OversetPartRows(
            SimpleNamespace(constrained_dofs=np.asarray((point_count - 1,))),
            np.arange(point_count, dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray((point_count - 1,), dtype=np.int64),
            jnp.zeros((1,), dtype=jnp.float64),
            np.zeros(point_count, dtype=np.bool_),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        )
        for name in parts
    }
    problems = tuple(
        SimpleNamespace(lift=jnp.zeros((point_count,), dtype=jnp.float64))
        for _ in parts
    )
    solver = _OversetSolver(
        composite,
        policy,
        counted,
        rows,
        {name: {"domain": 0.0, "wall_flux": None, "exterior_flux": None} for name in parts},
        {},
        {},
        {"local_action_realization": "verified-native-sparse-part-blocks"},
        {},
        problems,
        tuple(lambda problem, value: jnp.zeros_like(value) for _ in parts),
        route,
    )
    initial = {
        name: jnp.zeros((point_count,), dtype=jnp.float64)
        for name in parts
    }
    values, report, _ = _overset_composite_update(
        connectivity, initial, solver, reused=False, warm_repeats=2,
    )
    assert len(calls) == len(outputs) == 3
    assert all(
        comp is composite and value is calls[0][1] and solve_policy is policy
        for comp, value, solve_policy in calls
    )
    assert all(
        leaf.is_ready()
        for output in outputs
        for leaf in jax.tree_util.tree_leaves(output)
    )
    for name in parts:
        np.testing.assert_array_equal(values[name], outputs[0][0][name])
    assert report["passed"] is True
    assert len(solver.compiler["warm_samples_seconds"]) == 2
    assert all(
        isinstance(seconds, float) and np.isfinite(seconds) and seconds >= 0.0
        for seconds in solver.compiler["warm_samples_seconds"]
    )
    assert isinstance(solver.compiler["first_execution_seconds"], float)
    assert solver.compiler["first_execution_compilation_count"] == 0
    assert solver.compiler["warm_recompilation_count"] == 0

    def compiling_at_execution(
        comp: _OversetComposite, value: Array, solve_policy: phx.linalg.LinearSolvePolicy,
    ) -> tuple[dict[str, Array], phx.linalg.LinearSolveResult]:
        jax.jit(lambda operand: operand + 1)(
            jnp.zeros((31,), dtype=jnp.float64)
        ).block_until_ready()
        return compiled(comp, value, solve_policy)

    refusing_solver = solver._replace(compiled=compiling_at_execution)
    with pytest.raises(
        QualificationFailure, match="Composite execution must not compile at runtime"
    ):
        _overset_composite_update(
            connectivity, initial, refusing_solver, reused=False, warm_repeats=0
        )


def _cold_remap_fixture(
    shift: float, policy: phx.geometry.CommonRefinementPolicy,
    *, budget: CoordinateEnclosureBudget | None = None,
) -> phx.discretization.PreparedUnstructuredConservativeRemap:
    from tests._support.cell_meshes import triangle_grid

    source_mesh = triangle_grid(1, 1)
    target_mesh = source_mesh.with_coordinates(
        np.asarray(source_mesh.coordinates) + np.asarray((shift, 0.0)),
        numeric_version=f"cold-remap-target:{shift}",
    )
    source = phx.meshing.MeshPart(
        "background",
        phx.meshing.certify_cell_mesh(source_mesh, phx.SpatialCoordinateContract.si()),
    )
    target = phx.meshing.MeshPart(
        "background",
        phx.meshing.certify_cell_mesh(target_mesh, phx.SpatialCoordinateContract.si()),
    )
    return phx.meshing.prepare_overset_conservative_remap(
        source, target, policy=policy, budget=budget,
    )


def test_cold_remap_gate_requires_actual_complete_overlap_plan() -> None:
    remap = _cold_remap_fixture(0.0, phx.geometry.CommonRefinementPolicy())
    plan, evidence = _overset_require_complete_cold_remap(remap)
    assert plan is remap.plan
    assert evidence is remap.evidence
    assert plan.require_complete
    np.testing.assert_allclose(plan.apply(jnp.asarray((1.0, 2.0))), (1.0, 2.0))


@pytest.mark.parametrize("case", ("candidate-capacity", "coverage-gap", "partial"))
def test_cold_remap_refusal_preserves_real_status_resources_and_coverage(
    case: str,
) -> None:
    policy = phx.geometry.CommonRefinementPolicy(
        maximum_candidate_pairs=1 if case == "candidate-capacity" else 500000,
        coverage=phx.geometry.CommonRefinementCoverage.PARTIAL
        if case == "partial" else phx.geometry.CommonRefinementCoverage.COMPLETE,
    )
    remap = _cold_remap_fixture(0.0 if case == "candidate-capacity" else 0.25, policy)
    expected = {
        "candidate-capacity": phx.geometry.CommonRefinementStatus.RESOURCE_LIMIT,
        "coverage-gap": phx.geometry.CommonRefinementStatus.COVERAGE_GAP,
        "partial": phx.geometry.CommonRefinementStatus.SUCCESS,
    }[case]
    assert remap.status is expected
    with pytest.raises(_OversetColdRemapRefusal) as raised:
        _overset_require_complete_cold_remap(remap)
    failure = raised.value.report
    assert failure["status"] == expected.name
    assert failure["reason"] == remap.reason
    assert failure["plan_present"] is (remap.plan is not None)
    assert isinstance(remap.evidence, phx.geometry.CommonRefinementEvidence)
    coverage = failure["coverage"]
    assert coverage["status"] == remap.evidence.status.name
    assert coverage["evidence_id"] == remap.evidence.evidence_id
    np.testing.assert_array_equal(
        coverage["source_coverage_defects"], remap.evidence.source_coverage_defects,
    )
    np.testing.assert_array_equal(
        coverage["target_coverage_tolerances"], remap.evidence.target_coverage_tolerances,
    )
    assert failure["execution_evidence"] == (
        None if remap.execution_evidence is None else remap.execution_evidence.to_record()
    )
    assert json.loads(json.dumps(failure)) == failure
    if case == "partial":
        assert failure["plan_requires_complete"] is False
        assert coverage["source_gap_count"] + coverage["target_gap_count"] > 0
    else:
        assert failure["plan_present"] is False
    record = _complete_record(0.01)
    record["restart"] = {"passed": False, "conservative_remap_refusal": failure}
    assert not _overset_workflow_obligations(
        record, QualificationControls(), 1.0,
    )["cold_restart_ownership_continuation"]


def test_cold_remap_refusal_retains_actual_ended_request() -> None:
    from dataclasses import asdict

    from phydrax._meshcore import NativeExecutionScope
    from phydrax.discretization.finite_volume._automatic_remap import bind_remap_execution
    from phydrax.meshing._measurements import NativeExecutionRecord
    from phydrax.meshing.providers._native_publication import coordinate_budget

    limits = phx.meshing.MeshingLimits()
    budget = coordinate_budget(limits)
    with pytest.raises(_OversetColdRemapRefusal) as raised:
        with NativeExecutionScope(limits.native_limits()) as scope:
            with budget.bind_ticket(scope.ticket_view()):
                remap = _cold_remap_fixture(
                    0.0, phx.geometry.CommonRefinementPolicy(maximum_candidate_pairs=1),
                    budget=budget,
                )
                _overset_require_complete_cold_remap(remap, budget=budget)
    report = raised.value.to_record()
    assert report["status"] == "RESOURCE_LIMIT"
    assert report["coordinate_work_units"] == budget.work_units > 0
    assert report["request_execution_evidence"] == asdict(scope.completed_evidence())
    assert json.loads(json.dumps(report)) == report
    # The route's optional ended record is retained when its canonical owner
    # binds one; a route without such a record is never given invented counters.
    receipt = NativeExecutionRecord(scope.completed_evidence())
    bound = bind_remap_execution(remap, receipt)
    with pytest.raises(_OversetColdRemapRefusal) as bound_refusal:
        _overset_require_complete_cold_remap(bound)
    assert bound_refusal.value.report["execution_evidence"] == receipt.to_record()
