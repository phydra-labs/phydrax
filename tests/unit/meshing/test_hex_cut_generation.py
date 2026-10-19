"""Exercise retained PLC cuts through construction identity and phase evidence."""

import numpy as np
import pytest

from phydrax.discretization import (
    CellGeometrySpec,
    CellValidityPolicy,
    ExactPlcCellGeometrySource,
)
from phydrax.discretization._exact_plc_geometry import ExactPlcCellGeometryConvexSource
from phydrax.meshing import (
    MeshingLimits,
    NativeHexGridSchedule,
    NativePlcSource,
    PiecewiseLinearComplex,
    SizeControlStrength,
)
from phydrax.meshing._failures import (
    CutCornerLinkIncidence,
    CutStageError,
    CutStageRecord,
)
from phydrax.meshing._family_execution import coordinate_phase
from phydrax.meshing._hex_cut_generation import (
    BalancedCutHexConstruction,
    cut_stage,
    generate_balanced_cut_hexes,
)
from phydrax.meshing._hex_cut_preparation import CutCornerLinkError
from phydrax.meshing._volume_generation import generate_plc_volume, NativeVolumeSchedule
from tests._support.coordinate_references import native_meshing_request
from tests._support.meshing_fixtures import volume_specification


def _construct_reference_cuts(
    *, strength: SizeControlStrength
) -> BalancedCutHexConstruction:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    faces = ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    plc = PiecewiseLinearComplex(
        points,
        faces,
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        ("simplex",),
    )
    source = NativePlcSource(plc, "cut-reference-simplex", "cut-reference-authored")
    limits = MeshingLimits()
    # The reference needs the single admitted tet from the hard statistical epoch.
    specification = volume_specification(
        source, size=4.0, limits=limits, strength=strength
    )
    with native_meshing_request(limits) as (scope, budget, _):
        volume = generate_plc_volume(
            plc,
            specification,
            NativeVolumeSchedule(),
            validity_policy=CellValidityPolicy(),
            source_id=source.source_id,
            source_revision=source.source_revision,
            input_id=source.binding_id,
            scope=scope,
        )
        fidelity = volume.source_fidelity
        if fidelity is None or fidelity.declared_source is None:
            raise ValueError(
                "fixture.fidelity must retain declared source banks and charts; got missing source"
            )
        declared = fidelity.declared_source
        with coordinate_phase(scope=scope, budget=budget):
            exact_source = ExactPlcCellGeometrySource(
                declared.points,
                declared.faces,
                declared.segments,
                fidelity.witness_strata,
                fidelity.witness_entities,
                fidelity.witness_parameters,
                domain_source_id=source.source_id,
                domain_source_revision=source.source_revision,
                source_triangle_ids=declared.face_ids[declared.face_group_rows],
                source_triangle_bounds=declared.face_tolerances,
                source_segment_ids=declared.segment_ids[declared.segment_group_rows],
                source_segment_bounds=declared.segment_tolerances,
                budget=budget,
            )
            geometry = CellGeometrySpec.plc(volume.mesh, exact_source, budget=budget)
        # Whole-map portfolio bounds are fixed for this independently declared
        # simplex case; embedding and coverage remain separate acceptance obligations.
        schedule = NativeHexGridSchedule(
            "balanced_grid",
            minimum_scaled_jacobian=0.001,
            minimum_mean_ratio=0.001,
            maximum_aspect_ratio=1000.0,
        )
        return generate_balanced_cut_hexes(
            plc,
            volume,
            schedule,
            limits,
            0.75,
            source_geometry=geometry,
            scope=scope,
            budget=budget,
        )


def test_balanced_cut_construction_repeats_identity_and_earned_phase_records() -> None:
    first = _construct_reference_cuts(strength=SizeControlStrength.HARD)
    second = _construct_reference_cuts(strength=SizeControlStrength.HARD)
    assert first.construction_id == second.construction_id
    assert len(first.construction_id) == 64
    assert first.embedding.status == first.coverage.status == "certified"
    assert first.extraction.validity.all_certified
    geometry = first.extraction.geometry
    assert geometry is not None
    source = geometry.exact_source
    assert isinstance(source, ExactPlcCellGeometryConvexSource)
    parent_ids = np.asarray(
        first.extraction.source.entity_set(3).entity_ids, dtype=np.int64
    )
    np.testing.assert_array_equal(
        source.cell_parent_ids, parent_ids[first.extraction.parent_cells]
    )
    assert first.exact_grid_origin == second.exact_grid_origin
    assert (
        first.origin_preparation.identity_payload()
        == second.origin_preparation.identity_payload()
    )
    assert tuple(record.stage for record in first.stage_receipts) == (
        "source_and_origin_preparation",
        "source_clipping",
        "corner_closure",
        "exact_maps_embedding_quality",
        "original_scientific_ancestry",
        "original_domain_coverage",
    )
    assert all(record.failure is None for record in first.stage_receipts)
    assert first.cumulative_performed_work_units == sum(
        int(record.execution.work_counters.construction_work)
        for record in first.stage_receipts
    )
    assert first.cumulative_performed_work_units == second.cumulative_performed_work_units


def test_soft_reference_cuts_reach_quality_after_boundary_contact_certification() -> None:
    with pytest.raises(
        CutStageError, match=r"cut_quality.scaled Jacobian must respect bound 0\.001"
    ) as caught:
        _construct_reference_cuts(strength=SizeControlStrength.SOFT)
    records = caught.value.stage_records
    assert records[-1].stage == "exact_maps_embedding_quality"
    assert all(record.failure is None for record in records[:-1])


def test_cut_stage_error_retains_whole_prefix_and_unmodified_corner_incidence() -> None:
    limits = MeshingLimits()
    records: list[CutStageRecord] = []
    incidence = CutCornerLinkIncidence(17, 23, (31, 37, 41, 43), (47, 53, 59, 61))
    original = CutCornerLinkError(incidence)
    with native_meshing_request(limits) as (scope, _, _):
        with cut_stage("source_preparation", None, records, scope=scope) as preparation:
            with preparation.batch(1) as ticket:
                ticket.consume(1)
        with pytest.raises(CutStageError) as caught:
            with cut_stage("corner_closure", None, records, scope=scope):
                raise original
    error = caught.value
    assert error.cause is original
    assert error.__cause__ is original
    assert tuple(record.stage for record in error.stage_records) == (
        "source_preparation",
        "corner_closure",
    )
    assert error.stage_records[0] is records[0]
    assert error.stage_records[1] is records[1]
    assert error.stage_records[0].failure is None
    assert error.stage_records[1].failure == str(original)
    assert error.stage_records[1].link_incidence is incidence


def test_cut_stage_interruption_preserves_signal_and_ended_phase_record() -> None:
    limits = MeshingLimits()
    records: list[CutStageRecord] = []
    interrupt = KeyboardInterrupt(
        "fixture.cut_stage must finish; got injected interruption"
    )
    with native_meshing_request(limits) as (scope, _, _):
        with pytest.raises(KeyboardInterrupt) as caught:
            with cut_stage("source_clipping", None, records, scope=scope):
                raise interrupt
    assert caught.value is interrupt
    assert len(records) == 1
    assert records[0].stage == "source_clipping"
    assert records[0].failure == str(interrupt)
