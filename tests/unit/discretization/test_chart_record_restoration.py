#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Validate stored chart bindings without changing numerical record identity."""

import math
from dataclasses import replace

from phydrax import config as pxconfig


pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
import numpy as np
import pytest

from phydrax._meshcore import NativeExecutionLimits, NativeExecutionScope
from phydrax.discretization import (
    CellGeometryTransitionPolicy,
    CellValidityPolicy,
    prepare_sphere_chart_deformation,
    PreparedSphereChartDeformation,
    PreparedSurfaceChartDeformation,
    reconstruct_sphere_material_cell_geometry,
    SphereGeometryReconstruction,
    UnstructuredFiniteVolumePlan,
)
from phydrax.discretization._cell_geometry_transition_types import CellGeometryTransitionError
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget, CoordinateEnclosureResourceError,
)
from phydrax.discretization._sphere_chart_records import SphereChartValidation
from phydrax.meshing.qualification._surface import prepare_surface_fields
from phydrax.geometry import MeshCertificateLimits
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.mapped_cells import curved_surface_chart_fv
from tests._support.sphere_material import accepted_sphere_material


@pytest.fixture(scope="module")
def surface_chart() -> PreparedSurfaceChartDeformation:
    return curved_surface_chart_fv()[2]


@pytest.fixture(scope="module")
def sphere_chart() -> PreparedSphereChartDeformation:
    _, _, atlas = accepted_sphere_material(1.0)
    return prepare_sphere_chart_deformation(
        atlas, atlas, maximum_fidelity=0.3, maximum_displacement=0.5
    )


def test_surface_restore_retains_instance_and_numerical_content(
    surface_chart: PreparedSurfaceChartDeformation,
) -> None:
    identity = surface_chart.deformation_id
    measures = np.asarray(surface_chart.source_cell_measures).copy()
    restored = PreparedSurfaceChartDeformation.from_stored_fields(surface_chart)
    assert restored is surface_chart
    assert restored.deformation_id == identity
    np.testing.assert_array_equal(restored.source_cell_measures, measures)


def test_surface_restore_refuses_stale_geometry_binding(
    surface_chart: PreparedSurfaceChartDeformation,
) -> None:
    stale = replace(surface_chart, target_geometry_id="stale")
    with pytest.raises(ValueError, match=r"surface_chart\.require_bound\.binding"):
        PreparedSurfaceChartDeformation.from_stored_fields(stale)


def test_sphere_restore_retains_instance_and_numerical_content(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    identity = sphere_chart.deformation_id
    measures = np.asarray(sphere_chart.target_cell_measures).copy()
    restored = PreparedSphereChartDeformation.from_stored_fields(sphere_chart)
    assert restored is sphere_chart
    assert restored.deformation_id == identity
    np.testing.assert_array_equal(restored.target_cell_measures, measures)


def test_sphere_restore_refuses_stale_geometry_binding(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    stale = replace(sphere_chart, target_geometry_id="stale")
    with pytest.raises(ValueError, match=r"sphere_chart\.require_bound\.binding"):
        PreparedSphereChartDeformation.from_stored_fields(stale)


def test_sphere_request_validation_shares_h1_rows_only_inside_its_ledger(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    endpoints = (
        sphere_chart.source_mesh,
        sphere_chart.source_geometry,
        sphere_chart.target_mesh,
        sphere_chart.target_geometry,
    )
    key = ("actual-element-id", (0, 1, 2), "target-element-id", (2, 1, 0))
    rows = np.asarray([[0, 1, 2]], dtype=np.int64)
    coefficients = np.asarray([[0.25, 0.5, 0.25]], dtype=np.float64)
    with coordinate_reference_budget() as budget:
        validation = sphere_chart.require_bound(*endpoints, budget=budget)
        validation.h1_cache[key] = rows, coefficients, 0.0
        proved_work = budget.work_units
        sphere_chart.pieces.authenticate(budget=budget)
        authentication_work = budget.work_units - proved_work
        again = sphere_chart.require_bound(*endpoints, budget=budget)
        assert budget.work_units == proved_work + 2 * authentication_work
        assert again.h1_cache is validation.h1_cache
        assert again.h1_cache[key][0] is rows
        assert again.h1_cache[key][1] is coefficients
        validation.require(sphere_chart, *endpoints, budget=budget)
        changed = replace(sphere_chart, deformation_id="another-record")
        with pytest.raises(ValueError, match="foreign or stale proof"):
            validation.require(changed, *endpoints, budget=budget)
    with coordinate_reference_budget() as other:
        with pytest.raises(ValueError, match="foreign or stale proof"):
            validation.require(sphere_chart, *endpoints, budget=other)
        independent = sphere_chart.require_bound(*endpoints, budget=other)
        assert other.work_units > 0
        assert independent.h1_cache == {}
        assert independent.h1_cache is not validation.h1_cache


def test_sphere_request_validation_refuses_unissued_tokens(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    endpoints = (
        sphere_chart.source_mesh, sphere_chart.source_geometry,
        sphere_chart.target_mesh, sphere_chart.target_geometry,
    )
    binding = (
        sphere_chart.source_mesh.mesh_id, sphere_chart.target_mesh.mesh_id,
        sphere_chart.source_geometry_id, sphere_chart.target_geometry_id,
    )
    with coordinate_reference_budget() as budget:
        unissued = SphereChartValidation(sphere_chart, budget, binding)
        with pytest.raises(ValueError, match="foreign or stale proof"):
            unissued.require(sphere_chart, *endpoints, budget=budget)
        issued = sphere_chart.require_bound(*endpoints, budget=budget)
        issued.require(sphere_chart, *endpoints, budget=budget)
        separate_cache = SphereChartValidation(sphere_chart, budget, binding)
        with pytest.raises(ValueError, match="foreign or stale proof"):
            separate_cache.require(sphere_chart, *endpoints, budget=budget)


def test_surface_field_and_history_geometry_use_the_native_consumer_ledger() -> None:
    _, result, _ = accepted_sphere_material(1.0)
    memory = 256 * 1024**2

    def prepare(cap: int) -> tuple[int, int]:
        with NativeExecutionScope(
            NativeExecutionLimits(cap, 0, 0, memory, math.inf)
        ) as scope:
            with scope.batch(cap, bytes_upper=memory) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=cap, maximum_memory_bytes=memory
                )
                with budget.activate(), ticket.native_target():
                    prepare_surface_fields(result, budget=budget)
                    UnstructuredFiniteVolumePlan.from_cell_mesh(
                        result.mesh, field_name="surface-history",
                        component_names=("surface-history",), budget=budget,
                    ).prepare(cell_geometry=result.geometry, budget=budget)
                work = budget.work_units
        return work, scope.completed_evidence().work_counters.construction_work

    work, native = prepare(1_000_000)
    assert work > 0 and native == work
    assert prepare(work) == (work, work)
    with pytest.raises(CellGeometryTransitionError) as refusal:
        prepare(work - 1)
    assert refusal.value.limit == work - 1
    assert refusal.value.measured >= work
    assert isinstance(refusal.value.__cause__, CoordinateEnclosureResourceError)


@pytest.fixture(scope="module")
def sphere_reconstruction() -> SphereGeometryReconstruction:
    _, _, atlas = accepted_sphere_material(1.0)
    count = atlas.num_charts
    dimensions = np.empty((count, 3), dtype=np.int32)
    indices = np.empty((count, 3), dtype=np.int32)
    parameters = np.zeros((count, 3, 2), dtype=np.float64)
    patches = np.empty((count,), dtype=np.int32)
    paths: list[tuple[tuple[str, ...], ...]] = [()] * count
    entities: list[tuple[str, ...]] = [()] * count
    # Corner tokens retain source indices and occurrence paths; scientific IDs
    # must be resolved through the domain, not inferred from equal array rows.
    for chart_row, physical_row in enumerate(np.asarray(atlas.physical_rows)):
        row = int(physical_row)
        patches[row] = np.asarray(atlas.patches)[chart_row]
        row_paths, row_entities = [], []
        for corner, (dimension, source_index, path, values) in enumerate(
            atlas.corner_tokens[chart_row]
        ):
            dimensions[row, corner] = dimension
            indices[row, corner] = source_index
            parameters[row, corner, : len(values)] = [float(value) for value in values]
            candidates = tuple(
                local
                for local, (identifier, occurrence) in enumerate(
                    zip(
                        atlas.domain.source_indices[dimension],
                        atlas.domain.source_occurrences[dimension],
                        strict=True,
                    )
                )
                if identifier == source_index and occurrence == path
            )
            assert len(candidates) == 1
            row_paths.append(path)
            row_entities.append(atlas.domain.entity_id(dimension, candidates[0]))
        paths[row] = tuple(row_paths)
        entities[row] = tuple(row_entities)
    work, memory = 10_000_000, 256 * 1024**2
    with NativeExecutionScope(
        NativeExecutionLimits(2 * work, work, 0, memory, math.inf)
    ) as scope:
        with scope.batch(2 * work, queries_upper=work, bytes_upper=memory) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            return reconstruct_sphere_material_cell_geometry(
                atlas,
                atlas.source_mesh,
                atlas.source_geometry,
                cell_patches=patches,
                corner_source_dimensions=dimensions,
                corner_source_indices=indices,
                corner_source_parameters=parameters,
                corner_source_occurrence_paths=tuple(paths),
                corner_source_entity_ids=tuple(entities),
                maximum_fidelity=0.3,
                policy=CellGeometryTransitionPolicy(
                    reconstruction="bounded_chart_deformation"
                ),
                budget=budget,
                certificate_limits=MeshCertificateLimits(),
                validity_policy=CellValidityPolicy(),
            )


def test_corner_only_sphere_reconstruction_does_not_realize_unused_directions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._sphere_chart_deformation as owner
    import phydrax.geometry._sphere_material_atlas as material_owner

    domain, source, atlas = accepted_sphere_material(1.0)
    strata = material_owner.sphere_material_strata(domain, source)

    def forbid_direction(*args, **kwargs):
        raise AssertionError("prescribed corner nodes need no preliminary source realization")

    monkeypatch.setattr(material_owner, "corner_direction", forbid_direction)
    work, memory = 10_000_000, 256 * 1024**2
    with NativeExecutionScope(
        NativeExecutionLimits(2 * work, work, 0, memory, math.inf)
    ) as scope:
        with scope.batch(2 * work, queries_upper=work, bytes_upper=memory) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=work, maximum_memory_bytes=memory
            )
            prepared = owner._prepare_sphere_reconstruction_nodes(
                source_atlas=atlas,
                target_mesh=source.mesh,
                target_layout=source.geometry,
                cell_patches=strata.patches,
                corner_source_dimensions=strata.dimensions,
                corner_source_indices=strata.indices,
                corner_source_parameters=strata.parameters,
                corner_source_occurrence_paths=strata.occurrence_paths,
                corner_source_entity_ids=strata.entity_ids,
                maximum_fidelity=0.3,
                policy=CellGeometryTransitionPolicy(reconstruction="bounded_chart_deformation"),
                certificate_limits=MeshCertificateLimits(),
                validity_policy=CellValidityPolicy(),
                budget=budget,
                native_ticket=ticket,
            )
    assert prepared.directions is None
    assert prepared.prescribed_routes is not None
    assert prepared.planned_evaluations == 3 * prepared.count


def test_reconstruction_restore_retains_instance_and_work_evidence(
    sphere_reconstruction: SphereGeometryReconstruction,
) -> None:
    identity = sphere_reconstruction.reconstruction_id
    counts = (
        sphere_reconstruction.evaluation_count,
        sphere_reconstruction.preparation_work_units,
    )
    coordinates = np.asarray(sphere_reconstruction.geometry.coordinates).copy()
    restored = SphereGeometryReconstruction.from_stored_fields(sphere_reconstruction)
    assert restored is sphere_reconstruction
    assert restored.reconstruction_id == identity
    assert (restored.evaluation_count, restored.preparation_work_units) == counts
    np.testing.assert_array_equal(restored.geometry.coordinates, coordinates)


def test_reconstruction_restore_refuses_stale_topology_binding(
    sphere_reconstruction: SphereGeometryReconstruction,
) -> None:
    stale = replace(sphere_reconstruction, target_topology_id="stale")
    with pytest.raises(ValueError, match=r"sphere_reconstruction\.binding"):
        SphereGeometryReconstruction.from_stored_fields(stale)


def test_surface_restore_refuses_wrong_record_kind(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    with pytest.raises(TypeError, match="value"):
        PreparedSurfaceChartDeformation.from_stored_fields(
            sphere_chart,  # ty: ignore[invalid-argument-type]
        )


def test_sphere_restore_refuses_wrong_record_kind(
    surface_chart: PreparedSurfaceChartDeformation,
) -> None:
    with pytest.raises(TypeError, match="value"):
        PreparedSphereChartDeformation.from_stored_fields(
            surface_chart,  # ty: ignore[invalid-argument-type]
        )


def test_reconstruction_restore_refuses_wrong_record_kind(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    with pytest.raises(TypeError, match="value"):
        SphereGeometryReconstruction.from_stored_fields(
            sphere_chart,  # ty: ignore[invalid-argument-type]
        )


@pytest.mark.parametrize("record_name", ["surface_chart", "sphere_chart", "sphere_reconstruction"])
def test_explicit_restore_uses_actual_request_and_exact_work_boundary(
    record_name: str, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cold records re-earn proof on the caller, never a hidden host ticket."""
    import phydrax.discretization._sphere_chart_records as sphere_owner

    record = request.getfixturevalue(record_name)

    def forbid_ticket(*args, **kwargs):
        raise AssertionError("explicit restoration must not open a HostBatchTicket")

    monkeypatch.setattr(sphere_owner, "HostBatchTicket", forbid_ticket)
    memory = 256 * 1024**2

    def restore(cap: int) -> int:
        with NativeExecutionScope(
            NativeExecutionLimits(cap, 0, 0, memory, math.inf)
        ) as scope:
            with scope.batch(cap, bytes_upper=memory) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=cap, maximum_memory_bytes=memory
                )
                with budget.activate(), ticket.native_target():
                    restored = type(record).from_stored_fields(record, budget=budget)
                    assert restored is record
                    assert restored.source_geometry_id == record.source_geometry_id
                    assert restored.target_geometry_id == record.target_geometry_id
                work = budget.work_units
        assert scope.completed_evidence().work_counters.construction_work == work
        return work

    work = restore(10_000_000)
    assert work > 0
    assert restore(work) == work
    with pytest.raises(CoordinateEnclosureResourceError):
        restore(work - 1)


def test_explicit_reconstruction_restore_refuses_uncertified_status(
    sphere_reconstruction: SphereGeometryReconstruction,
) -> None:
    from phydrax.geometry._global_embedding import (
        EmbeddingCounts, GlobalEmbeddingCertificate,
    )
    from phydrax.geometry._mesh_certificates import MeshCertificateFinding

    certificate = sphere_reconstruction.target_embedding
    unresolved = GlobalEmbeddingCertificate(
        certificate.binding,
        certificate.validity_certificate_id,
        (MeshCertificateFinding("maximum_samples", "unresolved", "mesh"),),
        (),
        counts=EmbeddingCounts(
            certificate.cell_count, certificate.boundary_facet_count,
            certificate.shell_count, certificate.candidate_pair_count,
            certificate.ray_test_count, certificate.subdivision_piece_count,
            certificate.periodic_image_count,
        ),
    )
    assert unresolved.status == "unresolved"
    stale = replace(
        sphere_reconstruction,
        target_embedding=unresolved,
    )
    with coordinate_reference_budget() as budget:
        with pytest.raises(ValueError, match=r"sphere_chart\.require_bound\.validity"):
            SphereGeometryReconstruction.from_stored_fields(stale, budget=budget)


def test_explicit_sphere_restore_reearns_foreign_request_proof(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    endpoints = (
        sphere_chart.source_mesh, sphere_chart.source_geometry,
        sphere_chart.target_mesh, sphere_chart.target_geometry,
    )
    with coordinate_reference_budget() as original:
        issued = sphere_chart.require_bound(*endpoints, budget=original)
    with coordinate_reference_budget() as caller:
        with pytest.raises(ValueError, match="foreign or stale proof"):
            issued.require(sphere_chart, *endpoints, budget=caller)
        assert PreparedSphereChartDeformation.from_stored_fields(
            sphere_chart, budget=caller
        ) is sphere_chart
        work = caller.work_units
        sphere_chart.pieces.authenticate(budget=caller)
        authentication_work = caller.work_units - work
        work = caller.work_units
        assert work > 0
        current = sphere_chart.require_bound(*endpoints, budget=caller)
        current.require(sphere_chart, *endpoints, budget=caller)
        assert caller.work_units == work + 2 * authentication_work
        assert current.h1_cache is not issued.h1_cache


def test_sphere_piece_bank_preserves_exact_logical_rows(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    from phydrax.discretization import SphereChartPieceBank

    original = sphere_chart.pieces
    assert isinstance(original, SphereChartPieceBank)
    with coordinate_reference_budget() as budget:
        rows = tuple(original.iter_pieces(budget=budget))
        rebuilt = SphereChartPieceBank.from_pieces(rows, budget=budget)
        actual = tuple(rebuilt.iter_pieces(budget=budget))
    assert rebuilt.bank_id == original.bank_id
    assert len(actual) == len(rows)
    for expected, retained in zip(rows, actual, strict=True):
        assert retained.piece_id == expected.piece_id
        assert retained.source_cell_global_id == expected.source_cell_global_id
        assert retained.target_cell_global_id == expected.target_cell_global_id
        assert retained.occurrence_path == expected.occurrence_path
        assert retained.exact_source_reference_vertices == expected.exact_source_reference_vertices
        assert retained.exact_target_reference_vertices == expected.exact_target_reference_vertices
        assert retained.projective_map.map_id == expected.projective_map.map_id
        assert retained.projective_bounds == expected.projective_bounds
        np.testing.assert_array_equal(
            retained.source_reference_vertices, expected.source_reference_vertices
        )
        np.testing.assert_array_equal(
            retained.target_reference_vertices, expected.target_reference_vertices
        )


def test_sphere_piece_bank_roundtrip_uses_array_artifacts(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    import json

    from phydrax._model._structure import (
        model_from_logical_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.discretization import SphereChartPieceBank
    from phydrax.lifecycle import register_meshing_source_artifacts

    register_meshing_source_artifacts()
    bank = sphere_chart.pieces
    recipe = model_structure_recipe(bank)
    arrays = model_recipe_array_values(bank, recipe, prefix="sphere-piece-bank")
    assert len(json.dumps(recipe, separators=(",", ":")).encode()) <= 1024 * 1024
    assert len(
        json.dumps(model_structure_recipe(sphere_chart), separators=(",", ":")).encode()
    ) <= 1024 * 1024
    restored = model_from_logical_array_recipe(recipe, arrays, prefix="sphere-piece-bank")
    assert isinstance(restored, SphereChartPieceBank)
    assert restored is not bank
    with coordinate_reference_budget() as cold:
        restored.authenticate(budget=cold)
        assert cold.work_units > 0
        assert tuple(piece.piece_id for piece in restored.iter_pieces(budget=cold)) == (
            tuple(piece.piece_id for piece in bank)
        )
    assert restored.bank_id == bank.bank_id


@pytest.mark.parametrize(
    "field_name",
    [
        "atom_bytes", "atom_offsets", "text_bytes", "text_offsets",
        "occurrence_rows", "occurrence_paths", "map_rows", "map_representatives",
        "bound_rows", "piece_rows", "boundary_labels", "reference_vertices",
        "displacement_bounds",
    ],
)
def test_sphere_piece_bank_authenticates_every_mutable_array(
    sphere_chart: PreparedSphereChartDeformation, field_name: str,
) -> None:
    bank = sphere_chart.pieces
    # Warm pure row decoding first: it cannot become authority over changed bytes.
    _ = bank[0]
    changed = np.array(getattr(bank, field_name), copy=True)
    if changed.size == 0:
        changed = np.zeros((1,), dtype=changed.dtype)
    elif np.issubdtype(changed.dtype, np.floating):
        changed.flat[0] = np.nextafter(changed.flat[0], np.inf)
    else:
        changed.flat[0] ^= 1
    forged = replace(bank, **{field_name: changed})
    with coordinate_reference_budget() as budget:
        with pytest.raises(ValueError):
            forged.authenticate(budget=budget)
        with pytest.raises(ValueError):
            forged.metadata_rows(budget=budget)
    with pytest.raises(ValueError):
        _ = forged[0]


def test_sphere_piece_bank_row_access_cannot_mutate_private_derivation(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    bank = sphere_chart.pieces
    original = bank[0]
    expected = np.array(original.source_reference_vertices, copy=True)
    exposed = np.asarray(original.source_reference_vertices)
    if exposed.flags.writeable:
        exposed.flat[0] += 1.0
    else:
        with pytest.raises(ValueError):
            exposed.flat[0] += 1.0
    np.testing.assert_array_equal(bank[0].source_reference_vertices, expected)


def test_sphere_piece_bank_authentication_observed_work_boundary(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    bank = sphere_chart.pieces
    with coordinate_reference_budget() as budget:
        bank.authenticate(budget=budget)
        required = budget.required_work_units
    assert required > 0
    with coordinate_reference_budget() as admitted:
        with admitted.bound_stage(required, 10**8):
            bank.authenticate(budget=admitted)
    with coordinate_reference_budget() as refused:
        with pytest.raises(CoordinateEnclosureResourceError):
            with refused.bound_stage(required - 1, 10**8):
                bank.authenticate(budget=refused)


def test_sphere_request_proof_reauthenticates_changed_bank_bytes(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    endpoints = (
        sphere_chart.source_mesh, sphere_chart.source_geometry,
        sphere_chart.target_mesh, sphere_chart.target_geometry,
    )
    bank = sphere_chart.pieces
    original = bank.reference_vertices
    changed = np.array(original, copy=True)
    changed.flat[0] = np.nextafter(changed.flat[0], np.inf)
    with coordinate_reference_budget() as budget:
        validation = sphere_chart.require_bound(*endpoints, budget=budget)
        # Preserve the exact bank/deformation object identities used by the
        # issued proof; only public numerical bytes change.
        object.__setattr__(bank, "reference_vertices", changed)
        try:
            with pytest.raises(ValueError, match="piece_bank"):
                sphere_chart.require_bound(*endpoints, budget=budget)
            with pytest.raises(ValueError, match="piece_bank"):
                validation.require(sphere_chart, *endpoints, budget=budget)
        finally:
            object.__setattr__(bank, "reference_vertices", original)


@pytest.mark.parametrize(
    "field_name", ["source_cell_global_id", "target_cell_global_id", "geometry_entity_id", "occurrence_path"]
)
def test_sphere_piece_bank_refuses_foreign_row_identity(
    sphere_chart: PreparedSphereChartDeformation, field_name: str,
) -> None:
    from phydrax.discretization import SphereChartPieceBank

    piece = sphere_chart.pieces[0]
    original = getattr(piece, field_name)
    if isinstance(original, int):
        foreign = original + 1
    elif isinstance(original, str):
        foreign = original + "-foreign"
    else:
        foreign = (*original, "foreign-occurrence")
    changed = replace(piece, **{field_name: foreign})
    with coordinate_reference_budget() as budget:
        with pytest.raises(ValueError):
            SphereChartPieceBank.from_pieces((changed,), budget=budget)


def test_sphere_public_fraction_rows_do_not_expose_cached_atom_authority(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    bank = sphere_chart.pieces
    piece = bank[0]
    exposed = piece.exact_source_reference_vertices[0][0]
    expected = exposed.as_integer_ratio()
    # Python's Fraction numerical contract is immutable, but private slots can
    # still be assigned. Public row objects are not the private cache's atoms.
    object.__setattr__(exposed, "_numerator", expected[0] + expected[1])
    assert bank[0].exact_source_reference_vertices[0][0].as_integer_ratio() == expected
    with coordinate_reference_budget() as budget:
        bank.authenticate(budget=budget)
        actual = next(bank.iter_pieces(budget=budget))
        assert actual.exact_source_reference_vertices[0][0].as_integer_ratio() == expected


def test_sphere_bank_metadata_does_not_repeat_numerical_row_construction(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    bank = sphere_chart.pieces
    with coordinate_reference_budget() as budget:
        bank.authenticate(budget=budget)
        start = budget.work_units
        metadata = bank.metadata_rows(budget=budget)
        metadata_work = budget.work_units - start
        start = budget.work_units
        rows = tuple(bank.iter_pieces(budget=budget))
        numerical_work = budget.work_units - start
    assert metadata == tuple(
        (piece.piece_id, piece.geometry_entity_id, piece.occurrence_path) for piece in rows
    )
    assert metadata_work < numerical_work


@pytest.mark.parametrize(
    ("field_name", "foreign"),
    [
        ("dimension", 2),
        ("singular_tolerance", 1e-11),
        ("maximum_condition", 1e11),
        ("refinement_iterations", 2),
        ("plan_id", "foreign-plan"),
    ],
)
def test_stored_sphere_inverse_controls_bind_the_complete_plan(field_name, foreign):
    from copy import copy

    from phydrax.linalg._small_batched import SmallLinearSolvePlan

    plan = SmallLinearSolvePlan(3)
    with coordinate_reference_budget() as budget:
        assert SmallLinearSolvePlan.from_stored_fields(plan, budget=budget) is plan
        changed = copy(plan)
        object.__setattr__(changed, field_name, foreign)
        with pytest.raises(ValueError, match="plan_id"):
            SmallLinearSolvePlan.from_stored_fields(
                changed, budget=budget
            )


def test_private_sphere_operand_borrowing_does_not_repeat_public_normalization(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    bank = sphere_chart.pieces
    with coordinate_reference_budget() as budget:
        bank.authenticate(budget=budget)
        start = budget.work_units
        public = tuple(bank.iter_pieces(budget=budget))
        public_work = budget.work_units - start
        start = budget.work_units
        private = tuple(bank._iter_kernel_pieces(budget=budget))
        private_work = budget.work_units - start
        assert private_work < public_work
        assert tuple(piece.piece_id for piece in private) == tuple(
            piece.piece_id for piece in public
        )
        exposed = public[0].exact_source_reference_vertices[0][0]
        expected = private[0].exact_source_reference_vertices[0][0].as_integer_ratio()
        object.__setattr__(exposed, "_numerator", exposed.numerator + exposed.denominator)
        assert private[0].exact_source_reference_vertices[0][0].as_integer_ratio() == expected


def test_kernel_fan_reuses_actual_map_without_changing_canonical_payload(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    from phydrax.discretization._sphere_chart_piece_bank import (
        SphereChartPieceBank, SphereChartPieceBankBuilder,
    )

    piece = sphere_chart.pieces[0]
    # A codec-only repeated stream tests operand reuse, not material coverage.
    with coordinate_reference_budget() as public_budget:
        expected = SphereChartPieceBank.from_pieces(
            (piece, piece), budget=public_budget
        )
        public_work = public_budget.work_units
    with coordinate_reference_budget() as kernel_budget:
        builder = SphereChartPieceBankBuilder(budget=kernel_budget)
        try:
            mapping = builder._prepare_kernel_map(piece.projective_map)
            builder._append_piece(piece, mapping)
            builder._append_piece(piece, mapping)
            actual = builder.finish()
        finally:
            builder.close()
        assert kernel_budget.work_units < public_work
    assert actual.bank_id == expected.bank_id
    for name in (
        "atom_bytes", "atom_offsets", "map_rows", "map_representatives",
        "bound_rows", "piece_rows", "reference_vertices",
    ):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_private_fan_map_cannot_cross_builder_ownership(
    sphere_chart: PreparedSphereChartDeformation,
) -> None:
    from phydrax.discretization._sphere_chart_piece_bank import SphereChartPieceBankBuilder

    piece = sphere_chart.pieces[0]
    with coordinate_reference_budget() as budget:
        source = SphereChartPieceBankBuilder(budget=budget)
        target = SphereChartPieceBankBuilder(budget=budget)
        try:
            mapping = source._prepare_kernel_map(piece.projective_map)
            with pytest.raises(ValueError, match="piece_bank"):
                target._append_piece(piece, mapping)
        finally:
            source.close()
            target.close()
