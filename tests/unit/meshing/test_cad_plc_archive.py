from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path
from typing import Literal

import equinox as eqx
import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.lifecycle import Composition, CompositionEntry
from phydrax.meshing._association import GeometryAssociation, GeometryAssociationKind
from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, write_model_artifact
from phydrax.meshing._plc_source import PiecewiseLinearComplex


pytest_plugins = ("tests.unit.meshing.test_multiregion_certification",)


def _plc_identity(plc: PiecewiseLinearComplex, /) -> str:
    charts, embedding, fidelity = (
        plc.cad_source_charts,
        plc.cad_embedding,
        plc.cad_fidelity,
    )
    assert charts is not None and embedding is not None and fidelity is not None
    payload = {
        "kind": "piecewise-linear-complex",
        "vertices": array_tree_fingerprint(plc.vertices),
        "polygon_offsets": array_tree_fingerprint(plc.polygon_offsets),
        "polygon_vertices": array_tree_fingerprint(plc.polygon_vertices),
        "polygon_facets": array_tree_fingerprint(plc.polygon_facets),
        "facet_regions": array_tree_fingerprint(plc.facet_regions),
        "segments": array_tree_fingerprint(plc.segments),
        "vertex_ids": plc.vertex_ids,
        "vertex_occurrence_paths": plc.vertex_occurrence_paths,
        "facet_ids": plc.facet_ids,
        "facet_occurrence_paths": plc.facet_occurrence_paths,
        "segment_ids": plc.segment_ids,
        "segment_occurrence_paths": plc.segment_occurrence_paths,
        "plc_edge_vertices": plc.plc_edge_vertices,
        "plc_edge_ids": plc.plc_edge_ids,
        "plc_edge_occurrence_paths": plc.plc_edge_occurrence_paths,
        "region_ids": plc.region_ids,
        "region_occurrence_paths": plc.region_occurrence_paths,
        "boundary": plc.boundary,
        "cad_incidence": (
            charts.source_chart_id,
            plc.cad_fidelity_tolerance,
            plc.cad_maximum_normal_deviation,
            embedding.certificate_id,
            fidelity.certificate_id,
            array_tree_fingerprint(plc.cad_vertex_rows),
            array_tree_fingerprint(plc.cad_segment_edge_ids),
            array_tree_fingerprint(plc.cad_polygon_cell_ids),
        ),
    }
    if plc.geometry is not None:
        payload["geometry"] = array_tree_fingerprint(plc.geometry)
    if plc.facet_planes is not None and plc.facet_plane_tolerances is not None:
        payload["facet_planes"] = array_tree_fingerprint(plc.facet_planes)
        payload["facet_plane_tolerances"] = array_tree_fingerprint(
            plc.facet_plane_tolerances
        )
    return canonical_fingerprint(payload)


def _changed_interval(association: GeometryAssociation, /) -> GeometryAssociation:
    intervals = list(association.source_parameter_intervals)
    assert intervals[0] is not None
    first, last = intervals[0]
    intervals[0] = (first, (first + last) / 2)
    return GeometryAssociation(
        association.association_kind,
        association.source_id,
        association.source_revision,
        association.target_entity_set_id,
        association.target_global_ids,
        association.source_entity_ids,
        association.residuals,
        resolved=association.resolved,
        ambiguous=association.ambiguous,
        exact=association.exact,
        source_dimensions=association.source_dimensions,
        source_indices=association.source_indices,
        source_roles=association.source_roles,
        source_occurrence_paths=association.source_occurrence_paths,
        source_region_sides=association.source_region_sides,
        source_parameter_intervals=tuple(intervals),
        source_parameter_points=association.source_parameter_points,
        parameters=association.parameters,
        orientations=association.orientations,
        parent_dimensions=association.parent_dimensions,
        parent_ids=association.parent_ids,
        parent_association_id=association.parent_association_id,
        provenance=association.provenance,
    )


def _forged_plc(
    plc: PiecewiseLinearComplex,
    defect: Literal["coedge", "interval", "certificate", "status", "claim", "region"],
    /,
) -> PiecewiseLinearComplex:
    forged = copy.copy(plc)
    charts = plc.cad_source_charts
    assert charts is not None
    if defect in ("coedge", "interval"):
        charts = copy.copy(charts)
        if defect == "coedge":
            rows = np.array(charts.boundary_occurrences, copy=True)
            positions = np.argwhere(rows[..., 0] >= 0)
            assert positions.size
            row, slot = (int(value) for value in positions[0])
            patch, loop, use = (int(value) for value in rows[row, slot])
            uses = charts.domain.patches[patch].loops[loop]
            assert len(uses) > 1
            rows[row, slot, 2] = (use + 1) % len(uses)
            object.__setattr__(charts, "boundary_occurrences", rows)
        else:
            associations = list(charts.associations)
            index = next(
                i
                for i, value in enumerate(associations)
                if value.association_kind is GeometryAssociationKind.CURVE
            )
            associations[index] = _changed_interval(associations[index])
            object.__setattr__(charts, "associations", tuple(associations))
        object.__setattr__(charts, "source_chart_id", charts._identity())
        object.__setattr__(forged, "cad_source_charts", charts)
    elif defect in ("certificate", "status", "claim"):
        fidelity = copy.copy(plc.cad_fidelity)
        assert fidelity is not None
        if defect == "certificate":
            object.__setattr__(fidelity, "tolerance", fidelity.tolerance + 1.0)
        elif defect == "status":
            object.__setattr__(fidelity, "status", "violated")
        else:
            bound = fidelity.maximum_normal_deviation_bound
            object.__setattr__(
                fidelity,
                "maximum_normal_deviation_bound",
                1.0 if bound is None else bound + 1.0,
            )
        object.__setattr__(fidelity, "certificate_id", fidelity._derived_certificate_id())
        object.__setattr__(forged, "cad_fidelity", fidelity)
    else:
        regions = np.array(plc.facet_regions, copy=True)
        regions[0] = regions[0, ::-1]
        object.__setattr__(forged, "facet_regions", regions)
    object.__setattr__(forged, "complex_id", _plc_identity(forged))
    return forged


def _composition(plc: PiecewiseLinearComplex, /) -> Composition:
    charts = plc.cad_source_charts
    assert charts is not None
    source = CompositionEntry(
        charts,
        entry_id="cad/source-charts",
        role="model-state",
        owner_id=charts.source_id,
        structure_id=charts.source_chart_id,
        revision_id=charts.source_revision,
        semantics_id="cad-source-charts",
    )
    complex_entry = CompositionEntry(
        plc,
        entry_id="cad/plc",
        role="model-state",
        owner_id=charts.source_id,
        structure_id=plc.complex_id,
        revision_id=charts.source_revision,
        semantics_id="cad-piecewise-linear-complex",
        dependencies=(source.binding("revision"),),
    )
    return Composition((source, complex_entry), boundary_id="cad-plc-cold-replay")


def _run_cold(path: Path, expected_id: str, /, *, rejected: bool) -> str:
    script = (
        "import sys; from phydrax.lifecycle import Composition; "
        "from phydrax.meshing._model_codec import MODEL_ARCHIVE_LIMITS, read_model_artifact, restore_model_archive_value; "
        "c=read_model_artifact(sys.argv[1], expected_type=Composition, restore=restore_model_archive_value, limits=MODEL_ARCHIVE_LIMITS); "
        "p=c.value('cad/plc'); s=c.value('cad/source-charts'); "
        "assert p.complex_id == sys.argv[2]; assert p.cad_source_charts.source_chart_id == s.source_chart_id; "
        "assert p.cad_embedding.status == 'certified'; assert p.cad_fidelity.status == 'certified'; "
        "assert p.cad_segment_edge_ids is not None"
    )
    child = subprocess.run(
        [sys.executable, "-c", script, str(path), expected_id],
        check=False,
        capture_output=True,
        text=True,
    )
    if rejected:
        assert child.returncode != 0, "forged scientific authority was accepted"
        assert (
            "ValueError" in child.stderr or "ArrayArchiveCorruptionError" in child.stderr
        ), child.stderr
    else:
        assert child.returncode == 0, child.stderr
    return child.stderr


def test_cad_plc_composition_replays_in_cold_process(
    cad_plc: PiecewiseLinearComplex,
    tmp_path: Path,
) -> None:
    path = write_model_artifact(
        tmp_path / "cad-plc.phx", _composition(cad_plc), limits=MODEL_ARCHIVE_LIMITS
    )
    _run_cold(path, cad_plc.complex_id, rejected=False)


@pytest.mark.parametrize(
    "defect", ("coedge", "interval", "certificate", "status", "claim", "region")
)
def test_cad_plc_cold_archive_refuses_rehashed_forgery(
    cad_plc: PiecewiseLinearComplex,
    tmp_path: Path,
    defect: Literal["coedge", "interval", "certificate", "status", "claim", "region"],
) -> None:
    forged = _forged_plc(cad_plc, defect)
    if defect != "claim":
        assert forged.complex_id != cad_plc.complex_id
    path = write_model_artifact(
        tmp_path / f"forged-{defect}.phx",
        _composition(forged),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    _run_cold(path, forged.complex_id, rejected=True)


def _with_face_residuals(
    plc: PiecewiseLinearComplex, residual: float | None, /
) -> PiecewiseLinearComplex:
    """Replace stored face residual evidence; ``None`` re-rounds it by two ulps."""
    charts = copy.copy(plc.cad_source_charts)
    assert charts is not None
    faces = charts.mesh.entity_set(2).entity_set_id
    index = next(
        i
        for i, value in enumerate(charts.associations)
        if value.target_entity_set_id == faces
    )
    table = charts.associations[index]
    residuals = np.array(table.residuals, dtype=np.float64, copy=True)
    if residual is None:
        residuals = np.nextafter(np.nextafter(residuals, np.inf), np.inf)
    else:
        residuals[0] = residual
    associations = list(charts.associations)
    associations[index] = eqx.tree_at(lambda value: value.residuals, table, residuals)
    object.__setattr__(charts, "associations", tuple(associations))
    object.__setattr__(charts, "source_chart_id", charts._identity())
    forged = copy.copy(plc)
    object.__setattr__(forged, "cad_source_charts", charts)
    object.__setattr__(forged, "complex_id", _plc_identity(forged))
    return forged


def test_cad_plc_identity_excludes_platform_rounded_face_residuals(
    cad_plc: PiecewiseLinearComplex,
    tmp_path: Path,
) -> None:
    """Another platform's sampled chord residuals restore under one exact identity."""
    rounded = _with_face_residuals(cad_plc, None)
    charts, original = rounded.cad_source_charts, cad_plc.cad_source_charts
    assert charts is not None and original is not None
    assert charts.source_chart_id == original.source_chart_id
    assert rounded.complex_id == cad_plc.complex_id
    path = write_model_artifact(
        tmp_path / "rounded-residuals.phx",
        _composition(rounded),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    _run_cold(path, cad_plc.complex_id, rejected=False)


@pytest.mark.parametrize("residual", (-1.0e-16, float("nan"), float("inf")))
def test_cad_plc_cold_archive_refuses_forged_face_residual(
    cad_plc: PiecewiseLinearComplex,
    tmp_path: Path,
    residual: float,
) -> None:
    forged = _with_face_residuals(cad_plc, residual)
    assert forged.complex_id == cad_plc.complex_id
    path = write_model_artifact(
        tmp_path / "forged-residual.phx",
        _composition(forged),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    stderr = _run_cold(path, forged.complex_id, rejected=True)
    assert "AssociationResidualError" in stderr, stderr


def test_cad_plc_static_banks_are_exact_dynamic_columns(
    cad_plc: PiecewiseLinearComplex,
) -> None:
    from phydrax._archive_columns import TextRowsArchiveRecipe
    from phydrax._model._structure import (
        model_archive_projection,
        PiecewiseLinearComplexArchiveRecipe,
    )

    recipe = model_archive_projection(cad_plc)
    assert type(recipe) is PiecewiseLinearComplexArchiveRecipe
    decoded = recipe.decode()
    for name in (
        "vertex_ids",
        "vertex_occurrence_paths",
        "facet_ids",
        "facet_occurrence_paths",
        "segment_ids",
        "segment_occurrence_paths",
        "vertex_entity_ids",
        "facet_entity_ids",
        "segment_entity_ids",
        "plc_edge_vertices",
        "plc_edge_facets",
        "plc_edge_ids",
        "plc_edge_occurrence_paths",
        "plc_edge_entity_ids",
        "region_ids",
        "region_occurrence_paths",
        "region_entity_ids",
    ):
        assert getattr(decoded, name) == getattr(cad_plc, name)
    paths = recipe.fields["plc_edge_occurrence_paths"]
    assert type(paths) is TextRowsArchiveRecipe
    ends = np.array(paths.rows.ends, copy=True)
    assert ends.size
    ends[-1] += 1
    fields = dict(recipe.fields)
    fields["plc_edge_occurrence_paths"] = paths._replace(
        rows=paths.rows._replace(ends=ends)
    )
    with pytest.raises(ValueError):
        PiecewiseLinearComplexArchiveRecipe(fields).decode()


@pytest.mark.parametrize(
    ("left", "right"),
    (
        ((1,), (True,)),
        ((1,), (1.0,)),
        ((np.array([1], dtype=np.int32),), (np.array([1], dtype=np.int64),)),
        ({"control": 1}, {"control": True}),
        ({"control": 1}, {"other": 1}),
    ),
)
def test_cad_proof_recipe_comparison_refuses_kind_dtype_and_field_changes(
    left: object, right: object
) -> None:
    from phydrax.meshing._model_codec import _same_model_content

    assert not _same_model_content(left, right)
