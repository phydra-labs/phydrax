from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks._memory import host_allocation_peak
from phydrax._archive_columns import (
    ExactTensorArchiveRecipe,
    StaticColumnArchiveRecipe,
    TextRowsArchiveRecipe,
)
from phydrax._array_archive import ArrayArchiveCorruptionError, read_array_archive
from phydrax._model._structure import model_recipe_array_inventory
from phydrax.discretization._nested_reference_types import NestedReferenceWitnesses
from phydrax.geometry.implicit._source_charts import ImplicitRootChartEvidence
from phydrax.lifecycle._composition_rebind import Composition, CompositionEntry
from phydrax.meshing._archive_layer_columns import (
    ExactExpressionRowsArchiveRecipe,
    ImplicitChartRowsArchiveRecipe,
    layer_record_archive_recipe,
)
from phydrax.meshing._layer_curving_records import LayerColumnProjectionIdentity
from phydrax.meshing._model_codec import (
    MODEL_ARCHIVE_LIMITS,
    read_model_archive,
    read_model_archive_fields,
    read_model_artifact,
    write_model_archive,
    write_model_artifact,
)


def test_association_bank_preserves_exact_parameters_and_rejects_malformed_rows() -> None:
    from phydrax.meshing._association import (
        GeometryAssociation,
        GeometryAssociationBankArchiveRecipe,
        GeometryAssociationKind,
        GeometrySourceEntityRole,
    )

    first = GeometryAssociation(
        GeometryAssociationKind.CURVE,
        "source",
        "revision",
        "edges",
        np.asarray((7, 9)),
        ("curve-a", "curve-b"),
        np.asarray((-0.0, 0.0)),
        exact=True,
        source_roles=(GeometrySourceEntityRole.EDGE,) * 2,
        source_occurrence_paths=(("assembly", "a"), ("assembly", "b")),
        source_parameter_intervals=((Fraction(1, 3), Fraction(5, 3)), None),
        source_parameter_points=(None, Fraction(7, 11)),
        target_parameter_intervals=((Fraction(1, 4), Fraction(3, 4)), None),
    )
    second = GeometryAssociation(
        GeometryAssociationKind.PIECEWISE_LINEAR,
        "source",
        "revision",
        "faces",
        np.asarray((3,)),
        ("face",),
        np.asarray((0.0,)),
        exact=True,
        source_roles=(GeometrySourceEntityRole.FACET,),
        source_occurrence_paths=(("part",),),
        source_region_sides=(("material", None),),
        target_reference_pieces=(
            (
                (
                    21,
                    0,
                    (),
                    (
                        ((0, 1), (0, 1)),
                        ((1, 1), (0, 1)),
                        ((0, 1), (1, 1)),
                    ),
                ),
            ),
        ),
        source_boundary_witness_id="reference-witness",
    )
    bank = GeometryAssociationBankArchiveRecipe.from_associations((first, second))
    restored = bank.restore()
    assert tuple(value.association_id for value in restored) == (
        first.association_id,
        second.association_id,
    )
    assert np.signbit(np.asarray(restored[0].residuals)[0])
    rows = list(bank.rows.decode())
    rows[0] = (*rows[0], "forged")
    with pytest.raises(ValueError, match="trailing"):
        bank._replace(rows=TextRowsArchiveRecipe.encode(tuple(rows))).restore()
    metadata = list(bank.metadata.decode())
    metadata[0] = (*metadata[0][:-1], "self-rehashed-false-identity")
    with pytest.raises(ValueError, match="canonical owner"):
        bank._replace(metadata=TextRowsArchiveRecipe.encode(tuple(metadata))).restore()


def test_layer_exact_carrier_columns_roundtrip_and_reject_malformed_rows() -> None:
    from phydrax.meshing._archive_layer_columns import _decode_column
    from phydrax.meshing._layer_core_subdivision import (
        NearestWallPointEvidence,
        OverlapSourceStencils,
        RefinedColumnCertification,
        RefinedColumnEvidence,
    )

    overlap = OverlapSourceStencils(
        "source",
        "target",
        ((7, (("layer", 2, Fraction(1, 3)), ("plc", 4, Fraction(2, 3)))),),
    )
    column = RefinedColumnEvidence(
        ((2, Fraction(1, 3)), (5, Fraction(2, 3))),
        (7, 8),
        (
            NearestWallPointEvidence(
                Fraction(5, 7),
                (
                    (
                        11,
                        (Fraction(1, 2), Fraction(1, 3), Fraction(1, 6)),
                        Fraction(-3, 5),
                    ),
                ),
            ),
        ),
        Fraction(17, 9),
    )
    certification = RefinedColumnCertification(
        "source", "spec", "control", "policy", (column,)
    )
    for value in (overlap, certification):
        recipe = layer_record_archive_recipe(value)
        assert recipe is not None
        assert recipe.restore() == value
    with pytest.raises(ValueError, match="not reduced"):
        _decode_column(
            TextRowsArchiveRecipe.encode((("7", "layer", "2", "2", "4"),)), "overlap_rows"
        )
    with pytest.raises(ValueError):
        _decode_column(TextRowsArchiveRecipe.encode((("1", "2"),)), "refined_columns")


def test_facet_overlap_columns_preserve_exact_fields_and_reject_forgery() -> None:
    from phydrax.geometry._planar_coverage import FacetOverlap
    from phydrax.meshing._archive_layer_columns import _decode_column, _encode_column

    rows = (
        FacetOverlap(3, 9, (0, -1), Fraction(7, 11), "exact", "physical"),
        FacetOverlap(12, 4, (1, 2), Fraction(5, 3), "candidate", "reference"),
    )
    encoded = _encode_column(rows, "facet_overlaps")
    assert _decode_column(encoded, "facet_overlaps") == rows
    with pytest.raises(ValueError, match="not reduced"):
        _decode_column(
            TextRowsArchiveRecipe.encode(
                (("3", "9", "0", "-1", "2", "4", "exact", "physical"),)
            ),
            "facet_overlaps",
        )
    with pytest.raises(ValueError, match="invalid field set"):
        _decode_column(
            TextRowsArchiveRecipe.encode(
                (("3", "9", "0", "-1", "1", "2", "invented", "physical"),)
            ),
            "facet_overlaps",
        )


def _identity(count: int = 3) -> LayerColumnProjectionIdentity:
    return LayerColumnProjectionIdentity(
        np.zeros(count, np.int64),
        np.arange(count, dtype=np.int64),
        np.arange(count, dtype=np.int64),
        tuple(f"front/{i}" for i in range(count)),
        ((0, 1, 2),) * count,
        (("a", "b", "c"),) * count,
        ((10, 11, 12),) * count,
    )


def _composition(value: LayerColumnProjectionIdentity) -> Composition:
    return Composition(
        (
            CompositionEntry(
                value,
                entry_id="columns",
                role="model-state",
                owner_id="phydrax.meshing",
                structure_id=value.identity_id,
                revision_id=value.identity_id,
                semantics_id="layer-column-identity",
            ),
        ),
        boundary_id="column-archive",
    )


def _restore(value: Composition) -> Composition:
    stored = value.value("columns")
    rebuilt = LayerColumnProjectionIdentity(
        stored.layer_indices,
        stored.source_wall_cell_ids,
        stored.front_row_ids,
        stored.front_row_identities,
        stored.source_column_ids,
        stored.source_column_identities,
        stored.source_column_root_vertex_ids,
        allow_unbound=stored.allow_unbound,
    )
    assert rebuilt.identity_id == stored.identity_id
    return _composition(rebuilt)


def test_model_content_id_ignores_manifest_whitespace(tmp_path: Path) -> None:
    import json
    import zipfile

    from phydrax.meshing._model_codec import model_archive_content_id

    compact = write_model_artifact(
        tmp_path / "compact.phx",
        _composition(_identity()),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    legacy = tmp_path / "indented.phx"
    with (
        zipfile.ZipFile(compact) as source,
        zipfile.ZipFile(legacy, "w", compression=zipfile.ZIP_STORED) as destination,
    ):
        for name in source.namelist():
            payload = source.read(name)
            if name == "manifest.json":
                payload = json.dumps(
                    json.loads(payload), indent=2, sort_keys=True
                ).encode()
            destination.writestr(name, payload)
    assert model_archive_content_id(
        compact, limits=MODEL_ARCHIVE_LIMITS
    ) == model_archive_content_id(legacy, limits=MODEL_ARCHIVE_LIMITS)


def test_mapped_core_unbound_rows_roundtrip_without_fabricated_columns(
    tmp_path: Path,
) -> None:
    value = LayerColumnProjectionIdentity(
        np.asarray((0, -1)),
        np.asarray((10, -1)),
        np.asarray((0, -1)),
        ("front/0", ""),
        ((0, 1, 2), ()),
        (("a", "b", "c"), ()),
        ((10, 11, 12), ()),
        allow_unbound=True,
    )
    path = write_model_artifact(
        tmp_path / "mixed-columns.phx", _composition(value), limits=MODEL_ARCHIVE_LIMITS
    )
    restored = read_model_artifact(
        path, expected_type=Composition, restore=_restore, limits=MODEL_ARCHIVE_LIMITS
    ).value("columns")
    assert restored.allow_unbound
    assert restored.identity_id == value.identity_id
    with pytest.raises(ValueError, match="incomplete or misaligned"):
        LayerColumnProjectionIdentity(
            np.asarray((0, -1)),
            np.asarray((10, -1)),
            np.asarray((0, -1)),
            ("front/0", ""),
            ((0, 1, 2), ()),
            (("a", "b", "c"), ()),
            ((10, 11, 12), ()),
        )


def test_empty_exact_tensor_extents_are_backed_before_allocation() -> None:
    value = ((), (), ())
    recipe = ExactTensorArchiveRecipe.encode(value, rational=True)
    assert recipe.decode() == value
    with pytest.raises(ValueError, match="container evidence"):
        recipe._replace(shape=(10**12, 0)).decode()


def test_fidelity_root_and_nested_records_project_row_banks() -> None:
    import dataclasses

    from phydrax._model._structure import (
        _restore_dataclass,
        model_archive_projection,
        model_archive_root_projection,
        model_structure_matches_recipe,
        model_structure_recipe,
        restore_row_record_recipes,
    )
    from phydrax.geometry._source_fidelity_certificate import SourceFidelityCertificate
    from phydrax.meshing._model_codec import _register_model_archive_types

    fields = {field.name: None for field in dataclasses.fields(SourceFidelityCertificate)}
    fields.update(
        source_cell_global_ids=tuple(range(4096)),
        source_patch_ids=(0,) * 4096,
        source_occurrence_paths=((),) * 4096,
    )
    _register_model_archive_types()
    certificate = _restore_dataclass(SourceFidelityCertificate, fields)
    root = model_archive_root_projection(certificate)
    assert isinstance(root.source_cell_global_ids, ExactTensorArchiveRecipe)
    again = model_archive_root_projection(root)
    restored = restore_row_record_recipes(again)
    assert restored.source_cell_global_ids == certificate.source_cell_global_ids
    assert restored.source_patch_ids == certificate.source_patch_ids
    assert restored.source_occurrence_paths == certificate.source_occurrence_paths
    selected = restore_row_record_recipes(model_archive_projection(certificate))
    assert selected.source_cell_global_ids == certificate.source_cell_global_ids
    nested = model_archive_projection((certificate,))
    assert (
        restore_row_record_recipes(nested)[0].source_cell_global_ids
        == fields["source_cell_global_ids"]
    )
    recipe = model_structure_recipe(certificate)
    assert model_structure_matches_recipe(certificate, recipe)
    assert not model_structure_matches_recipe(certificate, {**recipe, "extra": True})
    for name in recipe["fields"]:
        changed = {
            **recipe,
            "fields": {**recipe["fields"], name: {"kind": "literal", "value": "changed"}},
        }
        assert not model_structure_matches_recipe(certificate, changed)


def test_nested_reference_exact_bank_uses_column_recipe() -> None:
    exact = (((Fraction(0),), (Fraction(1, 3),)),) * 4096
    witness = NestedReferenceWitnesses(
        np.arange(4096, dtype=np.int64),
        np.arange(4096, dtype=np.int64),
        np.zeros((4096, 2, 1)),
        fine_reference_exact_vertices=exact,
    )
    recipe = layer_record_archive_recipe(witness)
    assert recipe is not None
    assert isinstance(
        recipe.fields["fine_reference_exact_vertices"], ExactTensorArchiveRecipe
    )
    assert recipe.restore().fine_reference_exact_vertices == exact


def test_source_chart_witness_codec_sorts_rows_and_requires_ancestry_fields() -> None:
    from phydrax.geometry._source_boundary_records import SourceBoundaryChartWitness

    def witness(order: np.ndarray) -> SourceBoundaryChartWitness:
        return SourceBoundaryChartWitness(
            mesh_id="mesh",
            geometry_id="geometry",
            topology_id="topology",
            source_id="source",
            source_revision="revision",
            source_query_id="query",
            chart_dimension=2,
            reference_dimension=2,
            cell_global_ids=np.array([9, 3], dtype=np.int64)[order],
            entity_global_ids=np.array([9, 3], dtype=np.int64)[order],
            local_entity_indices=np.array([-1, -1], dtype=np.int32)[order],
            target_reference_vertices=np.tile(
                np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]), (2, 1, 1)
            )[order],
            source_chart_ids=tuple(("chart-a", "chart-b")[i] for i in order),
            source_entity_ids=tuple(("face-a", "face-b")[i] for i in order),
            occurrence_paths=((), ()),
            source_parameters=np.tile(
                np.array([[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]]), (2, 1, 1)
            )[order],
            orientations=np.array([1, -1], dtype=np.int8)[order],
        )

    first = witness(np.array([0, 1]))
    reordered = witness(np.array([1, 0]))
    assert first.witness_id == reordered.witness_id
    recipe = layer_record_archive_recipe(first)
    assert recipe is not None
    assert isinstance(recipe.fields["cad_root_cell_ids"], np.ndarray)
    assert recipe.restore().witness_id == first.witness_id
    missing = dict(recipe.fields)
    del missing["cad_plc"]
    with pytest.raises(ValueError, match="exact field set"):
        recipe._replace(fields=missing).restore()
    changed = dict(recipe.fields)
    changed["cad_root_cell_ids"] = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError, match="CAD ancestry"):
        recipe._replace(fields=changed).restore()


def test_face_image_archive_retains_exact_support_and_refuses_tampering(
    tmp_path: Path,
) -> None:
    from phydrax._model._structure import (
        model_archive_projection,
        restore_row_record_recipes,
        SourceBoundaryFaceImageWitnessArchiveRecipe,
        SourceBoundaryImagePullbackArchiveRecipe,
    )
    from phydrax.geometry._source_boundary_records import (
        SourceBoundaryFaceImageWitness,
        SourceBoundaryImagePullback,
    )
    from phydrax.meshing._model_codec import model_archive_content_id

    triangle = (
        ((0, 1), (0, 1), (0, 1)),
        ((1, 3), (0, 1), (0, 1)),
        ((0, 1), (1, 2), (0, 1)),
    )
    witness = SourceBoundaryFaceImageWitness(
        mesh_id="mesh",
        geometry_id="geometry",
        topology_id="topology",
        source_id="source",
        source_revision="revision",
        source_query_id="query",
        cell_global_ids=np.array([3], dtype=np.int64),
        entity_global_ids=np.array([5], dtype=np.int64),
        local_entity_indices=np.array([2], dtype=np.int64),
        source_chart_ids=("chart",),
        source_entity_ids=("face",),
        occurrence_paths=(("occurrence",),),
        source_region_sides=(("material", None),),
        orientations=np.array([1], dtype=np.int64),
        image_triangle_rational_tokens=(triangle,),
    )
    pullback = SourceBoundaryImagePullback(3, 2, triangle, "chart")
    projected = model_archive_projection((witness, pullback))
    image_recipe, pullback_recipe = projected
    assert type(image_recipe) is SourceBoundaryFaceImageWitnessArchiveRecipe
    assert type(pullback_recipe) is SourceBoundaryImagePullbackArchiveRecipe
    assert image_recipe.image_triangles.shape == (1, 3, 3)
    restored_witness, restored_pullback = restore_row_record_recipes(projected)
    assert restored_witness.witness_id == witness.witness_id
    assert restored_pullback == pullback

    changed_triangle = (
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(2, 3), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1, 2), Fraction(0)),
        ),
    )
    for changed in (
        image_recipe._replace(source_id="other-source"),
        image_recipe._replace(
            source_region_sides=StaticColumnArchiveRecipe.encode(((None, "material"),))
        ),
        image_recipe._replace(
            image_triangles=ExactTensorArchiveRecipe.encode(
                changed_triangle, rational=True
            )
        ),
    ):
        with pytest.raises(ValueError, match="canonical owner"):
            changed.restore()
    invalid_image = ExactTensorArchiveRecipe.encode(
        ((Fraction(0), Fraction(0)), (Fraction(1), Fraction(0))), rational=True
    )
    with pytest.raises(ValueError, match="3D points"):
        pullback_recipe._replace(image=invalid_image).restore()

    def composition(values: tuple[Any, ...]) -> Composition:
        entry = CompositionEntry(
            values,
            entry_id="image",
            role="model-state",
            owner_id="phydrax.geometry",
            structure_id=witness.witness_id,
            revision_id=witness.witness_id,
            semantics_id="exact-image-support",
        )
        return Composition((entry,), boundary_id="image-archive")

    path = write_model_artifact(
        tmp_path / "image.phx",
        composition((witness, pullback)),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    receipt = model_archive_content_id(path, limits=MODEL_ARCHIVE_LIMITS)
    restored = read_model_artifact(
        path,
        expected_type=Composition,
        restore=lambda raw: composition(raw.value("image")),
        limits=MODEL_ARCHIVE_LIMITS,
        expected_content_id=receipt,
    )
    reloaded_witness, reloaded_pullback = restored.value("image")
    assert reloaded_witness.witness_id == witness.witness_id
    assert reloaded_pullback == pullback
    with pytest.raises(ArrayArchiveCorruptionError, match="content ID"):
        read_model_artifact(
            path,
            expected_type=Composition,
            restore=lambda raw: raw,
            limits=MODEL_ARCHIVE_LIMITS,
            expected_content_id="0" * 64,
        )


def test_column_identity_archive_preserves_constructor_identity(tmp_path: Path) -> None:
    value = _identity(4096)
    path = write_model_artifact(
        tmp_path / "columns.phx", _composition(value), limits=MODEL_ARCHIVE_LIMITS
    )
    restored = read_model_artifact(
        path, expected_type=Composition, restore=_restore, limits=MODEL_ARCHIVE_LIMITS
    )
    assert restored.value("columns").identity_id == value.identity_id
    assert (
        restored.value("columns").source_column_identities
        == value.source_column_identities
    )


@pytest.mark.parametrize("change", ("length", "dtype", "index", "fields", "identity"))
def test_malformed_layer_columns_refuse_before_owner_replay(
    tmp_path: Path, change: str
) -> None:
    source = write_model_artifact(
        tmp_path / "source.phx", _composition(_identity()), limits=MODEL_ARCHIVE_LIMITS
    )
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)
    inventory = model_recipe_array_inventory(
        manifest["model_recipe"], prefix="model/leaves", limits=MODEL_ARCHIVE_LIMITS
    )
    if change == "fields":

        def remove_field(node: Any) -> bool:
            if isinstance(node, dict):
                if str(node.get("type", "")).endswith(":LayerRecordArchiveRecipe"):
                    node["items"][1]["items"].pop()
                    return True
                return any(remove_field(child) for child in node.values())
            if isinstance(node, list):
                return any(remove_field(child) for child in node)
            return False

        assert remove_field(manifest["model_recipe"])
    elif change == "identity":
        entry = next(item for item in inventory if "source_wall_cell_ids" in item.path)
        values = arrays[entry.name].copy()
        values[0] += 1
        arrays[entry.name] = values
    else:
        entry = next(
            item
            for item in inventory
            if "source_column_identities" in item.path and item.shape == (9,)
        )
        rows = arrays[entry.name].copy()
        if change == "length":
            rows = rows[:-1]
        elif change == "dtype":
            rows = rows.astype(np.float64)
        else:
            rows[0] = 10
        arrays[entry.name] = rows
    replayed = False

    def forbidden(value: Any) -> Any:
        nonlocal replayed
        replayed = True
        raise AssertionError("Malformed columns reached proof replay")

    with pytest.raises((ArrayArchiveCorruptionError, ValueError, TypeError)):
        changed = write_model_archive(
            tmp_path / "changed.phx",
            manifest=manifest,
            arrays=arrays,
            limits=MODEL_ARCHIVE_LIMITS,
        )
        read_model_artifact(
            changed,
            expected_type=Composition,
            restore=forbidden,
            limits=MODEL_ARCHIVE_LIMITS,
        )
    assert not replayed


def test_immutable_prerequisites_charge_shared_column_owners_once(
    tmp_path: Path,
) -> None:
    identity = _identity()
    path = write_model_artifact(
        tmp_path / "shared-columns.phx",
        _composition(identity),
        limits=MODEL_ARCHIVE_LIMITS,
    )
    _, physical = read_array_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    _, occurrences = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    count = sum(value.size for value in physical.values())
    assert count < sum(value.size for value in occurrences.values())
    limits = replace(MODEL_ARCHIVE_LIMITS, max_total_array_elements=count)
    selected = read_model_archive_fields(
        path,
        expected_type=Composition,
        fields=("entries",),
        limits=limits,
        immutable_arrays=True,
    )
    restored = selected["entries"][0].value
    assert restored.identity_id == identity.identity_id
    np.testing.assert_array_equal(
        restored.source_wall_cell_ids, identity.source_wall_cell_ids
    )


@pytest.mark.parametrize(
    ("field", "expected"),
    (
        pytest.param("boundary_id", "column-archive", id="root-field"),
        pytest.param("entries.0.entry_id", "columns", id="tuple-field"),
    ),
)
@pytest.mark.parametrize("change", ("index", "identity"))
@pytest.mark.parametrize("immutable_arrays", (False, True), ids=("mutable", "immutable"))
def test_selected_fields_reject_unselected_column_corruption(
    tmp_path: Path, change: str, immutable_arrays: bool, field: str, expected: str
) -> None:
    source = write_model_artifact(
        tmp_path / "source.phx", _composition(_identity()), limits=MODEL_ARCHIVE_LIMITS
    )
    assert read_model_archive_fields(
        source,
        expected_type=Composition,
        fields=(field,),
        limits=MODEL_ARCHIVE_LIMITS,
        immutable_arrays=immutable_arrays,
    ) == {field: expected}
    manifest, arrays = read_model_archive(source, limits=MODEL_ARCHIVE_LIMITS)
    inventory = model_recipe_array_inventory(
        manifest["model_recipe"], prefix="model/leaves", limits=MODEL_ARCHIVE_LIMITS
    )
    entry = next(
        item
        for item in inventory
        if (
            "source_wall_cell_ids" in item.path
            if change == "identity"
            else "source_column_identities" in item.path and item.shape == (9,)
        )
    )
    values = arrays[entry.name].copy()
    values[0] += 10
    arrays[entry.name] = values
    changed = write_model_archive(
        tmp_path / "changed.phx",
        manifest=manifest,
        arrays=arrays,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_archive_fields(
            changed,
            expected_type=Composition,
            fields=(field,),
            limits=MODEL_ARCHIVE_LIMITS,
            immutable_arrays=immutable_arrays,
        )


def test_implicit_primitive_columns_preserve_every_field_and_refuse_misalignment() -> (
    None
):
    row = ImplicitRootChartEvidence(
        row=0,
        target_cell_global_id=3,
        target_entity_global_id=4,
        target_local_entity_index=1,
        source_cell_key=(0, 1, 2, 3),
        root_edge_keys=(
            ((0, 0, 0), (1, 0, 0)),
            ((1, 0, 0), (1, 1, 0)),
            ((1, 1, 0), (0, 0, 0)),
        ),
        status="certified_root_chart",
        field_residual_bound=0.1,
        target_to_root_lower_bound=0.2,
        target_to_root_bound=0.3,
        source_normal_deviation_bounds=(0.4, 0.5),
        chart_jacobian_lower=0.6,
        chart_injectivity_margin=0.7,
        source_gradient_lower=0.8,
        detail="exact chart",
    )
    recipe = ImplicitChartRowsArchiveRecipe.encode((row, row))
    assert tuple(ImplicitRootChartEvidence(**item) for item in recipe.decode()) == (
        row,
        row,
    )
    assert (
        ImplicitChartRowsArchiveRecipe.encode(recipe.decode()).decode() == recipe.decode()
    )
    with pytest.raises(ValueError, match="content digest"):
        recipe._replace(bounds=recipe.bounds[:1]).decode()
    changed_bounds = recipe.bounds.copy()
    changed_bounds[0, 1] += 0.01  # Still inside the valid [lower, upper] interval.
    with pytest.raises(ValueError, match="content digest"):
        recipe._replace(bounds=changed_bounds).decode()


def test_layer_record_recipe_requires_exact_owner_field_set() -> None:
    recipe = layer_record_archive_recipe(_identity())
    assert recipe is not None
    fields = dict(recipe.fields)
    fields["unknown"] = 1
    with pytest.raises(ValueError, match="exact field set"):
        recipe._replace(fields=fields).restore()
    with pytest.raises(TypeError, match="Unknown"):
        recipe._replace(record_type="object").restore()


def test_exact_column_preserves_large_rationals_and_repeated_static_rows() -> None:
    values = ((Fraction(2**100 + 1, 3), Fraction(-1, 7)),) * 5
    assert ExactTensorArchiveRecipe.encode(values, rational=True).decode() == values
    column = StaticColumnArchiveRecipe.encode(values)
    assert len(column.table) == 1
    assert column.decode() == values


@pytest.mark.parametrize("count", (4096, 16384), ids=("small", "large"))
def test_exact_tensor_encoding_workspace_is_bounded_by_payload(count: int) -> None:
    values = tuple((row, row + 1) for row in range(count))
    recipe, peak = host_allocation_peak(
        lambda: ExactTensorArchiveRecipe.encode(values, rational=False)
    )
    assert recipe.decode() == values
    payload_bytes = recipe.text.data.nbytes + recipe.text.ends.nbytes
    assert peak <= 4 * payload_bytes + 64 * 1024


def test_exact_tensor_retains_canonical_decimal_archive_bytes() -> None:
    values = (
        (Fraction(2, 3), Fraction(-5, 7)),
        (Fraction(0), Fraction(1)),
    )
    recipe = ExactTensorArchiveRecipe.encode(values, rational=True)
    np.testing.assert_array_equal(
        recipe.text.data, np.frombuffer(b"2/3-5/701", dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        recipe.text.ends, np.asarray((3, 7, 8, 9), dtype=np.int64)
    )
    assert recipe.decode() == values


def test_exact_reference_map_columns_preserve_polynomial_expression_kinds() -> None:
    from types import MappingProxyType

    from phydrax.discretization._coordinate_enclosure import RationalPolynomial
    from phydrax.discretization._nested_reference_types import ReferenceCellMapWitnesses

    polynomial: Mapping[tuple[int, ...], Fraction] = MappingProxyType(
        {(1,): Fraction(1, 3)}
    )
    rational = RationalPolynomial({(1,): Fraction(2)}, {(0,): Fraction(3)})
    witness = ReferenceCellMapWitnesses(
        np.asarray([0, 1], np.int64),
        np.asarray([2, 3], np.int64),
        "interval",
        "interval",
        np.asarray([[[0.0], [1.0]], [[0.0], [1.0]]]),
        (((Fraction(0),), (Fraction(1),)),) * 2,
        ((polynomial,), (rational,)),
    )
    recipe = layer_record_archive_recipe(witness)
    assert recipe is not None
    restored = recipe.restore()
    assert type(restored) is ReferenceCellMapWitnesses
    assert (
        restored.source_reference_exact_vertices
        == witness.source_reference_exact_vertices
    )
    first, second = restored.source_reference_chart_arguments
    assert type(first[0]) is MappingProxyType
    assert first[0] == polynomial
    assert type(second[0]) is RationalPolynomial
    assert second[0].numerator == rational.numerator
    assert second[0].denominator == rational.denominator


def test_text_rows_preserve_empty_unicode_and_first_occurrence_order() -> None:
    rows = ((), ("alpha", "λ"), ("λ",), (), ("alpha", "β"))
    recipe = TextRowsArchiveRecipe.encode(rows)
    assert recipe.table.decode() == ("alpha", "λ", "β")
    assert recipe.decode() == rows
    assert TextRowsArchiveRecipe.encode(()).decode() == ()
    assert TextRowsArchiveRecipe.encode(((), ())).decode() == ((), ())
    bad_indices = recipe.rows.values.copy()
    bad_indices[0] = 2
    with pytest.raises(ValueError, match="first-occurrence"):
        recipe._replace(rows=recipe.rows._replace(values=bad_indices)).decode()
    with pytest.raises(ValueError, match="partition"):
        recipe._replace(
            rows=recipe.rows._replace(ends=np.asarray([1], np.int64))
        ).decode()
    with pytest.raises(TypeError, match="int64"):
        recipe._replace(
            rows=recipe.rows._replace(values=recipe.rows.values.astype(np.float64))
        ).decode()


def test_exact_expression_columns_deduplicate_and_refuse_malformed_terms() -> None:
    expression = ("mappingproxy", (((1,), Fraction(2**100 + 1, 3)),))
    rows = ((expression,),) * 32
    recipe = ExactExpressionRowsArchiveRecipe.encode(rows)
    assert recipe.kinds.shape == (1,)
    assert recipe.decode() == rows
    assert ExactExpressionRowsArchiveRecipe.encode(()).decode() == ()
    malformed = (
        recipe._replace(kinds=np.asarray([3], np.int8)),
        recipe._replace(polynomial_ends=np.asarray([2, 2], np.int64)),
        recipe._replace(rows=recipe.rows._replace(values=recipe.rows.values + 1)),
        recipe._replace(
            exponents=recipe.exponents._replace(
                values=recipe.exponents.values.astype(np.float64)
            )
        ),
    )
    for changed in malformed:
        with pytest.raises((TypeError, ValueError)):
            changed.decode()


def test_rational_term_rows_share_only_within_one_decode() -> None:
    from phydrax.meshing._archive_layer_columns import _decode_column, _encode_column

    numerator = (((1,), Fraction(2, 3)),)
    denominator = (((0,), Fraction(1)),)
    distinct = (((0,), Fraction(4)),)
    rows = (((numerator, denominator),), ((distinct, denominator),)) * 64
    recipe = _encode_column(rows, "rational_terms")
    restored = _decode_column(recipe, "rational_terms")
    assert restored == rows
    assert restored[0] is restored[2]
    assert restored[1] is restored[3]
    assert restored[0] is not restored[1]
    assert _decode_column(recipe, "rational_terms")[0] is not restored[0]


def _record_column(recipe: dict[str, Any], record_type: str, field: str, /) -> str:
    """Member name of one record's array column in an expanded model recipe."""
    from phydrax._model._structure import _recipe_array_nodes

    pending: list[Any] = [recipe]
    while pending:
        node = pending.pop()
        if isinstance(node, list):
            pending.extend(node)
            continue
        if not isinstance(node, dict):
            continue
        if (
            node.get("kind") == "namedtuple"
            and str(node.get("type", "")).endswith("LayerRecordArchiveRecipe")
            and node["items"][0]["value"] == record_type
        ):
            (column,) = (
                value for key, value in node["items"][1]["items"] if key["value"] == field
            )
            assert column["kind"] == "array"
            index = next(
                index
                for index, item in enumerate(_recipe_array_nodes(recipe))
                if item is column
            )
            return f"model/leaves/{index:06d}"
        pending.extend(node.values())
    raise AssertionError(f"No {record_type}.{field} column in the recipe.")


def _refuses_column(
    path: Path,
    expected_type: type,
    record_type: str,
    field: str,
    change: Any,
    /,
) -> None:
    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    name = _record_column(manifest["model_recipe"], record_type, field)
    changed = dict(arrays)
    changed[name] = change(np.array(arrays[name]))
    count = len(tuple(path.parent.iterdir()))
    tampered = write_model_archive(
        path.with_name(f"tampered-{record_type}-{count}.phx"),
        manifest=manifest,
        arrays=changed,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    from phydrax.meshing._model_codec import restore_model_archive_value

    with pytest.raises(ArrayArchiveCorruptionError):
        read_model_artifact(
            tampered,
            expected_type=expected_type,
            restore=restore_model_archive_value,
            limits=MODEL_ARCHIVE_LIMITS,
        )


def test_cell_map_digests_archive_as_fixed_width_byte_rows(tmp_path: Path) -> None:
    import phydrax as phx
    from phydrax.discretization._cell_mesh import CellMesh
    from phydrax.meshing._model_codec import restore_model_archive_value

    axis = np.linspace(0.0, 1.0, 4)
    points = np.stack(np.meshgrid(axis, axis), axis=-1).reshape((-1, 2))
    cells = [
        cell
        for j in range(3)
        for i in range(3)
        for cell in (
            (j * 4 + i, j * 4 + i + 1, j * 4 + i + 5),
            (j * 4 + i, j * 4 + i + 5, j * 4 + i + 4),
        )
    ]
    result = phx.meshing.certify_cell_mesh(
        CellMesh.from_triangles(points, np.asarray(cells, dtype=np.int32)),
        phx.SpatialCoordinateContract.si(),
    )
    digests = result.audit.validity.cell_map_digests
    path = write_model_artifact(
        tmp_path / "validity.phx", result, limits=MODEL_ARCHIVE_LIMITS
    )
    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    column = np.asarray(
        arrays[
            _record_column(
                manifest["model_recipe"], "CellValidityCertificate", "cell_map_digests"
            )
        ]
    )
    assert column.dtype == np.uint8 and column.shape == (len(cells), 32)
    assert column.tobytes() == digests
    restored = read_model_artifact(
        path,
        expected_type=type(result),
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    assert restored.result_id == result.result_id
    assert restored.audit.validity.cell_map_digests == digests

    def flipped(rows: np.ndarray) -> np.ndarray:
        rows[0, 0] ^= 1
        return rows

    _refuses_column(
        path, type(result), "CellValidityCertificate", "cell_map_digests", flipped
    )
    _refuses_column(
        path,
        type(result),
        "CellValidityCertificate",
        "cell_map_digests",
        lambda rows: rows.reshape((-1, 16)),
    )


def test_scope_unqualified_ids_archive_as_integer_column(tmp_path: Path) -> None:
    from phydrax.meshing._model_codec import restore_model_archive_value
    from phydrax.meshing._scope import MeshingEntityKind, MeshingScope

    scope = MeshingScope(
        "column-source",
        "column-revision",
        MeshingEntityKind.MESH,
        0,
        "column-vertices",
        np.arange(0, 40, 2, dtype=np.int64),
    )
    # Archive roots keep their registered dataclass; nested scopes use columns.
    value = Composition(
        (
            CompositionEntry(
                scope,
                entry_id="scope",
                role="model-state",
                owner_id="phydrax.meshing",
                structure_id=scope.scope_id,
                revision_id=scope.source_revision,
                semantics_id="meshing-scope",
            ),
        ),
        boundary_id="scope-column-archive",
    )
    path = write_model_artifact(
        tmp_path / "scope.phx", value, limits=MODEL_ARCHIVE_LIMITS
    )
    manifest, arrays = read_model_archive(path, limits=MODEL_ARCHIVE_LIMITS)
    column = np.asarray(
        arrays[
            _record_column(
                manifest["model_recipe"], "MeshingScope", "unqualified_entity_ids"
            )
        ]
    )
    assert column.dtype == np.int64
    assert tuple(column.tolist()) == scope.unqualified_entity_ids
    restored = read_model_artifact(
        path,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    ).value("scope")
    assert type(restored) is MeshingScope
    assert restored.scope_id == scope.scope_id
    assert restored.unqualified_entity_ids == scope.unqualified_entity_ids

    def changed(ids: np.ndarray) -> np.ndarray:
        ids[-1] += 1
        return ids

    _refuses_column(path, Composition, "MeshingScope", "unqualified_entity_ids", changed)
    _refuses_column(
        path,
        Composition,
        "MeshingScope",
        "unqualified_entity_ids",
        lambda ids: ids.astype(np.int32),
    )
