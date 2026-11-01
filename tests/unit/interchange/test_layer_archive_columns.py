from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._archive_columns import (
    ExactTensorArchiveRecipe,
    StaticColumnArchiveRecipe,
    TextRowsArchiveRecipe,
)
from phydrax._array_archive import ArrayArchiveCorruptionError
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
    )
    assert rebuilt.identity_id == stored.identity_id
    return _composition(rebuilt)


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
        restore_row_record_recipes,
    )
    from phydrax.geometry._source_fidelity_certificate import SourceFidelityCertificate

    fields = {field.name: None for field in dataclasses.fields(SourceFidelityCertificate)}
    fields.update(
        source_cell_global_ids=tuple(range(4096)),
        source_patch_ids=(0,) * 4096,
        source_occurrence_paths=((),) * 4096,
    )
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


@pytest.mark.parametrize("change", ("index", "identity"))
def test_selected_fields_reject_unselected_column_corruption(
    tmp_path: Path, change: str
) -> None:
    source = write_model_artifact(
        tmp_path / "source.phx", _composition(_identity()), limits=MODEL_ARCHIVE_LIMITS
    )
    assert read_model_archive_fields(
        source,
        expected_type=Composition,
        fields=("boundary_id",),
        limits=MODEL_ARCHIVE_LIMITS,
    ) == {"boundary_id": "column-archive"}
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
            fields=("boundary_id",),
            limits=MODEL_ARCHIVE_LIMITS,
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
