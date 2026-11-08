#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.fem import _spectral_hp_io as spectral_hp_io


def _cell_block(cell_type: Any, data: Any) -> Any:
    return SimpleNamespace(type=cell_type, data=np.asarray(data, dtype=np.int32))


@pytest.mark.parametrize("profile_kind", ("triangle", "quadrilateral"))
def test_swept_coordinate_element_recipe_retains_source_and_rejects_tampering(
    profile_kind: str, tmp_path: Any
) -> None:
    from phydrax.discretization._cell_geometry import swept_coordinate_element
    from phydrax.discretization.fem import lagrange_element
    from phydrax.meshing._model_codec import read_model_artifact, write_model_artifact
    from phydrax.solver._finite_element_schedule import (
        _accepted_element_recipe,
        FiniteElementAcceptedElementRecipe,
    )

    element = swept_coordinate_element(lagrange_element(profile_kind, 2))
    recipe = _accepted_element_recipe(element)
    assert recipe.recipe_kind == "swept"
    assert recipe.source_recipe is not None
    path = tmp_path / "swept-element.phx"
    write_model_artifact(path, recipe, limits=None)
    stored = read_model_artifact(
        path,
        expected_type=FiniteElementAcceptedElementRecipe,
        restore=lambda value: _accepted_element_recipe(value.rebuild()),
        limits=None,
    )
    rebuilt = stored.rebuild()
    assert rebuilt.element_id == element.element_id
    np.testing.assert_array_equal(rebuilt.reference_nodes, element.reference_nodes)
    forged_nodes = np.array(recipe.reference_nodes)
    forged_nodes[0, 0] += 0.25
    forged = FiniteElementAcceptedElementRecipe(
        recipe.recipe_kind,
        recipe.family,
        recipe.cell_kind,
        recipe.degree,
        recipe.continuity,
        recipe.representation,
        recipe.value_spec,
        recipe.form_identity,
        forged_nodes,
        recipe.entity_dofs,
        recipe.tabulator_id,
        recipe.element_id,
        recipe.source_recipe,
    )
    with pytest.raises(ValueError, match="mathematical fields"):
        forged.rebuild()


def test_swept_tabulator_equality_rechecks_all_source_fields() -> None:
    from dataclasses import fields

    from phydrax.discretization._cell_geometry import (
        swept_coordinate_element,
        SweptCoordinateTabulator,
    )
    from phydrax.discretization.fem import lagrange_element
    from phydrax.geometry._mesh_certificates import _same_tree_content

    source = lagrange_element("quadrilateral", 2)
    rebuilt = lagrange_element("quadrilateral", 2)
    first = SweptCoordinateTabulator(source)
    assert first == SweptCoordinateTabulator(rebuilt)
    assert hash(first) == hash(SweptCoordinateTabulator(rebuilt))
    assert _same_tree_content(first.source, rebuilt)
    assert _same_tree_content(
        swept_coordinate_element(source), swept_coordinate_element(rebuilt)
    )
    altered = object.__new__(type(source))
    for field in fields(source):
        value = getattr(source, field.name)
        if field.name == "reference_nodes":
            value = np.array(value)
            value[0, 0] += 0.125
        object.__setattr__(altered, field.name, value)
    assert altered.element_id == source.element_id
    assert first != SweptCoordinateTabulator(altered)
    assert hash(first) == hash(SweptCoordinateTabulator(altered))
    with pytest.raises(AttributeError):
        first.source = altered


def test_mesh_import_preserves_volume_and_facet_groups_with_loss_evidence(
    monkeypatch: Any, tmp_path: Any
) -> None:
    cells = (
        _cell_block("triangle", [[0, 1, 2], [0, 2, 3]]),
        _cell_block("line", [[0, 1], [1, 2], [2, 3], [3, 0]]),
        _cell_block("vertex", [[0]]),
    )
    source = SimpleNamespace(
        points=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
        ),
        cells=cells,
        cell_sets={
            "myocardium": (
                np.asarray([0, 1], dtype=np.int32),
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.int32),
            ),
            "base": (
                np.asarray([], dtype=np.int32),
                np.asarray([0], dtype=np.int32),
                np.asarray([], dtype=np.int32),
            ),
        },
        point_sets={},
        point_data={"temperature": np.arange(4.0)},
        cell_data={"material": (np.asarray([7, 7]), np.arange(4), np.asarray([0]))},
        field_data={"myocardium": np.asarray([7, 2], dtype=np.int32)},
        info={"generator": "deterministic-fixture", "revision": 3},
        gmsh_periodic=[
            (
                1,
                (10, 20),
                np.eye(4),
                np.asarray([[0, 1]], dtype=np.int32),
            )
        ],
    )
    # The reader admits the on-disk bytes through the bounded-resource boundary,
    # then decodes the staged copy with the module's meshio binding.
    (tmp_path / "case.msh").write_bytes(b"deterministic-fixture")
    monkeypatch.setattr(
        spectral_hp_io,
        "meshio",
        SimpleNamespace(read=lambda path, file_format: source),
    )

    imported = phx.discretization.read_finite_element_mesh(tmp_path / "case.msh")

    assert imported.volume_cells("myocardium") == (0, 1)
    assert len(imported.boundary_facets("base")) == 1
    assert imported.report.volume_names == ("myocardium",)
    assert imported.report.boundary_names == ("base",)
    assert imported.report.source_path.endswith("case.msh")
    assert imported.report.source_format == "gmsh"
    assert dict(imported.report.source_metadata)["info"] == (
        '{"generator":"deterministic-fixture","revision":3}'
    )
    assert dict(imported.report.source_entity_counts) == {
        "facet_cells": 4,
        "other_cells": 1,
        "points": 4,
        "volume_cells": 2,
    }
    assert dict(imported.report.imported_entity_counts)["facet_cells"] == 1
    assert dict(imported.report.dropped_entity_counts)["facet_cells"] == 3
    assert dict(imported.report.dropped_entity_counts)["other_cells"] == 1
    assert imported.report.lossy
    assert "dropped_ungrouped_facet_cells:3" in imported.report.losses
    assert "dropped_unsupported_cells:1" in imported.report.losses
    assert (
        imported.report.report_id
        == phx.discretization.read_finite_element_mesh(
            tmp_path / "case.msh"
        ).report.report_id
    )
    assert "gmsh_periodic_not_imported:1" in imported.report.losses
    assert "gmsh_periodic" in dict(imported.report.source_metadata)

    with pytest.raises(ValueError, match="Unknown imported volume group"):
        imported.volume_cells("missing")
