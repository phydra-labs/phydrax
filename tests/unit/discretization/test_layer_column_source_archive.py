"""Durable column-profile source values and fail-closed restored source validation."""

from dataclasses import fields
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax.discretization._cell_geometry import (
    coordinate_lagrange_element,
    LayerColumnCellGeometryElement,
)
from phydrax.lifecycle._meshing_sources import (
    read_meshing_source_closure,
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
    write_meshing_source_closure,
)


def test_layer_column_archive_retains_original_profile_tabulation(tmp_path: Path) -> None:
    profile = coordinate_lagrange_element("quadrilateral", 2)
    from tests._support.coordinate_references import coordinate_reference_budget

    with coordinate_reference_budget() as budget:
        element = LayerColumnCellGeometryElement(profile, fiber_graph=True, budget=budget)
        assert (
            element.element_id
            != LayerColumnCellGeometryElement(profile, budget=budget).element_id
        )
    receipt = write_meshing_source_closure(
        tmp_path / "column-profile", (profile, element)
    )
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    restored_profile, reopened = restored
    assert reopened.wall_element is restored_profile
    assert reopened.element_id == element.element_id
    assert reopened.station_axis == 2
    assert reopened.fiber_graph is True
    assert reopened.local_dof_count == 2 * profile.local_dof_count + 16
    points = jnp.asarray(
        ((0.2, 0.3, 0.0), (0.4, 0.6, 0.5), (0.7, 0.8, 1.0)), dtype=jnp.float64
    )
    original_values, original_gradients = element.tabulate(points)
    restored_values, restored_gradients = reopened.tabulate(points)
    np.testing.assert_array_equal(restored_values, original_values)
    np.testing.assert_array_equal(restored_gradients, original_gradients)
    assert np.any(np.asarray(restored_gradients) != 0.0)
    assert model_structure_recipe(restored) == model_structure_recipe((profile, element))


@pytest.mark.parametrize(
    ("malformation", "expected_field"),
    (
        ("station-axis", "station_axis"),
        ("coefficient-count", "local_dof_count"),
        ("corner-source", "corner_element"),
        ("fiber-graph", "element_id"),
    ),
)
def test_restored_layer_column_refuses_changed_authority(
    malformation: str, expected_field: str
) -> None:
    register_meshing_source_artifacts()
    from tests._support.coordinate_references import coordinate_reference_budget

    with coordinate_reference_budget() as budget:
        element = LayerColumnCellGeometryElement(
            coordinate_lagrange_element("quadrilateral", 2), budget=budget
        )
    if malformation == "corner-source":
        element = eqx.tree_at(
            lambda value: value.corner_element,
            element,
            coordinate_lagrange_element("quadrilateral", 2),
        )
    field_rows = {field.name: index for index, field in enumerate(fields(type(element)))}
    recipe = model_structure_recipe(element)
    arrays = model_recipe_array_values(element, recipe, prefix="column")
    if malformation == "station-axis":
        recipe["items"][field_rows["station_axis"]]["value"] = 0
    elif malformation == "coefficient-count":
        recipe["items"][field_rows["local_dof_count"]]["value"] += 1
    elif malformation == "fiber-graph":
        recipe["items"][field_rows["fiber_graph"]]["value"] = True
    restored = model_from_array_recipe(recipe, arrays, prefix="column")
    with pytest.raises(
        ValueError, match=rf"LayerColumnCellGeometryElement\.{expected_field} must"
    ):
        validate_meshing_source_closure(restored)
