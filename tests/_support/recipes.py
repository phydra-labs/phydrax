#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Navigation of canonical positional structure recipes for archive tests.

A registered dataclass recipe node stores its children as ``items`` in the
registered class's dataclass field order; there is no name-keyed form. Tests
that deliberately corrupt one named field locate it through that order.
Identity corruption retains every numerical leaf and invokes the registered
restore owners rather than mutating static PyTree metadata.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from phydrax._model._artifacts import artifact_value
from phydrax._model._structure import (
    model_from_logical_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax._strict import StrictModule


def recipe_field_names(node: Mapping[str, Any], /) -> tuple[str, ...]:
    """Registered field names of one canonical dataclass recipe node, in order."""
    if node.get("kind") != "dataclass":
        raise ValueError("Expected a canonical dataclass recipe node.")
    return tuple(field.name for field in dataclasses.fields(artifact_value(node["type"])))


def recipe_field_position(node: Mapping[str, Any], name: str, /) -> int:
    """Position of the registered field ``name`` in the node's ``items``."""
    return recipe_field_names(node).index(name)


def restore_with_identity(value: StrictModule, name: str, identity: str, /) -> object:
    """Restore a root identity corruption with every numerical leaf unchanged."""
    recipe = model_structure_recipe(value)
    arrays = model_recipe_array_values(value, recipe, prefix="identity-corruption")
    position = recipe_field_position(recipe, name)
    field = recipe["items"][position]
    if field.get("kind") != "literal" or not isinstance(field.get("value"), str):
        raise TypeError(
            f"recipe.{name} must contain a literal string identity; got {field}"
        )
    recipe["items"][position] = {"kind": "literal", "value": identity}
    return model_from_logical_array_recipe(recipe, arrays, prefix="identity-corruption")
